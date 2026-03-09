#include "pose_yolov8.hpp"
#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include "tools/logger.hpp"

namespace auto_aim
{
// 内部候选框结构
struct RawDetection
{
    float x1, y1, x2, y2;      // 在 640x640 坐标系下的浮点坐标
    float confidence;
    int class_id;
    int grid_h, grid_w;        // 可选项，用于调试
};

// NMS 函数
static void nms(std::vector<RawDetection> &dets, float iou_threshold)
{
    if (dets.empty()) return;
    // 按置信度降序排序
    std::sort(dets.begin(), dets.end(),
              [](const RawDetection &a, const RawDetection &b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> keep(dets.size(), true);
    for (size_t i = 0; i < dets.size(); ++i)
    {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < dets.size(); ++j)
        {
            if (!keep[j]) continue;
            // 计算 IoU
            float xi1 = std::max(dets[i].x1, dets[j].x1);
            float yi1 = std::max(dets[i].y1, dets[j].y1);
            float xi2 = std::min(dets[i].x2, dets[j].x2);
            float yi2 = std::min(dets[i].y2, dets[j].y2);
            float inter_area = std::max(0.0f, xi2 - xi1) * std::max(0.0f, yi2 - yi1);
            float union_area = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1) +
                               (dets[j].x2 - dets[j].x1) * (dets[j].y2 - dets[j].y1) - inter_area;
            float iou = inter_area / union_area;
            if (iou > iou_threshold)
                keep[j] = false;
        }
    }

    // 删除未保留的
    size_t idx = 0;
    for (size_t i = 0; i < dets.size(); ++i)
    {
        if (keep[i])
            dets[idx++] = dets[i];
    }
    dets.resize(idx);
}

// DFL 解码函数
static void dfl_decode(const float* box_input, int h, int w, float* box_output)
{
    // box_input: [64, h, w] 连续存储 (C,H,W)
    // box_output: [4, h, w] 连续存储
    const int reg_max = 16;   // 因为 64/4 = 16
    const int c_stride = h * w;
    // 对每个网格点进行解码
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            int idx = i * w + j;   // 在 h*w 平面上的线性索引
            // 对每个坐标分量（4个）分别处理
            for (int k = 0; k < 4; ++k)
            {
                const float* src = box_input + k * reg_max * c_stride + idx;
                // 计算 softmax 并加权求和
                float max_val = src[0];
                for (int m = 1; m < reg_max; ++m)
                {
                    if (src[m * c_stride] > max_val)
                        max_val = src[m * c_stride];
                }
                float sum_exp = 0.0f;
                float weight_sum = 0.0f;
                for (int m = 0; m < reg_max; ++m)
                {
                    float exp_val = std::exp(src[m * c_stride] - max_val);
                    sum_exp += exp_val;
                    weight_sum += exp_val * m;
                }
                float offset = weight_sum / sum_exp;   // 0~15
                // 存储结果到 box_output 的对应位置
                // box_output 布局为 [4, h, w]，所以 (k,i,j) 的索引为 k * c_stride + idx
                box_output[k * c_stride + idx] = offset;
            }
        }
    }
}

Pose_YOLOv8::Pose_YOLOv8(const std::string & config_path, bool debug)
  : debug_(debug)
{
  auto yaml = YAML::LoadFile(config_path);
  model_path_ = yaml["yolov8_model_path"].as<std::string>();
  use_roi_ = yaml["use_roi"].as<bool>();
  int x = yaml["roi"]["x"].as<int>(0);
  int y = yaml["roi"]["y"].as<int>(0);
  int w = yaml["roi"]["width"].as<int>(-1);
  int h = yaml["roi"]["height"].as<int>(-1);
  roi_ = cv::Rect(x, y, w, h);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  // 读取RKNN模型文件
  std::ifstream file(model_path_, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error("Failed to open RKNN model file: " + model_path_);
  std::streamsize file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  model_data_.resize(file_size);
  if (!file.read((char*)model_data_.data(), file_size))
    throw std::runtime_error("Failed to read RKNN model file");
  file.close();

  // 初始化RKNN
  int ret = rknn_init(&ctx_, model_data_.data(), model_data_.size(), 0, nullptr);
  if (ret != RKNN_SUCC)
    throw std::runtime_error("RKNN init failed: " + std::to_string(ret));

  // 查询输入输出信息
  ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  if (ret != RKNN_SUCC)
  {
    rknn_destroy(ctx_);
    throw std::runtime_error("Failed to query IO info");
  }
  std::cout << "RKNN model loaded. Inputs: " << io_num_.n_input
            << ", Outputs: " << io_num_.n_output << std::endl;

  // 准备输入结构体（固定为1个输入）
  memset(&rknn_input_, 0, sizeof(rknn_input_));
  rknn_input_.index = 0;
  rknn_input_.type = RKNN_TENSOR_UINT8;
  rknn_input_.fmt = RKNN_TENSOR_NHWC;
  rknn_input_.size = img_size_ * img_size_ * 3;  // 640*640*3

  // 分配输出结构体数组
  rknn_outputs_ = new rknn_output[io_num_.n_output];
  memset(rknn_outputs_, 0, sizeof(rknn_output) * io_num_.n_output);
  for (uint32_t i = 0; i < io_num_.n_output; ++i)
  {
    rknn_outputs_[i].index = i;
    rknn_outputs_[i].want_float = 1;   // 需要浮点输出
    rknn_outputs_[i].is_prealloc = 0;
  }

  input_buffer_.create(img_size_, img_size_, CV_8UC3);
}

std::list<Pose> Pose_YOLOv8::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty())                                              // 传入图像为空
  {
    tools::logger()->warn("Empty img!");
    return {};
  }

  cv::Mat roi_img;
  if (use_roi_)
  {
    if (roi_.width == -1) roi_.width = raw_img.cols;
    if (roi_.height == -1) roi_.height = raw_img.rows;
    roi_img = raw_img(roi_);
  }
  else
  {
    roi_img = raw_img;                                              // 没有使用roi则相同
  }

  // 将输入的图像转换为640*640，且保持图像的原始宽高比，多余的部分用黑色填充
  double scale;
  int pad_left, pad_top;
  // 输入图像，输出图像，输出参数
  letterbox(roi_img, input_buffer_, scale, pad_left, pad_top);
  cv::cvtColor(input_buffer_, input_buffer_, cv::COLOR_BGR2RGB);

  rknn_input_.buf = input_buffer_.data;                           //
  int ret = rknn_inputs_set(ctx_, io_num_.n_input, &rknn_input_);
  if (ret != RKNN_SUCC)
  {
    tools::logger()->error("Failed to set inputs");
    return {};
  }

  ret = rknn_run(ctx_, nullptr);                          // 运行推理
  if (ret != RKNN_SUCC)
  {
    tools::logger()->error("Failed to run inference");
    return {};
  }

  if (rknn_outputs_[0].buf != nullptr)                           // 检查缓冲区非空
  {
    // 用于释放之前通过 rknn_outputs_get 获取的输出缓冲区内存
    rknn_outputs_release(ctx_, io_num_.n_output, rknn_outputs_);
  }
  // 获取推理结果
  // 推理结果会放到rknn_outputs_ 数组中
  ret = rknn_outputs_get(ctx_, io_num_.n_output, rknn_outputs_, nullptr);
  if (ret != RKNN_SUCC)
  {
    tools::logger()->error("Failed to get outputs");
    return {};
  }

  // 解析输出
  std::list<Pose> results = parse_outputs(scale, pad_left, pad_top, raw_img, frame_count);

  return results;
}

void Pose_YOLOv8::letterbox(const cv::Mat &src, cv::Mat &dst, double &scale, int &pad_left, int &pad_top)
{
  // 实现保持宽高比的resize并填充到目标尺寸 dst 已预先分配为 640x640
  int src_h = src.rows, src_w = src.cols;
  int dst_h = dst.rows, dst_w = dst.cols;

  double scale_h = static_cast<double>(dst_h) / src_h;
  double scale_w = static_cast<double>(dst_w) / src_w;
  scale = std::min(scale_h, scale_w);

  int new_w = static_cast<int>(src_w * scale);
  int new_h = static_cast<int>(src_h * scale);

  pad_left = (dst_w - new_w) / 2;
  pad_top = (dst_h - new_h) / 2;

  // 先 resize 到新尺寸
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(new_w, new_h));

  // 填充黑色背景
  dst.setTo(cv::Scalar(0, 0, 0));
  resized.copyTo(dst(cv::Rect(pad_left, pad_top, new_w, new_h)));
}

std::list<Pose> Pose_YOLOv8::parse_outputs(double scale, int pad_left, int pad_top,
                                           const cv::Mat &raw_img, int frame_count)
{
  // 输出指针数组，方便索引
  std::vector<float*> outputs(io_num_.n_output);
  for (uint32_t i = 0; i < io_num_.n_output; ++i)
  {
    outputs[i] = (float*)rknn_outputs_[i].buf;
  }

  // 在 YOLOv8 中，模型会从三个不同大小的特征图上进行检测，这就是所谓的 三个尺度。
  // 这样做的目的是为了能够同时检测图像中 大小不同的目标——比如近处的人（大）和远处的人（小）。
  const std::vector<int> strides = {8, 16, 32};
  const std::vector<int> grid_sizes = {80, 40, 20};

  std::vector<RawDetection> all_detections;

  // 对每个尺度处理
  for (int idx = 0; idx < 3; ++idx)
  {
    int h = grid_sizes[idx];
    int w = h;
    int stride = strides[idx];

    // 获取三个输出的指针
    // 输出顺序：按照之前 ONNX 打印的顺序，先 80x80 的 3 个，然后 40x40 的 3 个，然后 20x20 的 3 个
    const float* box_data = outputs[idx * 3];                   // [1,64,h,w]
    const float* cls_data = outputs[idx * 3 + 1];               // [1,80,h,w]
    const float* obj_data = outputs[idx * 3 + 2];               // [1,1,h,w] 或类似

    // 解码 box：申请临时空间存放解码后的 [4, h, w]
    std::vector<float> decoded_box(4 * h * w);
    dfl_decode(box_data, h, w, decoded_box.data());

    // 遍历每个网格
    for (int i = 0; i < h; ++i)
    {
      for (int j = 0; j < w; ++j)
      {
        int flat_idx = i * w + j;                               // 在 h*w 平面上的索引

        // 获取 obj 分数
        float obj_score = obj_data[flat_idx];                   // 假设 obj_data 是 [1,1,h,w]，因此直接取
        if (obj_score < 0.01f) continue;                        // 快速过滤

        // 获取分类分数：cls_data 形状 [1,80,h,w]，80个类别连续存储
        const float* cls_ptr = cls_data + flat_idx;
        float max_cls = 0.0f;
        int best_cls = 0;
        for (int c = 0; c < num_classes_; ++c)
        {
          float val = cls_ptr[c * h * w];                       // 类别 c 在该位置的值
          if (val > max_cls)
          {
            max_cls = val;
            best_cls = c;
          }
        }
        float confidence = obj_score * max_cls;                 // 获得信任度
        if (confidence < conf_threshold_)
          continue;

        // 只保留人的类别（COCO 中 person 是 0），如果不是人则跳过
        if (best_cls != 0) continue;

        // 获取解码后的偏移量
        const float* box_ptr = decoded_box.data() + flat_idx; // 指向 [4, h, w] 中该位置的第一个分量
        // 每个分量的偏移量
        float dx1 = box_ptr[0 * h * w];                       // 左偏移
        float dy1 = box_ptr[1 * h * w];                       // 上偏移
        float dx2 = box_ptr[2 * h * w];                       // 右偏移
        float dy2 = box_ptr[3 * h * w];                       // 下偏移

        // 计算在 640x640 坐标系下的边界框
        float grid_cx = j + 0.5f;
        float grid_cy = i + 0.5f;
        float x1 = (grid_cx - dx1) * stride;
        float y1 = (grid_cy - dy1) * stride;
        float x2 = (grid_cx + dx2) * stride;
        float y2 = (grid_cy + dy2) * stride;

        // 裁剪到图像范围内（640x640）
        x1 = std::max(0.0f, std::min(639.0f, x1));
        y1 = std::max(0.0f, std::min(639.0f, y1));
        x2 = std::max(0.0f, std::min(639.0f, x2));
        y2 = std::max(0.0f, std::min(639.0f, y2));

        if (x2 <= x1 || y2 <= y1) continue;

        // 保存候选框
        RawDetection det;
        det.x1 = x1;
        det.y1 = y1;
        det.x2 = x2;
        det.y2 = y2;
        det.confidence = confidence;
        det.class_id = best_cls;
        all_detections.push_back(det);
      }
    }
  }

  // NMS
  nms(all_detections, nms_threshold_);

  // 将检测框从 640x640 坐标系映射回原始图像，并创建 Pose 对象
  std::list<Pose> poses;
  for (const auto &det : all_detections)
  {
    // 减去填充，除以缩放，得到在 ROI 图像中的坐标
    float x1 = (det.x1 - pad_left) / scale;
    float y1 = (det.y1 - pad_top) / scale;
    float x2 = (det.x2 - pad_left) / scale;
    float y2 = (det.y2 - pad_top) / scale;

    // 如果使用了 ROI，还需要加上偏移量得到原始图像坐标
    if (use_roi_)
    {
      x1 += offset_.x;
      y1 += offset_.y;
      x2 += offset_.x;
      y2 += offset_.y;
    }

    // 裁剪到原始图像边界
    x1 = std::max(0.0f, std::min(static_cast<float>(raw_img.cols - 1), x1));
    y1 = std::max(0.0f, std::min(static_cast<float>(raw_img.rows - 1), y1));
    x2 = std::max(0.0f, std::min(static_cast<float>(raw_img.cols - 1), x2));
    y2 = std::max(0.0f, std::min(static_cast<float>(raw_img.rows - 1), y2));

    if (x2 <= x1 || y2 <= y1) continue;

    cv::Rect bbox(static_cast<int>(x1), static_cast<int>(y1),
    static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
    Pose pose(bbox, det.confidence, det.class_id);
    poses.push_back(pose);
  }
  // cmake -B build
  // make -C build/ -j`nproc`
  // ./build/flower_test
  if (debug_)
  {
    draw_detections(raw_img, poses, frame_count);
  }
  

  return poses;
}

void Pose_YOLOv8::draw_detections(const cv::Mat &img, const std::list<Pose> &poses, int frame_count) const
{
  auto detection = img.clone();
  for (const auto &pose : poses)
  {
    cv::rectangle(detection, pose.bbox, cv::Scalar(0, 255, 0), 2);
    cv::circle(detection, pose.center, 3, cv::Scalar(0, 255, 0), -1);
  }
  if (use_roi_)
    cv::rectangle(detection, roi_, cv::Scalar(0, 255, 0), 2);
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("human detection", detection);
}

}  // namespace auto_aim