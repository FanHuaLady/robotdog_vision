#include "fire_yolov5.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <filesystem>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
Fire_YOLOV5::Fire_YOLOV5(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);                                    // 获取yaml

  model_path_ = yaml["yolov5_model_path"].as<std::string>();                        // 获得路径
  binary_threshold_ = yaml["threshold"].as<double>();                               // 无作用
  min_confidence_ = yaml["min_confidence"].as<double>();                            // 无作用
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  roi_ = cv::Rect(x, y, width, height);                                             // 感兴趣区
  offset_ = cv::Point2f(x, y);                                                      // 感兴趣的偏移
  use_roi_ = yaml["use_roi"].as<bool>();                                            // 是否启用感兴趣区
  use_traditional_ = yaml["use_traditional"].as<bool>();                            // 是否使用传统识别
  save_path_ = "imgs";                                                              // 保存图片
  std::filesystem::create_directory(save_path_);

  // 读取RKNN模型文件到内存
  // std::ifstream: 输入文件流，用于读取文件
  // model_path_: 模型文件路径（从YAML配置中读取）
  // std::ios::binary: 二进制模式打开文件，确保不进行字符转换
  // std::ios::ate: 打开后直接定位到文件末尾
  std::ifstream file(model_path_, std::ios::binary | std::ios::ate);
  if (!file.is_open())
  {
    throw std::runtime_error("Failed to open RKNN model file: " + model_path_);
  }

  std::streamsize file_size = file.tellg();                                     // 获取文件大小
  file.seekg(0, std::ios::beg);                                                    // 回到文件开头
  model_data_.resize(file_size);                                                   // 准备内存缓冲区
  // 把整个RKNN模型文件从磁盘读取到了内存中
  if (!file.read((char*)model_data_.data(), file_size))                            // 读取文件到内存
  {
    throw std::runtime_error("Failed to read RKNN model file");
  }
  file.close();                                                                    // 关闭文件

  // 1. 输出参数：RKNN上下文指针的地址
  // 2. 模型数据的内存地址
  // 3. 模型数据的大小（字节数）
  // 4. 标志位，通常为0
  // 5. 扩展参数，通常为nullptr
  int ret = rknn_init(&ctx_, model_data_.data(), model_data_.size(), 0, nullptr);
  if (ret != RKNN_SUCC)
  {
    throw std::runtime_error("RKNN initialization failed with error code: " + std::to_string(ret));
  }

  // rknn_query用于获取模型的各种信息
  // 1. RKNN上下文（从rknn_init获得）
  // 2. 查询类型：输入输出数量
  // 3. 输出参数：存储查询结果
  // 4. 输出参数的大小
  ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  if (ret != RKNN_SUCC)
  {
    rknn_destroy(ctx_);
    ctx_ = 0;
    throw std::runtime_error("Failed to query model IO info");
  }

  std::cout << "RKNN model loaded successfully. Inputs: " << io_num_.n_input
            << ", Outputs: " << io_num_.n_output << std::endl;

  memset(&rknn_input_, 0, sizeof(rknn_input_));
  rknn_input_.index = 0;
  rknn_input_.type = RKNN_TENSOR_UINT8;
  rknn_input_.fmt = RKNN_TENSOR_NHWC;
  rknn_input_.size = 640 * 640 * 3;  // 固定大小：640×640 RGB

  rknn_outputs_ = new rknn_output[io_num_.n_output];
  memset(rknn_outputs_, 0, sizeof(rknn_outputs_[0]) * io_num_.n_output);

  for (int i = 0; i < io_num_.n_output; i++)
  {
    rknn_outputs_[i].index = i;
    rknn_outputs_[i].want_float = 1;                                  // 需要浮点数输出
    rknn_outputs_[i].is_prealloc = 0;                                 // 让RKNN分配数据内存
  }

  input_buffer_.create(640, 640, CV_8UC3);
}

std::list<Fire> Fire_YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty())                                                // 没有图片直接返回
  {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Fire>();
  }

  cv::Mat bgr_img;                                                    // ROI后的图像
  if (use_roi_)
  {
    if (roi_.width == -1)
    {
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1)
    {
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  }
  else
  {
    bgr_img = raw_img;                                                // 不使用ROI
  }

  auto x_scale = static_cast<double>(640) / bgr_img.rows;      // x缩放
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  // 清空输入缓冲区为黑色
  input_buffer_.setTo(cv::Scalar(0, 0, 0));

  // 将图像缩放到缓冲区中的指定区域
  cv::Rect roi(0, 0, w, h);
  cv::resize(bgr_img, input_buffer_(roi), cv::Size(w, h));

  // 3. 转换颜色空间（直接在缓冲区上操作）
  cv::cvtColor(input_buffer_, input_buffer_, cv::COLOR_BGR2RGB);

  // 4. 设置输入结构体指向我们的缓冲区
  rknn_input_.buf = input_buffer_.data;

  // 设置输入
  int ret = rknn_inputs_set(ctx_, io_num_.n_input, &rknn_input_);
  if (ret != RKNN_SUCC)
  {
    tools::logger()->error("Failed to set RKNN inputs");
    return std::list<Fire>();
  }

  // 运行推理
  ret = rknn_run(ctx_, nullptr);
  if (ret != RKNN_SUCC)
  {
    tools::logger()->error("RKNN inference failed");
    return std::list<Fire>();
  }

  // 注意：如果之前调用过rknn_outputs_get，需要先释放RKNN分配的数据内存
  if (rknn_outputs_[0].buf != nullptr)
  {
    rknn_outputs_release(ctx_, io_num_.n_output, rknn_outputs_);
  }

  // 获取输出（RKNN会填充rknn_outputs_中的buf指针和size）
  ret = rknn_outputs_get(ctx_, io_num_.n_output, rknn_outputs_, nullptr);
  if (ret != RKNN_SUCC)
  {
    tools::logger()->error("Failed to get RKNN outputs");
    return std::list<Fire>();
  }
  // ===========================================

  // 解析输出
  float* output_data = (float*)rknn_outputs_[0].buf;
  std::list<Fire> result = parse(scale, output_data, raw_img, frame_count);

  // 注意：现在不在detect函数中释放RKNN输出内存
  // 我们将在下次调用rknn_outputs_get之前释放（见上面）
  // 或者在析构函数中统一处理

  return result;
}

std::list<Fire> Fire_YOLOV5::parse(
  double scale, float* output_data, const cv::Mat & bgr_img, int frame_count)
{
  const int NUM_BOXES = 25200;
  const int NUM_FEATURES = 6;

  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;

  // 遍历检测框
  for (int i = 0; i < NUM_BOXES; i++)
  {
    float* row_data = &output_data[i * NUM_FEATURES];
    float confidence = row_data[4];
    float class_score = row_data[5];

    float final_confidence = confidence * class_score;
    if (final_confidence < score_threshold_) continue;

    float x_center = row_data[0];
    float y_center = row_data[1];
    float width = row_data[2];
    float height = row_data[3];
    float x = x_center - width / 2.0f;
    float y = y_center - height / 2.0f;

    // 将坐标从640x640缩放到原始ROI图像尺寸
    x = x / static_cast<float>(scale);
    y = y / static_cast<float>(scale);
    width = width / static_cast<float>(scale);
    height = height / static_cast<float>(scale);

    // 如果使用了ROI，需要加上偏移量转换到原始图像坐标系
    if (use_roi_)
    {
      x += offset_.x;
      y += offset_.y;
    }

    // 确保边界框在图像范围内
    x = std::max(0.0f, x);
    y = std::max(0.0f, y);
    width = std::min(width, static_cast<float>(bgr_img.cols - x));
    height = std::min(height, static_cast<float>(bgr_img.rows - y));

    if (width <= 0 || height <= 0) continue;

    cv::Rect rect(static_cast<int>(x), static_cast<int>(y),
                  static_cast<int>(width), static_cast<int>(height));
    boxes.emplace_back(rect);
    confidences.push_back(final_confidence);
  }

  // 应用非极大值抑制(NMS)去除重叠框
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  // 创建Fire对象列表
  std::list<Fire> fire_list;
  for (int idx : indices)
  {
    Fire fire(boxes[idx], confidences[idx]);
    fire_list.push_back(fire);
  }

  if (debug_) draw_detections(bgr_img, fire_list, frame_count);

  return fire_list;
}

void Fire_YOLOV5::draw_detections(
  const cv::Mat & img, const std::list<Fire> & fires, int frame_count) const
{
  auto detection = img.clone();                                 // 创建输入图像的副本
  // 在图像左上角(10,30)位置显示当前帧号
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & fire : fires)                                   // 取出火结构体
  {
    cv::rectangle(detection, fire.bbox, cv::Scalar(0, 0, 255), 2);
    std::string label = fmt::format("Fire: {:.2f}", fire.confidence);
    cv::putText(detection, label,cv::Point(fire.bbox.x, fire.bbox.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    // 绘制中心点
    cv::circle(detection, fire.center, 3, cv::Scalar(0, 255, 0), -1);
  }
  if (use_roi_)
  {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  // 显示时缩小图片尺寸
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
}

}  // namespace auto_aim