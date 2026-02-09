#include "yolov5.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
YOLOV5::YOLOV5(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);                        // 获取yaml

  model_path_ = yaml["yolov5_model_path"].as<std::string>();            // 获得路径
  device_ = yaml["device"].as<std::string>();                           // CPU或者GPU
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();                // 未知
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();                                // 是否使用ROI
  use_traditional_ = yaml["use_traditional"].as<bool>();                // 是否使用传统方法
  roi_ = cv::Rect(x, y, width, height);                                 // 感兴趣区
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";                                                  // 保存图片路径
  std::filesystem::create_directory(save_path_);                     // 创建名为"imgs"的目录

  // core_是OpenVINO核心对象
  // read_model是从文件路径加载模型
  // 返回ov::Model对象，代表原始未编译的模型
  auto model = core_.read_model(model_path_);

  // 创建预处理/后处理处理器
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();                                           // 获取输入张量的【配置接口】

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 640, 640, 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);                // 颜色格式：BGR

  input.model().set_layout("NCHW");                                  // 设置模型期望的输入布局

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  // TODO: ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
  model = ppp.build();                                                 // 应用所有预处理/后处理配置

  // 模型编译
  // compiled_model_就是可以使用的模型了
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

std::list<Armor> YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty())
  {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;
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
    bgr_img = raw_img;
  }

  auto x_scale = static_cast<double>(640) / bgr_img.rows;
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  // 创建一个640×640的黑色背景
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  // 参数1：待缩放的图像
  // 参数2：缩放后图像存放的位置
  // 参数3：目标尺寸
  cv::resize(bgr_img, input(roi), {w, h});

  // 这里开始推理
  // 转换为OpenVINO张量
  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  // 创建推理请求并执行
  // compiled_model_：已编译的YOLO5模型
  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  // postprocess
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

  return parse(scale, output, raw_img, frame_count);
}

std::list<Armor> YOLOV5::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // 将输出从 (1, 25200, 22) 形状转置为 (22, 25200)
  // 这样每行是一个特征，每列是一个预测框，便于处理
  cv::transpose(output, output);

  std::vector<int> color_ids, num_ids;                                      // 颜色和数字
  std::vector<float> confidences;                                           // 置信度
  std::vector<cv::Rect> boxes;                                              // 边界框
  std::vector<std::vector<cv::Point2f>> armors_key_points;                  // 4个角点
  for (int r = 0; r < output.rows; r++)
  {
    double score = output.at<float>(r, 8);                                  // 第8列是目标置信度
    score = sigmoid(score);                                                 // 应用sigmoid激活函数

    if (score < score_threshold_) continue;                                 // 阈值过滤

    std::vector<cv::Point2f> armor_key_points;

    // ---------------------------------------------------------------------//
    // 这段代码的作用正是从YOLOv5模型的输出中提取检测到的装甲板的数字类别和颜色类别↓↓↓
    // 颜色和类别独热向量
    // output.row(r)
    // 获取 output 矩阵的第r行
    // 返回一个1×N的行向量（但仍然是一个Mat对象）
    // .colRange(13, 22)
    // 从刚才得到的行向量中取列索引13到21
    // 返回一个1×(22-13)=1×9的矩阵
    cv::Mat color_scores = output.row(r).colRange(9, 13);     // 颜色分类：第9-12列（4种颜色）
    cv::Mat classes_scores = output.row(r).colRange(13, 22);  // 数字类别：第13-22列（10个数字类别）
    cv::Point class_id, color_id;
    int _class_id, _color_id;
    double score_color, score_num;

    // 获取最大值的索引（颜色ID和数字ID）
    // cv::minMaxLoc 是 OpenCV 中一个用于查找数组中最小值和最大值及其位置的函数
    // 查找数字类别的最大值及其位置
    // classes_scores：包含10个数字类别的置信度分数
    // score_num：得到数字类别的最高置信度分数
    cv::minMaxLoc(classes_scores, NULL, &score_num, NULL, &class_id);
    cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);
    _class_id = class_id.x;                                               // 0-9，对应不同的装甲板数字
    _color_id = color_id.x;                                               // 0-3，对应不同的颜色

    // -------------------------------------------------------------------//
    // YOLOv5模型的输出中，前8列存储了装甲板的四个角点坐标
    // 这里获得了4个角点
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 0) / scale, output.at<float>(r, 1) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 6) / scale, output.at<float>(r, 7) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 4) / scale, output.at<float>(r, 5) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 2) / scale, output.at<float>(r, 3) / scale));

    // 初始化并计算四个角点的最小外接矩形（边界框）的范围
    float min_x = armor_key_points[0].x;
    float max_x = armor_key_points[0].x;
    float min_y = armor_key_points[0].y;
    float max_y = armor_key_points[0].y;

    for (int i = 1; i < armor_key_points.size(); i++)
    {
      if (armor_key_points[i].x < min_x) min_x = armor_key_points[i].x;
      if (armor_key_points[i].x > max_x) max_x = armor_key_points[i].x;
      if (armor_key_points[i].y < min_y) min_y = armor_key_points[i].y;
      if (armor_key_points[i].y > max_y) max_y = armor_key_points[i].y;
    }

    // 确定矩形
    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    // 保存
    color_ids.emplace_back(_color_id);                                  // 保存颜色
    num_ids.emplace_back(_class_id);                                    // 保存数字
    boxes.emplace_back(rect);                                           // 保存矩形
    confidences.emplace_back(score);                                    // 保存置信度
    armors_key_points.emplace_back(armor_key_points);                   // 保存角点
  }

  // 去除冗余的重叠检测框
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices)
  {
    if (use_roi_)
    {
      armors.emplace_back(
        color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    }
    else
    {
      armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  tmp_img_ = bgr_img;
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }
    // 使用传统方法二次矫正角点
    if (use_traditional_) detector_.detect(*it, bgr_img);

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

bool YOLOV5::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于神经网络的迭代
  // if (name_ok && !confidence_ok) save(armor);

  return name_ok && confidence_ok;
}

bool YOLOV5::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);

  // 保存异常的图案，用于神经网络的迭代
  // if (!name_ok) save(armor);

  return name_ok;
}

cv::Point2f YOLOV5::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLOV5::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
  cv::imshow("detection", detection);
}

void YOLOV5::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

double YOLOV5::sigmoid(double x)
{
  if (x > 0)
    return 1.0 / (1.0 + exp(-x));
  else
    return exp(x) / (1.0 + exp(x));
}

std::list<Armor> YOLOV5::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim