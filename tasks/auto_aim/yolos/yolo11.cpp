#include "yolo11.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
YOLO11::YOLO11(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);                    // 获取yaml

  model_path_ = yaml["yolo11_model_path"].as<std::string>();        // 获得路径
  device_ = yaml["device"].as<std::string>();                       // CPU或者GPU
  binary_threshold_ = yaml["threshold"].as<double>();               // 未知
  min_confidence_ = yaml["min_confidence"].as<double>();            // 未知
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();                            // 是否使用ROI
  roi_ = cv::Rect(x, y, width, height);                             // 感兴趣区
  offset_ = cv::Point2f(x, y);                                      // 未知

  save_path_ = "imgs";                                              // 保存图片路径
  std::filesystem::create_directory(save_path_);                 // 创建名为"imgs"的目录

  // core_是OpenVINO核心对象
  // read_model是从文件路径加载模型
  // 返回ov::Model对象，代表原始未编译的模型
  auto model = core_.read_model(model_path_);

  // 创建预处理/后处理处理器
  ov::preprocess::PrePostProcessor ppp(model);                      //
  auto & input = ppp.input();                                       // 获取输入张量的【配置接口】
  // 设置输入张量属性
  input.tensor()
    .set_element_type(ov::element::u8)                              // 元素类型：8位无符号整数（0-255）
    .set_shape({1, 640, 640, 3})                     // 形状：1个批次，640高，640宽，3通道
    .set_layout("NHWC")                                          // NHWC是OpenCV图像的自然布局
    .set_color_format(ov::preprocess::ColorFormat::BGR);            // 颜色格式：BGR

  input.model().set_layout("NCHW");                              // 设置模型期望的输入布局

  input.preprocess()
    .convert_element_type(ov::element::f32)                         // 数据类型转换：uint8 → float32
    .convert_color(ov::preprocess::ColorFormat::RGB)                // 颜色空间转换：BGR → RGB
    .scale(255.0);

  // TODO: ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
  model = ppp.build();                                              // 应用所有预处理/后处理配置

  // 模型编译
  // compiled_model_就是可以使用的模型了
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

std::list<Armor> YOLO11::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;                                                            // 被ROI处理过的图像
  tmp_img_ = raw_img;
  // 如果启用ROI，只处理感兴趣区域，提高效率
  if (use_roi_)
  {
    if (roi_.width == -1)                                                     // -1 表示该维度不裁切
    {
      roi_.width = raw_img.cols;                                              // 就自动填充为图像原始尺寸
    }
    if (roi_.height == -1)                                                    // -1 表示该维度不裁切
    {
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);                                                  // 从原始图像中提取ROI区域
  }
  else
  {
    bgr_img = raw_img;
  }

  // 这是计算图像缩放比例和尺寸，用于保持宽高比的预处理
  auto x_scale = static_cast<double>(640) / bgr_img.rows;             // 高度方向缩放比例
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);                       // 缩放
  auto h = static_cast<int>(bgr_img.rows * scale);                           // 缩放后的高度
  auto w = static_cast<int>(bgr_img.cols * scale);
  // 计算缩放比例，保持宽高比
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
  // compiled_model_：已编译的YOLO11模型
  auto infer_request = compiled_model_.create_infer_request();              // 推理请求对象，用于管理单次推理
  infer_request.set_input_tensor(input_tensor);                             // 将输入数据传递给模型
  infer_request.infer();                                                    // 运行YOLO模型

  // 获取输出张量
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();

  // 转换为OpenCV Mat
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

  // 这里是将这些参数传入parse函数
  // parse返回类型是std::list<Armor>
  // 参数1：图像缩放比例
  // 参数2：模型推理结果张量
  // 参数3：原始输入图像
  // 参数4：帧计数器
  return parse(scale, output, raw_img, frame_count);
}

// 此函数会返回装甲板对象列表
std::list<Armor> YOLO11::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // OpenCV的矩阵转置操作
  // 将矩阵的行和列进行交换。如果原始矩阵是 m×n，转置后变为 n×m
  cv::transpose(output, output);

  std::vector<int> ids;                                                   // 装甲板的类别ID
  std::vector<float> confidences;                                         // 检测置信度分数
  std::vector<cv::Rect> boxes;                                            // 检测框的边界【矩形】
  std::vector<std::vector<cv::Point2f>> armors_key_points;                // 关键点

  // output.rows：检测框的总数？？？
  for (int r = 0; r < output.rows; r++)
  {
    // 边界信息[x, y, w, h]
    auto xywh = output.row(r).colRange(0, 4);            // 边界信息[x, y, w, h]
    // 类别置信度
    auto scores = output.row(r).colRange(4, 4 + class_num_);
    // 关键点坐标
    auto one_key_points = output.row(r).colRange(4 + class_num_, 50);

    std::vector<cv::Point2f> armor_key_points;

    double score;
    cv::Point max_point;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &max_point);

    if (score < score_threshold_) continue;                               // 置信度过滤

    // 提取边界框参数
    auto x = xywh.at<float>(0);
    auto y = xywh.at<float>(1);
    auto w = xywh.at<float>(2);
    auto h = xywh.at<float>(3);
    auto left = static_cast<int>((x - 0.5 * w) / scale);
    auto top = static_cast<int>((y - 0.5 * h) / scale);
    auto width = static_cast<int>(w / scale);
    auto height = static_cast<int>(h / scale);

    // 提取关键点坐标
    for (int i = 0; i < 4; i++)
    {
      float x = one_key_points.at<float>(0, i * 2 + 0) / scale;
      float y = one_key_points.at<float>(0, i * 2 + 1) / scale;
      cv::Point2f kp = {x, y};
      armor_key_points.push_back(kp);
    }

    // 存储检测结果
    ids.emplace_back(max_point.x);                                      // 找到的最高置信度对应的列索引
    confidences.emplace_back(score);                                    // 存储当前检测框的置信度分数
    boxes.emplace_back(left, top, width, height);                   // 存储当前检测框的边界矩形
    armors_key_points.emplace_back(armor_key_points);                   // 存储当前检测框的4个关键点坐标
  }

  // 非极大值抑制（NMS）
  // 移除重叠度高且置信度低的重复检测框，保留最佳检测结果
  std::vector<int> indices;
  // 参数1：boxes所有检测框的边界矩形
  // 参数2：confidences对应的置信度分数
  // 参数3：score_threshold_置信度阈值（如0.5），过滤低置信度框
  // 参数4：nms_threshold_NMS阈值（如0.45），重叠度高于此值的框会被抑制
  // 参数5：indices输出参数，存储通过NMS的检测框索引
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices)
  {
    sort_keypoints(armors_key_points[i]);
    if (use_roi_)
    {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    }
    else
    {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  for (auto it = armors.begin(); it != armors.end();)
  {
    if (!check_name(*it))
    {
      it = armors.erase(it);
      continue;
    }

    if (!check_type(*it))
    {
      it = armors.erase(it);
      continue;
    }

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

bool YOLO11::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于神经网络的迭代
  // if (name_ok && !confidence_ok) save(armor);

  return name_ok && confidence_ok;
}

bool YOLO11::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);

  // 保存异常的图案，用于神经网络的迭代
  // if (!name_ok) save(armor);

  return name_ok;
}

cv::Point2f YOLO11::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLO11::sort_keypoints(std::vector<cv::Point2f> & keypoints)
{
  if (keypoints.size() != 4) {
    std::cout << "beyond 4!!" << std::endl;
    return;
  }

  std::sort(keypoints.begin(), keypoints.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.y < b.y;
  });

  std::vector<cv::Point2f> top_points = {keypoints[0], keypoints[1]};
  std::vector<cv::Point2f> bottom_points = {keypoints[2], keypoints[3]};

  std::sort(top_points.begin(), top_points.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.x < b.x;
  });

  std::sort(
    bottom_points.begin(), bottom_points.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });

  keypoints[0] = top_points[0];     // top-left
  keypoints[1] = top_points[1];     // top-right
  keypoints[2] = bottom_points[1];  // bottom-right
  keypoints[3] = bottom_points[0];  // bottom-left
}

void YOLO11::draw_detections(
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

  if (use_roi_)
  {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
  cv::imshow("detection", detection);
}

void YOLO11::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

std::list<Armor> YOLO11::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim