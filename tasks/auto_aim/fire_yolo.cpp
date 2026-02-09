#include "fire_yolo.hpp"

#include <yaml-cpp/yaml.h>

#include "yolos/fire_yolov5.hpp"

namespace auto_aim
{
Fire_YOLO::Fire_YOLO(const std::string & config_path, bool debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();                       // 获取配置文件的yolo_name

  if (yolo_name == "yolov5")
  {
    yolo_ = std::make_unique<Fire_YOLOV5>(config_path, debug);
  }
  else
  {
    throw std::runtime_error("Unknown yolo name: " + yolo_name + "!");
  }
}

std::list<Fire> Fire_YOLO::detect(const cv::Mat & img, int frame_count)
{
  return yolo_->detect(img, frame_count);
}

}  // namespace auto_aim