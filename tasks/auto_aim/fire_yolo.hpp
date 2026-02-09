#ifndef AUTO_AIM__FIRE_YOLO_HPP
#define AUTO_AIM__FIRE_YOLO_HPP

#include <opencv2/opencv.hpp>

#include "fire.hpp"

namespace auto_aim
{
class Fire_YOLOBase
{
public:
  virtual std::list<Fire> detect(const cv::Mat & img, int frame_count) = 0;
};

class Fire_YOLO
{
public:
  Fire_YOLO(const std::string & config_path, bool debug = true);

  std::list<Fire> detect(const cv::Mat & img, int frame_count = -1);

private:
  std::unique_ptr<Fire_YOLOBase> yolo_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__FIRE_YOLO_HPP