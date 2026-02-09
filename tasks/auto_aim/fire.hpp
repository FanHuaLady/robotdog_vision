#ifndef AUTO_AIM__FIRE_HPP
#define AUTO_AIM__FIRE_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace auto_aim
{

// 火焰数据结构
struct Fire
{
  cv::Rect bbox;                                          // 边界框
  cv::Point2f center;                                     // 中心点
  float confidence;                                       // 检测置信度

  Fire(const cv::Rect& box, float conf);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__FIRE_HPP