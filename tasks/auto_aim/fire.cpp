#include "fire.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace auto_aim
{

Fire::Fire(const cv::Rect& box, float conf)
  : bbox(box),confidence(conf)
{
  // 根据边界框计算中心点坐标
  center.x = box.x + box.width / 2.0f;
  center.y = box.y + box.height / 2.0f;
}

}  // namespace auto_aim