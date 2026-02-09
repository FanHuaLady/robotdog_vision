#ifndef AUTO_AIM__FIRE_YOLOV5_HPP
#define AUTO_AIM__FIRE_YOLOV5_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/fire.hpp"
#include "tasks/auto_aim/detector.hpp"

#include "tasks/auto_aim/fire_yolo.hpp"

extern "C" {
#include "rknn_api.h"
}

namespace auto_aim
{
class Fire_YOLOV5 : public Fire_YOLOBase
{
public:
  Fire_YOLOV5(const std::string & config_path, bool debug);

  std::list<Fire> detect(const cv::Mat & bgr_img, int frame_count) override;

private:
  std::string model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_, use_traditional_;

  const int class_num_ = 1;
  const float nms_threshold_ = 0.3;
  const float score_threshold_ = 0.5;
  double min_confidence_, binary_threshold_;

  // RKNN相关成员变量
  rknn_context ctx_;                       // RKNN上下文
  rknn_input_output_num io_num_;           // 输入输出数量信息
  std::vector<uint8_t> model_data_;        // 存储模型文件数据

  rknn_input rknn_input_;                  // 复用的输入结构体
  rknn_output* rknn_outputs_;              // 复用的输出结构体数组
  cv::Mat input_buffer_;                   // 复用的输入图像缓冲区

  cv::Rect roi_;
  cv::Point2f offset_;

  Detector detector_;

  std::list<Fire> parse(double scale, float* output_data, const cv::Mat & bgr_img, int frame_count);
  void draw_detections(const cv::Mat & img, const std::list<Fire> & fires, int frame_count) const;
};

}  // namespace auto_aim

#endif  //AUTO_AIM__FIRE_YOLOV5_HPP