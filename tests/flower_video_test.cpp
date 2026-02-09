#include <fmt/core.h>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/fire_yolo.hpp"
#include "tools/exiter.hpp"

const std::string keys =
  "{help h usage ? |                            | 输出命令行参数说明 }"
  "{config-path c  | configs/flower.yaml        | yaml配置文件的路径}"
  "{start-index s  | 0                          | 视频起始帧下标    }"
  "{end-index e    | 0                          | 视频结束帧下标    }"
  "{@input-path    | assets/demo/fire_demo      | avi和txt文件的路径}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help"))                                              // 检查是否有help参数
  {
    cli.printMessage();                                                      // 打印帮助信息
    return 0;
  }

  // 获取位置参数（索引从0开始）
  auto input_path = cli.get<std::string>(0);                            // 对应@input-path

  // 获取选项参数（通过名称）
  auto config_path = cli.get<std::string>("config-path");           // 配置文件路径
  auto start_index = cli.get<int>("start-index");
  auto end_index = cli.get<int>("end-index");

  // 用于退出
  tools::Exiter exiter;

  auto video_path = fmt::format("{}.avi", input_path);      // 组装路径

  cv::VideoCapture video(video_path);                                        // 视频输入
  auto_aim::Fire_YOLO yolo(config_path);                                     // 目标检测

  cv::Mat img;

  // 视频帧和姿态数据的同步对齐操作
  // 设置视频读取位置，从第start_index帧开始读取
  // cv::CAP_PROP_POS_FRAMES是OpenCV视频属性，表示以帧数为单位的当前位置
  video.set(cv::CAP_PROP_POS_FRAMES, start_index);                           //

  // 只有按退出才能退出 frame_count只是计数
  for (int frame_count = start_index; !exiter.exit(); frame_count++)
  {
    if (end_index > 0 && frame_count > end_index) break;                     // 视频自然结束会退出循环
    video.read(img);                                                      // 不断读出视频中的图片帧
    if (img.empty()) break;                                                  // 读出图片为空会退出循环

    auto fires = yolo.detect(img, frame_count);                      // 检测到的火焰列表

    auto key = cv::waitKey(30);
    if (key == 'q') break;
  }

  // 释放资源
  video.release();
  cv::destroyAllWindows();

  return 0;
}