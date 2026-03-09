#include "io/camera.hpp"
#include <opencv2/opencv.hpp>
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tasks/auto_aim/pose_yolo.hpp"
#include "io/flower_usb/flower_usb.hpp"   // 包含新类头文件

const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明}"
  "{config-path c  | configs/pose.yaml    | yaml配置文件路径 }"
  "{d display      | true                 | 显示视频流       }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help"))
  {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;

  auto config_path = cli.get<std::string>("config-path");
  auto display = cli.has("display");

  // 初始化摄像头
  io::Camera camera(config_path);
  // 初始化人体检测器
  auto_aim::Pose_YOLO yolo(config_path);

  // 创建异步 USB 发送器（设备路径固定，可根据需要改为从配置文件读取）
  io::Flower_USB usb_sender("/dev/ttyACM0");
  if (!usb_sender.is_open())
  {
    tools::logger()->warn("Failed to open USB device /dev/ttyACM0, continuing without USB.");
    // 程序可以继续运行，只是无法发送数据
  }

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  auto last_stamp = std::chrono::steady_clock::now();

  for (int frame_count = 0; !exiter.exit(); frame_count++)
  {
    camera.read(img, timestamp);
    if (img.empty())
    {
      tools::logger()->warn("Empty frame received, skipping...");
      continue;
    }

    auto detections = yolo.detect(img, frame_count);

    for (const auto& pose : detections)
    {
      char buffer[64];
      int len = snprintf(buffer, sizeof(buffer), "x:%.2f,y:%.2f\n",
                         pose.center.x, pose.center.y);
      // 异步发送：将字符串入队，立即返回
      usb_sender.send(std::string(buffer, len));
      // 可选：在终端打印坐标（调试用）
      printf("%s", buffer);
    }

    // 帧率计算与日志（可取消注释）
    // auto dt = tools::delta_time(timestamp, last_stamp);
    // last_stamp = timestamp;
    // tools::logger()->info("Frame {}: {:.2f} fps, {} detections",
    //                       frame_count, 1.0 / dt, detections.size());

    if (display)
    {
      // cv::imshow("Human Detection", img);
      if (cv::waitKey(1) == 'q')
        break;
    }
  }

  // usb_sender 析构时会自动停止线程、关闭设备，无需显式调用 close
  return 0;
}