#include "io/camera.hpp"

#include <opencv2/opencv.hpp>

#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tasks/auto_aim/pose_yolo.hpp"   // 你的检测类头文件（根据实际情况调整）
#include "io/flower_usb/flower_usb.hpp"

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

  // 初始化摄像头（根据配置文件）
  io::Camera camera(config_path);
  // 初始化人体检测器（根据配置文件）
  auto_aim::Pose_YOLO yolo(config_path);

  const char* usb_device = "/dev/ttyACM0";
  if (io::usb_open(usb_device) != 0)
  {
    tools::logger()->warn("Failed to open USB device {}, continuing without USB.", usb_device);
    // 可以选择退出或继续运行，这里设为警告并继续
  }

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  auto last_stamp = std::chrono::steady_clock::now();

  for (int frame_count = 0; !exiter.exit(); frame_count++)
  {
    // 读取一帧
    camera.read(img, timestamp);
    if (img.empty())
    {
      tools::logger()->warn("Empty frame received, skipping...");
      continue;
    }

    // 执行检测
    auto detections = yolo.detect(img, frame_count);

    for (const auto& pose : detections)
    {
      // 构造要发送的字符串
      char buffer[64];
      int len = snprintf(buffer, sizeof(buffer), "x:%.2f,y:%.2f\n",
                         pose.center.x, pose.center.y);
      // 发送
      int sent = io::usb_send(buffer, len);
      printf("%s",buffer);
      if (sent != len)
      {
        tools::logger()->error("Failed to send USB data");
      }
    }

    /*
    // 立即尝试读取回显（STM32 原样返回）
    char recv_buf[256];
    int n = io::usb_recv(recv_buf, sizeof(recv_buf) - 1);
    if (n > 0)
    {
      recv_buf[n] = '\0';
      tools::logger()->info("USB echo: {}", recv_buf);
    }
    else if (n < 0)
    {
      tools::logger()->error("USB recv error");
    }
    */
    // 计算帧率
    // auto dt = tools::delta_time(timestamp, last_stamp);
    // last_stamp = timestamp;

    // 日志输出
    // tools::logger()->info("Frame {}: {:.2f} fps, {} detections",
    //                       frame_count, 1.0 / dt, detections.size());

    if (display)
    {
      // cv::imshow("Human Detection", img);
      if (cv::waitKey(1) == 'q')
        break;
    }
  }

  io::usb_close();
  return 0;
}