#include "io/camera.hpp"
#include <opencv2/opencv.hpp>
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tasks/auto_aim/pose_yolo.hpp"
#include "io/flower_usb/flower_usb.hpp"

// cmake -B build
// make -C build/ -j`nproc`
// ./build/flower_test
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

  io::Camera camera(config_path);
  auto_aim::Pose_YOLO yolo(config_path);

  io::Flower_USB usb_sender("/dev/ttyACM0");
  if (!usb_sender.is_open())
  {
    tools::logger()->warn("Failed to open USB device /dev/ttyACM0, continuing without USB.");
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
      usb_sender.send(std::string(buffer, len));
    }

    // 接收并打印回复
    std::string reply = usb_sender.try_receive();
    while (!reply.empty())
    {
      tools::logger()->info("USB reply: {}", reply.c_str());
      reply = usb_sender.try_receive();
    }

    auto dt = tools::delta_time(timestamp, last_stamp);
    last_stamp = timestamp;
    tools::logger()->info("Frame {}: {:.2f} fps, {} detections",
                           frame_count, 1.0 / dt, detections.size());

    if (display)
    {
      // cv::imshow("Detection", img);
      if (cv::waitKey(1) == 'q')
        break;
    }
  }

  return 0;
}