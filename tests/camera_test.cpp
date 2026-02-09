#include "io/camera.hpp"

#include <opencv2/opencv.hpp>

#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明}"
  "{config-path c  | configs/camera.yaml | yaml配置文件路径 }"
  "{d display      | true                | 显示视频流       }";

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
  auto display = cli.has("display");                        // 是否显示视频流
  io::Camera camera(config_path);

  cv::Mat img;                                                            // opencv中用于存储和操作图像/矩阵数据的类
  std::chrono::steady_clock::time_point timestamp;                        //
  auto last_stamp = std::chrono::steady_clock::now(); // 获得当前时间
  while (!exiter.exit())
  {
    camera.read(img, timestamp);                                  // 读摄像头

    auto dt = tools::delta_time(timestamp, last_stamp);       // 计算时间差
    last_stamp = timestamp;

    tools::logger()->info("{:.2f} fps", 1 / dt);                   // 这是一个格式化日志输

    if (!display) continue;
    cv::imshow("img", img);                                 // 显示
    if (cv::waitKey(1) == 'q') break;
  }
}
