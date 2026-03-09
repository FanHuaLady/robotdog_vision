#ifndef FLOWER_USB_HPP
#define FLOWER_USB_HPP

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <atomic>
#include <stddef.h>

namespace io
{

class Flower_USB
{
public:
  Flower_USB(const char* device_path);              // 构造函数：打开 USB 设备，启动发送线程
  ~Flower_USB();                                    // 析构函数：停止线程，关闭设备

  void send(const std::string& data);               // 发送数据（线程安全，非阻塞）
  void stop();                                      // 显式停止发送线程（可选，一般由析构自动调用）

  bool is_open() const { return fd_ >= 0; }         // 检查设备是否成功打开

private:
  int fd_;                                          // USB 设备文件描述符
  std::thread sender_thread_;                       // 后台发送线程
  std::queue<std::string> queue_;                   // 数据队列
  std::mutex mutex_;                                // 队列锁
  std::condition_variable cond_;                    // 条件变量
  std::atomic<bool> running_;                       // 线程运行标志

  void sender_loop();                               // 发送线程主函数
  bool usb_send_raw(const char* data, size_t len);  // 实际的 USB 发送操作（封装 write 调用）
};

}

#endif // FLOWER_USB_HPP
