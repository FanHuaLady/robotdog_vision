#ifndef FLOWER_USB_HPP
#define FLOWER_USB_HPP

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <atomic>
#include <cstddef>

namespace io
{

class Flower_USB
{
public:
  Flower_USB(const char* device_path);              // 构造函数：打开 USB 设备，启动发送线程
  ~Flower_USB();                                     // 析构函数：停止线程，关闭设备

  void send(const std::string& data);                // 发送数据（线程安全，非阻塞）
  std::string try_receive();                         // 非阻塞接收一条完整行
  void stop();                                        // 显式停止线程

  bool is_open() const { return fd_ >= 0; }          // 检查设备是否成功打开

private:
  int fd_;                                            // USB 设备文件描述符
  std::atomic<bool> running_;                         // 线程运行标志

  // 发送相关
  std::thread sender_thread_;
  std::queue<std::string> send_queue_;
  std::mutex send_mutex_;
  std::condition_variable send_cond_;

  // 接收相关
  std::thread receiver_thread_;
  std::queue<std::string> recv_queue_;                // 存放已完整接收的行
  std::mutex recv_mutex_;
  std::string recv_buffer_;                            // 缓存不完整的行数据

  void sender_loop();
  void receiver_loop();
};

} // namespace io

#endif // FLOWER_USB_HPP