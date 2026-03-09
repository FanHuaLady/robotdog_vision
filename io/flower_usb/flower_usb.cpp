#include "flower_usb.hpp"
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <iostream>
#include <cstring>                                    // 提供 strerror
#include <cerrno>                                     // 提供 errno

namespace io
{
Flower_USB::Flower_USB(const char* device_path)
    : fd_(-1), running_(true)
{
    // 打开 USB 设备（以读写方式、非阻塞？可选项）
    fd_ = open(device_path, O_RDWR | O_NOCTTY);
    if (fd_ < 0)
    {
        // 记录错误，但继续运行（线程不会做实际发送）
        // 可以用日志系统，这里简单输出
        std::cerr << "Failed to open " << device_path << ": " << strerror(errno) << std::endl;
        return;
    }
    // 可设置串口参数（如波特率等），此处省略，假设已配置好
    // 启动发送线程
    sender_thread_ = std::thread(&Flower_USB::sender_loop, this);
}

Flower_USB::~Flower_USB()
{
    stop();
}

void Flower_USB::send(const std::string& data)
{
    if (!running_) return;                                            // 已停止，不再接收新数据
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(data);
    }
    cond_.notify_one();                                               // 唤醒发送线程
}

void Flower_USB::stop()
{
    if (!running_) return;
    running_ = false;
    cond_.notify_all();                                               // 唤醒发送线程，让其检查 running_ 并退出
    if (sender_thread_.joinable())
    {
        sender_thread_.join();                                        //
    }

    if (fd_ >= 0)
    {
        close(fd_);
        fd_ = -1;
    }
}

void Flower_USB::sender_loop()
{
    while (running_)                                                // running_ 控制循环是否继续
    {
        std::string data;
        {
            std::unique_lock<std::mutex> lock(mutex_);

            // 让发送线程在“无数据且未收到停止信号”时休眠，一旦有数据入队或要求停止，线程就会被唤醒并执行相应操作
            cond_.wait(lock, [this] { return !queue_.empty() || !running_; });

            if (!running_ && queue_.empty())                        // 收到停止信号且队列为空，退出循环
            {
                break;
            }
            if (!queue_.empty())
            {
                data = std::move(queue_.front());
                queue_.pop();
            }
        }
        // 发送数据（如果设备打开）
        if (fd_ >= 0 && !data.empty())                              // 发送数据
        {
            if (!usb_send_raw(data.c_str(), data.size()))
            {
                // 发送失败，可以记录日志，这里简单忽略
            }
        }
    }
}

bool Flower_USB::usb_send_raw(const char* data, size_t len)
{
    ssize_t ret = write(fd_, data, len);
    if (ret < 0)
    {
        // 错误处理
        return false;
    }
    // 如果发送部分字节，可以重试，为简化这里认为必须一次发完
    // 若未发完，可以缓存剩余部分，但简单场景下忽略
    return (size_t)ret == len;
}

}