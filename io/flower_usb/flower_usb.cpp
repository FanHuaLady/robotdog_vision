#include "flower_usb.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <errno.h>
#include <cstring>
#include <iostream>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cctype>   // for isprint

namespace io
{

Flower_USB::Flower_USB(const char* device_path)
    : fd_(-1), running_(true)
{
    fd_ = open(device_path, O_RDWR | O_NOCTTY);
    if (fd_ < 0)
    {
        std::cerr << "Failed to open " << device_path << ": " << strerror(errno) << std::endl;
        return;
    }

    // 设置为非阻塞模式
    int flags = fcntl(fd_, F_GETFL, 0);
    if (flags != -1)
    {
        fcntl(fd_, F_SETFL, flags | O_NONBLOCK);
    }

    // 设置为原始模式，避免行缓冲和转换
    struct termios tty;
    if (tcgetattr(fd_, &tty) == 0)
    {
        cfmakeraw(&tty);
        tcsetattr(fd_, TCSANOW, &tty);
    }

    // 启动线程
    sender_thread_ = std::thread(&Flower_USB::sender_loop, this);
    receiver_thread_ = std::thread(&Flower_USB::receiver_loop, this);
}

Flower_USB::~Flower_USB()
{
    stop();
}

void Flower_USB::send(const std::string& data)
{
    if (!running_) return;
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        send_queue_.push(data);
    }
    send_cond_.notify_one();
}

std::string Flower_USB::try_receive()
{
    std::lock_guard<std::mutex> lock(recv_mutex_);
    if (recv_queue_.empty()) {
        return {};
    }
    std::string line = std::move(recv_queue_.front());
    recv_queue_.pop();
    return line;
}

void Flower_USB::stop()
{
    if (!running_.exchange(false)) return;
    send_cond_.notify_all();  // 唤醒发送线程
    if (sender_thread_.joinable()) sender_thread_.join();
    if (receiver_thread_.joinable()) receiver_thread_.join();
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

void Flower_USB::sender_loop()
{
    while (running_)
    {
        std::string data;
        {
            std::unique_lock<std::mutex> lock(send_mutex_);
            send_cond_.wait(lock, [this] { return !send_queue_.empty() || !running_; });
            if (!running_ && send_queue_.empty()) break;
            if (!send_queue_.empty()) {
                data = std::move(send_queue_.front());
                send_queue_.pop();
            }
        }

        if (fd_ < 0 || data.empty()) continue;

        ssize_t ret = write(fd_, data.c_str(), data.size());
        if (ret < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // 暂时不可写，放回队尾稍后重试
                std::lock_guard<std::mutex> lock(send_mutex_);
                send_queue_.push(data);
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            } else {
                // 硬错误，丢弃数据
                std::cerr << "USB write error: " << strerror(errno) << std::endl;
            }
        } else if ((size_t)ret < data.size()) {
            // 部分发送，剩余部分重新入队
            std::string remaining = data.substr(ret);
            std::lock_guard<std::mutex> lock(send_mutex_);
            send_queue_.push(remaining);
            send_cond_.notify_one();  // 立即继续发送剩余部分
        }
        // 全部发送成功则继续循环
    }
}

void Flower_USB::receiver_loop()
{
    const size_t buf_size = 256;
    char buf[buf_size];
    while (running_)
    {
        if (fd_ < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        ssize_t n = read(fd_, buf, buf_size);
        if (n > 0) 
        {
            recv_buffer_.append(buf, n);
            size_t pos;
            while ((pos = recv_buffer_.find('\n')) != std::string::npos) {
                std::string line = recv_buffer_.substr(0, pos);
                if (!line.empty() && line.back() == '\r') line.pop_back(); // 去除可能结尾的 \r
                {
                    std::lock_guard<std::mutex> lock(recv_mutex_);
                    recv_queue_.push(line);
                }
                recv_buffer_.erase(0, pos + 1);
            }
        } else if (n == 0) {
            // 可能设备关闭，短暂休眠
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } else {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                // 读错误
                std::cerr << "USB read error: " << strerror(errno) << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } else {
                // 无数据，短暂休眠
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
}

} // namespace io