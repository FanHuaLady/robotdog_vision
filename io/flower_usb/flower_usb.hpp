#ifndef FLOWER_USB_HPP
#define FLOWER_USB_HPP

#include <stddef.h>

namespace io
{
/**
 * @brief 打开 USB 虚拟串口设备
 * @param device 设备路径，如 "/dev/ttyACM0"
 * @return 成功返回 0，失败返回 -1
 */
int usb_open(const char *device);

/**
 * @brief 关闭已打开的串口
 */
void usb_close(void);

/**
 * @brief 发送数据
 * @param data 数据缓冲区
 * @param len  要发送的字节数
 * @return 实际发送的字节数，失败返回 -1
 */
int usb_send(const char *data, size_t len);

/**
 * @brief 接收数据（非阻塞超时模式）
 * @param buf    接收缓冲区
 * @param maxlen 缓冲区最大长度
 * @return 实际接收的字节数（0 表示超时无数据），失败返回 -1
 */
int usb_recv(char *buf, size_t maxlen);

}

#endif // FLOWER_USB_HPP
