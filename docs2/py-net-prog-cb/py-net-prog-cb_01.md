# 第一章：套接字、IPv4 和简单的客户端/服务器编程

在本章中，我们将涵盖以下菜谱：

+   打印你的机器名称和 IPv4 地址

+   获取远程机器的 IP 地址

+   将 IPv4 地址转换为不同的格式

+   根据端口号和协议查找服务名

+   将整数从主机字节序转换为网络字节序，反之亦然

+   设置和获取默认套接字超时

+   优雅地处理套接字错误

+   修改套接字的发送/接收缓冲区大小

+   将套接字改为阻塞/非阻塞模式

+   重新使用套接字地址

+   从互联网时间服务器打印当前时间

+   编写 SNTP 客户端

+   编写简单的回声客户端/服务器应用程序

# 简介

本章通过一些简单的菜谱介绍了 Python 的核心网络库。Python 的`socket`模块既有基于类的工具，也有基于实例的工具。基于类的和基于实例的方法之间的区别在于前者不需要套接字对象的实例。这是一个非常直观的方法。例如，为了打印你的机器的 IP 地址，你不需要套接字对象。相反，你只需调用套接字的基于类的方法。另一方面，如果你需要向服务器应用程序发送一些数据，创建一个套接字对象来执行该显式操作会更直观。本章中提供的菜谱可以分为以下三个组：

+   在前几个菜谱中，已经使用了基于类的工具来提取有关主机、网络和任何目标服务的有用信息。

+   之后，使用基于实例的工具提供了一些更多的菜谱。演示了一些常见的套接字任务，包括操作套接字超时、缓冲区大小、阻塞模式等等。

+   最后，使用基于类的和基于实例的工具构建了一些客户端，它们执行一些实际的任务，例如将机器时间与互联网服务器同步或编写通用的客户端/服务器脚本。

你可以使用这些演示的方法来编写你自己的客户端/服务器应用程序。

# 打印你的机器名称和 IPv4 地址

有时候，你需要快速查找有关你的机器的一些信息，例如主机名、IP 地址、网络接口数量等等。使用 Python 脚本实现这一点非常简单。

## 准备工作

在开始编码之前，你需要在你的机器上安装 Python。Python 在大多数 Linux 发行版中都是预安装的。对于 Microsoft Windows 操作系统，你可以从 Python 网站下载二进制文件：[`www.python.org/download/`](http://www.python.org/download/)

你可以查阅你操作系统的文档，检查和审查你的 Python 设置。在你机器上安装 Python 之后，你可以通过在命令行中键入`python`来尝试打开 Python 解释器。这将显示解释器提示符`>>>`，应该类似于以下输出：

```py
~$ python 
Python 2.7.1+ (r271:86832, Apr 11 2011, 18:05:24) 
[GCC 4.5.2] on linux2 
Type "help", "copyright", "credits" or "license" for more information. >>> 

```

## 如何操作...

由于这个菜谱非常简短，您可以在 Python 解释器中交互式地尝试它。

首先，我们需要使用以下命令导入 Python 的`socket`库：

```py
>>> import socket

```

然后，我们调用`socket`库中的`gethostname()`方法，并将结果存储在变量中，如下所示：

```py
>>> host_name = socket.gethostname()
>>> print "Host name: %s" %host_name
Host name: debian6
>>> print "IP address: %s" %socket.gethostbyname(host_name)
IP address: 127.0.1.1

```

整个活动可以封装在一个独立的函数`print_machine_info()`中，它使用内置的 socket 类方法。

我们从通常的 Python `__main__`块调用我们的函数。在运行时，Python 将值分配给一些内部变量，例如`__name__`。在这种情况下，`__name__`指的是调用进程的名称。当从命令行运行此脚本时，如以下命令所示，名称将是`__main__`，但如果模块是从另一个脚本导入的，则名称将不同。这意味着当模块从命令行调用时，它将自动运行我们的`print_machine_info`函数；然而，当单独导入时，用户需要显式调用该函数。

列表 1.1 显示了如何获取我们的机器信息，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter -1 
# This program is optimized for Python 2.7\. It may run on any
# other Python version with/without modifications.

import socket

def print_machine_info():
    host_name = socket.gethostname()
    ip_address = socket.gethostbyname(host_name)
    print "Host name: %s" % host_name
    print "IP address: %s" % ip_address

if __name__ == '__main__':
    print_machine_info()
```

为了运行这个菜谱，您可以使用提供的源文件从命令行如下操作：

```py
$ python 1_1_local_machine_info.py

```

在我的机器上，以下输出如下：

```py
Host name: debian6
IP address: 127.0.0.1

```

这个输出将取决于您机器的系统主机配置而有所不同。

## 它是如何工作的...

`import socket`语句导入 Python 的核心网络库之一。然后，我们使用两个实用函数，`gethostname()`和`gethostbyname(host_name)`。您可以在命令行中输入`help(socket.gethostname)`来查看在线帮助信息。或者，您可以在您的网络浏览器中输入以下地址：[`docs.python.org/3/library/socket.html`](http://docs.python.org/3/library/socket.html)。您可以参考以下命令：

```py
gethostname(...)
 gethostname() -> string 
 Return the current host name. 

gethostbyname(...) 
 gethostbyname(host) -> address 
 Return the IP address (a string of the form '255.255.255.255') for a host.

```

第一个函数不接受任何参数，并返回当前或本地主机名。第二个函数接受一个`hostname`参数，并返回其 IP 地址。

# 获取远程机器的 IP 地址

有时候，您需要将一台机器的主机名转换为其对应的 IP 地址，例如，进行快速域名查找。这个菜谱介绍了一个简单的函数来完成这个任务。

## 如何操作...

如果您需要知道远程机器的 IP 地址，您可以使用内置库函数`gethostbyname()`。在这种情况下，您需要将远程主机名作为其参数传递。

在这种情况下，我们需要调用`gethostbyname()`类函数。让我们看看这个简短代码片段的内部。

列表 1.2 显示了如何获取远程机器的 IP 地址，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter – 1
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

import socket

def get_remote_machine_info():
    remote_host = 'www.python.org'
    try:
        print "IP address: %s" %socket.gethostbyname(remote_host)
    except socket.error, err_msg:
        print "%s: %s" %(remote_host, err_msg)

if __name__ == '__main__':
    get_remote_machine_info()
```

如果您运行前面的代码，它将给出以下输出：

```py
$ python 1_2_remote_machine_info.py 
IP address of www.python.org: 82.94.164.162

```

## 它是如何工作的...

这个配方将`gethostbyname()`方法封装在一个用户定义的函数`get_remote_machine_info()`中。在这个配方中，我们介绍了异常处理的概念。正如你所看到的，我们将主要函数调用封装在一个`try-except`块中。这意味着如果在执行此函数期间发生某些错误，此错误将由这个`try-except`块处理。

例如，让我们更改`remote_host`值，并将[www.python.org](http://www.python.org)替换为不存在的某个内容，例如`www.pytgo.org`。现在运行以下命令：

```py
$ python 1_2_remote_machine_info.py 
www.pytgo.org: [Errno -5] No address associated with hostname

```

`try-except`块捕获错误并向用户显示错误消息，指出没有与主机名`www.pytgo.org`关联的 IP 地址。

# 将 IPv4 地址转换为不同的格式

当你想要处理低级网络功能时，有时，IP 地址的常规字符串表示法并不很有用。它们需要转换为 32 位的打包二进制格式。

## 如何操作...

Python 套接字库有处理各种 IP 地址格式的实用工具。在这里，我们将使用其中的两个：`inet_aton()`和`inet_ntoa()`。

让我们创建一个`convert_ip4_address()`函数，其中将使用`inet_aton()`和`inet_ntoa()`进行 IP 地址转换。我们将使用两个示例 IP 地址，`127.0.0.1`和`192.168.0.1`。

列表 1.3 显示了`ip4_address_conversion`如下：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter – 1
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

import socket
from binascii import hexlify

def convert_ip4_address():
    for ip_addr in ['127.0.0.1', '192.168.0.1']:
        packed_ip_addr = socket.inet_aton(ip_addr)
        unpacked_ip_addr = socket.inet_ntoa(packed_ip_addr)
        print "IP Address: %s => Packed: %s, Unpacked: %s"\
	 %(ip_addr, hexlify(packed_ip_addr), unpacked_ip_addr)

if __name__ == '__main__':
    convert_ip4_address()
```

现在，如果你运行这个配方，你会看到以下输出：

```py
$ python 1_3_ip4_address_conversion.py 

IP Address: 127.0.0.1 => Packed: 7f000001, Unpacked: 127.0.0.1
IP Address: 192.168.0.1 => Packed: c0a80001, Unpacked: 192.168.0.1

```

## 它是如何工作的...

在这个配方中，两个 IP 地址已经使用`for-in`语句从字符串转换为 32 位打包格式。此外，还调用了来自`binascii`模块的 Python `hexlify`函数。这有助于以十六进制格式表示二进制数据。

# 根据端口和协议查找服务名称

如果你想要发现网络服务，确定使用 TCP 或 UDP 协议在哪些端口上运行哪些网络服务可能会有所帮助。

## 准备工作

如果你知道网络服务的端口号，你可以使用套接字库中的`getservbyport()`套接字类函数来查找服务名称。在调用此函数时，你可以选择性地提供协议名称。

## 如何操作...

让我们定义一个`find_service_name()`函数，其中将使用`getservbyport()`套接字类函数调用几个端口，例如`80, 25`。我们可以使用 Python 的`for-in`循环结构。

列表 1.4 显示了`finding_service_name`如下：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter -  1
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

import socket

def find_service_name():
    protocolname = 'tcp'
    for port in [80, 25]:
        print "Port: %s => service name: %s" %(port, socket.getservbyport(port, protocolname))
    print "Port: %s => service name: %s" %(53, socket.getservbyport(53, 'udp'))

if __name__ == '__main__':
    find_service_name()
```

如果你运行此脚本，你会看到以下输出：

```py
$ python 1_4_finding_service_name.py 

Port: 80 => service name: http
Port: 25 => service name: smtp
Port: 53 => service name: domain

```

## 它是如何工作的...

在这个配方中，使用`for-in`语句遍历一系列变量。因此，对于每次迭代，我们使用一个 IP 地址，以打包和未打包的格式转换它们。

# 将整数从主机字节序转换为网络字节序以及反向转换

如果你需要编写一个低级网络应用程序，可能需要处理两个机器之间通过线缆的低级数据传输。这种操作需要将数据从本地主机操作系统转换为网络格式，反之亦然。这是因为每个都有自己的数据特定表示。

## 如何操作...

Python 的 socket 库提供了从网络字节序转换为主机字节序以及相反方向的工具。你可能需要熟悉它们，例如，`ntohl()`/`htonl()`。

让我们定义一个`convert_integer()`函数，其中使用`ntohl()`/`htonl()` socket 类函数来转换 IP 地址格式。

列表 1.5 显示了`integer_conversion`如下：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.
import socket
def convert_integer():
    data = 1234
    # 32-bit
    print "Original: %s => Long  host byte order: %s, Network byte order: %s"\
    %(data, socket.ntohl(data), socket.htonl(data))
    # 16-bit
    print "Original: %s => Short  host byte order: %s, Network byte order: %s"\
    %(data, socket.ntohs(data), socket.htons(data))
if __name__ == '__main__':
    convert_integer()
```

如果你运行这个菜谱，你会看到以下输出：

```py
$ python 1_5_integer_conversion.py 
Original: 1234 => Long  host byte order: 3523477504, Network byte order: 3523477504
Original: 1234 => Short  host byte order: 53764, Network byte order: 53764

```

## 它是如何工作的...

在这里，我们取一个整数并展示如何将其在网络字节序和主机字节序之间转换。`ntohl()` socket 类函数将网络字节序从长格式转换为主机字节序。在这里，`n`代表网络，`h`代表主机；`l`代表长，`s`代表短，即 16 位。

# 设置和获取默认 socket 超时

有时，你需要操作 socket 库某些属性的默认值，例如，socket 超时。

## 如何操作...

你可以创建一个 socket 对象实例，并调用`gettimeout()`方法来获取默认超时值，以及调用`settimeout()`方法来设置特定的超时值。这在开发自定义服务器应用程序时非常有用。

我们首先在`test_socket_timeout()`函数内部创建一个 socket 对象。然后，我们可以使用 getter/setter 实例方法来操作超时值。

列表 1.6 显示了`socket_timeout`如下：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 1
# This program is optimized for Python 2.7\. It may run on any   
# other Python version with/without modifications

import socket

def test_socket_timeout():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print "Default socket timeout: %s" %s.gettimeout()
    s.settimeout(100)
    print "Current socket timeout: %s" %s.gettimeout()    

if __name__ == '__main__':
    test_socket_timeout()
```

运行前面的脚本后，你可以看到以下修改默认 socket 超时的方式：

```py
$ python 1_6_socket_timeout.py 
Default socket timeout: None
Current socket timeout: 100.0

```

## 它是如何工作的...

在这个代码片段中，我们首先通过将 socket 家族和 socket 类型作为 socket 构造函数的第一个和第二个参数传递来创建一个 socket 对象。然后，你可以通过调用`gettimeout()`来获取 socket 超时值，并通过调用`settimeout()`方法来更改该值。传递给`settimeout()`方法的超时值可以是秒（非负浮点数）或`None`。此方法用于操作阻塞-socket 操作。将超时设置为`None`将禁用 socket 操作的超时。

# 优雅地处理 socket 错误

在任何网络应用程序中，一端尝试连接，而另一端由于网络媒体故障或其他原因没有响应是非常常见的情况。Python 的 socket 库通过`socket.error`异常提供了一个优雅的方法来处理这些错误。在这个菜谱中，展示了几个示例。

## 如何操作...

让我们创建一些 try-except 代码块，并在每个块中放入一个潜在的错误类型。为了获取用户输入，可以使用 `argparse` 模块。此模块比仅使用 `sys.argv` 解析命令行参数更强大。在 try-except 块中，可以放置典型的套接字操作，例如创建套接字对象、连接到服务器、发送数据和等待回复。

以下示例代码展示了如何用几行代码说明这些概念。

列表 1.7 展示了 `socket_errors` 如下：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter – 1
# This program is optimized for Python 2.7\. It may run on any   
# other Python version with/without modifications.

import sys
import socket
import argparse 

def main():
    # setup argument parsing
    parser = argparse.ArgumentParser(description='Socket Error Examples')
    parser.add_argument('--host', action="store", dest="host", required=False)
    parser.add_argument('--port', action="store", dest="port", type=int, required=False)
    parser.add_argument('--file', action="store", dest="file", required=False)
    given_args = parser.parse_args()
    host = given_args.host
    port = given_args.port
    filename = given_args.file

    # First try-except block -- create socket 
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error, e:
        print "Error creating socket: %s" % e
        sys.exit(1)

    # Second try-except block -- connect to given host/port
    try:
        s.connect((host, port))
    except socket.gaierror, e:
        print "Address-related error connecting to server: %s" % e
        sys.exit(1)
    except socket.error, e:
        print "Connection error: %s" % e
        sys.exit(1)

    # Third try-except block -- sending data
    try:
        s.sendall("GET %s HTTP/1.0\r\n\r\n" % filename)
    except socket.error, e:
        print "Error sending data: %s" % e
        sys.exit(1)

    while 1:
        # Fourth tr-except block -- waiting to receive data from remote host
        try:
            buf = s.recv(2048)
        except socket.error, e:
            print "Error receiving data: %s" % e
            sys.exit(1)
        if not len(buf):
            break
        # write the received data
        sys.stdout.write(buf) 

if __name__ == '__main__':
    main()
```

## 它是如何工作的...

在 Python 中，可以使用 `argparse` 模块将命令行参数传递给脚本并在脚本中解析它们。这个模块在 Python 2.7 中可用。对于更早的 Python 版本，此模块可以在 **Python 包索引**（**PyPI**）中单独安装。你可以通过 `easy_install` 或 `pip` 来安装它。

在这个示例中，设置了三个参数：主机名、端口号和文件名。此脚本的用法如下：

```py
$ python 1_7_socket_errors.py –host=<HOST> --port=<PORT> --file=<FILE>

```

如果你尝试使用一个不存在的宿主，此脚本将打印出如下地址错误：

```py
$ python 1_7_socket_errors.py --host=www.pytgo.org --port=8080 --file=1_7_socket_errors.py 
Address-related error connecting to server: [Errno -5] No address associated with hostname

```

如果特定端口没有服务，并且你尝试连接到该端口，那么这将引发连接超时错误，如下所示：

```py
$ python 1_7_socket_errors.py --host=www.python.org --port=8080 --file=1_7_socket_errors.py 

```

由于主机 [www.python.org](http://www.python.org) 没有监听 8080 端口，这将返回以下错误：

```py
Connection error: [Errno 110] Connection timed out

```

然而，如果你向正确的端口发送一个任意请求，错误可能不会被应用程序级别捕获。例如，运行以下脚本不会返回错误，但 HTML 输出告诉我们这个脚本有什么问题：

```py
$ python 1_7_socket_errors.py --host=www.python.org --port=80 --file=1_7_socket_errors.py

HTTP/1.1 404 Not found
Server: Varnish
Retry-After: 0
content-type: text/html
Content-Length: 77
Accept-Ranges: bytes
Date: Thu, 20 Feb 2014 12:14:01 GMT
Via: 1.1 varnish
Age: 0
Connection: close

<html>
<head>
<title> </title>
</head>
<body>
unknown domain: </body></html>

```

在前面的示例中，使用了四个 try-except 块。除了第二个块使用 `socket.gaierror` 外，所有块都使用 `socket.error`。`socket.gaierror` 用于地址相关错误。还有两种其他类型的异常：`socket.herror` 用于旧版 C API，如果你在套接字上使用 `settimeout()` 方法，当该套接字发生超时时，将引发 `socket.timeout`。

# 修改套接字的发送/接收缓冲区大小

默认套接字缓冲区大小在很多情况下可能不合适。在这种情况下，可以将默认套接字缓冲区大小更改为更合适的值。

## 如何操作...

让我们使用套接字对象的 `setsockopt()` 方法来调整默认套接字缓冲区大小。

首先，定义两个常量：`SEND_BUF_SIZE`/`RECV_BUF_SIZE`，然后在函数中包装套接字实例对 `setsockopt()` 方法的调用。在修改之前检查缓冲区大小也是一个好主意。请注意，我们需要分别设置发送和接收缓冲区大小。

列表 1.8 展示了如何修改套接字的发送/接收缓冲区大小如下：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter – 1
# This program is optimized for Python 2.7\. It may run on any
# other Python version with/without modifications

import socket

SEND_BUF_SIZE = 4096
RECV_BUF_SIZE = 4096

def modify_buff_size():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM )

    # Get the size of the socket's send buffer
    bufsize = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    print "Buffer size [Before]:%d" %bufsize

    sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_SNDBUF,
            SEND_BUF_SIZE)
    sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_RCVBUF,
            RECV_BUF_SIZE)
    bufsize = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    print "Buffer size [After]:%d" %bufsize

if __name__ == '__main__':
    modify_buff_size()
```

如果你运行前面的脚本，它将显示套接字缓冲区大小的变化。以下输出可能因操作系统的本地设置而异：

```py
$ python 1_8_modify_buff_size.py 
Buffer size [Before]:16384
Buffer size [After]:8192

```

## 它是如何工作的...

你可以在套接字对象上调用`getsockopt()`和`setsockopt()`方法来分别检索和修改套接字对象的属性。`setsockopt()`方法接受三个参数：`level`、`optname`和`value`。在这里，`optname`接受选项名称，`value`是相应选项的值。对于第一个参数，所需的符号常量可以在套接字模块（`SO_*etc.`）中找到。

# 将套接字转换为阻塞/非阻塞模式

默认情况下，TCP 套接字被置于阻塞模式。这意味着控制权不会返回到你的程序，直到某个特定操作完成。例如，如果你调用`connect()` API，连接会阻塞你的程序直到操作完成。在许多情况下，你不想让程序永远等待，无论是等待服务器的响应还是等待任何错误来停止操作。例如，当你编写一个连接到 Web 服务器的 Web 浏览器客户端时，你应该考虑一个可以在操作过程中取消连接过程的功能。这可以通过将套接字置于非阻塞模式来实现。

## 如何操作...

让我们看看 Python 中可用的选项。在 Python 中，套接字可以被置于阻塞或非阻塞模式。在非阻塞模式下，如果任何 API 调用，例如`send()`或`recv()`，遇到任何问题，将会引发错误。然而，在阻塞模式下，这不会停止操作。我们可以创建一个普通的 TCP 套接字，并实验阻塞和非阻塞操作。

为了操纵套接字的阻塞特性，我们首先需要创建一个套接字对象。

然后，我们可以调用`setblocking(1)`来设置阻塞或`setblocking(0)`来取消阻塞。最后，我们将套接字绑定到特定端口并监听传入的连接。

列表 1.9 显示了套接字如何转换为阻塞或非阻塞模式，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 1
# This program is optimized for Python 2.7\. It may run on any
# other Python version with/without modifications

import socket

def test_socket_modes():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setblocking(1)
    s.settimeout(0.5)
    s.bind(("127.0.0.1", 0))

    socket_address = s.getsockname()
    print "Trivial Server launched on socket: %s" %str(socket_address)
    while(1):
        s.listen(1)

if __name__ == '__main__':
    test_socket_modes()
```

如果你运行这个菜谱，它将启动一个具有阻塞模式启用的简单服务器，如下面的命令所示：

```py
$ python 1_9_socket_modes.py 
Trivial Server launched on socket: ('127.0.0.1', 51410)

```

## 它是如何工作的...

在这个菜谱中，我们通过在`setblocking()`方法中将值设置为`1`来在套接字上启用阻塞。同样，你可以在该方法中将值`0`取消设置，使其变为非阻塞。

这个特性将在一些后续的菜谱中重用，其真正目的将在那里详细阐述。

# 重复使用套接字地址

你希望套接字服务器始终在特定的端口上运行，即使它在有意或意外关闭后也是如此。在某些情况下，如果你的客户端程序始终连接到该特定服务器端口，这很有用。因此，你不需要更改服务器端口。

## 如何操作...

如果你在一个特定的端口上运行 Python 套接字服务器，并在关闭后尝试重新运行它，你将无法使用相同的端口。它通常会抛出如下错误：

```py
Traceback (most recent call last):
 File "1_10_reuse_socket_address.py", line 40, in <module>
 reuse_socket_addr()
 File "1_10_reuse_socket_address.py", line 25, in reuse_socket_addr
 srv.bind( ('', local_port) )
 File "<string>", line 1, in bind
socket.error: [Errno 98] Address already in use

```

解决这个问题的方法是为套接字启用重用选项`SO_REUSEADDR`。

在创建套接字对象后，我们可以查询地址重用的状态，比如说一个旧状态。然后，我们调用`setsockopt()`方法来改变其地址重用状态。然后，我们遵循绑定到地址和监听传入客户端连接的常规步骤。在这个例子中，我们捕获`KeyboardInterrupt`异常，这样如果您按下*Ctrl* + *C*，Python 脚本将不会显示任何异常消息而终止。

列表 1.10 显示了如何如下重用套接字地址：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 1
# This program is optimized for Python 2.7\. It may run on any
# other Python version with/without modifications

import socket
import sys

def reuse_socket_addr():
    sock = socket.socket( socket.AF_INET, socket.SOCK_STREAM )

    # Get the old state of the SO_REUSEADDR option
    old_state = sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR )
    print "Old sock state: %s" %old_state

    # Enable the SO_REUSEADDR option
    sock.setsockopt( socket.SOL_SOCKET, socket.SO_REUSEADDR, 1 )
    new_state = sock.getsockopt( socket.SOL_SOCKET, socket.SO_REUSEADDR )
    print "New sock state: %s" %new_state

    local_port = 8282

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind( ('', local_port) )
    srv.listen(1)
    print ("Listening on port: %s " %local_port)
    while True:
        try:
            connection, addr = srv.accept()
            print 'Connected by %s:%s' % (addr[0], addr[1])
        except KeyboardInterrupt:
            break
        except socket.error, msg:
            print '%s' % (msg,)

if __name__ == '__main__':
    reuse_socket_e addr()
```

此菜谱的输出将类似于以下命令：

```py
$ python 1_10_reuse_socket_address.py 
Old sock state: 0
New sock state: 1
Listening on port: 8282 

```

## 它是如何工作的...

您可以从一个控制台窗口运行此脚本，并尝试从另一个控制台窗口通过键入`telnet localhost 8282`连接到该服务器。在关闭服务器程序后，您可以在同一端口上再次运行它。然而，如果您注释掉设置`SO_REUSEADDR`的行，服务器将无法再次运行。

# 从互联网时间服务器打印当前时间

许多程序依赖于准确的机器时间，例如 UNIX 中的`make`命令。您的机器时间可能不同，需要与网络中的另一个时间服务器同步。

## 准备工作

为了将您的机器时间与互联网上的一个时间服务器同步，您可以编写一个 Python 客户端。为此，将使用`ntplib`。在这里，客户端/服务器对话将使用**网络时间协议**（**NTP**）进行。如果您的机器上未安装`ntplib`，您可以使用以下命令通过`pip`或`easy_install`从`PyPI`获取它：

```py
$ pip install ntplib

```

## 如何做...

我们创建了一个`NTPClient`实例，然后通过传递 NTP 服务器地址来调用其上的`request()`方法。

列表 1.11 显示了如何如下从互联网时间服务器打印当前时间：

```py
 #!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 1
# This program is optimized for Python 2.7\. It may run on any
# other Python version with/without modifications

import ntplib
from time import ctime

def print_time():
    ntp_client = ntplib.NTPClient()
    response = ntp_client.request('pool.ntp.org')
    print ctime(response.tx_time)

if __name__ == '__main__':
    print_time()
```

在我的机器上，此菜谱显示了以下输出：

```py
$ python 1_11_print_machine_time.py 
Thu Mar 5 14:02:58 2012

```

## 它是如何工作的...

在这里，已经创建了一个 NTP 客户端，并向互联网 NTP 服务器之一`pool.ntp.org`发送了一个 NTP 请求。使用`ctime()`函数来打印响应。

# 编写 SNTP 客户端

与前面的菜谱不同，有时您不需要从 NTP 服务器获取精确的时间。您可以使用 NTP 的一个更简单的版本，称为简单网络时间协议。

## 如何做...

让我们创建一个不使用任何第三方库的纯 SNTP 客户端。

让我们先定义两个常量：`NTP_SERVER`和`TIME1970`。`NTP_SERVER`是我们客户端将要连接的服务器地址，而`TIME1970`是 1970 年 1 月 1 日的参考时间（也称为*纪元*）。您可以在[`www.epochconverter.com/`](http://www.epochconverter.com/)找到纪元时间的值或将其转换为纪元时间。实际的客户端创建一个 UDP 套接字（`SOCK_DGRAM`）来遵循 UDP 协议连接到服务器。然后，客户端需要发送 SNTP 协议数据（`'\x1b' + 47 * '\0'`）在一个数据包中。我们的 UDP 客户端使用`sendto()`和`recvfrom()`方法发送和接收数据。

当服务器以打包数组的形式返回时间信息时，客户端需要一个专门的`struct`模块来解包数据。有趣的数据位于数组的第 11 个元素。最后，我们需要从解包值中减去参考值`TIME1970`以获取实际当前时间。

列表 1.11 展示了如何编写一个 SNTP 客户端，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 1
# This program is optimized for Python 2.7\. It may run on any
# other Python version with/without modifications
import socket
import struct
import sys
import time

NTP_SERVER = "0.uk.pool.ntp.org"
TIME1970 = 2208988800L

def sntp_client():
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = '\x1b' + 47 * '\0'
    client.sendto(data, (NTP_SERVER, 123))
    data, address = client.recvfrom( 1024 )
    if data:
        print 'Response received from:', address
    t = struct.unpack( '!12I', data )[10]
    t -= TIME1970
    print '\tTime=%s' % time.ctime(t)

if __name__ == '__main__':
    sntp_client()
```

此配方打印从互联网时间服务器通过 SNTP 协议接收的当前时间，如下所示：

```py
$ python 1_12_sntp_client.py 
Response received from: ('87.117.251.2', 123) 
 Time=Tue Feb 25 14:49:38 2014 

```

## 它是如何工作的...

此 SNTP 客户端创建一个套接字连接并发送协议数据。在接收到 NTP 服务器（在这种情况下，`0.uk.pool.ntp.org`）的响应后，它使用`struct`解包数据。最后，它减去参考时间，即 1970 年 1 月 1 日，并使用 Python 时间模块的内置方法`ctime()`打印时间。

# 编写一个简单的回显客户端/服务器应用程序

在使用 Python 的基本套接字 API 进行测试后，我们现在创建一个套接字服务器和客户端。在这里，你将有机会利用你在前面的配方中获得的基本知识。

## 如何操作...

在这个例子中，服务器将回显从客户端接收到的任何内容。我们将使用 Python 的`argparse`模块从命令行指定 TCP 端口。服务器和客户端脚本都将接受此参数。

首先，我们创建服务器。我们首先创建一个 TCP 套接字对象。然后，我们设置地址重用，以便我们可以根据需要多次运行服务器。我们将套接字绑定到我们本地机器上的指定端口。在监听阶段，我们确保使用`listen()`方法的 backlog 参数来监听队列中的多个客户端。最后，我们等待客户端连接并发送一些数据到服务器。当数据被接收时，服务器将数据回显给客户端。

列表 1.13a 展示了如何编写一个简单的回显客户端/服务器应用程序，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter – 1
# This program is optimized for Python 2.7\. It may run on any
# other Python version with/without modifications.

import socket
import sys
import argparse

host = 'localhost'
data_payload = 2048
backlog = 5 

def echo_server(port):
    """ A simple echo server """
    # Create a TCP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Enable reuse address/port 
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Bind the socket to the port
    server_address = (host, port)
    print "Starting up echo server  on %s port %s" % server_address
    sock.bind(server_address)
    # Listen to clients, backlog argument specifies the max no. of queued connections
    sock.listen(backlog) 
    while True: 
        print "Waiting to receive message from client"
        client, address = sock.accept() 
        data = client.recv(data_payload) 
        if data:
            print "Data: %s" %data
            client.send(data)
            print "sent %s bytes back to %s" % (data, address)
        # end connection
        client.close() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Socket Server Example')
    parser.add_argument('--port', action="store", dest="port", type=int, required=True)
    given_args = parser.parse_args() 
    port = given_args.port
    echo_server(port)
```

在客户端代码中，我们使用端口号创建一个客户端套接字并连接到服务器。然后，客户端向服务器发送消息`Test message. This will be echoed`，并且客户端立即以几个段的形式接收到消息。在这里，构建了两个 try-except 块来捕获此交互会话中的任何异常。

列表 1-13b 展示了回显客户端，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter – 1
# This program is optimized for Python 2.7\. It may run on any
# other Python version with/without modifications.

import socket
import sys

import argparse

host = 'localhost'

def echo_client(port):
    """ A simple echo client """
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect the socket to the server
    server_address = (host, port)
    print "Connecting to %s port %s" % server_address
    sock.connect(server_address)

    # Send data
    try:
        # Send data
        message = "Test message. This will be echoed"
        print "Sending %s" % message
        sock.sendall(message)
        # Look for the response
        amount_received = 0
        amount_expected = len(message)
        while amount_received < amount_expected:
            data = sock.recv(16)
            amount_received += len(data)
            print "Received: %s" % data
    except socket.errno, e:
        print "Socket error: %s" %str(e)
    except Exception, e:
        print "Other exception: %s" %str(e)
    finally:
        print "Closing connection to the server"
        sock.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Socket Server Example')
    parser.add_argument('--port', action="store", dest="port", type=int, required=True)
    given_args = parser.parse_args() 
    port = given_args.port
    echo_client(port)
```

## 它是如何工作的...

为了查看客户端/服务器交互，在一个控制台中启动以下服务器脚本：

```py
$ python 1_13a_echo_server.py --port=9900 
Starting up echo server  on localhost port 9900 

Waiting to receive message from client 

```

现在，从另一个终端运行客户端，如下所示：

```py
$ python 1_13b_echo_client.py --port=9900 
Connecting to localhost port 9900 
Sending Test message. This will be echoed 
Received: Test message. Th 
Received: is will be echoe 
Received: d 
Closing connection to the server

```

当客户端连接到 localhost 时，服务器也会打印以下消息：

```py
Data: Test message. This will be echoed 
sent Test message. This will be echoed bytes back to ('127.0.0.1', 42961) 
Waiting to receive message from client

```
