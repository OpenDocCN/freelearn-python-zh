# 第七章：使用套接字进行编程

在 Python 中与各种客户端/服务器进行交互后，您可能会渴望为自己选择的任何协议创建自定义客户端和服务器。Python 在低级网络接口上提供了很好的覆盖。一切都始于 BSD 套接字接口。正如您可以想象的那样，Python 有一个`socket`模块，为您提供了与套接字接口一起工作所需的功能。如果您以前在 C/C++等其他语言中进行过套接字编程，您会喜欢 Python 的`socket`模块。

在本章中，我们将通过创建各种 Python 脚本来探索套接字模块。

本章的亮点如下：

+   套接字基础

+   使用 TCP 套接字

+   使用 UDP 套接字

+   TCP 端口转发

+   非阻塞套接字 I/O

+   使用 SSL/TLS 保护套接字

+   创建自定义 SSL 客户端/服务器

# 套接字基础

任何编程语言中的网络编程都可以从套接字开始。但是什么是套接字？简而言之，网络套接字是实体可以进行进程间通信的虚拟端点。例如，一台计算机中的一个进程与另一台计算机上的一个进程交换数据。我们通常将发起通信的第一个进程标记为客户端，后一个进程标记为服务器。

Python 有一种非常简单的方式来开始使用套接字接口。为了更好地理解这一点，让我们先了解一下整体情况。在下图中，显示了客户端/服务器交互的流程。这将让您了解如何使用套接字 API。

![套接字基础](img/B03711_07_01.jpg)

通过套接字进行客户端/服务器交互

在典型客户端和服务器之间的交互中，服务器进程必须做更多的工作，正如您可能已经想到的那样。创建套接字对象后，服务器进程将该套接字绑定到特定的 IP 地址和端口。这很像使用分机号的电话连接。在公司办公室中，新员工分配了他的办公电话后，通常会被分配一个新的分机号。因此，如果有人给这位员工打电话，可以使用他的电话号码和分机号建立连接。成功绑定后，服务器进程将开始监听新的客户端连接。对于有效的客户端会话，服务器进程可以接受客户端进程的请求。此时，我们可以说服务器和客户端之间的连接已经建立。

然后客户端/服务器进入请求/响应循环。客户端进程向服务器进程发送数据，服务器进程处理数据并返回响应给客户端。当客户端进程完成时，通过关闭连接退出。此时，服务器进程可能会回到监听状态。

上述客户端和服务器之间的交互是实际情况的一个非常简化的表示。实际上，任何生产服务器进程都有多个线程或子进程来处理来自成千上万客户端的并发连接，这些连接是通过各自的虚拟通道进行的。

# 使用 TCP 套接字

在 Python 中创建套接字对象非常简单。您只需要导入`socket`模块并调用`socket()`类：

```py
from socket import*
import socket

#create a TCP socket (SOCK_STREAM)
s = socket.socket(family=AF_INET, type=SOCK_STREAM, proto=0)
print('Socket created')
```

传统上，该类需要大量参数。以下是其中一些：

+   **套接字族**：这是套接字的域，例如`AF_INET`（大约 90％的互联网套接字属于此类别）或`AF_UNIX`，有时也会使用。在 Python 3 中，您可以使用`AF_BLUETOOTH`创建蓝牙套接字。

+   **套接字类型**：根据您的需求，您需要指定套接字的类型。例如，通过分别指定`SOCK_STREAM`和`SOCK_DGRAM`来创建基于 TCP 和 UDP 的套接字。

+   **协议**：这指定了套接字族和类型内协议的变化。通常，它被留空为零。

由于许多原因，套接字操作可能不成功。例如，如果作为普通用户没有权限访问特定端口，可能无法绑定套接字。这就是为什么在创建套接字或进行一些网络绑定通信时进行适当的错误处理是个好主意。

让我们尝试将客户端套接字连接到服务器进程。以下代码是一个连接到服务器套接字的 TCP 客户端套接字的示例：

```py
import socket
import sys 

if __name__ == '__main__':

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error as err:
        print("Failed to crate a socket")
        print("Reason: %s" %str(err))
        sys.exit();

    print('Socket created')

    target_host = input("Enter the target host name to connect: ")
    target_port = input("Enter the target port: ") 

    try:
        sock.connect((target_host, int(target_port)))
        print("Socket Connected to %s on port: %s" %(target_host, target_port))
    sock.shutdown(2)
    except socket.error as err:
        print("Failed to connect to %s on port %s" %(target_host, target_port))
        print("Reason: %s" %str(err))
        sys.exit();
```

如果您运行上述的 TCP 客户端，将显示类似以下的输出：

```py
**# python 7_1_tcp_client_socket.py**
**Socket created**
**Enter the target host name to connect: 'www.python.org'**
**Enter the target port: 80**
**Socket Connected to www.python.org on port: 80**

```

然而，如果由于某种原因套接字创建失败，比如无效的 DNS，将显示类似以下的输出：

```py
**# python 7_1_tcp_client_socket.py**
**Socket created**
**Enter the target host name to connect: www.asgdfdfdkflakslalalasdsdsds.invalid**
**Enter the target port: 80**
**Failed to connect to www.asgdfdfdkflakslalalasdsdsds.invalid on port 80**
**Reason: [Errno -2] Name or service not known**

```

现在，让我们与服务器交换一些数据。以下代码是一个简单 TCP 客户端的示例：

```py
import socket

HOST = 'www.linux.org' # or 'localhost'
PORT = 80
BUFSIZ = 4096
ADDR = (HOST, PORT)

if __name__ == '__main__':
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(ADDR)

    while True:
        data = 'GET / HTTP/1.0\r\n\r\n'
        if not data:
            break
        client_sock.send(data.encode('utf-8'))
        data = client_sock.recv(BUFSIZ)
        if not data:
            break
        print(data.decode('utf-8'))

    client_sock.close()
```

如果您仔细观察，您会发现上述的代码实际上创建了一个从 Web 服务器获取网页的原始 HTTP 客户端。它发送一个 HTTP 的`GET`请求来获取主页：

```py
**# python 7_2_simple_tcp_client.py**
**HTTP/1.1 200 OK**
**Date: Sat, 07 Mar 2015 16:23:02 GMT**
**Server: Apache**
**Last-Modified: Mon, 17 Feb 2014 03:19:34 GMT**
**Accept-Ranges: bytes**
**Content-Length: 111**
**Connection: close**
**Content-Type: text/html**

**<html><head><META HTTP-EQUIV="refresh" CONTENT="0;URL=/cgi- sys/defaultwebpage.cgi"></head><body></body></html>**

```

## 检查客户端/服务器通信

通过交换网络数据包进行的客户端和服务器之间的交互可以使用任何网络数据包捕获工具进行分析，比如 Wireshark。您可以配置 Wireshark 通过端口或主机过滤数据包。在这种情况下，我们可以通过端口 80 进行过滤。您可以在**捕获** | **选项**菜单下找到选项，并在**捕获过滤器**选项旁边的输入框中输入`port 80`，如下面的屏幕截图所示：

![检查客户端/服务器通信](img/B03711_07_02.jpg)

在**接口**选项中，我们选择捕获通过任何接口传递的数据包。现在，如果您运行上述的 TCP 客户端连接到[www.linux.org](http://www.linux.org/)，您可以在 Wireshark 中看到交换的数据包序列，如下面的屏幕截图所示：

![检查客户端/服务器通信](img/B03711_07_03.jpg)

正如您所见，前三个数据包通过客户端和服务器之间的三次握手过程建立了 TCP 连接。我们更感兴趣的是第四个数据包，它向服务器发出了 HTTP 的`GET`请求。如果您双击所选行，您可以看到 HTTP 请求的详细信息，如下面的屏幕截图所示：

![检查客户端/服务器通信](img/B03711_07_04.jpg)

如您所见，HTTP 的`GET`请求还有其他组件，比如`请求 URI`，版本等。现在您可以检查来自 Web 服务器的 HTTP 响应到您的客户端。它是在 TCP 确认数据包之后，也就是第六个数据包之后。在这里，服务器通常发送一个 HTTP 响应代码（在本例中是`200`），内容长度和数据或网页内容。这个数据包的结构如下面的屏幕截图所示：

![检查客户端/服务器通信](img/B03711_07_05.jpg)

通过对客户端和服务器之间的交互进行上述分析，您现在可以在基本层面上理解当您使用 Web 浏览器访问网页时发生了什么。在下一节中，您将看到如何创建自己的 TCP 服务器，并检查个人 TCP 客户端和服务器之间的交互。

## TCP 服务器

正如您从最初的客户端/服务器交互图中所理解的，服务器进程需要进行一些额外的工作。它需要绑定到套接字地址并监听传入的连接。以下代码片段显示了如何创建一个 TCP 服务器：

```py
import socket
from time import ctime

HOST = 'localhost'
PORT = 12345
BUFSIZ = 1024
ADDR = (HOST, PORT)

if __name__ == '__main__':
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(ADDR)
    server_socket.listen(5)
    server_socket.setsockopt( socket.SOL_SOCKET, socket.SO_REUSEADDR, 1 )

    while True:
        print('Server waiting for connection...')
        client_sock, addr = server_socket.accept()
        print('Client connected from: ', addr)

        while True:
            data = client_sock.recv(BUFSIZ)
            if not data or data.decode('utf-8') == 'END':
                break
            print("Received from client: %s" % data.decode('utf- 8'))
            print("Sending the server time to client: %s"  %ctime())
            try:
                client_sock.send(bytes(ctime(), 'utf-8'))
            except KeyboardInterrupt:
                print("Exited by user")
        client_sock.close()
    server_socket.close()
```

让我们修改之前的 TCP 客户端，向任何服务器发送任意数据。以下是一个增强型 TCP 客户端的示例：

```py
import socket

HOST = 'localhost'
PORT = 12345
BUFSIZ = 256

if __name__ == '__main__':
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = input("Enter hostname [%s]: " %HOST) or HOST
    port = input("Enter port [%s]: " %PORT) or PORT

    sock_addr = (host, int(port))
    client_sock.connect(sock_addr)

    payload = 'GET TIME'
    try:
        while True:
            client_sock.send(payload.encode('utf-8'))
            data = client_sock.recv(BUFSIZ)
            print(repr(data))
            more = input("Want to send more data to server[y/n] :")
            if more.lower() == 'y':
               payload = input("Enter payload: ")
            else:
                break
    except KeyboardInterrupt:
        print("Exited by user") 

    client_sock.close()
```

如果您在一个控制台中运行上述的 TCP 服务器，另一个控制台中运行 TCP 客户端，您可以看到客户端和服务器之间的以下交互。运行 TCP 服务器脚本后，您将得到以下输出：

```py
**# python 7_3_tcp_server.py** 
**Server waiting for connection...**
**Client connected from:  ('127.0.0.1', 59961)**
**Received from client: GET TIME**

**Sending the server time to client: Sun Mar 15 12:09:16 2015**
**Server waiting for connection...**

```

当您在另一个终端上运行 TCP 客户端脚本时，您将得到以下输出：

```py
**# python 7_4_tcp_client_socket_send_data.py** 
**Enter hostname [www.linux.org]: localhost**
**Enter port [80]: 12345**
**b'Sun Mar 15 12:09:16 2015'**
**Want to send more data to server[y/n] :n**

```

## 检查客户端/服务器交互

现在，您可以再次配置 Wireshark 来捕获数据包，就像上一节讨论的那样。但是，在这种情况下，您需要指定服务器正在侦听的端口（在上面的示例中是`12345`），如下面的屏幕截图所示：

![检查客户端/服务器交互](img/B03711_07_06.jpg)

由于我们在非标准端口上捕获数据包，Wireshark 不会在**数据**部分解码它（如上面屏幕截图的中间窗格所示）。但是，您可以在底部窗格上看到解码后的文本，服务器的时间戳显示在右侧。

# 使用 UDP 套接字

与 TCP 不同，UDP 不会检查交换的数据报中的错误。我们可以创建类似于 TCP 客户端/服务器的 UDP 客户端/服务器。唯一的区别是在创建套接字对象时，您必须指定`SOCK_DGRAM`而不是`SOCK_STREAM`。

让我们创建一个 UDP 服务器。使用以下代码创建 UDP 服务器：

```py
from socket import socket, AF_INET, SOCK_DGRAM
maxsize = 4096

sock = socket(AF_INET,SOCK_DGRAM)
sock.bind(('',12345))
while True:    
  data, addr = sock.recvfrom(maxsize)
    resp = "UDP server sending data"    
  sock.sendto(resp,addr)
```

现在，您可以创建一个 UDP 客户端，向 UDP 服务器发送一些数据，如下面的代码所示：

```py
from socket import socket, AF_INET, SOCK_DGRAM

MAX_SIZE = 4096
PORT = 12345

if __name__ == '__main__':
    sock = socket(AF_INET,SOCK_DGRAM)
    msg = "Hello UDP server"
    sock.sendto(msg.encode(),('', PORT))
    data, addr = sock.recvfrom(MAX_SIZE)
    print("Server says:")
    print(repr(data))
```

在上面的代码片段中，UDP 客户端发送一行文本`Hello UDP server`并从服务器接收响应。下面的屏幕截图显示了客户端发送到服务器的请求：

![使用 UDP 套接字](img/B03711_07_07.jpg)

下面的屏幕截图显示了服务器发送给客户端的响应。在检查 UDP 客户端/服务器数据包之后，我们可以很容易地看到 UDP 比 TCP 简单得多。它通常被称为无连接协议，因为没有涉及确认或错误检查。

![使用 UDP 套接字](img/B03711_07_08.jpg)

# TCP 端口转发

我们可以使用 TCP 套接字编程进行一些有趣的实验，比如设置 TCP 端口转发。这有很好的用例。例如，如果您在没有 SSL 能力进行安全通信的公共服务器上运行不安全的程序（FTP 密码可以在传输过程中以明文形式看到）。由于这台服务器可以从互联网访问，您必须确保密码是加密的，才能登录到服务器。其中一种方法是使用安全 FTP 或 SFTP。我们可以使用简单的 SSH 隧道来展示这种方法的工作原理。因此，您本地 FTP 客户端和远程 FTP 服务器之间的任何通信都将通过这个加密通道进行。

让我们运行 FTP 程序到同一个 SSH 服务器主机。但是从本地机器创建一个 SSH 隧道，这将给您一个本地端口号，并将直接连接您到远程 FTP 服务器守护程序。

Python 有一个第三方的`sshtunnel`模块，它是 Paramiko 的`SSH`库的包装器。以下是 TCP 端口转发的代码片段，显示了如何实现这个概念：

```py
import sshtunnel
from getpass import getpass

ssh_host = '192.168.56.101'
ssh_port = 22
ssh_user = 'YOUR_SSH_USERNAME'

REMOTE_HOST = '192.168.56.101'
REMOTE_PORT = 21

from sshtunnel import SSHTunnelForwarder
ssh_password = getpass('Enter YOUR_SSH_PASSWORD: ')

server = SSHTunnelForwarder(
    ssh_address=(ssh_host, ssh_port),
    ssh_username=ssh_user,
    ssh_password=ssh_password,
    remote_bind_address=(REMOTE_HOST, REMOTE_PORT))

server.start()
print('Connect the remote service via local port: %s'  %server.local_bind_port)
# work with FTP SERVICE via the `server.local_bind_port.
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Exiting user user request.\n")
    server.stop()
```

让我们捕获从本地机器`192.168.0.102`到远程机器`192.168.0.101`的数据包传输。您将看到所有网络流量都是加密的。当您运行上述脚本时，您将获得一个本地端口号。使用`ftp`命令连接到该本地端口号：

```py
**$ ftp <localhost> <local_bind_port>**

```

如果您运行上述命令，那么您将得到以下屏幕截图：

![TCP 端口转发](img/B03711_07_09.jpg)

在上面的屏幕截图中，您看不到任何 FTP 流量。正如您所看到的，首先我们连接到本地端口`5815`（参见前三个数据包），然后突然之间与远程主机建立了加密会话。您可以继续观察远程流量，但是没有 FTP 的痕迹。

如果您还可以在远程机器（`192.168.56.101`）上捕获数据包，您可以看到 FTP 流量，如下面的屏幕截图所示：

![TCP 端口转发](img/B03711_07_12.jpg)

有趣的是，您可以看到您的 FTP 密码从本地机器（通过 SSH 隧道）以明文形式发送到远程计算机，而不是通过网络发送，如下图所示：

TCP 端口转发

因此，您可以将任何敏感的网络流量隐藏在 SSL 隧道中。不仅 FTP，您还可以通过 SSH 通道加密传输远程桌面会话。

# 非阻塞套接字 I/O

在本节中，我们将看到一个小的示例代码片段，用于测试非阻塞套接字 I/O。如果您知道同步阻塞连接对您的程序不是必需的，这将非常有用。以下是非阻塞 I/O 的示例：

```py
import socket

if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(0)
    sock.settimeout(0.5)
    sock.bind(("127.0.0.1", 0))

    socket_address =sock.getsockname()
    print("Asynchronous socket server launched on socket: %s" %str(socket_address))
    while(1):
        sock.listen(1)
```

此脚本将以非阻塞方式运行套接字服务器并进行监听。这意味着您可以连接更多的客户端，他们不一定会因 I/O 而被阻塞。

# 使用 TLS/SSL 保护套接字

您可能已经遇到了使用**安全套接字层**（**SSL**）或更精确地说是**传输层安全**（**TLS**）进行安全网络通信的讨论，这已被许多其他高级协议采用。让我们看看如何使用 SSL 包装普通套接字连接。Python 具有内置的`ssl`模块，可以实现此目的。

在此示例中，我们希望创建一个普通的 TCP 套接字并连接到启用了 HTTPS 的 Web 服务器。然后，我们可以使用 SSL 包装该连接并检查连接的各种属性。例如，要检查远程 Web 服务器的身份，我们可以查看 SSL 证书中的主机名是否与我们期望的相同。以下是一个基于安全套接字的客户端的示例：

```py
import socket
import ssl
from ssl import wrap_socket, CERT_NONE, PROTOCOL_TLSv1, SSLError
from ssl import SSLContext
from ssl import HAS_SNI

from pprint import pprint

TARGET_HOST = 'www.google.com'
SSL_PORT = 443
# Use the path of CA certificate file in your system
CA_CERT_PATH = '/usr/local/lib/python3.3/dist- packages/requests/cacert.pem'

def ssl_wrap_socket(sock, keyfile=None, certfile=None, cert_reqs=None, ca_certs=None, server_hostname=None, ssl_version=None):

    context = SSLContext(ssl_version)
    context.verify_mode = cert_reqs

    if ca_certs:
        try:
            context.load_verify_locations(ca_certs)
        except Exception as e:
            raise SSLError(e)

    if certfile:
        context.load_cert_chain(certfile, keyfile)

    if HAS_SNI:  # OpenSSL enabled SNI
        return context.wrap_socket(sock, server_hostname=server_hostname)

    return context.wrap_socket(sock)

if __name__ == '__main__':
    hostname = input("Enter target host:") or TARGET_HOST
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect((hostname, 443))

    ssl_socket = ssl_wrap_socket(client_sock, ssl_version=PROTOCOL_TLSv1, cert_reqs=ssl.CERT_REQUIRED, ca_certs=CA_CERT_PATH, server_hostname=hostname)

    print("Extracting remote host certificate details:")
    cert = ssl_socket.getpeercert()
    pprint(cert)
    if not cert or ('commonName', TARGET_HOST) not in cert['subject'][4]:
        raise Exception("Invalid SSL cert for host %s. Check if this is a man-in-the-middle attack!" )
    ssl_socket.write('GET / \n'.encode('utf-8'))
    #pprint(ssl_socket .recv(1024).split(b"\r\n"))
    ssl_socket.close()
    client_sock.close()
```

如果运行上述示例，您将看到远程 Web 服务器（例如[`www.google.com`](http://www.google.com)）的 SSL 证书的详细信息。在这里，我们创建了一个 TCP 套接字并将其连接到 HTTPS 端口`443`。然后，该套接字连接使用我们的`ssl_wrap_socket()`函数包装成 SSL 数据包。此函数将以下参数作为参数：

+   `sock`：TCP 套接字

+   `keyfile`：SSL 私钥文件路径

+   `certfile`：SSL 公共证书路径

+   `cert_reqs`：确认是否需要来自另一方的证书以建立连接，以及是否需要验证测试

+   `ca_certs`：公共证书颁发机构证书路径

+   `server_hostname`：目标远程服务器的主机名

+   `ssl_version`：客户端要使用的预期 SSL 版本

在 SSL 套接字包装过程开始时，我们使用`SSLContext()`类创建了一个 SSL 上下文。这是必要的，以设置 SSL 连接的特定属性。除了使用自定义上下文外，我们还可以使用`ssl`模块默认提供的默认上下文，使用`create_default_context()`函数。您可以使用常量指定是否要创建客户端或服务器端套接字。以下是创建客户端套接字的示例：

```py
context = ssl.create_default_context(Purpose.SERVER_AUTH)
```

`SSLContext`对象接受 SSL 版本参数，在我们的示例中设置为`PROTOCOL_TLSv1`，或者您应该使用最新版本。请注意，SSLv2 和 SSLv3 已经被破解，严重的安全问题不能在任何生产代码中使用。

在上面的示例中，`CERT_REQUIRED`表示连接需要服务器证书，并且稍后将验证此证书。

如果已提供 CA 证书参数并提供了证书路径，则使用`load_verify_locations()`方法加载 CA 证书文件。这将用于验证对等服务器证书。如果您想在系统上使用默认证书路径，您可能会调用另一个上下文方法；`load_default_certs(purpose=Purpose.SERVER_AUTH)`。

当我们在服务器端操作时，通常使用`load_cert_chain()`方法加载密钥和证书文件，以便客户端可以验证服务器的真实性。

最后，调用`wrap_socket()`方法返回一个 SSL 包装套接字。请注意，如果`OpenSSL`库启用了**服务器名称指示**（**SNI**）支持，您可以在包装套接字时传递远程服务器的主机名。当远程服务器使用不同的 SSL 证书为单个 IP 地址使用不同的安全服务，例如基于名称的虚拟主机时，这将非常有用。

如果运行上述 SSL 客户端代码，您将看到远程服务器的 SSL 证书的各种属性，如下图所示。这用于通过调用`getpeercert()`方法验证远程服务器的真实性，并将其与返回的主机名进行比较。

![使用 TLS/SSL 保护套接字](img/B03711_07_13.jpg)

有趣的是，如果任何其他虚假的 Web 服务器想要假冒 Google 的 Web 服务器，除非您检查由认可的证书颁发机构签署的 SSL 证书，否则它根本无法做到这一点，除非认可的 CA 已被破坏/颠覆。对您的 Web 浏览器进行的这种形式的攻击通常被称为**中间人**（**MITM**）攻击。

## 检查标准 SSL 客户端/服务器通信

以下屏幕截图显示了 SSL 客户端与远程服务器之间的交互：

![检查标准 SSL 客户端/服务器通信](img/B03711_07_14.jpg)

让我们来看看客户端和服务器之间的 SSL 握手过程。在 SSL 握手的第一步中，客户端向远程服务器发送一个`Hello`消息，说明它在处理密钥文件、加密消息、进行消息完整性检查等方面的能力。在下面的屏幕截图中，您可以看到客户端向服务器呈现了一组`38`个密码套件，以选择相关的算法。它还发送了 TLS 版本号`1.0`和一个随机数，用于生成用于加密后续消息交换的主密钥。这有助于防止任何第三方查看数据包内容。在`Hello`消息中看到的随机数用于生成预主密钥，双方将进一步处理以得到主密钥，然后使用该密钥生成对称密钥。

![检查标准 SSL 客户端/服务器通信](img/B03711_07_15.jpg)

在服务器发送到客户端的第二个数据包中，服务器选择了密码套件`TLS_ECDHE_RSA_WITH_RC4_128_SHA`以连接到客户端。这大致意味着服务器希望使用 RSA 算法处理密钥，使用 RC4 进行加密，并使用 SHA 进行完整性检查（哈希）。这在以下屏幕截图中显示：

![检查标准 SSL 客户端/服务器通信](img/B03711_07_16.jpg)

在 SSL 握手的第二阶段，服务器向客户端发送 SSL 证书。如前所述，此证书由 CA 颁发。它包含序列号、公钥、有效期和主题和颁发者的详细信息。以下屏幕截图显示了远程服务器的证书。您能在数据包中找到服务器的公钥吗？

![检查标准 SSL 客户端/服务器通信](img/B03711_07_17.jpg)

在握手的第三阶段，客户端交换密钥并计算主密钥以加密消息并继续进一步通信。客户端还发送更改在上一阶段达成的密码规范的请求。然后指示开始加密消息。以下屏幕截图显示了这个过程：

![检查标准 SSL 客户端/服务器通信](img/B03711_07_18.jpg)

在 SSL 握手过程的最后一个任务中，服务器为客户端的特定会话生成了一个新的会话票证。这是由于 TLS 扩展引起的，客户端通过在客户端`Hello`消息中发送一个空的会话票证扩展来宣传其支持。服务器在其服务器`Hello`消息中回答一个空的会话票证扩展。这个会话票证机制使客户端能够记住整个会话状态，服务器在维护服务器端会话缓存方面变得不那么忙碌。以下截图显示了一个呈现 SSL 会话票证的示例：

![检查标准 SSL 客户端/服务器通信](img/B03711_07_19.jpg)

# 创建自定义 SSL 客户端/服务器

到目前为止，我们更多地处理 SSL 或 TLS 客户端。现在，让我们简要地看一下服务器端。由于您已经熟悉 TCP/UDP 套接字服务器创建过程，让我们跳过那部分，只集中在 SSL 包装部分。以下代码片段显示了一个简单 SSL 服务器的示例：

```py
import socket
import ssl

SSL_SERVER_PORT = 8000

if __name__ == '__main__':
    server_socket = socket.socket()
    server_socket.bind(('', SSL_SERVER_PORT))
    server_socket.listen(5)
    print("Waiting for ssl client on port %s" %SSL_SERVER_PORT)
    newsocket, fromaddr = server_socket.accept()
    # Generate your server's  public certificate and private key pairs.
    ssl_conn = ssl.wrap_socket(newsocket, server_side=True, certfile="server.crt", keyfile="server.key", ssl_version=ssl.PROTOCOL_TLSv1)
    print(ssl_conn.read())
    ssl_conn.write('200 OK\r\n\r\n'.encode())
    print("Served ssl client. Exiting...")
    ssl_conn.close()
    server_socket.close()
```

正如您所看到的，服务器套接字被`wrap_socket()`方法包装，该方法使用一些直观的参数，如`certfile`、`keyfile`和`SSL`版本号。您可以通过按照互联网上找到的任何逐步指南轻松生成证书。例如，[`www.akadia.com/services/ssh_test_certificate.html`](http://www.akadia.com/services/ssh_test_certificate.html)建议通过几个步骤生成 SSL 证书。

现在，让我们制作一个简化版本的 SSL 客户端，与上述 SSL 服务器进行通信。以下代码片段显示了一个简单 SSL 客户端的示例：

```py
from socket import socket
import ssl

from pprint import pprint

TARGET_HOST ='localhost'
TARGET_PORT = 8000
CA_CERT_PATH = 'server.crt'

if __name__ == '__main__':

    sock = socket()
    ssl_conn = ssl.wrap_socket(sock, cert_reqs=ssl.CERT_REQUIRED, ssl_version=ssl.PROTOCOL_TLSv1, ca_certs=CA_CERT_PATH)
    target_host = TARGET_HOST 
    target_port = TARGET_PORT 
    ssl_conn.connect((target_host, int(target_port)))
    # get remote cert
    cert = ssl_conn.getpeercert()
    print("Checking server certificate")
    pprint(cert)
    if not cert or ssl.match_hostname(cert, target_host):
        raise Exception("Invalid SSL cert for host %s. Check if this is a man-in-the-middle attack!" %target_host )
    print("Server certificate OK.\n Sending some custom request... GET ")
    ssl_conn.write('GET / \n'.encode('utf-8'))
    print("Response received from server:")
    print(ssl_conn.read())
    ssl_conn.close()
```

运行客户端/服务器将显示类似于以下截图的输出。您能否看到与我们上一个示例客户端/服务器通信相比有什么不同？

![创建自定义 SSL 客户端/服务器](img/B03711_07_20.jpg)

## 检查自定义 SSL 客户端/服务器之间的交互

让我们再次检查 SSL 客户端/服务器的交互，以观察其中的差异。第一个截图显示了整个通信序列。在以下截图中，我们可以看到服务器的`Hello`和证书合并在同一消息中。

![检查自定义 SSL 客户端/服务器之间的交互](img/B03711_07_21.jpg)

客户端的**客户端 Hello**数据包看起来与我们之前的 SSL 连接非常相似，如下截图所示：

![检查自定义 SSL 客户端/服务器之间的交互](img/B03711_07_22.jpg)

服务器的**服务器 Hello**数据包有点不同。您能识别出区别吗？密码规范不同，即`TLS_RSA_WITH_AES_256_CBC_SHA`，如下截图所示：

![检查自定义 SSL 客户端/服务器之间的交互](img/B03711_07_23.jpg)

**客户端密钥交换**数据包看起来也很熟悉，如下截图所示：

![检查自定义 SSL 客户端/服务器之间的交互](img/B03711_07_24.jpg)

以下截图显示了在此连接中提供的**新会话票证**数据包：

![检查自定义 SSL 客户端/服务器之间的交互](img/B03711_07_25.jpg)

现在让我们来看一下应用数据。那加密了吗？对于捕获的数据包，它看起来像垃圾。以下截图显示了隐藏真实数据的加密消息。这就是我们使用 SSL/TLS 想要实现的效果。

![检查自定义 SSL 客户端/服务器之间的交互](img/B03711_07_26.jpg)

# 总结

在本章中，我们讨论了使用 Python 的`socket`和`ssl`模块进行基本的 TCP/IP 套接字编程。我们演示了如何将简单的 TCP 套接字包装为 TLS，并用于传输加密数据。我们还发现了使用 SSL 证书验证远程服务器真实性的方法。还介绍了套接字编程中的一些其他小问题，比如非阻塞套接字 I/O。每个部分中的详细数据包分析帮助我们了解套接字编程练习中发生了什么。

在下一章中，我们将学习关于套接字服务器设计，特别是流行的多线程和事件驱动方法。
