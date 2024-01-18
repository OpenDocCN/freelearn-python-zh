# 基本网络 - 套接字编程

在本章中，您将学习套接字和三种互联网协议：`http`，`ftplib`和`urllib`。您还将学习Python中用于网络的`socket`模块。`http`是一个用于处理**超文本传输协议**（**HTTP**）的包。`ftplib`模块用于执行自动化的与FTP相关的工作。`urllib`是一个处理与URL相关的工作的包。

在本章中，您将学习以下内容：

+   套接字

+   `http`包

+   `ftplib`模块

+   `urllib`包

# 套接字

在本节中，我们将学习套接字。我们将使用Python的socket模块。套接字是用于机器之间通信的端点，无论是在本地还是通过互联网。套接字模块有一个套接字类，用于处理数据通道。它还具有用于网络相关任务的函数。要使用套接字模块的功能，我们首先需要导入套接字模块。

让我们看看如何创建套接字。套接字类有一个套接字函数，带有两个参数：`address_family` 和 `socket类型`。

以下是语法：

```py
 import socket            s = socket.socket(address_family, socket type)
```

`address_family` 控制OSI网络层协议。

**`socket类型`** 控制传输层协议。

Python支持三种地址族：`AF_INET`，`AF_INET6`和`AF_UNIX`。最常用的是`AF_INET`，用于互联网寻址。`AF_INET6`用于IPv6互联网寻址。`AF_UNIX`用于**Unix域套接字**（**UDS**），这是一种进程间通信协议。

有两种套接字类型：`SOCK_DGRAM` 和 `SOCK_STREAM`。`SOCK_DGRAM` 套接字类型用于面向消息的数据报传输；这些与UDP相关联。数据报套接字传递单个消息。`SOCK_STREAM` 用于面向流的传输；这些与TCP相关联。流套接字在客户端和服务器之间提供字节流。

套接字可以配置为服务器套接字和客户端套接字。当TCP/IP套接字都连接时，通信将是双向的。现在我们将探讨一个客户端-服务器通信的示例。我们将创建两个脚本：`server.py`和`client.py`。

`server.py`脚本如下：

```py
import socket host_name = socket.gethostname() port = 5000 s_socket = socket.socket() s_socket.bind((host_name, port)) s_socket.listen(2) conn, address = s_socket.accept() print("Connection from: " + str(address)) while True:
 recv_data = conn.recv(1024).decode() if not recv_data: break print("from connected user: " + str(recv_data)) recv_data = input(' -> ') conn.send(recv_data.encode()) conn.close()
```

现在我们将为客户端编写一个脚本。

`client.py`脚本如下：

```py
import socket host_name = socket.gethostname() port = 5000 c_socket = socket.socket() c_socket.connect((host_name, port)) msg = input(" -> ")  while msg.lower().strip() != 'bye': c_socket.send(msg.encode()) recv_data = c_socket.recv(1024).decode() print('Received from server: ' + recv_data) msg = input(" -> ") c_socket.close()
```

现在我们将在两个不同的终端中运行这两个程序。在第一个终端中，我们将运行`server.py`，在第二个终端中，运行`client.py`。

输出将如下所示：

| **终端1：** `python3 server.py` | **终端2：** `python3 client.py` |
| `student@ubuntu:~/work$ python3 server.py``连接来自：（'127.0.0.1'，35120）``来自连接的用户：来自客户端的问候`` -> 来自服务器的问候！` | `student@ubuntu:~/work$ python3 client.py``-> 来自客户端的问候``从服务器接收：来自服务器的问候！`` ->` |

# http包

在本节中，我们将学习`http`包。`http`包有四个模块：

+   `http.client`：这是一个低级HTTP协议客户端

+   `http.server`：这包含基本的HTTP服务器类

+   `http.cookies`：这用于使用cookie实现状态管理

+   `http.cookiejar`：此模块提供cookie持久性

在本节中，我们将学习`http.client`和`http.server`模块。

# http.client模块

我们将看到两个`http`请求：`GET` 和 `POST`。我们还将建立一个`http`连接。

首先，我们将探讨一个创建`http`连接的示例。为此，创建一个`make_connection.py`脚本，并在其中编写以下内容：

```py
import http.client con_obj = http.client.HTTPConnection('Enter_URL_name', 80, timeout=20) print(con_obj)
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work$ python3 make_connection.py <http.client.HTTPConnection object at 0x7f2c365dd898>
```

在上面的示例中，我们使用了指定超时的端口80上的URL建立了连接。

现在我们将看到`http`的`GET`请求方法；使用这个`GET`请求方法，我们将看到一个示例，其中我们获得响应代码以及头列表。创建一个`get_example.py`脚本，并在其中编写以下内容：

```py
import http.client con_obj = http.client.HTTPSConnection("www.imdb.com") con_obj.request("GET", "/") response = con_obj.getresponse()  print("Status: {}".format(response.status))  headers_list = response.getheaders()
print("Headers: {}".format(headers_list))  con_obj.close()
```

按照以下方式运行脚本：

```py
student@ubuntu:~/work$ python3 get_example.py
```

输出应该如下：

```py
Status: 200 Headers: [('Server', 'Server'), ('Date', 'Fri, 23 Nov 2018 09:49:12 GMT'), ('Content-Type', 'text/html;charset=UTF-8'), ('Transfer-Encoding', 'chunked'), ('Connection', 'keep-alive'), ('X-Frame-Options', 'SAMEORIGIN'), ('Content-Security-Policy', "frame-ancestors 'self' imdb.com *.imdb.com *.media-imdb.com withoutabox.com *.withoutabox.com amazon.com *.amazon.com amazon.co.uk *.amazon.co.uk amazon.de *.amazon.de translate.google.com images.google.com www.google.com www.google.co.uk search.aol.com bing.com www.bing.com"), ('Ad-Unit', 'imdb.home.homepage'), ('Entity-Id', ''), ('Section-Id', 'homepage'), ('Page-Id', 'homepage'), ('Content-Language', 'en-US'), ('Set-Cookie', 'uu=BCYsgIz6VTPefAjQB9YlJiZhwogwHmoU3sLx9YK-A61kPgvXEKwHSJKU3XeaxIoL8DBQGhYLuFvR%0D%0AqPV6VVvx70AV6eL_sGzVaRQQAKf-PUz2y0sTx9H4Yvib9iSYRPOzR5qHQkwuoHPKmpu2KsSbPaCb%0D%0AYbc-R6nz9ObkbQf6RAYm5sTAdf5lSqM2ZzCEhfIt_H3tWQqnK5WlihYwfMZS2AJdtGXGRnRvEHlv%0D%0AyA4Dcn9NyeX44-hAnS64zkDfDeGXoCUic_kH6ZnD5vv21HOiVodVKA%0D%0A; Domain=.imdb.com; Expires=Wed, 11-Dec-2086 13:03:18 GMT; Path=/; Secure'), ('Set-Cookie', 'session-id=134-6809939-6044806; Domain=.imdb.com; Expires=Wed, 11-Dec-2086 13:03:18 GMT; Path=/; Secure'), ('Set-Cookie', 'session-id-time=2173686551; Domain=.imdb.com; Expires=Wed, 11-Dec-2086 13:03:18 GMT; Path=/; Secure'), ('Vary', 'Accept-Encoding,X-Amzn-CDN-Cache,User-Agent'), ('x-amz-rid', '7SWEYTYH4TX8YR2CF5JT')]
```

在前面的示例中，我们使用了`HTTPSConnection`，因为该网站是通过`HTTPS`协议提供的。您可以根据您使用的网站使用`HTTPSConnection`或`HTTPConnection`。我们提供了一个URL，并使用连接对象检查了状态。之后，我们得到了一个标题列表。这个标题列表包含了从服务器返回的数据类型的信息。`getheaders()`方法将获取标题列表。

现在我们将看到一个`POST`请求的示例。我们可以使用`HTTP POST`将数据发布到URL。为此，创建一个`post_example.py`脚本，并在其中写入以下内容：

```py
import http.client import json con_obj = http.client.HTTPSConnection('www.httpbin.org') headers_list = {'Content-type': 'application/json'} post_text = {'text': 'Hello World !!'} json_data = json.dumps(post_text) con_obj.request('POST', '/post', json_data, headers_list) response = con_obj.getresponse() print(response.read().decode())
```

按照以下方式运行脚本：

```py
student@ubuntu:~/work$ python3 post_example.py
```

您应该得到以下输出：

```py
{
 "args": {}, "data": "{\"text\": \"Hello World !!\"}", "files": {}, "form": {}, "headers": { "Accept-Encoding": "identity", "Connection": "close", "Content-Length": "26", "Content-Type": "application/json", "Host": "www.httpbin.org" }, "json": { "text": "Hello World !!" }, "origin": "1.186.106.115", "url": "https://www.httpbin.org/post" }
```

在前面的示例中，我们首先创建了一个`HTTPSConnection`对象。接下来，我们创建了一个`post_text`对象，它发布了`Hello World`。之后，我们写了一个`POST`请求，得到了一个响应。

# http.server模块

在本节中，我们将学习`http`包中的一个模块，即`http.server`模块。这个模块定义了用于实现`HTTP`服务器的类。它有两种方法：`GET`和`HEAD`。通过使用这个模块，我们可以在网络上共享文件。您可以在任何端口上运行`http`服务器。确保端口号大于`1024`。默认端口号是`8000`。

您可以按照以下方式使用`http.server`。

首先，导航到您想要的目录，然后运行以下命令：

```py
student@ubuntu:~/Desktop$ python3 -m http.server 9000
```

现在打开您的浏览器，在地址栏中输入`localhost:9000`，然后按*Enter*。您将得到以下输出：

```py
student@ubuntu:~/Desktop$ python3 -m http.server 9000 Serving HTTP on 0.0.0.0 port 9000 (http://0.0.0.0:9000/) ... 127.0.0.1 - - [23/Nov/2018 16:08:14] code 404, message File not found 127.0.0.1 - - [23/Nov/2018 16:08:14] "GET /Downloads/ HTTP/1.1" 404 - 127.0.0.1 - - [23/Nov/2018 16:08:14] code 404, message File not found 127.0.0.1 - - [23/Nov/2018 16:08:14] "GET /favicon.ico HTTP/1.1" 404 - 127.0.0.1 - - [23/Nov/2018 16:08:21] "GET / HTTP/1.1" 200 - 127.0.0.1 - - [23/Nov/2018 16:08:21] code 404, message File not found 127.0.0.1 - - [23/Nov/2018 16:08:21] "GET /favicon.ico HTTP/1.1" 404 - 127.0.0.1 - - [23/Nov/2018 16:08:26] "GET /hello/ HTTP/1.1" 200 - 127.0.0.1 - - [23/Nov/2018 16:08:26] code 404, message File not found 127.0.0.1 - - [23/Nov/2018 16:08:26] "GET /favicon.ico HTTP/1.1" 404 - 127.0.0.1 - - [23/Nov/2018 16:08:27] code 404, message File not found 127.0.0.1 - - [23/Nov/2018 16:08:27] "GET /favicon.ico HTTP/1.1" 404 -
```

# ftplib模块

`ftplib`是Python中的一个模块，它提供了执行FTP协议的各种操作所需的所有功能。`ftplib`包含FTP客户端类，以及一些辅助函数。使用这个模块，我们可以轻松地连接到FTP服务器，检索多个文件并处理它们。通过导入`ftplib`模块，我们可以使用它提供的所有功能。

在本节中，我们将介绍如何使用`ftplib`模块进行FTP传输。我们将看到各种FTP对象。

# 下载文件

在本节中，我们将学习如何使用`ftplib`模块从另一台机器下载文件。为此，创建一个`get_ftp_files.py`脚本，并在其中写入以下内容：

```py
import os
from ftplib import FTP ftp = FTP('your-ftp-domain-or-ip')
with ftp:
 ftp.login('your-username','your-password') ftp.cwd('/home/student/work/') files = ftp.nlst()
    print(files) # Print the files for file in files:
        if os.path.isfile(file): print("Downloading..." + file) ftp.retrbinary("RETR " + file ,open("/home/student/testing/" + file, 'wb').write) ftp.close()
```

按照以下方式运行脚本：

```py
student@ubuntu:~/work$ python3 get_ftp_files.py
```

您应该得到以下输出：

```py
Downloading...hello Downloading...hello.c Downloading...sample.txt Downloading...strip_hello Downloading...test.py
```

在前面的示例中，我们使用`ftplib`模块从主机检索了多个文件。首先，我们提到了另一台机器的IP地址、用户名和密码。为了从主机获取所有文件，我们使用了`ftp.nlst()`函数，为了将这些文件下载到我们的计算机，我们使用了`ftp.retrbinary()`函数。

# 使用getwelcome()获取欢迎消息：

一旦建立了初始连接，服务器通常会返回一个欢迎消息。这条消息通过`getwelcome()`函数传递，有时包括免责声明或对用户相关的有用信息。

现在我们将看到一个`getwelcome()`的示例。创建一个`get_welcome_msg.py`脚本，并在其中写入以下内容：

```py
from ftplib import FTP ftp = FTP('your-ftp-domain-or-ip') ftp.login('your-username','your-password') welcome_msg = ftp.getwelcome() print(welcome_msg) ftp.close()
```

按照以下方式运行脚本：

```py
student@ubuntu:~/work$ python3 get_welcome_msg.py 220 (vsFTPd 3.0.3)
```

在前面的代码中，我们首先提到了另一台机器的IP地址、用户名和密码。我们使用了`getwelcome()`函数在建立初始连接后获取信息。

# 使用sendcmd()函数向服务器发送命令

在本节中，我们将学习`sendcmd()`函数。我们可以使用`sendcmd()`函数向服务器发送一个简单的`string`命令以获取字符串响应。客户端可以发送FTP命令，如`STAT`、`PWD`、`RETR`和`STOR`。`ftplib`模块有多个方法可以包装这些命令。这些命令可以使用`sendcmd()`或`voidcmd()`方法发送。例如，我们将发送一个`STAT`命令来检查服务器的状态。

创建一个`send_command.py`脚本，并在其中写入以下内容：

```py
from ftplib import FTP ftp = FTP('your-ftp-domain-or-ip') ftp.login('your-username','your-password') ftp.cwd('/home/student/') s_cmd_stat = ftp.sendcmd('STAT') print(s_cmd_stat) print() s_cmd_pwd = ftp.sendcmd('PWD') print(s_cmd_pwd) print() ftp.close()
```

运行脚本如下：

```py
student@ubuntu:~/work$ python3 send_command.py
```

您将获得以下输出：

```py
211-FTP server status:
 Connected to ::ffff:192.168.2.109 Logged in as student TYPE: ASCII No session bandwidth limit Session timeout in seconds is 300 Control connection is plain text Data connections will be plain text At session startup, client count was 1 vsFTPd 3.0.3 - secure, fast, stable 211 End of status
257 "/home/student" is the current directory
```

在上面的代码中，我们首先提到了另一台机器的IP地址，用户名和密码。接下来，我们使用`sendcmd()`方法发送`STAT`命令到另一台机器。然后，我们使用`sendcmd()`发送`PWD`命令。

# urllib包

像`http`一样，`urllib`也是一个包，其中包含用于处理URL的各种模块。`urllib`模块允许您通过脚本访问多个网站。我们还可以使用该模块下载数据，解析数据，修改标头等。

`urllib`有一些不同的模块，列在这里：

+   `urllib.request`：用于打开和读取URL。

+   `urllib.error`：包含`urllib.request`引发的异常。

+   `urllib.parse`：用于解析URL。

+   `urllib.robotparser`：用于解析`robots.txt`文件。

在本节中，我们将学习如何使用`urllib`打开URL以及如何从URL读取`html`文件。我们将看到一个简单的`urllib`使用示例。我们将导入`urllib.requests`。然后我们将打开URL的操作赋给一个变量，然后我们将使用`.read()`命令从URL读取数据。

创建一个`url_requests_example.py`脚本，并在其中写入以下内容：

```py
import urllib.request x = urllib.request.urlopen('https://www.imdb.com/') print(x.read())
```

运行脚本如下：

```py
student@ubuntu:~/work$ python3 url_requests_example.py
```

以下是输出：

```py
b'\n\n<!DOCTYPE html>\n<html\n    \n    >\n    <head>\n         \n        <meta charset="utf-8">\n        <meta http-equiv="X-UA-Compatible" content="IE=edge">\n\n    \n    \n    \n\n    \n    \n    \n\n    <meta name="apple-itunes-app" content="app-id=342792525, app-argument=imdb:///?src=mdot">\n\n\n\n        <script type="text/javascript">var IMDbTimer={starttime: new Date().getTime(),pt:\'java\'};</script>\n\n<script>\n    if (typeof uet == \'function\') {\n      uet("bb", "LoadTitle", {wb: 1});\n    }\n</script>\n  <script>(function(t){ (t.events = t.events || {})["csm_head_pre_title"] = new Date().getTime(); })(IMDbTimer);</script>\n        <title>IMDb - Movies, TV and Celebrities - IMDb</title>\n  <script>(function(t){ (t.events = t.events || {})["csm_head_post_title"] = new Date().getTime(); })(IMDbTimer);</script>\n<script>\n    if (typeof uet == \'function\') {\n      uet("be", "LoadTitle", {wb: 1});\n    }\n</script>\n<script>\n    if (typeof uex == \'function\') {\n      uex("ld", "LoadTitle", {wb: 1});\n    }\n</script>\n\n        <link rel="canonical" href="https://www.imdb.com/" />\n        <meta property="og:url" content="http://www.imdb.com/" />\n        <link rel="alternate" media="only screen and (max-width: 640px)" href="https://m.imdb.com/">\n\n<script>\n    if (typeof uet == \'function\') {\n      uet("bb", "LoadIcons", {wb: 1});\n    }\n</script>\n  <script>(function(t){ (t.events = t.events || {})["csm_head_pre_icon"] = new Date().getTime(); })(IMDbTimer);</script>\n        <link href="https://m.media-amazon.com/images/G/01/imdb/images/safari-favicon-517611381._CB483525257_.svg" mask rel="icon" sizes="any">\n        <link rel="icon" type="image/ico" href="https://m.media-amazon.com/images/G/01/imdb/images/favicon-2165806970._CB470047330_.ico" />\n        <meta name="theme-color" content="#000000" />\n        <link rel="shortcut icon" type="image/x-icon" href="https://m.media-amazon.com/images/G/01/imdb/images/desktop-favicon-2165806970._CB484110913_.ico" />\n        <link href="https://m.media-amazon.com/images/G/01/imdb/images/mobile/apple-touch-icon-web-4151659188._CB483525313_.png" rel="apple-touch-icon"> \n
```

在上面的示例中，我们使用了`read()`方法，该方法返回字节数组。这会以非人类可读的格式打印`Imdb`主页返回的HTML数据，但我们可以使用HTML解析器从中提取一些有用的信息。

# Python urllib响应标头

我们可以通过在响应对象上调用`info()`函数来获取响应标头。这将返回一个字典，因此我们还可以从响应中提取特定的标头数据。创建一个`url_response_header.py`脚本，并在其中写入以下内容：

```py
import urllib.request x = urllib.request.urlopen('https://www.imdb.com/') print(x.info())
```

运行脚本如下：

```py
student@ubuntu:~/work$ python3 url_response_header.py
```

以下是输出：

```py
Server: Server Date: Fri, 23 Nov 2018 11:22:48 GMT Content-Type: text/html;charset=UTF-8 Transfer-Encoding: chunked Connection: close X-Frame-Options: SAMEORIGIN Content-Security-Policy: frame-ancestors 'self' imdb.com *.imdb.com *.media-imdb.com withoutabox.com *.withoutabox.com amazon.com *.amazon.com amazon.co.uk *.amazon.co.uk amazon.de *.amazon.de translate.google.com images.google.com www.google.com www.google.co.uk search.aol.com bing.com www.bing.com Content-Language: en-US Set-Cookie: uu=BCYsJu-IKhmmXuZWHgogzgofKfB8CXXLkNXdfKrrvsCP-RkcSn29epJviE8uRML4Xl4E7Iw9V09w%0D%0Anl3qKv1bEVJ-hHWVeDFH6BF8j_MMf8pdVA2NWzguWQ2XbKvDXFa_rK1ymzWc-Q35RCk_Z6jTj-Mk%0D%0AlEMrKkFyxbDYxLMe4hSjUo7NGrmV61LY3Aohaq7zE-ZE8a6DhgdlcLfXsILNXTkv7L3hvbxmr4An%0D%0Af73atPNPOgyLTB2S615MnlZ3QpOeNH6E2fElDYXZnsIFEAb9FW2XfQ%0D%0A; Domain=.imdb.com; Expires=Wed, 11-Dec-2086 14:36:55 GMT; Path=/; Secure Set-Cookie: session-id=000-0000000-0000000; Domain=.imdb.com; Expires=Wed, 11-Dec-2086 14:36:55 GMT; Path=/; Secure Set-Cookie: session-id-time=2173692168; Domain=.imdb.com; Expires=Wed, 11-Dec-2086 14:36:55 GMT; Path=/; Secure Vary: Accept-Encoding,X-Amzn-CDN-Cache,User-Agent x-amz-rid: GJDGQQTNA4MH7S3KJJKV
```

# 总结

在本章中，我们学习了套接字，用于双向客户端-服务器通信。我们学习了三个互联网模块：`http`，`ftplib`和`urllib`。`http`包具有客户端和服务器的模块：`http.client`和`http.server`。使用`ftplib`，我们从另一台机器下载文件。我们还查看了欢迎消息和发送`send`命令。

在下一章中，我们将介绍构建和发送电子邮件。我们将学习有关消息格式和添加多媒体内容。此外，我们将学习有关SMTP，POP和IMAP服务器的知识。

# 问题

1.  什么是套接字编程？

1.  什么是RPC？

1.  导入用户定义模块或文件的不同方式是什么？

1.  列表和元组之间有什么区别？

1.  字典中是否可以有重复的键？

1.  `urllib`，`urllib2`和`requests`模块之间有什么区别？

# 进一步阅读

+   `ftplib`文档：[https://docs.python.org/3/library/ftplib.html](https://docs.python.org/3/library/ftplib.html)

+   `xmlrpc`文档：[https://docs.python.org/3/library/xmlrpc.html](https://docs.python.org/3/library/xmlrpc.html)
