# 前言

全部赞美归于上帝！我很高兴这本书现在已经出版，并且我想感谢所有为这本书的出版做出贡献的人。本书是 Python 网络编程的探索性指南。它触及了广泛的网络协议，如 TCP/UDP、HTTP/HTTPS、FTP、SMTP、POP3、IMAP、CGI 等。凭借 Python 的力量和交互性，它为网络和系统管理、Web 应用程序开发、与本地和远程网络交互、低级网络数据包捕获和分析等现实世界任务编写各种脚本带来了乐趣和快乐。本书的主要重点是让您在所涵盖的主题上获得实践经验。因此，本书涉及的理论较少，但内容丰富，实用性强。

本书以“DevOps”思维模式编写，在这种模式下，开发者也多少负责运营工作，即部署应用程序并管理其各个方面，例如远程服务器管理、监控、扩展和优化以获得更好的性能。本书向您介绍了一系列开源的第三方 Python 库，它们在各种用例中都非常易于使用。我每天都在使用这些库中的许多来享受自动化我的 DevOps 任务。例如，我使用 Fabric 来自动化软件部署任务，以及其他库用于其他目的，例如在互联网上搜索事物、屏幕抓取或从 Python 脚本中发送电子邮件。

我希望您会喜欢本书中提供的食谱，并将它们扩展以使它们更加强大和有趣。

# 本书涵盖的内容

第一章，*套接字、IPv4 和简单的客户端/服务器编程*，通过一系列小任务向您介绍 Python 的核心网络库，并使您能够创建您的第一个客户端/服务器应用程序。

第二章，*多路复用套接字 I/O 以获得更好的性能*，讨论了使用默认和第三方库扩展您的客户端/服务器应用程序的各种有用技术。

第三章，*IPv6、Unix 域套接字和网络接口*，更多地关注管理您的本地机器和照顾您的本地局域网。

第四章，*使用 HTTP 进行互联网编程*，使您能够创建具有各种功能的小型命令行浏览器，例如提交 Web 表单、处理 Cookies、管理部分下载、压缩数据以及通过 HTTPS 提供安全内容。

第五章，*电子邮件协议、FTP 和 CGI 编程*，为您带来自动化 FTP 和电子邮件任务（如操作您的 Gmail 账户、从脚本中读取或发送电子邮件或为您的 Web 应用程序创建留言簿）的乐趣。

第六章，*屏幕抓取和其他实用应用*，介绍了各种第三方 Python 库，它们执行一些实用任务，例如在谷歌地图上定位公司、从维基百科抓取信息、在 GitHub 上搜索代码库或从 BBC 读取新闻。

第七章，*跨机器编程*，让您体验通过 SSH 自动化系统管理和部署任务。您可以从笔记本电脑远程运行命令、安装软件包或设置新的网站。

第八章，*与 Web 服务协作 – XML-RPC、SOAP 和 REST*，介绍了各种 API 协议，如 XML-RPC、SOAP 和 REST。您可以通过编程方式向任何网站或 Web 服务请求信息并与它们交互。例如，您可以在亚马逊或谷歌上搜索产品。

第九章，*网络监控和安全*，介绍了捕获、存储、分析和操作网络数据包的各种技术。这鼓励您进一步使用简洁的 Python 脚本来调查您的网络安全问题。

# 您需要为这本书准备的东西

您需要一个运行良好的 PC 或笔记本电脑，最好使用任何现代 Linux 操作系统，如 Ubuntu、Debian、CentOS 等。本书中的大多数食谱在其他平台（如 Windows 和 Mac OS）上也能运行。

您还需要一个有效的互联网连接来安装文中提到的第三方软件库。如果您没有互联网连接，您可以下载这些第三方库并一次性安装。

以下是一个包含其下载 URL 的第三方库列表：

+   **ntplib**: [`pypi.python.org/pypi/ntplib/`](https://pypi.python.org/pypi/ntplib/)

+   **diesel**: [`pypi.python.org/pypi/diesel/`](https://pypi.python.org/pypi/diesel/)

+   **nmap**: [`pypi.python.org/pypi/python-nmap`](https://pypi.python.org/pypi/python-nmap)

+   **scapy**: [`pypi.python.org/pypi/scapy`](https://pypi.python.org/pypi/scapy)

+   **netifaces**: [`pypi.python.org/pypi/netifaces/`](https://pypi.python.org/pypi/netifaces/)

+   **netaddr**: [`pypi.python.org/pypi/netaddr`](https://pypi.python.org/pypi/netaddr)

+   **pyopenssl**: [`pypi.python.org/pypi/pyOpenSSL`](https://pypi.python.org/pypi/pyOpenSSL)

+   **pygeocoder**: [`pypi.python.org/pypi/pygocoder`](https://pypi.python.org/pypi/pygocoder)

+   **pyyaml**: [`pypi.python.org/pypi/PyYAML`](https://pypi.python.org/pypi/PyYAML)

+   **requests**: [`pypi.python.org/pypi/requests`](https://pypi.python.org/pypi/requests)

+   **feedparser**: [`pypi.python.org/pypi/feedparser`](https://pypi.python.org/pypi/feedparser)

+   **paramiko**: [`pypi.python.org/pypi/paramiko/`](https://pypi.python.org/pypi/paramiko/)

+   **fabric**: [`pypi.python.org/pypi/Fabric`](https://pypi.python.org/pypi/Fabric)

+   **supervisor**: [`pypi.python.org/pypi/supervisor`](https://pypi.python.org/pypi/supervisor)

+   **xmlrpclib**: [`pypi.python.org/pypi/xmlrpclib`](https://pypi.python.org/pypi/xmlrpclib)

+   **SOAPpy**: [`pypi.python.org/pypi/SOAPpy`](https://pypi.python.org/pypi/SOAPpy)

+   **bottlenose**: [`pypi.python.org/pypi/bottlenose`](https://pypi.python.org/pypi/bottlenose)

+   **construct**: [`pypi.python.org/pypi/construct/`](https://pypi.python.org/pypi/construct/)

运行某些食谱所需的非 Python 软件如下：

+   **postfix**: [`www.postfix.org/`](http://www.postfix.org/)

+   **openssh 服务器**: [`www.openssh.com/`](http://www.openssh.com/)

+   **mysql 服务器**: [`downloads.mysql.com/`](http://downloads.mysql.com/)

+   **apache2**: [`httpd.apache.org/download.cgi`](http://httpd.apache.org/download.cgi)

# 本书面向对象

如果你是一名网络程序员、系统/网络管理员或 Web 应用开发者，这本书非常适合你。你应该对 Python 编程语言和 TCP/IP 网络概念有基本的了解。然而，如果你是新手，你将在阅读本书的过程中逐渐理解这些概念。本书可作为任何网络编程学术课程中开发动手技能的补充材料。

# 习惯用法

在这本书中，你会发现许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名如下所示：

如果你需要知道远程机器的 IP 地址，你可以使用内置库函数`gethostbyname()`。

代码块设置如下：

```py
def test_socket_timeout():
   s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  print "Default socket timeout: %s" %s.gettimeout()
  s.settimeout(100)
  print "Current socket timeout: %s" %s.gettimeout()
```

任何命令行输入或输出都如下所示：

```py
$ python 2_5_echo_server_with_diesel.py --port=8800
[2013/04/08 11:48:32] {diesel} WARNING:Starting diesel <hand-rolledselect.epoll>
```

**新术语**和**重要词汇**以粗体显示。

### 注意

警告或重要注意事项以如下框的形式出现。

### 小贴士

小贴士和技巧看起来像这样。

# 读者反馈

我们欢迎读者的反馈。请告诉我们您对这本书的看法——您喜欢什么或可能不喜欢什么。读者反馈对我们开发您真正能从中获得最大收益的标题非常重要。

要发送一般反馈，只需将电子邮件发送到`<feedback@packtpub.com>`，并在邮件主题中提及书名。

如果你在某个领域有专业知识，并且对撰写或参与一本书感兴趣，请参阅我们的作者指南 [www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在你已经是 Packt 书籍的骄傲拥有者，我们有一些事情可以帮助你从购买中获得最大收益。

## 下载示例代码

您可以从 [`www.packtpub.com`](http://www.packtpub.com) 的账户下载您购买的 Packt 书籍的示例代码文件。如果您在其他地方购买了这本书，您可以访问 [`www.packtpub.com/support`](http://www.packtpub.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

## 错误清单

尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何错误清单，请通过访问 [`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**错误提交**表单链接，并输入您的错误详细信息来报告它们。一旦您的错误得到验证，您的提交将被接受，错误将被上传到我们的网站，或添加到该标题的错误清单中。任何现有的错误清单都可以通过从 [`www.packtpub.com/support`](http://www.packtpub.com/support) 选择您的标题来查看。

## 盗版

互联网上版权材料的盗版是一个跨所有媒体的持续问题。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何非法副本，无论形式如何，请立即提供位置地址或网站名称，以便我们可以追究补救措施。

请通过 `<copyright@packtpub.com>` 与我们联系，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们作者和提供有价值内容方面的帮助。

## 询问

如果你在本书的任何方面遇到问题，可以通过 `<questions@packtpub.com>` 联系我们，我们将尽力解决。
