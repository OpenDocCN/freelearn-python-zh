# 前言

最近，Python 开始受到越来越多的关注，最新的 Python 更新添加了许多可用于执行关键任务的包。我们的主要目标是帮助您利用 Python 包来检测和利用漏洞，并解决网络挑战。

本书将首先带您了解与网络和安全相关的 Python 脚本和库。然后，您将深入了解核心网络任务，并学习如何解决网络挑战。随后，本书将教您如何编写安全脚本，以检测网络或网站中的漏洞。通过本书，您将学会如何利用 Python 包实现端点保护，以及如何编写取证和加密脚本。

# 本书适合对象

本书非常适合网络工程师、系统管理员以及希望解决网络和安全挑战的任何安全专业人士。对 Python 及其网络和安全包感兴趣的安全研究人员和开发人员也会从本书中受益匪浅。

# 本书涵盖内容

第一章，*使用 Python 脚本*，向您介绍了 Python 语言、面向对象编程、数据结构、以及使用 Python 进行开发的方法和开发环境。

第二章，*系统编程包*，教授您有关系统编程的主要 Python 模块，涵盖主题包括读写文件、线程、套接字、多线程和并发。

第三章，*套接字编程*，为您提供了使用 socket 模块进行 Python 网络编程的一些基础知识。socket 模块公开了编写 TCP 和 UDP 客户端以及服务器所需的所有必要部分，用于编写低级网络应用程序。

第四章，*HTTP 编程*，涵盖了 HTTP 协议和主要的 Python 模块，如 urllib 标准库和 requests 包。我们还涵盖了 HTTP 身份验证机制以及如何使用 requests 模块来管理它们。

第五章，*分析网络流量*，为您提供了使用 Scapy 在 Python 中分析网络流量的一些基础知识。调查人员可以编写 Scapy 脚本来调查通过嗅探混杂网络接口的实时流量，或加载先前捕获的`pcap`文件。

第六章，*从服务器获取信息*，探讨了允许提取服务器公开的信息的模块，如 Shodan。我们还研究了获取服务器横幅和 DNS 服务器信息，并向您介绍了模糊处理。

第七章，*与 FTP、SSH 和 SNMP 服务器交互*，详细介绍了允许我们与 FTP、SSH 和 SNMP 服务器交互的 Python 模块。

第八章，*使用 Nmap 扫描器*，介绍了 Nmap 作为端口扫描器，并介绍了如何使用 Python 和 Nmap 实现网络扫描，以获取有关网络、特定主机以及在该主机上运行的服务的信息。此外，我们还介绍了编写例程以查找 Nmap 脚本中给定网络可能存在的漏洞。

第九章，*与 Metasploit 框架连接*，介绍了 Metasploit 框架作为利用漏洞的工具，并探讨了如何使用`python-msfprc`和`pymetasploit`模块。

第十章，“与漏洞扫描器交互”，介绍了 Nessus 和 Nexpose 作为漏洞扫描器，并为它们在服务器和 Web 应用程序中发现的主要漏洞提供了报告工具。此外，我们还介绍了如何使用 Python 中的`nessrest`和`Pynexpose`模块对它们进行程序化操作。

第十一章，“识别 Web 应用程序中的服务器漏洞”，涵盖了 OWASP 方法论中的 Web 应用程序中的主要漏洞，以及 Python 生态系统中用于 Web 应用程序漏洞扫描的工具。我们还介绍了如何测试服务器中的 openSSL 漏洞。

第十二章，“从文档、图片和浏览器中提取地理位置和元数据”，探讨了 Python 中用于从图片和文档中提取地理位置和元数据、识别 Web 技术以及从 Chrome 和 Firefox 中提取元数据的主要模块。

第十三章，“加密和隐写术”，深入探讨了 Python 中用于加密和解密信息的主要模块，如`pycrypto`和 cryptography。此外，我们还介绍了隐写术技术以及如何使用`stepic`模块在图片中隐藏信息。

# 为了充分利用本书

您需要在本地计算机上安装 Python 发行版，内存至少为 4GB。

在第九章、第十章和第十一章中，我们将使用一个名为 metasploitable 的虚拟机，用于进行与端口分析和漏洞检测相关的一些测试。可以从 SourceForge 页面下载：

[`sourceforge.net/projects/metasploitable/files/Metasploitable2`](https://sourceforge.net/projects/metasploitable/files/Metasploitable2)

对于第九章，您还需要安装 Kali Linux 发行版和 Python，以执行 Metasploit Framework。

在本书中，您可以找到基于 Python 2 和 3 版本的示例。虽然许多示例可以在 Python 2 中运行，但使用最新版本的 Python 3 会获得最佳体验。在撰写本文时，最新版本为 2.7.14 和 3.6.15，并且这些示例已针对这些版本进行了测试。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便直接通过电子邮件接收文件。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packt.com](http://www.packt.com)。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在“搜索”框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩软件解压文件夹：

+   Windows 上的 WinRAR/7-Zip

+   Mac 上的 Zipeg/iZip/UnRarX

+   Linux 上的 7-Zip/PeaZip

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security`](https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security)。如果代码有更新，将在现有的 GitHub 存储库中进行更新。

我们还有其他代码包，来自我们丰富的图书和视频目录，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。去看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图像。您可以在此处下载：[`www.packtpub.com/sites/default/files/downloads/9781788992510_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781788992510_ColorImages.pdf)

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名。例如："将下载的`WebStorm-10*.dmg`磁盘映像文件挂载为系统中的另一个磁盘。"

代码块设置如下：

```py
import requests
if __name__ == "__main__":
    response = requests.get("http://www.python.org")
    for header in response.headers.keys():
        print(header  + ":" + response.headers[header])
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```py
import requests
http_proxy = "http://<ip_address>:<port>"
proxy_dictionary = { "http" : http_proxy}
requests.get("http://example.org", proxies=proxy_dictionary)
```

任何命令行输入或输出都将按如下方式编写：

```py
$ pip install packagename
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以这种方式出现在文本中。例如："从管理面板中选择系统信息。"

警告或重要提示会以这种方式出现。提示和技巧会以这种方式出现。
