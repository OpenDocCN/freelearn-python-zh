# 前言

最近，Python开始受到越来越多的关注，最新的Python更新添加了许多可用于执行关键任务的包。我们的主要目标是帮助您利用Python包来检测和利用漏洞，并解决网络挑战。

本书将首先带您了解与网络和安全相关的Python脚本和库。然后，您将深入了解核心网络任务，并学习如何解决网络挑战。随后，本书将教您如何编写安全脚本，以检测网络或网站中的漏洞。通过本书，您将学会如何利用Python包实现端点保护，以及如何编写取证和加密脚本。

# 本书适合对象

本书非常适合网络工程师、系统管理员以及希望解决网络和安全挑战的任何安全专业人士。对Python及其网络和安全包感兴趣的安全研究人员和开发人员也会从本书中受益匪浅。

# 本书涵盖内容

[第1章](a1b0651a-c973-4af1-bc35-dd4e1fe8368a.xhtml)，*使用Python脚本*，向您介绍了Python语言、面向对象编程、数据结构、以及使用Python进行开发的方法和开发环境。

[第2章](a00521f9-8119-4877-aee1-b24e589cc432.xhtml)，*系统编程包*，教授您有关系统编程的主要Python模块，涵盖主题包括读写文件、线程、套接字、多线程和并发。

[第3章](bd1e16c1-2ce3-4edc-b61d-9845d978c2bd.xhtml)，*套接字编程*，为您提供了使用socket模块进行Python网络编程的一些基础知识。socket模块公开了编写TCP和UDP客户端以及服务器所需的所有必要部分，用于编写低级网络应用程序。

第4章，*HTTP编程*，涵盖了HTTP协议和主要的Python模块，如urllib标准库和requests包。我们还涵盖了HTTP身份验证机制以及如何使用requests模块来管理它们。

[第5章](40fd3a5e-4f71-4067-a0ce-6f0ba212af70.xhtml)，*分析网络流量*，为您提供了使用Scapy在Python中分析网络流量的一些基础知识。调查人员可以编写Scapy脚本来调查通过嗅探混杂网络接口的实时流量，或加载先前捕获的`pcap`文件。

[第6章](f294d743-c9f1-40f5-a9b7-9904d7f634b2.xhtml)，*从服务器获取信息*，探讨了允许提取服务器公开的信息的模块，如Shodan。我们还研究了获取服务器横幅和DNS服务器信息，并向您介绍了模糊处理。

[第7章](321a63e9-bf32-449a-9673-4991ab97234f.xhtml)，*与FTP、SSH和SNMP服务器交互*，详细介绍了允许我们与FTP、SSH和SNMP服务器交互的Python模块。

[第8章](ee538860-9660-4043-9296-143e62f27a61.xhtml)，*使用Nmap扫描器*，介绍了Nmap作为端口扫描器，并介绍了如何使用Python和Nmap实现网络扫描，以获取有关网络、特定主机以及在该主机上运行的服务的信息。此外，我们还介绍了编写例程以查找Nmap脚本中给定网络可能存在的漏洞。

[第9章](0125c9f4-5653-47c1-9097-375f4891a926.xhtml)，*与Metasploit框架连接*，介绍了Metasploit框架作为利用漏洞的工具，并探讨了如何使用`python-msfprc`和`pymetasploit`模块。

第10章，“与漏洞扫描器交互”，介绍了Nessus和Nexpose作为漏洞扫描器，并为它们在服务器和Web应用程序中发现的主要漏洞提供了报告工具。此外，我们还介绍了如何使用Python中的`nessrest`和`Pynexpose`模块对它们进行程序化操作。

第11章，“识别Web应用程序中的服务器漏洞”，涵盖了OWASP方法论中的Web应用程序中的主要漏洞，以及Python生态系统中用于Web应用程序漏洞扫描的工具。我们还介绍了如何测试服务器中的openSSL漏洞。

第12章，“从文档、图片和浏览器中提取地理位置和元数据”，探讨了Python中用于从图片和文档中提取地理位置和元数据、识别Web技术以及从Chrome和Firefox中提取元数据的主要模块。

第13章，“加密和隐写术”，深入探讨了Python中用于加密和解密信息的主要模块，如`pycrypto`和cryptography。此外，我们还介绍了隐写术技术以及如何使用`stepic`模块在图片中隐藏信息。

# 为了充分利用本书

您需要在本地计算机上安装Python发行版，内存至少为4GB。

在第9章、第10章和第11章中，我们将使用一个名为metasploitable的虚拟机，用于进行与端口分析和漏洞检测相关的一些测试。可以从SourceForge页面下载：

[https://sourceforge.net/projects/metasploitable/files/Metasploitable2](https://sourceforge.net/projects/metasploitable/files/Metasploitable2)

对于第9章，您还需要安装Kali Linux发行版和Python，以执行Metasploit Framework。

在本书中，您可以找到基于Python 2和3版本的示例。虽然许多示例可以在Python 2中运行，但使用最新版本的Python 3会获得最佳体验。在撰写本文时，最新版本为2.7.14和3.6.15，并且这些示例已针对这些版本进行了测试。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便直接通过电子邮件接收文件。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packt.com](http://www.packt.com)。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在“搜索”框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩软件解压文件夹：

+   Windows上的WinRAR/7-Zip

+   Mac上的Zipeg/iZip/UnRarX

+   Linux上的7-Zip/PeaZip

该书的代码包也托管在GitHub上，网址为[https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security](https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security)。如果代码有更新，将在现有的GitHub存储库中进行更新。

我们还有其他代码包，来自我们丰富的图书和视频目录，可在[https://github.com/PacktPublishing/](https://github.com/PacktPublishing/)上找到。去看看吧！

# 下载彩色图片

我们还提供了一个PDF文件，其中包含本书中使用的屏幕截图/图表的彩色图像。您可以在此处下载：[https://www.packtpub.com/sites/default/files/downloads/9781788992510_ColorImages.pdf](https://www.packtpub.com/sites/default/files/downloads/9781788992510_ColorImages.pdf)

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟URL、用户输入和Twitter用户名。例如："将下载的`WebStorm-10*.dmg`磁盘映像文件挂载为系统中的另一个磁盘。"

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
