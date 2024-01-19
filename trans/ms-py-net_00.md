# 前言

正如查尔斯·狄更斯在《双城记》中写道，“这是最好的时代，也是最坏的时代，这是智慧的时代，也是愚蠢的时代。”他看似矛盾的话语完美地描述了变革和过渡时期的混乱和情绪。毫无疑问，我们正在经历网络工程领域的快速变化。随着软件开发在网络的各个方面变得更加集成，传统的命令行界面和垂直集成的网络堆栈方法不再是管理今天网络的最佳方式。对于网络工程师来说，我们所看到的变化充满了兴奋和机遇，但对于那些需要快速适应和跟上的人来说，也是具有挑战性的。本书旨在通过提供一个实用指南来帮助网络专业人士缓解过渡，解决如何从传统平台发展到基于软件驱动实践的问题。

在这本书中，我们使用 Python 作为首选的编程语言，以掌握网络工程任务。Python 是一种易于学习的高级编程语言，可以有效地补充网络工程师的创造力和问题解决能力，以简化日常操作。Python 正在成为许多大型网络的一个组成部分，通过这本书，我希望与您分享我所学到的经验。

自第一版出版以来，我已经与许多读者进行了有趣而有意义的交流。第一版书的成功让我感到谦卑，并且我对所得到的反馈非常重视。在第二版中，我尝试使示例和技术更加相关。特别是，传统的 OpenFlow SDN 章节被一些网络 DevOps 工具所取代。我真诚地希望新的内容对你有所帮助。

变革的时代为技术进步提供了巨大的机遇。本书中的概念和工具在我的职业生涯中帮助了我很多，我希望它们也能对你有同样的帮助。

# 这本书适合谁

这本书非常适合已经管理网络设备组并希望扩展他们对使用 Python 和其他工具克服网络挑战的知识的 IT 专业人员和运维工程师。建议具有网络和 Python 的基本知识。

# 本书涵盖内容

第一章，*TCP/IP 协议套件和 Python 回顾*，回顾了构成当今互联网通信的基本技术，从 OSI 和客户端-服务器模型到 TCP、UDP 和 IP 协议套件。本章将回顾 Python 语言的基础知识，如类型、运算符、循环、函数和包。

第二章，*低级网络设备交互*，使用实际示例说明如何使用 Python 在网络设备上执行命令。它还将讨论在自动化中仅具有 CLI 界面的挑战。本章将使用 Pexpect 和 Paramiko 库进行示例。

第三章，*API 和意图驱动的网络*，讨论了支持**应用程序编程接口**（**API**）和其他高级交互方法的新型网络设备。它还说明了允许在关注网络工程师意图的同时抽象低级任务的工具。本章将使用 Cisco NX-API、Juniper PyEZ 和 Arista Pyeapi 的讨论和示例。

第四章，《Python 自动化框架- Ansible 基础》，讨论了 Ansible 的基础知识，这是一个基于 Python 的开源自动化框架。Ansible 比 API 更进一步，专注于声明性任务意图。在本章中，我们将介绍使用 Ansible 的优势、其高级架构，并展示一些与思科、Juniper 和 Arista 设备一起使用 Ansible 的实际示例。

第五章，《Python 自动化框架-进阶》，在前一章的基础上，涵盖了更高级的 Ansible 主题。我们将介绍条件、循环、模板、变量、Ansible Vault 和角色。还将介绍编写自定义模块的基础知识。

第六章，《Python 网络安全》，介绍了几种 Python 工具，帮助您保护网络。将讨论使用 Scapy 进行安全测试，使用 Ansible 快速实施访问列表，以及使用 Python 进行网络取证分析。

第七章，《Python 网络监控-第 1 部分》，涵盖了使用各种工具监控网络。本章包含了一些使用 SNMP 和 PySNMP 进行查询以获取设备信息的示例。还将展示 Matplotlib 和 Pygal 示例来绘制结果。本章将以使用 Python 脚本作为输入源的 Cacti 示例结束。

第八章，《Python 网络监控-第 2 部分》，涵盖了更多的网络监控工具。本章将从使用 Graphviz 根据 LLDP 信息绘制网络开始。我们将继续使用推送式网络监控的示例，使用 Netflow 和其他技术。我们将使用 Python 解码流数据包和 ntop 来可视化结果。还将概述 Elasticsearch 以及如何用于网络监控。

第九章，《使用 Python 构建网络 Web 服务》，向您展示如何使用 Python Flask Web 框架为网络自动化创建自己的 API。网络 API 提供了诸如将请求者与网络细节抽象化、整合和定制操作以及通过限制可用操作的暴露来提供更好的安全性等好处。

第十章，《AWS 云网络》，展示了如何使用 AWS 构建一个功能齐全且具有弹性的虚拟网络。我们将介绍诸如 CloudFormation、VPC 路由表、访问列表、弹性 IP、NAT 网关、Direct Connect 等虚拟私有云技术以及其他相关主题。

第十一章，《使用 Git 工作》，我们将说明如何利用 Git 进行协作和代码版本控制。本章将使用 Git 进行网络操作的实际示例。

第十二章，《Jenkins 持续集成》，使用 Jenkins 自动创建操作流水线，可以节省时间并提高可靠性。

第十三章，《网络的测试驱动开发》，解释了如何使用 Python 的 unittest 和 PyTest 创建简单的测试来验证我们的代码。我们还将看到编写用于验证可达性、网络延迟、安全性和网络事务的网络测试的示例。我们还将看到如何将这些测试集成到 Jenkins 等持续集成工具中。

# 为了充分利用本书

为了充分利用本书，建议具备一些基本的网络操作知识和 Python 知识。大多数章节可以任意顺序阅读，但第四章和第五章必须按顺序阅读。除了书的开头介绍的基本软件和硬件工具外，每个章节还会介绍与该章节相关的新工具。

强烈建议按照自己的网络实验室中显示的示例进行跟踪和练习。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的账户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packtpub.com/support](http://www.packtpub.com/support)注册，以便直接通过电子邮件接收文件。

您可以按照以下步骤下载代码文件：

1.  登录或注册网址为[www.packtpub.com](http://www.packtpub.com/support)。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在“搜索”框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩或提取文件夹：

+   WinRAR/7-Zip 适用于 Windows

+   Zipeg/iZip/UnRarX 适用于 Mac

+   7-Zip/PeaZip 适用于 Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Mastering-Python-Networking-Second-Edition`](https://github.com/PacktPublishing/Mastering-Python-Networking-Second-Edition)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有其他代码包，来自我们丰富的图书和视频目录，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。快去看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图片。您可以在此处下载：[`www.packtpub.com/sites/default/files/downloads/MasteringPythonNetworkingSecondEdition_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/MasteringPythonNetworkingSecondEdition_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。例如：“自动配置还生成了`vty`访问，用于 telnet 和 SSH。”

代码块设置如下：

```py
# This is a comment
print("hello world")
```

任何命令行输入或输出都按照以下格式编写：

```py
$ python
Python 2.7.12 (default, Dec 4 2017, 14:50:18)
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> exit()
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以这种形式出现在文本中。例如：“在‘拓扑设计’选项中，我将‘管理网络’选项设置为‘共享平面网络’，以便在虚拟路由器上使用 VMnet2 作为管理网络。”

警告或重要提示会以这种形式出现。提示和技巧会以这种形式出现。
