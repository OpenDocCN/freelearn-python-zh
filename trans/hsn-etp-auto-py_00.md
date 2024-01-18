# 前言

本书首先介绍了建立Python环境以执行自动化任务的设置，以及您将使用的模块、库和工具。

我们将使用简单的Python程序和Ansible探索网络自动化任务的示例。接下来，我们将带您了解如何使用Python Fabric自动化管理任务，您将学习执行服务器配置和管理以及系统管理任务，如用户管理、数据库管理和进程管理。随着您在本书中的进展，您将使用Python脚本自动化多个测试服务，并使用Python在虚拟机和云基础架构上执行自动化任务。在最后几章中，您将涵盖基于Python的攻击性安全工具，并学习如何自动化您的安全任务。

通过本书，您将掌握使用Python自动化多个系统管理任务的技能。

您可以访问作者的博客，链接如下：[https://basimaly.wordpress.com/.](https://basimaly.wordpress.com/)

# 这本书是为谁准备的

*使用Python进行企业自动化*适用于寻找Puppet和Chef等主要自动化框架替代方案的系统管理员和DevOps工程师。需要具备Python和Linux shell脚本的基本编程知识。

# 这本书涵盖了什么

[第1章](part0020.html#J2B80-9cfcdc5beecd470bbeda046372f0337f)，*设置Python环境*，探讨了如何下载和安装Python解释器以及名为*JetBrains PyCharm*的Python集成开发环境。该IDE提供智能自动完成、智能代码分析、强大的重构，并与Git、virtualenv、Vagrant和Docker集成。这将帮助您编写专业和健壮的Python代码。

[第2章](part0034.html#10DJ40-9cfcdc5beecd470bbeda046372f0337f)，*自动化中使用的常见库*，介绍了今天可用的用于自动化的Python库，并根据它们的用途（系统、网络和云）进行分类，并提供简单的介绍。随着您在本书中的进展，您将发现自己深入研究每一个库，并了解它们的用途。

[第3章](part0020.html#J2B80-9cfcdc5beecd470bbeda046372f0337f)，*设置您的网络实验室环境*，讨论了网络自动化的优点以及网络运营商如何使用它来自动化当前的设备。我们将探讨今天用于自动化来自思科、Juniper和Arista的网络节点的流行库。本章介绍了如何构建一个网络实验室来应用Python脚本。我们将使用一个名为EVE-NG的开源网络仿真工具。

[第4章](part0062.html#1R42S0-9cfcdc5beecd470bbeda046372f0337f)，*使用Python管理网络设备*，深入介绍了通过telnet和SSH连接使用netmiko、paramiko和telnetlib管理网络设备。我们将学习如何编写Python代码来访问交换机和路由器，并在终端上执行命令，然后返回输出。我们还将学习如何利用不同的Python技术来备份和推送配置。本章以当今现代网络环境中使用的一些用例结束。

[第5章](part0087.html#2IV0U0-9cfcdc5beecd470bbeda046372f0337f)，*从网络设备中提取有用数据*，解释了如何使用Python内部的不同工具和技术从返回的输出中提取有用数据并对其进行操作。此外，我们将使用一个名为*CiscoConfParse*的特殊库来审计配置。然后，我们将学习如何可视化数据，使用matplotlib生成吸引人的图表和报告。

第6章《使用Python和Jinja2生成配置文件》解释了如何为拥有数百个网络节点的站点生成通用配置。我们将学习如何编写模板，并使用Jinja2模板语言生成黄金配置。

第7章《Python脚本的并行执行》涵盖了如何并行实例化和执行Python代码。只要不相互依赖，这将使我们能够更快地完成自动化工作流程。

第8章《准备实验室环境》涵盖了实验室环境的安装过程和准备工作。我们将在不同的虚拟化器上安装我们的自动化服务器，无论是在CentOS还是Ubuntu上。然后我们将学习如何使用Cobbler自动安装操作系统。

第9章《使用Subprocess模块》解释了如何从Python脚本直接发送命令到操作系统shell并调查返回的输出。

第10章《使用Fabric运行系统管理任务》介绍了Fabric，这是一个用于通过SSH执行系统管理任务的Python库。它也用于大规模应用部署。我们将学习如何利用和发挥这个库来在远程服务器上执行任务。

第11章《生成系统报告》、《管理用户和系统监控》解释了从系统收集数据并生成定期报告对于任何系统管理员来说都是一项重要任务，自动化这项任务将帮助您及早发现问题并为其提供解决方案。在本章中，我们将看到一些经过验证的自动化从服务器收集数据并生成正式报告的方法。我们将学习如何使用Python和Ansible管理新用户和现有用户。此外，我们还将深入研究系统KPI的监控和日志分析。您还可以安排监控脚本定期运行，并将结果发送到您的邮箱。

第12章《与数据库交互》指出，如果你是数据库管理员或数据库开发人员，那么Python提供了许多库和模块，涵盖了管理和操作流行的DBMS（如MySQL、Postgress和Oracle）。在本章中，我们将学习如何使用Python连接器与DBMS进行交互。

第13章《系统管理的Ansible》探讨了配置管理软件中最强大的工具之一。当涉及系统管理时，Ansible非常强大，可以确保配置在数百甚至数千台服务器上同时精确复制。

第14章《创建和管理VMWare虚拟机》解释了如何在VMWare虚拟化器上自动创建VM。我们将探索使用VMWare官方绑定库在ESXi上创建和管理虚拟机的不同方法。

第15章《与Openstack API交互》解释了在创建私有云时，OpenStack在私有IaaS方面非常受欢迎。我们将使用Python模块，如requests，创建REST调用并与OpenStack服务（如nova、cinder和neutron）进行交互，并在OpenStack上创建所需的资源。在本章后期，我们将使用Ansible playbooks执行相同的工作流程。

[第16章](part0217.html#6EUA20-9cfcdc5beecd470bbeda046372f0337f)，*使用Python和Boto3自动化AWS*，介绍了如何使用官方的Amazon绑定（BOTO3）自动化常见的AWS服务，如EC2和S3，它提供了一个易于使用的API来访问服务。

[第17章](part0227.html#6OFFM0-9cfcdc5beecd470bbeda046372f0337f)，*使用SCAPY框架*，介绍了SCAPY，这是一个强大的Python工具，用于构建和制作数据包，然后将其发送到网络上。您可以构建任何类型的网络流并将其发送到网络上。它还可以帮助您捕获网络数据包并将其重放到网络上。

[第18章](part0240.html#74S700-9cfcdc5beecd470bbeda046372f0337f)，*使用Python构建网络扫描器*，提供了使用Python构建网络扫描器的完整示例。您可以扫描完整的子网以查找不同的协议和端口，并为每个扫描的主机生成报告。然后，我们将学习如何通过Git与开源社区（GitHub）共享代码。

# 为了充分利用本书

读者应该熟悉Python编程语言的基本编程范式，并且应该具有Linux和Linux shell脚本的基本知识。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便直接将文件发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)上登录或注册。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩软件解压缩文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在GitHub上，网址为[https://github.com/PacktPublishing/Hands-On-Enterprise-Automation-with-Python](https://github.com/PacktPublishing/Hands-On-Enterprise-Automation-with-Python)。如果代码有更新，将在现有的GitHub存储库上进行更新。

我们还有来自我们丰富的图书和视频目录的其他代码包，可在**[https://github.com/PacktPublishing/](https://github.com/PacktPublishing/)**上找到。去看看吧！

# 下载彩色图片

我们还提供了一个PDF文件，其中包含本书中使用的屏幕截图/图表的彩色图片。您可以在这里下载：[http://www.packtpub.com/sites/default/files/downloads/HandsOnEnterpriseAutomationwithPython_ColorImages.pdf](http://www.packtpub.com/sites/default/files/downloads/HandsOnEnterpriseAutomationwithPython_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码字词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟URL、用户输入和Twitter句柄。这是一个例子："一些大型软件包，如`matplotlib`或`django`，其中包含数百个模块，开发人员通常将相关模块分类到子目录中。"

代码块设置如下：

```py
from netmiko import ConnectHandler
from devices import R1,SW1,SW2,SW3,SW4

nodes = [R1,SW1,SW2,SW3,SW4]   for device in nodes:
  net_connect = ConnectHandler(**device)
  output = net_connect.send_command("show run")
  print output
```

当我们希望引起您对代码块的特定部分的注意时，相关的行或项目将以粗体显示：

```py
hostname {{hostname}}
```

任何命令行输入或输出都将按照以下方式编写：

```py
pip install jinja2 
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以这种方式出现在文本中。这是一个例子：

"从下载页面选择您的平台，然后选择x86或x64版本。"

警告或重要说明会以这种方式出现。提示和技巧会以这种方式出现。
