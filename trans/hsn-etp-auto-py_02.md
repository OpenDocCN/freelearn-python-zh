# 自动化中常用的库

本章将带您了解Python包的结构以及今天用于自动化系统和网络基础设施的常见库。有一个不断增长的Python包列表，涵盖了网络自动化、系统管理以及管理公共和私有云的功能。

此外，重要的是要理解如何访问模块源代码，以及Python包中的小部分如何相互关联，这样我们就可以修改代码，添加或删除功能，并再次与社区分享代码。

本章将涵盖以下主题：

+   理解Python包

+   常见的Python库

+   访问模块源代码

# 理解Python包

Python核心代码实际上是小而简单的。大部分功能都是通过添加第三方包和模块来实现的。

模块是一个包含函数、语句和类的Python文件，将在您的代码中使用。首先要做的是`import`模块，然后开始使用它的函数。

另一方面，一个**包**会收集相关的模块并将它们放在一个层次结构中。一些大型包，如`matplotlib`或`django`，其中包含数百个模块，开发人员通常会将相关的模块分类到子目录中。例如，`netmiko`包包含多个子目录，每个子目录包含连接到不同供应商的网络设备的模块：

![](../images/00031.jpeg)

这样做可以让包的维护者灵活地向每个模块添加或删除功能，而不会破坏全局包的操作。

# 包搜索路径

通常情况下，Python会在一些特定的系统路径中搜索模块。您可以通过导入`sys`模块并打印`sys.path`来打印这些路径。这实际上会返回`PYTHONPATH`环境变量和操作系统中的字符串；请注意结果只是一个普通的Python列表。您可以使用列表函数（如`insert()`）添加更多路径到搜索范围。

然而，最好是将包安装在默认搜索路径中，这样当与其他开发人员共享代码时，代码不会出错：

![](../images/00032.jpeg)

一个简单的包结构，只有一个模块，会是这样的：

![](../images/00033.jpeg)

每个包中的`__init__`文件（在全局目录或子目录中）会告诉Python解释器这个目录是一个Python包，每个以`.py`结尾的文件都是一个模块文件，可以在你的代码中导入。`init`文件的第二个功能是一旦导入包，就执行其中的任何代码。然而，大多数开发人员会将其留空，只是用它来标记目录为Python包。

# 常见的Python库

在接下来的章节中，我们将探讨用于网络、系统和云自动化的常见Python库。

# 网络Python库

如今的网络环境中包含来自许多供应商的多个设备，每个设备扮演不同的角色。设计和自动化网络设备的框架对于网络工程师来说至关重要，可以自动执行重复的任务，提高他们通常完成工作的方式，同时减少人为错误。大型企业和服务提供商通常倾向于设计一个能够自动执行不同网络任务并提高网络弹性和灵活性的工作流程。这个工作流程包含一系列相关的任务，共同形成一个流程或工作流程，当网络需要变更时将被执行。

网络自动化框架可以在无需人工干预的情况下执行一些任务：

+   问题的根本原因分析

+   检查和更新设备操作系统

+   发现节点之间的拓扑和关系

+   安全审计和合规性报告

+   根据应用程序需求从网络设备安装和撤销路由

+   管理设备配置和回滚

以下是用于自动化网络设备的一些Python库：

| **网络库** | **描述** | **链接** |
| --- | --- | --- |
| Netmiko | 一个支持SSH和Telnet的多供应商库，用于在网络设备上执行命令。支持的供应商包括Cisco、Arista、Juniper、HP、Ciena和许多其他供应商。 | [https://github.com/ktbyers/netmiko](https://github.com/ktbyers/netmiko) |
| NAPALM | 一个Python库，作为官方供应商API的包装器工作。它提供了连接到多个供应商设备并从中提取信息的抽象方法，同时以结构化格式返回输出。这可以很容易地被软件处理。 | [https://github.com/napalm-automation/napalm](https://github.com/napalm-automation/napalm) |
| PyEZ | 用于管理和自动化Juniper设备的Python库。它可以从Python客户端对设备执行CRUD操作。此外，它可以检索有关设备的信息，如管理IP、序列号和版本。返回的输出将以JSON或XML格式呈现。 | [https://github.com/Juniper/py-junos-eznc](https://github.com/Juniper/py-junos-eznc) |
| infoblox-client | 用于基于REST称为WAPI与infoblox NIOS进行交互的Python客户端。 | [https://github.com/infobloxopen/infoblox-client](https://github.com/infobloxopen/infoblox-client) |
| NX-API | 一个Cisco Nexus（仅限某些平台）系列API，通过HTTP和HTTPS公开CLI。您可以在提供的沙箱门户中输入show命令，它将被转换为对设备的API调用，并以JSON和XML格式返回输出。 | [https://developer.cisco.com/docs/nx-os/#!working-with-nx-api-cli](https://developer.cisco.com/docs/nx-os/#!working-with-nx-api-cli) |
| pyeapi | 一个Python库，作为Arista EOS eAPI的包装器，用于配置Arista EOS设备。该库支持通过HTTP和HTTPs进行eAPI调用。 | [https://github.com/arista-eosplus/pyeapi](https://github.com/arista-eosplus/pyeapi) |
| netaddr | 用于处理IPv4、IPv6和第2层地址（MAC地址）的Python库。它可以迭代、切片、排序和总结IP块。 | [https://github.com/drkjam/netaddr](https://github.com/drkjam/netaddr) |
| ciscoconfparse | 一个能够解析Cisco IOS风格配置并以结构化格式返回输出的Python库。该库还支持基于大括号分隔的配置的设备配置，如Juniper和F5。 | [https://github.com/mpenning/ciscoconfparse](https://github.com/mpenning/ciscoconfparse) |
| NSoT | 用于跟踪网络设备库存和元数据的数据库。它提供了一个基于Python Django的前端GUI。后端基于SQLite数据库存储数据。此外，它提供了使用pynsot绑定的库存的API接口。 | [https://github.com/dropbox/nsot](https://github.com/dropbox/nsot) |
| Nornir | 一个基于Python的新的自动化框架，可以直接从Python代码中使用，无需自定义**DSL**（**领域特定语言**）。Python代码称为runbook，包含一组可以针对存储在库存中的设备运行的任务（还支持Ansible库存格式）。任务可以利用其他库（如NAPALM）来获取信息或配置设备。 | [https://github.com/nornir-automation/nornir](https://github.com/nornir-automation/nornir) |

# 系统和云Python库

以下是一些可用于系统和云管理的Python软件包。像**Amazon Web Services**（**AWS**）和Google这样的公共云提供商倾向于以开放和标准的方式访问其资源，以便与组织的DevOps模型轻松集成。像持续集成、测试和部署这样的阶段需要对基础设施（虚拟化或裸金属服务器）进行*持续*访问，以完成代码生命周期。这无法手动完成，需要自动化：

| **库** | **描述** | **链接** |
| --- | --- | --- |
| ConfigParser | 用于解析和处理INI文件的Python标准库。 | [https://github.com/python/cpython/blob/master/Lib/configparser.py](https://github.com/python/cpython/blob/master/Lib/configparser.py) |
| Paramiko | Paramiko是SSHv2协议的Python（2.7、3.4+）实现，提供客户端和服务器功能。 | [https://github.com/paramiko/paramiko](https://github.com/paramiko/paramiko) |
| Pandas | 提供高性能、易于使用的数据结构和数据分析工具的库。 | [https://github.com/pandas-dev/pandas](https://github.com/pandas-dev/pandas) |
| `boto3` | 官方Python接口，用于管理不同的AWS操作，例如创建EC2实例和S3存储。 | [https://github.com/boto/boto3](https://github.com/boto/boto3) |
| `google-api-python-client` | Google Cloud Platform的官方API客户端库。 | [https://github.com/google/google-api-python-client](https://github.com/google/google-api-python-client) |
| `pyVmomi` | 来自VMWare的官方Python SDK，用于管理ESXi和vCenter。 | [https://github.com/vmware/pyvmomi](https://github.com/vmware/pyvmomi) |
| PyMYSQL | 用于与MySQL DBMS一起工作的纯Python MySQL驱动程序。 | [https://github.com/PyMySQL/PyMySQL](https://github.com/PyMySQL/PyMySQL) |
| Psycopg | 适用于Python的PostgresSQL适配器，符合DP-API 2.0标准。 | [http://initd.org/psycopg/](http://initd.org/psycopg/) |
| Django | 基于Python的高级开源Web框架。该框架遵循**MVT**（**Model, View, and Template**）架构设计，用于构建Web应用程序，无需进行Web开发和常见安全错误。 | [https://www.djangoproject.com/](https://www.djangoproject.com/) |
| Fabric | 用于在基于SSH的远程设备上执行命令和软件部署的简单Python工具。 | [https://github.com/fabric/fabric](https://github.com/fabric/fabric) |
| SCAPY | 一个出色的基于Python的数据包操作工具，能够处理各种协议，并可以使用任意组合的网络层构建数据包；它还可以将它们发送到网络上。 | [https://github.com/secdev/scapy](https://github.com/secdev/scapy) |
| Selenium | 用于自动化Web浏览器任务和Web验收测试的Python库。该库与Firefox、Chrome和Internet Explorer的Selenium Webdriver一起工作，以在Web浏览器上运行测试。 | [https://pypi.org/project/selenium/](https://pypi.org/project/selenium/) |

您可以在以下链接找到更多按不同领域分类的Python软件包：[https://github.com/vinta/awesome-python](https://github.com/vinta/awesome-python)。

# 访问模块源代码

您可以以两种方式访问您使用的任何模块的源代码。首先，转到[github.com](https://github.com/)上的`module`页面，查看所有文件、发布、提交和问题，就像下面的截图一样。我通过`netmiko`模块的维护者具有对所有共享代码的读取权限，并可以查看完整的提交列表和文件内容：

![](../images/00034.jpeg)

第二种方法是使用`pip`或PyCharm GUI将包本身安装到Python站点包目录中。`pip`实际上的操作是到GitHub下载模块内容并运行`setup.py`来安装和注册模块。你可以看到模块文件，但这次你对所有文件都有完全的读写权限，可以更改原始代码。例如，以下代码利用`netmiko`库连接到思科设备并在其上执行`show arp`命令：

```py
from netmiko import ConnectHandler

device = {"device_type": "cisco_ios",
  "ip": "10.10.88.110",
  "username": "admin",
  "password": "access123"}   net_connect = ConnectHandler(**device) output = net_connect.send_command("show arp")
```

如果我想看netmiko源代码，我可以去安装netmiko库的site-packages并列出所有文件*或*我可以在PyCharm中使用*Ctrl*和左键单击模块名称。这将在新标签中打开源代码：

![](../images/00035.jpeg)

# 可视化Python代码

你是否想知道Python自定义模块或类是如何制造的？开发人员如何编写Python代码并将其粘合在一起以创建这个漂亮而惊人的*x*模块？底层发生了什么？

文档是一个很好的开始，当然，我们都知道它通常不会随着每一步或开发人员添加的每一个细节而更新。

例如，我们都知道由Kirk Byers创建和维护的强大的netmiko库（[https://github.com/ktbyers/netmiko](https://github.com/ktbyers/netmiko)），它利用了另一个名为Paramiko的流行SSH库（[http://www.paramiko.org/](http://www.paramiko.org/)）。但我们不了解细节以及这些类之间的关系。如果你需要了解netmiko（或任何其他库）背后的魔力，以便处理请求并返回结果，请按照以下步骤（需要PyCharm专业版）。

PyCharm中的代码可视化和检查不受PyCharm社区版支持，只有专业版支持。

以下是你需要遵循的步骤：

1.  转到Python库位置文件夹中的netmiko模块源代码（通常在Windows上为`C:\Python27\Lib\site-packages`或在Linux上为`/usr/local/lib/pyhon2.7/dist-packages`）并从PyCharm中打开文件。

1.  右键单击地址栏中出现的模块名称，选择Diagram | Show Diagram。从弹出窗口中选择Python类图：

![](../images/00036.jpeg)

1.  PyCharm将开始在`netmiko`模块中的所有类和文件之间构建依赖树，然后在同一个窗口中显示它。请注意，这个过程可能需要一些时间，具体取决于你的计算机内存。此外，最好将图表保存为外部图像以查看：

![](../images/00037.jpeg)

根据生成的图表，你可以看到Netmiko支持许多供应商，如HP Comware，entrasys，Cisco ASA，Force10，Arista，Avaya等，所有这些类都继承自`netmiko.cisco_base_connection.CicsoSSHConnection`父类（我认为这是因为它们使用与思科相同的SSH风格）。这又继承自另一个名为`netmiko.cisco_base_connection.BaseConnection`的大父类。

此外，你可以看到Juniper有自己的类（`netmiko.juniper.juniper_ssh.JuniperSSH`），它直接连接到大父类。最后，我们连接到Python中所有父类的父类：`Object`类（记住最终在Python中的一切都是对象）。

你可以找到很多有趣的东西，比如*SCP传输*类和*SNMP*类，每个类都会列出用于初始化该类的方法和参数。

因此，`ConnectHandler`方法主要用于检查供应商类中的`device_type`可用性，并根据返回的数据使用相应的SSH类：

![](../images/00038.jpeg)

可视化代码的另一种方法是查看代码执行期间实际命中的模块和函数。这称为分析，它允许你在运行时检查函数。

首先，您需要像往常一样编写您的代码，然后右键单击空白处，选择“profile”而不是像平常一样运行代码：

![](../images/00039.jpeg)

等待代码执行。这次，PyCharm将检查从您的代码调用的每个文件，并为执行生成*调用图*，这样您就可以轻松地知道使用了哪些文件和函数，并计算每个文件的执行时间：

![](../images/00040.jpeg)

正如您在上一个图表中所看到的，我们在`profile_code.py`中的代码（图表底部）将调用`ConnectHandler()`函数，而后者将执行`__init__.py`，并且执行将继续。在图表的左侧，您可以看到在代码执行期间触及的所有文件。

# 摘要

在本章中，我们探讨了Python提供的一些最受欢迎的网络、系统和云包。此外，我们学习了如何访问模块源代码，并将其可视化，以更好地理解内部代码。我们查看了代码运行时的调用流程。在下一章中，我们将开始构建实验环境，并将我们的代码应用于其中。
