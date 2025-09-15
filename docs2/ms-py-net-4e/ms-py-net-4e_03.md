

# 第三章：API 和意图驱动的网络

在*第二章*，*低级网络设备交互*中，我们探讨了使用 Python 库（如 Pexpect、Paramiko、Netmiko 和 Nornir）与网络设备交互的方法。Paramiko 和类似的库使用持久会话，模拟用户在终端前输入命令。这种方法在某种程度上是可行的。向设备发送命令并捕获输出很容易。然而，当输出超过几行字符时，计算机程序解释输出就变得困难。例如，Paramiko 返回的输出是一系列字符，旨在由人类阅读。输出的结构由行和空格组成，对人类友好，但对计算机程序来说难以理解。

关键点在于：为了使我们的计算机程序自动化执行我们想要执行的大多数任务，我们需要解释返回的结果并根据返回的结果采取后续行动。当我们无法准确和可预测地解释返回的结果时，我们就无法有信心地执行下一个命令。

这是一个互联网社区普遍面临的问题，而不仅仅是网络自动化，每当计算机需要相互通信时。想象一下当计算机和人类阅读网页时的区别。人类看到的是浏览器解释的文字、图片和空间；而计算机看到的是原始 HTML 代码、Unicode 字符和二进制文件。当网站需要成为另一台计算机的 Web 服务时会发生什么？相同的 Web 资源需要同时容纳人类客户端和其他计算机程序。本质上，Web 服务器需要以最优化的方式将信息传输到另一台计算机。我们如何做到这一点？

答案是**应用程序编程接口**（**API**）。需要注意的是，API 是一个概念，而不是特定的技术或框架。根据维基百科的定义：

> 在计算机编程中，应用程序编程接口（API）是一组子程序定义、协议和工具，用于构建应用程序软件。一般而言，它是一组明确定义的软件组件之间通信的方法。一个好的 API 通过提供所有构建块，使得开发计算机程序变得更加容易，然后由程序员将这些构建块组合起来。

在我们的用例中，明确定义的通信方法将存在于我们的 Python 程序和目标设备之间。我们的网络设备提供的 API 为计算机程序（如我们的 Python 脚本）提供了一个独立的接口。确切的 API 实现是供应商和有时是产品特定的。一个供应商可能会偏好 XML，而另一个可能会使用 JSON；一个产品可能会使用 HTTPS 作为底层传输协议，而其他产品可能会提供用于设备的 Python 库，称为 SDK。在本章中，我们将看到许多不同的供应商和产品示例。

尽管存在差异，API 的概念仍然是相同的：它是一种针对其他计算机程序优化的通信方法。

在本章中，我们将探讨以下主题：

+   将**基础设施视为代码**（**IaC**）、意图驱动的网络和数据建模

+   Cisco NX-API、**应用中心基础设施**（**ACI**）和 Meraki 示例

+   Juniper **网络配置协议**（**NETCONF**）和 PyEZ

+   Arista eAPI 和 pyeapi

我们将从探讨为什么我们要将基础设施视为代码开始。

# 基础设施即代码 (IaC)

在一个完美的世界里，网络工程师和架构师在设计和管理网络时应该专注于他们希望网络实现的目标，而不是设备级别的交互。但我们都知道这个世界远非完美。许多年前，当我作为一家二线 ISP 的实习生工作时，我的第一个任务之一就是在客户现场安装一个路由器，以开通他们的分数帧中继链路（还记得那些吗？）。*我该如何做呢？* 我问。我被 handed down 一份开通帧中继链路的标准操作程序。

我去了客户现场，盲目地输入命令，看着绿色的指示灯闪烁，高兴地收拾好背包，为自己的工作感到自豪。尽管这项任务很令人兴奋，但我并没有完全理解我在做什么。我只是盲目地遵循指示，没有考虑我输入的命令的含义。如果指示灯是红色而不是绿色，我将如何进行故障排除？毫无疑问，我必须打电话给办公室，请求一位经验更丰富的工程师的帮助。

网络工程不仅仅是将命令输入到设备中；它是在尽可能减少摩擦的情况下，建立一个从一点到另一点提供服务的途径。我们必须使用的命令和必须解释的输出只是达到目的的手段。换句话说，我们应该专注于我们对网络的意图。**我们希望网络实现的目标比我们用来让设备做我们想要它做的事情的命令语法更重要**。如果我们进一步将描述我们意图的想法抽象为代码行，我们就可以潜在地描述我们的整个基础设施为特定的状态。基础设施将通过代码行来描述，并带有必要的软件或框架来强制执行该状态。

## 意图驱动的网络

自从本书第一版出版以来，随着主要网络供应商选择使用**基于意图的网络**（**IBN**）和**意图驱动的网络**（**IDN**）来描述他们的下一代设备，这两个术语的使用频率有所上升。这两个术语通常意味着相同的事情。*在我看来，IDN 是定义网络应处于的状态并拥有软件代码来强制执行该状态的想法*。例如，如果我的目标是阻止端口`80`从外部访问，这就是我应该将其声明为网络意图的方式。底层软件将负责了解配置和应用于边界路由器上的必要访问列表的语法以实现该目标。当然，IDN 是一个没有明确答案的关于确切实现的想法。我们用来强制执行我们声明的意图的软件可以是一个库、一个框架，或者我们从供应商那里购买的完整包。

当使用 API 时，我认为它使我们更接近 IDN 的状态。简而言之，因为我们抽象了在目标设备上执行特定命令的层，我们关注的是意图而不是特定命令。例如，回到我们的`block port 80`访问列表示例，我们可能在 Cisco 路由器上使用`access-list`和`access-group`，在 Juniper 路由器上使用`filter-list`。然而，通过使用 API，我们的程序可以开始询问执行者他们的意图，同时隐藏软件正在与哪种物理设备通信。我们甚至可以使用更高层次的声明性框架，例如我们在*第四章*，*Python 自动化框架*中将要介绍的 Ansible。但现在，让我们专注于网络 API。

## 屏幕抓取与 API 结构化输出

想象一个常见的场景，我们需要登录网络设备并确保设备上的所有接口都处于 up/up 状态（状态和协议都显示为`up`）。对于进入 Cisco NX-OS 设备的网络工程师来说，在终端中发出`show ip interface brief`命令就足够简单，可以很容易地从输出中看出哪个接口是 up 的：

```py
lax-edg-r1#sh ip int brief
Interface                  IP-Address      OK? Method Status                Protocol
GigabitEthernet0/0         192.168.2.51    YES NVRAM  up                    up
GigabitEthernet0/1         10.0.0.1        YES NVRAM  up                    up
Loopback0                  192.168.0.10    YES NVRAM  up 
```

行中断、空白和列标题的第一行很容易被肉眼区分。它们的存在是为了帮助我们对齐，比如，从第一行到第二行和第三行的每个接口的 IP 地址。如果我们把自己放在计算机的位置来捕获信息，所有这些空白和行中断只会让我们远离重要的输出，即：哪些接口处于 up/up 状态？为了说明这一点，我们可以看看`show interface brief`命令的 Paramiko 输出：

```py
>>> new_connection.send('show ip int brief/n') 
16
>>> output = new_connection.recv(5000)
>>> print(output)
b'show ip interface brief\r\nInterface                  IP-Address      OK? Method Status                Protocol\r\nGigabitEthernet0/0         192.168.2.51    YES NVRAM  up                    up      \r\nGigabitEthernet0/1         10.0.0.1        YES NVRAM  up                    up      \r\nLoopback0                  192.168.0.10    YES NVRAM  up                    up      \r\nlax-edg-r1#'
>>> 
```

如果我们要解析输出变量中包含的数据，我将以伪代码的形式（伪代码意味着实际代码的简化表示）这样做，以将文本减到所需的信息中：

1.  通过行中断分割每一行。

1.  我不需要包含`show ip interface brief`执行的命令的第一行，我会将其丢弃。

1.  从第二行取出直到主机名提示符的所有内容，并将其保存在一个变量中。

1.  对于其余的行，因为我们不知道有多少接口，我们将使用正则表达式语句来搜索行是否以接口名称开头，例如`lo`代表回环接口和`GigabitEthernet`代表以太网接口。

1.  我们需要将这一行分成三个部分，由空格分隔，每个部分分别包含接口名称、IP 地址以及接口状态。

1.  接口状态将通过空格进一步分割，以给我们协议、链路和管理状态。

呼吁，这只是一项大量工作，仅仅是为了人类一眼就能看出的事情！这些步骤是我们需要执行屏幕抓取非结构化文本时的步骤。这种方法有很多缺点，但我能看到的更大问题如下：

+   **可扩展性**：我们花费了大量时间在繁琐的细节上，以解析每个命令的输出。很难想象我们如何为通常运行的数百个命令做这件事。

+   **可预测性**：无法保证不同软件版本之间的输出保持一致。如果输出有细微的变化，它可能会使我们的信息收集斗争变得毫无意义。

+   **供应商和软件锁定**：一旦我们投入所有努力解析特定供应商和软件版本的输出，在这个例子中是 Cisco IOS，我们就需要为下一个我们选择的供应商重复这个过程。我不知道你怎么样，但如果我要评估一个新的供应商，如果我要再次重写所有的屏幕抓取代码，新的供应商将处于严重的入门劣势。

让我们比较一下 NX-API 调用相同`show ip interface brief`命令的输出。我们将在本章后面详细说明如何从设备获取此输出，但这里重要的是将以下输出与之前的屏幕抓取输出（完整输出在课程代码库中）进行比较：

```py
{
"ins_api":{
"outputs":{
"output":{
"body":{ "TABLE_intf":[
{
"ROW_intf":{
"admin-state":"up",
"intf-name":"Lo0",
"iod":84,
"ip-disabled":"FALSE",
"link-state":"up",
"prefix":"192.168.2.50",
"proto-state":"up"
}
},
{
"ROW_intf":{
"admin-state":"up",
"intf-name":"Eth2/1",
"iod":36,
"ip-disabled":"FALSE",
"link-state":"up",
"prefix":"10.0.0.6",
"proto-state":"up"
}
}
],
"TABLE_vrf":[
{
"ROW_vrf":{
"vrf-name-out":"default"
}
},
{
"ROW_vrf":{
"vrf-name-out":"default"
}
}
]
},
"code":"200",
"input":"show ip int brief",
"msg":"Success"
}
},
"sid":"eoc",
"type":"cli_show",
"version":"1.2"
}
} 
```

NX-API 可以返回 XML 或 JSON 格式的输出，这是 JSON 输出。我们可以立即看到输出是有结构的，可以直接映射到 Python 字典数据结构。一旦转换为 Python 字典，就不需要大量的解析——我们只需简单地选择键并检索与键关联的值。我们还可以从输出中看到，输出中有各种元数据，例如命令的成功或失败。如果命令失败，将有一条消息告诉发送者失败的原因。我们不再需要跟踪发出的命令，因为它已经通过`input`字段返回给你了。输出中还有其他有用的元数据，例如 NX-API 版本。

这种交换使得供应商和运营商的生活都变得更加容易。在供应商方面，他们可以轻松地传输配置和状态信息。当需要使用相同的数据结构暴露更多数据时，他们可以轻松地添加额外字段。在运营商方面，我们可以轻松地获取信息，并围绕它构建我们的基础设施自动化。所有人都同意，网络自动化和可编程性对网络供应商和运营商都有益。问题通常涉及自动化消息的传输、格式和结构。正如我们将在本章后面看到的那样，在 API 的伞形之下，有许多竞争技术。仅就传输语言而言，我们就有 REST API、NETCONF 和 RESTCONF 等。

## IaC 的数据建模

根据维基百科（[`en.wikipedia.org/wiki/Data_model`](https://en.wikipedia.org/wiki/Data_model)），数据模型的定义如下：

> 数据模型是一个抽象模型，它组织数据元素并标准化它们相互之间以及与现实世界实体属性的关系。例如，数据模型可能指定代表汽车的元素由多个其他元素组成，这些元素反过来又代表汽车的颜色和大小，并定义其所有者。

数据建模过程在以下图中展示：

![图示  自动生成描述](img/B18403_03_01.png)

图 3.1：数据示例过程

当将数据模型概念应用于网络时，我们可以称网络数据模型是一个抽象模型，它描述了我们的网络。如果我们仔细观察一个物理数据中心，一个二层以太网交换机可以被认为是一个包含映射到每个端口的 MAC 地址表的设备。我们的交换机数据模型描述了 MAC 地址应该如何保存在表中，包括键、附加特性（想想 VLAN 和私有 VLAN）以及更多。

同样，我们可以超越设备，在数据模型中映射整个数据中心。我们可以从接入层、分发层和核心层中的设备数量开始，它们是如何连接的，以及它们在生产环境中应该如何表现。

例如，如果我们有一个胖树网络，我们可以在模型中声明每个脊路由器有多少链路，它们应该包含多少路由，以及每个前缀会有多少下一跳。

记得我们讨论过 IaC 吗？这些特性可以以软件程序可以参考的格式映射出来，然后作为我们可以检查的理想状态。

## YANG 和 NETCONF

网络数据建模语言之一是 YANG，这是一个有趣的缩写，代表**“另一个下一代”**（尽管普遍认为，一些 IETF 工作组确实有幽默感）。它首次于 2010 年在 RFC 6020 中发布，并自此在供应商和运营商中获得了认可。

作为一种数据建模语言，YANG 用于建模设备的配置。它还可以表示由 NETCONF 协议、NETCONF 远程过程调用和 NETCONF 通知操作的状态数据。它的目标是提供协议（如 NETCONF）和底层供应商特定配置和操作语法之间的通用抽象层。我们将在本章后面查看一些 YANG 的示例。

现在我们已经讨论了基于 API 的设备管理和数据建模的高级概念，让我们来看看思科在其 API 结构中的几个示例。

# 思科 API 示例

在网络领域占据 800 磅巨猿地位的思科系统，没有错过网络自动化的趋势。在推动网络自动化的过程中，他们进行了各种内部开发、产品增强、合作以及许多外部收购。然而，由于产品线涵盖了路由器、交换机、防火墙、服务器（统一计算）、无线、协作软件和硬件以及分析软件，很难知道从哪里开始。

由于本书侧重于 Python 和网络，我们将本节中的思科示例范围限定在主要网络产品上。特别是，我们将涵盖以下内容：

+   带有 NX-API 的 Nexus

+   思科 NETCONF 和 YANG 示例

+   思科**应用中心基础设施**（**ACI**）

+   思科 Meraki 示例

对于本章中的 NX-API 和 NETCONF 示例，我们可以使用*第二章*，*低级网络设备交互*中提到的思科 DevNet 始终在线实验室设备，或者使用本地运行的思科 CML 虚拟实验室。

我们将使用与*第二章*，*低级网络设备交互*中相同的实验室拓扑，并专注于运行**NX-OSv**、**lax-cor-r1**和**nyc-cor-r1**的设备：

![图描述自动生成](img/B18403_03_02.png)

图 3.2：实验室 NX-OSv 设备

让我们先看看思科 NX-API 的示例。

## 思科 NX-API

Nexus 是思科数据中心交换机的主要产品线。NX-API（[`www.cisco.com/c/en/us/td/docs/switches/datacenter/nexus9000/sw/6-x/programmability/guide/b_Cisco_Nexus_9000_Series_NX-OS_Programmability_Guide/b_Cisco_Nexus_9000_Series_NX-OS_Programmability_Guide_chapter_011.html`](http://www.cisco.com/c/en/us/td/docs/switches/datacenter/nexus9000/sw/6-x/programmability/guide/b_Cisco_Nexus_9000_Series_NX-OS_Programmability_Guide/b_Cisco_Nexus_9000_Series_NX-OS_Programmability_Guide_chapter_011.html)）允许工程师通过 SSH、HTTP 和 HTTPS 等多种传输方式与交换机进行交互。

### 实验室准备

记得激活我们的 Python 虚拟环境。这应该是我们从现在开始进行实验室操作的一个基本要求：

```py
$ source venv/bin/activate 
```

`ncclient` ([`github.com/ncclient/ncclient`](https://github.com/ncclient/ncclient)) 库是一个用于 NETCONF 客户端的 Python 库。我们还将安装一个流行的 Python HTTP 客户端库，称为 Requests ([`pypi.org/project/requests/`](https://pypi.org/project/requests/))。我们可以通过 pip 安装这两个库：

```py
$ pip install ncclient==0.6.13
$ pip install requests==2.28.1 
```

Nexus 设备上的 NX-API 默认是关闭的，因此我们需要将其打开。我们还需要一个用户。在这种情况下，我们将仅使用现有的 `cisco` 用户：

```py
feature nxapi
username cisco password 5 $1$Nk7ZkwH0$fyiRmMMfIheqE3BqvcL0C1 role network-operator
username cisco role network-admin
username cisco passphrase lifetime 99999 warntime 14 gracetime 3 
```

对于我们的实验室，我们将打开 `nxapi` `http` 和 `nxapi` `sandbox` 配置；请注意，在生产环境中这两个都应该关闭：

```py
lax-cor-r1(config)# nxapi http port 80
lax-cor-r1(config)# nxapi sandbox 
```

我们现在准备好查看我们的第一个 NX-API 示例。

### NX-API 示例

NX-API 沙盒是玩转各种命令、数据格式，甚至直接从网页复制 Python 脚本的好方法。在上一步中，我们为了学习目的将其打开。再次强调，沙盒在生产环境中应该关闭。

使用 Nexus 设备的管理 IP 启动一个网页浏览器，查看基于我们已熟悉的 CLI 命令的各种消息格式、请求和响应：

![](img/B18403_03_03.png)

图 3.3：NX-API 开发者沙盒

在下面的示例中，我选择了 `JSON-RPC` 和 `CLI` 命令类型来执行 `show version` 命令。点击 **POST**，我们将看到 **请求** 和 **响应**：

![图形用户界面，文本，应用程序  自动生成的描述](img/B18403_03_04.png)

图 3.4：NX-API 沙盒示例输出

如果你对消息格式的可支持性不确定，或者如果你对你的代码中想要检索的值对应的响应数据字段键有疑问，沙盒会很有用。

在我们的第一个示例 `cisco_nxapi_1.py` 中，我们只是连接到 Nexus 设备并打印出首次连接时交换的能力：

```py
#!/usr/bin/env python3
from ncclient import manager
conn = manager.connect(
        host='192.168.2.50', 
        port=22, 
        username='cisco', 
        password='cisco', 
        hostkey_verify=False, 
        device_params={'name': 'nexus'}, 
        look_for_keys=False
        )
for value in conn.server_capabilities:
    print(value)
conn.close_session() 
```

在我们的示例中，我们使用 `ncclient` 库连接到设备。主机、端口、用户名和密码的连接参数是显而易见的。设备参数指定客户端连接到的设备类型。`hostkey_verify` 跳过了 SSH 的 `known_host` 要求；如果它未设置为 false，则主机指纹需要列在 `~/.ssh/known_hosts` 文件中。`look_for_keys` 选项禁用了公私钥认证，并使用用户名和密码组合进行认证。

输出将显示本版 NX-OS 支持的 XML 和 NETCONF 功能：

```py
(venv) $ python cisco_nxapi_1.py
urn:ietf:params:xml:ns:netconf:base:1.0
urn:ietf:params:netconf:base:1.0
urn:ietf:params:netconf:capability:validate:1.0
urn:ietf:params:netconf:capability:writable-running:1.0
urn:ietf:params:netconf:capability:url:1.0?scheme=file
urn:ietf:params:netconf:capability:rollback-on-error:1.0
urn:ietf:params:netconf:capability:candidate:1.0
urn:ietf:params:netconf:capability:confirmed-commit:1.0 
```

使用 ncclient 和通过 SSH 的 NETCONF 非常好，因为它使我们更接近原生实现和语法。我们将在本书的后面部分使用相同的库来比较其他供应商。对于 NX-API，我们还可以使用 HTTPS 和 JSON-RPC。在 **NX-API 开发者沙盒** 的早期截图，如果您注意到了，在 **请求** 框中有一个标记为 **Python** 的框。如果您点击它，您将能够根据 Requests 库自动转换成 Python 脚本。

对于来自 NX-API 沙盒的 `show version` 示例，以下 Python 脚本为我们自动生成。我在这里粘贴输出，没有进行任何修改：

```py
"""
NX-API-BOT 
"""
import requests
import json
"""
Modify these please
"""
url='http://YOURIP/ins'
switchuser='USERID'
switchpassword='PASSWORD'
myheaders={'content-type':'application/json-rpc'}
payload=[
  {
    "jsonrpc": "2.0",
    "method": "cli",
    "params": {
      "cmd": "show version",
      "version": 1.2
     },
     "id": 1
  }
]
response = requests.post(url,data=json.dumps(payload), headers=myheaders,auth=(switchuser,switchpassword)).json() 
```

在 `cisco_nxapi_2.py` 脚本中，您将看到我已经修改了 NX-API 沙盒生成的脚本中的 URL、用户名和密码。输出被解析，只包含软件版本。以下是输出：

```py
(venv) $ python cisco_nxapi_2.py
7.3(0)D1(1) 
```

使用这种方法最好的部分是，相同的总体语法结构适用于配置和 `show` 命令。这可以在 `cisco_nxapi_3.py` 文件中看到，该文件使用新的主机名配置设备。在命令执行后，您将看到设备主机名从 `lax-cor-r1` 更改为 `lax-cor-r1-new`：

```py
lax-cor-r1-new# sh run | i hostname
hostname lax-cor-r1-new 
```

对于多行配置，您可以使用 ID 字段来指定操作的顺序。这可以在 `cisco_nxapi_4.py` 中看到。以下有效载荷被列出，用于在接口配置模式下更改以太网 2/12 接口的描述：

```py
{
  "jsonrpc": "2.0",
  "method": "cli",
  "params": {
    "cmd": "interface ethernet 2/12",
    "version": 1.2
  },
  "id": 1
},
{
  "jsonrpc": "2.0",
  "method": "cli",
  "params": {
    "cmd": "description foo-bar",
    "version": 1.2
  },
  "id": 2
},
{
  "jsonrpc": "2.0",
  "method": "cli",
  "params": {
    "cmd": "end",
    "version": 1.2
  },
  "id": 3
},
{
  "jsonrpc": "2.0",
  "method": "cli",
  "params": {
    "cmd": "copy run start",
    "version": 1.2
  },
  "id": 4
  }
] 
```

我们可以通过查看 Nexus 设备的运行配置来验证先前配置脚本的结果：

```py
interface Ethernet2/12
  description foo-bar
  shutdown
  no switchport
  mac-address 0000.0000.002f 
```

在下一个示例中，我们将看到如何使用 YANG 与 NETCONF。

## Cisco YANG 模型

让我们通过一个示例来看看 Cisco 的 YANG 模型支持。首先，我们应该知道 YANG 模型只定义了通过 NETCONF 协议发送的架构类型，而没有规定数据应该是什么。其次，值得注意的是，NETCONF 作为独立协议存在，正如我们在 NX-API 部分所看到的。第三，YANG 在不同供应商和产品线上的支持性不同。例如，如果我们为运行 IOS-XE 的 Cisco CSR 1000v 运行能力交换脚本，我们可以在平台上看到支持的 YANG 模型：

```py
urn:cisco:params:xml:ns:yang:cisco-virtual-service?module=cisco- virtual-service&revision=2015-04-09
http://tail-f.com/ns/mibs/SNMP-NOTIFICATION-MIB/200210140000Z? module=SNMP-NOTIFICATION-MIB&revision=2002-10-14
urn:ietf:params:xml:ns:yang:iana-crypt-hash?module=iana-crypt-hash&revision=2014-04-04&features=crypt-hash-sha-512,crypt-hash-sha-256,crypt-hash-md5
urn:ietf:params:xml:ns:yang:smiv2:TUNNEL-MIB?module=TUNNEL-MIB&revision=2005-05-16
urn:ietf:params:xml:ns:yang:smiv2:CISCO-IP-URPF-MIB?module=CISCO-IP-URPF-MIB&revision=2011-12-29
urn:ietf:params:xml:ns:yang:smiv2:ENTITY-STATE-MIB?module=ENTITY-STATE-MIB&revision=2005-11-22
urn:ietf:params:xml:ns:yang:smiv2:IANAifType-MIB?module=IANAifType-MIB&revision=2006-03-31
<omitted> 
```

YANG 在不同供应商和产品线上的支持性有些不均匀。我在本书的代码库中包含了一个 `cisco_yang_1.py` 脚本，展示了如何使用 Cisco Devnet 提供的 Cisco IOS-XE 总是开启沙盒，通过 YANG 过滤器 `urn:ietf:params:xml:ns:yang:ietf-interfaces` 解析出 NETCONF XML 输出。

我们可以在 YANG GitHub 项目页面上看到最新的供应商支持（[`github.com/YangModels/yang/tree/master/vendor`](https://github.com/YangModels/yang/tree/master/vendor)）。

## Cisco ACI 示例

Cisco 应用中心基础设施，或称 ACI，旨在为其管理范围内定义的所有网络组件提供一个集中式控制器方法。在数据中心环境中，集中式控制器了解并管理脊线、叶子和机架顶部交换机，以及所有网络服务功能。这可以通过图形用户界面（GUI）、命令行界面（CLI）或 API 来实现。有些人可能会认为 ACI 是思科对基于控制器、软件定义网络更广泛解决方案的回应。

ACI API 遵循 REST 模型，使用 HTTP 动词（`GET`、`POST` 和 `DELETE`）来指定预期操作。在我们的示例中，我们可以使用 Cisco DevNet 永久在线实验室 ACI 设备([`devnetsandbox.cisco.com/RM/Topology`](https://devnetsandbox.cisco.com/RM/Topology))：

![包含图形用户界面的图片 描述自动生成](img/B18403_03_05.png)

图 3.5：Cisco DevNet 沙盒

始终检查最新的 Cisco DevNet 页面以获取最新的设备信息、用户名和密码，因为自本书编写以来它们可能已更改。

控制器是网络的“大脑”，它对所有网络设备保持可见性：

![图表 描述自动生成](img/B18403_03_06.png)

图 3.6：Cisco ACI 控制器

我们可以使用网页浏览器登录到控制器并查看不同的租户：

![图形用户界面 描述自动生成](img/B18403_03_07.png)

图 3.7：Cisco ACI 租户

让我们使用 Python 交互式提示符来看看我们如何与 ACI 控制器交互。我们将首先导入正确的库，并定义目标 URL 以及登录凭证：

```py
>>> import requests, json
>>> URL = 'https://sandboxapicdc.cisco.com'
>>> PASSWORD = "<password>"
>>> LOGIN = "admin"
>>> AUTH_URL = URL + '/api/aaaLogin.json' 
```

然后，我们可以发出请求并将响应转换为 JSON 格式：

```py
>>> r = requests.post(AUTH_URL, json={"aaaUser":{"attributes":{"name":LOGIN,"pwd":PASSWORD}}}, verify=False)
>>> r_json = r.json()
>>> r_json
{'totalCount': '1', 'imdata': [{'aaaLogin': {'attributes': {'token': _<skip>} 
```

我们可以从响应中获取令牌并将其用作未来请求控制器的身份验证 cookie。在下面的示例中，我们查询控制器租户部分中看到的 `cisco` 租户：

```py
>>> token = r_json["imdata"][0]["aaaLogin"]["attributes"]["token"]
>>> cookie = {'APIC-cookie':token}
>>> QUERY_URL = URL + '/api/node/class/fvTenant.json?query-target-filter=eq(fvTenant.name,"Cisco")'
>>> r_cisco = requests.get(QUERY_URL, cookies=cookie, verify=False)
>>> r_cisco.json()
{'totalCount': '1', 'imdata': [{'fvTenant': {'attributes': {'annotation': '', 'childAction': '', 'descr': '', 'dn': 'uni/tn-Cisco', 'extMngdBy': '', 'lcOwn': 'local', 'modTs': '2022-08-06T14:05:15.893+00:00', 'monPolDn': 'uni/tn-common/monepg-default', 'name': 'Cisco', 'nameAlias': '', 'ownerKey': '', 'ownerTag': '', 'status': '', 'uid': '15374', 'userdom': ':all:'}}}]}
>>> print(r_cisco.json()['imdata'][0]['fvTenant']['attributes']['dn'])
uni/tn-Cisco 
```

如您所见，我们只查询单个控制器设备，但我们可以获取控制器所了解的所有网络设备的高级视图。这相当不错！当然，缺点是 ACI 控制器目前只支持思科设备。

**Cisco IOS-XE**

在很大程度上，Cisco IOS-XE 脚本与为 NX-OS 编写的脚本在功能上相似。IOS-XE 具有额外的功能，可以提升 Python 网络编程能力，例如机箱内 Python 和虚拟机壳，[`developer.cisco.com/docs/ios-xe/#!on-box-python-and-guestshell-quick-start-guide/onbox-python`](https://developer.cisco.com/docs/ios-xe/#!on-box-python-and-guestshell-quick-start-guide/onbox-python)。

与 ACI 类似，Cisco Meraki 是一个集中式管理的控制器，它对多个有线和无线网络具有可见性。与 ACI 控制器不同，Meraki 是基于云的，因此它托管在本地场所之外。让我们在下一节中查看一些 Cisco Meraki 功能和示例。

# Cisco Meraki 控制器

Cisco Meraki 是一个基于云的集中式控制器，简化了设备的 IT 管理。其方法与 ACI 非常相似，唯一的区别是控制器有一个基于云的公共 URL。用户通常通过 GUI 接收 API 密钥，然后可以在 Python 脚本中使用它来检索组织 ID：

```py
#!/usr/bin/env python3
import requests
import pprint
myheaders={'X-Cisco-Meraki-API-Key': <skip>}
url ='https://dashboard.meraki.com/api/v0/organizations'
response = requests.get(url, headers=myheaders, verify=False)
pprint.pprint(response.json()) 
```

让我们执行脚本 `cisco_meraki_1.py`，这是一个简单的请求，针对的是由 Cisco DevNet 提供的始终开启的 Meraki 控制器：

```py
(venv) $ python cisco_meraki_1.py
 [{'id': '681155',
  'name': 'DeLab',
  'url': 'https://n6.meraki.com/o/49Gm_c/manage/organization/overview'},
 {'id': '865776',
  'name': 'Cisco Live US 2019',
  'url': 'https://n22.meraki.com/o/CVQqTb/manage/organization/overview'},
 {'id': '549236',
  'name': 'DevNet Sandbox',
  'url': 'https://n149.meraki.com/o/t35Mb/manage/organization/overview'},
 {'id': '52636',
  'name': 'Forest City - Other',
  'url': 'https://n42.meraki.com/o/E_utnd/manage/organization/overview'}] 
```

从那里，可以使用组织 ID 进一步检索信息，例如库存、网络信息等：

```py
#!/usr/bin/env python3
import requests
import pprint
myheaders={'X-Cisco-Meraki-API-Key': <skip>}
orgId = '549236'
url = 'https://dashboard.meraki.com/api/v0/organizations/' + orgId + '/networks'
response = requests.get(url, headers=myheaders, verify=False)
pprint.pprint(response.json()) 
```

让我们看看 `cisco_meraki_2.py` 脚本的输出：

```py
(venv) $ python cisco_meraki_2.py
<skip>
[{'disableMyMerakiCom': False,
  'disableRemoteStatusPage': True,
  'id': 'L_646829496481099586',
  'name': 'DevNet Always On Read Only',
  'organizationId': '549236',
  'productTypes': ['appliance', 'switch'],
  'tags': ' Sandbox ',
  'timeZone': 'America/Los_Angeles',
  'type': 'combined'},
 {'disableMyMerakiCom': False,
  'disableRemoteStatusPage': True,
  'id': 'N_646829496481152899',
  'name': 'test - mx65',
  'organizationId': '549236',
  'productTypes': ['appliance'],
  'tags': None,
  'timeZone': 'America/Los_Angeles',
  'type': 'appliance'},
<skip> 
```

我们已经看到了使用 NX-API、ACI 和 Meraki 控制器的 Cisco 设备的示例。在下一节中，让我们看看一些与 Juniper Networks 设备一起工作的 Python 示例。

# Juniper Networks 的 Python API

Juniper Networks 一直受到服务提供商群体的喜爱。如果我们退后一步，看看服务提供商行业，那么自动化网络设备成为他们需求列表中的首要任务是有道理的。在云规模数据中心出现之前，服务提供商需要管理的网络设备最多。例如，一个典型的企业网络可能在总部有少量冗余互联网连接，以及一些通过使用私有 **多协议标签交换**（**MPLS**）网络连接回总部的星型远程站点。但对于服务提供商来说，他们需要构建、配置、管理和排除 MPLS 连接及其底层网络的故障。他们通过销售带宽以及增值管理服务来赚钱。对于服务提供商来说，投资自动化以使用最少的工程小时数来保持网络正常运行是有意义的。在他们的情况下，网络自动化是他们竞争优势的关键。

在我看来，服务提供商的网络需求与云数据中心之间的区别在于，传统上，服务提供商将更多服务聚合到单个设备中。一个很好的例子是 MPLS，几乎所有主要服务提供商都提供但很少在企业或数据中心网络中使用。Juniper Networks 已经识别了网络可编程性的需求，并在满足服务提供商自动化需求方面表现出色。让我们看看 Juniper 的自动化 API 的一些示例。

## Juniper 和 NETCONF

NETCONF 是一个由 IETF 发布的标准，首次于 2006 年以 `RFC 4741` 的形式发布，后来在 `RFC 6241` 中进行了修订。Juniper Networks 对这两个 RFC 标准都做出了重大贡献。实际上，Juniper 是 `RFC 4741` 的唯一作者。Juniper 设备完全支持 NETCONF 是合情合理的，它还作为其大多数自动化工具和框架的底层层。NETCONF 的一些主要特性包括以下内容：

1.  它使用 **可扩展标记语言**（**XML**）进行数据编码。

1.  它使用 **远程过程调用**（**RPC**）。因此，如果 HTTP(s) 是传输协议，URL 端点是相同的，而操作意图则在请求体中指定。

1.  它从上到下基于层结构。层包括内容、操作、消息和传输：

![图描述自动生成，置信度中等](img/B18403_03_08.png)

图 3.8：NETCONF 模型

Juniper Networks 在其技术库中提供了一个广泛的 NETCONF XML 管理协议开发者指南 ([`www.juniper.net/techpubs/en_US/junos13.2/information-products/pathway-pages/netconf-guide/netconf.html#overview`](https://www.juniper.net/techpubs/en_US/junos13.2/information-products/pathway-pages/netconf-guide/netconf.html#overview))。让我们看看它的用法。

### 设备准备

要开始使用 NETCONF，让我们创建一个单独的用户以及开启所需的服务：

```py
set system login user juniper uid 2001
set system login user juniper class super-user
set system login user juniper authentication encrypted-password "$1$0EkA.XVf$cm80A0GC2dgSWJIYWv7Pt1" 
set system services ssh
set system services telnet
set system services netconf ssh port 830 
```

对于 Juniper 设备实验室，我正在使用一个较旧的、不受支持的平台，称为 **JunOS Olive**。它仅用于实验室目的。你可以使用你喜欢的搜索引擎查找一些关于 Juniper Olive 的有趣事实和历史。

在 Juniper 设备上，你总是可以在平面文件或 XML 格式下查看配置。当需要指定一行命令来更改配置时，`flat` 文件会很有用：

```py
netconf@foo> show configuration | display set 
set version 12.1R1.9
set system host-name foo set system domain-name bar
<omitted> 
```

当你需要查看配置的 XML 结构时，XML 格式会很有用：

```py
netconf@foo> show configuration | display xml
<rpc-reply >
<configuration junos:commit-seconds="1485561328" junos:commit- localtime="2017-01-27 23:55:28 UTC" junos:commit-user="netconf">
<version>12.1R1.9</version>
<system>
<host-name>foo</host-name>
<domain-name>bar</domain-name> 
```

我们在 *Cisco NX-API 中的实验室软件安装和设备准备* 部分安装了必要的 Linux 库和 `ncclient` Python 库。如果你还没有这样做，请回到该部分并安装必要的包。

我们现在可以查看我们的第一个 Juniper NETCONF 示例。

### Juniper NETCONF 示例

我们将使用一个非常直接的例子来执行 `show version`。我们将把这个文件命名为 `junos_netconf_1.py`：

```py
#!/usr/bin/env python3
from ncclient import manager
conn = manager.connect(
    host='192.168.2.70', 
    port='830', 
    username='juniper', 
    password='juniper!',
    timeout=10,
    device_params={'name':'junos'},
    hostkey_verify=False)
result = conn.command('show version', format='text')
print(result.xpath('output')[0].text)
conn.close_session() 
```

脚本中的所有字段都应该相当直观，除了 `device_params`。从 ncclient 0.4.1 版本开始，设备处理器被添加来指定不同的供应商或平台。例如，名称可以是 Juniper、CSR、Nexus 或华为。我们还添加了 `hostkey_verify=False`，因为我们正在使用 Juniper 设备的自签名证书。

返回的输出是使用 XML 编码的 `rpc-reply`，其中包含输出元素：

```py
<rpc-reply message-id="urn:uuid:7d9280eb-1384-45fe-be48- b7cd14ccf2b7">
<output>
Hostname: foo
Model: olive
JUNOS Base OS boot [12.1R1.9]
JUNOS Base OS Software Suite [12.1R1.9]
<omitted>
JUNOS Runtime Software Suite [12.1R1.9] JUNOS Routing Software Suite [12.1R1.9]
</output>
</rpc-reply> 
```

我们可以解析 XML 输出，只包括输出文本：

```py
print(result.xpath('output')[0].text) 
```

在 `junos_netconf_2.py` 中，我们将对设备进行配置更改。我们将从一些新的导入开始，用于构建新的 XML 元素和连接管理器对象：

```py
#!/usr/bin/env python3
from ncclient import manager
from ncclient.xml_ import new_ele, sub_ele
conn = manager.connect(host='192.168.2.70', port='830', username='juniper', password='juniper!', timeout=10, device_params={'name':'junos'}, hostkey_verify=False) 
```

我们将锁定配置并做出配置更改：

```py
# lock configuration and make configuration changes conn.lock()
# build configuration
config = new_ele('system')
sub_ele(config, 'host-name').text = 'master'
sub_ele(config, 'domain-name').text = 'python' 
```

在构建配置部分，我们创建了一个新的 `system` 元素，并包含 `host-name` 和 `domain-name` 子元素。如果您想知道层次结构，您可以从 XML 显示中看到 `system` 节点是 `host-name` 和 `domain-name` 的父节点：

```py
<system>
    <host-name>foo</host-name>
    <domain-name>bar</domain-name>
...
</system> 
```

配置构建完成后，脚本将推送配置并提交配置更改。这些是 Juniper 配置更改的正常最佳实践步骤（`lock`、`configure`、`unlock`、`commit`）。

```py
# send, validate, and commit config conn.load_configuration(config=config)
conn.validate()
commit_config = conn.commit()
print(commit_config.tostring)
# unlock config
conn.unlock()
# close session
conn.close_session() 
```

总体来说，NETCONF 步骤与我们在 CLI 步骤中会执行的操作非常匹配。请查看 `junos_netconf_3.py` 脚本以获取更可重用的代码。以下示例结合了逐步示例和一些 Python 函数：

```py
# make a connection object
def connect(host, port, user, password):
    connection = manager.connect(host=host, port=port,
        username=user, password=password, timeout=10,
        device_params={'name':'junos'}, hostkey_verify=False)
    return connection
# execute show commands
def show_cmds(conn, cmd):
    result = conn.command(cmd, format='text')
    return result
# push out configuration
def config_cmds(conn, config):
    conn.lock()
    conn.load_configuration(config=config)
    commit_config = conn.commit()
    return commit_config.tostring 
```

此文件可以独立执行，也可以导入到其他 Python 脚本中使用。

Juniper 还提供了一个名为 PyEZ 的 Python 库，用于与其设备一起使用。我们将在下一节中查看一些使用该库的示例。

## Juniper PyEZ 开发者指南

**PyEZ** 是一个高级 Python 库实现，与现有的 Python 代码集成得更好。通过利用围绕底层配置的 Python API，您可以在不深入了解 Junos CLI 的情况下执行常见操作和配置任务。

Juniper 在其技术库中维护了一个全面的 Junos PyEZ 开发者指南，网址为 [`www.juniper.net/techpubs/en_US/junos-pyez1.0/information-products/pathway-pages/junos-pyez-developer-guide.html#configuration`](https://www.juniper.net/techpubs/en_US/junos-pyez1.0/information-products/pathway-pages/junos-pyez-developer-guide.html#configuration)。如果您对使用 PyEZ 感兴趣，我强烈建议至少浏览一下指南中的各个主题。

### 安装和准备

每个操作系统的安装说明可以在 *安装 Junos PyEZ* ([`www.juniper.net/techpubs/en_US/junos-pyez1.0/topics/task/installation/junos-pyez-server-installing.html`](https://www.juniper.net/techpubs/en_US/junos-pyez1.0/topics/task/installation/junos-pyez-server-installing.html)) 页面上找到。

`PyEZ` 软件包可以通过 `pip` 进行安装：

```py
(venv) $ pip install junos-eznc 
```

在 Juniper 设备上，需要将 NETCONF 配置为 PyEZ 的底层 XML API：

```py
set system services netconf ssh port 830 
```

对于用户认证，我们可以使用密码认证或 SSH 密钥对。我们可以创建新用户或使用现有用户。对于 `ssh` 密钥认证，首先，如果你的管理主机尚未为 *第二章*，*低级网络设备交互* 生成密钥对，请先生成密钥对：

```py
$ ssh-keygen -t rsa 
```

默认情况下，公钥将在 `~/.ssh/` 下被命名为 `id_rsa.pub`，而私钥将在同一目录下命名为 `id_rsa`。将私钥视为你永远不会分享的密码。公钥可以自由分发。在我们的用例中，我们将公钥复制到 `/tmp` 目录，并启用 Python 3 HTTP 服务器模块以创建一个可访问的 URL：

```py
(venv) $ cp ~/.ssh/id_rsa.pub /tmp
(venv) $ cd /tmp
(venv) $ python3 -m http.server
(venv) Serving HTTP on 0.0.0.0 port 8000 ... 
```

从 Juniper 设备，我们可以创建用户并关联公钥，通过从 Python 3 网络服务器下载公钥：

```py
netconf@foo# set system login user echou class super-user authentication load-key-file http://<management host ip>:8000/id_rsa.pub
/var/home/netconf/...transferring.file........100% of 394 B 2482 kBps 
```

现在，如果我们尝试从管理站使用私钥 `ssh`，用户将被自动认证：

```py
(venv) $ ssh -i ~/.ssh/id_rsa <Juniper device ip>
--- JUNOS 12.1R1.9 built 2012-03-24 12:52:33 UTC
echou@foo> 
```

让我们确保两种认证方法都能与 PyEZ 一起工作。让我们尝试用户名和密码组合：

```py
>>> from jnpr.junos import Device
>>> dev = Device(host='<Juniper device ip, in our case 192.168.2.70>', user='juniper', password='juniper!')
>>> dev.open()
Device(192.168.2.70)
>>> dev.facts
{'serialnumber': '', 'personality': 'UNKNOWN', 'model': 'olive', 'ifd_style': 'CLASSIC', '2RE': False, 'HOME': '/var/home/juniper', 'version_info': junos.version_info(major=(12, 1), type=R, minor=1, build=9), 'switch_style': 'NONE', 'fqdn': 'foo.bar', 'hostname': 'foo', 'version': '12.1R1.9', 'domain': 'bar', 'vc_capable': False}
>>> dev.close() 
```

我们还可以尝试使用 SSH 密钥认证：

```py
>>> from jnpr.junos import Device
>>> dev1 = Device(host='192.168.2.70', user='echou', ssh_private_key_file='/home/echou/.ssh/id_rsa')
>>> dev1.open()
Device(192.168.2.70)
>>> dev1.facts
{'HOME': '/var/home/echou', 'model': 'olive', 'hostname': 'foo', 'switch_style': 'NONE', 'personality': 'UNKNOWN', '2RE': False, 'domain': 'bar', 'vc_capable': False, 'version': '12.1R1.9', 'serialnumber': '', 'fqdn': 'foo.bar', 'ifd_style': 'CLASSIC', 'version_info': junos.version_info(major=(12, 1), type=R, minor=1, build=9)}
>>> dev1.close() 
```

太好了！我们现在可以看看 PyEZ 的几个示例。

### PyEZ 示例

在之前的交互式提示中，我们看到当设备连接时，对象会自动检索一些关于设备的事实。在我们的第一个例子 `junos_pyez_1.py` 中，我们正在连接到设备并执行 `show interface em1` 的 RPC 调用：

```py
#!/usr/bin/env python3
from jnpr.junos import Device
import xml.etree.ElementTree as ET
import pprint
dev = Device(host='192.168.2.70', user='juniper', passwd='juniper!')
try:
    dev.open()
except Exception as err:
    print(err)
    sys.exit(1)
result = dev.rpc.get_interface_information(interface_name='em1', terse=True)
pprint.pprint(ET.tostring(result))
dev.close() 
```

`Device` 类有一个包含所有操作命令的 `rpc` 属性。这很棒，因为我们可以在 CLI 和 API 之间没有滑动。但是，我们需要找到 CLI 命令对应的 `xml rpc` 元素标签。在我们的第一个例子中，我们如何知道 `show interface em1` 等同于 `get_interface_information`？我们有三种方法来找出这些信息：

1.  我们可以参考 *Junos XML API 操作开发者参考*。

1.  我们可以使用 CLI 并显示 XML RPC 等价物，并将单词之间的破折号（-）替换为下划线（_）。

1.  我们也可以通过使用 PyEZ 库来编程实现。

我通常使用第二种方法直接获取输出：

```py
netconf@foo> show interfaces em1 | display xml rpc
<rpc-reply >
 <rpc>
 <get-interface-information>
 <interface-name>em1</interface-name>
 </get-interface-information>
 </rpc>
 <cli>
 <banner></banner>
 </cli>
</rpc-reply> 
```

这里是使用 PyEZ 编程的示例（第三种方法）：

```py
>>> dev1.display_xml_rpc('show interfaces em1', format='text') 
'<get-interface-information>/n <interface-name>em1</interface- name>/n</get-interface-information>/n' 
```

当然，我们也可以进行配置更改。在 `junos_pyez_2.py` 配置示例中，我们将从 PyEZ 导入一个额外的 `Config()` 方法：

```py
#!/usr/bin/env python3
from jnpr.junos import Device
from jnpr.junos.utils.config import Config 
```

我们将使用相同的块来连接到设备：

```py
dev = Device(host='192.168.2.70', user='juniper',
    passwd='juniper!')
try:
    dev.open()
except Exception as err:
    print(err)
    sys.exit(1) 
```

`new Config()` 方法将加载 XML 数据并执行配置更改：

```py
config_change = ""
<system>
  <host-name>master</host-name>
  <domain-name>python</domain-name>
</system>
""
cu = Config(dev)
cu.lock()
cu.load(config_change)
cu.commit()
cu.unlock()
dev.close() 
```

PyEZ 示例设计得很简单。希望它们能展示你如何利用 PyEZ 满足你的 Junos 自动化需求。在下面的例子中，让我们看看我们如何使用 Python 库与 Arista 网络设备一起工作。

# Arista Python API

**Arista Networks** 一直专注于大规模数据中心网络。在其公司简介页面上（[`www.arista.com/en/company/company-overview`](https://www.arista.com/en/company/company-overview)），它声明如下：

> “Arista Networks 是大型数据中心、校园和路由环境的数据驱动、客户端到云网络行业的领导者。”

注意，该声明特别提到了**大型数据中心**，我们知道那里服务器、数据库和网络设备都在爆炸式增长。自动化一直是 Arista 的领先特性之一是有道理的。事实上，它的操作系统背后有 Linux 的支持，这带来了许多附加的好处，例如 Linux 命令和平台上的内置 Python 解释器。从第一天起，Arista 就公开了将 Linux 和 Python 功能暴露给网络操作员。

与其他供应商一样，您可以直接通过 eAPI 与 Arista 设备交互，或者您可以选择利用他们的 Python 库。在本章中，我们将看到这两个示例。

## Arista eAPI 管理

几年前，Arista 的 eAPI 首次在 EOS 4.12 中引入。它通过 HTTP 或 HTTPS 传输一组显示或配置命令，并以 JSON 格式响应。一个重要的区别是，它是一个 RPC 和**JSON-RPC**，而不是在 HTTP 或 HTTPS 上提供的纯 RESTful API。区别在于我们使用相同的 HTTP 方法（`POST`）向相同的 URL 端点发出请求。但与使用 HTTP 动词（`GET`、`POST`、`PUT`、`DELETE`）来表示我们的操作不同，我们只需在请求体中声明我们的预期操作。在 eAPI 的情况下，我们将指定一个具有`runCmds`值的`method`键。

在以下示例中，我使用的是运行 EOS 4.16 的物理 Arista 交换机。

### eAPI 准备

Arista 设备上的 eAPI 代理默认是禁用的，因此在使用它之前，我们需要在设备上启用它：

```py
arista1(config)#management api http-commands
arista1(config-mgmt-api-http-cmds)#no shut
arista1(config-mgmt-api-http-cmds)#protocol https port 443 
arista1(config-mgmt-api-http-cmds)#no protocol http
arista1(config-mgmt-api-http-cmds)#vrf management 
```

如您所见，我们已经关闭了 HTTP 服务器，并使用 HTTPS 作为唯一的传输方式。默认情况下，管理接口位于名为**management**的 VRF 中。在我的拓扑中，我通过管理接口访问设备；因此，我为 eAPI 管理指定了 VRF。

您可以通过`show management api http-commands`命令检查 API 管理状态：

```py
arista1#sh management
api http-commands Enabled: Yes
HTTPS server: running, set to use port 443 HTTP server: shutdown, set to use port 80
Local HTTP server: shutdown, no authentication, set to use port 8080 
Unix Socket server: shutdown, no authentication
VRF: management 
Hits: 64
Last hit: 33 seconds ago Bytes in: 8250
Bytes out: 29862
Requests: 23
Commands: 42
Duration: 7.086 
seconds SSL Profile: none
QoS DSCP: 0
User Requests Bytes in Bytes out Last hit
----------- -------------- -------------- --------------- -----------
admin 23 8250 29862 33 seconds ago
URLs
-----------------------------------------
Management1 : https://192.168.199.158:443
arista1# 
```

启用代理后，我们可以在网页浏览器中通过访问设备的 IP 地址来访问 eAPI 的探索页面。如果您已更改了默认的访问端口，只需将其附加即可。认证与交换机上设置的认证方式相关联。我们将使用设备上配置的本地用户名和密码。默认情况下，将使用自签名证书：

![图形用户界面，文本，应用程序  自动生成的描述](img/B18403_03_09.png)

图 3.9：Arista EOS 探索

我们将被带到探索页面，在那里我们可以输入 CLI 命令，并获得我们请求主体的良好输出。例如，如果我想查看如何为`show version`创建请求主体，这就是我在探索器中看到的输出：

![图形用户界面，文本，应用程序  自动生成的描述](img/B18403_03_10.png)

图 3.10：Arista EOS 探索查看器

概览链接将带我们到示例使用和背景信息，而命令文档将作为 show 命令的参考点。每个命令参考将包含返回值字段名、类型和简要描述。Arista 的在线参考脚本使用`jsonrpclib`([`github.com/joshmarshall/jsonrpclib/`](https://github.com/joshmarshall/jsonrpclib/))，我们将使用它。

本节中的示例主要使用 Python 2.7，因为 jsonrpclib 长时间未移植到 Python 3。然而，根据 GitHub 拉取请求[`github.com/joshmarshall/jsonrpclib/issues/38`](https://github.com/joshmarshall/jsonrpclib/issues/38)，Python 3 支持应该被包含。

使用`pip`进行安装非常直接：

```py
(venv) $ pip install jsonrpclib 
```

### eAPI 示例

然后，我们可以编写一个简单的程序，名为`eapi_1.py`，来查看响应文本：

```py
#!/usr/bin/python2
from __future__ import print_function
from jsonrpclib import Server
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
switch = Server("https://admin:arista@192.168.199.158/command-api")
response = switch.runCmds( 1, [ "show version" ] )
print('Serial Number: ' + response[0]['serialNumber']) 
```

这是之前`runCmds()`方法返回的响应：

```py
[{u'memTotal': 3978148, u'internalVersion': u'4.16.6M- 3205780.4166M', u'serialNumber': u'<omitted>', u'systemMacAddress': u'<omitted>', u'bootupTimestamp': 1465964219.71, u'memFree': 277832, u'version': u'4.16.6M', u'modelName': u'DCS-7050QX-32-F', u'isIntlVersion': False, u'internalBuildId': u'373dbd3c-60a7-4736-8d9e-bf5e7d207689', u'hardwareRevision': u'00.00', u'architecture': u'i386'}] 
```

如您所见，结果是包含一个字典项的列表。如果我们需要获取序列号，我们可以简单地引用项号和键：

```py
print('Serial Number: ' + response[0]['serialNumber']) 
```

输出将只包含序列号：

```py
$ python eapi_1.py 
Serial Number: <omitted> 
```

要更熟悉命令参考，我建议点击 eAPI 页面上的**命令文档**链接，并将我们的输出与文档中**show version**的输出进行比较。

如前所述，与 REST 不同，JSON-RPC 客户端使用相同的 URL 端点来调用服务器资源。我们可以从之前的示例中看到`runCmds()`方法包含一个命令列表。对于配置命令的执行，你可以遵循相同的步骤，并通过命令列表配置设备。

这里是一个名为`eapi_2.py`的文件中的配置命令示例。在我们的例子中，我们编写了一个函数，该函数将`switch`对象和命令列表作为属性：

```py
#!/usr/bin/python2
from __future__ import print_function
from jsonrpclib import Server
import ssl, pprint
ssl._create_default_https_context = ssl._create_unverified_context
# Run Arista commands thru eAPI
def runAristaCommands(switch_object, list_of_commands):
    response = switch_object.runCmds(1, list_of_commands)
    return response
switch = Server("https://admin:arista@192.168.199.158/command-api") 
commands = ["enable", "configure", "interface ethernet 1/3", "switchport access vlan 100", "end", "write memory"]
response = runAristaCommands(switch, commands)
pprint.pprint(response) 
```

这是命令执行输出的结果：

```py
$ python2 eapi_2.py
[{}, {}, {}, {}, {}, {u'messages': [u'Copy completed successfully.']}] 
```

现在，快速检查`switch`以验证命令的执行情况：

```py
arista1#sh run int eth 1/3 
interface Ethernet1/3
    switchport access vlan 100 
arista1# 
```

总体来说，eAPI 相当直观且易于使用。大多数编程语言都有类似于`jsonrpclib`的库，这些库抽象了 JSON-RPC 的内部机制。只需几个命令，你就可以开始将 Arista EOS 自动化集成到你的网络中。

## Arista Pyeapi 库

Python 客户端库 Pyeapi([`pyeapi.readthedocs.io/en/master/index.html`](http://pyeapi.readthedocs.io/en/master/index.html))是围绕 eAPI 的本地 Python 库包装器。它提供了一套绑定来配置 Arista EOS 节点。为什么我们需要 Pyeapi，因为我们已经有了 eAPI？答案是“这取决于。”在 Pyeapi 和 eAPI 之间进行选择主要是一个判断决定。

如果你在一个非 Python 环境中，eAPI 可能是更好的选择。从我们的示例中，你可以看到 eAPI 的唯一要求是一个支持 JSON-RPC 的客户端。因此，它与大多数编程语言兼容。当我最初进入这个领域时，Perl 是脚本和网络自动化的主导语言。仍然有许多企业依赖于 Perl 脚本作为它们的主要自动化工具。如果你处于公司已经投入了大量资源且代码库不是 Python 语言的情况，使用带有 JSON-RPC 的 eAPI 将是一个不错的选择。

然而，对于我们这些喜欢用 Python 编程的人来说，像 Pyeapi 这样的本地 Python 库意味着在编写代码时感觉更自然。它确实使得将 Python 程序扩展以支持 EOS 节点变得更加容易。它也使得跟上 Python 的最新变化变得更加容易。例如，我们可以使用 Python 3 与 Pyeapi 一起使用（[`pyeapi.readthedocs.io/en/master/requirements.html`](https://pyeapi.readthedocs.io/en/master/requirements.html)）！

### Pyeapi 安装

使用`pip`安装非常简单：

```py
(venv) $ pip install pyeapi 
```

注意，pip 也会安装`netaddr`库，因为它是 Pyeapi 的指定要求之一（[`pyeapi.readthedocs.io/en/master/requirements.html`](http://pyeapi.readthedocs.io/en/master/requirements.html)）。

默认情况下，Pyeapi 客户端将在你的主目录中查找一个名为`eapi.conf`的 INI 风格的隐藏（前面带有点）文件。你可以通过指定`eapi.conf`文件路径来覆盖此行为。通常，将连接凭证与脚本本身分离并锁定是一个好主意。你可以查看 Arista Pyeapi 文档（[`pyeapi.readthedocs.io/en/master/configfile.html#configfile`](http://pyeapi.readthedocs.io/en/master/configfile.html#configfile)）以了解文件中包含的字段。

这里是我在实验室中使用的文件：

```py
cat ~/.eapi.conf
[connection:Arista1]
host: 192.168.199.158
username: admin
password: arista
transport: https 
```

第一行`[connection:Arista1]`包含我们将用于 Pyeapi 连接的名称；其余字段应该相当直观。你可以锁定文件，使其对使用此文件的用户为只读：

```py
$ chmod 400 ~/.eapi.conf
$ ls -l ~/.eapi.conf
-r-------- 1 echou echou 94 Jan 27 18:15 /home/echou/.eapi.conf 
```

现在 Pyeapi 已经安装好了，让我们来看一些示例。

### Pyeapi 示例

让我们从通过在交互式 Python shell 中创建一个对象来连接到 EOS 节点开始：

```py
>>> import pyeapi
>>> arista1 = pyeapi.connect_to('Arista1') 
```

我们可以向节点执行`show`命令并接收输出：

```py
>>> import pprint
>>> pprint.pprint(arista1.enable('show hostname'))
[{'command': 'show hostname',
'encoding': 'json',
'result': {'fqdn': 'arista1', 'hostname': 'arista1'}}] 
```

配置字段可以是单个命令或使用`config()`方法的命令列表：

```py
>>> arista1.config('hostname arista1-new')
[{}]
>>> pprint.pprint(arista1.enable('show hostname'))
[{'command': 'show hostname',
 'encoding': 'json',
 'result': {'fqdn': 'arista1-new', 'hostname': 'arista1-new'}}]
>>> arista1.config(['interface ethernet 1/3', 'description my_link']) [{}, {}] 
```

注意，命令缩写（`show run`与`show running-config`）和一些扩展可能不会工作：

```py
>>> pprint.pprint(arista1.enable('show run'))
Traceback (most recent call last):
...
File "/usr/local/lib/python3.5/dist-packages/pyeapi/eapilib.py", line 396, in send
raise CommandError(code, msg, command_error=err, output=out) pyeapi.eapilib.CommandError: Error [1002]: CLI command 2 of 2 'show run' failed: invalid command [incomplete token (at token 1: 'run')]
>>>
>>> pprint.pprint(arista1.enable('show running-config interface ethernet 1/3'))
Traceback (most recent call last):
...
pyeapi.eapilib.CommandError: Error [1002]: CLI command 2 of 2 'show running-config interface ethernet 1/3' failed: invalid command [incomplete token (at token 2: 'interface')] 
```

我们可以始终捕获结果并获取所需值：

```py
>>> result = arista1.enable('show running-config')
>>> pprint.pprint(result[0]['result']['cmds']['interface Ethernet1/3'])
{'cmds': {'description my_link': None, 'switchport access vlan 100': None}, 'comments': []} 
```

到目前为止，我们一直在使用 eAPI 对`show`和`configuration`命令进行操作。Pyeapi 提供了各种 API 来简化生活。在下面的示例中，我们将连接到节点，调用 VLAN API，并开始对设备的 VLAN 参数进行操作。让我们看看：

```py
>>> import pyeapi
>>> node = pyeapi.connect_to('Arista1')
>>> vlans = node.api('vlans')
>>> type(vlans)
<class 'pyeapi.api.vlans.Vlans'>
>>> dir(vlans)
[...'command_builder', 'config', 'configure', 'configure_interface', 'configure_vlan', 'create', 'default', 'delete', 'error', 'get', 'get_block', 'getall', 'items', 'keys', 'node', 'remove_trunk_group', 'set_name', 'set_state', 'set_trunk_groups', 'values']
>>> vlans.getall()
{'1': {'vlan_id': '1', 'trunk_groups': [], 'state': 'active', 'name': 'default'}}
>>> vlans.get(1)
{'vlan_id': 1, 'trunk_groups': [], 'state': 'active', 'name': 'default'}
>>> vlans.create(10) True
>>> vlans.getall()
{'1': {'vlan_id': '1', 'trunk_groups': [], 'state': 'active', 'name':
'default'}, '10': {'vlan_id': '10', 'trunk_groups': [], 'state': 'active', 'name': 'VLAN0010'}}
>>> vlans.set_name(10, 'my_vlan_10') True 
```

让我们验证设备上是否已创建了 VLAN 10：

```py
arista1#sh vlan
VLAN Name Status Ports
----- -------------------------------- --------- --------------------
-----
1 default active
10 my_vlan_10 active 
```

如我们所见，EOS 对象上的 Python 原生 API 是 Pyeapi 在 eAPI 之上的优势所在。它将底层属性抽象到设备对象中，使代码更简洁、更容易阅读。

要查看 Pyeapi API 的完整列表，请参阅官方文档（[`pyeapi.readthedocs.io/en/master/api_modules/_list_of_modules.html`](http://pyeapi.readthedocs.io/en/master/api_modules/_list_of_modules.html)）。

为了总结本节，让我们假设我们重复之前的步骤足够多次，以至于我们希望编写另一个 Python 类来节省我们的工作。

`pyeapi_1.py` 脚本如下所示：

```py
#!/usr/bin/env python3
import pyeapi
class my_switch():

    def __init__(self, config_file_location, device):
        # loads the config file
        pyeapi.client.load_config(config_file_location)
        self.node = pyeapi.connect_to(device)
        self.hostname = self.node.enable('show hostname')[0]['result']['hostname']
        self.running_config = self.node.enable('show running-config')
    def create_vlan(self, vlan_number, vlan_name):
        vlans = self.node.api('vlans')
        vlans.create(vlan_number)
        vlans.set_name(vlan_number, vlan_name) 
```

从脚本中我们可以看到，我们自动连接到节点，设置主机名，并在连接时加载 `running_config`。我们还创建了一个使用 VLAN API 创建 VLAN 的类方法。让我们在交互式外壳中尝试运行这个脚本：

```py
>>> import pyeapi_1
>>> s1 = pyeapi_1.my_switch('/tmp/.eapi.conf', 'Arista1')
>>> s1.hostname
'arista1'
>>> s1.running_config
[{'encoding': 'json', 'result': {'cmds': {'interface Ethernet27': {'cmds':
{}, 'comments': []}, 'ip routing': None, 'interface face Ethernet29':
{'cmds': {}, 'comments': []}, 'interface Ethernet26': {'cmds': {}, 'comments': []}, 'interface Ethernet24/4': h.':
<omitted>
'interface Ethernet3/1': {'cmds': {}, 'comments': []}}, 'comments': [],
'header': ['! device: arista1 (DCS-7050QX-32, EOS-4.16.6M)n!n']},
'command': 'show running-config'}]
>>> s1.create_vlan(11, 'my_vlan_11')
>>> s1.node.api('vlans').getall()
{'11': {'name': 'my_vlan_11', 'vlan_id': '11', 'trunk_groups': [], 'state':
'active'}, '10': {'name': 'my_vlan_10', 'vlan_id': '10', 'trunk_groups': [], 'state': 'active'}, '1': {'name': 'default', 'vlan_id': '1', 'trunk_groups': [], 'state': 'active'}}
>>> 
```

我们现在已经研究了网络领域三大供应商的 Python 脚本：思科系统、Juniper 网络和 Arista 网络。在下一节中，我们将探讨一个在相同领域获得一些动力的开源网络操作系统。

# VyOS 示例

VyOS 是一个完全开源的网络操作系统，它可以在广泛的硬件、虚拟机和云服务提供商上运行（[`vyos.io/`](https://vyos.io/））。由于其开源特性，它在开源社区中获得了广泛的支持。许多开源项目都将 VyOS 作为默认的测试平台。在章节的最后部分，我们将查看一个简短的 VyOS 示例。

VyOS 映像可以以各种格式下载：[https://wiki.vyos.net/wiki/Installation](https://wiki.vyos.net/wiki/Installation)。一旦下载并初始化，我们就可以在我们的管理主机上安装 Python 库：

```py
(venv) $ pip install vymgmt 
```

示例脚本 `vyos_1.py` 非常简单：

```py
#!/usr/bin/env python3
import vymgmt
vyos = vymgmt.Router('192.168.2.116', 'vyos', password='vyos')
vyos.login()
vyos.configure()
vyos.set("system domain-name networkautomationnerds.net")
vyos.commit()
vyos.save()
vyos.exit()
vyos.logout() 
```

我们可以执行脚本以更改系统域名：

```py
(venv) $ python vyos_1.py 
```

我们可以登录到设备以验证域名更改：

```py
vyos@vyos:~$ show configuration | match domain
domain-name networkautomationnerds.net 
```

如示例所示，我们使用的 VyOS 方法与其他来自专有供应商的示例非常相似。这主要是出于设计考虑，因为它们提供了一种从使用其他供应商设备到开源 VyOS 的便捷过渡。我们接近章节的结尾。还有一些其他库值得提及，并在开发中保持关注，我们将在下一节中介绍。

# 其他库

我们将通过提及几个关于供应商中立库的杰出努力来结束这一章，例如 Nornir ([`nornir.readthedocs.io/en/stable/index.html`](https://nornir.readthedocs.io/en/stable/index.html))、Netmiko ([`github.com/ktbyers/netmiko`](https://github.com/ktbyers/netmiko))、NAPALM ([`github.com/napalm-automation/napalm`](https://github.com/napalm-automation/napalm)) 和 Scrapli ([`carlmontanari.github.io/scrapli/`](https://carlmontanari.github.io/scrapli/))。我们在上一章中看到了它们的一些示例。这些供应商中立库中大多数可能需要更长时间来支持最新的平台或功能。然而，由于这些库是供应商中立的，如果你不喜欢工具的供应商锁定，这些库是不错的选择。使用供应商中立库的另一个好处是它们通常是开源的，因此你可以为新的功能和错误修复做出贡献。

# 摘要

在本章中，我们探讨了与 Cisco、Juniper、Arista 和 Vyatta 网络设备通信和管理的方法。我们探讨了与 NETCONF 和 REST 的直接通信，以及使用供应商提供的库，如 PyEZ 和 Pyeapi。这些都是不同层次的抽象，旨在提供一种无需人工干预即可通过编程管理网络设备的方法。

在 *第四章*，*Python 自动化框架* 中，我们将探讨一个更高层次的供应商中立抽象框架，称为 Ansible。Ansible 是一个用 Python 编写的开源通用自动化工具。它可以用于自动化服务器、网络设备、负载均衡器等等。当然，对于我们的目的，我们将专注于使用这个自动化框架来管理网络设备。

# 加入我们的图书社区

要加入这本书的社区——在那里你可以分享反馈、向作者提问，并了解新版本——请扫描下面的二维码：

[`packt.link/networkautomationcommunity`](https://packt.link/networkautomationcommunity)

![](img/QR_Code2903617220506617062.png)
