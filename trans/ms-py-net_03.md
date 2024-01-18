# API 和意图驱动的网络

在第二章中，*低级网络设备交互*，我们看了一下使用 Pexpect 和 Paramiko 与网络设备进行交互的方法。这两个工具都使用了一个模拟用户在终端前输入命令的持久会话。这在一定程度上是有效的。很容易发送命令以在设备上执行并捕获输出。然而，当输出超过几行字符时，计算机程序很难解释输出。Pexpect 和 Paramiko 返回的输出是一系列字符，是为人类阅读而设计的。输出的结构包括了人类友好的行和空格，但对计算机程序来说很难理解。

为了让我们的计算机程序自动化执行我们想要执行的许多任务，我们需要解释返回的结果，并根据返回的结果采取后续行动。当我们无法准确和可预测地解释返回的结果时，我们无法有信心执行下一个命令。

幸运的是，这个问题已经被互联网社区解决了。想象一下当计算机和人类都在阅读网页时的区别。人类看到的是浏览器解释的单词、图片和空格；计算机看到的是原始的 HTML 代码、Unicode 字符和二进制文件。当一个网站需要成为另一个计算机的网络服务时会发生什么？同样的网络资源需要同时适应人类客户和其他计算机程序。这个问题听起来是不是很熟悉？答案就是**应用程序接口**（**API**）。需要注意的是，根据维基百科的说法，API 是一个概念，而不是特定的技术或框架。

在计算机编程中，**应用程序编程接口**（**API**）是一组子程序定义、协议和用于构建应用软件的工具。一般来说，它是各种软件组件之间清晰定义的通信方法集。一个好的 API 通过提供所有构建块，使得开发计算机程序更容易，然后由程序员组合在一起。

在我们的用例中，清晰定义的通信方法集将在我们的 Python 程序和目标设备之间。我们的网络设备 API 提供了一个独立的接口供计算机程序使用。确切的 API 实现是特定于供应商的。一个供应商可能更喜欢 XML 而不是 JSON，有些可能提供 HTTPS 作为底层传输协议，而其他供应商可能提供 Python 库作为包装器。尽管存在差异，API 的概念仍然是相同的：它是一种为其他计算机程序优化的独立通信方法。

在本章中，我们将讨论以下主题：

+   将基础设施视为代码、意图驱动的网络和数据建模

+   思科 NX-API 和面向应用的基础设施

+   Juniper NETCONF 和 PyEZ

+   Arista eAPI 和 PyEAPI

# 基础设施即代码

在一个完美的世界里，设计和管理网络的网络工程师和架构师应该关注网络应该实现的目标，而不是设备级别的交互。在我作为当地 ISP 的实习生的第一份工作中，我兴奋地安装了一个路由器在客户现场，打开了他们的分段帧中继链路（还记得那些吗？）。我应该怎么做？我问道。我拿到了一个打开帧中继链路的标准操作流程。我去了客户现场，盲目地输入命令，看着绿灯闪烁，然后高兴地收拾行李，为自己的工作感到自豪。尽管第一份工作很令人兴奋，但我并没有完全理解我在做什么。我只是在按照指示行事，没有考虑我输入的命令的影响。如果灯是红色而不是绿色，我该如何排除故障？我想我会打电话回办公室求助（泪水可选）。

当然，网络工程不是关于在设备上输入命令，而是建立一种允许服务尽可能顺畅地从一点传递到另一点的方式。我们必须使用的命令和我们必须解释的输出只是达到目的的手段。换句话说，我们应该专注于网络的意图。我们想要网络实现的目标比我们用来让设备做我们想让它做的命令语法更重要。如果我们进一步提取描述我们意图的代码行的想法，我们可以潜在地将我们整个基础设施描述为特定状态。基础设施将在代码行中描述，并有必要的软件或框架强制执行该状态。

# 基于意图驱动的网络

自从这本书第一版出版以来，“基于意图的网络”这个术语在主要网络供应商选择将其用于描述其下一代设备后得到了更多的使用。在我看来，“基于意图驱动的网络”是定义网络应该处于的状态，并有软件代码来强制执行该状态的想法。举个例子，如果我的目标是阻止端口 80 被外部访问，那么我应该将这个作为网络意图声明。底层软件将负责知道配置和应用必要的访问控制列表的语法在边界路由器上实现这个目标。当然，“基于意图驱动的网络”是一个没有明确实现的想法。但这个想法很简单明了，我在此要主张我们应该更多地关注网络的意图，并摆脱设备级别的交互。

在使用 API 时，我认为这让我们更接近基于意图驱动的网络的状态。简而言之，因为我们抽象了在目标设备上执行的特定命令的层，我们关注的是我们的意图，而不是具体的命令。例如，回到我们的“阻止端口 80”的访问控制列表的例子，我们可能在思科上使用访问控制列表和访问组，而在 Juniper 上使用过滤列表。然而，在使用 API 时，我们的程序可以开始询问执行者的意图，同时掩盖他们正在与何种物理设备交流。我们甚至可以使用更高级的声明性框架，比如 Ansible，我们将在第四章中介绍，即《Python 自动化框架- Ansible 基础》。但现在，让我们专注于网络 API。

# 屏幕抓取与 API 结构化输出

想象一个常见的情景，我们需要登录到网络设备，并确保设备上的所有接口都处于 up/up 状态（状态和协议都显示为`up`）。对于人类网络工程师来说，登录到 Cisco NX-OS 设备，通过终端发出`show IP interface brief`命令就足够简单，可以轻松地从输出中看出哪个接口是 up 的：

```py
 nx-osv-2# show ip int brief
    IP Interface Status for VRF "default"(1)
    Interface IP Address Interface Status
    Lo0 192.168.0.2 protocol-up/link-up/admin-up
    Eth2/1 10.0.0.6 protocol-up/link-up/admin-up
    nx-osv-2#
```

换行符、空格和列标题的第一行很容易从人眼中区分出来。事实上，它们是为了帮助我们对齐，比如说，从第一行到第二行和第三行的每个接口的 IP 地址。如果我们把自己放在计算机的位置上，所有这些空格和换行只会让我们远离真正重要的输出，那就是：哪些接口处于 up/up 状态？为了说明这一点，我们可以看一下相同操作的 Paramiko 输出：

```py
 >>> new_connection.send('sh ip int briefn')
    16
    >>> output = new_connection.recv(5000)
    >>> print(output)
    b'sh ip int briefrrnIP Interface Status for VRF 
    "default"(1)rnInterface IP Address Interface 
    StatusrnLo0 192.168.0.2 protocol-up/link-up/admin-up 
    rnEth2/1 10.0.0.6 protocol-up/link-up/admin-up rnrnx-
    osv-2# '
    >>>
```

如果我们要解析出这些数据，我会以伪代码的方式进行如下操作（简化了我将要编写的代码的表示方式）：

1.  通过换行符分割每一行。

1.  我可能不需要包含`show ip interface brief`执行命令的第一行。目前，我认为我不需要它。

1.  删除第二行直到 VRF 的所有内容，并将其保存在一个变量中，因为我们想知道输出显示的是哪个 VRF。

1.  对于其余的行，因为我们不知道有多少个接口，我们将使用正则表达式语句来搜索行是否以可能的接口开头，比如`lo`表示环回接口，`Eth`表示以太网接口。

1.  我们需要通过空格将这行分成三个部分，每个部分包括接口名称、IP 地址，然后是接口状态。

1.  然后进一步使用斜杠(`/`)分割接口状态，以获取协议、链路和管理状态。

哇，这需要大量的工作，而人类一眼就能看出来！你可能能够优化代码和行数，但总的来说，当我们需要屏幕抓取一些结构不太清晰的东西时，这就是我们需要做的。这种方法有许多缺点，但我能看到的一些更大的问题列在下面：

+   **可扩展性**：我们花了很多时间来仔细解析每个命令的输出。很难想象我们如何能够对我们通常运行的数百个命令进行这样的操作。

+   **可预测性**：实际上并没有保证输出在不同软件版本之间保持不变。如果输出稍有变化，可能会使我们辛苦收集的信息变得毫无用处。

+   **供应商和软件锁定**：也许最大的问题是，一旦我们花费了所有这些时间来解析特定供应商和软件版本（在本例中为 Cisco NX-OS）的输出，我们需要重复这个过程来选择下一个供应商。我不知道你怎么看，但如果我要评估一个新的供应商，如果我不得不重新编写所有的屏幕抓取代码，那么新的供应商就处于严重的入门劣势。

让我们将其与相同`show IP interface brief`命令的 NX-API 调用输出进行比较。我们将在本章后面详细介绍如何从设备中获取此输出，但这里重要的是将以下输出与先前的屏幕抓取输出进行比较：

```py
    {
     "ins_api":{
     "outputs":{
     "output":{
     "body":{
     "TABLE_intf":[
       {
       "ROW_intf":{
       "admin-state":"up",
       "intf-name":"Lo0",
       "iod":84,
       "ip-disabled":"FALSE",
       "link-state":"up",
       "prefix":"192.168.0.2",
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

NX-API 可以返回 XML 或 JSON 格式的输出，这是我们正在查看的 JSON 输出。您可以立即看到输出是结构化的，并且可以直接映射到 Python 字典数据结构。无需解析-您只需选择键并检索与键关联的值。您还可以从输出中看到各种元数据，例如命令的成功或失败。如果命令失败，将显示一条消息，告诉发送者失败的原因。您不再需要跟踪已发出的命令，因为它已在“输入”字段中返回给您。输出中还有其他有用的元数据，例如 NX-API 版本。

这种类型的交换使供应商和运营商的生活更加轻松。对于供应商来说，他们可以轻松地传输配置和状态信息。当需要公开额外数据时，他们可以使用相同的数据结构添加额外字段。对于运营商来说，他们可以轻松地摄取信息并围绕它构建基础设施。一般认为自动化是非常需要的，也是一件好事。问题通常集中在自动化的格式和结构上。正如您将在本章后面看到的，API 的伞下有许多竞争技术。仅在传输方面，我们有 REST API、NETCONF 和 RESTCONF 等。最终，整体市场可能会决定未来的最终数据格式。与此同时，我们每个人都可以形成自己的观点，并帮助推动行业向前发展。

# 基础设施的数据建模

根据维基百科（[`en.wikipedia.org/wiki/Data_model`](https://en.wikipedia.org/wiki/Data_model)）的定义，数据模型的定义如下：

数据模型是一个抽象模型，它组织数据元素并规范它们之间以及与现实世界实体属性的关系。例如，数据模型可以指定代表汽车的数据元素由许多其他元素组成，这些元素反过来代表汽车的颜色和大小，并定义其所有者。

数据建模过程可以用以下图表来说明：

![](img/2c817be7-d41c-47bd-929a-130e1a63fd87.png)数据建模过程

当应用于网络时，我们可以将这个概念应用为描述我们的网络的抽象模型，无论是数据中心、校园还是全球广域网。如果我们仔细观察物理数据中心，可以将层 2 以太网交换机视为包含 MAC 地址映射到每个端口的设备。我们的交换机数据模型描述了 MAC 地址应该如何保存在表中，其中包括键、附加特性（考虑 VLAN 和私有 VLAN）等。同样，我们可以超越设备，将整个数据中心映射到一个模型中。我们可以从每个接入、分发和核心层中的设备数量开始，它们是如何连接的，以及它们在生产环境中应该如何行为。例如，如果我们有一个 fat-tree 网络，每个脊柱路由器应该有多少链接，它们应该包含多少路由，每个前缀应该有多少下一跳？这些特性可以以一种格式映射出来，可以与我们应该始终检查的理想状态进行对比。

**另一种下一代**（**YANG**）是一种相对新的网络数据建模语言，正在受到关注（尽管一般的看法是，一些 IETF 工作组确实有幽默感）。它首次在 2010 年的 RFC 6020 中发布，并且自那时以来在供应商和运营商中得到了广泛的应用。在撰写本文时，对 YANG 的支持在供应商和平台之间差异很大。因此，生产中的适应率相对较低。但是，这是一项值得关注的技术。

# 思科 API 和 ACI

思科系统是网络领域的 800 磅大猩猩，在网络自动化的趋势中没有落后。在推动网络自动化的过程中，他们进行了各种内部开发、产品增强、合作伙伴关系，以及许多外部收购。然而，由于产品线涵盖路由器、交换机、防火墙、服务器（统一计算）、无线、协作软件和硬件以及分析软件等，要知道从哪里开始是很困难的。

由于这本书侧重于 Python 和网络，我们将把这一部分范围限定在主要的网络产品上。特别是，我们将涵盖以下内容：

+   NX-API 的 Nexus 产品自动化

+   思科 NETCONF 和 YANG 示例

+   数据中心的思科应用中心基础设施

+   企业级思科应用中心基础设施

对于这里的 NX-API 和 NETCONF 示例，我们可以使用思科 DevNet 始终开启的实验室设备，或者在本地运行思科 VIRL。由于 ACI 是一个独立的产品，并且在以下 ACI 示例中与物理交换机一起许可使用，我建议使用 DevNet 实验室来了解这些工具。如果你是那些有自己的私人 ACI 实验室可以使用的幸运工程师之一，请随意在相关示例中使用它。

我们将使用与第二章中相同的实验拓扑，*低级网络设备交互*，只有一个设备运行 nx-osv 除外：

![](img/f905b772-b032-4d3a-b935-4d5cdd6b0faf.png) 实验室拓扑

让我们来看看 NX-API。

# 思科 NX-API

Nexus 是思科的主要数据中心交换机产品线。NX-API ([`www.cisco.com/c/en/us/td/docs/switches/datacenter/nexus9000/sw/6-x/programmability/guide/b_Cisco_Nexus_9000_Ser`](http://www.cisco.com/c/en/us/td/docs/switches/datacenter/nexus9000/sw/6-x/programmability/guide/b_Cisco_Nexus_9000_Series_NX-OS_Programmability_Guide/b_Cisco_Nexus_9000_Series_NX-OS_Programmability_Guide_chapter_011.html)[ies_NX-OS_Programmability_Guide/b_Cisco_Nexus_9000_Series_NX-OS_Programmability_Guide_chapter_011.html](http://www.cisco.com/c/en/us/td/docs/switches/datacenter/nexus9000/sw/6-x/programmability/guide/b_Cisco_Nexus_9000_Series_NX-OS_Programmability_Guide/b_Cisco_Nexus_9000_Series_NX-OS_Programmability_Guide_chapter_011.html))允许工程师通过各种传输方式与交换机进行交互，包括 SSH、HTTP 和 HTTPS。

# 实验室软件安装和设备准备

以下是我们将安装的 Ubuntu 软件包。你可能已经安装了一些软件包，比如`pip`和`git`：

```py
$ sudo apt-get install -y python3-dev libxml2-dev libxslt1-dev libffi-dev libssl-dev zlib1g-dev python3-pip git python3-requests
```

如果你使用的是 Python 2，使用以下软件包代替：`sudo apt-get install -y python-dev libxml2-dev libxslt1-dev libffi-dev libssl-dev zlib1g-dev python-pip git python-requests`。

`ncclient` ([`github.com/ncclient/ncclient`](https://github.com/ncclient/ncclient))库是一个用于 NETCONF 客户端的 Python 库。我们将从 GitHub 存储库中安装它，以便安装最新版本：

```py
$ git clone https://github.com/ncclient/ncclient
$ cd ncclient/
$ sudo python3 setup.py install
$ sudo python setup.py install #for Python 2
```

Nexus 设备上的 NX-API 默认关闭，因此我们需要打开它。我们可以使用已经创建的用户（如果你使用的是 VIRL 自动配置），或者为 NETCONF 过程创建一个新用户：

```py
feature nxapi
username cisco password 5 $1$Nk7ZkwH0$fyiRmMMfIheqE3BqvcL0C1 role network-operator
username cisco role network-admin
username cisco passphrase lifetime 99999 warntime 14 gracetime 3
```

对于我们的实验室，我们将同时打开 HTTP 和沙盒配置，因为它们在生产中应该关闭：

```py
nx-osv-2(config)# nxapi http port 80
nx-osv-2(config)# nxapi sandbox
```

我们现在准备看我们的第一个 NX-API 示例。

# NX-API 示例

NX-API 沙盒是一个很好的方式来玩各种命令、数据格式，甚至可以直接从网页上复制 Python 脚本。在最后一步，我们为了学习目的打开了它。在生产中应该关闭它。让我们打开一个网页浏览器，看看基于我们已经熟悉的 CLI 命令的各种消息格式、请求和响应。

![](img/3d7aec04-66a7-4ecb-95c2-f19b17fd399a.png)

在下面的例子中，我选择了`JSON-RPC`和`CLI`命令类型来执行`show version`命令：

![](img/48fbd88d-cc73-4a69-aa66-fff954aaa457.png)

如果你对消息格式的支持性不确定，或者对你想在代码中检索的值的响应数据字段键有疑问，沙盒会派上用场。

在我们的第一个例子中，我们只是连接到 Nexus 设备，并在连接时打印出交换的能力：

```py
    #!/usr/bin/env python3
    from ncclient import manager
    conn = manager.connect(
            host='172.16.1.90',
            port=22,
            username='cisco',
            password='cisco',
            hostkey_verify=False,
            device_params={'name': 'nexus'},
            look_for_keys=False)
    for value in conn.server_capabilities:
        print(value)
    conn.close_session()
```

主机、端口、用户名和密码的连接参数都很容易理解。设备参数指定了客户端连接的设备类型。当使用 ncclient 库时，在 Juniper NETCONF 部分会看到不同的响应。`hostkey_verify`绕过了 SSH 的`known_host`要求；如果不绕过，主机需要列在`~/.ssh/known_hosts`文件中。`look_for_keys`选项禁用了公钥私钥认证，而是使用用户名和密码进行认证。

如果你在 Python 3 和 Paramiko 中遇到问题，请随时使用 Python 2。希望在你阅读本节时，这个问题已经得到解决。

输出将显示这个版本的 NX-OS 支持的 XML 和 NETCONF 特性：

```py
$ python cisco_nxapi_1.py
urn:ietf:params:netconf:capability:writable-running:1.0
urn:ietf:params:netconf:capability:rollback-on-error:1.0
urn:ietf:params:netconf:capability:validate:1.0
urn:ietf:params:netconf:capability:url:1.0?scheme=file
urn:ietf:params:netconf:base:1.0
urn:ietf:params:netconf:capability:candidate:1.0
urn:ietf:params:netconf:capability:confirmed-commit:1.0
urn:ietf:params:xml:ns:netconf:base:1.0
```

使用 ncclient 和通过 SSH 的 NETCONF 非常好，因为它让我们更接近本地实现和语法。我们将在本书的后面使用相同的库。对于 NX-API，处理 HTTPS 和 JSON-RPC 可能更容易。在 NX-API 开发者沙箱的早期截图中，如果你注意到，在请求框中，有一个标有 Python 的框。如果你点击它，你将能够获得一个基于请求库自动生成的 Python 脚本。

以下脚本使用了一个名为`requests`的外部 Python 库。`requests`是一个非常流行的、自称为人类的 HTTP 库，被亚马逊、谷歌、NSA 等公司使用。你可以在官方网站上找到更多关于它的信息。

对于`show version`的例子，以下 Python 脚本是自动生成的。我将输出粘贴在这里，没有进行任何修改：

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
    response = requests.post(url,data=json.dumps(payload), 
    headers=myheaders,auth=(switchuser,switchpassword)).json()
```

在`cisco_nxapi_2.py`脚本中，你会看到我只修改了前面文件的 URL、用户名和密码。输出被解析为只包括软件版本。以下是输出：

```py
$ python3 cisco_nxapi_2.py
7.2(0)D1(1) [build 7.2(0)ZD(0.120)]
```

使用这种方法的最好之处在于，相同的总体语法结构既适用于配置命令，也适用于显示命令。这在`cisco_nxapi_3.py`文件中有所体现。对于多行配置，你可以使用 ID 字段来指定操作的顺序。在`cisco_nxapi_4.py`中，列出了用于更改接口 Ethernet 2/12 的描述的有效负载：

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

我们可以通过查看 Nexus 设备的运行配置来验证前面配置脚本的结果：

```py
hostname nx-osv-1-new
...
interface Ethernet2/12
description foo-bar
shutdown
no switchport
mac-address 0000.0000.002f 
```

在接下来的部分，我们将看一些关于 Cisco NETCONF 和 YANG 模型的例子。

# Cisco 和 YANG 模型

在本章的前面，我们探讨了使用数据建模语言 YANG 来表达网络的可能性。让我们通过例子再深入了解一下。

首先，我们应该知道 YANG 模型只定义了通过 NETCONF 协议发送的数据类型，而不规定数据应该是什么。其次，值得指出的是 NETCONF 存在作为一个独立的协议，正如我们在 NX-API 部分看到的那样。YANG 作为相对较新的技术，在各个供应商和产品线之间的支持性不够稳定。例如，如果我们对运行 IOS-XE 的 Cisco 1000v 运行相同的能力交换脚本，我们会看到这样的结果：

```py
 urn:cisco:params:xml:ns:yang:cisco-virtual-service?module=cisco-
 virtual-service&revision=2015-04-09
 http://tail-f.com/ns/mibs/SNMP-NOTIFICATION-MIB/200210140000Z?
 module=SNMP-NOTIFICATION-MIB&revision=2002-10-14
 urn:ietf:params:xml:ns:yang:iana-crypt-hash?module=iana-crypt-
 hash&revision=2014-04-04&features=crypt-hash-sha-512,crypt-hash-
 sha-256,crypt-hash-md5
 urn:ietf:params:xml:ns:yang:smiv2:TUNNEL-MIB?module=TUNNEL-
 MIB&revision=2005-05-16
 urn:ietf:params:xml:ns:yang:smiv2:CISCO-IP-URPF-MIB?module=CISCO-
 IP-URPF-MIB&revision=2011-12-29
 urn:ietf:params:xml:ns:yang:smiv2:ENTITY-STATE-MIB?module=ENTITY-
 STATE-MIB&revision=2005-11-22
 urn:ietf:params:xml:ns:yang:smiv2:IANAifType-MIB?module=IANAifType-
 MIB&revision=2006-03-31
 <omitted>
```

将此与我们在 NX-OS 中看到的输出进行比较。显然，IOS-XE 对 YANG 模型功能的支持要比 NX-OS 多。在整个行业范围内，当支持时，网络数据建模显然是可以跨设备使用的，这对于网络自动化是有益的。然而，鉴于供应商和产品的支持不均衡，我认为它还不够成熟，不能完全用于生产网络。对于本书，我包含了一个名为`cisco_yang_1.py`的脚本，演示了如何使用 YANG 过滤器`urn:ietf:params:xml:ns:yang:ietf-interfaces`来解析 NETCONF XML 输出的起点。

您可以在 YANG GitHub 项目页面上检查最新的供应商支持（[`github.com/YangModels/yang/tree/master/vendor`](https://github.com/YangModels/yang/tree/master/vendor)）。

# Cisco ACI

Cisco **Application Centric Infrastructure**（**ACI**）旨在为所有网络组件提供集中化的方法。在数据中心环境中，这意味着集中控制器知道并管理着脊柱、叶子和机架顶部交换机，以及所有网络服务功能。这可以通过 GUI、CLI 或 API 来实现。有人可能会认为 ACI 是思科对更广泛的基于控制器的软件定义网络的回应。

对于 ACI 而言，有点令人困惑的是 ACI 和 APIC-EM 之间的区别。简而言之，ACI 专注于数据中心操作，而 APIC-EM 专注于企业模块。两者都提供了对网络组件的集中视图和控制，但每个都有自己的重点和工具集。例如，很少见到任何主要数据中心部署面向客户的无线基础设施，但无线网络是当今企业的重要组成部分。另一个例子是网络安全的不同方法。虽然安全在任何网络中都很重要，但在数据中心环境中，许多安全策略被推送到服务器的边缘节点以实现可伸缩性。在企业安全中，策略在网络设备和服务器之间有一定的共享。

与 NETCONF RPC 不同，ACI API 遵循 REST 模型，使用 HTTP 动词（`GET`，`POST`，`DELETE`）来指定所需的操作。

我们可以查看`cisco_apic_em_1.py`文件，这是 Cisco 示例代码`lab2-1-get-network-device-list.py`的修改版本（[`github.com/CiscoDevNet/apicem-1.3-LL-sample-codes/blob/master/basic-labs/lab2-1-get-network-device-list.py`](https://github.com/CiscoDevNet/apicem-1.3-LL-sample-codes/blob/master/basic-labs/lab2-1-get-network-device-list.py)）。

在以下部分列出了没有注释和空格的缩写版本。

名为`getTicket()`的第一个函数在控制器上使用 HTTPS `POST`，路径为`/api/v1/ticket`，在标头中嵌入用户名和密码。此函数将返回仅在有限时间内有效的票证的解析响应：

```py
  def getTicket():
      url = "https://" + controller + "/api/v1/ticket"
      payload = {"username":"usernae","password":"password"}
      header = {"content-type": "application/json"}
      response= requests.post(url,data=json.dumps(payload), headers=header, verify=False)
      r_json=response.json()
      ticket = r_json["response"]["serviceTicket"]
      return ticket
```

然后，第二个函数调用另一个名为`/api/v1/network-devices`的路径，并在标头中嵌入新获取的票证，然后解析结果：

```py
url = "https://" + controller + "/api/v1/network-device"
header = {"content-type": "application/json", "X-Auth-Token":ticket}
```

这是 API 交互的一个常见工作流程。客户端将在第一个请求中使用服务器进行身份验证，并接收一个基于时间的令牌。此令牌将在后续请求中使用，并将作为身份验证的证明。

输出显示了原始 JSON 响应输出以及解析后的表格。执行针对 DevNet 实验室控制器的部分输出如下所示：

```py
    Network Devices =
    {
     "version": "1.0",
     "response": [
     {
     "reachabilityStatus": "Unreachable",
     "id": "8dbd8068-1091-4cde-8cf5-d1b58dc5c9c7",
     "platformId": "WS-C2960C-8PC-L",
    <omitted>
     "lineCardId": null,
     "family": "Wireless Controller",
     "interfaceCount": "12",
     "upTime": "497 days, 2:27:52.95"
     }
    ]
    }
    8dbd8068-1091-4cde-8cf5-d1b58dc5c9c7 Cisco Catalyst 2960-C Series
     Switches
    cd6d9b24-839b-4d58-adfe-3fdf781e1782 Cisco 3500I Series Unified
    Access Points
    <omitted>
    55450140-de19-47b5-ae80-bfd741b23fd9 Cisco 4400 Series Integrated 
    Services Routers
    ae19cd21-1b26-4f58-8ccd-d265deabb6c3 Cisco 5500 Series Wireless LAN 
    Controllers
```

正如您所看到的，我们只查询了一个控制器设备，但我们能够高层次地查看控制器所知道的所有网络设备。在我们的输出中，Catalyst 2960-C 交换机，3500 接入点，4400 ISR 路由器和 5500 无线控制器都可以进一步探索。当然，缺点是 ACI 控制器目前只支持 Cisco 设备。

# Juniper 网络的 Python API

Juniper 网络一直是服务提供商群体中的最爱。如果我们退一步看看服务提供商垂直领域，自动化网络设备将是他们需求清单的首要任务。在云规模数据中心出现之前，服务提供商是拥有最多网络设备的人。一个典型的企业网络可能在公司总部有几个冗余的互联网连接，还有一些以枢纽-辐射方式连接回总部，使用服务提供商的私有 MPLS 网络。对于服务提供商来说，他们需要构建、配置、管理和排除连接和底层网络的问题。他们通过销售带宽以及增值的托管服务来赚钱。对于服务提供商来说，投资于自动化以使用最少的工程小时数来保持网络运行是合理的。在他们的用例中，网络自动化是他们竞争优势的关键。

在我看来，服务提供商网络的需求与云数据中心相比的一个区别是，传统上，服务提供商将更多的服务聚合到单个设备中。一个很好的例子是**多协议标签交换**（**MPLS**），几乎所有主要的服务提供商都提供，但在企业或数据中心网络中很少使用。正如 Juniper 非常成功地发现了这一需求，并且在满足服务提供商自动化需求方面表现出色。让我们来看一下 Juniper 的一些自动化 API。

# Juniper 和 NETCONF

**网络配置协议**（**NETCONF**）是一个 IETF 标准，最早于 2006 年发布为[RFC 4741](https://tools.ietf.org/html/rfc4741)，后来修订为[RFC 6241](https://tools.ietf.org/html/rfc6241)。Juniper 网络对这两个 RFC 标准做出了重大贡献。事实上，Juniper 是 RFC 4741 的唯一作者。Juniper 设备完全支持 NETCONF 是合情合理的，并且它作为大多数自动化工具和框架的基础层。NETCONF 的一些主要特点包括以下内容：

1.  它使用**可扩展标记语言**（**XML**）进行数据编码。

1.  它使用**远程过程调用**（**RPC**），因此在使用 HTTP(s)作为传输方式时，URL 端点是相同的，而所需的操作在请求的正文中指定。

1.  它在概念上是基于自上而下的层。这些层包括内容、操作、消息和传输：

![](img/46006bbf-bde4-4219-b26e-451e09a7d384.png)NETCONF 模型

Juniper 网络在其技术库中提供了一个广泛的 NETCONF XML 管理协议开发者指南（[`www.juniper.net/techpubs/en_US/junos13.2/information-products/pathway-pages/netconf-guide/netconf.html#overview`](https://www.juniper.net/techpubs/en_US/junos13.2/information-products/pathway-pages/netconf-guide/netconf.html#overview)）。让我们来看一下它的用法。

# 设备准备

为了开始使用 NETCONF，让我们创建一个单独的用户，并打开所需的服务：

```py
 set system login user netconf uid 2001
 set system login user netconf class super-user
 set system login user netconf authentication encrypted-password
 "$1$0EkA.XVf$cm80A0GC2dgSWJIYWv7Pt1"
 set system services ssh
 set system services telnet
 set system services netconf ssh port 830
```

对于 Juniper 设备实验室，我正在使用一个名为**Juniper Olive**的较旧、不受支持的平台。它仅用于实验目的。您可以使用您喜欢的搜索引擎找出一些关于 Juniper Olive 的有趣事实和历史。

在 Juniper 设备上，您可以随时查看配置，无论是在一个平面文件中还是在 XML 格式中。当您需要指定一条命令来进行配置更改时，`flat`文件非常方便：

```py
 netconf@foo> show configuration | display set
 set version 12.1R1.9
 set system host-name foo
 set system domain-name bar
 <omitted>
```

在某些情况下，当您需要查看配置的 XML 结构时，XML 格式非常方便：

```py
 netconf@foo> show configuration | display xml
 <rpc-reply >
 <configuration junos:commit-seconds="1485561328" junos:commit-
 localtime="2017-01-27 23:55:28 UTC" junos:commit-user="netconf">
 <version>12.1R1.9</version>
 <system>
 <host-name>foo</host-name>
 <domain-name>bar</domain-name>
```

我们已经在 Cisco 部分安装了必要的 Linux 库和 ncclient Python 库。如果您还没有这样做，请参考该部分并安装必要的软件包。

我们现在准备好查看我们的第一个 Juniper NETCONF 示例。

# Juniper NETCONF 示例

我们将使用一个非常简单的示例来执行`show version`。我们将把这个文件命名为`junos_netconf_1.py`：

```py
  #!/usr/bin/env python3

  from ncclient import manager

  conn = manager.connect(
      host='192.168.24.252',
      port='830',
      username='netconf',
      password='juniper!',
      timeout=10,
      device_params={'name':'junos'},
      hostkey_verify=False)

  result = conn.command('show version', format='text')
  print(result)
  conn.close_session()
```

脚本中的所有字段应该都很容易理解，除了`device_params`。从 ncclient 0.4.1 开始，添加了设备处理程序，用于指定不同的供应商或平台。例如，名称可以是 juniper、CSR、Nexus 或 Huawei。我们还添加了`hostkey_verify=False`，因为我们使用的是 Juniper 设备的自签名证书。

返回的输出是用 XML 编码的`rpc-reply`，其中包含一个`output`元素：

```py
    <rpc-reply message-id="urn:uuid:7d9280eb-1384-45fe-be48-
    b7cd14ccf2b7">
    <output>
    Hostname: foo
 Model: olive
 JUNOS Base OS boot [12.1R1.9]
 JUNOS Base OS Software Suite [12.1R1.9]
 <omitted>
 JUNOS Runtime Software Suite [12.1R1.9]
 JUNOS Routing Software Suite [12.1R1.9]
    </output>
    </rpc-reply>
```

我们可以解析 XML 输出以仅包括输出文本：

```py
      print(result.xpath('output')[0].text)
```

在`junos_netconf_2.py`中，我们将对设备进行配置更改。我们将从一些新的导入开始，用于构建新的 XML 元素和连接管理器对象：

```py
      #!/usr/bin/env python3

      from ncclient import manager
      from ncclient.xml_ import new_ele, sub_ele

      conn = manager.connect(host='192.168.24.252', port='830', 
    username='netconf' , password='juniper!', timeout=10, 
    device_params={'name':'junos'}, hostkey_v erify=False)
```

我们将锁定配置并进行配置更改：

```py
      # lock configuration and make configuration changes
      conn.lock()

      # build configuration
      config = new_ele('system')
      sub_ele(config, 'host-name').text = 'master'
      sub_ele(config, 'domain-name').text = 'python'
```

在构建配置部分，我们创建一个`system`元素，其中包含`host-name`和`domain-name`子元素。如果你想知道层次结构，你可以从 XML 显示中看到`system`的节点结构是`host-name`和`domain-name`的父节点：

```py
     <system>
        <host-name>foo</host-name>
        <domain-name>bar</domain-name>
    ...
    </system>
```

配置构建完成后，脚本将推送配置并提交配置更改。这些是 Juniper 配置更改的正常最佳实践步骤（锁定、配置、解锁、提交）：

```py
      # send, validate, and commit config
      conn.load_configuration(config=config)
      conn.validate()
      commit_config = conn.commit()
      print(commit_config.tostring)

      # unlock config
      conn.unlock()

      # close session
      conn.close_session()
```

总的来说，NETCONF 步骤与 CLI 步骤非常相似。请查看`junos_netconf_3.py`脚本，以获取更多可重用的代码。以下示例将步骤示例与一些 Python 函数结合起来：

```py
# make a connection object
def connect(host, port, user, password):
    connection = manager.connect(host=host, port=port, username=user,
            password=password, timeout=10, device_params={'name':'junos'},
            hostkey_verify=False)
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

这个文件可以自行执行，也可以被导入到其他 Python 脚本中使用。

Juniper 还提供了一个名为 PyEZ 的 Python 库，可用于其设备。我们将在下一节中看一些使用该库的示例。

# 开发人员的 Juniper PyEZ

**PyEZ**是一个高级的 Python 实现，与现有的 Python 代码更好地集成。通过使用 Python API，您可以执行常见的操作和配置任务，而无需对 Junos CLI 有深入的了解。

Juniper 在其技术库的[`www.juniper.net/techpubs/en_US/junos-pyez1.0/information-products/pathway-pages/junos-pyez-developer-guide.html#configuration`](https://www.juniper.net/techpubs/en_US/junos-pyez1.0/information-products/pathway-pages/junos-pyez-developer-guide.html#configuration)上维护了一份全面的 Junos PyEZ 开发人员指南。如果您有兴趣使用 PyEZ，我强烈建议至少浏览一下指南中的各个主题。

# 安装和准备

每个操作系统的安装说明都可以在*安装 Junos PyEZ* ([`www.juniper.net/techpubs/en_US/junos-pyez1.0/topics/task/installation/junos-pyez-server-installing.html`](https://www.juniper.net/techpubs/en_US/junos-pyez1.0/topics/task/installation/junos-pyez-server-installing.html))页面上找到。我们将展示 Ubuntu 16.04 的安装说明。

以下是一些依赖包，其中许多应该已经在主机上运行之前的示例中了：

```py
$ sudo apt-get install -y python3-pip python3-dev libxml2-dev libxslt1-dev libssl-dev libffi-dev
```

`PyEZ`包可以通过 pip 安装。在这里，我已经为 Python 3 和 Python 2 都安装了：

```py
$ sudo pip3 install junos-eznc
$ sudo pip install junos-eznc
```

在 Juniper 设备上，NETCONF 需要配置为 PyEZ 的基础 XML API：

```py
set system services netconf ssh port 830
```

对于用户认证，我们可以使用密码认证或 SSH 密钥对。创建本地用户很简单：

```py
set system login user netconf uid 2001
set system login user netconf class super-user
set system login user netconf authentication encrypted-password "$1$0EkA.XVf$cm80A0GC2dgSWJIYWv7Pt1"
```

对于`ssh`密钥认证，首先在主机上生成密钥对：

```py
$ ssh-keygen -t rsa
```

默认情况下，公钥将被称为`id_rsa.pub`，位于`~/.ssh/`目录下，而私钥将被命名为`id_rsa`，位于相同的目录下。将私钥视为永远不共享的密码。公钥可以自由分发。在我们的用例中，我们将把公钥移动到`/tmp`目录，并启用 Python 3 HTTP 服务器模块以创建可访问的 URL：

```py
$ mv ~/.ssh/id_rsa.pub /tmp
$ cd /tmp
$ python3 -m http.server
Serving HTTP on 0.0.0.0 port 8000 ...
```

对于 Python 2，请改用`python -m SimpleHTTPServer`。

从 Juniper 设备中，我们可以通过从 Python 3 web 服务器下载公钥来创建用户并关联公钥：

```py
netconf@foo# set system login user echou class super-user authentication load-key-file http://192.168.24.164:8000/id_rsa.pub
/var/home/netconf/...transferring.file........100% of 394 B 2482 kBps
```

现在，如果我们尝试使用管理站的私钥进行 ssh，用户将自动进行身份验证：

```py
$ ssh -i ~/.ssh/id_rsa 192.168.24.252
--- JUNOS 12.1R1.9 built 2012-03-24 12:52:33 UTC
echou@foo>
```

让我们确保两种身份验证方法都可以与 PyEZ 一起使用。让我们尝试用户名和密码组合：

```py
Python 3.5.2 (default, Nov 17 2016, 17:05:23)
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from jnpr.junos import Device
>>> dev = Device(host='192.168.24.252', user='netconf', password='juniper!')
>>> dev.open()
Device(192.168.24.252)
>>> dev.facts
{'serialnumber': '', 'personality': 'UNKNOWN', 'model': 'olive', 'ifd_style': 'CLASSIC', '2RE': False, 'HOME': '/var/home/juniper', 'version_info': junos.version_info(major=(12, 1), type=R, minor=1, build=9), 'switch_style': 'NONE', 'fqdn': 'foo.bar', 'hostname': 'foo', 'version': '12.1R1.9', 'domain': 'bar', 'vc_capable': False}
>>> dev.close()
```

我们也可以尝试使用 SSH 密钥身份验证：

```py
>>> from jnpr.junos import Device
>>> dev1 = Device(host='192.168.24.252', user='echou', ssh_private_key_file='/home/echou/.ssh/id_rsa')
>>> dev1.open()
Device(192.168.24.252)
>>> dev1.facts
{'HOME': '/var/home/echou', 'model': 'olive', 'hostname': 'foo', 'switch_style': 'NONE', 'personality': 'UNKNOWN', '2RE': False, 'domain': 'bar', 'vc_capable': False, 'version': '12.1R1.9', 'serialnumber': '', 'fqdn': 'foo.bar', 'ifd_style': 'CLASSIC', 'version_info': junos.version_info(major=(12, 1), type=R, minor=1, build=9)}
>>> dev1.close()
```

太好了！我们现在准备好查看一些 PyEZ 的示例了。

# PyEZ 示例

在之前的交互提示中，我们已经看到设备连接时，对象会自动检索有关设备的一些事实。在我们的第一个示例`junos_pyez_1.py`中，我们连接到设备并执行了`show interface em1`的 RPC 调用：

```py
      #!/usr/bin/env python3
      from jnpr.junos import Device
      import xml.etree.ElementTree as ET
      import pprint

      dev = Device(host='192.168.24.252', user='juniper', passwd='juniper!')

      try:
          dev.open()
      except Exception as err:
          print(err)
          sys.exit(1)

      result = 
    dev.rpc.get_interface_information(interface_name='em1', terse=True)
      pprint.pprint(ET.tostring(result))

      dev.close()
```

设备类具有一个包含所有操作命令的`rpc`属性。这非常棒，因为我们在 CLI 和 API 中可以做的事情之间没有差距。问题在于我们需要找出`xml rpc`元素标签。在我们的第一个示例中，我们如何知道`show interface em1`等同于`get_interface_information`？我们有三种方法可以找出这些信息：

1.  我们可以参考*Junos XML API 操作开发人员参考*

1.  我们可以使用 CLI 显示 XML RPC 等效，并用下划线（`_`）替换单词之间的破折号（`-`）

1.  我们还可以通过使用 PyEZ 库来进行编程

我通常使用第二个选项直接获取输出：

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

以下是使用 PyEZ 进行编程的示例（第三个选项）：

```py
 >>> dev1.display_xml_rpc('show interfaces em1', format='text')
 '<get-interface-information>n <interface-name>em1</interface-
 name>n</get-interface-information>n'
```

当然，我们还需要进行配置更改。在`junos_pyez_2.py`配置示例中，我们将从 PyEZ 导入一个额外的`Config()`方法：

```py
      #!/usr/bin/env python3
      from jnpr.junos import Device
      from jnpr.junos.utils.config import Config
```

我们将利用相同的块连接到设备：

```py
      dev = Device(host='192.168.24.252', user='juniper', 
    passwd='juniper!')

      try:
          dev.open()
      except Exception as err:
          print(err)
          sys.exit(1)
```

`new Config()`方法将加载 XML 数据并进行配置更改：

```py
      config_change = """
      <system>
        <host-name>master</host-name>
        <domain-name>python</domain-name>
      </system>
      """

      cu = Config(dev)
      cu.lock()
      cu.load(config_change)
      cu.commit()
      cu.unlock()

      dev.close()
```

PyEZ 示例设计简单。希望它们能展示您如何利用 PyEZ 满足 Junos 自动化需求的方式。

# Arista Python API

**Arista Networks**一直专注于大型数据中心网络。在其公司简介页面（[`www.arista.com/en/company/company-overview`](https://www.arista.com/en/company/company-overview)）中，如下所述：

“Arista Networks 成立的目的是开创并提供面向大型数据中心存储和计算环境的软件驱动云网络解决方案。”

请注意，该声明特别指出了**大型数据中心**，我们已经知道这些数据中心充斥着服务器、数据库和网络设备。自动化一直是 Arista 的主要特点之一是有道理的。事实上，他们的操作系统背后有一个 Linux 支撑，允许许多附加功能，如 Linux 命令和内置的 Python 解释器。

与其他供应商一样，您可以直接通过 eAPI 与 Arista 设备交互，或者您可以选择利用他们的`Python`库。我们将看到两者的示例。我们还将在后面的章节中看到 Arista 与 Ansible 框架的集成。

# Arista eAPI 管理

几年前，Arista 的 eAPI 首次在 EOS 4.12 中引入。它通过 HTTP 或 HTTPS 传输一系列显示或配置命令，并以 JSON 形式回应。一个重要的区别是它是**远程过程调用**（**RPC**）和**JSON-RPC**，而不是纯粹通过 HTTP 或 HTTPS 提供的 RESTFul API。对于我们的意图和目的，不同之处在于我们使用相同的 HTTP 方法（`POST`）向相同的 URL 端点发出请求。我们不是使用 HTTP 动词（`GET`，`POST`，`PUT`，`DELETE`）来表达我们的动作，而是简单地在请求的正文中说明我们的意图动作。在 eAPI 的情况下，我们将为我们的意图指定一个`method`键和一个`runCmds`值。

在以下示例中，我使用运行 EOS 4.16 的物理 Arista 交换机。

# eAPI 准备

Arista 设备上的 eAPI 代理默认处于禁用状态，因此我们需要在设备上启用它才能使用：

```py
arista1(config)#management api http-commands
arista1(config-mgmt-api-http-cmds)#no shut
arista1(config-mgmt-api-http-cmds)#protocol https port 443
arista1(config-mgmt-api-http-cmds)#no protocol http
arista1(config-mgmt-api-http-cmds)#vrf management
```

如您所见，我们已关闭 HTTP 服务器，而是仅使用 HTTPS 作为传输。从几个 EOS 版本前开始，默认情况下，管理接口位于名为**management**的 VRF 中。在我的拓扑中，我通过管理接口访问设备；因此，我已指定了 eAPI 管理的 VRF。您可以通过"show management api http-commands"命令检查 API 管理状态：

```py
arista1#sh management api http-commands
Enabled: Yes
HTTPS server: running, set to use port 443
HTTP server: shutdown, set to use port 80
Local HTTP server: shutdown, no authentication, set to use port 8080
Unix Socket server: shutdown, no authentication
VRF: management
Hits: 64
Last hit: 33 seconds ago
Bytes in: 8250
Bytes out: 29862
Requests: 23
Commands: 42
Duration: 7.086 seconds
SSL Profile: none
QoS DSCP: 0
 User Requests Bytes in Bytes out Last hit
----------- -------------- -------------- --------------- --------------
 admin 23 8250 29862 33 seconds ago

URLs
-----------------------------------------
Management1 : https://192.168.199.158:443

arista1#
```

启用代理后，您将能够通过访问设备的 IP 地址来访问 eAPI 的探索页面。如果您已更改访问的默认端口，只需在末尾添加即可。认证与交换机上的认证方法绑定。我们将使用设备上本地配置的用户名和密码。默认情况下，将使用自签名证书：

![](img/3df6e19b-b674-427d-8fd8-e2b40dbfae9a.png)Arista EOS explorer

您将进入一个探索页面，在那里您可以输入 CLI 命令并获得请求正文的良好输出。例如，如果我想查看如何为`show version`制作请求正文，这就是我将从探索器中看到的输出：

![](img/8579af8e-eb72-4dc3-baab-1ab7d8c937be.png)Arista EOS explorer viewer

概述链接将带您进入示例用途和背景信息，而命令文档将作为 show 命令的参考点。每个命令引用都将包含返回值字段名称、类型和简要描述。Arista 的在线参考脚本使用 jsonrpclib ([`github.com/joshmarshall/jsonrpclib/`](https://github.com/joshmarshall/jsonrpclib/))，这是我们将使用的。然而，截至撰写本书时，它依赖于 Python 2.6+，尚未移植到 Python 3；因此，我们将在这些示例中使用 Python 2.7。

在您阅读本书时，可能会有更新的状态。请阅读 GitHub 拉取请求 ([`github.com/joshmarshall/jsonrpclib/issues/38`](https://github.com/joshmarshall/jsonrpclib/issues/38)) 和 GitHub README ([`github.com/joshmarshall/jsonrpclib/`](https://github.com/joshmarshall/jsonrpclib/)) 以获取最新状态。

安装很简单，使用`easy_install`或`pip`：

```py
$ sudo easy_install jsonrpclib
$ sudo pip install jsonrpclib
```

# eAPI 示例

然后，我们可以编写一个名为`eapi_1.py`的简单程序来查看响应文本：

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

请注意，由于这是 Python 2，在脚本中，我使用了`from __future__ import print_function`以便未来迁移更容易。与`ssl`相关的行适用于 Python 版本 > 2.7.9。更多信息，请参阅[`www.python.org/dev/peps/pep-0476/`](https://www.python.org/dev/peps/pep-0476/)。

这是我从先前的`runCms()`方法收到的响应：

```py
 [{u'memTotal': 3978148, u'internalVersion': u'4.16.6M-
 3205780.4166M', u'serialNumber': u'<omitted>', u'systemMacAddress':
 u'<omitted>', u'bootupTimestamp': 1465964219.71, u'memFree': 
 277832, u'version': u'4.16.6M', u'modelName': u'DCS-7050QX-32-F', 
 u'isIntlVersion': False, u'internalBuildId': u'373dbd3c-60a7-4736-
 8d9e-bf5e7d207689', u'hardwareRevision': u'00.00', u'architecture': 
 u'i386'}]
```

如您所见，结果是包含一个字典项的列表。如果我们需要获取序列号，我们可以简单地引用项目编号和键：

```py
     print('Serial Number: ' + response[0]['serialNumber'])
```

输出将仅包含序列号：

```py
$ python eapi_1.py
Serial Number: <omitted>
```

为了更熟悉命令参考，请点击 eAPI 页面上的命令文档链接，并将您的输出与文档中 show version 的输出进行比较。

如前所述，与 REST 不同，JSON-RPC 客户端使用相同的 URL 端点来调用服务器资源。您可以从前面的示例中看到，`runCmds()`方法包含一系列命令。对于配置命令的执行，您可以遵循相同的框架，并通过一系列命令配置设备。

这是一个名为`eapi_2.py`的配置命令示例。在我们的示例中，我们编写了一个函数，该函数将交换机对象和命令列表作为属性：

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

      switch = Server("https://admin:arista@192.168.199.158/command-
    api")

      commands = ["enable", "configure", "interface ethernet 1/3", 
    "switchport acc ess vlan 100", "end", "write memory"]

     response = runAristaCommands(switch, commands)
     pprint.pprint(response)

```

这是命令执行的输出：

```py
$ python2 eapi_2.py
[{}, {}, {}, {}, {}, {u'messages': [u'Copy completed successfully.']}]
```

现在，快速检查`switch`以验证命令的执行：

```py
arista1#sh run int eth 1/3
interface Ethernet1/3
    switchport access vlan 100
arista1# 
```

总的来说，eAPI 非常简单直接，易于使用。大多数编程语言都有类似于`jsonrpclib`的库，它们抽象了 JSON-RPC 的内部。通过几个命令，您就可以开始将 Arista EOS 自动化集成到您的网络中。

# Arista Pyeapi 库

Python 客户端 Pyeapi（[`pyeapi.readthedocs.io/en/master/index.html`](http://pyeapi.readthedocs.io/en/master/index.html)）库是一个原生的 Python 库，包装了 eAPI。它提供了一组绑定来配置 Arista EOS 节点。为什么我们需要 Pyeapi，当我们已经有 eAPI 了呢？在 Python 环境中选择 Pyeapi 还是 eAPI 主要是一个判断调用。

然而，如果你处于非 Python 环境中，eAPI 可能是一个不错的选择。从我们的例子中可以看出，eAPI 的唯一要求是一个支持 JSON-RPC 的客户端。因此，它与大多数编程语言兼容。当我刚开始进入这个领域时，Perl 是脚本和网络自动化的主导语言。仍然有许多企业依赖 Perl 脚本作为他们的主要自动化工具。如果你处于一个公司已经投入大量资源且代码基础是另一种语言而不是 Python 的情况下，使用支持 JSON-RPC 的 eAPI 可能是一个不错的选择。

然而，对于那些更喜欢用 Python 编程的人来说，一个原生的`Python`库意味着在编写我们的代码时更自然。它确实使得扩展 Python 程序以支持 EOS 节点更容易。它也使得更容易跟上 Python 的最新变化。例如，我们可以使用 Python 3 与 Pyeapi！

在撰写本书时，Python 3（3.4+）支持正式是一个正在进行中的工作，如文档中所述（[`pyeapi.readthedocs.io/en/master/requirements.html`](http://pyeapi.readthedocs.io/en/master/requirements.html)）。请查看文档以获取更多详细信息。

# Pyeapi 安装

使用 pip 进行安装非常简单：

```py
$ sudo pip install pyeapi
$ sudo pip3 install pyeapi
```

请注意，pip 还将安装 netaddr 库，因为它是 Pyeapi 的规定要求的一部分（[`pyeapi.readthedocs.io/en/master/requirements.html`](http://pyeapi.readthedocs.io/en/master/requirements.html)）。

默认情况下，Pyeapi 客户端将在您的主目录中查找一个 INI 风格的隐藏文件（前面带有一个句点）称为`eapi.conf`。您可以通过指定`eapi.conf`文件路径来覆盖这种行为，但通常最好将连接凭据与脚本本身分开。您可以查看 Arista Pyeapi 文档（[`pyeapi.readthedocs.io/en/master/configfile.html#configfile`](http://pyeapi.readthedocs.io/en/master/configfile.html#configfile)）以获取文件中包含的字段。这是我在实验室中使用的文件：

```py
cat ~/.eapi.conf
[connection:Arista1]
host: 192.168.199.158
username: admin
password: arista
transport: https
```

第一行`[connection:Arista1]`包含了我们将在 Pyeapi 连接中使用的名称；其余字段应该是相当容易理解的。您可以将文件锁定为只读，供用户使用此文件：

```py
$ chmod 400 ~/.eapi.conf
$ ls -l ~/.eapi.conf
-r-------- 1 echou echou 94 Jan 27 18:15 /home/echou/.eapi.conf
```

# Pyeapi 示例

现在，我们准备好查看用法了。让我们通过在交互式 Python shell 中创建一个对象来连接到 EOS 节点：

```py
Python 3.5.2 (default, Nov 17 2016, 17:05:23)
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pyeapi
>>> arista1 = pyeapi.connect_to('Arista1')
```

我们可以执行 show 命令到节点并接收输出：

```py
>>> import pprint
>>> pprint.pprint(arista1.enable('show hostname'))
[{'command': 'show hostname',
 'encoding': 'json',
 'result': {'fqdn': 'arista1', 'hostname': 'arista1'}}]
```

配置字段可以是单个命令，也可以是使用`config()`方法的命令列表：

```py
>>> arista1.config('hostname arista1-new')
[{}]
>>> pprint.pprint(arista1.enable('show hostname'))
[{'command': 'show hostname',
 'encoding': 'json',
 'result': {'fqdn': 'arista1-new', 'hostname': 'arista1-new'}}]
>>> arista1.config(['interface ethernet 1/3', 'description my_link'])
[{}, {}]
```

请注意，命令缩写（`show run`与`show running-config`）和一些扩展将不起作用：

```py
>>> pprint.pprint(arista1.enable('show run'))
Traceback (most recent call last):
...
 File "/usr/local/lib/python3.5/dist-packages/pyeapi/eapilib.py", line 396, in send
 raise CommandError(code, msg, command_error=err, output=out)
pyeapi.eapilib.CommandError: Error [1002]: CLI command 2 of 2 'show run' failed: invalid command [incomplete token (at token 1: 'run')]
>>>
>>> pprint.pprint(arista1.enable('show running-config interface ethernet 1/3'))
Traceback (most recent call last):
...
pyeapi.eapilib.CommandError: Error [1002]: CLI command 2 of 2 'show running-config interface ethernet 1/3' failed: invalid command [incomplete token (at token 2: 'interface')]
```

然而，您总是可以捕获结果并获得所需的值：

```py
>>> result = arista1.enable('show running-config')
>>> pprint.pprint(result[0]['result']['cmds']['interface Ethernet1/3'])
{'cmds': {'description my_link': None, 'switchport access vlan 100': None}, 'comments': []}
```

到目前为止，我们一直在使用 eAPI 进行 show 和配置命令。Pyeapi 提供了各种 API 来使生活更轻松。在下面的示例中，我们将连接到节点，调用 VLAN API，并开始对设备的 VLAN 参数进行操作。让我们来看一下：

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
>>> vlans.create(10)
True
>>> vlans.getall()
{'1': {'vlan_id': '1', 'trunk_groups': [], 'state': 'active', 'name': 'default'}, '10': {'vlan_id': '10', 'trunk_groups': [], 'state': 'active', 'name': 'VLAN0010'}}
>>> vlans.set_name(10, 'my_vlan_10')
True
```

让我们验证一下设备上是否创建了 VLAN 10：

```py
arista1#sh vlan
VLAN Name Status Ports
----- -------------------------------- --------- -------------------------------
1 default active
10 my_vlan_10 active
```

正如你所看到的，EOS 对象上的 Python 本机 API 确实是 Pyeapi 在 eAPI 之上的优势所在。它将底层属性抽象成设备对象，使代码更清晰、更易读。

要获取不断增加的 Pyeapi API 的完整列表，请查阅官方文档（[`pyeapi.readthedocs.io/en/master/api_modules/_list_of_modules.html`](http://pyeapi.readthedocs.io/en/master/api_modules/_list_of_modules.html)）。

总结本章，让我们假设我们重复了前面的步骤足够多次，以至于我们想写另一个 Python 类来节省一些工作。`pyeapi_1.py`脚本如下所示：

```py
      #!/usr/bin/env python3

      import pyeapi

      class my_switch():

          def __init__(self, config_file_location, device):
               # loads the config file
               pyeapi.client.load_config(config_file_location)
               self.node = pyeapi.connect_to(device)
               self.hostname = self.node.enable('show hostname')[0]
    ['result']['host name']
              self.running_config = self.node.enable('show running-
    config')

           def create_vlan(self, vlan_number, vlan_name):
               vlans = self.node.api('vlans')
               vlans.create(vlan_number)
               vlans.set_name(vlan_number, vlan_name) 
```

从脚本中可以看出，我们自动连接到节点并在连接时设置主机名和`running_config`。我们还创建了一个使用`VLAN` API 创建 VLAN 的类方法。让我们在交互式 shell 中尝试运行脚本：

```py
Python 3.5.2 (default, Nov 17 2016, 17:05:23)
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pyeapi_1
>>> s1 = pyeapi_1.my_switch('/tmp/.eapi.conf', 'Arista1')
>>> s1.hostname
'arista1'
>>> s1.running_config
[{'encoding': 'json', 'result': {'cmds': {'interface Ethernet27': {'cmds': {}, 'comments': []}, 'ip routing': None, 'interface face Ethernet29': {'cmds': {}, 'comments': []}, 'interface Ethernet26': {'cmds': {}, 'comments': []}, 'interface Ethernet24/4': h.': 
<omitted>
'interface Ethernet3/1': {'cmds': {}, 'comments': []}}, 'comments': [], 'header': ['! device: arista1 (DCS-7050QX-32, EOS-4.16.6M)n!n']}, 'command': 'show running-config'}]
>>> s1.create_vlan(11, 'my_vlan_11')
>>> s1.node.api('vlans').getall()
{'11': {'name': 'my_vlan_11', 'vlan_id': '11', 'trunk_groups': [], 'state': 'active'}, '10': {'name': 'my_vlan_10', 'vlan_id': '10', 'trunk_groups': [], 'state': 'active'}, '1': {'name': 'default', 'vlan_id': '1', 'trunk_groups': [], 'state': 'active'}}
>>>
```

# 供应商中立库

有几个优秀的供应商中立库，比如 Netmiko（[`github.com/ktbyers/netmiko`](https://github.com/ktbyers/netmiko)）和 NAPALM（[`github.com/napalm-automation/napalm`](https://github.com/napalm-automation/napalm)）。因为这些库并非来自设备供应商，它们有时会慢一步来支持最新的平台或功能。然而，由于这些库是供应商中立的，如果你不喜欢为你的工具绑定供应商，那么这些库是一个不错的选择。使用这些库的另一个好处是它们通常是开源的，所以你可以为新功能和错误修复做出贡献。

另一方面，由于这些库是由社区支持的，如果你需要依赖他人来修复错误或实现新功能，它们可能并不是理想的选择。如果你有一个相对较小的团队，仍然需要遵守工具的某些服务级保证，你可能最好使用供应商支持的库。

# 总结

在本章中，我们看了一些从思科、Juniper 和 Arista 管理网络设备的各种方法。我们既看了与 NETCONF 和 REST 等直接通信，也使用了供应商提供的库，比如 PyEZ 和 Pyeapi。这些都是不同的抽象层，旨在提供一种无需人工干预就能编程管理网络设备的方式。

在第四章中，*Python 自动化框架- Ansible 基础*，我们将看一下一个更高级的供应商中立抽象框架，称为**Ansible**。Ansible 是一个用 Python 编写的开源通用自动化工具。它可以用于自动化服务器、网络设备、负载均衡器等等。当然，对于我们的目的，我们将专注于使用这个自动化框架来管理网络设备。
