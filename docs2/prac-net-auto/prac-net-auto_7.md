# 网络自动化中的 SDN 概念

如我们迄今为止所看到的，有许多场景可以自动化网络，从日常或常规任务，到从单一基于控制器的架构管理基础设施。在这些概念的基础上，我们将现在获得一些关于在 **软件定义网络**（**SDNs**）中工作的额外见解，并查看一些与云平台一起工作的示例。

我们将要介绍的一些关键组件包括：

+   云平台自动化

+   网络自动化工具

+   基于控制器的网络架构

+   可编程网络设备

# 管理云平台

我们可以通过 Python 中的网络自动化技术来处理各种云提供商。从处理云实例，到启动新的虚拟机，控制完全访问，如 ACLs，以及创建特定网络层任务，如 VPN 和每个实例的网络配置，我们可以使用 Python 中可用的连接器或 API 自动化几乎所有事情。让我们看看在最受欢迎的云平台 **Amazon Web Services**（**AWS**）上使用 Python 的一些基本配置和连接。

AWS 通过其 SDK Boto 3 提供了一个广泛的 API。Boto 3 提供两种类型的 API 供使用，一种用于与直接 AWS 服务交互的低级 API 集合，以及一种用于与 AWS 快速交互的 Python 友好的高级 API 集合。除了 Boto 3 之外，我们还需要 AWS CLI，它用作 **命令行界面**（**CLI**）从本地机器与 AWS 交互。从 CLI 角度来看，这就像 DOS 对 Windows 一样是一个基于 CLI 的工具。

AWS CLI 和 Boto 3 的安装都使用 `pip` 完成：

+   要从 AWS CLI 安装，请使用以下命令：

```py
pip install awscli
```

+   要从 Boto 3 安装，请使用以下命令：

```py
pip install boto3
```

安装完成后，这些包就准备好使用了。然而，我们需要在 AWS 网络管理控制台中配置一个访问密钥，它将具有某种程度的限制（我们将在创建访问密钥时定义）。

让我们快速设置一个新的访问密钥，以便从我们的本地机器管理 AWS：

1.  登录 AWS 网络控制台，选择 IAM 选项：

![图片](img/d6dd3bb0-93e1-416b-b09b-dde6bf496236.jpg)

1.  点击添加用户以创建以下所示的用户名和密码对：

![图片](img/c34f5d6f-8428-4a07-b5db-5a6fd36980d4.jpg)

1.  选择用户名并确保勾选程序访问权限，以获取用于我们 Python 调用的访问 ID 和密钥：

![图片](img/201f9356-8f61-4288-8a8d-049d4ac83921.jpg)

1.  我们还需要用户成为某个特定组的成员（出于安全限制）。在我们的例子中，我们将其设置为管理员组的成员，该组在 AWS 实例上拥有完全权限：

![图片](img/e6bc8d59-f148-4f45-8f9a-2fb4acf477ad.jpg)

1.  如果我们正确地选择了选项，将创建一个用户，用户名为我们选择的 (`booktest`)，并带有访问密钥和秘密访问密钥：

![图片](img/d2158071-32a0-469e-911e-fbc96cb771bf.jpg)

1.  一旦我们有了这个密钥，我们就回到我们的 Python 安装，在命令提示符下调用 AWS CLI 命令`aws configure`：

![图片](img/b067f6ad-1546-486b-b4d8-baf6c6038c1d.jpg)

1.  根据提出的问题，我们从 AWS 网络控制台获取值并将其粘贴到 CLI 中。`默认输出格式`可以是`text`或`json`。然而，为了我们的自动化目的和与 Python 一起工作，我们会选择`json`而不是`text`。

一旦我们完成了这个后端配置，我们就可以通过在 Python 中调用 Boto 3 API 来测试我们的脚本。

让我们看看获取当前 AWS 账户中所有运行实例的示例，其中我们拥有密钥：

```py
import boto3
ec2 = boto3.resource('ec2')
for instance in ec2.instances.all():
    print (instance)
    print (instance.id, instance.state)
```

由于我们已通过`aws configure` CLI 命令配置了后端凭据和密钥，因此我们不需要在我们的脚本中指定任何凭据。

上述代码的输出如下：

![图片](img/0ab7156c-653b-4bbf-9abc-0da74b8ae3a2.jpg)

正如我们在前面的输出中看到的，我们得到了两个实例，它们是具有其实例 ID 的 EC2 实例。此外，我们还得到了一些当前配置实例的其他参数。在某些情况下，如果我们不想使用当前的预配置密钥，我们可以通过以下方式直接将值传递给 Boto 3 函数来调用 Python 程序：

```py
import boto3

aws_access_key_id = 'accesskey'
aws_secret_access_key = 'secretaccesskey'
region_name = 'us-east-2'

ec2 = boto3.client('ec2',aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key,region_name=region_name)
```

让我们看看获取每个实例的私有 IP 地址和实例 ID 的另一个示例：

```py
import boto3

ec2 = boto3.client('ec2')
response = ec2.describe_instances()
for item in response['Reservations']:
    for eachinstance in item['Instances']:
        print (eachinstance['InstanceId'],eachinstance['PrivateIpAddress'])
```

上述代码给出了以下输出：

![图片](img/e72a2534-f0bc-4b04-8209-2037ee842087.jpg)

使用 Boto 3 API，我们也可以在我们的订阅中启动新的实例。让我们看看使用 Boto 3 启动一个新的**虚拟机**（**VM**）的最终示例。

在我们调用 Python 来启动新的 VM 之前，我们需要选择用于实例的哪个**亚马逊机器镜像**（**AMI**）图像。要找出 AMI 图像值，我们需要打开如以下所示的 AWS 网络控制台中的 AMI：

![图片](img/b89b8f02-0d74-4548-968d-aa99cb7e16ce.jpg)

一旦我们完成了 AMI 的最终确定，我们就进入了简单的部分，即启动新的 VM：

```py
import boto3
ec2 = boto3.resource('ec2')
ec2.create_instances(ImageId='amid-imageid', MinCount=1, MaxCount=5)
```

脚本执行需要一些时间，结果值将是基于所选 AMI 图像 ID 配置的所有参数的实例。同样，我们可以使用 Boto 3 启动各种类型的实例，甚至可以启动新的安全过滤器，并确保我们拥有云控制自动化。

# 可编程网络设备

回顾历史实现，我们有一套固定的硬件或网络，专为向最终用户提供服务而设计。最终用户也有有限的连接选项来访问有限的网络或连接资源。随着用户数量的增加，一个简单的解决方案是添加额外的硬件或网络设备。然而，随着不同终端用户设备（如手机）的激增，以及最终用户对高数据需求和正常运行时间的要求，管理不断增加的硬件和额外连接变得复杂。

简单的设备故障或电缆故障可能会影响整个连接的硬件或网络设备，这将为最终用户造成广泛的中断，导致生产力和信任度下降。想象一下，一个大型**互联网服务提供商**（**ISP**）频繁出现故障，每次故障都会影响大量企业和家庭用户。如果一个新的 ISP 带着可靠性作为其独特的卖点进入市场，人们将毫不犹豫地跳转到新的提供商。实际上，这可能导致业务损失，最终，由于当前用户群中可靠性和信任度的下降，导致早期提供商关闭。

为了处理这种情况，出现了一种解决方案，即使用同一套设备或网络硬件，在相同的硬件平台上执行不同的功能。这是通过 SDN 和**可编程网络**（**PNs**）的组合实现的。

SDN 负责控制平面配置，以便数据能够自动重新路由到从特定源到目的地的最佳可用路径。例如，假设我们需要从源 A 到达目的地 D。到达 D 的最佳路径是 A -> C -> D。

现在，在处理传统流量流的情况下，除非 C 出现故障或实际上关闭，否则流量不会从 A -> B -> D 流动（除非在每个网络设备/设备上执行特殊的复杂配置）。在 SDN 环境中，使用 OpenFlow 作为底层协议，控制器将检测 A -> C -> D 路径上的任何问题，并根据某些问题（如路径中的数据包丢失或拥塞），做出智能决策，确保有新的路径让数据从 A -> B -> D 流动。

就像我们在本例中看到的那样，即使 C 没有物理问题，SDN 也已经负责确定数据流动的最佳路径，这有效地实现了对最终用户来说最佳的可实现性能，并且具有可靠性。

PN 是一个补充，它是一组网络层的硬件设备，可以根据需求编程以不同的方式行为。想象一下，一个交换机通过编写一段代码来改变其功能，充当一个路由器。假设我们迎来了大量新的终端用户，并且需要在网络层拥有高交换能力。一些设备现在可以充当交换机而不是路由器。这确保了两方面的好处：

+   根据需求和需求使用同一套硬件，硬件可以重用于处理新的场景，而不会通过添加额外的硬件集来增加网络的复杂性。

+   通过增加通过同一组设备安全传输流量的额外能力，更好地控制流量。这是通过为从一组特定设备流向流量的流量添加 ACLs 来实现的，甚至确保设备只处理特定类型的流量，并将剩余流量发送到专门编程以处理该特定流量的其他设备。想象一下，视频和语音流量从不同的设备流向，以确保使用我们目前拥有的相同硬件在特定设备上实现最佳性能和负载。

可编程网络设备（PNs）的一个主要组成部分是使用由各种网络供应商（如 Cisco、Arista 和 Juniper）提供的 API。通过调用这些 API，我们可以确保来自特定供应商的每个设备可以轻松地相互通信（以统一格式交换信息），并且可以根据 API 调用改变特定硬件的行为。在当今市场上一个常见的例子是 Cisco Nexus 9000 系列设备。这些是模块化或固定交换机（具有不同的变体），通过使用 OpenFlow，我们能够根据动态需求以编程方式改变它们的行为。

以这个开关为例，直接访问**专用集成电路**（**ASIC**）芯片级编程也被暴露出来，这确保了 ASIC 也可以根据需求以及软件级别的变化进行编程。在 SDN 部署的情况下，控制器可以利用 OpenFlow 以及这些交换机上暴露的 API 来控制这些交换机的角色。

Cisco 还为多个设备（主要在 Nexus 平台上）提供了一种**电源开启自动配置**（**PoAP**）功能，这有助于在新的设备启动时立即实现自动配置和部署。这个过程的基本概述是，如果一个启用了 PoAP 功能的 Nexus 设备启动后无法找到任何启动配置，它将在网络中定位一个**动态主机配置协议**（**DHCP**）服务器，并使用从该 DHCP 服务器获得的 IP 地址和 DNS 信息进行引导。它还会获取一个在设备上执行的定制脚本，该脚本包含下载和安装相关软件映像文件以及该设备的特定配置的指令。

这种类型功能的一个大优势是，我们只需打开电源并将其连接到具有 DHCP 功能以获取网络中新设备相关信息的网络，就可以在一到两分钟内启动新设备。想象一下，与需要数小时人工干预的传统方式相比，现在启动路由器的方式，路由器无需任何人工干预就能自行处理。

类似地，使用 API（**NX-API** 是用于 **Nexus API** 的底层术语），Cisco 也提供了关于数据包流和监控的更好可见性，并且可以使用任何语言（如 Python）编写的简单脚本，根据通过调用这些 API 返回的结果修改路径和流量。

以另一个例子为例，我们有一个网络设备供应商 Arista。Arista 引入了 Arista **可扩展操作系统**（EOS），这是一个高度模块化和基于 Linux 的网络操作系统。使用 Arista EOS，管理多个设备变得容易，因为它具有提供广泛 API（基于 Linux 内核和与 Arista 相关的附加 API）的能力，并调用 API 为各种供应商配置和部署多个终端节点。Arista 引入的一个名为 **智能系统升级**（SSU）的功能确保，当我们对 Arista 设备进行操作系统升级时，它将使用升级的操作系统版本重新启动其服务，但不会重启，以确保在升级期间最小化流量中断。这些功能确保即使在数据中心或多个设备上同时推出新的补丁和操作系统升级时，我们也有弹性和正常运行时间。

Arista EOS 通过提供一套 API 调用 **eAPI** 来为通过 API 管理的设备提供扩展功能。eAPI 可以通过从任何脚本或可编程语言中调用 eAPI 框架来配置 Arista 设备。让我们看看如何使用 eAPI 管理一个 Arista 交换机的非常基本的例子。

我们需要在 Arista 交换机上配置 eAPI：

```py
Arista> enable
Arista# configure terminal
Arista(config)# management api http-commands
Arista(config-mgmt-api-http-cmds)# no shutdown
Arista(config-mgmt-api-http-cmds)# protocol http
Arista(config-mgmt-api-http-cmds)#end
```

这确保了 Arista eAPI 功能在路由器上已启用，我们可以使用 HTTP 协议与 API 进行交互。我们还可以通过使用命令 `protocol https` 在 HTTPS 上可用的 eAPI 选项之间切换。

为了验证我们的配置是否正确，我们使用命令 `show management api http-commands`，如下所示：

```py
Arista# show management api http-commands 
Enabled: Yes 
HTTPS server: shutdown, set to use port 443 
HTTP server: running, set to use port 80
```

我们可以使用浏览器命令 `http://<路由器 IP>` 来检查 eAPI 框架是否现在可以访问。

Arista 的一些例子展示了我们使用 URL（在这种情况下我们启用了 HTTPS 而不是 HTTP）得到的输出：

![](img/8f8c0913-3109-4155-98bc-94f1f4da9513.jpg)

在这里，我们看到一组传递的命令（`show version` 和 `show hostname`），API 的响应确认了结果集。此外，命令响应文档选项卡显示了我们可以用于参考的可用 API：

![](img/8693d414-a4dc-4b05-b348-f0e4ab76c21e.jpg)

让我们看看如何在 Python 中调用相同的操作：

作为先决条件，我们需要安装 `jsonrpclib`，可以在 URL [`pypi.python.org/pypi/jsonrpclib.`](https://pypi.python.org/pypi/jsonrpclib) 找到。这个库用于解析 JSON 格式的 **远程过程调用**（RPC）。一旦完成，以下代码将产生与使用浏览器得到相同的一组值：

```py
from jsonrpclib import Server 
switch = Server( "https://admin:admin@172.16.130.16/command-api" ) 
response = switch.runCmds( 1, [ "show hostname" ] ) 
print ("Hello, my name is: ", response[0][ "hostname" ] )
response = switch.runCmds( 1, [ "show version" ] ) 
print ("My MAC address is: ", response[0][ "systemMacAddress" ] )
print ("My version is: ", response[0][ "version" ])
```

上述代码给出了以下输出：

```py
Hello, my name is: Arista 
My MAC address is: 08:00:27:0e:bf:31 
My version is: 4.14.5F
```

以类似的方式，Arista 也引入了一个 Python 库，可以作为`jsonrpclib`的替代品。可以在 URL [`pypi.python.org/pypi/pyeapi`](https://pypi.python.org/pypi/pyeapi)找到的`pyeapi`库，是 Arista EOS eAPI 的 Python 包装器。按照示例，以下是使用`pyeapi`访问相同设备集的方法。

从开发者页面，这里有一个示例，展示了我们如何使用`pyeapi`在 Arista 上进行 API 处理：

```py
>>> from pprint import pprint as pp
>>> node = pyeapi.connect(transport='https', host='veos03', username='eapi', password='secret', return_node=True)
>>> pp(node.enable('show version'))
[{'command': 'show version',
  'encoding': 'json',
  'result': {u'architecture': u'i386',
             u'bootupTimestamp': 1421765066.11,
             u'hardwareRevision': u'',
             u'internalBuildId': u'f590eed4-1e66-43c6-8943-cee0390fbafe',
             u'internalVersion': u'4.14.5F-2209869.4145F',
             u'memFree': 115496,
             u'memTotal': 2028008,
             u'modelName': u'vEOS',
             u'serialNumber': u'',
             u'systemMacAddress': u'00:0c:29:f5:d2:7d',
             u'version': u'4.14.5F'}}]
```

观察 Cisco 和 Arista（它们是云和 SDN 市场的主要参与者），我们可以结合 Arista eAPI 和 Cisco NX-API 来管理我们的整个数据中心库存，并在一些任务上工作，如新设备的提供或现有设备的升级，这些任务对业务流程的影响最小或没有影响，从而确保可扩展性、可靠性和正常运行时间。

# 基于控制器的网络布线

随着我们走出每个物理路径都是连接和设计用来从一个点到另一个点传输流量的传统硬件时代，以及数据包从一台设备到达另一台设备的可用性有限，SDN 确保我们有网络布线，以便我们的数据在源和目的地之间传输。

**网络布线**是一组通过一个共同的控制器连接在一起的不同网络设备，确保网络中的每个组件都优化了发送流量到每个节点的性能。底层的交换布线，即带有端口（如以太网、ATM 和 DSL）的物理交换板，也由控制器控制和编程，可以确保（通过创建路径或特定的端口）特定的数据类型可以穿越到达到其目的地。

在典型的网络设计中，我们有第 2 层（或交换域）和第 3 层（或路由域）。如果我们没有基于控制器的方案，每个网络组件都可以从其下一个连接的组件学习流量的行为（如第 2 层的**生成树协议**（**STP**））或某些路由协议（如第 3 层的 OSPF）。在这种情况下，每个设备都充当自己的控制器，并且只能有限地看到它直接连接的设备（也称为**邻居设备**）。在任何设备上都没有整个网络的单一视图，并且此外，每个组件（或单个控制器）都充当其邻居设备的单点故障。任何组件的故障都可能导致其邻居设备重新收敛，甚至由于连接组件的故障而隔离。

与基于控制器的环境相比，理论上每个设备拥有的连接数与其连接的端口数相同。因此，如果我们考虑在基于控制器的环境中连接的三个设备，由于它们之间的物理连接，每个设备之间都有多个连接。在组件（或设备）故障的情况下，控制器可以迅速做出智能决策，重新配置新路径，并改变其他两个网络设备的行为，以确保对交通的最小干扰，保持所有其他可用链路上的相同吞吐量和分布式负载。从理论上讲，控制器消除了每个设备的控制平面行为，并确保控制器管理的每个设备上更新了优化的转发表（用于将数据转发到特定目的地）。这是因为控制器开始充当主要组件，它可以看到每个设备，包括每个设备的每个入口和出口点以及从每个受管理的网络设备流过的数据类型的粒度。

根据厂商，像思科（拥有其开放网络环境）、瞻博网络（拥有其 QFabric 交换机）和艾维雅（拥有其 VENA 交换机）这样的主要厂商已经提供了充当控制器或配置为被控制器管理的功能。此外，随着控制器到管理器网络组件的引入，每个网络设备现在可以虚拟地成为客户端，控制器做出所有智能决策，从学习到转发表。

控制器充当多厂商网络设备和网络任务之间的抽象层。作为终端用户，某人可以配置控制器执行的具体任务，并且，使用来自不同厂商的底层 API 模型（使用 JSON 或 XML），控制器可以将这些具体任务转换为各种厂商特定的 API 调用，通过使用这些 API 向每个厂商设备发送这些具体指令来配置设备。**应用策略基础设施控制器**（**APIC**）组件负责控制和编程每个网络设备组件上的布线。

让我们看看 Cisco APIC 的示例以及一些基本用法。Cisco APIC 用于管理、自动化、监控和编程**应用中心基础设施**（**ACI**）。ACI 是一组对象，每个对象代表一个租户。租户可以称为基于业务分类的特定客户群、组或业务单元。例如，一个单一的组织可能将其整个基础设施转换为一个单一租户，而一个组织可以根据其功能（如 HR 和财务）将租户分离出来。租户可以进一步细分为上下文，每个上下文作为一个单独的转发平面，因此每个上下文中都可以使用相同的 IP 地址集，因为在每个上下文中，IP 地址集将被不同地处理。

上下文中包含**端点**（**EPs**）和**端点组**（**EPGs**）。这些 EPs 是物理组件，如硬件 NICs，而 EPGs 是 DNSs、IP 地址等项目的集合，它们为特定应用程序（如 Web 应用程序）定义了类似的功能。

使用 APIC 进行编程时，所需的主要组件如下：

+   **APIC Rest Python Adaptor**（**ARYA**）

这是一个由 Cisco 创建的工具，用于将 APIC 返回的 XML 或 JSON 对象转换为直接 Python 代码。在底层，它利用 COBRA SDK 执行此任务。这可以通过`pip install arya`在 Python 中安装。

+   **ACI SDK**

这是一个包含直接调用控制器 API 的 API 的 SDK。我们需要安装`acicobra`，可以在以下网址找到：[`www.cisco.com/c/en/us/td/docs/switches/datacenter/aci/apic/sw/1-x/api/python/install/b_Install_Cisco_APIC_Python_SDK_Standalone.html`](https://www.cisco.com/c/en/us/td/docs/switches/datacenter/aci/apic/sw/1-x/api/python/install/b_Install_Cisco_APIC_Python_SDK_Standalone.html)，以便将其调用到 Python 中。

一旦安装了此软件，以下是一些来自 Cisco 的示例，可以在以下网址找到：[`github.com/CiscoDevNet/python_code_samples_network/blob/master/acitoolkit_show_tenants/aci-show-tenants.py`](https://github.com/CiscoDevNet/python_code_samples_network/blob/master/acitoolkit_show_tenants/aci-show-tenants.py)。这可以帮助我们理解创建对象：

```py
#!/usr/bin/env python
"""
Simple application that logs on to the APIC and displays all
of the Tenants. 
Leverages the DevNet Sandbox - APIC Simulator Always On 
    Information at https://developer.cisco.com/site/devnet/sandbox/available-labs/data-center/index.gsp 

Code sample based off the ACI-Toolkit Code sample
https://github.com/datacenter/acitoolkit/blob/master/samples/aci-show-tenants.py 
"""

import sys
import acitoolkit.acitoolkit as ACI

# Credentials and information for the DevNet ACI Simulator Always-On Sandbox
APIC_URL = "https://sandboxapicdc.cisco.com/"
APIC_USER = "admin"
APIC_PASSWORD = "C1sco12345"

def main():
    """
    Main execution routine
    :return: None
    """

    # Login to APIC
    session = ACI.Session(APIC_URL, APIC_USER, APIC_PASSWORD)
    resp = session.login()
    if not resp.ok:
        print('%% Could not login to APIC')
        sys.exit(0)

    # Download all of the tenants
    print("TENANT")
    print("------")
    tenants = ACI.Tenant.get(session)
    for tenant in tenants:
        print(tenant.name)

if __name__ == '__main__':
    main()
```

在考虑前面的概念时，我们可以增强并确保控制器中的管理节点可以根据应用程序需求进行控制，而不是受硬件限制。这也确保了基础设施现在根据应用程序进行调整，而不是相反，应用程序性能受硬件限制。 

# 网络自动化工具

如前几章所述，我们在自动化网络方面有多种选择。从使用 Netmiko 为任何设备进行基本配置，到使用 Ansible 在网络中的各种设备上部署和创建配置，工程师们有许多选项可以根据不同的需求自动化网络。

Python 在创建自动化场景中被广泛使用，这得益于其对各种厂商和协议的开放社区支持。行业中的几乎每个主要参与者都支持 Python 编程，调整他们自己的工具或任何支持技术。网络自动化的另一个主要方面是为组织需求定制的解决方案。自服务 API 模型是一个很好的起点，确保一些手动完成的任务可以转换为 API，然后可以根据自动化需求利用任何语言。

让我们看看一个可以作为基本指南的示例，以了解自建或自定义自动化工具的优势。在 Cisco 中 `show ip bgp summary` 的输出与在 Juniper 中 `show bgp summary` 的输出相同。现在，作为一个需要验证两个厂商 BGP 的工程师，我需要理解这两个命令并解释输出。

想象一下，通过添加更多具有自己独特方式获取 BGP 输出的厂商，这会变得复杂，网络工程师需要接受多厂商环境的培训，以便能够从每个厂商获取相同类型的输出。

现在，假设我们创建一个 API（例如，`getbgpstatus`），它接受主机名作为输入。后端的 API 足够智能，能够使用 SNMP 获取厂商型号，并根据厂商发送特定的命令（如 Cisco 的 `show ip bgp summary` 或 Juniper 的 `show ip summary`），并将该输出解析为人类可读的格式，例如仅包含该 BGP 邻居的 IP 地址和状态。

例如，我们不是打印 `show ip bgp summary` 或 `show bgp summary` 的原始输出，而是像这样解析输出：

```py
IPaddress1 : Status is UP
IPaddress2 : Status is Down (Active)
```

此输出可以作为 JSON 值返回到 API 的调用中。

因此，假设我们可以通过 `http://localhost/networkdevices/getbgpstatus?device=devicex` 调用 API，后端的 API 将识别 `devicex` 是 Juniper、Cisco 还是其他厂商，并根据这一点，该厂商将获取并解析与该厂商相关的输出。该 API 调用的返回将是类似于前面示例中的 JSON 文本，我们可以在我们的自动化语言中解析它。

让我们看看另一个流行工具 SolarWinds 的基本示例。SolarWinds 有许多方面；它可以自动发现设备（基于 MIBs 和 SNMP），识别厂商，并从设备中获取相关信息。

让我们看看以下截图，了解基本的 SolarWinds 设备管理。SolarWinds 作为试用版免费提供下载。

SolarWinds 设备管理的先决条件如下：

1.  我们需要在 SolarWinds 中添加一个设备，如下所示：

![](img/d9163428-3e8e-45d1-aad5-c056a4ba0d2e.jpg)

如我们所见，SolarWinds 有能力发现设备（使用网络发现），或者我们可以添加一个特定的 IP 地址/主机名，并使用正确的 SNMP 字符串，以便 SolarWinds 识别该设备。

1.  一旦设备被检测到，它将显示为监控节点，如下面的屏幕截图所示：

![](img/8bc6e99c-2d59-43d2-baee-ddd63462bdf9.jpg)

注意 IP 地址（或主机名）旁边的绿色点。这表示节点是活跃的（可到达的），SolarWinds 可以正确地与节点交互。

**在设备发现之后可以执行的其他任务如下：**

一旦我们在 SolarWinds 中有了可用的节点或检测到的节点，以下是一些在 SolarWinds 中可以执行的其他任务（如下面的屏幕截图所示）：

![](img/2be18c1d-a31d-4045-8593-d0a06638e4e8.jpg)

我们已选择 CONFIGS 菜单，在该菜单下我们可以对设备执行配置管理。此外，如以下屏幕截图所示，我们还有创建小脚本的能力（就像我们在这里对`show running config`所做的那样），我们可以使用这些脚本在 SolarWinds 本身针对一组特定的设备执行（如下面的屏幕截图所示）：

![](img/e4aaef49-e77f-4c58-957e-8899d6f19977.jpg)

检索到的结果可以存储为文本文件，或者如果配置了，甚至可以发送回任何电子邮件客户端作为报告。同样，有一些任务（在 SolarWinds 中称为**作业**），可以按计划执行，如下面的屏幕截图所示：

![](img/2935a92f-e6ad-4ea6-9ec5-7aa7ea9da9e1.jpg)

如前一个屏幕截图所示，我们可以从设备下载配置，然后在下一步中选择所有或某些设备并安排作业。这在从以前的日期获取配置或在需要回滚到最后已知良好配置场景时非常有用。此外，有时需要对配置更改进行审计，以了解谁更改了什么以及更改了什么，SolarWinds 可以通过发送报告和警报来扩展这一功能。从编程的角度来看，我们还有额外的能力通过 Python 调用 SolarWinds API 来获取结果。

假设 OrionSDK 已经安装在 Python 中。如果没有，我们可以使用`pip install orionsdk`来安装它。

考虑以下示例：

```py
from orionsdk import SwisClient
import requests

npm_server = 'myserver'
username = "username"
password = "password"

verify = False
if not verify:
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

swis = SwisClient(npm_server, username, password)

results = swis.query("SELECT NodeID, DisplayName FROM Orion.Nodes Where Vendor= 'Cisco'")

for row in results['results']:
    print("{NodeID:<5}: {DisplayName}".format(**row))

```

由于 SolarWinds 支持直接 SQL 查询，我们使用以下查询：

```py
SELECT NodeID, DisplayName FROM Orion.Nodes Where Vendor= 'Cisco'
```

我们试图获取所有具有思科供应商的设备的`NodeID`和`DisplayName`（或设备名称）。一旦我们有了结果，我们将以格式化的方式打印结果。在我们的例子中，输出将是（假设我们的思科设备在 SolarWinds 中已添加为`mytestrouter1`和`mytestrouter2`）：

```py
>>> 
===================== RESTART: C:\a1\checksolarwinds.py =====================
101 : mytestrouter1
102 : mytestrouter2
>>>
```

使用这些自动化工具和 API 中的某些工具，我们可以确保我们的任务专注于实际工作，一些基本或核心任务（如从设备获取值等）被卸载到工具或 API 中处理。

现在我们从头开始创建一个基本的自动化工具，该工具使用 ping 测试来监控任何作为监控工具一部分的节点的可达性。我们可以将其称为 PingMesh 或 PingMatrix，因为该工具将生成一个基于 Web 的矩阵来显示路由器的可达性。

我们将要使用的拓扑如下：

![图片](img/730b75ac-2225-4408-b7e7-0b043391c012.jpg)

在这里，我们将使用四个路由器（`R1`到`R4`），以及`Cloud1`作为我们的监控源。每个路由器将尝试通过 ping 相互可达，并将结果报告给在`Cloud1`上运行的脚本，该脚本将解释结果并通过基于 Web 的 URL 显示网络矩阵。

前面拓扑的解释如下：

1.  我们试图做的是登录到每个路由器（最好是并行登录），从每个源 ping 每个目标，并报告每个目标的可达性状态。

1.  例如，如果我们想手动执行任务，我们会登录到`R1`并尝试从源 ping `R2`、`R3`和`R4`，以检查每个路由器从`R1`的可达性。`Cloud1`上的主脚本（充当控制器）将解释结果并相应地更新网络矩阵。

1.  在我们的案例中，所有路由器（以及控制器）都位于`192.168.255.x`子网中，因此它们可以通过简单的 ping 相互可达。

我们将创建两个独立的 Python 程序（一个作为调用各种节点上命令、从节点获取结果、解释结果并将解析后的数据发送到主程序的库）。主程序将负责调用库，并使用我们得到的结果创建 HTML 网络矩阵。

让我们先创建主程序中要调用的库或程序（我们称之为`getmeshvalues.py`）：

```py
#!/usr/bin/env python
import re
import sys
import os
import time
from netmiko import ConnectHandler
from threading import Thread
from random import randrange
username="cisco"
password="cisco"

splitlist = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

returns = {}
resultoutput={}
devlist=[]
cmdlist=""

def fetchallvalues(sourceip,sourcelist,delay,cmddelay):
    print ("checking for....."+sourceip)
    cmdend=" repeat 10" # this is to ensure that we ping for 10 packets
    splitsublist=splitlist(sourcelist,6) # this is to ensure we open not more than 6 sessions on router at a time
    threads_imagex= []
    for item in splitsublist:
        t = Thread(target=fetchpingvalues, args=(sourceip,item,cmdend,delay,cmddelay,))
        t.start()
        time.sleep(randrange(1,2,1)/20)
        threads_imagex.append(t)

    for t in threads_imagex:
        t.join() 

def fetchpingvalues(devip,destips,cmdend,delay,cmddelay):
    global resultoutput
    ttl="0"
    destip="none"
    command=""
    try:
        output=""
        device = ConnectHandler(device_type='cisco_ios', ip=devip, username=username, password=password, global_delay_factor=cmddelay)
        time.sleep(delay)
        device.clear_buffer()
        for destip in destips:
            command="ping "+destip+" source "+devip+cmdend
            output = device.send_command_timing(command,delay_factor=cmddelay)
            if ("round-trip" in output):
                resultoutput[devip+":"+destip]="True"
            elif ("Success rate is 0 percent" in output):
                resultoutput[devip+":"+destip]="False"
        device.disconnect()
    except:
        print ("Error connecting to ..."+devip)
        for destip in destips:
            resultoutput[devip+":"+destip]="False"

def getallvalues(allips):
    global resultoutput
    threads_imagex= []
    for item in allips:
        #print ("calling "+item)
        t = Thread(target=fetchallvalues, args=(item,allips,2,1,))
        t.start()
        time.sleep(randrange(1,2,1)/30)
        threads_imagex.append(t)
    for t in threads_imagex:
        t.join()
    dnew=sorted(resultoutput.items()) 
    return dnew

#print (getallvalues(["192.168.255.240","192.168.255.245","192.168.255.248","192.168.255.249","4.2.2.2"]))
```

在前面的代码中，我们创建了三个主要函数，我们在一个线程中调用这些函数（用于并行执行）。`getallvalues()`包含我们想要获取数据的 IP 地址列表。然后它将此信息传递给`fetchallvalues()`，带有特定设备信息以并行执行再次获取 ping 值。对于在路由器上执行命令并获取结果，我们调用`fetchpingvalues()`函数。

让我们看看这个代码的结果（通过删除调用函数的注释）。我们需要传递一个列表，其中包含我们想要验证的设备 IP。在我们的案例中，我们所有的有效路由器都在`192.168.255.x`范围内，`4.2.2.2`被用作一个不可达路由器的示例：

```py
print(getallvalues(["192.168.255.240","192.168.255.245","192.168.255.248","192.168.255.249","4.2.2.2"]))
```

前面的代码给出了以下输出：

![图片](img/27db4bff-4119-4e85-9d9f-015ca58654d2.jpg)

如我们在结果中看到的那样，我们从每个节点到其他节点获得了可达性的`True`或`False`。

例如，列表中的第一个项目`('192.168.255.240:192.168.255.240','True')`表示从源`192.168.255.240`到目标`192.168.255.240`（即相同的自 IP）是可达的。同样，同一列表中的下一个项目`('192.168.255.240:192.168.255.245','True')`确认从源 IP`192.168.255.240`到目标`192.168.255.245`我们通过 ping 有可达性。这些信息是创建基于结果的矩阵所必需的。接下来我们看到主要代码，其中我们获取这些结果并创建一个基于 Web 的矩阵页面。

接下来，我们需要创建主文件（我们将其命名为`pingmesh.py`）：

```py
import getmeshvalue
from getmeshvalue import getallvalues

getdevinformation={}
devicenamemapping={}
arraydeviceglobal=[]
pingmeshvalues={}

arraydeviceglobal=["192.168.255.240","192.168.255.245","192.168.255.248","192.168.255.249","4.2.2.2"]

devicenamemapping['192.168.255.240']="R1"
devicenamemapping['192.168.255.245']="R2"
devicenamemapping['192.168.255.248']="R3"
devicenamemapping['192.168.255.249']="R4"
devicenamemapping['4.2.2.2']="Random"

def getmeshvalues():
        global arraydeviceglobal
        global pingmeshvalues
        arraydeviceglobal=sorted(set(arraydeviceglobal))
        tval=getallvalues(arraydeviceglobal)
        pingmeshvalues = dict(tval)

getmeshvalues()

def createhtml():
    global arraydeviceglobal
    fopen=open("C:\pingmesh\pingmesh.html","w") ### this needs to be changed as web path of the html location

    head="""<html><head><meta http-equiv="refresh" content="60" ></head>"""
    head=head+"""<script type="text/javascript">
function updatetime() {
    var x = new Date(document.lastModified);
    document.getElementById("modified").innerHTML = "Last Modified: "+x+" ";
}
</script>"""+"<body onLoad='updatetime();'>"
    head=head+"<div style='display: inline-block;float: right;font-size: 80%'><h4><h4><p id='modified'></p></div>"
    head=head+"<div style='display: inline-block;float: left;font-size: 90%'></h4><center><h2>Network Health Dashboard<h2></div>"
    head=head+"<br><div><table border='1' align='center'><caption><b>Ping Matrix</b></caption>"
    head=head+"<center><br><br><br><br><br><br><br><br>"
    fopen.write(head)
    dval=""
    fopen.write("<tr><td>Devices</td>")
    for fromdevice in arraydeviceglobal:
        fopen.write("<td><b>"+devicenamemapping[fromdevice]+"</b></td>")
    fopen.write("</tr>")
    for fromdevice in arraydeviceglobal:
        fopen.write("<tr>")
        fopen.write("<td><b>"+devicenamemapping[fromdevice]+"</b></td>")
        for todevice in arraydeviceglobal:
            askvalue=fromdevice+":"+todevice
            if (askvalue in pingmeshvalues):
                getallvalues=pingmeshvalues.get(askvalue)
                bgcolor='lime'
                if (getallvalues == "False"):
                    bgcolor='salmon'
            fopen.write("<td align='center' font size='2' height='2' width='2' bgcolor='"+bgcolor+"'title='"+askvalue+"'>"+"<font color='white'><b>"+getallvalues+"</b></font></td>")
        fopen.write("</tr>\n")
    fopen.write("</table></div>")
    fopen.close()

createhtml()

print("All done!!!!")
```

在这种情况下，我们已设置以下映射：

```py
devicenamemapping['192.168.255.240']="R1"
devicenamemapping['192.168.255.245']="R2"
devicenamemapping['192.168.255.248']="R3"
devicenamemapping['192.168.255.249']="R4"
devicenamemapping['4.2.2.2']="Random"
```

最后一个名为`Random`的设备是一个测试设备，它不在我们的网络中，并且为了测试目的不可达。一旦执行，它将创建一个名为`pingmesh.html`的文件，该文件使用标准的 HTML 格式和最后刷新时钟（来自 JavaScript），以确认最后刷新发生的时间。如果我们想从任务计划程序（比如说每五分钟）执行脚本，并且任何人打开 HTML 页面都会知道探测发生的时间。HTML 文件需要放置或保存在一个映射到 Web 文件夹的文件夹中，以便可以使用 URL`http://<服务器>/pingmesh.html`访问。

当执行时，这里是 Python 脚本的输出：

![图片](img/9f1a81d1-b0b1-4f46-94d4-7ab8c48f8af6.jpg)

当将 HTML 文件放置在 Web 映射的 URL 中并调用时，看起来是这样的：

![图片](img/73a2b511-4c71-4ff9-bf5b-bc6376482369.jpg)

如我们所见，在 PingMatrix 中存在一整行和一整列红色，这意味着任何路由器到随机路由器以及从随机路由器到任何路由器的连接都不存在。绿色表示所有其他路由器之间的连接都正常。

此外，我们还在每个单元格上配置了工具提示，将鼠标悬停在该特定单元格上也会显示该特定单元格的源和目标 IP 地址映射，如下面的截图所示：

![图片](img/d5dd0c8b-dd5d-422c-9916-151750f43c15.jpg)

让我们看看另一个截图，其中我们关闭了 R2 以使其不可达：

![图片](img/b5b0edb8-bf44-4e23-8c27-f06c5f150f82.jpg)

现在，正如我们所见，`R2`的整个行和列都是红色，因此 PingMatrix 显示`R2`现在从其他任何地方都不可达，并且`R2`也无法在网络中到达其他人。

让我们看看一个最终的例子，在这个例子中，为了测试目的，我们故意使用扩展的 Cisco ACL 阻止`R2`到`R4`（反之亦然）的 ping 流量，这反过来又报告说`R4`和`R2`在 PingMatrix 中存在可达性问题：

![图片](img/2ee3fc82-d864-4bbe-89ff-b2de50f9edb5.jpg)

如前一个屏幕截图所示，随机路由器仍然显示为红色或错误，因为它不在我们的网络中，但现在它在`R2`和`R4`之间以及`R4`和`R2`之间显示为红色/错误。这让我们快速了解到，即使有多个路径可以到达每个节点，我们仍然存在两个节点之间的连接问题。

根据前面的示例，我们可以增强这个工具，以便轻松监控和理解任何路由/可达性问题，甚至使用我们对网络中所有连接的整体视图来检测链路断开连接问题。PingMesh/Matrix 可以扩展以检查延迟，甚至检查各个节点之间每个连接中的数据包丢失。此外，使用 syslog 或电子邮件功能（Python 中可用于发送 syslog 消息的特定 Python 库，甚至可以从 Python 代码发送电子邮件），在检测到故障或从 Python 脚本本身观察到高延迟的情况下，还可以生成警报或工单。

这个工具可以轻松地成为任何组织中的中心监控工具，工程师可以根据模式（如绿色或红色，以及如果需要其他颜色代码），对实际问题做出决策，并采取主动行动而不是被动行动，以确保网络的高可靠性和正常运行时间。

# 摘要

在本章中，我们学习了 SDN 控制器的基本功能、可编程布线以及一些网络自动化工具。我们还了解了如何与云平台协同工作，并通过一个从 Python 管理 AWS 云的实时示例，理解了我们可以如何使用自动化来控制云操作。

我们对控制器的作用有了深入的理解，并通过一些思科控制器的例子，详细介绍了控制器如何在程序/脚本中编程或调用以执行特定任务。我们还了解了某些流行的网络自动化工具的基础知识，例如 SolarWinds，并创建了一个内部基于 Web 的自动化工具，用于监控我们的网络，称为 PingMatrix 或 PingMesh。
