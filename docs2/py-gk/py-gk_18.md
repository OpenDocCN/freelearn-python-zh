# *第十四章*：使用 Python 进行网络自动化

传统上，网络是由网络专家构建和运营的，这在电信行业仍然是一个趋势。然而，这种管理和运营网络的手动方法速度较慢，有时由于人为错误而导致昂贵的网络中断。此外，为了获得一项新服务（如互联网服务），客户在提交新服务请求后需要等待数天，直到服务准备就绪。基于智能手机和移动应用的经验，您只需点击一下按钮即可启用新服务和应用，客户期望网络服务在几分钟内（如果不是几秒钟内）就绪。使用当前的网络管理方法是不可能的。传统的做法有时也会成为电信服务提供商引入新产品和服务的障碍。

**网络自动化**可以通过提供用于自动化网络管理和操作方面的软件来改善这些情况。网络自动化有助于消除配置网络设备中的人为错误，并通过自动化重复性任务显著降低运营成本。网络自动化有助于加速服务交付，并使电信服务提供商能够引入新服务。

Python 是网络自动化的流行选择。在本章中，我们将探讨 Python 在网络自动化方面的功能。Python 提供了**Paramiko**、**Netmiko**和**NAPALM**等库，可用于与网络设备交互。如果网络设备由**网络管理系统**（**NMS**）或网络控制器/编排器管理，Python 可以使用**REST**或**RESTCONF**协议与这些平台交互。没有监听网络中发生的实时事件，就无法实现端到端网络自动化。这些实时网络事件或实时流数据通常通过**Apache Kafka**等系统提供。我们还将探讨使用 Python 与事件驱动系统交互。

本章将涵盖以下主题：

+   介绍网络自动化

+   与网络设备交互

+   与网络管理系统集成

+   与基于事件的系统协同工作

完成本章后，您将了解如何使用 Python 库从网络设备获取数据，并将配置数据推送到这些设备。这些是任何网络自动化过程的基础步骤。

# 技术要求

本章的技术要求如下：

+   您需要在您的计算机上安装 Python 3.7 或更高版本。

+   您需要在 Python 上安装 Paramiko、Netmiko、NAPALM、ncclient 和 requests 库。

+   您需要能够访问一个或多个支持 SSH 协议的网络设备。

+   您需要能够访问诺基亚开发者实验室，以便能够访问诺基亚的 NMS（称为**网络服务平台**（**NSP**））。

本章的示例代码可以在[`github.com/PacktPublishing/Python-for-Geeks/tree/master/Chapter14`](https://github.com/PacktPublishing/Python-for-Geeks/tree/master/Chapter14)找到。

重要提示

在本章中，您需要访问物理或虚拟网络设备和网络管理系统来执行代码示例。这可能对每个人来说都不可能。您可以使用任何具有类似功能的网络设备。我们将更多地关注实现的 Python 方面，并使其方便地重用代码以用于任何其他设备或管理系统。

我们将首先通过提供网络自动化的介绍来开始我们的讨论。

# 介绍网络自动化

网络自动化是利用技术和软件来自动化管理和运营网络的过程。网络自动化的关键词是*自动化一个过程*，这意味着它不仅涉及部署和配置网络，还包括必须遵循的步骤以实现网络自动化。例如，有时自动化步骤包括在配置推送到网络之前从不同的利益相关者那里获得批准。自动化这样的批准步骤是网络自动化的一部分。因此，网络自动化过程可能因组织而异，取决于每个组织遵循的内部流程。这使得构建一个可以为许多客户开箱即用地执行自动化的单一平台具有挑战性。

有许多正在进行中的努力，旨在从网络设备供应商那里提供必要的平台，以帮助以最小的努力构建定制自动化。这些平台的几个例子包括思科的**网络服务编排器**（**NSO**）、Juniper Networks 的**Paragon Automation**平台和诺基亚的**NSP**。

这些自动化平台的一个挑战是它们通常是供应商锁定。这意味着供应商声称他们的平台可以管理和自动化其他供应商的网络设备，但实现多供应商自动化的过程既繁琐又昂贵。因此，电信服务提供商正在寻找超越供应商平台以实现自动化的方法。**Python**和**Ansible**是电信行业中用于自动化的两种流行编程语言。在我们深入探讨 Python 如何实现网络自动化之前，让我们探讨一下网络自动化的优点和挑战。

## 网络自动化的优点和挑战

我们已经强调了网络自动化的一些优点。我们可以总结关键优点如下：

+   **加速服务交付**：更快地向新客户提供服务，使您能够尽早开始服务计费，并拥有更多满意的客户。

+   **降低运营成本**：通过自动化重复性任务并通过工具和闭环自动化平台监控网络，可以降低网络的运营成本。

+   **消除幽默错误**：大多数网络中断都是由于人为错误造成的。网络自动化可以通过使用标准模板配置网络来消除这一原因。这些模板在生产投入之前都经过了深入评估和测试。

+   **一致的网络设置**：当人类配置网络时，不可能遵循一致的模板和命名约定，这对运营团队管理网络非常重要。网络自动化通过每次使用相同的脚本或模板配置网络，带来了设置网络的统一性。

+   **网络可见性**：通过网络自动化工具和平台，我们可以访问性能监控能力，并可以从端到端可视化我们的网络。通过在它们造成网络流量瓶颈之前检测到流量峰值和资源的高利用率，我们可以进行主动式网络管理。

网络自动化是实现数字化转型的一个必要条件，但实现它有一些成本和挑战。这些挑战如下：

+   **成本**：在构建或定制网络自动化软件时，总是会有成本。网络自动化是一个过程，每年都必须为其设定成本预算。

+   **人的抵触情绪**：在许多组织中，人力资源认为网络自动化是对他们工作的威胁，因此他们抵制采用网络自动化，尤其是在运营团队中。

+   **组织结构**：当网络自动化被用于不同的网络层和网络域，如 IT 和网络域时，它确实可以带来真正的**投资回报率（ROI）**。在许多组织中存在的挑战是，这些域由不同的部门拥有，每个部门都有自己的自动化策略和关于自动化平台的偏好。

+   **选择自动化平台/工具**：从思科或诺基亚等网络设备供应商中选择自动化平台，或者与惠普或埃森哲等第三方自动化平台合作，并不是一个容易的决定。在许多情况下，电信服务提供商最终会拥有多个供应商来构建他们的网络自动化，这给让这些供应商协同工作带来了一系列新的挑战。

+   **维护**：维护自动化工具和脚本与构建它们一样重要。这需要要么从自动化供应商那里购买必要的维护合同，要么设立一个内部团队来为这些自动化平台提供维护。

接下来，我们来看一下用例。

## 用例

可以使用 Python 或其他工具自动执行与网络管理相关的许多单调任务。但真正的益处是自动化那些如果手动执行会重复、易出错或令人厌烦的任务。从电信服务提供商的角度来看，以下是一些网络自动化应用的主要方面：

+   我们可以自动化网络设备的日常配置，例如创建新的 IP 接口和网络连接服务。手动执行这些任务非常耗时。

+   我们可以配置防火墙规则和政策来节省时间。创建防火墙规则配置是一项繁琐的活动，任何错误都可能导致在解决通信挑战时浪费时间。

+   当我们在网络中有成千上万的设备时，升级它们的软件是一个巨大的挑战，有时，这需要 1 到 2 年的时间才能完成。网络自动化可以加速这一活动，并方便地进行升级前后的检查，以确保无缝升级。

+   我们可以使用网络自动化将新的网络设备加入网络。如果设备要安装在客户的场所，我们可以通过自动化设备的加入过程来节省一次现场服务。这个过程也被称为**零接触配置**（**ZTP**）。

现在我们已经介绍了网络自动化，让我们来探讨如何使用不同的协议与网络设备进行交互。

# 与网络设备交互

Python 是网络自动化中一个流行的选择，因为它易于学习，可以直接与网络设备集成，也可以通过 NMS 集成。实际上，许多厂商，如诺基亚和思科，在其网络设备上支持 Python 运行时。在单个设备上下文中自动化任务和活动的设备 Python 运行时选项非常有用。在本节中，我们将重点关注设备外 Python 运行时选项。这个选项将使我们能够同时处理多个设备。

重要提示

在本节提供的所有代码示例中，我们将使用来自思科的虚拟网络设备（IOS XR 版本 7.1.2）。为了与 NMS 集成，我们将使用诺基亚 NSP 系统。

在使用 Python 与网络设备交互之前，我们将讨论可用于与网络设备通信的协议。

## 与网络设备交互的协议

当涉及到直接与网络设备通信时，我们可以使用几种协议，例如 **安全外壳协议**（**SSH**）、**简单网络管理协议**（**SNMP**）和 **网络配置**（**NETCONF**）。其中一些协议是建立在其他协议之上的。接下来将描述最常用的协议。

### SSH

SSH 是一种网络协议，用于在任意两个设备或计算机之间安全地通信。在信息发送到传输通道之前，两个实体之间的所有信息都将被加密。我们通常使用 SSH 客户端通过 `ssh` 命令连接到网络设备。SSH 客户端使用 `ssh` 命令的已登录操作系统用户的 **用户名**：

```py
ssh <server ip or hostname>
```

要使用除已登录用户以外的其他用户，我们可以指定**用户名**，如下所示：

```py
ssh username@<server IP or hostname>
```

一旦建立了 SSH 连接，我们可以发送 CLI 命令，要么从设备检索配置或操作信息，要么配置设备。**SSH 版本 2**（**SSHv2**）是用于与设备进行网络管理和甚至自动化目的的流行选择。

在*使用基于 SSH 的协议与网络设备交互*这一部分，我们将讨论如何使用 Python 库如 Paramiko、Netmiko 和 NAPALM 来使用 SSH 协议。SSH 也是许多高级网络管理协议的基础传输协议，如 NETCONF。

### SNMP

该协议已经成为了 30 多年来的网络管理事实上的标准，并且仍然被大量用于网络管理。然而，它正在被更先进和可扩展的协议如 NETCONF 和 gNMI 所取代。SNMP 可用于网络配置和网络监控，但它更常用于网络监控。在当今世界，它被认为是一种在 20 世纪 80 年代末引入的遗留协议，纯粹用于网络管理。

SNMP 协议依赖于**管理信息库**（**MIB**），这是一个设备模型。该模型是使用一种称为**管理信息结构**（**SMI**）的数据建模语言构建的。

### NETCONF

由**互联网工程任务组**（**IETF**）引入的 NETCONF 协议被认为是 SNMP 的继任者。NETCONF 主要用于配置网络设备，并预期所有新的网络设备都将支持它。NETCONF 基于以下四层：

+   **内容**：这是一个依赖于 YANG 建模的数据层。每个设备都为其提供的各种模块提供几个 YANG 模型。这些模型可以在[`github.com/YangModels/yang`](https://github.com/YangModels/yang)上探索。

+   `get`、`get-config`、`edit-config`和`delete-config`。

+   **消息**：这些是在 NETCONF 客户端和 NETCONF 代理之间交换的**远程过程调用**（**RPC**）消息。编码为 XML 的 NETCONF 操作和数据被封装在 RPC 消息中。

+   **传输**：这一层在客户端和服务器之间提供通信路径。NETCONF 消息可以使用 NETCONF over SSH 或使用 SSL 证书选项的 NETCONF over TLS。

NETCONF 协议基于通过 SSH 协议交换的 XML 消息，默认端口为`830`。网络设备通常管理两种类型的配置数据库。第一种类型称为**运行**数据库，它表示设备上的活动配置，包括操作数据。这是每个设备的强制数据库。第二种类型称为**候选**数据库，它表示在推送到运行数据库之前可以使用的候选配置。当存在候选数据库时，不允许直接对运行数据库进行配置更改。

我们将在*使用 NETCONF 与网络设备交互*部分讨论如何使用 Python 与 NETCONF 一起工作。

### RESTCONF

RESTCONF 是另一个*IETF*标准，它通过 RESTful 接口提供 NETCONF 功能的一个子集。与使用 XML 编码的 NETCONF RPC 调用不同，RESTCONF 提供基于 HTTP/HTTPS 的 REST 调用，可以选择使用 XML 或 JSON 消息。如果网络设备提供 RESTCONF 接口，我们可以使用 HTTP 方法（`GET`、`PATCH`、`PUT`、`POST`和`DELETE`）进行网络管理。当使用 RESTCONF 进行网络自动化时，我们必须理解它通过 HTTP/HTTPS 提供有限的 NETCONF 功能。NETCONF 操作，如提交、回滚和配置锁定，不支持通过 RESTCONF 进行。

### gRPC/gNMI

gNMI 是一个用于网络管理和遥测应用的 gRPC**网络管理接口**（**NMI**）。gRPC 是由 Google 开发的一种远程过程调用，用于低延迟和高性能的数据检索。gRPC 协议最初是为希望与具有严格延迟要求的云服务器通信的移动客户端开发的。gRPC 协议在通过**协议缓冲区**（**Protobufs**）传输结构化数据方面非常高效，这是该协议的关键组件。通过使用 Protobufs，数据以二进制格式打包，而不是 JSON 或 XML 等文本格式。这种格式不仅减少了数据的大小，而且与 JSON 或 XML 相比，在序列化和反序列化数据方面非常高效。此外，数据使用 HTTP/2 而不是 HTTP 1.1 进行传输。HTTP/2 提供了请求-响应模型和双向通信模型。这种双向通信模型使得客户端能够打开长连接，从而显著加快数据传输过程。这两种技术使得 gRPC 协议比 REST API 快 7 到 10 倍。

gNMI 是 gRPC 协议在网络管理和遥测应用中的特定实现。它也是一个 YANG 模型驱动的协议，与 NETCONF 类似，但与 NETCONF 相比，提供的操作非常少。这些操作包括`Get`、`Set`和`Subscribe`。gNMI 在遥测数据收集方面比在网络管理方面更受欢迎。主要原因在于它不像 NETCONG 那样为网络配置提供足够的灵活性，但在从远程系统收集数据，尤其是在实时或近实时时，它是一个优化的协议。

接下来，我们将讨论用于与网络设备交互的 Python 库。

## 使用基于 SSH 的 Python 库与网络设备交互

有几个 Python 库可用于使用 SSH 与网络设备交互。Paramiko、Netmiko 和 NAPALM 是三个可用的流行库，我们将在下一节中探讨它们。我们将从 Paramiko 开始。

### Paramiko

Paramiko 库是 Python 中 SSH v2 协议的抽象，包括服务器端和客户端功能。在这里，我们将只关注 Paramiko 库的客户端功能。

当我们与网络设备交互时，我们要么尝试获取配置数据，要么为某些对象推送新的配置。前者通过设备操作系统的*show*类型 CLI 命令实现，而后者可能需要执行配置 CLI 命令的特殊模式。这两种类型的命令在通过 Python 库工作时处理方式不同。

#### 获取设备配置

要使用 Paramiko 库连接到网络设备（作为 SSH 服务器），我们必须使用`paramiko.SSHClient`类的实例或直接使用低级的`paramiko.Transport`类。`Transport`类提供了低级方法，可以提供基于套接字的通信的直接控制。`SSHClient`类是一个包装类，在底层使用`Transport`类来管理会话，并在网络设备上实现 SSH 服务器。

我们可以使用 Paramiko 库与网络设备（在我们的例子中是 Cisco IOS XR）建立连接，并运行 show 命令（在我们的例子中是`show ip int brief`）如下：

```py
#show_cisco_int_pmk.py
import paramiko
host='HOST_ID'
port=22
username='xxx'
password='xxxxxx'
#cisco ios command to get a list of IP interfaces
cmd= 'show ip int brief \n'
def main():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.          AutoAddPolicy())
        ssh.connect(host, port, username, password)
        stdin, stdout, stderr = ssh.exec_command(cmd)
        output_lines = stdout.readlines()
        response = ''.join(output_lines)
        print(response)
    finally:
        ssh.close()
if __name__ == '__main__':
    main()
```

本代码示例的关键点如下：

+   我们创建了一个`SSHClient`实例，并与 SSH 服务器建立了连接。

+   由于我们不是使用主机密钥进行我们的 SSH 连接，所以我们应用了`set_missing_host_key_policy`方法以避免任何警告或错误。

+   一旦建立了 SSH 连接，我们就使用 SSH 传输向主机机器发送了我们的 show 命令`show ip int brief`，并接收了命令的输出作为 SSH 回复。

+   该程序的输出是一个包含`stdin`、`stdout`和`stderr`对象的元组。如果我们的命令执行成功，我们将从`stdout`对象中检索输出。

当在 Cisco IOS XR 设备上执行此程序时，输出如下：

```py
Mon Jul 19 12:03:41.631 UTC
Interface                   IP-Address      Status    Protocol 
Loopback0                   10.180.180.10   Up        Up
GigabitEthernet0/0/0/0      10.1.10.2       Up        Up
GigabitEthernet0/0/0/0.100  unassigned      Up        Down
GigabitEthernet0/0/0/1      unassigned      Up        Up
GigabitEthernet0/0/0/1.100  150.150.150.1   Up        Up
GigabitEthernet0/0/0/2      unassigned      Shutdown  Down 
```

如果你在这个程序上运行其他设备类型的程序，你必须根据你的设备类型更改已设置为`cmd`变量的命令。

Paramiko 库提供了对网络通信的低级控制，但由于许多网络设备对 SSH 协议的非标准或不完整实现，它有时可能会出现一些奇怪的问题。如果你在使用 Paramiko 与某些网络设备时遇到挑战，问题不是你或 Paramiko，而是设备期望你与之通信的方式。低级传输通道可以解决这些问题，但这需要一些复杂的编程。Netmiko 在这里提供了帮助。

### Netmiko

Netmiko 是一个基于 Paramiko 库构建的网络管理抽象库。它通过将每个网络设备视为不同类型来消除 Paramiko 的挑战。Netmiko 在底层使用 Paramiko，并隐藏了许多设备级通信细节。Netmiko 支持来自不同厂商的多种设备，例如 Cisco、Arista、Juniper 和 Nokia。

#### 获取设备配置

要使用`show`类型的 CLI 命令连接到网络设备，我们必须设置一个`device_type`定义，该定义用于连接到目标网络设备。这个`device_type`定义是一个字典，必须包括设备的类型、主机 IP 或设备的`22`端口。以下代码可以用来执行我们使用 Paramiko 库执行的相同`show`命令：

```py
#show_cisco_int_nmk.py
from netmiko import ConnectHandler
cisco_rtr = {
    "device_type": "cisco_ios",
    "host": "HOST_ID",
    "username": "xxx",
    "password": "xxxxxxx",
    #"global_delay_factor": 2,
}
def main():
    command = "show ip int brief"
    with ConnectHandler(**cisco_rtr) as net_connect:
        print(net_connect.find_prompt())
        print(net_connect.enable())
        output = net_connect.send_command(command)
    print(output)
```

本示例代码的关键点如下：

+   我们使用`ConnectHandler`类和上下文管理器创建了一个网络连接。上下文管理器将管理连接的生命周期。

+   Netmiko 提供了一个名为`find_prompt`的简单方法，用于获取目标设备的提示符，这对于解析许多网络设备的输出非常有用。对于*Cisco IOS XR*网络设备，这并不是必需的，但我们将其作为最佳实践。

+   Netmiko 还允许我们通过使用`enable`方法进入*启用*模式（这是一个命令行提示符，`#`），用于 Cisco IOS 设备。再次强调，对于本示例来说，这不是必需的，但将其作为最佳实践使用，尤其是在我们作为同一编程脚本的一部分推送 CLI 配置命令的情况下。

+   我们使用`send_command`方法执行了`show ip int brief`命令，并得到了与`show_cisco_int_pmk.py`程序相同的输出。

基于我们分享的相同`show`命令的代码示例，我们可以得出结论，与 Paramiko 相比，使用 Netmiko 要方便得多。

重要提示

设置正确的设备类型对于获得一致的结果非常重要，即使您使用的是同一厂商的设备也是如此。当使用配置设备的命令时，这一点尤为重要。错误的设备类型可能导致不一致的错误。

有时，我们执行的命令需要比正常的`show`命令更多的时间来完成。例如，我们可能想要将设备上的文件从一个位置复制到另一个位置，我们知道对于大文件来说，这可能需要几百秒。默认情况下，Netmiko 等待命令完成的几乎为*100*秒。我们可以通过添加如下类似的行作为设备定义的一部分来添加全局延迟因子：

```py
"global_delay_factor": 2
```

这将使该设备的所有命令的等待时间增加 2 倍。或者，我们可以通过`send_command`方法传递以下参数来为单个命令设置延迟因子：

```py
delay_factor=2 
```

当我们预期有显著的执行时间时，我们应该添加一个延迟因子。当我们需要添加延迟因子时，我们还应该在`send_command`方法中添加另一个作为参数的属性，这样如果我们看到命令提示符（例如，Cisco IOS 设备的`#`），就可以提前中断等待周期。这可以通过以下属性设置：

```py
expect_string=r'#'
```

#### 配置网络设备

在以下代码示例中，我们将提供一些用于配置目的的示例代码。使用 Netmiko 配置设备类似于执行`show`命令，因为 Netmiko 将负责启用配置终端（如果需要，根据设备类型）并优雅地退出配置终端。

对于我们的代码示例，我们将使用以下程序使用 Netmiko 设置接口的`description`：

```py
#config_cisco_int_nmk.py
from netmiko import ConnectHandler
cisco_rtr = {
    "device_type": "cisco_ios",
    "host": "HOST_ID",
    "username": "xxx",
    "password": "xxxxxx",
}
def main():
    commands = ["int Lo0 "description custom_description",       "commit"]
    with ConnectHandler(**cisco_rtr) as net_connect:
        output = net_connect.send_config_set(commands)
    print(output)
    print()
```

这个代码示例的关键点如下：

+   对于这个程序，我们创建了一个包含三个命令的列表（`int <interface id>`、`description <new description>`和`commit`）。前两个命令也可以作为一个单独的命令发送，但我们为了说明目的而将它们分开。`commit`命令用于保存更改。

+   当我们向设备发送配置命令时，我们使用 Netmiko 库中的`send_config_set`方法来设置配置目的的连接。成功执行此步骤取决于设备类型的正确设置。这是因为配置命令的行为因设备而异。

+   这组三个命令将为指定的接口添加或更新`description`属性。

除了设备配置提示符显示我们的命令外，这个程序不会期望有特殊的输出。控制台输出将如下所示：

```py
Mon Jul 19 13:21:16.904 UTC
RP/0/RP0/CPU0:cisco(config)#int Lo0
RP/0/RP0/CPU0:cisco(config-if)#description custom_description
RP/0/RP0/CPU0:cisco(config-if)#commit
Mon Jul 19 13:21:17.332 UTC
RP/0/RP0/CPU0:cisco(config-if)#
```

Netmiko 提供了更多功能，但我们将其留给你通过阅读其官方文档([`pypi.org/project/netmiko/`](https://pypi.org/project/netmiko/))来探索。本节中讨论的代码示例已在 Cisco 网络设备上测试过，但如果你使用的设备由 Netmiko 支持，可以通过更改设备类型和命令来使用相同的程序。

Netmiko 简化了网络设备交互的代码，但我们仍然在运行 CLI 命令以获取设备配置或将配置推送到设备。使用 Netmiko 进行编程并不容易，但另一个名为 NAPALM 的库可以帮助我们。

### NAPALM

**NAPALM** 是 **Network Automation and Programmability Abstraction Layer with Multivendor** 的缩写。这个库在 Netmiko 之上提供了更高层次的抽象，通过提供一组函数作为统一的 API 来与多个网络设备交互。它支持的设备数量不如 Netmiko 多。对于 NAPALM 的第三个版本，核心驱动程序适用于 **Arista EOS**、**Cisco IOS**、**Cisco IOS-XR**、**Cisco NX-OS** 和 **Juniper JunOS** 网络设备。然而，还有几个社区构建的驱动程序可用于与许多其他设备通信，例如 **Nokia SROS**、**Aruba AOS-CX** 和 **Ciena SAOS**。

与 Netmiko 一样，我们将为与网络设备交互构建 NAPALM 示例。在第一个示例中，我们将获取 IP 接口列表，而在第二个示例中，我们将为 IP 接口添加或更新 `description` 属性。这两个代码示例将执行我们使用 Paramiko 和 Netmiko 库执行的操作。

#### 获取设备配置

要获取设备配置，我们必须设置与我们的网络设备的连接。我们将在两个代码示例中都这样做。设置连接是一个三步过程，如下所述：

1.  要设置连接，我们必须根据支持的设备类型获取设备驱动程序类。这可以通过使用 NAPALM 库的 `get_network_driver` 函数来实现。

1.  一旦我们有了设备驱动程序类，我们可以通过向驱动程序类的构造函数提供例如 `host id`、`username` 和 `password` 等参数来创建设备对象。

1.  下一步是使用设备对象的 `open` 方法连接到设备。所有这些步骤都可以像下面这样实现为 Python 代码：

    ```py
    from napalm import get_network_driver 
    driver = get_network_driver('iosxr')
    device = driver('HOST_ID', 'xxxx', 'xxxx')
    device.open()
    ```

一旦设备的连接可用，我们可以调用 `get_interfaces_ip`（相当于 `show interfaces` CLI 命令）或 `get_facts`（相当于 `show version` CLI 命令）等方法。使用这两个方法的完整代码如下：

```py
#show_cisco_int_npm.py
from napalm import get_network_driver
import json
def main():
    driver = get_network_driver('iosxr')
    device = driver('HOST_ID', 'root', 'rootroot')
    try:
        device.open()
        print(json.dumps(device.get_interfaces_ip(), indent=2))
        #print(json.dumps(device.get_facts(), indent=2))
    finally:
        device.close()
```

最有趣的事实是，这个程序的输出默认是 JSON 格式。NAPALM 默认将 CLI 命令的输出转换为 Python 中易于消费的字典。以下是之前代码示例输出的一部分：

```py
{
  "Loopback0": {
    "ipv4": {
      "10.180.180.180": {
        "prefix_length": 32
      }
    }
  },
  "MgmtEth0/RP0/CPU0/0": {
    "ipv4": {
      "172.16.2.12": {
        "prefix_length": 24
      }
    }
  }
}
```

#### 配置网络设备

在以下代码示例中，我们使用 NAPALM 库为现有的 IP 接口添加或更新 `description` 属性：

```py
#config_cisco_int_npm.py
from napalm import get_network_driver
import json
def main():
    driver = get_network_driver('iosxr')
    device = driver('HOST_ID', 'xxx', 'xxxx')
    try:
        device.open()
        device.load_merge_candidate(config='interface Lo0 \n            description napalm_desc \n end\n')
        print(device.compare_config())
        device.commit_config()
    finally:
        device.close()
```

此代码示例的关键点如下：

+   要配置 IP 接口，我们必须使用 `load_merge_candidate` 方法，并将与 Netmiko 接口配置相同的 CLI 命令集传递给此方法。

+   接下来，我们使用 `compare_config` 方法比较了命令前后配置的差异。这表明了添加了哪些新配置以及删除了哪些配置。

+   我们使用 `commit_config` 方法提交了所有更改。

对于这个示例代码，输出将显示变化的差异，如下所示：

```py
--- 
+++ 
@@ -47,7 +47,7 @@
  !
 !
 interface Loopback0
- description my custom description
+ description napalm added new desc 
  ipv4 address 10.180.180.180 255.255.255.255
 !
 interface MgmtEth0/RP0/CPU0/0
```

在这里，以`-`开头的行是要删除的配置；任何以`+`开头的行是要添加的新配置。

通过这两个代码示例，我们已经向你展示了一个设备类型的基本 NAPALM 功能集。这个库可以同时配置多个设备，并且可以与不同的配置集一起工作。

在下一节中，我们将讨论使用 NETCONF 协议与网络设备交互。

## 使用 NETCONF 与网络设备交互

NETCONF 是为了模型（对象）驱动的网络管理而创建的，特别是为了网络配置。在使用 NETCONF 与网络设备一起工作时，了解设备以下两个功能是很重要的：

+   你可以理解你所拥有的设备的 YANG 模型。如果你希望以正确的格式发送消息，拥有这些知识是很重要的。以下是从各种供应商那里获取 YANG 模型的优秀来源：[`github.com/YangModels/yang`](https://github.com/YangModels/yang)。

+   你可以为你的网络设备上的 NETCONF 和 SSH 端口启用 NETCONF 协议。在我们的案例中，我们将使用 Cisco IOS XR 的虚拟设备，正如我们在之前的代码示例中所做的那样。

在开始任何网络管理相关活动之前，我们必须检查设备的 NETCONF 功能以及 NETCONF 数据源配置的详细信息。在本节的所有代码示例中，我们将使用一个名为`ncclient`的 Python NETCONF 客户端库。这个库提供了发送 NETCONF RPC 请求的便捷方法。我们可以使用`ncclient`库编写一个示例 Python 程序，以获取设备的功能和设备的完整配置，如下所示：

```py
#check_cisco_device.py
from ncclient import manager
with manager.connect(host='device_ip, username=xxxx,   password=xxxxxx, hostkey_verify=False) as conn:
   capabilities = []
   for capability in conn.server_capabilities:
      capabilities.append(capability)
   capabilities = sorted(capabilities)
   for cap in capabilities:
     print(cap)
   result = conn.get_config(source="running")
   print (result)
```

`ncclient`库中的`manager`对象用于通过 SSH 连接到设备，但使用 NETCONF 端口`830`（默认）。首先，我们通过连接实例获取服务器功能列表，然后以排序格式打印它们，以便于阅读。在代码示例的下一部分，我们通过`manager`类库的`get_config`方法启动了一个`get-config` NETCONF 操作。这个程序的输出非常长，显示了所有功能和设备配置。我们将其留给你去探索，并熟悉你设备的特性。

重要的是要理解本节的范围不是解释 NETCONF，而是学习如何使用 Python 和`ncclient`与 NETCONF 一起工作。为了实现这一目标，我们将编写两个代码示例：一个用于获取设备接口的配置，另一个是如何更新接口的描述，这与我们之前为 Python 库所做的是相同的。

### 通过 NETCONF 获取接口

在前面的章节中，我们了解到我们的设备（Cisco IOS XR）通过**OpenConfig**实现支持接口，该实现可在[`openconfig.net/yang/interfaces?module=openconfig-interfaces`](http://openconfig.net/yang/interfaces?module=openconfig-interfaces)找到。

我们还可以检查我们接口配置的 XML 格式，这是我们通过`get_config`方法获得的输出。在这个代码示例中，我们将简单地将一个带有接口配置的 XML 过滤器作为参数传递给`get_config`方法，如下所示：

```py
#show_all_interfaces.py
from ncclient import manager
with manager.connect(host='device_ip', username=xxx,                password='xxxx', hostkey_verify=False) as conn:
    result = conn.get_config("running", filter=('subtree', 
    '<interfaces xmlns= "http://openconfig.net/yang/      interfaces"/>'))
    print (result)
```

这个程序的输出是一个接口列表。为了说明目的，我们在这里只展示输出的一部分：

```py
<rpc-reply message-id="urn:uuid:f4553429-ede6-4c79-aeea-5739993cacf4" xmlns:nc="urn:ietf:params:xml:ns:netconf:base:1.0" xmlns="urn:ietf:params:xml:ns:netconf:base:1.0">
 <data>
  <interfaces xmlns="http://openconfig.net/yang/interfaces">
   <interface>
    <name>Loopback0</name>
    <config>
     <name>Loopback0</name>
     <description>Configured by NETCONF</description>
    </config>
<!—rest of the output is skipped -->
```

为了获取一组选择性的接口，我们将使用基于接口 YANG 模型的扩展版 XML 过滤器。对于下面的代码示例，我们将定义一个带有接口`name`属性的 XML 过滤器作为我们的过滤标准。由于这个 XML 过滤器是多行的，我们将单独将其定义为字符串对象。以下是带有 XML 过滤器的示例代码：

```py
#show_int_config.py
from ncclient import manager
# Create filter template for an interface
filter_temp = """
<filter>
    <interfaces xmlns="http://openconfig.net/yang/interfaces">
        <interface>
            <name>{int_name}</name>
        </interface>
    </interfaces>
</filter>"""
with manager.connect(host='device_ip', username=xxx,                password='xxxx', hostkey_verify=False) as conn:
    filter = filter_temp.format(int_name = "MgmtEth0/RP0/      CPU0/0")
    result = m.get_config("running", filter)
    print (result)
```

这个程序的输出将是一个单独的接口（根据我们设备的配置），如下所示：

```py
<?xml version="1.0"?>
<rpc-reply message-id="urn:uuid:c61588b3-1bfb-4aa4-a9de-2a98727e1e15" xmlns:nc="urn:ietf:params:xml:ns:netconf:base:1.0" xmlns="urn:ietf:params:xml:ns:netconf:base:1.0">
 <data>
  <interfaces xmlns="http://openconfig.net/yang/interfaces">
   <interface>
    <name>MgmtEth0/RP0/CPU0/0</name>
    <config>
     <name>MgmtEth0/RP0/CPU0/0</name>
    </config>
    <ethernet xmlns="http://openconfig.net/yang/interfaces/      ethernet">
     <config>
      <auto-negotiate>false</auto-negotiate>
     </config>
    </ethernet>
    <subinterfaces>
     <@!— ommitted sub interfaces details to save space -->
    </subinterfaces>
   </interface>
  </interfaces>
 </data>
</rpc-reply>
```

我们也可以在 XML 文件中定义 XML 过滤器，然后在 Python 程序中将文件内容读入字符串对象。如果我们计划广泛使用过滤器，另一个选项是使用*Jinja*模板。

接下来，我们将讨论如何更新接口的描述。

### 更新接口的描述

要配置接口属性，如`description`，我们必须使用在[`cisco.com/ns/yang/Cisco-IOS-XR-ifmgr-cfg`](http://cisco.com/ns/yang/Cisco-IOS-XR-ifmgr-cfg)可用的 YANG 模型。

此外，配置接口的 XML 块与我们用于获取接口配置的 XML 块不同。为了更新接口，我们必须使用以下模板，我们已在单独的文件中定义：

```py
<!--config-template.xml-->
<config xmlns:xc="urn:ietf:params:xml:ns:netconf:base:1.0">
 <interface-configurations xmlns="http://cisco.com/ns/yang/  Cisco-IOS-XR-ifmgr-cfg">
   <interface-configuration>
    <active>act</active>
    <interface-name>{int_name}</interface-name>
    <description>{int_desc}</description>
   </interface-configuration>
 </interface-configurations>
</config>
```

在这个模板中，我们设置了接口的`name`和`description`属性的占位符。接下来，我们将编写一个 Python 程序，该程序将读取这个模板并通过使用`ncclient`库的`edit_config`方法调用 NETCONF 的`edit-config`操作，将模板推送到设备的候选数据库：

```py
#config_cisco_int_desc.py
from ncclient import manager
nc_template = open("config-template.xml").read()
nc_payload = nc_template.format(int_name='Loopback0',                          int_desc="Configured by NETCONF")
with manager.connect(host='device_ip, username=xxxx,                     password=xxx, hostkey_verify=False) as nc:
    netconf_reply = nc.edit_config(nc_payload,       target="candidate")
    print(netconf_reply)
    reply = nc.commit()
    print(reply)
```

在这里，有两点很重要。首先，Cisco IOS XR 设备已被配置为仅通过候选数据库接受新的配置。如果我们尝试将`target`属性设置为`running`，它将失败。其次，我们必须在相同会话中在`edit-config`操作之后调用`commit`方法，以使新的配置生效。这个程序的输出将是 NETCONF 服务器两个 OK 回复，如下所示：

```py
<?xml version="1.0"?>
<rpc-reply message-id="urn:uuid:6d70d758-6a8e-407d-8cb8-10f500e9f297" xmlns:nc="urn:ietf:params:xml:ns:netconf:base:1.0" xmlns="urn:ietf:params:xml:ns:netconf:base:1.0">
 <ok/>
</rpc-reply>
<?xml version="1.0"?>
<rpc-reply message-id="urn:uuid:2a97916b-db5f-427d-9553-de1b56417d89" xmlns:nc="urn:ietf:params:xml:ns:netconf:base:1.0" xmlns="urn:ietf:params:xml:ns:netconf:base:1.0">
 <ok/>
</rpc-reply>
```

这就结束了我们使用 Python 进行 NETCONF 操作的讨论。我们使用`ncclient`库介绍了 NETCONF 的两个主要操作（`get-config`和`edit-config`）。

在下一节中，我们将探讨如何使用 Python 与网络管理系统集成。

# 与网络管理系统集成

网络管理系统或网络控制器是提供具有**图形用户界面**（**GUIs**）的网络管理应用程序的系统。这些系统包括网络库存、网络配置、故障管理和与网络设备的调解等应用程序。这些系统使用 SSH/NETCONF（用于网络配置）、SNMP（用于警报和设备监控）和 gRPC（用于遥测数据收集）等通信协议的组合与网络设备通信。这些系统还通过其脚本和工作流引擎提供自动化功能。

这些系统的最有价值之处在于，它们将网络设备的各项功能聚合到一个单一系统中（即自身），然后通过其**北向接口**（**NBIs**），通常是 REST 或 RESTCONF 接口提供。这些系统还通过基于事件系统的通知，如 Apache Kafka，提供实时事件（如警报）的通知。在本节中，我们将讨论使用 NMS 的 REST API 的几个示例。在*与事件驱动系统集成*部分，我们将探讨如何使用 Python 与 Apache Kafka 集成。

要与 NMS 一起工作，我们将使用诺基亚在线开发者门户提供的共享实验室([`network.developer.nokia.com/`](https://network.developer.nokia.com/))。这个实验室有几台诺基亚 IP 路由器和一台 NSP。这个共享实验室在撰写本书时免费提供（每天 3 小时）。您需要免费在开发者门户中创建一个账户。当您预订实验室使用时，您将收到一封电子邮件，其中包含如何连接到实验室的说明，以及必要的 VPN 详细信息。如果您是网络工程师并且可以访问任何其他 NMS 或控制器，您可以通过进行适当的调整使用该系统来完成本节中的练习。

要从诺基亚 NSP 消费 REST API，我们需要与 REST API 网关交互，该网关管理诺基亚 NSP 的多个 API 端点。我们可以通过使用位置服务开始与 REST API 网关一起工作，如以下所述。

## 使用位置服务端点

要了解可用的 API 端点，诺基亚 NSP 提供了一个位置服务端点，提供所有 API 端点的列表。在本节中，我们将使用 Python 的`requests`库来消费任何 REST API。`requests`库因其使用 HTTP 协议向服务器发送 HTML 请求而闻名，我们已在之前的章节中使用过它。要从诺基亚 NSP 系统中获取 API 端点列表，我们将使用以下 Python 代码调用位置服务 API：

```py
#location_services1.py
import requests
payload = {}
headers = {}
url = "https://<NSP URL>/rest-gateway/rest/api/v1/location/  services"
resp = requests.request("GET", url, headers=headers,   data=payload)
print(resp.text)
```

此 API 响应将为您提供几十个 API 端点，以 JSON 格式。您可以在诺基亚 NSP 的在线文档[`network.developer.nokia.com/api-documentation/`](https://network.developer.nokia.com/api-documentation/)中查看，了解每个 API 是如何工作的。如果我们正在寻找特定的 API 端点，我们可以在上述代码示例中更改`url`变量的值，如下所示：

```py
url = "https://<NSP URL>/rest-gateway/rest/api/v1/ location/services/endpoints?endPoint=/v1/auth/token
```

通过使用这个新的 API URL，我们试图找到一个用于授权令牌的 API 端点（`/v1/auth/token`）。使用这个新 URL 的代码示例输出如下：

```py
{ 
 "response": { 
  "status": 0, 
  "startRow": 0, 
  "endRow": 0, 
  "totalRows": 1, 
  "data": { 
   "endpoints": [ 
    { 
    "docUrl":"https://<NSP_URL>/rest-gateway/api-docs#!/      authent..", 
    "effectiveUrl": "https://<NSP_URL>/rest-gateway/rest/api", 
    "operation": "[POST]" 
    } 
   ] 
  }, 
  "errors": null 
 } 
}
```

注意，使用位置服务 API 不需要身份验证。但是，我们需要一个身份验证令牌来调用任何其他 API。在下一节中，我们将学习如何获取身份验证令牌。

## 获取身份验证令牌

作为下一步，我们将使用前一个代码示例的输出中的`effectiveUrl`来获取身份验证令牌。此 API 要求我们将`username`和`password`的*base64*编码作为 HTTP 头中的`Authorization`属性传递。调用此身份验证 API 的 Python 代码如下：

```py
#get_token.py
import requests
from base64 import b64encode
import json
#getting base64 encoding 
message = 'username'+ ':' +'password'
message_bytes = message.encode('UTF-8')
basic_token = b64encode(message_bytes)
payload = json.dumps({
  "grant_type": "client_credentials"
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Basic {}'.format(str(basic_token,'UTF-8'))
}
url = "https://<NSP SERVER URL>/rest-gateway/rest/api/v1/auth/  token"
resp = requests.request("POST", url, headers=headers,   data=payload)
token = resp.json()["access_token"]
print(resp)
When executing this Python code, we will get a token for one   hour to be used for any NSP API. 
{
  "access_token": "VEtOLVNBTXFhZDQ3MzE5ZjQtNWUxZjQ0YjNl",
  "refresh_token": "UkVUS04tU0FNcWF5ZlMTmQ0ZTA5MDNlOTY=",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

此外，还有一个刷新令牌可用，可以在令牌过期之前刷新令牌。一个最佳实践是每*30*分钟刷新一次令牌。我们可以使用相同的身份验证令牌 API 来刷新我们的令牌，但在 HTTP 请求体中发送以下属性：

```py
payload = json.dumps({
  "grant_type": "refresh_token",
  "refresh_token": "UkVUS04tU0FNcWF5ZlMTmQ0ZTA5MDNlOTY="
})
```

另一个好习惯是在不再需要令牌时撤销它。这可以通过使用以下 API 端点来实现：

```py
 url = "https://<NSP URL>rest-gateway/rest/api/v1/auth/  revocation"
```

## 获取网络设备和接口清单

一旦我们收到身份验证令牌，我们就可以使用 REST API 获取配置数据，以及添加新的配置。我们将从一个简单的代码示例开始，该示例将获取由 NSP 管理的网络中所有网络设备的列表。在这个代码示例中，我们将使用通过令牌 API 已经检索到的令牌：

```py
#get_network_devices.py
import requests
pload={}
headers = {
  'Authorization': 'Bearer {token}'.format(token)
}
url = "https://{{NSP_URL}}:8544/NetworkSupervision/rest/api/v1/  networkElements"
response = requests.request("GET", url, headers=headers,   data=pload)
print(response.text)
```

此程序的输出将是一个包含网络设备属性的网络设备列表。我们跳过了输出显示，因为这是一组大量数据。

在下面的代码示例中，我们将展示如何根据过滤器获取设备端口（接口）列表。请注意，我们也可以将过滤器应用于网络设备。对于本代码示例，我们将要求 NSP API 根据端口名称（在我们的例子中是`Port 1/1/1`）给我们提供一个端口列表：

```py
#get_ports_filter.py
import requests
payload={}
headers = {
  'Authorization': 'Bearer {token}'.format(token)
}
url = "https://{{server}}:8544/NetworkSupervision/rest/api/v1/  ports?filter=(name='Port 1/1/1')
response = requests.request("GET", url, headers=headers,   data=payload)
print(response.text)
```

此程序的输出将是从所有网络设备中调用名为`Port 1/1/1`的设备端口列表。使用单个 API 获取多个网络设备的端口是使用 NMS 的实际价值

接下来，我们将讨论如何使用 NMS API 更新网络资源。

## 更新网络设备端口

使用 NMS API 创建新对象或更新现有对象也很方便。我们将实现一个更新端口描述的案例，就像我们在之前的代码示例中所做的那样，使用 `Netmiko`、`NAPALM` 和 `ncclient`。要更新端口或接口，我们将使用一个不同的 API 端点，该端点来自 **网络功能管理器包**（**NFMP**）模块。NFMP 是诺基亚网络设备在诺基亚 NSP 平台下的 NMS 模块。让我们看看更新端口描述或对网络资源进行任何更改的步骤：

1.  要更新对象或在一个现有对象下创建新对象，我们需要使用具有以下筛选标准的 `v1/managedobjects/searchWithFilter` API：

    ```py
    #fullClassNames. The object's full class names are available in the Nokia NFMP object model documentation. We set filterExpression to search for a unique port based on the device site's ID and port name. The resultFilter attribute is used to limit the attributes that are returned by the API in the response. We are interested in the objectFullName attribute in the response of this API. 
    ```

1.  接下来，我们将使用一个名为 `v1/managedobjects/ofn` 的不同 API 端点来更新网络对象的属性。在我们的例子中，我们只更新描述属性。对于更新操作，我们必须在有效载荷中设置 `fullClassName` 属性以及描述属性的新值。对于 API 端点的 URL，我们将连接我们在上一步计算的 `port_ofn` 变量。该程序这部分内容的示例代码如下：

    ```py
    #update_port_desc.py (part 2)
    payload2 = json.dumps({
      "fullClassName": "equipment.PhysicalPort",
      "properties": {
        "description": "description added by a Python       program"
      }
    })
    url2 = "https:// NFMP_URL:8443/nfm-p/rest/api/v1/  managedobjects/"+port_ofn
    response = requests.request("PUT", url2, headers=headers,   data=payload2, verify=False)
    print(response.text)
    ```

网络自动化是指按照特定顺序创建和更新许多网络对象的过程。例如，我们可以在创建 IP 连接服务以连接两个或更多局域网之前更新一个端口。这类用例要求我们执行一系列任务来更新所有涉及的端口，以及许多其他对象。使用 NMS API，我们可以在程序中编排所有这些任务以实现自动化流程。

在下一节中，我们将探讨如何集成诺基亚 NSP 或类似系统以实现事件驱动通信。

# 集成事件驱动系统

在前面的章节中，我们讨论了如何使用请求-响应模型与网络设备和网络管理系统进行交互。在这个模型中，客户端向服务器发送请求，服务器作为对请求的回复发送响应。HTTP（REST API）和 SSH 协议基于请求-响应模型。这种模型在临时或定期配置系统或获取网络的操作状态时工作得很好。但是，如果网络中发生需要操作团队注意的事情怎么办？例如，假设设备上的硬件故障或线路电缆被切断。网络设备通常在这种情况下发出警报，并且这些警报必须立即通知操作员（通过电子邮件、短信或仪表板）。

我们可以使用请求-响应模型每秒（或每隔几秒）轮询网络设备，以检查网络设备的状态是否发生变化，或者是否有新的警报。然而，这种方式并不是网络设备资源的有效利用，并且会在网络中产生不必要的流量。那么，如果网络设备或 NMS 本身在关键资源状态发生变化或发出警报时主动联系感兴趣的客户端，会怎样呢？这种类型的模型被称为*事件驱动*模型，它是发送实时事件的一种流行通信方式。

事件驱动系统可以通过**webhooks**/**WebSockets**或使用**流式**方法来实现。WebSockets 通过 TCP/IP 套接字在 HTTP 1.1 上提供了一个双向传输通道。由于这种双向连接不使用传统的请求-响应模型，因此当我们需要在两个系统之间建立一对一连接时，WebSockets 是一种高效的方法。当我们需要两个程序之间进行实时通信时，这是最佳选择之一。所有标准浏览器都支持 WebSockets，包括 iPhone 和 Android 设备所提供的浏览器。它也是许多社交媒体平台、流媒体应用和在线游戏的流行选择。

WebSockets 是获取实时事件的一个轻量级解决方案。但是，当许多客户端希望从一个系统中接收事件时，使用流式方法可扩展且高效。基于流的基于事件模型通常遵循发布-订阅设计模式，并具有三个主要组件，如下所述：

+   **主题**：所有流式消息或事件通知都存储在主题下。我们可以将主题视为一个目录。这个主题帮助我们订阅感兴趣的主题，以避免接收所有事件。

+   **生产者**：这是一个将事件或消息推送到主题的程序或软件。这也被称为**发布者**。在我们的案例中，它将是一个 NSP 应用。

+   **消费者**：这是一个从主题中获取事件或消息的程序。这也被称为**订阅者**。在我们的案例中，这将是我们将要编写的 Python 程序。

事件驱动系统适用于网络设备以及网络管理系统。NMS 平台使用 gRPC 或 SNMP 等事件系统从网络设备接收实时事件，并为编排层或操作或监控应用程序提供聚合接口。在我们的示例中，我们将与诺基亚 NSP 平台的事件系统交互。诺基亚 NSP 系统提供了一个基于 Apache Kafka 的事件系统。Apache Kafka 是一个开源软件，用 Scala 和 Java 开发，它提供了一个基于**发布-订阅**设计模式的软件消息总线实现。在与 Apache Kafka 交互之前，我们将列举通过诺基亚 NSP 提供的以下关键**类别**（在 Apache Kafka 中用于主题的术语）列表：

+   `NSP-FAULT`：这个类别涵盖了与故障或警报相关的事件。

+   `NSP-PACKET-ALL`：这个类别用于所有网络管理事件，包括心跳事件。

+   `NSP-REAL-TIME-KPI`：这个类别代表实时流通知的事件。

+   `NSP-PACKET-STATS`：这个类别用于统计事件。

在诺基亚 NSP 文档中可以找到完整的类别列表。所有这些类别都提供了订阅特定类型事件的附加过滤器。在诺基亚 NSP 的上下文中，我们将与 Apache Kafka 交互以创建新的订阅，然后处理来自 Apache Kafka 系统的事件。我们将从订阅管理开始。

## 为 Apache Kafka 创建订阅

在从 Apache Kafka 接收任何事件或消息之前，我们必须订阅一个主题或类别。请注意，一个订阅仅对一类有效。订阅通常在 1 小时后过期，因此建议在过期前 30 分钟更新订阅。

要创建新的订阅，我们将使用`v1/notifications/subscriptions` API 和以下示例代码来获取新的订阅：

```py
#subscribe.py
import requests
token = <token obtain earlier>
url = "https://NSP_URL:8544/nbi-notification/api/v1/  notifications/subscriptions"
def create_subscription(category):
  headers = {'Authorization': 'Bearer {}'.format(token) }
  payload = {
      "categories": [
        {
          "name": "{}".format(category)
        }
      ]
  }
  response = requests.request("POST", url, json=payload,                               headers=headers, verify=False)
  print(response.text)
if __name__ == '__main__':
      create_subscription("NSP-PACKET-ALL")
```

该程序的输出将包括重要的属性，如`subscriptionId`、`topicId`和`expiresAt`等，如下所示：

```py
{
   "response":{
      "status":0,
      "startRow":0,
      "endRow":0,
      "totalRows":1,
      "data": {
         "subscriptionId":"440e4924-d236-4fba-b590-           a491661aae14",
         "clientId": null,
         "topicId":"ns-eg-440e4924-d236-4fba-b590-           a491661aae14",
         "timeOfSubscription":1627023845731,
         "expiresAt":1627027445731,
         "stage":"ACTIVE",
         "persisted":true
      },
      "errors":null
   }
}
```

`subscriptionId`属性用于稍后更新或删除订阅。Apache Kafka 将为该订阅创建一个特定的主题。它作为`topicId`属性提供给我们。我们将使用这个`topicId`属性来连接到 Apache Kafka 以接收事件。这就是为什么我们称 Apache Kafka 中的通用主题为类别。`expiresAt`属性表示此订阅将过期的时间。

一旦订阅准备就绪，我们就可以连接到 Apache Kafka 以接收事件，如下一小节所述。

## 处理来自 Apache Kafka 的事件

使用`kafka-python`库，编写一个基本的 Kafka 消费者只需要几行 Python 代码。要创建一个 Kafka 客户端，我们将使用`kafka-python`库中的`KafkaConsumer`类。我们可以使用以下示例代码来消费订阅主题的事件：

```py
#basic_consumer.py
topicid = 'ns-eg-ff15a252-f927-48c7-a98f-2965ab6c187d'
consumer = KafkaConsumer(topic_id,
                         group_id='120',
                         bootstrap_servers=[host_id], value_                          deserializer=lambda m: json.loads                          (m.decode('ascii')),
                         api_version=(0, 10, 1))
try:
    for message in consumer:
        if message is None:
            continue
        else:
            print(json.dumps(message.value, indent=4, sort_              keys=True))
except KeyboardInterrupt:
    sys.stderr.write('++++++ Aborted by user ++++++++\n')
finally:
    consumer.close()
```

重要提示：如果您使用的是 Python 3.7 或更高版本，则必须使用`kafka-python`库。如果您使用的是低于 3.7 版本的 Python，则可以使用`kafka`库。如果我们在 Python 3.7 或更高版本中使用`kafka`库，已知存在一些问题。例如，已知`async`在 Python 3.7 或更高版本中已成为关键字，但在`kafka`库中已被用作变量。当使用`kafka-python`库与 Python 3.7 或更高版本一起使用时，也存在 API 版本问题。这些问题可以通过设置正确的 API 版本作为参数（在这种情况下为`0.10.0`版本）来避免。

在本节中，我们向您展示了一个基本的 Kafka 消费者，但您可以通过访问本书提供的源代码中的更复杂示例来探索：[`github.com/nokia/NSP-Integration-Bootstrap/tree/master/kafka/kafka_cmd_consumer`](https://github.com/nokia/NSP-Integration-Bootstrap/tree/master/kafka/kafka_cmd_consumer)。

## 续订和删除订阅

我们可以使用与创建订阅相同的 API 端点来使用 Nokia NSP Kafka 系统续订订阅。我们将在 URL 末尾添加`subscriptionId`属性，以及`renewals`资源，如下所示：

```py
https://{{server}}:8544/nbi-notification/api/v1/notifications/subscriptions/<subscriptionId>/renewals
```

我们可以使用相同的 API 端点，通过在 URL 末尾添加`subscriptionId`属性，并使用 HTTP 的`Delete`方法来删除订阅。以下是一个删除请求的 API 端点示例：

```py
https://{{server}}:8544/nbi-notification/api/v1/notifications/subscriptions/<subscriptionId>
```

在这两种情况下，我们都不会在请求体中发送任何参数。

这就结束了我们关于使用请求-响应模型和事件驱动模型与 NMS 和网络控制器集成的讨论。这两种方法在与其他管理系统集成时都将为你提供一个良好的起点。

# 摘要

在本章中，我们介绍了网络自动化，包括其优势和它为电信服务提供商带来的挑战。我们还讨论了网络自动化的关键用例。在介绍之后，我们讨论了网络自动化与网络设备交互时可用的传输协议。网络自动化可以以多种方式采用。我们首先探讨了如何使用 Python 中的 SSH 协议直接与网络设备交互。我们使用了 Paramiko、Netmiko 和 NAPALM Python 库从设备获取配置，并详细说明了如何将此配置推送到网络设备。接下来，我们讨论了如何使用 Python 中的 NETCONF 与网络设备交互。我们提供了与 NETCONF 一起工作的代码示例，并使用 ncclient 库获取 IP 接口配置。我们还使用相同的库更新了网络设备上的 IP 接口。

在本章的最后部分，我们探讨了如何与网络管理系统，如诺基亚 NSP 进行交互。我们使用 Python 作为 REST API 客户端和 Kafka 消费者与诺基亚 NSP 系统进行交互。我们提供了一些代码示例，说明了如何获取认证令牌，然后向 NMS 发送 REST API 以检索配置数据并更新设备上的网络配置。

本章包含了一些代码示例，使您熟悉使用 Python 通过 SSH、NETCONF 协议以及使用 NMS 级 REST API 与设备交互。如果您是自动化工程师，并希望利用 Python 功能在您的领域脱颖而出，这种实际知识至关重要。

本章总结了本书的内容。我们不仅涵盖了 Python 的高级概念，还提供了在许多高级领域使用 Python 的见解，例如数据处理、无服务器计算、Web 开发、机器学习和网络自动化。

# 问题

1.  Paramiko 库中用于连接设备的常用类叫什么名字？

1.  NETCONF 有哪四层？

1.  你可以直接将配置推送到 NETCONF 中的 `running` 数据库吗？

1.  为什么 gNMI 在数据收集方面比网络配置更好？

1.  RESTCONF 是否提供与 NETCONF 相同的功能，但通过 REST 接口提供？

1.  Apache Kafka 中的发布者和消费者是什么？

# 进一步阅读

+   《*精通 Python 网络*》，作者 Eric Chou。

+   《*实用网络自动化* 第二版》，作者 Abhishek Ratan。

+   《*网络可编程性和自动化*》，作者 Jason Edelman。

+   《*Paramiko 官方文档*》可在 [`docs.paramiko.org/`](http://docs.paramiko.org/) 查找。

+   《*Netmiko 官方文档*》可在 [`ktbyers.github.io/`](https://ktbyers.github.io/) 查找。

+   《*NAPALM 官方文档*》可在 [`napalm.readthedocs.io/`](https://napalm.readthedocs.io/) 查找。

+   《*ncclient 官方文档*》可在 [`ncclient.readthedocs.io/`](https://ncclient.readthedocs.io/) 查找。

+   *NETCONF YANG 模型* 可以在 [`github.com/YangModels/yang`](https://github.com/YangModels/yang) 找到。

+   *诺基亚 NSP API 文档* 可在 [`network.developer.nokia.com/api-documentation/`](https://network.developer.nokia.com/api-documentation/) 找到。

# 答案

1.  `paramiko.SSHClient`类。

1.  内容、操作、消息和传输。

1.  如果网络设备不支持`candidate`数据库，它通常允许直接更新`running`数据库。

1.  gNMI 基于 gRPC，这是一个由谷歌引入的用于移动客户端和云应用之间 RPC 调用的协议。该协议针对数据传输进行了优化，这使得它在从网络设备收集数据方面比配置它们更有效率。

1.  RESTCONF 通过 REST 接口提供了 NETCONF 的大部分功能，但它并没有暴露 NETCONF 的所有操作。

1.  发布者是发送消息到 Kafka 主题（类别）作为事件的客户端程序，而消费者是读取并处理从 Kafka 主题消息的客户端应用程序。
