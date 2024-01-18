# 使用Scapy框架

Scapy是一个强大的Python工具，用于构建和制作数据包，然后将其发送到网络。您可以构建任何类型的网络流并将其发送到网络。它可以帮助您使用不同的数据包流测试您的网络，并操纵从源返回的响应。

本章将涵盖以下主题：

+   了解Scapy框架

+   安装Scapy

+   使用Scapy生成数据包和网络流

+   捕获和重放数据包

# 了解Scapy

Scapy ([https://scapy.net](https://scapy.net))是强大的Python工具之一，用于捕获、嗅探、分析和操纵网络数据包。它还可以构建分层协议的数据包结构，并将wiuthib流注入到网络中。您可以使用它在许多协议之上构建广泛的协议，并设置协议内每个字段的细节，或者更好地让Scapy发挥其魔力并选择适当的值，以便每个值都可以有一个有效的帧。如果用户没有覆盖，Scapy将尝试使用数据包的默认值。以下值将自动设置为每个流：

+   IP源根据目的地和路由表选择

+   校验和会自动计算

+   源Mac根据输出接口选择

+   以太网类型和IP协议由上层确定

Scapy可以编程地将帧注入到流中并重新发送。例如，您可以将802.1q VLAN ID注入到流中并重新发送，以执行对网络的攻击或分析。此外，您可以使用`Graphviz`和`ImageMagick`模块可视化两个端点之间的对话并绘制图形。

Scapy有自己的**领域特定语言**（**DSL**），使用户能够描述他想要构建或操纵的数据包，并以相同的结构接收答案。这与Python内置的数据类型（如列表和字典）非常好地配合和集成。我们将在示例中看到，从网络接收的数据包实际上是一个Python列表，我们可以对它们进行常规列表函数的迭代。

# 安装Scapy

Scapy支持Python 2.7.x和3.4+，从Scapy版本2.x开始。但是，对于低于2.3.3的版本，Scapy需要Python 2.5和2.7，或者3.4+用于之后的版本。由于我们已经安装了最新的Python版本，应该可以毫无问题地运行最新版本的Scapy。

此外，Scapy还有一个较旧的版本（1.x），已经不再支持Python 3，仅在Python 2.4上运行。

# 基于Unix的系统

要获取最新版本，您需要使用python pip：

```py
pip install scapy 
```

输出应该类似于以下屏幕截图：![](../images/00207.jpeg)

要验证Scapy是否成功安装，请访问Python控制台并尝试将`scapy`模块导入其中。如果控制台没有报告任何导入错误，则安装已成功完成：

![](../images/00208.jpeg)

需要一些附加软件包来可视化对话和捕获数据包。根据您的平台使用以下命令安装附加软件包：

# 在Debian和Ubuntu上安装

运行以下命令安装附加软件包：

```py
sudo apt-get install tcpdump graphviz imagemagick python-gnuplot python-cryptography python-pyx
```

# 在Red Hat/CentOS上安装

运行以下命令安装附加软件包：

```py
yum install tcpdump graphviz imagemagick python-gnuplot python-crypto python-pyx -y
```

如果在基于CentOS的系统上找不到上述软件包中的任何一个，请安装`epel`存储库并更新系统。

# Windows和macOS X支持

Scapy是专为基于Linux的系统构建和设计的。但它也可以在其他操作系统上运行。您可以在Windows和macOS上安装和移植它，每个平台都有一些限制。对于基于Windows的系统，您基本上需要删除WinPcap驱动程序，并改用Npcap驱动程序（不要同时安装两个版本，以避免任何冲突问题）。您可以在[http://scapy.readthedocs.io/en/latest/installation.html#windows](http://scapy.readthedocs.io/en/latest/installation.html#windows)上阅读有关Windows安装的更多信息。

对于macOS X，您需要安装一些Python绑定并使用libdnet和libpcap库。完整的安装步骤可在[http://scapy.readthedocs.io/en/latest/installation.html#mac-os-x](http://scapy.readthedocs.io/en/latest/installation.html#mac-os-x)上找到。

# 使用Scapy生成数据包和网络流

正如我们之前提到的，Scapy有自己的DSL语言，与Python集成。此外，您可以直接访问Scapy控制台，并开始直接从Linux shell发送和接收数据包：

```py
sudo scapy 
```

前面命令的输出如下：![](../images/00209.jpeg)

请注意，有一些关于一些缺少的*可选*软件包的警告消息，例如`matplotlib`和`PyX`，但这应该没问题，不会影响Scapy的核心功能。

我们可以首先通过运行`ls()`函数来检查Scapy中支持的协议。列出所有支持的协议：

```py
>>> ls()
```

输出非常冗长，如果在此处发布，将跨越多个页面，因此您可以快速查看终端，以检查它。

现在让我们开发一个hello world应用程序，并使用SCAPY运行它。该程序将向服务器的网关发送一个简单的ICMP数据包。我安装了Wireshark并配置它以监听将从自动化服务器（托管Scapy）接收流的网络接口。

现在，在Scapy终端上，执行以下代码：

```py
>>> send(IP(dst="10.10.10.1")/ICMP()/"Welcome to Enterprise Automation Course") 
```

返回到Wireshark，你应该看到通信：

![](../images/00210.jpeg)

让我们分析Scapy执行的命令：

+   **Send：**这是Scapy **Domain Specific Language** (**DSL**)中的内置函数，指示Scapy发送单个数据包（并不监听任何响应；它只发送一个数据包并退出）。

+   **IP：**现在，在这个类中，我们将开始构建数据包层。从IP层开始，我们需要指定将接收数据包的目标主机（在这种情况下，我们使用`dst`参数来指定目的地）。还要注意，我们可以在`src`参数中指定源IP；但是，Scapy将查询主机路由表并找到合适的源IP，并将其放入数据包中。您可以提供其他参数，例如**生存时间**（**TTL**），Scapy将覆盖默认值。

+   **/**：虽然它看起来像是Python中常用的普通除法运算符，但在Scapy DSL中，它用于区分数据包层，并将它们堆叠在一起。

+   **ICMP():**用于创建具有默认值的ICMP数据包的内置类。可以向函数提供的值之一是ICMP类型，它确定消息类型：`echo`，`echo reply`，`unreachable`等。

+   **欢迎来到企业自动化课程：**如果将字符串注入ICMP有效载荷中，Scapy将自动将其转换为适当的格式。

请注意，我们没有在堆栈中指定以太网层，并且没有提供任何mac地址（源或目的地）。这在Scapy中默认填充，以创建一个有效的帧。它将自动检查主机ARP表，并找到源接口的mac地址（如果存在，也是目的地），然后将它们格式化为以太网帧。

在继续下一个示例之前，需要注意的最后一件事是，您可以使用与我们之前用于列出所有支持的协议的`ls()`函数相同的函数，以获取每个协议的默认值，然后在调用协议时将其设置为任何其他值。

![](../images/00211.jpeg)

现在让我们做一些更复杂（和邪恶的）事情！假设我们有两台路由器之间形成VRRP关系，并且我们需要打破这种关系以成为新的主机，或者至少在网络中创建一个抖动问题，如下拓扑图所示：

![](../images/00212.jpeg)

请注意，配置为运行VRRP的路由器加入多播地址（`255.0.0.18`）以接收其他路由器的广告。VRRP数据包的目标MAC地址应该包含最后两个数字的VRRP组号。它还包含在路由器之间选举过程中使用的路由器优先级。我们将构建一个Scapy脚本，该脚本将发送一个具有比网络中配置的更高优先级的VRRP通告。这将导致我们的Scapy服务器被选为新的主机：

```py
from scapy.layers.inet import * from scapy.layers.vrrp import VRRP

vrrp_packet = Ether(src="00:00:5e:00:01:01",dst="01:00:5e:00:00:30")/IP(src="10.10.10.130", dst="224.0.0.18")/VRRP(priority=254, addrlist=["10.10.10.1"]) sendp(vrrp_packet, inter=2, loop=1) 
```

在这个例子中：

+   首先，我们从`scapy.layers`模块中导入了一些需要的层，我们将这些层叠加在一起。例如，`inet`模块包含了`IP()`、`Ether()`、`ARP()`、`ICMP()`等层。

+   此外，我们还需要VRRP层，可以从`scapy.layers.vrrp`中导入。

+   其次，我们将构建一个VRRP数据包并将其存储在`vrrp_packet`变量中。该数据包包含以太网帧内的mac地址中的VRRP组号。多播地址将位于IP层内。此外，我们将在VRRP层内配置一个更高的优先级号码。这样我们将拥有一个有效的VRRP通告，路由器将接受它。我们为每个层提供了信息，例如目标mac地址（VRRP MAC +组号）和多播IP（`225.0.0.18`）。

+   最后，我们使用了`sendp()`函数，并向其提供了一个精心制作的`vrrp_packet`。`sendp()`函数将在第2层发送数据包，与我们在上一个示例中使用的`send()`函数发送数据包的方式不同，后者是在第3层发送数据包。`sendp()`函数不会像`send()`函数那样尝试解析主机名，并且只会在第2层操作。此外，由于我们需要连续发送此通告，因此我们配置了`loop`和`inter`参数，以便每2秒发送一次通告。

脚本输出为：

![](../images/00213.jpeg)您可以将此攻击与ARP欺骗和VLAN跳跃攻击相结合，以便在第2层更改mac地址，切换到Scapy服务器的MAC地址，并执行**中间人**（**MITM**）攻击。

Scapy还包含一些执行扫描的类。例如，您可以使用`arping()`在网络范围内执行ARP扫描，并在其中指定IP地址的正则表达式格式。Scapy将向这些子网上的所有主机发送ARP请求并检查回复：

```py
from scapy.layers.inet import *  arping("10.10.10.*")
```

![](../images/00214.jpeg)

脚本输出为：

![](../images/00215.jpeg)

根据接收到的数据包，只有一个主机回复SCAPY，这意味着它是扫描子网上唯一的主机。回复中列出了主机的mac地址和IP地址。

# 捕获和重放数据包

Scapy具有监听网络接口并捕获其所有传入数据包的能力。它可以以与`tcpdump`相同的方式将其写入`pcap`文件，但是Scapy提供了额外的函数，可以再次读取和重放`pcap`文件。

从简单的数据包重放开始，我们将指示Scapy读取从网络中捕获的正常`pcap`文件（使用`tcpdump`或Scapy本身）并将其再次发送到网络。如果我们需要测试网络的行为是否通过特定的流量模式，这将非常有用。例如，我们可能已经配置了网络防火墙以阻止FTP通信。我们可以通过使用Scapy重放的FTP数据来测试防火墙的功能。

在这个例子中，我们有捕获的FTP `pcap`文件，我们需要将其重新发送到网络：

```py
from scapy.layers.inet import * from pprint import pprint
pkts = PcapReader("/root/ftp_data.pcap") #should be in wireshark-tcpdump format   for pkt in pkts:
  pprint(pkt.show()) 
```

`PcapReader()`将`pcap`文件作为输入，并对其进行分析，以单独获取每个数据包，并将其作为`pkts`列表中的一个项目添加。现在我们可以遍历列表并显示每个数据包的内容。

脚本输出为：

![](../images/00216.jpeg)

此外，您可以通过`get_layer()`函数获取特定层的信息，该函数访问数据包层。例如，如果我们有兴趣获取没有标头的原始数据，以便构建传输文件，我们可以使用以下脚本获取十六进制中所需的数据，然后稍后将其转换为ASCII：

```py
from scapy.layers.inet import * from pprint import pprint
pkts = PcapReader("/root/ftp_data.pcap") #should be in wireshark-tcpdump format   ftp_data = b"" for pkt in pkts:
  try:
  ftp_data += pkt.get_layer(Raw).load
    except:
  pass
```

请注意，我们必须用try-except子句包围`get_layer()`方法，因为某些层不包含原始数据（例如FTP控制消息）。Scapy会抛出错误，脚本将退出。此外，我们可以将脚本重写为一个`if`子句，只有在数据包中包含原始层时才会向`ftp_data`添加内容。

为了避免在读取`pcap`文件时出现任何错误，请确保将您的`pcap`文件保存（或导出）为Wireshark/tcpdump格式，如下所示，而不是默认格式：![](../images/00217.jpeg)

# 向数据包中注入数据

在将数据包重新发送到网络之前，我们可以操纵数据包并更改其内容。由于我们的数据包实际上存储为列表中的项目，我们可以遍历这些项目并替换特定信息。例如，我们可以更改MAC地址、IP地址，或者为每个数据包或符合特定条件的特定数据包添加附加层。但是，我们应该注意，在特定层（如IP和TCP）中操纵数据包并更改内容将导致整个层的校验和无效，接收方可能因此丢弃数据包。

Scapy有一个令人惊奇的功能（是的，我知道，我多次说了令人惊奇，但Scapy确实是一个很棒的工具）。如果我们在`pcap`文件中删除原始内容，它将基于新内容自动为我们计算校验和。

因此，我们将修改上一个脚本并更改一些数据包参数，然后在发送数据包到网络之前重新构建校验和：

```py

from scapy.layers.inet import * from pprint import pprint
pkts = PcapReader("/root/ftp_data.pcap") #should be in wireshark-tcpdump format     p_out = []   for pkt in pkts:
  new_pkt = pkt.payload

    try:
  new_pkt[IP].src = "10.10.88.100"
  new_pkt[IP].dst = "10.10.88.1"
  del (new_pkt[IP].chksum)
  del (new_pkt[TCP].chksum)
  except:
  pass    pprint(new_pkt.show())
  p_out.append(new_pkt) send(PacketList(p_out), iface="eth0")
```

在上一个脚本中：

+   我们使用`PcapReader()`类来读取FTP `pcap`文件的内容，并将数据包存储在`pkts`变量中。

+   然后，我们遍历数据包并将有效载荷分配给`new_pkt`，以便我们可以操纵内容。

+   请记住，数据包本身被视为来自该类的对象。我们可以访问`src`和`dst`成员，并将它们设置为任何所需的值。在这里，我们将目的地设置为网关，将源设置为与原始数据包不同的值。

+   设置新的IP值将使校验和无效，因此我们使用`del`关键字删除了IP和TCP校验和。Scapy将根据新数据包内容重新计算它们。

+   最后，我们将`new_pkt`附加到空的`p_out`列表中，并使用`send()`函数发送它。请注意，我们可以在发送函数中指定退出接口，或者只需离开它，Scapy将查询主机路由表；它将为每个数据包获取正确的退出接口。

脚本输出为：

![](../images/00218.jpeg)

此外，如果我们仍然在网关上运行Wireshark，我们会注意到Wireshark捕获了在重新计算后设置校验和值的`ftp`数据包流：

![](../images/00219.jpeg)

# 数据包嗅探

Scapy有一个名为`sniff()`的内置数据包捕获函数。默认情况下，如果您不指定任何过滤器或特定接口，它将监视所有接口并捕获所有数据包：

```py
from scapy.all import * from pprint import pprint

print("Begin capturing all packets from all interfaces. send ctrl+c to terminate and print summary") pkts = sniff()   pprint(pkts.summary())
```

脚本输出为：

![](../images/00220.jpeg)

当然，您可以提供过滤器和特定接口来监视是否满足条件。例如，在前面的输出中，我们可以看到混合了ICMP、TCP、SSH和DHCP流量命中了所有接口。如果我们只对在eth0上获取ICMP流量感兴趣，那么我们可以提供过滤器和`iface`参数来嗅探函数，并且它将只过滤所有流量并记录只有ICMP的数据：

```py
from scapy.all import * from pprint import pprint

print("Begin capturing all packets from all interfaces. send ctrl+c to terminate and print summary") pkts = sniff(iface="eth0", filter="icmp")   pprint(pkts.summary())
```

脚本输出如下：

![](../images/00221.jpeg)

请注意，我们只捕获eth0接口上的ICMP通信，所有其他数据包都由于应用在它们上的过滤器而被丢弃。*iface*值接受我们在脚本中使用的单个接口或要监视它们的接口列表。

`sniff`的高级功能之一是`stop_filter`，它是应用于每个数据包的Python函数，用于确定我们是否必须在该数据包之后停止捕获。例如，如果我们设置`stop_filter = lambda x: x.haslayer(TCP)`，那么一旦我们命中具有TCP层的数据包，我们将停止捕获。此外，`store`选项允许我们将数据包存储在内存中（默认情况下已启用）或在对每个数据包应用特定函数后丢弃它们。如果您正在从SCAPY中获取来自线缆的实时流量，并且不希望将其写入内存，那么将`sniff`函数中的store参数设置为false，然后SCAPY将在丢弃原始数据包之前应用您开发的任何自定义函数（例如获取数据包的一些信息或将其重新发送到不同的目的地等）。这将在嗅探期间节省一些内存资源。

# 将数据包写入pcap

最后，我们可以将嗅探到的数据包写入标准的`pcap`文件，并像往常一样使用Wireshark打开它。这是通过一个简单的`wrpcap()`函数实现的，它将数据包列表写入`pcap`文件。`wrpcap()`函数接受两个参数——第一个是文件位置的完整路径，第二个是在使用`sniff()`函数之前捕获的数据包列表：

```py
from scapy.all import *   print("Begin capturing all packets from all interfaces. send ctrl+c to terminate and print summary") pkts = sniff(iface="eth0", filter="icmp")   wrpcap("/root/icmp_packets_eth0.pcap",pkts)
```

# 摘要

在本章中，我们学习了如何利用Scapy框架构建任何类型的数据包，包含任何网络层，并用我们的值填充它。此外，我们还看到了如何在接口上捕获数据包并重放它们。
