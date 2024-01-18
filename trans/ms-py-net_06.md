# 使用Python进行网络安全

在我看来，网络安全是一个难以撰写的话题。原因不是技术上的，而是与设定正确的范围有关。网络安全的边界如此之广，以至于它们触及OSI模型的所有七层。从窃听的第1层到传输协议漏洞的第4层，再到中间人欺骗的第7层，网络安全无处不在。问题加剧了所有新发现的漏洞，有时似乎以每日的速度出现。这甚至没有包括网络安全的人为社会工程方面。

因此，在本章中，我想设定我们将讨论的范围。与迄今为止一样，我们将主要专注于使用Python来处理OSI第3和第4层的网络设备安全。我们将研究可以用于管理个别网络设备以实现安全目的的Python工具，以及使用Python作为连接不同组件的粘合剂。希望我们可以通过在不同的OSI层中使用Python来全面地处理网络安全。

在本章中，我们将研究以下主题：

+   实验室设置

+   Python Scapy用于安全测试

+   访问列表

+   使用Python进行Syslog和UFW的取证分析

+   其他工具，如MAC地址过滤列表、私有VLAN和Python IP表绑定。

# 实验室设置

本章中使用的设备与之前的章节有些不同。在之前的章节中，我们通过专注于手头的主题来隔离特定的设备。对于本章，我们将在我们的实验室中使用更多的设备，以便说明我们将使用的工具的功能。连接和操作系统信息很重要，因为它们对我们稍后将展示的安全工具产生影响。例如，如果我们想应用访问列表来保护服务器，我们需要知道拓扑图是什么样的，客户端的连接方向是什么。Ubuntu主机的连接与我们迄今为止看到的有些不同，因此如果需要，当您稍后看到示例时，请参考本实验室部分。

我们将使用相同的Cisco VIRL工具，其中包括四个节点：两个主机和两个网络设备。如果您需要关于Cisco VIRL的复习，请随时返回到[第2章](8cefc139-8dfa-4250-81bf-928231e20b22.xhtml)，*低级网络设备交互*，我们在那里首次介绍了这个工具：

![](assets/620c32a2-6ce0-471f-a165-264f14e09454.png)实验拓扑图列出的IP地址在您自己的实验室中将是不同的。它们在这里列出，以便在本章的其余部分中进行简单参考。

如图所示，我们将把顶部的主机重命名为客户端，底部的主机重命名为服务器。这类似于互联网客户端试图在我们的网络中访问公司服务器。我们将再次使用共享平面网络选项来访问设备进行带外管理：

![](assets/2365022c-82e9-4928-ab1e-620b2e95b72c.png)

对于两个交换机，我将选择**开放最短路径优先**（**OSPF**）作为`IGP`，并将两个设备放入区域`0`。默认情况下，`BGP`已打开，并且两个设备都使用AS 1。从配置自动生成中，连接到Ubuntu主机的接口被放入OSPF区域`1`，因此它们将显示为区间路由。NX-OSv的配置如下所示，IOSv的配置和输出类似：

```py
 interface Ethernet2/1
 description to iosv-1
 no switchport
 mac-address fa16.3e00.0001
 ip address 10.0.0.6/30
 ip router ospf 1 area 0.0.0.0
 no shutdown

 interface Ethernet2/2
 description to Client
 no switchport
 mac-address fa16.3e00.0002
 ip address 10.0.0.9/30
 ip router ospf 1 area 0.0.0.0
 no shutdown

 nx-osv-1# sh ip route
 <skip>
 10.0.0.12/30, ubest/mbest: 1/0
 *via 10.0.0.5, Eth2/1, [110/41], 04:53:02, ospf-1, intra
 192.168.0.2/32, ubest/mbest: 1/0
 *via 10.0.0.5, Eth2/1, [110/41], 04:53:02, ospf-1, intra
 <skip>
```

OSPF邻居和NX-OSv的BGP输出如下所示，IOSv的输出类似：

```py
nx-osv-1# sh ip ospf neighbors
 OSPF Process ID 1 VRF default
 Total number of neighbors: 1
 Neighbor ID Pri State Up Time Address Interface
 192.168.0.2 1 FULL/DR 04:53:00 10.0.0.5 Eth2/1

nx-osv-1# sh ip bgp summary
BGP summary information for VRF default, address family IPv4 Unicast
BGP router identifier 192.168.0.1, local AS number 1
BGP table version is 5, IPv4 Unicast config peers 1, capable peers 1
2 network entries and 2 paths using 288 bytes of memory
BGP attribute entries [2/288], BGP AS path entries [0/0]
BGP community entries [0/0], BGP clusterlist entries [0/0]

Neighbor V AS MsgRcvd MsgSent TblVer InQ OutQ Up/Down State/PfxRcd
192.168.0.2 4 1 321 297 5 0 0 04:52:56 1
```

我们网络中的主机正在运行Ubuntu 14.04，与迄今为止我们一直在使用的Ubuntu VM 16.04类似：

```py
cisco@Server:~$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description: Ubuntu 14.04.2 LTS
Release: 14.04
Codename: trusty
```

在两台Ubuntu主机上，有两个网络接口，`eth0`和`eth1`。`eth0`连接到管理网络（`172.16.1.0/24`），而`eth1`连接到网络设备（`10.0.0.x/30`）。设备环回的路由直接连接到网络块，远程主机网络通过默认路由静态路由到`eth1`：

```py
cisco@Client:~$ route -n
Kernel IP routing table
Destination Gateway Genmask Flags Metric Ref Use Iface
0.0.0.0 172.16.1.2 0.0.0.0 UG 0 0 0 eth0
10.0.0.4 10.0.0.9 255.255.255.252 UG 0 0 0 eth1
10.0.0.8 0.0.0.0 255.255.255.252 U 0 0 0 eth1
10.0.0.8 10.0.0.9 255.255.255.248 UG 0 0 0 eth1
172.16.1.0 0.0.0.0 255.255.255.0 U 0 0 0 eth0
192.168.0.1 10.0.0.9 255.255.255.255 UGH 0 0 0 eth1
192.168.0.2 10.0.0.9 255.255.255.255 UGH 0 0 0 eth1
```

为了验证客户端到服务器的路径，让我们ping和跟踪路由，确保我们的主机之间的流量通过网络设备而不是默认路由：

```py
## Our server IP is 10.0.0.14 cisco@Server:~$ ifconfig
<skip>
eth1 Link encap:Ethernet HWaddr fa:16:3e:d6:83:02
 inet addr:10.0.0.14 Bcast:10.0.0.15 Mask:255.255.255.252

## From the client ping toward server
cisco@Client:~$ ping -c 1 10.0.0.14
PING 10.0.0.14 (10.0.0.14) 56(84) bytes of data.
64 bytes from 10.0.0.14: icmp_seq=1 ttl=62 time=6.22 ms

--- 10.0.0.14 ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 6.223/6.223/6.223/0.000 ms

## Traceroute from client to server
cisco@Client:~$ traceroute 10.0.0.14
traceroute to 10.0.0.14 (10.0.0.14), 30 hops max, 60 byte packets
 1 10.0.0.9 (10.0.0.9) 11.335 ms 11.745 ms 12.113 ms
 2 10.0.0.5 (10.0.0.5) 24.221 ms 41.635 ms 41.638 ms
 3 10.0.0.14 (10.0.0.14) 37.916 ms 38.275 ms 38.588 ms
cisco@Client:~$
```

太好了！我们有了实验室，现在准备使用Python来查看一些安全工具和措施。

# Python Scapy

Scapy（[https://scapy.net](https://scapy.net/)）是一个功能强大的基于Python的交互式数据包构建程序。除了一些昂贵的商业程序外，据我所知，很少有工具可以做到Scapy所能做的。这是我在Python中最喜欢的工具之一。

Scapy的主要优势在于它允许您从非常基本的级别构建自己的数据包。用Scapy的创作者的话来说：

“Scapy是一个功能强大的交互式数据包操作程序。它能够伪造或解码大量协议的数据包，将它们发送到网络上，捕获它们，匹配请求和响应，等等……与大多数其他工具不同，您不会构建作者没有想象到的东西。这些工具是为了特定的目标而构建的，不能偏离太多。”

让我们来看看这个工具。

# 安装Scapy

在撰写本文时，Scapy 2.3.1支持Python 2.7。不幸的是，关于Scapy对Python 3的支持出现了一些问题，对于Scapy 2.3.3来说，这仍然是相对较新的。对于您的环境，请随时尝试使用版本2.3.3及更高版本的Python 3。在本章中，我们将使用Python 2.7的Scapy 2.3.1。如果您想了解选择背后的原因，请参阅信息侧栏。

关于Scapy在Python 3中的支持的长篇故事是，2015年有一个独立的Scapy分支，旨在仅支持Python 3。该项目被命名为`Scapy3k`。该分支与主要的Scapy代码库分道扬镳。如果您阅读本书的第一版，那是写作时提供的信息。关于PyPI上的`python3-scapy`和Scapy代码库的官方支持存在混淆。我们的主要目的是在本章中了解Scapy，因此我选择使用较旧的基于Python 2的Scapy版本。

在我们的实验室中，由于我们正在从客户端向目标服务器构建数据包源，因此需要在客户端上安装Scapy：

```py
cisco@Client:~$ sudo apt-get update
cisco@Client:~$ sudo apt-get install git
cisco@Client:~$ git clone https://github.com/secdev/scapy
cisco@Client:~$ cd scapy/
cisco@Client:~/scapy$ sudo python setup.py install
```

这是一个快速测试，以确保软件包已正确安装：

```py
cisco@Client:~/scapy$ python
Python 2.7.6 (default, Mar 22 2014, 22:59:56)
[GCC 4.8.2] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> from scapy.all import *
```

# 交互式示例

在我们的第一个示例中，我们将在客户端上构建一个**Internet控制消息协议**（**ICMP**）数据包，并将其发送到服务器。在服务器端，我们将使用`tcpdump`和主机过滤器来查看传入的数据包：

```py
## Client Side
cisco@Client:~/scapy$ sudo scapy
<skip>
Welcome to Scapy (2.3.3.dev274)
>>> send(IP(dst="10.0.0.14")/ICMP())
.
Sent 1 packets.
>>>

## Server Side
cisco@Server:~$ sudo tcpdump -i eth1 host 10.0.0.10
tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
listening on eth1, link-type EN10MB (Ethernet), capture size 65535 bytes
02:45:16.400162 IP 10.0.0.10 > 10.0.0.14: ICMP echo request, id 0, seq 0, length 8
02:45:16.400192 IP 10.0.0.14 > 10.0.0.10: ICMP echo reply, id 0, seq 0, length 8
```

正如您所看到的，使用Scapy构建数据包非常简单。Scapy允许您使用斜杠（`/`）作为分隔符逐层构建数据包。`send`函数在第3层级别操作，负责路由和第2层级别。还有一个`sendp()`替代方案，它在第2层级别操作，这意味着您需要指定接口和链路层协议。

让我们通过使用发送请求（`sr`）函数来捕获返回的数据包。我们使用`sr`的特殊变体，称为`sr1`，它只返回一个回答发送的数据包：

```py
>>> p = sr1(IP(dst="10.0.0.14")/ICMP())
>>> p
<IP version=4L ihl=5L tos=0x0 len=28 id=26713 flags= frag=0L ttl=62 proto=icmp chksum=0x71 src=10.0.0.14 dst=10.0.0.10 options=[] |<ICMP type=echo-reply code=0 chksum=0xffff id=0x0 seq=0x0 |>>
```

需要注意的一点是，`sr()`函数本身返回一个包含已回答和未回答列表的元组：

```py
>>> p = sr(IP(dst="10.0.0.14")/ICMP()) 
>>> type(p)
<type 'tuple'>

## unpacking
>>> ans,unans = sr(IP(dst="10.0.0.14")/ICMP())
>>> type(ans)
<class 'scapy.plist.SndRcvList'>
>>> type(unans)
<class 'scapy.plist.PacketList'>
```

如果我们只看已回答的数据包列表，我们可以看到它是另一个包含我们发送的数据包以及返回的数据包的元组：

```py
>>> for i in ans:
...     print(type(i))
...
<type 'tuple'>
>>> for i in ans:
...     print i
...
(<IP frag=0 proto=icmp dst=10.0.0.14 |<ICMP |>>, <IP version=4L ihl=5L tos=0x0 len=28 id=27062 flags= frag=0L ttl=62 proto=icmp chksum=0xff13 src=10.0.0.14 dst=10.0.0.10 options=[] |<ICMP type=echo-reply code=0 chksum=0xffff id=0x0 seq=0x0 |>>)
```

Scapy还提供了一个第7层的构造，比如`DNS`查询。在下面的例子中，我们正在查询一个开放的DNS服务器来解析`www.google.com`：

```py
>>> p = sr1(IP(dst="8.8.8.8")/UDP()/DNS(rd=1,qd=DNSQR(qname="www.google.com")))
>>> p
<IP version=4L ihl=5L tos=0x0 len=76 id=21743 flags= frag=0L ttl=128 proto=udp chksum=0x27fa src=8.8.8.8 dst=172.16.1.152 options=[] |<UDP sport=domain dport=domain len=56 chksum=0xc077 |<DNS id=0 qr=1L opcode=QUERY aa=0L tc=0L rd=1L ra=1L z=0L ad=0L cd=0L rcode=ok qdcount=1 ancount=1 nscount=0 arcount=0 qd=<DNSQR qname='www.google.com.' qtype=A qclass=IN |> an=<DNSRR rrname='www.google.com.' type=A rclass=IN ttl=299 rdata='172.217.3.164' |> ns=None ar=None |>>>
>>>
```

# 嗅探

Scapy还可以用于轻松捕获网络上的数据包：

```py
>>> a = sniff(filter="icmp and host 172.217.3.164", count=5)
>>> a.show()
0000 Ether / IP / TCP 192.168.225.146:ssh > 192.168.225.1:50862 PA / Raw
0001 Ether / IP / ICMP 192.168.225.146 > 172.217.3.164 echo-request 0 / Raw
0002 Ether / IP / ICMP 172.217.3.164 > 192.168.225.146 echo-reply 0 / Raw
0003 Ether / IP / ICMP 192.168.225.146 > 172.217.3.164 echo-request 0 / Raw
0004 Ether / IP / ICMP 172.217.3.164 > 192.168.225.146 echo-reply 0 / Raw
>>>
```

我们可以更详细地查看数据包，包括原始格式：

```py
>>> for i in a:
...     print i.show()
...
<skip>
###[ Ethernet ]###
 dst= <>
 src= <>
 type= 0x800
###[ IP ]###
 version= 4L
 ihl= 5L
 tos= 0x0
 len= 84
 id= 15714
 flags= DF
 frag= 0L
 ttl= 64
 proto= icmp
 chksum= 0xaa8e
 src= 192.168.225.146
 dst= 172.217.3.164
 options
###[ ICMP ]###
 type= echo-request
 code= 0
 chksum= 0xe1cf
 id= 0xaa67
 seq= 0x1
###[ Raw ]###
 load= 'xd6xbfxb1Xx00x00x00x00x1axdcnx00x00x00x00x00x10x11x12x13x14x15x16x17x18x19x1ax1bx1cx1dx1ex1f !"#$%&'()*+,-./01234567'
None
```

我们已经看到了Scapy的基本工作原理。让我们继续看看如何使用Scapy进行一些常见的安全测试。

# TCP端口扫描

任何潜在黑客的第一步几乎总是尝试了解网络上开放的服务，这样他们就可以集中精力进行攻击。当然，我们需要打开某些端口以为客户提供服务；这是我们需要接受的风险的一部分。但我们还应该关闭任何不必要暴露更大攻击面的其他开放端口。我们可以使用Scapy对我们自己的主机进行简单的TCP开放端口扫描。

我们可以发送一个`SYN`数据包，看服务器是否会返回`SYN-ACK`：

```py
>>> p = sr1(IP(dst="10.0.0.14")/TCP(sport=666,dport=23,flags="S"))
>>> p.show()
###[ IP ]###
 version= 4L
 ihl= 5L
 tos= 0x0
 len= 40
 id= 25373
 flags= DF
 frag= 0L
 ttl= 62
 proto= tcp
 chksum= 0xc59b
 src= 10.0.0.14
 dst= 10.0.0.10
 options
###[ TCP ]###
 sport= telnet
 dport= 666
 seq= 0
 ack= 1
 dataofs= 5L
 reserved= 0L
 flags= RA
 window= 0
 chksum= 0x9907
 urgptr= 0
 options= {}
```

请注意，在这里的输出中，服务器对TCP端口`23`响应了`RESET+ACK`。然而，TCP端口`22`（SSH）是开放的；因此返回了`SYN-ACK`：

```py
>>> p = sr1(IP(dst="10.0.0.14")/TCP(sport=666,dport=22,flags="S"))
>>> p.show()
###[ IP ]###
 version= 4L
<skip>
 proto= tcp
 chksum= 0x28b5
 src= 10.0.0.14
 dst= 10.0.0.10
 options
###[ TCP ]###
 sport= ssh
 dport= 666
<skip>
 flags= SA
<skip>
```

我们还可以扫描从`20`到`22`的一系列目标端口；请注意，我们使用`sr()`进行发送-接收，而不是`sr1()`发送-接收一个数据包的变体：

```py
>>> ans,unans = sr(IP(dst="10.0.0.14")/TCP(sport=666,dport=(20,22),flags="S"))
>>> for i in ans:
...     print i
...
(<IP frag=0 proto=tcp dst=10.0.0.14 |<TCP sport=666 dport=ftp_data flags=S |>>, <IP version=4L ihl=5L tos=0x0 len=40 id=4126 flags=DF frag=0L ttl=62 proto=tcp chksum=0x189b src=10.0.0.14 dst=10.0.0.10 options=[] |<TCP sport=ftp_data dport=666 seq=0 ack=1 dataofs=5L reserved=0L flags=RA window=0 chksum=0x990a urgptr=0 |>>)
(<IP frag=0 proto=tcp dst=10.0.0.14 |<TCP sport=666 dport=ftp flags=S |>>, <IP version=4L ihl=5L tos=0x0 len=40 id=4127 flags=DF frag=0L ttl=62 proto=tcp chksum=0x189a src=10.0.0.14 dst=10.0.0.10 options=[] |<TCP sport=ftp dport=666 seq=0 ack=1 dataofs=5L reserved=0L flags=RA window=0 chksum=0x9909 urgptr=0 |>>)
(<IP frag=0 proto=tcp dst=10.0.0.14 |<TCP sport=666 dport=ssh flags=S |>>, <IP version=4L ihl=5L tos=0x0 len=44 id=0 flags=DF frag=0L ttl=62 proto=tcp chksum=0x28b5 src=10.0.0.14 dst=10.0.0.10 options=[] |<TCP sport=ssh dport=666 seq=4187384571 ack=1 dataofs=6L reserved=0L flags=SA window=29200 chksum=0xaaab urgptr=0 options=[('MSS', 1460)] |>>)
>>>
```

我们还可以指定目标网络而不是单个主机。从`10.0.0.8/29`块中可以看到，主机`10.0.0.9`、`10.0.0.13`和`10.0.0.14`返回了`SA`，这对应于两个网络设备和主机：

```py
>>> ans,unans = sr(IP(dst="10.0.0.8/29")/TCP(sport=666,dport=(22),flags="S"))
>>> for i in ans:
...     print(i)
...
(<IP frag=0 proto=tcp dst=10.0.0.9 |<TCP sport=666 dport=ssh flags=S |>>, <IP version=4L ihl=5L tos=0x0 len=44 id=7304 flags= frag=0L ttl=64 proto=tcp chksum=0x4a32 src=10.0.0.9 dst=10.0.0.10 options=[] |<TCP sport=ssh dport=666 seq=541401209 ack=1 dataofs=6L reserved=0L flags=SA window=17292 chksum=0xfd18 urgptr=0 options=[('MSS', 1444)] |>>)
(<IP frag=0 proto=tcp dst=10.0.0.14 |<TCP sport=666 dport=ssh flags=S |>>, <IP version=4L ihl=5L tos=0x0 len=44 id=0 flags=DF frag=0L ttl=62 proto=tcp chksum=0x28b5 src=10.0.0.14 dst=10.0.0.10 options=[] |<TCP sport=ssh dport=666 seq=4222593330 ack=1 dataofs=6L reserved=0L flags=SA window=29200 chksum=0x6a5b urgptr=0 options=[('MSS', 1460)] |>>)
(<IP frag=0 proto=tcp dst=10.0.0.13 |<TCP sport=666 dport=ssh flags=S |>>, <IP version=4L ihl=5L tos=0x0 len=44 id=41992 flags= frag=0L ttl=254 proto=tcp chksum=0x4ad src=10.0.0.13 dst=10.0.0.10 options=[] |<TCP sport=ssh dport=666 seq=2167267659 ack=1 dataofs=6L reserved=0L flags=SA window=4128 chksum=0x1252 urgptr=0 options=[('MSS', 536)] |>>)
```

根据我们迄今为止学到的知识，我们可以编写一个简单的可重用脚本`scapy_tcp_scan_1.py`。我们从建议的导入`scapy`和`sys`模块开始，用于接收参数：

```py
  #!/usr/bin/env python2

  from scapy.all import *
  import sys
```

`tcp_scan()`函数与我们到目前为止看到的类似：

```py
  def tcp_scan(destination, dport):
      ans, unans = sr(IP(dst=destination)/TCP(sport=666,dport=dport,flags="S"))
      for sending, returned in ans:
          if 'SA' in str(returned[TCP].flags):
              return destination + " port " + str(sending[TCP].dport) + " is open"
          else:
              return destination + " port " + str(sending[TCP].dport) + " is not open"
```

然后我们可以从参数中获取输入，然后在`main()`中调用`tcp_scan()`函数：

```py
  def main():
      destination = sys.argv[1]
      port = int(sys.argv[2])
      scan_result = tcp_scan(destination, port)
      print(scan_result)

  if __name__ == "__main__":
      main()
```

请记住，访问低级网络需要root访问权限；因此，我们的脚本需要以`sudo`执行：

```py
cisco@Client:~$ sudo python scapy_tcp_scan_1.py "10.0.0.14" 23
<skip>
10.0.0.14 port 23 is not open
cisco@Client:~$ sudo python scapy_tcp_scan_1.py "10.0.0.14" 22
<skip>
10.0.0.14 port 22 is open
```

这是一个相对较长的TCP扫描脚本示例，演示了使用Scapy构建自己的数据包的能力。我们在交互式shell中测试了这些步骤，并用一个简单的脚本完成了使用。让我们看看Scapy在安全测试中的一些更多用法。

# Ping集合

假设我们的网络包含Windows、Unix和Linux机器的混合，用户添加了自己的**自带设备**（**BYOD**）；他们可能支持也可能不支持ICMP ping。我们现在可以构建一个文件，其中包含我们网络中三种常见ping的ICMP、TCP和UDP ping，在`scapy_ping_collection.py`中*：*

```py
#!/usr/bin/env python2

from scapy.all import *

def icmp_ping(destination):
    # regular ICMP ping
    ans, unans = sr(IP(dst=destination)/ICMP())
    return ans

def tcp_ping(destination, dport):
    # TCP SYN Scan
    ans, unans = sr(IP(dst=destination)/TCP(dport=dport,flags="S"))
    return ans

def udp_ping(destination):
    # ICMP Port unreachable error from closed port
    ans, unans = sr(IP(dst=destination)/UDP(dport=0))
    return ans
```

在这个例子中，我们还将使用`summary()`和`sprintf()`进行输出：

```py
def answer_summary(answer_list):
 # example of lambda with pretty print
    answer_list.summary(lambda(s, r): r.sprintf("%IP.src% is alive"))
```

如果你想知道为什么在前面的`answer_summary()`函数中有一个lambda，那是一种创建小型匿名函数的方法。基本上，它是一个没有名字的函数。关于它的更多信息可以在[https://docs.python.org/3.5/tutorial/controlflow.html#lambda-expressions](https://docs.python.org/3.5/tutorial/controlflow.html#lambda-expressions)找到。

然后我们可以在一个脚本中执行网络上的三种ping类型：

```py
def main():
    print("** ICMP Ping **")
    ans = icmp_ping("10.0.0.13-14")
    answer_summary(ans)
    print("** TCP Ping **")
    ans = tcp_ping("10.0.0.13", 22)
    answer_summary(ans)
    print("** UDP Ping **")
    ans = udp_ping("10.0.0.13-14")
    answer_summary(ans)

if __name__ == "__main__":
    main()
```

到目前为止，希望你会同意我的观点，通过拥有构建自己的数据包的能力，你可以控制你想要运行的操作和测试的类型。

# 常见攻击

在这个例子中，让我们看看如何构造我们的数据包来进行一些经典攻击，比如*Ping of Death* ([https://en.wikipedia.org/wiki/Ping_of_death](https://en.wikipedia.org/wiki/Ping_of_death)) 和 *Land Attack* ([https://en.wikipedia.org/wiki/Denial-of-service_attack](https://en.wikipedia.org/wiki/Denial-of-service_attack))。这可能是您以前必须使用类似的商业软件付费的网络渗透测试。使用Scapy，您可以在保持完全控制的同时进行测试，并在将来添加更多测试。

第一次攻击基本上发送了一个带有虚假IP头的目标主机，例如长度为2和IP版本3：

```py
def malformed_packet_attack(host):
    send(IP(dst=host, ihl=2, version=3)/ICMP()) 
```

`ping_of_death_attack`由常规的ICMP数据包组成，其负载大于65,535字节：

```py
def ping_of_death_attack(host):
    # https://en.wikipedia.org/wiki/Ping_of_death
    send(fragment(IP(dst=host)/ICMP()/("X"*60000)))
```

`land_attack`想要将客户端响应重定向回客户端本身，并耗尽主机的资源：

```py
  def land_attack(host):
      # https://en.wikipedia.org/wiki/Denial-of-service_attack
      send(IP(src=host, dst=host)/TCP(sport=135,dport=135))
```

这些都是相当古老的漏洞或经典攻击，现代操作系统不再容易受到攻击。对于我们的Ubuntu 14.04主机，前面提到的攻击都不会使其崩溃。然而，随着发现更多安全问题，Scapy是一个很好的工具，可以开始对我们自己的网络和主机进行测试，而不必等待受影响的供应商提供验证工具。这对于零日（未经事先通知发布的）攻击似乎在互联网上变得越来越常见尤其如此。 

# Scapy资源

我们在本章中花了相当多的精力来使用Scapy。这在一定程度上是因为我个人对这个工具的高度评价。我希望你同意Scapy是网络工程师工具箱中必备的伟大工具。Scapy最好的部分是它在一个积极参与的用户社区的不断发展。

我强烈建议至少阅读Scapy教程 [http://scapy.readthedocs.io/en/latest/usage.html#interactive-tutorial](http://scapy.readthedocs.io/en/latest/usage.html#interactive-tutorial)，以及您感兴趣的任何文档。

# 访问列表

网络访问列表通常是防范外部入侵和攻击的第一道防线。一般来说，路由器和交换机的数据包处理速度要比服务器快得多，因为它们利用硬件，如**三态内容可寻址存储器**（**TCAM**）。它们不需要查看应用层信息，而只需检查第3层和第4层信息，并决定是否可以转发数据包。因此，我们通常将网络设备访问列表用作保护网络资源的第一步。

作为一个经验法则，我们希望将访问列表尽可能靠近源（客户端）。因此，我们也相信内部主机，不信任我们网络边界之外的客户端。因此，访问列表通常放置在外部网络接口的入站方向上。在我们的实验场景中，这意味着我们将在直接连接到客户端主机的Ethernet2/2上放置一个入站访问列表。

如果您不确定访问列表的方向和位置，以下几点可能会有所帮助：

+   从网络设备的角度考虑访问列表

+   简化数据包，只涉及源和目的地IP，并以一个主机为例：

+   在我们的实验室中，来自我们服务器的流量将具有源IP`10.0.0.14`和目的IP`10.0.0.10`

+   来自客户端的流量将具有源IP`10.10.10.10`和目的IP`10.0.0.14`

显然，每个网络都是不同的，访问列表的构建方式取决于服务器提供的服务。但作为入站边界访问列表，您应该执行以下操作：

+   拒绝RFC 3030特殊使用地址源，如`127.0.0.0/8`

+   拒绝RFC 1918空间，如`10.0.0.0/8`

+   拒绝我们自己的空间作为源IP；在这种情况下，`10.0.0.12/30`

+   允许入站TCP端口`22`（SSH）和`80`（HTTP）到主机`10.0.0.14`

+   拒绝其他所有内容

# 使用Ansible实现访问列表

实现此访问列表的最简单方法是使用Ansible。我们在过去的两章中已经看过Ansible，但值得重申在这种情况下使用Ansible的优势：

+   **更容易管理**：对于长访问列表，我们可以利用`include`语句将其分解为更易管理的部分。然后其他团队或服务所有者可以管理这些较小的部分。

+   **幂等性**：我们可以定期安排playbook，并且只会进行必要的更改。

+   **每个任务都是明确的**：我们可以分开构造条目以及将访问列表应用到正确的接口。

+   **可重用性**：将来，如果我们添加额外的面向外部的接口，我们只需要将设备添加到访问列表的设备列表中。

+   **可扩展性**：您会注意到我们可以使用相同的playbook来构建访问列表并将其应用到正确的接口。我们可以从小处开始，根据需要在将来扩展到单独的playbook。

主机文件非常标准。为简单起见，我们直接将主机变量放在清单文件中：

```py
[nxosv-devices]
nx-osv-1 ansible_host=172.16.1.155 ansible_username=cisco ansible_password=cisco
```

我们暂时将在playbook中声明变量：

```py
---
- name: Configure Access List
  hosts: "nxosv-devices"
  gather_facts: false
  connection: local

  vars:
    cli:
      host: "{{ ansible_host }}"
      username: "{{ ansible_username }}"
      password: "{{ ansible_password }}"
      transport: cli
```

为了节省空间，我们将仅说明拒绝RFC 1918空间。实施拒绝RFC 3030和我们自己的空间将与用于RFC 1918空间的步骤相同。请注意，我们在playbook中没有拒绝`10.0.0.0/8`，因为我们当前的配置使用了`10.0.0.0`网络进行寻址。当然，我们可以首先执行单个主机许可，然后在以后的条目中拒绝`10.0.0.0/8`，但在这个例子中，我们选择忽略它：

```py
tasks:
  - nxos_acl:
      name: border_inbound
      seq: 20
      action: deny
      proto: tcp
      src: 172.16.0.0/12
      dest: any
      log: enable
      state: present
      provider: "{{ cli }}"
  - nxos_acl:
      name: border_inbound
      seq: 40
      action: permit
      proto: tcp
      src: any
      dest: 10.0.0.14/32
      dest_port_op: eq
      dest_port1: 22
      state: present
      log: enable
      provider: "{{ cli }}"
  - nxos_acl:
      name: border_inbound
      seq: 50
      action: permit
      proto: tcp
      src: any
      dest: 10.0.0.14/32
      dest_port_op: eq
      dest_port1: 80
      state: present
      log: enable
      provider: "{{ cli }}"
  - nxos_acl:
      name: border_inbound
      seq: 60
      action: permit
      proto: tcp
      src: any
      dest: any
      state: present
      log: enable
      established: enable
      provider: "{{ cli }}"
  - nxos_acl:
      name: border_inbound
      seq: 1000
      action: deny
      proto: ip
      src: any
      dest: any
      state: present
      log: enable
      provider: "{{ cli }}"
```

请注意，我们允许来自内部服务器的已建立连接返回。我们使用最终的显式`deny ip any any`语句作为高序号（`1000`），因此我们可以随后插入任何新条目。

然后我们可以将访问列表应用到正确的接口上：

```py
- name: apply ingress acl to Ethernet 2/2
  nxos_acl_interface:
    name: border_inbound
    interface: Ethernet2/2
    direction: ingress
    state: present
    provider: "{{ cli }}"
```

VIRL NX-OSv上的访问列表仅支持管理接口。您将看到此警告：警告：ACL可能不会按预期行为，因为只支持管理接口，如果您通过CLI配置此`ACL`。这个警告没问题，因为我们的目的只是演示访问列表的配置自动化。

对于单个访问列表来说，这可能看起来是很多工作。对于有经验的工程师来说，使用Ansible执行此任务将比只是登录设备并配置访问列表需要更长的时间。但是，请记住，这个playbook可以在将来多次重复使用，因此从长远来看可以节省时间。

根据我的经验，通常情况下，长访问列表中的一些条目将用于一个服务，另一些条目将用于另一个服务，依此类推。访问列表往往会随着时间的推移而有机地增长，很难跟踪每个条目的来源和目的。我们可以将它们分开，从而使长访问列表的管理变得更简单。

# MAC访问列表

在L2环境或在以太网接口上使用非IP协议的情况下，您仍然可以使用MAC地址访问列表来允许或拒绝基于MAC地址的主机。步骤与IP访问列表类似，但匹配将基于MAC地址。请记住，对于MAC地址或物理地址，前六个十六进制符号属于**组织唯一标识符**（**OUI**）。因此，我们可以使用相同的访问列表匹配模式来拒绝某个主机组。

我们正在使用`ios_config`模块在IOSv上进行测试。对于较旧的Ansible版本，更改将在每次执行playbook时推送出去。对于较新的Ansible版本，控制节点将首先检查更改，并且只在需要时进行更改。

主机文件和playbook的顶部部分与IP访问列表类似；`tasks`部分是使用不同模块和参数的地方：

```py
<skip>
  tasks:
    - name: Deny Hosts with vendor id fa16.3e00.0000
      ios_config:
        lines:
          - access-list 700 deny fa16.3e00.0000 0000.00FF.FFFF
          - access-list 700 permit 0000.0000.0000 FFFF.FFFF.FFFF
        provider: "{{ cli }}"
    - name: Apply filter on bridge group 1
      ios_config:
        lines:
          - bridge-group 1
          - bridge-group 1 input-address-list 700
        parents:
          - interface GigabitEthernet0/1
        provider: "{{ cli }}"   
```

随着越来越多的虚拟网络变得流行，L3信息有时对底层虚拟链接变得透明。在这些情况下，如果您需要限制对这些链接的访问，MAC访问列表成为一个很好的选择。

# Syslog搜索

有大量记录的网络安全漏洞发生在较长的时间内。在这些缓慢的漏洞中，我们经常看到日志中有可疑活动的迹象。这些迹象可以在服务器和网络设备的日志中找到。这些活动之所以没有被检测到，不是因为信息不足，而是因为信息太多。我们正在寻找的关键信息通常深藏在难以整理的大量信息中。

除了Syslog，**Uncomplicated Firewall**（**UFW**）是服务器日志信息的另一个很好的来源。它是iptables的前端，是一个服务器防火墙。UFW使管理防火墙规则变得非常简单，并记录了大量信息。有关UFW的更多信息，请参阅*其他工具*部分。

在这一部分，我们将尝试使用Python搜索Syslog文本，以便检测我们正在寻找的活动。当然，我们将搜索的确切术语取决于我们使用的设备。例如，思科提供了一个在Syslog中查找任何访问列表违规日志的消息列表。它可以在[http://www.cisco.com/c/en/us/about/security-center/identify-incidents-via-syslog.html](http://www.cisco.com/c/en/us/about/security-center/identify-incidents-via-syslog.html)上找到。

要更好地理解访问控制列表日志记录，请访问[http://www.cisco.com/c/en/us/about/security-center/access-control-list-logging.html](http://www.cisco.com/c/en/us/about/security-center/access-control-list-logging.html)。

对于我们的练习，我们将使用一个包含大约65,000行日志消息的Nexus交换机匿名Syslog文件，该文件已包含在适应书籍GitHub存储库中供您使用：

```py
$ wc -l sample_log_anonymized.log
65102 sample_log_anonymized.log
```

我们已经插入了一些来自思科文档（[http://www.cisco.com/c/en/us/support/docs/switches/nexus-7000-series-switches/118907-configure-nx7k-00.html](http://www.cisco.com/c/en/us/support/docs/switches/nexus-7000-series-switches/118907-configure-nx7k-00.html) ）的Syslog消息作为我们应该寻找的日志消息：

```py
2014 Jun 29 19:20:57 Nexus-7000 %VSHD-5-VSHD_SYSLOG_CONFIG_I: Configured from vty by admin on console0
2014 Jun 29 19:21:18 Nexus-7000 %ACLLOG-5-ACLLOG_FLOW_INTERVAL: Src IP: 10.1 0.10.1,
 Dst IP: 172.16.10.10, Src Port: 0, Dst Port: 0, Src Intf: Ethernet4/1, Pro tocol: "ICMP"(1), Hit-count = 2589
2014 Jun 29 19:26:18 Nexus-7000 %ACLLOG-5-ACLLOG_FLOW_INTERVAL: Src IP: 10.1 0.10.1, Dst IP: 172.16.10.10, Src Port: 0, Dst Port: 0, Src Intf: Ethernet4/1, Pro tocol: "ICMP"(1), Hit-count = 4561
```

我们将使用简单的正则表达式示例。如果您已经熟悉Python中的正则表达式，请随时跳过本节的其余部分。

# 使用RE模块进行搜索

对于我们的第一个搜索，我们将简单地使用正则表达式模块来查找我们正在寻找的术语。我们将使用一个简单的循环来进行以下操作：

```py
#!/usr/bin/env python3

import re, datetime

startTime = datetime.datetime.now()

with open('sample_log_anonymized.log', 'r') as f:
   for line in f.readlines():
       if re.search('ACLLOG-5-ACLLOG_FLOW_INTERVAL', line):
           print(line)

endTime = datetime.datetime.now()
elapsedTime = endTime - startTime
print("Time Elapsed: " + str(elapsedTime))
```

搜索日志文件大约花了6/100秒的时间：

```py
$ python3 python_re_search_1.py
2014 Jun 29 19:21:18 Nexus-7000 %ACLLOG-5-ACLLOG_FLOW_INTERVAL: Src IP: 10.1 0.10.1,

2014 Jun 29 19:26:18 Nexus-7000 %ACLLOG-5-ACLLOG_FLOW_INTERVAL: Src IP: 10.1 0.10.1,

Time Elapsed: 0:00:00.065436
```

建议编译搜索术语以进行更有效的搜索。这不会对我们产生太大影响，因为脚本已经非常快速。实际上，Python的解释性特性可能会使其变慢。但是，当我们搜索更大的文本主体时，这将产生影响，所以让我们做出改变：

```py
searchTerm = re.compile('ACLLOG-5-ACLLOG_FLOW_INTERVAL')

with open('sample_log_anonymized.log', 'r') as f:
   for line in f.readlines():
       if re.search(searchTerm, line):
           print(line)
```

时间结果实际上更慢：

```py
Time Elapsed: 0:00:00.081541
```

让我们扩展一下这个例子。假设我们有几个文件和多个要搜索的术语，我们将把原始文件复制到一个新文件中：

```py
$ cp sample_log_anonymized.log sample_log_anonymized_1.log
```

我们还将包括搜索`PAM: Authentication failure`术语。我们将添加另一个循环来搜索这两个文件：

```py
term1 = re.compile('ACLLOG-5-ACLLOG_FLOW_INTERVAL')
term2 = re.compile('PAM: Authentication failure')

fileList = ['sample_log_anonymized.log', 'sample_log_anonymized_1.log']

for log in fileList:
    with open(log, 'r') as f:
       for line in f.readlines():
           if re.search(term1, line) or re.search(term2, line):
               print(line) 
```

通过扩展我们的搜索术语和消息数量，我们现在可以看到性能上的差异：

```py
$ python3 python_re_search_2.py
2016 Jun 5 16:49:33 NEXUS-A %DAEMON-3-SYSTEM_MSG: error: PAM: Authentication failure for illegal user AAA from 172.16.20.170 - sshd[4425]

2016 Sep 14 22:52:26.210 NEXUS-A %DAEMON-3-SYSTEM_MSG: error: PAM: Authentication failure for illegal user AAA from 172.16.20.170 - sshd[2811]

<skip>

2014 Jun 29 19:21:18 Nexus-7000 %ACLLOG-5-ACLLOG_FLOW_INTERVAL: Src IP: 10.1 0.10.1,

2014 Jun 29 19:26:18 Nexus-7000 %ACLLOG-5-ACLLOG_FLOW_INTERVAL: Src IP: 10.1 0.10.1,

<skip>

Time Elapsed: 0:00:00.330697
```

当涉及性能调优时，这是一个永无止境的、不可能达到零的竞赛，性能有时取决于您使用的硬件。但重要的是定期使用Python对日志文件进行审计，这样您就可以捕捉到任何潜在违规的早期信号。

# 其他工具

还有其他网络安全工具可以使用Python进行自动化。让我们看看其中一些。

# 私有VLAN

**虚拟局域网**（**VLANs**）已经存在很长时间了。它们本质上是一个广播域，所有主机都可以连接到一个交换机，但被划分到不同的域，所以我们可以根据哪个主机可以通过广播看到其他主机来分隔主机。让我们看一个基于IP子网的映射。例如，在企业大楼中，我可能会看到每个物理楼层一个IP子网：第一层的`192.168.1.0/24`，第二层的`192.168.2.0/24`，依此类推。在这种模式下，我们为每个楼层使用1/24块。这清晰地划分了我的物理网络和逻辑网络。想要与自己的子网之外通信的主机将需要通过其第3层网关，我可以使用访问列表来强制执行安全性。

当不同部门位于同一楼层时会发生什么？也许财务和销售团队都在二楼，我不希望销售团队的主机与财务团队的主机在同一个广播域中。我可以进一步分割子网，但这可能变得乏味，并且会破坏先前设置的标准子网方案。这就是私有VLAN可以帮助的地方。

私有VLAN本质上将现有的VLAN分成子VLAN。私有VLAN中有三个类别：

+   **混杂（P）端口**：此端口允许从VLAN上的任何其他端口发送和接收第2层帧；这通常属于连接到第3层路由器的端口

+   **隔离（I）端口**：此端口只允许与P端口通信，并且它们通常连接到主机，当您不希望它与同一VLAN中的其他主机通信时

+   **社区（C）端口**：此端口允许与同一社区中的其他C端口和P端口通信

我们可以再次使用Ansible或迄今为止介绍的任何其他Python脚本来完成这项任务。到目前为止，我们应该有足够的练习和信心通过自动化来实现这个功能，所以我不会在这里重复步骤。在需要进一步隔离L2 VLAN中的端口时，了解私有VLAN功能将会很有用。

# 使用Python的UFW

我们简要提到了UFW作为Ubuntu主机上iptables的前端。以下是一个快速概述：

```py
$ sudo apt-get install ufw
$ sudo ufw status
$ sudo ufw default outgoing
$ sudo ufw allow 22/tcp
$ sudo ufw allow www
$ sudo ufw default deny incoming
```

我们可以查看UFW的状态：

```py
$ sudo ufw status verbose
Status: active
Logging: on (low)
Default: deny (incoming), allow (outgoing), disabled (routed)
New profiles: skip

To Action From
-- ------ ----
22/tcp ALLOW IN Anywhere
80/tcp ALLOW IN Anywhere
22/tcp (v6) ALLOW IN Anywhere (v6)
80/tcp (v6) ALLOW IN Anywhere (v6)
```

正如您所看到的，UFW的优势在于提供一个简单的界面来构建否则复杂的IP表规则。有几个与UFW相关的Python工具可以使事情变得更简单：

+   我们可以使用Ansible UFW模块来简化我们的操作。更多信息请访问[http://docs.ansible.com/ansible/ufw_module.html](http://docs.ansible.com/ansible/ufw_module.html)。因为Ansible是用Python编写的，我们可以进一步检查Python模块源代码中的内容。更多信息请访问[https://github.com/ansible/ansible/blob/devel/lib/ansible/modules/system/ufw.py.](https://github.com/ansible/ansible/blob/devel/lib/ansible/modules/system/ufw.py)

+   有Python包装器模块围绕UFW作为API（访问[https://gitlab.com/dhj/easyufw](https://gitlab.com/dhj/easyufw)）。如果您需要根据某些事件动态修改UFW规则，这可以使集成变得更容易。

+   UFW本身是用Python编写的。因此，如果您需要扩展当前的命令集，可以使用现有的Python知识。更多信息请访问[https://launchpad.net/ufw](https://launchpad.net/ufw)。

UFW被证明是保护您的网络服务器的好工具。

# 进一步阅读

Python是许多安全相关领域中常用的语言。我推荐的一些书籍如下：

+   **暴力Python**：T.J. O'Connor编写的黑客、取证分析师、渗透测试人员和安全工程师的食谱（ISBN-10：1597499579）

+   **黑帽Python**：Justin Seitz编写的黑客和渗透测试人员的Python编程（ISBN-10：1593275900）

我个人在A10 Networks的**分布式拒绝服务**（**DDoS**）研究工作中广泛使用Python。如果您有兴趣了解更多信息，可以免费下载指南：[https://www.a10networks.com/resources/ebooks/distributed-denial-service-ddos](https://www.a10networks.com/resources/ebooks/distributed-denial-service-ddos)。

# 总结

在本章中，我们使用Python进行了网络安全研究。我们使用Cisco VIRL工具在实验室中设置了主机和网络设备，包括NX-OSv和IOSv类型。我们对Scapy进行了介绍，它允许我们从头开始构建数据包。Scapy可以在交互模式下进行快速测试。在交互模式完成后，我们可以将步骤放入文件进行更可扩展的测试。它可以用于执行已知漏洞的各种网络渗透测试。

我们还研究了如何使用IP访问列表和MAC访问列表来保护我们的网络。它们通常是我们网络保护的第一道防线。使用Ansible，我们能够一致快速地部署访问列表到多个设备。

Syslog和其他日志文件包含有用的信息，我们应该定期查看以检测任何早期入侵的迹象。使用Python正则表达式，我们可以系统地搜索已知的日志条目，这些条目可以指引我们注意的安全事件。除了我们讨论过的工具之外，私有VLAN和UFW是我们可以用于更多安全保护的其他一些有用工具。

在[第7章](bfb06aa0-1deb-4432-80ae-f15e3644fa54.xhtml)中，*使用Python进行网络监控-第1部分*，我们将看看如何使用Python进行网络监控。监控可以让我们了解网络中正在发生的事情以及网络的状态。
