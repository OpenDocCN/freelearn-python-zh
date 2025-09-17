# 第九章：网络监控和安全

在本章中，我们将介绍以下菜谱：

+   在你的网络上嗅探数据包

+   使用 pcap dumper 将数据包保存到 pcap 格式

+   在 HTTP 数据包中添加额外的头部

+   扫描远程主机的端口

+   自定义数据包的 IP 地址

+   通过读取保存的 pcap 文件来回放流量

+   扫描数据包的广播

# 简介

本章介绍了关于网络安全监控和漏洞扫描的一些有趣的 Python 菜谱。我们首先使用 `pcap` 库在网络中嗅探数据包。然后，我们开始使用 `Scapy`，这是一个瑞士军刀式的库，可以执行许多类似任务。使用 `Scapy` 展示了一些常见的包分析任务，例如将数据包保存到 `pcap` 格式、添加额外的头部和修改数据包的 IP 地址。

本章还包括一些关于网络入侵检测的高级任务，例如，从保存的 `pcap` 文件中回放流量和广播扫描。

# 在你的网络上嗅探数据包

如果你对你本地网络上的数据包嗅探感兴趣，这个菜谱可以作为起点。记住，你可能无法嗅探除目标机器之外的数据包，因为良好的网络交换机只会转发指定给机器的流量。

## 准备工作

为了使这个菜谱工作，你需要安装 `pylibpcap` 库（版本 0.6.4 或更高版本）。它可以在 SourceForge 上找到（[`sourceforge.net/projects/pylibpcap/`](http://sourceforge.net/projects/pylibpcap/))。

你还需要安装 `construct` 库，这个库可以通过 PyPI 中的 `pip` 或 `easy_install` 安装，如下命令所示：

```py
$ easy_install construct

```

## 如何做...

我们可以提供命令行参数，例如网络接口名称和 TCP 端口号，以进行嗅探。

列表 9.1 给出了在网络上嗅探数据包的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 9
# This program is optimized for Python 2.6\. 
# It may run on any other version with/without modifications.

import argparse
import pcap
from construct.protocols.ipstack import ip_stack

def print_packet(pktlen, data, timestamp):
  """ Callback for printing the packet payload"""
  if not data:
    return

  stack = ip_stack.parse(data)
  payload = stack.next.next.next
  print payload

def main():
  # setup commandline arguments
  parser = argparse.ArgumentParser(description='Packet Sniffer')
  parser.add_argument('--iface', action="store", dest="iface", 
default='eth0')
  parser.add_argument('--port', action="store", dest="port", 
default=80, type=int)
  # parse arguments
  given_args = parser.parse_args()
  iface, port =  given_args.iface, given_args.port
  # start sniffing
  pc = pcap.pcapObject()
  pc.open_live(iface, 1600, 0, 100)
  pc.setfilter('dst port %d' %port, 0, 0)

  print 'Press CTRL+C to end capture'
  try:
    while True:
      pc.dispatch(1, print_packet)
  except KeyboardInterrupt:
    print 'Packet statistics: %d packets received, %d packets 
dropped, %d packets dropped by the interface' % pc.stats()

if __name__ == '__main__':
  main()
```

如果你运行此脚本并传递命令行参数，`--iface=eth0` 和 `--port=80`，此脚本将嗅探来自你网页浏览器的所有 HTTP 数据包。因此，在运行此脚本后，如果你在浏览器中访问 [`www.google.com`](http://www.google.com)，你可以看到如下所示的原始数据包输出：

```py
python 9_1_packet_sniffer.py --iface=eth0 --port=80 
Press CTRL+C to end capture
''
0000   47 45 54 20 2f 20 48 54 54 50 2f 31 2e 31 0d 0a   GET / HTTP/1.1..
0010   48 6f 73 74 3a 20 77 77 77 2e 67 6f 6f 67 6c 65   Host: www.google
0020   2e 63 6f 6d 0d 0a 43 6f 6e 6e 65 63 74 69 6f 6e   .com..Connection
0030   3a 20 6b 65 65 70 2d 61 6c 69 76 65 0d 0a 41 63   : keep-alive..Ac
0040   63 65 70 74 3a 20 74 65 78 74 2f 68 74 6d 6c 2c   cept: text/html,
0050   61 70 70 6c 69 63 61 74 69 6f 6e 2f 78 68 74 6d   application/xhtm
0060   6c 2b 78 6d 6c 2c 61 70 70 6c 69 63 61 74 69 6f   l+xml,applicatio
0070   6e 2f 78 6d 6c 3b 71 3d 30 2e 39 2c 2a 2f 2a 3b   n/xml;q=0.9,*/*;
0080   71 3d 30 2e 38 0d 0a 55 73 65 72 2d 41 67 65 6e   q=0.8..User-Agen
0090   74 3a 20 4d 6f 7a 69 6c 6c 61 2f 35 2e 30 20 28   t: Mozilla/5.0 (
00A0   58 31 31 3b 20 4c 69 6e 75 78 20 69 36 38 36 29   X11; Linux i686)
00B0   20 41 70 70 6c 65 57 65 62 4b 69 74 2f 35 33 37    AppleWebKit/537
00C0   2e 33 31 20 28 4b 48 54 4d 4c 2c 20 6c 69 6b 65   .31 (KHTML, like
00D0   20 47 65 63 6b 6f 29 20 43 68 72 6f 6d 65 2f 32    Gecko) Chrome/2
00E0   36 2e 30 2e 31 34 31 30 2e 34 33 20 53 61 66 61   6.0.1410.43 Safa
00F0   72 69 2f 35 33 37 2e 33 31 0d 0a 58 2d 43 68 72   ri/537.31..X-Chr
0100   6f 6d 65 2d 56 61 72 69 61 74 69 6f 6e 73 3a 20   ome-Variations: 
0110   43 50 71 31 79 51 45 49 6b 62 62 4a 41 51 69 59   CPq1yQEIkbbJAQiY
0120   74 73 6b 42 43 4b 4f 32 79 51 45 49 70 37 62 4a   tskBCKO2yQEIp7bJ
0130   41 51 69 70 74 73 6b 42 43 4c 65 32 79 51 45 49   AQiptskBCLe2yQEI
0140   2b 6f 50 4b 41 51 3d 3d 0d 0a 44 4e 54 3a 20 31   +oPKAQ==..DNT: 1
0150   0d 0a 41 63 63 65 70 74 2d 45 6e 63 6f 64 69 6e   ..Accept-Encodin
0160   67 3a 20 67 7a 69 70 2c 64 65 66 6c 61 74 65 2c   g: gzip,deflate,
0170   73 64 63 68 0d 0a 41 63 63 65 70 74 2d 4c 61 6e   sdch..Accept-Lan
0180   67 75 61 67 65 3a 20 65 6e 2d 47 42 2c 65 6e 2d   guage: en-GB,en-
0190   55 53 3b 71 3d 30 2e 38 2c 65 6e 3b 71 3d 30 2e   US;q=0.8,en;q=0.
01A0   36 0d 0a 41 63 63 65 70 74 2d 43 68 61 72 73 65   6..Accept-Charse
01B0   74 3a 20 49 53 4f 2d 38 38 35 39 2d 31 2c 75 74   t: ISO-8859-1,ut
01C0   66 2d 38 3b 71 3d 30 2e 37 2c 2a 3b 71 3d 30 2e   f-8;q=0.7,*;q=0.
01D0   33 0d 0a 43 6f 6f 6b 69 65 3a 20 50 52 45 46 3d   3..Cookie: PREF=

….

^CPacket statistics: 17 packets received, 0 packets dropped, 0 
packets dropped by the interface

```

## 它是如何工作的...

这个菜谱依赖于 `pcap` 库中的 `pcapObject()` 类来创建嗅探器的实例。在 `main()` 方法中，创建了这个类的实例，并使用 `setfilter()` 方法设置了一个过滤器，以便只捕获 HTTP 数据包。最后，`dispatch()` 方法开始嗅探并将嗅探到的数据包发送到 `print_packet()` 函数进行后处理。

在 `print_packet()` 函数中，如果数据包包含数据，则使用 `construct` 库的 `ip_stack.parse()` 方法提取有效载荷。这个库对于低级数据处理很有用。

# 使用 pcap dumper 将数据包保存到 pcap 格式

**pcap**格式，缩写自**数据包捕获**，是保存网络数据的常见文件格式。有关 pcap 格式的更多详细信息，请参阅[`wiki.wireshark.org/Development/LibpcapFileFormat`](http://wiki.wireshark.org/Development/LibpcapFileFormat)。

如果你想要将捕获到的网络数据包保存到文件，并在以后重新使用它们进行进一步处理，这个菜谱可以为你提供一个工作示例。

## 如何做到这一点...

在这个菜谱中，我们使用`Scapy`库来嗅探数据包并将其写入文件。所有`Scapy`的实用函数和定义都可以通过通配符导入来导入，如下面的命令所示：

```py
from scapy.all import *

```

这只是为了演示目的，不推荐用于生产代码。

`Scapy`的`sniff()`函数接受一个回调函数的名称。让我们编写一个回调函数，该函数将数据包写入文件。

列表 9.2 给出了使用 pcap dumper 将数据包保存为 pcap 格式的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 9
# This program is optimized for Python 2.7\. 
# It may run on any other version with/without modifications.

import os
from scapy.all import *

pkts = []
iter = 0
pcapnum = 0

def write_cap(x):
  global pkts
  global iter
  global pcapnum
  pkts.append(x)
  iter += 1
  if iter == 3:
    pcapnum += 1
    pname = "pcap%d.pcap" % pcapnum
    wrpcap(pname, pkts)
    pkts = []
    iter = 0

if __name__ == '__main__':
  print "Started packet capturing and dumping... Press CTRL+C to exit"
  sniff(prn=write_cap)

  print "Testing the dump file..."
  dump_file = "./pcap1.pcap"
  if os.path.exists(dump_file):
    print "dump fie %s found." %dump_file
    pkts = sniff(offline=dump_file)
    count = 0
    while (count <=2):
      print "----Dumping pkt:%s----" %count
      print hexdump(pkts[count])
      count += 1    
  else:
    print "dump fie %s not found." %dump_file
```

如果你运行此脚本，你将看到类似以下输出的输出：

```py
# python 9_2_save_packets_in_pcap_format.py 
^CStarted packet capturing and dumping... Press CTRL+C to exit
Testing the dump file...
dump fie ./pcap1.pcap found.
----Dumping pkt:0----
0000   08 00 27 95 0D 1A 52 54  00 12 35 02 08 00 45 00   ..'...RT..5...E.
0010   00 DB E2 6D 00 00 40 06  7C 9E 6C A0 A2 62 0A 00   ...m..@.|.l..b..
0020   02 0F 00 50 99 55 97 98  2C 84 CE 45 9B 6C 50 18   ...P.U..,..E.lP.
0030   FF FF 53 E0 00 00 48 54  54 50 2F 31 2E 31 20 32   ..S...HTTP/1.1 2
0040   30 30 20 4F 4B 0D 0A 58  2D 44 42 2D 54 69 6D 65   00 OK..X-DB-Time
0050   6F 75 74 3A 20 31 32 30  0D 0A 50 72 61 67 6D 61   out: 120..Pragma
0060   3A 20 6E 6F 2D 63 61 63  68 65 0D 0A 43 61 63 68   : no-cache..Cach
0070   65 2D 43 6F 6E 74 72 6F  6C 3A 20 6E 6F 2D 63 61   e-Control: no-ca
0080   63 68 65 0D 0A 43 6F 6E  74 65 6E 74 2D 54 79 70   che..Content-Typ
0090   65 3A 20 74 65 78 74 2F  70 6C 61 69 6E 0D 0A 44   e: text/plain..D
00a0   61 74 65 3A 20 53 75 6E  2C 20 31 35 20 53 65 70   ate: Sun, 15 Sep
00b0   20 32 30 31 33 20 31 35  3A 32 32 3A 33 36 20 47    2013 15:22:36 G
00c0   4D 54 0D 0A 43 6F 6E 74  65 6E 74 2D 4C 65 6E 67   MT..Content-Leng
00d0   74 68 3A 20 31 35 0D 0A  0D 0A 7B 22 72 65 74 22   th: 15....{"ret"
00e0   3A 20 22 70 75 6E 74 22  7D                        : "punt"}
None
----Dumping pkt:1----
0000   52 54 00 12 35 02 08 00  27 95 0D 1A 08 00 45 00   RT..5...'.....E.
0010   01 D2 1F 25 40 00 40 06  FE EF 0A 00 02 0F 6C A0   ...%@.@.......l.
0020   A2 62 99 55 00 50 CE 45  9B 6C 97 98 2D 37 50 18   .b.U.P.E.l..-7P.
0030   F9 28 1C D6 00 00 47 45  54 20 2F 73 75 62 73 63   .(....GET /subsc
0040   72 69 62 65 3F 68 6F 73  74 5F 69 6E 74 3D 35 31   ribe?host_int=51
0050   30 35 36 34 37 34 36 26  6E 73 5F 6D 61 70 3D 31   0564746&ns_map=1
0060   36 30 36 39 36 39 39 34  5F 33 30 30 38 30 38 34   60696994_3008084
0070   30 37 37 31 34 2C 31 30  31 39 34 36 31 31 5F 31   07714,10194611_1
0080   31 30 35 33 30 39 38 34  33 38 32 30 32 31 31 2C   105309843820211,
0090   31 34 36 34 32 38 30 35  32 5F 33 32 39 34 33 38   146428052_329438
00a0   36 33 34 34 30 38 34 2C  31 31 36 30 31 35 33 31   6344084,11601531
00b0   5F 32 37 39 31 38 34 34  37 35 37 37 31 2C 31 30   _279184475771,10
00c0   31 39 34 38 32 38 5F 33  30 30 37 34 39 36 35 39   194828_300749659
00d0   30 30 2C 33 33 30 39 39  31 39 38 32 5F 38 31 39   00,330991982_819
00e0   33 35 33 37 30 36 30 36  2C 31 36 33 32 37 38 35   35370606,1632785
00f0   35 5F 31 32 39 30 31 32  32 39 37 34 33 26 75 73   5_12901229743&us
0100   65 72 5F 69 64 3D 36 35  32 30 33 37 32 26 6E 69   er_id=6520372&ni
0110   64 3D 32 26 74 73 3D 31  33 37 39 32 35 38 35 36   d=2&ts=137925856
0120   31 20 48 54 54 50 2F 31  2E 31 0D 0A 48 6F 73 74   1 HTTP/1.1..Host
0130   3A 20 6E 6F 74 69 66 79  33 2E 64 72 6F 70 62 6F   : notify3.dropbo
0140   78 2E 63 6F 6D 0D 0A 41  63 63 65 70 74 2D 45 6E   x.com..Accept-En
0150   63 6F 64 69 6E 67 3A 20  69 64 65 6E 74 69 74 79   coding: identity
0160   0D 0A 43 6F 6E 6E 65 63  74 69 6F 6E 3A 20 6B 65   ..Connection: ke
0170   65 70 2D 61 6C 69 76 65  0D 0A 58 2D 44 72 6F 70   ep-alive..X-Drop
0180   62 6F 78 2D 4C 6F 63 61  6C 65 3A 20 65 6E 5F 55   box-Locale: en_U
0190   53 0D 0A 55 73 65 72 2D  41 67 65 6E 74 3A 20 44   S..User-Agent: D
01a0   72 6F 70 62 6F 78 44 65  73 6B 74 6F 70 43 6C 69   ropboxDesktopCli
01b0   65 6E 74 2F 32 2E 30 2E  32 32 20 28 4C 69 6E 75   ent/2.0.22 (Linu
01c0   78 3B 20 32 2E 36 2E 33  32 2D 35 2D 36 38 36 3B   x; 2.6.32-5-686;
01d0   20 69 33 32 3B 20 65 6E  5F 55 53 29 0D 0A 0D 0A    i32; en_US)....
None
----Dumping pkt:2----
0000   08 00 27 95 0D 1A 52 54  00 12 35 02 08 00 45 00   ..'...RT..5...E.
0010   00 28 E2 6E 00 00 40 06  7D 50 6C A0 A2 62 0A 00   .(.n..@.}Pl..b..
0020   02 0F 00 50 99 55 97 98  2D 37 CE 45 9D 16 50 10   ...P.U..-7.E..P.
0030   FF FF CA F1 00 00 00 00  00 00 00 00               ............
None

```

## 它是如何工作的...

这个菜谱使用`Scapy`库的`sniff()`和`wrpacp()`实用函数来捕获所有网络数据包并将它们写入文件。通过`sniff()`捕获数据包后，将调用该数据包的`write_cap()`函数。一些全局变量用于逐个处理数据包。例如，数据包存储在`pkts[]`列表中，并使用数据包和变量计数。当计数值为 3 时，将`pkts`列表写入名为`pcap1.pcap`的文件，将计数变量重置，以便我们可以继续捕获另外三个数据包并将它们写入`pcap2.pcap`，依此类推。

在`test_dump_file()`函数中，假设在当前工作目录中存在第一个转储文件，`pcap1.dump`。现在，使用带有离线参数的`sniff()`，从文件中捕获数据包而不是从网络中捕获。在这里，使用`hexdump()`函数逐个解码数据包。然后，将数据包的内容打印到屏幕上。

# 在 HTTP 数据包中添加额外头

有时，你可能希望通过提供包含自定义信息的自定义 HTTP 头来操纵应用程序。例如，添加一个授权头对于在数据包捕获代码中实现 HTTP 基本认证非常有用。

## 如何做到这一点...

让我们使用`Scapy`的`sniff()`函数嗅探数据包，并定义一个回调函数`modify_packet_header()`，该函数为某些数据包添加额外的头。

列表 9.3 给出了在 HTTP 数据包中添加额外头的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 9
# This program is optimized for Python 2.7\. 
# It may run on any other version with/without modifications.

from scapy.all import *

def modify_packet_header(pkt):
  """ Parse the header and add an extra header"""
  if pkt.haslayer(TCP) and pkt.getlayer(TCP).dport == 80 and 
pkt.haslayer(Raw):
    hdr = pkt[TCP].payload.__dict__        
    extra_item = {'Extra Header' : ' extra value'}
    hdr.update(extra_item)     
    send_hdr = '\r\n'.join(hdr)
    pkt[TCP].payload = send_hdr

    pkt.show()

    del pkt[IP].chksum
    send(pkt)

if __name__ == '__main__':
  # start sniffing
  sniff(filter="tcp and ( port 80 )", prn=modify_packet_header)
```

如果你运行此脚本，它将显示捕获到的数据包；打印其修改后的版本并将其发送到网络，如下面的输出所示。这可以通过其他数据包捕获工具如`tcpdump`或`wireshark`进行验证：

```py
$ python 9_3_add_extra_http_header_in_sniffed_packet.py 

###[ Ethernet ]###
 dst       = 52:54:00:12:35:02
 src       = 08:00:27:95:0d:1a
 type      = 0x800
###[ IP ]###
 version   = 4L
 ihl       = 5L
 tos       = 0x0
 len       = 525
 id        = 13419
 flags     = DF
 frag      = 0L
 ttl       = 64
 proto     = tcp
 chksum    = 0x171
 src       = 10.0.2.15
 dst       = 82.94.164.162
 \options   \
###[ TCP ]###
 sport     = 49273
 dport     = www
 seq       = 107715690
 ack       = 216121024
 dataofs   = 5L
 reserved  = 0L
 flags     = PA
 window    = 6432
 chksum    = 0x50f
 urgptr    = 0
 options   = []
###[ Raw ]###
 load      = 'Extra Header\r\nsent_time\r\nfields\r\naliastypes\r\npost_transforms\r\nunderlayer\r\nfieldtype\r\ntime\r\ninitialized\r\noverloaded_fields\r\npacketfields\r\npayload\r\ndefault_fields'
.
Sent 1 packets.

```

## 它是如何工作的...

首先，我们使用 `Scapy` 的 `sniff()` 函数设置数据包嗅探，指定 `modify_packet_header()` 作为每个数据包的回调函数。所有目标端口为 `80`（HTTP）且具有 TCP 和原始层的 TCP 数据包都被认为是修改对象。因此，当前数据包头部是从数据包的有效负载数据中提取出来的。

然后将额外的头部添加到现有的头部字典中。然后使用 `show()` 方法在屏幕上打印数据包，为了避免正确性检查失败，从数据包中移除数据包校验和。最后，数据包通过网络发送。

# 扫描远程主机的端口

如果你尝试通过特定端口连接到远程主机，有时你会收到“连接被拒绝”的消息。这可能是由于远程主机上的服务器可能已经关闭。在这种情况下，你可以尝试查看端口是否开放或处于监听状态。你可以扫描多个端口以识别机器上的可用服务。

## 如何做...

使用 Python 的标准套接字库，我们可以完成这个端口扫描任务。我们可以接受三个命令行参数：目标主机和起始端口和结束端口。

列表 9.4 提供了扫描远程主机端口的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 9
# This program is optimized for Python 2.7\. 
# It may run on any other version with/without modifications.

import argparse
import socket
import sys

def scan_ports(host, start_port, end_port):
  """ Scan remote hosts """
  #Create socket
  try:
    sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
  except socket.error,err_msg:
    print 'Socket creation failed. Error code: '+ str(err_msg[0]) 
+ ' Error mesage: ' + err_msg[1]
    sys.exit()

  #Get IP of remote host
  try:
    remote_ip = socket.gethostbyname(host)
  except socket.error,error_msg:
    print error_msg
  sys.exit()

  #Scan ports
  end_port += 1
  for port in range(start_port,end_port):
    try:
      sock.connect((remote_ip,port))
      print 'Port ' + str(port) + ' is open'
      sock.close()
      sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    except socket.error:
      pass # skip various socket errors

if __name__ == '__main__':
  # setup commandline arguments
  parser = argparse.ArgumentParser(description='Remote Port 
Scanner')
  parser.add_argument('--host', action="store", dest="host", 
default='localhost')
  parser.add_argument('--start-port', action="store", 
dest="start_port", default=1, type=int)
  parser.add_argument('--end-port', action="store", 
dest="end_port", default=100, type=int)
  # parse arguments
  given_args = parser.parse_args()
  host, start_port, end_port =  given_args.host, 
given_args.start_port, given_args.end_port
  scan_ports(host, start_port, end_port)
```

如果你运行此方法来扫描本地机器的端口 `1` 到 `100` 以检测开放端口，你将得到类似以下的结果：

```py
# python 9_4_scan_port_of_a_remote_host.py --host=localhost --start-port=1 --end-port=100
Port 21 is open
Port 22 is open
Port 23 is open
Port 25 is open
Port 80 is open

```

## 它是如何工作的...

这个方法演示了如何使用 Python 的标准套接字库扫描机器的开放端口。`scan_port()` 函数接受三个参数：主机名、起始端口和结束端口。然后，它分三步扫描整个端口范围。

使用 `socket()` 函数创建一个 TCP 套接字。

如果套接字创建成功，则使用 `gethostbyname()` 函数解析远程主机的 IP 地址。

如果找到了目标主机的 IP 地址，尝试使用 `connect()` 函数连接到该 IP。如果成功，则意味着端口是开放的。现在，使用 `close()` 函数关闭端口，并重复第一步以检查下一个端口。

# 自定义数据包的 IP 地址

如果你需要创建一个网络数据包并自定义源和目标 IP 或端口，这个方法可以作为起点。

## 如何做...

我们可以获取所有有用的命令行参数，例如网络接口名称、协议名称、源 IP、源端口、目标 IP、目标端口以及可选的 TCP 标志。

我们可以使用 `Scapy` 库创建自定义 TCP 或 UDP 数据包并将其发送到网络上。

列表 9.5 提供了自定义数据包 IP 地址的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 9
# This program is optimized for Python 2.7\. 
# It may run on any other version with/without modifications.

import argparse
import sys
import re
from random import randint

from scapy.all import IP,TCP,UDP,conf,send

def send_packet(protocol=None, src_ip=None, src_port=None, flags=None, dst_ip=None, dst_port=None, iface=None):
  """Modify and send an IP packet."""
  if protocol == 'tcp':
    packet = IP(src=src_ip, dst=dst_ip)/TCP(flags=flags, 
sport=src_port, dport=dst_port)
  elif protocol == 'udp':
  if flags: raise Exception(" Flags are not supported for udp")
    packet = IP(src=src_ip, dst=dst_ip)/UDP(sport=src_port, 
dport=dst_port)
  else:
    raise Exception("Unknown protocol %s" % protocol)

  send(packet, iface=iface)

if __name__ == '__main__':
  # setup commandline arguments
  parser = argparse.ArgumentParser(description='Packet Modifier')
  parser.add_argument('--iface', action="store", dest="iface", 
default='eth0')
  parser.add_argument('--protocol', action="store", 
dest="protocol", default='tcp')
  parser.add_argument('--src-ip', action="store", dest="src_ip", 
default='1.1.1.1')
  parser.add_argument('--src-port', action="store", 
dest="src_port", default=randint(0, 65535))
  parser.add_argument('--dst-ip', action="store", dest="dst_ip", 
default='192.168.1.51')
  parser.add_argument('--dst-port', action="store", 
dest="dst_port", default=randint(0, 65535))
  parser.add_argument('--flags', action="store", dest="flags", 
default=None)
  # parse arguments
  given_args = parser.parse_args()
  iface, protocol, src_ip,  src_port, dst_ip, dst_port, flags =  
given_args.iface, given_args.protocol, given_args.src_ip,\
  given_args.src_port, given_args.dst_ip, given_args.dst_port, 
given_args.flags
  send_packet(protocol, src_ip, src_port, flags, dst_ip, 
dst_port, iface)
```

为了运行此脚本，请输入以下命令：

```py
tcpdump src 192.168.1.66
tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
listening on eth0, link-type EN10MB (Ethernet), capture size 65535 bytes
^C18:37:34.309992 IP 192.168.1.66.60698 > 192.168.1.51.666: Flags [S], seq 0, win 8192, length 0

1 packets captured
1 packets received by filter
0 packets dropped by kernel

$ sudo python 9_5_modify_ip_in_a_packet.py 
WARNING: No route found for IPv6 destination :: (no default route?)
.
Sent 1 packets.

```

## 它是如何工作的...

此脚本定义了一个`send_packet()`函数，用于使用`Scapy`构建 IP 数据包。它将源地址和目标地址以及端口号提供给它。根据协议，例如 TCP 或 UDP，它构建正确的数据包类型。如果是 TCP，则使用标志参数；如果不是，则引发异常。

为了构建 TCP 数据包，`Sacpy`提供了`IP()`/`TCP()`函数。同样，为了创建 UDP 数据包，使用`IP()`/`UDP()`函数。

最后，使用`send()`函数发送修改后的数据包。

# 通过读取保存的 pcap 文件重放流量

在玩网络数据包时，您可能需要通过从之前保存的`pcap`文件中读取来重放流量。在这种情况下，您希望在发送之前读取`pcap`文件并修改源或目标 IP 地址。

## 如何操作...

让我们使用`Scapy`读取之前保存的`pcap`文件。如果您没有`pcap`文件，可以使用本章的*使用 pcap dumper 保存数据包到 pcap 格式*方法来创建一个。

然后，从命令行解析参数，并将解析后的原始数据包传递给`send_packet()`函数。

列表 9.6 给出了通过从保存的`pcap`文件中读取来重放流量的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 9
# This program is optimized for Python 2.7\. 
# It may run on any other version with/without modifications.

import argparse
from scapy.all import *

def send_packet(recvd_pkt, src_ip, dst_ip, count):
  """ Send modified packets"""
  pkt_cnt = 0
  p_out = []

  for p in recvd_pkt:
    pkt_cnt += 1
    new_pkt = p.payload
    new_pkt[IP].dst = dst_ip
    new_pkt[IP].src = src_ip
    del new_pkt[IP].chksum
    p_out.append(new_pkt)
    if pkt_cnt % count == 0:
      send(PacketList(p_out))
      p_out = []

  # Send rest of packet
  send(PacketList(p_out))
  print "Total packets sent: %d" %pkt_cnt

if __name__ == '__main__':
  # setup commandline arguments
  parser = argparse.ArgumentParser(description='Packet Sniffer')
  parser.add_argument('--infile', action="store", dest="infile", 
default='pcap1.pcap')
  parser.add_argument('--src-ip', action="store", dest="src_ip", 
default='1.1.1.1')
  parser.add_argument('--dst-ip', action="store", dest="dst_ip", 
default='2.2.2.2')
  parser.add_argument('--count', action="store", dest="count", 
default=100, type=int)
  # parse arguments
  given_args = ga = parser.parse_args()
  global src_ip, dst_ip
  infile, src_ip, dst_ip, count =  ga.infile, ga.src_ip, 
ga.dst_ip, ga.count
  try:
    pkt_reader = PcapReader(infile)
    send_packet(pkt_reader, src_ip, dst_ip, count)
  except IOError:
    print "Failed reading file %s contents" % infile
    sys.exit(1)
```

如果您运行此脚本，它将默认读取保存的`pcap`文件`pcap1.pcap`，并在修改源和目标 IP 地址为`1.1.1.1`和`2.2.2.2`后发送数据包，如下所示。如果您使用`tcpdump`实用程序，您可以看到这些数据包传输。

```py
# python 9_6_replay_traffic.py 
...
Sent 3 packets.
Total packets sent 3
----
# tcpdump src 1.1.1.1
tcpdump: verbose output suppressed, use -v or -vv for full protocol 
decode
listening on eth0, link-type EN10MB (Ethernet), capture size 65535 
bytes
^C18:44:13.186302 IP 1.1.1.1.www > ARennes-651-1-107-2.w2-
2.abo.wanadoo.fr.39253: Flags [P.], seq 2543332484:2543332663, ack 
3460668268, win 65535, length 179
1 packets captured
3 packets received by filter
0 packets dropped by kernel

```

## 它是如何工作的...

此方法使用`Scapy`的`PcapReader()`函数从磁盘读取保存的`pcap`文件`pcap1.pcap`，该函数返回一个数据包迭代器。如果提供了命令行参数，则解析它们。否则，使用前述输出中所示默认值。

将命令行参数和数据包列表传递给`send_packet()`函数。此函数将新数据包放入`p_out`列表中，并跟踪处理过的数据包。在每个数据包中，有效载荷被修改，从而改变了源和目标 IP 地址。此外，删除了`checksum`数据包，因为它基于原始 IP 地址。

处理完一个数据包后，它立即通过网络发送。之后，剩余的数据包一次性发送。

# 扫描数据包广播

如果您遇到检测网络广播的问题，这个方法就是为您准备的。我们可以学习如何从广播数据包中找到信息。

## 如何操作...

我们可以使用`Scapy`嗅探到达网络接口的数据包。在捕获每个数据包后，可以通过回调函数处理它们以获取有用的信息。

列表 9.7 给出了扫描数据包广播的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 9
# This program is optimized for Python 2.7\. 
# It may run on any other version with/without modifications.

from scapy.all import *
import os
captured_data = dict()

END_PORT = 1000

def monitor_packet(pkt):
  if IP in pkt:
    if not captured_data.has_key(pkt[IP].src):
      captured_data[pkt[IP].src] = []

    if TCP in pkt:
      if pkt[TCP].sport <=  END_PORT:
        if not str(pkt[TCP].sport) in captured_data[pkt[IP].src]:
           captured_data[pkt[IP].src].append(str(pkt[TCP].sport))

  os.system('clear')
  ip_list = sorted(captured_data.keys())
  for key in ip_list:
    ports=', '.join(captured_data[key])
    if len (captured_data[key]) == 0:
      print '%s' % key
    else:
      print '%s (%s)' % (key, ports)

if __name__ == '__main__':
  sniff(prn=monitor_packet, store=0)
```

如果你运行这个脚本，你可以列出广播流量的源 IP 和端口。以下是一个示例输出，其中 IP 的第一个八位字节已被替换：

```py
# python 9_7_broadcast_scanning.py
10.0.2.15
XXX.194.41.129 (80)
XXX.194.41.134 (80)
XXX.194.41.136 (443)
XXX.194.41.140 (80)
XXX.194.67.147 (80)
XXX.194.67.94 (443)
XXX.194.67.95 (80, 443)

```

## 它是如何工作的...

这个菜谱使用`Scapy`的`sniff()`函数在网络中嗅探数据包。它有一个`monitor_packet()`回调函数，用于处理数据包的后处理。根据协议，例如 IP 或 TCP，它将数据包排序到一个名为`captured_data`的字典中。

如果字典中尚未存在单个 IP，它将创建一个新的条目；否则，它将更新该特定 IP 的端口号。最后，它按行打印 IP 地址和端口。
