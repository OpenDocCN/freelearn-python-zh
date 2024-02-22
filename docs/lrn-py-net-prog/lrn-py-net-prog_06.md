# 第六章：IP 和 DNS

连接到网络的每台计算机都需要一个 IP 地址。在第一章中，介绍了 TCP/IP 网络编程。IP 地址使用数字标识符标记机器的网络接口，也标识了机器的位置，尽管可靠性有限。**域名系统**（**DNS**）是一种核心网络服务，将名称映射到 IP 地址，反之亦然。在本章中，我们将主要关注使用 Python 操作 IP 和 DNS 协议。除此之外，我们还将简要讨论**网络时间协议**（**NTP**），它有助于将时间与集中式时间服务器同步。以下主题将在此处讨论：

+   检索本地计算机的网络配置

+   操作 IP 地址

+   GeoIP 查找

+   使用 DNS

+   使用 NTP

# 检索本地计算机的网络配置

在做任何其他事情之前，让我们用 Python 语言问一下，*我的名字是什么？*。在网络术语中，这相当于找出机器的名称或主机的名称。在 shell 命令行上，可以使用`hostname`命令来发现这一点。在 Python 中，您可以使用 socket 模块来实现这一点。

```py
**>>> import socket**
**>>> socket.gethostname()**
**'debian6box.localdomain.loc'**

```

现在，我们想要查看本地计算机的 IP。这可以通过在 Linux 中使用`ifconfig`命令和在 Windows OS 中使用`ipconfig`命令来实现。但是，我们想要使用以下内置函数在 Python 中执行此操作：

```py
**>>> socket.gethostbyname('debian6box.localdomain.loc')**
**'10.0.2.15'**

```

如您所见，这是第一个网络接口的 IP。如果您的 DNS 或主机文件未正确配置，它还可以显示我们的环回接口（127.0.0.1）的 IP。在 Linux/UNIX 中，可以将以下行添加到您的`/etc/hosts`文件中以获取正确的 IP 地址：

```py
10.0.2.15       debian6box.localdomain.loc      debian6box
```

这个过程被称为基于主机文件的名称解析。您可以向 DNS 服务器发送查询，询问特定主机的 IP 地址。如果名称已经正确注册，那么您将从服务器收到响应。但是，在向远程服务器发出查询之前，让我们先了解一些关于网络接口和网络的更多信息。

在每个局域网中，主机被配置为充当网关，与*外部*世界通信。为了找到网络地址和子网掩码，我们可以使用 Python 第三方库 netifaces（版本> 0.10.0）。这将提取所有相关信息。例如，您可以调用`netifaces.gateways()`来查找配置为外部世界的网关。同样，您可以通过调用`netifaces.interfaces()`来枚举网络接口。如果您想要知道特定接口*eth0*的所有 IP 地址，那么可以调用`netifaces.ifaddresses('eth0')`。以下代码清单显示了如何列出本地计算机的所有网关和 IP 地址：

```py
#!/usr/bin/env python
import socket
import netifaces

if __name__ == '__main__':    
    # Find host info
    host_name = socket.gethostname()
    ip_address = socket.gethostbyname(host_name)
    print("Host name: {0}".format(host_name))

    # Get interfaces list
    ifaces = netifaces.interfaces()
    for iface in ifaces:
        ipaddrs = netifaces.ifaddresses(iface)
        if netifaces.AF_INET in ipaddrs:
            ipaddr_desc = ipaddrs[netifaces.AF_INET]
            ipaddr_desc = ipaddr_desc[0]
            print("Network interface: {0}".format(iface))
            print("\tIP address: {0}".format(ipaddr_desc['addr']))
            print("\tNetmask: {0}".format(ipaddr_desc['netmask']))
    # Find the gateway
    gateways = netifaces.gateways()
    print("Default gateway: {0}".format(gateways['default'][netifaces.AF_INET][0]))
```

如果您运行此代码，则会打印本地网络配置的摘要，类似于以下内容：

```py
**$ python 6_1_local_network_config.py**
**Host name: debian6box**
**Network interface: lo**
 **IP address: 127.0.0.1**
 **Netmask: 255.0.0.0**
**Network interface: eth0**
 **IP address: 10.0.2.15**
 **Netmask: 255.255.255.0**
**Default gateway: 10.0.2.2**

```

# 操作 IP 地址

通常，您需要操作 IP 地址并对其执行某种操作。Python3 具有内置的`ipaddress`模块，可帮助您执行此任务。它具有方便的函数来定义 IP 地址和 IP 网络，并查找许多有用的信息。例如，如果您想知道给定子网中存在多少 IP 地址，例如`10.0.1.0/255.255.255.0`或`10.0.2.0/24`，则可以使用此处显示的代码片段找到它们。此模块将提供几个类和工厂函数；例如，IP 地址和 IP 网络具有单独的类。每个类都有 IP 版本 4（IPv4）和 IP 版本 6（IPv6）的变体。以下部分演示了一些功能：

## IP 网络对象

让我们导入`ipaddress`模块并定义一个`net4`网络。

```py
**>>> import ipaddress as ip**
**>>> net4 = ip.ip_network('10.0.1.0/24')**

```

现在，我们可以找到一些有用的信息，比如`net4`的`netmask`、网络/广播地址等：

```py
**>>> net4.netmask**
**IP4Address(255.255.255.0)**

```

`net4`的`netmask`属性将显示为`IP4Address`对象。如果您正在寻找其字符串表示形式，则可以调用`str()`方法，如下所示：

```py
**>>> str(net4.netmask)**
**'255.255.255.0'**

```

同样，您可以通过执行以下操作找到`net4`的网络和广播地址：

```py
**>>> str(net4.network_address)**
**10.0.1.0**
**>>> str(net4.broadcast_address)**
**10.0.1.255**

```

`net4`总共有多少个地址？这可以通过使用以下命令找到：

```py
**>>> net4.num_addresses**
**256**

```

因此，如果我们减去网络和广播地址，那么总共可用的 IP 地址将是 254。我们可以在`net4`对象上调用`hosts()`方法。它将生成一个 Python 生成器，它将提供所有主机作为`IPv4Adress`对象。

```py
**>>> all_hosts = list(net4.hosts())**
**>>> len(all_hosts)**
**254**

```

您可以通过遵循标准的 Python 列表访问表示法来访问单个 IP 地址。例如，第一个 IP 地址将是以下内容：

```py
**>>> all_hosts[0]**
**IPv4Address('10.0.1.1')**

```

您可以通过使用列表表示法来访问最后一个 IP 地址，如下所示：

```py
**>>> all_hosts[-1]**
**IPv4Address('10.0.1.1')**

```

我们还可以从`IPv4Network`对象中找到子网信息，如下所示：

```py
**>>> subnets = list( net4.subnets())**
**>>> subnets**
**[ IPv4Network('10.0.1.0/25'), IPv4Network('10.0.1.128/25')  ]**

```

任何`IPv4Network`对象都可以告诉关于其父超网的信息，这与子网相反。

```py
**>>> net4.supernet()**
**IPv4Network('10.0.1.0/23')**

```

## 网络接口对象

在`ipaddress`模块中，一个方便的类用于详细表示接口的 IP 配置。IPv4 Interface 类接受任意地址并表现得像一个网络地址对象。让我们定义并讨论我们的网络接口`eth0`，如下截图所示：

![网络接口对象](img/6008OS_06_01.jpg)

正如您在前面的截图中所看到的，已经定义了一个带有`IPv4Address`类的网络接口 eth0。它具有一些有趣的属性，例如 IP、网络地址等。与网络对象一样，您可以检查地址是否为私有、保留或多播。这些地址范围已在各种 RFC 文档中定义。`ipaddress`模块的帮助页面将向您显示这些 RFC 文档的链接。您也可以在其他地方搜索这些信息。

## IP 地址对象

IP 地址类有许多其他有趣的属性。您可以对这些对象执行一些算术和逻辑操作。例如，如果一个 IP 地址大于另一个 IP 地址，那么您可以向 IP 地址对象添加数字，这将给您一个相应的 IP 地址。让我们在以下截图中看到这个演示：

![IP 地址对象](img/6008OS_06_02.jpg)

`ipaddress`模块的演示

在这里，已经定义了带有私有 IP 地址`192.168.1.1`的`eth0`接口，以及已经定义了另一个私有 IP 地址`192.168.2.1`的`eth1`。同样，回环接口`lo`定义为 IP 地址`127.0.0.1`。正如您所看到的，您可以向 IP 地址添加数字，它将给您相同序列的下一个 IP 地址。

您可以检查 IP 是否属于特定网络。在这里，网络 net 是由网络地址`192.168.1.0/24`定义的，并且已经测试了`eth0`和`eth1`的成员资格。还在这里测试了一些其他有趣的属性，比如`is_loopback`，`is_private`等。

## 为您的本地区域网络规划 IP 地址

如果您想知道如何选择合适的 IP 子网，那么您可以尝试使用`ipaddress`模块。以下代码片段将展示如何根据小型私有网络所需的主机 IP 地址数量选择特定子网的示例：

```py
#!/usr/bin/env python
import ipaddress as ip

CLASS_C_ADDR = '192.168.0.0'

if __name__ == '__main__':
    not_configed = True
    while not_configed:
        prefix = input("Enter the prefixlen (24-30): ")
        prefix = int(prefix)
        if prefix not in range(23, 31):
            raise Exception("Prefixlen must be between 24 and 30")
        net_addr = CLASS_C_ADDR + '/' + str(prefix)
        print("Using network address:%s " %net_addr)
        try:
            network = ip.ip_network(net_addr)
        except:
            raise Exception("Failed to create network object")
        print("This prefix will give %s IP addresses" %(network.num_addresses))
        print("The network configuration will be")
        print("\t network address: %s" %str(network.network_address))
        print("\t netmask: %s" %str(network.netmask))
        print("\t broadcast address: %s" %str(network.broadcast_address))
        first_ip, last_ip = list(network.hosts())[0], list(network.hosts())[-1] 
        print("\t host IP addresses: from %s to %s" %(first_ip, last_ip))
        ok = input("Is this configuration OK [y/n]? ")
        ok = ok.lower()
        if ok.strip() == 'y':
            not_configed = False
```

如果您运行此脚本，它将显示类似以下内容的输出：

```py
**# python 6_2_net_ip_planner.py** 
**Enter the prefixlen (24-30): 28**
**Using network address:192.168.0.0/28** 
**This prefix will give 16 IP addresses**
**The network configuration will be**
 **network address: 192.168.0.0**
 **netmask: 255.255.255.240**
 **broadcast address: 192.168.0.15**
 **host IP addresses: from 192.168.0.1 to 192.168.0.14**
**Is this configuration OK [y/n]? n**
**Enter the prefixlen (24-30): 26**
**Using network address:192.168.0.0/26** 
**This prefix will give 64 IP addresses**
**The network configuration will be**
 **network address: 192.168.0.0**
 **netmask: 255.255.255.192**
 **broadcast address: 192.168.0.63**
 **host IP addresses: from 192.168.0.1 to 192.168.0.62**
**Is this configuration OK [y/n]? y**

```

# GeoIP 查找

有时，许多应用程序需要查找 IP 地址的位置。例如，许多网站所有者可能对跟踪其访问者的位置以及根据国家、城市等标准对其 IP 进行分类感兴趣。有一个名为**python-geoip**的第三方库，它具有一个强大的接口，可以为您提供 IP 位置查询的答案。这个库由 MaxMind 提供，还提供了将最新版本的 Geolite2 数据库作为`python-geoip-geolite2`软件包进行发布的选项。这包括由 MaxMind 创建的 GeoLite2 数据，可在[www.maxmind.com](http://www.maxmind.com)上以知识共享署名-相同方式共享 3.0 未本地化许可证下获得。您也可以从他们的网站购买商业许可证。

让我们看一个如何使用这个 Geo-lookup 库的例子：

```py
import socket
from geoip import geolite2
import argparse

if __name__ == '__main__':
    # Setup commandline arguments
    parser = argparse.ArgumentParser(description='Get IP Geolocation info')
    parser.add_argument('--hostname', action="store", dest="hostname", required=True)

    # Parse arguments
    given_args = parser.parse_args()
    hostname =  given_args.hostname
    ip_address = socket.gethostbyname(hostname)
    print("IP address: {0}".format(ip_address))

    match = geolite2.lookup(ip_address)
    if match is not None:
        print('Country: ',match.country)
        print('Continent: ',match.continent) 
        print('Time zone: ', match.timezone) 
```

此脚本将显示类似以下的输出：

```py
**$ python 6_3_geoip_lookup.py --hostname=amazon.co.uk**
**IP address: 178.236.6.251**
**Country:  IE**
**Continent:  EU**
**Time zone:  Europe/Dublin**

```

您可以从开发者网站[`pythonhosted.org/python-geoip/`](http://pythonhosted.org/python-geoip/)上找到有关此软件包的更多信息。

## DNS 查找

IP 地址可以被翻译成称为域名的人类可读字符串。DNS 是网络世界中的一个重要主题。在本节中，我们将在 Python 中创建一个 DNS 客户端，并看看这个客户端将如何通过使用 Wirshark 与服务器通信。

PyPI 提供了一些 DNS 客户端库。我们将重点关注`dnspython`库，该库可在[`www.dnspython.org/`](http://www.dnspython.org/)上找到。您可以使用`easy_install`命令或`pip`命令安装此库：

```py
**$ pip install dnspython**

```

对主机的 IP 地址进行简单查询非常简单。您可以使用`dns.resolver`子模块，如下所示：

```py
**import dns.resolver**
**answers = dns.resolver.query('python.org', 'A')**
**for rdata in answers:**
 **print('IP', rdata.to_text())**

```

如果您想进行反向查找，那么您需要使用`dns.reversename`子模块，如下所示：

```py
**import dns.reversename**
**name = dns.reversename.from_address("127.0.0.1")**
**print name**
**print dns.reversename.to_address(name)**

```

现在，让我们创建一个交互式 DNS 客户端脚本，它将完成可能的记录查找，如下所示：

```py
import dns.resolver

if __name__ == '__main__':
    loookup_continue = True
    while loookup_continue:
        name = input('Enter the DNS name to resolve: ')
        record_type = input('Enter the query type [A/MX/CNAME]: ')
        answers = dns.resolver.query(name, record_type)
        if record_type == 'A':
            print('Got answer IP address: %s' %[x.to_text() for x in answers])
        elif record_type == 'CNAME':
            print('Got answer Aliases: %s' %[x.to_text() for x in answers])
        elif record_type == 'MX':
            for rdata in answers:
                print('Got answers for Mail server records:')
                print('Mailserver', rdata.exchange.to_text(), 'has preference', rdata.preference)
            print('Record type: %s is not implemented' %record_type)
        lookup_more = input("Do you want to lookup more records? [y/n]: " )
        if lookup_more.lower() == 'n':
            loookup_continue = False
```

如果您使用一些输入运行此脚本，那么您将得到类似以下的输出：

```py
**$ python 6_4_dns_client.py** 
**Enter the DNS name to resolve: google.com**
**Enter the query type [A/MX/CNAME]: MX**
**Got answers for Mail server records:**
**Mailserver alt4.aspmx.l.google.com. has preference 50**
**Got answers for Mail server records:**
**Mailserver alt2.aspmx.l.google.com. has preference 30**
**Got answers for Mail server records:**
**Mailserver alt3.aspmx.l.google.com. has preference 40**
**Got answers for Mail server records:**
**Mailserver aspmx.l.google.com. has preference 10**
**Got answers for Mail server records:**
**Mailserver alt1.aspmx.l.google.com. has preference 20**
**Do you want to lookup more records? [y/n]: y**
**Enter the DNS name to resolve: www.python.org**
**Enter the query type [A/MX/CNAME]: A**
**Got answer IP address: ['185.31.18.223']**
**Do you want to lookup more records? [y/n]: y**
**Enter the DNS name to resolve: pypi.python.org**
**Enter the query type [A/MX/CNAME]: CNAME**
**Got answer Aliases: ['python.map.fastly.net.']**
**Do you want to lookup more records? [y/n]: n**

```

## 检查 DNS 客户端/服务器通信

在以前的章节中，也许您注意到我们如何通过使用 Wireshark 捕获客户端和服务器之间的网络数据包。这是一个示例，显示了从 PyPI 安装 Python 软件包时的会话捕获：

![检查 DNS 客户端/服务器通信](img/6008OS_06_03.jpg)

FDNS 客户端/服务器通信

在 Wireshark 中，您可以通过导航到**捕获** | **选项** | **捕获过滤器**来指定`端口 53`。这将捕获所有发送到/从您的计算机的 DNS 数据包。

如您在以下截图中所见，客户端和服务器有几个请求/响应周期的 DNS 记录。它是从对主机地址（A）的标准请求开始的，然后是一个合适的响应。

![检查 DNS 客户端/服务器通信](img/6008OS_06_04.jpg)

如果您深入查看数据包，您可以看到来自服务器的响应的请求格式，如下截图所示：

![检查 DNS 客户端/服务器通信](img/6008OS_06_05.jpg)

# NTP 客户端

本章将涵盖的最后一个主题是 NTP。与集中式时间服务器同步时间是任何企业网络中的关键步骤。我们想要比较各个服务器之间的日志文件，并查看每个服务器上的时间戳是否准确；日志事件可能不会相互关联。许多认证协议，如 Kerberos，严格依赖于客户端报告给服务器的时间戳的准确性。在这里，将介绍第三方 Python `ntplib`库，然后调查 NTP 客户端和服务器之间的通信。

要创建一个 NTP 客户端，您需要调用 ntplib 的`NTPCLient`类。

```py
**import ntplib**
**from time import ctime**
**c = ntplib.NTPClient()**
**response = c.request('pool.ntp.org')**
**print ctime(response.tx_time)**

```

在这里，我们选择了`pool.ntp.org`，这是一个负载平衡的网络服务器。因此，一组 NTP 服务器将准备好响应客户端的请求。让我们从 NTP 服务器返回的响应中找到更多信息。

```py
import ntplib
from time import ctime

HOST_NAME = 'pool.ntp.org'

if __name__ == '__main__':
    params = {}
    client = ntplib.NTPClient()
    response = client.request(HOST_NAME)
    print('Received time: %s' %ctime(response.tx_time))
    print('ref_clock: ',ntplib.ref_id_to_text(response.ref_id, response.stratum))
    print('stratum: ',response.stratum)
    print('last_update: ', response.ref_time)
    print('offset:  %f' %response.offset)
    print('precision: ', response.precision)
    print('root_delay: %.6f' %response.root_delay)
    print('root_dispersion: %.6f' %response.root_dispersion)
```

详细的响应将如下所示：

```py
**$ python 6_5_ntp_client.py** 
**Received time: Sat Feb 28 17:08:29 2015**
**ref_clock:  213.136.0.252**
**stratum:  2**
**last_update:  1425142998.2**
**offset:  -4.777519**
**precision:  -23**
**root_delay: 0.019608**
**root_dispersion: 0.036987**

```

上述信息是 NTP 服务器提供给客户端的。这些信息可用于确定所提供的时间服务器的准确性。例如，stratum 值 2 表示 NTP 服务器将查询另一个具有直接附加时间源的 stratum 值 1 的 NTP 服务器。有关 NTP 协议的更多信息，您可以阅读[`tools.ietf.org/html/rfc958`](https://tools.ietf.org/html/rfc958)上的 RFC 958 文档，或访问[`www.ntp.org/`](http://www.ntp.org/)。

## 检查 NTP 客户端/服务器通信

您可以通过查看捕获的数据包来了解更多关于 NTP 的信息。为此，上述 NTP 客户端/服务器通信已被捕获，如下两个截图所示：

第一张截图显示了 NTP 客户端请求。如果您查看标志字段内部，您将看到客户端的版本号。

![检查 NTP 客户端/服务器通信](img/6008OS_06_06.jpg)

类似地，NTP 服务器的响应显示在以下截图中：

![检查 NTP 客户端/服务器通信](img/6008OS_06_07.jpg)

# 总结

在本章中，讨论了用于 IP 地址操作的标准 Python 库。介绍了两个第三方库`dnspython`和`ntplib`，分别用于与 DNS 和 NTP 服务器交互。正如您通过上述示例所看到的，这些库为您提供了与这些服务通信所需的接口。

在接下来的章节中，我们将介绍 Python 中的套接字编程。这是另一个对网络程序员来说非常有趣和受欢迎的主题。在那里，您将找到用于与 BSD 套接字编程的低级和高级 Python 库。
