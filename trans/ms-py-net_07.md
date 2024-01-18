# 使用Python进行网络监控-第1部分

想象一下，你在凌晨2点接到一个电话。电话那头的人说：“嗨，我们遇到了一个影响生产服务的困难问题。我们怀疑可能与网络有关。你能帮我们检查一下吗？”对于这种紧急的、开放式的问题，你会做什么？大多数情况下，脑海中浮现的第一件事是：在网络正常运行到出现问题之间发生了什么变化？很可能你会检查你的监控工具，看看最近几个小时内是否有任何关键指标发生了变化。更好的是，如果你收到了任何与指标基线偏差相关的监控警报。

在本书中，我们一直在讨论系统地对网络进行可预测的更改的各种方法，目标是尽可能使网络运行顺畅。然而，网络并不是静态的-远非如此-它们可能是整个基础设施中最流动的部分之一。根据定义，网络连接了基础设施的不同部分，不断地来回传递流量。有很多移动的部分可能导致您的网络停止按预期工作：硬件故障、软件错误、尽管有最好的意图，人为错误，等等。问题不在于事情是否会出错，而在于当它发生时，出了什么问题。我们需要监控我们的网络，以确保它按预期工作，并希望在它不按预期工作时得到通知。

在接下来的两章中，我们将看一些执行网络监控任务的各种方法。到目前为止，我们看到的许多工具可以通过Python进行绑定或直接管理。和我们看到的许多工具一样，网络监控涉及两个部分。首先，我们需要知道设备能够传输什么信息。其次，我们需要确定我们可以从中解释出什么有用的信息。

我们将看一些工具，让我们能够有效地监控网络：

+   **简单网络管理协议**（**SNMP**）

+   Matplotlib和Pygal可视化

+   MRTG和Cacti

这个列表并不详尽，网络监控领域显然没有缺乏商业供应商。然而，我们将要看的网络监控基础知识对于开源和商业工具都适用。

# 实验室设置

本章的实验室与[第6章](30262891-a82e-4bef-aae2-2e8fe530a16f.xhtml)中的实验室类似，*使用Python进行网络安全*，但有一个区别：网络设备都是IOSv设备。以下是这一点的说明：

![](assets/5e51171c-d3e7-46ad-b1bc-1d0afade785a.png)

两台Ubuntu主机将用于在网络中生成流量，以便我们可以查看一些非零计数器。

# SNMP

SNMP是一种标准化的协议，用于收集和管理设备。尽管该标准允许你使用SNMP进行设备管理，但根据我的经验，大多数网络管理员更喜欢将SNMP仅作为信息收集机制。由于SNMP在UDP上运行，UDP是无连接的，并且考虑到版本1和2中相对较弱的安全机制，通过SNMP进行设备更改往往会让网络运营商感到有些不安。SNMP版本3增加了加密安全性和协议的新概念和术语，但技术的适应方式在网络设备供应商之间存在差异。

SNMP在网络监控中被广泛使用，自1988年作为RFC 1065的一部分以来一直存在。操作很简单，网络管理器向设备发送`GET`和`SET`请求，设备与SNMP代理响应每个请求的信息。最广泛采用的标准是SNMPv2c，定义在RFC 1901 - RFC 1908中。它使用简单的基于社区的安全方案进行安全。它还引入了新功能，例如获取批量信息的能力。以下图显示了SNMP的高级操作：

![](assets/c3503470-34a9-43fc-8986-07ffaa47eb09.png)SNMP操作

设备中的信息存储在**管理信息库**（**MIB**）中。MIB使用包含**对象标识符**（**OID**）的分层命名空间，表示可以读取并反馈给请求者的信息。当我们谈论使用SNMP查询设备信息时，我们实际上是在谈论使用管理站点查询代表我们所需信息的特定OID。有一个常见的OID结构，例如系统和接口OID，这在供应商之间是共享的。除了常见的OID，每个供应商还可以提供特定于他们的企业级OID。

作为操作员，您需要努力将信息整合到环境中的OID结构中，以检索有用的信息。有时这可能是一个繁琐的过程，一次找到一个OID。例如，您可能会向设备OID发出请求，并收到一个值为10,000。那个值是什么？那是接口流量吗？是字节还是位？或者可能是数据包的数量？我们怎么知道？我们需要查阅标准或供应商文档才能找到答案。有一些工具可以帮助这个过程，比如MIB浏览器可以为值提供更多的元数据。但至少在我的经验中，为您的网络构建基于SNMP的监控工具有时会感觉像是一场猫鼠游戏，试图找到那个缺失的值。

从操作中可以得出一些要点：

+   实施严重依赖设备代理提供的信息量。这又取决于供应商如何对待SNMP：作为核心功能还是附加功能。

+   SNMP代理通常需要来自控制平面的CPU周期来返回一个值。这不仅对于具有大型BGP表的设备效率低下，而且在小间隔内使用SNMP查询数据也是不可行的。

+   用户需要知道OID才能查询数据。

由于SNMP已经存在一段时间，我假设您已经有了一些经验。让我们直接跳到软件包安装和我们的第一个SNMP示例。

# 设置

首先，让我们确保我们的设置中有SNMP管理设备和代理工作。SNMP捆绑包可以安装在我们实验室中的主机（客户端或服务器）或管理网络上的管理设备上。只要SNMP管理器可以通过IP与设备通信，并且受管设备允许入站连接，SNMP就可以工作。在生产中，您应该只在管理主机上安装软件，并且只允许控制平面中的SNMP流量。

在这个实验中，我们在管理网络上的Ubuntu主机和实验室中的客户端主机上都安装了SNMP以测试安全性：

```py
$ sudo apt-get install snmp
```

下一步将是在网络设备`iosv-1`和`iosv-2`上打开和配置SNMP选项。您可以在网络设备上配置许多可选参数，例如联系人、位置、机箱ID和SNMP数据包大小。这些选项是特定于设备的，您应该查看设备的文档。对于IOSv设备，我们将配置一个访问列表，以限制只有所需的主机可以查询设备，并将访问列表与SNMP社区字符串绑定。在我们的情况下，我们将使用`secret`作为只读社区字符串，`permit_snmp`作为访问列表名称。

```py
!
ip access-list standard permit_snmp
 permit 172.16.1.173 log
 deny any log
!
!
snmp-server community secret RO permit_snmp
!
```

SNMP社区字符串充当管理器和代理之间的共享密码；因此，每次要查询设备时都需要包含它。

正如本章前面提到的，与SNMP一起工作时找到正确的OID往往是战斗的一半。我们可以使用诸如思科IOS MIB定位器（[http://tools.cisco.com/ITDIT/MIBS/servlet/index](http://tools.cisco.com/ITDIT/MIBS/servlet/index)）这样的工具来查找要查询的特定OID。或者，我们可以从Cisco企业树的顶部`.1.3.6.1.4.1.9`开始遍历SNMP树。我们将执行遍历以确保SNMP代理和访问列表正在工作：

```py
$ snmpwalk -v2c -c secret 172.16.1.189 .1.3.6.1.4.1.9
iso.3.6.1.4.1.9.2.1.1.0 = STRING: "
Bootstrap program is IOSv
"
iso.3.6.1.4.1.9.2.1.2.0 = STRING: "reload" iso.3.6.1.4.1.9.2.1.3.0 = STRING: "iosv-1"
iso.3.6.1.4.1.9.2.1.4.0 = STRING: "virl.info"
...
```

我们还可以更具体地说明我们需要查询的OID：

```py
$ snmpwalk -v2c -c secret 172.16.1.189 .1.3.6.1.4.1.9.2.1.61.0
iso.3.6.1.4.1.9.2.1.61.0 = STRING: "cisco Systems, Inc.
170 West Tasman Dr.
San Jose, CA 95134-1706
U.S.A.
Ph +1-408-526-4000
Customer service 1-800-553-6387 or +1-408-526-7208
24HR Emergency 1-800-553-2447 or +1-408-526-7209
Email Address tac@cisco.com
World Wide Web http://www.cisco.com"
```

作为演示，如果我们在最后一个OID的末尾输入错误的值，例如从`0`到`1`的`1`位数，我们会看到这样的情况：

```py
$ snmpwalk -v2c -c secret 172.16.1.189 .1.3.6.1.4.1.9.2.1.61.1
iso.3.6.1.4.1.9.2.1.61.1 = No Such Instance currently exists at this OID
```

与API调用不同，没有有用的错误代码或消息；它只是简单地说明OID不存在。有时这可能非常令人沮丧。

最后要检查的是我们配置的访问列表将拒绝不需要的SNMP查询。因为我们在访问列表的允许和拒绝条目中都使用了`log`关键字，所以只有`172.16.1.173`被允许查询设备：

```py
*Mar 3 20:30:32.179: %SEC-6-IPACCESSLOGNP: list permit_snmp permitted 0 172.16.1.173 -> 0.0.0.0, 1 packet
*Mar 3 20:30:33.991: %SEC-6-IPACCESSLOGNP: list permit_snmp denied 0 172.16.1.187 -> 0.0.0.0, 1 packet
```

正如您所看到的，设置SNMP的最大挑战是找到正确的OID。一些OID在标准化的MIB-2中定义；其他的在树的企业部分下。尽管如此，供应商文档是最好的选择。有许多工具可以帮助，例如MIB浏览器；您可以将MIBs（同样由供应商提供）添加到浏览器中，并查看基于企业的OID的描述。当您需要找到您正在寻找的对象的正确OID时，像思科的SNMP对象导航器（[http://snmp.cloudapps.cisco.com/Support/SNMP/do/BrowseOID.do?local=en](http://snmp.cloudapps.cisco.com/Support/SNMP/do/BrowseOID.do?local=en)）这样的工具就变得非常有价值。

# PySNMP

PySNMP是由Ilya Etingof开发的跨平台、纯Python SNMP引擎实现（[https://github.com/etingof](https://github.com/etingof)）。它为您抽象了许多SNMP细节，正如优秀的库所做的那样，并支持Python 2和Python 3。

PySNMP需要PyASN1包。以下内容摘自维基百科：

<q>"ASN.1是一种标准和符号，描述了在电信和计算机网络中表示、编码、传输和解码数据的规则和结构。"</q>

PyASN1方便地提供了一个Python封装器，用于ASN.1。让我们首先安装这个包：

```py
cd /tmp
git clone https://github.com/etingof/pyasn1.git
cd pyasn1/
git checkout 0.2.3
sudo python3 setup.py install
```

接下来，安装PySNMP包：

```py
git clone https://github.com/etingof/pysnmp
cd pysnmp/
git checkout v4.3.10
sudo python3 setup.py install
```

由于`pysnmp.entity.rfc3413.oneliner`从版本5.0.0开始被移除（[https://github.com/etingof/pysnmp/blob/a93241007b970c458a0233c16ae2ef82dc107290/CHANGES.txt](https://github.com/etingof/pysnmp/blob/a93241007b970c458a0233c16ae2ef82dc107290/CHANGES.txt)），我们使用了较旧版本的PySNMP。如果您使用`pip`来安装包，示例可能会出现问题。

让我们看看如何使用PySNMP来查询与上一个示例中使用的相同的Cisco联系信息。我们将采取的步骤是从[http://pysnmp.sourceforge.net/faq/response-values-mib-resolution.html](http://pysnmp.sourceforge.net/faq/response-values-mib-resolution.html)中的PySNMP示例中略微修改的版本。我们将首先导入必要的模块并创建一个`CommandGenerator`对象：

```py
>>> from pysnmp.entity.rfc3413.oneliner import cmdgen
>>> cmdGen = cmdgen.CommandGenerator()
>>> cisco_contact_info_oid = "1.3.6.1.4.1.9.2.1.61.0"
```

我们可以使用`getCmd`方法执行SNMP。结果将被解包为各种变量；其中，我们最关心`varBinds`，其中包含查询结果：

```py
>>> errorIndication, errorStatus, errorIndex, varBinds = cmdGen.getCmd(
...     cmdgen.CommunityData('secret'),
...     cmdgen.UdpTransportTarget(('172.16.1.189', 161)),
...     cisco_contact_info_oid
... )
>>> for name, val in varBinds:
...     print('%s = %s' % (name.prettyPrint(), str(val)))
...
SNMPv2-SMI::enterprises.9.2.1.61.0 = cisco Systems, Inc.
170 West Tasman Dr.
San Jose, CA 95134-1706
U.S.A.
Ph +1-408-526-4000
Customer service 1-800-553-6387 or +1-408-526-7208
24HR Emergency 1-800-553-2447 or +1-408-526-7209
Email Address tac@cisco.com
World Wide Web http://www.cisco.com
>>>
```

请注意，响应值是PyASN1对象。`prettyPrint()`方法将一些这些值转换为人类可读的格式，但由于我们的结果没有被转换，我们将手动将其转换为字符串。

我们可以基于前面的交互式示例编写一个脚本。我们将其命名为`pysnmp_1.py`并进行错误检查。我们还可以在`getCmd()`方法中包含多个OID：

```py
#!/usr/bin/env/python3

from pysnmp.entity.rfc3413.oneliner import cmdgen

cmdGen = cmdgen.CommandGenerator()

system_up_time_oid = "1.3.6.1.2.1.1.3.0"
cisco_contact_info_oid = "1.3.6.1.4.1.9.2.1.61.0"

errorIndication, errorStatus, errorIndex, varBinds = cmdGen.getCmd(
    cmdgen.CommunityData('secret'),
    cmdgen.UdpTransportTarget(('172.16.1.189', 161)),
    system_up_time_oid,
    cisco_contact_info_oid
)

# Check for errors and print out results
if errorIndication:
    print(errorIndication)
else:
    if errorStatus:
        print('%s at %s' % (
            errorStatus.prettyPrint(),
            errorIndex and varBinds[int(errorIndex)-1] or '?'
            )
        )
    else:
        for name, val in varBinds:
            print('%s = %s' % (name.prettyPrint(), str(val)))

```

结果将被解包并列出两个OID的值：

```py
$ python3 pysnmp_1.py
SNMPv2-MIB::sysUpTime.0 = 660959
SNMPv2-SMI::enterprises.9.2.1.61.0 = cisco Systems, Inc.
170 West Tasman Dr.
San Jose, CA 95134-1706
U.S.A.
Ph +1-408-526-4000
Customer service 1-800-553-6387 or +1-408-526-7208
24HR Emergency 1-800-553-2447 or +1-408-526-7209
Email Address tac@cisco.com
World Wide Web http://www.cisco.com 
```

在接下来的示例中，我们将持久化我们从查询中收到的值，以便我们可以执行其他功能，比如使用数据进行可视化。在我们的示例中，我们将使用MIB-2树中的`ifEntry`来绘制与接口相关的值。您可以找到许多资源来映射`ifEntry`树；这里是我们之前访问过`ifEntry`的Cisco SNMP对象导航器网站的屏幕截图：

![](assets/dc9b8d4c-afc3-4aa7-8865-02b7faa9572d.png)SNMP ifEntry OID tree

一个快速测试将说明设备上接口的OID映射：

```py
$ snmpwalk -v2c -c secret 172.16.1.189 .1.3.6.1.2.1.2.2.1.2
iso.3.6.1.2.1.2.2.1.2.1 = STRING: "GigabitEthernet0/0"
iso.3.6.1.2.1.2.2.1.2.2 = STRING: "GigabitEthernet0/1"
iso.3.6.1.2.1.2.2.1.2.3 = STRING: "GigabitEthernet0/2"
iso.3.6.1.2.1.2.2.1.2.4 = STRING: "Null0"
iso.3.6.1.2.1.2.2.1.2.5 = STRING: "Loopback0"
```

从文档中，我们可以将`ifInOctets(10)`、`ifInUcastPkts(11)`、`ifOutOctets(16)`和`ifOutUcastPkts(17)`的值映射到它们各自的OID值。通过快速检查CLI和MIB文档，我们可以看到`GigabitEthernet0/0`数据包输出的值映射到OID`1.3.6.1.2.1.2.2.1.17.1`。我们将按照相同的过程来映射接口统计的其余OID。在CLI和SNMP之间进行检查时，请记住，值应该接近但不完全相同，因为在CLI输出和SNMP查询时间之间可能有一些流量：

```py
# Command Line Output
iosv-1#sh int gig 0/0 | i packets
 5 minute input rate 0 bits/sec, 0 packets/sec
 5 minute output rate 0 bits/sec, 0 packets/sec
 38532 packets input, 3635282 bytes, 0 no buffer
 53965 packets output, 4723884 bytes, 0 underruns

# SNMP Output
$ snmpwalk -v2c -c secret 172.16.1.189 .1.3.6.1.2.1.2.2.1.17.1
iso.3.6.1.2.1.2.2.1.17.1 = Counter32: 54070
```

如果我们处于生产环境中，我们可能会将结果写入数据库。但由于这只是一个例子，我们将把查询值写入一个平面文件。我们将编写`pysnmp_3.py`脚本来进行信息查询并将结果写入文件。在脚本中，我们已经定义了需要查询的各种OID：

```py
  # Hostname OID
  system_name = '1.3.6.1.2.1.1.5.0'

  # Interface OID
  gig0_0_in_oct = '1.3.6.1.2.1.2.2.1.10.1'
  gig0_0_in_uPackets = '1.3.6.1.2.1.2.2.1.11.1'
  gig0_0_out_oct = '1.3.6.1.2.1.2.2.1.16.1'
  gig0_0_out_uPackets = '1.3.6.1.2.1.2.2.1.17.1'
```

这些值在`snmp_query()`函数中被使用，输入为`host`、`community`和`oid`：

```py
  def snmp_query(host, community, oid):
      errorIndication, errorStatus, errorIndex, varBinds = cmdGen.getCmd(
      cmdgen.CommunityData(community),
      cmdgen.UdpTransportTarget((host, 161)),
      oid
      )
```

所有的值都被放在一个带有各种键的字典中，并写入一个名为`results.txt`的文件：

```py
  result = {}
  result['Time'] = datetime.datetime.utcnow().isoformat()
  result['hostname'] = snmp_query(host, community, system_name)
  result['Gig0-0_In_Octet'] = snmp_query(host, community, gig0_0_in_oct)
  result['Gig0-0_In_uPackets'] = snmp_query(host, community, gig0_0_in_uPackets)
  result['Gig0-0_Out_Octet'] = snmp_query(host, community, gig0_0_out_oct)
  result['Gig0-0_Out_uPackets'] = snmp_query(host, community, gig0_0_out_uPackets)

  with open('/home/echou/Master_Python_Networking/Chapter7/results.txt', 'a') as f:
      f.write(str(result))
      f.write('n')
```

结果将是一个显示查询时接口数据包的文件：

```py
# Sample output
$ cat results.txt
{'Gig0-0_In_Octet': '3990616', 'Gig0-0_Out_uPackets': '60077', 'Gig0-0_In_uPackets': '42229', 'Gig0-0_Out_Octet': '5228254', 'Time': '2017-03-06T02:34:02.146245', 'hostname': 'iosv-1.virl.info'}
{'Gig0-0_Out_uPackets': '60095', 'hostname': 'iosv-1.virl.info', 'Gig0-0_Out_Octet': '5229721', 'Time': '2017-03-06T02:35:02.072340', 'Gig0-0_In_Octet': '3991754', 'Gig0-0_In_uPackets': '42242'}
{'hostname': 'iosv-1.virl.info', 'Gig0-0_Out_Octet': '5231484', 'Gig0-0_In_Octet': '3993129', 'Time': '2017-03-06T02:36:02.753134', 'Gig0-0_In_uPackets': '42257', 'Gig0-0_Out_uPackets': '60116'}
{'Gig0-0_In_Octet': '3994504', 'Time': '2017-03-06T02:37:02.146894', 'Gig0-0_In_uPackets': '42272', 'Gig0-0_Out_uPackets': '60136', 'Gig0-0_Out_Octet': '5233187', 'hostname': 'iosv-1.virl.info'}
{'Gig0-0_In_uPackets': '42284', 'Time': '2017-03-06T02:38:01.915432', 'Gig0-0_In_Octet': '3995585', 'Gig0-0_Out_Octet': '5234656', 'Gig0-0_Out_uPackets': '60154', 'hostname': 'iosv-1.virl.info'}
...
```

我们可以使这个脚本可执行，并安排一个`cron`作业每五分钟执行一次：

```py
$ chmod +x pysnmp_3.py

# Crontab configuration
*/5 * * * * /home/echou/Master_Python_Networking/Chapter7/pysnmp_3.py
```

如前所述，在生产环境中，我们会将信息放入数据库。对于SQL数据库，您可以使用唯一ID作为主键。在NoSQL数据库中，我们可能会使用时间作为主索引（或键），因为它总是唯一的，然后是各种键值对。

我们将等待脚本执行几次，以便值被填充。如果您是不耐烦的类型，可以将`cron`作业间隔缩短为一分钟。在`results.txt`文件中看到足够多的值以制作有趣的图表后，我们可以继续下一节，看看如何使用Python来可视化数据。

# 用于数据可视化的Python

我们收集网络数据是为了深入了解我们的网络。了解数据含义的最佳方法之一是使用图形对其进行可视化。这对于几乎所有数据都是正确的，但特别适用于网络监控的时间序列数据。在过去一周内网络传输了多少数据？TCP协议在所有流量中的百分比是多少？这些都是我们可以通过使用数据收集机制（如SNMP）获得的值，我们可以使用一些流行的Python库生成可视化图形。

在本节中，我们将使用上一节从SNMP收集的数据，并使用两个流行的Python库Matplotlib和Pygal来对其进行图形化。

# Matplotlib

**Matplotlib** ([http://matplotlib.org/](http://matplotlib.org/))是Python语言及其NumPy数学扩展的2D绘图库。它可以用几行代码生成出版质量的图形，如绘图、直方图和条形图。

NumPy是Python编程语言的扩展。它是开源的，并广泛用于各种数据科学项目。您可以在[https://en.wikipedia.org/wiki/NumPy](https://en.wikipedia.org/wiki/NumPy)了解更多信息。

# 安装

安装可以使用Linux软件包管理系统完成，具体取决于您的发行版：

```py
$ sudo apt-get install python-matplotlib # for Python2
$ sudo apt-get install python3-matplotlib
```

# Matplotlib – 第一个示例

在以下示例中，默认情况下，输出图形会显示为标准输出。在开发过程中，最好先尝试最初的代码，并首先在标准输出上生成图形，然后再用脚本完成代码。如果您一直通过虚拟机跟随本书，建议您使用虚拟机窗口而不是SSH，这样您就可以看到图形。如果您无法访问标准输出，可以保存图形，然后在下载后查看（很快您将看到）。请注意，您需要在本节中的某些图形中设置`$DISPLAY`变量。

以下是本章可视化示例中使用的Ubuntu桌面的屏幕截图。在终端窗口中发出`plt.show()`命令后，`Figure 1`将出现在屏幕上。关闭图形后，您将返回到Python shell：

![](assets/6e2ca222-f974-43d2-aa0c-5cbbe61e2165.png)使用Ubuntu桌面的Matplotlib可视化

让我们先看看折线图。折线图只是给出了两个与*x*轴和*y*轴值对应的数字列表：

```py
>>> import matplotlib.pyplot as plt
>>> plt.plot([0,1,2,3,4], [0,10,20,30,40])
[<matplotlib.lines.Line2D object at 0x7f932510df98>]
>>> plt.ylabel('Something on Y')
<matplotlib.text.Text object at 0x7f93251546a0>
>>> plt.xlabel('Something on X')
<matplotlib.text.Text object at 0x7f9325fdb9e8>
>>> plt.show()
```

图形将显示为折线图：

![](assets/a829928b-284f-4292-ab0b-62b334bcba6f.png)Matplotlib折线图

或者，如果您无法访问标准输出或者首先保存了图形，可以使用`savefig()`方法：

```py
>>> plt.savefig('figure1.png')
or
>>> plt.savefig('figure1.pdf')
```

有了这些基本的图形绘制知识，我们现在可以绘制从SNMP查询中收到的结果了。

# 用于SNMP结果的Matplotlib

在我们的第一个Matplotlib示例中，即`matplotlib_1.py`，我们将除了`pyplot`之外还导入*dates*模块。我们将使用`matplotlib.dates`模块而不是Python标准库`dates`模块。与Python`dates`模块不同，`mapplotlib.dates`库将在内部将日期值转换为Matplotlib所需的浮点类型：

```py
  import matplotlib.pyplot as plt
  import matplotlib.dates as dates
```

Matplotlib提供了复杂的日期绘图功能；您可以在[http://matplotlib.org/api/dates_api.html](http://matplotlib.org/api/dates_api.html)找到更多信息。

在脚本中，我们将创建两个空列表，分别表示*x-*轴和*y-*轴的值。请注意，在第12行，我们使用内置的`eval()` Python函数将输入读取为字典，而不是默认的字符串：

```py
   x_time = []
   y_value = []

   with open('results.txt', 'r') as f:
       for line in f.readlines():
           line = eval(line)
           x_time.append(dates.datestr2num(line['Time']))
           y_value.append(line['Gig0-0_Out_uPackets'])
```

为了以人类可读的日期格式读取*x-*轴的值，我们需要使用`plot_date()`函数而不是`plot()`。我们还将微调图形的大小，并旋转*x-*轴上的值，以便我们可以完整地读取该值：

```py
  plt.subplots_adjust(bottom=0.3)
  plt.xticks(rotation=80)

  plt.plot_date(x_time, y_value)
  plt.title('Router1 G0/0')
  plt.xlabel('Time in UTC')
  plt.ylabel('Output Unicast Packets')
  plt.savefig('matplotlib_1_result.png')
  plt.show()
```

最终结果将显示Router1 Gig0/0和输出单播数据包，如下所示：

![](assets/ece57d54-836d-4b3e-86ef-ec605416081c.png)Router1 Matplotlib图

请注意，如果您喜欢直线而不是点，您可以在`plot_date()`函数中使用第三个可选参数：

```py
     plt.plot_date(x_time, y_value, "-")
```

我们可以重复输出八进制、输入单播数据包和输入的步骤作为单独的图形。然而，在我们接下来的例子中，也就是`matplotlib_2.py`中，我们将向您展示如何在相同的时间范围内绘制多个值，以及其他Matplotlib选项。

在这种情况下，我们将创建额外的列表，并相应地填充值：

```py
   x_time = []
   out_octets = []
   out_packets = []
   in_octets = []
   in_packets = []

   with open('results.txt', 'r') as f:
       for line in f.readlines():
   ...
           out_packets.append(line['Gig0-0_Out_uPackets'])
           out_octets.append(line['Gig0-0_Out_Octet'])
           in_packets.append(line['Gig0-0_In_uPackets'])
           in_octets.append(line['Gig0-0_In_Octet'])
```

由于我们有相同的*x-*轴值，我们可以将不同的*y-*轴值添加到同一图中：

```py
  # Use plot_date to display x-axis back in date format
  plt.plot_date(x_time, out_packets, '-', label='Out Packets')
  plt.plot_date(x_time, out_octets, '-', label='Out Octets')
  plt.plot_date(x_time, in_packets, '-', label='In Packets')
  plt.plot_date(x_time, in_octets, '-', label='In Octets')
```

还要在图中添加网格和图例：

```py
  plt.legend(loc='upper left')
  plt.grid(True)
```

最终结果将把所有值合并到一个图中。请注意，左上角的一些值被图例挡住了。您可以调整图形的大小和/或使用平移/缩放选项来在图形周围移动，以查看值：

![](assets/5c6c3d57-899b-402b-b554-c85e94fe3b24.png)Router 1 – Matplotlib多线图

Matplotlib中有许多其他绘图选项；我们当然不仅限于绘制图形。例如，我们可以使用以下模拟数据来绘制我们在线上看到的不同流量类型的百分比：

```py
#!/usr/bin/env python3
# Example from http://matplotlib.org/2.0.0/examples/pie_and_polar_charts/pie_demo_features.html
import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'TCP', 'UDP', 'ICMP', 'Others'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0) # Make UDP stand out

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
 shadow=True, startangle=90)
ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
```

上述代码导致了从`plt.show()`生成的饼图：

![](assets/6a9328fe-b860-4287-98e2-a56c469da3aa.png)Matplotlib饼图

# 附加的Matplotlib资源

Matplotlib是最好的Python绘图库之一，能够生成出版质量的图形。与Python一样，它的目标是使复杂的任务变得简单。在GitHub上有超过7550颗星（还在增加），它也是最受欢迎的开源项目之一。它的受欢迎程度直接转化为更快的错误修复、友好的用户社区和通用的可用性。学习这个包需要一点时间，但是非常值得努力。

在本节中，我们只是浅尝了Matplotlib的表面。您可以在[http://matplotlib.org/2.0.0/index.html](http://matplotlib.org/2.0.0/index.html)（Matplotlib项目页面）和[https://github.com/matplotlib/matplotlib](https://github.com/matplotlib/matplotlib)（Matplotlib GitHub存储库）找到更多资源。

在接下来的部分中，我们将看一下另一个流行的Python图形库：**Pygal**。

# Pygal

Pygal（[http://www.pygal.org/](http://www.pygal.org/)）是一个用Python编写的动态SVG图表库。在我看来，Pygal的最大优势是它能够轻松本地生成**可伸缩矢量图形**（**SVG**）格式的图形。SVG相对于其他图形格式有许多优势，但其中两个主要优势是它对Web浏览器友好，并且提供了可伸缩性而不会损失图像质量。换句话说，您可以在任何现代Web浏览器中显示生成的图像，并且可以放大和缩小图像，而不会丢失图形的细节。我提到了我们可以在几行Python代码中做到这一点吗？这有多酷？

# 安装

安装是通过`pip`完成的：

```py
$ sudo pip install pygal #Python 2
$ sudo pip3 install pygal
```

# Pygal - 第一个例子

让我们看一下Pygal文档中演示的线图示例，网址为[http://pygal.org/en/stable/documentation/types/line.html](http://pygal.org/en/stable/documentation/types/line.html)：

```py
>>> import pygal
>>> line_chart = pygal.Line()
>>> line_chart.title = 'Browser usage evolution (in %)'
>>> line_chart.x_labels = map(str, range(2002, 2013))
>>> line_chart.add('Firefox', [None, None, 0, 16.6, 25, 31, 36.4, 45.5, 46.3, 42.8, 37.1])
<pygal.graph.line.Line object at 0x7fa0bb009c50>
>>> line_chart.add('Chrome', [None, None, None, None, None, None, 0, 3.9, 10.8, 23.8, 35.3])
<pygal.graph.line.Line object at 0x7fa0bb009c50>
>>> line_chart.add('IE', [85.8, 84.6, 84.7, 74.5, 66, 58.6, 54.7, 44.8, 36.2, 26.6, 20.1])
<pygal.graph.line.Line object at 0x7fa0bb009c50>
>>> line_chart.add('Others', [14.2, 15.4, 15.3, 8.9, 9, 10.4, 8.9, 5.8, 6.7, 6.8, 7.5])
<pygal.graph.line.Line object at 0x7fa0bb009c50>
>>> line_chart.render_to_file('pygal_example_1.svg')
```

在这个例子中，我们创建了一个带有`x_labels`的线对象，自动呈现为11个单位的字符串。每个对象都可以以列表格式添加标签和值，例如Firefox、Chrome和IE。

这是在Firefox中查看的结果图：

![](assets/29abfeb0-b81b-4751-b7d3-ba3dc6bd8cb5.png)Pygal示例图

现在我们可以看到Pygal的一般用法，我们可以使用相同的方法来绘制我们手头上的SNMP结果。我们将在接下来的部分中进行这样做。

# Pygal用于SNMP结果

对于Pygal线图，我们可以大致按照Matplotlib示例的相同模式进行操作，其中我们通过读取文件创建值列表。我们不再需要将*x-*轴值转换为内部浮点数，就像我们为Matplotlib所做的那样；但是，我们确实需要将我们将在浮点数中收到的每个值中的数字转换为浮点数：

```py
  #!/usr/bin/env python3

  import pygal

  x_time = []
  out_octets = []
  out_packets = []
  in_octets = []
  in_packets = []

  with open('results.txt', 'r') as f:
      for line in f.readlines():
          line = eval(line)
          x_time.append(line['Time'])
          out_packets.append(float(line['Gig0-0_Out_uPackets']))
          out_octets.append(float(line['Gig0-0_Out_Octet']))
          in_packets.append(float(line['Gig0-0_In_uPackets']))
          in_octets.append(float(line['Gig0-0_In_Octet']))
```

我们可以使用我们看到的相同机制来构建线图：

```py
  line_chart = pygal.Line()
  line_chart.title = "Router 1 Gig0/0"
  line_chart.x_labels = x_time
  line_chart.add('out_octets', out_octets)
  line_chart.add('out_packets', out_packets)
  line_chart.add('in_octets', in_octets)
  line_chart.add('in_packets', in_packets)
  line_chart.render_to_file('pygal_example_2.svg')
```

结果与我们已经看到的类似，但是图表现在以SVG格式呈现，可以轻松地显示在网页上。它可以在现代Web浏览器中查看：

![](assets/6e383a39-e3b8-48c9-a6be-6fc05fcf4cb1.png)路由器1—Pygal多线图

就像Matplotlib一样，Pygal为图表提供了更多的选项。例如，要在Pygal中绘制我们之前看到的饼图，我们可以使用`pygal.Pie()`对象：

```py
#!/usr/bin/env python3

import pygal

line_chart = pygal.Pie()
line_chart.title = "Protocol Breakdown"
line_chart.add('TCP', 15)
line_chart.add('UDP', 30)
line_chart.add('ICMP', 45)
line_chart.add('Others', 10)
line_chart.render_to_file('pygal_example_3.svg')
```

生成的SVG文件将类似于Matplotlib生成的PNG：

![](assets/dd33b69f-99ec-45bb-bc0a-386ee0c65bcf.png)Pygal饼图

# 其他Pygal资源

Pygal为您从基本网络监控工具（如SNMP）收集的数据提供了更多可定制的功能和图形能力。在本节中，我们演示了简单的线图和饼图。您可以在此处找到有关项目的更多信息：

+   **Pygal文档**：[http://www.pygal.org/en/stable/index.html](http://www.pygal.org/en/stable/index.html)

+   **Pygal GitHub项目页面**：[https://github.com/Kozea/pygal](https://github.com/Kozea/pygal)

在接下来的部分中，我们将继续使用SNMP主题进行网络监控，但使用一个名为**Cacti**的功能齐全的网络监控系统。

# Cacti的Python

在我作为地区ISP的初级网络工程师工作的早期，我们使用开源跨平台**多路由器流量图**（**MRTG**）（[https://en.wikipedia.org/wiki/Multi_Router_Traffic_Grapher](https://en.wikipedia.org/wiki/Multi_Router_Traffic_Grapher)）工具来检查网络链路上的流量负载。我们几乎完全依赖于该工具进行流量监控。我真的很惊讶开源项目可以有多好和有用。这是第一个将SNMP、数据库和HTML的细节抽象化为网络工程师的开源高级网络监控系统之一。然后出现了**循环数据库工具**（**RRDtool**）（[https://en.wikipedia.org/wiki/RRDtool](https://en.wikipedia.org/wiki/RRDtool)）。在1999年的首次发布中，它被称为“正确的MRTG”。它极大地改进了后端的数据库和轮询器性能。

Cacti（[https://en.wikipedia.org/wiki/Cacti_(software)](https://en.wikipedia.org/wiki/Cacti_(software)）于2001年发布，是一个开源的基于Web的网络监控和图形工具，旨在作为RRDtool的改进前端。由于MRTG和RRDtool的传承，您会注意到熟悉的图表布局、模板和SNMP轮询器。作为一个打包工具，安装和使用将需要保持在工具本身的范围内。但是，Cacti提供了我们可以使用Python的自定义数据查询功能。在本节中，我们将看到如何将Python用作Cacti的输入方法。

# 安装

在Ubuntu上使用APT进行安装非常简单：

```py
$ sudo apt-get install cacti
```

这将触发一系列安装和设置步骤，包括MySQL数据库、Web服务器（Apache或lighttpd）和各种配置任务。安装完成后，导航到`http://<ip>/cacti`开始使用。最后一步是使用默认用户名和密码（`admin`/`admin`）登录；您将被提示更改密码。

一旦你登录，你可以按照文档添加设备并将其与模板关联。有一个预制的Cisco路由器模板可以使用。Cacti在[http://docs.cacti.net/](http://docs.cacti.net/)上有关于添加设备和创建第一个图形的良好文档，所以我们将快速查看一些你可以期望看到的屏幕截图：

![](assets/30d73b73-895c-472d-8723-22a9be73d6f3.png)

当你能看到设备的正常运行时间时，这是SNMP通信正在工作的一个标志：

![](assets/fdf75bec-2cb7-4e51-a6c8-87e9cf54ca4d.png)

你可以为设备添加接口流量和其他统计信息的图形：

![](assets/d02f3ac0-6883-4358-a3ae-deeebff30af9.png)

一段时间后，你会开始看到流量，如下所示：

![](assets/4e894e42-b8ca-4140-bf0c-e19f83ebef34.png)

我们现在准备看一下如何使用Python脚本来扩展Cacti的数据收集功能。

# Python脚本作为输入源

在我们尝试将Python脚本作为输入源之前，有两份文档我们应该阅读：

+   数据输入方法：[http://www.cacti.net/downloads/docs/html/data_input_methods.html](http://www.cacti.net/downloads/docs/html/data_input_methods.html)

+   使你的脚本与Cacti一起工作：[http://www.cacti.net/downloads/docs/html/making_scripts_work_with_cacti.html](http://www.cacti.net/downloads/docs/html/making_scripts_work_with_cacti.html)

有人可能会想知道使用Python脚本作为数据输入扩展的用例是什么。其中一个用例是为那些没有相应OID的资源提供监控，例如，如果我们想知道访问列表`permit_snmp`允许主机`172.16.1.173`进行SNMP查询的次数。我们知道我们可以通过CLI看到匹配的次数：

```py
iosv-1#sh ip access-lists permit_snmp | i 172.16.1.173
 10 permit 172.16.1.173 log (6362 matches)
```

然而，很可能与这个值没有关联的OID（或者我们可以假装没有）。这就是我们可以使用外部脚本生成一个可以被Cacti主机消耗的输出的地方。

我们可以重用我们在[第2章](8cefc139-8dfa-4250-81bf-928231e20b22.xhtml)中讨论的Pexpect脚本，`chapter1_1.py`。我们将其重命名为`cacti_1.py`。除了执行CLI命令并保存输出之外，一切都应该与原始脚本一样熟悉：

```py
for device in devices.keys():
...
    child.sendline('sh ip access-lists permit_snmp | i 172.16.1.173')
    child.expect(device_prompt)
    output = child.before
...
```

原始形式的输出如下：

```py
b'sh ip access-lists permit_snmp | i 172.16.1.173rn 10 permit 172.16.1.173 log (6428 matches)rn'
```

我们将使用`split()`函数对字符串进行处理，只留下匹配的次数并在脚本中将其打印到标准输出：

```py
print(str(output).split('(')[1].split()[0])
```

为了测试这一点，我们可以执行脚本多次来查看增量的数量：

```py
$ ./cacti_1.py
6428
$ ./cacti_1.py
6560
$ ./cacti_1.py
6758
```

我们可以将脚本设置为可执行，并将其放入默认的Cacti脚本位置：

```py
$ chmod a+x cacti_1.py
$ sudo cp cacti_1.py /usr/share/cacti/site/scripts/
```

Cacti文档，可在[http://www.cacti.net/downloads/docs/html/how_to.html](http://www.cacti.net/downloads/docs/html/how_to.html)上找到，提供了如何将脚本结果添加到输出图形的详细步骤。这些步骤包括将脚本添加为数据输入方法，将输入方法添加到数据源，然后创建一个图形进行查看：

![](assets/76e0529b-8252-41c3-b8c3-ab9f9cd550f3.png)

SNMP是提供网络监控服务给设备的常见方式。RRDtool与Cacti作为前端提供了一个良好的平台，可以通过SNMP用于所有的网络设备。

# 总结

在本章中，我们探讨了通过SNMP执行网络监控的方法。我们在网络设备上配置了与SNMP相关的命令，并使用了我们的网络管理VM与SNMP轮询程序来查询设备。我们使用了PySNMP模块来简化和自动化我们的SNMP查询。我们还学习了如何将查询结果保存在一个平面文件或数据库中，以便用于将来的示例。

在本章的后面，我们使用了两种不同的Python可视化包，即Matplotlib和Pygal，来绘制SNMP结果的图表。每个包都有其独特的优势。Matplotlib是一个成熟、功能丰富的库，在数据科学项目中被广泛使用。Pygal可以原生生成灵活且适合网络的SVG格式图表。我们看到了如何生成对网络监控相关的折线图和饼图。

在本章的末尾，我们看了一个名为Cacti的全面网络监控工具。它主要使用SNMP进行网络监控，但我们看到当远程主机上没有SNMP OID时，我们可以使用Python脚本作为输入源来扩展平台的监控能力。

在[第8章](5f7e76ef-d93a-4689-8054-8be72d41d69b.xhtml)中，《使用Python进行网络监控-第2部分》，我们将继续讨论我们可以使用的工具来监控我们的网络，并了解网络是否表现如预期。我们将研究使用NetFlow、sFlow和IPFIX进行基于流的监控。我们还将使用诸如Graphviz之类的工具来可视化我们的网络拓扑，并检测任何拓扑变化。最后，我们将使用Elasticsearch、Logstash和Kibana，通常被称为ELK堆栈，来监控网络日志数据以及其他与网络相关的输入。
