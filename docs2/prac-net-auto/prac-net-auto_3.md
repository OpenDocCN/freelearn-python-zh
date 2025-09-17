# 从网络中访问和挖掘数据

回顾一下，我们现在对编写 Python 脚本和如何从信息中获取有意义的数据有了相当的了解。我们已经介绍了如何编写 Python 脚本，与网络设备交互，并且还研究了 PowerShell 的基础，以便我们可以使用 PowerShell 和 Python 脚本。现在，我们将通过查看各种示例来深入了解 Python 的使用。在本章中，我们将专注于与各种网络设备一起工作，以挖掘或从设备中获取相关信息，处理这些信息以创建新的配置，并将其推回设备以增加或增强功能。

我们将研究一些我们可能遇到的一些常见场景，并尝试用 Python 来解决它们。这些示例或场景可以根据程序员的实际需求进行扩展，并可作为参考以实现复杂任务的自动化。

我们将介绍的一些关键概念如下：

+   设备配置

+   多供应商环境

+   IPv4 到 IPv6 的转换

+   办公室/数据中心搬迁

+   站点推广

+   交换机 BYOD 配置

+   设备操作系统升级

+   IP 配置/接口解析

# 设备配置

我们需要部署三个具有标准基本配置的路由器。基本配置在每个路由器上保持不变，但由于每个路由器都不同，我们需要自动化生成每个路由器的三个配置文件。假设所有路由器都具有标准硬件配置，具有相同类型的端口：

![图片](img/d50bf4ba-cbd3-4e52-b0fc-0259d1dd9c06.jpg)

如我们在图中所见，路由器 R1、R2 和 R3 的布线如下：

+   R1 `f1/0`（快速以太网 1/0）连接到 R2 `f1/0`

+   R1 `f0/0` 连接到 R3 `f0/0`

+   R2 `f0/1` 连接到 R3 `f0/1`

标准配置或模板如下：

```py
 hostname <hname>
 ip domain-lookup
 ip name-server <nameserver>
 logging host <loghost>
 username cisco privilege 15 password cisco
 enable password cisco
 ip domain-name checkmetest.router
line vty 0 4
 exec-timeout 5
```

增加一些复杂性，我们需要确保每个路由器的`名称服务器`不同。如果每个路由器将被部署在不同的网络中，这是我们想要的映射：

+   R1 -> 主机名 testindia

+   R2 -> 主机名 testusa

+   R3 -> 主机名 testUK

日志主机和名称服务器将取决于区域，因此映射如下：

+   **印度路由器**：日志服务器（1.1.1.1）和名称服务器（1.1.1.2）

+   **美国路由器**：日志服务器（2.1.1.1）和名称服务器（2.1.1.2）

+   **英国路由器**：日志服务器（3.1.1.1）和名称服务器（3.1.1.2）

执行请求任务的代码如下：

```py
ipdict={'india': '1.1.1.1,1.1.1.2', 'uk': '3.1.1.1,3.1.1.2', 'usa': '2.1.1.1,2.1.1.2'}

standardtemplate="""
hostname <hname>
ip domain-lookup
ip name-server <nameserver>
logging host <loghost>
username cisco privilege 15 password cisco
enable password cisco
ip domain-name checkmetest.router

line vty 0 4
 exec-timeout 5
"""

routerlist="R1,R2,R3"
routers=routerlist.split(",")
for router in routers:
print ("Now printing config for",router)
    if "R1" in router:
        hostname="testindia"
        getips=ipdict["india"]
        getips=getips.split(",")
        logserver=getips[0]
        nameserver=getips[1]
    if "R2" in router:
        hostname="testusa"
        getips=ipdict["usa"]
        getips=getips.split(",")
        logserver=getips[0]
        nameserver=getips[1]
    if "R3" in router:
        hostname="testUK"
        getips=ipdict["uk"]
        getips=getips.split(",")
        logserver=getips[0]
        nameserver=getips[1]
    generatedconfig=standardtemplate
    generatedconfig=generatedconfig.replace("<hname>",hostname)
    generatedconfig=generatedconfig.replace("<nameserver>",nameserver)
    generatedconfig=generatedconfig.replace("<loghost>",logserver)
    print (generatedconfig)
```

第一个列表是一个字典，它根据区域定义了日志主机和名称服务器配置。`standardtemplate`变量用于存储模板。如果我们需要将多行值存储在变量中，我们可以使用前面示例中看到的三引号格式。

现在，因为我们目前知道通用的或默认的主机名，我们可以逐个解析当前的主机名，并根据主机名值生成配置。此输出可以保存到文件中，也可以直接从脚本生成并粘贴到路由器上进行基本配置。同样，我们可以通过添加下一个示例中显示的 IP 地址来增强此脚本，格式为`<ipaddress>` `<subnet mask>`：

+   testindia f1/0: `10.0.0.1` `255.0.0.0`

+   testusa f1/0: `10.0.0.2` `255.0.0.0`

+   testindia f0/0: `11.0.0.1` `255.0.0.0`

+   testUK f0/0: `11.0.0.2` `255.0.0.0`

+   testusa f0/1: `12.0.0.1` `255.0.0.0`

+   testUK f0/1: `12.0.0.2` `255.0.0.0`

执行此任务的代码如下：

```py
def getipaddressconfig(routername):
    intconfig=""
    sampletemplate="""
    interface f0/0
     ip address ipinfof0/0
    interface f1/0
     ip address ipinfof1/0
    interface f0/1
     ip address ipinfof0/1
    """
    if (routername == "testindia"):
        f0_0="11.0.0.1 255.0.0.0"
        f1_0="10.0.0.1 255.0.0.0"
        sampletemplate=sampletemplate.replace("ipinfof0/0",f0_0)
        sampletemplate=sampletemplate.replace("ipinfof1/0",f1_0)
        sampletemplate=sampletemplate.replace("interface f0/1\n","")
        sampletemplate=sampletemplate.replace("ip address ipinfof0/1\n","")
    if (routername == "testusa"):
        f0_0="11.0.0.1 255.0.0.0"
        f0_1="12.0.0.1 255.0.0.0"
        sampletemplate=sampletemplate.replace("ipinfof0/0",f0_0)
        sampletemplate=sampletemplate.replace("ipinfof0/1",f0_1)
        sampletemplate=sampletemplate.replace("interface f1/0\n","")
        sampletemplate=sampletemplate.replace("ip address ipinfof1/0\n","")
    if (routername == "testUK"):
        f0_0="11.0.0.2 255.0.0.0"
        f0_1="12.0.0.2 255.0.0.0"
        sampletemplate=sampletemplate.replace("ipinfof0/0",f0_0)
        sampletemplate=sampletemplate.replace("ipinfof0/1",f0_1)
        sampletemplate=sampletemplate.replace("interface f1/0\n","")
        sampletemplate=sampletemplate.replace("ip address ipinfof1/0\n","")
    return sampletemplate

#calling this function
myfinaloutput=getipaddressconfig("testUK") #for UK router
myfinaloutput=getipaddressconfig("testindia") #for USA router
myfinaloutput=getipaddressconfig("testusa") #for India router
```

在这种情况下，我们定义了一个具有标准接口模板的函数。现在，该模板已根据函数中的调用值（即路由器名称）进行了修改，并进行了更新。同时，我们通过用两个双引号`""`（之间没有空格）表示的无用行替换它们来删除未使用的行。

一旦我们有了生成的配置，我们可以使用简单的文件处理操作来保存它：

```py
#Suppose our final value is in myfinaloutput and file name is myrouterconfig.txt
fopen=open("C:\check\myrouterconfig.txt","w")
fopen.write(myfinaloutput)
fopen.close()
```

如我们所见，通用模板和接口配置的输出可以连接或添加到名为`myfinaloutput`的变量中，现在它被保存在`C:\check`文件夹中的名为`myrouterconfig.txt`的文件中。

同样，我们可以通过添加更多特定任务的功能来增强脚本，例如**开放式最短路径优先**（**OSPF**）配置和**边界网关协议**（**BGP**）配置，根据特定的路由器名称创建增强和复杂的配置，并将它们存储在单独的`.txt`文件中，以便最终推送到网络设备。

# 多供应商环境

有时我们有许多供应商参与配置更改，甚至从头开始创建各种模板。我们有一些供应商，如 Arista、Cisco（IOS、NXOS）和 Juniper，在不同层次上参与网络设计。在处理此类情况时，我们需要清楚每个供应商正在工作的层次，并为每种类型的供应商创建动态模板。

在一个场景中，如果我们知道硬件平台和设备的作用（例如接入层、核心层或**机架顶部**（**TOR**）层），我们可以使用非常基本的参数快速生成配置。

如果一个设备处于生产状态，我们可以使用 SNMP 协议获取该设备的信息，并根据设备的返回类型创建动态值。

作为基本思路，我们可以查看[`wiki.opennms.org/wiki/Hardware_Inventory_Entity_MIB.`](https://wiki.opennms.org/wiki/Hardware_Inventory_Entity_MIB)

这包含了当前开放标准**管理信息库**（**MIB**）的信息，该信息由 SNMP 用于获取基本设备信息。

再次遵循良好的实践，我们应该确保我们创建一个通用的函数，该函数可以返回设备类型。此外，SNMP **对象标识符（OIDs**）可以深入到获取诸如当前接口数量、接口状态以及哪些接口是操作性的等信息，这样我们就可以根据设备的当前健康状况或从设备获取的信息快速做出智能决策。

我们将安装并使用 `PySNMP` 库来查询设备上的 SNMP。为了安装它，我们将使用之前的 `pip install pysnmp` 方法。

基本的 `PySNMP` 文档可以在以下 URL 中查看：

[`pynet.twb-tech.com/blog/snmp/python-snmp-intro.html`](https://pynet.twb-tech.com/blog/snmp/python-snmp-intro.html)

例如，我们将尝试获取网络设备的当前版本：

```py
from pysnmp.hlapi import *

errorIndication, errorStatus, errorIndex, varBinds = next(
    getCmd(SnmpEngine(),
           CommunityData('public', mpModel=0),
           UdpTransportTarget(('192.168.255.249', 161)),
           ContextData(),
           ObjectType(ObjectIdentity('SNMPv2-MIB', 'sysDescr', 0)))
)

if errorIndication:
    print(errorIndication)
elif errorStatus:
    print('%s at %s' % (errorStatus.prettyPrint(),
                        errorIndex and varBinds[int(errorIndex) - 1][0] or '?'))
else:
    for varBind in varBinds:
        print(' = '.join([x.prettyPrint() for x in varBind]))
```

当查询网络设备时，前面代码的示例输出如下：

![](img/77a31f3a-e4ed-4b2d-832d-148e5a11a9bf.jpg)

在我们的测试路由器上，我们使用 `snmp-server community public RO` 命令启用了 SNMP，并通过执行前面编写的 Python 代码，获取了 `RO` 字符串 public 以读取 `sysDescr.0` 值，这在 Cisco 标准中是截断的显示版本。

使用通过 SNMP 获取信息的方法，我们可以发现有哪些类型的设备，并根据输出，我们可以做出智能决策，例如生成特定设备的配置，而无需请求设备类型输入。

此外，这里有一个使用 PySNMP 获取路由器当前接口的示例：

```py
from pysnmp.entity.rfc3413.oneliner import cmdgen

cmdGen = cmdgen.CommandGenerator()

errorIndication, errorStatus, errorIndex, varBindTable = cmdGen.bulkCmd(
    cmdgen.CommunityData('public'),
    cmdgen.UdpTransportTarget(('192.168.255.249', 161)),
    0,25,
    '1.3.6.1.2.1.2.2.1.2'
)

# Check for errors and print out results
if errorIndication:
    print(errorIndication)
else:
    if errorStatus:
        print('%s at %s' % (
            errorStatus.prettyPrint(),
            errorIndex and varBindTable[-1][int(errorIndex)-1] or '?'
            )
        )
    else:
        for varBindTableRow in varBindTable:
            for name, val in varBindTableRow:
                print('%s = %s' % (name.prettyPrint(), val.prettyPrint()))
```

当在示例路由器上查询接口信息时的输出如下：

![](img/c0cae82a-c71e-4d5a-a69e-fe20240f9b9e.jpg)

如我们所见，我们使用 `bulkCmd` 方法，该方法遍历所有 SNMP 值并返回接口的输出。

OID `1.3.6.1.2.1.2.2.1.2` 被用作参考，从设备中获取这些值。

以类似的方式，我们可以利用不同厂商提供的可用 SNMP OIDs 从多个设备中获取特定信息，并根据返回的值执行预期的任务。

# IP 配置/接口解析

在许多情况下，我们需要解析接口配置以获取有用的信息。例如，从一个设备列表中找到所有 trunk 接口。另一个例子可能是找到所有 `admin-shutdown`（在路由器上关闭）的接口，或者甚至从接口中获取 IP 地址配置。

可能会有这样的情况，我们需要找出特定的 IP 地址或子网是否已在路由器上配置。

提取任何信息的一个好方法是使用正则表达式。正则表达式是一个用于匹配特定模式并从解析的文本中获取匹配的模式或验证是否存在特定模式的术语。

这里是 Python 中使用的一些最基本和最重要的正则表达式：

| `.` | 匹配除换行符之外的任何字符 |
| --- | --- |
| `^` | 匹配字符串的开始 |
| `$` | 匹配字符串的末尾 |
| `*` | 匹配 0 或更多重复 |
| `+` | 匹配 1 或更多重复 |
| `?` | 匹配 0 或 1 次重复 |
| `\A` | 仅匹配字符串的开始 |
| `\b` | 匹配一个空字符串，仅在单词的开始或结束时 |
| `\B` | 匹配一个空字符串，仅在它不是单词的开始或结束时 |
| `\d` | 匹配数字（例如 [0-9]） |
| `\D` | 匹配任何非数字字符（例如 [⁰-9]） |
| `\Z` | 仅匹配字符串的末尾 |
| `\` | 转义特殊字符 |
| `[]` | 匹配一组字符 |
| `[a-z]` | 匹配任何小写 ASCII 字母 |
| `[^]` | 匹配不在集合中的字符 |
| `A&#124;B` | 匹配 A 或 B 正则表达式（非贪婪） |
| `\s` | 匹配空白字符（例如 [ \t\n\r\f\v]） |
| `\S` | 匹配非空白字符（例如 [^ \t\n\r\f\v]） |
| `\w` | 匹配任何 Unicode 单词字符（例如 [a-zA-Z0-9_]） |
| `\W` | 匹配任何不是 Unicode 单词字符的字符（例如 [^a-zA-Z0-9_]） |

从这个字符串中，我的 IP 地址是 `10.10.10.20`，子网掩码是 `255.255.255.255`，我们需要使用正则表达式获取 IP 地址和子网掩码：

```py
import re
mystring='My ip address is 10.10.10.20 and by subnet mask is 255.255.255.255'

if (re.search("ip address",mystring)):
    ipaddregex=re.search("ip address is \d+.\d+.\d+.\d+",mystring)
    ipaddregex=ipaddregex.group(0)
    ipaddress=ipaddregex.replace("ip address is ","")
    print ("IP address is :",ipaddress)

if (re.search("subnet mask",mystring)):
    ipaddregex=re.search("subnet mask is \d+.\d+.\d+.\d+",mystring)
    ipaddregex=ipaddregex.group(0)
    ipaddress=ipaddregex.replace("subnet mask is ","")
    print ("Subnet mask is :",ipaddress)

```

执行上述代码的输出如下：

![图片](img/5842005f-74fc-4bdf-8fac-fb8cd53aa768.jpg)

如我们所见，我们使用的 IP 地址正则表达式是 `\d+.\d+.\d+.\d+`。这里的 `\d` 表示一个数字，而 `+` 表示多次重复，因为我们正在寻找由三个点分隔的多个数字的值。

然而，在我们的情况下，我们在两个地方有这种重复类型，一个在 IP 地址中，另一个在子网掩码中，因此我们修改正则表达式以搜索 `ip address is  \d+.\d+.\d+.\d+` 用于 IP 地址和 `subnet mask is \d+.\d+.\d+.\d+` 用于子网掩码。`if` 循环内部的 `re.search` 命令在找到匹配项时返回 true，如果没有找到匹配项则返回 false。在示例中，一旦我们在 `if` 条件中找到模式，我们再次使用 `re.search` 并使用 `.group(0)` 提取值，现在它包含匹配的正则表达式模式。

由于我们只关心 IP 地址和子网掩码，我们将其他字符串值替换为空白或无值，以便我们只获取特定的 IP 地址和子网掩码值。

此外，使用内置的 `socket` 库，可能存在检查 IP 地址（IPv4 或 IPv6）是否有效的理由。以下是一个示例：

```py
import socket

def validateipv4ip(address):
    try:
        socket.inet_aton(address)
        print ("Correct IPv4 IP")
    except socket.error:
        print ("wrong IPv4 IP")

def validateipv6ip(address):
    ### for IPv6 IP address validation
    try:
        socket.inet_pton(socket.AF_INET6,address)
        print ("Correct IPv6 IP")
    except socket.error:
        print ("wrong IPv6 IP")

#correct IPs:
validateipv4ip("2.2.2.1")
validateipv6ip("2001:0db8:85a3:0000:0000:8a2e:0370:7334")

#Wrong IPs:
validateipv4ip("2.2.2.500")
validateipv6ip("2001:0db8:85a3:0000:0000:8a2e")

```

上述代码的输出如下：

![图片](img/78441d58-8d1c-4c0d-945f-c447f5593419.jpg)

使用 `socket` 库，我们验证 IPv4 和 IPv6 IP 地址。

另一个任务，如我们之前提到的，是找到已启用 `trunk` 的接口：

```py
import re
sampletext="""
interface fa0/1
switchport mode trunk
no shut

interface fa0/0
no shut

interface fa1/0
switchport mode trunk
no shut

interface fa2/0
shut

interface fa2/1
switchport mode trunk
no shut

interface te3/1
switchport mode trunk
shut
"""

sampletext=sampletext.split("interface")
#check for interfaces that are in trunk mode
for chunk in sampletext:
    if ("mode trunk" in chunk):
        intname=re.search("(fa|te)\d+/\d+",chunk)
        print ("Trunk enabled on "+intname.group(0))

```

上述代码的输出如下：

![图片](img/09766937-eb10-44a3-9c56-c4fd4d5feac2.jpg)

在这里，我们需要找出分隔每个接口块的共同配置。正如我们在每个接口配置中看到的那样，单词`interface`分隔了每个接口的配置，因此我们使用`split`命令在接口工作上将配置分割成块。

一旦我们有了每个块，我们使用`(fa|te)\d+/\d+`的`re`模式来获取包含单词`trunk`的任何块上的接口名称。该模式表示任何以`fa`或`te`开头，后跟一个或多个数字和一个`\`，然后再次跟一个或多个数字的值将匹配。

同样，在相同的代码中，我们只想知道哪些配置为`trunk`的接口处于活动状态（未关闭）。以下是代码：

```py
import re
sampletext="""
interface fa0/1
switchport mode trunk
no shut

interface fa0/0
no shut

interface fa1/0
switchport mode trunk
no shut

interface fa2/0
shut

interface fa2/1
switchport mode trunk
no shut

interface te3/1
switchport mode trunk
shut
"""

sampletext=sampletext.split("interface")
#check for interfaces that are in trunk mode
for chunk in sampletext:
    if ("mode trunk" in chunk):
        if ("no shut" in chunk):
            intname=re.search("(fa|te)\d+/\d+",chunk)
            print ("Trunk enabled on "+intname.group(0))
```

上述代码的输出如下：

![图片](img/6cd711c9-5afc-47ef-ae62-6bf447e5fc36.jpg)

我们添加了一个额外条件，仅对那些除了`trunk`关键字外还有`no shut`的块进行操作。在这种情况下，我们只对满足这两个条件的块进行操作，在前面的示例中，`te3/1`不在列表中，因为它处于`shut`状态。

当验证任何 IP 配置时，我们可以解析配置，获取 IP 地址，验证每个 IP 地址（IPv4 或 IPv6），如果有任何不正确的值，则指出不正确的值。这有助于确保我们正在验证可能由于任何手动复制或粘贴操作而悄悄进入的 IP 地址。当然，这也意味着我们不会看到任何生产问题，因为配置已经使用此逻辑预先验证了正确性。

验证设备配置中任何给定的 IPv4 或 IPv6 地址的代码如下：

```py
import socket
import re

def validateipv4ip(address):
    try:
        socket.inet_aton(address)
    except socket.error:
        print ("wrong IPv4 IP",address)

def validateipv6ip(address):
    ### for IPv6 IP address validation
    try:
        socket.inet_pton(socket.AF_INET6,address)
    except socket.error:
        print ("wrong IPv6 IP", address)

sampletext="""
ip tacacs server 10.10.10.10
int fa0/1
ip address 25.25.25.298 255.255.255.255
no shut
ip name-server 100.100.100.200
int fa0/0
ipv6 address 2001:0db8:85a3:0000:0000:8a2e:0370:7334
ip logging host 90.90.91.92
int te0/2
ipv6 address 2602:306:78c5:6a40:421e:6813:d55:ce7f
no shut
exit

"""

sampletext=sampletext.split("\n")
for line in sampletext:
    if ("ipv6" in line):
        ipaddress=re.search("(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))",line)
        validateipv6ip(ipaddress.group(0))
    elif(re.search("\d+.\d+.\d+.\d+",line)):
        ipaddress=re.search("\d+.\d+.\d+.\d+",line)
        validateipv4ip(ipaddress.group(0))
```

上述代码的输出如下：

![图片](img/ad4c8a01-4998-4f15-86f5-626c6cbfa9db.jpg)

我们从`sampletext`的每一行中获取 IPv4 或 IPv6 IP 地址，然后将该信息解析到我们的 IP 验证函数中，如果有不正确的 IP 地址，它将打印出不正确的 IP 地址。

同样，我们可以通过创建特定的函数来验证配置的其他方面，并对任何给定的配置执行完整的健全性和验证检查。

# 设备操作系统升级

有时我们需要升级设备，如路由器、交换机和防火墙。在一个设备上执行升级很容易，但我们需要自动化来升级多个路由器。不同的设备有不同的升级 IOS 或 OS 映像的方式，自动化或脚本根据设备的不同而采用不同的方法。

以升级思科 IOS 路由器为例；需要执行两个基本步骤或任务：

1.  将相关的 OS 或 IOS 映像复制到`flash:`或`bootflash:`。

1.  更改配置以使用新映像重新加载路由器。

**任务 1：先决条件（复制相关的 OS 或 IOS 映像）：**

+   我们需要一个可以从路由器访问的 FTP 服务器，并且服务器上有我们在路由器上需要的 IOS 映像

+   我们需要映像、正确的 MD5 校验和以及映像大小以进行验证

任务 1 的示例代码如下：

```py
from netmiko import ConnectHandler
import time

def pushimage(imagename,cmd,myip,imgsize,md5sum=None):
    uname="cisco"
    passwd="cisco"
    print ("Now working on IP address: ",myip)
    device = ConnectHandler(device_type='cisco_ios', ip=myip, username=uname, password=passwd)
    outputx=device.send_command("dir | in Directory")
    outputx=outputx.split(" ")
    outputx=outputx[-1]
    outputx=outputx.replace("/","")
    precmds="file prompt quiet"
    postcmds="file prompt"
    xcheck=device.send_config_set(precmds)
    output = device.send_command_timing(cmd)
    flag=True
    devicex = ConnectHandler(device_type='cisco_ios', ip=myip, username=uname, password=passwd)
    outputx=devicex.send_command("dir")
    print (outputx)
    while (flag):
        time.sleep(30)
        outputx=devicex.send_command("dir | in "+imagename)
        print (outputx)
        if imgsize in outputx:
            print("Image copied with given size. Now validating md5")
            flag=False
        else:
            print (outputx)
        if (flag == False):
            cmd="verify /md5 "+imagename
            outputmd5=devicex.send_command(cmd,delay_factor=50)
        if (md5sum not in outputmd5):
            globalflag=True
            print ("Image copied but Md5 validation failed on ",myip)
        else:
            print ("Image copied and validated on ",myip)
    devicex.send_config_set(postcmds)
    devicex.disconnect()
    device.disconnect()

ipaddress="192.168.255.249"
imgname="c3745-adventerprisek9-mz.124-15.T14.bin"
imgsize="46509636"
md5sum="a696619869a972ec3a27742d38031b6a"
cmd="copy ftp://ftpuser:ftpuser@192.168.255.250/c3745-adventerprisek9-mz.124-15.T14.bin flash:"
pushimage(imgname,cmd,ipaddress,imgsize,md5sum)
```

这段代码将把 IOS 映像推送到路由器。`while`循环将继续监控代码复制的进度，直到目录中不满足特定的映像大小。当我们达到指定的映像大小时，脚本将移动到下一个动作，即验证 MD5 校验和。一旦 MD5 校验和被验证，它将打印出一个最终确认，表明 IOS 映像已复制并 MD5 验证。

我们可以在任何路由器上使用此功能，只需对映像名称、大小和不同映像集的 MD5 校验和进行一些调整。

这里需要注意的一个重要事项是`file prompt quiet`命令。在我们开始复制命令之前，需要执行此命令，因为它会抑制路由器中的任何确认提示。如果我们得到这些确认提示，处理所有提示将变得困难，从而增加了代码的复杂性。

通过添加此命令，我们抑制了确认，一旦代码复制完成，我们就将其启用到默认的文件提示状态。

**任务 2：将路由器的 bootvar 更改为新的 OS 映像**：

这是我们设置 Cisco 中的 bootvar，使其指向要加载的新 IOS 映像的地方：

```py
from netmiko import ConnectHandler
import time

uname="cisco"
passwd="cisco"
device = ConnectHandler(device_type='cisco_ios', ip="192.168.255.249", username=uname, password=passwd)
output=device.send_command("show run | in boot")
print ("Current config:")
print (output)
cmd="boot system flash:c3745-adventerprisek9-mz.124-15.T14.bin"
device.send_config_set(cmd)
print ("New config:")
output=device.send_command("show run | in boot")
print (output)
device.send_command("wr mem")
device.disconnect()
```

上述代码的输出如下：

![图片](img/82c715a9-ffb6-43f3-b9c3-90f3caa1a79e.jpg)

如我们所见，在这段代码中，我们使用新映像创建一个命令，并通过`send_config_set`方法将其发送到路由器。此方法在`config t`下执行命令。一旦完成，我们再次运行`show run | in boot`命令以获取新的输出，以验证 bootvar 现在是否指向新的 OS 映像。

如果一切正常，我们就运行`wr mem`来保存这个新配置。

一旦两个任务都完成，我们需要重新加载路由器以使更改生效。在重新加载之前，需要处理多个场景。可以直接使用`reload`命令作为任务 3 执行直接重新加载，但作为最佳实践，我们需要确保路由器上没有生产或实时流量，因为重新加载将中断当前的流量流。此外，建议登录到控制台以验证重新加载进度，并在重新加载失败时快速恢复。

# IPv4 到 IPv6 转换

将 IPv4 地址转换为 IPv6 地址有多种方法。在 Python 3 中，我们有内置的`ipaddress`模块：

```py
import ipaddress

def convertusingipaddress(ipv4address):
    print(ipaddress.IPv6Address('2002::' + ipv4address).compressed)

convertusingipaddress("10.10.10.10")
convertusingipaddress("192.168.100.1")
```

上述代码的输出如下：

![图片](img/4955c089-e699-458b-b816-1efcb3c6d5c1.jpg)

在`ipaddress`库中，有许多不同的方法或函数可供我们用于各种目的。文档和详细信息可以在[`docs.python.org/3/library/ipaddress.html.`](https://docs.python.org/3/library/ipaddress.html)找到。

# 站点部署

随着我们继续在多供应商环境中工作，有需求快速部署设备和配置以使特定地点上线运行。可以部署多种技术进行地点部署，这涉及到一组标准设备连接到标准端口，每个设备上都有标准的 IOS 或代码镜像，准备上架和供电。为了确定特定地点的标准**库存单位**（**SKU**），我们可以将其划分为 T 恤尺寸。在规划阶段，我们可以根据某些参数创建 T 恤尺寸，例如使用情况、负载和冗余。

在最低级别，比如说**超小尺寸**（**XS**）可以只有一个路由器和一台交换机，路由器终止于互联网链路。交换机连接到路由器的`FastEthernet 0/1`（100 Mbps）或`Gi0/1`（1000 Mbps）端口，最终用户直接将设备插入交换机以获取访问权限。基于这个 XS SKU（或 T 恤尺寸），我们可以确定每个路由器和交换机的硬件供应商，例如 Cisco、DLink 或其他网络设备提供商。接下来，当我们确定了硬件供应商后，我们开始生成配置模板。

配置模板通常基于两个标准：

+   设备的作用

+   硬件供应商

在相同的 XS 尺寸中，比如说我们有一个作为路由器的 Cisco 3064（运行 Cisco NXOS 的 Cisco Nexus）和一个在交换层中的 Alcatel 交换机。既然我们已经确定了硬件供应商和每个设备的作用，我们就可以轻松地创建模板配置。

如前所述，一旦我们有了标准硬件，我们还需要确保端口是标准的（例如，交换机的上行链路将从端口`Gi1/0`连接到路由器的`Gi1/1`）。这将帮助我们确保我们创建了一个几乎完整的模板，同时考虑了接口配置。

模板包含一个基本的配置，某些值将在以后确定。这是一个非常通用的布局，我们可以用各种输入的值来填充，例如识别可用的 IP 地址、序列中的下一个主机名，以及作为标准配置需要放置哪种路由：

![图片](img/11e0779b-374e-4781-9052-5e450fbd3b77.jpg)

如前图所示，中央 Python 脚本正在调用不同的函数（初始输入为供应商和基于角色的标准模板），并获取特定的信息，如可用的 IP 地址、下一个可用的主机名（如`rtr01`或`rtr05`）以及路由信息（如**增强型内部网关路由协议**（**EIGRP**）在网络上广播子网`10.10.10.0/255`）。这些输入以及更多（根据需求而定）都是独立的 Python 函数，模板根据 Python 函数的返回值进行更改。

例如，我们需要使用 Python 从 SQL 表中获取 IP 地址，其中 IP 地址显示为未分配（我们将使用 Python 中的`MySQLdb`库来完成此操作）：

```py
import MySQLdb

def getfreeip():
    # Open database connection
    db = MySQLdb.connect("testserver","user","pwd","networktable" )
    cursor = db.cursor()

    sql = "select top 1 from freeipaddress where isfree='true'"
    try:
       # Execute the SQL command
       cursor.execute(sql)
       # Fetch all the rows in a list of lists.
       results = cursor.fetchall()
       for eachrow in results:
          freeip=eachrow[0]
          return (freeip)
    except:
       print "Error: unable to fetch data"
       return "error in accessing table"
    db.close()

print (getfreeip())
```

这将从 SQL 表中返回一个空闲 IP 地址，我们可以将其调用到其他函数中以生成我们的配置。当然，一旦给出这个，我们还需要更新表以确保我们将记录中的`isfree`值设置为`false`，这样对函数的新调用将确保我们获取 SQL 表中的下一个空闲 IP 地址。

将所有这些结合起来，我们可以从多个表中获取详细信息，甚至调用特定工具的 API 以获取专门的信息，并且，将这些函数或方法的返回值作为输入，模板将使用这些返回值替换模板中指定的变量。一旦模板值填写完毕，输出将是最终生成的配置，该配置已准备好部署到路由器/网络设备上。

通过根据 T 恤尺寸规范创建此基线自动化，脚本可以再次被调用，以包含新的一组设备，例如负载均衡器、多个路由器，以及根据 T 恤尺寸和复杂度，不同角色的每个不同路由器。

在生成最终配置模板之后，下一步是将配置应用到路由器上。始终建议使用控制台执行此功能。一旦我们有了基本的配置以获取对设备的 SSH/Telnet 访问权限，我们就可以在控制台上保持会话打开，同时将剩余的配置推送到各个设备上。Netmiko 可用于此目的，目的是使用新生成的模板推送所有配置。

假设电缆已正确连接，符合标准，下一步是验证流量和配置。为此，我们再次依赖 Netmiko 来获取路由、日志以及诸如接口计数器和 BGP 路由器表等特定信息。

此外，我们还可以对 SNMP 输入进行工作以验证每个设备的当前运行健康状态。设备有时在测试条件下表现良好，但一旦生产或实时流量在其数据平面上，它可能会在硬件资源上激增，导致延迟或数据包丢失。SNMP 统计信息将给我们一个清晰的每个设备的健康状况，例如 CPU 使用率、内存使用率，甚至某些设备的当前温度及其模块，以显示 SKU 或 T 恤尺寸站点的整体健康状况。

# 办公/数据中心搬迁

有时候，我们需要将站点搬迁、关闭或迁移到不同的位置。这涉及到大量的预检查、预验证，并确保网络 PoD 的相同设置在另一个位置是活跃的。

在一个多厂商环境中，随着 T 恤尺码的增加，SKU 数量的增加，手动跟踪所有活跃会话、流量、当前接口状态和特定路由是困难的。使用 Python，我们可以创建一个自动化的基本清单，并确保在搬迁后，相同的清单可以作为后验证清单。

例如，我们创建一个基本的脚本，询问是否需要执行预检查/后检查，并将结果保存到名为`pre-check`和`post-check`的文件中：

```py
from netmiko import ConnectHandler
import time

def getoutput(cmd):
    uname="cisco"
    passwd="cisco"
    device = ConnectHandler(device_type='cisco_ios', ip="192.168.255.249", username=uname, password=passwd)
    output=device.send_command(cmd)
    return (output)

checkprepost=input("Do you want a pre or post check [pre|post]: ")
checkprepost=checkprepost.lower()
if ("pre" in checkprepost ):
    fname="precheck.txt"
else:
    fname="postcheck.txt"

file=open(fname,"w")
file.write(getoutput("show ip route"))
file.write("\n")
file.write(getoutput("show clock"))
file.write("\n")
file.write(getoutput("show ip int brief"))
file.write("\n")

print ("File write completed",fname)

file.close()
```

上述代码的输出如下：

![](img/e726fcd4-c421-400b-94c8-b54c2c0a1acf.jpg)

假设`precheck.txt`文件在迁移或搬迁之前在多个设备所在的现场被采集，而`postcheck.txt`文件是在搬迁之后在现场采集的。现在让我们编写一个简单的脚本，比较这两个文件并打印出差异。

Python 有一个名为`difflib`的库来执行这个任务：

```py
import difflib

file1 = "precheck.txt"
file2 = "postcheck.txt"

diff = difflib.ndiff(open(file1).readlines(),open(file2).readlines())
print (''.join(diff),)
```

上述代码的输出如下：

![](img/5657105f-a8e3-4959-ba83-bc03896f61a7.jpg)

如我们在`precheck.txt`和`postcheck.txt`中看到的，文件是逐行比较的。任何没有变化的内容都按原样显示，但任何有差异的内容都会用`-`或`+`来表示。行首的`-`符号指定了该特定行来自第一个文件（在我们的例子中是`precheck.txt`），而`+`符号表示相同的行已经在新的文件（即`postcheck.txt`）中输出。使用这种方法，我们可以快速验证`precheck`和`postcheck`之间的差异，并在迁移或搬迁后修复相关的问题。

有时候我们希望自动运行脚本以备份当前路由器的配置。在这种情况下，让我们假设搬迁计划在明天进行。在开始任何活动之前，我们想要确保我们有当前设备配置的备份。

一个简单的脚本就能解决问题：

```py
from netmiko import ConnectHandler

def takebackup(cmd,rname):
    uname="cisco"
    passwd="cisco"
    device = ConnectHandler(device_type='cisco_ios', ip=rname, username=uname, password=passwd)
    output=device.send_command(cmd)
    fname=rname+".txt"
    file=open(fname,"w")
    file.write(output)
    file.close()

# assuming we have two routers in network 
devices="rtr1,rtr2"
devices=devices.split(",")

for device in devices:
    takebackup("show run",device)
```

脚本将逐个解析设备列表中的每个设备，执行`show run`命令，并将其保存到给定的文件名中（文件名与给定的设备名或 IP 相同）。然而，下一个问题是如何确保它在预定的时间运行。在 Linux 中，我们有 cron 作业可以设置用于此目的，Windows 也有任务计划程序。

以下示例显示了在任务计划程序中创建任务的基本过程：

1.  在 Windows 中打开任务计划程序：

>![](img/84456129-2387-4ac2-a8bf-c1f53df878ae.jpg)

1.  在任务计划程序的右侧单击“创建基本任务”：

![](img/00a1e53b-6b7a-44ec-9cb1-f6d363d092cc.jpg)

1.  点击“下一步”并选择任务的频率：

![](img/9c3fc115-aea5-45be-8b1b-7c4c2bc786fc.jpg)

1.  点击下一步，选择时间，然后再次点击下一步。转到启动程序。在这个时候，您需要添加以下屏幕截图中显示的详细信息。我们必须在程序/脚本窗口中提供`python.exe`的完整路径，并在添加参数（可选）部分，用双引号括起 Python 脚本（带有`.py`扩展名）的完整路径：

![图片](img/49199f49-4c8e-436d-8156-a0433d3f1292.jpg)

1.  在最后一页，点击完成以提交更改并创建任务：

![图片](img/b23de7e1-ff39-460f-9d89-55bd5f3de113.jpg)

完成此操作后，您可以通过右键单击创建的任务并点击运行选项来手动运行它。如果任务成功，它将在任务计划程序窗口中返回相同的结果。如果一切正常，任务将在任务计划程序中指定的时间自动运行。

这也可以作为一个服务运行，并且可以按照固定的间隔，例如每天和每小时，根据我们希望在环境中运行脚本的频率来决定。

这些计划好的备份有时可以作为基线，也可以作为已知良好配置的场景。

# 交换机的自带设备（BYOD）配置

随着我们的网络变得更加可扩展，我们需要拓宽我们当前的网络架构设计，以纳入更好的交换机和路由器来满足需求。有时我们可能会有特定的需求，需要将特定的硬件添加到我们的网络中以满足这些需求。

另一个可能的需求可能是降低成本同时提高可扩展性。在这种情况下，我们需要添加不同供应商的交换机来满足需求。也可能会有对某个办公室或地点的非常具体的需求。在这种情况下，我们需要添加不同供应商的硬件来满足一些特定的要求。

我们刚才观察到的所有场景都有一个共同点。为了满足需求或特定要求，我们不能仅仅依赖网络上的单一供应商解决方案。我们需要一个随机集合的设备来确保满足特定的要求。这就是我们引入 BYOD（Bring Your Own Device，自带设备）概念的地方。BYOD 是一个新的标准，它接纳新的设计、硬件和架构，以适应我们当前的 SKU 或设计。它可能简单到只是通过无线方式将一部新手机添加到我们的企业网络，或者稍微复杂一些，比如将特定供应商的硬件添加到网络中。

架构师需要确保他们有良好的预测需求的方法，并了解当前的网络设计或硬件是否能够满足这些需求。在任何情况下，初始设计都需要确保当前技术支持跨厂商平台。在这个设计方法中存在一些冲突。例如，某些厂商，如思科，拥有邻居发现协议，**思科特定协议** (**CDP**)，它可以发现当前设备的正确思科设备作为邻居。然而，为了确保 CDP 能够发现并显示正确的信息，每个设备都需要是思科的。另一方面，我们有 **链路层发现协议** (**LLDP**)，它与 CDP 几乎相同，但它是开源的，因此许多其他厂商，包括思科，也有选择使用 LLDP 而不是 CDP 进行发现的机会。现在，思科 CDP 是一个思科特定协议；思科确保某些参数只能通过 CDP 交换或发现，因此参与 CDP 的每个设备都必须是思科的设备。

LLDP 作为开源协议，仅限于开放标准或 **互联网工程任务组** (**IETF**) 框架中的参数，并且所有支持 LLDP 的厂商仅遵守这些开放标准以实现跨平台和硬件兼容性。这也导致一些参与厂商不会发送或发现专门为该厂商设计的参数（例如思科）。回到之前提到的观点，在这种情况下，从第一天开始的设计架构需要确保多厂商或开源标准仅用于基本设计或架构。LLDP 的一个类似例子是使用开放标准如 OSPF 或 BGP 而不是 EIGRP，后者仅适用于思科设备。

如前所述，我们需要定义特定的角色，并根据我们在当前设计中引入的设备或硬件创建硬件或厂商模板，这些模板应基于我们作为 BYOD 策略引入的设备。保持开放标准的方法，我们需要确保创建的模板是通用的，并且可以稍后将其特定厂商的配置引入到设备中。

SNMP 是一种强大的协议，它有助于无缝地管理许多跨厂商或 BYOD 策略。通过基本配置启用 SNMP 并使用特定的只读社区字符串，我们可以用 Python 编写快速脚本以从 BYOD 设备获取基本信息。以一个例子来说明，假设我们需要知道以下两个设备的类型和厂商：

```py
from pysnmp.hlapi import *

def finddevices(ip):
    errorIndication, errorStatus, errorIndex, varBinds = next(
        getCmd(SnmpEngine(),
               CommunityData('public', mpModel=0),
               UdpTransportTarget((ip, 161)),
               ContextData(),
               ObjectType(ObjectIdentity('SNMPv2-MIB', 'sysDescr', 0)))
    )

    if errorIndication:
        print(errorIndication)
    elif errorStatus:
        print('%s at %s' % (errorStatus.prettyPrint(),
                            errorIndex and varBinds[int(errorIndex) - 1][0] or '?'))
    else:
        for varBind in varBinds:
            print(' = '.join([x.prettyPrint() for x in varBind]))

ipaddress="192.168.255.248,192.168.255.249"
ipaddress=ipaddress.split(",")
for ip in ipaddress:
    print (ip)
    finddevices(ip)
    print ("\n")    
```

之前代码的输出如下：

![](img/632e5f8d-0b08-408f-9861-148edd3bfc17.jpg)

如前述输出所示，现在我们只需要知道 IP 地址和开放标准 SNMP OID，即 SNMPv2-MIB。`sysDescr`将为两个设备提供输出。在这种情况下，我们可以看到一个是 Cisco 3600，另一个是 Cisco 3700。根据返回的信息，我们可以进行配置。

根据 BYOD 策略，需要执行各种其他任务。如果您想将一部手机连接到您的网络，所需的就是连接到企业网络，以及可以推送到设备以检查各种合规性检查（如操作系统和防病毒软件）的策略。根据这些结果，查询的设备可以被放置在另一个 VLAN 中，可以称为隔离 VLAN，它具有非常有限的可访问性，或者企业 VLAN，它可以完全访问企业资源。

以类似的方式，作为交换机 BYOD 策略的一部分，我们需要执行某些检查以确保设备适合成为我们网络设计的一部分。是的，我们需要为各种类型的设备保持开放政策，但需要在设备能够符合 BYOD 接受资格的松散耦合框架下。

让我们看看一个确保设备足够兼容以成为 BYOD 框架一部分的例子。核心要求是 Cisco 交换机，并且它应该有一个`FastEthernet0/0`作为其接口之一：

```py
from pysnmp.hlapi import *
from pysnmp.entity.rfc3413.oneliner import cmdgen

cmdGen = cmdgen.CommandGenerator()

def validateinterface(ip):
    errorIndication, errorStatus, errorIndex, varBindTable = cmdGen.bulkCmd(
        cmdgen.CommunityData('public'),
        cmdgen.UdpTransportTarget((ip, 161)),
        0,25,
        '1.3.6.1.2.1.2.2.1.2',
        '1.3.6.1.2.1.2.2.1.7'
    )
    flag=False
    # Check for errors and print out results
    if errorIndication:
        print(errorIndication)
    else:
        if errorStatus:
            print('%s at %s' % (
                errorStatus.prettyPrint(),
                errorIndex and varBindTable[-1][int(errorIndex)-1] or '?'
                )
            )
        else:
            for varBindTableRow in varBindTable:
                for name, val in varBindTableRow:
                    if ("FastEthernet0/0" in val.prettyPrint()):
                        flag=True
    if (flag):
        return True
    else:
        return False

def finddevice(ip):
    errorIndication, errorStatus, errorIndex, varBinds = next(
        getCmd(SnmpEngine(),
               CommunityData('public', mpModel=0),
               UdpTransportTarget((ip, 161)),
               ContextData(),
               ObjectType(ObjectIdentity('SNMPv2-MIB', 'sysDescr', 0)))
    )

    if errorIndication:
        print(errorIndication)
    elif errorStatus:
        print('%s at %s' % (errorStatus.prettyPrint(),
                            errorIndex and varBinds[int(errorIndex) - 1][0] or '?'))
    else:
        for varBind in varBinds:
            if ("Cisco" in varBind.prettyPrint()):
                return True
    return False

mybyoddevices="192.168.255.249,192.168.255.248"
mybyoddevices=mybyoddevices.split(",")
for ip in mybyoddevices:
    getvendorvalidation=False
    getipvalidation=False
    print ("Validating IP",ip)
    getipvalidation=validateinterface(ip)
    print ("Interface has fastethernet0/0 :",getipvalidation)
    getvendorvalidation=finddevice(ip)
    print ("Device is of vendor Cisco:",getvendorvalidation)
    if getipvalidation and getvendorvalidation:
        print ("Device "+ip+" has passed all validations and eligible for BYOD")
        print ("\n\n")
    else:
        print ("Device "+ip+" has failed validations and NOT eligible for BYOD")
        print ("\n\n")  
```

之前代码的输出如下：

![图片](img/addb3b92-1f37-493f-8e27-ade47fdfe9a7.jpg)

我们解析了两个设备，并使用开源 SNMP 获取厂商和接口信息。接下来，我们进行验证，并根据我们的条件返回`True`或`False`。所有检查条件都为`True`将导致设备作为 BYOD 被接受。

让我们稍微改变一下规则。假设任何具有以太网接口的设备都不符合 BYOD 资格：

```py
from pysnmp.hlapi import *
from pysnmp.entity.rfc3413.oneliner import cmdgen

cmdGen = cmdgen.CommandGenerator()

def validateinterface(ip):
    errorIndication, errorStatus, errorIndex, varBindTable = cmdGen.bulkCmd(
        cmdgen.CommunityData('public'),
        cmdgen.UdpTransportTarget((ip, 161)),
        0,25,
        '1.3.6.1.2.1.2.2.1.2',
        '1.3.6.1.2.1.2.2.1.7'
    )
    flag=False
    # Check for errors and print out results
    if errorIndication:
        print(errorIndication)
    else:
        if errorStatus:
            print('%s at %s' % (
                errorStatus.prettyPrint(),
                errorIndex and varBindTable[-1][int(errorIndex)-1] or '?'
                )
            )
        else:
            for varBindTableRow in varBindTable:
                for name, val in varBindTableRow:
                    if ((val.prettyPrint()).startswith("Ethernet")):
                        return False
                    if ("FastEthernet0/0" in val.prettyPrint()):
                        flag=True
    if (flag):
        return True
    else:
        return False

def finddevice(ip):
    errorIndication, errorStatus, errorIndex, varBinds = next(
        getCmd(SnmpEngine(),
               CommunityData('public', mpModel=0),
               UdpTransportTarget((ip, 161)),
               ContextData(),
               ObjectType(ObjectIdentity('SNMPv2-MIB', 'sysDescr', 0)))
    )

    if errorIndication:
        print(errorIndication)
    elif errorStatus:
        print('%s at %s' % (errorStatus.prettyPrint(),
                            errorIndex and varBinds[int(errorIndex) - 1][0] or '?'))
    else:
        for varBind in varBinds:
            if ("Cisco" in varBind.prettyPrint()):
                return True
    return False

mybyoddevices="192.168.255.249,192.168.255.248"
mybyoddevices=mybyoddevices.split(",")
for ip in mybyoddevices:
    getvendorvalidation=False
    getipvalidation=False
    print ("Validating IP",ip)
    getipvalidation=validateinterface(ip)
    print ("Device has No Ethernet only Interface(s) :",getipvalidation)
    getvendorvalidation=finddevice(ip)
    print ("Device is of vendor Cisco:",getvendorvalidation)
    if getipvalidation and getvendorvalidation:
        print ("Device "+ip+" has passed all validations and eligible for BYOD")
        print ("\n\n")
    else:
        print ("Device "+ip+" has failed validations and NOT eligible for BYOD")
        print ("\n\n")    
```

之前代码的输出如下：

![图片](img/b8fbf329-19c7-45c1-aa9b-96f25d9ed2b4.jpg)

如本例所示，我们对以`Ethernet`关键字开头的任何接口进行了验证。使用`string.startswith("given string")`函数来评估任何给定的字符串是否是其正在比较的字符串的开头。在我们的情况下，具有 IP 地址`192.168.255.248`的设备有一个仅以太网接口，该接口在验证仅以太网接口时返回了`True`值。由于这被认为是我们验证的失败，因此返回了`False`，并且脚本将其称为 BYOD 接受失败，因为这一特定条件失败了。

以类似的方式，我们可以对任何数量的设备进行验证和多项检查，并确保只有通过 BYOD 框架检查的设备被接受到网络中。

# 摘要

在本章中，我们研究了各种复杂场景，以了解新的站点迁移和验证是如何进行的。我们还探讨了多厂商配置的概念，创建模板，为设备生成配置，以及 IPv4 到 IPv6 迁移技术。

我们专注于特定数据的专项提取，例如 IP 地址，以及该数据的验证，以及数据失败或接受的条件。此外，还讨论了站点推广和 BYOD 策略，以及最佳实践，如 T 恤尺码和 BYOD 条件的验证。

在下一章中，我们将进一步深入，介绍 Web 启用框架。这将帮助我们创建自己的 API，并创建基于浏览器的 Python 脚本，这些脚本可以从任何地方执行。
