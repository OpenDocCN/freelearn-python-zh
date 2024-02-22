# 第四章：使用 Python 管理网络设备

现在我们对如何在不同操作系统中使用和安装 Python 以及如何使用 EVE-NG 构建网络拓扑有了相当的了解。在本章中，我们将发现如何利用今天用于自动化各种网络任务的许多网络自动化库。Python 可以在许多层面与网络设备进行交互。

首先，它可以通过套接字编程和 `socket` 模块处理低级层，这些模块作为运行 Python 的操作系统和网络设备之间的低级网络接口。此外，Python 模块通过 telnet、SSH 和 API 提供更高级的交互。在本章中，我们将深入探讨如何使用 Python 建立远程连接并使用 telnet 和 SSH 模块在远程设备上执行命令。

以下主题将被涵盖：

+   使用 Python 进行 telnet 到设备

+   Python 和 SSH

+   使用 `netaddr` 处理 IP 地址和网络

+   网络自动化示例用例

# 技术要求

以下工具应安装并在您的环境中可用：

+   Python 2.7.1x

+   PyCharm 社区版或专业版

+   EVE-NG 拓扑；请参阅第三章，*设置网络实验室环境*，了解如何安装和配置模拟器

您可以在以下 GitHub URL 找到本章中开发的完整脚本：[`github.com/TheNetworker/EnterpriseAutomation.git`](https://github.com/TheNetworker/EnterpriseAutomation.git)。

# Python 和 SSH

与 telnet 不同，SSH 提供了一个安全的通道，用于在客户端和服务器之间交换数据。在客户端和设备之间创建的隧道使用不同的安全机制进行加密，使得任何人都很难解密通信。对于需要安全管理网络节点的网络工程师来说，SSH 协议是首选。

Python 可以使用 SSH 协议与网络设备通信，利用一个名为 **Paramiko** 的流行库，该库支持认证、密钥处理（DSA、RSA、ECDSA 和 ED25519）以及其他 SSH 功能，如 `proxy` 命令和 SFTP。

# Paramiko 模块

Python 中最广泛使用的 SSH 模块称为 `Paramiko`，正如 GitHub 官方页面所说，Paramiko 的名称是 "paranoid" 和 "friend" 这两个世界的组合。该模块本身是使用 Python 编写和开发的，尽管一些核心功能如加密依赖于 C 语言。您可以在官方 GitHub 链接中了解更多有关贡献者和模块历史的信息：[`github.com/paramiko/paramiko`](https://github.com/paramiko/paramiko)。

# 模块安装

打开 Windows cmd 或 Linux shell 并执行以下命令，从 PyPI 下载最新的 `paramiko` 模块。它将下载附加的依赖包，如 `cyrptography`、`ipaddress` 和 `six`，并在您的计算机上安装它们：

```py
pip install paramiko
```

![](img/00066.jpeg)

您可以通过进入 Python shell 并导入 `paramiko` 模块来验证安装是否成功，如下面的屏幕截图所示。Python 应该成功导入它而不打印任何错误：

![](img/00067.jpeg)

# SSH 到网络设备

与每个 Python 模块一样，我们首先需要将其导入到我们的 Python 脚本中，然后我们将通过继承 `SSHClient()` 来创建一个 SSH 客户端。之后，我们将配置 Paramiko 自动添加任何未知的主机密钥并信任您和服务器之间的连接。然后，我们将使用 `connect` 函数并提供远程主机凭据：

```py
#!/usr/bin/python __author__ = "Bassim Aly" __EMAIL__ = "basim.alyy@gmail.com"   import paramiko
import time
Channel = paramiko.SSHClient()  Channel.set_missing_host_key_policy(paramiko.AutoAddPolicy()) Channel.connect(hostname="10.10.88.112", username='admin', password='access123', look_for_keys=False,allow_agent=False)   shell = Channel.invoke_shell()  
```

`AutoAddPolicy()` 是可以在 `set_missing_host_key_policy()` 函数内使用的策略之一。在实验室环境中，它是首选且可接受的。然而，在生产环境中，我们应该使用更严格的策略，比如 `WarningPolicy()` 或 `RejectPolicy()`。

最后，`invoke_shell()`将启动与我们的 SSH 服务器的交互式 shell 会话。您可以向其提供附加参数，如终端类型、宽度和高度。

Paramiko 连接参数：

+   `Look_For_Keys`：默认为`True`，它将强制 Paramiko 使用密钥对身份验证，用户使用私钥和公钥对来对网络设备进行身份验证。在我们的情况下，我们将其设置为`False`，因为我们将使用密码身份验证。

+   `allow_agent paramiko`：它可以连接到本地 SSH 代理操作系统。这在使用密钥时是必要的；在这种情况下，由于使用登录/密码进行身份验证，我们将禁用它。

最后一步是向设备终端发送一系列命令，如`show ip int b`和`show arp`，并将输出返回到我们的 Python shell：

```py
shell.send("enable\n") shell.send("access123\n") shell.send("terminal length 0\n") shell.send("show ip int b\n") shell.send("show arp \n") time.sleep(2) print shell.recv(5000)  Channel.close()
```

脚本输出为：

![](img/00068.jpeg)当您需要在远程设备上执行需要很长时间的命令时，最好使用`time.sleep()`来强制 Python 等待一段时间，直到设备生成输出并将其发送回 Python。否则，Python 可能会向用户返回空白输出。

# Netmiko 模块

`netmiko`模块是 paramiko 的增强版本，专门针对网络设备。虽然 paramiko 旨在处理对设备的 SSH 连接，并检查设备是服务器、打印机还是网络设备，但 Netmiko 是专为网络设备设计的，并更有效地处理 SSH 连接。此外，Netmiko 支持广泛的供应商和平台。

Netmiko 被认为是 paramiko 的包装器，并通过许多附加增强功能扩展了其功能，例如直接访问供应商启用模式，读取配置文件并将其推送到设备，登录期间禁用分页，并在每个命令后默认发送回车符`"\n"`。

# 供应商支持

Netmiko 支持许多供应商，并定期向受支持的列表中添加新供应商。以下是受支持的供应商列表，分为三组：定期测试，有限测试和实验性。您可以在模块 GitHub 页面上找到列表[`github.com/ktbyers/netmiko#supports`](https://github.com/ktbyers/netmiko#supports)。

以下截图显示了定期测试类别下受支持供应商的数量：

![](img/00069.jpeg)

以下截图显示了有限测试类别下受支持供应商的数量：

![](img/00070.jpeg)

以下截图显示了实验类别下受支持供应商的数量：

![](img/00071.jpeg)

# 安装和验证

要安装`netmiko`，打开 Windows cmd 或 Linux shell，并执行以下命令从 PyPI 获取最新包：

```py
pip install netmiko
```

![](img/00072.jpeg)

然后从 Python shell 导入 netmiko，以确保模块已正确安装到 Python site-packages 中：

```py
$python
>>>import netmiko
```

# 使用 netmiko 进行 SSH

现在是时候利用 netmiko 并看到它在 SSH 到网络设备并执行命令时的强大功能。默认情况下，netmiko 在会话建立期间在后台处理许多操作，例如添加未知的 SSH 密钥主机，设置终端类型、宽度和高度，并在需要时访问启用模式，然后通过运行特定于供应商的命令来禁用分页。您需要首先以字典格式定义设备，并提供五个强制键：

```py
R1 = {
  'device_type': 'cisco_ios',
  'ip': '10.10.88.110',
  'username': 'admin',
  'password': 'access123',
  'secret': 'access123',  }
```

第一个参数是`device_type`，用于定义平台供应商，以便执行正确的命令。然后，我们需要 SSH 的`ip`地址。如果已经通过 DNS 解析了设备主机名，这个参数可以是设备主机名，或者只是 IP 地址。然后我们提供`username`，`password`，和`secret`中的启用模式密码。请注意，您可以使用`getpass()`模块隐藏密码，并且只在脚本执行期间提示密码。

虽然变量内的键的顺序并不重要，但键的名称应与前面的示例中提供的完全相同，以便 netmiko 正确解析字典并开始建立与设备的连接。

接下来，我们将从 netmiko 模块中导入`ConnectHandler`函数，并给它定义的字典以开始连接。由于我们所有的设备都配置了启用模式密码，我们需要通过提供`.enable()`来访问启用模式到创建的连接。我们将使用`.send_command()`在路由器终端上执行命令，该命令将执行命令并将设备输出返回到变量中：

```py
from netmiko import ConnectHandler
connection = ConnectHandler(**R1)  connection.enable()  output = connection.send_command("show ip int b") print output
```

脚本输出为：

![](img/00073.jpeg)

请注意输出已经从设备提示和我们在设备上执行的命令中清除。默认情况下，Netmiko 会替换它们并生成一个经过清理的输出，可以通过正则表达式进行处理，我们将在下一章中看到。

如果您需要禁用此行为，并希望在返回的输出中看到设备提示和执行的命令，则需要为`.send_command()`函数提供额外的标志：

```py
output = connection.send_command("show ip int b",strip_command=False,strip_prompt=False) 
```

`strip_command=False`和`strip_prompt=False`标志告诉 netmiko 保留提示和命令，不要替换它们。它们默认为`True`，如果需要，可以切换它们：

![](img/00074.jpeg)

# 使用 netmiko 配置设备

Netmiko 可以用于通过 SSH 配置远程设备。它通过使用`.config`方法访问配置模式，然后应用以`list`格式给出的配置来实现这一点。列表本身可以在 Python 脚本中提供，也可以从文件中读取，然后使用`readlines()`方法转换为列表：

```py
from netmiko import ConnectHandler

SW2 = {
  'device_type': 'cisco_ios',
  'ip': '10.10.88.112',
  'username': 'admin',
  'password': 'access123',
  'secret': 'access123', }   core_sw_config = ["int range gig0/1 - 2","switchport trunk encapsulation dot1q",
  "switchport mode trunk","switchport trunk allowed vlan 1,2"]     print "########## Connecting to Device {0} ############".format(SW2['ip']) net_connect = ConnectHandler(**SW2) net_connect.enable()    print "***** Sending Configuration to Device *****" net_connect.send_config_set(core_sw_config) 
```

在上一个脚本中，我们做了与之前连接到 SW2 并进入启用模式相同的事情，但这次我们利用了另一个 netmiko 方法，称为`send_config_set()`，它以列表格式接收配置并访问设备配置模式并开始应用。我们有一个简单的配置，修改了`gig0/1`和`gig0/2`，并对它们应用了干线配置。您可以通过在设备上运行`show run`命令来检查命令是否成功执行；您应该会得到类似以下的输出：

![](img/00075.jpeg)

# netmiko 中的异常处理

当设计 Python 脚本时，我们假设设备正在运行，并且用户已经提供了正确的凭据，这并不总是情况。有时 Python 和远程设备之间存在网络连接问题，或者用户输入了错误的凭据。通常，如果发生这种情况，Python 会抛出异常并退出，这并不是最佳解决方案。

netmiko 中的异常处理模块`netmiko.ssh_exception`提供了一些可以处理这种情况的异常类。第一个是`AuthenticationException`，将捕获远程设备中的身份验证错误。第二个类是`NetMikoTimeoutException`，将捕获 netmiko 和设备之间的超时或任何连接问题。我们需要做的是用 try-except 子句包装我们的 ConnectHandler()方法，并捕获超时和身份验证异常：

```py

from netmiko import ConnectHandler
from netmiko.ssh_exception import AuthenticationException, NetMikoTimeoutException

device = {
  'device_type': 'cisco_ios',
  'ip': '10.10.88.112',
  'username': 'admin',
  'password': 'access123',
  'secret': 'access123', }     print "########## Connecting to Device {0} ############".format(device['ip']) try:
  net_connect = ConnectHandler(**device)
  net_connect.enable()    print "***** show ip configuration of Device *****"
  output = net_connect.send_command("show ip int b")
  print output

    net_connect.disconnect()   except NetMikoTimeoutException:
  print "=========== SOMETHING WRONG HAPPEN WITH {0} ============".format(device['ip'])   except AuthenticationException:
  print "========= Authentication Failed with {0} ============".format(device['ip'])   except Exception as unknown_error:
  print "============ SOMETHING UNKNOWN HAPPEN WITH {0} ============"
```

# 设备自动检测

Netmiko 提供了一种可以猜测设备类型并检测它的机制。它使用 SNMP 发现 OID 和在远程控制台上执行几个 show 命令的组合来检测路由器操作系统和类型，基于输出字符串。然后 netmiko 将加载适当的驱动程序到`ConnectHandler()`类中：

```py
#!/usr/local/bin/python __author__ = "Bassim Aly" __EMAIL__ = "basim.alyy@gmail.com"     from netmiko import SSHDetect, Netmiko

device = {
  'device_type': 'autodetect',
  'host': '10.10.88.110',
  'username': 'admin',
  'password': "access123", }   detect_device = SSHDetect(**device) device_type = detect_device.autodetect() print(device_type)  print(detect_device.potential_matches)    device['device_type'] = device_type
connection = Netmiko(**device)
```

在上一个脚本中：

+   设备字典中的`device_type`将是`autodetect`，这将告诉`netmiko`等待并不加载驱动程序，直到 netmiko 猜测到它。

+   然后我们指示 netmiko 使用`SSHDetect()`类执行设备检测。该类将使用 SSH 连接到设备，并执行一些发现命令来定义操作系统类型。返回的结果将是一个字典，并且最佳匹配将使用`autodetect()`函数分配给`device_type`变量。

+   您可以通过打印`potential_matches`来查看所有匹配的结果。

+   现在我们可以更新设备字典并为其分配新的`device_type`。

# 使用 Python 中的 telnet 协议

Telnet 是 TCP/IP 协议栈中可用的最古老的协议之一。它主要用于在服务器和客户端之间建立的连接上交换数据。它在服务器上使用 TCP 端口`23`来监听来自客户端的传入连接。

在我们的情况下，我们将创建一个充当 telnet 客户端的 Python 脚本，拓扑中的其他路由器和交换机将充当 telnet 服务器。Python 自带了一个名为`telnetlib`的库，因此我们不需要安装它。

通过从`telnetlib`模块中可用的`Telnet()`类实例化客户端对象后，我们可以使用`telnetlib`中可用的两个重要函数，即`read_until()`（用于读取输出）和`write()`（用于在远程设备上写入）。这两个函数用于与创建的通道进行交互，无论是写入还是读取返回的输出。

另外，重要的是要注意，使用`read_until()`读取通道将清除缓冲区，并且数据将不可用于进一步的读取。因此，如果您读取重要数据并且稍后将对其进行处理和处理，那么您需要在继续脚本之前将其保存为变量。

Telnet 数据以明文格式发送，因此您的凭据和密码可能会被执行中间人攻击的任何人捕获和查看。一些服务提供商和企业仍然使用它，并将其与 VPN 和 radius/tacacs 协议集成，以提供轻量级和安全的访问。

按照以下步骤来理解整个脚本：

1.  我们将在 Python 脚本中导入`telnetlib`模块，并将用户名和密码定义为变量，如下面的代码片段所示：

```py
import telnetlib
username = "admin" password = "access123" enable_password = "access123"
```

1.  我们将定义一个变量，用于与远程主机建立连接。请注意，在建立连接期间我们不会提供用户名或密码；我们只会提供远程主机的 IP 地址：

```py
cnx = telnetlib.Telnet(host="10.10.88.110") #here we're telnet to Gateway 
```

1.  现在我们将通过从通道返回的输出中读取`Username:`关键字并搜索来为 telnet 连接提供用户名。然后我们写入我们的管理员用户名。当我们需要输入 telnet 密码和启用密码时，也是使用相同的过程：

```py
cnx.read_until("Username:") cnx.write(username + "\n") cnx.read_until("Password:") cnx.write(password + "\n") cnx.read_until(">") cnx.write("en" + "\n") cnx.read_until("Password:") cnx.write(enable_password + "\n")  
```

重要的是要提供在建立 telnet 连接或连接时出现在控制台中的确切关键字，否则连接将进入无限循环。然后 Python 脚本将因错误而超时。

1.  最后，我们将在通道上写入`show ip interface brief`命令，并读取直到路由器提示`#`以获取输出。这应该可以让我们获取路由器中的接口配置：

```py
cnx.read_until("#") cnx.write("show ip int b" + "\n") output = cnx.read_until("#") print output
```

完整的脚本如下：

![](img/00076.jpeg)

脚本输出如下：

![](img/00077.jpeg)

请注意，输出包含执行的命令`show ip int b`，并且路由器提示“R1＃”被返回并打印在`stdout`中。我们可以使用内置的字符串函数如`replace()`来清除它们的输出：

```py
cleaned_output = output.replace("show ip int b","").replace("R1#","")
print cleaned_output 
```

![](img/00078.jpeg)

正如您注意到的，我们在脚本中以明文形式提供了密码和启用密码，这被认为是一个安全问题。在 Python 脚本中硬编码这些值也不是一个好的做法。稍后在下一节中，我们将隐藏密码并设计一个机制，仅在脚本运行时提供凭据。

此外，如果您想要执行跨多个页面的命令输出，比如`show running config`，那么您需要先发送`terminal length 0`来禁用分页，然后再连接到设备并发送命令。

# 使用 telnetlib 推送配置

在前一节中，我们通过执行`show ip int brief`来简化`telnetlib`的操作。现在我们需要利用它来将 VLAN 配置推送到拓扑中的四个交换机。我们可以使用 python 的`range()`函数创建一个 VLAN 列表，并迭代它以将 VLAN ID 推送到当前交换机。请注意，我们将交换机 IP 地址定义为列表中的一个项目，这个列表将是我们外部的`for`循环。此外，我将使用另一个内置模块称为`getpass`来隐藏控制台中的密码，并且只在脚本运行时提供它：

```py
#!/usr/bin/python import telnetlib
import getpass
import time

switch_ips = ["10.10.88.111", "10.10.88.112", "10.10.88.113", "10.10.88.114"] username = raw_input("Please Enter your username:") password = getpass.getpass("Please Enter your Password:") enable_password = getpass.getpass("Please Enter your Enable Password:")   for sw_ip in switch_ips:
  print "\n#################### Working on Device " + sw_ip + " ####################"
  connection = telnetlib.Telnet(host=sw_ip.strip())
  connection.read_until("Username:")
  connection.write(username + "\n")
  connection.read_until("Password:")
  connection.write(password + "\n")
  connection.read_until(">")
  connection.write("enable" + "\n")
  connection.read_until("Password:")
  connection.write(enable_password + "\n")
  connection.read_until("#")
  connection.write("config terminal" + "\n") # now i'm in config mode
  vlans = range(300,400)
  for vlan_id in vlans:
  print "\n********* Adding VLAN " + str(vlan_id) + "**********"
  connection.read_until("#")
  connection.write("vlan " + str(vlan_id) + "\n")
  time.sleep(1)
  connection.write("exit" + "\n")
  connection.read_until("#")
  connection.close()
```

在我们最外层的`for`循环中，我们正在迭代设备，然后在每次迭代（每个设备）中，我们从 300 到 400 生成一个 vlan 范围，并将它们推送到当前设备。

脚本输出为：

![](img/00079.jpeg)

此外，您还可以检查交换机控制台本身的输出（输出已省略）：

![](img/00080.jpeg)

# 使用 netaddr 处理 IP 地址和网络

处理和操作 IP 地址是网络工程师最重要的任务之一。Python 开发人员提供了一个了不起的库，可以理解 IP 地址并对其进行操作，称为`netaddr`。例如，假设您开发了一个应用程序，其中的一部分是获取`129.183.1.55/21`的网络和广播地址。您可以通过模块内部的两个内置方法`network`和`broadcast`轻松实现：

```py
net.network
129.183.0.
net.broadcast
129.183.0.0
```

总的来说，netaddr 提供以下功能的支持：

**第 3 层地址：**

+   IPv4 和 IPv6 地址、子网、掩码、前缀

+   迭代、切片、排序、总结和分类 IP 网络

+   处理各种范围格式（CIDR、任意范围和通配符、nmap）

+   基于集合的操作（并集、交集等）在 IP 地址和子网上

+   解析各种不同格式和符号

+   查找 IANA IP 块信息

+   生成 DNS 反向查找

+   超网和子网

**第 2 层地址：**

+   表示和操作 MAC 地址和 EUI-64 标识符

+   查找 IEEE 组织信息（OUI、IAB）

+   生成派生的 IPv6 地址

# Netaddr 安装

netaddr 模块可以使用 pip 安装，如下所示：

```py
pip install netaddr
```

作为成功安装模块的验证，您可以在安装后打开 PyCharm 或 Python 控制台，并导入模块。如果没有产生错误，则模块安装成功：

```py
python
>>>import netaddr
```

# 探索 netaddr 方法

`netaddr`模块提供了两种重要的方法来定义 IP 地址并对其进行操作。第一个称为`IPAddress()`，用于定义具有默认子网掩码的单个类 IP 地址。第二种方法是`IPNetwork()`，用于定义具有 CIDR 的无类别 IP 地址。

这两种方法都将 IP 地址作为字符串，并返回该字符串的 IP 地址或 IP 网络对象。可以对返回的对象执行许多操作。例如，我们可以检查 IP 地址是单播、多播、环回、私有、公共，甚至是有效还是无效。上述操作的输出要么是`True`，要么是`False`，可以在 Python 的`if`条件中使用。

此外，该模块支持比较操作，如`==`、`<`和`>`来比较两个 IP 地址，生成子网，还可以检索给定 IP 地址或子网所属的超网列表。最后，`netaddr`模块可以生成一个完整的有效主机列表（不包括网络 IP 和网络广播）：

```py
#!/usr/bin/python __author__ = "Bassim Aly" __EMAIL__ = "basim.alyy@gmail.com"  from netaddr import IPNetwork,IPAddress
def check_ip_address(ipaddr):
  ip_attributes = []
  ipaddress = IPAddress(ipaddr)    if ipaddress.is_private():
  ip_attributes.append("IP Address is Private")   else:
  ip_attributes.append("IP Address is public")   if ipaddress.is_unicast():
  ip_attributes.append("IP Address is unicast")
  elif ipaddress.is_multicast():
  ip_attributes.append("IP Address is multicast")   if ipaddress.is_loopback():
  ip_attributes.append("IP Address is loopback")    return "\n".join(ip_attributes)   def operate_on_ip_network(ipnet):    net_attributes = []
  net = IPNetwork(ipnet)   net_attributes.append("Network IP Address is " + str(net.network) + " and Netowrk Mask is " + str(net.netmask))    net_attributes.append("The Broadcast is " + str(net.broadcast) )   net_attributes.append("IP Version is " + str(net.version) )
  net_attributes.append("Information known about this network is " + str(net.info) )   net_attributes.append("The IPv6 representation is " + str(net.ipv6()))   net_attributes.append("The Network size is " + str(net.size))   net_attributes.append("Generating a list of ip addresses inside the subnet")   for ip in net:
  net_attributes.append("\t" + str(ip))   return "\n".join(net_attributes)  
ipaddr = raw_input("Please Enter the IP Address: ") print check_ip_address(ipaddr)    ipnet = raw_input("Please Enter the IP Network: ") print operate_on_ip_network(ipnet) 
```

前面的脚本首先使用`raw_input()`函数从用户那里请求 IP 地址和 IP 网络，然后将调用两个用户方法`check_ip_address()`和`operate_on_ip_network()`，并将输入的值传递给它们。第一个函数`check_ip_address()`将检查输入的 IP 地址并尝试生成有关 IP 地址属性的报告，例如它是单播 IP、多播、私有还是环回，并将输出返回给用户。

第二个函数`operate_on_ip_network()`接受 IP 网络并生成网络 ID、子网掩码、广播、版本、已知有关此网络的信息、IPv6 表示，最后生成此子网中的所有 IP 地址。

重要的是要注意，`net.info`仅对公共 IP 地址有效，而不适用于私有 IP 地址。

请注意，在使用它们之前，我们需要从`netaddr`模块导入`IP Network`和`IP Address`。

脚本输出如下：

![](img/00081.jpeg)

# 示例用例

随着我们的网络变得越来越大，并开始包含来自不同供应商的许多设备，我们需要创建模块化的 Python 脚本来自动化其中的各种任务。在接下来的几节中，我们将探讨三种用例，这些用例可以用来从我们的网络中收集不同的信息，并降低故障排除所需的时间，或者至少将网络配置恢复到其上次已知的良好状态。这将使网络工程师更专注于完成他们的工作，并为企业提供处理网络故障和恢复的自动化工作流程。

# 备份设备配置

备份设备配置是任何网络工程师的最重要任务之一。在这种用例中，我们将设计一个示例 Python 脚本，可用于不同的供应商和平台，以备份设备配置。我们将利用`netmiko`库来执行此任务。

结果文件应该以设备 IP 地址格式化，以便以后轻松访问或引用。例如，SW1 备份操作的结果文件应为`dev_10.10.88.111_.cfg`。

# 构建 Python 脚本

我们将首先定义我们的交换机。我们希望将它们的配置备份为文本文件，并提供由逗号分隔的凭据和访问详细信息。这将使我们能够在 Python 脚本中使用`split()`函数获取数据，并在`ConnectHandler`函数中使用它。此外，该文件可以轻松地从 Microsoft Excel 表或任何数据库中导出和导入。

文件结构如下：

`<device_ipaddress>,<username>,<password>,<enable_password>,<vendor>`

![](img/00082.jpeg)

现在我们将通过导入文件并使用`with open`子句来开始构建我们的 Python 脚本。我们使用文件上的`readlines()`将每一行作为列表中的一个项目。我们将创建一个`for`循环来遍历每一行，并使用`split()`函数来通过逗号分隔访问详细信息并将它们分配给变量：

```py
from netmiko import ConnectHandler
from datetime import datetime

with open("/media/bassim/DATA/GoogleDrive/Packt/EnterpriseAutomationProject/Chapter5_Using_Python_to_manage_network_devices/UC1_devices.txt") as devices_file:    devices = devices_file.readlines()   for line in devices:
    line = line.strip("\n")
  ipaddr = line.split(",")[0]
  username = line.split(",")[1]
  password = line.split(",")[2]
  enable_password = line.split(",")[3]    vendor = line.split(",")[4]    if vendor.lower() == "cisco":
  device_type = "cisco_ios"
  backup_command = "show running-config"    elif vendor.lower() == "juniper":
  device_type = "juniper"
  backup_command = "show configuration | display set"   
```

由于我们需要创建一个模块化和多供应商的脚本，我们需要在每一行中使用`if`子句检查供应商，并为当前设备分配正确的`device_type`和`backup_command`。

接下来，我们现在准备使用`netmiko`模块中可用的`.send_command()`方法建立与设备的 SSH 连接并在其上执行备份命令：

```py

print str(datetime.now()) + " Connecting to device {}" .format(ipaddr)   net_connect = ConnectHandler(device_type=device_type,
  ip=ipaddr,
  username=username,
  password=password,
  secret=enable_password) net_connect.enable() running_config = net_connect.send_command(backup_command)   print str(datetime.now()) + " Saving config from device {}" .format(ipaddr)   f = open( "dev_" + ipaddr + "_.cfg", "w") f.write(running_config) f.close() print "=============================================="
```

在最后几个语句中，我们打开了一个文件进行写入，并使其名称包含从我们的文本文件中收集的`ipaddr`变量。

脚本输出如下：

![](img/00083.jpeg)

另外，请注意备份配置文件是在项目主目录中创建的，其名称包含每个设备的 IP 地址：

![](img/00084.jpeg)您可以在 Linux 服务器上设计一个简单的 cron 作业，或在 Windows 服务器上安排一个作业，在特定时间运行先前的 Python 脚本。例如，脚本可以每天午夜运行一次，并将配置存储在`latest`目录中，以便团队以后可以参考。

# 创建您自己的访问终端

在 Python 中，以及一般的编程中，您就是供应商！您可以创建任何代码组合和程序，以满足您的需求。在第二个用例中，我们将创建我们自己的终端，通过`telnetlib`访问路由器。通过在终端中写入几个单词，它将被翻译成在网络设备中执行的多个命令并返回输出，这些输出可以只打印在标准输出中，也可以保存在文件中：

```py
#!/usr/bin/python __author__ = "Bassim Aly" __EMAIL__ = "basim.alyy@gmail.com"   import telnetlib

connection = telnetlib.Telnet(host="10.10.88.110") connection.read_until("Username:") connection.write("admin" + "\n") connection.read_until("Password:") connection.write("access123" + "\n") connection.read_until(">") connection.write("en" + "\n") connection.read_until("Password:") connection.write("access123" + "\n") connection.read_until("#") connection.write("terminal length 0" + "\n") connection.read_until("#") while True:
  command = raw_input("#:")
  if "health" in command.lower():
  commands = ["show ip int b",
  "show ip route",
  "show clock",
  "show banner motd"
  ]    elif "discover" in command.lower():
  commands = ["show arp",
  "show version | i uptime",
  "show inventory",   ]
  else:
  commands = [command]
  for cmd in commands:
  connection.write(cmd + "\n")
  output = connection.read_until("#")
  print output
        print "==================="  
```

首先，我们建立一个 telnet 连接到路由器，并输入用户访问详细信息，直到达到启用模式。然后，我们创建一个始终为`true`的无限`while`循环，并使用内置函数`raw_input()`来期望用户输入命令。当用户输入任何命令时，脚本将捕获它并直接执行到网络设备。

然而，如果用户输入了`health`或`discover`关键字，那么我们的终端将足够智能，以执行一系列命令来反映所需的操作。这在网络故障排除的情况下应该非常有用，您可以扩展它以进行任何日常操作。想象一下，您需要在两台路由器之间排除 OSPF 邻居关系问题。您只需要打开您自己的终端 Python 脚本，您已经教给它了一些用于故障排除所需的命令，并写入类似于`tshoot_ospf`的内容。一旦您的脚本看到这个魔术关键字，它将启动一系列多个命令，打印 OSPF 邻居关系状态、MTU 接口、在 OSPF 下广告的网络等，直到找到问题为止。

**脚本输出：**

通过在提示符中写入`health`来尝试我们脚本中的第一个命令：

![](img/00085.jpeg)

正如您所看到的，脚本返回了在设备中执行的多个命令的输出。

现在尝试第二个支持的命令，`discover`：

![](img/00086.jpeg)

这次脚本返回了发现命令的输出。在后面的章节中，我们可以解析返回的输出并从中提取有用的信息。

# 从 Excel 表中读取数据

网络和 IT 工程师总是使用 Excel 表来存储有关基础设施的信息，如 IP 地址、设备供应商和凭据。Python 支持从 Excel 表中读取信息并处理它，以便您以后在脚本中使用。

在这个用例中，我们将使用**Excel Read**（**xlrd**）模块来读取包含我们基础设施的主机名、IP、用户名、密码、启用密码和供应商信息的`UC3_devices.xlsx`文件，并使用这些信息来提供`netmiko`模块。

Excel 表将如下截图所示：

![](img/00087.jpeg)

首先，我们需要安装`xlrd`模块，使用`pip`，因为我们将使用它来读取 Microsoft Excel 表：

```py
pip install xlrd
```

XLRD 模块读取 Excel 工作簿并将行和列转换为矩阵。例如，如果您需要获取左侧的第一项，那么您将需要访问 row[0][0]。右侧的下一个项将是 row[0][1]，依此类推。

此外，当 xlrd 读取工作表时，它将通过每次读取一行来增加一个名为`nrows`（行数）的特殊计数器。类似地，它将通过每次读取列来增加`ncols`（列数），因此您可以通过这两个参数知道矩阵的大小：

![](img/00088.jpeg)

您可以使用`open_workbook()`函数提供`xlrd`的文件路径。然后，您可以通过使用`sheet_by_index()`或`sheet_by_name()`函数访问包含数据的工作表。对于我们的用例，我们的数据存储在第一个工作表（索引=0）中，并且文件路径存储在章节名称下。然后，我们将遍历工作表中的行，并使用`row()`函数访问特定行。返回的输出是一个列表，我们可以使用索引访问其中的任何项。

Python 脚本：

```py
__author__ = "Bassim Aly" __EMAIL__ = "basim.alyy@gmail.com"   from netmiko import ConnectHandler
from netmiko.ssh_exception import AuthenticationException, NetMikoTimeoutException
import xlrd
from pprint import pprint

workbook = xlrd.open_workbook(r"/media/bassim/DATA/GoogleDrive/Packt/EnterpriseAutomationProject/Chapter4_Using_Python_to_manage_network_devices/UC3_devices.xlsx")   sheet = workbook.sheet_by_index(0)   for index in range(1, sheet.nrows):
  hostname = sheet.row(index)[0].value
    ipaddr = sheet.row(index)[1].value
    username = sheet.row(index)[2].value
    password = sheet.row(index)[3].value
    enable_password = sheet.row(index)[4].value
    vendor = sheet.row(index)[5].value

    device = {
  'device_type': vendor,
  'ip': ipaddr,
  'username': username,
  'password': password,
  'secret': enable_password,    }
  # pprint(device)    print "########## Connecting to Device {0} ############".format(device['ip'])
  try:
  net_connect = ConnectHandler(**device)
  net_connect.enable()    print "***** show ip configuration of Device *****"
  output = net_connect.send_command("show ip int b")
  print output

        net_connect.disconnect()    except NetMikoTimeoutException:
  print "=======SOMETHING WRONG HAPPEN WITH {0}=======".format(device['ip'])    except AuthenticationException:
  print "=======Authentication Failed with {0}=======".format(device['ip'])   
```

```py
  except Exception as unknown_error:
  print "=======SOMETHING UNKNOWN HAPPEN WITH {0}======="   
```

# 更多用例

Netmiko 可以用于实现许多网络自动化用例。它可以用于在升级期间从远程设备上传、下载文件，从 Jinja2 模板加载配置，访问终端服务器，访问终端设备等等。您可以在[`github.com/ktbyers/pynet/tree/master/presentations/dfwcug/examples`](https://github.com/ktbyers/pynet/tree/master/presentations/dfwcug/examples)找到一些有用的用例列表：

![](img/00089.jpeg)

# 摘要

在本章中，我们开始了我们的 Python 网络自动化实践之旅。我们探索了 Python 中可用的不同工具，以建立与 telnet 和 SSH 远程节点的连接，并在其上执行命令。此外，我们学习了如何使用`netaddr`模块处理 IP 地址和网络子网。最后，我们通过两个实际用例加强了我们的知识。

在下一章中，我们将处理返回的输出并开始从中提取有用的信息。
