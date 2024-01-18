# Telnet 和 SSH 上的主机远程监控

在本章中，您将学习如何在配置了 Telnet 和 SSH 的服务器上进行基本配置。我们将首先使用 Telnet 模块，然后使用首选方法实现相同的配置：使用 Python 中的不同模块进行 SSH。您还将了解`telnetlib`、`subprocess`、`fabric`、`Netmiko`和`paramiko`模块的工作原理。在本章中，您必须具备基本的网络知识。

在本章中，我们将涵盖以下主题：

+   `telnetlib()`模块

+   `subprocess.Popen()`模块

+   使用 fabric 模块的 SSH

+   使用 Paramiko 库的 SSH

+   使用 Netmiko 库的 SSH

# telnetlib()模块

在本节中，我们将学习有关 Telnet 协议，然后我们将使用`telnetlib`模块在远程服务器上执行 Telnet 操作。

Telnet 是一种网络协议，允许用户与远程服务器通信。网络管理员通常使用它来远程访问和管理设备。要访问设备，请在终端中使用 Telnet 命令和远程服务器的 IP 地址或主机名。

Telnet 在默认端口号`23`上使用 TCP。要使用 Telnet，请确保它已安装在您的系统上。如果没有，请运行以下命令进行安装：

```py
$ sudo apt-get install telnetd
```

要使用简单的终端运行 Telnet，您只需输入以下命令：

```py
$ telnet ip_address_of_your_remote_server
```

Python 具有`telnetlib`模块，可通过 Python 脚本执行 Telnet 功能。在对远程设备或路由器进行 Telnet 之前，请确保它们已正确配置，如果没有，您可以通过在路由器的终端中使用以下命令进行基本配置：

```py
configure terminal
enable password 'set_Your_password_to_access_router'
username 'set_username' password 'set_password_for_remote_access'
line vty 0 4 
login local 
transport input all 
interface f0/0 
ip add 'set_ip_address_to_the_router' 'put_subnet_mask'
no shut 
end 
show ip interface brief 
```

现在，让我们看一下远程设备的 Telnet 示例。为此，请创建一个`telnet_example.py`脚本，并在其中写入以下内容：

```py
import telnetlib
import getpass
import sys

HOST_IP = "your host ip address"
host_user = input("Enter your telnet username: ")
password = getpass.getpass()

t = telnetlib.Telnet(HOST_IP)
t.read_until(b"Username:")
t.write(host_user.encode("ascii") + b"\n")
if password:
 t.read_until(b"Password:") t.write(password.encode("ascii") + b"\n")

t.write(b"enable\n")
t.write(b"enter_remote_device_password\n") #password of your remote device
t.write(b"conf t\n")
t.write(b"int loop 1\n")
t.write(b"ip add 10.1.1.1 255.255.255.255\n") t.write(b"int loop 2\n") t.write(b"ip add 20.2.2.2 255.255.255.255\n") t.write(b"end\n") t.write(b"exit\n") print(t.read_all().decode("ascii") )
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~$ python3 telnet_example.py Output: Enter your telnet username: student Password: 

server>enable Password: server#conf t Enter configuration commands, one per line.  End with CNTL/Z. server(config)#int loop 1 server(config-if)#ip add 10.1.1.1 255.255.255.255 server(config-if)#int loop 23 server(config-if)#ip add 20.2.2.2 255.255.255.255 server(config-if)#end server#exit
```

在前面的示例中，我们使用`telnetlib`模块访问并配置了 Cisco 路由器。在此脚本中，首先我们从用户那里获取用户名和密码，以初始化与远程设备的 Telnet 连接。当连接建立时，我们对远程设备进行了进一步的配置。Telnet 之后，我们将能够访问远程服务器或设备。但是 Telnet 协议有一个非常重要的缺点，那就是所有数据，包括用户名和密码都以文本方式通过网络发送，这可能会造成安全风险。因此，现在 Telnet 很少使用，并已被一个名为安全外壳（SSH）的非常安全的协议所取代。

# SSH

SSH 是一种网络协议，用于通过远程访问管理设备或服务器。SSH 使用公钥加密来确保安全。Telnet 和 SSH 之间的重要区别在于 SSH 使用加密，这意味着所有通过网络传输的数据都受到未经授权的实时拦截的保护。

用户访问远程服务器或设备必须安装 SSH 客户端。通过在终端中运行以下命令来安装 SSH：

```py
$ sudo apt install ssh
```

此外，在用户希望通信的远程服务器上，必须安装并运行 SSH 服务器。SSH 使用 TCP 协议，默认情况下在端口号`22`上运行。

您可以通过终端运行`ssh`命令如下：

```py
$ ssh host_name@host_ip_address
```

现在，您将学习如何使用 Python 中的不同模块进行 SSH，例如 subprocess、fabric、Netmiko 和 Paramiko。现在，我们将依次看到这些模块。

# subprocess.Popen()模块

`Popen`类处理进程的创建和管理。使用此模块，开发人员可以处理较少的常见情况。子程序执行将在新进程中完成。要在 Unix/Linux 上执行子程序，该类将使用`os.execvp()`函数。要在 Windows 中执行子程序，该类将使用`CreateProcess()`函数。

现在，让我们看一些`subprocess.Popen()`的有用参数：

```py
class subprocess.Popen(args, bufsize=0, executable=None, stdin=None, stdout=None,
 stderr=None, preexec_fn=None, close_fds=False, shell=False, cwd=None, env=None, universal_newlines=False, startupinfo=None, creationflags=0)
```

让我们看看每个参数：

+   `args`：它可以是程序参数的序列或单个字符串。如果`args`是一个序列，则将执行 args 中的第一项。如果 args 是一个字符串，则建议将 args 作为序列传递。

+   `shell`：shell 参数默认设置为`False`，它指定是否使用 shell 来执行程序。如果 shell 为`True`，则建议将 args 作为字符串传递。在 Linux 中，如果`shell=True`，shell 默认为`/bin/sh`。如果`args`是一个字符串，则该字符串指定要通过 shell 执行的命令。

+   `bufsize`：如果`bufsize`为`0`（默认为`0`），则表示无缓冲，如果`bufsize`为`1`，则表示行缓冲。如果`bufsize`是任何其他正值，则使用给定大小的缓冲区。如果`bufsize`是任何其他负值，则表示完全缓冲。

+   `executable`：它指定要执行的替换程序。

+   `stdin`、`stdout`和`stderr`：这些参数分别定义了标准输入、标准输出和标准错误。

+   `preexec_fn`：这是设置为可调用对象的，将在子进程中执行之前调用。

+   `close_fds`：在 Linux 中，如果`close_fds`为 true，则在执行子进程之前，除了`0`、`1`和`2`之外的所有文件描述符都将被关闭。在 Windows 中，如果`close_fds`为`true`，则子进程将不会继承任何句柄。

+   `env`：如果值不是`None`，则映射将为新进程定义环境变量。

+   `universal_newlines`：如果值为`True`，则`stdout`和`stderr`将以换行模式打开为文本文件。

现在，我们将看一个`subprocess.Popen()`的例子。为此，创建一个`ssh_using_sub.py`脚本，并在其中写入以下内容：

```py
import subprocess
import sys

HOST="your host username@host ip"
COMMAND= "ls"

ssh_obj = subprocess.Popen(["ssh", "%s" % HOST, COMMAND],
 shell=False,
 stdout=subprocess.PIPE,
 stderr=subprocess.PIPE)

result = ssh_obj.stdout.readlines()
if result == []:
 err = ssh_obj.stderr.readlines()
 print(sys.stderr, "ERROR: %s" % err)
else:
 print(result)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~$ python3 ssh_using_sub.py Output : student@192.168.0.106's password: [b'Desktop\n', b'Documents\n', b'Downloads\n', b'examples.desktop\n', b'Music\n', b'Pictures\n', b'Public\n', b'sample.py\n', b'spark\n', b'spark-2.3.1-bin-hadoop2.7\n', b'spark-2.3.1-bin-hadoop2.7.tgz\n', b'ssh\n', b'Templates\n', b'test_folder\n', b'test.txt\n', b'Untitled1.ipynb\n', b'Untitled.ipynb\n', b'Videos\n', b'work\n']
```

在前面的例子中，首先我们导入了 subprocess 模块，然后定义了要建立 SSH 连接的主机地址。之后，我们给出了一个在远程设备上执行的简单命令。在所有这些设置完成后，我们将这些信息放入`subprocess.Popen()`函数中。该函数执行函数内定义的参数，以创建与远程设备的连接。建立 SSH 连接后，执行我们定义的命令并提供结果。然后我们在终端上打印 SSH 的结果，如输出所示。

# 使用 fabric 模块的 SSH

Fabric 是一个 Python 库，也是一个用于 SSH 的命令行工具。它用于系统管理和应用程序在网络上的部署。我们还可以通过 SSH 执行 shell 命令。

要使用 fabric 模块，首先您必须使用以下命令安装它：

```py
$ pip3 install fabric3
```

现在，我们将看一个例子。创建一个`fabfile.py`脚本，并在其中写入以下内容：

```py
from fabric.api import * env.hosts=["host_name@host_ip"] env.password='your password'  def dir(): run('mkdir fabric') print('Directory named fabric has been created on your host network') def diskspace():
 run('df') 
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~$ fab dir Output: [student@192.168.0.106] Executing task 'dir' [student@192.168.0.106] run: mkdir fabric
Done. Disconnecting from 192.168.0.106... done.
```

在前面的例子中，首先我们导入了`fabric.api`模块，然后设置了主机名和密码以连接到主机网络。之后，我们设置了不同的任务以通过 SSH 执行。因此，为了执行我们的程序而不是 Python3 的`fabfile.py`，我们使用了`fab`实用程序（`fab dir`），然后我们声明应该从我们的`fabfile.py`执行所需的任务。在我们的情况下，我们执行了`dir`任务，在您的远程网络上创建了一个名为`'fabric'`的目录。您可以在您的 Python 文件中添加您的特定任务。它可以使用 fabric 模块的`fab`实用程序执行。

# 使用 Paramiko 库的 SSH

Paramiko 是一个实现 SSHv2 协议用于远程设备安全连接的库。Paramiko 是围绕 SSH 的纯 Python 接口。

在使用 Paramiko 之前，请确保您已经在系统上正确安装了它。如果尚未安装，可以通过在终端中运行以下命令来安装它：

```py
$ sudo pip3 install paramiko
```

现在，我们将看一个使用`paramiko`的例子。对于这个`paramiko`连接，我们使用的是 Cisco 设备。Paramiko 支持基于密码和基于密钥对的身份验证，以确保与服务器的安全连接。在我们的脚本中，我们使用基于密码的身份验证，这意味着我们检查密码，如果可用，将尝试使用普通用户名/密码身份验证。在对远程设备或多层路由器进行 SSH 之前，请确保它们已经正确配置，如果没有，您可以使用以下命令在多层路由器终端中进行基本配置：

```py
configure t
ip domain-name cciepython.com
crypto key generate rsa
How many bits in the modulus [512]: 1024
interface range f0/0 - 1
switchport mode access
switchport access vlan 1
no shut
int vlan 1
ip add 'set_ip_address_to_the_router' 'put_subnet_mask'
no shut
exit
enable password 'set_Your_password_to_access_router'
username 'set_username' password 'set_password_for_remote_access'
username 'username' privilege 15
line vty 0 4
login local
transport input all
end
```

现在，创建一个`pmiko.py`脚本，并在其中写入以下内容：

```py
import paramiko
import time

ip_address = "host_ip_address"
usr = "host_username"
pwd = "host_password"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(hostname=ip_address,username=usr,password=pwd)

print("SSH connection is successfully established with ", ip_address)
rc = c.invoke_shell() for n in range (2,6):
 print("Creating VLAN " + str(n)) rc.send("vlan database\n") rc.send("vlan " + str(n) +  "\n") rc.send("exit\n") time.sleep(0.5) time.sleep(1) output = rc.recv(65535) print(output) c.close
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~$ python3 pmiko.py Output: SSH connection is successfuly established with  192.168.0.70 Creating VLAN 2 Creating VLAN 3 Creating VLAN 4 Creating VLAN 5
```

在上面的例子中，首先我们导入了`paramiko`模块，然后定义了连接到远程设备所需的 SSH 凭据。提供凭据后，我们创建了`paramiko.SSHclient()`的实例`'c'`，这是用于与远程设备建立连接并执行命令或操作的主要客户端。创建`SSHClient`对象允许我们使用`.connect()`函数建立远程连接。然后，我们设置了`paramiko`连接的策略，因为默认情况下，`paramiko.SSHclient`将 SSH 策略设置为拒绝策略状态。这会导致策略拒绝任何未经验证的 SSH 连接。在我们的脚本中，我们使用`AutoAddPolicy()`函数来忽略 SSH 连接中断的可能性，该函数会自动添加服务器的主机密钥而无需提示。我们只能在测试目的中使用此策略，但出于安全目的，这不是生产环境中的好选择。

当建立 SSH 连接时，您可以在设备上进行任何配置或操作。在这里，我们在远程设备上创建了一些虚拟局域网。创建 VLAN 后，我们只是关闭了连接。

# 使用 Netmiko 库进行 SSH

在本节中，我们将学习 Netmiko。Netmiko 库是 Paramiko 的高级版本。它是一个基于 Paramiko 的“multi_vendor”库。Netmiko 简化了与网络设备的 SSH 连接，并对设备进行特定操作。在对远程设备或多层路由器进行 SSH 之前，请确保它们已经正确配置，如果没有，您可以使用 Paramiko 部分中提到的命令进行基本配置。

现在，让我们看一个例子。创建一个`nmiko.py`脚本，并在其中写入以下代码：

```py
from netmiko import ConnectHandler

remote_device={
 'device_type': 'cisco_ios',
 'ip':  'your remote_device ip address',
 'username': 'username',
 'password': 'password',
}

remote_connection = ConnectHandler(**remote_device)
#net_connect.find_prompt()

for n in range (2,6):
 print("Creating VLAN " + str(n))
 commands = ['exit','vlan database','vlan ' + str(n), 'exit']
 output = remote_connection.send_config_set(commands)
 print(output)

command = remote_connection.send_command('show vlan-switch brief')
print(command)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~$ python3 nmiko.py Output: Creating VLAN 2 config term Enter configuration commands, one per line.  End with CNTL/Z. server(config)#exit server #vlan database server (vlan)#vlan 2 VLAN 2 modified: server (vlan)#exit APPLY completed. Exiting.... server # .. .. .. .. switch# Creating VLAN 5 config term Enter configuration commands, one per line.  End with CNTL/Z. server (config)#exit server #vlan database server (vlan)#vlan 5 VLAN 5 modified: server (vlan)#exit APPLY completed. Exiting.... VLAN Name                             Status    Ports ---- -------------------------------- --------- ------------------------------- 1    default                          active    Fa0/0, Fa0/1, Fa0/2, Fa0/3, Fa0/4, Fa0/5, Fa0/6, Fa0/7, Fa0/8, Fa0/9, Fa0/10, Fa0/11, Fa0/12, Fa0/13, Fa0/14, Fa0/15 2    VLAN0002                         active 3    VLAN0003                         active 4    VLAN0004                         active 5    VLAN0005                         active 1002 fddi-default                    active 1003 token-ring-default         active 1004 fddinet-default               active 1005 trnet-default                    active
```

在上面的例子中，我们使用 Netmiko 库来进行 SSH，而不是 Paramiko。在这个脚本中，首先我们从 Netmiko 库中导入`ConnectHandler`，然后通过传递设备字典（在我们的情况下是`remote_device`）来建立与远程网络设备的 SSH 连接。连接建立后，我们执行配置命令，使用`send_config_set()`函数创建了多个虚拟局域网。

当我们使用这种类型（`.send_config_set()`）的函数在远程设备上传递命令时，它会自动将我们的设备设置为配置模式。在发送配置命令后，我们还传递了一个简单的命令来获取有关配置设备的信息。

# 摘要

在本章中，您学习了 Telnet 和 SSH。您还学习了不同的 Python 模块，如 telnetlib、subprocess、fabric、Netmiko 和 Paramiko，使用这些模块可以执行 Telnet 和 SSH。SSH 使用公钥加密以确保安全，并且比 Telnet 更安全。

在下一章中，我们将使用各种 Python 库，您可以使用这些库创建图形用户界面。

# 问题

1.  什么是客户端-服务器架构？

1.  如何在 Python 代码中运行特定于操作系统的命令？

1.  局域网和虚拟局域网之间有什么区别？

1.  以下代码的输出是什么？

```py
 List = [‘a’, ‘b’, ‘c’, ‘d’, ‘e’]
 Print(list [10:])
```

1.  编写一个 Python 程序来显示日历（提示：使用`calendar`模块）。

1.  编写一个 Python 程序来计算文本文件中的行数。

# 进一步阅读

+   Paramiko 文档：[`github.com/paramiko/paramiko`](https://github.com/paramiko/paramiko)

+   Fabric 文档：[`www.fabfile.org/`](http://www.fabfile.org/)
