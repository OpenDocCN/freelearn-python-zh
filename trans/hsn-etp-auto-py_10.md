# 使用Fabric运行系统管理任务

在上一章中，我们使用了`subprocess`模块在托管我们的Python脚本的机器内运行和生成系统进程，并将输出返回到终端。然而，许多自动化任务需要访问远程服务器以执行命令，这不容易使用子进程来实现。使用另一个Python模块`Fabric`就变得轻而易举。该库连接到远程主机并执行不同的任务，例如上传和下载文件，使用特定用户ID运行命令，并提示用户输入。`Fabric` Python模块是从一个中心点管理数十台Linux机器的强大工具。

本章将涵盖以下主题：

+   什么是Fabric？

+   执行您的第一个Fabric文件

+   其他有用的Fabric功能

# 技术要求

以下工具应安装并在您的环境中可用：

+   Python 2.7.1x。

+   PyCharm社区版或专业版。

+   EVE-NG拓扑。有关如何安装和配置系统服务器，请参阅第8章“准备实验环境”。

您可以在以下GitHub URL找到本章中开发的完整脚本：[https://github.com/TheNetworker/EnterpriseAutomation.git](https://github.com/TheNetworker/EnterpriseAutomation.git)。

# 什么是Fabric？

Fabric ([http://www.fabfile.org/](http://www.fabfile.org/))是一个高级Python库，用于连接到远程服务器（通过paramiko库）并在其上执行预定义的任务。它在托管fabric模块的机器上运行一个名为**fab**的工具。此工具将查找位于您运行工具的相同目录中的`fabfile.py`文件。`fabfile.py`文件包含您的任务，定义为从命令行调用的Python函数，以在服务器上启动执行。Fabric任务本身只是普通的Python函数，但它们包含用于在远程服务器上执行命令的特殊方法。此外，在`fabfile.py`的开头，您需要定义一些环境变量，例如远程主机、用户名、密码以及执行期间所需的任何其他变量：

![](../images/00154.jpeg)

# 安装

Fabric需要Python 2.5到2.7。您可以使用`pip`安装Fabric及其所有依赖项，也可以使用系统包管理器，如`yum`或`apt`。在这两种情况下，您都将在操作系统中准备好并可执行`fab`实用程序。

要使用`pip`安装`fabric`，请在自动化服务器上运行以下命令：

```py
pip install fabric
```

![](../images/00155.gif)

请注意，Fabric需要`paramiko`，这是一个常用的Python库，用于建立SSH连接。

您可以通过两个步骤验证Fabric安装。首先，确保您的系统中有`fab`命令可用：

```py
[root@AutomationServer ~]# which fab
/usr/bin/fab
```

验证的第二步是打开Python并尝试导入`fabric`库。如果没有抛出错误，则Fabric已成功安装：

```py
[root@AutomationServer ~]# python
Python 2.7.5 (default, Aug  4 2017, 00:39:18) 
[GCC 4.8.5 20150623 (Red Hat 4.8.5-16)] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> from fabric.api import *
>>>

```

# Fabric操作

`fabric`工具中有许多可用的操作。这些操作在fabfile中作为任务内的函数，但以下是`fabric`库中最重要操作的摘要。

# 使用运行操作

Fabric中`run`操作的语法如下：

```py
run(command, shell=True, pty=True, combine_stderr=True, quiet=False, warn_only=False, stdout=None, stderr=None)
```

这将在远程主机上执行命令，而`shell`参数控制是否在执行之前创建一个shell（例如`/bin/sh`）（相同的参数也存在于子进程中）。

命令执行后，Fabric将填充`.succeeded`或`.failed`，取决于命令输出。您可以通过调用以下内容来检查命令是否成功或失败：

```py
def run_ops():
  output = run("hostname")  
```

# 使用获取操作

Fabric `get`操作的语法如下：

```py
get(remote_path, local_path)
```

这将从远程主机下载文件到运行 `fabfile` 的机器，使用 `rsync` 或 `scp`。例如，当您需要将日志文件收集到服务器时，通常会使用此功能。

```py
def get_ops():
  try:
  get("/var/log/messages","/root/")
  except:
  pass
```

# 使用 put 操作

Fabric `put` 操作的语法如下：

```py
put(local_path, remote_path, use_sudo=False, mirror_local_mode=False, mode=None)
```

此操作将从运行 `fabfile`（本地）的机器上传文件到远程主机。使用 `use_sudo` 将解决上传到根目录时的权限问题。此外，您可以保持本地和远程服务器上的当前文件权限，或者您可以设置新的权限：

```py
def put_ops():
  try:
  put("/root/VeryImportantFile.txt","/root/")
  except:
  pass
```

# 使用 sudo 操作

Fabric `sudo` 操作的语法如下：

```py
sudo(command, shell=True, pty=True, combine_stderr=True, user=None, quiet=False, warn_only=False, stdout=None, stderr=None, group=None)
```

此操作可以被视为 `run()` 命令的另一个包装器。但是，`sudo` 操作将默认使用 root 用户名运行命令，而不管用于执行 `fabfile` 的用户名如何。它还包含一个用户参数，该参数可用于使用不同的用户名运行命令。此外，`user` 参数使用特定的 UID 执行命令，而 `group` 参数定义 GID：

```py
def sudo_ops():
  sudo("whoami") #it should print the root even if you use another account
```

# 使用提示操作

Fabric `prompt` 操作的语法如下：

```py
prompt(text, key=None, default='', validate=None)
```

用户可以使用 `prompt` 操作为任务提供特定值，并且输入将存储在变量中并被任务使用。请注意，您将为 `fabfile` 中的每个主机提示：

```py
def prompt_ops():
  prompt("please supply release name", default="7.4.1708")
```

# 使用重新启动操作

Fabric `reboot` 操作的语法如下：

```py
reboot(wait=120)
```

这是一个简单的操作，默认情况下重新启动主机。Fabric 将等待 120 秒然后尝试重新连接，但是您可以使用 `wait` 参数将此值更改为其他值：

```py
def reboot_ops():
  reboot(wait=60, use_sudo=True) 
```

有关其他支持的操作的完整列表，请查看 [http://docs.fabfile.org/en/1.14/api/core/operations.html](http://docs.fabfile.org/en/1.14/api/core/operations.html)。您还可以直接从 PyCharm 查看它们，方法是查看在键入 *Ctrl + 空格* 时弹出的所有自动完成函数。从 `fabric.operations` 导入 <*ctrl*+*space*> 在 `fabric.operations` 下：

![](../images/00156.jpeg)

# 执行您的第一个 Fabric 文件

现在我们知道操作的工作原理，所以我们将把它放在 `fabfile` 中，并创建一个可以与远程机器一起工作的完整自动化脚本。`fabfile` 的第一步是导入所需的类。其中大部分位于 `fabric.api` 中，因此我们将全局导入所有这些类到我们的 Python 脚本中：

```py
from fabric.api import *
```

下一步是定义远程机器的 IP 地址、用户名和密码。在我们的环境中，除了自动化服务器之外，我们还有两台机器分别运行 Ubuntu 16.04 和 CentOS 7.4，并具有以下详细信息：

| **机器类型** | **IP 地址** | **用户名** | **密码** |
| Ubuntu 16.04 | `10.10.10.140` | `root` | `access123` |
| CentOS 7.4 | `10.10.10.193` | `root` | `access123` |

我们将把它们包含在 Python 脚本中，如下面的片段所示：

```py
env.hosts = [
  '10.10.10.140', # ubuntu machine
  '10.10.10.193', # CentOS machine ]   env.user = "root"  env.password = "access123" 
```

请注意，我们使用名为 `env` 的变量，该变量继承自 `_AttributeDict` 类。在此变量内部，我们可以设置来自 SSH 连接的用户名和密码。您还可以通过设置 `env.use_ssh_config=True` 使用存储在 `.ssh` 目录中的 SSH 密钥；Fabric 将使用这些密钥进行身份验证。

最后一步是将任务定义为 Python 函数。任务可以使用前面的操作来执行命令。

以下是完整的脚本：

```py
from fabric.api import *    env.hosts = [
  '10.10.10.140', # ubuntu machine
  '10.10.10.193', # CentOS machine ]   env.user = "root" env.password = "access123"   def detect_host_type():
  output = run("uname -s")   if output.failed:
  print("something wrong happen, please check the logs")   elif output.succeeded:
  print("command executed successfully")   def list_all_files_in_directory():
  directory = prompt("please enter full path to the directory to list", default="/root")
  sudo("cd {0} ; ls -htlr".format(directory))     def main_tasks():
  detect_host_type()
  list_all_files_in_directory()
```

在上面的示例中，适用以下内容：

+   我们定义了两个任务。第一个任务将执行 `uname -s` 命令并返回输出，然后验证命令是否成功执行。该任务使用 `run()` 操作来完成。

+   第二个任务将使用两个操作：`prompt()` 和 `sudo()`。第一个操作将要求用户输入目录的完整路径，而第二个操作将列出目录中的所有内容。

+   最终任务`main_tasks()`将实际上将前面的两种方法组合成一个任务，以便我们可以从命令行调用它。

为了运行脚本，我们将上传文件到自动化服务器，并使用`fab`实用程序来运行它：

```py
fab -f </full/path/to/fabfile>.py <task_name>
```

在上一个命令中，如果您的文件名不是`fabfile.py`，则`-f`开关是不强制的。如果不是，您将需要向`fab`实用程序提供名称。此外，`fabfile`应该在当前目录中；否则，您将需要提供完整的路径。现在我们将通过执行以下命令来运行`fabfile`：

```py
fab -f fabfile_first.py main_tasks
```

第一个任务将被执行，并将输出返回到终端：

```py
[10.10.10.140] Executing task 'main_tasks'
[10.10.10.140] run: uname -s
[10.10.10.140] out: Linux
[10.10.10.140] out: 

command executed successfully 
```

现在，我们将进入`/var/log/`来列出内容：

```py

please enter full path to the directory to list [/root] /var/log/
[10.10.10.140] sudo: cd /var/log/ ; ls -htlr
[10.10.10.140] out: total 1.7M
[10.10.10.140] out: drwxr-xr-x 2 root   root 4.0K Dec  7 23:54 lxd
[10.10.10.140] out: drwxr-xr-x 2 root   root 4.0K Dec 11 15:47 sysstat
[10.10.10.140] out: drwxr-xr-x 2 root   root 4.0K Feb 22 18:24 dist-upgrade
[10.10.10.140] out: -rw------- 1 root   utmp    0 Feb 28 20:23 btmp
[10.10.10.140] out: -rw-r----- 1 root   adm    31 Feb 28 20:24 dmesg
[10.10.10.140] out: -rw-r--r-- 1 root   root  57K Feb 28 20:24 bootstrap.log
[10.10.10.140] out: drwxr-xr-x 2 root   root 4.0K Apr  4 08:00 fsck
[10.10.10.140] out: drwxr-xr-x 2 root   root 4.0K Apr  4 08:01 apt
[10.10.10.140] out: -rw-r--r-- 1 root   root  32K Apr  4 08:09 faillog
[10.10.10.140] out: drwxr-xr-x 3 root   root 4.0K Apr  4 08:09 installer

command executed successfully
```

如果您需要列出CentOS机器上`network-scripts`目录下的配置文件，也是一样的：

```py
 please enter full path to the directory to list [/root] /etc/sysconfig/network-scripts/ 
[10.10.10.193] sudo: cd /etc/sysconfig/network-scripts/ ; ls -htlr
[10.10.10.193] out: total 232K
[10.10.10.193] out: -rwxr-xr-x. 1 root root 1.9K Apr 15  2016 ifup-TeamPort
[10.10.10.193] out: -rwxr-xr-x. 1 root root 1.8K Apr 15  2016 ifup-Team
[10.10.10.193] out: -rwxr-xr-x. 1 root root 1.6K Apr 15  2016 ifdown-TeamPort
[10.10.10.193] out: -rw-r--r--. 1 root root  31K May  3  2017 network-functions-ipv6
[10.10.10.193] out: -rw-r--r--. 1 root root  19K May  3  2017 network-functions
[10.10.10.193] out: -rwxr-xr-x. 1 root root 5.3K May  3  2017 init.ipv6-global
[10.10.10.193] out: -rwxr-xr-x. 1 root root 1.8K May  3  2017 ifup-wireless
[10.10.10.193] out: -rwxr-xr-x. 1 root root 2.7K May  3  2017 ifup-tunnel
[10.10.10.193] out: -rwxr-xr-x. 1 root root 3.3K May  3  2017 ifup-sit
[10.10.10.193] out: -rwxr-xr-x. 1 root root 2.0K May  3  2017 ifup-routes
[10.10.10.193] out: -rwxr-xr-x. 1 root root 4.1K May  3  2017 ifup-ppp
[10.10.10.193] out: -rwxr-xr-x. 1 root root 3.4K May  3  2017 ifup-post
[10.10.10.193] out: -rwxr-xr-x. 1 root root 1.1K May  3  2017 ifup-plusb

<***output omitted for brevity>***
```

最后，Fabric将断开与两台机器的连接：

```py
[10.10.10.193] out: 

Done.
Disconnecting from 10.10.10.140... done.
Disconnecting from 10.10.10.193... done.
```

# 有关fab工具的更多信息

`fab`工具本身支持许多操作。它可以用来列出`fabfile`中的不同任务。它还可以在执行期间设置`fab`环境。例如，您可以使用`-H`或`--hosts`开关定义将在其上运行命令的主机，而无需在`fabfile`中指定。这实际上是在执行期间在`fabfile`中设置`env.hosts`变量：

```py
fab -H srv1,srv2
```

另一方面，您可以使用`fab`工具定义要运行的命令。这有点像Ansible的临时模式（我们将在[第13章](part0168.html#506UG0-9cfcdc5beecd470bbeda046372f0337f)中详细讨论这个问题，*系统管理的Ansible*）：

```py
fab -H srv1,srv2 -- ifconfig -a
```

如果您不想在`fabfile`脚本中以明文存储密码，那么您有两个选项。第一个是使用`-i`选项使用SSH身份文件（私钥），它在连接期间加载文件。

另一个选项是使用`-I`选项强制Fabric在连接到远程机器之前提示您输入会话密码。

请注意，如果在`fabfile`中指定了`env.password`参数，此选项将覆盖该参数。

`-D`开关将禁用已知主机，并强制Fabric不从`.ssh`目录加载`known_hosts`文件。您可以使用`-r`或`--reject-unknown-hosts`选项使Fabric拒绝连接到`known_hosts`文件中未定义的主机。

此外，您还可以使用`-l`或`--list`在`fabfile`中列出所有支持的任务，向`fab`工具提供`fabfile`名称。例如，将其应用到前面的脚本将生成以下输出：

```py
# fab -f fabfile_first.py -l
Available commands:

    detect_host_type
    list_all_files_in_directory
    main_tasks
```

您可以使用`-h`开关在命令行中查看`fab`命令的所有可用选项和参数，或者在[http://docs.fabfile.org/en/1.14/usage/fab.html](http://docs.fabfile.org/en/1.14/usage/fab.html)上查看。

# 使用Fabric发现系统健康

在这种用例中，我们将利用Fabric开发一个脚本，在远程机器上执行多个命令。脚本的目标是收集两种类型的输出：`discovery`命令和`health`命令。`discovery`命令收集正常运行时间、主机名、内核版本以及私有和公共IP地址，而`health`命令收集已使用的内存、CPU利用率、生成的进程数量和磁盘使用情况。我们将设计`fabfile`，以便我们可以扩展我们的脚本并向其中添加更多命令：

```py
#!/usr/bin/python __author__ = "Bassim Aly" __EMAIL__ = "basim.alyy@gmail.com"   from fabric.api import * from fabric.context_managers import * from pprint import pprint

env.hosts = [
  '10.10.10.140', # Ubuntu Machine
  '10.10.10.193', # CentOS Machine ]   env.user = "root" env.password = "access123"     def get_system_health():    discovery_commands = {
  "uptime": "uptime | awk '{print $3,$4}'",
  "hostname": "hostname",
  "kernel_release": "uname -r",
  "architecture": "uname -m",
  "internal_ip": "hostname -I",
  "external_ip": "curl -s ipecho.net/plain;echo",      }
  health_commands = {
  "used_memory": "free  | awk '{print $3}' | grep -v free | head -n1",
  "free_memory": "free  | awk '{print $4}' | grep -v shared | head -n1",
  "cpu_usr_percentage": "mpstat | grep -A 1 '%usr' | tail -n1 | awk '{print $4}'",
  "number_of_process": "ps -A --no-headers | wc -l",
  "logged_users": "who",
  "top_load_average": "top -n 1 -b | grep 'load average:' | awk '{print $10 $11 $12}'",
  "disk_usage": "df -h| egrep 'Filesystem|/dev/sda*|nvme*'"    }    tasks = [discovery_commands,health_commands]    for task in tasks:
  for operation,command in task.iteritems():
  print("============================={0}=============================".format(operation))
  output = run(command)
```

请注意，我们创建了两个字典：`discover_commands`和`health_commands`。每个字典都包含Linux命令作为键值对。键表示操作，而值表示实际的Linux命令。然后，我们创建了一个`tasks`列表来组合这两个字典。

最后，我们创建了一个嵌套的`for`循环。外部循环用于遍历列表项。内部`for`循环用于遍历键值对。使用Fabric的`run()`操作将命令发送到远程主机：

```py
# fab -f fabfile_discoveryAndHealth.py get_system_health
[10.10.10.140] Executing task 'get_system_health'
=============================uptime=============================
[10.10.10.140] run: uptime | awk '{print $3,$4}'
[10.10.10.140] out: 3:26, 2
[10.10.10.140] out: 

=============================kernel_release=============================
[10.10.10.140] run: uname -r
[10.10.10.140] out: 4.4.0-116-generic
[10.10.10.140] out: 

=============================external_ip=============================
[10.10.10.140] run: curl -s ipecho.net/plain;echo
[10.10.10.140] out: <Author_Masked_The_Output_For_Privacy>
[10.10.10.140] out: 

=============================hostname=============================
[10.10.10.140] run: hostname
[10.10.10.140] out: ubuntu-machine
[10.10.10.140] out: 

=============================internal_ip=============================
[10.10.10.140] run: hostname -I
[10.10.10.140] out: 10.10.10.140 
[10.10.10.140] out: 

=============================architecture=============================
[10.10.10.140] run: uname -m
[10.10.10.140] out: x86_64
[10.10.10.140] out: 

=============================disk_usage=============================
[10.10.10.140] run: df -h| egrep 'Filesystem|/dev/sda*|nvme*'
[10.10.10.140] out: Filesystem                            Size  Used Avail Use% Mounted on
[10.10.10.140] out: /dev/sda1                             472M   58M  390M  13% /boot
[10.10.10.140] out: 

=============================used_memory=============================
[10.10.10.140] run: free  | awk '{print $3}' | grep -v free | head -n1
[10.10.10.140] out: 75416
[10.10.10.140] out: 

=============================logged_users=============================
[10.10.10.140] run: who
[10.10.10.140] out: root     pts/0        2018-04-08 23:36 (10.10.10.130)
[10.10.10.140] out: root     pts/1        2018-04-08 21:23 (10.10.10.1)
[10.10.10.140] out: 

=============================top_load_average=============================
[10.10.10.140] run: top -n 1 -b | grep 'load average:' | awk '{print $10 $11 $12}'
[10.10.10.140] out: 0.16,0.03,0.01
[10.10.10.140] out: 

=============================cpu_usr_percentage=============================
[10.10.10.140] run: mpstat | grep -A 1 '%usr' | tail -n1 | awk '{print $4}'
[10.10.10.140] out: 0.02
[10.10.10.140] out: 

=============================number_of_process=============================
[10.10.10.140] run: ps -A --no-headers | wc -l
[10.10.10.140] out: 131
[10.10.10.140] out: 

=============================free_memory=============================
[10.10.10.140] run: free  | awk '{print $4}' | grep -v shared | head -n1
[10.10.10.140] out: 5869268
[10.10.10.140] out: 

```

`get_system_health`相同的任务也将在第二台服务器上执行，并将输出返回到终端：

```py
[10.10.10.193] Executing task 'get_system_health'
=============================uptime=============================
[10.10.10.193] run: uptime | awk '{print $3,$4}'
[10.10.10.193] out: 3:26, 2
[10.10.10.193] out: 

=============================kernel_release=============================
[10.10.10.193] run: uname -r
[10.10.10.193] out: 3.10.0-693.el7.x86_64
[10.10.10.193] out: 

=============================external_ip=============================
[10.10.10.193] run: curl -s ipecho.net/plain;echo
[10.10.10.193] out: <Author_Masked_The_Output_For_Privacy>
[10.10.10.193] out: 

=============================hostname=============================
[10.10.10.193] run: hostname
[10.10.10.193] out: controller329
[10.10.10.193] out: 

=============================internal_ip=============================
[10.10.10.193] run: hostname -I
[10.10.10.193] out: 10.10.10.193 
[10.10.10.193] out: 

=============================architecture=============================
[10.10.10.193] run: uname -m
[10.10.10.193] out: x86_64
[10.10.10.193] out: 

=============================disk_usage=============================
[10.10.10.193] run: df -h| egrep 'Filesystem|/dev/sda*|nvme*'
[10.10.10.193] out: Filesystem               Size  Used Avail Use% Mounted on
[10.10.10.193] out: /dev/sda1                488M   93M  360M  21% /boot
[10.10.10.193] out: 

=============================used_memory=============================
[10.10.10.193] run: free  | awk '{print $3}' | grep -v free | head -n1
[10.10.10.193] out: 287048
[10.10.10.193] out: 

=============================logged_users=============================
[10.10.10.193] run: who
[10.10.10.193] out: root     pts/0        2018-04-08 23:36 (10.10.10.130)
[10.10.10.193] out: root     pts/1        2018-04-08 21:23 (10.10.10.1)
[10.10.10.193] out: 

=============================top_load_average=============================
[10.10.10.193] run: top -n 1 -b | grep 'load average:' | awk '{print $10 $11 $12}'
[10.10.10.193] out: 0.00,0.01,0.02
[10.10.10.193] out: 

=============================cpu_usr_percentage=============================
[10.10.10.193] run: mpstat | grep -A 1 '%usr' | tail -n1 | awk '{print $4}'
[10.10.10.193] out: 0.00
[10.10.10.193] out: 

=============================number_of_process=============================
[10.10.10.193] run: ps -A --no-headers | wc -l
[10.10.10.193] out: 190
[10.10.10.193] out: 

=============================free_memory=============================
[10.10.10.193] run: free  | awk '{print $4}' | grep -v shared | head -n1
[10.10.10.193] out: 32524912
[10.10.10.193] out: 
```

最后，`fabric`模块将在执行所有任务后终止已建立的SSH会话并断开与两台机器的连接：

```py
Disconnecting from 10.10.10.140... done.
Disconnecting from 10.10.10.193... done.
```

请注意，我们可以重新设计之前的脚本，并将`discovery_commands`和`health_commands`作为Fabric任务，然后将它们包含在`get_system_health()`中。当我们执行`fab`命令时，我们将调用`get_system_health()`，它将执行另外两个函数；我们将得到与之前相同的输出。以下是修改后的示例脚本：

```py
#!/usr/bin/python __author__ = "Bassim Aly" __EMAIL__ = "basim.alyy@gmail.com"   from fabric.api import * from fabric.context_managers import * from pprint import pprint

env.hosts = [
  '10.10.10.140', # Ubuntu Machine
  '10.10.10.193', # CentOS Machine ]   env.user = "root" env.password = "access123"     def discovery_commands():
  discovery_commands = {
  "uptime": "uptime | awk '{print $3,$4}'",
  "hostname": "hostname",
  "kernel_release": "uname -r",
  "architecture": "uname -m",
  "internal_ip": "hostname -I",
  "external_ip": "curl -s ipecho.net/plain;echo",      }
  for operation, command in discovery_commands.iteritems():
  print("============================={0}=============================".format(operation))
  output = run(command)   def health_commands():
  health_commands = {
  "used_memory": "free  | awk '{print $3}' | grep -v free | head -n1",
  "free_memory": "free  | awk '{print $4}' | grep -v shared | head -n1",
  "cpu_usr_percentage": "mpstat | grep -A 1 '%usr' | tail -n1 | awk '{print $4}'",
  "number_of_process": "ps -A --no-headers | wc -l",
  "logged_users": "who",
  "top_load_average": "top -n 1 -b | grep 'load average:' | awk '{print $10 $11 $12}'",
  "disk_usage": "df -h| egrep 'Filesystem|/dev/sda*|nvme*'"    }
  for operation, command in health_commands.iteritems():
  print("============================={0}=============================".format(operation))
  output = run(command)   def get_system_health():
  discovery_commands()
  health_commands()
```

# Fabric的其他有用功能

Fabric还有其他有用的功能，如角色和上下文管理器。

# Fabric角色

Fabric可以为主机定义角色，并仅对角色成员运行任务。例如，我们可能有一堆数据库服务器，需要验证MySql服务是否正常运行，以及其他需要验证Apache服务是否正常运行的Web服务器。我们可以将这些主机分组到角色中，并根据这些角色执行函数：

```py
#!/usr/bin/python __author__ = "Bassim Aly" __EMAIL__ = "basim.alyy@gmail.com"   from fabric.api import *   env.hosts = [
  '10.10.10.140', # ubuntu machine
  '10.10.10.193', # CentOS machine
  '10.10.10.130', ]   env.roledefs = {
  'webapps': ['10.10.10.140','10.10.10.193'],
  'databases': ['10.10.10.130'], }   env.user = "root" env.password = "access123"   @roles('databases') def validate_mysql():
  output = run("systemctl status mariadb")     @roles('webapps') def validate_apache():
  output = run("systemctl status httpd") 
```

在前面的示例中，我们在设置`env.roledef`时使用了Fabric装饰器`roles`（从`fabric.api`导入）。然后，我们将webapp或数据库角色分配给每个服务器（将角色分配视为对服务器进行标记）。这将使我们能够仅在具有数据库角色的服务器上执行`validate_mysql`函数：

```py
# fab -f fabfile_roles.py validate_mysql:roles=databases
[10.10.10.130] Executing task 'validate_mysql'
[10.10.10.130] run: systemctl status mariadb
[10.10.10.130] out: ● mariadb.service - MariaDB database server
[10.10.10.130] out:    Loaded: loaded (/usr/lib/systemd/system/mariadb.service; enabled; vendor preset: disabled)
[10.10.10.130] out:    Active: active (running) since Sat 2018-04-07 19:47:35 EET; 1 day 2h ago
<output omitted>
```

# Fabric上下文管理器

在我们的第一个Fabric脚本`fabfile_first.py`中，我们有一个任务提示用户输入目录，然后切换到该目录并打印其内容。这是通过使用`;`来实现的，它将两个Linux命令连接在一起。但是，在其他操作系统上运行相同的命令并不总是有效。这就是Fabric上下文管理器发挥作用的地方。

上下文管理器在执行命令时维护目录状态。它通常通过`with`语句在Python中运行，并且在块内，您可以编写任何以前的Fabric操作。让我们通过一个示例来解释这个想法：

```py
from fabric.api import *
from fabric.context_managers import *   env.hosts = [
  '10.10.10.140', # ubuntu machine
  '10.10.10.193', # CentOS machine ]   env.user = "root" env.password = "access123"   def list_directory():
  with cd("/var/log"):
  run("ls")
```

在前面的示例中，首先我们在`fabric.context_managers`中全局导入了所有内容；然后，我们使用`cd`上下文管理器切换到特定目录。我们使用Fabric的`run()`操作在该目录上执行`ls`。这与在SSH会话中编写`cd /var/log ; ls`相同，但它提供了一种更Pythonic的方式来开发您的代码。

`with`语句可以嵌套。例如，我们可以用以下方式重写前面的代码：

```py
def list_directory_nested():
  with cd("/var/"):
  with cd("log"):
  run("ls")
```

另一个有用的上下文管理器是**本地更改目录**（**LCD**）。这与前面示例中的`cd`上下文管理器相同，但它在运行`fabfile`的本地机器上工作。我们可以使用它来将上下文更改为特定目录（例如，上传或下载文件到/从远程机器，然后自动切换回执行目录）：

```py
def uploading_file():
  with lcd("/root/"):
  put("VeryImportantFile.txt")
```

`prefix`上下文管理器将接受一个命令作为输入，并在`with`块内的任何其他命令之前执行它。例如，您可以在运行每个命令之前执行源文件或Python虚拟`env`包装器脚本来设置您的虚拟环境：

```py
def prefixing_commands():
  with prefix("source ~/env/bin/activate"):
  sudo('pip install wheel')
  sudo("pip install -r requirements.txt")
  sudo("python manage.py migrate")
```

实际上，这相当于在Linux shell中编写以下命令：

```py
source ~/env/bin/activate && pip install wheel
source ~/env/bin/activate && pip install -r requirements.txt
source ~/env/bin/activate && python manage.py migrate
```

最后一个上下文管理器是`shell_env(new_path, behavior='append')`，它可以修改包装命令的shell环境变量；因此，在该块内的任何调用都将考虑到修改后的路径：

```py
def change_shell_env():
  with shell_env(test1='val1', test2='val2', test3='val3'):
  run("echo $test1") #This command run on remote host
  run("echo $test2")
  run("echo $test3")
        local("echo $test1") #This command run on local host
```

请注意，在操作完成后，Fabric将将旧的环境恢复到原始状态。

# 摘要

Fabric是一个出色且强大的工具，可以自动化任务，通常在远程机器上执行。它与Python脚本很好地集成，可以轻松访问SSH套件。您可以为不同的任务开发许多fab文件，并将它们集成在一起，以创建包括部署、重启和停止服务器或进程在内的自动化工作流程。

在下一章中，我们将学习收集数据并为系统监控生成定期报告。
