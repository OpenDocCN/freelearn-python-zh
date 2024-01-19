# 第十一章：生成系统报告和系统监控

收集数据并生成定期系统报告是任何系统管理员的重要任务，并自动化这些任务可以帮助我们及早发现问题，以便为其提供解决方案。在本章中，我们将看到一些经过验证的方法，用于从服务器自动收集数据并将这些数据生成为正式报告。我们将学习如何使用 Python 和 Ansible 管理新用户和现有用户。此外，我们还将深入研究日志分析和监控系统**关键绩效指标**（**KPI**）。您可以安排监视脚本定期运行。

本章将涵盖以下主题：

+   从 Linux 收集数据

+   在 Ansible 中管理用户

# 从 Linux 收集数据

本机 Linux 命令提供有关当前系统状态和健康状况的有用数据。然而，这些 Linux 命令和实用程序都专注于从系统的一个方面获取数据。我们需要利用 Python 模块将这些详细信息返回给管理员并生成有用的系统报告。

我们将报告分为两部分。第一部分是使用`platform`模块获取系统的一般信息，而第二部分是探索 CPU 和内存等硬件资源。

我们将首先利用 Python 内置库中的`platform`模块。`platform`模块包含许多方法，可用于获取 Python 所在系统的详细信息：

```py
import platform
system = platform.system() print(system)
```

![](img/00157.jpeg)

在 Windows 机器上运行相同的脚本将产生不同的输出，反映当前的系统。因此，当我们在 Windows PC 上运行它时，我们将从脚本中获得`Windows`作为输出：

![](img/00158.gif)

另一个有用的函数是`uname()`，它与 Linux 命令(`uname -a`)执行相同的工作：检索机器的主机名、架构和内核，但以结构化格式呈现，因此您可以通过引用其索引来匹配任何值：

```py
import platform
from pprint import pprint
uname = platform.uname() pprint(uname)
```

![](img/00159.jpeg)

第一个值是系统类型，我们使用`system()`方法获取，第二个值是当前机器的主机名。

您可以使用 PyCharm 中的自动完成功能来探索并列出`platform`模块中的所有可用函数；您可以通过按下*CTRL* + *Q*来检查每个函数的文档：

![](img/00160.jpeg)

设计脚本的第二部分是使用 Linux 文件提供的信息来探索 Linux 机器中的硬件配置。请记住，CPU、内存和网络信息可以从`/proc/`下访问；我们将读取此信息并使用 Python 中的标准`open()`函数进行访问。您可以通过阅读和探索`/proc/`来获取有关可用资源的更多信息。

**脚本：**

这是导入`platform`模块的第一步。这仅适用于此任务：

```py
#!/usr/bin/python __author__ = "Bassim Aly" __EMAIL__ = "basim.alyy@gmail.com"   import platform
```

此片段包含了此练习中使用的函数；我们将设计两个函数 - `check_feature()`和`get_value_from_string()`：

```py
def check_feature(feature,string):
  if feature in string.lower():
  return True
  else:
  return False   def get_value_from_string(key,string):
  value = "NONE"
  for line in string.split("\n"):
  if key in line:
  value = line.split(":")[1].strip()
  return value
```

最后，以下是 Python 脚本的主体，其中包含获取所需信息的 Python 逻辑：

```py
cpu_features = [] with open('/proc/cpuinfo') as cpus:
  cpu_data = cpus.read()
  num_of_cpus = cpu_data.count("processor")
  cpu_features.append("Number of Processors: {0}".format(num_of_cpus))
  one_processor_data = cpu_data.split("processor")[1]
 print one_processor_data
    if check_feature("vmx",one_processor_data):
  cpu_features.append("CPU Virtualization: enabled")
  if check_feature("cpu_meltdown",one_processor_data):
  cpu_features.append("Known Bugs: CPU Metldown ")
  model_name = get_value_from_string("model name ",one_processor_data)
  cpu_features.append("Model Name: {0}".format(model_name))    cpu_mhz = get_value_from_string("cpu MHz",one_processor_data)
  cpu_features.append("CPU MHz: {0}".format((cpu_mhz)))   memory_features = [] with open('/proc/meminfo') as memory:
  memory_data = memory.read()
  total_memory = get_value_from_string("MemTotal",memory_data).replace(" kB","")
  free_memory = get_value_from_string("MemFree",memory_data).replace(" kB","")
  swap_memory = get_value_from_string("SwapTotal",memory_data).replace(" kB","")
  total_memory_in_gb = "Total Memory in GB: {0}".format(int(total_memory)/1024)
  free_memory_in_gb = "Free Memory in GB: {0}".format(int(free_memory)/1024)
  swap_memory_in_gb = "SWAP Memory in GB: {0}".format(int(swap_memory)/1024)
  memory_features = [total_memory_in_gb,free_memory_in_gb,swap_memory_in_gb]  
```

此部分用于打印从上一部分获得的信息：

```py
print("============System Information============")   print(""" System Type: {0} Hostname: {1} Kernel Version: {2} System Version: {3} Machine Architecture: {4} Python version: {5} """.format(platform.system(),
  platform.uname()[1],
  platform.uname()[2],
  platform.version(),
  platform.machine(),
  platform.python_version()))     print("============CPU Information============") print("\n".join(cpu_features))     print("============Memory Information============") print("\n".join(memory_features))
```

在上述示例中，执行了以下步骤：

1.  首先，我们打开了`/proc/cpuinfo`并读取了它的内容，然后将结果存储在`cpu_data`中。

1.  通过使用`count()`字符串函数计算文件中处理器的数量可以找到。

1.  然后，我们需要获取每个处理器可用的选项和特性。为此，我们只获取了一个处理器条目（因为它们通常是相同的），并将其传递给`check_feature()`函数。该方法接受我们想要在一个参数中搜索的特性，另一个是处理器数据，如果处理器数据中存在该特性，则返回`True`。

1.  处理器数据以键值对的形式可用。因此，我们设计了`get_value_from_string()`方法，它接受键名，并通过迭代处理器数据来搜索其对应的值；然后，我们将在每个返回的键值对上使用`:`分隔符进行拆分，以获取值。

1.  所有这些值都使用`append()`方法添加到`cpu_feature`列表中。

1.  然后，我们重复了相同的操作，使用内存信息获取总内存、空闲内存和交换内存。

1.  接下来，我们使用平台的内置方法，如`system()`、`uname()`和`python_version()`，来获取有关系统的信息。

1.  最后，我们打印了包含上述信息的报告。

脚本输出如下截图所示：

![](img/00161.jpeg)表示生成的数据的另一种方法是利用我们在第五章中使用的`matplotlib`库，以便随时间可视化数据。

# 通过电子邮件发送生成的数据

在前一节生成的报告中，提供了系统当前资源的良好概述。但是，我们可以调整脚本并扩展其功能，以便通过电子邮件发送所有细节给我们。这对于**网络运营中心**（**NoC**）团队非常有用，他们可以根据特定事件（硬盘故障、高 CPU 或丢包）从受监控系统接收电子邮件。Python 有一个名为`smtplib`的内置库，它利用**简单邮件传输协议**（**SMTP**）负责与邮件服务器发送和接收电子邮件。

这要求您的计算机上有本地电子邮件服务器，或者您使用其中一个免费的在线电子邮件服务，如 Gmail 或 Outlook。在本例中，我们将使用 SMTP 登录到[`www.gmail.com`](http://www.gmail.com)，并使用我们的数据发送电子邮件。

话不多说，我们将修改我们的脚本，并为其添加 SMTP 支持。

我们将所需的模块导入 Python。同样，`smtplib`和`platform`对于这个任务是必需的：

```py
#!/usr/bin/python __author__ = "Bassim Aly" __EMAIL__ = "basim.alyy@gmail.com"   import smtplib
imp        ort platform
```

这是函数的一部分，包含`check_feature()`和`get_value_from_string()`函数：

```py
def check_feature(feature,string):
  if feature in string.lower():
  return True
  else:
  return False   def get_value_from_string(key,string):
  value = "NONE"
  for line in string.split("\n"):
  if key in line:
  value = line.split(":")[1].strip()
  return value
```

最后，Python 脚本的主体如下，包含了获取所需信息的 Python 逻辑：

```py
cpu_features = [] with open('/proc/cpuinfo') as cpus:
  cpu_data = cpus.read()
  num_of_cpus = cpu_data.count("processor")
  cpu_features.append("Number of Processors: {0}".format(num_of_cpus))
  one_processor_data = cpu_data.split("processor")[1]
 if check_feature("vmx",one_processor_data):
  cpu_features.append("CPU Virtualization: enabled")
  if check_feature("cpu_meltdown",one_processor_data):
  cpu_features.append("Known Bugs: CPU Metldown ")
  model_name = get_value_from_string("model name ",one_processor_data)
  cpu_features.append("Model Name: {0}".format(model_name))    cpu_mhz = get_value_from_string("cpu MHz",one_processor_data)
  cpu_features.append("CPU MHz: {0}".format((cpu_mhz)))   memory_features = [] with open('/proc/meminfo') as memory:
  memory_data = memory.read()
  total_memory = get_value_from_string("MemTotal",memory_data).replace(" kB","")
  free_memory = get_value_from_string("MemFree",memory_data).replace(" kB","")
  swap_memory = get_value_from_string("SwapTotal",memory_data).replace(" kB","")
  total_memory_in_gb = "Total Memory in GB: {0}".format(int(total_memory)/1024)
  free_memory_in_gb = "Free Memory in GB: {0}".format(int(free_memory)/1024)
  swap_memory_in_gb = "SWAP Memory in GB: {0}".format(int(swap_memory)/1024)
  memory_features = [total_memory_in_gb,free_memory_in_gb,swap_memory_in_gb]   Data_Sent_in_Email = "" Header = """From: PythonEnterpriseAutomationBot <basim.alyy@gmail.com> To: To Administrator <basim.alyy@gmail.com> Subject: Monitoring System Report   """ Data_Sent_in_Email += Header
Data_Sent_in_Email +="============System Information============"   Data_Sent_in_Email +=""" System Type: {0} Hostname: {1} Kernel Version: {2} System Version: {3} Machine Architecture: {4} Python version: {5} """.format(platform.system(),
  platform.uname()[1],
  platform.uname()[2],
  platform.version(),
  platform.machine(),
  platform.python_version())     Data_Sent_in_Email +="============CPU Information============\n" Data_Sent_in_Email +="\n".join(cpu_features)     Data_Sent_in_Email +="\n============Memory Information============\n" Data_Sent_in_Email +="\n".join(memory_features)  
```

最后，我们需要为变量赋一些值，以便正确连接到`gmail`服务器：

```py
fromaddr = 'yyyyyyyyyyy@gmail.com' toaddrs  = 'basim.alyy@gmail.com' username = 'yyyyyyyyyyy@gmail.com' password = 'xxxxxxxxxx' server = smtplib.SMTP('smtp.gmail.com:587') server.ehlo() server.starttls() server.login(username,password)   server.sendmail(fromaddr, toaddrs, Data_Sent_in_Email) server.quit()
```

在前面的示例中，适用以下内容：

1.  第一部分与原始示例相同，但是不是将数据打印到终端，而是将其添加到`Data_Sent_in_Email`变量中。

1.  `Header`变量代表包含发件人地址、收件人地址和电子邮件主题的电子邮件标题。

1.  我们使用`smtplib`模块内的`SMTP()`类来连接到公共 Gmail SMTP 服务器并协商 TTLS 连接。这是连接到 Gmail 服务器时的默认方法。我们将 SMTP 连接保存在`server`变量中。

1.  现在，我们使用`login()`方法登录到服务器，最后，我们使用`sendmail()`函数发送电子邮件。`sendmail()`接受三个参数：发件人、收件人和电子邮件正文。

1.  最后，我们关闭与服务器的连接：

**脚本输出**

![](img/00162.jpeg)

# 使用时间和日期模块

很好；到目前为止，我们已经能够通过电子邮件发送从我们的服务器生成的自定义数据。然而，由于网络拥塞或邮件系统故障等原因，生成的数据和电子邮件的传递时间之间可能存在时间差异。因此，我们不能依赖电子邮件将传递时间与实际事件时间相关联。

因此，我们将使用 Python 的`datetime`模块来跟踪监视系统上的当前时间。该模块可以以许多属性格式化时间，如年、月、日、小时和分钟。

除此之外，`datetime`模块中的`datetime`实例实际上是 Python 中的一个独立对象（如 int、string、boolean 等）；因此，它在 Python 内部有自己的属性。

要将`datetime`对象转换为字符串，可以使用`strftime()`方法，该方法作为创建的对象内的属性可用。此外，它提供了一种通过以下指令格式化时间的方法：

| **指令** | **含义** |
| --- | --- |
| `%Y` | 返回年份，从 0001 到 9999 |
| `%m` | 返回月份 |
| `%d` | 返回月份的日期 |
| `%H` | 返回小时数，0-23 |
| `%M` | 返回分钟数，0-59 |
| `%S` | 返回秒数，0-59 |

因此，我们将调整我们的脚本，并将以下片段添加到代码中：

```py
from datetime import datetime
time_now = datetime.now() time_now_string = time_now.strftime("%Y-%m-%d %H:%M:%S")
Data_Sent_in_Email += "====Time Now is {0}====\n".format(time_now_string) 
```

首先，我们从`datetime`模块中导入了`datetime`类。然后，我们使用`datetime`类和`now()`函数创建了`time_now`对象，该函数返回正在运行系统上的当前时间。最后，我们使用`strftime()`，带有一个指令，以特定格式格式化时间并将其转换为字符串以进行打印（记住，该对象有一个`datetime`对象）。

脚本的输出如下：

![](img/00163.jpeg)

# 定期运行脚本

脚本的最后一步是安排脚本在一定时间间隔内运行。这可以是每天、每周、每小时或在特定时间。这可以通过 Linux 系统上的`cron`作业来完成。`cron`用于安排重复事件，如清理目录、备份数据库、旋转日志，或者你能想到的其他任何事情。

要查看当前计划的作业，使用以下命令：

```py
crontab -l
```

编辑`crontab`，使用`-e`开关。如果这是你第一次运行`cron`，系统会提示你使用你喜欢的编辑器（`nano`或`vi`）。

典型的`crontab`由五个星号组成，每个星号代表一个时间条目：

| **字段** | **值** |
| --- | --- |
| 分钟 | 0-59 |
| 小时 | 0-23 |
| 月份的日期 | 1-31 |
| 月份 | 1-12 |
| 星期几 | 0-6（星期日-星期六） |

例如，如果你需要安排一个工作在每周五晚上 9 点运行，你将使用以下条目：

```py
0 21 * * 5 /path/to/command
```

如果你需要每天凌晨 12 点执行一个命令（例如备份），使用以下`cron`作业：

```py
0 0 * * * /path/to/command
```

此外，你可以安排`cron`在*每个*特定的间隔运行。例如，如果你需要每`5`分钟运行一次作业，使用这个`cron`作业：

```py
*/5 * * * * /path/to/command
```

回到我们的脚本；我们可以安排它在每天上午 7:30 运行：

```py
30 7 * * * /usr/bin/python /root/Send_Email.py
```

最后，记得在退出之前保存`cron`作业。

最好提供 Linux 的完整命令路径，而不是相对路径，以避免任何潜在问题。

# 在 Ansible 中管理用户

现在，我们将讨论如何在不同系统中管理用户。

# Linux 系统

Ansible 提供了强大的用户管理模块，用于管理系统上的不同任务。我们有一个专门讨论 Ansible 的章节（第十三章，*系统管理的 Ansible*），但在本章中，我们将探讨其在管理公司基础设施上管理用户帐户的能力。

有时，公司允许所有用户访问 root 权限，以摆脱用户管理的麻烦；这在安全和审计方面不是一个好的解决方案。最佳实践是给予正确的用户正确的权限，并在用户离开公司时撤销这些权限。

Ansible 提供了一种无与伦比的方式来管理多台服务器上的用户，可以通过密码或无密码（SSH 密钥）访问。

在创建 Linux 系统中的用户时，还有一些其他需要考虑的事项。用户必须有一个 shell（如 Bash、CSH、ZSH 等）才能登录到服务器。此外，用户应该有一个主目录（通常在`/home`下）。最后，用户必须属于一个确定其特权和权限的组。

我们的第一个示例将是在远程服务器上使用临时命令创建一个带有 SSH 密钥的用户。密钥源位于`ansible` tower，而我们在`all`服务器上执行命令：

```py
ansible all -m copy -a "src=~/id_rsa dest=~/.ssh/id_rsa mode=0600"
```

第二个示例是使用 Playbook 创建用户：

```py
--- - hosts: localhost
  tasks:
    - name: create a username
      user:
        name: bassem
        password: "$crypted_value$"
        groups:
          - root
        state: present
        shell: /bin/bash
        createhome: yes
  home: /home/bassem
```

让我们来看一下任务的参数：

+   在我们的任务中，我们使用了一个包含多个参数的用户模块，比如`name`，用于设置用户的用户名。

+   第二个参数是`password`，用于设置用户的密码，但是以加密格式。您需要使用`mkpasswd`命令，该命令会提示您输入密码并生成哈希值。

+   `groups`是用户所属的组列表；因此，用户将继承权限。您可以在此字段中使用逗号分隔的值。

+   `state`用于告诉 Ansible 用户是要创建还是删除。

+   您可以在`shell`参数中定义用于远程访问的用户 shell。

+   `createhome`和`home`是用于指定用户主目录位置的参数。

另一个参数是`ssh_key_file`，用于指定 SSH 文件名。此外，`ssh_key_passphrase`将指定 SSH 密钥的密码。

# 微软 Windows

Ansible 提供了`win_user`模块来管理本地 Windows 用户帐户。在创建活动目录域或 Microsoft SQL 数据库（`mssql`）上的用户或在普通 PC 上创建默认帐户时，这非常有用。以下示例将创建一个名为`bassem`的用户，并为其设置密码`access123`。不同之处在于密码是以明文而不是加密值给出的，就像在基于 Unix 的系统中一样：

```py
- hosts: localhost
  tasks:
    - name: create user on windows machine
      win_user:
        name: bassem
        password: 'access123'
  password_never_expires: true
  account_disabled: no
  account_locked: no
  password_expired: no
  state: present
        groups:
          - Administrators
          - Users
```

`password_never_expires`参数将防止 Windows 在特定时间后使密码过期；这在创建管理员和默认帐户时非常有用。另一方面，如果将`password_expired`设置为`yes`，将要求用户在首次登录时输入新密码并更改密码。

`groups`参数将用户添加到列出的值或逗号分隔的组列表中。这将取决于`groups_action`参数，可以是`add`、`replace`或`remove`。

最后，状态将告诉 Ansible 应该对用户执行什么操作。此参数可以是`present`、`absent`或`query`。

# 总结

在本章中，我们学习了如何从 Linux 机器收集数据和报告，并使用时间和日期模块通过电子邮件进行警报。我们还学习了如何在 Ansible 中管理用户。

在下一章中，我们将学习如何使用 Python 连接器与 DBMS 进行交互。
