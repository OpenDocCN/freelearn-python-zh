# 使用Subprocess模块

运行和生成新的系统进程对于想要自动化特定操作系统任务或在脚本中执行一些命令的系统管理员非常有用。Python提供了许多库来调用外部系统实用程序，并与生成的数据进行交互。最早创建的库是`OS`模块，它提供了一些有用的工具来调用外部进程，比如`os.system`，`os.spwan`和`os.popen*`。然而，它缺少一些基本功能，因此Python开发人员引入了一个新的库，`subprocess`，它可以生成新的进程，与进程发送和接收，并处理错误和返回代码。目前，官方Python文档建议使用`subprocess`模块来访问系统命令，Python实际上打算用它来替换旧的模块。

本章将涵盖以下主题：

+   `Popen()`子进程

+   读取`stdin`，`stdout`和`stderr`

+   子进程调用套件

# popen()子进程

`subprocess`模块只实现了一个类：`popen()`。这个类的主要用途是在系统上生成一个新的进程。这个类可以接受运行进程的额外参数，以及`popen()`本身的额外参数：

| **参数** | **含义** |
| --- | --- |
| `args` | 一个字符串，或者程序参数的序列。 |
| `bufsize` | 它作为`open()`函数的缓冲参数提供，用于创建`stdin`/`stdout`/`stderr`管道文件对象。 |
| `executable` | 要执行的替换程序。 |
| `stdin`，`stdout`，`stderr` | 这些分别指定了执行程序的标准输入、标准输出和标准错误文件句柄。 |
| `shell` | 如果为`True`，则命令将通过shell执行（默认为`False`）。在Linux中，这意味着在运行子进程之前调用`/bin/sh`。 |
| `cwd` | 在执行子进程之前设置当前目录。 |
| `env` | 定义新进程的环境变量。 |

现在，让我们专注于`args`。`popen()`命令可以接受Python列表作为输入，其中第一个元素被视为命令，后续元素被视为命令`args`，如下面的代码片段所示：

```py
import subprocess
print(subprocess.Popen("ifconfig"))
```

**脚本输出**![](../images/00147.jpeg)

从命令返回的输出直接打印到您的Python终端。

`ifconfig`是一个用于返回网络接口信息的Linux实用程序。对于Windows用户，您可以通过在cmd上使用`ipconfig`命令来获得类似的输出。

我们可以重写上面的代码，使用列表而不是字符串，如下面的代码片段所示：

```py
print(subprocess.Popen(["ifconfig"]))
```

使用这种方法允许您将额外的参数添加到主命令作为列表项：

```py
print(subprocess.Popen(["sudo", "ifconfig", "enp60s0:0", "10.10.10.2", "netmask", "255.255.255.0", "up"])) enp60s0:0: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
        inet 10.10.10.2  netmask 255.255.255.0  broadcast 10.10.10.255
        ether d4:81:d7:cb:b7:1e  txqueuelen 1000  (Ethernet)
        device interrupt 16  
```

请注意，如果您将上一个命令提供为字符串而不是列表，就像我们在第一个示例中所做的那样，命令将失败，如下面的屏幕截图所示。子进程`Popen()`期望在每个列表元素中有一个可执行名称，而不是其他任何参数。![](../images/00148.jpeg)

另一方面，如果您想使用字符串方法而不是列表，您可以将`shell`参数设置为`True`。这将指示`Popen()`在命令之前附加`/bin/sh`，因此命令将在其后执行所有参数：

```py
print(subprocess.Popen("sudo ifconfig enp60s0:0 10.10.10.2 netmask 255.255.255.0 up", shell=True)) 
```

您可以将`shell=True`视为生成一个shell进程并将命令与参数传递给它。这可以通过使用`split()`节省您几行代码，以便直接从外部系统接收命令并运行它。

`subprocess`使用的默认shell是`/bin/sh`。如果您使用其他shell，比如`tch`或`csh`，您可以在`executable`参数中定义它们。还要注意，作为shell运行命令可能会带来安全问题，并允许*安全注入*。指示您的代码运行脚本的用户可以添加`"; rm -rf /"`，导致可怕的事情发生。

此外，您可以使用`cwd`参数在运行命令之前将目录更改为特定目录。当您需要在对其进行操作之前列出目录的内容时，这将非常有用：

```py
import subprocess
print(subprocess.Popen(["cat", "interfaces"], cwd="/etc/network"))  
```

![](../images/00149.jpeg)Ansible有一个类似的标志叫做`chdir:`。此参数将用于playbook任务中，在执行之前更改目录。

# 读取标准输入(stdin)、标准输出(stdout)和标准错误(stderr)

生成的进程可以通过三个通道与操作系统通信：

1.  标准输入（stdin）

1.  标准输出（stdout）

1.  标准错误（stderr）

在子进程中，`Popen()`可以与三个通道交互，并将每个流重定向到外部文件，或者重定向到一个称为`PIPE`的特殊值。另一个方法叫做`communicate()`，用于从`stdout`读取和写入`stdin`。`communicate()`方法可以从用户那里获取输入，并返回标准输出和标准错误，如下面的代码片段所示：

```py
import subprocess
p = subprocess.Popen(["ping", "8.8.8.8", "-c", "3"], stdin=subprocess.PIPE, stdout=subprocess.PIPE) stdout, stderr = p.communicate() print("""==========The Standard Output is========== {}""".format(stdout))   print("""==========The Standard Error is========== {}""".format(stderr))
```

![](../images/00150.jpeg)

同样，您可以使用`communicate()`中的输入参数发送数据并写入进程：

```py
import subprocess
p = subprocess.Popen(["grep", "subprocess"], stdout=subprocess.PIPE, stdin=subprocess.PIPE) stdout,stderr = p.communicate(input=b"welcome to subprocess module\nthis line is a new line and doesnot contain the require string")   print("""==========The Standard Output is========== {}""".format(stdout))   print("""==========The Standard Error is========== {}""".format(stderr))
```

在脚本中，我们在`communicate()`中使用了`input`参数，它将数据发送到另一个子进程，该子进程将使用`grep`命令搜索子进程关键字。返回的输出将存储在`stdout`变量中：

![](../images/00151.jpeg)

验证进程成功执行的另一种方法是使用返回代码。当命令成功执行且没有错误时，返回代码将为`0`；否则，它将是大于`0`的整数值：

```py
import subprocess

def ping_destination(ip):   p = subprocess.Popen(['ping', '-c', '3'],
  stdout=subprocess.PIPE,
  stderr=subprocess.PIPE)
  stdout, stderr = p.communicate(input=ip)
  if p.returncode == 0:
  print("Host is alive")
  return True, stdout
    else:
  print("Host is down")
  return False, stderr
 while True:
    print(ping_destination(raw_input("Please enter the host:"))) 
```

脚本将要求用户输入一个IP地址，然后调用`ping_destination()`函数，该函数将针对IP地址执行`ping`命令。`ping`命令的结果（成功或失败）将返回到标准输出，并且`communicate()`函数将使用结果填充返回代码：

![](../images/00152.jpeg)

首先，我们测试了Google DNS IP地址。主机是活动的，并且命令将成功执行，返回代码`=0`。函数将返回`True`并打印`主机是活动的`。其次，我们使用了`HostNotExist`字符串进行测试。函数将返回`False`到主程序并打印`主机已关闭`。此外，它将打印返回给子进程的命令标准输出（`Name or service not known`）。

您可以使用`echo $?`来检查先前执行的命令的返回代码（有时称为退出代码）。

# 子进程调用套件

子进程模块提供了另一个函数，使进程生成比使用`Popen()`更安全。子进程`call()`函数等待被调用的命令/程序完成读取输出。它支持与`Popen()`构造函数相同的参数，如`shell`、`executable`和`cwd`，但这次，您的脚本将等待程序完成并填充返回代码，而无需`communicate()`。

如果您检查`call()`函数，您会发现它实际上是`Popen()`类的一个包装器，但具有一个`wait()`函数，它会在返回输出之前等待命令结束：

![](../images/00153.jpeg)

```py
import subprocess
subprocess.call(["ifconfig", "docker0"], stdout=subprocess.PIPE, stderr=None, shell=False) 
```

如果您希望为您的代码提供更多保护，可以使用`check_call()`函数。它与`call()`相同，但会对返回代码进行另一个检查。如果它等于`0`（表示命令已成功执行），则将返回输出。否则，它将引发一个带有返回退出代码的异常。这将允许您在程序流中处理异常：

```py
import subprocess

try:
  result = subprocess.check_call(["ping", "HostNotExist", "-c", "3"]) except subprocess.CalledProcessError:
  print("Host is not found") 
```

使用`call()`函数的一个缺点是，您无法像使用`Popen()`那样使用`communicate()`将数据发送到进程。

# 总结

在本章中，我们学习了如何在系统中运行和生成新进程，以及我们了解了这些生成的进程如何与操作系统通信。我们还讨论了子进程模块和`subprocess`调用。

在下一章中，我们将看到如何在远程主机上运行和执行命令。
