# Python脚本的并行执行

Python已成为网络自动化的*事实*标准。许多网络工程师已经每天使用它来自动化网络任务，从配置到操作，再到解决网络问题。在本章中，我们将讨论Python中的一个高级主题：挖掘Python的多进程特性，并学习如何使用它来加速脚本执行时间。

本章将涵盖以下主题：

+   Python代码在操作系统中的执行方式

+   Python多进程库

# 计算机如何执行您的Python脚本

这是您计算机的操作系统执行Python脚本的方式：

1.  当您在shell中键入`python <your_awesome_automation_script>.py`时，Python（作为一个进程运行）指示您的计算机处理器安排一个线程（这是处理的最小单位）：

![](../images/00121.jpeg)

1.  分配的线程将开始逐行执行您的脚本。线程可以做任何事情，包括与I/O设备交互，连接到路由器，打印输出，执行数学方程等等。

1.  一旦脚本达到**文件结束**（**EOF**），线程将被终止并返回到空闲池中，供其他进程使用。然后，脚本被终止。

在Linux中，您可以使用`#strace –p <pid>`来跟踪特定线程的执行。

您为脚本分配的线程越多（并且得到处理器或操作系统允许的线程越多），脚本运行得越快。实际上，有时线程被称为**工作者**或**从属**。

我有一种感觉，你脑海中有这样一个小想法：为什么我们不从所有核心中为Python脚本分配大量线程，以便快速完成工作呢？

如果没有特殊处理，将大量线程分配给一个进程的问题是**竞争条件**。操作系统将为您的进程（在本例中是Python进程）分配内存，以供运行时所有线程访问 - *同时*。现在，想象一下其中一个线程在另一个线程实际写入数据之前读取了一些数据！您不知道线程尝试访问共享数据的顺序；这就是竞争条件：

![](../images/00122.jpeg)

一种可用的解决方案是使线程获取锁。事实上，默认情况下，Python被优化为作为单线程进程运行，并且有一个叫做**全局解释器锁**（**GIL**）的东西。为了防止线程之间的冲突，GIL不允许多个线程同时执行Python代码。

但是，为什么不使用多个进程，而不是多个线程呢？

多进程的美妙之处，与多线程相比，就在于你不必担心由于共享数据而导致数据损坏。每个生成的进程都将拥有自己分配的内存，其他Python进程无法访问。这使我们能够同时执行并行任务：

![](../images/00123.jpeg)

此外，从Python的角度来看，每个进程都有自己的GIL。因此，在这里没有资源冲突或竞争条件。

# Python多进程库

`multiprocessing`模块是Python的标准库，随Python二进制文件一起提供，并且从Python 2.6版本开始可用。还有`threading`模块，它允许您生成多个线程，但它们都共享相同的内存空间。多进程比线程具有更多的优势。其中之一是每个进程都有独立的内存空间，并且可以利用多个CPU和核心。

# 开始使用多进程

首先，您需要为Python脚本导入模块：

```py
import multiprocessing as mp
```

然后，用Python函数包装您的代码；这将允许进程针对此函数并将其标记为并行执行。

假设我们有连接到路由器并使用`netmiko`库在其上执行命令的代码，并且我们希望并行连接到所有设备。这是一个样本串行代码，将连接到每个设备并执行传递的命令，然后继续第二个设备，依此类推：

```py
from netmiko import ConnectHandler
from devices import R1, SW1, SW2, SW3, SW4

nodes = [R1, SW1, SW2, SW3, SW4]   for device in nodes:
  net_connect = ConnectHandler(**device)
  output = net_connect.send_command("show run")
  print output
```

Python文件`devices.py`创建在与我们的脚本相同的目录中，并以`dictionary`格式包含每个设备的登录详细信息和凭据：

```py
  R1 = {"device_type": "cisco_ios_ssh",
      "ip": "10.10.88.110",
      "port": 22,
      "username": "admin",
      "password": "access123",
      }

SW1 = {"device_type": "cisco_ios_ssh",
       "ip": "10.10.88.111",
       "port": 22,
       "username": "admin",
       "password": "access123",
       }

SW2 = {"device_type": "cisco_ios_ssh",
       "ip": "10.10.88.112",
       "port": 22,
       "username": "admin",
       "password": "access123",
       }

SW3 = {"device_type": "cisco_ios_ssh",
       "ip": "10.10.88.113",
       "port": 22,
       "username": "admin",
       "password": "access123",
       }

SW4 = {"device_type": "cisco_ios_ssh",
       "ip": "10.10.88.114",
       "port": 22,
       "username": "admin",
       "password": "access123",
       } 
```

现在，如果我们想要改用多进程模块，我们需要重新设计脚本并将代码移动到一个函数下；然后，我们将分配与设备数量相等的进程数（一个进程将连接到一个设备并执行命令），并将进程的目标设置为执行此函数：

```py
  from netmiko import ConnectHandler
from devices import R1, SW1, SW2, SW3, SW4
import multiprocessing as mp
from datetime import datetime

nodes = [R1, SW1, SW2, SW3, SW4]    def connect_to_dev(device):    net_connect = ConnectHandler(**device)
  output = net_connect.send_command("show run")
  print output

processes = []   start_time = datetime.now() for device in nodes:
  print("Adding Process to the list")
  processes.append(mp.Process(target=connect_to_dev, args=[device]))   print("Spawning the Process") for p in processes:
  p.start()   print("Joining the finished process to the main truck") for p in processes:
  p.join()   end_time = datetime.now() print("Script Execution tooks {}".format(end_time - start_time))   
```

在前面的例子中，适用以下内容：

+   我们将`multiprocess`模块导入为`mp`。模块中最重要的类之一是`Process`，它将我们的`netmiko connect`函数作为目标参数。此外，它接受将参数传递给目标函数。

+   然后，我们遍历我们的节点，并为每个设备创建一个进程，并将该进程附加到进程列表中。

+   模块中可用的`start()`方法用于生成并启动进程执行。

+   最后，脚本执行时间通过从脚本结束时间减去脚本开始时间来计算。

在幕后，执行主脚本的主线程将开始分叉与设备数量相等的进程。每个进程都针对一个函数，同时在所有设备上执行`show run`，并将输出存储在一个变量中，互不影响。

这是Python中进程的一个示例视图：

![](../images/00124.jpeg)

现在，当您执行完整的代码时，还需要做一件事。您需要将分叉的进程连接到主线程/主线程，以便顺利完成程序的执行：

```py
for p in processes:
  p.join()
```

在前面的例子中使用的`join()`方法与原始的字符串方法`join()`无关；它只是用来将进程连接到主线程。

# 进程之间的通信

有时，您将有一个需要在运行时与其他进程传递或交换信息的进程。多进程模块有一个`Queue`类，它实现了一个特殊的列表，其中一个进程可以插入和消耗数据。在这个类中有两个可用的方法：`get()`和`put()`。`put()`方法用于向`Queue`添加数据，而从队列获取数据则通过`get()`方法完成。在下一个示例中，我们将使用`Queue`来将数据从子进程传递到父进程：

```py
import multiprocessing
from netmiko import ConnectHandler
from devices import R1, SW1, SW2, SW3, SW4
from pprint import pprint

nodes = [R1, SW1, SW2, SW3, SW4]   def connect_to_dev(device, mp_queue):
  dev_id = device['ip']
  return_data = {}   net_connect = ConnectHandler(**device)   output = net_connect.send_command("show run")   return_data[dev_id] = output
    print("Adding the result to the multiprocess queue")
  mp_queue.put(return_data)   mp_queue = multiprocessing.Queue() processes = []   for device in nodes:
  p = multiprocessing.Process(target=connect_to_dev, args=[device, mp_queue])
  print("Adding Process to the list")
  processes.append(p)
  p.start()   for p in processes:
  print("Joining the finished process to the main truck")
  p.join()   results = [] for p in processes:
  print("Moving the result from the queue to the results list")
  results.append(mp_queue.get())   pprint(results)
```

在前面的例子中，适用以下内容：

+   我们从`multiprocess`模块中导入了另一个名为`Queue()`的类，并将其实例化为`mp_queue`变量。

+   然后，在进程创建过程中，我们将此队列作为参数与设备一起附加，因此每个进程都可以访问相同的队列并能够向其写入数据。

+   `connect_to_dev()`函数连接到每个设备并在终端上执行`show run`命令，然后将输出写入共享队列。

请注意，在将其添加到共享队列之前，我们将输出格式化为字典项`{ip:<command_output>}`，并使用`mp_queue.put()`将其添加到共享队列中。

+   在进程完成执行并加入主（父）进程之后，我们使用`mp_queue.get()`来检索结果列表中的队列项，然后使用`pprint`来漂亮地打印输出。

# 概要

在本章中，我们学习了Python多进程库以及如何实例化和并行执行Python代码。

在下一章中，我们将学习如何准备实验室环境并探索自动化选项以加快服务器部署速度。
