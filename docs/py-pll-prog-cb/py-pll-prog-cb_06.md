# 第六章：分布式 Python

本章将介绍一些重要的 Python 模块，用于分布式计算。特别是，我们将描述`socket`模块，它允许您通过客户端-服务器模型实现简单的分布式应用程序。

然后，我们将介绍 Celery 模块，这是一个强大的 Python 框架，用于管理分布式任务。最后，我们将描述`Pyro4`模块，它允许您调用在不同进程中使用的方法，可能在不同的机器上。

在本章中，我们将介绍以下内容：

+   介绍分布式计算

+   使用 Python 的 socket 模块

+   使用 Celery 进行分布式任务管理

+   使用`Pyro4`进行远程方法调用（RMI）

# 介绍分布式计算

*并行*和*分布式计算*是类似的技术，旨在增加特定任务的处理能力。通常，这些方法用于解决需要大量计算能力的问题。

当问题被分成许多小部分时，问题的各个部分可以同时由许多处理器计算。这允许问题上的处理能力比单个处理器提供的要多。

并行处理和分布式处理的主要区别在于，并行配置在单个系统内包含许多处理器，而分布式配置利用许多计算机的处理能力。

让我们看看其他的区别：

| **并行处理** | **分布式处理** |
| --- | --- |
| 并行处理具有提供可靠处理能力并具有非常低延迟的优势。 | 分布式处理在处理器的基础上并不是非常高效，因为数据必须通过网络传输，而不是通过单个系统的内部连接传输。 |
| 通过将所有处理能力集中在一个系统中，可以最大程度地减少由于数据传输而导致的速度损失。 | 由于数据传输会产生限制处理能力的瓶颈，因此每个处理器提供的处理能力远远低于并行系统中的任何处理器。 |
| 唯一的真正限制是系统中集成的处理器数量。 | 由于分布式系统中处理器数量没有实际上限，因此系统几乎可以无限扩展。 |

然而，在计算机应用的背景下，习惯上区分本地架构和分布式架构：

| **本地架构** | **分布式架构** |
| --- | --- |
| 所有组件都在同一台机器上。 | 应用程序和组件可以驻留在由网络连接的不同节点上。 |

使用分布式计算的优势主要在于程序的并发使用、数据的集中化以及处理负载的分布，但这些优势都伴随着更大的复杂性，特别是在各个组件之间的通信方面。

# 分布式应用程序的类型

分布式应用程序可以根据分布程度进行分类：

+   **客户端-服务器应用程序**

+   **多级应用程序**

# 客户端-服务器应用程序

只有两个级别，操作完全在服务器上进行。例如，我们可以提到经典的静态或动态网站。实现这些类型应用的工具是网络套接字，可以用多种语言进行编程，包括 C、C++、Java，当然还有 Python。

术语*客户端-服务器系统*指的是一个网络架构，其中客户端计算机或客户端终端通常连接到服务器以使用某项服务；例如，与其他客户端共享某些硬件/软件资源，或依赖底层协议架构。

# 客户端-服务器架构

客户端-服务器架构是一个实现处理和数据分布的系统。架构的中心元素是服务器。服务器可以从逻辑和物理角度来考虑。从物理角度来看，服务器是专门用于运行软件服务器的机器。

从逻辑上看，服务器是软件。服务器作为逻辑进程，为扮演请求者或客户端角色的其他进程提供服务。通常情况下，服务器直到客户端请求结果之前不会将结果发送给请求者。

区分客户端和服务器的一个特征是客户端可以与服务器启动事务，而服务器永远不能主动与客户端启动事务：

![](img/f9df2fb0-1a78-4826-b494-5e3b43effbf0.png)

客户端-服务器架构

事实上，客户端的具体任务是启动事务，请求特定服务，通知服务完成，并从服务器接收结果，如前图所示。

# 客户端-服务器通信

客户端和服务器之间的通信可以使用各种机制——从地理网络到本地网络，直到操作系统级别的应用程序之间的通信服务。此外，客户端-服务器架构必须独立于客户端和服务器之间存在的物理连接方法。

还应该注意的是，客户端-服务器进程不必驻留在物理上分离的系统上。事实上，服务器进程和客户端进程可以驻留在同一计算平台上。

在数据管理的背景下，客户端-服务器架构的主要目标是允许客户端应用程序访问服务器管理的数据。服务器（在逻辑上理解为软件）通常运行在远程系统上（例如，在另一个城市或本地网络上）。

因此，客户端-服务器应用程序通常与分布式处理相关联。

# TCP/IP 客户端-服务器架构

TCP/IP 连接在两个应用程序之间建立了点对点的连接。这种连接的两端由 IP 地址标记，IP 地址标识了工作站，而端口号使得可以在同一工作站上连接到独立应用程序的多个连接。

一旦连接建立，协议可以在其上交换数据，底层的 TCP/IP 协议负责将这些数据分成数据包，从连接的一端发送到另一端。特别是，TCP 协议负责组装和拆卸数据包，以及管理握手来保证连接的可靠性，而 IP 协议负责传输单个数据包和选择最佳的路由来沿着网络传输数据包。

这种机制是 TCP/IP 协议稳健性的基础，而 TCP/IP 协议的发展又是军事领域（ARPANET）发展的原因之一。

各种现有的标准应用程序（如 Web 浏览、文件传输和电子邮件）使用标准化的应用程序协议，如 HTTP、FTP、POP3、IMAP 和 SMTP。

每个特定的客户端-服务器应用程序必须定义和应用自己的专有应用程序协议。这可能涉及以固定大小的数据块交换数据（这是最简单的解决方案）。

# 多级应用程序

有更多级别可以减轻服务器的处理负载。实际上，被细分的是服务器端的功能，而客户端部分的特性基本保持不变，其任务是托管应用程序界面。这种架构的一个例子是三层模型，其结构分为三层或级别：

+   前端或演示层或界面

+   中间层或应用逻辑

+   后端或数据层或持久数据管理

这种命名方式通常用于 Web 应用程序。更一般地，可以将任何软件应用程序分为三个级别，如下所示：

+   **表示层**（**PL**）：这是数据的可视化部分（例如用户界面所需的模块和输入控件）。

+   **业务逻辑层**（**BLL**）：这是应用程序的主要部分，独立于用户可用的演示方法并保存在档案中，定义了各种实体及其关系。

+   **数据访问层**（**DAL**）：其中包含管理持久数据所需的一切（基本上是数据库管理系统）。

本章将介绍 Python 提出的一些分布式架构的解决方案。我们将首先描述`socket`模块，然后使用它来实现一些基本的客户端-服务器模型的示例。

# 使用 Python 套接字模块

套接字是一种软件对象，允许在远程主机（通过网络）或本地进程之间发送和接收数据，例如**进程间通信**（**IPC**）。

套接字是在伯克利作为**BSD Unix**项目的一部分发明的。它们基于 Unix 文件的输入和输出管理模型。事实上，打开、读取、写入和关闭套接字的操作与 Unix 文件的管理方式相同，但需要考虑的区别是用于通信的有用参数，如地址、端口号和协议。

套接字技术的成功和传播与互联网的发展息息相关。事实上，套接字与互联网的结合使得任何类型的机器之间的通信以及分散在世界各地的机器之间的通信变得非常容易（至少与其他系统相比是如此）。

# 准备就绪

Python 套接字模块公开了用于使用**BSD**（**Berkeley Software Distribution**的缩写）套接字接口进行网络通信的低级 C API。

该模块包括`Socket`类，其中包括管理以下任务的主要方法：

+   `socket([family [, type [, protocol]]])`: 使用以下参数构建套接字：

+   `family`地址，可以是`AF_INET（默认）`，`AF_INET6`，或`AF_UNIX`

+   `type`套接字，可以是`SOCK_STREAM（默认）`，`SOCK_DGRAM`，或者其他`"SOCK_"`常量之一

+   `protocol`号码（通常为零）

+   `gethostname()`: 返回机器的当前 IP 地址。

+   `accept()`: 返回以下一对值（`conn`和`address`），其中`conn`是套接字类型对象（用于在连接上发送/接收数据），而`address`是连接到连接的另一端的套接字的地址。

+   `bind(address)`: 将套接字与服务器的`address`关联。

该方法历史上接受`AF_INET`地址的一对参数，而不是单个元组。

+   `close()`: 提供选项，一旦与客户端的通信结束，就可以清理连接。套接字被关闭并由垃圾收集器收集。

+   `connect(address)`: 将远程套接字连接到地址。`address`格式取决于地址族。

# 如何做到...

在下面的示例中，服务器正在监听默认端口，并通过 TCP/IP 连接，客户端向服务器发送连接建立的日期和时间。

以下是`server.py`的服务器实现：

1.  导入相关的 Python 模块：

```py
import socket
import time
```

1.  使用给定的地址、套接字类型和协议号创建新的套接字：

```py
serversocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
```

1.  获取本地机器名称（`host`）：

```py
host=socket.gethostname()
```

1.  设置`port`号码：

```py
port=9999
```

1.  将套接字连接（绑定）到`host`和`port`：

```py
serversocket.bind((host,port))
```

1.  监听套接字的连接。`5`的参数指定了队列中的最大连接数。最大值取决于系统（通常为`5`），最小值始终为`0`：

```py
serversocket.listen(5)
```

1.  建立连接：

```py
while True:
```

1.  然后，接受连接。返回值是一对（`conn`，`address`），其中`conn`是用于发送和接收数据的新`socket`对象，`address`是与套接字关联的地址。一旦接受，将创建一个新的套接字，并且它将有自己的标识符。这个新的套接字只用于这个特定的客户端：

```py
clientsocket,addr=serversocket.accept()
```

1.  打印连接的地址和端口：

```py
print ("Connected with[addr],[port]%s"%str(addr))
```

1.  评估`currentTime`：

```py
currentTime=time.ctime(time.time())+"\r\n"
```

1.  以下语句将数据发送到套接字，并返回发送的字节数：

```py
clientsocket.send(currentTime.encode('ascii'))
```

1.  以下语句表示套接字关闭（即通信通道）；套接字上的所有后续操作都将失败。当套接字被拒绝时，它们会自动关闭，但始终建议使用`close()`操作关闭它们：

```py
clientsocket.close()
```

客户端（`client.py`）的代码如下：

1.  导入`socket`库：

```py
import socket
```

1.  然后创建`socket`对象：

```py
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
```

1.  获取本地机器名称（`host`）：

```py
host=socket.gethostname()
```

1.  设置`port`号码：

```py
port=9999
```

1.  建立到`host`和`port`的连接：

```py
s.connect((host,port))
```

可以接收的最大字节数不超过 1024 字节：（`tm=s.recv(1024)`）。

1.  现在，关闭连接并最终打印连接到服务器的连接时间：

```py
s.close()
print ("Time connection server:%s"%tm.decode('ascii'))
```

# 工作原理...

客户端和服务器分别创建它们的套接字，并在一个端口上监听它们。客户端向服务器发出连接请求。应该注意，我们可以有两个不同的端口号，因为一个可能只专用于出站流量，另一个可能只专用于入站流量。这取决于主机配置。

实际上，客户端的本地端口不一定与服务器的远程端口相符。服务器接收请求，如果接受，将创建一个新连接。现在，客户端和服务器通过专门为数据套接字连接的数据流创建的虚拟通道进行通信。

与第一阶段提到的一致，服务器创建数据套接字，因为第一个套接字专门用于处理请求。因此，可能有许多客户端使用服务器为它们创建的数据套接字与服务器进行通信。TCP 协议是面向连接的，这意味着当不再需要通信时，客户端会将此通知服务器，并关闭连接。

要运行示例，请执行服务器：

```py
C:\>python server.py 
```

然后，在不同的 Windows 终端中执行客户端：

```py
C:\>python client.py
```

客户端端的结果应报告地址（`addr`）并报告`port`已连接：

```py
Connected with[addr],port
```

但是，在服务器端，结果应该如下：

```py
Time connection server:Sun Mar 31 20:59:38 2019
```

# 还有更多...

通过对先前的代码进行小改动，我们可以创建一个简单的客户端-服务器应用程序进行文件传输。服务器实例化套接字并等待来自客户端的连接实例。一旦连接到服务器，客户端开始数据传输。

要传输的数据在`mytext.txt`文件中，按字节复制并通过调用`conn.send`函数发送到服务器。服务器然后接收数据并将其写入第二个文件`received.txt`。

`client2.py`的源代码如下：

```py
import socket
s =socket.socket()
host=socket.gethostname()
port=60000
s.connect((host,port))
s.send('HelloServer!'.encode())
with open('received.txt','wb') as f:
 print ('file opened')
 while True :
 print ('receiving data...')
 data=s.recv(1024)
 if not data:
 break
 print ('Data=>',data.decode())
 f.write(data)
f.close()
print ('Successfully get the file')
s.close()
print ('connection closed')
```

以下是`client.py`的源代码：

```py
import socket
port=60000
s =socket.socket()
host=socket.gethostname()
s.bind((host,port))
s.listen(15)
print('Server listening....')
while True :
 conn,addr=s.accept()
 print ('Got connection from',addr)
 data=conn.recv(1024)
 print ('Server received',repr(data.decode()))
 filename='mytext.txt'
 f =open(filename,'rb')
 l =f.read(1024)
 while True:
 conn.send(l)
 print ('Sent',repr(l.decode()))
 l =f.read(1024)
 f.close()
 print ('Done sending')
 conn.send('->Thank you for connecting'.encode())
 conn.close()
```

# 套接字类型

我们可以区分以下三种套接字类型，其特点是连接模式：

+   **流套接字**：这些是面向连接的套接字，它们基于可靠的协议，如 TCP 或 SCTP。

+   **数据报套接字**：这些套接字不是面向连接的（无连接）套接字，而是基于快速但不可靠的 UDP 协议。

+   **原始套接字**（原始 IP）：传输层被绕过，头部在应用层可访问。

# 流套接字

我们将只看到这种类型的套接字。由于它们基于 TCP 等传输层协议，它们保证可靠、全双工和面向连接的通信，具有可变长度的字节流。

通过这个套接字进行通信包括以下阶段：

1.  **套接字的创建**：客户端和服务器创建各自的套接字，并且服务器在端口上监听它们。由于服务器可以与不同客户端（但也可能是同一个客户端）创建多个连接，因此它需要一个队列来处理各种请求。

1.  **连接请求**：客户端请求与服务器建立连接。请注意，我们可以有不同的端口号，因为一个可能只分配给出站流量，另一个只分配给入站流量。这取决于主机配置。基本上，客户端的本地端口不一定与服务器的远程端口相符。服务器接收请求，如果接受，将创建一个新连接。在图中，客户端套接字的端口是`8080`，而服务器套接字的端口是`80`。

1.  **通信**：现在，客户端和服务器通过一个虚拟通道进行通信，介于客户端套接字和一个新的套接字（服务器端）之间，专门为此连接的数据流创建：一个数据套接字。正如在第一阶段中提到的，服务器创建数据套接字，因为第一个数据套接字专门用于处理请求。因此，可能有许多客户端与服务器通信，每个客户端都有服务器专门为其创建的数据套接字。

1.  **连接的关闭**：由于 TCP 是一种面向连接的协议，当不再需要通信时，客户端会通知服务器，服务器会释放数据套接字。

通过流套接字进行通信的阶段如下图所示：

![](img/4a04807a-ed49-44eb-9f21-2346ea59f835.png)

流套接字阶段

# 另请参阅

有关 Python 套接字的更多信息，请访问[`docs.python.org/3/howto/sockets.html`](https://docs.python.org/3/howto/sockets.html)。

# 使用 Celery 进行分布式任务管理

*Celery*是一个 Python 框架，通过遵循面向对象的中间件方法来管理分布式任务。其主要特点是处理许多小任务并将它们分发到许多计算节点上。最终，每个任务的结果将被重新处理，以组成整体解决方案。

要使用 Celery，需要一个消息代理。这是一个独立的（与 Celery 无关）软件组件，具有中间件的功能，用于向分布式任务工作者发送和接收消息。

事实上，消息代理（也称为消息中间件）处理通信网络中消息的交换：这种中间件的寻址方案不再是点对点类型，而是面向消息的寻址。

消息代理的参考架构，用于管理消息的交换，基于所谓的发布/订阅范式，如下所示：

![](img/081ba83c-57b4-44c6-aaa5-89f6d9e0989c.png)

消息代理架构

Celery 支持许多类型的代理。但是，更完整的是 RabbitMQ 和 Redis。

# 准备就绪

要安装 Celery，请使用`pip`安装程序，如下所示：

```py
C:\>pip install celery
```

然后，必须安装消息代理。有几种选择可用，但是对于我们的示例，建议从以下链接安装 RabbitMQ：[`www.rabbitmq.com/download.html`](http://www.rabbitmq.com/download.html)。

RabbitMQ 是一个实现 **高级消息队列协议** (**AMQP**) 的消息导向中间件。RabbitMQ 服务器是用 Erlang 编程语言编写的，因此在安装它之前，您需要从 [`www.erlang.org/download.html`](http://www.erlang.org/download.html) 下载并安装 Erlang。涉及的步骤如下：

1.  要检查 `celery` 的安装，首先启动消息代理（例如 RabbitMQ）。然后，输入以下内容：

```py
C:\>celery --version
```

1.  以下输出表示 `celery` 版本：

```py
4.2.2 (Windowlicker)
```

接下来，让我们了解如何使用 `celery` 模块创建和调用任务。

`celery` 提供以下两种方法来调用任务：

+   `apply_async(args[, kwargs[, ...]])`：这发送一个任务消息。

+   `delay(*args, **kwargs)`：这是发送任务消息的快捷方式，但不支持执行选项。

`delay` 方法更容易使用，因为它被调用为**常规函数**：`task.delay(arg1, arg2, kwarg1='x', kwarg2='y')`。然而，对于 `apply_async`，语法是 `task.apply_async (args=[arg1,arg2] kwargs={'kwarg1':'x','kwarg2': 'y'})`。

# Windows 设置

要在 Windows 环境中使用 Celery，必须执行以下过程：

1.  转到系统属性 | 环境变量 | 用户或系统变量 | 新建。

1.  设置以下值：

+   变量名：`FORKED_BY_MULTIPROCESSING`

+   变量值：`1`

进行此设置的原因是因为 Celery 依赖于 `billiard` 包 ([`github.com/celery/billiard`](https://github.com/celery/billiard))，它使用 `FORKED_BY_MULTIPROCESSING` 变量。

有关 Celery 在 Windows 上的设置的更多信息，请阅读 [`www.distributedpython.com/2018/08/21/celery-4-windows/`](https://www.distributedpython.com/2018/08/21/celery-4-windows/)。

# 如何做...

这里的任务是两个数字的和。为了执行这个简单的任务，我们必须组成 `addTask.py` 和 `addTask_main.py` 脚本文件：

1.  对于 `addTask.py`，开始导入 Celery 框架如下：

```py
from celery import Celery
```

1.  然后，定义任务。在我们的示例中，任务是两个数字的和：

```py
app = Celery('tasks', broker='amqp://guest@localhost//')
@app.task
def add(x, y):
 return x + y
```

1.  现在，导入之前定义的 `addTask.py` 文件到 `addtask_main.py` 中：

```py
import addTask
```

1.  然后，调用 `addTask.py` 执行两个数字的和：

```py
if __name__ == '__main__':
 result = addTask.add.delay(5,5)
```

# 工作原理...

要使用 Celery，首先要做的是运行 RabbitMQ 服务，然后执行 Celery 工作者服务器（即 `addTask.py` 文件脚本），方法是输入以下内容：

```py
C:\>celery -A addTask worker --loglevel=info
```

输出如下：

```py
Microsoft Windows [Versione 10.0.17134.648]
(c) 2018 Microsoft Corporation. Tutti i diritti sono riservati.

C:\Users\Giancarlo>cd C:\Users\Giancarlo\Desktop\Python Parallel Programming CookBook 2nd edition\Python Parallel Programming NEW BOOK\chapter_6 - Distributed Python\esempi

C:\Users\Giancarlo\Desktop\Python Parallel Programming CookBook 2nd edition\Python Parallel Programming NEW BOOK\chapter_6 - Distributed Python\esempi>celery -A addTask worker --loglevel=info

 -------------- celery@pc-giancarlo v4.2.2 (windowlicker)
---- **** -----
--- * *** * -- Windows-10.0.17134 2019-04-01 21:32:37
-- * - **** ---
- ** ---------- [config]
- ** ---------- .> app: tasks:0x1deb8f46940
- ** ---------- .> transport: amqp://guest:**@localhost:5672//
- ** ---------- .> results: disabled://
- *** --- * --- .> concurrency: 4 (prefork)
-- ******* ---- .> task events: OFF (enable -E to monitor tasks in this worker)
--- ***** -----
 -------------- [queues]
 .> celery exchange=celery(direct) key=celery
[tasks]
 . addTask.add

[2019-04-01 21:32:37,650: INFO/MainProcess] Connected to amqp://guest:**@127.0.0.1:5672//
[2019-04-01 21:32:37,745: INFO/MainProcess] mingle: searching for neighbors
[2019-04-01 21:32:39,353: INFO/MainProcess] mingle: all alone
[2019-04-01 21:32:39,479: INFO/SpawnPoolWorker-2] child process 10712 calling self.run()
[2019-04-01 21:32:39,512: INFO/SpawnPoolWorker-3] child process 10696 calling self.run()
[2019-04-01 21:32:39,536: INFO/MainProcess] celery@pc-giancarlo ready.
[2019-04-01 21:32:39,551: INFO/SpawnPoolWorker-1] child process 6084 calling self.run()
[2019-04-01 21:32:39,615: INFO/SpawnPoolWorker-4] child process 2080 calling self.run()
```

然后，使用 Python 启动第二个脚本：

```py
C:\>python addTask_main.py
```

最后，在第一个命令提示符中，结果应该如下所示：

```py
[2019-04-01 21:33:00,451: INFO/MainProcess] Received task: addTask.add[6fc350a9-e925-486c-bc41-c239ebd96041]
[2019-04-01 21:33:00,452: INFO/SpawnPoolWorker-2] Task addTask.add[6fc350a9-e925-486c-bc41-c239ebd96041] succeeded in 0.0s: 10
```

正如您所看到的，结果是 `10`。让我们专注于第一个脚本 `addTask.py`：在代码的前两行中，我们创建了一个使用 RabbitMQ 服务代理的 `Celery` 应用实例：

```py
from celery import Celery
app = Celery('addTask', broker='amqp://guest@localhost//')
```

`Celery` 函数的第一个参数是当前模块的名称（`addTask.py`），第二个是代理键盘参数；这表示用于连接代理（RabbitMQ）的 URL。

现在，让我们介绍要完成的任务。

每个任务必须使用 `@app.task` 注释（即装饰器）添加；装饰器帮助 `Celery` 确定哪些函数可以在任务队列中调度。

在装饰器之后，我们创建工作者可以执行的任务：这将是一个执行两个数字之和的简单函数：

```py
@app.task
def add(x, y):
 return x + y
```

在第二个脚本 `addTask_main.py` 中，我们使用 `delay()` 方法调用我们的任务：

```py
if __name__ == '__main__':
 result = addTask.add.delay(5,5)
```

让我们记住，这种方法是 `apply_async()` 方法的快捷方式，它可以更好地控制任务的执行。

# 还有更多...

Celery 的使用非常简单。可以通过以下命令执行：

```py
Usage: celery <command> [options]
```

这里，选项如下：

```py
positional arguments:
 args

optional arguments:
 -h, --help             show this help message and exit
 --version              show program's version number and exit

Global Options:
 -A APP, --app APP
 -b BROKER, --broker BROKER
 --result-backend RESULT_BACKEND
 --loader LOADER
 --config CONFIG
 --workdir WORKDIR
 --no-color, -C
 --quiet, -q
```

主要命令如下：

```py
+ Main:
| celery worker
| celery events
| celery beat
| celery shell
| celery multi
| celery amqp

+ Remote Control:
| celery status

| celery inspect --help
| celery inspect active
| celery inspect active_queues
| celery inspect clock
| celery inspect conf [include_defaults=False]
| celery inspect memdump [n_samples=10]
| celery inspect memsample
| celery inspect objgraph [object_type=Request] [num=200 [max_depth=10]]
| celery inspect ping
| celery inspect query_task [id1 [id2 [... [idN]]]]
| celery inspect registered [attr1 [attr2 [... [attrN]]]]
| celery inspect report
| celery inspect reserved
| celery inspect revoked
| celery inspect scheduled
| celery inspect stats

| celery control --help
| celery control add_consumer <queue> [exchange [type [routing_key]]]
| celery control autoscale [max [min]]
| celery control cancel_consumer <queue>
| celery control disable_events
| celery control election
| celery control enable_events
| celery control heartbeat
| celery control pool_grow [N=1]
| celery control pool_restart
| celery control pool_shrink [N=1]
| celery control rate_limit <task_name> <rate_limit (e.g., 5/s | 5/m | 
5/h)>
| celery control revoke [id1 [id2 [... [idN]]]]
| celery control shutdown
| celery control terminate <signal> [id1 [id2 [... [idN]]]]
| celery control time_limit <task_name> <soft_secs> [hard_secs]

+ Utils:
| celery purge
| celery list
| celery call
| celery result
| celery migrate
| celery graph
| celery upgrade

+ Debugging:
| celery report
| celery logtool

+ Extensions:
| celery flower
-------------------------------------------------------------
```

Celery 协议可以通过使用 Webhooks（[`developer.github.com/webhooks/`](https://developer.github.com/webhooks/)）在任何语言中实现。

# 另请参阅

+   有关 Celery 的更多信息，请访问[`www.celeryproject.org/`](http://www.celeryproject.org/)。

+   推荐的消息代理（[`en.wikipedia.org/wiki/Message_broker`](https://en.wikipedia.org/wiki/Message_broker)）是 RabbitMQ（[`en.wikipedia.org/wiki/RabbitMQ`](https://en.wikipedia.org/wiki/RabbitMQ)）或 Redis（[`en.wikipedia.org/wiki/Redis`](https://en.wikipedia.org/wiki/Redis)）。此外，还有 MongoDB（[`en.wikipedia.org/wiki/MongoDB`](https://en.wikipedia.org/wiki/MongoDB)）、Beanstalk、Amazon SQS（[`en.wikipedia.org/wiki/Amazon_Simple_Queue_Service`](https://en.wikipedia.org/wiki/Amazon_Simple_Queue_Service)）、CouchDB（[`en.wikipedia.org/wiki/Apache_CouchDB`](https://en.wikipedia.org/wiki/Apache_CouchDB)）和 IronMQ（[`www.iron.io/mq`](https://www.iron.io/mq)）。

# 使用 Pyro4 的 RMI

**Pyro**是**Python Remote Objects**的缩写。它的工作原理与 Java 的**RMI**（远程方法调用）完全相同，允许调用远程对象的方法（属于不同进程），就像对象是本地的一样（属于调用运行的同一进程）。

在面向对象的系统中使用 RMI 机制，可以在项目中获得统一性和对称性的重要优势，因为这种机制使得可以使用相同的概念工具对分布式进程之间的交互进行建模。

从下图中可以看出，`Pyro4`使对象以客户端/服务器的方式分布；这意味着`Pyro4`系统的主要部分可以从客户端调用者切换到远程对象，后者被调用来执行一个函数：

![](img/1782750b-b3f5-4ac1-8274-f01d46cff0ec.png)

RMI

需要注意的是，在远程调用过程中，始终存在两个不同的部分：一个客户端和一个接受并执行客户端调用的服务器。

# 准备工作

管理这种分布式方式的整个方法由`Pyro4`提供。要安装最新版本的`Pyro4`，请使用`pip`安装程序（这里使用 Windows 安装），并添加以下命令：

```py
C:\>pip install Pyro4
```

我们将使用`pyro_server.py`和`pyro_client.py`代码来完成这个示例。

# 如何做...

在这个例子中，我们将看到如何使用`Pyro4`中间件构建和使用简单的客户端-服务器通信。客户端的代码是`pyro_server.py`：

1.  导入`Pyro4`库：

```py
import Pyro4
```

1.  定义包含`welcomeMessage()`方法的`Server`类：

```py
class Server(object):
 @Pyro4.expose
 def welcomeMessage(self, name):
 return ("Hi welcome " + str (name))
```

请注意，装饰器`@Pyro4.expose`表示前面的方法将是远程可访问的。

1.  `startServer`函数包含了启动服务器所使用的所有指令：

```py
def startServer():
```

1.  接下来，构建`Server`类的`server`实例：

```py
server = Server()
```

1.  然后，定义`Pyro4`守护程序：

```py
daemon = Pyro4.Daemon()
```

1.  要执行此脚本，我们必须运行一个`Pyro4`语句来定位名字服务器：

```py
ns = Pyro4.locateNS()
```

1.  将对象服务器注册为*Pyro 对象*；它只会在 Pyro 守护程序内部知道：

```py
uri = daemon.register(server)
```

1.  现在，我们可以在名字服务器中注册对象服务器的名称：

```py
ns.register("server", uri)
```

1.  该函数以调用守护进程的`requestLoop`方法结束。这启动了服务器的事件循环，并等待调用：

```py
print("Ready. Object uri =", uri)
daemon.requestLoop()
```

1.  最后，通过`main`程序调用`startServer`：

```py
if __name__ == "__main__":
 startServer()
```

以下是客户端的代码（`pyro_client.py`）：

1.  导入`Pyro4`库：

```py
import Pyro4
```

1.  `Pyro4` API 使开发人员能够以透明的方式分发对象。在这个例子中，客户端脚本发送请求到服务器程序，以执行`welcomeMessage()`方法：

```py
uri = input("What is the Pyro uri of the greeting object? ").strip()
name = input("What is your name? ").strip()
```

1.  然后，创建远程调用：

```py
server = Pyro4.Proxy("PYRONAME:server")
```

1.  最后，客户端调用服务器，打印一条消息：

```py
print(server.welcomeMessage(name))
```

# 它是如何工作的...

上述示例由两个主要函数组成：`pyro_server.py`和`pyro_client.py`。

在`pyro_server.py`中，`Server`类对象提供`welcomeMessage()`方法，返回与客户端会话中插入的名称相等的字符串：

```py
class Server(object):
 @Pyro4.expose
 def welcomeMessage(self, name):
 return ("Hi welcome " + str (name))
```

`Pyro4`使用守护对象将传入调用分派给适当的对象。服务器必须创建一个管理其所有实例的守护进程。每个服务器都有一个守护进程，它知道服务器提供的所有 Pyro 对象：

```py
 daemon = Pyro4.Daemon()
```

至于`pyro_client.py`函数，首先执行远程调用并创建一个`Proxy`对象。特别是，`Pyro4`客户端使用代理对象将方法调用转发到远程对象，然后将结果传递回调用代码：

```py
server = Pyro4.Proxy("PYRONAME:server")
```

为了执行客户端-服务器连接，我们需要运行一个`Pyro4`名称服务器。在命令提示符中，输入以下内容：

```py
C:\>python -m Pyro4.naming
```

之后，您将看到以下消息：

```py
Not starting broadcast server for localhost.
NS running on localhost:9090 (127.0.0.1)
Warning: HMAC key not set. Anyone can connect to this server!
URI = PYRO:Pyro.NameServer@localhost:9090
```

前面的消息意味着名称服务器正在您的网络中运行。最后，我们可以在两个单独的 Windows 控制台中启动服务器和客户端脚本：

1.  要运行`pyro_server.py`，只需输入以下内容：

```py
C:\>python pyro_server.py
```

1.  之后，您将看到类似于这样的内容：

```py
Ready. Object uri = PYRO:obj_76046e1c9d734ad5b1b4f6a61ee77425@localhost:63269
```

1.  然后，输入以下内容运行客户端：

```py
C:\>python pyro_client.py
```

1.  将打印出以下消息：

```py
What is your name? 
```

1.  插入一个名称（例如，`Ruvika`）：

```py
What is your name? Ruvika
```

1.  将显示以下欢迎消息：

```py
Hi welcome Ruvika
```

# 还有更多...

`Pyro4`的功能之一是创建对象拓扑。例如，假设我们想要构建一个遵循链式拓扑结构的分布式架构，如下所示：

![](img/7f93a4ab-70f7-4ecb-8a0b-f5633c11bad1.png)

使用 Pyro4 链接对象

客户端向**服务器 1**发出请求，然后将请求转发到**服务器 2**，然后调用**服务器 3**。当**服务器 3**调用**服务器 1**时，链式调用结束。

# 实现链式拓扑

使用`Pyro4`实现链式拓扑，我们需要实现一个`chain`对象和`client`和`server`对象。`Chain`类允许通过处理输入消息并重建请求应该发送到的服务器地址来将调用重定向到下一个服务器。

还要注意，在这种情况下，`@Pyro4.expose`装饰器允许公开类（`chainTopology.py`）的所有方法：

```py
import Pyro4

@Pyro4.expose
class Chain(object):
 def __init__(self, name, next_server):
 self.name = name
 self.next_serverName = next_server
 self.next_server = None

 def process(self, message):
 if self.next_server is None:
 self.next_server = Pyro4.core.Proxy("PYRONAME:example.\
 chainTopology." + self.next_serverName)
```

如果链路关闭（最后一次调用是从`server_chain_3.py`到`server_chain_1.py`），则会打印出关闭消息：

```py
 if self.name in message:
 print("Back at %s;the chain is closed!" % self.name)
 return ["complete at " + self.name]
```

如果链中有下一个元素，则会打印出转发消息：

```py
 else:
 print("%s forwarding the message to the object %s" %\ 
 (self.name, self.next_serverName))
 message.append(self.name)
 result = self.next_server.process(message)
 result.insert(0, "passed on from " + self.name)
 return result
```

接下来是客户端的源代码（`client_chain.py`）：

```py
import Pyro4

obj = Pyro4.core.Proxy("PYRONAME:example.chainTopology.1")
print("Result=%s" % obj.process(["hello"]))
```

接下来是链中第一个服务器的源代码（即`server_1`），它是从客户端（`server_chain_1.py`）调用的。在这里，导入了相关的库。请注意，之前描述的`chainTopology.py`文件的导入：

```py
import Pyro4
import chainTopology
```

还要注意，服务器的源代码只有当前链和下一个链服务器的定义不同：

```py
current_server= "1"
next_server = "2"
```

其余代码行定义了与链中下一个元素的通信：

```py
servername = "example.chainTopology." + current_server
daemon = Pyro4.core.Daemon()
obj = chainTopology.Chain(current_server, next_server)
uri = daemon.register(obj)
ns = Pyro4.locateNS()
ns.register(servername, uri)
print("server_%s started " % current_server)
daemon.requestLoop()
```

要执行此示例，首先运行`Pyro4`名称服务器：

```py
C:\>python -m Pyro4.naming
Not starting broadcast server for localhost.
NS running on localhost:9090 (127.0.0.1)
Warning: HMAC key not set. Anyone can connect to this server!
URI = PYRO:Pyro.NameServer@localhost:9090
```

在三个不同的终端中运行三个服务器，分别输入它们（这里使用 Windows 终端）：

第一个服务器（`server_chain_1.py`）在第一个终端中：

```py
C:\>python server_chain_1.py
```

然后是第二个服务器（`server_chain_2.py`）在第二个终端中：

```py
C:\>python server_chain_2.py
```

最后，第三个服务器（`server_chain_3.py`）在第三个终端中：

```py
C:\>python server_chain_3.py
```

然后，从另一个终端运行`client_chain.py`脚本：

```py
C:\>python client_chain.py
```

这是在命令提示符中显示的输出：

```py
Result=['passed on from 1','passed on from 2','passed on from 3','complete at 1']
```

在返回任务完成的三个服务器之间传递转发请求后，将显示前面的消息。

此外，我们可以关注对象服务器在请求转发到链中的下一个对象时的行为（参见开始消息下方的消息）：

1.  **`server_1`**已启动，并将以下消息转发到**`server_2`**：

```py
server_1 started
1 forwarding the message to the object 2
```

1.  `server_2`将以下消息转发到`server_3`：

```py
server_2 started
2 forwarding the message to the object 3
```

1.  `server_3`将以下消息转发给`server_1`：

```py
server_3 started
3 forwarding the message to the object 1
```

1.  最后，消息返回到起始点（也就是`server_1`），链路关闭：

```py
server_1 started
1 forwarding the message to the object 2
Back at 1; the chain is closed!
```

# 另请参阅

`Pyro4`文档可在[`buildmedia.readthedocs.org/media/pdf/pyro4/stable/pyro4.pdf`](https://buildmedia.readthedocs.org/media/pdf/pyro4/stable/pyro4.pdf)上找到。

其中包含了 4.75 版本的描述和一些应用示例。
