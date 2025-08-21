# 第十三章：多进程-当单个 CPU 核心不够用时

在上一章中，我们讨论了影响性能的因素以及一些提高性能的方法。这一章实际上可以看作是性能提示列表的扩展。在本章中，我们将讨论多进程模块，这是一个使您的代码非常容易在多个 CPU 核心甚至多台机器上运行的模块。这是一个绕过前一章中讨论的**全局解释器锁**（**GIL**）的简单方法。

总之，本章将涵盖：

+   本地多进程

+   远程多进程

+   进程之间的数据共享和同步

# 多线程与多进程

在本书中，我们还没有真正涵盖多线程，但您可能以前看到过多线程代码。多线程和多进程之间的最大区别在于，多线程中的所有内容仍然在单个进程中执行。这实际上将性能限制在单个 CPU 核心。它实际上甚至限制了您的性能，因为代码必须处理 CPython 的 GIL 限制。

### 注意

GIL 是 Python 用于安全内存访问的全局锁。关于性能，它在第十二章中有更详细的讨论，*性能-跟踪和减少内存和 CPU 使用情况*。

为了说明多线程代码并不总是有助于性能，并且实际上可能比单线程代码稍慢，请看这个例子：

```py
import datetime
import threading

def busy_wait(n):
    while n > 0:
        n -= 1

if __name__ == '__main__':
    n = 10000000
    start = datetime.datetime.now()
    for _ in range(4):
        busy_wait(n)
    end = datetime.datetime.now()
    print('The single threaded loops took: %s' % (end - start))

    start = datetime.datetime.now()
    threads = []
    for _ in range(4):
        thread = threading.Thread(target=busy_wait, args=(n,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    end = datetime.datetime.now()
    print('The multithreaded loops took: %s' % (end - start))
```

使用 Python 3.5，它具有新的改进的 GIL 实现（在 Python 3.2 中引入），性能相当可比，但没有改进：

```py
# python3 test_multithreading.py
The single threaded loops took: 0:00:02.623443
The multithreaded loops took: 0:00:02.597900

```

使用仍然具有旧 GIL 的 Python 2.7，单线程变体的性能要好得多：

```py
# python2 test_multithreading.py
The single threaded loops took: 0:00:02.010967
The multithreaded loops took: 0:00:03.924950

```

从这个测试中，我们可以得出结论，Python 2 在某些情况下更快，而 Python 3 在其他情况下更快。你应该从中得出的结论是，没有性能原因特别选择 Python 2 还是 Python 3。只需注意，Python 3 在大多数情况下至少与 Python 2 一样快，如果不是这种情况，很快就会得到解决。

无论如何，对于 CPU 绑定的操作，线程不提供任何性能优势，因为它在单个处理器核心上执行。但是对于 I/O 绑定的操作，`threading`库确实提供了明显的好处，但在这种情况下，我建议尝试`asyncio`。`threading`的最大问题是，如果其中一个线程阻塞，主进程也会阻塞。

`multiprocessing`库提供了一个与`threading`库非常相似的 API，但是利用多个进程而不是多个线程。优点是 GIL 不再是问题，可以利用多个处理器核心甚至多台机器进行处理。

为了说明性能差异，让我们重复使用`multiprocessing`模块而不是`threading`进行测试：

```py
import datetime
import multiprocessing

def busy_wait(n):
    while n > 0:
        n -= 1

if __name__ == '__main__':
    n = 10000000
    start = datetime.datetime.now()

    processes = []
    for _ in range(4):
        process = multiprocessing.Process(
            target=busy_wait, args=(n,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    end = datetime.datetime.now()
    print('The multiprocessed loops took: %s' % (end - start))
```

运行时，我们看到了巨大的改进：

```py
# python3 test_multiprocessing.py
The multiprocessed loops took: 0:00:00.671249

```

请注意，这是在四核处理器上运行的，这就是为什么我选择了四个进程。`multiprocessing`库默认为`multiprocessing.cpu_count()`，它计算可用的 CPU 核心数，但该方法未考虑 CPU 超线程。这意味着在我的情况下它会返回 8，这就是为什么我将其硬编码为 4 的原因。

### 注意

重要的是要注意，因为`multiprocessing`库使用多个进程，代码需要从子进程中导入。结果是`multiprocessing`库无法在 Python 或 IPython shell 中工作。正如我们将在本章后面看到的那样，IPython 有自己的多进程处理方式。

# 超线程与物理 CPU 核心

在大多数情况下，超线程非常有用并提高了性能，但当您真正最大化 CPU 使用率时，通常最好只使用物理处理器数量。为了演示这如何影响性能，我们将再次运行上一节中的测试。这次使用 1、2、4、8 和 16 个进程来演示它如何影响性能。幸运的是，`multiprocessing`库有一个很好的`Pool`类来为我们管理进程：

```py
import sys
import datetime
import multiprocessing

def busy_wait(n):
    while n > 0:
        n -= 1

if __name__ == '__main__':
    n = 10000000
    start = datetime.datetime.now()
    if sys.argv[-1].isdigit():
        processes = int(sys.argv[-1])
    else:
        print('Please specify the number of processes')
        print('Example: %s 4' % ' '.join(sys.argv))
        sys.exit(1)

    with multiprocessing.Pool(processes=processes) as pool:
        # Execute the busy_wait function 8 times with parameter n
        pool.map(busy_wait, [n for _ in range(8)])

    end = datetime.datetime.now()
    print('The multithreaded loops took: %s' % (end - start))
```

池代码使得启动一组工作进程和处理队列变得更加简单。在这种情况下，我们使用了`map`，但还有其他几个选项，如`imap`，`map_async`，`imap_unordered`，`apply`，`apply_async`，`starmap`和`starmap_async`。由于这些方法与同名的`itertools`方法工作方式非常相似，因此不会为所有这些方法提供具体示例。

但现在，测试不同数量的进程：

```py
# python3 test_multiprocessing.py 1
The multithreaded loops took: 0:00:05.297707
# python3 test_multiprocessing.py 2
The multithreaded loops took: 0:00:02.701344
# python3 test_multiprocessing.py 4
The multithreaded loops took: 0:00:01.477845
# python3 test_multiprocessing.py 8
The multithreaded loops took: 0:00:01.579218
# python3 test_multiprocessing.py 16
The multithreaded loops took: 0:00:01.595239

```

您可能没有预料到这些结果，但这正是超线程的问题所在。一旦单个进程实际上使用了 CPU 核心的 100%，进程之间的任务切换实际上会降低性能。由于只有`4`个物理核心，其他`4`个核心必须争夺处理器核心上的任务。这场争斗需要时间，这就是为什么`4`个进程版本比`8`个进程版本稍快的原因。此外，调度效果也可以在使用`1`和`2`个核心的运行中看到。如果我们看单核版本，我们会发现它花了`5.3`秒，这意味着`4`个核心应该在`5.3 / 4 = 1.325`秒内完成，而实际上花了`1.48`秒。`2`核版本也有类似的效果，`2.7 / 2 = 1.35`秒，仍然比`4`核版本快。

如果您真的需要处理 CPU 绑定问题的性能，那么匹配物理 CPU 核心是最佳解决方案。如果您不希望始终最大化所有核心的使用，那么我建议将其保留为默认设置，因为超线程在其他情况下确实具有一些性能优势。

但这一切取决于您的用例，确切的方法是测试您特定情况的唯一方法：

+   磁盘 I/O 绑定？单个进程很可能是您最好的选择。

+   CPU 绑定？物理 CPU 核心数量是您最好的选择。

+   网络 I/O 绑定？从默认值开始，如果需要，进行调整。

+   没有明显的限制，但需要许多并行进程？也许您应该尝试`asyncio`而不是`multiprocessing`。

请注意，创建多个进程在内存和打开文件方面并不是免费的，而您可以拥有几乎无限数量的协程，但对于进程来说并非如此。根据您的操作系统配置，它可能在您甚至达到一百之前就达到最大值，即使您达到这些数字，CPU 调度也将成为瓶颈。

# 创建一个工作进程池

创建一个工作进程的处理池通常是一个困难的任务。您需要注意调度作业，处理队列，处理进程，以及最困难的部分是在进程之间处理同步而不会产生太多开销。

然而，使用`multiprocessing`，这些问题已经得到解决。您只需创建一个具有给定进程数的进程池，并在需要时添加任务即可。以下是`map`操作符的多进程版本的示例，并演示了处理不会使应用程序停滞：

```py
import time
import multiprocessing

def busy_wait(n):
    while n > 0:
        n -= 1

if __name__ == '__main__':
    n = 10000000
    items = [n for _ in range(8)]
    with multiprocessing.Pool() as pool:
        results = []
        start = time.time()
        print('Start processing...')
        for _ in range(5):
            results.append(pool.map_async(busy_wait, items))
        print('Still processing %.3f' % (time.time() - start))
        for result in results:
            result.wait()
            print('Result done %.3f' % (time.time() - start))
        print('Done processing: %.3f' % (time.time() - start))
```

处理本身非常简单。关键是池保持可用，您不需要等待它。只需在需要时添加作业，并在异步结果可用时使用它们：

```py
# python3 test_pool.py
Start processing...
Still processing 0.000
Result done 1.513
Result done 2.984
Result done 4.463
Result done 5.978
Result done 7.388
Done processing: 7.388

```

# 在进程之间共享数据

这确实是多进程、多线程和分布式编程中最困难的部分——要传递哪些数据，要跳过哪些数据。然而，理论上非常简单：尽可能不传输任何数据，不共享任何东西，保持一切本地。本质上是函数式编程范式，这就是为什么函数式编程与多进程非常搭配。不幸的是，在实践中，这并不总是可能的。`multiprocessing`库有几种共享数据的选项：`Pipe`、`Namespace`、`Queue`和其他一些选项。所有这些选项可能会诱使您一直在进程之间共享数据。这确实是可能的，但在许多情况下，性能影响要比分布式计算提供的额外性能更大。所有数据共享选项都需要在所有处理内核之间进行同步，这需要很长时间。特别是在分布式选项中，这些同步可能需要几毫秒，或者如果在全局范围内执行，可能会导致数百毫秒的延迟。

多进程命名空间的行为与常规对象的工作方式相同，只是有一个小差异，即所有操作都对多进程是安全的。有了这么多功能，命名空间仍然非常容易使用：

```py
import multiprocessing
manager = multiprocessing.Manager()
namespace = manager.Namespace()
namespace.spam = 123
namespace.eggs = 456
```

管道也没有那么有趣。它只是一个双向通信端点，允许读和写。在这方面，它只是为您提供了一个读取器和一个写入器，因此您可以组合多个进程/端点。在同步数据时，您必须始终记住的唯一一件事是，锁定需要时间。为了设置适当的锁，所有参与方都需要同意数据已被锁定，这是一个需要时间的过程。这个简单的事实比大多数人预期的要慢得多。

在常规硬盘设置上，由于锁定和磁盘延迟，数据库服务器无法处理同一行上超过大约 10 个事务每秒。使用延迟文件同步、固态硬盘和带电池备份的 RAID 缓存，该性能可以增加到，也许，每秒处理同一行上的 100 个事务。这些都是简单的硬件限制，因为您有多个进程尝试写入单个目标，您需要在进程之间同步操作，这需要很长时间。

### 注意

“数据库服务器”统计数据是所有提供安全和一致数据存储的数据库服务器的常见统计数据。

即使使用最快的硬件，同步也可能锁定所有进程并导致巨大的减速，因此如果可能的话，尽量避免在多个进程之间共享数据。简而言之，如果所有进程都从/向同一对象读取和写入，通常使用单个进程会更快。

# 远程进程

到目前为止，我们只在多个本地处理器上执行了我们的脚本，但实际上我们可以进一步扩展。使用`multiprocessing`库，实际上非常容易在远程服务器上执行作业，但文档目前仍然有点晦涩。实际上有几种以分布式方式执行进程的方法，但最明显的方法并不是最容易的方法。`multiprocessing.connection`模块具有`Client`和`Listener`类，可以以简单的方式促进客户端和服务器之间的安全通信。然而，通信并不同于进程管理和队列管理，这些功能需要额外的努力。在这方面，多进程库仍然有点简陋，但鉴于一些不同的进程，这是完全可能的。

## 使用多进程进行分布式处理

首先，我们将从一个包含一些常量的模块开始，这些常量应该在所有客户端和服务器之间共享，因此所有人都可以使用服务器的秘密密码和主机名。除此之外，我们将添加我们的质数计算函数，稍后我们将使用它们。以下模块中的导入将期望将此文件存储为`constants.py`，但是只要您修改导入和引用，可以随意将其命名为任何您喜欢的名称：

```py
host = 'localhost'
port = 12345
password = b'some secret password'

def primes(n):
    for i, prime in enumerate(prime_generator()):
        if i == n:
            return prime

def prime_generator():
    n = 2
    primes = set()
    while True:
        for p in primes:
            if n % p == 0:
                break
        else:
            primes.add(n)
            yield n
        n += 1
```

现在是时候创建实际的服务器，将函数和作业队列链接起来了。

```py
import constants
import multiprocessing
from multiprocessing import managers

queue = multiprocessing.Queue()
manager = managers.BaseManager(address=('', constants.port),
                               authkey=constants.password)

manager.register('queue', callable=lambda: queue)
manager.register('primes', callable=constants.primes)

server = manager.get_server()
server.serve_forever()
```

创建服务器后，我们需要一个发送作业的脚本，实际上将是一个常规客户端。这真的很简单，一个常规客户端也可以作为处理器，但为了保持事情合理，我们将它们用作单独的脚本。以下脚本将将 0 添加到 999 以进行处理：

```py
from multiprocessing import managers
import functions

manager = managers.BaseManager(
    address=(functions.host, functions.port),
    authkey=functions.password)
manager.register('queue')
manager.connect()

queue = manager.queue()
for i in range(1000):
    queue.put(i)
```

最后，我们需要创建一个客户端来实际处理队列：

```py
from multiprocessing import managers
import functions

manager = managers.BaseManager(
    address=(functions.host, functions.port),
    authkey=functions.password)
manager.register('queue')
manager.register('primes')
manager.connect()

queue = manager.queue()
while not queue.empty():
    print(manager.primes(queue.get()))
```

从前面的代码中，您可以看到我们如何传递函数；管理器允许注册可以从客户端调用的函数和类。通过这样，我们可以传递一个队列，从多进程类中，这对多线程和多进程都是安全的。现在我们需要启动进程本身。首先是保持运行的服务器：

```py
# python3 multiprocessing_server.py

```

之后，运行生产者生成质数生成请求：

```py
# python3 multiprocessing_producer.py

```

现在我们可以在多台机器上运行多个客户端，以获得前 1000 个质数。由于这些客户端现在打印出前 1000 个质数，输出有点太长，无法在这里显示，但您可以简单地在多台机器上并行运行此操作以生成您的输出：

```py
# python3 multiprocessing_client.py

```

您可以使用队列或管道将输出发送到不同的进程，而不是打印。但是，正如您所看到的，要并行处理事物仍然需要一些工作，并且需要一些代码同步才能正常工作。还有一些可用的替代方案，例如**ØMQ**、**Celery**和**IPyparallel**。哪种是最好和最合适的取决于您的用例。如果您只是想在多个 CPU 上处理任务，那么多进程和 IPyparallel 可能是您最好的选择。如果您正在寻找后台处理和/或轻松地将任务卸载到多台机器上，那么ØMQ 和 Celery 是更好的选择。

## 使用 IPyparallel 进行分布式处理

IPyparallel 模块（以前是 IPython Parallel）是一个模块，使得在多台计算机上同时处理代码变得非常容易。该库支持的功能比您可能需要的要多，但了解基本用法非常重要，以防您需要进行可以从多台计算机中受益的大量计算。首先，让我们从安装最新的 IPyparallel 包和所有 IPython 组件开始：

```py
pip install -U ipython[all] ipyparallel

```

### 注意

特别是在 Windows 上，使用 Anaconda 安装 IPython 可能更容易，因为它包含了许多科学、数学、工程和数据分析软件包的二进制文件。为了获得一致的安装，Anaconda 安装程序也适用于 OS X 和 Linux 系统。

其次，我们需要一个集群配置。从技术上讲，这是可选的，但由于我们将创建一个分布式 IPython 集群，使用特定配置来配置一切会更方便：

```py
# ipython profile create --parallel --profile=mastering_python
[ProfileCreate] Generating default config file: '~/.ipython/profile_mastering_python/ipython_config.py'
[ProfileCreate] Generating default config file: '~/.ipython/profile_mastering_python/ipython_kernel_config.py'
[ProfileCreate] Generating default config file: '~/.ipython/profile_mastering_python/ipcontroller_config.py'
[ProfileCreate] Generating default config file: '~/.ipython/profile_mastering_python/ipengine_config.py'
[ProfileCreate] Generating default config file: '~/.ipython/profile_mastering_python/ipcluster_config.py'

```

这些配置文件包含大量的选项，因此我建议搜索特定部分而不是逐个浏览它们。快速列出给我总共约 2500 行配置，分布在这五个文件中。文件名已经提供了关于配置文件目的的提示，但由于它们仍然有点令人困惑，我们将更详细地解释它们。

### ipython_config.py

这是通用的 IPython 配置文件；您可以在这里自定义关于您的 IPython shell 的几乎所有内容。它定义了您的 shell 应该如何显示，哪些模块应该默认加载，是否加载 GUI 等等。对于本章的目的并不是很重要，但如果您要经常使用 IPython，那么它绝对值得一看。您可以在这里配置的一件事是自动加载扩展，比如在上一章中讨论的`line_profiler`和`memory_profiler`。

```py
c.InteractiveShellApp.extensions = [
    'line_profiler',
    'memory_profiler',
]
```

### ipython_kernel_config.py

这个文件配置了您的 IPython 内核，并允许您覆盖/扩展`ipython_config.py`。要理解它的目的，重要的是要知道什么是 IPython 内核。在这个上下文中，内核是运行和审查代码的程序。默认情况下，这是`IPyKernel`，它是一个常规的 Python 解释器，但也有其他选项，如`IRuby`或`IJavascript`分别运行 Ruby 或 JavaScript。

其中一个更有用的选项是配置内核的监听端口和 IP 地址的可能性。默认情况下，端口都设置为使用随机数，但重要的是要注意，如果其他人在您运行内核时访问同一台机器，他们将能够连接到您的 IPython 内核，这在共享机器上可能是危险的。

### ipcontroller_config.py

`ipcontroller`是您的 IPython 集群的主进程。它控制引擎和任务的分发，并负责诸如日志记录之类的任务。

在性能方面最重要的参数是`TaskScheduler`设置。默认情况下，`c.TaskScheduler.scheme_name`设置为使用 Python LRU 调度程序，但根据您的工作负载，其他调度程序如`leastload`和`weighted`可能更好。如果您必须在如此大的集群上处理如此多的任务，以至于调度程序成为瓶颈，那么还有`plainrandom`调度程序，如果您的所有计算机具有类似的规格并且任务具有类似的持续时间，它会出奇地有效。

为了我们的测试目的，我们将控制器的 IP 设置为*，这意味着将接受**所有**IP 地址，并且将接受每个网络连接。如果您处于不安全的环境/网络，并且/或者没有任何允许您有选择地启用某些 IP 地址的防火墙，那么**不建议**使用这种方法！在这种情况下，我建议通过更安全的选项启动，例如`SSHEngineSetLauncher`或`WindowsHPCEngineSetLauncher`。

但是，假设您的网络确实是安全的，将工厂 IP 设置为所有本地地址：

```py
c.HubFactory.client_ip = '*'
c.RegistrationFactory.ip = '*'
```

现在启动控制器：

```py
# ipcontroller --profile=mastering_python
[IPControllerApp] Hub listening on tcp://*:58412 for registration.
[IPControllerApp] Hub listening on tcp://127.0.0.1:58412 for registration.
[IPControllerApp] Hub using DB backend: 'NoDB'
[IPControllerApp] hub::created hub
[IPControllerApp] writing connection info to ~/.ipython/profile_mastering_python/security/ipcontroller-client.json
[IPControllerApp] writing connection info to ~/.ipython/profile_mastering_python/security/ipcontroller-engine.json
[IPControllerApp] task::using Python leastload Task scheduler
[IPControllerApp] Heartmonitor started
[IPControllerApp] Creating pid file: .ipython/profile_mastering_python/pid/ipcontroller.pid
[scheduler] Scheduler started [leastload]
[IPControllerApp] client::client b'\x00\x80\x00A\xa7' requested 'connection_request'
[IPControllerApp] client::client [b'\x00\x80\x00A\xa7'] connected

```

注意已写入配置文件目录的安全目录中的文件。它包含了`ipengine`用于找到`ipcontroller`的身份验证信息。它包含端口、加密密钥和 IP 地址。

### ipengine_config.py

`ipengine`是实际的工作进程。这些进程运行实际的计算，因此为了加快处理速度，您需要在尽可能多的计算机上运行这些进程。您可能不需要更改此文件，但如果您想配置集中式日志记录或需要更改工作目录，则可能会有用。通常情况下，您不希望手动启动`ipengine`进程，因为您很可能希望在每台计算机上启动多个进程。这就是我们下一个命令`ipcluster`的用处。

### ipcluster_config.py

`ipcluster`命令实际上只是一个简单的快捷方式，可以同时启动`ipcontroller`和`ipengine`的组合。对于简单的本地处理集群，我建议使用这个，但是在启动分布式集群时，单独使用`ipcontroller`和`ipengine`可以很有用。在大多数情况下，该命令提供了足够的选项，因此您可能不需要单独的命令。

最重要的配置选项是`c.IPClusterEngines.engine_launcher_class`，因为它控制了引擎和控制器之间的通信方法。除此之外，它也是安全通信的最重要组件。默认情况下，它设置为`ipyparallel.apps.launcher.LocalControllerLauncher`，适用于本地进程，但如果您想要使用 SSH 与客户端通信，也可以选择`ipyparallel.apps.launcher.SSHEngineSetLauncher`。或者对于 Windows HPC，可以选择`ipyparallel.apps.launcher.WindowsHPCEngineSetLauncher`。

在所有机器上创建集群之前，我们需要传输配置文件。您可以选择传输所有文件，也可以选择仅传输 IPython 配置文件的`security`目录中的文件。

现在是时候启动集群了，因为我们已经单独启动了`ipcontroller`，所以我们只需要启动引擎。在本地机器上，我们只需要启动它，但其他机器还没有配置。一种选择是复制整个 IPython 配置文件目录，但实际上只需要复制`security/ipcontroller-engine.json`文件。在使用配置文件创建命令创建配置文件之后。因此，除非您打算复制整个 IPython 配置文件目录，否则需要再次执行配置文件创建命令：

```py
# ipython profile create --parallel --profile=mastering_python

```

之后，只需复制`ipcontroller-engine.json`文件，就完成了。现在我们可以启动实际的引擎了：

```py
# ipcluster engines --profile=mastering_python -n 4
[IPClusterEngines] IPython cluster: started
[IPClusterEngines] Starting engines with [daemon=False]
[IPClusterEngines] Starting 4 Engines with LocalEngineSetLauncher

```

请注意，这里的`4`是为四核处理器选择的，但任何数字都可以。默认情况下将使用逻辑处理器核心的数量，但根据工作负载，最好匹配物理处理器核心的数量。

现在我们可以从 IPython shell 运行一些并行代码。为了演示性能差异，我们将使用从 0 加到 10,000,000 的所有数字的简单总和。虽然不是非常繁重的任务，但连续执行 10 次时，常规的 Python 解释器需要一段时间：

```py
In [1]: %timeit for _ in range(10): sum(range(10000000))
1 loops, best of 3: 2.27 s per loop
```

然而，这一次，为了说明差异，我们将运行 100 次以演示分布式集群有多快。请注意，这只是一个三台机器集群，但速度仍然相当快：

```py
In [1]: import ipyparallel

In [2]: client = ipyparallel.Client(profile='mastering_python')

In [3]: view = client.load_balanced_view()

In [4]: %timeit view.map(lambda _: sum(range(10000000)), range(100)).wait()
1 loop, best of 3: 909 ms per loop
```

然而，更有趣的是在 IPyParallel 中定义并行函数。只需一个简单的装饰器，一个函数就被标记为并行：

```py
In [1]: import ipyparallel

In [2]: client = ipyparallel.Client(profile='mastering_python')

In [3]: view = client.load_balanced_view()

In [4]: @view.parallel()
   ...: def loop():
   ...:     return sum(range(10000000))
   ...:

In [5]: loop.map(range(10))
Out[5]: <AsyncMapResult: loop>
```

IPyParallel 库提供了许多其他有用的功能，但这超出了本书的范围。尽管 IPyParallel 是 Jupyter/IPython 的独立实体，但它与之整合良好，这使得它们很容易结合起来。

使用 IPyParallel 最方便的方法之一是通过 Jupyter/IPython 笔记本。为了演示，我们首先必须确保在 Jupyter Notebook 中启用并行处理，因为 IPython 笔记本默认情况下是单线程执行的：

```py
ipcluster nbextension enable

```

之后，我们可以启动`notebook`，看看它是怎么回事：

```py
# jupyter notebook
Unrecognized JSON config file version, assuming version 1
Loading IPython parallel extension
Serving notebooks from local directory: ./
0 active kernels
The Jupyter Notebook is running at: http://localhost:8888/
Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

```

使用 Jupyter Notebook，您可以在 Web 浏览器中创建脚本，稍后可以轻松与他人共享。这对于共享脚本和调试代码非常有用，特别是因为 Web 页面（与命令行环境相反）可以轻松显示图像。这对于绘制数据有很大帮助。这是我们笔记本的屏幕截图：

![ipcluster_config.py](img/4711_13_01.jpg)

# 总结

本章向我们展示了多进程的工作原理，我们如何可以汇集大量的工作，并且我们应该如何在多个进程之间共享数据。但更有趣的是，它还展示了我们如何可以在多台机器之间分发处理，这在加速繁重的计算方面非常有帮助。

您可以从本章中学到的最重要的一课是，您应该尽量避免在多个进程或服务器之间共享数据和同步，因为这样做会很慢，从而大大减慢应用程序的速度。在可能的情况下，保持计算和数据本地。

在下一章中，我们将学习如何在 C/C++中创建扩展，以提高性能并允许对内存和其他硬件资源进行低级访问。虽然 Python 通常会保护您免受愚蠢的错误，但 C 和 C++肯定不会。

|   | “C 使得自己踩到脚趾头很容易；C++让这变得更难，但一旦你踩到了，它会把整条腿都炸掉。” |   |
| --- | --- | --- |
|   | --*Bjarne Stroustrup（C++的创造者）* |
