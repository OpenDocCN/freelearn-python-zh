

# 第十四章：多进程 – 当单个 CPU 核心不够用时

在上一章中，我们讨论了`asyncio`，它可以使用`threading`和`multiprocessing`模块，但主要使用单线程/单进程并行化。在本章中，我们将看到如何直接使用多个线程或进程来加速我们的代码，以及需要注意的注意事项。实际上，本章可以被视为性能技巧列表的扩展。

`threading`模块使得在单个进程中并行运行代码成为可能。这使得`threading`对于与 I/O 相关的任务（如读写文件或网络通信）非常有用，但对于缓慢且计算量大的任务则不是一个有用的选项，这正是`multiprocessing`模块大放异彩的地方。

使用`multiprocessing`模块，你可以在多个进程中运行代码，这意味着你可以在多个 CPU 核心、多个处理器或甚至多台计算机上运行代码。这是绕过在*第十二章*，*性能 – 跟踪和减少你的内存和 CPU 使用*中讨论的**全局解释器锁**（**GIL**）的一种简单方法。

`multiprocessing`模块提供了一个相对容易使用的接口，具有许多便利功能，但`threading`模块相对基础，需要你手动创建和管理线程。为此，我们还有`concurrent.futures`模块，它提供了一种简单的方法来执行一系列任务，无论是通过线程还是进程。此接口也与我们在上一章中看到的`asyncio`功能部分可比。

总结来说，本章涵盖了：

+   全局解释器锁（GIL）

+   多线程与多进程的比较

+   锁定、死锁和线程安全

+   进程间的数据共享和同步

+   在多线程、多进程和单线程之间进行选择

+   超线程与物理核心的比较

+   使用`multiprocessing`和`ipyparallel`进行远程多进程

# 全局解释器锁（GIL）

GIL（全局解释器锁）在本书中已经被提及多次，但我们并没有对其进行详细讲解，它确实需要更多的解释才能继续本章的内容。

简而言之，名称已经解释了它的功能。它是一个 Python 解释器的全局锁，因此它一次只能执行一个语句。在并行计算中，**锁**或**互斥锁**（**互斥**）是一种同步原语，可以阻止并行执行。有了锁，你可以确保在你工作时没有人可以触摸你的变量。

Python 提供了几种类型的同步原语，如`threading.Lock`和`threading.Semaphore`。这些内容在本章的*线程和进程间共享数据*部分有更详细的介绍。

这意味着即使使用`threading`模块，你同时也只能执行一个 Python 语句。因此，当涉及到纯 Python 代码时，你的多线程解决方案**总是**会比单线程解决方案慢，因为`threading`引入了一些同步开销，而在这种情况下并没有提供任何好处。

让我们继续深入了解 GIL 的更多信息。

## 多线程的使用

由于 GIL 只允许同时执行一个 Python 语句，那么线程有什么用呢？其有效性很大程度上取决于你的目标。类似于第十三章中的`asyncio`示例，如果你正在等待外部资源，`threading`可以给你带来很多好处。

例如，如果你正在尝试获取一个网页，打开一个文件（记住`aiofiles`模块实际上使用线程），或者如果你想定期执行某些操作，`threading`可以非常有效。

当编写一个新应用程序时，如果将来有即使是微小的可能性成为 I/O 受限，我通常会建议你让它准备好使用`asyncio`。在以后的时间重新编写以适应`asyncio`可能是一项巨大的工作量。

`asyncio`相对于`threading`有几个优点：

+   `asyncio`通常比线程更快，因为你没有线程同步开销。

+   由于`asyncio`通常是单线程的，你不必担心线程安全问题（关于线程安全的内容将在本章后面详细介绍）。

## 我们为什么需要 GIL？

GIL 目前是 CPython 解释器的一个基本组成部分，因为它确保内存管理始终是一致的。

为了解释这是如何工作的，我们需要了解一些关于 CPython 解释器如何管理其内存的信息。

在 CPython 中，内存管理系统和垃圾回收系统依赖于引用计数。这意味着 CPython 会计算你有多少个名称链接到一个值。如果你有一行 Python 代码像这样：`a = SomeObject()`，这意味着这个`SomeObject`实例有 1 个引用，即`a`。如果我们执行`b = a`，引用计数将增加到 2。当引用计数达到 0 时，变量将在垃圾回收器运行时被删除。

你可以使用`sys.getrefcount(variable)`来检查引用的数量。你应该注意，对`sys.getrefcount()`的调用会增加你的引用计数 1，所以如果它返回 2，实际的数量是 1。

由于 GIL 确保同时只能执行一个 Python 语句，你永远不会遇到多个代码块同时操作内存的问题，或者内存被释放到实际上并未空闲的系统。

如果引用计数器没有正确管理，这很容易导致内存泄漏或 Python 解释器崩溃。记得我们在*第十一章*，*调试 – 解决错误*中看到的段错误？这就是没有 GIL 时可能发生的事情，它将立即杀死你的 Python 解释器。

## 我们为什么仍然有 GIL？

当 Python 最初被创建时，许多操作系统甚至没有线程的概念，所有常见的处理器都只有一个核心。简而言之，GIL 存在有两个主要原因：

+   初始时，创建一个处理线程的复杂解决方案并没有什么意义

+   GIL 是一个针对非常复杂问题的简单解决方案

幸运的是，这似乎并不是讨论的终点。最近（2021 年 5 月），吉多·范罗苏姆（Guido van Rossum）从退休状态中复出，他计划通过为线程创建子解释器来解决 GIL 的限制。当然，如何在实践中实现这一点还有待观察，但雄心勃勃的计划是将 CPython 3.15 的速度提升到 CPython 3.10 的 5 倍，这将是一个惊人的性能提升。

现在我们已经知道 GIL 限制了 CPython 线程，让我们看看我们如何创建和使用多个线程和进程。

# 多线程和多进程

`multiprocessing`模块在 Python 2.6 中被引入，它在处理 Python 中的多个进程方面是一个游戏规则的改变。具体来说，它使得绕过 GIL 的限制变得相当容易，因为每个进程都有自己的 GIL。

`multiprocessing`模块的使用在很大程度上与`threading`模块相似，但它有几个非常有用的额外功能，这些功能在多进程中使用时更有意义。或者，你也可以使用`concurrent.futures.ProcessPoolExecutor`，它的接口几乎与`concurrent.futures.ThreadPoolExecutor`相同。

这些相似之处意味着在许多情况下，你可以简单地替换模块，你的代码仍然会按预期运行。然而，不要被误导；尽管线程仍然可以使用相同的内存对象，并且只需要担心线程安全和死锁，但多个进程也存在这些问题，并且在共享内存、对象和结果时还会引入其他问题。

在任何情况下，处理并行代码都伴随着注意事项。这也是为什么使用多个线程或进程的代码有难以工作的声誉。许多这些问题并不像看起来那么可怕；如果你遵循一些规则，那就是了。

在我们继续示例代码之前，你应该意识到，在使用`multiprocessing`时，将你的代码放在`if __name__ == '__main__'`块中至关重要。当`multiprocessing`模块启动额外的 Python 进程时，它将执行相同的 Python 脚本，所以如果不使用这个块，你将陷入无限循环的启动进程。

在本节中，我们将介绍：

+   使用 `threading`、`multiprocessing` 和 `concurrent.futures` 的基本示例

+   清洁退出线程和进程

+   批处理

+   在进程间共享内存

+   线程安全性

+   死锁

+   线程局部变量

其中一些，如竞态条件和锁定，不仅限于线程，对`multiprocessing`也可能很有趣。

## 基本示例

要创建线程和进程，我们有几种选择：

+   `concurrent.futures`：一个易于使用的接口，用于在线程或进程中运行函数，类似于`asyncio`

+   `threading`：一个用于直接创建线程的接口

+   `multiprocessing`：一个具有许多实用和便利函数的接口，用于创建和管理多个 Python 进程

让我们看看每个示例。

### concurrent.futures

让我们从`concurrent.futures`模块的基本示例开始。在这个例子中，我们运行了两个并行运行和打印的计时器任务：

```py
import time
import concurrent.futures

def timer(name, steps, interval=0.1):
    '''timer function that sleeps 'steps * interval' '''
    for step in range(steps):
        print(name, step)
        time.sleep(interval)

if __name__ == '__main__':
    # Replace with concurrent.futures.ProcessPoolExecutor for
    # multiple processes instead of threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the function to the executor with some arguments
        executor.submit(timer, steps=3, name='a')
        # Sleep a tiny bit to keep the output order consistent
        time.sleep(0.1)
        executor.submit(timer, steps=3, name='b') 
```

在我们执行代码之前，让我们看看我们在这里做了什么。首先，我们创建了一个`timer`函数，该函数运行`time.sleep(interval)`并执行`steps`次。在休眠之前，它打印出`name`和当前的`step`，这样我们就可以轻松地看到发生了什么。

然后，我们使用`concurrent.futures.ThreadPoolExecutor`创建一个`executor`来执行函数。

最后，我们将要执行的函数及其相应的参数提交以启动两个线程。在启动它们之间，我们休眠了很短的时间，所以在这个例子中我们的输出是一致的。如果我们不执行`time.sleep(0.1)`，输出顺序将是随机的，因为有时`a`会更快，而有时`b`会更快。

包含短暂休眠的主要原因是测试。本书中的所有代码都可在 GitHub 上找到（[`github.com/mastering-python/code_2`](https://github.com/mastering-python/code_2)），并且会自动进行测试。

现在当我们执行这个脚本时，我们得到以下结果：

```py
$ python3 T_00_concurrent_futures.py
a 0
b 0
a 1
b 1
a 2
b 2 
```

如预期的那样，它们紧挨着运行，但由于我们添加了微小的`time.sleep(0.1)`，结果是一致地交织在一起。在这种情况下，我们使用默认参数启动了`ThreadPoolExecutor`，这导致没有特定名称的线程和自动计算的线程数。

线程数取决于 Python 版本。在 Python 3.8 之前，工作进程的数量等于机器中超线程 CPU 核心的数量乘以 5。因此，如果你的机器有 2 个启用超线程的核心，那么结果将是 4 个核心 * 5 = 20 个线程。对于 64 核的机器，这将导致 320 个线程，这可能会产生比好处更多的同步开销。

对于 Python 3.8 及以上版本，这已被更改为`min(32, cores + 4)`，这应该足以始终至少有 5 个线程用于 I/O 操作，但不会太多以至于在多核机器上使用大量资源。对于相同的`64`核机器，这仍然限制在`32`个线程。

在 `ProcessPoolExecutor` 的情况下，将使用包括超线程在内的处理器核心数。这意味着如果您的处理器有 4 个核心并且启用了超线程，您将默认获得 8 个进程。

自然地，传统的 `threading` 模块仍然是一个不错的选择，它提供了更多的控制，同时仍然拥有易于使用的接口。

在 Python 3 之前，`thread` 模块也作为线程的低级 API 可用。此模块仍然可用，但已重命名为 `_thread`。在内部，`concurrent.futures.ThreadPoolExecutor` 和 `threading` 都仍然使用它，但您通常不需要直接访问它。

### threading

现在我们将看看如何使用 `threading` 模块重新创建 `concurrent.futures` 的示例：

```py
import time
import threading

def timer(name, steps, interval=0.1):
    '''timer function that sleeps 'steps * interval' '''
    for step in range(steps):
        print(name, step)
        time.sleep(interval)

# Create the threads declaratively
a = threading.Thread(target=timer, kwargs=dict(name='a', steps=3))
b = threading.Thread(target=timer, kwargs=dict(name='b', steps=3))

# Start the threads
a.start()
# Sleep a tiny bit to keep the output order consistent
time.sleep(0.1)
b.start() 
```

`timer` 函数与前面的示例相同，所以这里没有差异。但是，执行方式略有不同。

在这种情况下，我们通过直接实例化 `threading.Thread()` 来创建线程，但继承 `threading.Thread` 也是一个选项，正如我们将在下一个示例中看到的。可以传递 `args` 和/或 `kwargs` 参数来提供 `target` 函数的参数，但如果您不需要它们或者已经使用 `functools.partial` 填充了它们，则这些参数是可选的。

在前面的示例中，我们创建了一个 `ThreadPoolExecutor()`，它会创建许多线程并在这些线程上运行函数。在这个示例中，我们明确创建线程来运行单个函数并在函数完成后退出。这对于长时间运行的背景线程非常有用，因为这个方法需要为每个函数设置和拆除线程。通常，启动线程的开销非常小，但它取决于您的 Python 解释器（CPython、PyPy 等）和操作系统。

现在对于相同的示例，但继承 `threading.Thread` 而不是对 `threading.Thread()` 的声明性调用：

```py
import time
import threading

class Timer(threading.Thread):
    def __init__(self, name, steps, interval=0.1):
        self.steps = steps
        self.interval = interval
        # Small gotcha: threading.Thread has a built-in name
        # parameter so be careful not to manually override it
        super().__init__(name=name)

    def run(self):
        '''timer function that sleeps 'steps * interval' '''
        for step in range(self.steps):
            print(self.name, step)
            time.sleep(self.interval)
a = Timer(name='a', steps=3)
b = Timer(name='b', steps=3)

# Start the threads
a.start()
# Sleep a tiny bit to keep the output order consistent
time.sleep(0.1)
b.start() 
```

代码大致与直接调用 `threading.Thread()` 的过程式版本相同，但有两大关键差异您需要注意：

+   `name` 是 `threading.Thread` 的一个保留属性。在 Linux/Unix 机器上，您的进程管理器（例如，`top`）可以显示此名称而不是 `/usr/bin/python3`。

+   默认的目标函数是 `run()`。请小心覆盖 `run()` 方法而不是 `start()` 方法，否则当您调用 `start()` 时，您的代码将*不会*在单独的线程中执行，而将像常规函数调用一样执行。

过程式和基于类的版本在内部使用完全相同的 API，并且同样强大，因此选择它们主要取决于个人偏好。

### multiprocessing

最后，我们也可以使用 `multiprocessing` 来重新创建早期的定时器脚本。首先是通过 `multiprocessing.Process()` 的过程调用：

```py
import time
import multiprocessing

def timer(name, steps, interval=0.1):
    '''timer function that sleeps 'steps * interval' '''
    for step in range(steps):
        print(name, step)
        time.sleep(interval)

if __name__ == '__main__':
    # Create the processes declaratively
    a = multiprocessing.Process(target=timer, kwargs=dict(name='a', steps=3))
    b = multiprocessing.Process(target=timer, kwargs=dict(name='b', steps=3))

    # Start the processes
    a.start()
    # Sleep a tiny bit to keep the output order consistent
    time.sleep(0.1)
    b.start() 
```

代码看起来几乎相同，只有一些小的变化。我们使用了`multiprocessing.Process`而不是`threading.Thread`，并且我们必须从`if __name__ == '__main__'`块中运行代码。除此之外，在这个简单的例子中，代码和执行都是相同的。

最后，为了完整性，让我们也看看基于类的版本：

```py
import time
import multiprocessing

class Timer(multiprocessing.Process):
    def __init__(self, name, steps, interval=0.1):
        self.steps = steps
        self.interval = interval
        # Similar to threading.Thread, multiprocessing.Process
        # also supports the name parameter but you are not
        # required to use it here.
        super().__init__(name=name)

    def run(self):
        '''timer function that sleeps 'steps * interval' '''
        for step in range(self.steps):
            print(self.name, step)
            time.sleep(self.interval)

if __name__ == '__main__':
    a = Timer(name='a', steps=3)
    b = Timer(name='b', steps=3)

    # Start the process
    a.start()
    # Sleep a tiny bit to keep the output order consistent
    time.sleep(0.1)
    b.start() 
```

再次强调，我们必须使用`if __name__ == '__main__'`块。但除此之外，代码与`threading`版本几乎相同。就像`threading`一样，选择过程式和基于类的风格仅取决于你个人的偏好。

现在我们已经知道了如何启动线程和进程，让我们看看我们如何可以干净地关闭它们。

## 清理退出长时间运行的线程和进程

`threading`模块主要用于处理外部资源的长时间运行的线程。一些示例场景：

+   当创建服务器并希望持续监听新的连接

+   当连接到 HTTP WebSockets 并且需要保持连接打开时

+   当你需要定期保存你的更改

自然地，这些场景也可以使用`multiprocessing`，但`threading`通常更方便，我们将在后面看到。

在某个时候，你可能需要从**外部**关闭线程；例如，在主脚本的退出过程中。等待自行退出的线程是微不足道的；你需要做的只是`future.result()`或`some_thread.join(timeout=...)`，然后你就完成了。更困难的部分是告诉线程自行关闭并在它仍在做某事时运行清理。

解决这个问题的唯一真正的方法，如果你幸运的话，是一个简单的`while`循环，它会一直运行，直到你给出停止信号，如下所示：

```py
import time
import threading

class Forever(threading.Thread):
    def __init__(self):
        self.stop = threading.Event()
        super().__init__()

    def run(self):
        while not self.stop.is_set():
            # Do whatever you need to do here
            time.sleep(0.1)

thread = Forever()
thread.start()
# Do whatever you need to do here
thread.stop.set()
thread.join() 
```

此代码使用`threading.Event()`作为标志来告诉线程在需要时退出。虽然你可以使用`bool`代替`threading.Event()`与当前的 CPython 解释器一起使用，但无法保证这将在未来的 Python 版本和/或其他类型的解释器中工作。目前这之所以对 CPython 来说是安全的，是因为由于 GIL，所有 Python 操作实际上都是单线程的。这就是为什么线程对于等待外部资源很有用，但会对你的 Python 代码的性能产生负面影响。

此外，如果你要将此代码翻译成多进程，你可以简单地用`multiprocessing.Event()`替换`threading.Event()`，并且它应该在没有其他更改的情况下继续工作，假设你没有与外部变量交互。在多个 Python 进程中，你不再受到单个 GIL 的保护，因此在修改变量时需要更加小心。关于这个话题的更多内容将在本章后面的*线程和进程间共享数据*部分中介绍。

现在我们有了`stop`事件，我们可以运行`stop.set()`，这样线程就知道何时退出，并在最多 0.1 秒的睡眠后退出。

这是一个理想的场景：有一个循环，循环条件会定期检查，循环间隔是你的最大线程关闭延迟。如果线程正忙于执行某些操作而没有检查`while`条件会发生什么？正如你可能猜到的，在这些场景中设置`stop`事件是无用的，你需要一个更强大的方法来退出线程。

处理这种场景，你有几种选择：

+   通过使用`asyncio`或`multiprocessing`来完全避免这个问题。在性能方面，`asyncio`无疑是你的最佳选择，但如果你的代码适合，`multiprocessing`也可以工作得很好。

+   通过在启动线程之前将`your_thread.daemon = True`设置为守护线程。这样，当主进程退出时，线程会自动终止，因此这不是一个优雅的关闭。你仍然可以使用`atexit`模块添加拆卸。

+   通过告诉操作系统发送终止/杀死信号或在主线程中从线程内部抛出异常来从外部杀死线程。你可能想尝试这种方法，但我强烈建议不要这样做。这不仅不可靠，还可能导致你的整个 Python 解释器崩溃，所以这绝对不是你应该考虑的选项。

我们已经在上一章中看到了如何使用`asyncio`，所以让我们看看我们如何使用`multiprocessing`来终止。然而，在我们开始之前，你应该注意，适用于`threading`的限制在很大程度上也适用于`multiprocessing`。虽然`multiprocessing`确实有一个内置的终止进程的解决方案，与线程不同，但这仍然不是一个干净的方法，它也不会（可靠地）运行你的退出处理程序、`finally`子句等。这意味着你应该*始终*首先尝试一个事件，但当然使用`multiprocessing.Event`而不是`threading.Event`。

为了说明我们如何强制终止或杀死一个线程（同时冒着内存损坏的风险）：

```py
import time
import multiprocessing

class Forever(multiprocessing.Process):
    def run(self):
        while True:
            # Do whatever you need to do here
            time.sleep(0.1)

if __name__ == '__main__':
    process = Forever()
    process.start()

    # Kill our "unkillable" process
    process.terminate()
    # Wait for 10 seconds to properly exit      
    process.join(10)

    # If it still didn't exit, kill it
    if process.exitcode is None:
        process.kill() 
```

在这个例子中，我们首先尝试一个常规的`terminate()`，它在 Unix 机器上发送`SIGTERM`信号，在 Windows 上是`TerminateProcess()`。如果那不起作用，我们再次尝试使用`kill()`，它在 Unix 上发送`SIGKILL`信号，目前在 Windows 上没有等效的信号，所以在 Windows 上`kill()`和`terminate()`方法的行为相同，并且两者都有效地终止了进程而没有进行拆卸。

## 使用`concurrent.futures`进行批处理

如我们在先前的例子中所见，以“点火并忘记”的方式启动线程或进程是足够简单的。然而，通常，你想要启动几个线程或进程，并等待它们全部完成。

这是一个`concurrent.futures`和`multiprocessing`真正闪耀的案例。它们允许您以与我们在第五章中看到的方式非常相似地调用`executor.map()`或`pool.map()`，即*第五章*，*函数式编程 – 可读性与简洁性之间的权衡*。实际上，您只需要创建一个要处理的项目列表，调用`[executor/pool].map()`函数，就完成了。如果您想找点乐子，可以使用`threading`模块构建类似的东西，但除此之外，它的用途很少。

为了测试我们的系统，让我们获取一些关于应该使用系统 DNS 解析系统的主机名的信息。由于它查询外部资源，我们使用线程时应该期待良好的结果，对吧？好吧...让我们试一试，看看结果如何：

```py
import timeit
import socket
import concurrent.futures

def getaddrinfo(*args):
    # Call getaddrinfo but ignore the given parameter
    socket.getaddrinfo('localhost', None)

def benchmark(threads, n=1000):
    if threads > 1:
        # Create the executor
        with concurrent.futures.ThreadPoolExecutor(threads) \
                as executor:
            executor.map(getaddrinfo, range(n))

    else:
        # Make sure to use 'list'. Otherwise the generator will
        # not execute because it is lazy
        list(map(getaddrinfo, range(n)))

if __name__ == '__main__':
    for threads in (1, 10, 50, 100):
        print(f'Testing with {threads} threads and n={10} took: ',
              end='')
        print('{:.1f}'.format(timeit.timeit(
            f'benchmark({threads})',
            setup='from __main__ import benchmark',
            number=10,
        ))) 
```

让我们分析一下这段代码。首先，我们有一个`getaddrinfo()`函数，它尝试通过您的操作系统获取关于主机名的一些信息，这是一个可能从多线程中受益的外部资源。

第二，我们有一个`benchmark()`函数，如果`threads`设置为大于 1 的数字，它将使用多个线程进行`map()`。如果没有，它将使用常规的`map()`。

最后，我们对`1`、`10`、`50`和`100`个线程进行了基准测试，其中`1`是常规的非线程方法。那么线程能帮我们多少呢？这个测试强烈依赖于您的计算机、操作系统、网络等，所以您的结果可能会有所不同，但这是我使用 CPython 3.10 在我的 OS X 机器上发生的情况：

```py
$ python3 T_07_thread_batch_processing.py
Testing with 1 threads and n=10 took: 2.1
Testing with 10 threads and n=10 took: 1.9
Testing with 50 threads and n=10 took: 1.9
Testing with 100 threads and n=10 took: 13.9 
```

您期待这些结果吗？虽然`1`个线程确实比`10`个线程和`50`个线程慢，但在`100`个线程时，我们明显看到了收益递减和拥有`100`个线程的开销。此外，由于`socket.getaddrinfo()`相当快，使用多线程的好处在这里相当有限。

如果我们从慢速网络文件系统读取大量文件，或者如果我们用它来并行获取多个网页，我们会看到更大的差异。这立即显示了线程的缺点：它只有在外部资源足够慢，以至于同步开销是合理的时才会带来好处。对于快速的外部资源，您可能会遇到减速，因为全局解释器锁（GIL）成为了瓶颈。CPython 一次只能执行一个语句，这可能会迅速变成问题。

当谈到性能时，您应该始终运行基准测试以查看最适合您情况的方法，尤其是在线程数量方面。正如您在早期示例中看到的，更多并不总是更好，100 线程版本比单线程版本慢得多。

那么，如果我们尝试使用进程而不是线程来执行相同的操作会怎样呢？为了简洁起见，我们将跳过实际的代码，因为我们实际上只需要将 `concurrent.futures.ThreadPoolExecutor()` 替换为 `concurrent.futures.ProcessPoolExecutor()`，然后我们就完成了。如果你感兴趣，测试的代码可以在 GitHub 上找到。当我们执行这段代码时，我们得到以下结果：

```py
$ python3 T_08_process_batch_processing.py
Testing with 1 processes and n=10 took: 2.1
Testing with 10 processes and n=10 took: 3.2
Testing with 50 processes and n=10 took: 8.3
Testing with 100 processes and n=10 took: 15.0 
```

如你所见，当我们使用多个进程时，我们得到了普遍较慢的结果。虽然多进程在 GIL 或单个 CPU 核心是限制时可以提供很多好处，但开销可能会在其他场景中影响你的性能。

## 使用多进程进行批量处理

在上一节中，我们看到了如何使用 `concurrent.futures` 进行批量处理。你可能想知道为什么我们想要直接使用 `multiprocessing`，而不是 `concurrent.futures` 可以为我们处理它。原因相当简单：`concurrent.futures` 是一个易于使用且非常简单的接口，用于 `threading` 和 `multiprocessing`，但 `multiprocessing` 提供了几个高级选项，这些选项可以非常方便，甚至可以在某些场景中帮助提高你的性能。

在之前的例子中，我们只看到了 `multiprocessing.Process`，它是 `threading.Thread` 的进程类似物。然而，在这种情况下，我们将使用 `multiprocessing.Pool`，它创建了一个与 `concurrent.futures` 执行器非常相似的进程池，但提供了几个额外的功能：

+   `map_async(func, iterable, [..., callback, ...])`

    `map_async()` 方法与 `concurrent.futures` 中的 `map()` 方法相似，但它返回一个 `AsyncResult` 对象的列表，这样你就可以在你需要的时候获取结果。

+   `imap(func, iterable[, chunksize])`

    `imap()` 方法实际上是 `map()` 的生成器版本。它的工作方式大致相同，但它不会预先加载可迭代对象中的项，因此如果你需要，可以安全地处理大型可迭代对象。如果你需要处理许多项，这可以 *大大* 提高速度。

+   `imap_unordered(func, iterable[, chunksize])`

    `imap_unordered()` 方法实际上与 `imap()` 相同，但它会在处理完结果后立即返回结果，这可以进一步提高性能。如果你的结果顺序不重要，可以尝试一下，因为它可以使你的代码更快。

+   `starmap(func, iterable[, chunksize])`

    `starmap()` 方法与 `map()` 方法非常相似，但通过像 `*args` 这样的方式支持多个参数。如果你要运行 `starmap(function, [(1, 2), (3, 4)])`，`starmap()` 方法将调用 `function(1, 2)` 和 `function(3, 4)`。这在与 `zip()` 结合使用时可以非常实用，以组合多个参数列表。

+   `starmap_async(func, iterable, [..., callback, ...])`

    如你所想，`starmap_async()` 实际上是非阻塞的 `starmap()` 方法，但它返回一个 `AsyncResult` 对象的列表，这样你就可以在你方便的时候获取它们。

`multiprocessing.Pool()`的使用在很大程度上类似于`concurrent.future.SomeExecutor()`，除了上面提到的额外方法之外。根据你的场景，它可能比`concurrent.futures`慢，速度相似，或者更快，所以总是确保为你的特定用例进行基准测试。以下这段基准代码应该会给你一个很好的起点：

```py
import timeit
import functools
import multiprocessing
import concurrent.futures

def triangle_number(n):
    total = 0
    for i in range(n + 1):
        total += i

    return total

def bench_mp(n, count, chunksize):
    with multiprocessing.Pool() as pool:
        # Generate a generator like [n, n, n, ..., n, n]
        iterable = (n for _ in range(count))
        list(pool.imap_unordered(triangle_number, iterable,
                                 chunksize=chunksize))

def bench_ft(n, count, chunksize):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Generate a generator like [n, n, n, ..., n, n]
        iterable = (n for _ in range(count))
        list(executor.map(triangle_number, iterable,
                          chunksize=chunksize))

if __name__ == '__main__':
    timer = functools.partial(timeit.timeit, number=5)

    n = 1000
    chunksize = 50
    for count in (100, 1000, 10000):
        # Using <6 formatting for consistent alignment
        args = ', '.join((
            f'n={n:<6}',
            f'count={count:<6}',
            f'chunksize={chunksize:<6}',
        ))
        time_mp = timer(
            f'bench_mp({args})',
            setup='from __main__ import bench_mp',
        )
        time_ft = timer(
            f'bench_ft({args})',
            setup='from __main__ import bench_ft',
        )

        print(f'{args} mp: {time_mp:.2f}, ft: {time_ft:.2f}') 
```

在我的机器上，这给出了以下结果：

```py
$ python3 T_09_multiprocessing_pool.py
n=1000  , count=100   , chunksize=50     mp: 0.71, ft: 0.42
n=1000  , count=1000  , chunksize=50     mp: 0.76, ft: 0.96
n=1000  , count=10000 , chunksize=50     mp: 1.12, ft: 1.40 
```

在我进行基准测试之前，我没有预料到`concurrent.futures`在某些情况下会快得多，而在其他情况下会慢得多。分析这些结果，你可以看到，使用`concurrent.futures`处理 1,000 个项目比在这个特定情况下使用多进程处理 10,000 个项目花费的时间要多。同样，对于 100 个项目，`multiprocessing`模块几乎慢了两倍。自然地，每次运行都会产生不同的结果，并且没有单一的选项会在每种情况下都表现良好，但这是需要记住的。

现在我们知道了如何在多个线程或进程中运行我们的代码，让我们看看我们如何安全地在线程/进程之间共享数据。

# 在线程和进程之间共享数据

数据共享实际上是关于多进程、多线程以及一般分布式编程中最困难的部分：要传递哪些数据，要共享哪些数据，以及要跳过哪些数据。然而，理论实际上非常简单：尽可能不要传输任何数据，不要共享任何数据，并保持一切局部。这本质上就是**函数式编程**范式，这也是为什么函数式编程与多进程结合得如此之好的原因。在实践中，遗憾的是，这并不总是可能的。`multiprocessing`库提供了几个共享数据的选择，但内部它们归结为两种不同的选项：

+   **共享内存**：这是迄今为止最快的解决方案，因为它几乎没有开销，但它只能用于不可变类型，并且仅限于通过`multiprocessing.sharedctypes`创建的少数几种类型和自定义对象。如果你只需要存储原始类型，如`int`、`float`、`bool`、`str`、`bytes`以及/或固定大小的列表或字典（其中子项是原始类型），这是一个极好的解决方案。

+   `multiprocessing.Manager`：`Manager`类提供了一系列存储和同步数据的选择，例如锁、信号量、队列、列表、字典等。如果可以序列化，就可以与`Manager`一起使用。

对于线程，解决方案甚至更简单：所有内存都是共享的，因此默认情况下，所有对象都可以从每个线程中访问。有一个例外称为线程局部变量，我们稍后会看到。

然而，共享内存有其自身的注意事项，正如我们将在“线程安全”部分看到的那样，在`threading`的情况下。由于多个线程和/或进程可以同时写入同一块内存，这是一个固有的风险操作。最坏的情况是，您的更改可能会因为冲突的写入而丢失；最坏的情况是，您的内存可能会损坏，这甚至可能导致解释器崩溃。幸运的是，Python 在保护您方面做得相当不错，所以如果您没有做任何太特别的事情，您不必担心解释器崩溃。

## 进程间的共享内存

Python 提供了几种不同的结构来确保进程间内存共享的安全性：

+   `multiprocessing.Value`

+   `multiprocessing.Array`

+   `multiprocessing.shared_memory.SharedMemory`

+   `multiprocessing.shared_memory.ShareableList`

让我们深入了解这些类型中的一些，以演示如何使用它们。

对于共享原始值，您可以使用`multiprocessing.Value`和`multiprocessing.Array`。它们基本上是相同的，但`Array`可以存储多个值，而`Value`只是一个单个值。作为参数，它们期望一个与 Python 中`array`模块工作方式相同的 typecode，这意味着它们映射到 C 类型。这导致`d`是一个双精度（浮点）数字，`i`是一个有符号整数，`b`是一个有符号字符等。

对于更多选项，请查看`array`模块的文档：[`docs.python.org/3/library/array.html`](https://docs.python.org/3/library/array.html)。

对于更高级的类型，您可以查看`multiprocessing.sharedctypes`模块，这也是`Value`和`Array`类起源的地方。

`multiprocessing.Value`和`multiprocessing.Array`都不难使用，但它们在我看来并不非常 Pythonic：

```py
import multiprocessing

some_int = multiprocessing.Value('i', 123)
with some_int.get_lock():
    some_int.value += 10
print(some_int.value)

some_double_array = multiprocessing.Array('d', [1, 2, 3])
with some_double_array.get_lock():
    some_double_array[0] += 2.5
print(some_double_array[:]) 
```

如果您需要共享内存并且性能对您来说很重要，请随意使用它们。然而，如果可能的话，我建议您避免使用它们（或者在可能的情况下避免共享内存），因为其使用起来至少是笨拙的。

`multiprocessing.shared_memory.SharedMemory`对象类似于`Array`，但它是一个更低级的结构。它为您提供了一个接口，可以读写一个可选的**命名**内存块，这样您也可以通过名称从其他进程访问它。此外，当您使用完毕后，*必须*调用`unlink()`来释放内存：

```py
from multiprocessing import shared_memory

# From process A we could write something
name = 'share_a'
share_a = shared_memory.SharedMemory(name, create=True, size=4)
share_a.buf[0] = 10

# From a different process, or the same one, we can access the data
share_a = shared_memory.SharedMemory(name)
print(share_a.buf[0])

# Make sure to clean up after. And only once!
share_a.unlink() 
```

如此例所示，第一次调用有一个`create=True`参数，用于请求操作系统内存。只有在那时（并且在调用`unlink()`之前），我们才能从其他（或相同的）进程引用该块。

再次强调，这并不是最 Pythonic 的接口，但它可以有效地共享内存。由于名称是可选的，否则会自动生成，因此您可以在创建共享内存块时省略它，并从`share_a.name`读取它。同样，像`Array`和`Value`对象一样，它也有一个固定的大小，不能在不替换它的情况下增长。

最后，我们有 `multiprocessing.shared_memory.ShareableList` 对象。虽然这个对象比 `Array` 和 `SharedMemory` 略为方便，因为它允许你灵活地处理类型（例如，`item[0]` 可以是 `str`，而 `item[1]` 可以是 `int`），但它仍然是一个难以使用的接口，并且不允许你调整大小。虽然你可以更改项目类型，但不能调整对象的大小，所以用较大的字符串替换数字将不起作用。至少它的使用比其他选项更符合 Python 风格：

```py
from multiprocessing import shared_memory

shared_list = shared_memory.ShareableList(['Hi', 1, False, None])
# Changing type from str to bool here
shared_list[0] = True
# Don't forget to unlink()
shared_list.shm.unlink() 
```

在看到这些进程间共享内存的选项后，你应该使用它们吗？是的，如果你需要高性能的话。

这应该是一个很好的迹象，为什么在并行处理中最好保持内存局部化。进程间共享内存是一个复杂的问题。即使有这些方法，它们是最快且最简单的，但仍然有点麻烦。

那么，内存共享对性能的影响有多大？让我们运行一些基准测试来看看共享变量和返回变量进行后处理之间的差异。首先，不使用共享内存作为性能基准的版本：

```py
import multiprocessing

def triangle_number_local(n):
    total = 0
    for i in range(n + 1):
        total += i

    return total

def bench_local(n, count):
    with multiprocessing.Pool() as pool:
        results = pool.imap_unordered(
            triangle_number_local,
            (n for _ in range(count)),
        )
        print('Sum:', sum(results)) 
```

`triangle_number_local()` 函数计算从 `n` 到包括 `n` 在内的所有数字之和，并返回它，类似于阶乘函数，但使用加法代替。

`bench_local()` 函数调用 `triangle_number_local()` 函数 `count` 次，并存储结果。之后，我们使用 `sum()` 函数来验证输出。

现在让我们看看使用共享内存的版本：

```py
import multiprocessing

class Shared:
    pass

def initializer(shared_value):
    Shared.value = shared_value

def triangle_number_shared(n):
    for i in range(n + 1):
        with Shared.value.get_lock():
            Shared.value.value += i

def bench_shared(n, count):
    shared_value = multiprocessing.Value('i', 0)

    # We need to explicitly share the shared_value. On Unix you
    # can work around this by forking the process, on Windows it
    # would not work otherwise
    pool = multiprocessing.Pool(
        initializer=initializer,
        initargs=(shared_value,),
    )

    iterable = (n for _ in range(count))
    list(pool.imap_unordered(triangle_number_shared, iterable))
    print('Sum:', shared_value.value)

    pool.close() 
```

在这种情况下，我们创建了一个 `Shared` 类作为命名空间来存储共享变量，但使用全局变量也是一个选项。

为了确保共享变量可用，我们需要使用 `initializer` 方法参数将其发送到 `pool` 中的所有工作进程。

此外，由于 `+=` 操作不是原子的（不是一个单一的操作，因为它需要 *fetch, add, set*），我们需要确保使用 `get_lock()` 方法锁定变量。

本章后面的 *线程安全* 部分将更详细地介绍何时需要锁定以及何时不需要。

为了运行基准测试，我们使用以下代码：

```py
import timeit

if __name__ == '__main__':
    n = 1000
    count = 100
    number = 5

    for function in 'bench_local', 'bench_shared':
        statement = f'{function}(n={n}, count={count})'
        result = timeit.timeit(
            statement, number=number,
            setup=f'from __main__ import {function}',
        )
        print(f'{statement}: {result:.3f}') 
```

现在当我们执行这个操作时，我们看到如果不共享内存会更好：

```py
bench_local(n=1000, count=100): 0.598
bench_shared(n=1000, count=100): 4.157 
```

使用共享内存的代码大约慢了 8 倍，这是有道理的，因为我的机器有 8 个核心。由于共享内存示例的大部分时间都花在锁定/解锁（只能由一个进程同时执行）上，我们实际上使代码再次在单个核心上运行。

我应该指出，这几乎是共享内存的最坏情况。由于所有函数所做的只是写入共享变量，大部分时间都花在锁定和解锁变量上。如果你在函数中实际进行处理，并且只写入结果，那会好得多。

你可能好奇我们如何正确地重写这个示例，同时仍然使用共享变量。在这种情况下，这相当简单，但这在很大程度上取决于你的具体用例，这可能不适合你：

```py
def triangle_number_shared_efficient(n):
    total = 0
    for i in range(n + 1):
        total += i

    with Shared.value.get_lock():
        Shared.value.value += total 
```

这段代码的运行速度几乎与 `bench_local()` 函数一样快。作为一个经验法则，只需记住尽可能减少锁的数量和写入次数。

使用管理器在进程间共享数据

现在我们已经看到了如何直接共享内存以获得最佳性能，让我们看看一个更方便、更灵活的解决方案：`multiprocessing.Manager` 类。

与共享内存限制我们只能使用原始类型相比，如果我们愿意牺牲一点性能，使用 `Manager` 我们可以非常容易地共享任何可以被序列化的东西。它使用的机制非常不同；它通过网络连接连接。这种方法的一个巨大优势是，你甚至可以在多个设备上使用它（我们将在本章后面看到）。

`Manager` 本身不是你经常使用的对象，尽管你可能会使用 `Manager` 提供的对象。列表很多，所以我们只详细说明其中的一些，但你总是可以查看 Python 文档以获取当前选项列表：[`docs.python.org/3/library/multiprocessing.html#managers`](https://docs.python.org/3/library/multiprocessing.html#managers)。

在使用 `multiprocessing` 共享数据时，最方便的选项之一是 `multiprocessing.Namespace` 对象。`Namespace` 对象的行为与常规对象非常相似，区别在于它可以作为共享内存对象从所有进程中访问。只要你的对象可以被序列化，你就可以将它们用作 `Namespace` 实例的属性。为了说明 `Namespace` 的用法：

```py
import multiprocessing

manager = multiprocessing.Manager()
namespace = manager.Namespace()

namespace.spam = 123
namespace.eggs = 456 
```

如你所见，你可以像对待常规对象一样简单地设置 `namespace` 的属性，但它们在所有进程之间是共享的。由于锁定现在是通过网络套接字进行的，所以开销甚至比共享内存还要大，所以只有在必须时才写入数据。将早期的共享内存示例直接转换为使用 `Namespace` 和显式的 `Lock`（`Namespace` 没有提供 `get_lock()` 方法）会产生以下代码：

```py
def triangle_number_namespace(namespace, lock, n):
    for i in range(n + 1):
        with lock:
            namespace.total += i

def bench_manager(n, count):
    manager = multiprocessing.Manager()
    namespace = manager.Namespace()
    namespace.total = 0
    lock = manager.Lock()
    with multiprocessing.Pool() as pool:
        list(pool.starmap(
            triangle_number_namespace,
            ((namespace, lock, n) for _ in range(count)),
        ))
        print('Sum:', namespace.total) 
```

与共享内存示例一样，这是一个非常低效的情况，因为我们为循环的每次迭代都进行了锁定，这一点非常明显。虽然本地版本大约需要 0.6 秒，共享内存版本大约需要 4 秒，但这个版本在实际上相同的操作中却需要惊人的 90 秒。

再次强调，我们可以通过减少在同步/锁定代码中花费的时间来轻松提高速度：

```py
def triangle_number_namespace_efficient(namespace, lock, n):
    total = 0
    for i in range(n + 1):
        total += i

    with lock:
        namespace.total += i 
```

当使用之前相同的基准代码对这一版本进行基准测试时，我们可以看到它仍然比本地版本得到的 0.6 秒慢得多：

```py
bench_local(n=1000, count=100): 0.637
bench_manager(n=1000, count=100): 1.476 
```

话虽如此，这至少比我们原本可能得到的 90 秒要可接受得多。

为什么这些锁如此慢？为了设置一个合适的锁，所有各方都需要同意数据被锁定，这是一个需要时间的过程。这个简单的事实比大多数人预期的要慢得多。运行`Manager`的服务器/进程需要向客户端确认它已经获得了锁；只有完成这一步后，客户端才能读取、写入并再次释放锁。

在常规的硬盘设置中，由于锁定和磁盘延迟，数据库服务器无法处理每秒超过大约 10 次*同一行*的事务。使用懒同步文件、SSD 和电池备份的 RAID 缓存，这种性能可以提高，以处理，也许，每秒 100 次同一行的事务。这些都是简单的硬件限制；因为你有多个进程试图写入单个目标，你需要同步进程之间的操作，这需要花费很多时间。

即使有最快的硬件，同步也可能锁定所有进程并产生巨大的减速，所以如果可能的话，尽量减少在多个进程之间共享数据。简单来说，如果所有进程都在不断从/向同一对象读取和写入，那么通常使用单个进程更快，因为锁定实际上会有效地限制你只能使用单个进程。

Redis，这是可用的最快的数据存储系统之一，直到 2020 年一直完全单线程超过十年，因为锁定开销不值得其带来的好处。即使是当前的线程版本，实际上也是一组具有自己内存空间的单线程服务器，以避免锁定。

## 线程安全

当与线程或进程一起工作时，你需要意识到你可能在某个时间点不是唯一修改变量的人。有许多场景中这不会成为问题，而且通常你很幸运，它不会影响你，但一旦发生，它可能会引起极其难以调试的错误。

例如，想象有两个代码块同时增加一个数字，想象一下可能会出错的情况。最初，让我们假设值是 10。在多个线程的情况下，这可能会导致以下序列：

1.  两个线程将数字取到本地内存中增加。目前对两个都是 10。

1.  两个线程将它们本地内存中的数字增加到 11。

1.  两个线程都将数字从本地内存（对两个都是 11）写回全局内存，所以全局数字现在是 11。

由于两个线程同时获取了数字，一个线程用它的增加覆盖了另一个线程的增加。所以，你现在的变量只增加了一次，而不是增加两次。

在许多情况下，CPython 中当前的 GIL 实现会在使用`threading`时保护你免受这些问题的影响，但你绝不应该把这个保护当作理所当然，并确保在多个线程可能同时更新你的变量时保护你的变量。

可能一个实际的代码示例可以使场景更加清晰：

```py
import time
import concurrent.futures

counter = 10

def increment(name):
    global counter
    current_value = counter
    print(f'{name} value before increment: {current_value}')
    counter = current_value + 1
    print(f'{name} value after increment: {counter}')

print(f'Before thread start: {counter}')

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(increment, range(3))
print(f'After thread finish: {counter}') 
```

如你所见，`increment`函数将`counter`存储在一个临时变量中，打印它，并在将其加 1 后写入`counter`。这个例子显然是有点牵强的，因为你通常会做`counter += 1`，这样可以减少意外行为的发生，但即使在这种情况下，你也没有保证你的结果是正确的。

为了说明这个脚本的输出：

```py
$ python3 T_12_thread_safety.py
Before thread start: 10
0 value before increment: 10
0 value after increment: 11
1 value before increment: 11
1 value after increment: 12
2 value before increment: 11
2 value after increment: 12
4 value before increment: 12
4 value after increment: 13
3 value before increment: 12
3 value after increment: 13
After thread finish: 13 
```

为什么最后结果是 13？纯粹是运气。我的一些尝试结果是 15，一些是 11，还有一些是 14。这就是线程安全问题如此难以调试的原因；在一个复杂的代码库中，很难找出导致错误的真正原因，而且你无法可靠地重现这个问题。

当在多线程/多进程系统中遇到奇怪且难以解释的错误时，确保查看它们在单线程运行时是否也会发生。这样的错误很容易犯，并且很容易被引入第三方代码，而这些代码并不是为了线程安全而设计的。

要使你的代码线程安全，你有几种不同的选择：

+   这可能看起来很明显，但如果你不并行地从多个线程/进程更新共享变量，那么就没有什么好担心的。

+   在修改你的变量时使用原子操作。原子操作是在单个指令中执行的，因此永远不会出现冲突。例如，增加一个数字可以是原子操作，其中获取、增加和更新都在单个指令中完成。在 Python 中，增加通常使用`counter += 1`来完成，这实际上是一个`counter = counter + 1`的缩写。你能看到这里的问题吗？Python 不会在内部增加`counter`，而是会将新的值写入变量`counter`，这意味着它不是原子操作。

+   使用锁来保护你的变量。

了解这些线程安全代码的选项后，你可能想知道哪些操作是线程安全的，哪些不是。幸运的是，Python 确实有一些关于这个问题的文档，我强烈建议你查看它，因为将来可能会发生变化：[`docs.python.org/3/faq/library.html#what-kinds-of-global-value-mutation-are-thread-safe`](https://docs.python.org/3/faq/library.html#what-kinds-of-global-value-mutation-are-thread-safe)。

对于当前的 CPython 版本（至少是 CPython 3.10 及以下版本），由于 GIL 在保护我们，我们可以假设这些操作是原子的，因此是线程安全的：

```py
L.append(x)
L1.extend(L2)
x = L[i]
x = L.pop()
L1[i:j] = L2
L.sort()
x = y
x.field = y
D[x] = y
D1.update(D2)
D.keys() 
```

这些不是原子的，也不是线程安全的：

```py
i = i+1
L.append(L[-1])
L[i] = L[j]
D[x] = D[x] + 1 
```

我们可以做些什么来使`i = i + 1`线程安全？最明显的解决方案是使用我们自己的锁，类似于 GIL：

```py
# This lock needs to be the same object for all threads
lock = threading.Lock()
i = 0

def increment():
    global i
    with lock():
        i += 1 
```

如你所见，我们可以很容易地使用锁来保护变量的更新。我应该指出，尽管我们在这个例子中使用了`global`变量，但同样的限制也适用于类实例的属性和其他变量。

自然，这同样适用于`多进程`，细微的区别在于变量在默认情况下不会在多个进程间共享，因此你需要做些事情来明确地引发问题。话虽如此，如果你从这些先前的共享内存和`Manager`示例中移除锁，它们会立即崩溃。

## 死锁

现在你已经知道了如何以线程安全的方式更新变量，你可能希望我们已经解决了线程的限制。不幸的是，事实正好相反。我们用来使变量更新线程安全的锁实际上可能引入另一个问题，这个问题可能更加难以解决：**死锁**。

当线程或进程在等待另一个线程/进程释放锁的同时持有锁时，可能会发生死锁。在某些情况下，甚至可能有一个线程/进程正在等待自己。为了说明这一点，让我们假设我们有锁`a`和`b`以及两个不同的线程。现在发生以下情况：

1.  线程 0 锁定`a`

1.  线程 1 锁定`b`

1.  线程 0 等待锁`b`

1.  线程 1 等待锁`a`

现在，线程 1 正在等待线程 0 完成，反之亦然。它们都不会完成，因为它们在互相等待。

为了说明这个场景：

```py
import time
import threading

a = threading.Lock()
b = threading.Lock()

def thread_0():
    print('thread 0 locking a')
    with a:
        time.sleep(0.1)
        print('thread 0 locking b')
        with b:
            print('thread 0 everything locked')

def thread_1():
    print('thread 1 locking b')
    with b:
        time.sleep(0.1)
        print('thread 1 locking a')
        with a:
            print('thread 1 everything locked')

threading.Thread(target=thread_0).start()
threading.Thread(target=thread_1).start() 
```

代码相对简单，但至少需要一些解释。如前所述，`thread_0`函数首先锁定`a`，然后是`b`，而`thread_1`则按相反的顺序进行。这就是导致死锁的原因；它们会各自等待对方完成。为了确保我们在这个例子中确实达到了死锁，我们有一个小的休眠，以确保`thread_0`在`thread_1`开始之前不会完成。在现实世界的场景中，你会在那段代码中放入一些需要花费时间的代码。

我们如何解决这类锁定问题？锁定策略和解决这些问题可以单独填满一章，并且有几种不同的锁定问题和解决方案。甚至可能存在一个活锁问题，其中两个线程都在尝试用相同的方法解决死锁问题，导致它们也互相等待，但锁却在不断变化。

要可视化活锁，可以想象一条狭窄的道路，两辆车从相反方向驶来。两辆车都会试图同时驾驶，并在注意到对方车辆移动时退后。重复这个过程，你就得到了一个活锁。

通常，你可以采用几种策略来避免死锁：

+   死锁只能在有多个锁的情况下发生，所以如果你的代码每次只获取一个锁，就不会有问题发生。

+   尽量保持锁的部分小，这样就不太可能意外地在那个块中添加另一个锁。这也可以帮助性能，因为锁可以使你的并行代码再次变成单线程。

+   这可能是解决死锁最重要的提示。*始终保持一致的锁定顺序。* 如果你总是以相同的顺序锁定，你就永远不会遇到死锁。让我们解释一下这如何帮助：在先前的例子和两个锁`a`和`b`中，问题发生是因为线程 0 正在等待`b`，而线程 1 正在等待`a`。如果它们都尝试先锁定`a`然后是`b`，我们就永远不会达到死锁状态，因为其中一个线程会锁定`a`，这会导致另一个线程在`b`被锁定之前就长时间停滞。

## 线程局部变量

我们已经看到了如何锁定变量，使得只有一个线程可以同时修改一个变量。我们也看到了在使用锁时如何防止死锁。如果我们想给一个线程提供一个独立的全局变量呢？这就是`threading.local`发挥作用的地方：它为你的当前线程提供了一个特定的上下文。例如，对于数据库连接来说，你可能希望每个线程都有自己的数据库连接，但传递连接很不方便，所以全局变量或连接管理器是一个更方便的选择。

这个部分不适用于`multiprocessing`，因为变量不会在进程之间自动共享。然而，一个派生的进程可以继承父进程的变量，因此必须小心显式初始化非共享资源。

让我们用一个简单的例子来说明线程局部变量的用法：

```py
import threading
import concurrent.futures

context = threading.local()

def init_counter():
    context.counter = 10

def increment(name):
    current_value = context.counter
    print(f'{name} value before increment: {current_value}')
    context.counter = current_value + 1
    print(f'{name} value after increment: {context.counter}')

init_counter()
print(f'Before thread start: {context.counter}')

with concurrent.futures.ThreadPoolExecutor(
        initializer=init_counter) as executor:
    executor.map(increment, range(5))

print(f'After thread finish: {context.counter}') 
```

这个例子在很大程度上与线程安全示例相同，但我们现在使用`threading.local()`作为上下文来设置`counter`变量，而不是使用全局的`counter`变量。在这里，我们还使用了`concurrent.futures.ThreadPoolExecutor`的一个额外功能，即`initializer`函数。由于线程局部变量只存在于那个线程中，并且不会自动复制到其他线程，所以所有线程（包括主线程）都需要单独设置`counter`。如果没有设置它，我们会得到一个`AttributeError`。

当运行代码时，我们可以看到所有线程都在独立地更新它们的变量，而不是我们在线程安全示例中看到的完全混合的版本：

```py
$ python3 T_15_thread_local.py
Before thread start: 10
0 value before increment: 10
0 value after increment: 11
1 value before increment: 10
2 value before increment: 11
1 value after increment: 11
3 value before increment: 10
3 value after increment: 11
2 value after increment: 12
4 value before increment: 10
4 value after increment: 11
After thread finish: 10 
```

如果可能的话，我总是建议从一个线程返回变量或将它们追加到后处理队列中，而不是更新全局变量或全局状态，因为这更快且更不容易出错。在这些情况下使用线程局部变量真的可以帮助你确保只有一个连接或集合类的实例。

现在我们已经知道了如何共享（或停止共享）变量，是时候了解使用线程而不是进程的优点和缺点了。我们现在应该对线程和进程的内存管理有一个基本的了解。有了所有这些选项，我们应该选择哪一个，为什么？

# 进程、线程，还是单线程？

现在我们已经知道了如何使用`multiprocessing`、`threading`和`concurrent.futures`，对于你的情况，你应该选择哪一个？

由于`concurrent.futures`实现了`threading`和`multiprocessing`，你可以在这个部分心理上将`threading`替换为`concurrent.futures.ThreadPoolExecutor`。当然，对于`multiprocessing`和`concurrent.futures.ProcessPoolExecutor`也是同样的道理。

当我们考虑单线程、多线程和多进程之间的选择时，有多个因素我们可以考虑。

你应该问自己的第一个也是最重要的问题是，你是否真的需要使用`threading`或`multiprocessing`。通常，代码已经足够快，你应该问问自己处理内存共享等潜在副作用的开销是否值得。不仅当涉及到并行处理时编写代码变得更加复杂，调试的复杂性也会随之增加。

其次，你应该问问自己是什么限制了你的表现。如果限制是外部 I/O，那么使用`asyncio`或`threading`来处理可能会有所帮助，但这并不能保证。

例如，如果你正在从慢速硬盘读取大量文件，线程可能甚至帮不上忙。如果硬盘是限制因素，无论你尝试什么，它都不会变快。所以在你重写整个代码库以使用`threading`之前，确保测试你的解决方案是否有任何成功的可能性。

假设你的 I/O 瓶颈可以得到缓解，那么你仍然可以选择`asyncio`与`threading`之间的选择。由于`asyncio`是所有可用选项中最快的，如果它与你的代码库兼容，我会选择这个解决方案，但当然使用`threading`也不是一个坏的选择。

如果由于 Python 代码中的大量计算，GIL 是你的瓶颈，那么`multiprocessing`可以帮到你很多。但即使在那些情况下，`multiprocessing`也不是你的唯一选择；对于许多慢速过程，使用快速库如`numpy`也可能有所帮助。

我非常喜爱`multiprocessing`库，它是迄今为止我见过的最简单的多进程代码实现之一，但它仍然伴随着一些需要注意的问题，比如更复杂的内存管理和死锁，正如我们所看到的。所以始终考虑你是否真的需要这个解决方案，以及你的问题是否适合多进程。如果大部分代码是用函数式编程编写的，那么实现起来可能非常简单；如果你需要与大量外部资源，如数据库，进行交互，那么实现起来可能非常困难。

## `threading`与`concurrent.futures`

当您有选择时，您应该使用`threading`还是`concurrent.futures`？在我看来，这取决于您想要做什么。

`threading`相对于`concurrent.futures`的优势是：

+   我们可以显式指定线程的名称，这在许多操作系统的任务管理器中可以看到。

+   我们可以显式创建并启动一个长时间运行的线程来执行一个函数，而不是依赖于线程池中的可用性。

如果您的场景允许您选择，我相信您应该使用`concurrent.futures`而不是`threading`，以下是一些原因：

+   使用`concurrent.futures`，您可以通过使用`concurrent.futures.ProcessPoolExecutor`而不是`concurrent.futures.ThreadPoolExecutor`在线程和进程之间切换。

+   使用`concurrent.futures`，您有`map()`方法可以轻松批量处理项目列表，而无需设置和关闭线程的（潜在）开销。

+   `concurrent.futures`方法返回的`concurrent.futures.Future`对象允许对结果和处理的细粒度控制。

## `multiprocessing`与`concurrent.futures`

当涉及到多进程时，我认为`concurrent.futures`接口相比线程提供的优势要少得多，尤其是`multiprocessing.Pool`基本上提供了与`concurrent.futures.ProcessPoolExecutor`几乎相同的接口。

`multiprocessing`相对于`concurrent.futures`的优势是：

+   许多高级映射方法，如`imap_unordered`和`starmap`。

+   对池有更多控制（即`terminate()`、`close()`）。

+   它可以在多台机器上使用。

+   您可以手动指定启动方法（`fork`、`spawn`或`forkserver`），这使您能够控制变量从父进程复制的方式。

+   您可以选择 Python 解释器。使用`multiprocessing.set_executable()`，您可以在运行 Python 3.9 的主进程的同时运行 Python 3.10 的进程池。

`concurrent.futures`相对于`multiprocessing`的优势是：

+   您可以轻松切换到`concurrent.futures.ThreadPoolExecutor`。

+   与`multiprocessing`使用的`AsyncResult`对象相比，返回的`Future`对象允许对结果处理进行更细粒度的控制。

个人而言，如果您不需要与`threads`兼容的映射方法，我更倾向于使用`multiprocessing`。

# 超线程与物理 CPU 核心

超线程是一种技术，它为物理核心提供额外的虚拟 CPU 核心。其理念是，由于这些虚拟 CPU 核心有独立的缓存和其他资源，您可以在多个任务之间更有效地切换。如果您在两个重负载进程之间进行任务切换，CPU 就不必卸载/重新加载所有缓存。然而，当涉及到实际的 CPU 指令处理时，它并不会帮助您。

当你真正最大化 CPU 使用时，通常最好只使用物理处理器数量。为了展示这如何影响性能，我们将使用几个进程数运行一个简单的测试。由于我的处理器有 `8` 个核心（如果包括超线程则是 `16` 个），我们将使用 `1`、`2`、`4`、`8`、`16` 和 `32` 个进程来展示它如何影响性能：

```py
import timeit
import multiprocessing

def busy_wait(n):
    while n > 0:
        n -= 1

def benchmark(n, processes, tasks):
    with multiprocessing.Pool(processes=processes) as pool:
        # Execute the busy_wait function 'tasks' times with
        # parameter n
        pool.map(busy_wait, [n for _ in range(tasks)])
    # Create the executor

if __name__ == '__main__':
    n = 100000
    tasks = 128
    for exponent in range(6):
        processes = int(2 ** exponent)
        statement = f'benchmark({n}, {processes}, {tasks})'
        result = timeit.timeit(
            statement,
            number=5,
            setup='from __main__ import benchmark',
        )
        print(f'{statement}: {result:.3f}') 
```

为了让处理器保持忙碌，我们在 `busy_wait()` 函数中使用从 `n` 到 `0` 的 `while` 循环。对于基准测试，我们使用具有给定进程数的 `multiprocessing.Pool()` 实例，并运行 `busy_wait(100000)` 128 次：

```py
$ python3 T_16_hyper_threading.py
benchmark(100000, 1): 3.400
benchmark(100000, 2): 1.894
benchmark(100000, 4): 1.208
benchmark(100000, 8): 0.998
benchmark(100000, 16): 1.124
benchmark(100000, 32): 1.787 
```

如你所见，在我的启用超线程的 `8` 核心 CPU 上，具有 `8` 线程的版本显然是最快的。尽管操作系统任务管理器显示 `16` 个核心，但使用超过 `8` 个物理核心并不总是更快的。此外，由于现代处理器的提升行为，你可以看到使用 `8` 个处理器仅比单线程版本快 `3.4` 倍，而不是预期的 `8` 倍加速。

这说明了在用指令大量加载处理器时超线程的问题。一旦单个进程实际上使用了 CPU 核心的 100%，进程之间的任务切换实际上会降低性能。由于只有 `8` 个物理核心，其他进程必须争夺在处理器核心上完成一些事情。别忘了系统上的其他进程以及操作系统本身也会消耗一些处理能力。

如果你确实因为 CPU 密集型问题而迫切需要性能，那么匹配物理 CPU 核心通常是最佳解决方案，但如果锁定是瓶颈，那么由于 CPU 提升行为，单线程可能比任何多线程解决方案都快。

如果你预期不会一直最大化所有核心，那么我建议不要将 `processes` 参数传递给 `multiprocessing.Pool()`，这将使其默认为 `os.cpu_count()`，它返回所有核心，包括超线程核心。

然而，这完全取决于你的用例，而确定唯一的方法是测试你的特定场景。作为一个经验法则，我建议以下做法：

+   磁盘 I/O 受限？单个进程可能是最佳选择。

+   CPU 受限？物理 CPU 核心的数量是你的最佳选择。

+   网络 I/O 受限？从默认值开始，如有需要则调整。这是在 8 核心处理器上仍然可以使用 128 线程的少数情况之一。

+   没有明显的限制，但需要许多（数百个）并行过程？也许你应该尝试使用 `asyncio` 而不是 `multiprocessing`。

注意，创建多个进程在内存和打开的文件方面并不是免费的；虽然你可以有几乎无限的协程数量，但对于进程来说并非如此。根据你的操作系统配置，你可能在达到 100 个进程之前就达到最大打开文件限制，即使你达到了这些数字，CPU 调度也将成为你的瓶颈。

如果我们的 CPU 核心不够用，我们应该怎么做？简单：使用更多的 CPU 核心。我们从哪里得到这些核心？多台计算机！是时候过渡到分布式计算了。

# 远程进程

到目前为止，我们只在多个本地处理器上执行了我们的脚本，但实际上我们可以做得更多。使用 `multiprocessing` 库，实际上在远程服务器上执行作业非常容易，但目前的文档仍然有些晦涩。实际上有几种方式可以以分布式的方式执行进程，但最明显的方法并不是最容易的方法。`multiprocessing.connection` 模块提供了 `Client` 和 `Listener` 类，它们以简单的方式促进了客户端和服务器之间的安全通信。

通信与进程管理和队列管理并不相同；这些功能需要额外的努力。`multiprocessing` 库在这方面仍然比较简单，但只要有几个不同的进程，这绝对是可能的。

## 使用多进程的分布式处理

我们将从一个模块开始，其中包含一些应该由所有客户端和服务器共享的常量，这样秘密密码和服务器的主机名就可以对所有客户端和服务器可用。除此之外，我们还将添加我们的素数计算函数，这些函数我们稍后会用到。以下模块中的导入将期望此文件存储为 `T_17_remote_multiprocessing/constants.py`，但只要你确保导入和引用正常工作，你可以随意命名：

```py
host = 'localhost'
port = 12345
password = b'some secret password' 
```

接下来，我们定义需要供服务器和客户端都可用到的函数。我们将将其存储为 `T_17_remote_multiprocessing/functions.py`：

```py
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

现在是时候创建实际的连接函数和作业队列的服务器了。我们将将其存储为 `T_17_remote_multiprocessing/server.py`：

```py
import multiprocessing
from multiprocessing import managers

import constants
import functions

queue = multiprocessing.Queue()
manager = managers.BaseManager(address=('', constants.port),
                               authkey=constants.password)

manager.register('queue', callable=lambda: queue)
manager.register('primes', callable=functions.primes)

server = manager.get_server()
server.serve_forever() 
```

在创建服务器之后，我们需要有一个客户端脚本来发送作业。你可以使用单个脚本进行发送和处理，但为了保持逻辑清晰，我们将使用单独的脚本。

以下脚本将把 `0` 到 `999` 添加到队列中进行处理。我们将将其存储为 `T_17_remote_multiprocessing/submitter.py`：

```py
from multiprocessing import managers

import constants

manager = managers.BaseManager(
    address=(constants.host, constants.port),
    authkey=constants.password)
manager.register('queue')
manager.connect()

queue = manager.queue()
for i in range(1000):
    queue.put(i) 
```

最后，我们需要创建一个客户端来实际处理队列。我们将将其存储为 `T_17_remote_multiprocessing/client.py`：

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

从前面的代码中，您可以看到我们如何传递函数；管理器允许注册可以从客户端调用的函数和类。有了这个，我们就传递了一个来自多进程类的队列，这个队列对多线程和多进程都是安全的。

现在，我们需要启动进程本身。首先，持续运行的服务器：

```py
$ python3 T_17_remote_multiprocessing/server.py 
```

然后，运行生产者以生成素数生成请求：

```py
$ python3 T_17_remote_multiprocessing/submitter.py 
```

现在，我们可以在多台机器上运行多个客户端以获取前 1,000 个素数。由于这些客户端现在正在打印前 1,000 个素数，输出有点太长，无法在此显示，但您可以简单地并行多次运行或在多台机器上运行以生成您的输出：

```py
$ python3 T_17_remote_multiprocessing/client.py 
```

如果您想将输出发送到不同的进程，您可以使用队列或管道代替打印。但是，如您所见，并行处理事物仍然需要一些工作，并且需要一些代码同步才能工作。有几个替代方案可用，例如 **Redis**、**ØMQ**、**Celery**、**Dask** 和 **IPython Parallel**。哪个是最好的和最适合取决于您的用例。如果您只是想处理多个 CPU 上的任务，那么 `multiprocessing`、Dask 和 IPython Parallel 可能是您最好的选择。如果您正在寻找后台处理和/或轻松地将任务卸载到多台机器上，那么 ØMQ 和 Celery 是更好的选择。

## 使用 Dask 进行分布式处理

Dask 库正在迅速成为分布式 Python 执行的标准。它与许多科学 Python 库（如 NumPy 和 Pandas）有非常紧密的集成，使得在许多情况下并行执行完全透明。这些库在 *第十五章*，*科学 Python 和绘图* 中有详细的介绍。

Dask 库提供了一个易于使用的并行接口，可以执行单线程、使用多个线程、使用多个进程，甚至使用多台机器。只要您牢记多线程、进程和多台机器的数据共享限制，您就可以轻松地在它们之间切换，以查看哪个最适合您的用例。

### 安装 Dask

Dask 库由多个包组成，您可能不需要所有这些包。总的来说，`Dask` 包只是核心，我们可以从几个附加功能中选择，这些可以通过 `pip install dask[extra]` 安装：

+   `array`：添加了一个类似于 `numpy.ndarray` 的数组接口。内部，这些结构由多个 `numpy.ndarray` 实例组成，分布在您的 Dask 集群中，以便于并行处理。

+   `dataframe`：类似于数组接口，这是一个 `pandas.DataFrame` 对象的集合。

+   `diagnostics`：添加了分析器、进度条，甚至一个完全交互式的仪表板，可以实时显示当前运行作业的信息。

+   `distributed`：运行 Dask 在多个系统上而不是仅本地所需的包。

+   `complete`：所有上述附加功能。

对于本章的演示，我们需要至少安装`distributed`扩展，因此您需要运行以下之一：

```py
$ pip3 install -U "dask[distributed]" 
```

或者：

```py
$ pip3 install -U "dask[complete]" 
```

如果您正在使用 Jupyter 笔记本进行实验，`diagnostics`扩展中的进度条也支持 Jupyter，这可能很有用。

### 基本示例

让我们从执行一些代码的基本示例开始，通过 Dask 而不显式设置集群。为了说明这如何有助于性能，我们将使用`busy-wait`循环来最大化 CPU 负载。在这种情况下，我们将使用`dask.distributed`子模块，它有一个与`concurrent.futures`非常相似的接口：

```py
import sys
import datetime

from dask import distributed

def busy_wait(n):
    while n > 0:
        n -= 1

def benchmark_dask(client):
    start = datetime.datetime.now()

    # Run up to 1 million
    n = 1000000
    tasks = int(sys.argv[1])  # Get number of tasks from argv

    # Submit the tasks to Dask
    futures = client.map(busy_wait, [n] * tasks, pure=False)
    # Gather the results; this blocks until the results are ready
    client.gather(futures)

    duration = datetime.datetime.now() - start
    per_second = int(tasks / duration.total_seconds())
    print(f'{tasks} tasks at {per_second} per '
          f'second, total time: {duration}')

if __name__ == '__main__':
    benchmark_dask(distributed.Client()) 
```

代码大部分都很直接，但有一些小细节需要注意。首先，当将任务提交给 Dask 时，您需要告诉 Dask 它是一个不纯的函数。

如果您还记得*第五章*，*函数式编程 – 可读性与简洁性之间的权衡*，函数式编程中的纯函数是没有副作用的一个；其输出是一致的，并且只依赖于输入。返回随机值的函数是不纯的，因为重复调用会返回不同的结果。

对于纯函数，Dask 会自动缓存结果。如果您有两个相同的调用，Dask 只会执行一次函数。

为了排队任务，我们需要使用`client.map()`或`client.submit()`等函数。这些函数在`concurrent.futures`的情况下与`executor.submit()`非常相似。

最后，我们需要从未来中获取结果。这可以通过调用`future.result()`或批量使用`client.gather(futures)`来完成。再次强调，这与`concurrent.futures`非常相似。

为了使代码更加灵活，我们使任务数量可配置，以便在您的系统上合理的时间内运行。如果您有一个速度慢得多或快得多的系统，您可能需要调整它以获得有用的结果。

当我们执行脚本时，我们得到以下结果：

```py
$ python3 T_18_dask.py 128
128 tasks at 71 per second, total time: 0:00:01.781836 
```

这就是您如何轻松地在所有 CPU 核心上执行一些代码。当然，我们也可以在单线程或分布式模式下进行测试；我们唯一需要改变的是如何初始化`distributed.Client()`。

### 单线程运行

让我们以单线程模式运行相同的代码：

```py
if __name__ == '__main__':
    benchmark_dask(distributed.Client()) 
```

现在如果我们运行它，我们可以看到 Dask 确实在之前使用了多个进程：

```py
$ python3 T_19_dask_single.py 128
128 tasks at 20 per second, total time: 0:00:06.142977 
```

这对于调试线程安全问题很有用。如果问题在单线程模式下仍然存在，那么线程安全问题可能不是你的问题。

### 在多台机器上分布式执行

为了实现更令人印象深刻的成就，让我们同时运行多台机器上的代码。要同时运行 Dask，有许多可用的部署选项：

+   使用`dask-scheduler`和`dask-worker`命令进行手动设置

+   使用`dask-ssh`命令通过 SSH 自动部署

+   直接部署到运行 Kubernetes、Hadoop 等现有计算集群

+   将应用程序部署到云服务提供商，如亚马逊、谷歌和微软 Azure

在这个例子中，我们将使用`dask-scheduler`，因为它是在几乎任何可以运行 Python 的机器上运行的解决方案。

注意，如果 Dask 版本和依赖项不匹配，可能会遇到错误，因此在开始之前更新到最新版本是一个好主意。

首先，我们启动`dask-scheduler`：

```py
$ dask-scheduler
[...]
distributed.scheduler - INFO - Scheduler at:  tcp://10.1.2.3:8786
distributed.scheduler - INFO - dashboard at:                :8787 
```

一旦`dask-scheduler`启动，它也将托管上面提到的仪表板，显示当前状态：`http://localhost:8787/status`。

现在，我们可以在所有需要参与的计算机上运行`dask-worker`进程：

```py
$ dask-worker --nprocs auto tcp://10.1.2.3:8786 
```

使用`--nprocs`参数，你可以设置要启动的进程数。设置为`auto`时，它将设置为包括超线程在内的 CPU 核心数。当设置为正数时，它将启动该确切数量的进程；当设置为负数时，该数值将添加到 CPU 核心数。

你的仪表板屏幕和控制台现在应该显示所有已连接的客户端。现在是时候再次运行我们的脚本了，但这次是分布式运行：

```py
if __name__ == '__main__':
    benchmark_dask(distributed.Client('localhost:8786')) 
```

这就是我们需要做的唯一一件事：配置调度器运行的位置。注意，我们也可以使用 IP 地址或主机名从其他机器连接。

让我们运行它并看看它是否变得更快：

```py
$ python3 T_20_dask_distributed.py 2048
[...]
2048 tasks at 405 per second, total time: 0:00:05.049570 
```

哇，这真是一个很大的区别！在单线程模式下我们每秒可以做`20`个任务，或者在多进程模式下每秒可以做`71`个任务，而现在我们每秒可以处理`405`个这样的任务。正如你所见，设置起来也非常简单。

Dask 库有许多更多选项来提高效率、限制内存、优先处理工作等等。我们甚至还没有涵盖通过链式连接任务或对捆绑结果运行`reduce`来组合任务。如果你的代码可以从同时在多个系统上运行中受益，我强烈建议考虑使用 Dask。

## 使用 ipyparallel 进行分布式处理

IPython Parallel 模块与 Dask 类似，使得同时处理多台计算机上的代码成为可能。需要注意的是，你可以在`ipyparallel`之上运行 Dask。该库支持比你可能需要的更多功能，但基本用法在需要执行可以受益于多台计算机的繁重计算时很重要。

首先，让我们安装最新的`ipyparallel`包和所有 IPython 组件：

```py
$ pip3 install -U "ipython[all]" ipyparallel 
```

尤其是在 Windows 上，使用 Anaconda 安装 IPython 可能更容易，因为它包括许多科学、数学、工程和数据分析包的二进制文件。为了获得一致的安装，Anaconda 安装程序也适用于 OS X 和 Linux 系统。

其次，我们需要一个集群配置。技术上，这是可选的，但既然我们打算创建一个分布式 IPython 集群，使用特定的配置文件来配置一切将更加方便：

```py
$ ipython profile create --parallel --profile=mastering_python
[ProfileCreate] Generating default config file: '~/.ipython/profile_mastering_python/ipython_config.py'
[ProfileCreate] Generating default config file: '~/.ipython/profile_mastering_python/ipython_kernel_config.py'
[ProfileCreate] Generating default config file: '~/.ipython/profile_mastering_python/ipcontroller_config.py'
[ProfileCreate] Generating default config file: '~/.ipython/profile_mastering_python/ipengine_config.py'
[ProfileCreate] Generating default config file: '~/.ipython/profile_mastering_python/ipcluster_config.py' 
```

这些配置文件包含大量的选项，所以我建议搜索特定的部分而不是逐个查看。快速列出这五个文件的总配置行数约为 2,500 行。文件名已经提供了关于配置文件用途的提示，但我们将通过解释它们的目的和一些最重要的设置来遍历这些文件。

### ipython_config.py

这是通用的 IPython 配置文件；你几乎可以在这里自定义 IPython shell 的任何内容。它定义了你的 shell 应该如何看起来，默认应该加载哪些模块，是否加载 GUI，以及更多。对于本章的目的来说，这并不是特别重要，但如果你打算更频繁地使用 IPython，它绝对值得一看。你可以在这里配置的一些事情包括自动加载扩展，例如在第十二章中讨论的`line_profiler`和`memory_profiler`。例如：

```py
c.InteractiveShellApp.extensions = [
    'line_profiler',
    'memory_profiler',
] 
```

### ipython_kernel_config.py

此文件配置你的 IPython 内核，并允许你覆盖/扩展`ipython_config.py`。为了理解其目的，了解 IPython 内核是什么很重要。在这个上下文中，内核是运行和检查代码的程序。默认情况下，这是`IPyKernel`，它是一个常规的 Python 解释器，但还有其他选项，如`IRuby`或`IJavascript`，分别用于运行 Ruby 或 JavaScript。

其中一个更有用的选项是配置内核的监听端口和 IP 地址。默认情况下，端口都设置为使用随机数，但重要的是要注意，如果你在运行内核时有人可以访问同一台机器，他们可以连接到你的 IPython 内核，这在共享机器上可能是危险的。

### ipcontroller_config.py

`ipcontroller`是 IPython 集群的主进程。它控制引擎和任务的分配，并负责诸如日志记录等任务。

在性能方面最重要的参数是`TaskScheduler`设置。默认情况下，`c.TaskScheduler.scheme_name`设置被设置为使用 Python LRU 调度器，但根据你的工作负载，其他如`leastload`和`weighted`可能更好。如果你必须在一个如此大的集群上处理如此多的任务，以至于调度器成为瓶颈，还有一个`plainrandom`调度器，如果所有机器的规格相似且任务持续时间相似，它的工作效果出奇地好。

为了我们的测试目的，我们将控制器的 IP 设置为`*`，这意味着将接受*所有*IP 地址，并且接受每个网络连接。如果您处于不安全的环境/网络中，并且/或者没有允许您选择性地启用某些 IP 地址的防火墙，那么这种方法*不推荐使用*！在这种情况下，我建议通过更安全的选择启动，例如`SSHEngineSetLauncher`或`WindowsHPCEngineSetLauncher`。

假设您的网络确实安全，将工厂 IP 设置为所有本地地址：

```py
c.HubFactory.client_ip = '*'
c.RegistrationFactory.ip = '*' 
```

现在启动控制器：

```py
$ ipcontroller --profile=mastering_python
[IPControllerApp] Hub listening on tcp://*:58412 for registration.
[IPControllerApp] Hub listening on tcp://127.0.0.1:58412 for registration.
...
 [IPControllerApp] writing connection info to ~/.ipython/profile_mastering_python/security/ipcontroller-client.json
[IPControllerApp] writing connection info to ~/.ipython/profile_mastering_python/security/ipcontroller-engine.json
... 
```

请注意写入配置目录安全目录的文件。它们包含`ipengine`用于查找和连接到`ipcontroller`的认证信息，例如加密密钥和端口信息。

### ipengine_config.py

`ipengine`是实际的工作进程。这些进程运行实际的计算，因此为了加快处理速度，您需要在尽可能多的机器上运行这些进程。您可能不需要更改此文件，但如果您想配置集中式日志记录或需要更改工作目录，它可能很有用。通常，您不希望手动启动`ipengine`进程，因为您很可能会希望每台计算机启动多个进程。这就是我们下一个命令`ipcluster`的作用所在。

### ipcluster_config.py

`ipcluster`命令实际上是一个简单的快捷方式，用于同时启动`ipcontroller`和`ipengine`的组合。对于简单的本地处理集群，我建议使用这个命令，但在启动分布式集群时，使用`ipcontroller`和`ipengine`分别使用可以提供更好的控制。在大多数情况下，该命令提供了足够多的选项，因此您可能不需要单独的命令。

最重要的配置选项是`c.IPClusterEngines.engine_launcher_class`，因为它控制着引擎和控制器之间的通信方法。此外，它也是进程之间安全通信最重要的组件。默认情况下，它设置为`ipyparallel.apps.launcher.LocalControllerLauncher`，这是为本地进程设计的，但如果您想使用 SSH 与客户端通信，`ipyparallel.apps.launcher.SSHEngineSetLauncher`也是一个选项。另外，还有`ipyparallel.apps.launcher.WindowsHPCEngineSetLauncher`用于 Windows HPC。

在我们可以在所有机器上创建集群之前，我们需要传输配置文件。您的选择是传输所有文件，或者简单地传输 IPython 配置文件`security`目录中的文件。

现在是启动集群的时候了。由于我们之前已经单独启动了`ipcontroller`，我们只需要启动引擎。在本地机器上，我们只需简单地启动它，但其他机器还没有配置。一个选择是复制整个 IPython 配置文件目录，但真正需要复制的只有一个文件，即`security/ipcontroller-engine.json`；在使用配置创建命令创建配置之后。所以，除非你打算复制整个 IPython 配置文件目录，否则你需要再次执行配置创建命令：

```py
$ ipython profile create --parallel --profile=mastering_python 
```

之后，只需复制`ipcontroller-engine.json`文件，任务就完成了。现在我们可以启动实际的引擎：

```py
$ ipcluster engines --profile=mastering_python -n 4
[IPClusterEngines] IPython cluster: started
[IPClusterEngines] Starting engines with [daemon=False]
[IPClusterEngines] Starting 4 Engines with LocalEngineSetLauncher 
```

注意，这里的`4`是为了四核处理器选择的，但任何数字都可以。默认情况下将使用逻辑处理器核心的数量，但根据工作负载，可能最好匹配物理处理器核心的数量。

现在，我们可以从我们的 IPython shell 中运行一些并行代码。为了展示性能差异，我们将使用从 0 到 10,000,000 的所有数字的简单求和。这不是一个特别重的任务，但当连续执行 10 次时，常规 Python 解释器会花费一些时间：

```py
In [1]: %timeit for _ in range(10): sum(range(10000000))
1 loops, best of 3: 2.27 s per loop 
```

然而，这次，为了说明差异，我们将运行它 100 次，以展示分布式集群有多快。请注意，这只是一个三机集群，但仍然快得多：

```py
In [1]: import ipyparallel

In [2]: client = ipyparallel.Client(profile='mastering_python')
In [3]: view = client.load_balanced_view()
In [4]: %timeit view.map(lambda _: sum(range(10000000)), range(100)).wait()
1 loop, best of 3: 909 ms per loop 
```

然而，更有趣的是在`ipyparallel`中定义并行函数。只需一个简单的装饰器，就可以将一个函数标记为并行：

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

`ipyparallel`库提供了更多有用的功能，但这本书的范围之外。尽管`ipyparallel`是 Jupyter/IPython 其他部分的独立实体，但它很好地整合在一起，使得它们结合变得足够容易。

练习

虽然为多线程和多进程做准备比为`asyncio`做准备不那么侵入性，但如果需要传递或共享变量，仍然需要一些思考。所以，这实际上是一个关于你想要让自己多难的问题。

看看你是否能创建一个作为单独进程的回声服务器和客户端。尽管我们没有涵盖`multiprocessing.Pipe()`，但我相信你无论如何都能处理它。它可以通过`a, b = multiprocessing.Pipe()`创建，你可以使用`[a/b].send()`和`[a/b].recv()`来使用它。

+   读取目录中的所有文件，并通过使用`threading`和`multiprocessing`或`concurrent.futures`（如果你想要一个更简单的练习）来读取每个文件，来计算文件的大小总和。如果你想增加难度，可以在运行时通过让线程/进程队列新项目来递归地遍历目录。

+   创建一个通过`multiprocessing.Queue()`等待项目入队的工人池。如果你能使其成为一个安全的 RPC（远程过程调用）类型操作，那么你将获得额外的分数。

+   应用你的函数式编程技能，以并行方式计算一些东西。也许并行排序？

与您在野外可能遇到的情况相比，所有这些练习仍然很遗憾地很简单。如果您真的想挑战自己，开始将这些技术（特别是内存共享）应用到您现有的或新项目中，并希望（或不是）遇到真正的挑战。

这些练习的示例答案可以在 GitHub 上找到：`github.com/mastering-python/exercises`。我们鼓励您提交自己的解决方案，并从他人的替代方案中学习。

# 概述

在本章中，我们涵盖了众多不同的主题，所以让我们总结一下：

+   Python GIL 是什么，为什么我们需要它，以及我们如何绕过它

+   何时使用线程，何时使用进程，以及何时使用 `asyncio`

+   使用 `threading` 和 `concurrent.futures` 在并行线程中运行代码

+   使用 `multiprocessing` 和 `concurrent.futures` 在并行进程中运行代码

+   在多台机器上运行分布式代码

+   在线程和进程之间共享数据

+   线程安全

+   死锁

您可以从本章中学到的最重要的课程是线程和进程之间数据同步**真的非常慢**。只要可能，您应该只向函数发送数据，并在完成时返回，中间不发送任何数据。即使在那种情况下，如果您可以发送更少的数据，请发送更少的数据。如果可能，请保持您的计算和数据本地化。

在下一章，我们将学习关于科学 Python 库和绘图的内容。这些库可以帮助您在创纪录的时间内执行困难的计算和数据处理。这些库大多数都高度优化性能，与多进程或 Dask 库配合得很好。

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：[`discord.gg/QMzJenHuJf`](https://discord.gg/QMzJenHuJf)

![二维码](img/QR_Code156081100001293319171.png)
