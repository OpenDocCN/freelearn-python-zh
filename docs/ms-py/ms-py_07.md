# 第七章：异步 IO - 无需线程的多线程

上一章向我们展示了同步协程的基本实现。然而，当涉及到外部资源时，同步协程是一个坏主意。只要一个远程连接停顿，整个进程就会挂起，除非你使用了多进程（在第十三章中有解释，*多进程 - 当单个 CPU 核心不够用*）或异步函数。

异步 IO 使得可以访问外部资源而无需担心减慢或阻塞应用程序。Python 解释器不需要主动等待结果，而是可以简单地继续执行其他任务，直到再次需要。这与 Node.js 和 JavaScript 中的 AJAX 调用的功能非常相似。在 Python 中，我们已经看到诸如`asyncore`、`gevent`和`eventlet`等库多年来已经实现了这一点。然而，随着`asyncio`模块的引入，使用起来变得更加容易。

本章将解释如何在 Python（特别是 3.5 及以上版本）中使用异步函数，以及如何重构代码，使其仍然能够正常运行，即使它不遵循标准的过程式编码模式来返回值。

本章将涵盖以下主题：

+   使用以下函数：

+   `async def`

+   `async for`

+   `async with`

+   `await`

+   并行执行

+   服务器

+   客户端

+   使用`Future`来获取最终结果

# 介绍 asyncio 库

`asyncio`库的创建是为了使异步处理更加容易，并且结果更加可预测。它的目的是取代`asyncore`模块，后者已经可用了很长时间（事实上自 Python 1.5 以来）。`asyncore`模块从来没有很好地可用，这促使了`gevent`和`eventlet`第三方库的创建。`gevent`和`eventlet`都比`asyncore`更容易实现异步编程，但我觉得随着`asyncio`的引入，它们已经基本过时了。尽管我不得不承认`asyncio`仍然有一些问题，但它正在积极开发中，这让我认为所有问题很快就会被核心 Python 库或第三方包解决。

`asyncio`库是在 Python 3.4 中正式引入的，但是可以通过 Python 包索引为 Python 3.3 提供后向端口。考虑到这一点，虽然本章的一些部分可以在 Python 3.3 上运行，但大部分是以 Python 3.5 和新引入的`async`和`await`关键字为基础编写的。

## 异步和等待语句

在继续任何示例之前，重要的是要了解 Python 3.4 和 Python 3.5 代码语法之间的关系。尽管`asyncio`库仅在 Python 3.4 中引入，但 Python 3.5 中已经替换了大部分通用语法。虽然不是强制性的，但更简单，因此推荐使用`async`和`await`的语法已经被引入。

### Python 3.4

对于传统的 Python 3.4 用法，需要考虑一些事项：

+   函数应使用`asyncio.coroutine`装饰器声明

+   应使用`yield from coroutine()`来获取异步结果

+   不直接支持异步循环，但可以使用`while True: yield from coroutine()`来模拟

以下是一个例子：

```py
import asyncio

@asyncio.coroutine
def sleeper():
    yield from asyncio.sleep(1)
```

### Python 3.5

在 Python 3.5 中，引入了一种新的语法来标记函数为异步的。可以使用`async`关键字来代替`asyncio.coroutine`装饰器。此外，Python 现在支持`await`语句，而不是令人困惑的`yield from`语法。`yield from`语句稍微令人困惑，因为它可能让人觉得正在交换值，而这并不总是情况。

以下是`async`语句：

```py
async def some_coroutine():
    pass
```

它可以代替装饰器：

```py
import asyncio

@asyncio.coroutine
def some_coroutine():
    pass
```

在 Python 3.5 中，以及很可能在未来的版本中，`coroutine`装饰器仍然受到支持，但如果不需要向后兼容性，我强烈推荐使用新的语法。

此外，我们可以使用更合乎逻辑的`await`语句，而不是`yield from`语句。因此，前面段落中的示例变得和以下示例一样简单：

```py
import asyncio

async def sleeper():
    await asyncio.sleep(1)
```

`yield from`语句源自 Python 中原始协程实现，并且是在同步协程中使用的`yield`语句的一个逻辑扩展。实际上，`yield from`语句仍然有效，而`await`语句只是它的一个包装器，增加了一些检查。在使用`await`时，解释器会检查对象是否是可等待对象，这意味着它需要是以下对象之一：

+   使用`async def`语句创建的本地协程

+   使用`asyncio.coroutine`装饰器创建的协程

+   实现`__await__`方法的对象

这个检查本身就使得`await`语句比`yield from`语句更可取，但我个人认为`await`更好地传达了语句的含义。

总之，要转换为新的语法，进行以下更改：

+   函数应该使用`async def`声明，而不是`def`

+   应该使用`await coroutine()`来获取异步结果

+   可以使用`async for ... in ...`创建异步循环

+   可以使用`async with ...`创建异步`with`语句

### 在 3.4 和 3.5 语法之间进行选择

除非你真的需要 Python 3.3 或 3.4 支持，我强烈推荐使用 Python 3.5 语法。新的语法更清晰，支持更多功能，比如异步`for`循环和`with`语句。不幸的是，它们并不完全兼容，所以你需要做出选择。在`async def`（3.5）中，我们不能使用`yield from`，但我们只需要用`await`替换`yield from`就可以解决这个问题。

## 单线程并行处理的简单示例

并行处理有很多用途：服务器同时处理多个请求，加快繁重任务的速度，等待外部资源等等。通用协程在某些情况下可以帮助处理多个请求和外部资源，但它们仍然是同步的，因此受到限制。使用`asyncio`，我们可以超越通用协程的限制，轻松处理阻塞资源，而不必担心阻塞主线程。让我们快速看一下代码如何在多个并行函数中不会阻塞：

```py
>>> import asyncio

>>> async def sleeper(delay):
...     await asyncio.sleep(delay)
...     print('Finished sleeper with delay: %d' % delay)

>>> loop = asyncio.get_event_loop()
>>> results = loop.run_until_complete(asyncio.wait((
...     sleeper(1),
...     sleeper(3),
...     sleeper(2),
... )))
Finished sleeper with delay: 1
Finished sleeper with delay: 2
Finished sleeper with delay: 3

```

即使我们按顺序开始了睡眠器，1、3、2，它们会按照相应的时间睡眠，`asyncio.sleep`结合`await`语句实际上告诉 Python，它应该继续处理需要实际处理的任务。普通的`time.sleep`实际上会阻塞 Python 任务，这意味着它们会按顺序执行。这使得它更加透明，可以处理任何类型的等待，我们可以将其交给`asyncio`，而不是让整个 Python 线程忙碌。因此，我们可以用`while True: fh.read()`来代替，只要有新数据就可以立即响应。

让我们分析一下这个例子中使用的组件：

+   `asyncio.coroutine`：这个装饰器使得可以从`async def`协程中进行 yield。除非你使用这种语法，否则没有真正需要这个装饰器，但如果只用作文档，这是一个很好的默认值。

+   `asyncio.sleep`：这是`time.sleep`的异步版本。这两者之间的主要区别是，`time.sleep`在睡眠时会让 Python 进程保持忙碌，而`asyncio.sleep`允许在事件循环中切换到不同的任务。这个过程与大多数操作系统中的任务切换的工作方式非常相似。

+   `asyncio.get_event_loop`：默认事件循环实际上是`asyncio`任务切换器；我们将在下一段解释更多关于这些的内容。

+   `asyncio.wait`：这是用于包装一系列协程或未来并等待结果的协程。等待时间是可配置的，等待方式也是可配置的（首先完成，全部完成，或者第一个异常）。

这应该解释了示例的基本工作原理：`sleeper`函数是异步协程，经过给定的延迟后退出。`wait`函数在退出之前等待所有协程完成，`event`循环用于在三个协程之间切换。

## `asyncio`的概念

`asyncio`库有几个基本概念，必须在我们进一步探讨示例和用法之前加以解释。前一段中显示的示例实际上使用了大部分这些概念，但对于如何以及为什么可能仍然有一些解释是有用的。

`asyncio`的主要概念是*协程*和*事件循环*。在其中，还有几个可用的辅助类，如`Streams`、`Futures`和`Processes`。接下来的几段将解释基础知识，以便你能够理解后面段落中的示例中的实现。

### 未来和任务

`asyncio.Future`类本质上是一个结果的承诺；如果结果可用，它会返回结果，并且一旦接收到结果，它将把结果传递给所有注册的回调函数。它在内部维护一个状态变量，允许外部方将未来标记为已取消。API 与`concurrent.futures.Future`类非常相似，但由于它们并不完全兼容，所以请确保不要混淆两者。

`Future`类本身并不那么方便使用，这就是`asyncio.Task`发挥作用的地方。`Task`类包装了一个协程，并自动处理执行、结果和状态。协程将通过给定的事件循环执行，或者如果没有给定，则通过默认事件循环执行。

这些类的创建并不是你需要直接担心的事情。这是因为推荐的方式是通过`asyncio.ensure_future`或`loop.create_task`来创建类。前者实际上在内部执行了`loop.create_task`，但如果你只想在主/默认事件循环上执行它而不必事先指定，那么这种方式更方便。使用起来非常简单。要手动创建自己的未来，只需告诉事件循环为你执行`create_task`。下面的示例由于所有的设置代码而有点复杂，但 C 的使用应该是清楚的。最重要的一点是事件循环应该被链接，以便任务知道如何/在哪里运行：

```py
>>> import asyncio

>>> async def sleeper(delay):
...     await asyncio.sleep(delay)
...     print('Finished sleeper with delay: %d' % delay)

# Create an event loop
>>> loop = asyncio.get_event_loop()

# Create the task
>>> result = loop.call_soon(loop.create_task, sleeper(1))

# Make sure the loop stops after 2 seconds
>>> result = loop.call_later(2, loop.stop)

# Start the loop and make it run forever. Or at least until the loop.stop gets
# called in 2 seconds.
>>> loop.run_forever()
Finished sleeper with delay: 1

```

现在，稍微了解一下调试异步函数。调试异步函数曾经非常困难，甚至是不可能的，因为没有好的方法来查看函数在哪里以及如何停滞。幸运的是，情况已经改变。在`Task`类的情况下，只需调用`task.get_stack`或`task.print_stack`就可以看到它当前所在的位置。使用方法可以简单到如下：

```py
>>> import asyncio

>>> async def stack_printer():
...     for task in asyncio.Task.all_tasks():
...         task.print_stack()

# Create an event loop
>>> loop = asyncio.get_event_loop()

# Create the task
>>> result = loop.run_until_complete(stack_printer())

```

### 事件循环

事件循环的概念实际上是`asyncio`中最重要的一个。你可能已经怀疑协程本身就是一切的关键，但没有事件循环，它们就毫无用处。事件循环就像任务切换器一样工作，就像操作系统在 CPU 上切换活动任务的方式一样。即使有多核处理器，仍然需要一个主进程告诉 CPU 哪些任务需要运行，哪些需要等待/休眠一段时间。这正是事件循环所做的：它决定要运行哪个任务。

#### 事件循环实现

到目前为止，我们只看到了`asyncio.get_event_loop`，它返回默认的事件循环和默认的事件循环策略。目前，有两种捆绑的事件循环实现：`async.SelectorEventLoop`和`async.ProactorEventLoop`实现。哪一种可用取决于您的操作系统。后一种事件循环仅在 Windows 机器上可用，并使用 I/O 完成端口，这是一个据说比`asyncio.SelectorEventLoop`的`Select`实现更快更高效的系统。如果性能是一个问题，这是需要考虑的事情。幸运的是，使用起来相当简单：

```py
import asyncio

loop = asyncio.ProActorEventLoop()
asyncio.set_event_loop(loop)
```

备用事件循环基于选择器，自 Python 3.4 以来，可以通过核心 Python 安装中的`selectors`模块获得。`selectors`模块是在 Python 3.4 中引入的，以便轻松访问低级异步 I/O 操作。基本上，它允许您通过使用 I/O 多路复用来打开和读取许多文件。由于`asyncio`为您处理了所有复杂性，通常不需要直接使用该模块，但如果需要，使用起来相当简单。以下是将函数绑定到标准输入的读事件（`EVENT_READ`）的示例。代码将简单地等待，直到其中一个注册的文件提供新数据：

```py
import sys
import selectors

def read(fh):
    print('Got input from stdin: %r' % fh.readline())

if __name__ == '__main__':
    # Create the default selector
    selector = selectors.DefaultSelector()

    # Register the read function for the READ event on stdin
    selector.register(sys.stdin, selectors.EVENT_READ, read)

    while True:
        for key, mask in selector.select():
            # The data attribute contains the read function here
            callback = key.data
            # Call it with the fileobj (stdin here)
            callback(key.fileobj)
```

有几种选择器可用，例如传统的`selectors.SelectSelector`（内部使用`select.select`），但也有更现代的解决方案，如`selectors.KqueueSelector`、`selectors.EpollSelector`和`selectors.DevpollSelector`。尽管默认情况下应该选择最有效的选择器，但在某些情况下，最有效的选择器可能不适合。在这些情况下，选择器事件循环允许您指定不同的选择器：

```py
import asyncio
import selectors

selector = selectors.SelectSelector()
loop = asyncio.SelectorEventLoop(selector)
asyncio.set_event_loop(loop)
```

应该注意的是，这些选择器之间的差异在大多数实际应用程序中通常太小而难以注意到。我遇到的唯一一种情况是在构建一个必须处理大量同时连接的服务器时，这种优化才会有所不同。当我说“大量”时，我指的是在单个服务器上有超过 100,000 个并发连接的问题，这只有少数人在这个星球上需要处理。

#### 事件循环策略

事件循环策略是创建和存储实际事件循环的对象。它们被设计为最大灵活性，但通常不需要修改。我能想到的唯一原因修改事件循环策略是如果您想要使特定事件循环在特定处理器和/或系统上运行，或者如果您希望更改默认事件循环类型。除此之外，它提供的灵活性超出了大多数人所需的范围。通过以下代码，使自己的事件循环（在这种情况下是`ProActorEventLoop`）成为默认事件循环是完全可能的：

```py
import asyncio

class ProActorEventLoopPolicy(
        asyncio.events.BaseDefaultEventLoopPolicy):
    _loop_factory = asyncio.SelectorEventLoop

policy = ProActorEventLoopPolicy()
asyncio.set_event_loop_policy(policy)
```

#### 事件循环使用

到目前为止，我们只看到了`loop.run_until_complete`方法。当然，还有其他一些方法。你最有可能经常使用的是`loop.run_forever`。这个方法，正如你所期望的那样，会一直运行下去，或者至少直到`loop.stop`被运行。

所以，假设我们现在有一个永远运行的事件循环，我们需要向其中添加任务。这就是事情变得有趣的地方。在默认事件循环中有很多选择：

+   `call_soon`：将项目添加到（FIFO）队列的末尾，以便按照插入的顺序执行函数。

+   `call_soon_threadsafe`：这与`call_soon`相同，只是它是线程安全的。`call_soon`方法不是线程安全的，因为线程安全需要使用全局解释器锁（GIL），这在线程安全时会使您的程序变成单线程。性能章节将更彻底地解释这一点。

+   `call_later`：在给定的秒数后调用函数。如果两个任务将同时运行，它们将以未定义的顺序运行。请注意，延迟是最小值。如果事件循环被锁定/忙碌，它可能会稍后运行。

+   `call_at`：在与`loop.time`的输出相关的特定时间调用函数。`loop.time`之后的每个整数都会增加一秒。

所有这些函数都返回`asyncio.Handle`对象。只要任务尚未执行，这些对象就允许通过`handle.cancel`函数取消任务。但是要小心取消来自其他线程，因为取消也不是线程安全的。要以线程安全的方式执行它，我们还必须将取消函数作为任务执行：`loop.call_soon_threadsafe(handle.cancel)`。以下是一个示例用法：

```py
>>> import time
>>> import asyncio

>>> t = time.time()

>>> def printer(name):
...     print('Started %s at %.1f' % (name, time.time() - t))
...     time.sleep(0.2)
...     print('Finished %s at %.1f' % (name, time.time() - t))

>>> loop = asyncio.get_event_loop()
>>> result = loop.call_at(loop.time() + .2, printer, 'call_at')
>>> result = loop.call_later(.1, printer, 'call_later')
>>> result = loop.call_soon(printer, 'call_soon')
>>> result = loop.call_soon_threadsafe(printer, 'call_soon_threadsafe')

>>> # Make sure we stop after a second
>>> result = loop.call_later(1, loop.stop)

>>> loop.run_forever()
Started call_soon at 0.0
Finished call_soon at 0.2
Started call_soon_threadsafe at 0.2
Finished call_soon_threadsafe at 0.4
Started call_later at 0.4
Finished call_later at 0.6
Started call_at at 0.6
Finished call_at at 0.8

```

你可能会想知道为什么我们在这里没有使用协程装饰器。原因是循环不允许直接运行协程。要通过这些调用函数运行协程，我们需要确保它被包装在`asyncio.Task`中。正如我们在前一段中看到的那样，这很容易——幸运的是：

```py
>>> import time
>>> import asyncio

>>> t = time.time()

>>> async def printer(name):
...     print('Started %s at %.1f' % (name, time.time() - t))
...     await asyncio.sleep(0.2)
...     print('Finished %s at %.1f' % (name, time.time() - t))

>>> loop = asyncio.get_event_loop()

>>> result = loop.call_at(
...     loop.time() + .2, loop.create_task, printer('call_at'))
>>> result = loop.call_later(.1, loop.create_task,
...     printer('call_later'))
>>> result = loop.call_soon(loop.create_task,
...     printer('call_soon'))

>>> result = loop.call_soon_threadsafe(
...     loop.create_task, printer('call_soon_threadsafe'))

>>> # Make sure we stop after a second
>>> result = loop.call_later(1, loop.stop)

>>> loop.run_forever()
Started call_soon at 0.0
Started call_soon_threadsafe at 0.0
Started call_later at 0.1
Started call_at at 0.2
Finished call_soon at 0.2
Finished call_soon_threadsafe at 0.2
Finished call_later at 0.3
Finished call_at at 0.4

```

这些调用方法可能看起来略有不同，但内部实际上都归结为通过`heapq`实现的两个队列。`loop._scheduled`用于计划操作，`loop._ready`用于立即执行。当调用`_run_once`方法（`run_forever`方法在`while True`循环中包装了这个方法）时，循环将首先尝试使用特定的循环实现（例如`SelectorEventLoop`）处理`loop._ready`堆中的所有项目。一旦`loop._ready`中的所有项目都被处理，循环将继续将`loop._scheduled`堆中的项目移动到`loop._ready`堆中，如果它们已经到期。

`call_soon`和`call_soon_threadsafe`都写入`loop._ready`堆。而`call_later`方法只是`call_at`的一个包装，其计划时间是当前值加上`asyncio.time`，它写入`loop._scheduled`堆。

这种处理方法的结果是，通过`call_soon*`方法添加的所有内容都将始终在通过`call_at`/`call_later`方法添加的所有内容之后执行。

至于`ensure_futures`函数，它将在内部调用`loop.create_task`来将协程包装在`Task`对象中，当然，这是`Future`对象的子类。如果出于某种原因需要扩展`Task`类，可以通过`loop.set_task_factory`方法轻松实现。

根据事件循环的类型，实际上有许多其他方法可以创建连接、文件处理程序等。这些将在后面的段落中通过示例进行解释，因为它们与事件循环的关系较小，更多地涉及使用协程进行编程。

### 进程

到目前为止，我们只是执行了特定的异步 Python 函数，但有些事情在 Python 中异步运行起来会更困难。例如，假设我们有一个长时间运行的外部应用程序需要运行。`subprocess`模块将是运行外部应用程序的标准方法，并且它运行得相当好。通过一些小心，甚至可以确保它们不会通过轮询输出来阻塞主线程。然而，这仍然需要轮询。然而，事件会更好，这样我们在等待结果时可以做其他事情。幸运的是，这很容易通过`asyncio.Process`安排。与`Future`和`Task`类似，这个类是通过事件循环创建的。在使用方面，这个类与`subprocess.Popen`类非常相似，只是函数已经变成了异步的。当然，这会导致轮询函数的消失。

首先，让我们看传统的顺序版本：

```py
>>> import time
>>> import subprocess
>>>
>>>
>>> t = time.time()
>>>
>>>
>>> def process_sleeper():
...     print('Started sleep at %.1f' % (time.time() - t))
...     process = subprocess.Popen(['sleep', '0.1'])
...     process.wait()
...     print('Finished sleep at %.1f' % (time.time() - t))
...
>>>
>>> for i in range(3):
...     process_sleeper()
Started sleep at 0.0
Finished sleep at 0.1
Started sleep at 0.1
Finished sleep at 0.2
Started sleep at 0.2
Finished sleep at 0.3

```

由于一切都是按顺序执行的，所以等待的时间是休眠命令休眠的 0.1 秒的三倍。因此，与其同时等待所有这些，这次让我们并行运行它们：

```py
>>> import time
>>> import subprocess 

>>> t = time.time()

>>> def process_sleeper():
...     print('Started sleep at %.1f' % (time.time() - t))
...     return subprocess.Popen(['sleep', '0.1'])
...
>>>
>>> processes = []
>>> for i in range(5):
...     processes.append(process_sleeper())
Started sleep at 0.0
Started sleep at 0.0
Started sleep at 0.0
Started sleep at 0.0
Started sleep at 0.0

>>> for process in processes:
...     returncode = process.wait()
...     print('Finished sleep at %.1f' % (time.time() - t))
Finished sleep at 0.1
Finished sleep at 0.1
Finished sleep at 0.1
Finished sleep at 0.1
Finished sleep at 0.1

```

虽然从运行时间上看这样做要好得多，但我们的程序结构现在有点混乱。我们需要两个循环，一个用于启动进程，另一个用于测量完成时间。此外，我们还必须将打印语句移到函数外部，这通常也是不可取的。这次，我们将尝试`asyncio`版本：

```py
>>> import time
>>> import asyncio

>>> t = time.time()

>>> async def async_process_sleeper():
...     print('Started sleep at %.1f' % (time.time() - t))
...     process = await asyncio.create_subprocess_exec('sleep', '0.1')
...     await process.wait()
...     print('Finished sleep at %.1f' % (time.time() - t))

>>> loop = asyncio.get_event_loop()
>>> for i in range(5):
...     task = loop.create_task(async_process_sleeper())

>>> future = loop.call_later(.5, loop.stop)

>>> loop.run_forever()
Started sleep at 0.0
Started sleep at 0.0
Started sleep at 0.0
Started sleep at 0.0
Started sleep at 0.0
Finished sleep at 0.1
Finished sleep at 0.1
Finished sleep at 0.1
Finished sleep at 0.1
Finished sleep at 0.1

```

如您所见，这种方式很容易同时运行多个应用程序。但这只是简单的部分；处理进程的难点在于交互式输入和输出。`asyncio`模块有几种措施可以使其更容易，但在实际处理结果时仍然可能会有困难。以下是调用 Python 解释器、执行一些代码并再次退出的示例：

```py
import asyncio

async def run_script():
    process = await asyncio.create_subprocess_shell(
        'python3',
        stdout=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
    )

    # Write a simple Python script to the interpreter
    process.stdin.write(b'\n'.join((
        b'import math',
        b'x = 2 ** 8',
        b'y = math.sqrt(x)',
        b'z = math.sqrt(y)',
        b'print("x: %d" % x)',
        b'print("y: %d" % y)',
        b'print("z: %d" % z)',
        b'for i in range(int(z)):',
        b'    print("i: %d" % i)',
    )))
    # Make sure the stdin is flushed asynchronously
    await process.stdin.drain()
    # And send the end of file so the Python interpreter will
    # start processing the input. Without this the process will
    # stall forever.
    process.stdin.write_eof()

    # Fetch the lines from the stdout asynchronously
    async for out in process.stdout:
        # Decode the output from bytes and strip the whitespace
        # (newline) at the right
        print(out.decode('utf-8').rstrip())

    # Wait for the process to exit
    await process.wait()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_script())
    loop.close()
```

代码足够简单，但这段代码中有一些对我们来说不明显但却需要的部分。虽然创建子进程和编写代码是相当明显的，但您可能会对`process.stdin.write_eof()`这一行感到疑惑。问题在于缓冲。为了提高性能，大多数程序默认会对输入和输出进行缓冲。在 Python 程序的情况下，结果是除非我们发送**文件结束**（**eof**），否则程序将继续等待更多的输入。另一种选择是关闭`stdin`流或以某种方式与 Python 程序通信，告诉它我们不会再发送任何输入。然而，这当然是需要考虑的事情。另一个选择是使用`yield` from `process.stdin.drain()`，但那只处理了代码的发送方；接收方可能仍在等待更多的输入。不过，让我们看一下输出：

```py
# python3 processes.py
x: 256
y: 16
z: 4
i: 0
i: 1
i: 2
i: 3

```

使用这种实现方式，我们仍然需要一个循环来从`stdout`流中获取所有的结果。不幸的是，`asyncio.StreamReader`（`process.stdout`所属的类）类尚不支持`async for`语法。如果支持的话，一个简单的`async for out in process.stdout`就可以工作了。一个简单的`yield from process.stdout.read()`也可以工作，但通常逐行阅读更方便使用。

如果可能的话，我建议您避免使用`stdin`向子进程发送数据，而是使用一些网络、管道或文件通信。正如我们将在下面的段落中看到的，这些更方便处理。

## 异步服务器和客户端

导致脚本和应用程序停滞的最常见原因之一是使用远程资源。使用`asyncio`，至少其中的大部分是很容易解决的。获取多个远程资源并为多个客户端提供服务比以前要容易得多，也更轻量级。虽然多线程和多进程也可以用于这些情况，但`asyncio`是一个更轻量级的替代方案，实际上更容易管理。创建客户端和服务器有两种主要方法。协程方式是使用`asyncio.open_connection`和`asyncio.start_server`。基于类的方法要求您继承`asyncio.Protocol`类。虽然它们本质上是相同的，但工作方式略有不同。

### 基本回显服务器

基本的客户端和服务器版本编写起来相当简单。`asyncio`模块负责所有底层连接处理，我们只需要连接正确的方法。对于服务器，我们需要一个处理传入连接的方法，对于客户端，我们需要一个创建连接的函数。为了说明发生了什么以及在何时发生，我们将添加一个专门的打印函数，打印自服务器进程启动以来的时间和给定的参数：

```py
import time
import sys
import asyncio

HOST = '127.0.0.1'
PORT = 1234

start_time = time.time()

def printer(start_time, *args, **kwargs):
    '''Simple function to print a message prefixed with the
    time relative to the given start_time'''
    print('%.1f' % (time.time() - start_time), *args, **kwargs)

async def handle_connection(reader, writer):
    client_address = writer.get_extra_info('peername')
    printer(start_time, 'Client connected', client_address)

    # Send over the server start time to get consistent
    # timestamps
    writer.write(b'%.2f\n' % start_time)
    await writer.drain()

    repetitions = int((await reader.readline()))
    printer(start_time, 'Started sending to', client_address)

    for i in range(repetitions):
        message = 'client: %r, %d\n' % (client_address, i)
        printer(start_time, message, end='')
        writer.write(message.encode())
        await writer.drain()

    printer(start_time, 'Finished sending to', client_address)
    writer.close()

async def create_connection(repetitions):
    reader, writer = await asyncio.open_connection(
        host=HOST, port=PORT)

    start_time = float((await reader.readline()))

    writer.write(repetitions.encode() + b'\n')
    await writer.drain()

    async for line in reader:
        # Sleeping a little to emulate processing time and make
        # it easier to add more simultaneous clients
        await asyncio.sleep(1)

        printer(start_time, 'Got line: ', line.decode(),
                end='')

    writer.close()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()

    if sys.argv[1] == 'server':
        server = asyncio.start_server(
            handle_connection,
            host=HOST,
            port=PORT,
        )
        running_server = loop.run_until_complete(server)

        try:
            result = loop.call_later(5, loop.stop)
            loop.run_forever()
        except KeyboardInterrupt:
            pass

        running_server.close()
        loop.run_until_complete(running_server.wait_closed())
    elif sys.argv[1] == 'client':
        loop.run_until_complete(create_connection(sys.argv[2]))

    loop.close()
```

现在我们将运行服务器和两个同时的客户端。由于这些是并行运行的，服务器输出当然有点奇怪。因此，我们将从服务器到客户端同步启动时间，并在所有打印语句前加上自服务器启动以来的秒数。

服务器：

```py
# python3 simple_connections.py server
0.4 Client connected ('127.0.0.1', 59990)
0.4 Started sending to ('127.0.0.1', 59990)
0.4 client: ('127.0.0.1', 59990), 0
0.4 client: ('127.0.0.1', 59990), 1
0.4 client: ('127.0.0.1', 59990), 2
0.4 Finished sending to ('127.0.0.1', 59990)
2.0 Client connected ('127.0.0.1', 59991)
2.0 Started sending to ('127.0.0.1', 59991)
2.0 client: ('127.0.0.1', 59991), 0
2.0 client: ('127.0.0.1', 59991), 1
2.0 Finished sending to ('127.0.0.1', 59991)

```

第一个客户端：

```py
# python3 simple_connections.py client 3
1.4 Got line:  client: ('127.0.0.1', 59990), 0
2.4 Got line:  client: ('127.0.0.1', 59990), 1
3.4 Got line:  client: ('127.0.0.1', 59990), 2

```

第二个客户端：

```py
# python3 simple_connections.py client 2
3.0 Got line:  client: ('127.0.0.1', 59991), 0
4.0 Got line:  client: ('127.0.0.1', 59991), 1

```

由于输入和输出都有缓冲区，我们需要在写入后手动排空输入，并在从对方读取输出时使用`yield from`。这正是与常规外部进程通信更困难的原因。进程的标准输入更侧重于用户输入而不是计算机输入，这使得使用起来不太方便。

### 注意

如果您希望使用`reader.read(BUFFER)`而不是`reader.readline()`，也是可能的。只是请注意，您需要明确地分隔数据，否则可能会意外地被附加。所有写操作都写入同一个缓冲区，导致一个长的返回流。另一方面，尝试在`reader.readline()`中没有新行(`\n`)的情况下进行写入将导致客户端永远等待。

# 摘要

在本章中，我们看到了如何在 Python 中使用`asyncio`进行异步 I/O。对于许多场景，`asyncio`模块仍然有些原始和未完成，但不应该有任何使用上的障碍。创建一个完全功能的服务器/客户端设置仍然有点复杂，但`asyncio`最明显的用途是处理基本的网络 I/O，如数据库连接和外部资源，如网站。特别是后者只需使用`asyncio`就可以实现几行代码，从您的代码中删除一些非常重要的瓶颈。

本章的重点是理解如何告诉 Python 在后台等待结果，而不是像通常那样简单地等待或轮询结果。在第十三章中，*多处理-当单个 CPU 核心不够用*，您将了解多处理，这也是处理停滞资源的选项。然而，多处理的目标实际上是使用多个处理器，而不是处理停滞资源。当涉及潜在缓慢的外部资源时，我建议您尽可能使用`asyncio`。

在基于`asyncio`库构建实用程序时，确保搜索预制库来解决您的问题，因为其中许多目前正在开发中。在撰写本章时，Python 3.5 尚未正式发布，因此很可能很快会出现更多使用`async/await`语法的文档和库。为了确保您不重复他人已完成的工作，请在撰写扩展`asyncio`的代码之前彻底搜索互联网。

下一章将解释一个完全不同的主题-使用元类构建类。常规类是使用 type 类创建的，但现在我们将看到如何扩展和修改默认行为，使类几乎可以做任何我们想要的事情。元类甚至可以实现自动注册插件，并以非常神奇的方式向类添加功能-简而言之，如何定制不仅类实例而且类定义本身。
