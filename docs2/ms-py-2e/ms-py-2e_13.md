# 13

# asyncio – 无线程的多线程

上一章向我们展示了如何跟踪应用程序的性能。在这一章中，我们将使用异步编程在需要等待**输入/输出**（**I/O**）操作时在函数之间切换。这有效地模拟了多线程或多进程的效果，而不会引入这些解决方案带来的开销。在下一章中，我们还将涵盖多线程和多进程的情况，其中 I/O 不是瓶颈，或者`asyncio`不是一个选项。

当你处理外部资源，如读取/写入文件、与 API 或数据库交互以及其他 I/O 操作时，使用`asyncio`可以带来巨大的好处。在正常情况下，单个阻塞的远程连接可以使整个进程挂起，而使用`asyncio`，它将简单地切换到你的代码的另一个部分。

本章将解释如何在 Python 中使用异步函数，以及如何重构代码，使其仍然可以工作，即使它不遵循标准的返回值的程序性编码模式。潜在的缺点是，与使用多线程和多进程类似，代码执行可能以意外的顺序进行。

本章节涵盖以下主题：

+   `asyncio`简介

+   `asyncio`基本概念，包括协程、事件循环、未来和任务

+   使用`async def`、`async for`、`async with`和`await`的函数

+   并行执行

+   使用`asyncio`的示例，包括客户端和服务器

+   调试`asyncio`

# `asyncio`简介

`asyncio`库的创建是为了使使用异步处理更加容易和可预测。它原本是作为`asyncore`模块的替代品，该模块已经存在很长时间（自 Python 1.5 以来），但并不那么易于使用。《asyncio》库正式引入 Python 3.4，并且随着每个新版本的 Python 发布，它都经历了许多改进。

简而言之，`asyncio`库允许你在需要等待 I/O 操作时切换到执行不同的函数。因此，而不是 Python 等待操作系统为你完成文件读取，在这个过程中阻塞整个应用程序，它可以在同时执行另一个函数中的有用操作。

## 向后兼容性和 async/await 语句

在我们继续任何示例之前，了解`asyncio`在 Python 版本中的变化是很重要的。尽管`asyncio`库是在 Python 3.4 中引入的，但大部分通用语法在 Python 3.5 中已经被替换。使用旧的 Python 3.4 语法仍然是可能的，但引入了一种更简单、因此更推荐的语法，即使用`await`。

在所有示例中，除非特别说明，否则本章将假设使用 Python 3.7 或更高版本。然而，如果你仍在运行较旧版本，请查看以下部分，这些部分说明了如何在较旧的系统上运行 `asyncio`。如果你有 Python 3.7+，可以自由跳转到标题为 *并行执行的基本示例* 的部分。

### Python 3.4

对于传统的 Python 3.4 使用，需要考虑以下几点：

+   函数应该使用 `asyncio.coroutine` 装饰器声明

+   应该使用 `yield from coroutine()` 来获取异步结果

+   异步循环不支持直接使用，但可以使用 `while True: yield from coroutine()` 来模拟

示例：

```py
import asyncio

@asyncio.coroutine
def main():
    print('Hello from main')
    yield from asyncio.sleep(1)

loop = asyncio.new_event_loop()
loop.run_until_complete(main())
loop.close() 
```

### Python 3.5

Python 3.5 的语法比 Python 3.4 版本更明显。虽然考虑到协程在早期 Python 版本中的起源，`yield from` 是可以理解的，但实际上这个名字并不适合这项工作。让 `yield` 用于生成器，而 `await` 用于协程。

+   应该使用 `async def` 而不是 `def` 来声明函数

+   应该使用 `await coroutine()` 来获取异步结果

+   可以使用 `async for ... in ...` 创建异步循环

示例：

```py
import asyncio

async def main():
    print('Hello from main')
    await asyncio.sleep(1)

loop = asyncio.new_event_loop()
loop.run_until_complete(main())
loop.close() 
```

### Python 3.7

自从 Python 3.7 以来，运行 `asyncio` 代码已经变得稍微容易一些，也更明显。如果你有使用较新 Python 版本的便利，你可以使用以下方法来运行你的 `async` 函数：

```py
import asyncio

async def main():
    print('Hello from main')
    await asyncio.sleep(1)

asyncio.run(main()) 
```

对于较旧的 Python 版本，我们需要一段相当高级的代码来正确地替换 `asyncio.run()`，但如果你不关心可能重用现有的事件循环（关于事件循环的详细信息可以在本章后面找到）并且自己处理任务的关闭，你可以使用以下代码：

```py
import asyncio

async def main():
    print('Hello from main')
    await asyncio.sleep(1)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    loop.run_until_complete(main())
finally:
    # Run the loop again to finish pending tasks
    loop.run_until_complete(asyncio.sleep(0))

    asyncio.set_event_loop(None)
    loop.close() 
```

或者一个更简短的版本，虽然不一定等效，但可以处理许多测试用例：

```py
import asyncio

async def main():
    print('Hello from main')
    await asyncio.sleep(1)

loop = asyncio.get_event_loop()
loop.run_until_complete(main()) 
```

如果可能的话，我当然会推荐使用 `asyncio.run()`。即使没有 `asyncio.run()`，你也可能会遇到与较旧版本的 Python 的库兼容性问题。

然而，如果你必须这样做，你可以在 Python Git 中找到 `asyncio.run()` 的源代码，这样你可以自己实现一个简化的版本：[`github.com/python/cpython/blob/main/Lib/asyncio/runners.py`](https://github.com/python/cpython/blob/main/Lib/asyncio/runners.py)。

## 并行执行的基本示例

当涉及到代码性能时，你通常会遇到以下两种瓶颈之一：

+   等待外部 I/O，如网络服务器、文件系统、数据库服务器、任何网络相关的内容以及其他

+   在进行大量计算的情况下，CPU

如果你的 CPU 由于大量计算而成为瓶颈，你需要求助于使用更快的算法、更快的处理器或将计算卸载到专用硬件（如显卡）。在这些情况下，`asyncio` 库对你帮助不大。

如果你的代码大部分时间都在等待用户、内核、文件系统或外部服务器，`asyncio`可以在很大程度上帮助你，同时它是一个相对简单且副作用较少的解决方案。正如我们将在*asyncio 概念*部分看到的那样，然而也有一些注意事项。使现有代码与`asyncio`兼容可能是一项大量工作。

让我们从一个非常简单的例子开始，以展示在需要等待时常规代码和`asyncio`代码之间的区别。

首先，执行两次 1 秒`sleep`的常规 Python 版本：

```py
>>> import time
>>> import asyncio

>>> def normal_sleep():
...     print('before sleep')
...     time.sleep(1)
...     print('after sleep')

>>> def normal_sleeps(n):
...     for _ in range(n):
...         normal_sleep()

# Normal execution
>>> start = time.time()
>>> normal_sleeps(2)
before sleep
after sleep
before sleep
after sleep
>>> print(f'duration: {time.time() - start:.0f}')
duration: 2 
```

现在让我们看看执行两次 1 秒`sleep`的`asyncio`版本：

```py
>>> async def asyncio_sleep():
...     print('before sleep')
...     await asyncio.sleep(1)
...     print('after sleep')

>>> async def asyncio_sleeps(n):
...     coroutines = []
...     for _ in range(n):
...         coroutines.append(asyncio_sleep())
...
...     await asyncio.gather(*coroutines)

>>> start = time.time()
>>> asyncio.run(asyncio_sleeps(2))
before sleep
before sleep
after sleep
after sleep
>>> print(f'duration: {time.time() - start:.0f}')
duration: 1 
```

如你所见，它仍然需要等待 1 秒钟来实际`sleep`，但它可以并行运行它们。`asyncio_sleep()`函数是同时开始的，正如`sleep 前`输出所示。

让我们分析这个例子中使用的组件：

+   `async def`: 这告诉 Python 解释器我们的函数是一个协程函数而不是常规函数。

+   `asyncio.sleep()`: 这是`time.sleep()`的异步版本。这两个函数之间最大的区别在于，`time.sleep()`在睡眠时会保持 Python 进程忙碌，而`asyncio.sleep()`则允许在事件循环中切换到不同的任务。这个过程与大多数操作系统中任务切换的工作原理非常相似。

+   `asyncio.run()`: 这是一个包装器，用于在默认事件循环中执行协程。这实际上是`asyncio`任务切换器；更多关于这一点将在下一节中介绍。

+   `asyncio.gather()`: 包装一系列可等待对象，并为你收集结果。等待时间可配置，等待的方式也可配置。你可以选择等待直到第一个结果出现，直到所有结果都可用，或者直到第一个异常发生。

这立即展示了`asyncio`代码的一些注意事项和陷阱。

如果我们不小心使用了`time.sleep()`而不是`asyncio.sleep()`，代码将需要 2 秒钟才能运行，并且在执行过程中会阻塞整个循环。更多关于这一点的内容将在下一节中介绍。

如果我们在最后没有使用`await asyncio.gather()`而是使用`await asyncio.sleep()`，代码将按顺序运行，而不是并行运行，正如你可能期望的那样。

现在我们已经看到了`asyncio`的一个基本示例，我们需要了解更多关于其内部结构，以便更明显地看到局限性。

## asyncio 概念

在进一步探讨示例和用法之前，`asyncio`库有几个基本概念需要解释。上一节中显示的示例实际上使用了其中的一些，但关于如何和为什么的解释可能仍然有用。

`asyncio`的主要概念是协程和事件循环。在这些概念中，有几种辅助类可用，例如`Streams`、`Futures`和`Processes`。接下来的几段将解释它们的基本知识，以便我们可以在后面的章节中理解实现示例。

### 协程、未来和任务

`coroutine`、`asyncio.Future`和`asyncio.Task`对象本质上是对结果的承诺；如果它们可用，则返回结果，并且如果它们尚未完成处理，则可以用来取消承诺的执行。应该注意的是，这些对象的创建并不能保证代码将被执行。实际的执行开始发生在你`await`结果或告诉事件循环执行承诺的时候。这将在下一节关于事件循环的讨论中介绍。

当使用`asyncio`时，你将遇到的最基本对象是`coroutine`。任何常规`async def`（如`asyncio.sleep()`）的结果都是一个`coroutine`对象。一旦你`await`那个`coroutine`，它将被执行，你将得到结果。

`asyncio.Future`和`asyncio.Task`类也可以通过`await`执行，但还允许你注册回调函数，以便在结果（或异常）可用时接收它们。此外，它们在内部维护一个状态变量，允许外部方取消未来并停止（或防止）其执行。API 与`concurrent.futures.Future`类非常相似，但它们并不完全兼容，所以请确保不要混淆这两个。

为了进一步澄清，所有这些都是可等待的，但具有不同的抽象级别：

+   `coroutine`：一个尚未被等待的调用`async def`的结果。你将主要使用这些。

+   `asyncio.Future`：一个表示最终结果的类。它不需要封装`coroutine`，结果可以手动设置。

+   `asyncio.Task`：`asyncio.Future`的一个实现，旨在封装`coroutine`以提供一个方便且一致的接口。

通常，这些类的创建不是你需要直接担心的事情；而不是自己创建类，推荐的方式是通过`asyncio.create_task()`或`loop.create_task()`。前者实际上内部执行`loop.create_task()`，但如果你只想通过`asyncio.get_running_loop()`在运行的事件循环上执行它，而不需要指定它，那么它会更方便。如果你需要出于某种原因扩展`Task`类，那通过`loop.set_task_factory()`方法很容易实现。

在 Python 3.7 之前，`asyncio.create_task()`被称为`asyncio.ensure_future()`。

### 事件循环

事件循环的概念实际上是`asyncio`中最重要的一点。你可能怀疑协程本身是关于一切的，但没有事件循环它们是无用的。事件循环充当任务切换器，类似于操作系统在 CPU 上切换活动任务的方式。即使有多核处理器，仍然需要一个主进程来告诉 CPU 哪些任务要运行，哪些需要等待或稍作休眠。这正是事件循环所做的：它决定运行哪个任务。

实际上，每次你执行`await`时，事件循环都会查看挂起的 awaitables，并继续执行当前挂起的那个。这也是单个事件循环危险的地方。如果你在协程中有一个慢速/阻塞函数，例如不小心使用`time.sleep()`而不是`asyncio.sleep()`，它将阻塞整个事件循环，直到它完成。

实际上，这意味着`await asyncio.sleep(5)`只能保证你的代码将等待至少 5 秒。如果在那个`await`期间，其他协程阻塞了事件循环 10 秒，那么`asyncio.sleep(5)`将至少需要 10 秒。

#### 事件循环实现

到目前为止，我们只看到了`asyncio.run()`，它内部使用`asyncio.get_event_loop()`来返回具有默认事件循环策略的默认事件循环。目前有两个捆绑的事件循环实现：

+   默认情况下在 Unix 和 Linux 系统上使用的`asyncio.SelectorEventLoop`实现。

+   仅在 Windows 上支持（且为默认）的`asyncio.ProactorEventLoop`实现。

内部，`asyncio.ProactorEventLoop`实现使用 I/O 完成端口，这是一个据说比 Windows 系统上`asyncio.SelectorEventLoop`的`select`实现更快更高效的系统。

`asyncio.SelectorEventLoop`是基于选择器的，自 Python 3.4 以来，通过核心 Python 模块中的`select`模块提供。有几种选择器可用：传统的`selectors.SelectSelector`，它内部使用`select.select`，但也包括更现代的解决方案，如`selectors.KqueueSelector`、`selectors.EpollSelector`和`selectors.DevpollSelector`。尽管`asyncio.SelectorEventLoop`默认会选择最有效的选择器，但在某些情况下，最有效的选择器可能以某种方式不适用。

最有效的选择器是通过排除法选择的。如果`select`模块具有`kqueue`属性，则将使用`KqueueSelector`。如果`kqueue`不可用，则将按照以下顺序选择下一个最佳选项：

1.  `KqueueSelector`：`kqueue`是 BSD 系统的事件通知接口。目前支持 FreeBSD、NetBSD、OpenBSD、DragonFly BSD 和 macOS（OS X）。

1.  `EpollSelector`：`epoll`是`kqueue`的 Linux 内核版本。

1.  `DevpollSelector`：此选择器使用`/dev/poll`，这是一个类似于`kqueue`和`epoll`的系统，但支持在 Solaris 系统上。

1.  `PollSelector`：`poll()`是一个系统调用，当有更新可用时将调用你的函数。实际实现取决于系统。

1.  `SelectSelector`：与`poll()`非常相似，但`select()`为所有文件描述符构建一个位图，并在每次更新时遍历该列表，这比`poll()`低效得多。

在这些情况下，选择器事件循环允许你指定不同的选择器：

```py
>>> import asyncio
>>> import selectors

>>> selector = selectors.SelectSelector()
>>> loop = asyncio.SelectorEventLoop(selector)
>>> asyncio.set_event_loop(loop) 
```

应该注意的是，这些之间的差异通常太小，在大多数实际应用中几乎察觉不到。这就是为什么我会建议尽可能忽略这些优化，因为它们可能效果甚微，如果使用不当甚至可能引起问题。我唯一遇到这些差异真正重要的情况是构建一个需要处理大量并发连接的服务器。这里的“大量”指的是单个服务器上超过 100,000 个并发连接，这是地球上只有少数人需要解决的问题。

如果你认为性能很重要（并且你正在运行 Linux/OS X），我建议查看 `uvloop`，这是一个基于 `libuv` 的非常快速的事件循环，`libuv` 是一个用 C 编写的异步 I/O 库，支持大多数平台。根据 `uvloop` 的基准测试，它可以让你的事件循环速度提高 2-4 倍。

#### 事件循环策略

事件循环策略仅仅是为你存储和创建事件循环的构造，并且考虑到最大程度的灵活性。我能想到修改事件循环策略的唯一原因可能是因为你想让特定的事件循环在特定的处理器和/或系统上运行，例如，如果你正在运行 Linux 或 OS X，则仅启用 `uvloop`。除此之外，它提供的灵活性比大多数人需要的都要多。如果你想将 `uvloop` 设置为默认循环（如果已安装），可以执行以下操作：

```py
import asyncio

class UvLoopPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        try:
            from uvloop import Loop
            return Loop()
        except ImportError:
            return super().new_event_loop()

asyncio.set_event_loop_policy(UvLoopPolicy()) 
```

除了覆盖 `new_event_loop()` 来自定义新事件循环的创建之外，你还可以通过覆盖 `get_event_loop()` 和 `set_event_loop()` 方法来覆盖事件循环的重用方式。我个人从未在启用 `uvloop` 之外使用过它。

#### 事件循环的使用

现在我们已经知道了事件循环是什么，它们的作用以及如何选择事件循环，让我们看看它们如何在 `asyncio.run()` 之外的应用。

如果你开始运行自己的事件循环，你可能会使用 `loop.run_forever()`，正如你所期望的，它会永远运行。或者至少直到运行了 `loop.stop()`。但你也可以使用 `loop.run_until_complete()` 运行单个任务。后者对于一次性操作非常有用，但在某些场景中可能会引起错误。如果你从一个非常小/快速的协程创建任务，那么任务可能没有时间运行，因此它将不会在下次执行 `loop.run_until_complete()` 或 `loop.run_forever()` 时执行。关于这一点，我们将在本章后面详细讨论；现在，我们将假设使用 `loop.run_forever()` 的长时间运行循环。

由于我们现在有一个永远运行的事件循环，我们需要向其中添加任务——这就是事情变得有趣的地方。默认事件循环中有许多可用的选择：

+   `call_soon()`：将一个项目添加到（FIFO）队列的末尾，这样函数将按照它们插入的顺序执行。

+   `call_soon_threadsafe()`: 与 `call_soon()` 相同，但它是线程安全的。`call_soon()` 方法不是线程安全的，因为线程安全需要使用 **全局解释器锁**（**GIL**），这实际上使得程序在线程安全时变为单线程。*第十四章，多进程——当单个 CPU 核心不够用时* 详细解释了 GIL 和线程安全。

+   `call_later()`: 在给定秒数后调用函数；如果两个任务会在同一时间运行，它们将以未定义的顺序执行。如果未定义的顺序是一个问题，你也可以选择使用 `asyncio.gather()` 或稍微增加两个任务中的一个的 `delay` 参数。请注意，`delay` 是一个最小值——如果事件循环被锁定/忙碌，它可能会稍后运行。

+   `call_at()`: 在与 `loop.time()` 输出相关的特定时间调用函数，`loop.time()` 是 `loop` 开始运行以来的秒数。所以，如果 `loop.time()` 的当前值为 `90`（这意味着 `loop` 从开始运行已经过去了 `90` 秒），那么你可以运行 `loop.call_at(95, ...)` 来在 `5` 秒后执行。

所有这些函数都返回 `asyncio.Handle` 对象。这些对象允许通过 `handle.cancel()` 函数取消尚未执行的任务。但是，请注意，从其他线程取消时，取消操作也不是线程安全的。为了以线程安全的方式执行它，我们必须也将取消函数作为一个任务来执行：`loop.call_soon_threadsafe(handle.cancel)`。

示例用法：

```py
>>> import time
>>> import asyncio

>>> def printer(name):
...     print(f'Started {name} at {loop.time() - offset:.1f}')
...     time.sleep(0.2)
...     print(f'Finished {name} at {loop.time() - offset:.1f}')

>>> loop = asyncio.new_event_loop()
>>> _ = loop.call_at(loop.time() + .2, printer, 'call_at')
>>> _ = loop.call_later(.1, printer, 'call_later')
>>> _ = loop.call_soon(printer, 'call_soon')
>>> _ = loop.call_soon_threadsafe(printer, 'call_soon_threadsafe')

>>> # Make sure we stop after a second
>>> _ = loop.call_later(1, loop.stop)

# Store the offset because the loop requires time to start
>>> offset = loop.time()

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

你可能想知道为什么我们在这里使用 `time.sleep()` 而不是 `asyncio.sleep()`。这是一个有意的选择，以展示这些函数中的任何一个都没有提供任何关于函数执行时间的保证，如果 `loop` 以某种方式被阻塞。尽管我们为 `loop.call_later()` 调用指定了 `0.1` 秒的延迟，但实际上它花了 `0.4` 秒才开始执行。如果我们使用 `asyncio.sleep()`，函数将并行运行。

`call_soon()`, `call_soon_threadsafe()` 和 `call_later()` 函数都是 `call_at()` 函数的包装器。在 `call_soon()` 的情况下，它只是将 `call_later()` 延迟设置为 `0` 进行包装，而 `call_at()` 则是添加了 `asyncio.time()` 延迟的 `call_soon()`。

根据事件循环的类型，实际上还有许多创建连接、文件处理器等其他方法，类似于 `asyncio.create_task()`。这些方法将在后面的章节中通过示例进行解释，因为它们与事件循环的关系较少，更多的是关于使用协程进行编程。

### 执行器

即使是简单的 `time.sleep()` 也可能完全阻塞你的事件循环，你可能想知道 `asyncio` 的实际用途是什么。这意味着你可能需要重写整个代码库以使其与 `asyncio` 兼容，对吧？理想情况下，这将是最好的解决方案，但我们可以通过使用执行器从 `asyncio` 代码中执行同步代码来绕过这个限制。`Executor` 创建了之前提到的另一种类型的 `Future`（`concurrent.futures.Future` 与 `asyncio.Future` 相比），并在单独的线程或进程中运行你的代码，以提供对同步代码的 `asyncio` 接口。

这是一个基本的例子，展示了通过执行器执行的同步 `time.sleep()` 以使其异步：

```py
>>> import time
>>> import asyncio

>>> def executor_sleep():
...     print('before sleep')
...     time.sleep(1)
...     print('after sleep')

>>> async def executor_sleeps(n):
...     loop = asyncio.get_running_loop()
...     futures = []
...     for _ in range(n):
...         future = loop.run_in_executor(None, executor_sleep)
...         futures.append(future)
...
...     await asyncio.gather(*futures)

>>> start = time.time()
>>> asyncio.run(executor_sleeps(2))
before sleep
before sleep
after sleep
after sleep
>>> print(f'duration: {time.time() - start:.0f}')
duration: 1 
```

因此，我们不是直接运行 `executor_sleep()`，而是通过 `loop.run_in_executor()` 创建一个未来。这使得 `asyncio` 通过默认执行器执行这个函数，这通常是 `concurrent.futures.ThreadPoolExecutor`，并在完成后返回结果。你需要意识到线程安全性，因为它是在单独的线程中处理的，但关于这个话题的更多内容将在下一章中介绍。

对于阻塞但不是 CPU 密集型操作（换句话说，没有重计算），默认基于线程的执行器将工作得很好。对于 CPU 密集型操作，它不会帮助你，因为操作仍然限制在单个 CPU 核心上。对于这些场景，我们可以使用 `concurrent.futures.ProcessPoolExecutor()`：

```py
import time
import asyncio
import concurrent.futures

def executor_sleep():
    print('before sleep')
    time.sleep(1)
    print('after sleep')

async def executor_sleeps(n):
    loop = asyncio.get_running_loop()
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as pool:
        for _ in range(n):
            future = loop.run_in_executor(pool, executor_sleep)
            futures.append(future)

        await asyncio.gather(*futures)

if __name__ == '__main__':
    start = time.time()
    asyncio.run(executor_sleeps(2))
    print(f'duration: {time.time() - start:.0f}') 
```

虽然这个例子看起来几乎与上一个例子相同，但内部机制相当不同，使用多个 Python 进程而不是多个线程带来了几个注意事项：

+   进程之间很难共享内存。这意味着你想传递作为参数的任何东西以及你需要返回的任何东西都必须由 `pickle` 进程支持，这样 Python 才能通过网络发送数据到另一个 Python 进程。这将在第十四章中详细解释。

+   主脚本必须从 `if __name__ == '__main__'` 块中运行，否则执行器最终会陷入无限循环，不断繁殖自己。

+   大多数资源不能在进程之间共享。这类似于无法共享内存，但不仅如此。如果你在主进程中有一个数据库连接，那么这个连接不能从该进程使用，因此它需要自己的连接。

+   终止/退出进程可能更困难，因为终止主进程并不总是能保证终止子进程。

+   根据你的操作系统，每个新的进程都将使用自己的内存，这会导致内存使用量大幅增加。

+   创建新进程通常比创建新线程要重得多，所以你有很多开销。

+   进程之间的同步比线程慢得多。

所有这些原因绝对不应该阻止你使用`ProcessPoolExecutor`，但你应该始终问自己你是否真的需要它。如果你需要并行运行许多重量级计算，它可以是一个惊人的解决方案。如果可能的话，我建议使用带有`ProcessPoolExecutor`的功能性编程。第十四章，“当单个 CPU 核心不够用时——多进程”，详细介绍了多进程。

现在我们对`asyncio`有了基本的了解，是时候继续一些`asyncio`可能很有用的示例了。

# 异步示例

导致脚本和应用程序停滞不前的最常见原因之一是使用远程资源，其中“远程”意味着与网络、文件系统或其他资源的任何交互。使用`asyncio`，至少大部分问题都可以轻松解决。从多个远程资源获取数据并服务于多个客户端比以前容易得多，也轻量得多。虽然在这种情况下也可以使用多线程和多进程，但`asyncio`是一个更轻量级的替代方案，在很多情况下实际上更容易管理。

接下来的几节将展示一些使用`asyncio`实现某些操作的示例。

在你开始实现自己的代码并复制这里的示例之前，我建议你在网上快速搜索你正在寻找的库，看看是否有可用的`asyncio`版本。

通常，查找“asyncio <协议>”会给你很好的结果。或者，许多库使用`aio`前缀作为库名称，例如`aiohttp`，这也可以帮助你搜索。

## 进程

到目前为止，我们只是在 Python 中执行了简单的`async`函数，如`asyncio.sleep()`，但有些事情在异步运行时稍微困难一些。例如，假设我们有一个运行时间较长的外部应用程序，我们希望在不完全阻塞主线程的情况下运行它。

在非阻塞模式下运行外部进程的选项通常有：

+   线程

+   多进程

+   轮询（定期检查）输出

第十四章涵盖了线程和多进程。

在没有求助于更复杂的解决方案，如线程和进程多线程，这些会引入可变同步问题的情况下，我们只剩下轮询。使用轮询，我们会在一定的时间间隔内检查是否有新的输出，这可能会因为轮询间隔而减慢你的结果。也就是说，如果你的轮询间隔是 1 秒，而进程在最后一次轮询后 0.1 秒生成输出，那么接下来的 0.9 秒都是浪费在等待上。为了缓解这种情况，你可以减少轮询间隔，当然，但轮询间隔越小，检查是否有结果所浪费的时间就越多。

使用`asyncio`，我们可以拥有轮询方法的优势，而无需在轮询间隔之间浪费时间。使用`asyncio.create_subprocess_shell`和`asyncio.create_subprocess_exec`，我们可以像其他协程一样`await`输出。类的使用与`subprocess.run`非常相似，只是函数已被异步化，从而消除了轮询函数，当然。

以下示例期望您的环境中可用`sleep`命令。在所有 Unix/Linux/BSD 系统中，默认情况下都是这种情况。在 Windows 上，默认情况下不可用，但可以轻松安装。可以使用`timeout`命令作为替代。

如果您希望使用`sleep`和其他 Unix 工具，我找到的最简单方法是安装 Git for Windows，并让它安装**可选的 Unix 工具**：

![图片](img/B15882_13_01.png)

图 13.1：Git for Windows 安装程序

首先，让我们看看通过`subprocess`模块运行外部进程（在这种情况下是`sleep`命令）的传统顺序脚本版本：

```py
>>> import time
>>> import subprocess

>>> def subprocess_sleep():
...     print(f'Started sleep at: {time.time() - start:.1f}')
...     process = subprocess.Popen(['sleep', '0.1'])
...     process.wait()
...     print(f'Finished sleep at: {time.time() - start:.1f}')

>>> start = time.time() 
```

在第一次`print()`之后，我们使用`subprocess.Popen()`运行带有参数`0.1`的`sleep`命令，使其睡眠 0.1 秒。与`subprocess.run()`不同，后者会阻塞您的 Python 进程并等待外部进程运行完成，`subprocess.Popen()`创建并启动进程并返回对运行进程的引用，但它不会自动等待输出。

这允许我们显式调用`process.wait()`来等待或轮询结果，正如我们将在下一个示例中看到的那样。内部，`subprocess.run()`实际上是一个方便的快捷方式，用于`subprocess.Popen()`的常见用法。

在运行代码时，我们会得到以下输出，正如您所期望的那样：

```py
>>> for _ in range(2):
...     subprocess_sleep()
Started sleep at: 0.0
Finished sleep at: 0.1
Started sleep at: 0.1
Finished sleep at: 0.2 
```

由于所有操作都是顺序执行的，所以它需要两倍于`sleep`命令睡眠时间的 0.1 秒。这当然是最坏的情况：它完全阻塞了正在运行的 Python 进程。

而不是在运行`sleep`命令后立即等待，我们现在将以并行方式启动所有进程，并且只有在它们都在后台启动后才开始等待结果：

```py
>>> import time
>>> import subprocess

>>> def subprocess_sleep():
...     print(f'Started sleep at: {time.time() - start:.1f}')
...     return subprocess.Popen(['sleep', '0.1'])

>>> start = time.time() 
```

如您所见，我们通过返回`subprocess.Popen()`而不执行`process.wait()`来返回进程。

现在我们立即启动所有进程，并且只有在它们都启动后才开始等待输出：

```py
>>> processes = []
>>> for _ in range(2):
...     processes.append(subprocess_sleep())
Started sleep at: 0.0
Started sleep at: 0.0 
```

进程现在应该在后台运行，所以让我们等待结果：

```py
>>> for process in processes:
...     returncode = process.wait()
...     print(f'Finished sleep at: {time.time() - start:.1f}')
Finished sleep at: 0.1
Finished sleep at: 0.1 
```

虽然在运行时看起来要好得多，但在我们运行`process.wait()`时，它仍然会阻塞主进程。它还要求以这种方式重新组织结构，即拆解（`Finished`打印语句）不在与启动进程相同的块中，正如早期示例中的情况。这意味着如果您的应用程序出现错误，您需要手动跟踪哪个进程失败，这有点不方便。

在 `asyncio` 版本中，我们再次可以回到在一个函数中处理与 `sleep` 命令相关的所有事情，这与第一个例子中的 `subprocess.Popen()` 非常相似：

```py
>>> import time
>>> import asyncio

>>> async def async_process_sleep():
...     print(f'Started sleep at: {time.time() - start:.1f}')
...     process = await asyncio.create_subprocess_exec('sleep', '0.1')
...     await process.wait()
...     print(f'Finished sleep at: {time.time() - start:.1f}')

>>> async def main():
...     coroutines = []
...     for _ in range(2):
...         coroutines.append(async_process_sleep())
...     await asyncio.gather(*coroutines)

>>> start = time.time()
>>> asyncio.run(main())
Started sleep at: 0.0
Started sleep at: 0.0
Finished sleep at: 0.1
Finished sleep at: 0.1 
```

如你所见，以这种方式同时运行多个应用程序是非常简单的。语法基本上与不阻塞或轮询时的 `subprocess` 相同。

如果你从这个长时间运行的 `asyncio` 事件循环中运行，并且不需要捕获结果，你可以跳过整个 `asyncio.gather()` 步骤，而是使用 `asyncio.create_task(async_process_sleep())` 代替。

## 交互式进程

启动进程是容易的部分；更困难的部分是与进程进行交互式输入和输出。`asyncio` 模块有几种措施来简化这部分，但在实际处理结果时仍然可能很困难。

这里有一个将 Python 解释器作为外部子进程调用的例子，执行一些代码，然后以简单的一次性方式退出：

```py
>>> import time
>>> import asyncio

>>> async def run_python_script(script):
...     print(f'Executing: {script!r}')
...     process = await asyncio.create_subprocess_exec(
...         'python3',
...         stdout=asyncio.subprocess.PIPE,
...         stdin=asyncio.subprocess.PIPE,
...     )
...     stdout, stderr = await process.communicate(script)
...     print(f'stdout: {stdout!r}')

>>> asyncio.run(run_python_script(b'print(2 ** 20)'))
Executing: b'print(2 ** 20)'
stdout: b'1048576\n' 
```

在这种情况下，我们向 `stdout`（标准输出）和 `stdin`（标准输入）添加了一个管道，这样我们就可以手动从 `stdout` 读取并写入 `stdin`。一旦进程启动，我们就可以使用 `process.communicate()` 向 `stdin` 写入，如果 `stdout` 和 `stderr` 可用，`process.communicate()` 将自动读取所有输出。由于我们没有声明 `stderr` 应该是什么，Python 会自动将所有 `process.stderr` 输出发送到 `sys.stderr`，因此在这里我们可以忽略 `stderr`，因为它将是 `None`。

现在真正的挑战在于我们想要具有通过 `stdin`/`stdout`/`stderr` 进行双向通信的交互式子进程，并且可以持续运行更长时间。当然，这也是可能的，但在双方都等待输入的情况下，避免死锁可能很困难。以下是一个非常简单的 Python 子进程示例，它实际上与上面的 `communicate()` 做的事情相同，但手动进行，以便你可以对进程的输入和输出有更精细的控制：

```py
>>> import asyncio

>>> async def run_script():
...     process = await asyncio.create_subprocess_exec(
...         'python3',
...         stdout=asyncio.subprocess.PIPE,
...         stdin=asyncio.subprocess.PIPE,
...     )
... 
...     # Write a simple Python script to the interpreter
...     process.stdin.write(b'print("Hi~")')
... 
...     # Make sure the stdin is flushed asynchronously
...     await process.stdin.drain()
...     # And send the end of file so the Python interpreter will
...     # start processing the input. Without this the process will
...     # stall forever.
...     process.stdin.write_eof()
... 
...     # Fetch the lines from the stdout asynchronously
...     async for line in process.stdout:
...         # Decode the output from bytes and strip the whitespace
...         # (newline) at the right
...         print('stdout:', line.rstrip())
... 
...     # Wait for the process to exit
...     await process.wait()

>>> asyncio.run(run_script())
stdout: b'Hi~' 
```

代码可能看起来与你预期的基本相同，但有一些部分在使用时并不明显，但却是必需的。虽然子进程的创建与前面的例子相同，但写入 `stdin` 的代码略有不同。

我们现在不再使用 `process.communicate()`，而是直接写入 `process.stdin` 管道。当你运行 `process.stdin.write()` 时，Python 将会 *尝试* 向流中写入，但由于进程尚未开始运行，可能无法写入。因此，我们需要手动使用 `process.stdin.drain()` 清空这些缓冲区。一旦完成，我们发送一个文件结束（`EOF`）字符，这样 Python 子进程就知道没有更多的输入了。

一旦输入被写入，我们需要从 Python 子进程读取输出。我们可以使用`process.stdout.readline()`在循环中完成此操作，但类似于我们可以使用`for line in open(filename)`，我们也可以使用`async for`循环逐行读取`process.stdout`，直到流关闭。

如果可能的话，我建议避免使用`stdin`向子进程发送数据，而应使用某种网络、管道或文件通信。正如我们将在下一节中看到回声客户端和服务器时，这些方法处理起来更加方便，并且不太可能发生死锁。

## 回声客户端和服务器

你可以得到的最基本的服务器类型是“回声”服务器，它会将接收到的所有消息发送回去。由于我们可以使用`asyncio`并行运行多个任务，因此我们可以在同一个脚本中运行服务器和客户端。当然，将它们分成两个进程也是可能的。

创建基本的客户端和服务器很容易：

```py
>>> import asyncio

>>> HOST = '127.0.0.1'
>>> PORT = 1234

>>> async def echo_client(message):
...     # Open the connection to the server
...     reader, writer = await asyncio.open_connection(HOST, PORT)
... 
...     print(f'Client sending {message!r}')
...     writer.write(message)
... 
...     # We need to drain and write the EOF to stop sending
...     writer.write_eof()
...     await writer.drain()
... 
...     async for line in reader:
...         print(f'Client received: {line!r}')
... 
...     writer.close()

>>> async def echo(reader, writer):
...     # Read all lines from the reader and send them back
...     async for line in reader:
...         print(f'Server received: {line!r}')
...         writer.write(line)
...         await writer.drain()
... 
...     writer.close()

>>> async def echo_server():
...     # Create a TCP server that listens on 'HOST'/'PORT' and
...     # calls 'echo' when a client connects.
...     server = await asyncio.start_server(echo, HOST, PORT)
... 
...     # Start listening
...     async with server:
...         await server.serve_forever()

>>> async def main():
...     # Create and run the echo server
...     server_task = asyncio.create_task(echo_server())
... 
...     # Wait a little for the server to start
...     await asyncio.sleep(0.01)
... 
...     # Create a client and send the message
...     await echo_client(b'test message')
... 
...     # Kill the server
...     server_task.cancel()

>>> asyncio.run(main())
Client sending b'test message'
Server received: b'test message'
Client received: b'test message' 
```

在这个例子中，我们可以看到我们使用`asyncio.create_task()`将服务器发送到后台。之后，我们必须等待一小段时间，以便后台任务开始工作，我们使用`asyncio.sleep()`来完成这个操作。`0.01`的睡眠时间是任意选择的（`0.001`可能也足够），但它应该足以让大多数系统与内核通信以创建监听套接字。一旦服务器开始运行，我们就启动客户端发送消息并等待响应。

自然地，这个例子可以用许多不同的方式编写。你不必使用`async for`，你可以使用`reader.readline()`读取到下一个换行符，或者你可以使用`reader.read(number_of_bytes)`读取特定数量的字符。这完全取决于你希望编写的协议。在 HTTP/1.1 协议的情况下，服务器期望一个`Connection: close`；在 SMTP 协议的情况下，应该发送一个`QUIT`消息。在我们的情况下，我们使用`EOF`字符作为指示符。

## 异步文件操作

你可能更希望某些操作是异步的，比如文件操作。尽管存储设备在近年来变得更快，但你并不总是使用快速的本地存储。例如，如果你想通过 Wi-Fi 连接写入网络驱动器，你可能会遇到相当多的延迟。通过使用`asyncio`，你可以确保这不会使你的整个解释器停滞不前。

不幸的是，目前还没有一种简单的方法可以在跨平台上通过`asyncio`执行文件操作，因为大多数操作系统都没有（可扩展的）异步文件操作支持。幸运的是，有人为这个问题创建了一个解决方案。`aiofiles`库在内部使用`threading`库为你提供一个`asyncio`接口来执行文件操作。虽然你可以轻松地使用`Executor`来为你处理文件操作，但`aiofiles`库是一个非常方便的包装器，我推荐使用它。

首先，安装库：

```py
$ pip3 install aiofiles 
```

现在我们可以使用 `aiofiles` 通过 `asyncio` 以非阻塞方式打开、读取和写入文件：

```py
>>> import asyncio
>>> import aiofiles

>>> async def main():
...     async with aiofiles.open('aiofiles.txt', 'w') as fh:
...         await fh.write('Writing to file')
...
...     async with aiofiles.open('aiofiles.txt', 'r') as fh:
...         async for line in fh:
...             print(line) 

>>> asyncio.run(main())
Writing to file 
```

`aiofiles` 的使用与常规的 `open()` 调用非常相似，除了在所有情况下都有 `async` 前缀。

## 创建异步生成器以支持异步 `for`

在前面的例子中，你可能想知道如何支持 `async for` 语句。本质上，这样做非常简单；你不再需要使用 `__iter__` 和 `__next__` 魔法函数在类中创建常规生成器，而是现在使用 `__aiter__` 和 `__anext__`：

```py
>>> import asyncio

>>> class AsyncGenerator:
...     def __init__(self, iterable):
...         self.iterable = iterable
...
...     async def __aiter__(self):
...         for item in self.iterable:
...             yield item

>>> async def main():
...     async_generator = AsyncGenerator([4, 2])
...
...     async for item in async_generator:
...         print(f'Got item: {item}')

>>> asyncio.run(main())
Got item: 4
Got item: 2 
```

实际上，代码与常规生成器和 `with` 语句相同，但你也可以从函数中访问 `asyncio` 代码。这些方法真正特殊的地方只是它们需要 `async` 前缀和名称中的 `a`，因此你得到 `__aiter__` 而不是 `__iter__`。

创建异步上下文管理器以支持异步 `with`

与异步生成器类似，我们也可以创建一个异步上下文管理器。现在，我们不再需要替换 `__iter__` 方法，而是用 `__enter__` 和 `__exit__` 方法分别替换 `__aenter__` 和 `__aexit__`。

实际上，代码与 `with` 语句相同，但你也可以从函数中访问 `asyncio` 代码：

```py
>>> import asyncio

>>> class AsyncContextManager:
...     async def __aenter__(self):
...         print('Hi :)')
...
...     async def __aexit__(self, exc_type, exc_value, traceback):
...         print('Bye :(')

>>> async def main():
...     async_context_manager = AsyncContextManager()
...
...     print('Before with')
...     async with async_context_manager:
...         print('During with')
...     print('After with')

>>> asyncio.run(main())
Before with
Hi :)
During with
Bye :(
After with 
```

与异步生成器类似，这些方法实际上并没有什么特殊之处。但特别是异步上下文管理器对于设置/清理方法非常有用，我们将在下一节中看到。

## 异步构造函数和析构函数

在某个时候，你可能想在构造函数和/或析构函数中运行一些异步代码，可能是为了初始化数据库连接或其他类型的网络连接。不幸的是，这实际上是不可能的。

自然地，使用 `__await__` 或元类，你可以绕过这一点来处理构造函数。并且使用 `asyncio.run(...)` 你可以为析构函数做类似的事情。但这两个都不是很好的解决方案——我建议重新结构化你的代码。

根据场景，我建议使用以下任一方法：

+   使用 `async with` 语句正确地进入/退出上下文管理器

+   在工厂模式中，一个 `async def` 为你生成和初始化类，同时还有一个 `async def close()` 作为异步析构函数

我们已经在上一节中看到了上下文管理器，这将是大多数情况下我会推荐的方法，例如创建数据库连接和/或事务，因为使用这种方法你不会意外忘记运行拆解操作。

工厂设计模式使用一个函数来简化对象的创建。在这种情况下，这意味着你将不再执行 `instance = SomeClass(...)`，而是使用 `instance = await SomeClass.create(...)`，这样你就可以有一个异步的初始化方法。

但当然，具有显式创建和关闭方法的工厂模式也是一个好选择：

```py
>>> import asyncio

>>> class SomeClass:
...     def __init__(self, *args, **kwargs):
...         print('Sync init')
...
...     async def init(self, *args, **kwargs):
...         print('Async init')
...
...     @classmethod
...     async def create(cls, *args, **kwargs):
...         # Create an instance of 'SomeClass' which calls the
...         # sync init: 'SomeClass.__init__(*args, **kwargs)'
...         self = cls(*args, **kwargs)
...         # Now we can call the async init:
...         await self.init(*args, **kwargs)
...         return self
...
...     async def close(self):
...         print('Async destructor')
...
...     def __del__(self):
...         print('Sync destructor')

>>> async def main():
...     # Note that we use 'SomeClass.create()' instead of
...     # 'SomeClass()' so we also run 'SomeClass().init()'
...     some_class = await SomeClass.create()
...     print('Using the class here')
...     await some_class.close()
...     del some_class

>>> asyncio.run(main())
Sync init
Async init
Using the class here
Async destructor
Sync destructor 
```

按照之前显示的操作顺序，您可以正确地创建和拆除`asyncio`类。作为一个安全措施（明确调用`close()`始终是更好的解决方案），您可以通过调用循环来向您的`__del__`添加一个`async`析构函数。

对于下一个示例，我们将使用`asyncpg`库，所以请确保首先安装它：

```py
$ pip3 install asyncpg 
```

现在，一个`asyncio`数据库连接到 PostgreSQL 可以像这样实现：

```py
import typing
import asyncio
import asyncpg

class AsyncPg:
    _connection: typing.Optional[asyncpg.Connection]

    async def init(self):
        self._connection = asyncpg.connect(...)

    async def close(self):
        await self._connection.close()

    def __del__(self):
        if self._connection:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
            else:
                loop.run_until_complete(self.close())

            self._connection = None 
```

您也可以创建一个注册表，以便轻松关闭所有创建的类，这样您就不会忘记在退出时这样做。但如果可能的话，我仍然建议使用上下文管理器风格的解决方案。您还可以通过创建`contextlib.ContextDecorator`的`async`版本来使用装饰器创建一个方便的快捷方式。

接下来，我们将探讨如何调试`asyncio`代码以及如何捕捉常见的错误。

# 调试`asyncio`

`asyncio`模块有一些特殊规定，使调试变得相对容易。鉴于`asyncio`中函数的异步性质，这是一个非常受欢迎的特性。虽然多线程/多进程函数或类的调试可能很困难——因为并发类可以并行更改环境变量——但使用`asyncio`，这同样困难，甚至更困难，因为`asyncio`后台任务在事件循环的堆栈中运行，而不是您的自己的堆栈。

如果您希望跳过本章的这一部分，我强烈建议您至少阅读关于*在所有任务完成之前退出*的部分。这涵盖了`asyncio`的一个**巨大的**陷阱。

调试`asyncio`的第一种也是最明显的方法是使用事件循环调试模式。我们有几个选项可以启用调试模式：

+   将`PYTHONASYNCIODEBUG`环境变量设置为`True`

+   使用`PYTHONDEVMODE`环境变量或通过执行带有`-X dev`命令行选项的 Python 来启用 Python 开发模式

+   将`debug=True`参数传递给`asyncio.run()`

+   调用`loop.set_debug()`

在这些方法中，我建议使用`PYTHONASYNCIODEBUG`或`PYTHONDEVMODE`环境变量，因为这些变量应用得非常早，因此可以捕获其他方法可能遗漏的几个错误。我们将在下一节关于忘记`await`语句的例子中看到这一点。

**关于设置环境变量的说明**

在大多数 Linux/Unix/Mac shell 会话中，可以使用`variable=value`作为前缀来设置环境变量：

```py
SOME_ENVIRONMENT_VARIABLE=value python3 script.py 
```

此外，您还可以使用`export`为当前 shell（当使用 ZSH 或 Bash）会话配置环境变量：

```py
export SOME_ENVIRONMENT_VARIABLE=value 
```

当前值可以使用以下行获取：

```py
echo $SOME_ENVIRONMENT_VARIABLE 
```

在 Windows 上，您可以使用`set`命令为您的本地 shell 会话配置环境变量：

```py
set SOME_ENVIRONMENT_VARIABLE=value 
```

当前值可以使用以下行获取：

```py
set SOME_ENVIRONMENT_VARIABLE 
```

当启用调试模式时，`asyncio`模块将检查一些常见的`asyncio`错误和问题。具体来说：

+   未调用的协程将引发异常。

+   从“错误”的线程调用协程会引发异常。这可能会发生，如果你有代码在不同的线程中运行，而当前事件循环是在不同的线程中运行的。这实际上是一个线程安全问题，这在*第十四章*中有介绍。

+   选择器的执行时间将被记录。

+   慢速协程（超过 100 毫秒）将被记录。这个超时可以通过`loop.slow_callback_duration`来修改。

+   当资源没有正确关闭时，将引发警告。

+   在执行前被销毁的任务将被记录。

让我们展示一些这些错误。

## 忘记 await 协程

这可能是最常见的`asyncio`错误，它已经咬了我很多次。做`some_coroutine()`而不是`await some_coroutine()`很容易，你通常会在已经太晚的时候发现它。

幸运的是，Python 可以帮助我们解决这个问题，所以让我们看看当将`PYTHONASYNCIODEBUG`设置为`1`时忘记`await`协程会发生什么：

```py
async def printer():
    print('This is a coroutine')

printer() 
```

这导致`printer`协程出现错误，我们忘记使用`await`：

```py
$ PYTHONASYNCIODEBUG=1 python3 T_13_forgot_await.py
T_13_forgot_await.py:5: RuntimeWarning: coroutine 'printer' was never awaited
  printer()
RuntimeWarning: Enable tracemalloc to get the object allocation traceback 
```

注意，这只会发生在事件循环被关闭时。事件循环无法知道你是否打算在稍后执行协程，因此这仍然可能很难调试。

这也是使用`PYTHONASYNCIODEBUG`环境变量而不是`loop.set_debug(True)`可能有所不同的案例之一。考虑一个场景，你可能有多个事件循环，并且忘记为它们全部启用调试模式，或者在一个忘记启用调试模式之前创建了协程，这意味着它不会被跟踪。

## 慢速阻塞函数

不考虑一个函数可能很慢并且会阻塞循环是很容易做到的。如果它有点慢但不足以让你注意到，除非你启用调试模式，否则你很可能永远不会发现它。让我们看看调试模式是如何帮助我们在这里的：

```py
import time
import asyncio

async def main():
    # Oh no... a synchronous sleep from asyncio code
    time.sleep(0.2)

asyncio.run(main(), debug=True) 
```

在这种情况下，我们“意外”地使用了`time.sleep()`而不是`asyncio.sleep()`。

对于这些问题，`debug=True`效果很好，但在开发时使用`PYTHONASYNCIODEBUG=1`永远不会有害：

```py
$ PYTHONASYNCIODEBUG=1 python3 T_14_slow_blocking_code.py
Executing <Task finished ...> took 0.204 seconds 
```

如我们所预期，我们得到了这个慢速函数的警告。

默认警告阈值设置为 100 毫秒，而我们睡眠了 200 毫秒，因此被报告了。如果需要，可以通过`loop.slow_callback_duration=<seconds>`来更改阈值。如果你正在使用较慢的系统，如树莓派，或者想要查找慢速代码，这可能很有用。

## 忘记检查结果或提前退出

使用`asyncio`编写代码的常见方式是使用`asyncio.create_task()`的 fire-and-forget，而不存储结果 future。虽然这本身并不是错误的，但如果你的代码中意外发生异常，如果没有启用调试模式，可能很难找到原因。

为了说明，我们将使用以下未捕获的异常，并在启用和未启用调试模式的情况下执行它：

```py
import asyncio

async def throw_exception():
    raise RuntimeError()

async def main():
    # Ignoring an exception from an async def
    asyncio.create_task(throw_exception())

asyncio.run(main()) 
```

如果我们不启用调试模式执行此操作，我们会得到以下输出：

```py
$ python3 T_15_forgotten_exception.py
Task exception was never retrieved
future: <Task finished ... at T_15_forgotten_exception.py:4> exception=RuntimeError()>
Traceback (most recent call last):
  File "T_15_forgotten_exception.py", line 5, in throw_exception
    raise RuntimeError()
RuntimeError 
```

虽然这很好地显示了异常发生的位置和发生的异常，但它并没有显示是谁或什么创建了协程。

现在如果我们启用调试模式重复同样的操作，我们得到以下结果：

```py
$ PYTHONASYNCIODEBUG=1 python3 T_15_forgotten_exception.py
Task exception was never retrieved
future: <Task finished ... at T_15_forgotten_exception.py:4> exception=RuntimeError() created at asyncio/tasks.py:361>
source_traceback: Object created at (most recent call last):
  File "T_15_forgotten_exception.py", line 13, in <module>
    asyncio.run(main())
  File "asyncio/runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "asyncio/base_events.py", line 629, in run_until_complete
    self.run_forever()
  File "asyncio/base_events.py", line 596, in run_forever
    self._run_once()
  File "asyncio/base_events.py", line 1882, in _run_once
    handle._run()
  File "asyncio/events.py", line 80, in _run
    self._context.run(self._callback, *self._args)
  File "T_15_forgotten_exception.py", line 10, in main
    asyncio.create_task(throw_exception())
  File "asyncio/tasks.py", line 361, in create_task
    task = loop.create_task(coro)
Traceback (most recent call last):
  File "T_15_forgotten_exception.py", line 5, in throw_exception
    raise RuntimeError()
RuntimeError 
```

这可能仍然有点难以阅读，但现在我们可以看到异常起源于`asyncio.create_task(throw_exception())`，我们甚至可以看到`asyncio.run(main())`的调用。

对于一个稍微大一点的代码库，这在追踪你的异常来源时可能是至关重要的。

## 在所有任务完成之前退出

注意这里，因为这个问题是极其微妙的，但如果你没有注意到，它可能会产生巨大的后果。

与忘记获取结果类似，当你在一个循环正在拆解时创建一个任务，该任务*不一定*会运行。在某些情况下，它没有运行的机会，你很可能不会注意到这一点。

看一下这个例子，我们有一个任务在生成另一个任务：

```py
import asyncio

async def sub_printer():
    print('Hi from the sub-printer')

async def printer():
    print('Before creating the sub-printer task')
    asyncio.create_task(sub_printer())
    print('After creating the sub-printer task')

async def main():
    asyncio.create_task(printer())

asyncio.run(main()) 
```

在这种情况下，即使是调试模式也无法帮助你。为了说明这一点，让我们看看当我们启用调试模式调用这个函数时会发生什么：

```py
$ PYTHONASYNCIODEBUG=1 python3 T_16_early_exit.py
Before creating the sub-printer task
After creating the sub-printer task 
```

对`sub_printer()`的调用似乎消失了。实际上并没有消失，但我们没有明确等待它完成，所以它从未有机会运行。

到目前为止，**最好的**解决方案是跟踪由`asyncio.create_task()`创建的所有 future，并在你的`main()`函数的末尾执行`await asyncio.gather(*futures)`。但这并不总是可行的选项——你可能无法访问由其他库创建的 future，或者 future 可能是在你无法轻松访问的作用域中创建的。那么你能做什么呢？

作为一个非常简单的解决方案，你可以在`main()`函数的末尾简单地等待：

```py
import asyncio

async def sub_printer():
    print('Hi from the sub-printer')

async def printer():
    print('Before creating the sub-printer task')
    asyncio.create_task(sub_printer())
    print('After creating the sub-printer task')

async def main():
    asyncio.create_task(printer())
    await asyncio.sleep(0.1)

asyncio.run(main()) 
```

对于这种情况，添加一点睡眠时间可以解决问题：

```py
$ python3 T_17_wait_for_exit.py
Before creating the sub-printer task
After creating the sub-printer task
Hi from the sub-printer 
```

但这只有在你的任务足够快或者你增加了睡眠时间的情况下才有效。如果我们有一个需要几秒钟的数据库拆解方法，我们仍然可能会遇到问题。作为一个非常粗略的解决方案，将此添加到你的代码中可能是有用的，因为当你缺少一个任务时，这会更为明显。

一个稍微好一点的解决方案是询问`asyncio`哪些任务仍在运行，并等待它们完成。这种方法的一个缺点是，如果你有一个永远运行的任务（换句话说，`while True`），你将永远等待脚本退出。

所以，让我们看看我们如何实现这样一个具有固定超时时间的功能，这样我们就不会永远等待：

```py
import asyncio

async def sub_printer():
    print('Hi from the sub-printer')

async def printer():
    print('Before creating the sub-printer task')
    asyncio.create_task(sub_printer())
    print('After creating the sub-printer task')

async def main():
    asyncio.create_task(printer())
    await shutdown()

async def shutdown(timeout=5):
    tasks = []
    # Collect all tasks from 'asyncio'
    for task in asyncio.all_tasks():
        # Make sure we skip our current task so we don't loop
        if task is not asyncio.current_task():
            tasks.append(task)

    for future in asyncio.as_completed(tasks, timeout=timeout):
        await future

asyncio.run(main()) 
```

这次，我们添加了一个`shutdown()`方法，它使用`asyncio.all_tasks()`从`asyncio`获取所有任务。在收集任务后，我们需要确保我们不会得到我们的当前任务，因为这会导致鸡生蛋的问题。`shutdown()`任务将在等待`shutdown()`任务完成时永远不会退出。

当所有任务都收集完毕后，我们使用`asyncio.as_completed()`等待它们完成并返回。如果等待时间超过`timeout`秒，`asyncio.as_completed()`将为我们抛出一个`asyncio.TimeoutError`。

您可以轻松地修改它以尝试取消所有任务，这样所有非受保护的任务将立即被取消。如果待处理任务在您的用例中不是关键的，您也可以将异常更改为警告。

`task = asyncio.shield(...)`可以防止`task.cancel()`和类似操作，就像洋葱一样。单个`asyncio.shield()`可以防止单个`task.cancel()`；要防止多次取消，您需要在循环中屏蔽，或者至少多次屏蔽。

最后，应该注意的是，这个解决方案也不是没有缺陷。在运行过程中，任务可能会产生新的任务；这不是这个实现所处理的，而且处理不当可能会导致永远等待。

现在我们知道了如何调试最常见的`asyncio`问题，是时候通过一些练习来结束了。

# 练习

在整个开发过程中，使用`asyncio`将需要积极的思考。除了`asyncio.run()`和类似的方法外，没有其他方法可以从同步代码中运行`async def`。这意味着您主`async def`和需要`asyncio`的代码之间的每个中间函数都必须是`async`的。

您可以将同步函数返回一个协程，这样父函数中的一个就可以在事件循环中运行它。但通常这会导致代码执行顺序非常混乱，所以我不会推荐走这条路。

简而言之，这意味着您尝试的任何启用`asyncio`调试设置的`asyncio`项目都是良好的实践。然而，我们可以提出一些挑战：

+   尝试创建一个`asyncio`基类，当您完成时可以自动注册所有实例，以便于轻松关闭/解构

+   使用 executors 为文件或网络操作等同步过程创建`asyncio`包装类

+   将您的脚本或项目转换为`asyncio`

这些练习的示例答案可以在 GitHub 上找到：`github.com/mastering-python/exercises`。我们鼓励您提交自己的解决方案，并从他人的替代方案中学习。

# 概述

在本章中，我们看到了：

+   `asyncio`的基本概念及其交互方式

+   如何使用`asyncio`运行外部进程

+   如何使用`asyncio`创建服务器和客户端

+   如何使用`asyncio`创建上下文管理器

+   如何使用`asyncio`创建生成器

+   如何调试使用`asyncio`时常见的错误

+   如何避免未完成任务的陷阱

到现在为止，您应该知道如何在等待结果的同时保持主循环的响应性，而无需求助于轮询。在 *第十四章*，“当单个 CPU 核心不够用时——多进程”，我们将学习 `threading` 和 `multiprocessing` 作为在并行运行多个函数时的 `asyncio` 替代方案。

对于新项目，我强烈建议从头开始使用 `asyncio`，因为它通常是处理外部资源的最快解决方案。然而，对于现有脚本，这可能是一个非常侵入性的过程。因此，了解 `threading` 和 `multiprocessing` 确实很重要，也因为 `asyncio` 可以利用它们，您应该了解线程和进程的安全性。

当基于 `asyncio` 库构建工具时，请确保搜索现成的库来解决您的问题，因为 `asyncio` 正在每年获得更多的采用。在许多情况下，有人已经为您创建了一个库。

接下来是使用 `threading` 和 `multiprocessing` 的并行执行。

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：[`discord.gg/QMzJenHuJf`](https://discord.gg/QMzJenHuJf)

![二维码](img/QR_Code156081100001293319171.png)
