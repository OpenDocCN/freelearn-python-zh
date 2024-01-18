# 并发

在本章中，我们将介绍以下食谱：

+   线程池-通过线程池并发运行任务

+   协程-通过协程交错执行代码

+   进程-将工作分派给多个子进程

+   期货-期货代表将来会完成的任务

+   计划任务-设置在特定时间运行的任务，或每隔几秒运行一次

+   在进程之间共享数据-管理可在多个进程中访问的变量

# 介绍

并发是在相同的时间段内运行两个或多个任务的能力，无论它们是并行的还是不并行的。Python提供了许多工具来实现并发和异步行为：线程、协程和进程。虽然其中一些由于设计（协程）或全局解释器锁（线程）的原因不允许真正的并行，但它们非常易于使用，并且可以用于执行并行I/O操作或以最小的工作量交错函数。当需要真正的并行时，Python中的多进程足够容易，可以成为任何类型软件的可行解决方案。

本章将介绍在Python中实现并发的最常见方法，将向您展示如何执行异步任务，这些任务将在后台等待特定条件，并且如何在进程之间共享数据。

# 线程池

线程在软件中实现并发的历史上一直是最常见的方式。

理论上，当系统允许时，这些线程可以实现真正的并行，但在Python中，全局解释器锁（GIL）不允许线程实际上利用多核系统，因为锁将允许单个Python操作在任何给定时间进行。

因此，线程在Python中经常被低估，但实际上，即使涉及GIL，它们也可以是运行I/O操作的非常方便的解决方案。

在使用协程时，我们需要一个`run`循环和一些自定义代码来确保I/O操作可以并行进行。使用线程，我们可以在线程中运行任何类型的函数，如果该函数进行某种I/O操作，例如从套接字或磁盘中读取，其他线程将同时进行。

线程的一个主要缺点是产生它们的成本。这经常被认为是协程可能是更好的解决方案的原因之一，但是有一种方法可以避免在需要线程时支付成本：`ThreadPool`。

`ThreadPool`是一组线程，通常在应用程序启动时启动，并且一直保持空闲，直到您实际上有一些工作要分派。这样，当我们有一个任务想要在单独的线程中运行时，我们只需将其发送到`ThreadPool`，`ThreadPool`将把它分配给它拥有的所有线程中的第一个可用线程。由于这些线程已经在那里运行，我们不必每次有工作要做时都支付产生线程的成本。

# 如何做...

此食谱的步骤如下：

1.  为了展示`ThreadPool`的工作原理，我们需要两个我们想要同时运行的操作。一个将从网络中获取一个URL，这可能需要一些时间：

```py
def fetch_url(url):
    """Fetch content of a given url from the web"""
    import urllib.request
    response = urllib.request.urlopen(url)
    return response.read()
```

1.  另一个将只是等待给定条件为真，一遍又一遍地循环，直到完成：

```py
def wait_until(predicate):
    """Waits until the given predicate returns True"""
    import time
    seconds = 0
    while not predicate():
        print('Waiting...')
        time.sleep(1.0)
        seconds += 1
    print('Done!')
    return seconds
```

1.  然后我们将只下载`https://httpbin.org/delay/3`，这将需要3秒，并且同时等待下载完成。

1.  为此，我们将在一个`ThreadPool`（四个线程）中运行这两个任务，并等待它们都完成：

```py
>>> from multiprocessing.pool import ThreadPool
>>> pool = ThreadPool(4)
>>> t1 = pool.apply_async(fetch_url, args=('https://httpbin.org/delay/3',))
>>> t2 = pool.apply_async(wait_until, args=(t1.ready, ))
Waiting...
>>> pool.close()
>>> pool.join()
Waiting...
Waiting...
Waiting...
Done!
>>> print('Total Time:', t2.get())
Total Time: 4
>>> print('Content:', t1.get())
Content: b'{"args":{},"data":"","files":{},"form":{},
            "headers":{"Accept-Encoding":"identity",
            "Connection":"close","Host":"httpbin.org",
            "User-Agent":"Python-urllib/3.5"},
            "origin":"99.199.99.199",
            "url":"https://httpbin.org/delay/3"}\n'
```

# 它是如何工作的...

`ThreadPool`由两个主要组件组成：一堆线程和一堆队列。在创建池时，一些协调线程与您在池初始化时指定的工作线程一起启动。

工作线程将负责实际运行分派给它们的任务，而编排线程将负责管理工作线程，例如在池关闭时告诉它们退出，或在它们崩溃时重新启动它们。

如果没有提供工作线程的数量，`TaskPool`将会启动与系统核心数量相同的线程，由`os.cpu_count()`返回。

一旦线程启动，它们将等待从包含要完成的工作的队列中获取内容。一旦队列有条目，工作线程将唤醒并消耗它，开始工作。

工作完成后，工作及其结果将放回结果队列，以便等待它们的人可以获取它们。

因此，当我们创建`TaskPool`时，实际上启动了四个工作线程，这些线程开始等待从任务队列中获取工作：

```py
>>> pool = ThreadPool(4)
```

然后，一旦我们为`TaskPool`提供了工作，实际上我们将两个函数排入任务队列，一旦有工作线程可用，它就会获取其中一个并开始运行：

```py
>>> t1 = pool.apply_async(fetch_url, args=('https://httpbin.org/delay/3',))
```

与此同时，`TaskPool`返回一个`AsyncResult`对象，该对象有两个有趣的方法：`AsyncResult.ready()`告诉我们结果是否准备好（任务完成），`AsyncResult.get()`在结果可用时返回结果。

我们排队的第二个函数是等待特定谓词为`True`的函数，在这种情况下，我们提供了` t1.ready`，这是先前`AsyncResult`的就绪方法：

```py
>>> t2 = pool.apply_async(wait_until, args=(t1.ready, ))
```

这意味着第二个任务将在第一个任务完成后完成，因为它将等待直到`t1.ready() == True`。

一旦这两个任务都在运行，我们告诉`pool`我们没有更多事情要做，这样它就可以在完成任务后退出：

```py
>>> pool.close()
```

然后我们等待`pool`退出：

```py
>>> pool.join()
```

这样，我们将等待两个任务都完成，然后退出`pool`启动的所有线程。

一旦我们知道所有任务都已完成（因为`pool.join()`返回），我们可以获取结果并打印它们：

```py
>>> print('Total Time:', t2.get())
Total Time: 4
>>> print('Content:', t1.get())
Content: b'{"args":{},"data":"","files":{},"form":{},
            "headers":{"Accept-Encoding":"identity",
            "Connection":"close","Host":"httpbin.org",
            "User-Agent":"Python-urllib/3.5"},
            "origin":"99.199.99.199",
            "url":"https://httpbin.org/delay/3"}\n'
```

如果我们有更多工作要做，我们将避免运行`pool.close()`和`pool.join()`方法，这样我们就可以将更多工作发送给`TaskPool`，一旦有空闲线程，工作就会完成。

# 还有更多...

当您有多个条目需要反复应用相同操作时，`ThreadPool`特别方便。假设您有一个包含四个URL的列表需要下载：

```py
urls = [
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/2",
    "https://httpbin.org/delay/3",
    "https://httpbin.org/delay/4"
]
```

在单个线程中获取它们将需要很长时间：

```py
def fetch_all_urls():
    contents = []
    for url in urls:
        contents.append(fetch_url(url))
    return contents
```

我们可以通过`timeit`模块运行函数来测试时间：

```py
>>> import timeit
>>> timeit.timeit(fetch_all_urls, number=1)
12.116707602981478
```

如果我们可以使用单独的线程来执行每个函数，那么获取所有提供的URL只需要最慢的一个的时间，因为下载将同时进行。

`ThreadPool`实际上为我们提供了`map`方法，该方法正是这样做的：它将一个函数应用于一系列参数：

```py
def fetch_all_urls_theraded():
    pool = ThreadPool(4)
    return pool.map(fetch_url, urls)
```

结果将是一个包含每次调用返回结果的列表，我们可以轻松测试这将比我们原始示例快得多：

```py
>>> timeit.timeit(fetch_all_urls_theraded, number=1)
4.660976745188236
```

# 协程

线程是大多数语言和用例中实现并发的最常见方式，但它们在成本方面很昂贵，而且虽然`ThreadPool`在涉及数千个线程的情况下可能是一个很好的解决方案，但通常不合理涉及数千个线程。特别是在涉及长期I/O时，您可能会轻松地达到数千个并发运行的操作（考虑一下HTTP服务器可能需要处理的并发HTTP请求数量），其中大多数任务将无所事事，只是大部分时间等待来自网络或磁盘的数据。

在这些情况下，异步I/O是首选的方法。与同步阻塞I/O相比，你的代码坐在那里等待读取或写入操作完成，异步I/O允许需要数据的任务启动读取操作，切换到做其他事情，一旦数据可用，就返回到原来的工作。

在某些情况下，可用数据的通知可能以信号的形式到来，这将中断并发运行的代码，但更常见的是，异步I/O是通过使用选择器（如`select`、`poll`或`epoll`）和一个事件循环来实现的，该事件循环将在选择器通知数据可用时立即恢复等待数据的函数。

这实际上导致了交错运行的功能，能够运行一段时间，达到需要一些I/O的时候，将控制权传递给另一个函数，只要它需要执行一些I/O，就会立即返回。通过暂停和恢复它们的执行来交错执行的函数称为**协程**，因为它们是协作运行的。

# 如何做...

在Python中，协程是通过`async def`语法实现的，并通过`asyncio`事件循环执行。

例如，我们可以编写一个函数，运行两个协程，从给定的秒数开始倒计时，并打印它们的进度。这将很容易让我们看到这两个协程是同时运行的，因为我们会看到一个协程的输出与另一个协程的输出交错出现：

```py
import asyncio

async def countdown(identifier, n):
    while n > 0:
        print('left:', n, '({})'.format(identifier))
        await asyncio.sleep(1)
        n -= 1

async def main():
    await asyncio.wait([
        countdown("A", 2),
        countdown("B", 3)
    ])
```

一旦创建了一个事件循环，并在其中运行`main`，我们将看到这两个函数在运行：

```py
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(main())
left: 2 (A)
left: 3 (B)
left: 1 (A)
left: 2 (B)
left: 1 (B)
```

一旦执行完成，我们可以关闭事件循环，因为我们不再需要它：

```py
>>> loop.close()
```

# 它是如何工作的...

我们协程世界的核心是**事件循环**。没有事件循环，就不可能运行协程（或者说，会变得非常复杂），所以我们代码的第一件事就是创建一个事件循环：

```py
>>> loop = asyncio.get_event_loop()
```

然后我们要求事件循环等待直到提供的协程完成：

```py
loop.run_until_complete(main())
```

`main`协程只启动两个`countdown`协程并等待它们完成。这是通过使用`await`来完成的，而`asyncio.wait`函数负责等待一堆协程：

```py
await asyncio.wait([
    countdown("A", 2),
    countdown("B", 3)
])
```

`await`在这里很重要，因为我们在谈论协程，所以除非它们被明确等待，否则我们的代码会立即向前移动，因此，即使我们调用了`asyncio.wait`，我们也不会等待。

在这种情况下，我们正在等待两个倒计时完成。第一个倒计时将从`2`开始，并由字符`A`标识，而第二个倒计时将从`3`开始，并由`B`标识。

`countdown`函数本身非常简单。它只是一个永远循环并打印剩下多少时间要等待的函数。

在每个循环之间等待一秒钟，这样就等待了预期的秒数：

```py
await asyncio.sleep(1)
```

你可能会想知道为什么我们使用`asyncio.sleep`而不是`time.sleep`，原因是，当使用协程时，你必须确保每个其他会阻塞的函数也是一个协程。这样，你就知道在你的函数被阻塞时，你会让其他协程继续向前移动。

通过使用`asyncio.sleep`，我们让事件循环在第一个协程等待时推进另一个`countdown`函数，因此，我们正确地交错执行了这两个函数。

这可以通过检查输出来验证。当使用`asyncio.sleep`时，输出将在两个函数之间交错出现：

```py
left 2 (A)
left 3 (B)
left 1 (A)
left 2 (B)
left 1 (B)
```

当使用`time.sleep`时，第一个协程必须完全完成，然后第二个协程才能继续向前移动：

```py
left 2 (A)
left 1 (A)
left 3 (B)
left 2 (B)
left 1 (B)
```

因此，使用协程时的一个一般规则是，每当要调用会阻塞的东西时，确保它也是一个协程，否则你将失去协程的并发属性。

# 还有更多...

我们已经知道协程最重要的好处是事件循环能够在它们等待I/O操作时暂停它们的执行，以便让其他协程继续。虽然目前没有支持协程的HTTP协议的内置实现，但很容易推出一个后备版本来重现我们同时下载网站的示例以跟踪它花费了多长时间。

至于`ThreadPool`示例，我们将需要`wait_until`函数，它将等待任何给定的谓词为真：

```py
async def wait_until(predicate):
    """Waits until the given predicate returns True"""
    import time
    seconds = 0
    while not predicate():
        print('Waiting...')
        await asyncio.sleep(1)
        seconds += 1
    print('Done!')
    return seconds
```

我们还需要一个`fetch_url`函数来下载URL的内容。由于我们希望这个函数作为协程运行，所以我们不能依赖`urllib`，否则它会永远阻塞而不是将控制权传递回事件循环。因此，我们将不得不使用`asyncio.open_connection`来读取数据，这将在纯TCP级别工作，因此需要我们自己实现HTTP支持：

```py
async def fetch_url(url):
    """Fetch content of a given url from the web"""
    url = urllib.parse.urlsplit(url)
    reader, writer = await asyncio.open_connection(url.hostname, 80)
    req = ('GET {path} HTTP/1.0\r\n'
           'Host: {hostname}\r\n'
           '\r\n').format(path=url.path or '/', hostname=url.hostname)
    writer.write(req.encode('latin-1'))
    while True:
        line = await reader.readline()
        if not line.strip():
            # Read until the headers, from here on is the actualy response.
            break
    return await reader.read()
```

在这一点上，可以交错两个协程，看到下载与等待同时进行，并且在预期时间内完成：

```py
>>> loop = asyncio.get_event_loop()
>>> t1 = asyncio.ensure_future(fetch_url('http://httpbin.org/delay/3'))
>>> t2 = asyncio.ensure_future(wait_until(t1.done))
>>> loop.run_until_complete(t2)
Waiting...
Waiting...
Waiting...
Waiting...
Done!
>>> loop.close()
>>> print('Total Time:', t2.result())
Total Time: 4
>>> print('Content:', t1.result())
Content: b'{"args":{},"data":"","files":{},"form":{},
            "headers":{"Connection":"close","Host":"httpbin.org"},
            "origin":"93.147.95.71",
            "url":"http://httpbin.org/delay/3"}\n'
```

# 进程

线程和协程是与Python GIL并存的并发模型，并利用I/O操作留下的执行时间来允许其他任务继续。在现代多核系统中，能够利用系统提供的全部性能并涉及真正的并行性并将工作分配到所有可用的核心上是非常好的。

Python标准库提供了非常精细的工具来处理多进程，这是在Python上利用并行性的一个很好的解决方案。由于多进程将导致多个独立的解释器，因此GIL不会成为障碍，并且与线程和协程相比，甚至可能更容易理解它们作为完全隔离的进程，需要合作，而不是考虑在同一系统中共享底层内存状态的多个线程/协程。

管理进程的主要成本通常是生成成本和确保您不会在任何奇怪的情况下分叉子进程的复杂性，从而导致在内存中复制不需要的数据或重用文件描述符。

`multiprocessing.ProcessPool`可以是解决所有这些问题的一个很好的解决方案，因为在软件开始时启动它将确保当我们有任务要提交给子进程时不必支付任何特定的成本。此外，通过在开始时仅创建进程，我们可以保证软件的状态可预测（并且大部分为空），被复制以创建子进程。

# 如何做...

就像在*ThreadPool*示例中一样，我们将需要两个函数，它们将作为我们在进程中并行运行的任务。

在进程的情况下，我们实际上不需要执行I/O来实现并发运行，因此我们的任务可以做任何事情。我将使用计算斐波那契数列并打印出进度，以便我们可以看到两个进程的输出是如何交错的：

```py
import os

def fib(n, seen):
    if n not in seen and n % 5 == 0:
        # Print out only numbers we didn't yet compute
        print(os.getpid(), '->', n)
        seen.add(n)

    if n < 2:
        return n
    return fib(n-2, seen) + fib(n-1, seen)
```

因此，现在我们需要创建运行`fib`函数并生成计算的多进程`Pool`：

```py
>>> from multiprocessing import Pool
>>> pool = Pool()
>>> t1 = pool.apply_async(fib, args=(20, set()))
>>> t2 = pool.apply_async(fib, args=(22, set()))
>>> pool.close()
>>> pool.join()
42588 -> 20
42588 -> 10
42588 -> 0
42589 -> 20
42588 -> 5
42589 -> 10
42589 -> 0
42589 -> 5
42588 -> 15
42589 -> 15
>>> t1.get()
6765
>>> t2.get()
17711
```

您可以看到两个进程的进程ID是如何交错的，一旦作业完成，就可以获得它们两者的结果。

# 它是如何工作的...

创建`multiprocessing.Pool`时，将通过`os.fork`或生成一个新的Python解释器创建与系统上的核心数量相等的进程（由`os.cpu_count()`指定），具体取决于底层系统支持的情况：

```py
>>> pool = Pool()
```

一旦启动了新进程，它们将都执行相同的操作：执行`worker`函数，该函数循环消耗发送到`Pool`的作业队列，并逐个运行它们。

这意味着如果我们创建了两个进程的`Pool`，我们将有两个工作进程。一旦我们要求`Pool`执行某些操作（通过`Pool.apply_async`，`Pool.map`或任何其他方法），作业（函数及其参数）将被放置在`multiprocessing.SimpleQueue`中，工作进程将从中获取。

一旦`worker`从队列中获取任务，它将运行它。如果有多个`worker`实例在运行，每个实例都会从队列中选择一个任务并运行它。

任务完成后，执行的函数结果将被推送回结果队列（与任务本身一起，以标识结果所属的任务），`Pool`将能够消耗结果并将其提供给最初启动任务的代码。

所有这些通信都发生在多个进程之间，因此它不能在内存中发生。相反，`multiprocessing.SimpleQueue`使用`pipe`，每个生产者将写入`pipe`，每个消费者将从`pipe`中读取。

由于`pipe`只能读取和写入字节，我们提交给`pool`的参数以及由`pool`执行的函数的结果通过`pickle`协议转换为字节。只要发送方和接收方都有相同的模块可用，它就能够在Python对象之间进行编组/解组。

因此，我们向`Pool`提交我们的请求：

```py
>>> t1 = pool.apply_async(fib, args=(20, set()))
```

`fib`函数，`20`和空集都被pickled并发送到队列中，供`Pool`的一个工作进程消耗。

与此同时，当工作进程正在获取数据并运行斐波那契函数时，我们加入池，以便我们的主进程将阻塞，直到池中的所有进程都完成：

```py
>>> pool.close()
>>> pool.join()
```

理论上，池中的进程永远不会完成（它将永远运行，不断地查找队列中的任务）。在调用`join`之前，我们关闭池。关闭池告诉池一旦它们完成当前正在做的事情，就*退出所有进程*。

然后，在`close`之后立即加入，我们等待直到池完成它现在正在做的事情，即为我们提供服务的两个请求。

与线程一样，`multiprocessing.Pool`返回`AsyncResult`对象，这意味着我们可以通过`AsyncResult.ready()`方法检查它们的完成情况，并且一旦准备就绪，我们可以通过`AsyncResult.get()`获取返回的值：

```py
>>> t1.get()
6765
>>> t2.get()
17711
```

# 还有更多...

`multiprocessing.Pool`的工作方式与`multiprocessing.pool.ThreadPool`几乎相同。实际上，它们共享很多实现，因为其中一个是另一个的子类。

但由于使用的底层技术不同，这会导致一些主要差异。一个基于线程，另一个基于子进程。

使用进程的主要好处是Python解释器锁不会限制它们的并行性，它们将能够实际并行运行。

另一方面，这是有成本的。使用进程在启动时间上更昂贵（fork一个进程通常比生成一个线程慢），而且在内存使用方面更昂贵，因为每个进程都需要有自己的内存状态。虽然大部分系统通过写时复制等技术大大降低了这些成本，但线程通常比进程便宜得多。

因此，通常最好在应用程序开始时只启动进程`pool`，这样生成进程的额外成本只需支付一次。

进程不仅更昂贵，而且与线程相比，它们不共享程序的状态；每个进程都有自己的状态和内存。因此，无法在`Pool`和执行任务的工作进程之间共享数据。所有数据都需要通过`pickle`编码并通过`pipe`发送到另一端进行消耗。与可以依赖共享队列的线程相比，这将产生巨大的成本，特别是当需要发送的数据很大时。

因此，通常最好避免在参数或返回值中涉及大文件或数据时涉及进程，因为该数据将不得不多次复制才能到达最终目的地。在这种情况下，最好将数据保存在磁盘上，并传递文件的路径。

# 未来

当启动后台任务时，它可能会与您的主流程并发运行，永远不会完成自己的工作（例如`ThreadPool`的工作线程），或者它可能是某种迟早会向您返回结果并且您可能正在等待该结果的东西（例如在后台下载URL内容的线程）。

这些第二种类型的任务都共享一个共同的行为：它们的结果将在`_future_`中可用。因此，通常将将来可用的结果称为`Future`。编程语言并不都具有完全相同的`futures`定义，而在Python中，`Future`是指将来会完成的任何函数，通常返回一个结果。

`Future`是可调用本身，因此与实际用于运行可调用的技术无关。您需要一种让可调用的执行继续进行的方法，在Python中，这由`Executor`提供。

有一些执行器可以将futures运行到线程、进程或协程中（在协程的情况下，循环本身就是执行器）。

# 如何做...

要运行一个future，我们将需要一个执行器（`ThreadPoolExecutor`、`ProcessPoolExecutor`）和我们实际想要运行的futures。为了举例说明，我们将使用一个返回加载网页所需时间的函数，以便对多个网站进行基准测试，以查看哪个网站速度最快：

```py
import concurrent.futures
import urllib.request
import time

def benchmark_url(url):
    begin = time.time()
    with urllib.request.urlopen(url) as conn:
        conn.read()
    return (time.time() - begin, url)

class UrlsBenchmarker:
    def __init__(self, urls):
        self._urls = urls

    def run(self, executor):
        futures = self._benchmark_urls(executor)
        fastest = min([
            future.result() for future in 
                concurrent.futures.as_completed(futures)
        ])
        print('Fastest Url: {1}, in {0}'.format(*fastest))

    def _benchmark_urls(self, executor):
        futures = []
        for url in self._urls:
            future = executor.submit(benchmark_url, url)
            future.add_done_callback(self._print_timing)
            futures.append(future)
        return futures

    def _print_timing(self, future):
        print('Url {1} downloaded in {0}'.format(
            *future.result()
        ))
```

然后我们可以创建任何类型的执行器，并让`UrlsBenchmarker`在其中运行其`futures`：

```py
>>> import concurrent.futures
>>> with concurrent.futures.ThreadPoolExecutor() as executor:
...     UrlsBenchmarker([
...             'http://time.com/',
...             'http://www.cnn.com/',
...             'http://www.facebook.com/',
...             'http://www.apple.com/',
...     ]).run(executor)
...
Url http://time.com/ downloaded in 1.0580978393554688
Url http://www.apple.com/ downloaded in 1.0482590198516846
Url http://www.facebook.com/ downloaded in 1.6707532405853271
Url http://www.cnn.com/ downloaded in 7.4976489543914795
Fastest Url: http://www.apple.com/, in 1.0482590198516846
```

# 它是如何工作的...

`UrlsBenchmarker`将通过`UrlsBenchmarker._benchmark_urls`为每个URL触发一个future：

```py
for url in self._urls:
    future = executor.submit(benchmark_url, url)
```

每个future将执行`benchmark_url`，该函数下载给定URL的内容并返回下载所用的时间，以及URL本身：

```py
def benchmark_url(url):
    begin = time.time()
    # download url here...
    return (time.time() - begin, url)
```

返回URL本身是必要的，因为`future`可以知道其返回值，但无法知道其参数。因此，一旦我们`submit`函数，我们就失去了它与哪个URL相关，并通过将其与时间一起返回，每当时间存在时我们将始终有URL可用。

然后对于每个`future`，通过`future.add_done_callback`添加一个回调：

```py
future.add_done_callback(self._print_timing)
```

一旦future完成，它将调用`UrlsBenchmarker._print_timing`，该函数打印运行URL所用的时间。这通知用户基准测试正在进行，并且已完成其中一个URL。

`UrlsBenchmarker._benchmark_urls` 然后会返回一个包含所有需要在列表中进行基准测试的URL的`futures`。

然后将该列表传递给`concurrent.futures.as_completed`。这将创建一个迭代器，按照完成的顺序返回所有`futures`，并且只有在它们完成时才返回。因此，我们知道通过迭代它，我们只会获取已经完成的`futures`，并且一旦消耗了所有已完成的`futures`，我们将阻塞等待新的future完成：

```py
[
    future.result() for future in 
        concurrent.futures.as_completed(futures)
]
```

因此，只有当所有`futures`都完成时，循环才会结束。

已完成`futures`的列表被`list`推导式所消耗，它将创建一个包含这些`futures`结果的列表。

由于结果都是以（时间，URL）形式存在，我们可以使用`min`来获取具有最短时间的结果，即下载时间最短的URL。

这是因为比较两个元组会按顺序比较元素：

```py
>>> (1, 5) < (2, 0)
True
>>> (2, 1) < (0, 5)
False
```

因此，在元组列表上调用`min`将抓取元组中第一个元素的最小值的条目：

```py
>>> min([(1, 2), (2, 0), (0, 7)])
(0, 7)
```

当有两个第一个元素具有相同值时，才会查看第二个元素：

```py
>>> min([(0, 7), (1, 2), (0, 3)])
(0, 3)
```

因此，我们获取具有最短时间的 URL（因为时间是由未来返回的元组中的第一个条目）并将其打印为最快的：

```py
fastest = min([
    future.result() for future in 
        concurrent.futures.as_completed(futures)
])
print('Fastest Url: {1}, in {0}'.format(*fastest))
```

# 还有更多...

未来执行器与 `multiprocessing.pool` 提供的工作进程池非常相似，但它们有一些差异可能会推动您朝一个方向或另一个方向。

主要区别可能是工作进程的启动方式。池会启动固定数量的工作进程，在创建池时同时创建和启动它们。因此，早期创建池会将生成工作进程的成本移到应用程序的开始。这意味着应用程序可能启动相当慢，因为它可能需要根据您请求的工作进程数量或系统核心数量来分叉许多进程。相反，执行器仅在需要时创建工作进程，并且它旨在在将来避免在有可用工作进程时创建新的工作进程。

因此，执行器通常更快地启动，但第一次将未来发送到执行器时会有更多的延迟，而池则将大部分成本集中在启动时间上。因此，如果您经常需要创建和销毁一组工作进程池的情况下，使用 `futures` 执行器可能更有效。

# 调度的任务

一种常见的后台任务是应该在任何给定时间自行在后台运行的操作。通常，这些通过 cron 守护程序或类似的系统工具进行管理，通过配置守护程序在提供的时间运行给定的 Python 脚本。

当您有一个主要应用程序需要周期性执行任务（例如过期缓存、重置密码链接、刷新待发送的电子邮件队列或类似任务）时，通过 cron 作业进行操作并不是可行的，因为您需要将数据转储到其他进程可以访问的地方：磁盘上、数据库上，或者任何类似的共享存储。

幸运的是，Python 标准库有一种简单的方法来安排在任何给定时间执行并与线程一起加入的任务。这可以是一个非常简单和有效的定时后台任务的解决方案。

# 如何做...

`sched` 模块提供了一个完全功能的调度任务执行器，我们可以将其与线程混合使用，创建一个后台调度器：

```py
import threading
import sched
import functools

class BackgroundScheduler(threading.Thread):
    def __init__(self, start=True):
        self._scheduler = sched.scheduler()
        self._running = True
        super().__init__(daemon=True)
        if start:
            self.start()

    def run_at(self, time, action, args=None, kwargs=None):
        self._scheduler.enterabs(time, 0, action, 
                                argument=args or tuple(), 
                                kwargs=kwargs or {})

    def run_after(self, delay, action, args=None, kwargs=None):
        self._scheduler.enter(delay, 0, action, 
                            argument=args or tuple(), 
                            kwargs=kwargs or {})

    def run_every(self, seconds, action, args=None, kwargs=None):
        @functools.wraps(action)
        def _f(*args, **kwargs):
            try:
                action(*args, **kwargs)
            finally:
                self.run_after(seconds, _f, args=args, kwargs=kwargs)
        self.run_after(seconds, _f, args=args, kwargs=kwargs)

    def run(self):
        while self._running:
            delta = self._scheduler.run(blocking=False)
            if delta is None:
                delta = 0.5
            self._scheduler.delayfunc(min(delta, 0.5))

    def stop(self):
        self._running = False
```

`BackgroundScheduler` 可以启动，并且可以向其中添加作业，以便在固定时间开始执行它们：

```py
>>> import time
>>> s = BackgroundScheduler()
>>> s.run_every(2, lambda: print('Hello World'))
>>> time.sleep(5)
Hello World
Hello World
>>> s.stop()
>>> s.join()
```

# 工作原理...

`BackgroundScheduler` 是 `threading.Thread` 的子类，因此它在我们的应用程序在做其他事情时在后台运行。注册的任务将在辅助线程中触发和执行，而不会妨碍主要代码：

```py
class BackgroundScheduler(threading.Thread):
        def __init__(self):
            self._scheduler = sched.scheduler()
            self._running = True
            super().__init__(daemon=True)
            self.start()
```

每当创建 `BackgroundScheduler` 时，它的线程也会启动，因此它立即可用。该线程将以 `daemon` 模式运行，这意味着如果程序在结束时仍在运行，它不会阻止程序退出。

通常 Python 在退出应用程序时会等待所有线程，因此将线程设置为 `daemon` 可以使其在无需等待它们的情况下退出。

`threading.Thread` 作为线程代码执行 `run` 方法。在我们的情况下，这是一个重复运行调度器中注册的任务的方法：

```py
def run(self):
    while self._running:
        delta = self._scheduler.run(blocking=False)
        if delta is None:
            delta = 0.5
        self._scheduler.delayfunc(min(delta, 0.5))
```

`_scheduler.run(blocking=False)` 表示从计划的任务中选择一个任务并运行它。然后，它返回在运行下一个任务之前仍需等待的时间。如果没有返回时间，这意味着没有要运行的任务。

通过 `_scheduler.delayfunc(min(delta, 0.5))`，我们等待下一个任务需要运行的时间，最多为半秒钟。

我们最多等待半秒钟，因为当我们等待时，调度的任务可能会发生变化。可能会注册一个新任务，我们希望确保它不必等待超过半秒钟才能被调度器捕捉到。

如果我们等待的时间正好是下一个任务挂起的时间，我们可能会运行，得到下一个任务在60秒内，然后开始等待60秒。但是，如果我们在等待时，用户注册了一个必须在5秒内运行的新任务，我们无论如何都会在60秒内运行它，因为我们已经在等待。通过等待最多0.5秒，我们知道需要半秒钟才能接收下一个任务，并且它将在5秒内正确运行。

等待少于下一个任务挂起的时间不会使任务运行得更快，因为调度程序不会运行任何已经超过其计划时间的任务。因此，如果没有要运行的任务，调度程序将不断告诉我们*你必须等待*，我们将等待半秒钟，直到达到下一个计划任务的计划时间为止。

`run_at`，`run_after`和`run_every`方法实际上是注册在特定时间执行函数的方法。

`run_at`和`run_after`只是包装调度程序的`enterabs`和`enter`方法，这些方法允许我们在特定时间或*n*秒后注册任务运行。

最有趣的函数可能是`run_every`，它每*n*秒运行一次任务：

```py
def run_every(self, seconds, action, args=None, kwargs=None):
    @functools.wraps(action)
    def _f(*args, **kwargs):
        try:
            action(*args, **kwargs)
        finally:
            self.run_after(seconds, _f, args=args, kwargs=kwargs)
    self.run_after(seconds, _f, args=args, kwargs=kwargs)
```

该方法接受必须运行的可调用对象，并将其包装成实际运行该函数的装饰器，但是一旦完成，它会将函数重新安排为再次执行。这样，它将一直运行，直到调度程序停止，并且每当它完成时，它都会再次安排。

# 在进程之间共享数据

在使用线程或协程时，数据是通过它们共享相同的内存空间而共享的。因此，只要注意避免竞争条件并提供适当的锁定，您就可以从任何线程访问任何对象。

相反，使用进程时，情况变得更加复杂，数据不会在它们之间共享。因此，在使用`ProcessPool`或`ProcessPoolExecutor`时，我们需要找到一种方法来在进程之间传递数据，并使它们能够共享一个公共状态。

Python标准库提供了许多工具来创建进程之间的通信渠道：`multiprocessing.Queues`，`multiprocessing.Pipe`，`multiprocessing.Value`和`multiprocessing.Array`可用于创建一个进程可以提供并且另一个进程可以消费的队列，或者在共享内存中共享的多个进程之间的值。

虽然所有这些都是可行的解决方案，但它们有一些限制：您必须在创建任何进程之前创建所有共享值，因此如果共享值的数量是可变的并且在存储类型方面受到限制，则它们就不可行。

相反，`multiprocessing.Manager`允许我们通过共享的`Namespace`存储任意数量的共享值。

# 如何做到...

以下是此配方的步骤：

1.  `管理器`应该在应用程序开始时创建，然后所有进程都能够从中设置和读取值：

```py
import multiprocessing

manager = multiprocessing.Manager()
namespace = manager.Namespace()
```

1.  一旦我们有了我们的`namespace`，任何进程都能够向其设置值：

```py
def set_first_variable():
    namespace.first = 42
p = multiprocessing.Process(target=set_first_variable)
p.start()
p.join()

def set_second_variable():
    namespace.second = dict(value=42)
p = multiprocessing.Process(target=set_second_variable)
p.start()
p.join()

import datetime
def set_custom_variable():
    namespace.last = datetime.datetime.utcnow()
p = multiprocessing.Process(target=set_custom_variable)
p.start()
p.join()
```

1.  任何进程都能够访问它们：

```py
>>> def print_variables():
...    print(namespace.first, namespace.second, namespace.last)
...
>>> p = multiprocessing.Process(target=print_variables)
>>> p.start()
>>> p.join()
42 {'value': 42} 2018-05-26 21:39:17.433112
```

无需提前创建变量或从主进程创建，只要进程能够访问`Namespace`，所有进程都能够读取或设置任何变量。

# 它是如何工作的...

`multiprocessing.Manager`类充当服务器，能够存储任何进程都能够访问的值，只要它具有对`Manager`和它想要访问的值的引用。

通过知道它正在侦听的套接字或管道的地址，可以访问`Manager`本身，每个具有对`Manager`实例的引用的进程都知道这些：

```py
>>> manager = multiprocessing.Manager()
>>> print(manager.address)
/tmp/pymp-4l33rgjq/listener-34vkfba3
```

然后，一旦您知道如何联系管理器本身，您需要能够告诉管理器要访问的对象。

可以通过拥有代表并确定该对象的`Token`来完成：

```py
>>> namespace = manager.Namespace()
>>> print(namespace._token)
Token(typeid='Namespace', 
      address='/tmp/pymp-092482xr/listener-yreenkqo', 
      id='7f78c7fd9630')
```

特别地，`Namespace`是一种允许我们在其中存储任何变量的对象。因此，通过仅使用`namespace`令牌就可以访问`Namespace`中存储的任何内容。

所有进程，因为它们是从同一个原始进程复制出来的，都具有`namespace`的令牌和管理器的地址，因此能够访问`namespace`，并因此设置或读取其中的值。

# 还有更多...

`multiprocessing.Manager` 不受限于与源自同一进程的进程一起工作。

可以创建一个在网络上监听的`Manager`，以便任何能够连接到它的进程可能能够访问其内容：

```py
>>> import multiprocessing.managers
>>> manager = multiprocessing.managers.SyncManager(
...     address=('localhost', 50000), 
...     authkey=b'secret'
... )
>>> print(manager.address)
('localhost', 50000)
```

然后，一旦服务器启动：

```py
>>> manager.get_server().serve_forever()
```

其他进程将能够通过使用与他们想要连接的管理器完全相同的参数创建一个`manager2`实例，然后显式连接：

```py
>>> manager2 = multiprocessing.managers.SyncManager(
...     address=('localhost', 50000), 
...     authkey=b'secret'
... )
>>> manager2.connect()
```

让我们在管理器中创建一个`namespace`并将一个值设置到其中：

```py
>>> namespace = manager.Namespace()
>>> namespace.value = 5
```

知道`namespace`的令牌值后，可以创建一个代理对象通过网络从`manager2`访问`namespace`：

```py
>>> from multiprocessing.managers import NamespaceProxy
>>> ns2 = NamespaceProxy(token, 'pickle', 
...                      manager=manager2, 
...                      authkey=b'secret')
>>> print(ns2.value)
5
```
