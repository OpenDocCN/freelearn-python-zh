# 并发

并发是让计算机同时做（或看起来同时做）多件事情的艺术。从历史上看，这意味着邀请处理器每秒多次在多个任务之间切换。在现代系统中，这也可以字面意义上意味着在单独的处理器核心上同时做两件或多件事。

并发本身并不是一个面向对象的主题，但 Python 的并发系统提供了面向对象的接口，正如我们在整本书中提到的。本章将介绍以下主题：

+   线程

+   多进程

+   Futures

+   AsyncIO

并发很复杂。基本概念相当简单，但可能出现的错误却难以追踪。然而，对于许多项目来说，并发是获得所需性能的唯一途径。想象一下，如果网络服务器不能在另一个用户完成之前响应用户的请求会怎样！我们不会深入探讨这有多么困难（需要另一本完整的书），但我们将看到如何在 Python 中实现基本的并发，以及一些常见的陷阱要避免。

# 线程

通常，并发是为了在程序等待 I/O 操作发生时继续执行工作。例如，服务器可以在等待前一个请求的数据到达时开始处理新的网络请求。或者，一个交互式程序可能在等待用户按下一个键时渲染动画或执行计算。记住，虽然一个人每分钟可以输入超过 500 个字符，但计算机每秒可以执行数十亿条指令。因此，在快速输入时，即使在单个按键之间，也可能发生大量的处理。

从理论上讲，你可以在程序内部管理所有这些活动之间的切换，但这几乎是不可能的。相反，我们可以依赖 Python 和操作系统来处理复杂的切换部分，同时我们创建看起来似乎是独立运行但实际上是同时运行的对象。这些对象被称为**线程**。让我们看看一个基本的例子：

```py
from threading import Thread

class InputReader(Thread):
    def run(self):
        self.line_of_text = input()

print("Enter some text and press enter: ")
thread = InputReader()
thread.start()

count = result = 1
while thread.is_alive():
    result = count * count
    count += 1

print("calculated squares up to {0} * {0} = {1}".format(count, result))
print("while you typed '{}'".format(thread.line_of_text))
```

这个例子运行了两个线程。你能看到它们吗？每个程序都有（至少）一个线程，称为主线程。从启动执行的代码就在这个线程中。更明显的第二个线程是`InputReader`类。

要构建一个线程，我们必须扩展`Thread`类并实现`run`方法。任何由`run`方法执行的代码都在一个单独的线程中执行。

新的线程在我们在对象上调用`start()`方法之前不会开始运行。在这种情况下，线程立即暂停以等待键盘输入。与此同时，原始线程从`start`被调用的地方继续执行。它开始在一个`while`循环中计算平方。`while`循环中的条件检查`InputReader`线程是否已经从其`run`方法退出；一旦它退出，它就会在屏幕上输出一些总结信息。

如果我们运行这个例子并输入字符串`hello world`，输出看起来如下：

```py
Enter some text and press enter:
hello world
calculated squares up to 2448265 * 2448265 = 5993996613696
```

当然，在输入字符串时，你会计算更多或更少的平方，因为数字与我们的相对打字速度以及我们运行的计算机的处理器速度都有关。当我在第一版和第三版之间更新这个例子时，我的新系统能够计算比之前多两倍的平方。

线程只有在调用`start`方法时才会以并发模式开始运行。如果我们想取消并发调用以比较其效果，我们可以在原始调用`thread.start()`的地方调用`thread.run()`。正如这里所示，输出是说明性的：

```py
    Enter some text and press enter:
    hello world
    calculated squares up to 1 * 1 = 1
    while you typed 'hello world'  
```

在这种情况下，没有第二个线程，`while` 循环从未执行。当我们输入时，我们浪费了很多 CPU 资源处于空闲状态。

使用线程有效有很多不同的模式。我们不会涵盖所有这些模式，但我们会查看一个常见的模式，这样我们就可以了解`join`方法。让我们检查加拿大每个省和地区的首府的当前温度：

```py
from threading import Thread
import time
from urllib.request import urlopen
from xml.etree import ElementTree

CITIES = {
    "Charlottetown": ("PE", "s0000583"),
    "Edmonton": ("AB", "s0000045"),
    "Fredericton": ("NB", "s0000250"),
    "Halifax": ("NS", "s0000318"),
    "Iqaluit": ("NU", "s0000394"),
    "Québec City": ("QC", "s0000620"),
    "Regina": ("SK", "s0000788"),
    "St. John's": ("NL", "s0000280"),
    "Toronto": ("ON", "s0000458"),
    "Victoria": ("BC", "s0000775"),
    "Whitehorse": ("YT", "s0000825"),
    "Winnipeg": ("MB", "s0000193"),
    "Yellowknife": ("NT", "s0000366"),
}

class TempGetter(Thread):
    def __init__(self, city):
        super().__init__()
        self.city = city
        self.province, self.code = CITIES[self.city]

    def run(self):
        url = (
            "http://dd.weatheroffice.ec.gc.ca/citypage_weather/xml/"
            f"{self.province}/{self.code}_e.xml"
        )
        with urlopen(url) as stream:
            xml = ElementTree.parse(stream)
            self.temperature = xml.find(
                "currentConditions/temperature"
            ).text

threads = [TempGetter(c) for c in CITIES]
start = time.time()
for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

for thread in threads:
    print(f"it is {thread.temperature}°C in {thread.city}")
print(
    "Got {} temps in {} seconds".format(
        len(threads), time.time() - start
    )
)
```

这段代码在启动线程之前构建了 10 个线程。注意我们如何覆盖构造函数将它们传递给`Thread`对象，同时记得调用`super`以确保`Thread`被正确初始化。

我们在一个线程中构建的数据可以从其他正在运行的线程中访问。`run` 方法中全局变量的引用说明了这一点。不那么明显的是，传递给构造函数的数据正在`主线程`中分配给`self`，但在第二个线程中被访问。这可能会让人困惑；仅仅因为一个方法在`Thread`实例上，并不意味着它会在那个线程中神奇地执行。

在启动 10 个线程之后，我们再次遍历它们，对每个线程调用`join()`方法。这个方法表示*等待线程完成后再做任何事情*。我们依次调用十次；这个`for`循环不会退出，直到所有十个线程都完成。

在这个阶段，我们可以打印出存储在每个线程对象上的温度。请注意，再次强调，我们可以从主线程访问在线程中构建的数据。在线程中，所有状态默认是共享的。

在我的 100 兆比特连接上执行前面的代码大约需要三分之一的秒，我们得到以下输出：

```py
it is 18.5°C in Charlottetown
it is 1.6°C in Edmonton
it is 16.6°C in Fredericton
it is 18.0°C in Halifax
it is -2.4°C in Iqaluit
it is 18.4°C in Québec City
it is 7.4°C in Regina
it is 11.8°C in St. John's
it is 20.4°C in Toronto
it is 9.2°C in Victoria
it is -5.1°C in Whitehorse
it is 5.1°C in Winnipeg
it is 1.6°C in Yellowknife
Got 13 temps in 0.29401135444641113 seconds
```

我现在是在九月写作，但北方已经低于冰点！如果我用单个线程（通过将`start()`调用改为`run()`并注释掉`join()`循环）运行这段代码，它需要接近四秒钟的时间，因为每个 0.3 秒的请求必须完成，下一个请求才能开始。这种数量级的速度提升仅仅展示了并发编程有多么有用。

# 线程的许多问题

线程可能很有用，尤其是在其他编程语言中，但现代的 Python 程序员倾向于避免使用它们，有几个原因。正如我们将看到的，还有其他方法可以编写并发编程，这些方法正在得到 Python 社区的更多关注。在我们继续之前，让我们讨论一些这些陷阱。

# 共享内存

线程的主要问题也是它们的最大优势。线程可以访问程序的所有内存和所有变量。这很容易导致程序状态的不一致性。

你是否遇到过这样一个房间，一个灯有两个开关，两个人同时打开它们？每个人（线程）都期望他们的动作会将灯（一个变量）打开，但结果是灯是关的，这与他们的期望不一致。现在想象如果这两个线程是在银行账户之间转账或管理车辆的巡航控制。

在线程编程中，解决这个问题的方法是对任何读取或（尤其是）写入共享变量的代码进行同步。有几种不同的方法可以做到这一点，但我们不会在这里深入讨论，以便我们可以专注于更 Pythonic 的结构。

同步解决方案是可行的，但很容易忘记应用它。更糟糕的是，由于不当使用同步而导致的错误很难追踪，因为线程执行操作的顺序不一致。我们无法轻易地重现错误。通常，最安全的方法是强制线程通过使用已经适当使用锁的轻量级数据结构进行通信。Python 提供了 `queue.Queue` 类来完成这项任务；其功能基本上与 `multiprocessing.Queue` 相同，我们将在下一节中讨论。

在某些情况下，这些缺点可能被允许共享内存的一个优点所抵消：它速度快。如果有多个线程需要访问一个巨大的数据结构，共享内存可以快速提供这种访问。然而，在 Python 中，由于两个在不同的 CPU 核心上运行的线程不可能同时执行计算，这个优势通常被抵消。这把我们带到了线程的第二个问题。

# 全局解释器锁

为了有效地管理内存、垃圾回收以及在本地库中对机器代码的调用，Python 有一个名为**全局解释器锁**或**GIL**的实用工具。它无法关闭，这意味着在 Python 中，线程对于其他语言中它们擅长的某一方面（并行处理）是无用的。对于我们的目的而言，GIL 的主要作用是防止任何两个线程在确切相同的时间进行工作，即使它们有工作要做。在这种情况下，“做工作”意味着使用 CPU，所以多个线程访问磁盘或网络是完全正常的；一旦线程开始等待某事，GIL 就会释放。这就是为什么天气示例可以工作。

GIL（全局解释器锁）受到了高度批评，主要是由那些不理解它是什么或者它给 Python 带来所有好处的人。如果我们的语言没有这种限制，那当然会很棒，但 Python 开发团队已经确定，它带来的价值大于其成本。它使得参考实现更容易维护和开发，而且在 Python 最初开发的单核处理器时代，它实际上使解释器运行得更快。然而，GIL 的净结果是限制了线程带来的好处，而没有减轻其成本。

虽然 GIL 是大多数人们使用的 Python 参考实现中的问题，但在一些非标准实现中，如 IronPython 和 Jython，这个问题已经被解决了。不幸的是，在出版时，这些实现中没有一个支持 Python 3。

# 线程开销

与我们稍后将要讨论的异步系统相比，线程的一个最终限制是维护每个线程的成本。每个线程都需要占用一定量的内存（在 Python 进程和操作系统内核中）来记录该线程的状态。在线程之间切换也会使用（少量）CPU 时间。这项工作在没有额外编码的情况下无缝发生（我们只需调用`start()`，其余的都会处理），但这项工作仍然需要发生。

这可以通过结构化我们的工作负载来在一定程度上缓解，使得线程可以被重用来执行多个任务。Python 提供了一个`ThreadPool`功能来处理这个问题。它作为多进程库的一部分提供，其行为与我们将要讨论的`ProcessPool`相同，所以让我们将这个讨论推迟到下一节。

# 多进程

多进程 API 最初是为了模仿线程 API 而设计的。然而，它已经发展，在 Python 3 的最新版本中，它更稳健地支持更多功能。多进程库是为了当需要并行执行 CPU 密集型任务且有多核可用时（几乎所有的计算机，甚至一个小巧的智能手表，都有多个核心）而设计的。当进程的大部分时间都在等待 I/O（例如，网络、磁盘、数据库或键盘）时，多进程并不有用，但对于并行计算来说，这是必经之路。

多进程模块会启动新的操作系统进程来完成工作。这意味着每个进程都运行着一个完全独立的 Python 解释器副本。让我们尝试使用与 `threading` API 提供的类似构造来并行化一个计算密集型操作，如下所示：

```py
from multiprocessing import Process, cpu_count
import time
import os

class MuchCPU(Process):
    def run(self):
        print(os.getpid())
        for i in range(200000000):
            pass

if __name__ == "__main__":
    procs = [MuchCPU() for f in range(cpu_count())]
    t = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("work took {} seconds".format(time.time() - t))
```

这个例子只是让 CPU 进行 2 亿次迭代。你可能不会认为这是有用的工作，但它可以在寒冷的日子里给你的笔记本电脑加热！

API 应该是熟悉的；我们实现了一个 `Process`（而不是 `Thread`）的子类，并实现了 `run` 方法。这个方法在执行一些激烈（如果方向错误）的工作之前，会打印出进程 ID（操作系统分配给机器上每个进程的唯一数字）。

请特别注意模块级别代码周围的 `if __name__ == '__main__':` 保护，这可以防止模块被导入时运行，而应该作为程序运行。这通常是一种良好的做法，但使用某些操作系统上的多进程时，这是至关重要的。幕后，多进程可能需要在新的进程中重新导入模块以执行 `run()` 方法。如果我们允许整个模块在那个时刻执行，它将开始递归地创建新进程，直到操作系统耗尽资源，导致你的电脑崩溃。

我们为机器上的每个处理器核心构建一个进程，然后启动并加入这些进程。在我的 2017 年代的 8 核 ThinkCenter 上，输出如下：

```py
25812
25813
25814
25815
25816
25817
25818
25819
work took 6.97506308555603 seconds
```

前四行是每个 `MuchCPU` 实例内部打印的进程 ID。最后一行显示在我的机器上，2 亿次迭代大约需要 13 秒。在这 13 秒内，我的进程监控显示我的四个核心都在以 100% 的速度运行。

如果我们在 `MuchCPU` 中使用 `threading.Thread` 而不是 `multiprocessing.Process` 作为子类，输出如下：

```py
26083
26083
26083
26083
26083
26083
26083
26083
work took 26.710845470428467 seconds
```

这次，四个线程在同一个进程中运行，运行时间超过三倍。这是 GIL 的代价；在其他语言中，线程版本至少会与多进程版本一样快。

我们可能期望它至少需要四倍的时间，但请记住，我的笔记本电脑上还运行着许多其他程序。在多进程版本中，这些程序也需要四个 CPU 中的一份。在多线程版本中，那些程序可以使用其他七个 CPU。

# 多进程池

通常，没有必要拥有比计算机上的处理器更多的进程。这有几个原因：

+   只有`cpu_count()`个进程可以同时运行

+   每个进程都消耗着 Python 解释器的完整副本的资源

+   进程间的通信代价高昂

+   创建进程需要一定的时间

考虑到这些限制，当程序启动时创建最多`cpu_count()`个进程，然后根据需要执行任务是有意义的。这比为每个任务启动一个新的进程要少得多开销。

实现这样一个基本的通信进程系列并不困难，但调试、测试和正确实现可能会很棘手。当然，其他 Python 开发者已经为我们以多进程池的形式实现了这一点。

池抽象化了确定主进程中执行什么代码以及子进程中运行什么代码的开销。池抽象限制了不同进程中的代码交互的地点，这使得跟踪变得更加容易。

与线程不同，多进程无法直接访问其他线程设置的变量。多进程提供了一些不同的方式来实现进程间通信。池无缝地隐藏了数据在进程间传递的过程。使用池看起来就像一个函数调用：你将数据传递给一个函数，它在另一个进程或多个进程中执行，当工作完成时，返回一个值。重要的是要理解，在底层，有很多工作正在进行以支持这一点：一个进程中的对象正在被序列化并通过操作系统进程管道传递。然后，另一个进程从管道中检索数据并反序列化它。请求的工作在子进程中完成，并产生一个结果。结果被序列化并通过管道返回。最终，原始进程反序列化并返回它。

所有这些序列化和将数据传递到管道中的操作都需要时间和内存。因此，将传递到池中和从池中返回的数据的数量和大小保持在最低限度是理想的，并且只有在需要对相关数据进行大量处理时，使用池才有利。

Pickling 对于即使是中等规模的 Python 操作来说也是一个昂贵的操作。将大对象序列化以在单独的进程中使用，通常比在原始进程中使用线程来完成工作要昂贵得多。确保您对程序进行性能分析，以确保多进程的开销实际上值得实现和维护的开销。

带着这些知识，让所有这些机械运转的代码出人意料地简单。让我们看看计算一组随机数的所有质因数的问题。这是各种密码学算法（更不用说对这些算法的攻击！）中常见且昂贵的部分。要破解用于保护您的银行账户的极其大的数字，需要多年的处理能力。以下实现虽然可读，但效率并不高，但这没关系，因为我们想看到它使用大量的 CPU 时间：

```py
import random
from multiprocessing.pool import Pool

def prime_factor(value):
    factors = []
    for divisor in range(2, value - 1):
        quotient, remainder = divmod(value, divisor)
        if not remainder:
            factors.extend(prime_factor(divisor))
            factors.extend(prime_factor(quotient))
            break
    else:
        factors = [value]
    return factors

if __name__ == "__main__":
 pool = Pool()

    to_factor = [random.randint(100000, 50000000) for i in range(20)]
 results = pool.map(prime_factor, to_factor)
    for value, factors in zip(to_factor, results):
        print("The factors of {} are {}".format(value, factors))
```

让我们专注于并行处理方面，因为计算因子的暴力递归算法已经很清晰了。我们首先构建一个多进程池实例。默认情况下，此池为运行在其上的机器中的每个 CPU 核心创建一个单独的进程。

`map`方法接受一个函数和一个可迭代对象。池将可迭代中的每个值序列化，并将其传递给一个可用的进程，该进程在它上面执行函数。当该进程完成其工作后，它将结果列表的因数序列化，并将其返回给池。然后，如果池有更多工作可用，它将承担下一项工作。

一旦所有池完成处理工作（这可能需要一些时间），结果列表就返回到原始进程，该进程一直在耐心地等待所有这些工作完成。

通常更有用使用类似的`map_async`方法，即使进程仍在工作，它也会立即返回。在这种情况下，结果变量将不会是一个值列表，而是一个承诺，稍后通过调用`results.get()`来返回一个值列表。这个承诺对象还具有`ready()`和`wait()`等方法，允许我们检查是否所有结果都已就绪。我将让您查阅 Python 文档以了解更多关于它们的使用方法。

或者，如果我们事先不知道我们想要获取结果的值的全部，我们可以使用`apply_async`方法来排队一个单独的任务。如果池有一个尚未工作的进程，它将立即开始；否则，它将保留任务，直到有可用的进程。

池也可以被`closed`，这拒绝接受任何进一步的任务，但会处理队列中当前的所有任务，或者`terminated`，这更进一步，拒绝启动队列中仍然存在的任何任务，尽管当前正在运行的任务仍然允许完成。

# 队列

如果我们需要对进程之间的通信有更多的控制，我们可以使用一个`Queue`。`Queue`数据结构对于从一个进程向一个或多个其他进程发送消息非常有用。任何可序列化的对象都可以发送到`Queue`中，但请记住，序列化可能是一个昂贵的操作，因此保持这样的对象小巧。为了说明队列，让我们构建一个小型搜索引擎，用于存储所有相关条目在内存中。

这不是构建基于文本的搜索引擎的最明智的方式，但我已经使用这种模式查询需要使用 CPU 密集型过程构建图表的数值数据，然后将其渲染给用户。

这个特定的搜索引擎并行扫描当前目录中的所有文件。为 CPU 上的每个核心构建一个进程。每个进程都被指示将一些文件加载到内存中。让我们看看执行加载和搜索的函数：

```py
def search(paths, query_q, results_q): 
    lines = [] 
    for path in paths: 
        lines.extend(l.strip() for l in path.open()) 

    query = query_q.get() 
    while query: 
        results_q.put([l for l in lines if query in l]) 
        query = query_q.get() 
```

记住，这个函数是在不同的进程（实际上，它是在`cpucount()`不同的进程中运行的）中运行的，与主线程不同。它传递一个`path.path`对象的列表，以及两个`multiprocessing.Queue`对象；一个用于传入查询，一个用于发送输出结果。这些队列自动将数据序列化到队列中，并通过管道传递到子进程。这两个队列在主进程中设置，并通过管道传递到子进程中的搜索函数。 

搜索代码在效率和功能方面都很愚蠢；它遍历存储在内存中的每一行，并将匹配的行放入列表中。这个列表被放入队列中，并返回给主进程。

让我们看看`main`进程，它设置了这些队列：

```py
if __name__ == '__main__': 
    from multiprocessing import Process, Queue, cpu_count 
    from path import path 
    cpus = cpu_count() 
    pathnames = [f for f in path('.').listdir() if f.isfile()] 
    paths = [pathnames[i::cpus] for i in range(cpus)] 
    query_queues = [Queue() for p in range(cpus)] 
    results_queue = Queue() 

    search_procs = [ 
        Process(target=search, args=(p, q, results_queue)) 
        for p, q in zip(paths, query_queues) 
    ] 
    for proc in search_procs: proc.start() 
```

为了更容易描述，让我们假设`cpu_count`是四。注意`import`语句是如何放在`if`保护中的？这是一个小的优化，可以防止在某些操作系统上在每个子进程中（它们不需要）导入它们。我们列出当前目录中的所有路径，然后将列表分成四个大致相等的部分。我们还构建了一个包含四个`Queue`对象的列表，用于将数据发送到每个子进程。最后，我们构建了一个**单个**的结果队列。这个队列被传递到所有四个子进程中。每个子进程都可以将数据放入队列中，它将在主进程中汇总。

现在让我们看看使搜索真正发生的代码：

```py
    for q in query_queues:
        q.put("def")
        q.put(None) # Signal process termination

    for i in range(cpus):
        for match in results_queue.get():
            print(match)
    for proc in search_procs:
        proc.join()
```

这段代码执行单个搜索，搜索`"def"`（因为它是充满 Python 文件的目录中的常见短语！）。

这种队列的使用实际上是可能成为分布式系统的本地版本。想象一下，如果搜索被发送到多台计算机，然后重新组合。现在想象一下，如果你可以访问谷歌数据中心数百万台计算机，你可能会理解为什么他们可以如此快速地返回搜索结果！

我们在这里不会讨论它，但多进程模块包含一个管理类，可以消除前面代码中的许多样板代码。甚至有一个版本的`multiprocessing.Manager`可以管理远程系统上的子进程，以构建一个基本的分布式应用程序。如果你对此感兴趣，请查看 Python 多进程文档。

# 多进程的问题

与线程一样，多进程也存在问题，其中一些我们已经讨论过。没有一种做并发的最佳方式；这在 Python 中尤其如此。我们总是需要检查并行问题，以确定众多可用解决方案中哪一个最适合该问题。有时，可能没有最佳解决方案。

在多进程的情况下，主要缺点是进程间共享数据成本高昂。正如我们讨论的，所有进程间的通信，无论是通过队列、管道还是更隐式的机制，都需要序列化对象。过度的序列化很快就会主导处理时间。多进程在需要将相对较小的对象在进程间传递，并且每个对象都需要完成大量工作时效果最好。另一方面，如果进程间不需要通信，可能根本不需要使用该模块；我们可以启动四个独立的 Python 进程（例如，通过在单独的终端中运行每个进程）并独立使用它们。

多进程的另一个主要问题是，与线程一样，很难确定变量或方法是在哪个进程中访问的。在多进程中，如果你从一个进程访问变量，它通常会覆盖当前运行进程中的变量，而另一个进程则保留旧值。这真的很令人困惑，因此不要这样做。

# Futures

让我们开始探讨一种更异步的并发实现方式。根据我们需要哪种类型的并发（倾向于 I/O 还是倾向于 CPU），Futures 会包装多进程或线程。它们并不能完全解决意外改变共享状态的问题，但它们允许我们以这样的方式组织代码，使得在发生这种情况时更容易追踪。

Futures 在不同的线程或进程之间提供了明确的边界。类似于多进程池，它们对于*调用和响应*类型的交互很有用，在这种交互中，处理可以在另一个线程中进行，然后在某个时刻（毕竟，它们的名字很合适），你可以请求结果。它实际上只是多进程池和线程池的包装，但它提供了一个更干净的 API，并鼓励编写更好的代码。

未来是一个封装函数调用的对象。该函数调用在*后台*运行，在一个线程或进程中。`future`对象有主线程可以使用的方法来检查未来是否完成，并在完成后获取结果。

让我们看看另一个文件搜索的例子。在上一个部分，我们实现了一个`unix grep`命令的版本。这次，我们将创建一个简单的`find`命令版本。该示例将搜索整个文件系统，查找包含给定字符序列的路径，如下所示：

```py
from concurrent.futures import ThreadPoolExecutor 
from pathlib import Path 
from os.path import sep as pathsep 
from collections import deque 

def find_files(path, query_string): 
    subdirs = [] 
    for p in path.iterdir(): 
        full_path = str(p.absolute()) 
        if p.is_dir() and not p.is_symlink(): 
            subdirs.append(p) 
        if query_string in full_path: 
                print(full_path) 

    return subdirs 

query = '.py' 
futures = deque() 
basedir = Path(pathsep).absolute() 

with ThreadPoolExecutor(max_workers=10) as executor: 
    futures.append( 
        executor.submit(find_files, basedir, query)) 
    while futures: 
        future = futures.popleft() 
        if future.exception(): 
            continue 
        elif future.done(): 
            subdirs = future.result() 
            for subdir in subdirs: 
                futures.append(executor.submit( 
                    find_files, subdir, query)) 
        else: 
            futures.append(future) 
```

这段代码由一个名为 `find_files` 的函数组成，它在单独的线程（或如果我们使用 `ProcessPoolExecutor` 代替，则是进程）中运行。这个函数没有什么特别之处，但请注意它没有访问任何全局变量。所有与外部环境的交互都传递到函数中或从函数返回。这不是一个技术要求，但这是在使用未来编程时保持你的大脑在头骨内的最佳方式。

在没有适当同步的情况下访问外部变量会导致称为 **竞争条件** 的情况。例如，想象有两个并发写入尝试增加一个整数计数器。它们同时开始，并且都读取值为 5。然后，它们都增加值并将结果写回为 6。但如果两个进程都在尝试增加一个变量，预期的结果应该是它增加两次，所以结果应该是 7。现代的智慧是，避免这样做最简单的方法是尽可能多地保持状态私有，并通过已知安全的结构，如队列或未来，来共享。

在开始之前，我们设置了一些变量；在这个例子中，我们将搜索包含字符 `'.py'` 的所有文件。我们有一个未来队列，我们稍后会讨论。`basedir` 变量指向文件系统的根目录：在 Unix 机器上是 `'/'`，在 Windows 机器上可能是 `C:\`。

首先，让我们简要地学习一下搜索理论。这个算法实现了并行广度优先搜索。而不是递归地使用深度优先搜索搜索每个目录，它将当前文件夹中的所有子目录添加到队列中，然后是每个这些文件夹的子目录，依此类推。

程序的核心部分被称为事件循环。我们可以将 `ThreadPoolExecutor` 作为上下文管理器来构建，这样它就会在完成后自动清理并关闭其线程。它需要一个 `max_workers` 参数来指示同时运行的线程数。如果有超过这个数量的工作提交，它将剩余的工作排队，直到有工作线程可用。当使用 `ProcessPoolExecutor` 时，这通常限制在机器上的 CPU 数量，但使用线程时，它可以高得多，这取决于一次有多少线程在等待 I/O。每个线程占用一定量的内存，所以它不应该太高。在磁盘速度而不是并行请求数量成为瓶颈之前，不需要太多线程。

一旦构建了执行器，我们就会使用根目录向其提交一个任务。`submit()` 方法立即返回一个 `Future` 对象，它承诺最终会给我们一个结果。这个未来对象被放入队列中。然后循环会反复从队列中移除第一个未来对象并检查它。如果它仍在运行，它会被添加回队列的末尾。否则，我们会通过调用 `future.exception()` 来检查函数是否抛出了异常。如果抛出了异常，我们只需忽略它（通常是一个权限错误，尽管真正的应用程序需要更小心地处理异常）。如果我们没有在这里检查这个异常，它会在我们调用 `result()` 时抛出，并且可以通过正常的 `try...except` 机制来处理。

假设没有发生异常，我们可以调用 `result()` 来获取返回值。由于函数返回一个不包含符号链接的子目录列表（我防止无限循环的懒惰方式），`result()` 返回相同的内容。这些新的子目录被提交给执行器，产生的未来对象被扔到队列中，以便在后续迭代中搜索其内容。

这就是开发基于未来的 I/O 密集型应用程序所需的所有内容。在底层，它使用的是我们之前讨论过的相同的线程或进程 API，但它提供了一个更易于理解的接口，并使得查看并发运行函数之间的边界更容易（只是不要尝试从未来内部访问全局变量！）。

# AsyncIO

AsyncIO 是 Python 并发编程的当前最佳实践。它将我们讨论过的未来的概念和事件循环与 第九章 中讨论的协程结合在一起，*迭代器模式*。结果是尽可能优雅和易于理解，尽管这并不是说很多！

AsyncIO 可以用于几种不同的并发任务，但它专门设计用于网络 I/O。大多数网络应用程序，尤其是在服务器端，花费大量时间等待从网络传入的数据。这可以通过为每个客户端使用单独的线程来解决，但线程会消耗内存和其他资源。AsyncIO 使用协程作为一种轻量级的线程。

该库提供自己的事件循环，消除了在前面示例中需要几行长的 `while` 循环的需求。然而，事件循环也有代价。当我们在一个事件循环上运行 `async` 任务中的代码时，该代码**必须**立即返回，既不阻塞 I/O 也不阻塞长时间运行的计算。当我们自己编写代码时，这是一件小事，但这意味着任何在 I/O 上阻塞的标准库或第三方函数都必须有非阻塞版本。

AsyncIO 通过创建一组使用`async`和`await`语法来立即将控制权返回给事件循环的协程来解决此问题。这些关键字替换了我们之前在原始协程中使用的`yield`、`yield from`和`send`语法，以及手动前进到第一个*send*位置的需要。结果是并发代码，我们可以像处理顺序代码一样推理它。事件循环负责检查阻塞调用是否完成，并执行任何后续任务，就像我们在上一节中手动执行的那样。

# AsyncIO 的实际应用

阻塞函数的一个典型例子是`time.sleep`调用。让我们使用这个调用的异步版本来展示 AsyncIO 事件循环的基本原理，如下所示：

```py
import asyncio
import random

async def random_sleep(counter):
    delay = random.random() * 5
    print("{} sleeps for {:.2f} seconds".format(counter, delay))
    await asyncio.sleep(delay)
    print("{} awakens".format(counter))

async def five_sleepers():
    print("Creating five tasks")
    tasks = [asyncio.create_task(random_sleep(i)) for i in range(5)]
    print("Sleeping after starting five tasks")
    await asyncio.sleep(2)
    print("Waking and waiting for five tasks")
    await asyncio.gather(*tasks)

asyncio.get_event_loop().run_until_complete(five_sleepers())
print("Done five tasks")
```

这是一个相当基础的例子，但它涵盖了 AsyncIO 编程的几个特性。最容易理解的是它的执行顺序，这基本上是从下到上。

下面是如何执行脚本的一个示例：

```py
Creating five tasks
Sleeping after starting five tasks
0 sleeps for 3.42 seconds
1 sleeps for 4.16 seconds
2 sleeps for 0.75 seconds
3 sleeps for 3.55 seconds
4 sleeps for 0.39 seconds
4 awakens
2 awakens
Waking and waiting for five tasks
0 awakens
3 awakens
1 awakens
Done five tasks
```

最后一行获取事件循环并指示它运行一个任务，直到它完成。相关的任务是`five_sleepers`。一旦该任务完成其工作，循环将退出，我们的代码将终止。作为异步程序员，我们不需要了解太多关于`run_until_complete`调用内部发生的事情，但请注意，有很多事情在进行中。这是一个增强版的协程版本，我们之前在章节中编写的未来循环，它知道如何处理迭代、异常、函数返回、并行调用等等。

在这个上下文中，任务是一个`asyncio`知道如何在事件循环上安排的对象。这包括以下内容：

+   使用`async`和`await`语法定义的协程。

+   使用`@asyncio.coroutine`装饰器和`yield from`语法（这是一个较旧的模型，已被`async`和`await`所取代）装饰的协程。

+   `asyncio.Future`对象。这些与我们在上一节中看到的`concurrent.futures`几乎相同，但用于`asyncio`。

+   任何可等待的对象，即具有`__await__`函数的对象。

在这个例子中，所有任务都是协程；我们将在后面的例子中看到一些其他的例子。

仔细观察一下那个`five_sleepers`未来。协程首先构建了五个`random_sleep`协程的实例。这些实例每个都被`asyncio.create_task`调用所包装，这会将未来添加到循环的任务队列中，以便它们可以在控制权返回到循环时立即执行。

每当我们调用`await`时，控制权都会返回。在这种情况下，我们调用`await asyncio.sleep`来暂停协程的执行两秒钟。在暂停期间，事件循环执行它已经排队的任务：即五个`random_sleep`任务。

当`five_sleepers`任务中的睡眠调用唤醒时，它调用`asyncio.gather`。这个函数接受任务作为可变参数，并在返回之前等待每个任务（以及其他事情，以保持循环安全运行）。

每个`random_sleep`协程打印一条启动信息，然后使用自己的`await`调用将控制权返回给事件循环一段时间。当睡眠完成时，事件循环将控制权返回给相关的`random_sleep`任务，该任务在返回之前打印其唤醒信息。

注意，任何少于两秒即可完成的任务都会在原始的`five_sleepers`协程唤醒并运行到调用`gather`任务之前输出它们自己的唤醒信息。由于事件队列现在为空（所有六个协程都已运行完成且不再等待任何任务），`run_until_complete`调用能够终止，程序结束。

`async`关键字作为文档，通知 Python 解释器（和程序员）协程包含`await`调用。它还做一些工作来准备协程在事件循环上运行。它的工作方式很像装饰器；事实上，在 Python 3.4 中，这被实现为一个`@asyncio.coroutine`装饰器。

# 读取 AsyncIO 未来对象

AsyncIO 协程按顺序执行每一行，直到遇到`await`语句，此时，它将控制权返回给事件循环。事件循环随后执行任何其他准备就绪的任务，包括原始协程等待的任务。每当子任务完成时，事件循环将结果发送回协程，以便它可以继续执行，直到遇到另一个`await`语句或返回。

这允许我们编写同步执行的代码，直到我们明确需要等待某事。因此，没有线程的非确定性行为，所以我们不必那么担心共享状态。

仍然建议避免在协程内部访问共享状态。这会使你的代码更容易推理。更重要的是，尽管理想的世界可能所有异步执行都在协程内部进行，但现实是有些未来对象在后台线程或进程中执行。坚持“无共享”哲学以避免大量难以调试的错误。

此外，AsyncIO 允许我们将代码的逻辑部分组合在一个单独的协程中，即使我们在等待其他地方的工作。作为一个具体的例子，即使在`random_sleep`协程中的`await asyncio.sleep`调用允许在事件循环中发生大量事情，但协程本身看起来像是有序地执行所有操作。这种无需担心等待任务完成的机制即可读取相关异步代码的能力是 AsyncIO 模块的主要优势。

# AsyncIO 网络编程

AsyncIO 是专门为与网络套接字一起使用而设计的，所以让我们实现一个 DNS 服务器。更准确地说，让我们实现 DNS 服务器的一个非常基础的功能。

DNS 的基本目的是将域名，例如[`www.python.org/`](https://www.python.org/)，翻译成 IP 地址，例如 IPv4 地址（例如`23.253.135.79`）或 IPv6 地址（例如`2001:4802:7901:0:e60a:1375:0:6`）。它必须能够执行许多类型的查询，并且知道如何联系其他 DNS 服务器，如果它没有所需的答案。我们不会实现这些功能，但以下示例能够直接响应标准的 DNS 查询，查找几个网站的 IP 地址：

```py
import asyncio
from contextlib import suppress

ip_map = {
    b"facebook.com.": "173.252.120.6",
    b"yougov.com.": "213.52.133.246",
    b"wipo.int.": "193.5.93.80",
    b"dataquest.io.": "104.20.20.199",
}

def lookup_dns(data):
    domain = b""
    pointer, part_length = 13, data[12]
    while part_length:
        domain += data[pointer : pointer + part_length] + b"."
        pointer += part_length + 1
        part_length = data[pointer - 1]

    ip = ip_map.get(domain, "127.0.0.1")

    return domain, ip

def create_response(data, ip):
    ba = bytearray
    packet = ba(data[:2]) + ba([129, 128]) + data[4:6] * 2
    packet += ba(4) + data[12:]
    packet += ba([192, 12, 0, 1, 0, 1, 0, 0, 0, 60, 0, 4])
    for x in ip.split("."):
        packet.append(int(x))
    return packet

class DNSProtocol(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        print("Received request from {}".format(addr[0]))
        domain, ip = lookup_dns(data)
        print(
            "Sending IP {} for {} to {}".format(
                domain.decode(), ip, addr[0]
            )
        )
        self.transport.sendto(create_response(data, ip), addr)

loop = asyncio.get_event_loop()
transport, protocol = loop.run_until_complete(
    loop.create_datagram_endpoint(
        DNSProtocol, local_addr=("127.0.0.1", 4343)
    )
)
print("DNS Server running")

with suppress(KeyboardInterrupt):
    loop.run_forever()
transport.close()
loop.close()
```

这个示例设置了一个字典，将几个域名愚蠢地映射到 IPv4 地址。随后是两个函数，它们从二进制 DNS 查询数据包中提取信息并构建响应。我们不会讨论这些；如果你想了解更多关于 DNS 的信息，请阅读 RFC（*请求评论*，定义大多数 IP 的格式）`1034`和`1035`。

你可以通过在另一个终端运行以下命令来测试这个服务：

```py
    nslookup -port=4343 facebook.com localhost  
```

让我们继续深入主题。AsyncIO 网络围绕紧密相连的概念——传输和协议。协议是一个类，当发生相关事件时，会调用其特定的方法。由于 DNS 运行在**UDP**（用户数据报协议）之上，我们构建我们的协议类作为`DatagramProtocol`的子类。这个类可以响应各种事件。我们特别关注初始连接的发生（仅此而已，这样我们就可以存储传输以供将来使用）以及`datagram_received`事件。对于 DNS，每个接收到的数据报都必须被解析并响应，此时，交互就结束了。

因此，当接收到数据报时，我们处理数据包，查找 IP，并使用我们未讨论的函数（它们是这个家族中的黑羊）构建响应。然后，我们指示底层传输使用其`sendto`方法将生成的数据包发送回请求的客户机。

传输本质上代表了一种通信流。在这种情况下，它抽象掉了在事件循环上使用 UDP 套接字发送和接收数据的所有繁琐操作。例如，还有类似的传输用于与 TCP 套接字和子进程交互。

UDP 传输是通过调用循环的`create_datagram_endpoint`协程来构建的。这会构建适当的 UDP 套接字并开始监听它。我们传递给它套接字需要监听的地址，以及，重要的是，我们创建的协议类，这样传输就知道在接收到数据时应该调用什么。

由于初始化套接字的过程需要相当多的时间，并且会阻塞事件循环，因此`create_datagram_endpoint`函数是一个协程。在我们的示例中，在等待这个初始化的过程中我们不需要做任何事情，所以我们用`loop.run_until_complete`来包装这个调用。事件循环负责管理未来，当它完成时，它会返回一个包含两个值的元组：新初始化的传输和从我们传入的类中构建的协议对象。

在幕后，传输已经在事件循环上设置了一个任务，该任务正在监听传入的 UDP 连接。然后我们只需要通过调用`loop.run_forever()`来启动事件循环的运行，以便任务可以处理这些数据包。当数据包到达时，它们在协议中被处理，一切正常工作。

唯一需要特别注意的另一件大事是，当我们完成传输（以及确实，事件循环）时，应该关闭它们。在这种情况下，代码运行得很好，不需要两个`close()`调用，但如果我们是在实时构建传输（或者只是做适当的错误处理！），我们就需要对此有更多的意识。

你可能对设置协议类和底层传输所需的样板代码感到沮丧。AsyncIO 在这些两个关键概念之上提供了一个抽象，称为流。我们将在下一个示例中看到流的示例。

# 使用执行器包装阻塞代码

AsyncIO 提供了一个自己的 futures 库版本，允许我们在没有适当的非阻塞调用可进行时在单独的线程或进程中运行代码。这允许我们将线程和进程与异步模型结合起来。这个特性的一个更有用的应用是在应用程序有 I/O 密集型和 CPU 密集型活动爆发时，能够获得两者的最佳效果。I/O 密集型部分可以在事件循环中发生，而 CPU 密集型工作可以分配到不同的进程中。为了说明这一点，让我们使用 AsyncIO 来实现*排序作为一项服务*：

```py
import asyncio
import json
from concurrent.futures import ProcessPoolExecutor

def sort_in_process(data):
    nums = json.loads(data.decode())
    curr = 1
    while curr < len(nums):
        if nums[curr] >= nums[curr - 1]:
            curr += 1
        else:
            nums[curr], nums[curr - 1] = nums[curr - 1], nums[curr]
            if curr > 1:
                curr -= 1

    return json.dumps(nums).encode()

async def sort_request(reader, writer):
    print("Received connection")
    length = await reader.read(8)
    data = await reader.readexactly(int.from_bytes(length, "big"))
    result = await asyncio.get_event_loop().run_in_executor(
        None, sort_in_process, data
    )
    print("Sorted list")
    writer.write(result)
    writer.close()
    print("Connection closed")

loop = asyncio.get_event_loop()
loop.set_default_executor(ProcessPoolExecutor())
server = loop.run_until_complete(
    asyncio.start_server(sort_request, "127.0.0.1", 2015)
)
print("Sort Service running")

loop.run_forever()
server.close()
loop.run_until_complete(server.wait_closed())
loop.close()
```

这是一个实现了一些非常愚蠢想法的好代码的例子。将排序作为一项服务整个想法相当荒谬。使用我们自己的排序算法而不是调用 Python 的`sorted`甚至更糟糕。我们使用的算法被称为冒泡排序，或者在某些情况下，*愚蠢排序*。这是一个纯 Python 实现的慢速排序算法。我们定义了自己的协议而不是使用野外存在的许多完全合适的应用协议之一。甚至使用多进程来实现并行性的想法也可能值得怀疑；我们最终还是将所有数据传递到和从子进程中。有时，从你正在编写的程序中退一步，问问自己你是否在尝试达到正确的目标，这很重要。

但忽略工作负载，让我们看看这个设计的一些智能特性。首先，我们在子进程中传入和传出字节。这比在主进程中解码 JSON 智能得多。这意味着（相对昂贵的）解码可以在不同的 CPU 上发生。此外，序列化的 JSON 字符串通常比序列化的列表小，因此进程间传递的数据更少。

第二，这两个方法非常线性；看起来代码是一行一行执行的。当然，在 AsyncIO 中，这是一个错觉，但我们不必担心共享内存或并发原语。

# 流

现在的排序服务示例应该已经很熟悉了，因为它与其他 AsyncIO 程序有类似的模板。然而，也有一些不同之处。我们使用了 `start_server` 而不是 `create_server`。这种方法通过 AsyncIO 的流而不是使用底层的传输/协议代码进行挂钩。它允许我们传入一个普通的协程，该协程接收读取器和写入器参数。这两个参数都代表可以读取和写入的字节流，就像文件或套接字一样。其次，因为这是一个 TCP 服务器而不是 UDP，当程序结束时需要一些套接字清理。这个清理是一个阻塞调用，所以我们必须在事件循环上运行 `wait_closed` 协程。

流相对简单易懂。读取是一个可能阻塞的调用，所以我们必须使用 `await` 来调用它。写入不会阻塞；它只是将数据放入队列，然后 AsyncIO 在后台发送出去。

我们在 `sort_request` 方法内部的代码进行了两次读取请求。首先，它从线路上读取 8 个字节，并使用大端表示法将它们转换为整数。这个整数表示客户端打算发送的字节数。因此，在下一个调用 `readexactly` 时，它读取这么多字节。`read` 和 `readexactly` 之间的区别在于，前者将读取请求的字节数，而后者将缓冲读取，直到接收到所有字节，或者直到连接关闭。

# 执行器

现在让我们看看执行器代码。我们导入与上一节中使用的完全相同的 `ProcessPoolExecutor`。注意，我们不需要一个特殊的 AsyncIO 版本。事件循环有一个方便的 `run_in_executor` 协程，我们可以用它来运行未来。默认情况下，循环在 `ThreadPoolExecutor` 中运行代码，但如果我们愿意，我们可以传入不同的执行器。或者，就像在这个例子中那样，我们可以在设置事件循环时通过调用 `loop.set_default_executor()` 来设置不同的默认值。

如你所回忆的那样，使用执行器与 futures 一起使用时，没有太多的样板代码。然而，当我们与 AsyncIO 一起使用时，则完全没有！协程会自动将函数调用包装在 future 中，并将其提交给执行器。我们的代码在 future 完成之前会阻塞，而事件循环会继续处理其他连接、任务或 futures。当 future 完成时，协程会唤醒并继续将数据写回客户端。

你可能想知道，是否在事件循环内部运行多个进程，而不是运行多个事件循环在不同进程中会更好。答案是：*可能吧*。然而，根据具体的问题空间，我们可能更倾向于运行具有单个事件循环的独立程序副本，而不是试图通过主多进程来协调一切。

# AsyncIO 客户端

由于它能够处理数以千计的并发连接，AsyncIO 非常常见于实现服务器。然而，它是一个通用的网络库，并为客户端进程提供全面支持。这非常重要，因为许多微服务运行的服务器充当其他服务器的客户端。

客户端可以比服务器简单得多，因为它们不需要设置来等待传入的连接。像大多数网络库一样，你只需打开一个连接，提交你的请求，并处理任何响应。主要区别在于，每次你进行可能阻塞的调用时，都需要使用 `await`。以下是我们在上一个章节中实现的排序服务的一个示例客户端：

```py
import asyncio
import random
import json

async def remote_sort():
 reader, writer = await asyncio.open_connection("127.0.0.1", 2015)
    print("Generating random list...")
    numbers = [random.randrange(10000) for r in range(10000)]
    data = json.dumps(numbers).encode()
    print("List Generated, Sending data")
    writer.write(len(data).to_bytes(8, "big"))
    writer.write(data)

    print("Waiting for data...")
 data = await reader.readexactly(len(data))
    print("Received data")
    sorted_values = json.loads(data.decode())
    print(sorted_values)
    print("\n")
    writer.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(remote_sort())
loop.close()
```

在本节中，我们已经涵盖了 AsyncIO 的多数要点，本章也涵盖了其他许多并发原语。并发是一个难以解决的问题，没有一种解决方案适合所有用例。设计并发系统最重要的部分是决定哪种可用的工具是解决该问题的正确选择。我们已经看到了几个并发系统的优缺点，现在对哪些是不同类型需求更好的选择有一些了解。

# 案例研究

为了总结本章和本书，让我们构建一个基本的图像压缩工具。它将接受黑白图像（每个像素 1 位，开或关），并尝试使用一种称为运行长度编码的非常基础的压缩形式来压缩它。你可能觉得黑白图像有点牵强。如果是这样，你可能还没有在 [`xkcd.com`](http://xkcd.com) 上享受足够的时间！

我已经为本章的示例代码包含了一些黑白 BMP 图像（这些图像易于读取数据并提供了大量改进文件大小的机会）。

运行长度编码将位序列替换为任何重复位字符串的数量。例如，字符串 000011000 可能被替换为 04 12 03，以表示四个零后面跟着两个一，然后是三个更多的零。为了使事情更有趣，我们将每行分解为 127 位块。

我并不是随意选择 127 位。127 个不同的值可以编码到 7 位中，这意味着如果一个行包含全部为 1 或全部为 0，我们可以将其存储在一个字节中，第一个位表示它是一个 0 行还是一个 1 行，其余七个位表示该位存在多少。

将图像分解成块有另一个优点：我们可以并行处理单个块，而无需它们相互依赖。然而，也存在一个主要的缺点：如果一个运行只有少数一或零，那么它在压缩文件中会占用`更多`的空间。当我们将长运行分解成块时，我们可能会创建更多的这些小运行，并使文件的大小膨胀。

我们有权根据需要设计压缩文件中字节的布局。在这个简单的例子中，我们的压缩文件将在文件开头存储两个字节的小端整数，代表完成文件的宽度和高度。然后，它将写入代表每行 127 位块的字节。

在我们开始设计一个并发系统来构建这样的压缩图像之前，我们应该问一个基本的问题：这个应用程序是 I/O 密集型还是 CPU 密集型？

实话实说，我的答案是*我不知道*。我不确定应用程序是否会花费更多的时间从磁盘加载数据并将其写回，还是在内存中进行压缩。我怀疑它本质上是一个 CPU 密集型应用程序，但一旦我们开始将图像字符串传递到子进程中，我们可能会失去任何并行化的好处。

我们将使用自下而上的设计来构建这个应用程序。这样，我们将有一些构建块可以组合成不同的并发模式，以比较它们。让我们从使用运行长度编码压缩 127 位块的代码开始：

```py
from bitarray import bitarray 
def compress_chunk(chunk): 
    compressed = bytearray() 
    count = 1 
    last = chunk[0] 
    for bit in chunk[1:]: 
        if bit != last: 
            compressed.append(count | (128 * last)) 
            count = 0 
            last = bit 
        count += 1 
    compressed.append(count | (128 * last)) 
    return compressed
```

这段代码使用`bitarray`类来操作单个 0 和 1。它作为一个第三方模块分发，你可以使用`pip install bitarray`命令进行安装。传递给`compress_chunks`的块是这个类的一个实例（尽管示例也可以用布尔值列表工作）。在这种情况下，`bitarray`的主要好处是，在进程间序列化时，它们占用的空间是布尔值列表或 1s 和 0s 的字节字符串的八分之一。因此，它们序列化得更快。它们也比进行大量位操作要容易一些。

该方法使用运行长度编码压缩数据，并返回包含打包数据的`bytearray`。就像`bitarray`是一个一和零的列表一样，`bytearray`就像是一个字节对象的列表（当然，每个字节当然包含八个一或零）。

执行压缩的算法相当简单（虽然我想指出，实现和调试它花了我两天时间——易于理解并不一定意味着容易编写！）。它首先将`last`变量设置为当前运行中位的类型（要么是`True`要么是`False`）。然后，它遍历位，逐个计数，直到找到一个不同的位。当找到时，它通过将字节的最左边位（128 位位置）设置为`last`变量包含的零或一，来构建一个新的字节。然后，它重置计数器并重复操作。一旦循环完成，它为最后一个运行创建一个最后的字节并返回结果。

当我们创建构建块时，让我们创建一个函数来压缩一行图像数据，如下所示：

```py
def compress_row(row): 
    compressed = bytearray() 
    chunks = split_bits(row, 127) 
    for chunk in chunks: 
        compressed.extend(compress_chunk(chunk)) 
    return compressed
```

此函数接受一个名为`row`的`bitarray`。它使用我们将很快定义的函数将其分割成每个宽度为 127 位的块。然后，它使用先前定义的`compress_chunk`压缩这些块，将结果连接到`bytearray`中，并返回它。

我们将`split_bits`定义为生成器，如下所示：

```py
def split_bits(bits, width): 
    for i in range(0, len(bits), width): 
        yield bits[i:i+width]
```

现在，由于我们还不确定这将在线程或进程中运行得更有效，让我们将这些函数包装在一个方法中，该方法在提供的执行器中运行一切：

```py
def compress_in_executor(executor, bits, width): 
    row_compressors = [] 
    for row in split_bits(bits, width): 
        compressor = executor.submit(compress_row, row) 
        row_compressors.append(compressor) 

    compressed = bytearray() 
    for compressor in row_compressors: 
        compressed.extend(compressor.result()) 
    return compressed
```

这个例子几乎不需要解释；它使用我们已定义的相同的`split_bits`函数将传入的位分割成基于图像宽度的行（为自下而上的设计欢呼！）。

注意，此代码可以压缩任何位序列，尽管它会使具有频繁位值变化的二进制数据膨胀，而不是压缩。黑白图像无疑是所讨论压缩算法的良好候选者。现在，让我们创建一个函数，使用第三方 pillow 模块加载图像文件，将其转换为位，并压缩它。我们可以通过使用古老的注释语句轻松地在执行器之间切换，如下所示：

```py
from PIL import Image 
def compress_image(in_filename, out_filename, executor=None): 
    executor = executor if executor else ProcessPoolExecutor() 
    with Image.open(in_filename) as image: 
        bits = bitarray(image.convert('1').getdata()) 
        width, height = image.size 

    compressed = compress_in_executor(executor, bits, width) 

    with open(out_filename, 'wb') as file: 
        file.write(width.to_bytes(2, 'little')) 
        file.write(height.to_bytes(2, 'little')) 
        file.write(compressed) 

def single_image_main(): 
    in_filename, out_filename = sys.argv[1:3] 
    #executor = ThreadPoolExecutor(4) 
    executor = ProcessPoolExecutor() 
    compress_image(in_filename, out_filename, executor) 
```

`image.convert()`调用将图像转换为黑白（一位）模式，而`getdata()`返回这些值的迭代器。我们将结果打包到`bitarray`中，以便它们可以更快地通过电线传输。当我们输出压缩文件时，我们首先写入图像的宽度和高度，然后是压缩数据，它作为`bytearray`到达，可以直接写入二进制文件。

编写完所有这些代码后，我们终于能够测试线程池或进程池是否提供了更好的性能。我创建了一个大（7200 x 5600 像素）的黑白图像，并通过这两个池运行它。`ProcessPool`在我的系统上处理图像大约需要 7.5 秒，而`ThreadPool`则始终需要大约 9 秒。因此，正如我们所怀疑的，在多个处理器上运行时，序列化和反序列化位和字节之间的成本几乎消耗了所有效率提升（尽管，查看我的 CPU 监视器，它确实完全利用了我机器上的所有四个核心）。

因此，看起来压缩单个图像最有效地是在单独的进程中完成的，但只是略微，因为我们需要在父进程和子进程之间传递大量数据。当进程间传递的数据量相当低时，多进程更有效。

因此，让我们扩展这个应用程序，以并行压缩目录中的所有位图。我们唯一需要传递给子进程的是文件名，因此与使用线程相比，我们应该获得速度上的提升。此外，为了有点疯狂，我们将使用现有的代码来压缩单个图像。这意味着我们将在每个子进程中运行`ProcessPoolExecutor`以创建更多的子进程，如下所示（我不建议在实际生活中这样做！）：

```py
from pathlib import Path 
def compress_dir(in_dir, out_dir): 
    if not out_dir.exists(): 
        out_dir.mkdir() 

    executor = ProcessPoolExecutor() 
    for file in ( 
            f for f in in_dir.iterdir() if f.suffix == '.bmp'): 
        out_file = (out_dir / file.name).with_suffix('.rle') 
        executor.submit( 
            compress_image, str(file), str(out_file)) 

def dir_images_main(): 
    in_dir, out_dir = (Path(p) for p in sys.argv[1:3]) 
    compress_dir(in_dir, out_dir)
```

此代码使用我们之前定义的`compress_image`函数，但为每个图像在单独的进程中运行它。它没有将执行器传递到函数中，因此`compress_image`在新的进程开始运行后创建`ProcessPoolExecutor`。

现在我们正在执行器内部运行执行器，我们可以使用四种线程和进程池的组合来压缩图像。它们各自有不同的时间特性，如下所示：

|  | **每图像进程池** | **每图像线程池** |
| --- | --- | --- |
| **每行进程池** | 42 秒 | 53 秒 |
| **每行线程池** | 34 秒 | 64 秒 |

如我们所预期，为每个图像使用线程，然后再为每行使用线程是最慢的配置，因为全局解释器锁（GIL）阻止我们在并行中进行任何工作。鉴于我们在使用单个图像时使用单独的进程稍微快一点，你可能会惊讶地看到，如果我们为每个图像在单独的进程中处理，使用`ThreadPool`功能处理行会更快。花点时间理解为什么会这样。

我的机器只有四个处理器核心。每个图像的每一行都在一个单独的池中处理，这意味着所有这些行都在争夺处理能力。当只有一个图像时，通过并行运行每一行，我们可以获得（非常适度的）加速。然而，当我们同时处理多个图像时，将所有这些行数据传递到和从子进程中的成本会积极地从每个其他图像中窃取处理时间。所以，如果我们可以在单独的处理器上处理每个图像，唯一需要序列化到子进程管道中的是几个文件名，我们就可以获得稳定的加速。

因此，我们看到不同的工作负载需要不同的并发范式。即使我们只是使用未来（futures），我们也必须就使用哪种执行器做出明智的决定。

还要注意，对于通常大小的图像，程序运行得足够快，以至于我们使用哪种并发结构实际上并不重要。事实上，即使我们根本不使用任何并发，我们可能也会得到几乎相同的使用体验。

此问题也可以直接使用线程和/或进程池模块来解决，尽管需要编写相当多的模板代码。你可能想知道是否 AsyncIO 在这里会有用。答案是：*可能没有*。大多数操作系统都没有很好的方法从文件系统中执行非阻塞读取，因此库最终仍然会将所有调用包装在未来的包装中。

为了完整性，以下是我在解压缩**运行长度编码**（**RLE**）图像时使用的代码，以确认算法是否正确工作（实际上，直到我修复了压缩和解压缩中的错误，我仍然不确定它是否完美——我应该使用测试驱动开发！）：

```py
from PIL import Image 
import sys 

def decompress(width, height, bytes): 
    image = Image.new('1', (width, height)) 

    col = 0 
    row = 0 
    for byte in bytes: 
        color = (byte & 128) >> 7 
        count = byte & ~128 
        for i in range(count): 
            image.putpixel((row, col), color) 
            row += 1 
        if not row % width: 
            col += 1 
            row = 0 
    return image 

with open(sys.argv[1], 'rb') as file: 
    width = int.from_bytes(file.read(2), 'little') 
    height = int.from_bytes(file.read(2), 'little') 

    image = decompress(width, height, file.read()) 
    image.save(sys.argv[2], 'bmp') 
```

这段代码相当直接。每个运行被编码在一个单独的字节中。它使用一些位运算来提取像素的颜色和运行的长度。然后，它将图像中该运行中的每个像素设置好，并在适当的间隔增加下一个要分析的像素的行和列。

# 练习

我们在本章中介绍了几种不同的并发范式，但仍不清楚何时使用哪一个。正如我们在案例研究中看到的，在做出承诺之前原型化几个不同的策略通常是一个好主意。

Python 3 中的并发是一个巨大的主题，一本这么大的书也无法涵盖关于它的所有知识。作为你的第一个练习，我鼓励你上网搜索，了解被认为是最新的 Python 并发最佳实践。

如果你最近在应用程序中使用了线程，请查看代码，看看如何通过使用未来（futures）使其更易于阅读且更少出错。比较线程和进程池中的未来（futures），看看是否可以通过使用多个 CPU 获得任何好处。

尝试实现一个用于基本 HTTP 请求的 AsyncIO 服务。如果您能将其做到网页浏览器可以渲染简单 GET 请求的程度，您就会对 AsyncIO 网络传输和协议有一个很好的理解。

确保您理解在访问共享数据时线程中发生的竞态条件。尝试编写一个使用多个线程以这种方式设置共享值的程序，使得数据故意变得损坏或无效。

记得我们在第六章“Python 数据结构”中讨论的链接收集器吗？您能否通过并行请求使其运行更快？对于这个任务，使用原始线程、未来对象还是 AsyncIO 更好？

尝试直接使用线程或多进程编写运行长度编码的示例。您是否获得了速度提升？代码是否更容易或更难理解？有没有办法通过并发或并行化来加快解压缩脚本的执行速度？

# 摘要

本章以一个不太面向对象的题目结束了我们对面向对象编程的探索。并发是一个难题，我们只是触及了表面。虽然底层操作系统的进程和线程抽象并没有提供一个接近面向对象的 API，但 Python 围绕它们提供了一些真正优秀的面向对象抽象。线程和多进程包都提供了对底层机制的面向对象接口。未来对象能够将许多杂乱的细节封装成一个单一的对象。AsyncIO 使用协程对象使我们的代码看起来像同步运行，同时在非常简单的循环抽象后面隐藏了丑陋和复杂的实现细节。

感谢您阅读《Python 3 面向对象编程》，第三版。我希望您享受了这次旅程，并渴望在您未来的所有项目中开始实现面向对象软件！
