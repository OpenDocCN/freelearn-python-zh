# 第十三章：并发

并发及其表现之一——并行处理——是软件工程领域中最广泛的主题之一。本书中的大部分章节也涵盖了广泛的领域，几乎所有这些章节都可以成为一本独立的书的大主题。但并发这个主题本身是如此庞大，以至于它可能需要数十个职位，我们仍然无法讨论其所有重要方面和模型。

这就是为什么我不会试图愚弄你，并且从一开始就声明我们几乎不会深入讨论这个话题。本章的目的是展示为什么你的应用程序可能需要并发，何时使用它，以及你可以在 Python 中使用的最重要的并发模型：

+   多线程

+   多处理

+   异步编程

我们还将讨论一些语言特性、内置模块和第三方包，这些都可以让你在代码中实现这些模型。但我们不会详细讨论它们。把本章的内容当作你进一步研究和阅读的起点。它在这里是为了引导你了解基本的想法，并帮助你决定是否真的需要并发，以及哪种方法最适合你的需求。

# 为什么要并发？

在回答“为什么要并发”之前，我们需要问“并发到底是什么？”

对第二个问题的答案可能会让一些人感到惊讶，他们曾经认为这是**并行处理**的同义词。但并发不同于并行。并发不是应用程序实现的问题，而只是程序、算法或问题的属性。并行只是处理并发问题的可能方法之一。

1976 年，Leslie Lamport 在他的《分布式系统中的时间、时钟和事件排序》一文中说：

> *"如果两个事件互不影响，则它们是并发的。"*

通过将事件推广到程序、算法或问题，我们可以说如果某事物可以被完全或部分分解为无序的组件（单元），那么它就是并发的。这些单元可以相互独立地进行处理，处理的顺序不会影响最终结果。这意味着它们也可以同时或并行处理。如果我们以这种方式处理信息，那么我们确实在处理并行处理。但这并非强制性的。

以分布式方式进行工作，最好利用多核处理器或计算集群的能力，是并发问题的自然结果。但这并不意味着这是处理并发的唯一有效方式。有很多用例，可以以非同步的方式处理并发问题，但不需要并行执行。

因此，一旦我们知道了并发到底是什么，就是时候解释这到底是怎么回事了。当问题是并发的时候，它给了你处理它的机会，以一种特殊的、更有效的方式。

我们经常习惯用经典的方式处理问题，通过一系列步骤来解决问题。这是我们大多数人思考和处理信息的方式——使用同步算法逐步进行。但这种信息处理方式并不适合解决大规模问题或需要同时满足多个用户或软件代理的需求：

+   处理工作的时间受单个处理单元（单台机器、CPU 核心等）性能的限制

+   在程序完成处理前，无法接受和处理新的输入

因此，通常处理并发问题的最佳方法是同时处理：

+   问题的规模如此之大，以至于在可接受的时间范围内或在可用资源范围内处理它们的唯一方法是将执行分配给能够并行处理工作的多个处理单元。

+   你的应用程序需要保持响应性（接受新输入），即使它还没有完成处理旧的输入

这涵盖了大多数情况下并发处理是一个合理选择的情况。第一组问题明显需要并行处理解决方案，因此通常使用多线程和多处理模型来解决。第二组问题不一定需要并行处理，因此实际解决方案取决于问题的细节。请注意，这组问题还涵盖了应用程序需要独立为多个客户（用户或软件代理）提供服务，而无需等待其他成功服务的情况。

另一件值得一提的事情是，前面两组并不是互斥的。很多时候，你需要保持应用程序的响应性，同时又无法在单个处理单元上处理输入。这就是为什么在并发性方面，不同的看似替代或冲突的方法经常同时使用的原因。这在开发 Web 服务器时尤其常见，可能需要使用异步事件循环，或者线程与多个进程的结合，以利用所有可用资源并在高负载下保持低延迟。

# 多线程

线程通常被开发人员认为是一个复杂的话题。虽然这种说法完全正确，但 Python 提供了高级类和函数，简化了线程的使用。CPython 对线程的实现带来了一些不便的细节，使它们比其他语言中的线程更少用。它们对于一些你可能想要解决的问题仍然完全合适，但不像在 C 或 Java 中那样多。在本节中，我们将讨论 CPython 中多线程的限制，以及 Python 线程是可行解决方案的常见并发问题。

## 什么是多线程？

线程是执行的线程的缩写。程序员可以将他或她的工作分成同时运行并共享相同内存上下文的线程。除非你的代码依赖于第三方资源，多线程在单核处理器上不会加快速度，甚至会增加一些线程管理的开销。多线程将受益于多处理器或多核机器，并将在每个 CPU 核心上并行执行每个线程，从而使程序更快。请注意，这是一个通用规则，对大多数编程语言都应该成立。在 Python 中，多核 CPU 上的多线程性能收益有一些限制，但我们将在后面讨论。为简单起见，现在假设这个说法是正确的。

相同上下文被线程共享的事实意味着你必须保护数据免受并发访问。如果两个线程在没有任何保护的情况下更新相同的数据，就会发生竞争条件。这被称为**竞争危害**，因为每个线程运行的代码对数据状态做出了错误的假设，可能会导致意外的结果发生。

锁机制有助于保护数据，线程编程一直是确保资源以安全方式被线程访问的问题。这可能非常困难，线程编程经常会导致难以调试的错误，因为它们很难重现。最糟糕的问题发生在由于糟糕的代码设计，两个线程锁定一个资源并尝试获取另一个线程已锁定的资源。它们将永远等待对方。这被称为**死锁**，非常难以调试。**可重入锁**通过确保线程不会尝试两次锁定资源来在一定程度上帮助解决这个问题。

然而，当线程用于专门为它们构建的工具的孤立需求时，它们可能会提高程序的速度。

多线程通常在系统内核级别得到支持。当计算机只有一个处理器和一个核心时，系统使用**时间片**机制。在这里，CPU 从一个线程快速切换到另一个线程，以至于产生线程同时运行的错觉。这也是在处理级别上完成的。在没有多个处理单元的情况下，并行性显然是虚拟的，并且在这样的硬件上运行多个线程并不会带来性能提升。无论如何，有时即使必须在单个核心上执行代码，实现代码的多线程仍然是有用的，我们稍后将看到一个可能的用例。

当执行环境具有多个处理器或多个 CPU 核心时，一切都会发生变化。即使使用时间片，进程和线程也会分布在 CPU 之间，从而提供更快地运行程序的能力。

## Python 如何处理线程

与其他一些语言不同，Python 使用多个内核级别的线程，每个线程都可以运行解释器级别的任何线程。但是，语言的标准实现——CPython——存在重大限制，使得在许多情况下线程的可用性降低。所有访问 Python 对象的线程都由一个全局锁串行化。这是因为解释器的许多内部结构以及第三方 C 代码都不是线程安全的，需要受到保护。

这种机制称为**全局解释器锁**（**GIL**），其在 Python/C API 级别的实现细节已经在第七章的*释放 GIL*部分中讨论过，*其他语言中的 Python 扩展*。GIL 的移除是 python-dev 电子邮件列表上偶尔出现的一个话题，并且被开发人员多次提出。遗憾的是，直到现在，没有人成功提供一个合理简单的解决方案，使我们能够摆脱这个限制。高度不可能在这个领域看到任何进展。更安全的假设是 GIL 将永远存在于 CPython 中。因此，我们需要学会如何与之共存。

那么在 Python 中使用多线程有什么意义呢？

当线程只包含纯 Python 代码时，使用线程加速程序几乎没有意义，因为 GIL 会串行化它。但请记住，GIL 只是强制只有一个线程可以在任何时候执行 Python 代码。在实践中，全局解释器锁会在许多阻塞系统调用上被释放，并且可以在不使用任何 Python/C API 函数的 C 扩展的部分中被释放。这意味着多个线程可以并行执行 I/O 操作或在某些第三方扩展中执行 C 代码。

对于使用外部资源或涉及 C 代码的非纯代码块，多线程对等待第三方资源返回结果是有用的。这是因为一个明确释放了 GIL 的休眠线程可以等待并在结果返回时唤醒。最后，每当程序需要提供响应式界面时，多线程都是答案，即使它使用时间片。程序可以在进行一些繁重的计算的同时与用户交互，所谓的后台。

请注意，GIL 并不是 Python 语言的每个实现都存在。这是 CPython、Stackless Python 和 PyPy 的限制，但在 Jython 和 IronPython 中并不存在（参见第一章，“Python 的当前状态”）。尽管 PyPy 也在开发无 GIL 版本，但在撰写本书时，它仍处于实验阶段，文档不完善。它基于软件事务内存，称为 PyPy-STM。很难说它何时（或是否）会正式发布为生产就绪的解释器。一切似乎表明这不会很快发生。

## 何时应该使用线程？

尽管有 GIL 的限制，但线程在某些情况下确实非常有用。它们可以帮助：

+   构建响应式界面

+   委托工作

+   构建多用户应用程序

### 构建响应式界面

假设您要求系统通过图形用户界面将文件从一个文件夹复制到另一个文件夹。任务可能会被推送到后台，并且界面窗口将由主线程不断刷新。这样您就可以实时了解整个过程的进展。您还可以取消操作。这比原始的`cp`或`copy` shell 命令少了一些烦恼，因为它在所有工作完成之前不提供任何反馈。

响应式界面还允许用户同时处理多个任务。例如，Gimp 可以让您在处理一张图片的同时处理另一张图片，因为这两个任务是独立的。

在尝试实现这样的响应界面时，一个很好的方法是将长时间运行的任务推送到后台，或者至少尝试为用户提供持续的反馈。实现这一点的最简单方法是使用线程。在这种情况下，它们的目的不是为了提高性能，而只是确保用户即使需要处理一些数据较长时间，也可以继续操作界面。

如果这样的后台任务执行大量 I/O 操作，您仍然可以从多核 CPU 中获得一些好处。这是一个双赢的局面。

### 委托工作

如果您的进程依赖于第三方资源，线程可能会真正加快一切。

让我们考虑一个函数的情况，该函数索引文件夹中的文件并将构建的索引推送到数据库中。根据文件的类型，该函数调用不同的外部程序。例如，一个专门用于 PDF，另一个专门用于 OpenOffice 文件。

您的函数可以为每个转换器设置一个线程，并通过队列将要完成的工作推送给它们中的每一个，而不是按顺序处理每个文件，执行正确的程序，然后将结果存储到数据库中。函数所花费的总时间将更接近最慢转换器的处理时间，而不是所有工作的总和。

转换器线程可以从一开始就初始化，并且负责将结果推送到数据库的代码也可以是一个消耗队列中可用结果的线程。

请注意，这种方法在某种程度上是多线程和多进程的混合。如果您将工作委托给外部进程（例如，使用`subprocess`模块的`run()`函数），实际上是在多个进程中进行工作，因此具有多进程的特征。但在我们的情况下，我们在单独的线程中等待处理结果，因此从 Python 代码的角度来看，这仍然主要是多线程。

线程的另一个常见用例是执行对外部服务的多个 HTTP 请求。例如，如果您想从远程 Web API 获取多个结果，同步执行可能需要很长时间。如果您在进行新请求之前等待每个先前的响应，您将花费大量时间等待外部服务的响应，并且每个请求都会增加额外的往返时间延迟。如果您正在与一个高效的服务（例如 Google Maps API）通信，很可能它可以同时处理大部分请求而不影响单独请求的响应时间。因此，合理的做法是在单独的线程中执行多个查询。请记住，在进行 HTTP 请求时，大部分时间都花在从 TCP 套接字中读取数据上。这是一个阻塞的 I/O 操作，因此在执行`recv()` C 函数时，CPython 会释放 GIL。这可以极大地提高应用程序的性能。

### 多用户应用程序

线程也被用作多用户应用程序的并发基础。例如，Web 服务器将用户请求推送到一个新线程中，然后变为空闲状态，等待新的请求。每个请求都有一个专用的线程简化了很多工作，但需要开发人员注意锁定资源。但是，当所有共享数据都被推送到处理并发事项的关系型数据库中时，这就不是问题了。因此，在多用户应用程序中，线程几乎像独立的进程一样运行。它们在同一个进程下只是为了简化在应用程序级别的管理。

例如，Web 服务器可以将所有请求放入队列，并等待线程可用以将工作发送到线程。此外，它允许内存共享，可以提高一些工作并减少内存负载。两个非常流行的 Python 符合 WSGI 标准的 Web 服务器：**Gunicorn**（参考[`gunicorn.org/`](http://gunicorn.org/)）和**uWSGI**（参考[`uwsgi-docs.readthedocs.org`](https://uwsgi-docs.readthedocs.org)），允许您以符合这一原则的方式使用带有线程工作进程的 HTTP 请求。

在多用户应用程序中使用多线程实现并发性比使用多进程要便宜。单独的进程会消耗更多资源，因为每个进程都需要加载一个新的解释器。另一方面，拥有太多线程也是昂贵的。我们知道 GIL 对 I/O 密集型应用程序并不是问题，但总有一个时刻，您需要执行 Python 代码。由于无法仅使用裸线程并行化应用程序的所有部分，因此在具有多核 CPU 和单个 Python 进程的机器上，您永远无法利用所有资源。这就是为什么通常最佳解决方案是多进程和多线程的混合——多个工作进程（进程）与多个线程同时运行。幸运的是，许多符合 WSGI 标准的 Web 服务器都允许这样的设置。

但在将多线程与多进程结合之前，要考虑这种方法是否真的值得所有的成本。这种方法使用多进程来更好地利用资源，另外使用多线程来实现更多的并发，应该比运行多个进程更轻。但这并不一定是真的。也许摆脱线程，增加进程的数量并不像你想象的那么昂贵？在选择最佳设置时，你总是需要对应用程序进行负载测试（参见第十章中的*负载和性能测试*部分，*测试驱动开发*）。另外，使用多线程的副作用是，你会得到一个不太安全的环境，共享内存会导致数据损坏或可怕的死锁。也许更好的选择是使用一些异步的方法，比如事件循环、绿色线程或协程。我们将在*异步编程*部分后面介绍这些解决方案。同样，如果没有合理的负载测试和实验，你无法真正知道哪种方法在你的情况下效果最好。

### 一个多线程应用的示例

为了了解 Python 线程在实践中是如何工作的，让我们构建一个示例应用程序，可以从实现多线程中获益。我们将讨论一个简单的问题，你可能在职业实践中不时遇到——进行多个并行的 HTTP 查询。这个问题已经被提到作为多线程的常见用例。

假设我们需要使用多个查询从某个网络服务获取数据，这些查询不能被批量处理成一个大的 HTTP 请求。作为一个现实的例子，我们将使用 Google Maps API 的地理编码端点。选择这个服务的原因如下：

+   它非常受欢迎，而且有很好的文档

+   这个 API 有一个免费的层，不需要任何身份验证密钥

+   在 PyPI 上有一个`python-gmaps`包，允许你与各种 Google Maps API 端点进行交互，非常容易使用

地理编码简单地意味着将地址或地点转换为坐标。我们将尝试将预定义的各种城市列表转换为纬度/经度元组，并在标准输出上显示结果与`python-gmaps`。就像下面的代码所示一样简单：

```py
>>> from gmaps import Geocoding
>>> api = Geocoding()
>>> geocoded = api.geocode('Warsaw')[0]
>>> print("{:>25s}, {:6.2f}, {:6.2f}".format(
...         geocoded['formatted_address'],
...         geocoded['geometry']['location']['lat'],
...         geocoded['geometry']['location']['lng'],
...     ))
Warsaw, Poland,  52.23,  21.01

```

由于我们的目标是展示多线程解决并发问题与标准同步解决方案相比的效果，我们将从一个完全不使用线程的实现开始。下面是一个循环遍历城市列表、查询 Google Maps API 并以文本格式表格显示有关它们地址和坐标的信息的程序代码：

```py
import time

from gmaps import Geocoding

api = Geocoding()

PLACES = (
    'Reykjavik', 'Vien', 'Zadar', 'Venice',
    'Wrocław', 'Bolognia', 'Berlin', 'Słubice',
    'New York', 'Dehli',
)

def fetch_place(place):
    geocoded = api.geocode(place)[0]

    print("{:>25s}, {:6.2f}, {:6.2f}".format(
        geocoded['formatted_address'],
        geocoded['geometry']['location']['lat'],
        geocoded['geometry']['location']['lng'],
    ))

def main():
    for place in PLACES:
        fetch_place(place)

if __name__ == "__main__":
    started = time.time()
    main()
    elapsed = time.time() - started

    print()
    print("time elapsed: {:.2f}s".format(elapsed))
```

在`main()`函数的执行周围，我们添加了一些语句，用于测量完成工作所花费的时间。在我的电脑上，这个程序通常需要大约 2 到 3 秒才能完成任务：

```py
$ python3 synchronous.py
 **Reykjavík, Iceland,  64.13, -21.82
 **Vienna, Austria,  48.21,  16.37
 **Zadar, Croatia,  44.12,  15.23
 **Venice, Italy,  45.44,  12.32
 **Wrocław, Poland,  51.11,  17.04
 **Bologna, Italy,  44.49,  11.34
 **Berlin, Germany,  52.52,  13.40
 **Slubice, Poland,  52.35,  14.56
 **New York, NY, USA,  40.71, -74.01
 **Dehli, Gujarat, India,  21.57,  73.22

time elapsed: 2.79s

```

### 注意

我们的脚本每次运行都会花费不同的时间，因为它主要取决于通过网络连接访问的远程服务。所以有很多不确定因素影响最终结果。最好的方法是进行更长时间的测试，多次重复，还要从测量中计算一些平均值。但为了简单起见，我们不会这样做。你将会看到，这种简化的方法对于说明目的来说已经足够了。

#### 每个项目使用一个线程

现在是时候改进了。我们在 Python 中没有进行太多的处理，长时间执行是由与外部服务的通信引起的。我们向服务器发送 HTTP 请求，它计算答案，然后我们等待直到响应被传送回来。涉及了大量的 I/O，因此多线程似乎是一个可行的选择。我们可以在单独的线程中同时启动所有请求，然后等待它们接收数据。如果我们正在通信的服务能够并发处理我们的请求，我们应该肯定会看到性能的提升。

那么让我们从最简单的方法开始。Python 提供了清晰且易于使用的抽象，通过`threading`模块可以轻松地操作系统线程。这个标准库的核心是`Thread`类，代表一个单独的线程实例。下面是`main()`函数的修改版本，它为每个地点创建并启动一个新线程，然后等待直到所有线程都完成：

```py
from threading import Thread

def main():
    threads = []
    for place in PLACES:
        thread = Thread(target=fetch_place, args=[place])
        thread.start()
        threads.append(thread)

    while threads:
        threads.pop().join()
```

这是一个快速而肮脏的改变，它有一些严重的问题，我们稍后会试图解决。它以一种有点轻率的方式解决问题，并不是编写可为成千上万甚至百万用户提供服务的可靠软件的方式。但嘿，它起作用：

```py
$ python3 threaded.py
 **Wrocław, Poland,  51.11,  17.04
 **Vienna, Austria,  48.21,  16.37
 **Dehli, Gujarat, India,  21.57,  73.22
 **New York, NY, USA,  40.71, -74.01
 **Bologna, Italy,  44.49,  11.34
 **Reykjavík, Iceland,  64.13, -21.82
 **Zadar, Croatia,  44.12,  15.23
 **Berlin, Germany,  52.52,  13.40
 **Slubice, Poland,  52.35,  14.56
 **Venice, Italy,  45.44,  12.32

time elapsed: 1.05s

```

所以当我们知道线程对我们的应用有益时，是时候以稍微理智的方式使用它们了。首先我们需要找出前面代码中的问题：

+   我们为每个参数启动一个新线程。线程初始化也需要一些时间，但这种小的开销并不是唯一的问题。线程还会消耗其他资源，比如内存和文件描述符。我们的示例输入有一个严格定义的项目数量，如果没有呢？你肯定不希望运行数量不受限制的线程，这取决于输入数据的任意大小。

+   在线程中执行的`fetch_place()`函数调用了内置的`print()`函数，实际上，你很少会想在主应用程序线程之外这样做。首先，这是因为 Python 中标准输出的缓冲方式。当多个线程之间交错调用这个函数时，你可能会遇到格式不正确的输出。另外，`print()`函数被认为是慢的。如果在多个线程中滥用使用，它可能导致串行化，这将抵消多线程的所有好处。

+   最后但同样重要的是，通过将每个函数调用委托给单独的线程，我们使得控制输入处理速率变得极其困难。是的，我们希望尽快完成工作，但很多时候外部服务会对单个客户端的请求速率设置严格限制。有时，合理设计程序以使其能够控制处理速率是很有必要的，这样你的应用就不会因滥用外部 API 的使用限制而被列入黑名单。

#### 使用线程池

我们要解决的第一个问题是程序运行的线程数量没有限制。一个好的解决方案是建立一个具有严格定义大小的线程工作池，它将处理所有并行工作，并通过一些线程安全的数据结构与工作线程进行通信。通过使用这种线程池方法，我们也将更容易解决刚才提到的另外两个问题。

因此，一般的想法是启动一些预定义数量的线程，这些线程将从队列中消耗工作项，直到完成。当没有其他工作要做时，线程将返回，我们将能够退出程序。用于与工作线程通信的结构的一个很好的候选是内置`queue`模块中的`Queue`类。它是一个先进先出（FIFO）队列实现，非常类似于`collections`模块中的`deque`集合，并且专门设计用于处理线程间通信。以下是一个修改后的`main()`函数的版本，它只启动了有限数量的工作线程，并使用一个新的`worker()`函数作为目标，并使用线程安全的队列与它们进行通信：

```py
from queue import Queue, Empty
from threading import Thread

THREAD_POOL_SIZE = 4

def worker(work_queue):
    while not work_queue.empty():
        try:
            item = work_queue.get(block=False)
        except Empty:
            break
        else:
            fetch_place(item)
            work_queue.task_done()

def main():
    work_queue = Queue()

    for place in PLACES:
        work_queue.put(place)

    threads = [
        Thread(target=worker, args=(work_queue,))
        for _ in range(THREAD_POOL_SIZE)
    ]

    for thread in threads:
        thread.start()

    work_queue.join()

    while threads:
        threads.pop().join()
```

运行修改后的程序的结果与之前的类似：

```py
$ python threadpool.py** 
 **Reykjavík, Iceland,  64.13, -21.82
 **Venice, Italy,  45.44,  12.32
 **Vienna, Austria,  48.21,  16.37
 **Zadar, Croatia,  44.12,  15.23
 **Wrocław, Poland,  51.11,  17.04
 **Bologna, Italy,  44.49,  11.34
 **Slubice, Poland,  52.35,  14.56
 **Berlin, Germany,  52.52,  13.40
 **New York, NY, USA,  40.71, -74.01
 **Dehli, Gujarat, India,  21.57,  73.22

time elapsed: 1.20s

```

运行时间将比每个参数一个线程的情况慢，但至少现在不可能用任意长的输入耗尽所有的计算资源。此外，我们可以调整`THREAD_POOL_SIZE`参数以获得更好的资源/时间平衡。

#### 使用双向队列

我们现在能够解决的另一个问题是线程中输出的潜在问题。最好将这样的责任留给启动其他线程的主线程。我们可以通过提供另一个队列来处理这个问题，该队列将负责从我们的工作线程中收集结果。以下是将所有内容与主要更改放在一起的完整代码：

```py
import time
from queue import Queue, Empty
from threading import Thread

from gmaps import Geocoding

api = Geocoding()

PLACES = (
    'Reykjavik', 'Vien', 'Zadar', 'Venice',
    'Wrocław', 'Bolognia', 'Berlin', 'Słubice',
    'New York', 'Dehli',
)

THREAD_POOL_SIZE = 4

def fetch_place(place):
    return api.geocode(place)[0]

def present_result(geocoded):
 **print("{:>25s}, {:6.2f}, {:6.2f}".format(
 **geocoded['formatted_address'],
 **geocoded['geometry']['location']['lat'],
 **geocoded['geometry']['location']['lng'],
 **))

def worker(work_queue, results_queue):
    while not work_queue.empty():
        try:
            item = work_queue.get(block=False)
        except Empty:
            break
        else:
 **results_queue.put(
 **fetch_place(item)
 **)
            work_queue.task_done()

def main():
    work_queue = Queue()
 **results_queue = Queue()

    for place in PLACES:
        work_queue.put(place)

    threads = [
 **Thread(target=worker, args=(work_queue, results_queue))
        for _ in range(THREAD_POOL_SIZE)
    ]

    for thread in threads:
        thread.start()

    work_queue.join()

    while threads:
        threads.pop().join()

 **while not results_queue.empty():
 **present_result(results_queue.get())

if __name__ == "__main__":
    started = time.time()
    main()
    elapsed = time.time() - started

    print()
    print("time elapsed: {:.2f}s".format(elapsed))
```

这消除了输出格式不正确的风险，如果`present_result()`函数执行更多的`print()`语句或执行一些额外的计算，我们可能会遇到这种情况。我们不希望从这种方法中获得任何性能改进，但实际上，由于`print()`执行缓慢，我们还减少了线程串行化的风险。这是我们的最终输出：

```py
$ python threadpool_with_results.py** 
 **Vienna, Austria,  48.21,  16.37
 **Reykjavík, Iceland,  64.13, -21.82
 **Zadar, Croatia,  44.12,  15.23
 **Venice, Italy,  45.44,  12.32
 **Wrocław, Poland,  51.11,  17.04
 **Bologna, Italy,  44.49,  11.34
 **Slubice, Poland,  52.35,  14.56
 **Berlin, Germany,  52.52,  13.40
 **New York, NY, USA,  40.71, -74.01
 **Dehli, Gujarat, India,  21.57,  73.22

time elapsed: 1.30s

```

#### 处理错误和速率限制

之前提到的您在处理这些问题时可能遇到的最后一个问题是外部服务提供商施加的速率限制。在编写本书时，谷歌地图 API 的官方速率限制为每秒 10 次请求和每天 2500 次免费和非身份验证请求。使用多个线程很容易耗尽这样的限制。问题更加严重，因为我们尚未涵盖任何故障场景，并且在多线程 Python 代码中处理异常比通常要复杂一些。

`api.geocode()` 函数在客户端超过谷歌速率时会引发异常，这是个好消息。但是这个异常会单独引发，并不会使整个程序崩溃。工作线程当然会立即退出，但主线程会等待所有存储在`work_queue`上的任务完成（使用`work_queue.join()`调用）。这意味着我们的工作线程应该优雅地处理可能的异常，并确保队列中的所有项目都被处理。如果没有进一步的改进，我们可能会陷入一种情况，其中一些工作线程崩溃，程序将永远不会退出。

让我们对我们的代码进行一些微小的更改，以便为可能发生的任何问题做好准备。在工作线程中出现异常的情况下，我们可以将错误实例放入`results_queue`队列，并将当前任务标记为已完成，就像没有错误时一样。这样我们可以确保主线程在`work_queue.join()`中等待时不会无限期地锁定。然后主线程可能检查结果并重新引发在结果队列中找到的任何异常。以下是可以更安全地处理异常的`worker()`和`main()`函数的改进版本：

```py
def worker(work_queue, results_queue):
    while True:
        try:
            item = work_queue.get(block=False)
        except Empty:
            break
        else:
 **try:
 **result = fetch_place(item)
 **except Exception as err:
 **results_queue.put(err)
 **else:
 **results_queue.put(result)
 **finally:
 **work_queue.task_done()

def main():
    work_queue = Queue()
    results_queue = Queue()

    for place in PLACES:
        work_queue.put(place)

    threads = [
        Thread(target=worker, args=(work_queue, results_queue))
        for _ in range(THREAD_POOL_SIZE)
    ]

    for thread in threads:
        thread.start()

    work_queue.join()

    while threads:
        threads.pop().join()

 **while not results_queue.empty():
 **result = results_queue.get()

 **if isinstance(result, Exception):
 **raise result

        present_result(result)
```

当我们准备处理异常时，就是我们的代码中断并超过速率限制的时候了。我们可以通过修改一些初始条件来轻松实现这一点。让我们增加地理编码的位置数量和线程池的大小：

```py
PLACES = (
    'Reykjavik', 'Vien', 'Zadar', 'Venice',
    'Wrocław', 'Bolognia', 'Berlin', 'Słubice',
    'New York', 'Dehli',
) * 10

THREAD_POOL_SIZE = 10
```

如果您的执行环境足够快，您应该很快就会收到类似的错误：

```py
$ python3 threadpool_with_errors.py
 **New York, NY, USA,  40.71, -74.01
 **Berlin, Germany,  52.52,  13.40
 **Wrocław, Poland,  51.11,  17.04
 **Zadar, Croatia,  44.12,  15.23
 **Vienna, Austria,  48.21,  16.37
 **Bologna, Italy,  44.49,  11.34
 **Reykjavík, Iceland,  64.13, -21.82
 **Venice, Italy,  45.44,  12.32
 **Dehli, Gujarat, India,  21.57,  73.22
 **Slubice, Poland,  52.35,  14.56
 **Vienna, Austria,  48.21,  16.37
 **Zadar, Croatia,  44.12,  15.23
 **Venice, Italy,  45.44,  12.32
 **Reykjavík, Iceland,  64.13, -21.82
Traceback (most recent call last):
 **File "threadpool_with_errors.py", line 83, in <module>
 **main()
 **File "threadpool_with_errors.py", line 76, in main
 **raise result
 **File "threadpool_with_errors.py", line 43, in worker
 **result = fetch_place(item)
 **File "threadpool_with_errors.py", line 23, in fetch_place
 **return api.geocode(place)[0]
 **File "...\site-packages\gmaps\geocoding.py", line 37, in geocode
 **return self._make_request(self.GEOCODE_URL, parameters, "results")
 **File "...\site-packages\gmaps\client.py", line 89, in _make_request
 **)(response)
gmaps.errors.RateLimitExceeded: {'status': 'OVER_QUERY_LIMIT', 'results': [], 'error_message': 'You have exceeded your rate-limit for this API.', 'url': 'https://maps.googleapis.com/maps/api/geocode/json?address=Wroc%C5%82aw&sensor=false'}

```

前面的异常当然不是由于错误的代码造成的。这个程序对于这个免费服务来说太快了。它发出了太多的并发请求，为了正确工作，我们需要有一种限制它们速率的方法。

限制工作的速度通常被称为节流。PyPI 上有一些包可以让您限制任何类型工作的速率，并且非常容易使用。但是我们不会在这里使用任何外部代码。节流是一个很好的机会，可以引入一些用于线程的锁原语，因此我们将尝试从头开始构建一个解决方案。

我们将使用的算法有时被称为令牌桶，非常简单：

1.  有一个预定义数量的令牌的桶。

1.  每个令牌对应于处理一个工作项的单个权限。

1.  每次工作线程请求单个或多个令牌（权限）时：

+   我们测量了从上次我们重新填充桶以来花费了多少时间

+   如果时间差允许，我们将用与此时间差相应的令牌数量重新填充桶

+   如果存储的令牌数量大于或等于请求的数量，我们会减少存储的令牌数量并返回该值

+   如果存储的令牌数量少于请求的数量，我们返回零

两个重要的事情是始终用零令牌初始化令牌桶，并且永远不允许它填充的令牌数量超过其速率可用的令牌数量，按照我们标准的时间量表达。如果我们不遵循这些预防措施，我们可能会以超过速率限制的突发方式释放令牌。因为在我们的情况下，速率限制以每秒请求的数量来表示，所以我们不需要处理任意的时间量。我们假设我们的测量基准是一秒，因此我们永远不会存储比该时间量允许的请求数量更多的令牌。以下是一个使用令牌桶算法进行节流的类的示例实现：

```py
From threading import Lock

class Throttle:
    def __init__(self, rate):
        self._consume_lock = Lock()
        self.rate = rate
        self.tokens = 0
        self.last = 0

    def consume(self, amount=1):
        with self._consume_lock:
            now = time.time()

            # time measument is initialized on first
            # token request to avoid initial bursts
            if self.last == 0:
                self.last = now

            elapsed = now - self.last

            # make sure that quant of passed time is big
            # enough to add new tokens
            if int(elapsed * self.rate):
                self.tokens += int(elapsed * self.rate)
                self.last = now

            # never over-fill the bucket
            self.tokens = (
                self.rate
                if self.tokens > self.rate
                else self.tokens
            )

            # finally dispatch tokens if available
            if self.tokens >= amount:
                self.tokens -= amount
            else:
                amount = 0

            return amount
```

使用这个类非常简单。假设我们在主线程中只创建了一个`Throttle`实例（例如`Throttle(10)`），并将其作为位置参数传递给每个工作线程。在不同的线程中使用相同的数据结构是安全的，因为我们使用`threading`模块中的`Lock`类的实例来保护其内部状态的操作。现在我们可以更新`worker()`函数的实现，以便在每个项目之前等待节流释放一个新的令牌：

```py
def worker(work_queue, results_queue, throttle):
    while True:
        try:
            item = work_queue.get(block=False)
        except Empty:
            break
        else:
 **while not throttle.consume():
 **pass

            try:
                result = fetch_place(item)
            except Exception as err:
                results_queue.put(err)
            else:
                results_queue.put(result)
            finally:
                work_queue.task_done()
```

# 多进程

坦率地说，多线程是具有挑战性的——我们在前一节已经看到了。最简单的方法只需要最少的工作。但是以明智和安全的方式处理线程需要大量的代码。

我们必须设置线程池和通信队列，优雅地处理来自线程的异常，并且在尝试提供速率限制功能时也要关心线程安全。只需十行代码就可以并行执行外部库中的一个函数！我们只是假设这是可以投入生产的，因为外部包的创建者承诺他的库是线程安全的。对于一个实际上只适用于执行 I/O 绑定任务的解决方案来说，这听起来像是一个很高的代价。

允许你实现并行的另一种方法是多进程。不受 GIL 约束的独立 Python 进程可以更好地利用资源。这对于在执行真正消耗 CPU 的任务的多核处理器上运行的应用程序尤为重要。目前，这是 Python 开发人员（使用 CPython 解释器）唯一可用的内置并发解决方案，可以让你利用多个处理器核心。

使用多个进程的另一个优势是它们不共享内存上下文。因此，更难破坏数据并引入死锁到你的应用程序中。不共享内存上下文意味着你需要额外的工作来在独立的进程之间传递数据，但幸运的是有许多很好的方法来实现可靠的进程间通信。事实上，Python 提供了一些原语，使进程间通信尽可能简单，就像线程之间一样。

在任何编程语言中启动新进程的最基本的方法通常是在某个时候**fork**程序。在 POSIX 系统（Unix、Mac OS 和 Linux）上，fork 是一个系统调用，在 Python 中通过`os.fork()`函数暴露出来，它将创建一个新的子进程。然后这两个进程在分叉后继续程序。下面是一个自我分叉一次的示例脚本：

```py
import os

pid_list = []

def main():
    pid_list.append(os.getpid())
    child_pid = os.fork()

    if child_pid == 0:
        pid_list.append(os.getpid())
        print()
        print("CHLD: hey, I am the child process")
        print("CHLD: all the pids i know %s" % pid_list)

    else:
        pid_list.append(os.getpid())
        print()
        print("PRNT: hey, I am the parent")
        print("PRNT: the child is pid %d" % child_pid)
        print("PRNT: all the pids i know %s" % pid_list)

if __name__ == "__main__":
    main()
```

以下是在终端中运行它的示例：

```py
$ python3 forks.py

PRNT: hey, I am the parent
PRNT: the child is pid 21916
PRNT: all the pids i know [21915, 21915]

CHLD: hey, I am the child process
CHLD: all the pids i know [21915, 21916]

```

请注意，在`os.fork()`调用之前，这两个进程的数据状态完全相同。它们都有相同的 PID 号（进程标识符）作为`pid_list`集合的第一个值。后来，两个状态分歧，我们可以看到子进程添加了`21916`的值，而父进程复制了它的`21915` PID。这是因为这两个进程的内存上下文是不共享的。它们有相同的初始条件，但在`os.fork()`调用后不能相互影响。

在分叉内存上下文被复制到子进程后，每个进程都处理自己的地址空间。为了通信，进程需要使用系统范围的资源或使用低级工具，比如**信号**。

不幸的是，在 Windows 下`os.fork`不可用，需要在新的解释器中生成一个新的进程来模拟 fork 功能。因此，它需要根据平台的不同而有所不同。`os`模块还公开了在 Windows 下生成新进程的函数，但最终你很少会使用它们。这对于`os.fork()`也是如此。Python 提供了一个很棒的`multiprocessing`模块，它为多进程提供了一个高级接口。这个模块的巨大优势在于它提供了一些我们在*一个多线程应用程序示例*部分中不得不从头编写的抽象。它允许你限制样板代码的数量，因此提高了应用程序的可维护性并减少了其复杂性。令人惊讶的是，尽管它的名字是`multiprocessing`模块，但它也为线程暴露了类似的接口，因此你可能希望对两种方法使用相同的接口。

## 内置的 multiprocessing 模块

`multiprocessing`提供了一种可移植的方式来处理进程，就像它们是线程一样。

这个模块包含一个`Process`类，它与`Thread`类非常相似，可以在任何平台上使用：

```py
from multiprocessing import Process
import os

def work(identifier):
    print(
        'hey, i am a process {}, pid: {}'
        ''.format(identifier, os.getpid())
    )

def main():
    processes = [
        Process(target=work, args=(number,))
        for number in range(5)
    ]
    for process in processes:
        process.start()

    while processes:
        processes.pop().join()

if __name__ == "__main__":
    main()
```

执行前述脚本将得到以下结果：

```py
$ python3 processing.py
hey, i am a process 1, pid: 9196
hey, i am a process 0, pid: 8356
hey, i am a process 3, pid: 9524
hey, i am a process 2, pid: 3456
hey, i am a process 4, pid: 6576

```

当进程被创建时，内存被分叉（在 POSIX 系统上）。进程的最有效使用方式是让它们在创建后独立工作，以避免开销，并从主线程检查它们的状态。除了复制的内存状态，`Process`类还在其构造函数中提供了额外的`args`参数，以便传递数据。

进程模块之间的通信需要一些额外的工作，因为它们的本地内存默认情况下不是共享的。为了简化这一点，多进程模块提供了一些进程之间通信的方式：

+   使用`multiprocessing.Queue`类，它几乎与`queue.Queue`相同，之前用于线程之间通信

+   使用`multiprocessing.Pipe`，这是一个类似套接字的双向通信通道

+   使用`multiprocessing.sharedctypes`模块，允许您在进程之间共享的专用内存池中创建任意 C 类型（来自`ctypes`模块）

`multiprocessing.Queue`和`queue.Queue`类具有相同的接口。唯一的区别是，第一个是设计用于多进程环境，而不是多线程环境，因此它使用不同的内部传输和锁定原语。我们已经看到如何在*一个多线程应用程序的示例*部分中使用 Queue，因此我们不会对多进程做同样的事情。使用方式完全相同，因此这样的例子不会带来任何新东西。

现在提供的更有趣的模式是`Pipe`类。它是一个双工（双向）通信通道，概念上与 Unix 管道非常相似。Pipe 的接口也非常类似于内置`socket`模块中的简单套接字。与原始系统管道和套接字的区别在于它允许您发送任何可挑选的对象（使用`pickle`模块）而不仅仅是原始字节。这使得进程之间的通信变得更加容易，因为您可以发送几乎任何基本的 Python 类型：

```py
from multiprocessing import Process, Pipe

class CustomClass:
    pass

def work(connection):
    while True:
        instance = connection.recv()

        if instance:
            print("CHLD: {}".format(instance))

        else:
            return

def main():
    parent_conn, child_conn = Pipe()

    child = Process(target=work, args=(child_conn,))

    for item in (
        42,
        'some string',
        {'one': 1},
        CustomClass(),
        None,
    ):
        print("PRNT: send {}:".format(item))
        parent_conn.send(item)

    child.start()
    child.join()

if __name__ == "__main__":
    main()
```

当查看前面脚本的示例输出时，您会发现您可以轻松传递自定义类实例，并且它们根据进程具有不同的地址：

```py
PRNT: send: 42
PRNT: send: some string
PRNT: send: {'one': 1}
PRNT: send: <__main__.CustomClass object at 0x101cb5b00>
PRNT: send: None
CHLD: recv: 42
CHLD: recv: some string
CHLD: recv: {'one': 1}
CHLD: recv: <__main__.CustomClass object at 0x101cba400>

```

在进程之间共享状态的另一种方法是使用`multiprocessing.sharedctypes`中提供的类在共享内存池中使用原始类型。最基本的是`Value`和`Array`。以下是`multiprocessing`模块官方文档中的示例代码：

```py
from multiprocessing import Process, Value, Array

def f(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]

if __name__ == '__main__':
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    p = Process(target=f, args=(num, arr))
    p.start()
    p.join()

    print(num.value)
    print(arr[:])
```

这个例子将打印以下输出：

```py
3.1415927
[0, -1, -2, -3, -4, -5, -6, -7, -8, -9]

```

在使用`multiprocessing.sharedctypes`时，您需要记住您正在处理共享内存，因此为了避免数据损坏的风险，您需要使用锁定原语。多进程提供了一些可用于线程的类，例如`Lock`、`RLock`和`Semaphore`，来做到这一点。`sharedctypes`类的缺点是它们只允许您共享`ctypes`模块中的基本 C 类型。如果您需要传递更复杂的结构或类实例，则需要使用 Queue、Pipe 或其他进程间通信通道。在大多数情况下，理应避免使用`sharedctypes`中的类型，因为它们会增加代码复杂性，并带来来自多线程的所有已知危险。

### 使用进程池

使用多进程而不是线程会增加一些实质性的开销。主要是因为它增加了内存占用，因为每个进程都有自己独立的内存上下文。这意味着允许无限数量的子进程甚至比在多线程应用程序中更加棘手。

在依赖多进程进行更好资源利用的应用程序中控制资源使用的最佳模式是以类似于*使用线程池*部分描述的方式构建进程池。

`multiprocessing`模块最好的地方是它提供了一个现成的`Pool`类，可以为你处理管理多个进程工作者的所有复杂性。这个池实现大大减少了所需的样板代码量和与双向通信相关的问题数量。你也不需要手动使用`join()`方法，因为`Pool`可以作为上下文管理器使用（使用`with`语句）。以下是我们以前的一个线程示例，重写为使用`multiprocessing`模块中的`Pool`类：

```py
from multiprocessing import Pool

from gmaps import Geocoding

api = Geocoding()

PLACES = (
    'Reykjavik', 'Vien', 'Zadar', 'Venice',
    'Wrocław', 'Bolognia', 'Berlin', 'Słubice',
    'New York', 'Dehli',
)

POOL_SIZE = 4

def fetch_place(place):
    return api.geocode(place)[0]

def present_result(geocoded):
    print("{:>25s}, {:6.2f}, {:6.2f}".format(
        geocoded['formatted_address'],
        geocoded['geometry']['location']['lat'],
        geocoded['geometry']['location']['lng'],
    ))

def main():
    with Pool(POOL_SIZE) as pool:
        results = pool.map(fetch_place, PLACES)

    for result in results:
        present_result(result)

if __name__ == "__main__":
    main()
```

正如你所看到的，现在代码要短得多。这意味着在出现问题时，现在更容易维护和调试。实际上，现在只有两行代码明确处理多进程。这是一个很大的改进，因为我们以前必须从头开始构建处理池。现在我们甚至不需要关心通信通道，因为它们是在`Pool`实现内部隐式创建的。

### 使用`multiprocessing.dummy`作为多线程接口

`multiprocessing`模块中的高级抽象，如`Pool`类，是比`threading`模块提供的简单工具更大的优势。但是，并不意味着多进程始终比多线程更好的方法。有很多情况下，线程可能是比进程更好的解决方案。特别是在需要低延迟和/或高资源效率的情况下。

但这并不意味着每当你想要使用线程而不是进程时，你就需要牺牲`multiprocessing`模块中的所有有用抽象。有`multiprocessing.dummy`模块，它复制了`multiprocessing`的 API，但使用多线程而不是 forking/spawning 新进程。

这使你可以减少代码中的样板，并且使接口更加可插拔。例如，让我们再次看一下我们以前示例中的`main()`函数。如果我们想要让用户控制他想要使用哪种处理后端（进程或线程），我们可以简单地替换`Pool`类：

```py
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool

def main(use_threads=False):
    if use_threads:
        pool_cls = ThreadPool
    else:
        pool_cls = ProcessPool

    with pool_cls(POOL_SIZE) as pool:
        results = pool.map(fetch_place, PLACES)

    for result in results:
        present_result(result)
```

# 异步编程

近年来，异步编程已经获得了很大的关注。在 Python 3.5 中，它最终获得了一些语法特性，巩固了异步执行的概念。但这并不意味着异步编程只能从 Python 3.5 开始。很多库和框架早在很久以前就提供了，大部分都起源于 Python 2 的旧版本。甚至有一个名为 Stackless 的 Python 的整个替代实现（见第一章，“Python 的当前状态”），它专注于这种单一的编程方法。其中一些解决方案，如 Twisted、Tornado 或 Eventlet，仍然拥有庞大和活跃的社区，并且真的值得了解。无论如何，从 Python 3.5 开始，异步编程比以往任何时候都更容易。因此，预计其内置的异步特性将取代较旧工具的大部分部分，或者外部项目将逐渐转变为基于 Python 内置的高级框架。

当试图解释什么是异步编程时，最简单的方法是将这种方法视为类似于线程但不涉及系统调度。这意味着异步程序可以并发处理问题，但其上下文在内部切换，而不是由系统调度程序切换。

但是，当然，我们不使用线程来同时处理异步程序中的工作。大多数解决方案使用一种不同的概念，根据实现的不同，它被命名为不同的名称。用来描述这种并发程序实体的一些示例名称是：

+   绿色线程或 greenlets（greenlet、gevent 或 eventlet 项目）

+   协程（Python 3.5 原生异步编程）

+   任务（Stackless Python）

这些主要是相同的概念，但通常以稍微不同的方式实现。出于明显的原因，在本节中，我们将只集中讨论 Python 从版本 3.5 开始原生支持的协程。

## 合作式多任务处理和异步 I/O

**合作式多任务处理**是异步编程的核心。在这种计算机多任务处理风格中，操作系统不负责启动上下文切换（到另一个进程或线程），而是每个进程在空闲时自愿释放控制，以实现多个程序的同时执行。这就是为什么它被称为*合作式*。所有进程都需要合作才能实现平稳的多任务处理。

这种多任务处理模型有时在操作系统中使用，但现在几乎不再作为系统级解决方案。这是因为一个设计不良的服务很容易破坏整个系统的稳定性。现在，线程和进程调度以及由操作系统直接管理的上下文切换是系统级并发的主要方法。但在应用程序级别，合作式多任务处理仍然是一个很好的并发工具。

在应用程序级别讨论合作式多任务处理时，我们不需要处理需要释放控制的线程或进程，因为所有执行都包含在一个单一的进程和线程中。相反，我们有多个任务（协程、任务和绿色线程），它们释放控制给处理任务协调的单个函数。这个函数通常是某种事件循环。

为了避免以后混淆（由于 Python 术语），从现在开始我们将把这样的并发任务称为*协程*。合作式多任务处理中最重要的问题是何时释放控制。在大多数异步应用程序中，控制权在 I/O 操作时释放给调度器或事件循环。无论程序是从文件系统读取数据还是通过套接字进行通信，这样的 I/O 操作总是与进程变得空闲的等待时间相关。等待时间取决于外部资源，因此释放控制是一个很好的机会，这样其他协程就可以做他们的工作，直到它们也需要等待。

这使得这种方法在行为上与 Python 中的多线程实现方式有些相似。我们知道 GIL 会对 Python 线程进行串行化，但在每次 I/O 操作时会释放。主要区别在于 Python 中的线程是作为系统级线程实现的，因此操作系统可以在任何时间点抢占当前运行的线程，并将控制权交给另一个线程。在异步编程中，任务永远不会被主事件循环抢占。这就是为什么这种多任务处理风格也被称为**非抢占式多任务处理**。

当然，每个 Python 应用程序都在一个操作系统上运行，那里有其他进程竞争资源。这意味着操作系统始终有权剥夺整个进程的控制权，并将控制权交给另一个进程。但当我们的异步应用程序恢复运行时，它会从系统调度器介入时暂停的地方继续运行。这就是为什么协程仍然被认为是非抢占式的。

## Python 的 async 和 await 关键字

`async`和`await`关键字是 Python 异步编程的主要构建模块。

在`def`语句之前使用的`async`关键字定义了一个新的协程。协程函数的执行可能在严格定义的情况下被暂停和恢复。它的语法和行为与生成器非常相似（参见第二章，“语法最佳实践-类级别下面”）。实际上，生成器需要在 Python 的旧版本中使用以实现协程。这是一个使用`async`关键字的函数声明的示例：

```py
async def async_hello():
    print("hello, world!")
```

使用`async`关键字定义的函数是特殊的。当调用时，它们不执行内部的代码，而是返回一个协程对象：

```py
>>> async def async_hello():
...     print("hello, world!")
...** 
>>> async_hello()
<coroutine object async_hello at 0x1014129e8>

```

协程对象在其执行被安排在事件循环中之前不会执行任何操作。`asyncio`模块可用于提供基本的事件循环实现，以及许多其他异步实用程序：

```py
>>> import asyncio
>>> async def async_hello():
...     print("hello, world!")
...** 
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(async_hello())
hello, world!
>>> loop.close()

```

显然，由于我们只创建了一个简单的协程，所以在我们的程序中没有涉及并发。为了真正看到一些并发，我们需要创建更多的任务，这些任务将由事件循环执行。

可以通过调用`loop.create_task()`方法或使用`asyncio.wait()`函数提供另一个对象来等待来添加新任务到循环中。我们将使用后一种方法，并尝试异步打印使用`range()`函数生成的一系列数字：

```py
import asyncio

async def print_number(number):
    print(number)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    loop.run_until_complete(
        asyncio.wait([
            print_number(number)
            for number in range(10)
        ])
    )
    loop.close()
```

`asyncio.wait()`函数接受一个协程对象的列表并立即返回。结果是一个生成器，产生表示未来结果（futures）的对象。正如其名称所示，它用于等待所有提供的协程完成。它返回生成器而不是协程对象的原因是为了与 Python 的先前版本向后兼容，这将在后面解释。运行此脚本的结果可能如下：

```py
$ python asyncprint.py** 
0
7
8
3
9
4
1
5
2
6

```

正如我们所看到的，数字的打印顺序与我们创建协程的顺序不同。但这正是我们想要实现的。

Python 3.5 中添加的第二个重要关键字是`await`。它用于等待协程或未来结果（稍后解释）的结果，并将执行控制权释放给事件循环。为了更好地理解它的工作原理，我们需要回顾一个更复杂的代码示例。

假设我们想创建两个协程，它们将在循环中执行一些简单的任务：

+   等待随机秒数

+   打印一些作为参数提供的文本和在睡眠中花费的时间

让我们从一个简单的实现开始，它存在一些并发问题，我们稍后将尝试使用额外的`await`使用来改进它：

```py
import time
import random
import asyncio

async def waiter(name):
    for _ in range(4):
        time_to_sleep = random.randint(1, 3) / 4
        time.sleep(time_to_sleep)
        print(
            "{} waited {} seconds"
            "".format(name, time_to_sleep)
        )

async def main():
    await asyncio.wait([waiter("foo"), waiter("bar")])

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
```

在终端中执行（使用`time`命令来测量时间），可能会得到以下输出：

```py
$ time python corowait.py** 
bar waited 0.25 seconds
bar waited 0.25 seconds
bar waited 0.5 seconds
bar waited 0.5 seconds
foo waited 0.75 seconds
foo waited 0.75 seconds
foo waited 0.25 seconds
foo waited 0.25 seconds

real	0m3.734s
user	0m0.153s
sys	0m0.028s

```

正如我们所看到的，这两个协程都完成了它们的执行，但不是以异步的方式。原因是它们都使用了`time.sleep()`函数，这是阻塞的，但没有释放控制给事件循环。这在多线程设置中可能效果更好，但我们现在不想使用线程。那么我们该如何解决这个问题呢？

答案是使用`asyncio.sleep()`，这是`time.sleep()`的异步版本，并使用`await`关键字等待其结果。我们已经在`main()`函数的第一个版本中使用了这个语句，但这只是为了提高代码的清晰度。显然，这并没有使我们的实现更加并发。让我们看一个改进的`waiter()`协程的版本，它使用`await asyncio.sleep()`：

```py
async def waiter(name):
    for _ in range(4):
        time_to_sleep = random.randint(1, 3) / 4
        await asyncio.sleep(time_to_sleep)
        print(
            "{} waited {} seconds"
            "".format(name, time_to_sleep)
        )
```

如果我们运行更新后的脚本，我们可以看到两个函数的输出如何交错：

```py
$ time python corowait_improved.py** 
bar waited 0.25 seconds
foo waited 0.25 seconds
bar waited 0.25 seconds
foo waited 0.5 seconds
foo waited 0.25 seconds
bar waited 0.75 seconds
foo waited 0.25 seconds
bar waited 0.5 seconds

real  0m1.953s
user  0m0.149s
sys   0m0.026s

```

这个简单改进的额外优势是代码运行得更快。总体执行时间小于所有睡眠时间的总和，因为协程合作地释放控制。

## 旧版本 Python 中的 asyncio

`asyncio`模块出现在 Python 3.4 中。因此，它是在 Python 3.5 之前唯一支持异步编程的版本。不幸的是，看起来这两个后续版本刚好足够引入兼容性问题。

无论喜欢与否，Python 中的异步编程核心早于支持此模式的语法元素。迟做总比不做好，但这造成了一种情况，即有两种语法可用于处理协程。

从 Python 3.5 开始，你可以使用`async`和`await`：

```py
async def main():
    await asyncio.sleep(0)
```

但对于 Python 3.4，你需要使用`asyncio.coroutine`装饰器和`yield from`语句：

```py
@asyncio.couroutine
def main():
    yield from asyncio.sleep(0)
```

另一个有用的事实是，`yield from`语句是在 Python 3.3 中引入的，并且在 PyPI 上有一个`asyncio`的后备。这意味着你也可以在 Python 3.3 中使用这个协作式多任务处理的实现。

## 异步编程的实际示例

正如本章中已经多次提到的那样，异步编程是处理 I/O 绑定操作的强大工具。所以现在是时候构建比简单打印序列或异步等待更实际的东西了。

为了保持一致，我们将尝试处理与多线程和多进程帮助解决的相同问题。因此，我们将尝试通过网络连接异步获取一些来自外部资源的数据。如果我们可以像在前面的部分中那样使用相同的`python-gmaps`包，那就太好了。不幸的是，我们不能。

`python-gmaps`的创建者有点懒，走了捷径。为了简化开发，他选择了`requests`包作为他的首选 HTTP 客户端库。不幸的是，`requests`不支持`async`和`await`的异步 I/O。还有一些其他项目旨在为`requests`项目提供一些并发性，但它们要么依赖于 Gevent（`grequests`，参见[`github.com/kennethreitz/grequests`](https://github.com/kennethreitz/grequests)），要么依赖于线程/进程池执行（`requests-futures`，参见[`github.com/ross/requests-futures`](https://github.com/ross/requests-futures)）。这两者都不能解决我们的问题。

### 注意

在你因为我在责备一个无辜的开源开发者而生气之前，冷静下来。`python-gmaps`包背后的人就是我。依赖项的选择不当是这个项目的问题之一。我只是喜欢偶尔公开批评自己。这对我来说应该是一个痛苦的教训，因为在我写这本书的时候，`python-gmaps`在其最新版本（0.3.1）中不能轻松地与 Python 的异步 I/O 集成。无论如何，这可能会在未来发生变化，所以一切都没有丢失。

知道在前面的示例中很容易使用的库的限制，我们需要构建一些填补这一空白的东西。Google Maps API 非常容易使用，所以我们将构建一个快速而简陋的异步实用程序，仅用于说明目的。Python 3.5 版本的标准库仍然缺少一个使异步 HTTP 请求像调用`urllib.urlopen()`一样简单的库。我们绝对不想从头开始构建整个协议支持，所以我们将从 PyPI 上可用的`aiohttp`包中得到一点帮助。这是一个非常有前途的库，为异步 HTTP 添加了客户端和服务器实现。这是一个建立在`aiohttp`之上的小模块，它创建了一个名为`geocode()`的辅助函数，用于向 Google Maps API 服务发出地理编码请求：

```py
import aiohttp

session = aiohttp.ClientSession()

async def geocode(place):
    params = {
        'sensor': 'false',
        'address': place
    }
    async with session.get(
        'https://maps.googleapis.com/maps/api/geocode/json',
        params=params
    ) as response:
        result = await response.json()
        return result['results']
```

假设这段代码存储在名为`asyncgmaps`的模块中，我们稍后会用到它。现在我们准备重写在讨论多线程和多进程时使用的示例。以前，我们习惯将整个操作分为两个独立的步骤：

1.  使用`fetch_place()`函数并行执行对外部服务的所有请求。

1.  使用`present_result()`函数在循环中显示所有结果。

但是，因为协作式多任务处理与使用多个进程或线程完全不同，我们可以稍微修改我们的方法。在“使用一个线程处理一个项目”部分提出的大部分问题不再是我们的关注点。协程是非抢占式的，因此我们可以在等待 HTTP 响应后立即显示结果。这将简化我们的代码并使其更清晰。

```py
import asyncio
# note: local module introduced earlier
from asyncgmaps import geocode, session

PLACES = (
    'Reykjavik', 'Vien', 'Zadar', 'Venice',
    'Wrocław', 'Bolognia', 'Berlin', 'Słubice',
    'New York', 'Dehli',
)

async def fetch_place(place):
    return (await geocode(place))[0]

async def present_result(result):
    geocoded = await result
    print("{:>25s}, {:6.2f}, {:6.2f}".format(
        geocoded['formatted_address'],
        geocoded['geometry']['location']['lat'],
        geocoded['geometry']['location']['lng'],
    ))

async def main():
    await asyncio.wait([
        present_result(fetch_place(place))
        for place in PLACES
    ])

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

    # aiohttp will raise issue about unclosed
    # ClientSession so we perform cleanup manually
    loop.run_until_complete(session.close())
    loop.close()
```

## 使用期货将非异步代码与异步集成

异步编程很棒，特别是对于对构建可扩展应用程序感兴趣的后端开发人员。实际上，这是构建高度并发服务器的最重要工具之一。

但现实是痛苦的。许多处理 I/O 绑定问题的流行软件包并不适用于异步代码。主要原因是：

+   Python 3 及其一些高级功能的采用率仍然较低

+   Python 初学者对各种并发概念的理解较低

这意味着现有的同步多线程应用程序和软件包的迁移通常是不可能的（由于架构约束）或成本太高。许多项目可以从合并异步多任务处理方式中受益，但最终只有少数项目会这样做。

这意味着现在，当尝试从头开始构建异步应用程序时，您将遇到许多困难。在大多数情况下，这将类似于“异步编程的实际示例”部分中提到的问题 - 接口不兼容和 I/O 操作的非异步阻塞。

当您遇到这种不兼容性时，您有时可以放弃`await`并同步获取所需的资源。但这将在等待结果时阻止其他协程执行其代码。从技术上讲，这是有效的，但也破坏了异步编程的所有收益。因此，最终，将异步 I/O 与同步 I/O 结合起来不是一个选择。这是一种“全有或全无”的游戏。

另一个问题是长时间运行的 CPU 绑定操作。当您执行 I/O 操作时，释放控制权不是一个问题。当从文件系统或套接字中读取/写入时，您最终会等待，因此使用`await`是您能做的最好的事情。但是当您需要实际计算某些东西并且知道这将需要一段时间时该怎么办？当然，您可以将问题切分成几部分，并在每次推进工作时释放控制权。但很快您会发现这不是一个好的模式。这样做可能会使代码混乱，也不能保证良好的结果。时间切片应该是解释器或操作系统的责任。

那么，如果您有一些使长时间同步 I/O 操作的代码，而您无法或不愿意重写。或者当您需要在主要设计为异步 I/O 的应用程序中进行一些重型 CPU 绑定操作时该怎么办？嗯...您需要使用一种变通方法。我所说的变通方法是多线程或多进程。

这可能听起来不好，但有时最好的解决方案可能是我们试图逃避的解决方案。在 Python 中，对 CPU 密集型任务的并行处理总是使用多进程更好。如果设置正确并小心处理，多线程可以同样好地处理 I/O 操作（快速且没有太多资源开销）如`async`和`await`。

所以有时当你不知道该怎么办，当某些东西简单地不适合你的异步应用程序时，使用一段代码将它推迟到单独的线程或进程。你可以假装这是一个协程，释放控制权给事件循环，最终在结果准备好时处理结果。幸运的是，Python 标准库提供了`concurrent.futures`模块，它也与`asyncio`模块集成。这两个模块一起允许你安排在线程或额外进程中执行的阻塞函数，就像它们是异步非阻塞的协程一样。

### 执行者和未来

在我们看到如何将线程或进程注入异步事件循环之前，我们将更仔细地看一下`concurrent.futures`模块，这将成为我们所谓的变通方法的主要组成部分。

`concurrent.futures`模块中最重要的类是`Executor`和`Future`。

`Executor`代表一个可以并行处理工作项的资源池。这在目的上似乎与`multiprocessing`模块的`Pool`和`dummy.Pool`类非常相似，但它有完全不同的接口和语义。它是一个不打算实例化的基类，并且有两个具体的实现：

+   `ThreadPoolExecutor`：这个代表一个线程池

+   `ProcessPoolExecutor`：这个代表一个进程池

每个执行者提供三种方法：

+   `submit(fn, *args, **kwargs)`：这个方法安排`fn`函数在资源池上执行，并返回代表可调用执行的`Future`对象

+   `map(func, *iterables, timeout=None, chunksize=1)`：这个方法以类似于`multiprocessing.Pool.map()`方法的方式在可迭代对象上执行 func 函数

+   `shutdown(wait=True)`：这个方法关闭执行者并释放它的所有资源

最有趣的方法是`submit()`，因为它返回一个`Future`对象。它代表一个可调用的异步执行，间接代表它的结果。为了获得提交的可调用的实际返回值，你需要调用`Future.result()`方法。如果可调用已经完成，`result()`方法不会阻塞它，只会返回函数的输出。如果不是这样，它会阻塞直到结果准备好。把它当作一个结果的承诺（实际上它和 JavaScript 中的 promise 概念是一样的）。你不需要立即在接收到它后解包它（用`result()`方法），但如果你试图这样做，它保证最终会返回一些东西：

```py
>>> def loudy_return():
...     print("processing")
...     return 42
...** 
>>> from concurrent.futures import ThreadPoolExecutor
>>> with ThreadPoolExecutor(1) as executor:
...     future = executor.submit(loudy_return)
...** 
processing
>>> future
<Future at 0x33cbf98 state=finished returned int>
>>> future.result()
42

```

如果你想使用`Executor.map()`方法，它在用法上与`multiprocessing`模块的`Pool`类的`map()`方法没有区别：

```py
def main():
    with ThreadPoolExecutor(POOL_SIZE) as pool:
        results = pool.map(fetch_place, PLACES)

    for result in results:
        present_result(result)
```

### 在事件循环中使用执行者

`Executor.submit()`方法返回的`Future`类实例在概念上与异步编程中使用的协程非常接近。这就是为什么我们可以使用执行者来实现协作式多任务和多进程或多线程的混合。

这个变通方法的核心是事件循环类的`BaseEventLoop.run_in_executor(executor, func, *args)`方法。它允许你在由`executor`参数表示的进程或线程池中安排`func`函数的执行。这个方法最重要的一点是它返回一个新的*awaitable*（一个可以用`await`语句*await*的对象）。因此，由于这个方法，你可以执行一个阻塞函数，它不是一个协程，就像它是一个协程一样，无论它需要多长时间来完成，它都不会阻塞。它只会阻止等待这样一个调用结果的函数，但整个事件循环仍然会继续运转。

一个有用的事实是，您甚至不需要创建自己的执行器实例。如果将`None`作为执行器参数传递，将使用`ThreadPoolExecutor`类以默认线程数（对于 Python 3.5，它是处理器数量乘以 5）。

因此，让我们假设我们不想重写导致我们头疼的`python-gmaps`包的有问题的部分。我们可以通过`loop.run_in_executor()`调用轻松地将阻塞调用推迟到单独的线程，同时将`fetch_place()`函数保留为可等待的协程：

```py
async def fetch_place(place):
    coro = loop.run_in_executor(None, api.geocode, place)
    result = await coro
    return result[0]
```

这样的解决方案并不像拥有完全异步库来完成工作那样好，但您知道*半瓶水总比没有水好*。

# 总结

这是一段漫长的旅程，但我们成功地克服了 Python 程序员可用的并发编程的最基本方法。

在解释并发到底是什么之后，我们迅速行动起来，通过多线程的帮助解剖了典型的并发问题之一。在确定了我们代码的基本缺陷并加以修复后，我们转向了多进程，看看它在我们的情况下会如何运作。

我们发现，使用`multiprocessing`模块比使用`threading`的基本线程要容易得多。但就在那之后，我们意识到我们也可以使用相同的 API 来处理线程，多亏了`multiprocessing.dummy`。因此，现在在多进程和多线程之间的选择只是更适合问题的解决方案，而不是哪种解决方案具有更好的接口。

说到问题的适应性，我们最终尝试了异步编程，这应该是 I/O 密集型应用程序的最佳解决方案，只是意识到我们不能完全忘记线程和进程。所以我们又回到了起点！

这就引出了本章的最终结论。并没有银弹。有一些方法可能更受您喜欢。有一些方法可能更适合特定的问题集，但您需要了解它们，以便取得成功。在现实场景中，您可能会发现自己在单个应用程序中使用整套并发工具和风格，这并不罕见。

上述结论是下一章第十四章*有用的设计模式*主题的绝佳引言。这是因为没有单一的模式可以解决您所有的问题。您应该尽可能了解尽可能多的模式，因为最终您将每天都使用它们。
