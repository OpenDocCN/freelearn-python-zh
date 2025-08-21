# 第十章：并发执行

“我们想要什么？现在！我们什么时候想要？更少的竞争条件！”- Anna Melzer

在这一章中，我打算稍微提高一下游戏水平，无论是在我将要介绍的概念上，还是在我将向你展示的代码片段的复杂性上。如果你觉得任务太艰巨，或者在阅读过程中意识到它变得太困难，可以随时跳过。等你准备好了再回来。

计划是离开熟悉的单线程执行范式，深入探讨可以描述为并发执行的内容。我只能浅尝这个复杂的主题，所以我不指望你在阅读完之后就成为并发性的大师，但我会像往常一样，尽量给你足够的信息，这样你就可以继续“走上这条路”，可以这么说。

我们将学习适用于这个编程领域的所有重要概念，并且我会尝试向你展示以不同风格编码的示例，以便让你对这些主题的基础有扎实的理解。要深入研究这个具有挑战性和有趣的编程分支，你将不得不参考 Python 文档中的*并发执行*部分（[`docs.python.org/3.7/library/concurrency.html`](https://docs.python.org/3.7/library/concurrency.html)），也许还要通过学习相关书籍来补充你的知识。

特别是，我们将探讨以下内容：

+   线程和进程背后的理论

+   编写多线程代码

+   编写多进程代码

+   使用执行器来生成线程和进程

+   使用`asyncio`进行编程的简短示例

让我们先把理论搞清楚。

# 并发与并行

并发和并行经常被误解为相同的事物，但它们之间有区别。并发是同时运行多个任务的能力，不一定是并行的。并行是同时做多件事情的能力。

想象一下，你带着另一半去剧院。有两条队列：VIP 和普通票。只有一个工作人员检查票，为了避免阻塞两个队列中的任何一个，他们先检查 VIP 队列的一张票，然后检查普通队列的一张票。随着时间的推移，两个队列都被处理了。这是并发的一个例子。

现在想象一下，另一个工作人员加入了，所以现在每个队列都有一个工作人员。这样，每个队列都将由自己的工作人员处理。这是并行的一个例子。

现代笔记本电脑处理器具有多个核心（通常是两到四个）。核心是属于处理器的独立处理单元。拥有多个核心意味着所讨论的 CPU 实际上具有并行执行任务的物理能力。在每个核心内部，通常会有一系列工作流的不断交替，这是并发执行。

请记住，我在这里故意保持讨论的泛化。根据你使用的系统，执行处理方式会有所不同，所以我将集中讨论那些对所有系统或至少大多数系统都通用的概念。

# 线程和进程-概述

线程可以被定义为一系列指令，可以由调度程序运行，调度程序是操作系统的一部分，决定哪个工作块将获得必要的资源来执行。通常，一个线程存在于一个进程内。进程可以被定义为正在执行的计算机程序的一个实例。

在之前的章节中，我们用类似`$ python my_script.py`的命令运行我们自己的模块和脚本。当运行这样的命令时，会创建一个 Python 进程。在其中，会生成一个主执行线程。脚本中的指令将在该线程内运行。

这只是一种工作方式，Python 实际上可以在同一个进程中使用多个线程，甚至可以生成多个进程。毫不奇怪，这些计算机科学的分支被称为多线程和多进程。

为了理解区别，让我们花点时间稍微深入地探讨线程和进程。

# 线程的快速解剖

一般来说，有两种不同类型的线程：

+   用户级线程：我们可以创建和管理以执行任务的线程

+   内核级线程：在内核模式下运行并代表操作系统执行操作的低级线程

鉴于 Python 在用户级别上运行，我们暂时不会深入研究内核线程。相反，我们将在本章的示例中探索几个用户级线程的示例。

线程可以处于以下任何状态：

+   新线程：尚未启动，也没有分配任何资源的线程。

+   可运行：线程正在等待运行。它具有运行所需的所有资源，一旦调度程序给予它绿灯，它将运行。

+   运行：正在执行指令流的线程。从这种状态，它可以返回到非运行状态，或者死亡。

+   非运行：已暂停的线程。这可能是由于另一个线程优先于它，或者仅仅是因为线程正在等待长时间运行的 IO 操作完成。

+   死亡：线程已经死亡，因为它已经到达了其执行流的自然结束，或者它已经被终止。

状态之间的转换是由我们的操作或调度程序引起的。不过，有一件事要记住；最好不要干涉线程的死亡。

# 终止线程

终止线程并不被认为是良好的做法。Python 不提供通过调用方法或函数来终止线程的能力，这应该是终止线程不是你想要做的事情的暗示。

一个原因是线程可能有子线程——从线程本身内部生成的线程——当其父线程死亡时会成为孤儿。另一个原因可能是，如果您要终止的线程持有需要正确关闭的资源，您可能会阻止这种情况发生，这可能会导致问题。

稍后，我们将看到如何解决这些问题的示例。

# 上下文切换

我们已经说过调度程序可以决定何时运行线程，或者暂停线程等。任何时候运行的线程需要被暂停以便另一个线程可以运行时，调度程序会以一种方式保存运行线程的状态，以便在以后的某个时间恢复执行，恢复到暂停的地方。

这个行为被称为上下文切换。人们也经常这样做。我们正在做一些文书工作，然后听到手机上的*叮铃声！*。我们停下文书工作，查看手机。当我们处理完可能是第 n 张有趣猫的照片后，我们回到文书工作。不过，我们并不是从头开始文书工作；我们只是继续之前离开的地方。

上下文切换是现代计算机的奇妙能力，但如果生成了太多线程，它可能会变得麻烦。调度程序将尝试给每个线程一点时间来运行，并且将花费大量时间保存和恢复分别暂停和重新启动的线程的状态。

为了避免这个问题，限制可以在任何给定时间点运行的线程数量（同样的考虑也适用于进程）是相当常见的。这是通过使用一个称为池的结构来实现的，其大小可以由程序员决定。简而言之，我们创建一个池，然后将任务分配给它的线程。当池中的所有线程都忙碌时，程序将无法生成新的线程，直到其中一个终止（并返回到池中）。池对于节省资源也非常有用，因为它为线程生态系统提供了回收功能。

当你编写多线程代码时，了解软件将在哪台机器上运行的信息是很有用的。这些信息，再加上一些分析（我们将在第十一章 *调试和故障排除*中学习），应该能够让我们正确地校准我们的池的大小。

# 全局解释器锁

2015 年 7 月，我参加了在毕尔巴鄂举行的 EuroPython 大会，我在那里做了一个关于测试驱动开发的演讲。摄像机操作员不幸地丢失了其中的前半部分，但我后来又有机会再做了几次那个演讲，所以你可以在网上找到完整版本。在会议上，我有幸见到了 Guido van Rossum 并与他交谈，我还参加了他的主题演讲。

他谈到的一个话题是臭名昭著的**全局解释器锁**（**GIL**）。GIL 是一个互斥锁，用于保护对 Python 对象的访问，防止多个线程同时执行 Python 字节码。这意味着即使你可以在 Python 中编写多线程代码，但在任何时间点只有一个线程在运行（每个进程，当然）。

在计算机编程中，互斥对象（mutex）是一个允许多个程序线程共享相同资源（如文件访问）但不是同时的程序对象。

这通常被视为语言的不良限制，许多开发人员以诅咒这个伟大的反派为傲。然而，事实并非如此，正如 Raymond Hettinger 在 2017 年 PyBay 大会上的并发性主题演讲中所美妙地解释的那样（[`bit.ly/2KcijOB`](https://bit.ly/2KcijOB)）。大约 10 分钟后，Raymond 解释说，实际上很容易从 Python 中删除 GIL。这需要大约一天的工作。然而，你为此付出的代价是在代码中需要在需要的地方自行应用锁。这会导致更昂贵的印记，因为大量的个别锁需要更长的时间来获取和释放，最重要的是，它引入了错误的风险，因为编写健壮的多线程代码并不容易，你可能最终不得不编写几十甚至几百个锁。

为了理解锁是什么，以及为什么你可能想要使用它，我们首先需要谈谈多线程编程的危险之一：竞争条件。

# 竞争条件和死锁

当涉及编写多线程代码时，你需要意识到当你的代码不再被线性执行时会出现的危险。我的意思是，多线程代码有可能在任何时间点被调度程序暂停，因为它决定给另一个指令流一些 CPU 时间。

这种行为使你面临不同类型的风险，其中最著名的两种是竞争条件和死锁。让我们简要谈谈它们。

# 竞争条件

**竞争条件**是系统行为的一种，其中过程的输出取决于其他无法控制的事件的顺序或时间。当这些事件不按程序员预期的顺序展开时，竞争条件就会成为一个错误。

通过一个例子来解释这一点会更容易理解。

想象一下你有两个运行的线程。两者都在执行相同的任务，即从一个位置读取一个值，对该值执行一个操作，将该值增加*1*单位，然后保存回去。假设该操作是将该值发布到 API。

# 情景 A - 竞争条件不会发生

线程*A*读取值（*1*），将*1*发送到 API，然后将其增加到*2*，并保存回去。就在这之后，调度程序暂停了线程*A*，并运行了线程*B*。线程*B*读取值（现在是*2*），将*2*发送到 API，将其增加到*3*，然后保存回去。

在这一点上，即使操作发生了两次，存储的值也是正确的：*1 + 2 = 3*。此外，API 已经正确地被调用了两次，分别是*1*和*2*。

# 情景 B - 竞争条件发生

线程*A*读取值（*1*），将其发送到 API，将其增加到*2*，但在它保存回去之前，调度程序决定暂停线程*A*，转而执行线程*B*。

线程*B*读取值（仍然是*1*！），将其发送到 API，将其增加到*2*，然后保存回去。然后调度程序再次切换到线程*A*。线程*A*通过简单保存增加后的值（*2*）来恢复其工作流。

在这种情况下，即使操作像情景 A 中发生了两次，保存的值是*2*，API 也被调用了两次，每次都是*1*。

在现实生活中，有多个线程和真实代码执行多个操作的情况下，程序的整体行为会爆炸成无数可能性。我们稍后会看到一个例子，并使用锁来解决它。

竞争条件的主要问题在于它使我们的代码变得不确定，这是不好的。在计算机科学中有一些领域使用了非确定性来实现某些目标，这是可以接受的，但通常情况下，你希望能够预测代码的行为，而竞争条件使这变得不可能。

# 锁来拯救

在处理竞争条件时，锁会拯救我们。例如，为了修复前面的例子，你只需要在该过程周围加上一个锁。锁就像一个守护者，只允许一个线程拿住它（我们说*获取*锁），并且直到该线程释放锁，其他线程都无法获取它。它们必须坐下等待，直到锁再次可用。

# 情景 C - 使用锁

线程*A*获取锁，读取值（*1*），发送到 API，增加到*2*，然后调度程序将其挂起。线程*B*获得了一些 CPU 时间，所以它尝试获取锁。但是锁还没有被线程*A*释放，所以线程*B*等待。调度程序可能会注意到这一点，并迅速决定切换回线程*A*。

线程*A*保存 2，并释放锁，使其对所有其他线程可用。

在这一点上，无论是线程*A*再次获取锁，还是线程*B*获取锁（因为调度程序可能已经决定再次切换），都不重要。该过程将始终被正确执行，因为锁确保当一个线程读取一个值时，它必须在任何其他线程也能读取该值之前完成该过程（ping API，增加和保存）。

标准库中有许多不同的锁可用。我绝对鼓励你阅读它们，以了解在编写多线程代码时可能遇到的所有危险，以及如何解决它们。

现在让我们谈谈死锁。

# 死锁

**死锁**是一种状态，在这种状态下，组中的每个成员都在等待其他成员采取行动，例如发送消息，更常见的是释放锁或资源。

一个简单的例子将帮助你理解。想象两个小孩在一起玩。找一个由两部分组成的玩具，给他们每人一部分。自然地，他们中没有一个会想把自己的那部分给另一个，他们会想让另一个释放他们手中的那部分。因此，他们中没有一个能够玩这个玩具，因为他们每个人都握着一半，会无限期地等待另一个孩子释放另一半。

别担心，在制作这个例子的过程中没有伤害到任何孩子。这一切都发生在我的脑海中。

另一个例子可能是让两个线程再次执行相同的过程。该过程需要获取两个资源，*A*和*B*，分别由单独的锁保护。线程*1*获取*A*，线程*2*获取*B*，然后它们将无限期地等待，直到另一个释放它所拥有的资源。但这不会发生，因为它们都被指示等待并获取第二个资源以完成该过程。线程可能比孩子更倔强。

你可以用几种方法解决这个问题。最简单的方法可能就是对资源获取应用顺序，这意味着获得*A*的线程也会获得其余的*B*、*C*等等。

另一种方法是在整个资源获取过程周围加锁，这样即使可能发生顺序错误，它仍然会在锁的上下文中进行，这意味着一次只有一个线程可以实际获取所有资源。

现在让我们暂停一下关于线程的讨论，来探讨进程。

# 进程的简单解剖

进程通常比线程更复杂。一般来说，它们包含一个主线程，但如果你选择的话也可以是多线程的。它们能够生成多个子线程，每个子线程都包含自己的寄存器和堆栈。每个进程都提供计算机执行程序所需的所有资源。

与使用多个线程类似，我们可以设计我们的代码以利用多进程设计。多个进程可能在多个核心上运行，因此使用多进程可以真正并行计算。然而，它们的内存占用略高于线程的内存占用，使用多个进程的另一个缺点是**进程间通信**（IPC）往往比线程间通信更昂贵。

# 进程的属性

UNIX 进程是由操作系统创建的。它通常包含以下内容：

+   进程 ID、进程组 ID、用户 ID 或组 ID

+   一个环境和工作目录

+   程序指令

+   寄存器、堆栈和堆

+   文件描述符

+   信号动作

+   共享库

+   进程间通信工具（管道、消息队列、信号量或共享内存）

如果你对进程感兴趣，打开一个 shell 并输入`$ top`。这个命令会显示并更新有关系统中正在运行的进程的排序信息。当我在我的机器上运行它时，第一行告诉我以下信息：

```py
$ top
Processes: 477 total, 4 running, 473 sleeping, 2234 threads
...
```

这让你对我们的计算机在我们并不真正意识到的情况下做了多少工作有了一个概念。

# 多线程还是多进程？

考虑到所有这些信息，决定哪种方法是最好的意味着要了解需要执行的工作类型，并且要了解将要专门用于执行该工作的系统。

这两种方法都有优势，所以让我们试着澄清一下主要的区别。

以下是使用多线程的一些优势：

+   线程都是在同一个进程中诞生的。它们共享资源，并且可以非常容易地相互通信。进程之间的通信需要更复杂的结构和技术。

+   生成线程的开销比生成进程的开销小。此外，它们的内存占用也更小。

+   线程在阻塞 IO 密集型应用程序方面非常有效。例如，当一个线程被阻塞等待网络连接返回一些数据时，工作可以轻松有效地切换到另一个线程。

+   因为进程之间没有共享资源，所以我们需要使用 IPC 技术，而且它们需要比线程之间通信更多的内存。

以下是使用多进程的一些优势：

+   我们可以通过使用进程来避免 GIL 的限制。

+   失败的子进程不会终止主应用程序。

+   线程存在诸如竞争条件和死锁等问题；而使用进程时，需要处理这些问题的可能性大大降低。

+   当线程数量超过一定阈值时，线程的上下文切换可能变得非常昂贵。

+   进程可以更好地利用多核处理器。

+   进程比多线程更擅长处理 CPU 密集型任务。

在本章中，我将为您展示多个示例的两种方法，希望您能对各种不同的技术有一个很好的理解。那么让我们开始编码吧！

# Python 中的并发执行

让我们从一些简单的例子开始，探索 Python 多线程和多进程的基础知识。

请记住，以下示例中的几个将产生取决于特定运行的输出。处理线程时，事情可能变得不确定，就像我之前提到的那样。因此，如果您遇到不同的结果，那是完全正常的。您可能会注意到，您的一些结果也会从一次运行到另一次运行有所不同。

# 开始一个线程

首先，让我们开始一个线程：

```py
# start.py
import threading

def sum_and_product(a, b):
    s, p = a + b, a * b
    print(f'{a}+{b}={s}, {a}*{b}={p}')

t = threading.Thread(
    target=sum_and_product, name='SumProd', args=(3, 7)
)
t.start()
```

在导入`threading`之后，我们定义一个函数：`sum_and_product`。这个函数计算两个数字的和和积，并打印结果。有趣的部分在函数之后。我们从`threading.Thread`实例化了`t`。这是我们的线程。我们传递了将作为线程主体运行的函数的名称，给它一个名称，并传递了参数`3`和`7`，它们将分别作为`a`和`b`传递到函数中。

创建了线程之后，我们使用同名方法启动它。

此时，Python 将在一个新线程中开始执行函数，当该操作完成时，整个程序也将完成，并退出。让我们运行它：

```py
$ python start.py
3+7=10, 3*7=21 
```

因此，开始一个线程非常简单。让我们看一个更有趣的例子，其中我们显示更多信息：

```py
# start_with_info.py
import threading
from time import sleep

def sum_and_product(a, b):
    sleep(.2)
    print_current()
    s, p = a + b, a * b
    print(f'{a}+{b}={s}, {a}*{b}={p}')

def status(t):
    if t.is_alive():
        print(f'Thread {t.name} is alive.')
    else:
        print(f'Thread {t.name} has terminated.')

def print_current():
    print('The current thread is {}.'.format(
        threading.current_thread()
    ))
    print('Threads: {}'.format(list(threading.enumerate())))

print_current()
t = threading.Thread(
    target=sum_and_product, name='SumPro', args=(3, 7)
)
t.start()
status(t)
t.join()
status(t)
```

在这个例子中，线程逻辑与之前的完全相同，所以你不需要为此而劳累，可以专注于我添加的（疯狂的！）大量日志信息。我们使用两个函数来显示信息：`status`和`print_current`。第一个函数接受一个线程作为输入，并通过调用其`is_alive`方法显示其名称以及它是否存活。第二个函数打印当前线程，然后枚举进程中的所有线程。这些信息来自`threading.current_thread`和`threading.enumerate`。

我在函数内部放置了`.2`秒的睡眠时间是有原因的。当线程启动时，它的第一条指令是休眠一会儿。调皮的调度程序会捕捉到这一点，并将执行切换回主线程。您可以通过输出中看到，在线程内部的`status(t)`的结果之前，您将看到`print_current`的结果。这意味着这个调用发生在线程休眠时。

最后，请注意我在最后调用了`t.join()`。这指示 Python 阻塞，直到线程完成。这是因为我希望最后一次对`status(t)`的调用告诉我们线程已经结束。让我们来看一下输出（为了可读性稍作调整）：

```py
$ python start_with_info.py
The current thread is
 <_MainThread(MainThread, started 140735733822336)>.
Threads: [<_MainThread(MainThread, started 140735733822336)>]
Thread SumProd is alive.
The current thread is <Thread(SumProd, started 123145375604736)>.
Threads: [
 <_MainThread(MainThread, started 140735733822336)>,
 <Thread(SumProd, started 123145375604736)>
]
3+7=10, 3*7=21
Thread SumProd has terminated.
```

正如你所看到的，一开始当前线程是主线程。枚举只显示一个线程。然后我们创建并启动`SumProd`。我们打印它的状态，我们得知它还活着。然后，这一次是从`SumProd`内部，我们再次显示当前线程的信息。当然，现在当前线程是`SumProd`，我们可以看到枚举所有线程返回了两个。打印结果后，我们通过最后一次对`status`的调用验证线程是否已经终止，正如预期的那样。如果你得到不同的结果（当然除了线程的 ID 之外），尝试增加睡眠时间，看看是否有任何变化。

# 启动一个进程

现在让我们看一个等价的例子，但是不使用线程，而是使用进程：

```py
# start_proc.py
import multiprocessing

...

p = multiprocessing.Process(
    target=sum_and_product, name='SumProdProc', args=(7, 9)
)
p.start()
```

代码与第一个示例完全相同，但我们实例化`multiprocessing.Process`而不是使用`Thread`。`sum_and_product`函数与以前相同。输出也是相同的，只是数字不同。

# 停止线程和进程

如前所述，一般来说，停止线程是一个坏主意，进程也是一样。确保你已经注意到处理和关闭所有打开的东西可能会非常困难。然而，有些情况下你可能希望能够停止一个线程，所以让我告诉你如何做：

```py
# stop.py
import threading
from time import sleep

class Fibo(threading.Thread):
    def __init__(self, *a, **kwa):
        super().__init__(*a, **kwa)
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        a, b = 0, 1
        while self._running:
            print(a, end=' ')
            a, b = b, a + b
            sleep(0.07)
        print()

fibo = Fibo()
fibo.start()
sleep(1)
fibo.stop()
fibo.join()
print('All done.')
```

对于这个例子，我们使用一个斐波那契生成器。我们之前见过它，所以我不会解释它。要关注的重要部分是`_running`属性。首先要注意的是类继承自`Thread`。通过重写`__init__`方法，我们可以将`_running`标志设置为`True`。当你以这种方式编写线程时，而不是给它一个目标函数，你只需在类中重写`run`方法。我们的`run`方法计算一个新的斐波那契数，然后睡眠约`0.07`秒。

在最后一段代码中，我们创建并启动了一个类的实例。然后我们睡眠一秒钟，这应该给线程时间产生大约 14 个斐波那契数。当我们调用`fibo.stop()`时，我们实际上并没有停止线程。我们只是将我们的标志设置为`False`，这允许`run`中的代码达到自然的结束。这意味着线程将自然死亡。我们调用`join`来确保线程在我们在控制台上打印`All done.`之前实际完成。让我们检查输出：

```py
$ python stop.py
0 1 1 2 3 5 8 13 21 34 55 89 144 233
All done.
```

检查打印了多少个数字：14，正如预期的那样。

这基本上是一种解决技术，允许你停止一个线程。如果你根据多线程范例正确设计你的代码，你就不应该总是不得不杀死线程，所以让这种需要成为你设计更好的警钟。

# 停止一个进程

当涉及到停止一个进程时，情况就不同了，而且没有麻烦。你可以使用`terminate`或`kill`方法，但请确保你知道自己在做什么，因为之前关于悬挂的开放资源的所有考虑仍然是正确的。

# 生成多个线程

只是为了好玩，现在让我们玩两个线程：

```py
# starwars.py
import threading
from time import sleep
from random import random

def run(n):
    t = threading.current_thread()
    for count in range(n):
        print(f'Hello from {t.name}! ({count})')
        sleep(0.2 * random())

obi = threading.Thread(target=run, name='Obi-Wan', args=(4, ))
ani = threading.Thread(target=run, name='Anakin', args=(3, ))
obi.start()
ani.start()
obi.join()
ani.join()
```

`run`函数简单地打印当前线程，然后进入`n`个周期的循环，在循环中打印一个问候消息，并睡眠一个随机的时间，介于`0`和`0.2`秒之间（`random()`返回一个介于`0`和`1`之间的浮点数）。

这个例子的目的是向你展示调度程序可能在线程之间跳转，所以让它们睡一会儿会有所帮助。让我们看看输出：

```py
$ python starwars.py
Hello from Obi-Wan! (0)
Hello from Anakin! (0)
Hello from Obi-Wan! (1)
Hello from Obi-Wan! (2)
Hello from Anakin! (1)
Hello from Obi-Wan! (3)
Hello from Anakin! (2)
```

正如你所看到的，输出在两者之间随机交替。每当发生这种情况时，你就知道调度程序已经执行了上下文切换。

# 处理竞争条件

现在我们有了启动线程和运行它们的工具，让我们模拟一个竞争条件，比如我们之前讨论过的条件：

```py
# race.py
import threading
from time import sleep
from random import random

counter = 0
randsleep = lambda: sleep(0.1 * random())

def incr(n):
    global counter
    for count in range(n):
        current = counter
        randsleep()
        counter = current + 1
        randsleep()

n = 5
t1 = threading.Thread(target=incr, args=(n, ))
t2 = threading.Thread(target=incr, args=(n, ))
t1.start()
t2.start()
t1.join()
t2.join()
print(f'Counter: {counter}')
```

在这个例子中，我们定义了`incr`函数，它接收一个数字`n`作为输入，并循环`n`次。在每个循环中，它读取计数器的值，通过调用我编写的一个小的 Lambda 函数`randsleep`来随机休眠一段时间（在`0`和`0.1`秒之间），然后将`counter`的值增加`1`。

我选择使用`global`来读/写`counter`，但实际上可以是任何东西，所以请随意尝试。

整个脚本基本上启动了两个线程，每个线程运行相同的函数，并获得`n = 5`。请注意，我们需要在最后加入两个线程的连接，以确保当我们打印计数器的最终值（最后一行）时，两个线程都完成了它们的工作。

当我们打印最终值时，我们期望计数器是 10，对吧？两个线程，每个循环五次，这样就是 10。然而，如果我们运行这个脚本，我们几乎永远不会得到 10。我自己运行了很多次，似乎总是在 5 和 7 之间。发生这种情况的原因是这段代码中存在竞争条件，我添加的随机休眠是为了加剧这种情况。如果你删除它们，仍然会存在竞争条件，因为计数器的增加是非原子的（这意味着一个可以被分解成多个步骤的操作，因此在其中间可以暂停）。然而，竞争条件发生的可能性非常低，所以添加随机休眠有所帮助。

让我们分析一下代码。`t1`获取计数器的当前值，比如`3`。然后，`t1`暂停一会儿。如果调度程序在那一刻切换上下文，暂停`t1`并启动`t2`，`t2`将读取相同的值`3`。无论之后发生什么，我们知道两个线程都将更新计数器为`4`，这是不正确的，因为在两次读取后，它应该已经增加到`5`。在更新后添加第二个随机休眠调用有助于调度程序更频繁地切换，并且更容易显示竞争条件。尝试注释掉其中一个，看看结果如何改变（它会发生戏剧性的变化）。

现在我们已经确定了问题，让我们通过使用锁来解决它。代码基本上是一样的，所以我只会向您展示发生了什么变化：

```py
# race_with_lock.py
incr_lock = threading.Lock()

def incr(n):
    global counter
    for count in range(n):
        with incr_lock:
            current = counter
            randsleep()
            counter = current + 1
            randsleep()
```

这一次我们创建了一个锁，来自`threading.Lock`类。我们可以手动调用它的`acquire`和`release`方法，或者我们可以使用上下文管理器在其中使用它，这看起来更好，而且可以为我们完成整个获取/释放的工作。请注意，我在代码中保留了随机休眠。然而，每次运行它，它现在会返回`10`。

区别在于：当第一个线程获取该锁时，即使它在睡眠时，调度程序稍后切换上下文也无所谓。第二个线程将尝试获取锁，Python 会坚决拒绝。因此，第二个线程将一直等待，直到锁被释放。一旦调度程序切换回第一个线程并释放锁，那么另一个线程将有机会（如果它首先到达那里，这并不一定保证）获取锁并更新计数器。尝试在该逻辑中添加一些打印，看看线程是否完美交替。我猜想它们不会，至少不是每次都会。记住`threading.current_thread`函数，以便能够看到哪个线程实际上打印了信息。

Python 在`threading`模块中提供了几种数据结构：Lock、RLock、Condition、Semaphore、Event、Timer 和 Barrier。我无法向您展示所有这些，因为不幸的是，我没有足够的空间来解释所有的用例，但阅读`threading`模块的文档（[`docs.python.org/3.7/library/threading.html`](https://docs.python.org/3.7/library/threading.html)）将是开始理解它们的好地方。

现在让我们看一个关于线程本地数据的例子。

# 线程的本地数据

`threading`模块提供了一种为线程实现本地数据的方法。本地数据是一个保存特定于线程的数据的对象。让我给你展示一个例子，并且让我偷偷加入一个`Barrier`，这样我就可以告诉你它是如何工作的：

```py
# local.py
import threading
from random import randint

local = threading.local()

def run(local, barrier):
    local.my_value = randint(0, 10**2)
    t = threading.current_thread()
    print(f'Thread {t.name} has value {local.my_value}')
    barrier.wait()
    print(f'Thread {t.name} still has value {local.my_value}')

count = 3
barrier = threading.Barrier(count)
threads = [
    threading.Thread(
        target=run, name=f'T{name}', args=(local, barrier)
    ) for name in range(count)
]
for t in threads:
    t.start()
```

我们首先定义`local`。这是保存特定于线程的数据的特殊对象。我们运行三个线程。它们中的每一个都将一个随机值赋给`local.my_value`，并将其打印出来。然后线程到达一个`Barrier`对象，它被编程为总共容纳三个线程。当第三个线程碰到屏障时，它们都可以通过。这基本上是一种确保*N*个线程达到某一点并且它们都等待，直到每一个都到达的好方法。

现在，如果`local`是一个普通的虚拟对象，第二个线程将覆盖`local.my_value`的值，第三个线程也会这样做。这意味着我们会看到它们在第一组打印中打印不同的值，但在第二组打印中它们将显示相同的值（最后一个）。但由于`local`的存在，这种情况不会发生。输出显示如下：

```py
$ python local.py
Thread T0 has value 61
Thread T1 has value 52
Thread T2 has value 38
Thread T2 still has value 38
Thread T0 still has value 61
Thread T1 still has value 52
```

注意错误的顺序，由于调度程序切换上下文，但所有值都是正确的。

# 线程和进程通信

到目前为止，我们已经看到了很多例子。所以，让我们探讨如何通过使用队列使线程和进程相互通信。让我们从线程开始。

# 线程通信

在这个例子中，我们将使用`queue`模块中的普通`Queue`：

```py
# comm_queue.py
import threading
from queue import Queue

SENTINEL = object()

def producer(q, n):
    a, b = 0, 1
    while a <= n:
        q.put(a)
        a, b = b, a + b
    q.put(SENTINEL)

def consumer(q):
    while True:
        num = q.get()
        q.task_done()
        if num is SENTINEL:
            break
        print(f'Got number {num}')

q = Queue()
cns = threading.Thread(target=consumer, args=(q, ))
prd = threading.Thread(target=producer, args=(q, 35))
cns.start()
prd.start()
q.join()
```

逻辑非常基本。我们有一个`producer`函数，它生成斐波那契数并将它们放入队列中。当下一个数字大于给定的`n`时，生产者退出`while`循环，并在队列中放入最后一件事：一个`SENTINEL`。`SENTINEL`是用来发出信号的任何对象，在我们的例子中，它向消费者发出信号，表示生产者已经完成。

有趣的逻辑部分在`consumer`函数中。它无限循环，从队列中读取值并将其打印出来。这里有几件事情需要注意。首先，看看我们如何调用`q.task_done()`？这是为了确认队列中的元素已被处理。这样做的目的是允许代码中的最后一条指令`q.join()`在所有元素都被确认时解除阻塞，以便执行可以结束。

其次，注意我们如何使用`is`运算符来与项目进行比较，以找到哨兵。我们很快会看到，当使用`multiprocessing.Queue`时，这将不再可能。在我们到达那里之前，你能猜到为什么吗？

运行这个例子会产生一系列行，比如`Got number 0`，`Got number 1`，依此类推，直到`34`，因为我们设置的限制是`35`，下一个斐波那契数将是`55`。

# 发送事件

另一种使线程通信的方法是触发事件。让我快速给你展示一个例子：

```py
# evt.py
import threading

def fire():
    print('Firing event...')
    event.set()

def listen():
    event.wait()
    print('Event has been fired')

event = threading.Event()
t1 = threading.Thread(target=fire)
t2 = threading.Thread(target=listen)
t2.start()
t1.start()
```

这里有两个线程分别运行`fire`和`listen`，分别触发和监听事件。要触发事件，调用`set`方法。首先启动的`t2`线程已经在监听事件，直到事件被触发。前面例子的输出如下：

```py
$ python evt.py
Firing event...
Event has been fired
```

在某些情况下，事件非常有用。想象一下，有一些线程正在等待连接对象准备就绪，然后才能开始使用它。它们可以等待事件，一个线程可以检查该连接，并在准备就绪时触发事件。事件很有趣，所以确保你进行实验，并考虑它们的用例。

# 使用队列进行进程间通信

让我们现在看看如何使用队列在进程之间进行通信。这个例子非常类似于线程的例子：

```py
# comm_queue_proc.py
import multiprocessing

SENTINEL = 'STOP'

def producer(q, n):
    a, b = 0, 1
    while a <= n:
        q.put(a)
        a, b = b, a + b
    q.put(SENTINEL)

def consumer(q):
    while True:
        num = q.get()
        if num == SENTINEL:
            break
        print(f'Got number {num}')

q = multiprocessing.Queue()
cns = multiprocessing.Process(target=consumer, args=(q, ))
prd = multiprocessing.Process(target=producer, args=(q, 35))
cns.start()
prd.start()
```

如您所见，在这种情况下，我们必须使用`multiprocessing.Queue`的实例作为队列，它不公开`task_done`方法。但是，由于这个队列的设计方式，它会自动加入主线程，因此我们只需要启动两个进程，一切都会正常工作。这个示例的输出与之前的示例相同。

在 IPC 方面，要小心。对象在进入队列时被 pickled，因此 ID 丢失，还有一些其他微妙的事情要注意。这就是为什么在这个示例中，我不能再使用对象作为 sentinel，并使用`is`进行比较，就像我在多线程版本中所做的那样。这个 sentinel 对象将在队列中被 pickled（因为这次`Queue`来自`multiprocessing`而不是之前的`queue`），并且在 unpickling 后会假定一个新的 ID，无法正确比较。在这种情况下，字符串`"STOP"`就派上了用场，你需要找到一个适合的 sentinel 值，它需要是永远不会与队列中的任何项目发生冲突的值。我把这留给你去参考文档，并尽可能多地了解这个主题。

队列不是进程之间通信的唯一方式。您还可以使用管道（`multiprocessing.Pipe`），它提供了从一个进程到另一个进程的连接（显然是管道），反之亦然。您可以在文档中找到大量示例；它们与我们在这里看到的并没有太大的不同。

# 线程和进程池

如前所述，池是设计用来保存*N*个对象（线程、进程等）的结构。当使用达到容量时，不会将工作分配给线程（或进程），直到其中一个当前正在工作的线程再次可用。因此，池是限制同时可以活动的线程（或进程）数量的绝佳方式，防止系统因资源耗尽而饥饿，或者计算时间受到过多的上下文切换的影响。

在接下来的示例中，我将利用`concurrent.futures`模块来使用`ThreadPoolExecutor`和`ProcessPoolExecutor`执行器。这两个类使用线程池（和进程池），以异步方式执行调用。它们都接受一个参数`max_workers`，它设置了执行器同时可以使用多少个线程（或进程）的上限。

让我们从多线程示例开始：

```py
# pool.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import randint
import threading

def run(name):
    value = randint(0, 10**2)
    tname = threading.current_thread().name
    print(f'Hi, I am {name} ({tname}) and my value is {value}')
    return (name, value)

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(run, f'T{name}') for name in range(5)
    ]
    for future in as_completed(futures):
        name, value = future.result()
        print(f'Thread {name} returned {value}')
```

在导入必要的部分之后，我们定义了`run`函数。它获取一个随机值，打印它，并返回它，以及它被调用时的`name`参数。有趣的部分就在函数之后。

如您所见，我们使用上下文管理器调用`ThreadPoolExecutor`，我们传递`max_workers=3`，这意味着池大小为`3`。这意味着任何时候只有三个线程是活动的。

我们通过使用列表推导式定义了一个未来对象列表，在其中我们在执行器对象上调用`submit`。我们指示执行器运行`run`函数，名称将从`T0`到`T4`。`future`是一个封装可调用异步执行的对象。

然后我们循环遍历`future`对象，因为它们已经完成。为此，我们使用`as_completed`来获取`future`实例的迭代器，它们在完成（完成或被取消）时立即返回。我们通过调用同名方法来获取每个`future`的结果，并简单地打印它。鉴于`run`返回一个元组`name`，`value`，我们期望结果是包含`name`和`value`的两元组。如果我们打印`run`的输出（请记住每个`run`可能略有不同），我们会得到：

```py
$ python pool.py
Hi, I am T0 (ThreadPoolExecutor-0_0) and my value is 5
Hi, I am T1 (ThreadPoolExecutor-0_0) and my value is 23
Hi, I am T2 (ThreadPoolExecutor-0_1) and my value is 58
Thread T1 returned 23
Thread T0 returned 5
Hi, I am T3 (ThreadPoolExecutor-0_0) and my value is 93
Hi, I am T4 (ThreadPoolExecutor-0_1) and my value is 62
Thread T2 returned 58
Thread T3 returned 93
Thread T4 returned 62
```

在继续阅读之前，你能告诉我为什么输出看起来像这样吗？你能解释发生了什么吗？花点时间思考一下。

所以，发生的是三个线程开始运行，所以我们得到三个“嗨，我是…”消息被打印出来。一旦它们都在运行，池就满了，所以我们需要等待至少一个线程完成，然后才能发生其他事情。在示例运行中，T0 和 T2 完成了（这是通过打印它们返回的内容来表示），所以它们返回到池中可以再次使用。它们被命名为 T3 和 T4，并最终所有三个 T1、T3 和 T4 都完成了。您可以从输出中看到线程是如何被实际重用的，以及前两个在完成后如何被重新分配给 T3 和 T4。

现在让我们看看相同的例子，但使用多进程设计：

```py
# pool_proc.py
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import randint
from time import sleep

def run(name):
    sleep(.05)
    value = randint(0, 10**2)
    print(f'Hi, I am {name} and my value is {value}')
    return (name, value)

with ProcessPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(run, f'P{name}') for name in range(5)
    ]
    for future in as_completed(futures):
        name, value = future.result()
        print(f'Process {name} returned {value}')
```

差异真的是微乎其微。这次我们使用 ProcessPoolExecutor，并且 run 函数完全相同，只是增加了一个小细节：在每次运行开始时我们休眠 50 毫秒。这是为了加剧行为并清楚地显示池的大小，仍然是三。如果我们运行示例，我们得到：

```py
$ python pool_proc.py
Hi, I am P0 and my value is 19
Hi, I am P1 and my value is 97
Hi, I am P2 and my value is 74
Process P0 returned 19
Process P1 returned 97
Process P2 returned 74
Hi, I am P3 and my value is 80
Hi, I am P4 and my value is 68
Process P3 returned 80
Process P4 returned 68
```

这个输出清楚地显示了池的大小为三。有趣的是，如果我们去掉对 sleep 的调用，大多数情况下输出将有五次打印“嗨，我是…”，然后是五次打印“进程 Px 返回…”。我们如何解释这个呢？很简单。当前三个进程完成并由 as_completed 返回时，所有三个都被要求返回它们的结果，无论返回什么，都会被打印出来。在这个过程中，执行器已经可以开始回收两个进程来运行最后两个任务，它们恰好在允许 for 循环中的打印发生之前打印它们的“嗨，我是…”消息。

这基本上意味着 ProcessPoolExecutor 非常快速和积极（在获取调度程序的注意方面），值得注意的是，这种行为在线程对应的情况下不会发生，如果您还记得，我们不需要使用任何人为的睡眠。

然而，要记住的重要事情是，即使是这样简单的例子，也可能稍微难以理解或解释。让这成为你的一课，这样你在为多线程或多进程设计编码时就能提高你的注意力到 110%。

现在让我们转到一个更有趣的例子。

# 使用一个过程为函数添加超时

大多数，如果不是所有，公开函数以进行 HTTP 请求的库，在执行请求时提供指定超时的能力。这意味着如果在*X*秒后（*X*是超时时间），请求还没有完成，整个操作将被中止，并且执行将从下一条指令继续。不过，并非所有函数都提供这个功能，所以当一个函数没有提供中断的能力时，我们可以使用一个过程来模拟这种行为。在这个例子中，我们将尝试将主机名翻译成 IPv4 地址。然而，socket 模块的 gethostbyname 函数不允许我们在操作上设置超时，所以我们使用一个过程来人为地实现。接下来的代码可能不那么直接，所以我鼓励您在阅读解释之前花一些时间去理解它：

```py
# hostres/util.py
import socket
from multiprocessing import Process, Queue

def resolve(hostname, timeout=5):
    exitcode, ip = resolve_host(hostname, timeout)
    if exitcode == 0:
        return ip
    else:
        return hostname

def resolve_host(hostname, timeout):
    queue = Queue()
    proc = Process(target=gethostbyname, args=(hostname, queue))
    proc.start()
    proc.join(timeout=timeout)

    if queue.empty():
        proc.terminate()
        ip = None
    else:
        ip = queue.get()
    return proc.exitcode, ip

def gethostbyname(hostname, queue):
    ip = socket.gethostbyname(hostname)
    queue.put(ip)
```

让我们从 resolve 开始。它只是接受一个主机名和一个超时时间，并用它们调用 resolve_host。如果退出代码是 0（这意味着进程正确终止），它返回对应于该主机的 IPv4。否则，它将主机名本身作为后备机制返回。

接下来，让我们谈谈 gethostbyname。它接受一个主机名和一个队列，并调用 socket.gethostbyname 来解析主机名。当结果可用时，它被放入队列。现在问题就出在这里。如果对 socket.gethostbyname 的调用时间超过我们想要分配的超时时间，我们需要终止它。

`resolve_host`函数正是这样做的。它接收`hostname`和`timeout`，起初只是创建一个`queue`。然后它生成一个以`gethostbyname`为`target`的新进程，并传递适当的参数。然后启动进程并加入，但带有一个`timeout`。

现在，成功的情况是这样的：对`socket.gethostbyname`的调用很快成功，IP 在队列中，进程在超时时间之前成功终止，当我们到达`if`部分时，队列不会为空。我们从中获取 IP，并返回它，以及进程退出代码。

在失败的情况下，对`socket.gethostbyname`的调用时间太长，进程在超时后被终止。因为调用失败，没有 IP 被插入到队列中，因此队列将为空。在`if`逻辑中，我们将 IP 设置为`None`，并像以前一样返回。`resolve`函数会发现退出代码不是`0`（因为进程不是幸福地终止，而是被杀死），并且将正确地返回主机名而不是 IP，我们无论如何都无法获取 IP。

在本章的源代码中，在本章的`hostres`文件夹中，我添加了一些测试，以确保这种行为是正确的。你可以在文件夹中的`README.md`文件中找到如何运行它们的说明。确保你也检查一下测试代码，它应该会很有趣。

# 案例示例

在本章的最后部分，我将向你展示三个案例，我们将看到如何通过采用不同的方法（单线程、多线程和多进程）来做同样的事情。最后，我将专门介绍`asyncio`，这是一个在 Python 中引入另一种异步编程方式的模块。

# 例一 - 并发归并排序

第一个例子将围绕归并排序算法展开。这种排序算法基于“分而治之”设计范式。它的工作方式非常简单。你有一个要排序的数字列表。第一步是将列表分成两部分，对它们进行排序，然后将结果合并成一个排序好的列表。让我用六个数字举个简单的例子。假设我们有一个列表，`v=[8, 5, 3, 9, 0, 2]`。第一步是将列表`v`分成两个包含三个数字的子列表：`v1=[8, 5, 3]`和`v2=[9, 0, 2]`。然后我们通过递归调用归并排序对`v1`和`v2`进行排序。结果将是`v1=[3, 5, 8]`和`v2=[0, 2, 9]`。为了将`v1`和`v2`合并成一个排序好的`v`，我们只需考虑两个列表中的第一个项目，并选择其中的最小值。第一次迭代会比较`3`和`0`。我们选择`0`，留下`v2=[2, 9]`。然后我们重复这个过程：比较`3`和`2`，我们选择`2`，现在`v2=[9]`。然后我们比较`3`和`9`。这次我们选择`3`，留下`v1=[5, 8]`，依此类推。接下来我们会选择`5`（`5`与`9`比较），然后选择`8`（`8`与`9`比较），最后选择`9`。这将给我们一个新的、排序好的`v`：`v=[0, 2, 3, 5, 8, 9]`。

我选择这个算法作为例子的原因有两个。首先，它很容易并行化。你将列表分成两部分，让两个进程对它们进行处理，然后收集结果。其次，可以修改算法，使其将初始列表分成任意*N ≥ 2*，并将这些部分分配给*N*个进程。重新组合就像处理两个部分一样简单。这个特性使它成为并发实现的一个很好的候选。

# 单线程归并排序

让我们看看所有这些是如何转化为代码的，首先学习如何编写我们自己的自制`mergesort`：

```py
# ms/algo/mergesort.py
def sort(v):
    if len(v) <= 1:
        return v
    mid = len(v) // 2
    v1, v2 = sort(v[:mid]), sort(v[mid:])
    return merge(v1, v2)

def merge(v1, v2):
    v = []
    h = k = 0
    len_v1, len_v2 = len(v1), len(v2)
    while h < len_v1 or k < len_v2:
        if k == len_v2 or (h < len_v1 and v1[h] < v2[k]):
            v.append(v1[h])
            h += 1
        else:
            v.append(v2[k])
            k += 1
    return v
```

让我们从`sort`函数开始。首先，我们遇到递归的基础，它说如果列表有`0`或`1`个元素，我们不需要对其进行排序，我们可以直接返回它。如果不是这种情况，我们计算中点(`mid`)，并在`v[:mid]`和`v[mid:]`上递归调用 sort。我希望你现在对切片语法非常熟悉，但以防万一你需要复习一下，第一个是`v`中到`mid`索引（不包括）的所有元素，第二个是从`mid`到末尾的所有元素。排序它们的结果分别分配给`v1`和`v2`。最后，我们调用`merge`，传递`v1`和`v2`。

`merge`的逻辑使用两个指针`h`和`k`来跟踪我们已经比较了`v1`和`v2`中的哪些元素。如果我们发现最小值在`v1`中，我们将其附加到`v`，并增加`h`。另一方面，如果最小值在`v2`中，我们将其附加到`v`，但这次增加`k`。该过程在一个`while`循环中运行，其条件与内部的`if`结合在一起，确保我们不会因为索引超出范围而出现错误。这是一个非常标准的算法，在网上可以找到许多不同的变体。

为了确保这段代码是可靠的，我编写了一个测试套件，位于`ch10/ms`文件夹中。我鼓励你去看一下。

现在我们有了构建模块，让我们看看如何修改它，使其能够处理任意数量的部分。

# 单线程多部分归并排序

算法的多部分版本的代码非常简单。我们可以重用`merge`函数，但我们需要重新编写`sort`函数：

```py
# ms/algo/multi_mergesort.py
from functools import reduce
from .mergesort import merge

def sort(v, parts=2):
    assert parts > 1, 'Parts need to be at least 2.'
    if len(v) <= 1:
        return v

    chunk_len = max(1, len(v) // parts)
    chunks = (
        sort(v[k: k + chunk_len], parts=parts)
        for k in range(0, len(v), chunk_len)
    )
    return multi_merge(*chunks)

def multi_merge(*v):
    return reduce(merge, v)
```

我们在第四章中看到了`reduce`，*函数，代码的构建模块*，当我们编写我们自己的阶乘函数时。它在`multi_merge`中的工作方式是合并`v`中的前两个列表。然后将结果与第三个合并，之后将结果与第四个合并，依此类推。

看一下`sort`的新版本。它接受`v`列表和我们想要将其分割成的部分数。我们首先检查我们传递了一个正确的`parts`数，它至少需要是两个。然后，就像以前一样，我们有递归的基础。最后，我们进入函数的主要逻辑，这只是前一个例子中看到的逻辑的多部分版本。我们使用`max`函数计算每个`chunk`的长度，以防列表中的元素少于部分数。然后，我们编写一个生成器表达式，对每个`chunk`递归调用`sort`。最后，我们通过调用`multi_merge`合并所有的结果。

我意识到在解释这段代码时，我没有像我通常那样详尽，我担心这是有意的。在归并排序之后的例子将会更加复杂，所以我想鼓励你尽可能彻底地理解前两个片段。

现在，让我们将这个例子推进到下一步：多线程。

# 多线程归并排序

在这个例子中，我们再次修改`sort`函数，这样，在初始分成块之后，它会为每个部分生成一个线程。每个线程使用单线程版本的算法来对其部分进行排序，然后最后我们使用多重归并技术来计算最终结果。翻译成 Python：

```py
# ms/algo/mergesort_thread.py
from functools import reduce
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed
from .mergesort import sort as _sort, merge

def sort(v, workers=2):
    if len(v) == 0:
        return v
    dim = ceil(len(v) / workers)
    chunks = (v[k: k + dim] for k in range(0, len(v), dim))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_sort, chunk) for chunk in chunks
        ]
        return reduce(
            merge,
            (future.result() for future in as_completed(futures))
        )
```

我们导入所有必需的工具，包括执行器、`ceiling`函数，以及从单线程版本的算法中导入的`sort`和`merge`。请注意，我在导入时将单线程的`sort`的名称更改为`_sort`。

在这个版本的`sort`中，我们首先检查`v`是否为空，如果不是，我们继续。我们使用`ceil`函数计算每个`chunk`的维度。它基本上做的是我们在前面片段中使用`max`的事情，但我想向你展示另一种解决问题的方法。

当我们有了维度，我们计算`chunks`并准备一个漂亮的生成器表达式来将它们提供给执行器。其余部分很简单：我们定义了一个未来对象列表，每个未来对象都是在执行器上调用`submit`的结果。每个未来对象在分配给它的`chunk`上运行单线程的`_sort`算法。

最后，当它们被`as_completed`函数返回时，结果将使用我们在之前的多部分示例中看到的相同技术进行合并。

# 多进程归并排序

为了执行最后一步，我们只需要修改前面代码中的两行。如果你在介绍性的例子中注意到了，你会知道我指的是哪两行。为了节省一些空间，我只会给你代码的差异：

```py
# ms/algo/mergesort_proc.py
...
from concurrent.futures import ProcessPoolExecutor, as_completed
...

def sort(v, workers=2):
    ...
    with ProcessPoolExecutor(max_workers=workers) as executor:
    ...
```

就是这样！你所要做的就是使用`ProcessPoolExecutor`而不是`ThreadPoolExecutor`，而不是生成线程，你正在生成进程。

你还记得我说过进程实际上可以在不同的核心上运行，而线程在同一个进程中运行，因此它们实际上并不是并行运行吗？这是一个很好的例子，向你展示选择其中一种方法的后果。因为代码是 CPU 密集型的，没有进行 IO 操作，分割列表并让线程处理块并没有任何优势。另一方面，使用进程有优势。我进行了一些性能测试（自己运行`ch10/ms/performance.py`模块，你会看到你的机器的性能如何），结果证明了我的期望：

```py
$ python performance.py

Testing Sort
Size: 100000
Elapsed time: 0.492s
Size: 500000
Elapsed time: 2.739s

Testing Sort Thread
Size: 100000
Elapsed time: 0.482s
Size: 500000
Elapsed time: 2.818s

Testing Sort Proc
Size: 100000
Elapsed time: 0.313s
Size: 500000
Elapsed time: 1.586s
```

这两个测试分别在两个包含 10 万和 50 万个项目的列表上运行。我为多线程和多进程版本使用了四个工作进程。在寻找模式时，使用不同的大小非常有用。正如你所看到的，前两个版本（单线程和多线程）的时间消耗基本相同，但在多进程版本中减少了约 50%。这略高于 50%，因为生成进程并处理它们是有代价的。但是，你肯定会欣赏到我在我的机器上有一个有两个内核的处理器。

这也告诉你，即使我在多进程版本中使用了四个工作进程，我仍然只能按比例并行化我的处理器核心数量。因此，两个或更多的工作进程几乎没有什么区别。

现在你已经热身了，让我们继续下一个例子。

# 第二个例子 - 批量数独求解器

在这个例子中，我们将探索一个数独求解器。我们不会详细讨论它，因为重点不是理解如何解决数独，而是向你展示如何使用多进程来解决一批数独谜题。

在这个例子中有趣的是，我们不再比较单线程和多线程版本，而是跳过这一点，将单线程版本与两个不同的多进程版本进行比较。一个将分配一个谜题给每个工作进程，所以如果我们解决了 1,000 个谜题，我们将使用 1,000 个工作进程（好吧，我们将使用一个* N *工作进程池，每个工作进程都在不断回收）。另一个版本将把初始批次的谜题按照池的大小进行划分，并在一个进程内批量解决每个块。这意味着，假设池的大小为四，将这 1,000 个谜题分成每个 250 个谜题的块，并将每个块分配给一个工作进程，总共有四个工作进程。

我将向您展示数独求解器的代码（不包括多进程部分），这是由 Peter Norvig 设计的解决方案，根据 MIT 许可证进行分发。他的解决方案非常高效，以至于在尝试重新实现自己的解决方案几天后，得到了相同的结果，我简单地放弃了并决定采用他的设计。不过，我进行了大量的重构，因为我对他选择的函数和变量名不满意，所以我将它们更改为更符合书本风格的名称。您可以在`ch10/sudoku/norvig`文件夹中找到原始代码、获取原始页面的链接以及原始的 MIT 许可证。如果您跟随链接，您将找到 Norvig 本人对数独求解器的非常详尽的解释。

# 什么是数独？

首先来看看。什么是数独谜题？数独是一种基于逻辑的数字填充谜题，起源于日本。目标是用数字填充*9x9*网格，使得每行、每列和每个*3x3*子网格（组成网格的子网格）都包含从*1*到*9*的所有数字。您从一个部分填充的网格开始，然后根据逻辑考虑逐渐添加数字。

从计算机科学的角度来看，数独可以被解释为一个适合*exact cover*类别的问题。唐纳德·克努斯，*计算机编程艺术*的作者（以及许多其他精彩的书籍），设计了一个算法，称为**Algorithm X**，用于解决这一类问题。一种名为**Dancing Links**的美丽而高效的 Algorithm X 实现，利用了循环双向链表的强大功能，可以用来解决数独。这种方法的美妙之处在于，它只需要数独的结构与 Dancing Links 算法之间的映射，而无需进行通常需要解决难题的逻辑推断，就能以光速到达解决方案。

许多年前，当我的空闲时间大于零时，我用 C#编写了一个 Dancing Links 数独求解器，我仍然在某个地方存档着，设计和编码过程非常有趣。我绝对鼓励您查阅相关文献并编写自己的求解器，如果您有时间的话，这是一个很好的练习。

在本例的解决方案中，我们将使用与人工智能中的**约束传播**相结合的**搜索**算法。这两种方法通常一起使用，使问题更容易解决。我们将看到在我们的例子中，它们足以让我们在几毫秒内解决一个困难的数独。

# 在 Python 中实现数独求解器

现在让我们来探索我重构后的求解器实现。我将分步向您展示代码，因为它非常复杂（而且在每个片段的顶部我不会重复源名称，直到我转移到另一个模块）：

```py
# sudoku/algo/solver.py
import os
from itertools import zip_longest, chain
from time import time

def cross_product(v1, v2):
    return [w1 + w2 for w1 in v1 for w2 in v2]

def chunk(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
```

我们从一些导入开始，然后定义了一些有用的函数：`cross_product`和`chunk`。它们确实做了名称所暗示的事情。第一个函数返回两个可迭代对象之间的叉积，而第二个函数返回`iterable`的一系列块，每个块都有`n`个元素，最后一个块可能会用给定的`fillvalue`填充，如果`iterable`的长度不是`n`的倍数。然后我们继续定义一些结构，这些结构将被求解器使用：

```py
digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits
squares = cross_product(rows, cols)
all_units = (
    [cross_product(rows, c) for c in cols]
    + [cross_product(r, cols) for r in rows]
    + [cross_product(rs, cs)
        for rs in chunk(rows, 3) for cs in chunk(cols, 3)]
)
units = dict(
    (square, [unit for unit in all_units if square in unit])
    for square in squares
)
peers = dict(
    (square, set(chain(*units[square])) - set([square]))
    for square in squares
)
```

不详细展开，让我们简单介绍一下这些对象。`squares`是网格中所有方块的列表。方块由诸如*A3*或*C7*之类的字符串表示。行用字母编号，列用数字编号，因此*A3*表示第一行第三列的方块。

`all_units`是所有可能的行、列和块的列表。每个元素都表示为属于行/列/块的方格的列表。`units`是一个更复杂的结构。它是一个有 81 个键的字典。每个键代表一个方格，相应的值是一个包含三个元素的列表：行、列和块。当然，这些是方格所属的行、列和块。

最后，`peers`是一个与`units`非常相似的字典，但每个键的值（仍然表示一个方格）是一个包含该方格的所有对等方格的集合。对等方格被定义为属于键中的方格所属的行、列和块的所有方格。这些结构将在解决谜题时用于计算解决方案。

在我们看一下解析输入行的函数之前，让我给你一个输入谜题的例子：

```py
1..3.......75...3..3.4.8.2...47....9.........689....4..5..178.4.....2.75.......1.
```

前九个字符代表第一行，然后另外九个代表第二行，依此类推。空方格用点表示：

```py
def parse_puzzle(puzzle):
    assert set(puzzle) <= set('.0123456789')
    assert len(puzzle) == 81

    grid = dict((square, digits) for square in squares)
    for square, digit in zip(squares, puzzle):
        if digit in digits and not place(grid, square, digit):
            return False  # Incongruent puzzle
    return grid

def solve(puzzle):
    grid = parse_puzzle(puzzle)
    return search(grid)
```

这个简单的`parse_puzzle`函数用于解析输入的谜题。我们在开始时进行了一些合理性检查，断言输入的谜题必须缩小为所有数字加一个点的子集。然后我们确保有 81 个输入字符，最后我们定义了`grid`，最初它只是一个有 81 个键的字典，每个键都是一个方格，都具有相同的值，即所有可能数字的字符串。这是因为在完全空的网格中，一个方格有潜力成为 1 到 9 之间的任何数字。

`for`循环绝对是最有趣的部分。我们解析输入谜题中的每个 81 个字符，将它们与网格中相应的方格相结合，并尝试“放置”它们。我用双引号括起来，因为正如我们将在一会儿看到的，`place`函数做的远不止简单地在给定的方格中设置一个给定的数字。如果我们发现无法在输入谜题中放置一个数字，这意味着输入无效，我们返回`False`。否则，我们可以继续并返回`grid`。

`parse_puzzle`函数用于`solve`函数中，它简单地解析输入的谜题，并在其上释放`search`。因此，接下来的内容是算法的核心：

```py
def search(grid):
    if not grid:
        return False
    if all(len(grid[square]) == 1 for square in squares):
        return grid  # Solved
    values, square = min(
        (len(grid[square]), square) for square in squares
        if len(grid[square]) > 1
    )
    for digit in grid[square]:
        result = search(place(grid.copy(), square, digit))
        if result:
            return result
```

这个简单的函数首先检查网格是否真的非空。然后它尝试查看网格是否已解决。已解决的网格将每个方格都有一个值。如果不是这种情况，它会循环遍历每个方格，并找到具有最少候选项的方格。如果一个方格的字符串值只有一个数字，这意味着一个数字已经放在了那个方格中。但如果值超过一个数字，那么这些就是可能的候选项，所以我们需要找到具有最少候选项的方格，并尝试它们。尝试一个有 23 个候选项的方格要比尝试一个有 23589 个候选项的方格好得多。在第一种情况下，我们有 50%的机会得到正确的值，而在第二种情况下，我们只有 20%。选择具有最少候选项的方格因此最大化了我们在网格中放置好数字的机会。

一旦找到候选项，我们按顺序尝试它们，如果其中任何一个成功，我们就解决了网格并返回。您可能已经注意到在搜索中使用了`place`函数。因此，让我们来探索它的代码：

```py
def place(grid, square, digit):
    """Eliminate all the other values (except digit) from
    grid[square] and propagate.
    Return grid, or False if a contradiction is detected.
    """
    other_vals = grid[square].replace(digit, '')
    if all(eliminate(grid, square, val) for val in other_vals):
        return grid
    return False
```

这个函数接受一个正在进行中的网格，并尝试在给定的方格中放置一个给定的数字。正如我之前提到的，*“放置”*并不那么简单。事实上，当我们放置一个数字时，我们必须在整个网格中传播该行为的后果。我们通过调用`eliminate`函数来做到这一点，该函数应用数独游戏的两种策略：

+   如果一个方格只有一个可能的值，就从该方格的对等方格中消除该值

+   如果一个单元只有一个值的位置，就把值放在那里

让我简要地举个例子。对于第一个点，如果你在一个方块中放入数字 7，那么你可以从属于该行、列和块的所有方块的候选数字列表中删除 7。

对于第二点，假设你正在检查第四行，而属于它的所有方块中，只有一个方块的候选数字中有数字 7。这意味着数字 7 只能放在那个确切的方块中，所以你应该继续把它放在那里。

接下来的函数`eliminate`应用了这两条规则。它的代码相当复杂，所以我没有逐行解释，而是添加了一些注释，留给你去理解：

```py
def eliminate(grid, square, digit):
    """Eliminate digit from grid[square]. Propagate when candidates
    are <= 2.
    Return grid, or False if a contradiction is detected.
    """
    if digit not in grid[square]:
        return grid  # already eliminated
    grid[square] = grid[square].replace(digit, '')

    ## (1) If a square is reduced to one value, eliminate value
    ## from peers.
    if len(grid[square]) == 0:
        return False  # nothing left to place here, wrong solution
    elif len(grid[square]) == 1:
        value = grid[square]
        if not all(
            eliminate(grid, peer, value) for peer in peers[square]
        ):
            return False

    ## (2) If a unit is reduced to only one place for a value,
    ## then put it there.
    for unit in units[square]:
        places = [sqr for sqr in unit if digit in grid[sqr]]
        if len(places) == 0:
            return False  # No place for this value
        elif len(places) == 1:
            # digit can only be in one place in unit,
            # assign it there
            if not place(grid, places[0], digit):
                return False
    return grid
```

模块中的其他函数对于本例来说并不重要，所以我会跳过它们。你可以单独运行这个模块；它首先对其数据结构进行一系列检查，然后解决我放在`sudoku/puzzles`文件夹中的所有数独难题。但这不是我们感兴趣的，对吧？我们想要看看如何使用多进程技术解决数独，所以让我们开始吧。

# 使用多进程解决数独

在这个模块中，我们将实现三个函数。第一个函数简单地解决一批数独难题，没有涉及多进程。我们将使用结果进行基准测试。第二个和第三个函数将使用多进程，一个是批量解决，一个是非批量解决，这样我们可以欣赏到它们之间的差异。让我们开始吧：

```py
# sudoku/process_solver.py
import os
from functools import reduce
from operator import concat
from math import ceil
from time import time
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from unittest import TestCase
from algo.solver import solve

@contextmanager
def timer():
    t = time()
    yield
    tot = time() - t
    print(f'Elapsed time: {tot:.3f}s')
```

经过一长串的导入后，我们定义了一个上下文管理器，我们将用它作为计时器。它获取当前时间的引用（`t`），然后进行 yield。在 yield 之后，才执行上下文管理器的主体。最后，在退出上下文管理器时，我们计算总共经过的时间`tot`，并打印出来。这是一个简单而优雅的上下文管理器，使用了装饰技术编写，非常有趣。现在让我们看看前面提到的三个函数：

```py
def batch_solve(puzzles):
    # Single thread batch solve.
    return [solve(puzzle) for puzzle in puzzles]
```

这是一个单线程的简单批量求解器，它将给我们一个用于比较的时间。它只是返回所有已解决的网格的列表。无聊。现在，看看下面的代码：

```py
def parallel_single_solver(puzzles, workers=4):
    # Parallel solve - 1 process per each puzzle
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = (
            executor.submit(solve, puzzle) for puzzle in puzzles
        )
        return [
            future.result() for future in as_completed(futures)
        ]
```

这个函数好多了。它使用`ProcessPoolExecutor`来使用一个`workers`池，每个池用于解决大约四分之一的难题。这是因为我们为每个难题生成一个`future`对象。逻辑与本章中已经看到的任何多进程示例非常相似。现在让我们看看第三个函数：

```py
def parallel_batch_solver(puzzles, workers=4):
    # Parallel batch solve - Puzzles are chunked into `workers`
    # chunks. A process for each chunk.
    assert len(puzzles) >= workers
    dim = ceil(len(puzzles) / workers)
    chunks = (
        puzzles[k: k + dim] for k in range(0, len(puzzles), dim)
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = (
            executor.submit(batch_solve, chunk) for chunk in chunks
        )
        results = (
            future.result() for future in as_completed(futures)
        )
        return reduce(concat, results)
```

最后一个函数略有不同。它不是为每个难题生成一个`future`对象，而是将所有难题的列表分成`workers`块，然后为每一块创建一个`future`对象。这意味着如果`workers`为八，我们将生成八个`future`对象。请注意，我们不是将`solve`传递给`executor.submit`，而是传递`batch_solve`，这就是诀窍所在。我之所以编写最后两个函数如此不同，是因为我很好奇我们从池中重复使用进程时所产生的开销的严重程度。

现在我们已经定义了这些函数，让我们使用它们：

```py
puzzles_file = os.path.join('puzzles', 'sudoku-topn234.txt')
with open(puzzles_file) as stream:
    puzzles = [puzzle.strip() for puzzle in stream]

# single thread solve
with timer():
    res_batch = batch_solve(puzzles)

# parallel solve, 1 process per puzzle
with timer():
    res_parallel_single = parallel_single_solver(puzzles)

# parallel batch solve, 1 batch per process
with timer():
    res_parallel_batch = parallel_batch_solver(puzzles)

# Quick way to verify that the results are the same, but
# possibly in a different order, as they depend on how the
# processes have been scheduled.
assert_items_equal = TestCase().assertCountEqual
assert_items_equal(res_batch, res_parallel_single)
assert_items_equal(res_batch, res_parallel_batch)
print('Done.')
```

我们使用了一组 234 个非常难的数独难题进行基准测试。正如你所看到的，我们只是在一个计时上下文中运行了三个函数，`batch_solve`，`parallel_single_solver`和`parallel_batch_solver`。我们收集结果，并且为了确保，我们验证所有运行是否产生了相同的结果。

当然，在第二次和第三次运行中，我们使用了多进程，所以我们不能保证结果的顺序与单线程`batch_solve`的顺序相同。这个小问题通过`assertCountEqual`得到了很好的解决，这是 Python 标准库中命名最糟糕的方法之一。我们可以在`TestCase`类中找到它，我们可以实例化它来引用我们需要的方法。我们实际上并没有运行单元测试，但这是一个很酷的技巧，我想向你展示一下。让我们看看运行这个模块的输出：

```py
$ python process_solver.py
Elapsed time: 5.368s
Elapsed time: 2.856s
Elapsed time: 2.818s
Done. 
```

哇。这非常有趣。首先，你可以再次看到我的机器有一个双核处理器，因为多进程运行的时间大约是单线程求解器所花时间的一半。然而，更有趣的是，两个多进程函数所花费的时间基本上没有区别。多次运行有时候会偏向一种方法，有时候会偏向另一种方法。要理解原因需要对参与游戏的所有组件有深入的了解，而不仅仅是进程，因此这不是我们可以在这里讨论的事情。不过，可以相当肯定的是，这两种方法在性能方面是可比较的。

在这本书的源代码中，你可以在`sudoku`文件夹中找到测试，并附有运行说明。花点时间去查看一下吧！

现在，让我们来看最后一个例子。

# 第三个例子 - 下载随机图片

这个例子编写起来很有趣。我们将从网站上下载随机图片。我会向你展示三个版本：一个串行版本，一个多进程版本，最后一个使用`asyncio`编写的解决方案。在这些例子中，我们将使用一个名为[`lorempixel.com`](http://lorempixel.com/)的网站，它提供了一个 API，你可以调用它来获取随机图片。如果你发现该网站宕机或运行缓慢，你可以使用一个很好的替代网站：[`lorempizza.com/`](https://lorempizza.com/)。

这可能是一个意大利人写的书的陈词滥调，但图片确实很漂亮。如果你想玩得开心，可以在网上寻找其他选择。无论你选择哪个网站，请理智一点，尽量不要通过发出一百万个请求来使其崩溃。这段代码的多进程和`asyncio`版本可能会相当激进！

让我们先来探索单线程版本的代码：

```py
# aio/randompix_serial.py
import os
from secrets import token_hex
import requests

PICS_FOLDER = 'pics'
URL = 'http://lorempixel.com/640/480/'

def download(url):
    resp = requests.get(URL)
    return save_image(resp.content)

def save_image(content):
    filename = '{}.jpg'.format(token_hex(4))
    path = os.path.join(PICS_FOLDER, filename)
    with open(path, 'wb') as stream:
        stream.write(content)
    return filename

def batch_download(url, n):
    return [download(url) for _ in range(n)]

if __name__ == '__main__':
    saved = batch_download(URL, 10)
    print(saved)
```

现在这段代码对你来说应该很简单了。我们定义了一个`download`函数，它向给定的`URL`发出请求，通过调用`save_image`保存结果，并将来自网站响应的主体传递给它。保存图片非常简单：我们使用`token_hex`创建一个随机文件名，只是因为这样很有趣，然后计算文件的完整路径，以二进制模式创建文件，并将响应的内容写入其中。我们返回`filename`以便在屏幕上打印它。最后，`batch_download`只是运行我们想要运行的`n`个请求，并将文件名作为结果返回。

你现在可以跳过`if __name__ ...`这一行，它将在第十二章中解释，*GUIs and Scripts*，这里并不重要。我们所做的就是调用`batch_download`并告诉它下载`10`张图片。如果你有编辑器，打开`pics`文件夹，你会看到它在几秒钟内被填充（还要注意：脚本假设`pics`文件夹存在）。

让我们加点料。让我们引入多进程（代码基本相似，所以我就不重复了）：

```py
# aio/randompix_proc.py
...
from concurrent.futures import ProcessPoolExecutor, as_completed
...

def batch_download(url, n, workers=4):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = (executor.submit(download, url) for _ in range(n))
        return [future.result() for future in as_completed(futures)]

...
```

这种技术现在对你来说应该很熟悉。我们只是将作业提交给执行器，并在结果可用时收集它们。因为这是 IO 绑定的代码，所以进程工作得相当快，而在进程等待 API 响应时，有大量的上下文切换。如果你查看`pics`文件夹，你会注意到它不再是线性地填充，而是分批次地填充。

现在让我们看看这个例子的`asyncio`版本。

# 使用 asyncio 下载随机图片

这段代码可能是整个章节中最具挑战性的，所以如果此刻对你来说太多了，不要感到难过。我添加了这个例子，只是作为一种引人入胜的手段，鼓励你深入了解 Python 异步编程的核心。另一个值得知道的是，可能有几种其他编写相同逻辑的方式，所以请记住，这只是可能的例子之一。

`asyncio`模块提供了基础设施，用于使用协程编写单线程并发代码，多路复用 IO 访问套接字和其他资源，运行网络客户端和服务器，以及其他相关原语。它在 Python 3.4 版本中添加，有人声称它将成为未来编写 Python 代码的*事实*标准。我不知道这是否属实，但我知道它绝对值得看一个例子：

```py
# aio/randompix_corout.py
import os
from secrets import token_hex
import asyncio
import aiohttp
```

首先，我们不能再使用`requests`，因为它不适用于`asyncio`。我们必须使用`aiohttp`，所以请确保你已经安装了它（它在这本书的要求中）：

```py
PICS_FOLDER = 'pics'
URL = 'http://lorempixel.com/640/480/'

async def download_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.read()
```

之前的代码看起来不太友好，但一旦你了解了背后的概念，就不会那么糟糕。我们定义了异步协程`download_image`，它以 URL 作为参数。

如果你不知道，协程是一种计算机程序组件，它通过允许在特定位置挂起和恢复执行来概括非抢占式多任务处理的子例程。子例程是作为一个单元打包的执行特定任务的程序指令序列。

在`download_image`中，我们使用`ClientSession`上下文管理器创建一个会话对象，然后通过使用另一个上下文管理器`session.get`获取响应。这些管理器被定义为异步的事实意味着它们能够在它们的`enter`和`exit`方法中暂停执行。我们使用`await`关键字返回响应的内容，这允许暂停。请注意，为每个请求创建一个会话并不是最佳的，但是为了这个例子的目的，我觉得保持代码尽可能简单，所以将其优化留给你作为一个练习。

让我们继续下一个片段：

```py
async def download(url, semaphore):
    async with semaphore:
        content = await download_image(url)
    filename = save_image(content)
    return filename

def save_image(content):
    filename = '{}.jpg'.format(token_hex(4))
    path = os.path.join(PICS_FOLDER, filename)
    with open(path, 'wb') as stream:
        stream.write(content)
    return filename
```

另一个协程`download`获取一个`URL`和一个`信号量`。它所做的就是获取图像的内容，通过调用`download_image`保存它，并返回`文件名`。这里有趣的地方是使用了`信号量`。我们将其用作异步上下文管理器，以便我们也可以暂停这个协程，并允许切换到其他东西，但更重要的不是*如何*，而是理解*为什么*我们要使用`信号量`。原因很简单，这个`信号量`有点类似于线程池。我们使用它来允许最多*N*个协程同时活动。我们在下一个函数中实例化它，并将 10 作为初始值传递。每当一个协程获取`信号量`时，它的内部计数器就会减少`1`，因此当有 10 个协程获取它时，下一个协程将会等待，直到`信号量`被一个已经完成的协程释放。这是一个不错的方式，试图限制我们从网站 API 中获取图像的侵略性。

`save_image`函数不是一个协程，它的逻辑已经在之前的例子中讨论过。现在让我们来到执行代码的部分：

```py
def batch_download(images, url):
    loop = asyncio.get_event_loop()
    semaphore = asyncio.Semaphore(10)
    cors = [download(url, semaphore) for _ in range(images)]
    res, _ = loop.run_until_complete(asyncio.wait(cors))
    loop.close()
    return [r.result() for r in res]

if __name__ == '__main__':
    saved = batch_download(20, URL)
    print(saved)
```

我们定义了`batch_download`函数，它接受一个数字`images`和要获取它们的 URL。它做的第一件事是创建一个事件循环，这是运行任何异步代码所必需的。事件循环是`asyncio`提供的中央执行设备。它提供了多种设施，包括：

+   注册、执行和取消延迟调用（超时）

+   为各种通信创建客户端和服务器传输

+   启动子进程和与外部程序通信的相关传输

+   将昂贵的函数调用委托给线程池

事件循环创建后，我们实例化信号量，然后继续创建一个期货列表`cors`。通过调用`loop.run_until_complete`，我们确保事件循环将一直运行，直到整个任务完成。我们将其喂给`asyncio.wait`的调用结果，它等待期货完成。

完成后，我们关闭事件循环，并返回每个期货对象产生的结果列表（保存图像的文件名）。请注意我们如何捕获对`loop.run_until_complete`的调用结果。我们并不真正关心错误，所以我们将第二个元组项赋值为`_`。这是一个常见的 Python 习惯用法，用于表明我们对该对象不感兴趣。

在模块的最后，我们调用`batch_download`，并保存了 20 张图片。它们分批次到达，整个过程受到只有 10 个可用位置的信号量的限制。

就是这样！要了解更多关于`asyncio`的信息，请参阅标准库中`asyncio`模块的文档页面（[`docs.python.org/3.7/library/asyncio.html`](https://docs.python.org/3.7/library/asyncio.html)）。这个例子编码起来很有趣，希望它能激励你努力学习并理解 Python 这一美妙的一面的复杂性。

# 总结

在本章中，我们学习了并发和并行。我们看到了线程和进程如何帮助实现其中的一个和另一个。我们探讨了线程的性质以及它们暴露给我们的问题：竞争条件和死锁。

我们学会了如何通过使用锁和谨慎的资源管理来解决这些问题。我们还学会了如何使线程通信和共享数据，并讨论了调度程序，即操作系统决定任何给定时间运行哪个线程的部分。然后我们转向进程，并探讨了它们的一些属性和特征。

在最初的理论部分之后，我们学会了如何在 Python 中实现线程和进程。我们处理了多个线程和进程，解决了竞争条件，并学会了防止线程错误地留下任何资源的解决方法。我们还探讨了 IPC，并使用队列在进程和线程之间交换消息。我们还使用了事件和屏障，这些是标准库提供的一些工具，用于在非确定性环境中控制执行流程。

在所有这些介绍性示例之后，我们深入研究了三个案例示例，展示了如何使用不同的方法解决相同的问题：单线程、多线程、多进程和`asyncio`。

我们学习了归并排序以及通常*分而治之*算法易于并行化。

我们学习了关于数独，并探讨了一种使用少量人工智能来运行高效算法的好方法，然后我们以不同的串行和并行模式运行了它。

最后，我们看到了如何使用串行、多进程和`asyncio`代码从网站上下载随机图片。后者无疑是整本书中最难的代码，它在本章中的存在是作为一种提醒，或者一种里程碑，鼓励读者深入学习 Python。

现在我们将转向更简单的、大多数是项目导向的章节，我们将在不同的背景下尝试不同的真实世界应用。
