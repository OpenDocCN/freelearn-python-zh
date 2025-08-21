# 第四章：在线程中使用 with 语句

`with`语句在 Python 中有时会让新手和有经验的 Python 程序员感到困惑。本章深入解释了`with`语句作为上下文管理器的概念，以及它在并发和并行编程中的使用，特别是在同步线程时使用锁。本章还提供了`with`语句最常见用法的具体示例。

本章将涵盖以下主题：

+   上下文管理的概念以及`with`语句作为上下文管理器在并发和并行编程中提供的选项

+   `with`语句的语法以及如何有效和高效地使用它

+   在并发编程中使用`with`语句的不同方式

# 技术要求

以下是本章的先决条件清单：

+   Python 3 必须安装在您的计算机上

+   在[`github.com/PacktPublishing/Mastering-Concurrency-in-Python`](https://github.com/PacktPublishing/Mastering-Concurrency-in-Python)下载 GitHub 存储库

+   在本章中，我们将使用名为`Chapter04`的子文件夹

+   查看以下视频以查看代码的实际操作：[`bit.ly/2DSGLEZ`](http://bit.ly/2DSGLEZ)

# 上下文管理

新的`with`语句首次在 Python 2.5 中引入，并且已经使用了相当长的时间。然而，即使对于有经验的 Python 程序员，对其使用仍然存在困惑。`with`语句最常用作上下文管理器，以正确管理资源，在并发和并行编程中是至关重要的，其中资源在并发或并行应用程序中跨不同实体共享。

# 从管理文件开始

作为一个有经验的 Python 用户，你可能已经看到`with`语句被用来在 Python 程序中打开和读取外部文件。从更低的层次来看，Python 中打开外部文件的操作会消耗资源——在这种情况下是文件描述符——你的操作系统会对这种资源设置一个限制。这意味着在你的系统上运行的单个进程同时打开的文件数量是有上限的。

让我们考虑一个快速的例子来进一步说明这一点。让我们看一下`Chapter04/example1.py`文件，如下所示的代码：

```py
# Chapter04/example1.py

n_files = 10
files = []

for i in range(n_files):
    files.append(open('output1/sample%i.txt' % i, 'w'))
```

这个快速程序简单地在`output1`文件夹中创建了 10 个文本文件：`sample0.txt`，`sample1.txt`，...，`sample9.txt`。对我们来说可能更感兴趣的是这些文件是在`for`循环中打开的，但没有关闭——这是编程中的一个不良实践，我们稍后会讨论。现在，假设我们想将`n_files`变量重新分配给一个大数——比如 10000，如下所示的代码：

```py
# Chapter4/example1.py

n_files = 10000
files = []

# method 1
for i in range(n_files):
    files.append(open('output1/sample%i.txt' % i, 'w'))
```

我们会得到类似以下的错误：

```py
> python example1.py
Traceback (most recent call last):
 File "example1.py", line 7, in <module>
OSError: [Errno 24] Too many open files: 'output1/sample253.txt'
```

仔细看错误消息，我们可以看到我的笔记本电脑只能同时处理 253 个打开的文件（顺便说一句，如果你在类 UNIX 系统上工作，运行`ulimit -n`会给你系统可以处理的文件数量）。更一般地说，这种情况是由所谓的**文件描述符泄漏**引起的。当 Python 在程序中打开一个文件时，该打开的文件实质上由一个整数表示。这个整数充当程序可以使用的参考点，以便访问该文件，同时不完全控制底层文件本身。

通过同时打开太多文件，我们的程序分配了太多文件描述符来管理打开的文件，因此出现了错误消息。文件描述符泄漏可能导致许多困难的问题——特别是在并发和并行编程中——即未经授权的对打开文件的 I/O 操作。解决这个问题的方法就是以协调的方式关闭打开的文件。让我们看看第二种方法中的`Chapter04/example1.py`文件。在`for`循环中，我们会这样做：

```py
# Chapter04/example1.py

n_files = 1000
files = []

# method 2
for i in range(n_files):
    f = open('output1/sample%i.txt' % i, 'w')
    files.append(f)
    f.close()
```

# 作为上下文管理器的 with 语句

在实际应用中，很容易通过忘记关闭它们来管理程序中打开的文件；有时也可能无法确定程序是否已经完成处理文件，因此我们程序员将无法决定何时适当地关闭文件。这种情况在并发和并行编程中更为常见，其中不同元素之间的执行顺序经常发生变化。

这个问题的一个可能解决方案，在其他编程语言中也很常见，就是每次想要与外部文件交互时都使用`try...except...finally`块。这种解决方案仍然需要相同级别的管理和显著的开销，并且在程序的易用性和可读性方面也没有得到很好的改进。这就是 Python 的`with`语句派上用场的时候。

`with`语句为我们提供了一种简单的方法，确保所有打开的文件在程序使用完毕时得到适当的管理和清理。使用`with`语句最显著的优势在于，即使代码成功执行或返回错误，`with`语句始终通过上下文处理和管理打开的文件。例如，让我们更详细地看看我们的`Chapter04/example1.py`文件：

```py
# Chapter04/example1.py

n_files = 254
files = []

# method 3
for i in range(n_files):
    with open('output1/sample%i.txt' % i, 'w') as f:
        files.append(f)
```

虽然这种方法完成了我们之前看到的第二种方法相同的工作，但它另外提供了一种更清晰和更易读的方式来管理我们的程序与之交互的打开文件。更具体地说，`with`语句帮助我们指示特定变量的范围——在这种情况下，指向打开文件的变量——因此也指明了它们的上下文。

例如，在前面代码的第三种方法中，`f`变量在`with`块的每次迭代中指示当前打开的文件，并且一旦我们的程序退出了`with`块（超出了`f`变量的范围），就再也无法访问它。这种架构保证了与文件描述符相关的所有清理都会适当地进行。因此`with`语句被称为上下文管理器。

# with 语句的语法

`with`语句的语法可以直观和简单。为了使用上下文管理器定义的方法包装一个块的执行，它由以下简单形式组成：

```py
with [expression] (as [target]):
    [code]
```

请注意，`with`语句中的`as [target]`部分实际上是不需要的，我们稍后会看到。此外，`with`语句也可以处理同一行上的多个项目。具体来说，创建的上下文管理器被视为多个`with`语句嵌套在一起。例如，看看以下代码：

```py
with [expression1] as [target1], [expression2] as [target2]:
    [code]
```

这被解释为：

```py
with [expression1] as [target1]:
    with [expression2] as [target2]:
        [code]
```

# 并发编程中的 with 语句

显然，打开和关闭外部文件并不太像并发。然而，我们之前提到`with`语句作为上下文管理器，不仅用于管理文件描述符，而且通常用于管理大多数资源。如果你在阅读第二章时发现管理`threading.Lock()`类的锁对象与管理外部文件类似，那么这里的比较就派上用场了。

作为一个提醒，锁是并发和并行编程中通常用于同步多线程的机制（即防止多个线程同时访问关键会话）。然而，正如我们将在第十二章中再次讨论的那样，*饥饿*，锁也是**死锁**的常见来源，其中一个线程**获取**了一个锁，但由于未处理的发生而从未**释放**它，从而停止整个程序。

# 死锁处理示例

让我们来看一个 Python 的快速示例。让我们看一下`Chapter04/example2.py`文件，如下所示：

```py
# Chapter04/example2.py

from threading import Lock

my_lock = Lock()

def get_data_from_file_v1(filename):
    my_lock.acquire()

    with open(filename, 'r') as f:
        data.append(f.read())

    my_lock.release()

data = []

try:
    get_data_from_file('output2/sample0.txt')
except FileNotFoundError:
    print('Encountered an exception...')

my_lock.acquire()
print('Lock can still be acquired.')
```

在这个例子中，我们有一个`get_data_from_file_v1()`函数，它接受外部文件的路径，从中读取数据，并将该数据附加到一个预先声明的名为`data`的列表中。在这个函数内部，一个名为`my_lock`的锁对象，在调用函数之前也是预先声明的，分别在读取文件之前和之后被获取和释放。

在主程序中，我们将尝试在一个不存在的文件上调用`get_data_from_file_v1()`，这是编程中最常见的错误之一。在程序的末尾，我们还会再次获取锁对象。重点是看看我们的编程是否能够处理读取不存在文件的错误，只使用我们已经有的`try...except`块。

运行脚本后，您会注意到我们的程序将打印出`try...except`块中指定的错误消息`遇到异常...`，这是预期的，因为找不到文件。但是，程序还将无法执行其余的代码；它永远无法到达代码的最后一行`print('Lock acquired.')`，并且将永远挂起（或者直到您按下*Ctrl* + *C*强制退出程序）。

这是一个死锁的情况，再次发生在`get_data_from_file_v1()`函数内部获取`my_lock`，但由于我们的程序在执行`my_lock.release()`之前遇到错误，锁从未被释放。这反过来导致程序末尾的`my_lock.acquire()`行挂起，因为无论如何都无法获取锁。因此，我们的程序无法达到最后一行代码`print('Lock acquired.')`。

然而，这个问题可以很容易地通过`with`语句轻松处理。在`example2.py`文件中，只需注释掉调用`get_data_from_file_v1()`的行，并取消注释调用`get_data_from_file_v2()`的行，您将得到以下结果：

```py
# Chapter04/example2.py

from threading import Lock

my_lock = Lock()

def get_data_from_file_v2(filename):
    with my_lock, open(filename, 'r') as f:
        data.append(f.read())

data = []

try:
    get_data_from_file_v2('output2/sample0.txt')
except:
    print('Encountered an exception...')

my_lock.acquire()
print('Lock acquired.')
```

在`get_data_from_file_v2()`函数中，我们有一对嵌套的`with`语句，如下所示：

```py
with my_lock:
    with open(filename, 'r') as f:
        data.append(f.read())
```

由于`Lock`对象是上下文管理器，简单地使用`with my_lock:`将确保锁对象被适当地获取和释放，即使在块内遇到异常。运行脚本后，您将得到以下输出：

```py
> python example2.py
Encountered an exception...
Lock acquired.
```

我们可以看到，这次我们的程序能够获取锁，并且在没有错误的情况下优雅地执行脚本的末尾。

# 总结

Python 中的`with`语句提供了一种直观和方便的方式来管理资源，同时确保错误和异常被正确处理。在并发和并行编程中，管理资源的能力更加重要，不同实体之间共享和利用各种资源，特别是通过在多线程应用程序中使用`with`语句与`threading.Lock`对象同步不同线程。

除了更好的错误处理和保证清理任务外，`with`语句还可以提供程序的额外可读性，这是 Python 为开发人员提供的最强大的功能之一。

在下一章中，我们将讨论 Python 目前最流行的用途之一：网络爬虫应用。我们将看看网络爬虫背后的概念和基本思想，Python 提供的支持网络爬虫的工具，以及并发如何显著帮助您的网络爬虫应用程序。

# 问题

+   文件描述符是什么，Python 中如何处理它？

+   当文件描述符没有得到谨慎处理时会出现什么问题？

+   锁是什么，Python 中如何处理它？

+   当锁没有得到谨慎处理时会出现什么问题？

+   上下文管理器背后的思想是什么？

+   Python 的`with`语句在上下文管理方面提供了哪些选项？

# 进一步阅读

有关更多信息，您可以参考以下链接：

+   *Python 并行编程食谱*，Zaccone 和 Giancarlo 著，Packt 出版，2015

+   *改进您的 Python：使用 with 语句和上下文管理器*，Jeff Knupp（[`jeffknupp.com/blog/2016/03/07/improve-your-python-the-with-statement-and-context-managers/`](https://jeffknupp.com/blog/2016/03/07/improve-your-python-the-with-statement-and-context-managers/)）

+   *复合语句*，Python 软件基金会（[`docs.python.org/3/reference/compound_stmts.html`](https://docs.python.org/3/reference/compound_stmts.html)）
