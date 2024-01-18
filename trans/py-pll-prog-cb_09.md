# 调试阶段

这最后一章将介绍两个重要的软件工程主题——调试和测试，这是软件开发过程中的重要步骤。

本章的第一部分专注于代码调试。错误是程序中的错误，可能会导致不同严重程度的问题，具体取决于情况。为了鼓励程序员寻找错误，使用称为**调试器**的特殊软件工具；使用这些软件工具，我们有能力通过利用特定的调试功能来找到程序中的错误或故障，这是一个专门用于识别受错误影响的软件部分的活动。

在第二部分，主要主题是*软件测试*：这是用于识别正在开发的软件产品中的*正确性*、*完整性*和*可靠性*缺陷的过程。

在这种情况下，我们将研究三个最重要的Python调试代码工具。这些是`winpdb-reborn`，涉及使用可视化工具进行调试；`pdb`，Python标准库中的调试器；和`rpdb`，其中`r`代表远程，意味着它是从远程机器进行代码调试。

使用`nose`进行应用程序测试

如今，这一活动得到了特定应用程序和调试器的支持，这些调试器通过逐步软件指令向程序员展示执行过程，允许同时查看和分析程序本身的输入和输出。

在本章中，我们将涵盖以下主题：

+   什么是调试？

+   什么是软件测试？

+   使用Winpdb Reborn进行调试

+   实现`rpdb`进行调试

+   什么是调试？

+   处理`unittest`

+   这些是用于开发单元测试的框架，其中单元是程序中独立操作的最小组件。

# 术语*调试*表示在软件使用后，识别代码部分中发现一个或多个错误（bug）的活动。

关于软件测试，我们将研究以下工具：`unittest`和`nose`。

什么是软件测试？

错误可以在程序的测试阶段定位；即在开发阶段尚未准备好供最终用户使用时，或者在最终用户使用程序时。找到错误后，将进行调试阶段，并确定错误所在的软件部分，有时非常复杂。

在这些工具可用于识别和纠正错误的活动之前（甚至现在，在没有它们的情况下），代码检查的最简单（但也是最不有效的）技术是打印文件或打印程序正在执行的指令。

调试是程序开发中最重要的操作之一。由于正在开发的软件的复杂性，通常非常困难。由于存在引入新错误或行为的风险，这甚至是微妙的，这些错误或行为与尝试纠正的错误不符。

尽管使用调试来完善软件的任务每次都是独一无二的，构成了一个独立的故事，但一些通用原则始终适用。特别是在软件应用程序的上下文中，可以识别以下四个*调试阶段*，总结如下图所示：

与`pdb`交互

当然，Python为开发人员提供了许多调试工具（请参阅[https://wiki.python.org/moin/PythonDebuggingTools](https://wiki.python.org/moin/PythonDebuggingTools)以获取Python调试器列表）。在本章中，我们将考虑Winpdb Reborn、`rpdb`和`pdb`。

# Python调试和测试

正如本章介绍中所述，软件测试是用于识别正在开发的软件产品中的正确性、完整性和可靠性缺陷的过程。

因此，通过这项活动，我们希望通过搜索缺陷或一系列指令和程序来确保产品的质量，当以特定输入数据和特定操作环境执行时，会产生故障。故障是用户不期望的软件行为；因此，它与规范以及为此类应用程序定义的隐式或显式要求不同。

因此，测试的目的是通过故障检测缺陷，以便在软件产品的正常使用中最小化此类故障发生的概率。测试无法确定产品在所有可能的执行条件下都能正确运行，但可以在特定条件下突出显示缺陷。

事实上，鉴于无法测试所有输入组合以及应用程序可能运行的可能软件和硬件环境，故障的概率无法降低到零，但必须降低到最低以使用户可以接受。

软件测试的一种特定类型是单元测试（我们将在本章中学习），其目的是隔离程序的每个部分，并展示其在实现上的正确性和完整性。它还可以及时发现任何缺陷，以便在集成之前轻松进行纠正。

此外，与在整个应用程序上执行测试相比，单元测试降低了识别和纠正缺陷的成本（时间和资源）。

# 使用Winpdb Reborn进行调试

**Winpdb Reborn**是最重要和最知名的Python调试器之一。该调试器的主要优势在于管理基于线程的代码的调试。

Winpdb Reborn基于RPDB2调试器，而Winpdb是RPDB2的GUI前端（参见：[https://github.com/bluebird75/winpdb/blob/master/rpdb2.py](https://github.com/bluebird75/winpdb/blob/master/rpdb2.py)）。

# 准备工作

安装Winpdb Reborn（*release 2.0.0 dev5*）最常用的方法是通过`pip`，因此您需要在控制台中输入以下内容：

```py
C:\>pip install winpdb-reborn
```

另外，如果您尚未在Python发行版中安装wxPython，则需要这样做。wxPython是Python语言的跨平台GUI工具包。

对于Python 2.x版本，请参考[https://sourceforge.net/projects/wxpython/files/wxPython/](https://sourceforge.net/projects/wxpython/files/wxPython/)。对于Python 3.x版本，wxPython会自动通过`pip`作为依赖项安装。

在下一节中，我们将通过一个Winpdb Reborn的简单示例来检查Winpdb Reborn的主要特性和图形界面的使用。

# 操作步骤...

假设我们想要分析以下使用线程库的Python应用程序。与下面的示例非常相似的示例已经在[第2章](c95be391-9558-4d2d-867e-96f61fbc5bbf.xhtml)的*如何定义线程子类*部分中描述过。在下面的示例中，我们使用`MyThreadClass`类来创建和管理三个线程的执行。以下是整个调试代码：

```py
import time
import os
from random import randint
from threading import Thread

class MyThreadClass (Thread):
 def __init__(self, name, duration):
 Thread.__init__(self)
 self.name = name
 self.duration = duration
 def run(self):
 print ("---> " + self.name + \
 " running, belonging to process ID "\
 + str(os.getpid()) + "\n")
 time.sleep(self.duration)
 print ("---> " + self.name + " over\n")
def main():
 start_time = time.time()

 # Thread Creation
 thread1 = MyThreadClass("Thread#1 ", randint(1,10))
 thread2 = MyThreadClass("Thread#2 ", randint(1,10))
 thread3 = MyThreadClass("Thread#3 ", randint(1,10))

 # Thread Running
 thread1.start()
 thread2.start()
 thread3.start()

 # Thread joining
 thread1.join()
 thread2.join()
 thread3.join()

 # End 
 print("End")

 #Execution Time
 print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
 main()
```

让我们看看以下步骤：

1.  打开控制台并输入包含示例文件`winpdb_reborn_code_example.py`的文件夹名称：

```py
 python -m winpdb .\winpdb_reborn_code_example.py
```

这在macOS上也适用，但您必须使用Python的框架构建。如果您在Anaconda中使用Winpdb Reborn，只需使用`pythonw`而不是`python`来启动Winpdb Reborn会话。

1.  如果安装成功，Winpdb Reborn GUI应该会打开：

![](assets/2fba89a5-4840-407f-8266-63aab36b85ba.png)

Windpdb Reborn GUI

1.  如下截图所示，我们在第12行和第23行（红色高亮显示）分别插入了两个断点（使用断点菜单）：

***![](assets/5a2ccb87-7020-4948-8b20-80a3f393b972.png)***

代码断点要了解断点是什么，请转到本教程的*更多内容...*部分。

1.  保持在源窗口中，将鼠标放在第23行，我们在那里插入了第二个断点，然后按下*F8*键，然后按下*F5*键。断点允许执行代码直到所选行。如您所见，命名空间指示我们正在考虑`MyThreadClass`类的实例，带有`thread#1`作为参数：

![](assets/1a27af00-4468-449f-88ee-ee0c6b9bb6a4.png)

命名空间

1.  调试器的另一个基本功能是**步入**功能，它可以检查不仅正在调试的代码，还可以检查库函数和调用执行的子例程。

1.  在开始删除之前的断点（菜单|断点|清除所有）之前，在第28行插入新的断点：

![](assets/b27f1402-d9ab-4924-bfa2-df62a894b50c.png)

第28行断点

1.  最后，按下*F5*键，应用程序将执行到第**28**行的断点。

1.  然后，按下*F7*。在这里，源窗口不再显示我们的示例代码，而是显示我们正在使用的`threading`库（请参见下一张截图）。

1.  因此，断点功能以及“步入”功能不仅允许调试所涉及的代码，还允许检查所有库函数和任何其他使用的子例程：

![](assets/70c4308b-78e8-4633-8a9c-7435c583eb9c.png)

执行“步入”后的第28行源窗口

# 工作原理...

在这个第一个例子中，我们已经熟悉了Winpdb Reborn工具。这个调试环境（像每个一般的环境一样）允许您在精确的点停止程序执行，检查执行堆栈，变量的内容，创建的对象的状态等等。

要使用Winpdb Reborn，只需记住以下基本步骤：

1.  在源代码（源窗口）中设置一些断点。

1.  通过“步入”功能检查函数。

1.  查看变量的状态（命名空间窗口）和执行堆栈（堆栈窗口）。

通过简单地用鼠标左键双击所需的行（您会看到所选行被红色下划线标出）来设置断点。一般来说，不建议在同一行上有多个命令；否则，将无法将断点与其中一些命令关联起来。

使用鼠标右键，可以选择性地*禁用断点*而不是删除它们（红色高亮将消失）。要删除所有断点，可以使用“清除所有”命令，该命令位于断点菜单中，如前所述。

当到达第一个断点时，最好关注正在分析的程序点的以下视图：

+   堆栈视图显示了执行堆栈的内容，当前挂起的各种方法的所有实例都会显示出来。通常，堆栈底部的是主方法，堆栈顶部的是包含已达到断点的方法。

+   命名空间视图显示了方法的局部变量，并允许您检查值。如果变量是对象的引用，则可以找出对象的唯一标识符并检查其状态。

一般来说，程序的执行可以通过与Winpdb Reborn命令栏上的图标（或*Fx*键）相关联的不同模式来管理。

最后，我们将指出以下重要的执行方法：

+   步入（*F7*键）：这会一次一行地恢复程序的执行，并调用库方法或子例程。

+   返回（*F12*键）：这允许您在Step Into功能被激活的确切点恢复执行。

+   下一个（*F6*键）：这会一次一行地恢复程序的执行，而不会在调用的任何方法中停止。

+   运行到行（*F8*键）：这会运行程序直到在指定行停止（等待新命令）。

# 还有更多...

正如您在Winpdb Reborn GUI截图中看到的，GUI分为五个主要窗口：

+   命名空间：在这个窗口中，显示了程序定义并在源文件中使用的各种变量和标识符的名称。

+   线程：显示当前执行线程，并以**TID**（线程ID）字段、线程名称和线程状态为特征。

+   堆栈：这里显示了要分析的程序的执行堆栈。堆栈也被称为**后进先出**（**LIFO**）数据结构，因为最后插入的元素是第一个被移除的。当程序调用一个函数时，被调用的函数必须知道如何返回调用控制，因此调用函数的返回地址被输入到程序执行堆栈中。程序执行堆栈还包含每次调用函数时使用的本地变量的内存。

+   控制台：这是一个命令行界面，因此允许用户和Winpdb Reborn之间进行文本交互。

+   源代码：这个窗口显示要调试的源代码。通过沿着代码行滚动，也可以在感兴趣的代码行上按下*F9*来插入断点。

断点是一个非常基本的调试工具。实际上，它允许您运行程序，但可以在所需的点或在发生某些条件时中断它，以获取有关正在运行的程序的信息***。***

有多种调试策略。在这里，我们列出其中一些：

+   **重现错误**：识别导致错误的输入数据。

+   **简化错误**：识别导致错误的可能最简单的数据。

+   **分而治之**：以步进模式执行主要过程，直到出现异常。导致异常的方法是在能够找到问题之前执行的最后一个方法，因此我们可以通过对该特定调用进行步进调试，然后再次按照方法的指示进行步进。

+   **谨慎进行**：在调试过程中，不断比较变量的当前值与您期望的值。

+   **检查所有细节**：在调试时不要忽视细节。如果您注意到源代码中有任何差异，最好做个记录。

+   **纠正错误**：只有在确信已经理解了问题时才纠正错误。

# 另请参阅

可以在[http://heather.cs.ucdavis.edu/~matloff/winpdb.html#usewin](http://heather.cs.ucdavis.edu/~matloff/winpdb.html#usewin)找到一个很好的Winpdb Reborn教程。

# 与pdb交互

`pdb`是用于进行交互式调试的Python模块。

`pdb`的主要特点如下：

+   使用断点

+   逐行交互处理源代码

+   堆栈帧分析

调试器是通过`pdb`类实现的。因此，它可以很容易地扩展新功能。

# 准备就绪

不需要安装`pdb`，因为它是Python标准库的一部分。可以使用以下主要用法模式启动它：

+   与命令行交互

+   使用Python解释器

+   在代码中插入指令（即`pdb`语句）进行调试

# 与命令行交互

最简单的方法就是将程序的名称作为输入。例如，对于`pdb_test.py`程序，如下所示：

```py
class Pdb_test(object):
 def __init__(self, parameter):
 self.counter = parameter

 def go(self):
 for j in range(self.counter):
 print ("--->",j)
 return

if __name__ == '__main__':
 Pdb_test(10).go()
```

通过从命令行执行，`pdb`加载要分析的源文件，并在找到的第一个语句处停止执行。在这种情况下，调试停在第1行（即`Pdb_test`类的定义处）：

```py
python -m pdb pdb_test.py
> .../pdb_test.py(1)<module>()
-> class Pdb_test(object):
(Pdb)
```

# 使用Python解释器

`pdb`模块可以通过使用`run()`命令以交互模式使用：

```py
>>> import pdb_test
>>> import pdb
>>> pdb.run('pdb_test.Pdb_test(10).go()')
> <string>(1)<module>()
(Pdb)
```

在这种情况下，`run()`语句来自调试器，并且会在评估第一个表达式之前停止执行。

# 在代码中插入指令进行调试

对于长时间运行的进程，在程序执行的后期出现问题的情况下，通过使用`pdb set_trace()`指令在程序内部启动调试器会更加方便：

```py
import pdb

class Pdb_test(object):
 def __init__(self, parameter):
 self.counter = parameter
 def go(self):
 for j in range(self.counter):
 pdb.set_trace()
 print ("--->",j)
 return

if __name__ == '__main__':
 Pdb_test(10).go()
```

`set_trace()`可以在程序的任何地方调用进行调试。例如，可以基于条件、异常处理程序或特定的控制指令分支进行调用。

在这种情况下，输出如下：

```py
-> print ("--->",j)
(P**db)** 
```

代码运行在`pdb.set_trace()`语句完成后立即停止。

# 如何操作...

要与`pdb`交互，需要使用其语言，该语言允许您在代码中移动，检查和修改变量的值，插入断点或浏览堆栈调用：

1.  使用`where`命令（或者紧凑形式`w`）查看正在运行的代码所在的行以及调用堆栈。在这种情况下，这是在`pdb_test.py`模块的`go()`方法的第17行：

```py
> python -m pdb pdb_test.py
-> class Pdb_test(object):
(Pdb) where
 c:\python35\lib\bdb.py(431)run()
-> exec(cmd, globals, locals)
 <string>(1)<module>()
(Pdb)
```

1.  使用`list`检查当前位置附近的代码行（由箭头指示）。在默认模式下，列出当前行周围的11行（之前5行和之后5行）：

```py
 (Pdb) list
 1 -> class Pdb_test(object):
 2 def __init__(self, parameter):
 3 self.counter = parameter
 4
 5 def go(self):
 6 for j in range(self.counter):
 7 print ("--->",j)
 8 return
 9
 10 if __name__ == '__main__':
 11 Pdb_test(10).go()
```

1.  如果`list`接收两个参数，则它们被解释为要显示的第一行和最后一行：

```py
 (Pdb) list 3,9
 3 self.counter = parameter
 4
 5 def go(self):
 6 for j in range(self.counter):
 7 print ("--->",j)
 8 return
 9
```

1.  使用`up`（或`u`）移动到堆栈上更旧的帧，使用`down`（或`d`）移动到更近的堆栈帧：

```py
(Pdb) up
> <string>(1)<module>()
(Pdb) up
> c:\python35\lib\bdb.py(431)run()
-> exec(cmd, globals, locals)
(Pdb) down
> <string>(1)<module>()
(Pdb) down
>....\pdb_test.py(1)<module>()
-> class Pdb_test(object):
(Pdb)
```

# 工作原理...

调试活动是按照运行程序的流程（跟踪）进行的。在每一行代码中，编码器实时显示指令执行的操作和变量中记录的值。通过这种方式，开发人员可以检查一切是否正常工作，或者确定故障的原因。

每种编程语言都有自己的调试器。但是，并没有适用于所有编程语言的有效调试器，因为每种语言都有自己的语法和语法。调试器逐步执行源代码。因此，调试器必须了解语言的规则，就像编译器一样。

# 还有更多...

在使用Python调试器时要牢记的最有用的`pdb`命令及其简写形式如下：

| **命令** | **操作** |
| `args` | 打印当前函数的参数列表 |
| `break` | 创建断点（需要参数） |
| `continue` | 继续程序执行 |
| `help` | 列出命令（或帮助）的命令（作为参数） |
| `jump` | 设置要执行的下一行 |
| `list` | 打印当前行周围的源代码 |
| `next` | 继续执行，直到到达当前函数中的下一行或返回 |
| `step` | 执行当前行，停在第一个可能的位置 |
| `pp` | 漂亮打印表达式的值 |
| **`quit`**或**`exit`** | 从`pdb`中退出 |
| `return` | 继续执行，直到当前函数返回 |

# 另请参阅

您可以通过观看这个有趣的视频教程了解更多关于`pdb`的信息：[https://www.youtube.com/watch?v=bZZTeKPRSLQ](https://www.youtube.com/watch?v=bZZTeKPRSLQ)。

# 实现用于调试的rpdb

在某些情况下，适合在远程位置调试代码；也就是说，在与运行调试器的机器不在同一台机器上的位置。为此，开发了`rpdb`。这是`pdb`的包装器，它使用TCP套接字与外部世界通信。

# 准备就绪

首先安装`rpdb`需要使用`pip`的主要步骤。对于Windows操作系统，只需输入以下内容：

```py
C:\>pip install rpdb
```

然后，你需要确保你的机器上有一个可用的**telnet**客户端。在Windows 10中，如果你打开命令提示符并输入`telnet`，那么操作系统会以错误响应，因为它在安装中默认不包含。

让我们看看如何通过几个简单的步骤安装它：

1.  以管理员模式打开命令提示符。

1.  点击Cortana按钮，然后输入`cmd`。

1.  在出现的列表中，右键单击“命令提示符”项目，然后选择“以管理员身份运行”。

1.  然后，当以管理员身份运行命令提示符时，输入以下命令：

```py
dism /online /Enable-Feature /FeatureName:TelnetClient
```

1.  等待几分钟，直到安装完成。如果过程成功，那么你会看到这个：

![](assets/92df2ec3-4eb3-43be-a8a0-f6351d39f5de.png)

1.  现在，你可以直接从提示符使用telnet。通过输入`telnet`，应该会出现以下窗口：

![](assets/1e25886f-dca3-41a6-8e2a-efaf80ff5b05.png)

在下面的示例中，让我们看看如何使用`rpdb`进行远程调试。

# 如何做...

让我们执行以下步骤：

1.  考虑以下示例代码：

```py
import threading

def my_func(thread_number):
 return print('my_func called by thread N°
 {}'.format(thread_number))

def main():
 threads = []
 for i in range(10):
 t = threading.Thread(target=my_func, args=(i,))
 threads.append(t)
 t.start()
 t.join()

if __name__ == "__main__":
 main()
```

1.  要使用`rpdb`，你需要插入以下代码行（在`import threading`语句之后）。实际上，这三行代码使得通过端口`4444`和IP地址`127.0.0.1`上的远程客户端使用`rpdb`成为可能：

```py
import rpdb
debugger = rpdb.Rpdb(port=4444)
rpdb.Rpdb().set_trace()
```

1.  如果在插入了这三行代码以启用`rpdb`的示例代码中运行代码，那么你应该在Python命令提示符上看到以下消息：

```py
pdb is running on 127.0.0.1:4444
```

1.  然后，你可以通过以下telnet连接远程调试示例代码：

```py
telnet localhost 4444
```

1.  应该打开以下窗口：

![](assets/9f1b0949-3689-4ca7-9cd9-502cdaf6a028.png)

1.  在示例代码中，注意第7行的箭头。代码没有运行，只是等待执行指令：

![](assets/1aa03050-e016-4563-94a9-b3418cef7ef4.png)

1.  例如，在这里，我们执行代码并重复输入“next”语句：

```py
 (Pdb) next
> c:\users\giancarlo\desktop\python parallel programming cookbook 2nd edition\python parallel programming new book\chapter_x- code debugging\rpdb_code_example.py(10)<module>()
-> def main():
(Pdb) next
> c:\users\giancarlo\desktop\python parallel programming cookbook 2nd edition\python parallel programming new book\chapter_x- code debugging\rpdb_code_example.py(18)<module>()
-> if __name__ == "__main__":
(Pdb) next
> c:\users\giancarlo\desktop\python parallel programming cookbook 2nd edition\python parallel programming new book\chapter_x- code debugging\rpdb_code_example.py(20)<module>()
-> main()
(Pdb) next
my_func called by thread N 0
my_func called by thread N 1
my_func called by thread N 2
my_func called by thread N 3
my_func called by thread N 4
my_func called by thread N 5
my_func called by thread N 6
my_func called by thread N 7
my_func called by thread N 8
my_func called by thread N 9
--Return--
> c:\users\giancarlo\desktop\python parallel programming cookbook 2nd edition\python parallel programming new book\chapter_x- code debugging\rpdb_code_example.py(20)<module>()->None
-> main()
(Pdb)
```

程序完成后，你仍然可以运行一个新的调试部分。现在，让我们看看在下一节中`rpdp`是如何工作的。

# 它是如何工作的...

在本节中，我们将看到如何简单地通过使用`next`语句在代码中移动，该语句继续执行，直到达到或返回当前函数中的下一行。

要使用`rpdb`，请按照以下步骤进行：

1.  导入相关的`rpdb`库：

```py
import rpdb
```

1.  设置`debugger`参数，指定telnet端口以连接以运行调试器：

```py
debugger = rpdb.Rpdb(port=4444)
```

1.  调用`set_trace()`指令，这使得进入调试模式成为可能：

```py
rpdb.Rpdb().set_trace()
```

在我们的情况下，我们将`set_trace()`指令放在`debugger`实例之后。实际上，我们可以将它放在代码的任何地方；例如，如果条件满足，或者在由异常管理的部分内。

第二步，是打开命令提示符并启动`telnet`，并设置与示例代码中`debugger`参数定义中指定的相同端口值：

```py
telnet localhost 4444
```

可以通过使用一个小的命令语言与`rpdb`调试器进行交互，该语言允许在调用堆栈之间移动，检查和修改变量的值，并控制调试器执行自己的程序的方式。

# 还有更多...

通过在`Pdb`提示符下输入`help`命令，可以显示可以与`rpdb`交互的命令列表：

```py
> c:\users\giancarlo\desktop\python parallel programming cookbook 2nd edition\python parallel programming new book\chapter_x- code debugging\rpdb_code_example.py(7)<module>()
-> def my_func(thread_number):
(Pdb) help

Documented commands (type help <topic>):
========================================
EOF   c   d   h list q rv undisplay
a cl debug help ll quit s unt
alias clear disable ignore longlist r source until
args commands display interact n restart step up
b condition down j next return tbreak w
break cont enable jump p retval u whatis
bt continue exit l pp run unalias where

Miscellaneous help topics:
==========================
pdb exec

(Pdb)
```

在最有用的命令中，这是我们在代码中插入断点的方法：

1.  输入`b`和行号来设置断点。在这里，断点设置在第`5`行和第`10`行：

```py
 (Pdb) b 5
Breakpoint 1 at c:\users\giancarlo\desktop\python parallel programming cookbook 2nd edition\python parallel programming new book\chapter_x- code debugging\rpdb_code_example.py:5
(Pdb) b 10
Breakpoint 2 at c:\users\giancarlo\desktop\python parallel programming cookbook 2nd edition\python parallel programming new book\chapter_x- code debugging\rpdb_code_example.py:10
```

1.  只需输入`b`命令即可显示已实施的断点列表：

```py
 (Pdb) b
Num Type Disp Enb Where
1 breakpoint keep yes at c:\users\giancarlo\desktop\python parallel programming cookbook 2nd edition\python parallel programming new book\chapter_x- code debugging\rpdb_code_example.py:5
2 breakpoint keep yes at c:\users\giancarlo\desktop\python parallel programming cookbook 2nd edition\python parallel programming new book\chapter_x- code debugging\rpdb_code_example.py:10
(Pdb)
```

每次添加新的断点时，都会分配一个数字标识符。这些标识符用于启用、禁用和交互式地删除断点。要禁用断点，请使用`disable`命令，告诉调试器在达到该行时不要停止。断点不会被遗忘，而是被忽略。

# 另请参阅

您可以在这个网站上找到关于`pdb`和`rpdb`的大量信息：[https://github.com/spiside/pdb-tutorial](https://github.com/spiside/pdb-tutorial)。

在接下来的两个部分中，我们将看一些用于实施单元测试的Python工具：

+   `unittest`

+   `nose`

# 处理单元测试

`unittest`模块是标准Python库提供的。它具有一套广泛的工具和程序，用于执行单元测试。在本节中，我们将简要介绍`unittest`模块的工作原理。

单元测试由两部分组成：

+   管理所谓的*测试系统*的代码

+   测试本身

# 准备工作

最简单的`unittest`模块可以通过`TestCase`子类获得，必须重写或添加适当的方法。

一个简单的`unittest`模块可以如下组成：

```py
import unittest

class SimpleUnitTest(unittest.TestCase):

 def test(self):
 self.assertTrue(True)

if __name__ == '__main__':
 unittest.main()
```

要运行`unittest`模块，需要包含`unittest.main()`，而我们有一个单一的方法`test()`，如果`True`曾经是`False`，则失败。

通过执行上面的例子，您将得到以下结果：

```py
-----------------------------------------------------------
Ran 1 test in 0.005s

OK
```

测试成功，因此得到结果`OK`。

在接下来的部分，我们将更详细地介绍`unittest`模块的工作原理。特别是，我们想研究单元测试的可能结果。

# 如何做...

让我们看看如何通过这个例子来描述测试的结果：

1.  导入相关模块：

```py
import unittest
```

1.  定义`outcomesTest`类，它的参数是`TestCase`子类：

```py
class OutcomesTest(unittest.TestCase):
```

1.  我们定义的第一个方法是`testPass`：

```py
 def testPass(self):
 return
```

1.  这是`TestFail`方法：

```py
 def testFail(self):
 self.failIf(True)
```

1.  接下来，我们有`TestError`方法：

```py
 def testError(self):
 raise RuntimeError('test error!')
```

1.  最后，我们有`main`函数，通过它我们回顾我们的过程：

```py
if __name__ == '__main__':
 unittest.main()
```

# 工作原理...

在这个例子中，展示了`unittest`的单元测试可能的结果。

可能的结果如下：

+   `ERROR`：测试引发了除`AssertionError`之外的异常。没有明确的方法来通过测试，因此测试状态取决于异常的存在（或不存在）。

+   `FAILED`：测试未通过，并引发了`AssertionError`异常。

+   `OK`：测试通过。

输出如下：

```py
===========================================================
ERROR: testError (__main__.OutcomesTest)
-----------------------------------------------------------
Traceback (most recent call last):
 File "unittest_outcomes.py", line 15, in testError
 raise RuntimeError('Errore nel test!')
RuntimeError: Errore nel test!

===========================================================
FAIL: testFail (__main__.OutcomesTest)
-----------------------------------------------------------
Traceback (most recent call last):
 File "unittest_outcomes.py", line 12, in testFail
 self.failIf(True)
AssertionError

-----------------------------------------------------------
Ran 3 tests in 0.000s

FAILED (failures=1, errors=1)
```

大多数测试都确认条件的真实性。根据测试作者的角度和代码的期望结果是否得到验证，编写验证真实性的测试有不同的方式。如果代码产生一个可以评估为真的值，那么应该使用`failUnless ()`和`assertTrue ()`方法。如果代码产生一个假值，那么使用`failIf ()`和`assertFalse ()`方法更有意义：

```py
import unittest

class TruthTest(unittest.TestCase):

 def testFailUnless(self):
 self.failUnless(True)

 def testAssertTrue(self):
 self.assertTrue(True)

 def testFailIf(self):
 self.assertFalse(False)

 def testAssertFalse(self):
 self.assertFalse(False)

if __name__ == '__main__':
 unittest.main()
```

结果如下：

```py
> python unittest_failwithmessage.py -v
testFail (__main__.FailureMessageTest) ... FAIL

===========================================================
FAIL: testFail (__main__.FailureMessageTest)
-----------------------------------------------------------
Traceback (most recent call last):
 File "unittest_failwithmessage.py", line 9, in testFail
 self.failIf(True, 'Il messaggio di fallimento va qui')
AssertionError: Il messaggio di fallimento va qui

-----------------------------------------------------------
Ran 1 test in 0.000s

FAILED (failures=1)
robby@robby-desktop:~/pydev/pymotw-it/dumpscripts$ python unittest_truth.py -v
testAssertFalse (__main__.TruthTest) ... ok
testAssertTrue (__main__.TruthTest) ... ok
testFailIf (__main__.TruthTest) ... ok
testFailUnless (__main__.TruthTest) ... ok

-----------------------------------------------------------
Ran 4 tests in 0.000s

OK
```

# 还有更多...

如前所述，如果测试引发的异常不是`AssertionError`，那么它将被视为错误。这对于发现在编辑已存在匹配测试的代码时发生的错误非常有用。

然而，有些情况下，您可能希望运行一个测试来验证某些代码是否实际产生异常。例如，在传递无效值作为对象的属性时。在这种情况下，`failUnlessRaises()`比在代码中捕获异常更清晰：

```py
import unittest

def raises_error(*args, **kwds):
 print (args, kwds)
 raise ValueError\
 ('Valore non valido:'+ str(args)+ str(kwds))

class ExceptionTest(unittest.TestCase):
 def testTrapLocally(self):
 try:
 raises_error('a', b='c')
 except ValueError:
 pass
 else:
 self.fail('Non si vede ValueError')

 def testFailUnlessRaises(self):
 self.assertRaises\
 (ValueError, raises_error, 'a', b='c')

if __name__ == '__main__':
 unittest.main()
```

两者的结果是一样的。然而，使用`failUnlessRaises()`的第二个测试的结果更短：

```py
> python unittest_exception.py -v
testFailUnlessRaises (__main__.ExceptionTest) ... ('a',) {'b': 'c'}
ok
testTrapLocally (__main__.ExceptionTest) ...('a',) {'b': 'c'}
ok

-----------------------------------------------------------
Ran 2 tests in 0.000s

OK
```

# 另请参阅

有关Python测试的更多信息可以在[https://realpython.com/python-testing/](https://realpython.com/python-testing/)找到。

# 使用nose进行应用测试

`nose`是一个重要的Python模块，用于定义单元测试。它允许我们编写简单的测试函数，使用`unittest.TestCase`的子类，但也可以编写*不是`unittest.TestCase`的子类*的测试类。

# 准备工作

使用`pip`安装`nose`：

```py
C:\>pip install nose
```

可以通过以下步骤下载和安装源包：[https://pypi.org/project/nose/](https://pypi.org/project/nose/)

1.  解压源包。

1.  `cd`到新目录。

然后，输入以下命令：

```py
C:\>python setup.py install
```

`nose`的一个优点是自动从以下位置收集测试：

+   Python源文件

+   在工作目录中找到的目录和包

要指定要运行的测试，请在命令行上传递相关的测试名称：

```py
C:\>nosetests only_test_this.py
```

指定的测试名称可以是文件或模块名称，并且可以通过使用冒号分隔模块或文件名和测试用例名称来指示要运行的测试用例。文件名可以是相对的或绝对的。

以下是一些例子：

```py
C:\>nosetests test.module
C:\>nosetests another.test:TestCase.test_method
C:\>nosetests a.test:TestCase
C:\>nosetests /path/to/test/file.py:test_function
```

您还可以使用`-w`开关更改`nose`查找测试的工作目录：

```py
C:\>nosetests -w /path/to/tests
```

但是，请注意，对多个`-w`参数的支持现已不推荐使用，并将在将来的版本中删除。但是，可以通过在没有`-w`开关的情况下指定目标目录来获得相同的行为：

```py
C:\>nosetests /path/to/tests /another/path/to/tests
```

通过使用插件，可以进一步自定义测试选择和加载。

测试结果输出与`unittest`相同，除了额外的功能，例如错误类和插件提供的功能，如输出捕获和断言内省。

在下一节中，我们将介绍使用`nose`测试类的方法。

# 如何做...

让我们执行以下步骤：

1.  导入相关的`nose.tools`*：*

```py
from nose.tools import eq_ 
```

1.  然后，设置`TestSuite`类。在这里，通过`eq_`函数测试类的方法：

```py
class TestSuite:
 def test_mult(self):
 eq_(2*2,4)

 def ignored(self):
 eq_(2*2,3)
```

# 工作原理...

开发人员可以独立开发单元测试，但最好使用标准产品，如`unittest`，并遵循常见的测试实践。

如您从以下示例中所见，使用`eq_`函数设置了测试方法。这类似于`unittest`的`assertEquals`，它验证两个参数是否相等：

```py
 def test_mult(self):
 eq_(2*2,4)

 def ignored(self):
 eq_(2*2,3)
```

这种测试实践，尽管意图良好，但显然有明显的局限性，例如不能随时间重复（例如，当软件模块更改时）所谓的**回归测试**。

以下是输出：

```py
C:\>nosetests -v testset.py
testset.TestSuite.test_mult ... ok

-----------------------------------------------------------
Ran 1 tests in 0.001s

OK
```

一般来说，测试不能识别程序中的所有错误，对于单元测试也是如此，因为根据定义，单元测试无法识别集成错误、性能问题和其他与系统相关的问题。一般来说，单元测试与其他软件测试技术结合使用时更有效。

与任何形式的测试一样，即使单元测试也不能证明错误的不存在，而只能*突出显示*它们的存在。

# 还有更多...

软件测试是一个组合数学问题。例如，每个布尔测试都需要至少两个测试，一个用于真条件，一个用于假条件。可以证明，对于每个功能代码行，需要三到五行代码进行测试。因此，对于任何非平凡代码，测试所有可能的输入组合是不现实的，除非有专门的测试用例生成工具。

为了从单元测试中获得期望的好处，开发过程中需要严格的纪律。不仅要跟踪已开发和执行的测试，还要跟踪对所讨论的单元的功能代码以及所有其他单元所做的所有更改。使用版本控制系统是必不可少的。如果单元的后续版本未通过先前通过的测试，则版本控制系统允许您突出显示其间发生的代码更改。

# 另请参阅

`nose`的有效教程可在[https://nose.readthedocs.io/en/latest/index.html](https://nose.readthedocs.io/en/latest/index.html)找到。
