# 系统编程包

在本章中，我们将介绍 Python 中的主要模块，用于与 Python 解释器、操作系统和执行命令。我们将回顾如何使用文件系统，读取和创建文件。此外，我们将回顾线程管理和其他用于多线程和并发的模块。我们将以对`socket.io`模块实现异步服务器的回顾结束本章。

本章将涵盖以下主题：

+   介绍 Python 中的系统模块

+   使用文件系统

+   Python 中的线程

+   Python 中的多线程和并发

+   Python `Socket.io`

# 技术要求

本章的示例和源代码可在 GitHub 存储库的`chapter 2`文件夹中找到：[`github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security.`](https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security.)

您需要一些关于操作系统中的命令执行的基本知识，并在本地计算机上安装 Python 发行版。

# 介绍 Python 中的系统模块

在本节中，我们将解释 Python 中用于与 Python 解释器、操作系统以及使用子进程模块执行命令的主要模块。

# 系统模块

`sys`模块将允许我们与解释器进行交互，并且它包含了与正在进行的执行相关的大部分信息，由解释器更新，以及一系列函数和低级对象。

`**sys.argv**`包含执行脚本的参数列表。列表中的第一项是脚本的名称，后面是参数列表。

例如，我们可能希望在运行时解析命令行参数。sys.argv 列表包含所有命令行参数。sys.argv[0]索引包含 Python 解释器脚本的名称。argv 数组中的其余项目包含下一个命令行参数。因此，如果我们传递了三个额外的参数，sys.argv 应该包含四个项目。

您可以在`**sys_arguments.py**`文件中找到以下代码：

```py
import sys
print "This is the name of the script:",sys.argv[0]
print "The number of arguments is: ",len(sys.argv)
print "The arguments are:",str(sys.argv)
print "The first argument is ",sys.argv[1]
```

前面的脚本可以使用一些参数执行，例如以下内容：

```py
$ python sys_arguments.py one two three
```

如果我们使用三个参数执行前面的脚本，我们可以看到以下结果：

![](img/65980509-35c9-4adc-abe0-556185adc047.png)

在此示例中，我们获得了许多系统变量：

![](img/75dd2dae-c18d-4937-a697-07d4de09d0b8.png)

这些是恢复该信息的主要属性和方法：

+   **sys.platform**：返回当前操作系统

+   **sys.stdin,sys,stdout,sys.stderr**：分别指向标准输入、标准输出和标准错误输出的文件对象

+   **sys.version**：返回解释器版本

+   **sys.getfilesystemencoding()**：返回文件系统使用的编码

+   **sys.getdefaultencoding()**：返回默认编码

+   **sys.path**：返回解释器在导入指令使用或在不使用完整路径的文件名时搜索模块的所有目录列表

您可以在 Python 在线模块文档中找到更多信息：[`docs.python.org/library/sys`](http://docs.python.org/library/sys)。

# 操作系统模块

操作系统(os)模块是访问操作系统中不同函数的最佳机制。使用此模块将取决于所使用的操作系统。如果使用此模块，我们将不得不根据从一个操作系统切换到另一个操作系统来调整脚本。

该模块允许我们与操作系统环境、文件系统和权限进行交互。在此示例中，我们检查作为命令行参数传递的文本文件的名称是否存在于当前执行路径中，并且当前用户是否具有对该文件的读取权限。

您可以在`os`模块子文件夹中的`check_filename.py`文件中找到以下代码：

```py
import sys
import os

if len(sys.argv) == 2:
    filename = sys.argv[1]
    if not os.path.isfile(filename):
        print '[-] ' + filename + ' does not exist.'
        exit(0)
if not os.access(filename, os.R_OK):
        print '[-] ' + filename + ' access denied.'
        exit(0)
```

# 当前工作目录的内容

在这个例子中，`os`模块用于使用`os.getcwd()`方法列出当前工作目录的内容。

您可以在`os`模块子文件夹中的`show_content_directory.py`文件中找到以下代码：

```py
import os
pwd = os.getcwd()
list_directory = os.listdir(pwd)
for directory in list_directory:
    print directory
```

这是上述代码的主要步骤：

1.  导入`os`模块。

1.  使用`os`模块，调用`**os.getcwd()**`方法检索当前工作目录路径，并将该值存储在 pwd 变量中。

1.  获取当前目录路径的目录列表。使用`**os.listdir()**`方法获取当前工作目录中的文件名和目录。

1.  遍历列表目录以获取文件和目录。

以下是从操作系统模块中恢复信息的主要方法：

+   **os.system()**：允许我们执行 shell 命令

+   **os.listdir(path)**：返回作为参数传递的目录的内容列表

+   **os.walk(path)**：导航提供的路径目录中的所有目录，并返回三个值：路径目录，子目录的名称以及当前目录路径中的文件名的列表。

在这个例子中，我们检查当前路径内的文件和目录。

您可以在`os`模块子文件夹中的**`check_files_directory.py`**文件中找到以下代码：

```py
import os
for root,dirs,files in os.walk(".",topdown=False):
    for name in files:
        print(os.path.join(root,name))
    for name in dirs:
        print name
```

# 确定操作系统

下一个脚本确定代码是否在 Windows OS 或 Linux 平台上运行。`**platform.system()**`方法告诉我们正在运行的操作系统。根据返回值，我们可以看到在 Windows 和 Linux 中 ping 命令是不同的。Windows OS 使用 ping -n 1 发送一个 ICMP ECHO 请求的数据包，而 Linux 或其他操作系统使用 ping -c 1。

您可以在`os`模块子文件夹中的**`operating_system.py`**文件中找到以下代码：

```py
import os
import platform
operating_system = platform.system()
print operating_system
if (operating_system == "Windows"):
    ping_command = "ping -n 1 127.0.0.1"
elif (operating_system == "Linux"):
    ping_command = "ping -c 1 127.0.0.1"
else :
    ping_command = "ping -c 1 127.0.0.1"
print ping_command
```

# 子进程模块

标准的子进程模块允许您从 Python 调用进程并与它们通信，将数据发送到输入(stdin)，并接收输出信息(stdout)。使用此模块是执行操作系统命令或启动程序（而不是传统的`os.system()`）并可选择与它们交互的推荐方法。

使用子进程运行子进程很简单。在这里，**Popen**构造函数**启动进程**。您还可以将数据从 Python 程序传输到子进程并检索其输出。使用**help(subprocess)**命令，我们可以看到相关信息：

![](img/5cb9d517-d7b9-4466-80cc-23aadaf26abc.png)

执行命令或调用进程的最简单方法是通过`call()`函数（从 Python 2.4 到 3.4）或`run()`（对于 Python 3.5+）。例如，以下代码执行列出当前路径中文件的命令。

您可以在`subprocess`子文件夹中的**`SystemCalls.py`**文件中找到此代码：

```py
import os
import subprocess
# using system
os.system("ls -la")
# using subprocess
subprocess.call(["ls", "-la"])
```

为了能够使用终端命令（例如清除或 cls 清理控制台，cd 移动到目录树中等），需要指定 shell = True 参数：

```py
>> subprocess.call("cls", shell=True)
```

在这个例子中，它要求用户写下他们的名字，然后在屏幕上打印一个问候语。通过子进程，我们可以使用 Popen 方法调用它，以编程方式输入一个名字，并将问候语作为 Python 字符串获取。

`Popen()`实例包括`terminate()`和`kill()`方法，分别用于终止或杀死进程。Linux 的发行版区分 SIGTERM 和 SIGKILL 信号：

```py
>>> p = subprocess.Popen(["python", "--version"])
>>> p.terminate()
```

与调用函数相比，Popen 函数提供了更多的灵活性，因为它在新进程中执行命令作为子程序。例如，在 Unix 系统上，该类使用`os.execvp()`。在 Windows 上，它使用 Windows `CreateProcess()`函数。

您可以在官方文档中找到有关 Popen 构造函数和 Popen 类提供的方法的更多信息：[`docs.python.org/2/library/subprocess.html#popen-constructor`](https://docs.python.org/3.5/library/subprocess.html#popen-constructor)。

在这个例子中，我们使用`subprocess`模块调用`ping`命令，并获取该命令的输出，以评估特定 IP 地址是否响应`ECHO_REPLY`。此外，我们使用`sys`模块来检查我们执行脚本的操作系统。

您可以在`PingScanNetWork.py`文件的 subprocess 子文件夹中找到以下代码：

```py
#!/usr/bin/env python
from subprocess import Popen, PIPE
import sys
import argparse
parser = argparse.ArgumentParser(description='Ping Scan Network')

# Main arguments
parser.add_argument("-network", dest="network", help="NetWork segment[For example 192.168.56]", required=True)
parser.add_argument("-machines", dest="machines", help="Machines number",type=int, required=True)

parsed_args = parser.parse_args()    
for ip in range(1,parsed_args.machines+1):
    ipAddress = parsed_args.network +'.' + str(ip)
    print "Scanning %s " %(ipAddress)
    if sys.platform.startswith('linux'):
    # Linux
        subprocess = Popen(['/bin/ping', '-c 1 ', ipAddress], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    elif sys.platform.startswith('win'):
    # Windows
        subprocess = Popen(['ping', ipAddress], stdin=PIPE, stdout=PIPE, stderr=PIPE)
stdout, stderr= subprocess.communicate(input=None)
print stdout
if "Lost = 0" in stdout or "bytes from " in stdout:
    print "The Ip Address %s has responded with a ECHO_REPLY!" %(stdout.split()[1])
```

要执行此脚本，我们需要将我们正在分析的网络和我们想要检查的机器编号作为参数传递：

```py
python PingScanNetWork.py -network 192.168.56 -machines 1
```

以下是扫描 129.168.56 网络和一个机器的结果：

![](img/6b324c6f-76c0-4706-88dc-bd48a61849d2.png)

# 在 Python 中处理文件系统

在本节中，我们解释了 Python 中用于处理文件系统、访问文件和目录、读取和创建文件以及使用和不使用上下文管理器的主要模块。

# 访问文件和目录

在本节中，我们将回顾如何处理文件系统并执行诸如浏览目录或逐个读取每个文件的任务。

# 递归浏览目录

在某些情况下，需要递归迭代主目录以发现新目录。在这个例子中，我们看到如何递归浏览目录并检索该目录中所有文件的名称：

```py
import os
 # you can change the "/" to a directory of your choice
 for file in os.walk("/"):
    print(file)
```

# 检查特定路径是否为文件或目录

我们可以检查某个字符串是否为文件或目录。为此，我们可以使用`os.path.isfile()`方法，如果是文件则返回`True`，如果是目录则返回`False`：

```py
 >>> import os
 >>> os.path.isfile("/")
 False
 >>> os.path.isfile("./main.py")
 True
```

# 检查文件或目录是否存在

如果您想要检查当前工作路径目录中是否存在文件，可以使用`os.path.exists()`函数，将要检查的文件或目录作为参数传递：

```py
 >>> import os
 >>> os.path.exists("./main.py")
 True
 >>> os.path.exists("./not_exists.py")
 False
```

# 在 Python 中创建目录

您可以使用`os.makedirs()`函数创建自己的目录：

```py
 >>> if not os.path.exists('my_dir'):
 >>>    os.makedirs('my_dir')
```

此代码检查`my_dir`目录是否存在；如果不存在，它将调用`os.makedirs` **('**`my_dir`**')**来创建该目录。

如果您在验证目录不存在后创建目录，在执行对`os.makedirs`('`my_dir`')的调用之前，可能会生成错误或异常。

如果您想要更加小心并捕获任何潜在的异常，您可以将对`os.makedirs('`my_dir`')`的调用包装在**try...except**块中：

```py
if not os.path.exists('my_dir'):
    try:
        os.makedirs('my_dir')
    except OSError as e:
       print e
```

# 在 Python 中读写文件

现在我们将回顾读取和写入文件的方法。

# 文件方法

这些是可以在文件对象上使用的函数。

+   **file.write(string)**：将字符串打印到文件，没有返回。

+   **file.read([bufsize])**：从文件中读取最多“bufsize”字节数。如果没有缓冲区大小选项运行，则读取整个文件。

+   **file.readline([bufsize])**：从文件中读取一行（保留换行符）。

+   **file.close()**：关闭文件并销毁文件对象。Python 会自动执行这个操作，但当您完成一个文件时，这仍然是一个好习惯。

# 打开文件

处理文件的经典方法是使用`open()`方法。这种方法允许您打开一个文件，返回一个文件类型的对象：

**open(name[, mode[, buffering]])**

文件的打开模式可以是 r（读取）、w（写入）和 a（追加）。我们可以在这些模式中添加 b（二进制）、t（文本）和+（打开读写）模式。例如，您可以在选项中添加“+”，这允许使用同一个对象进行读/写：

```py
>>> my_file=open("file.txt","r”)
```

要读取文件，我们有几种可能性：

+   `readlines()`方法读取文件的所有行并将它们连接成一个序列。如果您想一次读取整个文件，这个方法非常有用：` >>> allLines = file.readlines()`。

+   如果我们想逐行读取文件，我们可以使用`readline()`方法。这样，如果我们想逐行读取文件的所有行，我们可以将文件对象用作迭代器：

```py
>>> for line in file:
>>>  print line
```

# 使用上下文管理器

在 Python 中创建文件的多种方法，但最干净的方法是使用**with**关键字，这种情况下我们使用**上下文管理器方法**。

最初，Python 提供了 open 语句来打开文件。当我们使用 open 语句时，Python 将开发者的责任委托给开发者，当不再需要使用文件时关闭文件。这种做法会导致错误，因为开发者有时会忘记关闭文件。自 Python 2.5 以来，开发者可以使用 with 语句安全地处理这种情况。**with 语句**会自动关闭文件，即使发生异常也是如此。

with 命令允许对文件进行多种操作：

```py
>>> with open("somefile.txt", "r") as file:
>>> for line in file:
>>> print line
```

这样，我们就有了优势：文件会自动关闭，我们不需要调用`close()`方法。

您可以在文件名为`**create_file.py**`的文件中找到下面的代码

```py
def main():
    with open('test.txt', 'w') as file:
        file.write("this is a test file")

 if __name__ == '__main__':
    main()
```

上面的脚本使用上下文管理器打开一个文件，并将其作为文件对象返回。在这个块中，我们调用 file.write("this is a test file")，将其写入我们创建的文件。在这种情况下，with 语句会自动处理文件的关闭，我们不需要担心它。

有关 with 语句的更多信息，您可以查看官方文档[`docs.python.org/2/reference/compound_stmts.html#the-with-statement`](https://docs.python.org/2/reference/compound_stmts.html#the-with-statement)。

# 逐行读取文件

我们可以逐行迭代文件：

```py
>>> with open('test.txt', 'r') as file:
>>>    for line in file:
>>>        print(line)
```

在这个例子中，当我们处理文件时，我们将所有这些功能与异常管理结合起来。

您可以在**`create_file_exceptions.py`**文件中找到以下代码：

```py
def main():
    try:
        with open('test.txt', 'w') as file:
            file.write("this is a test file")
    except IOError as e:
        print("Exception caught: Unable to write to file", e)
    except Exception as e:
        print("Another error occurred ", e)
    else:
        print("File written to successfully")

if __name__ == '__main__':
    main()
```

# Python 中的线程

在本节中，我们将介绍线程的概念以及如何使用`Python`模块管理它们。

# 线程介绍

线程是可以由操作系统调度并在单个核心上以并发方式或在多个核心上以并行方式执行的流。线程可以与共享资源（如内存）交互，并且它们也可以同时或甚至并行地修改事物。

# 线程类型

有两种不同类型的线程：

+   **内核级线程**：低级线程，用户无法直接与它们交互。

+   **用户级线程**：高级线程，我们可以在我们的代码中与它们交互。

# 进程与线程

进程是完整的程序。它们有自己的 PID（进程 ID）和 PEB（进程环境块）。这些是进程的主要特点：

+   进程可以包含多个线程。

+   如果一个进程终止，相关的线程也会终止。

线程是一个类似于进程的概念：它们也是正在执行的代码。然而，线程是在一个进程内执行的，并且进程的线程之间共享资源，比如内存。这些是线程的主要特点：

+   线程只能与一个进程关联。

+   进程可以在线程终止后继续（只要还有至少一个线程）。

# 创建一个简单的线程

线程是程序在并行执行任务的机制。因此，在脚本中，我们可以在单个处理器上多次启动相同的任务。

在 Python 中处理线程有两种选择：

+   线程模块提供了编写多线程程序的原始操作。

+   线程模块提供了更方便的接口。

`thread`模块将允许我们使用多个线程：

在这个例子中，我们创建了四个线程，每个线程在屏幕上打印不同的消息，这些消息作为参数传递给`thread_message(message)`方法。

您可以在 threads 子文件夹中的**`threads_init.py`**文件中找到以下代码：

```py
import thread
import time

num_threads = 4

def thread_message(message):
  global num_threads
  num_threads -= 1
  print('Message from thread %s\n' %message)

while num_threads > 0:
  print "I am the %s thread" %num_threads
  thread.start_new_thread(thread_message,("I am the %s thread" %num_threads,))
  time.sleep(0.1)
```

如果我们调用 help(thread)命令，可以查看更多关于`start_new_thread()`方法的信息：

![](img/9fa9be64-d35f-47de-a4fb-d15a493af04b.png)

# 线程模块

除了`thread`模块，我们还有另一种使用`threading`模块的方法。线程模块依赖于`thread`模块为我们提供更高级、更完整和面向对象的 API。线程模块在某种程度上基于 Java 线程模型。

线程模块包含一个 Thread 类，我们必须扩展它以创建自己的执行线程。run 方法将包含我们希望线程执行的代码。如果我们想要指定自己的构造函数，它必须调用 threading.`Thread .__ init __ (self)`来正确初始化对象。

在 Python 中创建新线程之前，我们要检查 Python Thread 类的 init 方法构造函数，并查看需要传递的参数：

```py
# Python Thread class Constructor
 def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
```

Thread 类构造函数接受五个参数作为参数：

+   **group**：保留给未来扩展的特殊参数。

+   **target**：要由 run 方法()调用的可调用对象。

+   **name**：我们线程的名称。

+   **args**：用于目标调用的参数元组。

+   **kwargs**：调用基类构造函数的字典关键字参数。

如果我们在 Python 解释器控制台中调用**help(threading)**命令，可以获取有关`init()`方法的更多信息：

![](img/f0c2a3d8-c36e-4ec1-9e17-1c6c8ab60a9d.png)

让我们创建一个简单的脚本，然后用它来创建我们的第一个线程：

在 threads 子文件夹中的**`threading_init.py`**文件中，您可以找到以下代码：

```py
import threading

def myTask():
    print("Hello World: {}".format(threading.current_thread()))

 # We create our first thread and pass in our myTask function
 myFirstThread = threading.Thread(target=myTask)
 # We start out thread
 myFirstThread.start()
```

为了使线程开始执行其代码，只需创建我们刚刚定义的类的实例并调用其 start 方法即可。主线程的代码和我们刚刚创建的线程的代码将同时执行。

我们必须实例化一个 Thread 对象并调用`start()`方法。Run 是我们希望在每个线程内并行运行的逻辑，因此我们可以使用`run()`方法启动一个新线程。此方法将包含我们希望并行执行的代码。

在此脚本中，我们正在创建四个线程。

在 threads 子文件夹中的**`threading_example.py`**文件中，您可以找到以下代码：

```py
import threading

class MyThread(threading.Thread):

    def __init__ (self, message):
        threading.Thread.__init__(self)
        self.message = message

    def run(self):
        print self.message

threads = []
for num in range(0, 5):
    thread = MyThread("I am the "+str(num)+" thread")
    thread.name = num
    thread.start()
```

我们还可以使用`thread.join()`方法等待线程终止。join 方法用于使执行调用的线程在被调用的线程结束之前被阻塞。在这种情况下，它用于使主线程在子线程之前不结束其执行，否则可能导致某些平台在子线程结束执行之前终止子线程。join 方法可以接受浮点数作为参数，表示等待的最大秒数。

在 threads 子文件夹中的**`threading_join.py`**文件中，您可以找到以下代码：

```py
import threading

class thread_message(threading.Thread):
    def __init__ (self, message):
         threading.Thread.__init__(self)
         self.message = message

    def run(self):
         print self.message

threads = []
for num in range(0, 10):
 thread = thread_message("I am the "+str(num)+" thread")
 thread.start()
 threads.append(thread)

# wait for all threads to complete by entering them
for thread in threads:
 thread.join()
```

# Python 中的多线程和并发

在本节中，我们将介绍多线程和并发的概念，以及如何使用 Python 模块来管理它们。

# 多线程简介

多线程应用程序的理念是允许我们在额外的线程上有代码的副本并执行它们。这允许程序同时执行多个操作。此外，当一个进程被阻塞时，例如等待输入/输出操作，操作系统可以将计算时间分配给其他进程。

当我们提到多处理器时，我们指的是可以同时执行多个线程的处理器。这些处理器通常有两个或更多个线程，在内核中积极竞争执行时间，当一个线程停止时，处理内核开始执行另一个线程。

这些子进程之间的上下文变化非常快，给人一种计算机在并行运行进程的印象，这使我们能够进行多任务处理。

# Python 中的多线程

Python 有一个 API，允许我们使用多个线程编写应用程序。为了开始多线程，我们将在`python`类内部创建一个新线程，并将其命名为**`ThreadWorker.py`**。这个类继承自`threading.Thread`，并包含管理一个线程的代码：

```py
import threading
class ThreadWorker(threading.Thread):
    # Our workers constructor
    def __init__(self):
        super(ThreadWorker, self).__init__()
    def run(self):
        for i in range(10):
           print(i)
```

现在我们有了我们的线程工作类，我们可以开始在我们的主类上工作了。创建一个新的 python 文件，命名为`main.py`，并放入以下代码：

```py
import threading
from ThreadWorker import ThreadWorker 
def main():
    # This initializes ''thread'' as an instance of our Worker Thread
   thread = ThreadWorker()
    # This is the code needed to run our thread
    thread.start()

if __name__ == "__main__":  
    main()
```

有关线程模块的文档可在[`docs.python.org/3/library/threading.html`](https://docs.python.org/3/library/threading.html)找到。

# 经典 Python 线程的限制

Python 经典线程的一个主要问题是它们的执行并不完全是异步的。众所周知，Python 线程的执行并不完全是并行的，**添加多个线程**通常会使执行时间加倍。因此，执行这些任务会减少执行时间。

Python 中线程的执行受 GIL（全局解释器锁）控制，因此一次只能执行一个线程，无论机器有多少个处理器。

这样可以更容易地为 Python 编写 C 扩展，但它的缺点是会大大限制性能，因此尽管如此，在 Python 中，有时我们可能更有兴趣使用进程而不是线程，后者不会受到这种限制的影响。

默认情况下，线程更改是在每 10 个字节码指令执行时进行的，尽管可以使用 sys.setcheckinterval 函数进行修改。它还在线程使用 time.sleep 休眠或开始输入/输出操作时进行更改，这可能需要很长时间才能完成，因此，如果不进行更改，CPU 将长时间没有执行代码，等待 I/O 操作完成。

为了最小化 GIL 对我们应用程序性能的影响，最好使用-O 标志调用解释器，这将生成一个优化的字节码，指令更少，因此上下文更改更少。我们还可以考虑使用进程而不是线程，正如我们讨论的那样，比如`ProcessPoolExecutors`模块。

有关**GIL**的更多信息，请参阅[`wiki.python.org/moin/GlobalInterpreterLock`](https://wiki.python.org/moin/GlobalInterpreterLock)。

# 使用 ThreadPoolExecutor 在 Python 中进行并发

在这一部分，我们回顾了提供执行任务异步的接口的**ThreadPoolExecutor**类。

# 创建 ThreadPoolExecutor

我们可以用 init 构造函数定义我们的**ThreadPoolExecutor**对象：

```py
executor = ThreadPoolExecutor(max_workers=5)
```

如果我们将最大工作线程数作为参数传递给构造函数，我们就可以创建 ThreadPoolExecutor。在这个例子中，我们已经将最大线程数定义为五，这意味着这组子进程只会同时有五个线程在工作。

为了使用我们的`ThreadPoolExecutor`，我们可以调用`submit()`方法，该方法以一个函数作为参数，以异步方式执行该代码：

`executor.submit(myFunction())`

# ThreadPoolExecutor 实践

在这个例子中，我们分析了`ThreadPoolExecutor`类的对象的创建。我们定义了一个`view_thread()`函数，允许我们使用`threading.get_ident()`方法显示当前线程标识符。

我们定义了我们的主函数，其中 executor 对象被初始化为 ThreadPoolExecutor 类的一个实例，并在这个对象上执行一组新的线程。然后我们使用`threading.current_thread()`方法获得已执行的线程。

您可以在 concurrency 子文件夹中的**threadPoolConcurrency.py**文件中找到以下代码：

```py
#python 3
from concurrent.futures import ThreadPoolExecutor
import threading
import random

def view_thread():
 print("Executing Thread")
 print("Accesing thread : {}".format(threading.get_ident()))
 print("Thread Executed {}".format(threading.current_thread()))

def main():
 executor = ThreadPoolExecutor(max_workers=3)
 thread1 = executor.submit(view_thread)
 thread1 = executor.submit(view_thread)
 thread3 = executor.submit(view_thread)

if __name__ == '__main__':
 main()

```

我们看到脚本输出中的三个不同值是三个不同的线程标识符，我们获得了三个不同的守护线程：

![](img/459803de-3951-439d-8852-1c26a2480765.png)

# 使用上下文管理器执行 ThreadPoolExecutor

另一种实例化 ThreadPoolExecutor 的方法是使用`with`语句作为上下文管理器：

`with ThreadPoolExecutor(max_workers=2) as executor:`

在这个例子中，在我们的主函数中，我们将 ThreadPoolExecutor 作为上下文管理器使用，然后两次调用`future = executor.submit(message, (message))`来在线程池中处理每条消息。

你可以在 concurrency 子文件夹的`threadPoolConcurrency2.py`文件中找到以下代码：

```py
from concurrent.futures import ThreadPoolExecutor

def message(message):
 print("Processing {}".format(message))

def main():
 print("Starting ThreadPoolExecutor")
 with ThreadPoolExecutor(max_workers=2) as executor:
   future = executor.submit(message, ("message 1"))
   future = executor.submit(message, ("message 2"))
 print("All messages complete")

if __name__ == '__main__':
 main()
```

# Python Socket.io

在本节中，我们将回顾如何使用 socket.io 模块来创建基于 Python 的 Web 服务器。

# 介绍 WebSockets

WebSockets 是一种技术，通过 TCP 连接在客户端和服务器之间提供实时通信，并消除了客户端不断检查 API 端点是否有更新或新内容的需要。客户端创建到 WebSocket 服务器的单个连接，并保持等待以监听来自服务器的新事件或消息。

Websockets 的主要优势在于它们更有效，因为它们减少了网络负载，并以消息的形式向大量客户端发送信息。

# aiohttp 和 asyncio

aiohttp 是一个在 asyncio 中构建服务器和客户端应用程序的库。该库原生使用 websockets 的优势来异步通信应用程序的不同部分。

文档可以在[`aiohttp.readthedocs.io/en/stable`](http://aiohttp.readthedocs.io/en/stable/)找到。

asyncio 是一个帮助在 Python 中进行并发编程的模块。在 Python 3.6 中，文档可以在[`docs.python.org/3/library/asyncio.html`](https://docs.python.org/3/library/asyncio.html)找到。

# 使用 socket.io 实现服务器

Socket.IO 服务器可以在官方 Python 存储库中找到，并且可以通过 pip 安装：`pip install python-socketio.`

完整的文档可以在[`python-socketio.readthedocs.io/en/latest/`](https://python-socketio.readthedocs.io/en/latest/)找到。

以下是一个在 Python 3.5 中工作的示例，我们在其中使用 aiohttp 框架实现了一个 Socket.IO 服务器：

```py
from aiohttp import web
import socketio

socket_io = socketio.AsyncServer()
app = web.Application()
socket_io.attach(app)

async def index(request):
        return web.Response(text='Hello world from socketio' content_type='text/html')

# You will receive the new messages and send them by socket
@socket_io.on('message')
def print_message(sid, message):
    print("Socket ID: " , sid)
    print(message)

app.router.add_get('/', index)

if __name__ == '__main__':
    web.run_app(app)
```

在上面的代码中，我们实现了一个基于 socket.io 的服务器，该服务器使用了 aiohttp 模块。正如你在代码中看到的，我们定义了两种方法，`index()`方法，它将在“/”根端点接收到请求时返回一个响应消息，以及一个`print_message()`方法，其中包含`@socketio.on('message')`注释。这个注释使函数监听消息类型的事件，当这些事件发生时，它将对这些事件进行操作。

# 总结

在本章中，我们学习了 Python 编程的主要系统模块，如用于操作系统的 os 模块，用于文件系统的 sys 模块，以及用于执行命令的 sub-proccess 模块。我们还回顾了如何处理文件系统，读取和创建文件，管理线程和并发。

在下一章中，我们将探讨用于解析 IP 地址和域的 socket 包，并使用 TCP 和 UDP 协议实现客户端和服务器。

# 问题

1.  允许我们与 Python 解释器交互的主要模块是什么？

1.  允许我们与操作系统环境、文件系统和权限交互的主要模块是什么？

1.  用于列出当前工作目录内容的模块和方法是什么？

1.  执行命令或通过 call()函数调用进程的模块是什么？

1.  在 Python 中处理文件和管理异常的简单和安全方法是什么？

1.  进程和线程之间的区别是什么？

1.  Python 中用于创建和管理线程的主要模块是什么？

1.  Python 在处理线程时存在的限制是什么？

1.  哪个类提供了一个高级接口，用于以异步方式执行输入/输出任务？

1.  线程模块中的哪个函数确定了哪个线程执行了？

# 进一步阅读

在这些链接中，您将找到有关提到的工具的更多信息，以及我们讨论的一些模块的官方 Python 文档：

+   [`docs.python.org/3/tutorial/inputoutput.html`](https://docs.python.org/3/tutorial/inputoutput.html)

+   [`docs.python.org/3/library/threading.html`](https://docs.python.org/3/library/threading.html)

+   [`wiki.python.org/moin/GlobalInterpreterLock`](https://wiki.python.org/moin/GlobalInterpreterLock)

+   [`docs.python.org/3/library/concurrent.futures.html`](https://docs.python.org/3/library/concurrent.futures.html)

对于对使用 aiohttp 和 asyncio 等技术进行 Web 服务器编程感兴趣的读者，应该查看诸如 Flask（[`flask.pocoo.org`](http://flask.pocoo.org)）和 Django（[`www.djangoproject.com`](https://www.djangoproject.com)）等框架。
