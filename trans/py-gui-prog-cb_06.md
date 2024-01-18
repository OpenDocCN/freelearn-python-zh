# 第6章。线程和网络

在本章中，我们将使用Python 3创建线程、队列和TCP/IP套接字。

+   如何创建多个线程

+   启动一个线程

+   停止一个线程

+   如何使用队列

+   在不同模块之间传递队列

+   使用对话框小部件将文件复制到您的网络

+   使用TCP/IP通过网络进行通信

+   使用URLOpen从网站读取数据

# 介绍

在本章中，我们将使用线程、队列和网络连接扩展我们的Python GUI的功能。

### 注意

tkinter GUI是单线程的。每个涉及休眠或等待时间的函数都必须在单独的线程中调用，否则tkinter GUI会冻结。

当我们在Windows任务管理器中运行我们的Python GUI时，我们可以看到一个新的`python.exe`进程已经启动。

当我们给我们的Python GUI一个`.pyw`扩展名时，然后创建的进程将是`python.pyw`，可以在任务管理器中看到。

当创建一个进程时，该进程会自动创建一个主线程来运行我们的应用程序。这被称为单线程应用程序。

对于我们的Python GUI，单线程应用程序将导致我们的GUI在调用较长时间的任务时变得冻结，比如点击一个有几秒钟休眠的按钮。

为了保持我们的GUI响应，我们必须使用多线程，这就是我们将在本章中学习的内容。

我们还可以通过创建多个Python GUI的实例来创建多个进程，可以在任务管理器中看到。

进程在设计上是相互隔离的，彼此不共享公共数据。为了在不同进程之间进行通信，我们必须使用**进程间通信**（**IPC**），这是一种高级技术。

另一方面，线程确实共享公共数据、代码和文件，这使得在同一进程内的线程之间的通信比使用IPC更容易。

### 注意

关于线程的很好的解释可以在这里找到：[https://www.cs.uic.edu/~jbell/CourseNotes/OperatingSystems/4_Threads.html](https://www.cs.uic.edu/~jbell/CourseNotes/OperatingSystems/4_Threads.html)

在本章中，我们将学习如何保持我们的Python GUI响应，并且不会冻结。

# 如何创建多个线程

我们将使用Python创建多个线程。这是为了保持我们的GUI响应而必要的。

### 注意

线程就像编织由纱线制成的织物，没有什么可害怕的。

## 准备就绪

多个线程在同一计算机进程内存空间内运行。不需要进程间通信（IPC），这会使我们的代码变得复杂。在本节中，我们将通过使用线程来避免IPC。

## 如何做...

首先，我们将增加我们的`ScrolledText`小部件的大小，使其更大。让我们将`scrolW`增加到40，`scrolH`增加到10。

```py
# Using a scrolled Text control
scrolW  = 40; scrolH  =  10
self.scr = scrolledtext.ScrolledText(self.monty, width=scrolW, height=scrolH, wrap=tk.WORD)
self.scr.grid(column=0, row=3, sticky='WE', columnspan=3)
```

当我们现在运行结果的GUI时，`Spinbox`小部件相对于其上方的`Entry`小部件是居中对齐的，这看起来不好。我们将通过左对齐小部件来改变这一点。

在`grid`控件中添加`sticky='W'`，以左对齐`Spinbox`小部件。

```py
# Adding a Spinbox widget using a set of values
self.spin = Spinbox(self.monty, values=(1, 2, 4, 42, 100), width=5, bd=8, command=self._spin) 
self.spin.grid(column=0, row=2, sticky='W')
```

GUI可能看起来还不错，所以下一步，我们将增加`Entry`小部件的大小，以获得更平衡的GUI布局。

将宽度增加到24，如下所示：

```py
# Adding a Textbox Entry widget
self.name = tk.StringVar()
nameEntered = ttk.Entry(self.monty, width=24, textvariable=self.name)
nameEntered.grid(column=0, row=1, sticky='W')
```

让我们也稍微增加`Combobox`的宽度到14。

```py
ttk.Label(self.monty, text="Choose a number:").grid(column=1, row=0)
number = tk.StringVar()
numberChosen = ttk.Combobox(self.monty, width=14, textvariable=number)
numberChosen['values'] = (1, 2, 4, 42, 100)
numberChosen.grid(column=1, row=1)
numberChosen.current(0)
```

运行修改和改进的代码会导致一个更大的GUI，我们将在本节和下一节中使用它。

![如何做...](graphics/B04829_06_01.jpg)

为了在Python中创建和使用线程，我们必须从threading模块中导入`Thread`类。

```py
#======================
# imports
#======================
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import Menu  
from tkinter import Spinbox
import B04829_Ch06_ToolTip as tt

from threading import Thread

GLOBAL_CONST = 42
```

让我们在`OOP`类中添加一个在线程中创建的方法。

```py
class OOP():
    def methodInAThread(self):
        print('Hi, how are you?')
```

现在我们可以在代码中调用我们的线程方法，将实例保存在一个变量中。

```py
#======================
# Start GUI
#======================
oop = OOP()

# Running methods in Threads
runT = Thread(target=oop.methodInAThread)
oop.win.mainloop())
```

现在我们有一个线程化的方法，但当我们运行代码时，控制台上什么都没有打印出来！

我们必须先启动`Thread`，然后它才能运行，下一节将向我们展示如何做到这一点。

然而，在GUI主事件循环之后设置断点证明我们确实创建了一个`Thread`对象，这可以在Eclipse IDE调试器中看到。

![如何做...](graphics/B04829_06_02.jpg)

## 它是如何工作的...

在这个配方中，我们首先增加了GUI的大小，以便更好地看到打印到`ScrolledText`小部件中的结果，为了准备使用线程。

然后，我们从Python的`threading`模块中导入了`Thread`类。

之后，我们创建了一个在GUI内部从线程中调用的方法。

# 启动线程

这个配方将向我们展示如何启动一个线程。它还将演示为什么线程在长时间运行的任务期间保持我们的GUI响应是必要的。

## 准备工作

让我们首先看看当我们调用一个带有一些休眠的函数或方法时会发生什么，而不使用线程。

### 注意

我们在这里使用休眠来模拟一个现实世界的应用程序，该应用程序可能需要等待Web服务器或数据库响应，或者大文件传输或复杂计算完成其任务。

休眠是一个非常现实的占位符，并展示了涉及的原则。

在我们的按钮回调方法中添加一个循环和一些休眠时间会导致我们的GUI变得无响应，当我们尝试关闭GUI时，情况变得更糟。

```py
# Button callback
def clickMe(self):
  self.action.configure(text='Hello ' + self.name.get())
  # Non-threaded code with sleep freezes the GUI
  for idx in range(10):
    sleep(5)
    self.scr.insert(tk.INSERT, str(idx) + '\n')
```

![准备工作](graphics/B04829_06_03.jpg)

如果我们等待足够长的时间，方法最终会完成，但在此期间，我们的GUI小部件都不会响应点击事件。我们通过使用线程来解决这个问题。

### 注意

在之前的配方中，我们创建了一个要在线程中运行的方法，但到目前为止，线程还没有运行！

与常规的Python函数和方法不同，我们必须`start`一个将在自己的线程中运行的方法！

这是我们接下来要做的事情。

## 如何做...

首先，让我们将线程的创建移到它自己的方法中，然后从按钮回调方法中调用这个方法。

```py
# Running methods in Threads
def createThread(self):
  runT = Thread(target=self.methodInAThread)
  runT.start()
# Button callback
def clickMe(self):
  self.action.configure(text='Hello ' + self.name.get())
  self.createThread()
```

现在点击按钮会导致调用`createThread`方法，然后调用`methodInAThread`方法。

首先，我们创建一个线程并将其目标定位到一个方法。接下来，我们启动线程，该线程将在一个新线程中运行目标方法。

### 注意

GUI本身运行在它自己的线程中，这是应用程序的主线程。

![如何做...](graphics/B04829_06_04.jpg)

我们可以打印出线程的实例。

```py
# Running methods in Threads
def createThread(self):
  runT = Thread(target=self.methodInAThread)
  runT.start()
  print(runT)
```

现在点击按钮会创建以下输出：

![如何做...](graphics/B04829_06_05.jpg)

当我们点击按钮多次时，我们可以看到每个线程都被分配了一个唯一的名称和ID。

![如何做...](graphics/B04829_06_06.jpg)

现在让我们将带有`sleep`的循环代码移到`methodInAThread`方法中，以验证线程确实解决了我们的问题。

```py
def methodInAThread(self):
  print('Hi, how are you?')
  for idx in range(10):
    sleep(5)
    self.scr.insert(tk.INSERT, str(idx) + '\n')
```

当点击按钮时，数字被打印到`ScrolledText`小部件中，间隔五秒，我们可以在GUI的任何地方点击，切换标签等。我们的GUI再次变得响应，因为我们正在使用线程！

![如何做...](graphics/B04829_06_07.jpg)

## 它是如何工作的...

在这个配方中，我们在它们自己的线程中调用了GUI类的方法，并学会了我们必须启动这些线程。否则，线程会被创建，但只是坐在那里等待我们运行它的目标方法。

我们注意到每个线程都被分配了一个唯一的名称和ID。

我们通过在代码中插入`sleep`语句来模拟长时间运行的任务，这向我们表明线程确实可以解决我们的问题。

# 停止线程

我们必须启动一个线程来通过调用`start()`方法实际让它做一些事情，因此，直觉上，我们会期望有一个匹配的`stop()`方法，但实际上并没有这样的方法。在这个配方中，我们将学习如何将线程作为后台任务运行，这被称为守护线程。当关闭主线程，也就是我们的GUI时，所有守护线程也将自动停止。

## 准备工作

当我们在线程中调用方法时，我们也可以向方法传递参数和关键字参数。我们首先通过这种方式开始这个示例。

## 如何做到...

通过在线程构造函数中添加`args=[8]`并修改目标方法以期望参数，我们可以向线程方法传递参数。`args`的参数必须是一个序列，所以我们将我们的数字包装在Python列表中。

```py
def methodInAThread(self, numOfLoops=10):
  for idx in range(numOfLoops):
    sleep(1)
    self.scr.insert(tk.INSERT, str(idx) + '\n')
```

在下面的代码中，`runT`是一个局部变量，我们只能在创建`runT`的方法的范围内访问它。

```py

# Running methods in Threads
def createThread(self):
  runT = Thread(target=self.methodInAThread, args=[8])
  runT.start()
```

通过将局部变量转换为成员变量，我们可以在另一个方法中调用`isAlive`来检查线程是否仍在运行。

```py
# Running methods in Threads
def createThread(self):
  self.runT = Thread(target=self.methodInAThread, args=[8])
  self.runT.start()
  print(self.runT)
  print('createThread():', self.runT.isAlive())
```

在前面的代码中，我们将我们的局部变量`runT`提升为我们类的成员。这样做的效果是使我们能够从我们类的任何方法中评估`self.runT`变量。

这是通过以下方式实现的：

```py
    def methodInAThread(self, numOfLoops=10):
        for idx in range(numOfLoops):
            sleep(1)
            self.scr.insert(tk.INSERT, str(idx) + '\n')
        sleep(1)
        print('methodInAThread():', self.runT.isAlive())
```

当我们单击按钮然后退出GUI时，我们可以看到`createThread`方法中的打印语句被打印出来，但我们看不到`methodInAThread`的第二个打印语句。

相反，我们会得到一个运行时错误。

![如何做...](graphics/B04829_06_08.jpg)

线程预期完成其分配的任务，因此当我们在线程尚未完成时关闭GUI时，Python告诉我们我们启动的线程不在主事件循环中。

我们可以通过将线程转换为守护程序来解决这个问题，然后它将作为后台任务执行。

这给我们的是，一旦我们关闭我们的GUI，也就是我们的主线程启动其他线程，守护线程将干净地退出。

我们可以通过在启动线程之前调用`setDaemon(True)`方法来实现这一点。

```py
# Running methods in Threads
def createThread(self):
  runT = Thread(target=self.methodInAThread)
  runT.setDaemon(True)
  runT.start()
  print(runT)
```

当我们现在单击按钮并在线程尚未完成其分配的任务时退出我们的GUI时，我们不再收到任何错误。

![如何做...](graphics/B04829_06_09.jpg)

## 它是如何工作的...

虽然有一个启动线程运行的方法，但令人惊讶的是，实际上并没有一个等效的停止方法。

在这个示例中，我们正在一个线程中运行一个方法，该方法将数字打印到我们的`ScrolledText`小部件中。

当我们退出GUI时，我们不再对曾经向我们的小部件打印的线程感兴趣，因此，通过将线程转换为后台守护程序，我们可以干净地退出GUI。

# 如何使用队列

Python队列是一种实现先进先出范例的数据结构，基本上就像一个管道一样工作。你把东西塞进管道的一端，它就从管道的另一端掉出来。

这种队列填充和填充泥浆到物理管道的主要区别在于，在Python队列中，事情不会混在一起。你放一个单位进去，那个单位就会从另一边出来。接下来，你放另一个单位进去（比如，例如，一个类的实例），整个单位将作为一个完整的整体从另一端出来。

它以我们插入代码到队列的确切顺序从另一端出来。

### 注意

队列不是一个我们推送和弹出数据的堆栈。堆栈是一个后进先出（LIFO）的数据结构。

队列是容器，用于保存从潜在不同数据源输入队列的数据。我们可以有不同的客户端在有数据可用时向队列提供数据。无论哪个客户端准备好向我们的队列发送数据，我们都可以显示这些数据在小部件中或将其转发到其他模块。

在队列中使用多个线程完成分配的任务在接收处理的最终结果并显示它们时非常有用。数据被插入到队列的一端，然后以有序的方式从另一端出来，先进先出（FIFO）。

我们的GUI可能有五个不同的按钮小部件，每个按钮小部件都会启动我们想要在小部件中显示的不同任务（例如，一个ScrolledText小部件）。

这五个不同的任务完成所需的时间不同。

每当一个任务完成时，我们立即需要知道这一点，并在我们的GUI中显示这些信息。

通过创建一个共享的Python队列，并让五个任务将它们的结果写入这个队列，我们可以使用FIFO方法立即显示已完成的任务的结果。

## 准备工作

随着我们的GUI在功能和实用性上不断增加，它开始与网络、进程和网站进行通信，并最终必须等待数据可用于GUI表示。

在Python中创建队列解决了等待数据在我们的GUI中显示的问题。

## 如何做...

为了在Python中创建队列，我们必须从“queue”模块导入“Queue”类。在我们的GUI模块的顶部添加以下语句：

```py
from threading import Thread
from time import sleep
from queue import Queue
```

这让我们开始了。

接下来，我们创建一个队列实例。

```py
def useQueues(self):
    guiQueue = Queue()     # create queue instance
```

### 注意

在前面的代码中，我们创建了一个本地的“队列”实例，只能在这个方法中访问。如果我们希望从其他地方访问这个队列，我们必须使用“self”关键字将其转换为我们的类的成员，这将本地变量绑定到整个类，使其可以在类中的任何其他方法中使用。在Python中，我们经常在“__init__(self)”方法中创建类实例变量，但Python非常实用，使我们能够在代码中的任何地方创建这些成员变量。

现在我们有了一个队列的实例。我们可以通过打印它来证明它有效。

![如何做...](graphics/B04829_06_10.jpg)

为了将数据放入队列，我们使用“put”命令。为了从队列中取出数据，我们使用“get”命令。

```py
# Create Queue instance  
def useQueues(self):
    guiQueue = Queue()
    print(guiQueue)
    guiQueue.put('Message from a queue')
    print(guiQueue.get())
```

运行修改后的代码会导致消息首先被放入“队列”，然后被从“队列”中取出，并打印到控制台。

![如何做...](graphics/B04829_06_11.jpg)

我们可以将许多消息放入队列。

```py
# Create Queue instance  
def useQueues(self):
    guiQueue = Queue()
    print(guiQueue)
    for idx in range(10):
        guiQueue.put('Message from a queue: ' + str(idx))
    print(guiQueue.get())
```

我们将10条消息放入了“队列”，但我们只取出了第一条。其他消息仍然在“队列”内，等待以FIFO方式取出。

![如何做...](graphics/B04829_06_12.jpg)

为了取出放入“队列”的所有消息，我们可以创建一个无限循环。

```py
# Create Queue instance
def useQueues(self):
    guiQueue = Queue()
    print(guiQueue)
    for idx in range(10):
        guiQueue.put('Message from a queue: ' + str(idx))

    while True: 
        print(guiQueue.get())
```

![如何做...](graphics/B04829_06_13.jpg)

虽然这段代码有效，但不幸的是它冻结了我们的GUI。为了解决这个问题，我们必须在自己的线程中调用该方法，就像我们在之前的示例中所做的那样。

让我们在一个线程中运行我们的方法，并将其绑定到按钮事件：

```py
# Running methods in Threads
def createThread(self, num):
    self.runT = Thread(target=self.methodInAThread, args=[num])
    self.runT.setDaemon(True)
    self.runT.start()
    print(self.runT)
    print('createThread():', self.runT.isAlive())

    # textBoxes are the Consumers of Queue data
    writeT = Thread(target=self.useQueues, daemon=True)
    writeT.start()

# Create Queue instance  
def useQueues(self):
    guiQueue = Queue()
    print(guiQueue)
    for idx in range(10):
        guiQueue.put('Message from a queue: ' + str(idx))
    while True: 
        print(guiQueue.get())
```

当我们现在点击“按钮”时，我们不再会得到一个多余的弹出窗口，代码也能正常工作。

![如何做...](graphics/B04829_06_14.jpg)

## 它是如何工作的...

我们创建了一个“队列”，以FIFO（先进先出）的方式将消息放入队列的一侧。我们从“队列”中取出消息，然后将其打印到控制台（stdout）。

我们意识到我们必须在自己的“线程”中调用该方法。

# 在不同模块之间传递队列

在这个示例中，我们将在不同的模块之间传递“队列”。随着我们的GUI代码变得越来越复杂，我们希望将GUI组件与业务逻辑分离，将它们分离到不同的模块中。

模块化使我们可以重用代码，并使代码更易读。

一旦要在我们的GUI中显示的数据来自不同的数据源，我们将面临延迟问题，这就是“队列”解决的问题。通过在不同的Python模块之间传递“队列”的实例，我们正在分离模块功能的不同关注点。

### 注意

GUI代码理想情况下只关注创建和显示小部件。

业务逻辑模块的工作只是执行业务逻辑。

我们必须将这两个元素结合起来，理想情况下在不同模块之间尽可能少地使用关系，减少代码的相互依赖。

### 注意

避免不必要依赖的编码原则通常被称为“松耦合”。

为了理解松散耦合的重要性，我们可以在白板或纸上画一些框。一个框代表我们的GUI类和代码，而其他框代表业务逻辑、数据库等。

接下来，我们在框之间画线，绘制出这些框之间的相互依赖关系，这些框是我们的Python模块。

### 注意

我们在Python框之间的行数越少，我们的设计就越松散耦合。

## 准备工作

在上一个示例中，我们已经开始使用`Queues`。在这个示例中，我们将从我们的主GUI线程传递`Queue`的实例到其他Python模块，这将使我们能够从另一个模块向`ScrolledText`小部件写入内容，同时保持我们的GUI响应。

## 如何做...

首先，在我们的项目中创建一个新的Python模块。让我们称之为`Queues.py`。我们将在其中放置一个函数（暂时不需要OOP），并将队列的一个实例传递给它。

我们还传递了创建GUI表单和小部件的类的自引用，这使我们能够从另一个Python模块中使用所有GUI方法。

我们在按钮回调中这样做。

### 注意

这就是面向对象编程的魔力。在类的中间，我们将自己传递给类内部调用的函数，使用`self`关键字。

现在代码看起来像这样。

```py
import B04829_Queues as bq

class OOP():
    # Button callback
    def clickMe(self):
      # Passing in the current class instance (self)
        print(self)
        bq.writeToScrol(self)
```

导入的模块包含我们正在调用的函数，

```py
def writeToScrol(inst):
    print('hi from Queue', inst)
    inst.createThread(6)

```

我们已经在按钮回调中注释掉了对`createThread`的调用，因为我们现在是从我们的新模块中调用它。

```py
# Threaded method does not freeze our GUI
# self.createThread()
```

通过从类实例向另一个模块中的函数传递自引用，我们现在可以从其他Python模块访问所有GUI元素。

运行代码会创建以下结果。

![如何做...](graphics/B04829_06_15.jpg)

接下来，我们将创建`Queue`作为我们类的成员，并将对它的引用放在类的`__init__`方法中。

```py
class OOP():
    def __init__(self):
        # Create a Queue
        self.guiQueue = Queue()
```

现在我们可以通过简单地使用传入的类引用将消息放入队列中。

```py
def writeToScrol(inst):
    print('hi from Queue', inst)
    for idx in range(10):
        inst.guiQueue.put('Message from a queue: ' + str(idx))
    inst.createThread(6)
```

我们GUI代码中的`createThread`方法现在只从队列中读取数据，这些数据是由我们新模块中的业务逻辑填充的，这样就将逻辑与我们的GUI模块分离开来了。

```py
def useQueues(self):
    # Now using a class member Queue
    while True:
        print(self.guiQueue.get())
```

运行我们修改后的代码会产生相同的结果。我们没有破坏任何东西（至少目前没有）！

## 它是如何工作的...

为了将GUI小部件与表达业务逻辑的功能分开，我们创建了一个类，将队列作为这个类的成员，并通过将类的实例传递到不同Python模块中的函数中，我们现在可以访问所有GUI小部件以及`Queue`。

这个示例是一个使用面向对象编程的合理情况的例子。

# 使用对话框小部件将文件复制到您的网络

这个示例向我们展示了如何将文件从本地硬盘复制到网络位置。

我们将使用Python的tkinter内置对话框之一，这使我们能够浏览我们的硬盘。然后我们可以选择要复制的文件。

这个示例还向我们展示了如何使`Entry`小部件只读，并将我们的`Entry`默认设置为指定位置，这样可以加快浏览我们的硬盘的速度。

## 准备工作

我们将扩展我们在之前示例中构建的GUI的**Tab 2**。

## 如何做...

将以下代码添加到我们的GUI中`def createWidgets(self)`方法中，放在我们创建Tab Control 2的底部。

新小部件框的父级是`tab2`，我们在`createWidgets()`方法的开头创建了它。只要您将下面显示的代码放在`tab2`的创建物理下方，它就会起作用。

```py
###########################################################
    def createWidgets(self):
        tabControl = ttk.Notebook(self.win)  # Create Tab  
        tab2 = ttk.Frame(tabControl)         # Add a second tab
        tabControl.add(tab2, text='Tab 2')

# Create Manage Files Frame 
mngFilesFrame = ttk.LabelFrame(tab2, text=' Manage Files: ')
mngFilesFrame.grid(column=0, row=1, sticky='WE', padx=10, pady=5)

# Button Callback
def getFileName():
    print('hello from getFileName')

# Add Widgets to Manage Files Frame
lb = ttk.Button(mngFilesFrame, text="Browse to File...", command=getFileName)
lb.grid(column=0, row=0, sticky=tk.W) 

file = tk.StringVar()
self.entryLen = scrolW
self.fileEntry = ttk.Entry(mngFilesFrame, width=self.entryLen, textvariable=file)
self.fileEntry.grid(column=1, row=0, sticky=tk.W)

logDir = tk.StringVar()
self.netwEntry = ttk.Entry(mngFilesFrame, width=self.entryLen, textvariable=logDir)
self.netwEntry.grid(column=1, row=1, sticky=tk.W) 
        def copyFile():
        import shutil   
        src  = self.fileEntry.get()
        file = src.split('/')[-1]  
        dst  = self.netwEntry.get() + '\\'+ file
        try:
            shutil.copy(src, dst)   
            mBox.showinfo('Copy File to Network', 'Success: File copied.')
        except FileNotFoundError as err:
            mBox.showerror('Copy File to Network', '*** Failed to copy file! ***\n\n' + str(err))
        except Exception as ex:
            mBox.showerror('Copy File to Network', '*** Failed to copy file! ***\n\n' + str(ex))

        cb = ttk.Button(mngFilesFrame, text="Copy File To :   ", command=copyFile)
        cb.grid(column=0, row=1, sticky=tk.E)

        # Add some space around each label
        for child in mngFilesFrame.winfo_children(): 
            child.grid_configure(padx=6, pady=6)
```

这将在我们的GUI的**Tab 2**中添加两个按钮和两个输入。

我们还没有实现按钮回调函数的功能。

运行代码会创建以下GUI：

![如何做...](graphics/B04829_06_16.jpg)

点击**浏览文件...**按钮目前会在控制台上打印。

![如何做...](graphics/B04829_06_17.jpg)

我们可以使用tkinter的内置文件对话框，所以让我们在我们的Python GUI模块的顶部添加以下`import`语句。

```py
from tkinter import filedialog as fd
from os import path
```

现在我们可以在我们的代码中使用对话框。我们可以使用Python的os模块来查找GUI模块所在的完整路径，而不是硬编码路径。

```py
def getFileName():
    print('hello from getFileName')
    fDir  = path.dirname(__file__)
    fName = fd.askopenfilename(parent=self.win, initialdir=fDir)
```

单击浏览按钮现在会打开`askopenfilename`对话框。

![如何做...](graphics/B04829_06_18.jpg)

现在我们可以在这个目录中打开一个文件，或者浏览到另一个目录。在对话框中选择一个文件并单击**打开**按钮后，我们将保存文件的完整路径在`fName`本地变量中。

如果我们打开我们的Python `askopenfilename`对话框小部件时，能够自动默认到一个目录，这将是很好的，这样我们就不必一直浏览到我们正在寻找的特定文件要打开的地方。

最好通过回到我们的GUI **Tab 1**来演示如何做到这一点，这就是我们接下来要做的。

我们可以将默认值输入到Entry小部件中。回到我们的**Tab 1**，这非常容易。我们只需要在创建`Entry`小部件时添加以下两行代码即可。

```py
# Adding a Textbox Entry widget
self.name = tk.StringVar()
nameEntered = ttk.Entry(self.monty, width=24, textvariable=self.name)
nameEntered.grid(column=0, row=1, sticky='W')
nameEntered.delete(0, tk.END)
nameEntered.insert(0, '< default name >')
```

当我们现在运行GUI时，`nameEntered`输入框有一个默认值。

![如何做...](graphics/B04829_06_19.jpg)

我们可以使用以下Python语法获取我们正在使用的模块的完整路径，然后我们可以在其下创建一个新的子文件夹。我们可以将其作为模块级全局变量，或者我们可以在方法中创建子文件夹。

```py
# Module level GLOBALS
GLOBAL_CONST = 42
fDir   = path.dirname(__file__)
netDir = fDir + '\\Backup'

def __init__(self):
    self.createWidgets()       
    self.defaultFileEntries()

def defaultFileEntries(self):
    self.fileEntry.delete(0, tk.END)
    self.fileEntry.insert(0, fDir) 
    if len(fDir) > self.entryLen:
        self.fileEntry.config(width=len(fDir) + 3)
        self.fileEntry.config(state='readonly')

    self.netwEntry.delete(0, tk.END)
    self.netwEntry.insert(0, netDir) 
    if len(netDir) > self.entryLen:
        self.netwEntry.config(width=len(netDir) + 3)
```

我们为两个输入小部件设置默认值，并在设置它们后，将本地文件输入小部件设置为只读。

### 注意

这个顺序很重要。我们必须先填充输入框，然后再将其设置为只读。

在调用主事件循环之前，我们还选择**Tab 2**，不再将焦点设置到**Tab 1**的`Entry`中。在我们的tkinter `notebook`上调用`select`是从零开始的，所以通过传入值1，我们选择**Tab 2**...

```py
# Place cursor into name Entry
# nameEntered.focus()             
tabControl.select(1)
```

![如何做...](graphics/B04829_06_20.jpg)

由于我们不都在同一个网络上，这个示例将使用本地硬盘作为网络的示例。

UNC路径是通用命名约定，这意味着我们可以通过双反斜杠访问网络服务器，而不是在访问Windows PC上的本地硬盘时使用典型的`C:\`。

### 注意

你只需要使用UNC，并用`\\<server name> \<folder>\`替换`C:\`。

这个例子可以用来将我们的代码备份到一个备份目录，如果不存在，我们可以使用`os.makedirs`来创建它。

```py
# Module level GLOBALS
GLOBAL_CONST = 42

from os import makedirs
fDir   = path.dirname(__file__)
netDir = fDir + '\\Backup' 
if not path.exists(netDir):
    makedirs(netDir, exist_ok = True)
```

在选择要复制到其他地方的文件后，我们导入Python的`shutil`模块。我们需要文件源的完整路径，一个网络或本地目录路径，然后我们使用`shutil.copy`将文件名附加到我们将要复制的路径上。

### 注意

Shutil是shell utility的简写。

我们还可以通过消息框向用户提供反馈，指示复制是否成功或失败。为了做到这一点，导入`messagebox`并将其重命名为`mBox`。

在下面的代码中，我们将混合两种不同的方法来放置我们的导入语句。在Python中，我们有一些其他语言不提供的灵活性。

我们通常将所有的导入语句放在每个Python模块的顶部，这样可以清楚地看出我们正在导入哪些模块。

同时，现代编码方法是将变量的创建放在首次使用它们的函数或方法附近。

在下面的代码中，我们在Python模块的顶部导入了消息框，然后在一个函数中也导入了shutil Python模块。

为什么我们要这样做呢？

这样做会起作用吗？

答案是，是的，它确实有效，我们将这个导入语句放在一个函数中，因为这是我们的代码中唯一需要这个模块的地方。

如果我们从不调用这个方法，那么我们将永远不会导入这个方法所需的模块。

在某种意义上，您可以将这种技术视为惰性初始化设计模式。

如果我们不需要它，我们就不会在Python代码中导入它，直到我们真正需要它。

这里的想法是，我们的整个代码可能需要，比如说，二十个不同的模块。在运行时，真正需要哪些模块取决于用户的交互。如果我们从未调用`copyFile()`函数，那么就没有必要导入`shutil`。

一旦我们点击调用`copyFile()`函数的按钮，在这个函数中，我们就导入了所需的模块。

```py
from tkinter import messagebox as mBox

def copyFile():
    import shutil   
    src = self.fileEntry.get()
    file = src.split('/')[-1]  
    dst = self.netwEntry.get() + '\\'+ file
    try:
      shutil.copy(src, dst)   
      mBox.showinfo('Copy File to Network', 'Success: File copied.')
    except FileNotFoundError as err:
      mBox.showerror('Copy File to Network', '*** Failed to copy file! ***\n\n' + str(err))
    except Exception as ex:
      mBox.showerror('Copy File to Network', '*** Failed to copy file! ***\n\n' + str(ex))
```

当我们现在运行我们的GUI并浏览到一个文件并点击复制时，文件将被复制到我们在`Entry`小部件中指定的位置。

![如何做...](graphics/B04829_06_21.jpg)

如果文件不存在，或者我们忘记浏览文件并尝试复制整个父文件夹，代码也会让我们知道，因为我们使用了Python的内置异常处理能力。

![如何做...](graphics/B04829_06_22.jpg)

## 它是如何工作的...

我们正在使用Python shell实用程序将文件从本地硬盘复制到网络。由于大多数人都没有连接到相同的局域网，我们通过将代码备份到不同的本地文件夹来模拟复制。

我们正在使用tkinter的对话框控件，并且通过默认目录路径，我们可以提高复制文件的效率。

# 使用TCP/IP通过网络进行通信

这个示例向您展示了如何使用套接字通过TCP/IP进行通信。为了实现这一点，我们需要IP地址和端口号。

为了保持简单并独立于不断变化的互联网IP地址，我们将创建自己的本地TCP/IP服务器，并作为客户端，学习如何连接到它并从TCP/IP连接中读取数据。

我们将通过使用我们在以前的示例中创建的队列，将这种网络功能集成到我们的GUI中。

## 准备工作

我们将创建一个新的Python模块，它将是TCP服务器。

## 如何做...

在Python中实现TCP服务器的一种方法是从`socketserver`模块继承。我们子类化`BaseRequestHandler`，然后覆盖继承的`handle`方法。在很少的Python代码行中，我们可以实现一个TCP服务器模块。

```py
from socketserver import BaseRequestHandler, TCPServer

class RequestHandler(BaseRequestHandler):
    # override base class handle method
    def handle(self):
        print('Server connected to: ', self.client_address)
        while True:
            rsp = self.request.recv(512)
            if not rsp: break
            self.request.send(b'Server received: ' + rsp)

def startServer():
    serv = TCPServer(('', 24000), RequestHandler)
    serv.serve_forever()
```

我们将我们的`RequestHandler`类传递给`TCPServer`初始化程序。空的单引号是传递本地主机的快捷方式，这是我们自己的PC。这是IP地址127.0.0.1的IP地址。元组中的第二项是端口号。我们可以选择任何在本地PC上未使用的端口号。

我们只需要确保在TCP连接的客户端端口上使用相同的端口，否则我们将无法连接到服务器。当然，在客户端可以连接到服务器之前，我们必须先启动服务器。

我们将修改我们的`Queues.py`模块，使其成为TCP客户端。

```py
from socket import socket, AF_INET, SOCK_STREAM

def writeToScrol(inst):
    print('hi from Queue', inst)
    sock = socket(AF_INET, SOCK_STREAM)
    sock.connect(('localhost', 24000))
    for idx in range(10):
        sock.send(b'Message from a queue: ' + bytes(str(idx).encode()) )
        recv = sock.recv(8192).decode()
        inst.guiQueue.put(recv)      
    inst.createThread(6)
```

这是我们与TCP服务器通信所需的所有代码。在这个例子中，我们只是向服务器发送一些字节，服务器将它们发送回来，并在返回响应之前添加一些字符串。

### 注意

这显示了TCP通过网络进行通信的原理。

一旦我们知道如何通过TCP/IP连接到远程服务器，我们将使用由我们感兴趣的通信程序的协议设计的任何命令。第一步是在我们可以向驻留在服务器上的特定应用程序发送命令之前进行连接。

在`writeToScrol`函数中，我们将使用与以前相同的循环，但现在我们将把消息发送到TCP服务器。服务器修改接收到的消息，然后将其发送回给我们。接下来，我们将其放入GUI成员队列中，就像以前的示例一样，在其自己的`Thread`中运行。

### 注意

在Python 3中，我们必须以二进制格式通过套接字发送字符串。现在添加整数索引变得有点复杂，因为我们必须将其转换为字符串，对其进行编码，然后将编码后的字符串转换为字节！

```py
sock.send(b'Message from a queue: ' + bytes(str(idx).encode()) )
```

注意字符串前面的`b`，然后，嗯，所有其他所需的转换...

我们在OOP类的初始化程序中启动TCP服务器的线程。

```py
class OOP():
    def __init__(self):
    # Start TCP/IP server in its own thread
        svrT = Thread(target=startServer, daemon=True)
        svrT.start()
```

现在，在**Tab 1**上单击**Click Me!**按钮将在我们的`ScrolledText`小部件中创建以下输出，以及在控制台上，由于使用`Threads`，响应非常快。

![操作步骤...](graphics/B04829_06_23.jpg)

## 它是如何工作的...

我们创建了一个TCP服务器来模拟连接到本地区域网络或互联网上的服务器。我们将我们的队列模块转换为TCP客户端。我们在它们自己的后台线程中运行队列和服务器，这样我们的GUI非常响应。

# 使用URLOpen从网站读取数据

这个示例展示了我们如何使用Python的内置模块轻松读取整个网页。我们将首先以原始格式显示网页数据，然后解码它，然后在我们的GUI中显示它。

## 准备工作

我们将从网页中读取数据，然后在我们的GUI的`ScrolledText`小部件中显示它。

## 如何做...

首先，我们创建一个新的Python模块并命名为`URL.py`。

然后，我们导入所需的功能来使用Python读取网页。

我们可以用很少的代码来做到这一点。

我们将我们的代码包装在一个类似于Java和C#的`try…except`块中。这是Python支持的一种现代编码方法。

每当我们有可能不完整的代码时，我们可以尝试这段代码，如果成功，一切都很好。

如果`try…except`块中的代码块不起作用，Python解释器将抛出几种可能的异常，然后我们可以捕获。一旦我们捕获了异常，我们就可以决定接下来要做什么。

Python中有一系列的异常，我们还可以创建自己的类，继承并扩展Python异常类。

在下面显示的代码中，我们主要关注我们尝试打开的URL可能不可用，因此我们将我们的代码包装在`try…except`代码块中。

如果代码成功打开所请求的URL，一切都很好。

如果失败，可能是因为我们的互联网连接断开了，我们就会进入代码的异常部分，并打印出发生异常的信息。

### 注意

您可以在[https://docs.python.org/3.4/library/exceptions.html](https://docs.python.org/3.4/library/exceptions.html)了解更多关于Python异常处理的信息。

```py
from urllib.request import urlopen
link = 'http://python.org/' 
try:
    f = urlopen(link)
    print(f)
    html = f.read()
    print(html)
    htmldecoded = html.decode()
    print(htmldecoded)

except Exception as ex:
    print('*** Failed to get Html! ***\n\n' + str(ex))
```

通过在官方Python网站上调用`urlopen`，我们得到整个数据作为一个长字符串。

第一个打印语句将这个长字符串打印到控制台上。

然后我们对结果调用`decode`，这次我们得到了一千多行的网页数据，包括一些空白。

我们还打印调用`urlopen`的类型，它是一个`http.client.HTTPResponse`对象。实际上，我们首先打印出来。

![操作步骤...](graphics/B04829_06_24.jpg)

这是我们刚刚读取的官方Python网页。如果您是Web开发人员，您可能对如何处理解析数据有一些好主意。

![操作步骤...](graphics/B04829_06_25.jpg)

接下来，我们在我们的GUI中的`ScrolledText`小部件中显示这些数据。为了这样做，我们必须将我们的新模块连接到我们的GUI，从网页中读取数据。

为了做到这一点，我们需要一个对我们GUI的引用，而一种方法是通过将我们的新模块绑定到**Tab 1**按钮回调。

我们可以将从Python网页解码的HTML数据返回给`Button`小部件，然后将其放在`ScrolledText`控件中。

因此，让我们将我们的代码转换为一个函数，并将数据返回给调用代码。

```py
from urllib.request import urlopen
link = 'http://python.org/'
def getHtml():
    try:
        f = urlopen(link)
        #print(f)
        html = f.read()
        #print(html)
        htmldecoded = html.decode()
        #print(htmldecoded)     
    except Exception as ex:
        print('*** Failed to get Html! ***\n\n' + str(ex))
    else:
        return htmldecoded  
```

现在，我们可以通过首先导入新模块，然后将数据插入到小部件中，在我们的`button`回调方法中写入数据到`ScrolledText`控件。在调用`writeToScrol`之后，我们还给它一些休眠时间。

```py
import B04829_Ch06_URL as url

# Button callback
def clickMe(self):
  bq.writeToScrol(self)       
  sleep(2)
  htmlData = url.getHtml()
  print(htmlData)
  self.scr.insert(tk.INSERT, htmlData)
```

HTML数据现在显示在我们的GUI小部件中。

![操作步骤...](graphics/B04829_06_26.jpg)

## 它是如何工作的...

我们创建了一个新模块，将从网页获取数据的代码与我们的GUI代码分离。这总是一个好主意。我们读取网页数据，然后解码后返回给调用代码。然后我们使用按钮回调函数将返回的数据放入“ScrolledText”控件中。

本章向我们介绍了一些高级的Python编程概念，我们将它们结合起来，制作出一个功能性的GUI程序。
