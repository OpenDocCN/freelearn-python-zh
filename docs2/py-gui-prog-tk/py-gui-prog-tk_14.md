

# 第十四章：使用线程和队列进行异步编程

许多时候，在测试环境的简单性中完美运行的代码在现实世界中会遇到问题；不幸的是，ABQ 数据输入应用程序似乎就是这样。虽然你的网络功能在你的本地主机测试环境中运行得非常快，但实验室缓慢的 VPN 上行链路暴露了你编程中的一些不足。用户报告说，当进行网络事务时，应用程序会冻结或变得无响应。尽管它确实可以工作，但看起来不够专业，使用户感到烦恼。

为了解决这个问题，我们需要应用异步编程技术，我们将在以下主题中学习这些技术：

+   在 *Tkinter 事件队列* 中，我们将学习如何操作 Tkinter 的事件处理来提高应用程序的响应性。

+   在 *使用线程在后台运行代码* 中，我们将探讨使用 Python 的 `threading` 模块编写多线程应用程序。

+   在 *使用队列传递消息* 中，你将学习如何使用 `Queue` 对象来实现线程间通信。

+   在 *使用锁来保护共享资源* 中，我们将利用 `Lock` 对象来防止线程相互覆盖。

让我们开始吧！

# Tkinter 的事件队列

正如我们在 *第十一章* 中讨论的，*使用 unittest 创建自动化测试*，Tkinter 中的许多任务，如绘图和更新小部件，都是异步执行的，而不是在代码中调用时立即采取行动。更具体地说，你在 Tkinter 中执行的操作，如点击按钮、触发键绑定或跟踪，或调整窗口大小，都会在事件队列中放置一个 **事件**。在主循环的每次迭代中，Tkinter 从队列中提取所有挂起的事件，并逐个处理它们。对于每个事件，Tkinter 在继续处理队列中的下一个事件之前，执行与事件绑定的任何 **任务**（即回调或重绘小部件等内部操作）。

Tkinter 大致将任务优先级分为 **常规** 或 **空闲时执行**（通常称为 **空闲任务**）。在事件处理过程中，常规任务首先处理，当所有常规任务完成后，再处理空闲任务。大多数绘图或小部件更新任务被归类为空闲任务，而像回调函数这样的操作默认为常规优先级。

## 事件队列控制

大多数时候，我们通过依赖高级构造，如 `command` 回调和 `bind()`，从 Tkinter 获得所需的行为。然而，在某些情况下，我们可能希望直接与事件队列交互并手动控制事件的处理方式。我们已经看到了一些可用于此目的的功能，但让我们在这里更深入地了解一下。

### update() 方法

在 *第十一章*，*使用 unittest 创建自动化测试* 中，你学习了 `update()` 和 `update_idletasks()` 方法。为了复习，这些方法将导致 Tkinter 执行队列中当前所有事件的任何任务；`update()` 运行队列中所有当前等待的事件，直到完全清除，而 `update_idletasks()` 只运行空闲任务。

由于空闲任务通常较小且更安全，除非你发现它不起作用，否则建议使用 `update_idletasks()`。

注意，`update()` 和 `update_idletasks()` 将导致处理所有小部件的所有挂起事件，无论方法是在哪个小部件上调用。没有方法可以只处理特定小部件或 Tkinter 对象的事件。

### `after()` 方法

除了允许我们控制队列的处理外，Tkinter 小部件还有两种方法可以在延迟后向事件队列添加任意代码：`after()` 和 `after_idle()`。

`after()` 的基本用法如下：

```py
# basic_after_demo.py
import tkinter as tk
root = tk.Tk()
root.after(1000, root.quit)
root.mainloop() 
```

在这个例子中，我们将 `root.quit()` 方法设置为在 1 秒（1,000 毫秒）后运行。实际上发生的情况是，一个绑定到 `root.quit` 的事件被添加到事件队列中，但有一个条件，即它不应在从调用 `after()` 的那一刻起至少 1,000 毫秒内执行。在这段时间内，队列中的任何其他事件都将首先被处理。因此，虽然命令不会在 1,000 毫秒内执行，但它很可能会在之后执行，具体取决于事件队列中已经正在处理的其他内容。

`after_idle()` 方法也会将一个任务添加到事件队列中，但它不是提供一个明确的延迟，而是简单地将其添加为一个空闲任务，确保它将在任何常规任务之后运行。

在这两种方法中，任何附加到回调引用的额外参数都简单地作为位置参数传递给回调；例如：

```py
root.after(1000, print, 'hello', 'Python', 'programmers!') 
```

在这个例子中，我们将 `'hello'`、`'Python'` 和 `'programmers'` 参数传递给一个 `print()` 调用。这个语句将计划在 1 秒后尽可能快地运行 `print('hello', 'Python', 'programmers!')` 语句。

注意，`after()` 和 `after_idle()` 不能为传递的可调用对象接受关键字参数，只能接受位置参数。

使用 `after()` 方法计划的任务也可以使用 `after_cancel()` 方法取消计划。该方法接受一个任务 ID 号，该 ID 号是在我们调用 `after()` 时返回的。

例如，我们可以修改之前的例子如下：

```py
# basic_after_cancel_demo.py
import tkinter as tk
root = tk.Tk()
task_id = root.after(3000, root.quit)
tk.Button(
  root,
  text='Do not quit!', command=lambda: root.after_cancel(task_id)
).pack()
root.mainloop() 
```

在这个脚本中，我们保存了 `after()` 的返回值，这给我们提供了计划任务的 ID。然后，在我们的按钮回调中，我们调用 `after_cancel()`，传入 ID 值。在 3 秒内点击按钮会导致 `root.quit` 任务被取消，应用程序保持打开状态。

## 事件队列控制的一般用途

在**第十一章**，**使用 unittest 创建自动化测试**中，我们很好地使用了队列控制方法，以确保我们的测试运行得既快又高效，而无需等待人工交互。尽管如此，我们可以在实际应用程序中使用这些方法的不同方式，我们将在下面探讨。

### 平滑显示更改

在具有动态 GUI 更改的应用程序中，当窗口根据元素的出现和重新出现进行缩放时，这些更改的平滑性可能会略有下降。例如，在 ABQ 应用程序中，你可能会注意到登录后立即出现一个较小的应用程序窗口，随着 GUI 的构建，它很快就会被调整大小。这不是一个主要问题，但它会从整体上影响应用程序的展示。

我们可以通过在登录后使用`after()`延迟`deiconify()`调用来纠正这个问题。在`Application.__init__()`内部，让我们将那行代码修改如下：

```py
# application.py, inside Application.__init__()
    self.after(250, self.deiconify) 
```

现在，我们不再在登录后立即恢复应用程序窗口，而是将其延迟了四分之一秒。虽然对用户来说几乎察觉不到，但它给了 Tkinter 足够的时间在显示窗口之前构建和重绘 GUI，从而平滑了操作。

应该谨慎使用延迟代码，并且不要在延迟代码的稳定性或安全性依赖于其他进程先完成的情况下依赖它。这可能导致**竞态条件**，在这种情况下，一些不可预见的情况，如缓慢的磁盘或网络连接，可能导致你的延迟不足以正确地安排代码的执行顺序。在我们的应用程序中，我们的延迟仅仅是一个表面上的修复；如果在应用程序窗口完成绘制之前恢复窗口，不会发生灾难性的事故。

### 缓解 GUI 冻结

由于回调任务优先于屏幕更新任务，一个长时间阻塞代码执行的回调任务可能导致程序看起来冻结或卡在尴尬的位置，而重绘任务则等待其完成。解决这个问题的方法之一是使用`after()`和`update()`方法手动控制事件队列处理。为了了解这是如何工作的，我们将构建一个简单的应用程序，使用这些方法在长时间运行的任务期间保持 UI 响应。

从这个简单但缓慢的应用程序开始：

```py
# after_demo.py
import tkinter as tk
from time import sleep
class App(tk.Tk):
  def __init__(self):
    super().__init__()
    self.status = tk.StringVar()
    tk.Label(self, textvariable=self.status).pack()
    tk.Button(
      self, text="Run Process",
      command=self.run_process
    ).pack()
  def run_process(self):
    self.status.set("Starting process")
    sleep(2)
    for phase in range(1, 5):
      self.status.set(f"Phase {phase}")
      self.process_phase(phase, 2)
    self.status.set('Complete')
  def process_phase(self, n, length):
    # some kind of heavy processing here
    sleep(length)
App().mainloop() 
```

此应用程序使用`time.sleep()`来模拟多个阶段完成的一些重处理任务。GUI 向用户提供了一个按钮，用于启动进程，以及一个状态指示器来显示进度。

当用户点击按钮时，状态指示器应该**应该**执行以下操作：

+   显示**启动进程**2 秒钟。

+   显示**阶段 1**、**阶段 2**，通过**阶段 4**，每个阶段持续 2 秒钟。

+   最后，它应该显示为**完成**。

尽管如此，如果你尝试它，你会发现它并没有这样做。相反，当按钮按下时，它会立即冻结，并且不会解冻，直到所有阶段都完成并且状态显示为**完成**。为什么会发生这种情况？

当按钮点击事件由主循环处理时，`run_process()` 回调比任何绘图任务（因为那些是空闲任务）具有优先级，并且立即执行，阻塞主循环直到它返回。当回调调用 `self.status.set()` 时，`status` 变量的写事件被放入队列（它们最终会在 `Label` 小部件上触发重绘事件）。然而，当前队列的处理已被暂停，等待 `run_process()` 方法返回。当它最终返回时，所有等待在事件队列中的 `status` 更新都在一秒钟内执行。

为了使这个方法更好一些，让我们使用 `after()` 来安排 `run_process()`：

```py
# after_demo2.py
  def run_process(self):
    self.status.set("Starting process")
    self.after(50, self._run_phases)
  def _run_phases(self):
    for phase in range(1, 5):
      self.status.set(f"Phase {phase}")
      self.process_phase(phase, 2)
    self.status.set('Complete') 
```

这次，`run_process()` 的循环部分被拆分到一个单独的方法 `_run_phases()` 中。`run_process()` 方法本身只是设置起始状态，然后安排 `_run_phases()` 在 50 毫秒后运行。这个延迟给 Tkinter 时间来完成任何绘图任务并在启动长时间阻塞循环之前更新状态。在这种情况下，确切的时间并不重要，只要足够 Tkinter 完成绘图操作，但又不至于让用户注意到；50 毫秒似乎可以很好地完成这项工作。

尽管如此，我们仍然没有看到这个版本的各个阶段状态消息；它直接从 **开始进程** 跳到 **完成**，因为 `_run_phases()` 方法最终运行时仍然会阻塞事件循环。

为了解决这个问题，我们可以在循环中使用 `update_idletasks()`：

```py
# after_demo_update.py
  def _run_phases(self):
    for phase in range(1, 5):
      self.status.set(f"Phase {phase}")
      self.update_idletasks()
      self.process_phase(phase, 2)
    self.status.set('Complete') 
```

通过强制 Tkinter 在开始长时间阻塞方法之前运行队列中的剩余空闲任务，我们的 GUI 保持更新。不幸的是，这种方法存在一些缺点：

+   首先，单个任务在运行时仍然会阻塞应用程序。无论我们如何将其拆分，当处理过程的各个单元执行时，应用程序仍然会被冻结。

+   其次，这种方法在关注点分离方面存在问题。在实际应用中，我们的处理阶段很可能会在某种后端或模型类中运行。这些类不应该操作 GUI 小部件。

虽然这些队列控制方法可以用于管理 GUI 层面的进程，但很明显，我们需要一个更好的解决方案来处理像 ABQ 网络上传功能这样的慢速后台进程。对于这些，我们需要使用更强大的工具：线程。

# 使用线程在后台运行代码

书中到目前为止所写的所有代码都可以描述为**单线程**；也就是说，每个语句一次执行一个，前一个语句完成之前，下一个语句才开始执行。即使像我们的 Tkinter 事件队列这样的异步元素可能会改变任务执行的顺序，但它们仍然一次只执行一个任务。这意味着像慢速网络事务或文件读取这样的长时间运行的过程不可避免地会在运行时冻结我们的应用程序。

要看到这个效果，请运行示例代码中包含的`sample_rest_service.py`脚本，该脚本对应于第十四章（确保运行的是第十四章版本，而不是第十三章版本！）现在运行 ABQ 数据输入，确保你今天数据库中有一些数据，并运行 REST 上传。上传应该需要大约 20 秒，在这段时间内，服务脚本应该会打印出类似以下的状态消息：

```py
File 0% uploaded
File 5% uploaded
File 10% uploaded
File 15% uploaded
File 20% uploaded
File 25% uploaded 
```

同时，我们的 GUI 应用程序会冻结。你会发现你无法与任何控件交互，移动或调整大小可能会导致一个空白的灰色窗口。只有当上传过程完成时，你的应用程序才会再次变得响应。

为了真正解决这个问题，我们需要创建一个**多线程**应用程序，其中多个代码片段可以同时运行，而无需相互等待。在 Python 中，我们可以使用`threading`模块来实现这一点。

## 线程模块

多线程应用程序编程可能相当具有挑战性，但标准库的`threading`模块使得使用线程变得尽可能简单。

为了演示`threading`的基本用法，让我们首先创建一个故意慢速的函数：

```py
# basic_threading_demo.py
from time import sleep
def print_slowly(string):
  words = string.split()
  for word in words:
    sleep(1)
    print(word) 
```

这个函数接受一个字符串，并以每秒一个单词的速度打印它。这将模拟一个长时间运行、计算密集型的过程，并给我们一些反馈，表明它仍在运行。

让我们为这个函数创建一个 Tkinter GUI 前端：

```py
# basic_threading_demo.py
import tkinter as tk
# print_slowly() function goes here
# ...
class App(tk.Tk):
  def __init__(self):
    super().__init__()
    self.text = tk.StringVar()
    tk.Entry(self, textvariable=self.text).pack()
    tk.Button(
      self, text="Run unthreaded",
      command=self.print_unthreaded
    ).pack()
  def print_unthreaded(self):
    print_slowly(self.text.get())
App().mainloop() 
```

这个简单的应用程序有一个文本输入框和一个按钮；当按钮被按下时，输入框中的文本会被发送到`print_slowly()`函数。运行此代码，然后在`Entry`小部件中输入或粘贴一个长句子。

当你点击按钮时，你会看到整个应用程序冻结，因为单词被打印到控制台。这是因为所有操作都在单个执行线程中运行。

现在让我们添加线程代码：

```py
# basic_threading_demo.py
from threading import Thread
# at the end of App.__init__()
    tk.Button(
      self, text="Run threaded",
      command=self.print_threaded
    ).pack()
  def print_threaded(self):
    thread = Thread(
      target=print_slowly,
      args=(self.text.get(),)
    )
    thread.start() 
```

这次，我们导入了`Thread`类并创建了一个名为`print_threaded()`的新回调。这个回调使用一个`Thread`对象在其自己的执行线程中运行`print_slowly()`。

`Thread`对象接受一个`target`参数，该参数指向将在新执行线程中运行的调用函数。它还可以接受一个`args`元组，该元组包含要传递给`target`参数的参数，以及一个`kwargs`字典，它也会在`target`函数的参数列表中展开。

要执行 `Thread` 对象，我们调用它的 `start()` 方法。此方法不会阻塞，因此 `print_threaded()` 回调立即返回，允许 Tkinter 继续其事件循环，同时 `thread` 在后台执行。

如果你尝试运行此代码，你会看到在打印句子时 GUI 不会冻结。无论句子有多长，GUI 整个过程中都保持响应。

### Tkinter 和线程安全

线程引入了大量的复杂性到代码库中，并不是所有的代码都是为在多线程环境中正确行为而编写的。

我们将考虑到线程的代码称为 **线程安全**。

经常有人说 Tkinter 不是线程安全的；这并不完全正确。假设你的 Tcl/Tk 二进制文件已经编译了线程支持（Linux、Windows 和 macOS 的官方 Python 发行版中包含的），Tkinter 应该在多线程程序中运行良好。然而，Python 文档警告我们，在跨线程调用中，Tkinter 仍然存在一些边缘情况，其行为可能不正确。

避免这些问题的最佳方式是将我们的 Tkinter 代码保持在单个线程中，并将线程的使用限制在非 Tkinter 代码（如我们的模型类）中。

关于 Tkinter 和线程的更多信息可以在 [`docs.python.org/3/library/tkinter.html#threading-model`](https://docs.python.org/3/library/tkinter.html#threading-model) 找到。

## 将我们的网络函数转换为线程执行

将一个函数传递给 `Thread` 对象的 `target` 参数是在线程中运行代码的一种方法；一个更灵活且强大的方法是继承 `Thread` 类，并用你想要执行的代码覆盖其 `run()` 方法。为了演示这种方法，让我们更新在 *第十三章*，*连接到云端* 中为 ABQ 数据录入创建的企业 REST 上传功能，使其在单独的线程中运行缓慢的上传操作。

首先，打开 `models.py` 并导入 `Thread` 类，如下所示：

```py
# models.py, at the top
from threading import Thread 
```

我们不是让一个 `CorporateRestModel` 方法来执行上传，而是将创建一个基于 `Thread` 的类，其实例将能够在单独的线程中执行上传操作。我们将它命名为 `ThreadedUploader`。

要执行其上传，`ThreadedUploader` 实例需要一个端点 URL 和本地文件路径；我们可以简单地将这些传递给对象在其初始化器中。它还需要访问认证会话；这会带来更多的问题。我们可能能够通过将我们的认证 `Session` 对象传递给线程来解决这个问题，但在编写本文时，关于 `Session` 对象是否线程安全存在很大的不确定性，因此最好避免在线程之间共享它们。

然而，我们实际上并不需要整个 `Session` 对象，只需要认证令牌或会话 cookie。

结果表明，当我们对 REST 服务器进行身份验证时，一个名为`session`的 cookie 被放置在我们的 cookie jar 中，我们可以通过检查终端中的`Session.cookies`对象来查看，如下所示：

```py
# execute this with the sample REST server running in another terminal
>>> import requests
>>> s = requests.Session()
>>> s.post('http://localhost:8000/auth', data={'username': 'test', 'password': 'test'})
<Response [200]>
>>> dict(s.cookies)
{'session': 'eyJhdXRoZW50aWNhdGVkIjp0cnVlfQ.YTu7xA.c5ZOSuHQbckhasRFRF'} 
```

`cookies`属性是一个`requests.CookieJar`对象，它在许多方面都像字典一样工作。每个 cookie 都有一个唯一的名称，可以用来检索 cookie 本身。在这种情况下，我们的会话 cookie 被称为`session`。

由于 cookie 本身只是一个字符串，我们可以安全地将其传递给另一个线程。一旦到达那里，我们将创建一个新的`Session`对象，并将 cookie 传递给它，之后它就可以验证请求了。

不可变对象，包括字符串、整数和浮点数，总是线程安全的。由于不可变对象在创建后不能被更改，我们不必担心两个线程会同时尝试更改对象。

让我们以以下方式开始我们的新上传器类：

```py
# models.py
class ThreadedUploader(Thread):
  def __init__(self, session_cookie, files_url, filepath):
    super().__init__()
    self.files_url = files_url
    self.filepath = filepath
    # Create the new session and hand it the cookie
    self.session = requests.Session()
    self.session.cookies['session'] = session_cookie 
```

初始化方法首先调用超类初始化器以设置`Thread`对象，然后将传递的`files_url`和`filepath`字符串分配给实例属性。

接下来，我们创建一个新的`Session`对象，并将传递的 cookie 值添加到 cookie jar 中，通过将其分配给`session`键（与原始会话的 cookie jar 中使用的相同键）。现在我们有了执行上传过程所需的所有信息。在线程中要执行的实际过程在其`run()`方法中实现，我们将在下面添加它：

```py
 def run(self, *args, **kwargs):
    with open(self.filepath, 'rb') as fh:
      files = {'file': fh}
      response = self.session.put(
        self.files_url, files=files
      )
      response.raise_for_status() 
```

注意，这段代码基本上是模型`upload()`方法的代码，只是函数参数已经被更改为了实例属性。

现在，让我们转到我们的模型，看看我们如何使用这个类。

Python 文档建议，在子类化`Thread`时，只重写`run()`和`__init__()`方法。其他方法应保持不变以确保正确操作。

### 使用线程化上传器

现在我们已经创建了一个线程化的上传器，我们只需要让`CorporateRestModel`使用它。找到你的模型类，然后按照以下方式重写`upload_file()`方法：

```py
# models.py, inside CorporateRestModel
  def upload_file(self, filepath):
    """PUT a file on the server"""
    cookie = self.session.cookies.get('session')
    uploader = ThreadedUploader(
      cookie, self.files_url, filepath
    )
    uploader.start() 
```

在这里，我们首先从我们的`Session`对象中提取会话 cookie，然后将其与 URL 和文件路径一起传递给`ThreadedUploader`初始化器。最后，我们调用线程的`start()`方法以开始上传的执行。

现在，再次尝试你的 REST 上传，你会发现应用程序不会冻结。干得好！然而，它还没有完全按照我们希望的方式表现...

记住，你重写了`run()`方法，但调用的是`start()`方法。混淆这些会导致你的代码要么什么都不做，要么像正常单线程调用一样阻塞。

### 使用队列传递消息

我们已经解决了程序冻结的问题，但现在我们遇到了一些新的问题。最明显的问题是我们的回调立即显示一个消息框，声称我们已经成功上传了文件，尽管你可以从服务器输出中看到该过程仍在后台进行。一个更微妙但更严重的问题是，我们没有收到错误通知。如果你在上传过程中尝试终止测试服务（因此回调应该失败），它仍然会立即声称上传成功，尽管你可以在终端上看到正在抛出异常。这里发生了什么？

这里的问题首先是 `Thread.start()` 方法不会阻塞代码执行。当然，这是我们想要的，但现在这意味着我们的成功对话框不会等待上传过程完成才显示。一旦启动新线程，主线程中的代码就会与新线程并行执行，立即显示成功对话框。

第二个问题是，在其自己的线程中运行的代码无法将线程的 `run()` 方法中引起的异常传递回主线程。这些异常是在新线程中抛出的，并且只能在新线程中被捕获。就我们的主线程而言，`try` 块中的代码执行得很好。事实上，上传操作无法通信失败或成功。

为了解决这些问题，我们需要一种方式让 GUI 和模型线程进行通信，以便上传线程可以将错误或进度消息发送回主线程以适当处理。我们可以使用**队列**来实现这一点。

### 队列对象

Python 的 `queue.Queue` 类提供了一个**先进先出**（**FIFO**）的数据结构。Python 对象可以使用 `put()` 方法放入 `Queue` 对象中，并使用 `get()` 方法检索；要查看这是如何工作的，请在 Python shell 中执行以下操作：

```py
>>> from queue import Queue
>>> q = Queue()
>>> q.put('My item')
>>> q.get()
'My item' 
```

这可能看起来并不特别令人兴奋；毕竟，你可以用 `list` 对象做同样的事情。然而，使 `Queue` 有用之处在于它是线程安全的。一个线程可以将消息放置在队列上，另一个线程可以检索它们并相应地做出反应。

默认情况下，队列的 `get()` 方法将阻塞执行，直到接收到一个项目。这种行为可以通过将 `False` 作为其第一个参数传递或使用 `get_nowait()` 方法来改变。在不等待模式下，该方法将立即返回，如果队列为空，则抛出异常。

要查看这是如何工作的，请在 shell 中执行以下操作：

```py
>>> q = Queue()
>>> q.get_nowait()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/lib/python3.9/queue.py", line 199, in get_nowait
    return self.get(block=False)
  File "/usr/lib/python3.9/queue.py", line 168, in get
    raise Empty
_queue.Empty 
```

我们还可以使用 `empty()` 或 `qsize()` 方法检查队列是否为空；例如：

```py
>>> q.empty()
True
>>> q.qsize()
0
>>> q.put(1)
>>> q.empty()
False
>>> q.qsize()
1 
```

如你所见，`empty()` 返回一个布尔值，表示队列是否为空，而 `qsize()` 返回队列中的项目数量。`Queue` 类还有其他一些在更高级的多线程情况下有用的方法，但 `get()`、`put()` 和 `empty()` 将足以解决我们的问题。

### 使用队列在线程之间进行通信

在编辑我们的应用程序代码之前，让我们创建一个简单的示例应用程序，以确保我们理解如何使用 `Queue` 在线程之间进行通信。

从一个长时间运行的线程开始：

```py
# threading_queue_demo.py
from threading import Thread
from time import sleep
class Backend(Thread):
  def __init__(self, queue, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.queue = queue
  def run(self):
    self.queue.put('ready')
    for n in range(1, 5):
      self.queue.put(f'stage {n}')
      print(f'stage {n}')
      sleep(2)
    self.queue.put('done') 
```

`Backend` 对象是 `Thread` 的一个子类，它接受一个 `Queue` 对象作为参数，并将其保存为实例属性。它的 `run()` 方法使用 `print()` 和 `sleep()` 模拟一个长时间运行的四阶段过程。在开始、结束和每个阶段之前，我们使用 `queue.put()` 将状态消息放入队列模块。

现在，我们将使用 Tkinter 为此过程创建一个前端：

```py
# threading_queue_demo.py
import tkinter as tk
from queue import Queue
class App(tk.Tk):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.status = tk.StringVar(self, value='ready')
    tk.Label(self, textvariable=self.status).pack()
    tk.Button(self, text="Run process", command=self.go).pack()
    self.queue = Queue() 
```

这个简单的应用程序包含一个绑定到 `status` 控制变量的 `Label` 对象，一个绑定到回调方法 `go()` 的 `Button` 小部件，以及存储为实例变量的 `Queue` 对象。想法是，当我们点击 **运行进程** 按钮时，`go()` 方法将运行我们的 `Backend` 类，并通过 `status` 控制变量将队列中的消息显示在标签上。

让我们创建 `go()` 方法：

```py
 def go(self):
    p = Backend(self.queue)
    p.start() 
```

`go()` 方法创建 `Backend` 类的一个实例，传入应用程序的 `Queue` 对象，并启动它。因为现在两个线程都有一个对 `queue` 的引用，我们可以用它来在它们之间进行通信。我们已经看到 `Backend` 如何将状态消息放置在队列上，那么 `App()` 应该如何检索它们呢？

也许我们可以启动一个循环，如下所示：

```py
 def go(self):
    p = Backend(self.queue)
    p.start()
    while True:
      status = self.queue.get()
      self.status.set(status)
      if status == 'done':
        break 
```

当然，这不会起作用，因为循环会阻塞；Tkinter 事件循环会卡在执行 `go()` 上，冻结 GUI，并违背使用第二个线程的目的。相反，我们需要一种方法来定期轮询 `queue` 对象以获取状态消息，并在收到消息时更新状态。

我们将首先编写一个可以检查队列并相应响应的方法：

```py
 def check_queue(self):
    msg = ''
    while not self.queue.empty():
      msg = self.queue.get()
      self.status.set(msg) 
```

使用 `Queue.empty()` 方法，我们首先找出队列是否为空。如果是，我们不想做任何事情，因为默认情况下 `get()` 会阻塞，直到它收到消息，我们不希望阻塞执行。如果 `queue` 对象包含项目，我们将想要获取这些项目并将它们发送到我们的 `status` 变量。我们正在使用 `while` 循环这样做，这样我们只有在队列为空时才离开函数。

当然，这只会执行一次检查；我们希望继续轮询队列模块，直到线程发送 `done` 消息。因此，如果我们的状态不是 `done`，我们需要调度另一个队列检查。

这可以通过在 `check_queue()` 的末尾调用 `after()` 来完成，如下所示：

```py
 if msg != 'done':
      self.after(100, self.check_queue) 
```

现在 `check_queue()` 将执行其工作，然后每 `100` 毫秒调度自己再次运行，直到状态为 `done`。剩下的只是在 `go()` 的末尾启动进程，如下所示：

```py
 def go(self):
    p = Backend(self.queue)
    p.start()
    **self.check_queue()** 
```

如果你运行这个应用程序，你会看到我们能够实时地（相对地）获得状态消息。与我们在本章早期创建的单线程应用程序不同，即使在任务运行时，也没有冻结。

## 向我们的线程上传器添加通信队列

让我们应用我们对队列的知识来解决`ThreadedUploader`类的问题。首先，我们将更新初始化器签名，以便我们可以传入一个`Queue`对象，然后将其存储为实例属性，如下所示：

```py
# models.py, in ThreadedUploader
  def __init__(
    self, session_cookie, files_url, filepath, **queue**
  ):
  # ...
  **self.queue = queue** 
```

正如我们在示例应用程序中所做的那样，我们将在`CorporateRestModel`对象中创建`Queue`对象，以便上传者和模型都可以引用它。此外，我们将保存队列作为模型的公共属性，以便应用程序对象也可以引用它。为此，我们首先需要将`Queue`导入到`models.py`中，所以请在顶部添加此导入：

```py
# models.py, at the top
from queue import Queue 
```

现在，回到`CorporateRestModel`初始化器中，创建一个`Queue`对象：

```py
# models.py, inside CorporateRestModel
  def __init__(self, base_url):
    #...
    **self.queue = Queue()** 
```

接下来，我们需要更新`upload_file()`方法，以便它将队列传递给`ThreadedUploader`对象：

```py
 def upload_file(self, filepath):
    cookie = self.session.cookies.get('session')
    uploader = ThreadedUploader(
      cookie, self.files_url, filepath, **self.queue**
    )
    uploader.start() 
```

现在，GUI 可以从`rest_model.queue`访问队列，我们可以使用这个连接从我们的上传线程向 GUI 发送消息。然而，在我们可以使用这个连接之前，我们需要开发一个通信协议。

## 创建通信协议

现在我们已经建立了一个线程间通信的通道，我们必须决定我们的两个线程将如何通信。换句话说，上传线程将确切地在队列上放置什么，以及我们的应用程序线程应该如何响应它？我们可以在队列中随意放入任何东西，并在应用程序端继续编写`if`语句来处理出现的任何内容，但更好的方法是通过对定义一个简单的协议来标准化通信。

我们的上传线程将主要发送状态相关信息回应用程序，以便它可以在消息框或状态栏上显示正在发生的事情的更新。我们将创建一个消息格式，我们可以使用它来确定线程正在做什么，并将此信息传达给用户。

消息结构将看起来像这样：

| 字段 | 描述 |
| --- | --- |
| `status` | 表示消息类型的单个单词，例如 info 或 error |
| `subject` | 总结消息的简短句子 |
| `body` | 包含关于消息详细信息的较长的字符串 |

我们可以使用字典或类创建这样的结构，但像这样简单的命名字段集合是一个很好的用例。`collections.namedtuple()`函数允许我们快速创建只包含命名属性的迷你类。

创建`namedtuple`类的样子如下：

```py
from collections import namedtuple
MyClass = namedtuple('MyClass', ['prop1', 'prop2']) 
```

这相当于编写：

```py
class MyClass():
  def __init__(self, prop1, prop2):
    self.prop1 = prop1
    self.prop2 = prop2 
```

`namedtuple()`方法比创建一个类要快得多，并且与字典不同，它强制统一性——也就是说，每个`MyClass`对象都必须有`prop1`和`prop2`属性，而字典从不要求有特定的键。

在`models.py`文件的顶部，让我们导入`namedtuple`并使用它来定义一个名为`Message`的类：

```py
# models.py, at the top
from collections import namedtuple
Message = namedtuple('Message', ['status', 'subject', 'body']) 
```

现在我们已经创建了`Message`类，创建一个新的`Message`对象就像创建任何其他类的实例一样：

```py
message = Message(
  'info', 'Testing the class', 
  'We are testing the Message class'
) 
```

让我们在队列中实现这些`Message`对象的使用。

## 从上传器发送消息

现在我们已经建立了一个协议，是时候将其付诸实践了。定位`ThreadedUploader`类，让我们更新`run()`方法以发送消息，从信息性消息开始：

```py
# models.py, in ThreadedUploader
  def run(self, *args, **kwargs):
    self.queue.put(
      Message(
        'info', 'Upload Started', 
        f'Begin upload of {self.filepath}'
      )
    ) 
```

我们的第一条消息只是一个信息性消息，表明上传开始。接下来，我们将开始上传并返回一些指示操作成功或失败的消息：

```py
 with open(self.filepath, 'rb') as fh:
      files = {'file': fh}
      response = self.session.put(
        self.files_url, files=files
      )
    try:
      response.raise_for_status()
    except Exception as e:
      self.queue.put(Message('error', 'Upload Error', str(e)))
    else:
      self.queue.put(
        Message(
          'done',  'Upload Succeeded',
          f'Upload of {self.filepath} to REST succeeded'
        )
      ) 
```

如前所述，我们通过打开文件并向网络服务发出`PUT`请求开始上传过程。这次，我们在`try`块中运行`raise_for_status()`。如果操作中捕获到异常，我们在队列中放置一个状态为`error`的消息以及异常的文本。如果我们成功，我们在队列中放置一个成功消息。

这就是我们的`ThreadedUploader`需要做的；现在我们需要转向 GUI 以实现对这些消息的响应。

## 处理队列消息

在`Application`对象中，我们需要添加一些代码来监控队列，并在从线程发送消息时采取适当的行动。正如我们在队列演示应用程序中所做的那样，我们将创建一个方法，使用 Tkinter 事件循环定期轮询队列并处理从模型的队列对象发送的任何消息。

这样启动`Application._check_queue()`方法：

```py
# application.py, inside Application
  def _check_queue(self, queue):
    while not queue.empty():
      item = queue.get() 
```

该方法接受一个`Queue`对象，首先检查它是否有任何项目。如果有，它检索一个。一旦我们有一个，我们需要检查它并根据`status`值确定如何处理它。

首先，让我们处理一个`done`状态；在`if`块下添加此代码：

```py
# application.py, inside Application._check_queue()
      if item.status == 'done':
        messagebox.showinfo(
          item.status,
          message=item.subject,
          detail=item.body
        )
        self.status.set(item.subject)
        return 
```

当我们的上传成功完成时，我们想要显示一个消息框并设置状态，然后返回而不做其他任何事情。

`Message`对象的`status`、`subject`和`body`属性很好地映射到消息框的`title`、`message`和`detail`参数，所以我们直接将它们传递给它。我们还通过设置`status`变量在应用程序的状态栏中显示消息的主题。

接下来，我们将处理队列中的`error`消息：

```py
 elif item.status == 'error':
        messagebox.showerror(
          item.status,
          message=item.subject,
          detail=item.body
        )
        self.status.set(item.subject)
        return 
```

再次显示一个消息框，这次使用`showerror()`。我们还想退出方法，因为线程可能已经退出，我们不需要安排下一次队列检查。

最后，让我们处理`info`状态：

```py
 else:
        self.status.set(f'{item.subject}: {item.body}') 
```

信息性消息并不真正需要模态消息框，所以我们只是将它们发送到状态栏。

在这个方法中，我们最后需要确保如果线程仍在运行，它会被再次调用。由于`done`和`error`消息会导致方法返回，如果我们已经到达函数的这个点，线程仍在运行，我们应该继续轮询它。因此，我们将添加一个对`after()`的调用：

```py
 self.after(100, self._check_queue, queue) 
```

`_check_queue()`编写完成后，我们只需要消除`_upload_to_corporate_rest()`末尾围绕`rest_model.upload_file()`的异常处理，并调用`_check_queue()`代替：

```py
# application.py, in Application._upload_to_corporate_rest()
        rest_model.upload_file(csvfile)
        self._check_queue(self.rest_queue) 
```

这个调用不需要使用`after()`来调度，因为第一次调用很可能没有消息，导致`_check_queue()`只是调度其下一次调用并返回。

现在我们已经完成了更新，启动测试服务器和应用程序，再次尝试 REST 上传。观察状态栏，你会看到进度条被显示出来，当过程完成时会显示一个消息框。尝试关闭 HTTP 服务器，你应该会立即看到一个错误消息弹出。

# 使用锁来保护共享资源

虽然我们的应用程序在慢速文件上传期间不再冻结是件好事，但它也引发了一个潜在的问题。假设用户在第一个上传正在进行时尝试启动第二个 REST 上传？继续尝试这个操作；启动示例 HTTP 服务器和应用程序，并尝试快速连续启动两个 REST 上传，以便第二个在上一个完成之前开始。注意 REST 服务器的输出；根据你的时间，你可能会看到一些令人困惑的日志消息，百分比上下波动，因为两个线程同时上传文件。

当然，我们的示例 REST 服务器只是使用`sleep()`模拟慢速链接；实际的文件上传发生得非常快，不太可能引起问题。在真正慢速网络的情况下，并发上传可能会更成问题。虽然接收服务器可能足够健壮，可以合理地处理两个尝试上传相同文件的线程，但最好我们一开始就避免这种情况。

我们需要一种某种类型的标志，它可以在线程之间共享，以指示一个线程是否正在上传，这样其他线程就会知道不要这样做。我们可以使用`threading`模块的`Lock`对象来实现这一点。

## 理解锁对象

锁是一个非常简单的对象，有两个状态：**获取**和**释放**。当`Lock`对象处于释放状态时，任何线程都可以调用它的`acquire()`方法将其置于获取状态。一旦一个线程获取了锁，`acquire()`方法将阻塞，直到通过调用其`release()`方法释放锁。这意味着如果另一个线程调用`acquire()`，它的执行将等待直到第一个线程释放锁。

要了解这是如何工作的，请查看本章前面创建的`basic_threading_demo.py`脚本。从终端提示符运行该脚本，在`Entry`小部件中输入一个句子，然后点击**运行线程化**按钮。

正如我们之前提到的，句子以每秒一个单词的速度打印到终端输出。但现在，连续两次快速点击**运行线程**按钮。注意，输出是一团糟的重复单词，因为两个线程同时向终端输出文本。你可以想象在类似的情况下，多个线程会对文件或网络会话造成多大的破坏。

为了纠正这个问题，让我们创建一个锁。首先，从`threading`模块导入`Lock`并创建一个实例：

```py
# basic_threading_demo_with_lock.py
from threading import Thread, Lock
print_lock = Lock() 
```

现在，在`print_slowly()`函数内部，让我们在方法周围添加对`acquire()`和`release()`的调用，如下所示：

```py
def print_slowly(string):
  print_lock.acquire()
  words = string.split()
  for word in words:
    sleep(1)
    print(word)
  print_lock.release() 
```

将此文件保存为`basic_threading_demo_with_lock.py`并再次运行。现在，当你多次点击**运行线程**按钮时，每次运行都会等待前一个运行释放锁后再开始。这样，我们可以在保持应用程序响应的同时强制线程相互等待。

`Lock`对象也可以用作上下文管理器，这样在进入块时调用`acquire()`，在退出时调用`release()`。因此，我们可以将前面的示例重写如下：

```py
 with print_lock:
    words = string.split()
    for word in words:
      sleep(1)
      print(word) 
```

## 使用锁对象防止并发上传

让我们将对`Lock`对象的理解应用到防止对公司的 REST 服务器并发上传。首先，我们需要将`Lock`导入到`models.py`中，如下所示：

```py
from threading import Thread, **Lock** 
```

接下来，我们将创建一个`Lock`对象作为`ThreadedUploader`类的类属性，如下所示：

```py
class ThreadedUploader(Thread):
  **rest_upload_lock = Lock()** 
```

回想一下*第四章*，*使用类组织代码*，分配给类属性的实例是共享的。因此，通过将锁作为类属性创建，任何`ThreadedUploader`线程都可以访问这个锁。

现在，在`run()`方法内部，我们需要使用我们的锁。最干净的方法是将其用作上下文管理器，如下所示：

```py
# models.py, inside ThreadedUploader.run()
    with self.upload_lock:
      with open(self.filepath, 'rb') as fh:
        files = {'file': fh}
        response = self.session.put(
          self.files_url, files=files
        )
        #... remainder of method in this block 
```

无论`put()`调用是返回还是抛出异常，上下文管理器都会确保在块退出时调用`release()`，以便其他对`run()`的调用可以获取锁。

添加此代码后，再次运行测试 HTTP 服务器和应用程序，并尝试快速连续启动两个 REST 上传。现在你应该会看到第二个上传直到第一个完成才启动。

### 线程和 GIL

每当我们讨论 Python 中的线程时，了解 Python 的全局解释器锁（GIL）及其对线程的影响是非常重要的。

GIL 是一种锁机制，通过防止多个线程同时执行 Python 命令来保护 Python 的内存管理。类似于我们在`ThreadedUploader`类中实现的锁，GIL 可以被看作是一个只能由一个线程一次持有的令牌；持有令牌的线程可以执行 Python 指令，其余的则必须等待。

这可能看起来像是违背了 Python 多线程的理念，然而，有两个因素可以减轻 GIL 的影响：

+   首先，GIL 只限制 Python 代码的执行；许多库在其它语言中执行代码。例如，Tkinter 执行 TCL 代码，而 `psycopg2` 执行编译后的 C 代码。这类非 Python 代码可以在单独的线程中运行，同时 Python 代码在另一个线程中运行。

+   其次，**输入/输出**（**I/O**）操作，如磁盘访问或网络请求，可以与 Python 代码并发运行。例如，当我们使用 `requests` 发起 HTTP 请求时，在等待服务器响应的过程中，全局解释器锁（GIL）会被释放。

GIL 真正限制多线程效用的情况是当我们有计算密集型的 Python 代码时。在典型的以数据为导向的应用程序（如 ABQ）中的慢速操作很可能是 I/O 基于的操作，对于计算密集型的情况，我们可以使用非 Python 库，如 `numpy`。即便如此，了解 GIL 并知道它可能会影响多线程设计的有效性仍然是好的。

# 摘要

在本章中，你学习了如何使用异步和多线程编程技术从你的程序中移除无响应行为。你学习了如何使用 `after()` 和 `update()` 方法与 Tkinter 的事件队列进行交互和控制，以及如何将这些方法应用于解决应用程序中的问题。你还学习了如何使用 Python 的 `threading` 模块在后台运行进程，以及如何利用 `Queue` 对象在线程之间进行通信。最后，你学习了如何使用 `Lock` 对象防止共享资源被破坏。

在下一章中，我们将探索 Tkinter 中最强大的小部件：Canvas。我们将学习如何绘制图像和动画化它们，以及创建有用和富有信息量的图表。
