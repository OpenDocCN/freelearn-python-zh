# 异步编程

在本章中，我们将介绍以下食谱：

+   调度操作

+   在线程上运行方法

+   执行 HTTP 请求

+   将线程与进度条连接起来

+   取消已调度的操作

+   处理空闲任务

+   生成单独的进程

# 介绍

与任何其他编程语言一样，Python 允许您将进程执行分成多个可以在时间上独立执行的单元，称为**线程**。当启动 Python 程序时，它会在**主线程**中开始执行。

Tkinter 的主循环必须从主线程开始，负责处理所有 GUI 的事件和更新。默认情况下，我们的应用程序代码，如回调和事件处理程序，也将在此线程中执行。

然而，如果我们在这个线程中启动一个长时间运行的操作，主线程的执行将会被阻塞，因此 GUI 将会冻结，并且不会响应用户事件。

在本章中，我们将介绍几种方法来实现应用程序的响应性，同时在后台执行单独的操作，并了解如何与它们交互。

# 调度操作

在 Tkinter 中防止阻塞主线程的基本技术是调度一个在超时后被调用的操作。

在本食谱中，我们将介绍如何使用`after()`方法在 Tkinter 中实现这一点，该方法可以从所有 Tkinter 小部件类中调用。

# 准备就绪

以下代码展示了一个回调如何阻塞主循环的简单示例。

该应用程序由一个按钮组成，当单击时会被禁用，等待 5 秒，然后再次启用。一个简单的实现如下：

```py
import time
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.button = tk.Button(self, command=self.start_action,
                                text="Wait 5 seconds")
        self.button.pack(padx=20, pady=20)

    def start_action(self):
        self.button.config(state=tk.DISABLED)
        time.sleep(5)
        self.button.config(state=tk.NORMAL)

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

如果运行上述程序，您会注意到**等待 5 秒**按钮根本没有被禁用，但点击它会使 GUI 冻结 5 秒。我们可以直接注意到按钮样式的变化，看起来是活动的而不是禁用的；此外，标题栏在 5 秒时间到之前将不会响应鼠标点击：

![](img/c38a83be-7aae-4da8-9461-e4c44b592c65.png)

如果我们包含了其他小部件，比如输入框和滚动条，这也会受到影响。

现在，我们将看看如何通过调度操作而不是挂起线程执行来实现所需的功能。

# 如何做...

`after()`方法允许您注册一个回调函数，在 Tkinter 的主循环中延迟指定的毫秒数后调用。您可以将这些注册的警报视为应该在系统空闲时立即处理的事件。

因此，我们将使用`self.after(5000, callback)`替换对`time.sleep(5)`的调用。我们使用`self`实例，因为`after()`方法也可以在根`Tk`实例中使用，并且从子小部件中调用它不会有任何区别：

```py
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.button = tk.Button(self, command=self.start_action,
                                text="Wait 5 seconds")
        self.button.pack(padx=50, pady=20)

    def start_action(self):
        self.button.config(state=tk.DISABLED)
        self.after(5000, lambda: self.button.config(state=tk.NORMAL))

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

使用上述方法，应用程序在调度操作被调用之前是响应的。按钮的外观将变为禁用状态，我们也可以像往常一样与标题栏交互：

![](img/064d3045-79e4-43cc-ae25-e13f2ab909e7.png)

# 工作原理...

从前面部分提到的示例中，您可能会认为`after()`方法会在给定的毫秒数经过后准确执行回调。

然而，它只是请求 Tkinter 注册一个警报，保证不会在指定的时间之前执行；因此，如果主线程忙碌，实际执行时间是没有上限的。

我们还应该记住，在调度操作之后，方法的执行立即继续。以下示例说明了这种行为：

```py
print("First")
self.after(1000, lambda: print("Third"))
print("Second")
```

上述代码段将分别在 1 秒后打印`"First"`，`"Second"`和`"Third"`。在此期间，主线程将保持 GUI 响应，并且用户可以像往常一样与应用程序交互。

通常，我们希望防止同一后台操作的运行超过一次，因此最好禁用触发执行的小部件。

不要忘记，任何预定的函数都将在主线程上执行，因此仅仅使用`after()`是不足以防止 GUI 冻结的；还重要的是避免执行长时间运行的方法作为回调。

在下一个示例中，我们将看看如何利用单独的线程执行这些阻塞操作。

# 还有更多...

`after()`方法返回一个预定警报的标识符，可以将其传递给`after_cancel()`方法以取消回调的执行。

在另一个示例中，我们将看到如何使用这种方法实现停止预定回调的功能。

# 另请参阅

+   *取消预定操作*示例

# 在线程上运行方法

由于主线程应该负责更新 GUI 和处理事件，因此其余的后台操作必须在单独的线程中执行。

Python 的标准库包括`threading`模块，用于使用高级接口创建和控制多个线程，这将允许我们使用简单的类和方法。

值得一提的是，CPython——参考 Python 实现——受**GIL**（**全局解释器锁**）的固有限制，这是一种防止多个线程同时执行 Python 字节码的机制，因此它们无法在单独的核心上运行，无法充分利用多处理器系统。如果尝试使用`threading`模块来提高应用程序的性能，应该记住这一点。

# 如何做...

以下示例将`time.sleep()`的线程暂停与通过`after()`调度的操作结合起来：

```py
import time
import threading
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.button = tk.Button(self, command=self.start_action,
                                text="Wait 5 seconds")
        self.button.pack(padx=50, pady=20)

    def start_action(self):
        self.button.config(state=tk.DISABLED)
        thread = threading.Thread(target=self.run_action)
        print(threading.main_thread().name)
        print(thread.name)
        thread.start()
        self.check_thread(thread)

    def check_thread(self, thread):
        if thread.is_alive():
            self.after(100, lambda: self.check_thread(thread))
        else:
            self.button.config(state=tk.NORMAL)

    def run_action(self):
        print("Starting long running action...")
        time.sleep(5)
        print("Long running action finished!")

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

# 它是如何工作的...

要创建一个新的`Thread`对象，可以使用带有`target`关键字参数的构造函数，在调用其`start()`方法时将在单独的线程上调用它。

在前面的部分中，我们在当前应用程序实例上使用了对`run_action`方法的引用：

```py
    thread = threading.Thread(target=self.run_action)
    thread.start()
```

然后，我们使用`after()`定期轮询线程状态，直到线程完成为止：

```py
    def check_thread(self, thread):
        if thread.is_alive():
            self.after(100, lambda: self.check_thread(thread))
        else:
            self.button.config(state=tk.NORMAL)
```

在前面的代码片段中，我们设置了`100`毫秒的延迟，因为没有必要以更频繁的频率进行轮询。当然，这个数字可能会根据线程操作的性质而变化。

这个时间线可以用以下序列图表示：

![](img/fea2c979-482b-44a4-a0b5-c4ed260cd966.jpg)

**Thread-1**上的矩形表示它忙于执行**time.sleep(5)**的时间。与此同时，**MainThread**只定期检查状态，没有操作长到足以导致 GUI 冻结。

# 还有更多...

在这个示例中，我们简要介绍了`Thread`类，但同样重要的是指出一些关于在 Python 程序中实例化和使用线程的细节。

# 线程方法 - start、run 和 join

在我们的示例中，我们调用了`start()`，因为我们希望在单独的线程中执行该方法并继续执行当前线程。

另一方面，如果我们调用了`join()`方法，主线程将被阻塞，直到新线程终止。因此，即使我们使用多个线程，它也会导致我们想要避免的相同的“冻结”行为。

最后，`run()`方法是线程实际执行其可调用目标操作的地方。当我们扩展`Thread`类时，我们将覆盖它，就像下一个示例中一样。

作为一个经验法则，始终记住从主线程调用`start()`以避免阻塞它。

# 参数化目标方法

在使用`Thread`类的构造函数时，可以通过`args`参数指定目标方法的参数：

```py
    def start_action(self):
        self.button.config(state=tk.DISABLED)
        thread = threading.Thread(target=self.run_action, args=(5,))
        thread.start()
        self.check_thread(thread)

    def run_action(self, timeout):
        # ...
```

请注意，由于我们正在使用当前实例引用目标方法，因此`self`参数会自动传递。在新线程需要访问来自调用方实例的信息的情况下，这可能很方便。

# 执行 HTTP 请求

通过 HTTP 与远程服务器通信是异步编程的常见用例。客户端执行请求，该请求使用 TCP/IP 协议在网络上传输；然后，服务器处理信息并将响应发送回客户端。

执行此操作所需的时间可能会从几毫秒到几秒不等，但在大多数情况下，可以安全地假设用户可能会注意到这种延迟。

# 做好准备

互联网上有很多第三方网络服务可以免费访问以进行原型设计。但是，我们不希望依赖外部服务，因为其 API 可能会更改，甚至可能会下线。

对于这个示例，我们将实现我们自己的 HTTP 服务器，该服务器将生成一个随机的 JSON 响应，该响应将打印在我们单独的 GUI 应用程序中：

```py
import time
import json
import random
from http.server import HTTPServer, BaseHTTPRequestHandler

class RandomRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Simulate latency
        time.sleep(3)

        # Write response headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        # Write response body
        body = json.dumps({'random': random.random()})
        self.wfile.write(bytes(body, "utf8"))

def main():
    """Starts the HTTP server on port 8080"""
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, RandomRequestHandler)
    httpd.serve_forever()

if __name__ == "__main__":
    main()
```

要启动此服务器，请运行`server.py`脚本，并保持进程运行以接受本地端口`8080`上的传入 HTTP 请求。

# 如何做...

我们的客户端应用程序包括一个简单的标签，用于向用户显示信息，以及一个按钮，用于向我们的本地服务器执行新的 HTTP 请求：

```py
import json
import threading
import urllib.request
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HTTP request example")
        self.label = tk.Label(self,
                              text="Click 'Start' to get a random 
                              value")
        self.button = tk.Button(self, text="Start",
                                command=self.start_action)
        self.label.pack(padx=60, pady=10)
        self.button.pack(pady=10)

    def start_action(self):
        self.button.config(state=tk.DISABLED)
        thread = AsyncAction()
        thread.start()
        self.check_thread(thread)

    def check_thread(self, thread):
        if thread.is_alive():
            self.after(100, lambda: self.check_thread(thread))
        else:
            text = "Random value: {}".format(thread.result)
            self.label.config(text=text)
            self.button.config(state=tk.NORMAL)

class AsyncAction(threading.Thread):
    def run(self):
        self.result = None
        url = "http://localhost:8080"
        with urllib.request.urlopen(url) as f:
            obj = json.loads(f.read().decode("utf-8"))
            self.result = obj["random"]

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

当请求完成时，标签显示服务器中生成的随机值，如下所示：

![](img/1dafeaf9-7935-446d-a1fb-b04232acb21b.png)

通常情况下，当异步操作正在运行时，按钮会被禁用，以避免在处理前一个请求之前执行新的请求。

# 工作原理...

在这个示例中，我们扩展了`Thread`类，以使用更面向对象的方法实现必须在单独线程中运行的逻辑。这是通过覆盖其`run()`方法来完成的，该方法将负责执行对本地服务器的 HTTP 请求：

```py
class AsyncAction(threading.Thread):
    def run(self):
        # ...
```

有很多 HTTP 客户端库，但在这里，我们将简单地使用标准库中的`urllib.request`模块。该模块包含`urlopen()`函数，可以接受 URL 字符串并返回一个 HTTP 响应，可以作为上下文管理器使用，即可以使用`with`语句安全地读取和关闭。

服务器返回一个 JSON 文档，如下所示（您可以通过在浏览器中打开`http://localhost:8080`URL 来检查）：

```py
{"random": 0.0915826359180778}
```

为了将字符串解码为对象，我们将响应内容传递给`json`模块的`loads()`函数。由于这样，我们可以像使用字典一样访问随机值，并将其存储在`result`属性中，该属性初始化为`None`，以防止主线程在发生错误时读取未设置的字段：

```py
def run(self):
    self.result = None
    url = "http://localhost:8080"
    with urllib.request.urlopen(url) as f:
        obj = json.loads(f.read().decode("utf-8"))
        self.result = obj["random"]
```

然后，GUI 定期轮询线程状态，就像我们在前面的示例中看到的那样：

```py
    def check_thread(self, thread):
        if thread.is_alive():
            self.after(100, lambda: self.check_thread(thread))
        else:
            text = "Random value: {}".format(thread.result)
            self.label.config(text=text)
            self.button.config(state=tk.NORMAL)
```

这里，主要的区别在于一旦线程不再活动，我们可以检索`result`属性的值，因为它在执行结束之前已经设置。

# 另请参阅

+   *在线程上运行方法*示例

# 将线程与进度条连接起来

进度条是后台任务状态的有用指示器，显示相对于进度的逐步填充部分。它们经常用于长时间运行的操作，因此通常将它们与执行这些任务的线程连接起来，以向最终用户提供视觉反馈。

# 做好准备

我们的示例应用程序将包括一个水平进度条，一旦用户点击“开始”按钮，它将增加固定数量的进度：

![](img/39ccaa8f-15d9-4d73-bfc0-e0be22d147d2.png)

# 如何做...

为了模拟后台任务的执行，进度条的增量将由一个不同的线程生成，该线程将在每个步骤之间暂停 1 秒。

通信将使用同步队列进行，这允许我们以线程安全的方式交换信息：

```py
import time
import queue
import threading
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as mb

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Progressbar example")
        self.queue = queue.Queue()
        self.progressbar = ttk.Progressbar(self, length=300,
                                           orient=tk.HORIZONTAL)
        self.button = tk.Button(self, text="Start",
                                command=self.start_action)

        self.progressbar.pack(padx=10, pady=10)
        self.button.pack(padx=10, pady=10)

    def start_action(self):
        self.button.config(state=tk.DISABLED)
        thread = AsyncAction(self.queue, 20)
        thread.start()
        self.poll_thread(thread)

    def poll_thread(self, thread):
        self.check_queue()
        if thread.is_alive():
            self.after(100, lambda: self.poll_thread(thread))
        else:
            self.button.config(state=tk.NORMAL)
            mb.showinfo("Done!", "Async action completed")

    def check_queue(self):
        while self.queue.qsize():
            try:
                step = self.queue.get(0)
                self.progressbar.step(step * 100)
            except queue.Empty:
                pass

class AsyncAction(threading.Thread):
    def __init__(self, queue, steps):
        super().__init__()
        self.queue = queue
        self.steps = steps

    def run(self):
        for _ in range(self.steps):
            time.sleep(1)
            self.queue.put(1 / self.steps)

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

# 它是如何工作的...

`Progressbar`是`tkinter.ttk`模块中包含的一个主题小部件。我们将在第八章中深入探讨这个模块，探索它定义的新小部件，但到目前为止，我们只需要将`Progressbar`作为常规小部件使用。

我们还需要导入`queue`模块，该模块定义了同步集合，如`Queue`。在多线程环境中，同步性是一个重要的主题，因为如果在完全相同的时间访问共享资源，可能会出现意外的结果，我们将这些不太可能但可能发生的情况定义为**竞争条件**。

通过这些添加，我们的`App`类包含了这些新的语句：

```py
# ...
import queue
import tkinter.ttk as ttk

class App(tk.Tk):
    def __init__(self):
        # ...
        self.queue = queue.Queue()
 self.progressbar = ttk.Progressbar(self, length=300,
 orient=tk.HORIZONTAL)
```

与以前的示例一样，`start_action()`方法启动一个线程，传递队列和将模拟长时间运行任务的步数：

```py
    def start_action(self):
        self.button.config(state=tk.DISABLED)
        thread = AsyncAction(self.queue, 20)
        thread.start()
        self.poll_thread(thread)
```

我们的`AsyncAction`子类定义了一个自定义构造函数来接收这些参数，这些参数将在`run()`方法中使用：

```py
class AsyncAction(threading.Thread):
    def __init__(self, queue, steps):
        super().__init__()
        self.queue = queue
        self.steps = steps

    def run(self):
        for _ in range(self.steps):
            time.sleep(1)
            self.queue.put(1 / self.steps)
```

循环暂停线程的执行 1 秒，并根据`steps`属性中指示的次数将增量添加到队列中。

从应用程序实例中读取队列，从`check_queue()`中检查队列中添加的项目：

```py
    def check_queue(self):
        while self.queue.qsize():
            try:
                step = self.queue.get(0)
                self.progressbar.step(step * 100)
            except queue.Empty:
                pass
```

从`poll_thread()`定期调用以下方法，该方法轮询线程状态并使用`after()`再次调度自己，直到线程完成执行：

```py
    def poll_thread(self, thread):
        self.check_queue()
        if thread.is_alive():
            self.after(100, lambda: self.poll_thread(thread))
        else:
            self.button.config(state=tk.NORMAL)
            mb.showinfo("Done!", "Async action completed")
```

# 另请参阅

+   *在线程上运行方法*食谱

# 取消预定的操作

Tkinter 的调度机制不仅提供了延迟回调执行的方法，还提供了取消它们的方法，如果它们尚未执行。考虑一个可能需要太长时间才能完成的操作，因此我们希望让用户通过按下按钮或关闭应用程序来停止它。

# 准备工作

我们将从第一个食谱中获取示例，并添加一个 Stop 按钮，以允许我们取消预定的操作。

这个按钮只有在操作被预定时才会启用，这意味着一旦单击左按钮，用户可以等待 5 秒，或者单击 Stop 按钮立即再次启用它：

![](img/d3b3fcb2-7afe-4875-83aa-f97e554a4787.png)

# 如何做到这一点...

`after_cancel()`方法通过获取先前调用`after()`返回的标识符来取消预定操作的执行。在这个例子中，这个值存储在`scheduled_id`属性中：

```py
import time
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.button = tk.Button(self, command=self.start_action,
                                text="Wait 5 seconds")
        self.cancel = tk.Button(self, command=self.cancel_action,
                                text="Stop", state=tk.DISABLED)
        self.button.pack(padx=30, pady=20, side=tk.LEFT)
        self.cancel.pack(padx=30, pady=20, side=tk.LEFT)

    def start_action(self):
        self.button.config(state=tk.DISABLED)
        self.cancel.config(state=tk.NORMAL)
        self.scheduled_id = self.after(5000, self.init_buttons)

    def init_buttons(self):
        self.button.config(state=tk.NORMAL)
        self.cancel.config(state=tk.DISABLED)

    def cancel_action(self):
        print("Canceling scheduled", self.scheduled_id)
        self.after_cancel(self.scheduled_id)
        self.init_buttons()

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

# 它是如何工作的...

要取消回调，我们首先需要`after()`返回的警报标识符。我们将把这个标识符存储在`scheduled_id`属性中，因为我们将在一个单独的方法中需要它：

```py
    def start_action(self):
        self.button.config(state=tk.DISABLED)
        self.cancel.config(state=tk.NORMAL)
        self.scheduled_id = self.after(5000, self.init_buttons)
```

然后，该字段被传递给`Stop`按钮的回调函数中的`after_cancel()`：

```py
    def cancel_action(self):
        print("Canceling scheduled", self.scheduled_id)
        self.after_cancel(self.scheduled_id)
        self.init_buttons()
```

在我们的情况下，一旦单击`Start`按钮，将其禁用是很重要的，因为如果`start_action()`被调用两次，`scheduled_id`将被覆盖，而`Stop`按钮只能取消最后一个预定的操作。

顺便说一句，如果我们使用已经执行过的警报标识符调用`after_cancel()`，它将没有效果。

# 还有更多...

在本节中，我们介绍了如何取消预定的警报，但是如果此回调正在轮询后台线程的状态，您可能会想知道如何停止线程。

不幸的是，没有官方的 API 可以优雅地停止`Thread`实例。如果您已经定义了一个自定义子类，您可能需要在其`run()`方法中定期检查的标志。

```py
class MyAsyncAction(threading.Thread):
    def __init__(self):
        super().__init__()
        self.do_stop = False

    def run(self):
        # Start execution...
        if not self.do_stop:
            # Continue execution...
```

然后，当调用`after_cancel()`时，这个标志可以通过设置`thread.do_stop = True`来外部修改，也可以停止线程。

显然，这种方法将严重依赖于`run()`方法内部执行的操作，例如，如果它由一个循环组成，那么您可以在每次迭代之间执行此检查。

从 Python 3.4 开始，您可以使用`asyncio`模块，其中包括管理异步操作的类和函数，包括取消。尽管这个模块超出了本书的范围，但如果您面对更复杂的情况，我们建议您探索一下。

# 处理空闲任务

有些情况下，某个操作会导致程序执行时出现短暂的暂停。它甚至可能不到一秒就完成，但对于用户来说仍然是可察觉的，因为它在 GUI 中引入了短暂的暂停。

在这个配方中，我们将讨论如何处理这些情况，而无需在单独的线程中处理整个任务。

# 准备工作

我们将从*Scheduling actions*配方中取一个例子，但超时时间为 1 秒，而不是 5 秒。

# 如何做...

当我们将按钮的状态更改为`DISABLED`时，回调函数继续执行，因此按钮的状态实际上直到系统处于空闲状态时才会更改，这意味着它必须等待`time.sleep()`完成。

但是，我们可以强制 Tkinter 在特定时刻更新所有挂起的 GUI 更新，如下面的脚本所示：

```py
import time
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.button = tk.Button(self, command=self.start_action,
                                text="Wait 1 second")
        self.button.pack(padx=30, pady=20)

    def start_action(self):
        self.button.config(state=tk.DISABLED)
        self.update_idletasks()
        time.sleep(1)
        self.button.config(state=tk.NORMAL)

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

# 工作原理...

在前面部分提到的代码片段中，关键是调用`self.update_idletasks()`。由于这一点，按钮状态的更改在调用`time.sleep()`之前由 Tkinter 处理。因此，在回调被暂停的一秒钟内，按钮具有期望的外观，而不是 Tkinter 在调用回调之前设置的`ACTIVE`状态。

我们使用`time.sleep()`来说明一个语句执行时间长，但足够短，可以考虑将其移到新线程中的情况——在现实世界的场景中，这将是一个更复杂的计算操作。

# 生成单独的进程

在某些情况下，仅使用线程可能无法实现应用程序所需的功能。例如，您可能希望调用用不同语言编写的单独程序。

在这种情况下，我们还需要使用`subprocess`模块从 Python 进程中调用目标程序。

# 准备工作

以下示例执行对指定 DNS 或 IP 地址的 ping 操作：

![](img/2d22640c-8c92-43f6-8cec-59f2f6dcb862.png)

# 如何做...

像往常一样，我们定义一个自定义的`AsyncAction`方法，但在这种情况下，我们使用 Entry 小部件中设置的值调用`subprocess.run()`。

这个函数启动一个单独的子进程，与线程不同，它使用单独的内存空间。这意味着为了获得`ping`命令的结果，我们必须将打印到标准输出的结果进行管道传输，并在我们的 Python 程序中读取它：

```py
import threading
import subprocess
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.entry = tk.Entry(self)
        self.button = tk.Button(self, text="Ping!",
                                command=self.do_ping)
        self.output = tk.Text(self, width=80, height=15)

        self.entry.grid(row=0, column=0, padx=5, pady=5)
        self.button.grid(row=0, column=1, padx=5, pady=5)
        self.output.grid(row=1, column=0, columnspan=2,
                         padx=5, pady=5)

    def do_ping(self):
        self.button.config(state=tk.DISABLED)
        thread = AsyncAction(self.entry.get())
        thread.start()
        self.poll_thread(thread)

    def poll_thread(self, thread):
        if thread.is_alive():
            self.after(100, lambda: self.poll_thread(thread))
        else:
            self.button.config(state=tk.NORMAL)
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, thread.result)

class AsyncAction(threading.Thread):
    def __init__(self, ip):
        super().__init__()
        self.ip = ip

    def run(self):
        self.result = subprocess.run(["ping", self.ip], shell=True,
                                     stdout=subprocess.PIPE).stdout

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

# 工作原理...

`run()`函数执行数组参数中指定的子进程。默认情况下，结果只包含进程的返回代码，因此我们还传递了`stdout`选项和`PIPE`常量，以指示应将标准输出流进行管道传输。

我们使用关键字参数`shell`设置为`True`来调用这个函数，以避免为`ping`子进程打开新的控制台：

```py
    def run(self):
        self.result = subprocess.run(["ping", self.ip], shell=True,
                                     stdout=subprocess.PIPE).stdout
```

最后，当主线程验证该操作已完成时，将输出打印到 Text 小部件：

```py
    def poll_thread(self, thread):
        if thread.is_alive():
            self.after(100, lambda: self.poll_thread(thread))
        else:
            self.button.config(state=tk.NORMAL)
 self.output.delete(1.0, tk.END)
 self.output.insert(tk.END, thread.result)
```
