# 第四章：处理并发

正如我们在上一章中看到的，当处理任何大型企业应用程序时，我们会处理大量数据。这些数据以同步方式处理，并且只有在特定进程的数据处理完成后才发送结果。当处理的数据不大时，这种模型是完全可以接受的。但是考虑一种情况，需要在生成响应之前处理大量数据。那么会发生什么？答案是，应用程序响应时间变慢。

我们需要一个更好的解决方案。一种允许我们并行处理数据，从而获得更快应用程序响应的解决方案。但是我们如何实现这一点呢？问题的答案是**并发...**

# 技术要求

本书中的代码清单可以在[`github.com/PacktPublishing/Hands-On-Enterprise-Application-Development-with-Python`](https://github.com/PacktPublishing/Hands-On-Enterprise-Application-Development-with-Python)的`chapter04`目录下找到。

可以通过运行以下命令克隆代码示例：

```py
git clone https://github.com/PacktPublishing/Hands-On-Enterprise-Application-Development-with-Python
```

本章中提到的代码示例需要运行 Python 3.6 及以上版本。虚拟环境是将依赖项与系统隔离的首选选项。

# 并发的需求

大多数情况下，当我们构建相当简单的应用程序时，我们不需要并发。简单的顺序编程就可以很好地工作，一个步骤在另一个步骤完成后执行。但随着应用程序用例变得越来越复杂，并且有越来越多的任务可以轻松地推入后台以改善应用程序的用户体验，我们最终围绕并发的概念展开。

并发本身就是一个不同的东西，并且使编程任务变得更加复杂。但是，尽管增加了复杂性，但并发也带来了许多功能，以改善应用程序的用户体验。

在我们深入讨论为什么我们...

# GUI 应用程序中的并发

我们已经习惯使用的硬件每年都变得越来越强大。如今，即使是我们智能手机内部的 CPU 也具有四核或八核的配置。这些配置允许并行运行多个进程或线程。不利用并发的硬件改进将是对之前提到的硬件改进的浪费。如今，当我们在智能手机上打开应用程序时，大多数应用程序都有两个或更多个线程在运行，尽管我们大部分时间都不知道。

让我们考虑一个相当简单的例子，在我们的设备上打开一个照片库应用程序。当我们打开照片库时，一个应用程序进程就会启动。这个进程负责加载应用程序的 GUI。GUI 在主线程中运行，允许我们与应用程序进行交互。现在，这个应用程序还会生成另一个后台线程，负责遍历操作系统的文件系统并加载照片的缩略图。从文件系统加载缩略图可能是一个繁琐的任务，并且可能需要一些时间，这取决于需要加载多少缩略图。

尽管我们注意到缩略图正在慢慢加载，但在整个过程中，我们的应用程序 GUI 仍然保持响应，并且我们可以与之交互，查看进度等。所有这些都是通过并发编程实现的。

想象一下，如果这里没有使用并发。应用程序将在主线程中加载缩略图。这将导致 GUI 在主线程完成加载缩略图之前变得无响应。这不仅会非常不直观，而且还会导致糟糕的用户体验，而我们通过并发编程避免了这种情况。

现在我们已经对并发编程如何证明其巨大用处有了一个大致的了解，让我们看看它如何帮助我们设计和开发企业应用程序，以及它可以实现什么。

# 企业应用程序中的并发

企业应用程序通常很大，通常涉及大量用户发起的操作，如数据检索、更新等。现在，让我们以我们的 BugZot 应用程序为例，用户可能会在提交错误报告时附上图形附件。这在提交可能影响应用程序 UI 或在 UI 上显示错误的错误时是一个常见的过程。现在，每个用户可能会提交图像，这些图像可能在质量上有所不同，因此它们的大小可能会有所不同。这可能涉及到非常小的尺寸的图像和尺寸非常大且分辨率很高的图像。作为应用程序开发人员，您可能知道以 100%质量存储图像可能会...

# 使用 Python 进行并发编程

Python 提供了多种实现并行或并发的方法。所有这些方法都有各自的优缺点，在实现方式上有根本的不同，需要根据使用情况做出选择。

Python 提供的实现并发的方法之一是在线程级别上进行，允许应用程序启动多个线程，每个线程执行一个任务。这些线程提供了一种易于使用的并发机制，并在单个 Python 解释器进程内执行，因此非常轻量级。

另一种实现并行的机制是通过使用多个进程代替多个线程。通过这种方法，每个进程在其自己的独立 Python 解释器进程内执行一个单独的任务。这种方法为多线程 Python 程序在**全局解释器锁**（**GIL**）存在的情况下可能面临的问题提供了一些解决方法，但也可能增加管理多个进程和增加内存使用量的额外开销。

因此，让我们首先看看如何使用线程实现并发，并讨论它们所附带的好处和缺点。

# 多线程的并发

在大多数现代处理器系统中，多线程的使用是司空见惯的。随着 CPU 配备多个核心和诸如超线程等技术的出现，允许单个核心同时运行多个线程，应用程序开发人员不会浪费任何一个利用这些技术提供的优势的机会。

作为一种编程语言，Python 通过使用线程模块支持多线程的实现，允许开发人员在应用程序中利用线程级别的并行性。

以下示例展示了如何使用 Python 中的线程模块构建一个简单的程序：

```py
# simple_multithreading.pyimport threadingclass SimpleThread(threading.Thread): ...
```

# 线程同步

正如我们在前一节中探讨的，虽然在 Python 中可以很容易地实现线程，但它们也有自己的陷阱，需要在编写面向生产用例的应用程序时予以注意。如果在应用程序开发时不注意这些陷阱，它们将产生难以调试的行为，这是并发程序所以闻名的。

因此，让我们试着找出如何解决前一节讨论的问题。如果我们仔细思考，我们可以将问题归类为多个线程同步的问题。应用程序的最佳行为是同步对文件的写入，以便在任何给定时间只有一个线程能够写入文件。这将强制确保在已经执行的线程完成其写入之前，没有线程可以开始写入操作。

为了实现这种同步，我们可以利用锁的力量。锁提供了一种简单的实现同步的方法。例如，将要开始写操作的线程将首先获取锁。如果锁获取成功，线程就可以继续执行其写操作。现在，如果在中间发生上下文切换，并且另一个线程即将开始写操作，它将被阻塞，因为锁已经被获取。这将防止线程在已经运行的写操作之间写入数据。

在 Python 多线程中，我们可以通过使用`threading.Lock`类来实现锁。该类提供了两种方法来方便地获取和释放锁。当线程想要在执行操作之前获取锁时，会调用`acquire()`方法。一旦锁被获取，线程就会继续执行操作。一旦线程的操作完成，线程调用`release()`方法释放锁，以便其他可能正在等待它的线程可以获取锁。

让我们看看如何使用锁来同步我们的 JSON 到 YAML 转换器示例中的线程操作。以下代码示例展示了锁的使用：

```py
import threading
import json
import yaml

class JSONConverter(threading.Thread):
        def __init__(self, json_file, yaml_file, lock):
                threading.Thread.__init__(self)
                self.json_file = json_file
                self.yaml_file = yaml_file
      self.lock = lock

        def run(self):
                print("Starting read for {}".format(self.json_file))
                self.json_reader = open(self.json_file, 'r')
                self.json = json.load(self.json_reader)
                self.json_reader.close()
                print("Read completed for {}".format(self.json_file))
                print("Writing {} to YAML".format(self.json_file))
      self.lock.acquire() # We acquire a lock before writing
                self.yaml_writer = open(self.yaml_file, 'a+')
                yaml.dump(self.json, self.yaml_writer)
                self.yaml_writer.close()
                self.lock.release() # Release the lock once our writes are done
                print("Conversion completed for {}".format(self.json_file))

files = ['file1.json', 'file2.json', 'file3.json']
write_lock = threading.Lock()
conversion_threads = []

for file in files:
        converter = JSONConverter(file, 'converted.yaml', write_lock)
        conversion_threads.append(converter)
        converter.start()

for cthread in conversion_threads:
        cthread.join()

print("Exiting")
```

在这个例子中，我们首先通过创建`threading.Lock`类的实例来创建一个`lock`变量。然后将这个实例传递给所有需要同步的线程。当一个线程需要进行写操作时，它首先通过获取锁来开始写操作。一旦这些写操作完成，线程释放锁，以便其他线程可以获取锁。

如果一个线程获取了锁但忘记释放它，程序可能会陷入死锁状态，因为没有其他线程能够继续。应该谨慎地确保一旦线程完成其操作，获取的锁就被释放，以避免死锁。

# 可重入锁

除了提供多线程的一般锁定机制的`threading.Lock`类之外，其中锁只能被获取一次直到释放，Python 还提供了另一种可能对实现递归操作的程序有用的锁定机制。这种锁，称为可重入锁，使用`threading.RLock`类实现，可以被递归函数使用。该类提供了与锁类提供的类似方法：`acquire()`和`release()`，分别用于获取和释放已获取的锁。唯一的区别是当递归函数在调用堆栈中多次调用`acquire()`时发生。当相同的函数一遍又一遍地调用获取方法时，...

# 条件变量

让我们想象一下，不知何故，我们有一种方法可以告诉我们的`Thread-1`等待，直到`Thread-2`提供了一些数据可供使用。这正是条件变量允许我们做的。它们允许我们同步依赖于共享资源的两个线程。为了更好地理解这一点，让我们看一下以下代码示例，它创建了两个线程，一个用于输入电子邮件 ID，另一个负责发送电子邮件：

```py
# condition_variable.py
import threading

class EmailQueue(threading.Thread):

    def __init__(self, email_queue, max_items, condition_var):
        threading.Thread.__init__(self)
        self.email_queue = email_queue
        self.max_items = max_items
        self.condition_var = condition_var
        self.email_recipients = []

    def add_recipient(self, email):
        self.email_recipients.append(email)

    def run(self):
        while True:
            self.condition_var.acquire()
            if len(self.email_queue) == self.max_items:
                print("E-mail queue is full. Entering wait state...")
                self.condition_var.wait()
                print("Received consume signal. Populating queue...")
            while len(self.email_queue) < self.max_items:
                if len(self.email_recipients) == 0:
                    break
                email = self.email_recipients.pop()
                self.email_queue.append(email)
                self.condition_var.notify()
            self.condition_var.release()

class EmailSender(threading.Thread):

    def __init__(self, email_queue, condition_var):
        threading.Thread.__init__(self)
        self.email_queue = email_queue
        self.condition_var = condition_var

    def run(self):
        while True:
            self.condition_var.acquire()
            if len(self.email_queue) == 0:
                print("E-mail queue is empty. Entering wait state...")
                self.condition_var.wait()
                print("E-mail queue populated. Resuming operations...")
            while len(self.email_queue) is not 0:
                email = self.email_queue.pop()
                print("Sending email to {}".format(email))
            self.condition_var.notify()
            self.condition_var.release()

queue = []
MAX_QUEUE_SIZE = 100
condition_var = threading.Condition()

email_queue = EmailQueue(queue, MAX_QUEUE_SIZE, condition_var)
email_sender = EmailSender(queue, condition_var)
email_queue.start()
email_sender.start()
email_queue.add_recipient("joe@example.com")
```

在这个代码示例中，我们定义了两个类，分别是`EmailQueue`，它扮演生产者的角色，并在电子邮件队列中填充需要发送电子邮件的电子邮件地址。然后还有另一个类`EmailSender`，它扮演消费者的角色，从电子邮件队列中获取电子邮件地址并发送邮件给它们。

现在，在`EmailQueue`的`__init__`方法中，我们接收一个 Python 列表作为参数，这个列表将作为队列使用，一个定义列表最多应该容纳多少项的变量，以及一个条件变量。

接下来，我们有一个方法`add_recipient`，它将一个新的电子邮件 ID 附加到`EmailQueue`的内部数据结构中，以临时保存电子邮件地址，直到它们被添加到发送队列中。

现在，让我们进入`run()`方法，这里发生了真正的魔术。首先，我们启动一个无限循环，使线程始终处于运行模式。接下来，我们通过调用条件变量的`acquire()`方法来获取锁。我们这样做是为了防止线程在意外时间切换上下文时对我们的数据结构进行任何形式的破坏。

一旦我们获得了锁，我们就会检查我们的电子邮件队列是否已满。如果已满，我们会打印一条消息，并调用条件变量的`wait()`方法。对`wait()`方法的调用会释放条件变量获取的锁，并使线程进入阻塞状态。只有在条件变量上调用`notify()`方法时，这种阻塞状态才会结束。现在，当线程通过`notify()`接收到信号时，它会继续其操作，首先检查内部队列中是否有一些数据。如果它在内部队列中找到了一些数据，那么它会用这些数据填充电子邮件队列，并调用条件变量的`notify()`方法来通知`EmailSender`消费者线程。现在，让我们来看看`EmailSender`类。

在这里不需要逐行阅读，让我们把重点放在`EmailSender`类的`run()`方法上。由于这个线程需要始终运行，我们首先启动一个无限循环来做到这一点。然后，我们要做的下一件事是，在共享条件变量上获取锁。一旦我们获得了锁，我们现在可以操作共享的`email_queue`数据结构。因此，我们的消费者首先要做的事情是检查电子邮件队列是否为空。如果发现队列为空，我们的消费者将调用条件变量的`wait()`方法，有效地释放锁并进入阻塞状态，直到电子邮件队列中有一些数据为止。这会导致控制权转移到负责填充队列的`EmailQueue`类。

现在，一旦电子邮件队列中有一些电子邮件 ID，消费者将开始发送邮件。一旦队列耗尽，它通过调用条件变量的`notify`方法向`EmailSender`类发出信号。这将允许`EmailSender`继续其操作，填充电子邮件队列。

让我们看看当我们尝试执行前面的示例程序时会发生什么：

```py
python condition_variable.py 
E-mail queue is empty. Entering wait state...
E-mail queue populated. Resuming operations...
Sending email to joe@example.com
E-mail queue is empty. Entering wait state...
```

通过这个例子，我们现在了解了在 Python 中如何使用条件变量来解决生产者-消费者问题。有了这些知识，现在让我们来看看在我们的应用程序中进行多线程时可能出现的一些问题。

# 多线程的常见陷阱

多线程提供了许多好处，但也伴随着一些陷阱。如果不加以避免，这些陷阱在应用程序投入生产时可能会带来痛苦的经历。这些陷阱通常会导致意外行为，可能只会偶尔发生，也可能在特定模块的每次执行时都会发生。这其中令人痛苦的是，当这些问题是由多个线程的执行引起时，很难调试这些问题，因为很难预测特定线程何时执行。因此，在开发阶段讨论这些常见陷阱发生的原因以及如何在开发阶段避免它们是值得的。

一些常见的原因是...

# 竞争条件

在多线程的上下文中，竞争条件是指两个或更多个线程尝试同时修改共享数据结构的情况，但由于线程的调度和执行方式，共享数据结构被修改成一种使其处于不一致状态的方式。

这个声明是否令人困惑？别担心，让我们通过一个例子来理解它：

考虑我们之前的 JSON 转 YAML 转换器问题的例子。现在，假设我们在将转换后的 YAML 输出写入文件时没有使用锁。现在假设我们有两个名为`writer-1`和`writer-2`的线程，它们负责向共同的 YAML 文件写入。现在，想象一下，`writer-1`和`writer-2`线程都开始了写入文件的操作，并且操作系统安排线程执行的方式是，`writer-1`开始写入文件。现在，当`writer-1`线程正在写入文件时，操作系统决定该线程完成了其时间配额，并将该线程与`writer-2`线程交换。现在，需要注意的一点是，当被交换时，`writer-1`线程尚未完成写入所有数据。现在，`writer-2`线程开始执行并完成了在 YAML 文件中的数据写入。在`writer-2`线程完成后，操作系统再次开始执行`writer-1`线程，它开始再次写入剩余的数据到 YAML 文件，然后完成。

现在，当我们打开 YAML 文件时，我们看到的是一个文件，其中包含了两个写入线程混合在一起的数据，因此，使我们的文件处于不一致的状态。`writer-1`和`writer-2`线程之间发生的问题被称为竞争条件。

竞争条件属于非常难以调试的问题类别，因为线程执行的顺序取决于机器和操作系统。因此，在一个部署上可能出现的问题在另一个部署上可能不会出现。

那么，我们如何避免竞争条件？嗯，我们已经有了问题的答案，而且我们最近刚刚使用过它们。所以，让我们来看看一些可以预防竞争条件发生的方法：

+   **在关键区域使用锁**：关键区域指的是代码中共享变量被线程修改的区域。为了防止竞争条件在关键区域发生，我们可以使用锁。锁本质上会导致除了持有锁的线程外，所有其他线程都会被阻塞。需要修改共享资源的所有其他线程只有在当前持有锁的线程释放锁时才能执行。可以使用的锁的类别包括互斥锁（一次只能由一个线程持有）、可重入锁（允许递归函数对同一共享资源进行多次锁定）和条件对象（可用于在生产者-消费者类型的环境中同步执行）。

+   **使用线程安全的数据结构**：预防竞争条件的另一种方法是使用线程安全的数据结构。线程安全的数据结构是指能够自动管理多个线程对其所做修改并串行化其操作的数据结构。Python 提供的一个线程安全的共享数据结构是队列。当操作涉及多个线程时，可以轻松地使用队列。

现在，我们对竞争条件是什么，它是如何发生的，以及如何避免有了一个概念。有了这个想法，让我们来看看由于我们预防竞争条件而可能出现的其他问题之一。

# 死锁

死锁是指两个或更多个线程永远被阻塞，因为它们彼此依赖或者一个资源永远不会被释放。让我们通过一个简单的例子来理解死锁是如何发生的：

考虑我们之前的 JSON 转 YAML 转换器的例子。现在，假设我们在线程中使用了锁，这样当一个线程开始向文件写入时，它首先对文件进行互斥锁定。现在，在线程释放这个互斥锁之前，其他线程无法执行。

因此，让我们想象一下有两个线程`writer-1`和`writer-2`，它们试图写入共同的输出文件。现在，当`writer-1`开始执行时，它首先在文件上获取锁并开始操作。...

# GIL 的故事

如果有人告诉你，即使你创建了一个多线程程序，只有一个线程可以同时执行？这种情况在系统只包含一个一次只能执行一个线程的单核心时是真实的，多个运行线程的幻觉是由 CPU 频繁地在线程之间切换而产生的。

但这种情况在 Python 的一个实现中也是真实的。Python 的原始实现，也称为 CPython，包括一个全局互斥锁，也称为 GIL，它只允许一个线程同时执行 Python 字节码。这有效地限制了应用程序一次只能执行一个线程。

GIL 是在 CPython 中引入的，因为 CPython 解释器不是线程安全的。GIL 通过交换运行多个线程的属性来有效地解决了线程安全问题。

GIL 的存在在 Python 社区中一直是一个备受争议的话题，有很多提案旨在消除它，但由于各种原因，包括对单线程应用程序性能的影响、破坏对 GIL 存在依赖的功能的向后兼容性等，没有一个提案被纳入 Python 的生产版本。

那么，GIL 的存在对于你的多线程应用程序意味着什么呢？实际上，如果你的应用程序利用多线程来执行 I/O 工作负载，那么由于大部分 I/O 发生在 GIL 之外，你可能不会受到 GIL 的性能损失影响，因此多个线程可以被复用。只有当应用程序使用多个线程执行需要大量操作应用程序特定数据结构的 CPU 密集型任务时，GIL 的影响才会被感知到。由于所有数据结构操作都涉及 Python 字节码的执行，GIL 将通过不允许多个线程同时执行严重限制多线程应用程序的性能。

那么，GIL 引起的问题是否有解决方法？答案是肯定的，但应该采用哪种解决方案完全取决于应用程序的用例。以下选项可能有助于避免 GIL：

+   **切换 Python 实现：**如果你的应用程序并不一定依赖于底层的 Python 实现，并且可以切换到另一个实现，那么有一些 Python 实现是没有 GIL 的。一些没有 GIL 的实现包括：Jython 和 IronPython，它们可以完全利用多处理器系统来执行多线程应用程序。

+   **利用多进程：**Python 在构建考虑并发的程序时有很多选择。我们探讨了多线程，这是实现并发的选项之一，但受到 GIL 的限制。实现并发的另一个选项是使用 Python 的多进程能力，它允许启动多个进程并行执行任务。由于每个进程在自己的 Python 解释器实例中运行，因此 GIL 在这里不成问题，并允许充分利用多处理器系统。

了解了 GIL 对多线程应用程序的影响，现在让我们讨论多进程如何帮助你克服并发的限制。

# 多进程并发

Python 语言提供了一些非常简单的方法来实现应用程序的并发。我们在 Python 线程库中看到了这一点，对于 Python 的多进程能力也是如此。

如果您想要借助多进程在程序中构建并发，那么借助 Python 的多进程库和该库提供的 API，实现起来非常容易。

那么，当我们说我们将使用多进程来实现并发时，我们是什么意思呢？让我们试着回答这个问题。通常，当我们谈论并发时，有两种方法可以帮助我们实现它。其中一种方法是运行单个应用程序实例，并允许其使用多个线程。...

# Python 多进程模块

Python 提供了一种简单的方法来实现多进程程序。这种实现的便利性得益于 Python 的多进程模块，该模块提供了重要的类，如 Process 类用于启动新进程；Queue 和 Pipe 类用于促进多个进程之间的通信；等等。

以下示例快速概述了如何使用 Python 的多进程库创建一个作为单独进程执行的 URL 加载器：

```py
# url_loader.py
from multiprocessing import Process
import urllib.request

def load_url(url):
    url_handle = urllib.request.urlopen(url)
    url_data = url_handle.read()
    # The data returned by read() call is in the bytearray format. We need to
    # decode the data before we can print it.
    html_data = url_data.decode('utf-8')
    url_handle.close()
    print(html_data)

if __name__ == '__main__':
    url = 'http://www.w3c.org'
    loader_process = Process(target=load_url, args=(url,))
    print("Spawning a new process to load the url")
    loader_process.start()
    print("Waiting for the spawned process to exit")
    loader_process.join()
    print("Exiting…")
```

在这个例子中，我们使用 Python 的多进程库创建了一个简单的程序，它在后台加载一个 URL 并将其信息打印到 stdout。有趣的地方在于理解我们如何轻松地在程序中生成一个新的进程。所以，让我们来看看。为了实现多进程，我们首先从 Python 的多进程模块中导入 Process 类。下一步是创建一个函数，该函数以要加载的 URL 作为参数，然后使用 Python 的 urllib 模块加载该 URL。一旦 URL 加载完成，我们就将来自 URL 的数据打印到 stdout。

接下来，我们定义程序开始执行时运行的代码。在这里，我们首先定义了我们想要加载的 URL，并将其存储在 url 变量中。接下来的部分是我们通过创建 Process 类的对象在程序中引入多进程。对于这个对象，我们将目标参数提供为我们想要执行的函数。这类似于我们在使用 Python 线程库时已经习惯的目标方法。Process 构造函数的下一个参数是 args 参数，它接受在调用目标函数时需要传递给目标函数的参数。

要生成一个新的进程，我们调用 Process 对象的 start()方法。这将在一个新的进程中启动我们的目标函数并执行其操作。我们做的最后一件事是等待这个生成的进程退出，通过调用 Process 类的 join()方法。

这就是在 Python 中创建多进程应用程序的简单方法。

现在，我们知道如何在 Python 中创建多进程应用程序，但是如何在多个进程之间分配特定的任务呢？嗯，这很容易。以下代码示例修改了我们之前示例中的入口代码，以利用多进程模块中的 Pool 类的功能来实现这一点：

```py
from multiprocessing import Pool
if __name__ == '__main__':
    url = ['http://www.w3c.org', 'http://www.microsoft.com', '[http://www.wikipedia.org', '[http://www.packt.com']
    with Pool(4) as loader_pool:
      loader_pool.map(load_url, url)
```

在这个例子中，我们使用多进程库中的 Pool 类创建了一个包含四个进程的进程池来执行我们的代码。然后使用 Pool 类的 map 方法，将输入数据映射到执行函数中的一个单独的进程中，以实现并发。

现在，我们有多个进程在处理我们的任务。但是，如果我们想让这些进程相互通信怎么办。例如，在之前的 URL 加载问题中，我们希望进程返回数据而不是在 stdout 上打印数据怎么办？答案在于使用管道，它为进程之间提供了双向通信的机制。

以下示例利用管道使 URL 加载器将从 URL 加载的数据发送回父进程：

```py
# url_load_pipe.py
from multiprocessing import Process, Pipe
import urllib.request

def load_url(url, pipe):
    url_handle = urllib.request.urlopen(url)
    url_data = url_handle.read()
    # The data returned by read() call is in the bytearray format. We need to
    # decode the data before we can print it.
    html_data = url_data.decode('utf-8')
    url_handle.close()
    pipe.send(html_data)

if __name__ == '__main__':
    url = 'http://www.w3c.org'
    parent_pipe, child_pipe = Pipe()
    loader_process = Process(target=load_url, args=(url, child_pipe))
    print("Spawning a new process to load the url")
    loader_process.start()
    print("Waiting for the spawned process to exit")
    html_data = parent_pipe.recv()
    print(html_data)
    loader_process.join()
    print("Exiting…")
```

在这个例子中，我们使用管道为父进程和子进程提供双向通信机制。当我们在代码的`__main__`部分调用`pipe`构造函数时，构造函数返回一对连接对象。每个连接对象都包含一个`send()`和一个`recv()`方法，用于在两端之间进行通信。使用`send()`方法从`child_pipe`发送的数据可以通过`parent_pipe`的`recv()`方法读取，反之亦然。

如果两个进程同时从管道的同一端读取或写入数据，可能会导致管道中的数据损坏。尽管，如果进程使用两个不同的端口或两个不同的管道，这就不成问题了。只有可以通过 pickle 序列化的数据才能通过管道发送。这是 Python 多进程模块的一个限制。

# 同步进程

与同步线程的操作一样重要的是，多进程上下文中的操作同步也很重要。由于多个进程可能访问相同的共享资源，它们对共享资源的访问需要进行序列化。为了帮助实现这一点，我们在这里也有锁的支持。

以下示例展示了如何在多进程模块的上下文中使用锁来同步多个进程的操作，通过获取与 URL 相关联的 HTML 并将其写入一个共同的本地文件：

```py
# url_loader_locks.pyfrom multiprocessing import Process, Lockimport urllib.requestdef load_url(url, lock): url_handle = urllib.request.urlopen(url) ...
```

# 总结

在这一章中，我们探讨了如何在 Python 应用程序中实现并发以及它的用途。在这个探索过程中，我们揭示了 Python 多线程模块的功能，以及如何使用它来生成多个线程来分配工作负载。然后，我们继续了解如何同步这些线程的操作，并了解了多线程应用程序可能出现的各种问题，如果不加以处理。然后，本章继续探讨了全局解释器锁（GIL）在某些 Python 实现中所施加的限制，以及它如何影响多线程工作负载。为了探索克服 GIL 所施加的限制的可能方法，我们继续了解了 Python 的多进程模块的使用，以及它如何帮助我们利用多处理器系统的全部潜力，通过使用多个进程而不是多个线程来实现并行处理。

# 问题

1.  Python 是通过哪些不同的方法实现并发应用程序的？

1.  如果已经获得锁的线程突然终止会发生什么？

1.  当应用程序接收到终止信号时，如何终止执行线程？

1.  如何在多个进程之间共享状态？

1.  有没有一种方法可以创建一个进程池，然后用于处理任务队列中的任务？
