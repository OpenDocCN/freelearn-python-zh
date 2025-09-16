# 7

# 并发和异步模式

在上一章中，我们介绍了架构设计模式：这些模式有助于解决复杂项目带来的某些独特挑战。接下来，我们需要讨论并发和异步模式，这是我们的解决方案目录中的另一个重要类别。

并发允许你的程序同时管理多个操作，充分利用现代处理器的全部能力。这就像一位厨师并行准备多道菜，每个步骤都精心编排，以确保所有菜肴同时准备好。另一方面，异步编程允许你的应用程序在等待操作完成时继续执行其他任务，例如将食物订单发送到厨房，并在订单准备好之前为其他顾客提供服务。

在本章中，我们将涵盖以下主要内容：

+   线程池模式

+   工作模型模式

+   未来与承诺模式

+   响应式编程中的观察者模式

+   其他并发和异步模式

# 技术要求

请参阅第一章中提出的各项要求。本章讨论的代码的附加技术要求如下：

+   Faker，使用 `pip` `install faker`

+   ReactiveX，使用 `pip` `install reactivex`

# 线程池模式

首先，了解什么是线程很重要。在计算机中，线程是操作系统可以调度的最小的处理单元。

线程就像可以在计算机上同时运行的执行轨迹，这使得许多活动可以同时进行，从而提高性能。它们在需要多任务处理的应用程序中尤为重要，例如处理多个 Web 请求或执行多个计算。

现在，让我们转向线程池模式本身。想象一下，你有很多任务要完成，但启动每个任务（在这种情况下，创建一个线程）在资源和时间上可能很昂贵。这就像每次有工作要做时都雇佣一个新员工，工作完成后又让他们离开。这个过程可能效率低下且成本高昂。通过维护一个或多个可以一次性创建并多次重用的工作线程集合，线程池模式有助于降低这种低效。当一个线程完成一个任务后，它不会终止，而是回到线程池中，等待另一个可以再次使用的任务。

什么是工作线程？

工作线程是特定任务或任务集的执行线程。工作线程用于将处理任务从主线程卸载，通过异步执行耗时或资源密集型任务来帮助保持应用程序的响应性。

除了更快的应用程序性能外，还有两个好处：

+   **降低开销**：通过重用线程，应用程序避免了为每个任务创建和销毁线程的开销

+   **更好的资源管理**：线程池限制了线程的数量，防止了由于创建过多线程而可能发生的资源耗尽

## 现实世界中的例子

在现实生活中，想象一家小餐馆，餐馆里有有限数量的厨师（线程）为顾客烹饪餐点（任务）。由于厨房空间（系统资源）的限制，餐馆一次只能容纳一定数量的厨师同时工作。当新的订单到来时，如果所有厨师都在忙碌，订单就会在队列中等待，直到有可用的厨师。这样，餐馆通过其可用的厨师有效地管理订单流，确保所有厨师都得到有效利用，而不会压垮厨房或需要为每个新订单雇佣更多员工。

在软件中也有很多例子：

+   网络服务器经常使用线程池来处理传入的客户请求。这允许它们同时为多个客户提供服务，而不需要为每个请求创建新的线程。

+   数据库使用线程池来管理连接，确保总有一池连接可供传入的查询使用。

+   任务调度器使用线程池来执行计划的任务，例如*cron*作业、备份或更新。

## 线程池模式的用例

有三个用例，其中线程池模式有助于：

+   **批量处理**：当你有许多可以并行执行的任务时，线程池可以将它们分配给其工作线程

+   **负载均衡**：线程池可以用来在工作线程之间均匀地分配工作负载，确保没有单个线程承担过多的工作

+   **资源优化**：通过重用线程，线程池最小化了系统资源的使用，例如内存和 CPU 时间

## 实现线程池模式

首先，让我们停下来分析一下对于给定的应用程序，线程池是如何工作的：

1.  当应用程序启动时，线程池创建一定数量的工作线程。这是初始化。线程的数量可以是固定的，也可以根据应用程序的需求动态调整。

1.  然后，我们有任务提交步骤。当有任务要执行时，它被提交到池中，而不是直接创建一个新的线程。任务可以是任何需要执行的内容，例如处理用户输入、处理网络请求或执行计算。

1.  下一步是任务执行。池将任务分配给一个可用的工作线程。如果所有线程都在忙碌，任务可能会在队列中等待，直到有线程变得可用。

1.  一旦线程完成了其任务，它不会死亡。相反，它返回到池中，准备好被分配新的任务。

对于我们的示例，让我们看看一些代码，其中我们创建了一个包含五个工作线程的线程池来处理一组任务。我们将使用`concurrent.futures`模块中的`ThreadPoolExecutor`类。

我们首先导入示例中需要的模块，如下所示：

```py
from concurrent.futures import ThreadPoolExecutor
import time
```

然后，我们创建一个函数来模拟任务，在这个例子中，我们简单地使用 `time.sleep(1)`：

```py
def task(n):
    print(f"Executing task {n}")
    time.sleep(1)
    print(f"Task {n} completed")
```

然后，我们使用一个 `ThreadPoolExecutor` 类的实例，该实例创建了一个最大工作者线程数为 5 的线程池，并向线程池提交了 10 个任务。因此，工作者线程会取走这些任务并执行它们。一旦工作者线程完成一个任务，它会从队列中取走另一个。代码如下：

```py
with ThreadPoolExecutor(max_workers=5) as executor:
    for i in range(10):
        executor.submit(task, i)
```

当运行示例代码时，使用 `ch07/thread_pool.py` Python 命令，你应该得到以下输出：

```py
Executing task 0
Executing task 1
Executing task 2
Executing task 3
Executing task 4
Task 0 completed
Task 4 completed
Task 3 completed
Task 1 completed
Executing task 6
Executing task 7
Executing task 8
Task 2 completed
Executing task 5
Executing task 9
Task 8 completed
Task 6 completed
Task 9 completed
Task 5 completed
Task 7 completed
```

我们看到，任务的完成顺序与提交顺序不同。这表明它们是使用线程池中可用的线程并发执行的。

# Worker Model 模式

Worker Model 模式的背后思想是将一个大型任务或多个任务划分为更小、更易于管理的单元，称为工作者，这些单元可以并行处理。这种并发和并行处理方法不仅加速了处理时间，还提高了应用程序的性能。

工作者可以是单个应用程序内的线程（正如我们在线程池模式中刚刚看到的），同一台机器上的独立进程，甚至是分布式系统中的不同机器。

Worker Model 模式的优点如下：

+   **可扩展性**：易于通过添加更多工作者进行扩展，这在分布式系统中特别有益，因为可以在多台机器上处理任务

+   **效率**：通过将任务分配给多个工作者，系统可以更好地利用可用的计算资源，并行处理任务

+   **灵活性**：Worker Model 模式可以适应各种处理策略，从简单的基于线程的工作者到跨越多服务器的复杂分布式系统

## 现实世界示例

考虑一个快递服务，包裹（任务）由一组快递员（工作者）递送。每个快递员从配送中心（任务队列）取走一个包裹并递送。快递员的数量可以根据需求变化；在繁忙时期可以增加更多快递员，而在较安静时可以减少。

在大数据处理中，Worker Model 模式通常被采用，其中每个工作者负责映射或减少数据的一部分。

在 RabbitMQ 或 Kafka 等系统中，Worker Model 模式用于并发处理队列中的消息。

我们还可以引用图像处理服务。需要同时处理多个图像的服务通常使用 Worker Model 模式在多个工作者之间分配负载。

## Worker Model 模式的用例

Worker Model 模式的用例之一是*数据转换*。当你有一个需要转换的大型数据集时，你可以将工作分配给多个工作者。

另一个用例是*任务并行化*。在任务彼此独立的应用程序中，Worker Model 模式可以非常有效。

第三个用例是 *分布式计算*，工作模型模式可以扩展到多台机器，使其适用于分布式计算环境。

## 实现工作模型模式

在讨论实现示例之前，让我们了解工作模型模式是如何工作的。工作模型模式涉及三个组件：工作者、任务队列和可选的调度器：

+   **工作者**：在这个模型中的主要角色。每个工作者可以独立于其他工作者执行任务的一部分。根据实现方式，工作者可能一次处理一个任务或同时处理多个任务。

+   **任务队列**：一个中央组件，其中存储着等待处理的任务。工作者通常从这个队列中拉取任务，确保任务在他们之间高效分配。队列充当了一个缓冲区，将任务提交与任务处理解耦。

+   **调度器**：在某些实现中，调度器组件根据可用性、负载或优先级将任务分配给工作进程。这有助于优化任务分配和资源利用。

现在我们来看一个并行执行函数的例子。

我们首先导入示例中需要的模块，如下所示：

```py
from multiprocessing import Process, Queue
import time
```

然后，我们创建一个 `worker()` 函数，我们将用它来运行任务。该函数接受一个参数 `task_queue` 对象，其中包含要执行的任务。代码如下：

```py
def worker(task_queue):
    while not task_queue.empty():
        task = task_queue.get()
        print(f"Worker {task} is processing")
        time.sleep(1)
        print(f"Worker {task} completed")
```

在 `main()` 函数中，我们首先创建一个任务队列，一个 `multiprocessing.Queue` 实例。然后，我们创建 10 个任务并将它们添加到队列中：

```py
def main():
    task_queue = Queue()
    for i in range(10):
        task_queue.put(i)
```

接着，我们创建了五个工作进程，使用 `multiprocessing.Process` 类，并启动它们。每个工作进程从队列中取一个任务来执行，然后取另一个任务，直到队列为空。然后，我们通过循环启动每个工作进程（使用 `p.start()`），这意味着相关的任务将并发执行。之后，我们创建另一个循环，在这个循环中使用进程的 `.join()` 方法，这样程序会等待这些进程完成工作。这部分代码如下：

```py
    processes = [
        Process(target=worker, args=(task_queue,))
        for _ in range(5)
    ]
    # Start the worker processes
    for p in processes:
        p.start()
    # Wait for all worker processes to finish
    for p in processes:
        p.join()
    print("All tasks completed.")
```

当运行示例代码时，使用 `ch07/worker_model.py` Python 命令，你应该得到以下输出，其中可以看到 5 个工作者以并发方式从任务队列中处理任务，直到所有 10 个任务完成：

```py
Worker 0 is processing
Worker 1 is processing
Worker 2 is processing
Worker 3 is processing
Worker 4 is processing
Worker 0 completed
Worker 5 is processing
Worker 1 completed
Worker 6 is processing
Worker 2 completed
Worker 7 is processing
Worker 3 completed
Worker 8 is processing
Worker 4 completed
Worker 9 is processing
Worker 5 completed
Worker 6 completed
Worker 7 completed
Worker 8 completed
Worker 9 completed
All tasks completed.
```

这展示了我们实现的工作模型模式。这种模式特别适用于任务独立且可以并行处理的情况。

# 未来和承诺模式

在异步编程范式下，Future 表示一个尚未知晓但最终会提供的值。当一个函数启动异步操作时，它不会阻塞直到操作完成并得到结果，而是立即返回一个 Future。这个 `Future` 对象充当了稍后可用的实际结果的占位符。

未来对象通常用于 I/O 操作、网络请求和其他耗时的异步任务。它们允许程序在等待操作完成的同时继续执行其他任务。这种特性被称为*非阻塞*。

一旦未来被实现，结果可以通过未来对象访问，通常是通过回调、轮询或阻塞，直到结果可用。

承诺是未来对象的可写、可控对应物。它代表异步操作的生产者端，最终将为相关的未来对象提供结果。当操作完成时，承诺通过一个值或错误被履行或拒绝，然后解决未来对象。

承诺可以被链式调用，允许一系列异步操作以清晰和简洁的方式执行。

通过允许程序在等待异步操作的同时继续执行，应用程序变得更加响应。另一个好处是*可组合性*：多个异步操作可以以干净和可管理的方式组合、排序或并行执行。

## 实际例子

从木匠那里订购定制餐桌提供了未来和承诺模式的实际例子。当你下单时，你会收到一个预计完成日期和设计草图（未来对象），代表木匠交付桌子的承诺。随着木匠的工作进行，这个承诺逐渐得到履行。完成餐桌的交付解决了未来对象，标志着木匠对你承诺的履行。

我们也可以在数字领域找到几个例子，如下所示：

+   **在线购物订单跟踪**：当你在线下单时，网站会立即为你提供订单确认和跟踪号码（未来对象）。随着你的订单被处理、发货和交付，状态更新（承诺的履行）会在跟踪页面上实时反映，最终确定最终的交付状态。

+   **食品配送应用**：通过食品配送应用下单后，你会收到一个预计的配送时间（未来对象）。应用会持续更新订单状态——从准备到取货和配送（承诺正在履行）——直到食物送到你家门口，此时未来对象因订单完成而得到解决。

+   **客户支持工单**：当你在一个网站上提交支持工单时，你会立即收到一个工单号码和一条消息，说明有人会回复你（未来对象）。幕后，支持团队根据优先级或接收顺序处理工单。一旦你的工单得到处理，你会收到回复，履行了你最初提交工单时做出的承诺。

## 未来和承诺模式的用例

至少有四种情况下推荐使用未来和承诺模式：

1.  **数据处理管道**：在数据处理管道中，数据通常需要经过多个阶段才能达到最终形式。通过用 Future 表示每个阶段，你可以有效地管理数据的异步流动。例如，一个阶段的输出可以作为下一个阶段的输入，但由于每个阶段都返回一个 Future，后续阶段不需要阻塞等待前一个阶段完成。

1.  **任务调度**：任务调度系统，如操作系统或高级应用程序中的系统，可以使用 Future 来表示计划在未来运行的任务。当任务被调度时，会返回一个 Future 来表示该任务的最终完成。这允许系统或应用程序跟踪任务的状态，而不会阻塞执行。

1.  **复杂数据库查询或事务**：异步执行数据库查询对于保持应用程序的响应性至关重要，尤其是在用户体验至关重要的 Web 应用程序中。通过使用 Future 来表示数据库操作的结果，应用程序可以发起一个查询并立即将控制权返回给用户界面或调用函数。Future 最终会解析为查询结果，允许应用程序更新 UI 或处理数据，而无需在等待数据库响应时冻结或变得无响应。

1.  **文件输入输出操作**：文件输入输出操作可能会显著影响应用程序的性能，尤其是在主线程上同步执行时。通过应用 Future 和 Promise 模式，文件输入输出操作被卸载到后台进程，并返回一个 Future 来表示操作的完成。这种方法允许应用程序在读取或写入文件的同时继续运行其他任务或响应用户交互。一旦 I/O 操作完成，Future 就会解析，应用程序可以处理或显示文件数据。

在这些用例中，Future 和 Promise 模式促进了异步操作，允许应用程序通过不阻塞主线程执行长时间运行的任务，保持响应性和高效性。

## 使用`concurrent.futures`实现 Future 和 Promise 模式

要了解如何实现 Future 和 Promise 模式，你必须首先理解其机制的三个步骤。接下来，让我们逐一分析：

1.  **初始化**：初始化步骤涉及使用一个函数启动异步操作，在该函数中，不是等待操作完成，而是函数立即返回一个“Future”对象。此对象充当稍后可用的结果的占位符。内部，异步函数创建一个“Promise”对象。此对象负责处理异步操作的结果。Promise 与 Future 相关联，这意味着 Promise 的状态（无论是已履行还是被拒绝）将直接影响 Future。

1.  **执行**：在执行步骤中，操作独立于主程序流程进行。这允许程序保持响应性并继续执行其他任务。一旦异步任务完成，其结果需要传达给启动操作的部分程序。操作的结果（无论是成功的结果还是错误）传递给先前创建的 Promise。

1.  **解析**：如果操作成功，Promise 将“履行”结果。如果操作失败，Promise 将“拒绝”错误。Promise 的履行或拒绝解决 Future。通常通过回调或后续函数使用结果，这是一段指定如何处理结果的代码。Future 提供机制（例如，方法或运算符）来指定这些回调，这些回调将在 Future 解决后执行。

在我们的示例中，我们使用`ThreadPoolExecutor`类的实例异步执行任务。submit 方法返回一个将最终包含计算结果的`Future`对象。我们首先导入所需的模块，如下所示：

```py
from concurrent.futures import ThreadPoolExecutor, as_completed
```

然后，我们定义一个用于执行的任务的函数：

```py
def square(x):
    return x * x
```

我们提交任务并获取`Future`对象，然后我们收集完成的 Future 对象。`as_completed`函数允许我们遍历完成的`Future`对象并检索它们的结果：

```py
with ThreadPoolExecutor() as executor:
    future1 = executor.submit(square, 2)
    future2 = executor.submit(square, 3)
    future3 = executor.submit(square, 4)
    futures = [future1, future2, future3]
    for future in as_completed(futures):
        print(f"Result: {future.result()}")
```

运行示例时，使用`ch07/future_and_promise/future.py` Python 命令，你应该得到以下输出：

```py
Result: 16
Result: 4
Result: 9
```

这展示了我们的实现。

## 实现 Future 和 Promise 模式 - 使用 asyncio

Python 的`asyncio`库提供了另一种使用异步编程执行任务的方法。它特别适用于 I/O 密集型任务。让我们看看使用这种技术的第二个示例。

什么是 asyncio？

`asyncio`库提供了对异步 I/O、事件循环、协程和其他并发相关任务的支撑。因此，使用`asyncio`，开发者可以编写高效处理 I/O 密集型操作的代码。

协程和 async/await

协程是一种特殊的函数，可以在某些点暂停和恢复其执行，同时允许其他协程在此期间运行。协程使用`async`关键字声明。此外，协程可以使用`await`关键字从其他协程中等待。

我们导入`asyncio`模块，它包含我们所需的一切：

```py
import asyncio
```

然后，我们创建一个用于计算并返回数字平方的函数。我们还想进行 I/O 密集型操作，因此我们使用`asyncio.sleep()`。请注意，在`asyncio`风格的编程中，这样的函数使用组合关键字`async def`定义——它是一个协程。`asyncio.sleep()`函数本身也是一个协程，因此我们确保在调用它时使用`await`关键字：

```py
async def square(x):
    # Simulate some IO-bound operation
    await asyncio.sleep(1)
    return x * x
```

然后，我们转向创建我们的`main()`函数。我们使用`asyncio.ensure_future()`函数来创建我们想要的`Future`对象，传递`square(x)`，其中`x`是要平方的数字。我们创建了三个`Future`对象，`future1`、`future2`和`future3`。然后，我们使用`asyncio.gather()`协程等待我们的 Future 完成并收集结果。`main()`函数的代码如下：

```py
async def main():
    fut1 = asyncio.ensure_future(square(2))
    fut2 = asyncio.ensure_future(square(3))
    fut3 = asyncio.ensure_future(square(4))
    results = await asyncio.gather(fut1, fut2, fut3)
    for result in results:
        print(f"Result: {result}")
```

在我们的代码文件末尾，我们有常见的`if __name__ == "__main__":`块。由于我们正在编写基于`asyncio`的代码，所以这里的新颖之处在于我们需要通过调用`asyncio.run(main())`来运行`asyncio`的事件循环：

```py
if __name__ == "__main__":
    asyncio.run(main())
```

要测试示例，运行`ch07/future_and_promise/async.py` Python 命令。你应该得到以下类似的输出：

```py
Result: 4
Result: 9
Result: 16
```

结果的顺序可能会根据运行程序的人和时间而变化。实际上，这是不可预测的。你可能已经注意到了我们之前示例中的类似行为。这通常是并发或异步代码的一般情况。

这个简单的例子表明，当我们需要高效处理 I/O 密集型任务（如网络爬取或 API 调用）时，`asyncio`是 Future 和 Promise 模式的合适选择。

# 响应式编程中的观察者模式

观察者模式（在*第五章*，*行为设计模式*中介绍）在通知一个对象或一组对象给定对象的状态发生变化时非常有用。这种传统的观察者模式允许我们响应某些对象变化事件。它为许多情况提供了一个很好的解决方案，但在我们必须处理许多事件，其中一些相互依赖的情况下，传统的方法可能会导致复杂、难以维护的代码。这就是另一个称为响应式编程的范式给我们提供了一个有趣的选择的地方。简单来说，响应式编程的概念是在保持我们的代码干净的同时，对许多事件（事件流）做出反应。

让我们关注 ReactiveX ([`reactivex.io`](http://reactivex.io))，它是响应式编程的一部分。ReactiveX 的核心是一个称为可观察的概念。根据其官方网站，ReactiveX 是关于提供异步编程 API，这些 API 被称为可观察流。这个概念被添加到我们已讨论的观察者理念中。

想象一个 Observable 就像一条河流，它将数据或事件流向一个 Observer。这个 Observable 依次发送项目。这些项目通过由不同步骤或操作组成的路径旅行，直到它们到达一个 Observer，该 Observer 接受或消费它们。

## 现实世界的例子

机场的航班信息显示系统在响应式编程中类似于一个 Observable。这样的系统会持续流式传输有关航班状态的更新，包括到达、出发、延误和取消。这个类比说明了观察者（旅客、航空公司员工和机场服务人员订阅以接收更新）如何订阅一个 Observable（航班显示系统）并对连续的更新流做出反应，从而允许对实时信息做出动态响应。

电子表格应用程序也可以被视为响应式编程的一个例子，基于其内部行为。在几乎所有电子表格应用程序中，交互式地更改工作表中的任何单元格都会导致立即重新评估直接或间接依赖于该单元格的所有公式，并更新显示以反映这些重新评估。

ReactiveX 思想在多种语言中得到实现，包括 Java（RxJava）、Python（RxPY）和 JavaScript（RxJS）。Angular 框架使用 RxJS 来实现 Observable 模式。

## 在响应式编程中使用观察者模式的用例

一个用例是集合管道的概念，由马丁·福勒在他的博客中讨论（[`martinfowler.com/articles/collection-pipeline`](https://martinfowler.com/articles/collection-pipeline)）。

集合管道，由马丁·福勒描述

集合管道是一种编程模式，其中你将一些计算组织为一系列操作，这些操作通过将一个集合作为一个操作的输出并传递给下一个操作来组合。

在处理数据时，我们还可以使用 Observable 对对象序列执行“映射和归约”或“按组”等操作。

最后，可以创建用于各种函数的 Observables，例如按钮事件、请求和 Twitter 动态。

## 在响应式编程中实现观察者模式

对于这个例子，我们决定构建一个包含（虚构）人名的列表的流（在`ch07/observer_rx/people.txt`文本文件中），以及基于它的 Observable。

注意

提供了一个包含虚构人名的文本文件作为本书示例文件的一部分（`ch07/observer_rx/people.txt`）。但每当需要时，可以使用辅助脚本（`ch07/observer_rx/peoplelist.py`）生成一个新的文件，这个脚本将在下一分钟介绍。

这样的名字列表示例看起来可能如下所示：

```py
Peter Brown, Gabriel Hunt, Gary Martinez, Heather Fernandez, Juan White, Alan George, Travis Davidson, David Adams, Christopher Morris, Brittany Thomas, Brian Allen, Stefanie Lutz, Craig West, William Phillips, Kirsten Michael, Daniel Brennan, Derrick West, Amy Vazquez, Carol Howard, Taylor Abbott,
```

回到我们的实现。我们首先导入所需的模块：

```py
from pathlib import Path
import reactivex as rx
from reactivex import operators as ops
```

我们定义了一个函数`firstnames_from_db()`，它从一个包含名字的文本文件（读取文件内容）返回一个 Observable，使用`flat_map()`、`filter()`和`map()`方法进行转换，并使用一个新的操作`group_by()`来从另一个序列中发射项目——文件中找到的第一个名字及其出现次数：

```py
def firstnames_from_db(path: Path):
    file = path.open()
    # collect and push stored people firstnames
    return rx.from_iterable(file).pipe(
        ops.flat_map(
            lambda content: rx.from_iterable(
                content.split(", ")
            )
        ),
        ops.filter(lambda name: name != ""),
        ops.map(lambda name: name.split()[0]),
        ops.group_by(lambda firstname: firstname),
        ops.flat_map(
            lambda grp: grp.pipe(
                ops.count(),
                ops.map(lambda ct: (grp.key, ct)),
            )
        ),
    )
```

然后，在`main()`函数中，我们定义了一个每 5 秒发射数据的 Observable，将其发射与从`firstnames_from_db(db_file)`返回的内容合并，在将`db_file`设置为包含人名的文本文件之后，如下所示：

```py
def main():
    db_path = Path(__file__).parent / Path("people.txt")
    # Emit data every 5 seconds
    rx.interval(5.0).pipe(
        ops.flat_map(lambda i: firstnames_from_db(db_path))
    ).subscribe(lambda val: print(str(val)))
    # Keep alive until user presses any key
    input("Starting... Press any key and ENTER, to quit\n")
```

这里是对示例的总结（完整的代码在`ch07/observer_rx/rx_peoplelist.py`文件中）：

1.  我们导入所需的模块和类。

1.  我们定义了一个`firstnames_from_db()`函数，它从一个文本文件返回一个 Observable，该文件是数据的来源。我们从该文件收集并推送存储的人名。

1.  最后，在`main()`函数中，我们定义了一个每 5 秒发射数据的 Observable，将其发射与调用`firstnames_from_db()`函数返回的内容合并。

要测试示例，请运行`ch07/observer_rx/rx_peoplelist.py` Python 命令。你应该得到以下输出（这里只显示了一部分）：

```py
Starting... Press any key and ENTER, to quit
('Peter', 1)
('Gabriel', 1)
('Gary', 1)
('Heather', 1)
('Juan', 1)
('Alan', 1)
('Travis', 1)
('David', 1)
('Christopher', 1)
('Brittany', 1)
('Brian', 1)
('Stefanie', 1)
('Craig', 1)
('William', 1)
('Kirsten', 1)
('Daniel', 1)
('Derrick', 1)
```

一旦你按下一个键并在键盘上按下*Enter*，发射就会中断，程序停止。

### 处理新的数据流

我们的测试是成功的，但从某种意义上说，它是静态的；数据流仅限于当前文本文件中的内容。我们现在需要生成多个数据流。我们可以使用一种基于第三方模块 Faker（[`pypi.org/project/Faker`](https://pypi.org/project/Faker)）的技术来生成文本文件中的假数据。生成数据的代码免费提供给你（在`ch07/observer_rx/peoplelist.py`文件中），如下所示：

```py
from faker import Faker
import sys
fake = Faker()
args = sys.argv[1:]
if len(args) == 1:
    output_filename = args[0]
    persons = []
    for _ in range(0, 20):
        p = {"firstname": fake.first_name(), "lastname": fake.last_name()}
        persons.append(p)
    persons = iter(persons)
    data = [f"{p['firstname']} {p['lastname']}" for p in persons]
    data = ", ".join(data) + ", "
    with open(output_filename, "a") as f:
        f.write(data)
else:
    print("You need to pass the output filepath!")
```

现在，让我们看看执行这两个程序（`ch07/observer_rx/peoplelist.py`和`ch07/observer_rx/rx_peoplelis.py`）会发生什么：

+   从一个命令行窗口或终端，你可以通过传递正确的文件路径到脚本中生成人名；你会执行以下命令：`python ch07/observer_rx/peoplelist.py ch07/observer_rx/people.txt`。

+   从第二个 shell 窗口，你可以通过执行`python ch07/observer_rx/rx_peoplelist.py`命令来运行实现 Observable 的程序。

那么，这两个命令的输出是什么？

创建了一个新的`people.txt`文件版本（其中包含用逗号分隔的随机名字），以替换现有文件。每次你重新运行该命令（`python ch07/observer_rx/peoplelist.py`），都会向文件中添加一组新的名字。

第二个命令给出的输出类似于第一次执行时的输出；区别在于现在发射的不是相同的数据集。现在，可以在源中生成新数据并发射。

# 其他并发和异步模式

开发者可能会使用一些其他的并发和异步模式。我们可以引用以下模式：

+   **演员模型**：一个处理并发计算的概念模型。它定义了一些规则，说明演员实例应该如何行为：一个演员可以做出局部决策，创建更多演员，发送更多消息，并确定如何对收到的下一个消息做出反应。

+   `asyncio`库）。

+   **消息传递**：用于并行计算、**面向对象编程**（**OOP**）和**进程间通信**（**IPC**），其中软件实体通过相互传递消息来通信和协调它们的行为。

+   **背压**：一种管理通过软件系统中的数据流并防止组件过载的机制。它允许系统通过向生产者发出信号以减慢速度，直到消费者能够赶上，从而优雅地处理过载。

每个模式都有其适用场景和权衡。知道它们存在很有趣，但我们无法讨论所有可用的模式和技巧。

# 摘要

在本章中，我们讨论了并发和异步模式，这些模式对于编写高效、响应性软件，能够同时处理多个任务非常有用。

线程池模式是并发编程中的一个强大工具，它提供了一种有效管理资源并提高应用程序性能的方法。它帮助我们提高应用程序性能，同时减少开销并更好地管理资源，因为线程池限制了线程的数量。

虽然线程池模式侧重于重用固定数量的线程来执行任务，但工作模型模式更多地关注于在可能可扩展和灵活的工作实体之间动态分配任务。这种模式特别适用于任务独立且可以并行处理的情况。

未来和承诺模式促进了异步操作，通过不阻塞主线程执行长时间运行的任务，使应用程序保持响应性和高效。

我们还讨论了反应式编程中的观察者模式。这个模式的核心思想是对数据流和事件做出反应，就像我们在自然界中看到的水流一样。在计算世界中，我们有大量的这种想法的例子。我们讨论了一个 ReactiveX 的例子，这为读者提供了一个如何接近这种编程范式并继续通过 ReactiveX 官方文档进行自己研究的介绍。

最后，我们提到了还有其他并发和异步模式。每个模式都有其适用场景和权衡，但我们无法在一本书中涵盖所有这些模式。

在下一章中，我们将讨论性能设计模式。
