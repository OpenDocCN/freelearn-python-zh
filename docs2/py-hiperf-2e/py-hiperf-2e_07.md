# 第七章：并行处理

通过使用多个核心的并行处理，您可以在给定的时间框架内增加程序可以完成的计算量，而无需更快的处理器。主要思想是将问题划分为独立的子单元，并使用多个核心并行解决这些子单元。

并行处理对于解决大规模问题是必要的。公司每天产生大量数据，需要存储在多台计算机上并进行分析。科学家和工程师在超级计算机上运行并行代码以模拟大型系统。

并行处理允许您利用多核 CPU 以及与高度并行问题配合得非常好的 GPU。在本章中，我们将涵盖以下主题：

+   并行处理基础简介

+   使用 `multiprocessing` Python 库并行化简单问题的说明

+   使用简单的 `ProcessPoolExecutor` 接口

+   在 Cython 和 OpenMP 的帮助下使用多线程并行化我们的程序

+   使用 Theano 和 Tensorflow 自动实现并行性

+   使用 Theano、Tensorflow 和 Numba 在 GPU 上执行代码

# 并行编程简介

为了并行化一个程序，有必要将问题划分为可以独立（或几乎独立）运行的子单元。

当子单元完全独立于彼此时，该问题被称为**令人尴尬的并行**。数组上的元素级操作是一个典型的例子——操作只需要知道它当前处理的元素。另一个例子是我们的粒子模拟器。由于没有相互作用，每个粒子可以独立于其他粒子进化。令人尴尬的并行问题很容易实现，并且在并行架构上表现良好。

其他问题可能被划分为子单元，但必须共享一些数据以执行它们的计算。在这些情况下，实现过程不太直接，并且由于通信成本可能导致性能问题。

我们将通过一个例子来说明这个概念。想象一下，你有一个粒子模拟器，但这次粒子在一定的距离内吸引其他粒子（如图所示）。为了并行化这个问题，我们将模拟区域划分为区域，并将每个区域分配给不同的处理器。如果我们一次进化系统的一步，一些粒子将与相邻区域的粒子相互作用。为了执行下一次迭代，需要与相邻区域的新粒子位置进行通信。

![图片](img/B06440_07CHPNO_01.png)

进程之间的通信成本高昂，可能会严重阻碍并行程序的性能。在并行程序中处理数据通信存在两种主要方式：

+   **共享内存**

+   **分布式内存**

在共享内存中，子单元可以访问相同的内存空间。这种方法的优点是，你不需要显式处理通信，因为从共享内存中写入或读取就足够了。然而，当多个进程同时尝试访问和更改相同的内存位置时，就会出现问题。应谨慎使用同步技术来避免此类冲突。

在分布式内存模型中，每个进程与其他进程完全隔离，并拥有自己的内存空间。在这种情况下，进程间的通信是显式处理的。与共享内存相比，通信开销通常更昂贵，因为数据可能需要通过网络接口传输。

实现共享内存模型中的并行性的一个常见方法是**线程**。线程是从进程派生出来的独立子任务，并共享资源，如内存。这一概念在以下图中进一步说明。线程产生多个执行上下文并共享相同的内存空间，而进程提供多个具有自己内存空间和需要显式处理通信的执行上下文。

![图片](img/B06440_07CHPNO_02.png)

Python 可以创建和处理线程，但它们不能用来提高性能；由于 Python 解释器的设计，一次只能允许一个 Python 指令运行--这种机制被称为**全局解释器锁**（**GIL**）。发生的情况是，每次线程执行 Python 语句时，线程都会获取一个锁，当执行完成后，释放相同的锁。由于锁一次只能被一个线程获取，因此当某个线程持有锁时，其他线程将无法执行 Python 语句。

尽管 GIL 阻止了 Python 指令的并行执行，但在可以释放锁的情况（如耗时的 I/O 操作或 C 扩展）中，线程仍然可以用来提供并发性。

为什么不移除全局解释器锁（GIL）呢？在过去的几年里，已经尝试过多次移除 GIL，包括最近的 gilectomy 实验。首先，移除 GIL 并不是一件容易的事情，它需要修改大多数 Python 数据结构。此外，这种细粒度的锁定可能会带来高昂的成本，并可能在单线程程序中引入显著的性能损失。尽管如此，一些 Python 实现（如 Jython 和 IronPython）并不使用 GIL。

可以通过使用进程而不是线程来完全绕过 GIL。进程不共享相同的内存区域，彼此独立--每个进程都有自己的解释器。进程有一些缺点：启动一个新的进程通常比启动一个新的线程慢，它们消耗更多的内存，并且进程间通信可能较慢。另一方面，进程仍然非常灵活，并且随着它们可以在多台机器上分布而具有更好的可扩展性。

# 图形处理单元

图形处理单元是专为计算机图形应用设计的特殊处理器。这些应用通常需要处理 3D 场景的几何形状，并将像素数组输出到屏幕上。GPU 执行的操作涉及浮点数的数组和矩阵运算。

GPU 被设计用来非常高效地运行与图形相关的操作，它们通过采用高度并行的架构来实现这一点。与 CPU 相比，GPU 拥有更多的（数千个）小型处理单元。GPU 旨在每秒产生大约 60 帧的数据，这比 CPU 的典型响应时间慢得多，而 CPU 具有更高的时钟速度。

GPU 具有与标准 CPU 非常不同的架构，并且专门用于计算浮点运算。因此，为了为 GPU 编译程序，有必要利用特殊的编程平台，如 CUDA 和 OpenCL。

**统一计算设备架构**（**CUDA**）是 NVIDIA 的专有技术。它提供了一个可以从其他语言访问的 API。CUDA 提供了 NVCC 工具，可以用来编译用类似于 C（CUDA C）的语言编写的 GPU 程序，以及实现高度优化的数学例程的众多库。

**OpenCL**是一种开放技术，具有编写可编译为多种目标设备（多个厂商的 CPU 和 GPU）的并行程序的能力，对于非 NVIDIA 设备来说是一个不错的选择。

GPU 编程在纸面上听起来很美妙。然而，不要急着丢弃你的 CPU。GPU 编程很复杂，并且只有特定的用例能从 GPU 架构中受益。程序员需要意识到在主内存之间进行数据传输所产生的成本，以及如何实现算法以利用 GPU 架构。

通常，GPU 在增加每单位时间内可以执行的操作数量（也称为**吞吐量**）方面表现很好；然而，它们需要更多的时间来准备数据以进行处理。相比之下，CPU 在从头开始生成单个结果方面要快得多（也称为**延迟**）。

对于正确的问题，GPU 提供了极端（10 到 100 倍）的加速。因此，它们通常构成了一个非常经济的（同样的加速将需要数百个 CPU）解决方案，用于提高数值密集型应用的性能。我们将在*自动并行性*部分说明如何在 GPU 上执行一些算法。

# 使用多个进程

标准的`multiprocessing`模块可以通过启动几个进程来快速并行化简单任务，同时避免 GIL 问题。它的接口易于使用，包括处理任务提交和同步的几个实用工具。

# Process 和 Pool 类

您可以通过继承 `multiprocessing.Process` 来创建一个独立运行的进程。您可以扩展 `__init__` 方法来初始化资源，并且可以通过实现 `Process.run` 方法来编写将在子进程中执行的部分代码。在以下代码中，我们定义了一个 `Process` 类，它将等待一秒钟并打印其分配的 `id`：

```py
    import multiprocessing 
    import time 

    class Process(multiprocessing.Process): 
        def __init__(self, id): 
            super(Process, self).__init__() 
            self.id = id 

        def run(self): 
            time.sleep(1) 
            print("I'm the process with id: {}".format(self.id))

```

要启动进程，我们必须实例化 `Process` 类并调用 `Process.start` 方法。请注意，您不能直接调用 `Process.run`；对 `Process.start` 的调用将创建一个新的进程，然后该进程将调用 `Process.run` 方法。我们可以在前面的代码片段末尾添加以下行来创建并启动新进程：

```py
    if __name__ == '__main__': 
        p = Process(0) 
        p.start()

```

在 `Process.start` 之后的指令将立即执行，而不需要等待 `p` 进程完成。要等待任务完成，您可以使用 `Process.join` 方法，如下所示：

```py
    if __name__ == '__main__': 
       p = Process(0) 
       p.start() 
       p.join()

```

我们可以启动四个不同的进程，它们将以相同的方式并行运行。在串行程序中，所需的总时间将是四秒钟。由于执行是并发的，因此结果的时间将是秒。在以下代码中，我们创建了四个将并发执行的进程：

```py
    if __name__ == '__main__': 
        processes = Process(1), Process(2), Process(3), Process(4) 
        [p.start() for p in processes]

```

注意，并行进程的执行顺序是不可预测的，最终取决于操作系统如何调度它们的执行。您可以通过多次执行程序来验证此行为；运行之间的顺序可能会不同。

`multiprocessing` 模块提供了一个方便的接口，使得将任务分配和分配给 `multiprocessing.Pool` 类中驻留的一组进程变得容易。

`multiprocessing.Pool` 类会启动一组进程——称为 **工作者**——并允许我们通过 `apply`/`apply_async` 和 `map`/`map_async` 方法提交任务。

`Pool.map` 方法将函数应用于列表中的每个元素，并返回结果列表。其用法与内置的（串行）`map` 相当。

要使用并行映射，您首先需要初始化一个 `multiprocessing.Pool` 对象。它将工作者数量作为其第一个参数；如果没有提供，则该数字将与系统中的核心数相等。您可以通过以下方式初始化一个 `multiprocessing.Pool` 对象：

```py
    pool = multiprocessing.Pool() 
    pool = multiprocessing.Pool(processes=4)

```

让我们看看 `pool.map` 的实际应用。如果您有一个计算数字平方的函数，您可以通过调用 `Pool.map` 并传递函数和输入列表作为参数来将该函数映射到列表上，如下所示：

```py
    def square(x): 
        return x * x 

    inputs = [0, 1, 2, 3, 4] 
    outputs = pool.map(square, inputs)

```

`Pool.map_async` 函数与 `Pool.map` 类似，但返回的是 `AsyncResult` 对象而不是实际的结果。当我们调用 `Pool.map` 时，主程序的执行会停止，直到所有工作进程完成结果的处理。使用 `map_async`，会立即返回 `AsyncResult` 对象而不阻塞主程序，计算在后台进行。然后我们可以使用 `AsyncResult.get` 方法在任何时候检索结果，如下面的代码所示：

```py
    outputs_async = pool.map_async(square, inputs) 
    outputs = outputs_async.get()

```

`Pool.apply_async` 将一个由单个函数组成的任务分配给一个工作进程。它接受函数及其参数，并返回一个 `AsyncResult` 对象。我们可以使用 `apply_async` 获得类似于 `map` 的效果，如下所示：

```py
    results_async = [pool.apply_async(square, i) for i in range(100))] 
    results = [r.get() for r in results_async]

```

# 执行器接口

从版本 3.2 开始，可以使用 `concurrent.futures` 模块中提供的 `Executor` 接口并行执行 Python 代码。我们在上一章中已经看到了 `Executor` 接口的使用，当时我们使用 `ThreadPoolExecutor` 来并发执行多个任务。在本小节中，我们将演示 `ProcessPoolExecutor` 类的使用。

`ProcessPoolExecutor` 提供了一个非常简洁的接口，至少与功能更丰富的 `multiprocessing.Pool` 相比是这样的。可以通过传递 `max_workers` 参数（默认情况下，`max_workers` 将是可用的 CPU 核心数）来实例化 `ProcessPoolExecutor`，类似于 `ThreadPoolExecutor`。`ProcessPoolExecutor` 可用的主要方法有 `submit` 和 `map`。

`submit` 方法将接受一个函数并返回一个 `Future`（见最后一章），该 `Future` 将跟踪提交函数的执行。方法 `map` 与 `Pool.map` 函数类似，但返回的是一个迭代器而不是列表：

```py
    from concurrent.futures import ProcessPoolExecutor

    executor = ProcessPoolExecutor(max_workers=4)
    fut = executor.submit(square, 2)
    # Result:
    # <Future at 0x7f5b5c030940 state=running>

    result = executor.map(square, [0, 1, 2, 3, 4])
    list(result)
    # Result:
    # [0, 1, 4, 9, 16]

```

要从一个或多个 `Future` 实例中提取结果，可以使用 `concurrent.futures.wait` 和 `concurrent.futures.as_completed` 函数。`wait` 函数接受一个 `future` 列表，并将阻塞程序的执行，直到所有 futures 完成它们的执行。然后可以使用 `Future.result` 方法提取结果。`as_completed` 函数也接受一个函数，但会返回一个结果迭代器：

```py
    from concurrent.futures import wait, as_completed

    fut1 = executor.submit(square, 2)
    fut2 = executor.submit(square, 3)
    wait([fut1, fut2])
    # Then you can extract the results using fut1.result() and fut2.result()

    results = as_completed([fut1, fut2])
    list(results)
    # Result:
    # [4, 9]

```

或者，你可以使用 `asyncio.run_in_executor` 函数生成 futures，并使用 `asyncio` 库提供的所有工具和语法来操作结果，这样你就可以同时实现并发和并行。

# π 的蒙特卡洛近似

作为例子，我们将实现一个典型的、显而易见的并行程序——**蒙特卡洛逼近π**。想象一下，我们有一个边长为 2 个单位的正方形；其面积为 4 个单位。现在，我们在正方形内画一个半径为 1 个单位的圆；圆的面积将是 *π * r²*。通过将 *r* 的值代入前面的方程，我们得到圆面积的数值为 *π * (1)² = π*。你可以参考以下图表来获取图形表示。

如果我们在这个图形上随机射击很多点，一些点将落在圆内，我们将它们称为**击中**，而剩余的点，**未击中**，将位于圆外。圆的面积将与击中的数量成正比，而正方形的面积将与射击的总数成正比。为了得到 *π* 的值，只需将圆的面积（等于 *π*）除以正方形的面积（等于 4）即可：

```py
    hits/total = area_circle/area_square = pi/4 
    pi = 4 * hits/total

```

![](img/image_07_003.png)

我们在项目中将采用以下策略：

+   在范围（**-1**，**1**）内生成大量的均匀随机（*x*，*y*）数字

+   通过检查是否 *x**2 + y**2* <= *1* 来测试这些数字是否位于圆内

编写并行程序的第一步是编写一个串行版本并验证其是否工作。在实际场景中，你希望将并行化作为优化过程的最后一步。首先，我们需要识别出慢速部分，其次，并行化耗时且只能提供与处理器数量相等的最大加速。串行程序的实现如下：

```py
    import random 

    samples = 1000000 
    hits = 0 

    for i in range(samples): 
        x = random.uniform(-1.0, 1.0) 
        y = random.uniform(-1.0, 1.0) 

        if x**2 + y**2 <= 1: 
            hits += 1 

    pi = 4.0 * hits/samples

```

随着样本数量的增加，我们的近似精度将提高。你可以注意到每个循环迭代都是独立的——这个问题是显而易见的并行问题。

为了并行化这段代码，我们可以编写一个名为 `sample` 的函数，它对应于一次击中-未击中的检查。如果样本击中圆，函数将返回 `1`；否则，它将返回 `0`。通过多次运行 `sample` 并汇总结果，我们将得到总的击中次数。我们可以使用 `apply_async` 在多个处理器上运行 `sample` 并以以下方式获取结果：

```py
    def sample(): 
        x = random.uniform(-1.0, 1.0) 
        y = random.uniform(-1.0, 1.0) 

        if x**2 + y**2 <= 1: 
            return 1 
        else: 
            return 0 

    pool = multiprocessing.Pool() 
    results_async = [pool.apply_async(sample) for i in range(samples)] 
    hits = sum(r.get() for r in results_async)

```

我们可以将这两个版本包裹在 `pi_serial` 和 `pi_apply_async` 函数中（你可以在 `pi.py` 文件中找到它们的实现）并比较执行速度，如下所示：

```py
$ time python -c 'import pi; pi.pi_serial()'
real    0m0.734s
user    0m0.731s
sys     0m0.004s
$ time python -c 'import pi; pi.pi_apply_async()'
real    1m36.989s
user    1m55.984s
sys     0m50.386

```

如前所述的基准测试所示，我们的第一个并行版本实际上削弱了我们的代码。原因是实际计算所需的时间与发送和分配任务给工作者的开销相比很小。

为了解决这个问题，我们必须使开销与计算时间相比可以忽略不计。例如，我们可以要求每个工作进程一次处理多个样本，从而减少任务通信开销。我们可以编写一个`sample_multiple`函数，它处理多个命中并修改我们的并行版本，通过将问题分成 10 份；更密集的任务如下面的代码所示：

```py
    def sample_multiple(samples_partial): 
        return sum(sample() for i in range(samples_partial)) 

    n_tasks = 10 
    chunk_size = samples/n_tasks 
    pool = multiprocessing.Pool() 
    results_async = [pool.apply_async(sample_multiple, chunk_size) 
                     for i in range(n_tasks)] 
    hits = sum(r.get() for r in results_async)

```

我们可以将这个功能封装在一个名为`pi_apply_async_chunked`的函数中，并按如下方式运行：

```py
$ time python -c 'import pi; pi.pi_apply_async_chunked()'
real    0m0.325s
user    0m0.816s
sys     0m0.008s

```

结果要好得多；我们的程序速度提高了不止一倍。你还可以注意到`user`指标大于`real`；总 CPU 时间大于总时间，因为同时有多个 CPU 在工作。如果你增加样本数量，你会注意到通信与计算的比率降低，从而提供更好的加速。

当处理令人尴尬的并行问题时，一切都很简单。然而，有时你必须在进程之间共享数据。

# 同步和锁

即使`multiprocessing`使用进程（它们有自己独立的内存），它也允许你定义某些变量和数组作为共享内存。你可以使用`multiprocessing.Value`定义一个共享变量，将数据类型作为字符串传递（`i`表示整数，`d`表示双精度，`f`表示浮点数等）。你可以通过`value`属性更新变量的内容，如下面的代码片段所示：

```py
    shared_variable = multiprocessing.Value('f') 
    shared_variable.value = 0

```

当使用共享内存时，你应该意识到并发访问。想象一下，你有一个共享的整数变量，每个进程多次增加其值。你可以定义一个进程类如下：

```py
    class Process(multiprocessing.Process): 

        def __init__(self, counter): 
            super(Process, self).__init__() 
            self.counter = counter 

        def run(self): 
            for i in range(1000): 
                self.counter.value += 1

```

你可以在主程序中初始化共享变量，并将其传递给`4`个进程，如下面的代码所示：

```py
    def main(): 
        counter = multiprocessing.Value('i', lock=True) 
        counter.value = 0 

        processes = [Process(counter) for i in range(4)] 
        [p.start() for p in processes] 
        [p.join() for p in processes] # processes are done 
        print(counter.value)

```

如果你运行这个程序（代码目录中的`shared.py`），你会注意到`counter`的最终值不是 4000，而是有随机值（在我的机器上，它们在 2000 到 2500 之间）。如果我们假设算术是正确的，我们可以得出结论，并行化存在问题。

发生的情况是多个进程同时尝试访问相同的共享变量。这种情况最好通过查看以下图表来解释。在串行执行中，第一个进程读取（数字`0`），增加它，并写入新值（`1`）；第二个进程读取新值（`1`），增加它，并再次写入（`2`）。

在并行执行中，两个进程同时读取（`0`），增加它，并写入值（`1`），导致错误答案：

![图片](img/B06440_07CHPNO_04.png)

为了解决这个问题，我们需要同步对这个变量的访问，以确保一次只有一个进程可以访问、增加和写入共享变量的值。这个功能由 `multiprocessing.Lock` 类提供。锁可以通过 `acquire` 方法或 `release` 以及将锁用作上下文管理器来获取和释放。由于锁一次只能被一个进程获取，这种方法防止了多个进程同时执行受保护的代码段。

我们可以定义一个全局锁，并使用它作为上下文管理器来限制对计数器的访问，如下代码片段所示：

```py
    lock = multiprocessing.Lock() 

    class Process(multiprocessing.Process): 

        def __init__(self, counter): 
            super(Process, self).__init__() 
            self.counter = counter 

        def run(self): 
            for i in range(1000): 
                with lock: # acquire the lock 
                    self.counter.value += 1 
                # release the lock

```

同步原语，如锁，对于解决许多问题是必不可少的，但它们应该保持最小，以提高程序的性能。

`multiprocessing` 模块还包括其他通信和同步工具；您可以参考官方文档[`docs.python.org/3/library/multiprocessing.html`](http://docs.python.org/3/library/multiprocessing.html)以获取完整参考。

# Parallel Cython with OpenMP

Cython 提供了一个方便的接口，通过 **OpenMP** 执行共享内存并行处理。这允许您直接在 Cython 中编写非常高效的并行代码，而无需创建 C 包装器。

OpenMP 是一个规范和一个 API，旨在编写多线程、并行程序。OpenMP 规范包括一系列 C 预处理器指令，用于管理线程，并提供通信模式、负载均衡和其他同步功能。几个 C/C++ 和 Fortran 编译器（包括 GCC）实现了 OpenMP API。

我们可以通过一个小示例引入 Cython 并行功能。Cython 在 `cython.parallel` 模块中提供了一个基于 OpenMP 的简单 API。实现并行化的最简单方法是使用 `prange`，这是一个自动在多个线程中分配循环操作的构造。

首先，我们可以在 `hello_parallel.pyx` 文件中编写一个程序的串行版本，该程序计算 NumPy 数组中每个元素的平方。我们定义了一个函数 `square_serial`，它接受一个缓冲区作为输入，并将输入数组元素的平方填充到输出数组中；`square_serial` 如下代码片段所示：

```py
    import numpy as np 

    def square_serial(double[:] inp): 
        cdef int i, size 
        cdef double[:] out 
        size = inp.shape[0] 
        out_np = np.empty(size, 'double') 
        out = out_np 

        for i in range(size): 
            out[i] = inp[i]*inp[i] 

        return out_np

```

实现对数组元素进行循环的并行版本涉及用 `prange` 替换 `range` 调用。有一个注意事项——要使用 `prange`，循环体必须是解释器无关的。如前所述，我们需要释放 GIL，由于解释器调用通常获取 GIL，因此需要避免它们以利用线程。

在 Cython 中，您可以使用 `nogil` 上下文释放 GIL，如下所示：

```py
    with nogil: 
        for i in prange(size): 
            out[i] = inp[i]*inp[i]

```

或者，您可以使用 `prange` 的 `nogil=True` 选项，这将自动将循环体包装在 `nogil` 块中：

```py
    for i in prange(size, nogil=True): 
        out[i] = inp[i]*inp[i]

```

尝试在 `prange` 块中调用 Python 代码将产生错误。禁止的操作包括函数调用、对象初始化等。为了在 `prange` 块中启用此类操作（你可能想这样做以进行调试），你必须使用 `with gil` 语句重新启用 GIL：

```py
    for i in prange(size, nogil=True): 
        out[i] = inp[i]*inp[i] 
        with gil:   
            x = 0 # Python assignment

```

现在，我们可以通过将其编译为 Python 扩展模块来测试我们的代码。为了启用 OpenMP 支持，需要更改 `setup.py` 文件，使其包含编译选项 `-fopenmp`。这可以通过在 `distutils` 中使用 `distutils.extension.Extension` 类并传递给 `cythonize` 来实现。完整的 `setup.py` 文件如下：

```py
    from distutils.core import setup 
    from distutils.extension import Extension 
    from Cython.Build import cythonize 

    hello_parallel = Extension('hello_parallel', 
                               ['hello_parallel.pyx'], 
                                extra_compile_args=['-fopenmp'], 
                                extra_link_args=['-fopenmp']) 

    setup( 
       name='Hello', 
       ext_modules = cythonize(['cevolve.pyx', hello_parallel]), 
    )

```

使用 `prange`，我们可以轻松地将我们的 `ParticleSimulator` 的 Cython 版本并行化。以下代码包含 `cevolve.pyx` Cython 模块的 `c_evolve` 函数，该函数在 第四章，*C Performance with Cython* 中编写：

```py
    def c_evolve(double[:, :] r_i,double[:] ang_speed_i, 
                 double timestep,int nsteps): 

        # cdef declarations 

        for i in range(nsteps): 
            for j in range(nparticles): 
                # loop body

```

首先，我们将反转循环的顺序，使得最外层的循环将并行执行（每个迭代都是独立的）。由于粒子之间没有相互作用，我们可以安全地改变迭代的顺序，如下面的代码片段所示：

```py
        for j in range(nparticles): 
            for i in range(nsteps): 

                # loop body

```

接下来，我们将用 `prange` 替换外层循环的 `range` 调用，并移除获取 GIL 的调用。由于我们的代码已经通过静态类型进行了增强，因此可以安全地应用 `nogil` 选项，如下所示：

```py
    for j in prange(nparticles, nogil=True)

```

现在，我们可以通过将它们包装在基准函数中来比较这些函数，以评估任何性能改进：

```py
    In [3]: %timeit benchmark(10000, 'openmp') # Running on 4 processors
    1 loops, best of 3: 599 ms per loop 
    In [4]: %timeit benchmark(10000, 'cython') 
    1 loops, best of 3: 1.35 s per loop

```

有趣的是，我们通过编写 `prange` 的并行版本实现了 2 倍的速度提升。

# 自动并行化

如我们之前提到的，由于 GIL，普通的 Python 程序在实现线程并行化方面有困难。到目前为止，我们通过使用单独的进程来解决这个问题；然而，启动进程比启动线程花费更多的时间和内存。

我们还看到，绕过 Python 环境使我们能够在已经很快的 Cython 代码上实现 2 倍的速度提升。这种策略使我们能够实现轻量级并行化，但需要单独的编译步骤。在本节中，我们将进一步探讨这种策略，使用能够自动将我们的代码转换为并行版本的专用库，以实现高效的执行。

实现自动并行化的包示例包括现在熟悉的即时编译器 `numexpr` 和 Numba。其他包已经开发出来，用于自动优化和并行化数组密集型表达式和矩阵密集型表达式，这在特定的数值和机器学习应用中至关重要。

**Theano** 是一个项目，它允许你在数组（更一般地说，*张量*）上定义数学表达式，并将它们编译为快速语言，如 C 或 C++。Theano 实现的大多数操作都是可并行化的，并且可以在 CPU 和 GPU 上运行。

**Tensorflow** 是另一个库，与 Theano 类似，旨在表达密集型数学表达式，但它不是将表达式转换为专门的 C 代码，而是在高效的 C++ 引擎上执行操作。

当手头的问题可以用矩阵和逐元素操作的链式表达来表示时（例如 *神经网络*），Theano 和 Tensorflow 都是非常理想的。

# 开始使用 Theano

Theano 在某种程度上类似于编译器，但增加了能够表达、操作和优化数学表达式以及能够在 CPU 和 GPU 上运行代码的额外好处。自 2010 年以来，Theano 在版本更新中不断改进，并被其他几个 Python 项目采用，作为在运行时自动生成高效计算模型的一种方式。

在 Theano 中，你首先通过指定变量和转换来使用纯 Python API *定义* 你想要运行的函数。然后，这个规范将被编译成机器代码以执行。

作为第一个例子，让我们看看如何实现一个计算数字平方的函数。输入将由一个标量变量 `a` 表示，然后我们将对其进行转换以获得其平方，表示为 `a_sq`。在下面的代码中，我们将使用 `T.scalar` 函数来定义变量，并使用正常的 `**` 运算符来获取一个新的变量：

```py
    import theano.tensor as T
    import theano as th
    a = T.scalar('a')
    a_sq = a ** 2
    print(a_sq)
    # Output:
    # Elemwise{pow,no_inplace}.0

```

如你所见，没有计算特定的值，我们应用的是纯符号转换。为了使用这个转换，我们需要生成一个函数。要编译一个函数，你可以使用 `th.function` 工具，它将输入变量的列表作为其第一个参数，输出转换（在我们的情况下是 `a_sq`）作为其第二个参数：

```py
    compute_square = th.function([a], a_sq)

```

Theano 将花费一些时间将表达式转换为高效的 C 代码并编译它，所有这些都是在后台完成的！`th.function` 的返回值将是一个可用的 Python 函数，其用法在下一行代码中演示：

```py
    compute_square(2)
    4.0

```

令人意外的是，`compute_square` 函数正确地返回了输入值的平方。然而，请注意，返回类型不是整数（与输入类型相同），而是一个浮点数。这是因为 Theano 的默认变量类型是 `float64`。你可以通过检查 `a` 变量的 `dtype` 属性来验证这一点：

```py
    a.dtype
    # Result: 
    # float64

```

与我们之前看到的 Numba 相比，Theano 的行为非常不同。Theano 不编译通用的 Python 代码，也不进行任何类型推断；定义 Theano 函数需要更精确地指定涉及的类型。

Theano 的真正力量来自于其对数组表达式的支持。使用 `T.vector` 函数可以定义一维向量；返回的变量支持与 NumPy 数组相同的广播操作语义。例如，我们可以取两个向量并计算它们平方的逐元素和，如下所示：

```py
    a = T.vector('a')
    b = T.vector('b')
    ab_sq = a**2 + b**2
    compute_square = th.function([a, b], ab_sq)

    compute_square([0, 1, 2], [3, 4, 5])
    # Result:
    # array([  9.,  17.,  29.])

```

想法再次是使用 Theano API 作为迷你语言来组合各种 Numpy 数组表达式，这些表达式将被编译成高效的机器代码。

Theano 的一个卖点是其执行算术简化以及自动梯度计算的能力。有关更多信息，请参阅官方文档（[`deeplearning.net/software/theano/introduction.html`](http://deeplearning.net/software/theano/introduction.html)）。

为了演示 Theano 在熟悉的使用场景中的功能，我们可以再次实现我们的π的并行计算。我们的函数将接受两个随机坐标的集合作为输入，并返回π的估计值。输入的随机数将被定义为名为`x`和`y`的向量，我们可以使用标准元素级操作来测试它们在圆内的位置，这些操作我们将存储在`hit_test`变量中：

```py
    x = T.vector('x')
    y = T.vector('y')

    hit_test = x ** 2 + y ** 2 < 1

```

在这个阶段，我们需要计算`hit_test`中`True`元素的数量，这可以通过取其和来完成（它将被隐式转换为整数）。为了获得π的估计值，我们最终需要计算击中次数与总试验次数的比率。计算过程在下面的代码块中展示：

```py
    hits = hit_test.sum()
    total = x.shape[0]
    pi_est = 4 * hits/total

```

我们可以使用`th.function`和`timeit`模块来基准测试 Theano 实现的执行。在我们的测试中，我们将传递两个大小为 30,000 的数组，并使用`timeit.timeit`实用程序多次执行`calculate_pi`函数：

```py
    calculate_pi = th.function([x, y], pi_est)

    x_val = np.random.uniform(-1, 1, 30000)
    y_val = np.random.uniform(-1, 1, 30000)

    import timeit
    res = timeit.timeit("calculate_pi(x_val, y_val)", 
    "from __main__ import x_val, y_val, calculate_pi", number=100000)
    print(res)
    # Output:
    # 10.905971487998613

```

此函数的串行执行大约需要 10 秒。Theano 能够通过实现元素级和矩阵操作使用专门的包（如 OpenMP 和**基本线性代数子程序**（**BLAS**）线性代数例程）来自动并行化代码。可以通过配置选项启用并行执行。

在 Theano 中，你可以在导入时通过修改`theano.config`对象中的变量来设置配置选项。例如，你可以发出以下命令来启用 OpenMP 支持：

```py
import theano
theano.config.openmp = True
theano.config.openmp_elemwise_minsize = 10

```

与 OpenMP 相关的参数如下：

+   `openmp_elemwise_minsize`：这是一个整数，表示应该启用元素级并行化的数组的最小大小（对于小数组，并行化的开销可能会损害性能）

+   `openmp`：这是一个布尔标志，用于控制 OpenMP 编译的激活（默认情况下应该被激活）

通过在执行代码之前设置`OMP_NUM_THREADS`环境变量，可以控制分配给 OpenMP 执行的线程数量。

我们现在可以编写一个简单的基准测试来演示实际中的 OpenMP 使用。在一个名为`test_theano.py`的文件中，我们将放置π估计示例的完整代码：

```py
    # File: test_theano.py
    import numpy as np
    import theano.tensor as T
    import theano as th
    th.config.openmp_elemwise_minsize = 1000
    th.config.openmp = True

    x = T.vector('x')
    y = T.vector('y')

    hit_test = x ** 2 + y ** 2 <= 1
    hits = hit_test.sum()
    misses = x.shape[0]
    pi_est = 4 * hits/misses

    calculate_pi = th.function([x, y], pi_est)

    x_val = np.random.uniform(-1, 1, 30000)
    y_val = np.random.uniform(-1, 1, 30000)

    import timeit
    res = timeit.timeit("calculate_pi(x_val, y_val)", 
                        "from __main__ import x_val, y_val, 
                        calculate_pi", number=100000)
    print(res)

```

在这个阶段，我们可以从命令行运行代码，并通过设置`OMP_NUM_THREADS`环境变量来评估线程数量增加时的扩展性：

```py
    $ OMP_NUM_THREADS=1 python test_theano.py
    10.905971487998613
    $ OMP_NUM_THREADS=2 python test_theano.py
    7.538279129999864
    $ OMP_NUM_THREADS=3 python test_theano.py
    9.405846934998408
    $ OMP_NUM_THREADS=4 python test_theano.py
    14.634153957000308

```

有趣的是，当使用两个线程时，会有轻微的加速，但随着线程数量的增加，性能会迅速下降。这意味着对于这个输入大小，使用超过两个线程并不有利，因为启动新线程和同步共享数据所付出的代价高于从并行执行中获得的加速。

实现良好的并行性能可能相当棘手，因为它将取决于特定的操作以及它们如何访问底层数据。一般来说，衡量并行程序的性能至关重要，而获得显著的加速则是一项试错的工作。

例如，我们可以看到使用略微不同的代码，并行性能会迅速下降。在我们的击中测试中，我们直接使用了 `sum` 方法，并依赖于对 `hit_tests` 布尔数组的显式转换。如果我们进行显式转换，Theano 将生成略微不同的代码，从多个线程中获得的益处较少。我们可以修改 `test_theano.py` 文件来验证这一效果：

```py
    # Older version
    # hits = hit_test.sum()
    hits = hit_test.astype('int32').sum()

```

如果我们重新运行我们的基准测试，我们会看到线程数量对运行时间没有显著影响。尽管如此，与原始版本相比，时间有所显著改善：

```py
    $ OMP_NUM_THREADS=1 python test_theano.py
    5.822126664999814
    $ OMP_NUM_THREADS=2 python test_theano.py
    5.697357518001809
    $ OMP_NUM_THREADS=3 python test_theano.py 
    5.636914656002773
    $ OMP_NUM_THREADS=4 python test_theano.py
    5.764030176000233

```

# 分析 Theano

考虑到衡量和分析性能的重要性，Theano 提供了强大且信息丰富的分析工具。要生成分析数据，只需在 `th.function` 中添加 `profile=True` 选项即可：

```py
    calculate_pi = th.function([x, y], pi_est, profile=True)

```

分析器将在函数运行时收集数据（例如，通过 `timeit` 或直接调用）。可以通过发出 `summary` 命令将分析摘要打印到输出，如下所示：

```py
    calculate_pi.profile.summary()

```

要生成分析数据，我们可以在添加 `profile=True` 选项后重新运行我们的脚本（对于这个实验，我们将 `OMP_NUM_THREADS` 环境变量设置为 1）。此外，我们将我们的脚本恢复到执行 `hit_tests` 隐式转换的版本。

您也可以使用 `config.profile` 选项全局设置分析。

`calculate_pi.profile.summary()` 打印的输出相当长且信息丰富。其中一部分在下一块文本中报告。输出由三个部分组成，分别按 `Class`、`Ops` 和 `Apply` 排序。在我们的例子中，我们关注的是 `Ops`，它大致对应于 Theano 编译代码中使用的函数。正如你所见，大约 80% 的时间用于对两个数字进行逐元素平方和求和，其余时间用于计算总和：

```py
Function profiling
==================
  Message: test_theano.py:15

... other output
Time in 100000 calls to Function.__call__: 1.015549e+01s
... other output

Class
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
.... timing info by class

Ops
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
 80.0%    80.0%       6.722s       6.72e-05s     C     100000        1   Elemwise{Composite{LT((sqr(i0) + sqr(i1)), i2)}}
  19.4%    99.4%       1.634s       1.63e-05s     C     100000        1   Sum{acc_dtype=int64}
   0.3%    99.8%       0.027s       2.66e-07s     C     100000        1   Elemwise{Composite{((i0 * i1) / i2)}}
   0.2%   100.0%       0.020s       2.03e-07s     C     100000        1   Shape_i{0}
   ... (remaining 0 Ops account for   0.00%(0.00s) of the runtime)

Apply
------
<% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
... timing info by apply

```

这些信息与我们第一次基准测试的结果一致。当使用两个线程时，代码从大约 11 秒减少到大约 8 秒。从这些数字中，我们可以分析时间是如何被花费的。

在这 11 秒中，80% 的时间（大约 8.8 秒）用于执行元素级操作。这意味着，在完全并行的情况下，增加两个线程将使速度提高 4.4 秒。在这种情况下，理论上的执行时间将是 6.6 秒。考虑到我们获得了大约 8 秒的计时，看起来线程使用有一些额外的开销（1.4 秒）。

# Tensorflow

Tensorflow 是另一个用于快速数值计算和自动并行化的库。它于 2015 年由 Google 以开源项目形式发布。Tensorflow 通过构建类似于 Theano 的数学表达式来工作，但计算不是编译成机器代码，而是在用 C++ 编写的外部引擎上执行。Tensorflow 支持在一台或多台 CPU 和 GPU 上执行和部署并行代码。

Tensorflow 的使用方式与 Theano 非常相似。要在 Tensorflow 中创建一个变量，你可以使用 `tf.placeholder` 函数，该函数接受一个数据类型作为输入：

```py
    import tensorflow as tf

    a = tf.placeholder('float64')

```

Tensorflow 的数学表达式可以非常类似于 Theano，除了命名约定略有不同以及对于 NumPy 语义的支持更加有限。

Tensorflow 并不像 Theano 那样将函数编译成 C 语言和机器代码，而是将定义的数学函数（包含变量和转换的数据结构称为 **计算图**）序列化，并在特定设备上执行。设备配置和上下文可以通过 `tf.Session` 对象来完成。

一旦定义了所需的表达式，就需要初始化一个 `tf.Session`，并可以使用 `Session.run` 方法来执行计算图。在以下示例中，我们展示了如何使用 Tensorflow API 实现一个简单的元素级平方和：

```py
    a = tf.placeholder('float64')
    b = tf.placeholder('float64')
    ab_sq = a**2 + b**2

    with tf.Session() as session:
        result = session.run(ab_sq, feed_dict={a: [0, 1, 2], 
                                               b: [3, 4, 5]})
        print(result)
    # Output:
    # array([  9.,  17.,  29.])

```

Tensorflow 的并行性是通过其智能执行引擎自动实现的，通常无需过多调整就能很好地工作。然而，请注意，它主要适用于涉及定义复杂函数（使用大量矩阵乘法并计算其梯度）的深度学习工作负载。

我们现在可以使用 Tensorflow 的功能来复制 pi 的估计示例，并对其执行速度和并行性进行基准测试，与 Theano 实现进行比较。我们将这样做：

+   定义我们的 `x` 和 `y` 变量，并使用广播操作进行碰撞测试。

+   使用 `tf.reduce_sum` 函数计算 `hit_tests` 的总和。

+   使用 `inter_op_parallelism_threads` 和 `intra_op_parallelism_threads` 配置选项初始化一个 `Session` 对象。这些选项控制不同类别的并行操作使用的线程数。请注意，使用这些选项创建的第一个 `Session` 将设置整个脚本（甚至未来的 `Session` 实例）的线程数。

我们现在可以编写一个名为`test_tensorflow.py`的脚本，其中包含以下代码。请注意，线程数作为脚本的第一个参数传递（`sys.argv[1]`）：

```py
    import tensorflow as tf
    import numpy as np
    import time
    import sys

    NUM_THREADS = int(sys.argv[1])
    samples = 30000

    print('Num threads', NUM_THREADS)
    x_data = np.random.uniform(-1, 1, samples)
    y_data = np.random.uniform(-1, 1, samples)

    x = tf.placeholder('float64', name='x')
    y = tf.placeholder('float64', name='y')

    hit_tests = x ** 2 + y ** 2 <= 1.0
    hits = tf.reduce_sum(tf.cast(hit_tests, 'int32'))

    with tf.Session
        (config=tf.ConfigProto
            (inter_op_parallelism_threads=NUM_THREADS,
             intra_op_parallelism_threads=NUM_THREADS)) as sess:
        start = time.time()
        for i in range(10000):
            sess.run(hits, {x: x_data, y: y_data})
        print(time.time() - start)

```

如果我们多次运行脚本，并使用不同的`NUM_THREADS`值，我们会看到性能与 Theano 相当，并且通过并行化提高的速度提升相当有限：

```py
 $ python test_tensorflow.py 1
    13.059704780578613
 $ python test_tensorflow.py 2
    11.938535928726196
 $ python test_tensorflow.py 3
    12.783955574035645
 $ python test_tensorflow.py 4
    12.158143043518066

```

使用 Tensorflow 和 Theano 等软件包的主要优势是支持在机器学习算法中常用到的并行矩阵运算。这非常有效，因为这些操作可以在专门为以高吞吐量执行这些操作而设计的 GPU 硬件上实现令人印象深刻的性能提升。

# 在 GPU 上运行代码

在本小节中，我们将演示使用 Theano 和 Tensorflow 的 GPU 使用方法。作为一个例子，我们将基准测试 GPU 上非常简单的矩阵乘法执行时间，并将其与 CPU 上的运行时间进行比较。

本小节中的代码需要具备 GPU。为了学习目的，可以使用 Amazon EC2 服务([`aws.amazon.com/ec2`](https://aws.amazon.com/ec2))来请求一个启用 GPU 的实例。

以下代码使用 Theano 执行简单的矩阵乘法。我们使用`T.matrix`函数初始化一个二维数组，然后使用`T.dot`方法执行矩阵乘法：

```py
    from theano import function, config
    import theano.tensor as T
    import numpy as np
    import time

    N = 5000

    A_data = np.random.rand(N, N).astype('float32')
    B_data = np.random.rand(N, N).astype('float32')

    A = T.matrix('A')
    B = T.matrix('B')

    f = function([A, B], T.dot(A, B))

    start = time.time()
    f(A_data, B_data)

    print("Matrix multiply ({}) took {} seconds".format(N, time.time() - start))
    print('Device used:', config.device)

```

可以通过设置`config.device=gpu`选项让 Theano 在 GPU 上执行此代码。为了方便起见，我们可以使用`THEANO_FLAGS`环境变量从命令行设置配置值，如下所示。在将前面的代码复制到`test_theano_matmul.py`文件后，我们可以通过以下命令来基准测试执行时间：

```py
 $ THEANO_FLAGS=device=gpu python test_theano_gpu.py 
    Matrix multiply (5000) took 0.4182612895965576 seconds
    Device used: gpu

```

我们可以使用`device=cpu`配置选项在 CPU 上类似地运行相同的代码：

```py
 $ THEANO_FLAGS=device=cpu python test_theano.py 
    Matrix multiply (5000) took 2.9623231887817383 seconds
    Device used: cpu

```

如您所见，在这个例子中，GPU 比 CPU 版本快 7.2 倍！

为了进行比较，我们可以使用 Tensorflow 基准测试等效代码。Tensorflow 版本的实现将在下一个代码片段中报告。与 Theano 版本的主要区别如下：

+   `tf.device`配置管理器的使用，用于指定目标设备（`/cpu:0`或`/gpu:0`）

+   矩阵乘法是通过`tf.matmul`运算符来执行的：

```py
    import tensorflow as tf
    import time
    import numpy as np
    N = 5000

    A_data = np.random.rand(N, N)
    B_data = np.random.rand(N, N)

    # Creates a graph.

 with tf.device('/gpu:0'):
        A = tf.placeholder('float32')
        B = tf.placeholder('float32')

        C = tf.matmul(A, B)

    with tf.Session() as sess:
        start = time.time()
        sess.run(C, {A: A_data, B: B_data})
        print('Matrix multiply ({}) took: {}'.format(N, time.time() - start))

```

如果我们使用适当的`tf.device`选项运行`test_tensorflow_matmul.py`脚本，我们将获得以下计时结果：

```py
 # Ran with tf.device('/gpu:0')
    Matrix multiply (5000) took: 1.417285680770874

 # Ran with tf.device('/cpu:0')
    Matrix multiply (5000) took: 2.9646761417388916 

```

如您所见，在这个简单案例中，性能提升相当显著（但不如 Theano 版本好）。

实现自动 GPU 计算的另一种方法是现在熟悉的 Numba。使用 Numba，可以将 Python 代码编译成可以在 GPU 上运行的程序。这种灵活性允许进行高级 GPU 编程以及更简化的接口。特别是，Numba 使得编写 GPU 就绪的通用函数变得极其简单。

在下一个示例中，我们将演示如何编写一个通用函数，该函数对两个数字应用指数函数并求和结果。正如我们已经在第五章中看到的，*探索编译器*，这可以通过使用`nb.vectorize`函数（我们还将明确指定`cpu`目标）来实现：

```py
    import numba as nb
    import math
    @nb.vectorize(target='cpu')
    def expon_cpu(x, y):
        return math.exp(x) + math.exp(y)

```

使用`target='cuda'`选项可以将`expon_cpu`通用函数编译为 GPU 设备。此外，请注意，对于 CUDA 通用函数，必须指定输入类型。`expon_gpu`的实现如下：

```py
    @nb.vectorize(['float32(float32, float32)'], target='cuda')
    def expon_gpu(x, y):
        return math.exp(x) + math.exp(y)

```

我们现在可以通过在两个大小为 1,000,000 的数组上应用这两个函数来基准测试这两个函数的执行。请注意，我们在测量时间之前执行函数以触发 Numba 即时编译：

```py
    import numpy as np
    import time

    N = 1000000
    niter = 100

    a = np.random.rand(N).astype('float32')
    b = np.random.rand(N).astype('float32')

    # Trigger compilation
    expon_cpu(a, b)
    expon_gpu(a, b)

    # Timing
    start = time.time()
    for i in range(niter):
       expon_cpu(a, b)
    print("CPU:", time.time() - start)

    start = time.time()
    for i in range(niter): 
        expon_gpu(a, b) 
    print("GPU:", time.time() - start) 
    # Output:
    # CPU: 2.4762887954711914
    # GPU: 0.8668839931488037

```

多亏了 GPU 执行，我们能够将 CPU 版本的速度提高 3 倍。请注意，在 GPU 上传输数据相当昂贵；因此，只有对于非常大的数组，GPU 执行才具有优势。

# 摘要

并行处理是提高大数据集性能的有效方法。令人尴尬的并行问题是非常好的并行执行候选者，可以轻松实现以实现良好的性能扩展。

在本章中，我们介绍了 Python 并行编程的基础。我们学习了如何通过使用 Python 标准库中的工具来生成进程来规避 Python 线程限制。我们还探讨了如何使用 Cython 和 OpenMP 实现多线程程序。

对于更复杂的问题，我们学习了如何使用 Theano、Tensorflow 和 Numba 包自动编译针对 CPU 和 GPU 设备并行执行的密集数组表达式。

在下一章中，我们将学习如何使用 dask 和 PySpark 等库在多个处理器和机器上编写和执行并行程序。
