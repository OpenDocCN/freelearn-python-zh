# 开始使用PyCUDA

在上一章中，我们设置了编程环境。现在，有了我们的驱动程序和编译器牢固地安装好，我们将开始实际的GPU编程！我们将首先学习如何使用PyCUDA进行一些基本和基础的操作。我们将首先看看如何查询我们的GPU - 也就是说，我们将首先编写一个小的Python程序，告诉我们GPU的特性，如核心数量、架构和内存。然后，我们将花一些时间熟悉如何在Python和GPU之间传输内存，使用PyCUDA的`gpuarray`类以及如何使用这个类进行基本计算。本章的其余部分将花在展示如何编写一些基本函数（我们将称之为**CUDA内核**），我们可以直接启动到GPU上。

本章的学习成果如下：

+   使用PyCUDA确定GPU特性，如内存容量或核心数量

+   理解主机（CPU）和设备（GPU）内存之间的区别，以及如何使用PyCUDA的`gpuarray`类在主机和设备之间传输数据

+   如何使用只有`gpuarray`对象进行基本计算

+   如何使用PyCUDA的`ElementwiseKernel`函数在GPU上执行基本的逐元素操作

+   理解函数式编程概念的reduce/scan操作，以及如何制作基本的缩减或扫描CUDA内核

# 技术要求

本章需要一台安装了现代NVIDIA GPU（2016年以后）的Linux或Windows 10 PC，并安装了所有必要的GPU驱动程序和CUDA Toolkit（9.0以后）。还需要一个合适的Python 2.7安装（如Anaconda Python 2.7），并安装了PyCUDA模块。

本章的代码也可以在GitHub上找到：[https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)。

有关先决条件的更多信息，请查看本书的*前言*；有关软件和硬件要求，请查看[https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)中的`README`部分。

# 查询您的GPU

在我们开始编程GPU之前，我们应该真正了解一些关于其技术能力和限制的知识。我们可以通过进行所谓的**GPU查询**来确定这一点。GPU查询是一个非常基本的操作，它将告诉我们GPU的具体技术细节，如可用的GPU内存和核心数量。NVIDIA在`samples`目录中（适用于Windows和Linux）包含了一个纯CUDA-C编写的命令行示例`deviceQuery`，我们可以运行它来执行此操作。让我们看一下作者的Windows 10笔记本电脑（Microsoft Surface Book 2，配备了GTX 1050 GPU）上产生的输出：

![](assets/ef6b22de-9871-49b2-ad73-4e7aff2017ac.png)

让我们来看看这里显示的所有技术信息的一些基本要点。首先，我们看到只安装了一个GPU，设备0 - 可能主机计算机安装了多个GPU并使用它们，因此CUDA将为每个*GPU设备*指定一个独立的编号。有些情况下，我们可能需要明确指定设备编号，所以了解这一点总是很好的。我们还可以看到我们拥有的具体设备类型（这里是GTX 1050），以及我们正在使用的CUDA版本。现在我们还要注意两件事：核心的总数（这里是640），以及设备上的全局内存总量（在本例中为2,048兆字节，即2千兆字节）。

虽然您可以从`deviceQuery`中看到许多其他技术细节，但核心数量和内存量通常是您第一次在新GPU上运行时应该关注的前两件事，因为它们可以让您最直接地了解新设备的容量。

# 使用PyCUDA查询您的GPU

现在，最后，我们将通过用Python编写我们自己的版本的`deviceQuery`来开始我们的GPU编程之旅。在这里，我们主要关注设备上可用内存的数量，计算能力，多处理器的数量和CUDA核心的总数。

我们将从以下方式初始化CUDA：

```py
import pycuda.driver as drv
drv.init()
```

请注意，我们将始终需要使用`pycuda.driver.init()`或通过导入PyCUDA的`autoinit`子模块`import pycuda.autoinit`来初始化PyCUDA！

我们现在可以立即检查我们的主机计算机上有多少个GPU设备：

```py
print 'Detected {} CUDA Capable device(s)'.format(drv.Device.count())
```

让我们在IPython中输入这个并看看会发生什么：

![](assets/9c6850ad-552d-48ed-a6d5-4145c4f7407f.png)

太好了！到目前为止，我已经验证了我的笔记本确实有一个GPU。现在，让我们通过添加几行代码来迭代可以通过`pycuda.driver.Device`（按编号索引）单独访问的每个设备，以提取有关此GPU（以及系统上的任何其他GPU）的更多有趣信息。设备的名称（例如，GeForce GTX 1050）由`name`函数给出。然后我们使用`compute_capability`函数获取设备的**计算能力**和`total_memory`函数获取设备的总内存量。

**计算能力**可以被视为每个NVIDIA GPU架构的*版本号*；这将为我们提供一些关于设备的重要信息，否则我们无法查询，我们将在一分钟内看到。

我们将这样写：

```py
for i in range(drv.Device.count()):

     gpu_device = drv.Device(i)
     print 'Device {}: {}'.format( i, gpu_device.name() )
     compute_capability = float( '%d.%d' % gpu_device.compute_capability() )
     print '\t Compute Capability: {}'.format(compute_capability)
     print '\t Total Memory: {} megabytes'.format(gpu_device.total_memory()//(1024**2))
```

现在，我们准备查看PyCUDA以Python字典类型形式提供给我们的GPU的一些剩余属性。我们将使用以下行将其转换为由字符串索引属性的字典：

```py
    device_attributes_tuples = gpu_device.get_attributes().iteritems()
     device_attributes = {}

     for k, v in device_attributes_tuples:
         device_attributes[str(k)] = v
```

现在，我们可以使用以下内容确定设备上的*多处理器*数量：

```py
    num_mp = device_attributes['MULTIPROCESSOR_COUNT']
```

GPU将其各个核心划分为称为**流处理器（SMs）**的较大单元； GPU设备将具有多个SM，每个SM将根据设备的计算能力具有特定数量的CUDA核心。要明确：每个多处理器的核心数并不是由GPU直接指示的-这是由计算能力隐含给我们的。我们将不得不查阅NVIDIA的一些技术文件以确定每个多处理器的核心数（参见[http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)），然后创建一个查找表来给出每个多处理器的核心数。我们使用`compute_capability`变量来查找核心数：

```py
    cuda_cores_per_mp = { 5.0 : 128, 5.1 : 128, 5.2 : 128, 6.0 : 64, 6.1 : 128, 6.2 : 128}[compute_capability]
```

现在我们可以通过将这两个数字相乘来最终确定设备上的总核心数：

```py
    print '\t ({}) Multiprocessors, ({}) CUDA Cores / Multiprocessor: {} CUDA Cores'.format(num_mp, cuda_cores_per_mp, num_mp*cuda_cores_per_mp)
```

现在，我们可以通过迭代字典中剩余的键并打印相应的值来完成我们的程序：

```py
    device_attributes.pop('MULTIPROCESSOR_COUNT')

     for k in device_attributes.keys():
         print '\t {}: {}'.format(k, device_attributes[k])
```

所以，现在我们终于完成了本文的第一个真正的GPU程序！（也可在[https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA/blob/master/3/deviceQuery.py](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA/blob/master/3/deviceQuery.py)找到）。现在，我们可以按如下方式运行它：

![](assets/59a5907a-0a76-4c08-bfe6-349d9ce48c71.png)

现在，我们可以有点自豪，因为我们确实可以编写一个程序来查询我们的GPU！现在，让我们真正开始学习*使用*我们的GPU，而不仅仅是观察它。

# 使用PyCUDA的gpuarray类

就像 NumPy 的 `array` 类是 NumPy 环境中数值编程的基石一样，PyCUDA 的 `gpuarray` 类在 Python 中的 GPU 编程中扮演着类似的重要角色。它具有你从 NumPy 中熟悉和喜爱的所有功能——多维向量/矩阵/张量形状结构、数组切片、数组展开，以及用于逐点计算的重载运算符（例如 `+`、`-`、`*`、`/` 和 `**`）。

`gpuarray` 对于任何新手 GPU 程序员来说都是一个不可或缺的工具。在我们继续之前，我们将花费这一部分时间来了解这种特定的数据结构，并对其有一个深入的理解。

# 使用 gpuarray 将数据传输到 GPU 和从 GPU 中传输数据

正如我们在 Python 中编写的先前的 `deviceQuery` 程序所示，GPU 有自己的内存，与主机计算机的内存分开，这被称为**设备内存**。（有时这更具体地被称为**全局设备内存**，以区分它与 GPU 上的其他缓存内存、共享内存和寄存器内存。）在大多数情况下，我们将 GPU 上的（全局）设备内存视为我们在 C 中动态分配的堆内存（使用 `malloc` 和 `free` 函数）或 C++（使用 `new` 和 `delete` 运算符）；在 CUDA C 中，这进一步复杂化，需要在 CPU 和 GPU 空间之间来回传输数据（使用诸如 `cudaMemcpyHostToDevice` 和 `cudaMemcpyDeviceToHost` 的命令），同时跟踪 CPU 和 GPU 空间中的多个指针，并执行适当的内存分配（`cudaMalloc`）和释放（`cudaFree`）。

幸运的是，PyCUDA 通过 `gpuarray` 类涵盖了所有的内存分配、释放和数据传输的开销。正如所述，这个类类似于 NumPy 数组，使用矢量/矩阵/张量形状结构信息来处理数据。`gpuarray` 对象甚至根据其生命周期自动执行清理，因此当我们完成后，我们不必担心释放存储在 `gpuarray` 对象中的任何 GPU 内存。

那么我们如何使用它将数据从主机传输到 GPU 呢？首先，我们必须将我们的主机数据包含在某种形式的 NumPy 数组中（我们称之为 `host_data`），然后使用 `gpuarray.to_gpu(host_data)` 命令将其传输到 GPU 并创建一个新的 GPU 数组。

现在让我们在 GPU 中执行一个简单的计算（在 GPU 上的常数点乘），然后使用 `gpuarray.get` 函数将 GPU 数据检索到一个新的数组中。让我们加载 IPython 并看看它是如何工作的（请注意，这里我们将使用 `import pycuda.autoinit` 初始化 PyCUDA）：

![](assets/14eef3e5-273f-45c9-b42f-99d07628f9d8.png)

需要注意的一点是，当我们设置 NumPy 数组时，我们特别指定了主机上的数组类型为 NumPy `float32` 类型，并使用 `dtype` 选项；这与 C/C++ 中的浮点类型直接对应。一般来说，当我们发送数据到 GPU 时，最好使用 NumPy 明确设置数据类型。原因有两个：首先，由于我们使用 GPU 来提高应用程序的性能，我们不希望使用不必要的类型造成不必要的计算时间或内存开销；其次，由于我们很快将在内联 CUDA C 中编写代码的部分，我们必须非常具体地指定类型，否则我们的代码将无法正确工作，要记住 C 是一种静态类型语言。

记得为将要传输到 GPU 的 NumPy 数组明确设置数据类型。这可以在 `numpy.array` 类的构造函数中使用 `dtype` 选项来完成。

# 使用 gpuarray 进行基本的逐点算术运算

在最后一个示例中，我们看到我们可以使用（重载的）Python乘法运算符（`*`）来将`gpuarray`对象中的每个元素乘以一个标量值（这里是2）；请注意，逐点操作本质上是可并行化的，因此当我们在`gpuarray`对象上使用此操作时，PyCUDA能够将每个乘法操作分配到单个线程上，而不是依次串行计算每个乘法（公平地说，一些版本的NumPy可以使用现代x86芯片中的高级SSE指令进行这些计算，因此在某些情况下性能将与GPU相当）。明确一点：在GPU上执行的这些逐点操作是并行的，因为一个元素的计算不依赖于任何其他元素的计算。

为了了解这些运算符的工作原理，我建议读者加载IPython并在GPU上创建一些`gpuarray`对象，然后玩几分钟，看看这些运算符是否与NumPy中的数组类似。以下是一些灵感：

![](assets/fd5469e2-c573-472e-a1a9-6da1dc61ddc5.png)

现在，我们可以看到`gpuarray`对象的行为是可预测的，并且与NumPy数组的行为一致。（请注意，我们将不得不使用`get`函数从GPU中获取输出！）现在让我们比较一下CPU和GPU计算时间，看看在何时是否有任何优势进行这些操作。

# 速度测试

让我们编写一个小程序（`time_calc0.py`），对CPU上的标量乘法和GPU上的相同操作进行速度比较测试。然后，我们将使用NumPy的`allclose`函数比较两个输出值。我们将生成一个包含5000万个随机32位浮点值的数组（这将大约占用48兆字节的数据，因此在任何稍微现代的主机和GPU设备上都应该完全可行），然后我们将计算在两个设备上将数组乘以2所需的时间。最后，我们将比较输出值以确保它们相等。操作如下：

```py
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time
host_data = np.float32( np.random.random(50000000) )

t1 = time()
host_data_2x =  host_data * np.float32(2)
t2 = time()

print 'total time to compute on CPU: %f' % (t2 - t1)
device_data = gpuarray.to_gpu(host_data)

t1 = time()
device_data_2x =  device_data * np.float32( 2 )
t2 = time()

from_device = device_data_2x.get()
print 'total time to compute on GPU: %f' % (t2 - t1)

print 'Is the host computation the same as the GPU computation? : {}'.format(np.allclose(from_device, host_data_2x) )
```

（您可以在之前提供给您的存储库中找到`time_calc0.py`文件。）

现在，让我们加载IPython并运行几次，以了解这些的一般速度，并查看是否有任何变化。（这里，这是在2017年的微软Surface Book 2上运行的，配备了Kaby Lake i7处理器和GTX 1050 GPU。）：

![](assets/a278a2f1-e099-4907-805f-708f2884a7c3.png)

我们首先注意到，每次计算的CPU计算时间大致相同（大约0.08秒）。然而，我们注意到，第一次运行时，GPU计算时间比CPU计算时间慢得多（1.09秒），并且在随后的运行中变得快得多，在每次后续运行中保持大致恒定（在7或9毫秒的范围内）。如果您退出IPython，然后再次运行程序，将发生相同的情况。这种现象的原因是什么？好吧，让我们使用IPython内置的`prun`分析器进行一些调查工作。（这类似于[第1章](f9c54d0e-6a18-49fc-b04c-d44a95e011a2.xhtml)中介绍的`cProfiler`模块，*为什么要进行GPU编程？*）

首先，让我们将我们的程序作为文本加载到IPython中，然后通过Python的`exec`命令运行我们的分析器：

```py
with open('time_calc0.py','r') as f:
     time_calc_code = f.read()
```

现在，我们在IPython控制台中键入`%prun -s cumulative exec(time_calc_code)`（带有前导`%`）并查看哪些操作花费了最多的时间：

![](assets/7dfbfc79-dcc1-4cc8-b7b6-7f11103f54e6.png)

现在，有一些可疑的对Python模块文件`compiler.py`的调用；这些调用总共大约需要一秒钟，比在这里进行GPU计算所需的时间略少。现在让我们再次运行一下，看看是否有任何差异：

![](assets/da5995b8-f05d-45d7-950c-f921d79b3886.png)

请注意，这一次没有调用`compiler.py`。为什么呢？由于PyCUDA库的性质，GPU代码通常在给定的Python会话中首次运行时使用NVIDIA的`nvcc`编译器进行编译和链接；然后它被缓存，如果再次调用代码，则不必重新编译。这甚至可能包括*简单*的操作，比如标量乘法！（我们最终会看到，通过使用[第10章](5383b46f-8dc6-4e17-ab35-7f6bd35f059f.xhtml)中的预编译代码或使用NVIDIA自己的线性代数库与Scikit-CUDA模块一起使用CUDA库，可以改善这一点，我们将在[第7章](55146879-4b7e-4774-9a8b-cc5c80c04ed8.xhtml)中看到）。

在PyCUDA中，GPU代码通常在运行时使用NVIDIA的`nvcc`编译器进行编译，然后从PyCUDA中调用。这可能会导致意外的减速，通常是在给定的Python会话中首次运行程序或GPU操作时。

# 使用PyCUDA的ElementWiseKernel执行逐点计算

现在，让我们看看如何使用PyCUDA的`ElementWiseKernel`函数直接在GPU上编写我们自己的逐点（或等效地，*逐元素*）操作。这就是我们之前对C/C++编程的了解将变得有用的地方——我们将不得不在CUDA C中编写一点*内联代码*，这些代码是由NVIDIA的`nvcc`编译器在外部编译的，然后通过PyCUDA在运行时由我们的代码启动。

在本文中，我们经常使用术语**kernel**；通过*kernel*，我们总是指的是由CUDA直接启动到GPU上的函数。我们将使用PyCUDA的几个函数来生成不同类型的kernel的模板和设计模式，以便更轻松地过渡到GPU编程。

让我们直接开始；我们将从头开始重写代码，使用CUDA-C将`gpuarray`对象的每个元素乘以2；我们将使用PyCUDA的`ElementwiseKernel`函数来生成我们的代码。您应该尝试直接在IPython控制台中输入以下代码。（不那么冒险的人可以从本文的Git存储库中下载，文件名为`simple_element_kernel_example0.py`）：

```py
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time
from pycuda.elementwise import ElementwiseKernel
host_data = np.float32( np.random.random(50000000) )
gpu_2x_ker = ElementwiseKernel(
"float *in, float *out",
"out[i] = 2*in[i];",
"gpu_2x_ker")
```

让我们看看这是如何设置的；当然，这是几行内联C。我们首先在第一行中设置输入和输出变量（`"float *in, float *out"`），这通常是指指向GPU上已分配内存的C指针的形式。在第二行中，我们使用`"out[i] = 2*in[i];"`定义了我们的逐元素操作，它将把`in`中的每个点乘以2，并将其放在`out`的相应索引中。

请注意，PyCUDA会自动为我们设置整数索引`i`。当我们使用`i`作为我们的索引时，`ElementwiseKernel`将自动在GPU的许多核心中并行化我们的计算。最后，我们给我们的代码片段起了一个内部CUDA C kernel的名称（`"gpu_2x_ker"`）。由于这是指CUDA C的命名空间而不是Python的，因此将其与Python中的名称相同是可以的（也很方便）。

现在，让我们进行速度比较：

```py
def speedcomparison():
    t1 = time()
    host_data_2x =  host_data * np.float32(2)
    t2 = time()
    print 'total time to compute on CPU: %f' % (t2 - t1)
    device_data = gpuarray.to_gpu(host_data)
    # allocate memory for output
    device_data_2x = gpuarray.empty_like(device_data)
    t1 = time()
    gpu_2x_ker(device_data, device_data_2x)
    t2 = time()
    from_device = device_data_2x.get()
    print 'total time to compute on GPU: %f' % (t2 - t1)
    print 'Is the host computation the same as the GPU computation? : {}'.format(np.allclose(from_device, host_data_2x) )

if __name__ == '__main__':
    speedcomparison()
```

现在，让我们运行这个程序：

![](assets/02db7f9f-e682-41fb-af4c-2833d054a746.png)

哇！看起来不太好。让我们从IPython中运行`speedcomparison()`函数几次：

![](assets/9514e819-e7cd-42c3-b1af-d5ea832c6864.png)

正如我们所看到的，第一次使用给定的GPU函数后，速度显著增加。与前面的例子一样，这是因为PyCUDA在首次调用给定的GPU kernel函数时使用`nvcc`编译器编译我们的内联CUDA C代码。代码编译后，它将被缓存并在给定的Python会话的其余部分中重复使用。

现在，在我们继续之前，让我们再讨论一些重要的事情，这是非常微妙的。我们定义的小内核函数操作C浮点指针；这意味着我们将不得不在GPU上分配一些空的内存，该内存由`out`变量指向。再次看一下`speedcomparison()`函数中的这部分代码：

```py
device_data = gpuarray.to_gpu(host_data)
# allocate memory for output
device_data_2x = gpuarray.empty_like(device_data)
```

与之前一样，我们通过`gpuarray.to_gpu`函数将一个NumPy数组（`host_data`）发送到GPU，该函数会自动将数据分配到GPU并从CPU空间复制过来。我们将把这个数组插入到我们内核函数的`in`部分。在下一行，我们使用`gpuarray.empty_like`函数在GPU上分配空的内存。这类似于C中的普通`malloc`，分配一个与`device_data`大小和数据类型相同的数组，但不复制任何内容。现在我们可以将其用于内核函数的`out`部分。现在我们来看一下`speedcomparison()`中的下一行，看看如何将我们的内核函数启动到GPU上（忽略我们用于计时的行）：

```py
gpu_2x_ker(device_data, device_data_2x)
```

再次，我们设置的变量直接对应于我们用`ElementwiseKernel`定义的第一行（这里是`"float *in, float *out"`）。

# 曼德勃罗重新审视

让我们再次从[第1章](f9c54d0e-6a18-49fc-b04c-d44a95e011a2.xhtml)“为什么使用GPU编程？”中生成曼德勃罗集的问题。原始代码可以在存储库的`1`文件夹中找到，文件名为`mandelbrot0.py`，在继续之前，您应该再次查看一下。我们看到该程序有两个主要组成部分：第一个是生成曼德勃罗集，第二个是将曼德勃罗集转储到PNG文件中。在第一章中，我们意识到我们只能并行生成曼德勃罗集，并且考虑到这占程序运行时间的大部分，这将是一个很好的候选算法，可以将其转移到GPU上。让我们看看如何做到这一点。（我们将避免重复定义曼德勃罗集，因此如果您需要更深入的复习，请重新阅读[第1章](f9c54d0e-6a18-49fc-b04c-d44a95e011a2.xhtml)“为什么使用GPU编程？”中的“曼德勃罗重新审视”部分）

首先，让我们基于原始程序中的`simple_mandelbrot`创建一个新的Python函数。我们将其称为`gpu_mandelbrot`，这将接受与之前完全相同的输入：

```py
def gpu_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):
```

我们将从这里开始以稍微不同的方式进行。我们将首先构建一个复杂的晶格，其中包含我们将分析的复平面中的每个点。

在这里，我们将使用一些NumPy矩阵类型的技巧轻松生成晶格，然后将结果从NumPy `matrix`类型转换为二维NumPy `array`类型（因为PyCUDA只能处理NumPy `array`类型，而不能处理`matrix`类型）。请注意我们非常小心地设置我们的NumPy类型：

```py
    real_vals = np.matrix(np.linspace(real_low, real_high, width), dtype=np.complex64)
    imag_vals = np.matrix(np.linspace( imag_high, imag_low, height), dtype=np.complex64) * 1j
    mandelbrot_lattice = np.array(real_vals + imag_vals.transpose(), dtype=np.complex64)  
```

因此，我们现在有一个表示我们将生成曼德勃罗集的晶格的二维复杂数组；正如我们将看到的，我们可以在GPU内非常容易地操作这个数组。现在让我们将我们的晶格传输到GPU，并分配一个数组来表示我们的曼德勃罗集：

```py
    # copy complex lattice to the GPU
    mandelbrot_lattice_gpu = gpuarray.to_gpu(mandelbrot_lattice)    
    # allocate an empty array on the GPU
    mandelbrot_graph_gpu = gpuarray.empty(shape=mandelbrot_lattice.shape, dtype=np.float32)
```

重申一下——`gpuarray.to_array`函数只能操作NumPy `array`类型，因此我们在将其发送到GPU之前一定要对其进行类型转换。接下来，我们必须使用`gpuarray.empty`函数在GPU上分配一些内存，指定数组的大小/形状和类型。同样，您可以将其视为类似于C中的`malloc`；请记住，由于`gpuarray`对象析构函数在作用域结束时自动处理内存清理，因此我们不必在以后释放或`free`这些内存。

当您使用PyCUDA函数`gpuarray.empty`或`gpuarray.empty_like`在GPU上分配内存时，由于`gpuarray`对象的析构函数管理所有内存清理，因此您不必在以后释放此内存。

现在我们准备启动内核；我们唯一需要做的更改是改变

我们还没有编写生成曼德勃罗集的内核函数，但让我们先写出这个函数的其余部分应该是怎样的：

```py
    mandel_ker( mandelbrot_lattice_gpu, mandelbrot_graph_gpu, np.int32(max_iters), np.float32(upper_bound))

    mandelbrot_graph = mandelbrot_graph_gpu.get()

    return mandelbrot_graph
```

这就是我们希望我们的新内核行为的方式——第一个输入将是我们生成的复点阵（NumPy `complex64`类型），第二个将是指向二维浮点数组的指针（NumPy `float32`类型），它将指示哪些元素是曼德勃罗集的成员，第三个将是一个整数，表示每个点的最大迭代次数，最后一个输入将是用于确定曼德勃罗类成员资格的每个点的上限。请注意，我们在将所有输入传递给GPU时非常小心！

下一行将从GPU中检索我们生成的曼德勃罗集回到CPU空间，并返回结束值。（请注意，`gpu_mandelbrot`的输入和输出与`simple_mandelbrot`完全相同）。

现在让我们看看如何正确定义我们的GPU内核。首先，让我们在头部添加适当的`include`语句：

```py
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
```

我们现在准备编写我们的GPU内核！我们将在这里展示它，然后逐行讨论：

```py
mandel_ker = ElementwiseKernel(
"pycuda::complex<float> *lattice, float *mandelbrot_graph, int max_iters, float upper_bound",
"""
mandelbrot_graph[i] = 1;
pycuda::complex<float> c = lattice[i]; 
pycuda::complex<float> z(0,0);
for (int j = 0; j < max_iters; j++)
    {  
     z = z*z + c;     
     if(abs(z) > upper_bound)
         {
          mandelbrot_graph[i] = 0;
          break;
         }
    }         
""",
"mandel_ker")
```

首先，我们使用传递给`ElementwiseKernel`的第一个字符串设置我们的输入。我们必须意识到当我们在CUDA-C中工作时，特定的C数据类型将直接对应于特定的Python NumPy数据类型。再次注意，当数组被传递到CUDA内核时，它们被CUDA视为C指针。在这里，CUDA C `int`类型与NumPy `int32`类型完全对应，而CUDA C `float`类型对应于NumPy `float32`类型。然后使用内部PyCUDA类模板进行复杂类型的转换——这里PyCUDA `::complex<float>`对应于Numpy `complex64`。

让我们看看第二个字符串的内容，它用三个引号（`"""`）分隔。这使我们能够在字符串中使用多行；当我们在Python中编写更大的内联CUDA内核时，我们将使用这个。

虽然我们传入的数组在Python中是二维数组，但CUDA只会将它们视为一维数组，并由`i`索引。同样，`ElementwiseKernel`会自动为我们跨多个核心和线程索引`i`。我们将输出中的每个点初始化为1，如`mandelbrot_graph[i] = 1;`，因为`i`将在曼德勃罗集的每个元素上进行索引；我们将假设每个点都是成员，除非证明相反。（再次说明，曼德勃罗集是在两个维度上的，实部和虚部，但`ElementwiseKernel`将自动将所有内容转换为一维集合。当我们再次在Python中与数据交互时，曼德勃罗集的二维结构将被保留。）

我们像在Python中一样为我们的`c`值设置适当的点阵点，如`pycuda::complex<float> c = lattice[i];`，并用`pycuda::complex<float> z(0,0);`将我们的`z`值初始化为`0`（第一个零对应于实部，而第二个对应于虚部）。然后我们使用一个新的迭代器`j`进行循环，如`for(int j = 0; j < max_iters; j++)`。（请注意，这个算法不会在`j`或任何其他索引上并行化——只有`i`！这个`for`循环将在`j`上串行运行——但整个代码片段将在`i`上并行化。）

然后，我们使用`z = z*z + c;`设置`*z*`的新值，按照曼德勃罗算法。如果这个元素的绝对值超过了上限（`if(abs(z) > upper_bound)`），我们将这个点设置为0（`mandelbrot_graph[i] = 0;`），并用`break`关键字跳出循环。

在传递给`ElementwiseKernel`的最终字符串中，我们为内核赋予其内部CUDA C名称，这里是`"mandel_ker"`。

我们现在准备启动内核；我们唯一需要做的更改是将主函数中的引用从`simple_mandelbrot`更改为`gpu_mandelbrot`，然后我们就可以开始了。让我们从IPython中启动：

![](assets/f00d5080-4975-4023-9f14-397a8e007ac4.png)

让我们检查转储的图像，以确保这是正确的：

![](assets/6fa6851a-bcce-46d0-a63d-8023766da21a.png)

这肯定是在第一章中生成的相同Mandelbrot图像，所以我们已经成功地将其实现到了GPU上！现在让我们看看我们得到的速度增加：在第一章中，我们花了14.61秒来生成这张图；而在这里，只花了0.894秒。请记住，PyCUDA还必须在运行时编译和链接我们的CUDA C代码，并且需要花费时间来进行与GPU的内存传输。即使有了所有这些额外的开销，它仍然是一个非常值得的速度增加！（您可以在Git存储库中找到我们的GPU Mandelbrot的代码，文件名为`gpu_mandelbrot0.py`。）

# 对函数式编程的简要探讨

在我们继续之前，让我们简要回顾一下Python中用于**函数式编程**的两个函数——`map`和`reduce`。它们都被认为是*函数式*，因为它们都对*函数*进行操作。我们发现这些有趣，因为它们都对应于编程中的常见设计模式，所以我们可以替换输入中的不同函数，以获得多种不同（和有用的）操作。

让我们首先回顾Python中的`lambda`关键字。这允许我们定义一个**匿名函数**——在大多数情况下，这些可以被视为`一次性`函数，或者只希望使用一次的函数，或者可以在一行上定义的函数。让我们现在打开IPython并定义一个将数字平方的小函数，如`pow2 = lambda x : x**2`。让我们在一些数字上测试一下：

![](assets/f7154a51-4486-4292-9ed0-89415e526394.png)

让我们回顾一下`map`作用于两个输入值：一个函数和给定函数可以作用的对象`列表`。`map`输出原始列表中每个元素的函数输出列表。现在让我们将我们的平方操作定义为一个匿名函数，然后将其输入到map中，并使用最后几个数字的列表进行检查，如`map(lambda x : x**2, [2,3,4])`：

![](assets/c41f370e-9cba-40ba-a5df-e84857044437.png)

我们看到`map`作为`ElementwiseKernel`！这实际上是函数式编程中的标准设计模式。现在，让我们看看`reduce`；它不是接收一个列表并直接输出相应列表，而是接收一个列表，在其上执行递归二进制操作，并输出一个单例。让我们通过键入`reduce(lambda x, y : x + y, [1,2,3,4])`来了解这种设计模式。当我们在IPython中键入这个时，我们将看到这将输出一个单个数字10，这确实是*1+2+3+4*的和。您可以尝试用乘法替换上面的求和，并看到这确实适用于递归地将一长串数字相乘。一般来说，我们使用*可结合的二进制操作*进行缩减操作；这意味着，无论我们在列表的连续元素之间以何种顺序执行操作，都将始终得到相同的结果，前提是列表保持有序。（这与*交换律*不同。）

现在我们将看看PyCUDA如何处理类似于`reduce`的编程模式——使用**并行扫描**和**归约内核**。

# 并行扫描和归约内核基础

让我们看一下PyCUDA中一个复制reduce功能的基本函数——`InclusiveScanKernel`。（您可以在`simple_scankernal0.py`文件名下找到代码。）让我们执行一个在GPU上对一小组数字求和的基本示例：

```py
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.scan import InclusiveScanKernel
seq = np.array([1,2,3,4],dtype=np.int32)
seq_gpu = gpuarray.to_gpu(seq)
sum_gpu = InclusiveScanKernel(np.int32, "a+b")
print sum_gpu(seq_gpu).get()
print np.cumsum(seq)
```

我们通过首先指定输入/输出类型（这里是NumPy `int32`）和字符串`"a+b"`来构建我们的内核。在这里，`InclusiveScanKernel`自动在GPU空间中设置了名为`a`和`b`的元素，因此您可以将此字符串输入视为Python中的`lambda a,b: a + b`的类似物。我们实际上可以在这里放置任何（可结合的）二进制操作，只要我们记得用C语言编写它。

当我们运行`sum_gpu`时，我们会得到一个与输入数组大小相同的数组。数组中的每个元素表示计算中的每个步骤的值（我们可以看到，NumPy `cumsum`函数给出了相同的输出）。最后一个元素将是我们正在寻找的最终输出，对应于reduce的输出：

![](assets/98e28110-698a-4ab5-a827-9c1a8a2e31d4.png)

让我们尝试一些更具挑战性的东西；让我们找到一个`float32`数组中的最大值：

```py
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.scan import InclusiveScanKernel
seq = np.array([1,100,-3,-10000, 4, 10000, 66, 14, 21],dtype=np.int32)
seq_gpu = gpuarray.to_gpu(seq)
max_gpu = InclusiveScanKernel(np.int32, "a > b ? a : b")
print max_gpu(seq_gpu).get()[-1]
print np.max(seq)
```

（您可以在名为`simple_scankernal1.py`的文件中找到完整的代码。）

在这里，我们所做的主要更改是用`a > b ? a : b`替换了`a + b`字符串。 （在Python中，这将在`reduce`语句中呈现为`lambda a, b: max(a,b)`）。在这里，我们使用了一个技巧，使用C语言的`?`运算符来给出`a`和`b`中的最大值。最后，我们显示了输出数组中结果元素的最后一个值，这将恰好是最后一个元素（我们总是可以用Python中的`[-1]`索引来检索）。

现在，让我们最后再看一个用于生成GPU内核的PyCUDA函数——`ReductionKernel`。实际上，`ReductionKernel`的作用类似于`ElementwiseKernel`函数，后面跟着一个并行扫描内核。哪种算法是使用`ReductionKernel`实现的一个好选择？首先想到的是线性代数中的点积。让我们记住计算两个向量的点积有两个步骤：

1.  将向量逐点相乘

1.  对结果的逐点乘积求和

这两个步骤也称为*乘法和累加*。现在让我们设置一个内核来执行这个计算：

```py
dot_prod = ReductionKernel(np.float32, neutral="0", reduce_expr="a+b", map_expr="vec1[i]*vec2[i]", arguments="float *vec1, float *vec2") 
```

首先，注意我们为内核使用的数据类型（`float32`）。然后，我们使用`arguments`设置了我们的CUDA C内核的输入参数（这里是两个代表每个向量的浮点数组，用`float *`表示），并使用`map_expr`设置了逐点计算，这里是逐点乘法。与`ElementwiseKernel`一样，这是按`i`索引的。我们设置了`reduce_expr`，与`InclusiveScanKernel`一样。这将对数组执行元素操作的结果进行减少类型的操作。最后，我们使用`neutral`设置了*中性元素*。这是一个将作为`reduce_expr`的标识的元素；在这里，我们设置`neutral=0`，因为`0`在加法下始终是标识（在乘法下，1是标识）。稍后在本书中更深入地讨论并行前缀时，我们将看到为什么我们必须设置这个。

# 摘要

我们首先学习了如何从PyCUDA查询我们的GPU，并用此方法在Python中重新创建了CUDA的`deviceQuery`程序。然后我们学习了如何使用PyCUDA的`gpuarray`类及其`to_gpu`和`get`函数将NumPy数组传输到GPU的内存中。我们通过观察如何使用`gpuarray`对象来进行基本的GPU计算来感受了使用`gpuarray`对象的感觉，并且我们学会了使用IPython的`prun`分析器进行一些调查工作。我们发现，由于PyCUDA启动NVIDIA的`nvcc`编译器来编译内联CUDA C代码，有时在会话中首次运行PyCUDA的GPU函数时会出现一些任意的减速。然后我们学习了如何使用`ElementwiseKernel`函数来编译和启动逐元素操作，这些操作会自动并行化到GPU上。我们对Python中的函数式编程进行了简要回顾（特别是`map`和`reduce`函数），最后，我们介绍了如何使用`InclusiveScanKernel`和`ReductionKernel`函数在GPU上进行一些基本的reduce/scan类型计算。

现在我们已经掌握了编写和启动内核函数的绝对基础知识，我们应该意识到PyCUDA已经通过其模板为我们覆盖了编写内核的大部分开销。我们将在下一章学习CUDA内核执行的原则，以及CUDA如何将内核中的并发线程排列成抽象的**网格**和**块**。

# 问题

1.  在`simple_element_kernel_example0.py`中，我们在测量GPU计算时间时不考虑与GPU之间的内存传输。尝试使用Python时间命令测量`gpuarray`函数`to_gpu`和`get`的时间。考虑内存传输时间后，你会认为将这个特定函数卸载到GPU上值得吗？

1.  在[第1章](f9c54d0e-6a18-49fc-b04c-d44a95e011a2.xhtml)中，*为什么进行GPU编程？*，我们讨论了安德尔定律，这让我们对将程序的部分内容卸载到GPU上可能获得的收益有了一些了解。在本章中我们看到的两个问题，安德尔定律没有考虑到的是什么？

1.  修改`gpu_mandel0.py`以使用越来越小的复数格点，并将其与程序的CPU版本进行比较。我们可以选择足够小的格点，以至于CPU版本实际上比GPU版本更快吗？

1.  创建一个使用`ReductionKernel`的内核，该内核在GPU上获取两个相同长度的`complex64`数组，并返回两个数组中的绝对最大元素。

1.  如果一个`gpuarray`对象在Python中到达作用域的末尾会发生什么？

1.  你认为我们在使用`ReductionKernel`时为什么需要定义`neutral`？

1.  如果在`ReductionKernel`中我们设置`reduce_expr ="a > b ? a : b"`，并且我们正在操作int32类型，那么我们应该将"`neutral`"设置为什么？
