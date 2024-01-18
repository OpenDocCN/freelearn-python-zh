# 异构计算

本章将帮助我们通过Python语言探索**图形处理单元**（**GPU**）编程技术。GPU的不断演进揭示了这些架构如何为执行复杂计算带来巨大好处。

GPU当然不能取代CPU。然而，它们是一个结构良好的异构代码，能够利用两种类型处理器的优势，事实上，可以带来相当大的优势。

我们将研究异构编程的主要开发环境，即**PyCUDA**和**Numba**环境，用于**CUDA**和**PyOpenCL**环境，它们是Python版本的**OpenCL**框架。

在本章中，我们将涵盖以下内容：

+   理解异构计算

+   理解GPU架构

+   理解GPU编程

+   处理PyCUDA

+   使用PyCUDA进行异构编程

+   使用PyCUDA实现内存管理

+   介绍PyOpenCL

+   使用PyOpenCL构建应用程序

+   使用PyOpenCL进行逐元素表达

+   评估PyOpenCL应用程序

+   使用Numba进行GPU编程

让我们从详细了解异构计算开始。

# 理解异构计算

多年来，对越来越复杂计算的更好性能的追求导致了在计算机使用方面采用新技术。其中之一称为*异构计算*，旨在以一种有利于时间计算效率的方式与不同（或异构）处理器合作。

在这种情况下，主程序运行的处理器（通常是CPU）被称为“主机”，而协处理器（例如GPU）被称为“设备”。后者通常与主机物理上分离，并管理自己的内存空间，这也与主机的内存分开。

特别是，受到市场需求的影响，GPU已经演变成高度并行的处理器，将GPU从图形渲染设备转变为可并行化和计算密集型的通用计算设备。

事实上，除了在屏幕上渲染图形之外，使用GPU进行其他任务被称为异构计算。

最后，良好的GPU编程任务是充分利用图形卡提供的高级并行性和数学能力，并尽量减少它所带来的所有缺点，例如主机和设备之间的物理连接延迟。

# 理解GPU架构

GPU是用于矢量处理图形数据以从多边形基元渲染图像的专用CPU/核心。良好的GPU程序的任务是充分利用图形卡提供的高级并行性和数学能力，并尽量减少它所带来的所有缺点，例如主机和设备之间的物理连接延迟。

GPU具有高度并行的结构，可以以高效的方式处理大型数据集。这一特性与硬件性能程序的快速改进相结合，引起了科学界对使用GPU进行除了渲染图像之外的其他用途的关注。

GPU（参见下图）由称为**流多处理器**（**SMs**）的多个处理单元组成，代表了并行逻辑的第一级别。事实上，每个SM同时独立地工作：

![](assets/12715105-d093-49e4-8e05-cdb976bc755c.png)GPU架构

每个SM被分成一组**流处理器**（**SPs**），具有可以顺序运行线程的核心。SP代表执行逻辑的最小单位和更细粒度的并行级别。

为了最好地编程这种类型的架构，我们需要介绍GPU编程，这将在下一节中描述。

# 理解GPU编程

GPU已经变得越来越可编程。事实上，它们的指令集已经扩展，允许执行更多的任务。

如今，在GPU上，可以执行经典的CPU编程指令，如循环和条件，内存访问和浮点计算。两个主要的独立显卡制造商——**NVIDIA**和**AMD**——已经开发了他们的GPU架构，为开发人员提供了相关的开发环境，允许使用不同的编程语言，包括Python进行编程。

目前，开发人员在非纯粹与图形相关的环境中编程使用GPU的软件时，有宝贵的工具。在异构计算的主要开发环境中，我们有CUDA和OpenCL。

现在让我们详细看一下它们。

# CUDA

CUDA是NVIDIA的专有硬件架构，也是相关开发环境的名称。目前，CUDA拥有数十万活跃开发人员，这表明在并行编程环境中对这项技术的兴趣正在增长。

CUDA为最常用的编程语言提供扩展，包括Python。最知名的CUDA Python扩展如下：

+   PyCUDA ([https://mathema.tician.de/software/PyCUDA/](https://mathema.tician.de/software/pycuda/))

+   Numba ([http://numba.pydata.org](http://numba.pydata.org))

我们将在接下来的章节中使用这些扩展。

# OpenCL

并行计算中的第二个主角是OpenCL，与其NVIDIA对应物不同，它是开放标准，不仅可以与不同制造商的GPU一起使用，还可以与不同类型的微处理器一起使用。

然而，OpenCL是一个更完整和多功能的解决方案，因为它没有CUDA的成熟和简单易用。

OpenCL Python扩展是PyOpenCL ([https://mathema.tician.de/software/pyopencl/](https://mathema.tician.de/software/pyopencl/))。

在接下来的章节中，将分析CUDA和OpenCL编程模型及其Python扩展，并附带一些有趣的应用示例。

# 处理PyCUDA

PyCUDA是Andreas Klöckner提供的一个绑定库，通过它可以访问CUDA的Python API。其主要特点包括自动清理，与对象的生命周期相关联，从而防止泄漏，对模块和缓冲区的方便抽象，对驱动程序的完全访问以及内置的错误处理。它也非常轻巧。

该项目是根据MIT许可证开源的，文档非常清晰，而且在线可以找到许多不同的来源来提供帮助和支持。PyCUDA的主要目的是让开发人员以最小的抽象从Python调用CUDA，并支持CUDA元编程和模板化。

# 准备就绪

请按照Andreas Klöckner主页上的说明([https://mathema.tician.de/software/pycuda/](https://mathema.tician.de/software/pycuda/))安装PyCUDA。

下一个编程示例具有双重功能：

+   第一步是验证PyCUDA是否正确安装。

+   第二步是读取并打印GPU卡的特性。

# 如何做到...

让我们按照以下步骤进行：

1.  通过第一条指令，我们导入了Python驱动程序（即`pycuda.driver`）到我们PC上安装的CUDA库：

```py
import pycuda.driver as drv
```

1.  初始化CUDA。还要注意，在`pycuda.driver`模块中的任何其他指令之前必须调用以下指令：

```py
drv.init()
```

1.  枚举PC上的GPU卡数量：

```py
print ("%d device(s) found." % drv.Device.count())
```

1.  对于每个存在的GPU卡，打印设备的型号名称、计算能力和设备上的总内存量（以千字节为单位）：

```py
for ordinal i n range(drv.Device.count()): 
       dev = drv.Device(ordinal) 
       print ("Device #%d: %s" % (ordinal, dev.name()) 
       print ("Compute Capability: %d.%d"% dev.compute_capability()) 
       print ("Total Memory: %s KB" % (dev.total_memory()//(1024))) 
```

# 它是如何工作的...

执行非常简单。在第一行代码中，导入并初始化了`pycuda.driver`：

```py
import pycuda.driver as drv  
drv.init() 
```

`pycuda.driver`模块公开了CUDA编程接口的驱动级别，比CUDA C运行时级别的编程接口更灵活，并且具有一些运行时中不存在的功能。

然后，它循环进入`drv.Device.count()`函数，并且对于每个GPU卡，都会打印出卡的名称和其主要特征（计算能力和总内存）：

```py
print ("Device #%d: %s" % (ordinal, dev.name()))  
print ("Compute Capability: %d.%d" % dev.compute_capability()) 
print ("Total Memory: %s KB" % (dev.total_memory()//(1024))) 
```

执行以下代码：

```py
C:\>python dealingWithPycuda.py
```

完成后，安装的GPU将显示在屏幕上，如下例所示：

```py
1 device(s) found.
Device #0: GeForce GT 240
Compute Capability: 1.2
Total Memory: 1048576 KB
```

# 还有更多...

CUDA编程模型（因此也包括Python包装器PyCUDA）是通过对C语言标准库的特定扩展来实现的。这些扩展就像在标准C库中的函数调用一样创建，允许简单地处理包括主机和设备代码在内的异构编程模型。这两个逻辑部分的管理由`nvcc`编译器完成。

以下是其简要描述：

1.  *将*设备代码与主机代码分开。

1.  *调用*默认编译器（例如GCC）来编译主机代码。

1.  *构建*设备代码为二进制形式（`.cubin`对象）或汇编形式（`PTX`对象）：

![](assets/6c16c259-1075-4eb4-bf70-ad0b6ac78a12.png)

PyCUDA执行模型

PyCUDA在执行期间执行所有前述步骤，与CUDA应用程序相比，这会增加应用程序的加载时间。

# 另请参阅

+   CUDA编程指南可在此处找到：[https://docs.nvidia.com/CUDA/CUDA-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

+   PyCUDA文档可在此处找到：[https://documen.tician.de/PyCUDA/](https://documen.tician.de/pycuda/)

# 使用PyCUDA进行异构编程

CUDA编程模型（因此也包括PyCUDA的编程模型）旨在在CPU和GPU上共同执行软件应用程序，以便在CPU上执行应用程序的顺序部分，并在GPU上执行可以并行化的部分。不幸的是，计算机并不足够聪明，无法自主地理解如何分配代码，因此开发人员需要指示哪些部分应由CPU和GPU运行。

事实上，CUDA应用程序由串行组件和并行组件（称为内核）组成，串行组件由系统CPU或主机执行，而并行组件由GPU或设备执行。

内核被定义为*网格*，反过来可以分解为顺序分配给各个多处理器的块，从而实现*粗粒度并行*。在块内部，有一个基本的计算单元，线程，具有非常*细粒度的并行性*。一个线程只能属于一个块，并且由整个内核的唯一索引标识。为了方便起见，可以使用二维索引来表示块，三维索引来表示线程。内核之间是顺序执行的。另一方面，块和线程是并行执行的。运行的线程数量（并行）取决于它们在块中的组织以及它们对资源的请求，与设备中可用的资源相比。

要可视化先前表达的概念，请参考[https://sites.google.com/site/computationvisualization/programming/cuda/article1](https://sites.google.com/site/computationvisualization/programming/cuda/article1)中的（*图5*）。

块的设计旨在保证可伸缩性。事实上，如果您有一个具有两个多处理器的架构和另一个具有四个多处理器的架构，那么GPU应用程序可以在两个架构上执行，显然具有不同的时间和并行级别。

根据PyCUDA编程模型执行异构程序的结构如下：

1.  *在*主机上分配内存。

1.  *将*数据从主机内存传输到设备内存。

1.  *通过调用内核函数*运行设备。

1.  *将结果从设备内存传输到主机内存。

1.  *释放*在设备上分配的内存。

以下图表显示了根据PyCUDA编程模型的程序执行流程：

![](assets/d58c37c0-992d-4f12-b1e3-b152038611bc.png)PyCUDA编程模型

在下一个例子中，我们将通过一个具体的编程方法来构建PyCUDA应用程序。

# 如何做...

为了展示PyCUDA编程模型，我们考虑需要将5×5矩阵的所有元素加倍的任务：

1.  我们导入了执行所需的库：

```py
import PyCUDA.driver as CUDA 
import PyCUDA.autoinit 
from PyCUDA.compiler import SourceModule 
import numpy 
```

1.  我们导入的`numpy`库允许我们构建问题的输入，即一个5×5矩阵，其值是随机选择的：

```py
a = numpy.random.randn(5,5) 
a = a.astype(numpy.float32) 
```

1.  因此，构建的矩阵必须从主机内存复制到设备内存。为此，我们在设备上分配了一个内存空间（`a*_*gpu`），用于包含矩阵`a`。为此，我们使用了`mem_alloc`函数，其主题是分配的内存空间。特别是，由`a.nbytes`参数表示的矩阵`a`的字节数如下：

```py
a_gpu = cuda.mem_alloc(a.nbytes) 
```

1.  之后，我们可以使用`memcpy_htod`函数将矩阵从主机传输到设备上专门创建的内存区域：

```py
cuda.memcpy_htod(a_gpu, a) 
```

1.  在设备内部，`doubleMatrix`内核函数将运行。它的目的是将输入矩阵的每个元素乘以`2`。正如你所看到的，`doubleMatrix`函数的语法类似于C语言，而`SourceModule`语句是NVIDIA编译器（`nvcc`编译器）的真正指令，它创建了一个模块，这个模块只包含`doubleMatrix`函数：

```py
mod = SourceModule(""" 
  __global__ void doubles_matrix(float *a){ 
    int idx = threadIdx.x + threadIdx.y*4; 
    a[idx] *= 2;} 
  """)
```

1.  通过`func`参数，我们识别了`mod`模块中包含的`doubleMatrix`函数：

```py
func = mod.get_function("doubles_matrix") 
```

1.  最后，我们运行内核函数。为了成功地在设备上执行内核函数，CUDA用户必须指定内核的输入和执行线程块的大小。在下面的情况中，输入是先前复制到设备上的`a_gpu`矩阵，而线程块的维度是`(5,5,1)`：

```py
func(a_gpu, block=(5,5,1)) 
```

1.  因此，我们分配了一个大小等于输入矩阵`a`的内存区域：

```py
a_doubled = numpy.empty_like(a) 
```

1.  然后，我们将分配给设备的内存区域`a_gpu`的内容复制到先前定义的内存区域`a_doubled`中：

```py
cuda.memcpy_dtoh(a_doubled, a_gpu) 
```

1.  最后，我们打印输入矩阵`a`的内容和输出矩阵，以验证实现的质量：

```py
print ("ORIGINAL MATRIX") 
print (a) 
print ("DOUBLED MATRIX AFTER PyCUDA EXECUTION") 
print (a_doubled) 
```

# 它是如何工作的...

让我们首先看一下为这个例子导入了哪些库：

```py
import PyCUDA.driver as CUDA 
import PyCUDA.autoinit 
from PyCUDA.compiler import SourceModule 
```

特别是，`autoinit`导入自动识别我们系统上可用于执行的GPU，而`SourceModule`是NVIDIA编译器（`nvcc`）的指令，允许我们识别必须编译并上传到设备的对象。

然后，我们使用`numpy`库构建了5×5输入矩阵：

```py
import numpy 
a = numpy.random.randn(5,5) 
```

在这种情况下，矩阵中的元素被转换为单精度模式（因为执行此示例的图形卡仅支持单精度）：

```py
a = a.astype(numpy.float32) 
```

然后，我们将数组从主机复制到设备，使用以下两个操作：

```py
a_gpu = CUDA.mem_alloc(a.nbytes) 
CUDA.memcpy_htod(a_gpu, a) 
```

请注意，在执行内核函数期间，设备和主机内存可能永远不会进行通信。因此，为了在设备上并行执行内核函数，与内核函数相关的所有输入数据也必须存在于设备的内存中。

还应该注意，`a_gpu`矩阵是线性化的，即是一维的，因此我们必须对其进行管理。

此外，所有这些操作都不需要内核调用。这意味着它们是由主机直接执行的。

`SourceModule`实体允许定义`doubleMatrix`内核函数。`__global__`是`nvcc`指令，表示`doubleMatrix`函数将由设备处理：

```py
mod = SourceModule(""" 
  __global__ void doubleMatrix(float *a) 
```

让我们考虑内核的主体。`idx`参数是矩阵索引，由`threadIdx.x`和`threadIdx.y`线程坐标标识：

```py
 int idx = threadIdx.x + threadIdx.y*4; 
    a[idx] *= 2; 
```

然后，`mod.get_function("doubleMatrix")`返回`func`参数的标识符：

```py
func = mod.get_function("doubleMatrix ") 
```

为了执行内核，我们需要配置执行上下文。这意味着通过在`func`调用内部使用块参数来设置属于块网格的线程的三维结构：

```py
func(a_gpu, block = (5, 5, 1)) 
```

`block = (5, 5, 1)`告诉我们，我们正在调用一个具有`a_gpu`线性化输入矩阵和大小为`5`的单个线程块的内核函数（即在*x*方向上`*5*`个线程，在*y*方向上`*5*`个线程，在*z*方向上1个线程，总共*16*个线程）。请注意，每个线程执行相同的内核代码（总共25个线程）。

在GPU设备中计算后，我们使用数组来存储结果：

```py
a_doubled = numpy.empty_like(a) 
CUDA.memcpy_dtoh(a_doubled, a_gpu) 
```

要运行示例，请在命令提示符上键入以下内容：

```py
C:\>python heterogenousPycuda.py
```

输出应该是这样的：

```py
ORIGINAL MATRIX
[[-0.59975582 1.93627465 0.65337795 0.13205571 -0.46468592]
[ 0.01441949 1.40946579 0.5343408 -0.46614054 -0.31727529]
[-0.06868593 1.21149373 -0.6035406 -1.29117763 0.47762445]
[ 0.36176383 -1.443097 1.21592784 -1.04906416 -1.18935871]
[-0.06960868 -1.44647694 -1.22041082 1.17092752 0.3686313 ]] 
DOUBLED MATRIX AFTER PyCUDA EXECUTION
[[-1.19951165 3.8725493 1.3067559 0.26411143 -0.92937183]
[ 0.02883899 2.81893158 1.0686816 -0.93228108 -0.63455057]
[-0.13737187 2.42298746 -1.2070812 -2.58235526 0.95524889]
[ 0.72352767 -2.886194 2.43185568 -2.09812832 -2.37871742]
[-0.13921736 -2.89295388 -2.44082164 2.34185504 0.73726263 ]]
```

# 还有更多...

使得CUDA的关键特性与其他并行模型（通常在CPU上使用）根本不同的是，为了高效，它需要成千上万的线程处于活动状态。这是由GPU的典型结构实现的，它使用轻量级线程，并且还允许非常快速和高效地创建和修改执行上下文。

请注意，线程的调度直接与GPU架构及其固有的并行性相关联。事实上，一块线程分配给一个单个SM。在这里，线程进一步分成称为warp的组。属于同一warp的线程由*warp调度程序*管理。为了充分利用SM的固有并行性，同一warp的线程必须执行相同的指令。如果不满足这个条件，我们就称为*线程分歧*。

# 另请参阅

+   有关使用PyCUDA的完整教程，请访问以下网站：[https://documen.tician.de/pycuda/tutorial.html](https://documen.tician.de/pycuda/tutorial.html)。

+   在Windows 10上安装PyCUDA，请查看以下链接：[https://github.com/kdkoadd/Win10-PyCUDA-Install](https://github.com/kdkoadd/Win10-PyCUDA-Install)。

# 使用PyCUDA实现内存管理

PyCUDA程序应遵守由SM的结构和内部组织所规定的对线程性能的限制。事实上，GPU提供的各种类型的内存的知识和正确使用对于实现最大效率至关重要。在那些启用了CUDA使用的GPU卡中，有四种类型的内存，如下所示：

+   **寄存器**：每个线程被分配一个内存寄存器，只有分配的线程才能访问，即使线程属于同一块。

+   **共享内存**：每个块都有自己的共享内存，线程属于它。即使这个内存也非常快。

+   **常量内存**：网格中的所有线程都可以常量访问内存，但只能读取。其中的数据在整个应用程序的持续时间内存在。

+   **全局内存**：网格中的所有线程，因此所有内核都可以访问全局内存。此外，数据的持久性与常量内存完全相同：

![](assets/b43f49cb-fad9-4e64-8752-610540f62d30.png)

GPU内存模型

# 准备工作

为了获得最佳性能，PyCUDA程序必须充分利用每种类型的内存。特别是，它必须充分利用共享内存，最小化对全局内存的访问。

为了做到这一点，问题域通常被细分，以便一个线程块能够在一个封闭的数据子集中执行其处理。这样，操作单个块的线程将共同在同一个共享内存区域上工作，优化访问。

每个线程的基本步骤如下：

1.  *从全局内存加载数据到共享内存。

1.  *同步*块的所有线程，以便每个人都可以安全地读取其他线程填充的位置和共享内存。

1.  *处理*共享内存的数据。进行新的同步是必要的，以确保共享内存已经更新了结果。

1.  *将*结果写入全局内存。

为了澄清这种方法，在接下来的部分中，我们将介绍一个基于计算两个矩阵乘积的示例。

# 如何做...

下面的代码片段显示了在标准方法中计算两个矩阵*M×N*的乘积，这是基于顺序方法的。输出矩阵`P`的每个元素都是通过从矩阵`M`中取一个行元素和从矩阵`N`中取一个列元素得到的：

```py
void SequentialMatrixMultiplication(float*M,float *N,float *P, int width){ 
  for (int i=0; i< width; ++i) 
      for(int j=0;j < width; ++j) { 
          float sum = 0; 
          for (int k = 0 ; k < width; ++k) { 
              float a = M[I * width + k]; 
              float b = N[k * width + j]; 
              sum += a * b; 
                     } 
         P[I * width + j] = sum; 
    } 
} 
P[I * width + j] = sum; 
```

在这种情况下，如果每个线程都被赋予计算矩阵的每个元素的任务，那么对内存的访问将主导算法的执行时间。

我们可以依靠一个线程块来计算一个输出子矩阵。这样，访问相同内存块的线程合作优化访问，从而最小化总计算时间：

1.  第一步是加载实现算法所需的所有模块：

```py
import numpy as np 
from pycuda import driver, compiler, gpuarray, tools 
```

1.  然后，初始化GPU设备：

```py
import pycuda.autoinit 
```

1.  我们实现了`kernel_code_template`，它实现了两个矩阵的乘积，分别用`a`和`b`表示，而结果矩阵用参数`c`表示。注意，`MATRIX_SIZE`参数将在下一步中定义：

```py
kernel_code_template = """ 
__global__ void MatrixMulKernel(float *a, float *b, float *c) 
{ 
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    float Pvalue = 0; 
    for (int k = 0; k < %(MATRIX_SIZE)s; ++k) { 
        float Aelement = a[ty * %(MATRIX_SIZE)s + k]; 
        float Belement = b[k * %(MATRIX_SIZE)s + tx]; 
        Pvalue += Aelement * Belement; 
    } 
    c[ty * %(MATRIX_SIZE)s + tx] = Pvalue; 
}""" 
```

1.  以下参数将用于设置矩阵的维度。在这种情况下，大小为5×5：

```py
MATRIX_SIZE = 5
```

1.  我们定义两个输入矩阵，`a_cpu`和`b_cpu`，它们将包含随机浮点值：

```py
a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32) 
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
```

1.  然后，我们在主机设备上计算两个矩阵`a`和`b`的乘积：

```py
c_cpu = np.dot(a_cpu, b_cpu) 
```

1.  我们在设备（GPU）上分配了与输入矩阵大小相同的内存区域：

```py
a_gpu = gpuarray.to_gpu(a_cpu)  
b_gpu = gpuarray.to_gpu(b_cpu) 
```

1.  我们在GPU上分配了一个内存区域，大小与两个矩阵的乘积得到的输出矩阵相同。在这种情况下，得到的矩阵`c_gpu`的大小为5×5：

```py
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32) 
```

1.  以下的`kernel_code`重新定义了`kernel_code_template`，但设置了`matrix_size`参数：

```py
kernel_code = kernel_code_template % { 
    'MATRIX_SIZE': MATRIX_SIZE} 
```

1.  `SourceModule`指令告诉`nvcc`（*NVIDIA CUDA Compiler*）它将需要创建一个模块，其中包含先前定义的`kernel_code`：

```py
mod = compiler.SourceModule(kernel_code) 
```

1.  最后，我们从模块`mod`中取出`MatrixMulKernel`函数，给它起名为`matrixmul`：

```py
matrixmul = mod.get_function("MatrixMulKernel")
```

1.  我们执行两个矩阵`a_gpu`和`b_gpu`之间的乘积，得到`c_gpu`矩阵。线程块的大小被定义为`MATRIX_SIZE, MATRIX_SIZE, 1`：

```py
matrixmul( 
    a_gpu, b_gpu,  
    c_gpu,  
    block = (MATRIX_SIZE, MATRIX_SIZE, 1))
```

1.  打印输入矩阵：

```py
print ("-" * 80) 
print ("Matrix A (GPU):") 
print (a_gpu.get()) 
print ("-" * 80) 
print ("Matrix B (GPU):") 
print (b_gpu.get()) 
print ("-" * 80) 
print ("Matrix C (GPU):") 
print (c_gpu.get()) 
```

1.  为了检查在GPU上执行的计算的有效性，我们比较了两种实现的结果，一种是在主机设备（CPU）上执行的，另一种是在设备（GPU）上执行的。为此，我们使用了`numpy allclose`指令，它验证了两个逐元素数组在容差为`1e-05`的情况下是否相等：

```py
np.allclose(c_cpu, c_gpu.get()) 
```

# 它是如何工作的...

考虑PyCUDA编程工作流程。准备输入矩阵、输出矩阵以及存储结果的位置：

```py
MATRIX_SIZE = 5 
a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32) 
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32) 
c_cpu = np.dot(a_cpu, b_cpu) 
```

然后，我们使用`gpuarray.to_gpu()` PyCUDA函数将这些矩阵传输到GPU设备：

```py
a_gpu = gpuarray.to_gpu(a_cpu)  
b_gpu = gpuarray.to_gpu(b_cpu) 
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32) 
```

算法的核心是以下的核函数。需要指出的是，`__global__`关键字指定这个函数是一个核函数，这意味着它将由设备（GPU）执行，而这是在主机代码（CPU）调用后执行的：

```py
__global__ void MatrixMulKernel(float *a, float *b, float *c){
 int tx = threadIdx.x;
 int ty = threadIdx.y;
 float Pvalue = 0;
 for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
 float Aelement = a[ty * %(MATRIX_SIZE)s + k];
 float Belement = b[k * %(MATRIX_SIZE)s + tx];
 Pvalue += Aelement * Belement;}
 c[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
}
```

`threadIdx.x`和`threadIdy.y`是坐标，允许在二维块网格中识别线程。请注意，网格块内的线程执行相同的内核代码，但是在不同的数据片段上。如果我们将并行版本与顺序版本进行比较，那么我们立即注意到循环索引*i*和*j*已被`threadIdx.x`和`threadIdx.y`索引所取代。

这意味着在并行版本中，我们只会有一个循环迭代。实际上，`MatrixMulKernel`内核将在一个5×5并行线程的网格上执行。

这个条件在下图中表示：

![](assets/65aa99d1-d699-4329-883c-543cb7ef15de.png)

示例的网格和线程块组织

然后，我们通过比较两个结果矩阵来验证产品计算：

```py
np.allclose(c_cpu, c_gpu.get())
```

输出如下：

```py
C:\>python memManagementPycuda.py

---------------------------------------------------------------------
Matrix A (GPU):
[[ 0.90780383 -0.4782407 0.23222363 -0.63184392 1.05509627]
 [-1.27266967 -1.02834761 -0.15528528 -0.09468858 1.037099 ]
 [-0.18135822 -0.69884419 0.29881889 -1.15969539 1.21021318]
 [ 0.20939326 -0.27155793 -0.57454145 0.1466181 1.84723163]
 [ 1.33780348 -0.42343542 -0.50257754 -0.73388749 -1.883829 ]]
---------------------------------------------------------------------
Matrix B (GPU):
[[ 0.04523897 0.99969769 -1.04473436 1.28909719 1.10332143]
 [-0.08900332 -1.3893919 0.06948703 -0.25977209 -0.49602833]
 [-0.6463753 -1.4424541 -0.81715286 0.67685211 -0.94934392]
 [ 0.4485206 -0.77086055 -0.16582981 0.08478995 1.26223004]
 [-0.79841441 -0.16199949 -0.35969591 -0.46809086 0.20455229]]
---------------------------------------------------------------------
Matrix C (GPU):
[[-1.19226956 1.55315971 -1.44614291 0.90420711 0.43665022]
 [-0.73617989 0.28546685 1.02769876 -1.97204924 -0.65403283]
 [-1.62555301 1.05654192 -0.34626681 -0.51481217 -1.35338223]
 [-1.0040834 1.00310731 -0.4568972 -0.90064859 1.47408712]
 [ 1.59797418 3.52156591 -0.21708387 2.31396151 0.85150564]]
---------------------------------------------------------------------

TRUE
```

# 还有更多...

在单线程块中，共享内存中分配的数据的可见性有限。很容易看出，PyCUDA编程模型适用于特定类别的应用程序。

特别是这些应用程序必须具备的特征涉及到许多数学运算，具有高度的数据并行性（即在大量数据上重复相同操作的序列）。

具有这些特征的应用领域都属于以下科学领域：密码学、计算化学以及图像和信号分析。

# 另请参阅

+   可以在以下链接找到更多使用PyCUDA的示例：[https://github.com/zamorays/miniCursoPycuda](https://github.com/zamorays/miniCursoPycuda)。

# 介绍PyOpenCL

PyOpenCL是PyCUDA的姊妹项目。它是一个绑定库，可以从Python完全访问OpenCL的API，也是由Andreas Klöckner开发的。它具有许多与PyCUDA相同的概念，包括对超出范围对象的清理、对数据结构的部分抽象和错误处理，而且开销很小。该项目在MIT许可下可用；其文档非常好，网上可以找到大量指南和教程。

PyOpenCL的主要重点是提供Python和OpenCL之间的轻量级连接，但它还包括对模板和元程序的支持。PyOpenCL程序的流程几乎与OpenCL的C或C++程序完全相同。主机程序准备调用设备程序，启动它，然后等待结果。

# 准备工作

PyOpenCL安装的主要参考资料是Andreas Klöckner的主页：[https://mathema.tician.de/software/pyopencl/](https://mathema.tician.de/software/pyopencl/)。

如果您正在使用Anaconda，则建议执行以下步骤：

1.  从以下链接安装最新的Anaconda发行版，其中包括Python 3.7：[https://www.anaconda.com/distribution/#download-section](https://www.anaconda.com/distribution/#download-section)。对于本节，已安装了Windows Installer的Anaconda 2019.07。

1.  从此链接获取PyOpenCL预构建二进制文件，链接为：[https://www.lfd.uci.edu/~gohlke/pythonlibs/](https://www.lfd.uci.edu/~gohlke/pythonlibs/)。选择正确的OS和CPython版本组合。在这里，我们使用`pyopencl-2019.1+cl12-cp37-cp37m-win_amd64.whl`。

1.  使用`pip`来安装之前的软件包。只需在Anaconda Prompt中输入以下内容：

```py
**(base) C:\> pip install <directory>\pyopencl-2019.1+cl12-cp37-cp37m-win_amd64.whl** 
```

`<directory>`是PyOpenCL软件包所在的文件夹。

此外，以下符号表示我们正在使用Anaconda Prompt：

```py
**(base) C:\>**
```

# 操作步骤如下...

在以下示例中，我们将使用PyOpenCL的一个函数来列举它将运行的GPU的特性。

我们实现的代码非常简单和逻辑：

1.  在第一步中，我们导入`pyopencl`库：

```py
import pyopencl as cl
```

1.  我们构建一个函数，其输出将为我们提供正在使用的GPU硬件的特征：

```py
def print_device_info() :
 print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
 for platform in cl.get_platforms():
 print('=' * 60)
 print('Platform - Name: ' + platform.name)
 print('Platform - Vendor: ' + platform.vendor)
 print('Platform - Version: ' + platform.version)
 print('Platform - Profile: ' + platform.profile)

 for device in platform.get_devices():
 print(' ' + '-' * 56)
 print(' Device - Name: ' \
 + device.name)
 print(' Device - Type: ' \
 + cl.device_type.to_string(device.type))
 print(' Device - Max Clock Speed: {0} Mhz'\
 .format(device.max_clock_frequency))
 print(' Device - Compute Units: {0}'\
 .format(device.max_compute_units))
 print(' Device - Local Memory: {0:.0f} KB'\
 .format(device.local_mem_size/1024.0))
 print(' Device - Constant Memory: {0:.0f} KB'\
 .format(device.max_constant_buffer_size/1024.0))
 print(' Device - Global Memory: {0:.0f} GB'\
 .format(device.global_mem_size/1073741824.0))
 print(' Device - Max Buffer/Image Size: {0:.0f} MB'\
 .format(device.max_mem_alloc_size/1048576.0))
 print(' Device - Max Work Group Size: {0:.0f}'\
 .format(device.max_work_group_size))
 print('\n')
```

1.  因此，我们实现了`main`函数，该函数调用了先前实现的`print_device_info`函数：

```py
if __name__ == "__main__":
 print_device_info()
```

# 它是如何工作的...

以下命令用于导入`pyopencl`库：

```py
import pyopencl as cl
```

这使我们可以使用**`get_platforms`**方法，该方法返回一个平台实例列表，即系统中设备的列表：

```py
for platform in cl.get_platforms():
```

然后，对于找到的每个设备，显示以下主要特性：

+   名称和设备类型

+   最大时钟速度

+   计算单元

+   本地/常量/全局内存

此示例的输出如下：

```py
(base) C:\>python deviceInfoPyopencl.py

=============================================================
OpenCL Platforms and Devices
============================================================
Platform - Name: NVIDIA CUDA
Platform - Vendor: NVIDIA Corporation
Platform - Version: OpenCL 1.2 CUDA 10.1.152
Platform - Profile: FULL_PROFILE
 --------------------------------------------------------
 Device - Name: GeForce 840M
 Device - Type: GPU
 Device - Max Clock Speed: 1124 Mhz
 Device - Compute Units: 3
 Device - Local Memory: 48 KB
 Device - Constant Memory: 64 KB
 Device - Global Memory: 2 GB
 Device - Max Buffer/Image Size: 512 MB
 Device - Max Work Group Size: 1024
============================================================
Platform - Name: Intel(R) OpenCL
Platform - Vendor: Intel(R) Corporation
Platform - Version: OpenCL 2.0
Platform - Profile: FULL_PROFILE
 --------------------------------------------------------
 Device - Name: Intel(R) HD Graphics 5500
 Device - Type: GPU
 Device - Max Clock Speed: 950 Mhz
 Device - Compute Units: 24
 Device - Local Memory: 64 KB
 Device - Constant Memory: 64 KB
 Device - Global Memory: 3 GB
 Device - Max Buffer/Image Size: 808 MB
 Device - Max Work Group Size: 256
 --------------------------------------------------------
 Device - Name: Intel(R) Core(TM) i7-5500U CPU @ 2.40GHz
 Device - Type: CPU
 Device - Max Clock Speed: 2400 Mhz
 Device - Compute Units: 4
 Device - Local Memory: 32 KB
 Device - Constant Memory: 128 KB
 Device - Global Memory: 8 GB
 Device - Max Buffer/Image Size: 2026 MB
 Device - Max Work Group Size: 8192
```

# 还有更多...

OpenCL目前由Khronos Group管理，这是一个非营利性公司联盟，他们合作定义了这个（以及许多其他）标准的规范和符合参数，用于为每种类型的平台创建特定于OpenCL的驱动程序。

这些驱动程序还提供了用于编译使用内核语言编写的程序的函数：这些函数被转换为通常是特定于供应商的某种形式的中间语言中的程序，然后在参考架构上执行。

有关OpenCL的更多信息可以在以下链接找到：[https://www.khronos.org/registry/OpenCL/](https://www.khronos.org/registry/OpenCL/)。

# 另请参阅

+   PyOpenCL文档可在此处找到：[https://documen.tician.de/pyopencl/](https://documen.tician.de/pyopencl/)。

+   PyOpenCL的最佳介绍之一，即使有点过时，可以在以下链接找到：[http://www.drdobbs.com/open-source/easy-opencl-with-python/240162614](http://www.drdobbs.com/open-source/easy-opencl-with-python/240162614)。

# 使用PyOpenCL构建应用程序

为PyOpenCL构建程序的第一步是编写主机应用程序。这是在CPU上执行的，其任务是管理可能在GPU卡（即设备）上执行内核。

*内核*是可执行代码的基本单位，类似于C函数。它可以是数据并行或任务并行。然而，PyOpenCL的基石是利用并行性。

一个基本概念是*程序*，它是一组内核和其他函数，类似于动态库。因此，我们可以将内核中的指令分组，并将不同的内核分组到一个程序中。

程序可以从应用程序中调用。我们有执行队列，指示内核执行的顺序。但是，在某些情况下，这些可以在不遵循原始顺序的情况下启动。

最后，我们可以列出使用PyOpenCL开发应用程序的基本元素：

+   **设备**：这标识了内核代码要在其中执行的硬件。请注意，PyOpenCL应用程序可以在CPU和GPU板上运行（以及PyCUDA），还可以在嵌入式设备（如**可编程门阵列**（**FPGAs**））上运行。

+   **程序**：这是一组内核，其任务是选择在设备上运行哪个内核。

+   **内核**：这是要在设备上执行的代码。内核是类似C的函数，这意味着它可以在支持PyOpenCL驱动程序的任何设备上编译。

+   **命令队列**：这在设备上对内核的执行进行排序。

+   **上下文**：这是一组设备，允许设备接收内核并传输数据。

以下图表显示了此数据结构如何在主机应用程序中工作：

![](assets/60b38941-28e9-4b2f-a920-d59c4f426b1e.png)

PyOpenCL编程模型

再次，我们观察到一个程序可以包含更多的函数在设备上运行，并且每个内核仅封装了程序中的单个函数。

# 如何做到这一点...

在以下示例中，我们展示了使用PyOpenCL构建应用程序的基本步骤：要执行的任务是两个向量的求和。为了有一个可读的输出，我们将考虑每个具有100个元素的两个向量：结果向量的每个第i个元素将等于**`vector_a`**的第i个元素加上**`vector_b`**的第i个元素的和：

1.  让我们从导入所有必要的库开始：

```py
import numpy as np 
import pyopencl as cl 
import numpy.linalg as la 
```

1.  我们定义要相加的向量的大小，如下所示：

```py
vector_dimension = 100 
```

1.  在这里，定义了输入向量`vector_a`和`vector_b`：

```py
vector_a = np.random.randint(vector_dimension,size=vector_dimension) 
vector_b = np.random.randint(vector_dimension,size=vector_dimension) 
```

1.  接着，我们定义**`platform`**、**`device`**、**`context`**和**`queue`**：

```py
platform = cl.get_platforms()[1] 
device = platform.get_devices()[0] 
context = cl.Context([device]) 
queue = cl.CommandQueue(context) 
```

1.  现在，是时候组织将包含输入向量的内存区域了：

```py
mf = cl.mem_flags 
a_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,\ hostbuf=vector_a) 
b_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,\ hostbuf=vector_b) 
```

1.  最后，我们使用`Program`方法构建应用程序内核：

```py
program = cl.Program(context, """ 
__kernel void vectorSum(__global const int *a_g, __global const int *b_g, __global int *res_g) { 
  int gid = get_global_id(0); 
  res_g[gid] = a_g[gid] + b_g[gid]; 
} 
""").build()
```

1.  然后，我们分配了结果矩阵的内存：

```py
res_g = cl.Buffer(context, mf.WRITE_ONLY, vector_a.nbytes) 
```

1.  然后，我们调用内核函数：

```py
program.vectorSum(queue, vector_a.shape, None, a_g, b_g, res_g) 
```

1.  用于存储结果的内存空间在主机内存区域中分配（`res_np`）：

```py
res_np = np.empty_like(vector_a) 
```

1.  将计算结果复制到创建的内存区域中：

```py
cl._enqueue_copy(queue, res_np, res_g) 
```

1.  最后，我们打印结果：

```py
print ("PyOPENCL SUM OF TWO VECTORS") 
print ("Platform Selected = %s" %platform.name ) 
print ("Device Selected = %s" %device.name) 
print ("VECTOR LENGTH = %s" %vector_dimension) 
print ("INPUT VECTOR A") 
print (vector_a) 
print ("INPUT VECTOR B") 
print (vector_b) 
print ("OUTPUT VECTOR RESULT A + B ") 
print (res_np) 
```

1.  然后，我们进行了简单的检查，以验证求和操作是否正确：

```py
assert(la.norm(res_np - (vector_a + vector_b))) < 1e-5 
```

# 它是如何工作的...

在接下来的几行中，在相关的导入之后，我们定义输入向量*：*

```py
vector_dimension = 100 
vector_a = np.random.randint(vector_dimension, size= vector_dimension) 
vector_b = np.random.randint(vector_dimension, size= vector_dimension) 
```

每个向量包含100个整数项，这些项是通过`numpy`函数随机选择的：

```py
np.random.randint(max integer , size of the vector) 
```

然后，我们使用`get_platform()`方法选择平台来进行计算：

```py
platform = cl.get_platforms()[1] 
```

然后，选择相应的设备。这里，`platform.get_devices()[0]`对应于Intel(R) HD Graphics 5500显卡：

```py
device = platform.get_devices()[0]
```

在接下来的步骤中，定义了上下文和队列；PyOpenCL提供了上下文（选择的设备）和队列（选择的上下文）的方法：

```py
context = cl.Context([device]) 
queue = cl.CommandQueue(context) 
```

为了在所选设备中执行计算，将输入向量复制到设备的内存中：

```py
mf = cl.mem_flags 
a_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,\
hostbuf=vector_a) 
b_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,\
 hostbuf=vector_b) 
```

然后，我们为结果向量准备缓冲区：

```py
res_g = cl.Buffer(context, mf.WRITE_ONLY, vector_a.nbytes) 
```

在这里，定义了内核代码：

```py
program = cl.Program(context, """ 
__kernel void vectorSum(__global const int *a_g, __global const int *b_g, __global int *res_g) { 
  int gid = get_global_id(0); 
  res_g[gid] = a_g[gid] + b_g[gid];} 
""").build()
```

`vectorSum`是内核的名称，参数列表定义了输入参数和输出数据类型的数据类型（都是整数向量）。在内核主体内，两个向量的和定义如下步骤：

1.  *初始化*向量的索引：`int gid = get_global_id(0)`。

1.  *求和*向量的分量：`res_g[gid] = a_g[gid] + b_g[gid]`。

在OpenCL（因此在PyOpenCL中），缓冲区附加到上下文（[https://documen.tician.de/pyopencl/runtime.html#pyopencl.Context](https://documen.tician.de/pyopencl/runtime.html#pyopencl.Context)），一旦缓冲区在设备上使用，就会移动到设备上。

最后，我们在设备中执行`vectorSum`：

```py
program.vectorSum(queue, vector_a.shape, None, a_g, b_g, res_g)
```

为了检查结果，我们使用`assert`语句。这会测试结果，并在条件为假时触发错误：

```py
assert(la.norm(res_np - (vector_a + vector_b))) < 1e-5
```

输出应该如下所示：

```py
(base) C:\>python vectorSumPyopencl.py 
PyOPENCL SUM OF TWO VECTORS
Platform Selected = Intel(R) OpenCL
Device Selected = Intel(R) HD Graphics 5500
VECTOR LENGTH = 100
INPUT VECTOR A
 [45 46 0 97 96 98 83 7 51 21 72 70 59 65 79 92 98 24 56 6 70 64 59 0
 96 78 15 21 4 89 14 66 53 20 34 64 48 20 8 53 82 66 19 53 11 17 39 11
 89 97 51 53 7 4 92 82 90 78 31 18 72 52 44 17 98 3 36 69 25 87 86 68
 85 16 58 4 57 64 97 11 81 36 37 21 51 22 17 6 66 12 80 50 77 94 6 70
 21 86 80 69]
 INPUT VECTOR B
[25 8 76 57 86 96 58 89 26 31 28 92 67 47 72 64 13 93 96 91 91 36 1 75
 2 40 60 49 24 40 23 35 80 60 61 27 82 38 66 81 95 79 96 23 73 19 5 43
 2 47 17 88 46 76 64 82 31 73 43 17 35 28 48 89 8 61 23 17 56 7 84 36
 95 60 34 9 4 5 74 59 6 89 84 98 25 50 38 2 3 43 64 96 47 79 12 82
 72 0 78 5]
 OUTPUT VECTOR RESULT A + B
[70 54 76 154 182 194 141 96 77 52 100 162 126 112 151 156 111 117 152 
 97 161 100 60 75 98 118 75 70 28 129 37 101 133 80 95 91 130 58 74 134 
 177 145 115 76 84 36 44 54 91 144 68 141 53 80 156 164 121 151 74 35 
 107 80 92 106 106 64 59 86 81 94 170 104 80 76 92 13 61 69 171 70 87 
 125 121 119 76 72 55 8 69 55 144 146 124 173 18 152 93 86 158 74] 
```

# 还有更多...

在本节中，我们已经看到PyOpenCL执行模型，就像PyCUDA一样，涉及一个管理一个或多个异构设备的主机处理器。特别是，每个PyOpenCL命令以源代码的形式从主机发送到设备，该源代码是通过内核函数定义的。

然后，将源代码加载到参考架构的程序对象中，将程序编译成参考架构，并创建与程序相关的内核对象。

内核对象可以在可变数量的工作组中执行，创建一个*n*维计算矩阵，使其能够有效地将问题的工作负载在*n*维（1、2或3）中进行有效划分。这些工作组又由多个并行工作项组成。

根据设备的并行计算能力平衡每个工作组的工作负载是实现良好应用程序性能的关键参数之一。

工作负载的错误平衡，以及每个设备的特定特性（如传输延迟、吞吐量和带宽），可能会导致性能的大幅损失，或者在执行时没有考虑任何动态获取设备计算能力信息的系统时，会损害代码的可移植性。

然而，准确使用这些技术可以通过结合不同计算单元的计算结果来达到高水平的性能。

# 另请参阅

有关PyOpenCL编程的更多信息，请访问[https://pydanny-event-notes.readthedocs.io/en/latest/PyConPL2012/async_via_pyopencl.html](https://pydanny-event-notes.readthedocs.io/en/latest/PyConPL2012/async_via_pyopencl.html)。

# 使用PyOpenCL进行逐元素表达式

逐元素功能允许我们在单个计算步骤中对复杂表达式（由更多操作数组成）进行评估。

# 入门指南

`ElementwiseKernel(context, argument, operation, name, optional_parameters)`方法在PyOpenCL中实现以处理逐元素表达式。

主要参数如下：

+   `context`是将执行逐元素操作的设备或设备组。

+   `argument`是计算中涉及的所有参数的类似C的参数列表。

+   `operation`是表示要在参数列表上执行的操作的字符串。

+   `name`是与`Elementwisekernel`关联的内核名称。

+   `optional_parameters`在此示例中并不重要。

# 操作步骤...

在这里，我们考虑再次添加两个整数向量的任务：

1.  开始导入相关库：

```py
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
```

1.  定义上下文元素（`context`）和命令队列（`queue`）：

```py
context = cl.create_some_context()
queue = cl.CommandQueue(context)
```

1.  在这里，我们设置了输入和输出向量的向量维度和空间分配：

```py
vector_dim = 100 
vector_a=cl_array.to_device(queue,np.random.randint(100,\
size=vector_dim)) 
vector_b = cl_array.to_device(queue,np.random.randint(100,\ 
size=vector_dim)) 
result_vector = cl_array.empty_like(vector_a) 
```

1.  我们将`elementwiseSum`设置为`ElementwiseKernel`的应用程序，然后将其设置为一组定义要应用于输入向量的操作的参数：

```py
elementwiseSum = cl.elementwise.ElementwiseKernel(context, "int *a,\
int *b, int *c", "c[i] = a[i] + b[i]", "sum")
elementwiseSum(vector_a, vector_b, result_vector)
```

1.  最后，我们打印结果：

```py
print ("PyOpenCL ELEMENTWISE SUM OF TWO VECTORS")
print ("VECTOR LENGTH = %s" %vector_dimension)
print ("INPUT VECTOR A")
print (vector_a)
print ("INPUT VECTOR B")
print (vector_b)
print ("OUTPUT VECTOR RESULT A + B ")
print (result_vector)
```

# 工作原理...

在脚本的前几行中，我们导入了所有请求的模块。

为了初始化上下文，我们使用`cl.create_some_context()`方法。这会询问用户必须使用哪个上下文来执行计算：

```py
Choose platform:
[0] <pyopencl.Platform 'NVIDIA CUDA' at 0x1c0a25aecf0>
[1] <pyopencl.Platform 'Intel(R) OpenCL' at 0x1c0a2608400>
```

然后，我们需要实例化将接收`ElementwiseKernel`的队列：

```py
queue = cl.CommandQueue(context)
```

输入和输出向量被实例化。输入向量`vector_a`和`vector_b`是使用`random.randint` NumPy函数获得的随机值的整数向量。然后使用PyOpenCL语句将这些向量复制到设备中：

```py
cl.array_to_device(queue,array)
```

在`ElementwiseKernel`中，创建了一个对象：

```py
elementwiseSum = cl.elementwise.ElementwiseKernel(context,\
 "int *a, int *b, int *c", "c[i] = a[i] + b[i]", "sum")
```

请注意，所有参数都以C参数列表的形式的字符串格式化（它们都是整数）。操作是类似C的代码片段，执行操作，即输入向量元素的和。将用于编译内核的函数的名称是`sum`。

最后，我们使用之前定义的参数调用`elementwiseSum`函数：

```py
elementwiseSum(vector_a, vector_b, result_vector)
```

示例最后通过打印输入向量和获得的结果结束。输出如下所示：

```py
(base) C:\>python elementwisePyopencl.py

Choose platform:
[0] <pyopencl.Platform 'NVIDIA CUDA' at 0x1c0a25aecf0>
[1] <pyopencl.Platform 'Intel(R) OpenCL' at 0x1c0a2608400>
Choice [0]:1 
Choose device(s):
[0] <pyopencl.Device 'Intel(R) HD Graphics 5500' on 'Intel(R) OpenCL' at 0x1c0a1640db0>
[1] <pyopencl.Device 'Intel(R) Core(TM) i7-5500U CPU @ 2.40GHz' on 'Intel(R) OpenCL' at 0x1c0a15e53f0>
Choice, comma-separated [0]:0 PyOpenCL ELEMENTWISE SUM OF TWO VECTORS
VECTOR LENGTH = 100
INPUT VECTOR A
[24 64 73 37 40 4 41 85 19 90 32 51 6 89 98 56 97 53 34 91 82 89 97 2
 54 65 90 90 91 75 30 8 62 94 63 69 31 99 8 18 28 7 81 72 14 53 91 80
 76 39 8 47 25 45 26 56 23 47 41 18 89 17 82 84 10 75 56 89 71 56 66 61
 58 54 27 88 16 20 9 61 68 63 74 84 18 82 67 30 15 25 25 3 93 36 24 27
 70 5 78 15] 
INPUT VECTOR B
[49 18 69 43 51 72 37 50 79 34 97 49 51 29 89 81 33 7 47 93 70 52 63 90
 99 95 58 33 41 70 84 87 20 83 74 43 78 34 94 47 89 4 30 36 34 56 32 31
 56 22 50 52 68 98 52 80 14 98 43 60 20 49 15 38 74 89 99 29 96 65 89 41
 72 53 89 31 34 64 0 47 87 70 98 86 41 25 34 10 44 36 54 52 54 86 33 38
 25 49 75 53] 
OUTPUT VECTOR RESULT A + B
[73 82 142 80 91 76 78 135 98 124 129 100 57 118 187 137 130 60 81 184 
 152 141 160 92 153 160 148 123 132 145 114 95 82 177 137 112 109 133 
 102 65 117 11 111 108 48 109 123 111 132 61 58 99 93 143 78 136 37 145 
 84 78 109 66 97 122 84 164 155 118 167 121 155 102 130 107 116 119 50 
 84 9 108 155 133 172 170 59 107 101 40 59 61 79 55 147 122 57 65 
 95 54 153 68] 
```

# 还有更多...

PyCUDA也具有逐元素功能：

```py
ElementwiseKernel(arguments,operation,name,optional_parameters)
```

此功能与为PyOpenCL构建的函数几乎具有相同的参数，除了上下文参数。通过PyCUDA实现的本节中的相同示例具有以下列表：

```py
import pycuda.autoinit 
import numpy 
from pycuda.elementwise import ElementwiseKernel 
import numpy.linalg as la 

vector_dimension=100 
input_vector_a = np.random.randint(100,size= vector_dimension) 
input_vector_b = np.random.randint(100,size= vector_dimension) 
output_vector_c = gpuarray.empty_like(input_vector_a) 

elementwiseSum = ElementwiseKernel(" int *a, int * b, int *c",\ 
                             "c[i] = a[i] + b[i]"," elementwiseSum ") 
elementwiseSum(input_vector_a, input_vector_b,output_vector_c) 

print ("PyCUDA ELEMENTWISE SUM OF TWO VECTORS") 
print ("VECTOR LENGTH = %s" %vector_dimension) 
print ("INPUT VECTOR A") 
print (vector_a) 
print ("INPUT VECTOR B") 
print (vector_b) 
print ("OUTPUT VECTOR RESULT A + B ") 
print (result_vector) 
```

# 另请参阅

在以下链接中，您将找到PyOpenCL应用程序的有趣示例：[https://github.com/romanarranz/PyOpenCL](https://github.com/romanarranz/PyOpenCL)。

# 评估PyOpenCL应用程序

在本节中，我们将使用PyOpenCL库对CPU和GPU之间的性能进行比较测试。

事实上，在研究要实现的算法的性能之前，了解所拥有的计算平台所提供的计算优势也是很重要的。

# 入门指南

计算系统的特定特征会干扰计算时间，因此它们代表了一个非常重要的方面。

在以下示例中，我们将进行一项测试，以便监视系统的性能：

+   GPU：GeForce 840 M

+   CPU：Intel Core i7 – 2.40 GHz

+   RAM：8 GB

# 操作步骤...

在以下测试中，将评估并比较数学运算的计算时间，例如两个具有浮点元素的向量的求和。为了进行比较，将在两个单独的函数上执行相同的操作。

第一个函数仅由CPU计算，而第二个函数是通过使用PyOpenCL库编写的，以使用GPU卡。测试是在大小为10,000个元素的向量上执行的。

以下是代码：

1.  导入相关库。注意导入`time`库以计算计算时间，以及`linalg`库，它是`numpy`库的线性代数工具：

```py
from time import time 
import pyopencl as cl   
import numpy as np    
import deviceInfoPyopencl as device_info 
import numpy.linalg as la 
```

1.  然后，我们定义输入向量。它们都包含`10000`个浮点数的随机元素：

```py
a = np.random.rand(10000).astype(np.float32) 
b = np.random.rand(10000).astype(np.float32) 
```

1.  以下函数计算两个向量在CPU（主机）上的和：

```py
def test_cpu_vector_sum(a, b): 
    c_cpu = np.empty_like(a) 
    cpu_start_time = time() 
    for i in range(10000): 
            for j in range(10000): 
                    c_cpu[i] = a[i] + b[i] 
    cpu_end_time = time() 
    print("CPU Time: {0} s".format(cpu_end_time - cpu_start_time)) 
    return c_cpu 
```

1.  以下函数计算两个向量在GPU（设备）上的和：

```py
def test_gpu_vector_sum(a, b): 
    platform = cl.get_platforms()[0] 
    device = platform.get_devices()[0] 
    context = cl.Context([device]) 
    queue = cl.CommandQueue(context,properties=\
 cl.command_queue_properties.PROFILING_ENABLE)
```

1.  在`test_gpu_vector_sum`函数中，我们准备内存缓冲区来包含输入向量和输出向量：

```py
 a_buffer = cl.Buffer(context,cl.mem_flags.READ_ONLY \ 
                | cl.mem_flags.COPY_HOST_PTR, hostbuf=a) 
    b_buffer = cl.Buffer(context,cl.mem_flags.READ_ONLY \ 
                | cl.mem_flags.COPY_HOST_PTR, hostbuf=b) 
    c_buffer = cl.Buffer(context,cl.mem_flags.WRITE_ONLY, b.nbytes) 
```

1.  同样，在`test_gpu_vector_sum`函数中，我们定义了将在设备上计算两个向量的和的内核：

```py
 program = cl.Program(context, """ 
    __kernel void sum(__global const float *a,\ 
                      __global const float *b,\ 
                      __global float *c){ 
        int i = get_global_id(0); 
        int j; 
        for(j = 0; j < 10000; j++){ 
            c[i] = a[i] + b[i];} 
    }""").build() 
```

1.  然后，在开始计算之前，我们重置`gpu_start_time`变量。之后，我们计算两个向量的和，然后评估计算时间：

```py
 gpu_start_time = time() 
    event = program.sum(queue, a.shape, None,a_buffer, b_buffer,\ 
 c_buffer) 
    event.wait() 
    elapsed = 1e-9*(event.profile.end - event.profile.start) 
    print("GPU Kernel evaluation Time: {0} s".format(elapsed)) 
    c_gpu = np.empty_like(a) 
    cl._enqueue_read_buffer(queue, c_buffer, c_gpu).wait() 
    gpu_end_time = time() 
    print("GPU Time: {0} s".format(gpu_end_time - gpu_start_time)) 
    return c_gpu 
```

1.  最后，我们执行测试，调用之前定义的两个函数：

```py
if __name__ == "__main__": 
    device_info.print_device_info() 
    cpu_result = test_cpu_vector_sum(a, b) 
    gpu_result = test_gpu_vector_sum(a, b) 
    assert (la.norm(cpu_result - gpu_result)) < 1e-5 
```

# 工作原理...

如前所述，测试包括在CPU上通过`test_cpu_vector_sum`函数执行计算任务，然后通过`test_gpu_vector_sum`函数在GPU上执行。

两个函数都报告执行时间。

关于在CPU上进行测试的函数`test_cpu_vector_sum`，它由对`10000`个向量元素进行双重计算循环组成：

```py
 cpu_start_time = time() 
               for i in range(10000): 
                         for j in range(10000): 
                             c_cpu[i] = a[i] + b[i] 
               cpu_end_time = time() 
```

总CPU时间是以下时间之间的差异：

```py
 CPU Time = cpu_end_time - cpu_start_time 
```

至于`test_gpu_vector_sum`函数，通过查看执行内核，可以看到以下内容：

```py
 __kernel void sum(__global const float *a, 
                      __global const float *b, 
                      __global float *c){ 
        int i=get_global_id(0); 
        int j; 
        for(j=0;j< 10000;j++){ 
            c[i]=a[i]+b[i];} 
```

两个向量的和是通过单个计算循环执行的。

结果，可以想象，是对`test_gpu_vector_sum`函数执行时间的实质性减少：

```py
(base) C:\>python testApplicationPyopencl.py 

============================================================
OpenCL Platforms and Devices
============================================================
Platform - Name: NVIDIA CUDA
Platform - Vendor: NVIDIA Corporation
Platform - Version: OpenCL 1.2 CUDA 10.1.152
Platform - Profile: FULL_PROFILE
 --------------------------------------------------------
 Device - Name: GeForce 840M
 Device - Type: GPU
 Device - Max Clock Speed: 1124 Mhz
 Device - Compute Units: 3
 Device - Local Memory: 48 KB
 Device - Constant Memory: 64 KB
 Device - Global Memory: 2 GB
 Device - Max Buffer/Image Size: 512 MB
 Device - Max Work Group Size: 1024
============================================================
Platform - Name: Intel(R) OpenCL
Platform - Vendor: Intel(R) Corporation
Platform - Version: OpenCL 2.0
Platform - Profile: FULL_PROFILE
 --------------------------------------------------------
 Device - Name: Intel(R) HD Graphics 5500
 Device - Type: GPU
 Device - Max Clock Speed: 950 Mhz
 Device - Compute Units: 24
 Device - Local Memory: 64 KB
 Device - Constant Memory: 64 KB
 Device - Global Memory: 3 GB
 Device - Max Buffer/Image Size: 808 MB
 Device - Max Work Group Size: 256
 --------------------------------------------------------
 Device - Name: Intel(R) Core(TM) i7-5500U CPU @ 2.40GHz
 Device - Type: CPU
 Device - Max Clock Speed: 2400 Mhz
 Device - Compute Units: 4
 Device - Local Memory: 32 KB
 Device - Constant Memory: 128 KB
 Device - Global Memory: 8 GB
 Device - Max Buffer/Image Size: 2026 MB
 Device - Max Work Group Size: 8192

CPU Time: 39.505873918533325 s
GPU Kernel evaluation Time: 0.013606592 s
GPU Time: 0.019981861114501953 s 
```

即使测试不具有计算上的广泛性，它也提供了有关GPU卡潜力的有用指示。

# 还有更多...

OpenCL是一个标准化的跨平台API，用于开发利用异构系统中的并行计算的应用程序。与CUDA的相似之处令人瞩目，包括从内存层次结构到线程和工作项之间的直接对应关系。

即使在编程层面，也有许多相似的方面和具有相同功能的扩展。

然而，由于OpenCL能够支持各种硬件，它具有更复杂的设备管理模型。另一方面，OpenCL旨在实现不同制造商产品之间的代码可移植性。

CUDA由于其更高的成熟度和专用硬件，提供了简化的设备管理和更高级别的API，使其更可取，但前提是您正在处理特定的架构（即NVIDIA显卡）。

CUDA和OpenCL库以及PyCUDA和PyOpenCL库的优缺点在以下部分中进行了解释。

# OpenCL和PyOpenCL的优点

优点如下：

+   它们允许在不同类型的微处理器的异构系统中使用。

+   相同的代码在不同的系统上运行。

# OpenCL和PyOpenCL的缺点

缺点如下：

+   复杂的设备管理

+   APIs不够稳定

# CUDA和PyCUDA的优点

优点如下：

+   具有非常高抽象级别的APIs

+   许多编程语言的扩展

+   庞大的文档和非常庞大的社区

# CUDA和PyCUDA的缺点

缺点如下：

+   仅支持最新的NVIDIA GPU作为设备

+   减少了对CPU和GPU的异构性

# 另请参阅

Andreas Klöckner在[https://www.bu.edu/pasi/courses/gpu-programming-with-pyopencl-and-pycuda/](https://www.bu.edu/pasi/courses/gpu-programming-with-pyopencl-and-pycuda/)和[https://www.youtube.com/results?search_query=pyopenCL+and+pycuda](https://www.youtube.com/results?search_query=pyopenCL+and+pycuda)上提供了一系列关于PyCuda和PyOpenCL的GPU编程讲座。

# 使用Numba进行GPU编程

Numba是一个提供基于CUDA的API的Python编译器。它主要设计用于数值计算任务，就像NumPy库一样。特别是，`numba`库管理和处理NumPy提供的数组数据类型。

事实上，利用数据并行性，这是涉及数组的数值计算中固有的选择，对于GPU加速器来说是一个自然的选择。

Numba编译器通过为Python函数指定签名类型（或装饰器）并在运行时启用编译来工作（这种类型的编译也称为*即时编译*）。

最重要的装饰器如下：

+   `jit`：这允许开发人员编写类似CUDA的函数。当遇到时，编译器将装饰器下的代码翻译成伪汇编PTX语言，以便GPU执行。

+   `autojit`：这为*延迟编译*过程注释了一个函数，这意味着具有此签名的函数只编译一次。

+   `vectorize`：这创建了一个所谓的**NumPy通用函数**（**ufunc**），它接受一个函数并使用矢量参数并行执行它。

+   `guvectorize`：这构建了所谓的**NumPy广义通用函数**（**gufunc**）。`gufunc`对象可以操作整个子数组。

# 准备工作

Numba（版本0.45）兼容Python 2.7和3.5或更高版本，以及NumPy版本1.7到1.16。

要安装`numba`，建议使用Anaconda框架，因此，只需从Anaconda Prompt中输入以下内容：

```py
(base) C:\> conda install numba
```

此外，为了充分利用`numba`，必须安装`cudatoolkit`库：

```py
(base) C:\> conda install cudatoolkit
```

之后，可以验证CUDA库和GPU是否被正确检测到。

从Anaconda Prompt打开Python解释器：

```py
(base) C:\> python
Python 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>
```

第一个测试涉及检查CUDA库（`cudatoolkit`）是否正确安装：

```py
>>> import numba.cuda.api
>>> import numba.cuda.cudadrv.libs
>>> numba.cuda.cudadrv.libs.test()
```

以下输出显示了安装的质量，其中所有检查都返回了积极的结果：

```py
Finding cublas from Conda environment
 located at C:\Users\Giancarlo\Anaconda3\Library\bin\cublas64_10.dll
 trying to open library... ok
Finding cusparse from Conda environment
 located at C:\Users\Giancarlo\Anaconda3\Library\bin\cusparse64_10.dll
 trying to open library... ok
Finding cufft from Conda environment
 located at C:\Users\Giancarlo\Anaconda3\Library\bin\cufft64_10.dll
 trying to open library... ok
Finding curand from Conda environment
 located at C:\Users\Giancarlo\Anaconda3\Library\bin\curand64_10.dll
 trying to open library... ok
Finding nvvm from Conda environment
 located at C:\Users\Giancarlo\Anaconda3\Library\bin\nvvm64_33_0.dll
 trying to open library... ok
Finding libdevice from Conda environment
 searching for compute_20... ok
 searching for compute_30... ok
 searching for compute_35... ok
 searching for compute_50... ok
True

```

在第二次测试中，我们验证了显卡的存在：

```py
>>> numba.cuda.api.detect()
```

输出显示找到的显卡以及是否支持它：

```py
Found 1 CUDA devices
id 0 b'GeForce 840M' [SUPPORTED]
 compute capability: 5.0
 pci device id: 0
 pci bus id: 8
Summary:
 1/1 devices are supported
True
```

# 如何做...

在这个例子中，我们使用`@guvectorize`注释演示了Numba编译器的使用。

要执行的任务是矩阵乘法：

1.  从`numba`库和`numpy`模块导入`guvectorize`：

```py
from numba import guvectorize 
import numpy as np 
```

1.  使用`@guvectorize`装饰器，我们定义了`matmul`函数，它将执行矩阵乘法任务：

```py
@guvectorize(['void(int64[:,:], int64[:,:], int64[:,:])'], 
             '(m,n),(n,p)->(m,p)') 
def matmul(A, B, C): 
    m, n = A.shape 
    n, p = B.shape 
    for i in range(m): 
        for j in range(p): 
            C[i, j] = 0 
            for k in range(n): 
                C[i, j] += A[i, k] * B[k, j] 
```

1.  输入矩阵的大小为10×10，元素为整数：

```py
dim = 10 
A = np.random.randint(dim,size=(dim, dim)) 
B = np.random.randint(dim,size=(dim, dim)) 
```

1.  最后，我们在之前定义的输入矩阵上调用`matmul`函数：

```py
C = matmul(A, B) 
```

1.  我们打印输入矩阵和结果矩阵：

```py
print("INPUT MATRIX A") 
print(":\n%s" % A) 
print("INPUT MATRIX B") 
print(":\n%s" % B) 
print("RESULT MATRIX C = A*B") 
print(":\n%s" % C) 
```

# 它是如何工作的...

`@guvectorize`装饰器适用于数组参数，按顺序使用四个参数来指定`gufunc`签名：

+   前三个参数指定要管理的数据类型和整数数组：`void(int64[:,:], int64[:,:], int64[:,:])`。

+   `@guvectorize`的最后一个参数指定如何操作矩阵维度：`(m,n),(n,p)->(m,p)`。

然后，定义了矩阵乘法操作，其中`A`和`B`是输入矩阵，`C`是输出矩阵：*A(m,n)* B(n,p) = C(m,p)*，其中*m*、*n*和*p*是矩阵维度。

矩阵乘积是通过三个`for`循环以及矩阵索引执行的：

```py
 for i in range(m): 
            for j in range(p): 
                C[i, j] = 0 
                for k in range(n): 
                      C[i, j] += A[i, k] * B[k, j] 
```

这里使用`randint` NumPy函数构建了10×10维度的输入矩阵：

```py
dim = 10
A = np.random.randint(dim,size=(dim, dim))
B = np.random.randint(dim,size=(dim, dim))
```

最后，使用这些矩阵作为参数调用 `matmul` 函数，并打印出结果矩阵 `C`：

```py
C = matmul(A, B)
print("RESULT MATRIX C = A*B")
print(":\n%s" % C)
```

要执行此示例，请键入以下内容：

```py
(base) C:\>python matMulNumba.py
```

结果显示了输入的两个矩阵以及它们的乘积得到的矩阵：

```py
INPUT MATRIX A
:
[[8 7 1 3 1 0 4 9 2 2]
 [3 6 2 7 7 9 8 4 4 9]
 [8 9 9 9 1 1 1 1 8 0]
 [0 5 0 7 1 3 2 0 7 3]
 [4 2 6 4 1 2 9 1 0 5]
 [3 0 6 5 1 0 4 3 7 4]
 [0 9 7 2 1 4 3 3 7 3]
 [1 7 2 7 1 8 0 3 4 1]
 [5 1 5 0 7 7 2 3 0 9]
 [4 6 3 6 0 3 3 4 1 2]]
INPUT MATRIX B
:
[[2 1 4 6 6 4 9 9 5 2]
 [8 6 7 6 5 9 2 1 0 9]
 [4 1 2 4 8 2 9 5 1 4]
 [9 9 1 5 0 5 1 1 7 1]
 [8 7 8 3 9 1 4 3 1 5]
 [7 2 5 8 3 5 8 5 6 2]
 [5 3 1 4 3 7 2 9 9 5]
 [8 7 9 3 4 1 7 8 0 4]
 [3 0 4 2 3 8 8 8 6 2]
 [8 6 7 1 8 3 0 8 8 9]]
RESULT MATRIX C = A*B
:
[[225 172 201 161 170 172 189 230 127 169]
 [400 277 289 251 278 276 240 324 295 273]
 [257 171 177 217 208 254 265 224 176 174]
 [187 130 116 117 94 175 105 128 152 114]
 [199 133 117 143 168 156 143 214 188 157]
 [180 118 124 113 152 149 175 213 167 122]
 [238 142 186 165 188 215 202 200 139 192]
 [237 158 162 176 122 185 169 140 137 130]
 [249 160 220 159 249 125 201 241 169 191]
 [209 152 142 154 131 160 147 161 132 137]]
```

# 还有更多...

使用 PyCUDA 编写缩减操作的算法可能非常复杂。为此，Numba 提供了 `@reduce` 装饰器，用于将简单的二进制操作转换为*缩减内核*。

缩减操作将一组值缩减为单个值。缩减操作的典型示例是计算数组所有元素的总和。例如，考虑以下元素数组：1, 2, 3, 4, 5, 6, 7, 8。

顺序算法按照图表中显示的方式运行，即一个接一个地添加数组的元素：

![](assets/7e5ea317-7653-4c24-96f3-8ea106d866df.png)

顺序求和

并行算法按照以下模式运行：

![](assets/3704575d-6b42-4dc7-b4f9-01093cb44870.png)

并行求和

很明显，后者具有更短的执行时间优势。

通过使用 Numba 和 `@reduce` 装饰器，我们可以编写一个算法，用几行代码对从 1 到 10,000 的整数数组进行并行求和：

```py
import numpy 
from numba import cuda 

@cuda.reduce 
def sum_reduce(a, b): 
    return a + b 

A = (numpy.arange(10000, dtype=numpy.int64)) + 1
print(A) 
got = sum_reduce(A)
print(got) 
```

可以通过输入以下命令执行前面的示例：

```py
(base) C:\>python reduceNumba.py
```

提供以下结果：

```py
vector to reduce = [ 1 2 3 ... 9998 9999 10000]
result = 50005000
```

# 另请参阅

在以下存储库中，您可以找到许多 Numba 的示例：[https://github.com/numba/numba-examples](https://github.com/numba/numba-examples)。您可以在[https://nyu-cds.github.io/python-numba/05-cuda/](https://nyu-cds.github.io/python-numba/05-cuda/)找到有关 Numba 和 CUDA 编程的有趣介绍。
