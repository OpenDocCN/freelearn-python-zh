# 核心、线程、块和网格

本章中，我们将看到如何编写有效的**CUDA 核心**。在 GPU 编程中，**核心**（我们可以互换使用术语**CUDA 核心**或**核心函数**）是一个可以直接从**主机**（CPU）启动到**设备**（GPU）的并行函数，而**设备函数**是一个只能从核心函数或另一个设备函数调用的函数。（一般来说，设备函数看起来和行为像普通的串行 C/C++函数，只是它们在 GPU 上运行，并且从核心函数并行调用。）

然后我们将了解 CUDA 如何使用**线程**、**块**和**网格**的概念来抽象 GPU 的一些基础技术细节（例如核心、warp 和流多处理器，我们将在本书的后面部分介绍），以及我们如何使用这些概念来减轻并行编程中的认知负担。我们将学习关于线程同步（块级和网格级），以及在 CUDA 中使用**全局**和**共享****内存**进行线程间通信。最后，我们将深入了解如何在 GPU 上实现我们自己的并行前缀类型算法（即我们在上一章中介绍的扫描/归约类型函数），这使我们能够将本章学到的所有原则付诸实践。

本章的学习成果如下：

+   理解核心和设备函数之间的区别

+   如何在 PyCUDA 中编译和启动核心，并在核心内使用设备函数

+   在启动核心的上下文中有效使用线程、块和网格，以及如何在核心内使用`threadIdx`和`blockIdx`

+   如何以及为什么在核心内同步线程，使用`__syncthreads()`来同步单个块中的所有线程，以及主机来同步整个块网格中的所有线程

+   如何使用设备全局和共享内存进行线程间通信

+   如何使用我们新获得的关于核心的所有知识来正确实现并行前缀和的 GPU 版本

# 技术要求

本章需要一台带有现代 NVIDIA GPU（2016 年以后）的 Linux 或 Windows 10 PC，并安装了所有必要的 GPU 驱动程序和 CUDA Toolkit（9.0 及以上）。还需要一个合适的 Python 2.7 安装（如 Anaconda Python 2.7），并安装了 PyCUDA 模块。

本章的代码也可以在 GitHub 上找到：

[`github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA`](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)

有关先决条件的更多信息，请查看本书的*前言*；有关软件和硬件要求，请查看[`github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA`](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)中的`README`部分。

# 核心

与上一章一样，我们将学习如何在 Python 代码中以内联 CUDA C 编写 CUDA 核心函数，并使用 PyCUDA 将它们启动到我们的 GPU 上。在上一章中，我们使用 PyCUDA 提供的模板来编写符合特定设计模式的核心，相比之下，我们现在将看到如何从头开始编写我们自己的核心，以便我们可以编写各种各样的核心，这些核心可能不属于 PyCUDA 涵盖的任何特定设计模式，并且我们可以更精细地控制我们的核心。当然，这些收益将以编程复杂性增加为代价；我们特别需要了解**线程**、**块**和**网格**及其在核心中的作用，以及如何**同步**我们的核心正在执行的线程，以及如何在线程之间交换数据。

让我们从简单开始，尝试重新创建我们在上一章中看到的一些逐元素操作，但这次不使用`ElementwiseKernel`函数；我们现在将使用`SourceModule`函数。这是 PyCUDA 中非常强大的函数，允许我们从头构建一个内核，所以通常最好从简单开始。

# PyCUDA SourceModule 函数

我们将使用 PyCUDA 的`SourceModule`函数将原始内联 CUDA C 代码编译为可用的内核，我们可以从 Python 中启动。我们应该注意，`SourceModule`实际上将代码编译为**CUDA 模块**，这类似于 Python 模块或 Windows DLL，只是它包含一组编译的 CUDA 代码。这意味着我们必须使用 PyCUDA 的`get_function`“提取”我们想要使用的内核的引用，然后才能实际启动它。让我们从如何使用`SourceModule`的基本示例开始。

与以前一样，我们将从制作最简单的内核函数之一开始，即将向量乘以标量。我们将从导入开始：

```py
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
```

现在我们可以立即开始编写我们的内核：

```py
ker = SourceModule("""
__global__ void scalar_multiply_kernel(float *outvec, float scalar, float *vec)
{
 int i = threadIdx.x;
 outvec[i] = scalar*vec[i];
}
""")
```

因此，让我们停下来，对比一下在`ElementwiseKernel`中是如何完成的。首先，在 CUDA C 中声明内核函数时，我们要在前面加上`__global__`关键字。这将使编译器将该函数标识为内核。我们总是将其声明为`void`函数，因为我们总是通过传递指向一些空内存块的指针来获得输出值。我们可以像声明任何标准 C 函数一样声明参数：首先是`outvec`，这将是我们的输出缩放向量，当然是浮点数组指针。接下来是`scalar`，用一个简单的`float`表示；注意这不是一个指针！如果我们希望将简单的单例输入值传递给我们的内核，我们总是可以在不使用指针的情况下这样做。最后，我们有我们的输入向量`vec`，当然是另一个浮点数组指针。

单例输入参数可以直接从主机传递给内核函数，而无需使用指针或分配的设备内存。

让我们在继续测试之前先深入了解内核。我们记得`ElementwiseKernel`会自动并行化多个 GPU 线程，通过 PyCUDA 为我们设置的值`i`；每个单独线程的标识由`threadIdx`值给出，我们可以通过以下方式检索：`int i = threadIdx.x;`。

`threadIdx`用于告诉每个单独的线程其身份。这通常用于确定应在输入和输出数据数组上处理哪些值的索引。（这也可以用于使用标准 C 控制流语句（如`if`或`switch`）为特定线程分配不同的任务。）

现在，我们准备像以前一样并行执行标量乘法：`outvec[i] = scalar*vec[i];`。

现在，让我们测试这段代码：我们首先必须从我们刚刚使用`SourceModule`编译的 CUDA 模块中*提取*编译的内核函数的引用。我们可以使用 Python 的`get_function`来获取这个内核引用，如下所示：

```py
scalar_multiply_gpu = ker.get_function("scalar_multiply_kernel")
```

现在，我们必须在 GPU 上放一些数据来实际测试我们的内核。让我们设置一个包含 512 个随机值的浮点数组，然后使用`gpuarray.to_gpu`函数将这些值复制到 GPU 的全局内存中的数组中。（我们将在 GPU 和 CPU 上将这个随机向量乘以一个标量，并查看输出是否匹配。）我们还将使用`gpuarray.empty_like`函数在 GPU 的全局内存中分配一块空内存块：

```py
testvec = np.random.randn(512).astype(np.float32)
testvec_gpu = gpuarray.to_gpu(testvec)
outvec_gpu = gpuarray.empty_like(testvec_gpu)
```

现在，我们准备启动我们的内核。我们将标量值设置为`2`。（再次，由于标量是单例，我们不必将该值复制到 GPU，但是我们必须小心确保正确地进行类型转换。）在这里，我们必须使用`block`和`grid`参数明确设置线程数为`512`。我们现在准备启动：

```py
scalar_multiply_gpu( outvec_gpu, np.float32(2), testvec_gpu, block=(512,1,1), grid=(1,1,1))
```

现在我们可以使用`gpuarray`输出对象中的`get`函数来检查输出是否与预期输出匹配，并将其与 NumPy 的`allclose`函数进行比较：

```py
print "Does our kernel work correctly? : {}".format(np.allclose(outvec_gpu.get() , 2*testvec) )
```

（此示例的代码可在存储库中的`simple_scalar_multiply_kernel.py`文件中的`4`下找到。）

现在我们开始去掉前一章中学到的 PyCUDA 内核模板的训练轮——我们现在可以直接用纯 CUDA C 编写内核，并启动它在 GPU 上使用特定数量的线程。但是，在继续使用内核之前，我们必须更多地了解 CUDA 如何将线程结构化为抽象单位**块**和**网格**。

# 线程、块和网格

到目前为止，在本书中，我们一直认为**线程**这个术语是理所当然的。让我们退后一步，看看这究竟意味着——线程是在 GPU 的单个核心上执行的一系列指令—*核心*和*线程*不应被视为同义词！事实上，可以启动使用的线程数量比 GPU 上的核心数量多得多。这是因为，类似于英特尔芯片可能只有四个核心，但在 Linux 或 Windows 中运行数百个进程和数千个线程，操作系统的调度程序可以快速在这些任务之间切换，使它们看起来是同时运行的。GPU 以类似的方式处理线程，允许在成千上万的线程上进行无缝计算。

多个线程在 GPU 上以抽象单位**块**执行。您应该回忆一下我们如何从标量乘法内核中的`threadIdx.x`获得线程 ID；末尾有一个`x`，因为还有`threadIdx.y`和`threadIdx.z`。这是因为您可以在三个维度上对块进行索引，而不仅仅是一个维度。为什么我们要这样做？让我们回忆一下有关从第一章中计算 Mandelbrot 集的示例，*为什么使用 GPU 编程？*和第三章，*使用 PyCUDA 入门*。这是在二维平面上逐点计算的。因此，对于这样的算法，我们可能更倾向于在两个维度上对线程进行索引。同样，在某些情况下，使用三个维度可能是有意义的——在物理模拟中，我们可能需要在 3D 网格内计算移动粒子的位置。

块进一步以称为**网格**的抽象批次执行，最好将其视为*块的块*。与块中的线程一样，我们可以使用`blockIdx.x`、`blockIdx.y`和`blockIdx.z`给出的常量值在网格中的最多三个维度上对每个块进行索引。让我们看一个示例来帮助我们理解这些概念；为了简单起见，我们这里只使用两个维度。

# 康威的生命游戏

《生命游戏》（通常简称为 LIFE）是一种细胞自动机模拟，由英国数学家约翰·康威于 1970 年发明。听起来很复杂，但实际上非常简单——LIFE 是一个零玩家的“游戏”，由一个二维二进制格子组成，其中的“细胞”被认为是“活着的”或“死了的”。这个格子通过以下一组规则进行迭代更新：

+   任何活细胞周围少于两个活邻居的细胞会死亡

+   任何活细胞周围有两个或三个邻居的细胞会存活

+   任何活细胞周围有三个以上的邻居的细胞会死亡

+   任何死细胞周围恰好有三个邻居的细胞会复活

这四条简单的规则产生了一个复杂的模拟，具有有趣的数学特性，而且在动画时也非常美观。然而，在晶格中有大量的细胞时，它可能运行得很慢，并且通常在纯串行 Python 中编程时会导致动画不流畅。然而，这是可以并行化的，因为很明显晶格中的每个细胞可以由一个单独的 CUDA 线程管理。

我们现在将 LIFE 实现为一个 CUDA 核函数，并使用 `matplotlib.animation` 模块来进行动画。这对我们来说现在很有趣，因为我们将能够在这里应用我们对块和网格的新知识。

我们将首先包括适当的模块，如下所示：

```py
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
```

现在，让我们通过 `SourceModule` 来编写我们的核函数。我们将首先使用 C 语言的 `#define` 指令来设置一些我们将在整个核函数中使用的常量和宏。让我们看看我们将设置的前两个，`_X` 和 `_Y`：

```py
ker = SourceModule("""
#define _X  ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y  ( threadIdx.y + blockIdx.y * blockDim.y )
```

首先让我们记住这里 `#define` 的工作原理——它会在编译时用定义的值（在括号中）直接替换 `_X` 或 `_Y` 的任何文本，也就是说，它为我们创建了宏。（作为个人风格的问题，我通常会在所有的 C 宏之前加上下划线。）

在 C 和 C++中，`#define` 用于创建**宏**。这意味着 `#define` 不会创建任何函数或设置正确的常量变量——它只允许我们在编译之前通过交换文本来在我们的代码中以简写方式编写东西。

现在，让我们具体讨论一下 `_X` 和 `_Y` 的含义——这将是单个 CUDA 线程在我们用于 LIFE 的二维晶格上的笛卡尔 *x* 和 *y* 值。我们将在一个二维网格上启动核函数，由二维块组成，这些块将对应整个细胞晶格。我们将使用线程和块常量来找到晶格上的笛卡尔点。让我们看一些图表来说明这一点。驻留在二维 CUDA 块中的线程可以被可视化如下：

![](img/b18aaa4b-b830-44f1-9371-80c57b0ab285.png)

此时，你可能会想知道为什么我们不在一个单独的块上启动我们的核函数，这样我们就可以将 `_X` 设置为 `threadIdx.x`，将 `_Y` 设置为 `threadIdx.y`，然后就完成了。这是由于 CUDA 对我们施加了块大小的限制——目前只支持由最多 1024 个线程组成的块。这意味着我们只能将我们的细胞晶格的尺寸最大设为 32 x 32，这将导致一个相当无聊的模拟，最好在 CPU 上完成，所以我们将在网格上启动多个块。（我们当前块的尺寸将由 `blockDim.x` 和 `blockDim.y` 给出，这将帮助我们确定目标 *x* 和 *y* 坐标，正如我们将看到的。）

同样，和之前一样，我们可以确定我们在二维网格中的块是哪个，使用 `blockIdx.x` 和 `blockIdx.y`：

![](img/45159b32-9f17-4a39-a50f-0e4bfb47b1f2.png)

在我们稍微思考一下数学之后，应该很清楚 `_X` 应该被定义为 `(threadIdx.x + blockIdx.x * blockDim.x)`，而 `_Y` 应该被定义为 `(threadIdx.y + blockIdx.y * blockDim.y)`。（添加括号是为了不干扰宏插入代码时的运算顺序。）现在，让我们继续定义剩下的宏：

```py
#define _WIDTH  ( blockDim.x * gridDim.x )
#define _HEIGHT ( blockDim.y * gridDim.y  )

#define _XM(x)  ( (x + _WIDTH) % _WIDTH )
#define _YM(y)  ( (y + _HEIGHT) % _HEIGHT )
```

`_WIDTH`和`_HEIGHT`宏将分别给出我们单元格格子的宽度和高度，这应该从图表中清楚地看出。让我们讨论`_XM`和`_YM`宏。在我们的 LIFE 实现中，我们将端点“环绕”到格子的另一侧 - 例如，我们将考虑`-1`的*x*值为`_WIDTH - 1`，*y*值为`-1`为`_HEIGHT - 1`，同样地，我们将考虑`_WIDTH`的*x*值为`0`，*y*值为`_HEIGHT`为`0`。我们为什么需要这个？当我们计算给定单元格的存活邻居数时，我们可能处于某个边缘，邻居可能是外部点 - 定义这些宏来调制我们的点将自动为我们覆盖这一点。请注意，在使用 C 的模运算符之前，我们必须添加宽度或高度 - 这是因为，与 Python 不同，C 中的模运算符对于整数可以返回负值。

我们现在有一个最终的宏要定义。我们记得 PyCUDA 将二维数组作为一维指针传递到 CUDA C 中；二维数组从 Python 以**按行**的方式传递到一维 C 指针中。这意味着我们必须将格子上给定的笛卡尔（*x*，*y*）点转换为指向格子对应的指针中的一维点。在这里，我们可以这样做：

```py
#define _INDEX(x,y)  ( _XM(x)  + _YM(y) * _WIDTH )
```

由于我们的单元格格子是按行存储的，我们必须将*y*值乘以宽度以偏移到对应行的点。我们现在终于可以开始我们的 LIFE 实现了。让我们从 LIFE 最重要的部分开始 - 计算给定单元格的存活邻居数。我们将使用 CUDA **设备函数**来实现这一点，如下所示：

```py
__device__ int nbrs(int x, int y, int * in)
{
     return ( in[ _INDEX(x -1, y+1) ] + in[ _INDEX(x-1, y) ] + in[ _INDEX(x-1, y-1) ] \
                   + in[ _INDEX(x, y+1)] + in[_INDEX(x, y - 1)] \
                   + in[ _INDEX(x+1, y+1) ] + in[ _INDEX(x+1, y) ] + in[ _INDEX(x+1, y-1) ] );
}

```

设备函数是以串行方式编写的 C 函数，由内核中的单个 CUDA 线程调用。也就是说，这个小函数将由我们内核中的多个线程并行调用。我们将把我们的单元格格子表示为 32 位整数的集合（1 表示活细胞，0 表示死细胞），所以这对我们的目的是有效的；我们只需要添加周围当前单元格的值。

CUDA **设备函数**是由内核中的单个 CUDA 线程调用的串行 C 函数。虽然这些函数本身是串行的，但它们可以由多个 GPU 线程并行运行。设备函数本身不能由主机计算机启动到 GPU 上，只能由内核启动。

我们现在准备编写 LIFE 的内核实现。实际上，我们已经完成了大部分的艰苦工作 - 我们检查当前线程单元格的邻居数量，检查当前单元格是生还是死，然后使用适当的 switch-case 语句来根据 LIFE 的规则确定下一次迭代的状态。我们将使用两个整数指针数组作为内核 - 一个将用作输入参考上一次迭代（`lattice`），另一个将用作输出我们将计算的迭代（`lattice_out`）的参考。

```py
__global__ void conway_ker(int * lattice_out, int * lattice  )
{
   // x, y are the appropriate values for the cell covered by this thread
   int x = _X, y = _Y;

   // count the number of neighbors around the current cell
   int n = nbrs(x, y, lattice);

    // if the current cell is alive, then determine if it lives or dies for the next generation.
    if ( lattice[_INDEX(x,y)] == 1)
       switch(n)
       {
          // if the cell is alive: it remains alive only if it has 2 or 3 neighbors.
          case 2:
          case 3: lattice_out[_INDEX(x,y)] = 1;
                  break;
          default: lattice_out[_INDEX(x,y)] = 0;                   
       }
    else if( lattice[_INDEX(x,y)] == 0 )
         switch(n)
         {
            // a dead cell comes to life only if it has 3 neighbors that are alive.
            case 3: lattice_out[_INDEX(x,y)] = 1;
                    break;
            default: lattice_out[_INDEX(x,y)] = 0;         
         }

}
""")

conway_ker = ker.get_function("conway_ker")

```

我们记得用三重括号关闭内联 CUDA C 段落，然后用`get_function`获取对我们的 CUDA C 内核的引用。由于内核只会一次更新格子，我们将在 Python 中设置一个简短的函数，它将涵盖更新格子的所有开销以用于动画：

```py
def update_gpu(frameNum, img, newLattice_gpu, lattice_gpu, N):    
```

`frameNum`参数只是 Matplotlib 动画模块对于我们可以忽略的更新函数所需的一个值，而`img`将是我们单元格格子的代表图像，这是动画模块所需的，将被迭代显示。

让我们关注另外三个参数—`newLattice_gpu`和`lattice_gpu`将是我们保持持久的 PyCUDA 数组，因为我们希望避免在 GPU 上重新分配内存块。`lattice_gpu`将是细胞数组的当前一代，对应于内核中的`lattice`参数，而`newLattice_gpu`将是下一代晶格。`N`将指示晶格的高度和宽度（换句话说，我们将使用*N x N*晶格）。

我们使用适当的参数启动内核，并设置块和网格大小如下：

```py
    conway_ker(newLattice_gpu, lattice_gpu, grid=(N/32,N/32,1), block=(32,32,1) )    
```

我们将块大小设置为 32 x 32，使用`(32, 32, 1)`；因为我们只使用两个维度来表示我们的细胞晶格，所以我们可以将*z*维度设置为 1。请记住，块的线程数限制为 1,024 个线程—*32 x 32 = 1024*，所以这样可以工作。（请记住，32 x 32 没有什么特别之处；如果需要，我们可以使用 16 x 64 或 10 x 10 等值，只要总线程数不超过 1,024。）

CUDA 块中的线程数最多为 1,024。

现在我们来看一下网格值—在这里，因为我们使用 32 的维度，很明显*N*（在这种情况下）应该是 32 的倍数。这意味着在这种情况下，我们只能使用 64 x 64、96 x 96、128 x 128 和 1024 x 1024 等晶格。同样，如果我们想使用不同大小的晶格，那么我们将不得不改变块的维度。（如果这不太清楚，请查看之前的图表，并回顾一下我们如何在内核中定义宽度和高度宏。）

现在我们可以使用`get()`函数从 GPU 的内存中获取最新生成的晶格，并为我们的动画设置图像数据。最后，我们使用 PyCUDA 切片操作`[:]`将新的晶格数据复制到当前数据中，这将复制 GPU 上先前分配的内存，这样我们就不必重新分配了：

```py
    img.set_data(newLattice_gpu.get() )    
    lattice_gpu[:] = newLattice_gpu[:]

    return img
```

让我们设置一个大小为 256 x 256 的晶格。现在我们将使用`numpy.random`模块中的 choice 函数为我们的晶格设置初始状态。我们将随机用 1 和 0 填充一个*N x N*的整数图表；通常，如果大约 25%的点是 1，其余的是 0，我们可以生成一些有趣的晶格动画，所以我们就这样做吧：

```py
if __name__ == '__main__':
    # set lattice size
    N = 256

    lattice = np.int32( np.random.choice([1,0], N*N, p=[0.25, 0.75]).reshape(N, N) )
    lattice_gpu = gpuarray.to_gpu(lattice)
```

最后，我们可以使用适当的`gpuarray`函数在 GPU 上设置晶格，并相应地设置 Matplotlib 动画，如下所示：

```py
lattice_gpu = gpuarray.to_gpu(lattice)
    lattice_gpu = gpuarray.to_gpu(lattice)
    newLattice_gpu = gpuarray.empty_like(lattice_gpu) 

    fig, ax = plt.subplots()
    img = ax.imshow(lattice_gpu.get(), interpolation='nearest')
    ani = animation.FuncAnimation(fig, update_gpu, fargs=(img, newLattice_gpu, lattice_gpu, N, ) , interval=0, frames=1000, save_count=1000) 

    plt.show()
```

现在我们可以运行我们的程序并享受展示（代码也可以在 GitHub 存储库的`4`目录下的`conway_gpu.py`文件中找到）：

![](img/bb012845-31d4-4511-a697-9eef0e2772b2.png)

# 线程同步和互通

现在我们将讨论 GPU 编程中的两个重要概念—**线程同步**和**线程互通**。有时，我们需要确保每个线程在继续任何进一步的计算之前都已经到达了代码中完全相同的行；我们称之为线程同步。同步与线程互通相辅相成，也就是说，不同的线程之间传递和读取输入；在这种情况下，我们通常希望确保所有线程在传递数据之前都处于计算的相同步骤。我们将从学习 CUDA `__syncthreads`设备函数开始，该函数用于同步内核中的单个块。

# 使用 __syncthreads()设备函数

在我们之前的康威生命游戏的示例中，我们的内核每次由主机启动时只更新了晶格一次。在这种情况下，同步所有在启动的内核中的线程没有问题，因为我们只需要处理已经准备好的晶格的上一个迭代。

现在假设我们想做一些稍微不同的事情——我们想重新编写我们的内核，以便在给定的细胞点阵上执行一定数量的迭代，而不是由主机一遍又一遍地重新启动。这一开始可能看起来很琐碎——一个天真的解决方案将是只需在内联`conway_ker`内核中放置一个整数参数来指示迭代次数和一个`for`循环，进行一些额外的琐碎更改，然后就完成了。

然而，这引发了**竞争条件**的问题；这是多个线程读取和写入相同内存地址以及由此可能产生的问题。我们的旧`conway_ker`内核通过使用两个内存数组来避免这个问题，一个严格用于读取，一个严格用于每次迭代写入。此外，由于内核只执行单次迭代，我们实际上是在使用主机来同步线程。

我们希望在 GPU 上进行多次完全同步的 LIFE 迭代；我们还希望使用单个内存数组来存储点阵。我们可以通过使用 CUDA 设备函数`__syncthreads()`来避免竞争条件。这个函数是一个**块级同步屏障**——这意味着在一个块内执行的每个线程在到达`__syncthreads()`实例时都会停止，并等待直到同一块内的每个其他线程都到达`__syncthreads()`的同一调用，然后线程才会继续执行后续的代码行。

`__syncthreads()`只能同步单个 CUDA 块内的线程，而不能同步 CUDA 网格内的所有线程！

让我们现在创建我们的新内核；这将是之前 LIFE 内核的修改，它将执行一定数量的迭代然后停止。这意味着我们不会将其表示为动画，而是作为静态图像，因此我们将在开始时加载适当的 Python 模块。（此代码也可在 GitHub 存储库的`conway_gpu_syncthreads.py`文件中找到）：

```py
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt 
```

现在，让我们再次设置计算 LIFE 的内核：

```py
ker = SourceModule("""
```

当然，我们的 CUDA C 代码将放在这里，这将大致与以前相同。我们只需要对内核进行一些更改。当然，我们可以保留设备函数`nbrs`。在我们的声明中，我们将只使用一个数组来表示细胞点阵。我们可以这样做，因为我们将使用适当的线程同步。我们还必须用一个整数表示迭代次数。我们设置参数如下：

```py
__global__ void conway_ker(int * lattice, int iters) {
```

我们将继续与以前类似，只是使用`for`循环进行迭代：

```py
 int x = _X, y = _Y; 
 for (int i = 0; i < iters; i++)
 {
     int n = nbrs(x, y, lattice); 
     int cell_value;
```

让我们回想一下以前，我们直接在数组中设置新的细胞点阵值。在这里，我们将在`cell_value`变量中保存值，直到块内的所有线程都同步。我们以前也是类似地进行，使用`__syncthreads`阻止执行，直到确定了当前迭代的所有新细胞值，然后才在点阵数组中设置值：

```py
 if ( lattice[_INDEX(x,y)] == 1)
 switch(n)
 {
 // if the cell is alive: it remains alive only if it has 2 or 3 neighbors.
 case 2:
 case 3: cell_value = 1;
 break;
 default: cell_value = 0; 
 }
 else if( lattice[_INDEX(x,y)] == 0 )
 switch(n)
 {
 // a dead cell comes to life only if it has 3 neighbors that are alive.
 case 3: cell_value = 1;
 break;
 default: cell_value = 0; 
 } 
 __syncthreads();
 lattice[_INDEX(x,y)] = cell_value; 
 __syncthreads();
 } 
}
""")
```

我们现在将像以前一样启动内核并显示输出，迭代点阵 100 万次。请注意，由于每个块的线程限制为 1,024 个，我们在网格中只使用一个块，大小为 32 x 32。（再次强调，`__syncthreads`仅在块中的所有线程上工作，而不是在网格中的所有线程上工作，这就是为什么我们在这里限制自己使用单个块的原因）：

```py
conway_ker = ker.get_function("conway_ker")
if __name__ == '__main__':
 # set lattice size
 N = 32
 lattice = np.int32( np.random.choice([1,0], N*N, p=[0.25, 0.75]).reshape(N, N) )
 lattice_gpu = gpuarray.to_gpu(lattice)
 conway_ker(lattice_gpu, np.int32(1000000), grid=(1,1,1), block=(32,32,1))
 fig = plt.figure(1)
 plt.imshow(lattice_gpu.get())
```

当我们运行程序时，我们将得到以下所需的输出（这是随机 LIFE 点阵在一百万次迭代后会收敛到的结果！）：

![](img/38be0537-84a4-447c-a25b-0f60c15726b1.png)

# 使用共享内存

从先前的例子中，我们可以看到内核中的线程可以使用 GPU 全局内存中的数组进行相互通信；虽然可以使用全局内存进行大多数操作，但使用**共享内存**可以加快速度。这是一种专门用于单个 CUDA 块内线程相互通信的内存类型；与全局内存相比，使用共享内存的优势在于纯线程间通信速度更快。不过，与全局内存相反，存储在共享内存中的内存不能直接被主机访问——共享内存必须首先由内核自己复制回全局内存。

在继续之前，让我们先退一步思考一下我们的意思。让我们看看我们刚刚看到的迭代 LIFE 内核中声明的一些变量。首先看看`x`和`y`，这两个整数保存着特定线程单元的笛卡尔坐标。请记住，我们正在使用`_X`和`_Y`宏设置它们的值。（尽管编译器优化，我们希望将这些值存储在变量中以减少计算，因为直接使用`_X`和`_Y`将在我们的代码中引用这些宏时每次重新计算`x`和`y`的值）：

```py
 int x = _X, y = _Y; 
```

我们注意到，对于每个单个线程，点阵中将对应于`x`和`y`的唯一笛卡尔点。同样，我们使用一个变量`n`，它声明为`int n = nbrs(x, y, lattice);`，来表示特定单元周围的存活邻居的数量。这是因为，当我们通常在 CUDA 中声明变量时，它们默认是每个单独线程的本地变量。请注意，即使我们在线程内部声明数组如`int a[10];`，也将有一个大小为 10 的数组，它是每个线程的本地数组。

本地线程数组（例如，在内核内部声明`int a[10];`）和指向全局 GPU 内存的指针（例如，以`int * b`形式作为内核参数传递的值）可能看起来和行为类似，但实际上非常不同。对于内核中的每个线程，将有一个单独的`a`数组，其他线程无法读取，但将有一个单独的`b`，它将保存相同的值，并且对所有线程都是同样可访问的。

我们准备使用共享内存。这使我们能够声明在单个 CUDA 块内的线程之间共享的变量和数组。这种内存比使用全局内存指针（我们到目前为止一直在使用的）要快得多，同时减少了指针分配内存的开销。

假设我们想要一个大小为 10 的共享整数数组。我们声明如下——`__shared__ int a[10] `。请注意，我们不必局限于数组；我们可以按如下方式创建共享的单例变量：`__shared__ int x`。

让我们重新编写 LIFE 的迭代版本的一些行，以利用共享内存。首先，让我们将输入指针重命名为`p_lattice`，这样我们可以在我们的共享数组上使用这个变量名，并在我们的代码中懒惰地保留所有对“lattice”的引用。由于我们将坚持使用 32 x 32 个单元的点阵，我们设置新的共享`lattice`数组如下：

```py
__global__ void conway_ker_shared(int * p_lattice, int iters)
{
 int x = _X, y = _Y;
 __shared__ int lattice[32*32];
```

现在我们必须将全局内存`p_lattice`数组中的所有值复制到`lattice`中。我们将以完全相同的方式索引我们的共享数组，因此我们可以在这里使用我们旧的`_INDEX`宏。请注意，在复制后我们确保在我们继续 LIFE 算法之前放置`__syncthreads()`，以确保所有对 lattice 的内存访问完全完成：

```py
 lattice[_INDEX(x,y)] = p_lattice[_INDEX(x,y)];
 __syncthreads();
```

内核的其余部分与以前完全相同，只是我们必须将共享的点阵复制回 GPU 数组。我们这样做，然后关闭内联代码：

```py
 __syncthreads();
 p_lattice[_INDEX(x,y)] = lattice[_INDEX(x,y)];
 __syncthreads();
} """)
```

现在我们可以像以前一样运行，使用完全相同的测试代码。（此示例可以在 GitHub 存储库中的`conway_gpu_syncthreads_shared.py`中找到。）

# 并行前缀算法

现在，我们将利用我们对 CUDA 核心的新知识来实现**并行前缀算法**，也称为**扫描设计模式**。我们已经在上一章中以 PyCUDA 的`InclusiveScanKernel`和`ReductionKernel`函数的形式看到了这种简单的例子。现在让我们更详细地了解这个想法。

这个设计模式的核心动机是，我们有一个二元运算符![](img/9388a619-6713-4ea7-93d8-a85fc2fd8094.png)，也就是说，一个作用于两个输入值并给出一个输出值的函数（比如—+，![](img/362dcb88-5323-4213-8ea4-03a9785d4984.png)，![](img/2abe6459-7144-4bdf-9569-b0a70726e422.png)（最大值），![](img/380e1f66-b930-42d9-b7d0-a565b74b858f.png)（最小值）），和元素的集合![](img/d10897a0-55d1-4a8f-9862-fd43a6f729ea.png)，我们希望从中高效地计算![](img/d1dace09-f460-4cee-abb6-81691be4dcf6.png)。此外，我们假设我们的二元运算符![](img/2c1163a9-8a0f-480d-be93-f5cbaa606034.png)是**可结合的**—这意味着，对于任意三个元素*x*、*y*和*z*，我们总是有:![](img/7f9f94dd-6751-4a0e-abcc-32333a71812d.png)。

我们希望保留部分结果，也就是*n-1*个子计算—![](img/cadd1c5c-4dfa-45f8-bca5-e3e810c6187b.png)。并行前缀算法的目的是高效地产生这些*n*个和。在串行操作中，通常需要*O(n)*的时间来产生这些*n*个和，我们希望降低时间复杂度。

当使用术语“并行前缀”或“扫描”时，通常意味着一个产生所有这些*n*个结果的算法，而“减少”/“归约”通常意味着只产生单个最终结果，![](img/864d06dd-ccd8-48c3-8a0a-fd437cd6436e.png)。（这是 PyCUDA 的情况。）

实际上，并行前缀算法有几种变体，我们将首先从最简单（也是最古老的）版本开始，这就是所谓的天真并行前缀算法。

# 天真并行前缀算法

这个算法是原始版本的**天真并行前缀算法**；这个算法是“天真的”，因为它假设给定*n*个输入元素，![](img/2a2cf114-8fcf-41ec-83de-66803d5c7345.png)，进一步假设*n*是*二进制*的（也就是说，![](img/f23d2bc1-c7c4-44fc-a947-ac7571697516.png)对于某个正整数*k*），我们可以在*n*个处理器（或*n*个线程）上并行运行算法。显然，这将对我们可以处理的集合的基数*n*施加严格的限制。然而，只要满足这些条件，我们就有一个很好的结果，即其计算时间复杂度仅为*O(log n)*。我们可以从算法的伪代码中看到这一点。在这里，我们将用![](img/79ff2d6e-25d9-478d-8a3c-5fe79faa77cf.png)表示输入值，用![](img/be8218cd-e2fd-4453-87fd-50f3cf71308c.png)表示输出值：

```py
input: x0, ..., xn-1 initialize:
for k=0 to n-1:
    yk := xk begin:
parfor i=0 to n-1 :
    for j=0 to log2(n):
        if i >= 2j :
            yi := yi  yi - 2^j else:
            continue
        end if
    end for
end parfor
end
output: y0, ..., yn-1
```

现在，我们可以清楚地看到这将花费*O(log n)*的渐近时间，因为外部循环是在`parfor`上并行化的，内部循环需要*log2*。经过几分钟的思考，很容易看出*y[i]*值将产生我们期望的输出。

现在让我们开始我们的实现；在这里，我们的二元运算符将简单地是加法。由于这个例子是说明性的，这个核心代码将严格地针对 1,024 个线程。

让我们先设置好头文件，然后开始编写我们的核心代码：

```py
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from time import time

naive_ker = SourceModule("""
__global__ void naive_prefix(double *vec, double *out)
{
     __shared__ double sum_buf[1024]; 
     int tid = threadIdx.x; 
     sum_buf[tid] = vec[tid];

```

因此，让我们看看我们有什么：我们将我们的输入元素表示为双精度 GPU 数组，即`double *vec`，并用`double *out`表示输出值。我们声明一个共享内存`sum_buf`数组，我们将用它来计算我们的输出。现在，让我们看看算法本身的实现：

```py
 int iter = 1;
 for (int i=0; i < 10; i++)
 {
     __syncthreads();
     if (tid >= iter )
     {
         sum_buf[tid] = sum_buf[tid] + sum_buf[tid - iter]; 
     } 
     iter *= 2;
 }
 __syncthreads();
```

当然，这里没有`parfor`，它是隐式的，通过`tid`变量表示线程编号。我们还可以通过从初始化为 1 的变量开始，然后在每次 i 的迭代中乘以 2 来省略*log[2]*和*2^i*的使用。（请注意，如果我们想更加技术化，我们可以使用位移运算符来实现这一点。）我们将`i`的迭代限制在 10 次，因为*2¹⁰ = 1024*。现在我们将结束我们的新内核如下：

```py
 __syncthreads();
 out[tid] = sum_buf[tid];
 __syncthreads();

}
""")
naive_gpu = naive_ker.get_function("naive_prefix")

```

现在让我们看一下内核后面的测试代码：

```py
if __name__ == '__main__':
 testvec = np.random.randn(1024).astype(np.float64)
 testvec_gpu = gpuarray.to_gpu(testvec)

 outvec_gpu = gpuarray.empty_like(testvec_gpu)
 naive_gpu( testvec_gpu , outvec_gpu, block=(1024,1,1), grid=(1,1,1))

 total_sum = sum( testvec)
 total_sum_gpu = outvec_gpu[-1].get()

 print "Does our kernel work correctly? : {}".format(np.allclose(total_sum_gpu , total_sum) )
```

我们只关心输出中的最终总和，我们使用`outvec_gpu[-1].get()`来检索它，需要回想一下，"-1"索引给出了 Python 数组中的最后一个成员。这将是`vec`中每个元素的总和；部分总和在`outvec_gpu`的先前值中。（此示例可以在 GitHub 存储库中的`naive_prefix.py`文件中看到。）

由于其性质，并行前缀算法必须在*n*个线程上运行，对应于一个大小为 n 的数组，其中*n*是二进制的（这意味着*n*是 2 的某个幂）。然而，我们可以将这个算法扩展到任意非二进制大小，假设我们的运算符具有**单位元素**（或等效地，**中性元素**）——也就是说，存在某个值*e*，使得对于任何*x*值，我们有 ![](img/9a41546d-6056-4ca9-bfdd-0e9ec13ae195.png)。在运算符为+的情况下，单位元素是 0；在运算符为 ![](img/04590efc-4d27-477f-a7bb-96da2f050011.png) 的情况下，它是 1；然后我们只需用一系列*e*值填充 ![](img/f1e4ffaf-26a6-4f45-b9f4-20340837e38c.png) 的元素，以便我们有新集合的二进制基数 ![](img/8c22f00e-f896-44d2-99f2-7d5ecc3b35eb.png)。

# 包含与排他前缀

让我们停下来，做一个非常微妙但非常重要的区分。到目前为止，我们一直关心接受形式为 ![](img/cb300a9a-6dff-4df9-b962-91cb62ccd143.png) 的输入，并产生形式为 ![](img/244e529c-37f3-46ba-a492-b34765482abb.png) 的总和数组作为输出。产生这种输出的前缀算法称为**包含**；在**包含前缀算法**的情况下，每个索引处的相应元素包含在输出数组的相同索引处的总和中。这与**排他前缀算法**形成对比。**排他前缀算法**不同之处在于，它同样接受形式为 ![](img/1bba2db0-de4c-42d3-a676-d482b67584c2.png) 的*n*个输入值，并产生长度为*n*的输出数组 ![](img/c789f125-403d-4264-b4f4-a57d73c11977.png)。

这很重要，因为一些高效的前缀算法的变体天生就是排他的。我们将在下一个小节中看到一个例子。

请注意，排他算法产生的输出与包含算法几乎相同，只是右移并省略了最后一个值。因此，我们可以从任一算法中轻松获得等效输出，只要我们保留 ![](img/179c86ba-3a68-4b15-98bf-24d9e930e963.png) 的副本。

# 一个高效的并行前缀算法

在我们继续使用新算法之前，我们将从两个角度看待朴素算法。在理想情况下，计算时间复杂度为*O(log n)*，但这仅当我们有足够数量的处理器用于我们的数据集时成立；当数据集的基数（元素数量）*n*远大于处理器数量时，这将成为*O(n log n)*时间复杂度的算法。

让我们定义一个与我们的二进制运算符 ![](img/b89f2449-e123-451e-a958-6a782b932821.png)相关的新概念——这里并行算法的**工作**是执行期间所有线程对此运算符的调用次数。同样，**跨度**是线程在内核执行期间进行调用的次数；而整个算法的**跨度**与每个单独线程的最长跨度相同，这将告诉我们总执行时间。

我们寻求特别减少算法在所有线程中执行的工作量，而不仅仅是关注跨度。在简单前缀的情况下，所需的额外工作在可用处理器数量不足时会花费更多时间；这些额外工作将溢出到有限数量的可用处理器中。

我们将介绍一种新的算法，它是**工作高效**的，因此更适合有限数量的处理器。这包括两个独立的部分——**向上扫描（或减少）阶段**和**向下扫描阶段**。我们还应该注意，我们将看到的算法是一种独占前缀算法。

**向上扫描阶段**类似于单个减少操作，以产生由减少算法给出的值，即 ![](img/f380e618-5246-4853-b1eb-976537a28ab6.png) ；在这种情况下，我们保留所需的部分和（![](img/b739e65c-64e0-4a07-ae84-740401252763.png)）以实现最终结果。然后，向下扫描阶段将对这些部分和进行操作，并给出最终结果。让我们看一些伪代码，从向上扫描阶段开始。（接下来的小节将立即深入到伪代码的实现中。）

# 工作高效的并行前缀（向上扫描阶段）

这是向上扫描的伪代码。（注意`parfor`覆盖`j`变量，这意味着此代码块可以并行化，由`j`索引的线程）：

```py
input: x0, ..., xn-1initialize:
    for i = 0 to n - 1:
        yi := xi
begin:
for k=0 to log2(n) - 1:
    parfor j=0 to n - 1: 
        if j is divisible by 2k+1:
            yj+2^(k+1)-1 = yj+2^k-1  yj +2^(k+1) -1else:
            continue
end
output: y0, ..., yn-1

```

# 工作高效的并行前缀（向下扫描阶段）

现在让我们继续向下扫描，它将操作向上扫描的输出：

```py
input: x0, ..., xn-1 initialize:
    for i = 0 to n - 2:
        yi := xi
    yn-1 := 0
begin:
for k = log2(n) - 1 to 0:
    parfor j = 0 to n - 1: 
        if j is divisible by 2k+1:
            temp := yj+2^k-1
            yj+2^k-1 := yj+2^(k+1)-1
            yj+2^(k+1)-1 := yj+2^(k+1)-1  temp
        else:
            continue
end
output: y0 , y1 , ..., yn-1
```

# 工作高效的并行前缀 — 实现

作为本章的压轴，我们将编写该算法的实现，该算法可以在大于 1,024 的任意大小的数组上操作。这意味着这将在网格和块上操作；因此，我们将不得不使用主机进行同步；此外，这将要求我们为向上扫描和向下扫描阶段实现两个单独的内核，这将作为两个阶段的`parfor`循环，以及作为向上和向下扫描的外部`for`循环的 Python 函数。 

让我们从向上扫描内核开始。由于我们将从主机迭代重新启动此内核，我们还需要一个指示当前迭代（`k`）的参数。我们将使用两个数组进行计算，以避免竞争条件——`x`（用于当前迭代）和`x_old`（用于之前的迭代）。我们声明内核如下：

```py
up_ker = SourceModule("""
__global__ void up_ker(double *x, double *x_old, int k)
{
```

现在让我们设置`tid`变量，它将是网格中*所有*块中*所有*线程中当前线程的标识。我们使用与我们之前看到的康威生命游戏的原始网格级实现相同的技巧：

```py
int tid =  blockIdx.x*blockDim.x + threadIdx.x;
```

我们现在将使用 C 位移运算符直接从`k`生成 2^k 和 2^(k+1 )。我们现在将`j`设置为`tid`乘以`_2k1`—这将使我们能够删除“如果`j`可被 2^(k+1)整除”的部分，如伪代码中所示，从而使我们只启动所需数量的线程：

```py
 int _2k = 1 << k;
 int _2k1 = 1 << (k+1);

 int j = tid* _2k1;
```

我们可以使用 CUDA C 中的左位移运算符（`<<`）轻松生成二进制（2 的幂）整数。请记住，整数 1（即 2⁰）表示为 0001，2（2¹）表示为 0010，4（2²）表示为 0100，依此类推。因此，我们可以使用`1 << k`操作计算 2^k。

现在我们可以运行上扫描阶段的单行代码，注意`j`确实可以被 2^(k+1)整除，因为它的构造方式是这样的：

```py

 x[j + _2k1 - 1] = x_old[j + _2k -1 ] + x_old[j + _2k1 - 1];
}
""")
```

我们已经编写完我们的核心代码了！但这当然不是完整的上扫描实现。我们还需要在 Python 中完成其余部分。让我们拿到我们的核心代码并开始实现。这基本上是按照伪代码进行的，我们应该记住，我们通过使用`[:]`从`x_gpu`复制到`x_old_gpu`来更新`x_old_gpu`，这将保留内存分配，并仅复制新数据而不是重新分配。还要注意，我们根据要启动的线程数量设置我们的块和网格大小 - 我们尝试保持我们的块大小为 32 的倍数（这是本文中的经验法则，我们在第十一章中详细介绍为什么我们特别使用 32，*CUDA 性能优化*）。我们应该在文件开头加上`from __future__ import division`，因为我们将使用 Python 3 风格的除法来计算我们的块和核心大小。

有一点需要提到的是，我们假设`x`的长度是二进制长度 32 或更大 - 如果您希望将其操作在其他大小的数组上，可以通过用零填充我们的数组来轻松修改，然而：

```py

up_gpu = up_ker.get_function("up_ker")

def up_sweep(x):
    x = np.float64(x)
    x_gpu = gpuarray.to_gpu(np.float64(x) )
    x_old_gpu = x_gpu.copy()
    for k in range( int(np.log2(x.size) ) ) : 
        num_threads = int(np.ceil( x.size / 2**(k+1)))
        grid_size = int(np.ceil(num_threads / 32))

        if grid_size > 1:
            block_size = 32
        else:
            block_size = num_threads

        up_gpu(x_gpu, x_old_gpu, np.int32(k) , block=(block_size,1,1), grid=(grid_size,1,1))
        x_old_gpu[:] = x_gpu[:]

    x_out = x_gpu.get()
    return(x_out)
```

现在我们将开始编写下扫描。同样，让我们从核心开始，它将具有伪代码中内部`parfor`循环的功能。它与之前类似 - 再次使用两个数组，因此在这里使用`temp`变量与伪代码中是不必要的，我们再次使用位移运算符来获得 2^k 和 2^(k+1)的值。我们计算`j`与之前类似：

```py
down_ker = SourceModule("""
__global__ void down_ker(double *y, double *y_old, int k)
{
 int j = blockIdx.x*blockDim.x + threadIdx.x;

 int _2k = 1 << k;
 int _2k1 = 1 << (k+1);

 int j = tid*_2k1;

 y[j + _2k - 1 ] = y_old[j + _2k1 - 1];
 y[j + _2k1 - 1] = y_old[j + _2k1 - 1] + y_old[j + _2k - 1];
}
""")

down_gpu = down_ker.get_function("down_ker")
```

现在我们可以编写一个 Python 函数，该函数将迭代地启动核心，这对应于下扫描阶段的外部`for`循环。这类似于上扫描阶段的 Python 函数。从伪代码中看到的一个重要区别是，我们必须从外部`for`循环中的最大值迭代到最小值；我们可以使用 Python 的`reversed`函数来做到这一点。现在我们可以实现下扫描阶段：

```py

def down_sweep(y):
    y = np.float64(y)
    y[-1] = 0
    y_gpu = gpuarray.to_gpu(y)
    y_old_gpu = y_gpu.copy()
    for k in reversed(range(int(np.log2(y.size)))):
        num_threads = int(np.ceil( y.size / 2**(k+1)))
        grid_size = int(np.ceil(num_threads / 32))

        if grid_size > 1:
            block_size = 32
        else:
            block_size = num_threads

        down_gpu(y_gpu, y_old_gpu, np.int32(k), block=(block_size,1,1), grid=(grid_size,1,1))
        y_old_gpu[:] = y_gpu[:]
    y_out = y_gpu.get()
    return(y_out)
```

在实现了上扫描和下扫描阶段之后，我们的最后任务是轻而易举地完成：

```py
def efficient_prefix(x):
        return(down_sweep(up_sweep(x)))

```

我们现在已经完全实现了一个与主机同步的高效并行前缀算法！（这个实现可以在存储库中的`work-efficient_prefix.py`文件中找到，还有一些测试代码。）

# 总结

我们从康威的*生命游戏*实现开始，这给了我们一个关于 CUDA 核心的许多线程是如何在块-网格张量类型结构中组织的想法。然后，我们通过 CUDA 函数`__syncthreads()`深入研究了块级同步，以及通过使用共享内存进行块级线程互通；我们还看到单个块有一定数量的线程，我们可以操作，所以在创建将使用多个块跨越更大网格的核心时，我们必须小心使用这些功能。

我们概述了并行前缀算法的理论，并最后实现了一个天真的并行前缀算法，作为一个单个核心，可以操作大小受限的数组，该数组与`___syncthreads`同步，并在内部执行`for`和`parfor`循环，并且实现了一个高效的并行前缀算法，该算法跨两个核心和三个 Python 函数实现，可以操作任意大小的数组，核心充当算法的内部`parfor`循环，而 Python 函数有效地充当外部`for`循环并同步核心启动。

# 问题

1.  更改`simple_scalar_multiply_kernel.py`中的随机向量，使其长度为 10,000，并修改内核定义中的`i`索引，以便它可以在网格形式的多个块中使用。看看现在是否可以通过将块和网格参数设置为`block=(100,1,1)`和`grid=(100,1,1)`来启动这个内核超过 10,000 个线程。

1.  在上一个问题中，我们启动了一个同时使用 10,000 个线程的内核；截至 2018 年，没有 NVIDIA GPU 具有超过 5,000 个核心。为什么这仍然有效并产生预期的结果？

1.  天真的并行前缀算法在数据集大小为*n*的情况下，时间复杂度为 O(*log n*)，假设我们有*n*个或更多处理器。假设我们在具有 640 个核心的 GTX 1050 GPU 上使用天真的并行前缀算法。在`n >> 640`的情况下，渐近时间复杂度会变成什么？

1.  修改`naive_prefix.py`以在大小为 1,024 的数组上运行（可能是非二进制的）。

1.  `__syncthreads()` CUDA 设备函数只在单个块中同步线程。我们如何在整个网格的所有块中同步所有线程？

1.  您可以通过这个练习说服自己，第二个前缀和算法确实比天真的前缀和算法更有效。假设我们有一个大小为 32 的数据集。在这种情况下，第一个和第二个算法所需的“加法”操作的确切数量是多少？

1.  在高效并行前缀的实现中，我们使用 Python 函数来迭代我们的内核并同步结果。为什么我们不能在内核中使用`for`循环，同时小心地使用`__syncthreads()`？

1.  为什么在 CUDA C 中实现天真的并行前缀在一个单独的内核中处理自己的同步比使用两个内核和 Python 函数实现高效并行前缀更有意义，并且让主机处理同步更有意义？
