# 使用已编译的GPU代码

在本书的过程中，我们通常依赖PyCUDA库自动为我们接口我们的内联CUDA-C代码，使用即时编译和与Python代码的链接。然而，我们可能还记得，有时编译过程可能需要一段时间。在[第3章](6ab0cd69-e439-4cfb-bf1a-4247ec58c94e.xhtml)中，*使用PyCUDA入门*，我们甚至详细看到编译过程如何导致减速，以及内联代码何时被编译和保留可能是相当随意的。在某些情况下，这可能会给应用程序带来不便，或者在实时系统的情况下甚至是不可接受的。

为此，我们最终将看到如何从Python使用预编译的GPU代码。特别是，我们将看看三种不同的方法来做到这一点。首先，我们将看看如何通过编写一个主机端CUDA-C函数来间接启动CUDA内核。这种方法将涉及使用标准Python Ctypes库调用主机端函数。其次，我们将把我们的内核编译成所谓的PTX模块，这实际上是一个包含已编译二进制GPU的DLL文件。然后，我们可以使用PyCUDA加载此文件并直接启动我们的内核。最后，我们将通过查看如何编写我们自己的完整Ctypes接口来结束本章，以使用CUDA Driver API。然后，我们可以使用Driver API中的适当函数加载我们的PTX文件并启动内核。

本章的学习成果如下：

+   使用Ctypes模块启动编译后（主机端）的代码

+   使用Ctypes使用主机端CUDA C包装器从Python启动内核

+   如何将CUDA C模块编译为PTX文件

+   如何将PTX模块加载到PyCUDA中以启动预编译的内核

+   如何编写自定义Python接口以使用CUDA Driver API

# 使用Ctypes启动编译后的代码

我们现在将简要概述Python标准库中的Ctypes模块。Ctypes用于调用来自Linux`.so`（共享对象）或Windows.DLL（动态链接库）预编译二进制文件的函数。这将使我们摆脱纯Python的世界，并与已用编译语言编写的库和代码进行接口，特别是C和C++ - 恰好Nvidia只为与我们的CUDA设备进行接口提供这样的预编译二进制文件，因此如果我们想绕过PyCUDA，我们将不得不使用Ctypes。

让我们从一个非常基本的例子开始：我们将向您展示如何直接从Ctypes调用`printf`。打开一个IPython实例，键入`import ctypes`。现在我们将看看如何从Ctypes调用标准的`printf`函数。首先，我们必须导入适当的库：在Linux中，通过键入`libc = ctypes.CDLL('libc.so.6')`加载LibC库（在Windows中，将`'libc.so.6'`替换为`'msvcrt.dll'`）。现在我们可以通过在IPython提示符中键入`libc.printf("Hello from ctypes!\n")`直接调用`printf`。自己试试吧！

现在让我们试试其他东西：从IPython键入`libc.printf("Pi is approximately %f.\n", 3.14)`；您应该会收到一个错误。这是因为`3.14`没有适当地从Python浮点变量转换为C双精度变量 - 我们可以使用Ctypes这样做：

```py
libc.printf("Pi is approximately %f.\n", ctypes.c_double(3.14)) 
```

输出应该如预期那样。与从PyCUDA启动CUDA内核的情况一样，我们必须同样小心地将输入转换为Ctypes函数。

请务必确保将任何从Python使用Ctypes调用的函数的输入适当地转换为适当的C数据类型（在Ctypes中，这些类型以c_开头：`c_float`、`c_double`、`c_char`、`c_int`等）。

# 再次重温Mandelbrot集

让我们重新审视一下我们在[第1章](f9c54d0e-6a18-49fc-b04c-d44a95e011a2.xhtml)和[第3章](6ab0cd69-e439-4cfb-bf1a-4247ec58c94e.xhtml)中看到的Mandelbrot集合，*为什么使用GPU编程？*和*使用PyCUDA入门*。首先，我们将编写一个完整的CUDA核函数，它将根据一组特定的参数计算Mandelbrot集合，以及一个适当的主机端包装函数，我们稍后可以从Ctypes接口调用。我们将首先将这些函数编写到一个单独的CUDA-C`.cu`源文件中，然后使用NVCC编译成DLL或`.so`二进制文件。最后，我们将编写一些Python代码，以便我们可以运行我们的二进制代码并显示Mandelbrot集合。

我们现在将运用我们对Ctypes的知识，从Python中启动一个预编译的CUDA核函数，而不需要PyCUDA的任何帮助。这将要求我们在CUDA-C中编写一个主机端*核函数启动器*包装函数，我们可以直接调用，它本身已经编译成了一个动态库二进制文件，其中包含任何必要的GPU代码——即在Windows上的动态链接库（DLL）二进制文件，或者在Linux上的共享对象（so）二进制文件。

当然，我们将首先编写我们的CUDA-C代码，所以打开你最喜欢的文本编辑器并跟着做。我们将从标准的`include`语句开始：

```py
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
```

我们现在将直接开始编写我们的核函数。请注意代码中的`extern "C"`，这将允许我们在外部链接到这个函数：

```py
extern "C" __global__ void mandelbrot_ker(float * lattice, float * mandelbrot_graph, int max_iters, float upper_bound_squared, int lattice_size)
{
```

让我们思考一分钟关于这将如何工作：我们将使用一个单一的一维数组来存储实部和虚部，称为`lattice`，其长度为`lattice_size`。我们将使用这个数组来计算一个形状为(`lattice_size`, `lattice_size`)的二维Mandelbrot图形，存储在预先分配的数组`mandelbrot_graph`中。我们将指定每个点检查发散的迭代次数为`max_iters`，通过使用`upper_bound_squared`提供其平方值来指定之前的最大上限值。（我们稍后会看一下使用平方的动机。）

我们将在一维网格/块结构上启动这个核函数，每个线程对应于Mandelbrot集合图像中的一个点。然后我们可以确定相应点的实部/虚部lattice值，如下所示：

```py
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( tid < lattice_size*lattice_size )
    {
        int i = tid % lattice_size;
        int j = lattice_size - 1 - (tid / lattice_size);

        float c_re = lattice[i];
        float c_im = lattice[j];
```

让我们花一分钟来谈谈这个。首先，记住我们可能需要使用稍多于必要的线程，所以重要的是我们检查线程ID是否对应于输出图像中的某个点，使用`if`语句。让我们还记住，输出数组`mandelbrot_graph`将以按行方式存储为一维数组，表示为二维图像，我们将使用`tid`作为写入该数组的索引。我们将使用`i`和`j`，以及复平面上图形的`x`和`y`坐标。由于lattice是一系列从小到大排序的实值，我们将不得不颠倒它们的顺序以获得适当的虚值。另外，请注意，我们将在这里使用普通的浮点数，而不是某种结构或对象来表示复数值。由于每个复数中都有实部和虚部，我们将在这里使用两个浮点数来存储与该线程的lattice点对应的复数（`c_re`和`c_im`）。

我们将设置两个更多的变量来处理发散检查，`z_re`和`z_im`，并在检查发散之前将该线程的图上的点的初始值设置为`1`：

```py
        float z_re = 0.0f;
        float z_im = 0.0f;

        mandelbrot_graph[tid] = 1;
```

现在我们将检查发散；如果在`max_iters`次迭代后发散，我们将把点设置为`0`。否则，它将保持为1：

```py
        for (int k = 0; k < max_iters; k++)
        {
            float temp;

            temp = z_re*z_re - z_im*z_im + c_re;
            z_im = 2*z_re*z_im + c_im;
            z_re = temp;

            if ( (z_re*z_re + z_im*z_im) > upper_bound_squared )
            {
                mandelbrot_graph[tid] = 0;
                break;
            }
        }
```

在我们继续之前，让我们谈一分钟关于这一块代码。让我们记住，曼德勃罗集的每次迭代都是通过复数乘法和加法来计算的，例如，`z_new = z*z + c`。由于我们不是使用将为我们处理复数值的类，前面的操作正是我们需要做的，以计算`z`的新的实部和虚部值。我们还需要计算绝对值并查看是否超过特定值——记住，复数的绝对值，`c = x + iy`，是用*√(x²+y²)*来计算的。在这里计算上限的平方然后将其插入内核中，实际上会节省我们一些时间，因为这样可以节省我们在每次迭代中计算`z_re*z_re + z_im*z_im`的平方根的时间。

我们现在基本上已经完成了这个内核——我们只需要关闭`if`语句并从内核返回，然后我们就完成了：

```py
    }
    return;
}
```

然而，我们还没有完全完成。我们需要编写一个只有`extern "C"`的主机端包装函数，在Linux的情况下，以及在Windows的情况下，只有`extern "C" __declspec(dllexport)`。 （与编译的CUDA内核相反，如果我们想要能够从Ctypes在Windows中访问主机端函数，这个额外的单词是必要的。）我们放入这个函数的参数将直接对应于进入内核的参数，除了这些参数将存储在主机上：

```py
extern "C" __declspec(dllexport) void launch_mandelbrot(float * lattice,  float * mandelbrot_graph, int max_iters, float upper_bound, int lattice_size)
{
```

现在，我们将需要分配足够的内存来存储在GPU上的晶格和输出，然后使用`cudaMemcpy`将晶格复制到GPU上：

```py
    int num_bytes_lattice = sizeof(float) * lattice_size;
    int num_bytes_graph = sizeof(float)* lattice_size*lattice_size;

    float * d_lattice;
    float * d_mandelbrot_graph;

    cudaMalloc((float **) &d_lattice, num_bytes_lattice);
    cudaMalloc((float **) &d_mandelbrot_graph, num_bytes_graph);

    cudaMemcpy(d_lattice, lattice, num_bytes_lattice, cudaMemcpyHostToDevice);
```

像我们的许多其他内核一样，我们将在一维网格上启动大小为32的一维块。我们将取输出点数除以32的上限值，以确定网格大小，如下所示：

```py
    int grid_size = (int)  ceil(  ( (double) lattice_size*lattice_size ) / ( (double) 32 ) );
```

现在我们准备使用传统的CUDA-C三角形括号来启动我们的内核，以指定网格和块大小。请注意，我们在这里提前求出了上限的平方：

```py
    mandelbrot_ker <<< grid_size, 32 >>> (d_lattice,  d_mandelbrot_graph, max_iters, upper_bound*upper_bound, lattice_size);
```

现在我们只需要在完成这些操作后将输出复制到主机上，然后在适当的数组上调用`cudaFree`。然后我们可以从这个函数返回：

```py
    cudaMemcpy(mandelbrot_graph, d_mandelbrot_graph, num_bytes_graph, cudaMemcpyDeviceToHost);    
    cudaFree(d_lattice);
    cudaFree(d_mandelbrot_graph);
}
```

有了这个，我们已经完成了所有需要的CUDA-C代码。将其保存到名为`mandelbrot.cu`的文件中，然后继续下一步。

您还可以从[https://github.com/btuomanen/handsongpuprogramming/blob/master/10/mandelbrot.cu](https://github.com/btuomanen/handsongpuprogramming/blob/master/10/mandelbrot.cu)下载此文件。

# 编译代码并与Ctypes进行接口

现在让我们将刚刚编写的代码编译成DLL或`.so`二进制文件。这实际上相当简单：如果你是Linux用户，请在命令行中输入以下内容将此文件编译成`mandelbrot.so`：

```py
nvcc -Xcompiler -fPIC -shared -o mandelbrot.so mandelbrot.cu
```

如果你是Windows用户，请在命令行中输入以下内容将文件编译成`mandelbrot.dll`：

```py
nvcc -shared -o mandelbrot.dll mandelbrot.cu
```

现在我们可以编写我们的Python接口。我们将从适当的导入语句开始，完全排除PyCUDA，只使用Ctypes。为了方便使用，我们将直接从Ctypes导入所有类和函数到默认的Python命名空间中，如下所示：

```py
from __future__ import division
from time import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from ctypes import *
```

让我们使用Ctypes为`launch_mandelbrot`主机端函数设置一个接口。首先，我们将不得不加载我们编译的DLL或`.so`文件，Linux用户当然需要将文件名更改为`mandelbrot.so`：

```py
mandel_dll = CDLL('./mandelbrot.dll')
```

现在我们可以从库中获取对`launch_mandelbrot`的引用，就像这样；我们将简称它为`mandel_c`：

```py
mandel_c = mandel_dll.launch_mandelbrot
```

现在，在使用Ctypes调用函数之前，我们将不得不让Ctypes知道输入类型是什么。让我们记住，对于`launch_mandelbrot`，输入是`float-pointer`，`float-pointer`，`integer`，`float`和`integer`。我们使用`argtypes`参数设置这一点，使用适当的Ctypes数据类型（`c_float`，`c_int`），以及Ctypes的`POINTER`类：

```py
mandel_c.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_float, c_int]
```

现在让我们编写一个Python函数来为我们运行这个。我们将使用`breadth`指定正方形输出图像的宽度和高度，并且在复杂格的实部和虚部中指定最小和最大值。我们还将指定最大迭代次数以及上限：

```py
def mandelbrot(breadth, low, high, max_iters, upper_bound):
```

现在，我们将使用NumPy的`linspace`函数创建我们的格点数组，就像这样：

```py
 lattice = np.linspace(low, high, breadth, dtype=np.float32)
```

让我们记住，我们将不得不传递一个预先分配的浮点数组给`launch_mandelbrot`，以便以输出图的形式得到输出。我们可以通过调用NumPy的`empty`命令来设置一个适当形状和大小的数组，这将充当C的`malloc`调用：

```py
    out = np.empty(shape=(lattice.size,lattice.size), dtype=np.float32)
```

现在，我们准备计算Mandelbrot图。请注意，我们可以通过使用它们的`ctypes.data_as`方法和相应的类型将NumPy数组传递给C。在我们这样做之后，我们可以返回输出；也就是说，Mandelbrot图以二维NumPy数组的形式：

```py
 mandel_c(lattice.ctypes.data_as(POINTER(c_float)), out.ctypes.data_as(POINTER(c_float)), c_int(max_iters), c_float(upper_bound), c_int(lattice.size) ) 
 return out
```

现在，让我们编写我们的主函数来计算、计时和使用Matplotlib查看Mandelbrot图：

```py
if __name__ == '__main__':
    t1 = time()
    mandel = mandelbrot(512,-2,2,256, 2)
    t2 = time()
    mandel_time = t2 - t1
    print 'It took %s seconds to calculate the Mandelbrot graph.' % mandel_time
    plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.show()
```

我们现在将尝试运行这个。您应该会得到一个看起来与[第1章](f9c54d0e-6a18-49fc-b04c-d44a95e011a2.xhtml)的Mandelbrot图以及[第3章](6ab0cd69-e439-4cfb-bf1a-4247ec58c94e.xhtml)的*为什么使用GPU编程*和*使用PyCUDA入门*中的图形完全相同的输出：

![](assets/0620985f-949b-4f6b-bba4-1be7b0bf7eff.png)

这个Python示例的代码也可以在GitHub存储库的`mandelbrot_ctypes.py`文件中找到。

# 编译和启动纯PTX代码

我们刚刚看到了如何从Ctypes调用纯C函数。在某些方面，这可能看起来有点不够优雅，因为我们的二进制文件必须包含主机代码以及编译后的GPU代码，这可能看起来很麻烦。我们是否可以只使用纯粹的编译后的GPU代码，然后适当地将其启动到GPU上，而不是每次都编写一个C包装器？幸运的是，我们可以。

NVCC编译器将CUDA-C编译为**PTX**（**Parallel Thread Execution**），这是一种解释的伪汇编语言，与NVIDIA的各种GPU架构兼容。每当您使用NVCC将使用CUDA核心的程序编译为可执行的EXE、DLL、`.so`或ELF文件时，该文件中将包含该核心的PTX代码。我们还可以直接编译具有PTX扩展名的文件，其中将仅包含从编译后的CUDA .cu文件中编译的GPU核心。幸运的是，PyCUDA包括一个接口，可以直接从PTX加载CUDA核心，使我们摆脱了即时编译的枷锁，同时仍然可以使用PyCUDA的所有其他好功能。

现在让我们将刚刚编写的Mandelbrot代码编译成一个PTX文件；我们不需要对它进行任何更改。只需在Linux或Windows的命令行中输入以下内容：

```py
nvcc -ptx -o mandelbrot.ptx mandelbrot.cu
```

现在让我们修改上一节的Python程序，以使用PTX代码。我们将从导入中删除`ctypes`并添加适当的PyCUDA导入：

```py
from __future__ import division
from time import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pycuda
from pycuda import gpuarray
import pycuda.autoinit
```

现在让我们使用PyCUDA的`module_from_file`函数加载PTX文件，就像这样：

```py
mandel_mod = pycuda.driver.module_from_file('./mandelbrot.ptx')
```

现在我们可以使用`get_function`来获取对我们的核心的引用，就像我们用PyCUDA的`SourceModule`一样：

```py
mandel_ker = mandel_mod.get_function('mandelbrot_ker')
```

我们现在可以重写Mandelbrot函数，以处理使用适当的`gpuarray`对象和`typecast`输入的核心。（我们不会逐行讨论这个，因为在这一点上它的功能应该是显而易见的。）：

```py
def mandelbrot(breadth, low, high, max_iters, upper_bound):
    lattice = gpuarray.to_gpu(np.linspace(low, high, breadth, dtype=np.   
    out_gpu = gpuarray.empty(shape=(lattice.size,lattice.size), dtype=np.float32)
    gridsize = int(np.ceil(lattice.size**2 / 32))
    mandel_ker(lattice, out_gpu, np.int32(256), np.float32(upper_bound**2), np.int32(lattice.size), grid=(gridsize, 1, 1), block=(32,1,1))
    out = out_gpu.get()

    return out
```

`main`函数将与上一节完全相同：

```py
if __name__ == '__main__':
    t1 = time()
    mandel = mandelbrot(512,-2,2,256,2)
    t2 = time()
    mandel_time = t2 - t1
    print 'It took %s seconds to calculate the Mandelbrot graph.' % mandel_time
    plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.show()
```

现在，尝试运行这个来确保输出是正确的。您可能还会注意到与Ctypes版本相比的一些速度改进。

这段代码也可以在本书GitHub存储库的“10”目录下的`mandelbrot_ptx.py`文件中找到。

# 为CUDA Driver API编写包装器

现在我们将看看如何使用Ctypes编写我们自己的包装器，用于一些预打包的二进制CUDA库函数。特别是，我们将为CUDA驱动程序API编写包装器，这将允许我们执行所有基本GPU使用所需的操作，包括GPU初始化、内存分配/传输/释放、内核启动和上下文创建/同步/销毁。这是一个非常强大的知识；它将允许我们在不经过PyCUDA的情况下使用GPU，也不需要编写任何繁琐的主机端C函数包装器。

现在我们将编写一个小模块，它将作为**CUDA驱动程序API**的包装库。让我们谈一分钟这对我们意味着什么。驱动程序API与**CUDA Runtime API**略有不同，技术性稍高，后者是我们在CUDA-C文本中一直在使用的。驱动程序API旨在与常规C/C++编译器一起使用，而不是与NVCC一起使用，具有一些不同的约定，如使用`cuLaunchKernel`函数启动内核，而不是使用`<<< gridsize, blocksize >>>`括号表示法。这将允许我们直接访问使用Ctypes从PTX文件启动内核所需的函数。

让我们通过将所有Ctypes导入模块的命名空间，并导入sys模块来开始编写此模块。我们将通过使用`sys.platform`检查系统的操作系统（`nvcuda.dll`或`libcuda.so`）来加载适当的库文件，使我们的模块可以在Windows和Linux上使用。

```py
from ctypes import *
import sys
if 'linux' in sys.platform:
 cuda = CDLL('libcuda.so')
elif 'win' in sys.platform:
 cuda = CDLL('nvcuda.dll')
```

我们已经成功加载了CUDA驱动程序API，现在我们可以开始为基本GPU使用编写必要函数的包装器。随着我们的进行，我们将查看每个驱动程序API函数的原型，这通常是在编写Ctypes包装器时必要的。

鼓励读者在官方Nvidia CUDA驱动程序API文档中查找我们将在本节中使用的所有函数，该文档可在此处找到：[https://docs.nvidia.com/cuda/cuda-driver-api/](https://docs.nvidia.com/cuda/cuda-driver-api/)。

让我们从驱动程序API中最基本的函数`cuInit`开始，它将初始化驱动程序API。这需要一个用于标志的无符号整数作为输入参数，并返回类型为CUresult的值，实际上只是一个整数值。我们可以这样编写我们的包装器：

```py
cuInit = cuda.cuInit
cuInit.argtypes = [c_uint]
cuInit.restype = int
```

现在让我们开始下一个函数`cuDeviceCount`，它将告诉我们在计算机上安装了多少个NVIDIA GPU。它以整数指针作为其唯一输入，实际上是通过引用返回的单个整数输出值。返回值是另一个CUresult整数——所有函数都将使用CUresult，这是所有驱动程序API函数的错误值的标准化。例如，如果我们看到任何函数返回`0`，这意味着结果是`CUDA_SUCCESS`，而非零结果将始终意味着错误或警告：

```py
cuDeviceGetCount = cuda.cuDeviceGetCount
cuDeviceGetCount.argtypes = [POINTER(c_int)]
cuDeviceGetCount.restype = int
```

现在让我们为`cuDeviceGet`编写一个包装器，它将通过引用在第一个输入中返回设备句柄。这将对应于第二个输入中给定的序号GPU。第一个参数的类型是`CUdevice *`，实际上只是一个整数指针：

```py
cuDeviceGet = cuda.cuDeviceGet
cuDeviceGet.argtypes = [POINTER(c_int), c_int]
cuDeviceGet.restype = int
```

让我们记住，每个CUDA会话都需要至少一个CUDA上下文，可以将其类比为在CPU上运行的进程。由于Runtime API会自动处理这一点，在这里我们将不得不在使用设备之前手动在设备上创建上下文（使用设备句柄），并且在CUDA会话结束时销毁此上下文。

我们可以使用`cuCtxCreate`函数创建一个CUDA上下文，它当然会创建一个上下文。让我们看看文档中列出的原型：

```py
 CUresult cuCtxCreate ( CUcontext* pctx, unsigned int flags, CUdevice dev )
```

当然，返回值是`CUresult`。第一个输入是指向名为`CUcontext`的类型的指针，实际上它本身是CUDA内部使用的特定C结构的指针。由于我们从Python对`CUcontext`的唯一交互将是保持其值以在其他函数之间传递，我们可以将`CUcontext`存储为C `void *`类型，用于存储任何类型的通用指针地址。由于这实际上是指向CU上下文的指针（再次，它本身是指向内部数据结构的指针——这是另一个按引用返回值），我们可以将类型设置为普通的`void *`，这是Ctypes中的`c_void_p`类型。第二个值是无符号整数，而最后一个值是要在其上创建新上下文的设备句柄——让我们记住这实际上只是一个整数。我们现在准备为`cuCtxCreate`创建包装器：

```py
cuCtxCreate = cuda.cuCtxCreate
cuCtxCreate.argtypes = [c_void_p, c_uint, c_int]
cuCtxCreate.restype = int
```

您可以始终在C/C++（Ctypes中的`c_void_p`）中使用`void *`类型指向任意数据或变量，甚至结构和对象，其定义可能不可用。

下一个函数是`cuModuleLoad`，它将为我们加载一个PTX模块文件。第一个参数是一个CUmodule的引用（同样，我们可以在这里使用`c_void_p`），第二个是文件名，这将是一个典型的以空字符结尾的C字符串——这是一个`char *`，或者在Ctypes中是`c_char_p`：

```py
cuModuleLoad = cuda.cuModuleLoad
cuModuleLoad.argtypes = [c_void_p, c_char_p]
cuModuleLoad.restype = int
```

下一个函数用于同步当前CUDA上下文中的所有启动操作，并称为`cuCtxSynchronize`（不带参数）：

```py
cuCtxSynchronize = cuda.cuCtxSynchronize
cuCtxSynchronize.argtypes = []
cuCtxSynchronize.restype = int
```

下一个函数用于从加载的模块中检索内核函数句柄，以便我们可以将其启动到GPU上，这与PyCUDA的`get_function`方法完全对应，这一点我们已经看过很多次了。文档告诉我们原型是`CUresult cuModuleGetFunction ( CUfunction* hfunc, CUmodule hmod, const char* name )`。现在我们可以编写包装器：

```py
cuModuleGetFunction = cuda.cuModuleGetFunction
 cuModuleGetFunction.argtypes = [c_void_p, c_void_p, c_char_p ]
 cuModuleGetFunction.restype = int
```

现在让我们为标准动态内存操作编写包装器；这些将是必要的，因为我们将不再有使用PyCUDA gpuarray对象的虚荣。这些实际上与我们之前使用过的CUDA运行时操作几乎相同；也就是说，`cudaMalloc`，`cudaMemcpy`和`cudaFree`：

```py
cuMemAlloc = cuda.cuMemAlloc
cuMemAlloc.argtypes = [c_void_p, c_size_t]
cuMemAlloc.restype = int

cuMemcpyHtoD = cuda.cuMemcpyHtoD
cuMemcpyHtoD.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemAlloc.restype = int

cuMemcpyDtoH = cuda.cuMemcpyDtoH
cuMemcpyDtoH.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemcpyDtoH.restype = int

cuMemFree = cuda.cuMemFree
cuMemFree.argtypes = [c_void_p] 
cuMemFree.restype = int
```

现在，我们将为`cuLaunchKernel`函数编写一个包装器。当然，这是我们将用来在GPU上启动CUDA内核的函数，前提是我们已经初始化了CUDA Driver API，设置了上下文，加载了一个模块，分配了内存并配置了输入，并且已经从加载的模块中提取了内核函数句柄。这个函数比其他函数复杂一些，所以我们将看一下原型：

```py
CUresult cuLaunchKernel ( CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra )  
```

第一个参数是我们要启动的内核函数的句柄，我们可以表示为`c_void_p`。六个`gridDim`和`blockDim`参数用于指示网格和块的维度。无符号整数`sharedMemBytes`用于指示在内核启动时为每个块分配多少字节的共享内存。`CUstream hStream`是一个可选参数，我们可以使用它来设置自定义流，或者如果希望使用默认流，则设置为NULL（0），我们可以在Ctypes中表示为`c_void_p`。最后，`kernelParams`和`extra`参数用于设置内核的输入；这些有点复杂，所以现在只需知道我们也可以将这些表示为`c_void_p`：

```py
cuLaunchKernel = cuda.cuLaunchKernel
cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_void_p, c_void_p, c_void_p]
cuLaunchKernel.restype = int
```

现在我们还有最后一个函数要为`cuCtxDestroy`编写一个包装器。我们在CUDA会话结束时使用它来销毁GPU上的上下文。唯一的输入是一个`CUcontext`对象，由`c_void_p`表示：

```py
cuCtxDestroy = cuda.cuCtxDestroy
cuCtxDestroy.argtypes = [c_void_p]
cuCtxDestroy.restype = int
```

让我们把这个保存到`cuda_driver.py`文件中。我们现在已经完成了Driver API包装器模块！接下来，我们将看看如何仅使用我们的模块和Mandelbrot PTX加载一个PTX模块并启动一个内核。

这个示例也可以在本书的GitHub存储库中的`cuda_driver.py`文件中找到。

# 使用CUDA Driver API

我们现在将翻译我们的小曼德布洛特生成程序，以便我们可以使用我们的包装库。让我们从适当的导入语句开始；注意我们如何将所有的包装器加载到当前命名空间中：

```py
from __future__ import division
from time import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from cuda_driver import *
```

让我们把所有的GPU代码放到`mandelbrot`函数中，就像以前一样。我们将从使用`cuInit`初始化CUDA Driver API开始，然后检查系统上是否安装了至少一个GPU，否则会引发异常：

```py
def mandelbrot(breadth, low, high, max_iters, upper_bound):
 cuInit(0)
 cnt = c_int(0)
 cuDeviceGetCount(byref(cnt))
 if cnt.value == 0:
  raise Exception('No GPU device found!')
```

注意这里的`byref`：这是Ctypes中引用操作符(`&`)的等价物。我们现在将再次应用这个想法，记住设备句柄和CUDA上下文可以用Ctypes表示为`c_int`和`c_void_p`：

```py
 cuDevice = c_int(0)
 cuDeviceGet(byref(cuDevice), 0)
 cuContext = c_void_p()
 cuCtxCreate(byref(cuContext), 0, cuDevice)
```

我们现在将加载我们的PTX模块，记得要使用`c_char_p`将文件名转换为C字符串：

```py
 cuModule = c_void_p()
 cuModuleLoad(byref(cuModule), c_char_p('./mandelbrot.ptx'))
```

现在我们将在主机端设置晶格，以及一个名为`graph`的用于在主机端存储输出的全零NumPy数组。我们还将为晶格和图形输出在GPU上分配内存，然后使用`cuMemcpyHtoD`将晶格复制到GPU上：

```py
 lattice = np.linspace(low, high, breadth, dtype=np.float32)
 lattice_c = lattice.ctypes.data_as(POINTER(c_float))
 lattice_gpu = c_void_p(0)
 graph = np.zeros(shape=(lattice.size, lattice.size), dtype=np.float32)
 cuMemAlloc(byref(lattice_gpu), c_size_t(lattice.size*sizeof(c_float)))
 graph_gpu = c_void_p(0)
 cuMemAlloc(byref(graph_gpu), c_size_t(lattice.size**2 * sizeof(c_float)))
 cuMemcpyHtoD(lattice_gpu, lattice_c, c_size_t(lattice.size*sizeof(c_float)))
```

现在我们将使用`cuModuleGetFunction`获取Mandelbrot内核的句柄，并设置一些输入：

```py
 mandel_ker = c_void_p(0)
 cuModuleGetFunction(byref(mandel_ker), cuModule, c_char_p('mandelbrot_ker'))
 max_iters = c_int(max_iters)
 upper_bound_squared = c_float(upper_bound**2)
 lattice_size = c_int(lattice.size)
```

下一步有点复杂。在继续之前，我们必须了解参数是如何通过`cuLaunchKernel`传递到CUDA内核中的。让我们先看看CUDA-C中是如何工作的。

我们将输入参数在`kernelParams`中表达为一个`void *`值的数组，它们本身是指向我们希望插入内核的输入的指针。对于我们的曼德布洛特内核，它看起来像这样：

```py
void * mandel_params [] = {&lattice_gpu, &graph_gpu, &max_iters, &upper_bound_squared, &lattice_size};
```

现在让我们看看如何在Ctypes中表达这一点，这并不是立即显而易见的。首先，让我们将所有的输入放入一个Python列表中，按正确的顺序：

```py
mandel_args0 = [lattice_gpu, graph_gpu, max_iters, upper_bound_squared, lattice_size ]
```

现在我们需要每个值的指针，将其类型转换为`void *`类型。让我们使用Ctypes函数`addressof`来获取每个Ctypes变量的地址（类似于`byref`，只是不绑定到特定类型），然后将其转换为`c_void_p`。我们将这些值存储在另一个列表中：

```py
mandel_args = [c_void_p(addressof(x)) for x in mandel_args0]
```

现在让我们使用Ctypes将这个Python列表转换成一个`void *`指针数组，就像这样：

```py
 mandel_params = (c_void_p * len(mandel_args))(*mandel_args)
```

现在我们可以设置网格的大小，就像以前一样，并使用`cuLaunchKernel`启动我们的内核，使用这组参数。然后我们在之后同步上下文：

```py
 gridsize = int(np.ceil(lattice.size**2 / 32))
 cuLaunchKernel(mandel_ker, gridsize, 1, 1, 32, 1, 1, 10000, None, mandel_params, None)
 cuCtxSynchronize()
```

我们现在将使用`cuMemcpyDtoH`将数据从GPU复制到我们的NumPy数组中，使用NumPy的`array.ctypes.data`成员，这是一个C指针，将允许我们直接从C中访问数组作为堆内存的一部分。我们将使用Ctypes的类型转换函数`cast`将其转换为`c_void_p`：

```py
 cuMemcpyDtoH( cast(graph.ctypes.data, c_void_p), graph_gpu,  c_size_t(lattice.size**2 *sizeof(c_float)))
```

我们现在完成了！让我们释放在GPU上分配的数组，并通过销毁当前上下文来结束我们的GPU会话。然后我们将把图形NumPy数组返回给调用函数：

```py
 cuMemFree(lattice_gpu)
 cuMemFree(graph_gpu)
 cuCtxDestroy(cuContext)
 return graph
```

现在我们可以像以前一样设置我们的`main`函数：

```py
if __name__ == '__main__':
 t1 = time()
 mandel = mandelbrot(512,-2,2,256, 2)
 t2 = time()
 mandel_time = t2 - t1
 print 'It took %s seconds to calculate the Mandelbrot graph.' % mandel_time

 fig = plt.figure(1)
 plt.imshow(mandel, extent=(-2, 2, -2, 2))
 plt.show()
```

现在尝试运行这个函数，确保它产生与我们刚刚编写的其他曼德布洛特程序相同的输出。

恭喜你——你刚刚编写了一个直接接口到低级CUDA Driver API，并成功使用它启动了一个内核！

这个程序也可以在本书的GitHub存储库中的目录下的`mandelbrot_driver.py`文件中找到。

# 总结

我们从简要概述Python Ctypes库开始了本章，该库用于直接与编译的二进制代码进行接口，特别是用C/C++编写的动态库。然后，我们看了如何使用CUDA-C编写一个启动CUDA内核的基于C的包装器，然后使用这个包装器间接地从Python启动我们的CUDA内核，方法是使用Ctypes编写一个对这个函数的接口。然后，我们学习了如何将CUDA内核编译成PTX模块二进制文件，可以将其视为一个带有CUDA内核函数的DLL，并看到如何使用PyCUDA加载PTX文件并启动预编译的内核。最后，我们编写了一系列CUDA Driver API的Ctypes包装器，并看到我们如何使用这些包装器执行基本的GPU操作，包括从PTX文件启动预编译的内核到GPU上。

我们现在将进入本书中可能是最技术性的一章：第11章，《CUDA性能优化》。在本章中，我们将学习关于NVIDIA GPU的一些技术细节，这将帮助我们提高应用程序的性能水平。

# 问题

1.  假设您使用`nvcc`将包含主机和内核代码的单个`.cu`文件编译成EXE文件，还编译成PTX文件。哪个文件将包含主机函数，哪个文件将包含GPU代码？

1.  如果我们使用CUDA Driver API，为什么要销毁上下文？

1.  在本章开始时，当我们首次看到如何使用Ctypes时，请注意在调用`printf`之前，我们必须将浮点值3.14强制转换为Ctypes的`c_double`对象。然而，在本章中我们可以看到许多不需要将类型转换为Ctypes的工作案例。你认为为什么`printf`在这里是一个例外呢？

1.  假设您想要向我们的Python CUDA Driver接口模块添加功能以支持CUDA流。您将如何在Ctypes中表示单个流对象？

1.  为什么在`mandelbrot.cu`中的函数要使用`extern "C"`？

1.  再次查看`mandelbrot_driver.py`。为什么我们在GPU内存分配和主机/GPU内存传输之后*不*使用`cuCtxSynchronize`函数，而只在单个内核调用之后使用？
