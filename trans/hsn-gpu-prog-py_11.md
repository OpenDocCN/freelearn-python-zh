# CUDA 中的性能优化

在这个倒数第二章中，我们将介绍一些相当高级的 CUDA 功能，可以用于低级性能优化。我们将首先学习动态并行性，它允许内核在 GPU 上启动和管理其他内核，并看看我们如何使用它直接在 GPU 上实现快速排序。我们将学习关于矢量化内存访问，可以用于增加从 GPU 全局内存读取时的内存访问加速。然后我们将看看如何使用 CUDA 原子操作，这些是线程安全的函数，可以在没有线程同步或*mutex*锁的情况下操作共享数据。我们将学习关于 Warp，它是 32 个或更少线程的基本块，在这些线程可以直接读取或写入彼此的变量，然后简要地涉足 PTX 汇编的世界。我们将通过直接在我们的 CUDA-C 代码中内联编写一些基本的 PTX 汇编来做到这一点，这本身将内联在我们的 Python 代码中！最后，我们将把所有这些小的低级调整结合到一个最终的例子中，我们将应用它们来制作一个快速的求和内核，并将其与 PyCUDA 的求和进行比较。

本章的学习成果如下：

+   CUDA 中的动态并行性

+   在 GPU 上使用动态并行性实现快速排序

+   使用矢量化类型加速设备内存访问

+   使用线程安全的 CUDA 原子操作

+   基本的 PTX 汇编

+   将所有这些概念应用于编写性能优化的求和内核

# 动态并行性

首先，我们将看一下**动态并行性**，这是 CUDA 中的一个功能，允许内核在 GPU 上启动和管理其他内核，而无需主机的任何交互或输入。这也使得许多通常在 GPU 上可用的主机端 CUDA-C 功能也可用于 GPU，例如设备内存分配/释放，设备到设备的内存复制，上下文范围的同步和流。

让我们从一个非常简单的例子开始。我们将创建一个小内核，覆盖*N*个线程，每个线程将向终端打印一条简短的消息，然后递归地启动另一个覆盖*N-1*个线程的内核。这个过程将持续，直到*N*达到 1。(当然，除了说明动态并行性如何工作之外，这个例子将毫无意义。)

让我们从 Python 中的`import`语句开始：

```py
from __future__ import division
import numpy as np
from pycuda.compiler import DynamicSourceModule
import pycuda.autoinit
```

请注意，我们必须导入`DynamicSourceModule`而不是通常的`SourceModule`！这是因为动态并行性功能需要编译器设置特定的配置细节。否则，这将看起来和行为像通常的`SourceModule`操作。现在我们可以继续编写内核：

```py
DynamicParallelismCode='''
__global__ void dynamic_hello_ker(int depth)
{
 printf("Hello from thread %d, recursion depth %d!\\n", threadIdx.x, depth);
 if (threadIdx.x == 0 && blockIdx.x == 0 && blockDim.x > 1)
 {
  printf("Launching a new kernel from depth %d .\\n", depth);
  printf("-----------------------------------------\\n");
  dynamic_hello_ker<<< 1, blockDim.x - 1 >>>(depth + 1);
 }
}'''
```

在这里需要注意的最重要的一点是：我们必须小心，只有一个线程启动下一个迭代的内核，使用一个良好放置的`if`语句来检查`threadIdx`和`blockIdx`的值。如果我们不这样做，那么每个线程将在每个深度迭代中启动远多于必要的内核实例。另外，注意我们可以以正常方式启动内核，使用通常的 CUDA-C 三重括号表示法——我们不必使用任何晦涩或低级命令来利用动态并行性。

在使用 CUDA 动态并行性功能时，一定要小心避免不必要的内核启动。这可以通过指定的线程启动下一个内核迭代来实现。

现在让我们结束这一切：

```py
dp_mod = DynamicSourceModule(DynamicParallelismCode)
hello_ker = dp_mod.get_function('dynamic_hello_ker')
hello_ker(np.int32(0), grid=(1,1,1), block=(4,1,1))
```

现在我们可以运行前面的代码，这将给我们以下输出：

![](img/5147fd83-75b9-4dd4-8e5f-07a8cf013436.png)

这个例子也可以在本书的 GitHub 存储库中的`dynamic_hello.py`文件中找到。

# 具有动态并行性的快速排序

现在让我们来看一个稍微有趣和实用的动态并行应用——**快速排序算法**。实际上，这是一个非常适合并行化的算法，我们将会看到。

让我们先简要回顾一下。快速排序是一种递归和原地排序算法，平均和最佳情况下的性能为*O(N log N)*，最坏情况下的性能为*O(N²)*。快速排序通过在未排序的数组中选择一个称为*枢轴*的任意点，然后将数组分成一个左数组（其中包含所有小于枢轴的点）、一个右数组（其中包含所有等于或大于枢轴的点），枢轴位于两个数组之间。如果一个或两个数组的长度大于 1，那么我们将在一个或两个子数组上再次递归调用快速排序，此时枢轴点已经处于最终位置。

在纯 Python 中，可以使用函数式编程一行代码实现快速排序：

`qsort = lambda xs : [] if xs == [] else qsort(filter(lambda x: x < xs[-1] , xs[0:-1])) + [xs[-1]] + qsort(filter(lambda x: x >= xs[-1] , xs[0:-1]))`

我们可以看到并行性将在快速排序递归调用右数组和左数组时发挥作用——我们可以看到这将从一个线程在初始大数组上操作开始，但当数组变得非常小时，应该有许多线程在它们上工作。在这里，我们实际上将通过在每个*单个线程上*启动所有内核来实现这一点！

让我们开始吧，并从导入语句开始。（我们将确保从标准随机模块中导入`shuffle`函数，以备后面的示例。）

```py
from __future__ import division
import numpy as np
from pycuda.compiler import DynamicSourceModule
import pycuda.autoinit
from pycuda import gpuarray
from random import shuffle
```

现在我们将编写我们的快速排序内核。我们将为分区步骤编写一个`device`函数，它将接受一个整数指针、要分区的子数组的最低点和子数组的最高点。此函数还将使用此子数组的最高点作为枢轴。最终，在此函数完成后，它将返回枢轴的最终位置。

```py
DynamicQuicksortCode='''
__device__ int partition(int * a, int lo, int hi)
{
 int i = lo;
 int pivot = a[hi];
 int temp;

 for (int k=lo; k<hi; k++)
 {
  if (a[k] < pivot)
  {
   temp = a[k];
   a[k] = a[i];
   a[i] = temp;
   i++;
  }
 }

 a[hi] = a[i];
 a[i] = pivot;

 return i;
}
```

现在我们可以编写实现此分区函数的内核，将其转换为并行快速排序。我们将使用 CUDA-C 约定来处理流，这是我们到目前为止还没有见过的：要在 CUDA-C 中的流`s`中启动内核`k`，我们使用`k<<<grid, block, sharedMemBytesPerBlock, s>>>(...)`。通过在这里使用两个流，我们可以确保它们是并行启动的。（考虑到我们不会使用共享内存，我们将把第三个启动参数设置为“0”。）流对象的创建和销毁应该是不言自明的：

```py
__global__ void quicksort_ker(int *a, int lo, int hi)
{

 cudaStream_t s_left, s_right; 
 cudaStreamCreateWithFlags(&s_left, cudaStreamNonBlocking);
 cudaStreamCreateWithFlags(&s_right, cudaStreamNonBlocking);

 int mid = partition(a, lo, hi);

 if(mid - 1 - lo > 0)
   quicksort_ker<<< 1, 1, 0, s_left >>>(a, lo, mid - 1);
 if(hi - (mid + 1) > 0)
   quicksort_ker<<< 1, 1, 0, s_right >>>(a, mid + 1, hi);

 cudaStreamDestroy(s_left);
 cudaStreamDestroy(s_right);

}
'''
```

现在让我们随机洗牌一个包含 100 个整数的列表，并让我们的内核为我们进行排序。请注意我们如何在单个线程上启动内核：

```py
qsort_mod = DynamicSourceModule(DynamicQuicksortCode)

qsort_ker = qsort_mod.get_function('quicksort_ker')

if __name__ == '__main__':
    a = range(100)
    shuffle(a)

    a = np.int32(a)

    d_a = gpuarray.to_gpu(a)

    print 'Unsorted array: %s' % a

    qsort_ker(d_a, np.int32(0), np.int32(a.size - 1), grid=(1,1,1), block=(1,1,1))

    a_sorted = list(d_a.get())

    print 'Sorted array: %s' % a_sorted
```

此程序也可以在本书的 GitHub 存储库中的`dynamic_quicksort.py`文件中找到。

# 矢量化数据类型和内存访问

现在我们将看一下 CUDA 的矢量化数据类型。这些是标准数据类型的*矢量化*版本，例如 int 或 double，它们可以存储多个值。32 位类型的*矢量化*版本的大小最多为 4（例如`int2`，`int3`，`int4`和`float4`），而 64 位变量只能矢量化为原始大小的两倍（例如`double2`和`long2`）。对于大小为 4 的矢量化变量，我们使用 C 的“struct”表示法访问每个单独的元素，成员为`x`，`y`，`z`和`w`，而对于 3 个成员的变量，我们使用`x`，`y`和`z`，对于 2 个成员的变量，我们只使用`x`和`y`。

现在这些可能看起来毫无意义，但这些数据类型可以用来提高从全局内存加载数组的性能。现在，让我们进行一个小测试，看看我们如何可以从整数数组中加载一些 int4 变量，以及从双精度数组中加载 double2 变量——我们将不得不使用 CUDA 的`reinterpret_cast`运算符来实现这一点：

```py
from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

VecCode='''
__global__ void vec_ker(int *ints, double *doubles) { 

 int4 f1, f2;

 f1 = *reinterpret_cast<int4*>(ints);
 f2 = *reinterpret_cast<int4*>(&ints[4]);

 printf("First int4: %d, %d, %d, %d\\n", f1.x, f1.y, f1.z, f1.w);
 printf("Second int4: %d, %d, %d, %d\\n", f2.x, f2.y, f2.z, f2.w);

 double2 d1, d2;

 d1 = *reinterpret_cast<double2*>(doubles);
 d2 = *reinterpret_cast<double2*>(&doubles[2]);

 printf("First double2: %f, %f\\n", d1.x, d1.y);
 printf("Second double2: %f, %f\\n", d2.x, d2.y);

}'''
```

请注意，我们必须使用`dereference`运算符`*`来设置矢量化变量，以及我们必须通过引用（`&ints[4]`，`&doubles[2]`）跳转到下一个地址来加载第二个`int4`和`double2`，使用数组上的引用运算符`&`：

![](img/9b0e1417-e3c7-4896-9672-279ced733a2f.png)

这个例子也可以在本书的 GitHub 存储库中的`vectorized_memory.py`文件中找到。

# 线程安全的原子操作

我们现在将学习 CUDA 中的**原子操作**。原子操作是非常简单的、线程安全的操作，输出到单个全局数组元素或共享内存变量，否则可能会导致竞争条件。

让我们想一个例子。假设我们有一个内核，并且在某个时候我们设置了一个名为`x`的局部变量跨所有线程。然后我们想找到所有*x*中的最大值，然后将这个值设置为我们用`__shared__ int x_largest`声明的共享变量。我们可以通过在每个线程上调用`atomicMax(&x_largest, x)`来实现这一点。

让我们看一个原子操作的简单例子。我们将为两个实验编写一个小程序：

+   将变量设置为 0，然后为每个线程添加 1

+   找到所有线程中的最大线程 ID 值

让我们首先将`tid`整数设置为全局线程 ID，然后将全局`add_out`变量设置为 0。过去，我们会通过一个单独的线程使用`if`语句来改变变量，但现在我们可以使用`atomicExch(add_out, 0)`来跨所有线程进行操作。让我们导入并编写到这一点的内核：

```py
from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv

AtomicCode='''
__global__ void atomic_ker(int *add_out, int *max_out) 
{

 int tid = blockIdx.x*blockDim.x + threadIdx.x;

 atomicExch(add_out, 0);
```

应该注意的是，虽然原子操作确实是线程安全的，但它们绝不保证所有线程将同时访问它们，它们可能会在不同的时间由不同的线程执行。这可能会有问题，因为我们将在下一步中修改`add_out`。这可能会导致`add_out`在一些线程部分修改后被重置。让我们进行块同步以防止这种情况发生：

```py
 __syncthreads();
```

现在我们可以使用`atomicAdd`来为每个线程添加`1`到`add_out`，这将给我们总线程数：

```py
 atomicAdd(add_out, 1);
```

现在让我们通过使用`atomicMax`来检查`tid`的最大值是多少。然后我们可以关闭我们的 CUDA 内核：

```py
 atomicMax(max_out, tid);

}
'''
```

现在我们将添加测试代码；让我们尝试在 1 个包含 100 个线程的块上启动这个。我们这里只需要两个变量，所以我们将不得不分配一些大小为 1 的`gpuarray`对象。然后我们将打印输出：

```py
atomic_mod = SourceModule(AtomicCode)
atomic_ker = atomic_mod.get_function('atomic_ker')

add_out = gpuarray.empty((1,), dtype=np.int32)
max_out = gpuarray.empty((1,), dtype=np.int32)

atomic_ker(add_out, max_out, grid=(1,1,1), block=(100,1,1))

print 'Atomic operations test:'
print 'add_out: %s' % add_out.get()[0]
print 'max_out: %s' % max_out.get()[0]
```

现在我们准备运行这个：

![](img/d0799721-50a9-45e2-a9d2-6c3f8d097bcb.png)

这个例子也可以在本书的 GitHub 存储库中的`atomic.py`文件中找到。

# Warp shuffling

我们现在将看看所谓的**warp shuffling**。这是 CUDA 中的一个特性，允许存在于同一个 CUDA Warp 中的线程通过直接读写对方的寄存器（即它们的本地堆栈空间变量）进行通信，而无需使用*shared*变量或全局设备内存。Warp shuffling 实际上比其他两个选项快得多，更容易使用。这几乎听起来太好了，所以肯定有一个*陷阱*——确实，*陷阱*是这只在存在于同一个 CUDA Warp 上的线程之间起作用，这限制了对大小为 32 或更小的线程组的洗牌操作。另一个限制是我们只能使用 32 位或更小的数据类型。这意味着我们不能在 Warp 中洗牌 64 位*长长*整数或*双精度*浮点值。

只有 32 位（或更小）的数据类型可以与 CUDA Warp shuffling 一起使用！这意味着虽然我们可以使用整数、浮点数和字符，但不能使用双精度或*长长*整数！

在我们继续编码之前，让我们简要回顾一下 CUDA Warps。（在继续之前，您可能希望回顾第六章中名为“Warp Lockstep Property”的部分，“调试和分析您的 CUDA 代码”）。CUDA **Warp** 是 CUDA 中的最小执行单元，由 32 个线程或更少组成，运行在精确的 32 个 GPU 核心上。就像网格由块组成一样，块同样由一个或多个 Warps 组成，取决于块使用的线程数 - 如果一个块由 32 个线程组成，那么它将使用一个 Warp，如果它使用 96 个线程，它将由三个 Warps 组成。即使 Warp 的大小小于 32，它也被视为完整的 Warp：这意味着只有一个单个线程的块将使用 32 个核心。这也意味着 33 个线程的块将由两个 Warps 和 31 个核心组成。

要记住我们在第六章中看到的内容，“调试和分析您的 CUDA 代码”，Warp 具有所谓的**Lockstep Property**。这意味着 Warp 中的每个线程将完全并行地迭代每条指令，与 Warp 中的每个其他线程完全一致。也就是说，单个 Warp 中的每个线程将同时执行相同的指令，*忽略*任何不适用于特定线程的指令 - 这就是为什么要尽量避免单个 Warp 中线程之间的任何分歧。NVIDIA 将这种执行模型称为**Single Instruction Multiple Thread**，或**SIMT**。到目前为止，您应该明白为什么我们一直在文本中始终使用 32 个线程的块！

在我们开始之前，我们需要学习另一个术语 - Warp 中的**lane**是 Warp 内特定线程的唯一标识符，它将介于 0 和 31 之间。有时，这也被称为**Lane ID**。

让我们从一个简单的例子开始：我们将使用`__shfl_xor`命令在我们的 Warp 内的所有偶数和奇数编号的 Lanes（线程）之间交换特定变量的值。这实际上非常快速和容易做到，所以让我们编写我们的内核并查看一下：

```py
from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

ShflCode='''
__global__ void shfl_xor_ker(int *input, int * output) {

int temp = input[threadIdx.x];

temp = __shfl_xor (temp, 1, blockDim.x);

output[threadIdx.x] = temp;

}'''
```

除了`__shfl_xor`之外，这里的一切对我们来说都很熟悉。这是一个 CUDA 线程如何看待这个函数的方式：这个函数从当前线程接收`temp`的值作为输入。它对当前线程的二进制 Lane ID 执行一个`XOR`操作，这将是它的左邻居（如果该线程的 Lane 的最低有效位是二进制中的`1`）或右邻居（如果最低有效位是二进制中的“0”）。然后将当前线程的`temp`值发送到其邻居，同时检索邻居的 temp 值，这就是`__shfl_xor`。这将作为输出返回到`temp`中。然后我们在输出数组中设置值，这将交换我们的输入数组值。

现在让我们编写剩下的测试代码，然后检查输出：

```py
shfl_mod = SourceModule(ShflCode)
shfl_ker = shfl_mod.get_function('shfl_xor_ker')

dinput = gpuarray.to_gpu(np.int32(range(32)))
doutout = gpuarray.empty_like(dinput)

shfl_ker(dinput, doutout, grid=(1,1,1), block=(32,1,1))

print 'input array: %s' % dinput.get()
print 'array after __shfl_xor: %s' % doutout.get()
```

上述代码的输出如下：

![](img/c0f8d38e-aae3-4dee-8d83-ac869ba587bc.png)

在我们继续之前，让我们做一个更多的 Warp-shuffling 示例 - 我们将实现一个操作，对 Warp 中所有线程的单个本地变量进行求和。让我们回顾一下第四章中的 Naive Parallel Sum 算法，“内核、线程、块和网格”，这个算法非常快速，但做出了一个*天真*的假设，即我们有与数据片段一样多的处理器 - 这是生活中为数不多的几种情况之一，我们实际上会有这么多处理器，假设我们正在处理大小为 32 或更小的数组。我们将使用`__shfl_down`函数在单个 Warp 中实现这一点。`__shfl_down`接受第一个参数中的线程变量，并通过第二个参数中指示的步数*移动*变量在线程之间，而第三个参数将指示 Warp 的总大小。

让我们立即实现这个。再次，如果您不熟悉 Naive Parallel Sum 或不记得为什么这应该起作用，请查看第四章，*内核、线程、块和网格*。我们将使用`__shfl_down`实现一个直接求和，然后在包括整数 0 到 31 的数组上运行这个求和。然后我们将与 NumPy 自己的`sum`函数进行比较，以确保正确性：

```py
from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

ShflSumCode='''
__global__ void shfl_sum_ker(int *input, int *out) {

 int temp = input[threadIdx.x];

 for (int i=1; i < 32; i *= 2)
     temp += __shfl_down (temp, i, 32);

 if (threadIdx.x == 0)
     *out = temp;

}'''

shfl_mod = SourceModule(ShflSumCode)
shfl_sum_ker = shfl_mod.get_function('shfl_sum_ker')

array_in = gpuarray.to_gpu(np.int32(range(32)))
out = gpuarray.empty((1,), dtype=np.int32)

shfl_sum_ker(array_in, out, grid=(1,1,1), block=(32,1,1))

print 'Input array: %s' % array_in.get()
print 'Summed value: %s' % out.get()[0]
print 'Does this match with Python''s sum? : %s' % (out.get()[0] == sum(array_in.get()) )
```

这将给我们以下输出：

![](img/9ed86d23-a9fb-4770-add7-d80587b7aa01.png)

本节中的示例也可在本书 GitHub 存储库的`Chapter11`目录下的`shfl_sum.py`和`shfl_xor.py`文件中找到。

# 内联 PTX 汇编

我们现在将初步了解编写 PTX（Parallel Thread eXecution）汇编语言，这是一种伪汇编语言，适用于所有 Nvidia GPU，反过来由即时（JIT）编译器编译为特定 GPU 的实际机器代码。虽然这显然不是用于日常使用，但如果必要，它将让我们在比 C 甚至更低的级别上工作。一个特定的用例是，如果没有其他源代码可用，您可以轻松地反汇编 CUDA 二进制文件（主机端可执行文件/库或 CUDA .cubin 二进制文件）并检查其 PTX 代码。这可以在 Windows 和 Linux 中使用`cuobjdump.exe -ptx cuda_binary`命令来完成。

如前所述，我们将只涵盖从 CUDA-C 内部使用 PTX 的一些基本用法，它具有特定的语法和用法，类似于在 GCC 中使用内联主机端汇编语言。让我们开始编写我们的 GPU 代码：

```py
from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

PtxCode='''
```

我们将通过将代码编写到单独的设备函数中进行几个小实验。让我们从一个简单的函数开始，将一个输入变量设置为零。（我们可以在 CUDA 中使用 C++的传址运算符`&`，我们将在`device`函数中使用它。）

```py
__device__ void set_to_zero(int &x)
{
 asm("mov.s32 %0, 0;" : "=r"(x));
}
```

让我们在继续之前先分解一下。`asm`当然会告诉`nvcc`编译器我们将使用汇编，所以我们必须将代码放入引号中，以便正确处理。`mov`指令只是复制一个常量或其他值，并将其输入到**寄存器**中。（寄存器是 GPU 或 CPU 用于存储或操作值的最基本类型的芯片存储单元；这是 CUDA 中大多数*本地*变量的使用方式。）`mov.s32`中的`.s32`部分表示我们正在使用带符号的 32 位整数变量——PTX 汇编没有 C 中数据的*类型*，因此我们必须小心使用正确的特定操作。`%0`告诉`nvcc`使用与此处字符串的第`0`个参数对应的寄存器，并用逗号将此与`mov`的下一个*输入*分隔开，这是常量`0`。然后我们以分号结束汇编行，就像我们在 C 中一样，并用引号关闭这个汇编代码字符串。然后我们将使用冒号（而不是逗号！）来指示我们想要在我们的代码中使用的变量。`"=r"`表示两件事：`=`将告诉`nvcc`寄存器将被写入为输出，而`r`表示这应该被处理为 32 位整数数据类型。然后我们将要由汇编器处理的变量放在括号中，然后关闭`asm`，就像我们对任何 C 函数一样。

所有这些都是为了将一个变量的值设置为 0！现在，让我们编写一个小的设备函数，用于为我们添加两个浮点数：

```py
__device__ void add_floats(float &out, float in1, float in2)
{
 asm("add.f32 %0, %1, %2 ;" : "=f"(out) : "f"(in1) , "f"(in2));
}
```

让我们停下来注意一些事情。首先，当然，我们使用`add.f32`来表示我们要将两个 32 位浮点值相加。我们还使用`"=f"`表示我们将写入一个寄存器，`f`表示我们将只从中读取。还要注意我们如何使用冒号来分隔`write`寄存器和`only read`寄存器，以供`nvcc`使用。

在继续之前，让我们再看一个简单的例子，即类似于 C 中的`++`运算符的函数，它将整数增加`1`：

```py
__device__ void plusplus(int &x)
{
 asm("add.s32 %0, %0, 1;" : "+r"(x));
}
```

首先，请注意我们将“0th”参数用作输出和第一个输入。接下来，请注意我们使用的是`+r`而不是`=r`——`+`告诉`nvcc`这个寄存器在这个指令中将被读取和写入。

现在我们不会变得更复杂了，因为即使在汇编语言中编写一个简单的`if`语句也是相当复杂的。但是，让我们看一些更多的示例，这些示例在使用 CUDA Warps 时会很有用。让我们从一个小函数开始，这个函数将给出当前线程的 lane ID；这是非常有用的，实际上比使用 CUDA-C 更直接，因为 lane ID 实际上存储在一个称为`%laneid`的特殊寄存器中，我们无法在纯 C 中访问它。（请注意代码中我们使用了两个`%`符号，这将告诉`nvcc`直接在`%laneid`引用的汇编代码中使用`%`，而不是将其解释为`asm`命令的参数。）

```py
__device__ int laneid()
{
 int id; 
 asm("mov.u32 %0, %%laneid; " : "=r"(id)); 
 return id;
}
```

现在让我们编写另外两个函数，这些函数对处理 CUDA Warps 将会很有用。请记住，只能使用洗牌命令在 Warp 之间传递 32 位变量。这意味着要在 Warp 上传递 64 位变量，我们必须将其拆分为两个 32 位变量，分别将这两个变量洗牌到另一个线程，然后将这两个 32 位值重新组合成原始的 64 位变量。对于将 64 位双精度拆分为两个 32 位整数，我们可以使用`mov.b64`命令，注意我们必须使用`d`来表示 64 位浮点双精度：

请注意我们在以下代码中使用`volatile`，这将确保这些命令在编译后按原样执行。我们这样做是因为有时编译器会对内联汇编代码进行自己的优化，但对于这样特别敏感的操作，我们希望按照原样执行。

```py
__device__ void split64(double val, int & lo, int & hi)
{
 asm volatile("mov.b64 {%0, %1}, %2; ":"=r"(lo),"=r"(hi):"d"(val));
}

__device__ void combine64(double &val, int lo, int hi)
{
 asm volatile("mov.b64 %0, {%1, %2}; ":"=d"(val):"r"(lo),"r"(hi));
}
```

现在让我们编写一个简单的内核，用于测试我们编写的所有 PTX 汇编设备函数。然后我们将它启动在一个单个线程上，以便我们可以检查一切：

```py
__global__ void ptx_test_ker() { 

 int x=123;

 printf("x is %d \\n", x);

 set_to_zero(x);

 printf("x is now %d \\n", x);

 plusplus(x);

 printf("x is now %d \\n", x);

 float f;

 add_floats(f, 1.11, 2.22 );

 printf("f is now %f \\n", f);

 printf("lane ID: %d \\n", laneid() );

 double orig = 3.1415;

 int t1, t2;

 split64(orig, t1, t2);

 double recon;

 combine64(recon, t1, t2);

 printf("Do split64 / combine64 work? : %s \\n", (orig == recon) ? "true" : "false"); 

}'''

ptx_mod = SourceModule(PtxCode)
ptx_test_ker = ptx_mod.get_function('ptx_test_ker')
ptx_test_ker(grid=(1,1,1), block=(1,1,1))
```

现在我们将运行前面的代码：

![](img/bebb9f8f-a21c-4b06-b3e9-3686b8c96b41.png)

此示例也可在本书的 GitHub 存储库中的`Chapter11`目录下的`ptx_assembly.py`文件中找到。

# 性能优化的数组求和

对于本书的最后一个示例，我们将为给定的双精度数组制作一个标准的数组求和内核，但这一次我们将使用本章学到的所有技巧，使其尽可能快速。我们将检查我们求和内核的输出与 NumPy 的`sum`函数，然后我们将使用标准的 Python `timeit`函数运行一些测试，以比较我们的函数与 PyCUDA 自己的`gpuarray`对象的`sum`函数相比如何。

让我们开始导入所有必要的库，然后从一个`laneid`函数开始，类似于我们在上一节中使用的函数：

```py
from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv
from timeit import timeit

SumCode='''
__device__ void __inline__ laneid(int & id)
{
 asm("mov.u32 %0, %%laneid; " : "=r"(id)); 
}
```

让我们注意一些事情——注意我们在设备函数的声明中放了一个新的内联语句。这将有效地将我们的函数变成一个宏，当我们从内核中调用这个函数时，会减少一点时间。另外，注意我们通过引用设置了`id`变量，而不是返回一个值——在这种情况下，实际上可能有两个整数寄存器应该被使用，并且应该有一个额外的复制命令。这样可以确保这种情况不会发生。

让我们以类似的方式编写其他设备函数。我们需要再编写两个设备函数，以便我们可以将 64 位双精度数拆分和合并为两个 32 位变量：

```py
__device__ void __inline__ split64(double val, int & lo, int & hi)
{
 asm volatile("mov.b64 {%0, %1}, %2; ":"=r"(lo),"=r"(hi):"d"(val));
}

__device__ void __inline__ combine64(double &val, int lo, int hi)
{
 asm volatile("mov.b64 %0, {%1, %2}; ":"=d"(val):"r"(lo),"r"(hi));
}
```

让我们开始编写内核。我们将接收一个名为 input 的双精度数组，然后将整个总和输出到`out`，`out`应该初始化为`0`。我们将首先获取当前线程的 lane ID，并使用矢量化内存加载将两个值从全局内存加载到当前线程中：

```py
__global__ void sum_ker(double *input, double *out) 
{

 int id;
 laneid(id);

 double2 vals = *reinterpret_cast<double2*> ( &input[(blockDim.x*blockIdx.x + threadIdx.x) * 2] );
```

现在让我们将双精度变量`vals`中的这些值求和到一个新的双精度变量`sum_val`中，它将跟踪本线程的所有求和。我们将创建两个 32 位整数`s1`和`s2`，我们将使用它们来分割这个值，并使用 Warp Shuffling 与其他线程共享一个`temp`变量来重构值：

```py
 double sum_val = vals.x + vals.y;

 double temp;

 int s1, s2;
```

现在让我们再次在 Warp 上使用 Naive Parallel 求和，这将与在 Warp 上对 32 位整数求和相同，只是我们将在每次迭代中使用`sum_val`和`temp`上的`split64`和`combine64`PTX 函数：

```py
 for (int i=1; i < 32; i *= 2)
 {

     // use PTX assembly to split
     split64(sum_val, s1, s2);

     // shuffle to transfer data
     s1 = __shfl_down (s1, i, 32);
     s2 = __shfl_down (s2, i, 32);

     // PTX assembly to combine
     combine64(temp, s1, s2);
     sum_val += temp;
 }
```

现在我们完成了，让我们让每个 Warp 的`0th`线程将其结束值添加到`out`，使用线程安全的`atomicAdd`：

```py
 if (id == 0)
     atomicAdd(out, sum_val);

}'''
```

我们现在将编写我们的测试代码，使用`timeit`操作来测量我们的内核和 PyCUDA 对 10000*2*32 个双精度数组进行 20 次迭代的平均时间：

```py
sum_mod = SourceModule(SumCode)
sum_ker = sum_mod.get_function('sum_ker')

a = np.float64(np.random.randn(10000*2*32))
a_gpu = gpuarray.to_gpu(a)
out = gpuarray.zeros((1,), dtype=np.float64)

sum_ker(a_gpu, out, grid=(int(np.ceil(a.size/64)),1,1), block=(32,1,1))
drv.Context.synchronize()

print 'Does sum_ker produces the same value as NumPy\'s sum (according allclose)? : %s' % np.allclose(np.sum(a) , out.get()[0])

print 'Performing sum_ker / PyCUDA sum timing tests (20 each)...'

sum_ker_time = timeit('''from __main__ import sum_ker, a_gpu, out, np, drv \nsum_ker(a_gpu, out, grid=(int(np.ceil(a_gpu.size/64)),1,1), block=(32,1,1)) \ndrv.Context.synchronize()''', number=20)
pycuda_sum_time = timeit('''from __main__ import gpuarray, a_gpu, drv \ngpuarray.sum(a_gpu) \ndrv.Context.synchronize()''', number=20)

print 'sum_ker average time duration: %s, PyCUDA\'s gpuarray.sum average time duration: %s' % (sum_ker_time, pycuda_sum_time)
print '(Performance improvement of sum_ker over gpuarray.sum: %s )' % (pycuda_sum_time / sum_ker_time)
```

让我们从 IPython 中运行这个。确保你已经先运行了`gpuarray.sum`和`sum_ker`，以确保我们不会计算`nvcc`的编译时间：

![](img/b1e04079-149b-4da4-9524-6bc4ef455108.png)

因此，虽然求和通常相当无聊，但我们可以因为我们巧妙地利用硬件技巧来加速这样一个单调乏味的算法而感到兴奋。

此示例可在本书的 GitHub 存储库的`Chapter11`目录下的`performance_sum_ker.py`文件中找到。

# 总结

我们开始这一章时学习了动态并行性，这是一种允许我们直接在 GPU 上从其他内核启动和管理内核的范式。我们看到了我们如何可以使用这个来直接在 GPU 上实现快速排序算法。然后我们学习了 CUDA 中的矢量化数据类型，并看到了我们如何可以使用这些类型来加速从全局设备内存中读取数据。然后我们学习了 CUDA Warps，这是 GPU 上的小单位，每个 Warp 包含 32 个线程或更少，并且我们看到了单个 Warp 内的线程如何可以直接读取和写入彼此的寄存器，使用 Warp Shuffling。然后我们看了一下如何在 PTX 汇编中编写一些基本操作，包括确定 lane ID 和将 64 位变量分割为两个 32 位变量等导入操作。最后，我们通过编写一个新的性能优化求和内核来结束了这一章，该内核用于双精度数组，应用了本章学到的几乎所有技巧。我们看到，这实际上比双精度数组长度为 500,000 的标准 PyCUDA 求和更快。

我们已经完成了本书的所有技术章节！你应该为自己感到骄傲，因为现在你肯定是一个技艺高超的 GPU 程序员。现在我们将开始最后一章，在这一章中，我们将简要介绍一些不同的路径，可以帮助你应用和扩展你的 GPU 编程知识。

# 问题

1.  在原子操作示例中，尝试在启动内核之前将网格大小从 1 更改为 2，同时保持总块大小为 100。如果这给出了`add_out`的错误输出（除了 200 以外的任何值），那么为什么是错误的，考虑到`atomicExch`是线程安全的呢？

1.  在原子操作示例中，尝试移除`__syncthreads`，然后在原始参数的网格大小为 1，块大小为 100 的情况下运行内核。如果这给出了`add_out`的错误输出（除了 100 以外的任何值），那么为什么是错误的，考虑到`atomicExch`是线程安全的呢？

1.  为什么我们不必使用`__syncthreads`来同步大小为 32 或更小的块？

1.  我们发现`sum_ker`对于长度为 640,000（`10000*2*32`）的随机值数组比 PyCUDA 的求和操作快大约五倍。如果你尝试在这个数字的末尾加上一个零（也就是乘以 10），你会注意到性能下降到`sum_ker`只比 PyCUDA 的求和快大约 1.5 倍的程度。如果你在这个数字的末尾再加上一个零，你会注意到`sum_ker`只比 PyCUDA 的求和快 75%。你认为这是为什么？我们如何改进`sum_ker`以在更大的数组上更快？

1.  哪种算法执行了更多的加法操作（计算 C +运算符的调用和将 atomicSum 视为单个操作）：`sum_ker`还是 PyCUDA 的`sum`？
