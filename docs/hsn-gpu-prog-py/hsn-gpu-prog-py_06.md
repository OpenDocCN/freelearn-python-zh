# 第六章：调试和分析您的 CUDA 代码

在本章中，我们将最终学习如何使用多种不同的方法和工具调试和分析我们的 GPU 代码。虽然我们可以使用 Spyder 和 PyCharm 等 IDE 轻松调试纯 Python 代码，但我们无法使用这些工具来调试实际的 GPU 代码，记住 GPU 代码本身是用 CUDA-C 编写的，PyCUDA 提供了一个接口。调试 CUDA 内核的第一种最简单的方法是使用`printf`语句，我们实际上可以直接在 CUDA 内核中调用它来打印到标准输出。我们将看到如何在 CUDA 的上下文中使用`printf`以及如何有效地应用它进行调试。

接下来，我们将填补 CUDA-C 编程中的一些空白，以便我们可以直接在 NVIDIA Nsight IDE 中编写 CUDA 程序，这将允许我们为我们一直在编写的一些代码创建 CUDA-C 的测试用例。我们将看看如何使用`nvcc`命令行编译 CUDA-C 程序，以及如何在 Nsight IDE 中进行编译。然后，我们将看看如何在 Nsight 中进行调试，并使用 Nsight 了解 CUDA lockstep 属性。最后，我们将概述 NVIDIA 命令行和 Visual Profilers 以对我们的代码进行分析。

本章的学习成果包括以下内容：

+   有效地使用`printf`作为 CUDA 内核的调试工具

+   在 Python 之外编写完整的 CUDA-C 程序，特别是用于创建调试的测试用例

+   使用`nvcc`编译器在命令行上编译 CUDA-C 程序

+   使用 NVIDIA Nsight IDE 开发和调试 CUDA 程序

+   了解 CUDA warp lockstep 属性以及为什么我们应该避免单个 CUDA warp 内的分支分歧

+   学会有效使用 NVIDIA 命令行和 Visual Profilers 进行 GPU 代码的调试

# 技术要求

本章需要一台安装了现代 NVIDIA GPU（2016 年以后）的 Linux 或 Windows 10 PC，并安装了所有必要的 GPU 驱动程序和 CUDA Toolkit（9.0 及以上）。还需要一个合适的 Python 2.7 安装（如 Anaconda Python 2.7），并安装了 PyCUDA 模块。

本章的代码也可以在 GitHub 上找到：[`github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA`](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)。

有关先决条件的更多信息，请查看本书的*前言*，有关软件和硬件要求，请查看[`github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA`](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)中的 README。

# 在 CUDA 内核中使用 printf

也许会让人惊讶，但我们实际上可以直接从 CUDA 内核中将文本打印到标准输出；不仅如此，每个单独的线程都可以打印自己的输出。当我们调试内核时，这将特别方便，因为我们可能需要监视特定变量或代码中特定点的计算值，这也将使我们摆脱使用调试器逐步进行调试的束缚。从 CUDA 内核中打印输出的方法是使用 C/C++编程中最基本的函数，大多数人在编写他们的第一个 C 程序`Hello world`时会学到的函数：`printf`。当然，`printf`是将字符串打印到标准输出的标准函数，实际上在 C 编程语言中相当于 Python 的`print`函数。

现在让我们简要回顾一下如何在 CUDA 中使用`printf`。首先要记住的是，`printf`总是以字符串作为其第一个参数；因此，在 C 中打印"Hello world!"是用`printf("Hello world!\n");`来完成的。（当然，`\n`表示"新行"或"返回"，它将输出在终端上移到下一行。）`printf`还可以在我们想要直接在 C 中打印任何常量或变量的情况下，采用可变数量的参数：如果我们想要将`123`整数打印到输出，我们可以使用`printf("%d", 123);`（其中`%d`表示字符串后面跟着一个整数。）

类似地，我们使用`%f`，`%e`或`%g`来打印浮点值（其中`%f`是十进制表示法，`%e`是科学表示法，`%g`是最短的表示法，无论是十进制还是科学表示法）。我们甚至可以连续打印几个值，记得按正确的顺序放置这些指示符：`printf("%d is a prime number, %f is close to pi, and %d is even.\n", 17, 3.14, 4);`将在终端上打印"17 is a prime number, 3.14 is close to pi, and 4 is even."。

现在，在这本书的近一半时，我们终于要开始创建我们的第一个并行`Hello world`程序了！我们首先导入适当的模块到 Python 中，然后编写我们的内核。我们将首先打印每个单独线程的线程和网格标识（我们只会在一维块和网格中启动这个，所以我们只需要`x`值）：

```py
ker = SourceModule('''
__global__ void hello_world_ker()
{
    printf("Hello world from thread %d, in block %d!\\n", threadIdx.x, blockIdx.x);
```

让我们停下来注意一下，我们写的是`\\n`而不是`\n`。这是因为 Python 中的三引号本身会将`\n`解释为"新行"，所以我们必须使用双反斜杠来表示我们是字面意思，以便将`\n`直接传递给 CUDA 编译器。

我们现在将打印有关块和网格维度的一些信息，但我们希望确保它在每个线程完成其初始`printf`命令后打印。我们可以通过放入`__syncthreads();`来确保每个单独的线程在执行第一个`printf`函数后同步。

现在，我们只想将块和网格的维度打印到终端一次；如果我们在这里放置`printf`语句，每个线程都会打印相同的信息。我们可以通过只有一个指定的线程打印输出来实现这一点；让我们选择第 0 个块的第 0 个线程，这是唯一保证存在的线程，无论我们选择的块和网格的维度如何。我们可以通过 C 的`if`语句来实现这一点：

```py
 if(threadIdx.x == 0 && blockIdx.x == 0)
 {
```

我们现在将打印出我们的块和网格的维度，并关闭`if`语句，这将是我们的 CUDA 内核的结束：

```py
 printf("-------------------------------------\\n");
 printf("This kernel was launched over a grid consisting of %d blocks,\\n", gridDim.x);
 printf("where each block has %d threads.\\n", blockDim.x);
 }
}
''')
```

我们现在将提取内核，然后在由两个块组成的网格上启动它，每个块有五个线程：

```py
hello_ker = ker.get_function("hello_world_ker")
hello_ker( block=(5,1,1), grid=(2,1,1) )
```

让我们现在运行这个程序（该程序也可以在存储库中的`hello-world_gpu.py`的`6`下找到）：

![](img/2c945416-8c8a-4eae-8d57-ef1479dd2798.png)

# 使用 printf 进行调试

让我们通过一个例子来看看如何使用`printf`调试 CUDA 内核，然后再继续。这种方法并没有确切的科学依据，但通过经验可以学会这种技能。我们将从一个 CUDA 内核开始，用于矩阵乘法，但其中有几个错误。（鼓励读者随着我们的步骤阅读代码，该代码可在存储库中的`6`目录中的`broken_matrix_ker.py`文件中找到。）

在继续之前，让我们简要回顾一下矩阵乘法。假设我们有两个矩阵*A*和*B*，我们将它们相乘得到另一个相同大小的矩阵*C*，如下所示：![](img/60101910-7794-4484-b69f-7ddaad3994db.png)。我们通过迭代所有元组![](img/f550491f-6b50-49dd-be38-817b363525b9.png)来做到这一点，并将![](img/54c310e8-02c5-4bbc-b645-9f33465d4e77.png)的值设置为*A*的第*i*行和*B*的第*j*列的点积：![](img/615e6730-f5b3-48d2-9ac2-fb9a7dc535de.png)。

换句话说，我们将输出矩阵*C*中的每个*i, j*元素设置如下：![](img/8870cb22-35f8-4612-aa78-b1cc92fd38f9.png)

假设我们已经编写了一个内核，用于执行矩阵乘法，它接受表示输入矩阵的两个数组，一个额外的预先分配的浮点数组，输出将写入其中，并且一个表示每个矩阵的高度和宽度的整数（我们将假设所有矩阵都是相同大小和正方形的）。这些矩阵都将以一维的`float *`数组以行优先的一维布局表示。此外，这将被实现为每个 CUDA 线程处理输出矩阵中的单个行/列元组。

我们进行了一个小的测试案例，并将其与 CUDA 中矩阵乘法的输出进行了检查，对于两个 4 x 4 矩阵，它作为断言检查失败，如下所示：

```py
test_a = np.float32( [xrange(1,5)] * 4 )
test_b = np.float32([xrange(14,10, -1)]*4 )
output_mat = np.matmul(test_a, test_b)

test_a_gpu = gpuarray.to_gpu(test_a)
test_b_gpu = gpuarray.to_gpu(test_b)
output_mat_gpu = gpuarray.empty_like(test_a_gpu)

matrix_ker(test_a_gpu, test_b_gpu, output_mat_gpu, np.int32(4), block=(2,2,1), grid=(2,2,1))

assert( np.allclose(output_mat_gpu.get(), output_mat) )
```

我们现在将运行这个程序，并且不出所料地得到以下输出：

![](img/7fb29cbf-af3a-4fd1-a75a-cd05d3f56a44.png)

现在让我们来看一下 CUDA C 代码，其中包括一个内核和一个设备函数：

```py
ker = SourceModule('''
// row-column dot-product for matrix multiplication
__device__ float rowcol_dot(float *matrix_a, float *matrix_b, int row, int col, int N)
{
 float val = 0;

 for (int k=0; k < N; k++)
 {
     val += matrix_a[ row + k*N ] * matrix_b[ col*N + k];
 }
 return(val);
}

// matrix multiplication kernel that is parallelized over row/column tuples.

__global__ void matrix_mult_ker(float * matrix_a, float * matrix_b, float * output_matrix, int N)
{
 int row = blockIdx.x + threadIdx.x;
 int col = blockIdx.y + threadIdx.y;

 output_matrix[col + row*N] = rowcol_dot(matrix_a, matrix_b, col, row, N);
}
''')
```

我们的目标是在我们的 CUDA 代码中聪明地放置`printf`调用，以便我们可以监视内核和设备函数中的许多适当的值和变量；我们还应该确保在每个`printf`调用中打印出线程和块号。

让我们从内核的入口点开始。我们看到两个变量`row`和`col`，所以我们应该立即检查这些。让我们在设置它们之后立即放上一行代码（因为这是在两个维度上并行化的，我们应该打印`threadIdx`和`blockIdx`的*x*和*y*值）：

```py
printf("threadIdx.x,y: %d,%d blockIdx.x,y: %d,%d -- row is %d, col is %d.\\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, row, col);
```

再次运行代码，我们得到了这个输出：

![](img/579c6bb8-3bee-4788-b606-aff8d5d15812.png)

有两件事情立即引人注目：行和列元组有重复的值（每个单独的元组应该只表示一次），而且行和列的值从未超过两，而它们都应该达到三（因为这个单元测试使用 4 x 4 的矩阵）。这应该告诉我们，我们正在错误地计算行和列的值；确实，我们忘记了将`blockIdx`的值乘以`blockDim`的值来找到目标行/列值。我们按照以下方式修复这个问题：

```py
int row = blockIdx.x*blockDim.x + threadIdx.x;
int col = blockIdx.y*blockDim.y + threadIdx.y;
```

然而，如果我们再次运行程序，我们仍然会得到一个断言错误。让我们保留原始的`printf`调用，这样我们就可以在继续进行的过程中监视这些值。我们看到在内核中有一个对设备函数`rowcol_dot`的调用，所以我们决定去看一下。让我们首先确保变量被正确传递到设备函数中，通过在开始处放置这个`printf`调用：

```py
printf("threadIdx.x,y: %d,%d blockIdx.x,y: %d,%d -- row is %d, col is %d, N is %d.\\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, row, col, N);
```

当我们运行程序时，会有更多的行输出，但我们会看到一行说—`threadIdx.x,y: 0,0 blockIdx.x,y: 1,0 -- row is 2, col is 0.`，还有另一行说—`threadIdx.x,y: 0,0 blockIdx.x,y: 1,0 -- row is 0, col is 2, N is 4`。通过`threadIdx`和`blockIdx`的值，我们看到这是同一个块中的同一个线程，但`row`和`col`的值是颠倒的。实际上，当我们查看`rowcol_dot`设备函数的调用时，我们看到`row`和`col`确实是与设备函数声明中的相反。我们修复了这个问题，但当我们再次运行程序时，又出现了另一个断言错误。

让我们在设备函数中的`for`循环内放置另一个`printf`调用；这当然是*点积*，用于在矩阵`A`的行与矩阵`B`的列之间执行点积。我们将检查我们正在相乘的矩阵的值，以及`k`；我们还将只查看第一个线程的值，否则我们将得到一个不连贯的输出混乱。

```py
if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
            printf("Dot-product loop: k value is %d, matrix_a value is %f, matrix_b is %f.\\n", k, matrix_a[ row + k*N ], matrix_b[ col*N + k]);

```

在继续之前，让我们看一下为我们的单元测试设置的`A`和`B`矩阵的值：

![](img/b2580d2a-f34e-4d83-abbd-43ebc163aba7.png)

我们看到，当我们在列之间切换时，两个矩阵都会变化，但在行之间变化时是恒定的。因此，根据矩阵乘法的性质，矩阵`A`的值应该在我们的`for`循环中随着`k`的变化而变化，而`B`的值应该保持恒定。让我们再次运行程序并检查相关的输出：

![](img/47f4367a-3282-4718-9ec9-e818e0a16ffb.png)

因此，看起来我们没有以正确的方式访问矩阵的元素；记住这些矩阵是以行方式存储的，我们修改索引，以便以正确的方式访问它们的值：

```py
val += matrix_a[ row*N + k ] * matrix_b[ col + k*N];
```

再次运行程序将不会产生断言错误。恭喜，您刚刚使用唯一的`printf`调试了一个 CUDA 内核！

# 用 CUDA-C 填补空白

我们现在将介绍如何编写一个完整的 CUDA-C 程序的基础知识。我们将从小处开始，将我们刚刚在上一节中调试的小矩阵乘法测试程序的*修复*版本翻译成纯 CUDA-C 程序，然后使用 NVIDIA 的`nvcc`编译器从命令行编译成本机 Windows 或 Linux 可执行文件（我们将在下一节中看到如何使用 NVIDIA 的 Nsight IDE，所以现在我们只使用文本编辑器和命令行）。同样，鼓励读者在我们进行翻译时查看我们正在翻译的 Python 代码，该代码在存储库中作为`matrix_ker.py`文件可用。

现在，让我们打开我们最喜欢的文本编辑器，创建一个名为`matrix_ker.cu`的新文件。扩展名将表明这是一个 CUDA-C 程序，可以使用`nvcc`编译器进行编译。

CUDA-C 程序和库源代码的文件名总是使用`.cu`文件扩展名。

让我们从头开始——正如 Python 在程序开头使用`import`关键字导入库一样，我们回忆 C 语言使用`#include`。在继续之前，我们需要包含一些导入库。

让我们从这些开始：

```py
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
```

让我们简要地思考一下我们需要这些的原因：`cuda_runtime.h`是一个头文件，其中包含了我们程序所需的所有特定 CUDA 数据类型、函数和结构的声明。我们需要为我们编写的任何纯 CUDA-C 程序包含这个头文件。`stdio.h`当然为我们提供了主机的所有标准 I/O 函数，如`printf`，我们需要`stdlib.h`来使用主机上的`malloc`和`free`动态内存分配函数。

请记住，始终在每个纯 CUDA-C 程序的开头放置`#include <cuda_runtime.h>`！

现在，在我们继续之前，我们记得我们最终将不得不检查我们的内核输出与正确的已知输出，就像我们在 NumPy 的`allclose`函数中所做的那样。不幸的是，在 C 中，我们没有像 Python 中的 NumPy 那样的标准或易于使用的数值数学库。往往，如果是一些简单的东西，编写自己的等效函数会更容易，就像在这种情况下一样。这意味着我们现在将明确地制作我们自己的等效于 NumPy 的`allclose`。我们将这样做：我们将使用 C 中的`#define`宏来设置一个名为`_EPSILON`的值，它将作为一个常数来指示输出和期望输出之间的最小值，以便被认为是相同的，我们还将设置一个名为`_ABS`的宏，它将告诉我们两个数字之间的绝对差异。我们这样做如下：

```py
#define _EPSILON 0.001
#define _ABS(x) ( x > 0.0f ? x : -x )
```

现在我们可以创建我们自己的`allclose`版本。这将接受两个浮点指针和一个整数值`len`，我们循环遍历两个数组并检查它们：如果任何点的差异超过`_EPSILON`，我们返回-1，否则我们返回 0，表示这两个数组确实匹配。

我们注意到一件事：由于我们使用 CUDA-C，我们在函数定义之前加上`__host__`，以表明这个函数是打算在 CPU 上运行而不是在 GPU 上运行的：

```py
__host__ int allclose(float *A, float *B, int len)
{

  int returnval = 0;

  for (int i = 0; i < len; i++)
  {
    if ( _ABS(A[i] - B[i]) > _EPSILON )
    {
      returnval = -1;
      break;
    }
  }

  return(returnval);
}
```

现在我们可以将设备和内核函数剪切并粘贴到这里，就像它们在我们的 Python 版本中出现的那样：

```py

__device__ float rowcol_dot(float *matrix_a, float *matrix_b, int row, int col, int N)
{
  float val = 0;

  for (int k=0; k < N; k++)
  {
        val += matrix_a[ row*N + k ] * matrix_b[ col + k*N];
  }

  return(val);
}

__global__ void matrix_mult_ker(float * matrix_a, float * matrix_b, float * output_matrix, int N)
{

    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

  output_matrix[col + row*N] = rowcol_dot(matrix_a, matrix_b, row, col, N);
}
```

再次，与`__host__`相比，注意 CUDA 设备函数前面有`__device__`，而 CUDA 内核前面有`__global__`。

现在，就像在任何 C 程序中一样，我们需要编写`main`函数，它将在主机上运行，我们将在其中设置我们的测试案例，并从中显式地启动我们的 CUDA 内核到 GPU 上。再次，与普通的 C 相比，我们将明确指定这也要在 CPU 上运行，并使用`__host__`：

```py
__host__ int main()
{
```

我们将要做的第一件事是选择和初始化我们的 GPU。我们可以使用`cudaSetDevice`来这样做：

```py
cudaSetDevice(0);
```

`cudaSetDevice(0)`将选择默认的 GPU。如果您的系统中安装了多个 GPU，您可以选择并使用它们，而不是使用`cudaSetDevice(1)`，`cudaSetDevice(2)`等。

现在我们将设置`N`，就像在 Python 中一样，以指示矩阵的高度/宽度。由于我们的测试案例将只包括 4 x 4 矩阵，我们将其设置为`4`。由于我们将使用动态分配的数组和指针，我们还必须设置一个值，以指示我们的测试矩阵将需要的字节数。矩阵将由*N* x *N*浮点数组成，我们可以使用 C 中的`sizeof`关键字确定浮点数所需的字节数：

```py
int N = 4;
int num_bytes = sizeof(float)*N*N;
```

现在我们设置我们的测试矩阵如下；这些将与我们在 Python 测试程序中看到的`test_a`和`test_b`矩阵完全对应（请注意我们如何使用`h_`前缀来表示这些数组存储在主机上，而不是设备上）：

```py

 float h_A[] = { 1.0, 2.0, 3.0, 4.0, \
                 1.0, 2.0, 3.0, 4.0, \
                 1.0, 2.0, 3.0, 4.0, \
                 1.0, 2.0, 3.0, 4.0 };

 float h_B[] = { 14.0, 13.0, 12.0, 11.0, \
                 14.0, 13.0, 12.0, 11.0, \
                 14.0, 13.0, 12.0, 11.0, \
                 14.0, 13.0, 12.0, 11.0 };
```

现在我们设置另一个数组，它将指示先前测试矩阵的矩阵乘法的预期输出。我们将不得不明确计算这一点，并将这些值放入我们的 C 代码中。最终，我们将在程序结束时将其与 GPU 输出进行比较，但让我们先设置它并将其放在一边：

```py
float h_AxB[] = { 140.0, 130.0, 120.0, 110.0, \
                 140.0, 130.0, 120.0, 110.0, \
                 140.0, 130.0, 120.0, 110.0, \
                 140.0, 130.0, 120.0, 110.0 };
```

现在我们声明一些指针，这些指针将存在于 GPU 上的数组，并且我们将复制`h_A`和`h_B`的值并指向 GPU 的输出。请注意，我们只是使用标准的浮点指针。还要注意前缀`d_` - 这是另一个标准的 CUDA-C 约定，表示这些将存在于设备上：

```py
float * d_A;
float * d_B;
float * d_output;
```

现在，我们将使用`cudaMalloc`在设备上为`d_A`和`d_B`分配一些内存，这几乎与 C 中的`malloc`相同；这就是 PyCUDA `gpuarray`函数（如`empty`或`to_gpu`）在本书中无形地调用我们分配 GPU 上的内存数组的方式：

```py
cudaMalloc((float **) &d_A, num_bytes);
cudaMalloc((float **) &d_B, num_bytes);
```

让我们思考一下这是如何工作的：在 C 函数中，我们可以通过在变量前加上一个取地址运算符（`&`）来获取变量的地址；如果有一个整数`x`，我们可以用`&x`来获取它的地址。`&x`将是一个指向整数的指针，因此它的类型将是`int *`。我们可以使用这个来将参数的值设置到 C 函数中，而不仅仅使用纯返回值。

由于`cudaMalloc`通过参数设置指针而不是使用返回值（与常规的`malloc`相反），我们必须使用取地址运算符，它将是一个指向指针的指针，因为它是一个指向浮点指针的指针，所以我们必须使用括号显式地转换这个值，因为`cudaMalloc`可以分配任何类型的数组。最后，在第二个参数中，我们必须指示在 GPU 上分配多少字节；我们之前已经设置了`num_bytes`，以便它是我们需要保存由浮点数组成的 4 x 4 矩阵所需的字节数，所以我们将其插入并继续。

现在我们可以使用两次`cudaMemcpy`函数将`h_A`和`h_B`的值分别复制到`d_A`和`d_B`中：

```py
cudaMemcpy(d_A, h_A, num_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, num_bytes, cudaMemcpyHostToDevice);
```

`cudaMemcpy`总是以目标指针作为第一个参数，源指针作为第二个参数，要复制的字节数作为第三个参数，并且还有一个最后的参数。最后一个参数将指示我们是使用`cudaMemcpyHostToDevice`从主机到 GPU 进行复制，使用`cudaMemcpyDeviceToHost`从 GPU 到主机进行复制，还是在 GPU 上的两个数组之间进行复制`cudaMemcpyDeviceToDevice`。

我们现在将分配一个数组来保存我们在 GPU 上进行矩阵乘法的输出，使用`cudaMalloc`的另一个调用：

```py
cudaMalloc((float **) &d_output, num_bytes);
```

最后，当我们想要检查内核的输出时，我们将在主机上设置一些存储 GPU 输出的内存。让我们设置一个常规的 C 浮点指针，并使用`malloc`分配内存，就像我们通常做的那样：

```py
float * h_output;
h_output = (float *) malloc(num_bytes);
```

现在，我们几乎准备好启动我们的内核。CUDA 使用一个名为`dim3`的数据结构来指示内核启动的块和网格大小；我们将设置这些，因为我们想要一个 2 x 2 维度的网格和也是 2 x 2 维度的块：

```py
dim3 block(2,2,1);
dim3 grid(2,2,1);
```

现在我们准备启动我们的内核；我们使用三角形括号来指示 CUDA-C 编译器内核应该启动的块和网格大小：

```py
matrix_mult_ker <<< grid, block >>> (d_A, d_B, d_output, N);
```

现在，当然，在我们可以将内核的输出复制回主机之前，我们必须确保内核已经执行完毕。我们通过调用`cudaDeviceSynchronize`来做到这一点，这将阻止主机向 GPU 发出更多命令，直到内核执行完毕：

```py
cudaDeviceSynchronize();
```

现在我们可以将内核的输出复制到我们在主机上分配的数组中：

```py
cudaMemcpy(h_output, d_output, num_bytes, cudaMemcpyDeviceToHost);
```

再次，我们同步：

```py
cudaDeviceSynchronize();
```

在检查输出之前，我们意识到我们不再需要在 GPU 上分配的任何数组。我们通过在每个数组上调用`cudaFree`来释放这些内存：

```py
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_output);
```

我们已经完成了 GPU 的工作，所以我们调用`cudaDeviceReset`：

```py
cudaDeviceReset();
```

现在，我们最终检查我们在主机上复制的输出，使用我们在本章开头编写的`allclose`函数。如果实际输出与预期输出不匹配，我们打印一个错误并返回`-1`，否则，我们打印匹配并返回 0。然后我们在程序的`main`函数上放一个闭括号：

```py
if (allclose(h_AxB, h_output, N*N) < 0)
 {
     printf("Error! Output of kernel does not match expected output.\n");
     free(h_output);
     return(-1);
 }
 else
 {
     printf("Success! Output of kernel matches expected output.\n");
     free(h_output);
     return(0);
 }
}
```

请注意，由于我们已经为`h_output`分配了内存，在这两种情况下，我们最后调用了标准的 C free 函数。

现在保存我们的文件，并从命令行编译成 Windows 或 Linux 可执行文件，使用`nvcc matrix_ker.cu -o matrix_ker`。这应该输出一个二进制可执行文件，`matrix_ker.exe`（在 Windows 中）或`matrix_ker`（在 Linux 中）。让我们尝试编译和运行它：

![](img/ada8e5c8-9443-40ce-81d9-03100f570855.png)

恭喜，您刚刚创建了您的第一个纯 CUDA-C 程序！（此示例在存储库中作为`matrix_ker.cu`在`7`下可用。）

# 使用 Nsight IDE 进行 CUDA-C 开发和调试

现在让我们学习如何使用 Nsight IDE 开发 CUDA-C 程序。我们将看到如何导入我们刚刚编写的程序，并在 Nsight 内部进行编译和调试。请注意，由于在 Windows 下它实际上是 Visual Studio IDE 的插件，在 Linux 下是 Eclipse IDE 的插件，因此 Windows 和 Linux 版本的 Nsight 之间存在差异。我们将在接下来的两个子部分中涵盖两者；如果不适用于您的操作系统，请随意跳过。

# 在 Windows 中使用 Visual Studio 的 Nsight

打开 Visual Studio，点击文件，然后选择新建|项目....会弹出一个窗口，您可以在其中设置项目类型：选择 NVIDIA 下拉项，然后选择 CUDA 9.2：

![](img/e5b3c450-388f-44ac-9d77-8f7cf574bf69.png)

给项目取一个合适的名称，然后点击确定。在解决方案资源管理器窗口中应该会出现一个项目，其中包含一个简单的预制 CUDA 测试程序，由一个源文件`kernel.cu`组成，其中包含一个简单的并行加法内核和测试代码。如果您想查看这是否编译和运行，请点击顶部标有本地 Windows 调试器的绿色向右箭头。一个终端应该弹出，显示内核的一些文本输出，然后立即关闭。

如果您在从 Visual Studio 运行后关闭基于 Windows 终端的应用程序时遇到问题，请尝试在主函数的末尾添加`getchar();`，这将使终端保持打开状态，直到您按下一个键。（或者，您也可以在程序的末尾使用调试器断点。）

现在，让我们添加刚刚编写的 CUDA-C 程序。在解决方案资源管理器窗口中，右键单击`kernel.cu`，然后单击`kernel.cu`上的删除。现在，右键单击项目名称，选择添加，然后选择现有项目。现在我们可以选择一个现有文件，找到`matrix_ker.cu`的路径，并将其添加到项目中。点击 IDE 顶部标有本地 Windows 调试器的绿色箭头，程序应该会在 Windows 终端中再次编译和运行。这就是我们可以在 Visual Studio 中设置和编译完整的 CUDA 程序的全部步骤。

现在让我们看看如何调试我们的 CUDA 内核。让我们首先在代码的入口点`matrix_mult_ker`处添加一个断点，我们在那里设置了`row`和`col`的值。我们可以通过在窗口的行号左侧的灰色列上单击来添加此断点；每个我们添加的断点都应该在那里显示一个红点。（您可以忽略 Visual Studio 编辑器可能在您的代码下方放置的任何红色波浪线；这是因为 CUDA 不是 Visual Studio 的*本地*语言）：

![](img/b4408001-ab3d-42fb-9741-1e1a3c896491.png)

现在我们可以开始调试。从顶部菜单中选择 Nsight 下拉菜单，然后选择开始 CUDA 调试。这里可能有两个选项，开始 CUDA 调试（Next-Gen）和开始 CUDA 调试（Legacy）。无论选择哪一个都可以，但是根据您的 GPU，可能会在 Next-Gen 上遇到问题；在这种情况下，请选择 Legacy。

您的程序应该启动，并且调试器应该在我们刚刚设置的内核中的断点处停止。让我们按*F10*跳过这一行，现在看看`row`变量是否被正确设置。让我们在变量资源管理器中的本地窗口中查看：

![](img/4bbfaa46-46cb-44cc-ac0b-055dc9ce89ca.png)

通过检查`threadIdx`和`blockIdx`的值，我们可以看到我们当前位于网格中的第一个块中的第一个线程；`row`设置为`0`，这确实对应于正确的值。现在，让我们检查一些不同线程的`row`值。为了做到这一点，我们必须在 IDE 中切换**线程焦点**；我们可以通过单击上面的 Nsight 下拉菜单，然后选择 Windows|CUDA Debug Focus...来实现这一点。应该会出现一个新菜单，允许您选择一个新的线程和块。在菜单中将线程从 0, 0, 0 更改为 1, 0, 0，然后单击确定：

![](img/c5f2b196-8ce2-45b6-807a-11f8d86fe86b.png)

当您再次检查变量时，您应该看到为此线程设置了正确的`row`值：

![](img/f2f27821-5df5-4ea7-a847-385967ba54a0.png)

简而言之，这就是您在 Visual Studio 中使用 Nsight 进行调试的方法。我们现在已经掌握了如何在 Windows 中使用 Nsight/Visual Studio 调试 CUDA 程序的基础知识，我们可以像调试常规 Windows 程序一样使用所有常规约定（设置断点，启动调试器，继续/恢复，跳过，步入和步出）。主要的区别在于您必须知道如何在 CUDA 线程和块之间切换以检查变量，否则它基本上是一样的。

# 在 Linux 中使用 Nsight 与 Eclipse

现在我们将看到如何在 Linux 中使用 Nsight。您可以从桌面上选择它打开 Nsight，也可以使用`nsight`命令从命令行运行它。Nsight IDE 将打开。从 IDE 顶部，单击文件，然后从下拉菜单中选择新建...，然后选择新建 CUDA C/C++项目。将出现一个新窗口，在这里选择 CUDA Runtime 项目。给项目取一个合适的名字，然后点击下一步。您将被提示提供进一步的设置选项，但默认设置对我们的目的来说现在可以工作得很好。（请确保注意这里第三和第四屏幕中源文件和项目路径的位置。）您将进入最终屏幕，在这里您可以按完成来创建项目：

![](img/e2f72961-a2fe-4c06-a4c2-2ab6653c140b.png)

最后，您将在项目视图中看到您的新项目和一些占位代码；从 CUDA 9.2 开始，这将包括一个倒数内核示例。

现在我们可以导入我们的代码。您可以使用 Nsight 中的编辑器删除默认源文件中的所有代码并剪切粘贴，或者您可以手动从项目的源目录中删除文件，手动将`matrix_ker.cu`文件复制到源目录中，然后选择刷新 Nsight 中的源目录视图，然后按*F5*。现在可以使用*Ctrl* + *B*构建项目，并使用*F11*运行它。我们程序的输出应该出现在 IDE 的 Console 子窗口中，如下所示：

![](img/1e287f83-ed1a-416e-aca7-511b375c1568.png)

现在，我们可以在 CUDA 代码中设置断点；让我们在内核的入口点设置一个断点，那里设置了`row`值。我们将光标放在 Eclipse 编辑器中的该行上，然后按*Ctrl* + *Shift* + *B*进行设置。

现在，我们可以通过按*F11*（或单击 bug 图标）开始调试。程序应该在`main`函数的开头暂停，所以按*F8*继续到第一个断点。您应该在 IDE 中看到我们的 CUDA 内核中的第一行被箭头指向。让我们通过按*F6*跳过当前行，确保`row`已经设置。

现在，我们可以轻松地在 CUDA 网格中切换不同的线程和块，以检查它们当前持有的值：从 IDE 顶部，单击窗口下拉菜单，然后单击显示视图，然后选择 CUDA。应该会打开一个显示当前运行内核的窗口，从这里您可以看到此内核正在运行的所有块的列表。

点击第一个，从这里你将能够看到块内运行的所有单个线程：

![](img/429490aa-3a80-48af-8833-9cc6a4988a7f.png)

现在，我们可以通过单击“变量”选项卡来查看与第一个块中的第一个线程对应的变量，这里，row 应该是 0，正如我们所期望的：

![](img/2d42ba20-c3bb-4f76-acc9-054fa1cc3960.png)

现在，我们可以通过再次转到 CUDA 选项卡，选择适当的线程并切换回来，来检查不同线程的值。让我们留在同一个块中，但这次选择线程（1,0,0），再次检查 row 的值：

![](img/ca99eb7a-04e5-40a0-982e-8a605b3cc697.png)

我们看到 row 的值现在是 1，正如我们所期望的。

现在，我们已经掌握了如何从 Nisight/Eclipse 在 Linux 中调试 CUDA 程序的基础知识，我们可以像调试其他 IDE 中的常规 Linux 程序一样使用所有常规约定（设置断点、启动调试器、继续/恢复、步进、步入和步出）。主要的区别在于我们必须知道如何在 CUDA 线程和块之间切换以检查变量，否则，它基本上是一样的。

# 使用 Nisight 来理解 CUDA 中的 warp lockstep 属性

我们现在将使用 Nisight 逐步执行一些代码，以帮助我们更好地理解一些 CUDA GPU 架构，以及内核内的**分支**是如何处理的。这将使我们对如何编写更有效的 CUDA 内核有一些见解。通过分支，我们指的是 GPU 如何处理 CUDA 内核中的`if`、`else`或`switch`等控制流语句。特别是，我们对内核内的**分支分歧**如何处理感兴趣，即当内核中的一个线程满足成为`if`语句的条件时，而另一个线程不满足条件并成为`else`语句时会发生什么：它们是分歧的，因为它们执行不同的代码片段。

让我们写一个小的 CUDA-C 程序作为实验：我们将从一个小内核开始，如果其`threadIdx.x`值是偶数，则打印一个输出，如果是奇数，则打印另一个输出。然后，我们编写一个`main`函数，将这个内核启动到由 32 个不同线程组成的一个单一块上：

```py
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void divergence_test_ker()
{
    if( threadIdx.x % 2 == 0)
        printf("threadIdx.x %d : This is an even thread.\n", threadIdx.x);
    else
        printf("threadIdx.x %d : This is an odd thread.\n", threadIdx.x);
}

__host__ int main()
{
    cudaSetDevice(0);
    divergence_test_ker<<<1, 32>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
}
```

（此代码也可在存储库中的`divergence_test.cu`中找到。）

如果我们从命令行编译和运行这个程序，我们可能天真地期望偶数和奇数线程之间会有交错的字符串序列；或者它们可能会随机交错——因为所有线程都是并发运行并且大约在同一时间分支，这是有道理的。

相反，每次我们运行这个程序，我们总是得到这个输出：

![](img/38c855ca-4bd8-4ab6-9e6b-371a9c9c0bb1.png)

所有与偶数线程对应的字符串都先打印出来，而所有与奇数线程对应的字符串都在第二次打印出来。也许 Nisight 调试器可以解释一些问题；让我们像在上一节中那样将这个小程序导入 Nisight 项目，并在内核的第一个`if`语句处设置断点。然后我们将执行*step over*，这样调试器就会在第一个`printf`语句处停下来。由于 Nisight 中的默认线程是（0,0,0），这应该满足了第一个`if`语句，所以它会一直停在那里，直到调试器继续。

让我们切换到一个奇数线程，比如（1,0,0），看看它现在在我们的程序中的位置：

![](img/58816167-07da-44ae-954a-4b6c3e911bbd.png)

非常奇怪！线程（1,0,0）在执行中也与线程（0,0,0）处于相同的位置。实际上，如果我们在这里检查每一个其他奇数线程，它们都会停在同一个地方——在一个所有奇数线程应该跳过的`printf`语句处。

这是什么？这被称为**warp 锁步特性**。CUDA 架构中的一个**warp**是一个由 32 个“通道”组成的单元，在这个单元中，我们的 GPU 执行内核和网格，其中每个通道将执行一个线程。warp 的一个主要限制是，所有在一个 warp 上执行的线程必须以**锁步**的方式执行相同的代码；这意味着并非每个线程确实运行相同的代码，而只是忽略对它不适用的步骤。（这被称为锁步，因为就像一群士兵一起*锁步*行进一样——不管他们是否想要行进！）

锁步特性意味着如果一个 warp 上运行的单个线程在单个`if`语句中与其他 31 个线程产生分歧，所有其他 31 个线程的执行都会延迟，直到这个单独的异常线程完成并从其孤立的`if`分歧中返回。这是在编写内核时应该牢记的一个特性，也是为什么在 CUDA 编程中，分支分歧应该尽量减少的一般规则。

# 使用 NVIDIA nvprof 分析器和 Visual Profiler

最后，我们将简要概述命令行 Nvidia `nvprof`分析器。与 Nsight IDE 相比，我们可以自由使用我们编写的任何 Python 代码——我们不必在这里强制编写完整的、纯粹的 CUDA-C 测试函数代码。

我们可以使用`nvprof program`命令对二进制可执行程序进行基本分析；同样，我们可以使用`python`命令作为第一个参数，使用脚本作为第二个参数来对 Python 脚本进行分析，如下所示：`nvprof python program.py`。让我们使用`nvprof matrix_ker`对我们之前编写的简单矩阵乘法 CUDA-C 可执行程序进行分析： 

![](img/41fb00d0-582b-4b22-81ea-b9ae1988ad84.png)

我们看到这与我们最初使用的 Python cProfiler 模块输出非常相似，我们用它来分析第一章中的 Mandelbrot 算法——只是现在，这专门告诉我们有关执行的所有 CUDA 操作。因此，当我们专门想要在 GPU 上进行优化时，我们可以使用它，而不必关心在主机上执行的任何 Python 或其他命令。（如果我们添加`--print-gpu-trace`命令行选项，我们可以进一步分析每个单独的 CUDA 内核操作，包括块和网格大小的启动参数。）

让我们再看一个技巧，帮助我们*可视化*程序所有操作的执行时间；我们将使用`nvprof`来转储一个文件，然后可以由 NVIDIA Visual Profiler 读取，以图形方式显示给我们。我们将使用上一章的示例`multi-kernel_streams.py`（在存储库的`5`下可用）来做这个。让我们回忆一下，这是我们对 CUDA 流概念的介绍示例之一，它允许我们同时执行和组织多个 GPU 操作。我们将使用`-o`命令行选项将输出转储到一个带有`.nvvp`文件后缀的文件中，如下所示：`nvprof -o m.nvvp python multi-kernel_streams.py`。现在我们可以使用`nvvp m.nvvp`命令将此文件加载到 NVIDIA Visual Profiler 中。

我们应该在所有 CUDA 流上看到一个时间线（记住，此程序中使用的内核名称为`mult_ker`）：

![](img/37df8c10-9b93-4d86-bb74-ff00920e7470.png)

不仅可以看到所有的内核启动，还可以看到内存分配、内存复制和其他操作。这对于直观和视觉上理解程序如何随着时间使用 GPU 是有用的。

# 总结

我们在本章开始时看到了如何在 CUDA 内核中使用`printf`来输出来自各个线程的数据；我们特别看到了这对于调试代码有多么有用。然后，我们涵盖了 CUDA-C 中我们知识的一些空白，以便我们可以编写完整的测试程序，将其编译成适当的可执行二进制文件：这里有很多开销在我们之前是隐藏的，我们必须非常谨慎。接下来，我们看到了如何在 Nsight IDE 中创建和编译项目以及如何使用它进行调试。我们看到了如何在 CUDA 内核中停止我们设置的任何断点，并在不同的本地变量之间切换以查看各个线程。我们还使用了 Nsight 调试器来了解 warp 锁步属性以及为什么在 CUDA 内核中避免分支分歧很重要。最后，我们对 NVIDIA 命令行`nvprof`分析器和 Visual Profiler 进行了非常简要的概述，用于分析我们的 GPU 代码。

# 问题

1.  在我们编写的第一个 CUDA-C 程序中，我们在使用`cudaMalloc`在 GPU 上分配内存数组之后没有使用`cudaDeviceSynchronize`命令。为什么这是不必要的？（提示：回顾上一章。）

1.  假设我们有一个单个内核，它在由两个块组成的网格上启动，每个块有 32 个线程。假设第一个块中的所有线程都执行一个`if`语句，而第二个块中的所有线程都执行相应的`else`语句。第二个块中的所有线程是否必须像第一个块中的线程一样“锁步”执行`if`语句中的命令？

1.  如果我们执行类似的代码片段，只是在由 64 个线程执行的一个单一块组成的网格上执行，其中前 32 个线程执行一个`if`，而后 32 个执行一个`else`语句，会怎么样？

1.  `nvprof`分析器可以为我们测量哪些 Python 的 cProfiler 无法测量的内容？

1.  列举一些我们可能更喜欢使用`printf`来调试 CUDA 内核的情境，以及其他一些情境，其中使用 Nsight 来调试 CUDA 内核可能更容易。

1.  `cudaSetDevice`命令在 CUDA-C 中的目的是什么？

1.  为什么我们在每次 CUDA-C 中的内核启动或内存复制后都必须使用`cudaDeviceSynchronize`？
