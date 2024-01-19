# 使用 Scikit-CUDA 与 CUDA 库

在本章中，我们将介绍三个用于简化数值和科学计算的标准 CUDA 库。我们将首先看一下**cuBLAS**，这是 NVIDIA 针对 CUDA 的**基本线性代数子程序**（**BLAS**）规范的实现。（cuBLAS 是 NVIDIA 对 BLAS 的各种优化的 CPU 实现的回应，例如免费/开源的 OpenBLAS 或英特尔的专有数学核心库。）接下来我们将看一下**cuFFT**，它可以在 GPU 上执行几乎每种**快速傅里叶变换**（**FFT**）的变体。我们将看看如何在图像处理中使用 cuFFT 进行滤波。然后我们将看一下**cuSolver**，它可以执行比 cuBLAS 中更复杂的线性代数运算，例如**奇异值分解**（**SVD**）或乔列斯基分解。

到目前为止，我们主要处理了一个作为我们与 CUDA 网关的单个 Python 模块——PyCUDA。虽然 PyCUDA 是一个非常强大和多功能的 Python 库，但它的主要目的是提供一个网关来编写、编译和启动 CUDA 内核，而不是提供一个接口给 CUDA 库。幸运的是，有一个免费的 Python 模块可用，它提供了一个用户友好的包装器接口给这些库。这就是 Scikit-CUDA。

虽然您不必了解 PyCUDA 甚至理解 GPU 编程就能欣赏 Scikit-CUDA，但它与 PyCUDA 兼容，例如，Scikit-CUDA 可以轻松地与 PyCUDA 的`gpuarray`类一起使用，这使您可以轻松地在我们自己的 CUDA 内核例程和 Scikit-CUDA 之间传递数据。此外，大多数例程也可以与 PyCUDA 的 stream 类一起使用，这将允许我们正确地同步我们自己的自定义 CUDA 内核和 Scikit-CUDA 的包装器。

请注意，除了这三个列出的库之外，Scikit-CUDA 还为专有的 CULA 库提供了包装器，以及开源的 MAGMA 库。这两者在功能上与官方的 NVIDIA 库有很多重叠。由于这些库在标准 CUDA 安装中默认未安装，我们将选择不在本章中涵盖它们。感兴趣的读者可以分别在[`www.culatools.com`](http://www.culatools.com)和[`icl.utk.edu/magma/`](http://icl.utk.edu/magma/)了解更多关于 CULA 和 MAGMA 的信息。建议读者查看 Scikit-CUDA 的官方文档，网址为：[`media.readthedocs.org/pdf/scikit-cuda/latest/scikit-cuda.pdf`](https://media.readthedocs.org/pdf/scikit-cuda/latest/scikit-cuda.pdf)。

本章的学习成果如下：

+   了解如何安装 Scikit-CUDA

+   了解标准 CUDA 库的基本目的和区别

+   了解如何使用低级 cuBLAS 函数进行基本线性代数

+   了解如何使用 SGEMM 和 DGEMM 操作来测量 GPU 在 FLOPS 中的性能

+   了解如何使用 cuFFT 在 GPU 上执行 1D 或 2D FFT 操作

+   了解如何使用 FFT 创建 2D 卷积滤波器，并将其应用于简单的图像处理

+   了解如何使用 cuSolver 执行奇异值分解（SVD）

+   了解如何使用 cuSolver 的 SVD 算法执行基本主成分分析

# 技术要求

本章需要一台安装了现代 NVIDIA GPU（2016 年及以后）的 Linux 或 Windows 10 PC，并安装了所有必要的 GPU 驱动程序和 CUDA Toolkit（9.0 及以后）。还需要一个合适的 Python 2.7 安装（如 Anaconda Python 2.7），其中包括 PyCUDA 模块。

本章的代码也可以在 GitHub 上找到，网址为：[`github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA`](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)。

有关先决条件的更多信息，请查看本书的前言。有关软件和硬件要求的更多信息，请查看[`github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA`](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)上的 README 文件。

# 安装 Scikit-CUDA

建议您直接从 GitHub 安装最新稳定版本的 Scikit-CUDA：[`github.com/lebedov/scikit-cuda`](https://github.com/lebedov/scikit-cuda)。

将软件包解压缩到一个目录中，然后在此处打开命令行，并键入`python setup.py install`来安装模块。然后，您可以运行单元测试以确保已使用`python setup.py test`执行了正确的安装。（此方法建议 Windows 和 Linux 用户均使用。）或者，可以直接使用`pip install scikit-cuda`从 PyPI 存储库安装 Scikit-CUDA。

# 使用 cuBLAS 进行基本线性代数

我们将从学习如何使用 Scikit-CUDA 的 cuBLAS 包装器开始这一章。让我们花一点时间讨论 BLAS。BLAS（基本线性代数子程序）是一个基本线性代数库的规范，最早是在 1970 年代标准化的。BLAS 函数被分为几个类别，被称为*级别*。

Level 1 BLAS 函数包括纯粹在向量上的操作——向量-向量加法和缩放（也称为*ax+y*操作，或 AXPY），点积和范数。Level 2 BLAS 函数包括一般矩阵-向量操作（GEMV），例如矩阵与向量的乘法，而 Level 3 BLAS 函数包括“一般矩阵-矩阵”（GEMM）操作，例如矩阵-矩阵乘法。最初，这些库是在 1970 年代完全用 FORTRAN 编写的，因此您应该考虑到在使用和命名上可能存在一些看似过时的遗留问题，这可能对今天的新用户来说显得繁琐。

cuBLAS 是 NVIDIA 自己对 BLAS 规范的实现，当然是经过优化以充分利用 GPU 的并行性。Scikit-CUDA 提供了与 PyCUDA `gpuarray`对象兼容的 cuBLAS 包装器，以及与 PyCUDA 流兼容的包装器。这意味着我们可以通过 PyCUDA 将这些函数与我们自己的自定义 CUDA-C 内核耦合和接口，以及在多个流上同步这些操作。

# 使用 cuBLAS 进行 Level-1 AXPY

现在让我们从 cuBLAS 开始进行基本的 Level-1 *ax + y*（或 AXPY）操作。让我们停下来，回顾一下线性代数的一点，并思考这意味着什么。在这里，*a*被认为是一个标量；也就是说，一个实数，比如-10、0、1.345 或 100。*x*和*y*被认为是某个向量空间中的向量，![](img/47a6873c-3e1b-4b3c-95e8-d1a3a4f796eb.png)。这意味着*x*和*y*是实数的 n 元组，因此在![](img/d0e81dc7-0fa8-4bce-a264-941fee2e3ad7.png)的情况下，这些值可以是`[1,2,3]`或`[-0.345, 8.15, -15.867]`。*ax*表示*x*的缩放乘以*a*，因此如果*a*是 10 且*x*是先前的第一个值，则*ax*是*x*的每个单独值乘以*a*；也就是`[10, 20, 30]`。最后，和*ax + y*表示我们将两个向量中每个槽的每个单独值相加以产生一个新的向量，假设*y*是给定的第二个向量，结果将如下所示-`[9.655, 28.15, 14.133]`。

现在让我们在 cuBLAS 中做这个。首先，让我们导入适当的模块：

```py
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
```

现在让我们导入 cuBLAS：

```py
from skcuda import cublas
```

我们现在可以设置我们的向量数组并将它们复制到 GPU。请注意，我们使用 32 位（单精度）浮点数：

```py
a = np.float32(10)
x = np.float32([1,2,3])
y = np.float32([-.345,8.15,-15.867])
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)
```

现在我们必须创建一个**cuBLAS 上下文**。这在本质上类似于 CUDA 上下文，在第五章中我们讨论过，只是这一次它被显式用于管理 cuBLAS 会话。`cublasCreate`函数创建一个 cuBLAS 上下文，并将其句柄作为输出。我们需要保存这个句柄，只要我们打算在此会话中使用 cuBLAS：

```py
cublas_context_h = cublas.cublasCreate()
```

现在我们可以使用`cublasSaxpy`函数。`S`代表单精度，这是我们需要的，因为我们正在处理 32 位浮点数组：

```py
cublas.cublasSaxpy(cublas_context_h, x_gpu.size, a, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
```

让我们讨论一下我们刚刚做的事情。还要记住，这是一个直接包装到低级 C 函数的函数，因此输入可能更像 C 函数而不是真正的 Python 函数。简而言之，这执行了一个“AXPY”操作，最终将输出数据放入`y_gpu`数组中。让我们逐个讨论每个输入参数。

第一个输入始终是 CUDA 上下文句柄。然后我们必须指定向量的大小，因为这个函数最终将在 C 指针上操作；我们可以使用 gpuarray 的`size`参数来做到这一点。已经将我们的标量类型转换为 NumPy 的`float32`变量，我们可以将`a`变量直接作为标量参数传递。然后我们使用`gpudata`参数将`x_gpu`数组的底层 C 指针传递给这个函数。然后我们将第一个数组的**步长**设置为 1：步长指定每个输入值之间应该走多少步。 （相反，如果您正在使用来自行向矩阵的列的向量，则将步长设置为矩阵的宽度。）然后我们放入`y_gpu`数组的指针，并将其步长也设置为 1。

我们已经完成了计算；现在我们必须明确销毁我们的 cuBLAS 上下文：

```py
cublas.cublasDestroy(cublas_context)
```

现在我们可以使用 NumPy 的`allclose`函数来验证这是否接近，就像这样：

```py
print 'This is close to the NumPy approximation: %s' % np.allclose(a*x + y , y_gpu.get())
```

再次注意，最终输出被放入了`y_gpu`数组中，这也是一个输入。

始终记住，BLAS 和 CuBLAS 函数是就地操作，以节省时间和内存，而不是进行新的分配调用。这意味着输入数组也将用作输出！

我们刚刚看到如何使用`cublasSaxpy`函数执行`AXPY`操作。

让我们讨论突出的大写字母 S。正如我们之前提到的，这代表单精度，即 32 位实数浮点值（`float32`）。如果我们想要操作 64 位实数浮点值数组（NumPy 和 PyCUDA 中的`float64`），那么我们将使用`cublasDaxpy`函数；对于 64 位单精度复数值（`complex64`），我们将使用`cublasCaxpy`，而对于 128 位双精度复数值（`complex128`），我们将使用`cublasZaxpy`。

我们可以通过检查函数名称其余部分之前的字母来确定 BLAS 或 CuBLAS 函数操作的数据类型。使用单精度实数的函数总是以 S 开头，双精度实数以 D 开头，单精度复数以 C 开头，双精度复数以 Z 开头。

# 其他一级 cuBLAS 函数

让我们看看其他一些一级函数。我们不会深入介绍它们的操作，但步骤与我们刚刚介绍的类似：创建一个 cuBLAS 上下文，使用适当的数组指针调用函数（可以通过 PyCUDA 的`gpuarray`的`gpudata`参数访问），并相应地设置步长。另一件需要记住的事情是，如果函数的输出是单个值而不是数组（例如，点积函数），则函数将直接将该值输出到主机，而不是在必须从 GPU 中取出的内存数组中。（我们只会在这里介绍单精度实数版本，但其他数据类型的相应版本可以通过用适当的字母替换 S 来使用。）

我们可以对两个单精度实数`gpuarray`，`v_gpu`和`w_gpu`进行点积。再次，1 是为了确保我们在这个计算中使用步长 1！再次回想一下，点积是两个向量的逐点乘积的和：

```py
dot_output = cublas.cublasSdot(cublas_context_h, v_gpu.size, v_gpu.gpudata, 1, w_gpu.gpudata, 1)
```

我们还可以执行向量的 L2 范数，就像这样（回想一下，对于一个向量*x*，这是它的 L2 范数，或长度，可以用![](img/839337d6-db29-481e-8467-bcd415a2ad7c.png)公式来计算）：

```py
l2_output = cublas.cublasSnrm2(cublas_context_h, v_gpu.size, v_gpu.gpudata, 1)
```

# cuBLAS 中的二级 GEMV

让我们看看如何进行`GEMV`矩阵-向量乘法。对于一个*m* x *n*矩阵*A*，一个 n 维向量*x*，一个*m*维向量*y*，以及标量*alpha*和*beta*，这被定义为以下操作：

![](img/0b6277ff-e027-45fe-ad2e-d312ea1a38f5.png)

现在让我们在继续之前看一下函数的布局：

```py
cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)  
```

让我们逐个检查这些输入：

+   `handle`指的是 cuBLAS 上下文句柄。

+   `trans`指的是矩阵的结构——我们可以指定是否要使用原始矩阵、直接转置或共轭转置（对于复数矩阵）。这很重要要记住，因为这个函数将期望矩阵`A`以**列主**格式存储。

+   `m`和`n`是我们想要使用的矩阵`A`的行数和列数。

+   `alpha`是浮点值*α*。

+   `A`是*m x n*矩阵*A*。

+   `lda`指示矩阵的主维度，矩阵的总大小实际上是`lda` x `n`。这在列主格式中很重要，因为如果`lda`大于`m`，这可能会导致 cuBLAS 在尝试访问`A`的值时出现问题，因为该矩阵的基础结构是一个一维数组。

+   然后我们有`x`及其步长`incx`；`x`是被`A`相乘的向量的基础 C 指针。记住，`x`的大小必须是`n`；也就是说，`A`的列数。

+   `beta`是浮点值*β*。

+   最后，我们有`y`及其步长`incy`作为最后的参数。我们应该记住，`y`的大小应该是`m`，或者`A`的行数。

让我们通过生成一个 10 x 100 的随机值矩阵`A`，和一个包含 100 个随机值的向量`x`来测试这个。我们将初始化`y`为一个包含 10 个零的矩阵。我们将 alpha 设置为 1，beta 设置为 0，以便直接进行矩阵乘法而不进行缩放：

```py
m = 10
n = 100
alpha = 1
beta = 0
A = np.random.rand(m,n).astype('float32')
x = np.random.rand(n).astype('float32')
y = np.zeros(m).astype('float32')
```

我们现在必须将`A`转换为**列主**（或按列）格式。NumPy 默认将矩阵存储为**行主**（或按行）格式，这意味着用于存储矩阵的基础一维数组会遍历所有第一行的值，然后遍历所有第二行的值，依此类推。您应该记住，转置操作会将矩阵的列与其行交换。然而，结果将是转置矩阵的新一维数组将以列主格式表示原始矩阵。我们可以通过`A.T.copy()`这样做，将`A`的转置矩阵的副本以及`x`和`y`一起复制到 GPU 上：

```py
A_columnwise = A.T.copy()
A_gpu = gpuarray.to_gpu(A_columnwise) 
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)
```

由于我们现在已经正确地在 GPU 上存储了按列的矩阵，我们可以通过使用`_CUBLAS_OP`字典将`trans`变量设置为不进行转置：

```py
trans = cublas._CUBLAS_OP['N']
```

由于矩阵的大小与我们想要使用的行数完全相同，我们现在将`lda`设置为`m`。*x*和*y*向量的步长再次为 1。我们现在已经设置好了所有需要的值，现在可以创建我们的 CuBLAS 上下文并存储它的句柄，就像这样：

```py
lda = m 
incx = 1
incy = 1
handle = cublas.cublasCreate()
```

我们现在可以启动我们的函数。记住，`A`，`x`和`y`实际上是 PyCUDA `gpuarray`对象，所以我们必须使用`gpudata`参数输入到这个函数中。除了这个，这很简单：

```py
cublas.cublasSgemv(handle, trans, m, n, alpha, A_gpu.gpudata, lda, x_gpu.gpudata, incx, beta, y_gpu.gpudata, incy)
```

现在我们可以销毁我们的 cuBLAS 上下文并检查返回值以确保它是正确的：

```py
cublas.cublasDestroy(handle)
print 'cuBLAS returned the correct value: %s' % np.allclose(np.dot(A,x), y_gpu.get())
```

# cuBLAS 中的三级 GEMM 用于测量 GPU 性能

现在我们将看看如何使用 CuBLAS 执行**通用矩阵-矩阵乘法**（**GEMM**）。实际上，我们将尝试制作一些比我们在 cuBLAS 中看到的最后几个示例更实用的东西-我们将使用这个作为我们的 GPU 性能指标，以确定它可以执行的**每秒浮点运算次数**（**FLOPS**）的数量，这将是两个单独的值：单精度和双精度的情况。使用 GEMM 是评估 FLOPS 中计算硬件性能的标准技术，因为它比使用纯时钟速度（MHz 或 GHz）更好地理解了纯计算能力。

如果您需要简要回顾，请回想一下我们在上一章中深入讨论了矩阵-矩阵乘法。如果您忘记了这是如何工作的，强烈建议您在继续本节之前复习一下这一章。

首先，让我们看看 GEMM 操作是如何定义的：

![](img/6732ed55-6eea-497a-adcb-95731cc211b9.png)

这意味着我们执行*A*和*B*的矩阵乘法，将结果缩放为*alpha*，然后加到我们已经通过*beta*缩放的*C*矩阵中，将最终结果放在*C*中。

让我们考虑一下执行实值 GEMM 操作的最终结果需要执行多少浮点运算，假设*A*是一个*m* x *k*（其中*m*是行，*k*是列）矩阵，*B*是一个*k* x *n*矩阵，C 是一个*m* x *n*矩阵。首先，让我们计算*AB*需要多少操作。让我们取*A*的一列并将其乘以*B*：这将导致*m*行中的每一行需要*k*次乘法和*k-1*次加法，这意味着这是*m*行上的*km + (k-1)m*总操作。*B*中有*n*列，因此计算*AB*将总共需要*kmn + (k-1)mn = 2kmn - mn*次操作。现在，我们使用*alpha*来缩放*AB*，这将是*m**n*次操作，因为这是矩阵*AB*的大小；类似地，通过*beta*缩放*C*是另外*m**n*次操作。最后，我们将这两个结果矩阵相加，这又是*mn*次操作。这意味着在给定的 GEMM 操作中，我们将有*2kmn - mn + 3mn = 2kmn + 2mn = 2mn(k+1)*次浮点运算。

现在我们唯一需要做的就是运行一个计时的 GEMM 操作，注意矩阵的不同大小，并将*2kmn + 2mn*除以总时间持续时间来计算我们 GPU 的 FLOPS。结果数字将非常大，因此我们将以 GFLOPS 的形式表示这一点-也就是说，每秒可以计算多少十亿（10⁹）次操作。我们可以通过将 FLOPS 值乘以 10^(-9)来计算这一点。

现在我们准备开始编写代码。让我们从我们的导入语句开始，以及`time`函数：

```py
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import cublas
from time import time
```

现在我们将为我们的矩阵大小设置`m`，`n`和`k`变量。我们希望我们的矩阵相对较大，以便时间持续足够长，以避免除以 0 的错误。以下值对于截至 2018 年中旬或更早发布的任何 GPU 来说应该足够了；拥有更新卡的用户可能考虑增加这些值：

```py
m = 5000
n = 10000
k = 10000
```

现在我们将编写一个计算单精度和双精度 GFLOPS 的函数。如果我们希望使用双精度，则将输入值设置为'D'，否则设置为'S'：

```py
def compute_gflops(precision='S'):

if precision=='S':
    float_type = 'float32'
elif precision=='D':
    float_type = 'float64'
else:
    return -1
```

现在让我们生成一些随机矩阵，这些矩阵具有我们将用于计时的适当精度。GEMM 操作与我们之前看到的 GEMV 操作类似，因此我们必须在将它们复制到 GPU 之前对其进行转置。（由于我们只是在计时，这一步并不是必要的，但记住这一点是个好习惯。）

我们将为 GEMM 设置一些其他必要的变量，这些变量在这一点上应该是不言自明的（`transa`，`lda`，`ldb`等）：

```py
A = np.random.randn(m, k).astype(float_type)
B = np.random.randn(k, n).astype(float_type)
C = np.random.randn(m, n).astype(float_type)
A_cm = A.T.copy()
B_cm = B.T.copy()
C_cm = C.T.copy()
A_gpu = gpuarray.to_gpu(A_cm)
B_gpu = gpuarray.to_gpu(B_cm)
C_gpu = gpuarray.to_gpu(C_cm)
alpha = np.random.randn()
beta = np.random.randn()
transa = cublas._CUBLAS_OP['N']
transb = cublas._CUBLAS_OP['N']
lda = m
ldb = k
ldc = m
```

现在我们可以开始计时了！首先，我们将创建一个 cuBLAS 上下文：

```py
t = time()
handle = cublas.cublasCreate()
```

现在我们将启动 GEMM。请记住，对于实数情况有两个版本：`cublasSgemm`用于单精度，`cublasDgemm`用于双精度。我们可以使用一个小小的 Python 技巧执行适当的函数：我们将用`cublas%sgemm`和适当的参数写一个字符串，然后通过附加`% precision`将`%s`替换为 D 或 S。然后我们将使用`exec`函数将这个字符串作为 Python 代码执行，就像这样：

```py
exec('cublas.cublas%sgemm(handle, transa, transb, m, n, k, alpha, A_gpu.gpudata, lda, B_gpu.gpudata, ldb, beta, C_gpu.gpudata, ldc)' % precision)
```

现在我们可以销毁 cuBLAS 上下文，并得到我们计算的最终时间：

```py
cublas.cublasDestroy(handle)
t = time() - t
```

然后我们需要使用我们推导出的方程计算 GFLOPS，并将其作为这个函数的输出返回：

```py
gflops = 2*m*n*(k+1)*(10**-9) / t 
return gflops
```

现在我们可以设置我们的主函数。我们将输出单精度和双精度情况下的 GFLOPS：

```py
if __name__ == '__main__':
    print 'Single-precision performance: %s GFLOPS' % compute_gflops('S')
    print 'Double-precision performance: %s GFLOPS' % compute_gflops('D')
```

现在在运行这个程序之前，让我们做一点功课——去[`www.techpowerup.com`](https://www.techpowerup.com)搜索你的 GPU，然后注意两件事——单精度浮点性能和双精度浮点性能。我现在使用的是 GTX 1050，它的列表声称在单精度上有 1,862 GFLOPS 性能，在双精度上有 58.20 GFLOPS 性能。让我们现在运行这个程序，看看这是否符合事实：

![](img/0d014970-b902-4cd8-804a-433bf0b83d77.png)

看哪，它成功了！

这个程序也可以在这本书的存储库目录下的`cublas_gemm_flops.py`文件中找到。

# 使用 cuFFT 进行快速傅里叶变换

现在让我们看看如何使用 cuFFT 进行一些基本的**快速傅里叶变换**（**FFT**）。首先，让我们简要回顾一下傅里叶变换到底是什么。如果你上过高级微积分或分析课程，你可能已经见过傅里叶变换被定义为一个积分公式，就像这样：

![](img/d1c79c32-6eab-4a52-a2ef-4af0bc192f5c.png)

这个程序将*f*作为*x*的时间域函数。这给了我们一个相应的频域函数，对应于"ξ"。这事实上是一个极其有用的工具，几乎触及到所有科学和工程的分支。

让我们记住积分可以被看作是一个求和；同样地，有一个对应的离散、有限版本的傅里叶变换，称为**离散傅里叶变换**（**DFT**）。这对长度为*n*的向量进行操作，并允许它们在频域中进行分析或修改。*x*的 DFT 被定义如下：

![](img/8b60bac2-8488-4c5f-9d90-3ac7eb73bd62.png)

换句话说，我们可以将一个向量*x*乘以复杂的*N* x *N*矩阵 ![](img/96b3a1fb-9202-44fa-8b1d-381398412504.png)

（这里，*k*对应行号，*n*对应列号）来找到它的 DFT。我们还应该注意到逆公式，它让我们从它的 DFT 中检索*x*（在这里用*x*的 DFT 替换*y*，输出将是原始的*x*）：

![](img/25229fe6-66c3-4ca5-b8d7-96ba5e639917.png)

通常，计算矩阵-向量操作的计算复杂度对于长度为*N*的向量是 O(*N²*)。然而，由于 DFT 矩阵中的对称性，这总是可以通过使用 FFT 减少到 O(*N log N*)。让我们看看如何使用 FFT 与 CuBLAS，然后我们将继续一个更有趣的例子。

# 一个简单的一维 FFT

让我们首先看看如何使用 cuBLAS 计算简单的一维 FFT。首先，我们将简要讨论 Scikit-CUDA 中的 cuFFT 接口。

这里有两个子模块，我们可以使用`cufft`和`fft`访问 cuFFT 库。`cufft`包括了一系列 cuFFT 库的低级封装，而`fft`提供了一个更加用户友好的接口；在本章中我们将只使用`fft`。

让我们从适当的导入开始，记得包括 Scikit-CUDA 的`fft`子模块：

```py
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import fft
```

现在我们将设置一些随机数组并将其复制到 GPU。我们还将设置一个空的 GPU 数组，用于存储 FFT（请注意，我们使用实数 float32 数组作为输入，但输出将是 complex64 数组，因为傅立叶变换总是复值）：

```py
x = np.asarray(np.random.rand(1000), dtype=np.float32 )
x_gpu = gpuarray.to_gpu(x)
x_hat = gpuarray.empty_like(x_gpu, dtype=np.complex64)
```

我们现在将为正向 FFT 变换设置一个 cuFFT 计划。这是 cuFFT 用来确定变换的形状，以及输入和输出数据类型的对象：

```py
plan = fft.Plan(x_gpu.shape,np.float32,np.complex64)
```

我们还将为逆 FFT 计划对象设置一个计划。请注意，这一次我们从`complex64`到实数`float32`：

```py
inverse_plan = fft.Plan(x.shape, in_dtype=np.complex64, out_dtype=np.float32)
```

现在，我们必须将`x_gpu`的正向 FFT 转换为`x_hat`，并将`x_hat`的逆 FFT 转换回`x_gpu`。请注意，在逆 FFT 中设置了`scale=True`；我们这样做是为了告诉 cuFFT 将逆 FFT 按 1/N 进行缩放：

```py
fft.fft(x_gpu, x_hat, plan)
fft.ifft(x_hat, x_gpu, inverse_plan, scale=True)
```

我们现在将检查`x_hat`与`x`的 NumPy FFT，以及`x_gpu`与`x`本身：

```py
y = np.fft.fft(x)
print 'cuFFT matches NumPy FFT: %s' % np.allclose(x_hat.get(), y, atol=1e-6)
print 'cuFFT inverse matches original: %s' % np.allclose(x_gpu.get(), x, atol=1e-6)
```

如果你运行这个程序，你会发现`x_hat`与`y`不匹配，然而，莫名其妙的是，`x_gpu`与`x`匹配。这是怎么可能的？好吧，让我们记住`x`是实数；如果你看一下离散傅立叶变换是如何计算的，你可以数学上证明实向量的输出在 N/2 之后会重复为它们的复共轭。虽然 NumPy FFT 会完全计算这些值，但 cuFFT 通过只计算输入为实数时的输出的前半部分来节省时间，并将其余输出设置为`0`。你应该通过检查前面的变量来验证这一点。

因此，如果我们将前面代码中的第一个打印语句更改为仅比较 CuFFT 和 NumPy 之间的前 N/2 个输出，那么这将返回 true：

```py
print 'cuFFT matches NumPy FFT: %s' % np.allclose(x_hat.get()[0:N//2], y[0:N//2], atol=1e-6)
```

# 使用 FFT 进行卷积

我们现在将看看如何使用 FFT 执行**卷积**。首先让我们回顾一下卷积的确切定义：给定两个一维向量*x*和*y*，它们的卷积定义如下：

![](img/34574397-830b-446f-8e5f-34b468e76b3e.png)

这对我们很有意义，因为如果*x*是一些长的连续信号，而*y*只有一小部分局部非零值，那么*y*将作为*x*的滤波器；这本身有许多应用。首先，我们可以使用滤波器平滑信号*x*（在数字信号处理和图像处理中很常见）。我们还可以使用它来收集信号*x*的样本，以表示信号或压缩它（在数据压缩或压缩感知领域很常见），或者使用滤波器为机器学习中的信号或图像识别收集特征。这个想法构成了卷积神经网络的基础。

当然，计算机无法处理无限长的向量（至少目前还不能），所以我们将考虑**循环卷积**。在循环卷积中，我们处理两个长度为*n*的向量，其索引小于 0 或大于 n-1 的部分将会环绕到另一端；也就是说，*x*[-1] = *x*[n-1]，*x*[-2] = *x*[n-2]，*x*[n] = *x*[0]，*x*[n+1] = *x*[1]，依此类推。我们定义*x*和*y*的循环卷积如下：

![](img/69ee6cc9-5c17-40f2-a4f5-675f5a0a9ee2.png)

事实证明，我们可以很容易地使用 FFT 执行循环卷积；我们可以对*x*和*y*执行 FFT，对输出进行逐点乘法，然后对最终结果执行逆 FFT。这个结果被称为**卷积定理**，也可以表示如下：

![](img/d4a9eaac-c5fb-4ec2-8544-35b8e7388209.png)

我们将在两个维度上进行这个操作，因为我们希望将结果应用到信号处理上。虽然我们只看到了一维卷积和 FFT 的数学运算，但二维卷积和 FFT 的工作方式与一维类似，只是索引更复杂一些。然而，我们选择跳过这一点，以便我们可以直接进入应用。

# 使用 cuFFT 进行 2D 卷积

现在我们将制作一个小程序，使用基于 cuFFT 的二维卷积对图像进行**高斯滤波**。高斯滤波是一种使用所谓的高斯滤波器平滑粗糙图像的操作。之所以这样命名，是因为它基于统计学中的高斯（正态）分布。这是高斯滤波器在两个维度上以标准偏差σ定义的方式：

![](img/06f381b9-2e9d-48aa-95e7-265b988f144d.png)

当我们用滤波器对离散图像进行卷积时，有时我们会将滤波器称为**卷积核**。通常，图像处理工程师会简单地称之为核，但由于我们不想将其与 CUDA 核混淆，我们将始终使用完整术语卷积核。在这里，我们将使用离散版本的高斯滤波器作为我们的卷积核。

让我们从适当的导入开始；请注意，我们将在这里使用 Scikit-CUDA 子模块`linalg`。这将为我们提供比 cuBLAS 更高级的接口。由于我们在这里处理图像，我们还将导入 Matplotlib 的`pyplot`子模块。还要注意，我们将在此处使用 Python 3 风格的除法，从第一行开始；这意味着如果我们使用`/`运算符除两个整数，那么返回值将是浮点数，无需类型转换（我们使用`//`运算符进行整数除法）：

```py
from __future__ import division
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import fft
from skcuda import linalg
from matplotlib import pyplot as plt
```

让我们立即开始编写卷积函数。这将接受两个相同大小的 NumPy 数组`x`和`y`。我们将把它们转换为 complex64 数组，然后如果它们的大小不相同，就返回`-1`：

```py
def cufft_conv(x , y):
    x = x.astype(np.complex64)
    y = y.astype(np.complex64)

    if (x.shape != y.shape):
        return -1
```

现在我们将设置我们的 FFT 计划和逆 FFT 计划对象：

```py
plan = fft.Plan(x.shape, np.complex64, np.complex64)
inverse_plan = fft.Plan(x.shape, np.complex64, np.complex64)
```

现在我们可以将我们的数组复制到 GPU。我们还将设置一些空数组，用于保存这些数组的 FFT 的适当大小，另外一个数组将保存最终卷积的输出，`out_gpu`：

```py
 x_gpu = gpuarray.to_gpu(x)
 y_gpu = gpuarray.to_gpu(y)

 x_fft = gpuarray.empty_like(x_gpu, dtype=np.complex64)
 y_fft = gpuarray.empty_like(y_gpu, dtype=np.complex64)
 out_gpu = gpuarray.empty_like(x_gpu, dtype=np.complex64)
```

现在我们可以进行我们的 FFT：

```py
fft.fft(x_gpu, x_fft, plan)
fft.fft(y_gpu, y_fft, plan)
```

我们现在将使用`linalg.multiply`函数在`x_fft`和`y_fft`之间执行逐点（Hadamard）乘法。我们将设置`overwrite=True`，以便将最终值写入`y_fft`：

```py
linalg.multiply(x_fft, y_fft, overwrite=True)
```

现在我们将调用逆 FFT，将最终结果输出到`out_gpu`。我们将这个值传输到主机并返回它：

```py
fft.ifft(y_fft, out_gpu, inverse_plan, scale=True)
conv_out = out_gpu.get()
return conv_out
```

我们还没有完成。我们的卷积核将比输入图像小得多，因此我们将不得不调整我们的两个 2D 数组的大小（卷积核和图像），使它们相等，并在它们之间执行逐点乘法。我们不仅应该确保它们相等，还需要确保我们在数组上执行**零填充**，并适当地将卷积核居中。零填充意味着我们在图像的两侧添加零缓冲区，以防止环绕错误。如果我们使用 FFT 来执行卷积，记住它是循环卷积，因此边缘将始终环绕。当我们完成卷积后，我们可以去除图像外部的缓冲区，得到最终的输出图像。

让我们创建一个名为`conv_2d`的新函数，它接受卷积核`ker`和图像`img`。填充后的图像大小将是（`2*ker.shape[0] + img.shape[0]`，`2*ker.shape[1] + img.shape[1]`）。让我们首先设置填充的卷积核。我们将创建一个这个大小的零矩阵，然后将左上角子矩阵设置为我们的卷积核，如下所示：

```py
def conv_2d(ker, img):

    padded_ker = np.zeros( (img.shape[0] + 2*ker.shape[0], img.shape[1] + 2*ker.shape[1] )).astype(np.float32)
    padded_ker[:ker.shape[0], :ker.shape[1]] = ker
```

现在我们需要移动卷积核，使其中心精确地位于坐标（0,0）处。我们可以使用 NumPy 的`roll`命令来实现这一点：

```py
padded_ker = np.roll(padded_ker, shift=-ker.shape[0]//2, axis=0)
padded_ker = np.roll(padded_ker, shift=-ker.shape[1]//2, axis=1)
```

现在我们需要对输入图像进行填充：

```py
padded_img = np.zeros_like(padded_ker).astype(np.float32)
padded_img[ker.shape[0]:-ker.shape[0], ker.shape[1]:-ker.shape[1]] = img
```

现在我们有两个大小相同且格式正确的数组。我们现在可以使用我们刚刚在这里编写的`cufft_conv`函数：

```py
out_ = cufft_conv(padded_ker, padded_img)
```

现在我们可以去除图像外部的零缓冲区。然后返回结果：

```py
output = out_[ker.shape[0]:-ker.shape[0], ker.shape[1]:-ker.shape[1]]

return output
```

我们还没有完成。让我们编写一些小函数来设置我们的高斯滤波器，然后我们可以继续将其应用于图像。我们可以使用 lambda 函数一行代码来编写基本滤波器本身：

```py
gaussian_filter = lambda x, y, sigma : (1 / np.sqrt(2*np.pi*(sigma**2)) )*np.exp( -(x**2 + y**2) / (2 * (sigma**2) ))
```

现在我们可以编写一个使用此滤波器输出离散卷积核的函数。卷积核的高度和长度将为`2*sigma + 1`，这是相当标准的：

请注意，我们通过将其值求和为`total_`并将其除以来规范化高斯核的值。

```py
def gaussian_ker(sigma):
    ker_ = np.zeros((2*sigma+1, 2*sigma+1))
    for i in range(2*sigma + 1):
        for j in range(2*sigma + 1):
            ker_[i,j] = gaussian_filter(i - sigma, j - sigma, sigma)
    total_ = np.sum(ker_.ravel())
    ker_ = ker_ */* total*_* return ker_
```

我们现在准备在图像上测试这个！作为我们的测试案例，我们将使用高斯滤波来模糊这本书编辑*Akshada Iyer*的彩色 JPEG 图像。（此图像在 GitHub 存储库的`Chapter07`目录中，文件名为`akshada.jpg`。）我们将使用 Matplotlib 的`imread`函数来读取图像；默认情况下，这将存储为 0 到 255 范围内的无符号 8 位整数数组。我们将将其强制转换为浮点数组并对其进行规范化，以便所有值的范围为 0 到 1。

对于本书印刷版的读者注意：尽管本书的印刷版是灰度图像，但这是一幅彩色图像。然后，我们将设置一个空的零数组，用于存储模糊的图像：

```py
if __name__ == '__main__':
    akshada = np.float32(plt.imread('akshada.jpg')) / 255
    akshada_blurred = np.zeros_like(akshada)
```

让我们设置我们的卷积核。在这里，标准差为 15 应该足够：

```py
ker = gaussian_ker(15)
```

现在我们可以模糊图像。由于这是一幅彩色图像，我们将不得不分别对每个颜色层（红色、绿色和蓝色）应用高斯滤波；这在图像数组的第三维中进行索引：

```py
for k in range(3):
    akshada_blurred[:,:,k] = conv_2d(ker, akshada[:,:,k])
```

现在让我们通过使用一些 Matplotlib 技巧并排查看 Before 和 After 图像：

```py
fig, (ax0, ax1) = plt.subplots(1,2)
fig.suptitle('Gaussian Filtering', fontsize=20)
ax0.set_title('Before')
ax0.axis('off')
ax0.imshow(akshada)
ax1.set_title('After')
ax1.axis('off')
ax1.imshow(akshada_blurred)
plt.tight_layout()
plt.subplots_adjust(top=.85)
plt.show()
```

现在我们可以运行程序并观察高斯滤波的效果：

![](img/ae723dc0-ccbd-4163-9c08-bb7caad3aa74.png)

该程序在此书的存储库中的`Chapter07`目录中的名为`conv_2d.py`的文件中可用。

# 使用 Scikit-CUDA 中的 cuSolver

我们现在将看看如何使用 Scikit-CUDA 的`linalg`子模块中的 cuSolver。同样，这为 cuBLAS 和 cuSolver 提供了一个高级接口，因此我们不必陷入细节。

正如我们在介绍中指出的，cuSolver 是一个用于执行比 cuBLAS 更高级的线性代数运算的库，例如奇异值分解、LU/QR/Cholesky 分解和特征值计算。由于 cuSolver、cuBLAS 和 cuFFT 一样，是另一个庞大的库，我们只会花时间来看数据科学和机器学习中最基本的操作之一——SVD。

如果您想进一步了解此库，请参考 NVIDIA 关于 cuSOLVER 的官方文档：[`docs.NVIDIA.com/cuda/cusolver/index.html`](https://docs.nvidia.com/cuda/cusolver/index.html)。

# 奇异值分解（SVD）

SVD 接受任何*m* x *n*矩阵*A*，然后返回三个矩阵—*U*、*Σ*和*V*。在这里，*U*是一个*m* x *m*酉矩阵，*Σ*是一个*m* x *n*对角矩阵，*V*是一个*n* x *n*酉矩阵。这里，*酉*表示矩阵的列形成一个正交归一基；*对角*表示矩阵中的所有值都为零，除了可能沿着对角线的值。

SVD 的重要性在于它将*A*分解为这些矩阵，使得我们有*A = UΣV^T*；此外，*Σ*对角线上的值将全部为正值或零，并且被称为奇异值。我们很快将看到一些应用，但您应该记住，SVD 的计算复杂度为 O(*mn²*)——对于大矩阵来说，使用 GPU 绝对是一个好主意，因为这个算法是可并行化的。

现在让我们看看如何计算矩阵的 SVD。让我们进行适当的导入语句：

```py
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import linalg
```

现在我们将生成一个相对较大的随机矩阵并将其传输到 GPU：

```py
a = np.random.rand(1000,5000).astype(np.float32)
a_gpu = gpuarray.to_gpu(a)
```

现在我们可以执行 SVD。这将有三个输出，对应于我们刚刚描述的矩阵。第一个参数将是我们刚刚复制到 GPU 的矩阵数组。然后我们需要指定我们要使用 cuSolver 作为此操作的后端：

```py
U_d, s_d, V_d = linalg.svd(a_gpu,  lib='cusolver')
```

现在让我们将这些数组从 GPU 复制到主机：

```py
U = U_d.get()
s = s_d.get()
V = V_d.get()
```

`s`实际上存储为一维数组；我们将不得不创建一个大小为 1000 x 5000 的零矩阵，并将这些值沿对角线复制。我们可以使用 NumPy 的`diag`函数来做到这一点，再加上一些数组切片：

```py
S = np.zeros((1000,5000))
S[:1000,:1000] = np.diag(s)
```

我们现在可以在主机上使用 NumPy 的`dot`函数对这些值进行矩阵相乘，以验证它们是否与我们的原始数组匹配：

```py
print 'Can we reconstruct a from its SVD decomposition? : %s' % np.allclose(a, np.dot(U, np.dot(S, V)), atol=1e-5)
```

由于我们只使用 float32，并且我们的矩阵相对较大，引入了一些数值误差；我们不得不将“容差”级别（`atol`）设置得比平常高一点，但仍然足够小，以验证这两个数组是足够接近的。

# 使用 SVD 进行主成分分析（PCA）

**主成分分析**（**PCA**）是主要用于降维的工具。我们可以使用它来查看数据集，并找出哪些维度和线性子空间最显著。虽然有几种实现方法，但我们将向您展示如何使用 SVD 执行 PCA。

我们将这样做——我们将使用一个存在于 10 个维度中的数据集。我们将首先创建两个在前面有很大权重的向量，其他位置为 0：

```py
vals = [ np.float32([10,0,0,0,0,0,0,0,0,0]) , np.float32([0,10,0,0,0,0,0,0,0,0]) ]
```

然后我们将添加 9000 个额外的向量：其中 6000 个将与前两个向量相同，只是加了一点随机白噪声，剩下的 3000 个将只是随机白噪声：

```py
for i in range(3000):
    vals.append(vals[0] + 0.001*np.random.randn(10))
    vals.append(vals[1] + 0.001*np.random.randn(10))
    vals.append(0.001*np.random.randn(10))
```

我们现在将`vals`列表转换为`float32`的 NumPy 数组。我们对行进行平均，并从每行中减去这个值。（这是 PCA 的必要步骤。）然后我们转置这个矩阵，因为 cuSolver 要求输入矩阵的行数少于或等于列数：

```py
vals = np.float32(vals)
vals = vals - np.mean(vals, axis=0)
v_gpu = gpuarray.to_gpu(vals.T.copy())
```

我们现在将运行 cuSolver，就像我们之前做的那样，并将输出值从 GPU 复制出来：

```py
U_d, s_d, V_d = linalg.svd(v_gpu, lib='cusolver')

u = U_d.get()
s = s_d.get()
v = V_d.get()
```

现在我们准备开始我们的调查工作。让我们打开 IPython，仔细查看`u`和`s`。首先，让我们看看 s；它的值实际上是**主要值**的平方根，所以我们将它们平方然后看一看：

![](img/28321a31-a6fb-49e8-974f-1b2caecfe01b.png)

您会注意到，前两个主要值的数量级为 10⁵，而其余的分量的数量级为 10^(-3)。这告诉我们，实际上只有一个二维子空间与这些数据相关，这并不令人意外。这些是第一个和第二个值，它们将对应于第一个和第二个主成分，即相应的向量。让我们来看看这些向量，它们将存储在`U`中：

！[](assets/9d685877-5a4c-4449-8da4-68b1b21d1e66.png)

您会注意到这两个向量在前两个条目中有很大的权重，数量级为 10^(-1)；其余的条目都是 10^(-6)或更低，相对不相关。考虑到我们在前两个条目中使数据偏向，这正是我们应该期望的。这就是 PCA 背后的思想。

# 摘要

我们从查看如何使用 Scikit-CUDA 库的 cuBLAS 包装器开始了本章；在这里，我们必须记住许多细节，比如何使用列主存储，或者输入数组是否会被就地覆盖。然后，我们看了如何使用 Scikit-CUDA 的 cuFFT 执行一维和二维 FFT，以及如何创建一个简单的卷积滤波器。然后，我们向您展示了如何将其应用于图像的简单高斯模糊效果。最后，我们看了如何使用 cuSolver 在 GPU 上执行奇异值分解（SVD），这通常是一个非常计算密集的操作，但在 GPU 上可以很好地并行化。我们通过查看如何使用 SVD 进行基本的 PCA 来结束本章。

# 问题

1.  假设你得到了一个工作，需要将一些旧的遗留 FORTRAN BLAS 代码转换成 CUDA。你打开一个文件，看到一个名为 SBLAH 的函数，另一个名为 ZBLEH。你能在不查找的情况下告诉这两个函数使用的数据类型吗？

1.  你能修改 cuBLAS level-2 GEMV 示例，直接将矩阵`A`复制到 GPU，而不是在主机上进行转置以设置为列优先吗？

1.  使用 cuBLAS 32 位实数点积（`cublasSdot`）来实现使用一个按行矩阵和一个步幅为 1 的向量进行矩阵-向量乘法。

1.  使用`cublasSdot`实现矩阵-矩阵乘法。

1.  你能实现一种精确测量性能测量示例中的 GEMM 操作的方法吗？

1.  在一维 FFT 的示例中，尝试将`x`强制转换为`complex64`数组，然后将 FFT 和逆 FFT 计划都设置为`complex64`值。然后确认`np.allclose(x, x_gpu.get())`是否为真，而不检查数组的前一半。你认为为什么现在这样做会有效？

1.  请注意，在卷积示例中，模糊图像周围有一个暗边。为什么模糊图像中有这个暗边，而原始图像中没有呢？你能想到一种方法来减轻这个问题吗？
