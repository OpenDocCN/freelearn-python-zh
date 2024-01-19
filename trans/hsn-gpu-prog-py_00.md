# 前言

问候和祝福！本文是关于使用 Python 和 CUDA 进行 GPU 编程的入门指南。**GPU**可能代表**图形编程单元**，但我们应该明确，这本书*不是*关于图形编程——它本质上是**通用 GPU 编程**的介绍，简称为**GPGPU 编程**。在过去的十年中，已经清楚地表明 GPU 除了渲染图形之外，也非常适合进行计算，特别是需要大量计算吞吐量的并行计算。为此，NVIDIA 发布了 CUDA 工具包，这使得几乎任何具有一些 C 编程知识的人都可以更轻松地进入 GPGPU 编程的世界。

《使用 Python 和 CUDA 进行 GPU 编程实践》的目标是尽快让您进入 GPGPU 编程的世界。我们努力为每一章设计了有趣和有趣的示例和练习；特别是，我们鼓励您在阅读时在您喜欢的 Python 环境中键入这些示例并运行它们（Spyder、Jupyter 和 PyCharm 都是合适的选择）。这样，您最终将学会所有必要的函数和命令，并获得编写 GPGPU 程序的直觉。

最初，GPGPU 并行编程似乎非常复杂和令人望而生畏，特别是如果您过去只做过 CPU 编程。有很多新概念和约定您必须学习，可能会让您觉得自己从零开始。在这些时候，您必须相信学习这个领域的努力不是徒劳的。通过一点主动性和纪律性，当您阅读完本文时，这个主题将对您来说变得轻而易举。

愉快的编程！

# 这本书是为谁准备的

这本书特别针对一个人，那就是我自己，2014 年，当时我正在尝试为数学博士研究开发基于 GPU 的模拟。我正在研究多本关于 GPU 编程的书籍和手册，试图对这个领域有一点了解；大多数文本似乎乐意在每一页向读者抛出无尽的硬件原理和术语，而实际的*编程*则被放在了次要位置。

这本书主要面向那些想要实际进行 GPU 编程，但不想陷入技术细节和硬件原理的人。在本书中，我们将使用适当的 C/C++（CUDA C）来编程 GPU，但我们将通过 PyCUDA 模块将其*内联*到 Python 代码中。PyCUDA 允许我们只编写我们需要的必要低级 GPU 代码，同时自动处理编译、链接和在 GPU 上启动代码的所有冗余工作。

# 本书涵盖的内容

第一章《为什么要进行 GPU 编程？》给出了一些我们应该学习这个领域的动机，以及如何应用阿姆达尔定律来估计将串行程序转换为利用 GPU 的潜在性能改进。

第二章《设置 GPU 编程环境》解释了如何在 Windows 和 Linux 下设置适当的 Python 和 C++开发环境以进行 CUDA 编程。

第三章《使用 PyCUDA 入门》展示了我们在使用 Python 编程 GPU 时最基本的技能。我们将特别看到如何使用 PyCUDA 的 gpuarray 类将数据传输到 GPU 和从 GPU 传输数据，以及如何使用 PyCUDA 的 ElementwiseKernel 函数编译简单的 CUDA 核函数。

[第四章]（5a5f4317-50c7-4ce6-9d04-ac3be4c6d28b.xhtml），*核心，线程，块和网格*，教授了编写有效的 CUDA 核心的基础知识，这些核心是在 GPU 上启动的并行函数。我们将看到如何编写 CUDA 设备函数（由 CUDA 核心直接调用的“串行”函数），并了解 CUDA 的抽象网格/块结构及其在启动核心中的作用。

[第五章]（ea648e20-8c72-44a9-880d-11469d0e291f.xhtml），*流，事件，上下文和并发*，涵盖了 CUDA 流的概念，这是一种允许我们在 GPU 上同时启动和同步许多内核的功能。我们将看到如何使用 CUDA 事件来计时内核启动，以及如何创建和使用 CUDA 上下文。

[第六章]（6d1c808f-1dc2-4454-b0b8-d0a36bc3c908.xhtml），*调试和分析您的 CUDA 代码*，填补了我们在纯 CUDA C 编程方面的一些空白，并向我们展示了如何使用 NVIDIA Nsight IDE 进行调试和开发，以及如何使用 NVIDIA 分析工具。

[第七章]（55146879-4b7e-4774-9a8b-cc5c80c04ed8.xhtml），*使用 CUDA 库与 Scikit-CUDA*，通过 Python Scikit-CUDA 模块简要介绍了一些重要的标准 CUDA 库，包括 cuBLAS，cuFFT 和 cuSOLVER。

[第八章]（d374ea77-f9e5-4d38-861d-5295ef3e3fbf.xhtml），*CUDA 设备函数库和 Thrust*，向我们展示了如何在我们的代码中使用 cuRAND 和 CUDA Math API 库，以及如何使用 CUDA Thrust C++容器。

[第九章]（3562f1e0-a53d-470f-9b4d-94fa41b1b2fa.xhtml），*实现深度神经网络*，作为一个巅峰，我们将学习如何从头开始构建整个深度神经网络，应用我们在文本中学到的许多想法。

[第十章]（5383b46f-8dc6-4e17-ab35-7f6bd35f059f.xhtml），*使用已编译的 GPU 代码*，向我们展示了如何将我们的 Python 代码与预编译的 GPU 代码进行接口，使用 PyCUDA 和 Ctypes。

[第十一章]（e853faad-3ee4-4df7-9cdb-98f74e435527.xhtml），*CUDA 性能优化*，教授了一些非常低级的性能优化技巧，特别是与 CUDA 相关的技巧，例如 warp shuffling，矢量化内存访问，使用内联 PTX 汇编和原子操作。

[第十二章]（2d464c61-de29-49fa-826a-a7437c368d6a.xhtml），*从这里出发*，概述了您将拥有的一些教育和职业道路，这些道路将建立在您现在扎实的 GPU 编程基础之上。

# 为了充分利用本书

这实际上是一个非常技术性的主题。为此，我们将不得不对读者的编程背景做一些假设。为此，我们将假设以下内容：

+   您在 Python 中具有中级的编程经验。

+   您熟悉标准的 Python 科学包，如 NumPy，SciPy 和 Matplotlib。

+   您具有任何基于 C 的编程语言的中级能力（C，C ++，Java，Rust，Go 等）。

+   您了解 C 中的动态内存分配概念（特别是如何使用 C 的`malloc`和`free`函数）。

GPU 编程主要适用于非常科学或数学性质的领域，因此许多（如果不是大多数）示例将利用一些数学知识。因此，我们假设读者对大学一年级或二年级的数学有一定了解，包括：

+   三角学（正弦函数：sin，cos，tan…）

+   微积分（积分，导数，梯度）

+   统计学（均匀分布和正态分布）

+   线性代数（向量，矩阵，向量空间，维度）。

如果您还没有学习过这些主题，或者已经有一段时间了，不要担心，因为我们将在学习过程中尝试回顾一些关键的编程和数学概念。

我们在这里将做另一个假设。请记住，本文中我们只会使用 CUDA，这是 NVIDIA 硬件的专有编程语言。因此，在开始之前，我们需要拥有一些特定的硬件。因此，我假设读者可以访问以下内容：

+   64 位 x86 英特尔/AMD PC

+   4 GB 或更多的 RAM

+   入门级 NVIDIA GTX 1050 GPU（Pascal 架构）或更高版本

读者应该知道，大多数旧的 GPU 可能会在本文中的大多数示例中正常工作，但本文中的示例仅在 Windows 10 下的 GTX 1050 和 Linux 下的 GTX 1070 上进行了测试。有关设置和配置的具体说明在第二章中给出，*设置您的 GPU 编程环境*。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packt.com/support](http://www.packt.com/support)并注册，文件将直接发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packt.com](http://www.packt.com)。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA`](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有来自我们丰富书籍和视频目录的其他代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上找到。去看看吧！

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图像。您可以在此处下载：[`www.packtpub.com/sites/default/files/downloads/9781788993913_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781788993913_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。例如：“我们现在可以使用`cublasSaxpy`函数。”

代码块设置如下：

```py
cublas.cublasDestroy(handle)
print 'cuBLAS returned the correct value: %s' % np.allclose(np.dot(A,x), y_gpu.get())
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```py
def compute_gflops(precision='S'):

if precision=='S':
    float_type = 'float32'
elif precision=='D':
    float_type = 'float64'
else:
    return -1
```

任何命令行输入或输出都以以下方式编写：

```py
$ run cublas_gemm_flops.py
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以这种方式出现在文本中。

警告或重要说明会出现在这样的地方。提示和技巧会出现在这样的地方。
