# 第十二章：从这里往哪里走

这本书就像一次冒险的登山旅程一样……但现在，最终，我们已经到达了我们的徒步旅行的终点。我们现在站在介绍性 GPU 编程的山顶上，我们骄傲地回望我们的故乡串行编程村，微笑着想着我们旧的一维编程传统的天真，我们曾认为在 Unix 中*fork*一个进程就是我们对*并行编程*概念的全部理解。我们经历了许多险阻和危险才到达这一点，我们甚至可能犯了一些错误，比如在 Linux 中安装了一个损坏的 NVIDIA 驱动模块，或者在父母家度假时通过缓慢的 100k 连接下载了错误的 Visual Studio 版本。但这些挫折只是暂时的，留下的伤口变成了使我们更加强大对抗（GPU）自然力量的老茧。

然而，在我们的眼角，我们可以看到离我们站立的地方几米远处有两个木制标志；我们把目光从我们过去的小村庄移开，现在看着它们。第一个标志指向我们当前所面对的方向，上面只有一个词——过去。另一个指向相反的方向，也只有一个词——未来。我们转身朝着指向未来的方向走去，看到一个大而闪亮的大都市展现在我们面前，一直延伸到地平线，招手着我们。现在我们终于喘过气来，可以开始走向未来了…

在本章中，我们将介绍一些你现在可以继续学习 GPU 编程相关领域的选项。无论你是想要建立一个职业生涯，作为一个业余爱好者为了乐趣而做这个，作为一个工程学生为了课程而学习 GPU，作为一个程序员或工程师试图增强你的技术背景，还是作为一个学术科学家试图将 GPU 应用到一个研究项目中，你现在都有很多选择。就像我们比喻的大都市一样，迷失其中很容易，很难确定我们应该去哪里。我们希望在这最后一章中提供类似于简短的导游，为你提供一些下一步可以去的选项。

我们现在将在本章中看一下以下路径：

+   高级 CUDA 和 GPGPU 编程

+   图形

+   机器学习和计算机视觉

+   区块链技术

# 进一步学习 CUDA 和 GPGPU 编程的知识

你首先可以选择的是，当然是学习更多关于 CUDA 和**通用 GPU**（**GPGPU**）编程的知识。在这种情况下，你可能已经找到了一个很好的应用，并且想要编写更高级或优化的 CUDA 代码。你可能对它本身感兴趣，或者你想找一份 CUDA/GPU 程序员的工作。有了这本书提供的坚实的 GPU 编程基础，我们现在将看一些这个领域的高级主题，我们现在已经准备好学习了。

# 多 GPU 系统

首先想到的一个主要话题是学习如何为安装了多个 GPU 的系统编程。许多专业工作站和服务器都安装了多个 GPU，目的是处理需要不止一个，而是几个顶级 GPU 的数据。为此，存在一个称为多 GPU 编程的子领域。其中大部分工作集中在负载平衡上，即使用每个 GPU 的最大容量，确保没有一个 GPU 被过多的工作饱和，而另一个则未被充分利用。另一个话题是 Inter-GPU 通信，通常涉及一个 GPU 直接使用 CUDA 的 GPUDirect **点对点**（**P2P**）内存访问将内存数组复制到另一个 GPU 或从另一个 GPU 复制。

NVIDIA 在这里提供了有关多 GPU 编程的简要介绍：[`www.nvidia.com/docs/IO/116711/sc11-multi-gpu.pdf`](https://www.nvidia.com/docs/IO/116711/sc11-multi-gpu.pdf)。

# 集群计算和 MPI

另一个主题是集群计算，即编写程序，使其集体利用包含 GPU 的多台服务器。这些是*服务器农场*，它们在著名互联网公司（如 Facebook 和 Google）的数据处理设施中，以及政府和军方使用的科学超级计算设施中广泛存在。集群通常使用一种称为**消息传递接口**（**MPI**）的编程范式，它是与诸如 C++或 Fortran 之类的语言一起使用的接口，允许您编程连接到同一网络的许多计算机。

有关在 MPI 中使用 CUDA 的更多信息，请参阅此处：[`devblogs.nvidia.com/introduction-cuda-aware-mpi/`](https://devblogs.nvidia.com/introduction-cuda-aware-mpi/)。

# OpenCL 和 PyOpenCL

CUDA 并不是唯一可以用来编程 GPU 的语言。CUDA 最主要的竞争对手是称为开放计算语言（Open Computing Language）或 OpenCL。CUDA 是一个封闭的专有系统，只能在 NVIDIA 硬件上运行，而 OpenCL 是一个由非营利性 Khronos Group 开发和支持的开放标准。OpenCL 不仅可以用于编程 NVIDIA GPU，还可以用于 AMD Radeon GPU 甚至 Intel HD GPU - 大多数主要技术公司都承诺在其产品中支持 OpenCL。此外，PyCUDA 的作者，UIUC 的 Andreas Kloeckner 教授，还编写了另一个出色（且免费）的 Python 库，名为 PyOpenCL，它提供了一个同样用户友好的 OpenCL 接口，几乎与 PyCUDA 具有相同的语法和概念。

有关 OpenCL 的信息由 NVIDIA 在这里提供：[`developer.nvidia.com/opencl`](https://developer.nvidia.com/opencl)。

Andreas Kloeckner 的网站上提供了有关免费 PyOpenCL 库的信息：

[`mathema.tician.de/software/pyopencl/`](https://mathema.tician.de/software/pyopencl/)。

# 图形

显然，GPU 中的 G 代表图形，而在本书中我们并没有看到太多关于图形的内容。尽管机器学习应用现在是 NVIDIA 的支柱产业，但一切都始于渲染出漂亮的图形。我们将在这里提供一些资源，让您开始学习，无论您是想开发视频游戏引擎、渲染 CGI 电影，还是开发 CAD 软件。CUDA 实际上可以与图形应用程序并驾齐驱，并且实际上已经用于专业软件，如 Adobe 的 Photoshop 和 After Effects，以及许多最近的视频游戏，如*Mafia*和*Just Cause*系列。我们将简要介绍一些您可能考虑从这里开始的主要 API。

# OpenGL

OpenGL 是一个行业开放标准，自 90 年代初就存在。虽然在某些方面它显得有些陈旧，但它是一个稳定的 API，得到了广泛支持，如果你编写了一个使用它的程序，几乎可以保证在任何相对现代的 GPU 上都能运行。CUDA 示例文件夹实际上包含了许多示例，说明了 OpenGL 如何与 CUDA 接口（特别是在`2_Graphics`子目录中），因此感兴趣的读者可以考虑查看这些示例。（在 Windows 中，默认位置为`C:\ProgramData\NVIDIA Corporation\CUDA Samples`，在 Linux 中为`/usr/local/cuda/samples`。）

有关 OpenGL 的信息可以直接从 NVIDIA 这里获取：[`developer.nvidia.com/opengl`](https://developer.nvidia.com/opengl)。

PyCUDA 还提供了与 NVIDIA OpenGL 驱动程序的接口。有关信息，请访问此处：[`documen.tician.de/pycuda/gl.html`](https://documen.tician.de/pycuda/gl.html)。

# DirectX 12

DirectX 12 是微软著名且得到良好支持的图形 API 的最新版本。虽然这是 Windows PC 和微软 Xbox 游戏机的专有，但这些系统显然拥有数亿用户的广泛安装基础。此外，除了 NVIDIA 显卡，Windows PC 还支持各种 GPU，并且 Visual Studio IDE 提供了很好的易用性。DirectX 12 实际上支持低级 GPGPU 编程类型的概念，并且可以利用多个 GPU。

微软的 DirectX 12 编程指南可以在这里找到：[`docs.microsoft.com/en-us/windows/desktop/direct3d12/directx-12-programming-guide`](https://docs.microsoft.com/en-us/windows/desktop/direct3d12/directx-12-programming-guide)。

# Vulkan

Vulkan 可以被认为是 DirectX 12 的开放等效物，由 Khronos Group 开发，作为 OpenGL 的*next-gen*继任者。除了 Windows，Vulkan 也支持 macOS 和 Linux，以及索尼 PlayStation 4，任天堂 Switch 和 Xbox One 游戏机。Vulkan 具有与 DirectX 12 相同的许多功能，例如准 GPGPU 编程。Vulkan 对 DirectX 12 提供了一些严肃的竞争，例如 2016 年的《毁灭战士》重制版。

Khronos Group 在这里提供了《Vulkan 初学者指南》：[`www.khronos.org/blog/beginners-guide-to-vulkan`](https://www.khronos.org/blog/beginners-guide-to-vulkan)。

# 机器学习和计算机视觉

当然，本章的重点是机器学习及其兄弟计算机视觉。不用说，机器学习（特别是深度神经网络和卷积神经网络的子领域）是如今让 NVIDIA 首席执行官黄仁勋有饭吃的东西。（好吧，我们承认这是本十年的轻描淡写……）如果你需要提醒为什么 GPU 在这个领域如此适用和有用，请再看一下第九章，*实现深度神经网络*。大量的并行计算和数学运算，以及用户友好的数学库，使 NVIDIA GPU 成为机器学习行业的硬件支柱。

# 基础知识

虽然现在你已经了解了低级 GPU 编程的许多复杂性，但你不会立即将这些知识应用到机器学习中。如果你在这个领域没有基本技能，比如如何对数据集进行基本统计分析，你真的应该停下来熟悉一下。斯坦福大学教授 Andrew Ng，谷歌 Brain 的创始人，在网上和 YouTube 上提供了许多免费的材料。Ng 教授的工作通常被认为是机器学习教育材料的金标准。

Ng 教授在这里提供了一门免费的机器学习入门课程：[`www.ml-class.org`](http://www.ml-class.org)。

# cuDNN

NVIDIA 提供了一个针对深度神经网络基元的优化 GPU 库，名为 cuDNN。这些基元包括前向传播、卷积、反向传播、激活函数（如 sigmoid、ReLU 和 tanh）和梯度下降。cuDNN 是大多数主流深度神经网络框架（如 Tensorflow）在 NVIDIA GPU 上的后端使用。这是 NVIDIA 免费提供的，但必须单独从 CUDA Toolkit 下载。

有关 cuDNN 的更多信息可以在这里找到：[`developer.nvidia.com/cudnn`](https://developer.nvidia.com/cudnn)。

# Tensorflow 和 Keras

Tensorflow 当然是谷歌著名的神经网络框架。这是一个免费的开源框架，可用于 Python 和 C++，自 2015 年以来一直向公众提供。

Tensorflow 的教程可以在 Google 这里找到：[`www.tensorflow.org/tutorials/`](https://www.tensorflow.org/tutorials/)。

Keras 是一个更高级的库，为 Tensorflow 提供了更*用户友好*的接口，最初由 Google Brain 的 Francois Chollet 编写。读者实际上可以考虑从 Keras 开始，然后再转向 Tensorflow。

有关 Keras 的信息在这里：[`keras.io/`](https://keras.io/)。

# Chainer

Chainer 是另一个神经网络 API，由目前在日本东京大学攻读博士学位的 Seiya Tokui 开发。虽然它比 Tensorflow 更不为人知，但由于其令人难以置信的速度和效率而备受尊重。此外，读者可能会对 Chainer 特别感兴趣，因为最初是使用 PyCUDA 开发的。（后来改用了 CuPy，这是一个为了提供更类似于 NumPy 的接口而开发的 PyCUDA 分支。）

有关 Chainer 的信息在这里：[`chainer.org/`](https://chainer.org/)。

# OpenCV

自 2001 年以来，开源计算机视觉库（OpenCV）一直存在。该库提供了许多来自经典计算机视觉和图像处理的工具，在深度神经网络时代仍然非常有用。近年来，OpenCV 中的大多数算法都已移植到 CUDA，并且与 PyCUDA 接口非常容易。

有关 OpenCV 的信息在这里：[`opencv.org/`](https://opencv.org/)。

# 区块链技术

最后但并非最不重要的是**区块链技术**。这是支持比特币和以太坊等加密货币的基础加密技术。这当然是一个非常新的领域，最初由比特币神秘创造者中本聪在 2008 年发表的白皮书中描述。在其发明后几乎立即应用了 GPU——生成货币单位归结为蛮力破解加密谜题，而 GPU 可以并行尝试蛮力破解比一般公众今天可用的任何其他硬件更多的组合。这个过程被称为**挖矿**。

有兴趣了解区块链技术的人建议阅读中本聪关于比特币的原始白皮书，网址在这里：[`bitcoin.org/bitcoin.pdf`](https://bitcoin.org/bitcoin.pdf)。

GUIMiner 是一个开源的基于 CUDA 的比特币矿工，网址在这里：[`guiminer.org/`](https://guiminer.org/)。

# 总结

在本章中，我们介绍了一些对于那些有兴趣进一步了解 GPU 编程背景的选项和路径，这超出了本书的范围。我们介绍的第一条路径是扩展您在纯 CUDA 和 GPGPU 编程方面的背景——您可以学习一些本书未涵盖的内容，包括使用多个 GPU 和网络集群的编程系统。我们还讨论了除 CUDA 之外的一些并行编程语言/API，如 MPI 和 OpenCL。接下来，我们讨论了一些对于有兴趣将 GPU 应用于渲染图形的人来说非常知名的 API，如 Vulkan 和 DirectX 12。然后，我们研究了机器学习，并深入了解了一些您应该具备的基本背景以及一些用于开发深度神经网络的主要框架。最后，我们简要介绍了区块链技术和基于 GPU 的加密货币挖矿。

作为作者，我想对每个人都说*谢谢*，感谢你们阅读到这里，到了结尾。GPU 编程是我遇到的最棘手的编程子领域之一，我希望我的文字能帮助你掌握基本要点。作为读者，你现在应该可以放纵自己，享用一块你能找到的最丰富、最高热量的巧克力蛋糕——只要知道你*赚*了它。（但只能吃一块！）

# 问题

1.  使用谷歌或其他搜索引擎找到至少一个未在本章中介绍的 GPU 编程应用。

1.  尝试找到至少一种编程语言或 API，可以用来编程 GPU，而这在本章中没有提到。

1.  查找谷歌的新张量处理单元（TPU）芯片。这些与 GPU 有何不同？

1.  你认为使用 Wi-Fi 还是有线以太网电缆将计算机连接成集群是更好的主意？
