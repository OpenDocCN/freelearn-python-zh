# 从这里往哪里走

这本书就像一次冒险的登山旅程一样……但现在，最终，我们已经到达了我们的徒步旅行的终点。我们现在站在介绍性GPU编程的山顶上，我们骄傲地回望我们的故乡串行编程村，微笑着想着我们旧的一维编程传统的天真，我们曾认为在Unix中*fork*一个进程就是我们对*并行编程*概念的全部理解。我们经历了许多险阻和危险才到达这一点，我们甚至可能犯了一些错误，比如在Linux中安装了一个损坏的NVIDIA驱动模块，或者在父母家度假时通过缓慢的100k连接下载了错误的Visual Studio版本。但这些挫折只是暂时的，留下的伤口变成了使我们更加强大对抗（GPU）自然力量的老茧。

然而，在我们的眼角，我们可以看到离我们站立的地方几米远处有两个木制标志；我们把目光从我们过去的小村庄移开，现在看着它们。第一个标志指向我们当前所面对的方向，上面只有一个词——过去。另一个指向相反的方向，也只有一个词——未来。我们转身朝着指向未来的方向走去，看到一个大而闪亮的大都市展现在我们面前，一直延伸到地平线，招手着我们。现在我们终于喘过气来，可以开始走向未来了…

在本章中，我们将介绍一些你现在可以继续学习GPU编程相关领域的选项。无论你是想要建立一个职业生涯，作为一个业余爱好者为了乐趣而做这个，作为一个工程学生为了课程而学习GPU，作为一个程序员或工程师试图增强你的技术背景，还是作为一个学术科学家试图将GPU应用到一个研究项目中，你现在都有很多选择。就像我们比喻的大都市一样，迷失其中很容易，很难确定我们应该去哪里。我们希望在这最后一章中提供类似于简短的导游，为你提供一些下一步可以去的选项。

我们现在将在本章中看一下以下路径：

+   高级CUDA和GPGPU编程

+   图形

+   机器学习和计算机视觉

+   区块链技术

# 进一步学习CUDA和GPGPU编程的知识

你首先可以选择的是，当然是学习更多关于CUDA和**通用GPU**（**GPGPU**）编程的知识。在这种情况下，你可能已经找到了一个很好的应用，并且想要编写更高级或优化的CUDA代码。你可能对它本身感兴趣，或者你想找一份CUDA/GPU程序员的工作。有了这本书提供的坚实的GPU编程基础，我们现在将看一些这个领域的高级主题，我们现在已经准备好学习了。

# 多GPU系统

首先想到的一个主要话题是学习如何为安装了多个GPU的系统编程。许多专业工作站和服务器都安装了多个GPU，目的是处理需要不止一个，而是几个顶级GPU的数据。为此，存在一个称为多GPU编程的子领域。其中大部分工作集中在负载平衡上，即使用每个GPU的最大容量，确保没有一个GPU被过多的工作饱和，而另一个则未被充分利用。另一个话题是Inter-GPU通信，通常涉及一个GPU直接使用CUDA的GPUDirect **点对点**（**P2P**）内存访问将内存数组复制到另一个GPU或从另一个GPU复制。

NVIDIA在这里提供了有关多GPU编程的简要介绍：[https://www.nvidia.com/docs/IO/116711/sc11-multi-gpu.pdf](https://www.nvidia.com/docs/IO/116711/sc11-multi-gpu.pdf)。

# 集群计算和MPI

另一个主题是集群计算，即编写程序，使其集体利用包含GPU的多台服务器。这些是*服务器农场*，它们在著名互联网公司（如Facebook和Google）的数据处理设施中，以及政府和军方使用的科学超级计算设施中广泛存在。集群通常使用一种称为**消息传递接口**（**MPI**）的编程范式，它是与诸如C++或Fortran之类的语言一起使用的接口，允许您编程连接到同一网络的许多计算机。

有关在MPI中使用CUDA的更多信息，请参阅此处：[https://devblogs.nvidia.com/introduction-cuda-aware-mpi/](https://devblogs.nvidia.com/introduction-cuda-aware-mpi/)。

# OpenCL和PyOpenCL

CUDA并不是唯一可以用来编程GPU的语言。CUDA最主要的竞争对手是称为开放计算语言（Open Computing Language）或OpenCL。CUDA是一个封闭的专有系统，只能在NVIDIA硬件上运行，而OpenCL是一个由非营利性Khronos Group开发和支持的开放标准。OpenCL不仅可以用于编程NVIDIA GPU，还可以用于AMD Radeon GPU甚至Intel HD GPU - 大多数主要技术公司都承诺在其产品中支持OpenCL。此外，PyCUDA的作者，UIUC的Andreas Kloeckner教授，还编写了另一个出色（且免费）的Python库，名为PyOpenCL，它提供了一个同样用户友好的OpenCL接口，几乎与PyCUDA具有相同的语法和概念。

有关OpenCL的信息由NVIDIA在这里提供：[https://developer.nvidia.com/opencl](https://developer.nvidia.com/opencl)。

Andreas Kloeckner的网站上提供了有关免费PyOpenCL库的信息：

[https://mathema.tician.de/software/pyopencl/](https://mathema.tician.de/software/pyopencl/)。

# 图形

显然，GPU中的G代表图形，而在本书中我们并没有看到太多关于图形的内容。尽管机器学习应用现在是NVIDIA的支柱产业，但一切都始于渲染出漂亮的图形。我们将在这里提供一些资源，让您开始学习，无论您是想开发视频游戏引擎、渲染CGI电影，还是开发CAD软件。CUDA实际上可以与图形应用程序并驾齐驱，并且实际上已经用于专业软件，如Adobe的Photoshop和After Effects，以及许多最近的视频游戏，如*Mafia*和*Just Cause*系列。我们将简要介绍一些您可能考虑从这里开始的主要API。

# OpenGL

OpenGL是一个行业开放标准，自90年代初就存在。虽然在某些方面它显得有些陈旧，但它是一个稳定的API，得到了广泛支持，如果你编写了一个使用它的程序，几乎可以保证在任何相对现代的GPU上都能运行。CUDA示例文件夹实际上包含了许多示例，说明了OpenGL如何与CUDA接口（特别是在`2_Graphics`子目录中），因此感兴趣的读者可以考虑查看这些示例。（在Windows中，默认位置为`C:\ProgramData\NVIDIA Corporation\CUDA Samples`，在Linux中为`/usr/local/cuda/samples`。）

有关OpenGL的信息可以直接从NVIDIA这里获取：[https://developer.nvidia.com/opengl](https://developer.nvidia.com/opengl)。

PyCUDA还提供了与NVIDIA OpenGL驱动程序的接口。有关信息，请访问此处：[https://documen.tician.de/pycuda/gl.html](https://documen.tician.de/pycuda/gl.html)。

# DirectX 12

DirectX 12是微软著名且得到良好支持的图形API的最新版本。虽然这是Windows PC和微软Xbox游戏机的专有，但这些系统显然拥有数亿用户的广泛安装基础。此外，除了NVIDIA显卡，Windows PC还支持各种GPU，并且Visual Studio IDE提供了很好的易用性。DirectX 12实际上支持低级GPGPU编程类型的概念，并且可以利用多个GPU。

微软的DirectX 12编程指南可以在这里找到：[https://docs.microsoft.com/en-us/windows/desktop/direct3d12/directx-12-programming-guide](https://docs.microsoft.com/en-us/windows/desktop/direct3d12/directx-12-programming-guide)。

# Vulkan

Vulkan可以被认为是DirectX 12的开放等效物，由Khronos Group开发，作为OpenGL的*next-gen*继任者。除了Windows，Vulkan也支持macOS和Linux，以及索尼PlayStation 4，任天堂Switch和Xbox One游戏机。Vulkan具有与DirectX 12相同的许多功能，例如准GPGPU编程。Vulkan对DirectX 12提供了一些严肃的竞争，例如2016年的《毁灭战士》重制版。

Khronos Group在这里提供了《Vulkan初学者指南》：[https://www.khronos.org/blog/beginners-guide-to-vulkan](https://www.khronos.org/blog/beginners-guide-to-vulkan)。

# 机器学习和计算机视觉

当然，本章的重点是机器学习及其兄弟计算机视觉。不用说，机器学习（特别是深度神经网络和卷积神经网络的子领域）是如今让NVIDIA首席执行官黄仁勋有饭吃的东西。（好吧，我们承认这是本十年的轻描淡写……）如果你需要提醒为什么GPU在这个领域如此适用和有用，请再看一下[第9章](3562f1e0-a53d-470f-9b4d-94fa41b1b2fa.xhtml)，*实现深度神经网络*。大量的并行计算和数学运算，以及用户友好的数学库，使NVIDIA GPU成为机器学习行业的硬件支柱。

# 基础知识

虽然现在你已经了解了低级GPU编程的许多复杂性，但你不会立即将这些知识应用到机器学习中。如果你在这个领域没有基本技能，比如如何对数据集进行基本统计分析，你真的应该停下来熟悉一下。斯坦福大学教授Andrew Ng，谷歌Brain的创始人，在网上和YouTube上提供了许多免费的材料。Ng教授的工作通常被认为是机器学习教育材料的金标准。

Ng教授在这里提供了一门免费的机器学习入门课程：[http://www.ml-class.org](http://www.ml-class.org)。

# cuDNN

NVIDIA提供了一个针对深度神经网络基元的优化GPU库，名为cuDNN。这些基元包括前向传播、卷积、反向传播、激活函数（如sigmoid、ReLU和tanh）和梯度下降。cuDNN是大多数主流深度神经网络框架（如Tensorflow）在NVIDIA GPU上的后端使用。这是NVIDIA免费提供的，但必须单独从CUDA Toolkit下载。

有关cuDNN的更多信息可以在这里找到：[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)。

# Tensorflow和Keras

Tensorflow当然是谷歌著名的神经网络框架。这是一个免费的开源框架，可用于Python和C++，自2015年以来一直向公众提供。

Tensorflow的教程可以在Google这里找到：[https://www.tensorflow.org/tutorials/](https://www.tensorflow.org/tutorials/)。

Keras是一个更高级的库，为Tensorflow提供了更*用户友好*的接口，最初由Google Brain的Francois Chollet编写。读者实际上可以考虑从Keras开始，然后再转向Tensorflow。

有关Keras的信息在这里：[https://keras.io/](https://keras.io/)。

# Chainer

Chainer是另一个神经网络API，由目前在日本东京大学攻读博士学位的Seiya Tokui开发。虽然它比Tensorflow更不为人知，但由于其令人难以置信的速度和效率而备受尊重。此外，读者可能会对Chainer特别感兴趣，因为最初是使用PyCUDA开发的。（后来改用了CuPy，这是一个为了提供更类似于NumPy的接口而开发的PyCUDA分支。）

有关Chainer的信息在这里：[https://chainer.org/](https://chainer.org/)。

# OpenCV

自2001年以来，开源计算机视觉库（OpenCV）一直存在。该库提供了许多来自经典计算机视觉和图像处理的工具，在深度神经网络时代仍然非常有用。近年来，OpenCV中的大多数算法都已移植到CUDA，并且与PyCUDA接口非常容易。

有关OpenCV的信息在这里：[https://opencv.org/](https://opencv.org/)。

# 区块链技术

最后但并非最不重要的是**区块链技术**。这是支持比特币和以太坊等加密货币的基础加密技术。这当然是一个非常新的领域，最初由比特币神秘创造者中本聪在2008年发表的白皮书中描述。在其发明后几乎立即应用了GPU——生成货币单位归结为蛮力破解加密谜题，而GPU可以并行尝试蛮力破解比一般公众今天可用的任何其他硬件更多的组合。这个过程被称为**挖矿**。

有兴趣了解区块链技术的人建议阅读中本聪关于比特币的原始白皮书，网址在这里：[https://bitcoin.org/bitcoin.pdf](https://bitcoin.org/bitcoin.pdf)。

GUIMiner是一个开源的基于CUDA的比特币矿工，网址在这里：[https://guiminer.org/](https://guiminer.org/)。

# 总结

在本章中，我们介绍了一些对于那些有兴趣进一步了解GPU编程背景的选项和路径，这超出了本书的范围。我们介绍的第一条路径是扩展您在纯CUDA和GPGPU编程方面的背景——您可以学习一些本书未涵盖的内容，包括使用多个GPU和网络集群的编程系统。我们还讨论了除CUDA之外的一些并行编程语言/API，如MPI和OpenCL。接下来，我们讨论了一些对于有兴趣将GPU应用于渲染图形的人来说非常知名的API，如Vulkan和DirectX 12。然后，我们研究了机器学习，并深入了解了一些您应该具备的基本背景以及一些用于开发深度神经网络的主要框架。最后，我们简要介绍了区块链技术和基于GPU的加密货币挖矿。

作为作者，我想对每个人都说*谢谢*，感谢你们阅读到这里，到了结尾。GPU编程是我遇到的最棘手的编程子领域之一，我希望我的文字能帮助你掌握基本要点。作为读者，你现在应该可以放纵自己，享用一块你能找到的最丰富、最高热量的巧克力蛋糕——只要知道你*赚*了它。（但只能吃一块！）

# 问题

1.  使用谷歌或其他搜索引擎找到至少一个未在本章中介绍的GPU编程应用。

1.  尝试找到至少一种编程语言或API，可以用来编程GPU，而这在本章中没有提到。

1.  查找谷歌的新张量处理单元（TPU）芯片。这些与GPU有何不同？

1.  你认为使用Wi-Fi还是有线以太网电缆将计算机连接成集群是更好的主意？
