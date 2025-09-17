# 第九章：为高性能设计

在前面的章节中，我们学习了如何使用 Python 标准库和第三方包中可用的各种工具来评估和改进 Python 应用程序的性能。在本章中，我们将提供一些关于如何处理不同类型应用程序的一般性指南，并展示一些被多个 Python 项目普遍采用的优秀实践。

在本章中，我们将学习以下内容：

+   为通用、数值计算和大数据应用选择正确的性能技术

+   结构化 Python 项目

+   使用虚拟环境和容器化隔离 Python 安装

+   使用 Travis CI 设置持续集成

# 选择合适的策略

许多软件包可用于提高程序的性能，但我们如何确定我们程序的最佳优化策略？多种因素决定了使用哪种方法的决策。在本节中，我们将尽可能全面地回答这个问题，基于广泛的应用类别。

首先要考虑的是应用程序的类型。Python 是一种服务于多个非常多样化的社区的语言，这些社区包括网络服务、系统脚本、游戏、机器学习等等。这些不同的应用程序将需要对程序的不同部分进行优化努力。

例如，一个网络服务可以被优化以拥有非常短的反应时间。同时，它必须能够尽可能少地使用资源来回答尽可能多的请求（也就是说，它将尝试实现更低的延迟），而数值代码可能需要几周时间才能运行。即使有显著的启动开销（在这种情况下，我们感兴趣的是吞吐量），提高系统可能处理的数据量也很重要。

另一个方面是我们正在开发的平台和架构。虽然 Python 支持许多平台和架构，但许多第三方库可能对某些平台的支持有限，尤其是在处理绑定到 C 扩展的包时。因此，有必要检查目标平台和架构上库的可用性。

此外，一些架构，如嵌入式系统和小型设备，可能存在严重的 CPU 和内存限制。这是一个需要考虑的重要因素，例如，一些技术（如多进程）可能会消耗太多内存或需要执行额外的软件。

最后，业务需求同样重要。很多时候，软件产品需要快速迭代和快速更改代码的能力。一般来说，您希望将软件栈保持尽可能简单，以便修改、测试、部署以及引入额外的平台支持在短时间内变得容易且可行。这也适用于团队——安装软件栈和开始开发应该尽可能顺利。因此，通常应优先选择纯 Python 库而不是扩展，除非是经过良好测试的库，如 NumPy。此外，各种业务方面将有助于确定哪些操作需要首先优化（始终记住，*过早优化是万恶之源*）。

# 通用应用程序

通用应用程序，例如 Web 应用程序或移动应用程序后端，通常涉及对远程服务和数据库的调用。对于此类情况，利用异步框架可能很有用，例如在第六章中介绍的框架，*实现并发*；这将提高应用程序逻辑、系统设计、响应性，并且，它还将简化网络故障的处理。

异步编程的使用也使得实现和使用微服务变得更加容易。虽然没有标准定义，但可以将**微服务**视为专注于应用程序特定方面（例如，认证）的远程服务。

微服务的理念是您可以通过组合通过简单协议（例如 gRPC、REST 调用或通过专用消息队列）通信的不同微服务来构建应用程序。这种架构与所有服务都由同一 Python 进程处理的单体应用程序形成对比。

微服务的优势包括应用程序不同部分之间的强解耦。小型、简单的服务可以由不同的团队实现和维护，并且可以在不同时间更新和部署。这也使得微服务可以轻松复制，以便处理更多用户。此外，由于通信是通过简单的协议进行的，因此微服务可以用更适合特定应用程序的语言实现。

如果服务的性能不满意，应用程序通常可以在不同的 Python 解释器上执行，例如 PyPy（前提是所有第三方扩展都兼容）以实现足够的速度提升。否则，算法策略以及将瓶颈迁移到 Cython 通常足以实现令人满意的表现。

# 数值代码

如果你的目标是编写数值代码，一个很好的策略是直接从 NumPy 实现开始。使用 NumPy 是一个安全的赌注，因为它在许多平台上都可用且经过测试，并且，如我们在前面的章节中看到的，许多其他包将 NumPy 数组视为一等公民。

当正确编写时（例如，通过利用我们在第二章中学习的广播和其他技术，*纯 Python 优化*），NumPy 的性能已经非常接近由 C 代码实现的本地性能，并且不需要进一步优化。尽管如此，某些算法使用 NumPy 的数据结构和方法难以高效表达。当这种情况发生时，两个非常好的选择可以是 Numba 或 Cython。

Cython 是一个非常成熟的工具，被许多重要项目广泛使用，例如 `scipy` 和 `scikit-learn`。Cython 代码通过其明确的静态类型声明，使其非常易于理解，大多数 Python 程序员都不会在掌握其熟悉的语法上遇到问题。此外，没有“魔法”和良好的检查工具使得程序员可以轻松预测其性能，并对如何进行更改以实现最佳性能做出有根据的猜测。

然而，Cython 也有一些缺点。Cython 代码在执行之前需要编译，这打破了 Python 编辑-运行周期的便利性。这也要求目标平台上有兼容的 C 编译器。这还使得分发和部署变得复杂，因为需要为每个目标平台测试多个平台、架构、配置和编译器。

另一方面，Numba API 只需要定义纯 Python 函数，这些函数会即时编译，保持快速的 Python 编辑-运行周期。一般来说，Numba 需要在目标平台上安装 LLVM 工具链。请注意，截至版本 0.30，Numba 函数的**即时编译**（**AOT**）有一些有限的支持，这样 Numba 编译的函数就可以打包和部署，而无需安装 Numba 和 LLVM。

注意，Numba 和 Cython 通常都预包装了所有依赖项（包括编译器），可以在 conda 包管理器的默认通道中找到。因此，在 conda 包管理器可用的平台上，Cython 的部署可以大大简化。

如果 Cython 和 Numba 仍然不够用怎么办？虽然这通常不是必需的，但另一种策略是实现一个纯 C 模块（可以使用编译器标志或手动调整进行进一步优化），然后使用 `cffi` 包（[`cffi.readthedocs.io/en/latest/`](https://cffi.readthedocs.io/en/latest/)) 或 Cython 从 Python 模块中调用它。

使用 NumPy、Numba 和 Cython 是在串行代码上获得近似最优性能的有效策略。对于许多应用来说，串行代码当然足够了，即使最终计划是拥有并行算法，仍然非常值得为调试目的而工作在串行参考实现上，因为串行实现在小数据集上可能更快。

并行实现根据特定应用的复杂性有很大差异。在许多情况下，程序可以很容易地表示为一系列独立的计算，随后进行某种形式的*聚合*，并可以使用简单的基于进程的接口进行并行化，例如 `multiprocessing.Pool` 或 `ProcessPoolExecutor`，这些接口的优点是能够在不费太多周折的情况下并行执行通用 Python 代码。

为了避免启动多个进程的时间和内存开销，可以使用线程。NumPy 函数通常释放 GIL，是线程并行化的良好候选者。此外，Cython 和 Numba 提供特殊的 `nogil` 语句以及自动并行化，这使得它们适合简单的、轻量级的并行化。

对于更复杂的用例，你可能需要显著改变算法。在这些情况下，Dask 数组是一个不错的选择，它们几乎可以无缝替换标准的 NumPy。Dask 的进一步优势是操作非常透明，并且易于调整。

专门的应用程序，如深度学习和计算机图形学，它们大量使用线性代数例程，可能从 Theano 和 Tensorflow 等软件包中受益，这些软件包能够实现高度性能的自动并行化，并内置 GPU 支持。

最后，可以使用 `mpi4py` 在基于 MPI 的超级计算机上部署并行 Python 脚本（通常大学的研究人员可以访问）。

# 大数据

大型数据集（通常大于 1 TB）变得越来越普遍，大量资源已经投入到了开发能够收集、存储和分析这些数据的技术中。通常，选择使用哪个框架取决于数据最初是如何存储的。

许多时候，即使完整的数据集不适合单个机器，仍然可以制定策略来提取答案，而无需探测整个数据集。例如，经常可以通过提取一个小的、有趣的数据子集来回答问题，这些数据子集可以轻松加载到内存中，并使用高度方便和高效的库（如 Pandas）进行分析。通过过滤或随机采样数据点，通常可以找到足够好的答案来回答业务问题，而无需求助于大数据工具。

如果公司的软件主体是用 Python 编写的，并且你有自由选择软件栈的权限，那么使用 Dask 分布式是有意义的。这个软件包的设置非常简单，并且与 Python 生态系统紧密集成。使用 Dask 的 `array` 和 `DataFrame`，通过适配 NumPy 和 Pandas 代码，可以非常容易地扩展你现有的 Python 算法。

很常见，一些公司可能已经设置了一个 Spark 集群。在这种情况下，PySpark 是最佳选择，并且鼓励使用 SparkSQL 以获得更高的性能。Spark 的一个优点是它允许使用其他语言，例如 Scala 和 Java。

# 组织你的源代码

典型的 Python 项目的仓库结构至少包括一个包含 `README.md` 文件的目录、一个包含应用程序或库源代码的 Python 模块或包，以及一个 `setup.py` 文件。项目可能采用不同的约定来遵守公司政策或使用的特定框架。在本节中，我们将说明一些在社区驱动的 Python 项目中常见的实践，这些实践可能包括我们在前面章节中介绍的一些工具。

一个名为 `myapp` 的 Python 项目的典型目录结构可以看起来像这样。现在，我们将阐述每个文件和目录的作用：

```py
    myapp/ 
      README.md
      LICENSE
      setup.py
      myapp/
        __init__.py
        module1.py
        cmodule1.pyx
        module2/
           __init__.py
      src/
        module.c
        module.h
      tests/
        __init__.py
        test_module1.py
        test_module2.py
      benchmarks/
        __init__.py
        test_module1.py
        test_module2.py
      docs/
      tools/

```

`README.md` 是一个包含有关软件的一般信息的文本文件，例如项目范围、安装、快速入门和有用的链接。如果软件公开发布，则使用 `LICENSE` 文件来指定其使用的条款和条件。

Python 软件通常使用 `setup.py` 文件中的 `setuptools` 库进行打包。正如我们在前面的章节中看到的，`setup.py` 也是编译和分发 Cython 代码的有效方式。

`myapp` 包包含应用程序的源代码，包括 Cython 模块。有时，除了它们的 Cython 优化版本之外，维护纯 Python 实现也很方便。通常，模块的 Cython 版本以 c 前缀命名（例如，前一个示例中的 `cmodule1.pyx`）。

如果需要外部 `.c` 和 `.h` 文件，这些文件通常存储在顶级（`myapp`）项目目录下的一个额外的 `src/` 目录中。

`tests/` 目录包含应用程序的测试代码（通常以单元测试的形式），可以使用测试运行器（如 `unittest` 或 `pytest`）运行。然而，一些项目更喜欢将 `tests/` 目录放置在 `myapp` 包内部。由于高性能代码是持续调整和重写的，因此拥有一个可靠的测试套件对于尽早发现错误和通过缩短测试-编辑-运行周期来提高开发者体验至关重要。

基准测试可以放在`benchmarks`目录中；将基准测试与测试分离的优势在于，基准测试可能需要更多的时间来执行。基准测试也可以在构建服务器上运行（见*持续集成*部分），作为一种简单的方法来比较不同版本的性能。虽然基准测试通常比单元测试运行时间更长，但最好将它们的执行时间尽可能缩短，以避免资源浪费。

最后，`docs/`目录包含用户和开发者文档以及 API 参考。这通常还包括文档工具的配置文件，例如`sphinx`。其他工具和脚本可以放在`tools/`目录中。

# 隔离、虚拟环境和容器

在代码测试和执行时拥有隔离环境的重要性，通过观察当你请求朋友运行你的一个 Python 脚本时会发生什么，就会变得非常明显。发生的情况是，你提供安装 Python 版本 X 及其依赖包`Y`、`X`的指令，并要求他们在自己的机器上复制并执行该脚本。

在许多情况下，你的朋友会继续操作，为其平台下载 Python 以及依赖库，并尝试执行脚本。然而，可能会发生（很多时候都会发生）脚本会失败，因为他们的计算机操作系统与你不同，或者安装的库版本与你机器上安装的版本不同。在其他时候，可能会有不正确删除的先前安装，这会导致难以调试的冲突和很多挫败感。

避免这种情况的一个非常简单的方法是使用虚拟环境。虚拟环境通过隔离 Python、相关可执行文件和第三方包来创建和管理多个 Python 安装。自 Python 3.3 版本以来，标准库包括了`venv`模块（之前称为**virtualenv**），这是一个用于创建和管理简单隔离环境的工具。基于`venv`的虚拟环境中的 Python 包可以使用`setup.py`文件或通过`pip`进行安装。

在处理高性能代码时，提供精确和具体的库版本至关重要。库在发布之间不断进化，算法的变化可能会显著影响性能。例如，流行的库，如`scipy`和`scikit-learn`，通常会将其部分代码和数据结构移植到 Cython，因此用户安装正确的版本以获得最佳性能非常重要。

# 使用 conda 环境

大多数情况下，使用`venv`是一个不错的选择。然而，当编写高性能代码时，通常会发生一些高性能库也需要安装非 Python 软件的情况。这通常涉及额外的设置编译器和高性能本地库（在 C、C++或 Fortran 中），Python 软件包会链接到这些库。由于`venv`和`pip`旨在仅处理 Python 软件包，因此这些工具对这种场景的支持不佳。

`conda`软件包管理器是专门为处理此类情况而创建的。可以使用`conda create`命令创建虚拟环境。该命令接受`-n`参数（`-n`代表`--name`），用于指定新创建的环境和我们要安装的软件包。如果我们想创建一个使用 Python 版本`3.5`和最新版 NumPy 的环境，我们可以使用以下命令：

```py
$ conda create -n myenv Python=3.5 numpy

```

Conda 将负责从其存储库获取相关软件包并将它们放置在隔离的 Python 安装中。要启用虚拟环境，可以使用`source activate`命令：

```py
$ source activate myenv

```

执行此命令后，默认 Python 解释器将切换到我们之前指定的版本。你可以使用`which`命令轻松验证 Python 可执行文件的位置，该命令返回可执行文件的全路径：

```py
(myenv) $ which python
/home/gabriele/anaconda/envs/myenv/bin/python

```

到目前为止，你可以自由地在虚拟环境中添加、删除和修改软件包，而不会影响全局 Python 安装。可以使用`conda install <package name>`命令或通过`pip`安装更多软件包。

虚拟环境的优点在于，你可以以良好的隔离方式安装或编译你想要的任何软件。这意味着，如果由于某种原因，你的环境被损坏，你可以将其擦除并从头开始。

要删除`myenv`环境，首先需要将其停用，然后使用`conda env remove`命令，如下所示：

```py
(myenv) $ source deactivate
$ conda env remove -n myenv

```

如果软件包在标准`conda`存储库中不可用怎么办？一个选项是查看它是否在`conda-forge`社区频道中可用。要搜索`conda-forge`中的软件包，可以在`conda search`命令中添加`-c`选项（代表`--channel`）：

```py
$ conda search -c conda-forge scipy

```

该命令将列出与`scipy`查询字符串匹配的一系列软件包和版本。另一个选项是在**Anaconda Cloud**上托管公共频道中搜索该软件包。可以通过安装`anaconda-client`软件包来下载 Anaconda Cloud 的命令行客户端：

```py
$ conda install anaconda-client

```

客户端安装完成后，你可以使用`anaconda`命令行客户端来搜索软件包。在以下示例中，我们演示了如何查找`chemview`软件包：

```py
$ anaconda search chemview 
Using Anaconda API: https://api.anaconda.org
Run 'anaconda show <USER/PACKAGE>' to get more details:
Packages:
 Name                      | Version | Package Types   | Platforms 
 ------------------------- | ------  | --------------- | ---------------
 cjs14/chemview            | 0.3     | conda           | linux-64, win-64, osx-64
 : WebGL Molecular Viewer for IPython notebook.
 gabrielelanaro/chemview   | 0.7     | conda           | linux-64, osx-64
 : WebGL Molecular Viewer for IPython notebook. 

```

然后，可以通过指定适当的频道（使用`-c`选项）轻松执行安装：

```py
$ conda install -c gabrielelanaro chemlab

```

# 虚拟化和容器

虚拟化作为一种在同一台机器上运行多个操作系统以更好地利用物理资源的方法，已经存在很长时间了。

实现虚拟化的一种方法是通过使用*虚拟机*。虚拟机通过创建虚拟硬件资源，例如 CPU、内存和设备，并使用这些资源在同一台机器上安装和运行多个操作系统。通过在操作系统（称为*宿主*）上安装虚拟化软件（称为*管理程序*），可以实现虚拟化。管理程序能够创建、管理和监控虚拟机及其相应的操作系统（称为*客户机*）。

重要的是要注意，尽管名为虚拟环境，但它们与虚拟机无关。虚拟环境是 Python 特定的，通过 shell 脚本设置不同的 Python 解释器来实现。

容器是一种通过创建与宿主操作系统分离的环境并仅包含必要依赖项来隔离应用程序的方法。容器是操作系统功能，允许您共享由操作系统内核提供的硬件资源（多个实例）。与虚拟机不同，容器并不抽象硬件资源，而只是共享操作系统的内核。

容器在利用硬件资源方面非常高效，因为它们通过内核直接访问。因此，它们是高性能应用的绝佳解决方案。它们也易于创建和销毁，可以快速在隔离环境中测试应用程序。容器还用于简化部署（尤其是微服务）以及开发构建服务器，如前文所述。

在第八章“分布式处理”中，我们使用了**docker**来轻松设置 PySpark 安装。Docker 是目前最受欢迎的容器化解决方案之一。安装 Docker 的最佳方式是遵循官方网站上的说明（[`www.docker.com/`](https://www.docker.com/)）。安装后，可以使用 docker 命令行界面轻松创建和管理容器。

您可以使用`docker run`命令启动一个新的容器。在下面的示例中，我们将演示如何使用`docker run`在 Ubuntu 16.04 容器中执行 shell 会话。为此，我们需要指定以下参数：

+   `-i`指定我们正在尝试启动一个交互式会话。也可以在不交互的情况下执行单个 docker 命令（例如，当启动 Web 服务器时）。

+   `-t <image name>`指定要使用哪个系统镜像。在下面的示例中，我们使用`ubuntu:16.04`镜像。

+   ` /bin/bash`，这是在容器内运行的命令，如下所示：

```py
 $ docker run -i -t ubuntu:16.04 /bin/bash
 root@585f53e77ce9:/#

```

此命令将立即带我们进入一个独立的、隔离的 shell，我们可以在其中与系统互动并安装软件，而不会触及主机操作系统。使用容器是测试不同 Linux 发行版上的安装和部署的非常好的方法。在完成交互式 shell 的工作后，我们可以输入 `exit` 命令返回到主机系统。

在上一章中，我们也使用了端口和分离选项 `-p` 和 `-d` 来运行可执行文件 `pyspark`。`-d` 选项只是要求 Docker 在后台运行命令。而 `-p <host_port>:<guest_port>` 选项则是必要的，用于将主机操作系统的网络端口映射到客户系统；没有这个选项，Jupyter Notebook 就无法从运行在主机系统中的浏览器访问。

我们可以使用 `docker ps` 监控容器的状态，如下面的片段所示。`-a` 选项（代表 *all*）用于输出有关所有容器的信息，无论它们当前是否正在运行：

```py
$ docker ps -a
CONTAINER ID IMAGE        COMMAND     CREATED       STATUS     PORTS NAMES
585f53e77ce9 ubuntu:16.04 "/bin/bash" 2 minutes ago Exited (0)       2 minutes ago pensive_hamilton

```

`docker ps` 提供的信息包括一个十六进制标识符 `585f53e77ce9` 以及一个可读名称 `pensive_hamilton`，这两个都可以用于在其他 Docker 命令中指定容器。它还包括有关执行命令、创建时间和执行当前状态的其他信息。

您可以使用 `docker start` 命令恢复已退出的容器的执行。要获取对容器的 shell 访问权限，您可以使用 `docker attach`。这两个命令都可以跟容器 ID 或其可读名称：

```py
$ docker start pensive_hamilton 
pensive_hamilton
$ docker attach pensive_hamilton 
root@585f53e77ce9:/#

```

您可以使用 `docker run` 命令后跟容器标识符轻松地删除容器：

```py
$ docker rm pensive_hamilton

```

如您所见，您可以在不到一秒的时间内自由执行命令、运行、停止和恢复容器。使用 Docker 容器进行交互式操作是测试新包和进行实验的好方法，而不会干扰主机操作系统。由于您可以同时运行多个容器，Docker 还可以用来模拟分布式系统（用于测试和学习目的），而无需拥有昂贵的计算集群。

Docker 还允许您创建自己的系统镜像，这对于分发、测试、部署和文档用途非常有用。这将是下一小节的主题。

# 创建 Docker 镜像

Docker 镜像是现成的、预配置的系统。可以使用 `docker run` 命令访问和安装可在 **DockerHub** ([`hub.docker.com/`](https://hub.docker.com/)) 上找到的 Docker 镜像，这是一个维护者上传现成镜像以测试和部署各种应用程序的在线服务。

创建 Docker 镜像的一种方法是在现有的容器上使用 `docker commit` 命令。`docker commit` 命令接受容器引用和输出镜像名称作为参数：

```py
$ docker commit <container_id> <new_image_name>

```

使用这种方法可以保存特定容器的快照，但如果图像从系统中删除，重新创建图像的步骤也会丢失。

创建图像的更好方法是使用**Dockerfile**构建。Dockerfile 是一个文本文件，它提供了从另一个图像开始构建图像的指令。例如，我们将展示我们在上一章中用于设置带有 Jupyter 笔记本支持的 PySpark 的 Dockerfile 的内容。完整的文件如下所示。

每个 Dockerfile 都需要一个起始图像，可以使用`FROM`命令声明。在我们的例子中，起始图像是`jupyter/scipy-notebook`，它可以通过 DockerHub ([`hub.docker.com/r/jupyter/scipy-notebook/`](https://hub.docker.com/r/jupyter/scipy-notebook/))获取。

一旦我们定义了起始图像，我们就可以开始使用一系列`RUN`和`ENV`命令来发出 shell 命令，安装包和执行其他配置。在下面的示例中，你可以识别出 Java 运行时环境（`openjdk-7-jre-headless`）的安装，以及下载 Spark 和设置相关环境变量。`USER`指令可以用来指定执行后续命令的用户：

```py
 FROM jupyter/scipy-notebook
 MAINTAINER Jupyter Project <jupyter@googlegroups.com>
 USER root

    # Spark dependencies
 ENV APACHE_SPARK_VERSION 2.0.2
 RUN apt-get -y update && 
        apt-get install -y --no-install-recommends 
        openjdk-7-jre-headless && 
        apt-get clean && 
        rm -rf /var/lib/apt/lists/*
 RUN cd /tmp && 
        wget -q http://d3kbcqa49mib13.cloudfront.net/spark-
        ${APACHE_SPARK_VERSION}-bin-hadoop2.6.tgz    && 
        echo "ca39ac3edd216a4d568b316c3af00199
              b77a52d05ecf4f9698da2bae37be998a 
              *spark-${APACHE_SPARK_VERSION}-bin-hadoop2.6.tgz" | 
        sha256sum -c - && 
        tar xzf spark-${APACHE_SPARK_VERSION}
        -bin-hadoop2.6.tgz -C /usr/local && 
        rm spark-${APACHE_SPARK_VERSION}-bin-hadoop2.6.tgz
 RUN cd /usr/local && ln -s spark-${APACHE_SPARK_VERSION}
        -bin-hadoop2.6 spark

    # Spark and Mesos config
 ENV SPARK_HOME /usr/local/spark
 ENV PYTHONPATH $SPARK_HOME/python:$SPARK_HOME/python/lib/
        py4j-0.10.3-src.zip
 ENV SPARK_OPTS --driver-java-options=-Xms1024M 
        --driver-java-options=-
        Xmx4096M --driver-java-options=-Dlog4j.logLevel=info

 USER $NB_USER

```

可以使用以下命令从 Dockerfile 所在的目录创建图像。`-t`选项可以用来指定存储图像时使用的标签。以下行可以创建名为`pyspark`的图像，该图像来自前面的 Dockerfile：

```py
$ docker build -t pyspark .

```

命令将自动检索起始图像`jupyter/scipy-notebook`，并生成一个新图像，命名为`pyspark`。

# 持续集成

持续集成是确保应用程序在每次开发迭代中保持无错误的好方法。持续集成背后的主要思想是频繁地运行项目的测试套件，通常在一个单独的构建服务器上，该服务器直接从主项目仓库拉取代码。

通过在机器上手动设置 Jenkins ([`jenkins.io/`](https://jenkins.io/))、Buildbot ([`buildbot.net/`](http://buildbot.net/))和 Drone ([`github.com/drone/drone`](https://github.com/drone/drone))等软件来设置构建服务器可以完成。这是一个方便且成本低的解决方案，特别是对于小型团队和私人项目。

大多数开源项目都利用了 Travis CI ([`travis-ci.org/`](https://travis-ci.org/))，这是一个能够从你的仓库自动构建和测试你的代码的服务，因为它与 GitHub 紧密集成。截至今天，Travis CI 为开源项目提供免费计划。许多开源 Python 项目利用 Travis CI 来确保程序在多个 Python 版本和平台上正确运行。

通过包含一个包含项目构建说明的 `.travis.yml` 文件，并注册账户后激活 Travis CI 网站上的构建（[`travis-ci.org/`](https://travis-ci.org/)），可以从 GitHub 仓库轻松设置 Travis CI。

这里展示了一个高性能应用的 `.travis.yml` 示例。该文件包含使用 YAML 语法编写的几个部分，用于指定构建和运行软件的说明。

`python` 部分指定了要使用的 Python 版本。`install` 部分将下载并设置 conda 以进行测试、安装依赖项和设置项目。虽然这一步不是必需的（可以使用 `pip` 代替），但 conda 是高性能应用的优秀包管理器，因为它包含有用的本地包。

`script` 部分包含测试代码所需的代码。在这个例子中，我们限制自己只运行测试和基准测试：

```py
 language: python
 python:
      - "2.7"
      - "3.5"
 install:      # Setup miniconda
      - sudo apt-get update
      - if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
          wget https://repo.continuum.io/miniconda/
          Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
        else
          wget https://repo.continuum.io/miniconda/
          Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
        fi
      - bash miniconda.sh -b -p $HOME/miniconda
      - export PATH="$HOME/miniconda/bin:$PATH"
      - hash -r
      - conda config --set always_yes yes --set changeps1 no
      - conda update -q conda
      # Installing conda dependencies
      - conda create -q -n test-environment python=
        $TRAVIS_PYTHON_VERSION numpy pandas cython pytest
      - source activate test-environment
      # Installing pip dependencies
      - pip install pytest-benchmark
      - python setup.py install

 script:
      pytest tests/
      pytest benchmarks/

```

每次将新代码推送到 GitHub 仓库（以及其他可配置的事件）时，Travis CI 将启动一个容器，安装依赖项，并运行测试套件。在开源项目中使用 Travis CI 是一种很好的实践，因为它是对项目状态的一种持续反馈，同时也通过持续测试的 `.travis.yml` 文件提供最新的安装说明。

# 摘要

决定优化软件的策略是一个复杂且微妙的工作，它取决于应用程序类型、目标平台和业务需求。在本章中，我们提供了一些指导方针，以帮助你思考和选择适合你自己的应用程序的适当软件堆栈。

高性能数值应用有时需要管理第三方包的安装和部署，这些包可能需要处理外部工具和本地扩展。在本章中，我们介绍了如何构建你的 Python 项目，包括测试、基准测试、文档、Cython 模块和 C 扩展。此外，我们还介绍了持续集成服务 Travis CI，它可以用于为托管在 GitHub 上的项目启用持续测试。

最后，我们还学习了可以使用虚拟环境和 docker 容器来测试应用程序，这可以极大地简化部署并确保多个开发者可以访问相同的平台。
