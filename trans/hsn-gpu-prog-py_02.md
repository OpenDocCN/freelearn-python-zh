# 第二章：设置 GPU 编程环境

现在我们将看到如何在 Windows 和 Linux 下设置适当的 GPU 编程环境。在这两种情况下，我们都需要采取几个步骤。我们将逐步进行这些步骤，注意 Linux 和 Windows 之间的任何差异。当然，您可以随意跳过或忽略不适用于您选择的操作系统的任何部分或注释。

读者应注意，本章仅涵盖 64 位 Intel/AMD PC 的两个平台——Ubuntu LTS（长期支持）版本和 Windows 10。请注意，任何基于 Ubuntu LTS 的 Linux 操作系统（如 Xubuntu，Kubuntu 或 Linux Mint）也同样适用于通用的 Unity/GNOME-based Ubuntu 版本。

我们建议使用 Python 2.7 而不是 Python 3.x。 Python 2.7 在本文中使用的所有库中都有稳定的支持，并且我们已经在 Windows 和 Linux 平台上使用 Python 2.7 测试了本书中给出的每个示例。 Python 3.x 用户可以使用本书，但应该注意 Python 2.7 和 Python 3.x 之间的区别。本书中的一些示例已经在 Python 3.7 上进行了测试，但需要标准更改，例如在 Python `print`函数中添加括号。

Packt 作者 Sebastian Raschka 博士在[`sebastianraschka.com/Articles/2014_python_2_3_key_diff.html`](https://sebastianraschka.com/Articles/2014_python_2_3_key_diff.html)提供了 Python 2.7 和 3.x 之间的关键区别列表。

我们特别建议 Windows 和 Linux 用户使用 Anaconda Python 2.7 版本，因为它可以在用户基础上安装，无需`sudo`或`管理员`权限，包含本文所需的所有数据科学和可视化模块，并使用快速预优化的 NumPy/SciPy 包，这些包利用了英特尔的**数学核心库**（**MKL**）。 （默认的 Linux `/usr/bin/python`安装对于本文也应该足够，但您可能需要手动安装一些包，如 NumPy 和 Matplotlib。）

Anaconda Python（2.7 和 3.x 版本）可以在[`www.anaconda.com/download/.`](https://www.anaconda.com/download/)下载到所有平台上。

其他受支持平台的用户（例如 macOS，Windows 7/8，Windows Server 2016，Red Hat/Fedora，OpenSUSE 和 CENTOS）应查阅官方的 NVIDIA CUDA 文档（[`docs.nvidia.com/cuda/`](https://docs.nvidia.com/cuda/)）以获取更多详细信息。此外，还有其他硬件选择：对于对嵌入式系统或具有树莓派等开发板经验的读者，可能希望从基于 ARM 的 NVIDIA Jetson 开发板开始，而对于对云计算或 Web 编程感兴趣的读者，可能考虑远程使用适当的 Azure 或 AWS 实例。在这些情况下，鼓励读者阅读官方文档以设置其驱动程序，编译器和 CUDA 工具包。本章中的一些步骤可能适用，也可能不适用。

本章的学习目标是：

+   确保我们拥有适当的硬件

+   安装 NVIDIA GPU 驱动程序

+   设置适当的 C/C++编程环境

+   安装 NVIDIA CUDA 工具包

+   为 GPU 编程设置 Python 环境

# 技术要求

本章建议安装 Anaconda Python 2.7，网址为[`www.anaconda.com/download/.`](https://www.anaconda.com/download/)

本章的代码也可以在 GitHub 上找到，网址为[`github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA.`](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)

有关先决条件的更多信息，请查看本书的前言；有关软件和硬件要求，请查看[`github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA`](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)中的 README 部分。

# 确保我们拥有正确的硬件

对于本书，我们建议您至少具备以下硬件：

+   64 位英特尔/AMD PC

+   4GB RAM

+   NVIDIA GeForce GTX 1050 GPU（或更高）

这种配置将确保您可以轻松学习 GPU 编程，在本书中运行所有示例，并且还可以运行一些其他新的有趣的基于 GPU 的软件，如 Google 的 TensorFlow（一种机器学习框架）或 Vulkan SDK（一种尖端的图形 API）。

**请注意，您必须拥有 NVIDIA 品牌的 GPU 才能使用本书！** CUDA Toolkit 专为 NVIDIA 卡而设计，因此无法用于编程 Intel HD 或 Radeon GPU。

正如所述，我们将假设您使用的是 Windows 10 或 Ubuntu LTS（长期支持）版本。

Ubuntu LTS 发布通常具有 14.04、16.04、18.04 等形式的版本号。

Ubuntu LTS，大体上来说，是最主流的 Linux 版本，可以确保与新软件和工具包的最大兼容性。请记住，有许多基于 Ubuntu 的 Linux 变体，如 Linux Mint 或 Xubuntu，这些通常同样有效。（我个人发现 Linux Mint 在配备 GPU 的笔记本电脑上开箱即用效果相当不错。）

我们应该注意，我们假设您至少拥有一款入门级 GTX 1050（Pascal）GPU，或者在任何更新的架构中具有相当的性能。请注意，本书中的许多示例很可能在大多数旧 GPU 上运行，但作者只在 GTX 1050（在 Windows 10 下）和 GTX 1070（在 Linux 下）上进行了测试。虽然这些示例尚未在旧 GPU 上进行测试，但 2014 年的入门级 Maxwell 架构 GPU，如 GTX 750，也应足以满足本文的要求。

如果您使用的是台式 PC，请确保在继续之前已经按照所有包含的说明物理安装了 GPU。

# 检查您的硬件（Linux）

现在，我们将在 Linux 中进行一些基本检查，以确保我们拥有正确的硬件。首先让我们打开一个终端并切换到 bash 命令行——您可以通过在 Ubuntu 中快速按下组合键*Ctrl* + *Alt* + *T*来快速完成这一步。

现在，通过输入`lscpu`并按*Enter*来检查我们的处理器。会出现大量信息，但只需查看第一行，确保架构确实是 x86_64：

![](img/e6983512-f18c-418c-9307-e1169baf3591.png)

接下来，通过在 bash 提示符下输入`free -g`并再次按*Enter*来检查我们的内存容量。这将告诉我们在第一行的第一个条目中我们拥有的总内存量（以 GB 为单位），以及在接下来的行中交换空间中的内存量：

![](img/c3146279-5d0d-47b3-a8d9-cc7db17d807b.png)

这绝对是足够的内存。

最后，让我们看看我们是否有适当的 GPU。NVIDIA GPU 通过 PCI 总线与我们的 PC 通信，因此我们可以使用`lspci`命令列出所有 PCI 硬件。通常会列出许多其他硬件，因此让我们使用`grep`命令仅过滤出 NVIDIA GPU，输入`lspci | grep -e "NVIDIA"`在 bash 提示符下：

![](img/4365ea85-449d-4d80-b0c6-4514fdebe0e5.png)

这是一款 GTX 1070，幸运的是它超出了我们至少需要 GTX 1050 的要求。

# 检查您的硬件（Windows）

首先，我们必须打开 Windows 面板。我们可以通过按下*Windows* + *R*，然后在提示符处输入`Control Panel`来实现这一点，如下面的屏幕截图所示：

![](img/80e71c36-7fdf-4962-bb3a-107afc76022b.png)

Windows 控制面板将弹出。现在点击系统和安全，然后选择以下屏幕上的系统。这将立即告诉我们我们拥有多少 RAM 以及我们是否拥有 64 位处理器：

![](img/73eb0d90-a13d-42e1-a0ac-7941a3f8ff5e.png)

要检查我们的 GPU，请点击此窗口左上角的设备管理器。然后 Windows 设备管理器将弹出；然后您可以选择显示适配器下拉框来检查您系统上的 GPU：

![](img/e084b0d1-48a8-4280-86a5-dd505ff38c0c.png)

# 安装 GPU 驱动程序

如果您已经安装了 GPU 的驱动程序，您可能可以跳过此步骤；此外，一些版本的 CUDA 已经预先打包了最新的驱动程序。通常情况下，CUDA 对您安装的驱动程序非常挑剔，甚至可能无法与 CUDA Toolkit 驱动程序一起工作，因此您可能需要尝试几种不同的驱动程序，直到找到一个可用的。

一般来说，Windows 具有更好的 CUDA 驱动程序兼容性和更用户友好的安装比 Linux。Windows 用户可以考虑跳过此步骤，只使用与 CUDA Toolkit 捆绑的驱动程序，我们稍后将在本章中安装。然而，我们强烈建议 Linux 用户（特别是 Linux 笔记本用户）在继续之前，密切遵循本节中的所有步骤。

# 安装 GPU 驱动程序（Linux）

在 Ubuntu 中，NVIDIA GPU 的默认驱动程序是一个名为 Nouveau 的开源驱动程序；不幸的是，这在 CUDA 中根本不起作用，因此我们必须安装专有驱动程序。我们必须将特殊的`graphics-drivers`存储库添加到我们的软件包管理器中，以便能够将专有 NVIDIA 驱动程序下载到我们的 Ubuntu 系统中。我们通过在 bash 提示符中输入以下行来添加存储库：

```py
sudo add-apt-repository ppa:graphics-drivers/ppa
```

由于这是一个`sudo`超级用户命令，您将需要输入您的密码。我们现在通过输入以下行来将系统与新的存储库同步：

```py
sudo apt-get update
```

我们现在应该准备安装我们的驱动程序。从 Ubuntu 桌面，按下*Windows* + *R*，然后输入`software and drivers`：

![](img/33637818-a7d1-4fb9-8b47-9c09741b001c.png)

软件和驱动程序设置菜单应该出现。从这里，点击标记为附加驱动程序的选项卡。您应该看到一系列可用的稳定专有驱动程序供您的 GPU 选择；选择您看到的最新的一个（在我的情况下，它是`nvidia-driver-396`，如下所示）：

![](img/d75d9ff2-f44a-459c-ba72-7ded556d763b.png)

选择最新的驱动程序后，点击应用更改。您将再次被提示输入您的`sudo`密码，然后驱动程序将安装；进度条应该出现。请注意，这个过程可能需要很长时间，而且可能会出现您的计算机“挂起”的情况；这个过程可能需要超过一个小时，所以请耐心等待。

最后，当过程完成时，重启您的计算机，返回到 Ubuntu 桌面。现在输入*Windows* + *A*，然后输入`nvidia-settings`（或者，从 bash 提示符中运行此程序）。NVIDIA X Server 设置管理器应该出现，并指示您正在使用适当的驱动程序版本：

![](img/bdab7346-2e9f-4615-b61b-a91f6b6b8588.png)

# 安装 GPU 驱动程序（Windows）

重申一下-通常建议读者最初跳过此步骤，然后安装包含在 CUDA Toolkit 中的驱动程序。

Windows 的最新驱动程序可以直接从 NVIDIA 的[`www.nvidia.com/Download/`](http://www.nvidia.com/Download/)下载。只需从下拉菜单中选择适用于您 GPU 的适当的 Windows 10 驱动程序，这些是可执行（`.exe`）文件。只需通过双击文件管理器中的文件来安装驱动程序。

# 建立 C++编程环境

现在我们已经安装了驱动程序，我们必须设置我们的 C/C++编程环境；Python 和 CUDA 都对它们可能集成的编译器和 IDE 有特殊要求，所以您可能需要小心。对于 Ubuntu Linux 用户，标准存储库的编译器和 IDE 通常可以完美地与 CUDA 工具包集成，而 Windows 用户可能需要更加小心。

# 设置 GCC、Eclipse IDE 和图形依赖项（Linux）

从 Ubuntu 桌面打开终端（*Ctrl* + *Alt* + *T*）。我们首先更新`apt`存储库如下：

```py
sudo apt-get update
```

现在我们可以用一行额外的命令安装我们需要的 CUDA 一切：

```py
sudo apt-get install build-essential binutils gdb eclipse-cdt
```

在这里，`build-essential`是带有`gcc`和`g++`编译器以及其他实用程序（如 make）的软件包；`binutils`有一些通用的实用程序，如 LD 链接器；`gdb`是调试器；Eclipse 是我们将要使用的 IDE。

让我们还安装一些额外的依赖项，这将允许我们使用以下命令运行 CUDA 工具包中包含的一些图形（OpenGL）演示：

```py
sudo apt-get install freeglut3 freeglut3-dev libxi-dev libxmu-dev
```

现在您应该可以安装 CUDA 工具包了。

# 在 Windows 上设置 Visual Studio

在撰写本文时，只有一个版本的 Visual Studio 似乎完美地与 Python 和最新的 CUDA 工具包集成在一起——Visual Studio 2015；也就是说，Visual Studio 版本 14.0。

虽然可能可以在较新版本的 Visual Studio（例如 2017）下进行子安装，但我们建议读者直接在系统上安装带有 C/C++支持的 Visual Studio 2015。

Visual Studio Community 2015，这个软件的免费版本，可以在[`visualstudio.microsoft.com/vs/older-downloads/`](https://visualstudio.microsoft.com/vs/older-downloads/)下载。

在这里，我们将进行最小化安装，只安装 CUDA 所需的组件。我们运行安装软件，并选择自定义安装：

![](img/64390e7f-5b9c-4768-9d4e-b7b4116d57c7.png)

点击下一步，然后点击编程语言的下拉框，然后选择 Visual C++（如果您需要其他包或编程语言，可以随意选择，但是对于 GPU 编程，我们只需要 Visual C++）：

![](img/63cef2a7-be52-499c-af90-810042581d5d.png)

这个安装过程可能需要一些时间。完成后，我们将准备安装 CUDA 工具包。

# 安装 CUDA 工具包

最后，我们开始接近我们的目标！现在我们通过访问[`developer.nvidia.com/cuda-downloads`](https://developer.nvidia.com/cuda-downloads)来下载我们的 CUDA 工具包。选择适当的操作系统，您将看到几个选项。对于 Windows 和 Linux，都有网络和本地安装选项。我倾向于在 Windows 和 Linux 下都使用本地安装选项，因为我更喜欢一次性下载整个软件包；如果有任何网络问题，那么您可以确保在安装 CUDA 工具包时不会发生问题。

# 安装 CUDA 工具包（Linux）

对于 Linux 用户，您将看到使用`.deb`包和`.run`文件的选择；对于大多数用户，我建议使用`.deb`文件，因为这将自动安装 CUDA 需要的任何缺少的软件包。`.run`文件安装在系统的**高级软件包工具**（APT）系统之外，它只是将适当的文件复制到系统的`/usr`二进制和库目录。如果您不想干扰系统的 APT 系统或存储库，并且对 Linux 有很好的理解，那么`.run`文件可能更合适。无论哪种情况，请仔细遵循网站上关于安装软件包的说明，这些说明可能会因版本而略有不同。

安装包完成后，您可能需要配置您的`PATH`和`LD_SYSTEM_CONFIG`环境变量，以便您的系统可以找到 CUDA 所需的适当的二进制可执行文件和库文件。我建议您通过将以下行附加到您用户目录中的`.bashrc`文件的末尾来完成这个步骤。使用您喜欢的文本编辑器，如`gedit`、`nano`、`emacs`或`vim`打开`~/.bashrc`文件，然后在文件的最底部添加以下行：

```py
export PATH="/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
```

保存文件然后退出终端。您现在可以通过打开一个新的终端并输入`nvcc --version`然后按*Enter*来确保您已正确安装了工具包，这将给您工具包编译器的版本信息。（`nvcc`是命令行 CUDA C 编译器，类似于`gcc`编译器。）

# 安装 CUDA Toolkit（Windows）

对于 Windows 用户，您可以通过双击`.exe`文件并按照屏幕上的提示来安装包。

安装完成后，重置您的系统。我们现在将通过检查`nvcc`编译器来确保 CUDA 已正确安装。在开始菜单下，点击`Visual Studio 2015`文件夹，然后点击 VS2015 x64 Native Tools Command Prompt。一个终端窗口将弹出；现在输入`nvcc --version`并按*Enter*，这应该会给您 NVIDIA 编译器的版本信息。

# 为 GPU 编程设置我们的 Python 环境

使用我们的编译器、集成开发环境和 CUDA 工具包正确安装在我们的系统上，我们现在可以为 GPU 编程设置一个合适的 Python 环境。这里有很多选择，但我们明确建议您使用 Anaconda Python Distribution。Anaconda Python 是一个独立且用户友好的分发版，可以直接安装在您的用户目录中，而且不需要任何管理员或`sudo`级别的系统访问权限来安装、使用或更新。

请记住，Anaconda Python 有两种版本——Python 2.7 和 Python 3。由于 Python 3 目前对我们将要使用的一些库的支持不是很好，我们将在本书中使用 Python 2.7，这仍然是广泛使用的。

您可以通过访问[`www.anaconda.com/download`](https://www.anaconda.com/download)来安装 Anaconda Python，选择您的操作系统，然后选择下载分发版的 Python 2.7 版本。按照 Anaconda 网站上给出的说明安装分发版，这相对比较简单。现在我们可以为 GPU 编程设置本地 Python 安装。

我们现在将设置本书中可能是最重要的 Python 包：Andreas Kloeckner 的 PyCUDA 包。

# 安装 PyCUDA（Linux）

在 Linux 中打开一个命令行。通过在 bash 提示符下输入`which python`并按*Enter*来确保您的`PATH`变量正确设置为使用本地 Anaconda 安装的 Python（而不是系统范围的安装）（Anaconda 应该在安装过程中自动配置您的`.bashrc`）；这应该告诉您 Python 二进制文件在您的本地`~/anaconda2/bin`目录中，而不是在`/usr/bin`目录中。如果不是这种情况，请打开一个文本编辑器，并在您的`~/.bashrc`文件的末尾放置以下行`export PATH="/home/${USER}/anaconda2/bin:${PATH}"`，保存后，打开一个新的终端，然后再次检查。

有几种安装 PyCUDA 的选项。最简单的选项是从 PyPI 存储库安装最新稳定版本，方法是输入`pip install pycuda`。您还可以按照 PyCUDA 官方网站上的说明安装最新版本的 PyCUDA，网址为[`mathema.tician.de/software/pycuda/`](https://mathema.tician.de/software/pycuda/)。请注意，如果您希望从不同的来源重新安装 PyCUDA，请确保首先使用`pip uninstall pycuda`卸载它。

# 创建一个环境启动脚本（Windows）

Windows 用户需要特别注意，他们的 Visual Studio 和 Anaconda Python 环境变量是否设置正确，以便使用 PyCUDA；否则，Python 将无法找到 NVIDIA 的`nvcc` CUDA 编译器或 Microsoft 的`cl.exe` C++编译器。幸运的是，包含了批处理脚本，可以自动为我们设置这些环境，但我们必须小心，每次想要进行 GPU 编程时都要执行这些脚本。

因此，我们将创建一个批处理脚本，通过连续调用这两个脚本来启动适当的 IDE 或命令行环境。 （此脚本也可在[`github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA/blob/master/2/launch-python-cuda-environment.bat`](https://github.com/btuomanen/handsongpuprogramming/blob/master/2/launch-python-cuda-environment.bat)上找到。）

请务必首先打开 Windows 记事本，并跟随操作：

首先找到您的 Visual Studio 的`vcvars.bat`文件的位置；对于 Visual Studio 2015，它位于`C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat`。

在文本编辑器中输入以下行，然后按*Enter*：

```py
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
```

现在，我们需要调用 Anaconda 的`activate.bat`脚本来设置 Anaconda Python 环境变量；标准路径是`Anaconda2\Scripts\activate.bat`。我们还必须指示此脚本的参数是 Anaconda 库的位置。在我的情况下，我的启动脚本中的第二行将是`call "C:\Users\%username%\Anaconda2\Scripts\activate.bat" C:\Users\%username%\Anaconda2`。

最后，我们的批处理脚本的最后一行将启动您喜欢的任何环境——IDE 或命令行提示符，它将继承前两个脚本设置的所有必要环境和系统变量。如果您喜欢旧的标准 DOS 风格命令提示符，这行应该只是`cmd`。如果您喜欢从 PowerShell 工作，请将其更改为`powershell`。在某些情况下，特别是用于访问命令行`pip`和`conda`来更新 Python 库时，需要使用命令行。

最后，将此文件保存到桌面，并命名为`launch-python-cuda-environment.bat`。现在，您可以通过双击此文件来启动我们的 Python GPU 编程环境。

（请记住，如果您希望使用 Jupyter Notebook 或 Spyder Python IDE，您可以简单地通过`jupyter-notebook`或`spyder`从命令行启动它们，或者您可以制作一个批处理脚本，只需用适当的 IDE 启动命令替换`cmd`。）

# 安装 PyCUDA（Windows）

由于大多数 Python 库主要是由 Linux 用户编写和为 Linux 用户编写的，建议您从 Christoph Gohlke 的网站上安装预构建的 PyCUDA wheel 二进制文件，网址为：[`www.lfd.uci.edu/~gohlke/pythonlibs/#pycuda`](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycuda)。下载一个文件，文件名为`pycuda-2017.1.1+cuda(VERSION)-cp27-cp27m-win_amd64.whl`，其中版本是您的 CUDA 版本号。现在，您可以通过在命令行中输入以下内容并用您的 PyCUDA wheel 的完整路径和文件名替换`pycuda.whl`来安装 PyCUDA：

```py
pip install pycuda.whl
```

（或者，您可以尝试使用`pip install pycuda`从 PyPI 存储库安装 PyCUDA，或者按照 PyCUDA 网站上的说明操作。）

# 测试 PyCUDA

最后，我们到了可以看到我们的 GPU 编程环境是否真正起作用的时候。我们将运行下一章的一个小程序，该程序将查询我们的 GPU 并提供有关型号号码、内存、核心数量、架构等相关信息。从存储库中的目录`3`中获取 Python 文件（`deviceQuery.py`），也可以在[`github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA/blob/master/3/deviceQuery.py`](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA/blob/master/3/deviceQuery.py)上找到。

如果您使用的是 Windows，请确保通过在桌面上启动我们在上一节中创建的`.bat`文件来启动 GPU 编程环境。否则，如果您使用的是 Linux，请打开一个 bash 终端。现在输入以下命令并按*Enter*键——`python deviceQuery.py`。

这将输出许多行数据，但前几行应该表明 PyCUDA 已经检测到您的 GPU，并且您应该在下一行看到型号号码：

![](img/51a20697-a1c6-435b-95a0-9845a9506ab6.png)

恭喜，您现在已经准备好进入 GPU 编程的世界了！

# 总结

为 GPU 编程设置 Python 环境可能是一个非常微妙的过程。本文建议 Windows 和 Linux 用户都使用 Anaconda Python 2.7 发行版。首先，我们应该确保我们有正确的硬件进行 GPU 编程；一般来说，64 位 Windows 或 Linux PC，带有 4GB RAM 和 2016 年或之后的任何入门级 NVIDIA GPU 将足够满足我们的需求。Windows 用户应该注意使用一个既适用于 CUDA 工具包又适用于 Anaconda 的 Visual Studio 版本（如 VS 2015），而 Linux 用户在安装 GPU 驱动程序时应特别小心，并在其`.bashrc`文件中设置适当的环境变量。此外，Windows 用户应该创建一个适当的启动脚本，用于设置 GPU 编程环境，并应该使用预编译的 PyCUDA 库安装文件。

现在，我们的编程环境已经设置好了，接下来的一章我们将学习 GPU 编程的基础知识。我们将看到如何将数据写入 GPU 的内存，以及如何在 CUDA C 中编写一些非常简单的*逐元素*GPU 函数。（如果你看过经典的 1980 年代电影《功夫小子》，那么你可能会把下一章看作是 GPU 编程的“上蜡，下蜡”。）

# 问题

1.  我们可以在主处理器内置的英特尔 HD GPU 上运行 CUDA 吗？离散的 AMD Radeon GPU 呢？

1.  这本书的示例是使用 Python 2.7 还是 Python 3.7？

1.  在 Windows 中，我们使用什么程序来查看我们安装了什么 GPU 硬件？

1.  在 Linux 中，我们使用什么命令行程序来查看我们安装了什么 GPU 硬件？

1.  在 Linux 中，我们使用什么命令来确定系统有多少内存？

1.  如果我们不想改变我们的 Linux 系统的 APT 存储库，我们应该使用`run`还是`deb`安装程序来安装 CUDA？
