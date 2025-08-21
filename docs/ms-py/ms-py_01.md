# 第一章：入门-每个项目一个环境

Python 哲学的一个方面一直以来都是最重要的，也将永远如此——可读性，或者说 Pythonic 代码。这本书将帮助你掌握编写 Python 的方式：可读、美观、明确，尽可能简单。简而言之，它将是 Pythonic 代码。这并不是说复杂的主题不会被涵盖。当然会，但每当 Python 的哲学受到影响时，你将被警告何时何地使用这种技术是合理的。

本书中的大部分代码将在 Python 2 和 Python 3 上运行，但主要目标是 Python 3。这样做有三个原因：

1.  Python 3 于 2008 年发布，这在快速变化的软件世界中已经是很长的时间了。它不再是新鲜事物，而是稳定的、可用的，最重要的是，它是未来。

1.  Python 2 的开发在 2009 年基本停止了。某些功能已经从 Python 3 回溯到 Python 2，但任何新的开发都将首先针对 Python 3。

1.  Python 3 已经成熟。我必须承认，Python 3.2 和更早版本仍存在一些小问题，这使得很难编写能在 Python 2 和 3 上运行的代码，但 Python 3.3 在这方面有了很大的改进，我认为它已经成熟。这一点可以从 Python 3.4 和 3.5 中略有修改的语法以及许多非常有用的功能得到证实，这些都在本书中有所涵盖。

总之，Python 3 是对 Python 2 的改进。我自己也是长期的怀疑论者，但我没有看到不使用 Python 3 进行新项目的理由，甚至将现有项目迁移到 Python 3 通常只需要进行一些小的更改。有了 Python 3.5 中的`async with`等新功能，你会想要升级来尝试一下。

这一章将向你展示如何正确设置环境，创建一个新的隔离环境，并确保在不同的机器上运行相同代码时获得类似的结果。大多数 Python 程序员已经在使用`virtualenv`创建虚拟 Python 环境，但在 Python 3.3 中引入的`venv`命令是一个非常好的替代方案。它本质上是`virtualenv`包的一个克隆，但稍微简单一些，并且与 Python 捆绑在一起。虽然它的使用方法大部分类似于`virtualenv`，但有一些有趣的变化值得知道。

其次，我们将讨论`pip`命令。使用`ensurepip`包通过`venv`自动安装`pip`命令，这是在 Python 3.4 中引入的一个包。这个包会在现有的 Python 库中自动引导`pip`，同时保持独立的 Python 和`pip`版本。在 Python 3.4 之前，`venv`没有`pip`，需要手动安装。

最后，我们将讨论如何安装使用`distutils`创建的包。纯 Python 包通常很容易安装，但涉及 C 模块时可能会有挑战。

在本章中，将涵盖以下主题：

+   使用`venv`创建虚拟 Python 环境

+   使用`ensurepip`引导 pip 的引导

+   使用`pip`基于`distutils`（C/C++）安装包

# 使用 venv 创建虚拟 Python 环境

大多数 Python 程序员可能已经熟悉`venv`或`virtualenv`，但即使你不熟悉，现在开始使用也不算晚。`venv`模块旨在隔离你的 Python 环境，这样你就可以安装特定于当前项目的包，而不会污染全局命名空间。此外，由于包是本地安装的，你不需要系统（root/administrator）访问权限来安装它们。

结果是，您可以确保在本地开发机器和生产机器上具有完全相同版本的软件包，而不会干扰其他软件包。例如，有许多 Django 软件包需要 Django 项目的特定版本。使用`venv`，您可以轻松地为项目 A 安装 Django 1.4，为项目 B 安装 Django 1.8，而它们永远不会知道其他环境中安装了不同的版本。默认情况下，甚至配置了这样的环境，以便全局软件包不可见。这样做的好处是，要获得环境中安装的所有软件包的确切列表，只需`pip freeze`即可。缺点是，一些较重的软件包（例如`numpy`）将必须在每个单独的环境中安装。不用说，哪种选择对您的项目最好取决于项目。对于大多数项目，我会保持默认设置，即不具有全局软件包，但是在处理具有大量 C/C++扩展的项目时，简单地启用全局站点软件包会很方便。原因很简单；如果您没有编译器可用，安装软件包可能会很困难，而全局安装对于 Windows 有可执行文件，对于 Linux/Unix 有可安装软件包可用。

### 注意

`venv`模块（[`docs.python.org/3/library/venv.html`](https://docs.python.org/3/library/venv.html)）可以看作是`virtualenv`工具（[`virtualenv.pypa.io/`](https://virtualenv.pypa.io/)）的一个略微简化的版本，自 Python 3.3 版本以来已经捆绑在一起（参见 PEP 0405 -- Python 虚拟环境：[`www.python.org/dev/peps/pep-0405/`](https://www.python.org/dev/peps/pep-0405/)）。

`virtualenv`包通常可以用作`venv`的替代品，特别是对于不捆绑`venv`的较旧的 Python 版本（3.3 以下）来说，这一点尤为重要。

## 创建您的第一个 venv

创建环境非常容易。基本命令是`pyvenv PATH_TO_THE_NEW_VIRTUAL_ENVIRONMENT`，所以让我们试一试。请注意，此命令适用于 Linux、Unix 和 Mac；Windows 命令将很快跟进：

```py
# pyvenv test_venv
# . ./test_venv/bin/activate
(test_venv) #

```

### 注意

一些 Ubuntu 版本（特别是 14.04 LTS）通过不在`ensurepip`中包含完整的`pyvenv`包来削弱 Python 安装。标准的解决方法是调用`pyvenv --without-pip test_env`，这需要通过`pip`主页上提供的`get_pip.py`文件手动安装`pip`。

这将创建一个名为`test_venv`的环境，第二行激活该环境。

在 Windows 上，一切都略有不同，但总体上是相似的。默认情况下，`pyvenv`命令不会在您的 PATH 中，因此运行该命令略有不同。三个选项如下：

+   将`Python\Tools\Scripts\`目录添加到您的 PATH

+   运行模块：

```py
python -m venv test_venv

```

+   直接运行脚本：

```py
python Python\Tools\Scripts\pyvenv.py test_venv

```

为了方便起见，我建议您无论如何将`Scripts`目录添加到您的 PATH，因为许多其他应用程序/脚本（如`pip`）也将安装在那里。

以下是 Windows 的完整示例：

```py
C:\envs>python -m venv test_venv
C:\envs>test_venv\Scripts\activate.bat
(test_venv) C:\envs>

```

### 提示

在使用 Windows PowerShell 时，可以通过使用`test_venv\Scripts\Activate.ps1`来激活环境。请注意，这里确实需要反斜杠。

## venv 参数

到目前为止，我们只是创建了一个普通的和常规的`venv`，但是有一些非常有用的标志可以根据您的需求定制您的`venv`。

首先，让我们看一下`venv`的帮助：

| 参数 | 描述 |
| --- | --- |
| `--system-site-packages` | 它使虚拟环境可以访问`system-site-packages`目录 |
| `--symlinks` | 尝试在平台不默认使用符号链接时使用`symlinks`而不是副本 |
| `--copies` | 尝试使用副本而不是符号链接，即使符号链接是平台的默认值 |
| `--clear` | 在环境创建之前删除环境目录的内容，如果存在的话 |
| `--upgrade` | 升级环境目录以使用 Python 的这个版本，假设 Python 已经被原地升级 |
| `--without-pip` | 这将跳过在虚拟环境中安装或升级 pip（pip 默认情况下是引导的） |

要注意的最重要的参数是`--system-site-packages`，它可以在环境中启用全局站点包。这意味着如果你在全局 Python 版本中安装了一个包，它也将在你的环境中可用。但是，如果你尝试将其更新到不同的版本，它将被安装在本地。在可能的情况下，我建议禁用`--system-site-packages`标志，因为它可以为你提供一个简单的环境，而不会有太多的变量。否则，简单地更新系统包可能会破坏你的虚拟环境，更糟糕的是，没有办法知道哪些包是本地需要的，哪些只是为其他目的安装的。

要为现有环境启用这个功能，你可以简单地再次运行环境创建命令，但这次加上`--system-site-packages`标志以启用全局站点包。

要再次禁用它，你可以简单地运行环境创建命令，不带标志。这将保留在环境中安装的本地包，但会从你的 Python 范围中删除全局包。

### 提示

在使用`virtualenvwrapper`时，也可以通过在激活的环境中使用`toggleglobalsitepackages`命令来完成这个操作。

`--symlinks`和`--copies`参数通常可以忽略，但了解它们的区别很重要。这些参数决定文件是从基本 Python 目录复制还是创建符号链接。

### 注意

符号链接是 Linux/Unix/Mac 的东西；它不是复制文件，而是创建一个符号链接，告诉系统在哪里找到实际的文件。

默认情况下，`venv`会尝试创建符号链接，如果失败，它会退而使用复制。自从 Windows Vista 和 Python 3.2 以来，这在 Windows 上也得到支持，所以除非你使用的是一个非常旧的系统，你很可能会在你的环境中使用符号链接。符号链接的好处是它节省了磁盘空间，并且与你的 Python 安装保持同步。缺点是，如果你的系统的 Python 版本升级了，它可能会破坏你的环境中安装的包，但这可以通过使用`pip`重新安装包来轻松解决。

最后，`--upgrade`参数在系统 Python 版本被原地升级后非常有用。这个参数的最常见用法是在使用复制（而不是符号链接）环境后修复损坏的环境。

## virtualenv 和 venv 之间的区别

由于`venv`模块本质上是`virtualenv`的一个简化版本，它们大部分是相同的，但有些地方是不同的。此外，由于`virtualenv`是一个与 Python 分开分发的包，它确实有一些优势。

以下是`venv`相对于`virtualenv`的优势：

+   `venv`随 Python 3.3 及以上版本一起分发，因此不需要单独安装

+   `venv`简单直接，除了基本必需品之外没有其他功能

`virtualenv`相对于`venv`的优势：

+   `virtualenv`是在 Python 之外分发的，因此可以单独更新。

+   `virtualenv`适用于旧的 Python 版本，但建议使用 Python 2.6 或更高版本。然而，使用较旧版本（1.9.x 或更低版本）可以支持 Python 2.5。

+   它支持方便的包装器，比如`virtualenvwrapper` ([`virtualenvwrapper.readthedocs.org/`](http://virtualenvwrapper.readthedocs.org/))

简而言之，如果`venv`对您足够了，就使用它。如果您使用的是旧版本的 Python 或需要一些额外的便利，比如`virtualenvwrapper`，则使用`virtualenv`。这两个项目本质上是做同样的事情，并且已经努力使它们之间易于切换。两者之间最大和最显著的区别是`virtualenv`支持的 Python 版本的种类。

# 使用 ensurepip 引导 pip

自 2008 年推出以来，`pip`软件包管理器一直在逐渐取代`easy_install`。自 Python 3.4 以来，它甚至已成为默认选项，并与 Python 捆绑在一起。从 Python 3.4 开始，它默认安装在常规 Python 环境和`pyvenv`中；在此之前，需要手动安装。要在 Python 3.4 及以上版本自动安装`pip`，需要使用`ensurepip`库。这是一个处理`pip`的自动安装和/或升级的库，因此至少与`ensurepip`捆绑的版本一样新。

## ensurepip 用法

使用`ensurepip`非常简单。只需运行 python `-m ensurepip`来保证`pip`的版本，或者运行 python `-m ensurepip --upgrade`来确保`pip`至少是与`ensurepip`捆绑的版本一样新。

除了安装常规的`pip`快捷方式外，这还将安装`pipX`和`pipX.Y`链接，允许您选择特定的 Python 版本。当同时使用 Python 2 和 Python 3 时，这允许您使用`pip2`和`pip3`在 Python 2 和 Python 3 中安装软件包。这意味着如果您在 Python 3.5 上使用 python `-m ensurepip`，您将在您的环境中安装`pip`、`pip3`和`pip3.5`命令。

## 手动 pip 安装

如果您使用的是 Python 3.4 或更高版本，`ensurepip`软件包非常好。然而，在此之下，您需要手动安装`pip`。实际上，这非常容易。只需要两个步骤：

1.  下载`get-pip.py`文件：[`bootstrap.pypa.io/get-pip.py`](https://bootstrap.pypa.io/get-pip.py)。

1.  执行`get-pip.py`文件：python `get-pip.py`。

### 提示

如果`ensurepip`命令由于权限错误而失败，提供`--user`参数可能会有用。这允许您在用户特定的站点包目录中安装`pip`，因此不需要 root/admin 访问权限。

# 安装 C/C++软件包

大多数 Python 软件包纯粹是 Python，并且安装起来非常容易，只需简单的`pip install packagename`就可以了。然而，有些情况涉及到编译，安装不再是简单的 pip install，而是需要搜索几个小时以查看安装某个软件包所需的依赖关系。

特定的错误消息会根据项目和环境而异，但这些错误中有一个共同的模式，了解您所看到的内容可以在寻找解决方案时提供很大帮助。

例如，在标准的 Ubuntu 机器上安装`pillow`时，您会得到几页错误、警告和其他消息，最后是这样的：

```py
 **x86_64-linux-gnu-gcc: error: build/temp.linux-x86_64-3.4/libImaging/Jpeg2KDecode.o: No such file or directory
 **x86_64-linux-gnu-gcc: error: build/temp.linux-x86_64-3.4/libImaging/Jpeg2KEncode.o: No such file or directory
 **x86_64-linux-gnu-gcc: error: build/temp.linux-x86_64-3.4/libImaging/BoxBlur.o: No such file or directory
 **error: command 'x86_64-linux-gnu-gcc' failed with exit status 1

 **----------------------------------------
Command "python3 -c "import setuptools, tokenize;__file__='/tmp/pip-build-_f0ryusw/pillow/setup.py';exec(compile(getattr(tokenize, 'open', open)(__file__).read().replace('\r\n', '\n'), __file__, 'exec'))" install --record /tmp/pip-kmmobum2-record/install-record.txt --single-version-externally-managed --compile --install-headers include/site/python3.4/pillow" failed with error code 1 in /tmp/pip-build-_f0ryusw/pillow

```

看到这样的消息后，您可能会想要搜索其中的一行，比如`x86_64-linux-gnu-gcc: error: build/temp.linux-x86_64-3.4/libImaging/Jpeg2KDecode.o: No such file or directory`。虽然这可能会给您一些相关的结果，但很可能不会。在这种安装中的技巧是向上滚动，直到看到有关缺少头文件的消息。这是一个例子：

```py
 **In file included from libImaging/Imaging.h:14:0,
 **from libImaging/Resample.c:16:
 **libImaging/ImPlatform.h:10:20: fatal error: Python.h: No such file or directory
 **#include "Python.h"
 **^
 **compilation terminated.

```

这里的关键消息是缺少`Python.h`。这些是 Python 头文件的一部分，需要用于 Python 中大多数 C/C++软件包的编译。根据操作系统的不同，解决方案也会有所不同，不幸的是。因此，我建议您跳过本段中与您的情况无关的部分。

## Debian 和 Ubuntu

在 Debian 和 Ubuntu 中，要安装的软件包是`python3-dev`或`python2-dev`（如果您仍在使用 Python 2）。要执行的命令如下：

```py
# sudo apt-get install python3-dev

```

但是，这只安装了开发头文件。如果您希望编译器和其他头文件与安装捆绑在一起，那么`build-dep`命令也非常有用。以下是一个示例：

```py
# sudo apt-get build-dep python3

```

## Red Hat、CentOS 和 Fedora

Red Hat、CentOS 和 Fedora 是基于 rpm 的发行版，它们使用`yum`软件包管理器来安装所需的软件。大多数开发头文件都可以通过`<package-name>-devel`获得，并且可以轻松安装。要安装 Python 3 开发头文件，请使用以下命令：

```py
# sudo apt-get install python3-devel

```

为了确保您具有构建软件包（如 Python）所需的所有要求，例如开发头文件和编译器，`yum-builddep`命令是可用的：

```py
# yum-builddep python3

```

## OS X

在实际安装软件包之前，OS X 上的安装过程包括三个步骤。

首先，您需要安装 Xcode。这可以通过 OS X App Store 完成，网址为[`itunes.apple.com/en/app/xcode/id497799835?mt=12`](https://itunes.apple.com/en/app/xcode/id497799835?mt=12)。

然后，您需要安装 Xcode 命令行工具：

```py
# xcode-select --install

```

最后，您需要安装**Homebrew**软件包管理器。步骤可在[`brew.sh/`](http://brew.sh/)找到，但安装命令如下：

```py
# /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

```

### 注意

其他软件包管理器，如`Macports`，也是可能的，但`Homebrew`目前是 OS X 上开发和社区最活跃的软件包管理器。

完成所有这些步骤后，您应该有一个可用的 Homebrew 安装。可以使用`brew doctor`命令验证`Homebrew`的工作情况。如果输出中没有主要错误，那么您应该准备通过 brew 安装您的第一个软件包。现在我们只需要安装 Python，就完成了：

```py
# brew install python3

```

## Windows

在 Windows 上，手动编译 C Python 软件包通常是一个非常不容易的任务。大多数软件包都是针对 Linux/Unix 系统编写的（OS X 属于 Unix 类别），而 Windows 对开发人员来说只是一个附带的功能。结果是，由于测试软件包的人很少，许多库需要手动安装，因此在 Windows 上编译软件包非常繁琐。因此，除非您确实必须这样做，否则请尽量避免在 Windows 上手动编译 Python 软件包。大多数软件包都可以通过一些搜索获得可安装的二进制下载，并且还有诸如 Anaconda 之类的替代方案，其中包括大多数重要的 C Python 软件包的二进制软件包。

如果您仍然倾向于手动编译 C Python 软件包，那么还有另一种选择，通常是更简单的替代方案。Cygwin 项目（[`cygwin.com/`](http://cygwin.com/)）试图使 Linux 应用程序在 Windows 上原生运行。这通常是一个比让软件包与 Visual Studio 配合工作更容易的解决方案。

如果您确实希望选择 Visual Studio 路径，我想指向第十四章，*C/C++扩展、系统调用和 C/C++库*，其中涵盖了手动编写 C/C++扩展以及有关您的 Python 版本所需的 Visual Studio 版本的一些信息。

# 摘要

随着`pip`和`venv`等包的加入，我觉得 Python 3 已经成为一个完整的包，应该适合大多数人。除了遗留应用程序外，再也没有理由不选择 Python 3 了。2008 年初版的 Python 3 相比于同年发布的成熟的 Python 2.6 版本确实有些粗糙，但在这方面已经发生了很多变化。最后一个重要的 Python 2 版本是 Python 2.7，发布于 2010 年；在软件世界中，这是非常非常长的时间。虽然 Python 2.7 仍然在接受维护，但它将不会获得 Python 3 正在获得的任何惊人的新功能——比如默认的 Unicode 字符串、`dict`生成器（第六章，*生成器和协程-无限，一步一步*）以及`async`方法（第七章，*异步 IO-无需线程的多线程*）。

完成本章后，您应该能够创建一个干净且可重现的虚拟环境，并知道如果 C/C++包的安装失败应该去哪里查找。

这一章最重要的笔记如下：

+   为了创建一个干净简洁的环境，请使用`venv`。如果需要与 Python 2 兼容，请使用`virtualenv`。

+   如果 C/C++包安装失败，请查找有关缺少包含文件的错误。

下一章将介绍 Python 风格指南，重要的规则以及它们的重要性。可读性是 Python 哲学中最重要的方面之一，您将学习编写更干净、更易读的 Python 代码的方法和风格。
