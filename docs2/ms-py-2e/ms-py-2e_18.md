

# 第十八章：打包 – 创建您自己的库或应用程序

到目前为止的章节已经涵盖了如何编写、测试和调试 Python 代码。有了这一切，只剩下了一件事：打包和分发您的 Python 库和应用程序。为了创建可安装的包，我们将使用随 Python 一起打包的 `setuptools` 包。如果您之前创建过包，可能会记得 `distribute` 和 `distutils2`，但非常重要的一点是，这些已经被 `setuptools` 和 `distutils` 取代，您不应再使用它们！

我们有几种包类型和打包方法需要涵盖：

+   使用 PEP 517/518 的 `pyproject.toml` 文件构建新式包

+   使用 `setup.py` 文件进行高级包构建

+   包类型：wheels、eggs、源码包和其他

+   安装可执行文件和自定义 `setuptools` 命令

+   包含 C/C++ 扩展的包

+   在包上运行测试

# 包的介绍

当涉及到打包时，Python 的历史非常混乱。在 Python 存在的几十年里，我们有过 `distutils`、`distutils2`、`distribute`、`buildout`、`setuptools`、`packaging`、`distlib`、`poetry` 以及其他几个库。所有这些项目都是出于改善现状的最好意图而开始的，不幸的是，它们的成功程度各不相同。而且，这还不包括所有不同的包类型，如 wheels、源码包和二进制包，如 eggs、Red Hat `.rpm` 文件和 Windows `.exe`/`.msi` 文件。

好消息是，尽管打包有着复杂的历史，但近几年来情况已经有所改善，并且有了很大的进步。构建包变得更加容易，现在维护一个稳定的项目依赖状态也变得容易可行。

## 包的类型

Python 有（曾经有）许多种包类型，但如今真正重要的只有两种：

+   **Wheels**：这些是小型、可直接安装的 `.zip` 文件，扩展名为 `.whl`，只需解压即可，而无需像构建源包那样进行编译。此外，这些可以是源码或二进制，具体取决于包的类型。

+   **源码包**：这些可以有许多扩展，例如 `.zip`、`.tar`、`.tar.gz`、`.tar.bz2`、`.tar.xz` 和 `.tar.Z`。它们包含安装和构建包所需的 Python/C 等 source 和数据文件。

现在我们将更详细地介绍这些格式。

### Wheels – 新的 eggs

对于纯 Python 包，源码包一直足够使用。然而，对于二进制 C/C++ 包，这却是一个不太方便的选项。C/C++ 源码包的问题在于需要编译，这需要不仅需要一个编译器，还经常需要系统上的头文件和库。而二进制包通常不需要编译器或安装任何其他库，因为所需的库已经包含在包中；Python 本身就足够了。

传统上，Python 使用 `.egg` 格式用于二进制软件包。`.egg` 格式本质上是一个重命名的 `.zip` 文件，包含源代码和元数据，在二进制 `.egg` 文件的情况下还包含编译后的二进制文件。虽然这个想法很棒，但 `.egg` 文件从未真正完全解决问题；一个 `.egg` 文件可能匹配多个环境，但这并不能保证它实际上能在那些系统上运行。

正因如此，引入了 `wheel` 格式（`PEP-427`），这是一种可以包含源代码和二进制文件的软件包格式，可以在 Windows、Linux、macOS X 和其他系统上安装，而无需编译器。

作为额外的好处，轮子可以更快地安装纯 Python 和二进制软件包，因为没有构建、安装或后处理步骤，并且它们更小。安装轮子只需将 `.whl` 文件提取到 Python 环境的 `site-packages` 目录，然后您就完成了。

二进制轮子相对于蛋文件（eggs）解决的最大问题是文件命名。使用轮子（wheels），这一过程简单且一致，因此仅通过文件名就可以检查是否存在兼容的轮子。文件使用以下格式：

```py
{distribution}-{version}[-{build tag}]-{python tag}-{abi tag}-{platform tag}.whl 
```

让我们深入探讨这些字段，看看它们的可能值。但首先，语法。花括号 `{}` 和 `}` 之间的名称是字段，方括号 `[` 和 `]` 之间的括号表示可选字段：

+   `分发`: 软件包的名称，例如 `numpy`、`scipy` 等。

+   `版本`: 该软件包的版本，例如 `1.2.3`。

+   `构建标签`: 作为多个匹配轮子的一个可选的区分器。

+   `python 标签`: Python 版本和平台。对于 CPython 3.10，这将是一个 `cp310`。对于 PyPy 3.10，这将是一个 `pp310`。更多关于这个话题的信息可以在 `PEP-425` 中找到。对于纯 Python 软件包，这可以是支持 Python 3 的 `py3`，或者支持 Python 2 和 Python 3 的通用软件包的 `py2.py3`。

+   `abi 标签`: ABI（应用程序二进制接口）标签表示所需的 Python ABI。例如，`cp310d` 表示启用调试的 CPython 3.10。更多详情可以在 `PEP-3149` 中找到。对于纯 Python 软件包，这通常是 `none`。

+   `平台标签`: 平台标签告诉您它将在哪些操作系统上运行。这可以是 32 位或 64 位 Windows 的 `win32` 或 `win_amd64`。对于 macOS X，这可能类似于 `macosx_11_0_arm64`。对于纯 Python 软件包，这通常是 `any`。

在所有这些不同选项中，你可能已经猜到，为了支持许多平台，你需要许多轮子。这实际上是一件好事，因为它解决了蛋文件（egg files）的一个大问题，即可安装文件并不总是能正常工作。如果你能找到适合你系统的匹配轮子，你可以期待它没有任何问题就能运行。

这些轮子的构建时间是一个缺点。例如，`numpy` 包在写作时拥有 29 个不同的轮子。每个轮子的构建时间在 15 到 45 分钟之间，如果我们按每个轮子平均 30 分钟来计算，那么每个 `numpy` 发布版本就需要 15 小时的构建时间。当然，它们可以并行构建，但这仍然是一个需要考虑的因素。

尽管我们有 29 个不同的轮子可供 `numpy` 使用，但仍有许多平台，如 FreeBSD，没有支持，因此源包的需求仍然存在。

### 源包

源包是所有 Python 包类型中最灵活的。它们包含源代码、构建脚本，以及可能包含许多其他文件，如文档和测试。这些允许你在你的系统上构建和/或编译包。源包可能有多种不同的扩展名，如 `.zip` 和 `.tar.bz2`，但基本上是整个项目目录和相关文件的一个略微精简版本。

由于这些包通常不仅包含直接的源文件，还包含测试和文档，因此它们占用的空间更多，安装速度比轮子慢。以 `numpy` 的源包为例，我目前看到有 1941 个文件，而轮子只包含 710 个文件。这种差异实际上也可能是有用的，因为你可能需要测试文件或文档。如果你希望跳过二进制文件，因为你希望有原始源代码，或者如果你想为你的特定系统进行优化构建，你可以选择通过告诉 `pip` 跳过二进制文件来安装源文件。

从源代码而不是二进制文件安装包可以导致二进制文件更小和/或更快，因为它只会链接到系统上可用的库，而不是通用库。

连接到 PostgreSQL 数据库的 `psycopg` 包是这方面的一个好例子。它提供了三种可能的安装选项，通过 `pip` 安装，优先级从高到低：

+   `psycopg[c]`: 用于本地构建和编译的 Python 和 C 源代码

+   `psycopg[binary]`: Python 源代码和预编译的二进制文件

+   `psycopg`: 仅 Python 源代码；在这种情况下，你需要在你的系统上安装 `libpq` 库，它通过 `ctypes` 访问

要安装而不使用任何预编译的二进制文件：

```py
$ pip3 install --no-binary ... 
```

由于源包附带构建脚本，仅安装本身就可能存在风险。虽然轮子只会解压而不会运行任何内容，但源包在安装过程中会执行构建脚本。曾经有一个名为俄罗斯轮盘赌的包在 PyPI 上，安装时会有 1/6 的几率删除系统上的文件，以此来展示这种方法的危险性。

我个人认为，在安装过程中执行构建脚本的安全风险远不如在计划安装之前对包进行审查重要。无论您是否实际执行代码，在您的系统上安装可能有害的包都是一件坏事。

## 包工具

那么，我们今天还需要和使用的安装工具是什么？

`distribute`、`distutils`和`distutils2`包已被`setuptools` largely 取代。要安装基于`setup.py`的源包，通常需要`setuptools`，而`setuptools`与`pip`捆绑在一起，因此您应该已经具备了这个要求。当涉及到安装 wheel 时，需要`wheel`包；这也方便地捆绑在`pip`中。在大多数系统上，这意味着一旦安装了 Python，您就应该拥有安装额外包所需的一切。

不幸的是，Ubuntu Linux 发行版是一个值得注意的例外，它附带了一个损坏的 Python 安装，缺少`pip`和`ensurepip`命令。这可以通过单独安装`pip`来修复：

```py
$ apt install python3-pip 
```

如果这不起作用，您可以通过运行`get-pip.py`脚本安装`pip`：[`bootstrap.pypa.io/get-pip.py`](https://bootstrap.pypa.io/get-pip.py)

由于`setuptools`和`pip`在过去几年中已经得到了相当多的开发，因此无论如何升级这些包都是一个好主意：

```py
$ pip3 install --upgrade pip setuptools wheel 
```

现在我们已经安装了所有先决条件，我们可以继续构建自己的包。

# 包版本控制

虽然有众多版本控制方案可用，但许多 Python 包以及 Python 本身都使用 PEP-440 进行版本规范。

有些人坚持使用稍微严格一点的版本，称为**语义版本控制**（**SemVer**），但两者在很大程度上是兼容的。

简短而简化的解释是使用如 `1.2` 或 `1.2.3` 这样的版本号。例如，查看版本 `1.2.3`：

+   `1` 是主版本，表示破坏 API 的不兼容更改

+   `2` 是次要版本，表示向后兼容的功能添加

+   `3` 是补丁版本，用于向后兼容的 bug 修复

在主版本的情况下，一些库选择使版本非连续，并使用日期作为版本，例如 `2022.5`。

预发布版本，如 alpha 和 beta，可以通过次要版本中的字母来指定。选项有 `a` 表示 alpha，`b` 表示 beta，`rc` 表示发布候选。例如，对于 `1.2 alpha 3`，结果是 `1.2a3`。

在语义版本控制的情况下，这通过在末尾添加预发布标识符来处理，例如 `1.2.3-beta` 或 `1.2.3-beta.1` 用于多个 beta 版本。

最后，PEP-440 允许使用后发布版本，例如使用 `1.2.post3` 代替 `1.2.3` 进行次要 bug 修复，以及类似地使用 `1.2.dev2` 进行开发版本。

无论你使用哪种版本控制系统，在开始你的项目之前都要仔细考虑。不考虑未来可能会在长期内造成问题。一个例子是 Windows。一些应用程序在支持 Windows 10 时遇到了麻烦，因为版本号的字母顺序将 Windows 10 放在 Windows 8 之下（毕竟，1 小于 8）。

# 构建包

Python 包传统上使用包含（部分）构建脚本的 `setup.py` 文件进行构建。这种方法通常依赖于 `setuptools`，并且仍然是大多数包的标准，但如今我们有更简单的方法可用。如果你的项目要求不高，你可以使用一个小的 `pyproject.toml` 文件，这可能会更容易维护。

让我们尝试这两种方法，看看构建一个基本的 Python 包有多容易。

## 使用 pyproject.toml 进行打包

`pyproject.toml` 文件允许根据所使用的工具轻松地进行打包。它是在 2015 年通过 `PEP-517` 和 `PEP-518` 引入的。这种方法是为了改进 `setup.py` 文件而创建的，通过引入构建时依赖项、自动配置，并使其更容易以 DRY（不要重复自己）的方式工作。

TOML 代表“Tom 的明显、最小化语言”，在某种程度上与 YAML 和 INI 文件相似，但更简单。由于它是一种如此简单的语言，它可以很容易地包含在像 `pip` 这样的包中，几乎没有开销。这使得它在需要扁平结构且不需要复杂功能（如继承和包含）的场景中非常完美。

在我们继续之前，我们需要澄清一些事情。当我们谈论 `setup.py` 文件时，我们通常实际上是在谈论 `setuptools` 库。与 Python 打包在一起的 `distutils` 库也可以使用，但由于 `pip` 依赖于 `setuptools`，它通常是更好的选择；它具有更多功能，并且与 `pip` 一起更新，而不是与你的 Python 安装一起更新。

类似于 `setup.py` 通常意味着 `setuptools`，使用 `pyproject.toml` 我们也有多个库可用于构建和管理 `PEP-517` 风格的包。这种方法创建标准并依赖社区项目进行实现，在 Python 中过去已经工作得相当好，这使得它是一个明智的选择。这种方法的例子是 Python 网络服务器网关接口（WSGI），它作为 `PEP-333` 引入，目前有几种优秀的实现可用。

`PEP-517` 的参考解决方案是 `pep517` 库，它虽然可用但功能相当有限。另一个选择是 `build` 库，由 Python 包权威机构（PyPA）维护，它也维护了 Python 包索引（PyPI）。虽然这个库可用，但在功能方面也相当有限，我并不推荐使用。

在我看来，最好的选择无疑是 `poetry` 工具。`poetry` 工具不仅为你处理包的构建，还负责：

+   并行快速安装依赖项

+   创建虚拟环境

+   为可运行的脚本创建易于访问的入口点

+   通过指定智能版本约束（例如，主版本和次版本，将在本章后面详细说明）来管理依赖项

+   构建包

+   发布到 PyPI

+   使用 `pyenv` 处理多个 Python 版本

对于大多数情况，`pyproject.toml` 可以完全替代传统的 `setup.py` 文件，但也有一些情况你需要一些额外的工具。

在构建 C/C++ 扩展和其他情况时，你可能需要一个 `setup.py` 文件或以其他方式指定如何构建扩展。这个选项之一是使用 `poetry` 工具并将构建脚本添加到 `pyproject.toml` 工具中。我们将在关于 C/C++ 扩展的部分进一步讨论这个问题。

可编辑安装（即 `pip install -e ...`）直到 2021 年才成为可能，但已被 PEP-660 解决。

### 创建一个基本包

让我们从使用 `poetry` 在当前目录中创建一个基本的 `pyproject.toml` 文件开始：

```py
$ poetry new .
Created package t_00_basic_pyproject in . 
```

由于我们的父目录名为 `t_00_basic_pyproject`，`poetry` 自动将其作为新项目名称。或者，你也可以执行 `poetry new some_project_name`，它将为你创建一个目录。

`poetry` 命令为我们创建了以下文件：

```py
README.rst
pyproject.toml
t_00_basic_pyproject
t_00_basic_pyproject/__init__.py
tests
tests/__init__.py
tests/test_t_00_basic_pyproject.py 
```

这是一个非常简单的模板，包含足够的内容来启动你的项目。`t_00_basic_pyproject/__init__.py` 文件包含版本（默认为 `0.1.0`）和 `tests/test_t_00_basic_pyproject.py` 文件作为示例测试来测试这个版本。然而，更有趣的部分是 `pyproject.toml` 文件，所以现在让我们看看它：

```py
[tool.poetry]
name = "T_00_basic_pyproject"
version = "0.1.0"
description = ""
authors = ["Rick van Hattem <Wolph@wol.ph>"]

[tool.poetry.dependencies]
python = "³.10"

[tool.poetry.dev-dependencies]
pytest = "⁵.2"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api" 
```

如你所见，`poetry` 已经自动配置了名称和版本。它还通过查看我的系统上的 `git` 配置将我添加为作者。你可以通过运行以下命令轻松配置：

```py
$ git config --global user.email 'your@email.tld'
$ git config --global user.name 'Your Name' 
```

接下来，我们可以看到它自动将 Python 3.10 设置为要求，并添加 `pytest` 5.2 作为开发依赖项。对于构建包，它添加了 `poetry-core` 作为依赖项，这是 `poetry` 的 `setuptools` 等价物。

### 安装开发包

对于开发目的，我们通常以 **可编辑模式** 安装包。在可编辑模式下，包不会被复制到你的 `site-packages` 目录，而是创建到你的源目录的链接，因此所有对源目录的更改都会立即生效。如果没有可编辑模式，每次你对包进行更改时，你都需要执行 `pip install`，这对于开发来说非常不方便。

使用 `pip`，你可以通过以下命令以可编辑模式安装：

```py
$ pip3 install --editable <package-directory> 
```

对于在当前目录中安装，你可以使用 `.` 作为目录，结果如下：

```py
$ pip3 install --editable . 
```

使用 `poetry` 命令，以可编辑模式安装（或对于 `poetry` 的旧版本，可能是类似的功能）会自动发生。它还为我们处理虚拟环境的创建，同时使用 `pyenv` 来处理 `pyproject.toml` 文件中指定的 Python 版本。要安装包及其所有依赖项，您只需运行：

```py
$ poetry install 
```

如果您希望直接访问创建的虚拟环境中的所有命令，您可以使用：

```py
$ poetry shell
(name-of-your-project) $ 
```

`poetry shell` 命令会启动一个新的 shell，将当前项目的名称添加到您的命令行前缀中，并将虚拟环境脚本目录添加到您的 `PATH` 环境变量中。这会导致 `python` 和 `pip` 等命令在您的虚拟环境中执行。

### 添加代码和数据

在基本示例中，我们没有指定哪个目录包含源代码，或者 `t_00_basic_pyproject` 目录必须包含在目录中。默认情况下，这会隐式处理，但我们可以修改 `pyproject.toml` 文件来显式包含目录或文件模式作为 Python 源：

```py
[tool.poetry]
...
packages = [
    {include="T_00_basic_pyproject"},
    {include="some_directory/**/*.py"},
] 
```

注意，添加 `packages` 参数会禁用自动检测包，因此您需要在此处指定所有包含的包。

要包含其他数据文件，例如文档，我们可以使用 `include` 和 `exclude` 参数。`exclude` 参数会覆盖由 `packages` 参数包含的文件：

```py
[tool.poetry]
...
include = ["CHANGELOG.rst"]
exclude = ["T_00_basic_pyproject/local.py"] 
```

对于一个基本项目，您可能不需要查看这个。但，就像往常一样，明确总是优于隐式，所以我建议您快速查看一下，以防止意外地将错误的文件包含在您的包中。

### 添加可执行命令

一些包，如 `numpy`，仅是库，这意味着它们被导入但没有可运行的命令。其他包，如 `pip` 和 `poetry`，包含可运行的脚本，在安装过程中作为新命令安装。毕竟，当 `poetry` 包被安装后，您可以从 shell 中使用 `poetry` 命令。

要创建我们自己的命令，我们需要指定新命令的名称、模块和相应的函数，这样 `poetry` 就会知道要运行什么。例如：

```py
[tool.poetry.scripts]
our_command = 'T_00_basic_pyproject.main:run' 
```

这将执行名为 `T_00_basic_pyproject/main.py` 的文件中的 `run()` 函数。安装包后，您可以从您的 shell 中执行 `our_command` 来运行脚本。在 `poetry` 开发期间，您可以使用 `poetry run our_command`，这将自动在 `poetry` 创建的虚拟环境中运行命令。

### 管理依赖项

我们创建的 `pyproject.toml` 文件已经为开发和构建添加了一些要求，但您可能还想向项目中添加其他依赖项。例如，如果我们想添加进度条，我们可以运行以下命令：

```py
$ poetry add progressbar2
Using version ⁴.0.0 for progressbar2
... 
```

这会自动为我们安装 `progressbar2` 包，并将其添加到 `pyproject.toml` 文件中，如下所示：

```py
[tool.poetry.dependencies]
...
progressbar2 = "⁴.0.0" 
```

此外，`poetry` 将创建或更新一个 `poetry.lock` 文件，其中包含已安装的确切包版本，因此在新环境中可以轻松地重现安装。在上面的例子中，我们只是告诉 `poetry` 安装 `progressbar2` 的任何版本，这导致 `poetry` 将版本要求设置为 `⁴.0.0`，但我们也可以放宽这些要求，这样 `poetry` 将自动安装包的最新补丁、小版本或主要版本。 

默认情况下，`poetry` 将将依赖项添加到 `[tool.poetry.dependencies]` 部分，但您也可以使用 `--dev` 或 `-D` 命令行参数将它们添加为开发依赖项。如果您想添加其他类型的依赖项，例如 `build-system` 依赖项或测试依赖项，那么您需要手动编辑 `pyproject.toml` 文件。

版本指定符期望与 SemVer 兼容的版本，并按以下方式工作。为了允许更新非主要版本，您可以使用连字符 (`^`)。它查看第一个非零数字，因此 `¹.2.3` 的行为与 `⁰.1.2` 不同，如下所示：

+   `¹.2.3` 表示 `>=1.2.3` 且 `<2.0.0`

+   `¹.2` 表示 `>=1.2.0` 且 `<2.0.0`

+   `¹` 表示 `>=1.0.0` 且 `<2.0.0`

+   `⁰.1.2` 表示 `>=0.1.2` 且 `<0.2.0`

接下来是波浪号 (`~`) 要求，它们指定了最小版本，但允许进行小版本更新。它们比连字符版本简单一些，实际上指定了数字应该从哪里开始：

+   `~1.2.3` 表示 `>=1.2.3` 且 `<1.3.0>`

+   `~1.2` 表示 `>=1.2.0` 且 `<1.3.0>`

+   `~1` 表示 `>=1.0.0` 且 `<2.0.0`。请注意，上述两个选项都允许进行小版本更新，而这是唯一允许进行主要版本更新的选项。

使用星号 (`*`) 也可以进行通配符要求：

+   `1.2.*` 表示 `>=1.2.0` 且 `<1.3.0`

+   `1.*` 表示 `>=1.0.0` 且 `<2.0.0`

版本控制系统与 `requirements.txt` 中使用的格式兼容，允许使用如下版本：

+   `>= 1.2.3`

+   `>= 1.2.3, <1.4.0`

+   `>= 1.2.3, <1.4.0, != 1.3.0`

+   `!= 1.5.0`

我个人更喜欢这种最后的语法，因为它很清晰，不需要太多的先验知识，但您当然可以使用您喜欢的任何一种。默认情况下，`poetry` 在添加依赖项时会使用 `¹.2.3` 格式。

现在，假设我们有一个类似 `progressbar2 = "³.5"` 的要求，并且我们在 `poetry.lock` 文件中有 `3.5.0` 版本。如果我们运行 `poetry install`，它将安装确切的 `3.5.0` 版本，因为我们知道这个版本是好的。

作为开发者，您可能希望将那个依赖项更新到新版本，以便测试新版本是否也能正常工作。这也是我们可以向 `poetry` 提出的要求：

```py
$ poetry update
Updating dependencies
...
Package operations: 0 installs, 1 update, 0 removals
  • Updating progressbar2 (3.5.0 -> 3.55.0) 
```

现在 `poetry` 将会自动在 `pyproject.toml` 的约束范围内升级包并更新 `poetry.lock` 文件。

### 构建包

现在我们已经配置了 `pyproject.toml` 文件和所需的依赖项，我们可以使用 `poetry` 构建包。幸运的是，这非常简单。构建包只需要一个命令：

```py
$ poetry build
Building T_00_basic_pyproject (0.1.0)
  - Building sdist
  - Built T_00_basic_pyproject-0.1.0.tar.gz
  - Building wheel
  - Built T_00_basic_pyproject-0.1.0-py3-none-any.whl 
```

只需一个命令，`poetry` 就为我们创建了一个源包和一个 wheel。所以，如果您一直在关注，您会意识到我们实际上可以用两个命令创建和构建一个包：`poetry new` 和 `poetry build`。

### 构建 C/C++ 扩展

在我们开始本节之前，我需要提供一点免责声明。截至撰写本文时（2021 年底），构建 C/C++ 扩展并不是 `poetry` 的稳定和受支持的功能，这意味着它将来可能会被不同的机制所取代。然而，目前有一个可用的解决方案用于构建 C/C++ 扩展，并且未来的版本可能会以类似的方式工作。

如果您现在正在寻找一个稳定且受良好支持的解决方案，我建议您选择基于 `setup.py` 的项目，这将在本章后面进行介绍。

我们需要首先修改我们的 `pyproject.toml` 文件，并在 `[tool.poetry]` 部分添加以下行：

```py
build = "build_extension.py" 
```

如果您希望使用 PyPA 构建命令，请确保不要将文件命名为 `build.py`。

一旦完成，当我们运行 `poetry build` 时，`poetry` 将会执行 `build_extension.py` 文件，因此现在我们需要创建 `build_extension.py` 文件，以便 `setuptools` 为我们构建扩展：

```py
import pathlib
import setuptools

# Get the current directory
PROJECT_PATH = pathlib.Path(__file__).parent

# Create the extension object with the references to the C source
sum_of_squares = setuptools.Extension('sum_of_squares', sources=[
    # Get the relative path to sum_of_squares.c
    str(PROJECT_PATH / 'sum_of_squares.c'),
])

def build(setup_kwargs):
    setup_kwargs['ext_modules'] = [sum_of_squares] 
```

此脚本基本上与您会放入 `setup.py` 文件中的内容相同。原因是它实际上是在注入相同的函数调用。如果您仔细查看 `build()` 函数，您会看到它更新了 `setup_kwargs` 并在该函数中设置了 `ext_modules` 项。该参数直接传递给 `setuptools.setup()` 函数。本质上，我们只是在模拟使用 `setup.py` 文件。

注意，对于我们的 C 文件，我们使用了来自 *第十七章*，*C/C++ 扩展、系统调用和 C/C++ 库* 的 `sum_of_squares.c` 文件。您会看到其余的代码在很大程度上与我们在 *第十七章* 中使用的 `setup.py` 文件相似。

当我们执行 `poetry build` 命令时，`poetry` 将会内部调用 `setuptools` 并构建二进制轮：

```py
$ poetry build
Building T_01_pyproject_extensions (0.1.0)
  - Building sdist
  - Built T_01_pyproject_extensions-0.1.0.tar.gz
  - Building wheel
running build
running build_py
creating build
...
running build_ext
building 'sum_of_squares' extension
... 
```

这样，我们就完成了。我们现在有一个包含构建的 C 扩展的 wheel 文件。

## 使用 setuptools 和 setup.py 或 setup.cfg 打包

`setup.py` 文件是构建 Python 包的传统方法，但仍然被广泛使用，并且是一种创建包的非常灵活的方法。

*第十七章* 已经向我们展示了构建扩展时的几个示例，但让我们重申并回顾一下实际上最重要的部分做了什么。您将在整个章节中使用的核心函数是 `setuptools.setup()`。

Python 附带的标准`distutils`包在大多数情况下也足够使用，但我仍然推荐使用`setuptools`。`setuptools`包具有许多`distutils`所缺乏的出色功能，并且由于它包含在`pip`中，几乎所有的 Python 环境都会提供`setuptools`。

在我们继续之前，确保您拥有`pip`、`wheel`和`setuptools`的最新版本总是一个好主意：

```py
$ pip3 install -U pip wheel setuptools 
```

`setuptools`和`distutils`包在过去几年中发生了显著变化，2014 年之前编写的文档/示例很可能已经过时。请小心不要实现已弃用的示例，并且我建议跳过任何使用`distutils`的文档/示例。

作为`setup.py`文件的替代或补充，您还可以使用`setup.cfg`文件来配置所有元数据。这使用 INI 格式，对于不需要（或不想）Python 语法开销的简单元数据来说可能更方便一些。

您甚至可以选择仅使用`setup.cfg`并跳过`setup.py`；然而，如果您这样做，您将需要一个单独的构建工具。对于这些情况，我建议安装 PyPA 的`build`库：

```py
$ pip3 install build
... 
```

### 创建一个基本包

现在我们已经具备了所有先决条件，让我们使用`setup.py`文件创建一个包。虽然最基础的`setuptools.setup()`调用在技术上不需要任何参数，但如果您计划将包发布到 PyPI，您确实应该包括至少`name`、`version`、`packages`、`url`、`author`和`author_email`字段。以下是一个包含这些字段的非常基础的示例：

```py
import setuptools

if __name__ == '__main__':
    setuptools.setup(
        name='T_02_basic_setup_py',
        version='0.1.0',
        packages=setuptools.find_packages(),
        url='https://wol.ph/',
        author='Rick van Hattem',
        author_email='wolph@wol.ph',
    ) 
```

作为将这些配置为`setup()`参数的替代方案，您还可以使用一个`setup.cfg`文件，它使用 INI 格式，但实际工作方式与之前相同：

```py
[metadata]
name = T_03_basic_setup_cfg
version = 0.1.0
url='https://wol.ph/',
author='Rick van Hattem',
author_email='wolph@wol.ph',

[options]
packages = find: 
```

`setup.cfg`的主要优势是它比`setup.py`文件更简洁、更简单的文件格式。例如，看看`packages`部分；`setuptools.find_packages()`比`find:`要详细得多。

缺点是您需要将一个`setup.cfg`文件与一个`setup.py`或`pyproject.toml`文件配对，才能构建它。仅凭`setup.cfg`本身不足以构建一个包，这使得`setup.cfg`成为将元数据与设置代码分离的一种既简洁又清晰的方式。此外，许多库如`pytest`和`tox`都原生支持`setup.cfg`文件，因此您也可以通过该文件进行配置。

为了将`setup.cfg`和/或`setup.py`与`pyproject.toml`文件配对，我们需要将这些行添加到`pyproject.toml`文件中：

```py
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta" 
```

注意，仅凭一个`pyproject.toml`文件本身并不能为您提供`poetry`支持；为了获得`poetry`支持，您需要添加一个`[tool.poetry]`部分。

### 安装开发版本的包

要安装用于本地开发的包，我们可以再次使用 `-e` 或 `--editable` 标志，如本章中 `poetry` 部分所述。这将在你的源目录和 `site-packages` 目录之间创建一个链接，以便实际使用源文件，而不是让 `setuptools` 将所有源文件复制到 `site-packages` 目录。

简而言之，从项目目录中，你可以使用 `setup.py` 文件：

```py
$ python3 setup.py develop 
```

或者 `pip`：

```py
$ pip3 install -e . 
```

### 添加包

在基本示例中，你可以看到我们使用 `find_packages()` 作为包的参数。这个函数会自动检测所有源目录，通常作为默认值是足够的，但有时你需要更多的控制。`find_packages()` 函数还允许你添加 `include` 或 `exclude` 参数，如果你希望从包中排除测试和其他文件，例如这样：

```py
setuptools.find_packages(
    include=['a', 'b', 'c.*'],
    exclude=['a.excluded'],
) 
```

`find_packages()` 的参数也可以翻译到具有不同语法的 `setup.cfg` 文件中：

```py
[options]
packages = find:

[options.packages.find]
include =
    a
    b
    c.*
exclude = a.excluded 
```

### 添加包数据

在大多数情况下，你可能不需要包含包数据，例如测试数据或文档文件，但有些情况下需要额外的文件。例如，Web 应用程序可能附带 `html`、`javascript` 和 `css` 文件。

在包含与你的包一起的额外文件方面，有几个不同的选项。首先，了解你的源包默认包含哪些文件是很重要的：

+   包目录中的 Python 源文件及其所有子目录

+   `setup.py`、`setup.cfg` 和 `pyproject.toml` 文件

+   如果有的话，包括自述文件，例如 `README.rst`、`README.txt` 和 `README.md`

+   包含包名称、版本、入口点、文件哈希等元数据文件的元数据文件

对于 Python 轮子（wheels），列表甚至更短，默认情况下只打包 Python 源文件和元数据文件。

这意味着如果我们想包含其他文件，我们需要指定这些文件需要被添加。我们有两种不同的选项来将其他类型的数据添加到我们的包中。

首先，我们可以将 `include_package_data` 标志作为 `setup()` 的参数启用：

```py
 setuptools.setup(
        ...
        include_package_data=True,
    ) 
```

一旦启用该标志，我们就可以在 `MANIFEST.in` 文件中指定我们想要的文件模式。此文件包含包含、排除和其他模式的模式。`include` 和 `exclude` 命令使用模式进行匹配。这些模式是 glob-style 模式（有关文档，请参阅 `glob` 模块：[`docs.python.org/3/library/glob.html`](https://docs.python.org/3/library/glob.html)），对于 `include` 和 `exclude` 命令都有三种变体：

+   `include`/`exclude`：这些命令只针对给定的路径，不涉及其他

+   `recursive-include`/`recursive-exclude`：这些命令与 `include`/`exclude` 命令类似，但会递归地处理给定的路径

+   `global-include`/`global-exclude`：对这些要非常小心，因为它们会在源树中的任何位置包含或排除这些文件

除了 `include`/`exclude` 命令之外，还有两个其他命令：`graft` 和 `prune` 命令，它们包括或排除给定目录下的所有目录。这对于测试和文档可能很有用，因为它们可以包含非标准文件。除了这些示例之外，几乎总是更好的做法是明确包含你需要的文件，并忽略所有其他文件。以下是一个示例 `MANIFEST.in` 文件：

```py
# Include all documentation files
include-recursive *.rst
include LICENSE

# Include docs and tests
graft tests
graft docs

# Skip compiled python files
global-exclude *.py[co]

# Remove all build directories
prune docs/_build
prune build
prune dist 
```

或者，我们可以使用 `package_data` 和 `exclude_package_data` 参数，并将它们添加到 `setup.py` 中：

```py
 setuptools.setup(
        ...
        package_data={
            # Include all documentation files
            '': ['*.rst'],

            # Include docs and tests
            'tests': ['*'],
            'docs': ['*'],
        },
        exclude_package_data={
            '': ['*.pyc', '*.pyo'],
            'dist': ['*'],
            'build': ['*'],
        },
    ) 
```

自然，这些也有等效的 `setup.cfg` 格式：

```py
[options]
...
include_package_data=True,

[options.package_data]
# Include all documentation files
* = *.rst

# Include docs and tests
tests = *
docs = *

[options.exclude_package_data]
* = *.pyc, *.pyo
dist = *
build = * 
```

注意，这些参数使用 `package_data` 而不是 `data` 是有原因的。所有这些都需要你使用一个包。这意味着数据只有在它位于一个合适的 Python 包（换句话说，如果它包含一个 `__init__.py`）中时才会被包含。

你可以选择你喜欢的任何格式和方法。

### 管理依赖项

当你使用 `setup.py` 或 `setup.cfg` 文件时，你不会得到 `poetry` 提供的简单依赖项管理。添加新的依赖项并不困难，除了你需要添加要求并自己安装包，而不是在一个命令中完成所有操作。

就像 `pyproject.toml` 一样，你可以声明多种类型的依赖项：

+   `[build-system] requires`：这是构建项目所需的依赖项。这些通常是 `setuptools` 和 `wheel`，对于基于 `setuptools` 的包；对于 `poetry`，这将是一个 `poetry-core`。

+   `[options] install_requires`：这些是运行包所需的依赖项。例如，像 `pandas` 这样的项目将需要 `numpy`。

+   `[options.extras_require] NAME_OF_EXTRA`：如果你的项目在特定情况下有可选的依赖项，额外的依赖项可以帮助。例如，要安装具有 `redis` 支持的 `portalocker`，你可以运行此命令：

```py
$ pip3 install "portalocker[redis]" 
```

如果你有过创建包的经验，你可能想知道为什么这里没有显示 `tests_require`。原因是自从添加了 `extras_require` 之后，就不再真正需要它了。你可以简单地添加一个额外的 `tests` 和 `docs` 的要求。

下面是一个向 `setup.py` 文件添加一些要求的示例：

```py
setuptools.setup(
    ...
    setup_requires=['pytest-runner'],
    install_requires=['portalocker'],
    extras_require={
        'docs': ['sphinx'],
        'tests': ['pytest'],
    },
) 
```

在 `setup.cfg` 文件中，这是等效的：

```py
[build-system]
requires =
    setuptools
    wheel

[options]
install_requires =
    portalocker

[options.extras_require]
docs = sphinx
tests = pytest 
```

### 添加可执行命令

就像基于 `pyproject.toml` 的项目一样，我们也可以使用 `setup.py` 或 `setup.cfg` 文件来指定可执行命令。要添加一个类似于我们可以运行 `pip` 或 `ipython` 命令的基本可执行命令，我们可以在 `setup.py` 文件中添加 `entry_points`：

```py
setuptools.setup(
...
    entry_points={
        'console_scripts': [
            'our_command = T_02_basic_setup_py.main:run',
        ],
    }, 
```

或者 `setup.cfg` 的等效格式：

```py
[options.entry_points]
console_scripts =
    our_command = T_03_basic_setup_cfg.main:run 
```

一旦安装了这个包，你就可以从你的 shell 中运行 `our_command`，就像你运行 `pip` 或 `ipython` 命令一样。

从上面的示例中，您可能会想知道我们是否有除了`console_scripts`之外的其他选项，答案是肯定的。一个例子是`distutils.commands`，它可以用来向`setup.py`添加额外的命令。通过在那个命名空间中添加一个命令，您可以这样做：

```py
$ python3 setup.py our_command 
```

然而，这种行为最突出的例子是`pytest`库。`pytest`库使用这些入口点来自动检测与`pytest`兼容的插件。我们可以轻松地创建自己的等效版本：

```py
[options.entry_points]
our.custom.plugins =
    some_plugin = T_03_basic_setup_cfg.some_plugin:run 
```

一旦安装了这样的软件包，您可以通过`importlib`查询它们，如下所示：

```py
>>> from importlib import metadata

>>> metadata.entry_points()['our.custom.plugins']
[EntryPoint(name='some_plugin', value='...some_plugin:run', ...] 
```

这是一个非常有用的功能，可以自动在库之间注册插件。

### 构建软件包

要实际构建软件包，我们有几种选择。如果可用，我个人会使用`setup.py`文件：

```py
$ python3 setup.py build sdist bdist_wheel
running build
...
creating 'dist/T_02_basic_setup-0.1.0-py3-none-any.whl' and adding ... 
```

如果您只有`setup.cfg`和`pyproject.toml`可用，您需要安装一个包来调用构建器。除了`poetry`之外，PyPA 还提供了一个名为`build`的工具，用于创建构建软件包的隔离环境：

```py
$ python3 -m build
* Creating venv isolated environment...
...
Successfully built T_02_basic_setup-0.1.0.tar.gz and T_02_basic_setup-0.1.0-py3-none-any.whl 
```

轮和源包都写入到`dist`目录，它们已准备好发布。

# 发布软件包

现在我们已经构建了软件包，我们需要将它们实际发布到 PyPI。我们可以使用几种不同的选项，但让我们先讨论一些可选的软件包元数据。

## 添加 URL

我们的`setup.py`和`setup.cfg`文件已经包含了一个`url`参数，该参数将用作 PyPI 上的软件包主页。然而，我们可以通过配置`project_urls`设置添加更多相关的 URL，这是一个名称/URL 对的任意映射。对于`settings.py`：

```py
 setuptools.setup(
        ...
        project_urls=dict(
            docs='https://progressbar-2.readthedocs.io/',
        ),
    ) 
```

或者对于`settings.cfg`：

```py
[options]
project_urls=
    docs=https://progressbar-2.readthedocs.io/ 
```

类似地，对于使用`poetry`的`pyproject.toml`：

```py
[tool.poetry.urls]
docs='https://progressbar-2.readthedocs.io/' 
```

## PyPI trove 分类器

为了提高您的软件包在 PyPI 上的曝光度，添加一些分类器可能很有用。一些分类器，如 Python 版本和许可证，会自动为您添加，但指定您正在编写的库或应用程序的类型可能很有用。

对于对您的软件包感兴趣的人，有许多有用的分类器示例：

+   **开发状态**：这可以从“规划”到“成熟”不等，告诉用户应用程序是否已准备好投入生产。当然，人们对什么是稳定或测试版的定义各不相同，所以这通常只被视为一个提示。

+   **框架**：您正在使用或扩展的框架。这可能包括 Jupyter、IPython、Django、Flask 等等。

+   **主题**：这是否是一个软件开发包、科学、游戏等等。

可以在 PyPI 网站上找到完整的分类器列表：[`pypi.org/classifiers/`](https://pypi.org/classifiers/)

## 上传到 PyPI

将您的软件包上传并发布到 PyPI 非常简单。也许太简单了，正如我们将在`twine`的案例中看到的那样。

在我们开始之前，为了防止你意外地将你的包发布到 PyPI，你应该了解 PyPI 测试服务器：[`packaging.python.org/en/latest/guides/using-testpypi/`](https://packaging.python.org/en/latest/guides/using-testpypi/)

在 `poetry` 的情况下，我们可以这样配置测试仓库：

```py
$ poetry config repositories.testpypi https://test.pypi.org/simple/
$ poetry config pypi-token.testpypi <token> 
```

首先，如果你使用 `poetry`，那么它就像这样简单：

```py
$ poetry publish --repository=testpypi 
```

如果你没有使用 `poetry` 并且不想使用兼容 `poetry` 的 `pyproject.toml`，你需要一个不同的解决方案。PyPA 的官方解决方案是使用由 PyPA 维护的 `twine` 工具。在你使用 `python3 -m build` 构建包之后，你可以使用 `twine` 进行上传：

警告！如果你已经认证，此命令将立即注册并上传包到 `pypi.org`。这就是为什么添加了 `--repository testpypi` 来上传到测试 PyPI 服务器的原因。如果你省略该参数，你将立即将你的包发布到 PyPI。

```py
$ twine upload --repository testpypi dist/* 
```

在你开始将你的包发布到 PyPI 之前，你应该问自己几个问题：

+   包是否处于工作状态？

+   你计划支持这个包吗？

不幸的是，PyPI 仓库充满了声称有可用包名的人的空包。

# C/C++ 扩展

上一章和本章前面的部分已经简要介绍了 C/C++ 组件的编译，但这个主题足够复杂，足以拥有一个单独的部分，提供更深入的说明。

为了方便，我们将从一个基本的 `setup.py` 文件开始，该文件编译一个 C 扩展：

```py
import setuptools

sum_of_squares = setuptools.Extension('sum_of_squares', sources=[
    # Get the relative path to sum_of_squares.c
    str(PROJECT_PATH / 'sum_of_squares.c'),
])

setuptools.setup(
    name='T_04_C_extensions',
    version='0.1.0',
    ext_modules=[sum_of_squares],
) 
```

在开始使用这些扩展之前，你应该学习以下 `setup.py` 命令：

+   `build_ext`：此命令构建 C/C++ 扩展，以便在包以开发/可编辑模式安装时使用。

+   `clean`：这个命令会清理 `build` 命令的结果。这通常不是必需的，但有时检测需要重新编译以工作的文件是不正确的。如果你遇到奇怪或意外的错误，请先尝试清理项目。

你可以选择使用 PyPA 的 `build` 命令来代替 `python3 setup.py build_ext`，但这并不是一个方便的开发选项。如果你使用 `python3 setup.py build`，你可以重用你的 `build` 目录并选择性地构建你的 C/C++ 扩展，这为你节省了大量时间，特别是对于较大的 C/C++ 模块。PyPA 的 `build` 命令旨在生成干净、可用于生产的包，强烈推荐用于部署和发布，但不推荐用于开发。

## 正规的 C/C++ 扩展

`setuptools.Extension` 类告诉 `setuptools`，名为 `sum_of_squares` 的模块使用源文件 `sum_of_squares.c`。这只是扩展的最简单版本——一个名称和一组源文件——但通常你需要的不仅仅是 C 文件，还需要来自其他库的一些头文件。

一个典型的例子是用于图像处理的 `pillow` 库。当库正在构建时，它会自动检测系统上可用的库，并基于此添加扩展。对于 `.jpeg` 支持，你需要安装 `libjpeg`；对于 `.tiff` 图像，你需要 `libtiff`；等等。由于这些扩展包括二进制库，因此需要一些额外的编译标志和 C 头文件。基本的 PIL 模块本身并不太复杂，但 `setup.py` 文件充满了自动检测代码，用于检测哪些 `libs`（库）可用，以及匹配的 C 宏定义以启用这些库。

C 中的宏是预处理器指令。这些指令在真正的编译步骤发生之前执行，这使得它们非常适合条件代码。例如，你可以有一个依赖于 `DEBUG` 标志的调试代码的条件块：

```py
#ifdef DEBUG
/* your debug code here */
#endif 
```

如果设置了 `DEBUG`，代码将成为编译二进制的一部分。如果没有设置此标志，代码块将永远不会出现在最终的二进制文件中。这导致二进制文件更小、运行更快，因为这些条件是在编译时而不是在运行时发生的。

这里是一个来自较旧版本的 `pillow` `setup.py` 文件的 `Extension` 部分示例：

```py
exts = [(Extension("PIL._imaging", files, libraries=libs,
    define_macros=defs))] 
```

新的版本相当不同，`pillow` 项目的 `setup.py` 文件目前有超过 1,000 行。`freetype` 扩展也有类似之处：

```py
if feature.freetype:
    exts.append(Extension(
        "PIL._imagingft", ["_imagingft.c"], libraries=["freetype"])) 
```

添加和编译 C/C++ 扩展确实可能具有挑战性，所以如果你需要处理这个问题，我建议从像 `pillow` 和 `numpy` 这样的项目中汲取灵感。它们可能有点复杂，但应该为你提供一个很好的起点，几乎涵盖了所有场景。

## Cython 扩展

在处理扩展方面，`setuptools` 库比常规的 `distutils` 库要聪明一些：它实际上为 `Extension` 类增加了一个小技巧。还记得在第十二章中对 `cython` 的简要介绍吗？关于性能？`setuptools` 库使得编译 Cython 扩展变得更加方便。Cython 手册建议使用以下类似代码：

```py
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("src/*.pyx")
) 
```

这种方法的缺点是，如果没有安装 Cython，`setup.py` 将会因为 `ImportError` 而出错：

```py
$ python3 setup.py build
Traceback (most recent call last):
  File "setup.py", line 2, in <module>
    import Cython
ImportError: No module named 'Cython' 
```

为了防止这个问题，我们将让 `setuptools` 处理 Cython 编译：

```py
import setuptools

setuptools.setup(
    name='T_05_cython',
    version='0.1.0',
    ext_modules=[
        setuptools.Extension(
            'sum_of_squares',
            sources=['T_05_cython/sum_of_squares.pyx'],
        ),
    ],
    setup_requires=['cython'],
) 
```

现在，如果需要，Cython 将会自动安装，代码将正常工作：

```py
$ python3 setup.py build
running build
running build_ext
cythoning T_05_cython/sum_of_squares.pyx to T_05_cython/sum_of_squares.c
building 'sum_of_squares' extension
... 
```

然而，为了开发目的，Cython 也提供了一种更简单的方法，不需要手动构建，即 `pyximport`：

```py
$ python3
>>> import pyximport

>>> pyximport.install()
(None, <pyximport.pyximport.PyxImporter object at ...>)

>>> from T_05_cython import sum_of_squares

>>> sum_of_squares.sum_of_squares(10)
14 
```

这样就可以轻松地运行 `pyx` 文件而无需显式编译。

# 测试

在*第十章*的*测试和日志记录 – 准备错误*中，我们看到了许多 Python 测试系统中的几个。正如您所怀疑的，其中至少有一些与 `setup.py` 集成。应该注意的是，`setuptools` 甚至有一个专门的 `test` 命令（在撰写本文时），但此命令已被弃用，`setuptools` 文档现在建议使用 `tox`。虽然我是 `tox` 的忠实粉丝，但对于即时本地开发，它通常会带来相当大的开销。我发现直接执行 `py.test` 会更快，因为您可以快速测试仅更改的代码部分。

## unittest

在开始之前，我们应该为我们的包创建一个测试脚本。对于实际的测试，请参阅*第十章*；在这种情况下，我们只是使用一个空操作测试，`test.py`：

```py
import unittest

class Test(unittest.TestCase):
    def test(self):
        pass 
```

标准的 `python setup.py test` 命令已被弃用，因此我们将直接运行 `unittest`：

```py
$ python3 -m unittest -v test
running test
... 
```

`unittest` 库仍然相当有限，因此我建议直接跳转到 `py.test`。

## py.test

`py.test` 包目前会自动注册为 `setuptools` 中的一个额外命令，因此安装后您可以直接运行 `python3 setup.py pytest`。然而，由于 `setuptools` 正在积极减少与 `setup.py` 的所有交互，我建议直接使用 `py.test` 或 `tox` 调用。

如前所述，建议使用 `tox` 来初始化环境并全面测试项目。然而，对于快速本地开发，我建议安装 `pytest` 模块并直接运行测试。

注意，可能仍然有一些旧的文档建议使用 `pytest-runner`、带有别名或自定义命令的 `python setup.py test`，或者生成一个 `runtests.py` 文件，但所有这些解决方案都已弃用，不应再使用。

配置 `py.test` 时，我们有几种选项取决于您的偏好。以下所有文件都将有效：

+   `pytest.ini`

+   `pyproject.toml`

+   `tox.ini`

+   `setup.cfg`

对于我维护的项目，我已经将测试需求定义为额外的依赖项，因此可以使用（例如）`pip3 install -e "./progressbar2[tests]"` 来安装。之后，您可以轻松地运行 `py.test` 来以 `tox` 运行测试的方式运行测试。当然，`tox` 也可以使用相同的额外依赖项来安装需求，这确保了您使用的是相同的测试环境。

要在您的 `setup.cfg`（或 `setup.py` / `pyproject.toml` 的等效文件）中启用此功能：

```py
[options.extras_require]
tests = pytest 
```

对于本地开发，我们现在可以以可编辑模式安装包和额外依赖项以进行快速测试：

```py
$ pip3 install -e '.[tests]' 
```

这样就足以能够直接使用 `py.test` 进行测试：

```py
$ py.test 
```

要使用 `tox` 进行测试，您需要创建一个 `tox.ini` 文件，但为此，我建议您查看*第十章*。

# 练习

现在您已经到达了本书的结尾，当然有很多事情可以尝试。您可以构建和发布自己的应用程序和库，或者扩展现有的库和应用。

在尝试本章的示例时，请注意不要意外地将包发布到 PyPI，如果不是你的意图。只需一个`twine`命令就可能导致意外注册和上传包，而 PyPI 上已经充斥着没有实际用途的包。

对于一些实际练习：

+   创建一个`setuptools`命令来提升你包的版本

+   通过交互式询问进行主要、次要或补丁升级来扩展版本提升命令

+   尝试将现有的项目从`setup.py`转换为`pyproject.toml`结构

这些练习的示例答案可以在 GitHub 上找到：`github.com/mastering-python/exercises`。我们鼓励你提交自己的解决方案，并从他人的解决方案中学习。

# 摘要

在阅读完这一章后，你应该能够创建包含纯 Python 文件、额外数据、编译的 C/C++扩展、文档和测试的 Python 包。有了所有这些工具，你现在能够制作出高质量的 Python 包，这些包可以轻松地在其他项目和包中重用。

Python 基础设施使得创建新包并将你的项目拆分为多个子项目变得非常简单。这允许你创建简单且可重用的包，因为一切都可以轻松测试，从而减少错误。虽然你不应该过度拆分包，但如果一个脚本或模块有其自身的目的，那么它就是单独打包的候选者。

*

随着这一章的结束，我们来到了这本书的结尾。我真诚地希望你喜欢阅读它，并了解了一些新的有趣的话题。任何和所有的反馈都将非常受重视，所以请随时通过我的网站[`wol.ph/`](https://wol.ph/)联系我。

# 加入我们的 Discord 社区

加入我们的社区 Discord 空间，与作者和其他读者进行讨论：[`discord.gg/QMzJenHuJf`](https://discord.gg/QMzJenHuJf)

![二维码](img/QR_Code156081100001293319171.png)
