# 第一章：正确开始——使用 Virtualenv

现代软件开发中的一个巨大困难是依赖管理。一般来说，软件项目的依赖关系包括所需的库或组件，以使项目能够正确运行。对于 Flask 应用程序（更一般地说，对于 Python 应用程序），大多数依赖关系由特别组织和注释的源文件组成。创建后，这些源文件包可以包含在其他项目中，依此类推。对于一些人来说，这种依赖链可能变得难以管理，当链中的任何库发生细微变化时，可能会导致一系列不兼容性，从而使进一步的开发陷入停滞。在 Python 世界中，正如您可能已经知道的那样，可重用的源文件的基本单元是 Python 模块（包含定义和语句的文件）。一旦您在本地文件系统上创建了一个模块，并确保它在系统的 PYTHONPATH 中，将其包含在新创建的项目中就像指定导入一样简单，如下所示：

```py
import the_custom_module

```

其中`the_custom_module.py`是一个存在于执行程序系统的`$PYTHONPATH`中的文件。

### 注意：

`$PYTHONPATH`可以包括对压缩存档（`.zip`文件夹）的路径，除了正常的文件路径。

当然，故事并不会在这里结束。虽然最初在本地文件系统中散布模块可能很方便，但当您想要与他人共享一些您编写的代码时会发生什么？通常，这将涉及通过电子邮件/Dropbox 发送相关文件，然而，这显然是一个非常繁琐且容易出错的解决方案。幸运的是，这是一个已经被考虑过并且已经在缓解常见问题方面取得了一些进展的问题。其中最重要的进展之一是本章的主题，以及如何利用以下创建可重用的、隔离的代码包的技术来简化 Flask 应用程序的开发：

+   使用 pip 和 setuptools 进行 Python 打包

+   使用 virtualenv 封装虚拟环境

各种 Python 打包范例/库提出的解决方案远非完美；与热情的 Python 开发者争论的一种肯定方式是宣称*打包问题*已经解决！我们在这方面还有很长的路要走，但通过改进 setuptools 和其他用于构建、维护和分发可重用 Python 代码的库，我们正在逐步取得进展。

在本章中，当我们提到一个包时，我们实际上要谈论的是一个分发——一个从远程源安装的软件包——而不是一个使用`the__init__.py`约定来划分包含我们想要导入的模块的文件夹结构的集合。

# Setuptools 和 pip

当开发人员希望使他们的代码更广泛可用时，首要步骤之一将是创建一个与 setuptools 兼容的包。

现代 Python 版本的大多数发行版将已经安装了 setuptools。如果您的系统上没有安装它，那么获取它相对简单，官方文档中还提供了额外的说明：

```py
wget https://bootstrap.pypa.io/ez_setup.py -O - | python

```

安装了 setuptools 之后，创建兼容包的基本要求是在项目的根目录创建一个`setup.py`文件。该文件的主要内容应该是调用`setup()`函数，并带有一些强制（和许多可选）参数，如下所示：

```py
from setuptools import setup

setup(
 name="My Great Project",
 version="0.0.1",
 author="Jane Doe",
 author_email="jane@example.com",
 description= "A brief summary of the project.",
 license="BSD",
 keywords="example tutorial flask",
 url="http://example.com/my-great-project",
 packages=['foobar','tests'],
 long_description="A much longer project description.",
 classifiers=[
 "Development Status :: 3 - Alpha",
 "Topic :: Utilities",
 "License :: OSI Approved :: BSD License",
 ],
)

```

### 提示

**下载示例代码**

您可以从[`www.packtpub.com`](http://www.packtpub.com)的帐户中下载您购买的所有 Packt Publishing 图书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便直接通过电子邮件接收文件。

一旦软件包被创建，大多数开发人员将选择使用 setuptools 本身提供的内置工具将他们新创建的软件包上传到 PyPI——几乎所有 Python 软件包的官方来源。虽然使用特定的公共 PyPI 存储库并不是必需的（甚至可以设置自己的个人软件包索引），但大多数 Python 开发人员都希望在这里找到他们的软件包。

这将引出拼图中另一个至关重要的部分——`pip` Python 软件包安装程序。如果您已安装 Python 2.7.9 或更高版本，则`pip`将已经存在。某些发行版可能已经为您预安装了它，或者它可能存在于系统级软件包中。对于类似 Debian 的 Linux 发行版，可以通过以下命令安装它：

```py
apt-get install python-pip

```

同样，其他基于 Linux 的发行版将有他们自己推荐的软件包管理器。如果您更愿意获取源代码并手动安装，只需获取文件并使用 Python 解释器运行即可：

```py
$ curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py
$ python get-pip.py

```

Pip 是一个用于安装 Python 软件包的工具（本身也是一个 Python 软件包）。虽然它不是唯一的选择，但`pip`是迄今为止使用最广泛的。

### 注意

`pip`的前身是`easy_install`，在 Python 社区中已经大部分被后者取代。`easy_install`模块存在一些相当严重的问题，比如允许部分安装、无法卸载软件包而需要用户手动删除相关的`.egg`文件，以及包含有用的成功和错误消息的控制台输出，允许开发人员在出现问题时确定最佳操作方式。

可以在命令行中调用 pip 来在本地文件系统上安装科学计算软件包，比如说：

```py
$ pip install numpy

```

上述命令将查询默认的 PyPI 索引，寻找名为`numpy`的软件包，并将最新版本下载到系统的特定位置，通常是`/usr/local/lib/pythonX.Y/site-packages`（`X`和`Y`是`pip`指向的 Python 版本的主/次版本）。此操作可能需要 root 权限，因此需要`sudo`或类似的操作来完成。

虚拟环境的许多好处之一是，它们通常避免了对已安装软件包进行系统级更改时可能出现的权限提升要求。

一旦此操作成功完成，您现在可以将`numpy`软件包导入新模块，并使用它提供的所有功能：

```py
import numpy

x = numpy.array([1, 2, 3])
sum = numpy.sum(x)
print sum  # prints 6

```

一旦我们安装了这个软件包（或者其他任何软件包），就没有什么可以阻止我们以通常的方式获取其他软件包。此外，我们可以通过将它们的名称作为`install`命令的附加参数来一次安装多个软件包：

```py
$ pip install scipy pandas # etc.

```

# 避免依赖地狱，Python 的方式

新开发人员可能会想要安装他们遇到的每个有趣的软件包。这样做的话，他们可能会意识到这很快就会变成一个卡夫卡式的情况，先前安装的软件包可能会停止工作，新安装的软件包可能会表现得不可预测，如果它们成功安装的话。前述方法的问题，正如你们中的一些人可能已经猜到的那样，就是冲突的软件包依赖关系。例如，假设我们安装了软件包`A`；它依赖于软件包`Q`的版本 1 和软件包`R`的版本 1。软件包`B`依赖于软件包`R`的版本 2（其中版本 1 和 2 不兼容）。Pip 将愉快地为您安装软件包`B`，这将升级软件包`R`到版本 2。这将使软件包`A`在最好的情况下完全无法使用，或者在最坏的情况下，使其以未记录和不可预测的方式行为。

Python 生态系统已经提出了一个解决从俗称为**依赖地狱**中产生的基本问题的解决方案。虽然远非完美，但它允许开发人员规避在 Web 应用程序开发中可能出现的许多最简单的软件包版本依赖冲突。

`virtualenv`工具（Python 3.3 中的默认模块`venv`）是必不可少的，以确保最大限度地减少陷入依赖地狱的机会。以下引用来自`virtualenv`官方文档的介绍部分：

> *它创建一个具有自己安装目录的环境，不与其他 virtualenv 环境共享库（也可以选择不访问全局安装的库）。*

更简洁地说，`virtualenv`允许您为每个 Python 应用程序（或任何 Python 代码）创建隔离的环境。

### 注意

`virtualenv`工具不会帮助您管理 Python 基于 C 的扩展的依赖关系。例如，如果您从`pip`安装`lxml`软件包，它将需要您拥有正确的`libxml2`和`libxslt`系统库和头文件（它将链接到）。`virtualenv`工具将无法帮助您隔离这些系统级库。

# 使用 virtualenv

首先，我们需要确保在本地系统中安装了`virtualenv`工具。这只是从 PyPI 存储库中获取它的简单事情：

```py
$ pip install virtualenv

```

### 注意

出于明显的原因，应该在可能已经存在的任何虚拟环境之外安装这个软件包。

## 创建新的虚拟环境

创建新的虚拟环境很简单。以下命令将在指定路径创建一个新文件夹，其中包含必要的结构和脚本，包括默认 Python 二进制文件的完整副本：

```py
$ virtualenv <path/to/env/directory>

```

如果我们想创建一个位于`~/envs/testing`的环境，我们首先要确保父目录存在，然后调用以下命令：

```py
$ mkdir -p ~/envs
$ virtualenv ~/envs/testing

```

在 Python 3.3+中，一个大部分与 API 兼容的`virtualenv`工具被添加到默认语言包中。模块的名称是`venv`，然而，允许您创建虚拟环境的脚本的名称是`pyvenv`，可以以与先前讨论的`virtualenv`工具类似的方式调用：

```py
$ mkdir -p ~/envs
$ pyvenv ~/envs/testing

```

## 激活和停用虚拟环境

创建虚拟环境不会自动激活它。环境创建后，我们需要激活它，以便对 Python 环境进行任何修改（例如安装软件包）将发生在隔离的环境中，而不是我们系统的全局环境中。默认情况下，激活虚拟环境将更改当前活动用户的提示字符串（`$PS1`），以便显示所引用的虚拟环境的名称：

```py
$ source ~/envs/testing/bin/activate
(testing) $ # Command prompt modified to display current virtualenv

```

Python 3.3+的命令是相同的：

```py
$ source ~/envs/testing/bin/activate
(testing) $ # Command prompt modified to display current virtualenv

```

当您运行上述命令时，将发生以下一系列步骤：

1.  停用任何已激活的环境。

1.  使用`virtualenv bin/`目录的位置在您的`$PATH`变量之前添加，例如`~/envs/testing/bin:$PATH`。

1.  如果存在，则取消设置`$PYTHONHOME`。

1.  修改您的交互式 shell 提示，以包括当前活动的`virtualenv`的名称。

由于`$PATH`环境变量的操作，通过激活环境的 shell 调用的 Python 和`pip`二进制文件（以及通过`pip`安装的其他二进制文件）将包含在`~/envs/testing/bin`中。

## 向现有环境添加包

我们可以通过简单激活它，然后以以下方式调用`pip`来轻松向虚拟环境添加包：

```py
$ source ~/envs/testing/bin/activate
(testing)$ pip install numpy

```

这将把`numpy`包安装到测试环境中，只有测试环境。您的全局系统包不受影响，以及任何其他现有环境。

## 从现有环境中卸载包

卸载`pip`包也很简单：

```py
$ source ~/envs/testing/bin/activate
(testing)$ pip uninstall numpy

```

这将仅从测试环境中删除`numpy`包。

这是 Python 软件包管理存在相对重要的一个地方：卸载一个包不会卸载它的依赖项。例如，如果安装包`A`并安装依赖包`B`和`C`，则以后卸载包`A`将不会卸载`B`和`C`。

# 简化常见操作-使用 virtualenvwrapper 工具

我经常使用的一个工具是`virtualenvwrapper`，它是一组非常智能的默认值和命令别名，使得使用虚拟环境更直观。现在让我们将其安装到我们的全局系统中：

```py
$ pip install virtualenvwrapper

```

### 注意

这也将安装`virtualenv`包，以防它尚未存在。

接下来，您需要将以下行添加到您的 shell 启动文件的末尾。这很可能是`~/.bashrc`，但是如果您已将默认 shell 更改为其他内容，例如`zsh`，那么它可能会有所不同（例如`~/.zshrc`）：

```py
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh

```

在上述代码块的第一行指示使用`virtualenvwrapper`创建的新虚拟环境应存储在`$HOME/.virtualenvs`中。您可以根据需要修改此设置，但我通常将其保留为一个很好的默认值。我发现将所有虚拟环境放在我的主目录中的同一个隐藏文件夹中可以减少个别项目中的混乱，并使误将整个虚拟环境添加到版本控制变得更加困难。

### 注意

将整个虚拟环境添加到版本控制可能看起来像一个好主意，但事情从来不像看起来那么简单。一旦运行稍微（或完全）不同的操作系统的人决定下载您的项目，其中包括可能包含针对您自己的架构编译的`C`模块的包的完整`virtualenv`文件夹，他们将很难使事情正常工作。

相反，pip 支持并且许多开发人员使用的常见模式是在虚拟环境中冻结已安装包的当前状态，并将其保存到`requirements.txt`文件中：

```py
(testing) $ pip freeze > requirements.txt

```

然后，该文件可以添加到**版本控制系统**（**VCS**）中。由于该文件的目的是声明应用程序所需的依赖关系，而不是提供它们或指示如何构建它们，因此您的项目的用户可以自由选择以任何方式获取所需的包。通常，他们会通过`pip`安装它们，`pip`可以很好地处理要求文件：

```py
(testing) $ pip install –r  requirements.txt

```

第二行在当前 shell 环境中添加了一些方便的别名，以创建、激活、切换和删除环境：

+   `mkvirtualenv test`：这将创建一个名为 test 的环境并自动激活它。

+   `mktmpenv test`：这将创建一个名为 test 的临时环境并自动激活它。一旦调用停用脚本，此环境将被销毁。

+   `workon app`：这将切换到 app 环境（已经创建）。

+   `workon`（`alias lsvirtualenv`）：当您不指定环境时，这将打印所有可用的现有环境。

+   `deactivate`：如果有的话，这将禁用当前活动的环境。

+   `rmvirtualenv app`：这将完全删除 app 环境。

我们将使用以下命令创建一个环境来安装我们的应用程序包：

```py
$ mkvirtualenv app1

```

这将创建一个空的`app1`环境并激活它。您应该在 shell 提示符中看到一个（`app1`）标签。

### 注意

如果您使用的是 Bash 或 ZSH 之外的 shell，此环境标签可能会出现也可能不会出现。这样的工作方式是，用于激活虚拟环境的脚本还会修改当前的提示字符串（`PS1`环境变量），以便指示当前活动的`virtualenv`。因此，如果您使用非常特殊或非标准的 shell 配置，则有可能无法正常工作。

# 摘要

在本章中，我们看到了任何非平凡的 Python 应用程序都面临的最基本的问题之一：库依赖管理。值得庆幸的是，Python 生态系统已经开发了被广泛采用的`virtualenv`工具，用于解决开发人员可能遇到的最常见的依赖问题子集。

此外，我们还介绍了一个工具`virtualenvwrapper`，它抽象了一些使用`virtualenv`执行的最常见操作。虽然我们列出了这个软件包提供的一些功能，但`virtualenvwrapper`可以做的事情更加广泛。我们只是在这里介绍了基础知识，但如果您整天都在使用 Python 虚拟环境，深入了解这个工具能做什么是不可或缺的。
