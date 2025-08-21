# 第五章：编写包

本章重点介绍了编写和发布 Python 包的可重复过程。其意图是：

+   在开始真正工作之前缩短设置所需的时间

+   提供一种标准化的编写包的方式

+   简化测试驱动开发方法的使用

+   促进发布过程

它分为以下四个部分：

+   所有包的**常见模式**，描述了所有 Python 包之间的相似之处，以及`distutils`和`setuptools`如何发挥核心作用

+   什么是**命名空间包**以及它们为何有用

+   如何在**Python 包索引**（**PyPI**）中注册和上传包，重点放在安全性和常见陷阱上

+   独立可执行文件作为打包和分发 Python 应用程序的替代方式

# 创建一个包

Python 打包一开始可能有点令人不知所措。这主要是因为对于创建 Python 包的适当工具的混乱。不管怎样，一旦您创建了第一个包，您会发现这并不像看起来那么困难。此外，了解适当的、最新的打包工具也会有很大帮助。

即使您不打算将代码作为开源分发，您也应该知道如何创建包。了解如何制作自己的包将使您更深入地了解打包生态系统，并有助于您使用 PyPI 上可用的第三方代码。

此外，将您的闭源项目或其组件作为源分发包可用，可以帮助您在不同环境中部署代码。利用 Python 打包生态系统在代码部署中的优势将在下一章中更详细地描述。在这里，我们将专注于创建这样的分发的适当工具和技术。

## Python 打包工具的混乱状态

很长一段时间以来，Python 打包的状态非常混乱，花了很多年时间才将这个话题组织起来。一切都始于 1998 年引入的`distutils`包，后来在 2003 年由`setuptools`进行了增强。这两个项目开启了一个漫长而复杂的分叉、替代项目和完全重写的故事，试图一劳永逸地修复 Python 的打包生态系统。不幸的是，大多数尝试都没有成功。效果恰恰相反。每个旨在取代`setuptools`或`distutils`的新项目都增加了已经围绕打包工具的巨大混乱。一些这样的分叉被合并回它们的祖先（比如`distribute`是`setuptools`的一个分叉），但有些被遗弃了（比如`distutils2`）。

幸运的是，这种状态正在逐渐改变。一个名为**Python 打包管理机构**（**PyPA**）的组织成立，旨在恢复打包生态系统的秩序和组织。由 PyPA 维护的**Python 打包用户指南**（[`packaging.python.org`](https://packaging.python.org)）是关于最新打包工具和最佳实践的权威信息来源。把它视为关于打包的最佳信息来源，以及本章的补充阅读。该指南还包含了与打包相关的更改和新项目的详细历史，因此如果您已经了解一些内容但想确保仍在使用适当的工具，它将非常有用。

远离其他流行的互联网资源，比如**打包者指南**。它已经过时，没有维护，大部分已经过时。它可能只是出于历史原因有趣，而 Python 打包用户指南实际上是这个旧资源的一个分支。

### 由于 PyPA，Python 打包的当前格局

除了为打包提供权威指南外，PyPA 还维护打包项目和新官方打包方面的标准化过程。PyPA 的所有项目都可以在 GitHub 的一个组织下找到：[`github.com/pypa`](https://github.com/pypa)。

其中一些在书中已经提到。最显著的是：

+   `pip`

+   `virtualenv`

+   `twine`

+   `warehouse`

请注意，其中大多数是在该组织之外启动的，并且只有在成熟和广泛使用的解决方案下才移至 PyPA 赞助下。

由于 PyPA 的参与，逐渐放弃鸡蛋格式，转而使用 wheels 进行构建分发已经在进行中。未来可能会带来更多新的变化。PyPA 正在积极开发`warehouse`，旨在完全取代当前的 PyPI 实现。这将是包装历史上的一大步，因为`pypi`是一个如此古老和被忽视的项目，只有少数人能够想象在没有完全重写的情况下逐渐改进它。

### 工具推荐

Python Packaging User Guide 给出了一些建议，推荐使用一些工具来处理软件包。它们通常可以分为两组：用于安装软件包的工具和用于创建和分发软件包的工具。

PyPA 推荐的第一组工具已经在第一章中提到过，但为了保持一致，让我们在这里重复一下：

+   使用`pip`从 PyPI 安装软件包

+   使用`virtualenv`或`venv`来实现 Python 环境的应用级隔离

Python Packaging User Guide 给出了一些建议，推荐用于创建和分发软件包的工具如下：

+   使用`setuptools`来定义项目并创建**源分发**

+   使用**wheels**而不是**eggs**来创建**构建分发**

+   使用`twine`将软件包分发上传到 PyPI

## 项目配置

显而易见，组织大型应用程序代码的最简单方法是将其拆分为几个软件包。这使得代码更简单，更易于理解，维护和更改。它还最大化了每个软件包的可重用性。它们就像组件一样。

### setup.py

必须分发的软件包的根目录包含一个`setup.py`脚本。它定义了`distutils`模块中描述的所有元数据，作为对标准`setup()`函数的参数的组合。尽管`distutils`是一个标准库模块，但建议您使用`setuptools`包，它对标准`distutils`提供了几个增强功能。

因此，此文件的最小内容是：

```py
from setuptools import setup

setup(
    name='mypackage',
)
```

`name`给出了软件包的完整名称。从那里，脚本提供了几个命令，可以使用`--help-commands`选项列出：

```py
$ python3 setup.py --help-commands
Standard commands:
 **build             build everything needed to install
 **clean             clean up temporary files from 'build' command
 **install           install everything from build directory
 **sdist             create a source distribution (tarball, zip file)
 **register          register the distribution with the PyP
 **bdist             create a built (binary) distribution
 **check             perform some checks on the package
 **upload            upload binary package to PyPI

Extra commands:
 **develop           install package in 'development mode'
 **alias             define a shortcut to invoke one or more commands
 **test              run unit tests after in-place build
 **bdist_wheel       create a wheel distribution

usage: setup.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]
 **or: setup.py --help [cmd1 cmd2 ...]
 **or: setup.py --help-commands
 **or: setup.py cmd --help

```

实际的命令列表更长，可以根据可用的`setuptools`扩展而变化。它被截断以仅显示对本章最重要和相关的命令。**标准** **命令**是`distutils`提供的内置命令，而**额外** **命令**是由第三方软件包创建的命令，例如`setuptools`或定义和注册新命令的任何其他软件包。另一个软件包注册的额外命令是由`wheel`软件包提供的`bdist_wheel`。

### setup.cfg

`setup.cfg`文件包含`setup.py`脚本命令的默认选项。如果构建和分发软件包的过程更复杂，并且需要传递许多可选参数给`setup.py`命令，这将非常有用。这允许您在每个项目的代码中存储这些默认参数。这将使您的分发流程独立于项目，并且还可以提供关于如何构建和分发软件包给用户和其他团队成员的透明度。

`setup.cfg`文件的语法与内置的`configparser`模块提供的语法相同，因此类似于流行的 Microsoft Windows INI 文件。以下是一个设置配置文件的示例，其中提供了一些`global`，`sdist`和`bdist_wheel`命令的默认值：

```py
[global]
quiet=1

[sdist]
formats=zip,tar

[bdist_wheel]
universal=1
```

此示例配置将确保始终使用两种格式（ZIP 和 TAR）创建源分发，并且将创建通用轮（与 Python 版本无关）的构建轮分发。此外，通过全局`quiet`开关，每个命令的大部分输出都将被抑制。请注意，这仅用于演示目的，可能不是默认情况下抑制每个命令的输出的合理选择。

### MANIFEST.in

使用`sdist`命令构建分发时，`distutils`浏览包目录，寻找要包含在存档中的文件。`distutils`将包括：

+   由`py_modules`，`packages`和`scripts`选项隐含的所有 Python 源文件

+   `ext_modules`选项中列出的所有 C 源文件

与 glob 模式`test/test*.py`匹配的文件是：`README`，`README.txt`，`setup.py`和`setup.cfg`。

此外，如果您的包处于子版本或 CVS 下，`sdist`将浏览文件夹，如`.svn`，以寻找要包含的文件。还可以通过扩展与其他版本控制系统集成。`sdist`构建一个列出所有文件并将它们包含到存档中的`MANIFEST`文件。

假设您不使用这些版本控制系统，并且需要包含更多文件。现在，您可以在与`setup.py`相同的目录中定义一个名为`MANIFEST.in`的模板，用于`MANIFEST`文件，其中您指示`sdist`包含哪些文件。

此模板每行定义一个包含或排除规则，例如：

```py
include HISTORY.txt
include README.txt
include CHANGES.txt
include CONTRIBUTORS.txt
include LICENSE
recursive-include *.txt *.py
```

`MANIFEST.in`的完整命令列表可以在官方`distutils`文档中找到。

### 最重要的元数据

除了要分发的软件包的名称和版本外，`setup`可以接收的最重要的参数是：

+   `description`: 这包括几句话来描述该包

+   `long_description`: 这包括一个可以使用 reStructuredText 的完整描述

+   `keywords`: 这是定义该包的关键字列表

+   `author`: 这是作者的姓名或组织

+   `author_email`: 这是联系电子邮件地址

+   `url`: 这是项目的 URL

+   `license`: 这是许可证（GPL，LGPL 等）

+   `packages`: 这是包中所有名称的列表；`setuptools`提供了一个称为`find_packages`的小函数来计算这个列表

+   `namespace_packages`: 这是命名空间包的列表

### Trove classifiers

PyPI 和`distutils`提供了一组分类应用程序的解决方案，称为**trove classifiers**。所有分类器形成一个类似树状的结构。每个分类器都是一种字符串形式，其中每个命名空间都由`::`子字符串分隔。它们的列表作为`classifiers`参数提供给`setup()`函数的包定义。以下是 PyPI 上某个项目（这里是`solrq`）的分类器示例列表：

```py
from setuptools import setup

setup(
    name="solrq",
    # (...)

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Internet :: WWW/HTTP :: Indexing/Search',
    ],
)
```

它们在包定义中是完全可选的，但为`setup()`接口中可用的基本元数据提供了有用的扩展。除其他外，trove classifiers 可能提供有关支持的 Python 版本或系统、项目的开发阶段或代码发布的许可证的信息。许多 PyPI 用户通过分类搜索和浏览可用的软件包，因此适当的分类有助于软件包达到其目标。

Trove classifiers 在整个打包生态系统中起着重要作用，不应被忽视。没有组织验证软件包的分类，因此您有责任为您的软件包提供适当的分类器，并且不要给整个软件包索引引入混乱。

撰写本书时，PyPI 上有 608 个分类器，分为九个主要类别：

+   开发状态

+   环境

+   框架

+   预期受众

+   许可证

+   自然语言

+   操作系统

+   编程语言

+   主题

新的分类器会不时地被添加，因此在您阅读时这些数字可能会有所不同。当前可用的所有 trove 分类器的完整列表可通过`setup.py register --list-classifiers`命令获得。

### 常见模式

为了分发而创建一个包对于经验不足的开发人员来说可能是一项繁琐的任务。`setuptools`或`distuitls`在它们的`setup()`函数调用中接受的大部分元数据可以手动提供，忽略了这些可能在项目的其他部分中可用的事实：

```py
from setuptools import setup

setup(
    name="myproject",
    version="0.0.1",
    description="mypackage project short description",
    long_description="""
        Longer description of mypackage project
        possibly with some documentation and/or
        usage examples
    """,
    install_requires=[
        'dependency1',
        'dependency2',
        'etc',
    ]
)
```

虽然这肯定会起作用，但在长期内很难维护，并且留下了未来错误和不一致的可能性。`setuptools`和`distutils`都无法自动从项目源中提取各种元数据信息，因此您需要自己提供它们。在 Python 社区中有一些常见的模式用于解决最常见的问题，如依赖管理、版本/自述文件的包含等。至少了解其中一些是值得的，因为它们如此受欢迎，以至于它们可以被视为包装习语。

#### 从包中自动包含版本字符串

**PEP 440（版本标识和依赖规范）**文档规定了版本和依赖规范的标准。这是一个长篇文档，涵盖了接受的版本规范方案以及 Python 包装工具中版本匹配和比较应该如何工作。如果您正在使用或计划使用复杂的项目版本编号方案，那么阅读这个文档是义不容辞的。如果您使用的是由点分隔的一个、两个、三个或更多数字组成的简单方案，那么您可以放弃阅读 PEP 440。如果您不知道如何选择适当的版本方案，我强烈建议遵循语义化版本控制，这已经在第一章中提到过了。

另一个问题是在包或模块中包含版本说明符的位置。有 PEP 396（模块版本号），它正好处理这个问题。请注意，它只是信息性的，并且具有*延迟*状态，因此它不是标准跟踪的一部分。无论如何，它描述了现在似乎是*事实*标准。根据 PEP 396，如果包或模块有指定的版本，它应该被包含为包根（`__init__.py`）或模块文件的`__version__`属性。另一个*事实*标准是还包括包含版本部分的`VERSION`属性的元组。这有助于用户编写兼容性代码，因为如果版本方案足够简单，这样的版本元组可以很容易地进行比较。

PyPI 上有很多包都遵循这两个标准。它们的`__init__.py`文件包含如下所示的版本属性：

```py
# version as tuple for simple comparisons
VERSION = (0, 1, 1)
# string created from tuple to avoid inconsistency
__version__ = ".".join([str(x) for x in VERSION])
```

延迟 PEP 396 的另一个建议是，distutils 的`setup()`函数中提供的版本应该从`__version__`派生，或者反之亦然。Python 包装用户指南提供了单一源项目版本的多种模式，每种模式都有其自己的优点和局限性。我个人比较喜欢的是比较长的模式，它没有包含在 PyPA 的指南中，但它的优点是将复杂性限制在`setup.py`脚本中。这个样板假设版本说明符由包的`__init__`模块的`VERSION`属性提供，并提取这些数据以包含在`setup()`调用中。以下是一些虚构包的`setup.py`脚本的摘录，展示了这种方法：

```py
from setuptools import setup
import os

def get_version(version_tuple):
    # additional handling of a,b,rc tags, this can
    # be simpler depending on your versioning scheme
    if not isinstance(version_tuple[-1], int):
        return '.'.join(
            map(str, version_tuple[:-1])
        ) + version_tuple[-1]

    return '.'.join(map(str, version_tuple))

# path to the packages __init__ module in project
# source tree
init = os.path.join(
    os.path.dirname(__file__), 'src', 'some_package', '__init__.py'
)

version_line = list(
    filter(lambda l: l.startswith('VERSION'), open(init))
)[0]

# VERSION is a tuple so we need to eval its line of code.
# We could simply import it from the package but we
# cannot be sure that this package is importable before
# finishing its installation
VERSION = get_version(eval(version_line.split('=')[-1]))

setup(
    name='some-package',
    version=VERSION,
    # ...
)
```

#### 自述文件

Python Packaging Index 可以在 PyPI 门户网站的软件包页面上显示项目的 readme 或`long_description`的值。你可以使用 reStructuredText ([`docutils.sourceforge.net/rst.html`](http://docutils.sourceforge.net/rst.html))标记编写这个描述，因此在上传时它将被格式化为 HTML。不幸的是，目前只有 reStructuredText 作为 PyPI 上的文档标记可用。在不久的将来，这种情况不太可能改变。更有可能的是，当我们看到`warehouse`项目完全取代当前的 PyPI 实现时，将支持更多的标记语言。不幸的是，`warehouse`的最终发布日期仍然未知。

然而，许多开发人员出于各种原因希望使用不同的标记语言。最受欢迎的选择是 Markdown，这是 GitHub 上默认的标记语言——大多数开源 Python 开发目前都在这里进行。因此，通常，GitHub 和 Markdown 爱好者要么忽视这个问题，要么提供两个独立的文档文本。提供给 PyPI 的描述要么是项目 GitHub 页面上可用的简短版本，要么是在 PyPI 上呈现不佳的纯 Markdown 格式。

如果你想为你的项目的 README 使用不同于 reStructuredText 标记语言的东西，你仍然可以以可读的形式在 PyPI 页面上提供它作为项目描述。诀窍在于使用`pypandoc`软件包将你的其他标记语言转换为 reStructuredText，同时上传到 Python Package Index 时要有一个回退到你的 readme 文件的纯内容，这样如果用户没有安装`pypandoc`，安装就不会失败：

```py
try:
    from pypandoc import convert

    def read_md(f):
        return convert(f, 'rst')

except ImportError:
    convert = None
    print(
        "warning: pypandoc module not found, could not convert Markdown to RST"
    )

    def read_md(f):
        return open(f, 'r').read()  # noqa

README = os.path.join(os.path.dirname(__file__), 'README.md')

setup(
    name='some-package',
    long_description=read_md(README),
    # ...
)
```

#### 管理依赖

许多项目需要安装和/或使用一些外部软件包。当依赖列表非常长时，就会出现如何管理的问题。在大多数情况下，答案非常简单。不要过度设计问题。保持简单，并在你的`setup.py`脚本中明确提供依赖列表：

```py
from setuptools import setup
setup(
    name='some-package',
    install_requires=['falcon', 'requests', 'delorean']
    # ...
)
```

一些 Python 开发人员喜欢使用`requirements.txt`文件来跟踪他们软件包的依赖列表。在某些情况下，你可能会找到理由这样做，但在大多数情况下，这是该项目代码未正确打包的遗留物。无论如何，即使像 Celery 这样的知名项目仍然坚持这种约定。因此，如果你不愿意改变你的习惯，或者你在某种程度上被迫使用要求文件，那么至少要做到正确。以下是从`requirements.txt`文件中读取依赖列表的一种流行习语：

```py
from setuptools import setup
import os

def strip_comments(l):
    return l.split('#', 1)[0].strip()

def reqs(*f):
    return list(filter(None, [strip_comments(l) for l in open(
        os.path.join(os.getcwd(), *f)).readlines()]))

setup(
    name='some-package',
    install_requires=reqs('requirements.txt')
    # ...
)
```

## 自定义设置命令

`distutils`允许你创建新的命令。新的命令可以通过入口点进行注册，这是由`setuptools`引入的一种将软件包定义为插件的简单方法。

入口点是通过`setuptools`提供的一种通过一些 API 公开的类或函数的命名链接。任何应用程序都可以扫描所有已注册的软件包，并将链接的代码用作插件。

要链接新的命令，可以在设置调用中使用`entry_points`元数据：

```py
setup(
    name="my.command",
    entry_points="""
        [distutils.commands]
        my_command  = my.command.module.Class
    """
)
```

所有命名链接都被收集在命名部分中。当`distutils`被加载时，它会扫描在`distutils.commands`下注册的链接。

这种机制被许多提供可扩展性的 Python 应用程序使用。

## 在开发过程中使用软件包

使用`setuptools`主要是关于构建和分发软件包。然而，你仍然需要知道如何使用它们直接从项目源安装软件包。原因很简单。在提交软件包到 PyPI 之前，测试包装代码是否正常工作是很重要的。测试的最简单方法是安装它。如果你将一个有问题的软件包发送到存储库，那么为了重新上传它，你需要增加版本号。

在最终分发之前测试代码是否打包正确可以避免不必要的版本号膨胀，显然也可以节省时间。此外，在同时处理多个相关包时，直接从自己的源代码使用`setuptools`进行安装可能是必不可少的。

### setup.py install

`install`命令将包安装到 Python 环境中。如果之前没有进行构建，它将尝试构建包，然后将结果注入 Python 树中。当提供源分发时，可以将其解压缩到临时文件夹，然后使用此命令安装。`install`命令还将安装在`install_requires`元数据中定义的依赖项。这是通过查看 Python 包索引中的包来完成的。

在安装包时，除了使用裸`setup.py`脚本之外，还可以使用`pip`。由于它是 PyPA 推荐的工具，即使在本地环境中安装包用于开发目的时，也应该使用它。为了从本地源安装包，请运行以下命令：

```py
pip install <project-path>

```

### 卸载包

令人惊讶的是，`setuptools`和`distutils`缺乏`uninstall`命令。幸运的是，可以使用`pip`卸载任何 Python 包：

```py
pip uninstall <package-name>

```

在系统范围的包上尝试卸载可能是一种危险的操作。这是为什么对于任何开发都使用虚拟环境如此重要的另一个原因。

### setup.py develop 或 pip -e

使用`setup.py install`安装的包将被复制到当前环境的 site-packages 目录中。这意味着每当您对该包的源代码进行更改时，都需要重新安装它。这在密集开发过程中经常是一个问题，因为很容易忘记需要再次进行安装。这就是为什么`setuptools`提供了额外的`develop`命令，允许我们以**开发模式**安装包。此命令在部署目录（site-packages）中创建对项目源代码的特殊链接，而不是将整个包复制到那里。包源代码可以在不需要重新安装的情况下进行编辑，并且可以像正常安装一样在`sys.path`中使用。

`pip`还允许以这种模式安装包。这种安装选项称为*可编辑模式*，可以在`install`命令中使用`-e`参数启用：

```py
pip install -e <project-path>

```

# 命名空间包

*Python 之禅*，您可以通过在解释器会话中编写`import this`来阅读，关于命名空间说了以下内容：

> *命名空间是一个了不起的想法——让我们做更多这样的事情！*

这可以以至少两种方式理解。第一种是在语言环境中的命名空间。我们都在不知不觉中使用命名空间：

+   模块的全局命名空间

+   函数或方法调用的本地命名空间

+   内置名称的命名空间

另一种命名空间可以在打包级别提供。这些是**命名空间包**。这通常是一个被忽视的功能，可以在组织的包生态系统或非常庞大的项目中非常有用。

## 这有什么用呢？

命名空间包可以被理解为一种在元包级别以上对相关包或模块进行分组的方式，其中每个包都可以独立安装。

如果您的应用程序组件是独立开发、打包和版本化的，但您仍希望从相同的命名空间访问它们，命名空间包尤其有用。这有助于明确每个包属于哪个组织或项目。例如，对于一些虚构的 Acme 公司，通用命名空间可以是`acme`。结果可能会导致创建一个通用的`acme`命名空间包，用于容纳该组织的其他包。例如，如果 Acme 的某人想要贡献一个与 SQL 相关的库，他可以创建一个新的`acme.sql`包，并将其注册到`acme`中。

重要的是要了解普通包和命名空间包之间的区别以及它们解决的问题。通常（没有命名空间包），您将创建一个带有以下文件结构的`acme`包和`sql`子包/子模块：

```py
$ tree acme/
acme/
├── acme
│   ├── __init__.py
│   └── sql
│       └── __init__.py
└── setup.py

2 directories, 3 files

```

每当您想要添加一个新的子包，比如`templating`，您都被迫将其包含在`acme`的源树中：

```py
$ tree acme/
acme/
├── acme
│   ├── __init__.py
│   ├── sql
│   │   └── __init__.py
│   └── templating
│       └── __init__.py
└── setup.py

3 directories, 4 files

```

这种方法使得独立开发`acme.sql`和`acme.templating`几乎不可能。`setup.py`脚本还必须为每个子包指定所有的依赖关系，因此不可能（或者至少非常困难）只安装一些`acme`组件。而且，如果一些子包有冲突的要求，这是一个无法解决的问题。

使用命名空间包，您可以独立存储每个子包的源树：

```py
$ tree acme.sql/
acme.sql/
├── acme
│   └── sql
│       └── __init__.py
└── setup.py

2 directories, 2 files

$ tree acme.templating/
acme.templating/
├── acme
│   └── templating
│       └── __init__.py
└── setup.py

2 directories, 2 files

```

您还可以在 PyPI 或您使用的任何包索引中独立注册它们。用户可以选择从`acme`命名空间安装哪些子包，但他们永远不会安装通用的`acme`包（它不存在）：

```py
$ pip install acme.sql acme.templating

```

请注意，独立的源树不足以在 Python 中创建命名空间包。如果您不希望您的包互相覆盖，您需要做一些额外的工作。此外，根据您的 Python 语言版本目标，正确的处理可能会有所不同。这方面的细节在接下来的两节中描述。

## PEP 420 - 隐式命名空间包

如果您只使用和针对 Python 3，那么对您来说有个好消息。**PEP 420（隐式命名空间包）**引入了一种新的定义命名空间包的方法。它是标准跟踪的一部分，并且自 3.3 版本以来成为语言的官方部分。简而言之，如果一个目录包含 Python 包或模块（包括命名空间包），并且不包含`__init__.py`文件，则被视为命名空间包。因此，以下是在上一节中介绍的文件结构示例：

```py
$ tree acme.sql/
acme.sql/
├── acme
│   └── sql
│       └── __init__.py
└── setup.py

2 directories, 2 files

$ tree acme.templating/
acme.templating/
├── acme
│   └── templating
│       └── __init__.py
└── setup.py

2 directories, 2 files

```

它们足以定义`acme`是 Python 3.3 及更高版本中的命名空间包。使用设置工具的最小`setup.py`脚本将如下所示：

```py
from setuptools import setup

setup(
 **name='acme.templating',
 **packages=['acme.templating'],
)

```

不幸的是，在撰写本书时，`setuptools.find_packages()`不支持 PEP 420。无论如何，这在将来可能会改变。此外，明确定义包列表的要求似乎是易于集成命名空间包的一个非常小的代价。

## 在以前的 Python 版本中的命名空间包

PEP 420 布局中的命名空间包在 Python 3.3 之前的版本中无法工作。然而，这个概念非常古老，在像 Zope 这样的成熟项目中经常被使用，因此肯定可以使用它们，但不能进行隐式定义。在 Python 的旧版本中，有几种方法可以定义包应该被视为命名空间。

最简单的方法是为每个组件创建一个文件结构，类似于普通包布局而不是命名空间包，并将一切交给`setuptools`。因此，`acme.sql`和`acme.templating`的示例布局可能如下所示：

```py
$ tree acme.sql/
acme.sql/
├── acme
│   ├── __init__.py
│   └── sql
│       └── __init__.py
└── setup.py

2 directories, 3 files

$ tree acme.templating/
acme.templating/
├── acme
│   ├── __init__.py
│   └── templating
│       └── __init__.py
└── setup.py

2 directories, 3 files

```

请注意，对于`acme.sql`和`acme.templating`，还有一个额外的源文件`acme/__init__.py`。这个文件必须保持空白。如果我们将这个名称作为`setuptools.setup()`函数的`namespace_packages`关键字参数的值提供，`acme`命名空间包将被创建：

```py
from setuptools import setup

setup(
    name='acme.templating',
    packages=['acme.templating'],
    namespace_packages=['acme'],
)
```

最简单并不意味着最好。为了注册一个新的命名空间，`setuptools`将在您的`__init__.py`文件中调用`pkg_resources.declare_namespace()`函数。即使`__init__.py`文件是空的，也会发生这种情况。无论如何，正如官方文档所说，声明命名空间在`__init__.py`文件中是您自己的责任，`setuptools`的这种隐式行为可能会在将来被取消。为了安全和"未来证明"，您需要在文件`acme/__init__.py`中添加以下行：

```py
__import__('pkg_resources').declare_namespace(__name__)
```

# 上传软件包

没有组织的方式存储、上传和下载软件包将是无用的。Python 软件包索引是 Python 社区中开源软件包的主要来源。任何人都可以自由上传新软件包，唯一的要求是在 PyPI 网站上注册-[`pypi.python.org/pypi`](https://pypi.python.org/pypi)。

当然，您不仅限于这个索引，所有打包工具都支持使用替代软件包存储库。这对于在内部组织中分发闭源代码或用于部署目的尤其有用。如何使用这样的打包工具以及如何创建自己的软件包索引的说明将在下一章中解释。在这里，我们只关注向 PyPI 上传开源软件，只简要提及如何指定替代存储库。

## PyPI- Python 软件包索引

Python 软件包索引，如前所述，是开源软件包分发的官方来源。从中下载不需要任何帐户或权限。您唯一需要的是一个可以从 PyPI 下载新分发包的软件包管理器。您应该首选`pip`。

### 上传到 PyPI-或其他软件包索引

任何人都可以注册并上传软件包到 PyPI，只要他或她已经注册了帐户。软件包与用户绑定，因此，默认情况下，只有注册软件包名称的用户是其管理员，并且可以上传新的分发包。这可能对于更大的项目来说是一个问题，因此有一个选项可以将其他用户设计为软件包维护者，以便他们能够上传新的分发包。

上传软件包的最简单方法是使用`setup.py`脚本的`upload`命令：

```py
$ python setup.py <dist-commands> upload

```

在这里，`<dist-commands>`是一个创建要上传的分发包的命令列表。只有在同一次`setup.py`执行期间创建的分发包才会上传到存储库。因此，如果您要同时上传源分发包、构建分发包和 wheel 软件包，那么您需要发出以下命令：

```py
$ python setup.py sdist bdist bdist_wheel upload

```

在使用`setup.py`上传时，您不能重复使用已构建的分发包，并且被迫在每次上传时重新构建它们。这可能有些合理，但对于大型或复杂的项目来说可能不方便，因为创建分发包可能需要相当长的时间。`setup.py upload`的另一个问题是，它可能在某些 Python 版本上使用明文 HTTP 或未经验证的 HTTPS 连接。这就是为什么建议使用`twine`作为`setup.py upload`的安全替代品。

Twine 是与 PyPI 交互的实用程序，目前只提供一个功能-安全地上传软件包到存储库。它支持任何打包格式，并始终确保连接是安全的。它还允许您上传已经创建的文件，因此您可以在发布之前测试分发包。`twine`的一个示例用法仍然需要调用`setup.py`来构建分发包：

```py
$ python setup.py sdist bdist_wheel
$ twine upload dist/*

```

如果您尚未注册此软件包，则上传将失败，因为您需要先注册它。这也可以使用`twine`来完成：

```py
$ twine register dist/*

```

### .pypirc

`.pypirc`是一个存储有关 Python 软件包存储库信息的配置文件。它应该位于您的主目录中。该文件的格式如下：

```py
[distutils]
index-servers =
    pypi
    other

[pypi]
repository: <repository-url>
username: <username>
password: <password>

[other]
repository: https://example.com/pypi
username: <username>
password: <password>
```

`distutils`部分应该有`index-servers`变量，列出所有描述所有可用存储库和其凭据的部分。对于每个存储库部分，只有三个变量可以修改：

+   `存储库`：这是软件包存储库的 URL（默认为[`www.python.org/pypi`](https://www.python.org/pypi)）

+   `用户名`：这是在给定存储库中进行授权的用户名

+   `密码`：这是用于授权的用户密码，以明文形式

请注意，以明文形式存储存储库密码可能不是明智的安全选择。您可以始终将其留空，并在必要时提示输入密码。

`.pypirc`文件应该受到为 Python 构建的每个打包工具的尊重。虽然这对于每个与打包相关的实用程序来说可能并不正确，但它得到了最重要的工具的支持，如`pip`、`twine`、`distutils`和`setuptools`。

## 源包与构建包

Python 软件包通常有两种类型的分发：

+   源分发

+   构建（二进制）分发

源分发是最简单和最独立于平台的。对于纯 Python 软件包，这是毫无疑问的。这种分发只包含 Python 源代码，这些源代码应该已经非常易于移植。

更复杂的情况是，当您的软件包引入一些扩展时，例如用 C 编写的扩展。只要软件包用户在其环境中具有适当的开发工具链，源分发仍将起作用。这主要包括编译器和适当的 C 头文件。对于这种情况，构建的分发格式可能更适合，因为它可能已经为特定平台提供了构建好的扩展。

### sdist

`sdist`命令是最简单的可用命令。它创建一个发布树，其中复制了运行软件包所需的一切。然后将此树存档在一个或多个存档文件中（通常只创建一个 tarball）。存档基本上是源树的副本。

这个命令是从目标系统独立地分发软件包的最简单方法。它创建一个包含存档的`dist`文件夹，可以进行分发。为了使用它，必须向`setup`传递一个额外的参数来提供版本号。如果不给它一个`version`值，它将使用`version = 0.0.0`：

```py
from setuptools import setup

setup(name='acme.sql', version='0.1.1')
```

这个数字对于升级安装是有用的。每次发布软件包时，都会提高这个数字，以便目标系统知道它已经更改。

让我们使用这个额外的参数运行`sdist`命令：

```py
$ python setup.py sdist
running sdist
...
creating dist
tar -cf dist/acme.sql-0.1.1.tar acme.sql-0.1.1
gzip -f9 dist/acme.sql-0.1.1.tar
removing 'acme.sql-0.1.1' (and everything under it)
$ ls dist/
acme.sql-0.1.1.tar.gz

```

### 注意

在 Windows 下，归档将是一个 ZIP 文件。

版本用于标记存档的名称，可以在任何安装了 Python 的系统上分发和安装。在`sdist`分发中，如果软件包包含 C 库或扩展，目标系统负责编译它们。这在基于 Linux 的系统或 Mac OS 中非常常见，因为它们通常提供编译器，但在 Windows 下很少见。这就是为什么当软件包打算在多个平台上运行时，应该始终使用预构建的分发进行分发。

### bdist 和 wheels

为了能够分发预构建的分发，`distutils`提供了`build`命令，它在四个步骤中编译软件包：

+   `build_py`：这将通过对其进行字节编译并将其复制到构建文件夹中来构建纯 Python 模块。

+   `build_clib`：当软件包包含任何 C 库时，使用 C 编译器构建 C 库并在构建文件夹中创建一个静态库。

+   `build_ext`：这将构建 C 扩展并将结果放在构建文件夹中，如`build_clib`。

+   `build_scripts`：这将构建标记为脚本的模块。当第一行被设置为（`!#`）时，它还会更改解释器路径，并修复文件模式，使其可执行。

这些步骤中的每一步都是可以独立调用的命令。编译过程的结果是一个包含了安装软件包所需的一切的构建文件夹。`distutils`包中还没有交叉编译器选项。这意味着命令的结果始终特定于它所构建的系统。

当需要创建一些 C 扩展时，构建过程使用系统编译器和 Python 头文件（`Python.h`）。这个**include**文件是从 Python 构建源代码时就可用的。对于打包的发行版，可能需要额外的系统发行版包。至少在流行的 Linux 发行版中，通常被命名为`python-dev`。它包含了构建 Python 扩展所需的所有必要的头文件。

使用的 C 编译器是系统编译器。对于基于 Linux 的系统或 Mac OS X，分别是**gcc**或**clang**。对于 Windows，可以使用 Microsoft Visual C++（有免费的命令行版本可用），也可以使用开源项目 MinGW。这可以在`distutils`中配置。

`build`命令由`bdist`命令用于构建二进制分发。它调用`build`和所有依赖的命令，然后以与`sdist`相同的方式创建存档。

让我们在 Mac OS X 下为`acme.sql`创建一个二进制发行版：

```py
$ python setup.py bdist
running bdist
running bdist_dumb
running build
...
running install_scripts
tar -cf dist/acme.sql-0.1.1.macosx-10.3-fat.tar .
gzip -f9 acme.sql-0.1.1.macosx-10.3-fat.tar
removing 'build/bdist.macosx-10.3-fat/dumb' (and everything under it)
$ ls dist/
acme.sql-0.1.1.macosx-10.3-fat.tar.gz    acme.sql-0.1.1.tar.gz

```

请注意，新创建的存档名称包含了系统名称和它构建的发行版名称（*Mac OS X 10.3*）。

在 Windows 下调用相同的命令将创建一个特定的分发存档：

```py
C:\acme.sql> python.exe setup.py bdist
...
C:\acme.sql> dir dist
25/02/2008  08:18    <DIR>          .
25/02/2008  08:18    <DIR>          ..
25/02/2008  08:24            16 055 acme.sql-0.1.win32.zip
 **1 File(s)         16 055 bytes
 **2 Dir(s)  22 239 752 192 bytes free

```

如果软件包包含 C 代码，除了源分发外，释放尽可能多的不同二进制分发是很重要的。至少，对于那些没有安装 C 编译器的人来说，Windows 二进制分发是很重要的。

二进制发行版包含一个可以直接复制到 Python 树中的树。它主要包含一个文件夹，该文件夹被复制到 Python 的`site-packages`文件夹中。它还可能包含缓存的字节码文件（在 Python 2 上为`*.pyc`文件，在 Python 3 上为`__pycache__/*.pyc`）。

另一种构建分发是由`wheel`包提供的“wheels”。当安装（例如，使用`pip`）时，`wheel`会向`distutils`添加一个新的`bdist_wheel`命令。它允许创建特定于平台的分发（目前仅适用于 Windows 和 Mac OS X），为普通的`bdist`分发提供了替代方案。它旨在取代`setuptools`早期引入的另一种分发——eggs。Eggs 现在已经过时，因此不会在这里介绍。使用 wheels 的优势列表非常长。以下是 Python Wheels 页面（[`pythonwheels.com/`](http://pythonwheels.com/)）中提到的优势：

+   纯 Python 和本地 C 扩展包的更快安装

+   避免安装时的任意代码执行。（避免`setup.py`）

+   在 Windows 或 OS X 上安装 C 扩展不需要编译器

+   允许更好的缓存用于测试和持续集成

+   创建`.pyc`文件作为安装的一部分，以确保它们与使用的 Python 解释器匹配

+   跨平台和机器上的安装更一致

根据 PyPA 的建议，wheels 应该是您的默认分发格式。不幸的是，Linux 的特定平台 wheels 目前还不可用，因此如果您必须分发带有 C 扩展的软件包，那么您需要为 Linux 用户创建`sdist`分发。

# 独立可执行文件

创建独立的可执行文件是 Python 代码打包材料中常常被忽视的一个话题。这主要是因为 Python 在其标准库中缺乏适当的工具，允许程序员创建简单的可执行文件，用户可以在不需要安装 Python 解释器的情况下运行。

编译语言在一个重要方面比 Python 具有优势，那就是它们允许为给定的系统架构创建可执行应用程序，用户可以以一种不需要了解底层技术的方式运行。Python 代码在作为包分发时需要 Python 解释器才能运行。这给没有足够技术能力的用户带来了很大的不便。

开发人员友好的操作系统，比如 Mac OS X 或大多数 Linux 发行版，都预装了 Python。因此，对于他们的用户，基于 Python 的应用程序仍然可以作为依赖于主脚本文件中特定**解释器指令**的源代码包进行分发，这通常被称为**shebang**。对于大多数 Python 应用程序，这采用以下形式：

```py
#!/usr/bin/env python
```

这样的指令，当作为脚本的第一行使用时，将默认标记为由给定环境的 Python 版本解释。当然，这可以更详细地表达，需要特定的 Python 版本，比如`python3.4`、`python3`或`python2`。请注意，这将在大多数流行的 POSIX 系统中工作，但根据定义，这在任何情况下都不具备可移植性。这个解决方案依赖于特定的 Python 版本的存在，以及`env`可执行文件确切地位于`/usr/bin/env`。这些假设都可能在某些操作系统上失败。另外，shebang 在 Windows 上根本不起作用。此外，即使对于经验丰富的开发人员，在 Windows 上启动 Python 环境也可能是一个挑战，因此你不能指望非技术用户能够自己做到这一点。

另一件要考虑的事情是在桌面环境中的简单用户体验。用户通常希望可以通过简单点击桌面上的应用程序来运行它们。并非每个桌面环境都支持将 Python 应用程序作为源代码分发后以这种方式运行。

因此，最好能够创建一个二进制分发，它可以像任何其他编译的可执行文件一样工作。幸运的是，可以创建一个既包含 Python 解释器又包含我们项目的可执行文件。这允许用户打开我们的应用程序，而不必关心 Python 或任何其他依赖项。

## 独立的可执行文件何时有用？

独立的可执行文件在用户体验的简单性比用户能够干预应用程序代码更重要的情况下是有用的。请注意，仅仅将应用程序作为可执行文件分发只会使代码阅读或修改变得更加困难，而不是不可能。这不是保护应用程序代码的方法，应该只用作使与应用程序交互的方式更简单的方法。

独立的可执行文件应该是为非技术终端用户分发应用程序的首选方式，似乎也是为 Windows 分发 Python 应用程序的唯一合理方式。

独立的可执行文件通常是一个不错的选择：

+   依赖于目标操作系统上可能不容易获得的特定 Python 版本的应用程序

+   依赖于修改后的预编译的 CPython 源代码的应用程序

+   具有图形界面的应用程序

+   具有许多用不同语言编写的二进制扩展的项目

+   游戏

## 流行的工具

Python 没有任何内置支持来构建独立的可执行文件。幸运的是，有一些社区项目在解决这个问题，取得了不同程度的成功。最值得注意的四个是：

+   PyInstaller

+   cx_Freeze

+   py2exe

+   py2app

它们每一个在使用上都略有不同，而且每一个都有略微不同的限制。在选择工具之前，您需要决定要针对哪个平台，因为每个打包工具只能支持特定的操作系统集。

最好的情况是在项目的早期阶段就做出这样的决定。当然，这些工具都不需要在您的代码中进行深入的交互，但是如果您早期开始构建独立的软件包，您可以自动化整个过程，并节省未来的集成时间和成本。如果您把这个留到以后，您可能会发现项目构建得非常复杂，以至于没有任何可用的工具可以使用。为这样的项目提供一个独立的可执行文件将是困难的，并且会花费大量的时间。

### PyInstaller

PyInstaller（[`www.pyinstaller.org/`](http://www.pyinstaller.org/)）是目前将 Python 软件包冻结为独立可执行文件的最先进的程序。它在目前所有可用的解决方案中提供了最广泛的多平台兼容性，因此是最推荐的。PyInstaller 支持的平台有：

+   Windows（32 位和 64 位）

+   Linux（32 位和 64 位）

+   Mac OS X（32 位和 64 位）

+   FreeBSD、Solaris 和 AIX

支持的 Python 版本是 Python 2.7 和 Python 3.3、3.4 和 3.5。它可以在 PyPI 上找到，因此可以使用`pip`在您的工作环境中安装它。如果您在安装时遇到问题，您可以随时从项目页面下载安装程序。

不幸的是，不支持跨平台构建（交叉编译），因此如果您想为特定平台构建独立的可执行文件，那么您需要在该平台上执行构建。随着许多虚拟化工具的出现，这在今天并不是一个大问题。如果您的计算机上没有安装特定的系统，您可以随时使用 Vagrant，它将为您提供所需的操作系统作为虚拟机。

简单应用程序的使用很容易。假设我们的应用程序包含在名为`myscript.py`的脚本中。这是一个简单的“Hello world！”应用程序。我们想为 Windows 用户创建一个独立的可执行文件，并且我们的源代码位于文件系统中的`D://dev/app`下。我们的应用程序可以使用以下简短的命令进行打包：

```py
$ pyinstaller myscript.py

2121 INFO: PyInstaller: 3.1
2121 INFO: Python: 2.7.10
2121 INFO: Platform: Windows-7-6.1.7601-SP1
2121 INFO: wrote D:\dev\app\myscript.spec
2137 INFO: UPX is not available.
2138 INFO: Extending PYTHONPATH with paths
['D:\\dev\\app', 'D:\\dev\\app']
2138 INFO: checking Analysis
2138 INFO: Building Analysis because out00-Analysis.toc is non existent
2138 INFO: Initializing module dependency graph...
2154 INFO: Initializing module graph hooks...
2325 INFO: running Analysis out00-Analysis.toc
(...)
25884 INFO: Updating resource type 24 name 2 language 1033

```

PyInstaller 的标准输出即使对于简单的应用程序也非常长，因此为了简洁起见，在前面的示例中进行了截断。如果在 Windows 上运行，目录和文件的结果结构将如下所示：

```py
$ tree /0066
│   myscript.py
│   myscript.spec
│
├───build
│   └───myscript
│           myscript.exe
│           myscript.exe.manifest
│           out00-Analysis.toc
│           out00-COLLECT.toc
│           out00-EXE.toc
│           out00-PKG.pkg
│           out00-PKG.toc
│           out00-PYZ.pyz
│           out00-PYZ.toc
│           warnmyscript.txt
│
└───dist
 **└───myscript
 **bz2.pyd
 **Microsoft.VC90.CRT.manifest
 **msvcm90.dll
 **msvcp90.dll
 **msvcr90.dll
 **myscript.exe
 **myscript.exe.manifest
 **python27.dll
 **select.pyd
 **unicodedata.pyd
 **_hashlib.pyd

```

`dist/myscript`目录包含了可以分发给用户的构建应用程序。请注意，整个目录必须被分发。它包含了运行我们的应用程序所需的所有附加文件（DLL、编译的扩展库等）。可以使用`pyinstaller`命令的`--onefile`开关获得更紧凑的分发：

```py
$ pyinstaller --onefile myscript.py
(...)
$ tree /f
├───build
│   └───myscript
│           myscript.exe.manifest
│           out00-Analysis.toc
│           out00-EXE.toc
│           out00-PKG.pkg
│           out00-PKG.toc
│           out00-PYZ.pyz
│           out00-PYZ.toc
│           warnmyscript.txt
│
└───dist
 **myscript.exe

```

使用`--onefile`选项构建时，您需要分发给其他用户的唯一文件是`dist`目录中找到的单个可执行文件（这里是`myscript.exe`）。对于小型应用程序，这可能是首选选项。

运行`pyinstaller`命令的一个副作用是创建`*.spec`文件。这是一个自动生成的 Python 模块，包含了如何从您的源代码创建可执行文件的规范。例如，我们已经在以下代码中使用了这个：

```py
# -*- mode: python -*-

block_cipher = None

a = Analysis(['myscript.py'],
             pathex=['D:\\dev\\app'],
             binaries=None,
             datas=None,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='myscript',
          debug=False,
          strip=False,
          upx=True,
          console=True )
```

这个`.spec`文件包含了之前指定的所有`pyinstaller`参数。如果您对构建进行了大量的自定义，这将非常有用，因为这可以代替必须存储您的配置的构建脚本。创建后，您可以将其用作`pyinstaller`命令的参数，而不是您的 Python 脚本：

```py
$ pyinstaller.exe myscript.spec

```

请注意，这是一个真正的 Python 模块，因此您可以使用自己已经了解的语言对其进行扩展并对构建过程进行更复杂的自定义。当您针对许多不同的平台时，自定义`.spec`文件尤其有用。此外，并非所有的`pyinstaller`选项都可以通过命令行参数使用，只有在修改`.spec`文件时才能使用。

PyInstaller 是一个功能强大的工具，使用起来对于大多数程序来说非常简单。无论如何，如果您有兴趣将其作为分发应用程序的工具，建议仔细阅读其文档。

### cx_Freeze

cx_Freeze ([`cx-freeze.sourceforge.net/`](http://cx-freeze.sourceforge.net/))是另一个用于创建独立可执行文件的工具。它比 PyInstaller 更简单，但也支持三个主要平台：

+   Windows

+   Linux

+   Mac OS X

与 PyInstaller 一样，它不允许我们执行跨平台构建，因此您需要在分发到的相同操作系统上创建您的可执行文件。cx_Freeze 的主要缺点是它不允许我们创建真正的单文件可执行文件。使用它构建的应用程序需要与相关的 DLL 文件和库一起分发。假设我们有与*PyInstaller*部分中的相同应用程序，那么示例用法也非常简单：

```py
$ cxfreeze myscript.py

copying C:\Python27\lib\site-packages\cx_Freeze\bases\Console.exe -> D:\dev\app\dist\myscript.exe
copying C:\Windows\system32\python27.dll -> D:\dev\app\dist\python27.dll
writing zip file D:\dev\app\dist\myscript.exe
(...)
copying C:\Python27\DLLs\bz2.pyd -> D:\dev\app\dist\bz2.pyd
copying C:\Python27\DLLs\unicodedata.pyd -> D:\dev\app\dist\unicodedata.pyd

```

生成的文件结构如下：

```py
$ tree /f
│   myscript.py
│
└───dist
 **bz2.pyd
 **myscript.exe
 **python27.dll
 **unicodedata.pyd

```

cx_Freeze 不是提供自己的构建规范格式（就像 PyInstaller 一样），而是扩展了`distutils`包。这意味着您可以使用熟悉的`setup.py`脚本配置独立可执行文件的构建方式。如果您已经使用`setuptools`或`distutils`来分发软件包，那么 cx_Freeze 非常方便，因为额外的集成只需要对`setup.py`脚本进行小的更改。以下是一个使用`cx_Freeze.setup()`创建 Windows 独立可执行文件的`setup.py`脚本示例：

```py
import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": ["os"], "excludes": ["tkinter"]}

setup(
    name="myscript",
    version="0.0.1",
    description="My Hello World application!",
    options={
        "build_exe": build_exe_options
    },
    executables=[Executable("myscript.py")]
)
```

有了这样一个文件，可以使用添加到`setup.py`脚本的新`build_exe`命令来创建新的可执行文件：

```py
$ python setup.py build_exe

```

cx_Freeze 的使用似乎比 PyInstaller 和`distutils`集成更容易一些，这是一个非常有用的功能。不幸的是，这个项目可能会给经验不足的开发人员带来一些麻烦：

+   在 Windows 下使用`pip`进行安装可能会有问题

+   官方文档非常简短，某些地方缺乏说明

### py2exe 和 py2app

py2exe ([`www.py2exe.org/`](http://www.py2exe.org/))和 py2app ([`pythonhosted.org/py2app/`](https://pythonhosted.org/py2app/))是另外两个集成到 Python 打包中的程序，可以通过`distutils`或`setuptools`创建独立可执行文件。它们被一起提到，因为它们在使用和限制方面非常相似。py2exe 和 py2app 的主要缺点是它们只针对单个平台：

+   py2exe 允许构建 Windows 可执行文件

+   py2app 允许构建 Mac OS X 应用程序

由于使用方法非常相似，只需要修改`setup.py`脚本，这些软件包似乎互补。py2app 项目的官方文档提供了以下`setup.py`脚本示例，可以根据所使用的平台使用正确的工具（py2exe 或 py2app）构建独立可执行文件：

```py
import sys
from setuptools import setup

mainscript = 'MyApplication.py'

if sys.platform == 'darwin':
    extra_options = dict(
        setup_requires=['py2app'],
        app=[mainscript],
        # Cross-platform applications generally expect sys.argv to
        # be used for opening files.
        options=dict(py2app=dict(argv_emulation=True)),
    )
elif sys.platform == 'win32':
    extra_options = dict(
        setup_requires=['py2exe'],
        app=[mainscript],
    )
else:
    extra_options = dict(
        # Normally unix-like platforms will use "setup.py install"
        # and install the main script as such
        scripts=[mainscript],
    )

setup(
    name="MyApplication",
    **extra_options
)
```

使用这样的脚本，您可以使用`python setup.py py2exe`命令构建 Windows 可执行文件，并使用`python setup.py py2app`构建 Mac OS X 应用程序。当然，跨编译是不可能的。

尽管 cx_Freeze 的一些限制和弹性不如 PyInstaller 或 cx_Freeze，但了解总是有 py2exe 和 py2app 项目。在某些情况下，PyInstaller 或 cx_Freeze 可能无法正确地构建项目的可执行文件。在这种情况下，值得检查其他解决方案是否能够处理我们的代码。

## 可执行软件包中 Python 代码的安全性

重要的是要知道，独立可执行文件并不以任何方式使应用程序代码安全。从这种可执行文件中反编译嵌入的代码并不是一件容易的任务，但肯定是可行的。更重要的是，这种反编译的结果（如果使用适当的工具进行）可能看起来与原始源代码非常相似。

这个事实使得独立的 Python 可执行文件对于泄漏应用程序代码可能会损害组织的闭源项目来说并不是一个可行的解决方案。因此，如果你的整个业务可以通过简单地复制应用程序的源代码来复制，那么你应该考虑其他分发应用程序的方式。也许提供软件作为服务对你来说会是更好的选择。

### 使反编译变得更加困难

正如已经说过的，目前没有可靠的方法可以防止应用程序被反编译。但是，有一些方法可以使这个过程变得更加困难。但更困难并不意味着不太可能。对于我们中的一些人来说，最具诱惑力的挑战是最困难的挑战。我们都知道，这个挑战的最终奖励是非常高的：您试图保护的代码。

通常，反编译的过程包括几个步骤：

1.  从独立可执行文件中提取项目的字节码的二进制表示。

1.  将二进制表示映射到特定 Python 版本的字节码。

1.  将字节码转换为 AST。

1.  直接从 AST 重新创建源代码。

提供确切的解决方案来阻止开发人员对独立可执行文件进行逆向工程将是毫无意义的，因为这是显而易见的原因。因此，这里只提供了一些阻碍反编译过程或贬值其结果的想法：

+   删除运行时可用的任何代码元数据（文档字符串），因此最终结果会变得不太可读

+   修改 CPython 解释器使用的字节码值，以便从二进制转换为字节码，然后再转换为 AST 需要更多的工作

+   使用经过复杂修改的 CPython 源代码版本，即使可用应用程序的反编译源代码也无法在没有反编译修改后的 CPython 二进制文件的情况下使用

+   在将源代码捆绑成可执行文件之前，使用混淆脚本对源代码进行混淆，这样在反编译后源代码的价值就会降低

这些解决方案使开发过程变得更加困难。上述一些想法需要对 Python 运行时有很深的理解，但它们每一个都充满了许多陷阱和缺点。大多数情况下，它们只是推迟了不可避免的结果。一旦你的技巧被破解，所有额外的努力都将成为时间和资源的浪费。

不允许您的闭源代码以任何形式直接发货给用户是唯一可靠的方法。只有在您组织的其他方面保持严密的安全性时，这才是真实的。

# 摘要

本章描述了 Python 的打包生态系统的细节。现在，在阅读完本章之后，您应该知道哪些工具适合您的打包需求，以及您的项目需要哪些类型的分发。您还应该知道常见问题的流行技术以及如何为您的项目提供有用的元数据。

我们还讨论了独立可执行文件的话题，这些文件非常有用，特别是在分发桌面应用程序时。

下一章将广泛依赖我们在这里学到的知识，展示如何以可靠和自动化的方式有效处理代码部署。
