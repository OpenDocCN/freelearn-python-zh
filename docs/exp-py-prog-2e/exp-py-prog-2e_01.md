# 第一章：Python 的当前状态

Python 对开发人员来说是很好的。

无论您或您的客户使用什么操作系统，它都可以工作。除非您编写特定于平台的内容，或者使用特定于平台的库，您可以在 Linux 上工作，然后在其他系统上部署，例如。然而，这已经不再是不寻常的事情了（Ruby、Java 和许多其他语言也是这样工作的）。再加上我们将在本书中发现的其他优点，Python 成为公司主要开发语言的明智选择。

本书专注于 Python 的最新版本 3.5，并且所有的代码示例都是用这个版本的语言编写的，除非另一个版本被明确提到。因为这个版本还没有被广泛使用，本章包含了一些关于 Python 3 当前现状的描述，以便向读者介绍它，以及一些关于 Python 开发的现代方法的介绍信息。本章涵盖以下主题：

+   如何在 Python 2 和 Python 3 之间保持兼容性

+   如何在应用程序和操作系统级别上处理环境隔离问题，以便进行开发

+   如何增强 Python 提示

+   如何使用 pip 安装软件包

一本书总是以一些开胃菜开始的。因此，如果您已经熟悉 Python（尤其是最新的 3.x 分支），并且知道如何正确地隔离开发环境，您可以跳过本章的前两节，快速阅读其他部分。它们描述了一些非必要但可以极大提高 Python 生产力的工具和资源。但是一定要阅读有关应用级环境隔离和 pip 的部分，因为它们的安装对本书的其余部分是强制性的。

# 我们现在在哪里，我们将走向何方？

Python 的历史始于 20 世纪 80 年代末，但它的 1.0 发布日期是在 1994 年，所以它并不是一种非常年轻的语言。这里可以提到一系列主要的 Python 发布日期，但真正重要的是一个日期：2008 年 12 月 3 日——Python 3.0 的发布日期。

在撰写本文时，距离第一个 Python 3 发布已经过去了七年。自 PEP 404 的创建以来已经过去了四年，这是官方文件，它“取消发布”了 Python 2.8，并正式关闭了 2.x 分支。尽管已经过去了很长时间，但在 Python 社区中存在着一个特定的二分法——虽然语言发展非常迅速，但有一大群用户不愿意向前发展。

# 为什么以及 Python 如何改变？

答案很简单——Python 之所以改变，是因为有这样的需求。竞争是不会停止的。每隔几个月，就会有一种新语言突然冒出来，声称解决了所有先前语言的问题。大多数这样的项目很快就失去了开发人员的注意，它们的流行程度是由突然的热潮所推动的。

无论如何，这都是一些更大问题的迹象。人们设计新语言，是因为他们发现现有的语言无法以最佳方式解决他们的问题。不承认这种需求是愚蠢的。此外，Python 的越来越广泛的使用表明它在许多方面都可以，也应该得到改进。

Python 的许多改进通常是由其使用领域的特定需求驱动的。最重要的一个是 Web 开发，这需要改进来处理 Python 中的并发性。

一些变化仅仅是由于 Python 项目的年龄和成熟度所致。多年来，它已经收集了一些混乱的东西，例如非组织化和冗余的标准库模块或一些糟糕的设计决策。首先，Python 3 发布旨在为语言带来重大的清理和更新，但时间表明这个计划有些事与愿违。很长一段时间以来，它被许多开发人员视为一种好奇心，但希望情况正在改变。

# 了解最新变化-PEP 文档

Python 社区有一种处理变化的成熟方式。虽然关于 Python 语言的推测性想法大多在特定的邮件列表上讨论（`<python-ideas@python.org>`），但没有新的文件称为 PEP，就不会发生重大变化。**PEP**是**Python Enhancement Proposal**的缩写。它是提出对 Python 的改变的文件，也是社区讨论的起点。这些文件的整个目的、格式和工作流程也是以 Python Enhancement Proposal 的形式标准化的，准确地说，是 PEP 1 文档（[`www.python.org/dev/peps/pep-0001`](http://www.python.org/dev/peps/pep-0001)）。

PEP 文档对 Python 非常重要，根据主题的不同，它们有不同的用途：

+   **通知**：它们总结了核心 Python 开发人员所需的信息，并通知 Python 发布时间表

+   **标准化**：它们提供代码风格、文档或其他指南

+   **设计**：它们描述了提出的功能。

所有提出的 PEP 的列表都可以在一个文档中找到——PEP 0（[`www.python.org/dev/peps/`](https://www.python.org/dev/peps/)）。由于它们在一个地方很容易获得，而且实际的 URL 也很容易猜到，它们通常在书中按编号引用。 

对于那些想知道 Python 语言发展方向的人，但没有时间跟踪 Python 邮件列表上的讨论，PEP 0 文档可以是一个很好的信息来源。它显示了哪些文件已经被接受但尚未实施，以及哪些仍在考虑中。

PEP 还有其他用途。人们经常问类似的问题：

+   为什么 A 功能以这种方式工作？

+   为什么 Python 没有 B 功能？

在大多数这样的情况下，详细答案可以在特定的 PEP 文档中找到，其中已经提到了这样的功能。有很多 PEP 文档描述了提出但未被接受的 Python 语言功能。这些文件被留作历史参考。

# 写作本书时的 Python 3 采用情况

那么，由于新的激动人心的功能，Python 3 在其社区中被广泛采用吗？可惜，还没有。跟踪 Python 3 分支与大多数流行软件包兼容性的流行页面 Python 3 Wall of Superpowers（[`python3wos.appspot.com`](https://python3wos.appspot.com)）直到不久前还被称为 Python 3 Wall of Shame。这种情况正在改变，提到的页面上列出的软件包表格每个月都在慢慢变得“更绿”。但是，这并不意味着所有构建应用程序的团队很快就会只使用 Python 3。当所有流行软件包都在 Python 3 上可用时，常见的借口——我们使用的软件包尚未移植——将不再有效。

这种情况的主要原因是，将现有应用程序从 Python 2 迁移到 Python 3 始终是一个挑战。有一些工具，比如 2to3 可以执行自动代码转换，但不能保证结果是 100%正确。此外，这样翻译的代码可能不如原始形式那样表现良好，需要手动调整。将现有复杂代码库移植到 Python 3 可能需要巨大的努力和成本，一些组织可能无法承担。但这样的成本可以分摊。一些良好的软件架构设计方法，如面向服务的架构或微服务，可以帮助逐步实现这一目标。新项目组件（服务或微服务）可以使用新技术编写，现有项目可以逐个移植。

从长远来看，转移到 Python 3 对项目只会产生有益的影响。根据 PEP-404，在 Python 2.x 分支中将不再发布 2.8 版本。此外，将来可能会有一段时间，像 Django、Flask 和 numpy 这样的所有主要项目都将放弃任何 2.x 兼容性，只能在 Python 3 上使用。

我对这个话题的个人看法可能是有争议的。我认为对于社区来说，最好的激励是在创建新软件包时完全放弃对 Python 2 的支持。当然，这大大限制了这类软件的影响范围，但这可能是改变那些坚持使用 Python 2.x 的人思维方式的唯一途径。

# Python 3 和 Python 2 之间的主要区别

已经说过 Python 3 与 Python 2 破坏了向后兼容性。但这并不意味着完全重新设计。也不意味着每个为 2.x 版本编写的 Python 模块都将在 Python 3 下停止工作。可以编写完全跨兼容的代码，将在两个主要版本上运行，而无需额外的工具或技术，但通常只适用于简单的应用程序。

## 我为什么要在意？

尽管我在本章前面提到了我对 Python 2 兼容性的个人看法，但现在不可能就这样忘记它。仍然有一些有用的包（比如在第六章中提到的 fabric，*部署代码*）真的值得使用，但在不久的将来可能不会被移植。

有时候，我们可能会受到我们所在组织的限制。现有的遗留代码可能非常复杂，以至于移植它在经济上是不可行的。因此，即使我们决定从现在开始只在 Python 3 世界中生活，也不可能在一段时间内完全不使用 Python 2。

如今，要成为专业开发人员，很难不回馈社区，因此帮助开源开发者将现有软件包添加 Python 3 兼容性是偿还使用它们所带来的“道义债务”的好方法。当然，这是不可能做到的，而不知道 Python 2 和 Python 3 之间的差异。顺便说一句，这对于那些刚接触 Python 3 的人来说也是一个很好的练习。

## 主要的语法差异和常见陷阱

Python 文档是每个版本之间差异的最佳参考。无论如何，为了方便读者，本节总结了最重要的差异。这并不改变文档对于那些尚不熟悉 Python 3 的人来说是必读的事实（参见[`docs.python.org/3.0/whatsnew/3.0.html`](https://docs.python.org/3.0/whatsnew/3.0.html)）。

Python 3 引入的破坏性更改通常可以分为几个组：

+   语法更改，其中一些语法元素被移除/更改，其他元素被添加

+   标准库的更改

+   数据类型和集合的更改

### 语法更改

使现有代码难以运行的语法更改是最容易发现的——它们将导致代码根本无法运行。具有新语法元素的 Python 3 代码将无法在 Python 2 上运行，反之亦然。被移除的元素将使 Python 2 代码与 Python 3 明显不兼容。具有这些问题的运行代码将立即导致解释器失败，引发`SyntaxError`异常。以下是一个破损脚本的示例，其中恰好有两个语句，由于语法错误，都不会被执行：

```py
print("hello world")
print "goodbye python2"
```

在 Python 3 上运行时的实际结果如下：

```py
$ python3 script.py
 **File "script.py", line 2
 **print "goodbye python2"
 **^
SyntaxError: Missing parentheses in call to 'print'

```

这样的差异列表有点长，而且，任何新的 Python 3.x 版本可能会不时地添加新的语法元素，这些元素会在早期的 Python 版本（甚至在同一个 3.x 分支上）上引发错误。其中最重要的部分在第二章和第三章中都有所涵盖，因此这里没有必要列出所有这些内容。

从 Python 2.7 中删除或更改的事项列表较短，因此以下是最重要的事项：

+   `print`不再是语句，而是一个函数，因此括号现在是必需的。

+   捕获异常从`except exc, var`变为`except exc as var`。

+   `<>`比较运算符已被移除，改用`!=`。

+   `from module import *`（[`docs.python.org/3.0/reference/simple_stmts.html#import`](https://docs.python.org/3.0/reference/simple_stmts.html#import)）现在只允许在模块级别上使用，不再在函数内部使用。

+   `from .[module] import name`现在是相对导入的唯一接受的语法。所有不以点字符开头的导入都被解释为绝对导入。

+   `sort()`函数和列表的`sorted()`方法不再接受`cmp`参数。应该使用`key`参数代替。

+   整数的除法表达式，如 1/2 会返回浮点数。截断行为是通过`//`运算符实现的，比如`1//2`。好处是这也可以用于浮点数，所以`5.0//2.0 == 2.0`。

### 标准库中的变化

标准库中的重大变化是在语法变化之后最容易捕捉到的。每个后续版本的 Python 都会添加、弃用、改进或完全删除标准库模块。这样的过程在 Python 的旧版本（1.x 和 2.x）中也是常见的，因此在 Python 3 中并不令人震惊。在大多数情况下，根据被移除或重新组织的模块（比如`urlparse`被移动到`urllib.parse`），它将在导入时立即引发异常。这使得这类问题很容易被捕捉到。无论如何，为了确保所有这类问题都得到覆盖，完整的测试代码覆盖是必不可少的。在某些情况下（例如，当使用延迟加载模块时），通常在导入时注意到的问题在一些模块作为函数调用的代码中使用之前不会出现。这就是为什么在测试套件中确保每行代码实际执行非常重要。

### 提示

**延迟加载模块**

延迟加载模块是在导入时不加载的模块。在 Python 中，`import`语句可以包含在函数内部，因此导入将在函数调用时发生，而不是在导入时发生。在某些情况下，这种模块的加载可能是一个合理的选择，但在大多数情况下，这是对设计不佳的模块结构的一种变通方法（例如，避免循环导入），并且通常应该避免。毫无疑问，没有理由去延迟加载标准库模块。

### 数据类型和集合的变化

Python 表示数据类型和集合的变化需要开发人员在尝试保持兼容性或简单地将现有代码移植到 Python 3 时付出最大的努力。虽然不兼容的语法或标准库变化很容易被注意到并且最容易修复，但集合和类型的变化要么不明显，要么需要大量重复的工作。这样的变化列表很长，再次，官方文档是最好的参考。

然而，这一部分必须涵盖 Python 3 中字符串文字处理方式的变化，因为尽管这是一个非常好的变化，现在使事情更加明确，但它似乎是 Python 3 中最具争议和讨论的变化。

所有字符串文字现在都是 Unicode，`bytes`文字需要`b`或`B`前缀。对于 Python 3.0 和 3.1，使用`u`前缀（如`u"foo"`）已被删除，并将引发语法错误。放弃该前缀是所有争议的主要原因。这使得在不同分支的 Python 版本中创建兼容的代码变得非常困难——版本 2.x 依赖于该前缀以创建 Unicode 文字。该前缀在 Python 3.3 中被重新引入以简化集成过程，尽管没有任何语法意义。

## 用于维护跨版本兼容性的流行工具和技术

在 Python 版本之间保持兼容性是一项挑战。这可能会增加很多额外的工作，具体取决于项目的规模，但绝对是可行的，也是值得做的。对于旨在在许多环境中重复使用的软件包，这是绝对必须的。没有明确定义和测试过的兼容性边界的开源软件包很不可能变得流行，但也是，从不离开公司网络的封闭的第三方代码可以从在不同环境中进行测试中获益。

值得注意的是，虽然本部分主要关注 Python 的各个版本之间的兼容性，但这些方法也适用于与外部依赖项（如不同的软件包版本、二进制库、系统或外部服务）保持兼容性。

整个过程可以分为三个主要领域，按重要性排序：

+   定义和记录目标兼容性边界以及如何管理它们

+   在每个环境和每个声明为兼容的依赖版本中进行测试

+   实施实际的兼容性代码

定义什么被认为是兼容的是整个过程中最重要的部分，因为它为代码的用户（开发人员）提供了对其工作方式和未来可能发生变化的期望和假设的能力。我们的代码可以作为不同项目中的依赖项使用，这些项目可能也致力于管理兼容性，因此理解其行为方式的能力至关重要。

虽然本书试图总是提供几种选择，而不是对特定选项给出绝对建议，但这是少数例外之一。到目前为止，定义未来兼容性可能如何改变的最佳方法是使用*语义化版本*（[`semver.org/`](http://semver.org/)），或简称 semver。它描述了一种广泛接受的标准，通过版本说明符仅由三个数字组成，标记了代码变化的范围。它还提供了一些建议，关于如何处理弃用策略。以下是其摘要的一部分：

给定版本号`MAJOR.MINOR.PATCH`，递增：

+   当您进行不兼容的 API 更改时，使用`MAJOR`版本

+   在向后兼容的方式中添加功能时的`MINOR`版本

+   当您进行向后兼容的错误修复时，使用`PATCH`版本

预发布和构建元数据的附加标签可作为`MAJOR.MINOR.PATCH`格式的扩展。

当涉及测试时，令人沮丧的事实是，为了确保代码与每个声明的依赖版本和每个环境（这里是 Python 版本）兼容，必须在这些组合的每个组合中进行测试。当项目具有大量依赖项时，这当然可能是不可能的，因为随着每个新版本的依赖项，组合的数量会迅速增长。因此，通常需要做出一些权衡，以便运行完整的兼容性测试不会花费很长时间。在第十章中介绍了一些帮助测试所谓矩阵的工具，*测试驱动开发*，讨论了测试。

### 注意

使用遵循 semver 的项目的好处通常是只需要测试主要版本，因为次要和补丁版本保证不包含向后不兼容的更改。只有在这样的项目可以信任不违反这样的合同时才成立。不幸的是，每个人都会犯错误，并且许多项目甚至在补丁版本上也会发生向后不兼容的更改。然而，由于 semver 声明了次要和补丁版本更改的严格兼容性，违反它被认为是一个错误，因此可以在补丁版本中修复。

兼容性层的实现是最后的，也是最不重要的，如果该兼容性的边界被明确定义并经过严格测试。但是，仍然有一些工具和技术，每个对这样一个主题感兴趣的程序员都应该知道。

最基本的是 Python 的`__future__`模块。它将一些新版本 Python 的功能移回到旧版本，并采用 import 语句的形式：

```py
from __future__ import <feature>
```

`future`语句提供的功能是与语法相关的元素，不能通过其他方式轻松处理。此语句仅影响其使用的模块。以下是 Python 2.7 交互会话的示例，它从 Python 3.0 中引入了 Unicode 文字：

```py
Python 2.7.10 (default, May 23 2015, 09:40:32) [MSC v.1500 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> type("foo")  # old literals
<type 'str'>
>>> from __future__ import unicode_literals
>>> type("foo")  # now is unicode
<type 'unicode'>

```

以下是所有可用的`__future__`语句选项的列表，关心 2/3 兼容性的开发人员应该知道：

+   `division`：这添加了 Python 3 的除法运算符（PEP 238）

+   `absolute_import`：这使得以点字符开头的`import`语句的每种形式都被解释为绝对导入（PEP 328）

+   `print_function`：这将把`print`语句改为函数调用，因此`print`周围的括号变得必需（PEP 3112）

+   `unicode_literals`：这使得每个字符串文字都被解释为 Unicode 文字（PEP 3112）

`__future__`语句选项的列表非常短，只涵盖了一些语法特性。像元类语法（这是一个高级特性，涵盖在第三章中，*语法最佳实践-类级别以上*）这样的其他更改，要维护起来就困难得多。可靠地处理多个标准库重组也不能通过`future`语句解决。幸运的是，有一些工具旨在提供一致的可用兼容性层。最常见的是 Six（[`pypi.python.org/pypi/six/`](https://pypi.python.org/pypi/six/)），它提供了整个通用的 2/3 兼容性样板作为单个模块。另一个有前途但稍微不那么受欢迎的工具是 future 模块（[`python-future.org/`](http://python-future.org/)）。

在某些情况下，开发人员可能不希望在一些小包中包含额外的依赖项。一个常见的做法是额外的模块，它收集所有兼容性代码，通常命名为`compat.py`。以下是从`python-gmaps`项目（[`github.com/swistakm/python-gmaps`](https://github.com/swistakm/python-gmaps)）中获取的这样一个`compat`模块的示例：

```py
# -*- coding: utf-8 -*-
import sys

if sys.version_info < (3, 0, 0):
    import urlparse  # noqa

    def is_string(s):
        return isinstance(s, basestring)

else:
    from urllib import parse as urlparse  # noqa

    def is_string(s):
        return isinstance(s, str)
```

即使在依赖于 Six 进行 2/3 兼容性的项目中，这样的`compat.py`模块也很受欢迎，因为这是一种非常方便的方式来存储处理与用作依赖项的不同版本的包的兼容性的代码。

### 提示

**下载示例代码**

您可以从[`www.packtpub.com`](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便文件直接通过电子邮件发送给您。

您可以按照以下步骤下载代码文件：

+   使用您的电子邮件地址和密码登录或注册我们的网站。

+   将鼠标指针悬停在顶部的**支持**选项卡上。

+   单击**代码下载和勘误**。

+   在**搜索**框中输入书名。

+   选择您要下载代码文件的书籍。

+   从下拉菜单中选择您购买此书的位置。

+   点击**代码下载**。

下载文件后，请确保使用最新版本的解压或提取文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Expert-Python-Programming_Second-Edition`](https://github.com/PacktPublishing/Expert-Python-Programming_Second-Edition)。我们还有其他丰富的书籍和视频代码包可供使用，网址为[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。去看看吧！

# 不仅仅是 CPython

主要的 Python 实现是用 C 语言编写的，称为**CPython**。当语言发展时，C 实现会相应地进行更改。除了 C，Python 还有其他几种实现，它们试图跟上主流。它们中的大多数都比 CPython 落后了几个里程碑，但为在特定环境中使用和推广该语言提供了绝佳的机会。

## 我为什么要在意呢？

有很多可用的替代 Python 实现。关于这个主题的 Python Wiki 页面（[`wiki.python.org/moin/PythonImplementations`](https://wiki.python.org/moin/PythonImplementations)）列出了 20 多种不同的语言变体、方言或 Python 解释器的实现，这些实现都是用其他语言而不是 C 构建的。其中一些只实现了核心语言语法、特性和内置扩展的子集，但至少有一些几乎完全兼容 CPython。最重要的是要知道，虽然其中一些只是玩具项目或实验，但大多数是为了解决一些真正的问题而创建的——这些问题要么无法用 CPython 解决，要么需要开发人员付出太多的努力。这些问题的例子有：

+   在嵌入式系统上运行 Python 代码

+   与 Java 或.NET 等运行时框架编写的代码集成，或者与不同语言编写的代码集成

+   在 Web 浏览器中运行 Python 代码

本节提供了一个主观上最受欢迎和最新的选择的简短描述，这些选择目前适用于 Python 程序员。

## Stackless Python

Stackless Python 自称是 Python 的增强版本。Stackless 之所以被命名为 Stackless，是因为它避免依赖 C 调用堆栈来进行自己的堆栈。实际上，它是修改过的 CPython 代码，还添加了一些当时核心 Python 实现中缺失的新功能。其中最重要的是由解释器管理的微线程，它是普通线程的一种廉价而轻量级的替代，普通线程必须依赖于系统内核上下文切换和任务调度。

最新可用的版本是 2.7.9 和 3.3.5，分别实现了 Python 2.7 和 3.3 版本。Stackless 提供的所有附加功能都通过内置的`stackless`模块作为该发行版中的一个框架暴露出来。

Stackless 并不是 Python 的最受欢迎的替代实现，但它值得知道，因为它引入的想法对语言社区产生了很大的影响。核心切换功能是从 Stackless 中提取出来的，并发布为一个名为`greenlet`的独立包，现在已经成为许多有用的库和框架的基础。此外，它的大多数功能已经在 PyPy 中重新实现，PyPy 是另一个稍后将介绍的 Python 实现。参考[`stackless.readthedocs.org/`](http://stackless.readthedocs.org/)。

## Jython

Jython 是语言的 Java 实现。它将代码编译成 Java 字节码，并允许开发人员在其 Python 模块中无缝使用 Java 类。Jython 允许人们在复杂的应用系统中使用 Python 作为顶级脚本语言，例如 J2EE。它还将 Java 应用程序引入 Python 世界。使 Apache Jackrabbit（这是一个基于 JCR 的文档存储库 API；请参见[`jackrabbit.apache.org`](http://jackrabbit.apache.org)）在 Python 程序中可用是 Jython 允许的一个很好的例子。

Jython 的最新可用版本是 Jython 2.7，对应于语言的 2.7 版本。它被宣传为几乎实现了所有核心 Python 标准库，并使用相同的回归测试套件。Jython 3.x 的版本正在开发中。

与 CPython 实现相比，Jython 的主要区别是：

+   真正的 Java 垃圾回收，而不是引用计数

+   缺乏全局解释器锁（GIL）允许更好地利用多核在多线程应用程序中

该语言实现的主要弱点是不支持 CPython 扩展 API，因此不支持用 C 编写的 Python 扩展将无法在 Jython 中运行。这可能会在未来发生变化，因为计划在 Jython 3.x 中支持 CPython 扩展 API。

一些 Python Web 框架，如 Pylons，被认为正在推动 Jython 的发展，使其在 Java 世界中可用。参见[`www.jython.org`](http://www.jython.org)。

## IronPython

IronPython 将 Python 引入了.NET Framework。该项目得到了微软的支持，IronPython 的主要开发人员在这里工作。这对于推广一种语言来说是非常重要的实现。除了 Java，.NET 社区是最大的开发者社区之一。值得注意的是，微软提供了一套免费的开发工具，可以将 Visual Studio 变成一个功能齐全的 Python IDE。这被分发为名为**PVTS**（**Visual Studio 的 Python 工具**）的 Visual Studio 插件，并且作为开源代码在 GitHub 上可用（[`microsoft.github.io/PTVS`](http://microsoft.github.io/PTVS)）。

最新的稳定版本是 2.7.5，与 Python 2.7 兼容。与 Jython 类似，Python 3.x 实现周围也有一些开发，但目前还没有稳定版本。尽管.NET 主要在 Microsoft Windows 上运行，但也可以在 Mac OS X 和 Linux 上运行 IronPython。这要归功于 Mono，一个跨平台的开源.NET 实现。

IronPython 相对于 CPython 的主要区别或优势如下：

+   与 Jython 类似，缺乏全局解释器锁（GIL）允许更好地利用多核在多线程应用程序中

+   用 C#和其他.NET 语言编写的代码可以轻松集成到 IronPython 中，反之亦然

+   可以在所有主要的 Web 浏览器中通过 Silverlight 运行

IronPython 的弱点与 Jython 非常相似，因为它不支持 CPython 扩展 API。这对于希望使用诸如 numpy 之类的大部分基于 C 扩展的软件包的开发人员来说非常重要。有一个名为 ironclad 的项目（参见[`github.com/IronLanguages/ironclad`](https://github.com/IronLanguages/ironclad)），旨在允许使用这些扩展与 IronPython 无缝集成，尽管其最后已知支持的版本是 2.6，开发似乎已经停止。参见[`ironpython.net/`](http://ironpython.net/)。

## PyPy

PyPy 可能是最令人兴奋的实现，因为其目标是将 Python 重写为 Python。在 PyPy 中，Python 解释器本身就是用 Python 编写的。我们有一个 C 代码层来执行 Python 的 CPython 实现的基本工作。然而，在 PyPy 实现中，这个 C 代码层被重写为纯 Python。

这意味着您可以在执行时更改解释器的行为，并实现在 CPython 中无法轻松完成的代码模式。

PyPy 目前旨在与 Python 2.7 完全兼容，而 PyPy3 与 Python 3.2.5 版本兼容。

过去，PyPy 主要因理论原因而引人关注，它吸引了那些喜欢深入了解语言细节的人。它并不常用于生产，但这在多年来已经发生了改变。如今，许多基准测试显示，令人惊讶的是，PyPy 通常比 CPython 实现要快得多。该项目有自己的基准测试网站，跟踪每个版本的性能，使用数十种不同的基准测试进行测量（参见[`speed.pypy.org/`](http://speed.pypy.org/)）。这清楚地表明，启用 JIT 的 PyPy 至少比 CPython 快几倍。这和 PyPy 的其他特性使越来越多的开发人员决定在生产环境中切换到 PyPy。

PyPy 与 CPython 实现相比的主要区别是：

+   使用垃圾收集而不是引用计数

+   集成跟踪 JIT 编译器可以显著提高性能

+   从 Stackless Python 借用的应用级 Stackless 功能

与几乎所有其他替代 Python 实现一样，PyPy 缺乏对 CPython 扩展 API 的全面官方支持。尽管如此，它至少通过其 CPyExt 子系统提供了对 C 扩展的某种支持，尽管文档贫乏且功能尚不完整。此外，社区正在努力将 NumPy 移植到 PyPy，因为这是最受欢迎的功能。参见[`pypy.org`](http://pypy.org)。

# Python 开发的现代方法

选择的编程语言的深入理解是作为专家最重要的事情。这对于任何技术来说都是真实的。然而，如果不了解特定语言社区内的常用工具和实践，要开发出优秀的软件是非常困难的。Python 没有任何一个特性是其他语言中找不到的。因此，在语法、表现力或性能的直接比较中，总会有一个或多个领域更好的解决方案。但 Python 真正脱颖而出的领域是围绕该语言构建的整个生态系统。多年来，Python 社区一直在完善标准实践和库，帮助更快地创建更可靠的软件。

提到的生态系统中最明显和重要的部分是大量解决各种问题的免费开源软件包。编写新软件总是一个昂贵且耗时的过程。能够重用现有代码而不是“重复造轮子”大大减少了开发的时间和成本。对一些公司来说，这是他们的项目经济可行的唯一原因。

因此，Python 开发人员花了很多精力来创建工具和标准，以处理他人创建的开源软件包。从虚拟隔离环境、改进的交互式 shell 和调试器，到帮助发现、搜索和分析**PyPI**（**Python 软件包索引**）上可用的大量软件包的程序。

# Python 环境的应用级隔离

如今，许多操作系统都将 Python 作为标准组件。大多数 Linux 发行版和基于 Unix 的系统，如 FreeBSD、NetBSD、OpenBSD 或 OS X，都默认安装了 Python，或者可以通过系统软件包存储库获得。其中许多甚至将其用作一些核心组件的一部分——Python 驱动 Ubuntu（Ubiquity）、Red Hat Linux（Anaconda）和 Fedora（再次是 Anaconda）的安装程序。

由于这个事实，PyPI 上的许多软件包也可以作为系统软件包管理工具（如`apt-get`（Debian，Ubuntu），`rpm`（Red Hat Linux）或`emerge`（Gentoo））管理的本地软件包。尽管应该记住，可用库的列表非常有限，而且与 PyPI 相比，它们大多已经过时。这就是为什么`pip`应该始终被用来获取最新版本的新软件包，作为**PyPA**（**Python Packaging Authority**）的建议。尽管它是 CPython 2.7.9 和 3.4 版本的独立软件包，但它默认随每个新版本捆绑发布。安装新软件包就像这样简单：

```py
pip install <package-name>

```

除其他功能外，`pip`允许强制使用特定版本的软件包（使用`pip install package-name==version`语法）并升级到最新可用版本（使用`--upgrade`开关）。本书中介绍的大多数命令行工具的完整使用说明可以通过简单地运行带有`-h`或`--help`开关的命令来轻松获得，但这里有一个示例会话，演示了最常用的选项：

```py
$ pip show pip
---
Metadata-Version: 2.0
Name: pip
Version: 7.1.2
Summary: The PyPA recommended tool for installing Python packages.
Home-page: https://pip.pypa.io/
Author: The pip developers
Author-email: python-virtualenv@groups.google.com
License: MIT
Location: /usr/lib/python2.7/site-packages
Requires:

$ pip install 'pip<7.0.0'
Collecting pip<7.0.0
 **Downloading pip-6.1.1-py2.py3-none-any.whl (1.1MB)
 **100% |████████████████████████████████| 1.1MB 242kB/s
Installing collected packages: pip
 **Found existing installation: pip 7.1.2
 **Uninstalling pip-7.1.2:
 **Successfully uninstalled pip-7.1.2
Successfully installed pip-6.1.1
You are using pip version 6.1.1, however version 7.1.2 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.

$ pip install --upgrade pip
You are using pip version 6.1.1, however version 7.1.2 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.
Collecting pip
 **Using cached pip-7.1.2-py2.py3-none-any.whl
Installing collected packages: pip
 **Found existing installation: pip 6.1.1
 **Uninstalling pip-6.1.1:
 **Successfully uninstalled pip-6.1.1
Successfully installed pip-7.1.2

```

在某些情况下，`pip`可能不是默认可用的。从 Python 3.4 版本开始（也是 Python 2.7.9），它始终可以使用`ensurepip`模块进行引导：

```py
$ python -m ensurepip
Ignoring indexes: https://pypi.python.org/simple
Requirement already satisfied (use --upgrade to upgrade): setuptools in /usr/lib/python2.7/site-packages
Collecting pip
Installing collected packages: pip
Successfully installed pip-6.1.1

```

有关如何为旧版本的 Python 安装 pip 的最新信息，请访问项目的文档页面[`pip.pypa.io/en/stable/installing/`](https://pip.pypa.io/en/stable/installing/)。

## 为什么要隔离？

`pip`可用于安装系统范围的软件包。在基于 Unix 和 Linux 的系统上，这将需要超级用户权限，因此实际调用将是：

```py
sudo pip install <package-name>

```

请注意，这在 Windows 上不是必需的，因为它默认不提供 Python 解释器，通常由用户手动安装 Python 而不需要超级用户权限。

无论如何，不建议直接从 PyPI 全局安装系统范围的软件包，应该避免这样做。这似乎与之前的使用`pip`是 PyPA 建议的说法相矛盾，但这其中有一些严重的原因。如前所述，Python 往往是许多通过操作系统软件包存储库可用的软件包的重要组成部分，并且可能支持许多重要的服务。系统分发维护者在选择正确的软件包版本以匹配各种软件包依赖关系方面付出了很多努力。经常情况下，从系统软件包存储库中可用的 Python 软件包包含自定义补丁或者保持过时，只是为了确保与其他一些系统组件的兼容性。使用`pip`强制更新此类软件包到破坏某些向后兼容性的版本可能会破坏一些关键的系统服务。

仅在本地计算机上进行开发目的的这样做也不是一个好的借口。这样鲁莽地使用`pip`几乎总是在自找麻烦，并最终会导致非常难以调试的问题。这并不意味着全局安装来自 PyPI 的软件包是严格禁止的，但它应该始终是有意识地并且了解相关风险的情况下进行的。

幸运的是，有一个简单的解决方案——环境隔离。有各种工具可以在不同系统抽象级别上隔离 Python 运行时环境。主要思想是将项目依赖项与不同项目和/或系统服务所需的软件包隔离开来。这种方法的好处是：

+   它解决了“项目 X 依赖于 1.x 版本，但项目 Y 需要 4.x”困境。开发人员可以在具有不同依赖关系的多个项目上工作，甚至可能发生冲突，而不会影响彼此。

+   项目不再受限于系统分发存储库中提供的软件包版本。

+   由于新的包版本只在这样的环境中可用，所以不会有破坏依赖于特定包版本的其他系统服务的风险。

+   项目依赖的包列表可以很容易地被“冻结”，因此很容易重现它们。

最简单和最轻量级的隔离方法是使用应用级虚拟环境。它们只专注于隔离 Python 解释器和其中可用的包。它们非常容易设置，通常足以确保在开发小型项目和包时进行适当的隔离。

不幸的是，在某些情况下，这可能不足以确保足够的一致性和可重现性。对于这种情况，系统级别的隔离是工作流程的一个很好的补充，本章后面将解释一些可用的解决方案。

## 流行的解决方案

有几种隔离 Python 运行时的方法。最简单和最明显的方法，尽管最难维护，是手动更改`PATH`和`PYTHONPATH`环境变量和/或将 Python 二进制文件移动到不同的位置，以影响它发现可用包的方式，并将其更改为我们想要存储项目依赖项的自定义位置。幸运的是，有几种可用的工具可以帮助维护虚拟环境以及安装包在系统中的存储方式。主要有：`virtualenv`、`venv`和`buildout`。它们在底层的操作实际上与我们手动操作的相同。实际的策略取决于具体的工具实现，但通常它们更方便使用并且可以提供额外的好处。

### virtualenv

Virtualenv 是这个列表中迄今为止最受欢迎的工具。它的名字简单地代表虚拟环境。它不是标准 Python 发行版的一部分，因此需要使用`pip`来获取。它是值得系统范围内安装的包之一（在 Linux 和基于 Unix 的系统上使用`sudo`）。

一旦安装完成，可以使用以下命令创建一个新的虚拟环境：

```py
virtualenv ENV

```

在这里，`ENV`应该被新环境的期望名称替换。这将在当前工作目录路径中创建一个新的`ENV`目录。它将包含几个新的目录：

+   `bin/`：这是存储新 Python 可执行文件和其他包提供的脚本/可执行文件的地方。

+   `lib/`和`include/`：这些目录包含了虚拟环境中新 Python 的支持库文件。新的包将安装在`ENV/lib/pythonX.Y/site-packages/`中。

一旦创建了新的环境，就需要在当前 shell 会话中使用 Unix 的 source 命令激活它：

```py
source ENV/bin/activate

```

这会通过影响其环境变量改变当前 shell 会话的状态。为了让用户意识到他已经激活了虚拟环境，它会通过在其开头添加`(ENV)`字符串来改变 shell 提示。以下是一个创建新环境并激活它的示例会话：

```py
$ virtualenv example
New python executable in example/bin/python
Installing setuptools, pip, wheel...done.
$ source example/bin/activate
(example)$ deactivate
$** 

```

关于`virtualenv`的重要事情是，它完全依赖于存储在文件系统上的状态。它不提供任何额外的能力来跟踪应该安装在其中的包。这些虚拟环境不可移植，不应该移动到另一个系统/机器上。这意味着需要为每个新的应用部署从头开始创建新的虚拟环境。因此，`virtualenv`用户使用的一个良好的实践是将所有项目依赖项存储在`requirements.txt`文件中（这是命名约定），如下面的代码所示：

```py
# lines followed by hash (#) are treated as a comments

# strict version names are best for reproducibility
eventlet==0.17.4
graceful==0.1.1

# for projects that are well tested with different
# dependency versions the relative version specifiers 
# are acceptable too
falcon>=0.3.0,<0.5.0

# packages without versions should be avoided unless
# latest release is always required/desired
pytz
```

有了这样的文件，所有依赖项都可以很容易地使用`pip`进行安装，因为它接受 requirements 文件作为其输出。

```py
pip install -r requirements.txt

```

需要记住的是，要求文件并不总是理想的解决方案，因为它并没有定义确切的依赖项列表，只有要安装的依赖项。因此，整个项目在开发环境中可以正常工作，但如果要求文件过时并且不反映环境的实际状态，它将无法在其他环境中启动。当然，有`pip freeze`命令可以打印当前环境中的所有软件包，但不应该盲目使用它——它会输出所有内容，甚至是仅用于测试而不在项目中使用的软件包。书中提到的另一个工具`buildout`解决了这个问题，因此对于一些开发团队来说，它可能是更好的选择。

### 注意

对于 Windows 用户，在 Windows 下，`virtualenv`使用不同的命名方式来命名其目录的内部结构。您需要使用`Scripts/`，`Libs/`和`Include/`，而不是`bin/`，`lib/`，`include/`，以更好地匹配该操作系统上的开发约定。激活/停用环境的命令也不同；您需要使用`ENV/Scripts/activate.bat`和`ENV/Scripts/deactivate.bat`，而不是在`activate`和`deactivate`脚本上使用`source`。

### venv

虚拟环境很快在社区内得到了很好的建立，并成为了一个受欢迎的工具。从 Python 3.3 开始，创建虚拟环境得到了标准库的支持。使用方式几乎与 Virtualenv 相同，尽管命令行选项的命名约定有很大不同。新的`venv`模块提供了一个`pyvenv`脚本来创建一个新的虚拟环境。

```py
pyvenv ENV

```

这里，`ENV`应该被新环境的期望名称所替换。此外，现在可以直接从 Python 代码中创建新环境，因为所有功能都是从内置的`venv`模块中公开的。其他用法和实现细节，如环境目录的结构和激活/停用脚本，大部分与 Virtualenv 相同，因此迁移到这个解决方案应该是简单而无痛的。

对于使用较新版本 Python 的开发人员，建议使用`venv`而不是 Virtualenv。对于 Python 3.3，切换到`venv`可能需要更多的努力，因为在这个版本中，它不会默认在新环境中安装`setuptools`和`pip`，因此用户需要手动安装它们。幸运的是，这在 Python 3.4 中已经改变，而且由于`venv`的可定制性，可以覆盖其行为。详细信息在 Python 文档中有解释（参见[`docs.python.org/3.5/library/venv.html`](https://docs.python.org/3.5/library/venv.html)），但一些用户可能会发现它太棘手，会选择在特定版本的 Python 中继续使用 Virtualenv。

### buildout

Buildout 是一个强大的用于引导和部署用 Python 编写的应用程序的工具。书中还将解释一些其高级功能。很长一段时间以来，它也被用作创建隔离的 Python 环境的工具。因为 Buildout 需要一个声明性的配置，必须在依赖关系发生变化时进行更改，而不是依赖于环境状态，因此这些环境更容易复制和管理。

很不幸，这已经改变了。自 2.0.0 版本以来，`buildout`软件包不再尝试提供与系统 Python 安装的任何级别的隔离。隔离处理留给其他工具，如 Virtualenv，因此仍然可以拥有隔离的 Buildouts，但事情变得有点复杂。必须在隔离的环境中初始化 Buildout 才能真正实现隔离。

与 Buildout 的旧版本相比，这有一个主要缺点，因为它依赖于其他解决方案进行隔离。编写此代码的开发人员不再能确定依赖关系描述是否完整，因为一些软件包可以通过绕过声明性配置进行安装。当然，这个问题可以通过适当的测试和发布程序来解决，但它给整个工作流程增加了一些复杂性。

总之，Buildout 不再是提供环境隔离的解决方案，但其声明性配置可以提高虚拟环境的可维护性和可重现性。

## 选择哪一个？

没有一种最佳解决方案适用于所有用例。在一个组织中适用的东西可能不适合其他团队的工作流程。此外，每个应用程序都有不同的需求。小型项目可以轻松依赖于单独的`virtualenv`或`venv`，但更大的项目可能需要`buildout`的额外帮助来执行更复杂的组装。

之前没有详细描述的是，Buildout 的旧版本（buildout<2.0.0）允许在与 Virtualenv 提供的类似结果的隔离环境中组装项目。不幸的是，该项目的 1.x 分支不再维护，因此不鼓励将其用于此目的。

我建议尽可能使用`venv`模块而不是 Virtualenv。因此，这应该是针对 Python 版本 3.4 及更高版本的项目的默认选择。在 Python 3.3 中使用`venv`可能有点不方便，因为缺乏对`setuptools`和`pip`的内置支持。对于针对更广泛的 Python 运行时（包括替代解释器和 2.x 分支）的项目，似乎 Virtualenv 是最佳选择。

# 系统级环境隔离

在大多数情况下，软件实现可以快速迭代，因为开发人员重用许多现有组件。不要重复自己——这是许多程序员的流行规则和座右铭。使用其他软件包和模块将它们包含在代码库中只是这种文化的一部分。可以被视为“重复使用组件”的还有二进制库、数据库、系统服务、第三方 API 等。甚至整个操作系统也应该被视为被重复使用。

基于 Web 的应用程序的后端服务是这类应用程序可以有多复杂的一个很好的例子。最简单的软件堆栈通常由几个层组成（从最低层开始）：

+   数据库或其他类型的存储设备

+   Python 实现的应用程序代码

+   像 Apache 或 NGINX 这样的 HTTP 服务器

当然，这样的堆栈可能会更简单，但这是非常不可能的。事实上，大型应用程序通常非常复杂，很难区分单个层。大型应用程序可以使用许多不同的数据库，分为多个独立的进程，并使用许多其他系统服务进行缓存、排队、日志记录、服务发现等。遗憾的是，复杂性没有限制，代码似乎只是遵循热力学第二定律。

真正重要的是，并非所有软件堆栈元素都可以在 Python 运行时环境的级别上进行隔离。无论是 NGINX 这样的 HTTP 服务器还是 PostgreSQL 这样的 RDBMS，它们通常在不同系统上有不同版本。确保开发团队中的每个人使用每个组件的相同版本是非常困难的，没有适当的工具。理论上，一个团队中所有开发人员在一个项目上工作时能够在他们的开发环境中获得相同版本的服务。但是，如果他们不使用与生产环境相同的操作系统，所有这些努力都是徒劳的。而且，强迫程序员在他所钟爱的系统之外工作是不可能的。

问题在于可移植性仍然是一个巨大的挑战。并非所有服务在生产环境中都能像在开发者的机器上那样完全相同地工作，这种情况很可能不会改变。即使是 Python，尽管已经付出了很多工作来使其跨平台，但在不同的系统上可能会有不同的行为。通常，这些情况都有很好的文档记录，而且只会发生在直接依赖系统调用的地方，但依赖程序员记住一长串兼容性怪癖的能力是相当容易出错的策略。

解决这个问题的一个流行的解决方案是通过将整个系统隔离为应用程序环境。这通常是通过利用不同类型的系统虚拟化工具来实现的。虚拟化当然会降低性能，但对于具有硬件虚拟化支持的现代计算机来说，性能损失通常是可以忽略不计的。另一方面，可能获得的潜在收益列表非常长：

+   开发环境可以完全匹配生产中使用的系统版本和服务，有助于解决兼容性问题

+   系统配置工具（如 Puppet、Chef 或 Ansible）的定义（如果使用）可以被重用于配置开发环境

+   如果创建这样的环境是自动化的，新加入的团队成员可以轻松地加入项目

+   开发人员可以直接使用低级别的系统功能，这些功能可能在他们用于工作的操作系统上不可用，例如，在 Windows 中不可用的**FUSE**（用户空间文件系统）

## 使用 Vagrant 创建虚拟开发环境

Vagrant 目前似乎是提供创建和管理开发环境的最流行的工具。它适用于 Windows、Mac OS 和一些流行的 Linux 发行版（参见[`www.vagrantup.com`](https://www.vagrantup.com)）。它没有任何额外的依赖。Vagrant 以虚拟机或容器的形式创建新的开发环境。具体的实现取决于虚拟化提供者的选择。VirtualBox 是默认提供者，并且它已经与 Vagrant 安装程序捆绑在一起，但也有其他提供者可用。最显著的选择是 VMware、Docker、LXC（Linux 容器）和 Hyper-V。

Vagrant 中提供的最重要的配置是一个名为`Vagrantfile`的单个文件。它应该独立于每个项目。它提供的最重要的内容如下：

+   虚拟化提供者的选择

+   用作虚拟机镜像的 Box

+   配置方法的选择

+   虚拟机和虚拟机主机之间的共享存储

+   需要在虚拟机和其主机之间转发的端口

`Vagrantfile`的语法语言是 Ruby。示例配置文件提供了一个很好的模板来启动项目，并且有很好的文档，因此不需要了解这种语言。可以使用一个命令创建模板配置：

```py
vagrant init

```

这将在当前工作目录中创建一个名为`Vagrantfile`的新文件。通常最好将此文件存储在相关项目源的根目录。这个文件已经是一个有效的配置，将使用默认提供者和基础盒子镜像创建一个新的虚拟机。默认情况下不启用任何配置。添加了`Vagrantfile`后，可以使用以下命令启动新的虚拟机：

```py
vagrant up

```

初始启动可能需要几分钟，因为实际的盒子必须从网络上下载。每次启动已经存在的虚拟机时，还会有一些初始化过程，这可能需要一些时间，具体取决于所使用的提供者、盒子和系统性能。通常，这只需要几秒钟。一旦新的 Vagrant 环境启动并运行，开发人员可以使用以下简写连接到 SSH：

```py
vagrant ssh

```

这可以在`Vagrantfile`位置下的项目源树中的任何位置完成。为了开发者的方便起见，我们将在上面的目录中查找配置文件，并将其与相关的 VM 实例进行匹配。然后，它建立安全外壳连接，因此开发环境可以像任何普通的远程机器一样进行交互。唯一的区别是整个项目源树（根定义为`Vagrantfile`位置）在 VM 的文件系统下的`/vagrant/`下是可用的。

## 容器化与虚拟化

容器是完整机器虚拟化的替代方案。这是一种轻量级的虚拟化方法，其中内核和操作系统允许运行多个隔离的用户空间实例。操作系统在容器和主机之间共享，因此在理论上需要的开销比完整虚拟化要少。这样的容器只包含应用程序代码和其系统级依赖项，但从内部运行的进程的角度来看，它看起来像一个完全隔离的系统环境。

软件容器主要得益于 Docker 而变得流行；这是其中一种可用的实现。Docker 允许以称为`Dockerfile`的简单文本文档描述其容器。根据这些定义，可以构建和存储容器。它还支持增量更改，因此如果容器中添加了新内容，则不需要从头开始重新创建。

不同的工具，如 Docker 和 Vagrant，似乎在功能上有重叠，但它们之间的主要区别在于这些工具被构建的原因。如前所述，Vagrant 主要是作为开发工具构建的。它允许用单个命令引导整个虚拟机，但不允许简单地打包并部署或发布。另一方面，Docker 则是专门为此而构建的——准备完整的容器，可以作为一个整体包发送和部署到生产环境。如果实施得当，这可以极大地改善产品部署的过程。因此，在开发过程中使用 Docker 和类似的解决方案（例如 Rocket）只有在它也必须在生产环境中的部署过程中使用时才有意义。仅在开发过程中用于隔离目的可能会产生太多的开销，而且还有一个不一致的缺点。

# 流行的生产力工具

生产力工具是一个有点模糊的术语。一方面，几乎每个发布并在网上可用的开源代码包都是一种提高生产力的工具——它提供了现成的解决方案，使得没有人需要花时间去解决它（理想情况下）。另一方面，有人可能会说整个 Python 都是关于生产力的。这两者无疑都是真的。这种语言和围绕它的社区几乎所有的东西似乎都是为了使软件开发尽可能地高效。

这创造了一个积极的反馈循环。由于编写代码是有趣且容易的，许多程序员会利用业余时间创建使其更容易和有趣的工具。这个事实将被用作生产力工具的一个非常主观和非科学的定义的基础——一种使开发更容易和更有趣的软件。

自然而然，生产力工具主要关注开发过程的某些元素，如测试、调试和管理软件包，并不是它们帮助构建的产品的核心部分。在某些情况下，它们甚至可能根本没有在项目的代码库中被提及，尽管它们每天都在使用。

最重要的生产力工具`pip`和`venv`在本章的前面已经讨论过。其中一些工具有针对特定问题的软件包，如性能分析和测试，并在本书中有它们自己的章节。本节专门介绍了其他一些值得一提的工具，但在本书中没有专门的章节可以介绍它们。

## 自定义 Python shell - IPython，bpython，ptpython 等。

Python 程序员在交互式解释器会话中花费了大量时间。它非常适合测试小的代码片段，访问文档，甚至在运行时调试代码。默认的交互式 Python 会话非常简单，不提供诸如制表符补全或代码内省助手之类的许多功能。幸运的是，默认的 Python shell 可以很容易地扩展和自定义。

交互提示可以通过启动文件进行配置。启动时，它会查找`PYTHONSTARTUP`环境变量，并执行由该变量指向的文件中的代码。一些 Linux 发行版提供了一个默认的启动脚本，通常位于您的主目录中。它被称为`.pythonstartup`。制表符补全和命令历史记录通常会提供以增强提示，并且基于`readline`模块。（您需要`readline`库。）

如果您没有这样的文件，可以轻松创建一个。以下是一个添加了使用`<Tab>`键和历史记录的最简单启动文件的示例：

```py
# python startup file
import readline
import rlcompleter
import atexit
import os

# tab completion
readline.parse_and_bind('tab: complete')

# history file
histfile = os.path.join(os.environ['HOME'], '.pythonhistory')
try:
    readline.read_history_file(histfile)

except IOError:
    pass

atexit.register(readline.write_history_file, histfile)
del os, histfile, readline, rlcompleter
```

在您的主目录中创建此文件，并将其命名为`.pythonstartup`。然后，在环境中添加一个`PYTHONSTARTUP`变量，使用您文件的路径：

### 设置`PYTHONSTARTUP`环境变量

如果您正在运行 Linux 或 Mac OS X，最简单的方法是在您的主文件夹中创建启动脚本。然后，将其链接到设置为系统 shell 启动脚本的`PYTHONSTARTUP`环境变量。例如，Bash 和 Korn shells 使用`.profile`文件，您可以插入一行如下：

```py
export PYTHONSTARTUP=~/.pythonstartup

```

如果您正在运行 Windows，可以在系统首选项中以管理员身份设置新的环境变量，并将脚本保存在一个常用位置，而不是使用特定的用户位置。

编写`PYTHONSTARTUP`脚本可能是一个很好的练习，但独自创建一个良好的自定义 shell 是一项只有少数人能够抽出时间来完成的挑战。幸运的是，有一些自定义 Python shell 实现极大地改善了 Python 交互式会话的体验。

### IPython

IPyhton（[`ipython.scipy.org`](http://ipython.scipy.org)）提供了一个扩展的 Python 命令行。提供的功能中，最有趣的是：

+   动态对象内省

+   从提示中访问系统 shell

+   直接支持分析

+   调试设施

现在，IPython 是一个名为 Jupyter 的更大项目的一部分，它提供了可以用许多不同语言编写的具有实时代码的交互式笔记本。

### bpython

bpython（[`bpython-interpreter.org/`](http://bpython-interpreter.org/)）将自己宣传为 Python 解释器的时髦界面。以下是该项目页面上强调的一些内容：

+   内联语法高亮

+   类似 Readline 的自动完成，建议在您输入时显示

+   任何 Python 函数的预期参数列表

+   自动缩进

+   Python 3 支持

### ptpython

ptpython（[`github.com/jonathanslenders/ptpython/`](https://github.com/jonathanslenders/ptpython/)）是另一种高级 Python shell 主题的方法。在这个项目中，核心提示工具的实现可作为一个名为`prompt_toolkit`的单独包使用（来自同一作者）。这使您可以轻松创建各种美观的交互式命令行界面。

它经常与 bpython 在功能上进行比较，但主要区别在于它启用了与 IPython 和其语法的兼容模式，从而启用了额外的功能，如`%pdb`，`%cpaste`或`%profile`。

## 交互式调试器

代码调试是软件开发过程的一个重要组成部分。许多程序员可能会花费大部分时间仅使用广泛的日志记录和`print`语句作为他们的主要调试工具，但大多数专业开发人员更喜欢依赖某种调试器。

Python 已经内置了一个名为`pdb`的交互式调试器（参见[`docs.python.org/3/library/pdb.html`](https://docs.python.org/3/library/pdb.html)）。它可以从命令行上调用现有的脚本，因此如果程序异常退出，Python 将进入事后调试：

```py
python -m pdb script.py

```

事后调试虽然有用，但并不涵盖每种情况。它仅在应用程序以某种异常退出时有用。在许多情况下，有错误的代码只是表现异常，但并不会意外退出。在这种情况下，可以使用这个单行习惯用法在特定代码行上设置自定义断点：

```py
import pdb; pdb.set_trace()

```

这将导致 Python 解释器在运行时在此行开始调试会话。

`pdb`对于追踪问题非常有用，乍一看，它可能看起来非常熟悉，就像著名的 GDB（GNU 调试器）。由于 Python 是一种动态语言，`pdb`会话非常类似于普通的解释器会话。这意味着开发人员不仅限于追踪代码执行，还可以调用任何代码，甚至执行模块导入。

遗憾的是，由于其根源（`bdb`），对`pdb`的第一次体验可能会有点压倒性，因为存在着诸如`h`、`b`、`s`、`n`、`j`和`r`等神秘的短字母调试器命令。每当有疑问时，在调试器会话期间键入`help pdb`命令将提供广泛的用法和额外信息。

pdb 中的调试器会话也非常简单，不提供诸如制表符补全或代码高亮之类的附加功能。幸运的是，PyPI 上有一些包可提供这些功能，这些功能可以从前一节提到的替代 Python shell 中使用。最值得注意的例子有：

+   `ipdb`：这是一个基于`ipython`的独立包

+   `ptpdb`：这是一个基于`ptpython`的独立包

+   `bpdb`：这是与`bpython`捆绑在一起的

# 有用的资源

网络上充满了对 Python 开发人员有用的资源。最重要和明显的资源已经在前面提到过，但为了保持这个列表的一致性，这里重复一遍：

+   Python 文档

+   PyPI—Python 包索引

+   PEP 0—Python 增强提案索引

其他资源，如书籍和教程，虽然有用，但往往很快就会过时。不会过时的是由社区积极策划或定期发布的资源。其中最值得推荐的两个是：

+   Awesome-python ([`github.com/vinta/awesome-python`](https://github.com/vinta/awesome-python))，其中包括一个经过策划的流行包和框架的列表

+   Python Weekly ([`www.pythonweekly.com/`](http://www.pythonweekly.com/))是一个流行的新闻通讯，每周向订阅者提供数十个新的有趣的 Python 包和资源

这两个资源将为读者提供数月的额外阅读。

# 总结

本章从 Python 2 和 3 之间的差异开始，提出了如何处理当前情况的建议，其中大部分社区都在两个世界之间挣扎。然后，它涉及到了由于语言的两个主要版本之间的不幸分裂而出现的 Python 开发的现代方法。这些主要是环境隔离问题的不同解决方案。本章以对流行的生产工具和进一步参考的流行资源的简短总结结束。
