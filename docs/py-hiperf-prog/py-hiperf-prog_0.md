# 前言

Python 是一种以其简单性、优雅性和出色的社区支持而闻名的编程语言。得益于大量高质量第三方库的令人印象深刻，Python 被用于许多领域。

在性能关键的应用程序中，通常首选低级语言，如 C、C++和 Fortran。用这些语言编写的程序性能极好，但编写和维护起来都很困难。

Python 是一种更容易处理的编程语言，可以用来快速编写复杂的应用程序。由于其与 C 语言的紧密集成，Python 能够避免动态语言相关的性能下降。你可以使用快速的 C 扩展来编写性能关键代码，同时保留 Python 在其余应用程序中的所有便利性。

在这本书中，你将学习如何逐步使用基本和高级技术找到并加速程序中的慢速部分。

这本书的风格是实用的；每个概念都通过示例进行解释和说明。这本书还讨论了常见的错误，并教授如何避免它们。本书使用的工具相当流行且经过实战检验；你可以确信它们在未来将保持相关性和良好的支持。

这本书从基础知识开始，在此基础上构建，因此，我建议你按顺序阅读章节。

不要忘了享受乐趣！

# 这本书涵盖的内容

第一章, *基准测试和性能分析* 展示了你如何找到程序中需要优化的部分。我们将使用不同用例的工具，并解释如何分析和解释性能分析统计数据。

第二章, *使用 NumPy 进行快速数组操作* 是关于 NumPy 包的指南。NumPy 是 Python 中进行数组计算的框架。它提供了一个干净且简洁的 API，以及高效的数组操作。

第三章, *使用 Cython 进行 C 性能优化* 是关于 Cython 的教程：一种作为 Python 和 C 之间桥梁的语言。Cython 可以使用 Python 语法的超集来编写代码，并将其编译为高效的 C 扩展。

第四章, *并行处理* 是对并行编程的介绍。在本章中，你将学习并行编程与串行编程的不同之处，以及如何并行化简单问题。我们还将解释如何使用多进程、`IPython.parallel` 和 `cython.parallel` 为多核编写代码。

# 这本书你需要的东西

这本书需要安装 Python。除非另有说明，否则示例适用于 Python 2.7 和 Python 3.3。

在这本书中，我们将利用一些流行的 Python 包：

+   **NumPy**（版本 1.7.1 或更高版本）：此软件包可以从官方网站下载（[`www.scipy.org/scipylib/download.html`](http://www.scipy.org/scipylib/download.html)），并在大多数 Linux 发行版中可用

+   **Cython**（版本 0.19.1 或更高版本）：安装说明可在官方网站上找到（[`docs.cython.org/src/quickstart/install.html`](http://docs.cython.org/src/quickstart/install.html)）；请注意，您还需要一个 C 编译器，例如 GCC（GNU 编译器集合），来编译您的 C 扩展

+   **IPython**（版本 0.13.2 或更高版本）：安装说明可在官方网站上找到（[`ipython.org/install.html`](http://ipython.org/install.html)）

本书是在 Ubuntu 13.10 操作系统上编写和测试的。示例代码很可能在 Mac OS X 上运行，只需进行少量或无需修改。

我对 Windows 用户的建议是安装 Anaconda Python 发行版（[`store.continuum.io/cshop/anaconda/`](https://store.continuum.io/cshop/anaconda/)），它包含适合科学编程的完整环境。

一个方便的替代方案是使用免费的`wakari.io`服务：一个基于云的 Linux 和 Python 环境，包括所需的软件包及其工具和实用程序。无需设置。

在第一章中，“基准测试和性能分析”，我们将使用 KCachegrind ([`sourceforge.net/projects/kcachegrind/`](http://sourceforge.net/projects/kcachegrind/))，它适用于 Linux。KCachegrind 还有一个 Windows 版本——QcacheGrind，它也可以从源代码在 Mac OS X 上安装。

# 本书的目标读者

本书面向中级到高级的 Python 程序员，他们开发的是性能关键型应用程序。由于大多数示例都来自科学应用，本书非常适合希望加快其数值代码的科学家和工程师。

然而，本书的范围很广，概念可以应用于任何领域。由于本书涵盖了基本和高级主题，因此包含了不同 Python 熟练程度程序员的实用信息。

# 习惯用法

在本书中，您将找到多种文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名都按照以下格式显示：“`plot`函数包含在`matplotlib`中，可以将我们的粒子显示在笛卡尔网格上的点，而`FuncAnimation`类可以动画化我们的粒子随时间的变化。”

代码块按照以下方式设置：

```py
from matplotlib import pyplot as plt
from matplotlib import animation

def visualize(simulator):

    X = [p.x for p in simulator.particles]
    Y = [p.y for p in simulator.particles]
```

当我们希望引起您对代码块中特定部分的注意时，相关的行或项目将以粗体显示：

```py
In [1]: import purepy
In [2]: %timeit purepy.loop()
100 loops, best of 3: 8.26 ms per loop
In [3]: %timeit purepy.comprehension()
100 loops, best of 3: 5.39 ms per loop
In [4]: %timeit purepy.generator()
100 loops, best of 3: 5.07 ms per loop

```

任何命令行输入或输出都按照以下格式编写：

```py
$ time python simul.py # Performance Tuned
real    0m0.756s
user    0m0.714s
sys    0m0.036s

```

**新术语**和**重要词汇**将以粗体显示。您在屏幕上、菜单或对话框中看到的单词，例如，将以文本中的这种形式出现：“您可以通过双击矩形来导航到**调用图**或**调用者地图**选项卡。”

### 注意

警告或重要注意事项将以这样的框显示。

### 小贴士

小贴士和技巧将以这样的形式出现。

# 读者反馈

我们始终欢迎读者的反馈。让我们知道您对这本书的看法——您喜欢什么或可能不喜欢什么。读者反馈对我们开发您真正从中受益的标题非常重要。

如要向我们发送一般反馈，请发送电子邮件至`<feedback@packtpub.com>`，并在邮件主题中提及书籍标题。

如果你在某个领域有专业知识，并且对撰写或为书籍做出贡献感兴趣，请参阅我们的作者指南，网址为[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在您已经是 Packt 书籍的骄傲拥有者，我们有一些事情可以帮助您充分利用您的购买。

# 下载示例代码

您可以从您在[`www.packtpub.com`](http://www.packtpub.com)的账户下载所有已购买的 Packt 书籍的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)，并注册以将文件直接通过电子邮件发送给您。

# 错误清单

尽管我们已经尽一切努力确保我们内容的准确性，但错误仍然可能发生。如果您在我们的某本书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进此书的后续版本。如果您发现任何错误清单，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**错误清单提交表单**链接，并输入您的错误清单详情。一旦您的错误清单得到验证，您的提交将被接受，错误清单将被上传到我们的网站，或添加到该标题的错误清单部分。任何现有的错误清单都可以通过从[`www.packtpub.com/support`](http://www.packtpub.com/support)选择您的标题来查看。

# 侵权

互联网上版权材料的侵权是一个跨所有媒体的持续问题。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何非法副本，无论形式如何，请立即向我们提供位置地址或网站名称，以便我们可以追究补救措施。

请通过`<copyright@packtpub.com>`与我们联系，并提供涉嫌侵权材料的链接。

我们感谢您帮助我们保护作者，并确保我们能够为您提供有价值的内容。

# 询问

如果你在本书的任何方面遇到问题，可以通过`<questions@packtpub.com>`联系我们，我们将尽力解决。
