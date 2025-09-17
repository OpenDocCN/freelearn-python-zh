# 前言

近年来，Python 编程语言因其直观、有趣的语法以及丰富的顶级第三方库而受到极大的欢迎。Python 已成为许多入门和高级大学课程以及科学和工程等数值密集型领域的首选语言。其主要应用还在于机器学习、系统脚本和 Web 应用程序。

参考 Python 解释器 CPython，与底层语言（如 C、C++和 Fortran）相比，通常被认为效率不高。CPython 性能不佳的原因在于程序指令是由解释器处理的，而不是编译成高效的机器代码。虽然使用解释器有几个优点，例如可移植性和额外的编译步骤，但它确实在程序和机器之间引入了一个额外的间接层，这导致了效率降低的执行。

多年来，已经开发了许多策略来克服 CPython 的性能不足。本书旨在填补这一空白，并将教你如何始终如一地实现 Python 程序的良好性能。

本书将吸引广泛的读者，因为它涵盖了数值和科学代码的优化策略，以及提高 Web 服务和应用程序响应时间的策略。

本书可以从头到尾阅读；然而，章节被设计成是自包含的，这样如果你已经熟悉了前面的主题，你也可以跳到感兴趣的章节。

# 本书涵盖的内容

第一章*，基准和性能分析*，将教你如何评估 Python 程序的性能，以及如何识别和隔离代码中缓慢部分的实用策略。

第二章*，纯 Python 优化*，讨论了如何通过使用 Python 标准库和纯 Python 第三方模块中可用的有效数据结构和算法来提高你的运行时间。

第三章*，使用 NumPy 和 Pandas 进行快速数组操作*，是关于 NumPy 和 Pandas 包的指南。掌握这些包将允许你使用表达性强、简洁的接口实现快速数值算法。

第四章*，使用 Cython 进行 C 性能优化*，是关于 Cython 语言的教程，它使用与 Python 兼容的语法生成高效的 C 代码。

第五章，*探索编译器*，涵盖了可以用来将 Python 编译成高效机器代码的工具。本章将教你如何使用 Numba，一个针对 Python 函数的优化编译器，以及 PyPy，一个可以在运行时执行和优化 Python 程序的替代解释器。

第六章*，实现并发*，是异步和响应式编程的指南。我们将学习关键术语和概念，并演示如何使用 asyncio 和 RxPy 框架编写干净、并发的代码。

第七章*，并行处理*，是关于在多核处理器和 GPU 上实现并行编程的介绍。在本章中，您将学习如何使用 multiprocessing 模块以及通过使用 Theano 和 Tensorflow 表达您的代码来实现并行性。

第八章*，分布式处理*，通过专注于在分布式系统上运行并行算法来解决大规模问题和大数据，扩展了前一章的内容。本章将涵盖 Dask、PySpark 和 mpi4py 库。

第九章*，针对高性能的设计*，讨论了开发、测试和部署高性能 Python 应用程序的一般优化策略和最佳实践。

# 您需要为此书准备的内容

本书中的软件在 Python 3.5 和 Ubuntu 16.04 版本上进行了测试。然而，大多数示例也可以在 Windows 和 Mac OS X 操作系统上运行。

推荐的安装 Python 及其相关库的方式是通过 Anaconda 发行版，可以从[`www.continuum.io/downloads`](https://www.continuum.io/downloads)下载，适用于 Linux、Windows 和 Mac OS X。

# 本书面向的对象

本书旨在帮助 Python 开发者提高其应用程序的性能；预期读者具备 Python 的基本知识。

# 习惯用法

在本书中，您将找到多种文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称如下所示：“总结来说，我们将实现一个名为`ParticleSimulator.evolve_numpy`的方法，并将其与纯 Python 版本（重命名为`ParticleSimulator.evolve_python`）进行基准测试。”

代码块设置如下：

```py
    def square(x):
    return x * x

    inputs = [0, 1, 2, 3, 4]
    outputs = pool.map(square, inputs)

```

当我们希望引起您对代码块中特定部分的注意时，相关的行或项目将以粗体显示：

```py
    def square(x):
    return x * x

    inputs = [0, 1, 2, 3, 4]
 outputs = pool.map(square, inputs)

```

任何命令行输入或输出都如下所示：

```py
$ time python -c 'import pi; pi.pi_serial()' 
real 0m0.734s
user 0m0.731s
sys 0m0.004s

```

**新术语**和**重要词汇**以粗体显示。屏幕上看到的词汇，例如在菜单或对话框中，在文本中如下显示：“在右侧，点击 Callee Map 标签将显示函数成本的图表。”

警告或重要提示以如下框的形式出现。

小贴士和技巧以如下形式出现。

# 读者反馈

我们始终欢迎读者的反馈。请告诉我们您对这本书的看法——您喜欢或不喜欢的地方。读者反馈对我们很重要，因为它帮助我们开发出您真正能从中获得最大价值的标题。

要向我们发送一般反馈，只需发送电子邮件至`feedback@packtpub.com`，并在邮件主题中提及书籍的标题。

如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在，您已经是 Packt 图书的骄傲拥有者，我们有一些事情可以帮助您从您的购买中获得最大收益。

# 下载示例代码

您可以从[`www.packtpub.com`](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  使用您的电子邮件地址和密码登录或注册我们的网站。

1.  将鼠标指针悬停在顶部的 SUPPORT 标签上。

1.  点击代码下载与错误清单。

1.  在搜索框中输入书籍名称。

1.  选择您想要下载代码文件的书籍。

1.  从下拉菜单中选择您购买此书的来源。

1.  点击代码下载。

文件下载完成后，请确保使用最新版本的软件解压或提取文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

本书代码包也托管在 GitHub 上，地址为[`github.com/PacktPublishing/Python-High-Performance-Second-Edition`](https://github.com/PacktPublishing/Python-High-Performance-Second-Edition)。我们还有其他来自我们丰富图书和视频目录的代码包可供在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 下载本书的颜色图像

我们还为您提供了一个包含本书中使用的截图/图表的颜色图像的 PDF 文件。这些颜色图像将帮助您更好地理解输出的变化。您可以从[`www.packtpub.com/sites/default/files/downloads/PythonHighPerformanceSecondEdition_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/PythonHighPerformanceSecondEdition_ColorImages.pdf)下载此文件。

# 错误清单

尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以避免其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误表部分。

要查看之前提交的勘误表，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在勘误表部分。

# 侵权

互联网上版权材料的侵权是一个持续存在的问题，跨越所有媒体。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何非法副本，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过`copyright@packtpub.com`与我们联系，并提供疑似侵权材料的链接。

我们感谢您在保护我们作者和我们为您提供有价值内容的能力方面的帮助。

# 问题

如果您对本书的任何方面有问题，您可以联系`questions@packtpub.com`，我们将尽力解决问题。
