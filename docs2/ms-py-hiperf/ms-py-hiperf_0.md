# 前言

这本书的想法来源于 Packt 出版社的友好人士。他们希望找到能够深入研究 Python 高性能及其相关主题的人，无论是配置、可用的工具（如剖析器和其他性能增强技术），甚至是标准 Python 实现的替代方案。

话虽如此，我欢迎你加入*精通 Python 高性能*的行列。在这本书中，我们将涵盖与性能改进相关的所有内容。对主题的了解并非严格必要（尽管这不会有害），但需要了解 Python 编程语言，尤其是在一些特定于 Python 的章节中。

我们将从介绍配置的基本概念开始，包括它在开发周期中的位置以及与之相关的益处。之后，我们将转向完成工作所需的核心工具（剖析器和可视化剖析器）。然后，我们将探讨一系列优化技术，最后到达一个完全实用的章节，将提供一个实际的优化示例。

# 本书涵盖内容

第一章，*配置 101*，为那些不了解配置艺术的人提供相关信息。

第二章，*剖析器*，告诉你如何使用本书中提到的核心工具。

第三章，*可视化——使用 GUI 来理解剖析器输出*，涵盖了如何使用 pyprof2calltree 和 RunSnakeRun 工具。它还帮助开发者通过不同的可视化技术理解 cProfile 的输出。

第四章，*优化一切*，讲述了优化的基本过程以及一套每个 Python 开发者考虑其他选项之前应该遵循的良好/推荐实践。

第五章，*多线程与多进程*，讨论了多线程和多进程，并解释了何时以及如何应用它们。

第六章，*通用优化选项*，描述并展示了如何安装和使用 Cython 和 PyPy 来提高代码性能。

第七章，*使用 Numba、Parakeet 和 pandas 进行闪电般的数值计算*，讨论了帮助优化处理数字的 Python 脚本的工具。这些特定的工具（Numba、Parakeet 和 pandas）有助于使数值计算更快。

第八章，“将一切付诸实践”，提供了一个关于分析器的实际示例，找出其瓶颈，并使用本书中提到的工具和技术将其移除。最后，我们将比较使用每种技术的结果。

# 您需要为这本书准备的内容

在执行本书中提到的代码之前，您的系统必须安装以下软件：

+   Python 2.7

+   行分析器 1.0b2

+   Kcachegrind 0.7.4

+   RunSnakeRun 2.0.4

+   Numba 0.17

+   Parakeet 的最新版本

+   pandas 0.15.2

# 本书面向的对象

由于本书涉及与 Python 代码分析优化相关的所有主题，因此所有级别的 Python 开发者都将从本书中受益。

唯一的基本要求是您需要对 Python 编程语言有一些基本了解。

# 约定

在这本书中，您将找到许多不同的文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称如下所示："我们可以在`PROFILER`函数内部打印/收集我们认为相关的信息。"

代码块按照以下方式设置：

```py
import sys

def profiler(frame, event, arg):
    print 'PROFILER: %r %r' % (event, arg)

sys.setprofile(profiler)
```

当我们希望将您的注意力引向代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
Traceback (most recent call last): 
  File "cprof-test1.py", line 7, in <module> 
    runRe() ...
  File "/usr/lib/python2.7/cProfile.py", line 140, in runctx 
    exec cmd in globals, locals 
 File "<string>", line 1, in <module> 
NameError: name 're' is not defined 
```

任何命令行输入或输出都按照以下方式编写：

```py
$ sudo apt-get install python-dev libxml2-dev libxslt-dev

```

**新术语**和**重要词汇**以粗体显示。您在屏幕上看到的单词，例如在菜单或对话框中，在文本中如下所示："再次，当第一个函数调用选择了**调用者映射**时，我们可以看到我们脚本的整个映射。"

### 注意

警告或重要注意事项以如下方式出现在框中。

### 小贴士

小贴士和技巧看起来是这样的。

# 读者反馈

我们始终欢迎读者的反馈。告诉我们您对这本书的看法——您喜欢或不喜欢什么。读者反馈对我们很重要，因为它帮助我们开发出您真正能从中获得最大收益的标题。

要向我们发送一般反馈，只需发送电子邮件至`<feedback@packtpub.com>`，并在邮件主题中提及书的标题。

如果您在某个主题上具有专业知识，并且您有兴趣撰写或为本书做出贡献，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在您已经是 Packt 图书的骄傲拥有者，我们有一些事情可以帮助您从您的购买中获得最大收益。

## 下载示例代码

您可以从[`www.packtpub.com`](http://www.packtpub.com)下载您购买的所有 Packt 出版物的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

## 下载本书的彩色图像

我们还为您提供了一个包含本书中使用的截图/图表彩色图像的 PDF 文件。彩色图像将帮助您更好地理解输出中的变化。您可以从：[`www.packtpub.com/sites/default/files/downloads/9300OS_GraphicBundle.pdf`](https://www.packtpub.com/sites/default/files/downloads/9300OS_GraphicBundle.pdf)下载此文件。

## 勘误

尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在我们的某本书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误****提交****表**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。

要查看之前提交的勘误表，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在**勘误**部分。

## 侵权

互联网上版权材料的侵权是一个持续存在的问题，涉及所有媒体。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何形式的非法副本，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过`<copyright@packtpub.com>`与我们联系，并提供涉嫌侵权材料的链接。

我们感谢您在保护我们作者和我们为您提供有价值内容的能力方面的帮助。

## 询问

如果您本书的任何方面有问题，您可以通过`<questions@packtpub.com>`联系我们，我们将尽力解决问题。
