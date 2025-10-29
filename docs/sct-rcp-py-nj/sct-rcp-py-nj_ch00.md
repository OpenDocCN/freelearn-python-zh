# 前言

许多读者可能觉得自己已经掌握了 Python 语言，并知道编写利用语言最佳特性的应用程序所需的一切。本书旨在深入探讨 Python 及相关技术的一些方面，这些方面一些开发者从未经历过。

本书将揭示与标准库实现相关的 Python 的未知或误解的方面，并提供对模块实际工作方式的理解。本书展示了集合和数学模块的正确实现，以及如小数和分数等数字，这些数字将帮助读者拓宽视野。在详细学习内部特殊方法之前，读者将学习装饰器、上下文管理器、协程和生成器函数。本书探讨了 CPython 解释器，涵盖了可以改变环境功能以及改进正常 Python 体验的替代交互式外壳的命令选项。读者将游览 PyPy 项目，在那里他们将接触到几种提高应用程序速度和并发性的新方法。对最新版本的几个 Python 增强提案进行了审查，以了解 Python 未来的发展。最后，它提供了有关记录 Python 代码的不同方法的详细信息。

# 这本书面向谁

本书旨在为想要学习如何以新方式使用 Python 来提高应用程序性能的 Python 软件开发者而写。为了充分利用本书，必须具备 Python 的实际知识。

# 这本书涵盖的内容

第一章，*使用 Python 模块*，探讨了 Python 包、模块和命名空间，使用虚拟环境，以及打包 Python 代码以进行分发。

第二章，*利用 Python 解释器*，探讨了 Python 命令行选项，定制交互会话，在 Windows OS 上使用 Python，以及替代 Python 交互式外壳。

第三章，*使用装饰器*，回顾了 Python 函数，并展示了如何使用装饰器来改进它们。

第四章，*使用 Python 集合*，涵盖了容器，并深入探讨了 Python 中可用的集合。

第五章，*生成器、协程和并行处理*，专注于 Python 中的迭代以及它与生成器的协同工作，然后转向并发和并行处理。

第六章，*使用 Python 的数学模块*，深入探讨了 Python 如何实现各种数学运算。

第七章，*使用 PyPy 提高 Python 性能*，概述了使用即时编译来提高 Python 性能。

第八章，*Python 增强提案*，讨论了如何处理 Python 语言的改进，并查看了一些当前的提案。

第九章，*使用 LyX 进行文档编制*，展示了不同的技术和工具来编制代码文档。

# 要充分利用本书

虽然许多主题以即使是初学者也能理解基本原理的方式进行了覆盖，但需要具备 Python 的中级知识。具体来说，假设您有使用交互式 Python 解释器和编写 Python 文件的经验，以及如何导入模块和如何使用面向对象原则的经验。

本书使用 Python 3.6 进行示例，除非另有说明。虽然简要讨论了替代实现，但本书假设使用的是基本的 CPython 实现。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)登录或注册。

1.  选择 SUPPORT 标签。

1.  点击代码下载与勘误。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

下载文件后，请确保使用最新版本的软件解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

书籍的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Secret-Recipes-of-the-Python-Ninja`](https://github.com/PacktPublishing/Secret-Recipes-of-the-Python-Ninja)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的图书和视频目录的代码包可供下载，网址为**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从[`www.packtpub.com/sites/default/files/downloads/SecretRecipesofthePythonNinja_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/SecretRecipesofthePythonNinja_ColorImages.pdf)下载它。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“`sqrt(x)`函数返回`√x`。”

代码块设置为以下格式：

```py
def print_funct(arg):
    print(arg)
    if __name__ == "__main__":
        import sys
        print_funct(sys.argv[1])
```

任何命令行输入或输出都按照以下方式编写：

```py
>>> import random
>>> random.randint(0, 1000)
607
```

**粗体**: 表示新术语、重要单词或屏幕上出现的单词。例如，菜单或对话框中的单词在文本中会这样显示。例如：“例如，在创建这本书的过程中，这位作者在创建教程的 PDF 副本时遇到了问题，因为在将 EPS 图像转换为 PDF 图像时不断出现错误。”

警告或重要提示会这样显示。

小贴士和技巧会这样显示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**: 请通过`feedback@packtpub.com`发送电子邮件，并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请通过`questions@packtpub.com`发送电子邮件给我们。

**勘误**: 尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告，我们将不胜感激。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果您在互联网上以任何形式遇到我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packtpub.com`与我们联系，并附上材料的链接。

**如果您有兴趣成为作者**: 如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用了这本书，为什么不在您购买它的网站上留下评论呢？潜在的读者可以看到并使用您的客观意见来做出购买决定，Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问[packtpub.com](https://www.packtpub.com/)。[packtpub.com](https://www.packtpub.com/)
