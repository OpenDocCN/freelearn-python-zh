# 前言

Qt 是 GUI 应用程序开发中最广泛使用和最灵活的框架之一。本书将向您介绍使用 PyQt 和 PySide 等 Python 绑定开发基于 Python 的 Qt GUI 应用程序。我们将从描述 PyQt 和 PySide GUI 元素开始，这将成为本书的参考。完成本书后，您将能够从头开始创建美观的 Python GUI 应用程序。

# 本书面向对象

本书面向希望开发现代、响应迅速且吸引人的 GUI 和跨平台应用程序的 Python 开发者。不需要具备 Qt 或 QML 的先验知识。

# 本书涵盖的内容

第一章，*Python 和 Qt 简介*，描述了 Qt 的基础知识，并将其与 Python 进行比较。它还讨论了使用 Python 实现各种 GUI 的领域。

第二章，*QML 概述*，介绍了 Qt 建模语言。

第三章，*Qt Quick 库*，讨论了 Qt Quick 库的实现。

第四章，*PyQt 和 PySide 入门*，介绍了 PyQt、PySide 以及 GUI 应用程序构建中最有用的方法，以及样式化的概念。

第五章，*使用 QWidget 和主窗口*，展示了如何在 GUI 应用程序中使用小部件和框架，并讨论了 PyQt5、PySide2 及其小部件和框架的功能。

第六章，*使用框架、标签和文本字段*，处理 PyQt5 和 PySide2 的标签、文本字段、组合框和相关元素特性。

第七章，*使用组合框和列表视图*，讨论了 PyQt5 和 PySide2 中的组合框、列表视图和表格，以及这些元素的功能和特性。

第八章，*实现按钮*，描述了在 PyQt5 和 PySide2 GUI 应用程序中实现按钮和其他控件元素。

第九章，*图形基础*，描述了用于处理图形、绘图和绘画、图形视图、图形场景和其他可视化组件的 PyQt 和 PySide 类。

第十章，*图形表示*，专注于在应用程序中使用的图形和其它可视化组件的 PyQt 和 PySide 类。

第十一章，*图形效果和多媒体*，专注于在应用程序中表示图形的 PyQt 和 PySide 类、图形动画以及使用应用程序中的附加图形工具。

第十二章，*文件、打印机和其他对话框*，处理了 PyQt 和 PySide 类用于处理文件，如打开、保存和创建。它还关注输入对话框和信息框。

第十三章，*创建布局*，专注于 PyQt 和 PySide 类用于定位元素，并解释了如何创建布局以使您的 GUI 应用程序响应并具有分辨率无关性。

第十四章，*实现数据库*，展示了如何创建数据库以及如何与 SQL/NoSQL 一起工作，并添加持久性。

第十五章，*信号、槽和事件处理器*，介绍了信号和槽，并解释了如何添加 PyQt 和 PySide 元素以及元素之间的功能特性和通信可能性。此外，它还描述了 PyQt 和 PySide GUI 中的事件以及如何实现事件处理器。

第十六章，*线程和多进程*，展示了使用 PyQt 和 PySide 创建 GUI 时的线程和多进程方法的实现。此外，它还关注构建大型应用程序的主要原则以及数据库处理。

第十七章，*完成应用程序、嵌入式设备和测试*，完成了应用程序，进行了测试，并描述了将应用程序部署到嵌入式平台的基础知识。

# 为了充分利用本书

本书致力于在 Python 编程语言及其相关工具的背景下进行 Python 的实现开发。为了充分利用这些材料，读者必须对 Python 有一定的了解。Qt 有许多子版本；请确保您安装了 Qt 5、Python 2 和 Python 3，以便代码与本书兼容。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了此书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)上登录或注册。

1.  选择 SUPPORT 标签。

1.  点击代码下载与勘误表。

1.  在搜索框中输入书名，并遵循屏幕上的说明。

下载文件后，请确保使用最新版本的以下软件解压缩或提取文件夹：

+   Windows 上的 WinRAR/7-Zip

+   Mac 上的 Zipeg/iZip/UnRarX

+   Linux 上的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，地址为[`github.com/PacktPublishing/Hands-On-Qt-for-Python-Developers`](https://github.com/PacktPublishing/Hands-On-Qt-for-Python-Developers)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的书籍和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 下载彩色图像

我们还提供包含本书中使用的截图/图表的彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789612790_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781789612790_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“您可以通过遵循说明从 Qt 官方源下载并安装 PySide2 绑定，或者您可以通过`pip`从**PyPI**（Python 包索引）安装它。”

代码块设置如下：

```py
...
class UApp(UTools):

    def __init__(self):
        UTools.__init__(self)
        self.uaps1 = "I am a coder, and I do"
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
import "Path/To/JavaScript/File" as Identifier
```

任何命令行输入或输出都如下所示：

```py
> pip install PyQt5
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“导航到 选择一个模板 | 应用程序 | Qt Quick 应用程序 - 空白 然后点击 选择....”

警告或重要注意事项如下所示。

小技巧和技巧如下所示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`与我们联系。

**勘误**：尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告这一点。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上以任何形式遇到我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用过这本书，为何不在您购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解 Packt 的更多信息，请访问 [packt.com](http://www.packt.com/).
