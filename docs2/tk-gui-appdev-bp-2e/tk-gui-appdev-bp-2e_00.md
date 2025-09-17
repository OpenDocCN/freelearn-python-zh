# 前言

《Tkinter GUI 应用程序开发蓝图》，**第二版**将指导您通过使用 Python 和 Tkinter（Python 内置的 GUI 模块）开发实际应用图形界面的过程。

本书旨在突出 Tkinter 的特点和能力，同时展示在编写 GUI 程序过程中涉及的最佳实践，无论你选择哪个库来构建你的应用程序。在这里，你将学习如何使用 Tkinter 和 Python 开发令人兴奋、有趣且实用的 GUI 应用程序。

我们希望带你踏上一段充满乐趣的旅程，穿越超过 10 个来自不同问题领域的项目。随着我们在每个项目中开发新的应用，这本书也逐步建立了一个常用策略目录，用于开发现实世界中的应用。

# 这本书面向的对象

软件开发者、科学家、研究人员、工程师、学生以及那些对 Python 有基本了解的编程爱好者会发现这本书既有趣又有信息量。一个有编程背景且对 Python 充满热情的新手，通过一点额外的研究就能填补知识上的空白。

熟悉其他编程语言基本编程结构的开发者也可以通过阅读一些关于 Python 的简要内容来跟上进度。假设没有 GUI 编程经验。

# 本书涵盖的内容

第一章，*认识 Tkinter*，从零开始，提供了 Tkinter 的概述，并涵盖了如何创建根窗口、向根窗口添加小部件、使用几何管理器处理布局以及处理事件等细节。

第二章，*制作文本编辑器*，通过过程式编程风格开发了一个文本编辑器。它让读者首次体验了 Tkinter 的几个特性以及开发真实应用程序的感觉。

第三章，*可编程鼓机*，使用面向对象编程开发了一个能够演奏用户创作的节奏的鼓机。该应用程序还可以保存创作，并在以后编辑或重放它们。在这里，你将学习使用以模型优先的哲学设计 GUI 应用程序以及编写多线程 GUI 应用程序的技术。

第四章，《棋局》，介绍了使用模型-视图-控制器（MVC）架构构建 GUI 应用程序的关键方面。它还教授了如何将现实世界中的对象（棋类游戏）建模成程序可以操作的形式。此外，它向读者介绍了 Tkinter 画布小部件的强大功能。

第五章，*构建音频播放器*，涵盖了在使用外部库的同时展示如何使用许多不同的 Tkinter 小部件的概念。最重要的是，它展示了如何创建自己的 Tkinter 小部件，从而扩展 Tkinter 的多功能性。

第六章，*画布应用*，详细探讨了 Tkinter 的 Canvas 小部件。正如您将看到的，Canvas 小部件确实是 Tkinter 的亮点。本章还介绍了 GUI 框架的概念，从而为您的所有未来程序创建了可重用的代码。

第七章，《钢琴辅导》，展示了如何使用 JSON 表示给定的领域信息，然后将创建的数据应用于创建一个交互式应用程序。它还讨论了程序响应性的概念以及如何使用 Tkinter 来处理它。

第八章，*Canvas 中的乐趣*，致力于利用 Tkinter 画布小部件强大的可视化能力。它从几个重要的数学领域选取实例，构建不同种类的有用且美观的模拟。

第九章, 《多个趣味项目》，通过一系列小型但实用的项目进行讲解，展示了来自不同领域的问题，例如动画、网络编程、套接字编程、数据库编程、异步编程和多线程编程。

第十章，*杂项提示*，讨论了 GUI 编程中一些重要的方面，尽管这些内容在前几章中没有涉及，但它们在许多 GUI 程序中构成了一个共同的主题。

# 为了最大限度地利用这本书

我们假设读者对 Python 编程语言的基本结构有入门级别的熟悉程度。我们使用 Python 3.6 版本和 Tkinter 8.6，建议坚持使用这些确切版本以避免兼容性问题。

本书讨论的程序是在 Linux Mint 平台上开发的。然而，考虑到 Tkinter 的多平台能力，您可以在其他平台如 Windows、Mac OS 以及其他 Linux 发行版上轻松工作。各章节中提到了下载和安装其他项目特定模块和软件的链接。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便直接将文件通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在 [www.packtpub.com](http://www.packtpub.com/support) 登录或注册。

1.  选择“支持”标签。

1.  点击代码下载与勘误表。

1.  在搜索框中输入书籍名称，并遵循屏幕上的指示。

一旦文件下载完成，请确保您使用最新版本解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，地址为[`github.com/PacktPublishing/Tkinter-GUI-Application-Development-Blueprints-Second-Edition`](https://github.com/PacktPublishing/Tkinter-GUI-Application-Development-Blueprints-Second-Edition)。我们还有其他来自我们丰富图书和视频目录的代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到。去看看吧！

# 下载彩色图片

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/TkinterGUIApplicationDevelopmentBlueprintsSecondEdition_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/TkinterGUIApplicationDevelopmentBlueprintsSecondEdition_ColorImages.pdf).

# 使用的约定

本书使用了多种文本约定。

`CodeInText`: 表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：

代码块设置如下：

```py
def toggle_play_button_state(self):
  if self.now_playing:
    self.play_button.config(state="disabled")
  else:
    self.play_button.config(state="normal")
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
def on_loop_button_toggled(self):
  self.loop = self.to_loop.get()
  self.keep_playing = self.loop
  if self.now_playing:
 self.now_playing = self.loop
  self.toggle_play_button_state()
```

任何命令行输入或输出都应按照以下格式编写：

```py
>>> import pyglet
>>> help(pyglet.media)
```

**粗体**: 表示新术语、重要单词或屏幕上出现的单词。例如，菜单或对话框中的单词在文本中会像这样显示。以下是一个例子：“在我们的例子中，我们将向文件、编辑和关于菜单中添加菜单项。”

警告或重要提示会像这样显示。

小贴士和技巧看起来像这样。

# 联系我们

我们始终欢迎读者的反馈。

**总体反馈**：请发送邮件至 `feedback@packtpub.com` 并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请发送邮件至 `questions@packtpub.com`。

**勘误**: 尽管我们已经尽最大努力确保内容的准确性，错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告这一点，我们将不胜感激。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果你在互联网上以任何形式遇到我们作品的非法副本，如果你能提供位置地址或网站名称，我们将不胜感激。请通过发送链接至`copyright@packtpub.com`与我们联系。

**如果您想成为一名作者**：如果您在某个领域有专业知识，并且对撰写或参与一本书籍感兴趣，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评价。一旦您阅读并使用了这本书，为何不在购买它的网站上留下评价呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 公司可以了解您对我们产品的看法，而我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解 Packt 的更多信息，请访问 [packtpub.com](https://www.packtpub.com/).
