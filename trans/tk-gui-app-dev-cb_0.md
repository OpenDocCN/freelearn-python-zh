# 前言

作为一种更多用途的编程语言之一，Python 以其“电池包含”哲学而闻名，其中包括其标准库中丰富的模块集；Tkinter 是用于构建桌面应用程序的库。Tkinter 是建立在 Tk GUI 工具包之上的，是快速 GUI 开发的常见选择，复杂的应用程序可以从该库的全部功能中受益。本书涵盖了 Tkinter 和 Python GUI 开发的所有问题和解决方案。

*Tkinter GUI 应用程序开发食谱*首先概述了 Tkinter 类，同时提供了有关基本主题的示例，例如布局模式和事件处理。接下来，本书介绍了如何开发常见的 GUI 模式，例如输入和保存数据，通过菜单和对话框导航，以及在后台执行长时间操作。然后，您可以使您的应用程序有效地利用网络资源，并在画布上执行图形操作以及相关任务，例如检测项目之间的碰撞。最后，本书介绍了使用主题小部件，这是 Tk 小部件的扩展，具有更本地的外观和感觉。

通过本书，您将深入了解 Tkinter 类，并知道如何使用它们构建高效和丰富的 GUI 应用程序。

# 这本书是为谁准备的

这本书的目标读者是熟悉 Python 语言基础知识（语法、数据结构和面向对象编程）的开发人员，希望学习 GUI 开发常见挑战的有效解决方案，并希望发现 Tkinter 可以提供的有趣功能，以构建复杂的应用程序。

您不需要有 Tkinter 或其他 GUI 开发库的先前经验，因为本书的第一部分将通过介绍性用例教授库的基础知识。

# 本书涵盖的内容

第一章，*开始使用 Tkinter*，介绍了 Tkinter 程序的结构，并向您展示如何执行最常见的任务，例如创建小部件和处理用户事件。

第二章，*窗口布局*，演示了如何使用几何管理器放置小部件并改进大型应用程序的布局。

第三章，*自定义小部件*，深入探讨了 Tkinter 小部件的配置和外观自定义。

第四章，*对话框和菜单*，教会您如何通过菜单和对话框改进 Tkinter 应用程序的导航。

第五章，*面向对象编程和 MVC*，教会您如何在 Tkinter 应用程序中有效应用设计模式。

第六章，*异步编程*，涵盖了执行长时间操作而不冻结应用程序的几个方法——这是 GUI 开发中经常出现的问题。

第七章，*画布和图形*，探索了画布小部件以及您可以添加到画布的项目类型以及如何操作它们。

第八章，*主题小部件*，教会您如何使用 Tk 主题小部件集扩展 Tkinter 应用程序。

# 充分利用本书

要开始并运行，用户需要安装以下技术：

+   Python 3.x

+   任何操作系统

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便直接通过电子邮件接收文件。

您可以按照以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)登录或注册。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

文件下载后，请确保使用最新版本的软件解压或提取文件夹。

+   Windows 的 WinRAR/7-Zip

+   Mac 的 Zipeg/iZip/UnRarX

+   Linux 的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Tkinter-GUI-Application-Development-Cookbook`](https://github.com/PacktPublishing/Tkinter-GUI-Application-Development-Cookbook)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有来自我们丰富书籍和视频目录的其他代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。去看看吧！

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图像。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/TkinterGUIApplicationDevelopmentCookbook_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/TkinterGUIApplicationDevelopmentCookbook_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词，数据库表名，文件夹名，文件名，文件扩展名，路径名，虚拟 URL，用户输入和 Twitter 句柄。这是一个例子："`delete()`方法接受两个参数，指示应删除的字符范围。"

代码块设置如下：

```py
from tkinter import * 

root = Tk() 
btn = Button(root, text="Click me!") 
btn.config(command=lambda: print("Hello, Tkinter!"))
btn.pack(padx=120, pady=30)
root.title("My Tkinter app")
root.mainloop()
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项将以粗体显示：

```py
def show_caption(self, event):
    caption = tk.Label(self, ...)
    caption.place(in_=event.widget, x=event.x, y=event.y)
    # ...
```

**粗体**：表示一个新术语，一个重要单词，或者您在屏幕上看到的单词。例如，菜单或对话框中的单词会在文本中以这种方式出现。这是一个例子："第一个将被标记为选择文件。"

警告或重要提示会以这种方式出现。提示和技巧会以这种方式出现。
