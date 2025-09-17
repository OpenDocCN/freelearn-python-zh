# 前言

写一本书远不止是应用语法和标点符号规则。同样，开发应用程序也需要更多编程语言和库 API 的知识。仅仅掌握语法规则和函数调用本身并不足以设计出能够使用户能够执行工作、保护宝贵数据并产生完美输出的应用程序。作为程序员，我们还需要能够将用户请求和期望转化为有效的界面设计，并选择最佳技术来实现它们。我们需要能够组织大型代码库，测试它们，并以保持其可管理性和避免粗心错误的方式维护它们。

本书的目标远不止是一本特定 GUI 工具包的参考手册。当我们走过一个虚构的工作场所场景时，您将体验到在一个小型办公室环境中作为应用程序程序员的感受。除了学习 Tkinter 和一些其他有用的库外，您还将学习许多您需要从编写短脚本过渡到编写功能齐全的图形应用程序的技能。完成本书后，您应该有信心能够为工作环境开发一个简单但实用的数据导向应用程序。

# 本书面向的对象

这本书是为那些已经学习了 Python 基础知识但尚未编写过多复杂脚本的新手而写的。我们将一步步引导您设计和创建一个更大的应用程序，并介绍一些有助于您作为程序员进步的技能。

它也针对那些已经使用 Python 进行数据科学、Web 开发或系统管理，但现在想扩展到创建 GUI 应用程序的人。我们将介绍创建本地 GUI 应用程序所需的知识和技能。

最后，这本书也可能对那些只想学习 Tkinter 的资深 Python 程序员有所帮助，因为书中详细介绍了使用 Tkinter 库的细微之处。

# 本书涵盖的内容

*第一章*，*Tkinter 简介*，向您介绍了 Tkinter 库的基础知识，并引导您创建一个基本的 Tkinter 应用程序。它还将介绍 IDLE 作为 Tkinter 应用程序的示例。

*第二章*，*设计 GUI 应用程序*，讲述了将一组用户需求转化为可实施设计的过程。

*第三章*，*使用 Tkinter 和 Ttk 小部件创建基本表单*，展示了如何创建一个基本的数据输入应用程序，该应用程序将输入的数据追加到 CSV 文件中。

*第四章*，*使用类组织我们的代码*，将向您介绍通用的面向对象编程技术，以及 Tkinter 特定类用法，这将使我们的 GUI 程序更易于维护和理解。

*第五章*，*通过验证和自动化减少用户错误*，展示了如何在我们的表单输入中自动填充和验证数据。

*第六章*，*为应用程序的扩展规划*，使你熟悉如何智能地将单个文件脚本拆分为多个文件，如何构建可以导入的 Python 模块，以及如何将大型代码库的关注点分离以使其更易于管理。

*第七章*，*使用 Menu 和 Tkinter 对话框创建菜单*，概述了使用 Tkinter 创建主菜单的过程。它还将展示如何使用几种内置对话框类型来实现常见的菜单功能。

*第八章*，*使用 Treeview 和 Notebook 导航记录*，详细介绍了使用 Ttk Treeview 和 Notebook 构建数据记录导航系统，以及将我们的应用程序从仅追加模式转换为全读写更新功能。

*第九章*，*使用样式和主题改进外观*，告诉你如何更改应用程序的颜色、字体和小部件样式，以及如何使用它们使应用程序更易用和吸引人。

*第十章*，*维护跨平台兼容性*，概述了 Python 和 Tkinter 技术，以保持你的应用程序在 Windows、macOS 和 Linux 系统上平稳运行。

*第十一章*，*使用 unittest 创建自动化测试*，讨论了如何通过自动化单元测试和集成测试验证你的代码。

*第十二章*，*使用 SQL 改进数据存储*，带你了解如何将我们的应用程序从 CSV 平面文件存储转换为 SQL 数据库存储。你还将了解所有关于 SQL 和关系数据模型的内容。

*第十三章*，*连接到云*，涵盖了如何处理网络资源，如 HTTP 服务器、REST 服务和 SFTP 服务器。你将学习如何与这些服务交互以下载和上传数据和文件。

*第十四章*，*使用 Thread 和 Queue 进行异步编程*，解释了如何使用异步和多线程编程在长时间运行过程中保持应用程序的响应性。

*第十五章*，*使用 Canvas 小部件可视化数据*，教你如何使用 Tkinter Canvas 小部件创建可视化和动画。你还将学习如何集成 Matplotlib 图表并构建一个简单的游戏。

*第十六章*，*使用 setuptools 和 cxFreeze 进行打包*，探讨了如何准备你的 Python 应用程序以作为 Python 包或独立可执行文件进行分发。

# 为了最大限度地利用本书

本书假设你已了解 Python 3 的基础知识。你应该知道如何使用内置类型和函数编写和运行简单的脚本，如何定义自己的函数，以及如何从标准库中导入模块。

您可以在运行当前版本的 Microsoft Windows、Apple macOS 或 GNU/Linux 发行版的计算机上阅读本书。请确保您已安装 Python 3 和 Tcl/Tk（*第一章*，*Tkinter 简介*包含 Windows、macOS 和 Linux 的安装说明），并且您有一个您感到舒适的代码编辑环境（我们建议使用 IDLE，因为它与 Python 一起提供并使用 Tkinter。我们不推荐使用 Jupyter、Spyder 或类似的环境，这些环境针对的是分析 Python 而不是应用开发）。在后面的章节中，您将需要访问互联网，以便您可以安装 Python 包和 PostgreSQL 数据库。

## 下载示例代码文件

本书代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Python-GUI-Programming-with-Tkinter-2E`](https://github.com/PacktPublishing/Python-GUI-Programming-with-Tkinter-2E)。我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们！

## 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`static.packt-cdn.com/downloads/9781801815925_ColorImages.pdf`](https://static.packt-cdn.com/downloads/9781801815925_ColorImages.pdf)。

## 使用的约定

本书使用了多种文本约定。

`CodeInText`: 表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。例如：“将代码保存在`solve_the_worlds_problems.py`中，并在终端提示符下键入`python solve_the_worlds_problems.py`来执行它。”

代码块设置如下：

```py
import tkinter as tk
root = tk.TK()
def solve():
  raise NotImplemented("Sorry!")
tk.Button(
  root, text="Solve the world's problems", command=solve
).pack()
root.mainloop() 
```

当我们希望您注意代码块中的特定部分，特别是指明现有代码的更改时，相关的行或项目将以粗体显示：

```py
import tkinter as tk
**from** **tkinter** **import** **messagebox**
root = tk.TK()
def solve():
  **messagebox.showinfo(****'The answer?'****,** **'Bananas?'****)**
tk.Button(
  root, text="Solve the world's problems", command=solve
).pack()
root.mainloop() 
```

注意，本书中使用的所有 Python 代码都使用 2 个空格缩进，而不是传统的 4 个空格缩进。

任何命令行输入或输出都使用`$`表示提示符，如下所示：

```py
$ mkdir Bananas
$ cp plantains.txt Bananas/ 
```

旨在 Python 外壳或 REPL 的命令行输入以`>>>`提示符打印，如下所示：

```py
>>> print('This should be run in a Python shell')
'This should be run in a Python shell' 
```

从外壳期望的输出将打印在没有提示符的行上。

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词，例如在菜单或对话框中。例如：“从**管理**面板中选择**系统信息**。”

警告或重要注意事项看起来像这样。

小贴士和技巧看起来像这样。

### 执行 Python 和 pip

当我们需要指导读者在本书中执行 Python 脚本时，我们将指示如下命令行：

```py
$ python myscript.py 
```

根据您的操作系统或 Python 配置，`python`命令可能执行的是 Python 2.x 而不是 Python 3.x。您可以通过运行以下命令来验证这一点：

```py
$ python --version
Python 3.9.7 
```

如果此命令在您的系统上输出 Python 2 而不是 Python 3，您需要更改任何 `python` 命令，以便您的代码在 Python 3 中执行。通常，这意味着使用 `python3` 命令，如下所示：

```py
$ python3 myscript.py 
```

同样的注意事项适用于用于从 Python 包索引安装库的 `pip` 命令。您可能需要使用 `pip3` 命令来安装库到您的 Python 3 环境中，例如：

```py
$ pip3 install --user requests 
```

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：请发送电子邮件至 `feedback@packtpub.com`，并在邮件主题中提及书籍的标题。如果您对本书的任何方面有疑问，请通过 `questions@packtpub.com` 发送电子邮件给我们。

**勘误**：尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告此错误。请访问 [`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上以任何形式发现我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过 `copyright@packtpub.com` 联系我们，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [`authors.packtpub.com`](https://authors.packtpub.com/)。

## 评论

请留下评论。一旦您阅读并使用了这本书，为何不在您购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解 Packt 的更多信息，请访问 [packtpub.com](http://packtpub.com)。

# 分享您的想法

读完 *Python GUI Programming with Tkinter, Second Edition* 后，我们很乐意听到您的想法！请[点击此处直接跳转到该书的 Amazon 评论页面](https://www.packtpub.com/)并分享您的反馈。

您的评论对我们和科技社区非常重要，并将帮助我们确保我们提供高质量的内容。
