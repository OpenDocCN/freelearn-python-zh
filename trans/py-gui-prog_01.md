# Tkinter 简介

欢迎，Python 程序员！如果您已经掌握了 Python 的基础知识，并希望开始设计强大的 GUI 应用程序，那么这本书就是为您准备的。

到目前为止，您无疑已经体验到了 Python 的强大和简单。也许您已经编写了 Web 服务，进行了数据分析，或者管理了服务器。也许您已经编写了游戏，自动化了例行任务，或者只是在代码中玩耍。但现在，您已经准备好去应对 GUI 了。

在如此强调网络、移动和服务器端编程的今天，开发简单的桌面 GUI 应用程序似乎越来越像是一门失传的艺术；许多经验丰富的开发人员从未学会创建这样的应用程序。真是一种悲剧！桌面计算机在工作和家庭计算中仍然发挥着至关重要的作用，能够为这个无处不在的平台构建简单、功能性的应用程序的能力应该成为每个软件开发人员工具箱的一部分。幸运的是，对于 Python 程序员来说，由于 Tkinter，这种能力完全可以实现。

在本章中，您将涵盖以下主题：

+   发现 Tkinter——一个快速、有趣、易学的 GUI 库，直接内置在 Python 标准库中

+   了解 IDLE——一个使用 Tkinter 编写并与 Python 捆绑在一起的编辑器和开发环境

+   创建两个“Hello World”应用程序，以学习编写 Tkinter GUI 的基础知识

# 介绍 Tkinter 和 Tk

Tk 的小部件库起源于“工具命令语言”（Tcl）编程语言。Tcl 和 Tk 是由约翰·奥斯特曼（John Ousterman）在 20 世纪 80 年代末担任伯克利大学教授时创建的，作为一种更简单的方式来编写在大学中使用的工程工具。由于其速度和相对简单性，Tcl/Tk 在学术、工程和 Unix 程序员中迅速流行起来。与 Python 本身一样，Tcl/Tk 最初是在 Unix 平台上诞生的，后来才迁移到 macOS 和 Windows。Tk 的实际意图和 Unix 根源仍然影响着它的设计，与其他工具包相比，它的简单性仍然是一个主要优势。

Tkinter 是 Python 对 Tk GUI 库的接口，自 1994 年以来一直是 Python 标准库的一部分，随着 Python 1.1 版本的发布，它成为了 Python 的事实标准 GUI 库。Tkinter 的文档以及进一步学习的链接可以在标准库文档中找到：[`docs.python.org/3/library/tkinter.html`](https://docs.python.org/3/library/tkinter.html)。

# 选择 Tkinter

想要构建 GUI 的 Python 程序员有几种工具包选择；不幸的是，Tkinter 经常被诋毁或被忽视为传统选项。公平地说，它并不是一种时髦的技术，无法用时髦的流行词和光辉的炒作来描述。然而，Tkinter 不仅适用于各种应用程序，而且具有以下无法忽视的优势：

+   它在标准库中：除了少数例外，Tkinter 在 Python 可用的任何地方都可以使用。无需安装`pip`，创建虚拟环境，编译二进制文件或搜索网络安装包。对于需要快速完成的简单项目来说，这是一个明显的优势。

+   它是稳定的：虽然 Tkinter 的开发并没有停止，但它是缓慢而渐进的。API 已经稳定多年，主要变化是额外的功能和错误修复。您的 Tkinter 代码可能会在未来数十年内保持不变。

+   它只是一个 GUI 工具包：与一些其他 GUI 库不同，Tkinter 没有自己的线程库、网络堆栈或文件系统 API。它依赖于常规的 Python 库来实现这些功能，因此非常适合将 GUI 应用于现有的 Python 代码。

+   它简单而直接：Tkinter 是直接、老派的面向对象的 GUI 设计。要使用 Tkinter，您不必学习数百个小部件类、标记或模板语言、新的编程范式、客户端-服务器技术或不同的编程语言。

当然，Tkinter 并非完美。它还具有以下缺点：

+   **外观和感觉**：它经常因其外观和感觉而受到批评，这些外观和感觉仍带有一些 1990 年代 Unix 世界的痕迹。在过去几年中，由于 Tk 本身的更新和主题化小部件库的添加，这方面已经有了很大改进。我们将在本书中学习如何修复或避免 Tkinter 更古老的默认设置。

+   **复杂的小部件**：它还缺少更复杂的小部件，比如富文本或 HTML 渲染小部件。正如我们将在本书中看到的，Tkinter 使我们能够通过定制和组合其简单小部件来创建复杂的小部件。

Tkinter 可能不是游戏用户界面或时尚商业应用的正确选择；但是，对于数据驱动的应用程序、简单实用程序、配置对话框和其他业务逻辑应用程序，Tkinter 提供了所需的一切以及更多。

# 安装 Tkinter

Tkinter 包含在 Python 标准库中，适用于 Windows 和 macOS 发行版。这意味着，如果您在这些平台上安装了 Python，您无需执行任何操作来安装 Tkinter。

但是，我们将专注于本书中的 Python 3.x；因此，您需要确保已安装了这个版本。

# 在 Windows 上安装 Python 3

您可以通过以下步骤从[python.org](https://www.python.org/)网站获取 Windows 的 Python 3 安装程序：

1.  转到[`www.python.org/downloads/windows`](http://www.python.org)。

1.  选择最新的 Python 3 版本。在撰写本文时，最新版本为 3.6.4，3.7 版本预计将在发布时推出。

1.  在文件部分，选择适合您系统架构的 Windows 可执行安装程序（32 位 Windows 选择 x86，64 位 Windows 选择 x86_64）。

1.  启动下载的安装程序。

1.  单击“自定义安装”。确保 tcl/tk 和 IDLE 选项已被选中（默认情况下应该是这样）。

1.  按照所有默认设置继续安装程序。

# 在 macOS 上安装 Python 3

截至目前，macOS 内置 Python 2 和 Tcl/Tk 8.5。但是，Python 2 计划在 2020 年停用，本书中的代码将无法与其一起使用，因此 macOS 用户需要安装 Python 3 才能跟随本书学习。

让我们按照以下步骤在 macOS 上安装 Python3：

1.  转到[`www.python.org/downloads/mac-osx/`](http://www.python.org)。

1.  选择最新的 Python 3 版本。在撰写本文时，最新版本为 3.6.4，但在出版时应该会有 3.7 版本。

1.  在文件部分，选择并下载`macOS 64 位/32 位安装程序`**。**

1.  启动您下载的`.pkg`文件，并按照安装向导的步骤进行操作，选择默认设置。

目前在 macOS 上没有推荐的升级到 Tcl/Tk 8.6 的方法，尽管如果您愿意，可以使用第三方工具来完成。我们的大部分代码将与 8.5 兼容，不过当某些内容仅适用于 8.6 时会特别提到。

# 在 Linux 上安装 Python 3 和 Tkinter

大多数 Linux 发行版都包括 Python 2 和 Python 3，但 Tkinter 并不总是捆绑在其中或默认安装。

要查看 Tkinter 是否已安装，请打开终端并尝试以下命令：

```py
python3 -m tkinter
```

这将打开一个简单的窗口，显示有关 Tkinter 的一些信息。如果您收到`ModuleNotFoundError`，则需要使用软件包管理器为 Python 3 安装您发行版的 Tkinter 包。在大多数主要发行版中，包括 Debian、Ubuntu、Fedora 和 openSUSE，这个包被称为`python3-tk`。

# 介绍 IDLE

IDLE 是一个集成开发环境，随 Windows 和 macOS Python 发行版捆绑提供（在大多数 Linux 发行版中通常也可以找到，通常称为 IDLE 或 IDLE3）。IDLE 使用 Tkinter 用 Python 编写，它不仅为 Python 提供了一个编辑环境，还是 Tkinter 的一个很好的示例。因此，虽然许多 Python 编码人员可能不认为 IDLE 的基本功能集是专业级的，而且您可能已经有了首选的 Python 代码编写环境，但我鼓励您在阅读本书时花一些时间使用 IDLE。

让我们熟悉 IDLE 的两种主要模式：**shell**模式和**editor**模式。

# 使用 IDLE 的 shell 模式

当您启动 IDLE 时，您将开始进入 shell 模式，这只是一个类似于在终端窗口中键入`python`时获得的 Python **Read-Evaluate-Print-Loop**（**REPL**）。

查看下面的屏幕截图中的 shell 模式：

![](img/589e283e-d8fa-4c1a-93b3-ffec82450966.png)

IDLE 的 shell 具有一些很好的功能，这些功能在命令行 REPL 中无法获得，如语法高亮和制表符补全。REPL 对 Python 开发过程至关重要，因为它使您能够实时测试代码并检查类和 API，而无需编写完整的脚本。我们将在后面的章节中使用 shell 模式来探索模块的特性和行为。如果您没有打开 shell 窗口，可以通过单击“开始”，然后选择“运行”，并搜索 Python shell 来打开一个。

# 使用 IDLE 的编辑器模式

编辑器模式用于创建 Python 脚本文件，稍后可以运行。当本书告诉您创建一个新文件时，这是您将使用的模式。要在编辑器模式中打开新文件，只需在菜单中导航到 File | New File，或者在键盘上按下*Ctrl* + *N*。

以下是一个可以开始输入脚本的窗口：

![](img/16738154-6cfd-4030-a682-8da92f4125dd.png)

您可以通过在编辑模式下按下*F5*而无需离开 IDLE 来运行脚本；输出将显示在一个 shell 窗口中。

# IDLE 作为 Tkinter 示例

在我们开始使用 Tkinter 编码之前，让我们快速看一下您可以通过检查 IDLE 的一些 UI 来做些什么。导航到主菜单中的 Options | Configure IDLE，打开 IDLE 的配置设置，您可以在那里更改 IDLE 的字体、颜色和主题、键盘快捷键和默认行为，如下面的屏幕截图所示：

![](img/0084883d-16d3-4c43-8b4d-ff027d48fd5f.png)

考虑一些构成此用户界面的组件：

+   有下拉列表和单选按钮，允许您在不同选项之间进行选择

+   有许多按钮，您可以单击以执行操作

+   有一个文本窗口可以显示多彩的文本

+   有包含组件组的标记帧

这些组件中的每一个都被称为**widget**；我们将在本书中遇到这些小部件以及更多内容，并学习如何像这里使用它们。然而，我们将从更简单的东西开始。

# 创建一个 Tkinter Hello World

通过执行以下步骤学习 Tkinter 的基础知识，创建一个简单的`Hello World` Tkinter 脚本：

1.  在 IDLE 或您喜欢的编辑器中创建一个新文件，输入以下代码，并将其保存为`hello_tkinter.py`：

```py
"""Hello World application for Tkinter"""

from tkinter import *
from tkinter.ttk import *

root = Tk()
label = Label(root, text="Hello World")
label.pack()
root.mainloop()
```

1.  通过按下*F5*在 IDLE 中运行此命令，或者在终端中键入以下命令：

```py
python3 hello_tkinter.py
```

您应该看到一个非常小的窗口弹出，其中显示了“Hello World”文本，如下面的屏幕截图所示：

![](img/c8342aa6-557d-416b-8058-e13fa2af884c.png)

1.  关闭窗口并返回到编辑器屏幕。让我们分解这段代码并谈谈它的作用：

+   `from tkinter import *`：这将 Tkinter 库导入全局命名空间。这不是最佳实践，因为它会填充您的命名空间，您可能会意外覆盖很多类，但对于非常小的脚本来说还可以。

+   `from tkinter.ttk import *`: 这导入了`ttk`或**主题**Tk 部件库。我们将在整本书中使用这个库，因为它添加了许多有用的部件，并改善了现有部件的外观。由于我们在这里进行了星号导入，我们的 Tk 部件将被更好看的`ttk`部件替换（例如，我们的`Label`对象）。

+   `root = Tk()`: 这将创建我们的根或主应用程序对象。这代表应用程序的主要顶层窗口和主执行线程，因此每个应用程序应该有且只有一个 Tk 的实例。

+   `label = Label(root, text="Hello World")`: 这将创建一个新的`Label`对象。顾名思义，`Label`对象只是用于显示文本（或图像）的部件。仔细看这一行，我们可以看到以下内容：

+   我们传递给`Label()`的第一个参数是`parent`或主部件。Tkinter 部件按层次结构排列，从根窗口开始，每个部件都包含在另一个部件中。每次创建部件时，第一个参数将是包含新部件的部件对象。在这种情况下，我们将`Label`对象放在主应用程序窗口上。

+   第二个参数是一个关键字参数，指定要显示在`Label`对象上的文本。

+   我们将新的`Label`实例存储在一个名为`label`的变量中，以便以后可以对其进行更多操作。

+   `label.pack()`: 这将新的标签部件放在其`parent`部件上。在这种情况下，我们使用`pack()`方法，这是您可以使用的三种**几何管理器**方法中最简单的一种。我们将在以后的章节中更详细地了解这些内容。

+   `root.mainloop()`: 这最后一行启动我们的主事件循环。这个循环负责处理所有事件——按键、鼠标点击等等——并且会一直运行直到程序退出。这通常是任何 Tkinter 脚本的最后一行，因为它之后的任何代码都不会在主窗口关闭之前运行。

花点时间玩弄一下这个脚本，在`root.mainloop()`调用之前添加更多的部件。你可以添加更多的`Label`对象，或者尝试`Button`（创建一个可点击的按钮）或`Entry`（创建一个文本输入字段）。就像`Label`一样，这些部件都是用`parent`对象（使用`root`）和`text`参数初始化的。不要忘记调用`pack()`将你的部件添加到窗口中。

你也可以尝试注释掉`ttk`导入，看看小部件外观是否有所不同。根据你的操作系统，外观可能会有所不同。

# 创建一个更好的 Hello World Tkinter

像我们刚才做的那样创建 GUI 对于非常小的脚本来说还可以，但更可扩展的方法是子类化 Tkinter 部件，以创建我们将随后组装成一个完成的应用程序的组件部件。

**子类化**只是一种基于现有类创建新类的方法，只添加或更改新类中不同的部分。我们将在本书中广泛使用子类化来扩展 Tkinter 部件的功能。

让我们构建一个更健壮的`Hello World`脚本，演示一些我们将在本书的其余部分中使用的模式。看一下以下步骤：

1.  创建一个名为`better_hello_tkinter.py`的文件，并以以下行开始：

```py
"""A better Hello World for Tkinter"""
import tkinter as tk
from tkinter import ttk
```

这一次，我们不使用星号导入；相反，我们将保持 Tkinter 和`ttk`对象在它们自己的命名空间中。这样可以避免全局命名空间被混乱，消除潜在的错误源。

星号导入（`from module import *`）在 Python 教程和示例代码中经常见到，但在生产代码中应该避免使用。Python 模块可以包含任意数量的类、函数或变量；当你进行星号导入时，你会导入所有这些内容，这可能导致一个导入覆盖从另一个模块导入的对象。如果你发现一个模块名在重复输入时很麻烦，可以将其别名为一个简短的名称，就像我们对 Tkinter 所做的那样。

1.  接下来，我们创建一个名为`HelloView`的新类，如下所示：

```py
class HelloView(tk.Frame):
    """A friendly little module"""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
```

我们的类是从`Tkinter.Frame`继承的。`Frame`类是一个通用的 Tk 小部件，通常用作其他小部件的容器。我们可以向`Frame`类添加任意数量的小部件，然后将整个内容视为单个小部件。这比在单个主窗口上单独放置每个按钮、标签和输入要简单得多。构造函数的首要任务是调用`super().__init__()`。`super()`函数为我们提供了对超类的引用（在本例中是我们继承的类，即`tk.Frame`）。通过调用超类构造函数并传递`*args`和`**kwargs`，我们的新`HelloWidget`类可以接受`Frame`可以接受的任何参数。

在较旧的 Python 版本中，`super()`必须使用子类的名称和对当前实例的引用来调用，例如`super(MyChildClass, self)`。Python 3 允许您无需参数调用它，但您可能会遇到使用旧调用的代码。

1.  接下来，我们将创建两个 Tkinter 变量对象来存储名称和问候语字符串，如下所示：

```py
        self.name = tk.StringVar()
        self.hello_string = tk.StringVar()
        self.hello_string.set("Hello World")
```

Tkinter 有一系列变量类型，包括`StringVar`、`IntVar`、`DoubleVar`和`BooleanVar`。您可能会想知道为什么我们要使用这些，当 Python 已经为所有这些（以及更多！）提供了完全良好的数据类型。Tkinter 变量不仅仅是数据的容器：它们具有常规 Python 变量缺乏的特殊功能，例如自动传播对所有引用它们的小部件的更改或在它们更改时触发事件的能力。在这里，我们将它们用作一种访问小部件中的数据的方式，而无需保留或传递对小部件本身的引用。

注意，将值设置为 Tkinter 变量需要使用`set()`方法，而不是直接赋值。同样，检索数据需要使用`get()`方法。在这里，我们将`hello_string`的值设置为`Hello World`。我们通过创建`Label`对象和`Entry`来开始构建我们的视图，如下所示：

```py
        name_label = ttk.Label(self, text="Name:")
        name_entry = ttk.Entry(self, textvariable=self.name)
```

`Label()`的调用看起来很熟悉，但`Entry`对象获得了一个新的参数：`textvariable`。通过将 Tkinter `StringVar`变量传递给此参数，`Entry`框的内容将绑定到该变量，我们可以在不需要引用小部件的情况下访问它。每当用户在`Entry`对象中输入文本时，`self.name`将立即在出现的任何地方更新。

1.  现在，让我们创建`Button`，如下所示：

```py
        ch_button = ttk.Button(self, text="Change", 
            command=self.on_change)
```

在上述代码中，我们再次有一个新的参数`command`，它接受对 Python 函数或方法的引用。我们通过这种方式传递的函数或方法称为回调，正如你所期望的那样，当单击按钮时将调用此回调。这是将函数绑定到小部件的最简单方法；稍后，我们将学习一种更灵活的方法，允许我们将各种按键、鼠标点击和其他小部件事件绑定到函数或方法调用。

确保此时不要实际调用回调函数——它应该是`self.on_change`，而不是`self.on_change()`。回调函数应该是对函数或方法的引用，而不是它的输出。

1.  让我们创建另一个`Label`，如下所示，这次用于显示我们的文本：

```py
        hello_label = ttk.Label(self, textvariable=self.hello_string,
            font=("TkDefaultFont", 64), wraplength=600)
```

在这里，我们将另一个`StringVar`变量`self.hello_string`传递给`textvariable`参数；在标签上，`textvariable`变量决定了将显示什么。通过这样做，我们可以通过简单地更改`self.hello_string`来更改标签上的文本。我们还将使用`font`参数设置一个更大的字体，该参数采用格式为`(font_name, font_size)`的元组。

您可以在这里输入任何字体名称，但它必须安装在系统上才能工作。Tk 有一些内置的别名，可以映射到每个平台上合理的字体，例如这里使用的`TkDefaultFont`。我们将在第八章“使用样式和主题改善外观”中学习更多关于在 Tkinter 中使用字体的知识。

`wraplength`参数指定文本在换行到下一行之前可以有多宽。我们希望当文本达到窗口边缘时换行；默认情况下，标签文本不会换行，因此会在窗口边缘被截断。通过将换行长度设置为 600 像素，我们的文本将在屏幕宽度处换行。

1.  到目前为止，我们已经创建了小部件，但尚未放置在`HelloView`上。让我们安排我们的小部件如下：

```py
        name_label.grid(row=0, column=0, sticky=tk.W)
        name_entry.grid(row=0, column=1, sticky=(tk.W + tk.E))
                ch_button.grid(row=0, column=2, sticky=tk.E)
                hello_label.grid(row=1, column=0, columnspan=3)
```

在这种情况下，我们使用`grid()`几何管理器添加我们的小部件，而不是之前使用的`pack()`几何管理器。顾名思义，`grid()`允许我们使用行和列在它们的`parent`对象上定位小部件，就像电子表格或 HTML 表格一样。我们的前三个小部件在第 0 行的三列中排列，而`hello_label`将在第二行（第 1 行）。`sticky`参数采用基本方向（`N`、`S`、`E`或`W`—您可以使用字符串或 Tkinter 常量），指定内容必须粘附到单元格的哪一侧。您可以将这些加在一起，以将小部件粘附到多个侧面；例如，通过将`name_entry`小部件粘附到东侧和西侧，它将拉伸以填满整个列的宽度。`grid()`调用`hello_label`使用`columnspan`参数。正如您可能期望的那样，这会导致小部件跨越三个网格列。由于我们的第一行为网格布局建立了三列，如果我们希望这个小部件填满应用程序的宽度，我们需要跨越所有三列。最后，我们将通过调整网格配置来完成`__init__()`方法：

```py
        self.columnconfigure(1, weight=1)
```

在上述代码中，`columnconfigure()`方法用于更改小部件的网格列。在这里，我们告诉它要比其他列更加重视第 1 列（第二列）。通过这样做，网格的第二列（我们的输入所在的位置）将水平扩展并压缩周围的列到它们的最小宽度。还有一个`rowconfigure()`方法，用于对网格行进行类似的更改。

1.  在完成`HelloModule`类之前，我们必须创建`ch_button`的回调，如下所示：

```py
def on_change(self):
    if self.name.get().strip():
        self.hello_string.set("Hello " + self.name.get())
    else:
        self.hello_string.set("Hello World")
```

要获取文本输入的值，我们调用其文本变量的`get()`方法。如果这个变量包含任何字符（请注意我们去除了空格），我们将设置我们的问候文本来问候输入的名字；否则，我们将只是问候整个世界。

通过使用`StringVar`对象，我们不必直接与小部件交互。这使我们不必在我们的类中保留大量小部件引用，但更重要的是，我们的变量可以从任意数量的来源更新或更新到任意数量的目的地，而无需明确编写代码来执行此操作。

1.  创建了`HelloView`后，我们转到实际的应用程序类，如下所示：

```py
class MyApplication(tk.Tk):
    """Hello World Main Application"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Hello Tkinter")
        self.geometry("800x600")
        self.resizable(width=False, height=False)
```

这次，我们对`Tk`进行了子类化，它将代表我们的主要应用程序对象。在 Tkinter 世界中，是否这样做是最佳实践存在一些争议。由于应用程序中只能有一个`Tk`对象，如果我们将来想要多个`MyApplication`对象，这可能会在某种程度上造成问题；对于简单的单窗口应用程序，这是完全可以的。

1.  与我们的模块一样，我们调用`super().__init__()`并传递任何参数。请注意，这次我们不需要一个`parent`小部件，因为`Tk`对象是根窗口，没有`parent`。然后有以下三个调用来配置我们的应用程序窗口：

+   `self.title()`: 这个调用设置窗口标题，通常出现在任务列表和/或我们的 OS 环境中的窗口栏中。

+   `self.geometry()`: 此调用以像素为单位设置窗口的大小，格式为`x * y`（宽度 x 高度）。

+   `self.resizable()`: 此调用设置程序窗口是否可以调整大小。我们在这里禁用调整大小，宽度和高度都禁用。

1.  我们通过将视图添加到主窗口来完成我们的应用程序类，如下所示：

```py
        HelloView(self).grid(sticky=(tk.E + tk.W + tk.N + tk.S))
        self.columnconfigure(0, weight=1)
```

请注意，我们在一行代码中创建和放置`HelloView`。我们在不需要保留对小部件的引用的情况下这样做，但由于`grid()`不返回值，如果您想在代码中稍后访问小部件，则必须坚持使用两个语句的版本。

因为我们希望视图填充应用程序窗口，我们的`grid()`调用将其固定在单元格的所有边上，我们的`columnconfigure()`调用会导致第一列扩展。请注意，我们省略了`row`和`column`参数，没有它们，`grid()`将简单地使用下一个可用行的第一列（在本例中为`0`，`0`）。

1.  定义了我们的类之后，我们将开始实际执行代码，如下所示：

```py
if __name__ == '__main__':
    app = MyApplication()
    app.mainloop()
```

在 Python 中，`if __name__ == '__main__':`是一个常见的习语，用于检查脚本是否直接运行，例如当我们在终端上键入`python3 better_hello_world.py`时。如果我们将此文件作为模块导入到另一个 Python 脚本中，此检查将为 false，并且之后的代码将不会运行。在此检查下方放置程序的主执行代码是一个良好的做法，这样您可以在更大的应用程序中安全地重用您的类和函数。

请记住，`MyApplication`是`Tk`的子类，因此它充当根窗口。我们只需要创建它，然后启动它的主循环。看一下以下的屏幕截图：

![](img/2687700f-7aa1-434a-be4e-0aeaf4ec4d4f.png)

对于“Hello World”应用程序来说，这显然是过度的，但它演示了使用子类将我们的应用程序分成模块的用法，这将大大简化我们构建更大程序时的布局和代码组织。

# 摘要

现在您已经安装了 Python 3，学会了使用 IDLE，品尝了 Tkinter 的简单性和强大性，并且已经看到了如何开始为更复杂的应用程序进行结构化，现在是时候开始编写一个真正的应用程序了。

在下一章中，您将开始在 ABQ AgriLabs 的新工作，并面临一个需要用您的编程技能和 Tkinter 解决的问题。您将学习如何分解这个问题，制定程序规范，并设计一个用户友好的应用程序，这将成为解决方案的一部分。
