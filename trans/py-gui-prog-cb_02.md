# 第二章。布局管理

在本章中，我们将使用 Python 3 来布局我们的 GUI：

+   在标签框架小部件内排列几个标签

+   使用填充在小部件周围添加空间

+   小部件如何动态扩展 GUI

+   通过在框架内嵌套框架来对齐 GUI 小部件

+   创建菜单栏

+   创建选项卡小部件

+   使用网格布局管理器

# 介绍

在这一章中，我们将探讨如何在小部件内部排列小部件，以创建我们的 Python GUI。学习 GUI 布局设计的基础知识将使我们能够创建外观出色的 GUI。有一些技术将帮助我们实现这种布局设计。

网格布局管理器是内置在 tkinter 中的最重要的布局工具之一，我们将使用它。

我们可以很容易地使用 tk 来创建菜单栏，选项卡控件（又名 Notebooks）以及许多其他小部件。

tk 中默认缺少的一个小部件是状态栏。

在本章中，我们将不费力地手工制作这个小部件，但这是可以做到的。

# 在标签框架小部件内排列几个标签

`LabelFrame`小部件允许我们以有组织的方式设计我们的 GUI。我们仍然使用网格布局管理器作为我们的主要布局设计工具，但通过使用`LabelFrame`小部件，我们可以更好地控制 GUI 设计。

## 准备工作

我们开始向我们的 GUI 添加越来越多的小部件，并且我们将在接下来的示例中使 GUI 完全功能。在这里，我们开始使用`LabelFrame`小部件。我们将重用上一章最后一个示例中的 GUI。

## 如何做...

在 Python 模块的底部朝向主事件循环上方添加以下代码：

```py
# Create a container to hold labels
labelsFrame = ttk.LabelFrame(win, text=' Labels in a Frame ') # 1
labelsFrame.grid(column=0, row=7)

# Place labels into the container element # 2
ttk.Label(labelsFrame, text="Label1").grid(column=0, row=0)
ttk.Label(labelsFrame, text="Label2").grid(column=1, row=0)
ttk.Label(labelsFrame, text="Label3").grid(column=2, row=0)

# Place cursor into name Entry
nameEntered.focus()
```

![操作步骤...](img/B04829_02_01.jpg)

### 注意

通过更改我们的代码，我们可以轻松地垂直对齐标签，如下所示。请注意，我们唯一需要更改的是列和行编号。

```py
# Place labels into the container element – vertically # 3
ttk.Label(labelsFrame, text="Label1").grid(column=0, row=0)
ttk.Label(labelsFrame, text="Label2").grid(column=0, row=1)
ttk.Label(labelsFrame, text="Label3").grid(column=0, row=2)
```

![操作步骤...](img/B04829_02_01_1.jpg)

## 它是如何工作的...

注释＃1：在这里，我们将创建我们的第一个 ttk LabelFrame 小部件并为框架命名。父容器是`win`，即我们的主窗口。

在注释＃2 之后的三行代码创建标签名称并将它们放置在 LabelFrame 中。我们使用重要的网格布局工具来排列 LabelFrame 内的标签。此布局管理器的列和行属性赋予我们控制 GUI 布局的能力。

### 注意

我们标签的父级是 LabelFrame，而不是主窗口的`win`实例变量。我们可以在这里看到布局层次的开始。

突出显示的注释＃3 显示了通过列和行属性轻松更改布局的方法。请注意，我们如何将列更改为 0，并且如何通过按顺序编号行值来垂直叠加我们的标签。

### 注意

ttk 的名称代表“主题 tk”。tk-themed 小部件集是在 Tk 8.5 中引入的。

## 还有更多...

在本章的后面的一个示例中，我们将嵌套 LabelFrame(s)在 LabelFrame(s)中，以控制我们的 GUI 布局。

# 使用填充在小部件周围添加空间

我们的 GUI 正在很好地创建。接下来，我们将通过在它们周围添加一点空间来改善我们小部件的视觉效果，以便它们可以呼吸...

## 准备工作

尽管 tkinter 可能曾经以创建丑陋的 GUI 而闻名，但自 8.5 版本以来（随 Python 3.4.x 一起发布），这种情况发生了显著变化。您只需要知道如何使用可用的工具和技术。这就是我们接下来要做的。

## 如何做...

首先展示了围绕小部件添加间距的程序化方法，然后我们将使用循环以更好的方式实现相同的效果。

我们的 LabelFrame 看起来有点紧凑，因为它与主窗口向底部融合在一起。让我们现在来修复这个问题。

通过添加`padx`和`pady`修改以下代码行：

```py
labelsFrame.grid(column=0, row=7, padx=20, pady=40)
```

现在我们的 LabelFrame 有了一些空间：

![操作步骤...](img/B04829_02_02.jpg)

## 它是如何工作的...

在 tkinter 中，通过使用名为`padx`和`pady`的内置属性来水平和垂直地添加空间。这些属性可以用于在许多小部件周围添加空间，分别改善水平和垂直对齐。我们在 LabelFrame 的左右两侧硬编码了 20 像素的空间，并在框架的顶部和底部添加了 40 像素。现在我们的 LabelFrame 比以前更加突出。

### 注意

上面的屏幕截图只显示了相关的更改。

我们可以使用循环在 LabelFrame 内包含的标签周围添加空间：

```py
for child in labelsFrame.winfo_children(): 
    child.grid_configure(padx=8, pady=4)
```

现在 LabelFrame 小部件内的标签周围也有一些空间：

![它是如何工作的...](img/B04829_02_02_1.jpg)

`grid_configure()`函数使我们能够在主循环显示 UI 元素之前修改它们。因此，我们可以在首次创建小部件时，而不是硬编码数值，可以在文件末尾的布局中工作，然后在创建 GUI 之前进行间距调整。这是一个不错的技巧。

`winfo_children()`函数返回属于`labelsFrame`变量的所有子项的列表。这使我们能够循环遍历它们并为每个标签分配填充。

### 注意

要注意的一件事是标签右侧的间距实际上并不明显。这是因为 LabelFrame 的标题比标签的名称长。我们可以通过使标签的名称更长来进行实验。

```py
ttk.Label(labelsFrame, text="Label1 -- sooooo much loooonger...").grid(column=0, row=0)
```

现在我们的 GUI 看起来像下面这样。请注意，现在在长标签旁边的右侧添加了一些空间。最后一个点没有触及 LabelFrame，如果没有添加的空间，它就会触及。

![它是如何工作的...](img/B04829_02_02_2.jpg)

我们还可以删除 LabelFrame 的名称，以查看`padx`对定位我们的标签的影响。

![它是如何工作的...](img/B04829_02_02_3.jpg)

# 小部件如何动态扩展 GUI

您可能已经注意到在之前的屏幕截图中，并通过运行代码，小部件具有扩展自身以视觉显示其文本所需的能力。

### 注意

Java 引入了动态 GUI 布局管理的概念。相比之下，像 VS.NET 这样的可视化开发 IDE 以可视化方式布局 GUI，并且基本上是在硬编码 UI 元素的 x 和 y 坐标。

使用`tkinter`，这种动态能力既带来了优势，也带来了一点挑战，因为有时我们的 GUI 会在我们不希望它太动态时动态扩展！好吧，我们是动态的 Python 程序员，所以我们可以想出如何最好地利用这种奇妙的行为！

## 准备工作

在上一篇食谱的开头，我们添加了一个标签框小部件。这将一些控件移动到第 0 列的中心。我们可能不希望这种修改影响我们的 GUI 布局。接下来，我们将探讨一些修复这个问题的方法。

## 如何做...

让我们首先注意一下 GUI 布局中正在发生的微妙细节，以更好地理解它。

我们正在使用网格布局管理器小部件，并且它以从零开始的网格布局排列我们的小部件。

| 第 0 行；第 0 列 | 第 0 行；第 1 列 | 第 0 行；第 2 列 |
| --- | --- | --- |
| 第 1 行；第 0 列 | 第 1 行；第 1 列 | 第 1 行；第 2 列 |

使用网格布局管理器时，任何给定列的宽度由该列中最长的名称或小部件确定。这会影响所有行。

通过添加 LabelFrame 小部件并给它一个比某些硬编码大小小部件（如左上角的标签和下面的文本输入）更长的标题，我们动态地将这些小部件移动到第 0 列的中心，并在这些小部件的左右两侧添加空间。

顺便说一句，因为我们为 Checkbutton 和 ScrolledText 小部件使用了 sticky 属性，它们仍然附着在框架的左侧。

让我们更详细地查看本章第一个示例的屏幕截图：

![如何做...](img/B04829_02_02_4.jpg)

我们添加了以下代码来创建 LabelFrame，然后将标签放入此框架中：

```py
# Create a container to hold labels
labelsFrame = ttk.LabelFrame(win, text=' Labels in a Frame ')
labelsFrame.grid(column=0, row=7)
```

由于 LabelFrame 的 text 属性（显示为 LabelFrame 的标题）比我们的**Enter a name:**标签和下面的文本框条目都长，这两个小部件会动态地居中于列 0 的新宽度。

列 0 中的 Checkbutton 和 Radiobutton 小部件没有居中，因为我们在创建这些小部件时使用了`sticky=tk.W`属性。

对于 ScrolledText 小部件，我们使用了`sticky=tk.WE`，这将小部件绑定到框架的西（即左）和东（即右）两侧。

让我们从 ScrolledText 小部件中删除 sticky 属性，并观察这个改变的影响。

```py
scr = scrolledtext.ScrolledText(win, width=scrolW, height=scrolH, wrap=tk.WORD)
#### scr.grid(column=0, sticky='WE', columnspan=3)
scr.grid(column=0, columnspan=3)
```

现在我们的 GUI 在 ScrolledText 小部件的左侧和右侧都有新的空间。因为我们使用了`columnspan=3`属性，我们的 ScrolledText 小部件仍然跨越了所有三列。

![操作步骤...](img/B04829_02_02_5.jpg)

如果我们移除`columnspan=3`，我们会得到以下 GUI，这不是我们想要的。现在我们的 ScrolledText 只占据列 0，并且由于其大小，它拉伸了布局。

![操作步骤...](img/B04829_02_02_6.jpg)

将我们的布局恢复到添加 LabelFrame 之前的方法之一是调整网格列位置。将列值从 0 更改为 1。

```py
labelsFrame.grid(column=1, row=7, padx=20, pady=40)
```

现在我们的 GUI 看起来像这样：

![操作步骤...](img/B04829_02_03.jpg)

## 它是如何工作的...

因为我们仍在使用单独的小部件，所以我们的布局可能会混乱。通过将 LabelFrame 的列值从 0 移动到 1，我们能够将控件放回到它们原来的位置，也是我们喜欢它们的位置。至少最左边的标签、文本、复选框、滚动文本和单选按钮小部件现在位于我们打算的位置。第二个标签和文本`Entry`位于列 1，它们自己对齐到了**Labels in a Frame**小部件的长度中心，所以我们基本上将我们的对齐挑战移到了右边一列。这不太明显，因为**Choose a number:**标签的大小几乎与**Labels in a Frame**标题的大小相同，因此列宽已经接近 LabelFrame 生成的新宽度。

## 还有更多...

在下一个教程中，我们将嵌入框架以避免我们在本教程中刚刚经历的小部件意外错位。

# 通过嵌入框架来对齐 GUI 小部件

如果我们在框架中嵌入框架，我们将更好地控制 GUI 布局。这就是我们将在本教程中做的事情。

## 准备工作

Python 及其 GUI 模块的动态行为可能会对我们真正想要的 GUI 外观造成一些挑战。在这里，我们将嵌入框架以获得对布局的更多控制。这将在不同 UI 元素之间建立更强的层次结构，使视觉外观更容易实现。

我们将继续使用我们在上一个教程中创建的 GUI。

## 如何做...

在这里，我们将创建一个顶级框架，其中将包含其他框架和小部件。这将帮助我们将 GUI 布局调整到我们想要的样子。

为了做到这一点，我们将不得不将我们当前的控件嵌入到一个中央 ttk.LabelFrame 中。这个 ttk.LabelFrame 是主父窗口的子窗口，所有控件都是这个 ttk.LabelFrame 的子控件。

在我们的教程中到目前为止，我们已经直接将所有小部件分配给了我们的主 GUI 框架。现在我们将只将我们的 LabelFrame 分配给我们的主窗口，之后，我们将使这个 LabelFrame 成为所有小部件的父容器。

这在我们的 GUI 布局中创建了以下层次结构：

![操作步骤...](img/B04829_02_30.jpg)

在这个图表中，**win**是指我们的主 GUI tkinter 窗口框架的变量；**monty**是指我们的 LabelFrame 的变量，并且是主窗口框架（**win**）的子窗口；**aLabel**和所有其他小部件现在都放置在 LabelFrame 容器（**monty**）中。

在我们的 Python 模块顶部添加以下代码（参见注释＃1）：

```py
# Create instance
win = tk.Tk()

# Add a title       
win.title("Python GUI")    

# We are creating a container frame to hold all other widgets # 1
monty = ttk.LabelFrame(win, text=' Monty Python ')
monty.grid(column=0, row=0)
```

接下来，我们将修改所有以下控件，使用`monty`作为父控件，替换`win`。以下是如何做到这一点的示例：

```py
# Modify adding a Label
aLabel = ttk.Label(monty, text="A Label")
```

![如何做到...](img/B04829_02_04.jpg)

请注意，现在所有的小部件都包含在**Monty Python** LabelFrame 中，它用几乎看不见的细线将它们全部包围起来。接下来，我们可以重置**Labels in a Frame**小部件到左侧，而不会弄乱我们的 GUI 布局：

![如何做到...](img/B04829_02_04_1.jpg)

哎呀-也许不是。虽然我们在另一个框架中的框架很好地对齐到了左侧，但它又把我们的顶部小部件推到了中间（默认）。

为了将它们对齐到左侧，我们必须使用`sticky`属性来强制我们的 GUI 布局。通过将其分配为"W"（西），我们可以控制小部件左对齐。

```py
# Changing our Label
ttk.Label(monty, text="Enter a name:").grid(column=0, row=0, sticky='W')
```

![如何做到...](img/B04829_02_04_2.jpg)

## 它是如何工作的...

请注意我们对齐了标签，但没有对下面的文本框进行对齐。我们必须使用`sticky`属性来左对齐我们想要左对齐的所有控件。我们可以在一个循环中做到这一点，使用`winfo_children()`和`grid_configure(sticky='W')`属性，就像我们在本章的第 2 个配方中做的那样。

`winfo_children()`函数返回属于父控件的所有子控件的列表。这使我们能够循环遍历所有小部件并更改它们的属性。

### 注意

使用 tkinter 来强制左、右、上、下的命名与 Java 非常相似：west、east、north 和 south，缩写为："W"等等。我们还可以使用以下语法：tk.W 而不是"W"。

在以前的配方中，我们将"W"和"E"组合在一起，使我们的 ScrolledText 小部件使用"WE"附加到其容器的左侧和右侧。我们可以添加更多的组合："NSE"将使我们的小部件拉伸到顶部、底部和右侧。如果我们的表单中只有一个小部件，例如一个按钮，我们可以使用所有选项使其填满整个框架："NSWE"。我们还可以使用元组语法：`sticky=(tk.N, tk.S, tk.W, tk.E)`。

让我们把非常长的标签改回来，并将条目对齐到第 0 列的左侧。

```py
ttk.Label(monty, text="Enter a name:").grid(column=0, row=0, sticky='W')

name = tk.StringVar()
nameEntered = ttk.Entry(monty, width=12, textvariable=name)
nameEntered.grid(column=0, row=1, sticky=tk.W)
```

![它是如何工作的...](img/B04829_02_04_3.jpg)

### 注意

为了分离我们的**Labels in a Frame** LabelFrame 对我们的 GUI 布局的影响，我们不能将这个 LabelFrame 放入与其他小部件相同的 LabelFrame 中。相反，我们直接将它分配给主 GUI 表单（`win`）。

我们将在以后的章节中做到这一点。

# 创建菜单栏

在这个配方中，我们将向我们的主窗口添加一个菜单栏，向菜单栏添加菜单，然后向菜单添加菜单项。

## 准备工作

我们将首先学习如何添加菜单栏、几个菜单和一些菜单项的技巧，以展示如何做到这一点的原则。单击菜单项将不会产生任何效果。接下来，我们将为菜单项添加功能，例如，单击**Exit**菜单项时关闭主窗口，并显示**Help** | **About**对话框。

我们将继续扩展我们在当前和上一章中创建的 GUI。

## 如何做到...

首先，我们必须从`tkinter`中导入`Menu`类。在 Python 模块的顶部添加以下代码，即导入语句所在的地方： 

```py
from tkinter import Menu
```

接下来，我们将创建菜单栏。在模块的底部添加以下代码，就在我们创建主事件循环的地方上面：

```py
menuBar = Menu(win)                      # 1
win.config(menu=menuBar)
```

现在我们在菜单栏中添加一个菜单，并将一个菜单项分配给菜单。

```py
fileMenu = Menu(menuBar)                 # 2
fileMenu.add_command(label="New")
menuBar.add_cascade(label="File", menu=fileMenu)
```

运行此代码将添加一个菜单栏，其中有一个菜单，其中有一个菜单项。

![如何做到...](img/B04829_02_05.jpg)

接下来，我们在我们添加到菜单栏的第一个菜单中添加第二个菜单项。

```py
fileMenu.add_command(label="New")
fileMenu.add_command(label="Exit")        # 3
menuBar.add_cascade(label="File", menu=fileMenu)
```

![如何做到...](img/B04829_02_05_1.jpg)

我们可以通过在现有的 MenuItems 之间添加以下代码（＃4）来添加一个分隔线。

```py
fileMenu.add_command(label="New")
fileMenu.add_separator()               # 4
fileMenu.add_command(label="Exit")
```

![如何做到...](img/B04829_02_05_2.jpg)

通过将`tearoff`属性传递给菜单的构造函数，我们可以删除默认情况下出现在菜单中第一个 MenuItem 上方的第一条虚线。

```py
# Add menu items
fileMenu = Menu(menuBar, tearoff=0)      # 5
```

![如何做...](img/B04829_02_05_3.jpg)

我们将添加第二个菜单，它将水平放置在第一个菜单的右侧。我们将给它一个菜单项，我们将其命名为`关于`，为了使其工作，我们必须将这第二个菜单添加到菜单栏。

**文件**和**帮助** | **关于**是非常常见的 Windows GUI 布局，我们都很熟悉，我们可以使用 Python 和 tkinter 创建相同的菜单。

菜单的创建顺序和命名可能一开始有点令人困惑，但一旦我们习惯了 tkinter 要求我们如何编码，这实际上变得有趣起来。

```py
helpMenu = Menu(menuBar, tearoff=0)            # 6
helpMenu.add_command(label="About")
menuBar.add_cascade(label="Help", menu=helpMenu)
```

![如何做...](img/B04829_02_05_4.jpg)

此时，我们的 GUI 有一个菜单栏和两个包含一些菜单项的菜单。单击它们并没有太多作用，直到我们添加一些命令。这就是我们接下来要做的。在创建菜单栏之前添加以下代码：

```py
def _quit():         # 7
    win.quit()
    win.destroy()
    exit()
```

接下来，我们将**文件** | **退出**菜单项绑定到这个函数，方法是在菜单项中添加以下命令：

```py
fileMenu.add_command(label="Exit", command=_quit)    # 8
```

现在，当我们点击`退出`菜单项时，我们的应用程序确实会退出。

## 它是如何工作的...

在注释＃1 中，我们调用了菜单的`tkinter`构造函数，并将菜单分配给我们的主 GUI 窗口。我们在实例变量中保存了一个名为`menuBar`的引用，并在下一行代码中，我们使用这个实例来配置我们的 GUI，以使用`menuBar`作为我们的菜单。

注释＃2 显示了我们首先添加一个菜单项，然后创建一个菜单。这似乎有点不直观，但这就是 tkinter 的工作原理。`add_cascade()`方法将菜单项垂直布局在一起。

注释＃3 显示了如何向菜单添加第二个菜单项。

在注释＃4 中，我们在两个菜单项之间添加了一个分隔线。这通常用于将相关的菜单项分组并将它们与不太相关的项目分开（因此得名）。

注释＃5 禁用了虚线以使我们的菜单看起来更好。

### 注意

在不禁用此默认功能的情况下，用户可以从主窗口“撕下”菜单。我发现这种功能价值不大。随意双击虚线（在禁用此功能之前）进行尝试。

如果您使用的是 Mac，这个功能可能没有启用，所以您根本不用担心。

![它是如何工作的...](img/B04829_02_05_5.jpg)

注释＃6 向您展示了如何向菜单栏添加第二个菜单。我们可以继续使用这种技术添加菜单。

注释＃7 创建了一个函数来干净地退出我们的 GUI 应用程序。这是结束主事件循环的推荐 Pythonic 方式。

在＃8 中，我们将在＃7 中创建的函数绑定到菜单项，使用`tkinter`命令属性。每当我们想要我们的菜单项实际执行某些操作时，我们必须将它们中的每一个绑定到一个函数。

### 注意

我们使用了推荐的 Python 命名约定，通过在退出函数之前加上一个下划线，以表示这是一个私有函数，不应该由我们代码的客户端调用。

## 还有更多...

在下一章中，我们将添加**帮助** | **关于**功能，介绍消息框等等。

# 创建选项卡小部件

在这个配方中，我们将创建选项卡小部件，以进一步组织我们在 tkinter 中编写的扩展 GUI。

## 准备工作

为了改进我们的 Python GUI，我们将从头开始，使用最少量的代码。在接下来的配方中，我们将从以前的配方中添加小部件，并将它们放入这个新的选项卡布局中。

## 如何做...

创建一个新的 Python 模块，并将以下代码放入该模块：

```py
import tkinter as tk                    # imports
from tkinter import ttk
win = tk.Tk()                           # Create instance      
win.title("Python GUI")                 # Add a title 
tabControl = ttk.Notebook(win)          # Create Tab Control
tab1 = ttk.Frame(tabControl)            # Create a tab 
tabControl.add(tab1, text='Tab 1')      # Add the tab
tabControl.pack(expand=1, fill="both")  # Pack to make visible
win.mainloop()                          # Start GUI
```

这创建了以下 GUI：

![如何做...](img/B04829_02_06.jpg)

尽管目前还不是非常令人印象深刻，但这个小部件为我们的 GUI 设计工具包增加了另一个非常强大的工具。它在上面的极简示例中有自己的限制（例如，我们无法重新定位 GUI，也不显示整个 GUI 标题）。

在以前的示例中，我们使用网格布局管理器来创建更简单的 GUI，我们可以使用更简单的布局管理器之一，“pack”是其中之一。

在上述代码中，我们将 tabControl ttk.Notebook“pack”到主 GUI 表单中，扩展选项卡控件以填充所有边缘。

![如何做...](img/B04829_02_06_0.jpg)

我们可以向我们的控件添加第二个选项卡并在它们之间切换。

```py
tab2 = ttk.Frame(tabControl)            # Add a second tab
tabControl.add(tab2, text='Tab 2')      # Make second tab visible
win.mainloop()                          # Start GUI
```

现在我们有两个标签。单击**Tab 2**以使其获得焦点。

![如何做...](img/B04829_02_06_1.jpg)

我们真的很想看到我们的窗口标题。因此，为了做到这一点，我们必须向我们的选项卡中添加一个小部件。该小部件必须足够宽，以动态扩展我们的 GUI 以显示我们的窗口标题。我们正在将 Ole Monty 和他的孩子们重新添加。

```py
monty = ttk.LabelFrame(tab1, text=' Monty Python ')
monty.grid(column=0, row=0, padx=8, pady=4)
ttk.Label(monty, text="Enter a name:").grid(column=0, row=0, sticky='W')
```

现在我们在**Tab1**中有我们的**Monty Python**。

![如何做...](img/B04829_02_06_2.jpg)

我们可以继续将到目前为止创建的所有小部件放入我们新创建的选项卡控件中。

![如何做...](img/B04829_02_06_3.jpg)

现在所有的小部件都驻留在**Tab1**中。让我们将一些移动到**Tab2**。首先，我们创建第二个 LabelFrame，作为我们将移动到**Tab2**的小部件的容器：

```py
monty2 = ttk.LabelFrame(tab2, text=' The Snake ')
monty2.grid(column=0, row=0, padx=8, pady=4)
```

接下来，我们通过指定新的父容器`monty2`，将复选框和单选按钮移动到**Tab2**。以下是一个示例，我们将其应用于所有移动到**Tab2**的控件：

```py
chVarDis = tk.IntVar()
check1 = tk.Checkbutton(monty2, text="Disabled", variable=chVarDis, state='disabled')
```

当我们运行代码时，我们的 GUI 现在看起来不同了。**Tab1**的小部件比以前少了，当它包含我们以前创建的所有小部件时。

![如何做...](img/B04829_02_06_4.jpg)

现在我们可以单击**Tab 2**并查看我们移动的小部件。

![如何做...](img/B04829_02_06_5.jpg)

单击移动的 Radiobutton(s)不再产生任何效果，因此我们将更改它们的操作以重命名文本属性，这是 LabelFrame 小部件的标题，以显示 Radiobuttons 的名称。当我们单击**Gold** Radiobutton 时，我们不再将框架的背景设置为金色，而是在这里替换 LabelFrame 文本标题。Python“ The Snake”现在变成“Gold”。

```py
# Radiobutton callback function
def radCall():
    radSel=radVar.get()
    if   radSel == 0: monty2.configure(text='Blue')
    elif radSel == 1: monty2.configure(text='Gold')
    elif radSel == 2: monty2.configure(text='Red')
```

现在，选择任何 RadioButton 小部件都会导致更改 LabelFrame 的名称。

![如何做...](img/B04829_02_06_6.jpg)

## 它是如何工作的...

创建第二个选项卡后，我们将一些最初驻留在**Tab1**中的小部件移动到**Tab2**。添加选项卡是组织我们不断增加的 GUI 的另一种绝佳方式。这是处理 GUI 设计中复杂性的一种非常好的方式。我们可以将小部件分组放置在它们自然属于的组中，并通过使用选项卡使我们的用户摆脱混乱。

### 注意

在`tkinter`中，通过`Notebook`小部件创建选项卡是通过`Notebook`小部件完成的，这是允许我们添加选项卡控件的工具。 tkinter 笔记本小部件，就像许多其他小部件一样，具有我们可以使用和配置的附加属性。探索我们可以使用的 tkinter 小部件的其他功能的绝佳起点是官方网站：[`docs.python.org/3.1/library/tkinter.ttk.html#notebook`](https://docs.python.org/3.1/library/tkinter.ttk.html#notebook)

# 使用网格布局管理器

网格布局管理器是我们可以使用的最有用的布局工具之一。我们已经在许多示例中使用了它，因为它非常强大。

## 准备工作...

在这个示例中，我们将回顾一些网格布局管理器的技术。我们已经使用过它们，在这里我们将进一步探讨它们。

## 如何做...

在本章中，我们已经创建了行和列，这实际上是 GUI 设计的数据库方法（MS Excel 也是如此）。我们硬编码了前四行，但然后忘记了给下一行一个我们希望它驻留的位置的规范。

Tkinter 在我们不知不觉中为我们填充了这个。

以下是我们在代码中所做的：

```py
check3.grid(column=2, row=4, sticky=tk.W, columnspan=3)
scr.grid(column=0, sticky='WE', columnspan=3)              # 1
curRad.grid(column=col, row=6, sticky=tk.W, columnspan=3)
labelsFrame.grid(column=0, row=7)
```

Tkinter 自动添加了我们没有指定任何特定行的缺失行（在注释＃1 中强调）。我们可能没有意识到这一点。

我们将复选框布置在第 4 行，然后“忘记”为我们的 ScrolledText 小部件指定行，我们通过 scr 变量引用它，然后我们添加了要布置在第 6 行的 Radiobutton 小部件。

这很好用，因为 tkinter 自动递增了我们的 ScrolledText 小部件的行位置，以使用下一个最高的行号，即第 5 行。

查看我们的代码，没有意识到我们“忘记”将我们的 ScrolledText 小部件明确定位到第 5 行，我们可能会认为那里什么都没有。

因此，我们可以尝试以下操作。

如果我们将变量`curRad`设置为使用第 5 行，我们可能会得到一个不愉快的惊喜：

![如何做...](img/B04829_02_07.jpg)

## 它是如何工作的...

注意我们的 RadioButton(s)行突然出现在我们的 ScrolledText 小部件的中间！这绝对不是我们想要的 GUI 样式！

### 注意

如果我们忘记显式指定行号，默认情况下，`tkinter`将使用下一个可用的行。

我们还使用了`columnspan`属性来确保我们的小部件不会被限制在一列。以下是我们如何确保我们的 ScrolledText 小部件跨越 GUI 的所有列：

```py
# Using a scrolled Text control    
scrolW = 30; scrolH = 3
scr = ScrolledText(monty, width=scrolW, height=scrolH, wrap=tk.WORD)
scr.grid(column=0, sticky='WE', columnspan=3)
```
