# 第5章。Matplotlib图表

在这一章中，我们将使用Python 3和Matplotlib模块创建美丽的图表。

+   使用Matplotlib创建美丽的图表

+   Matplotlib - 使用pip下载模块

+   Matplotlib - 使用whl扩展名下载模块

+   创建我们的第一个图表

+   在图表上放置标签

+   如何给图表加上图例

+   调整图表的比例

+   动态调整图表的比例

# 介绍

在本章中，我们将创建美丽的图表，以直观地表示数据。根据数据源的格式，我们可以在同一图表中绘制一个或多个数据列。

我们将使用Python Matplotlib模块来创建我们的图表。

为了创建这些图形图表，我们需要下载额外的Python模块，有几种安装方法。

本章将解释如何下载Matplotlib Python模块，所有其他所需的Python模块，以及如何做到这一点的方法。

在安装所需的模块之后，我们将创建自己的Python图表。

# 使用Matplotlib创建美丽的图表

这个示例向我们介绍了Matplotlib Python模块，它使我们能够使用Python 3创建可视化图表。

以下URL是开始探索Matplotlib世界的好地方，并将教您如何创建本章中未提及的许多图表：

[http://matplotlib.org/users/screenshots.html](http://matplotlib.org/users/screenshots.html)

## 准备工作

为了使用Matplotlib Python模块，我们首先必须安装该模块，以及诸如numpy等其他相关的Python模块。

如果您使用的Python版本低于3.4.3，我建议您升级Python版本，因为在本章中我们将使用Python pip模块来安装所需的Python模块，而pip是在3.4.3及以上版本中安装的。

### 注意

可以使用较早版本的Python 3安装pip，但这个过程并不是很直观，因此最好升级到3.4.3或更高版本。

## 如何做...

以下图片是使用Python和Matplotlib模块创建的令人难以置信的图表的示例。

我从[http://matplotlib.org/](http://matplotlib.org/)网站复制了以下代码，它创建了这个令人难以置信的图表。该网站上有许多示例，我鼓励您尝试它们，直到找到您喜欢创建的图表类型。

以下是创建图表的代码，包括空格在内，不到25行的Python代码。

```py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
```

运行代码会创建以下图片中显示的图表：

![如何做...](graphics/B04829_05_01.jpg)

使用Python 3.4或更高版本与Eclipse PyDev插件运行代码可能会显示一些未解决的导入错误。这似乎是PyDev或Java中的一个错误。

如果您使用Eclipse进行开发，请忽略这些错误，因为代码将成功运行。

## 它是如何工作的...

为了创建如前面截图所示的美丽图表，我们需要下载其他几个Python模块。

以下示例将指导我们如何成功下载所有所需的模块，从而使我们能够创建自己的美丽图表。

# Matplotlib - 使用pip下载模块

下载额外的Python模块的常规方法是使用pip。pip模块预装在最新版本的Python（3.4及以上）中。

### 注意

如果您使用的是较旧版本的Python，可能需要自己下载pip和setuptools。

除了使用Python安装程序外，还有其他几个预编译的Windows可执行文件，可以让我们轻松安装Matplotlib等Python模块。

这个示例将展示如何通过Windows可执行文件成功安装Matplotlib，以及使用pip安装Matplotlib库所需的其他模块。

## 准备工作

我们所需要做的就是在我们的PC上安装一个Python 3.4（或更高版本）的发行版，以便下载所需的Python模块来使用Matplotlib模块。

## 如何做...

我们可以通过官方Matplotlib网站上的Windows可执行文件来安装Matplotlib。

确保安装与您正在使用的Python版本匹配的Matplotlib版本。例如，如果您在64位操作系统（如Microsoft Windows 7）上安装了Python 3.4，则下载并安装`Matplotlib-1.4.3.win-amd64-py3.4.exe`。

### 注意

可执行文件名称中的"amd64"表示您正在安装64位版本。如果您使用32位x86系统，则安装amd64将不起作用。如果您安装了32位版本的Python并下载了64位Python模块，则可能会出现类似的问题。

![如何做...](graphics/B04829_05_02.jpg)

运行可执行文件将启动我们，并且看起来像这样：

![如何做...](graphics/B04829_05_03.jpg)

我们可以通过查看我们的Python安装目录来验证我们是否成功安装了Matplotlib。

安装成功后，Matplotlib文件夹将添加到site-packages。在Windows上使用默认安装，site-packages文件夹的完整路径是：

`C:\Python34\Lib\site-packages\matplotlib\`

![如何做...](graphics/B04829_05_04.jpg)

在官方Matplotlib网站上最简单的绘图示例需要使用Python numpy模块，所以让我们下载并安装这个模块。

### 注意

Numpy是一个数学模块，它使Matplotlib图表的绘制成为可能，但远不止于Matplotlib。如果您正在开发的软件需要大量的数学计算，您肯定会想要查看numpy。

有一个优秀的网站，为我们提供了几乎所有Python模块的快速链接。它作为一个很好的时间节省者，指出了成功使用Matplotlib所需的其他Python模块，并给我们提供了下载这些模块的超链接，这使我们能够快速轻松地安装它们。

### 注意

这是链接：

[http://www.lfd.uci.edu/~gohlke/pythonlibs/](http://www.lfd.uci.edu/~gohlke/pythonlibs/)

![如何做...](graphics/B04829_05_05.jpg)

注意安装程序包的文件扩展名都以whl结尾。为了使用它们，我们必须安装Python wheel模块，我们使用pip来做到这一点。

### 注意

Wheels是Python分发的新标准，旨在取代eggs。

您可以在以下网站找到更多详细信息：

[http://pythonwheels.com/](http://pythonwheels.com/)

最好以管理员身份运行Windows命令处理器，以避免潜在的安装错误。

![如何做...](graphics/B04829_05_06.jpg)

## 它是如何工作的...

下载Python模块的常见方法是使用pip，就像上面所示的那样。为了安装Matplotlib所需的所有模块，我们可以从主网站下载它们的下载格式已更改为使用whl格式。

下一个配方将解释如何使用wheel安装Python模块。

# Matplotlib - 使用whl扩展名下载模块

我们将使用几个Matplotlib需要的额外Python模块，在这个配方中，我们将使用Python的新模块分发标准wheel来下载它们。

### 注意

您可以在以下网址找到新的wheel标准的Python增强提案（PEP）：[https://www.python.org/dev/peps/pep-0427/](https://www.python.org/dev/peps/pep-0427/)

## 准备工作

为了下载带有whl扩展名的Python模块，必须首先安装Python wheel模块，这在前面的配方中已经解释过了。

## 如何做...

让我们从网上下载`numpy-1.9.2+mkl-cp34-none-win_amd64.whl`。安装了wheel模块后，我们可以使用pip来安装带有whl文件扩展名的软件包。

### 注意

Pip随Python 3.4.3及以上版本一起提供。如果您使用的是较旧版本的Python，我建议安装pip，因为它可以让安装所有其他额外的Python模块变得更加容易。

一个更好的建议可能是将您的Python版本升级到最新的稳定版本。当您阅读本书时，最有可能的是Python 3.5.0或更高版本。

Python是免费软件。升级对我们来说是没有成本的。

浏览到要安装的软件包所在的文件夹，并使用以下命令进行安装：

```py
**pip install numpy-1.9.2+mkl-cp34-none-win_amd64.whl**

```

![如何做...](graphics/B04829_05_07.jpg)

现在我们可以使用官方网站上最简单的示例应用程序创建我们的第一个Matplotlib图表。之后，我们将创建自己的图表。

![如何做...](graphics/B04829_05_08.jpg)

我们还没有准备好运行前面的代码，这表明我们需要下载更多的模块。虽然一开始需要下载更多的模块可能会有点烦人，但实际上这是一种代码重用的形式。

因此，让我们使用pip和wheel下载并安装six和所有其他所需的模块（如dateutil、pyparsing等），直到我们的代码能够工作并从只有几行Python代码中创建一个漂亮的图表。

我们可以从刚刚用来安装numpy的同一个网站下载所有所需的模块。这个网站甚至列出了我们正在安装的模块所依赖的所有其他模块，并提供了跳转到这个网站上的安装软件的超链接。

### 注意

如前所述，安装Python模块的URL是：[http://www.lfd.uci.edu/~gohlke/pythonlibs/](http://www.lfd.uci.edu/~gohlke/pythonlibs/)

## 它是如何工作的...

使我们能够从一个便利的地方下载许多Python模块的网站还提供其他Python模块。并非所有显示的依赖项都是必需的。这取决于您正在开发的内容。随着您使用Matplotlib库的旅程的推进，您可能需要下载和安装其他模块。

![它是如何工作的...](graphics/B04829_05_09.jpg)

# 创建我们的第一个图表

现在我们已经安装了所有所需的Python模块，我们可以使用Matplotlib创建自己的图表。

我们可以只用几行Python代码创建图表。

## 准备工作

使用前一个示例中的代码，我们现在可以创建一个看起来类似于下一个示例的图表。

## 如何做...

使用官方网站上提供的最少量的代码，我们可以创建我们的第一个图表。嗯，几乎。网站上显示的示例代码在导入`show`方法并调用它之前是无法工作的。

![如何做...](graphics/B04829_05_10.jpg)

我们可以简化代码，甚至通过使用官方Matplotlib网站提供的许多示例之一来改进它。

![如何做...](graphics/B04829_05_11.jpg)

## 它是如何工作的...

Python Matplotlib模块，结合诸如numpy之类的附加组件，创建了一个非常丰富的编程环境，使我们能够轻松进行数学计算并在可视化图表中绘制它们。

Python numpy方法`arange`并不打算安排任何事情。它的意思是创建“一个范围”，在Python中用于内置的“range”运算符。`linspace`方法可能会造成类似的混淆。谁是“lin”，在什么“空间”？

事实证明，该名称意味着“线性间隔向量”。

pyglet函数`show`显示我们创建的图形。在成功创建第一个图形后，调用`show()`会产生一些副作用，当您尝试绘制另一个图形时。

# 在图表上放置标签

到目前为止，我们已经使用了默认的Matplotlib GUI。现在我们将使用Matplotlib创建一些tkinter GUI。

这将需要更多的Python代码行和导入更多的库，但这是值得的，因为我们正在通过画布控制我们的绘画。

我们将标签放在水平轴和垂直轴上，也就是*x*和*y*。

我们将通过创建一个Matplotlib图形来实现这一点。

我们还将学习如何使用子图，这将使我们能够在同一个窗口中绘制多个图形。

## 准备工作

安装必要的Python模块并知道在哪里找到官方在线文档和教程后，我们现在可以继续创建Matplotlib图表。

## 如何做...

虽然`plot`是创建Matplotlib图表的最简单方法，但是结合`Canvas`使用`Figure`创建一个更定制的图表，看起来更好，还可以让我们向其添加按钮和其他小部件。

```py
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
#--------------------------------------------------------------
fig = Figure(figsize=(12, 8), facecolor='white')
#--------------------------------------------------------------
# axis = fig.add_subplot(111)   # 1 row,  1 column, only graph
axis = fig.add_subplot(211)     # 2 rows, 1 column, Top graph
#--------------------------------------------------------------
xValues = [1,2,3,4]
yValues = [5,7,6,8]
axis.plot(xValues, yValues)

axis.set_xlabel('Horizontal Label')
axis.set_ylabel('Vertical Label')

# axis.grid()                   # default line style 
axis.grid(linestyle='-')        # solid grid lines
#--------------------------------------------------------------
def _destroyWindow():
    root.quit()
    root.destroy() 
#--------------------------------------------------------------
root = tk.Tk() 
root.withdraw()
root.protocol('WM_DELETE_WINDOW', _destroyWindow)   
#--------------------------------------------------------------
canvas = FigureCanvasTkAgg(fig, master=root)
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
#--------------------------------------------------------------
root.update()
root.deiconify()
root.mainloop()
```

运行上述代码会得到以下图表：

![如何做...](graphics/B04829_05_12.jpg)

在导入语句之后的第一行代码中，我们创建了一个`Figure`对象的实例。接下来，我们通过调用`add_subplot(211)`向这个图添加子图。211中的第一个数字告诉图要添加多少个图，第二个数字确定列数，第三个数字告诉图以什么顺序显示图。

我们还添加了一个网格并更改了其默认线型。

尽管我们在图表中只显示一个图，但通过选择2作为子图的数量，我们将图向上移动，这导致图表底部出现额外的空白。这第一个图现在只占据屏幕的50％，这会影响在显示时此图的网格线有多大。

### 注意

通过取消注释`axis =`和`axis.grid()`的代码来尝试该代码，以查看不同的效果。

我们可以通过将它们分配到第二个位置使用`add_subplot(212)`来添加更多的子图。

```py
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
#--------------------------------------------------------------
fig = Figure(figsize=(12, 8), facecolor='white')
#--------------------------------------------------------------
axis = fig.add_subplot(211)     # 2 rows, 1 column, Top graph
#--------------------------------------------------------------
xValues = [1,2,3,4]
yValues = [5,7,6,8]
axis.plot(xValues, yValues)

axis.set_xlabel('Horizontal Label')
axis.set_ylabel('Vertical Label')

axis.grid(linestyle='-')        # solid grid lines
#--------------------------------------------------------------
axis1 = fig.add_subplot(212)    # 2 rows, 1 column, Bottom graph
#--------------------------------------------------------------
xValues1 = [1,2,3,4]
yValues1 = [7,5,8,6]
axis1.plot(xValues1, yValues1)
axis1.grid()                    # default line style 
#--------------------------------------------------------------
def _destroyWindow():
    root.quit()
    root.destroy() 
#--------------------------------------------------------------
root = tk.Tk() 
root.withdraw()
root.protocol('WM_DELETE_WINDOW', _destroyWindow)   
#--------------------------------------------------------------
canvas = FigureCanvasTkAgg(fig, master=root)
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
#--------------------------------------------------------------
root.update()
root.deiconify()
root.mainloop()
```

现在运行略微修改的代码会将axis1添加到图表中。对于底部图的网格，我们将线型保留为默认值。

![如何做...](graphics/B04829_05_13.jpg)

## 工作原理...

我们导入了必要的Matplotlib模块来创建一个图和一个画布，用于在其上绘制图表。我们为*x*和*y*轴给出了一些值，并设置了很多配置选项中的一些。

我们创建了自己的tkinter窗口来显示图表并自定义了绘图的位置。

正如我们在前几章中看到的，为了创建一个tkinter GUI，我们首先必须导入tkinter模块，然后创建`Tk`类的实例。我们将这个类实例分配给一个我们命名为`root`的变量，这是在示例中经常使用的名称。

我们的tkinter GUI直到我们启动主事件循环才会变得可见，为此，我们使用`root.mainloop()`。

避免在这里使用Matplotlib默认GUI并改为使用tkinter创建自己的GUI的一个重要原因是，我们想要改善默认Matplotlib GUI的外观，而使用tkinter可以很容易地实现这一点。

如果我们使用tkinter构建GUI，就不会再出现那些过时的按钮出现在Matplotlib GUI底部。

同时，Matplotlib GUI具有我们的tkinter GUI没有的功能，即当我们在图表内移动鼠标时，我们实际上可以看到Matplotlib GUI中的x和y坐标。 x和y坐标位置显示在右下角。

# 如何给图表添加图例

一旦我们开始绘制多条数据点的线，事情可能会变得有点不清楚。通过向我们的图表添加图例，我们可以知道哪些数据是什么，它们实际代表什么。

我们不必选择不同的颜色来表示不同的数据。Matplotlib会自动为每条数据点的线分配不同的颜色。

我们所要做的就是创建图表并向其添加图例。

## 准备工作

在这个示例中，我们将增强上一个示例中的图表。我们只会绘制一个图表。

## 如何做...

首先，我们将在同一图表中绘制更多的数据线，然后我们将向图表添加图例。

我们通过修改上一个示例中的代码来实现这一点。

```py
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
#--------------------------------------------------------------
fig = Figure(figsize=(12, 5), facecolor='white')
#--------------------------------------------------------------
axis  = fig.add_subplot(111)                  # 1 row, 1 column

xValues  = [1,2,3,4]

yValues0 = [6,7.5,8,7.5]
yValues1 = [5.5,6.5,8,6]
yValues2 = [6.5,7,8,7]

t0, = axis.plot(xValues, yValues0)
t1, = axis.plot(xValues, yValues1)
t2, = axis.plot(xValues, yValues2)

axis.set_ylabel('Vertical Label')
axis.set_xlabel('Horizontal Label')

axis.grid()

fig.legend((t0, t1, t2), ('First line', 'Second line', 'Third line'), 'upper right')

#--------------------------------------------------------------
def _destroyWindow():
    root.quit()
    root.destroy() 
#--------------------------------------------------------------
root = tk.Tk() 
root.withdraw()
root.protocol('WM_DELETE_WINDOW', _destroyWindow)
#--------------------------------------------------------------
canvas = FigureCanvasTkAgg(fig, master=root)
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
#--------------------------------------------------------------
root.update()
root.deiconify()
root.mainloop()
```

运行修改后的代码会创建以下图表，图例位于右上角：

![如何做...](graphics/B04829_05_14.jpg)

在这个示例中，我们只绘制了一个图表，我们通过更改`fig.add_subplot(111)`来实现这一点。我们还通过`figsize`属性略微修改了图表的大小。

接下来，我们创建了三个包含要绘制的值的Python列表。当我们绘制数据时，我们将图表的引用保存在本地变量中。

我们通过传入一个包含三个图表引用的元组，另一个包含随后在图例中显示的字符串的元组来创建图例，并在第三个参数中定位图例在图表中的位置。

Matplotlib的默认设置为正在绘制的线条分配了一个颜色方案。

我们可以通过在绘制每个轴时设置属性来轻松地将这些默认颜色设置更改为我们喜欢的颜色。

我们通过使用颜色属性并为其分配一个可用的颜色值来实现这一点。

```py
t0, = axis.plot(xValues, yValues0, color = 'purple')
t1, = axis.plot(xValues, yValues1, color = 'red')
t2, = axis.plot(xValues, yValues2, color = 'blue')
```

请注意，t0、t1和t2的变量赋值后面的逗号不是错误，而是为了创建图例而需要的。

在每个变量后面的逗号将列表转换为元组。如果我们省略这一点，我们的图例将不会显示。

代码仍将运行，只是没有预期的图例。

### 注意

当我们在t0 =赋值后移除逗号时，我们会得到一个错误，第一行不再出现在图中。图表和图例仍然会被创建，但图例中不再出现第一行。

![如何做...](graphics/B04829_05_15.jpg)

## 它是如何工作的...

我们通过在同一图表中绘制三条数据线并为其添加图例来增强了我们的图表，以区分这三条线绘制的数据。

# 调整图表的比例

在以前的示例中，当我们创建我们的第一个图表并增强它们时，我们硬编码了这些值的视觉表示方式。

虽然这对我们使用的值很有帮助，但我们经常从非常大的数据库中绘制图表。

根据数据的范围，我们为垂直y维度的硬编码值可能并不总是最佳解决方案，这可能会使我们的图表中的线条难以看清。

## 准备工作

我们将改进我们在上一个示例中的代码。如果您没有输入所有以前示例中的代码，只需下载本章的代码，它将让您开始（然后您可以通过使用Python创建GUI、图表等来玩得很开心）。

## 如何做...

将上一个示例中的`yValues1`代码行修改为使用50作为第三个值。

```py
axis  = fig.add_subplot(111)        # 1 row, 1 column

xValues  = [1,2,3,4]

yValues0 = [6,7.5,8,7.5]
yValues1 = [5.5,6.5,50,6]           # one very high value
yValues2 = [6.5,7,8,7]
```

与上一个示例中创建图表的代码唯一的区别是一个数据值。

通过更改一个与所有其他值的平均范围不接近的值，数据的视觉表示已经发生了戏剧性的变化，我们失去了关于整体数据的许多细节，现在主要看到一个高峰。

![如何做...](graphics/B04829_05_16.jpg)

到目前为止，我们的图表已根据它们所呈现的数据自动调整。

虽然这是Matplotlib的一个实用功能，但这并不总是我们想要的。我们可以通过限制垂直y维度来限制图表的比例。

```py
yValues0 = [6,7.5,8,7.5]
yValues1 = [5.5,6.5,50,6]           # one very high value (50)
yValues2 = [6.5,7,8,7]

axis.set_ylim(5, 8)                 # limit the vertical display
```

现在，`axis.set_ylim(5, 8)`这行代码限制了起始值为5，垂直显示的结束值为8。

现在，当我们创建图表时，高值峰值不再像以前那样有影响。

![如何做...](graphics/B04829_05_17.jpg)

## 它是如何工作的...

我们增加了数据中的一个值，这产生了戏剧性的效果。通过设置图表的垂直和水平显示限制，我们可以看到我们最感兴趣的数据。

像刚才显示的那样的尖峰也可能非常有趣。这一切取决于我们要寻找什么。数据的视觉表示具有很大的价值。

### 注意

一图胜千言。

# 动态调整图表的比例

在上一个示例中，我们学习了如何限制我们图表的缩放。在这个示例中，我们将进一步通过在表示数据之前动态调整缩放来设置限制并分析我们的数据。

## 准备工作

我们将通过动态读取数据、对其进行平均并调整我们的图表来增强上一个示例中的代码。

虽然我们通常会从外部来源读取数据，在这个示例中，我们使用Python列表创建我们要绘制的数据，如下面的代码所示。

## 如何做...

我们通过将数据分配给xvalues和yvalues变量来在我们的Python模块中创建自己的数据。

在许多图表中，x和y坐标系的起始点通常是(0, 0)。这通常是一个好主意，所以让我们相应地调整我们的图表坐标代码。

让我们修改代码以限制x和y两个维度：

```py
xValues  = [1,2,3,4]

yValues0 = [6,7.5,8,7.5]
yValues1 = [5.5,6.5,50,6]           # one very high value (50)
yValues2 = [6.5,7,8,7]              

axis.set_ylim(0, 8)                 # lower limit (0)
axis.set_xlim(0, 8)                 # use same limits for x
```

现在我们已经为x和y设置了相同的限制，我们的图表可能看起来更加平衡。当我们运行修改后的代码时，我们得到了以下结果：

![如何做...](graphics/B04829_05_18.jpg)

也许从(0, 0)开始并不是一个好主意...

我们真正想做的是根据数据的范围动态调整我们的图表，同时限制过高或过低的值。

我们可以通过解析要在图表中表示的所有数据，同时设置一些明确的限制来实现这一点。

修改代码如下所示：

```py
xValues  = [1,2,3,4]

yValues0 = [6,7.5,8,7.5]
yValues1 = [5.5,6.5,50,6]              # one very high value (50)
yValues2 = [6.5,7,8,7]              
yAll = [yValues0, yValues1, yValues2]  # list of lists

# flatten list of lists retrieving minimum value
minY = min([y for yValues in yAll for y in yValues])

yUpperLimit = 20
# flatten list of lists retrieving max value within defined limit
maxY = max([y for yValues in yAll for y in yValues if y < yUpperLimit])

# dynamic limits
axis.set_ylim(minY, maxY)                 
axis.set_xlim(min(xValues), max(xValues))                

t0, = axis.plot(xValues, yValues0)
t1, = axis.plot(xValues, yValues1)
t2, = axis.plot(xValues, yValues2)
```

运行代码会得到以下图表。我们动态调整了它的x和y维度。请注意，现在y维度从5.5开始，而不是之前的5.0。图表也不再从(0, 0)开始，这为我们提供了更有价值的关于我们的数据的信息。

![如何做...](graphics/B04829_05_19.jpg)

我们正在为y维度数据创建一个列表的列表，然后使用一个列表推导包装成对Python的`min()`和`max()`函数的调用。

如果列表推导似乎有点高级，它们基本上是一个非常压缩的循环。

它们还被设计为比常规编程循环更快。

在创建上述图表的Python代码中，我们创建了三个包含要绘制的y维度数据的列表。然后我们创建了另一个包含这三个列表的列表，从而创建了一个列表的列表。

就像这样：

```py
yValues0 = [6,7.5,8,7.5]
yValues1 = [5.5,6.5,50,6]              # one very high value (50)
yValues2 = [6.5,7,8,7]              
yAll = [yValues0, yValues1, yValues2]  # list of lists
```

我们对获取所有y维度数据的最小值以及包含在这三个列表中的最大值感兴趣。

我们可以通过Python列表推导来实现这一点。

```py
# flatten list of lists retrieving minimum value
minY = min([y for yValues in yAll for y in yValues])
```

在运行列表推导后，`minY`为5.5。

上面的一行代码是列表推导，它遍历三个列表中包含的所有数据的所有值，并使用Python的`min`关键字找到最小值。

在同样的模式中，我们找到了我们希望绘制的数据中包含的最大值。这次，我们还在列表推导中设置了一个限制，忽略了所有超过我们指定限制的值，就像这样：

```py
yUpperLimit = 20
# flatten list of lists retrieving max value within defined limit
maxY = max([y for yValues in yAll for y in yValues if y < yUpperLimit])
```

在使用我们选择的限制条件运行上述代码后，`maxY`的值为8（而不是50）。

我们根据预定义条件选择20作为图表中显示的最大值，对最大值应用了限制。

对于x维度，我们只需在Matplotlib方法中调用`min()`和`max()`来动态调整图表的限制。

## 工作原理...

在这个示例中，我们创建了几个Matplotlib图表，并调整了其中一些可用属性。我们还使用核心Python动态控制了图表的缩放。
