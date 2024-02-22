# 第九章。使用 wxPython 库扩展我们的 GUI

在本章中，我们将使用 wxPython 库增强我们的 Python GUI。

+   如何安装 wxPython 库

+   如何在 wxPython 中创建我们的 GUI

+   使用 wxPython 快速添加控件

+   尝试在主 tkinter 应用程序中嵌入主 wxPython 应用程序

+   尝试将我们的 tkinter GUI 代码嵌入到 wxPython 中

+   如何使用 Python 控制两个不同的 GUI 框架

+   如何在两个连接的 GUI 之间通信

# 介绍

在本章中，我们将介绍另一个 Python GUI 工具包，它目前不随 Python 一起发布。它被称为 wxPython。

这个库有两个版本。原始版本称为 Classic，而最新版本称为开发项目的代号 Phoenix。

在本书中，我们仅使用 Python 3 进行编程，因为新的 Phoenix 项目旨在支持 Python 3，这就是我们在本章中使用的 wxPython 版本。

首先，我们将创建一个简单的 wxPython GUI，然后我们将尝试将我们在本书中开发的基于 tkinter 的 GUI 与新的 wxPython 库连接起来。

### 注意

wxPython 是 Python 绑定到 wxWidgets 的库。

wxPython 中的 w 代表 Windows 操作系统，x 代表 Unix 操作系统，如 Linux 和 OS X。

如果同时使用这两个 GUI 工具包出现问题，我们将尝试使用 Python 解决任何问题，然后我们将使用 Python 内的**进程间通信**（**IPC**）来确保我们的 Python 代码按我们希望的方式工作。

# 如何安装 wxPython 库

wxPython 库不随 Python 一起发布，因此，为了使用它，我们首先必须安装它。

这个步骤将向我们展示在哪里以及如何找到正确的版本来安装，以匹配已安装的 Python 版本和正在运行的操作系统。

### 注意

wxPython 第三方库已经存在了 17 年多，这表明它是一个强大的库。

## 准备工作

为了在 Python 3 中使用 wxPython，我们必须安装 wxPython Phoenix 版本。

## 如何做...

在网上搜索 wxPython 时，我们可能会在[www.wxpython.org](http://www.wxpython.org)找到官方网站。

![如何做...](img/B04829_09_01.jpg)

如果我们点击 MS Windows 的下载链接，我们可以看到几个 Windows 安装程序，所有这些安装程序都仅适用于 Python 2.x。

![如何做...](img/B04829_09_02.jpg)

使用 Python 3 和 wxPython，我们必须安装 wxPython/Phoenix 库。我们可以在快照构建链接中找到安装程序：

[`wxpython.org/Phoenix/snapshot-builds/`](http://wxpython.org/Phoenix/snapshot-builds/)

从这里，我们可以选择与我们的 Python 版本和操作系统版本匹配的 wxPython/Phoenix 版本。我正在使用运行在 64 位 Windows 7 操作系统上的 Python 3.4。

![如何做...](img/B04829_09_03.jpg)

Python wheel（.whl）安装程序包有一个编号方案。

对我们来说，这个方案最重要的部分是我们正在安装的 wxPython/Phoenix 版本是为 Python 3.4（安装程序名称中的 cp34）和 Windows 64 位操作系统（安装程序名称中的 win_amd64）。

![如何做...](img/B04829_09_04.jpg)

成功下载 wxPython/Phoenix 包后，我们现在可以转到该包所在的目录，并使用 pip 安装此包。

![如何做...](img/B04829_09_05.jpg)

我们的 Python`site-packages`文件夹中有一个名为`wx`的新文件夹。

![如何做...](img/B04829_09_06.jpg)

### 注意

`wx`是 wxPython/Phoenix 库安装的文件夹名称。我们将在 Python 代码中导入此模块。

我们可以通过执行来自官方 wxPython/Phoenix 网站的简单演示脚本来验证我们的安装是否成功。官方网站的链接是[`wxpython.org/Phoenix/docs/html/`](http://wxpython.org/Phoenix/docs/html/)。

```py
import wx
app = wx.App()
frame = wx.Frame(None, -1, "Hello World")
frame.Show()
app.MainLoop()
```

运行上述 Python 3 脚本将使用 wxPython/Phoenix 创建以下 GUI。

![如何做...](img/B04829_09_07.jpg)

## 工作原理...

在这个食谱中，我们成功安装了与 Python 3 兼容的正确版本的 wxPython 工具包。我们找到了这个 GUI 工具包的 Phoenix 项目，这是当前和活跃的开发线。Phoenix 将在未来取代 Classic wxPython 工具包，特别适用于与 Python 3 良好地配合使用。

成功安装了 wxPython/Phoenix 工具包后，我们只用了五行代码就创建了一个 GUI。

### 注意

我们之前使用 tkinter 实现了相同的结果。

# 如何在 wxPython 中创建我们的 GUI

在这个食谱中，我们将开始使用 wxPython GUI 工具包创建我们的 Python GUI。

我们将首先使用随 Python 一起提供的 tkinter 重新创建我们之前创建的几个小部件。

然后，我们将探索一些使用 tkinter 更难创建的 wxPython GUI 工具包提供的小部件。

## 准备工作

前面的食谱向您展示了如何安装与您的 Python 版本和操作系统匹配的正确版本的 wxPython。

## 如何做...

开始探索 wxPython GUI 工具包的一个好地方是访问以下网址：[`wxpython.org/Phoenix/docs/html/gallery.html`](http://wxpython.org/Phoenix/docs/html/gallery.html)

这个网页显示了许多 wxPython 小部件。点击任何一个小部件，我们会进入它们的文档，这是一个非常好的和有用的功能，可以快速了解 wxPython 控件。

![如何做...](img/B04829_09_08.jpg)

以下屏幕截图显示了 wxPython 按钮小部件的文档。

![如何做...](img/B04829_09_09.jpg)

我们可以非常快速地创建一个带有标题、菜单栏和状态栏的工作窗口。当鼠标悬停在菜单项上时，状态栏会显示菜单项的文本。这可以通过编写以下代码来实现：

```py
# Import wxPython GUI toolkit
import wx

# Subclass wxPython frame
class GUI(wx.Frame):
    def __init__(self, parent, title, size=(200,100)):
        # Initialize super class
        wx.Frame.__init__(self, parent, title=title, size=size)

        # Change the frame background color 
        self.SetBackgroundColour('white')

        # Create Status Bar
        self.CreateStatusBar() 

        # Create the Menu
        menu= wx.Menu()

        # Add Menu Items to the Menu
        menu.Append(wx.ID_ABOUT, "About", "wxPython GUI")
        menu.AppendSeparator()
        menu.Append(wx.ID_EXIT,"Exit"," Exit the GUI")

        # Create the MenuBar
        menuBar = wx.MenuBar()

        # Give the MenuBar a Title
        menuBar.Append(menu,"File") 

        # Connect the MenuBar to the frame
        self.SetMenuBar(menuBar)  

        # Display the frame
        self.Show()

# Create instance of wxPython application
app = wx.App()

# Call sub-classed wxPython GUI increasing default Window size
GUI(None, "Python GUI using wxPython", (300,150))

# Run the main GUI event loop
app.MainLoop()
```

这创建了以下使用 wxPython 库编写的 Python GUI。

![如何做...](img/B04829_09_10.jpg)

在前面的代码中，我们继承自`wx.Frame`。在下面的代码中，我们继承自`wx.Panel`，并将`wx.Frame`传递给我们的类的`__init__()`方法。

### 注意

在 wxPython 中，顶级 GUI 窗口称为框架。没有框架就不能有 wxPython GUI，框架必须作为 wxPython 应用程序的一部分创建。

我们在代码底部同时创建应用程序和框架。

为了向我们的 GUI 添加小部件，我们必须将它们附加到一个面板上。面板的父级是框架（我们的顶级窗口），我们放置在面板中的小部件的父级是面板。

以下代码向一个面板小部件添加了一个多行文本框小部件。我们还向面板小部件添加了一个按钮小部件，当点击时，会向文本框打印一些文本。

以下是完整的代码：

```py
import wx               # Import wxPython GUI toolkit
class GUI(wx.Panel):    # Subclass wxPython Panel
    def __init__(self, parent):

        # Initialize super class
        wx.Panel.__init__(self, parent)

        # Create Status Bar
        parent.CreateStatusBar() 

        # Create the Menu
        menu= wx.Menu()

        # Add Menu Items to the Menu
        menu.Append(wx.ID_ABOUT, "About", "wxPython GUI")
        menu.AppendSeparator()
        menu.Append(wx.ID_EXIT,"Exit"," Exit the GUI")

        # Create the MenuBar
        menuBar = wx.MenuBar()

        # Give the Menu a Title
        menuBar.Append(menu,"File") 

        # Connect the MenuBar to the frame
        parent.SetMenuBar(menuBar)  

        # Create a Print Button
        button = wx.Button(self, label="Print", pos=(0,60))

        # Connect Button to Click Event method 
        self.Bind(wx.EVT_BUTTON, self.printButton, button)

        # Create a Text Control widget 
        self.textBox = wx.TextCtrl(
self, size=(280,50), style=wx.TE_MULTILINE)

    def printButton(self, event):
        self.textBox.AppendText(
"The Print Button has been clicked!") 

app = wx.App()      # Create instance of wxPython application
frame = wx.Frame(None, title="Python GUI using wxPython", size=(300,180))     # Create frame
GUI(frame)          # Pass frame into GUI
frame.Show()        # Display the frame
app.MainLoop()      # Run the main GUI event loop
```

运行前面的代码并点击我们的 wxPython 按钮小部件会产生以下 GUI 输出：

![如何做...](img/B04829_09_11.jpg)

## 工作原理...

在这个食谱中，我们使用成熟的 wxPython GUI 工具包创建了自己的 GUI。只需几行 Python 代码，我们就能创建一个带有“最小化”、“最大化”和“退出”按钮的完全功能的 GUI。我们添加了一个菜单栏，一个多行文本控件和一个按钮。我们还创建了一个状态栏，当我们选择菜单项时会显示文本。我们将所有这些小部件放入了一个面板容器小部件中。

我们将按钮连接到文本控件以打印文本。

当鼠标悬停在菜单项上时，状态栏会显示一些文本。

# 使用 wxPython 快速添加控件

在这个食谱中，我们将重新创建我们在本书中早期使用 tkinter 创建的 GUI，但这次，我们将使用 wxPython 库。我们将看到使用 wxPython GUI 工具包创建我们自己的 Python GUI 是多么简单和快速。

我们不会重新创建我们在之前章节中创建的整个功能。例如，我们不会国际化我们的 wxPython GUI，也不会将其连接到 MySQL 数据库。我们将重新创建 GUI 的视觉方面并添加一些功能。

### 注意

比较不同的库可以让我们选择使用哪些工具包来开发我们自己的 Python GUI，并且我们可以在我们自己的 Python 代码中结合几个工具包。

## 准备工作

确保你已经安装了 wxPython 模块以便按照这个步骤进行。

## 如何做...

首先，我们像以前在 tkinter 中那样创建我们的 Python`OOP`类，但这次我们继承并扩展了`wx.Frame`类。出于清晰的原因，我们不再将我们的类称为`OOP`，而是将其重命名为`MainFrame`。

### 注意

在 wxPython 中，主 GUI 窗口被称为 Frame。

我们还创建了一个回调方法，当我们单击“退出”菜单项时关闭 GUI，并将浅灰色的“元组”声明为我们 GUI 的背景颜色。

```py
import wx
BACKGROUNDCOLOR = (240, 240, 240, 255)

class MainFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)

        self.createWidgets()
        self.Show()

    def exitGUI(self, event):     # callback
        self.Destroy()

    def createWidgets(self):   
        self.CreateStatusBar()      # wxPython built-in method
        self.createMenu()
       self.createNotebook()
```

接下来，我们通过创建 wxPython`Notebook`类的实例并将其分配为我们自己的名为`Widgets`的自定义类的父类，向我们的 GUI 添加一个选项卡控件。

`notebook`类实例变量的父类是`wx.Panel`。

```py
    def createNotebook(self):
        panel = wx.Panel(self)
        notebook = wx.Notebook(panel)
        widgets = Widgets(notebook) # Custom class explained below
        notebook.AddPage(widgets, "Widgets")
        notebook.SetBackgroundColour(BACKGROUNDCOLOR) 
        # layout
        boxSizer = wx.BoxSizer()
        boxSizer.Add(notebook, 1, wx.EXPAND)
        panel.SetSizerAndFit(boxSizer)  
```

### 注意

在 wxPython 中，选项卡小部件被命名为`Notebook`，就像在 tkinter 中一样。

每个`Notebook`小部件都需要一个父类，并且为了在 wxPython 中布局`Notebook`中的小部件，我们使用不同类型的 sizers。

### 注意

wxPython sizers 是类似于 tkinter 的网格布局管理器的布局管理器。

接下来，我们向我们的 Notebook 页面添加控件。我们通过创建一个从`wx.Panel`继承的单独类来实现这一点。

```py
class Widgets(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.createWidgetsFrame()
        self.addWidgets()
        self.layoutWidgets()
```

我们通过将 GUI 代码模块化为小方法来遵循 Python OOP 编程最佳实践，这样可以使我们的代码易于管理和理解。

```py
    #------------------------------------------------------
    def createWidgetsFrame(self):
        self.panel = wx.Panel(self)
        staticBox = wx.StaticBox( self.panel, -1, "Widgets Frame" )    
        self.statBoxSizerV = wx.StaticBoxSizer(staticBox, 
                                               wx.VERTICAL)
    #-----------------------------------------------------
    def layoutWidgets(self):         
        boxSizerV = wx.BoxSizer( wx.VERTICAL )
        boxSizerV.Add( self.statBoxSizerV, 1, wx.ALL )
        self.panel.SetSizer( boxSizerV )
        boxSizerV.SetSizeHints( self.panel )

    #------------------------------------------------------
    def addWidgets(self):
        self.addCheckBoxes()        
        self.addRadioButtons()
        self.addStaticBoxWithLabels()
```

### 注意

在使用 wxPython StaticBox 小部件时，为了成功地对其进行布局，我们使用了`StaticBoxSizer`和常规的`BoxSizer`的组合。wxPython StaticBox 与 tkinter 的 LabelFrame 小部件非常相似。

在 tkinter 中，将一个`StaticBox`嵌入另一个`StaticBox`很简单，但在 wxPython 中使用起来有点不直观。使其工作的一种方法如下所示：

```py
    def addStaticBoxWithLabels(self):
        boxSizerH = wx.BoxSizer(wx.HORIZONTAL)
        staticBox = wx.StaticBox( self.panel, -1, 
"Labels within a Frame" )
        staticBoxSizerV = wx.StaticBoxSizer( staticBox, wx.VERTICAL )
        boxSizerV = wx.BoxSizer( wx.VERTICAL )
        staticText1 = wx.StaticText( self.panel, -1,
"Choose a number:" )
        boxSizerV.Add( staticText1, 0, wx.ALL)
        staticText2 = wx.StaticText( self.panel, -1,"Label 2")
        boxSizerV.Add( staticText2, 0, wx.ALL )
        #------------------------------------------------------
        staticBoxSizerV.Add( boxSizerV, 0, wx.ALL )
        boxSizerH.Add(staticBoxSizerV)
        #------------------------------------------------------
        boxSizerH.Add(wx.TextCtrl(self.panel))
        # Add local boxSizer to main frame
        self.statBoxSizerV.Add( boxSizerH, 1, wx.ALL )
```

首先，我们创建一个水平的`BoxSizer`。接下来，我们创建一个垂直的`StaticBoxSizer`，因为我们想在这个框架中以垂直布局排列两个标签。

为了将另一个小部件排列到嵌入的`StaticBox`的右侧，我们必须将嵌入的`StaticBox`及其子控件和下一个小部件都分配给水平的`BoxSizer`，然后将这个`BoxSizer`（现在包含了我们的嵌入的`StaticBox`和其他小部件）分配给主`StaticBox`。

这听起来令人困惑吗？

你只需要尝试使用这些 sizers 来感受如何使用它们。从这个步骤的代码开始，注释掉一些代码，或者修改一些 x 和 y 坐标来看看效果。

阅读官方的 wxPython 文档也是很有帮助的。

### 注意

重要的是要知道在代码中的哪里添加不同的 sizers 以实现我们希望的布局。

为了在第一个下面创建第二个`StaticBox`，我们创建单独的`StaticBoxSizers`并将它们分配给同一个面板。

```py
class Widgets(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.panel = wx.Panel(self)
        self.createWidgetsFrame()
        self.createManageFilesFrame()
        self.addWidgets()
        self.addFileWidgets()
        self.layoutWidgets()

    #----------------------------------------------------------
    def createWidgetsFrame(self):
        staticBox = wx.StaticBox( 
self.panel, -1, "Widgets Frame", size=(285, -1) )   
        self.statBoxSizerV = wx.StaticBoxSizer(
staticBox, wx.VERTICAL)   

    #----------------------------------------------------------
    def createManageFilesFrame(self):
        staticBox = wx.StaticBox( 
self.panel, -1, "Manage Files", size=(285, -1) )   
        self.statBoxSizerMgrV = wx.StaticBoxSizer(
staticBox, wx.VERTICAL)

    #----------------------------------------------------------
    def layoutWidgets(self):         
        boxSizerV = wx.BoxSizer( wx.VERTICAL )
        boxSizerV.Add( self.statBoxSizerV, 1, wx.ALL )
        boxSizerV.Add( self.statBoxSizerMgrV, 1, wx.ALL )

        self.panel.SetSizer( boxSizerV )
        boxSizerV.SetSizeHints( self.panel )

    #----------------------------------------------------------
    def addFileWidgets(self):   
        boxSizerH = wx.BoxSizer(wx.HORIZONTAL)
        boxSizerH.Add(wx.Button(
self.panel, label='Browse to File...'))   
        boxSizerH.Add(wx.TextCtrl(
self.panel, size=(174, -1), value= "Z:\\" ))

        boxSizerH1 = wx.BoxSizer(wx.HORIZONTAL)
        boxSizerH1.Add(wx.Button(
self.panel, label='Copy File To:    ')) 
        boxSizerH1.Add(wx.TextCtrl(
self.panel, size=(174, -1), value= "Z:\\Backup" ))    

        boxSizerV = wx.BoxSizer(wx.VERTICAL)
        boxSizerV.Add(boxSizerH)
        boxSizerV.Add(boxSizerH1)        

        self.statBoxSizerMgrV.Add( boxSizerV, 1, wx.ALL )
```

以下代码实例化了主事件循环，运行我们的 wxPython GUI 程序。

```py
#======================
# Start GUI
#======================
app = wx.App()
MainFrame(None, title="Python GUI using wxPython", size=(350,450))
app.MainLoop()
```

我们的 wxPython GUI 的最终结果如下：

![如何做...](img/B04829_09_12.jpg)

## 工作原理...

我们在几个类中设计和布局我们的 wxPython GUI。

在我们的 Python 模块的底部部分完成这些操作后，我们创建了一个 wxPython 应用程序的实例。接下来，我们实例化我们的 wxPython GUI 代码。

之后，我们调用主 GUI 事件循环，该循环执行在此应用程序进程中运行的所有 Python 代码。这将显示我们的 wxPython GUI。

### 注意

我们放置在创建应用程序和调用其主事件循环之间的任何代码都成为我们的 wxPython GUI。

可能需要一些时间来真正熟悉 wxPython 库及其 API，但一旦我们了解如何使用它，这个库就真的很有趣，是构建自己的 Python GUI 的强大工具。还有一个可与 wxPython 一起使用的可视化设计工具：[`www.cae.tntech.edu/help/programming/wxdesigner-getting-started/view`](http://www.cae.tntech.edu/help/programming/wxdesigner-getting-started/view)

这个示例使用面向对象编程来学习如何使用 wxPython GUI 工具包。

# 尝试将主要的 wxPython 应用程序嵌入到主要的 tkinter 应用程序中

现在，我们已经使用 Python 内置的 tkinter 库以及 wxWidgets 库的 wxPython 包装器创建了相同的 GUI，我们确实需要结合使用这些技术创建的 GUI。

### 注意

wxPython 和 tkinter 库都有各自的优势。在诸如[`stackoverflow.com/`](http://stackoverflow.com/)的在线论坛上，我们经常看到诸如哪个更好？应该使用哪个 GUI 工具包？这表明我们必须做出“二选一”的决定。我们不必做出这样的决定。

这样做的主要挑战之一是每个 GUI 工具包都必须有自己的事件循环。

在这个示例中，我们将尝试通过从我们的 tkinter GUI 中调用它来嵌入一个简单的 wxPython GUI。

## 准备工作

我们将重用在第一章中构建的 tkinter GUI。

## 如何做...

我们从一个简单的 tkinter GUI 开始，看起来像这样：

![如何做...](img/B04829_09_13.jpg)

接下来，我们将尝试调用在本章前一篇示例中创建的简单 wxPython GUI。

这是以简单的非面向对象编程方式完成此操作的整个代码：

```py
#===========================================================
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

win = tk.Tk()    

win.title("Python GUI")
aLabel = ttk.Label(win, text="A Label")
aLabel.grid(column=0, row=0)    
ttk.Label(win, text="Enter a name:").grid(column=0, row=0)
name = tk.StringVar()
nameEntered = ttk.Entry(win, width=12, textvariable=name)
nameEntered.grid(column=0, row=1)
ttk.Label(win, text="Choose a number:").grid(column=1, row=0)
number = tk.StringVar()
numberChosen = ttk.Combobox(win, width=12, textvariable=number)
numberChosen['values'] = (1, 2, 4, 42, 100)
numberChosen.grid(column=1, row=1)
numberChosen.current(0)
scrolW  = 30
scrolH  =  3
scr = scrolledtext.ScrolledText(win, width=scrolW, height=scrolH, wrap=tk.WORD)
scr.grid(column=0, sticky='WE', columnspan=3)
nameEntered.focus()  

#===========================================================
def wxPythonApp():
    import wx
    app = wx.App()
    frame = wx.Frame(None, -1, "wxPython GUI", size=(200,150))
    frame.SetBackgroundColour('white')
    frame.CreateStatusBar()
    menu= wx.Menu()
    menu.Append(wx.ID_ABOUT, "About", "wxPython GUI")
    menuBar = wx.MenuBar()
    menuBar.Append(menu,"File") 
    frame.SetMenuBar(menuBar)     
    frame.Show()
    app.MainLoop()

action = ttk.Button(win, text="Call wxPython GUI", command= wxPythonApp ) 
action.grid(column=2, row=1)

#======================
# Start GUI
#======================
win.mainloop()
```

运行上述代码后，单击 tkinter `Button`控件后，从我们的 tkinter GUI 启动了一个 wxPython GUI。

![如何做...](img/B04829_09_14.jpg)

## 它是如何工作的...

重要的是，我们将整个 wxPython 代码放入了自己的函数中，我们将其命名为`def wxPythonApp()`。

在按钮单击事件的回调函数中，我们只需调用此代码。

### 注意

需要注意的一点是，在继续使用 tkinter GUI 之前，我们必须关闭 wxPython GUI。

# 尝试将我们的 tkinter GUI 代码嵌入到 wxPython 中

在这个示例中，我们将与上一个示例相反，尝试从 wxPython GUI 中调用我们的 tkinter GUI 代码。

## 准备工作

我们将重用在本章前一篇示例中创建的一些 wxPython GUI 代码。

## 如何做...

我们将从一个简单的 wxPython GUI 开始，看起来像这样：

![如何做...](img/B04829_09_15.jpg)

接下来，我们将尝试调用一个简单的 tkinter GUI。

这是以简单的非面向对象编程方式完成此操作的整个代码：

```py
#=============================================================
def tkinterApp():
    import tkinter as tk
    from tkinter import ttk
    win = tk.Tk()    
    win.title("Python GUI")
    aLabel = ttk.Label(win, text="A Label")
    aLabel.grid(column=0, row=0)    
    ttk.Label(win, text="Enter a name:").grid(column=0, row=0)
    name = tk.StringVar()
    nameEntered = ttk.Entry(win, width=12, textvariable=name)
    nameEntered.grid(column=0, row=1)
    nameEntered.focus()  
    def buttonCallback():
        action.configure(text='Hello ' + name.get())
    action = ttk.Button(win, text="Print", command=buttonCallback)
    action.grid(column=2, row=1)
    win.mainloop()

#=============================================================
import wx
app = wx.App()
frame = wx.Frame(None, -1, "wxPython GUI", size=(200,180))
frame.SetBackgroundColour('white')
frame.CreateStatusBar()
menu= wx.Menu()
menu.Append(wx.ID_ABOUT, "About", "wxPython GUI")
menuBar = wx.MenuBar()
menuBar.Append(menu,"File") 
frame.SetMenuBar(menuBar) 
textBox = wx.TextCtrl(frame, size=(180,50), style=wx.TE_MULTILINE)

def tkinterEmbed(event):
    tkinterApp()

button = wx.Button(frame, label="Call tkinter GUI", pos=(0,60)) 
frame.Bind(wx.EVT_BUTTON, tkinterEmbed, button)
frame.Show()

#======================
# Start wxPython GUI
#======================
app.MainLoop()
```

运行上述代码后，单击 wxPython `Button`小部件后，从我们的 wxPython GUI 启动了一个 tkinter GUI。然后我们可以在 tkinter 文本框中输入文本。通过单击其按钮，按钮文本将更新为该名称。

![如何做...](img/B04829_09_16.jpg)

在启动 tkinter 事件循环后，wxPython GUI 仍然可以响应，因为我们可以在 tkinter GUI 运行时输入`TextCtrl`小部件。

### 注意

在上一个示例中，我们在关闭 wxPython GUI 之前无法使用我们的 tkinter GUI。意识到这种差异可以帮助我们的设计决策，如果我们想要结合这两种 Python GUI 技术。

通过多次单击 wxPython GUI 按钮，我们还可以创建几个 tkinter GUI 实例。但是，只要有任何 tkinter GUI 仍在运行，我们就不能关闭 wxPython GUI。我们必须先关闭它们。

![如何做...](img/B04829_09_17.jpg)

## 它是如何工作的...

在这个示例中，我们与上一个示例相反，首先使用 wxPython 创建 GUI，然后在其中使用 tkinter 创建了几个 GUI 实例。

当一个或多个 tkinter GUI 正在运行时，wxPython GUI 仍然保持响应。但是，单击 tkinter 按钮只会更新第一个实例中的按钮文本。

# 如何使用 Python 来控制两种不同的 GUI 框架

在这个配方中，我们将探讨如何从 Python 控制 tkinter 和 wxPython GUI 框架。在上一章中，我们已经使用 Python 的线程模块来保持我们的 GUI 响应，所以在这里我们将尝试使用相同的方法。

我们将看到事情并不总是按照直觉的方式工作。

然而，我们将改进我们的 tkinter GUI，使其在我们从中调用 wxPython GUI 的实例时不再无响应。

## 准备工作

这个配方将扩展本章的一个先前配方，我们试图将一个主要的 wxPython GUI 嵌入到我们的 tkinter GUI 中。

## 如何做...

当我们从 tkinter GUI 创建了一个 wxPython GUI 的实例时，我们就不能再使用 tkinter GUI 控件，直到关闭了 wxPython GUI 的一个实例。让我们现在改进一下。

我们的第一次尝试可能是在 tkinter 按钮回调函数中使用线程。

例如，我们的代码可能是这样的：

```py
def wxPythonApp():
    import wx
    app = wx.App()
    frame = wx.Frame(None, -1, "wxPython GUI", size=(200,150))
    frame.SetBackgroundColour('white')
    frame.CreateStatusBar()
    menu= wx.Menu()
    menu.Append(wx.ID_ABOUT, "About", "wxPython GUI")
    menuBar = wx.MenuBar()
    menuBar.Append(menu,"File") 
    frame.SetMenuBar(menuBar)     
    frame.Show()
    app.MainLoop()

def tryRunInThread():
    runT = Thread(target=wxPythonApp)
    runT.setDaemon(True)    
    runT.start()
    print(runT)
    print('createThread():', runT.isAlive())    

action = ttk.Button(win, text="Call wxPython GUI", command=tryRunInThread)
```

起初，这似乎是有效的，这是直观的，因为 tkinter 控件不再被禁用，我们可以通过单击按钮创建几个 wxPython GUI 的实例。我们还可以在其他 tkinter 小部件中输入和选择。

![如何做...](img/B04829_09_18.jpg)

然而，一旦我们试图关闭 GUI，我们会从 wxWidgets 得到一个错误，我们的 Python 可执行文件会崩溃。

![如何做...](img/B04829_09_19.jpg)

为了避免这种情况，我们可以改变代码，只让 wxPython 的`app.MainLoop`在一个线程中运行，而不是尝试在一个线程中运行整个 wxPython 应用程序。

```py
def wxPythonApp():
    import wx
    app = wx.App()
    frame = wx.Frame(None, -1, "wxPython GUI", size=(200,150))
    frame.SetBackgroundColour('white')
    frame.CreateStatusBar()
    menu= wx.Menu()
    menu.Append(wx.ID_ABOUT, "About", "wxPython GUI")
    menuBar = wx.MenuBar()
    menuBar.Append(menu,"File") 
    frame.SetMenuBar(menuBar)     
    frame.Show()

    runT = Thread(target=app.MainLoop)
    runT.setDaemon(True)    
    runT.start()
    print(runT)
    print('createThread():', runT.isAlive())

action = ttk.Button(win, text="Call wxPython GUI", command=wxPythonApp) 
action.grid(column=2, row=1)
```

## 它是如何工作的...

我们最初尝试在一个线程中运行整个 wxPython GUI 应用程序，但这并不起作用，因为 wxPython 的主事件循环期望成为应用程序的主线程。

我们找到了一个解决方法，只在一个线程中运行 wxPython 的`app.MainLoop`，这样就可以欺骗它认为它是主线程。

这种方法的一个副作用是，我们不能再单独关闭所有的 wxPython GUI 实例。至少其中一个只有在我们关闭创建线程为守护进程的 wxPython GUI 时才关闭。

我不太确定为什么会这样。直觉上，人们可能期望能够关闭所有守护线程，而不必等待创建它们的主线程先关闭。

这可能与引用计数器没有被设置为零，而我们的主线程仍在运行有关。

在实际层面上，这是当前的工作方式。

# 如何在两个连接的 GUI 之间进行通信

在之前的配方中，我们找到了连接 wxPython GUI 和 tkinter GUI 的方法，相互调用彼此。

虽然两个 GUI 成功同时运行，但它们实际上并没有真正相互通信，因为它们只是互相启动。

在这个配方中，我们将探讨使这两个 GUI 相互通信的方法。

## 准备工作

阅读之前的一些配方可能是为这个配方做好准备的好方法。

在这个配方中，我们将使用与之前配方相比略有修改的 GUI 代码，但大部分基本的 GUI 构建代码是相同的。

## 如何做...

在之前的配方中，我们的主要挑战之一是如何将两个设计为应用程序的唯一 GUI 工具包的 GUI 技术结合起来。我们找到了各种简单的方法来将它们结合起来。

我们将再次从 tkinter GUI 的主事件循环中启动 wxPython GUI，并在 tkinter 进程中启动 wxPython GUI 的自己的线程。

为了做到这一点，我们将使用一个共享的全局多进程 Python 队列。

### 注意

虽然在这个配方中最好避免全局数据，但它们是一个实际的解决方案，Python 全局变量实际上只在它们被声明的模块中是全局的。

这是使两个 GUI 在一定程度上相互通信的 Python 代码。为了节省空间，这不是纯粹的面向对象编程代码。

我们也没有展示所有部件的创建代码。该代码与之前的示例中相同。

```py
# Ch09_Communicate.py
import tkinter as tk
from tkinter import ttk
from threading import Thread

win = tk.Tk()       
win.title("Python GUI")   

from multiprocessing import Queue
sharedQueue = Queue()
dataInQueue = False

def putDataIntoQueue(data):
    global dataInQueue
    dataInQueue =  True
    sharedQueue.put(data)

def readDataFromQueue():
    global dataInQueue
    dataInQueue = False
    return sharedQueue.get() 
#===========================================================
import wx               
class GUI(wx.Panel):    
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        parent.CreateStatusBar() 
        button = wx.Button(self, label="Print", pos=(0,60))
        self.Bind(wx.EVT_BUTTON, self.writeToSharedQueue, button)

    #--------------------------------------------------------
    def writeToSharedQueue(self, event):
        self.textBox.AppendText(
                        "The Print Button has been clicked!\n") 
        putDataIntoQueue('Hi from wxPython via Shared Queue.\n')
        if dataInQueue: 
            data = readDataFromQueue()
            self.textBox.AppendText(data)

            text.insert('0.0', data) # insert data into GUI

#============================================================
def wxPythonApp():
        app = wx.App()
        frame = wx.Frame(
            None, title="Python GUI using wxPython", size=(300,180))
        GUI(frame)          
        frame.Show()        
        runT = Thread(target=app.MainLoop)
        runT.setDaemon(True)    
        runT.start()
        print(runT)
        print('createThread():', runT.isAlive())
#============================================================
action = ttk.Button(win, text="Call wxPython GUI", command=wxPythonApp) 
action.grid(column=2, row=1)

#======================
# Start GUI
#======================
win.mainloop()
```

首先运行上述代码会创建程序的 tkinter 部分，当我们在这个 GUI 中点击按钮时，它会运行 wxPython GUI。与之前一样，两者同时运行，但这次，两个 GUI 之间有了额外的通信层级。

![操作步骤...](img/B04829_09_20.jpg)

在上述截图的左侧显示了 tkinter GUI，通过点击**Call wxPython GUI**按钮，我们调用了一个 wxPython GUI 的实例。我们可以通过多次点击按钮来创建多个实例。

### 注意

所有创建的 GUI 都保持响应。它们不会崩溃或冻结。

在任何一个 wxPython GUI 实例上点击**Print**按钮会向其自己的`TextCtrl`部件写入一句话，然后也会向自己以及 tkinter GUI 写入另一行。您需要向上滚动以在 wxPython GUI 中看到第一句话。

### 注意

这种工作方式是通过使用模块级队列和 tkinter 的`Text`部件来实现的。

重要的一点是，我们创建一个线程来运行 wxPython 的`app.MainLoop`，就像我们在之前的示例中所做的那样。

```py
def wxPythonApp():
        app = wx.App()
        frame = wx.Frame(
None, title="Python GUI using wxPython", size=(300,180))
        GUI(frame)          
        frame.Show()        
        runT = Thread(target=app.MainLoop)
        runT.setDaemon(True)    
        runT.start()
```

我们创建了一个从`wx.Panel`继承并命名为`GUI`的类。然后我们在上述代码中实例化了这个类。

我们在这个类中创建了一个按钮点击事件回调方法，然后调用了上面编写的过程代码。因此，该类可以访问这些函数并将数据写入共享队列。

```py
    #------------------------------------------------------
    def writeToSharedQueue(self, event):
        self.textBox.AppendText(
"The Print Button has been clicked!\n") 
        putDataIntoQueue('Hi from wxPython via Shared Queue.\n')
        if dataInQueue: 
            data = readDataFromQueue()
            self.textBox.AppendText(data)
            text.insert('0.0', data) # insert data into tkinter
```

我们首先检查在上述方法中是否已将数据放入共享队列，如果是这样，我们就将公共数据打印到两个 GUI 中。

### 注意

`putDataIntoQueue()`将数据放入队列，`readDataFromQueue()`将其读取出来并保存在`data`变量中。

`text.insert('0.0', data)`是将这些数据从**Print**按钮的 wxPython 回调方法写入到 tkinter GUI 中的代码行。

以下是在代码中被调用并使其工作的过程函数（不是方法，因为它们没有绑定）。

```py
from multiprocessing import Queue
sharedQueue = Queue()
dataInQueue = False

def putDataIntoQueue(data):
    global dataInQueue
    dataInQueue =  True
    sharedQueue.put(data)

def readDataFromQueue():
    global dataInQueue
    dataInQueue = False
    return sharedQueue.get()
```

我们使用一个名为`dataInQueue`的简单布尔标志来通知数据何时可用于队列中。

## 工作原理

在这个示例中，我们成功地以类似的方式将我们之前独立的两个 GUI 结合在一起，但彼此之间没有交流。然而，在这个示例中，我们通过使一个 GUI 启动另一个 GUI，并通过一个简单的多进程 Python 队列机制，进一步连接它们，我们能够使它们相互通信，将数据从共享队列写入到两个 GUI 中。

有许多非常先进和复杂的技术可用于连接不同的进程、线程、池、锁、管道、TCP/IP 连接等。

在 Python 精神中，我们找到了一个对我们有效的简单解决方案。一旦我们的代码变得更加复杂，我们可能需要重构它，但这是一个很好的开始。
