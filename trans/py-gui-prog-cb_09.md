# 第9章。使用wxPython库扩展我们的GUI

在本章中，我们将使用wxPython库增强我们的Python GUI。

+   如何安装wxPython库

+   如何在wxPython中创建我们的GUI

+   使用wxPython快速添加控件

+   尝试在主tkinter应用程序中嵌入主wxPython应用程序

+   尝试将我们的tkinter GUI代码嵌入到wxPython中

+   如何使用Python控制两个不同的GUI框架

+   如何在两个连接的GUI之间通信

# 介绍

在本章中，我们将介绍另一个Python GUI工具包，它目前不随Python一起发布。它被称为wxPython。

这个库有两个版本。原始版本称为Classic，而最新版本称为开发项目的代号Phoenix。

在本书中，我们仅使用Python 3进行编程，因为新的Phoenix项目旨在支持Python 3，这就是我们在本章中使用的wxPython版本。

首先，我们将创建一个简单的wxPython GUI，然后我们将尝试将我们在本书中开发的基于tkinter的GUI与新的wxPython库连接起来。

### 注意

wxPython是Python绑定到wxWidgets的库。

wxPython中的w代表Windows操作系统，x代表Unix操作系统，如Linux和OS X。

如果同时使用这两个GUI工具包出现问题，我们将尝试使用Python解决任何问题，然后我们将使用Python内的**进程间通信**（**IPC**）来确保我们的Python代码按我们希望的方式工作。

# 如何安装wxPython库

wxPython库不随Python一起发布，因此，为了使用它，我们首先必须安装它。

这个步骤将向我们展示在哪里以及如何找到正确的版本来安装，以匹配已安装的Python版本和正在运行的操作系统。

### 注意

wxPython第三方库已经存在了17年多，这表明它是一个强大的库。

## 准备工作

为了在Python 3中使用wxPython，我们必须安装wxPython Phoenix版本。

## 如何做...

在网上搜索wxPython时，我们可能会在[www.wxpython.org](http://www.wxpython.org)找到官方网站。

![如何做...](graphics/B04829_09_01.jpg)

如果我们点击MS Windows的下载链接，我们可以看到几个Windows安装程序，所有这些安装程序都仅适用于Python 2.x。

![如何做...](graphics/B04829_09_02.jpg)

使用Python 3和wxPython，我们必须安装wxPython/Phoenix库。我们可以在快照构建链接中找到安装程序：

[http://wxpython.org/Phoenix/snapshot-builds/](http://wxpython.org/Phoenix/snapshot-builds/)

从这里，我们可以选择与我们的Python版本和操作系统版本匹配的wxPython/Phoenix版本。我正在使用运行在64位Windows 7操作系统上的Python 3.4。

![如何做...](graphics/B04829_09_03.jpg)

Python wheel（.whl）安装程序包有一个编号方案。

对我们来说，这个方案最重要的部分是我们正在安装的wxPython/Phoenix版本是为Python 3.4（安装程序名称中的cp34）和Windows 64位操作系统（安装程序名称中的win_amd64）。

![如何做...](graphics/B04829_09_04.jpg)

成功下载wxPython/Phoenix包后，我们现在可以转到该包所在的目录，并使用pip安装此包。

![如何做...](graphics/B04829_09_05.jpg)

我们的Python“site-packages”文件夹中有一个名为“wx”的新文件夹。

![如何做...](graphics/B04829_09_06.jpg)

### 注意

“wx”是wxPython/Phoenix库安装的文件夹名称。我们将在Python代码中导入此模块。

我们可以通过执行来自官方wxPython/Phoenix网站的简单演示脚本来验证我们的安装是否成功。官方网站的链接是[http://wxpython.org/Phoenix/docs/html/](http://wxpython.org/Phoenix/docs/html/)。

```py
import wx
app = wx.App()
frame = wx.Frame(None, -1, "Hello World")
frame.Show()
app.MainLoop()
```

运行上述Python 3脚本将使用wxPython/Phoenix创建以下GUI。

![如何做...](graphics/B04829_09_07.jpg)

## 工作原理...

在这个食谱中，我们成功安装了与Python 3兼容的正确版本的wxPython工具包。我们找到了这个GUI工具包的Phoenix项目，这是当前和活跃的开发线。Phoenix将在未来取代Classic wxPython工具包，特别适用于与Python 3良好地配合使用。

成功安装了wxPython/Phoenix工具包后，我们只用了五行代码就创建了一个GUI。

### 注意

我们之前使用tkinter实现了相同的结果。

# 如何在wxPython中创建我们的GUI

在这个食谱中，我们将开始使用wxPython GUI工具包创建我们的Python GUI。

我们将首先使用随Python一起提供的tkinter重新创建我们之前创建的几个小部件。

然后，我们将探索一些使用tkinter更难创建的wxPython GUI工具包提供的小部件。

## 准备工作

前面的食谱向您展示了如何安装与您的Python版本和操作系统匹配的正确版本的wxPython。

## 如何做...

开始探索wxPython GUI工具包的一个好地方是访问以下网址：[http://wxpython.org/Phoenix/docs/html/gallery.html](http://wxpython.org/Phoenix/docs/html/gallery.html)

这个网页显示了许多wxPython小部件。点击任何一个小部件，我们会进入它们的文档，这是一个非常好的和有用的功能，可以快速了解wxPython控件。

![如何做...](graphics/B04829_09_08.jpg)

以下屏幕截图显示了wxPython按钮小部件的文档。

![如何做...](graphics/B04829_09_09.jpg)

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

这创建了以下使用wxPython库编写的Python GUI。

![如何做...](graphics/B04829_09_10.jpg)

在前面的代码中，我们继承自`wx.Frame`。在下面的代码中，我们继承自`wx.Panel`，并将`wx.Frame`传递给我们的类的`__init__()`方法。

### 注意

在wxPython中，顶级GUI窗口称为框架。没有框架就不能有wxPython GUI，框架必须作为wxPython应用程序的一部分创建。

我们在代码底部同时创建应用程序和框架。

为了向我们的GUI添加小部件，我们必须将它们附加到一个面板上。面板的父级是框架（我们的顶级窗口），我们放置在面板中的小部件的父级是面板。

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

运行前面的代码并点击我们的wxPython按钮小部件会产生以下GUI输出：

![如何做...](graphics/B04829_09_11.jpg)

## 工作原理...

在这个食谱中，我们使用成熟的wxPython GUI工具包创建了自己的GUI。只需几行Python代码，我们就能创建一个带有“最小化”、“最大化”和“退出”按钮的完全功能的GUI。我们添加了一个菜单栏，一个多行文本控件和一个按钮。我们还创建了一个状态栏，当我们选择菜单项时会显示文本。我们将所有这些小部件放入了一个面板容器小部件中。

我们将按钮连接到文本控件以打印文本。

当鼠标悬停在菜单项上时，状态栏会显示一些文本。

# 使用wxPython快速添加控件

在这个食谱中，我们将重新创建我们在本书中早期使用tkinter创建的GUI，但这次，我们将使用wxPython库。我们将看到使用wxPython GUI工具包创建我们自己的Python GUI是多么简单和快速。

我们不会重新创建我们在之前章节中创建的整个功能。例如，我们不会国际化我们的wxPython GUI，也不会将其连接到MySQL数据库。我们将重新创建GUI的视觉方面并添加一些功能。

### 注意

比较不同的库可以让我们选择使用哪些工具包来开发我们自己的Python GUI，并且我们可以在我们自己的Python代码中结合几个工具包。

## 准备工作

确保你已经安装了wxPython模块以便按照这个步骤进行。

## 如何做...

首先，我们像以前在tkinter中那样创建我们的Python“OOP”类，但这次我们继承并扩展了“wx.Frame”类。出于清晰的原因，我们不再将我们的类称为“OOP”，而是将其重命名为“MainFrame”。

### 注意

在wxPython中，主GUI窗口被称为Frame。

我们还创建了一个回调方法，当我们单击“退出”菜单项时关闭GUI，并将浅灰色的“元组”声明为我们GUI的背景颜色。

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

接下来，我们通过创建wxPython“Notebook”类的实例并将其分配为我们自己的名为“Widgets”的自定义类的父类，向我们的GUI添加一个选项卡控件。

“notebook”类实例变量的父类是“wx.Panel”。

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

在wxPython中，选项卡小部件被命名为“Notebook”，就像在tkinter中一样。

每个“Notebook”小部件都需要一个父类，并且为了在wxPython中布局“Notebook”中的小部件，我们使用不同类型的sizers。

### 注意

wxPython sizers是类似于tkinter的网格布局管理器的布局管理器。

接下来，我们向我们的Notebook页面添加控件。我们通过创建一个从“wx.Panel”继承的单独类来实现这一点。

```py
class Widgets(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.createWidgetsFrame()
        self.addWidgets()
        self.layoutWidgets()
```

我们通过将GUI代码模块化为小方法来遵循Python OOP编程最佳实践，这样可以使我们的代码易于管理和理解。

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

在使用wxPython StaticBox小部件时，为了成功地对其进行布局，我们使用了“StaticBoxSizer”和常规的“BoxSizer”的组合。wxPython StaticBox与tkinter的LabelFrame小部件非常相似。

在tkinter中，将一个“StaticBox”嵌入另一个“StaticBox”很简单，但在wxPython中使用起来有点不直观。使其工作的一种方法如下所示：

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

首先，我们创建一个水平的“BoxSizer”。接下来，我们创建一个垂直的“StaticBoxSizer”，因为我们想在这个框架中以垂直布局排列两个标签。

为了将另一个小部件排列到嵌入的“StaticBox”的右侧，我们必须将嵌入的“StaticBox”及其子控件和下一个小部件都分配给水平的“BoxSizer”，然后将这个“BoxSizer”（现在包含了我们的嵌入的“StaticBox”和其他小部件）分配给主“StaticBox”。

这听起来令人困惑吗？

你只需要尝试使用这些sizers来感受如何使用它们。从这个步骤的代码开始，注释掉一些代码，或者修改一些x和y坐标来看看效果。

阅读官方的wxPython文档也是很有帮助的。

### 注意

重要的是要知道在代码中的哪里添加不同的sizers以实现我们希望的布局。

为了在第一个下面创建第二个“StaticBox”，我们创建单独的“StaticBoxSizers”并将它们分配给同一个面板。

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

以下代码实例化了主事件循环，运行我们的wxPython GUI程序。

```py
#======================
# Start GUI
#======================
app = wx.App()
MainFrame(None, title="Python GUI using wxPython", size=(350,450))
app.MainLoop()
```

我们的wxPython GUI的最终结果如下：

![如何做...](graphics/B04829_09_12.jpg)

## 工作原理...

我们在几个类中设计和布局我们的wxPython GUI。

在我们的Python模块的底部部分完成这些操作后，我们创建了一个wxPython应用程序的实例。接下来，我们实例化我们的wxPython GUI代码。

之后，我们调用主GUI事件循环，该循环执行在此应用程序进程中运行的所有Python代码。这将显示我们的wxPython GUI。

### 注意

我们放置在创建应用程序和调用其主事件循环之间的任何代码都成为我们的wxPython GUI。

可能需要一些时间来真正熟悉wxPython库及其API，但一旦我们了解如何使用它，这个库就真的很有趣，是构建自己的Python GUI的强大工具。还有一个可与wxPython一起使用的可视化设计工具：[http://www.cae.tntech.edu/help/programming/wxdesigner-getting-started/view](http://www.cae.tntech.edu/help/programming/wxdesigner-getting-started/view)

这个示例使用面向对象编程来学习如何使用wxPython GUI工具包。

# 尝试将主要的wxPython应用程序嵌入到主要的tkinter应用程序中

现在，我们已经使用Python内置的tkinter库以及wxWidgets库的wxPython包装器创建了相同的GUI，我们确实需要结合使用这些技术创建的GUI。

### 注意

wxPython和tkinter库都有各自的优势。在诸如[http://stackoverflow.com/](http://stackoverflow.com/)的在线论坛上，我们经常看到诸如哪个更好？应该使用哪个GUI工具包？这表明我们必须做出“二选一”的决定。我们不必做出这样的决定。

这样做的主要挑战之一是每个GUI工具包都必须有自己的事件循环。

在这个示例中，我们将尝试通过从我们的tkinter GUI中调用它来嵌入一个简单的wxPython GUI。

## 准备工作

我们将重用在[第1章](ch01.html "第1章。创建GUI表单并添加小部件")中构建的tkinter GUI。

## 如何做...

我们从一个简单的tkinter GUI开始，看起来像这样：

![如何做...](graphics/B04829_09_13.jpg)

接下来，我们将尝试调用在本章前一篇示例中创建的简单wxPython GUI。

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

运行上述代码后，单击tkinter `Button`控件后，从我们的tkinter GUI启动了一个wxPython GUI。

![如何做...](graphics/B04829_09_14.jpg)

## 它是如何工作的...

重要的是，我们将整个wxPython代码放入了自己的函数中，我们将其命名为`def wxPythonApp()`。

在按钮单击事件的回调函数中，我们只需调用此代码。

### 注意

需要注意的一点是，在继续使用tkinter GUI之前，我们必须关闭wxPython GUI。

# 尝试将我们的tkinter GUI代码嵌入到wxPython中

在这个示例中，我们将与上一个示例相反，尝试从wxPython GUI中调用我们的tkinter GUI代码。

## 准备工作

我们将重用在本章前一篇示例中创建的一些wxPython GUI代码。

## 如何做...

我们将从一个简单的wxPython GUI开始，看起来像这样：

![如何做...](graphics/B04829_09_15.jpg)

接下来，我们将尝试调用一个简单的tkinter GUI。

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

运行上述代码后，单击wxPython `Button`小部件后，从我们的wxPython GUI启动了一个tkinter GUI。然后我们可以在tkinter文本框中输入文本。通过单击其按钮，按钮文本将更新为该名称。

![如何做...](graphics/B04829_09_16.jpg)

在启动tkinter事件循环后，wxPython GUI仍然可以响应，因为我们可以在tkinter GUI运行时输入`TextCtrl`小部件。

### 注意

在上一个示例中，我们在关闭wxPython GUI之前无法使用我们的tkinter GUI。意识到这种差异可以帮助我们的设计决策，如果我们想要结合这两种Python GUI技术。

通过多次单击wxPython GUI按钮，我们还可以创建几个tkinter GUI实例。但是，只要有任何tkinter GUI仍在运行，我们就不能关闭wxPython GUI。我们必须先关闭它们。

![如何做...](graphics/B04829_09_17.jpg)

## 它是如何工作的...

在这个示例中，我们与上一个示例相反，首先使用wxPython创建GUI，然后在其中使用tkinter创建了几个GUI实例。

当一个或多个tkinter GUI正在运行时，wxPython GUI仍然保持响应。但是，单击tkinter按钮只会更新第一个实例中的按钮文本。

# 如何使用Python来控制两种不同的GUI框架

在这个配方中，我们将探讨如何从Python控制tkinter和wxPython GUI框架。在上一章中，我们已经使用Python的线程模块来保持我们的GUI响应，所以在这里我们将尝试使用相同的方法。

我们将看到事情并不总是按照直觉的方式工作。

然而，我们将改进我们的tkinter GUI，使其在我们从中调用wxPython GUI的实例时不再无响应。

## 准备工作

这个配方将扩展本章的一个先前配方，我们试图将一个主要的wxPython GUI嵌入到我们的tkinter GUI中。

## 如何做...

当我们从tkinter GUI创建了一个wxPython GUI的实例时，我们就不能再使用tkinter GUI控件，直到关闭了wxPython GUI的一个实例。让我们现在改进一下。

我们的第一次尝试可能是在tkinter按钮回调函数中使用线程。

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

起初，这似乎是有效的，这是直观的，因为tkinter控件不再被禁用，我们可以通过单击按钮创建几个wxPython GUI的实例。我们还可以在其他tkinter小部件中输入和选择。

![如何做...](graphics/B04829_09_18.jpg)

然而，一旦我们试图关闭GUI，我们会从wxWidgets得到一个错误，我们的Python可执行文件会崩溃。

![如何做...](graphics/B04829_09_19.jpg)

为了避免这种情况，我们可以改变代码，只让wxPython的`app.MainLoop`在一个线程中运行，而不是尝试在一个线程中运行整个wxPython应用程序。

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

我们最初尝试在一个线程中运行整个wxPython GUI应用程序，但这并不起作用，因为wxPython的主事件循环期望成为应用程序的主线程。

我们找到了一个解决方法，只在一个线程中运行wxPython的`app.MainLoop`，这样就可以欺骗它认为它是主线程。

这种方法的一个副作用是，我们不能再单独关闭所有的wxPython GUI实例。至少其中一个只有在我们关闭创建线程为守护进程的wxPython GUI时才关闭。

我不太确定为什么会这样。直觉上，人们可能期望能够关闭所有守护线程，而不必等待创建它们的主线程先关闭。

这可能与引用计数器没有被设置为零，而我们的主线程仍在运行有关。

在实际层面上，这是当前的工作方式。

# 如何在两个连接的GUI之间进行通信

在之前的配方中，我们找到了连接wxPython GUI和tkinter GUI的方法，相互调用彼此。

虽然两个GUI成功同时运行，但它们实际上并没有真正相互通信，因为它们只是互相启动。

在这个配方中，我们将探讨使这两个GUI相互通信的方法。

## 准备工作

阅读之前的一些配方可能是为这个配方做好准备的好方法。

在这个配方中，我们将使用与之前配方相比略有修改的GUI代码，但大部分基本的GUI构建代码是相同的。

## 如何做...

在之前的配方中，我们的主要挑战之一是如何将两个设计为应用程序的唯一GUI工具包的GUI技术结合起来。我们找到了各种简单的方法来将它们结合起来。

我们将再次从tkinter GUI的主事件循环中启动wxPython GUI，并在tkinter进程中启动wxPython GUI的自己的线程。

为了做到这一点，我们将使用一个共享的全局多进程Python队列。

### 注意

虽然在这个配方中最好避免全局数据，但它们是一个实际的解决方案，Python全局变量实际上只在它们被声明的模块中是全局的。

这是使两个GUI在一定程度上相互通信的Python代码。为了节省空间，这不是纯粹的面向对象编程代码。

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

首先运行上述代码会创建程序的tkinter部分，当我们在这个GUI中点击按钮时，它会运行wxPython GUI。与之前一样，两者同时运行，但这次，两个GUI之间有了额外的通信层级。

![操作步骤...](graphics/B04829_09_20.jpg)

在上述截图的左侧显示了tkinter GUI，通过点击**Call wxPython GUI**按钮，我们调用了一个wxPython GUI的实例。我们可以通过多次点击按钮来创建多个实例。

### 注意

所有创建的GUI都保持响应。它们不会崩溃或冻结。

在任何一个wxPython GUI实例上点击**Print**按钮会向其自己的`TextCtrl`部件写入一句话，然后也会向自己以及tkinter GUI写入另一行。您需要向上滚动以在wxPython GUI中看到第一句话。

### 注意

这种工作方式是通过使用模块级队列和tkinter的`Text`部件来实现的。

重要的一点是，我们创建一个线程来运行wxPython的`app.MainLoop`，就像我们在之前的示例中所做的那样。

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

我们首先检查在上述方法中是否已将数据放入共享队列，如果是这样，我们就将公共数据打印到两个GUI中。

### 注意

`putDataIntoQueue()`将数据放入队列，`readDataFromQueue()`将其读取出来并保存在`data`变量中。

`text.insert('0.0', data)`是将这些数据从**Print**按钮的wxPython回调方法写入到tkinter GUI中的代码行。

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

在这个示例中，我们成功地以类似的方式将我们之前独立的两个GUI结合在一起，但彼此之间没有交流。然而，在这个示例中，我们通过使一个GUI启动另一个GUI，并通过一个简单的多进程Python队列机制，进一步连接它们，我们能够使它们相互通信，将数据从共享队列写入到两个GUI中。

有许多非常先进和复杂的技术可用于连接不同的进程、线程、池、锁、管道、TCP/IP连接等。

在Python精神中，我们找到了一个对我们有效的简单解决方案。一旦我们的代码变得更加复杂，我们可能需要重构它，但这是一个很好的开始。
