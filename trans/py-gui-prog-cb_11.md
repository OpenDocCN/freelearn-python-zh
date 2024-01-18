# 第十一章。最佳实践

在本章中，我们将探讨与 Python GUI 相关的最佳实践。

+   避免意大利面代码

+   使用 __init__ 连接模块

+   混合下降和 OOP 编码

+   使用代码命名约定

+   何时不使用 OOP

+   成功使用设计模式的方法

+   避免复杂性

# 介绍

在本章中，我们将探讨可以帮助我们以高效的方式构建 GUI 并使其易于维护和扩展的不同最佳实践。

这些最佳实践也将帮助我们调试 GUI，使其成为我们想要的样子。

# 避免意大利面代码

在这个配方中，我们将探讨创建意大利面代码的典型方式，然后我们将看到如何避免这样的代码的更好方式。

### 注意

意大利面代码是一种功能交织在一起的代码。

## 准备就绪

我们将使用内置的 Python 库 tkinter 来创建一个新的简单 GUI。

## 如何做...

在网上搜索并阅读文档后，我们可能会开始编写以下代码来创建我们的 GUI：

```py
# Spaghetti Code #############################
def PRINTME(me):print(me)
import tkinter 
x=y=z=1
PRINTME(z) 
from tkinter import *
scrolW=30;scrolH=6
win=tkinter.Tk()
if x:chVarUn=tkinter.IntVar()
from tkinter import ttk
WE='WE'
import tkinter.scrolledtext
outputFrame=tkinter.ttk.LabelFrame(win,text=' Type into the scrolled text control: ')
scr=tkinter.scrolledtext.ScrolledText(outputFrame,width=scrolW,height=scrolH,wrap=tkinter.WORD)
e='E'
scr.grid(column=1,row=1,sticky=WE)
outputFrame.grid(column=0,row=2,sticky=e,padx=8)
lFrame=None
if y:chck2=tkinter.Checkbutton(lFrame,text="Enabled",variable=chVarUn)
wE='WE'
if y==x:PRINTME(x) 
lFrame=tkinter.ttk.LabelFrame(win,text="Spaghetti")
chck2.grid(column=1,row=4,sticky=tkinter.W,columnspan=3)  
PRINTME(z)
lFrame.grid(column=0,row=0,sticky=wE,padx=10,pady=10) 
chck2.select()
try: win.mainloop()
except:PRINTME(x)
chck2.deselect()
if y==x:PRINTME(x) 
# End Pasta #############################
```

运行上述代码会产生以下 GUI：

![如何做...](img/B04829_11_01.jpg)

这并不是我们打算的 GUI。我们希望它看起来更像这样：

![如何做...](img/B04829_11_02.jpg)

虽然意大利面代码创建了一个 GUI，但代码非常难以调试，因为代码中有很多混乱。

以下是产生所需 GUI 的代码：

```py
#======================
# imports
#======================
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

#======================
# Create instance
#======================
win = tk.Tk()   

#======================
# Add a title       
#====================== 
win.title("Python GUI")

#=========================
# Disable resizing the GUI
#=========================
win.resizable(0,0)

#=============================================================
# Adding a LabelFrame, Textbox (Entry) and Combobox  
#=============================================================
lFrame = ttk.LabelFrame(win, text="Python GUI Programming Cookbook")
lFrame.grid(column=0, row=0, sticky='WE', padx=10, pady=10)

#=============================================================
# Using a scrolled Text control    
#=============================================================
outputFrame = ttk.LabelFrame(win, text=' Type into the scrolled text control: ')
outputFrame.grid(column=0, row=2, sticky='E', padx=8)
scrolW  = 30
scrolH  =  6
scr = scrolledtext.ScrolledText(outputFrame, width=scrolW, height=scrolH, wrap=tk.WORD)
scr.grid(column=1, row=0, sticky='WE')

#=============================================================
# Creating a checkbutton
#=============================================================
chVarUn = tk.IntVar()
check2 = tk.Checkbutton(lFrame, text="Enabled", variable=chVarUn)
check2.deselect()
check2.grid(column=1, row=4, sticky=tk.W, columnspan=3) 

#======================
# Start GUI
#======================
win.mainloop()
```

## 它是如何工作的...

在这个配方中，我们将意大利面代码与良好的代码进行了比较。良好的代码比意大利面代码有很多优势。

它有清晰的注释部分。

意大利面代码：

```py
def PRINTME(me):print(me)
import tkinter 
x=y=z=1
PRINTME(z) 
from tkinter import *
```

良好的代码：

```py
#======================
# imports
#======================
import tkinter as tk
from tkinter import ttk
```

它具有自然的流程，遵循小部件在 GUI 主窗体中的布局方式。

在意大利面代码中，底部的 LabelFrame 在顶部的 LabelFrame 之前创建，并且与导入语句和一些小部件创建混合在一起。

意大利面代码：

```py
import tkinter.scrolledtext
outputFrame=tkinter.ttk.LabelFrame(win,text=' Type into the scrolled text control: ')
scr=tkinter.scrolledtext.ScrolledText(outputFrame,width=scrolW,height=scrolH,wrap=tkinter.WORD)
e='E'
scr.grid(column=1,row=1,sticky=WE)
outputFrame.grid(column=0,row=2,sticky=e,padx=8)
lFrame=None
if y:chck2=tkinter.Checkbutton(lFrame,text="Enabled",variable=chVarUn)
wE='WE'
if y==x:PRINTME(x) 
lFrame=tkinter.ttk.LabelFrame(win,text="Spaghetti")
```

良好的代码：

```py
#=============================================================
# Adding a LabelFrame, Textbox (Entry) and Combobox  
#=============================================================
lFrame = ttk.LabelFrame(win, text="Python GUI Programming Cookbook")
lFrame.grid(column=0, row=0, sticky='WE', padx=10, pady=10)

#=============================================================
# Using a scrolled Text control    
#=============================================================
outputFrame = ttk.LabelFrame(win, text=' Type into the scrolled text control: ')
outputFrame.grid(column=0, row=2, sticky='E', padx=8)
```

它不包含不必要的变量赋值，也没有`print`函数，当阅读代码时，它不会做人们期望的调试。

意大利面代码：

```py
def PRINTME(me):print(me)
x=y=z=1
e='E'
WE='WE'
scr.grid(column=1,row=1,sticky=WE)
wE='WE'
if y==x:PRINTME(x) 
lFrame.grid(column=0,row=0,sticky=wE,padx=10,pady=10) 
PRINTME(z)
try: win.mainloop()
except:PRINTME(x)
chck2.deselect()
if y==x:PRINTME(x) 
```

良好的代码：

没有上述任何一种。

`import`语句只导入所需的模块。它们不会在整个代码中混乱。没有重复的`import`语句。没有`import *`语句。

意大利面代码：

```py
import tkinter 1
x=y=z=1
PRINTME(z) 
from tkinter import *
scrolW=30;scrolH=6
win=tkinter.Tk()
if x:chVarUn=tkinter.IntVar()
from tkinter import ttk
WE='WE'
import tkinter.scrolledtext
```

良好的代码：

```py
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
```

选择的变量名相当有意义。没有不必要使用数字`1`而不是`True`的`if`语句。

意大利面代码：

```py
x=y=z=1
if x:chVarUn=tkinter.IntVar()
wE='WE'
```

良好的代码：

```py
#=============================================================
# Using a scrolled Text control    
#=============================================================
outputFrame = ttk.LabelFrame(win, text=' Type into the scrolled text control: ')
outputFrame.grid(column=0, row=2, sticky='E', padx=8)
scrolW  = 30
scrolH  =  6
scr = scrolledtext.ScrolledText(outputFrame, width=scrolW, height=scrolH, wrap=tk.WORD)
scr.grid(column=1, row=0, sticky='WE')
```

我们没有失去预期的窗口标题，我们的复选框最终出现在正确的位置。我们还使包围复选框的“LabelFrame”可见。

意大利面代码：

我们失去了窗口标题，也没有显示顶部的“LabelFrame”。复选框最终出现在错误的位置。

良好的代码：

```py
#======================
# Create instance
#======================
win = tk.Tk()   

#======================
# Add a title       
#====================== 
win.title("Python GUI")

#=============================================================
# Adding a LabelFrame, Textbox (Entry) and Combobox  
#=============================================================
lFrame = ttk.LabelFrame(win, text="Python GUI Programming Cookbook")
lFrame.grid(column=0, row=0, sticky='WE', padx=10, pady=10)

#=============================================================
# Creating a checkbutton
#=============================================================
chVarUn = tk.IntVar()
check2 = tk.Checkbutton(lFrame, text="Enabled", variable=chVarUn)
check2.deselect()
check2.grid(column=1, row=4, sticky=tk.W, columnspan=3) 

#======================
# Start GUI
#======================
win.mainloop()
```

# 使用 __init__ 连接模块

当我们使用 Eclipse IDE 的 PyDev 插件创建一个新的 Python 项目时，它会自动创建一个`__init__.py`模块。当不使用 Eclipse 时，我们也可以手动创建它。

### 注意

`__init__.py`模块通常是空的，然后大小为 0 千字节。

我们可以使用这个通常为空的模块来连接不同的 Python 模块，通过在其中输入代码。这个配方将展示如何做到这一点。

## 准备就绪

我们将创建一个类似于我们在上一个配方中创建的 GUI 的新 GUI。

## 如何做...

随着我们的项目变得越来越大，我们自然地将其拆分为几个 Python 模块。使用现代 IDE（如 Eclipse）时，惊人地复杂，找到位于不同子文件夹中的模块，无论是在需要导入它的代码的上方还是下方。

绕过这个限制的一个实际方法是使用`__init__.py`模块。

### 注意

在 Eclipse 中，我们可以将 Eclipse 内部项目环境设置为某些文件夹，我们的 Python 代码将找到它。但是在 Eclipse 之外，例如在命令窗口中运行时，Python 模块导入机制有时会不匹配，代码将无法运行。

这是一个空的`__init__.py`模块的截图，当在 Eclipse 代码编辑器中打开时，它的名称不是`__init__`，而是属于的 PyDev 包的名称。代码编辑器左侧的“1”是行号，而不是在这个模块中编写的任何代码。这个空的`__init__.py`模块中绝对没有代码。

![如何做...](img/B04829_11_03.jpg)

这个文件是空的，但它确实存在。

![如何做...](img/B04829_11_04.jpg)

当我们运行以下代码并点击`clickMe Button`时，我们会得到代码后面显示的结果。这是一个常规的 Python 模块，尚未使用`__init__.py`模块。

### 注意

`__init__.py`模块与 Python 类的`__init__(self)`方法不同。

```py
#  Ch11_GUI__init.py
#======================
# imports
#======================
import tkinter as tk
from tkinter import ttk

#======================
# Create instance
#======================
win = tk.Tk()   

#======================
# Add a title       
#====================== 
win.title("Python GUI")

#=============================================================
# Adding a LabelFrame and a Button
#=============================================================
lFrame = ttk.LabelFrame(win, text="Python GUI Programming Cookbook")
lFrame.grid(column=0, row=0, sticky='WE', padx=10, pady=10)

def clickMe():
    from tkinter import messagebox
    messagebox.showinfo('Message Box', 'Hi from same Level.')

button = ttk.Button(lFrame, text="Click Me ", command=clickMe)
button.grid(column=1, row=0, sticky=tk.S)  

#======================
# Start GUI
#======================
win.mainloop()
```

![如何做...](img/B04829_11_05.jpg)

在上面的代码中，我们创建了以下函数，它导入 Python 的消息框，然后使用它来显示消息框窗口：

```py
def clickMe():
    from tkinter import messagebox
    messagebox.showinfo('Message Box', 'Hi from same Level.')
```

当我们将`clickMe()`消息框代码移动到嵌套的目录文件夹中，并尝试将其`import`到我们的 GUI 模块中时，我们遇到了一些问题。

我们在 Python 模块所在的位置下创建了三个子文件夹。然后，我们将`clickMe()`消息框代码放入一个新的 Python 模块中，我们将其命名为`MessageBox.py`。这个模块位于`Folder3`中，比我们的 Python 模块所在的位置低三级。

我们需要导入`MessageBox.py`模块，以便使用这个模块包含的`clickMe()`函数。

起初，它似乎可以工作，因为似乎我们可以导入新的嵌套模块，因为我们没有从 Eclipse IDE 中得到任何错误或警告。

我们使用 Python 的相对导入语法：

```py
from .Folder1.Folder2.Folder3.MessageBox import clickme
```

这可以在以下截图中看到：

![如何做...](img/B04829_11_06.jpg)

我们已经删除了本地的`clickMe()`函数，现在我们的回调应该使用导入的`clickMe()`函数，但它并没有按预期工作。我们运行代码时，没有得到预期的弹出窗口，而是得到了一个导入系统错误：

![如何做...](img/B04829_11_07.jpg)

我们可以通过转到 PyDev 项目属性并将自己添加为外部库，在 Eclipse 中将包含新函数的子文件夹作为外部库。这似乎并不直观，但它确实有效。

![如何做...](img/B04829_11_08.jpg)

现在，当我们注释掉文件夹结构，并直接从嵌套到三个级别的模块中导入函数时，代码会按预期工作。

```py
#======================
# imports
#======================
import tkinter as tk
from tkinter import ttk
# from .Folder1.Folder2.Folder3.MessageBox import clickMe
from MessageBox import clickMe
```

这个函数在消息框中显示不同的文本：

![如何做...](img/B04829_11_09.jpg)

实现相同结果的更好方法是使用 Python 内置的`__init__.py`模块。

删除之前特定于 Eclipse 的外部库依赖后，我们现在可以直接使用这个模块。

### 注意

我们将代码放入这个模块中，如果我们将`__init__.py`模块导入到我们的程序中，它将在我们的所有其他代码之前运行，截至 Python 3.4.3。

忽略 PyDev 未解析的导入（带有红色圈和交叉）错误。这个`import`是必要的；它使我们的代码运行，并且整个 Python 导入机制工作。

![如何做...](img/B04829_11_10.jpg)

将`__init__.py`模块导入到我们的程序后，我们可以使用它。检查它是否工作的第一个测试是在这个模块中编写一个打印语句。

![如何做...](img/B04829_11_11.jpg)

通过添加以下代码，我们可以以编程方式找出我们的位置：

![如何做...](img/B04829_11_12.jpg)

现在，我们可以通过向相同的`__init__.py`模块添加以下代码来从内部初始化我们的 Python 搜索路径：

```py
print('hi from GUI init\n')
from sys import path
from pprint import pprint
#=======================================================
# Required setup for the PYTONPATH in order to find
# all package folders
#=======================================================
from site import addsitedir
from os import getcwd, chdir, pardir
for _ in range(10):
    curFull = getcwd()
    curDir = curFull.split('\\')[-1] 
    if 'B04829_Ch11_Code' == curDir:
        addsitedir(curFull)
        addsitedir(curFull + '\\Folder1\\Folder2\\Folder3\\')
        break
    chdir(pardir)
pprint(path)
```

当我们现在运行我们的 GUI 代码时，我们得到了相同预期的窗口，但我们已经移除了对 Eclipse `PYTHONPATH`变量的依赖。

现在我们可以成功地在 Eclipse PyDev 插件之外运行相同的代码。

### 注意

我们的代码变得更加 Pythonic。

## 它是如何工作的...

在这个示例中，我们发现了使用 PyDev 插件的局限性，这个插件是免费的，与出色的免费 Eclipse IDE 一起提供。

我们首先在 Eclipse IDE 中找到了一个解决方法，然后通过变得 Pythonic 而独立于这个 IDE。

### 注意

通常使用纯 Python 是最好的方法。

# 混合下降和面向对象编码

Python 是一种面向对象的编程语言，但并不总是使用 OOP 是有意义的。对于简单的脚本任务，传统的瀑布式编码风格仍然是合适的。

在这个示例中，我们将创建一个新的 GUI，将下降式编码风格与更现代的 OOP 编码风格混合在一起。

我们将创建一个 OOP 风格的类，当我们在 Python GUI 中使用瀑布样式创建小部件时，它将在鼠标悬停在小部件上时显示工具提示。

### 注意

下降和瀑布式编码风格是相同的。这意味着我们必须在调用下面的代码之前将代码物理放置在上面的代码之上。在这种范式中，当我们执行代码时，代码从程序的顶部字面上下降到程序的底部。

## 准备工作

在这个示例中，我们将使用 tkinter 创建一个 GUI，这类似于我们在本书第一章中创建的 GUI。

## 如何做...

在 Python 中，我们可以通过使用`self`关键字将函数绑定到类，将它们转换为方法。这是 Python 的一个真正美妙的能力，它允许我们创建可理解和可维护的大型系统。

有时，当我们只编写简短的脚本时，使用 OOP 并没有意义，因为我们发现自己不得不用`self`关键字大量添加变量，当代码不需要时，代码会变得不必要地庞大。

让我们首先使用 tkinter 创建一个 Python GUI，并以瀑布式编码风格编写它。

以下代码创建了 GUI：

```py
#======================
# imports
#======================
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

#======================
# Create instance
#======================
win = tk.Tk()   

#======================
# Add a title       
#====================== 
win.title("Python GUI")

#=========================
# Disable resizing the GUI
#=========================
win.resizable(0,0)

#=============================================================
# Adding a LabelFrame, Textbox (Entry) and Combobox  
#=============================================================
lFrame = ttk.LabelFrame(win, text="Python GUI Programming Cookbook")
lFrame.grid(column=0, row=0, sticky='WE', padx=10, pady=10)

#=============================================================
# Labels
#=============================================================
ttk.Label(lFrame, text="Enter a name:").grid(column=0, row=0)
ttk.Label(lFrame, text="Choose a number:").grid(column=1, row=0, sticky=tk.W)

#=============================================================
# Buttons click command
#=============================================================
def clickMe(name, number):
    messagebox.showinfo('Information Message Box', 'Hello '+name+
                        ', your number is: ' + number)

#=============================================================
# Creating several controls in a loop
#=============================================================
names         = ['name0', 'name1', 'name2']
nameEntries   = ['nameEntry0', 'nameEntry1', 'nameEntry2']

numbers       = ['number0', 'number1', 'number2']
numberEntries = ['numberEntry0', 'numberEntry1', 'numberEntry2']

buttons = []

for idx in range(3):
    names[idx] = tk.StringVar()
    nameEntries[idx] = ttk.Entry(lFrame, width=12, textvariable=names[idx])
    nameEntries[idx].grid(column=0, row=idx+1)
    nameEntries[idx].delete(0, tk.END)
    nameEntries[idx].insert(0, '<name>')

    numbers[idx] = tk.StringVar()
    numberEntries[idx] = ttk.Combobox(lFrame, width=14, textvariable=numbers[idx])
    numberEntries[idx]['values'] = (1+idx, 2+idx, 4+idx, 42+idx, 100+idx)
    numberEntries[idx].grid(column=1, row=idx+1)
    numberEntries[idx].current(0)

    button = ttk.Button(lFrame, text="Click Me "+str(idx+1), command=lambda idx=idx: clickMe(names[idx].get(), numbers[idx].get()))
    button.grid(column=2, row=idx+1, sticky=tk.W)  
    buttons.append(button)
#======================
# Start GUI
#======================
win.mainloop()
```

当我们运行代码时，我们得到了 GUI，它看起来像这样：

![如何做...](img/B04829_11_13.jpg)

我们可以通过添加工具提示来改进我们的 Python GUI。这样做的最佳方式是将创建工具提示功能的代码与我们的 GUI 隔离开来。

我们通过创建一个具有工具提示功能的单独类来实现这一点，然后在创建 GUI 的同一 Python 模块中创建这个类的实例。

使用 Python，我们不需要将我们的`ToolTip`类放入一个单独的模块中。我们可以将它放在过程化代码的上面，然后从这段代码下面调用它。

代码现在看起来像这样：

```py
#======================
# imports
#======================
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

#-----------------------------------------------
class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

#-----------------------------------------------
def createToolTip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event): toolTip.showtip(text)
    def leave(event): toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

#-----------------------------------------------
# further down the module we call the createToolTip function
#-----------------------------------------------

for idx in range(3):
    names[idx] = tk.StringVar()
    nameEntries[idx] = ttk.Entry(
lFrame, width=12, textvariable=names[idx])
    nameEntries[idx].grid(column=0, row=idx+1)
    nameEntries[idx].delete(0, tk.END)
    nameEntries[idx].insert(0, '<name>')

    numbers[idx] = tk.StringVar()
    numberEntries[idx] = ttk.Combobox(
lFrame, width=14, textvariable=numbers[idx])
    numberEntries[idx]['values'] = (
1+idx, 2+idx, 4+idx, 42+idx, 100+idx)
    numberEntries[idx].grid(column=1, row=idx+1)
    numberEntries[idx].current(0)

    button = ttk.Button(
lFrame, text="Click Me "+str(idx+1), command=lambda idx=idx: clickMe(names[idx].get(), numbers[idx].get()))
    button.grid(column=2, row=idx+1, sticky=tk.W)  
    buttons.append(button)

#-----------------------------------------------
    # Add Tooltips to more widgets
    createToolTip(nameEntries[idx], 'This is an Entry widget.') 
    createToolTip(
numberEntries[idx], 'This is a DropDown widget.') 
    createToolTip(buttons[idx], 'This is a Button widget.')
#-----------------------------------------------
```

运行代码会在我们悬停鼠标在小部件上时为它们创建工具提示。

![如何做...](img/B04829_11_14.jpg)

## 它是如何工作的...

在这个示例中，我们以过程化的方式创建了一个 Python GUI，然后在模块的顶部添加了一个类。

我们可以很容易地在同一个 Python 模块中混合和匹配过程化和 OOP 编程。

# 使用代码命名约定

本书中以前的示例没有使用结构化的代码命名约定。这个示例将向您展示遵循代码命名方案的价值，因为它帮助我们找到我们想要扩展的代码，并提醒我们程序的设计。

## 准备工作

在这个示例中，我们将查看本书第一章中的 Python 模块名称，并将它们与更好的命名约定进行比较。

## 如何做...

在本书的第一章中，我们创建了我们的第一个 Python GUI。我们通过逐步增加不同的代码模块名称来改进我们的 GUI。

它看起来像这样：

![如何做...](img/B04829_11_15.jpg)

虽然这是一种典型的编码方式，但它并没有提供太多的意义。当我们在开发过程中编写 Python 代码时，很容易递增数字。

稍后回到这段代码时，我们不太清楚哪个 Python 模块提供了哪些功能，有时，我们最后增加的模块不如之前的版本好。

### 注意

清晰的命名约定确实有所帮助。

我们可以将第一章中的模块名称与第八章中的模块名称进行比较，后者更有意义。

![操作步骤...](img/B04829_11_16.jpg)

虽然不完美，但为不同的 Python 模块选择的名称表明了每个模块的责任。当我们想要添加更多单元测试时，清楚地知道它们位于哪个模块中。

以下是另一个示例，演示如何使用代码命名约定在 Python 中创建 GUI：

![操作步骤...](img/B04829_11_17.jpg)

### 注意

将单词`PRODUCT`替换为您当前正在开发的产品。

整个应用程序都是一个 GUI。所有部分都是相互连接的。`DEBUG.py`模块仅用于调试我们的代码。调用 GUI 的主要函数在与所有其他模块相比时，其名称是颠倒的。它以`Gui`开头，并以`.pyw`扩展名结尾。

它是唯一具有这个扩展名的 Python 模块。

根据这个命名约定，如果您对 Python 足够熟悉，很明显，要运行这个 GUI，您需要双击`Gui_PRODUCT.pyw`模块。

所有其他 Python 模块都包含为 GUI 提供功能并执行底层业务逻辑以实现此 GUI 目的的功能。

## 工作原理...

Python 代码模块的命名约定对于保持高效并记住我们最初的设计非常有帮助。当我们需要调试和修复缺陷或添加新功能时，它们是首要资源。

### 注意

通过数字递增模块名称并不是非常有意义，最终会浪费开发时间。

另一方面，命名 Python 变量更像是自由形式。Python 推断类型，因此我们不必指定变量将是`<list>`类型（它可能不是，或者实际上，在代码的后面部分，它可能会变成不同的类型）。

为变量命名的一个好主意是使它们具有描述性，并且不要缩写得太多。

如果我们希望指出某个变量设计为`<list>`类型，则使用完整单词`list`比使用`lst`更直观。

这与使用`number`而不是`num`类似。

在为变量命名时，使用非常描述性的名称是一个好主意，但有时可能会太长。在苹果的 Objective-C 语言中，一些变量和函数名字非常极端：`thisIsAMethodThatDoesThisAndThatAndAlsoThatIfYouPassInNIntegers:1:2:3`

### 注意

在为变量、方法和函数命名时要遵循常识。

# 何时不使用面向对象编程

Python 内置了面向对象编程的能力，但与此同时，我们也可以编写不需要使用面向对象编程的脚本。

对于某些任务，面向对象编程是没有意义的。

这个示例将展示何时不使用面向对象编程。

## 准备工作

在这个示例中，我们将创建一个类似于之前示例的 Python GUI。我们将比较面向对象编程的代码和非面向对象的替代编程方式。

## 如何做...

让我们首先使用**OOP**方法创建一个新的 GUI。以下代码将创建下面代码中显示的 GUI：

```py
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import Menu

class OOP():
    def __init__(self): 
        self.win = tk.Tk()         
        self.win.title("Python GUI")      
        self.createWidgets()

    def createWidgets(self):    
        tabControl = ttk.Notebook(self.win)     
        tab1 = ttk.Frame(tabControl)            
        tabControl.add(tab1, text='Tab 1')    
        tabControl.pack(expand=1, fill="both")  
        self.monty = ttk.LabelFrame(tab1, text=' Monty Python ')
        self.monty.grid(column=0, row=0, padx=8, pady=4)        

        ttk.Label(self.monty, text="Enter a name:").grid(
column=0, row=0, sticky='W')
        self.name = tk.StringVar()
        nameEntered = ttk.Entry(
self.monty, width=12, textvariable=self.name)
        nameEntered.grid(column=0, row=1, sticky='W')

        self.action = ttk.Button(self.monty, text="Click Me!")   
        self.action.grid(column=2, row=1)

        ttk.Label(self.monty, 
text="Choose a number:").grid(column=1, row=0)
        number = tk.StringVar()
        numberChosen = ttk.Combobox(self.monty, 
width=12, textvariable=number)
        numberChosen['values'] = (42)
        numberChosen.grid(column=1, row=1)
        numberChosen.current(0)

        scrolW = 30; scrolH = 3
        self.scr = scrolledtext.ScrolledText(
self.monty, width=scrolW, height=scrolH, wrap=tk.WORD)
        self.scr.grid(column=0, row=3, sticky='WE', columnspan=3)

        menuBar = Menu(tab1)
        self.win.config(menu=menuBar)
        fileMenu = Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label="File", menu=fileMenu)
        helpMenu = Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label="Help", menu=helpMenu)

        nameEntered.focus()     
#==========================
oop = OOP()
oop.win.mainloop()
```

![操作步骤...](img/B04829_11_18.jpg)

我们可以通过稍微重构我们的代码来实现相同的 GUI，而不使用面向对象的方法。首先，我们移除`OOP`类及其`__init__`方法。

接下来，我们将所有方法移到左侧，并移除`self`类引用，将它们转换为未绑定的函数。

我们还删除了先前代码中的任何其他`self`引用。然后，我们将`createWidgets`函数调用移动到函数声明点下方。我们将它放在`mainloop`调用的正上方。

最终，我们实现了相同的 GUI，但没有使用 OOP。

重构后的代码如下所示：

```py
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import Menu   

def createWidgets():    
    tabControl = ttk.Notebook(win)     
    tab1 = ttk.Frame(tabControl)            
    tabControl.add(tab1, text='Tab 1')    
    tabControl.pack(expand=1, fill="both")  
    monty = ttk.LabelFrame(tab1, text=' Monty Python ')
    monty.grid(column=0, row=0, padx=8, pady=4)        

    ttk.Label(monty, text="Enter a name:").grid(
column=0, row=0, sticky='W')
    name = tk.StringVar()
    nameEntered = ttk.Entry(monty, width=12, textvariable=name)
    nameEntered.grid(column=0, row=1, sticky='W')

    action = ttk.Button(monty, text="Click Me!")   
    action.grid(column=2, row=1)

    ttk.Label(monty, text="Choose a number:").grid(
column=1, row=0)
    number = tk.StringVar()
    numberChosen = ttk.Combobox(
monty, width=12, textvariable=number)
    numberChosen['values'] = (42)
    numberChosen.grid(column=1, row=1)
    numberChosen.current(0)

    scrolW = 30; scrolH = 3
    scr = scrolledtext.ScrolledText(
monty, width=scrolW, height=scrolH, wrap=tk.WORD)
    scr.grid(column=0, row=3, sticky='WE', columnspan=3)

    menuBar = Menu(tab1)
    win.config(menu=menuBar)
    fileMenu = Menu(menuBar, tearoff=0)
    menuBar.add_cascade(label="File", menu=fileMenu)
    helpMenu = Menu(menuBar, tearoff=0)
    menuBar.add_cascade(label="Help", menu=helpMenu)

    nameEntered.focus()     
#======================
win = tk.Tk()         
win.title("Python GUI")   
createWidgets()
win.mainloop()
```

## 它是如何工作的...

Python 使我们能够在有意义的时候使用 OOP。其他语言如 Java 和 C#强制我们始终使用 OOP 方法进行编码。在这个示例中，我们探讨了一个不适合使用 OOP 的情况。

### 注意

如果代码库增长，OOP 方法将更具扩展性，但是如果确定只需要这个代码，那么就没有必要经过 OOP。

# 成功使用设计模式的方法

在这个示例中，我们将使用工厂设计模式为我们的 Python GUI 创建小部件。

在以前的示例中，我们要么手动创建小部件，要么在循环中动态创建小部件。

使用工厂设计模式，我们将使用工厂来创建我们的小部件。

## 准备工作

我们将创建一个 Python GUI，其中有三个按钮，每个按钮都有不同的样式。

## 如何做...

在我们的 Python GUI 模块顶部，在导入语句的下方，我们创建了几个类：

```py
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import Menu

class ButtonFactory():
    def createButton(self, type_):
        return buttonTypes[type_]()

class ButtonBase():     
    relief     ='flat'
    foreground ='white'
    def getButtonConfig(self):
        return self.relief, self.foreground

class ButtonRidge(ButtonBase):
    relief     ='ridge'
    foreground ='red'        

class ButtonSunken(ButtonBase):
    relief     ='sunken'
    foreground ='blue'        

class ButtonGroove(ButtonBase):
    relief     ='groove'
    foreground ='green'        

buttonTypes = [ButtonRidge, ButtonSunken, ButtonGroove] 

class OOP():
    def __init__(self): 
        self.win = tk.Tk()         
        self.win.title("Python GUI")      
        self.createWidgets()
```

我们创建一个基类，我们的不同按钮样式类都继承自该基类，并且每个类都覆盖了`relief`和`foreground`配置属性。所有子类都从这个基类继承`getButtonConfig`方法。该方法返回一个元组。

我们还创建了一个按钮工厂类和一个保存我们按钮子类名称的列表。我们将列表命名为`buttonTypes`，因为我们的工厂将创建不同类型的按钮。

在模块的下方，我们使用相同的`buttonTypes`列表创建按钮小部件。

```py
    def createButtons(self):

        factory = ButtonFactory()

        # Button 1
        rel = factory.createButton(0).getButtonConfig()[0]
        fg  = factory.createButton(0).getButtonConfig()[1]
        action = tk.Button(self.monty, 
text="Button "+str(0+1), relief=rel, foreground=fg)   
        action.grid(column=0, row=1)  

        # Button 2
        rel = factory.createButton(1).getButtonConfig()[0]
        fg  = factory.createButton(1).getButtonConfig()[1]
        action = tk.Button(self.monty, 
text="Button "+str(1+1), relief=rel, foreground=fg)   
        action.grid(column=1, row=1)  

        # Button 3
        rel = factory.createButton(2).getButtonConfig()[0]
        fg  = factory.createButton(2).getButtonConfig()[1]
        action = tk.Button(self.monty, 
text="Button "+str(2+1), relief=rel, foreground=fg)   
       action.grid(column=2, row=1)   
```

首先，我们创建一个按钮工厂的实例，然后我们使用我们的工厂来创建我们的按钮。

### 注意

`buttonTypes`列表中的项目是我们子类的名称。

我们调用`createButton`方法，然后立即调用基类的`getButtonConfig`方法，并使用点表示法检索配置属性。

当我们运行整个代码时，我们会得到以下 Python tkinter GUI：

![如何做...](img/B04829_11_19.jpg)

我们可以看到，我们的 Python GUI 工厂确实创建了不同的按钮，每个按钮都有不同的样式。它们在文本颜色和 relief 属性上有所不同。

## 它是如何工作的...

在这个示例中，我们使用工厂设计模式创建了几个具有不同样式的小部件。我们可以轻松地使用这种设计模式来创建整个 GUI。

设计模式是我们软件开发工具箱中非常令人兴奋的工具。

# 避免复杂性

在这个示例中，我们将扩展我们的 Python GUI，并学习处理软件开发工作不断增加的复杂性的方法。

我们的同事和客户喜欢我们用 Python 创建的 GUI，并要求为我们的 GUI 添加越来越多的功能。

这增加了复杂性，很容易破坏我们最初的良好设计。

## 准备工作

我们将创建一个类似于之前示例中的新 Python GUI，并将以小部件的形式添加许多功能。

## 如何做...

我们将从一个具有两个选项卡并且看起来像这样的 Python GUI 开始：

![如何做...](img/B04829_11_20.jpg)

我们收到的第一个新功能请求是为**Tab 1**添加功能，清除`scrolledtext`小部件。

足够简单。我们只需在**Tab 1**中添加另一个按钮。

```py
        # Adding another Button
        self.action = ttk.Button(.
self.monty, text="Clear Text", command=self.clearScrol)   
        self.action.grid(column=2, row=2)
```

我们还必须创建回调方法以添加所需的功能，我们在类的顶部定义它，并在创建小部件的方法之外。

```py
    # Button callback
    def clickMe(self):
        self.action.configure(text='Hello ' + self.name.get())

    # Button callback Clear Text   
    def clearScrol(self):
        self.scr.delete('1.0', tk.END)
```

现在我们的 GUI 有一个新按钮，当我们点击它时，我们清除`ScrolledText`小部件的文本。

![如何做...](img/B04829_11_21.jpg)

为了添加这个功能，我们不得不在同一个 Python 模块中的两个地方添加代码。

我们在`createWidgets`方法中插入了新按钮（未显示），然后我们创建了一个新的回调方法，当我们的新按钮被点击时调用。我们将这段代码放在第一个按钮的回调之下。

我们的下一个功能请求是添加更多功能。业务逻辑封装在另一个 Python 模块中。我们通过向**Tab 1**添加三个按钮来调用这个新功能。我们使用循环来实现这一点。

```py
        # Adding more Feature Buttons
        for idx in range(3):
            b = ttk.Button(
self.monty, text="Feature" + str(idx+1))   
            b.grid(column=idx, row=4)
```

我们的 GUI 现在看起来是这样的：

![如何做...](img/B04829_11_22.jpg)

接下来，我们的客户要求更多功能，我们使用相同的方法。我们的 GUI 现在看起来是这样的：

![如何做...](img/B04829_11_23.jpg)

### 注意

这并不太糟糕。当我们为另外 50 个新功能请求时，我们开始怀疑我们的方法是否仍然是最好的方法...

处理我们的 GUI 必须处理的不断增加的复杂性的一种方法是添加选项卡。通过添加更多的选项卡，并将相关功能放入自己的选项卡中，我们可以控制复杂性，并使我们的 GUI 更直观。

这是创建我们的新**Tab 3**的代码，下面是我们的新 Python GUI：

```py
# Tab Control 3  -----------------------------------------
        tab3 = ttk.Frame(tabControl)          # Add a tab
        tabControl.add(tab3, text='Tab 3')    # Make tab visible

        monty3 = ttk.LabelFrame(tab3, text=' New Features ')
        monty3.grid(column=0, row=0, padx=8, pady=4)

        # Adding more Feature Buttons
        startRow = 4
        for idx in range(24):
            if idx < 2:
                colIdx = idx
                col = colIdx
            else:
                col += 1
            if not idx % 3: 
                startRow += 1
                col = 0

            b = ttk.Button(monty3, text="Feature " + str(idx+1))
            b.grid(column=col, row=startRow)    

        # Add some space around each label
        for child in monty3.winfo_children(): 
            child.grid_configure(padx=8)
```

![如何做...](img/B04829_11_24.jpg)

## 它是如何工作的...

在这个示例中，我们向我们的 GUI 添加了几个新的小部件，以便为我们的 Python GUI 添加更多功能。我们看到，越来越多的新功能请求很容易使我们精美的 GUI 设计变得不太清楚如何使用 GUI。

### 注意

突然间，小部件占据了世界...

我们看到了如何通过将大功能分解为较小的部分，并将它们安排在功能相关的区域中，通过模块化我们的 GUI 来处理复杂性。

尽管复杂性有许多方面，但模块化和重构代码通常是处理软件代码复杂性的非常好的方法。

### 注意

在编程中，有时候我们会遇到障碍，陷入困境。我们不断地撞击这堵墙，但什么也没有发生。

有时候我们觉得想要放弃。

然而，奇迹确实会发生...

如果我们继续撞击这堵墙，在某个时刻，这堵墙将倒塌，道路将会开放。

在那个时候，我们可以在软件宇宙中留下积极的印记。
