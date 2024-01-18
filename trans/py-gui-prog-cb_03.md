# 第3章。外观定制

在本章中，我们将使用Python 3自定义我们的GUI：

+   创建消息框-信息、警告和错误

+   如何创建独立的消息框

+   如何创建tkinter窗体的标题

+   更改主根窗口的图标

+   使用旋转框控件

+   小部件的浮雕、凹陷和凸起外观

+   使用Python创建工具提示

+   如何使用画布小部件

# 介绍

在本章中，我们将通过更改一些属性来自定义GUI中的一些小部件。我们还介绍了一些tkinter提供给我们的新小部件。

*使用Python创建工具提示*示例将创建一个ToolTip面向对象的类，它将成为我们到目前为止一直在使用的单个Python模块的一部分。

# 创建消息框-信息、警告和错误

消息框是一个弹出窗口，向用户提供反馈。它可以是信息性的，暗示潜在问题，甚至是灾难性的错误。

使用Python创建消息框非常容易。

## 准备工作

我们将为上一个示例中创建的“帮助”|“关于”菜单项添加功能。在大多数应用程序中，单击“帮助”|“关于”菜单时向用户提供的典型反馈是信息性的。我们从这个信息开始，然后变化设计模式以显示警告和错误。

## 如何做...

将以下代码添加到导入语句所在的模块顶部：

```py
from tkinter import messagebox as mBox
```

接下来，我们将创建一个回调函数来显示一个消息框。我们必须将回调的代码放在我们将回调附加到菜单项的代码上面，因为这仍然是过程性的而不是面向对象的代码。

将此代码添加到创建帮助菜单的行的上方：

```py
# Display a Message Box
# Callback function
def _msgBox():
    mBox.showinfo('Python Message Info Box', 'A Python GUI created using tkinter:\nThe year is 2015.')   

# Add another Menu to the Menu Bar and an item
helpMenu = Menu(menuBar, tearoff=0)
helpMenu.add_command(label="About", command=_msgBox)
```

现在单击“帮助”|“关于”会导致以下弹出窗口出现：

![操作步骤...](graphics/B04829_03_01.jpg)

让我们将这段代码转换为警告消息框弹出窗口。注释掉上一行并添加以下代码：

```py
# Display a Message Box
def _msgBox():
#    mBox.showinfo('Python Message Info Box', 'A Python GUI 
#      created using tkinter:\nThe year is 2015.')
    mBox.showwarning('Python Message Warning Box', 'A Python GUI created using tkinter:\nWarning: There might be a bug in this code.')
```

运行上面的代码现在会导致以下略微修改的消息框出现：

![操作步骤...](graphics/B04829_03_02.jpg)

显示错误消息框很简单，通常警告用户存在严重问题。如上所述，注释掉并添加此代码，如我们在这里所做的：

```py
# Display a Message Box
def _msgBox():
#    mBox.showinfo('Python Message Info Box', 'A Python GUI 
#      created using tkinter:\nThe year is 2015.')
#    mBox.showwarning('Python Message Warning Box', 'A Python GUI 
#      created using tkinter:\nWarning: There might be a bug in 
#      this code.')
    mBox.showerror('Python Message Error Box', 'A Python GUI created using tkinter:\nError: Houston ~ we DO have a serious PROBLEM!')
```

![操作步骤...](graphics/B04829_03_03.jpg)

## 工作原理...

我们添加了另一个回调函数，并将其附加为处理单击事件的委托。现在，当我们单击“帮助”|“关于”菜单时，会发生一个动作。我们正在创建和显示最常见的弹出式消息框对话框。它们是模态的，因此用户在点击“确定”按钮之前无法使用GUI。

在第一个示例中，我们显示了一个信息框，可以看到其左侧的图标。接下来，我们创建警告和错误消息框，它们会自动更改与弹出窗口关联的图标。我们只需指定要显示哪个mBox。

有不同的消息框显示多个“确定”按钮，我们可以根据用户的选择来编程我们的响应。

以下是一个简单的例子，说明了这种技术：

```py
# Display a Message Box
def _msgBox():
    answer = mBox.askyesno("Python Message Dual Choice Box", "Are you sure you really wish to do this?")
    print(answer)
```

运行此GUI代码会导致弹出一个用户响应可以用来分支的窗口，通过将其保存在“answer”变量中来驱动此事件驱动的GUI循环的答案。

![工作原理...](graphics/B04829_03_04.jpg)

在Eclipse中使用控制台输出显示，单击“是”按钮会导致将布尔值“True”分配给“answer”变量。

![工作原理...](graphics/B04829_03_05.jpg)

例如，我们可以使用以下代码：

```py
If answer == True:
    <do something>
```

# 如何创建独立的消息框

在这个示例中，我们将创建我们的tkinter消息框作为独立的顶层GUI窗口。

我们首先注意到，这样做会多出一个窗口，因此我们将探索隐藏此窗口的方法。

在上一个示例中，我们通过我们主GUI表单中的“帮助”|“关于”菜单调用了tkinter消息框。

那么为什么我们希望创建一个独立的消息框呢？

一个原因是我们可能会自定义我们的消息框，并在我们的GUI中重用它们。我们可以将它们从我们的主GUI代码中分离出来，而不是在我们设计的每个Python GUI中复制和粘贴相同的代码。这可以创建一个小的可重用组件，然后我们可以将其导入到不同的Python GUI中。

## 准备工作

我们已经在上一个食谱中创建了消息框的标题。我们不会重用上一个食谱中的代码，而是会用很少的Python代码构建一个新的GUI。

## 如何做...

我们可以像这样创建一个简单的消息框：

```py
from tkinter import messagebox as mBox
mBox.showinfo('A Python GUI created using tkinter:\nThe year is 2015')
```

这将导致这两个窗口：

![如何做...](graphics/B04829_03_06.jpg)

这看起来不像我们想象的那样。现在我们有两个窗口，一个是不需要的，第二个是其文本显示为标题。

哎呀。

现在让我们来修复这个问题。我们可以通过添加一个单引号或双引号，后跟一个逗号来更改Python代码。

```py
mBox.showinfo('', 'A Python GUI created using tkinter:\nThe year is 2015')
```

![如何做...](graphics/B04829_03_07.jpg)

第一个参数是标题，第二个是弹出消息框中显示的文本。通过添加一对空的单引号或双引号，后跟一个逗号，我们可以将我们的文本从标题移到弹出消息框中。

我们仍然需要一个标题，而且我们肯定想摆脱这个不必要的第二个窗口。

### 注意

在像C#这样的语言中，会出现第二个窗口的相同现象。基本上是一个DOS风格的调试窗口。许多程序员似乎不介意有这个额外的窗口漂浮。从GUI编程的角度来看，我个人觉得这很不雅。我们将在下一步中删除它。

第二个窗口是由Windows事件循环引起的。我们可以通过抑制它来摆脱它。

添加以下代码：

```py
from tkinter import messagebox as mBox
from tkinter import Tk
root = Tk()
root.withdraw()
mBox.showinfo('', 'A Python GUI created using tkinter:\nThe year is 2015')
```

现在我们只有一个窗口。`withdraw()`函数移除了我们不希望漂浮的调试窗口。

![如何做...](graphics/B04829_03_08.jpg)

为了添加标题，我们只需将一些字符串放入我们的空第一个参数中。

例如：

```py
from tkinter import messagebox as mBox
from tkinter import Tk
root = Tk()
root.withdraw()
mBox.showinfo('This is a Title', 'A Python GUI created using tkinter:\nThe year is 2015')
```

现在我们的对话框有了标题：

![如何做...](graphics/B04829_03_09.jpg)

## 它是如何工作的...

我们将更多参数传递给消息框的tkinter构造函数，以添加窗体的标题并在消息框中显示文本，而不是将其显示为标题。这是由于我们传递的参数的位置。如果我们省略空引号或双引号，那么消息框小部件将把第一个参数的位置作为标题，而不是要在消息框中显示的文本。通过传递一个空引号后跟一个逗号，我们改变了消息框显示我们传递给函数的文本的位置。

我们通过在我们的主根窗口上调用`withdraw()`方法来抑制tkinter消息框小部件自动创建的第二个弹出窗口。

# 如何创建tkinter窗体的标题

更改tkinter主根窗口的标题的原则与前一个食谱中讨论的原则相同。我们只需将一个字符串作为小部件的构造函数的第一个参数传递进去。

## 准备工作

与弹出对话框窗口不同，我们创建主根窗口并给它一个标题。

在这个食谱中显示的GUI是上一章的代码。它不是在本章中基于上一个食谱构建的。

## 如何做...

以下代码创建了主窗口并为其添加了标题。我们已经在以前的食谱中做过这个。在这里，我们只关注GUI的这个方面。

```py
import tkinter as tk
win = tk.Tk()               # Create instance
win.title("Python GUI")     # Add a title
```

![如何做...](graphics/B04829_03_10.jpg)

## 它是如何工作的...

通过使用内置的tkinter `title` 属性，为主根窗口添加标题。在创建`Tk()`实例后，我们可以使用所有内置的tkinter属性来自定义我们的GUI。

# 更改主根窗口的图标

自定义GUI的一种方法是给它一个与tkinter默认图标不同的图标。下面是我们如何做到这一点。

## 准备工作

我们正在改进上一个配方的GUI。我们将使用一个随Python一起提供的图标，但您可以使用任何您认为有用的图标。确保您在代码中有图标所在的完整路径，否则可能会出错。

### 注意

虽然可能会有点混淆，上一章的这个配方指的是哪个配方，最好的方法就是只下载本书的代码，然后逐步执行代码以理解它。

## 如何做...

将以下代码放在主事件循环的上方某处。示例使用了我安装Python 3.4的路径。您可能需要调整它以匹配您的安装目录。

请注意GUI左上角的“feather”默认图标已更改。

```py
# Change the main windows icon
win.iconbitmap(r'C:\Python34\DLLs\pyc.ico')
```

![如何做...](graphics/B04829_03_11.jpg)

## 它是如何工作的...

这是另一个与Python 3.x一起提供的tkinter的属性。 `iconbitmap`是我们使用的属性，通过传递图标的绝对（硬编码）路径来改变主根窗口的图标。这将覆盖tkinter的默认图标，用我们选择的图标替换它。

### 注意

在上面的代码中，使用绝对路径的字符串中的“r”来转义反斜杠，因此我们可以使用“raw”字符串，而不是写`C:\\`，这让我们可以写更自然的单个反斜杠`C:\`。这是Python为我们创建的一个巧妙的技巧。

# 使用旋转框控件

在这个示例中，我们将使用`Spinbox`小部件，并且还将绑定键盘上的*Enter*键到我们的小部件之一。

## 准备工作

我们正在使用我们的分页GUI，并将在`ScrolledText`控件上方添加一个`Spinbox`小部件。这只需要我们将`ScrolledText`行值增加一，并在`Entry`小部件上面的行中插入我们的新`Spinbox`控件。

## 如何做...

首先，我们添加了`Spinbox`控件。将以下代码放在`ScrolledText`小部件上方：

```py
# Adding a Spinbox widget
spin = Spinbox(monty, from_=0, to=10)
spin.grid(column=0, row=2)
```

这将修改我们的GUI，如下所示：

![如何做...](graphics/B04829_03_12.jpg)

接下来，我们将减小`Spinbox`小部件的大小。

```py
spin = Spinbox(monty, from_=0, to=10, width=5)
```

![如何做...](graphics/B04829_03_13.jpg)

接下来，我们添加另一个属性来进一步自定义我们的小部件，`bd`是`borderwidth`属性的简写表示。

```py
spin = Spinbox(monty, from_=0, to=10, width=5 , bd=8)
```

![如何做...](graphics/B04829_03_14.jpg)

在这里，我们通过创建回调并将其链接到控件来为小部件添加功能。

这将把Spinbox的选择打印到`ScrolledText`中，也打印到标准输出。名为`scr`的变量是我们对`ScrolledText`小部件的引用。

```py
# Spinbox callback 
def _spin():
    value = spin.get()
    print(value)
    scr.insert(tk.INSERT, value + '\n')

spin = Spinbox(monty, from_=0, to=10, width=5, bd=8, command=_spin)
```

![如何做...](graphics/B04829_03_15.jpg)

除了使用范围，我们还可以指定一组值。

```py
# Adding a Spinbox widget using a set of values
spin = Spinbox(monty, values=(1, 2, 4, 42, 100), width=5, bd=8, command=_spin) 
spin.grid(column=0, row=2)
```

这将创建以下GUI输出：

![如何做...](graphics/B04829_03_16.jpg)

## 它是如何工作的...

请注意，在第一个屏幕截图中，我们的新`Spinbox`控件默认为宽度20，推出了此列中所有控件的列宽。这不是我们想要的。我们给小部件一个从0到10的范围，并且默认显示`to=10`值，这是最高值。如果我们尝试将`from_/to`范围从10到0反转，tkinter不会喜欢。请自行尝试。

在第二个屏幕截图中，我们减小了`Spinbox`控件的宽度，这使其与列的中心对齐。

在第三个屏幕截图中，我们添加了Spinbox的`borderwidth`属性，这自动使整个`Spinbox`看起来不再是平的，而是三维的。

在第四个屏幕截图中，我们添加了一个回调函数，以显示在`ScrolledText`小部件中选择的数字，并将其打印到标准输出流中。我们添加了“\n”以打印在新行上。请注意默认值不会被打印。只有当我们单击控件时，回调函数才会被调用。通过单击默认为10的向上箭头，我们可以打印值“10”。

最后，我们将限制可用的值为硬编码集。这也可以从数据源（例如文本或XML文件）中读取。

# 小部件的Relief、sunken和raised外观

我们可以通过一个属性来控制`Spinbox`小部件的外观，使它们看起来是凸起的、凹陷的，或者是凸起的格式。

## 准备工作

我们将添加一个`Spinbox`控件来演示`Spinbox`控件的`relief`属性的可用外观。

## 如何做...

首先，让我们增加`borderwidth`以区分我们的第二个`Spinbox`和第一个`Spinbox`。

```py
# Adding a second Spinbox widget 
spin = Spinbox(monty, values=(0, 50, 100), width=5, bd=20, command=_spin) 
spin.grid(column=1, row=2)
```

![如何做...](graphics/B04829_03_17.jpg)

我们上面的两个Spinbox小部件具有相同的`relief`样式。唯一的区别是，我们右边的新小部件的边框宽度要大得多。

在我们的代码中，我们没有指定使用哪个`relief`属性，所以`relief`默认为`tk.SUNKEN`。

以下是可以设置的可用`relief`属性选项：

| tk.SUNKEN | tk.RAISED | tk.FLAT | tk.GROOVE | tk.RIDGE |

通过将不同的可用选项分配给`relief`属性，我们可以为这个小部件创建不同的外观。

将`tk.RIDGE`的`relief`属性分配给它，并将边框宽度减小到与我们的第一个`Spinbox`小部件相同的值，结果如下GUI所示：

![如何做...](graphics/B04829_03_18.jpg)

## 它是如何工作的...

首先，我们创建了一个第二个`Spinbox`，对齐到第二列（索引==1）。它默认为`SUNKEN`，所以它看起来类似于我们的第一个`Spinbox`。我们通过增加第二个控件（右边的控件）的边框宽度来区分这两个小部件。

接下来，我们隐式地设置了`Spinbox`小部件的`relief`属性。我们使边框宽度与我们的第一个`Spinbox`相同，因为通过给它一个不同的`relief`，不需要改变任何其他属性，差异就变得可见了。

# 使用Python创建工具提示

这个示例将向我们展示如何创建工具提示。当用户将鼠标悬停在小部件上时，将以工具提示的形式提供额外的信息。

我们将把这些额外的信息编码到我们的GUI中。

## 准备工作

我们正在为我们的GUI添加更多有用的功能。令人惊讶的是，向我们的控件添加工具提示应该很简单，但实际上并不像我们希望的那样简单。

为了实现这种期望的功能，我们将把我们的工具提示代码放入自己的面向对象编程类中。

## 如何做...

在导入语句的下面添加这个类：

```py
class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, _cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 27
        y = y + cy + self.widget.winfo_rooty() +27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))

        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
   background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))

        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

#===========================================================
def createToolTip( widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)
```

在**面向对象编程**（**OOP**）方法中，我们在Python模块中创建一个新的类。Python允许我们将多个类放入同一个Python模块中，并且还可以在同一个模块中“混合和匹配”类和常规函数。

上面的代码正在做这个。

`ToolTip`类是一个Python类，为了使用它，我们必须实例化它。

如果你不熟悉面向对象的编程，“实例化一个对象以创建类的实例”可能听起来相当无聊。

这个原则非常简单，非常类似于通过`def`语句创建一个Python函数，然后在代码中稍后调用这个函数。

以非常相似的方式，我们首先创建一个类的蓝图，并通过在类的名称后面添加括号将其分配给一个变量：

```py
class AClass():
    pass
instanceOfAClass = AClass()
print(instanceOfAClass)
```

上面的代码打印出一个内存地址，并且显示我们的变量现在引用了这个类实例。

面向对象编程的很酷的一点是，我们可以创建同一个类的许多实例。

在我们之前的代码中，我们声明了一个Python类，并明确地让它继承自所有Python类的基础对象。我们也可以将其省略，就像我们在`AClass`代码示例中所做的那样，因为它是所有Python类的默认值。

在`ToolTip`类中发生的所有必要的工具提示创建代码之后，我们接下来转到非面向对象的Python编程，通过在其下方创建一个函数。

我们定义了函数`createToolTip()`，它期望我们的GUI小部件之一作为参数传递进来，这样当我们将鼠标悬停在这个控件上时，我们就可以显示一个工具提示。

`createToolTip()`函数实际上为我们为每个调用它的小部件创建了`ToolTip`类的一个新实例。

我们可以为我们的Spinbox小部件添加一个工具提示，就像这样：

```py
# Add a Tooltip
createToolTip(spin, 'This is a Spin control.')
```

以及我们所有其他GUI小部件的方式完全相同。我们只需传入我们希望显示一些额外信息的工具提示的小部件的父级。对于我们的ScrolledText小部件，我们使变量`scr`指向它，所以这就是我们传递给我们的ToolTip创建函数构造函数的内容。

```py
# Using a scrolled Text control    
scrolW  = 30; scrolH  =  3
scr = scrolledtext.ScrolledText(monty, width=scrolW, height=scrolH, wrap=tk.WORD)
scr.grid(column=0, row=3, sticky='WE', columnspan=3)

# Add a Tooltip to the ScrolledText widget
createToolTip(scr, 'This is a ScrolledText widget.')
```

## 它是如何工作的...

这是本书中面向对象编程的开始。这可能看起来有点高级，但不用担心，我们会解释一切，它确实有效！

嗯，实际上运行这段代码并没有起作用，也没有任何区别。

在创建微调器之后，添加以下代码：

```py
# Add a Tooltip
createToolTip(spin, 'This is a Spin control.')
```

现在，当我们将鼠标悬停在微调小部件上时，我们会得到一个工具提示，为用户提供额外的信息。

![它是如何工作的...](graphics/B04829_03_19.jpg)

我们调用创建工具提示的函数，然后将对小部件的引用和我们希望在悬停鼠标在小部件上时显示的文本传递进去。

本书中的其余示例将在合适的情况下使用面向对象编程。在这里，我们展示了可能的最简单的面向对象编程示例。作为默认，我们创建的每个Python类都继承自`object`基类。作为一个真正的务实的编程语言，Python简化了类的创建过程。

我们可以写成这样：

```py
class ToolTip(object):
    pass
```

我们也可以通过省略默认的基类来简化它：

```py
class ToolTip():
    pass
```

在同样的模式中，我们可以继承和扩展任何tkinter类。

# 如何使用画布小部件

这个示例展示了如何通过使用tkinter画布小部件为我们的GUI添加戏剧性的颜色效果。

## 准备工作

通过为其添加更多的颜色，我们将改进我们先前的代码和GUI的外观。

## 如何做...

首先，我们将在我们的GUI中创建第三个选项卡，以便隔离我们的新代码。

以下是创建新的第三个选项卡的代码：

```py
# Tab Control introduced here --------------------------------
tabControl = ttk.Notebook(win)          # Create Tab Control

tab1 = ttk.Frame(tabControl)            # Create a tab 
tabControl.add(tab1, text='Tab 1')      # Add the tab

tab2 = ttk.Frame(tabControl)            # Add a second tab
tabControl.add(tab2, text='Tab 2')      # Make second tab visible

tab3 = ttk.Frame(tabControl)            # Add a third tab
tabControl.add(tab3, text='Tab 3')      # Make second tab visible

tabControl.pack(expand=1, fill="both")  # Pack to make visible
# ~ Tab Control introduced here -------------------------------
```

接下来，我们使用tkinter的另一个内置小部件，即画布。很多人喜欢这个小部件，因为它具有强大的功能。

```py
# Tab Control 3 -------------------------------
tab3 = tk.Frame(tab3, bg='blue')
tab3.pack()
for orangeColor in range(2):
    canvas = tk.Canvas(tab3, width=150, height=80, highlightthickness=0, bg='orange')
    canvas.grid(row=orangeColor, column=orangeColor)
```

## 它是如何工作的...

以下屏幕截图显示了通过运行上述代码并单击新的**Tab 3**创建的结果。当你运行代码时，它真的是橙色和蓝色的。在这本无色的书中，这可能不太明显，但这些颜色是真实的；你可以相信我。

您可以通过在线搜索来查看绘图和绘制功能。我不会在这本书中深入探讨这个小部件（但它确实很酷）。

![它是如何工作的...](graphics/B04829_03_20.jpg)
