# 第一章。创建 GUI 表单并添加小部件

在本章中，我们开始使用 Python 3 创建令人惊叹的 GUI：

+   创建我们的第一个 Python GUI

+   防止 GUI 大小调整

+   将标签添加到 GUI 表单

+   创建按钮并更改其文本属性

+   文本框小部件

+   将焦点设置为小部件并禁用小部件

+   组合框小部件

+   创建具有不同初始状态的复选按钮

+   使用单选按钮小部件

+   使用滚动文本小部件

+   在循环中添加多个小部件

# 介绍

在本章中，我们将在 Python 中开发我们的第一个 GUI。我们从构建运行的 GUI 应用程序所需的最少代码开始。然后，每个示例都向 GUI 表单添加不同的小部件。

在前两个示例中，我们展示了仅包含几行代码的完整代码。在接下来的示例中，我们只展示要添加到前面示例中的代码。

在本章结束时，我们将创建一个工作的 GUI 应用程序，其中包括各种状态的标签、按钮、文本框、组合框和复选按钮，以及可以更改 GUI 背景颜色的单选按钮。

# 创建我们的第一个 Python GUI

Python 是一种非常强大的编程语言。它附带了内置的 tkinter 模块。只需几行代码（确切地说是四行），我们就可以构建我们的第一个 Python GUI。

## 准备工作

要遵循此示例，需要一个可用的 Python 开发环境。Python 附带的 IDLE GUI 足以开始。IDLE 是使用 tkinter 构建的！

### 注意

本书中的所有示例都是在 Windows 7 64 位操作系统上使用 Python 3.4 开发的。它们尚未在任何其他配置上进行测试。由于 Python 是一种跨平台语言，预计每个示例的代码都可以在任何地方运行。

如果您使用的是 Mac，它确实内置了 Python，但可能缺少一些模块，例如我们将在本书中使用的 tkinter。

我们正在使用 Python 3，Python 的创建者有意选择不与 Python 2 向后兼容。

如果您使用的是 Mac 或 Python 2，您可能需要从[www.python.org](http://www.python.org)安装 Python 3，以便成功运行本书中的示例。

## 如何做...

以下是创建结果 GUI 所需的四行 Python 代码：

```py
import tkinter as tk     # 1
win = tk.Tk()            # 2
win.title("Python GUI")  # 3
win.mainloop()           # 4
```

执行此代码并欣赏结果：

![如何做...](img/B04829_01_01.jpg)

## 工作原理...

在第 1 行中，我们导入内置的`tkinter`模块，并将其别名为`tk`以简化我们的 Python 代码。在第 2 行中，我们通过调用其构造函数（括号附加到`Tk`将类转换为实例）创建`Tk`类的实例。我们使用别名`tk`，这样我们就不必使用更长的单词`tkinter`。我们将类实例分配给名为`win`（窗口的缩写）的变量。由于 Python 是一种动态类型的语言，我们在分配给它之前不必声明此变量，并且我们不必给它指定特定的类型。*Python 从此语句的分配中推断出类型*。Python 是一种强类型的语言，因此每个变量始终都有一个类型。我们只是不必像其他语言那样事先指定其类型。这使得 Python 成为一种非常强大和高效的编程语言。

### 注意

关于类和类型的一点说明：

在 Python 中，每个变量始终都有一个类型。我们不能创建一个没有分配类型的变量。然而，在 Python 中，我们不必事先声明类型，就像在 C 编程语言中一样。

Python 足够聪明，可以推断类型。在撰写本文时，C#也具有这种能力。

使用 Python，我们可以使用`class`关键字而不是`def`关键字来创建自己的类。

为了将类分配给变量，我们首先必须创建我们类的一个实例。我们创建实例并将此实例分配给我们的变量。

```py
class AClass(object):
    print('Hello from AClass')

classInstance = AClass()
```

现在变量`classInstance`的类型是`AClass`。

如果这听起来令人困惑，不要担心。我们将在接下来的章节中介绍面向对象编程。

在第 3 行，我们使用类的实例变量(`win`)通过`title`属性给我们的窗口设置了一个标题。在第 4 行，通过在类实例`win`上调用`mainloop`方法来启动窗口的事件循环。在我们的代码中到目前为止，我们创建了一个实例并设置了一个属性*但是 GUI 直到我们启动主事件循环之前都不会显示*。

### 注意

事件循环是使我们的 GUI 工作的机制。我们可以把它看作是一个无限循环，我们的 GUI 在其中等待事件发送给它。按钮点击在我们的 GUI 中创建一个事件，或者我们的 GUI 被调整大小也会创建一个事件。

我们可以提前编写所有的 GUI 代码，直到我们调用这个无限循环(`win.mainloop()`在上面显示的代码中)用户的屏幕上什么都不会显示。

当用户点击红色的**X**按钮或者我们编程结束 GUI 的小部件时，事件循环就会结束。当事件循环结束时，我们的 GUI 也会结束。

## 还有更多...

这个示例使用了最少量的 Python 代码来创建我们的第一个 GUI 程序。然而，在本书中，我们会在合适的时候使用 OOP。

# 阻止 GUI 的大小可调整

## 准备工作

这个示例扩展了之前的示例。因此，有必要自己输入第 1 个示例的代码到你自己的项目中，或者从[`www.packtpub.com/support`](https://www.packtpub.com/support)下载代码。

## 如何做...

我们正在阻止 GUI 的大小可调整。

```py
import tkinter as tk        # 1 imports

win = tk.Tk()               # 2 Create instance
win.title("Python GUI")     # 3 Add a title       

win.resizable(0, 0)         # 4 Disable resizing the GUI

win.mainloop()              # 5 Start GUI
```

运行这段代码会创建这个 GUI：

![如何做...](img/B04829_01_02.jpg)

## 它是如何工作的...

第 4 行阻止 Python GUI 的大小可调整。

运行这段代码将会得到一个类似于我们在第 1 个示例中创建的 GUI。然而，用户不能再调整它的大小。同时，注意窗口工具栏中的最大化按钮是灰色的。

为什么这很重要？因为一旦我们向我们的表单添加小部件，调整大小可能会使我们的 GUI 看起来不如我们希望的那样好。我们将在下一个示例中向我们的 GUI 添加小部件。

`Resizable()`是`Tk()`类的一个方法，通过传入`(0, 0)`，我们阻止了 GUI 的大小可调整。如果我们传入其他值，我们就会硬编码 GUI 的 x 和 y 的启动大小，*但这不会使它不可调整大小*。

我们还在我们的代码中添加了注释，为本书中包含的示例做准备。

### 注意

在 Visual Studio .NET 等可视化编程 IDE 中，C#程序员通常不会考虑阻止用户调整他们用这种语言开发的 GUI。这会导致 GUI 质量较差。添加这一行 Python 代码可以让我们的用户欣赏我们的 GUI。

# 向 GUI 表单添加标签

## 准备工作

我们正在扩展第一个示例。我们将保持 GUI 可调整大小，所以不要使用第二个示例中的代码(或者将第 4 行的`win.resizable`注释掉)。

## 如何做...

为了向我们的 GUI 添加一个`Label`小部件，我们从`tkinter`中导入了`ttk`模块。请注意这两个导入语句。

```py
# imports                  # 1
import tkinter as tk       # 2
from tkinter import ttk    # 3
```

在示例 1 和 2 底部的`win.mainloop()`上面添加以下代码。

```py
# Adding a Label           # 4
ttk.Label(win, text="A Label").grid(column=0, row=0) # 5
```

运行这段代码会向我们的 GUI 添加一个标签：

![如何做...](img/B04829_01_03.jpg)

## 它是如何工作的...

在上面的代码的第 3 行，我们从`tkinter`中导入了一个单独的模块。`ttk`模块有一些高级的小部件，可以让我们的 GUI 看起来很棒。在某种意义上，`ttk`是`tkinter`中的一个扩展。

我们仍然需要导入`tkinter`本身，但是我们必须指定我们现在也想要从`tkinter`中使用`ttk`。

### 注意

`ttk`代表"themed tk"。它改善了我们的 GUI 外观和感觉。

上面的第 5 行在调用`mainloop`之前向 GUI 添加了标签(这里没有显示以保持空间。请参见示例 1 或 2)。

我们将我们的窗口实例传递给`ttk.Label`构造函数，并设置文本属性。这将成为我们的`Label`将显示的文本。

我们还使用了*网格布局管理器*，我们将在第二章中更深入地探讨*布局管理*。

请注意我们的 GUI 突然变得比以前的食谱小得多。

它变得如此之小的原因是我们在表单中添加了一个小部件。没有小部件，`tkinter`使用默认大小。添加小部件会导致优化，通常意味着尽可能少地使用空间来显示小部件。

如果我们使标签的文本更长，GUI 将自动扩展。我们将在第二章中的后续食谱中介绍这种自动表单大小调整，*布局管理*。

## 还有更多...

尝试调整和最大化带有标签的 GUI，看看会发生什么。

# 创建按钮并更改它们的文本属性

## 准备就绪

这个食谱扩展了上一个食谱。您可以从 Packt Publishing 网站下载整个代码。

## 如何做...

我们正在添加一个按钮，当点击时执行一个动作。在这个食谱中，我们将更新上一个食谱中添加的标签，以及更新按钮的文本属性。

```py
# Modify adding a Label                                      # 1
aLabel = ttk.Label(win, text="A Label")                      # 2
aLabel.grid(column=0, row=0)                                 # 3

# Button Click Event Callback Function                       # 4
def clickMe():                                               # 5
    action.configure(text="** I have been Clicked! **")
    aLabel.configure(foreground='red')

# Adding a Button                                            # 6
action = ttk.Button(win, text="Click Me!", command=clickMe)  # 7
action.grid(column=1, row=0)                                 # 8
```

点击按钮之前：

![如何做...](img/B04829_01_04.jpg)

点击按钮后，标签的颜色已经改变，按钮的文本也改变了。动作！

![如何做...](img/B04829_01_05.jpg)

## 它是如何工作的...

在第 2 行，我们现在将标签分配给一个变量，在第 3 行，我们使用这个变量来定位表单中的标签。我们将需要这个变量来在`clickMe()`函数中更改它的属性。默认情况下，这是一个模块级变量，因此只要我们在调用它的函数上方声明变量，我们就可以在函数内部访问它。

第 5 行是一旦按钮被点击就被调用的事件处理程序。

在第 7 行，我们创建按钮并将命令绑定到`clickMe()`函数。

### 注意

GUI 是事件驱动的。点击按钮会创建一个事件。我们使用`ttk.Button`小部件的命令属性绑定事件发生时回调函数中的操作。请注意我们没有使用括号；只有名称`clickMe`。

我们还将标签的文本更改为包含`red`，就像印刷版中一样，否则可能不太明显。当您运行代码时，您会看到颜色确实改变了。

第 3 行和第 8 行都使用了网格布局管理器，这将在下一章中讨论。这样可以对齐标签和按钮。

## 还有更多...

我们将继续向我们的 GUI 中添加更多的小部件，并在本书的其他章节中利用许多内置属性。

# 文本框小部件

在`tkinter`中，典型的文本框小部件称为`Entry`。在这个食谱中，我们将向我们的 GUI 添加这样一个`Entry`。我们将通过描述`Entry`为用户做了什么来使我们的标签更有用。

## 准备就绪

这个食谱是基于*创建按钮并更改它们的文本属性*食谱的。

## 如何做...

```py
# Modified Button Click Function   # 1
def clickMe():                     # 2
    action.configure(text='Hello ' + name.get())

# Position Button in second row, second column (zero-based)
action.grid(column=1, row=1)

# Changing our Label               # 3
ttk.Label(win, text="Enter a name:").grid(column=0, row=0) # 4

# Adding a Textbox Entry widget    # 5
name = tk.StringVar()              # 6
nameEntered = ttk.Entry(win, width=12, textvariable=name) # 7
nameEntered.grid(column=0, row=1)  # 8
```

现在我们的 GUI 看起来是这样的：

![如何做...](img/B04829_01_06.jpg)

输入一些文本并点击按钮后，GUI 发生了以下变化：

![如何做...](img/B04829_01_07.jpg)

## 它是如何工作的...

在第 2 行，我们获取`Entry`小部件的值。我们还没有使用面向对象编程，那么我们怎么能访问甚至还没有声明的变量的值呢？

在 Python 过程式编码中，如果不使用面向对象编程类，我们必须在尝试使用该名称的语句上方物理放置一个名称。那么为什么这样会起作用呢（它确实起作用）？

答案是按钮单击事件是一个回调函数，当用户单击按钮时，此函数中引用的变量是已知且存在的。

生活很美好。

第 4 行给我们的标签一个更有意义的名称，因为现在它描述了它下面的文本框。我们将按钮移动到标签旁边，以视觉上将两者关联起来。我们仍然使用网格布局管理器，将在第二章中详细解释，*布局管理*。

第 6 行创建了一个变量`name`。这个变量绑定到`Entry`，在我们的“clickMe（）”函数中，我们可以通过在这个变量上调用“get（）”来检索`Entry`框的值。这非常有效。

现在我们看到，虽然按钮显示了我们输入的整个文本（以及更多），但文本框`Entry`小部件没有扩展。原因是我们在第 7 行中将其硬编码为宽度为 12。

### 注意

Python 是一种动态类型的语言，并且从赋值中推断类型。这意味着如果我们将一个字符串赋给变量“name”，那么该变量将是字符串类型，如果我们将一个整数赋给“name”，那么该变量的类型将是整数。

使用 tkinter，我们必须将变量`name`声明为类型“tk.StringVar（）”才能成功使用它。原因是 Tkinter 不是 Python。我们可以从 Python 中使用它，但它不是相同的语言。

# 将焦点设置为小部件并禁用小部件

尽管我们的图形用户界面正在不断改进，但在 GUI 出现时让光标立即出现在`Entry`小部件中会更方便和有用。在这里，我们学习如何做到这一点。

## 准备工作

这个示例扩展了以前的示例。

## 如何做...

Python 真的很棒。当 GUI 出现时，我们只需调用先前创建的`tkinter`小部件实例上的“focus（）”方法，就可以将焦点设置为特定控件。在我们当前的 GUI 示例中，我们将`ttk.Entry`类实例分配给了一个名为`nameEntered`的变量。现在我们可以给它焦点。

将以下代码放在启动主窗口事件循环的模块底部之上，就像以前的示例一样。如果出现错误，请确保将变量调用放在声明它们的代码下面。我们目前还没有使用面向对象编程，所以这仍然是必要的。以后，将不再需要这样做。

```py
nameEntered.focus()            # Place cursor into name Entry
```

在 Mac 上，您可能必须先将焦点设置为 GUI 窗口，然后才能将焦点设置为该窗口中的`Entry`小部件。

添加这一行 Python 代码将光标放入我们的文本`Entry`框中，使文本`Entry`框获得焦点。一旦 GUI 出现，我们就可以在不必先单击它的情况下在这个文本框中输入。

![如何做...](img/B04829_01_08.jpg)

### 注意

请注意，光标现在默认驻留在文本`Entry`框内。

我们也可以禁用小部件。为此，我们在小部件上设置一个属性。通过添加这一行 Python 代码，我们可以使按钮变为禁用状态：

```py
action.configure(state='disabled')    # Disable the Button Widget
```

添加上述一行 Python 代码后，单击按钮不再产生任何动作！

![如何做...](img/B04829_01_09.jpg)

## 它是如何工作的...

这段代码是不言自明的。我们将焦点设置为一个控件并禁用另一个小部件。在编程语言中良好的命名有助于消除冗长的解释。在本书的后面，将有一些关于如何在工作中编程或在家练习编程技能时进行高级提示。

## 还有更多...

是的。这只是第一章。还有更多内容。

# 组合框小部件

在这个示例中，我们将通过添加下拉组合框来改进我们的 GUI，这些下拉组合框可以具有初始默认值。虽然我们可以限制用户只能选择某些选项，但与此同时，我们也可以允许用户输入他们希望的任何内容。

## 准备工作

这个示例扩展了以前的示例。

## 如何做...

我们正在使用网格布局管理器在`Entry`小部件和`Button`之间插入另一列。以下是 Python 代码。

```py
ttk.Label(win, text="Choose a number:").grid(column=1, row=0)  # 1
number = tk.StringVar()                         # 2
numberChosen = ttk.Combobox(win, width=12, textvariable=number) #3
numberChosen['values'] = (1, 2, 4, 42, 100)     # 4
numberChosen.grid(column=1, row=1)              # 5
numberChosen.current(0)                         # 6
```

将此代码添加到以前的示例中后，将创建以下 GUI。请注意，在前面的代码的第 4 行中，我们将默认值的元组分配给组合框。然后这些值出现在下拉框中。如果需要，我们也可以在应用程序运行时更改它们（通过输入不同的值）。

![如何做...](img/B04829_01_10.jpg)

## 它是如何工作的...

第 1 行添加了第二个标签以匹配新创建的组合框（在第 3 行创建）。第 2 行将框的值分配给特殊`tkinter`类型的变量（`StringVar`），就像我们在之前的示例中所做的那样。

第 5 行将两个新控件（标签和组合框）与我们之前的 GUI 布局对齐，第 6 行在 GUI 首次可见时分配要显示的默认值。这是`numberChosen['values']`元组的第一个值，字符串`"1"`。我们在第 4 行没有在整数元组周围放置引号，但它们被转换为字符串，因为在第 2 行，我们声明值为`tk.StringVar`类型。

屏幕截图显示用户所做的选择（**42**）。这个值被分配给`number`变量。

## 还有更多...

如果我们希望限制用户只能选择我们编程到`Combobox`中的值，我们可以通过将*state 属性*传递给构造函数来实现。修改前面代码中的第 3 行：

```py
numberChosen = ttk.Combobox(win, width=12, textvariable=number, state='readonly')
```

现在用户不能再在`Combobox`中输入值。我们可以通过在我们的按钮单击事件回调函数中添加以下代码行来显示用户选择的值：

```py
# Modified Button Click Callback Function
def clickMe():
    action.configure(text='Hello ' + name.get()+ ' ' + numberChosen.get())
```

选择一个数字，输入一个名称，然后单击按钮，我们得到以下 GUI 结果，现在还显示了所选的数字：

![还有更多...](img/B04829_01_11.jpg)

# 创建具有不同初始状态的复选按钮

在这个示例中，我们将添加三个`Checkbutton`小部件，每个小部件都有不同的初始状态。

## 准备就绪

这个示例扩展了之前的示例。

## 如何做...

我们创建了三个`Checkbutton`小部件，它们的状态不同。第一个是禁用的，并且其中有一个复选标记。用户无法移除此复选标记，因为小部件被禁用。

第二个`Checkbutton`是启用的，并且默认情况下没有复选标记，但用户可以单击它以添加复选标记。

第三个`Checkbutton`既启用又默认选中。用户可以随意取消选中和重新选中小部件。

```py
# Creating three checkbuttons    # 1
chVarDis = tk.IntVar()           # 2
check1 = tk.Checkbutton(win, text="Disabled", variable=chVarDis, state='disabled')                     # 3
check1.select()                  # 4
check1.grid(column=0, row=4, sticky=tk.W) # 5

chVarUn = tk.IntVar()            # 6
check2 = tk.Checkbutton(win, text="UnChecked", variable=chVarUn)
check2.deselect()                # 8
check2.grid(column=1, row=4, sticky=tk.W) # 9                  

chVarEn = tk.IntVar()            # 10
check3 = tk.Checkbutton(win, text="Enabled", variable=chVarEn)
check3.select()                  # 12
check3.grid(column=2, row=4, sticky=tk.W) # 13
```

运行新代码将得到以下 GUI：

![如何做...](img/B04829_01_12.jpg)

## 它是如何工作的...

在第 2、6 和 10 行，我们创建了三个`IntVar`类型的变量。在接下来的一行中，对于这些变量中的每一个，我们创建一个`Checkbutton`，传入这些变量。它们将保存`Checkbutton`的状态（未选中或选中）。默认情况下，它们是 0（未选中）或 1（选中），因此变量的类型是`tkinter`整数。

我们将这些`Checkbutton`小部件放在我们的主窗口中，因此传递给构造函数的第一个参数是小部件的父级；在我们的情况下是`win`。我们通过其`text`属性为每个`Checkbutton`提供不同的标签。

将网格的 sticky 属性设置为`tk.W`意味着小部件将对齐到网格的西侧。这与 Java 语法非常相似，意味着它将对齐到左侧。当我们调整 GUI 的大小时，小部件将保持在左侧，并不会向 GUI 的中心移动。

第 4 和 12 行通过调用这两个`Checkbutton`类实例的`select()`方法向`Checkbutton`小部件中放入复选标记。

我们继续使用网格布局管理器来排列我们的小部件，这将在第二章*布局管理*中详细解释。

# 使用单选按钮小部件

在这个示例中，我们将创建三个`tkinter Radiobutton`小部件。我们还将添加一些代码，根据选择的`Radiobutton`来更改主窗体的颜色。

## 准备就绪

这个示例扩展了之前的示例。

## 如何做...

我们将以下代码添加到之前的示例中：

```py
# Radiobutton Globals   # 1
COLOR1 = "Blue"         # 2
COLOR2 = "Gold"         # 3
COLOR3 = "Red"          # 4

# Radiobutton Callback  # 5
def radCall():          # 6
   radSel=radVar.get()
   if   radSel == 1: win.configure(background=COLOR1)
   elif radSel == 2: win.configure(background=COLOR2)
   elif radSel == 3: win.configure(background=COLOR3)

# create three Radiobuttons   # 7
radVar = tk.IntVar()          # 8
rad1 = tk.Radiobutton(win, text=COLOR1, variable=radVar, value=1,               command=radCall)              # 9
rad1.grid(column=0, row=5, sticky=tk.W)  # 10

rad2 = tk.Radiobutton(win, text=COLOR2, variable=radVar, value=2, command=radCall)                             # 11
rad2.grid(column=1, row=5, sticky=tk.W)  # 12

rad3 = tk.Radiobutton(win, text=COLOR3, variable=radVar, value=3, command=radCall)                             # 13
rad3.grid(column=2, row=5, sticky=tk.W)  # 14
```

运行此代码并选择名为**Gold**的`Radiobutton`将创建以下窗口：

![如何做...](img/B04829_01_13.jpg)

## 它是如何工作的...

在 2-4 行中，我们创建了一些模块级全局变量，我们将在每个单选按钮的创建以及在创建改变主窗体背景颜色的回调函数（使用实例变量`win`）中使用这些变量。

我们使用全局变量使代码更容易更改。通过将颜色的名称分配给一个变量，并在多个地方使用这个变量，我们可以轻松地尝试不同的颜色。我们只需要更改一行代码，而不是全局搜索和替换硬编码的字符串（容易出错），其他所有东西都会工作。这被称为**DRY 原则**，代表**不要重复自己**。这是我们将在本书的后续食谱中使用的面向对象编程概念。

### 注意

我们分配给变量（`COLOR1`，`COLOR2...`）的颜色名称是`tkinter`关键字（从技术上讲，它们是*符号名称*）。如果我们使用不是`tkinter`颜色关键字的名称，那么代码将无法工作。

第 6 行是*回调函数*，根据用户的选择改变我们主窗体（`win`）的背景。

在第 8 行，我们创建了一个`tk.IntVar`变量。重要的是，我们只创建了一个变量供所有三个单选按钮使用。从上面的截图中可以看出，无论我们选择哪个`Radiobutton`，所有其他的都会自动为我们取消选择。

第 9 到 14 行创建了三个单选按钮，将它们分配给主窗体，并传入要在回调函数中使用的变量，以创建改变主窗口背景的操作。

### 注意

虽然这是第一个改变小部件颜色的食谱，但老实说，它看起来有点丑。本书中的大部分后续食谱都会解释如何使我们的 GUI 看起来真正令人惊叹。

## 还有更多...

这里是一小部分可用的符号颜色名称，您可以在官方 tcl 手册页面上查找：

[`www.tcl.tk/man/tcl8.5/TkCmd/colors.htm`](http://www.tcl.tk/man/tcl8.5/TkCmd/colors.htm)

| 名称 | 红 | 绿 | 蓝 |
| --- | --- | --- | --- |
| alice blue | 240 | 248 | 255 |
| AliceBlue | 240 | 248 | 255 |
| Blue | 0 | 0 | 255 |
| 金色 | 255 | 215 | 0 |
| 红色 | 255 | 0 | 0 |

一些名称创建相同的颜色，因此`alice blue`创建的颜色与`AliceBlue`相同。在这个食谱中，我们使用了符号名称`Blue`，`Gold`和`Red`。

# 使用滚动文本小部件

`ScrolledText`小部件比简单的`Entry`小部件大得多，跨越多行。它们就像记事本一样的小部件，自动换行，并在文本大于`ScrolledText`小部件的高度时自动启用垂直滚动条。

## 准备工作

这个食谱扩展了之前的食谱。您可以从 Packt Publishing 网站下载本书每一章的代码。

## 如何做...

通过添加以下代码行，我们创建了一个`ScrolledText`小部件：

```py
# Add this import to the top of the Python Module    # 1
from tkinter import scrolledtext      # 2

# Using a scrolled Text control       # 3
scrolW  = 30                          # 4
scrolH  =  3                          # 5
scr = scrolledtext.ScrolledText(win, width=scrolW, height=scrolH, wrap=tk.WORD)                         # 6
scr.grid(column=0, columnspan=3)      # 7
```

我们实际上可以在我们的小部件中输入文字，如果我们输入足够多的单词，行将自动换行！

![操作步骤...](img/B04829_01_14.jpg)

一旦我们输入的单词超过了小部件可以显示的高度，垂直滚动条就会启用。所有这些都是开箱即用的，我们不需要编写任何额外的代码来实现这一点。

![操作步骤...](img/B04829_01_15.jpg)

## 它是如何工作的...

在第 2 行，我们导入包含`ScrolledText`小部件类的模块。将其添加到模块顶部，就在其他两个`import`语句的下面。

第 4 和 5 行定义了我们即将创建的`ScrolledText`小部件的宽度和高度。这些是硬编码的值，我们将它们传递给第 6 行中`ScrolledText`小部件的构造函数。

这些值是通过实验找到的*魔术数字*，可以很好地工作。您可以尝试将`srcolW`从 30 更改为 50，并观察效果！

在第 6 行，我们通过传入`wrap=tk.WORD`在小部件上设置了一个属性。

通过将`wrap`属性设置为`tk.WORD`，我们告诉`ScrolledText`小部件按单词换行，这样我们就不会在单词中间换行。默认选项是`tk.CHAR`，它会在单词中间换行。

第二个屏幕截图显示，垂直滚动条向下移动，因为我们正在阅读一个较长的文本，它不能完全适应我们创建的`SrolledText`控件的 x，y 维度。

将网格小部件的`columnspan`属性设置为`3`，使`SrolledText`小部件跨越所有三列。如果我们不设置这个属性，我们的`SrolledText`小部件将只驻留在第一列，这不是我们想要的。

# 在循环中添加多个小部件

到目前为止，我们已经通过基本上复制和粘贴相同的代码，然后修改变化（例如，列号）来创建了几个相同类型的小部件（例如`Radiobutton`）。在这个示例中，我们开始重构我们的代码，使其不那么冗余。

## 准备工作

我们正在重构上一个示例代码的一些部分，所以你需要将那个代码应用到这个示例中。

## 如何做到...

```py
# First, we change our Radiobutton global variables into a list.
colors = ["Blue", "Gold", "Red"]              # 1

# create three Radiobuttons using one variable
radVar = tk.IntVar()

Next we are selecting a non-existing index value for radVar.
radVar.set(99)                                # 2

Now we are creating all three Radiobutton widgets within one loop.

for col in range(3):                          # 3
    curRad = 'rad' + str(col)  
    curRad = tk.Radiobutton(win, text=colors[col], variable=radVar,     value=col, command=radCall)
    curRad.grid(column=col, row=5, sticky=tk.W)

We have also changed the callback function to be zero-based, using the list instead of module-level global variables. 

# Radiobutton callback function                # 4
def radCall():
   radSel=radVar.get()
   if   radSel == 0: win.configure(background=colors[0])
   elif radSel == 1: win.configure(background=colors[1])
   elif radSel == 2: win.configure(background=colors[2])
```

运行此代码将创建与以前相同的窗口，但我们的代码更清晰，更易于维护。这将有助于我们在下一个示例中扩展我们的 GUI。

## 它是如何工作的...

在第 1 行，我们将全局变量转换为列表。

在第 2 行，我们为名为`radVar`的`tk.IntVar`变量设置了默认值。这很重要，因为在上一个示例中，我们将`Radiobutton`小部件的值设置为 1，但在我们的新循环中，使用 Python 的基于零的索引更方便。如果我们没有将默认值设置为超出`Radiobutton`小部件范围的值，当 GUI 出现时，将选择一个单选按钮。虽然这本身可能并不那么糟糕，*它不会触发回调*，我们最终会选择一个不起作用的单选按钮（即更改主窗体的颜色）。

在第 3 行，我们用循环替换了之前硬编码创建`Radiobutton`小部件的三个部分，这样做是一样的。它只是更简洁（代码行数更少）和更易于维护。例如，如果我们想创建 100 个而不仅仅是 3 个`Radiobutton`小部件，我们只需要改变 Python 的 range 运算符中的数字。我们不必输入或复制粘贴 97 个重复代码段，只需一个数字。

第 4 行显示了修改后的回调，实际上它位于前面的行之上。我们将其放在下面是为了强调这个示例的更重要的部分。

## 还有更多...

这个示例结束了本书的第一章。接下来章节中的所有示例都将在我们迄今为止构建的 GUI 基础上进行扩展，大大增强它。
