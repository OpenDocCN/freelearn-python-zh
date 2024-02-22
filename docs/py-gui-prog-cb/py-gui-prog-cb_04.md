# 第四章。数据和类

在本章中，我们将使用 Python 3 来使用数据和 OOP 类：

+   如何使用 StringVar()

+   如何从小部件获取数据

+   使用模块级全局变量

+   如何在类中编码可以改进 GUI

+   编写回调函数

+   创建可重用的 GUI 组件

# 介绍

在本章中，我们将把 GUI 数据保存到 tkinter 变量中。

我们还将开始使用**面向对象编程**（**OOP**）来扩展现有的 tkinter 类，以扩展 tkinter 的内置功能。这将使我们创建可重用的 OOP 组件。

# 如何使用 StringVar()

在 tkinter 中有一些内置的编程类型，它们与我们习惯用 Python 编程的类型略有不同。StringVar()就是这些 tkinter 类型之一。

这个示例将向您展示如何使用 StringVar()类型。

## 准备工作

我们正在学习如何将 tkinter GUI 中的数据保存到变量中，以便我们可以使用这些数据。我们可以设置和获取它们的值，与 Java 的 getter/setter 方法非常相似。

这里是 tkinter 中可用的一些编码类型：

| `strVar = StringVar()` | # 保存一个字符串；默认值是一个空字符串"" |
| --- | --- |
| `intVar = IntVar()` | # 保存一个整数；默认值是 0 |
| `dbVar = DoubleVar()` | # 保存一个浮点数；默认值是 0.0 |
| `blVar = BooleanVar()` | # 保存一个布尔值，对于 false 返回 0，对于 true 返回 1 |

### 注意

不同的语言称带有小数点的数字为浮点数或双精度数。Tkinter 将 Python 中称为浮点数据类型的内容称为 DoubleVar。根据精度级别，浮点数和双精度数据可能不同。在这里，我们将 tkinter 的 DoubleVar 翻译成 Python 中的 Python 浮点类型。

## 如何做...

我们正在创建一个新的 Python 模块，下面的截图显示了代码和生成的输出：

![如何做...](img/B04829_04_01.jpg)

首先，我们导入 tkinter 模块并将其别名为`tk`。

接下来，我们使用这个别名通过在`Tk`后面加括号来创建`Tk`类的一个实例，这样就调用了类的构造函数。这与调用函数的机制相同，只是这里我们创建了一个类的实例。

通常我们使用分配给变量`win`的实例来在代码中稍后启动主事件循环。但是在这里，我们不显示 GUI，而是演示如何使用 tkinter 的 StringVar 类型。

### 注意

我们仍然必须创建`Tk()`的一个实例。如果我们注释掉这一行，我们将从 tkinter 得到一个错误，因此这个调用是必要的。

然后我们创建一个 tkinter StringVar 类型的实例，并将其分配给我们的 Python`strData`变量。

之后，我们使用我们的变量调用 StringVar 的`set()`方法，并在设置为一个值后，然后获取该值并将其保存在一个名为`varData`的新变量中，然后打印出它的值。

在 Eclipse PyDev 控制台中，可以看到输出打印到控制台的底部，这是**Hello StringVar**。

接下来，我们将打印 tkinter 的 IntVar、DoubleVar 和 BooleanVar 类型的默认值。

![如何做...](img/B04829_04_02.jpg)

## 它是如何工作的...

如前面的截图所示，默认值并没有像我们预期的那样被打印出来。

在线文献提到了默认值，但在调用它们的`get`方法之前，我们不会看到这些值。否则，我们只会得到一个自动递增的变量名（例如在前面的截图中可以看到的 PY_VAR3）。

将 tkinter 类型分配给 Python 变量并不会改变结果。我们仍然没有得到默认值。

在这里，我们专注于最简单的代码（创建 PY_VAR0）：

![它是如何工作的...](img/B04829_04_03.jpg)

该值是 PY_VAR0，而不是预期的 0，直到我们调用`get`方法。现在我们可以看到默认值。我们没有调用`set`，所以一旦我们在每种类型上调用`get`方法，就会看到自动分配给每种 tkinter 类型的默认值。

![它是如何工作的...](img/B04829_04_04.jpg)

注意`IntVar`实例的默认值为 0 被打印到控制台，我们将其保存在`intData`变量中。我们还可以在屏幕截图的顶部看到 Eclipse PyDev 调试器窗口中的值。

# 如何从小部件中获取数据

当用户输入数据时，我们希望在我们的代码中对其进行处理。这个配方展示了如何在变量中捕获数据。在上一个配方中，我们创建了几个 tkinter 类变量。它们是独立的。现在我们正在将它们连接到我们的 GUI，使用我们从 GUI 中获取的数据并将其存储在 Python 变量中。

## 准备工作

我们将继续使用我们在上一章中构建的 Python GUI。

## 如何做...

我们正在将来自我们的 GUI 的值分配给一个 Python 变量。

在我们的模块底部，就在主事件循环之上，添加以下代码：

```py
strData = spin.get()
print("Spinbox value: " + strData)

# Place cursor into name Entry
nameEntered.focus()      
#======================
# Start GUI
#======================
win.mainloop()
```

运行代码会给我们以下结果：

![如何做...](img/B04829_04_05.jpg)

我们正在检索`Spinbox`控件的当前值。

### 注意

我们将我们的代码放在 GUI 主事件循环之上，因此打印发生在 GUI 变得可见之前。如果我们想要在显示 GUI 并改变`Spinbox`控件的值之后打印出当前值，我们将不得不将代码放在回调函数中。

我们使用以下代码创建了我们的 Spinbox 小部件，将可用值硬编码到其中：

```py
# Adding a Spinbox widget using a set of values
spin = Spinbox(monty, values=(1, 2, 4, 42, 100), width=5, bd=8, command=_spin) 
spin.grid(column=0, row=2)
```

我们还可以将数据的硬编码从`Spinbox`类实例的创建中移出，并稍后设置它。

```py
# Adding a Spinbox widget assigning values after creation
spin = Spinbox(monty, width=5, bd=8, command=_spin) 
spin['values'] = (1, 2, 4, 42, 100)
spin.grid(column=0, row=2)
```

无论我们如何创建小部件并将数据插入其中，因为我们可以通过在小部件实例上使用`get()`方法来访问这些数据，所以我们可以访问这些数据。

## 它是如何工作的...

为了从使用 tkinter 编写的 GUI 中获取值，我们使用 tkinter 的`get()`方法来获取我们希望获取值的小部件的实例。

在上面的例子中，我们使用了 Spinbox 控件，但对于所有具有`get()`方法的小部件，原理是相同的。

一旦我们获得了数据，我们就处于一个纯粹的 Python 世界，而 tkinter 确实帮助我们构建了我们的 GUI。现在我们知道如何从我们的 GUI 中获取数据，我们可以使用这些数据。

# 使用模块级全局变量

封装是任何编程语言中的一个主要优势，它使我们能够使用 OOP 进行编程。Python 既是 OOP 又是过程化的。我们可以创建局部化到它们所在模块的全局变量。它们只对这个模块全局，这是一种封装的形式。为什么我们想要这样做？因为随着我们向我们的 GUI 添加越来越多的功能，我们希望避免命名冲突，这可能导致我们代码中的错误。

### 注意

我们不希望命名冲突在我们的代码中创建错误！命名空间是避免这些错误的一种方法，在 Python 中，我们可以通过使用 Python 模块（这些是非官方的命名空间）来实现这一点。

## 准备工作

我们可以在任何模块的顶部和函数之外声明模块级全局变量。

然后我们必须使用`global` Python 关键字来引用它们。如果我们在函数中忘记使用`global`，我们将意外创建新的局部变量。这将是一个错误，而且是我们真的不想做的事情。

### 注意

Python 是一种动态的、强类型的语言。我们只会在运行时注意到这样的错误（忘记使用全局关键字来限定变量的范围）。

## 如何做...

将第 15 行中显示的代码添加到我们在上一章和上一章中使用的 GUI 中，这将创建一个模块级的全局变量。我们使用了 C 风格的全大写约定，这并不真正“Pythonic”，但我认为这确实强调了我们在这个配方中要解决的原则。

![如何做...](img/B04829_04_06.jpg)

运行代码会导致全局变量的打印。注意**42**被打印到 Eclipse 控制台。

![如何做...](img/B04829_04_07.jpg)

## 它是如何工作的...

我们在我们的模块顶部定义一个全局变量，稍后，在我们的模块底部，我们打印出它的值。

那起作用。

在我们的模块底部添加这个函数：

![它是如何工作的...](img/B04829_04_08.jpg)

在上面，我们正在使用模块级全局变量。很容易出错，因为`global`被遮蔽，如下面的屏幕截图所示：

![它是如何工作的...](img/B04829_04_09.jpg)

请注意，即使我们使用相同的变量名，`42`也变成了`777`。

### 注意

Python 中没有编译器警告我们在本地函数中覆盖全局变量。这可能导致在运行时调试时出现困难。

使用全局限定符（第 234 行）打印出我们最初分配的值（42），如下面的屏幕截图所示：

![它是如何工作的...](img/B04829_04_10.jpg)

但是，要小心。当我们取消本地全局时，我们打印出本地的值，而不是全局的值：

![它是如何工作的...](img/B04829_04_11.jpg)

尽管我们使用了`global`限定符，但本地变量似乎会覆盖它。我们从 Eclipse PyDev 插件中得到了一个警告，即我们的`GLOBAL_CONST = 777`没有被使用，但运行代码仍然打印出 777，而不是预期的 42。

这可能不是我们期望的行为。使用`global`限定符，我们可能期望指向先前创建的全局变量。

相反，似乎 Python 在本地函数中创建了一个新的全局变量，并覆盖了我们之前创建的全局变量。

全局变量在编写小型应用程序时非常有用。它们可以帮助在同一 Python 模块中的方法和函数之间共享数据，并且有时 OOP 的开销是不合理的。

随着我们的程序变得越来越复杂，使用全局变量所获得的好处很快就会减少。

### 注意

最好避免使用全局变量，并通过在不同范围中使用相同的名称而意外地遮蔽变量。我们可以使用面向对象编程来代替使用全局变量。

我们在过程化代码中玩了全局变量，并学会了如何导致难以调试的错误。在下一章中，我们将转向面向对象编程，这可以消除这些类型的错误。

# 如何在类中编码可以改进 GUI

到目前为止，我们一直在以过程化的方式编码。这是来自 Python 的一种快速脚本化方法。一旦我们的代码变得越来越大，我们就需要进步到面向对象编程。

为什么？

因为，除了许多其他好处之外，面向对象编程允许我们通过使用方法来移动代码。一旦我们使用类，我们就不再需要在调用代码的代码上方物理放置代码。这使我们在组织代码方面具有很大的灵活性。

我们可以将相关代码写在其他代码旁边，不再担心代码不会运行，因为代码不在调用它的代码上方。

我们可以通过编写引用未在该模块中创建的方法的模块来将其推向一些相当花哨的极端。它们依赖于运行时状态在代码运行时创建了这些方法。

### 注意

如果我们调用的方法在那时还没有被创建，我们会得到一个运行时错误。

## 准备就绪

我们只需将整个过程化代码简单地转换为面向对象编程。我们只需将其转换为一个类，缩进所有现有代码，并在所有变量之前添加`self`。

这非常容易。

虽然起初可能感觉有点烦人，必须在所有东西之前加上`self`关键字，使我们的代码更冗长（嘿，我们浪费了这么多纸...）；但最终，这将是值得的。

## 如何做...

一开始，一切都乱了，但我们很快就会解决这个明显的混乱。

请注意，在 Eclipse 中，PyDev 编辑器通过在代码编辑器的右侧部分将其标记为红色来提示编码问题。

也许我们毕竟不应该使用面向对象编程，但这就是我们所做的，而且理由非常充分。

![如何做...](img/B04829_04_12.jpg)

我们只需使用`self`关键字在所有变量之前添加，并通过使用`self`将函数绑定到类中，这样官方和技术上将函数转换为方法。

### 注意

函数和方法之间有区别。Python 非常清楚地表明了这一点。方法绑定到一个类，而函数则没有。我们甚至可以在同一个 Python 模块中混合使用这两种方法。

让我们用`self`作为前缀来消除红色，这样我们就可以再次运行我们的代码。

![如何做...](img/B04829_04_13.jpg)

一旦我们对所有在红色中突出显示的错误做了这些，我们就可以再次运行我们的 Python 代码。

`clickMe`函数现在绑定到类上，正式成为一个方法。

不幸的是，以过程式方式开始，然后将其转换为面向对象的方式并不像我上面说的那么简单。代码变得一团糟。这是以面向对象的方式开始编程的一个很好的理由。

### 注意

Python 擅长以简单的方式做事。简单的代码通常变得更加复杂（因为一开始很容易）。一旦变得过于复杂，将我们的过程式代码重构为真正的面向对象的代码变得越来越困难。

我们正在将我们的过程式代码转换为面向对象的代码。看看我们陷入的所有麻烦，仅仅将 200 多行的 Python 代码转换为面向对象的代码可能表明，我们可能最好从一开始就开始使用面向对象的方式编码。

实际上，我们确实破坏了一些之前工作正常的功能。现在无法使用 Tab 2 和点击单选按钮了。我们必须进行更多的重构。

过程式代码之所以容易，是因为它只是从上到下的编码。现在我们把我们的代码放入一个类中，我们必须把所有的回调函数移到方法中。这样做是可以的，但确实需要一些工作来转换我们的原始代码。

我们的过程式代码看起来像这样：

```py
# Button Click Function
def clickMe():
    action.configure(text='Hello ' + name.get())

# Changing our Label
ttk.Label(monty, text="Enter a name:").grid(column=0, row=0, sticky='W')

# Adding a Textbox Entry widget
name = tk.StringVar()
nameEntered = ttk.Entry(monty, width=12, textvariable=name)
nameEntered.grid(column=0, row=1, sticky='W')

# Adding a Button
action = ttk.Button(monty, text="Click Me!", command=clickMe)
action.grid(column=2, row=1)

The new OOP code looks like this:
class OOP():
    def __init__(self): 
        # Create instance
        self.win = tk.Tk()   

        # Add a title       
        self.win.title("Python GUI")      
        self.createWidgets()

    # Button callback
    def clickMe(self):
        self.action.configure(text='Hello ' + self.name.get())

    # … more callback methods 

    def createWidgets(self):    
        # Tab Control introduced here -----------------------
        tabControl = ttk.Notebook(self.win)     # Create Tab Control

        tab1 = ttk.Frame(tabControl)            # Create a tab 
        tabControl.add(tab1, text='Tab 1')      # Add the tab

        tab2 = ttk.Frame(tabControl)            # Create second tab
        tabControl.add(tab2, text='Tab 2')      # Add second tab 

        tabControl.pack(expand=1, fill="both")  # Pack make visible
#======================
# Start GUI
#======================
oop = OOP()
oop.win.mainloop()
```

我们将回调方法移到模块顶部，放在新的面向对象类内部。我们将所有的部件创建代码放入一个相当长的方法中，在类的初始化器中调用它。

从技术上讲，在低级代码的深处，Python 确实有一个构造函数，但 Python 让我们摆脱了对此的任何担忧。这已经为我们处理了。

相反，除了一个“真正的”构造函数之外，Python 还为我们提供了一个初始化器。

我们强烈建议使用这个初始化器。我们可以用它来向我们的类传递参数，初始化我们希望在类实例内部使用的变量。

### 注意

在 Python 中，同一个 Python 模块中可以存在多个类。

与 Java 不同，它有一个非常严格的命名约定（没有这个约定它就无法工作），Python 要灵活得多。

### 注意

我们可以在同一个 Python 模块中创建多个类。与 Java 不同，我们不依赖于必须与每个类名匹配的文件名。

Python 真的很棒！

一旦我们的 Python GUI 变得庞大，我们将把一些类拆分成它们自己的模块，但与 Java 不同，我们不必这样做。在这本书和项目中，我们将保持一些类在同一个模块中，同时，我们将把一些其他类拆分成它们自己的模块，将它们导入到可以被认为是一个 main()函数的地方（这不是 C，但我们可以像 C 一样思考，因为 Python 非常灵活）。

到目前为止，我们所做的是将`ToolTip`类添加到我们的 Python 模块中，并将我们的过程式 Python 代码重构为面向对象的 Python 代码。

在这里，在这个示例中，我们可以看到一个 Python 模块中可以存在多个类。

确实很酷！

![如何做...](img/B04829_04_14.jpg)

`ToolTip`类和`OOP`类都驻留在同一个 Python 模块中。

![如何做...](img/B04829_04_15.jpg)

## 它是如何工作的...

在这个示例中，我们将我们的过程式代码推进到面向对象编程（OOP）代码。

Python 使我们能够以实用的过程式风格编写代码，就像 C 编程语言一样。

与此同时，我们有选择以面向对象的方式编码，就像 Java、C#和 C++一样。

# 编写回调函数

起初，回调函数可能看起来有点令人生畏。您调用函数，传递一些参数，现在函数告诉您它真的很忙，会回电话给您！

你会想：“这个函数会*永远*回调我吗？”“我需要*等*多久？”

在 Python 中，即使回调函数也很容易，是的，它们通常会回调你。

它们只需要先完成它们分配的任务（嘿，是你编码它们的第一次...）。

让我们更多地了解一下当我们将回调编码到我们的 GUI 中时会发生什么。

我们的 GUI 是事件驱动的。在创建并显示在屏幕上之后，它通常会等待事件发生。它基本上在等待事件被发送到它。我们可以通过点击其动作按钮之一来向我们的 GUI 发送事件。

这创建了一个事件，并且在某种意义上，我们通过发送消息“调用”了我们的 GUI。

现在，我们发送消息到我们的 GUI 后应该发生什么？

点击按钮后发生的事情取决于我们是否创建了事件处理程序并将其与此按钮关联。如果我们没有创建事件处理程序，点击按钮将没有任何效果。

事件处理程序是一个回调函数（或方法，如果我们使用类）。

回调方法也是被动的，就像我们的 GUI 一样，等待被调用。

一旦我们的 GUI 被点击按钮，它将调用回调函数。

回调通常会进行一些处理，完成后将结果返回给我们的 GUI。

### 注意

在某种意义上，我们可以看到我们的回调函数在回调我们的 GUI。

## 准备就绪

Python 解释器会运行项目中的所有代码一次，找到任何语法错误并指出它们。如果语法不正确，您无法运行 Python 代码。这包括缩进（如果不导致语法错误，错误的缩进通常会导致错误）。

在下一轮解析中，解释器解释我们的代码并运行它。

在运行时，可以生成许多 GUI 事件，通常是回调函数为 GUI 小部件添加功能。

## 如何做...

这是 Spinbox 小部件的回调：

![如何做...](img/B04829_04_16.jpg)

## 它是如何工作的...

我们在`OOP`类中创建了一个回调方法，当我们从 Spinbox 小部件中选择一个值时，它会被调用，因为我们通过`command`参数（`command=self._spin`）将该方法绑定到小部件。我们使用下划线前缀来暗示这个方法应该像一个私有的 Java 方法一样受到尊重。

Python 故意避免了私有、公共、友好等语言限制。

在 Python 中，我们使用命名约定。预期用双下划线包围关键字的前后缀应该限制在 Python 语言中，我们不应该在我们自己的 Python 代码中使用它们。

但是，我们可以使用下划线前缀来提供一个提示，表明这个名称应该被视为私有助手。

与此同时，如果我们希望使用本来是 Python 内置名称的名称，我们可以在后面加上一个下划线。例如，如果我们希望缩短列表的长度，我们可以这样做：

```py
len_ = len(aList)
```

通常，下划线很难阅读，容易忽视，因此在实践中这可能不是最好的主意。

# 创建可重用的 GUI 组件

我们正在使用 Python 创建可重用的 GUI 组件。

在这个示例中，我们将简化操作，将我们的`ToolTip`类移动到其自己的模块中。接下来，我们将导入并在 GUI 的几个小部件上使用它来显示工具提示。

## 准备就绪

我们正在构建我们之前的代码。

## 如何做...

我们将首先将我们的`ToolTip`类拆分为一个单独的 Python 模块。我们将稍微增强它，以便传入控件小部件和我们希望在悬停鼠标在控件上时显示的工具提示文本。

我们创建了一个新的 Python 模块，并将`ToolTip`类代码放入其中，然后将此模块导入我们的主要模块。

然后，我们通过创建几个工具提示来重用导入的`ToolTip`类，当鼠标悬停在几个 GUI 小部件上时可以看到它们。

将我们通用的`ToolTip`类代码重构到自己的模块中有助于我们重用这些代码。我们使用 DRY 原则，将我们的通用代码放在一个地方，这样当我们修改代码时，导入它的所有模块将自动获得我们模块的最新版本，而不是复制/粘贴/修改。

### 注意

DRY 代表不要重复自己，我们将在以后的章节中再次讨论它。

我们可以通过将选项卡 3 的图像转换为可重用组件来做类似的事情。

为了保持本示例的代码简单，我们删除了选项卡 3，但您可以尝试使用上一章的代码进行实验。

![如何做...](img/B04829_04_17.jpg)

```py

# Add a Tooltip to the Spinbox
tt.createToolTip(self.spin, 'This is a Spin control.')

# Add Tooltips to more widgets
tt.createToolTip(nameEntered, 'This is an Entry control.')
tt.createToolTip(self.action, 'This is a Button control.')
tt.createToolTip(self.scr, 'This is a ScrolledText control.')
```

这也适用于第二个选项卡。

![如何做...](img/B04829_04_18.jpg)

新的代码结构现在看起来像这样：

![如何做...](img/B04829_04_19.jpg)

导入语句如下所示：

![如何做...](img/B04829_04_20.jpg)

而在单独的模块中分解（重构）的代码如下所示：

![如何做...](img/B04829_04_21.jpg)

## 工作原理...

在前面的屏幕截图中，我们可以看到显示了几条工具提示消息。主窗口的工具提示可能有点烦人，所以最好不要为主窗口显示工具提示，因为我们真的希望突出显示各个小部件的功能。主窗体有一个解释其目的的标题；不需要工具提示。