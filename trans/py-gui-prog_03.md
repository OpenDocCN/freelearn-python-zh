# 使用Tkinter和ttk小部件创建基本表单

好消息！您的设计已经得到主管的审查和批准。现在是时候开始实施了！

在本章中，您将涵盖以下主题：

+   根据设计评估您的技术选择

+   了解我们选择的Tkinter和`ttk`小部件

+   实现和测试表单和应用程序

让我们开始编码吧！

# 评估我们的技术选择

我们对设计的第一次实现将是一个非常简单的应用程序，它提供了规范的核心功能和很少的其他功能。这被称为**最小可行产品**或**MVP**。一旦我们建立了MVP，我们将更好地了解如何将其发展成最终产品。

在我们开始之前，让我们花点时间评估我们的技术选择。

# 选择技术

当然，我们将使用Python和Tkinter构建这个表单。然而，值得问一下，Tkinter是否真的是应用程序的良好技术选择。在选择用于实现此表单的GUI工具包时，我们需要考虑以下几点：

+   **您目前的专业知识和技能**：您的专业是Python，但在创建GUI方面经验不足。为了最快的交付时间，您需要一个能够很好地与Python配合使用并且不难学习的选项。您还希望选择一些已经建立并且稳定的东西，因为您没有时间跟上工具包的新发展。Tkinter在这里适用。

+   **目标平台**：您将在Windows PC上开发应用程序，但它需要在Debian Linux上运行，因此GUI的选择应该是跨平台的。它将在一台又老又慢的计算机上运行，因此您的程序需要节约资源。Tkinter在这里也适用。

+   **应用功能**：您的应用程序需要能够显示基本表单字段，验证输入的数据，并将其写入CSV。Tkinter可以处理这些前端要求，Python可以轻松处理CSV文件。

鉴于Python的可用选项，Tkinter是一个不错的选择。它学习曲线短，轻量级，在您的开发和目标平台上都很容易获得，并且包含了表单所需的功能。

Python还有其他用于GUI开发的选项，包括**PyQT**、**Kivy**和**wxPython**。与Tkinter相比，它们各自有不同的优势和劣势，但如果发现Tkinter不适合某个项目，其中一个可能是更好的选择。

# 探索Tkinter小部件

当我们设计应用程序时，我们挑选了一个小部件类，它最接近我们需要的每个字段。这些是`Entry`、`Spinbox`、`Combobox`、`Checkbutton`和`Text`小部件。我们还确定我们需要`Button`和`LabelFrame`小部件来实现应用程序布局。在我们开始编写我们的类之前，让我们来看看这些小部件。

我们的一些小部件在Tkinter中，另一些在`ttk`主题小部件集中，还有一些在两个库中都有。我们更喜欢`ttk`版本，因为它们在各个平台上看起来更好。请注意我们从哪个库导入每个小部件。

# 输入小部件

`ttk.Entry`小部件是一个基本的、单行字符输入，如下截图所示：

![](assets/93ab1723-3880-43fe-8866-779aeb33dd64.png)

您可以通过执行以下代码来创建一个输入：

```py
my_entry = ttk.Entry(parent, textvariable=my_text_var)
```

在上述代码中，`ttk.Entry`的常用参数如下：

+   `parent`：此参数为输入设置了`parent`小部件。

+   `textvariable`：这是一个Tkinter `StringVar`变量，其值将绑定到此`input`小部件。

+   `show`：此参数确定在您输入框中键入时将显示哪个字符。默认情况下，它是您键入的字符，但这可以被替换（例如，对于密码输入，您可以指定`*`或点来代替显示）。

+   `Entry`：像所有的`ttk`小部件一样，此小部件支持额外的格式和样式选项。

在所有上述参数中，使用“textvariable”参数是可选的；没有它，我们可以使用其“get()”方法提取“Entry”小部件中的值。然而，将变量绑定到我们的“input”小部件具有一些优势。首先，我们不必保留或传递对小部件本身的引用。这将使得在后面的章节中更容易将我们的软件重新组织为单独的模块。此外，对输入值的更改会自动传播到变量，反之亦然。

# Spinbox小部件

“ttk.Spinbox”小部件向常规“Entry”小部件添加了增量和减量按钮，使其适用于数字数据。

在Python 3.7之前，“Spinbox”只在Tkinter中可用，而不是在`ttk`中。如果您使用的是Python 3.6或更早版本，请改用`Tkinter.Spinbox`小部件。示例代码使用了Tkinter版本以确保兼容性。

创建“Spinbox”小部件如下：

```py
my_spinbox = tk.Spinbox(
    parent,
    from_=0.5,
    to=52.0,
    increment=.01,
    textvariable=my_double_var)
```

如前面的代码所示，“Spinbox”小部件需要一些额外的构造函数参数来控制增量和减量按钮的行为，如下所示：

+   **`from_`**：此参数确定箭头递减到的最低值。需要添加下划线，因为“from”是Python关键字；在Tcl/`Tk`中只是“from”。

+   **`to`**：此参数确定箭头递增到的最高值。

+   **`increment`**：此参数表示箭头递增或递减的数量。

+   **`values`**：此参数接受一个可以通过递增的字符串或数字值列表。

请注意，如果使用了“from_”和“to”，则两者都是必需的；也就是说，您不能只指定一个下限，这样做将导致异常或奇怪的行为。

查看以下截图中的“Spinbox”小部件：

![](assets/0af19ce1-0ba8-4436-9c18-41fe6f315936.png)

“Spinbox”小部件不仅仅是用于数字，尽管这主要是我们将要使用它的方式。它也可以接受一个字符串列表，可以使用箭头按钮进行选择。因为它可以用于字符串或数字，所以“textvariable”参数接受`StringVar`、`IntVar`或`DoubleVar`数据类型。

请注意，这些参数都不限制可以输入到“Spinbox”小部件中的内容。它只不过是一个带有按钮的“Entry”小部件，您不仅可以输入有效范围之外的值，还可以输入字母和符号。这样做可能会导致异常，如果您已将小部件绑定到非字符串变量。

# Combobox小部件

“ttk.Combobox”参数是一个“Entry”小部件，它添加了一个下拉选择菜单。要填充下拉菜单，只需传入一个带有用户可以选择的字符串列表的“values”参数。

您可以执行以下代码来创建一个“Combobox”小部件：

```py
combobox = ttk.Combobox(
    parent, textvariable=my_string_var,
    values=["Option 1", "Option 2", "Option 3"])
```

上述代码将生成以下小部件：

![](assets/2b06ad94-ddbd-45f4-8a00-fa095da6b4d9.png)如果您习惯于HTML的“<SELECT>”小部件或其他工具包中的下拉小部件，“ttk.Combobox”小部件可能对您来说有些陌生。它实际上是一个带有下拉菜单以选择一些预设字符串的“Entry”小部件。就像“Spinbox”小部件一样，它不限制可以输入的值。

# Checkbutton小部件

“ttk.Checkbutton”小部件是一个带有标签的复选框，用于输入布尔数据。与“Spinbox”和“Combobox”不同，它不是从“Entry”小部件派生的，其参数如下所示：

+   `text`：此参数设置小部件的标签。

+   `variable`：此参数是`BooleanVar`，绑定了复选框的选中状态。

+   `textvariable`：与基于`Entry`的小部件不同，此参数可用于将变量绑定到小部件的标签文本。您不会经常使用它，但您应该知道它存在，以防您错误地将变量分配给它。

您可以执行以下代码来创建一个“Checkbutton”小部件：

```py
my_checkbutton = ttk.Checkbutton(
    parent, text="Check to make this option True",
    variable=my_boolean_var)
```

“Checkbox”小部件显示为一个带有标签的可点击框，如下截图所示：

![](assets/8a97f210-253f-438e-afb5-b7f49e04fed9.png)

# 文本小部件

“Text”小部件不仅仅是一个多行“Entry”小部件。它具有强大的标记系统，允许您实现多彩的文本，超链接样式的可点击文本等。与其他小部件不同，它不能绑定到Tkinter的“StringVar”，因此设置或检索其内容需要通过其“get()”、“insert()”和“delete()”方法来完成。

在使用这些方法进行读取或修改时，您需要传入一个或两个**索引**值来选择您要操作的字符或字符范围。这些索引值是字符串，可以采用以下任何格式：

+   由点分隔的行号和字符号。行号从1开始，字符从0开始，因此第一行上的第一个字符是“1.0”，而第四行上的第十二个字符将是“4.11”。

+   “end”字符串或Tkinter常量“END”，表示字段的结束。

+   一个数字索引加上单词“linestart”、“lineend”、“wordstart”和“wordend”中的一个，表示相对于数字索引的行或单词的开始或结束。例如，“6.2 wordstart”将是包含第六行第三个字符的单词的开始；“2.0 lineend”将是第二行的结束。

+   前述任何一个，加上加号或减号运算符，以及一定数量的字符或行。例如，“2.5 wordend - 1 chars”将是第二行第六个字符所在的单词结束前的字符。

以下示例显示了使用“Text”小部件的基础知识：

```py
# create the widget.  Make sure to save a reference.
mytext = tk.Text(parent)

# insert a string at the beginning
mytext.insert('1.0', "I love my text widget!")

# insert a string into the current text
mytext.insert('1.2', 'REALLY ')

# get the whole string
mytext.get('1.0', tk.END)

# delete the last character.
# Note that there is always a newline character
# at the end of the input, so we backup 2 chars.
mytext.delete('end - 2 chars')
```

运行上述代码，您将获得以下输出：

![](assets/52c2e54d-1453-4eee-99a0-70c08c082001.png)

在这个表单中的“Notes”字段中，我们只需要一个简单的多行“Entry”；所以，我们现在只会使用“Text”小部件的最基本功能。

# 按钮小部件

“ttk.Button”小部件也应该很熟悉。它只是一个可以用鼠标或空格键单击的简单按钮，如下截图所示：

![](assets/83d80f01-f102-4d59-878d-c674bf8a3d54.png)

就像“Checkbutton”小部件一样，此小部件使用“text”和“textvariable”配置选项来控制按钮上的标签。`Button`对象不接受`variable`，但它们确实接受`command`参数，该参数指定单击按钮时要运行的Python函数。

以下示例显示了“Button”对象的使用：

```py
tvar = tk.StringVar()
def swaptext():
    if tvar.get() == 'Hi':
        tvar.set('There')
    else:
        tvar.set('Hi')

my_button = ttk.Button(parent, textvariable=tvar, command=swaptext)
```

# LabelFrame小部件

我们选择了“ttk.LabelFrame”小部件来对我们的应用程序中的字段进行分组。顾名思义，它是一个带有标签的`Frame`（通常带有一个框）。`LabelFrame`小部件在构造函数中接受一个`text`参数，用于设置标签，该标签位于框架的左上角。

Tkinter和“ttk”包含许多其他小部件，其中一些我们将在本书的后面遇到。Python还附带了一个名为“tix”的小部件库，其中包含几十个小部件。但是，“tix”已经非常过时，我们不会在本书中涵盖它。不过，您应该知道它的存在。

# 实现应用程序

要启动我们的应用程序脚本，请创建一个名为“ABQ data entry”的文件夹，并在其中创建一个名为“data_entry_app.py”的文件。

我们将从以下样板代码开始：

```py
import tkinter as tk
from tkinter import ttk

# Start coding here

class Application(tk.Tk):
    """Application root window"""

if __name__ == "__main__":
    app = Application()
    app.mainloop()
```

运行此脚本应该会给您一个空白的Tk窗口。

# 使用LabelInput类节省一些时间

我们表单上的每个“input”小部件都有一个与之关联的标签。在一个小应用程序中，我们可以分别创建标签和输入，然后将每个标签添加到“parent”框架中，如下所示：

```py
form = Frame()
label = Label(form, text='Name')
name_input = Entry(form)
label.grid(row=0, column=0)
name_input.grid(row=1, column=0)
```

这样做很好，你可以为你的应用程序这样做，但它也会创建大量乏味、重复的代码，并且移动输入意味着改变两倍的代码。由于`label`和`input`小部件是一起的，创建一个小的包装类来包含它们并建立一些通用默认值会很聪明。

在编码时，要注意包含大量重复代码的部分。您通常可以将此代码抽象为类、函数或循环。这样做不仅可以节省您的输入，还可以确保一致性，并减少您需要维护的代码总量。

让我们看看以下步骤：

1.  我们将这个类称为`LabelInput`，并在我们的代码顶部定义它，就在`Start coding here`注释下面：

```py
"""Start coding here"""
class LabelInput(tk.Frame):
    """A widget containing a label and input together."""

    def __init__(self, parent, label='', input_class=ttk.Entry,
         input_var=None, input_args=None, label_args=None,
         **kwargs):
        super().__init__(parent, **kwargs)
        input_args = input_args or {}
        label_args = label_args or {}
        self.variable = input_var
```

1.  我们将基于`Tkinter.Frame`类，就像我们在`HelloWidget`中所做的一样。我们的构造函数接受以下参数：

+   `parent`：这个参数是对`parent`小部件的引用；我们创建的所有小部件都将以此作为第一个参数。

+   `label`：这是小部件标签部分的文本。

+   `input_class`：这是我们想要创建的小部件类。它应该是一个实际的可调用类对象，而不是一个字符串。如果留空，将使用`ttk.Entry`。

+   `input_var`：这是一个Tkinter变量，用于分配输入。这是可选的，因为有些小部件不使用变量。

+   `input_args`：这是`input`构造函数的任何额外参数的可选字典。

+   `label_args`：这是`label`构造函数的任何额外参数的可选字典。

+   `**kwargs`：最后，我们在`**kwargs`中捕获任何额外的关键字参数。这些将传递给`Frame`构造函数。

1.  在构造函数中，我们首先调用`super().__init__()`，并传入`parent`和额外的关键字参数。然后，我们确保`input_args`和`label_args`都是字典，并将我们的输入变量保存为`self.variable`的引用。

不要诱使使用空字典（`{}`）作为方法关键字参数的默认值。如果这样做，当方法定义被评估时会创建一个字典，并被类中的所有对象共享。这会对您的代码产生一些非常奇怪的影响！接受的做法是对于可变类型如字典和列表，传递`None`，然后在方法体中用空容器替换`None`。

1.  我们希望能够使用任何类型的`input`小部件，并在我们的类中适当处理它；不幸的是，正如我们之前学到的那样，不同小部件类的构造函数参数和行为之间存在一些小差异，比如`Combobox`和`Checkbutton`使用它们的`textvariable`参数的方式。目前，我们只需要区分`Button`和`Checkbutton`等按钮小部件处理变量和标签文本的方式。为了处理这个问题，我们将添加以下代码：

```py
        if input_class in (ttk.Checkbutton, ttk.Button, 
        ttk.Radiobutton):
            input_args["text"] = label
            input_args["variable"] = input_var
        else:
            self.label = ttk.Label(self, text=label, **label_args)
            self.label.grid(row=0, column=0, sticky=(tk.W + tk.E))
            input_args["textvariable"] = input_var
```

1.  对于按钮类型的小部件，我们以不同的方式执行以下任务：

+   我们不是添加一个标签，而是设置`text`参数。所有按钮都使用这个参数来添加一个`label`到小部件中。

+   我们将变量分配给`variable`，而不是分配给`textvariable`。

1.  对于其他`input`类，我们设置`textvariable`并创建一个`Label`小部件，将其添加到`LabelInput`类的第一行。

1.  现在我们需要创建`input`类，如下所示：

```py
        self.input = input_class(self, **input_args)
        self.input.grid(row=1, column=0, sticky=(tk.W + tk.E))
```

1.  这很简单：我们用扩展为关键字参数的`input_args`字典调用传递给构造函数的`input_class`类。然后，我们将其添加到第`1`行的网格中。

1.  最后，我们配置`grid`布局，将我们的单列扩展到整个小部件，如下所示：

```py
        self.columnconfigure(0, weight=1)
```

1.  当创建自定义小部件时，我们可以做的一件好事是为其几何管理器方法添加默认值，这将节省我们大量的编码。例如，我们将希望所有的`LabelInput`对象填充它们所放置的整个网格单元。我们可以通过覆盖方法将`sticky=(tk.W + tk.E)`添加为默认值，而不是在每个`LabelInput.grid()`调用中添加它：

```py
    def grid(self, sticky=(tk.E + tk.W), **kwargs):
        super().grid(sticky=sticky, **kwargs)
```

通过将其定义为默认参数，我们仍然可以像往常一样覆盖它。所有`input`小部件都有一个`get()`方法，返回它们当前的值。为了节省一些重复的输入，我们将在`LabelInput`类中实现一个`get()`方法，它将简单地将请求传递给输入或其变量。接下来添加这个方法：

```py
    def get(self):
        try:
            if self.variable:
                return self.variable.get()
            elif type(self.input) == tk.Text:
                return self.input.get('1.0', tk.END)
            else:
                return self.input.get()
        except (TypeError, tk.TclError):
            # happens when numeric fields are empty.
            return ''
```

我们在这里使用`try`块，因为在某些条件下，例如当数字字段为空时（空字符串无法转换为数字值），Tkinter变量将抛出异常，如果调用`get()`。在这种情况下，我们将简单地从表单中返回一个空值。此外，我们需要以不同的方式处理`tk.Text`小部件，因为它们需要一个范围来检索文本。我们总是希望从这个表单中获取所有文本，所以我们在这里指定。作为`get()`的补充，我们将实现一个`set()`方法，将请求传递给变量或小部件，如下所示：

```py
    def set(self, value, *args, **kwargs):
        if type(self.variable) == tk.BooleanVar:
                self.variable.set(bool(value))
        elif self.variable:
                self.variable.set(value, *args, **kwargs)
        elif type(self.input) in (ttk.Checkbutton, 
        ttk.Radiobutton):
            if value:
                self.input.select()
            else:
                self.input.deselect()
        elif type(self.input) == tk.Text:
            self.input.delete('1.0', tk.END)
            self.input.insert('1.0', value)
        else: # input must be an Entry-type widget with no variable
            self.input.delete(0, tk.END)
            self.input.insert(0, value)
```

`.set()`方法抽象了各种Tkinter小部件设置其值的差异：

+   如果我们有一个`BooleanVar`类的变量，将`value`转换为`bool`并设置它。`BooleanVar.set()`只接受`bool`，而不是其他假值或真值。这确保我们的变量只获得实际的布尔值。

+   如果我们有任何其他类型的变量，只需将`value`传递给其`.set()`方法。

+   如果我们没有变量，并且是一个按钮样式的类，我们使用`.select()`和`.deselect()`方法来根据变量的真值选择和取消选择按钮。

+   如果它是一个`tk.Text`类，我们可以使用它的`.delete`和`.insert`方法。

+   否则，我们使用`input`的`.delete`和`.insert`方法，这些方法适用于`Entry`、`Spinbox`和`Combobox`类。我们必须将这个与`tk.Text`输入分开，因为索引值的工作方式不同。

这可能并不涵盖每种可能的`input`小部件，但它涵盖了我们计划使用的以及我们以后可能需要的一些。虽然构建`LabelInput`类需要很多工作，但我们将看到现在定义表单要简单得多。

# 构建表单

我们不直接在主应用程序窗口上构建我们的表单，而是将我们的表单构建为自己的对象。最初，这样做可以更容易地维护一个良好的布局，而在将来，这将使我们更容易扩展我们的应用程序。让我们执行以下步骤来构建我们的表单：

1.  一旦再次子类化`Tkinter.Frame`来构建这个模块。在`LabelInput`类定义之后，开始一个新的类，如下所示：

```py
class DataRecordForm(tk.Frame):
    """The input form for our widgets"""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
```

这应该现在很熟悉了。我们子类化`Frame`，定义我们的构造函数，并调用`super().__init__()`来初始化底层的`Frame`对象。

1.  现在我们将创建一个结构来保存表单中所有`input`小部件的引用，如下所示：

```py
        # A dict to keep track of input widgets
        self.inputs = {}
```

在创建`input`小部件时，我们将在字典中存储对它们的引用，使用字段名作为键。这将使我们以后更容易检索所有的值。

# 添加LabelFrame和其他小部件

我们的表单被分成了带有标签和框的各个部分。对于每个部分，我们将创建一个`LabelFrame`小部件，并开始向其中添加我们的`LabelInput`小部件，执行以下步骤：

1.  让我们从执行以下代码开始记录信息框架：

```py
        recordinfo = tk.LabelFrame(self, text="Record Information")
```

记住，`LabelFrame`的`text`参数定义了标签的文本。这个小部件将作为记录信息组中所有输入的`parent`小部件传递。

1.  现在，我们将添加`input`小部件的第一行，如下所示：

```py
        self.inputs['Date'] = LabelInput(recordinfo, "Date",
            input_var=tk.StringVar())
        self.inputs['Date'].grid(row=0, column=0)

        self.inputs['Time'] = LabelInput(recordinfo, "Time",
            input_class=ttk.Combobox, input_var=tk.StringVar(),
            input_args={"values": ["8:00", "12:00", "16:00", "20:00"]})
        self.inputs['Time'].grid(row=0, column=1)

        self.inputs['Technician'] = LabelInput(recordinfo, 
        "Technician",
            input_var=tk.StringVar())
        self.inputs['Technician'].grid(row=0, column=2)
```

1.  `Date`和`Technician`输入是简单的文本输入；我们只需要将`parent`，`label`和`input`变量传递给我们的`LabelInput`构造函数。对于`Time`输入，我们指定一个可能值的列表，这些值将用于初始化`Combobox`小部件。

1.  让我们按照以下方式处理第2行：

```py
        # line 2
        self.inputs['Lab'] = LabelInput(recordinfo, "Lab",
            input_class=ttk.Combobox, input_var=tk.StringVar(),
            input_args={"values": ["A", "B", "C", "D", "E"]})
        self.inputs['Lab'].grid(row=1, column=0)

       self.inputs['Plot'] = LabelInput(recordinfo, "Plot",
            input_class=ttk.Combobox, input_var=tk.IntVar(),
           input_args={"values": list(range(1, 21))})
        self.inputs['Plot'].grid(row=1, column=1)

        self.inputs['Seed sample'] = LabelInput(
            recordinfo, "Seed sample", input_var=tk.StringVar())
        self.inputs['Seed sample'].grid(row=1, column=2)

        recordinfo.grid(row=0, column=0, sticky=tk.W + tk.E)
```

1.  这里，我们有两个`Combobox`小部件和另一个`Entry`。这些创建方式与第1行中的方式类似。`Plot`的值只需要是1到20的数字列表；我们可以使用Python内置的`range()`函数创建它。完成记录信息后，我们通过调用`grid()`将其`LabelFrame`添加到表单小部件。其余字段以基本相同的方式定义。例如，我们的环境数据将如下所示：

```py
        # Environment Data
        environmentinfo = tk.LabelFrame(self, text="Environment Data")
        self.inputs['Humidity'] = LabelInput(
            environmentinfo, "Humidity (g/m³)",
            input_class=tk.Spinbox, input_var=tk.DoubleVar(),
            input_args={"from_": 0.5, "to": 52.0, "increment": .01})
        self.inputs['Humidity'].grid(row=0, column=0)
```

1.  在这里，我们添加了我们的第一个`Spinbox`小部件，指定了有效范围和增量；您可以以相同的方式添加`Light`和`Temperature`输入。请注意，我们的`grid()`坐标已经从`0, 0`重新开始；这是因为我们正在开始一个新的父对象，所以坐标重新开始。

所有这些嵌套的网格可能会让人困惑。请记住，每当在小部件上调用`.grid()`时，坐标都是相对于小部件父级的左上角。父级的坐标是相对于其父级的，依此类推，直到根窗口。

这一部分还包括唯一的`Checkbutton`小部件：

```py
        self.inputs['Equipment Fault'] = LabelInput(
            environmentinfo, "Equipment Fault",
            input_class=ttk.Checkbutton,
            input_var=tk.BooleanVar())
        self.inputs['Equipment Fault'].grid(
            row=1, column=0, columnspan=3)
```

1.  对于`Checkbutton`，没有真正的参数可用，尽管请注意我们使用`BooleanVar`来存储其值。现在，我们继续进行植物数据部分：

```py
        plantinfo = tk.LabelFrame(self, text="Plant Data")

        self.inputs['Plants'] = LabelInput(
            plantinfo, "Plants",
            input_class=tk.Spinbox,
            input_var=tk.IntVar(),
            input_args={"from_": 0, "to": 20})
        self.inputs['Plants'].grid(row=0, column=0)

        self.inputs['Blossoms'] = LabelInput(
            plantinfo, "Blossoms",
            input_class=tk.Spinbox,
            input_var=tk.IntVar(),
            input_args={"from_": 0, "to": 1000})
        self.inputs['Blossoms'].grid(row=0, column=1)
```

请注意，与我们的十进制`Spinboxes`不同，我们没有为整数字段设置增量；这是因为它默认为`1.0`，这正是我们想要的整数字段。

1.  尽管从技术上讲`Blossoms`没有最大值，但我们也使用`1000`作为最大值；我们的`Lab` `Technicians`向我们保证它永远不会接近1000。由于`Spinbox`需要`to`和`from_`，如果我们使用其中一个，我们将使用这个值。

您还可以指定字符串`infinity`或`-infinity`作为值。这些可以转换为`float`值，其行为是适当的。

1.  `Fruit`字段和三个`Height`字段将与这些基本相同。继续创建它们，确保遵循适当的`input_args`值和`input_var`类型的数据字典。通过添加以下注释完成我们的表单字段：

```py
# Notes section
self.inputs['Notes'] = LabelInput(
    self, "Notes",
    input_class=tk.Text,
    input_args={"width": 75, "height": 10}
)
self.inputs['Notes'].grid(sticky="w", row=3, column=0)
```

1.  这里不需要`LabelFrame`，因此我们只需将注释的`LabelInput`框直接添加到表单中。`Text`小部件采用`width`和`height`参数来指定框的大小。我们将为注释输入提供一个非常大的尺寸。

# 从我们的表单中检索数据

现在我们已经完成了表单，我们需要一种方法来从中检索数据，以便应用程序对其进行处理。我们将创建一个返回表单数据字典的方法，并且与我们的`LabelInput`对象一样，遵循Tkinter的约定将其命名为`get()`。

在你的表单类中添加以下方法：

```py
    def get(self):
        data = {}
        for key, widget in self.inputs.items():
            data[key] = widget.get()
        return data
```

代码很简单：我们遍历包含我们的`LabelInput`对象的实例的`inputs`对象，并通过对每个变量调用`get()`来构建一个新字典。

这段代码展示了可迭代对象和一致命名方案的强大之处。如果我们将输入存储为表单的离散属性，或者忽略了规范化`get()`方法，我们的代码将不够优雅。

# 重置我们的表单

我们的表单类几乎完成了，但还需要一个方法。在每次保存表单后，我们需要将其重置为空字段；因此，让我们通过执行以下步骤添加一个方法来实现：

1.  将此方法添加到表单类的末尾：

```py
    def reset(self):
        for widget in self.inputs.values():
            widget.set('')
```

1.  与我们的`get()`方法一样，我们正在遍历`input`字典并将每个`widget`设置为空值。

1.  为了确保我们的应用程序行为一致，我们应该在应用程序加载后立即调用`reset()`，清除我们可能不想要的任何`Tk`默认设置。

1.  回到`__init__()`的最后一行，并添加以下代码行：

```py
        self.reset()
```

# 构建我们的应用程序类

让我们看看构建我们的应用程序类的以下步骤：

1.  在`Application`类文档字符串（读作`Application root window`的行）下面移动，并开始为`Application`编写一个`__init__()`方法，如下所示：

```py
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("ABQ Data Entry Application")
        self.resizable(width=False, height=False)
```

1.  再次调用`super().__init__()`，传递任何参数或关键字参数。

请注意，我们这里没有传入`parent`小部件，因为`Application`是根窗口。

1.  我们调用`.title()`来设置我们应用程序的标题字符串；这不是必需的，但它肯定会帮助运行多个应用程序的用户快速在他们的桌面环境中找到我们的应用程序。

1.  我们还通过调用`self.resizable`禁止窗口的调整大小。这也不是严格必要的，但它使我们暂时更容易控制我们的布局。让我们开始添加我们的应用程序组件，如下所示：

```py
        ttk.Label(
            self,
            text="ABQ Data Entry Application",
            font=("TkDefaultFont", 16)
        ).grid(row=0)
```

1.  应用程序将从顶部开始，显示一个`Label`对象，以比正常字体大的字体显示应用程序的名称。请注意，我们这里没有指定`column`；我们的主应用程序布局只有一列，所以没有必要严格指定`column`，因为它默认为`0`。接下来，我们将添加我们的`DataRecordForm`如下：

```py
        self.recordform = DataRecordForm(self)
        self.recordform.grid(row=1, padx=10)
```

1.  我们使用`padx`参数向左和向右添加了10像素的填充。这只是在表单的边缘周围添加了一些空白，使其更易读。

1.  接下来，让我们添加保存按钮，如下所示：

```py
        self.savebutton = ttk.Button(self, text="Save", 
        command=self.on_save)
        self.savebutton.grid(sticky=tk.E, row=2, padx=10)
```

1.  我们给按钮一个`command`值为`self.on_save`；我们还没有编写该方法，所以在运行代码之前我们需要这样做。

当编写用于GUI事件的方法或函数时，惯例是使用格式`on_EVENTNAME`，其中`EVENTNAME`是描述触发它的事件的字符串。我们也可以将此方法命名为`on_save_button_click()`，但目前`on_save()`就足够了。

1.  最后，让我们添加状态栏，如下所示：

```py
        # status bar
        self.status = tk.StringVar()
        self.statusbar = ttk.Label(self, textvariable=self.status)
        self.statusbar.grid(sticky=(tk.W + tk.E), row=3, padx=10)
```

1.  我们首先创建一个名为`self.status`的字符串变量，并将其用作`ttk.Label`的`textvariable`。我们的应用程序只需要在类内部调用`self.status.set()`来更新状态。通过将状态栏添加到应用程序小部件的底部，我们的GUI完成了。

# 保存到CSV

当用户点击保存时，需要发生以下一系列事件：

1.  打开一个名为`abq_data_record_CURRENTDATE.csv`的文件

1.  如果文件不存在，它将被创建，并且字段标题将被写入第一行

1.  数据字典从`DataEntryForm`中检索

1.  数据被格式化为CSV行并附加到文件

1.  表单被清除，并通知用户记录已保存

我们将需要一些其他Python库来帮助我们完成这个任务：

1.  首先，我们需要一个用于我们文件名的日期字符串。Python的`datetime`库可以帮助我们。

1.  接下来，我们需要能够检查文件是否存在。Python的`os`库有一个用于此的函数。

1.  最后，我们需要能够写入CSV文件。Python在标准库中有一个CSV库，这里非常适用。

让我们看看以下步骤：

1.  回到文件顶部，并在Tkinter导入之前添加以下导入：

```py
from datetime import datetime
import os
import csv
```

1.  现在，回到`Application`类，并开始`on_save()`方法，如下所示：

```py
   def on_save(self):
        datestring = datetime.today().strftime("%Y-%m-%d")
        filename = "abq_data_record_{}.csv".format(datestring)
        newfile = not os.path.exists(filename)
```

1.  我们要做的第一件事是创建我们的日期字符串。`datetime.today()`方法返回当前日期的午夜`datetime`；然后我们使用`strftime()`将其格式化为年-月-日的ISO日期字符串（使用数字01到12表示月份）。这将被插入到我们规范的文件名模板中，并保存为`filename`。

1.  接下来，我们需要确定文件是否已经存在；`os.path.exists()`将返回一个布尔值，指示文件是否存在；我们对这个值取反，并将其存储为`newfile`。

1.  现在，让我们从`DataEntryForm`获取数据：

```py
        data = self.recordform.get()
```

1.  获得数据后，我们需要打开文件并将数据写入其中。添加以下代码：

```py
        with open(filename, 'a') as fh:
            csvwriter = csv.DictWriter(fh, fieldnames=data.keys())
            if newfile:
                csvwriter.writeheader()
            csvwriter.writerow(data)
```

`with open(filename, 'a') as fh:`语句以追加模式打开我们生成的文件名，并为我们提供一个名为`fh`的文件句柄。追加模式意味着我们不能读取或编辑文件中的任何现有行，只能添加到文件的末尾，这正是我们想要的。

`with`关键字与**上下文管理器**对象一起使用，我们调用`open()`返回的就是这样的对象。上下文管理器是特殊的对象，它定义了在`with`块之前和之后要运行的代码。通过使用这种方法打开文件，它们将在块结束时自动正确关闭。

1.  接下来，我们使用文件句柄创建一个`csv.DictWriter`对象。这个对象将允许我们将数据字典写入CSV文件，将字典键与CSV的标题行标签匹配。这对我们来说比默认的CSV写入对象更好，后者每次都需要正确顺序的字段。

1.  要配置这一点，我们首先必须将`fieldnames`参数传递给`DictWriter`构造函数。我们的字段名称是从表单中获取的`data`字典的键。如果我们正在处理一个新文件，我们需要将这些字段名称写入第一行，我们通过调用`DictWriter.writeheader()`来实现。

1.  最后，我们使用`DictWriter`对象的`.writerow()`方法将我们的`data`字典写入新行。在代码块的末尾，文件会自动关闭和保存。

# 完成和测试

此时，您应该能够运行应用程序，输入数据，并将其保存到CSV文件中。试试看！您应该会看到类似以下截图的东西：

![](assets/9708ab24-6d9f-4276-935b-454f6110dc31.png)

也许您注意到的第一件事是，单击保存没有明显的效果。表单保持填充状态，没有任何指示已经完成了什么。我们应该修复这个问题。

我们将执行以下两件事来帮助这里：

1.  首先，在我们的状态栏中放置一个通知，说明记录已保存以及本次会话已保存多少条记录。对于第一部分，将以下代码行添加到`Application`构造函数的末尾，如下所示：

```py
        self.records_saved = 0
```

1.  其次，在保存后清除表单，以便可以开始下一个记录。然后将以下代码行添加到`on_save()`方法的末尾，如下所示：

```py
        self.records_saved += 1
        self.status.set(
            "{} records saved this session".format(self.records_saved))
```

这段代码设置了一个计数器变量，用于跟踪自应用程序启动以来保存的记录数。

1.  保存文件后，我们增加值，然后设置我们的状态以指示已保存多少条记录。用户将能够看到这个数字增加，并知道他们的按钮点击已经做了一些事情。

1.  接下来，我们将在保存后重置表单。将以下代码追加到`Application.on_save()`的末尾，如下所示：

```py
        self.recordform.reset()
```

这将清空表单，并准备好下一个记录的输入。

1.  现在，再次运行应用程序。它应该清除并在保存记录时给出状态指示。

# 摘要

嗯，我们在这一章取得了长足的进步！您将您的设计从规范和一些图纸转化为一个运行的应用程序，它已经涵盖了您需要的基本功能。您学会了如何使用基本的Tkinter和`ttk`小部件，并创建自定义小部件，以节省大量重复的工作。

在下一章中，我们将解决`input`小部件的问题。我们将学习如何自定义`input`小部件的行为，防止错误的按键，并验证数据，以确保它在我们规范中规定的容差范围内。在此过程中，我们将深入研究Python类，并学习更多高效和优雅的代码技巧。
