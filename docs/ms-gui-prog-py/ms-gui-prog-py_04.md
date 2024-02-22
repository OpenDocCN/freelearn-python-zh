# 第三章：使用信号和插槽处理事件

将小部件组合成一个漂亮的表单是设计应用程序的一个很好的第一步，但是为了 GUI 能够发挥作用，它需要连接到实际执行操作的代码。为了在 PyQt 中实现这一点，我们需要了解 Qt 最重要的功能之一，**信号和插槽**。

在本章中，我们将涵盖以下主题：

+   信号和插槽基础

+   创建自定义信号和插槽

+   自动化我们的日历表单

# 技术要求

除了第一章中列出的基本要求外，*使用 PyQt 入门*，您还需要来自第二章*使用 QtWidgets 构建全面表单*的日历表单代码和 Qt Designer 文件。您可能还希望从我们的 GitHub 存储库[`github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter03`](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter03)下载示例代码。

查看以下视频，看看代码是如何运行的：[`bit.ly/2M5OFQo`](http://bit.ly/2M5OFQo)

# 信号和插槽基础

**信号**是对象的特殊属性，可以在对应的**事件**类型中发出。事件可以是用户操作、超时或异步方法调用的完成等。

**插槽**是可以接收信号并对其做出响应的对象方法。我们连接信号到插槽，以配置应用程序对事件的响应。

所有从`QObject`继承的类（这包括 Qt 中的大多数类，包括所有`QWidget`类）都可以发送和接收信号。每个不同的类都有适合该类功能的一组信号和插槽。

例如，`QPushButton`有一个`clicked`信号，每当用户点击按钮时就会发出。`QWidget`类有一个`close()`插槽，如果它是顶级窗口，就会导致它关闭。我们可以这样连接两者：

```py
self.quitbutton = qtw.QPushButton('Quit')
self.quitbutton.clicked.connect(self.close)
self.layout().addWidget(self.quitbutton)
```

如果您将此代码复制到我们的应用程序模板中并运行它，您会发现单击“退出”按钮会关闭窗口并结束程序。在 PyQt5 中连接信号到插槽的语法是`object1.signalName.connect(object2.slotName)`。

您还可以在创建对象时通过将插槽作为关键字参数传递给信号来进行连接。例如，前面的代码可以重写如下：

```py
self.quitbutton = qtw.QPushButton('Quit', clicked=self.close)
self.layout().addWidget(self.quitbutton)
```

C++和旧版本的 PyQt 使用非常不同的信号和插槽语法，它使用`SIGNAL()`和`SLOT()`包装函数。这些在 PyQt5 中不存在，所以如果您在遵循旧教程或非 Python 文档，请记住这一点。

信号还可以携带数据，插槽可以接收。例如，`QLineEdit`有一个`textChanged`信号，随信号发送进小部件的文本一起。该行编辑还有一个接受字符串参数的`setText()`插槽。我们可以这样连接它们：

```py
self.entry1 = qtw.QLineEdit()
self.entry2 = qtw.QLineEdit()
self.layout().addWidget(self.entry1)
self.layout().addWidget(self.entry2)
self.entry1.textChanged.connect(self.entry2.setText)
```

在这个例子中，我们将`entry1`的`textChanged`信号连接到`entry2`的`setText()`插槽。这意味着每当`entry1`中的文本发生变化时，它将用输入的文本信号`entry2`；`entry2`将把自己的文本设置为接收到的字符串，导致它镜像`entry1`中输入的任何内容。

在 PyQt5 中，插槽不必是官方的 Qt 插槽方法；它可以是任何 Python 可调用对象，比如自定义方法或内置函数。例如，让我们将`entry2`小部件的`textChanged`连接到老式的`print()`：

```py
self.entry2.textChanged.connect(print)
```

现在，您会发现对`entry2`的每次更改都会打印到控制台。`textChanged`信号基本上每次触发时都会调用`print()`，并传入信号携带的文本。

信号甚至可以连接到其他信号，例如：

```py
self.entry1.editingFinished.connect(lambda: print('editing finished'))
self.entry2.returnPressed.connect(self.entry1.editingFinished)
```

我们已经将`entry2`小部件的`returnPressed`信号（每当用户在小部件上按下*return*/*Enter*时发出）连接到`entry1`小部件的`editingFinished`信号，而`editingFinished`信号又连接到一个打印消息的`lambda`函数。当你连接一个信号到另一个信号时，事件和数据会从一个信号传递到下一个信号。最终结果是在`entry2`上触发`returnPressed`会导致`entry1`发出`editingFinished`，然后运行`lambda`函数。

# 信号和槽连接的限制

尽管 PyQt 允许我们将信号连接到任何 Python 可调用对象，但有一些规则和限制需要牢记。与 Python 不同，C++是一种**静态类型**语言，这意味着变量和函数参数必须给定一个类型（`string`、`integer`、`float`或许多其他类型），并且存储在变量中或传递给该函数的任何值必须具有匹配的类型。这被称为**类型安全**。

原生的 Qt 信号和槽是类型安全的。例如，假设我们尝试将行编辑的`textChanged`信号连接到按钮的`clicked`信号，如下所示：

```py
self.entry1.textChanged.connect(self.quitbutton.clicked)
```

这是行不通的，因为`textChanged`发出一个字符串，而`clicked`发出（并且因此期望接收）一个布尔值。如果你运行这个，你会得到这样的错误：

```py
QObject::connect: Incompatible sender/receiver arguments
        QLineEdit::textChanged(QString) --> QPushButton::clicked(bool)
Traceback (most recent call last):
  File "signal_slots_demo.py", line 57, in <module>
    mw = MainWindow()
  File "signal_slots_demo.py", line 32, in __init__
    self.entry1.textChanged.connect(self.quitbutton.clicked)
TypeError: connect() failed between textChanged(QString) and clicked()
```

槽可以有多个实现，每个实现都有自己的**签名**，允许相同的槽接受不同的参数类型。这被称为**重载**槽。只要我们的信号签名与任何重载的槽匹配，我们就可以建立连接，Qt 会确定我们连接到哪一个。

当连接到一个是 Python 函数的槽时，我们不必担心参数类型，因为 Python 是**动态类型**的（尽管我们需要确保我们的 Python 代码对传递给它的任何对象都做正确的事情）。然而，与对 Python 函数的任何调用一样，我们确实需要确保传入足够的参数来满足函数签名。

例如，让我们向`MainWindow`类添加一个方法，如下所示：

```py
def needs_args(self, arg1, arg2, arg3):
        pass
```

这个实例方法需要三个参数（`self`会自动传递）。让我们尝试将按钮的`clicked`信号连接到它：

```py
self.badbutton = qtw.QPushButton("Bad")
self.layout().addWidget(self.badbutton)
self.badbutton.clicked.connect(self.needs_args)
```

这段代码本身并不反对连接，但当你点击按钮时，程序会崩溃并显示以下错误：

```py
TypeError: needs_args() missing 2 required positional arguments: 'arg2' and 'arg3'
Aborted (core dumped)
```

由于`clicked`信号只发送一个参数，函数调用是不完整的，会抛出异常。可以通过将`arg2`和`arg3`变成关键字参数（添加默认值），或者创建一个以其他方式填充它们的包装函数来解决这个问题。

顺便说一句，槽接收的参数比信号发送的参数少的情况并不是问题。Qt 只是从信号中丢弃额外的数据。

因此，例如，将`clicked`连接到一个没有参数的方法是没有问题的，如下所示：

```py
        # inside __init__()
        self.goodbutton = qtw.QPushButton("Good")
        self.layout().addWidget(self.goodbutton)
        self.goodbutton.clicked.connect(self.no_args)
        # ...

    def no_args(self):
        print('I need no arguments')
```

# 创建自定义信号和槽

为按钮点击和文本更改设置回调是信号和槽的常见和非常明显的用法，但这实际上只是开始。在本质上，信号和槽机制可以被看作是应用程序中任何两个对象进行通信的一种方式，同时保持**松散耦合**。

松散耦合是指保持两个对象彼此需要了解的信息量最少。这是设计大型复杂应用程序时必须保留的重要特性，因为它隔离了代码并防止意外的破坏。相反的是紧密耦合，其中一个对象的代码严重依赖于另一个对象的内部结构。

为了充分利用这一功能，我们需要学习如何创建自己的自定义信号和槽。

# 使用自定义信号在窗口之间共享数据

假设您有一个弹出表单窗口的程序。当用户完成填写表单并提交时，我们需要将输入的数据传回主应用程序类进行处理。我们可以采用几种方法来解决这个问题；例如，主应用程序可以监视弹出窗口的**提交**按钮的单击事件，然后在销毁对话框之前从其字段中获取数据。但这种方法要求主窗体了解弹出对话框的所有部件，而且任何对弹出窗口的重构都可能破坏主应用程序窗口中的代码。

让我们尝试使用信号和槽的不同方法。从第一章中打开我们应用程序模板的新副本，*PyQt 入门*，并开始一个名为`FormWindow`的新类，就像这样：

```py
class FormWindow(qtw.QWidget):

    submitted = qtc.pyqtSignal(str)
```

在这个类中我们定义的第一件事是一个名为`submitted`的自定义信号。要定义自定义信号，我们需要调用`QtCore.pyqtSignal()`函数。`pyqtSignal()`的参数是我们的信号将携带的数据类型，在这种情况下是`str`。我们可以在这里使用 Python `type`对象，或者命名 C++数据类型的字符串（例如`'QString'`）。

现在让我们通过定义`__init__()`方法来构建表单，如下所示：

```py
    def __init__(self):
        super().__init__()
        self.setLayout(qtw.QVBoxLayout())

        self.edit = qtw.QLineEdit()
        self.submit = qtw.QPushButton('Submit', clicked=self.onSubmit)

        self.layout().addWidget(self.edit)
        self.layout().addWidget(self.submit)
```

在这里，我们定义了一个用于数据输入的`QLineEdit`和一个用于提交表单的`QPushButton`。按钮单击信号绑定到一个名为`onSubmit`的方法，我们将在下面定义：

```py
    def onSubmit(self):
        self.submitted.emit(self.edit.text())
        self.close()
```

在这个方法中，我们调用`submitted`信号的`emit()`方法，传入`QLineEdit`的内容。这意味着任何连接的槽都将使用从`self.edit.text()`检索到的字符串进行调用。

发射信号后，我们关闭`FormWindow`。

在我们的`MainWindow`构造函数中，让我们构建一个使用它的应用程序：

```py
    def __init__(self):
        super().__init__()
        self.setLayout(qtw.QVBoxLayout())

        self.label = qtw.QLabel('Click "change" to change this text.')
        self.change = qtw.QPushButton("Change", clicked=self.onChange)
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.change)
        self.show()
```

在这里，我们创建了一个`QLabel`和一个`QPushButton`，并将它们添加到垂直布局中。单击按钮时，按钮调用一个名为`onChange()`的方法。

`onChange()`方法看起来像这样：

```py
    def onChange(self):
        self.formwindow = FormWindow()
        self.formwindow.submitted.connect(self.label.setText)
        self.formwindow.show()
```

这个方法创建了一个`FormWindow`的实例。然后将我们的自定义信号`FormWindow.submitted`绑定到标签的`setText`槽；`setText`接受一个字符串作为参数，而我们的信号发送一个字符串。

如果您运行此应用程序，您会看到当您提交弹出窗口表单时，标签中的文本确实会更改。

这种设计的美妙之处在于`FormWindow`不需要知道任何关于`MainWindow`的东西，而`MainWindow`只需要知道`FormWindow`有一个`submitted`信号，该信号发射输入的字符串。只要相同的信号发射相同的数据，我们可以轻松修改任一类的结构和内部，而不会对另一类造成问题。

`QtCore`还包含一个`pyqtSlot()`函数，我们可以将其用作装饰器，表示 Python 函数或方法旨在作为槽使用。

例如，我们可以装饰我们的`MainWindow.onChange()`方法来声明它为一个槽：

```py
    @qtc.pyqtSlot()
    def onChange(self):
        # ...
```

这纯粹是可选的，因为我们可以使用任何 Python 可调用对象作为槽，尽管这确实给了我们强制类型安全的能力。例如，如果我们希望要求`onChange()`始终接收一个字符串，我们可以这样装饰它：

```py
    @qtc.pyqtSlot(str)
    def onChange(self):
        # ...
```

如果您这样做并运行程序，您会看到我们尝试连接`clicked`信号会失败：

```py
Traceback (most recent call last):
  File "form_window.py", line 47, in <module>
    mw = MainWindow()
  File "form_window.py", line 31, in __init__
    self.change = qtw.QPushButton("Change", clicked=self.onChange)
TypeError: decorated slot has no signature compatible with clicked(bool)
```

除了强制类型安全外，将方法声明为槽还会减少其内存使用量，并提供一点速度上的改进。因此，虽然这完全是可选的，但对于只会被用作槽的方法来说，这可能值得做。

# 信号和槽的重载

就像 C++信号和槽可以被重载以接受不同的参数签名一样，我们也可以重载我们自定义的 PyQt 信号和槽。例如，假设如果在我们的弹出窗口中输入了一个有效的整数字符串，我们希望将其作为字符串和整数发射出去。

为了做到这一点，我们首先必须重新定义我们的信号：

```py
    submitted = qtc.pyqtSignal([str], [int, str])
```

我们不仅传入单个变量类型，而是传入两个变量类型的列表。每个列表代表一个信号签名的参数列表。因此，我们在这里注册了两个信号：一个只发送字符串，一个发送整数和字符串。

在`FormWindow.onSubmit()`中，我们现在可以检查行编辑中的文本，并使用适当的签名发送信号：

```py
    def onSubmit(self):
        if self.edit.text().isdigit():
            text = self.edit.text()
            self.submitted[int, str].emit(int(text), text)
        else:
            self.submitted[str].emit(self.edit.text())
        self.close()
```

在这里，我们测试`self.edit`中的文本，以查看它是否是有效的数字字符串。如果是，我们将其转换为`int`，并使用整数和文本版本的文本发出`submitted`信号。选择签名的语法是在信号名称后跟一个包含参数类型列表的方括号。

回到主窗口，我们将定义两种新方法来处理这些信号：

```py
    @qtc.pyqtSlot(str)
    def onSubmittedStr(self, string):
        self.label.setText(string)

    @qtc.pyqtSlot(int, str)
    def onSubmittedIntStr(self, integer, string):
        text = f'The string {string} becomes the number {integer}'
        self.label.setText(text)
```

我们已经创建了两个插槽——一个接受字符串，另一个接受整数和字符串。现在我们可以将`FormWindow`中的两个信号连接到适当的插槽，如下所示：

```py
    def onChange(self):
        self.formwindow = FormWindow()
        self.formwindow.submitted[str].connect(self.onSubmittedStr)
        self.formwindow.submitted[int, str].connect(self.onSubmittedIntStr)
```

运行脚本，您会发现输入一串数字会打印与字母数字字符串不同的消息。

# 自动化我们的日历表单

要了解信号和插槽在实际应用程序中的使用方式，让我们拿我们在第二章 *使用 QtWidgets 构建表单*中构建的日历表单，并将其转换为一个可工作的日历应用程序。为此，我们需要进行以下更改：

+   应用程序需要一种方法来存储我们输入的事件。

+   全天复选框应在选中时禁用时间输入。

+   在日历上选择一天应该用当天的事件填充事件列表。

+   在事件列表中选择一个事件应该用事件的详细信息填充表单。

+   单击“添加/更新”应该更新保存的事件详细信息，如果选择了事件，或者如果没有选择事件，则添加一个新事件。

+   单击删除应该删除所选事件。

+   如果没有选择事件，删除应该被禁用。

+   选择“新建…”作为类别应该打开一个对话框，允许我们输入一个新的类别。如果我们选择输入一个，它应该被选中。

我们将首先使用我们手工编码的表单进行这一过程，然后讨论如何使用 Qt Designer 文件解决同样的问题。

# 使用我们手工编码的表单

要开始，请将您的`calendar_form.py`文件从第二章 *使用 QtWidgets 构建表单*复制到一个名为`calendar_app.py`的新文件中，并在编辑器中打开它。我们将开始编辑我们的`MainWindow`类，并将其完善为一个完整的应用程序。

为了处理存储事件，我们将在`MainWindow`中创建一个`dict`属性，如下所示：

```py
class MainWindow(qtw.QWidget):

    events = {}
```

我们不打算将数据持久化到磁盘，尽管如果您愿意，您当然可以添加这样的功能。`dict`中的每个项目将使用`date`对象作为其键，并包含一个包含该日期上所有事件详细信息的`dict`对象列表。数据的布局将看起来像这样：

```py
    events = {
        QDate:  {
            'title': "String title of event",
            'category': "String category of event",
            'time': QTime() or None if "all day",
            'detail':  "String details of event"
        }
    }
```

接下来，让我们深入研究表单自动化。最简单的更改是在单击“全天”复选框时禁用时间输入，因为这种自动化只需要处理内置信号和插槽。

在`__init__()`方法中，我们将添加这段代码：

```py
        self.allday_check.toggled.connect(self.event_time.setDisabled)
```

`QCheckBox.toggled`信号在复选框切换开或关时发出，并发送一个布尔值，指示复选框是（更改后）未选中（`False`）还是选中（`True`）。这与`setDisabled`很好地连接在一起，它将在`True`时禁用小部件，在`False`时启用它。

# 创建和连接我们的回调方法

我们需要的其余自动化不适用于内置的 Qt 插槽，因此在连接更多信号之前，我们需要创建一些将用于实现插槽的方法。我们将把所有这些方法创建为`MainWindow`类的方法。

在开始处理回调之前，我们将创建一个实用方法来清除表单，这是几个回调方法将需要的。它看起来像这样：

```py
    def clear_form(self):
        self.event_title.clear()
        self.event_category.setCurrentIndex(0)
        self.event_time.setTime(qtc.QTime(8, 0))
        self.allday_check.setChecked(False)
        self.event_detail.setPlainText('')
```

基本上，这个方法会遍历我们表单中的字段，并将它们全部设置为默认值。不幸的是，这需要为每个小部件调用不同的方法，所以我们必须把它全部写出来。

现在让我们来看看回调方法。

# populate_list()方法

第一个实际的回调方法是`populate_list()`，它如下所示：

```py
    def populate_list(self):
        self.event_list.clear()
        self.clear_form()
        date = self.calendar.selectedDate()
        for event in self.events.get(date, []):
            time = (
                event['time'].toString('hh:mm')
                if event['time']
                else 'All Day'
            )
            self.event_list.addItem(f"{time}: {event['title']}")
```

这将在日历选择更改时调用，并且其工作是使用该天的事件重新填充`event_list`小部件。它首先清空列表和表单。然后，它使用其`selectedDate()`方法从日历小部件中检索所选日期。

然后，我们循环遍历所选日期的`self.events`字典的事件列表，构建一个包含时间和事件标题的字符串，并将其添加到`event_list`小部件中。请注意，我们的事件时间是一个`QTime`对象，因此要将其用作字符串，我们需要使用它的`toString()`方法进行转换。

有关如何将时间值格式化为字符串的详细信息，请参阅[`doc.qt.io/qt-5/qtime.html`](https://doc.qt.io/qt-5/qtime.html)中的`QTime`文档。

为了连接这个方法，在`__init__()`中，我们添加了这段代码：

```py
        self.calendar.selectionChanged.connect(self.populate_list)
```

`selectionChanged`信号在日历上选择新日期时发出。它不发送任何数据，因此我们的回调函数不需要任何数据。

# populate_form()方法

接下来的回调是`populate_form()`，当选择事件时将调用它并填充事件详细信息表单。它开始如下：

```py
    def populate_form(self):
        self.clear_form()
        date = self.calendar.selectedDate()
        event_number = self.event_list.currentRow()
        if event_number == -1:
            return
```

在这里，我们首先清空表单，然后从日历中检索所选日期，并从事件列表中检索所选事件。当没有选择事件时，`QListWidget.currentRow()`返回值为`-1`；在这种情况下，我们将只是返回，使表单保持空白。

方法的其余部分如下：

```py
        event_data = self.events.get(date)[event_number]

        self.event_category.setCurrentText(event_data['category'])
        if event_data['time'] is None:
            self.allday_check.setChecked(True)
        else:
            self.event_time.setTime(event_data['time'])
        self.event_title.setText(event_data['title'])
        self.event_detail.setPlainText(event_data['detail'])
```

由于列表小部件上显示的项目与`events`字典中存储的顺序相同，因此我们可以使用所选项目的行号来从所选日期的列表中检索事件。

一旦数据被检索，我们只需要将每个小部件设置为保存的值。

回到`__init__()`中，我们将连接槽如下：

```py
        self.event_list.itemSelectionChanged.connect(
            self.populate_form
        )
```

`QListWidget`在选择新项目时发出`itemSelectionChanged`。它不发送任何数据，因此我们的回调函数也不需要任何数据。

# save_event()方法

`save_event()`回调将在单击添加/更新按钮时调用。它开始如下：

```py
    def save_event(self):
        event = {
            'category': self.event_category.currentText(),
            'time': (
                None
                if self.allday_check.isChecked()
                else self.event_time.time()
                ),
            'title': self.event_title.text(),
            'detail': self.event_detail.toPlainText()
            }
```

在这段代码中，我们现在调用访问器方法来从小部件中检索值，并将它们分配给事件字典的适当键。

接下来，我们将检索所选日期的当前事件列表，并确定这是添加还是更新：

```py
        date = self.calendar.selectedDate()
        event_list = self.events.get(date, [])
        event_number = self.event_list.currentRow()

        if event_number == -1:
            event_list.append(event)
        else:
            event_list[event_number] = event
```

请记住，如果没有选择项目，`QListWidget.currentRow()`会返回`-1`。在这种情况下，我们希望将新事件追加到列表中。否则，我们将所选事件替换为我们的新事件字典：

```py
        event_list.sort(key=lambda x: x['time'] or qtc.QTime(0, 0))
        self.events[date] = event_list
        self.populate_list()
```

为了完成这个方法，我们将使用时间值对列表进行排序。请记住，我们对全天事件使用`None`，因此它们将首先通过在排序中用`QTime`的 0:00 替换它们来进行排序。

排序后，我们用新排序的列表替换当前日期的事件列表，并用新列表重新填充`QListWidget`。

我们将通过在`__init__()`中添加以下代码来连接`add_button`小部件的`clicked`事件：

```py
        self.add_button.clicked.connect(self.save_event)
```

# delete_event()方法

`delete_event`方法将在单击删除按钮时调用，它如下所示：

```py
    def delete_event(self):
        date = self.calendar.selectedDate()
        row = self.event_list.currentRow()
        del(self.events[date][row])
        self.event_list.setCurrentRow(-1)
        self.clear_form()
        self.populate_list()
```

再次，我们检索当前日期和当前选择的行，并使用它们来定位我们想要删除的`self.events`中的事件。在从列表中删除项目后，我们通过将`currentRow`设置为`-1`来将列表小部件设置为无选择。然后，我们清空表单并填充列表小部件。

请注意，我们不需要检查当前选择的行是否为`-1`，因为我们计划在没有选择行时禁用删除按钮。

这个回调很容易连接到`__init__()`中的`del_button`：

```py
        self.del_button.clicked.connect(self.delete_event)
```

# 检查`_delete _btn()`方法

我们的最后一个回调是最简单的，它看起来像这样：

```py
    def check_delete_btn(self):
        self.del_button.setDisabled(
            self.event_list.currentRow() == -1)
```

这个方法只是检查当前事件列表小部件中是否没有事件被选中，并相应地启用或禁用删除按钮。

回到`__init__()`，让我们连接到这个回调：

```py
        self.event_list.itemSelectionChanged.connect(
            self.check_delete_btn)
        self.check_delete_btn()
```

我们将这个回调连接到`itemSelectionChanged`信号。请注意，我们已经将该信号连接到另一个插槽。信号可以连接到任意数量的插槽而不会出现问题。我们还直接调用该方法，以便`del_button`一开始就被禁用。

# 构建我们的新类别弹出表单

我们应用程序中的最后一个功能是能够向组合框添加新类别。我们需要实现的基本工作流程是：

1.  当用户更改事件类别时，检查他们是否选择了“新…”

1.  如果是这样，打开一个新窗口中的表单，让他们输入一个类别

1.  当表单提交时，发出新类别的名称

1.  当发出该信号时，向组合框添加一个新类别并选择它

1.  如果用户选择不输入新类别，则将组合框默认为“选择类别…”

让我们从实现我们的弹出表单开始。这将与我们在本章前面讨论过的表单示例一样，它看起来像这样：

```py
class CategoryWindow(qtw.QWidget):

    submitted = qtc.pyqtSignal(str)

    def __init__(self):
        super().__init__(None, modal=True)
        self.setLayout(qtw.QVBoxLayout())
        self.layout().addWidget(
            qtw.QLabel('Please enter a new catgory name:'))
        self.category_entry = qtw.QLineEdit()
        self.layout().addWidget(self.category_entry)
        self.submit_btn = qtw.QPushButton(
            'Submit',
            clicked=self.onSubmit)
        self.layout().addWidget(self.submit_btn)
        self.cancel_btn = qtw.QPushButton(
            'Cancel',
            clicked=self.close
            )
        self.layout().addWidget(self.cancel_btn)
        self.show()

    @qtc.pyqtSlot()
    def onSubmit(self):
        if self.category_entry.text():
            self.submitted.emit(self.category_entry.text())
        self.close()
```

这个类与我们的`FormWindow`类相同，只是增加了一个标签和一个取消按钮。当点击`cancel_btn`小部件时，将调用窗口的`close()`方法，导致窗口关闭而不发出任何信号。

回到`MainWindow`，让我们实现一个方法，向组合框添加一个新类别：

```py
    def add_category(self, category):
        self.event_category.addItem(category)
        self.event_category.setCurrentText(category)
```

这种方法非常简单；它只是接收一个类别文本，将其添加到组合框的末尾，并将组合框选择设置为新类别。

现在我们需要编写一个方法，每当选择“新…”时，它将创建我们弹出表单的一个实例：

```py
    def on_category_change(self, text):
        if text == 'New…':
            dialog = CategoryWindow()
            dialog.submitted.connect(self.add_category)
            self.event_category.setCurrentIndex(0)
```

这种方法接受已更改类别的`text`值，并检查它是否为“新…”。如果是，我们创建我们的`CategoryWindow`对象，并将其`submitted`信号连接到我们的`add_category()`方法。然后，我们将当前索引设置为`0`，这是我们的“选择类别…”选项。

现在，当`CategoryWindow`显示时，用户要么点击取消，窗口将关闭并且组合框将被设置为“选择类别…”，就像`on_category_change()`留下的那样，要么用户将输入一个类别并点击提交，这样`CategoryWindow`将发出一个带有新类别的`submitted`信号。`add_category()`方法将接收到新类别，将其添加，并将组合框设置为它。

我们的日历应用现在已经完成；启动它并试试吧！

# 使用 Qt Designer .ui 文件

现在让我们回过头来使用我们在第二章中创建的 Qt Designer 文件，*使用 QtWidgets 构建表单*。这将需要一种完全不同的方法，但最终产品将是一样的。

要完成本节的工作，您需要第二章中的`calendar_form.ui`文件，*使用 QtWidgets 构建表单*，以及第二个`.ui`文件用于类别窗口。您可以自己练习构建这个表单，也可以使用本章示例代码中包含的表单。如果选择自己构建，请确保将每个对象命名为我们在上一节的代码中所做的那样。

# 在 Qt Designer 中连接插槽

Qt Designer 对于连接信号和插槽到我们的 GUI 的能力有限。对于 Python 开发人员，它主要只能用于在同一窗口中的小部件之间连接内置的 Qt 信号到内置的 Qt 插槽。连接信号到 Python 可调用对象或自定义的 PyQt 信号实际上是不可能的。

在日历 GUI 中，我们确实有一个原生的 Qt 信号-槽连接示例——`allday_check`小部件连接到`event_time`小部件。让我们看看如何在 Qt Designer 中连接这些：

1.  在 Qt Designer 中打开`calendar_form.ui`文件

1.  在屏幕右下角找到 Signal/Slot Editor 面板

1.  点击+图标添加一个新的连接

1.  在 Sender 下，打开弹出菜单，选择`allday_check`

1.  在 Signal 下，选择 toggled(bool)

1.  对于 Receiver，选择`event_time`

1.  最后，对于 Slot，选择 setDisabled(bool)

生成的条目应该是这样的：

![](img/f8145b67-f651-4977-9b7f-6e0c23b01bf5.png)

如果你正在构建自己的`category_window.ui`文件，请确保你还将取消按钮的`clicked`信号连接到类别窗口的`closed`槽。

# 将.ui 文件转换为 Python

如果你在文本编辑器中打开你的`calendar_form.ui`文件，你会看到它既不是 Python 也不是 C++，而是你设计的 GUI 的 XML 表示。PyQt 为我们提供了几种选择，可以在 Python 应用程序中使用`.ui`文件。

第一种方法是使用 PyQt 附带的`pyuic5`工具将 XML 转换为 Python。在存放`.ui`文件的目录中打开命令行窗口，运行以下命令：

```py
$ pyuic5 calendar_form.ui
```

这将生成一个名为`calendar_form.py`的文件。如果你在代码编辑器中打开这个文件，你会看到它包含一个`Ui_MainWindow`类的单个类定义，如下所示：

```py
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(799, 600)
        # ... etc
```

注意这个类既不是`QWidget`的子类，也不是`QObject`的子类。这个类本身不会显示我们构建的窗口。相反，这个类将在另一个小部件内部构建我们设计的 GUI，我们必须用代码创建它。

为了做到这一点，我们将这个类导入到另一个脚本中，创建一个`QWidget`作为容器，并将`setupUi()`方法与我们的小部件容器作为参数一起调用。

不要试图编辑或添加代码到生成的 Python 文件中。如果你想使用 Qt Designer 更新你的 GUI，当你生成新文件时，你会丢失所有的编辑。把生成的代码当作第三方库来对待。

首先，从第一章，*PyQt 入门*中复制 PyQt 应用程序模板到存放`calendar_form.py`的目录，并将其命名为`calendar_app.py`。

在文件顶部像这样导入`Ui_MainWindow`类：

```py
from calendar_form import Ui_MainWindow
```

我们可以以几种方式使用这个类，但最干净的方法是通过将它作为`MainWindow`的第二个父类进行**多重继承**。

更新`MainWindow`类定义如下：

```py
class MainWindow(qtw.QWidget, Ui_MainWindow):
```

注意我们窗口的基类（第一个父类）仍然是`QWidget`。这个基类需要与我们最初设计表单时选择的基类匹配（参见第二章，*使用 QtWidgets 构建表单*）。

现在，在构造函数内部，我们可以调用`setupUi`，像这样：

```py
    def __init__(self):
        super().__init__()
        self.setupUi(self)
```

如果你在这一点运行应用程序，你会看到日历 GUI 都在那里，包括我们在`allday_check`和`event_time`之间的连接。然后，你可以将其余的连接和修改添加到`MainWindow`构造函数中，如下所示：

```py
        # disable the first category item
        self.event_category.model().item(0).setEnabled(False)
        # Populate the event list when the calendar is clicked
        self.calendar.selectionChanged.connect(self.populate_list)
        # Populate the event form when an item is selected
        self.event_list.itemSelectionChanged.connect(
            self.populate_form)
        # Save event when save is hit
        self.add_button.clicked.connect(self.save_event)
        # connect delete button
        self.del_button.clicked.connect(self.delete_event)
        # Enable 'delete' only when an event is selected
        self.event_list.itemSelectionChanged.connect(
            self.check_delete_btn)
        self.check_delete_btn()
        # check for selection of "new…" for category
        self.event_category.currentTextChanged.connect(
            self.on_category_change)
```

这个类的回调方法与我们在代码中定义的方法是相同的。继续把它们复制到`MainWindow`类中。

使用`pyuic5`创建的`Ui_`类的另一种方法是将其实例化为容器小部件的属性。我们将尝试在类别窗口中使用这个方法；在文件顶部添加这个类：

```py
class CategoryWindow(qtw.QWidget):

    submitted = qtc.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.ui = Ui_CategoryWindow()
        self.ui.setupUi(self)
        self.show()
```

在将`Ui_CategoryWindow`对象创建为`CategoryWindow`的属性之后，我们调用它的`setupUi()`方法来在`CategoryWindow`上构建 GUI。然而，我们所有对小部件的引用现在都在`self.ui`命名空间下。因此，例如，`category_entry`不是`self.category_entry`，而是`self.ui.category_entry`。虽然这种方法稍微冗长，但如果你正在构建一个特别复杂的类，它可能有助于避免名称冲突。

# 自动信号和插槽连接

再次查看由`pyuic5`生成的`Ui_`类，并注意`setupUi`中的最后一行代码：

```py
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
```

`connectSlotsByName()`是一种方法，它将通过将信号与以`on_object_name_signal()`格式命名的方法进行匹配来自动连接信号和插槽，其中`object_name`与`PyQt`对象的`objectName`属性匹配，`signal`是其内置信号之一的名称。

例如，在我们的`CategoryWindow`中，我们希望创建一个回调，当单击`submit_btn`时运行（如果您制作了自己的`.ui`文件，请确保您将提交按钮命名为`submit_btn`）。如果我们将回调命名为`on_submit_btn_clicked()`，那么这将自动发生。

代码如下：

```py
    @qtc.pyqtSlot()
    def on_submit_btn_clicked(self):
        if self.ui.category_entry.text():
            self.submitted.emit(self.ui.category_entry.text())
        self.close()
```

如果我们使名称匹配，我们就不必在任何地方显式调用`connect()`；回调将自动连接。

您也可以在手工编码的 GUI 中使用`connectSlotsByName()`；您只需要显式设置每个小部件的`objectName`属性，以便该方法有东西与名称匹配。仅仅变量名是行不通的。

# 在不进行转换的情况下使用.ui 文件

如果您不介意在运行时进行一些转换开销，实际上可以通过使用 PyQt 的`uic`库（`pyuic5`基于此库）在程序内部动态转换您的`.ui`文件，从而避免手动转换这一步。

让我们尝试使用我们的`MainWindow` GUI。首先将您对`Ui_MainWindow`的导入注释掉，并导入`uic`，如下所示：

```py
#from calendar_form import Ui_MainWindow
from PyQt5 import uic
```

然后，在您的`MainWindow`类定义之前，调用`uic.loadUiType()`，如下所示：

```py
MW_Ui, MW_Base = uic.loadUiType('calendar_form.ui')
```

`loadUiType()`接受一个`.ui`文件的路径，并返回一个包含生成的 UI 类和其基于的 Qt 基类（在本例中为`QWidget`）的元组。

然后，我们可以将这些用作我们的`MainWindow`类的父类，如下所示：

```py
class MainWindow(MW_Base, MW_Ui):
```

这种方法的缺点是额外的转换时间，但带来了更简单的构建和更少的文件维护。这是在早期开发阶段采取的一个很好的方法，当时您可能经常在 GUI 设计上进行迭代。

# 摘要

在本章中，您学习了 Qt 的对象间通信功能，即信号和插槽。您学会了如何使用它们来自动化表单行为，将功能连接到用户事件，并在应用程序的不同窗口之间进行通信。

在下一章中，我们将学习`QMainWindow`，这是一个简化常见应用程序组件构建的类。您将学会如何快速创建菜单、工具栏和对话框，以及如何保存设置。

# 问题

尝试这些问题来测试您对本章的了解：

1.  查看下表，并确定哪些连接实际上可以进行，哪些会导致错误。您可能需要在文档中查找这些信号和插槽的签名：

| # | 信号 | 插槽 |
| --- | --- | --- |
| 1 | `QPushButton.clicked` | `QLineEdit.clear` |
| 2 | `QComboBox.currentIndexChanged` | `QListWidget.scrollToItem` |
| 3 | `QLineEdit.returnPressed` | `QCalendarWidget.setGridVisible` |
| 4 | `QLineEdit.textChanged` | `QTextEdit.scrollToAnchor` |

1.  在信号对象上，`emit()`方法在信号被绑定（即连接到插槽）之前是不存在的。重写我们第一个`calendar_app.py`文件中的`CategoryWindow.onSubmit()`方法，以防`submitted`未绑定的可能性。

1.  您在 Qt 文档中找到一个对象，该对象的插槽需要`QString`作为参数。您能连接发送 Python 的`str`的自定义信号吗？

1.  您在 Qt 文档中找到一个对象，该对象的插槽需要`QVariant`作为参数。您可以将哪些内置的 Python 类型发送到这个插槽？

1.  您正在尝试创建一个对话框窗口，该窗口需要时间，并在用户完成编辑值时发出。您正在尝试使用自动插槽连接，但您的代码没有做任何事情。确定缺少了什么：

```py
    class TimeForm(qtw.QWidget):

        submitted = qtc.pyqtSignal(qtc.QTime)

        def __init__(self):
        super().__init__()
        self.setLayout(qtw.QHBoxLayout())
        self.time_inp = qtw.QTimeEdit(self)
        self.layout().addWidget(self.time_inp)

        def on_time_inp_editingFinished(self):
        self.submitted.emit(self.time_inp.time())
        self.destroy()
```

1.  你在 Qt Designer 中为一个计算器应用程序创建了一个`.ui`文件，现在你试图让它在代码中工作，但是它不起作用。在下面的源代码中你做错了什么？

```py
    from calculator_form import Ui_Calculator

    class Calculator(qtw.QWidget):
        def __init__(self):
            self.ui = Ui_Calculator(self)
            self.ui.setupGUI(self.ui)
            self.show()
```

1.  你正在尝试创建一个新的按钮类，当点击时会发出一个整数值；不幸的是，当你点击按钮时什么也不会发生。看看下面的代码，试着让它工作起来：

```py
    class IntegerValueButton(qtw.QPushButton):

        clicked = qtc.pyqtSignal(int)

        def __init__(self, value, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.value = value
            self.clicked.connect(
                lambda: self.clicked.emit(self.value))
```

# 进一步阅读

查看以下资源以获取更多信息：

+   PyQt 关于信号和槽支持的文档可以在这里找到：[`pyqt.sourceforge.net/Docs/PyQt5/signals_slots.html`](http://pyqt.sourceforge.net/Docs/PyQt5/signals_slots.html)

+   PyQt 关于使用 Qt Designer 的文档可以在这里找到：[`pyqt.sourceforge.net/Docs/PyQt5/designer.html`](http://pyqt.sourceforge.net/Docs/PyQt5/designer.html)
