# 第二十一章：问题的答案

# 第一章

1.  **Qt 是用 C++编写的，这种语言与 Python 非常不同。这两种语言之间有哪些主要区别？在我们使用 Python 中的 Qt 时，这些区别可能会如何体现？**

C++语言的差异以多种方式影响 PyQt，例如：

+   +   它的静态类型和类型安全的函数意味着在某些情况下，PyQt 对可以调用的函数和可以传递的变量相当严格。

+   C++中缺乏内置数据类型意味着 Qt 提供了丰富的数据类型选择，其中许多我们必须在 Python 中使用，因为类型安全。

+   在 C++中常见但在 Python 中很少见的`enum`类型在 Qt 中普遍存在。

1.  GUI 由小部件组成。在计算机上打开一些 GUI 应用程序，尝试识别尽可能多的小部件。

一些例子可能包括以下内容：

+   +   按钮

+   复选框

+   单选按钮

+   标签

+   文本编辑

+   滑块

+   图像区域

+   组合框

1.  **假设以下程序崩溃。找出原因，并修复它以显示一个窗口：**

```py
 from PyQt5.QtWidgets import *
 app = QWidget()
 app.show()
 QApplication().exec()
```

代码应该如下所示：

```py
   from PyQt5.QtWidgets import *

   app = QApplication([])
   window = QWidget()
   window.show()
   app.exe()
```

记住在任何`QWidget`对象之前必须存在一个`QApplication()`对象，并且它必须用列表作为参数创建。

1.  **`QWidget`类有一个名为`statusTip`的属性。以下哪些最有可能是该属性的访问方法的名称：**

1.  1.  `getStatusTip()`和`setStatusTip()`

1.  `statusTip()`和`setStatusTip()`

1.  `get_statusTip()`和`change_statusTip()`

答案**b**是正确的。在大多数情况下，`property`的访问器是`property()`和`setProperty()`。

1.  `QDate`是用于包装日历日期的类。你期望在三个主要的 Qt 模块中的哪一个找到它？

`QDate`在`QtCore`中。`QtCore`保存了与 GUI 不一定相关的数据类型类。

1.  `QFont`是定义屏幕字体的类。你期望在三个主要的 Qt 模块中的哪一个找到它？

`QFont`在`QtGui`中。字体与 GUI 相关，但不是小部件或布局，所以你期望它在`QtGui`中。

1.  **你能使用 Qt Designer 重新创建`hello_world.py`吗？确保设置`windowTitle`。**

基于`QWidget`创建一个新项目。然后选择主窗口小部件，并在属性窗格中设置`windowTitle`。

# 第二章

1.  **你如何创建一个全屏的`QWidget`，没有窗口框架，并使用沙漏光标？**

代码看起来像这样：

```py
   widget = QWidget(cursor=qtc.Qt.WaitCursor)
   widget.setWindowState(qtc.Qt.WindowFullScreen)
   widget.setWindowFlags(qtc.Qt.FramelessWindowHint)
```

1.  假设你被要求为计算机库存数据库设计一个数据输入表单。为以下字段选择最佳的小部件：

+   +   **计算机制造**：公司购买的八个品牌之一

+   **处理器速度**：CPU 速度（GHz）

+   **内存量**：RAM 的数量，以 MB 为单位

+   **主机名**：计算机的主机名

+   **视频制造**：视频硬件是 Nvidia、AMD 还是 Intel

+   **OEM 许可**：计算机是否使用 OEM 许可

这个表格列出了一些可能的答案：

| 字段 | 小部件 | 解释 |
| --- | --- | --- |
| 计算机制造 | `QComboBox` | 用于在许多值列表中进行选择，组合框是理想的选择 |
| 处理器速度 | `QDoubleSpinBox` | 十进制值的最佳选择 |
| 内存量 | `QSpinBox` | 整数值的最佳选择 |
| 主机名 | `QLineEdit` | 主机名只是一个单行文本字符串 |
| 视频制造 | `QComboBox`，`QRadioButton` | 组合框可以工作，但只有三个选择，单选按钮也是一个选项 |
| OEM 许可 | `QCheckBox` | `QCheckBox`是布尔值的一个很好的选择 |

1.  **数据输入表单包括一个需要`XX-999-9999X`格式的`库存编号`字段，其中`X`是从`A`到`Z`的大写字母，不包括`O`和`I`，`9`是从`0`到`9`的数字。你能创建一个验证器类来验证这个输入吗？**

查看示例代码中的`inventory_validator.py`。

1.  查看以下计算器表单：

![](img/1ecc9365-5e6d-40b1-9764-b07adf8f0aff.png)

**可能使用了哪些布局来创建它？**

很可能是一个带有嵌套`QGridLayout`布局的`QVBoxLayout`，用于按钮区域，或者是一个使用列跨度的单个`QGridLayout`布局的前两行。

1.  **参考前面的计算器表单，当表单被调整大小时，你如何使按钮网格占据任何额外的空间？**

在每个小部件上设置`sizePolicy`属性为`QtWidgets.QSizePolicy.Expanding`，垂直和水平都是。

1.  **计算器表单中最顶部的小部件是一个`QLCDNumber`小部件。你能找到关于这个小部件的 Qt 文档吗？它有哪些独特的属性？什么时候会用到它？**

`QLCDNumber`的文档在[`doc.qt.io/qt-5/qlcdnumber.html`](https://doc.qt.io/qt-5/qlcdnumber.html)。它的独特属性是`digitCount`、`intValue`、`mode`、`segmentStyle`、`smallDecimalPoint`和`value`。它适用于显示任何类型的数字，包括八进制、十六进制和二进制。

1.  **从你的模板代码开始，在代码中构建计算器表单。**

在示例代码中查看`calculator_form.py`。

1.  **在 Qt Designer 中构建计算器表单。**

在示例代码中查看`calculator_form.ui`。

# 第三章

1.  **查看下表，并确定哪些连接实际上可以被建立，哪些会导致错误。你可能需要在文档中查找这些信号和槽的签名：**

| # | 信号 | 槽 |
| --- | --- | --- |
| 1 | `QPushButton.clicked` | `QLineEdit.clear` |
| 2 | `QComboBox.currentIndexChanged` | `QListWidget.scrollToItem` |
| 3 | `QLineEdit.returnPressed` | `QCalendarWidget.setGridVisible` |
| 4 | `QLineEdit.textChanged` | `QTextEdit.scrollToAnchor` |

答案如下：

1.  1.  可以，因为`clicked`的布尔参数可以被`clear`忽略

1.  不行，因为`currentIndexChanged`发送的是`int`，但`scrollToItem`期望一个项目和一个滚动提示

1.  不行，因为`returnPressed`不发送任何参数，而`setGridVisible`期望一个参数

1.  可以，因为`textChanged`发送一个字符串，而`scrollToAnchor`接受它

1.  **在信号对象上，`emit()`方法直到信号被绑定（即连接到槽）之前都不存在。重新编写我们第一个`calendar_app.py`文件中的`CategoryWindow.onSubmit()`方法，以防`submitted`未被绑定的可能性。**

我们需要捕获`AttributeError`，像这样：

```py
        def onSubmit(self):
            if self.category_entry.text():
                try:
                    self.submitted.emit(self.category_entry.text())
                except AttributeError:
                    pass
            self.close()
```

1.  **你在 Qt 文档中找到一个对象，它的槽需要`QString`作为参数。你能连接你自定义的信号，发送一个 Python `str`对象吗？**

可以，因为 PyQt 会自动在`QString`和 Python `str`对象之间转换。

1.  **你在 Qt 文档中找到一个对象，它的槽需要`QVariant`作为参数。你可以发送哪些内置的 Python 类型到这个槽？**

任何一个都可以发送。`QVariant`是一个通用对象容器，可以容纳任何其他类型的对象。

1.  **你正在尝试创建一个对话框窗口，它需要时间，并在用户完成编辑数值时发出信号。你试图使用自动槽连接，但你的代码没有做任何事情。确定以下代码缺少什么：**

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

首先，你忘记调用`connectSlotsByName()`。另外，你没有设置`self.time_inp`的对象名称。你的代码应该像这样：

```py
    class TimeForm(qtw.QWidget):

        submitted = qtc.pyqtSignal(qtc.QTime)

        def __init__(self):
            super().__init__()
            self.setLayout(qtw.QHBoxLayout())
            self.time_inp = qtw.QTimeEdit(
                self, objectName='time_inp')
            self.layout().addWidget(self.time_inp)
            qtc.QMetaObject.connectSlotsByName(self)

        def on_time_inp_editingFinished(self):
            self.submitted.emit(self.time_inp.time())
            self.destroy()
```

1.  **你在 Qt Designer 中为一个计算器应用程序创建了一个`.ui`文件，并尝试在代码中让它工作，但是没有成功。你做错了什么？查看以下源代码：**

```py
    from calculator_form import Ui_Calculator

    class Calculator(qtw.QWidget):
        def __init__(self):
            self.ui = Ui_Calculator(self)
            self.ui.setupGUI(self.ui)
            self.show()
```

这里有四个问题：

+   +   首先，你忘记调用`super().__init__()`

+   其次，你将`self`传递给`Ui_Calculator`，它不需要任何参数

+   第三，你调用了`self.ui.setupGUI()`；应该是`self.ui.setupUi()`

+   最后，你将`self.ui`传递给`setupUi()`；你应该传递一个对包含小部件的引用，即`self`

1.  **你正在尝试创建一个新的按钮类，当点击按钮时会发出一个整数值；不幸的是，当你点击按钮时什么也不会发生。查看以下代码并尝试让它工作：**

```py
    class IntegerValueButton(qtw.QPushButton):

        clicked = qtc.pyqtSignal(int)

        def __init__(self, value, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.value = value
            self.clicked.connect(
                lambda: self.clicked.emit(self.value))
```

答案是将`__init__()`的最后一行更改为以下内容：

```py
 super().clicked.connect(
             lambda: self.clicked.emit(self.value))
```

因为我们用自己的信号覆盖了内置的`clicked`属性，`self.clicked`不再指向按钮被点击时发出的信号。我们必须调用`super().clicked`来获得对父类`clicked`信号的引用。

# 第四章

1.  **你想要使用`calendar_app.py`脚本中的`QMainWindow`，来自第三章，*使用信号和槽处理事件*。你会如何进行转换？**

最简单的方法是以下：

+   +   将`MainWindow`重命名为类似`CalendarForm`的东西

+   基于`QMainWindow`创建一个新的`MainWindow`类

+   在`MainWindow`内创建一个`CalendarForm`的实例，并将其设置为中央小部件

1.  **你正在开发一个应用程序，并已将子菜单名称添加到菜单栏，但尚未填充任何子菜单。你的同事说在他测试时他的桌面上没有出现任何菜单名称。你的代码看起来是正确的；这里可能出了什么问题？**

你的同事正在使用一个默认不显示空菜单文件夹的平台（如 macOS）。

1.  **你正在开发一个代码编辑器，并希望创建一个侧边栏面板与调试器进行交互。哪个`QMainWindow`特性对这个任务最合适？**

`QDockWidget`是最合适的，因为它允许你将任何类型的小部件构建到可停靠窗口中。工具栏不是一个好选择，因为它主要设计用于按钮。

1.  **以下代码无法正常工作；无论点击什么都会继续。为什么它不起作用，你如何修复它？**

```py
    answer = qtw.QMessageBox.question(
        None, 'Continue?', 'Run this program?')
    if not answer:
        sys.exit()
```

`QMessageBox.question()`不返回布尔值；它返回与点击的按钮类型匹配的常量。匹配`No`按钮的常量的实际整数值是`65536`，在 Python 中评估为`True`。代码应该如下所示：

```py
    answer = qtw.QMessageBox.question(
        None, 'Continue?', 'Run this program?')
    if answer == qtw.QMessageBox.No:
        sys.exit()
```

1.  **你正在通过子类化`QDialog`来构建一个自定义对话框。你需要将对话框中输入的信息传递回主窗口对象。以下哪种方法不起作用？**

+   1.  **传入一个可变对象，并使用对话框的`accept()`方法来改变它的值。**

1.  **覆盖对象的`accept()`方法，并使其返回输入值的`dict`。**

+   1.  **覆盖对话框的`accepted`信号，使其传递输入值的`dict`。将此信号连接到主窗口类中的回调。**

答案**a**和**c**都可以。答案**b**不行，因为`accept`的返回值在调用`exec()`时对话框没有返回。`exec()`只返回一个布尔值，指示对话框是被接受还是被拒绝。

1.  **你正在 Linux 上开发一个名为 SuperPhoto 的照片编辑器。你已经编写了代码并保存了用户设置，但是在`~/.config/`中找不到`SuperPhoto.conf`。查看代码并确定出了什么问题：**

```py
    settings = qtc.QSettings()
    settings.setValue('config_file', 'SuperPhoto.conf')
    settings.setValue('default_color', QColor('black'))
    settings.sync()
```

`QSettings`使用的配置文件（或在 Windows 上的注册表键）由传递给构造函数的公司名称和应用程序名称确定。代码应该如下所示：

```py
 settings = qtc.QSettings('My Company', 'SuperPhoto')
 settings.setValue('default_color', QColor('black'))
```

另外，注意`sync()`不需要显式调用。它会被 Qt 事件循环自动调用。

1.  **你正在从设置对话框保存偏好设置，但出于某种原因，保存的设置返回的结果非常奇怪。这里有什么问题？看看以下代码：**

```py
    settings = qtc.QSettings('My Company', 'SuperPhoto')
    settings.setValue('Default Name', dialog.default_name_edit.text)
    settings.setValue('Use GPS', dialog.gps_checkbox.isChecked)
    settings.setValue('Default Color', dialog.color_picker.color)
```

问题在于你实际上没有调用小部件的访问函数。因此，`settings`存储了访问函数的引用。在下一次程序启动时，这些引用是无意义的，因为新的对象被创建在新的内存位置。请注意，如果你保存函数引用，`settings`不会抱怨。

# 第五章

1.  **假设我们有一个设计良好的模型-视图应用程序，以下代码是模型还是视图的一部分？**

```py
  def save_as(self):
    filename, _ = qtw.QFileDialog(self)
    self.data.save_file(filename)
```

这是视图代码，因为它创建了一个 GUI 元素（文件对话框），并似乎回调到可能是一个模型的东西（`self.data`）。

1.  **您能否至少列举两件模型绝对不应该做的事情，以及视图绝对不应该做的两件事情？**

模型绝对不应该做的事情的例子包括创建或直接更改 GUI 元素，为演示格式化数据，或关闭应用程序。视图绝对不应该做的事情的例子包括将数据保存到磁盘，对存储的数据执行转换（如排序或算术），或从模型以外的任何地方读取数据。

1.  `QAbstractTableModel`和`QAbstractTreeModel`都在名称中带有`abstract`。在这种情况下，`abstract`是什么意思？在 C++中，它的含义与 Python 中的含义不同吗？

在任何编程语言中，抽象类是指不打算实例化为对象的类；它们只应该被子类化，并覆盖所需的方法。在 Python 中，这是暗示的，但不是强制的；在 C++中，标记为`abstract`的类将无法实例化。

1.  **以下哪种模型类型——列表、表格或树——最适合以下数据集？**

+   1.  **用户的最近文件**

1.  **Windows 注册表**

1.  **Linux `syslog`记录**

1.  **博客文章**

1.  **个人称谓（例如，先生，夫人或博士）**

1.  **分布式版本控制历史**

虽然有争议，但最有可能的答案如下：

1.  1.  列表

1.  树

1.  表

1.  表

1.  列表

1.  树

1.  **为什么以下代码失败了？**

```py
  class DataModel(QAbstractTreeModel):
    def rowCount(self, node):
      if node > 2:
        return 1
      else:
        return len(self._data[node])
```

`rowCount()`的参数是指向父节点的`QModelIndex`对象。它不能与整数进行比较（`if node > 2`）。

1.  **当插入列时，您的表模型工作不正常。您的`insertColumns()`方法有什么问题？**

```py
    def insertColumns(self, col, count, parent):
      for row in self._data:
        for i in range(count):
          row.insert(col, '')
```

在修改数据之前，您忽略了调用`self.beginInsertColumns()`，并在完成后调用`self.endInsertColumns()`。

1.  **当鼠标悬停时，您希望您的视图显示项目数据作为工具提示。您将如何实现这一点？**

您需要在模型的`data()`方法中处理`QtCore.Qt.TooltipRole`。代码示例如下：

```py
        def data(self, index, role):
            if role in (
                qtc.Qt.DisplayRole,
                qtc.Qt.EditRole,
                qtc.Qt.ToolTipRole
            ):
                return self._data[index.row()][index.column()]
```

# 第六章

1.  **您正在准备分发您的文本编辑器应用程序，并希望确保用户无论使用什么平台，都会默认获得等宽字体。您可以使用哪两种方法来实现这一点？**

第一种方法是将默认字体的`styleHint`设置为`QtGui.QFont.Monospace`。第二种方法是找到一个适当许可的等宽字体，将其捆绑到 Qt 资源文件中，并将字体设置为您捆绑的字体。

1.  **尽可能地，尝试使用`QFont`模仿以下文本：**

![](img/7bcc4ce2-2313-4c4a-81c0-6897c8e32149.png)

代码如下：

```py
   font = qtg.QFont('Times', 32, qtg.QFont.Bold)
   font.setUnderline(True)
   font.setOverline(True)
   font.setCapitalization(qtg.QFont.SmallCaps)
```

1.  **您能解释`QImage`，`QPixmap`和`QIcon`之间的区别吗？**

`QPixmap`和`QImage`都代表单个图像，但`QPixmap`经过优化用于显示，而`QImage`经过优化用于内存中的图像处理。`QIcon`不是单个图像，而是一组可以绑定到小部件或操作状态的图像。

1.  您已经为您的应用程序定义了以下`.qrc`文件，运行了`pyrcc5`，并在脚本中导入了资源库。如何将这个图像加载到`QPixmap`中？

```py
   <RCC>
      <qresource prefix="foodItems">
        <file alias="pancakes.png">pc_img.45234.png</file>
      </qresource>
   </RCC>
```

代码应该如下所示：

```py
   pancakes_pxm = qtg.QPixmap(":/foodItems/pancakes.png")
```

1.  **使用`QPalette`，如何使用`tile.png`图像铺设`QWidget`对象的背景？**

代码应该如下所示：

```py
   widget = qtw.QWidget()
   palette = widget.palette()
   tile_brush = qtg.QBrush(
       qtg.QColor('black'),
       qtg.QPixmap('tile.png')
   )
   palette.setBrush(qtg.QPalette.Window, tile_brush)
   widget.setPalette(palette)
```

1.  **您试图使用 QSS 使删除按钮变成粉色，但没有成功。您的代码有什么问题？**

```py
   deleteButton = qtw.QPushButton('Delete')
   form.layout().addWidget(deleteButton)
   form.setStyleSheet(
      form.styleSheet() + 'deleteButton{ background-color: #8F8; }'
   )
```

您的代码有两个问题。首先，您的`deleteButton`没有分配`objectName`。QSS 对您的 Python 变量名称一无所知；它只知道 Qt 对象名称。其次，您的样式表没有使用`#`符号前缀对象名称。更正后的代码应该如下所示：

```py
   deleteButton = qtw.QPushButton('Delete')
   deleteButton.setObjectName('deleteButton')
   form.layout().addWidget(deleteButton)
   form.setStyleSheet(
      form.styleSheet() + 
      '#deleteButton{ background-color: #8F8; }'
   )
```

1.  **哪种样式表字符串将把您的`QLineEdit`小部件的背景颜色变成黑色？**

```py
   stylesheet1 = "QWidget {background-color: black;}"
   stylesheet2 = ".QWidget {background-color: black;}"
```

`stylesheet1`将把任何`QWidget`子类的背景变成黑色，包括`QLineEdit`。`stylesheet2`只会把实际`QWidget`对象的背景变成黑色；子类将保持不受影响。

1.  **使用下拉框构建一个简单的应用程序，允许您将 Qt 样式更改为系统上安装的任何样式。包括一些其他小部件，以便您可以看到它们在不同样式下的外观。**

在本章的示例代码中查看`question_8_answer.py`。

1.  **您对学习如何为 PyQt 应用程序设置样式感到非常高兴，并希望创建一个`QProxyStyle`类，该类将强制 GUI 中的所有像素图像为`smile.gif`。您会如何做？提示：您需要研究一些`QStyle`的绘图方法，而不是本章讨论的方法。**

该类如下所示：

```py
   class SmileyStyley(qtw.QProxyStyle):

       def drawItemPixmap(
           self, painter, rectangle, alignment, pixmap):
           smile = qtg.QPixmap('smile.gif')
           super().drawItemPixmap(
               painter, rectangle, alignment, smile)
```

1.  **以下动画不起作用；找出为什么不起作用：**

```py
    class MyWidget(qtw.QWidget):
        def __init__(self):
            super().__init__()
            animation = qtc.QPropertyAnimation(
                self, b'windowOpacity')
            animation.setStartValue(0)
            animation.setEndValue(1)
            animation.setDuration(10000)
            animation.start()
```

简短的答案是`animation`应该是`self.animation`。动画没有父对象，当它们被添加到布局时，它们不会像小部件一样被**重新父化**。因此，当构造函数退出时，`animation`就会超出范围并被销毁。故事的寓意是，保存您的动画作为实例变量。

# 第七章

1.  **使用`QSoundEffect`，您为呼叫中心编写了一个实用程序，允许他们回顾录制的电话呼叫。他们正在转移到一个新的电话系统，该系统将电话呼叫存储为 MP3 文件。您需要对您的实用程序进行任何更改吗？**

是的。您需要使用`QMediaPlayer`而不是`QSoundEffect`，或者编写一个解码 MP3 到 WAV 的层，因为`QSoundEffect`无法播放压缩音频。

1.  `cool_songs`是一个 Python 列表，其中包含您最喜欢的歌曲的路径字符串。要以随机顺序播放这些歌曲，您需要做什么？

您需要将路径转换为`QUrl`对象，将它们添加到`QMediaPlaylist`，将`playbackMode`设置为`Random`，然后将其传递给`QMediaPlayer`。代码如下：

```py
   playlist = qtmm.QMediaPlaylist()
   for song in cool_songs:
       url = qtc.QUrl.fromLocalFile(song)
       content = qtmm.QMediaContent(url)
       playlist.addMedia(content)
   playlist.setPlaybackMode(qtmm.QMediaPlaylist.Random)
   player = qtmm.QMediaPlayer()
   player.setPlaylist(playlist)
   player.play()
```

1.  **您已在系统上安装了`audio/mpeg`编解码器，但以下代码不起作用。找出其中的问题：**

```py
   recorder = qtmm.QAudioRecorder()
   recorder.setCodec('audio/mpeg')
   recorder.record()
```

`QAudioRecorder`没有`setCodec`方法。录制中使用的编解码器设置在`QAudioEncoderSettings`对象上设置。代码应该如下所示：

```py
   recorder = qtmm.QAudioRecorder()
   settings = qtmm.QAudioEncoderSettings()
   settings.setCodec('audio/mpeg')
   recorder.setEncodingSettings(settings)
   recorder.record()
```

1.  在几个不同的 Windows、macOS 和 Linux 系统上运行`audio_test.py`和`video_test.py`。输出有什么不同？有哪些项目在所有系统上都受支持？

答案将取决于您选择的系统。

1.  `QCamera`类的属性包括几个控制对象，允许您管理相机的不同方面。其中之一是`QCameraFocus`。在 Qt 文档中查看`QCameraFocus`，并编写一个简单的脚本，显示取景器并让您调整数字变焦。

在包含的代码示例中查看`question_5_example_code.py`。

1.  **您已经注意到录制到您的船长日志视频日志中的音频相当响亮。您想添加一个控件来调整它；您会如何做？**

`QMediaRecorder`有一个`volume()`插槽，就像`QAudioRecorder`一样。您需要创建一个`QSlider`（或任何其他控件小部件），并将其`valueChanged`或`sliderMoved`信号连接到录制器的`volume()`插槽。

1.  **在`captains_log.py`中实现一个停靠窗口小部件，允许您控制尽可能多的音频和视频录制方面。您可以包括焦点、缩放、曝光、白平衡、帧速率、分辨率、音频音量、音频质量等内容。**

这里就靠你自己了！

# 第八章

1.  **您正在设计一个应用程序，该应用程序将向本地网络发出状态消息，您将使用管理员工具进行监控。哪种类型的套接字对象是一个不错的选择？**

在这里最好使用`QUdpSocket`，因为它允许广播数据包，并且状态数据包不需要 TCP 的开销。

1.  您的 GUI 类有一个名为`self.socket`的`QTcpSocket`对象。您已经将其`readyRead`信号连接到以下方法，但它没有起作用。发生了什么，您该如何修复它？

```py
       def on_ready_read(self):
           while self.socket.hasPendingDatagrams():
               self.process_data(self.socket.readDatagram())
```

`QTcpSocket`没有`hasPendingDatagrams()`或`readDatagram()`方法。TCP 套接字使用数据流而不是数据包。这个方法需要重写以使用`QDataStream`对象提取数据。

1.  使用`QTcpServer`实现一个简单的服务，监听端口`8080`并打印接收到的任何请求。让它用您选择的字节字符串回复客户端。

在示例代码中查看`question_3_tcp_server.py`。通过运行脚本并将 Web 浏览器指向[`localhost:8080`](http://localhost:8080)来进行测试。

1.  您正在为应用程序创建一个下载函数，以便检索一个大型数据文件以导入到您的应用程序中。代码不起作用。阅读代码并决定您做错了什么：

```py
       def download(self, url):
        self.manager = qtn.QNetworkAccessManager(
            finished=self.on_finished)
        self.request = qtn.QNetworkRequest(qtc.QUrl(url))
        reply = self.manager.get(self.request)
        with open('datafile.dat', 'wb') as fh:
            fh.write(reply.readAll())
```

您试图同步使用`QNetworkAccessManager.get()`，但它是设计用于异步使用的。您需要连接一个回调到网络访问管理器的`finished`信号，而不是从`get()`中检索回复对象，它携带完成的回复。

1.  修改您的`poster.py`脚本，以便将键值数据发送为 JSON，而不是 HTTP 表单数据。

在示例代码中查看`question_5_json_poster.py`文件。

# 第九章

1.  编写一个 SQL `CREATE`语句，用于构建一个表来保存电视节目表。确保它具有日期、时间、频道和节目名称的字段。还要确保它具有主键和约束，以防止无意义的数据（例如在同一频道上同时播放两个节目，或者一个节目没有时间或日期）。

一个示例可能如下所示：

```py
   CREATE TABLE tv_schedule AS (
       id INTEGER PRIMARY KEY,
       channel TEXT NOT NULL,
       date DATE NOT NULL,
       time TIME NOT NULL,
       program TEXT NOT NULL,
       UNIQUE(channel, date, time)
   )
```

1.  以下 SQL 查询返回语法错误；您能修复它吗？

```py
DELETE * FROM my_table IF category_id == 12;
```

这里有几个问题：

+   +   `DELETE`不接受字段列表，因此必须删除`*`。

+   `IF`是错误的关键字。它应该使用`WHERE`。

+   `==`不是 SQL 运算符。与 Python 不同，SQL 使用单个`=`进行赋值和比较操作。

生成的 SQL 应该如下所示：

```py
   DELETE FROM my_table WHERE category_id = 12;
```

1.  以下 SQL 查询不正确；您能修复它吗？

```py
INSERT INTO flavors(name) VALUES ('hazelnut', 'vanilla', 'caramel', 'onion');
```

`VALUES`子句中的每组括号表示一行。由于我们只插入一列，每行应该只有一个值。因此，我们的语句应该如下所示：

```py
   INSERT INTO flavors(name) VALUES ('hazelnut'), ('vanilla'), ('caramel'), ('onion');
```

1.  `QSqlDatabase`的文档可以在[`doc.qt.io/qt-5/qsqldatabase.html`](https://doc.qt.io/qt-5/qsqldatabase.html)找到。详细了解如何使用多个数据库连接，例如对同一数据库进行只读和读写连接。您将如何创建两个连接并对每个连接进行特定的查询？

关键是多次使用唯一连接名称调用`addDatabase()`；一个示例如下：

```py
   db1 = qts.QSqlDatabase.addDatabase('QSQLITE', 'XYZ read-only')
   db1.setUserName('readonlyuser')
   # etc...
   db1.open()
   db2 = qts.QSqlDatabase.addDatabase('QSQLITE', 'XYZ read-write')
   db2.setUserName('readwriteuser')
   # etc...
   db2.open()

   # Keep the database reference for querying:
   query = qts.QSqlQuery('SELECT * FROM my_table', db1)

   # Or retrieve it using its name:
   db = qts.QSqlDatabase.database('XYZ read-write')
   db.exec('INSERT INTO my_table VALUES (1, 2, 3)')
```

1.  使用`QSqlQuery`，编写代码将`dict`对象中的数据安全地插入`coffees`表中：

```py
data = {'brand': 'generic', 'name': 'cheap coffee', 'roast': 
    'light'}
# Your code here:
```

为了安全起见，我们将使用`QSqlQuery`的`prepare()`方法：

```py
   data = {'brand': 'generic', 'name': 'cheap coffee', 'roast': 
       'Light'}
   query = QSqlQuery()
   query.prepare(
       'INSERT INTO coffees(coffee_brand, coffee_name, roast_id) '
       'VALUES (:brand, :name,
       '(SELECT id FROM roasts WHERE description == :roast))'
   )
   query.bindValue(':brand', data['brand'])
   query.bindValue(':name', data['name'])
   query.bindValue(':roast', data['roast'])
   query.exec()
```

1.  您已经创建了一个`QSqlTableModel`对象，并将其附加到`QTableView`。您知道表中有数据，但在视图中没有显示。查看代码并决定问题出在哪里：

```py
flavor_model = qts.QSqlTableModel()
flavor_model.setTable('flavors')
flavor_table = qtw.QTableView()
flavor_table.setModel(flavor_model)
mainform.layout().addWidget(flavor_table)
```

您没有在模型上调用`select()`。在这样做之前，它将是空的。

1.  以下是附加到`QLineEdit`的`textChanged`信号的回调。解释为什么这不是一个好主意：

```py
def do_search(self, text):
    self.sql_table_model.setFilter(f'description={text}')
    self.sql_table_model.select()
```

问题在于您正在接受任意用户输入并将其传递给表模型的`filter()`字符串。这个字符串被直接附加到表模型的内部 SQL 查询中，从而使您的数据库容易受到 SQL 注入。为了使其安全，您需要采取措施来清理`text`或切换 SQL 表模型以使用`prepare()`来创建一个准备好的语句。

1.  您决定在您的咖啡列表的烘焙组合框中使用颜色而不是名称。为了实现这一点，您需要做出哪些改变？

您需要更改`roast_id`上设置的`QSqlRelation`所使用的显示字段为`color`。然后，您需要为`coffee_list`创建一个自定义委托，用于创建颜色图标（参见第六章，*Qt 应用程序的样式*）并在组合框中使用它们而不是文本标签。

# 第十章

1.  创建代码以每十秒调用`self.every_ten_seconds()`方法。

假设我们在一个类的`__init__()`方法中，它看起来像这样：

```py
           self.timer = qtc.QTimer()
           self.timer.setInterval(10000)
           self.timer.timeout.connect(self.every_ten_seconds)
```

1.  以下代码错误地使用了`QTimer`。你能修复它吗？

```py
   timer = qtc.QTimer()
   timer.setSingleShot(True)
   timer.setInterval(1000)
   timer.start()
   while timer.remainingTime():
       sleep(.01)
   run_delayed_command()
```

`QTimer`与`while`循环同步使用。这会创建阻塞代码。可以异步完成相同的操作，如下所示：

```py
   qtc.QTimer.singleShot(1000, run_delayed_command)
```

1.  您创建了以下计算单词数的工作类，并希望将其移动到另一个线程以防止大型文档减慢 GUI。但是，它没有工作；您需要对这个类做出哪些改变？

```py
   class Worker(qtc.QObject):

    counted = qtc.pyqtSignal(int)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

    def count_words(self):
        content = self.parent.textedit.toPlainText()
        self.counted.emit(len(content.split()))
```

该类依赖于通过共同的父级访问小部件，因为`Worker`类必须由包含小部件的 GUI 类作为父级。您需要更改此类，使以下内容适用：

+   +   它没有父小部件。

+   它以其他方式访问内容，比如通过一个槽。

1.  以下代码是阻塞的，而不是在单独的线程中运行。为什么会这样？

```py
   class Worker(qtc.QThread):

       def set_data(data):
           self.data = data

       def run(self):n
           start_complex_calculations(self.data)

    class MainWindow(qtw.QMainWindow):

        def __init__(self):
            super().__init__()
            form = qtw.QWidget()
            self.setCentralWidget(form)
            form.setLayout(qtw.QFormLayout())

            worker = Worker()
            line_edit = qtw.QLineEdit(textChanged=worker.set_data)
            button = qtw.QPushButton('Run', clicked=worker.run)
            form.layout().addRow('Data:', line_edit)
            form.layout().addRow(button)
            self.show()
```

按钮回调指向`Worker.run()`。它应该指向`QThread`对象的`start()`方法。

1.  这个工作类会正确运行吗？如果不会，为什么？

```py
   class Worker(qtc.QRunnable):

       finished = qtc.pyqtSignal()

       def run(self):
           calculate_navigation_vectors(30)
           self.finished.emit()
```

不，`QRunnable`对象不能发出信号，因为它们不是从`QObject`继承的，也没有事件循环。在这种情况下，最好使用`QThread`。

1.  以下代码是一个`QRunnable`类的`run()`方法，用于处理来自科学设备的大型数据文件输出。文件由数百万行空格分隔的数字组成。这段代码可能会被 Python GIL 减慢吗？您能使 GIL 干扰的可能性更小吗？

```py
       def run(self):
           with open(self.file, 'r') as fh:
               for row in fh:
                   numbers = [float(x) for x in row.split()]
                   if numbers:
                       mean = sum(numbers) / len(numbers)
                       numbers.append(mean)
                   self.queue.put(numbers)
```

读取文件是一个 I/O 绑定的操作，不需要获取 GIL。但是，进行数学计算和类型转换是一个 CPU 绑定的任务，需要获取 GIL。这可以通过在非 Python 数学库（如 NumPy）中进行计算来减轻。

1.  以下是你正在编写的多线程 TCP 服务器应用程序中`QRunnable`中的`run()`方法。所有线程共享通过`self.datastream`访问的服务器套接字实例。但是，这段代码不是线程安全的。你需要做什么来修复它？

```py
       def run(self):
           message = get_http_response_string()
           message_len = len(message)
           self.datastream.writeUInt32(message_len)
           self.datastream.writeQString(message)
```

由于您不希望两个线程同时写入数据流，您将希望使用`QMutex`来确保只有一个线程可以访问。在定义了一个名为`qmutex`的共享互斥对象之后，代码将如下所示：

```py
       def run(self):
           message = get_http_response_string()
           message_len = len(message)
           with qtc.QMutexLocker(self.qmutex):
               self.datastream.writeUInt32(message_len)
               self.datastream.writeQString(message)
```

# 第十一章

1.  以下 HTML 显示不像您想要的那样。找出尽可能多的错误：

```py
<table>
<thead background=#EFE><th>Job</th><th>Status</th></thead>
<tr><td>Backup</td><font text-color='green'>Success!</font></td></tr>
<tr><td>Cleanup<td><font text-style='bold'>Fail!</font></td></tr>
</table>
```

这里有几个错误：

+   +   `<thead>`部分缺少围绕单元格的`<tr>`标签。

+   在下一行中，第二个单元格缺少开放的`<td>`标签。

+   另外，没有`text-color`属性。它只是`color`。

+   在下一行中，第一个单元格缺少闭合的`</td>`标签。

+   还有没有`text-style`属性。文本应该只是用`<b>`标签包装起来。

1.  以下 Qt HTML 片段有什么问题？

```py
<p>There is nothing <i>wrong</i> with your television <b>set</p></b>
<table><row><data>french fries</data>
<data>$1.99</data></row></table>
<font family='Tahoma' color='#235499'>Can you feel the <strikethrough>love</strikethrough>code tonight?</font>
<label>Username</label><input type='text' name='username'></input>
<img source='://mypix.png'>My picture</img>
```

问题如下：

1.  1.  最后两个闭合标签被切换了。嵌套标签必须在外部标签之前关闭。

1.  没有`<row>`或`<data>`这样的标签。正确的标签应该分别是`<tr>`和`<td>`。

1.  有两个问题——`<font>`没有`family`属性，应该是`face`；另外，没有`<strikethrough>`标签，应该是`<s>`。

1.  Qt 不支持`<label>`或`<input>`标签。此外，`<input>`不使用闭合标签。

1.  `<img>`没有`source`属性；它应该是`src`。它也没有使用闭合标签，也不能包含文本内容。

1.  **这段代码应该实现一个目录。为什么它不能正常工作？**

```py
   <ul>
     <li><a href='Section1'>Section 1</a></li>
     <li><a href='Section2'>Section 2</a></li>
   </ul>
   <div id=Section1>
     <p>This is section 1</p>
   </div>
   <div id=Section2>
     <p>This is section 2</p>
   </div>
```

这不是文档锚点的工作方式。正确的代码如下：

```py
   <ul>
     <li><a href='#Section1'>Section 1</a></li>
     <li><a href='#Section2'>Section 2</a></li>
   </ul>
   <a name='Section1'></a>
   <div id=Section1>
     <p>This is section 1</p>
   </div>
   <a name='Section2'></a>
   <div id=Section2>
     <p>This is section 2</p>
   </div>
```

请注意`href`前面的井号(`#`)，表示这是一个内部锚点，以及上面的`<a>`标签，其中包含一个包含部分名称的`name`属性（不包括井号！）。

1.  **使用`QTextCursor`，您需要在文档的右侧添加一个侧边栏。解释一下您将如何做到这一点。**

这样做的步骤如下：

+   1.  创建一个`QTextFrameFormat`对象

1.  将框架格式的`position`属性配置为右浮动

1.  将文本光标定位在根框中

1.  在光标上调用`insertFrame()`，并将框架对象作为第一个参数

1.  使用光标插入方法插入侧边栏内容

1.  **您正在尝试使用`QTextCursor`创建一个文档。它应该有一个顶部和底部框架；在顶部框架中，应该有一个标题，在底部框架中，应该有一个无序列表。请更正此代码，使其实现这一点：**

```py
   document = qtg.QTextDocument()
   cursor = qtg.QTextCursor(document)
   top_frame = cursor.insertFrame(qtg.QTextFrameFormat())
   bottom_frame = cursor.insertFrame(qtg.QTextFrameFormat())

   cursor.insertText('This is the title')
   cursor.movePosition(qtg.QTextCursor.NextBlock)
   cursor.insertList(qtg.QTextListFormat())
   for item in ('thing 1', 'thing 2', 'thing 3'):
       cursor.insertText(item)
```

这段代码的主要问题在于它未能正确移动光标，因此内容没有被创建在正确的位置。以下是更正后的代码：

```py
   document = qtg.QTextDocument()
   cursor = qtg.QTextCursor(document)
   top_frame = cursor.insertFrame(qtg.QTextFrameFormat())
   cursor.setPosition(document.rootFrame().lastPosition())
   bottom_frame = cursor.insertFrame(qtg.QTextFrameFormat())

   cursor.setPosition(top_frame.lastPosition())
   cursor.insertText('This is the title')
   # This won't get us to the next frame:
   #cursor.movePosition(qtg.QTextCursor.NextBlock)
   cursor.setPosition(bottom_frame.lastPosition())
   cursor.insertList(qtg.QTextListFormat())
   for i, item in enumerate(('thing 1', 'thing 2', 'thing 3')):
       # don't forget to add a block for each item after the first:
       if i > 0:
           cursor.insertBlock()
       cursor.insertText(item)
```

1.  **您正在创建自己的`QPrinter`子类以在页面大小更改时添加信号。以下代码会起作用吗？**

```py
   class MyPrinter(qtps.QPrinter):

       page_size_changed = qtc.pyqtSignal(qtg.QPageSize)

       def setPageSize(self, size):
           super().setPageSize(size)
           self.page_size_changed.emit(size)
```

不幸的是，不会。因为`QPrinter`不是从`QObject`派生的，所以它不能有信号。您将会收到这样的错误：

```py
   TypeError: MyPrinter cannot be converted to PyQt5.QtCore.QObject in this context
```

1.  **`QtPrintSupport`包含一个名为`QPrinterInfo`的类。使用这个类，在您的系统上打印出所有打印机的名称、制造商和型号以及默认页面大小的列表。**

代码如下：

```py
   for printer in qtps.QPrinterInfo.availablePrinters():
       print(
           printer.printerName(),
           printer.makeAndModel(),
           printer.defaultPageSize())
```

# 第十二章

1.  **在这个方法中添加代码，以在图片底部用蓝色写下您的名字：**

```py
       def create_headshot(self, image_file, name):
           image = qtg.QImage()
           image.load(image_file)
           # your code here

           # end of your code
           return image
```

您的代码将需要创建`QPainter`和`QPen`，然后写入图像：

```py
       def create_headshot(self, image_file, name):
           image = qtg.QImage()
           image.load(image_file)

           # your code here
           painter = qtg.QPainter(image)
           pen = qtg.QPen(qtg.QColor('blue'))
           painter.setPen(pen)
           painter.drawText(image.rect(), qtc.Qt.AlignBottom, name)

           # end of your code
           return image
```

1.  **给定一个名为`painter`的`QPainter`对象，写一行代码在绘图设备的左上角绘制一个 80×80 像素的八边形。参考[`doc.qt.io/qt-5/qpainter.html#drawPolygon`](https://doc.qt.io/qt-5/qpainter.html#drawPolygon)中的文档。**

有几种方法可以创建和绘制多边形，但最简单的方法是将一系列`QPoint`对象传递给`drawPolygon()`：

```py
   painter.drawPolygon(
       qtc.QPoint(0, 20), qtc.QPoint(20, 0),
       qtc.QPoint(60, 0), qtc.QPoint(80, 20),
       qtc.QPoint(80, 60), qtc.QPoint(60, 80),
       qtc.QPoint(20, 80), qtc.QPoint(0, 60)
   )
```

当然，您也可以使用`QPainterPath`对象。

1.  **您正在创建一个自定义小部件，但不知道为什么文本显示为黑色。以下是您的`paintEvent()`方法；看看您能否找出问题所在：**

```py
   def paintEvent(self, event):
       black_brush = qtg.QBrush(qtg.QColor('black'))
       white_brush = qtg.QBrush(qtg.QColor('white'))
       painter = qtg.QPainter()
       painter.setBrush(black_brush)
       painter.drawRect(0, 0, self.width(), self.height())
       painter.setBrush(white_brush)
       painter.drawText(0, 0, 'Test Text')
```

问题在于您设置了`brush`，但文本是用`pen`绘制的。默认的笔是黑色。要解决这个问题，创建一个设置为白色的`pen`，并在绘制文本之前将其传递给`painter.setPen()`。

1.  **油炸模因是一种使用极端压缩、饱和度和其他处理方式的模因风格，使模因图像看起来故意低质量。向您的模因生成器添加一个功能，可选择使模因油炸。您可以尝试的一些方法包括减少颜色位深度和调整图像中颜色的色调和饱和度。**

在这里要有创意，但是可以参考附带源代码中的`question_4_example_code.py`文件。

1.  **您想要对一个圆进行水平移动的动画。在以下代码中，您需要改变什么才能使圆形动起来？**

```py
   scene = QGraphicsScene()
   scene.setSceneRect(0, 0, 800, 600)
   circle = scene.addEllipse(0, 0, 10, 10)
   animation = QPropertyAnimation(circle, b'x')
   animation.setStartValue(0)
   animation.setEndValue(600)
   animation.setDuration(5000)
   animation.start()
```

您的`circle`对象不能像现在这样进行动画处理，因为它是一个`QGraphicsItem`。要使用`QPropertyAnimation`对对象的属性进行动画处理，它必须是`QObject`的后代。您需要将您的圆构建为`QGraphicsObject`的子类；然后，您可以对其进行动画处理。

1.  **以下代码有什么问题，它试图使用渐变刷设置`QPainter`？**

```py
   gradient = qtg.QLinearGradient(
       qtc.QPointF(0, 100), qtc.QPointF(0, 0))
   gradient.setColorAt(20, qtg.QColor('red'))
   gradient.setColorAt(40, qtg.QColor('orange'))
   gradient.setColorAt(60, qtg.QColor('green'))
   painter = QPainter()
   painter.setGradient(gradient)
```

这里有两个问题：

1.  1.  `setColorAt`的第一个参数不是像素位置，而是一个表示为浮点数的百分比，介于`0`和`1`之间。

1.  没有`QPainter.setGradient()`方法。渐变必须传递到`QPainter`构造函数中。

1.  看看你是否可以实现以下游戏改进：

+   +   脉动子弹

+   击中坦克时爆炸

+   声音（参见第七章，*使用 QtMultimedia 处理音频-视觉*，在这里寻求帮助）

+   背景动画

+   多个子弹

你自己来吧。玩得开心！

# 第十三章

1.  OpenGL 渲染管线的哪些步骤是可用户定义的？为了渲染任何东西，必须定义哪些步骤？你可能需要参考[`www.khronos.org/opengl/wiki/Rendering_Pipeline_Overview`](https://www.khronos.org/opengl/wiki/Rendering_Pipeline_Overview)上的文档。

顶点处理和片段着色器步骤是可用户定义的。至少，你必须创建一个顶点着色器和一个片段着色器。可选步骤包括几何着色器和镶嵌步骤，这些步骤是顶点处理的一部分。

1.  你正在为一个 OpenGL 2.1 程序编写着色器。以下看起来正确吗？

```py
   #version 2.1

   attribute highp vec4 vertex;

   void main (void)
   {
   gl_Position = vertex;
   }
```

你的版本字符串是错误的。它应该是`#version 120`，因为它指定了 GLSL 的版本，而不是 OpenGL 的版本。版本也被指定为一个没有句号的三位数。

1.  以下是顶点着色器还是片段着色器？你如何判断？

```py
   attribute highp vec4 value1;
   varying highp vec3 x[4];
   void main(void)
   {
     x[0] = vec3(sin(value1[0] * .4));
     x[1] = vec3(cos(value1[1]));
     gl_Position = value1;
     x[2] = vec3(10 * x[0])
   }
```

这是一个顶点着色器；有一些线索：

+   +   它有一个属性变量，它分配给`gl_Position`。

+   它有一个可变变量，它正在分配值。

1.  给定以下顶点着色器，你需要写什么代码来为这两个变量分配简单的值？

```py
   attribute highp vec4 coordinates;
   uniform highp mat4 matrix1;

   void main(void){
     gl_Position = matrix1 * coordinates;
   }
```

假设你的`QOpenGLShaderProgram`对象保存为`self.program`，需要以下代码：

```py
   c_handle = self.program.attributeLocation('coordinates')
   m_handle = self.program.uniformLocation('matrix1')
   self.program.setAttributeValue(c_handle, coordinate_value)
   self.program.setUniformValue(m_handle, matrix)
```

1.  你启用面剔除以节省一些处理能力，但发现你的绘图中的几何体没有渲染。可能出了什么问题？

顶点被以错误的顺序绘制。记住，逆时针绘制一个基元会导致远处的面被剔除；顺时针绘制会导致近处的面被剔除。

1.  以下代码对我们的 OpenGL 图像做了什么？

```py
   matrix = qtg.QMatrix4x4()
   matrix.perspective(60, 4/3, 2, 10)
   matrix.translate(1, -1, -4)
   matrix.rotate(45, 1, 0, 0)
```

单独来看，什么也没有。这段代码只是创建一个 4x4 矩阵，并对其进行一些变换操作。然而，如果我们将其传递到一个应用其值到顶点的着色器中，它将创建一个透视投影，将我们的对象移动到空间中，并旋转图像。实际的`matrix`对象只不过是一组数字的矩阵。

1.  尝试演示，并看看你是否可以添加以下功能中的任何一个：

+   +   一个更有趣的形状（金字塔、立方体等）

+   移动对象的更多控件

+   阴影和光效果

+   在对象中动画形状的变化

你自己来吧！

# 第十四章

1.  考虑以下数据集的描述。你会为每个建议哪种图表样式？

+   1.  按日期的 Web 服务器点击次数

1.  每个销售人员每月的销售数据

1.  去年各公司部门支持票的百分比

1.  豆类植物的产量与植物的高度的图表，几百个植物

答案是主观的，但作者建议以下内容：

1.  1.  线图或样条线图，因为它可以说明交通趋势

1.  条形图或堆叠图，因为这样可以让你比较销售人员的销售情况

1.  饼图，因为它代表一组百分比加起来等于 100

1.  散点图，因为你想展示大量数据的一般趋势

1.  以下代码中哪个图表组件尚未配置，结果会是什么？

```py
   data_list = [
       qtc.QPoint(2, 3),
       qtc.QPoint(4, 5),
       qtc.QPoint(6, 7)]
   chart = qtch.QChart()
   series = qtch.QLineSeries()
   series.append(data_list)
   view = qtch.QChartView()
   view.setChart(chart)
   view.show()
```

轴尚未配置。此图表可以显示，但轴上将没有参考标记，并且可能无法直观地进行缩放。

1.  以下代码有什么问题？

```py
   mainwindow = qtw.QMainWindow()
   chart = qtch.QChart()
   series = qtch.QPieSeries()
   series.append('Half', 50)
   series.append('Other Half', 50)
   mainwindow.setCentralWidget(chart)
   mainwindow.show()
```

`QChart`不是一个小部件，不能添加到布局或设置为中央小部件。它必须附加到`QChartView`。

1.  **你想创建一个比较 Bob 和 Alice 季度销售额的柱状图。需要添加什么代码？（注意这里不需要轴。）**

```py
   bob_sales = [2500, 1300, 800]
   alice_sales = [1700, 1850, 2010]

   chart = qtch.QChart()
   series = qtch.QBarSeries()
   chart.addSeries(series)

   # add code here

   # end code
   view = qtch.QChartView()
   view.setChart(chart)
   view.show()
```

我们需要为 Bob 和 Alice 创建柱状图，并将它们附加到系列中：

```py
   bob_set = qtch.QBarSet('Bob')
   alice_set = qtch.QBarSet('Alice')
   bob_set.append(bob_sales)
   alice_set.append(alice_sales)
   series.append(bob_set)
   series.append(alice_set)
```

1.  **给定一个名为`chart`的`QChart`对象，编写代码使图表具有黑色背景和蓝色数据图。**

为此，设置`backgroundBrush`和`theme`属性：

```py
   chart.setBackgroundBrush(
       qtg.QBrush(qtc.Qt.black))
   chart.setTheme(qtch.QChart.ChartThemeBlueIcy)
```

1.  **使用你在上一个图表中使用的技术来为系统监视器脚本中的另外两个图表设置样式。尝试不同的画刷和笔，看看是否可以找到其他需要设置的属性。**

你现在是自己一个人了！

1.  **`QPolarChart`是`QChart`的一个子类，允许你构建极坐标图。查阅 Qt 文档中关于极坐标图的使用，并看看你是否可以创建一个适当数据集的极坐标图。**

你现在是自己一个人了！

1.  **`psutil.cpu_percent()`接受一个可选参数`percpu`，它将创建一个显示每个 CPU 核心使用信息的值列表。更新你的应用程序以使用这个选项，并分别在一个图表上显示每个 CPU 核心的活动。**

你现在还是自己一个人；不过别担心，你可以做到的！

# 第十五章

1.  **你刚刚购买了一个预装了 Raspbian 的树莓派来运行你的 PyQt5 应用程序。当你尝试运行你的应用程序时，你会遇到一个错误，试图导入`QtNetworkAuth`，而你的应用程序依赖于它。可能的问题是什么？**

可能你的 Raspbian 安装版本是 9。版本 9 具有 Qt 5.7，其中没有`QtNetworkAuth`模块。你需要升级到更新的 Raspbian 版本。

1.  **你为一个传统扫描仪设备编写了一个 PyQt 前端。你的代码通过一个名为`scanutil.exe`的专有驱动程序实用程序与扫描仪通信。它目前在 Windows 10 PC 上运行，但你的雇主希望通过将其移植到树莓派来节省成本。这是一个好主意吗？**

不幸的是，不是这样。如果你的应用程序依赖于专有的 Windows x86 二进制文件，那么该程序将无法在树莓派上运行。要切换到树莓派，你需要一个为 ARM 平台编译的二进制文件，可以在树莓派支持的操作系统之一上运行（此外，该操作系统需要能够运行 Python 和 Qt）。

1.  **你已经获得了一个新的传感器，并想要用树莓派试验它。它有三个连接，标有 Vcc、GND 和 Data。你将如何将其连接到树莓派？你还需要更多的信息吗？**

你真的需要更多的信息，但这里有足够的信息让你开始：

+   +   **Vcc**是输入电压的缩写。你将不得不将其连接到树莓派上的 5V 或 3V3 引脚。你需要查阅制造商的文档，以确定哪种连接方式可行。

+   **GND**意味着地线，你可以将其连接到树莓派上的任何地线引脚。

+   **Data**可能是你想要连接到可编程 GPIO 引脚之一的连接。很可能你需要某种库来使其工作，所以你应该向制造商咨询。

1.  **你试图点亮连接到树莓派左侧第四个 GPIO 引脚的 LED。这段代码有什么问题？**

```py
   GPIO.setmode(GPIO.BCM)
   GPIO.setup(8, GPIO.OUT)
   GPIO.output(8, 1)
```

GPIO 引脚模式设置为`BCM`，这意味着你使用的引脚号错误。将模式设置为`BOARD`，或者使用正确的 BCM 引脚号（`14`）。

1.  **你试图调暗连接到 GPIO 引脚`12`的 LED。这段代码有效吗？**

```py
   GPIO.setmode(GPIO.BOARD)
   GPIO.setup(12, GPIO.OUT)
   GPIO.output(12, 0.5)
```

这段代码不起作用，因为引脚只能是开或关。要模拟半电压，你需要使用脉冲宽度调制，就像下面的例子中所示：

```py
   GPIO.setmode(GPIO.BOARD)
   GPIO.setup(12, GPIO.OUT)
   pwm = GPIO.PWM(12, 60)
   pwm.start(0)
   pwm.ChangeDutyCycle(50)
```

1.  **你有一个带有数据引脚的运动传感器，当检测到运动时会变为`HIGH`。它连接到引脚`8`。以下是你的驱动代码：**

```py
   class MotionSensor(qtc.QObject):

       detection = qtc.pyqtSignal()

       def __init__(self):
           super().__init__()
           GPIO.setmode(GPIO.BOARD)
           GPIO.setup(8, GPIO.IN)
           self.state = GPIO.input(8)

       def check(self):
           state = GPIO.input(8)
           if state and state != self.state:
               detection.emit()
           self.state = state
```

**你的主窗口类创建了一个`MotionSensor`对象，并将其`detection`信号连接到一个回调方法。然而，没有检测到任何东西。缺少了什么？**

您没有调用`MotionSensor.check()`。您应该通过添加一个调用`check()`的`QTimer`对象来实现轮询。

1.  **以创造性的方式结合本章中的两个电路；例如，您可以创建一个根据湿度和温度改变颜色的灯。**

这里就靠你自己了！

# 第十六章

1.  **以下代码给出了一个属性错误；怎么了？**

```py
   from PyQt5 import QtWebEngine as qtwe
   w = qtwe.QWebEngineView()
```

您想要导入`QtWebEngineWidgets`，而不是`QtWebEngine`。后者用于与 Qt 的 QML 前端一起使用。

1.  **以下代码应该将`UrlBar`类与`QWebEngineView`连接起来，以便在按下*返回*/*Enter*键时加载输入的 URL。但是它不起作用；怎么了？**

```py
   class UrlBar(qtw.QLineEdit):

       url_request = qtc.pyqtSignal(str)

       def __init__(self):
           super().__init__()
           self.returnPressed.connect(self.request)

       def request(self):
           self.url_request.emit(self.text())

   mywebview = qtwe.QWebEngineView()
   myurlbar = UrlBar()
   myurlbar.url_request(mywebview.load)
```

`QWebEngineView.load()`需要一个`QUrl`对象，而不是一个字符串。`url_request`信号将栏的文本作为字符串直接发送到`load()`。它应该首先将其包装在`QUrl`对象中。

1.  **以下代码的结果是什么？**

```py
   class WebView(qtwe.QWebEngineView):

    def createWindow(self, _):

        return self
```

每当浏览器操作请求创建新的选项卡或窗口时，都会调用`QWebEngineView.createWindow()`，并且预计返回一个`QWebEngineView`对象，该对象将用于新窗口或选项卡。通过返回`self`，这个子类强制任何尝试创建新窗口的链接或调用只是在同一个窗口中导航。

1.  **查看[`doc.qt.io/qt-5/qwebengineview.html`](https://doc.qt.io/qt-5/qwebengineview.html)上的`QWebEngineView`文档。您将如何在浏览器中实现缩放功能？**

首先，您需要在`MainWindow`上实现回调函数，以设置当前 Web 视图的`zoomFactor`属性：

```py
   def zoom_in(self):
        webview = self.tabs.currentWidget()
        webview.setZoomFactor(webview.zoomFactor() * 1.1)

    def zoom_out(self):
        webview = self.tabs.currentWidget()
        webview.setZoomFactor(webview.zoomFactor() * .9)
```

然后，在`MainWindow.__init__()`中，您只需要创建控件来调用这些方法：

```py
   navigation.addAction('Zoom In', self.zoom_in)
   navigation.addAction('Zoom Out', self.zoom_out)
```

1.  **顾名思义，`QWebEngineView`表示模型-视图架构中的视图部分。在这个设计中，哪个类代表模型？**

`QWebEnginePage`似乎是这里最清晰的候选者，因为它存储和控制 Web 内容的呈现。

1.  **给定名为`webview`的`QWebEngineView`，编写代码来确定`webview`上是否启用了 JavaScript。**

代码必须查询视图的`QWebEngineSettings`对象，就像这样：

```py
   webview.settings().testAttribute(
       qtwe.QWebEngineSettings.JavascriptEnabled)
```

1.  **您在我们的浏览器示例中看到`runJavaScript()`可以将整数值传递给回调函数。编写一个简单的演示脚本来测试可以返回哪些其他类型的 JavaScript 对象，以及它们在 Python 代码中的显示方式。**

在示例代码中查看`chapter_7_return_value_test.py`。

# 第十七章

1.  **您已经在名为`Scan & Print Tool-box.py`的文件中编写了一个 PyQt 应用程序。您想将其转换为模块样式的组织；您应该做出什么改变？**

脚本的名称应该更改，因为空格、和符号和破折号不是 Python 模块名称中使用的有效字符。例如，您可以将模块名称更改为`scan_and_print_toolbox`。

1.  **您的 PyQt5 数据库应用程序有一组包含应用程序使用的查询的`.sql`文件。当您的应用程序是与`.sql`文件在同一个目录中的单个脚本时，它可以工作，但是现在您已经将其转换为模块样式的组织，就无法找到查询。你应该怎么办？**

最好的做法是将您的`.sql`文件放入 Qt 资源文件中，并将其作为 Python 模块的一部分。如果无法使用 Qt 资源文件，您将需要使用`path`模块和内置的`file`变量将相对路径转换为绝对路径

1.  **在将新应用程序上传到代码共享站点之前，您正在编写一个详细的`README.rst`文件来记录您的新应用程序。分别应该使用哪些字符来标记您的一级、二级和三级标题？**

实际上并不重要，只要使用可接受字符列表中的字符即可：

```py
   = - ` : ' " ~ ^ _ * + # < >
```

RST 解释器应该考虑遇到的第一个标题字符表示一级；第二个表示二级；第三个表示三级。

1.  您正在为您的项目创建一个`setup.py`脚本，以便您可以将其上传到 PyPI。您想要包括项目的 FAQ 页面的 URL。您该如何实现这一点？

您需要向`project_urls`字典中添加一个`key: value`对，就像这样：

```py
   setup(
       project_urls={
           'Project FAQ': 'https://example.com/faq',
       }
   )
```

1.  您在`setup.py`文件中指定了`include_package_data=True`，但由于某种原因，`docs`文件夹没有包含在您的分发包中。出了什么问题？

`include_package_data`只影响包（模块）内的数据文件。如果您想要包括模块外的文件，您需要使用`MANIFEST.in`文件。

1.  您运行了`pyinstaller fight_fighter3.py`来将您的新游戏打包为可执行文件。不过出了些问题；您可以在哪里找到构建过程的日志？

首先，您需要查看`build/fight_fighter3/warn-fight_fighter3.txt`。您可能需要通过使用`--log-level DEBUG`参数调用 PyInstaller 来增加调试输出。

1.  尽管名字是这样，但 PyInstaller 实际上不能生成安装程序或包来安装您的应用程序。研究一些适合您平台的选项。

您需要自己解决这个问题，尽管一个流行的选项是**Nullsoft Scriptable Install System**（**NSIS**）。
