# 使用QtWidgets构建表单

应用程序开发的第一步之一是原型设计应用程序的GUI。有了各种各样的现成小部件，PyQt使这变得非常容易。最重要的是，当我们完成后，我们可以直接将我们的原型代码移植到实际应用程序中。

在这一章中，我们将通过以下主题熟悉基本的表单设计：

+   创建基本的QtWidgets小部件

+   放置和排列小部件

+   验证小部件

+   构建一个日历应用程序的GUI

# 技术要求

要完成本章，您需要从[第1章](bce5f3b1-2979-4f78-817b-3986e7974725.xhtml) *PyQt入门*中获取所有内容，以及来自[https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter02](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter02)的示例代码。

查看以下视频以查看代码的实际效果：[http://bit.ly/2M2R26r](http://bit.ly/2M2R26r)

# 创建基本的QtWidgets小部件

`QtWidgets`模块包含数十个小部件，有些简单和标准，有些复杂和独特。在本节中，我们将介绍八种最常见的小部件及其基本用法。

在开始本节之前，从[第1章](bce5f3b1-2979-4f78-817b-3986e7974725.xhtml) *PyQt入门*中复制您的应用程序模板，并将其保存到名为`widget_demo.py`的文件中。当我们逐个示例进行时，您可以将它们添加到您的`MainWindow.__init__()`方法中，以查看这些对象的工作方式。

# QWidget

`QWidget`是所有其他小部件的父类，因此它拥有的任何属性和方法也将在任何其他小部件中可用。单独使用时，`QWidget`对象可以作为其他小部件的容器，填充空白区域，或作为顶层窗口的基类。

创建小部件就像这样简单：

```py
        # inside MainWindow.__init__()
        subwidget = qtw.QWidget(self)
```

请注意我们将`self`作为参数传递。如果我们正在创建一个小部件以放置在或在另一个小部件类中使用，就像我们在这里做的那样，将父小部件的引用作为第一个参数传递是一个好主意。指定父小部件将确保在父小部件被销毁和清理时，子小部件也被销毁，并限制其可见性在父小部件内部。

正如您在[第1章](bce5f3b1-2979-4f78-817b-3986e7974725.xhtml)中学到的，*PyQt入门*，PyQt也允许我们为任何小部件的属性指定值。

例如，我们可以使用`toolTip`属性来设置此小部件的工具提示文本（当鼠标悬停在小部件上时将弹出）：

```py
        subwidget = qtw.QWidget(self, toolTip='This is my widget')
```

阅读`QWidget`的C++文档（位于[https://doc.qt.io/qt-5/qwidget.html](https://doc.qt.io/qt-5/qwidget.html)）并注意类的属性。请注意，每个属性都有指定的数据类型。在这种情况下，`toolTip`需要`QString`。每当需要`QString`时，我们可以使用常规Unicode字符串，因为PyQt会为我们进行转换。然而，对于更奇特的数据类型，如`QSize`或`QColor`，我们需要创建适当的对象。请注意，这些转换是在后台进行的，因为Qt对数据类型并不宽容。

例如，这段代码会导致错误：

```py
        subwidget = qtw.QWidget(self, toolTip=b'This is my widget')
```

这将导致`TypeError`，因为PyQt不会将`bytes`对象转换为`QString`。因此，请确保检查小部件属性或方法调用所需的数据类型，并使用兼容的类型。

# QWidget作为顶层窗口

当创建一个没有父级的`QWidget`并调用它的`show()`方法时，它就成为了一个顶层窗口。当我们将其用作顶层窗口时，例如我们在`MainWindow`实例中所做的那样，我们可以设置一些特定于窗口的属性。其中一些显示在下表中：

| 属性 | 参数类型 | 描述 |
| --- | --- | --- |
| `windowTitle` | 字符串 | 窗口的标题。 |
| `windowIcon` | `QIcon` | 窗口的图标。 |
| `modal` | 布尔值 | 窗口是否为模态。 |
| `cursor` | `Qt.CursorShape` | 当小部件悬停时使用的光标。 |
| `windowFlags` | `Qt.WindowFlags` | 操作系统应如何处理窗口（对话框、工具提示、弹出窗口）。 |

`cursor`的参数类型是枚举的一个例子。枚举只是一系列命名的值，Qt在属性受限于一组描述性值的任何地方定义枚举。`windowFlags`的参数是标志的一个例子。标志类似于枚举，不同之处在于它们可以组合（使用管道运算符`|`），以便传递多个标志。

在这种情况下，枚举和标志都是`Qt`命名空间的一部分，位于`QtCore`模块中。因此，例如，要在小部件悬停时将光标设置为箭头光标，您需要找到`Qt`中引用箭头光标的正确常量，并将小部件的`cursor`属性设置为该值。要在窗口上设置标志，指示操作系统它是`sheet`和`popup`窗口，您需要找到`Qt`中表示这些窗口标志的常量，用管道组合它们，并将其作为`windowFlags`的值传递。

创建这样一个`QWidget`窗口可能是这样的：

```py
window = qtw.QWidget(cursor=qtc.Qt.ArrowCursor)
window.setWindowFlags(qtc.Qt.Sheet|qtc.Qt.Popup)
```

在本书的其余部分学习配置Qt小部件时，我们将遇到更多的标志和枚举。

# QLabel

`QLabel`是一个配置为显示简单文本和图像的`QWidget`对象。

创建一个看起来像这样的：

```py
        label = qtw.QLabel('Hello Widgets!', self)
```

注意这次指定的父窗口小部件是第二个参数，而第一个参数是标签的文本。

这里显示了一些常用的`QLabel`属性：

| 属性 | 参数 | 描述 |
| --- | --- | --- |
| `text` | string | 标签上显示的文本。 |
| `margin` | 整数 | 文本周围的空间（以像素为单位）。 |
| `indent` | 整数 | 文本缩进的空间（以像素为单位）。 |
| `wordWrap` | 布尔值 | 是否换行。 |
| `textFormat` | `Qt.TextFormat` | 强制纯文本或富文本，或自动检测。 |
| `pixmap` | `QPixmap` | 要显示的图像而不是文本。 |

标签的文本存储在其`text`属性中，因此可以使用相关的访问器方法来访问或更改，如下所示：

```py
        label.setText("Hi There, Widgets!")
        print(label.text())
```

`QLabel`可以显示纯文本、富文本或图像。Qt中的富文本使用类似HTML的语法；默认情况下，标签将自动检测您的字符串是否包含任何格式标记，并相应地显示适当类型的文本。例如，如果我们想要使我们的标签加粗并在文本周围添加边距，我们可以这样做：

```py
        label = qtw.QLabel('<b>Hello Widgets!</b>', self, margin=10)
```

我们将在[第6章](c3eb2567-0e73-4c37-9a9e-a0e2311e106c.xhtml) *Qt应用程序样式*和[第11章](a9b58d41-a0ec-41f8-8f59-39ae2bc921ee.xhtml) *使用QTextDocument创建富文本*中学习更多关于使用图像、富文本和字体的知识。

# QLineEdit

`QLineEdit`类是一个单行文本输入小部件，您可能经常在数据输入或登录表单中使用。`QLineEdit`可以不带参数调用，只带有父窗口小部件，或者将默认字符串值作为第一个参数，如下所示：

```py
        line_edit = qtw.QLineEdit('default value', self)
```

还有许多我们可以传递的属性：

| 属性 | 参数 | 描述 |
| --- | --- | --- |
| `text` | string | 盒子的内容。 |
| `readOnly` | 布尔值 | 字段是否可编辑。 |
| `clearButtonEnabled` | 布尔值 | 是否添加清除按钮。 |
| `placeholderText` | string | 字段为空时显示的文本。 |
| `maxLength` | 整数 | 可输入的最大字符数。 |
| `echoMode` | `QLineEdit.EchoMode` | 切换文本输入时显示方式（例如用于密码输入）。 |

让我们给我们的行编辑小部件添加一些属性：

```py
        line_edit = qtw.QLineEdit(
            'default value',
            self,
            placeholderText='Type here',
            clearButtonEnabled=True,
            maxLength=20
        )
```

这将用默认文本'默认值'填充小部件。当字段为空或有一个清除字段的小`X`按钮时，它将显示一个占位符字符串'在此输入'。它还限制了可以输入的字符数为`20`。

# QPushButton和其他按钮

`QPushButton`是一个简单的可点击按钮小部件。与`QLabel`和`QLineEdit`一样，它可以通过第一个参数调用，该参数指定按钮上的文本，如下所示：

```py
        button = qtw.QPushButton("Push Me", self)
```

我们可以在`QPushButton`上设置的一些更有用的属性包括以下内容：

| 属性 | 参数 | 描述 |
| --- | --- | --- |
| `checkable` | 布尔值 | 按钮是否在按下时保持开启状态。 |
| `checked` | 布尔值 | 对于`checkable`按钮，按钮是否被选中。 |
| `icon` | `QIcon` | 要显示在按钮上的图标图像。 |
| `shortcut` | `QKeySequence` | 一个激活按钮的键盘快捷键。 |

`checkable`和`checked`属性允许我们将此按钮用作反映开/关状态的切换按钮，而不仅仅是执行操作的单击按钮。所有这些属性都来自`QPushButton`类的父类`QAbstractButton`。这也是其他几个按钮类的父类，列在这里：

| 类 | 描述 |
| --- | --- |
| `QCheckBox` | 复选框可以是开/关的布尔值，也可以是开/部分开/关的三态值。 |
| `QRadioButton` | 类似复选框，但在具有相同父级的按钮中只能选中一个按钮。 |
| `QToolButton` | 用于工具栏小部件的特殊按钮。 |

尽管每个按钮都有一些独特的特性，但在核心功能方面，这些按钮在我们创建和配置它们的方式上是相同的。

让我们将我们的按钮设置为可选中，默认选中，并给它一个快捷键：

```py
        button = qtw.QPushButton(
            "Push Me",
            self,
            checkable=True,
            checked=True,
            shortcut=qtg.QKeySequence('Ctrl+p')
        )
```

请注意，`shortcut`选项要求我们传入一个`QKeySequence`，它是`QtGui`模块的一部分。这是一个很好的例子，说明属性参数通常需要包装在某种实用类中。`QKeySequence`封装了一个键组合，这里是*Ctrl*键（或macOS上的*command*键）和*P*。

键序列可以指定为字符串，例如前面的示例，也可以使用`QtCOre.Qt`模块中的枚举值。例如，我们可以将前面的示例写为`QKeySequence(qtc.Qt.CTRL + qtc.Qt.Key_P)`。

# QComboBox

**combobox**，也称为下拉或选择小部件，是一个在点击时呈现选项列表的小部件，其中必须选择一个选项。`QCombobox`可以通过将其`editable`属性设置为`True`来允许文本输入自定义答案。

让我们创建一个`QCombobox`对象，如下所示：

```py
        combobox = qtw.QComboBox(self)
```

现在，我们的`combobox`菜单中没有项目。`QCombobox`在构造函数中不提供使用选项初始化小部件的方法；相反，我们必须创建小部件，然后使用`addItem()`或`insertItem()`方法来填充其菜单选项，如下所示：

```py
        combobox.addItem('Lemon', 1)
        combobox.addItem('Peach', 'Ohh I like Peaches!')
        combobox.addItem('Strawberry', qtw.QWidget)
        combobox.insertItem(1, 'Radish', 2)
```

`addItem()`方法接受标签和数据值的字符串。正如你所看到的，这个值可以是任何东西——整数，字符串，Python类。可以使用`QCombobox`对象的`currentData()`方法检索当前选定项目的值。通常最好——尽管不是必需的——使所有项目的值都是相同类型的。

`addItem()`将始终将项目附加到菜单的末尾；要在之前插入它们，使用`insertItem()`方法。它的工作方式完全相同，只是它接受一个索引（整数值）作为第一个参数。项目将插入到列表中的该索引处。如果我们想节省时间，不需要为我们的项目设置`data`属性，我们也可以使用`addItems()`或`insertItems()`传递一个选项列表。

`QComboBox`的一些其他重要属性包括以下内容：

| 属性 | 参数 | 描述 |
| --- | --- | --- |
| `currentData` | （任何） | 当前选定项目的数据对象。 |
| `currentIndex` | 整数 | 当前选定项目的索引。 |
| `currentText` | string | 当前选定项目的文本。 |
| `editable` | 布尔值 | `combobox`是否允许文本输入。 |
| `insertPolicy` | `QComboBox.InsertPolicy` | 输入的项目应该插入列表中的位置。 |

`currentData`的数据类型是`QVariant`，这是Qt的一个特殊类，用作任何类型数据的容器。在C++中更有用，因为它们为多种数据类型可能有用的情况提供了一种绕过静态类型的方法。PyQt会自动将`QVariant`对象转换为最合适的Python类型，因此我们很少需要直接使用这种类型。

让我们更新我们的`combobox`，以便我们可以将项目添加到下拉列表的顶部：

```py
        combobox = qtw.QComboBox(
            self,
            editable=True,
            insertPolicy=qtw.QComboBox.InsertAtTop
        )
```

现在这个`combobox`将允许输入任何文本；文本将被添加到列表框的顶部。新项目的`data`属性将为`None`，因此这实际上只适用于我们仅使用可见字符串的情况。

# QSpinBox

一般来说，旋转框是一个带有箭头按钮的文本输入，旨在*旋转*一组递增值。`QSpinbox`专门用于处理整数或离散值（例如下拉框）。

一些有用的`QSpinBox`属性包括以下内容：

| 属性 | 参数 | 描述 |
| --- | --- | --- |
| `value` | 整数 | 当前旋转框值，作为整数。 |
| `cleanText` | string | 当前旋转框值，作为字符串（不包括前缀和后缀）。 |
| `maximum` | 整数 | 方框的最大整数值。 |
| `minimum` | 整数 | 方框的最小值。 |
| `prefix` | string | 要添加到显示值的字符串。 |
| `suffix` | string | 要附加到显示值的字符串。 |
| `singleStep` | 整数 | 当使用箭头时增加或减少值的数量。 |
| `wrapping` | 布尔值 | 当使用箭头时是否从范围的一端包装到另一端。 |

让我们在脚本中创建一个`QSpinBox`对象，就像这样：

```py
        spinbox = qtw.QSpinBox(
            self,
            value=12,
            maximum=100,
            minimum=10,
            prefix='$',
            suffix=' + Tax',
            singleStep=5
        )
```

这个旋转框从值`12`开始，并允许输入从`10`到`100`的整数，以`$<value> + Tax`的格式显示。请注意，框的非整数部分不可编辑。还要注意，虽然增量和减量箭头移动`5`，但我们可以输入不是`5`的倍数的值。

`QSpinBox`将自动忽略非数字的按键，或者会使值超出可接受范围。如果输入了一个太低的值，当焦点从`spinbox`移开时，它将被自动更正为有效值；例如，如果您在前面的框中输入了`9`并单击了它，它将被自动更正为`90`。

`QDoubleSpinBox`与`QSpinBox`相同，但设计用于十进制或浮点数。

要将`QSpinBox`用于离散文本值而不是整数，您需要对其进行子类化并重写其验证方法。我们将在*验证小部件*部分中进行。

# QDateTimeEdit

旋转框的近亲是`QDateTimeEdit`，专门用于输入日期时间值。默认情况下，它显示为一个旋转框，允许用户通过每个日期时间值字段进行制表，并使用箭头递增/递减它。该小部件还可以配置为使用日历弹出窗口。

更有用的属性包括以下内容：

| 属性 | 参数 | 描述 |
| --- | --- | --- |
| `date` | `QDate`或`datetime.date` | 日期值。 |
| `time` | `QTime`或`datetime.time` | 时间值。 |
| `dateTime` | `QDateTime`或`datetime.datetime` | 组合的日期时间值。 |
| `maximumDate`，`minimumDate` | `QDate`或`datetime.date` | 可输入的最大和最小日期。 |
| `maximumTime`，`minimumTime` | `QTime`或`datetime.time` | 可输入的最大和最小时间。 |
| `maximumDateTime`，`minimumDateTime` | `QDateTime`或`datetime.datetime` | 可输入的最大和最小日期时间。 |
| `calendarPopup` | 布尔值 | 是否显示日历弹出窗口或像旋转框一样行为。 |
| `displayFormat` | string | 日期时间应如何格式化。 |

让我们像这样创建我们的日期时间框：

```py
       datetimebox = qtw.QDateTimeEdit(
            self,
            date=qtc.QDate.currentDate(),
            time=qtc.QTime(12, 30),
            calendarPopup=True,
            maximumDate=qtc.QDate(2030, 1, 1),
            maximumTime=qtc.QTime(17, 0),
            displayFormat='yyyy-MM-dd HH:mm'
        )
```

这个日期时间小部件将使用以下属性创建：

+   当前日期将设置为12:30

+   当焦点集中时，它将显示日历弹出窗口

+   它将禁止在2030年1月1日之后的日期

+   它将禁止在最大日期后的17:00（下午5点）之后的时间

+   它将以年-月-日小时-分钟的格式显示日期时间

请注意，`maximumTime`和`minimumTime`只影响`maximumDate`和`minimumDate`的值，分别。因此，即使我们指定了17:00的最大时间，只要在2030年1月1日之前，您也可以输入18:00。相同的概念也适用于最小日期和时间。

日期时间的显示格式是使用包含每个项目的特定替换代码的字符串设置的。这里列出了一些常见的代码：

| 代码 | 意义 |
| --- | --- |
| `d` | 月份中的日期。 |
| `M` | 月份编号。 |
| `yy` | 两位数年份。 |
| `yyyy` | 四位数年份。 |
| `h` | 小时。 |
| `m` | 分钟。 |
| `s` | 秒。 |
| `A` | 上午/下午，如果使用，小时将切换到12小时制。 |

日，月，小时，分钟和秒都默认省略前导零。要获得前导零，只需将字母加倍（例如，`dd`表示带有前导零的日期）。代码的完整列表可以在[https://doc.qt.io/qt-5/qdatetime.html](https://doc.qt.io/qt-5/qdatetime.html)找到。

请注意，所有时间、日期和日期时间都可以接受来自Python标准库的`datetime`模块以及Qt类型的对象。因此，我们的框也可以这样创建：

```py
        import datetime
        datetimebox = qtw.QDateTimeEdit(
            self,
            date=datetime.date.today(),
            time=datetime.time(12, 30),
            calendarPopup=True,
            maximumDate=datetime.date(2020, 1, 1),
            minimumTime=datetime.time(8, 0),
            maximumTime=datetime.time(17, 0),
            displayFormat='yyyy-MM-dd HH:mm'
        )
```

你选择使用哪一个取决于个人偏好或情境要求。例如，如果您正在使用其他Python模块，`datetime`标准库对象可能更兼容。如果您只需要为小部件设置默认值，`QDateTime`可能更方便，因为您可能已经导入了`QtCore`。

如果您需要更多对日期和时间输入的控制，或者只是想将它们拆分开来，Qt有`QTimeEdit`和`QDateEdit`小部件。它们就像这个小部件一样，只是分别处理时间和日期。

# QTextEdit

虽然`QLineEdit`用于单行字符串，但`QTextEdit`为我们提供了输入多行文本的能力。`QTextEdit`不仅仅是一个简单的纯文本输入，它是一个完整的所见即所得编辑器，可以配置为支持富文本和图像。

这里显示了`QTextEdit`的一些更有用的属性：

| 属性 | 参数 | 描述 |
| --- | --- | --- |
| `plainText` | 字符串 | 框的内容，纯文本格式。 |
| `html` | 字符串 | 框的内容，富文本格式。 |
| `acceptRichText` | 布尔值 | 框是否允许富文本。 |
| `lineWrapColumnOrWidth` | 整数 | 文本将换行的像素或列。 |
| `lineWrapMode` | `QTextEdit.LineWrapMode` | 行换行模式使用列还是像素。 |
| `overwriteMode` | 布尔值 | 是否激活覆盖模式；`False`表示插入模式。 |
| `placeholderText` | 字符串 | 字段为空时显示的文本。 |
| `readOnly` | 布尔值 | 字段是否只读。 |

让我们创建一个文本编辑器，如下所示：

```py
        textedit = qtw.QTextEdit(
            self,
            acceptRichText=False,
            lineWrapMode=qtw.QTextEdit.FixedColumnWidth,
            lineWrapColumnOrWidth=25,
            placeholderText='Enter your text here'
            )
```

这将创建一个纯文本编辑器，每行只允许输入`25`个字符，当为空时显示短语`'在此输入您的文本'`。

我们将在[第11章](a9b58d41-a0ec-41f8-8f59-39ae2bc921ee.xhtml)中深入了解`QTextEdit`和富文本文档，*使用QTextDocument创建富文本*。

# 放置和排列小部件

到目前为止，我们已经创建了许多小部件，但如果运行程序，您将看不到它们。虽然我们的小部件都属于父窗口，但它们还没有放置在上面。在本节中，我们将学习如何在应用程序窗口中排列我们的小部件，并将它们设置为适当的大小。

# 布局类

布局对象定义了子小部件在父小部件上的排列方式。Qt提供了各种布局类，每个类都有适合不同情况的布局策略。

使用布局类的工作流程如下：

1.  从适当的布局类创建布局对象

1.  使用`setLayout()`方法将布局对象分配给父小部件的`layout`属性

1.  使用布局的`addWidget()`方法向布局添加小部件

您还可以使用`addLayout()`方法将布局添加到布局中，以创建更复杂的小部件排列。让我们来看看Qt提供的一些基本布局类。

# QHBoxLayout和QVBoxLayout

`QHBoxLayout`和`QVBoxLayout`都是从`QBoxLayout`派生出来的，这是一个非常基本的布局引擎，它简单地将父对象分成水平或垂直框，并按顺序放置小部件。`QHBoxLayout`是水平定向的，小部件按添加顺序从左到右放置。`QVBoxLayout`是垂直定向的，小部件按添加顺序从上到下放置。

让我们在`MainWindow`小部件上尝试`QVBoxLayout`：

```py
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)
```

一旦布局对象存在，我们可以使用`addWidget()`方法开始向其中添加小部件：

```py
        layout.addWidget(label)
        layout.addWidget(line_edit)
```

如您所见，如果运行程序，小部件将逐行添加。如果我们想要将多个小部件添加到一行中，我们可以像这样在布局中嵌套一个布局：

```py
        sublayout = qtw.QHBoxLayout()
        layout.addLayout(sublayout)

        sublayout.addWidget(button)
        sublayout.addWidget(combobox)
```

在这里，我们在主垂直布局的下一个单元格中添加了一个水平布局，然后在子布局中插入了三个更多的小部件。这三个小部件在主布局的一行中并排显示。大多数应用程序布局可以通过简单地嵌套框布局来完成。

# QGridLayout

嵌套框布局涵盖了很多内容，但在某些情况下，您可能希望以统一的行和列排列小部件。这就是`QGridLayout`派上用场的地方。顾名思义，它允许您以表格结构放置小部件。

像这样创建一个网格布局对象：

```py
        grid_layout = qtw.QGridLayout()
        layout.addLayout(grid_layout)
```

向`QGridLayout`添加小部件类似于`QBoxLayout`类的方法，但还需要传递坐标：

```py
        grid_layout.addWidget(spinbox, 0, 0)
        grid_layout.addWidget(datetimebox, 0, 1)
        grid_layout.addWidget(textedit, 1, 0, 2, 2)
```

这是`QGridLayout.addWidget()`的参数，顺序如下：

1.  要添加的小部件

1.  行号（垂直坐标），从`0`开始

1.  列号（水平坐标），从`0`开始

1.  行跨度，或者小部件将包含的行数（可选）

1.  列跨度，或者小部件将包含的列数（可选）

因此，我们的`spinbox`小部件放置在第`0`行，第`0`列，即左上角；我们的`datetimebox`放置在第`0`行，第`1`列，即右上角；我们的`textedit`放置在第`1`行，第`0`列，并且跨越了两行两列。

请记住，网格布局保持所有列的宽度一致，所有行的高度一致。因此，如果您将一个非常宽的小部件放在第`2`行，第`1`列，所有行中位于第`1`列的小部件都会相应地被拉伸。如果希望每个单元格独立拉伸，请改用嵌套框布局。

# QFormLayout

在创建数据输入表单时，通常会在标签旁边放置标签。Qt为这种情况提供了一个方便的两列网格布局，称为`QFormLayout`。

让我们向我们的GUI添加一个表单布局：

```py
        form_layout = qtw.QFormLayout()
        layout.addLayout(form_layout)
```

使用`addRow()`方法可以轻松添加小部件：

```py
        form_layout.addRow('Item 1', qtw.QLineEdit(self))
        form_layout.addRow('Item 2', qtw.QLineEdit(self))
        form_layout.addRow(qtw.QLabel('<b>This is a label-only row</b>'))
```

这个方便的方法接受一个字符串和一个小部件，并自动为字符串创建`QLabel`小部件。如果只传递一个小部件（如`QLabel`），该小部件跨越两列。这对于标题或部分标签非常有用。

`QFormLayout`不仅仅是对`QGridLayout`的方便，它还在跨不同平台使用时自动提供成语化的行为。例如，在Windows上使用时，标签是左对齐的；在macOS上使用时，标签是右对齐的，符合平台的设计指南。此外，当在窄屏幕上查看（如移动设备），布局会自动折叠为单列，标签位于输入框上方。在任何需要两列表单的情况下使用这种布局是非常值得的。

# 控制小部件大小

如果您按照当前的设置运行我们的演示并将其扩展以填满屏幕，您会注意到主布局的每个单元格都会均匀拉伸以填满屏幕，如下所示：

![](assets/2b113c24-8f5f-4608-a786-8a0e4d6b40bd.png)

这并不理想。顶部的标签实际上不需要扩展，并且底部有很多空间被浪费。据推测，如果用户要扩展此窗口，他们会这样做以获得更多的输入小部件空间，就像我们的`QTextEdit`。我们需要为GUI提供一些关于如何调整小部件的大小以及在窗口从其默认大小扩展或收缩时如何调整它们的指导。

在任何工具包中，控制小部件的大小可能会有些令人困惑，但Qt的方法可能尤其令人困惑，因此让我们一步一步来。

我们可以简单地使用其`setFixedSize()`方法为任何小部件设置固定大小，就像这样：

```py
        # Fix at 150 pixels wide by 40 pixels high
        label.setFixedSize(150, 40)
```

`setFixedSize`仅接受像素值，并且设置为固定大小的小部件在任何情况下都不能改变这些像素大小。以这种方式调整小部件的大小的问题在于它没有考虑不同字体、不同文本大小或应用程序窗口的大小或布局发生变化的可能性，这可能导致小部件对其内容太小或过大。我们可以通过设置`minimumSize`和`maximumSize`使其稍微灵活一些，就像这样：

```py
        # setting minimum and maximum sizes
        line_edit.setMinimumSize(150, 15)
        line_edit.setMaximumSize(500, 50)
```

如果您运行此代码并调整窗口大小，您会注意到`line_edit`在窗口扩展和收缩时具有更大的灵活性。但是，请注意，小部件不会收缩到其`minimumSize`以下，但即使有空间可用，它也不一定会使用其`maximumSize`。

因此，这仍然远非理想。与其关心每个小部件消耗多少像素，我们更希望它根据其内容和在界面中的角色合理而灵活地调整大小。Qt正是使用*大小提示*和*大小策略*的概念来实现这一点。

大小提示是小部件的建议大小，并由小部件的`sizeHint()`方法返回。此大小可能基于各种动态因素；例如，`QLabel`小部件的`sizeHint()`值取决于其包含的文本的长度和换行。由于它是一个方法而不是属性，因此为小部件设置自定义`sizeHint()`需要您对小部件进行子类化并重新实现该方法。幸运的是，这并不是我们经常需要做的事情。

大小策略定义了小部件在调整大小请求时如何响应其大小提示。这是作为小部件的`sizePolicy`属性设置的。大小策略在`QtWidgets.QSizePolicy.Policy`枚举中定义，并使用`setSizePolicy`访问器方法分别为小部件的水平和垂直尺寸设置。可用的策略在此处列出：

| 策略 | 描述 |
| --- | --- |
| 固定 | 永远不要增长或缩小。 |
| 最小 | 不要小于`sizeHint`。扩展并不有用。 |
| 最大 | 不要大于`sizeHint`，如果有必要则缩小。 |
| 首选 | 尝试是`sizeHint`，但如果有必要则缩小。扩展并不有用。这是默认值。 |
| 扩展 | 尝试是`sizeHint`，如果有必要则缩小，但尽可能扩展。 |
| 最小扩展 | 不要小于`sizeHint`，但尽可能扩展。 |
| 忽略 | 完全忘记`sizeHint`，尽可能占用更多空间。 |

因此，例如，如果我们希望SpinBox保持固定宽度，以便旁边的小部件可以扩展，我们将这样做：

```py
      spinbox.setSizePolicy(qtw.QSizePolicy.Fixed,qtw.QSizePolicy.Preferred)
```

或者，如果我们希望我们的`textedit`小部件尽可能填满屏幕，但永远不要缩小到其`sizeHint()`值以下，我们应该像这样设置其策略：

```py
        textedit.setSizePolicy(
            qtw.QSizePolicy.MinimumExpanding,
            qtw.QSizePolicy.MinimumExpanding
        )
```

当您有深度嵌套的布局时，调整小部件的大小可能有些不可预测；有时覆盖`sizeHint()`会很方便。在Python中，可以使用Lambda函数快速实现这一点，就像这样：

```py
        textedit.sizeHint = lambda : qtc.QSize(500, 500)
```

请注意，`sizeHint()`必须返回`QtCore.QSize`对象，而不仅仅是整数元组。

在使用框布局时，控制小部件大小的最后一种方法是在将小部件添加到布局时设置一个`stretch`因子。拉伸是`addWidget()`的可选第二个参数，它定义了每个小部件的比较拉伸。

这个例子展示了`stretch`因子的使用：

```py
        stretch_layout = qtw.QHBoxLayout()
        layout.addLayout(stretch_layout)
        stretch_layout.addWidget(qtw.QLineEdit('Short'), 1)
        stretch_layout.addWidget(qtw.QLineEdit('Long'), 2)
```

`stretch`只适用于`QHBoxLayout`和`QVBoxLayout`类。

在这个例子中，我们添加了一个拉伸因子为`1`的行编辑，和一个拉伸因子为`2`的第二个。当你运行这个程序时，你会发现第二个行编辑的长度大约是第一个的两倍。

请记住，拉伸不会覆盖大小提示或大小策略，因此根据这些因素，拉伸比例可能不会完全按照指定的方式进行。

# 容器小部件

我们已经看到我们可以使用`QWidget`作为其他小部件的容器。Qt还为我们提供了一些专门设计用于包含其他小部件的特殊小部件。我们将看看其中的两个：`QTabWidget`和`QGroupBox`。

# QTabWidget

`QTabWidget`，有时在其他工具包中被称为**笔记本小部件**，允许我们通过选项卡选择多个*页面*。它们非常适用于将复杂的界面分解为更容易用户接受的较小块。

使用`QTabWidget`的工作流程如下：

1.  创建`QTabWidget`对象

1.  在`QWidget`或其他小部件类上构建一个UI页面

1.  使用`QTabWidget.addTab()`方法将页面添加到选项卡小部件

让我们试试吧；首先，创建选项卡小部件：

```py
        tab_widget = qtw.QTabWidget()
        layout.addWidget(tab_widget)
```

接下来，让我们将我们在*放置和排列小部件*部分下构建的`grid_layout`移动到一个容器小部件下：

```py
        container = qtw.QWidget(self)
        grid_layout = qtw.QGridLayout()
        # comment out this line:
        #layout.addLayout(grid_layout)
        container.setLayout(grid_layout)
```

最后，让我们将我们的`container`小部件添加到一个新的选项卡中：

```py
        tab_widget.addTab(container, 'Tab the first')
```

`addTab()`的第二个参数是选项卡上将显示的标题文本。可以通过多次调用`addTab()`来添加更多的选项卡，就像这样：

```py
        tab_widget.addTab(subwidget, 'Tab the second')
```

`insertTab()`方法也可以用于在末尾以外的其他位置添加新的选项卡。

`QTabWidget`有一些我们可以自定义的属性，列在这里：

| 属性 | 参数 | 描述 |
| --- | --- | --- |
| `movable` | 布尔值 | 选项卡是否可以重新排序。默认值为`False`。 |
| `tabBarAutoHide` | 布尔值 | 当只有一个选项卡时，选项卡栏是隐藏还是显示。 |
| `tabPosition` | `QTabWidget.TabPosition` | 选项卡出现在小部件的哪一侧。默认值为North（顶部）。 |
| `tabShape` | `QTabWidget.TabShape` | 选项卡的形状。可以是圆角或三角形。 |
| `tabsClosable` | 布尔值 | 是否在选项卡上显示一个关闭按钮。 |
| `useScrollButtons` | 布尔值 | 是否在有许多选项卡时使用滚动按钮或展开。 |

让我们修改我们的`QTabWidget`，使其在小部件的左侧具有可移动的三角形选项卡：

```py
        tab_widget = qtw.QTabWidget(
            movable=True,
            tabPosition=qtw.QTabWidget.West,
            tabShape=qtw.QTabWidget.Triangular
        )
```

`QStackedWidget`类似于选项卡小部件，只是它不包含用于切换页面的内置机制。如果您想要构建自己的选项卡切换机制，您可能会发现它很有用。

# QGroupBox

`QGroupBox`提供了一个带有标签的面板，并且（取决于平台样式）有边框。它对于在表单上将相关的输入分组在一起非常有用。我们创建`QGroupBox`的方式与创建`QWidget`容器的方式相同，只是它可以有一个边框和一个框的标题，例如：

```py
        groupbox = qtw.QGroupBox('Buttons')
        groupbox.setLayout(qtw.QHBoxLayout())
        groupbox.layout().addWidget(qtw.QPushButton('OK'))
        groupbox.layout().addWidget(qtw.QPushButton('Cancel'))
        layout.addWidget(groupbox)
```

在这里，我们创建了一个带有`Buttons`标题的分组框。我们给它一个水平布局，并添加了两个按钮小部件。

请注意，在这个例子中，我们没有像以前那样给布局一个自己的句柄，而是创建了一个匿名的`QHBoxLayout`，然后使用小部件的`layout()`访问器方法来检索一个引用，以便添加小部件。在某些情况下，您可能更喜欢这种方法。

分组框相当简单，但它确实有一些有趣的属性：

| 属性 | 参数 | 描述 |
| --- | --- | --- |
| `title` | 字符串 | 标题文本。 |
| `checkable` | 布尔值 | groupbox是否有一个复选框来启用/禁用它的内容。 |
| `checked` | 布尔值 | 一个可勾选的groupbox是否被勾选（启用）。 |
| `alignment` | `QtCore.Qt.Alignment` | 标题文本的对齐方式。 |
| `flat` | 布尔值 | 盒子是平的还是有框架。 |

`checkable`和`checked`属性非常有用，用于希望用户能够禁用表单的整个部分的情况（例如，如果与运输地址相同，则禁用订单表单的帐单地址部分）。

让我们重新配置我们的`groupbox`，如下所示：

```py
        groupbox = qtw.QGroupBox(
            'Buttons',
            checkable=True,
            checked=True,
            alignment=qtc.Qt.AlignHCenter,
            flat=True
        )
```

请注意，现在按钮可以通过简单的复选框切换禁用，并且框架的外观不同。

如果您只想要一个有边框的小部件，而没有标签或复选框功能，`QFrame`类可能是一个更好的选择。

# 验证小部件

尽管Qt提供了各种现成的输入小部件，例如日期和数字，但有时我们可能会发现需要一个具有非常特定约束的小部件。这些输入约束可以使用`QValidator`类创建。

工作流程如下：

1.  通过子类化`QtGui.QValidator`创建自定义验证器类

1.  用我们的验证逻辑覆盖`validate()`方法

1.  将我们自定义类的一个实例分配给小部件的`validator`属性

一旦分配给可编辑小部件，`validate()`方法将在用户更新小部件的值时被调用（例如，在`QLineEdit`中的每次按键），并确定输入是否被接受。

# 创建IPv4输入小部件

为了演示小部件验证，让我们创建一个验证**互联网协议版本4**（**IPv4**）地址的小部件。IPv4地址必须是4个整数，每个整数在`0`和`255`之间，并且每个数字之间有一个点。

让我们首先创建我们的验证器类。在`MainWindow`类之前添加这个类：

```py
class IPv4Validator(qtg.QValidator):
    """Enforce entry of IPv4 Addresses"""
```

接下来，我们需要重写这个类的`validate()`方法。`validate()`接收两个信息：一个包含建议输入的字符串和输入发生的索引。它将返回一个指示输入是`可接受`、`中间`还是`无效`的值。如果输入是可接受或中间的，它将被接受。如果无效，它将被拒绝。

用于指示输入状态的值是`QtValidator.Acceptable`、`QtValidator.Intermediate`或`QtValidator.Invalid`。

在Qt文档中，我们被告知验证器类应该只返回状态常量。然而，在PyQt中，实际上需要返回一个包含状态、字符串和位置的元组。不幸的是，这似乎没有很好的记录，如果您忘记了这一点，错误就不直观。

让我们开始构建我们的IPv4验证逻辑如下：

1.  在点字符上拆分字符串：

```py
            def validate(self, string, index):
                octets = string.split('.')
```

1.  如果有超过`4`个段，该值无效：

```py
            if len(octets) > 4:
                state = qtg.QValidator.Invalid
```

1.  如果任何填充的段不是数字字符串，则该值无效：

```py
            elif not all([x.isdigit() for x in octets if x != '']):
                state = qtg.QValidator.Invalid
```

1.  如果不是每个填充的段都可以转换为0到255之间的整数，则该值无效：

```py
            elif not all([0 <= int(x) <= 255 for x in octets if x != '']):
                state = qtg.QValidator.Invalid
```

1.  如果我们已经进行了这些检查，该值要么是中间的，要么是有效的。如果段少于四个，它是中间的：

```py
            elif len(octets) < 4:
                state = qtg.QValidator.Intermediate
```

1.  如果有任何空段，该值是中间的：

```py
            elif any([x == '' for x in octets]):
                state = qtg.QValidator.Intermediate
```

1.  如果值通过了所有这些测试，它是可接受的。我们可以返回我们的元组：

```py
            else:
                state = qtg.QValidator.Acceptable
            return (state, string, index)
```

要使用此验证器，我们只需要创建一个实例并将其分配给一个小部件：

```py
        # set the default text to a valid value
        line_edit.setText('0.0.0.0')
        line_edit.setValidator(IPv4Validator())
```

如果您现在运行演示，您会看到行编辑现在限制您输入有效的IPv4地址。

# 使用QSpinBox进行离散值

正如您在*创建基本QtWidgets小部件*部分中学到的，`QSpinBox`可以用于离散的字符串值列表，就像组合框一样。`QSpinBox`有一个内置的`validate()`方法，它的工作方式就像`QValidator`类的方法一样，用于限制小部件的输入。要使旋转框使用离散字符串列表，我们需要对`QSpinBox`进行子类化，并覆盖`validate()`和另外两个方法，`valueFromText()`和`textFromValue()`。

让我们创建一个自定义的旋转框类，用于从列表中选择项目；在`MainWindow`类之前，输入以下内容：

```py
class ChoiceSpinBox(qtw.QSpinBox):
    """A spinbox for selecting choices."""

    def __init__(self, choices, *args, **kwargs):
        self.choices = choices
        super().__init__(
            *args,
            maximum=len(self.choices) - 1,
            minimum=0,
            **kwargs
        )
```

我们正在对`qtw.QSpinBox`进行子类化，并覆盖构造函数，以便我们可以传入一个选择列表或元组，将其存储为`self.choices`。然后我们调用`QSpinBox`构造函数；请注意，我们设置了`maximum`和`minimum`，以便它们不能设置在我们选择的范围之外。我们还传递了任何额外的位置或关键字参数，以便我们可以利用所有其他`QSpinBox`属性设置。

接下来，让我们重新实现`valueFromText()`，如下所示：

```py
    def valueFromText(self, text):
        return self.choices.index(text)
```

这个方法的目的是能够返回一个整数索引值，给定一个与显示的选择项匹配的字符串。我们只是返回传入的任何字符串的列表索引。

接下来，我们需要重新实现补充方法`textFromValue()`：

```py
    def textFromValue(self, value):
        try:
            return self.choices[value]
        except IndexError:
            return '!Error!'
```

这个方法的目的是将整数索引值转换为匹配选择的文本。在这种情况下，我们只是返回给定索引处的字符串。如果以某种方式小部件传递了超出范围的值，我们将返回`!Error!`作为字符串。由于此方法用于确定在设置特定值时框中显示的内容，如果以某种方式值超出范围，这将清楚地显示错误条件。

最后，我们需要处理`validate()`。就像我们的`QValidator`类一样，我们需要创建一个方法，该方法接受建议的输入和编辑索引，并返回一个包含验证状态、字符串值和索引的元组。

我们将像这样编写它：

```py
    def validate(self, string, index):
        if string in self.choices:
            state = qtg.QValidator.Acceptable
        elif any([v.startswith(string) for v in self.choices]):
            state = qtg.QValidator.Intermediate
        else:
            state = qtg.QValidator.Invalid
        return (state, string, index)
```

在我们的方法中，如果输入字符串在`self.choices`中找到，我们将返回`Acceptable`，如果任何选择项以输入字符串开头（包括空字符串），我们将返回`Intermediate`，在任何其他情况下我们将返回`Invalid`。

有了这个类创建，我们可以在我们的`MainWindow`类中创建一个小部件：

```py
        ratingbox = ChoiceSpinBox(
            ['bad', 'average', 'good', 'awesome'],
            self
        )
        sublayout.addWidget(ratingbox)
```

`QComboBox`对象和具有文本选项的`QSpinBox`对象之间的一个重要区别是，旋转框项目缺少`data`属性。只能返回文本或索引。最适合用于诸如月份、星期几或其他可转换为整数值的顺序列表。

# 构建一个日历应用程序GUI

现在是时候将我们所学到的知识付诸实践，实际构建一个简单的功能性GUI。我们的目标是构建一个简单的日历应用程序，看起来像这样：

![](assets/e7ff923c-5442-4c41-ba36-c81f7435b740.png)

我们的界面还不能正常工作；现在，我们只关注如何创建和布局组件，就像屏幕截图中显示的那样。我们将以两种方式实现这一点：一次只使用代码，第二次使用Qt Designer。

这两种方法都是有效的，而且都可以正常工作，尽管您会看到，每种方法都有优点和缺点。

# 在代码中构建GUI

通过复制[第1章](bce5f3b1-2979-4f78-817b-3986e7974725.xhtml)中的应用程序模板，创建一个名为`calendar_form.py`的新文件，*PyQt入门*。

然后我们将配置我们的主窗口；在`MainWindow`构造函数中，从这段代码开始：

```py
        self.setWindowTitle("My Calendar App")
        self.resize(800, 600)
```

这段代码将设置我们窗口的标题为适当的内容，并设置窗口的固定大小为800 x 600。请注意，这只是初始大小，用户可以调整窗体的大小。

# 创建小部件

现在，让我们创建所有的小部件：

```py
        self.calendar = qtw.QCalendarWidget()
        self.event_list = qtw.QListWidget()
        self.event_title = qtw.QLineEdit()
        self.event_category = qtw.QComboBox()
        self.event_time = qtw.QTimeEdit(qtc.QTime(8, 0))
        self.allday_check = qtw.QCheckBox('All Day')
        self.event_detail = qtw.QTextEdit()
        self.add_button = qtw.QPushButton('Add/Update')
        self.del_button = qtw.QPushButton('Delete')
```

这些都是我们在GUI中将要使用的所有小部件。其中大部分我们已经介绍过了，但有两个新的：`QCalendarWidget`和`QListWidget`。

`QCalendarWidget`正是您所期望的：一个完全交互式的日历，可用于查看和选择日期。虽然它有许多可以配置的属性，但对于我们的需求，默认配置就可以了。我们将使用它来允许用户选择要查看和编辑的日期。

`QListWidget`用于显示、选择和编辑列表中的项目。我们将使用它来显示保存在特定日期的事件列表。

在我们继续之前，我们需要使用一些项目配置我们的`event_category`组合框以进行选择。以下是此框的计划：

+   当没有选择时，将其读为“选择类别…”作为占位符

+   包括一个名为`New…`的选项，也许允许用户输入新类别。

+   默认情况下包括一些常见类别，例如`工作`、`会议`和`医生`

为此，请添加以下内容：

```py
        # Add event categories
        self.event_category.addItems(
            ['Select category…', 'New…', 'Work',
             'Meeting', 'Doctor', 'Family']
            )
        # disable the first category item
        self.event_category.model().item(0).setEnabled(False)
```

`QComboBox`实际上没有占位符文本，因此我们在这里使用了一个技巧来模拟它。我们像往常一样使用`addItems()`方法添加了我们的组合框项目。接下来，我们使用`model()`方法检索其数据模型，该方法返回一个`QStandardItemModel`实例。数据模型保存组合框中所有项目的列表。我们可以使用模型的`item()`方法来访问给定索引（在本例中为`0`）处的实际数据项，并使用其`setEnabled()`方法来禁用它。

简而言之，我们通过禁用组合框中的第一个条目来模拟占位符文本。

我们将在[第5章](61ff4931-02af-474a-996c-5da827e0684f.xhtml)中了解更多关于小部件数据模型的知识，*使用模型视图类创建数据接口*。

# 构建布局

我们的表单将需要一些嵌套布局才能将所有内容放置到正确的位置。让我们分解我们提议的设计，并确定如何创建此布局：

+   应用程序分为左侧的日历和右侧的表单。这表明主要布局使用`QHBoxLayout`。

+   右侧的表单是一个垂直堆叠的组件，表明我们应该使用`QVBoxLayout`在右侧排列事物。

+   右下角的事件表单可以大致布局在网格中，因此我们可以在那里使用`QGridLayout`。

我们将首先创建主布局，然后添加日历：

```py
        main_layout = qtw.QHBoxLayout()
        self.setLayout(main_layout)
        main_layout.addWidget(self.calendar)
```

我们希望日历小部件填充布局中的任何额外空间，因此我们将根据需要设置其大小策略：

```py
        self.calendar.setSizePolicy(
            qtw.QSizePolicy.Expanding,
            qtw.QSizePolicy.Expanding
        )
```

现在，在右侧创建垂直布局，并添加标签和事件列表：

```py
        right_layout = qtw.QVBoxLayout()
        main_layout.addLayout(right_layout)
        right_layout.addWidget(qtw.QLabel('Events on Date'))
        right_layout.addWidget(self.event_list)
```

如果有更多的垂直空间，我们希望事件列表填满所有可用的空间。因此，让我们将其大小策略设置如下：

```py
        self.event_list.setSizePolicy(
            qtw.QSizePolicy.Expanding,
            qtw.QSizePolicy.Expanding
        )
```

GUI的下一部分是事件表单及其标签。我们可以在这里使用另一个标签，但设计建议这些表单字段在此标题下分组在一起，因此`QGroupBox`更合适。

因此，让我们创建一个带有`QGridLayout`的组框来容纳我们的事件表单：

```py
        event_form = qtw.QGroupBox('Event')
        right_layout.addWidget(event_form)
        event_form_layout = qtw.QGridLayout()
        event_form.setLayout(event_form_layout)
```

最后，我们需要将剩余的小部件添加到网格布局中：

```py
        event_form_layout.addWidget(self.event_title, 1, 1, 1, 3)
        event_form_layout.addWidget(self.event_category, 2, 1)
        event_form_layout.addWidget(self.event_time, 2, 2,)
        event_form_layout.addWidget(self.allday_check, 2, 3)
        event_form_layout.addWidget(self.event_detail, 3, 1, 1, 3)
        event_form_layout.addWidget(self.add_button, 4, 2)
        event_form_layout.addWidget(self.del_button, 4, 3)
```

我们将网格分为三列，并使用可选的列跨度参数将我们的标题和详细字段跨越所有三列。

现在我们完成了！此时，您可以运行脚本并查看您完成的表单。当然，它目前还没有做任何事情，但这是我们[第3章](dbb86a9b-0050-490e-94da-1f4661d8bc66.xhtml)的主题，*使用信号和槽处理事件*。

# 在Qt Designer中构建GUI

让我们尝试构建相同的GUI，但这次我们将使用Qt Designer构建它。

# 第一步

首先，按照[第1章](bce5f3b1-2979-4f78-817b-3986e7974725.xhtml)中描述的方式启动Qt Designer，然后基于小部件创建一个新表单，如下所示：

![](assets/eda1e482-5f33-4176-a699-f3da3c3f43a7.png)

现在，单击小部件，我们将使用右侧的属性面板配置其属性：

1.  将对象名称更改为`MainWindow`

1.  在**几何**下，将宽度更改为`800`，高度更改为`600`

1.  将窗口标题更改为`我的日历应用程序`

接下来，我们将开始添加小部件。在左侧的小部件框中滚动查找**日历小部件**，然后将其拖放到主窗口上。选择日历并编辑其属性：

1.  将名称更改为`calendar`

1.  将水平和垂直大小策略更改为`扩展`

要设置我们的主要布局，右键单击主窗口（不是日历），然后选择布局|**水平布局**。这将在主窗口小部件中添加一个`QHBoxLayout`。请注意，直到至少有一个小部件放在主窗口上，您才能这样做，这就是为什么我们首先添加了日历小部件。

# 构建右侧面板

现在，我们将为表单的右侧添加垂直布局。将一个垂直布局拖到日历小部件的右侧。然后将一个标签小部件拖到垂直布局中。确保标签在层次结构中列为垂直布局的子对象，而不是同级对象：

![](assets/ba44242d-5956-4cd8-9e1a-492662eec464.png)

如果您在将小部件拖放到未展开的布局上遇到问题，您也可以将其拖放到**对象检查器**面板中的层次结构中。

双击标签上的文本，将其更改为日期上的事件。

接下来，将一个列表小部件拖到垂直布局中，使其出现在标签下面。将其重命名为`event_list`，并检查其属性，确保其大小策略设置为`扩展`。

# 构建事件表单

在小部件框中找到组框，并将其拖到列表小部件下面。双击文本，并将其更改为`事件`。

将一个行编辑器拖到组框上，确保它显示为组框对象检查器中的子对象。将对象名称更改为`event_title`。

现在，右键单击组框，选择布局，然后选择**在网格中布局**。这将在组框中创建一个网格布局。

将一个组合框拖到下一行。将一个时间编辑器拖到其右侧，然后将一个复选框拖到其右侧。将它们分别命名为`event_category`，`event_time`和`allday_check`。双击复选框文本，并将其更改为`全天`。

要向组合框添加选项，右键单击框并选择**编辑项目**。这将打开一个对话框，我们可以在其中输入我们的项目，所以点击+按钮添加`选择类别…`，就像第一个一样，然后`新建…`，然后一些随机类别（如`工作`，`医生`，`会议`）。

不幸的是，我们无法在Qt Designer中禁用第一项。当我们在应用程序中使用我们的表单时，我们将在[第3章](dbb86a9b-0050-490e-94da-1f4661d8bc66.xhtml)中讨论如何处理这个问题，*使用信号和槽处理事件*。

注意，添加这三个小部件会将行编辑器推到右侧。我们需要修复该小部件的列跨度。单击行编辑器，抓住右边缘的手柄，将其向右拖动，直到它扩展到组框的宽度。

现在，抓住一个文本编辑器，将其拖到其他小部件下面。注意它被挤压到第一列，所以就像行编辑一样，将其向右拖动，直到填满整个宽度。将文本编辑器重命名为`event_detail`。

最后，将两个按钮小部件拖到表单底部。确保将它们拖到第二列和第三列，留下第一列为空。将它们重命名为`add_button`和`del_button`，将文本分别更改为`添加/更新`和`删除`。

# 预览表单

将表单保存为`calendar_form.ui`，然后按下*Ctrl* + *R*进行预览。您应该看到一个完全功能的表单，就像原始截图中显示的那样。要实际使用这个文件，我们需要将其转换为Python代码并将其导入到实际的脚本中。在我们对表单进行一些额外修改之后，我们将在[第3章](dbb86a9b-0050-490e-94da-1f4661d8bc66.xhtml)中进行讨论，*使用信号和槽处理事件*。

# 总结

在本章中，我们介绍了Qt中一些最受欢迎的小部件类。您学会了如何创建它们，自定义它们，并将它们添加到表单中。我们讨论了各种控制小部件大小的方法，并练习了在Python代码和Qt Designer所见即所得应用程序中构建简单应用程序表单的方法。

在下一章中，我们将学习如何使这个表单真正做一些事情，同时探索Qt的核心通信和事件处理系统。保持你的日历表单方便，因为我们将对它进行更多修改，并从中制作一个功能应用程序。

# 问题

尝试这些问题来测试你从本章学到的知识：

1.  你会如何创建一个全屏、没有窗口框架，并使用沙漏光标的`QWidget`？

1.  你被要求为计算机库存数据库设计一个数据输入表单。为以下字段选择最好的小部件使用：

+   **计算机制造商**：你公司购买的八个品牌之一

+   **处理器速度**：CPU速度，以GHz为单位

+   **内存量**：内存量，以MB为单位

+   **主机名**：计算机的主机名

+   **视频制作**：视频硬件是Nvidia、AMD还是Intel

+   **OEM许可证**：计算机是否使用原始设备制造商（OEM）许可证

1.  数据输入表单包括一个需要`XX-999-9999X`格式的`库存编号`字段，其中`X`是从`A`到`Z`的大写字母，不包括`O`和`I`，`9`是从`0`到`9`的数字。你能创建一个验证器类来验证这个输入吗？

1.  看看下面的计算器表单——可能使用了哪些布局来创建它？

![](assets/1b7c100d-6694-48e8-8bc0-e15dc8c0aba7.png)

1.  参考前面的计算器表单，当表单被调整大小时，你会如何使按钮网格占用任何额外的空间？

1.  计算器表单中最顶层的小部件是一个`QLCDNumber`小部件。你能找到关于这个小部件的Qt文档吗？它有哪些独特的属性？你什么时候会使用它？

1.  从你的模板代码开始，在代码中构建计算器表单。

1.  在Qt Designer中构建计算器表单。

# 进一步阅读

查看以下资源，了解本章涉及的主题的更多信息：

+   `QWidget`属性文档列出了所有`QWidget`的属性，这些属性被所有子类继承，网址为[https://doc.qt.io/qt-5/qwidget.html#properties](https://doc.qt.io/qt-5/qwidget.html#properties)

+   `Qt`命名空间文档列出了Qt中使用的许多全局枚举，网址为[https://doc.qt.io/qt-5/qt.html#WindowState-enum](https://doc.qt.io/qt-5/qt.html#WindowState-enum)

+   Qt布局管理教程提供了有关布局和大小调整的详细信息，网址为[https://doc.qt.io/qt-5/layout.html](https://doc.qt.io/qt-5/layout.html)

+   `QDateTime`文档提供了有关在Qt中处理日期和时间的更多信息，网址为[https://doc.qt.io/qt-5/qdatetime.html](https://doc.qt.io/qt-5/qdatetime.html)

+   有关`QCalendarWidget`的更多信息可以在[https://doc.qt.io/qt-5/qcalendarwidget.html](https://doc.qt.io/qt-5/qcalendarwidget.html)找到。
