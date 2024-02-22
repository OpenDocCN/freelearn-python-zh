# 第七章：样式化 Qt 应用程序

很容易欣赏到 Qt 默认提供的清晰、本地外观。但对于不那么商业化的应用程序，普通的灰色小部件和标准字体并不总是设置正确的语气。即使是最沉闷的实用程序或数据输入应用程序偶尔也会受益于添加图标或谨慎调整字体以增强可用性。幸运的是，Qt 的灵活性使我们能够自己控制应用程序的外观和感觉。

在本章中，我们将涵盖以下主题：

+   使用字体、图像和图标

+   配置颜色、样式表和样式

+   创建动画

# 技术要求

在本章中，您将需要第一章中列出的所有要求，*PyQt 入门*，以及第四章中的 Qt 应用程序模板，*使用 QMainWindow 构建应用程序*。

此外，您可能需要 PNG、JPEG 或 GIF 图像文件来使用；您可以使用示例代码中包含的这些文件：[`github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter06`](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter06)。

查看以下视频，了解代码的运行情况：[`bit.ly/2M5OJj6`](http://bit.ly/2M5OJj6)

# 使用字体、图像和图标

我们将通过自定义应用程序的字体、显示一些静态图像和包含动态图标来开始样式化我们的 Qt 应用程序。但在此之前，我们需要创建一个**图形用户界面**（**GUI**），以便我们可以使用。我们将创建一个游戏大厅对话框，该对话框将用于登录到一个名为**Fight Fighter**的虚构多人游戏。

要做到这一点，打开应用程序模板的新副本，并将以下 GUI 代码添加到`MainWindow.__init__()`中：

```py
        self.setWindowTitle('Fight Fighter Game Lobby')
        cx_form = qtw.QWidget()
        self.setCentralWidget(cx_form)
        cx_form.setLayout(qtw.QFormLayout())
        heading = qtw.QLabel("Fight Fighter!")
        cx_form.layout().addRow(heading)

        inputs = {
            'Server': qtw.QLineEdit(),
            'Name': qtw.QLineEdit(),
            'Password': qtw.QLineEdit(
                echoMode=qtw.QLineEdit.Password),
            'Team': qtw.QComboBox(),
            'Ready': qtw.QCheckBox('Check when ready')
        }
        teams = ('Crimson Sharks', 'Shadow Hawks',
                  'Night Terrors', 'Blue Crew')
        inputs['Team'].addItems(teams)
        for label, widget in inputs.items():
            cx_form.layout().addRow(label, widget)
        self.submit = qtw.QPushButton(
            'Connect',
            clicked=lambda: qtw.QMessageBox.information(
                None, 'Connecting', 'Prepare for Battle!'))
        self.reset = qtw.QPushButton('Cancel', clicked=self.close)
        cx_form.layout().addRow(self.submit, self.reset)
```

这是相当标准的 Qt GUI 代码，您现在应该对此很熟悉；我们通过将输入放入`dict`对象中并在循环中将它们添加到布局中，节省了一些代码行，但除此之外，它相对直接。根据您的操作系统和主题设置，对话框框可能看起来像以下截图：

![](img/ce8de21f-49a5-46c4-aab8-3f45ba3b8c26.png)

正如您所看到的，这是一个不错的表单，但有点单调。因此，让我们探讨一下是否可以改进样式。

# 设置字体

我们要解决的第一件事是字体。每个`QWidget`类都有一个`font`属性，我们可以在构造函数中设置，也可以使用`setFont()`访问器来设置。`font`的值必须是一个`QtGui.QFont`对象。

以下是您可以创建和使用`QFont`对象的方法：

```py
        heading_font = qtg.QFont('Impact', 32, qtg.QFont.Bold)
        heading_font.setStretch(qtg.QFont.ExtraExpanded)
        heading.setFont(heading_font)
```

`QFont`对象包含描述文本将如何绘制到屏幕上的所有属性。构造函数可以接受以下任何参数：

+   一个表示字体系列的字符串

+   一个浮点数或整数，表示点大小

+   一个`QtGui.QFont.FontWeight`常量，指示权重

+   一个布尔值，指示字体是否应该是斜体

字体的其余方面，如`stretch`属性，可以使用关键字参数或访问器方法进行配置。我们还可以创建一个没有参数的`QFont`对象，并按照以下方式进行程序化配置：

```py
        label_font = qtg.QFont()
        label_font.setFamily('Impact')
        label_font.setPointSize(14)
        label_font.setWeight(qtg.QFont.DemiBold)
        label_font.setStyle(qtg.QFont.StyleItalic)

        for inp in inputs.values():
            cx_form.layout().labelForField(inp).setFont(label_font)
```

在小部件上设置字体不仅会影响该小部件，还会影响所有子小部件。因此，我们可以通过在`cx_form`上设置字体而不是在单个小部件上设置字体来为整个表单配置字体。

# 处理缺失的字体

现在，如果所有平台和**操作系统**（**OSes**）都提供了无限数量的同名字体，那么您需要了解的就是`QFont`。不幸的是，情况并非如此。大多数系统只提供了少数内置字体，并且这些字体中只有少数是跨平台的，甚至是平台的不同版本通用的。因此，Qt 有一个处理缺失字体的回退机制。

例如，假设我们要求 Qt 使用一个不存在的字体系列，如下所示：

```py
        button_font = qtg.QFont(
            'Totally Nonexistant Font Family XYZ', 15.233)
```

Qt 不会在此调用时抛出错误，甚至不会注册警告。相反，在未找到请求的字体系列后，它将回退到其`defaultFamily`属性，该属性利用了操作系统或桌面环境中设置的默认字体。

`QFont`对象实际上不会告诉我们发生了什么；如果查询它以获取信息，它只会告诉您已配置了什么：

```py
        print(f'Font is {button_font.family()}')
        # Prints: "Font is Totally Nonexistent Font Family XYZ"
```

要发现实际使用的字体设置，我们需要将我们的`QFont`对象传递给`QFontInfo`对象：

```py
        actual_font = qtg.QFontInfo(button_font).family()
        print(f'Actual font used is {actual_font}')
```

如果运行脚本，您会看到，很可能实际上使用的是默认的屏幕字体：

```py
$ python game_lobby.py
Font is Totally Nonexistent Font Family XYZ
Actual font used is Bitstream Vera Sans
```

虽然这确保了用户不会在窗口中没有任何文本，但如果我们能让 Qt 更好地了解应该使用什么样的字体，那就更好了。

我们可以通过设置字体的`styleHint`和`styleStrategy`属性来实现这一点，如下所示：

```py
        button_font.setStyleHint(qtg.QFont.Fantasy)
        button_font.setStyleStrategy(
            qtg.QFont.PreferAntialias |
            qtg.QFont.PreferQuality
        )
```

`styleHint`建议 Qt 回退到的一般类别，在本例中是`Fantasy`类别。这里的其他选项包括`SansSerif`、`Serif`、`TypeWriter`、`Decorative`、`Monospace`和`Cursive`。这些选项对应的内容取决于操作系统和桌面环境的配置。

`styleStrategy`属性告诉 Qt 与所选字体的能力相关的更多技术偏好，比如抗锯齿、OpenGL 兼容性，以及大小是精确匹配还是四舍五入到最接近的非缩放大小。策略选项的完整列表可以在[`doc.qt.io/qt-5/qfont.html#StyleStrategy-enum`](https://doc.qt.io/qt-5/qfont.html#StyleStrategy-enum)找到。

设置这些属性后，再次检查字体，看看是否有什么变化：

```py
        actual_font = qtg.QFontInfo(button_font)
        print(f'Actual font used is {actual_font.family()}'
              f' {actual_font.pointSize()}')
        self.submit.setFont(button_font)
        self.cancel.setFont(button_font)
```

根据系统的配置，您应该看到与之前不同的结果：

```py
$ python game_lobby.py
Actual font used is Impact 15
```

在这个系统上，`Fantasy`被解释为`Impact`，而`PreferQuality`策略标志强制最初奇怪的 15.233 点大小成为一个漂亮的`15`。

此时，根据系统上可用的字体，您的应用程序应该如下所示：

![](img/0ac336f7-5ade-4387-a6c5-48a27c8fd1b1.png)

字体也可以与应用程序捆绑在一起；请参阅本章中的*使用 Qt 资源文件*部分。

# 添加图像

Qt 提供了许多与应用程序中使用图像相关的类，但是，对于在 GUI 中简单显示图片，最合适的是`QPixmap`。`QPixmap`是一个经过优化的显示图像类，可以加载许多常见的图像格式，包括 PNG、BMP、GIF 和 JPEG。

要创建一个，我们只需要将`QPixmap`传递给图像文件的路径：

```py
        logo = qtg.QPixmap('logo.png')
```

一旦加载，`QPixmap`对象可以显示在`QLabel`或`QButton`对象中，如下所示：

```py
        heading.setPixmap(logo)
```

请注意，标签只能显示字符串或像素图，但不能同时显示两者。

为了优化显示，`QPixmap`对象只提供了最小的编辑功能；但是，我们可以进行简单的转换，比如缩放：

```py
        if logo.width() > 400:
            logo = logo.scaledToWidth(
                400, qtc.Qt.SmoothTransformation)
```

在这个例子中，我们使用了像素图的`scaledToWidth()`方法，使用平滑的转换算法将标志的宽度限制为`400`像素。

`QPixmap`对象如此有限的原因是它们实际上存储在显示服务器的内存中。`QImage`类似，但是它将数据存储在应用程序内存中，因此可以进行更广泛的编辑。我们将在第十二章中更多地探讨这个类，创建*使用 QPainter 进行 2D 图形*。

`QPixmap`还提供了一个方便的功能，可以生成简单的彩色矩形，如下所示：

```py
        go_pixmap = qtg.QPixmap(qtc.QSize(32, 32))
        stop_pixmap = qtg.QPixmap(qtc.QSize(32, 32))
        go_pixmap.fill(qtg.QColor('green'))
        stop_pixmap.fill(qtg.QColor('red'))
```

通过在构造函数中指定大小并使用`fill()`方法，我们可以创建一个简单的彩色矩形像素图。这对于显示颜色样本或用作快速的图像替身非常有用。

# 使用图标

现在考虑工具栏或程序菜单中的图标。当菜单项被禁用时，您期望图标以某种方式变灰。同样，如果用户使用鼠标指针悬停在按钮或项目上，您可能期望它被突出显示。为了封装这种状态相关的图像显示，Qt 提供了`QIcon`类。`QIcon`对象包含一组与小部件状态相映射的像素图。

以下是如何创建一个`QIcon`对象：

```py
        connect_icon = qtg.QIcon()
        connect_icon.addPixmap(go_pixmap, qtg.QIcon.Active)
        connect_icon.addPixmap(stop_pixmap, qtg.QIcon.Disabled)
```

创建图标对象后，我们使用它的`addPixmap()`方法将一个`QPixmap`对象分配给小部件状态。这些状态包括`Normal`、`Active`、`Disabled`和`Selected`。

当禁用时，`connect_icon`图标现在将是一个红色的正方形，或者当启用时将是一个绿色的正方形。让我们将其添加到我们的提交按钮，并添加一些逻辑来切换按钮的状态：

```py
        self.submit.setIcon(connect_icon)
        self.submit.setDisabled(True)
        inputs['Server'].textChanged.connect(
            lambda x: self.submit.setDisabled(x == '')
        )
```

如果您在此时运行脚本，您会看到红色的正方形出现在提交按钮上，直到“服务器”字段包含数据为止，此时它会自动切换为绿色。请注意，我们不必告诉图标对象本身切换状态；一旦分配给小部件，它就会跟踪小部件状态的任何更改。

图标可以与`QPushButton`、`QToolButton`和`QAction`对象一起使用；`QComboBox`、`QListView`、`QTableView`和`QTreeView`项目；以及大多数其他您可能合理期望有图标的地方。

# 使用 Qt 资源文件

在程序中使用图像文件的一个重要问题是确保程序可以在运行时找到它们。传递给`QPixmap`构造函数或`QIcon`构造函数的路径被解释为绝对路径（即，如果它们以驱动器号或路径分隔符开头），或者相对于当前工作目录（您无法控制）。例如，尝试从代码目录之外的某个地方运行您的脚本：

```py
$ cd ..
$ python ch05/game_lobby.py
```

您会发现您的图像都丢失了！当`QPixmap`找不到文件时不会抱怨，它只是不显示任何东西。如果没有图像的绝对路径，您只能在脚本从相对路径相关的确切目录运行时找到它们。

不幸的是，指定绝对路径意味着您的程序只能从文件系统上的一个位置工作，这对于您计划将其分发到多个平台是一个重大问题。

PyQt 为我们提供了一个解决这个问题的解决方案，即**PyQt 资源文件**，我们可以使用**PyQt 资源编译器**工具创建。基本过程如下：

1.  编写一个 XML 格式的**Qt 资源集合**文件（.qrc），其中包含我们要包括的所有文件的路径

1.  运行`pyrcc5`工具将这些文件序列化并压缩到包含在 Python 模块中的数据中

1.  将生成的 Python 模块导入我们的应用程序脚本

1.  现在我们可以使用特殊的语法引用我们的资源

让我们逐步走过这个过程——假设我们有一些队徽，以 PNG 文件的形式，我们想要包含在我们的程序中。我们的第一步是创建`resources.qrc`文件，它看起来像下面的代码块：

```py
<RCC>
  <qresource prefix="teams">
    <file>crimson_sharks.png</file>
    <file>shadow_hawks.png</file>
    <file>night_terrors.png</file>
    <file alias="blue_crew.png">blue_crew2.png</file>
  </qresource>
</RCC>
```

我们已经将这个文件放在与脚本中列出的图像文件相同的目录中。请注意，我们添加了一个`prefix`值为`teams`。前缀允许您将资源组织成类别。另外，请注意，最后一个文件有一个指定的别名。在我们的程序中，我们可以使用这个别名而不是文件的实际名称来访问这个资源。

现在，在命令行中，我们将运行`pyrcc5`，如下所示：

```py
$ pyrcc5 -o resources.py resources.qrc
```

这里的语法是`pyrcc5 -o outputFile.py inputFile.qrc`。这个命令应该生成一个包含您的资源数据的 Python 文件。如果您花一点时间打开文件并检查它，您会发现它主要只是一个分配给`qt_resource_data`变量的大型`bytes`对象。

回到我们的主要脚本中，我们只需要像导入任何其他 Python 文件一样导入这个文件：

```py
import resources
```

文件不一定要叫做`resources.py`；实际上，任何名称都可以。你只需要导入它，文件中的代码将确保资源对 Qt 可用。

现在资源文件已导入，我们可以使用资源语法指定像素图路径：

```py
        inputs['Team'].setItemIcon(
            0, qtg.QIcon(':/teams/crimson_sharks.png'))
        inputs['Team'].setItemIcon(
            1, qtg.QIcon(':/teams/shadow_hawks.png'))
        inputs['Team'].setItemIcon(
            2, qtg.QIcon(':/teams/night_terrors.png'))
        inputs['Team'].setItemIcon(
            3, qtg.QIcon(':/teams/blue_crew.png'))
```

基本上，语法是`:/prefix/file_name_or_alias.extension`。

因为我们的数据存储在一个 Python 文件中，我们可以将它放在一个 Python 库中，它将使用 Python 的标准导入解析规则来定位文件。

# Qt 资源文件和字体

资源文件不仅限于图像；实际上，它们可以用于包含几乎任何类型的二进制文件，包括字体文件。例如，假设我们想要在程序中包含我们喜欢的字体，以确保它在所有平台上看起来正确。

与图像一样，我们首先在`.qrc`文件中包含字体文件：

```py
<RCC>
  <qresource prefix="teams">
    <file>crimson_sharks.png</file>
    <file>shadow_hawks.png</file>
    <file>night_terrors.png</file>
    <file>blue_crew.png</file>
  </qresource>
  <qresource prefix="fonts">
    <file>LiberationSans-Regular.ttf</file>
  </qresource>
</RCC>
```

在这里，我们添加了一个前缀`fonts`并包含了对`LiberationSans-Regular.ttf`文件的引用。运行`pyrcc5`对这个文件进行处理后，字体被捆绑到我们的`resources.py`文件中。

要在代码中使用这个字体，我们首先要将它添加到字体数据库中，如下所示：

```py
        libsans_id = qtg.QFontDatabase.addApplicationFont(
            ':/fonts/LiberationSans-Regular.ttf')
```

`QFontDatabase.addApplicationFont()`将传递的字体文件插入应用程序的字体数据库并返回一个 ID 号。然后我们可以使用该 ID 号来确定字体的系列字符串；这可以传递给`QFont`，如下所示：

```py
        family = qtg.QFontDatabase.applicationFontFamilies(libsans_id)[0]
        libsans = qtg.QFont(family)
        inputs['Team'].setFont(libsans)
```

在分发应用程序之前，请确保检查字体的许可证！请记住，并非所有字体都可以自由分发。

我们的表单现在看起来更像游戏了；运行应用程序，它应该看起来类似于以下截图：

![](img/93a4b1be-2c07-49b2-9a87-7d2b740a59fc.png)

# 配置颜色、样式表和样式

字体和图标改善了我们表单的外观，但现在是时候摆脱那些机构灰色调，用一些颜色来替换它们。在本节中，我们将看一下 Qt 为自定义应用程序颜色提供的三种不同方法：操纵**调色板**、使用**样式表**和覆盖**应用程序样式**。

# 使用调色板自定义颜色

由`QPalette`类表示的调色板是一组映射到颜色角色和颜色组的颜色和画笔的集合。

让我们解开这个声明：

+   在这里，**color**是一个文字颜色值，由`QColor`对象表示

+   **画笔**将特定颜色与样式（如图案、渐变或纹理）结合在一起，由`QBrush`类表示

+   **颜色角色**表示小部件使用颜色的方式，例如前景、背景或边框

+   **颜色组**指的是小部件的交互状态；它可以是`Normal`、`Active`、`Disabled`或`Inactive`

当小部件在屏幕上绘制时，Qt 的绘图系统会查阅调色板，以确定用于渲染小部件的每个部分的颜色和画笔。要自定义这一点，我们可以创建自己的调色板并将其分配给一个小部件。

首先，我们需要获取一个`QPalette`对象，如下所示：

```py
        app = qtw.QApplication.instance()
        palette = app.palette()
```

虽然我们可以直接创建一个`QPalette`对象，但 Qt 文档建议我们在运行的`QApplication`实例上调用`palette()`来检索当前配置样式的调色板的副本。

您可以通过调用`QApplication.instance()`来随时检索`QApplication`对象的副本。

现在我们有了调色板，让我们开始覆盖一些规则：

```py
        palette.setColor(
            qtg.QPalette.Button,
            qtg.QColor('#333')
        )
        palette.setColor(
            qtg.QPalette.ButtonText,
            qtg.QColor('#3F3')
        )
```

`QtGui.QPalette.Button`和`QtGui.QPalette.ButtonText`是颜色角色常量，正如你可能猜到的那样，它们分别代表所有 Qt 按钮类的背景和前景颜色。我们正在用新颜色覆盖它们。

要覆盖特定按钮状态的颜色，我们需要将颜色组常量作为第一个参数传递：

```py
        palette.setColor(
            qtg.QPalette.Disabled,
            qtg.QPalette.ButtonText,
            qtg.QColor('#F88')
        )
        palette.setColor(
            qtg.QPalette.Disabled,
            qtg.QPalette.Button,
            qtg.QColor('#888')
        )
```

在这种情况下，我们正在更改按钮处于`Disabled`状态时使用的颜色。

要应用这个新的调色板，我们必须将它分配给一个小部件，如下所示：

```py
        self.submit.setPalette(palette)
        self.cancel.setPalette(palette)
```

`setPalette()`将提供的调色板分配给小部件和所有子小部件。因此，我们可以创建一个单独的调色板，并将其分配给我们的`QMainWindow`类，以将其应用于所有对象，而不是分配给单个小部件。

# 使用 QBrush 对象

如果我们想要比纯色更花哨的东西，那么我们可以使用`QBrush`对象。画笔可以填充颜色、图案、渐变或纹理（即基于图像的图案）。

例如，让我们创建一个绘制白色点划填充的画笔：

```py
        dotted_brush = qtg.QBrush(
            qtg.QColor('white'), qtc.Qt.Dense2Pattern)
```

`Dense2Pattern`是 15 种可用图案之一。（你可以参考[`doc.qt.io/qt-5/qt.html#BrushStyle-enum`](https://doc.qt.io/qt-5/qt.html#BrushStyle-enum)获取完整列表。）其中大多数是不同程度的点划、交叉点划或交替线条图案。

图案有它们的用途，但基于渐变的画笔可能更适合现代风格。然而，创建一个可能会更复杂，如下面的代码所示：

```py
        gradient = qtg.QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, qtg.QColor('navy'))
        gradient.setColorAt(0.5, qtg.QColor('darkred'))
        gradient.setColorAt(1, qtg.QColor('orange'))
        gradient_brush = qtg.QBrush(gradient)
```

要在画笔中使用渐变，我们首先必须创建一个渐变对象。在这里，我们创建了一个`QLinearGradient`对象，它实现了基本的线性渐变。参数是渐变的起始和结束坐标，我们指定为主窗口的左上角（0, 0）和右下角（宽度，高度）。

Qt 还提供了`QRadialGradient`和`QConicalGradient`类，用于提供额外的渐变选项。

创建对象后，我们使用`setColorAt()`指定颜色停止。第一个参数是 0 到 1 之间的浮点值，指定起始和结束之间的百分比，第二个参数是渐变应该在该点的`QColor`对象。

创建渐变后，我们将其传递给`QBrush`构造函数，以创建一个使用我们的渐变进行绘制的画笔。

我们现在可以使用`setBrush()`方法将我们的画笔应用于调色板，如下所示：

```py
        window_palette = app.palette()
        window_palette.setBrush(
            qtg.QPalette.Window,
            gradient_brush
        )
        window_palette.setBrush(
            qtg.QPalette.Active,
            qtg.QPalette.WindowText,
            dotted_brush
        )
        self.setPalette(window_palette)
```

就像`QPalette.setColor()`一样，我们可以分配我们的画笔，无论是否指定了特定的颜色组。在这种情况下，我们的渐变画笔将用于绘制主窗口，而我们的点画画笔只有在小部件处于活动状态时才会使用（即当前活动窗口）。

# 使用 Qt 样式表（QSS）自定义外观

对于已经使用过 Web 技术的开发人员来说，使用调色板、画笔和颜色对象来设计应用程序可能会显得啰嗦和不直观。幸运的是，Qt 为您提供了一种称为 QSS 的替代方案，它与 Web 开发中使用的**层叠样式表**（**CSS**）非常相似。这是一种简单的方法，可以对我们的小部件进行一些简单的更改。

您可以按照以下方式使用 QSS：

```py
        stylesheet = """
        QMainWindow {
            background-color: black;
        }
        QWidget {
            background-color: transparent;
            color: #3F3;
        }
        QLineEdit, QComboBox, QCheckBox {
            font-size: 16pt;
        }"""
        self.setStyleSheet(stylesheet)
```

在这里，样式表只是一个包含样式指令的字符串，我们可以将其分配给小部件的`styleSheet`属性。

这个语法对于任何使用过 CSS 的人来说应该很熟悉，如下所示：

```py
WidgetClass {
    property-name: value;
    property-name2: value2;
}
```

如果此时运行程序，你会发现（取决于你的系统主题），它可能看起来像以下的截图：

![](img/5aa10114-bea6-4980-b9a0-6d011e653a36.png)

在这里，界面大部分变成了黑色，除了文本和图像。特别是我们的按钮和复选框与背景几乎无法区分。那么，为什么会发生这种情况呢？

当您向小部件类添加 QSS 样式时，样式更改会传递到所有其子类。由于我们对`QWidget`进行了样式设置，所有其他`QWidget`派生类（如`QCheckbox`和`QPushButton`）都继承了这种样式。

让我们通过覆盖这些子类的样式来修复这个问题，如下所示：

```py
        stylesheet += """
        QPushButton {
            background-color: #333;
        }
        QCheckBox::indicator:unchecked {
            border: 1px solid silver;
            background-color: darkred;
        }
        QCheckBox::indicator:checked {
            border: 1px solid silver;
            background-color: #3F3;
        }
        """
        self.setStyleSheet(stylesheet)
```

就像 CSS 一样，将样式应用于更具体的类会覆盖更一般的情况。例如，我们的`QPushButton`背景颜色会覆盖`QWidget`背景颜色。

请注意在`QCheckBox`中使用冒号 - QSS 中的双冒号允许我们引用小部件的子元素。在这种情况下，这是`QCheckBox`类的指示器部分（而不是其标签部分）。我们还可以使用单个冒号来引用小部件状态，就像在这种情况下，我们根据复选框是否选中或未选中来设置不同的样式。

如果您只想将更改限制为特定类，而不是其任何子类，只需在名称后添加一个句点（`。`），如下所示：

```py
        stylesheet += """
        .QWidget {
           background: url(tile.png);
        }
        """
```

前面的示例还演示了如何在 QSS 中使用图像。就像在 CSS 中一样，我们可以提供一个包装在`url()`函数中的文件路径。

如果您已经使用`pyrcc5`序列化了图像，QSS 还接受资源路径。

如果要将样式应用于特定小部件而不是整个小部件类，有两种方法可以实现。

第一种方法是依赖于`objectName`属性，如下所示：

```py
        self.submit.setObjectName('SubmitButton')
        stylesheet += """
        #SubmitButton:disabled {
            background-color: #888;
            color: darkred;
        }
        """
```

在我们的样式表中，对象名称前必须加上一个

`#`符号用于将其标识为对象名称，而不是类。

在单个小部件上设置样式的另一种方法是调用 t

使用小部件的`setStyleSheet()`方法和一些样式表指令，如下所示：

```py
        for inp in ('Server', 'Name', 'Password'):
            inp_widget = inputs[inp]
            inp_widget.setStyleSheet('background-color: black')
```

如果我们要直接将样式应用于我们正在调用的小部件，我们不需要指定类名或对象名；我们可以简单地传递属性和值。

经过所有这些更改，我们的应用程序现在看起来更像是一个游戏 GUI：

![](img/ec30b7cf-f46b-4f79-955c-9af210c2281f.png)

# QSS 的缺点

正如您所看到的，QSS 是一种非常强大的样式方法，对于任何曾经从事 Web 开发的开发人员来说都是可访问的；但是，它确实有一些缺点。

QSS 是对调色板和样式对象的抽象，必须转换为实际系统。这使它们在大型应用程序中变得更慢，这也意味着没有默认样式表可以检索和编辑 - 每次都是从头开始。

正如我们已经看到的，当应用于高级小部件时，QSS 可能会产生不可预测的结果，因为它通过类层次结构继承。

最后，请记住，QSS 是 CSS 2.0 的一个较小子集，带有一些添加或更改 - 它不是 CSS。因此，过渡、动画、flexbox 容器、相对单位和其他现代 CSS 好东西完全不存在。因此，尽管 Web 开发人员可能会发现其基本语法很熟悉，但有限的选项集可能会令人沮丧，其不同的行为也会令人困惑。

# 使用 QStyle 自定义外观

调色板和样式表可以帮助我们大大定制 Qt 应用程序的外观，对于大多数情况来说，这就是您所需要的。要真正深入了解 Qt 应用程序外观的核心，我们需要了解样式系统。

每个运行的 Qt 应用程序实例都有一个样式，负责告诉图形系统如何绘制每个小部件或 GUI 组件。样式是动态和可插拔的，因此不同的 OS 平台具有不同的样式，用户可以安装自己的 Qt 样式以在 Qt 应用程序中使用。这就是 Qt 应用程序能够在不同的操作系统上具有本机外观的原因。

在第一章中，*使用 PyQt 入门*，我们学到`QApplication`在创建时应传递`sys.argv`的副本，以便它可以处理一些特定于 Qt 的参数。其中一个参数是`-style`，它允许用户为其 Qt 应用程序设置自定义样式。

例如，让我们使用`Windows`样式运行第三章中的日历应用程序，*使用信号和槽处理事件*：

```py
$ python3 calendar_app.py -style Windows
```

现在尝试使用`Fusion`样式，如下所示：

```py
$ python3 calendar_app.py -style Fusion
```

请注意外观上的差异，特别是输入控件。

样式中的大小写很重要；**windows**不是有效的样式，而**Windows**是！

常见 OS 平台上可用的样式如下表所示：

| OS | 样式 |
| --- | --- |
| Windows 10 | `windowsvista`，`Windows`和`Fusion` |
| macOS | `macintosh`，`Windows`和`Fusion` |
| Ubuntu 18.04 | `Windows`和`Fusion` |

在许多 Linux 发行版中，可以从软件包存储库中获取其他 Qt 样式。可以通过调用`QtWidgets.QStyleFactory.keys()`来获取当前安装的样式列表。

样式也可以在应用程序内部设置。为了检索样式类，我们需要使用`QStyleFactory`类，如下所示：

```py
if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    windows_style = qtw.QStyleFactory.create('Windows')
    app.setStyle(windows_style)
```

`QStyleFactory.create()`将尝试查找具有给定名称的已安装样式，并返回一个`QCommonStyle`对象；如果未找到请求的样式，则它将返回`None`。然后可以使用样式对象来设置我们的`QApplication`对象的`style`属性。（`None`的值将导致其使用默认值。）

如果您计划在应用程序中设置样式，最好在绘制任何小部件之前尽早进行，以避免视觉故障。

# 自定义 Qt 样式

构建 Qt 样式是一个复杂的过程，需要深入了解 Qt 的小部件和绘图系统，很少有开发人员需要创建一个。但是，我们可能希望覆盖运行样式的某些方面，以完成一些无法通过调色板或样式表的操作来实现的事情。我们可以通过对`QtWidgets.QProxyStyle`进行子类化来实现这一点。

代理样式是我们可以使用来覆盖实际运行样式的方法的覆盖层。这样，用户选择的实际样式是什么并不重要，我们的代理样式的方法（在实现时）将被使用。

例如，让我们创建一个代理样式，强制所有屏幕文本都是大写的，如下所示：

```py
class StyleOverrides(qtw.QProxyStyle):

    def drawItemText(
        self, painter, rect,
        flags, palette, enabled,
        text, textRole
    ):
        """Force uppercase in all text"""
        text = text.upper()
        super().drawItemText(
            painter, rect, flags,
            palette, enabled, text,
            textRole
        )
```

`drawItemText()`是在必须将文本绘制到屏幕时在样式上调用的方法。它接收许多参数，但我们最关心的是要绘制的`text`参数。我们只是要拦截此文本，并在将所有参数传回`super().drawTextItem()`之前将其转换为大写。

然后可以将此代理样式应用于我们的`QApplication`对象，方式与任何其他样式相同：

```py
if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    proxy_style= StyleOverrides()
    app.setStyle(proxy_style)
```

如果此时运行程序，您会看到所有文本现在都是大写。任务完成！

# 绘制小部件

现在让我们尝试一些更有野心的事情。让我们将所有的`QLineEdit`输入框更改为绿色的圆角矩形轮廓。那么，我们如何在代理样式中做到这一点呢？

第一步是弄清楚我们要修改的小部件的元素是什么。这些可以在`QStyle`类的枚举常量中找到，它们分为三个主要类别：

+   `PrimitiveElement`，其中包括基本的非交互式 GUI 元素，如框架或背景

+   `ControlElement`，其中包括按钮或选项卡等交互元素

+   `ComplexControl`，其中包括复杂的交互元素，如组合框和滑块

这些类别中的每个项目都由`QStyle`的不同方法绘制；在这种情况下，我们想要修改的是`PE_FrameLineEdit`元素，这是一个原始元素（由`PE_`前缀表示）。这种类型的元素由`QStyle.drawPrimitive()`绘制，因此我们需要在代理样式中覆盖该方法。

将此方法添加到`StyleOverrides`中，如下所示：

```py
    def drawPrimitive(
        self, element, option, painter, widget
    ):
        """Outline QLineEdits in Green"""
```

要控制元素的绘制，我们需要向其`painter`对象发出命令，如下所示：

```py
        self.green_pen = qtg.QPen(qtg.QColor('green'))
        self.green_pen.setWidth(4)
        if element == qtw.QStyle.PE_FrameLineEdit:
            painter.setPen(self.green_pen)
            painter.drawRoundedRect(widget.rect(), 10, 10)
        else:
            super().drawPrimitive(element, option, painter, widget)
```

绘图对象和绘图将在第十二章中完全介绍，*使用 QPainter 创建 2D 图形*，但是，现在要理解的是，如果`element`参数匹配`QStyle.PE_FrameLineEdit`，则前面的代码将绘制一个绿色的圆角矩形。否则，它将将参数传递给超类的`drawPrimitive()`方法。

请注意，在绘制矩形后，我们不调用超类方法。如果我们这样做了，那么超类将在我们的绿色矩形上方绘制其样式定义的小部件元素。

正如你在这个例子中看到的，使用`QProxyStyle`比使用调色板或样式表要复杂得多，但它确实让我们几乎无限地控制我们的小部件的外观。

无论你使用 QSS 还是样式和调色板来重新设计应用程序都没有关系；然而，强烈建议你坚持使用其中一种。否则，你的样式修改可能会相互冲突，并在不同平台和桌面设置上产生不可预测的结果。

# 创建动画

没有什么比动画的巧妙使用更能为 GUI 增添精致的边缘。在颜色、大小或位置的变化之间平滑地淡入淡出的动态 GUI 元素可以为任何界面增添现代感。

Qt 的动画框架允许我们使用`QPropertyAnimation`类在我们的小部件上创建简单的动画。在本节中，我们将探讨如何使用这个类来为我们的游戏大厅增添一些动画效果。

因为 Qt 样式表会覆盖另一个基于小部件和调色板的样式，所以你需要注释掉所有这些动画的样式表代码才能正常工作。

# 基本属性动画

`QPropertyAnimation`对象用于动画小部件的单个 Qt 属性。该类会自动在两个数值属性值之间创建插值步骤序列，并在一段时间内应用这些变化。

例如，让我们动画我们的标志，让它从左向右滚动。你可以通过添加一个属性动画对象来开始，如下所示：

```py
        self.heading_animation = qtc.QPropertyAnimation(
            heading, b'maximumSize')
```

`QPropertyAnimation`需要两个参数：一个要被动画化的小部件（或其他类型的`QObject`类），以及一个指示要被动画化的属性的`bytes`对象（请注意，这是一个`bytes`对象，而不是一个字符串）。

接下来，我们需要配置我们的动画对象如下：

```py
        self.heading_animation.setStartValue(qtc.QSize(10, logo.height()))
        self.heading_animation.setEndValue(qtc.QSize(400, logo.height()))
        self.heading_animation.setDuration(2000)
```

至少，我们需要为属性设置一个`startValue`值和一个`endValue`值。当然，这些值必须是属性所需的数据类型。我们还可以设置毫秒为单位的`duration`（默认值为 250）。

配置好后，我们只需要告诉动画开始，如下所示：

```py
        self.heading_animation.start()
```

有一些要求限制了`QPropertyAnimation`对象的功能：

+   要动画的对象必须是`QObject`的子类。这包括所有小部件，但不包括一些 Qt 类，如`QPalette`。

+   要动画的属性必须是 Qt 属性（不仅仅是 Python 成员变量）。

+   属性必须具有读写访问器方法，只需要一个值。例如，`QWidget.size`可以被动画化，但`QWidget.width`不能，因为没有`setWidth()`方法。

+   属性值必顺为以下类型之一：`int`、`float`、`QLine`、`QLineF`、`QPoint`、`QPointF`、`QSize`、`QSizeF`、`QRect`、`QRectF`或`QColor`。

不幸的是，对于大多数小部件，这些限制排除了我们可能想要动画的许多方面，特别是颜色。幸运的是，我们可以解决这个问题。

# 动画颜色

正如你在本章前面学到的，小部件颜色不是小部件的属性，而是调色板的属性。调色板不能被动画化，因为`QPalette`不是`QObject`的子类，而且`setColor()`需要的不仅仅是一个单一的值。

颜色是我们想要动画的东西，为了实现这一点，我们需要对小部件进行子类化，并将其颜色设置为 Qt 属性。

让我们用一个按钮来做到这一点；在脚本的顶部开始一个新的类，如下所示：

```py
class ColorButton(qtw.QPushButton):

    def _color(self):
        return self.palette().color(qtg.QPalette.ButtonText)

    def _setColor(self, qcolor):
        palette = self.palette()
        palette.setColor(qtg.QPalette.ButtonText, qcolor)
        self.setPalette(palette)
```

在这里，我们有一个`QPushButton`子类，其中包含用于调色板`ButtonText`颜色的访问器方法。但是，请注意这些是 Python 方法；为了对此属性进行动画处理，我们需要`color`成为一个实际的 Qt 属性。为了纠正这一点，我们将使用`QtCore.pyqtProperty()`函数来包装我们的访问器方法，并在底层 Qt 对象上创建一个属性。

您可以按照以下方式操作：

```py
    color = qtc.pyqtProperty(qtg.QColor, _color, _setColor)
```

我们使用的属性名称将是 Qt 属性的名称。传递的第一个参数是属性所需的数据类型，接下来的两个参数是 getter 和 setter 方法。

`pyqtProperty()`也可以用作装饰器，如下所示：

```py
    @qtc.pyqtProperty(qtg.QColor)
    def backgroundColor(self):
        return self.palette().color(qtg.QPalette.Button)

    @backgroundColor.setter
    def backgroundColor(self, qcolor):
        palette = self.palette()
        palette.setColor(qtg.QPalette.Button, qcolor)
        self.setPalette(palette)
```

请注意，在这种方法中，两个方法必须使用我们打算创建的属性名称相同的名称。

现在我们的属性已经就位，我们需要用`ColorButton`对象替换我们的常规`QPushButton`对象：

```py
        # Replace these definitions
        # at the top of the MainWindow constructor
        self.submit = ColorButton(
            'Connect',
            clicked=lambda: qtw.QMessageBox.information(
                None,
                'Connecting',
                'Prepare for Battle!'))
        self.cancel = ColorButton(
            'Cancel',
            clicked=self.close)
```

经过这些更改，我们可以如下地对颜色值进行动画处理：

```py
        self.text_color_animation = qtc.QPropertyAnimation(
            self.submit, b'color')
        self.text_color_animation.setStartValue(qtg.QColor('#FFF'))
        self.text_color_animation.setEndValue(qtg.QColor('#888'))
        self.text_color_animation.setLoopCount(-1)
        self.text_color_animation.setEasingCurve(
            qtc.QEasingCurve.InOutQuad)
        self.text_color_animation.setDuration(2000)
        self.text_color_animation.start()
```

这个方法非常有效。我们还在这里添加了一些额外的配置设置：

+   `setLoopCount()`将设置动画重新启动的次数。值为`-1`将使其永远循环。

+   `setEasingCurve()`改变了值插值的曲线。我们选择了`InOutQuad`，它减缓了动画开始和结束的速率。

现在，当您运行脚本时，请注意颜色从白色渐变到灰色，然后立即循环回白色。如果我们希望动画从一个值移动到另一个值，然后再平稳地返回，我们可以使用`setKeyValue()`方法在动画的中间放置一个值：

```py
        self.bg_color_animation = qtc.QPropertyAnimation(
            self.submit, b'backgroundColor')
        self.bg_color_animation.setStartValue(qtg.QColor('#000'))
        self.bg_color_animation.setKeyValueAt(0.5, qtg.QColor('darkred'))
        self.bg_color_animation.setEndValue(qtg.QColor('#000'))
        self.bg_color_animation.setLoopCount(-1)
        self.bg_color_animation.setDuration(1500)
```

在这种情况下，我们的起始值和结束值是相同的，并且我们在动画的中间添加了一个值为 0.5（动画进行到一半时）设置为第二个颜色。这个动画将从黑色渐变到深红色，然后再返回。您可以添加任意多个关键值并创建相当复杂的动画。

# 使用动画组

随着我们向 GUI 添加越来越多的动画，我们可能会发现有必要将它们组合在一起，以便我们可以将动画作为一个组来控制。这可以使用动画组类`QParallelAnimationGroup`和`QSequentialAnimationGroup`来实现。

这两个类都允许我们向组中添加多个动画，并作为一个组开始、停止、暂停和恢复动画。

例如，让我们将按钮动画分组如下：

```py
        self.button_animations = qtc.QParallelAnimationGroup()
        self.button_animations.addAnimation(self.text_color_animation)
        self.button_animations.addAnimation(self.bg_color_animation)
```

`QParallelAnimationGroup`在调用其`start()`方法时会同时播放所有动画。相反，`QSequentialAnimationGroup`将按添加的顺序依次播放其动画，如下面的代码块所示：

```py
        self.all_animations = qtc.QSequentialAnimationGroup()
        self.all_animations.addAnimation(self.heading_animation)
        self.all_animations.addAnimation(self.button_animations)
        self.all_animations.start()
```

通过像我们在这里所做的那样将动画组添加到其他动画组中，我们可以将复杂的动画安排成一个对象，可以一起启动、停止、暂停和恢复。

注释掉所有其他动画的`start()`调用并启动脚本。请注意，按钮动画仅在标题动画完成后开始。

我们将在*第十二章* *使用 QPainter 进行 2D 图形*中探索更多`QPropertyAnimation`的用法。

# 总结

在本章中，我们学习了如何自定义 PyQt 应用程序的外观和感觉。我们还学习了如何操纵屏幕字体并添加图像。此外，我们还学习了如何以对路径更改具有弹性的方式打包图像和字体资源。我们还探讨了如何使用调色板和样式表改变应用程序的颜色和外观，以及如何覆盖样式方法来实现几乎无限的样式更改。最后，我们探索了使用 Qt 的动画框架进行小部件动画，并学习了如何向我们的类添加自定义 Qt 属性，以便我们可以对其进行动画处理。

在下一章中，我们将使用`QtMultimedia`库探索多媒体应用程序的世界。您将学习如何使用摄像头拍照和录制视频，如何显示视频内容，以及如何录制和播放音频。

# 问题

尝试这些问题来测试您从本章学到的知识：

1.  您正在准备分发您的文本编辑器应用程序，并希望确保用户无论使用什么平台，都会默认获得等宽字体。您可以使用哪两种方法来实现这一点？

1.  尽可能地，尝试使用`QFont`模仿以下文本：

![](img/07c03999-3b51-4ee3-8a01-aaaf1e4cf5c3.png)

1.  您能解释一下`QImage`，`QPixmap`和`QIcon`之间的区别吗？

1.  您已为应用程序定义了以下`.qrc`文件，运行了`pyrcc5`，并在脚本中导入了资源库。您会如何将此图像加载到`QPixmap`中？

```py
   <RCC>
      <qresource prefix="foodItems">
        <file alias="pancakes.png">pc_img.45234.png</file>
      </qresource>
   </RCC>
```

1.  使用`QPalette`，如何使用`tile.png`图像在`QWidget`对象的背景上铺砌？

1.  您试图使用 QSS 使删除按钮变成粉色，但没有成功。您的代码有什么问题？

```py
   deleteButton = qtw.QPushButton('Delete')
   form.layout().addWidget(deleteButton)
   form.setStyleSheet(
      form.styleSheet() + 'deleteButton{ background-color: #8F8; }'
   )
```

1.  哪个样式表字符串将把您的`QLineEdit`小部件的背景颜色变成黑色？

```py
   stylesheet1 = "QWidget {background-color: black;}"
   stylesheet2 = ".QWidget {background-color: black;}"
```

1.  构建一个简单的应用程序，其中包含一个下拉框，允许您将 Qt 样式更改为系统上安装的任何样式。包括一些其他小部件，以便您可以看到它们在不同样式下的外观。

1.  您对学习如何为 PyQt 应用程序设置样式感到非常高兴，并希望创建一个`QProxyStyle`类，该类将强制 GUI 中的所有像素图像为`smile.gif`。您会如何做？提示：您需要研究`QStyle`的一些其他绘图方法，而不是本章讨论的方法。

1.  以下动画不起作用；找出它为什么不起作用：

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

# 进一步阅读

有关更多信息，请参考以下内容：

+   有关字体如何解析的更详细描述可以在[`doc.qt.io/qt-5/qfont.html#details`](https://doc.qt.io/qt-5/qfont.html#details)的`QFont`文档中找到

+   这个 C++中的 Qt 样式示例([`doc.qt.io/qt-5/qtwidgets-widgets-styles-example.html`](https://doc.qt.io/qt-5/qtwidgets-widgets-styles-example.html))演示了如何创建一个全面的 Qt 代理样式

+   Qt 的动画框架概述在[`doc.qt.io/qt-5/animation-overview.html`](https://doc.qt.io/qt-5/animation-overview.html)提供了如何使用属性动画以及它们的限制的额外细节
