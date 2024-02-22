# 第十二章：使用`QPainter`创建 2D 图形

我们已经看到 Qt 提供了大量的小部件，具有广泛的样式和自定义功能。然而，有时我们需要直接控制屏幕上的绘制内容；例如，我们可能想要编辑图像，创建一个独特的小部件，或者构建一个交互式动画。在所有这些任务的核心是 Qt 中一个谦卑而勤奋的对象，称为`QPainter`。

在本章中，我们将在三个部分中探索 Qt 的**二维**（**2D**）图形功能：

+   使用`QPainter`进行图像编辑

+   使用`QPainter`创建自定义小部件

+   使用`QGraphicsScene`动画 2D 图形

# 技术要求

本章需要基本的 Python 和 PyQt5 设置，这是您在整本书中一直在使用的。您可能还希望从[`github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter12`](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter12)下载示例代码。

您还需要`psutil`库，可以使用以下命令从 PyPI 安装：

```py
$ pip install --user psutil
```

最后，有一些图像在手边会很有帮助，您可以用它们作为示例数据。

查看以下视频以查看代码的运行情况：[`bit.ly/2M5xzlL`](http://bit.ly/2M5xzlL)

# 使用`QPainter`进行图像编辑

在 Qt 中，可以使用`QPainter`对象在`QImage`对象上绘制图像。在第六章中，*Qt 应用程序的样式*，您了解了`QPixmap`对象，它是一个表示图形图像的显示优化对象。`QImage`对象是一个类似的对象，它针对编辑而不是显示进行了优化。为了演示如何使用`QPainter`在`QImage`对象上绘制图像，我们将构建一个经典的表情包生成器应用程序。

# 生成表情包的图形用户界面

从第四章中创建 Qt 应用程序模板的副本，*使用 QMainWindow 构建应用程序*，并将其命名为`meme_gen.py`。我们将首先构建用于表情包生成器的 GUI 表单。

# 编辑表单

在创建实际表单之前，我们将通过创建一些自定义按钮类稍微简化我们的代码：一个用于设置颜色的`ColorButton`类，一个用于设置字体的`FontButton`类，以及一个用于选择图像的`ImageFileButton`类。

`ColorButton`类的开始如下：

```py
class ColorButton(qtw.QPushButton):

   changed = qtc.pyqtSignal()

    def __init__(self, default_color, changed=None):
        super().__init__()
        self.set_color(qtg.QColor(default_color))
        self.clicked.connect(self.on_click)
        if changed:
            self.changed.connect(changed)
```

这个按钮继承自`QPushButton`，但做了一些改动。我们定义了一个`changed`信号来跟踪按钮值的变化，并添加了一个关键字选项，以便可以像内置信号一样使用关键字连接这个信号。

我们还添加了指定默认颜色的功能，该颜色将传递给`set_color`方法：

```py
    def set_color(self, color):
        self._color = color
        pixmap = qtg.QPixmap(32, 32)
        pixmap.fill(self._color)
        self.setIcon(qtg.QIcon(pixmap))
```

这种方法将传递的颜色值存储在实例变量中，然后生成给定颜色的`pixmap`对象，用作按钮图标（我们在第六章中看到了这种技术，*Qt 应用程序的样式*）。

按钮的`clicked`信号连接到`on_click()`方法：

```py
    def on_click(self):
        color = qtw.QColorDialog.getColor(self._color)
        if color:
            self.set_color(color)
            self.changed.emit()
```

这种方法打开`QColorDialog`，允许用户选择颜色，并且如果选择了颜色，则设置其颜色并发出`changed`信号。

`FontButton`类将与前一个类几乎相同：

```py
class FontButton(qtw.QPushButton):

    changed = qtc.pyqtSignal()

    def __init__(self, default_family, default_size, changed=None):
        super().__init__()
        self.set_font(qtg.QFont(default_family, default_size))
        self.clicked.connect(self.on_click)
        if changed:
            self.changed.connect(changed)

    def set_font(self, font):
        self._font = font
        self.setFont(font)
        self.setText(f'{font.family()} {font.pointSize()}')
```

与颜色按钮类似，它定义了一个可以通过关键字连接的`changed`信号。它采用默认的字体和大小，用于生成存储在按钮的`_font`属性中的默认`QFont`对象，使用`set_font()`方法。

`set_font()`方法还会更改按钮的字体和文本为所选的字体和大小。

最后，`on_click()`方法处理按钮点击：

```py
    def on_click(self):
        font, accepted = qtw.QFontDialog.getFont(self._font)
        if accepted:
            self.set_font(font)
            self.changed.emit()
```

与颜色按钮类似，我们显示一个`QFontDialog`对话框，并且如果用户选择了字体，则相应地设置按钮的字体。

最后，`ImageFileButton`类将与前两个类非常相似：

```py
class ImageFileButton(qtw.QPushButton):

    changed = qtc.pyqtSignal()

    def __init__(self, changed=None):
        super().__init__("Click to select…")
        self._filename = None
        self.clicked.connect(self.on_click)
        if changed:
            self.changed.connect(changed)

    def on_click(self):
        filename, _ = qtw.QFileDialog.getOpenFileName(
            None, "Select an image to use",
            qtc.QDir.homePath(), "Images (*.png *.xpm *.jpg)")
        if filename:
            self._filename = filename
            self.setText(qtc.QFileInfo(filename).fileName())
            self.changed.emit()
```

唯一的区别是对话框现在是一个`getOpenFileName`对话框，允许用户选择 PNG、XPM 或 JPEG 文件。

`QImage`实际上可以处理各种各样的图像文件。您可以在[`doc.qt.io/qt-5/qimage.html#reading-and-writing-image-files`](https://doc.qt.io/qt-5/qimage.html#reading-and-writing-image-files)找到这些信息，或者调用`QImageReader.supportedImageFormats()`。出于简洁起见，我们在这里缩短了列表。

现在这些类已经创建，让我们为编辑表情包属性构建一个表单：

```py
class MemeEditForm(qtw.QWidget):

    changed = qtc.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setLayout(qtw.QFormLayout())
```

这个表单将与我们在之前章节中创建的表单非常相似，但是，与其在表单提交时使用`submitted`信号不同，`changed`信号将在任何表单项更改时触发。这将允许我们实时显示任何更改，而不需要按按钮。

我们的第一个控件将是设置源图像的文件名：

```py
        self.image_source = ImageFileButton(changed=self.on_change)
        self.layout().addRow('Image file', self.image_source)
```

我们将把每个小部件的`changed`信号（或类似的信号）链接到一个名为`on_change()`的方法上，该方法将收集表单中的数据并发射`MemeEditForm`的`changed`信号。

不过，首先让我们添加字段来控制文本本身：

```py
        self.top_text = qtw.QPlainTextEdit(textChanged=self.on_change)
        self.bottom_text = qtw.QPlainTextEdit(textChanged=self.on_change)
        self.layout().addRow("Top Text", self.top_text)
        self.layout().addRow("Bottom Text", self.bottom_text)
        self.text_color = ColorButton('white', changed=self.on_change)
        self.layout().addRow("Text Color", self.text_color)
        self.text_font = FontButton('Impact', 32, changed=self.on_change)
        self.layout().addRow("Text Font", self.text_font)
```

我们的表情包将在图像的顶部和底部分别绘制文本，并且我们使用了`ColorButton`和`FontButton`类来创建文本颜色和字体的输入。再次，我们将每个小部件的适当`changed`信号连接到一个`on_changed()`实例方法。

让我们通过添加控件来绘制文本的背景框来完成表单 GUI：

```py
        self.text_bg_color = ColorButton('black', changed=self.on_change)
        self.layout().addRow('Text Background', self.text_bg_color)
        self.top_bg_height = qtw.QSpinBox(
            minimum=0, maximum=32,
            valueChanged=self.on_change, suffix=' line(s)')
        self.layout().addRow('Top BG height', self.top_bg_height)
        self.bottom_bg_height = qtw.QSpinBox(
            minimum=0, maximum=32,
            valueChanged=self.on_change, suffix=' line(s)')
        self.layout().addRow('Bottom BG height', self.bottom_bg_height)
        self.bg_padding = qtw.QSpinBox(
            minimum=0, maximum=100, value=10,
            valueChanged=self.on_change, suffix=' px')
        self.layout().addRow('BG Padding', self.bg_padding)
```

这些字段允许用户在图像太丰富而无法阅读时在文本后面添加不透明的背景。控件允许您更改顶部和底部背景的行数、框的颜色和填充。

这样就处理了表单布局，现在我们来处理`on_change()`方法：

```py
    def get_data(self):
        return {
            'image_source': self.image_source._filename,
            'top_text': self.top_text.toPlainText(),
            'bottom_text': self.bottom_text.toPlainText(),
            'text_color': self.text_color._color,
            'text_font': self.text_font._font,
            'bg_color': self.text_bg_color._color,
            'top_bg_height': self.top_bg_height.value(),
            'bottom_bg_height': self.bottom_bg_height.value(),
            'bg_padding': self.bg_padding.value()
        }

    def on_change(self):
        self.changed.emit(self.get_data())
```

首先，我们定义了一个`get_data()`方法，该方法从表单的小部件中组装一个值的`dict`对象并返回它们。如果我们需要显式地从表单中提取数据，而不是依赖信号，这将非常有用。`on_change()`方法检索这个`dict`对象并用`changed`信号发射它。

# 主 GUI

创建了表单小部件后，现在让我们组装我们的主 GUI。

让我们从`MainView.__init__()`开始：

```py
        self.setWindowTitle('Qt Meme Generator')
        self.max_size = qtc.QSize(800, 600)
        self.image = qtg.QImage(
            self.max_size, qtg.QImage.Format_ARGB32)
        self.image.fill(qtg.QColor('black'))
```

我们将从设置窗口标题开始，然后定义生成的表情包图像的最大尺寸。我们将使用这个尺寸来创建我们的`QImage`对象。由于在程序启动时我们没有图像文件，所以我们将生成一个最大尺寸的黑色占位图像，使用`fill()`方法来实现，就像我们用像素图一样。然而，当创建一个空白的`QImage`对象时，我们需要指定一个图像格式来用于生成的图像。在这种情况下，我们使用 ARGB32 格式，可以用于制作具有透明度的全彩图像。

在创建主 GUI 布局时，我们将使用这个图像：

```py
        mainwidget = qtw.QWidget()
        self.setCentralWidget(mainwidget)
        mainwidget.setLayout(qtw.QHBoxLayout())
        self.image_display = qtw.QLabel(pixmap=qtg.QPixmap(self.image))
        mainwidget.layout().addWidget(self.image_display)
        self.form = MemeTextForm()
        mainwidget.layout().addWidget(self.form)
        self.form.changed.connect(self.build_image)
```

这个 GUI 是一个简单的两面板布局，左边是一个`QLabel`对象，用于显示我们的表情包图像，右边是用于编辑的`MemeTextForm()`方法。我们将表单的`changed`信号连接到一个名为`build_image()`的`MainWindow`方法，其中包含我们的主要绘图逻辑。请注意，我们不能直接在`QLabel`对象中显示`QImage`对象；我们必须先将其转换为`QPixmap`对象。

# 使用 QImage 进行绘制

既然我们的 GUI 已经准备好了，现在是时候创建`MainView.build_image()`了。这个方法将包含所有的图像处理和绘制方法。

我们将从添加以下代码开始：

```py
    def build_image(self, data):
        if not data.get('image_source'):
            self.image.fill(qtg.QColor('black'))
        else:
            self.image.load(data.get('image_source'))
            if not (self.max_size - self.image.size()).isValid():
                # isValid returns false if either dimension is negative
                self.image = self.image.scaled(
                    self.max_size, qtc.Qt.KeepAspectRatio)
```

我们的第一个任务是设置我们的表情包的基本图像。如果在表单数据中没有 `image_source` 值，那么我们将用黑色填充我们的 `QImage` 对象，为我们的绘图提供一个空白画布。如果我们有图像来源，那么我们可以通过将其文件路径传递给 `QImage.load()` 来加载所选图像。如果我们加载的图像大于最大尺寸，我们将希望将其缩小，使其小于最大宽度和高度，同时保持相同的纵横比。

检查图像在任一维度上是否太大的一种快速方法是从最大尺寸中减去它的尺寸。如果宽度或高度大于最大值，则其中一个维度将为负，这使得减法表达式产生的 `QSize` 对象无效。

`QImage.scaled()` 方法将返回一个新的 `QImage` 对象，该对象已经按照提供的 `QSize` 对象进行了缩放。通过指定 `KeepAspectRatio`，我们的宽度和高度将分别进行缩放，以使结果大小与原始大小具有相同的纵横比。

现在我们有了我们的图像，我们可以开始在上面绘画。

# `QPainter` 对象

最后，让我们来认识一下 `QPainter` 类！`QPainter` 可以被认为是屏幕内部的一个小机器人，我们可以为它提供一个画笔和一个笔，然后发出绘图命令。

让我们创建我们的绘画“机器人”：

```py
        painter = qtg.QPainter(self.image)
```

绘图者的构造函数接收一个它将绘制的对象的引用。要绘制的对象必须是 `QPaintDevice` 的子类；在这种情况下，我们传递了一个 `QImage` 对象，它是这样一个类。传递的对象将成为绘图者的画布，在这个画布上，当我们发出绘图命令时，绘图者将进行绘制。

为了了解基本绘画是如何工作的，让我们从顶部和底部的背景块开始。我们首先要弄清楚我们需要绘制的矩形的边界：

```py
        font_px = qtg.QFontInfo(data['text_font']).pixelSize()
        top_px = (data['top_bg_height'] * font_px) + data['bg_padding']
        top_block_rect = qtc.QRect(
            0, 0, self.image.width(), top_px)
        bottom_px = (
            self.image.height() - data['bg_padding']
            - (data['bottom_bg_height'] * font_px))
        bottom_block_rect = qtc.QRect(
            0, bottom_px, self.image.width(), self.image.height())
```

`QPainter` 使用的坐标从绘画表面的左上角开始。因此，坐标 `(0, 0)` 是屏幕的左上角，而 `(width, height)` 将是屏幕的右下角。

为了计算我们顶部矩形的高度，我们将所需行数乘以我们选择的字体的像素高度（我们从 `QFontInfo` 中获取），最后加上填充量。我们最终得到一个从原点(`(0, 0)`)开始并在框的图像的完整宽度和高度处结束的矩形。这些坐标用于创建一个表示框区域的 `QRect` 对象。

对于底部的框，我们需要从图像的底部计算；这意味着我们必须首先计算矩形的高度，然后从框的高度中*减去*它。然后，我们构造一个从左侧开始并延伸到右下角的矩形。

`QRect` 坐标必须始终从左上到右下定义。

现在我们有了我们的矩形，让我们来绘制它们：

```py
        painter.setBrush(qtg.QBrush(data['bg_color']))
        painter.drawRect(top_block_rect)
        painter.drawRect(bottom_block_rect)
```

`QPainter` 有许多用于创建线条、圆圈、多边形和其他形状的绘图函数。在这种情况下，我们使用 `drawRect()`，它用于绘制矩形。为了定义这个矩形的填充，我们将绘图者的 `brush` 属性设置为一个 `QBrush` 对象，该对象设置为我们选择的背景颜色。绘图者的 `brush` 值决定了它将用什么颜色和图案来填充任何形状。

除了 `drawRect()`，`QPainter` 还包含一些其他绘图方法，如下所示：

| 方法 | 用于绘制 |
| --- | --- |
| `drawEllipse()` | 圆和椭圆 |
| `drawLine()` | 直线 |
| `drawRoundedRect()` | 带有圆角的矩形 |
| `drawPolygon()` | 任何类型的多边形 |
| `drawPixmap()` | `QPixmap` 对象 |
| `drawText()` | 文本 |

为了将我们的表情包文本放在图像上，我们需要使用 `drawText()`：

```py
        painter.setPen(data['text_color'])
        painter.setFont(data['text_font'])
        flags = qtc.Qt.AlignHCenter | qtc.Qt.TextWordWrap
        painter.drawText(
            self.image.rect(), flags | qtc.Qt.AlignTop, data['top_text'])
        painter.drawText(
            self.image.rect(), flags | qtc.Qt.AlignBottom,
            data['bottom_text'])
```

在绘制文本之前，我们需要给画家一个`QPen`对象来定义文本颜色，并给一个`QFont`对象来定义所使用的字体。画家的`QPen`确定了画家绘制的文本、形状轮廓、线条和点的颜色。

为了控制文本在图像上的绘制位置，我们可以使用`drawText()`的第一个参数，它是一个`QRect`对象，用于定义文本的边界框。然而，由于我们不知道我们要处理多少行文本，我们将使用整个图像作为边界框，并使用垂直对齐来确定文本是在顶部还是底部写入。

使用`QtCore.Qt.TextFlag`和`QtCore.Qt.AlignmentFlag`枚举的标志值来配置对齐和自动换行等行为。在这种情况下，我们为顶部和底部文本指定了居中对齐和自动换行，然后在`drawText()`调用中添加了垂直对齐选项。

`drawText()`的最后一个参数是实际的文本，我们从我们的`dict`数据中提取出来。

现在我们已经绘制了文本，我们需要做的最后一件事是在图像显示标签中设置图像：

```py
        self.image_display.setPixmap(qtg.QPixmap(self.image))
```

在这一点上，你应该能够启动程序并创建一个图像。试试看吧！

# 保存我们的图像

创建一个时髦的迷因图像后，我们的用户可能想要保存它，以便他们可以将其上传到他们最喜欢的社交媒体网站。为了实现这一点，让我们回到`MainWindow.__init_()`并创建一个工具栏：

```py
        toolbar = self.addToolBar('File')
        toolbar.addAction("Save Image", self.save_image)
```

当然，你也可以使用菜单选项或其他小部件来做到这一点。无论如何，我们需要定义由此操作调用的`save_image()`方法：

```py
    def save_image(self):
        save_file, _ = qtw.QFileDialog.getSaveFileName(
            None, "Save your image",
            qtc.QDir.homePath(), "PNG Images (*.png)")
        if save_file:
            self.image.save(save_file, "PNG")
```

要将`QImage`文件保存到磁盘，我们需要使用文件路径字符串和第二个字符串定义图像格式调用其`save()`方法。在这种情况下，我们将使用`QFileDialog.getSaveFileName()`来检索保存位置，并以`PNG`格式保存。

如果你运行你的迷因生成器，你应该会发现它看起来像下面的截图：

![](img/ce619532-1f47-4b59-bcbf-e28f4e9401a2.png)

作为额外的练习，尝试想出一些其他你想在迷因上绘制的东西，并将这个功能添加到代码中。

# 使用 QPainter 创建自定义小部件

`QPainter`不仅仅是一个专门用于在图像上绘制的工具；它实际上是为 Qt 中所有小部件绘制所有图形的工作马。换句话说，你在 PyQt 应用程序中看到的每个小部件的每个像素都是由`QPainter`对象绘制的。我们可以控制`QPainter`来创建一个纯自定义的小部件。

为了探索这个想法，让我们创建一个 CPU 监视器应用程序。获取 Qt 应用程序模板的最新副本，将其命名为`cpu_graph.py`，然后我们将开始。

# 构建一个 GraphWidget

我们的 CPU 监视器将使用区域图显示实时 CPU 活动。图表将通过颜色渐变进行增强，高值将以不同颜色显示，低值将以不同颜色显示。图表一次只显示配置数量的值，随着从右侧添加新值，旧值将滚动到小部件的左侧。

为了实现这一点，我们需要构建一个自定义小部件。我们将其命名为`GraphWidget`，并开始如下：

```py
class GraphWidget(qtw.QWidget):
    """A widget to display a running graph of information"""

    crit_color = qtg.QColor(255, 0, 0)  # red
    warn_color = qtg.QColor(255, 255, 0)  # yellow
    good_color = qtg.QColor(0, 255, 0)  # green

    def __init__(
        self, *args, data_width=20,
        minimum=0, maximum=100,
        warn_val=50, crit_val=75, scale=10,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
```

自定义小部件从一些类属性开始，用于定义*good*、*warning*和*critical*值的颜色。如果你愿意，可以随意更改这些值。

我们的构造函数接受一些关键字参数，如下所示：

+   `data_width`：这指的是一次将显示多少个值

+   `minimum`和`maximum`：要显示的最小和最大值

+   `warn_val`和`crit_val`：这些是颜色变化的阈值值

+   `Scale`：这指的是每个数据点将使用多少像素

我们的下一步是将所有这些值保存为实例属性：

```py
        self.minimum = minimum
        self.maximum = maximum
        self.warn_val = warn_val
        self.scale = scale
        self.crit_val = crit_val
```

为了存储我们的值，我们需要类似 Python `list`的东西，但受限于固定数量的项目。Python 的`collections`模块为此提供了完美的对象：`deque`类。

让我们在代码块的顶部导入这个类：

```py
from collections import deque
```

`deque`类可以接受一个`maxlen`参数，这将限制其长度。当新项目附加到`deque`类时，将其推到其`maxlen`值之外，旧项目将从列表的开头删除，以使其保持在限制之下。这对于我们的图表非常完美，因为我们只想在图表中同时显示固定数量的数据点。

我们将创建我们的`deque`类如下：

```py
        self.values = deque([self.minimum] * data_width, maxlen=data_width)
        self.setFixedWidth(data_width * scale)
```

`deque`可以接受一个`list`作为参数，该参数将用于初始化其数据。在这种情况下，我们使用一个包含最小值的`data_width`项的`list`进行初始化，并将`deque`类的`maxlen`值设置为`data_width`。

您可以通过将包含 1 个项目的列表乘以*N*在 Python 中快速创建*N*个项目的列表，就像我们在这里所做的那样；例如，`[2] * 4`将创建一个列表`[2, 2, 2, 2]`。

我们通过将小部件的固定宽度设置为`data_width * scale`来完成`__init__()`方法，这代表了我们想要显示的总像素数。

接下来，我们需要一个方法来向我们的`deque`类添加一个新值，我们将其称为`add_value()`：

```py
    def add_value(self, value):
        value = max(value, self.minimum)
        value = min(value, self.maximum)
        self.values.append(value)
        self.update()
```

该方法首先通过将我们的值限制在最小值和最大值之间，然后将其附加到`deque`对象上。这还有一个额外的效果，即将`deque`对象的开头弹出第一项，使其保持在`data_width`值。

最后，我们调用`update()`，这是一个`QWidget`方法，告诉小部件重新绘制自己。我们将在下一步处理这个绘图过程。

# 绘制小部件

`QWidget`类，就像`QImage`一样，是`QPaintDevice`的子类；因此，我们可以使用`QPainter`对象直接在小部件上绘制。当小部件收到重新绘制自己的请求时（类似于我们发出`update()`的方式），它调用其`paintEvent()`方法。我们可以用我们自己的绘图命令覆盖这个方法，为我们的小部件定义一个自定义外观。

让我们按照以下方式开始该方法：

```py
    def paintEvent(self, paint_event):
        painter = qtg.QPainter(self)
```

`paintEvent()`将被调用一个参数，一个`QPaintEvent`对象。这个对象包含有关请求重绘的事件的信息 - 最重要的是，需要重绘的区域和矩形。对于复杂的小部件，我们可以使用这些信息来仅重绘请求的部分。对于我们简单的小部件，我们将忽略这些信息，只重绘整个小部件。

我们定义了一个指向小部件本身的画家对象，因此我们向画家发出的任何命令都将在我们的小部件上绘制。让我们首先创建一个背景：

```py
        brush = qtg.QBrush(qtg.QColor(48, 48, 48))
        painter.setBrush(brush)
        painter.drawRect(0, 0, self.width(), self.height())
```

就像我们在我们的模因生成器中所做的那样，我们正在定义一个画刷，将其给我们的画家，并画一个矩形。

请注意，我们在这里使用了`drawRect()`的另一种形式，它直接取坐标而不是`QRect`对象。`QPainter`对象的许多绘图函数都有取稍微不同类型参数的替代版本，以增加灵活性。

接下来，让我们画一些虚线，显示警告和临界的阈值在哪里。为此，我们需要将原始数据值转换为小部件上的*y*坐标。由于这将经常发生，让我们创建一个方便的方法来将值转换为*y*坐标：

```py
    def val_to_y(self, value):
        data_range = self.maximum - self.minimum
        value_fraction = value / data_range
        y_offset = round(value_fraction * self.height())
        y = self.height() - y_offset
        return y
```

要将值转换为*y*坐标，我们首先需要确定值代表数据范围的什么比例。然后，我们将该分数乘以小部件的高度，以确定它应该离小部件底部多少像素。然后，因为像素坐标从顶部开始计数*向下*，我们必须从小部件的高度中减去我们的偏移量，以确定*y*坐标。

回到`paintEvent()`，让我们使用这个方法来画一个警告阈值线：

```py
        pen = qtg.QPen()
        pen.setDashPattern([1, 0])
        warn_y = self.val_to_y(self.warn_val)
        pen.setColor(self.warn_color)
        painter.setPen(pen)
        painter.drawLine(0, warn_y, self.width(), warn_y)
```

由于我们正在绘制一条线，我们需要设置绘图者的`pen`属性。`QPen.setDashPattern()`方法允许我们通过向其传递`1`和`0`值的列表来为线定义虚线模式，表示绘制或未绘制的像素。在这种情况下，我们的模式将在绘制像素和空像素之间交替。

创建了笔之后，我们使用我们的新转换方法将`warn_val`值转换为*y*坐标，并将笔的颜色设置为`warn_color`。我们将配置好的笔交给我们的绘图者，并指示它在我们计算出的*y*坐标处横跨小部件的宽度绘制一条线。

同样的方法可以用来绘制我们的临界阈值线：

```py
        crit_y = self.val_to_y(self.crit_val)
        pen.setColor(self.crit_color)
        painter.setPen(pen)
        painter.drawLine(0, crit_y, self.width(), crit_y)
```

我们可以重用我们的`QPen`对象，但请记住，每当我们对笔或刷子进行更改时，我们都必须重新分配给绘图者。绘图者传递了笔或刷子的副本，因此我们对对象进行的更改*在*分配给绘图者之后不会隐式传递给使用的笔或刷子。

在第六章中，*Qt 应用程序的样式*，您学习了如何创建一个渐变对象并将其应用于`QBrush`对象。在这个应用程序中，我们希望使用渐变来绘制我们的数据值，使得高值在顶部为红色，中等值为黄色，低值为绿色。

让我们定义一个`QLinearGradient`渐变对象如下：

```py
        gradient = qtg.QLinearGradient(
            qtc.QPointF(0, self.height()), qtc.QPointF(0, 0))
```

这个渐变将从小部件的底部（`self.height()`）到顶部（`0`）进行。这一点很重要要记住，因为在定义颜色停止时，`0`位置表示渐变的开始（即小部件的底部），`1`位置将表示渐变的结束（即顶部）。

我们将设置我们的颜色停止如下：

```py
        gradient.setColorAt(0, self.good_color)
        gradient.setColorAt(
            self.warn_val/(self.maximum - self.minimum),
            self.warn_color)
        gradient.setColorAt(
            self.crit_val/(self.maximum - self.minimum),
            self.crit_color)
```

类似于我们计算*y*坐标的方式，在这里，我们通过将警告和临界值除以最小值和最大值之间的差来确定数据范围表示的警告和临界值的分数。这个分数是`setColorAt()`需要的第一个参数。

现在我们有了一个渐变，让我们为绘制数据设置我们的绘图者：

```py
        brush = qtg.QBrush(gradient)
        painter.setBrush(brush)
        painter.setPen(qtc.Qt.NoPen)
```

为了使我们的面积图看起来平滑和连贯，我们不希望图表部分有任何轮廓。为了阻止`QPainter`勾勒形状，我们将我们的笔设置为一个特殊的常数：`QtCore.Qt.NoPen`。

为了创建我们的面积图，每个数据点将由一个四边形表示，其中右上角将是当前数据点，左上角将是上一个数据点。宽度将等于我们在构造函数中设置的`scale`属性。

由于我们将需要每个数据点的*上一个*值，我们需要从一点开始进行一些簿记：

```py
        self.start_value = getattr(self, 'start_value', self.minimum)
        last_value = self.start_value
        self.start_value = self.values[0]
```

我们需要做的第一件事是确定一个起始值。由于我们需要在当前值*之前*有一个值，我们的第一项需要一个开始绘制的地方。我们将创建一个名为`start_value`的实例变量，它在`paintEvent`调用之间保持不变，并存储初始值。然后，我们将其赋值给`last_value`，这是一个本地变量，将用于记住循环的每次迭代的上一个值。最后，我们将起始值更新为`deque`对象的第一个值，以便*下一次*调用`paintEvent`。

现在，让我们开始循环遍历数据并计算每个点的`x`和`y`值：

```py
        for indx, value in enumerate(self.values):
            x = (indx + 1) * self.scale
            last_x = indx * self.scale
            y = self.val_to_y(value)
            last_y = self.val_to_y(last_value)
```

多边形的两个*x*坐标将是（1）值的索引乘以比例，和（2）比例乘以值的索引加一。对于*y*值，我们将当前值和上一个值传递给我们的转换方法。这四个值将使我们能够绘制一个四边形，表示从一个数据点到下一个数据点的变化。

要绘制该形状，我们将使用一个称为`QPainterPath`的对象。在数字图形中，**路径**是由单独的线段或形状组合在一起构建的对象。`QPainterPath`对象允许我们通过在代码中逐个绘制每一边来创建一个独特的形状。

接下来，让我们使用我们计算出的`x`和`y`数据开始绘制我们的路径对象：

```py
            path = qtg.QPainterPath()
            path.moveTo(x, self.height())
            path.lineTo(last_x, self.height())
            path.lineTo(last_x, last_y)
            path.lineTo(x, y)
```

要绘制路径，我们首先创建一个`QPainterPath`对象。然后我们使用它的`moveTo()`方法设置绘制的起始点。然后我们使用`lineTo()`方法连接路径的四个角，以在点之间绘制一条直线。最后一个连接我们的结束点和起始点是自动完成的。

请注意，此时我们实际上并没有在屏幕上绘制；我们只是在定义一个对象，我们的绘图器可以使用其当前的画笔和笔将其绘制到屏幕上。

让我们绘制这个对象：

```py
            painter.drawPath(path)
            last_value = value
```

我们通过绘制路径和更新最后一个值到当前值来完成了这个方法。当然，这条由直线组成的路径相当乏味——我们本可以只使用绘图器的`drawPolygon()`方法。使用`QPainterPath`对象的真正威力在于利用它的非线性绘制方法。

例如，如果我们希望我们的图表是平滑和圆润的，而不是锯齿状的，那么我们可以使用**立方贝塞尔曲线**来绘制最后一条线（即形状的顶部），而不是直线：

```py
            #path.lineTo(x, y)
            c_x = round(self.scale * .5) + last_x
            c1 = (c_x, last_y)
            c2 = (c_x, y)
            path.cubicTo(*c1, *c2, x, y)
```

贝塞尔曲线使用两个控制点来定义其曲线。每个控制点都会将线段拉向它自己——第一个控制点拉动线段的前半部分，第二个控制点拉动线段的后半部分：

![](img/004dbd50-bd0b-40cc-8967-020135f4a640.png)

我们将第一个控制点设置为最后的 y 值，将第二个控制点设置为当前的 y 值——这两个值都是开始和结束 x 值的中间值。这给我们在上升斜坡上一个 S 形曲线，在下降斜坡上一个反 S 形曲线，从而产生更柔和的峰值和谷值。

在应用程序中设置`GraphWidget`对象后，您可以尝试在曲线和线命令之间切换以查看差异。

# 使用 GraphWidget

我们的图形小部件已经完成，所以让我们转到`MainWindow`并使用它。

首先创建您的小部件并将其设置为中央小部件：

```py
        self.graph = GraphWidget(self)
        self.setCentralWidget(self.graph)
```

接下来，让我们创建一个方法，该方法将读取当前的 CPU 使用情况并将其发送到`GraphWidget`。为此，我们需要从`psutil`库导入`cpu_percent`函数：

```py
from psutil import cpu_percent
```

现在我们可以编写我们的图形更新方法如下：

```py
    def update_graph(self):
        cpu_usage = cpu_percent()
        self.graph.add_value(cpu_usage)
```

`cpu_percent()`函数返回一个从 0 到 100 的整数，反映了计算机当前的 CPU 利用率。这非常适合直接发送到我们的`GraphWidget`，其默认范围是 0 到 100。

现在我们只需要定期调用这个方法来更新图形；在`MainWindow.__init__()`中，添加以下代码：

```py
        self.timer = qtc.QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_graph)
        self.timer.start()
```

这只是一个`QTimer`对象，您在第十章中学到的，*使用 QTimer 和 QThread 进行多线程处理*，设置为每秒调用一次`update_graph()`。

如果现在运行应用程序，您应该会得到类似于这样的结果：

![](img/9f215c0b-8fcb-4e0d-acbf-b344e4bf5605.png)

注意我们的贝塞尔曲线所创建的平滑峰值。如果切换回直线代码，您将看到这些峰值变得更加尖锐。

如果您的 CPU 太强大，无法提供有趣的活动图，请尝试对`update_graph()`进行以下更改以更好地测试小部件：

```py
    def update_graph(self):
        import random
        cpu_usage = random.randint(1, 100)
        self.graph.add_value(cpu_usage)
```

这将只输出介于`1`和`100`之间的随机值，并且应该产生一些相当混乱的结果。

看到这个 CPU 图表实时动画可能会让您对 Qt 的动画能力产生疑问。在下一节中，我们将学习如何使用`QPainter`和 Qt 图形视图框架一起创建 Qt 中的 2D 动画。

# 使用 QGraphicsScene 进行 2D 图形动画

在简单的小部件和图像编辑中，对`QPaintDevice`对象进行绘制效果很好，但在我们想要绘制大量的 2D 对象，并可能实时地对它们进行动画处理的情况下，我们需要一个更强大的对象。Qt 提供了 Graphics View Framework，这是一个基于项目的模型视图框架，用于组合复杂的 2D 图形和动画。

为了探索这个框架的运作方式，我们将创建一个名为**Tankity Tank Tank Tank**的游戏。

# 第一步

这个坦克游戏将是一个两人对战游戏，模拟了你可能在经典的 1980 年代游戏系统上找到的简单动作游戏。一个玩家将在屏幕顶部，一个在底部，两辆坦克将不断从左到右移动，每个玩家都试图用一颗子弹射击对方。

要开始，将您的 Qt 应用程序模板复制到一个名为`tankity_tank_tank_tank.py`的新文件中。从文件顶部的`import`语句之后开始，我们将添加一些常量：

```py
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BORDER_HEIGHT = 100
```

这些常量将在整个游戏代码中用于计算大小和位置。实际上，我们将立即在`MainWindow.__init__()`中使用其中的两个：

```py
        self.resize(qtc.QSize(SCREEN_WIDTH, SCREEN_HEIGHT))
        self.scene = Scene()
        view = qtw.QGraphicsView(self.scene)
        self.setCentralWidget(view)
```

这是我们将要添加到`MainWindow`中的所有代码。在将窗口调整大小为我们的宽度和高度常量之后，我们将创建两个对象，如下：

+   第一个是`Scene`对象。这是一个我们将要创建的自定义类，是从`QGraphicsScene`派生的。`QGraphicsScene`是这个模型视图框架中的模型，表示包含各种图形项目的 2D 场景。

+   第二个是`QGraphicsView`对象，它是框架的视图组件。这个小部件的工作只是渲染场景并将其显示给用户。

我们的`Scene`对象将包含游戏的大部分代码，所以我们将下一步构建那部分。

# 创建一个场景

`Scene`类将是我们游戏的主要舞台，并将管理游戏中涉及的各种对象，如坦克、子弹和墙壁。它还将显示分数并跟踪其他游戏逻辑。

让我们这样开始：

```py
class Scene(qtw.QGraphicsScene):

    def __init__(self):
        super().__init__()
        self.setBackgroundBrush(qtg.QBrush(qtg.QColor('black')))
        self.setSceneRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
```

我们在这里做的第一件事是通过设置`backgroundBrush`属性将我们的场景涂成黑色。这个属性自然地需要一个`QBrush`对象，它将用来填充场景的背景。我们还设置了`sceneRect`属性，它描述了场景的大小，设置为我们的宽度和高度常量的`QRect`对象。

要开始在场景上放置对象，我们可以使用它的许多 add 方法之一：

```py
        wall_brush = qtg.QBrush(qtg.QColor('blue'), qtc.Qt.Dense5Pattern)
        floor = self.addRect(
            qtc.QRectF(0, SCREEN_HEIGHT - BORDER_HEIGHT,
                       SCREEN_WIDTH, BORDER_HEIGHT),
            brush=wall_brush)
        ceiling = self.addRect(
            qtc.QRectF(0, 0, SCREEN_WIDTH, BORDER_HEIGHT),
            brush=wall_brush)
```

在这里，我们使用`addRect()`在场景上绘制了两个矩形——一个在底部作为地板，一个在顶部作为天花板。就像`QPainter`类一样，`QGraphicsScene`有方法来添加椭圆、像素图、线、多边形、文本和其他这样的项目。然而，与绘图程序不同，`QGraphicsScene`方法不仅仅是将像素绘制到屏幕上；相反，它们创建了`QGraphicsItem`类（或其子类）的项目。我们随后可以查询或操作所创建的项目。

例如，我们可以添加一些文本项目来显示我们的分数，如下所示：

```py
        self.top_score = 0
        self.bottom_score = 0
        score_font = qtg.QFont('Sans', 32)
        self.top_score_display = self.addText(
            str(self.top_score), score_font)
        self.top_score_display.setPos(10, 10)
        self.bottom_score_display = self.addText(
            str(self.bottom_score), score_font)
        self.bottom_score_display.setPos(
            SCREEN_WIDTH - 60, SCREEN_HEIGHT - 60)
```

在这里，在创建文本项目之后，我们正在操作它们的属性，并使用`setPos()`方法设置每个文本项目的位置。

我们还可以更新项目中的文本；例如，让我们创建方法来更新我们的分数：

```py
    def top_score_increment(self):
        self.top_score += 1
        self.top_score_display.setPlainText(str(self.top_score))

    def bottom_score_increment(self):
        self.bottom_score += 1
        self.bottom_score_display.setPlainText(str(self.bottom_score))
```

如果你把`QPainter`比作在纸上绘画，那么把`QGraphicsItems`添加到`QGraphicsScene`类就相当于在毛毯图上放置毛毡形状。项目*在*场景上，但它们不是场景的一部分，因此它们可以被改变或移除。

# 创建坦克

我们的游戏将有两辆坦克，一辆在屏幕顶部，一辆在底部。这些将在`Scene`对象上绘制，并进行动画处理，以便玩家可以左右移动它们。在第六章中，*Qt 应用程序的样式*，您学到了可以使用`QPropertyAnimation`进行动画处理，但是*只有*被动画处理的属性属于`QObject`的后代。`QGraphicsItem`不是`QObject`的后代，但`QGraphicsObject`对象将两者结合起来，为我们提供了一个可以进行动画处理的图形项。

因此，我们需要将我们的`Tank`类构建为`QGraphicsObject`的子类：

```py
class Tank(qtw.QGraphicsObject):

    BOTTOM, TOP = 0, 1
    TANK_BM = b'\x18\x18\xFF\xFF\xFF\xFF\xFF\x66'
```

这个类首先定义了两个常量，`TOP`和`BOTTOM`。这将用于表示我们是在屏幕顶部还是底部创建坦克。

`TANK_BM`是一个包含坦克图形的 8×8 位图数据的`bytes`对象。我们很快就会看到这是如何工作的。

首先，让我们开始构造函数：

```py
    def __init__(self, color, y_pos, side=TOP):
        super().__init__()
        self.side = side
```

我们的坦克将被赋予颜色、*y*坐标和`side`值，该值将是`TOP`或`BOTTOM`。我们将使用这些信息来定位和定向坦克。

接下来，让我们使用我们的`bytes`字符串为我们的坦克创建一个位图：

```py
        self.bitmap = qtg.QBitmap.fromData(
            qtc.QSize(8, 8), self.TANK_BM)
```

`QBitmap`对象是`QPixmap`的单色图像的特殊情况。通过将大小和`bytes`对象传递给`fromData()`静态方法，我们可以生成一个简单的位图对象，而无需单独的图像文件。

为了理解这是如何工作的，请考虑`TANK_BM`字符串。因为我们将其解释为 8×8 图形，所以该字符串中的每个字节（8 位）对应于图形的一行。

如果您将每一行转换为二进制数字并将它们按每行一个字节的方式排列，它将如下所示：

```py
00011000
00011000
11111111
11111111
11111111
11111111
11111111
01100110
```

由 1 创建的形状实质上是该位图将采用的形状。当然，8x8 的图形将非常小，所以我们应该将其放大。此外，这辆坦克显然是指向上的，所以如果我们是顶部的坦克，我们需要将其翻转过来。

我们可以使用`QTransform`对象来完成这两件事：

```py
        transform = qtg.QTransform()
        transform.scale(4, 4)  # scale to 32x32
        if self.side == self.TOP:  # We're pointing down
            transform.rotate(180)
        self.bitmap = self.bitmap.transformed(transform)
```

`QTransform`对象表示要在`QPixmap`或`QBitmap`上执行的一组变换。创建变换对象后，我们可以设置要应用的各种变换，首先是缩放操作，然后是添加`rotate`变换（如果坦克在顶部）。`QTransform`对象可以传递给位图的`transformed()`方法，该方法返回一个应用了变换的新`QBitmap`对象。

该位图是单色的，默认情况下是黑色。要以其他颜色绘制，我们将需要一个设置为所需颜色的`QPen`（而不是刷子！）对象。让我们使用我们的`color`参数按如下方式创建它：

```py
        self.pen = qtg.QPen(qtg.QColor(color))
```

`QGraphicsObject`对象的实际外观是通过重写`paint()`方法确定的。让我们按照以下方式创建它：

```py
    def paint(self, painter, option, widget):
        painter.setPen(self.pen)
        painter.drawPixmap(0, 0, self.bitmap)
```

`paint()`的第一个参数是`QPainter`对象，Qt 已经创建并分配给绘制对象。我们只需要对该绘图程序应用命令，它将根据我们的要求绘制图像。我们将首先将`pen`属性设置为我们创建的笔，然后使用绘图程序的`drawPixmap()`方法来绘制我们的位图。

请注意，我们传递给`drawPixmap()`的坐标不是`QGraphicsScene`类的坐标，而是`QGraphicsObject`对象本身的边界矩形内的坐标。因此，我们需要确保我们的对象返回一个适当的边界矩形，以便我们的图像被正确绘制。

为了做到这一点，我们需要重写`boundingRect()`方法：

```py
    def boundingRect(self):
        return qtc.QRectF(0, 0, self.bitmap.width(),
                          self.bitmap.height())
```

在这种情况下，我们希望我们的`boundingRect()`方法返回一个与位图大小相同的矩形。

回到`Tank.__init__()`，让我们定位我们的坦克：

```py
        if self.side == self.BOTTOM:
            y_pos -= self.bitmap.height()
        self.setPos(0, y_pos)
```

`QGraphicsObject.setPos()`方法允许您使用像素坐标将对象放置在其分配的`QGraphicsScene`上的任何位置。由于像素坐标始终从对象的左上角计数，如果对象在屏幕底部，我们需要调整对象的*y*坐标，使其自身高度升高，以便坦克的*底部*距离屏幕顶部`y_pos`像素。

对象的位置始终表示其左上角的位置。

现在我们想要让我们的坦克动起来；每个坦克将在*x*轴上来回移动，在触碰屏幕边缘时会反弹。

让我们创建一个`QPropertyAnimation`方法来实现这一点：

```py
        self.animation = qtc.QPropertyAnimation(self, b'x')
        self.animation.setStartValue(0)
        self.animation.setEndValue(SCREEN_WIDTH - self.bitmap.width())
        self.animation.setDuration(2000)
```

`QGraphicsObject`对象具有定义其在场景上的*x*和*y*坐标的`x`和`y`属性，因此将对象进行动画处理就像是将我们的属性动画指向这些属性。我们将从`0`开始动画`x`，并以屏幕的宽度结束；但是，为了防止我们的坦克离开边缘，我们需要从该值中减去位图的宽度。最后，我们设置两秒的持续时间。

属性动画可以向前或向后运行。因此，要启用左右移动，我们只需要切换动画运行的方向。让我们创建一些方法来做到这一点：

```py
    def toggle_direction(self):
        if self.animation.direction() == qtc.QPropertyAnimation.Forward:
            self.left()
        else:
            self.right()

    def right(self):
        self.animation.setDirection(qtc.QPropertyAnimation.Forward)
        self.animation.start()

    def left(self):
        self.animation.setDirection(qtc.QPropertyAnimation.Backward)
        self.animation.start()
```

改变方向只需要设置动画对象的`direction`属性为`Forward`或`Backward`，然后调用`start()`来应用它。

回到`__init__()`，让我们使用`toggle_direction()`方法来创建*反弹*：

```py
        self.animation.finished.connect(self.toggle_direction)
```

为了使游戏更有趣，我们还应该让我们的坦克从屏幕的两端开始：

```py
        if self.side == self.TOP:
            self.toggle_direction()
        self.animation.start()
```

设置动画后，通过调用`start()`来启动它。这处理了坦克的动画；现在是时候装载我们的武器了。

# 创建子弹

在这个游戏中，每个坦克一次只能在屏幕上有一个子弹。这简化了我们的游戏代码，但也使游戏保持相对具有挑战性。

为了实现这些子弹，我们将创建另一个名为`Bullet`的`QGraphicsObject`对象，它被动画化沿着*y*轴移动。

让我们开始我们的`Bullet`类如下：

```py
class Bullet(qtw.QGraphicsObject):

    hit = qtc.pyqtSignal()

    def __init__(self, y_pos, up=True):
        super().__init__()
        self.up = up
        self.y_pos = y_pos
```

子弹类首先通过定义`hit`信号来表示它击中了敌方坦克。构造函数接受一个`y_pos`参数来定义子弹的起始点，并且一个布尔值来指示子弹是向上还是向下移动。这些参数被保存为实例变量。

接下来，让我们按照以下方式定义子弹的外观：

```py
    def boundingRect(self):
        return qtc.QRectF(0, 0, 10, 10)

    def paint(self, painter, options, widget):
        painter.setBrush(qtg.QBrush(qtg.QColor('yellow')))
        painter.drawRect(0, 0, 10, 10)
```

我们的子弹将简单地是一个 10×10 的黄色正方形，使用绘图器的`drawRect()`方法创建。这对于复古游戏来说是合适的，但是为了好玩，让我们把它变得更有趣。为此，我们可以将称为`QGraphicsEffect`的类应用于`QGraphicsObject`。`QGraphicsEffect`类可以实时地对对象应用视觉效果。我们通过创建`QGraphicEffect`类的子类实例并将其分配给子弹的`graphicsEffect`属性来实现这一点，如下所示：

```py
        blur = qtw.QGraphicsBlurEffect()
        blur.setBlurRadius(10)
        blur.setBlurHints(
            qtw.QGraphicsBlurEffect.AnimationHint)
 self.setGraphicsEffect(blur)
```

添加到`Bullet.__init__()`的这段代码创建了一个模糊效果并将其应用到我们的`QGraphicsObject`类。请注意，这是应用在对象级别上的，而不是在绘画级别上，因此它适用于我们绘制的任何像素。我们已将模糊半径调整为 10 像素，并添加了`AnimationHint`对象，告诉我们正在应用于动画对象的效果，并激活某些性能优化。

说到动画，让我们按照以下方式创建子弹的动画：

```py
        self.animation = qtc.QPropertyAnimation(self, b'y')
        self.animation.setStartValue(y_pos)
        end = 0 if up else SCREEN_HEIGHT
        self.animation.setEndValue(end)
        self.animation.setDuration(1000)
```

动画被配置为使子弹从当前的`y_pos`参数到屏幕的顶部或底部花费一秒的时间，具体取决于子弹是向上还是向下射击。不过我们还没有开始动画，因为我们不希望子弹在射击前就开始移动。

射击将在`shoot()`方法中发生，如下所示：

```py
    def shoot(self, x_pos):
        self.animation.stop()
        self.setPos(x_pos, self.y_pos)
        self.animation.start()
```

当玩家射出子弹时，我们首先停止任何可能发生的动画。由于一次只允许一颗子弹，快速射击只会导致子弹重新开始（虽然这并不是非常现实，但这样做可以使游戏更具挑战性）。

然后，将子弹重新定位到*x*坐标并传递到`shoot()`方法和坦克的*y*坐标。最后，启动动画。这个想法是，当玩家射击时，我们将传入坦克当前的*x*坐标，子弹将从那个位置直线飞出。

让我们回到我们的`Tank`类，并添加一个`Bullet`对象。在`Tank.__init__()`中，添加以下代码：

```py
        bullet_y = (
            y_pos - self.bitmap.height()
            if self.side == self.BOTTOM
            else y_pos + self.bitmap.height()
        )
        self.bullet = Bullet(bullet_y, self.side == self.BOTTOM)
```

为了避免我们的子弹击中自己的坦克，我们希望子弹从底部坦克的正上方或顶部坦克的正下方开始，这是我们在第一条语句中计算出来的。由于我们的坦克不会上下移动，这个位置是一个常数，我们可以将它传递给子弹的构造函数。

为了让坦克射出子弹，我们将在`Tank`类中创建一个名为`shoot()`的方法：

```py
    def shoot(self):
        if not self.bullet.scene():
            self.scene().addItem(self.bullet)
        self.bullet.shoot(self.x())
```

我们需要做的第一件事是将子弹添加到场景中（如果尚未添加或已被移除）。我们可以通过检查子弹的`scene`属性来确定这一点，如果对象不在场景中，则返回`None`。

然后，通过传入坦克的*x*坐标来调用子弹的`shoot()`方法。

# 碰撞检测

如果子弹击中目标后什么都不发生，那么子弹就没有什么用。为了在子弹击中坦克时发生一些事情，我们需要实现**碰撞检测**。我们将在`Bullet`类中实现这一点，要求它在移动时检查是否击中了任何东西。

首先在`Bullet`中创建一个名为`check_colllision()`的方法：

```py
    def check_collision(self):
        colliding_items = self.collidingItems()
        if colliding_items:
            self.scene().removeItem(self)
            for item in colliding_items:
                if type(item).__name__ == 'Tank':
                    self.hit.emit()
```

`QGraphicsObject.collidingItems()`返回一个列表，其中包含任何与此项的边界矩形重叠的`QGraphicsItem`对象。这不仅包括我们的`Tank`对象，还包括我们在`Scene`类中创建的`floor`和`ceiling`项，甚至是另一个坦克的`Bullet`对象。如果我们的子弹触碰到这些物品中的任何一个，我们需要将其从场景中移除；为此，我们调用`self.scene().removeItem(self)`来消除子弹。

然后，我们需要检查我们碰撞的物品中是否有`Tank`对象。我们只需检查被击中的对象的类型和名称即可。如果我们击中了坦克，我们就会发出`hit`信号。（我们可以安全地假设它是另一个坦克，因为我们的子弹移动的方式）

每次`Bullet`对象移动时都需要调用这个方法，因为每次移动都可能导致碰撞。幸运的是，`QGraphicsObject`方法有一个`yChanged`信号，每当它的*y*坐标发生变化时就会发出。

因此，在`Bullet.__init__()`方法中，我们可以添加一个连接，如下所示：

```py
        self.yChanged.connect(self.check_collision)
```

我们的坦克和子弹对象现在已经准备就绪，所以让我们回到`Scene`对象来完成我们的游戏。

# 结束游戏

回到`Scene.__init__()`，让我们创建我们的两辆坦克：

```py
        self.bottom_tank = Tank(
            'red', floor.rect().top(), Tank.BOTTOM)
        self.addItem(self.bottom_tank)

        self.top_tank = Tank(
            'green', ceiling.rect().bottom(), Tank.TOP)
        self.addItem(self.top_tank)
```

底部坦克位于地板上方，顶部坦克位于天花板下方。现在我们可以将它们的子弹的`hit`信号连接到适当的分数增加方法：

```py
        self.top_tank.bullet.hit.connect(self.top_score_increment)
        self.bottom_tank.bullet.hit.connect(self.bottom_score_increment)
```

到目前为止，我们的游戏几乎已经完成了：

![](img/492381be-ba26-4e11-91b5-de1470a9ef5a.png)

当然，还有一个非常重要的方面还缺失了——控制！

我们的坦克将由键盘控制；我们将为底部玩家分配箭头键进行移动和回车键进行射击，而顶部玩家将使用*A*和*D*进行移动，空格键进行射击。

为了处理按键，我们需要重写`Scene`对象的`keyPressEvent()`方法：

```py
    def keyPressEvent(self, event):
        keymap = {
            qtc.Qt.Key_Right: self.bottom_tank.right,
            qtc.Qt.Key_Left: self.bottom_tank.left,
            qtc.Qt.Key_Return: self.bottom_tank.shoot,
            qtc.Qt.Key_A: self.top_tank.left,
            qtc.Qt.Key_D: self.top_tank.right,
            qtc.Qt.Key_Space: self.top_tank.shoot
        }
        callback = keymap.get(event.key())
        if callback:
            callback()
```

`keyPressEvent()`在`Scene`对象聚焦时每当用户按下键盘时被调用。它是唯一的参数，是一个`QKeyEvent`对象，其`key()`方法返回`QtCore.Qt.Key`枚举中的常量，告诉我们按下了什么键。在这个方法中，我们创建了一个`dict`对象，将某些键常量映射到我们的坦克对象的方法。每当我们接收到一个按键，我们尝试获取一个回调方法，如果成功，我们调用这个方法。

游戏现在已经准备好玩了！找个朋友（最好是你不介意和他共享键盘的人）并开始玩吧。

# 总结

在本章中，您学习了如何在 PyQt 中使用 2D 图形。我们学习了如何使用`QPainter`对象编辑图像并创建自定义小部件。然后，您学习了如何使用`QGraphicsScene`方法与`QGraphicsObject`类结合使用，创建可以使用自动逻辑或用户输入控制的动画场景。

在下一章中，我们将为我们的图形添加一个额外的维度，探索在 PyQt 中使用 OpenGL 3D 图形。您将学习一些 OpenGL 编程的基础知识，以及如何将其集成到 PyQt 应用程序中。

# 问题

尝试这些问题来测试你从本章学到的知识：

1.  在这个方法中添加代码，以在图片底部用蓝色写下你的名字：

```py
       def create_headshot(self, image_file, name):
           image = qtg.QImage()
           image.load(image_file)
           # your code here

           # end of your code
           return image
```

1.  给定一个名为`painter`的`QPainter`对象，写一行代码在绘图设备的左上角绘制一个 80×80 像素的八边形。您可以参考[`doc.qt.io/qt-5/qpainter.html#drawPolygon`](https://doc.qt.io/qt-5/qpainter.html#drawPolygon)中的文档进行指导。

1.  您正在创建一个自定义小部件，但不知道为什么文本显示为黑色。以下是您的`paintEvent()`方法；看看你能否找出问题：

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

1.  深炸迷因是一种使用极端压缩、饱和度和其他处理来使迷因图像故意看起来低质量的迷因风格。在你的迷因生成器中添加一个功能，可以选择使迷因深炸。你可以尝试的一些事情包括减少颜色位深度和调整图像中颜色的色调和饱和度。

1.  您想要动画一个圆在屏幕上水平移动。更改以下代码以动画圆：

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

1.  以下代码尝试使用渐变刷设置`QPainter`对象。找出其中的问题所在：

```py
   gradient = qtg.QLinearGradient(
       qtc.QPointF(0, 100), qtc.QPointF(0, 0))
   gradient.setColorAt(20, qtg.QColor('red'))
   gradient.setColorAt(40, qtg.QColor('orange'))
   gradient.setColorAt(60, qtg.QColor('green'))
   painter = QPainter()
   painter.setGradient(gradient)
```

1.  看看你是否可以实现一些对我们创建的游戏的改进：

+   +   脉动子弹

+   坦克被击中时爆炸

+   声音（参见第七章，*使用 QtMultimedia 处理音频-视觉*，以获取指导）

+   背景动画

+   多个子弹

# 进一步阅读

有关更多信息，请参阅以下内容：

+   有关`QPainter`和 Qt 绘图系统的深入讨论可以在[`doc.qt.io/qt-5/paintsystem.html`](https://doc.qt.io/qt-5/paintsystem.html)找到

+   Qt 图形视图框架的概述可以在[`doc.qt.io/qt-5/graphicsview.html`](https://doc.qt.io/qt-5/graphicsview.html)找到

+   动画框架的概述可以在[`doc.qt.io/qt-5/animation-overview.html`](https://doc.qt.io/qt-5/animation-overview.html)找到
