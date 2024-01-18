# 使用图形

在每个应用程序中，图形在使其更加用户友好方面起着重要作用。图形使概念更容易理解。在本章中，我们将涵盖以下主题：

+   显示鼠标坐标

+   显示鼠标点击和释放的坐标

+   显示鼠标按钮点击的点

+   在两次鼠标点击之间绘制一条线

+   绘制不同类型的线

+   绘制所需大小的圆

+   在两次鼠标点击之间绘制一个矩形

+   以所需的字体和大小绘制文本

+   创建显示不同图形工具的工具栏

+   使用 Matplotlib 绘制一条线

+   使用 Matplotlib 绘制条形图

# 介绍

为了在 Python 中进行绘制和绘画，我们将使用几个类。其中最重要的是`QPainter`类。

这个类用于绘图。它可以绘制线条、矩形、圆形和复杂的形状。在使用`QPainter`绘图时，可以使用`QPainter`类的笔来定义绘图的颜色、笔/刷的粗细、样式，以及线条是实线、虚线还是点划线等。

本章中使用了`QPainter`类的几种方法来绘制不同的形状。以下是其中的一些：

+   `QPainter::drawLine()`: 该方法用于在两组*x*和*y*坐标之间绘制一条线

+   `QPainter::drawPoints()`: 该方法用于在通过提供的*x*和*y*坐标指定的位置绘制一个点

+   `QPainter::drawRect()`: 该方法用于在两组*x*和*y*坐标之间绘制一个矩形

+   `QPainter::drawArc()`: 该方法用于从指定的中心位置绘制弧，介于两个指定的角度之间，并具有指定的半径

+   `QPainter::drawText()`: 该方法用于以指定的字体样式、颜色和大小绘制文本

为了实际显示图形所需的不同类和方法，让我们遵循一些操作步骤。

# 显示鼠标坐标

要用鼠标绘制任何形状，您需要知道鼠标按钮的点击位置，鼠标拖动到何处以及鼠标按钮释放的位置。只有在知道鼠标按钮点击的坐标后，才能执行命令来绘制不同的形状。在这个教程中，我们将学习在表单上显示鼠标移动到的*x*和*y*坐标。

# 操作步骤...

在这个教程中，我们将跟踪鼠标移动，并在表单上显示鼠标移动的*x*和*y*坐标。因此，在这个应用程序中，我们将使用两个 Label 小部件，一个用于显示消息，另一个用于显示鼠标坐标。创建此应用程序的完整步骤如下：

1.  让我们创建一个基于没有按钮的对话框模板的应用程序。

1.  通过将两个 Label 小部件拖放到表单上，向表单添加两个`QLabel`小部件。

1.  将第一个 Label 小部件的文本属性设置为`This app will display x,y coordinates where mouse is moved on`。

1.  删除第二个 Label 小部件的文本属性，因为它的文本属性将通过代码设置。

1.  将应用程序保存为`demoMousetrack.ui`。

表单现在将显示如下截图所示：

![](img/42107aa5-261d-42aa-8327-ed81cf97bbae.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。使用`pyuic5`实用程序将 XML 文件转换为 Python 代码。书籍的源代码包中可以看到生成的 Python 脚本`demoMousetrack.py`。

1.  将`demoMousetrack.py`脚本视为头文件，并将其从中调用用户界面设计的文件中导入。

1.  创建另一个名为`callMouseTrack.pyw`的 Python 文件，并将`demoMousetrack.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoMousetrack import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.setMouseTracking(True)
        self.ui.setupUi(self)
        self.show()
    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()
        text = "x: {0}, y: {1}".format(x, y)
        self.ui.label.setText(text)
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

为了使应用程序跟踪鼠标，使用了一个方法`setMouseTracking(True)`。这个方法将感应鼠标移动，每当鼠标移动时，它将调用`mouseMoveEvent()`方法。在`mouseMoveEvent()`中，对`event`对象调用`x`和`y`方法以获取鼠标位置的*x*和*y*坐标值。*x*和*y*坐标分别赋给`x`和`y`变量。通过标签小部件以所需的格式显示*x*和*y*坐标的值。

运行应用程序时，将会收到一条消息，提示鼠标移动时将显示其*x*和*y*坐标值。当您在表单上移动鼠标时，鼠标位置的*x*和*y*坐标将通过第二个标签小部件显示，如下截图所示：

![](img/da24bcb9-6682-437e-96ac-72ca5f5d78f5.png)

# 显示鼠标按下和释放的坐标

在这个示例中，我们将学习显示鼠标按下的*x*和*y*坐标，以及鼠标释放的坐标。

# 如何做...

两种方法，`mousePressEvent()`和`mouseReleaseEvent()`，在这个示例中将起到重要作用。当鼠标按下时，`mousePressEvent()`方法将自动被调用，并在鼠标按下事件发生时显示*x*和*y*坐标。同样，`mouseReleaseEvent()`方法将在鼠标按钮释放时自动被调用。两个标签小部件将用于显示鼠标按下和释放的坐标。以下是创建这样一个应用程序的步骤：

1.  让我们基于没有按钮的对话框模板创建一个应用程序。

1.  通过将三个标签小部件拖放到表单上，向表单添加三个`QLabel`小部件。

1.  将第一个标签小部件的文本属性设置为`显示鼠标按下和释放的*x*和*y*坐标`。

1.  删除第二个和第三个标签小部件的文本属性，因为它们的文本属性将通过代码设置。

1.  将第二个标签小部件的 objectName 属性设置为`labelPress`，因为它将用于显示鼠标按下的位置的*x*和*y*坐标。

1.  将第三个标签小部件的 objectName 属性设置为`labelRelease`，因为它将用于显示鼠标释放的位置的*x*和*y*坐标。

1.  将应用程序保存为`demoMouseClicks.ui`。

表单现在将显示如下截图所示：

![](img/349370c8-045c-4665-a596-a08ee0f4ca45.png)

使用 Qt Designer 创建的用户界面存储在一个`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。使用`pyuic5`实用程序将 XML 文件转换为 Python 代码。生成的 Python 脚本`demoMouseClicks.py`可以在本书的源代码包中看到。

1.  将`demoMouseClicks.py`脚本视为头文件，并将其导入到您将调用其用户界面设计的文件中。

1.  创建另一个名为`callMouseClickCoordinates.pyw`的 Python 文件，并将`demoMouseClicks.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoMouseClicks import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.show()
    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            x = event.x()
            y = event.y()
            text = "x: {0}, y: {1}".format(x, y)
            self.ui.labelPress.setText('Mouse button pressed at 
            '+text)
    def mouseReleaseEvent(self, event):
        x = event.x()
        y = event.y()
        text = "x: {0}, y: {1}".format(x, y)
        self.ui.labelRelease.setText('Mouse button released at 
        '+text)
        self.update()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

当单击鼠标时，会自动调用两个方法。当按下鼠标按钮时，会调用`mousePressEvent()`方法，当释放鼠标按钮时，会调用`mouseReleaseEvent()`方法。为了显示鼠标点击和释放的位置的*x*和*y*坐标，我们使用这两种方法。在这两种方法中，我们只需在`event`对象上调用`x()`和`y()`方法来获取鼠标位置的*x*和*y*坐标值。获取的*x*和*y*值将分别赋给`x`和`y`变量。`x`和`y`变量中的值将以所需的格式进行格式化，并通过两个 Label 部件显示出来。

运行应用程序时，将会收到一个消息，显示鼠标按下和释放的位置的*x*和*y*坐标。

当你按下鼠标按钮并释放它时，鼠标按下和释放的位置的*x*和*y*坐标将通过两个 Label 部件显示出来，如下截图所示：

![](img/d2b7047e-aabd-43b8-9b9a-dff3be87f80e.png)

# 显示鼠标点击的点

在这个教程中，我们将学习在窗体上显示鼠标点击的点。这里的点指的是一个小圆点。也就是说，无论用户在哪里按下鼠标，都会在那个坐标处出现一个小圆点。你还将学会定义小圆点的大小。

# 如何做...

在这个教程中，将使用`mousePressEvent()`方法，因为它是在窗体上按下鼠标时自动调用的方法。在`mousePressEvent()`方法中，我们将执行命令来显示所需大小的点或圆点。以下是了解如何在单击鼠标的地方在窗体上显示一个点或圆点的步骤：

1.  让我们基于没有按钮的对话框模板创建一个应用程序。

1.  通过拖放 Label 部件将`QLabel`部件添加到窗体中。

1.  将 Label 部件的文本属性设置为“单击鼠标以显示一个点的位置”。

1.  将应用程序保存为`demoDrawDot.ui`。

窗体现在将显示如下截图所示：

![](img/ec7eda2f-85a9-454a-80a7-41c9fc88a2ea.png)

使用 Qt Designer 创建的用户界面存储在一个`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。使用`pyuic5`工具将 XML 文件转换为 Python 代码。生成的 Python 脚本`demoDrawDot.py`可以在本书的源代码包中找到。

1.  将`demoDrawDot.py`脚本视为头文件，并将其从用户界面设计中调用的文件中导入。

1.  创建另一个名为`callDrawDot.pyw`的 Python 文件，并将`demoDrawDot.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt
from demoDrawDot import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.pos1 = [0,0]
        self.show()
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        pen = QPen(Qt.black, 5)
        qp.setPen(pen)
        qp.drawPoint(self.pos1[0], self.pos1[1])
        qp.end()
    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.pos1[0], self.pos1[1] = event.pos().x(), 
            event.pos().y()
            self.update()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

因为我们想要显示鼠标点击的点，所以使用了`mousePressEvent()`方法。在`mousePressEvent()`方法中，对`event`对象调用`pos().x()`和`pos().y()`方法来获取*x*和*y*坐标的位置，并将它们分配给`pos1`数组的`0`和`1`元素。也就是说，`pos1`数组被初始化为鼠标点击的*x*和*y*坐标值。在初始化`pos1`数组之后，调用`self.update()`方法来调用`paintEvent()`方法。

在`paintEvent()`方法中，通过名称为`qp`的`QPainter`类对象定义了一个对象。通过名称为 pen 的`QPen`类对象设置了笔的粗细和颜色。最后，通过在`pos1`数组中定义的位置调用`drawPoint()`方法显示一个点。

运行应用程序时，将会收到一条消息，指出鼠标按钮点击的地方将显示一个点。当您点击鼠标时，一个点将出现在那个位置，如下截图所示：

![](img/ba0d76c7-9948-4b42-9843-1fd8aaf1b17c.png)

# 在两次鼠标点击之间画一条线

在这个示例中，我们将学习如何在两个点之间显示一条线，从鼠标按钮点击的地方到鼠标按钮释放的地方。这个示例的重点是理解如何处理鼠标按下和释放事件，如何访问鼠标按钮点击和释放的*x*和*y*坐标，以及如何在鼠标按钮点击的位置和鼠标按钮释放的位置之间绘制一条线。

# 如何操作...

这个示例中的主要方法是`mousePressEvent()`、`mouseReleaseEvent()`和`paintEvent()`。`mousePressEvent()`和`mouseReleaseEvent()`方法在鼠标按钮被点击或释放时自动执行。这两种方法将用于访问鼠标按钮被点击和释放的*x*和*y*坐标。最后，`paintEvent()`方法用于在`mousePressEvent()`和`mouseReleaseEvent()`方法提供的坐标之间绘制一条线。以下是创建此应用程序的逐步过程：

1.  让我们基于没有按钮的对话框模板创建一个应用程序。

1.  通过拖放标签小部件到表单上，向表单添加一个`QLabel`小部件。

1.  将标签小部件的文本属性设置为`单击鼠标并拖动以绘制所需大小的线`。

1.  将应用程序保存为`demoDrawLine.ui`。

表单现在将显示如下截图所示：

![](img/14d862a2-330c-4b2a-a5a9-c82308f27209.png)

使用 Qt Designer 创建的用户界面存储在一个`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。`pyuic5`实用程序用于将 XML 文件转换为 Python 代码。生成的 Python 脚本`demoDrawLine.py`可以在书的源代码包中看到。

1.  将`demoDrawLine.py`脚本视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callDrawLine.pyw`的 Python 文件，并将`demoDrawLine.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtGui import QPainter
from demoDrawLine import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.pos1 = [0,0]
        self.pos2 = [0,0]
        self.show()
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        qp.drawLine(self.pos1[0], self.pos1[1], self.pos2[0], 
        self.pos2[1])
        qp.end()
    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.pos1[0], self.pos1[1] = event.pos().x(), 
            event.pos().y()
    def mouseReleaseEvent(self, event):
            self.pos2[0], self.pos2[1] = event.pos().x(), 
            event.pos().y()
            self.update()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

由于我们想要在鼠标按钮点击和释放的位置之间显示一条线，我们将使用两种方法，`mousePressEvent()`和`mouseReleaseEvent()`。顾名思义，`mousePressEvent()`方法在鼠标按钮按下时自动调用。同样，`mouseReleaseEvent()`方法在鼠标按钮释放时自动调用。在这两种方法中，我们将简单地保存鼠标按钮点击和释放的*x*和*y*坐标的值。在这个应用程序中定义了两个数组`pos1`和`pos2`，其中`pos1`存储鼠标按钮点击的位置的*x*和*y*坐标，`pos2`数组存储鼠标按钮释放的位置的*x*和*y*坐标。一旦鼠标按钮点击和释放的位置的*x*和*y*坐标被分配给`pos1`和`pos2`数组，`self.update()`方法在`mouseReleaseEvent()`方法中被调用以调用`paintEvent()`方法。在`paintEvent()`方法中，调用`drawLine()`方法，并将存储在`pos1`和`pos2`数组中的*x*和*y*坐标传递给它，以在鼠标按下和鼠标释放的位置之间绘制一条线。

运行应用程序时，您将收到一条消息，要求在需要绘制线条的位置之间单击并拖动鼠标按钮。因此，单击鼠标按钮并保持鼠标按钮按下，将其拖动到所需位置，然后释放鼠标按钮。将在鼠标按钮单击和释放的位置之间绘制一条线，如下面的屏幕截图所示：

![](img/af765355-e32c-4cbc-8c68-130b778aa710.png)

# 绘制不同类型的线条

在本示例中，我们将学习在两个点之间显示不同类型的线条，从鼠标单击位置到释放鼠标按钮的位置。用户将显示不同的线条类型可供选择，例如实线、虚线、虚线点线等。线条将以所选线条类型绘制。

# 如何做...

用于定义绘制形状的笔的大小或厚度的是`QPen`类。在这个示例中，使用`QPen`类的`setStyle()`方法来定义线条的样式。以下是绘制不同样式线条的逐步过程：

1.  让我们基于没有按钮的对话框模板创建一个应用程序。

1.  通过在表单上拖放一个标签小部件来向表单添加一个`QLabel`小部件。

1.  通过拖放一个列表小部件项目在表单上添加一个`QListWidget`小部件。

1.  将标签小部件的文本属性设置为`从列表中选择样式，然后单击并拖动以绘制一条线`。

1.  将应用程序保存为`demoDrawDiffLine.ui`。

1.  列表小部件将用于显示不同类型的线条，因此右键单击列表小部件并选择“编辑项目”选项以向列表小部件添加几种线条类型。单击打开的对话框框底部的+（加号）按钮，并添加几种线条类型，如下面的屏幕截图所示：

![](img/2d9c24db-f1d3-4c5b-86f8-946e1db3c09b.png)

1.  将列表小部件项目的 objectName 属性设置为`listWidgetLineType`。

表单现在将显示如下屏幕截图所示：

![](img/3da8852a-3a65-41e2-94d6-9a50803b8a8f.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。`pyuic5`实用程序用于将 XML 文件转换为 Python 代码。生成的 Python 脚本`demoDrawDiffLine.py`可以在本书的源代码包中看到。

1.  将`demoDrawDiffLine.py`脚本视为头文件，并将其导入到您将调用其用户界面设计的文件中。

1.  创建另一个名为`callDrawDiffLine.pyw`的 Python 文件，并将`demoDrawDiffLine.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt
from demoDrawDiffLine import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.lineType="SolidLine"
        self.pos1 = [0,0]
        self.pos2 = [0,0]
        self.show()
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        pen = QPen(Qt.black, 4)
        self.lineTypeFormat="Qt."+self.lineType
        if self.lineTypeFormat == "Qt.SolidLine":
            pen.setStyle(Qt.SolidLine)
            elif self.lineTypeFormat == "Qt.DashLine":
            pen.setStyle(Qt.DashLine)
            elif self.lineTypeFormat =="Qt.DashDotLine":
                pen.setStyle(Qt.DashDotLine)
            elif self.lineTypeFormat =="Qt.DotLine":
                pen.setStyle(Qt.DotLine)
            elif self.lineTypeFormat =="Qt.DashDotDotLine":
                pen.setStyle(Qt.DashDotDotLine)
                qp.setPen(pen)
                qp.drawLine(self.pos1[0], self.pos1[1], 
                self.pos2[0], self.pos2[1])
                qp.end()
    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.pos1[0], self.pos1[1] = event.pos().x(), 
            event.pos().y()
    def mouseReleaseEvent(self, event):
        self.lineType=self.ui.listWidgetLineType.currentItem()
        .text()
        self.pos2[0], self.pos2[1] = event.pos().x(), 
        event.pos().y()
        self.update()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

必须在鼠标按下和鼠标释放位置之间绘制一条线，因此我们将在此应用程序中使用两种方法，`mousePressEvent()`和`mouseReleaseEvent()`。当单击鼠标左键时，`mousePressEvent()`方法会自动调用。同样，当鼠标按钮释放时，`mouseReleaseEvent()`方法会自动调用。

在这两种方法中，我们将保存鼠标单击和释放时的*x*和*y*坐标的值。在这个应用程序中定义了两个数组`pos1`和`pos2`，其中`pos1`存储鼠标单击的位置的*x*和*y*坐标，`pos2`数组存储鼠标释放的位置的*x*和*y*坐标。在`mouseReleaseEvent()`方法中，我们从列表小部件中获取用户选择的线类型，并将所选的线类型分配给`lineType`变量。此外，在`mouseReleaseEvent()`方法中调用了`self.update()`方法来调用`paintEvent()`方法。在`paintEvent()`方法中，您定义了一个宽度为`4`像素的画笔，并将其分配为黑色。此外，您为画笔分配了一个与用户从列表小部件中选择的线类型相匹配的样式。最后，调用`drawLine()`方法，并将存储在`pos1`和`pos2`数组中的*x*和*y*坐标传递给它，以在鼠标按下和鼠标释放位置之间绘制一条线。所选的线将以从列表小部件中选择的样式显示。

运行应用程序时，您将收到一条消息，要求从列表中选择线类型，并在需要线的位置之间单击并拖动鼠标按钮。因此，在选择所需的线类型后，单击鼠标按钮并保持鼠标按钮按下，将其拖动到所需位置，然后释放鼠标按钮。将在鼠标按钮单击和释放的位置之间绘制一条线，以所选的样式显示在列表中。以下截图显示了不同类型的线：

![](img/9b53d16b-2706-4bad-b105-9f86f5163ee6.png)

# 绘制所需大小的圆

在这个示例中，我们将学习如何绘制一个圆。用户将点击并拖动鼠标来定义圆的直径，圆将根据用户指定的直径进行绘制。

# 如何做...

一个圆实际上就是从 0 到 360 度绘制的弧。弧的长度，或者可以说是圆的直径，由鼠标按下事件和鼠标释放事件的距离确定。在鼠标按下事件到鼠标释放事件之间内部定义了一个矩形，并且圆在该矩形内绘制。以下是创建此应用程序的完整步骤：

1.  让我们创建一个基于无按钮对话框模板的应用程序。

1.  通过拖放一个标签小部件到表单上，向表单添加一个`QLabel`小部件。

1.  将标签小部件的文本属性设置为`单击鼠标并拖动以绘制所需大小的圆`。

1.  将应用程序保存为`demoDrawCircle.ui`。表单现在将显示如下截图所示：

![](img/53947d76-a10c-4516-9a77-6e50d2412f11.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，它是一个 XML 文件。通过应用`pyuic5`实用程序将 XML 文件转换为 Python 代码。您可以在本书的源代码包中找到生成的 Python 代码`demoDrawCircle.py`。

1.  将`demoDrawCircle.py`脚本视为头文件，并将其导入到您将调用其用户界面设计的文件中。

1.  创建另一个名为`callDrawCircle.pyw`的 Python 文件，并将`demoDrawCircle.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtGui import QPainter
from demoDrawCircle import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.pos1 = [0,0]
        self.pos2 = [0,0]
        self.show()
    def paintEvent(self, event):
        width = self.pos2[0]-self.pos1[0]
        height = self.pos2[1] - self.pos1[1]
        qp = QPainter()
        qp.begin(self)
        rect = QtCore.QRect(self.pos1[0], self.pos1[1], width, 
        height)
        startAngle = 0
        arcLength = 360 *16
        qp.drawArc(rect, startAngle, arcLength)
        qp.end()
    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.pos1[0], self.pos1[1] = event.pos().x(), 
            event.pos().y()
    def mouseReleaseEvent(self, event):
        self.pos2[0], self.pos2[1] = event.pos().x(), 
        event.pos().y()
        self.update()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

为了在鼠标按下和释放的位置之间绘制直径定义的圆，我们将使用两种方法，`mousePressEvent()`和`mouseReleaseEvent()`。当鼠标按钮按下时，`mousePressEvent()`方法会自动调用，当鼠标按钮释放时，`mouseReleaseEvent()`方法会自动调用。在这两种方法中，我们将简单地保存鼠标按下和释放的*x*和*y*坐标的值。定义了两个数组`pos1`和`pos2`，其中`pos1`数组存储鼠标按下的位置的*x*和*y*坐标，`pos2`数组存储鼠标释放的位置的*x*和*y*坐标。在`mouseReleaseEvent()`方法中调用的`self.update()`方法将调用`paintEvent()`方法。在`paintEvent()`方法中，通过找到鼠标按下和鼠标释放位置的*x*坐标之间的差异来计算矩形的宽度。类似地，通过找到鼠标按下和鼠标释放事件的*y*坐标之间的差异来计算矩形的高度。

圆的大小将等于矩形的宽度和高度，也就是说，圆将在用户用鼠标指定的边界内创建。 

此外，在`paintEvent()`方法中，调用了`drawArc()`方法，并将矩形、弧的起始角度和弧的长度传递给它。起始角度被指定为`0`。

运行应用程序时，会收到一条消息，要求点击并拖动鼠标按钮以定义要绘制的圆的直径。因此，点击鼠标按钮并保持鼠标按钮按下，将其拖动到所需位置，然后释放鼠标按钮。将在鼠标按下和释放的位置之间绘制一个圆，如下截图所示：

![](img/361dd5cf-84ec-4f7e-91da-d01e1547202f.png)

# 在两次鼠标点击之间绘制一个矩形

在这个示例中，我们将学习在表单上显示鼠标按下和释放的两个点之间的矩形。

# 如何做...

这是一个非常简单的应用程序，其中使用`mousePressEvent()`和`mouseReleaseEvent()`方法来分别找到鼠标按下和释放的位置的*x*和*y*坐标。然后，调用`drawRect()`方法来从鼠标按下的位置到鼠标释放的位置绘制矩形。创建此应用程序的逐步过程如下：

1.  让我们基于没有按钮的对话框模板创建一个应用程序。

1.  在表单上通过拖放标签小部件添加一个`QLabel`小部件。

1.  将标签小部件的文本属性设置为`点击鼠标并拖动以绘制所需大小的矩形`。

1.  将应用程序保存为`demoDrawRectangle.ui`。表单现在将显示如下截图所示：

![](img/23480372-2821-4af3-abba-cb0ad86dc49b.png)

使用 Qt Designer 创建的用户界面存储在一个`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。使用`pyuic5`工具将 XML 文件转换为 Python 代码。生成的 Python 脚本`demoDrawRectangle.py`可以在本书的源代码包中找到。

1.  将`demoDrawRectangle.py`脚本视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callDrawRectangle.pyw`的 Python 文件，并将`demoDrawRectangle.py`的代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtGui import QPainter
from demoDrawRectangle import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.pos1 = [0,0]
        self.pos2 = [0,0]
        self.show()
    def paintEvent(self, event):
        width = self.pos2[0]-self.pos1[0]
        height = self.pos2[1] - self.pos1[1]
        qp = QPainter()
        qp.begin(self)
        qp.drawRect(self.pos1[0], self.pos1[1], width, height)
        qp.end()
    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.pos1[0], self.pos1[1] = event.pos().x(), 
            event.pos().y()
    def mouseReleaseEvent(self, event):
        self.pos2[0], self.pos2[1] = event.pos().x(), 
        event.pos().y()
        self.update()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

为了在鼠标按钮按下和释放的位置之间绘制矩形，我们将使用两种方法，`mousePressEvent()`和`mouseReleaseEvent()`。当鼠标按钮按下时，`mousePressEvent()`方法会自动被调用，当鼠标按钮释放时，`mouseReleaseEvent()`方法会自动被调用。在这两种方法中，我们将简单地保存鼠标按钮单击和释放时的*x*和*y*坐标的值。定义了两个数组`pos1`和`pos2`，其中`pos1`数组存储鼠标按钮单击的位置的*x*和*y*坐标，`pos2`数组存储鼠标按钮释放的位置的*x*和*y*坐标。在`mouseReleaseEvent()`方法中调用的`self.update()`方法将调用`paintEvent()`方法。在`paintEvent()`方法中，矩形的宽度通过找到鼠标按下和鼠标释放位置的*x*坐标之间的差异来计算。同样，矩形的高度通过找到鼠标按下和鼠标释放事件的*y*坐标之间的差异来计算。

此外，在`paintEvent()`方法中，调用了`drawRect()`方法，并将存储在`pos1`数组中的*x*和*y*坐标传递给它。此外，矩形的宽度和高度也传递给`drawRect()`方法，以在鼠标按下和鼠标释放位置之间绘制矩形。

运行应用程序时，您将收到一条消息，要求单击并拖动鼠标按钮以在所需位置之间绘制矩形。因此，单击鼠标按钮并保持鼠标按钮按下，将其拖动到所需位置，然后释放鼠标按钮。

在鼠标按钮单击和释放的位置之间将绘制一个矩形，如下截图所示：

![](img/46b5d873-feca-40d4-a47d-8cd1a9adf217.png)

# 以所需的字体和大小绘制文本

在这个教程中，我们将学习如何以特定的字体和特定的字体大小绘制文本。在这个教程中将需要四个小部件，如文本编辑，列表小部件，组合框和按钮。文本编辑小部件将用于输入用户想要以所需字体和大小显示的文本。列表小部件框将显示用户可以从中选择的不同字体名称。组合框小部件将显示用户可以选择以定义文本大小的字体大小。按钮小部件将启动操作，也就是说，单击按钮后，文本编辑小部件中输入的文本将以所选字体和大小显示。

# 如何操作...

`QPainter`类是本教程的重点。`QPainter`类的`setFont()`和`drawText()`方法将在本教程中使用。`setFont()`方法将被调用以设置用户选择的字体样式和字体大小，`drawText()`方法将以指定的字体样式和大小绘制用户在文本编辑小部件中编写的文本。以下是逐步学习这些方法如何使用的过程：

1.  让我们创建一个基于无按钮对话框模板的应用程序。

1.  将`QLabel`，`QTextEdit`，`QListWidget`，`QComboBox`和`QPushButton`小部件通过拖放标签小部件，文本编辑小部件，列表小部件框，组合框小部件和按钮小部件添加到表单中。

1.  将标签小部件的文本属性设置为“在最左边的框中输入一些文本，选择字体和大小，然后单击绘制文本按钮”。

1.  列表小部件框将用于显示不同的字体，因此右键单击列表小部件框，选择“编辑项目”选项，向列表小部件框添加一些字体名称。单击打开的对话框底部的+（加号）按钮，并添加一些字体名称，如下截图所示：

![](img/4af8bf5d-9571-4dfc-adfc-3ff8071a0010.png)

1.  组合框小部件将用于显示不同的字体大小，因此我们需要向组合框小部件添加一些字体大小。右键单击组合框小部件，然后选择“编辑项目”选项。

1.  单击打开的对话框框底部的+（加号）按钮，并添加一些字体大小，如下面的屏幕截图所示：

![](img/4a210911-d3db-4235-b58c-be82fae13bb0.png)

1.  将推送按钮小部件的文本属性设置为“绘制文本”。

1.  将列表小部件框的 objectName 属性设置为`listWidgetFont`。

1.  将组合框小部件的 objectName 属性设置为`comboBoxFontSize`。

1.  将推送按钮小部件的 objectName 属性设置为 pushButtonDrawText。

1.  将应用程序保存为`demoDrawText.ui`。

表单现在将显示如下的屏幕截图：

![](img/685adb85-c7f3-40db-b4ae-5ad6090a29ea.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，它是一个 XML 文件。通过应用`pyuic5`实用程序将 XML 文件转换为 Python 代码。您可以在本书的源代码包中找到生成的 Python 代码`demoDrawText.py`。

1.  将`demoDrawText.py`脚本视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callDrawText.pyw`的 Python 文件，并将`demoDrawText.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt
from demoDrawText import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonDrawText.clicked.connect(self.
        dispText)
        self.textToDraw=""
        self.fontName="Courier New"
        self.fontSize=5
        self.show()
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        qp.setPen(QColor(168, 34, 3))
        qp.setFont(QFont(self.fontName, self.fontSize))
        qp.drawText(event.rect(), Qt.AlignCenter, 
        self.textToDraw)
        qp.end()
    def dispText(self):
        self.fontName=self.ui.listWidgetFont.currentItem().
        text()
        self.fontSize=int(self.ui.comboBoxFontSize.itemText(
        self.ui.comboBoxFontSize.currentIndex()))
        self.textToDraw=self.ui.textEdit.toPlainText()
        self.update()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的…

推送按钮小部件的 click()事件连接到`dispText()`方法，也就是说，每当点击推送按钮时，将调用`dispText()`方法。

在`dispText()`方法中，访问从列表小部件框中选择的字体名称，并将其分配给`fontName`变量。此外，访问从组合框中选择的字体大小，并将其分配给`fontSize`变量。除此之外，获取并分配在文本编辑小部件中编写的文本给`textToDraw`变量。最后，调用`self.update()`方法；它将调用`paintEvent()`方法。

在`paintEvent()`方法中，调用`drawText()`方法，将以`fontName`变量分配的字体样式和`fontSize`变量中指定的字体大小绘制在文本编辑小部件中编写的文本。运行应用程序后，您将在极左边看到一个文本编辑小部件，字体名称显示在列表小部件框中，字体大小通过组合框小部件显示。您需要在文本编辑小部件中输入一些文本，从列表小部件框中选择一个字体样式，从组合框小部件中选择一个字体大小，然后单击“绘制文本”按钮。单击“绘制文本”按钮后，文本编辑小部件中编写的文本将以所选字体和所选字体大小显示，如下面的屏幕截图所示：

![](img/b50a34fb-8383-4fec-b359-c0a6d2bea7f1.png)

# 创建一个显示不同图形工具的工具栏

在这个示例中，我们将学习创建一个显示三个工具栏按钮的工具栏。这三个工具栏按钮显示线条、圆圈和矩形的图标。当用户从工具栏中单击线条工具栏按钮时，他/她可以在表单上单击并拖动鼠标以在两个鼠标位置之间绘制一条线。类似地，通过单击圆圈工具栏按钮，用户可以通过单击和拖动鼠标在表单上绘制一个圆圈。

# 如何做…

这个示例的重点是帮助您理解如何通过工具栏向用户提供应用程序中经常使用的命令，使它们易于访问和使用。您将学习创建工具栏按钮，定义它们的快捷键以及它们的图标。为工具栏按钮定义图标，您将学习创建和使用资源文件。逐步清晰地解释了每个工具栏按钮的创建和执行过程：

1.  让我们创建一个新应用程序来了解创建工具栏涉及的步骤。

1.  启动 Qt Designer 并创建一个基于主窗口的应用程序。您将获得一个带有默认菜单栏的新应用程序。

1.  您可以右键单击菜单栏，然后从弹出的快捷菜单中选择“删除菜单栏”选项来删除菜单栏。

1.  要添加工具栏，右键单击“主窗口”模板，然后从上下文菜单中选择“添加工具栏”。将在菜单栏下方添加一个空白工具栏，如下截图所示：

![](img/cf2901c2-01df-4d2a-88ff-69dd27d17455.png)

我们想要创建一个具有三个工具栏按钮的工具栏，分别是线条、圆形和矩形。由于这三个工具栏按钮将代表三个图标图像，我们假设已经有了图标文件，即扩展名为`.ico`的线条、圆形和矩形文件。

1.  要将工具添加到工具栏中，在“操作编辑器”框中创建一个操作；工具栏中的每个工具栏按钮都由一个操作表示。操作编辑器框通常位于属性编辑器窗口下方。

1.  如果“操作编辑器”窗口不可见，请从“视图”菜单中选择“操作编辑器”。操作编辑器窗口将显示如下：

![](img/cf7c369d-b4c5-49f3-b5e1-441e3e6143de.png)

1.  在“操作编辑器”窗口中，选择“新建”按钮，为第一个工具栏按钮创建一个操作。您将获得一个对话框，以输入新操作的详细信息。

1.  在文本框中，指定操作的名称为“Circle”。

1.  在“对象名称”框中，操作对象的名称将自动显示，前缀为文本“action”。

1.  在“工具提示”框中，输入任何描述性文本。

1.  在“快捷方式”框中，按下*Ctrl* + *C*字符，将`Ctrl + C`分配为绘制圆形的快捷键。

1.  图标下拉列表显示两个选项，选择资源…和选择文件。

1.  您可以通过单击“选择文件…”选项或从资源文件中为操作分配图标图像：

![](img/e086d3f2-763a-4856-9cbb-e1dc5cd6ba7e.png)

您可以在资源文件中选择多个图标，然后该资源文件可以在不同的应用程序中使用。

1.  选择“选择资源…”选项。您将获得“选择资源”对话框，如下截图所示：

![](img/276ca224-3288-4f68-b55a-b73768c3e92c.png)

由于尚未创建任何资源，对话框为空。您会在顶部看到两个图标。第一个图标代表编辑资源，第二个图标代表重新加载。单击“编辑资源”图标后，您将看到如下对话框：

![](img/990531d8-648a-46ec-aacb-cc2ef781da4b.png)

现在让我们看看如何通过以下步骤创建资源文件：

1.  第一步是创建一个资源文件或加载一个现有的资源文件。底部的前三个图标分别代表新资源文件、编辑资源文件和删除。

1.  单击“新建资源文件”图标。将提示您指定资源文件的名称。

1.  让我们将新资源文件命名为`iconresource`。该文件将以扩展名`.qrc`保存。

1.  下一步是向资源文件添加前缀。前缀/路径窗格下的三个图标分别是添加前缀、添加文件和删除。

1.  单击“添加前缀”选项，然后将提示您输入前缀名称。

1.  将前缀输入为“Graphics”。添加前缀后，我们准备向资源文件添加我们的三个图标，圆形、矩形和线条。请记住，我们有三个扩展名为`.ico`的图标文件。

1.  单击“添加文件”选项以添加图标。单击“添加文件”选项后，将要求您浏览到驱动器/目录并选择图标文件。

1.  逐个选择三个图标文件。添加完三个图标后，编辑资源对话框将显示如下：

![](img/4a5b9237-59a2-458c-9f2a-49c9e5d6fc7c.png)

1.  单击“确定”按钮后，资源文件将显示三个可供选择的图标。

1.  由于我们想要为圆形操作分配一个图标，因此单击圆形图标，然后单击“确定”按钮：

![](img/136162d0-9399-4687-99fa-6c23c97afe3a.png)

所选的圆形图标将被分配给 actionCircle。

1.  类似地，为矩形和线条工具栏按钮创建另外两个操作，`actionRectangle`和`actionLine`。添加了这三个操作后，操作编辑器窗口将显示如下：

![](img/03c156e2-1e98-4390-92d0-9b0d0fe8d940.png)

1.  要在工具栏中显示工具栏按钮，从操作编辑器窗口中单击一个操作，并保持按住状态，将其拖动到工具栏中。

1.  将应用程序保存为`demoToolBars.ui`。

将三个操作拖动到工具栏后，工具栏将显示如下：

![](img/314fa220-5b14-4a10-bafb-314fc5d3ce72.png)

`pyuic5`命令行实用程序将把`.ui`（XML）文件转换为 Python 代码，生成的代码将被命名为`demoToolBars.py`。您可以在本书的源代码包中找到`demoToolBars.py`脚本。我们创建的`iconresource.qrc`文件必须在我们继续之前转换为 Python 格式。以下命令行将资源文件转换为 Python 脚本：

```py
pyrcc5 iconresource.qrc -o iconresource_rc.py
```

1.  创建一个名为`callToolBars.pyw`的 Python 脚本，导入代码`demoToolBar.py`，以调用工具栏并绘制从工具栏中选择的图形。脚本文件将如下所示：

```py
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPainter
from demoToolBars import *

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.pos1 = [0,0]
        self.pos2 = [0,0]
        self.toDraw=""
        self.ui.actionCircle.triggered.connect(self.drawCircle)
        self.ui.actionRectangle.triggered.connect(self.
        drawRectangle)
        self.ui.actionLine.triggered.connect(self.drawLine)
        self.show()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.toDraw=="rectangle":
            width = self.pos2[0]-self.pos1[0]
            height = self.pos2[1] - self.pos1[1]    
            qp.drawRect(self.pos1[0], self.pos1[1], width, 
            height)
        if self.toDraw=="line":
            qp.drawLine(self.pos1[0], self.pos1[1], 
            self.pos2[0], self.pos2[1])
        if self.toDraw=="circle":
            width = self.pos2[0]-self.pos1[0]
            height = self.pos2[1] - self.pos1[1]          
            rect = QtCore.QRect(self.pos1[0], self.pos1[1], 
            width, height)
            startAngle = 0
            arcLength = 360 *16
            qp.drawArc(rect, startAngle, arcLength)     
            qp.end()

    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.pos1[0], self.pos1[1] = event.pos().x(), 
            event.pos().y()

    def mouseReleaseEvent(self, event):
        self.pos2[0], self.pos2[1] = event.pos().x(), 
        event.pos().y()   
        self.update()

    def drawCircle(self):
        self.toDraw="circle"

    def drawRectangle(self):
        self.toDraw="rectangle"

    def drawLine(self):
        self.toDraw="line"

app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())
```

# 它是如何工作的...

每个工具栏按钮的操作的 triggered()信号都连接到相应的方法。actionCircle 工具栏按钮的 triggered()信号连接到`drawCircle()`方法，因此每当从工具栏中选择圆形工具栏按钮时，将调用`drawCircle()`方法。类似地，`actionRectangle`和`actionLine`的 triggered()信号分别连接到`drawRectangle()`和`drawLine()`方法。在`drawCircle()`方法中，一个变量`toDraw`被赋予一个字符串`circle`。`toDraw`变量将用于确定在`paintEvent()`方法中要绘制的图形。`toDraw`变量可以分配任何三个字符串之一，`line`、`circle`或`rectangle`。在`toDraw`变量的值上应用条件分支，相应地，将调用绘制线条、矩形或圆形的方法。

绘制线条、圆形或矩形的大小由鼠标点击确定；用户需要在窗体上单击鼠标，拖动鼠标并释放它到想要绘制线条、圆形或矩形的位置。换句话说，线条的长度、矩形的宽度和高度以及圆形的直径将由鼠标确定。

使用`pos1`和`pos2`两个数组来存储鼠标单击位置和鼠标释放位置的*x*和*y*坐标。*x*和*y*坐标值通过`mousePressEvent()`和`mouseReleaseEvent()`两种方法分配给`pos1`和`pos2`数组。当鼠标按钮被单击时，`mousePressEvent()`方法会自动调用，当鼠标按钮释放时，`mouseReleaseEvent()`方法会自动调用。

在`mouseReleaseEvent()`方法中，分配鼠标释放的位置的*x*和*y*坐标值后，调用`self.update()`方法来调用`paintEvent()`方法。在`paintEvent()`方法中，基于分配给`toDraw`变量的字符串进行分支。如果`toDraw`变量被分配了字符串`line`（由`drawLine()`方法），则将调用`QPainter`类的`drawLine()`方法来在两个鼠标位置之间绘制线。类似地，如果`toDraw`变量被分配了字符串`circle`（由`drawCircle()`方法），则将调用`QPainter`类的`drawArc()`方法来绘制由鼠标位置提供的直径的圆。如果`toDraw`变量由`drawRectangle()`方法分配了字符串`rectangle`，则将调用`QPainter`类的`drawRect()`方法来绘制由鼠标位置提供的宽度和高度的矩形。

运行应用程序后，您将在工具栏上找到三个工具栏按钮，圆形、矩形和线，如下截图所示（左）。点击圆形工具栏按钮，然后在表单上点击鼠标按钮，并保持鼠标按钮按下，拖动以定义圆的直径，然后释放鼠标按钮。将从鼠标按钮点击的位置到释放鼠标按钮的位置绘制一个圆（右）：

![](img/179df12e-a471-46a1-9a71-2178ca4063ce.png)

要绘制一个矩形，点击矩形工具，点击鼠标按钮在表单上的一个位置，并保持鼠标按钮按下，拖动以定义矩形的高度和宽度。释放鼠标按钮时，将在鼠标按下和鼠标释放的位置之间绘制一个矩形（左）。类似地，点击线工具栏按钮，然后在表单上点击鼠标按钮。保持鼠标按钮按下，将其拖动到要绘制线的位置。释放鼠标按钮时，将在鼠标按下和释放的位置之间绘制一条线（右）：

![](img/243bd53e-551a-4efa-a6ed-349fc4e2e1a6.png)

# 使用 Matplotlib 绘制一条线

在本示例中，我们将学习使用 Matplotlib 绘制通过特定*x*和*y*坐标的线。

Matplotlib 是一个 Python 2D 绘图库，使绘制线条、直方图、条形图等复杂的任务变得非常容易。该库不仅可以绘制图表，还提供了一个 API，可以在应用程序中嵌入图表。

# 准备工作

您可以使用以下语句安装 Matplotlib：

```py
pip install matplotlib
```

假设我们要绘制一条线，使用以下一组*x*和*y*坐标：

```py
x=10, y=20
x=20, y=40
x=30, y=60
```

在*x*轴上，`x`的值从`0`开始向右增加，在*y*轴上，`y`的值在底部为`0`，向上移动时增加。因为最后一对坐标是`30`，`60`，所以图表的最大`x`值为`30`，最大`y`值为`60`。

本示例中将使用`matplotlib.pyplot`的以下方法：

+   `title()`: 该方法用于设置图表的标题

+   `xlabel()`: 该方法用于在*x*轴上显示特定文本

+   `ylabel()`: 该方法用于在*y*轴上显示特定文本

+   `plot()`: 该方法用于在指定的*x*和*y*坐标处绘制图表

# 如何操作...

创建一个名为`demoPlotLine.py`的 Python 脚本，并在其中编写以下代码：

```py
import matplotlib.pyplot as graph
graph.title('Plotting a Line!')
graph.xlabel('x - axis')
graph.ylabel('y - axis')
x = [10,20,30]
y = [20,40,60]
graph.plot(x, y)
graph.show()
```

# 工作原理...

您在脚本中导入`matplotlib.pyplot`并将其命名为 graph。使用`title()`方法，您设置图表的标题。然后，调用`xlabel()`和`ylabel()`方法来定义*x*轴和*y*轴的文本。因为我们想要使用三组*x*和*y*坐标绘制一条线，所以定义了两个名为*x*和*y*的数组。在这两个数组中分别定义了我们想要绘制的三个*x*和*y*坐标值的值。调用`plot()`方法，并将这两个*x*和*y*数组传递给它，以使用这两个数组中定义的三个*x*和*y*坐标值绘制线。调用 show 方法显示绘图。

运行应用程序后，您会发现绘制了一条通过指定的*x*和*y*坐标的线。此外，图表将显示指定的标题，绘制一条线！除此之外，您还可以在*x*轴和*y*轴上看到指定的文本，如下截图所示：

![](img/b2c02ed5-8d94-4c5c-a52a-bec1ed0b908e.png)

# 使用 Matplotlib 绘制条形图

在本示例中，我们将学习使用 Matplotlib 绘制条形图，比较过去三年业务增长。您将提供 2016 年、2017 年和 2018 年的利润百分比，应用程序将显示代表过去三年利润百分比的条形图。

# 准备工作

假设组织过去三年的利润百分比如下：

+   2016 年：利润为 70%

+   2017 年：利润为 90%

+   2018 年：利润为 80%

您想显示代表利润百分比的条形，并沿*x*轴显示年份：2016 年、2017 年和 2018 年。沿*y*轴，您希望显示代表利润百分比的条形。 *y*轴上的`y`值将从底部的`0`开始增加，向顶部移动时增加，最大值为顶部的`100`。

本示例将使用`matplotlib.pyplot`的以下方法：

+   `title()`: 用于设置图表的标题

+   `bar()`: 从两个提供的数组绘制条形图；一个数组将代表*x*轴的数据，第二个数组将代表*y*轴的数据

+   `plot()`: 用于在指定的*x*和*y*坐标处绘图

# 如何做...

创建一个名为`demoPlotBars.py`的 Python 脚本，并在其中编写以下代码：

```py
import matplotlib.pyplot as graph
years = ['2016', '2017', '2018']
profit = [70, 90, 80]
graph.bar(years, profit)
graph.title('Growth in Business')
graph.plot(100)
graph.show()
```

# 工作原理...

您在脚本中导入`matplotlib.pyplot`并将其命名为 graph。您定义两个数组，years 和 profit，其中 years 数组将包含 2016 年、2017 年和 2018 年的数据，以表示我们想要比较利润的年份。类似地，profit 数组将包含代表过去三年利润百分比的值。然后，调用`bar()`方法，并将这两个数组 years 和 profit 传递给它，以显示比较过去三年利润的条形图。调用`title()`方法显示标题，业务增长。调用`plot()`方法指示*y*轴上的最大`y`值。最后，调用`show()`方法显示条形图。

运行应用程序后，您会发现绘制了一根条形图，显示了组织在过去三年的利润。 *x*轴显示年份，*y*轴显示利润百分比。此外，图表将显示指定的标题，业务增长，如下截图所示：

![](img/9b933396-c535-4c40-bc3f-d099b006d5db.png)
