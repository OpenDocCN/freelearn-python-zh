# 实现动画

在本章中，您将学习如何对给定的图形图像应用运动，从而实现动画。动画在解释任何机器、过程或系统的实际工作中起着重要作用。在本章中，我们将涵盖以下主题：

+   显示2D图形图片

+   点击按钮使球移动

+   制作一个弹跳的球

+   使球根据指定的曲线进行动画

# 介绍

要在Python中查看和管理2D图形项，我们需要使用一个名为`QGraphicsScene`的类。为了显示`QGraphicsScene`的内容，我们需要另一个名为`QGraphicsView`的类的帮助。基本上，`QGraphicsView`提供了一个可滚动的视口，用于显示`QGraphicsScene`的内容。`QGraphicsScene`充当多个图形项的容器。它还提供了几种标准形状，如矩形和椭圆，包括文本项。还有一点：`QGraphicsScene`使用OpenGL来渲染图形。OpenGL非常高效，可用于显示图像和执行多媒体处理任务。`QGraphicsScene`类提供了几种方法，可帮助添加或删除场景中的图形项。也就是说，您可以通过调用`addItem`函数向场景添加任何图形项。同样，要从图形场景中删除项目，可以调用`removeItem`函数。

# 实现动画

要在Python中应用动画，我们将使用`QPropertyAnimation`类。PyQt中的`QPropertyAnimation`类帮助创建和执行动画。`QPropertyAnimation`类通过操纵Qt属性（如小部件的几何形状、位置等）来实现动画。以下是`QPropertyAnimation`的一些方法：

+   `start()`: 该方法开始动画

+   `stop()`: 该方法结束动画

+   `setStartValue()`: 该方法用于指定动画的起始值

+   `setEndValue()`: 该方法用于指定动画的结束值

+   `setDuration()`: 该方法用于设置动画的持续时间（毫秒）

+   `setKeyValueAt()`: 该方法在给定值处创建关键帧

+   `setLoopCount()`: 该方法设置动画中所需的重复次数

# 显示2D图形图像

在本教程中，您将学习如何显示2D图形图像。我们假设您的计算机上有一个名为`scene.jpg`的图形图像，并将学习如何在表单上显示它。本教程的重点是了解如何使用Graphics View小部件来显示图像。

# 操作步骤...

显示图形的过程非常简单。您首先需要创建一个`QGraphicsScene`对象，该对象又利用`QGraphicsView`类来显示其内容。然后通过调用`QGraphicsScene`类的`addItem`方法向`QGraphicsScene`类添加图形项，包括图像。以下是在屏幕上显示2D图形图像的步骤：

1.  基于无按钮对话框模板创建一个新应用程序。

1.  将Graphics View小部件拖放到其中。

1.  将应用程序保存为`demoGraphicsView.ui`。表单将显示如下截图所示：

![](assets/bae96b96-83cd-4d38-9ed1-0185b3781174.png)

`pyuic5`命令实用程序将`.ui`（XML）文件转换为Python代码。生成的Python脚本`demoGraphicsView.py`可以在本书的源代码包中找到。

1.  创建一个名为`callGraphicsView.pyw`的Python脚本，导入代码`demoGraphicsView.py`，以调用用户界面设计，从磁盘加载图像，并通过Graphics View显示它。Python脚本文件`callGraphicsView.pyw`将包括以下代码：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from demoGraphicsView import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.scene = QGraphicsScene(self)
        pixmap= QtGui.QPixmap()
        pixmap.load("scene.jpg")
        item=QGraphicsPixmapItem(pixmap)
        self.scene.addItem(item)
        self.ui.graphicsView.setScene(self.scene)
if __name__=="__main__":
    app = QApplication(sys.argv)
    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_())
```

# 工作原理...

在此应用程序中，您正在使用Graphics View来显示图像。您向Graphics View小部件添加了一个图形场景，并添加了`QGraphicsPixmapItem`。如果要将图像添加到图形场景中，需要以`pixmap`项目的形式提供。首先，您需要将图像表示为`pixmap`，然后在将其添加到图形场景之前将其显示为`pixmap`项目。您需要创建`QPixmap`的实例，并通过其`load()`方法指定要通过其显示的图像。然后，通过将`pixmap`传递给`QGraphicsPixmapItem`的构造函数，将`pixmap`项目标记为`pixmapitem`。然后，通过`addItem`将`pixmapitem`添加到场景中。如果`pixmapitem`比`QGraphicsView`大，则会自动启用滚动。

在上面的代码中，我使用了文件名为`scene.jpg`的图像。请将文件名替换为您的磁盘上可用的图像文件名，否则屏幕上将不显示任何内容。

使用了以下方法：

+   `QGraphicsView.setScene`：此方法（self，`QGraphicsScene` scene）将提供的场景分配给`GraphicView`实例以进行显示。如果场景已经在视图中显示，则此函数不执行任何操作。设置场景时，将生成`QGraphicsScene.changed`信号，并调整视图的滚动条以适应场景的大小。

+   `addItem`：此方法将指定的项目添加到场景中。如果项目已经在不同的场景中，则首先将其从旧场景中移除，然后添加到当前场景中。运行应用程序时，将通过`GrahicsView`小部件显示`scene.jpg`图像，如下面的屏幕截图所示：

![](assets/70652385-2de8-405b-b118-4b9f403460bf.png)

# 点击按钮使球移动

在本教程中，您将了解如何在对象上应用基本动画。本教程将包括一个按钮和一个球，当按下按钮时，球将开始向地面动画。

# 操作步骤...

为了制作这个教程，我们将使用`QPropertyAnimation`类。`QPropertyAnimation`类的`setStartValue()`和`setEndValue()`方法将用于分别定义动画需要开始和结束的坐标。`setDuration()`方法将被调用以指定每次动画移动之间的延迟时间（以毫秒为单位）。以下是应用动画的逐步过程：

1.  基于无按钮对话框模板创建一个新应用程序。

1.  将一个Label小部件和一个Push Button小部件拖放到表单上。

1.  将Push Button小部件的文本属性设置为`Move Down`。我们假设您的计算机上有一个名为`coloredball.jpg`的球形图像。

1.  选择其pixmap属性以将球图像分配给Label小部件。

1.  在pixmap属性中，从两个选项中选择Resource和Choose File，选择Choose File选项，浏览您的磁盘，并选择`coloredball.jpg`文件。球的图像将出现在Label小部件的位置。

1.  将Push Button小部件的objectName属性设置为`pushButtonPushDown`，Label小部件的objectName属性设置为`labelPic`。

1.  使用名称`demoAnimation1.ui`保存应用程序。应用程序将显示如下屏幕截图所示：

![](assets/d58c8775-5b6f-4b2f-8c3a-cdeefbcef278.png)

使用Qt Designer创建的用户界面存储在`.ui`文件中，这是一个需要转换为Python代码的XML文件。在应用`pyuic5`命令实用程序时，`.ui`文件将被转换为Python脚本。生成的Python脚本`demoAnimation1.py`可以在本书的源代码包中看到。

1.  将`demoAnimation1.py`脚本视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callAnimation1.pyw`的Python文件，并将`demoAnimation1.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtCore import QRect, QPropertyAnimation
from demoAnimation1 import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonMoveDown.clicked.connect(self.
        startAnimation)
        self.show()
    def startAnimation(self):
        self.anim = QPropertyAnimation(self.ui.labelPic, 
        b"geometry")
        self.anim.setDuration(10000)
        self.anim.setStartValue(QRect(160, 70, 80, 80))
        self.anim.setEndValue(QRect(160, 70, 220, 220))
        self.anim.start()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

您可以看到，具有objectName属性`pushButtonMoveDown`的推送按钮小部件的click()事件连接到`startAnimation`方法；当点击推送按钮时，将调用`startAnimation`方法。在`startAnimation`方法中，创建一个`QPropertyAnimation`类的对象并命名为`anim`。在创建`QPropertyAnimation`实例时，传递两个参数；第一个是要应用动画的标签小部件，第二个是定义要将动画应用于对象属性的属性。因为您想要对球的几何图形应用动画，所以在定义`QPropertyAnimation`对象时，将`b"geometry"`作为第二个属性传递。之后，将动画的持续时间指定为`10000`毫秒，这意味着您希望每隔10,000毫秒更改对象的几何图形。通过`setStartValue`方法，指定要开始动画的矩形区域，并通过调用`setEndValue`方法，指定要停止动画的矩形区域。通过调用`start`方法，启动动画；因此，球从通过`setStartValue`方法指定的矩形区域向下移动，直到达到通过`setEndValue`方法指定的矩形区域。

运行应用程序时，您会在屏幕上找到一个推送按钮和一个代表球图像的标签小部件，如下截图所示（左）。点击Move Down推送按钮后，球开始向地面动画，并在通过`setEndValue`方法指定的区域停止动画，如下截图所示（右）：

![](assets/36217d0c-f023-4596-9f9b-433a7f9502ac.png)

# 制作一个弹跳的球

在这个示例中，您将制作一个弹跳的球；当点击按钮时，球向地面掉落，触及地面后，它会反弹到顶部。在这个示例中，您将了解如何在对象上应用基本动画。这个示例将包括一个推送按钮和一个球，当按下推送按钮时，球将开始向地面动画。

# 如何做...

要使球看起来像是在弹跳，我们需要首先使其向地面动画，然后从地面向天空动画。为此，我们将三次调用`QPropertyAnimation`类的`setKeyValueAt`方法。前两次调用`setKeyValueAt`方法将使球从顶部向底部动画。第三次调用`setKeyValueAt`方法将使球从底部向顶部动画。在三个`setKeyValueAt`方法中提供坐标，以使球以相反方向弹跳，而不是从哪里来的。以下是了解如何使球看起来像在弹跳的步骤：

1.  基于没有按钮的对话框模板创建一个新的应用程序。

1.  将一个标签小部件和一个推送按钮小部件拖放到表单上。

1.  将推送按钮小部件的文本属性设置为`Bounce`。我们假设您的计算机上有一个名为`coloredball.jpg`的球形图像。

1.  要将球形图像分配给标签小部件，请选择其pixmap属性。

1.  在pixmap属性中，从两个选项`Choose Resource`和`Choose File`中选择`Choose File`选项，浏览您的磁盘，并选择`coloredball.jpg`文件。球的图像将出现在标签小部件的位置。

1.  将推送按钮小部件的objectName属性设置为`pushButtonBounce`，标签小部件的objectName属性设置为`labelPic`。

1.  将应用程序保存为`demoAnimation3.ui`。

应用程序将显示如下截图所示：

![](assets/f6f06312-aec8-4d84-864c-1635ffe30319.png)

使用Qt Designer创建的用户界面存储在`.ui`文件中，这是一个XML文件，需要转换为Python代码。在应用`pyuic5`命令实用程序时，`.ui`文件将被转换为Python脚本。生成的Python脚本`demoAnimation3.py`可以在本书的源代码包中找到。

1.  将`demoAnimation3.py`脚本视为头文件，并将其导入到您将调用其用户界面设计的文件中。

1.  创建另一个名为`callAnimation3.pyw`的Python文件，并将`demoAnimation3.py`代码导入其中。

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtCore import QRect, QPropertyAnimation
from demoAnimation3 import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonBounce.clicked.connect(self.
        startAnimation)
        self.show()
    def startAnimation(self):
        self.anim = QPropertyAnimation(self.ui.labelPic, 
        b"geometry")
        self.anim.setDuration(10000)
        self.anim.setKeyValueAt(0, QRect(0, 0, 100, 80));
        self.anim.setKeyValueAt(0.5, QRect(160, 160, 200, 180));
        self.anim.setKeyValueAt(1, QRect(400, 0, 100, 80));
        self.anim.start()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

您可以看到，具有objectName属性`pushButtonMoveDown`的Push按钮小部件的click()事件与`startAnimation`方法连接在一起；当单击按钮时，将调用`startAnimation`方法。在`startAnimation`方法中，您创建一个`QPropertyAnimation`类的对象，并将其命名为`anim`。在创建`QPropertyAnimation`实例时，您传递两个参数：第一个是要应用动画的Label小部件，第二个是定义要将动画应用于对象属性的属性。因为您想要将动画应用于球的几何属性，所以在定义`QPropertyAnimation`对象时，将`b"geometry"`作为第二个属性传递。之后，您将动画的持续时间指定为`10000`毫秒，这意味着您希望每隔10,000毫秒更改对象的几何形状。通过`setKeyValue`方法，您指定要开始动画的区域，通过这种方法指定左上角区域，因为您希望球从左上角向地面掉落。通过对`setKeyValue`方法的第二次调用，您提供了球掉落到地面的区域。您还指定了掉落的角度。球将对角线向下掉落到地面。通过调用第三个`setValue`方法，您指定动画停止的结束值，在这种情况下是在右上角。通过对`setKeyValue`方法的这三次调用，您使球对角线向下掉落到地面，然后反弹回右上角。通过调用`start`方法，您启动动画。

运行应用程序时，您会发现Push按钮和Label小部件代表球图像显示在屏幕左上角，如下面的屏幕截图所示（左侧）。

单击Bounce按钮后，球开始沿对角线向下动画移动到地面，如中间屏幕截图所示，触地后，球反弹回屏幕的右上角，如右侧所示：

![](assets/b97a5f52-81a4-4e91-afde-843f7bfe0f4a.png)

# 根据指定的曲线使球动起来

创建一个具有所需形状和大小的曲线，并设置一个球在单击按钮时沿着曲线的形状移动。在这个示例中，您将了解如何实现引导动画。

# 如何做...

`QPropertyAnimation`类的`setKeyValueAt`方法确定动画的方向。对于引导动画，您在循环中调用`setKeyValueAt`方法。在循环中将曲线的坐标传递给`setKeyValueAt`方法，以使球沿着曲线动画。以下是使对象按预期动画的步骤：

1.  基于无按钮对话框模板创建一个新的应用程序。

1.  将一个Label小部件和一个Push按钮小部件拖放到表单上。

1.  将Push按钮小部件的文本属性设置为`Move With Curve`。

1.  假设您的计算机上有一个名为`coloredball.jpg`的球形图像，您可以使用其pixmap属性将此球形图像分配给Label小部件。

1.  在`pixmap`属性中，您会找到两个选项，选择资源和选择文件；选择选择文件选项，浏览您的磁盘，并选择`coloredball.jpg`文件。球的图像将出现在`Label`小部件的位置。

1.  将`Push Button`小部件的`objectName`属性设置为`pushButtonMoveCurve`，将`Label`小部件的`objectName`属性设置为`labelPic`。

1.  将应用程序保存为`demoAnimation4.ui`。应用程序将显示如下截图所示：

![](assets/0ab85891-3816-483b-9ede-704837a20332.png)

使用Qt Designer创建的用户界面存储在`.ui`文件中，是一个XML文件。通过应用`pyuic5`实用程序，将XML文件转换为Python代码。您可以在本书的源代码包中找到生成的Python代码`demoAnimation4.py`。

1.  将`demoAnimation4.py`脚本视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callAnimation4.pyw`的Python文件，并将`demoAnimation4.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtCore import QRect, QPointF, QPropertyAnimation, pyqtProperty
from PyQt5.QtGui import QPainter, QPainterPath
from demoAnimation4 import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonMoveCurve.clicked.connect(self.
        startAnimation)
        self.path = QPainterPath()
        self.path.moveTo(30, 30)
        self.path.cubicTo(30, 30, 80, 180, 180, 170)
        self.ui.labelPic.pos = QPointF(20, 20)
        self.show()
    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        qp.drawPath(self.path)
        qp.end()
    def startAnimation(self):
        self.anim = QPropertyAnimation(self.ui.labelPic, b'pos')
        self.anim.setDuration(4000)
        self.anim.setStartValue(QPointF(20, 20))
        positionValues = [n/80 for n in range(0, 50)]
        for i in positionValues:
            self.anim.setKeyValueAt(i,  
            self.path.pointAtPercent(i))
            self.anim.setEndValue(QPointF(160, 150))
            self.anim.start()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

首先，让曲线出现在屏幕上。这是将指导球动画的曲线；也就是说，它将作为动画的路径。您定义了`QPainterPath`类的实例并将其命名为`path`。您调用`QPainterPath`类的`moveTo`方法来指定路径或曲线的起始位置。调用`cubicTo`方法来指定球动画的曲线路径。

您会发现`Push Button`小部件的`objectName`属性为`pushButtonMoveCurve`的点击事件与`startAnimation`方法相连接；当单击`Push Button`小部件时，将调用`startAnimation()`方法。在`startAnimation`方法中，您创建了`QPropertyAnimation`类的对象并将其命名为`anim`。在创建`QPropertyAnimation`实例时，您传递了两个参数：第一个是要应用动画的`Label`小部件，第二个是定义要将动画应用于对象属性的属性。因为您想要将动画应用于球的位置，所以在定义`QPropertyAnimation`对象时，您将`b'pos'`作为第二个属性传递。之后，您将动画的持续时间指定为`4000`毫秒，这意味着您希望每`4000`毫秒更改球的位置。使用`QPropertyAnimation`类的`setStartValue()`方法，您指定了希望球进行动画的坐标。您设置了指定球需要沿着移动的值的`for`循环。您通过在`for`循环内调用`setKeyValue`方法来指定球的动画路径。因为球需要在路径中指定的每个点绘制，所以您通过调用`pointAtPercent()`方法并将其传递给`setKeyValueAt()`方法来设置球需要绘制的点。您还需要通过调用`setEndValue()`方法来设置动画需要停止的位置。

不久之后，您会指定动画的开始和结束位置，指定动画的路径，并调用`paintEvent()`方法来在路径的每一点重新绘制球。

运行应用程序后，您会在屏幕左上角（截图的左侧）找到`Push Button`小部件和代表球形图像的`Label`小部件，并在单击`Move With Curve`按钮后，球会沿着绘制的曲线开始动画，并在曲线结束的地方停止（截图的右侧）：

![](assets/4a89a80e-ad09-4699-bae1-03d9950497f4.png)
