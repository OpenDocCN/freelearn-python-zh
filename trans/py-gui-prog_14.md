# 事件处理-信号和插槽

在本章中，我们将学习以下主题：

+   使用信号/插槽编辑器

+   从一个*Line Edit*小部件复制并粘贴文本到另一个*Line Edit*小部件

+   转换数据类型并制作一个小型计算器

+   使用旋转框小部件

+   使用滚动条和滑块

+   使用列表小部件

+   从一个列表小部件中选择多个列表项，并在另一个列表中显示它们

+   将项目添加到列表小部件中

+   在列表小部件中执行操作

+   使用组合框小部件

+   使用字体组合框小部件

+   使用进度条小部件

# 介绍

事件处理是每个应用程序中的重要机制。应用程序不仅应该识别事件，还必须采取相应的行动来服务事件。在任何事件上采取的行动决定了应用程序的进程。每种编程语言都有不同的处理或监听事件的技术。让我们看看Python如何处理其事件。

# 使用信号/插槽编辑器

在PyQt中，事件处理机制也被称为**信号**和**插槽**。事件可以是在小部件上单击或双击的形式，或按下*Enter*键，或从单选按钮、复选框等中选择选项。每个小部件在应用事件时都会发出一个信号，该信号需要连接到一个方法，也称为插槽。插槽是指包含您希望在发生信号时执行的代码的方法。大多数小部件都有预定义的插槽；您不必编写代码来将预定义的信号连接到预定义的插槽。

您甚至可以通过导航到工具栏中的编辑|编辑信号/插槽工具来编辑信号/插槽。

# 如何做...

要编辑放置在表单上的不同小部件的信号和插槽，您需要执行以下步骤切换到信号和插槽编辑模式：

1.  您可以按*F4*键，导航到编辑|编辑信号/插槽选项，或从工具栏中选择编辑信号/插槽图标。该模式以箭头的形式显示所有信号和插槽连接，指示小部件与其相应插槽的连接。

您还可以在此模式下创建小部件之间的新信号和插槽连接，并删除现有信号。

1.  要在表单中的两个小部件之间建立信号和插槽连接，请通过在小部件上单击鼠标，将鼠标拖向要连接的另一个小部件，然后释放鼠标按钮来选择小部件。

1.  在拖动鼠标时取消连接，只需按下*Esc*键。

1.  在释放鼠标到达目标小部件时，将出现“连接对话框”，提示您从源小部件中选择信号和从目标小部件中选择插槽。

1.  选择相应的信号和插槽后，选择“确定”以建立信号和插槽连接。

以下屏幕截图显示了将*Push Button*拖动到*Line Edit*小部件上：

![](assets/44ba5e6d-dc82-49fc-a7b8-db7d21c9ff08.png)

1.  在*Line Edit*小部件上释放鼠标按钮后，您将获得预定义信号和插槽的列表，如下图所示：

![](assets/6236b767-2cf0-4da3-bc9b-f585ceb40395.png)您还可以在“配置连接”对话框中选择取消以取消信号和插槽连接。

1.  连接后，所选信号和插槽将显示为箭头中的标签，连接两个小部件。

1.  要修改信号和插槽连接，请双击连接路径或其标签之一，以显示“配置连接”对话框。

1.  从“配置连接”对话框中，您可以根据需要编辑信号或插槽。

1.  要删除信号和插槽连接，请在表单上选择其箭头，然后按*删除*键。

信号和插槽连接也可以在任何小部件和表单之间建立。为此，您可以执行以下步骤：

1.  选择小部件，拖动鼠标，并释放鼠标按钮到表单上。连接的终点会变成电气接地符号，表示已经在表单上建立了连接。

1.  要退出信号和插槽编辑模式，导航到Edit | Edit Widgets或按下*F3*键。

# 从一个Line Edit小部件复制文本并粘贴到另一个

这个教程将让您了解一个小部件上执行的事件如何调用相关小部件上的预定义动作。因为我们希望在点击推按钮时从一个Line Edit小部件复制内容，所以我们需要在推按钮的pressed()事件发生时调用`selectAll()`方法。此外，我们需要在推按钮的released()事件发生时调用`copy()`方法。要在点击另一个推按钮时将剪贴板中的内容粘贴到另一个Line Edit小部件中，我们需要在另一个推按钮的clicked()事件发生时调用`paste()`方法。

# 准备就绪

让我们创建一个包含两个Line Edit和两个Push Button小部件的应用程序。点击第一个推按钮时，第一个Line Edit小部件中的文本将被复制，点击第二个推按钮时，从第一个Line Edit小部件中复制的文本将被粘贴到第二个Line Edit小部件中。

让我们根据无按钮对话框模板创建一个新应用程序，执行以下步骤：

1.  通过从小部件框中将Line Edit和Push Button小部件拖放到表单上，开始添加`QLineEdit`和`QPushButton`。

在编辑时预览表单，选择Form、Preview，或使用*Ctrl* + *R*。

1.  要在用户在表单上选择推按钮时复制Line Edit小部件的文本，您需要将推按钮的信号连接到Line Edit的插槽。让我们学习如何做到这一点。

# 如何操作...

最初，表单处于小部件编辑模式，要应用信号和插槽连接，您需要首先切换到信号和插槽编辑模式：

1.  从工具栏中选择编辑信号/插槽图标，切换到信号和插槽编辑模式。

1.  在表单上，选择推按钮，拖动鼠标到Line Edit小部件上，然后释放鼠标按钮。配置连接对话框将弹出，允许您在Push Button和Line Edit小部件之间建立信号和插槽连接，如下截图所示：

![](assets/a35a1ce3-4efe-436f-9082-068d9263932a.png)

1.  从pushButton (QPushButton)选项卡中选择pressed()事件或信号，从lineEdit (QLineEdit)选项卡中选择selectAll()插槽。

Push Button小部件与Line Edit的连接信号将以箭头的形式显示，表示两个小部件之间的信号和插槽连接，如下截图所示：

![](assets/76b97b24-9475-4591-a360-15a724c1cc19.png)

1.  将Push Button小部件的文本属性设置为`Copy`，表示它将复制Line Edit小部件中输入的文本。

1.  接下来，我们将重复点击推按钮并将其拖动到Line Edit小部件上，以连接push按钮的released()信号与Line Edit小部件的copy()插槽。在表单上，您将看到另一个箭头，表示两个小部件之间建立的第二个信号和插槽连接，如下截图所示：

![](assets/3a3fca9d-bd78-4f3a-9756-17182366be2c.png)

1.  为了粘贴复制的内容，将一个推按钮和一个Line Edit小部件拖放到表单上。

1.  将Push Button小部件的文本属性设置为`Paste`。

1.  点击推按钮，按住鼠标按钮拖动，然后释放到Line Edit小部件上。

1.  从配置连接对话框中，选择pushButton (QPushButton)列中的clicked()事件和lineEdit (QLineEdit)列中的paste()插槽。

1.  将表单保存为`demoSignal1.ui`。表单现在将显示如下截图所示：

![](assets/4d966c42-f1e8-4711-8c50-3ac51135e31c.png)

表单将保存在扩展名为`.ui`的文件中。`demoSignal1.ui`文件将包含表单的所有信息，包括其小部件、布局等。`.ui`文件是一个XML文件，需要使用`pyuic5`实用程序将其转换为Python代码。生成的Python代码文件`demoSignal1.py`可以在本书的源代码包中找到。在`demoSignal1.py`文件中，您会发现它从`QtCore`和`QtGui`两个模块中导入了所有内容，因为您将需要它们来开发GUI应用程序：

+   `QtCore`：`QtCore`模块构成了所有基于Qt的应用程序的基础。它包含了最基本的类，如`QCoreApplication`、`QObject`等。这些类执行重要的任务，如事件处理、实现信号和槽机制、I/O操作、处理字符串等。该模块包括多个类，包括`QFile`、`QDir`、`QIODevice`、`QTimer`、`QString`、`QDate`和`QTime`。

+   `QtGui`：顾名思义，`QtGUI`模块包含了开发跨平台GUI应用程序所需的类。该模块包含了GUI类，如`QCheckBox`、`QComboBox`、`QDateTimeEdit`、`QLineEdit`、`QPushButton`、`QPainter`、`QPaintDevice`、`QApplication`、`QTextEdit`和`QTextDocument`。

1.  将`demoSignalSlot1.py`文件视为头文件，并将其导入到您将调用其用户界面设计的文件中。

1.  创建另一个名为`calldemoSignal1.pyw`的Python文件，并将`demoSignal1.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoSignalSlot1 import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.show()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

`sys`模块被导入，因为它提供了对存储在`sys.argv`列表中的命令行参数的访问。这是因为每个PyQt GUI应用程序必须有一个`QApplication`对象，以提供对应用程序目录、屏幕大小等信息的访问，因此您创建了一个`QApplication`对象。为了使PyQt能够使用和应用命令行参数（如果有的话），您在创建`QApplication`对象时传递命令行参数。您创建了`MyForm`的一个实例，并调用其`show()`方法，该方法向`QApplication`对象的事件队列中添加了一个新事件。这个新事件用于显示`MyForm`类中指定的所有小部件。调用`app.exec_`方法来启动`QApplication`对象的事件循环。一旦事件循环开始，`MyForm`类中使用的顶级小部件以及其子小部件将被显示。所有系统生成的事件以及用户交互事件都将被添加到事件队列中。应用程序的事件循环不断检查是否发生了事件。发生事件时，事件循环会处理它并调用相关的槽或方法。在关闭应用程序的顶级小部件时，PyQt会删除该小部件，并对应用程序进行清理终止。

在PyQt中，任何小部件都可以用作顶级窗口。`super().__init__()`方法从`MyForm`类中调用基类构造函数，即从`MyForm`类中调用`QDialog`类的构造函数，以指示通过该类显示`QDialog`是一个顶级窗口。

通过调用Python代码中创建的类的`setupUI()`方法来实例化用户界面设计（`Ui_Dialog`）。我们创建了`Ui_Dialog`类的一个实例，该类是在Python代码中创建的，并调用了它的`setupUi()`方法。对话框小部件将被创建为所有用户界面小部件的父级，并显示在屏幕上。请记住，`QDialog`、`QMainWindow`以及PyQt的所有小部件都是从`QWidget`派生的。

运行应用程序时，您将获得两对行编辑和按钮小部件。在一个行编辑小部件中输入文本，当您单击复制按钮时，文本将被复制。

现在，单击粘贴按钮后，复制的文本将粘贴在第二个行编辑小部件中，如下截图所示：

![](assets/4555d238-8d6a-4315-bcb7-6f4b37d2e1b4.png)

# 转换数据类型并创建一个小型计算器

接受单行数据最常用的小部件是行编辑小部件，行编辑小部件的默认数据类型是字符串。为了对两个整数值进行任何计算，需要将行编辑小部件中输入的字符串数据转换为整数数据类型，然后将计算结果（将是数值数据类型）转换回字符串类型，然后通过标签小部件显示。这个示例正是这样做的。

# 如何做...

为了了解用户如何接受数据以及如何进行类型转换，让我们创建一个基于对话框无按钮模板的应用程序，执行以下步骤：

1.  通过拖放三个标签、两个行编辑和四个按钮小部件到表单上，向表单添加三个`QLabel`、两个`QLineEdit`和一个`QPushButton`小部件。

1.  将两个标签小部件的文本属性设置为`输入第一个数字`和`输入第二个数字`。

1.  将三个标签的objectName属性设置为`labelFirstNumber`，`labelSecondNumber`和`labelResult`。

1.  将两个行编辑小部件的objectName属性设置为`lineEditFirstNumber`和`lineEditSecondNumber`。

1.  将四个按钮小部件的objectName属性分别设置为`pushButtonPlus`，`pushButtonSubtract`，`pushButtonMultiply`和`pushButtonDivide`。

1.  将按钮的文本属性分别设置为`+`，`-`，`x`和`/`。

1.  删除第三个标签的默认文本属性，因为Python脚本将设置该值，并在添加两个数字值时显示它。 

1.  不要忘记在设计师中拖动标签小部件，以确保它足够长，可以显示通过Python脚本分配给它的文本。

1.  将UI文件保存为`demoCalculator.ui`。

1.  您还可以通过在属性编辑器窗口中的geometry下设置宽度属性来增加标签小部件的宽度：

![](assets/78cf810c-5dc9-488b-8c9b-99978a6d18d5.png)

`.ui`文件以XML格式，需要转换为Python代码。生成的Python代码`demoCalculator.py`可以在本书的源代码包中看到。

1.  创建一个名为`callCalculator.pyw`的Python脚本，导入Python代码`demoCalculator.py`来调用用户界面设计，并获取输入的行编辑小部件中的值，并显示它们的加法。Python脚本`callCalculator.pyw`中的代码如下所示：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoCalculator import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonPlus.clicked.connect(self.addtwonum)
        self.ui.pushButtonSubtract.clicked.connect
        (self.subtracttwonum)
        self.ui.pushButtonMultiply.clicked.connect
        (self.multiplytwonum)
        self.ui.pushButtonDivide.clicked.connect(self.dividetwonum)
        self.show()
    def addtwonum(self):
        if len(self.ui.lineEditFirstNumber.text())!=0:
                a=int(self.ui.lineEditFirstNumber.text())
        else:
                a=0
        if len(self.ui.lineEditSecondNumber.text())!=0:
                b=int(self.ui.lineEditSecondNumber.text())
        else:
                b=0
                sum=a+b
        self.ui.labelResult.setText("Addition: " +str(sum))
    def subtracttwonum(self):
        if len(self.ui.lineEditFirstNumber.text())!=0:
                a=int(self.ui.lineEditFirstNumber.text())
        else:
                a=0
        if len(self.ui.lineEditSecondNumber.text())!=0:
                b=int(self.ui.lineEditSecondNumber.text())
        else:
                b=0
                diff=a-b
        self.ui.labelResult.setText("Substraction: " +str(diff))
    def multiplytwonum(self):
        if len(self.ui.lineEditFirstNumber.text())!=0:
                a=int(self.ui.lineEditFirstNumber.text())
        else:
                a=0
        if len(self.ui.lineEditSecondNumber.text())!=0:
                b=int(self.ui.lineEditSecondNumber.text())
        else:
                b=0
                mult=a*b
        self.ui.labelResult.setText("Multiplication: " +str(mult))
    def dividetwonum(self):
        if len(self.ui.lineEditFirstNumber.text())!=0:
                a=int(self.ui.lineEditFirstNumber.text())
        else:
                a=0
        if len(self.ui.lineEditSecondNumber.text())!=0:
                b=int(self.ui.lineEditSecondNumber.text())
        else:
                b=0
                division=a/b
        self.ui.labelResult.setText("Division: "+str(round
        (division,2)))
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

此代码中使用了以下四个函数：

+   `len()`: 这个函数返回字符串中的字符数

+   `str()`: 这个函数将传递的参数转换为字符串数据类型

+   `int()`: 这个函数将传递的参数转换为整数数据类型

+   `round()`: 这个函数将传递的数字四舍五入到指定的小数位

`pushButtonPlus`的`clicked()`事件连接到`addtwonum()`方法，以显示在两个行编辑小部件中输入的数字的总和。在`addtwonum()`方法中，首先验证`lineEditFirstNumber`和`lineEditSecondNumber`，以确保用户是否将任一行编辑留空，如果是，则该行编辑的值为零。

检索两个行编辑小部件中输入的值，通过`int()`转换为整数，并赋值给两个变量`a`和`b`。计算`a`和`b`变量中的值的总和，并存储在`sum`变量中。通过`str`方法将变量`sum`中的结果转换为字符串格式，并通过`labelResult`显示，如下截图所示：

![](assets/c49fe2fa-a965-4712-9a07-2b38c64867e5.png)

类似地，`pushButtonSubtract`的`clicked()`事件连接到`subtracttwonum()`方法，以显示两个行编辑小部件中输入的数字的减法。再次，在验证两个行编辑小部件之后，检索并将其输入的值转换为整数。对这两个数字进行减法运算，并将结果分配给`diff`变量。

最后，通过`str()`方法将`diff`变量中的结果转换为字符串格式，并通过`labelResult`显示，如下面的屏幕截图所示：

![](assets/45370293-94db-4db0-975b-215463fc4a0b.png)

类似地，`pushButtonMultiply`和`pushButtonDivide`的`clicked()`事件分别连接到`multiplytwonum()`和`dividetwonum()`方法。这些方法将两个行编辑小部件中输入的值相乘和相除，并通过`labelResult`小部件显示它们。

乘法的结果如下所示：

![](assets/ed7ceb86-a691-4815-b23a-b6fa3495afbf.png)

除法的结果如下所示：

![](assets/984be4f4-3428-475e-9a52-702b4dfd181d.png)

# 使用旋转框小部件

旋转框小部件用于显示整数值、浮点值和文本。它对用户施加了约束：用户不能输入任意数据，但只能从旋转框显示的可用选项中进行选择。旋转框小部件默认显示初始值，可以通过选择上/下按钮或在键盘上按上/下箭头键来增加或减少该值。您可以通过单击或手动输入来选择要显示的值。

# 准备就绪

旋转框小部件可以使用两个类`QSpinBox`和`QDoubleSpinBox`创建，其中`QSpinBox`仅显示整数值，而`QDoubleSpinBox`类显示浮点值。`QSpinBox`提供的方法如下所示：

+   `value()`: 此方法返回从旋转框中选择的当前整数值。

+   `text()`: 此方法返回旋转框显示的文本。

+   `setPrefix()`: 此方法分配要添加到旋转框返回值之前的前缀文本。

+   `setSuffix()`: 此方法分配要附加到旋转框返回值的后缀文本。

+   `cleanText()`: 此方法返回旋转框的值，不带后缀、前缀或前导或尾随空格。

+   `setValue()`: 此方法分配值给旋转框。

+   `setSingleStep()`: 此方法设置旋转框的步长。步长是旋转框的增量/减量值，即旋转框的值将通过选择上/下按钮或使用`setValue()`方法增加或减少的值。

+   `setMinimum()`: 此方法设置旋转框的最小值。

+   `setMaximum()`: 此方法设置旋转框的最大值。

+   `setWrapping()`: 此方法将布尔值true传递给此方法，以启用旋转框中的包装。包装意味着当按下上按钮显示最大值时，旋转框返回到第一个值（最小值）。

`QSpinBox`类发出的信号如下：

+   valueChanged(): 当通过选择上/下按钮或使用`setValue()`方法更改旋转框的值时，将发出此信号。

+   `editingFinished()`: 当焦点离开旋转框时发出此信号

用于处理旋转框中浮点值的类是`QDoubleSpinBox`。所有前述方法也受`QDoubleSpinBox`类的支持。它默认显示值，保留两位小数。要更改精度，请使用`round()`，它会显示值，保留指定数量的小数位；该值将四舍五入到指定数量的小数位。

旋转框的默认最小值、最大值、单步值和值属性分别为0、99、1和0；双精度旋转框的默认值为0.000000、99.990000、1.000000和0.000000。

让我们创建一个应用程序，该应用程序将要求用户输入书的价格，然后输入客户购买的书的数量，并显示书的总金额。此外，该应用程序将提示您输入1公斤糖的价格，然后输入用户购买的糖的数量。在输入糖的数量时，应用程序将显示糖的总量。书籍和糖的数量将分别通过微调框和双精度微调框输入。

# 如何做...

要了解如何通过微调框接受整数和浮点值并在进一步计算中使用，让我们基于无按钮模板创建一个新的应用程序，并按照以下步骤操作：

1.  让我们开始拖放三个标签，一个微调框，一个双精度微调框和四个行编辑小部件。

1.  两个标签小部件的文本属性设置为`Book Price value`和`Sugar Price`，第三个标签小部件的objectName属性设置为`labelTotalAmount`。

1.  将四个行编辑小部件的objectName属性设置为`lineEditBookPrice`，`lineEditBookAmount`，`lineEditSugarPrice`和`lineEditSugarAmount`。

1.  将Spin Box小部件的objectName属性设置为`spinBoxBookQty`，将Double Spin Box小部件的objectName属性设置为`doubleSpinBoxSugarWeight`。

1.  删除第三个标签小部件TextLabe的默认文本属性，因为您将在程序中设置其文本以显示总金额。

1.  删除第三个标签小部件的文本属性后，它将变得不可见。

1.  禁用两个行编辑小部件`lineEditBookAmount`和`lineEditSugarAmount`，通过取消选中它们的属性编辑器窗口中的启用属性，因为您希望它们显示不可编辑的值。

1.  使用名称`demoSpinner.ui`保存应用程序：

![](assets/4af31882-d0d5-42af-8146-943ccd17fa15.png)

1.  使用`pyuic5`命令实用程序，`.ui`（XML）文件将转换为Python代码。生成的Python代码文件`demoSpinner.py`可以在本书的源代码中看到。

1.  创建一个名为`calldemoSpinner.pyw`的Python脚本文件，导入代码`demoSpinner.py`，使您能够调用显示通过微调框选择的数字并计算总书籍金额和总糖量的用户界面设计。`calldemoSpinner.pyw`文件将显示如下：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoSpinBox import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.spinBoxBookQty.editingFinished.connect(self.
        result1)
        self.ui.doubleSpinBoxSugarWeight.editingFinished.connect
        (self.result2)
        self.show()
    def result1(self):
        if len(self.ui.lineEditBookPrice.text())!=0:
                bookPrice=int(self.ui.lineEditBookPrice.text())
        else:
                bookPrice=0
                totalBookAmount=self.ui.spinBoxBookQty.value() * 
                bookPrice
                self.ui.lineEditBookAmount.setText(str
                (totalBookAmount))
    def result2(self):
        if len(self.ui.lineEditSugarPrice.text())!=0:
                sugarPrice=float(self.ui.lineEditSugarPrice.
                text())
        else:
                sugarPrice=0
                totalSugarAmount=self.ui.
                doubleSpinBoxSugarWeight.value() * sugarPrice
                self.ui.lineEditSugarAmount.setText(str(round
                (totalSugarAmount,2)))
                totalBookAmount=int(self.ui.lineEditBookAmount.
                text())
                totalAmount=totalBookAmount+totalSugarAmount
                self.ui.labelTotalAmount.setText(str(round
                (totalAmount,2)))
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

在此代码中，您可以看到两个微调框的`editingFinished`信号附加到`result1`和`result2`函数。这意味着当焦点离开任何微调框时，将调用相应的方法。当用户使用鼠标移动到其他微调框或按Tab键时，焦点将离开小部件：

+   在`result1`方法中，您从Spin Box小部件中检索购买的书的数量的整数值，并将其乘以在`lineEditBookPrice`小部件中输入的书的价格，以计算总书费。然后通过`lineEditBookAmount`小部件显示总书费。

+   类似地，在`result2`方法中，您从双精度微调框中检索购买的糖的重量的浮点值，并将其乘以在`lineEditSugarPrice`小部件中输入的每公斤糖的价格，以计算总糖成本，然后通过`lineEditSugarAmount`小部件显示。书的成本和糖的成本的总和最终通过`labelTotalAmount`小部件显示，如下面的屏幕截图所示：

![](assets/ef46d187-e1cd-41f5-bcdc-05f1af89982d.png)

# 使用滚动条和滑块

滚动条在查看无法出现在有限可见区域的大型文档或图像时非常有用。滚动条水平或垂直出现，指示您在文档或图像中的当前位置以及不可见区域的大小。使用这些滚动条提供的滑块手柄，您可以访问文档或图像的隐藏部分。

滑块是选择两个值之间的整数值的一种方式。也就是说，滑块可以表示一系列最小和最大值，并且用户可以通过将滑块手柄移动到滑块中所需位置来选择此范围内的值。

# 准备就绪

滚动条用于查看大于视图区域的文档或图像。要显示水平或垂直滚动条，您可以使用`HorizontalScrollBar`和`VerticalScrollBar`小部件，它们是`QScrollBar`类的实例。这些滚动条有一个滑块手柄，可以移动以查看不可见的区域。滑块手柄的位置指示文档或图像内的位置。滚动条具有以下控件：

+   **滑块手柄**: 此控件用于快速移动到文档或图像的任何部分。

+   **滚动箭头**: 这些是滚动条两侧的箭头，用于查看当前不可见的文档或图像的所需区域。使用这些滚动箭头时，滑块手柄的位置移动以显示文档或图像内的当前位置。

+   **页面控制**: 页面控制是滑块手柄拖动的滚动条的背景。单击背景时，滑块手柄向单击位置移动一个页面。滑块手柄移动的量可以通过pageStep属性指定。页面步进是用户按下*Page Up*和*Page Down*键时滑块移动的量。您可以使用`setPageStep()`方法设置pageStep属性的量。

用于设置和检索滚动条的值的特定方法是`value()`方法，这里进行了描述。

`value()`方法获取滑块手柄的值，即其距离滚动条起始位置的距离值。当滑块手柄在垂直滚动条的顶部边缘或水平滚动条的左边缘时，您会得到滚动条的最小值；当滑块手柄在垂直滚动条的底部边缘或水平滚动条的右边缘时，您会得到滚动条的最大值。您也可以通过键盘将滑块手柄移动到其最小和最大值，分别按下*Home*和*End*键。让我们来看看以下方法：

+   `setValue()`: 此方法将值分配给滚动条，并根据分配的值设置滑块手柄在滚动条中的位置

+   `minimum()`: 此方法返回滚动条的最小值

+   `maximum()`: 此方法返回滚动条的最大值

+   `setMinimum()`: 此方法将最小值分配给滚动条

+   `setMaximum()`: 此方法将最大值分配给滚动条

+   `setSingleStep()`: 此方法设置单步值

+   `setPageStep()`: 此方法设置页面步进值

`QScrollBar`仅提供整数值。

通过`QScrollBar`类发出的信号如下所示：

+   valueChanged(): 当滚动条的值发生变化时发出此信号，即当其滑块手柄移动时

+   sliderPressed(): 当用户开始拖动滑块手柄时发出此信号

+   sliderMoved(): 当用户拖动滑块手柄时发出此信号

+   sliderReleased(): 当用户释放滑块手柄时发出此信号

+   actionTriggered(): 当用户交互改变滚动条时发出此信号

滑块通常用于表示某个整数值。与滚动条不同，滚动条大多用于显示大型文档或图像，滑块是交互式的，是输入或表示整数值的更简单的方式。也就是说，通过移动和定位其手柄沿水平或垂直槽，可以使水平或垂直滑块表示某个整数值。为了显示水平和垂直滑块，使用了`HorizontalSlider`和`VerticalSlider`小部件，它们是`QSlider`类的实例。与我们在滚动条中看到的方法类似，滑块在移动滑块手柄时也会生成信号，例如`valueChanged()`，`sliderPressed()`，`sliderMoved()`，`sliderReleased()`等等。

滚动条和滑块中的滑块手柄表示在最小和最大范围内的值。要更改默认的最小和最大值，可以通过为minimum、maximum、singleStep和pageStep属性分配值来更改它们的值。

滑块的最小值、最大值、singleStep、pageStep和value属性的默认值分别为0、99、1、10和0。

让我们创建一个应用程序，其中包括水平和垂直滚动条，以及水平和垂直滑块。水平滚动条和滑块将分别表示血糖水平和血压。也就是说，移动水平滚动条时，患者的血糖水平将通过行编辑小部件显示。同样，移动水平滑块时，将表示血压，并通过行编辑小部件显示。

垂直滚动条和滑块将分别表示心率和胆固醇水平。移动垂直滚动条时，心率将通过行编辑小部件显示，移动垂直滑块时，胆固醇水平将通过行编辑小部件显示。

# 操作步骤...

为了理解水平和垂直滚动条的工作原理，以及水平和垂直滑块的工作原理，了解滚动条和滑块在值更改时如何生成信号，以及如何将相应的槽或方法与它们关联，执行以下步骤：

1.  让我们创建一个新的对话框应用程序，没有按钮模板，并将水平和垂直滚动条和滑块拖放到表单上。

1.  将四个标签小部件和一个行编辑小部件放置到显示滚动条和滑块手柄值的位置。

1.  将四个标签小部件的text属性分别设置为`血糖水平`，`血压`，`脉搏率`和`胆固醇`。

1.  将水平滚动条的objectName属性设置为`horizontalScrollBarSugarLevel`，垂直滚动条的objectName属性设置为`verticalScrollBarPulseRate`，水平滑块的objectName属性设置为`horizontalSliderBloodPressure`，垂直滑块的objectName属性设置为`verticalSliderCholestrolLevel`。

1.  将行编辑小部件的objectName属性设置为`lineEditResult`。

1.  将应用程序保存为名称为`demoSliders.ui`的文件。表单将显示如下截图所示：

![](assets/e9694361-d6d4-4f0d-a343-05e70ae1ea91.png)

`pyuic5`命令实用程序将把`.ui`（XML）文件转换为Python代码。生成的Python文件`demoScrollBar.py`可以在本书的源代码包中找到。

1.  创建一个名为`callScrollBar.pyw`的Python脚本文件，导入代码`demoScrollBar.py`，以调用用户界面设计并同步滚动条和滑块手柄的移动。该脚本还将通过标签小部件显示滚动条和滑块手柄的值。Python脚本`callScrollBar.pyw`将显示如下：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoScrollBar import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.horizontalScrollBarSugarLevel.valueChanged.connect
        (self.scrollhorizontal)
        self.ui.verticalScrollBarPulseRate.valueChanged.connect
        (self.scrollvertical)
        self.ui.horizontalSliderBloodPressure.valueChanged.connect
        (self.sliderhorizontal)
        self.ui.verticalSliderCholestrolLevel.valueChanged.connect
        (self.slidervertical)
        self.show()
    def scrollhorizontal(self,value):
        self.ui.lineEditResult.setText("Sugar Level : "+str(value))
    def scrollvertical(self, value):
        self.ui.lineEditResult.setText("Pulse Rate : "+str(value))
    def sliderhorizontal(self, value):
        self.ui.lineEditResult.setText("Blood Pressure :  
        "+str(value))
    def slidervertical(self, value):
        self.ui.lineEditResult.setText("Cholestrol Level : 
        "+str(value))
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

在此代码中，您正在将每个窗口部件的`valueChanged()`信号与相应的函数连接起来，以便如果窗口部件的滚动条或滑块移动，将调用相应的函数来执行所需的任务。例如，当水平滚动条的滑块移动时，将调用`scrollhorizontal`函数。`scrollhorizontal`函数通过Label窗口部件显示滚动条表示的值，即血糖水平。

同样，当垂直滚动条或滑块的滑块移动时，将调用`scrollvertical`函数，并且垂直滚动条的滑块的值，即心率，将通过Label窗口部件显示，如下面的屏幕截图所示：

![](assets/267f54e8-3456-4f03-aca5-19072ac1f550.png)

同样，当水平和垂直滑块移动时，血压和胆固醇水平会相应地显示，如下面的屏幕截图所示：

![](assets/69d859b6-7ea2-4da1-8bc1-b9458ee448fc.png)

# 使用List窗口部件

要以更简单和可扩展的格式显示多个值，可以使用List窗口部件，它是`QListWidget`类的实例。List窗口部件显示多个项目，不仅可以查看，还可以编辑和删除。您可以逐个添加或删除列表项目，也可以使用其内部模型集合地设置列表项目。

# 准备工作

列表中的项目是`QListWidgetItem`类的实例。`QListWidget`提供的方法如下所示：

+   `insertItem()`: 此方法将提供的文本插入到List窗口部件的指定位置。

+   `insertItems()`: 此方法从提供的列表中的指定位置开始插入多个项目。

+   `count()`: 此方法返回列表中项目数量的计数。

+   `takeItem()`: 此方法从列表窗口中指定的行中移除并返回项目。

+   `currentItem()`: 此方法返回列表中的当前项目。

+   `setCurrentItem()`: 此方法用指定的项目替换列表中的当前项目。

+   `addItem()`: 此方法将具有指定文本的项目附加到List窗口部件的末尾。

+   `addItems()`: 此方法将提供的列表中的项目附加到List窗口部件的末尾。

+   `clear()`: 此方法从List窗口部件中移除所有项目。

+   `currentRow()`: 此方法返回当前选定列表项的行号。如果未选择列表项，则返回值为`-1`。

+   `setCurrentRow()`: 此方法选择List窗口部件中的指定行。

+   `item()`: 此方法返回指定行处的列表项。

`QListWidget`类发出的信号如下所示：

+   currentRowChanged(): 当当前列表项的行更改时发出此信号

+   currentTextChanged(): 当当前列表项中的文本更改时发出此信号

+   currentItemChanged(): 当当前列表项的焦点更改时发出此信号

# 如何做...

因此，让我们创建一个应用程序，通过List窗口部件显示特定的诊断测试，并且当用户从List窗口部件中选择任何测试时，所选测试将通过Label窗口部件显示。以下是创建应用程序的逐步过程：

1.  创建一个没有按钮模板的对话框的新应用程序，并将两个Label窗口部件和一个List窗口部件拖放到表单上。

1.  将第一个Label窗口部件的文本属性设置为“选择诊断测试”。

1.  将List窗口部件的objectName属性设置为`listWidgetDiagnosis`。

1.  将Label窗口部件的objectName属性设置为`labelTest`。

1.  删除`labelTest`窗口部件的默认文本属性，因为我们将通过代码通过此窗口部件显示所选的诊断测试。

1.  要通过List窗口部件显示诊断测试，请右键单击它，并从打开的上下文菜单中选择“编辑项目”选项。

1.  逐个添加诊断测试，然后在输入每个测试后单击底部的+按钮，如下截图所示：

![](assets/f88c1e1d-2934-4bd7-a0ba-37acd3757fca.png)

1.  使用名称`demoListWidget1.ui`保存应用程序。表单将显示如下截图所示：

![](assets/46334250-f059-4b61-a0d8-413c80e98db3.png)

`pyuic5`命令实用程序将把`.ui`（XML）文件转换为Python代码。生成的Python代码`demoListWidget1.py`可以在本书的源代码包中看到。

1.  创建一个名为`callListWidget1.pyw`的Python脚本文件，导入代码`demoListWidget1.py`，以调用用户界面设计和从列表窗口中显示所选的诊断测试的代码。Python脚本`callListWidget1.pyw`中的代码如下所示：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoListWidget1 import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.listWidgetDiagnosis.itemClicked.connect(self.
        dispSelectedTest)
        self.show()
    def dispSelectedTest(self):
        self.ui.labelTest.setText("You have selected 
        "+self.ui.listWidgetDiagnosis.currentItem().text())
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

您可以看到列表窗口的`itemClicked`事件连接到`dispSelectedTest()`方法。也就是说，单击列表窗口中的任何列表项时，将调用`dispSelectedTest()`方法，该方法使用列表窗口的`currentItem`方法通过名为`labelTest`的标签显示列表窗口的所选项目。

运行应用程序时，您将看到列表窗口显示一些诊断测试；从列表窗口中选择一个测试，该测试将通过Label窗口显示，如下截图所示：

![](assets/989c3543-3303-45f5-ae47-1f09e11d3090.png)

# 从一个列表窗口中选择多个列表项，并在另一个列表窗口中显示它们

在前面的应用程序中，您只从列表窗口中选择了单个诊断测试。如果我想要从列表窗口中进行多重选择怎么办？在进行多重选择的情况下，您需要另一个列表窗口来存储所选的诊断测试，而不是使用行编辑窗口。

# 如何做...

让我们创建一个应用程序，通过列表窗口显示特定的诊断测试，当用户从列表窗口中选择任何测试时，所选测试将显示在另一个列表窗口中：

1.  因此，创建一个没有按钮模板的对话框的新应用程序，并将两个Label窗口小部件和两个列表窗口拖放到表单上。

1.  将第一个Label窗口小部件的文本属性设置为`诊断测试`，另一个设置为`已选择的测试为`。

1.  将第一个列表窗口的objectName属性设置为`listWidgetDiagnosis`，第二个列表窗口的设置为`listWidgetSelectedTests`。

1.  要通过列表窗口显示诊断测试，请右键单击它，从打开的上下文菜单中选择“编辑项目”选项。

1.  逐个添加诊断测试，然后在输入每个测试后单击底部的+按钮。

1.  要从列表窗口启用多重选择，请选择`listWidgetDiagnosis`窗口小部件，并从属性编辑器窗口中将selectionMode属性从`SingleSelection`更改为`MultiSelection`。

1.  使用名称`demoListWidget2.ui`保存应用程序。表单将显示如下截图所示：

![](assets/62a52de1-daaf-4e26-a240-17cd71be81d7.png)

通过使用`pyuic5`实用程序，XML文件`demoListWidget2.ui`将被转换为Python代码，即`demoListWidget2.py`文件。可以在本书的源代码包中看到从`demoListWidget2.py`文件生成的Python代码。

1.  创建一个名为`callListWidget2.pyw`的Python脚本文件，导入代码`demoListWidget2.py`，以调用用户界面设计和显示从列表窗口中选择的多个诊断测试的代码。Python脚本`callListWidget2.pyw`将显示如下：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoListWidget2 import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.listWidgetDiagnosis.itemSelectionChanged.connect
        (self.dispSelectedTest)
        self.show()
    def dispSelectedTest(self):
        self.ui.listWidgetSelectedTests.clear()
        items = self.ui.listWidgetDiagnosis.selectedItems()
        for i in list(items):
            self.ui.listWidgetSelectedTests.addItem(i.text())
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

您可以看到，第一个列表小部件的`itemSelectionChanged`事件连接到`dispSelectedTest()`方法。也就是说，在从第一个列表小部件中选择或取消选择任何列表项目时，将调用`dispSelectedTest()`方法。`dispSelectedTest()`方法调用列表小部件上的`selectedItems()`方法以获取所有选定项目的列表。然后，使用`for`循环，通过在第二个列表小部件上调用`addItem()`方法，将所有选定的项目添加到第二个列表小部件中。

运行应用程序时，您将看到列表小部件显示一些诊断测试；从第一个列表小部件中选择任意数量的测试，所有选定的测试将通过第二个列表小部件项目显示，如下截图所示：

![](assets/e8efd76e-aedb-4ecc-ac45-fd7f615126a9.png)

# 向列表小部件添加项目

虽然您可以通过属性编辑器手动向列表小部件添加项目，但有时需要通过代码动态向列表小部件添加项目。让我们创建一个应用程序，解释向列表小部件添加项目的过程。

在此应用程序中，您将使用标签、行编辑、按钮和列表小部件。列表小部件项目最初将为空，并要求用户将所需的食物项目输入到行编辑中，并选择“添加到列表”按钮。然后将输入的食物项目添加到列表小部件项目中。所有后续的食物项目将添加到上一个条目下方。

# 如何做...

执行以下步骤以了解如何向列表小部件项目添加项目：

1.  我们将从基于无按钮对话框模板创建一个新应用程序开始，并将标签、行编辑、按钮和列表小部件拖放到表单中。

1.  将标签和按钮小部件的文本属性分别设置为“您最喜欢的食物项目”和“添加到列表”。

1.  将行编辑小部件的objectName属性设置为`lineEditFoodItem`，按钮的objectName设置为`pushButtonAdd`，列表小部件的objectName设置为`listWidgetSelectedItems`。

1.  将应用程序保存为`demoListWidget3.ui`。表单将显示如下截图所示：

![](assets/f7f06759-0127-48e4-93fd-83b746974b69.png)

在执行`pyuic5`实用程序时，XML文件`demoListWidget3.ui`将被转换为Python代码`demoListWidget3.py`。生成的Python文件`demoListWidget3.py`的代码可以在本书的源代码包中找到。

1.  创建一个名为`callListWidget3.pyw`的Python脚本文件，导入Python代码`demoListWidget3.py`以调用用户界面设计，并将用户在行编辑中输入的食物项目添加到列表小部件中。`callListWidget3.pyw`文件中的Python代码将如下所示：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoListWidget3 import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonAdd.clicked.connect(self.addlist)
        self.show()
    def addlist(self):
        self.ui.listWidgetSelectedItems.addItem(self.ui.
        lineEditFoodItem.text())
        self.ui.lineEditFoodItem.setText('')
        self.ui.lineEditFoodItem.setFocus()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

将按钮小部件的clicked()事件连接到`addlist`函数。因此，在在行编辑小部件中输入要添加到列表小部件中的文本后，当用户选择“添加到列表”按钮时，将调用`addlist`函数。`addlist`函数检索在行编辑中输入的文本，并将其添加到列表小部件中。然后，清除行编辑小部件中的文本，并将焦点设置在它上面，使用户能够输入不同的文本。

在下面的截图中，您可以看到用户在行编辑小部件中输入的文本在用户选择“添加到列表”按钮时添加到列表小部件中：

![](assets/29ad3de8-401c-4098-a53c-c40863d88069.png)

# 在列表小部件中执行操作

在这个示例中，您将学习如何在List Widget中执行不同的操作。List Widget基本上用于显示一组相似的项目，使用户能够选择所需的项目。因此，您需要向List Widget添加项目。此外，您可能需要编辑List Widget中的任何项目。有时，您可能需要从List Widget中删除项目。您可能还希望对List Widget执行的另一个操作是删除其中的所有项目，清除整个List Widget项目。在学习如何向List Widget添加、编辑和删除项目之前，让我们先了解列表项的概念。

# 准备工作

List Widget包含多个列表项。这些列表项是`QListWidgetItem`类的实例。可以使用`insertItem()`或`addItem()`方法将列表项插入List Widget中。列表项可以是文本或图标形式，并且可以被选中或取消选中。`QListWidgetItem`提供的方法如下。

# `QListWidgetItem`类提供的方法

让我们来看看`QListWidgetItem`类提供的以下方法：

+   `setText()`: 这个方法将指定的文本分配给列表项

+   `setIcon()`: 这个方法将指定的图标分配给列表项

+   `checkState()`: 这个方法根据列表项是选中还是未选中状态返回布尔值

+   `setHidden()`: 这个方法将布尔值true传递给这个方法以隐藏列表项

+   `isHidden()`: 如果列表项被隐藏，这个方法返回true

我们已经学会了向List Widget添加项目。如果您想编辑List Widget中的现有项目，或者您想从List Widget中删除项目，或者您想从List Widget中删除所有项目呢？

让我们通过创建一个应用程序来学习在列表小部件上执行不同的操作。这个应用程序将显示Line Edit，List Widget和一对Push Button小部件。您可以通过在Line Edit中输入文本，然后单击“Add”按钮来向List Widget添加项目。同样，您可以通过单击List Widget中的项目，然后单击“Edit”按钮来编辑List Widget中的任何项目。不仅如此，您甚至可以通过单击“Delete”按钮来删除List Widget中的任何项目。如果您想清除整个List Widget，只需单击“Delete All”按钮。

# 如何做....

执行以下步骤以了解如何在列表小部件上应用不同的操作；如何向列表小部件添加、编辑和删除项目；以及如何清除整个列表小部件：

1.  打开Qt Designer，基于无按钮模板创建一个新应用程序，并将一个标签、一个Line Edit、四个Push Button和List Widget小部件拖放到表单上。

1.  将标签小部件的文本属性设置为`Enter an item`。

1.  将四个Push Button小部件的文本属性设置为`Add`，`Edit`，`Delete`和`Delete All`。

1.  将四个Push Button小部件的objectName属性设置为`psuhButtonAdd`，`pushButtonEdit`，`pushButtonDelete`和`pushButtonDeleteAll`。

1.  将应用程序保存为`demoListWidgetOp.ui`。

表单将显示如下截图所示：

![](assets/a15bca18-c4e6-4674-8be7-c898809bc6dc.png)

需要使用`pyuic5`命令实用程序将XML文件`demoListWidgetOp.ui`转换为Python脚本。本书的源代码包中可以看到生成的Python文件`demoListWidgetOp.py`。

1.  创建一个名为`callListWidgetOp.pyw`的Python脚本文件，导入Python代码`demoListWidgetOp.py`，使您能够调用用户界面设计并在List Widget中添加、删除和编辑列表项。Python脚本`callListWidgetOp.pyw`中的代码如下所示：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication, QInputDialog, QListWidgetItem
from demoListWidgetOp import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.listWidget.addItem('Ice Cream')
        self.ui.listWidget.addItem('Soda')
        self.ui.listWidget.addItem('Coffee')
        self.ui.listWidget.addItem('Chocolate')
        self.ui.pushButtonAdd.clicked.connect(self.addlist)
        self.ui.pushButtonEdit.clicked.connect(self.editlist)
        self.ui.pushButtonDelete.clicked.connect(self.delitem)
        self.ui.pushButtonDeleteAll.clicked.connect
        (self.delallitems)
        self.show()
    def addlist(self):
        self.ui.listWidget.addItem(self.ui.lineEdit.text())
        self.ui.lineEdit.setText('')
        self.ui.lineEdit.setFocus()
    def editlist(self):
        row=self.ui.listWidget.currentRow()
        newtext, ok=QInputDialog.getText(self, "Enter new text", 
        "Enter new text")
        if ok and (len(newtext) !=0):
                self.ui.listWidget.takeItem(self.ui.listWidget.
                currentRow())
                self.ui.listWidget.insertItem(row,
                QListWidgetItem(newtext))
    def delitem(self):
        self.ui.listWidget.takeItem(self.ui.listWidget.
        currentRow())
    def delallitems(self):
        self.ui.listWidget.clear()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

`pushButtonAdd`的clicked()事件连接到`addlist`函数。同样，`pushButtonEdit`，`pushButtonDelete`和`pushButtonDeleteAll`对象的clicked()事件分别连接到`editlist`，`delitem`和`delallitems`函数。也就是说，单击任何按钮时，将调用相应的函数。`addlist`函数调用`addItem`函数来添加在Line Edit部件中输入的文本。`editlist`函数使用List Widget上的`currentRow`方法来找出要编辑的列表项目。

调用`QInputDialog`类的`getText`方法来提示用户输入新文本或编辑文本。在对话框中单击OK按钮后，当前列表项目将被对话框中输入的文本替换。`delitem`函数调用List Widget上的`takeItem`方法来删除当前行，即所选的列表项目。`delallitems`函数调用List Widget上的`clear`方法来清除或删除List Widget中的所有列表项目。

运行应用程序后，您将在Line Edit部件下方找到一个空的List Widget、Line Edit和Add按钮。在Line Edit部件中添加任何文本，然后单击添加按钮将该项目添加到List Widget中。在List Widget中添加了四个项目后，可能会显示如下截图所示：

![](assets/f3ab5ae8-fe93-46e6-bc4d-4edfb5f2f9bb.png)

让我们向List Widget中再添加一个项目Pizza。在Line Edit部件中输入`Pizza`，然后单击添加按钮。Pizza项目将被添加到List Widget中，如下截图所示：

![](assets/dac59a1d-1586-4ae6-96c2-86f6cbe5c05b.png)

假设我们要编辑List Widget中的Pizza项目，点击List Widget中的Pizza项目，然后点击编辑按钮。单击编辑按钮后，将弹出一个对话框，提示您输入一个新项目来替换Pizza项目。让我们在对话框中输入`Cold Drink`，然后单击OK按钮，如下截图所示：

![](assets/eb1432ea-258f-4eb5-b620-2b28fc01782f.png)

在下面的截图中，您可以看到列表部件中的Pizza项目被文本Cold Drink替换：

![](assets/d19c2923-1402-4399-8c88-3b3883a1cce3.png)

要从列表部件中删除任何项目，只需点击列表部件中的项目，然后点击删除按钮。让我们点击列表部件中的Coffee项目，然后点击删除按钮；如下截图所示，Coffee项目将从列表部件中删除：

![](assets/877ff850-e8c9-48eb-896c-27e14a12e9d7.png)

单击删除所有按钮后，整个List Widget项目将变为空，如下截图所示：

![](assets/bbf1152c-f749-4aad-82c3-184593fc8a81.png)

# 使用组合框部件

组合框用于从用户那里获取输入，并应用约束；也就是说，用户将以弹出列表的形式看到某些选项，他/她只能从可用选项中选择。与List Widget相比，组合框占用更少的空间。`QComboBox`类用于显示组合框。您不仅可以通过组合框显示文本，还可以显示`pixmaps`。以下是`QComboBox`类提供的方法：

| **方法** | **用途** |
| --- | --- |
| `setItemText()` | 设置或更改组合框中项目的文本。 |
| `removeItem()` | 从组合框中删除特定项目。 |
| `clear()` | 从组合框中删除所有项目。 |
| `currentText()` | 返回当前项目的文本，即当前选择的项目。 |
| `setCurrentIndex()` | 设置组合框的当前索引，即将组合框中的所需项目设置为当前选择的项目。 |
| `count()` | 返回组合框中项目的计数。 |
| `setMaxCount()` | 设置允许在组合框中的最大项目数。 |
| `setEditable()` | 使组合框可编辑，即用户可以编辑组合框中的项目。 |
| `addItem()` | 将指定内容附加到组合框中。 |
| `addItems()` | 将提供的每个字符串附加到组合框中。 |
| `itemText()` | 返回组合框中指定索引位置的文本。 |
| `currentIndex()` | 返回组合框中当前选择项目的索引位置。如果组合框为空或组合框中当前未选择任何项目，则该方法将返回`-1`作为索引。 |

以下是由`QComboBox`生成的信号：

| **信号** | **描述** |
| --- | --- |
| currentIndexChanged() | 当组合框的索引更改时发出，即用户在组合框中选择了一些新项目。 |
| activated() | 当用户更改索引时发出。 |
| highlighted() | 当用户在组合框中突出显示项目时发出。 |
| editTextChanged() | 当可编辑组合框的文本更改时发出。 |

为了实际了解组合框的工作原理，让我们创建一个示例。这个示例将通过一个组合框显示特定的银行账户类型，并提示用户选择他/她想要开设的银行账户类型。通过组合框选择的银行账户类型将通过`Label`小部件显示在屏幕上。

# 如何做…

以下是创建一个应用程序的步骤，该应用程序利用组合框显示某些选项，并解释了如何显示来自组合框的所选选项：

1.  创建一个没有按钮的对话框的新应用程序模板，从小部件框中拖动两个Label小部件和一个Combo Box小部件，并将它们放到表单中。

1.  将第一个Label小部件的文本属性设置为`选择您的账户类型`。

1.  删除第二个Label小部件的默认文本属性，因为其文本将通过代码设置。

1.  将组合框小部件的objectName属性设置为`comboBoxAccountType`。

1.  第二个Label小部件将用于显示用户选择的银行账户类型，因此将第二个Label小部件的objectName属性设置为`labelAccountType`。

1.  由于我们希望组合框小部件显示特定的银行账户类型，因此右键单击组合框小部件，并从打开的上下文菜单中选择编辑项目选项。

1.  逐个向组合框小部件添加一些银行账户类型。

1.  将应用程序保存为`demoComboBox.ui`。

1.  单击对话框底部显示的+按钮，将银行账户类型添加到组合框小部件中，如下截图所示：

![](assets/f949873b-a195-402c-b874-ed1b68424b5f.png)

1.  在添加所需的银行账户类型后，单击“确定”按钮退出对话框。表单现在将显示如下截图所示：

![](assets/d8e57706-8767-492e-9c85-d22cfaf04fbb.png)

使用Qt Designer创建的用户界面存储在`.ui`文件中，这是一个XML文件，需要转换为Python代码。可以使用`pyuic5`实用程序从XML文件生成Python代码。生成的文件`demoComboBox.py`可以在本书的源代码包中看到。

1.  将`demoComboBox.py`文件视为头文件，并将其导入到将调用其用户界面设计的文件中，这样您就可以访问组合框。

1.  创建另一个名为`callComboBox.pyw`的Python文件，并将`demoComboBox.py`的代码导入其中。Python脚本`callComboBox.pyw`中的代码如下所示：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoComboBox import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.comboBoxAccountType.currentIndexChanged.connect
        (self.dispAccountType)
        self.show()

    def dispAccountType(self):
        self.ui.labelAccountType.setText("You have selected 
        "+self.ui.comboBoxAccountType.itemText(self.ui.
        comboBoxAccountType.currentIndex())) 

if __name__=="__main__":   
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理…

在`demoComboBox.py`文件中，创建了一个名为顶级对象的类，其名称为`Ui_ prepended`。也就是说，对于顶级对象`Dialog`，创建了`Ui_Dialog`类，并存储了我们小部件的接口元素。该类包括两种方法，`setupUi`和`retranslateUi`。

`setupUi`方法创建了在Qt Designer中定义用户界面时使用的小部件。此方法还设置了小部件的属性。`setupUi`方法接受一个参数，即应用程序的顶层小部件，即`QDialog`的一个实例。`retranslateUi`方法用于翻译界面。

在`callComboBox.pyw`文件中，每当用户从组合框中选择任何项目时，`currentIndexChanged`信号将被发射，并且`currentIndexChanged`信号连接到`dispAccountType`方法，因此每当从组合框中选择任何项目时，`dispAccountType`方法将被调用。

在`dispAccountType`方法中，通过调用`QComboBox`类的`currentIndex`方法来访问当前选定的索引号，并将获取的索引位置传递给`QComboBox`类的`itemText`方法，以获取当前选定的组合框项目的文本。然后通过标签小部件显示当前选定的组合框项目。

运行应用程序时，您将看到一个下拉框显示四种银行账户类型：储蓄账户、活期账户、定期存款账户和定期存款账户，如下截图所示：

![](assets/1bb48fee-e2aa-42b9-8837-da3970018460.png)

从组合框中选择一个银行账户类型后，所选的银行账户类型将通过标签小部件显示，如下截图所示：

![](assets/7f09f615-47ce-4645-ae6d-308a5ee1430d.png)

# 使用字体组合框小部件

字体组合框小部件，顾名思义，显示一个可选择的字体样式列表。如果需要，所选的字体样式可以应用到所需的内容中。

# 准备工作

为了实际理解字体组合框小部件的工作原理，让我们创建一个示例。这个示例将显示一个字体组合框小部件和一个文本编辑小部件。用户可以在文本编辑小部件中输入所需的内容。在文本编辑小部件中输入文本后，当用户从字体组合框小部件中选择任何字体样式时，所选字体将被应用到文本编辑小部件中输入的内容。

# 如何做…

以下是显示活动字体组合框小部件并将所选字体应用于文本编辑小部件中的文本的步骤：

1.  创建一个没有按钮的对话框模板的新应用程序，并从小部件框中拖动两个标签小部件、一个字体组合框小部件和一个文本编辑小部件，并将它们放到表单上。

1.  将第一个标签小部件的文本属性设置为`选择所需的字体`，将第二个标签小部件的文本属性设置为`输入一些文本`。

1.  将应用程序保存为`demoFontComboBox.ui`。表单现在将显示如下截图所示：

![](assets/6422b45f-9351-46f8-a76a-87798511ab90.png)

使用Qt Designer创建的用户界面存储在一个`.ui`文件中，这是一个XML文件，需要转换为Python代码。转换为Python代码后，生成的文件`demoFontComboBox.py`将在本书的源代码包中可见。上述代码将被用作头文件，并被导入到需要GUI的文件中，也就是说，设计的用户界面可以通过简单地导入上述代码在任何Python脚本中访问。

1.  创建另一个名为`callFontFontComboBox.pyw`的Python文件，并将`demoFontComboBox.py`代码导入其中。

Python脚本`callFontComboBox.pyw`中的代码如下所示：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoFontComboBox import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        myFont=QtGui.QFont(self.ui.fontComboBox.itemText(self.ui.
        fontComboBox.currentIndex()),15)
        self.ui.textEdit.setFont(myFont)
        self.ui.fontComboBox.currentFontChanged.connect
        (self.changeFont)
        self.show()
    def changeFont(self):
        myFont=QtGui.QFont(self.ui.fontComboBox.itemText(self.ui.
        fontComboBox.currentIndex()),15)
        self.ui.textEdit.setFont(myFont)
if __name__=="__main__":   
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

在`callFontComboBox.pyw`文件中，每当用户从字体组合框小部件中选择任何字体样式时，将发射`currentFontChanged`信号，并且该信号连接到`changeFont`方法，因此每当从字体组合框小部件中选择任何字体样式时，将调用`changeFont()`方法。

在`changeFont()`方法中，通过调用两个方法来访问所选的字体样式。首先调用的是`QFontComboBox`类的`currentIndex()`方法，该方法获取所选字体样式的索引号。然后调用的是`itemText()`方法，并将当前所选字体样式的索引位置传递给该方法，以访问所选的字体样式。然后将所选的字体样式应用于文本编辑小部件中的内容。

运行应用程序时，您将看到一个字体组合框小部件，显示系统中可用的字体样式，如下截图所示：

![](assets/f886b743-3310-4daa-9245-852c3926c46a.png)

在文本编辑小部件中输入一些文本，并从字体组合框中选择所需的字体。所选的字体样式将应用于文本编辑小部件中的文本，如下截图所示：

![](assets/058646f2-31d3-4a67-adf1-60a195d177fd.png)

# 使用进度条小部件

进度条小部件在表示任何任务的进度时非常有用。无论是从服务器下载文件，还是在计算机上进行病毒扫描，或者其他一些关键任务，进度条小部件都有助于通知用户任务完成的百分比和待处理的百分比。随着任务的完成，进度条小部件不断更新，指示任务的进展。

# 准备工作

为了理解如何更新进度条以显示任何任务的进度，让我们创建一个示例。这个示例将显示一个进度条小部件，指示下载文件所需的总时间。当用户点击推送按钮开始下载文件时，进度条小部件将从0%逐渐更新到100%；也就是说，随着文件的下载，进度条将更新。当文件完全下载时，进度条小部件将显示100%。

# 如何做…

最初，进度条小部件为0%，为了使其增加，我们需要使用循环。随着进度条小部件表示的任务向完成的进展，循环将增加其值。循环值的每次增加都会增加进度条小部件的一些进度。以下是逐步过程，展示了如何更新进度条：

1.  从没有按钮的对话框模板创建一个新应用程序，并从小部件框中拖动一个标签小部件、一个进度条小部件和一个推送按钮小部件，然后将它们放到表单上。

1.  将标签小部件的文本属性设置为`下载文件`，将推送按钮小部件的文本属性设置为`开始下载`。

1.  将推送按钮小部件的objectName属性设置为`pushButtonStart`。

1.  将应用程序保存为`demoProgressBar.ui`。现在表单将显示如下截图所示：

![](assets/b1652ac9-b923-490a-b769-fb30070ac1f3.png)

使用Qt Designer创建的用户界面存储在`.ui`文件中，这是一个XML文件，需要转换为Python代码。生成的Python代码`demoProgressBar.py`可以在本书的源代码包中找到。上述代码将用作头文件，并导入到需要GUI的文件中；也就是说，代码中设计的用户界面可以通过简单导入上述代码在任何Python脚本中访问。

1.  创建另一个名为`callProgressBar.pyw`的Python文件，并将`demoProgressBar.py`代码导入其中。Python脚本`callProgressBar.pyw`中的代码如下所示：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoProgressBar import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonStart.clicked.connect(self.updateBar)
        self.show()

    def updateBar(self):
        x = 0
        while x < 100:
            x += 0.0001
            self.ui.progressBar.setValue(x)

if __name__=="__main__":   
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理…

在`callProgressBar.pyw`文件中，因为我们希望在按下按钮时进度条显示其进度，所以将进度条的clicked()事件连接到`updateBar()`方法，因此当按下按钮时，将调用`updateBar()`方法。在`updateBar()`方法中，使用了一个`while`循环，从`0`到`100`循环。一个变量`x`被初始化为值`0`。在while循环的每次迭代中，变量`x`的值增加了`0.0001`。在更新进度条时，将`x`变量的值应用于进度条。也就是说，每次while循环的迭代中，变量`x`的值都会增加，并且变量`x`的值会用于更新进度条。因此，进度条将从0%开始逐渐增加，直到达到100%。

在运行应用程序时，最初，您会发现进度条小部件为0%，底部有一个带有标题“开始下载”的按钮（请参见以下屏幕截图）。单击“开始下载”按钮，您会看到进度条开始逐渐显示进度。进度条会持续增加，直到达到100%，表示文件已完全下载：

![](assets/07e662a1-24bf-46c4-97b3-be0ce58f4577.png)
