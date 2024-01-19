# 使用 Qt 组件创建用户界面

在本章中，我们将学习使用以下小部件：

+   显示欢迎消息

+   使用单选按钮小部件

+   分组单选按钮

+   以复选框形式显示选项

+   显示两组复选框

# 介绍

我们将学习使用 Qt 工具包创建 GUI 应用程序。Qt 工具包，简称 Qt，是由 Trolltech 开发的跨平台应用程序和 UI 框架，用于开发 GUI 应用程序。它可以在多个平台上运行，包括 Windows、macOS X、Linux 和其他 UNIX 平台。它也被称为小部件工具包，因为它提供了按钮、标签、文本框、推按钮和列表框等小部件，这些小部件是设计 GUI 所必需的。它包括一组跨平台的类、集成工具和跨平台 IDE。为了创建实时应用程序，我们将使用 Python 绑定的 Qt 工具包，称为 PyQt5。

# PyQt

PyQt 是一个用于跨平台应用程序框架的 Python 绑定集合，结合了 Qt 和 Python 的所有优势。使用 PyQt，您可以在 Python 代码中包含 Qt 库，从而能够用 Python 编写 GUI 应用程序。换句话说，PyQt 允许您通过 Python 代码访问 Qt 提供的所有功能。由于 PyQt 依赖于 Qt 库来运行，因此在安装 PyQt 时，所需版本的 Qt 也会自动安装在您的计算机上。

GUI 应用程序可能包括一个带有多个对话框的主窗口，或者只包括一个对话框。一个小型 GUI 应用程序通常至少包括一个对话框。对话框应用程序包含按钮。它不包含菜单栏、工具栏、状态栏或中央小部件，而主窗口应用程序通常包括所有这些。

对话框有以下两种类型：

+   **模态**：这种对话框会阻止用户与应用程序的其他部分进行交互。对话框是用户可以与之交互的应用程序的唯一部分。在对话框关闭之前，无法访问应用程序的其他部分。

+   **非模态**：这种对话框与模态对话框相反。当非模态对话框处于活动状态时，用户可以自由地与对话框和应用程序的其他部分进行交互。

# 创建 GUI 应用程序的方式

有以下两种方式编写 GUI 应用程序：

+   使用简单文本编辑器从头开始

+   使用 Qt Designer，一个可视化设计工具，可以快速使用拖放功能创建用户界面

您将使用 Qt Designer 在 PyQt 中开发 GUI 应用程序，因为这是一种快速简便的设计用户界面的方法，无需编写一行代码。因此，双击桌面上的图标启动 Qt Designer。

打开时，Qt Designer 会要求您为新应用程序选择模板，如下截图所示：

![](img/4c0a403e-cd4c-427f-93aa-fcfdc443eefd.png)

Qt Designer 提供了适用于不同类型应用程序的多个模板。您可以选择其中任何一个模板，然后单击“创建”按钮。

Qt Designer 为新应用程序提供以下预定义模板：

+   带有底部按钮的对话框：此模板在右下角创建一个带有确定和取消按钮的表单。

+   带有右侧按钮的对话框：此模板在右上角创建一个带有确定和取消按钮的表单。

+   没有按钮的对话框：此模板创建一个空表单，您可以在其中放置小部件。对话框的超类是`QDialog`。

+   主窗口：此模板提供一个带有菜单栏和工具栏的主应用程序窗口，如果不需要可以删除。

+   小部件：此模板创建一个表单，其超类是`QWidget`而不是`QDialog`。

每个 GUI 应用程序都有一个顶级小部件，其余的小部件称为其子级。顶级小部件可以是`QDialog`、`QWidget`或`QMainWindow`，具体取决于您需要的模板。如果要基于对话框模板创建应用程序，则顶级小部件或您继承的第一个类将是`QDialog`。类似地，要基于主窗口模板创建应用程序，顶级小部件将是`QMainWindow`，要基于窗口小部件模板创建应用程序，您需要继承`QWidget`类。如前所述，用于用户界面的其余小部件称为这些类的子小部件。

Qt Designer 在顶部显示菜单栏和工具栏。它在左侧显示一个包含各种小部件的窗口小部件框，用于开发应用程序，分组显示。您只需从表单中拖放您想要的小部件即可。您可以在布局中排列小部件，设置它们的外观，提供初始属性，并将它们的信号连接到插槽。

# 显示欢迎消息

在这个示例中，用户将被提示输入他/她的名字，然后点击一个按钮。点击按钮后，将出现一个欢迎消息，“你好”，后面跟着用户输入的名字。对于这个示例，我们需要使用三个小部件，标签、行编辑和按钮。让我们逐个了解这些小部件。

# 理解标签小部件

标签小部件是`QLabel`类的一个实例，用于显示消息和图像。因为标签小部件只是显示计算结果，不接受任何输入，所以它们只是用于在屏幕上提供信息。

# 方法

以下是`QLabel`类提供的方法：

+   `setText()`: 该方法将文本分配给标签小部件

+   `setPixmap()`: 该方法将`pixmap`，`QPixmap`类的一个实例，分配给标签小部件

+   `setNum()`: 该方法将整数或双精度值分配给标签小部件

+   `clear()`: 该方法清除标签小部件中的文本

`QLabel`的默认文本是 TextLabel。也就是说，当您通过拖放标签小部件将`QLabel`类添加到表单时，它将显示 TextLabel。除了使用`setText()`，您还可以通过在属性编辑器窗口中设置其文本属性来为选定的`QLabel`对象分配文本。

# 理解行编辑小部件

行编辑小部件通常用于输入单行数据。行编辑小部件是`QLineEdit`类的一个实例，您不仅可以输入，还可以编辑数据。除了输入数据，您还可以在行编辑小部件中撤消、重做、剪切和粘贴数据。

# 方法

以下是`QLineEdit`类提供的方法：

+   `setEchoMode()`: 它设置行编辑小部件的回显模式。也就是说，它确定如何显示行编辑小部件的内容。可用选项如下：

+   `Normal`: 这是默认模式，它以输入的方式显示字符

+   `NoEcho`: 它关闭了行编辑的回显，也就是说，它不显示任何内容

+   `Password`: 该选项用于密码字段，不会显示文本；而是用户输入的文本将显示为星号

+   `PasswordEchoOnEdit`: 在编辑密码字段时显示实际文本，否则将显示文本的星号

+   `maxLength()`: 该方法用于指定可以在行编辑小部件中输入的文本的最大长度。

+   `setText()`: 该方法用于为行编辑小部件分配文本。

+   `text()`: 该方法访问在行编辑小部件中输入的文本。

+   `clear()`: 该方法清除或删除行编辑小部件的全部内容。

+   `setReadOnly()`:当将布尔值 true 传递给此方法时，它将使 LineEdit 小部件变为只读，即不可编辑。用户无法对通过 LineEdit 小部件显示的内容进行任何更改，但只能复制。

+   `isReadOnly()`:如果 LineEdit 小部件处于只读模式，则此方法返回布尔值 true，否则返回 false。

+   `setEnabled()`:默认情况下，LineEdit 小部件是启用的，即用户可以对其进行更改。但是，如果将布尔值 false 传递给此方法，它将禁用 LineEdit 小部件，因此用户无法编辑其内容，但只能通过`setText()`方法分配文本。

+   `setFocus()`:此方法将光标定位在指定的 LineEdit 小部件上。

# 了解 PushButton 小部件

要在应用程序中显示一个按钮，您需要创建一个`QPushButton`类的实例。在为按钮分配文本时，您可以通过在文本中的任何字符前加上一个和字符来创建快捷键。例如，如果分配给按钮的文本是`Click Me`，则字符`C`将被下划线标记，表示它是一个快捷键，用户可以通过按*Alt* + *C*来选择按钮。按钮在激活时发出 clicked()信号。除了文本，图标也可以显示在按钮中。在按钮中显示文本和图标的方法如下：

+   `setText()`:此方法用于为按钮分配文本

+   `setIcon()`:此方法用于为按钮分配图标

# 如何做...

让我们基于没有按钮的对话框模板创建一个新应用程序。如前所述，此应用程序将提示用户输入姓名，并在输入姓名后单击按钮后，应用程序将显示一个 hello 消息以及输入的姓名。以下是创建此应用程序的步骤：

1.  具有默认文本的另一个 Label 应该具有`labelResponse`的 objectName 属性

1.  从显示小部件类别中拖动一个 Label 小部件，并将其放在表单上。不要更改此 Label 小部件的文本属性，并将其文本属性保留为其默认值 TextLabel。这是因为此 Label 小部件的文本属性将通过代码设置，即将用于向用户显示 hello 消息。

1.  从输入小部件类别中拖动一个 LineEdit，并将其放在表单上。将其 objectName 属性设置为`lineEditName`。

1.  从按钮类别中拖动一个 PushButton 小部件，并将其放在表单上。将其 text 属性设置为`Click`。您可以通过以下三种方式之一更改 PushButton 小部件的 text 属性：通过双击 PushButton 小部件并覆盖默认文本，通过右键单击 PushButton 小部件并从弹出的上下文菜单中选择更改文本...选项，或者通过从属性编辑器窗口中选择文本属性并覆盖默认文本。

1.  将 PushButton 小部件的 objectName 属性设置为`ButtonClickMe`。

1.  将应用程序保存为`demoLineEdit.ui`。现在，表单将显示如下截图所示：

![](img/d296bf34-b970-46a3-af33-77336aebb427.png)

您使用 Qt Designer 创建的用户界面存储在一个`.ui`文件中，其中包括所有表单的信息：其小部件、布局等。`.ui`文件是一个 XML 文件，您需要将其转换为 Python 代码。这样，您可以在视觉界面和代码中实现的行为之间保持清晰的分离。

1.  要使用`.ui`文件，您首先需要将其转换为 Python 脚本。您将用于将`.ui`文件转换为 Python 脚本的命令实用程序是`pyuic5`。在 Windows 中，`pyuic5`实用程序与 PyQt 捆绑在一起。要进行转换，您需要打开命令提示符窗口并导航到保存文件的文件夹，并发出以下命令：

```py
C:\Pythonbook\PyQt5>pyuic5 demoLineEdit.ui -o demoLineEdit.py
```

假设我们将表单保存在此位置：`C:\Pythonbook\PyQt5>`。上述命令显示了`demoLineEdit.ui`文件转换为 Python 脚本`demoLineEdit.py`的过程。

此方法生成的 Python 代码不应手动修改，因为任何更改都将在下次运行`pyuic5`命令时被覆盖。

生成的 Python 脚本文件`demoLineEdit.py`的代码可以在本书的源代码包中找到。

1.  将`demoLineEdit.py`文件中的代码视为头文件，并将其导入到将调用其用户界面设计的文件中。

头文件是指那些被导入到当前文件中的文件。导入这些文件的命令通常写在脚本的顶部，因此被称为头文件。

1.  让我们创建另一个名为`callLineEdit.py`的 Python 文件，并将`demoLineEdit.py`的代码导入其中，如下所示：

```py
import sys from PyQt5.QtWidgets import QDialog, QApplication
from demoLineEdit import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.ButtonClickMe.clicked.connect(self.dispmessage)
        self.show()
    def dispmessage(self):
        self.ui.labelResponse.setText("Hello "
        +self.ui.lineEditName.text())
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

`demoLineEdit.py`文件非常容易理解。创建了一个名为顶级对象的类，前面加上`Ui_`。由于我们应用程序中使用的顶级对象是`Dialog`，因此创建了`Ui_Dialog`类，并存储了我们小部件的界面元素。该类有两个方法，`setupUi()`和`retranslateUi()`。`setupUi()`方法设置小部件；它创建了您在 Qt Designer 中定义用户界面时使用的小部件。该方法逐个创建小部件，并设置它们的属性。`setupUi()`方法接受一个参数，即创建用户界面（子小部件）的顶级小部件。在我们的应用程序中，它是`QDialog`的一个实例。`retranslateUi()`方法翻译界面。

让我们逐条理解`callLineEdit.py`的作用：

1.  它导入了必要的模块。`QWidget`是 PyQt5 中所有用户界面对象的基类。

1.  它创建了一个继承自基类`QDialog`的新`MyForm`类。

1.  它为`QDialog`提供了默认构造函数。默认构造函数没有父级，没有父级的小部件称为窗口。

1.  PyQt5 中的事件处理使用信号和槽。信号是一个事件，槽是在发生信号时执行的方法。例如，当您单击一个按钮时，会发生一个`clicked()`事件，也称为信号。`connect()`方法将信号与槽连接起来。在这种情况下，槽是一个方法：`dispmessage()`。也就是说，当用户单击按钮时，将调用`dispmessage()`方法。`clicked()`在这里是一个事件，事件处理循环等待事件发生，然后将其分派以执行某些任务。事件处理循环会继续工作，直到调用`exit()`方法或主窗口被销毁为止。

1.  它通过`QApplication()`方法创建了一个名为`app`的应用程序对象。每个 PyQt5 应用程序都必须创建`sys.argv`应用程序对象，其中包含从命令行传递的参数列表，并在创建应用程序对象时传递给方法。`sys.argv`参数有助于传递和控制脚本的启动属性。

1.  使用`MyForm`类的一个实例被创建，名为`w`。

1.  `show()`方法将在屏幕上显示小部件。

1.  `dispmessage()`方法执行按钮的事件处理。它显示 Hello 文本，以及在行编辑小部件中输入的名称。

1.  `sys.exit()`方法确保干净退出，释放内存资源。

`exec_()`方法有一个下划线，因为`exec`是 Python 关键字。

在执行上述程序时，您将获得一个带有行编辑和按钮小部件的窗口，如下截图所示。当选择按钮时，将执行`dispmessage()`方法，显示 Hello 消息以及输入在行编辑小部件中的用户名：

![](img/1aa5d22d-74f6-4f2d-9664-791751d4bedb.png)

# 使用单选按钮小部件

这个示例通过单选按钮显示特定的航班类型，当用户选择单选按钮时，将显示与该航班相关的价格。我们需要首先了解单选按钮的工作原理。

# 了解单选按钮

当您希望用户只能从可用选项中选择一个选项时，单选按钮小部件非常受欢迎。这些选项被称为互斥选项。当用户选择一个选项时，先前选择的选项将自动取消选择。单选按钮小部件是`QRadioButton`类的实例。每个单选按钮都有一个关联的文本标签。单选按钮可以处于选定（已选中）或未选定（未选中）状态。如果您想要两个或更多组单选按钮，其中每组允许单选按钮的互斥选择，请将它们放入不同的按钮组（`QButtonGroup`的实例）中。`QRadioButton`提供的方法如下所示。

# 方法

`QRadioButton`类提供以下方法：

+   `isChecked()`: 如果按钮处于选定状态，则此方法返回布尔值 true。

+   `setIcon()`: 此方法显示带有单选按钮的图标。

+   `setText()`: 此方法为单选按钮分配文本。如果您想为单选按钮指定快捷键，请在文本中使用和号（`&`）前置所选字符。快捷字符将被下划线标记。

+   `setChecked()`: 要使任何单选按钮默认选定，将布尔值 true 传递给此方法。

# 信号描述

`QRadioButton`发射的信号如下：

+   toggled(): 当按钮从选中状态变为未选中状态，或者反之时，将发射此信号

+   点击（）：当按钮被激活（即按下并释放）或者按下其快捷键时，将发射此信号

+   stateChanged(): 当单选按钮从选中状态变为未选中状态，或者反之时，将发射此信号

为了理解单选按钮的概念，让我们创建一个应用程序，询问用户选择航班类型，并通过单选按钮以`头等舱`，`商务舱`和`经济舱`的形式显示三个选项。通过单选按钮选择一个选项后，将显示该航班的价格。

# 如何做...

让我们基于没有按钮的对话框模板创建一个新的应用程序。这个应用程序将显示不同的航班类型以及它们各自的价格。当用户选择一个航班类型时，它的价格将显示在屏幕上：

1.  将两个标签小部件和三个单选按钮小部件拖放到表单上。

1.  将第一个标签小部件的文本属性设置为`选择航班类型`，并删除第二个标签小部件的文本属性。第二个标签小部件的文本属性将通过代码设置；它将用于显示所选航班类型的价格。

1.  将三个单选按钮小部件的文本属性设置为`头等舱 $150`，`商务舱 $125`和`经济舱 $100`。

1.  将第二个标签小部件的 objectName 属性设置为`labelFare`。三个单选按钮的默认对象名称分别为`radioButton`，`radioButton_2`和`radioButton_3`。将这三个单选按钮的 objectName 属性更改为`radioButtonFirstClass`，`radioButtonBusinessClass`和`radioButtonEconomyClass`。

1.  将应用程序保存为`demoRadioButton1.ui`。

看一下以下的屏幕截图：

![](img/545e660b-5490-4767-8101-c8f7ab6dea60.png)

`demoRadioButton1.ui`应用程序是一个 XML 文件，需要通过`pyuic5`命令实用程序转换为 Python 代码。本书的源代码包中可以看到生成的 Python 代码`demoRadioButton1.py`。

1.  将`demoRadioButton1.py`文件作为头文件导入到您即将创建的 Python 脚本中，以调用用户界面设计。

1.  在 Python 脚本中，编写代码根据用户选择的单选按钮显示飞行类型。将源文件命名为`callRadioButton1.py`；其代码如下所示：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoRadioButton1 import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.radioButtonFirstClass.toggled.connect(self.
        dispFare)
        self.ui.radioButtonBusinessClass.toggled.connect(self.
        dispFare)
        self.ui.radioButtonEconomyClass.toggled.connect(self.
        dispFare)
        self.show()
    def dispFare(self):
        fare=0
        if self.ui.radioButtonFirstClass.isChecked()==True:
            fare=150
        if self.ui.radioButtonBusinessClass.isChecked()==True:
            fare=125
        if self.ui.radioButtonEconomyClass.isChecked()==True:
            fare=100
        self.ui.labelFare.setText("Air Fare is "+str(fare))
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理

单选按钮的 toggled()事件连接到`dispFare()`函数，该函数将显示所选航班类型的价格。在`dispFare()`函数中，您检查单选按钮的状态。因此，如果选择了`radioButtonFirstClass`，则将值`150`分配给票价变量。同样，如果选择了`radioButtonBusinessClass`，则将值`125`分配给`fare`变量。同样，当选择`radioButtonEconomyClass`时，将值`100`分配给`fare`变量。最后，通过`labelFare`显示`fare`变量中的值。

在执行上一个程序时，您会得到一个对话框，其中显示了三种飞行类型，并提示用户选择要用于旅行的飞行类型。选择飞行类型后，所选飞行类型的价格将显示出来，如下面的屏幕截图所示：

![](img/1f097ec8-af6f-4177-8496-d35da2a29cc3.png)

# 分组单选按钮

在这个应用程序中，我们将学习创建两组单选按钮。用户可以从任一组中选择单选按钮，相应地结果或文本将出现在屏幕上。

# 准备工作

我们将显示一个对话框，其中显示不同尺码的衬衫和不同的付款方式。选择衬衫尺码和付款方式后，所选的衬衫尺码和付款方式将显示在屏幕上。我们将创建两组单选按钮，一组是衬衫尺码，另一组是付款方式。衬衫尺码组显示四个单选按钮，显示四种不同尺码的衬衫，例如 M、L、XL 和 XXL，其中 M 代表中号，L 代表大号，依此类推。付款方式组显示三个单选按钮，分别是借记/信用卡、网上银行和货到付款。用户可以从任一组中选择任何单选按钮。当用户选择任何衬衫尺码或付款方式时，所选的衬衫尺码和付款方式将显示出来。

# 如何做到...

让我们逐步重新创建前面的应用程序：

1.  基于无按钮对话框模板创建一个新应用程序。

1.  拖放三个 Label 小部件和七个 Radio Button 小部件。在这七个单选按钮中，我们将四个单选按钮排列在一个垂直布局中，将另外三个单选按钮排列在第二个垂直布局中。这两个布局将有助于将这些单选按钮分组。单选按钮是互斥的，只允许从布局或组中选择一个单选按钮。

1.  将前两个 Label 小部件的文本属性分别设置为`选择您的衬衫尺码`和`选择您的付款方式`。

1.  删除第三个 Label 小部件的文本属性，因为我们将通过代码显示所选的衬衫尺码和付款方式。

1.  在属性编辑器窗口中，增加所有小部件的字体大小，以增加它们在应用程序中的可见性。

1.  将前四个单选按钮的文本属性设置为`M`、`L`、`XL`和`XXL`。将这四个单选按钮排列成一个垂直布局。

1.  将接下来的三个单选按钮的文本属性设置为`借记/信用卡`、`网上银行`和`货到付款`。将这三个单选按钮排列成第二个垂直布局。请记住，这些垂直布局有助于将这些单选按钮分组。

1.  将前四个单选按钮的对象名称更改为`radioButtonMedium`、`radioButtonLarge`、`radioButtonXL`和`radioButtonXXL`。

1.  将第一个`VBoxLayout`布局的 objectName 属性设置为`verticalLayout`。`VBoxLayout`布局将用于垂直对齐单选按钮。

1.  将下一个三个单选按钮的对象名称更改为`radioButtonDebitCard`，`radioButtonNetBanking`和`radioButtonCashOnDelivery`。

1.  将第二个`QVBoxLayout`对象的 objectName 属性设置为`verticalLayout_2`。

1.  将第三个标签小部件的`objectName`属性设置为`labelSelected`。通过此标签小部件，将显示所选的衬衫尺寸和付款方式。

1.  将应用程序保存为`demoRadioButton2.ui`。

1.  现在，表单将显示如下截图所示：

![](img/8c185e1e-852e-46ca-bb90-d767f275c3ab.png)

然后，`.ui`（XML）文件通过`pyuic5`命令实用程序转换为 Python 代码。您可以在本书的源代码包中找到 Python 代码`demoRadioButton2.py`。

1.  将`demoRadioButton2.py`文件作为头文件导入我们的程序，以调用用户界面设计并编写代码，通过标签小部件显示所选的衬衫尺寸和付款方式，当用户选择或取消选择任何单选按钮时。

1.  让我们将程序命名为`callRadioButton2.pyw`；其代码如下所示：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from demoRadioButton2 import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.radioButtonMedium.toggled.connect(self.
        dispSelected)
        self.ui.radioButtonLarge.toggled.connect(self.
        dispSelected)
        self.ui.radioButtonXL.toggled.connect(self.dispSelected)
        self.ui.radioButtonXXL.toggled.connect(self.
        dispSelected)
        self.ui.radioButtonDebitCard.toggled.connect(self.
        dispSelected)
        self.ui.radioButtonNetBanking.toggled.connect(self.
        dispSelected)
        self.ui.radioButtonCashOnDelivery.toggled.connect(self.
        dispSelected)
        self.show()
    def dispSelected(self):
        selected1="";
        selected2=""
        if self.ui.radioButtonMedium.isChecked()==True:
            selected1="Medium"
        if self.ui.radioButtonLarge.isChecked()==True:
            selected1="Large"
        if self.ui.radioButtonXL.isChecked()==True:
            selected1="Extra Large"
        if self.ui.radioButtonXXL.isChecked()==True:
            selected1="Extra Extra Large"
        if self.ui.radioButtonDebitCard.isChecked()==True:
            selected2="Debit/Credit Card"
        if self.ui.radioButtonNetBanking.isChecked()==True:
            selected2="NetBanking"
        if self.ui.radioButtonCashOnDelivery.isChecked()==True:
            selected2="Cash On Delivery"
        self.ui.labelSelected.setText("Chosen shirt size is 
        "+selected1+" and payment method as " + selected2)
if __name__=="__main__":
    app = QApplication(sys.argv)
```

```py
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

所有单选按钮的`toggled()`事件都连接到`dispSelected()`函数，该函数将显示所选的衬衫尺寸和付款方式。在`dispSelected()`函数中，您检查单选按钮的状态，以确定它们是选中还是未选中。根据第一个垂直布局中选择的单选按钮，`selected1`变量的值将设置为`中号`、`大号`、`特大号`或`特特大号`。类似地，从第二个垂直布局中，根据所选的单选按钮，`selected2`变量的值将初始化为`借记卡/信用卡`、`网上银行`或`货到付款`。最后，通过`labelSelected`小部件显示分配给`selected1`变量和`selected`变量的衬衫尺寸和付款方式。运行应用程序时，会弹出对话框，提示您选择衬衫尺寸和付款方式。选择衬衫尺寸和付款方式后，所选的衬衫尺寸和付款方式将通过标签小部件显示，如下截图所示：

![](img/fbe0eafa-412a-4a05-8700-c88f95f491ee.png)

# 以复选框形式显示选项

在创建应用程序时，您可能会遇到需要为用户提供多个选项以供选择的情况。也就是说，您希望用户从一组选项中选择一个或多个选项。在这种情况下，您需要使用复选框。让我们更多地了解复选框。

# 准备就绪

而单选按钮只允许在组中选择一个选项，复选框允许您选择多个选项。也就是说，选择复选框不会影响应用程序中的其他复选框。复选框显示为文本标签，是`QCheckBox`类的一个实例。复选框可以处于三种状态之一：选中（已选中）、未选中（未选中）或三态（未更改）。三态是一种无变化状态；用户既没有选中也没有取消选中复选框。

# 方法应用

以下是`QCheckBox`类提供的方法：

+   `isChecked()`: 如果复选框被选中，此方法返回布尔值 true，否则返回 false。

+   `setTristate()`: 如果您不希望用户更改复选框的状态，请将布尔值 true 传递给此方法。用户将无法选中或取消选中复选框。

+   `setIcon()`: 此方法用于显示复选框的图标。

+   `setText()`: 此方法将文本分配给复选框。要为复选框指定快捷键，请在文本中的首选字符前加上一个和字符。快捷字符将显示为下划线。

+   `setChecked()`: 为了使复选框默认显示为选中状态，请将布尔值 true 传递给此方法。

# 信号描述

`QCheckBox`发出的信号如下：

+   clicked(): 当复选框被激活（即按下并释放）或按下其快捷键时，将发出此信号

+   stateChanged(): 每当复选框从选中到未选中或反之亦然时，将发出此信号

理解复选框小部件，让我们假设您经营一家餐厅，销售多种食物，比如比萨。比萨可以搭配不同的配料，比如额外的奶酪，额外的橄榄等，每种配料的价格也会显示出来。用户可以选择普通比萨并加上一个或多个配料。您希望的是，当选择了配料时，比萨的总价，包括所选的配料，会显示出来。

# 操作步骤...

本教程的重点是理解当复选框的状态从选中到未选中或反之时如何触发操作。以下是创建这样一个应用程序的逐步过程：

1.  首先，基于无按钮的对话框模板创建一个新应用程序。

1.  将三个标签小部件和三个复选框小部件拖放到表单上。

1.  将前两个标签小部件的文本属性设置为`Regular Pizza $10`和`Select your extra toppings`。

1.  在属性编辑器窗口中，增加所有三个标签和复选框的字体大小，以增加它们在应用程序中的可见性。

1.  将三个复选框的文本属性设置为`Extra Cheese $1`，`Extra Olives $1`和`Extra Sausages $2`。三个复选框的默认对象名称分别为`checkBox`，`checkBox_2`和`checkBox_3`。

1.  分别更改为`checkBoxCheese`，`checkBoxOlives`和`checkBoxSausages`。

1.  将标签小部件的 objectName 属性设置为`labelAmount`。

1.  将应用程序保存为`demoCheckBox1.ui`。现在，表单将显示如下截图所示：

![](img/a49fb108-5909-488f-98fb-addd10717ebf.png)

然后，通过`pyuic5`命令实用程序将`.ui`（XML）文件转换为 Python 代码。在本书的源代码包中可以看到生成的`demoCheckBox1.py`文件中的 Python 代码。

1.  将`demoCheckBox1.py`文件作为头文件导入我们的程序，以调用用户界面设计并编写代码，通过标签小部件计算普通比萨的总成本以及所选的配料，当用户选择或取消选择任何复选框时。

1.  让我们将程序命名为`callCheckBox1.pyw`；其代码如下所示：

```py
import sys
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from demoCheckBox1 import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.checkBoxCheese.stateChanged.connect(self.
        dispAmount)
        self.ui.checkBoxOlives.stateChanged.connect(self.
        dispAmount)
        self.ui.checkBoxSausages.stateChanged.connect(self.
        dispAmount)
        self.show()
    def dispAmount(self):
        amount=10
        if self.ui.checkBoxCheese.isChecked()==True:
            amount=amount+1
        if self.ui.checkBoxOlives.isChecked()==True:
            amount=amount+1
        if self.ui.checkBoxSausages.isChecked()==True:
            amount=amount+2
        self.ui.labelAmount.setText("Total amount for pizza is 
        "+str(amount))
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

将复选框的 stateChanged()事件连接到`dispAmount`函数，该函数将计算所选配料的比萨的成本。在`dispAmount`函数中，您检查复选框的状态，以找出它们是选中还是未选中。被选中的复选框的配料成本被添加并存储在`amount`变量中。最后，存储在`amount`变量中的金额加法通过`labelAmount`显示出来。运行应用程序时，会弹出对话框提示您选择要添加到普通比萨中的配料。选择任何配料后，普通比萨的金额以及所选的配料将显示在屏幕上，如下截图所示：

![](img/4ed74906-5046-4431-99fb-2da485c83932.png)每当任何复选框的状态改变时，`dispAmount`函数将被调用。因此，只要勾选或取消任何复选框，总金额将通过标签小部件显示出来。

# 显示两组复选框

在这个应用程序中，我们将学习如何制作两组复选框。用户可以从任一组中选择任意数量的复选框，相应的结果将显示出来。

# 准备工作

我们将尝试显示一家餐厅的菜单，那里供应不同类型的冰淇淋和饮料。我们将创建两组复选框，一组是冰淇淋，另一组是饮料。冰淇淋组显示四个复选框，显示四种不同类型的冰淇淋，薄荷巧克力片、曲奇面团等，以及它们的价格。饮料组显示三个复选框，咖啡、苏打水等，以及它们的价格。用户可以从任一组中选择任意数量的复选框。当用户选择任何冰淇淋或饮料时，所选冰淇淋和饮料的总价格将显示出来。

# 操作步骤...

以下是创建应用程序的步骤，解释了如何将复选框排列成不同的组，并在任何组的任何复选框的状态发生变化时采取相应的操作：

1.  基于没有按钮的对话框模板创建一个新的应用程序。

1.  将四个标签小部件、七个复选框小部件和两个分组框小部件拖放到表单上。

1.  将前三个标签小部件的文本属性分别设置为`菜单`，`选择您的冰淇淋`和`选择您的饮料`。

1.  删除第四个标签小部件的文本属性，因为我们将通过代码显示所选冰淇淋和饮料的总金额。

1.  通过属性编辑器，增加所有小部件的字体大小，以增加它们在应用程序中的可见性。

1.  将前四个复选框的文本属性设置为`Mint Choclate Chips $4`，`Cookie Dough $2`，`Choclate Almond $3`和`Rocky Road $5`。将这四个复选框放入第一个分组框中。

1.  将接下来三个复选框的文本属性设置为`Coffee $2`，`Soda $3`和`Tea $1`。将这三个复选框放入第二个分组框中。

1.  将前四个复选框的对象名称更改为`checkBoxChoclateChips`，`checkBoxCookieDough`，`checkBoxChoclateAlmond`和`checkBoxRockyRoad`。

1.  将第一个分组框的`objectName`属性设置为`groupBoxIceCreams`。

1.  将接下来三个复选框的`objectName`属性更改为`checkBoxCoffee`，`checkBoxSoda`和`checkBoxTea`。

1.  将第二个分组框的`objectName`属性设置为`groupBoxDrinks`。

1.  将第四个标签小部件的`objectName`属性设置为`labelAmount`。

1.  将应用程序保存为`demoCheckBox2.ui`。通过这个标签小部件，所选冰淇淋和饮料的总金额将显示出来，如下面的屏幕截图所示：

![](img/0c376114-b3d1-4706-8f83-f503e597ac9f.png)

然后，通过`pyuic5`命令实用程序将`.ui`（XML）文件转换为 Python 代码。您可以在本书的源代码包中找到生成的 Python 代码`demoCheckbox2.py`文件。

1.  在我们的程序中将`demoCheckBox2.py`文件作为头文件导入，以调用用户界面设计，并编写代码来通过标签小部件计算冰淇淋和饮料的总成本，当用户选择或取消选择任何复选框时。

1.  让我们将程序命名为`callCheckBox2.pyw`；其代码如下所示：

```py
import sys
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from demoCheckBox2 import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.checkBoxChoclateAlmond.stateChanged.connect
        (self.dispAmount)
        self.ui.checkBoxChoclateChips.stateChanged.connect(self.
        dispAmount)
        self.ui.checkBoxCookieDough.stateChanged.connect(self.
        dispAmount)
        self.ui.checkBoxRockyRoad.stateChanged.connect(self.
        dispAmount)
        self.ui.checkBoxCoffee.stateChanged.connect(self.
        dispAmount)
        self.ui.checkBoxSoda.stateChanged.connect(self.
        dispAmount)
        self.ui.checkBoxTea.stateChanged.connect(self.
        dispAmount)
        self.show()
    def dispAmount(self):
        amount=0
        if self.ui.checkBoxChoclateAlmond.isChecked()==True:
            amount=amount+3
        if self.ui.checkBoxChoclateChips.isChecked()==True:
            amount=amount+4
        if self.ui.checkBoxCookieDough.isChecked()==True:
            amount=amount+2
        if self.ui.checkBoxRockyRoad.isChecked()==True:
            amount=amount+5
        if self.ui.checkBoxCoffee.isChecked()==True:
            amount=amount+2
        if self.ui.checkBoxSoda.isChecked()==True:
            amount=amount+3
        if self.ui.checkBoxTea.isChecked()==True:
            amount=amount+1
        self.ui.labelAmount.setText("Total amount is 
        $"+str(amount))
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

所有复选框的`stateChanged()`事件都连接到`dispAmount`函数，该函数将计算所选冰淇淋和饮料的成本。在`dispAmount`函数中，您检查复选框的状态，以找出它们是选中还是未选中。选中复选框的冰淇淋和饮料的成本被添加并存储在`amount`变量中。最后，通过`labelAmount`小部件显示存储在`amount`变量中的金额的总和。运行应用程序时，会弹出对话框提示您选择要订购的冰淇淋或饮料。选择冰淇淋或饮料后，所选项目的总金额将显示出来，如下面的屏幕截图所示：

![](img/9e489150-d6f2-4a81-bb85-67494f09c7fd.png)
