# 理解对话框

在本章中，我们将学习如何使用以下类型的对话框：

+   输入对话框

+   使用输入对话框

+   使用颜色对话框

+   使用字体对话框

+   使用文件对话框

# 介绍

在所有应用程序中都需要对话框来从用户那里获取输入，还要指导用户输入正确的数据。交互式对话框也使应用程序变得非常用户友好。基本上有以下两种类型的对话框：

+   **模态对话框**：模态对话框是一种要求用户输入强制信息的对话框。这种对话框在关闭之前不允许用户使用应用程序的其他部分。也就是说，用户需要在模态对话框中输入所需的信息，关闭对话框后，用户才能访问应用程序的其余部分。

+   **非模态或无模式对话框**：这些对话框使用户能够与应用程序的其余部分和对话框进行交互。也就是说，用户可以在保持无模式对话框打开的同时继续与应用程序的其余部分进行交互。这就是为什么无模式对话框通常用于从用户那里获取非必要或非关键信息。

# 输入对话框

使用`QInputDialog`类来创建输入对话框。`QInputDialog`类提供了一个对话框，用于从用户那里获取单个值。提供的输入对话框包括一个文本字段和两个按钮，OK 和 Cancel。文本字段使我们能够从用户那里获取单个值，该单个值可以是字符串、数字或列表中的项目。以下是`QInputDialog`类提供的方法，用于接受用户不同类型的输入：

+   `getInt()`:该方法显示一个旋转框以接受整数。要从用户那里得到一个整数，您需要使用以下语法：

```py
getInt(self, window title, label before LineEdit widget, default value, minimum, maximum and step size)
```

看一下下面的例子：

```py
quantity, ok = QInputDialog.getInt(self, "Order Quantity", "Enter quantity:", 2, 1, 100, 1)
```

前面的代码提示用户输入数量。如果用户没有输入任何值，则默认值`2`将被赋给`quantity`变量。用户可以输入`1`到`100`之间的任何值。

+   `getDouble()`:该方法显示一个带有浮点数的旋转框，以接受小数值。要从用户那里得到一个小数值，您需要使用以下语法：

```py
getDouble(self, window title, label before LineEdit widget, default value, minimum, maximum and number of decimal places desired)
```

看一下下面的例子：

```py
price, ok = QInputDialog.getDouble(self, "Price of the product", "Enter price:", 1.50,0, 100, 2)
```

前面的代码提示用户输入产品的价格。如果用户没有输入任何值，则默认值`1.50`将被赋给`price`变量。用户可以输入`0`到`100`之间的任何值。

+   `getText()`:该方法显示一个 Line Edit 小部件，以从用户那里接受文本。要从用户那里获取文本，您需要使用以下语法：

```py
getText(self, window title, label before LineEdit widget)
```

看一下下面的例子：

```py
name, ok = QtGui.QInputDialog.getText(self, 'Get Customer Name', 'Enter your name:')
```

前面的代码将显示一个标题为“获取客户名称”的输入对话框。对话框还将显示一个 Line Edit 小部件，允许用户输入一些文本。在 Line Edit 小部件之前还将显示一个 Label 小部件，显示文本“输入您的姓名:”。在对话框中输入的客户姓名将被赋给`name`变量。

+   `getItem()`:该方法显示一个下拉框，显示多个可供选择的项目。要从下拉框中获取项目，您需要使用以下语法：

```py
getItem(self, window title, label before combo box, array , current item, Boolean Editable)
```

这里，`array`是需要在下拉框中显示的项目列表。`current item`是在下拉框中被视为当前项目的项目。`Editable`是布尔值，如果设置为`True`，则意味着用户可以编辑下拉框并输入自己的文本。当`Editable`设置为`False`时，这意味着用户只能从下拉框中选择项目，但不能编辑项目。看一下下面的例子：

```py
countryName, ok = QInputDialog.getItem(self, "Input Dialog", "List of countries", countries, 0, False)
```

上述代码将显示一个标题为“输入对话框”的输入对话框。对话框显示一个下拉框，其中显示了通过 countries 数组的元素显示的国家列表。下拉框之前的 Label 小部件显示文本“国家列表”。从下拉框中选择的国家名称将被分配给`countryName`变量。用户只能从下拉框中选择国家，但不能编辑任何国家名称。

# 使用输入对话框

输入对话框可以接受任何类型的数据，包括整数、双精度和文本。在本示例中，我们将学习如何从用户那里获取文本。我们将利用输入对话框来了解用户所居住的国家的名称。

输入对话框将显示一个显示不同国家名称的下拉框。通过名称选择国家后，所选的国家名称将显示在文本框中。

# 如何做...

让我们根据没有按钮的对话框模板创建一个新的应用程序，执行以下步骤：

1.  由于应用程序将提示用户通过输入对话框选择所居住的国家，因此将一个 Label 小部件、一个 Line Edit 小部件和一个 Push Button 小部件拖放到表单中。

1.  将 Label 小部件的文本属性设置为“你的国家”。

1.  将 Push Button 小部件的文本属性设置为“选择国家”。

1.  将 Line Edit 小部件的 objectName 属性设置为`lineEditCountry`。

1.  将 Push Button 小部件的 objectName 属性设置为`pushButtonCountry`。

1.  将应用程序保存为`demoInputDialog.ui`。

现在表单将如下所示：

![](img/3a59be73-9686-4b0a-92a6-c5ea8a7bf4b3.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。

1.  要进行转换，您需要打开一个命令提示符窗口，导航到保存文件的文件夹，并发出以下命令行：

```py
C:\Pythonbook\PyQt5>pyuic5 demoInputDialog.ui -o demoInputDialog.py
```

您可以在本书的源代码包中找到生成的 Python 脚本`demoInputDialog.py`。

1.  将`demoInputDialog.py`脚本视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callInputDialog.pyw`的 Python 文件，并将`demoInputDialog.py`的代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication, QInputDialog
from demoInputDialog import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonCountry.clicked.connect(self.dispmessage)
        self.show()
    def dispmessage(self):
        countries = ("Albania", "Algeria", "Andorra", "Angola",   
        "Antigua and Barbuda", "Argentina", "Armenia", "Aruba", 
        "Australia", "Austria", "Azerbaijan")
        countryName, ok = QInputDialog.getItem(self, "Input  
        Dialog", "List of countries", countries, 0, False)
        if ok and countryName:
            self.ui.lineEditCountry.setText(countryName)
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

在`demoInputDialog.py`文件中，创建一个名为顶层对象的类，前面加上`Ui_`。也就是说，对于顶层对象 Dialog，创建了`Ui_Dialog`类，并存储了我们小部件的接口元素。该类有两个方法，`setupUi()`和`retranslateUi()`。

`setupUi()`方法创建了在 Qt Designer 中定义用户界面中使用的小部件。此方法还设置了小部件的属性。`setupUi()`方法接受一个参数，即应用程序的顶层小部件，即`QDialog`的一个实例。`retranslateUi()`方法翻译了界面。

在`callInputDialog.pyw`文件中，可以看到 Push Button 小部件的单击事件连接到`dispmessage()`方法，该方法用于选择国家；当用户单击推送按钮时，将调用`dispmessage()`方法。`dispmessage()`方法定义了一个名为 countries 的字符串数组，其中包含了几个国家名称的数组元素。之后，调用`QInputDialog`类的`getItem`方法，打开一个显示下拉框的输入对话框。当用户单击下拉框时，它会展开，显示分配给`countries`字符串数组的国家名称。当用户选择一个国家，然后单击对话框中的 OK 按钮，所选的国家名称将被分配给`countryName`变量。然后，所选的国家名称将通过 Line Edit 小部件显示出来。

运行应用程序时，您将得到一个空的 Line Edit 小部件和一个名为“选择国家”的推送按钮，如下截图所示：

![](img/eb1fe98b-1676-4764-b7e0-ca45a599d5b5.png)

单击“选择国家”按钮后，输入对话框框将打开，如下截图所示。输入对话框显示一个组合框以及两个按钮“确定”和“取消”。单击组合框，它将展开显示所有国家名称，如下截图所示：

![](img/33a4befd-4197-4c3e-920c-fc7ce186433d.png)

从组合框中选择国家名称，然后单击“确定”按钮后，所选国家名称将显示在行编辑框中，如下截图所示：

![](img/362a4ebc-bc81-4595-826b-21c961807936.png)

# 使用颜色对话框

在本教程中，我们将学习使用颜色对话框显示颜色调色板，允许用户从调色板中选择预定义的颜色或创建新的自定义颜色。

该应用程序包括一个框架，当用户从颜色对话框中选择任何颜色时，所选颜色将应用于框架。除此之外，所选颜色的十六进制代码也将通过 Label 小部件显示。

在本教程中，我们将使用`QColorDialog`类，该类提供了一个用于选择颜色值的对话框小部件。

# 如何做到...

让我们根据以下步骤创建一个基于无按钮对话框模板的新应用程序：

1.  将一个 Push Button、一个 Frame 和一个 Label 小部件拖放到表单上。

1.  将 Push Button 小部件的文本属性设置为“选择颜色”。

1.  将 Push Button 小部件的 objectName 属性设置为`pushButtonColor`。

1.  将 Frame 小部件的 objectName 属性设置为`frameColor`。

1.  将 Label 小部件设置为`labelColor`。

1.  将应用程序保存为`demoColorDialog.ui`。

表格现在将如下所示：

![](img/134219b5-01b1-4f44-bd45-52d5cdd09caa.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件。您可以使用`pyuic5`实用程序将 XML 文件转换为 Python 代码。生成的 Python 脚本`demoColorDialog.py`可以在本书的源代码包中找到。`demoColorDialog.py`脚本将用作头文件，并将在另一个 Python 脚本文件中导入，该文件将调用此用户界面设计。

1.  创建另一个名为`callColorDialog.pyw`的 Python 文件，并将`demoColorDialog.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication, QColorDialog
from PyQt5.QtGui import QColor
from demoColorDialog import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        col = QColor(0, 0, 0)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.frameColor.setStyleSheet("QWidget { background-
        color: %s }" % col.name())
        self.ui.pushButtonColor.clicked.connect(self.dispcolor)
        self.show()
    def dispcolor(self):
        col = QColorDialog.getColor()
        if col.isValid():
        self.ui.frameColor.setStyleSheet("QWidget { background-  
        color: %s }" % col.name())
        self.ui.labelColor.setText("You have selected the color with 
        code: " + str(col.name()))
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

在`callColorDialog.pyw`文件中，您可以看到按钮的 click()事件连接到`dispcolor()`方法；也就是说，当用户单击“选择颜色”按钮时，将调用`dispcolor()`方法。`dispmessage()`方法调用`QColorDialog`类的`getColor()`方法，打开一个显示不同颜色的对话框。用户不仅可以从对话框中选择任何预定义的基本颜色，还可以创建新的自定义颜色。选择所需的颜色后，当用户从颜色对话框中单击“确定”按钮时，所选颜色将通过在 Frame 小部件类上调用`setStyleSheet()`方法来分配给框架。此外，所选颜色的十六进制代码也通过 Label 小部件显示。

运行应用程序时，最初会看到一个按钮“选择颜色”，以及一个默认填充为黑色的框架，如下截图所示：

![](img/49d02cee-9ab4-4642-8246-6ab2bd1c733b.png)

单击“选择颜色”按钮，颜色对话框将打开，显示以下截图中显示的基本颜色。颜色对话框还可以让您创建自定义颜色：

![](img/b1c886f2-3210-4155-b51b-7083e10959cf.png)

选择颜色后，单击“确定”按钮，所选颜色将应用于框架，并且所选颜色的十六进制代码将通过 Label 小部件显示，如下截图所示：

![](img/594e7f78-ec74-4168-a3ac-caa5adae3102.png)

# 使用字体对话框

在本教程中，我们将学习使用字体对话框为所选文本应用不同的字体和样式。

在这个应用程序中，我们将使用 Text Edit 小部件和 Push Button 小部件。点击按钮后，将打开字体对话框。从字体对话框中选择的字体和样式将应用于 Text Edit 小部件中的文本。

在这个示例中，我们将使用`QFontDialog`类，该类显示一个用于选择字体的对话框小部件。

# 如何做...

让我们根据无按钮模板创建一个新的应用程序，执行以下步骤：

1.  将一个 Push Button 和一个 Text Edit 小部件拖放到表单上。

1.  将 Push Button 小部件的文本属性设置为`Choose Font`。

1.  将 Push Button 小部件的 objectName 属性设置为`pushButtonFont`。

1.  将应用程序保存为`demoFontDialog.ui`。

1.  执行上述步骤后，应用程序将显示如下截图所示：

![](img/2c394b22-bf10-464e-828c-e79074e2cb4d.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件。使用`pyuic5`命令，您可以将 XML 文件转换为 Python 代码。生成的 Python 脚本`demoFontDialog.py`可以在本书的源代码包中找到。`demoFontDialog.py`脚本将被用作头文件，并将在另一个 Python 脚本文件中导入，该文件将调用此用户界面设计。

1.  创建另一个名为`callFontDialog.pyw`的 Python 文件，并将`demoFontDialog.py`代码导入其中。

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication, QFontDialog
from demoFontDialog import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonFont.clicked.connect(self.changefont)
        self.show()
    def changefont(self):
        font, ok = QFontDialog.getFont()
        if ok:
        self.ui.textEdit.setFont(font)
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

在`callFontDialog.pyw`文件中，您可以看到将 push button 的 click()事件连接到`changefont()`方法；也就是说，当用户点击 Choose Font 按钮时，将调用`change()`方法。`changefont()`方法调用`QFontDialog`类的`getFont()`方法，打开一个对话框，显示不同的字体、字体样式、大小和效果。选择字体、字体样式、大小或效果后，将在示例框中显示文本的选择效果。选择所需的字体、字体样式、大小和效果后，当用户点击 OK 按钮时，所选的选择将被分配给`font`变量。随后，在`TextEdit`类上调用`setFont()`方法，将所选的字体和样式应用于通过 Text Edit 小部件显示的文本。

运行应用程序后，您将看到一个按钮，Change Font 小部件和 Text Edit 小部件，如下截图所示：

![](img/7cf1dce1-7e0a-4cfe-978e-601477340c7c.png)

要查看从字体对话框中选择的字体的影响，您需要在 Text Edit 小部件中输入一些文本，如下截图所示：

![](img/e93a5f1c-e881-4674-85ee-e3386739f82a.png)

选择 Change Font 按钮后，字体对话框将打开，如下截图所示。您可以看到不同的字体名称将显示在最左边的选项卡上。中间选项卡显示不同的字体样式，使您可以使文本以粗体、斜体、粗斜体和常规形式显示。最右边的选项卡显示不同的大小。在底部，您可以看到不同的复选框，使您可以使文本显示为下划线、删除线等。从任何选项卡中选择选项，所选字体和样式对示例框中显示的示例文本的影响可见。选择所需的字体和样式后，点击 OK 按钮关闭字体对话框：

![](img/4448d866-eb2e-407b-a9ef-97ac098f20d8.png)

所选字体和样式的效果将显示在 Text Edit 小部件中显示的文本上，如下截图所示：

![](img/48b1df37-7b6a-4b51-acce-79ffce0be77e.png)

# 使用文件对话框

在这个示例中，我们将学习使用文件对话框，了解如何执行不同的文件操作，如打开文件和保存文件。

我们将学习创建一个包含两个菜单项 Open 和 Save 的文件菜单。单击 Open 菜单项后，将打开文件打开对话框，帮助浏览和选择要打开的文件。打开文件的文件内容将显示在文本编辑框中。用户甚至可以在需要时更新文件内容。在对文件进行所需的修改后，当用户从文件菜单中单击 Save 选项时，文件内容将被更新。

# 准备工作

在这个教程中，我们将使用`QFileDialog`类，该类显示一个对话框，允许用户选择文件或目录。文件可以用于打开和保存。

在这个教程中，我将使用`QFileDialog`类的以下两种方法：

+   `getOpenFileName()`: 该方法打开文件对话框，使用户可以浏览目录并打开所需的文件。`getOpenFileName()`方法的语法如下：

```py
file_name = QFileDialog.getOpenFileName(self, dialog_title, path, filter)
```

在上述代码中，`filter`表示文件扩展名；它确定要显示的文件类型，例如如下所示：

```py
file_name = QFileDialog.getOpenFileName(self, 'Open file', '/home')

In the preceding example, file dialog is opened that shows all the files of home directory to browse from.

file_name = QFileDialog.getOpenFileName(self, 'Open file', '/home', "Images (*.png *.jpg);;Text files (.txt);;XML files (*.xml)")
```

在上面的示例中，您可以看到来自`home`目录的文件。对话框中将显示扩展名为`.png`、`.jpg`、`.txt`和`.xml`的文件。

+   `getSaveFileName()`: 该方法打开文件保存对话框，使用户可以以所需的名称和所需的文件夹保存文件。`getSaveFileName()`方法的语法如下：

```py
file_name = QFileDialog.getSaveFileName(self, dialog_title, path, filter, options)
```

`options`表示如何运行对话框的各种选项，例如，请查看以下代码：

```py
file_name, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)

In the preceding example, the File Save dialog box will be opened allowing you to save the files with the desired extension. If you don't specify the file extension, then it will be saved with the default extension, .txt.
```

# 如何操作...

让我们基于主窗口模板创建一个新的应用程序。主窗口模板默认包含顶部的菜单：

1.  我们甚至可以使用两个按钮来启动文件打开对话框和文件保存对话框，但使用菜单项来启动文件操作将给人一种实时应用程序的感觉。

1.  主窗口模板中的默认菜单栏显示“Type Here”代替菜单名称。

1.  “Type Here”选项表示用户可以输入所需的菜单名称，替换“Type Here”文本。让我们输入`File`，在菜单栏中创建一个菜单。

1.  按下*Enter*键后，术语“Type Here”将出现在文件菜单下的菜单项中。

1.  在文件菜单中将 Open 作为第一个菜单项。

1.  在创建第一个菜单项 Open 后按下*Enter*键后，术语“Type Here”将出现在 Open 下方。

1.  用菜单项 Save 替换 Type Here。

1.  创建包含两个菜单项 Open 和 Save 的文件菜单后

1.  应用程序将显示如下截图所示：

![](img/b905aec6-f952-4af1-aeb4-ebab19256086.png)

在属性编辑器窗口下方的操作编辑器窗口中，可以看到 Open 和 Save 菜单项的默认对象名称分别为`actionOpen`和`actionSave`。操作编辑器窗口中的 Shortcut 选项卡目前为空，因为尚未为任何菜单项分配快捷键：

![](img/f68b6bba-5a63-43b9-b8b2-5d1fbc82a9e3.png)

1.  要为 Open 菜单项分配快捷键，双击`actionOpen`菜单项的 Shortcut 选项卡中的空白处。您将得到如下截图所示的对话框：

![](img/836d5b52-ff7e-4240-b084-25b0a393679f.png)

文本、对象名称和工具提示框会自动填充默认文本。

1.  单击 Shortcut 框以将光标放置在该框中，并按下*Ctrl*和*O*键，将*Ctrl* + *O*分配为 Open 菜单项的快捷键。

1.  在`actionSave`菜单项的 Shortcut 选项卡的空白处双击，并在打开的对话框的 Shortcut 框中按下*Ctrl* + *S*。

1.  在为两个菜单项 Open 和 Save 分配快捷键后。操作编辑器窗口将显示如下截图所示：

![](img/5e2020b6-9dfd-4a42-a2b4-304bcd498027.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件。在应用`pyuic5`命令后，XML 文件将被转换为 Python 代码。生成的 Python 脚本`demoFileDialog.py`可以在本书的源代码包中找到。`demoFileDialog.py`脚本将用作头文件，并将在另一个 Python 脚本文件中导入，该文件将调用此用户界面设计、“文件”菜单及其相应的菜单项。

1.  创建另一个名为`callFileDialog.pyw`的 Python 文件，并将`demoFileDialog.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QFileDialog
from demoFileDialog import *
class MyForm(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actionOpen.triggered.connect(self.openFileDialog)
        self.ui.actionSave.triggered.connect(self.saveFileDialog)
        self.show()
    def openFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 
        '/home')
        if fname[0]:
            f = open(fname[0], 'r')
        with f:
            data = f.read()
            self.ui.textEdit.setText(data)
    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,
        "QFileDialog.
        getSaveFileName()","","All Files (*);;Text Files (*.txt)",   
        options=options)
        f = open(fileName,'w')
        text = self.ui.textEdit.toPlainText()
        f.write(text)
        f.close()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

在`callFileDialog.pyw`文件中，您可以看到具有`objectName`、`actionOpen`的“打开”菜单项的 click()事件连接到`openFileDialog`方法；当用户单击“打开”菜单项时，将调用`openFileDialog`方法。类似地，“保存”菜单项的 click()事件与`objectName`、`actionSave`连接到`saveFileDialog`方法；当用户单击“保存”菜单项时，将调用`saveFileDialog`方法。

在`openFileDialog`方法中，通过调用`QFileDialog`类的`getOpenFileName`方法打开文件对话框。打开文件对话框使用户能够浏览目录并选择要打开的文件。选择文件后，当用户单击“确定”按钮时，所选文件名将被分配给`fname`变量。文件以只读模式打开，并且文件内容被读取并分配给文本编辑小部件；也就是说，文件内容显示在文本编辑小部件中。

在文本编辑小部件中显示的文件内容进行更改后，当用户从文件对话框中单击“保存”菜单项时，将调用`saveFileDialog()`方法。

在`saveFileDialog()`方法中，调用`QFileDialog`类上的`getSaveFileName()`方法，将打开文件保存对话框。您可以在相同位置使用相同名称保存文件，或者使用其他名称。如果在相同位置提供相同的文件名，则单击“确定”按钮后，将会出现一个对话框，询问您是否要用更新的内容覆盖原始文件。提供文件名后，该文件将以写入模式打开，并且文本编辑小部件中的内容将被读取并写入文件。也就是说，文本编辑小部件中可用的更新文件内容将被写入提供的文件名。

运行应用程序后，您会发现一个带有两个菜单项“打开”和“保存”的文件菜单，如下面的屏幕截图所示。您还可以看到“打开”和“保存”菜单项的快捷键：

![](img/e8b9094f-1160-4842-b63f-ed080a134dd5.png)

单击文件菜单中的“打开”菜单项，或按下快捷键*Ctrl* + *O*，您将获得打开文件对话框，如下面的屏幕截图所示。您可以浏览所需的目录并选择要打开的文件。选择文件后，您需要从对话框中单击“打开”按钮：

![](img/147ba560-0280-49a9-b9fb-8c6a6c6d98ef.png)

所选文件的内容将显示在文本编辑框中，如下面的屏幕截图所示：

![](img/3155d9b5-2c8f-4450-b4a5-b16fb00e9a18.png)

在文本编辑框中显示的文件内容进行修改后，当用户从文件菜单中单击“保存”菜单项时，将调用`getSaveFileName`方法以显示保存文件对话框。让我们使用原始名称保存文件，然后单击“保存”按钮，如下面的屏幕截图所示：

![](img/7e2a8d8f-0369-42b2-b5f6-32582404514d.png)

因为文件将以相同的名称保存，您将收到一个对话框，询问是否要用新内容替换原始文件，如下面的屏幕截图所示。单击“是”以使用新内容更新文件：

![](img/f629f0a0-3d64-4e81-9aa3-13fefe6f9a13.png)
