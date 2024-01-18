# 理解布局

在本章中，我们将重点关注以下主题：

+   使用水平布局

+   使用垂直布局

+   使用网格布局

+   使用表单布局

# 理解布局

正如其名称所示，布局用于以所需格式排列小部件。在布局中排列某些小部件时，自动将某些尺寸和对齐约束应用于小部件。例如，增大窗口的尺寸时，布局中的小部件也会增大，以利用增加的空间。同样，减小窗口的尺寸时，布局中的小部件也会减小。以下问题出现了：布局如何知道小部件的推荐尺寸是多少？

基本上，每个小部件都有一个名为sizeHint的属性，其中包含小部件的推荐尺寸。当窗口调整大小并且布局大小也改变时，通过小部件的sizeHint属性，布局管理器知道小部件的尺寸要求。

为了在小部件上应用尺寸约束，可以使用以下两个属性：

+   最小尺寸：如果窗口大小减小，小部件仍然不会变得比最小尺寸属性中指定的尺寸更小。

+   最大尺寸：同样，如果窗口增大，小部件不会变得比最大尺寸属性中指定的尺寸更大。

当设置了前述属性时，sizeHint属性中指定的值将被覆盖。

要在布局中排列小部件，只需选择所有小部件，然后单击工具栏上的“布局管理器”。另一种方法是右键单击以打开上下文菜单。从上下文菜单中，可以选择“布局”菜单选项，然后从弹出的子菜单中选择所需的布局。

在选择所需的布局后，小部件将以所选布局布置，并且在运行时不可见的小部件周围会有一条红线表示布局。要查看小部件是否正确布置，可以通过选择“表单”、“预览”或*Ctrl* + *R*来预览表单。要打破布局，选择“表单”、“打破布局”，输入*Ctrl* + *O*，或从工具栏中选择“打破布局”图标。

布局可以嵌套。

以下是Qt Designer提供的布局管理器：

+   水平布局

+   垂直布局

+   网格布局

+   表单布局

# 间隔器

为了控制小部件之间的间距，使用水平和垂直间隔器。当两个小部件之间放置水平间隔器时，两个小部件将被推到尽可能远的左右两侧。如果窗口大小增加，小部件的尺寸不会改变，额外的空间将被间隔器占用。同样，当窗口大小减小时，间隔器会自动减小，但小部件的尺寸不会改变。

间隔器会扩展以填充空白空间，并在空间减小时收缩。

让我们看看在水平框布局中排列小部件的步骤。

# 使用水平布局

水平布局将小部件在一行中排列，即使用水平布局水平对齐小部件。让我们通过制作一个应用程序来理解这个概念。

# 如何做...

在这个应用程序中，我们将提示用户输入电子邮件地址和密码。这个配方的主要重点是理解如何水平对齐两对标签和行编辑小部件。以下是创建此应用程序的逐步过程：

1.  让我们创建一个基于没有按钮的对话框模板的应用程序，并通过将两个标签、两个行编辑和一个按钮小部件拖放到表单上，来添加两个`QLabel`、两个`QLineEdit`和一个`QPushButton`小部件。

1.  将两个标签小部件的文本属性设置为`姓名`和`电子邮件地址`。

1.  还要将按钮小部件的文本属性设置为`提交`。

1.  由于此应用程序的目的是了解布局而不是其他任何内容，因此我们不会设置应用程序中任何小部件的objectName属性。

现在表单将显示如下截图所示：

![](assets/a58fadae-ef2c-4415-b9f1-a50a23a0f840.png)

1.  我们将在每对Label和LineEdit小部件上应用水平布局。因此，单击文本为`Name`的Label小部件，并保持按住*Ctrl*键，然后单击其旁边的LineEdit小部件。

您可以使用*Ctrl* +左键选择多个小部件。

1.  选择Label和LineEdit小部件后，右键单击并从打开的上下文菜单中选择布局菜单选项。

1.  选择布局菜单选项后，屏幕上将出现几个子菜单选项；选择水平布局子菜单选项。两个Label和LineEdit小部件将水平对齐，如下截图所示：

![](assets/bdc1cac6-e00f-4061-8483-08bdba5442ce.png)

1.  如果您想要打破布局怎么办？这很简单：您可以随时通过选择布局并右键单击来打破任何布局。上下文菜单将弹出；从上下文菜单中选择布局菜单选项，然后选择打破布局子菜单选项。

1.  要水平对齐文本为`Email Address`的第二对Label小部件和其旁边的LineEdit小部件，请重复步骤6和7中提到的相同过程。这对Label和LineEdit小部件也将水平对齐，如下截图所示。

您可以看到一个红色的矩形围绕着这两个小部件。这个红色的矩形是水平布局窗口：

![](assets/4539ec46-5518-4979-b6a9-5fd9edd6ecf3.png)

1.  要在第一对Label和LineEdit小部件之间创建一些空间，请从小部件框的间隔器选项卡中拖动水平间隔器小部件，并将其放置在文本为`Name`的Label小部件和其旁边的LineEdit小部件之间。

水平间隔器小部件最初占据两个小部件之间的默认空间。间隔器显示为表单上的蓝色弹簧。

1.  通过拖动其节点来调整水平间隔器的大小，以限制LineEdit小部件的宽度，如下截图所示：

![](assets/76bf6cf9-c68b-43bd-8db9-9931a47f0906.png)

1.  从第一对Label和LineEdit小部件的水平布局小部件的红色矩形中选择，并将其向右拖动，使其宽度等于第二对小部件。

1.  拖动水平布局小部件时，水平间隔器将增加其宽度，以消耗两个小部件之间的额外空白空间，如下截图所示：

![](assets/8404860e-69fe-4ea9-a15a-d36755fb2ef8.png)

1.  将应用程序保存为`demoHorizontalLayout.ui`。

使用Qt Designer创建的用户界面存储在`.ui`文件中，这是一个XML文件，我们需要将其转换为Python代码。要进行转换，您需要打开命令提示符窗口并导航到保存文件的文件夹，然后发出以下命令行：

```py
C:\Pythonbook\PyQt5>pyuic5 demoHorizontalLayout.ui -o demoHorizontalLayout.py
```

Python脚本文件`demoHorizontalLayout.py`可能包含以下代码：

```py
from PyQt5 import QtCore, QtGui, QtWidgets
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(483, 243)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(120, 130, 111, 
        23))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(20, 30, 271, 27))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.
        QSizePolicy.Expanding,QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.widget1 = QtWidgets.QWidget(Dialog)
        self.widget1.setGeometry(QtCore.QRect(20, 80, 276, 27))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.
        widget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.widget1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_2.addWidget(self.lineEdit_2)
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "Submit"))
        self.label.setText(_translate("Dialog", "Name"))
        self.label_2.setText(_translate("Dialog", "Email Address"))
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

您可以在代码中看到，一个具有默认objectName属性`lineEdit`的LineEdit小部件和一个具有默认objectName属性为**label**的Label小部件被放置在表单上。使用水平布局小部件水平对齐LineEdit和Label小部件。水平布局小部件具有默认的objectName属性`horizontalLayout`。在对齐Label和LineEdit小部件时，两个小部件之间的水平空间被减小。因此，在Label和LineEdit小部件之间保留了一个间隔。第二对Label具有默认的objectName属性`label_2`和LineEdit小部件具有默认的objectName属性`lineEdit_2`，通过具有默认objectName属性`horizontalLayout_2`的水平布局水平对齐。

运行应用程序后，您会发现两对标签和行编辑小部件水平对齐，如下面的屏幕截图所示：

![](assets/0e8766db-b9a9-4dde-ab8e-fb23c08ebe42.png)

# 使用垂直布局

垂直布局将选定的小部件垂直排列，以列的形式一个接一个地排列。在下面的应用程序中，您将学习如何在垂直布局中放置小部件。

# 如何做...

在这个应用程序中，我们将提示用户输入姓名和电子邮件地址。用于输入姓名和电子邮件地址的标签和文本框，以及提交按钮，将通过垂直布局垂直排列。以下是创建应用程序的步骤：

1.  启动Qt Designer并基于无按钮对话框模板创建一个应用程序，然后通过将两个标签、两个行编辑和一个 `QPushButton` 小部件拖放到表单上，向表单添加两个`QLabel`、两个`QlineEdit`和一个 `QPushButton` 小部件。

1.  将两个标签小部件的文本属性设置为`Name`和`Email Address`。

1.  将提交按钮的文本属性设置为`Submit`。因为这个应用程序的目的是理解布局，而不是其他任何东西，所以我们不会设置应用程序中任何小部件的objectName属性。表单现在将显示如下屏幕截图所示：

![](assets/6eb17343-a136-4675-af5d-5c062e558a71.png)

1.  在对小部件应用垂直布局之前，我们需要将小部件水平对齐。因此，我们将在每对标签和行编辑小部件上应用水平布局小部件。因此，点击文本为`Name`的标签小部件，并保持*Ctrl*键按下，然后点击其旁边的行编辑小部件。

1.  在选择标签和行编辑小部件后，右键单击鼠标按钮，并从打开的上下文菜单中选择布局菜单选项。

1.  选择布局菜单选项后，屏幕上会出现几个子菜单选项。选择水平布局子菜单选项。标签和行编辑小部件将水平对齐。

1.  要水平对齐文本为`Email Address`的第二对标签和其旁边的行编辑小部件，请重复前面步骤5和6中提到的相同过程。您会看到一个红色矩形围绕着这两个小部件。这个红色矩形是水平布局窗口。

1.  要在第一对标签和行编辑小部件之间创建一些空间，请从小部件框的间隔器选项卡中拖动水平间隔器小部件，并将其放在文本为`Name`的标签小部件和其旁边的行编辑小部件之间。水平间隔器将最初占据两个小部件之间的默认空间。

1.  从第一对标签和行编辑小部件中选择 Horizontal Layout 小部件的红色矩形，并将其向右拖动，使其宽度等于第二对的宽度。

1.  拖动水平布局小部件时，水平间隔器将增加其宽度，以消耗两个小部件之间的额外空白空间，如下面的屏幕截图所示：

![](assets/6a7bfcbd-9828-44f9-a489-35a95e46eb8d.png)

1.  现在，选择三个项目：第一个水平布局窗口、第二个水平布局窗口和提交按钮。在这些多重选择过程中保持*Ctrl*键按下。

1.  选择这三个项目后，右键单击以打开上下文菜单。

1.  从上下文菜单中选择布局菜单选项，然后选择垂直布局子菜单选项。这三个项目将垂直对齐，并且提交按钮的宽度将增加以匹配最宽布局的宽度，如下面的屏幕截图所示：

![](assets/2331a0a3-0d14-4fb3-bc0f-647a0926d26d.png)

1.  您还可以从工具栏中选择垂直布局图标，以将小部件排列成垂直布局。

1.  如果要控制提交按钮的宽度，可以使用此小部件的minimumSize和maximumSize属性。您会注意到两个水平布局之间的垂直空间大大减少了。

1.  要在两个水平布局之间创建一些空间，请从小部件框的间隔器选项卡中拖动垂直间隔器小部件，并将其放置在两个水平布局之间。

垂直间隔器最初将占据两个水平布局之间的默认空间

1.  要在第二个水平布局和提交按钮之间创建垂直空间，请拖动垂直间隔器，并将其放置在第二个水平布局和提交按钮之间。

1.  选择垂直布局的红色矩形，并向下拖动以增加其高度。

1.  拖动垂直布局小部件时，垂直间隔器将增加其高度，以消耗两个水平布局和提交按钮之间的额外空白空间，如下面的屏幕截图所示：

![](assets/205ffd7b-a853-4a5e-a6b2-ee5ecd78cc3a.png)

1.  将应用程序保存为`demoverticalLayout.ui`。

由于我们知道使用Qt Designer创建的用户界面存储在`.ui`文件中，这是一个XML文件，需要将其转换为Python代码。要进行转换，您需要打开命令提示符窗口，并导航到保存文件的文件夹，然后发出以下命令：

```py
C:PyQt5>pyuic5 demoverticalLayout.ui -o demoverticalLayout.py
```

Python脚本文件`demoverticalLayout.py`可能包含以下代码：

```py
from PyQt5 import QtCore, QtGui, QtWidgets
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(407, 211)
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(20, 30, 278, 161))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.
        QSizePolicy.Expanding,QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.verticalLayout.addLayout(self.horizontalLayout)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.
        QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_2.addWidget(self.lineEdit_2)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.
        QSizePolicy.Minimum,QtWidgets.QSizePolicy.
        Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.pushButton = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Name"))
        self.label_2.setText(_translate("Dialog", "Email Address"))
        self.pushButton.setText(_translate("Dialog", "Submit"))
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

您可以在代码中看到，具有默认objectName `lineEdit`属性的Line Edit小部件和具有默认objectName `label`属性的Label小部件被放置在表单上，并使用具有默认objectName属性`horizontalLayout`的水平布局进行水平对齐。在对齐标签和行编辑小部件时，两个小部件之间的水平空间减小了。因此，在标签和行编辑小部件之间保留了一个间隔器。第二对，具有默认objectName `label_2`属性的Label小部件和具有默认objectName `lineEdit_2`属性的Line Edit小部件，使用具有默认objectName `horizontalLayout_2`属性的水平布局进行水平对齐。然后，使用具有默认`objectName`属性`verticalLayout`的垂直布局对前两个水平布局和具有默认objectName `pushButton`属性的提交按钮进行垂直对齐。通过在它们之间放置一个水平间隔器，增加了第一对标签和行编辑小部件之间的水平空间。类似地，通过在它们之间放置一个名为`spacerItem1`的垂直间隔器，增加了两个水平布局之间的垂直空间。此外，还在第二个水平布局和提交按钮之间放置了一个名为`spacerItem2`的垂直间隔器，以增加它们之间的垂直空间。

运行应用程序后，您会发现两对标签和行编辑小部件以及提交按钮垂直对齐，如下面的屏幕截图所示：

![](assets/dd0b7f67-3b15-4336-b924-46487100a488.png)

# 使用网格布局

网格布局将小部件排列在可伸缩的网格中。要了解网格布局小部件如何排列小部件，让我们创建一个应用程序。

# 如何做...

在这个应用程序中，我们将制作一个简单的登录表单，提示用户输入电子邮件地址和密码，然后点击提交按钮。在提交按钮下方，将有两个按钮，取消和忘记密码。该应用程序将帮助您了解这些小部件如何以网格模式排列。以下是创建此应用程序的步骤：

1.  启动Qt Designer，并基于无按钮的对话框模板创建一个应用程序，然后通过拖放两个Label、两个Line Edit和三个Push Button小部件到表单上，将两个`QLabel`、两个`QlineEdit`和三个`QPushButton`小部件添加到表单上。

1.  将两个Label小部件的文本属性设置为`Name`和`Email Address`。

1.  将三个Push Button小部件的文本属性设置为`Submit`，`Cancel`和`Forgot Password`。

1.  因为此应用程序的目的是了解布局而不是其他任何内容，所以我们不会设置应用程序中任何小部件的objectName属性。

1.  为了增加两个Line Edit小部件之间的垂直空间，从Widget Box的间隔符选项卡中拖动垂直间隔符小部件，并将其放置在两个Line Edit小部件之间。垂直间隔符将最初占据两个Line Edit小部件之间的空白空间。

1.  为了在第二个Line Edit小部件和提交按钮之间创建垂直空间，拖动垂直间隔符小部件并将其放置在它们之间。

应用程序将显示如下截图所示：

![](assets/756be524-e34c-4be9-a03c-da094cecfe9a.png)

1.  通过按下*Ctrl*键并单击表单上的所有小部件来选择表单上的所有小部件。

1.  选择所有小部件后，右键单击鼠标按钮以打开上下文菜单。

1.  从上下文菜单中，选择布局菜单选项，然后选择网格布局子菜单选项。

小部件将按照网格中所示的方式对齐：

![](assets/cf4b1ac4-bbd7-4c78-b5c9-e7af16062762.png)

1.  为了增加提交和取消按钮之间的垂直空间，从Widget Box的间隔符选项卡中拖动垂直间隔符小部件，并将其放置在它们之间。

1.  为了增加取消和忘记密码按钮之间的水平空间，从间隔符选项卡中拖动水平间隔符小部件，并将其放置在它们之间。

现在表格将显示如下截图所示：

![](assets/742b1811-bfea-49f2-a3b3-5aa7e38033f8.png)

1.  将应用程序保存为`demoGridLayout.ui`。

使用Qt Designer创建的用户界面存储在`.ui`文件中，这是一个XML文件，需要转换为Python代码。要进行转换，您需要打开命令提示符窗口并导航到保存文件的文件夹，然后发出以下命令：

```py
C:PyQt5>pyuic5 demoGridLayout.ui -o demoGridLayout.py
```

Python脚本文件`demoGridLayout.py`可能包含以下代码：

```py
from PyQt5 import QtCore, QtGui, QtWidgets
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(369, 279)
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(20, 31, 276, 216))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 4, 0, 1, 5)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.
        QSizePolicy.Minimum,QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 5, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 2)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 2, 2, 1, 3)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 2, 1, 3)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.
        QSizePolicy.Minimum,QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 3, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.
        QSizePolicy.Minimum,QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem2, 1, 2, 1, 3)
        self.pushButton_2 = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 6, 0, 1, 3)
        self.pushButton_3 = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 6, 4, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.
        QSizePolicy.Expanding,QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem3, 6, 3, 1, 1)
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "Submit"))
        self.label.setText(_translate("Dialog", "Name"))
        self.label_2.setText(_translate("Dialog", "Email Address"))
        self.pushButton_2.setText(_translate("Dialog", "Cancel"))
        self.pushButton_3.setText(_translate("Dialog", 
        "Forgot Password"))
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
```

# 工作原理...

您可以在代码中看到，具有默认objectName`lineEdit`属性的Line Edit小部件和具有默认objectName`label`属性的Label小部件放置在表单上。类似地，第二对具有默认objectName`label_2`属性的Label小部件和具有默认objectName`lineEdit_2`属性的Line Edit小部件也放置在表单上。通过在它们之间放置名为`spacerItem1`的垂直间隔符，增加了两对Label和Line Edit小部件之间的垂直空间。还在表单上放置了一个文本为`Submit`，objectName为`pushButton`的Push Button小部件。同样，通过在具有objectName`label_2`的第二个Label和具有objectName`pushButton`的Push Button小部件之间放置名为`spacerItem2`的垂直间隔符，增加了它们之间的垂直空间。另外两个具有默认objectName属性`pushButton_2`和`pushButton_3`的push按钮也放置在表单上。所有小部件都以默认对象名称`gridLayout`排列在一个可伸缩的网格布局中。具有object名称`pushButton`和`pushButton_2`的两个push按钮之间的垂直空间通过在它们之间放置名为`spacerItem3`的垂直间隔符来增加。

运行应用程序时，您会发现两对Label和Line Edit小部件以及提交、取消和忘记密码按钮都排列在一个可伸缩的网格中，如下截图所示：

![](assets/d581b1d5-be49-44a1-8060-c92f029cd11a.png)

# 使用表单布局

表单布局被认为是几乎所有应用程序中最需要的布局。当显示产品、服务等以及接受用户或客户的反馈或其他信息时，需要这种两列布局。

# 做好准备

表单布局以两列格式排列小部件。就像任何网站的注册表单或任何订单表单一样，表单被分成两列，左侧列显示标签或文本，右侧列显示空文本框。同样，表单布局将小部件排列在左列和右列。让我们使用一个应用程序来理解表单布局的概念。

# 如何做...

在这个应用程序中，我们将创建两列，一列用于显示消息，另一列用于接受用户输入。除了两对用于从用户那里获取输入的Label和Line Edit小部件之外，该应用程序还将有两个按钮，这些按钮也将按照表单布局排列。以下是创建使用表单布局排列小部件的应用程序的步骤：

1.  启动Qt Designer，并基于无按钮的对话框模板创建一个应用程序，然后通过拖放两个Label、两个LineEdit和两个PushButton小部件到表单上，添加两个`QLabel`、两个`QLineEdit`和两个`QPushButton`小部件。

1.  将两个Label小部件的文本属性设置为`Name`和`Email Address`。

1.  将两个Push Button小部件的文本属性设置为`Cancel`和`Submit`。

1.  因为这个应用程序的目的是理解布局，而不是其他任何东西，所以我们不会设置应用程序中任何小部件的objectName属性。

应用程序将显示如下屏幕截图所示：

![](assets/dc2d8dc1-37b5-43f8-a14a-cd377b5f520a.png)

1.  通过按下*Ctrl*键并单击表单上的所有小部件来选择所有小部件。

1.  选择所有小部件后，右键单击鼠标按钮以打开上下文菜单。

1.  从上下文菜单中，选择布局菜单选项，然后选择表单布局子菜单选项中的布局。

小部件将在表单布局小部件中对齐，如下面的屏幕截图所示：

![](assets/a757b71e-e5ae-4091-a2ac-7a538975ed84.png)

1.  为了增加两个Line Edit小部件之间的垂直空间，请从Widget Box的间隔器选项卡中拖动垂直间隔器小部件，并将其放置在它们之间。

1.  为了增加第二个Line Edit小部件和提交按钮之间的垂直空间，请从间隔器选项卡中拖动垂直间隔器小部件，并将其放置在它们之间。

1.  选择表单布局小部件的红色矩形，并垂直拖动以增加其高度。两个垂直间隔器将自动增加高度，以利用小部件之间的空白空间。

表单现在将显示如下屏幕截图所示：

![](assets/7558844e-df8a-4249-9389-bfe217334b5f.png)

1.  将应用程序保存为`demoFormLayout.ui`。

使用Qt Designer创建的用户界面存储在`.ui`文件中，这是一个XML文件，需要转换为Python代码。要进行转换，您需要打开命令提示符窗口，并导航到保存文件的文件夹，然后发出以下命令：

```py
C:PyQt5>pyuic5 demoFormLayout.ui -o demoFormLayout.py
```

Python脚本文件`demoFormLayout.py`可能包含以下代码：

```py
from PyQt5 import QtCore, QtGui, QtWidgets
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(407, 211)
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(20, 30, 276, 141))
        self.widget.setObjectName("widget")
        self.formLayout = QtWidgets.QFormLayout(self.widget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.
        LabelRole,self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.
        FieldRole,self.lineEdit)
        self.label_2 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.
        LabelRole,self.label_2)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.
        FieldRole, self.lineEdit_2)
        self.pushButton_2 = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.
        LabelRole,self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.
        FieldRole,self.pushButton)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.
        QSizePolicy.Minimum,QtWidgets.QSizePolicy.Expanding)
        self.formLayout.setItem(1, QtWidgets.QFormLayout.FieldRole, 
        spacerItem)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.
        QSizePolicy.Minimum,QtWidgets.QSizePolicy.Expanding)
        self.formLayout.setItem(3, QtWidgets.QFormLayout.FieldRole, 
        spacerItem1)
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Name"))
        self.label_2.setText(_translate("Dialog", "Email Address"))
        self.pushButton_2.setText(_translate("Dialog", "Cancel"))
        self.pushButton.setText(_translate("Dialog", "Submit"))
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

您可以在代码中看到，一个带有默认objectName `lineEdit`属性的Line Edit小部件和一个带有默认objectName `labels`属性的Label小部件被放置在表单上。同样，第二对，一个带有默认objectName `label_2`属性的Label小部件和一个带有默认objectName `lineEdit_2`属性的Line Edit小部件被放置在表单上。两个带有object names `pushButton`和`pushButton_2`的按钮被放置在表单上。所有六个小部件都被选中，并使用默认objectName `formLayout`属性的表单布局小部件以两列格式对齐。

运行应用程序时，您会发现两对Label和Line Edit小部件以及取消和提交按钮被排列在表单布局小部件中，如下面的屏幕截图所示：

![](assets/d7c6b285-ad2e-4171-8937-c2346b4f9166.png)
