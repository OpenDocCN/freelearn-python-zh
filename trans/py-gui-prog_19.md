# 数据库处理

数据库处理在任何应用程序中都起着重要作用，因为数据需要存储以备将来使用。您需要存储客户信息、用户信息、产品信息、订单信息等。在本章中，您将学习与数据库处理相关的每项任务：

+   创建数据库

+   创建数据库表

+   在指定的数据库表中插入行

+   显示指定数据库表中的行

+   在指定的数据库表中导航行

+   在数据库表中搜索特定信息

+   创建登录表单-应用认证程序

+   更新数据库表-更改用户密码

+   从数据库表中删除一行

我们将使用 SQLite 进行数据库处理。在我们进入本章的更深入之前，让我们快速介绍一下 SQLite。

# 介绍

SQLite 是一个非常易于使用的数据库引擎。基本上，它是一个轻量级数据库，适用于存储在单个磁盘文件中的小型应用程序。它是一个非常受欢迎的数据库，用于手机、平板电脑、小型设备和仪器。SQLite 不需要单独的服务器进程，甚至不需要任何配置。

为了使这个数据库在 Python 脚本中更容易使用，Python 标准库包括一个名为`sqlite3`的模块。因此，要在任何 Python 应用程序中使用 SQLite，您需要使用`import`语句导入`sqlite3`模块，如下所示：

```py
import sqlite3
```

使用任何数据库的第一步是创建一个`connect`对象，通过它您需要与数据库建立连接。以下示例建立到`ECommerce`数据库的连接：

```py
conn = sqlite3.connect('ECommerce.db')
```

如果数据库已经存在，此示例将建立到`ECommerce`数据库的连接。如果数据库不存在，则首先创建数据库，然后建立连接。

您还可以使用`connect`方法中的`:memory:`参数在内存中创建临时数据库。

```py
conn = sqlite3.connect(':memory:')
```

您还可以使用`:memory:`特殊名称在 RAM 中创建数据库。

一旦与数据库相关的工作结束，您需要使用以下语句关闭连接：

```py
conn.close()
```

# 创建游标对象

要使用数据库表，您需要获取一个`cursor`对象，并将 SQL 语句传递给`cursor`对象以执行它们。以下语句创建一个名为`cur`的`cursor`对象：

```py
cur = conn.cursor()
```

使用`cursor`对象`cur`，您可以执行 SQL 语句。例如，以下一组语句创建一个包含三列`id`、`EmailAddress`和`Password`的`Users`表：

```py
# Get a cursor object
cur = conn.cursor() cur.execute('''CREATE TABLE Users(id INTEGER PRIMARY KEY, EmailAddress TEXT, Password TEXT)''') conn.commit()
```

请记住，您需要通过在连接对象上调用`commit()`方法来提交对数据库的更改，否则对数据库所做的所有更改都将丢失。

以下一组语句将删除`Users`表：

```py
# Get a cursor object
cur = conn.cursor() cur.execute('''DROP TABLE Users''') conn.commit()
```

# 创建数据库

在这个示例中，我们将提示用户输入数据库名称，然后点击按钮。点击按钮后，如果指定的数据库不存在，则创建它，如果已经存在，则连接它。

# 如何做…

按照逐步过程在 SQLite 中创建数据库：

1.  让我们基于没有按钮的对话框模板创建一个应用程序。

1.  通过拖放两个标签小部件、一个行编辑小部件和一个按钮小部件到表单上，添加两个`QLabel`小部件、一个`QLineEdit`小部件和一个`QPushButton`小部件。

1.  将第一个标签小部件的文本属性设置为`输入数据库名称`。

1.  删除第二个标签小部件的文本属性，因为这是已经建立的。

1.  将行编辑小部件的对象名称属性设置为`lineEditDBName`。

1.  将按钮小部件的对象名称属性设置为`pushButtonCreateDB`。

1.  将第二个标签小部件的对象名称属性设置为`labelResponse`。

1.  将应用程序保存为`demoDatabase.ui`。表单现在将显示如下截图所示：

![](img/14fd5deb-fdad-4905-8ce4-b20927514f75.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。通过应用`pyuic5`实用程序，将 XML 文件转换为 Python 代码。生成的 Python 脚本`demoDatabase.py`可以在本书的源代码包中看到。

1.  将`demoDatabase.py`脚本视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callDatabase.pyw`的 Python 文件，并将`demoDatabase.py`代码导入其中：

```py
import sqlite3, sys
from PyQt5.QtWidgets import QDialog, QApplication
from sqlite3 import Error
from demoDatabase import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonCreateDB.clicked.connect(self.
        createDatabase)
        self.show()
    def createDatabase(self):
        try:
            conn = sqlite3.connect(self.ui.lineEditDBName.
            text()+".db")
            self.ui.labelResponse.setText("Database is created")
        except Error as e:
            self.ui.labelResponse.setText("Some error has 
            occurred")
        finally:
            conn.close()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

您可以在脚本中看到，具有 objectName 属性`pushButtonCreateDB`的按钮的 click()事件与`createDatabase()`方法连接在一起。这意味着每当单击按钮时，就会调用`createDatabase()`方法。在`createDatabase()`方法中，调用了`sqlite3`类的`connect()`方法，并将用户在 Line Edit 小部件中输入的数据库名称传递给`connect()`方法。如果在创建数据库时没有发生错误，则通过 Label 小部件显示消息“数据库已创建”以通知用户；否则，通过 Label 小部件显示消息“发生了一些错误”以指示发生错误。

运行应用程序时，将提示您输入数据库名称。假设我们输入数据库名称为`Ecommerce`。单击“创建数据库”按钮后，将创建数据库并收到消息“数据库已创建”：

![](img/6f1378e9-e4d2-44b9-ae9b-4d586e8ebc92.png)

# 创建数据库表

在这个示例中，我们将学习如何创建一个数据库表。用户将被提示指定数据库名称，然后是要创建的表名称。该示例使您能够输入列名及其数据类型。单击按钮后，将在指定的数据库中创建具有定义列的表。

# 如何做...

以下是创建一个 GUI 的步骤，使用户能够输入有关要创建的数据库表的所有信息。使用此 GUI，用户可以指定数据库名称、列名，并且还可以选择列类型：

1.  让我们基于没有按钮的对话框模板创建一个应用程序。

1.  通过拖放五个 Label、三个 Line Edit、一个 Combo Box 和两个 Push Button 小部件到表单上，添加五个 QLabel、三个 QLineEdit、一个 QComboBox 和两个 QPushButton 小部件。

1.  将前四个 Label 小部件的文本属性设置为`输入数据库名称`，`输入表名称`，`列名`和`数据类型`。

1.  删除第五个 Label 小部件的文本属性，因为这是通过代码建立的。

1.  将两个 push 按钮的文本属性设置为`添加列`和`创建表`。

1.  将三个 Line Edit 小部件的 objectName 属性设置为`lineEditDBName`、`lineEditTableName`和`lineEditColumnName`。

1.  将 Combo Box 小部件的 objectName 属性设置为`ComboBoxDataType`。

1.  将两个 push 按钮的 objectName 属性设置为`pushButtonAddColumn`和`pushButtonCreateTable`。

1.  将第五个 Label 小部件的 objectName 属性设置为`labelResponse`。

1.  将应用程序保存为`demoCreateTable.ui`。表单现在将显示如下截图所示：

![](img/25a283f1-b10b-41da-a13d-ba1d67c4ae55.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。使用`pyuic5`命令将 XML 文件转换为 Python 代码。本书的源代码包中可以看到生成的 Python 脚本`demoCreateTable.py`。

1.  将`demoCreateTable.py`脚本视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callCreateTable.pyw`的 Python 文件，并将`demoCreateTable.py`代码导入其中：

```py
import sqlite3, sys
from PyQt5.QtWidgets import QDialog, QApplication
from sqlite3 import Error
from demoCreateTable import *
tabledefinition=""
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonCreateTable.clicked.connect( 
        self.createTable)
        self.ui.pushButtonAddColumn.clicked.connect(self.
        addColumns)
        self.show()
    def addColumns(self):
        global tabledefinition
        if tabledefinition=="":
            tabledefinition="CREATE TABLE IF NOT EXISTS "+   
            self.ui.lineEditTableName.text()+" ("+ 
            self.ui.lineEditColumnName.text()+"  "+                                                                                                                      
            self.ui.comboBoxDataType.itemText(self.ui.
            comboBoxDataType.currentIndex())
        else:
            tabledefinition+=","+self.ui.lineEditColumnName
            .text()+" "+ self.ui.comboBoxDataType.itemText
            (self.ui.comboBoxDataType.currentIndex())
            self.ui.lineEditColumnName.setText("")
            self.ui.lineEditColumnName.setFocus()
    def createTable(self):
        global tabledefinition
        try:
            conn = sqlite3.connect(self.ui.lineEditDBName.
            text()+".db")
            self.ui.labelResponse.setText("Database is    
            connected")
            c = conn.cursor()
            tabledefinition+=");"
            c.execute(tabledefinition)
            self.ui.labelResponse.setText("Table is successfully  
            created")
        except Error as e:
            self.ui.labelResponse.setText("Error in creating 
            table")
        finally:
            conn.close()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理…

在脚本中可以看到，具有 objectName 属性`pushButtonCreateTable`的按钮的 click()事件与`createTable()`方法相连。这意味着每当单击此按钮时，将调用`createTable()`方法。类似地，具有 objectName 属性`pushButtonAddColumn`的按钮的 click()事件与`addColumns()`方法相连。也就是说，单击此按钮将调用`addColumns()`方法。

在`addColumns()`方法中，定义了`CREATE TABLE SQL`语句，其中包括在 LineEdit 小部件中输入的列名和从组合框中选择的数据类型。用户可以向表中添加任意数量的列。

在`createTable()`方法中，首先建立与数据库的连接，然后执行`addColumns()`方法中定义的`CREATE TABLE SQL`语句。如果成功创建表，将通过最后一个 Label 小部件显示一条消息，通知您表已成功创建。最后，关闭与数据库的连接。

运行应用程序时，将提示您输入要创建的数据库名称和表名称，然后输入该表中所需的列。假设您要在`ECommerce`表中创建一个`Users`表，其中包括`EmailAddress`和`Password`两列，这两列都假定为文本类型。

`Users`表中的第一列名为`Email Address`，如下面的屏幕截图所示：

![](img/e6805d8b-ee5e-45dd-9d1b-f32c6aa86b7b.png)

让我们在`Users`表中定义另一列，称为`Password`，类型为文本，然后点击 Create Table 按钮。如果成功创建了指定列数的表，将通过最后一个 Label 小部件显示消息“表已成功创建”，如下面的屏幕截图所示：

![](img/37ac3783-9136-42eb-a3d5-cefc0d2d003f.png)

为了验证表是否已创建，我将使用一种可视化工具，该工具可以让您创建、编辑和查看数据库表及其中的行。这个可视化工具是 SQLite 的 DB Browser，我从[`sqlitebrowser.org/`](http://sqlitebrowser.org/)下载了它。在启动 DB Browser for SQLite 后，点击主菜单下方的“打开数据库”选项卡。浏览并选择当前文件夹中的`ECommerce`数据库。`ECommerce`数据库显示了一个包含两列`EmailAddress`和`Password`的`Users`表，如下面的屏幕截图所示，证实数据库表已成功创建：

![](img/e64bbee5-b2a2-4f18-b53f-ee14777e310e.png)

# 在指定的数据库表中插入行

在本教程中，我们将学习如何向表中插入行。我们假设一个名为`Users`的表已经存在于名为`ECommerce`的数据库中，包含两列`EmailAddress`和`Password`。

在分别输入电子邮件地址和密码后，当用户点击“插入行”按钮时，将会将行插入到指定的数据库表中。

# 操作步骤…

以下是向存在于 SQLite 中的数据库表中插入行的步骤：

1.  让我们创建一个基于无按钮对话框模板的应用程序。

1.  通过拖放五个 Label 小部件、四个 LineEdit 小部件和一个 PushButton 小部件将它们添加到表单中。

1.  将前四个 Label 小部件的文本属性设置为“输入数据库名称”、“输入表名称”、“电子邮件地址”和“密码”。

1.  删除第五个 Label 小部件的文本属性，这是通过代码建立的。

1.  将按钮的文本属性设置为“插入行”。

1.  将四个 Line Edit 小部件的 objectName 属性设置为`lineEditDBName`、`lineEditTableName`、`lineEditEmailAddress`和`lineEditPassword`。

1.  将 Push Button 小部件的 objectName 属性设置为`pushButtonInsertRow`。

1.  将第五个 Label 小部件的 objectName 属性设置为`labelResponse`。由于我们不希望密码显示出来，我们希望用户输入密码时显示星号。

1.  为此，选择用于输入密码的 Line Edit 小部件，并从 Property Editor 窗口中选择 echoMode 属性，并将其设置为 Password，而不是默认的 Normal，如下屏幕截图所示：

![](img/24010fe7-0bcd-490a-bdbf-82206bfbc3a2.png)

echoMode 属性显示以下四个选项：

+   +   Normal: 这是默认属性，当在 Line Edit 小部件中键入字符时显示。

+   NoEcho: 在 Line Edit 小部件中键入时不显示任何内容，也就是说，您甚至不会知道输入的文本长度。

+   Password: 主要用于密码。在 Line Edit 小部件中键入时显示星号。

+   PasswordEchoOnEdit: 在 Line Edit 小部件中键入密码时显示密码，尽管输入的内容会很快被星号替换。

1.  将应用程序保存为`demoInsertRowsInTable.ui`。表单现在将显示如下屏幕截图所示：

![](img/e1494e00-f106-4262-91bb-46543a2899ad.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。通过应用`pyuic5`实用程序，XML 文件将被转换为 Python 代码。生成的 Python 脚本`demoInsertRowsInTable.py`可以在本书的源代码包中找到。

1.  创建另一个名为`callInsertRows.pyw`的 Python 文件，并将`demoInsertRowsInTable.py`代码导入其中。Python 脚本`callInsertRows.pyw`中的代码如下所示：

```py
import sqlite3, sys
from PyQt5.QtWidgets import QDialog, QApplication
from sqlite3 import Error
from demoInsertRowsInTable import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonInsertRow.clicked.connect(self.
        InsertRows)
        self.show()
    def InsertRows(self):
        sqlStatement="INSERT INTO "+  
        self.ui.lineEditTableName.text() +"   
        VALUES('"+self.ui.lineEditEmailAddress.text()+"', 
        '"+self.ui.lineEditPassword.text()+"')"
        try:
            conn = sqlite3.connect(self.ui.lineEditDBName.
            text()+ ".db")
        with conn:
            cur = conn.cursor()
            cur.execute(sqlStatement)
            self.ui.labelResponse.setText("Row successfully 
            inserted")
        except Error as e:
            self.ui.labelResponse.setText("Error in inserting  
            row")
        finally:
            conn.close()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

您可以在脚本中看到，具有 objectName 属性`pushButtonInsertRow`的 push 按钮的单击事件连接到`InsertRows()`方法。这意味着每当单击此 push 按钮时，将调用`InsertRows()`方法。在`InsertRows()`方法中，定义了一个`INSERT SQL`语句，用于获取在 Line Edit 小部件中输入的电子邮件地址和密码。与输入数据库名称的 Line Edit 小部件建立连接。然后，执行`INSERT SQL`语句，将新行添加到指定的数据库表中。最后，关闭与数据库的连接。

运行应用程序时，将提示您指定数据库名称、表名称以及两个列`Email Address`和`Password`的数据。输入所需信息后，单击插入行按钮，将向表中添加新行，并显示消息“成功插入行”，如下屏幕截图所示：

![](img/69102eea-f463-4a74-9b7e-45251474e952.png)

为了验证行是否插入了`Users`表，我将使用一个名为 DB Browser for SQLite 的可视化工具。这是一个很棒的工具，可以让您创建、编辑和查看数据库表及其中的行。您可以从[`sqlitebrowser.org/`](http://sqlitebrowser.org/)下载 DB Browser for SQLite。启动 DB Browser for SQLite 后，您需要首先打开数据库。要这样做，请单击主菜单下方的打开数据库选项卡。浏览并选择当前文件夹中的`Ecommerce`数据库。`Ecommerce`数据库显示`Users`表。单击执行 SQL 按钮；您会得到一个小窗口来输入 SQL 语句。编写一个 SQL 语句，`select * from Users`，然后单击窗口上方的运行图标。

在`Users`表中输入的所有行将以表格格式显示，如下屏幕截图所示。确认我们在本教程中制作的应用程序运行良好：

![](img/1a25fa44-24c3-4eec-9392-3540580713b0.png)

# 在指定的数据库表中显示行

在这个示例中，我们将学习从给定数据库表中获取行并通过表小部件以表格格式显示它们。我们假设一个名为`Users`的表包含两列，`EmailAddress`和`Password`，已经存在于名为`ECommerce`的数据库中。此外，我们假设`Users`表中包含一些行。

# 如何做…

按照以下逐步过程访问 SQLite 数据库表中的行：

1.  让我们基于没有按钮的对话框模板创建一个应用程序。

1.  通过拖放三个标签小部件、两个行编辑小部件、一个按钮和一个表小部件到表单上，向表单添加三个`QLabel`小部件、两个`QLineEdit`小部件、一个`QPushButton`小部件和一个`QTableWidget`小部件。

1.  将两个标签小部件的文本属性设置为`输入数据库名称`和`输入表名称`。

1.  删除第三个标签小部件的文本属性，因为它的文本属性将通过代码设置。

1.  将按钮的文本属性设置为`显示行`。

1.  将两个行编辑小部件的 objectName 属性设置为`lineEditDBName`和`lineEditTableName`。

1.  将按钮小部件的 objectName 属性设置为`pushButtonDisplayRows`。

1.  将第三个标签小部件的 objectName 属性设置为`labelResponse`。

1.  将应用程序保存为`demoDisplayRowsOfTable.ui`。表单现在将显示如下截图所示：

![](img/087f96d5-d68f-46ed-b734-460c08612c8e.png)

将通过表小部件显示的`Users`表包含两列。

1.  选择表小部件，并在属性编辑器窗口中选择其 columnCount 属性。

1.  将 columnCount 属性设置为`2`，将 rowCount 属性设置为`3`，如下截图所示：

![](img/546d14f3-3cac-43df-81dd-b36542f6cc78.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。通过应用`pyuic5`实用程序，XML 文件将被转换为 Python 代码。生成的 Python 脚本`demoInsertRowsInTable.py`可以在本书的源代码包中找到。

1.  将`demoInsertRowsInTable.py`脚本视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callDisplayRows.pyw`的 Python 文件，并将`demoDisplayRowsOfTable.py`代码导入其中：

```py
import sqlite3, sys
from PyQt5.QtWidgets import QDialog, QApplication,QTableWidgetItem
from sqlite3 import Error
from demoDisplayRowsOfTable import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonDisplayRows.clicked. 
            connect(self.DisplayRows)
        self.show()

    def DisplayRows(self):
        sqlStatement="SELECT * FROM "+ 
            self.ui.lineEditTableName.text()
        try:
            conn = sqlite3.connect(self.ui.lineEditDBName.
            text()+ ".db")
            cur = conn.cursor()
            cur.execute(sqlStatement)
            rows = cur.fetchall()
            rowNo=0
        for tuple in rows:
            self.ui.labelResponse.setText("")
            colNo=0
        for columns in tuple:
            oneColumn=QTableWidgetItem(columns)
            self.ui.tableWidget.setItem(rowNo, colNo, oneColumn)
            colNo+=1
            rowNo+=1
        except Error as e:
            self.ui.tableWidget.clear()
            self.ui.labelResponse.setText("Error in accessing  
            table")
        finally:
            conn.close()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理…

您可以在脚本中看到，按钮的 click()事件与 objectName 属性`pushButtonDisplayRows`连接到`DisplayRows()`方法。这意味着每当单击此按钮时，将调用`DisplayRows()`方法。在`DisplayRows()`方法中，定义了一个`SQL SELECT`语句，该语句从在行编辑小部件中指定的表中获取行。还与在行编辑小部件中输入的数据库名称建立了连接。然后执行`SQL SELECT`语句。在光标上执行`fetchall()`方法，以保留从数据库表中访问的所有行。

执行`for`循环以一次访问接收到的行中的一个元组，并再次在元组上执行`for`循环以获取该行中每一列的数据。在表小部件中显示分配给行每一列的数据。在显示第一行的数据后，从行中选择第二行，并重复该过程以在表小部件中显示第二行的数据。两个嵌套的`for`循环一直执行，直到通过表小部件显示所有行。

运行应用程序时，您将被提示指定数据库名称和表名。输入所需信息后，单击“显示行”按钮，指定数据库表的内容将通过表部件显示，如下截图所示：

![](img/5e0fe826-4244-4c96-8cde-7e30a582d172.png)

# 浏览指定数据库表的行

在本教程中，我们将学习逐个从给定数据库表中获取行。也就是说，运行应用程序时，将显示数据库表的第一行。应用程序中提供了四个按钮，称为 Next、Previous、First 和 Last。顾名思义，单击 Next 按钮将显示序列中的下一行。类似地，单击 Previous 按钮将显示序列中的上一行。单击 Last 按钮将显示数据库表的最后一行，单击 First 按钮将显示数据库表的第一行。

# 如何做…

以下是了解如何逐个访问和显示数据库表中的行的步骤：

1.  让我们基于没有按钮的对话框模板创建一个应用程序。

1.  通过拖放三个标签部件、两个行编辑部件和四个按钮部件将它们添加到表单上。

1.  将两个标签部件的文本属性设置为`Email Address`和`Password`。

1.  删除第三个标签部件的文本属性，因为它的文本属性将通过代码设置。

1.  将四个按钮的文本属性设置为`First Row`、`Previous`、`Next`和`Last Row`。

1.  将两个行编辑部件的 objectName 属性设置为`lineEditEmailAddress`和`lineEditPassword`。

1.  将四个按钮的 objectName 属性设置为`pushButtonFirst`、`pushButtonPrevious`、`pushButtonNext`和`pushButtonLast`。

1.  将第三个标签部件的 objectName 属性设置为`labelResponse`。因为我们不希望密码被显示，我们希望用户输入密码时出现星号。

1.  选择用于输入密码的行编辑部件（lineEditPassword），从属性编辑器窗口中选择 echoMode 属性，并将其设置为 Password，而不是默认的 Normal。

1.  将应用程序保存为`demoShowRecords`。表单现在将显示如下截图所示：

![](img/2a4ab6c3-609c-4fd9-96b0-b375a38941dd.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件，应用`pyuic5`命令后，XML 文件可以转换为 Python 代码。书籍的源代码包中可以看到生成的 Python 脚本`demoShowRecords.py`。

1.  将`demoShowRecords.py`脚本视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callShowRecords.pyw`的 Python 文件，并将`demoShowRecords.py`代码导入其中。

```py
import sqlite3, sys
from PyQt5.QtWidgets import QDialog, QApplication,QTableWidgetItem
from sqlite3 import Error
from demoShowRecords import *
rowNo=1
sqlStatement="SELECT EmailAddress, Password FROM Users"
conn = sqlite3.connect("ECommerce.db")
cur = conn.cursor()
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        cur.execute(sqlStatement)
        self.ui.pushButtonFirst.clicked.connect(self.
        ShowFirstRow)
        self.ui.pushButtonPrevious.clicked.connect(self.
        ShowPreviousRow)
        self.ui.pushButtonNext.clicked.connect(self.ShowNextRow)
        self.ui.pushButtonLast.clicked.connect(self.ShowLastRow)
        self.show()
    def ShowFirstRow(self):
        try:
            cur.execute(sqlStatement)
            row=cur.fetchone()
        if row:
            self.ui.lineEditEmailAddress.setText(row[0])
            self.ui.lineEditPassword.setText(row[1])
        except Error as e:
            self.ui.labelResponse.setText("Error in accessing  
            table")
    def ShowPreviousRow(self):
        global rowNo
        rowNo -= 1
        sqlStatement="SELECT EmailAddress, Password FROM Users  
        where rowid="+str(rowNo)
        cur.execute(sqlStatement)
        row=cur.fetchone()
        if row: 
            self.ui.labelResponse.setText("")
            self.ui.lineEditEmailAddress.setText(row[0])
            self.ui.lineEditPassword.setText(row[1])
        else:
            rowNo += 1
            self.ui.labelResponse.setText("This is the first  
            row")
        def ShowNextRow(self):
            global rowNo
            rowNo += 1
            sqlStatement="SELECT EmailAddress, Password FROM  
            Users where rowid="+str(rowNo)
            cur.execute(sqlStatement)
            row=cur.fetchone()
            if row:
                self.ui.labelResponse.setText("")
                self.ui.lineEditEmailAddress.setText(row[0])
                self.ui.lineEditPassword.setText(row[1])
            else:
                rowNo -= 1
                self.ui.labelResponse.setText("This is the last  
                row")
    def ShowLastRow(self):
        cur.execute(sqlStatement)
        for row in cur.fetchall():
            self.ui.lineEditEmailAddress.setText(row[0])
            self.ui.lineEditPassword.setText(row[1])
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的…

您可以在脚本中看到，具有 objectName 属性`pushButtonFirst`的按钮的 click()事件连接到`ShowFirstRow()`方法，具有 objectName 属性`pushButtonPrevious`的按钮连接到`ShowPreviousRow()`方法，具有 objectName 属性`pushButtonNext`的按钮连接到`ShowNextRow()`方法，具有 objectName 属性`pushButtonLast`的按钮连接到`ShowLastRow()`方法。

每当单击按钮时，将调用相关方法。

在`ShowFirstRow()`方法中，执行了一个`SQL SELECT`语句，获取了`Users`表的电子邮件地址和密码列。在光标上执行了`fetchone()`方法，以访问执行`SQL SELECT`语句后接收到的第一行。`EmailAddress`和`Password`列中的数据通过屏幕上的两个 Line Edit 小部件显示出来。如果在访问行时发生错误，错误消息`Error in accessing table`将通过标签小部件显示出来。

为了获取上一行，我们使用了一个全局变量`rowNo`，它被初始化为`1`。在`ShowPreviousRow()`方法中，全局变量`rowNo`的值减少了`1`。然后，执行了一个`SQL SELECT`语句，获取了`Users`表的`EmailAddress`和`Password`列，其中`rowid=rowNo`。因为`rowNo`变量减少了`1`，所以`SQL SELECT`语句将获取序列中的上一行。在光标上执行了`fetchone()`方法，以访问接收到的行，`EmailAddress`和`Password`列中的数据通过屏幕上的两个 Line Edit 小部件显示出来。

如果已经显示了第一行，则点击“上一个”按钮，它将通过标签小部件简单地显示消息“This is the first row”。

在访问序列中的下一行时，我们使用全局变量`rowNo`。在`ShowNextRow()`方法中，全局变量`rowNo`的值增加了`1`。然后，执行了一个`SQL SELECT`语句，获取了`Users`表的`EmailAddress`和`Password`列，其中`rowid=rowNo`；因此，访问了下一行，即`rowid`比当前行高`1`的行。在光标上执行了`fetchone()`方法，以访问接收到的行，`EmailAddress`和`Password`列中的数据通过屏幕上的两个 Line Edit 小部件显示出来。

如果您正在查看数据库表中的最后一行，然后点击“下一个”按钮，它将通过标签小部件简单地显示消息“This is the last row”。

在`ShowLastRow()`方法中，执行了一个`SQL SELECT`语句，获取了`Users`表的`EmailAddress`和`Password`列。在光标上执行了`fetchall()`方法，以访问数据库表中其余的行。使用`for`循环，将`row`变量从执行`SQL SELECT`语句后接收到的行中移动到最后一行。最后一行的`EmailAddress`和`Password`列中的数据通过屏幕上的两个 Line Edit 小部件显示出来。

运行应用程序后，您将在屏幕上看到数据库表的第一行，如下截图所示。如果现在点击“上一个”按钮，您将收到消息“This is the first row”。

![](img/ab778559-635d-4314-b72e-c2ae0545b23f.png)

点击“下一个”按钮后，序列中的下一行将显示在屏幕上，如下截图所示：

![](img/e9aadf57-6616-4a5a-89c4-10a602b9eb67.png)

点击“最后一行”按钮后，数据库表中的最后一行将显示出来，如下截图所示：

![](img/a7f26b14-4a30-43ca-b31e-1f9b7aeebb8a.png)

# 搜索数据库表中的特定信息

在这个示例中，我们将学习如何在数据库表中执行搜索，以获取所需的信息。我们假设用户忘记了他们的密码。因此，您将被提示输入数据库名称、表名称和需要密码的用户的电子邮件地址。如果数据库表中存在使用提供的电子邮件地址的用户，则将搜索、访问并在屏幕上显示该用户的密码。

# 如何做…

按照以下步骤了解如何在 SQLite 数据库表中搜索数据：

1.  让我们基于没有按钮的对话框模板创建一个应用程序。

1.  通过拖放五个 Label 小部件、四个 LineEdit 小部件和一个 PushButton 小部件到表单上，向表单添加五个`QLabel`小部件、四个`QLineEdit`小部件和一个`QPushButton`小部件。

1.  将前三个 Label 小部件的文本属性设置为`输入数据库名称`、`输入表名称`和`电子邮件地址`。

1.  删除第四个 Label 小部件的文本属性，这是通过代码建立的。

1.  将第五个 Label 小部件的文本属性设置为`Password`。

1.  将 PushButton 的文本属性设置为`搜索`。

1.  将四个 LineEdit 小部件的 objectName 属性设置为`lineEditDBName`、`lineEditTableName`、`lineEditEmailAddress`和`lineEditPassword`。

1.  将 PushButton 小部件的 objectName 属性设置为`pushButtonSearch`。

1.  将第四个 Label 小部件的 objectName 属性设置为`labelResponse`。

1.  将应用程序保存为`demoSearchRows.ui`。表单现在将显示如下截图所示：

![](img/3809324e-2973-4898-b19b-678430a5da8a.png)

使用 Qt Designer 创建的用户界面存储在一个`.ui`文件中，这是一个需要通过`pyuic5`命令应用转换为 Python 代码的 XML 文件。书籍的源代码包中可以看到生成的 Python 脚本`demoSearchRows.py`。

1.  将`demoSearchRows.py`脚本视为头文件，并将其导入到您将调用其用户界面设计的文件中。

1.  创建另一个名为`callSearchRows.pyw`的 Python 文件，并将`demoSearchRows.py`代码导入其中：

```py
import sqlite3, sys
from PyQt5.QtWidgets import QDialog, QApplication
from sqlite3 import Error
from demoSearchRows import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonSearch.clicked.connect(self.
        SearchRows)
        self.show()
    def SearchRows(self):
        sqlStatement="SELECT Password FROM  
        "+self.ui.lineEditTableName.text()+" where EmailAddress  
        like'"+self.ui.lineEditEmailAddress.text()+"'"
    try:
        conn = sqlite3.connect(self.ui.lineEditDBName.text()+
        ".db")
        cur = conn.cursor()
        cur.execute(sqlStatement)
        row = cur.fetchone()
    if row==None:
        self.ui.labelResponse.setText("Sorry, No User found with  
        this email address")
        self.ui.lineEditPassword.setText("")
```

```py
    else:
        self.ui.labelResponse.setText("Email Address Found,  
        Password of this User is :")
        self.ui.lineEditPassword.setText(row[0])
    except Error as e:
        self.ui.labelResponse.setText("Error in accessing row")
    finally:
        conn.close()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理...

您可以在脚本中看到，具有 objectName 属性`pushButtonSearch`的 PushButton 的 click()事件连接到`SearchRows()`方法。这意味着每当单击 PushButton 时，都会调用`SearchRows()`方法。在`SearchRows()`方法中，对`sqlite3`类调用`connect()`方法，并将用户在 LineEdit 小部件中输入的数据库名称传递给`connect()`方法。建立与数据库的连接。定义一个 SQL `search`语句，从所提供的表中获取`Password`列，该表中的电子邮件地址与提供的电子邮件地址匹配。在给定的数据库表上执行`search` SQL 语句。在光标上执行`fetchone()`方法，从执行的 SQL 语句中获取一行。如果获取的行不是`None`，即数据库表中有一行与给定的电子邮件地址匹配，则访问该行中的密码，并将其分配给 object 名称为`lineEditPassword`的 LineEdit 小部件以进行显示。最后，关闭与数据库的连接。

如果在执行 SQL 语句时发生错误，即找不到数据库、表名输入错误，或者给定表中不存在电子邮件地址列，则会通过具有 objectName 属性`labelResponse`的 Label 小部件显示错误消息“访问行时出错”。

运行应用程序后，我们会得到一个对话框，提示我们输入数据库名称、表名和表中的列名。假设我们想要找出在`ECommerce`数据库的`Users`表中，邮箱地址为`bmharwani@yahoo.com`的用户的密码。在框中输入所需信息后，当点击搜索按钮时，用户的密码将从表中获取，并通过行编辑小部件显示，如下截图所示：

![](img/ffecd662-0444-472c-adc9-b0709ffc4ed4.png)

如果在 Users 表中找不到提供的电子邮件地址，您将收到消息“抱歉，找不到使用此电子邮件地址的用户”，该消息将通过 Label 小部件显示，如下所示：

![](img/fa36f35c-539b-4640-b78b-5e75d9cd1447.png)

# 创建一个登录表单 - 应用认证程序

在本教程中，我们将学习如何访问特定表中的行，并将其与提供的信息进行比较。

我们假设数据库`ECommerce`已经存在，并且`ECommerce`数据库中也存在名为`Users`的表。`Users`表包括两列，`EmailAddress`和`Password`。此外，我们假设`Users`表中包含一些行。用户将被提示在登录表单中输入其电子邮件地址和密码。将在`Users`表中搜索指定的电子邮件地址。如果在`Users`表中找到电子邮件地址，则将比较该行中的密码与输入的密码。如果两个密码匹配，则显示欢迎消息；否则，显示指示电子邮件地址或密码不匹配的错误消息。

# 如何做…

以下是了解如何将数据库表中的数据与用户输入的数据进行比较并对用户进行身份验证的步骤：

1.  让我们基于没有按钮的对话框模板创建一个应用程序。

1.  在表单中通过拖放三个 Label 小部件、两个 Line Edit 小部件和一个 Push Button 小部件，添加三个`QLabel`小部件、两个`QLineEdit`小部件和一个`QPushButton`小部件。

1.  将前两个 Label 小部件的文本属性设置为`电子邮件地址`和`密码`。

1.  通过代码删除第三个 Label 小部件的文本属性。

1.  将按钮的文本属性设置为`登录`。

1.  将两个 Line Edit 小部件的 objectName 属性设置为`lineEditEmailAddress`和`lineEditPassword`。

1.  将 Push Button 小部件的 objectName 属性设置为`pushButtonSearch`。

1.  将第三个 Label 小部件的 objectName 属性设置为`labelResponse`。

1.  将应用程序保存为`demoSignInForm.ui`。表单现在将显示如下截图所示：

![](img/8ed32324-b44a-44ce-93b7-6696182ccd07.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。通过应用`pyuic5`命令，可以将 XML 文件转换为 Python 代码。生成的 Python 脚本`demoSignInForm.py`可以在本书的源代码包中找到。

1.  将`demoSignInForm.py`文件视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callSignInForm.pyw`的 Python 文件，并将`demoSignInForm.py`代码导入其中：

```py
import sqlite3, sys
from PyQt5.QtWidgets import QDialog, QApplication
from sqlite3 import Error
from demoSignInForm import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonSearch.clicked.connect(self.
        SearchRows)
        self.show()
    def SearchRows(self):
        sqlStatement="SELECT EmailAddress, Password FROM Users   
        where EmailAddress like'"+self.ui.lineEditEmailAddress.
        text()+"'and Password like '"+ self.ui.lineEditPassword.
        text()+"'"
        try:
            conn = sqlite3.connect("ECommerce.db")
            cur = conn.cursor()
            cur.execute(sqlStatement)
            row = cur.fetchone()
        if row==None:
            self.ui.labelResponse.setText("Sorry, Incorrect  
            email address or password ")
        else:
            self.ui.labelResponse.setText("You are welcome ")
        except Error as e:
            self.ui.labelResponse.setText("Error in accessing 
            row")
        finally:
            conn.close()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理…

您可以在脚本中看到，具有 objectName 属性`pushButtonSearch`的按钮的单击事件与`SearchRows()`方法相连。这意味着每当单击按钮时，都会调用`SearchRows()`方法。在`SearchRows()`方法中，调用`sqlite3`类的`connect()`方法与`ECommerce`数据库建立连接。定义了一个 SQL `search`语句，该语句从`Users`表中获取`EmailAddress`和`Password`列，这些列的电子邮件地址与提供的电子邮件地址匹配。在`Users`表上执行`search` SQL 语句。在光标上执行`fetchone()`方法，以从执行的 SQL 语句中获取一行。如果获取的行不是`None`，即数据库表中存在与给定电子邮件地址和密码匹配的行，则会通过具有 objectName 属性`labelResponse`的 Label 小部件显示欢迎消息。最后，关闭与数据库的连接。

如果在执行 SQL 语句时发生错误，如果找不到数据库，或者表名输入错误，或者`Users`表中不存在电子邮件地址或密码列，则通过具有 objectName 属性`labelResponse`的 Label 小部件显示错误消息“访问行时出错”。

运行应用程序时，您将被提示输入电子邮件地址和密码。输入正确的电子邮件地址和密码后，当您单击“登录”按钮时，您将收到消息“欢迎”，如下截图所示：

![](img/0b0a8b36-ce5e-46f4-9667-fdd73d9cded9.png)

但是，如果电子邮件地址或密码输入不正确，您将收到消息“抱歉，电子邮件地址或密码不正确”，如下截图所示：

![](img/827e320d-80a2-4214-bc21-a1020968ee47.png)

# 更新数据库表-更改用户密码

在这个示例中，您将学习如何更新数据库中的任何信息。在几乎所有应用程序中，更改密码都是一个非常常见的需求。在这个示例中，我们假设一个名为`ECommerce`的数据库已经存在，并且`ECommerce`数据库中也存在一个名为`Users`的表。`Users`表包含两列，`EmailAddress`和`Password`。此外，我们假设`Users`表中已经包含了一些行。用户将被提示在表单中输入他们的电子邮件地址和密码。将搜索`Users`表以查找指定的电子邮件地址和密码。如果找到具有指定电子邮件地址和密码的行，则将提示用户输入新密码。新密码将被要求输入两次，也就是说，用户将被要求在新密码框和重新输入新密码框中输入他们的新密码。如果两个框中输入的密码匹配，密码将被更改，也就是说，旧密码将被新密码替换。

# 如何做...

从数据库表中删除数据的过程非常关键，执行此类应用程序的任何错误都可能导致灾难。以下是从给定数据库表中删除任何行的步骤：

1.  让我们基于没有按钮的对话框模板创建一个应用程序。

1.  通过拖放五个标签小部件、四个行编辑小部件和一个按钮小部件将它们添加到表单上。

1.  将前四个标签小部件的文本属性设置为`电子邮件地址`、`旧密码`、`新密码`和`重新输入新密码`。

1.  删除第五个标签小部件的文本属性，这是通过代码建立的。将按钮的文本属性设置为`更改密码`。

1.  将四个行编辑小部件的 objectName 属性设置为`lineEditEmailAddress`、`lineEditOldPassword`、`lineEditNewPassword`和`lineEditRePassword`。由于我们不希望密码显示在与密码相关联的任何行编辑小部件中，我们希望用户输入密码时显示星号。

1.  依次从属性编辑器窗口中选择三个行编辑小部件。

1.  选择 echoMode 属性，并将其设置为`Password`，而不是默认的 Normal。

1.  将按钮小部件的 objectName 属性设置为`pushButtonChangePassword`。

1.  将第五个标签小部件的 objectName 属性设置为`labelResponse`。

1.  将应用程序保存为`demoChangePassword.ui`。表单现在将显示如下截图所示：

![](img/7ac023e9-4e95-4f52-b1ec-a2d92853d777.png)

使用 Qt Designer 创建的用户界面存储在`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。`pyuic5`命令用于将 XML 文件转换为 Python 代码。本书的源代码包中可以看到生成的 Python 脚本`demoChangePassword.py`。 

1.  将`demoChangePassword.py`脚本视为头文件，并将其导入到您将调用其用户界面设计的文件中。

1.  创建另一个名为`callChangePassword.pyw`的 Python 文件，并将`demoChangePassword.py`代码导入其中：

```py
import sqlite3, sys
from PyQt5.QtWidgets import QDialog, QApplication
from sqlite3 import Error
from demoChangePassword import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonChangePassword.clicked.connect(self.
        ChangePassword)
        self.show()
    def ChangePassword(self):
        selectStatement="SELECT EmailAddress, Password FROM   
        Users where EmailAddress like '"+self.ui.
        lineEditEmailAddress.text()+"'and Password like '"+         
        self.ui.lineEditOldPassword.text()+"'"
        try:
            conn = sqlite3.connect("ECommerce.db")
            cur = conn.cursor()
            cur.execute(selectStatement)
            row = cur.fetchone()
        if row==None:
            self.ui.labelResponse.setText("Sorry, Incorrect  
            email address or password")
        else:
        if self.ui.lineEditNewPassword.text()==  
          self.ui.lineEditRePassword.text():
            updateStatement="UPDATE Users set Password = '" +             
            self.ui.lineEditNewPassword.text()+"' WHERE   
            EmailAddress like'"+self.ui.lineEditEmailAddress.
            text()+"'"
        with conn:
            cur.execute(updateStatement)
            self.ui.labelResponse.setText("Password successfully 
            changed")
        else:
            self.ui.labelResponse.setText("The two passwords 
            don't match")
        except Error as e:
            self.ui.labelResponse.setText("Error in accessing 
            row")
        finally:
            conn.close()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

您可以在脚本中看到，具有 objectName 属性`pushButtonChangePassword`的按钮的 click()事件与`ChangePassword()`方法相连。这意味着每当单击按钮时，都会调用`ChangePassword()`方法。在`ChangePassword()`方法中，调用`sqlite3`类的`connect()`方法与`ECommerce`数据库建立连接。定义了一个 SQL `SELECT`语句，该语句从`Users`表中获取与在 LineEdit 小部件中输入的电子邮件地址和密码匹配的`EmailAddress`和`Password`列。在`Users`表上执行 SQL `SELECT`语句。在光标上执行`fetchone()`方法，以从执行的 SQL 语句中获取一行。如果获取的行不是`None`，即数据库表中有一行，则确认两个 LineEdit 小部件`lineEditNewPassword`和`lineEditRePassword`中输入的新密码是否完全相同。如果两个密码相同，则执行`UPDATE` SQL 语句来更新`Users`表，将密码更改为新密码。

如果两个密码不匹配，则不会对数据库表进行更新，并且通过 Label 小部件显示消息“两个密码不匹配”。

如果在执行 SQL `SELECT`或`UPDATE`语句时发生错误，则会通过具有 objectName 属性`labelResponse`的 Label 小部件显示错误消息“访问行时出错”。

运行应用程序时，您将被提示输入电子邮件地址和密码，以及新密码。如果电子邮件地址或密码不匹配，则会通过 Label 小部件显示错误消息“抱歉，电子邮件地址或密码不正确”，如下面的屏幕截图所示：

![](img/77868907-dab6-407a-ba11-6a3de7557bf2.png)

如果输入的电子邮件地址和密码正确，但在新密码和重新输入新密码框中输入的新密码不匹配，则屏幕上会显示消息“两个密码不匹配”，如下面的屏幕截图所示：

![](img/d50c910f-cd03-4074-8b50-2e3c424b120c.png)

如果电子邮件地址和密码都输入正确，也就是说，如果在数据库表中找到用户行，并且在新密码和重新输入新密码框中输入的新密码匹配，则更新`Users`表，并且在成功更新表后，屏幕上会显示消息“密码已成功更改”，如下面的屏幕截图所示：

![](img/03512e11-26e0-47bc-a2b2-4ab3b27a238d.png)

# 从数据库表中删除一行

在本教程中，我们将学习如何从数据库表中删除一行。我们假设名为`ECommerce`的数据库已经存在，并且`ECommerce`数据库中也存在名为`Users`的表。`Users`表包含两列，`EmailAddress`和`Password`。此外，我们假设`User`表中包含一些行。用户将被提示在表单中输入他们的电子邮件地址和密码。将在`Users`表中搜索指定的电子邮件地址和密码。如果在`Users`表中找到具有指定电子邮件地址和密码的任何行，则将提示您确认是否确定要删除该行。如果单击“是”按钮，则将删除该行。

# 如何做…

从数据库表中删除数据的过程非常关键，执行此类应用程序时的任何错误都可能导致灾难。以下是从给定数据库表中删除任何行的步骤：

1.  让我们基于没有按钮的对话框模板创建一个应用程序。

1.  通过将四个 Label 小部件、两个 LineEdit 小部件和三个 PushButton 小部件拖放到表单上，向表单添加四个`QLabel`小部件、两个`QLineEdit`小部件和三个`QPushButton`小部件。

1.  将前三个 Label 小部件的文本属性设置为`电子邮件地址`，`密码`和`你确定吗？`

1.  删除第四个 Label 小部件的文本属性，这是通过代码建立的。

1.  将三个按钮的文本属性设置为`删除用户`，`是`和`否`。

1.  将两个 Line Edit 小部件的 objectName 属性设置为`lineEditEmailAddress`和`lineEditPassword`。

1.  将三个 Push Button 小部件的 objectName 属性设置为`pushButtonDelete`，`pushButtonYes`和`pushButtonNo`。

1.  将第四个 Label 小部件的 objectName 属性设置为`labelResponse`。

1.  将应用程序保存为`demoDeleteUser.ui`。表单现在将显示如下截图所示：

![](img/da1a933e-c763-40db-8e88-c2e13c944c60.png)

使用 Qt Designer 创建的用户界面存储在一个`.ui`文件中，这是一个 XML 文件，需要转换为 Python 代码。使用`pyuic5`命令将 XML 文件转换为 Python 代码。生成的 Python 脚本`demoDeleteUser.py`可以在本书的源代码包中找到。

1.  将`demoDeleteUser.py`脚本视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callDeleteUser.pyw`的 Python 文件，并将`demoDeleteUser.py`代码导入其中：

```py
import sqlite3, sys
from PyQt5.QtWidgets import QDialog, QApplication
from sqlite3 import Error
from demoDeleteUser import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonDelete.clicked.connect(self.
        DeleteUser)
        self.ui.pushButtonYes.clicked.connect(self.
        ConfirmDelete)
        self.ui.labelSure.hide()
        self.ui.pushButtonYes.hide()
        self.ui.pushButtonNo.hide()
        self.show()
    def DeleteUser(self):
        selectStatement="SELECT * FROM Users where EmailAddress  
        like'"+self.ui.lineEditEmailAddress.text()+"' 
        and Password like '"+ self.ui.lineEditPassword.
        text()+"'"
        try:
            conn = sqlite3.connect("ECommerce.db")
            cur = conn.cursor()
            cur.execute(selectStatement)
            row = cur.fetchone()
        if row==None:
            self.ui.labelSure.hide()
            self.ui.pushButtonYes.hide()
            self.ui.pushButtonNo.hide()
            self.ui.labelResponse.setText("Sorry, Incorrect 
            email address or password ")
        else:
            self.ui.labelSure.show()
            self.ui.pushButtonYes.show()
            self.ui.pushButtonNo.show()
            self.ui.labelResponse.setText("")
        except Error as e:
            self.ui.labelResponse.setText("Error in accessing 
            user account")
        finally:
            conn.close()
    def ConfirmDelete(self):
        deleteStatement="DELETE FROM Users where EmailAddress    
        like '"+self.ui.lineEditEmailAddress.text()+"' 
        and Password like '"+ self.ui.lineEditPassword.
        text()+"'"
        try:
            conn = sqlite3.connect("ECommerce.db")
            cur = conn.cursor()
        with conn:
            cur.execute(deleteStatement)
            self.ui.labelResponse.setText("User successfully 
            deleted")
        except Error as e:
            self.ui.labelResponse.setText("Error in deleting 
            user account")
        finally:
            conn.close()
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理…

在这个应用程序中，带有文本“你确定吗？”的 Label 小部件和两个按钮 Yes 和 No 最初是隐藏的。只有当用户输入的电子邮件地址和密码在数据库表中找到时，这三个小部件才会显示出来。这三个小部件使用户能够确认他们是否真的想要删除该行。因此，对这三个小部件调用`hide()`方法，使它们最初不可见。此外，将具有 objectName 属性`pushButtonDelete`的按钮的 click()事件连接到`DeleteUser()`方法。这意味着每当单击删除按钮时，都会调用`DeleteUser()`方法。类似地，具有 objectName 属性`pushButtonYes`的按钮的 click()事件连接到`ConfirmDelete()`方法。这意味着当用户通过单击 Yes 按钮确认删除该行时，将调用`ConfirmDelete()`方法。

在`DeleteUser()`方法中，首先搜索是否存在与输入的电子邮件地址和密码匹配的`Users`表中的任何行。在`sqlite3`类上调用`connect()`方法，与`ECommerce`数据库建立连接。定义了一个 SQL `SELECT`语句，从`Users`表中获取`EmailAddress`和`Password`列，其电子邮件地址和密码与提供的电子邮件地址和密码匹配。在`Users`表上执行 SQL `SELECT`语句。在游标上执行`fetchone()`方法，从执行的 SQL 语句中获取一行。如果获取的行不是`None`，即数据库表中存在与给定电子邮件地址和密码匹配的行，则会使三个小部件，Label 和两个按钮可见。用户将看到消息“你确定吗？”，然后是两个带有文本 Yes 和 No 的按钮。

如果用户单击 Yes 按钮，则会执行`ConfirmDelete()`方法。在`ConfirmDelete()`方法中，定义了一个 SQL `DELETE`方法，用于从`Users`表中删除与输入的电子邮件地址和密码匹配的行。在与`ECommerce`数据库建立连接后，执行 SQL `DELETE`方法。如果成功从`Users`表中删除了行，则通过 Label 小部件显示消息“用户成功删除”；否则，将显示错误消息“删除用户帐户时出错”。

在运行应用程序之前，我们将启动一个名为 SQLite 数据库浏览器的可视化工具。该可视化工具使我们能够创建，编辑和查看数据库表及其中的行。使用 SQLite 数据库浏览器，我们将首先查看“用户”表中的现有行。之后，应用程序将运行并删除一行。再次从 SQLite 数据库浏览器中，我们将确认该行是否真的已从“用户”表中删除。

因此，启动 SQLite 数据库浏览器并在主菜单下方点击“打开数据库”选项卡。浏览并从当前文件夹中选择“电子商务”数据库。 “电子商务”数据库显示由两列“电子邮件地址”和“密码”组成的“用户”表。单击“执行 SQL”按钮以编写 SQL 语句。在窗口中，写入 SQL 语句`select * from Users`，然后单击运行图标。 “用户”表中的所有现有行将显示在屏幕上。您可以在以下屏幕截图中看到“用户”表有两行：

![](img/f8bf5ccc-af6f-405e-a4e6-775a76665be4.png)

运行应用程序后，您将被提示输入您的电子邮件地址和密码。如果您输入错误的电子邮件地址和密码，您将收到消息“抱歉，电子邮件地址或密码不正确”，如下面的屏幕截图所示：

![](img/4e593290-b796-4841-8621-20bb4bba5aa5.png)

在输入正确的电子邮件地址和密码后，当您点击删除用户按钮时，三个小部件——标签小部件和两个按钮，将变为可见，并且您会收到消息“您确定吗？”，以及两个按钮“Yes”和“No”，如下面的屏幕截图所示：

![](img/889013f0-8d85-4888-a358-90ea905de172.png)

点击“Yes”按钮后，“用户”表中与提供的电子邮件地址和密码匹配的行将被删除，并且通过标签小部件显示确认消息“用户成功删除”，如下面的屏幕截图所示：

![](img/69261dfc-02f4-4a0d-a063-7d4c1fd4ba6c.png)

让我们通过可视化工具检查行是否实际上已从用户表中删除。因此，启动 SQLite 数据库浏览器并在主菜单下方点击“打开数据库”选项卡。浏览并从当前文件夹中选择“电子商务”数据库。 “电子商务”数据库将显示“用户”表。单击“执行 SQL”按钮以编写 SQL 语句。在窗口中，写入 SQL 语句`select * from Users`，然后单击运行图标。 “用户”表中的所有现有行将显示在屏幕上。

在运行应用程序之前，我们看到“用户”表中有两行。这次，您只能在“用户”表中看到一行（请参阅下面的屏幕截图），证实已从“用户”表中删除了一行：

![](img/332e5a96-cbdf-4514-93c9-6a1aa708de21.png)
