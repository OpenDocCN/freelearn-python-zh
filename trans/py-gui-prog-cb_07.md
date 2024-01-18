# 第7章。通过我们的GUI将数据存储在MySQL数据库中

在本章中，我们将通过连接到MySQL数据库来增强我们的Python GUI。

+   从Python连接到MySQL数据库

+   配置MySQL连接

+   设计Python GUI数据库

+   使用SQL INSERT命令

+   使用SQL UPDATE命令

+   使用SQL DELETE命令

+   从我们的MySQL数据库中存储和检索数据

# 介绍

在我们可以连接到MySQL服务器之前，我们必须先访问MySQL服务器。本章的第一个步骤将向您展示如何安装免费的MySQL服务器社区版。

成功连接到我们的MySQL服务器运行实例后，我们将设计并创建一个数据库，该数据库将接受一本书的标题，这可能是我们自己的日记或者是我们在互联网上找到的引用。我们将需要书的页码，这可能为空白，然后我们将使用我们在Python 3中构建的GUI将我们喜欢的引用从一本书、日记、网站或朋友中`插入`到我们的MySQL数据库中。

我们将使用我们的Python GUI来插入、修改、删除和显示我们喜欢的引用，以发出这些SQL命令并显示数据。

### 注意

**CRUD**是您可能遇到的一个数据库术语，它缩写了四个基本的SQL命令，代表**创建**、**读取**、**更新**和**删除**。

# 从Python连接到MySQL数据库

在我们可以连接到MySQL数据库之前，我们必须先连接到MySQL服务器。

为了做到这一点，我们需要知道MySQL服务器的IP地址以及它所监听的端口。

我们还必须是一个注册用户，并且需要密码才能被MySQL服务器验证。

## 准备工作

您需要访问一个正在运行的MySQL服务器实例，并且您还需要具有管理员权限才能创建数据库和表。

在官方MySQL网站上有一个免费的MySQL社区版可用。您可以从以下网址在本地PC上下载并安装它：[http://dev.mysql.com/downloads/](http://dev.mysql.com/downloads/)

### 注意

在本章中，我们使用的是MySQL社区服务器（GPL）版本：5.6.26。

## 如何做...

为了连接到MySQL，我们首先需要安装一个特殊的Python连接器驱动程序。这个驱动程序将使我们能够从Python与MySQL服务器通信。

该驱动程序可以在MySQL网站上免费获得，并附带一个非常好的在线教程。您可以从以下网址安装它：

[http://dev.mysql.com/doc/connector-python/en/index.html](http://dev.mysql.com/doc/connector-python/en/index.html)

### 注意

确保选择与您安装的Python版本匹配的安装程序。在本章中，我们使用Python 3.4的安装程序。

![如何做...](graphics/B04829_07_01.jpg)

在安装过程的最后，目前有一点小小的惊喜。当我们启动`.msi`安装程序时，我们会短暂地看到一个显示安装进度的MessageBox，但然后它就消失了。我们没有收到安装是否成功的确认。

验证我们是否安装了正确的驱动程序，让Python能够与MySQL通信，一种方法是查看Python site-packages目录。

如果您的site-packages目录看起来类似于以下屏幕截图，并且您看到一些新文件的名称中带有`mysql_connector_python`，那么我们确实安装了一些东西...

![如何做...](graphics/B04829_07_02.jpg)

上述提到的官方MySQL网站附带一个教程，网址如下：

[http://dev.mysql.com/doc/connector-python/en/connector-python-tutorials.html](http://dev.mysql.com/doc/connector-python/en/connector-python-tutorials.html)

在线教程示例中关于验证安装Connector/Python驱动程序是否成功的部分有点误导，因为它试图连接到一个员工数据库，这个数据库在我的社区版中并没有自动创建。

验证我们的Connector/Python驱动程序是否真的安装了的方法是，只需连接到MySQL服务器而不指定特定的数据库，然后打印出连接对象。

### 注意

用你在MySQL安装中使用的真实凭据替换占位符括号名称`<adminUser>`和`<adminPwd>`。

如果您安装了MySQL社区版，您就是管理员，并且在MySQL安装过程中会选择用户名和密码。

```py
import mysql.connector as mysql

conn = mysql.connect(user=<adminUser>, password=<adminPwd>,
                     host='127.0.0.1')
print(conn)

conn.close()
```

如果运行上述代码导致以下输出打印到控制台，则表示正常。

![如何做...](graphics/B04829_07_03.jpg)

如果您无法连接到MySQL服务器，那么在安装过程中可能出了问题。如果是这种情况，请尝试卸载MySQL，重新启动您的PC，然后再次运行MySQL安装程序。仔细检查您下载的MySQL安装程序是否与您的Python版本匹配。如果您安装了多个版本的Python，有时会导致混淆，因为您最后安装的版本会被添加到Windows路径环境变量中，并且一些安装程序只会使用在此位置找到的第一个Python版本。

当我安装了Python 32位版本并且我困惑为什么一些我下载的模块无法工作时，这种情况发生了。

安装程序下载了32位模块，这些模块与64位版本的Python不兼容。

## 它是如何工作的...

为了将我们的GUI连接到MySQL服务器，如果我们想创建自己的数据库，我们需要能够以管理员权限连接到服务器。

如果数据库已经存在，那么我们只需要连接、插入、更新和删除数据的授权权限。

在下一个教程中，我们将在MySQL服务器上创建一个新的数据库。

# 配置MySQL连接

在上一个教程中，我们使用了最短的方式通过将用于身份验证的凭据硬编码到`connection`方法中来连接到MySQL服务器。虽然这是早期开发的快速方法，但我们绝对不希望将我们的MySQL服务器凭据暴露给任何人，除非我们*授予*特定用户对数据库、表、视图和相关数据库命令的权限。

通过将凭据存储在配置文件中，通过MySQL服务器进行身份验证的一个更安全的方法是我们将在本教程中实现的。

我们将使用我们的配置文件连接到MySQL服务器，然后在MySQL服务器上创建我们自己的数据库。

### 注意

我们将在所有接下来的教程中使用这个数据库。

## 准备工作

需要具有管理员权限的运行中的MySQL服务器才能运行本教程中显示的代码。

### 注意

上一个教程展示了如何安装免费的MySQL服务器社区版。管理员权限将使您能够实现这个教程。

## 如何做...

首先，在`MySQL.py`代码的同一模块中创建一个字典。

```py
# create dictionary to hold connection info
dbConfig = {
    'user': <adminName>,      # use your admin name 
    'password': <adminPwd>,   # use your admin password
    'host': '127.0.0.1',      # IP address of localhost
    }
```

接下来，在连接方法中，我们解压字典的值。而不是写成，

```py
mysql.connect('user': <adminName>,  'password': <adminPwd>, 'host': '127.0.0.1') 
```

我们使用`(**dbConfig)`，这与上面的方法相同，但更简洁。

```py
import mysql.connector as mysql
# unpack dictionary credentials 
conn = mysql.connect(**dbConfig)
print(conn)
```

这将导致与MySQL服务器的相同成功连接，但不同之处在于连接方法不再暴露任何关键任务信息。

### 注意

数据库服务器对你的任务至关重要。一旦你丢失了宝贵的数据...并且找不到任何最近的备份时，你就会意识到这一点！

![如何做...](graphics/B04829_07_04.jpg)

现在，在同一个Python模块中将相同的用户名、密码、数据库等放入字典中并不能消除任何人浏览代码时看到凭据的风险。

为了增加数据库安全性，我们首先将字典移到自己的Python模块中。让我们称这个新的Python模块为`GuiDBConfig.py`。

然后我们导入这个模块并解压凭据，就像之前做的那样。

```py
import GuiDBConfig as guiConf
# unpack dictionary credentials 
conn = mysql.connect(**guiConf.dbConfig)
print(conn)
```

### 注意

一旦我们将这个模块放在一个安全的地方，与其余代码分开，我们就为我们的MySQL数据实现了更高级别的安全性。

现在我们知道如何连接到MySQL并具有管理员权限，我们可以通过发出以下命令来创建我们自己的数据库：

```py
GUIDB = 'GuiDB'

# unpack dictionary credentials 
conn = mysql.connect(**guiConf.dbConfig)

cursor = conn.cursor()

try:
    cursor.execute("CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(GUIDB))

except mysql.Error as err:
    print("Failed to create DB: {}".format(err))

conn.close()
```

为了执行对MySQL的命令，我们从连接对象创建一个游标对象。

游标通常是数据库表中特定行的位置，我们可以在表中向上或向下移动，但在这里我们使用它来创建数据库本身。

我们将Python代码包装到`try...except`块中，并使用MySQL的内置错误代码告诉我们是否出现了任何问题。

我们可以通过执行创建数据库的代码两次来验证此块是否有效。第一次，它将在MySQL中创建一个新数据库，第二次将打印出一个错误消息，说明此数据库已经存在。

我们可以通过使用完全相同的游标对象语法执行以下MySQL命令来验证哪些数据库存在。

我们不是发出`CREATE DATABASE`命令，而是创建一个游标并使用它来执行`SHOW DATABASES`命令，然后获取并打印到控制台输出的结果。

```py
import mysql.connector as mysql
import GuiDBConfig as guiConf

# unpack dictionary credentials 
conn = mysql.connect(**guiConf.dbConfig)

cursor = conn.cursor()

cursor.execute("SHOW DATABASES")
print(cursor.fetchall())

conn.close()
```

### 注意

我们通过在游标对象上调用`fetchall`方法来检索结果。

运行此代码会显示我们的MySQL服务器实例中当前存在哪些数据库。从输出中可以看到，MySQL附带了几个内置数据库，例如`information_schema`等。我们已成功创建了自己的`guidb`数据库，如输出所示。所有其他数据库都是MySQL附带的。

![如何操作...](graphics/B04829_07_05.jpg)

请注意，尽管我们在创建时指定了数据库的混合大小写字母为GuiDB，但`SHOW DATABASES`命令显示MySQL中所有现有数据库的小写形式，并将我们的数据库显示为`guidb`。

## 它是如何工作的...

为了将我们的Python GUI连接到MySQL数据库，我们首先必须知道如何连接到MySQL服务器。这需要建立一个连接，只有当我们能够提供所需的凭据时，MySQL才会接受这个连接。

虽然将字符串放入一行Python代码很容易，但在处理数据库时，我们必须非常谨慎，因为今天的个人沙箱开发环境，明天很容易就可能变成全球网络上可以访问的环境。

您不希望危害数据库安全性，这个配方的第一部分展示了通过将MySQL服务器的连接凭据放入一个单独的文件，并将此文件放在外部世界无法访问的位置，来更安全地放置连接凭据的方法，我们的数据库系统将变得更加安全。

在真实的生产环境中，MySQL服务器安装、连接凭据和dbConfig文件都将由IT系统管理员处理，他们将使您能够导入dbConfig文件以连接到MySQL服务器，而您不知道实际的凭据是什么。解压dbConfig不会像我们的代码那样暴露凭据。

第二部分在MySQL服务器实例中创建了我们自己的数据库，我们将在接下来的配方中扩展并使用这个数据库，将其与我们的Python GUI结合使用。

# 设计Python GUI数据库

在开始创建表并向其中插入数据之前，我们必须设计数据库。与更改本地Python变量名称不同，一旦创建并加载了数据的数据库模式就不那么容易更改。

在删除表之前，我们必须提取数据，然后`DROP`表，并以不同的名称重新创建它，最后重新导入原始数据。

你明白了...

设计我们的GUI MySQL数据库首先意味着考虑我们希望我们的Python应用程序如何使用它，然后选择与预期目的相匹配的表名。

## 准备工作

我们正在使用前一篇中创建的MySQL数据库。需要运行一个MySQL实例，前两篇文章介绍了如何安装MySQL和所有必要的附加驱动程序，以及如何创建本章中使用的数据库。

## 操作步骤…

首先，我们将在前几篇中创建的两个标签之间在我们的Python GUI中移动小部件，以便更好地组织我们的Python GUI以连接到MySQL数据库。

我们重命名了几个小部件，并将访问MySQL数据的代码分离到以前称为Tab 1的位置，我们将不相关的小部件移动到我们在早期配方中称为Tab 2的位置。

我们还调整了一些内部Python变量名，以便更好地理解我们的代码。

### 注意

代码可读性是一种编码美德，而不是浪费时间。

我们重构后的Python GUI现在看起来像下面的截图。我们将第一个标签重命名为MySQL，并创建了两个tkinter LabelFrame小部件。我们将顶部的一个标记为Python数据库，它包含两个标签和六个tkinter输入小部件加上三个按钮，我们使用tkinter网格布局管理器将它们排列在四行三列中。

我们将书名和页数输入到输入小部件中，点击按钮将导致插入、检索或修改书籍引用。

底部的LabelFrame有一个**图书引用**的标签，这个框架中的ScrolledText小部件将显示我们的书籍和引用。

![操作步骤…](graphics/B04829_07_06.jpg)

我们将创建两个SQL表来保存我们的数据。第一个将保存书名和书页的数据。然后我们将与第二个表连接，第二个表将保存书籍引用。

我们将通过主键到外键关系将这两个表连接在一起。

所以，现在让我们创建第一个数据库表。

在这之前，让我们先验证一下我们的数据库确实没有表。根据在线MySQL文档，查看数据库中存在的表的命令如下。

### 注意

13.7.5.38 `SHOW` `TABLES` 语法：

```py
SHOW [FULL] TABLES [{FROM | IN} db_name]
    [LIKE 'pattern' | WHERE expr]
```

需要注意的是，在上述语法中，方括号中的参数（如`FULL`）是可选的，而花括号中的参数（如`FROM`）是`SHOW TABLES`命令描述中所需的。在`FROM`和`IN`之间的管道符号表示MySQL语法要求其中一个。

```py
# unpack dictionary credentials 
conn = mysql.connect(**guiConf.dbConfig)
# create cursor 
cursor = conn.cursor()
# execute command
cursor.execute("SHOW TABLES FROM guidb")
print(cursor.fetchall())

# close connection to MySQL
conn.close()
```

当我们在Python中执行SQL命令时，我们得到了预期的结果，即一个空列表，显示我们的数据库当前没有表。

![操作步骤…](graphics/B04829_07_07.jpg)

我们还可以通过执行`USE <DB>`命令首先选择数据库。现在，我们不必将其传递给`SHOW TABLES`命令，因为我们已经选择了要交谈的数据库。

以下代码创建了与之前相同的真实结果：

```py
cursor.execute("USE guidb")
cursor.execute("SHOW TABLES")
```

现在我们知道如何验证我们的数据库中是否有表，让我们创建一些表。创建了两个表之后，我们将使用与之前相同的命令验证它们是否真的进入了我们的数据库。

我们通过执行以下代码创建了第一个名为`Books`的表。

```py
# connect by unpacking dictionary credentials
conn = mysql.connect(**guiConf.dbConfig)

# create cursor 
cursor = conn.cursor()

# select DB
cursor.execute("USE guidb")

# create Table inside DB
cursor.execute("CREATE TABLE Books (       \
      Book_ID INT NOT NULL AUTO_INCREMENT, \
      Book_Title VARCHAR(25) NOT NULL,     \
      Book_Page INT NOT NULL,              \
      PRIMARY KEY (Book_ID)                \
    ) ENGINE=InnoDB")

# close connection to MySQL
conn.close()
```

我们可以通过执行以下命令验证表是否在我们的数据库中创建了。

![操作步骤…](graphics/B04829_07_08.jpg)

现在的结果不再是一个空列表，而是一个包含元组的列表，显示了我们刚刚创建的`books`表。

我们可以使用MySQL命令行客户端查看表中的列。为了做到这一点，我们必须以root用户身份登录。我们还必须在命令的末尾添加一个分号。

### 注意

在Windows上，您只需双击MySQL命令行客户端的快捷方式，这个快捷方式会在MySQL安装过程中自动安装。

如果您的桌面上没有快捷方式，您可以在典型默认安装的以下路径找到可执行文件：

`C:\Program Files\MySQL\MySQL Server 5.6\bin\mysql.exe`

如果没有运行MySQL客户端的快捷方式，您必须传递一些参数：

+   `C:\Program Files\MySQL\MySQL Server 5.6\bin\mysql.exe`

+   `--defaults-file=C:\ProgramData\MySQL\MySQL Server 5.6\my.ini`

+   `-uroot`

+   `-p`

双击快捷方式，或使用完整路径到可执行文件的命令行并传递所需的参数，将打开MySQL命令行客户端，提示您输入root用户的密码。

如果您记得在安装过程中为root用户分配的密码，那么可以运行`SHOW COLUMNS FROM books;`命令，如下所示。这将显示我们的`books`表的列从我们的guidb。

### 注意

在MySQL客户端执行命令时，语法不是Pythonic的。

![如何做…](graphics/B04829_07_09.jpg)

接下来，我们将创建第二个表，用于存储书籍和期刊引用。我们将通过执行以下代码来创建它：

```py
# select DB
cursor.execute("USE guidb")

# create second Table inside DB
cursor.execute("CREATE TABLE Quotations ( \
        Quote_ID INT,                     \
        Quotation VARCHAR(250),           \
        Books_Book_ID INT,                \
        FOREIGN KEY (Books_Book_ID)       \
            REFERENCES Books(Book_ID)     \
            ON DELETE CASCADE             \
    ) ENGINE=InnoDB")
```

执行`SHOW TABLES`命令现在显示我们的数据库有两个表。

![如何做…](graphics/B04829_07_10.jpg)

我们可以通过使用Python执行SQL命令来查看列。

![如何做…](graphics/B04829_07_11.jpg)

使用MySQL客户端可能以更好的格式显示数据。我们还可以使用Python的漂亮打印（`pprint`）功能。

![如何做…](graphics/B04829_07_12.jpg)

MySQL客户端仍然以更清晰的格式显示我们的列，当您运行此客户端时可以看到。

## 工作原理

我们设计了Python GUI数据库，并重构了我们的GUI，以准备使用我们的新数据库。然后我们创建了一个MySQL数据库，并在其中创建了两个表。

我们通过Python和随MySQL服务器一起提供的MySQL客户端验证了表是否成功进入我们的数据库。

在下一个步骤中，我们将向我们的表中插入数据。

# 使用SQL INSERT命令

本步骤介绍了整个Python代码，向您展示如何创建和删除MySQL数据库和表，以及如何显示我们的MySQL实例中现有数据库、表、列和数据。

在创建数据库和表之后，我们将向本步骤中创建的两个表中插入数据。

### 注意

我们正在使用主键到外键的关系来连接两个表的数据。

我们将在接下来的两个步骤中详细介绍这是如何工作的，我们将修改和删除我们的MySQL数据库中的数据。

## 准备工作

本步骤基于我们在上一个步骤中创建的MySQL数据库，并向您展示如何删除和重新创建GuiDB。

### 注意

删除数据库当然会删除数据库中表中的所有数据，因此我们还将向您展示如何重新插入这些数据。

## 如何做…

我们的`MySQL.py`模块的整个代码都在本章的代码文件夹中，可以从Packt Publishing的网站上下载。它创建数据库，向其中添加表，然后将数据插入我们创建的两个表中。

在这里，我们将概述代码，而不显示所有实现细节，以节省空间，因为显示整个代码需要太多页面。

```py
import mysql.connector as mysql
import GuiDBConfig as guiConf

class MySQL():
    # class variable
    GUIDB  = 'GuiDB'   

    #------------------------------------------------------
    def connect(self):
        # connect by unpacking dictionary credentials
        conn = mysql.connector.connect(**guiConf.dbConfig)

        # create cursor 
        cursor = conn.cursor()    

        return conn, cursor

    #------------------------------------------------------
    def close(self, cursor, conn):
        # close cursor

    #------------------------------------------------------
    def showDBs(self):
        # connect to MySQL

    #------------------------------------------------------
    def createGuiDB(self):
        # connect to MySQL

    #------------------------------------------------------
    def dropGuiDB(self):
        # connect to MySQL

    #------------------------------------------------------
    def useGuiDB(self, cursor):
        '''Expects open connection.'''
        # select DB

    #------------------------------------------------------
    def createTables(self):
        # connect to MySQL

        # create Table inside DB

    #------------------------------------------------------
    def dropTables(self):
        # connect to MySQL

    #------------------------------------------------------
    def showTables(self):
        # connect to MySQL

    #------------------------------------------------------
    def insertBooks(self, title, page, bookQuote):
        # connect to MySQL

        # insert data

    #------------------------------------------------------
    def insertBooksExample(self):
        # connect to MySQL

        # insert hard-coded data

    #------------------------------------------------------
    def showBooks(self):
        # connect to MySQL

    #------------------------------------------------------
    def showColumns(self):
        # connect to MySQL

    #------------------------------------------------------
    def showData(self):
        # connect to MySQL

#------------------------------------------------------
if __name__ == '__main__': 

    # Create class instance
    mySQL = MySQL()
```

运行上述代码会在我们创建的数据库中创建以下表和数据。

![如何做…](graphics/B04829_07_13.jpg)

## 工作原理

我们已经创建了一个MySQL数据库，建立了与之的连接，然后创建了两个表，用于存储喜爱的书籍或期刊引用的数据。

我们在两个表之间分配数据，因为引用往往相当大，而书名和书页码非常短。通过这样做，我们可以提高数据库的效率。

### 注意

在SQL数据库语言中，将数据分隔到单独的表中称为规范化。

# 使用SQL UPDATE命令

这个配方将使用前一个配方中的代码，对其进行更详细的解释，然后扩展代码以更新我们的数据。

为了更新我们之前插入到MySQL数据库表中的数据，我们使用SQL `UPDATE`命令。

## 准备工作

这个配方是基于前一个配方的，所以请阅读和研究前一个配方，以便理解本配方中修改现有数据的编码。

## 如何做…

首先，我们将通过运行以下Python到MySQL命令来显示要修改的数据：

```py
import mysql.connector as mysql
import GuiDBConfig as guiConf

class MySQL():
    # class variable
    GUIDB  = 'GuiDB'
    #------------------------------------------------------
    def showData(self):
        # connect to MySQL
        conn, cursor = self.connect()   

        self.useGuiDB(cursor)      

        # execute command
        cursor.execute("SELECT * FROM books")
        print(cursor.fetchall())

        cursor.execute("SELECT * FROM quotations")
        print(cursor.fetchall())

        # close cursor and connection
        self.close(cursor, conn)
#==========================================================
if __name__ == '__main__': 
    # Create class instance
    mySQL = MySQL()
    mySQL.showData()
```

运行代码会产生以下结果：

![如何做…](graphics/B04829_07_14.jpg)

也许我们不同意“四人帮”的观点，所以让我们修改他们著名的编程引语。

### 注意

四人帮是创作了世界著名书籍《设计模式》的四位作者，这本书对整个软件行业产生了深远影响，使我们认识到、思考并使用软件设计模式进行编码。

我们将通过更新我们最喜爱的引语数据库来实现这一点。

首先，我们通过搜索书名来检索主键值，然后将该值传递到我们对引语的搜索中。

```py
    #------------------------------------------------------
    def updateGOF(self):
        # connect to MySQL
        conn, cursor = self.connect()   

        self.useGuiDB(cursor)      

        # execute command
        cursor.execute("SELECT Book_ID FROM books WHERE Book_Title = 'Design Patterns'")
        primKey = cursor.fetchall()[0][0]
        print(primKey)

        cursor.execute("SELECT * FROM quotations WHERE Books_Book_ID = (%s)", (primKey,))
        print(cursor.fetchall())

        # close cursor and connection
        self.close(cursor, conn) 
#==========================================================
if __name__ == '__main__': 
    # Create class instance
    mySQL = MySQL()
    mySQL.updateGOF()
```

这给我们带来了以下结果：

![如何做…](graphics/B04829_07_15.jpg)

现在我们知道了引语的主键，我们可以通过执行以下命令来更新引语。

```py
    #------------------------------------------------------
    def updateGOF(self):
        # connect to MySQL
        conn, cursor = self.connect()   

        self.useGuiDB(cursor)      

        # execute command
        cursor.execute("SELECT Book_ID FROM books WHERE Book_Title = 'Design Patterns'")
        primKey = cursor.fetchall()[0][0]
        print(primKey)

        cursor.execute("SELECT * FROM quotations WHERE Books_Book_ID = (%s)", (primKey,))
        print(cursor.fetchall())

        cursor.execute("UPDATE quotations SET Quotation = (%s) WHERE Books_Book_ID = (%s)", \
                       ("Pythonic Duck Typing: If it walks like a duck and talks like a duck it probably is a duck...", primKey))

        # commit transaction
        conn.commit ()

        cursor.execute("SELECT * FROM quotations WHERE Books_Book_ID = (%s)", (primKey,))
        print(cursor.fetchall())

        # close cursor and connection
        self.close(cursor, conn)
#==========================================================
if __name__ == '__main__': 
    # Create class instance
    mySQL = MySQL()
    #------------------------
    mySQL.updateGOF()
    book, quote = mySQL.showData()    
    print(book, quote)
```

通过运行上述代码，我们使这个经典的编程更加Pythonic。

如下截图所示，在运行上述代码之前，我们的`Book_ID 1`标题通过主外键关系与引语表的`Books_Book_ID`列相关联。

这是《设计模式》书中的原始引语。

然后，我们通过SQL `UPDATE`命令更新了与该ID相关的引语。

ID都没有改变，但现在与`Book_ID 1`相关联的引语已经改变，如下所示在第二个MySQL客户端窗口中。

![如何做…](graphics/B04829_07_16.jpg)

## 工作原理…

在这个配方中，我们从数据库和之前配方中创建的数据库表中检索现有数据。我们向表中插入数据，并使用SQL `UPDATE`命令更新我们的数据。

# 使用SQL DELETE命令

在这个配方中，我们将使用SQL `DELETE`命令来删除我们在前面配方中创建的数据。

虽然删除数据乍一看似乎很简单，但一旦我们在生产中拥有一个相当大的数据库设计，事情可能就不那么容易了。

因为我们通过主外键关系设计了GUI数据库，当我们删除某些数据时，不会出现孤立记录，因为这种数据库设计会处理级联删除。

## 准备工作

这个配方使用了MySQL数据库、表以及本章前面配方中插入到这些表中的数据。为了展示如何创建孤立记录，我们将不得不改变其中一个数据库表的设计。

## 如何做…

我们通过只使用两个数据库表来保持我们的数据库设计简单。

虽然在删除数据时这样做是有效的，但总会有可能出现孤立记录。这意味着我们在一个表中删除数据，但在另一个SQL表中却没有删除相关数据。

如果我们创建`quotations`表时没有与`books`表建立外键关系，就可能出现孤立记录。

```py
        # create second Table inside DB -- 
        # No FOREIGN KEY relation to Books Table
        cursor.execute("CREATE TABLE Quotations ( \
                Quote_ID INT AUTO_INCREMENT,      \
                Quotation VARCHAR(250),           \
                Books_Book_ID INT,                \
                PRIMARY KEY (Quote_ID)            \
            ) ENGINE=InnoDB")  
```

在向`books`和`quotations`表中插入数据后，如果我们执行与之前相同的`delete`语句，我们只会删除`Book_ID 1`的书籍，而与之相关的引语`Books_Book_ID 1`则会被留下。

这是一个孤立的记录。不再存在`Book_ID`为`1`的书籍记录。

![如何做…](graphics/B04829_07_17.jpg)

这种情况可能会造成混乱，我们可以通过使用级联删除来避免这种情况。

我们在创建表时通过添加某些数据库约束来实现这一点。在之前的示例中，当我们创建包含引用的表时，我们使用外键约束创建了我们的“引用”表，明确引用了书籍表的主键，将两者联系起来。

```py
        # create second Table inside DB
        cursor.execute("CREATE TABLE Quotations ( \
                Quote_ID INT AUTO_INCREMENT,      \
                Quotation VARCHAR(250),           \
                Books_Book_ID INT,                \
                PRIMARY KEY (Quote_ID),           \
                FOREIGN KEY (Books_Book_ID)       \
                    REFERENCES Books(Book_ID)     \
                    ON DELETE CASCADE             \
            ) ENGINE=InnoDB")  
```

“外键”关系包括`ON DELETE CASCADE`属性，这基本上告诉我们的MySQL服务器，在删除与这些外键相关的记录时，删除这个表中的相关记录。

### 注意

在创建表时，如果不指定`ON DELETE CASCADE`属性，我们既不能删除也不能更新我们的数据，因为`UPDATE`是`DELETE`后跟`INSERT`。

由于这种设计，不会留下孤立的记录，这正是我们想要的。

### 注意

在MySQL中，我们必须指定`ENGINE=InnoDB`才能使用外键。

让我们显示我们数据库中的数据。

```py
#==========================================================
if __name__ == '__main__': 
    # Create class instance
    mySQL = MySQL()
      mySQL.showData()
```

这显示了我们数据库表中的以下数据：

![操作方法…](graphics/B04829_07_18.jpg)

这显示了我们有两条通过主键到外键关系相关的记录。

当我们现在删除“书籍”表中的记录时，我们期望“引用”表中的相关记录也将通过级联删除被删除。

让我们尝试通过在Python中执行以下SQL命令来执行此操作：

```py
import mysql.connector as mysql
import GuiDBConfig as guiConf

class MySQL():
    #------------------------------------------------------
    def deleteRecord(self):
        # connect to MySQL
        conn, cursor = self.connect()   

        self.useGuiDB(cursor)      

        # execute command
        cursor.execute("SELECT Book_ID FROM books WHERE Book_Title = 'Design Patterns'")
        primKey = cursor.fetchall()[0][0]
        # print(primKey)

        cursor.execute("DELETE FROM books WHERE Book_ID = (%s)", (primKey,))

        # commit transaction
        conn.commit ()

        # close cursor and connection
        self.close(cursor, conn)    
#==========================================================
if __name__ == '__main__': 
    # Create class instance
    mySQL = MySQL()
    #------------------------
    mySQL.deleteRecord()
    mySQL.showData()   
```

在执行前面的删除记录命令后，我们得到了以下新结果：

![操作方法…](graphics/B04829_07_19.jpg)

### 注意

著名的“设计模式”已经从我们喜爱的引用数据库中消失了…

## 工作原理…

通过通过主键到外键关系进行级联删除，通过设计我们的数据库，我们在这个示例中触发了级联删除。

这可以保持我们的数据完整和完整。

### 注意

在这个示例和示例代码中，我们有时引用相同的表名，有时以大写字母开头，有时全部使用小写字母。

这适用于MySQL的Windows默认安装，但在Linux上可能不起作用，除非我们更改设置。

这是官方MySQL文档的链接：[http://dev.mysql.com/doc/refman/5.0/en/identifier-case-sensitivity.html](http://dev.mysql.com/doc/refman/5.0/en/identifier-case-sensitivity.html)

在下一个示例中，我们将使用我们的Python GUI中的`MySQL.py`模块的代码。

# 从我们的MySQL数据库中存储和检索数据

我们将使用我们的Python GUI将数据插入到我们的MySQL数据库表中。我们已经重构了之前示例中构建的GUI，以便连接和使用数据库。

我们将使用两个文本框输入小部件，可以在其中输入书名或期刊标题和页码。我们还将使用一个ScrolledText小部件来输入我们喜爱的书籍引用，然后将其存储在我们的MySQL数据库中。

## 准备工作

这个示例将建立在我们之前创建的MySQL数据库和表的基础上。

## 操作方法…

我们将使用我们的Python GUI来插入、检索和修改我们喜爱的引用。我们已经重构了我们GUI中的MySQL选项卡，为此做好了准备。

![操作方法…](graphics/B04829_07_20.jpg)

为了让按钮起作用，我们将把它们连接到回调函数，就像我们在之前的示例中所做的那样。

我们将在按钮下方的ScrolledText小部件中显示数据。

为了做到这一点，我们将像之前一样导入`MySQL.py`模块。所有与我们的MySQL服务器实例和数据库通信的代码都驻留在这个模块中，这是一种封装代码的形式，符合面向对象编程的精神。

我们将“插入引用”按钮连接到以下回调函数。

```py
        # Adding a Button
        self.action = ttk.Button(self.mySQL, text="Insert Quote", command=self.insertQuote)   
        self.action.grid(column=2, row=1)
    # Button callback
    def insertQuote(self):
        title = self.bookTitle.get()
        page = self.pageNumber.get()
        quote = self.quote.get(1.0, tk.END)
        print(title)
        print(quote)
        self.mySQL.insertBooks(title, page, quote)  
```

当我们现在运行我们的代码时，我们可以从我们的Python GUI中将数据插入到我们的MySQL数据库中。

![操作方法…](graphics/B04829_07_21.jpg)

输入书名和书页以及书籍或电影中的引用后，通过单击“插入引用”按钮将数据插入到我们的数据库中。

我们当前的设计允许标题、页面和引语。我们还可以插入我们最喜欢的电影引语。虽然电影没有页面，但我们可以使用页面列来插入引语在电影中发生的大致时间。

接下来，我们可以通过发出与之前使用的相同命令来验证所有这些数据是否已经进入了我们的数据库表。

![如何做...](graphics/B04829_07_22.jpg)

在插入数据之后，我们可以通过单击**获取引语**按钮来验证它是否已经进入了我们的两个MySQL表中，然后显示我们插入到两个MySQL数据库表中的数据，如上所示。

单击**获取引语**按钮会调用与按钮单击事件关联的回调方法。这给了我们在我们的ScrolledText小部件中显示的数据。

```py
# Adding a Button
        self.action1 = ttk.Button(self.mySQL, text="Get Quotes", command=self.getQuote)   
        self.action1.grid(column=2, row=2)
    # Button callback
    def getQuote(self):
        allBooks = self.mySQL.showBooks()  
        print(allBooks)
        self.quote.insert(tk.INSERT, allBooks)
```

我们使用`self.mySQL`类实例变量来调用`showBooks()`方法，这是我们导入的MySQL类的一部分。

```py
from B04829_Ch07_MySQL import MySQL
class OOP():
    def __init__(self):
        # create MySQL instance
        self.mySQL = MySQL()

class MySQL():
    #------------------------------------------------------
    def showBooks(self):
        # connect to MySQL
        conn, cursor = self.connect()    

        self.useGuiDB(cursor)    

        # print results
        cursor.execute("SELECT * FROM Books")
        allBooks = cursor.fetchall()
        print(allBooks)

        # close cursor and connection
        self.close(cursor, conn)   

        return allBooks  
```

## 它是如何工作的...

在这个示例中，我们导入了包含所有连接到我们的MySQL数据库并知道如何插入、更新、删除和显示数据的编码逻辑的Python模块。

我们现在已经将我们的Python GUI连接到了这个SQL逻辑。
