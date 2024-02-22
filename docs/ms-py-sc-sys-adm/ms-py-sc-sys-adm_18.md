# 第十八章：MySQL 和 SQLite 数据库管理

在本章中，您将学习有关 MySQL 和 SQLite 数据库管理的知识。您将学习如何安装 MySQL 和 SQLite。您还将学习如何创建用户，授予权限，创建数据库，创建表，将数据插入表中，并查看表中的所有记录，特定记录，并更新和删除数据。

在本章中，您将学习以下内容：

+   MySQL 数据库管理

+   SQLite 数据库管理

# MySQL 数据库管理

本节将介绍使用 Python 进行 MySQL 数据库管理。您已经知道 Python 有各种模块用于`mysql`数据库管理。因此，我们将在这里学习有关 MySQLdb 模块的知识。`mysqldb`模块是 MySQL 数据库服务器的接口，用于提供 Python 数据库 API。

让我们学习如何安装 MySQL 和 Python 的`mysqldb`包。为此，请在终端中运行以下命令：

```py
$ sudo apt install mysql-server
```

此命令安装 MySQL 服务器和各种其他软件包。在安装软件包时，我们被提示为 MySQL root 帐户输入密码：

+   以下代码用于检查是否安装了`mysqldb`包：

```py
$ apt-cache search MySQLdb
```

+   以下是用于安装 MySQL 的 Python 接口：

```py
$ sudo apt-get install python3-mysqldb
```

+   现在，我们将检查`mysql`是否安装正确。为此，在终端中运行以下命令：

```py
student@ubuntu:~$ sudo mysql -u root -p 
```

一旦命令运行，您将获得以下输出：

```py
Enter password: Welcome to the MySQL monitor.  Commands end with ; or \g. Your MySQL connection id is 10 Server version: 5.7.24-0ubuntu0.18.04.1 (Ubuntu)
Copyright (c) 2000, 2018, Oracle and/or its affiliates. All rights reserved.
Oracle is a registered trademark of Oracle Corporation and/or its affiliates. Other names may be trademarks of their respective owners. Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.
mysql>
```

通过运行`sudo mysql -u root -p`，您将获得`mysql`控制台。有一些命令用于列出数据库和表，并使用数据库来存储我们的工作。我们将逐一看到它们：

+   这是用于列出所有数据库的：

```py
show databases;
```

+   这是用于使用数据库的：

```py
use database_name;
```

每当我们退出 MySQL 控制台并在一段时间后再次登录时，我们必须使用`use database_name;`语句。使用此命令的目的是我们的工作将保存在我们的数据库中。我们可以通过以下示例详细了解这一点：

+   以下代码用于列出所有表：

```py
show tables;
```

这些是我们用于列出数据库，使用数据库和列出表的命令。

现在，我们将使用`mysql`控制台中的 create database 语句创建数据库。现在，使用`mysql -u root -p`打开`mysql`控制台，然后输入您在安装时输入的密码，然后按*Enter*。接下来，创建您的数据库。在本节中，我们将创建一个名为`test`的数据库，并在本节中将使用该数据库：

```py
student@ubuntu:~/work/mysql_testing$ sudo mysql -u root -p  Output: Enter password: Welcome to the MySQL monitor.  Commands end with ; or \g. Your MySQL connection id is 16 Server version: 5.7.24-0ubuntu0.18.04.1 (Ubuntu)
Copyright (c) 2000, 2018, Oracle and/or its affiliates. All rights reserved.
Oracle is a registered trademark of Oracle Corporation and/or its affiliates. Other names may be trademarks of their respective owners. Type 'help;' or '\h' for help. Type '\c' to clear the current input statement. mysql> mysql> show databases; +--------------------+ | Database           | +--------------------+ | information_schema | | mysql              | | performance_schema | | sys                | +--------------------+ 4 rows in set (0.10 sec)
mysql> create database test; Query OK, 1 row affected (0.00 sec)
mysql> show databases; +--------------------+ | Database           | +--------------------+ | information_schema | | mysql              | | performance_schema | | sys                | | test               | +--------------------+ 5 rows in set (0.00 sec)
mysql> use test; Database changed mysql>
```

首先，我们使用 show databases 列出了所有数据库。接下来，我们使用 create `database`语句创建了我们的数据库 test。然后，我们再次执行 show databases 以查找我们的数据库是否已创建。我们的数据库现在已创建。接下来，我们使用该数据库来存储我们正在进行的工作。

现在，我们将创建一个用户并授予该用户权限。运行以下命令：

```py
mysql> create user 'test_user'@'localhost' identified by 'test123'; Query OK, 0 rows affected (0.06 sec) mysql> grant all on test.* to 'test_user'@'localhost'; Query OK, 0 rows affected (0.02 sec)
mysql>
```

我们创建了一个名为`test_user`的用户；该用户的密码为`test123`。接下来，我们授予我们的`test_user`用户所有权限。现在，通过运行`quit;`或`exit;`命令退出`mysql`控制台。

现在，我们将看一些示例，获取数据库版本，创建表，将一些数据插入表中，更新数据和删除数据。

# 获取数据库版本

首先，我们将看一个获取数据库版本的示例。为此，我们将创建一个`get_database_version.py`脚本，并在其中编写以下内容：

```py
import MySQLdb as mdb import sys  con_obj = mdb.connect('localhost', 'test_user', 'test123', 'test') cur_obj = con_obj.cursor() cur_obj.execute("SELECT VERSION()") version = cur_obj.fetchone() print ("Database version: %s " % version) con_obj.close()
```

在运行此脚本之前，非常重要遵循先前的步骤；不应跳过它们。

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work/mysql_testing$ python3 get_database_version.py Output: Database version: 5.7.24-0ubuntu0.18.04.1
```

在上面的示例中，我们得到了数据库版本。首先，我们导入了 MySQLdb 模块。然后我们编写了连接字符串。在连接字符串中，我们提到了我们的用户名、密码和数据库名称。接下来，我们创建了一个游标对象，用于执行 SQL 查询。在`execute()`中，我们传递了一个 SQL 查询。`fetchone()`检索查询结果的下一行。接下来，我们打印了结果。`close()`方法关闭了数据库连接。

# 创建表和插入数据

现在，我们将创建一个表，并向其中插入一些数据。为此，创建一个`create_insert_data.py`脚本，并在其中写入以下内容：

```py
import MySQLdb as mdb con_obj = mdb.connect('localhost', 'test_user', 'test123', 'test') with con_obj:
 cur_obj = con_obj.cursor() cur_obj.execute("DROP TABLE IF EXISTS books") cur_obj.execute("CREATE TABLE books(Id INT PRIMARY KEY AUTO_INCREMENT, Name VARCHAR(100))") cur_obj.execute("INSERT INTO books(Name) VALUES('Harry Potter')") cur_obj.execute("INSERT INTO books(Name) VALUES('Lord of the rings')") cur_obj.execute("INSERT INTO books(Name) VALUES('Murder on the Orient Express')") cur_obj.execute("INSERT INTO books(Name) VALUES('The adventures of Sherlock Holmes')") cur_obj.execute("INSERT INTO books(Name) VALUES('Death on the Nile')") print("Table Created !!") print("Data inserted Successfully !!")
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work/mysql_testing$ python3 create_insert_data.py Output: Table Created !! Data inserted Successfully !!
```

要检查您的表是否成功创建，请打开您的`mysql`控制台并运行以下命令：

```py
student@ubuntu:~/work/mysql_testing$ sudo mysql -u root -p Enter password: Welcome to the MySQL monitor.  Commands end with ; or \g. Your MySQL connection id is 6 Server version: 5.7.24-0ubuntu0.18.04.1 (Ubuntu)
Copyright (c) 2000, 2018, Oracle and/or its affiliates. All rights reserved.
Oracle is a registered trademark of Oracle Corporation and/or its affiliates. Other names may be trademarks of their respective owners.
Type 'help;' or '\h' for help. Type '\c' to clear the current input statement. mysql> mysql> mysql> use test; Reading table information for completion of table and column names You can turn off this feature to get a quicker startup with -A Database changed mysql> show tables; +----------------+ | Tables_in_test | +----------------+ | books          | +----------------+ 1 row in set (0.00 sec)
```

您可以看到您的 books 表已创建。

# 检索数据

要从表中检索数据，我们使用`select`语句。现在，我们将从我们的 books 表中检索数据。为此，创建一个`retrieve_data.py`脚本，并在其中写入以下内容：

```py
import MySQLdb as mdb con_obj = mdb.connect('localhost', 'test_user', 'test123', 'test') with con_obj:
 cur_obj = con_obj.cursor() cur_obj.execute("SELECT * FROM books") records = cur_obj.fetchall() for r in records: print(r)
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work/mysql_testing$ python3 retrieve_data.py Output: (1, 'Harry Potter') (2, 'Lord of the rings') (3, 'Murder on the Orient Express') (4, 'The adventures of Sherlock Holmes') (5, 'Death on the Nile')
```

在上面的示例中，我们从表中检索了数据。我们使用了 MySQLdb 模块。我们编写了一个连接字符串并创建了一个游标对象来执行 SQL 查询。在`execute()`中，我们编写了一个 SQL`select`语句。最后，我们打印了记录。

# 更新数据

现在，如果我们想对记录进行一些更改，我们可以使用 SQL`update`语句。我们将看一个`update`语句的示例。为此，创建一个`update_data.py`脚本，并在其中写入以下内容：

```py
import MySQLdb as mdb con_obj = mdb.connect('localhost', 'test_user', 'test123', 'test') cur_obj = con_obj.cursor() cur_obj.execute("UPDATE books SET Name = 'Fantastic Beasts' WHERE Id = 1")try:
 con_obj.commit() except:
 con_obj.rollback()
```

按以下方式运行脚本：

```py
student@ubuntu:~/work/mysql_testing$ python3 update_data.py
```

现在，要检查您的记录是否已更新，请按以下方式运行`retrieve_data.py`：

```py
student@ubuntu:~/work/mysql_testing$ python3 retrieve_data.py Output: (1, 'Fantastic Beasts') (2, 'Lord of the rings') (3, 'Murder on the Orient Express') (4, 'The adventures of Sherlock Holmes') (5, 'Death on the Nile')
```

您可以看到 ID 为`1`的数据已更新。在上面的示例中，在`execute()`中，我们编写了一个`update`语句，将更新 ID 为`1`的数据。

# 删除数据

要从表中删除特定记录，请使用`delete`语句。我们将看一个删除数据的示例。创建一个`delete_data.py`脚本，并在其中写入以下内容：

```py
import MySQLdb as mdb con_obj = mdb.connect('localhost', 'test_user', 'test123', 'test') cur_obj = con_obj.cursor() cur_obj.execute("DELETE FROM books WHERE Id = 5"); try:
 con_obj.commit() except:
 con_obj.rollback() 
```

按以下方式运行脚本：

```py
student@ubuntu:~/work/mysql_testing$ python3 delete_data.py
```

现在，要检查您的记录是否已删除，请按以下方式运行`retrieve_data.py`脚本：

```py
student@ubuntu:~/work/mysql_testing$ python3 retrieve_data.py Output: (1, 'Fantastic Beasts') (2, 'Lord of the rings') (3, 'Murder on the Orient Express') (4, 'The adventures of Sherlock Holmes')
```

您可以看到，您的 ID 为`5`的记录已被删除。在上面的示例中，我们使用了`delete`语句来删除特定记录。在这里，我们删除了 ID 为`5`的记录。您还可以根据自己选择的任何字段名删除记录。

# SQLite 数据库管理

在本节中，我们将学习如何安装和使用 SQLite。Python 有`sqlite3`模块来执行 SQLite 数据库任务。SQLite 是一个无服务器、零配置、事务性 SQL 数据库引擎。SQLite 非常快速和轻量级。整个数据库存储在单个磁盘文件中。

现在，我们将首先安装 SQLite。在终端中运行以下命令：

```py
$ sudo apt install sqlite3
```

在本节中，我们将学习以下操作：创建数据库、创建表、向表中插入数据、检索数据，以及从表中更新和删除数据。我们将逐个查看每个操作。

现在，首先，我们将看如何在 SQLite 中创建数据库。要创建数据库，您只需在终端中输入以下命令：

```py
$ sqlite3 test.db
```

运行此命令后，您将在终端中打开`sqlite`控制台，如下所示：

```py
student@ubuntu:~$ sqlite3 test.db SQLite version 3.22.0 2018-01-22 18:45:57 Enter ".help" for usage hints. sqlite>
```

您的数据库已通过简单运行`sqlite3 test.db`创建。

# 连接到数据库

现在，我们将看到如何连接到数据库。为此，我们将创建一个脚本。Python 已经在标准库中包含了一个`sqlite3`模块。我们只需要在使用 SQLite 时导入它。创建一个`connect_database.py`脚本，并在其中写入以下内容：

```py
import sqlite3 con_obj = sqlite3.connect('test.db') print ("Database connected successfully !!")
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work $ python3 connect_database.py Output: Database connected successfully !!
```

在前面的例子中，我们导入了`sqlite3`模块来执行功能。现在，检查您的目录，您将在目录中找到创建的`test.db`文件。

# 创建表

现在，我们将在我们的数据库中创建一个表。为此，我们将创建一个`create_table.py`脚本，并在其中写入以下内容：

```py
import sqlite3 con_obj = sqlite3.connect("test.db") with con_obj:
 cur_obj = con_obj.cursor() cur_obj.execute("""CREATE TABLE books(title text, author text)""")  print ("Table created")
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work $ python3 create_table.py Output: Table created
```

在前面的例子中，我们使用`CREATE TABLE`语句创建了一个名为 books 的表。首先，我们使用`test.db`建立了与数据库的连接。接下来，我们创建了一个游标对象，用于在数据库上执行 SQL 查询。

# 插入数据

现在，我们将向我们的表中插入数据。为此，我们将创建一个`insert_data.py`脚本，并在其中写入以下内容：

```py
import sqlite3 con_obj = sqlite3.connect("test.db") with con_obj:
 cur_obj = con_obj.cursor() cur_obj.execute("INSERT INTO books VALUES ('Pride and Prejudice', 'Jane Austen')") cur_obj.execute("INSERT INTO books VALUES ('Harry Potter', 'J.K Rowling')") cur_obj.execute("INSERT INTO books VALUES ('The Lord of the Rings', 'J. R. R. Tolkien')") cur_obj.execute("INSERT INTO books VALUES ('Murder on the Orient Express', 'Agatha Christie')") cur_obj.execute("INSERT INTO books VALUES ('A Study in Scarlet', 'Arthur Conan Doyle')") con_obj.commit() print("Data inserted Successfully !!")
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 insert_data.py Output: Data inserted Successfully !!
```

在前面的例子中，我们向我们的表中插入了一些数据。为此，我们在 SQL 语句中使用了`insert`。通过使用`commit()`，我们告诉数据库保存所有当前的事务。

# 检索数据

现在，我们将从表中检索数据。为此，创建一个`retrieve_data.py`脚本，并在其中写入以下内容：

```py
import sqlite3 con_obj = sqlite3.connect('test.db') cur_obj = con_obj.execute("SELECT title, author from books")
for row in cur_obj:
 print ("Title = ", row[0]) print ("Author = ", row[1], "\n") con_obj.close()
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 retrieve_data.py Output: Title =  Pride and Prejudice Author =  Jane Austen
Title =  Harry Potter Author =  J.K Rowling
Title =  The Lord of the Rings Author =  J. R. R. Tolkien
Title =  Murder on the Orient Express Author =  Agatha Christie
Title =  A Study in Scarlet Author =  Arthur Conan Doyle
```

在前面的例子中，我们导入了`sqlite3`模块。接下来，我们连接到了我们的`test.db`数据库。为了检索数据，我们使用了`select`语句。最后，我们打印了检索到的数据。

您还可以在`sqlite3`控制台中检索数据。为此，首先启动 SQLite 控制台，然后按照以下方式检索数据：

```py
student@ubuntu:~/work/sqlite3_testing$ sqlite3 test.db Output: SQLite version 3.22.0 2018-01-22 18:45:57 Enter ".help" for usage hints. sqlite> sqlite> select * from books; Pride and Prejudice|Jane Austen Harry Potter|J.K Rowling The Lord of the Rings|J. R. R. Tolkien Murder on the Orient Express|Agatha Christie A Study in Scarlet|Arthur Conan Doyle sqlite>
```

# 更新数据

我们可以使用`update`语句从我们的表中更新数据。现在，我们将看一个更新数据的例子。为此，创建一个`update_data.py`脚本，并在其中写入以下内容：

```py
import sqlite3 con_obj = sqlite3.connect("test.db") with con_obj:
            cur_obj = con_obj.cursor()
 sql = """ UPDATE books SET author = 'John Smith' WHERE author = 'J.K Rowling' """ cur_obj.execute(sql) print("Data updated Successfully !!")
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work $ python3 update_data.py Output: Data updated Successfully !!
```

现在，要检查数据是否实际上已更新，请运行`retrieve_data.py`，或者您可以转到 SQLite 控制台并运行`select * from books;`。您将得到更新后的输出如下：

```py
By running retrieve_data.py: Output: student@ubuntu:~/work$ python3 retrieve_data.py Title =  Pride and Prejudice Author =  Jane Austen
Title =  Harry Potter Author =  John Smith
Title =  The Lord of the Rings Author =  J. R. R. Tolkien
Title =  Murder on the Orient Express Author =  Agatha Christie
Title =  A Study in Scarlet Author =  Arthur Conan Doyle
Checking on SQLite console: Output: student@ubuntu:~/work$ sqlite3 test.db SQLite version 3.22.0 2018-01-22 18:45:57 Enter ".help" for usage hints. sqlite> sqlite> select * from books; Pride and Prejudice|Jane Austen Harry Potter|John Smith The Lord of the Rings|J. R. R. Tolkien Murder on the Orient Express|Agatha Christie A Study in Scarlet|Arthur Conan Doyle sqlite>
```

# 删除数据

现在，我们将看一个从表中删除数据的例子。我们将使用`delete`语句来做到这一点。创建一个`delete_data.py`脚本，并在其中写入以下内容：

```py
import sqlite3 con_obj = sqlite3.connect("test.db") with con_obj:
 cur_obj = con_obj.cursor()            sql = """
 DELETE FROM books WHERE author = 'John Smith' """ cur_obj.execute(sql) print("Data deleted successfully !!")
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work $ python3 delete_data.py Output: Data deleted successfully !!
```

在前面的例子中，我们从表中删除了一条记录。我们使用了`delete` SQL 语句。现在，要检查数据是否成功删除，请运行`retrieve_data.py`或启动 SQLite 控制台，如下所示：

```py
By running retrieve_data.py Output: student@ubuntu:~/work$ python3 retrieve_data.py Title =  Pride and Prejudice Author =  Jane Austen
Title =  The Lord of the Rings Author =  J. R. R. Tolkien
Title =  Murder on the Orient Express Author =  Agatha Christie
Title =  A Study in Scarlet Author =  Arthur Conan Doyle
```

您可以看到作者是`john smith`的记录已被删除：

```py
Checking on SQLite console: Output: student@ubuntu:~/work$ sqlite3 test.db SQLite version 3.22.0 2018-01-22 18:45:57 Enter ".help" for usage hints. sqlite> sqlite> select * from books; Pride and Prejudice|Jane Austen The Lord of the Rings|J. R. R. Tolkien Murder on the Orient Express|Agatha Christie A Study in Scarlet|Arthur Conan Doyle sqlite>
```

# 总结

在本章中，我们学习了 MySQL 以及 SQLite 数据库管理。我们创建了数据库和表。然后我们在表中插入了一些记录。使用`select`语句，我们检索了记录。我们还学习了更新和删除数据。

# 问题

1.  数据库用于什么？

1.  数据库中的 CRUD 是什么？

1.  我们可以连接远程数据库吗？如果可以，请举例说明。

1.  我们可以在 Python 代码中编写触发器和存储过程吗？

1.  什么是 DML 和 DDL 语句？

# 进一步阅读

+   使用 PyMySQL 库：[`zetcode.com/python/pymysql/`](http://zetcode.com/python/pymysql/)

+   MySQLdb，Python 连接指南：[`mysqlclient.readthedocs.io/`](https://mysqlclient.readthedocs.io/)

+   SQLite 数据库的 DB-API 2.0 接口：[`docs.python.org/3/library/sqlite3.html`](https://docs.python.org/3/library/sqlite3.html)
