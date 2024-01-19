# 与数据库交互

在之前的章节中，我们使用了许多 Python 工具和实用程序生成了多种不同的报告。在本章中，我们将利用 Python 库连接到外部数据库，并提交我们生成的数据。然后，外部应用程序可以访问这些数据以获取信息。

Python 提供了广泛的库和模块，涵盖了管理和处理流行的**数据库管理系统**（**DBMSes**），如 MySQL、PostgreSQL 和 Oracle。在本章中，我们将学习如何与 DBMS 交互，并填充我们自己的数据。

本章将涵盖以下主题：

+   在自动化服务器上安装 MySQL

+   从 Python 访问 MySQL 数据库

# 在自动化服务器上安装 MySQL

我们需要做的第一件事是设置一个数据库。在接下来的步骤中，我们将介绍如何在我们在第八章中创建的自动化服务器上安装 MySQL 数据库。基本上，您需要一个具有互联网连接的基于 Linux 的机器（CentOS 或 Ubuntu）来下载 SQL 软件包。MySQL 是一个使用关系数据库和 SQL 语言与数据交互的开源 DBMS。在 CentOS 7 中，MySQL 被另一个分支版本 MariaDB 取代；两者具有相同的源代码，但 MariaDB 中有一些增强功能。

按照以下步骤安装 MariaDB：

1.  使用`yum`软件包管理器（或`apt`，在基于 Debian 的系统中）下载`mariadb-server`软件包，如下摘录所示：

```py
yum install mariadb-server -y
```

1.  安装完成后，启动`mariadb`守护程序。此外，我们需要使用`systemd`命令在操作系统启动时启用它：

```py
systemctl enable mariadb ; systemctl start mariadb

Created symlink from /etc/systemd/system/multi-user.target.wants/mariadb.service to /usr/lib/systemd/system/mariadb.service.
```

1.  通过运行以下命令验证数据库状态，并确保输出包含`Active:active (running)`：

```py
systemctl status mariadb

● mariadb.service - MariaDB database server
 Loaded: loaded (/usr/lib/systemd/system/mariadb.service; enabled; vendor preset: disabled)
 Active: active (running) since Sat 2018-04-07 19:47:35 EET; 1min 34s ago
```

# 保护安装

安装完成后的下一个逻辑步骤是保护它。MariaDB 包括一个安全脚本，可以更改 MySQL 配置文件中的选项，比如创建用于访问数据库的 root 密码和允许远程访问。运行以下命令启动脚本：

```py
mysql_secure_installation
```

第一个提示要求您提供 root 密码。这个 root 密码不是 Linux 的 root 用户名，而是 MySQL 数据库的 root 密码；由于这是一个全新的安装，我们还没有设置它，所以我们将简单地按*Enter*进入下一步：

```py
Enter current password for root (enter for none): <PRESS_ENTER>
```

脚本将建议为 root 设置密码。我们将通过按`Y`并输入新密码来接受建议：

```py
Set root password? [Y/n] Y
New password:EnterpriseAutomation
Re-enter new password:EnterpriseAutomation
Password updated successfully!
Reloading privilege tables..
 ... Success! 
```

以下提示将建议删除匿名用户对数据库的管理和访问权限，这是强烈建议的：

```py
Remove anonymous users? [Y/n] y
 ... Success!
```

您可以从远程机器向托管在自动化服务器上的数据库运行 SQL 命令；这需要您为 root 用户授予特殊权限，以便他们可以远程访问数据库：

```py
Disallow root login remotely? [Y/n] n
 ... skipping.
```

最后，我们将删除任何人都可以访问的测试数据库，并重新加载权限表，以确保所有更改立即生效：

```py
Remove test database and access to it? [Y/n] y
 - Dropping test database...
 ... Success!
 - Removing privileges on test database...
 ... Success!

Reload privilege tables now? [Y/n] y
 ... Success!

Cleaning up...

All done!  If you've completed all of the above steps, your MariaDB
installation should now be secure.

Thanks for using MariaDB!
```

我们已经完成了安装的保护；现在，让我们验证它。

# 验证数据库安装

在 MySQL 安装后的第一步是验证它。我们需要验证`mysqld`守护程序是否已启动并正在侦听端口`3306`。我们将通过运行`netstat`命令和在侦听端口上使用`grep`来做到这一点：

```py
netstat -antup | grep -i 3306
tcp   0   0 0.0.0.0:3306      0.0.0.0:*         LISTEN      3094/mysqld
```

这意味着`mysqld`服务可以接受来自端口`3306`上的任何 IP 的传入连接。

如果您的机器上运行着`iptables`，您需要向`INPUT`链添加一个规则，以允许远程主机连接到 MySQL 数据库。还要验证`SELINUX`是否具有适当的策略。

第二次验证是通过使用`mysqladmin`实用程序连接到数据库。这个工具包含在 MySQL 客户端中，允许您在 MySQL 数据库上远程（或本地）执行命令：

```py
mysqladmin -u root -p ping
Enter password:EnterpriseAutomation 
mysqld is alive
```

| **切换名称** | **含义** |
| --- | --- |
| `-u` | 指定用户名。 |
| `-p` | 使 MySQL 提示您输入用户名的密码。 |
| `ping` | 用于验证 MySQL 数据库是否存活的操作名称。 |

输出表明 MySQL 安装已成功完成，我们准备进行下一步。

# 从 Python 访问 MySQL 数据库

Python 开发人员创建了`MySQLdb`模块，该模块提供了一个工具，可以从 Python 脚本中与数据库进行交互和管理。可以使用 Python 的`pip`或操作系统包管理器（如`yum`或`apt`）安装此模块。

要安装该软件包，请使用以下命令：

```py
yum install MySQL-python
```

按以下方式验证安装：

```py
[root@AutomationServer ~]# python
Python 2.7.5 (default, Aug  4 2017, 00:39:18) 
[GCC 4.8.5 20150623 (Red Hat 4.8.5-16)] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import MySQLdb
>>> 
```

由于模块已经成功导入，我们知道 Python 模块已成功安装。

现在，通过控制台访问数据库，并创建一个名为`TestingPython`的简单数据库，其中包含一个表。然后我们将从 Python 连接到它：

```py
[root@AutomationServer ~]# mysql -u root -p
Enter password: EnterpriseAutomation
Welcome to the MariaDB monitor.  Commands end with ; or \g.
Your MariaDB connection id is 12
Server version: 5.5.56-MariaDB MariaDB Server

Copyright (c) 2000, 2017, Oracle, MariaDB Corporation Ab and others.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

MariaDB [(none)]> CREATE DATABASE TestingPython;
Query OK, 1 row affected (0.00 sec)
```

在前述声明中，我们使用 MySQL 实用程序连接到数据库，然后使用 SQL 的`CREATE`命令创建一个空的新数据库。

您可以使用以下命令验证新创建的数据库：

```py
MariaDB [(none)]> SHOW DATABASES;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| TestingPython      |
| mysql              |
```

```py
| performance_schema |
+--------------------+
4 rows in set (0.00 sec)
```

在 SQL 命令中不一定要使用大写字母；但是，这是最佳实践，以便将它们与变量和其他操作区分开来。

我们需要切换到新的数据库：

```py
MariaDB [(none)]> use TestingPython;
Database changed
```

现在，执行以下命令在数据库中创建一个新表：

```py
MariaDB [TestingPython]> CREATE TABLE TestTable (id INT PRIMARY KEY, fName VARCHAR(30), lname VARCHAR(20), Title VARCHAR(10));
Query OK, 0 rows affected (0.00 sec)
```

在创建表时，应指定列类型。例如，`fname`是一个最大长度为 30 个字符的字符串，而`id`是一个整数。

验证表的创建如下：

```py
MariaDB [TestingPython]> SHOW TABLES;
+-------------------------+
| Tables_in_TestingPython |
+-------------------------+
| TestTable               |
+-------------------------+
1 row in set (0.00 sec)

MariaDB [TestingPython]> describe TestTable;
+-------+-------------+------+-----+---------+-------+
| Field | Type        | Null | Key | Default | Extra |
+-------+-------------+------+-----+---------+-------+
| id    | int(11)     | NO   | PRI | NULL    |       |
| fName | varchar(30) | YES  |     | NULL    |       |
| lname | varchar(20) | YES  |     | NULL    |       |
| Title | varchar(10) | YES  |     | NULL    |       |
+-------+-------------+------+-----+---------+-------+
4 rows in set (0.00 sec)

```

# 查询数据库

此时，我们的数据库已准备好接受一些 Python 脚本。让我们创建一个新的 Python 文件，并提供数据库参数：

```py
import MySQLdb
SQL_IP ="10.10.10.130" SQL_USERNAME="root" SQL_PASSWORD="EnterpriseAutomation" SQL_DB="TestingPython"   sql_connection = MySQLdb.connect(SQL_IP,SQL_USERNAME,SQL_PASSWORD,SQL_DB) print sql_connection
```

提供的参数（`SQL_IP`、`SQL_USERNAME`、`SQL_PASSWORD`和`SQL_DB`）是建立连接并对端口`3306`上的数据库进行身份验证所需的。

以下表格列出了参数及其含义：

| **参数** | **含义** |
| --- | --- |
| `host` | 具有`mysql`安装的服务器 IP 地址。 |
| `user` | 具有对连接数据库的管理权限的用户名。 |
| `passwd` | 使用`mysql_secure_installation`脚本创建的密码。 |
| `db` | 数据库名称。 |

输出将如下所示：

```py
<_mysql.connection open to '10.10.10.130' at 1cfd430>
```

返回的对象表明已成功打开到数据库的连接。让我们使用此对象创建用于执行实际命令的 SQL 游标：

```py
cursor = sql_connection.cursor() cursor.execute("show tables")
```

您可以有许多与单个连接关联的游标，对一个游标的任何更改都会立即报告给其他游标，因为您已经打开了相同的连接。

游标有两个主要方法：`execute()`和`fetch*()`。

`execute()`方法用于向数据库发送命令并返回查询结果，而`fetch*()`方法有三种不同的用法：

| **方法名称** | **描述** |
| --- | --- |
| `fetchone()` | 从输出中获取一个记录，而不管返回的行数。 |
| `fetchmany(num)` | 返回方法内指定的记录数。 |
| `fetchall()` | 返回所有记录。 |

由于`fetchall()`是一个通用方法，可以获取一个记录或所有记录，我们将使用它：

```py
output = cursor.fetchall()
print(output) # python mysql_simple.py
(('TestTable',),)
```

# 向数据库中插入记录

`MySQLdb`模块允许我们使用相同的游标操作将记录插入到数据库中。请记住，`execute()`方法可用于插入和查询。毫不犹豫，我们将稍微修改我们的脚本，并提供以下`insert`命令：

```py
#!/usr/bin/python __author__ = "Bassim Aly" __EMAIL__ = "basim.alyy@gmail.com"   import MySQLdb

SQL_IP ="10.10.10.130" SQL_USERNAME="root" SQL_PASSWORD="EnterpriseAutomation" SQL_DB="TestingPython"   sql_connection = MySQLdb.connect(SQL_IP,SQL_USERNAME,SQL_PASSWORD,SQL_DB)   employee1 = {
  "id": 1,
  "fname": "Bassim",
  "lname": "Aly",
  "Title": "NW_ENG" }   employee2 = {
  "id": 2,
  "fname": "Ahmed",
  "lname": "Hany",
  "Title": "DEVELOPER" }   employee3 = {
  "id": 3,
  "fname": "Sara",
  "lname": "Mosaad",
  "Title": "QA_ENG" }   employee4 = {
  "id": 4,
  "fname": "Aly",
  "lname": "Mohamed",
  "Title": "PILOT" }   employees = [employee1,employee2,employee3,employee4]   cursor = sql_connection.cursor()   for record in employees:
  SQL_COMMAND = """INSERT INTO TestTable(id,fname,lname,Title) VALUES ({0},'{1}','{2}','{3}')""".format(record['id'],record['fname'],record['lname'],record['Title'])    print SQL_COMMAND
    try:
  cursor.execute(SQL_COMMAND)
  sql_connection.commit()
  except:
  sql_connection.rollback()   sql_connection.close()
```

在前面的例子中，以下内容适用：

+   我们将四个员工记录定义为字典。每个记录都有`id`、`fname`、`lname`和`title`，并具有不同的值。

+   然后，我们使用`employees`对它们进行分组，这是一个`list`类型的变量。

+   创建一个`for`循环来迭代`employees`列表，在循环内部，我们格式化了`insert` SQL 命令，并使用`execute()`方法将数据推送到 SQL 数据库。请注意，在`execute`函数内部不需要在命令后添加分号(`;`)，因为它会自动添加。

+   在每次成功执行 SQL 命令后，将使用`commit()`操作来强制数据库引擎提交数据；否则，连接将被回滚。

+   最后，使用`close()`函数来终止已建立的 SQL 连接。

关闭数据库连接意味着所有游标都被发送到 Python 垃圾收集器，并且将无法使用。还要注意，当关闭连接而不提交更改时，它会立即使数据库引擎回滚所有事务。

脚本的输出如下：

```py
# python mysql_insert.py
INSERT INTO TestTable(id,fname,lname,Title) VALUES (1,'Bassim','Aly','NW_ENG')
INSERT INTO TestTable(id,fname,lname,Title) VALUES (2,'Ahmed','Hany','DEVELOPER')
INSERT INTO TestTable(id,fname,lname,Title) VALUES (3,'Sara','Mosad','QA_ENG')
INSERT INTO TestTable(id,fname,lname,Title) VALUES (4,'Aly','Mohamed','PILOT')
```

您可以通过 MySQL 控制台查询数据库，以验证数据是否已提交到数据库：

```py
MariaDB [TestingPython]> select * from TestTable;
+----+--------+---------+-----------+
| id | fName  | lname   | Title     |
+----+--------+---------+-----------+
|  1 | Bassim | Aly     | NW_ENG    |
|  2 | Ahmed  | Hany    | DEVELOPER |
|  3 | Sara   | Mosaad  | QA_ENG    |
|  4 | Aly    | Mohamed | PILOT     |
+----+--------+---------+-----------+
```

现在，回到我们的 Python 代码，我们可以再次使用`execute()`函数；这次，我们使用它来选择在`TestTable`中插入的所有数据：

```py
import MySQLdb

SQL_IP ="10.10.10.130" SQL_USERNAME="root" SQL_PASSWORD="EnterpriseAutomation" SQL_DB="TestingPython"   sql_connection = MySQLdb.connect(SQL_IP,SQL_USERNAME,SQL_PASSWORD,SQL_DB) # print sql_connection   cursor = sql_connection.cursor() cursor.execute("select * from TestTable")   output = cursor.fetchall() print(output)
```

脚本的输出如下：

```py
python mysql_show_all.py 
((1L, 'Bassim', 'Aly', 'NW_ENG'), (2L, 'Ahmed', 'Hany', 'DEVELOPER'), (3L, 'Sara', 'Mosaa    d', 'QA_ENG'), (4L, 'Aly', 'Mohamed', 'PILOT'))
```

在上一个示例中，`id`值后的`L`字符可以通过再次将数据转换为整数（在 Python 中）来解决，使用`int()`函数。

游标内另一个有用的属性是`.rowcount`。这个属性将指示上一个`.execute()`方法返回了多少行。

# 总结

在本章中，我们学习了如何使用 Python 连接器与 DBMS 交互。我们在自动化服务器上安装了一个 MySQL 数据库，然后进行了验证。然后，我们使用 Python 脚本访问了 MySQL 数据库，并对其进行了操作。

在下一章中，我们将学习如何使用 Ansible 进行系统管理。
