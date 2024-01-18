# 使用 Qt SQL 探索 SQL

大约 40 年来，使用**结构化查询语言**（通常称为 SQL）管理的**关系数据库**一直是存储、检索和分析世界数据的事实标准技术。无论您是创建业务应用程序、游戏、Web 应用程序还是其他应用，如果您的应用处理大量数据，您几乎肯定会使用 SQL。虽然 Python 有许多可用于连接到 SQL 数据库的模块，但 Qt 的`QtSql`模块为我们提供了强大和方便的类，用于将 SQL 数据集成到 PyQt 应用程序中。

在本章中，您将学习如何构建基于数据库的 PyQt 应用程序，我们将涵盖以下主题：

+   SQL 基础知识

+   使用 Qt 执行 SQL 查询

+   使用模型视图小部件与 SQL

# 技术要求

除了您自第一章以来一直在使用的基本设置，*开始使用 PyQt*，您还需要在 GitHub 存储库中找到的示例代码，网址为[`github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter09`](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter09)。

您可能还会发现拥有**SQLite**的副本对练习 SQL 示例很有帮助。SQLite 是免费的，可以从[`sqlite.org/download.html`](https://sqlite.org/download.html)下载。

查看以下视频，了解代码的实际操作：[`bit.ly/2M5xu1r`](http://bit.ly/2M5xu1r)

# SQL 基础知识

在我们深入了解`QtSql`提供的内容之前，您需要熟悉 SQL 的基础知识。本节将为您快速概述如何在 SQL 数据库中创建、填充、更改和查询数据。如果您已经了解 SQL，您可能希望跳到本章的 PyQt 部分。

SQL 在语法和结构上与 Python 非常不同。它是一种**声明式**语言，意味着我们描述我们想要的结果，而不是用于获得结果的过程。与 SQL 数据库交互时，我们执行**语句**。每个语句由一个 SQL**命令**和一系列**子句**组成，每个子句进一步描述所需的结果。语句以分号结束。

尽管 SQL 是标准化的，但所有 SQL 数据库实现都提供其自己的对标准语言的修改和扩展。我们将学习 SQL 的 SQLite 方言，它与标准 SQL 相当接近。

与 Python 不同，SQL 通常是不区分大小写的语言；但是，长期以来，将 SQL 关键字写成大写字母是一种惯例。这有助于它们与数据和对象名称区分开。我们将在本书中遵循这个惯例，但对于您的代码来说是可选的。

# 创建表

SQL 数据库由关系组成，也称为**表**。表是由行和列组成的二维数据结构。表中的每一行代表我们拥有信息的单个项目，每一列代表我们正在存储的信息类型。

使用`CREATE TABLE`命令定义表，如下所示：

```py
CREATE TABLE coffees (
        id  INTEGER PRIMARY KEY,
        coffee_brand TEXT NOT NULL,
        coffee_name TEXT NOT NULL,
        UNIQUE(coffee_brand, coffee_name)
        );
```

`CREATE TABLE`语句后面跟着表名和列定义列表。在这个例子中，`coffees`是我们正在创建的表的名称，列定义在括号内。每一列都有一个名称，一个数据类型，以及描述有效值的任意数量的**约束**。

在这种情况下，我们有三列：

+   `id`是一个整数列。它被标记为**主键**，这意味着它将是一个可以用来标识行的唯一值。

+   `coffee_brand`和`coffee_name`都是文本列，具有`NOT NULL`约束，这意味着它们不能有`NULL`值。

约束也可以在多个列上定义。在字段后添加的`UNIQUE`约束不是字段，而是一个表级约束，确保每行的`coffee _brand`和`coffee _name`的组合对于每行都是唯一的。

`NULL`是 SQL 中 Python 的`None`的等价物。它表示信息的缺失。

SQL 数据库至少支持文本、数字、日期、时间和二进制对象数据类型；但不少数据库实现会通过扩展 SQL 来支持额外的数据类型，比如货币或 IP 地址类型。许多数据库还有数字类型的`SMALL`和`BIG`变体，允许开发人员微调列使用的存储空间。

尽管简单的二维表很有用，但 SQL 数据库的真正威力在于将多个相关表连接在一起，例如：

```py
CREATE TABLE roasts (
        id INTEGER PRIMARY KEY,
        description TEXT NOT NULL UNIQUE,
        color TEXT NOT NULL UNIQUE
        );

CREATE TABLE coffees (
        id  INTEGER PRIMARY KEY,
        coffee_brand TEXT NOT NULL,
        coffee_name TEXT NOT NULL,
        roast_id INTEGER REFERENCES roasts(id),
        UNIQUE(coffee_brand, coffee_name)
        );

CREATE TABLE reviews (
        id INTEGER PRIMARY KEY,
        coffee_id REFERENCES coffees(id),
        reviewer TEXT NOT NULL,
        review_date DATE NOT NULL DEFAULT CURRENT_DATE,
        review TEXT NOT NULL
        );
```

`coffees`中的`roast_id`列保存与`roasts`的主键匹配的值，如`REFERENCES`约束所示。每个`coffees`记录不需要在每条咖啡记录中重写烘焙的描述和颜色，而是简单地指向`roasts`中保存有关该咖啡烘焙信息的行。同样，`reviews`表包含`coffee_id`列，它指向一个单独的`coffees`条目。这些关系称为**外键关系**，因为该字段引用另一个表的键。

在多个相关表中对数据进行建模可以减少重复，并强制执行数据一致性。想象一下，如果所有三个表中的数据合并成一张咖啡评论表，那么同一款咖啡产品的两条评论可能会指定不同的烘焙程度。这是不可能的，而且在关系型数据表中也不会发生。

# 插入和更新数据

创建表后，我们可以使用`INSERT`语句添加新的数据行，语法如下：

```py
INSERT INTO table_name(column1, column2, ...)
    VALUES (value1, value2, ...), (value3, value4, ...);
```

例如，让我们向`roasts`中插入一些行：

```py
INSERT INTO roasts(description, color) VALUES
    ('Light', '#FFD99B'),
    ('Medium', '#947E5A'),
    ('Dark', '#473C2B'),
    ('Burnt to a Crisp', '#000000');
```

在这个例子中，我们为`roasts`表中的每条新记录提供了`description`和`color`值。`VALUES`子句包含一个元组列表，每个元组代表一行数据。这些元组中的值的数量和数据类型*必须*与指定的列的数量和数据类型匹配。

请注意，我们没有包括所有的列——`id`缺失。我们在`INSERT`语句中不指定的任何字段都将获得默认值，除非我们另有规定，否则默认值为`NULL`。

在 SQLite 中，`INTEGER PRIMARY KEY`字段具有特殊行为，其默认值在每次插入时自动递增。因此，此查询产生的`id`值将为`1`（`Light`），`2`（`Medium`），`3`（`Dark`）和`4`（`Burnt to a Crisp`）。

这一点很重要，因为我们需要该键值来插入记录到我们的`coffees`表中：

```py
INSERT INTO coffees(coffee_brand, coffee_name, roast_id) VALUES
    ('Dumpy''s Donuts', 'Breakfast Blend', 2),
    ('Boise''s Better than Average', 'Italian Roast', 3),
    ('Strawbunks', 'Sumatra', 2),
    ('Chartreuse Hillock', 'Pumpkin Spice', 1),
    ('Strawbunks', 'Espresso', 3),
    ('9 o''clock', 'Original Decaf', 2);
```

与 Python 不同，SQL 字符串文字*必须*只使用单引号。双引号字符串被解释为数据库对象的名称，比如表或列。要在字符串中转义单引号，请使用两个单引号，就像我们在前面的查询中所做的那样。

由于我们的外键约束，不可能在`coffees`中插入包含不存在于`roasts`中的`roast_id`的行。例如，这将返回一个错误：

```py
INSERT INTO coffees(coffee_brand, coffee_name, roast_id) VALUES
    ('Minwell House', 'Instant', 48);
```

请注意，我们可以在`roast_id`字段中插入`NULL`；除非该列被定义为`NOT NULL`约束，否则`NULL`是唯一不需要遵守外键约束的值。

# 更新现有行

要更新表中的现有行，您可以使用`UPDATE`语句，如下所示：

```py
UPDATE coffees SET roast_id = 4 WHERE id = 2;
```

`SET`子句后面是要更改的字段的值分配列表，`WHERE`子句描述了必须为真的条件，如果要更新特定行。在这种情况下，我们将把`id`列为`2`的记录的`roast_id`列的值更改为`4`。

SQL 使用单个等号来进行赋值和相等操作。它永远不会使用 Python 使用的双等号。

更新操作也可以影响多条记录，就像这样：

```py
UPDATE coffees SET roast_id = roast_id + 1
    WHERE coffee_brand LIKE 'Strawbunks';
```

在这种情况下，我们通过将`Strawbunks`咖啡的所有`roast_id`值增加 1 来增加。每当我们在查询中引用列的值时，该值将是同一行中的列的值。

# 选择数据

SQL 中最重要的操作可能是`SELECT`语句，用于检索数据。一个简单的`SELECT`语句看起来像这样：

```py
SELECT reviewer, review_date
FROM reviews
WHERE  review_date > '2019-03-01'
ORDER BY reviewer DESC;
```

`SELECT`命令后面跟着一个字段列表，或者跟着`*`符号，表示*所有字段*。`FROM`子句定义了数据的来源；在这种情况下，是`reviews`表。`WHERE`子句再次定义了必须为真的条件才能包括行。在这种情况下，我们只包括比 2019 年 3 月 1 日更新的评论，通过比较每行的`review_date`字段（它是一个`DATE`类型）和字符串`'2019-03-01'`（SQLite 将其转换为`DATE`以进行比较）。最后，`ORDER BY`子句确定了结果集的排序方式。

# 表连接

`SELECT`语句总是返回一个值表。即使你的结果集只有一个值，它也会在一个行和一列的表中，而且没有办法从一个查询中返回多个表。然而，我们可以通过将数据合并成一个表来从多个表中提取数据。

这可以在`FROM`子句中使用`JOIN`来实现，例如：

```py
SELECT coffees.coffee_brand,
    coffees.coffee_name,
    roasts.description AS roast,
    COUNT(reviews.id) AS reviews
FROM coffees
    JOIN roasts ON coffees.roast_id = roasts.id
    LEFT OUTER JOIN reviews ON reviews.coffee_id = coffees.id
GROUP BY coffee_brand, coffee_name, roast
ORDER BY reviews DESC;
```

在这种情况下，我们的`FROM`子句包含两个`JOIN`语句。第一个将`coffees`与`roasts`通过匹配`coffees`中的`roast_id`字段和`roasts`中的`id`字段进行连接。第二个通过匹配`reviews`表中的`coffee_id`列和`coffees`表中的`id`列进行连接。

连接略有不同：请注意`reviews`连接是一个`LEFT OUTER JOIN`。这意味着我们包括了`coffees`中没有任何匹配`reviews`记录的行；默认的`JOIN`是一个`INNER`连接，意味着只有在两个表中都有匹配记录的行才会显示。

在这个查询中，我们还使用了一个**聚合函数**，`COUNT()`。`COUNT()`函数只是计算匹配的行数。聚合函数要求我们指定一个`GROUP BY`子句，列出将作为聚合基础的字段。换句话说，对于每个`coffee_brand`、`coffee_name`和`roast`的唯一组合，我们将得到数据库中评论记录的总数。其他标准的聚合函数包括`SUM`（用于对所有匹配值求和）、`MIN`（返回所有匹配值的最小值）和`MAX`（返回所有匹配值的最大值）。不同的数据库实现还包括它们自己的自定义聚合函数。

# SQL 子查询

`SELECT`语句可以通过将其放在括号中嵌入到另一个 SQL 语句中。这被称为**子查询**。它可以嵌入的确切位置取决于查询预期返回的数据类型：

+   如果语句将返回一个单行单列，它可以嵌入到期望单个值的任何地方

+   如果语句将返回一个单列多行，它可以嵌入到期望值列表的任何地方

+   如果语句将返回多行多列，它可以嵌入到期望值表的任何地方

考虑这个查询：

```py
SELECT coffees.coffee_brand, coffees.coffee_name
FROM coffees
    JOIN (
    SELECT * FROM roasts WHERE id > (
        SELECT id FROM roasts WHERE description = 'Medium'
            )) AS dark_roasts
    ON coffees.roast_id = dark_roasts.id
WHERE coffees.id IN (
    SELECT coffee_id FROM reviews WHERE reviewer = 'Maxwell');
```

这里有三个子查询。第一个位于`FROM`子句中：

```py
    (SELECT * FROM roasts WHERE id > (
        SELECT id FROM roasts WHERE description = 'Medium'
            )) AS dark_roasts
```

因为它以`SELECT *`开头，我们可以确定它将返回一个数据表（或者没有数据，但这不重要）。因此，它可以在`FROM`子句中使用，因为这里期望一个表。请注意，我们需要使用`AS`关键字给子查询一个名称。在`FROM`子句中使用子查询时，这是必需的。

这个子查询包含了它自己的子查询：

```py
        SELECT id FROM roasts WHERE description = 'Medium'
```

这个查询很可能会给我们一个单一的值，所以我们在期望得到单一值的地方使用它；在这种情况下，作为大于表达式的操作数。如果由于某种原因，这个查询返回了多行，我们的查询将会返回一个错误。

我们最终的子查询在`WHERE`子句中：

```py
    SELECT coffee_id FROM reviews WHERE reviewer = 'Maxwell'
```

这个表达式保证只返回一列，但可能返回多行。因此，我们将其用作`IN`关键字的参数，该关键字期望一个值列表。

子查询很强大，但如果我们对数据的假设不正确，有时也会导致减速和错误。

# 学习更多

我们在这里只是简单地介绍了 SQL 的基础知识，但这应该足够让您开始创建和使用简单的数据库，并涵盖了本章中将要使用的 SQL。在本章末尾的*进一步阅读*部分中，您将看到如何将 SQL 知识与 PyQt 结合起来创建数据驱动的应用程序。

# 使用 Qt 执行 SQL 查询

使用不同的 SQL 实现可能会令人沮丧：不仅 SQL 语法有细微差异，而且用于连接它们的 Python 库在它们实现的各种方法上经常不一致。虽然在某些方面，它不如更知名的 Python SQL 库方便，但`QtSQL`确实为我们提供了一种一致的抽象 API，以一致的方式处理各种数据库产品。正确利用时，它还可以为我们节省大量代码。

为了学习如何在 PyQt 中处理 SQL 数据，我们将为本章*SQL 基础*中创建的咖啡数据库构建一个图形前端。

可以使用以下命令从示例代码创建完整版本的数据库：

`$ sqlite3 coffee.db -init coffee.sql`。在前端工作之前，您需要创建这个数据库文件。

# 构建一个表单

我们的咖啡数据库有三个表：咖啡产品列表、烘焙列表和产品评论表。我们的 GUI 将设计如下：

+   它将有一个咖啡品牌和产品列表

+   当我们双击列表中的项目时，它将打开一个表单，显示关于咖啡的所有信息，以及与该产品相关的所有评论

+   它将允许我们添加新产品和新评论，或编辑任何现有信息

让我们首先从第四章中复制您的基本 PyQt 应用程序模板，*使用 QMainWindow 构建应用程序*，保存为`coffee_list1.py`。然后，像这样添加一个`QtSQL`的导入：

```py
from PyQt5 import QtSql as qts
```

现在我们要创建一个表单，显示关于我们的咖啡产品的信息。基本表单如下：

```py
class CoffeeForm(qtw.QWidget):

    def __init__(self, roasts):
        super().__init__()
        self.setLayout(qtw.QFormLayout())
        self.coffee_brand = qtw.QLineEdit()
        self.layout().addRow('Brand: ', self.coffee_brand)
        self.coffee_name = qtw.QLineEdit()
        self.layout().addRow('Name: ', self.coffee_name)
        self.roast = qtw.QComboBox()
        self.roast.addItems(roasts)
        self.layout().addRow('Roast: ', self.roast)
        self.reviews = qtw.QTableWidget(columnCount=3)
        self.reviews.horizontalHeader().setSectionResizeMode(
            2, qtw.QHeaderView.Stretch)
        self.layout().addRow(self.reviews)
```

这个表单有品牌、名称和咖啡烘焙的字段，以及一个用于显示评论的表格小部件。请注意，构造函数需要`roasts`，这是一个咖啡烘焙的列表，用于组合框；我们希望从数据库中获取这些，而不是将它们硬编码到表单中，因为新的烘焙可能会被添加到数据库中。

这个表单还需要一种方法来显示咖啡产品。让我们创建一个方法，它将获取咖啡数据并对其进行审查，并用它填充表单：

```py
    def show_coffee(self, coffee_data, reviews):
        self.coffee_brand.setText(coffee_data.get('coffee_brand'))
        self.coffee_name.setText(coffee_data.get('coffee_name'))
        self.roast.setCurrentIndex(coffee_data.get('roast_id'))
        self.reviews.clear()
        self.reviews.setHorizontalHeaderLabels(
            ['Reviewer', 'Date', 'Review'])
        self.reviews.setRowCount(len(reviews))
        for i, review in enumerate(reviews):
            for j, value in enumerate(review):
                self.reviews.setItem(i, j, qtw.QTableWidgetItem(value))
```

这个方法假设`coffee_data`是一个包含品牌、名称和烘焙 ID 的`dict`对象，而`reviews`是一个包含评论数据的元组列表。它只是遍历这些数据结构，并用数据填充每个字段。

在`MainWindow.__init__()`中，让我们开始主 GUI：

```py
        self.stack = qtw.QStackedWidget()
        self.setCentralWidget(self.stack)
```

我们将使用`QStackedWidget`在我们的咖啡列表和咖啡表单小部件之间进行切换。请记住，这个小部件类似于`QTabWidget`，但没有选项卡。

在我们可以构建更多 GUI 之前，我们需要从数据库中获取一些信息。让我们讨论如何使用`QtSQL`连接到数据库。

# 连接和进行简单查询

要使用`QtSQL`与 SQL 数据库，我们首先必须建立连接。这有三个步骤：

+   创建连接对象

+   配置连接对象

+   打开连接

在`MainWindow.__init__()`中，让我们创建我们的数据库连接：

```py
        self.db = qts.QSqlDatabase.addDatabase('QSQLITE')
```

我们不是直接创建`QSqlDatabase`对象，而是通过调用静态的`addDatabase`方法创建一个，其中包含我们将要使用的数据库驱动程序的名称。在这种情况下，我们使用的是 Qt 的 SQLite3 驱动程序。Qt 5.12 内置了九个驱动程序，包括 MySQL（`QMYSQL`）、PostgreSQL（`QPSQL`）和 ODBC 连接（包括 Microsoft SQL Server）（`QODBC`）。完整的列表可以在[`doc.qt.io/qt-5/qsqldatabase.html#QSqlDatabase-2`](https://doc.qt.io/qt-5/qsqldatabase.html#QSqlDatabase-2)找到。

一旦我们的数据库对象创建好了，我们需要用任何必需的连接设置来配置它，比如主机、用户、密码和数据库名称。对于 SQLite，我们只需要指定一个文件名，如下所示：

```py
        self.db.setDatabaseName('coffee.db')
```

我们可以配置的一些属性包括以下内容：

+   `hostName`—数据库服务器的主机名或 IP

+   `port`—数据库服务侦听的网络端口

+   `userName`—连接的用户名

+   `password`—用于身份验证的密码

+   `connectOptions`—附加连接选项的字符串

所有这些都可以使用通常的访问器方法进行配置或查询（例如`hostName()`和`setHostName()`）。如果你使用的是 SQLite 之外的其他东西，请查阅其文档，看看你需要配置哪些设置。

连接对象配置好之后，我们可以使用`open()`方法打开连接。这个方法返回一个布尔值，表示连接是否成功。如果失败，我们可以通过检查连接对象的`lastError`属性来找出失败的原因。

这段代码演示了我们可能会这样做：

```py
        if not self.db.open():
            error = self.db.lastError().text()
            qtw.QMessageBox.critical(
                None, 'DB Connection Error',
                'Could not open database file: '
                f'{error}')
            sys.exit(1)
```

在这里，我们调用`self.db.open()`，如果失败，我们从`lastError`中检索错误并在对话框中显示它。`lastError()`调用返回一个`QSqlError`对象，其中包含有关错误的数据和元数据；要提取实际的错误文本，我们调用它的`text()`方法。

# 获取有关数据库的信息

一旦我们的连接实际连接上了，我们就可以使用它来开始检查数据库。例如，`tables()`方法列出数据库中的所有表。我们可以使用这个方法来检查所有必需的表是否存在，例如：

```py
        required_tables = {'roasts', 'coffees', 'reviews'}
        tables = self.db.tables()
        missing_tables = required_tables - set(tables)
        if missing_tables:
            qtw.QMessageBox.critica(
                None, 'DB Integrity Error'
                'Missing tables, please repair DB: '
                f'{missing_tables}')
            sys.exit(1)
```

在这里，我们比较数据库中存在的表和必需表的集合。如果我们发现任何缺失，我们将显示错误并退出。

`set`对象类似于列表，不同之处在于其中的所有项目都是唯一的，并且它们允许进行一些有用的比较。在这种情况下，我们正在减去集合以找出`required_tables`中是否有任何不在`tables`中的项目。

# 进行简单的查询

与我们的 SQL 数据库交互依赖于`QSqlQuery`类。这个类表示对 SQL 引擎的请求，可以用来准备、执行和检索有关查询的数据和元数据。

我们可以使用数据库对象的`exec()`方法向数据库发出 SQL 查询：

```py
        query = self.db.exec('SELECT count(*) FROM coffees')
```

`exec()`方法从我们的字符串创建一个`QSqlQuery`对象，执行它，并将其返回给我们。然后我们可以从`query`对象中检索我们查询的结果：

```py
        query.next()
        count = query.value(0)
        print(f'There are {count} coffees in the database.')
```

重要的是要对这里发生的事情有一个心理模型，因为这并不是非常直观的。正如你所知，SQL 查询总是返回一张数据表，即使只有一行和一列。`QSqlQuery`有一个隐式的*游标*，它将指向数据的一行。最初，这个游标指向无处，但调用`next()`方法将它移动到下一个可用的数据行，这种情况下是第一行。然后使用`value()`方法来检索当前选定行中给定列的值（`value(0)`将检索第一列，`value(1)`将检索第二列，依此类推）。

所以，这里发生的情况类似于这样：

+   查询被执行并填充了数据。游标指向无处。

+   我们调用`next()`将光标指向第一行。

+   我们调用`value(0)`来检索行的第一列的值。

要从`QSqlQuery`对象中检索数据列表或表，我们只需要重复最后两个步骤，直到`next()`返回`False`（表示没有下一行要指向）。例如，我们需要一个咖啡烘焙的列表来填充我们的表单，所以让我们检索一下：

```py
        query = self.db.exec('SELECT * FROM roasts ORDER BY id')
        roasts = []
        while query.next():
            roasts.append(query.value(1))
```

在这种情况下，我们要求查询从`roasts`表中获取所有数据，并按`id`排序。然后，我们在查询对象上调用`next()`，直到它返回`False`；每次，提取第二个字段的值（`query.value(1)`）并将其附加到我们的`roasts`列表中。

现在我们有了这些数据，我们可以创建我们的`CoffeeForm`并将其添加到应用程序中：

```py
        self.coffee_form = CoffeeForm(roasts)
        self.stack.addWidget(self.coffee_form)
```

除了使用`value()`检索值之外，我们还可以通过调用`record()`方法来检索整行。这将返回一个包含当前行数据的`QSqlRecord`对象（如果没有指向任何行，则返回一个空记录）。我们将在本章后面使用`QSqlRecord`。

# 准备好的查询

很多时候，数据需要从应用程序传递到 SQL 查询中。例如，我们需要编写一个方法，通过 ID 号查找单个咖啡，以便我们可以在我们的表单中显示它。

我们可以开始编写该方法，就像这样：

```py
    def show_coffee(self, coffee_id):
        query = self.db.exec(f'SELECT * FROM coffees WHERE id={coffee_id}')
```

在这种情况下，我们使用格式化字符串直接将`coffee_id`的值放入我们的查询中。不要这样做！

使用字符串格式化或连接构建 SQL 查询可能会导致所谓的**SQL 注入漏洞**，其中传递一个特制的值可能会暴露或破坏数据库中的数据。在这种情况下，我们假设`coffee_id`将是一个整数，但假设一个恶意用户能够向这个函数发送这样的字符串：

```py
0; DELETE FROM coffees;
```

我们的字符串格式化将评估这一点，并生成以下 SQL 语句：

```py
SELECT * FROM coffees WHERE id=0; DELETE FROM coffees;
```

结果将是我们的`coffees`表中的所有行都将被删除！虽然在这种情况下可能看起来微不足道或荒谬，但 SQL 注入漏洞是许多数据泄露和黑客丑闻背后的原因，这些你在新闻中读到的。在处理重要数据时（还有比咖啡更重要的东西吗？），保持防御是很重要的。

执行此查询并保护数据库免受此类漏洞的正确方法是使用准备好的查询。**准备好的查询**是一个包含我们可以绑定值的变量的查询。数据库驱动程序将适当地转义我们的值，以便它们不会被意外地解释为 SQL 代码。

这个版本的代码使用了一个准备好的查询：

```py
        query1 = qts.QSqlQuery(self.db)
        query1.prepare('SELECT * FROM coffees WHERE id=:id')
        query1.bindValue(':id', coffee_id)
        query1.exec()
```

在这里，我们明确地创建了一个连接到我们的数据库的空`QSqlQuery`对象。然后，我们将 SQL 字符串传递给`prepare()`方法。请注意我们查询中使用的`:id`字符串；冒号表示这是一个变量。一旦我们有了准备好的查询，我们就可以开始将查询中的变量绑定到我们代码中的变量，使用`bindValue()`。在这种情况下，我们将`：id` SQL 变量绑定到我们的`coffee_id` Python 变量。

一旦我们的查询准备好并且变量被绑定，我们调用它的`exec()`方法来执行它。

一旦执行，我们可以从查询对象中提取数据，就像以前做过的那样：

```py
        query1.next()
        coffee = {
            'id': query1.value(0),
            'coffee_brand': query1.value(1),
            'coffee_name': query1.value(2),
            'roast_id': query1.value(3)
        }
```

让我们尝试相同的方法来检索咖啡的评论数据：

```py
        query2 = qts.QSqlQuery()
        query2.prepare('SELECT * FROM reviews WHERE coffee_id=:id')
        query2.bindValue(':id', coffee_id)
        query2.exec()
        reviews = []
        while query2.next():
            reviews.append((
                query2.value('reviewer'),
                query2.value('review_date'),
                query2.value('review')
            ))
```

请注意，这次我们没有将数据库连接对象传递给`QSqlQuery`构造函数。由于我们只有一个连接，所以不需要将数据库连接对象传递给`QSqlQuery`；`QtSQL`将自动在任何需要数据库连接的方法调用中使用我们的默认连接。

还要注意，我们使用列名而不是它们的编号从我们的`reviews`表中获取值。这同样有效，并且是一个更友好的方法，特别是在有许多列的表中。

我们将通过填充和显示我们的咖啡表单来完成这个方法：

```py
        self.coffee_form.show_coffee(coffee, reviews)
        self.stack.setCurrentWidget(self.coffee_form)
```

请注意，准备好的查询只能将*值*引入查询中。例如，您不能准备这样的查询：

```py
      query.prepare('SELECT * from :table ORDER BY :column')
```

如果您想构建包含可变表或列名称的查询，不幸的是，您将不得不使用字符串格式化。在这种情况下，请注意可能出现 SQL 注入的潜在风险，并采取额外的预防措施，以确保被插入的值是您认为的值。

# 使用 QSqlQueryModel

手动将数据填充到表小部件中似乎是一项繁琐的工作；如果您回忆起第五章，*使用模型视图类创建数据接口*，Qt 为我们提供了可以为我们完成繁琐工作的模型视图类。我们可以对`QAbstractTableModel`进行子类化，并创建一个从 SQL 查询中填充的模型，但幸运的是，`QtSql`已经以`QSqlQueryModel`的形式提供了这个功能。

正如其名称所示，`QSqlQueryModel`是一个使用 SQL 查询作为数据源的表模型。我们将使用它来创建我们的咖啡产品列表，就像这样：

```py
        coffees = qts.QSqlQueryModel()
        coffees.setQuery(
            "SELECT id, coffee_brand, coffee_name AS coffee "
            "FROM coffees ORDER BY id")
```

创建模型后，我们将其`query`属性设置为 SQL `SELECT`语句。模型的数据将从此查询返回的表中获取。

与`QSqlQuery`一样，我们不需要显式传递数据库连接，因为只有一个。如果您有多个活动的数据库连接，您应该将要使用的连接传递给`QSqlQueryModel()`。

一旦我们有了模型，我们就可以在`QTableView`中使用它，就像这样：

```py
        self.coffee_list = qtw.QTableView()
        self.coffee_list.setModel(coffees)
        self.stack.addWidget(self.coffee_list)
        self.stack.setCurrentWidget(self.coffee_list)
```

就像我们在第五章中所做的那样，*使用模型视图类创建数据接口*，我们创建了`QTableView`并将模型传递给其`setModel()`方法。然后，我们将表视图添加到堆叠小部件中，并将其设置为当前可见的小部件。

默认情况下，表视图将使用查询的列名作为标题标签。我们可以通过使用模型的`setHeaderData()`方法来覆盖这一点，就像这样：

```py
        coffees.setHeaderData(1, qtc.Qt.Horizontal, 'Brand')
        coffees.setHeaderData(2, qtc.Qt.Horizontal, 'Product')
```

请记住，`QSqlQueryModel`对象处于只读模式，因此无法将此表视图设置为可编辑，以便更改关于我们咖啡列表的详细信息。我们将在下一节中看看如何使用可编辑的 SQL 模型，*在没有 SQL 的情况下使用模型视图小部件*。不过，首先让我们完成我们的 GUI。

# 完成 GUI

现在我们的应用程序既有列表又有表单小部件，让我们在它们之间启用一些导航。首先，创建一个工具栏按钮，用于从咖啡表单切换到列表：

```py
        navigation = self.addToolBar("Navigation")
        navigation.addAction(
            "Back to list",
            lambda: self.stack.setCurrentWidget(self.coffee_list))
```

接下来，我们将配置我们的列表，以便双击项目将显示包含该咖啡记录的咖啡表单。请记住，我们的`MainView.show_coffee()`方法需要咖啡的`id`值，但列表小部件的`itemDoubleClicked`信号携带了点击的模型索引。让我们在`MainView`上创建一个方法来将一个转换为另一个：

```py
    def get_id_for_row(self, index):
        index = index.siblingAtColumn(0)
        coffee_id = self.coffee_list.model().data(index)
        return coffee_id
```

由于`id`在模型的列`0`中，我们使用`siblingAtColumn(0)`从被点击的任意行中检索列`0`的索引。然后我们可以通过将该索引传递给`model().data()`来检索`id`值。

现在我们有了这个，让我们为`itemDoubleClicked`信号添加一个连接：

```py
        self.coffee_list.doubleClicked.connect(
            lambda x: self.show_coffee(self.get_id_for_row(x)))
```

在这一点上，我们对我们的咖啡数据库有一个简单的只读应用程序。我们当然可以继续使用当前的 SQL 查询方法来管理我们的数据，但 Qt 提供了一种更优雅的方法。我们将在下一节中探讨这种方法。

# 在没有 SQL 的情况下使用模型视图小部件

在上一节中使用了`QSqlQueryModel`之后，您可能会想知道这种方法是否可以进一步泛化，直接访问表并避免完全编写 SQL 查询。您可能还想知道我们是否可以避开`QSqlQueryModel`的只读限制。对于这两个问题的答案都是*是*，这要归功于`QSqlTableModel`和`QSqlRelationalTableModels`。

要了解这些是如何工作的，让我们回到应用程序的起点重新开始：

1.  从一个新的模板副本开始，将其命名为`coffee_list2.py`。添加`QtSql`的导入和第一个应用程序中的数据库连接代码。现在让我们开始使用表模型构建。对于简单的情况，我们想要从单个数据库表创建模型，我们可以使用`QSqlTableModel`：

```py
self.reviews_model = qts.QSqlTableModel()
self.reviews_model.setTable('reviews')
```

1.  `reviews_model`现在是`reviews`表的可读/写表模型。就像我们在第五章中使用 CSV 表模型编辑 CSV 文件一样，我们可以使用这个模型来查看和编辑`reviews`表。对于需要从连接表中查找值的表，我们可以使用`QSqlRelationalTableModel`：

```py
self.coffees_model = qts.QSqlRelationalTableModel()
self.coffees_model.setTable('coffees')
```

1.  再一次，我们有一个可以用来查看和编辑 SQL 表中数据的表模型；这次是`coffees`表。但是，`coffees`表有一个引用`roasts`表的`roast_id`列。`roast_id`对应于应用程序用户没有意义，他们更愿意使用烘焙的`description`列。为了在我们的模型中用`roasts.description`替换`roast_id`，我们可以使用`setRelation()`函数将这两个表连接在一起，就像这样：

```py
        self.coffees_model.setRelation(
            self.coffees_model.fieldIndex('roast_id'),
            qts.QSqlRelation('roasts', 'id', 'description')
        )
```

这个方法接受两个参数。第一个是我们要连接的主表的列号，我们可以使用模型的`fieldIndex()`方法按名称获取。第二个是`QSqlRelation`对象，它表示外键关系。它所需的参数是表名（`roasts`），连接表中的相关列（`roasts.id`），以及此关系的显示字段（`description`）。

设置这种关系的结果是，我们的表视图将使用与`roasts`中的`description`列相关的值，而不是`roast_id`值，当我们将`coffee_model`连接到视图时。

1.  在我们可以将模型连接到视图之前，我们需要再走一步：

```py
self.mapper.model().select()
```

每当我们配置或重新配置`QSqlTableModel`或`QSqlRelationalTableModel`时，我们必须调用它的`select()`方法。这会导致模型生成并运行 SQL 查询，以刷新其数据并使其可用于视图。

1.  现在我们的模型准备好了，我们可以在视图中尝试一下：

```py
        self.coffee_list = qtw.QTableView()
        self.coffee_list.setModel(self.coffees_model)
```

1.  在这一点上运行程序，您应该会得到类似这样的东西：

![](img/85eb90e0-609f-4e31-9fff-1299a9c8b9b1.png)

请注意，由于我们的关系表模型，我们有一个包含烘焙描述的`description`列，而不是`roast_id`列。正是我们想要的。

还要注意，在这一点上，您可以查看和编辑咖啡列表中的任何值。`QSqlRelationalTableModel`默认是可读/写的，我们不需要对视图进行任何调整来使其可编辑。但是，它可能需要一些改进。

# 代理和数据映射

虽然我们可以编辑列表，但我们还不能添加或删除列表中的项目；在继续进行咖啡表单之前，让我们添加这个功能。

首先创建一些指向`MainView`方法的工具栏操作：

```py
        toolbar = self.addToolBar('Controls')
        toolbar.addAction('Delete Coffee(s)', self.delete_coffee)
        toolbar.addAction('Add Coffee', self.add_coffee)
```

现在我们将为这些操作编写`MainView`方法：

```py
    def delete_coffee(self):
        selected = self.coffee_list.selectedIndexes()
        for index in selected or []:
            self.coffees_model.removeRow(index.row())

    def add_coffee(self):
        self.stack.setCurrentWidget(self.coffee_list)
        self.coffees_model.insertRows(
            self.coffees_model.rowCount(), 1)
```

要从模型中删除一行，我们可以调用其`removeRow()`方法，传入所需的行号。这可以从`selectedIndexes`属性中获取。要添加一行，我们调用模型的`insertRows()`方法。这段代码应该很熟悉，来自第五章，*使用模型-视图类创建数据接口*。

现在，如果您运行程序并尝试添加一行，注意您基本上会得到一个`QLineEdit`，用于在每个单元格中输入数据。这对于咖啡品牌和产品名称等文本字段来说是可以的，但对于烘焙描述，更合理的是使用一些限制我们使用正确值的东西，比如下拉框。

在 Qt 的模型-视图系统中，决定为数据绘制什么小部件的对象称为**代理**。代理是视图的属性，通过设置我们自己的代理对象，我们可以控制数据的呈现方式以进行查看或编辑。

在由`QSqlRelationalTableModel`支持的视图的情况下，我们可以利用一个名为`QSqlRelationalDelegate`的现成委托，如下所示：

```py
self.coffee_list.setItemDelegate(qts.QSqlRelationalDelegate())
```

`QSqlRelationalDelegate`自动为已设置`QSqlRelation`的任何字段提供组合框。通过这个简单的更改，您应该发现`description`列现在呈现为一个组合框，其中包含来自`roasts`表的可用描述值。好多了！

# 数据映射

现在我们的咖啡列表已经很完善了，是时候处理咖啡表单了，这将允许我们显示和编辑单个产品及其评论的详细信息

让我们从表单的咖啡详情部分的 GUI 代码开始：

```py
class CoffeeForm(qtw.QWidget):

    def __init__(self, coffees_model, reviews_model):
        super().__init__()
        self.setLayout(qtw.QFormLayout())
        self.coffee_brand = qtw.QLineEdit()
        self.layout().addRow('Brand: ', self.coffee_brand)
        self.coffee_name = qtw.QLineEdit()
        self.layout().addRow('Name: ', self.coffee_name)
        self.roast = qtw.QComboBox()
        self.layout().addRow('Roast: ', self.roast)
```

表单的这一部分是我们在咖啡列表中显示的完全相同的信息，只是现在我们使用一系列不同的小部件来显示单个记录。将我们的`coffees`表模型连接到视图是直接的，但是我们如何将模型连接到这样的表单呢？一个答案是使用`QDataWidgetMapper`对象。

`QDataWidgetMapper`的目的是将模型中的字段映射到表单中的小部件。为了了解它是如何工作的，让我们将一个添加到`CoffeeForm`中：

```py
        self.mapper = qtw.QDataWidgetMapper(self)
        self.mapper.setModel(coffees_model)
        self.mapper.setItemDelegate(
            qts.QSqlRelationalDelegate(self))
```

映射器位于模型和表单字段之间，将它们之间的列进行转换。为了确保数据从表单小部件正确写入到模型中的关系字段，我们还需要设置适当类型的`itemDelegate`，在这种情况下是`QSqlRelationalDelegate`。

现在我们有了映射器，我们需要使用`addMapping`方法定义字段映射：

```py
        self.mapper.addMapping(
            self.coffee_brand,
            coffees_model.fieldIndex('coffee_brand')
        )
        self.mapper.addMapping(
            self.coffee_name,
            coffees_model.fieldIndex('coffee_name')
        )
        self.mapper.addMapping(
            self.roast,
            coffees_model.fieldIndex('description')
        )
```

`addMapping()`方法接受两个参数：一个小部件和一个模型列编号。我们使用模型的`fieldIndex()`方法通过名称检索这些列编号，但是您也可以在这里直接使用整数。

在我们可以使用我们的组合框之前，我们需要用选项填充它。为此，我们需要从我们的关系模型中检索`roasts`模型，并将其传递给组合框：

```py
        roasts_model = coffees_model.relationModel(
            self.coffees_model.fieldIndex('description'))
        self.roast.setModel(roasts_model)
        self.roast.setModelColumn(1)
```

`relationalModel()`方法可用于通过传递字段编号从我们的`coffees_model`对象中检索单个表模型。请注意，我们通过请求`description`的字段索引而不是`roast_id`来检索字段编号。在我们的关系模型中，`roast_id`已被替换为`description`。

虽然咖啡列表`QTableView`可以同时显示所有记录，但是我们的`CoffeeForm`设计为一次只显示一条记录。因此，`QDataWidgetMapper`具有*当前记录*的概念，并且只会使用当前记录的数据填充小部件。

因此，为了在我们的表单中显示数据，我们需要控制映射器指向的记录。`QDataWidgetMapper`类有五种方法来浏览记录表：

| 方法 | 描述 |
| --- | --- |
| `toFirst()` | 转到表中的第一条记录。 |
| `toLast()` | 转到表中的最后一条记录。 |
| `toNext()` | 转到表中的下一条记录。 |
| `toPrevious()` | 返回到上一个记录。 |
| `setCurrentIndex()` | 转到特定的行号。 |

由于我们的用户正在选择列表中的任意咖啡进行导航，我们将使用最后一个方法`setCurrentIndex()`。我们将在我们的`show_coffee()`方法中使用它，如下所示：

```py
    def show_coffee(self, coffee_index):
        self.mapper.setCurrentIndex(coffee_index.row())
```

`setCurrentIndex()`接受一个与模型中的行号对应的整数值。请注意，这与我们在应用程序的先前版本中使用的咖啡`id`值不同。在这一点上，我们严格使用模型索引值。

现在我们有了工作中的`CoffeeForm`，让我们在`MainView`中创建一个，并将其连接到我们咖啡列表的信号：

```py
        self.coffee_form = CoffeeForm(
            self.coffees_model,
            self.reviews_model
        )
        self.stack.addWidget(self.coffee_form)
        self.coffee_list.doubleClicked.connect(
            self.coffee_form.show_coffee)
        self.coffee_list.doubleClicked.connect(
            lambda: self.stack.setCurrentWidget(self.coffee_form))
```

由于我们使用索引而不是行号，我们可以直接将我们的`doubleClicked`信号连接到表单的`show_coffee()`方法。我们还将它连接到一个 lambda 函数，以将当前小部件更改为表单。

在这里，让我们继续创建一个工具栏操作来返回到列表：

```py
toolbar.addAction("Back to list", self.show_list)
```

相关的回调看起来是这样的：

```py
def show_list(self):
    self.coffee_list.resizeColumnsToContents()
    self.coffee_list.resizeRowsToContents()
    self.stack.setCurrentWidget(self.coffee_list)
```

为了适应在`CoffeeForm`中编辑时可能发生的数据可能的更改，我们将调用`resizeColumnsToContents()`和`resizeRowsToContents()`。然后，我们只需将堆栈小部件的当前小部件设置为`coffee_list`。

# 过滤数据

在这个应用程序中，我们需要处理的最后一件事是咖啡表单的评论部分：

1.  记住，评论模型是`QSqlTableModel`，我们将其传递给`CoffeeForm`构造函数。我们可以很容易地将它绑定到`QTableView`，就像这样：

```py
        self.reviews = qtw.QTableView()
        self.layout().addRow(self.reviews)
        self.reviews.setModel(reviews_model)
```

1.  这在我们的表单中添加了一个评论表。在继续之前，让我们解决一些视图的外观问题：

```py
        self.reviews.hideColumn(0)
        self.reviews.hideColumn(1)
        self.reviews.horizontalHeader().setSectionResizeMode(
            4, qtw.QHeaderView.Stretch)
```

表格的前两列是`id`和`coffee_id`，这两个都是我们不需要为用户显示的实现细节。代码的最后一行导致第四个字段（`review`）扩展到小部件的右边缘。

如果你运行这个，你会看到我们这里有一个小问题：当我们查看咖啡的记录时，我们不想看到*所有*的评论在表中。我们只想显示与当前咖啡产品相关的评论。

1.  我们可以通过对表模型应用**过滤器**来实现这一点。在`show_coffee()`方法中，我们将添加以下代码：

```py
        id_index = coffee_index.siblingAtColumn(0)
        self.coffee_id = int(self.coffees_model.data(id_index))
        self.reviews.model().setFilter(f'coffee_id = {self.coffee_id}')
        self.reviews.model().setSort(3, qtc.Qt.DescendingOrder)
        self.reviews.model().select()
        self.reviews.resizeRowsToContents()
        self.reviews.resizeColumnsToContents()
```

我们首先从我们的咖啡模型中提取选定的咖啡的`id`号码。这可能与行号不同，这就是为什么我们要查看所选行的第 0 列的值。我们将它保存为一个实例变量，因为以后可能会用到它。

1.  接下来，我们调用评论模型的`setFilter()`方法。这个方法接受一个字符串，它会被直接附加到用于从 SQL 表中选择数据的查询的`WHERE`子句中。同样，`setSort()`将设置`ORDER BY`子句。在这种情况下，我们按评论日期排序，最近的排在前面。

不幸的是，`setFilter()`中没有办法使用绑定变量，所以如果你想插入一个值，你必须使用字符串格式化。正如你所学到的，这会使你容易受到 SQL 注入漏洞的影响，所以在插入数据时要非常小心。在这个例子中，我们将`coffee_id`转换为`int`，以确保它不是 SQL 注入代码。

设置了过滤和排序属性后，我们需要调用`select()`来应用它们。然后，我们可以调整行和列以适应新的内容。现在，表单应该只显示当前选定咖啡的评论。

# 使用自定义委托

评论表包含一个带有日期的列；虽然我们可以使用常规的`QLineEdit`编辑日期，但如果我们能使用更合适的`QDateEdit`小部件会更好。与我们的咖啡列表视图不同，Qt 没有一个现成的委托可以为我们做到这一点。幸运的是，我们可以很容易地创建我们自己的委托：

1.  在`CoffeeForm`类的上面，让我们定义一个新的委托类：

```py
class DateDelegate(qtw.QStyledItemDelegate):

    def createEditor(self, parent, option, proxyModelIndex):
        date_inp = qtw.QDateEdit(parent, calendarPopup=True)
        return date_inp
```

委托类继承自`QStyledItemDelegate`，它的`createEditor()`方法负责返回将用于编辑数据的小部件。在这种情况下，我们只需要创建`QDateEdit`并返回它。我们可以根据需要配置小部件；例如，在这里我们启用了日历弹出窗口。

请注意，我们正在传递`parent`参数——这很关键！如果你不明确传递父小部件，你的委托小部件将弹出在它自己的顶层窗口中。

对于我们在评论表中的目的，这就是我们需要改变的全部内容。在更复杂的场景中，可能需要覆盖一些其他方法：

+   +   `setModelData()`方法负责从小部件中提取数据并将其传递给模型。如果需要在模型中更新之前将小部件的原始数据转换或准备好，你可能需要覆盖这个方法。

+   `setEditorData()`方法负责从模型中检索数据并将其写入小部件。如果模型数据不适合小部件理解，你可能需要重写这个方法。

+   `paint()`方法将编辑小部件绘制到屏幕上。你可以重写这个方法来构建一个自定义小部件，或者根据数据的不同来改变小部件的外观。如果你重写了这个方法，你可能还需要重写`sizeHint()`和`updateEditorGeometry()`来确保为你的自定义小部件提供足够的空间。

1.  一旦我们创建了自定义委托类，我们需要告诉我们的表视图使用它：

```py
        self.dateDelegate = DateDelegate()
        self.reviews.setItemDelegateForColumn(
            reviews_model.fieldIndex('review_date'),
            self.dateDelegate)
```

在这种情况下，我们创建了一个`DateDelegate`的实例，并告诉`reviews`视图在`review_date`列上使用它。现在，当你编辑评论日期时，你会得到一个带有日历弹出窗口的`QDateEdit`。

# 在表视图中插入自定义行

我们要实现的最后一个功能是在我们的评论表中添加和删除行：

1.  我们将从一些按钮开始：

```py
        self.new_review = qtw.QPushButton(
            'New Review', clicked=self.add_review)
        self.delete_review = qtw.QPushButton(
            'Delete Review', clicked=self.delete_review)
        self.layout().addRow(self.new_review, self.delete_review)
```

1.  删除行的回调足够简单：

```py
    def delete_review(self):
        for index in self.reviews.selectedIndexes() or []:
            self.reviews.model().removeRow(index.row())
        self.reviews.model().select()
```

就像我们在`MainView.coffee_list`中所做的一样，我们只需遍历所选的索引并按行号删除它们。

1.  添加新行会出现一个问题：我们可以添加行，但我们需要确保它们设置为使用当前选定的`coffee_id`。为此，我们将使用`QSqlRecord`对象。这个对象代表了来自`QSqlTableModel`的单行，并且可以使用模型的`record()`方法创建。一旦我们有了一个空的`record`对象，我们就可以用值填充它，并将其写回模型。我们的回调从这里开始：

```py
    def add_review(self):
        reviews_model = self.reviews.model()
        new_row = reviews_model.record()
        defaults = {
            'coffee_id': self.coffee_id,
            'review_date': qtc.QDate.currentDate(),
            'reviewer': '',
            'review': ''
        }
        for field, value in defaults.items():
            index = reviews_model.fieldIndex(field)
            new_row.setValue(index, value)
```

首先，我们通过调用`record()`从`reviews_model`中提取一个空记录。这样做很重要，因为它将被预先填充所有模型的字段。接下来，我们需要设置这些值。默认情况下，所有字段都设置为`None`（SQL `NULL`），所以如果我们想要默认值或者我们的字段有`NOT NULL`约束，我们需要覆盖这个设置。

在这种情况下，我们将`coffee_id`设置为当前显示的咖啡 ID（我们保存为实例变量，很好对吧？），并将`review_date`设置为当前日期。我们还将`reviewer`和`review`设置为空字符串，因为它们有`NOT NULL`约束。请注意，我们将`id`保留为`None`，因为在字段上插入`NULL`将导致它使用其默认值（在这种情况下，将是自动递增的整数）。

1.  设置好`dict`后，我们遍历它并将值写入记录的字段。现在我们需要将这个准备好的记录插入模型：

```py
        inserted = reviews_model.insertRecord(-1, new_row)
        if not inserted:
            error = reviews_model.lastError().text()
            print(f"Insert Failed: {error}")
        reviews_model.select()
```

`QSqlTableModel.insertRecord()`接受插入的索引（`-1`表示表的末尾）和要插入的记录，并返回一个简单的布尔值，指示插入是否成功。如果失败，我们可以通过调用`lastError().text()`来查询模型的错误文本。

1.  最后，我们在模型上调用`select()`。这将用我们插入的记录重新填充视图，并允许我们编辑剩下的字段。

到目前为止，我们的应用程序已经完全功能。花一些时间插入新的记录和评论，编辑记录，并删除它们。

# 总结

在本章中，你学习了关于 SQL 数据库以及如何在 PyQt 中使用它们。你学习了使用 SQL 创建关系数据库的基础知识，如何使用`QSqlDatabase`类连接数据库，以及如何在数据库上执行查询。你还学习了如何通过使用`QtSql`中可用的 SQL 模型视图类来构建优雅的数据库应用程序，而无需编写 SQL。

在下一章中，你将学习如何创建异步应用程序，可以处理缓慢的工作负载而不会锁定你的应用程序。你将学习如何有效地使用`QTimer`类，以及如何安全地利用`QThread`。我们还将介绍使用`QTheadPool`来实现高并发处理。

# 问题

尝试这些问题来测试你对本章的了解：

1.  编写一个 SQL“CREATE”语句，用于创建一个用于保存电视节目表的表。确保它有日期、时间、频道和节目名称的字段。还要确保它有主键和约束，以防止无意义的数据（例如同一频道上同时播放两个节目，或者没有时间或日期的节目）。

1.  以下 SQL 查询返回语法错误；你能修复吗？

```py
DELETE * FROM my_table IF category_id == 12;
```

1.  以下 SQL 查询不正确；你能修复吗？

```py
INSERT INTO flavors(name) VALUES ('hazelnut', 'vanilla', 'caramel', 'onion');
```

1.  “QSqlDatabase”的文档可以在[`doc.qt.io/qt-5/qsqldatabase.html`](https://doc.qt.io/qt-5/qsqldatabase.html)找到。了解如何使用多个数据库连接；例如，对同一数据库创建一个只读连接和一个读写连接。你将如何创建两个连接并对每个连接进行特定查询？

1.  使用“QSqlQuery”，编写代码将“dict”对象中的数据安全地插入到“coffees”表中：

```py
data = {'brand': 'generic', 'name': 'cheap coffee',
    'roast': 'light'}
# Your code here:
```

1.  你创建了一个“QSqlTableModel”对象并将其附加到“QTableView”。你知道表中有数据，但在视图中没有显示。查看代码并决定问题出在哪里：

```py
flavor_model = qts.QSqlTableModel()
flavor_model.setTable('flavors')
flavor_table = qtw.QTableView()
flavor_table.setModel(flavor_model)
mainform.layout().addWidget(flavor_table)
```

1.  以下是附加到“QLineEdit”的“textChanged”信号的回调函数。解释为什么这不是一个好主意：

```py
def do_search(self, text):
    self.sql_table_model.setFilter(f'description={text}')
    self.sql_table_model.select()
```

1.  你决定在咖啡列表的“烘焙”组合框中使用颜色而不是名称。你需要做哪些改变来实现这一点？

# 进一步阅读

查看以下资源以获取更多信息：

+   SQLite 中使用的 SQL 语言指南可以在[`sqlite.org/lang.html`](https://sqlite.org/lang.html)找到

+   可以在[`doc.qt.io/qt-5/qtsql-index.html`](https://doc.qt.io/qt-5/qtsql-index.html)找到“QtSQL”模块及其使用的概述
