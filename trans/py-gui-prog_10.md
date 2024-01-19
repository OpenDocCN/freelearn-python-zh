# 使用 SQL 改进数据存储

随着时间的推移，实验室出现了一个越来越严重的问题：CSV 文件到处都是！冲突的副本，丢失的文件，非数据输入人员更改的记录，以及其他与 CSV 相关的挫折正在困扰着项目。很明显，单独的 CSV 文件不适合作为存储实验数据的方式。需要更好的东西。

该设施有一个安装了 PostgreSQL 数据库的较旧的 Linux 服务器。您被要求更新您的程序，以便将数据存储在 PostgreSQL 数据库中，而不是在 CSV 文件中。这将是对您的应用程序的重大更新！

在本章中，您将学习以下主题：

+   安装和配置 PostgreSQL 数据库系统

+   在数据库中构建数据以获得良好的性能和可靠性

+   SQL 查询的基础知识

+   使用`psycopg2`库将您的程序连接到 PostgreSQL

# PostgreSQL

PostgreSQL（通常发音为 post-gress）是一个免费的、开源的、跨平台的关系数据库系统。它作为一个网络服务运行，您可以使用客户端程序或软件库进行通信。在撰写本文时，该项目刚刚发布了 10.0 版本。

尽管 ABQ 提供了一个已安装和配置的 PostgreSQL 服务器，但您需要为开发目的在您的工作站上下载并安装该软件。

共享的生产资源，如数据库和网络服务，永远不应该用于测试或开发。始终在您自己的工作站或单独的服务器上设置这些资源的独立开发副本。

# 安装和配置 PostgreSQL

要下载 PostgreSQL，请访问[`www.postgresql.org/download/`](https://www.postgresql.org/download)。EnterpriseDB 公司为 Windows、macOS 和 Linux 提供了安装程序，这是一个为 PostgreSQL 提供付费支持的商业实体。这些软件包包括服务器、命令行客户端和 pgAdmin 图形客户端。

要安装软件，请使用具有管理权限的帐户启动安装程序，并按照安装向导中的屏幕进行操作。

安装后，启动 pgAdmin，并通过选择 Object | Create | Login/Group Role 来为自己创建一个新的管理员用户。确保访问特权选项卡以检查超级用户，并访问定义选项卡以设置密码。然后，通过选择 Object | Create | Database 来创建一个数据库。确保将您的用户设置为所有者。要在数据库上运行 SQL 命令，请选择您的数据库并单击 Tools | Query Tool。

喜欢使用命令行的 MacOS 或 Linux 用户也可以使用以下命令：

```py
sudo -u postgres createuser -sP myusername
sudo -u postgres createdb -O myusername mydatabasename
psql -d mydatabasename -U myusername
```

尽管 Enterprise DB 为 Linux 提供了二进制安装程序，但大多数 Linux 用户更喜欢使用其发行版提供的软件包。您可能会得到一个稍旧的 PostgreSQL 版本，但对于大多数基本用例来说这并不重要。请注意，pgAdmin 通常是单独的软件包的一部分，最新版本（pgAdmin 4）可能不可用。不过，您应该没有问题遵循本章使用旧版本。

# 使用 psycopg2 连接

要从我们的应用程序进行 SQL 查询，我们需要安装一个可以直接与我们的数据库通信的 Python 库。最受欢迎的选择是`psycopg2`。`psycopg2`库不是 Python 标准库的一部分。您可以在[`initd.org/psycopg/docs/install.html`](http://initd.org/psycopg/docs/install.html)找到最新的安装说明；但是，首选方法是使用`pip`。

对于 Windows、macOS 和 Linux，以下命令应该有效：

```py
pip install --user psycopg2-binary
```

如果这不起作用，或者您更愿意从源代码安装它，请在网站上检查要求。`psycopg2`库是用 C 编写的，而不是 Python，因此它需要 C 编译器和其他几个开发包。Linux 用户通常可以从其发行版的软件包管理系统中安装`psycopg2`。我们将在本章后面深入研究`psycopg2`的使用。

# SQL 和关系数据库基础知识

在我们开始使用 Python 与 PostgreSQL 之前，您至少需要对 SQL 有基本的了解。如果您已经有了，可以跳到下一节；否则，准备好接受关系数据库和 SQL 的超短速成课程。

三十多年来，关系数据库系统一直是存储业务数据的事实标准。它们更常被称为**SQL 数据库**，因为与它们交互的**结构化查询语言**（**SQL**）。

SQL 数据库由表组成。表类似于我们的 CSV 文件，因为它具有表示单个项目的行和表示与每个项目关联的数据值的列。SQL 表与我们的 CSV 文件有一些重要的区别。首先，表中的每一列都被分配了一个严格执行的数据类型；就像当您尝试将`abcd`作为`int`使用时，Python 会产生错误一样，当您尝试将字母插入到数字或其他非字符串列中时，SQL 数据库会抱怨。SQL 数据库通常支持文本、数字、日期和时间、布尔值、二进制数据等数据类型。

SQL 表还可以具有约束，进一步强制执行插入到表中的数据的有效性。例如，可以给列添加唯一约束，这可以防止两行具有相同的值，或者添加非空约束，这意味着每一行都必须有一个值。

SQL 数据库通常包含许多表；这些表可以连接在一起，以表示更复杂的数据结构。通过将数据分解为多个链接的表，可以以比我们的二维纯文本 CSV 文件更有效和更具弹性的方式存储数据。

# 基本的 SQL 操作

SQL 是一个用于对表格数据进行大规模操作的强大而表达性的语言，但基础知识可以很快掌握。SQL 作为单独的查询来执行，这些查询要么定义数据，要么在数据库中操作数据。SQL 方言在不同的关系数据库产品之间略有不同，但它们大多数支持 ANSI/ISO 标准 SQL 进行核心操作。虽然我们将在本章中使用 PostgreSQL，但我们编写的大多数 SQL 语句都可以在不同的数据库中使用。

要遵循本节，连接到您的 PostgreSQL 数据库服务器上的空数据库，可以使用`psql`命令行工具、pgAdmin 4 图形工具或您选择的其他数据库客户端软件。

# 与 Python 的语法差异

如果您只在 Python 中编程过，那么最初可能会觉得 SQL 很奇怪，因为规则和语法非常不同。

我们将介绍各个命令和关键字，但以下是与 Python 不同的一些一般区别：

+   **SQL（大部分）不区分大小写**：尽管为了可读性的目的，按照惯例，将 SQL 关键字输入为全大写，但大多数 SQL 实现不区分大小写。这里有一些小的例外，但大部分情况下，您可以以最容易的方式输入 SQL 的大小写。

+   **空格不重要**：在 Python 中，换行和缩进可以改变代码的含义。在 SQL 中，空格不重要，语句以分号结尾。查询中的缩进和换行只是为了可读性。

+   **SQL 是声明性的**：Python 可以被描述为一种命令式编程语言：我们通过告诉 Python 如何做来告诉 Python 我们想要它做什么。SQL 更像是一种声明性语言：我们描述我们想要的，SQL 引擎会找出如何做。

当我们查看特定的 SQL 代码示例时，我们会遇到其他语法差异。

# 定义表和插入数据

SQL 表是使用`CREATE TABLE`命令创建的，如下面的 SQL 查询所示：

```py
CREATE TABLE musicians (id SERIAL PRIMARY KEY, name TEXT NOT NULL, born DATE, died DATE CHECK(died > born));
```

在这个例子中，我们正在创建一个名为`musicians`的表。在名称之后，我们指定了一系列列定义。每个列定义都遵循`column_name data_type constraints`的格式。

在这种情况下，我们有以下四列：

+   `id`列将是任意的行 ID。它的类型是`SERIAL`，这意味着它将是一个自动递增的整数字段，其约束是`PRIMARY KEY`，这意味着它将用作行的唯一标识符。

+   `name`字段的类型是`TEXT`，因此它可以容纳任意长度的字符串。它的`NOT NULL`约束意味着在该字段中不允许`NULL`值。

+   `born`和`died`字段是`DATE`字段，因此它们只能容纳日期值。`born`字段没有约束，但`died`有一个`CHECK`约束，强制其值必须大于任何给定行的`born`的值。

虽然不是必需的，但为每个表指定一个主键是一个好习惯。主键可以是一个字段，也可以是多个字段的组合，但对于任何给定的行，值必须是唯一的。例如，如果我们将`name`作为主键字段，那么我们的表中不能有两个同名的音乐家。

要向该表添加数据行，我们使用`INSERT INTO`命令如下：

```py
INSERT INTO musicians (name, born, died) VALUES ('Robert Fripp', '1946-05-16', NULL),   ('Keith Emerson', '1944-11-02', '2016-03-11'), ('Greg Lake', '1947-11-10', '2016-12-7'),   ('Bill Bruford', '1949-05-17', NULL), ('David Gilmour', '1946-03-06', NULL);
```

`INSERT INTO`命令接受表名和一个可选的列表，指定接收数据的字段；其他字段将接收它们的默认值（如果在`CREATE`语句中没有另外指定，则为`NULL`）。`VALUES`关键字表示要跟随的数据值列表，格式为逗号分隔的元组列表。每个元组对应一个表行，必须与在表名之后指定的字段列表匹配。

请注意，字符串由单引号字符括起来。与 Python 不同，单引号和双引号在 SQL 中具有不同的含义：单引号表示字符串文字，而双引号用于包含空格或需要保留大小写的对象名称。如果我们在这里使用双引号，将导致错误。

让我们创建并填充一个`instruments`表：

```py
CREATE TABLE instruments (id SERIAL PRIMARY KEY, name TEXT NOT NULL);
INSERT INTO instruments (name) VALUES ('bass'), ('drums'), ('guitar'), ('keyboards');
```

请注意，`VALUES`列表必须始终在每一行周围使用括号，即使每行只有一个值。

表在创建后可以使用`ALTER TABLE`命令进行更改，如下所示：

```py
ALTER TABLE musicians ADD COLUMN main_instrument INT REFERENCES instruments(id);
```

`ALTER TABLE`命令接受表名，然后是改变表的某个方面的命令。在这种情况下，我们正在添加一个名为`main_instrument`的新列，它将是一个整数。我们指定的`REFERENCES`约束称为**外键**约束；它将`main_instrument`的可能值限制为`instruments`表中现有的 ID 号码。

# 从表中检索数据

要从表中检索数据，我们使用`SELECT`语句如下：

```py
SELECT name FROM musicians;
```

`SELECT`命令接受一个列或以逗号分隔的列列表，后面跟着一个`FROM`子句，指定包含指定列的表或表。此查询要求从`musicians`表中获取`name`列。

它的输出如下：

| `name` |
| --- |
| `Bill Bruford` |
| `Keith Emerson` |
| `Greg Lake` |
| `Robert Fripp` |
| `David Gilmour` |

我们还可以指定一个星号，表示所有列，如下面的查询所示：

```py
SELECT * FROM musicians;
```

前面的 SQL 查询返回以下数据表：

| `ID` | `name` | `born` | `died` | `main_instrument` |
| --- | --- | --- | --- | --- |
| `4` | `Bill Bruford` | `1949-05-17` |  |  |
| `2` | `Keith Emerson` | `1944-11-02` | `2016-03-11` |  |
| `3` | `Greg Lake` | `1947-11-10` | `2016-12-07` |  |
| `1` | `Robert Fripp` | `1946-05-16` |  |  |
| `5` | `David Gilmour` | `1946-03-06` |  |  |

为了过滤掉我们不想要的行，我们可以指定一个`WHERE`子句，如下所示：

```py
SELECT name FROM musicians WHERE died IS NULL;
```

`WHERE`命令必须跟随一个条件语句；满足条件的行将被显示，而不满足条件的行将被排除。在这种情况下，我们要求没有死亡日期的音乐家的名字。

我们可以使用`AND`和`OR`运算符指定复杂条件如下：

```py
SELECT name FROM musicians WHERE born < '1945-01-01' AND died IS NULL;
```

在这种情况下，我们只会得到 1945 年之前出生且尚未去世的音乐家。

`SELECT`命令也可以对字段进行操作，或者按照某些列重新排序结果：

```py
SELECT name, age(born), (died - born)/365 AS "age at death" FROM musicians ORDER BY born DESC;
```

在这个例子中，我们使用`age()`函数来确定音乐家的年龄。我们还对`died`和`born`日期进行数学运算，以确定那些已故者的死亡年龄。请注意，我们使用`AS`关键字来重命名或别名生成的列。

当运行此查询时，请注意，对于没有死亡日期的人，`age at death`为`NULL`。对`NULL`值进行数学或逻辑运算总是返回`NULL`。

`ORDER BY`子句指定结果应该按照哪些列进行排序。它还接受`DESC`或`ASC`的参数来指定降序或升序。我们在这里按出生日期降序排序输出。请注意，每种数据类型都有其自己的排序规则，就像在 Python 中一样。日期按照它们的日历位置排序，字符串按照字母顺序排序，数字按照它们的数值排序。

# 更新行，删除行，以及更多的 WHERE 子句

要更新或删除现有行，我们使用`UPDATE`和`DELETE FROM`关键字与`WHERE`子句一起选择受影响的行。

删除很简单，看起来像这样：

```py
DELETE FROM instruments WHERE id=4;
```

`DELETE FROM`命令将删除与`WHERE`条件匹配的任何行。在这种情况下，我们匹配主键以确保只删除一行。如果没有行与`WHERE`条件匹配，将不会删除任何行。然而，请注意，`WHERE`子句在技术上是可选的：`DELETE FROM instruments`将简单地删除表中的所有行。

更新类似，只是包括一个`SET`子句来指定新的列值如下：

```py
UPDATE musicians SET main_instrument=3 WHERE id=1;
UPDATE musicians SET main_instrument=2 WHERE name='Bill Bruford';
```

在这里，我们将`main_instrument`设置为两位音乐家对应的`instruments`主键值。我们可以通过主键、名称或任何有效的条件集来选择要更新的音乐家记录。与`DELETE`一样，省略`WHERE`子句会影响所有行。

`SET`子句中可以更新任意数量的列：

```py
UPDATE musicians SET main_instrument=4, name='Keith Noel Emerson' WHERE name LIKE 'Keith%';
```

额外的列更新只需用逗号分隔。请注意，我们还使用`LIKE`运算符与`%`通配符一起匹配记录。`LIKE`可用于文本和字符串数据类型，以匹配部分数值。标准 SQL 支持两个通配符字符：`%`，匹配任意数量的字符，`_`，匹配单个字符。

我们也可以匹配转换后的列值：

```py
UPDATE musicians SET main_instrument=1 WHERE LOWER(name) LIKE '%lake';
```

在这里，我们使用`LOWER`函数将我们的字符串与列值的小写版本进行匹配。这不会永久改变表中的数据；它只是临时更改值以进行检查。

标准 SQL 规定`LIKE`是区分大小写的匹配。PostgreSQL 提供了一个`ILIKE`运算符，它可以进行不区分大小写的匹配，还有一个`SIMILAR TO`运算符，它使用更高级的正则表达式语法进行匹配。

# 子查询

与其每次使用`instruments`表的原始主键值，我们可以像以下 SQL 查询中所示使用子查询：

```py
UPDATE musicians SET main_instrument=(SELECT id FROM instruments WHERE name='guitar') WHERE name IN ('Robert Fripp', 'David Gilmour');
```

子查询是 SQL 查询中的 SQL 查询。如果可以保证子查询返回单个值，它可以用在任何需要使用文字值的地方。在这种情况下，我们让我们的数据库来确定`guitar`的主键是什么，并将其插入我们的`main_instrument`值。

在`WHERE`子句中，我们还使用`IN`运算符来匹配一个值列表。这允许我们匹配一个值列表。

`IN`可以与子查询一起使用，如下所示：

```py
SELECT name FROM musicians WHERE main_instrument IN (SELECT id FROM instruments WHERE name like '%r%')
```

由于`IN`是用于与值列表一起使用的，任何返回单列的查询都是有效的。

返回多行和多列的子查询可以在任何可以使用表的地方使用：

```py
SELECT name FROM (SELECT * FROM musicians WHERE died IS NULL) AS living_musicians;
```

请注意，`FROM`子句中的子查询需要一个别名；我们将子查询命名为`living_musicians`。

# 连接表

子查询是使用多个表的一种方法，但更灵活和强大的方法是使用`JOIN`。

`JOIN`在 SQL 语句的`FROM`子句中使用如下：

```py
SELECT musicians.name, instruments.name as main_instrument FROM musicians JOIN instruments ON musicians.main_instrument = instruments.id;
```

`JOIN`语句需要一个`ON`子句，指定用于匹配每个表中的行的条件。`ON`子句就像一个过滤器，就像`WHERE`子句一样；你可以想象`JOIN`创建一个包含来自两个表的每个可能组合的新表，然后过滤掉不匹配`ON`条件的行。表通常通过匹配共同字段中的值进行连接，比如在外键约束中指定的那些字段。在这种情况下，我们的`musicians.main_instrument`列包含`instrument`表的`id`值，所以我们可以基于此连接这两个表。

连接用于实现以下四种类型的表关系：

+   一对一连接将第一个表中的一行精确匹配到第二个表中的一行。

+   多对一连接将第一个表中的多行精确匹配到第二个表中的一行。

+   一对多连接将第一个表中的一行匹配到第二个表中的多行。

+   多对多连接匹配两个表中的多行。这种连接需要使用一个中间表。

早期的查询显示了一个多对一的连接，因为许多音乐家可以有相同的主要乐器。当一个列的值应该限制在一组选项时，通常会使用多对一连接，比如我们的 GUI 可能会用`ComboBox`小部件表示的字段。连接的表称为**查找表**。

如果我们要反转它，它将是一对多：

```py
SELECT instruments.name AS instrument, musicians.name AS musician FROM instruments JOIN musicians ON musicians.main_instrument = instruments.id;
```

一对多连接通常在记录有与之关联的子记录列表时使用；在这种情况下，每个乐器都有一个将其视为主要乐器的音乐家列表。连接的表通常称为**详细表**。

前面的 SQL 查询将给出以下输出：

| `instrument` | `musician` |
| --- | --- |
| `drums` | `Bill Bruford` |
| `keyboards` | `Keith Emerson` |
| `bass` | `Greg Lake` |
| `guitar` | `Robert Fripp` |
| `guitar` | `David Gilmour` |

请注意，`guitar`在乐器列表中重复了。当两个表连接时，结果的行不再指代相同类型的对象。乐器表中的一行代表一个乐器。`musician`表中的一行代表一个音乐家。这个表中的一行代表一个`instrument`-`musician`关系。

但假设我们想要保持输出，使得一行代表一个乐器，但仍然可以在每行中包含有关关联音乐家的信息。为了做到这一点，我们需要使用聚合函数和`GROUP BY`子句来聚合匹配的音乐家行，如下面的 SQL 查询所示：

```py
SELECT instruments.name AS instrument, count(musicians.id) as musicians FROM instruments JOIN musicians ON musicians.main_instrument = instruments.id GROUP BY instruments.name;
```

`GROUP BY`子句指定输出表中的每一行代表什么列。不在`GROUP BY`子句中的输出列必须使用聚合函数减少为单个值。在这种情况下，我们使用`count()`函数来计算与每个乐器关联的音乐家记录的总数。标准 SQL 包含几个更多的聚合函数，如`min()`、`max()`和`sum()`，大多数 SQL 实现也扩展了这些函数。

多对一和一对多连接并不能完全涵盖数据库需要建模的每种可能情况；很多时候，需要一个多对多的关系。

为了演示多对多连接，让我们创建一个名为`bands`的新表，如下所示：

```py
CREATE TABLE bands (id SERIAL PRIMARY KEY, name TEXT NOT NULL);
INSERT INTO bands(name) VALUES ('ABWH'), ('ELP'), ('King Crimson'), ('Pink Floyd'), ('Yes');
```

一个乐队有多位音乐家，音乐家也可以是多个乐队的一部分。我们如何在音乐家和乐队之间创建关系？如果我们在`musicians`表中添加一个`band`字段，这将限制每个音乐家只能属于一个乐队。如果我们在`band`表中添加一个`musician`字段，这将限制每个乐队只能有一个音乐家。为了建立连接，我们需要创建一个**连接表**，其中每一行代表一个音乐家在一个乐队中的成员资格。

按照惯例，我们称之为`musicians_bands`：

```py
CREATE TABLE musicians_bands (musician_id INT REFERENCES musicians(id), band_id INT REFERENCES bands(id), PRIMARY KEY (musician_id, band_id));
INSERT INTO musicians_bands(musician_id, band_id) VALUES (1, 3), (2, 2), (3, 2), (3, 3), (4, 1), (4, 2), (4, 5), (5,4);
```

`musicians_bands`表只包含两个外键字段，一个指向音乐家的 ID，一个指向乐队的 ID。请注意，我们使用两个字段的组合作为主键，而不是创建或指定一个字段作为主键。有多行具有相同的两个值是没有意义的，因此这种组合可以作为一个合适的主键。要编写使用这种关系的查询，我们的`FROM`子句需要指定两个`JOIN`语句：一个从`musicians`到`musicians_bands`，一个从`bands`到`musicians_bands`。

例如，让我们获取每位音乐家所在乐队的名字：

```py
SELECT musicians.name, array_agg(bands.name) AS bands FROM musicians JOIN musicians_bands ON musicians.id = musicians_bands.musician_id JOIN bands ON bands.id = musicians_bands.band_id GROUP BY musicians.name ORDER BY musicians.name ASC;
```

这个查询使用连接表将`音乐家`和`乐队`联系起来，然后显示音乐家的名字以及他们所在乐队的聚合列表，并按音乐家的名字排序。

前面的 SQL 查询给出了以下输出：

| `name` | `bands` |
| --- | --- |
| `Bill Bruford` | `{ABWH,"King Crimson",Yes}` |
| `David Gilmour` | `{"Pink Floyd"}` |
| `Greg Lake` | `{ELP,"King Crimson"}` |
| `Keith Emerson` | `{ELP}` |
| `Robert Fripp` | ``{"King Crimson"}`` |

这里使用的`array_agg()`函数将字符串值聚合成数组结构。这种方法和`ARRAY`数据类型是特定于 PostgreSQL 的。没有用于聚合字符串值的 SQL 标准函数，但大多数 SQL 实现都有解决方案。

# 学习更多

这是对 SQL 概念和语法的快速概述；我们已经涵盖了你需要了解的大部分内容，但还有很多东西需要学习。PostgreSQL 手册，可在[`www.postgresql.org/docs/manuals/`](https://www.postgresql.org/docs/manuals/)上找到，是 SQL 语法和 PostgreSQL 特定功能的重要资源和参考。

# 建模关系数据

我们的应用目前将数据存储在一个单独的 CSV 文件中；这种文件通常被称为**平面文件**，因为数据已经被压缩成了两个维度。虽然这种格式对我们的应用程序来说可以接受，并且可以直接转换成 SQL 表，但更准确和有用的数据模型需要更复杂的结构。

# 规范化

将平面数据文件拆分成多个表的过程称为**规范化**。规范化是一个涉及一系列级别的过程，称为**范式**，逐步消除重复并创建更精确的数据模型。虽然有许多范式，但大多数常见业务数据中遇到的问题都可以通过符合前三个范式来解决。

粗略地说，这需要以下条件：

+   **第一范式**要求每个字段只包含一个值，并且必须消除重复的列。

+   **第二范式**还要求每个值必须依赖于整个主键。换句话说，如果一个表有主键字段`A`、`B`和`C`，并且列`X`的值仅取决于列`A`的值，而不考虑`B`或`C`，那么该表就违反了第二范式。

+   **第三范式**还要求表中的每个值只依赖于主键。换句话说，给定一个具有主键`A`和数据字段`X`和`Y`的表，`Y`的值不能依赖于`X`的值。

符合这些规范的数据消除了冗余、冲突或未定义数据情况的可能性。

# 实体关系图

帮助规范化我们的数据并为关系数据库做好准备的一种有效方法是分析数据并创建一个**实体-关系图**，或**ERD**。 ERD 是一种用图表表示数据库存储信息和这些信息之间关系的方法。

这些东西被称为**实体**。**实体**是一个唯一可识别的对象；它对应于单个表的单行。实体具有属性，对应于其表的列。实体与其他实体有关系，这对应于我们在 SQL 中定义的外键关系。

让我们考虑实验室场景中的实体及其属性和关系：

+   有实验室。每个实验室都有一个名字。

+   有地块。每个地块都属于一个实验室，并有一个编号。在地块中种植种子样本。

+   有实验室技术人员，每个人都有一个名字。

+   有实验室检查，由实验室技术人员在特定实验室进行。每个检查都有日期和时间。

+   有地块检查，这是在实验室检查期间在地块上收集的数据。每个地块检查都记录了各种植物和环境数据。

以下是这些实体和关系的图表：

![](img/7fd91062-81f2-4025-a39e-26abb3216732.png)

在前面的图表中，实体由矩形表示。我们有五个实体：**实验室**，**地块**，**实验室技术人员**，**实验室检查**和**地块检查**。每个实体都有属性，用椭圆形表示。关系由菱形表示，其中的文字描述了左到右的关系。例如，**实验室技术人员**执行**实验室检查**，**实验室检查**在**实验室**中进行。请注意关系周围的小**1**和**n**字符：这些显示了关系是一对多，多对一还是多对多。

这个图表代表了我们数据的一个相当规范化的结构。要在 SQL 中实现它，我们只需为每个实体创建一个表，为每个属性创建一个列，并为每个关系创建一个外键关系（可能包括一个中间表）。在我们这样做之前，让我们考虑 SQL 数据类型。

# 分配数据类型

标准 SQL 定义了 16 种数据类型，包括各种大小的整数和浮点数类型、固定大小或可变大小的 ASCII 或 Unicode 字符串、日期和时间类型以及位类型。几乎每个 SQL 引擎都会扩展这些类型，以适应二进制数据、特殊类型的字符串或数字等。许多数据类型似乎有点多余，而且有几个别名在不同的实现之间可能是不同的。选择列的数据类型可能会令人困惑！

对于 PostgreSQL，以下图表提供了一些合理的选择：

| **存储的数据** | **推荐类型** | **备注** |
| --- | --- | --- |
| 固定长度字符串 | `CHAR` | 需要长度。 |
| 短到中等长度的字符串 | `VARCHAR` | 需要一个最大长度参数，例如，`VARCHAR(256)`。 |
| 长、自由格式文本 | `TEXT` | 无限长度，性能较慢。 |
| 较小的整数 | `SMALLINT` | 最多±32,767。 |
| 大多数整数 | `INT` | 最多约±21 亿。 |
| 较大的整数 | `BIGINT` | 最多约±922 万亿。 |
| 小数 | `NUMERIC` | 接受可选的长度和精度参数。 |
| 整数主键 | `SERIAL`，`BIGSERIAL` | 自动递增整数或大整数。 |
| 布尔 | `BOOLEAN` |  |
| 日期和时间 | `TIMESTAMP WITH TIMEZONE` | 存储日期、时间和时区。精确到 1 微秒。 |
| 无时间的日期 | `DATE` |  |
| 无日期的时间 | `TIME` | 可以有或没有时区。 |

这些类型可能在大多数应用中满足您的绝大多数需求，我们将在我们的 ABQ 数据库中使用其中的一部分。在创建表时，我们将参考我们的数据字典，并为我们的列选择适当的数据类型。

注意不要选择过于具体或限制性的数据类型。任何数据最终都可以存储在`TEXT`字段中；选择更具体的类型的目的主要是为了能够使用特定类型的运算符、函数或排序。如果不需要这些，可以考虑使用更通用的类型。例如，电话号码和美国社会安全号码可以纯粹用数字表示，但这并不意味着要将它们作为`INTEGER`或`NUMERIC`字段；毕竟，你不会用它们进行算术运算！

# 创建 ABQ 数据库

现在我们已经对数据进行了建模，并对可用的数据类型有了一定的了解，是时候建立我们的数据库了。首先，在您的 SQL 服务器上创建一个名为`abq`的数据库，并将自己设为所有者。

接下来，在您的项目根目录下，创建一个名为`sql`的新目录。在`sql`文件夹中，创建一个名为`create_db.sql`的文件。我们将从这个文件开始编写我们的数据库创建代码。

# 创建我们的表

我们创建表的顺序很重要。在外键关系中引用的任何表都需要在定义关系之前存在。因此，最好从查找表开始，并遵循一对多关系的链，直到所有表都被创建。在我们的 ERD 中，这将使我们从大致左上到右下。

# 创建查找表

我们需要创建以下三个查找表：

+   `labs`：这个查找表将包含我们实验室的 ID 字符串。

+   `lab_techs`：这个查找表将包含实验室技术员的姓名，通过他们的员工 ID 号进行标识。

+   `plots`：这个查找表将为每个物理地块创建一行，由实验室和地块号标识。它还将跟踪地块中种植的当前种子样本。

将用于创建这些表的 SQL 查询添加到`create_db.sql`中，如下所示：

```py
CREATE TABLE labs (id CHAR(1) PRIMARY KEY);
CREATE TABLE lab_techs (id SMALLINT PRIMARY KEY, name VARCHAR(512) UNIQUE NOT NULL);
CREATE TABLE plots (lab_id CHAR(1) NOT NULL REFERENCES labs(id), 
    plot SMALLINT NOT NULL, current_seed_sample CHAR(6), 
    PRIMARY KEY(lab_id, plot), 
    CONSTRAINT valid_plot CHECK (plot BETWEEN 1 AND 20));
```

在我们可以使用我们的数据库之前，查找表将需要被填充：

+   `labs`应该有值`A`到`E`，代表五个实验室。

+   `lab_techs`需要我们四名实验室技术员的姓名和 ID 号：`J Simms`（`4291`）、`P Taylor`（`4319`）、`Q Murphy`（`4478`）和`L Taniff`（`5607`）。

+   `plots`需要所有 100 个地块，每个实验室的地块号为`1`到`20`。种子样本在四个值之间轮换，如`AXM477`、`AXM478`、`AXM479`和`AXM480`。

您可以手动使用 pgAdmin 填充这些表，或者使用包含在示例代码中的`db_populate.sql`脚本。

# 实验室检查表

`lab_check`表是一个技术人员在给定日期的给定时间检查实验室的所有地块的一个实例，如下所示的 SQL 查询：

```py
CREATE TABLE lab_checks(
    date DATE NOT NULL, time TIME NOT NULL, 
    lab_id CHAR(1) NOT NULL REFERENCES labs(id), 
    lab_tech_id SMALLINT NOT NULL REFERENCES lab_techs(id), 
    PRIMARY KEY(date, time, lab_id));
```

`date`、`time`和`lab_id`列一起唯一标识了实验室检查，因此我们将它们指定为主键列。执行检查的实验室技术员的 ID 是这个表中唯一的属性。

# 地块检查表

地块检查是在单个地块收集的实际数据记录。这些是实验室检查的一部分，因此必须参考现有的实验室检查。

我们将从主键列开始：

```py
CREATE TABLE plot_checks(date DATE NOT NULL, time TIME NOT NULL,
lab_id CHAR(1) NOT NULL REFERENCES labs(id), plot SMALLINT NOT NULL,
```

这是`lab_check`表的主键加上`plot`号；它的键约束看起来像这样：

```py
PRIMARY KEY(date, time, lab_id, plot),
FOREIGN KEY(date, time, lab_id)
    REFERENCES lab_checks(date, time, lab_id),
FOREIGN KEY(lab_id, plot) REFERENCES plots(lab_id, plot),
```

现在我们可以添加属性列：

```py
seed_sample CHAR(6) NOT NULL, 
humidity NUMERIC(4, 2) CHECK (humidity BETWEEN 0.5 AND 52.0),
light NUMERIC(5, 2) CHECK (light BETWEEN 0 AND 100),
temperature NUMERIC(4, 2) CHECK (temperature BETWEEN 4 AND 40),
equipment_fault BOOLEAN NOT NULL,
blossoms SMALLINT NOT NULL CHECK (blossoms BETWEEN 0 AND 1000),
plants SMALLINT NOT NULL CHECK (plants BETWEEN 0 AND 20),
fruit SMALLINT NOT NULL CHECK (fruit BETWEEN 0 AND 1000),
max_height NUMERIC(6, 2) NOT NULL CHECK (max_height BETWEEN 0 AND 1000),
min_height NUMERIC(6, 2) NOT NULL CHECK (min_height BETWEEN 0 AND 1000),
median_height NUMERIC(6, 2) NOT NULL 
    CHECK (median_height BETWEEN min_height AND max_height),
notes TEXT);
```

请注意我们对数据类型和`CHECK`约束的使用，以复制我们的`data`字典中的限制。使用这些，我们利用了数据库的功能来防止无效数据。

# 创建视图

在完成数据库设计之前，我们将创建一个视图，以简化对我们数据的访问。视图在大多数方面都像表一样，但不包含实际数据；它实际上只是一个存储的`SELECT`查询。我们的视图将为与 GUI 交互更容易地格式化我们的数据。

视图是使用`CREATE VIEW`命令创建的，如下所示：

```py
CREATE VIEW data_record_view AS (
```

在括号内，我们放置将为我们的视图返回表数据的`SELECT`查询：

```py
SELECT pc.date AS "Date", to_char(pc.time, 'FMHH24:MI') AS "Time",
    lt.name AS "Technician", pc.lab_id AS "Lab", pc.plot AS "Plot",
    pc.seed_sample AS "Seed sample", pc.humidity AS "Humidity",
    pc.light AS "Light", pc.temperature AS "Temperature",
    pc.plants AS "Plants", pc.blossoms AS "Blossoms", pc.fruit AS 
    "Fruit",
    pc.max_height AS "Max Height", pc.min_height AS "Min Height",
    pc.median_height AS "Median Height", pc.notes AS "Notes"
FROM plot_checks AS pc JOIN lab_checks AS lc ON pc.lab_id = lc.lab_id AND pc.date = lc.date AND pc.time = lc.time JOIN lab_techs AS lt ON lc.lab_tech_id = lt.id);
```

我们正在选择`plot_checks`表，并通过外键关系将其与`lab_checks`和`lab_techs`连接起来。请注意，我们使用`AS`关键字给这些表起了别名。像这样的简短别名可以帮助使大查询更易读。我们还将每个字段别名为应用程序数据结构中使用的名称。这些必须用双引号括起来，以允许使用空格并保留大小写。通过使列名与应用程序中的`data`字典键匹配，我们就不需要在应用程序代码中翻译字段名。

诸如 PostgreSQL 之类的 SQL 数据库引擎在连接和转换表格数据方面非常高效。在可能的情况下，利用这种能力，让数据库为了您的应用程序的方便而进行数据格式化工作。

这完成了我们的数据库创建脚本。在您的 PostgreSQL 客户端中运行此脚本，并验证已创建四个表和视图。

# 将 SQL 集成到我们的应用程序中

将我们的应用程序转换为 SQL 后端将不是一项小任务。该应用程序是围绕 CSV 文件的假设构建的，尽管我们已经注意到了分离我们的关注点，但许多事情都需要改变。

让我们分解一下我们需要采取的步骤：

+   我们需要编写一个 SQL 模型

+   我们的`Application`类将需要使用 SQL 模型

+   记录表格需要重新排序以优先考虑我们的键，使用新的查找和使用数据库自动填充

+   记录列表将需要调整以适应新的数据模型和主键

在这个过程中，我们将需要修复其他错误或根据需要实现一些新的 UI 元素。让我们开始吧！

# 创建一个新模型

我们将从`models.py`开始导入`psycopg2`和`DictCursor`：

```py
import psycopg2 as pg
from psycopg2.extras import DictCursor
```

`DictCursor`将允许我们以 Python 字典而不是默认的元组获取结果，这在我们的应用程序中更容易处理。

开始一个名为`SQLModel`的新模型类，并从`CSVModel`复制`fields`属性。

首先清除`Technician`、`Lab`和`Plot`的值列表，并将`Technician`设置为`FT.string_list`类型：

```py
class SQLModel:
    fields = {
        ...
        "Technician": {'req': True, 'type': FT.string_list, 
                       'values': []},
        "Lab": {'req': True, 'type': FT.string_list, 'values': []},
        "Plot": {'req': True, 'type': FT.string_list,'values': []},

```

这些列表将从我们的查找表中填充，而不是硬编码到模型中。

我们将在`__init__()`方法中完成这些列表的填充：

```py
    def __init__(self, host, database, user, password):
        self.connection = pg.connect(host=host, database=database,
            user=user, password=password, cursor_factory=DictCursor)

        techs = self.query("SELECT * FROM lab_techs ORDER BY name")
        labs = self.query("SELECT id FROM labs ORDER BY id")
        plots = self.query(
        "SELECT DISTINCT plot FROM plots ORDER BY plot")
        self.fields['Technician']['values'] = [x['name'] for x in 
        techs]
        self.fields['Lab']['values'] = [x['id'] for x in labs]
        self.fields['Plot']['values'] = [str(x['plot']) for x in plots]
```

`__init__()`接受我们基本的数据库连接细节，并使用`psycopg2.connect()`建立与数据库的连接。因为我们将`DictCursor`作为`cursor_factory`传入，这个连接将返回所有数据查询的字典列表。

然后，我们查询数据库以获取我们三个查找表中的相关列，并使用列表推导式来展平每个查询的结果以获得`values`列表。

这里使用的`query`方法是我们需要接下来编写的包装器：

```py
    def query(self, query, parameters=None):
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, parameters)
        except (pg.Error) as e:
            self.connection.rollback()
            raise e
        else:
            self.connection.commit()
            if cursor.description is not None:
                return cursor.fetchall()
```

使用`psycopg2`查询数据库涉及从连接生成`cursor`对象，然后使用查询字符串和可选参数数据调用其`execute()`方法。默认情况下，所有查询都在事务中执行，这意味着它们在我们提交更改之前不会生效。如果查询因任何原因（SQL 语法错误、约束违反、连接问题等）引发异常，事务将进入损坏状态，并且必须在我们再次使用连接之前回滚（恢复事务的初始状态）。因此，我们将在`try`块中执行我们的查询，并在任何`psycopg2`相关异常（所有都是从`pg.Error`继承的）的情况下使用`connection.rollback()`回滚事务。

在查询执行后从游标中检索数据时，我们使用 `fetchall()` 方法，它将所有结果作为列表检索。但是，如果查询不是返回数据的查询（例如 `INSERT`），`fetchall()` 将抛出异常。为了避免这种情况，我们首先检查 `cursor.description`：如果查询返回了数据（即使是空数据集），`cursor.description` 将包含有关返回表的元数据（例如列名）。如果没有，则为 `None`。

让我们通过编写 `get_all_records()` 方法来测试我们的 `query()` 方法：

```py
    def get_all_records(self, all_dates=False):
        query = ('SELECT * FROM data_record_view '
            'WHERE NOT %(all_dates)s OR "Date" = CURRENT_DATE '
            'ORDER BY "Date", "Time", "Lab", "Plot"')
        return self.query(query, {'all_dates': all_dates})
```

由于我们的用户习惯于仅使用当天的数据，因此默认情况下只显示该数据，但如果我们需要检索所有数据，我们可以添加一个可选标志。我们可以在大多数 SQL 实现中使用 `CURRENT_DATE` 常量获取当前日期，我们在这里使用了它。为了使用我们的 `all_dates` 标志，我们正在使用准备好的查询。

语法 `%(all_dates)s` 定义了一个参数；它告诉 `psycopg2` 检查包含的参数字典，以便将其值替换到查询中。`psycopg2` 库将自动以一种安全的方式执行此操作，并正确处理各种数据类型，如 `None` 或布尔值。

始终使用准备好的查询将数据传递到 SQL 查询中。永远不要使用字符串格式化或连接！不仅比你想象的更难以正确实现，而且可能会导致意外或恶意的数据库损坏。

接下来，让我们创建 `get_record()`：

```py
def get_record(self, date, time, lab, plot):
    query = ('SELECT * FROM data_record_view '
        'WHERE "Date" = %(date)s AND "Time" = %(time)s '
        'AND "Lab" = %(lab)s AND "Plot" = %(plot)s')
    result = self.query(
        query, {"date": date, "time": time, "lab": lab, "plot": plot})
    return result[0] if result else {}
```

我们不再处理像我们的 `CSVModel` 那样的行号，因此此方法需要所有四个关键字段来检索记录。再次，我们使用了准备好的查询，为这四个字段指定参数。请注意参数括号的右括号后面的 `s`；这是一个必需的格式说明符，应始终为 `s`。

即使只有一行，`query()` 也会以列表的形式返回结果。我们的应用程序期望从 `get_record()` 中获得一个单行字典，因此我们的 `return` 语句会在列表不为空时提取 `result` 中的第一项，如果为空则返回一个空的 `dict`。

检索实验室检查记录非常类似：

```py
    def get_lab_check(self, date, time, lab):
        query = ('SELECT date, time, lab_id, lab_tech_id, '
            'lt.name as lab_tech FROM lab_checks JOIN lab_techs lt '
            'ON lab_checks.lab_tech_id = lt.id WHERE '
            'lab_id = %(lab)s AND date = %(date)s AND time = %(time)s')
        results = self.query(
            query, {'date': date, 'time': time, 'lab': lab})
        return results[0] if results else {}
```

在此查询中，我们使用连接来确保我们有技术员名称可用，而不仅仅是 ID。这种方法将在我们的 `save_record()` 方法和表单数据自动填充方法中非常有用。

`save_record()` 方法将需要四个查询：对 `lab_checks` 和 `plot_checks` 的 `INSERT` 和 `UPDATE` 查询。为了保持方法相对简洁，让我们将查询字符串创建为类属性。

我们将从实验室检查查询开始：

```py
    lc_update_query = ('UPDATE lab_checks SET lab_tech_id = '
        '(SELECT id FROM lab_techs WHERE name = %(Technician)s) '
        'WHERE date=%(Date)s AND time=%(Time)s AND lab_id=%(Lab)s')
    lc_insert_query = ('INSERT INTO lab_checks VALUES (%(Date)s, 
        '%(Time)s, %(Lab)s,(SELECT id FROM lab_techs '
        'WHERE name=%(Technician)s))')
```

这些查询非常简单，但请注意我们使用子查询来填充每种情况中的 `lab_tech_id`。我们的应用程序不知道实验室技术员的 ID 是什么，因此我们需要通过名称查找 ID。另外，请注意我们的参数名称与应用程序字段中使用的名称相匹配。这将使我们无需重新格式化从表单获取的记录数据。

地块检查查询更长，但并不复杂：

```py
    pc_update_query = (
        'UPDATE plot_checks SET seed_sample = %(Seed sample)s, '
        'humidity = %(Humidity)s, light = %(Light)s, '
        'temperature = %(Temperature)s, '
        'equipment_fault = %(Equipment Fault)s, '
        'blossoms = %(Blossoms)s, plants = %(Plants)s, '
        'fruit = %(Fruit)s, max_height = %(Max Height)s, '
        'min_height = %(Min Height)s, median_height = '
        '%(Median Height)s, notes = %(Notes)s '
        'WHERE date=%(Date)s AND time=%(Time)s '
        'AND lab_id=%(Lab)s AND plot=%(Plot)s')

    pc_insert_query = (
        'INSERT INTO plot_checks VALUES (%(Date)s, %(Time)s, %(Lab)s,'
        ' %(Plot)s, %(Seed sample)s, %(Humidity)s, %(Light)s,'
        ' %(Temperature)s, %(Equipment Fault)s, %(Blossoms)s,'
        ' %(Plants)s, %(Fruit)s, %(Max Height)s, %(Min Height)s,'
        ' %(Median Height)s, %(Notes)s)')
```

有了这些查询，我们可以开始 `save_record()` 方法：

```py
    def save_record(self, record):
        date = record['Date']
        time = record['Time']
        lab = record['Lab']
        plot = record['Plot']
```

`CSVModel.save_record()` 方法接受一个 `record` 字典和一个 `rownum`，但是我们不再需要 `rownum`，因为它没有意义。我们所有的关键信息已经在记录中。为了方便起见，我们将提取这四个字段并为它们分配本地变量名。

当我们尝试在这个数据库中保存记录时，有三种可能性：

+   实验室检查或地块检查记录都不存在。两者都需要创建。

+   实验室检查存在，但地块检查不存在。如果用户想要更正技术员的值，则需要更新实验室检查，而地块检查需要添加。

+   实验室检查和地块检查都存在。两者都需要使用提交的值进行更新。

为了确定哪种可能性是真实的，我们将利用我们的 `get_` 方法：

```py
        if self.get_lab_check(date, time, lab):
            lc_query = self.lc_update_query
        else:
            lc_query = self.lc_insert_query
        if self.get_record(date, time, lab, plot):
            pc_query = self.pc_update_query
        else:
            pc_query = self.pc_insert_query
```

对于实验室检查和地块检查，我们尝试使用我们的键值从各自的表中检索记录。如果找到了一个，我们将使用我们的更新查询；否则，我们将使用我们的插入查询。

现在，我们只需使用`record`作为参数列表运行这些查询。

```py
        self.query(lc_query, record)
        self.query(pc_query, record)
```

请注意，`psycopg2`不会因为我们传递了一个在查询中没有引用的额外参数的字典而出现问题，因此我们不需要费心从`record`中过滤不需要的项目。

这里还有一件事情要做：记住我们的`Application`需要跟踪更新和插入的行。由于我们不再处理行号，只有数据库模型知道是否执行了插入或更新。

让我们创建一个实例属性来共享这些信息：

```py
        if self.get_record(date, time, lab, plot):
            pc_query = self.pc_update_query
            self.last_write = 'update'
        else:
            pc_query = self.pc_insert_query
            self.last_write = 'insert'
```

现在`Application`可以在调用`save_record()`后检查`last_write`的值，以确定执行了哪种操作。

这个模型还需要最后一个方法；因为我们的数据库知道每个地块当前种子样本是什么，我们希望我们的表单自动为用户填充这些信息。我们需要一个方法，它接受一个`lab`和`plot_id`，并返回种子样本名称。

我们将称其为`get_current_seed_sample()`。

```py
    def get_current_seed_sample(self, lab, plot):
        result = self.query('SELECT current_seed_sample FROM plots '
            'WHERE lab_id=%(lab)s AND plot=%(plot)s',
            {'lab': lab, 'plot': plot})
        return result[0]['current_seed_sample'] if result else ''
```

这次，我们的`return`语句不仅仅是提取结果的第一行，而是提取该第一行中`current_seed_sample`列的值。如果没有`result`，我们将返回一个空字符串。

这完成了我们的模型类；现在让我们将其合并到应用程序中。

# 调整 SQL 后端的 Application 类

`Application`类需要的第一件事是数据库连接信息，以传递给模型。

对于主机和数据库名称，我们可以只需向我们的`SettingsModel`添加设置：

```py
    variables = {
        ...
        'db_host': {'type': 'str', 'value': 'localhost'},
        'db_name': {'type': 'str', 'value': 'abq'}
```

这些可以保存在我们的 JSON`config`文件中，可以编辑以从开发切换到生产，但我们的用户名和密码需要用户输入。为此，我们需要构建一个登录对话框。

# 构建登录窗口

Tkinter 没有为我们提供现成的登录对话框，但它提供了一个通用的`Dialog`类，可以被子类化以创建自定义对话框。

从`tkinter.simpledialog`中导入这个类到我们的`views.py`文件：

```py
from tkinter.simpledialog import Dialog
```

让我们从我们的类声明和`__init__()`方法开始：

```py
class LoginDialog(Dialog):

    def __init__(self, parent, title, error=''):
        self.pw = tk.StringVar()
        self.user = tk.StringVar()
        self.error = tk.StringVar(value=error)
        super().__init__(parent, title=title)
```

我们的类将像往常一样接受一个`parent`，一个窗口`title`，以及一个可选的`error`，如果需要重新显示带有`error`消息的对话框（例如，如果密码错误）。`__init__()`的其余部分为密码、用户名和`error`字符串设置了一些 Tkinter 变量；然后，它以通常的方式调用`super()`结束。

表单本身不是在`__init__()`中定义的；相反，我们需要重写`body()`方法：

```py
    def body(self, parent):
        lf = tk.Frame(self)
        ttk.Label(lf, text='Login to ABQ', font='Sans 20').grid()
```

我们做的第一件事是制作一个框架，并使用大字体在第一行添加一个标题标签。

接下来，我们将检查是否有`error`字符串，如果有，以适当的样式显示它。

```py
        if self.error.get():
            tk.Label(lf, textvariable=self.error,
                     bg='darkred', fg='white').grid()
```

现在我们将添加用户名和密码字段，并将我们的框架打包到对话框中。

```py
        ttk.Label(lf, text='User name:').grid()
        self.username_inp = ttk.Entry(lf, textvariable=self.user)
        self.username_inp.grid()
        ttk.Label(lf, text='Password:').grid()
        self.password_inp = ttk.Entry(lf, show='*', 
        textvariable=self.pw)
        self.password_inp.grid()
        lf.pack()
        return self.username_inp
```

注意我们在密码输入中使用`show`选项，它用我们指定的字符替换任何输入的文本，以创建一个隐藏的文本字段。另外，请注意我们从方法中返回用户名输入小部件。`Dialog`在显示时将聚焦在这里返回的小部件上。

`Dialog`自动提供`OK`和`Cancel`按钮；我们想知道点击了哪个按钮，如果是`OK`按钮，检索输入的信息。

点击 OK 会调用`apply()`方法，因此我们可以重写它来设置一个`result`值。

```py
        def apply(self):
            self.result = (self.user.get(), self.pw.get())
```

`Dialog`默认创建一个名为`result`的属性，其值设置为`None`。但是现在，如果我们的用户点击了 OK，`result`将是一个包含用户名和密码的元组。我们将使用这个属性来确定点击了什么，输入了什么。

# 使用登录窗口

为了使用对话框，我们的应用程序需要一个方法，它将在无限循环中显示对话框，直到用户单击取消或提供的凭据成功验证。

在`Application`中启动一个新的`database_login()`方法：

```py
        def database_login(self):
            error = ''
            db_host = self.settings['db_host'].get()
            db_name = self.settings['db_name'].get()
            title = "Login to {} at {}".format(db_name, db_host)
```

我们首先设置一个空的`error`字符串和一个`title`字符串，以传递给我们的`LoginDialog`类。

现在我们将开始无限循环：

```py
        while True:
            login = v.LoginDialog(self, title, error)
            if not login.result:
                break
```

在循环内部，我们创建一个`LoginDialog`，它将阻塞，直到用户单击其中一个按钮。对话框返回后，如果`login.result`是`None`，则用户已单击取消，因此我们会跳出循环并退出方法。

如果我们有一个非`None`的`login.result`，我们将尝试用它登录：

```py
        else:
            username, password = login.result
            try:
                self.data_model = m.SQLModel(
                 db_host, db_name, username, password)
            except m.pg.OperationalError:
                error = "Login Failed"
            else:
                break
```

从`result`元组中提取`username`和`password`后，我们尝试用它创建一个`SQLModel`实例。如果凭据失败，`psycopg2.connect`将引发`OperationalError`，在这种情况下，我们将简单地填充我们的`error`字符串，让无限循环再次迭代。

如果数据模型创建成功，我们只需跳出循环并退出方法。

回到`__init__()`，在设置我们的设置之后，让我们让`database_login()`开始工作：

```py
        self.database_login()
        if not hasattr(self, 'data_model'):
            self.destroy()
            return
```

在调用`self.database_login()`之后，`Application`要么有一个`data_model`属性（因为登录成功），要么没有（因为用户单击了取消）。如果没有，我们将通过销毁主窗口并立即从`__init__()`返回来退出应用程序。

当然，在这个逻辑生效之前，我们需要删除`CSVModel`的创建：

```py
        # Delete this line:
        self.data_model = m.CSVModel(filename=self.filename.get())
```

# 修复一些模型不兼容性

理论上，我们应该能够用相同的方法调用交换一个新模型，我们的应用程序对象将正常工作，但情况并非完全如此。我们需要做一些小的修复来让`Application`与我们的新模型一起工作。

# DataRecordForm 创建

首先，让我们在`Application.__init__()`中修复`DataRecordForm`的实例化：

```py
        # The data record form
        self.recordform = v.DataRecordForm(
            self, self.data_model.fields, self.settings, 
            self.callbacks)
```

以前，我们从`CSVModel`的静态类属性中提取了`fields`参数。我们现在需要从我们的数据模型实例中提取它，因为实例正在设置一些值。

# 修复 open_record()方法

接下来，我们需要修复我们的`open_record()`方法。它目前需要一个`rownum`，但我们不再有行号；我们有`date`、`time`、`lab`和`plot`。

为了反映这一点，用`rowkey`替换所有`rownum`的实例：

```py
    def open_record(self, rowkey=None):
        if rowkey is None:
        # ...etc
```

最后，在`get_record()`调用中扩展`rowkey`，因为它期望四个位置参数：

```py
        record = self.data_model.get_record(*rowkey)
```

# 修复 on_save()方法

`on_save()`的错误处理部分是好的，但在`if errors:`块之后，我们将开始改变事情：

```py
        data = self.recordform.get()
        try:
            self.data_model.save_record(data)
```

我们不再需要提取行号或将其传递给`save_record()`，并且我们可以删除对`IndexError`的处理，因为`SQLModel`不会引发该异常。我们还需要重写`inserted_rows`和`updated_rows`的更新。

在调用`self.status.set()`之后，删除此方法中的所有代码，并用以下代码替换：

```py
        key = (data['Date'], data['Time'], data['Lab'], data['Plot'])
        if self.data_model.last_write == 'update':
            self.updated_rows.append(key)
        else:
            self.inserted_rows.append(key)
        self.populate_recordlist()
        if self.data_model.last_write == 'insert':
            self.recordform.reset()
```

从传递给方法的`data`中构建主键元组后，我们使用`last_write`的值将其附加到正确的列表中。最后，在插入的情况下重置记录表单。

# 创建新的回调

我们希望为我们的记录表单有两个回调。当用户输入`lab`和`plot`值时，我们希望自动填充当前种植在该`plot`中的正确`seed`值。此外，当`date`、`time`和`lab`值已输入，并且我们有匹配的现有实验室检查时，我们应该填充执行该检查的实验室技术人员的姓名。

当然，如果我们的用户不希望数据自动填充，我们也不应该做这些事情。

让我们从`get_current_seed_sample()`方法开始：

```py
    def get_current_seed_sample(self, *args):
        if not (hasattr(self, 'recordform')
            and self.settings['autofill sheet data'].get()):
            return
        data = self.recordform.get()
        plot = data['Plot']
        lab = data['Lab']
        if plot and lab:
            seed = self.data_model.get_current_seed_sample(lab, plot)
            self.recordform.inputs['Seed sample'].set(seed)
```

我们首先检查是否已创建记录表单对象，以及用户是否希望数据自动填充。如果不是，我们退出该方法。接下来，我们从表单的当前数据中获取`plot`和`lab`。如果我们两者都有，我们将使用它们从模型中获取`seed`样本值，并相应地设置表单的`Seed sample`值。

我们将以类似的方式处理实验技术值：

```py
    def get_tech_for_lab_check(self, *args):
        if not (hasattr(self, 'recordform')
            and self.settings['autofill sheet data'].get()):
            return
        data = self.recordform.get()
        date = data['Date']
        time = data['Time']
        lab = data['Lab']

        if all([date, time, lab]):
            check = self.data_model.get_lab_check(date, time, lab)
            tech = check['lab_tech'] if check else ''
            self.recordform.inputs['Technician'].set(tech)
```

这一次，我们需要`date`、`time`和`lab`参数来获取实验检查记录。因为我们不能确定是否存在与这些值匹配的检查，所以如果我们找不到匹配的实验检查，我们将把`tech`设置为空字符串。

将这两种方法添加到`callbacks`字典中，`Application`类应该准备就绪。

# 更新我们的视图以适应 SQL 后端

让我们回顾一下我们需要在视图中进行的更改：

+   重新排列我们的字段，将所有主键放在前面

+   修复我们表单的`load_record()`方法，使其与新的关键结构配合使用

+   为我们的表单添加触发器以填充`Technician`和`Seed sample`

+   修复我们的记录列表以适应新的关键

让我们从我们的记录表单开始。

# 数据记录表单

我们的第一个任务是移动字段。这实际上只是剪切和粘贴代码，然后修复我们的`grid()`参数。将它们放在正确的键顺序中：Date、Time、Lab、Plot。然后，将 Technician 和 Seed sample 留在 Record Information 部分的末尾。

它应该看起来像这样：

![](img/c9aa7446-02f5-4cf4-b810-7b805ae2dd1b.png)

这种更改的原因是，所有可能触发 Technician 或 Seed sample 自动填充的字段将出现在这些字段之前。如果它们中的任何一个出现在之后，我们将无用地自动填充用户已经填写的字段。

在`__init__()`的末尾，让我们添加触发器来填充 Technician 和 Seed sample：

```py
        for field in ('Lab', 'Plot'):
            self.inputs[field].variable.trace(
                'w', self.callbacks['get_seed_sample'])
        for field in ('Date', 'Time', 'Lab'):
            self.inputs[field].variable.trace(
                'w', self.callbacks['get_check_tech'])
```

我们正在对实验检查和绘图的关键变量进行跟踪；如果它们中的任何一个发生变化，我们将调用适当的回调函数来自动填充表单。

在`load_record()`中，为了清晰起见，用`rowkey`替换`rownum`，然后修复标签`text`，使其有意义：

```py
        self.record_label.config(
            text='Record for Lab {2}, Plot {3} at {0} {1}'
            .format(*rowkey))
```

对于`DataRecordForm`的最后一个更改涉及一个小的可用性问题。随着我们自动填充表单，确定下一个需要聚焦的字段变得越来越令人困惑。我们将通过创建一个方法来解决这个问题，该方法找到并聚焦表单中的第一个空字段。

我们将称之为`focus_next_empty()`：

```py
    def focus_next_empty(self):
        for labelwidget in self.inputs.values():
            if (labelwidget.get() == ''):
                labelwidget.input.focus()
                break
```

在这个方法中，我们只是迭代所有的输入并检查它们当前的值。当我们找到一个返回空字符串时，我们将聚焦它，然后打破循环，这样就不会再检查了。我们可以删除`DataRecordForm.reset()`中对聚焦字段的任何调用，并将其替换为对此方法的调用。您还可以将其添加到我们应用程序的自动填充方法`get_current_seed_sample()`和`get_tech_for_lab_check()`中。

# 记录列表

在`RecordList`中，`Row`列不再包含我们希望显示的有用信息。

我们无法删除它，但我们可以使用这段代码隐藏它：

```py
self.treeview.config(show='headings')
```

`show`配置选项接受两个值中的任意一个或两个：`tree`和`headings`。`tree`参数代表`#0`列，因为它用于展开`tree`。`headings`参数代表其余的列。通过在这里只指定`headings`，`#0`列被隐藏了。

我们还需要处理我们的`populate()`方法，它在很大程度上依赖于`rownum`。

我们将从更改填充值的`for`循环开始：

```py
        for rowdata in rows:
            rowkey = (str(rowdata['Date']), rowdata['Time'],
            rowdata['Lab'], str(rowdata['Plot']))
            values = [rowdata[key] for key in valuekeys]
```

我们可以删除`enumerate()`调用，只需处理行数据，从中提取`rowkey`元组，通过获取`Date`、`Time`、`Lab`和`Plot`。这些需要转换为字符串，因为它们作为 Python 对象（如`date`和`int`）从数据库中出来，我们需要将它们与`inserted`和`updated`中的键进行匹配，这些键都是字符串值（因为它们是从我们的表单中提取的）。

让我们进行比较并设置我们的行标签：

```py
        if self.inserted and rowkey in self.inserted:
            tag = 'inserted'
        elif self.updated and rowkey in self.updated:
            tag = 'updated'
        else:
            tag = ''
```

现在，我们需要决定如何处理我们行的`iid`值。`iid`值必须是字符串；当我们的主键是整数时，这不是问题（可以轻松转换为字符串），但是我们的元组必须以某种方式进行序列化，以便我们可以轻松地反转。

解决这个问题的一个简单方法是将我们的元组转换为一个分隔的字符串：

```py
        stringkey = '{}|{}|{}|{}'.format(*rowkey)
```

任何不会出现在数据中的字符都可以作为分隔符；在这种情况下，我们选择使用管道字符。

现在我们可以在`treeview`中使用键的字符串版本：

```py
        self.treeview.insert('', 'end', iid=stringkey,
            text=stringkey, values=values, tag=tag)
```

该方法的最后部分将键盘用户聚焦在第一行。以前，为了聚焦第一行，我们依赖于第一个`iid`始终为`0`的事实。现在它将是一些数据相关的元组，所以我们必须在设置选择和焦点之前检索第一个`iid`。

我们可以使用`Treeview.identify_row()`方法来实现这一点：

```py
        if len(rows) > 0:
            firstrow = self.treeview.identify_row(0)
            self.treeview.focus_set()
            self.treeview.selection_set(firstrow)
            self.treeview.focus(firstrow)
```

`identify_row()`方法接受行号并返回该行的`iid`。一旦我们有了这个，我们就可以将它传递给`selection_set()`和`focus()`。

我们最后的更改是`on_open_record()`方法。由于我们使用了我们序列化的元组作为`iid`值，显然我们需要将其转换回一个可以传递回`on_open_record()`方法的元组。

这就像调用`split()`一样简单：

```py
        self.callbacks'on_open_record')
```

这修复了我们所有的视图代码，我们的程序已经准备好运行了！

# 最后的更改

呼！这是一次相当艰难的旅程，但你还没有完成。作业是，您需要更新您的单元测试以适应数据库和登录。最好的方法是模拟数据库和登录对话框。

还有一些 CSV 后端的残留物，比如文件菜单中的选择目标... 项目。您可以删除这些 UI 元素，但是将后端代码保留下来可能会在不久的将来派上用场。

# 总结

在本章中，您了解了关系数据库和 SQL，用于处理它们的语言。您学会了对数据进行建模和规范化，以减少不一致性的可能性，以及如何将平面文件转换为关系数据。您学会了如何使用`psycopg2`库，并经历了将应用程序转换为使用 SQL 后端的艰巨任务。

在下一章中，我们将接触云。我们需要使用不同的网络协议联系一些远程服务器来交换数据。您将了解有关 Python 标准库模块的信息，用于处理 HTTP 和 FTP，并使用它们来下载和上传数据。
