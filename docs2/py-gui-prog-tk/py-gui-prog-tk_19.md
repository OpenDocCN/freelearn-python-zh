# B

# 快速 SQL 教程

超过三十年，关系型数据库系统一直是存储商业数据的实际标准。它们更常见地被称为 SQL 数据库，这是由于与它们交互所使用的 **结构化查询语言**（**SQL**）。尽管对 SQL 的全面处理需要一本自己的书，但本附录将简要介绍其基本概念和语法，这将足以跟随本书中对其的使用。

# SQL 概念

SQL 数据库由 **表** 组成。表就像 CSV 或电子表格文件，因为它有行表示单个项目，列表示与每个项目关联的数据值。尽管 SQL 表与电子表格有一些重要区别：

+   首先，表中的每一列都被分配了一个 **数据类型**，这是严格强制执行的。就像 Python 会在你尝试将 `"abcd"` 转换为 `int` 或 `0.03` 转换为 `date` 时产生错误一样，如果尝试将字母插入到数字列或十进制值插入到日期列中，SQL 数据库也会返回错误。SQL 数据库通常支持基本数据类型，如文本、数字、日期和时间、布尔值和二进制数据；此外，一些实现还有一些专门的数据类型，用于诸如 IP 地址、JSON 数据、货币或图像等事物。

+   SQL 表也可以有 **约束**，这进一步确保了插入到表中的数据的有效性。例如，一列可以指定一个 **唯一约束**，这防止两行在该列中有相同的值，或者一个 **非空约束**，这意味着每一行都必须有一个值。

SQL 数据库通常包含许多表，这些表可以连接起来以表示更复杂的数据结构。通过将数据分解成多个链接的表，我们可以以比二维纯文本 CSV 文件更高效和更具弹性的方式存储它。

## 与 Python 的语法差异

如果你以前只使用过 Python 编程，SQL 可能一开始会感觉有些奇怪，因为规则和语法非常不同。我们将介绍单个命令和关键字，但这里有一些与 Python 的一般差异：

+   SQL 是（大部分）**不区分大小写**的：虽然为了可读性目的，通常会将 SQL 关键字全部大写，但大多数 SQL 实现并不区分大小写。这里和那里有一些小的例外，但总的来说，你可以用对你最容易的任何大小写来输入 SQL。

+   **空白**没有意义：在 Python 中，换行和缩进可以改变代码的含义。在 SQL 中，空白没有意义，语句以分号结束。查询中的缩进和新行只是为了可读性。

+   SQL 是**声明式**的：Python 可以被描述为一种**命令式编程语言**：我们通过告诉 Python 如何做来告诉它我们想要做什么。SQL 更像是一种声明式语言：我们*描述*我们想要完成的事情，而 SQL 引擎会找出如何完成它。

我们在查看具体的 SQL 代码示例时，会遇到额外的语法差异。

# SQL 操作和语法

SQL 是一种强大且表达丰富的语言，用于对表格数据进行大量操作，但基本概念可以快速掌握。SQL 代码作为单独的查询执行，这些查询要么定义、操作，要么选择数据库中的数据。不同的关系数据库产品之间的 SQL 方言略有不同，但它们大多数都支持**ANSI/ISO 标准 SQL**的核心操作。

尽管这里涵盖的大多数基本概念和关键字在 SQL 实现中都是通用的，但我们将在本节的示例中使用 PostgreSQL 的方言。如果您想在不同的 SQL 实现上尝试这些示例，请准备好对语法进行一些调整。

要跟随本节内容，请连接到您 PostgreSQL 数据库服务器上的一个空数据库，无论是使用`psql`命令行工具、`pgAdmin`图形工具，还是您选择的任何其他数据库客户端软件。

## 定义表和插入数据

SQL 表是通过使用`CREATE TABLE`命令创建的，如下面的 SQL 查询所示：

```py
CREATE TABLE musicians (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  born DATE,
  died DATE CHECK(died > born)
 ); 
```

在这个例子中，我们正在创建一个名为`musicians`的表。在名称之后，我们指定一个列定义列表。每个列定义遵循以下格式：`column_name data_type constraints`。

让我们分解一下我们定义的这些字段的细节：

+   `id`列将为行提供一个任意的 ID 值。它的类型是`SERIAL`，这意味着它将是一个自动增长的整数字段，并且它的约束是`PRIMARY KEY`，这意味着它将被用作行的唯一标识符。

+   `name`字段是`TEXT`类型，因此它可以存储任意长度的字符串。它的`NOT NULL`约束意味着这个字段不允许有`NULL`值。

+   `born`和`died`字段是`DATE`类型，因此它们只能存储日期值。

+   `born`字段没有约束，但`died`字段有一个`CHECK`约束，强制其值必须大于任何给定行的`born`值。

虽然这不是必需的，但为每个表指定一个**主键**是一个好的实践。主键可以是一个字段，也可以是字段的组合，但任何给定行的值必须是唯一的。例如，如果我们把`name`设为主键字段，我们表中就不能有两个同名音乐家。

要向这个表中添加数据行，我们使用以下格式的`INSERT INTO`命令：

```py
INSERT INTO musicians (name, born, died)
VALUES
  ('Robert Fripp','1946-05-16', NULL),
  ('Keith Emerson', '1944-11-02', '2016-03-11'),
  ('Greg Lake', '1947-11-10', '2016-12-7'),
  ('Bill Bruford', '1949-05-17', NULL),
  ('David Gilmour', '1946-03-06', NULL); 
```

`INSERT INTO` 命令接受一个表名和一个可选的字段列表，指定接收数据字段；其他字段将接收它们的默认值（如果未在 `CREATE` 语句中指定，则为 `NULL`）。`VALUES` 关键字表示将跟随一系列数据值，格式为逗号分隔的元组列表。每个元组对应一行表，必须与表名之后指定的字段列表的顺序相匹配。

注意，字符串由单引号字符分隔。与 Python 不同，单引号和双引号在 SQL 中的含义不同：单引号表示字符串字面量，而双引号用于包含空格或需要保留大小写的对象名称。例如，如果我们把我们的表命名为 `Musicians of the '70s`，由于空格、撇号和大小写，我们需要用双引号括住那个名称。

使用双引号括起来的字符串字面量会导致错误，例如：

```py
INSERT INTO musicians (name, born, died)
VALUES
  ("Brian May", "1947-07-19", NULL);
-- Produces error:
ERROR:  column "Brian May" does not exist 
```

为了使我们的数据库更有趣，让我们创建并填充另一个表；这次是一个 `instruments` 表：

```py
CREATE TABLE instruments (id SERIAL PRIMARY KEY, name TEXT NOT NULL);
INSERT INTO instruments (name)
VALUES ('bass'), ('drums'), ('guitar'), ('keyboards'), ('sax'); 
```

注意，`VALUES` 列表必须始终在每个行周围使用括号，即使每行只有一个值。

要将 `musicians` 表与 `instruments` 表相关联，我们需要向其中添加一个列。可以使用 `ALTER TABLE` 命令在创建表之后更改表。例如，我们可以添加我们的新列如下：

```py
ALTER TABLE musicians
  ADD COLUMN main_instrument INT REFERENCES instruments(id); 
```

`ALTER TABLE` 命令接受一个表名，然后是一个改变表某些方面的命令。在这种情况下，我们正在添加一个名为 `main_instrument` 的新列，它将是一个整数。

我们指定的 `REFERENCES` 约束称为**外键约束**；它限制 `main_instrument` 的可能值只能为 `instruments` 表中存在的 ID 号。

## 从表中检索数据

要从表中检索数据，我们可以使用 `SELECT` 语句，如下所示：

```py
SELECT name FROM musicians; 
```

`SELECT` 命令接受一个列或逗号分隔的列列表，后跟一个 `FROM` 子句，该子句指定包含指定列的表或表。此查询请求 `musicians` 表中的 `name` 列。

其输出如下：

| name |
| --- |
| Bill Bruford |
| Keith Emerson |
| Greg Lake |
| Robert Fripp |
| David Gilmour |

除了列列表之外，我们还可以指定一个星号，表示“所有列”。例如：

```py
SELECT * FROM musicians; 
```

前面的 SQL 查询返回以下数据表：

| ID | name | born | died | main_instrument |
| --- | --- | --- | --- | --- |
| 4 | Bill Bruford | 1949-05-17 |  |  |
| 2 | Keith Emerson | 1944-11-02 | 2016-03-11 |  |
| 3 | Greg Lake | 1947-11-10 | 2016-12-07 |  |
| 1 | Robert Fripp | 1946-05-16 |  |  |
| 5 | David Gilmour | 1946-03-06 |  |  |

要过滤掉我们不需要的行，我们可以指定一个 `WHERE` 子句，如下所示：

```py
SELECT name FROM musicians WHERE died IS NULL; 
```

`WHERE`命令必须后跟一个条件表达式，该表达式评估为`True`或`False`；评估为`True`的表达式的行被显示，而评估为`False`的行被省略。

在这种情况下，我们要求的是那些`died`日期为`NULL`的音乐家的名字。我们可以通过使用`AND`和`OR`运算符组合表达式来指定更复杂的条件，如下所示：

```py
SELECT name FROM musicians WHERE born < '1945-01-01' AND died IS NULL; 
```

在这种情况下，我们只会得到在 1945 年之前出生且数据库中没有死亡日期的音乐家。

`SELECT`命令也可以对字段进行操作，或根据某些列重新排序结果：

```py
SELECT name, age(born), (died - born)/365 AS "age at death"
FROM musicians
ORDER BY born DESC; 
```

在这个例子中，我们使用`age()`函数根据出生日期确定音乐家的年龄。我们还对`died`和`born`日期进行数学运算，以确定已故者的死亡年龄。请注意，我们使用`AS`关键字来**别名**或重命名生成的列。

当你运行这个查询时，你应该得到如下输出：

| 姓名 | 年龄 | 死亡年龄 |
| --- | --- | --- |
| 比尔·布鲁福德 | 72 岁 4 个月 18 天 |  |
| 格雷格·莱克 | 73 岁 10 个月 24 天 | 69 |
| 罗伯特·弗里普 | 75 岁 4 个月 19 天 |  |
| 大卫·吉尔莫尔 | 75 岁 6 个月 29 天 |  |
| 吉思·艾默森 | 76 岁 11 个月 2 天 | 71 |

注意，对于没有死亡日期的人，“死亡年龄”是`NULL`。对`NULL`值进行数学或逻辑运算始终返回`NULL`答案。

`ORDER BY`子句指定了按哪个列或列列表排序结果。它还接受`DESC`或`ASC`参数来指定降序或升序，分别。

我们在这里按出生日期降序排列了输出。请注意，每种数据类型都有自己的排序规则，就像在 Python 中一样。日期按日历位置排序，字符串按字母顺序排序，数字按其数值排序。

## 更新行、删除行和更多的 WHERE 子句

要更新或删除现有行，我们使用`UPDATE`和`DELETE FROM`关键字与`WHERE`子句结合来选择受影响的行。

删除相对简单；例如，如果我们想删除`id`值为`5`的`instrument`记录，它看起来会是这样：

```py
DELETE FROM instruments WHERE id=5; 
```

`DELETE FROM`命令将删除任何匹配`WHERE`条件的行。在这种情况下，我们匹配主键以确保只删除一行。如果没有行匹配`WHERE`条件，则不会删除任何行。请注意，然而，`WHERE`子句在技术上不是必需的：`DELETE FROM instruments`将简单地删除表中的所有行。

更新类似，但它包括一个`SET`子句来指定新列值，如下所示：

```py
UPDATE musicians SET main_instrument=3 WHERE id=1;
UPDATE musicians SET main_instrument=2 WHERE name='Bill Bruford'; 
```

在这里，我们将`musicians`表中的`main_instrument`设置为`instruments`表中标识我们想要与每位音乐家关联的乐器的主键值。

我们可以使用主键、名称或任何条件的组合来选择我们想要更新的`musician`记录。就像`DELETE`一样，省略`WHERE`子句会导致查询影响所有行。

`SET`子句中可以更新任意数量的列；例如：

```py
UPDATE musicians
  SET main_instrument=4, name='Keith Noel Emerson'
  WHERE name LIKE 'Keith%'; 
```

要更新的附加列只需用逗号分隔。请注意，我们还在使用`LIKE`运算符和`%`通配符的同时匹配记录。`LIKE`可以与文本和字符串数据类型一起使用来匹配部分值。标准 SQL 支持两个通配符：%匹配零个或多个字符，_ 匹配单个字符。

我们也可以匹配转换后的列值：

```py
UPDATE musicians SET main_instrument=1 WHERE LOWER (name) LIKE '%lake'; 
```

在这里，我们使用了`LOWER`函数来将我们的字符串与列值的 lowercase 版本进行匹配。这不会永久更改表中的数据；它只是暂时更改值以用于比较目的。

标准 SQL 指定`LIKE`是区分大小写的匹配。PostgreSQL 提供了一个`ILIKE`运算符，它执行不区分大小写的匹配，以及一个`SIMILAR TO`运算符，它使用更高级的正则表达式语法进行匹配。

## 子查询

使用无意义的键值插入数据并不非常用户友好。为了使插入这些值更加直观，我们可以使用**子查询**，如下面的 SQL 查询所示：

```py
UPDATE musicians
SET main_instrument=(
  SELECT id FROM instruments WHERE name='guitar'
)
WHERE name IN ('Robert Fripp', 'David Gilmour'); 
```

子查询是 SQL 查询中的 SQL 查询。如果你的子查询可以保证返回单个值，它可以在任何使用文字值的地方使用。

在这种情况下，我们让数据库来做找出`'guitar'`的主键是什么的工作，并将返回的整数插入到我们的`main_instrument`值中。

在`WHERE`子句中，我们也使用了`IN`运算符来匹配音乐家的姓名。就像 Python 的`in`关键字一样，这个 SQL 关键字允许我们匹配值列表。`IN`也可以与子查询一起使用；例如：

```py
SELECT name FROM musicians
WHERE main_instrument IN (
  SELECT id FROM instruments WHERE name LIKE '%r%'
) 
```

在这个例子中，我们要求数据库给我们每个主要乐器包含字母“r”的音乐家。由于`IN`是用来与值列表一起使用的，任何返回单个列和任意行数的查询都是有效的。在这种情况下，我们的子查询返回了几个只有`id`列的行，所以它与`IN`配合得很好。

返回多行和多列的子查询可以在任何可以使用表的地方使用；例如，我们可以在`FROM`子句中使用子查询，如下所示：

```py
SELECT name
FROM (
  SELECT * FROM musicians WHERE died IS NULL
) AS living_musicians; 
```

在这种情况下，SQL 将我们的子查询视为数据库中的一个表。请注意，在`FROM`子句中使用的子查询需要别名；我们将此子查询别名为`living_musicians`。

## 表连接

子查询是使用多个表的一种方法，但更灵活和强大的方法是使用`JOIN`。`JOIN`用于 SQL 语句的`FROM`子句中，例如：

```py
SELECT musicians.name, instruments.name as main_instrument
FROM musicians
  JOIN instruments ON musicians.main_instrument = instrument.id; 
```

`JOIN`语句需要一个`ON`子句，该子句指定用于匹配每个表中行的条件。`ON`子句的作用就像一个过滤器，就像`WHERE`子句一样；你可以想象`JOIN`创建了一个包含来自两个表的所有可能行组合的新表，然后过滤掉不符合`ON`条件的行。

表通常通过匹配公共字段中的值来连接，例如在外键约束中指定的那些。在这种情况下，我们的`musicians.main_instrument`列包含来自`instrument`表的`id`值，因此我们可以根据这个值将两个表连接起来。

连接用于实现四种类型的表关系：

+   **一对一连接**将第一表中的单行与第二表中的单行精确匹配。

+   **多对一连接**将第一表中的多行与第二表中的单行精确匹配。

+   **一对多连接**将第一表中的一行与第二表中的多行匹配。

+   **多对多连接**匹配两个表中的多行。这种连接需要使用一个中间表。

之前的查询显示了一个多对一连接，因为许多音乐家可能有相同的主体乐器。当一列的值应该限制为一组选项时，通常使用多对一连接，例如我们的 GUI 可能用`Combobox`小部件表示的字段。连接的表通常被称为**查找表**。

如果我们反转最后的查询，它将是一对多：

```py
SELECT instruments.name AS instrument, musicians.name AS musician
FROM instruments
  JOIN musicians ON musicians.main_instrument = instruments.id; 
```

一对多连接通常用于一个记录有一个与其关联的子记录列表的情况；在这种情况下，每种乐器都有一个将其视为主要乐器的音乐家列表。连接的表通常被称为**详细表**。前面的 SQL 查询将给出以下输出：

| 乐器 | 音乐家 |
| --- | --- |
| 鼓 | 比尔·布鲁福德 |
| 键盘 | 凯斯·埃默森 |
| 贝斯 | 格雷格·莱克 |
| 吉他 | 罗伯特·弗里普 |
| 吉他 | 大卫·吉尔莫 |

注意到`guitar`在`instrument`列表中重复了。当两个表连接时，结果表的行不再指代相同的实体。`instrument`表中的一行代表一种乐器。

`musician`表中的一行代表一位音乐家。*这张*表中的一行代表一种乐器与音乐家的关系。

假设我们想要保持输出，使得一行代表一种乐器，但仍然在每一行中包含有关关联音乐家的信息。为此，我们需要使用**聚合函数**和`GROUP BY`子句将匹配的音乐家行组合起来，如下面的 SQL 查询所示：

```py
SELECT instruments.name AS instrument,
  count(musicians.id) as musicians
FROM instruments
  JOIN musicians ON musicians.main_instrument = instruments.id
GROUP BY instruments.name; 
```

`GROUP BY`子句指定哪些列或列描述了输出表中的每一行代表的内容。不在`GROUP BY`子句中的输出列必须使用聚合函数将其减少到单个值。

在这种情况下，我们使用`count()`聚合函数来计算与每种乐器关联的音乐家记录总数。其输出如下所示：

| instrument | musicians |
| --- | --- |
| drums | 1 |
| keyboards | 1 |
| bass | 1 |
| guitar | 2 |

标准 SQL 包含更多聚合函数，如 `min()`、`max()` 和 `sum()`，并且大多数 SQL 实现也扩展了它们自己的函数。

多对一和一对多连接并不能涵盖数据库需要建模的每一种可能情况；很多时候，需要多对多关系。

为了演示多对多连接，让我们创建一个新的表 `bands`，如下所示：

```py
CREATE TABLE bands (id SERIAL PRIMARY KEY, name TEXT NOT NULL);
INSERT INTO bands(name)
VALUES ('ABWH'), ('ELP'), ('King Crimson'), ('Pink Floyd'), ('Yes'); 
```

一个乐队有多位音乐家，而音乐家也可以属于多个乐队。我们如何创建音乐家和乐队之间的关系？如果我们向 `musicians` 表中添加一个 `band` 字段，这将限制每位音乐家只能属于一个乐队。如果我们向 `band` 表中添加一个 `musician` 字段，这将限制每个乐队只能有一位音乐家。为了建立这种联系，我们需要创建一个 **连接表**，其中每一行代表一位音乐家在乐队中的成员资格。

创建 `musicians_bands` 表如下所示：

```py
CREATE TABLE musicians_bands (
  musician_id INT REFERENCES musicians(id),
  band_id INT REFERENCES bands(id),
  PRIMARY KEY (musician_id, band_id)
);
INSERT INTO musicians_bands(musician_id, band_id)
VALUES (1, 3), (2, 2), (3, 2), (3, 3),
  (4, 1), (4, 2), (4, 5), (5,4); 
```

`musicians_bands` 表仅包含两个外键字段，一个指向音乐家的 ID，另一个指向乐队的 ID。

注意，我们不是创建或指定一个字段作为主键，而是使用这两个字段的组合作为主键。如果有多个行具有相同的两个值，那就没有意义，所以这种组合可以作为一个可接受的主键。

要编写使用这种关系的查询，我们的 `FROM` 子句需要指定两个 `JOIN` 语句：一个是从 `musicians` 到 `musicians_bands`，另一个是从 `bands` 到 `musicians_bands`。

例如，让我们获取每位音乐家所属乐队的名称：

```py
SELECT musicians.name, array_agg(bands.name) AS bands
FROM musicians
  JOIN musicians_bands ON musicians.id = musicians_bands.musician_id
  JOIN bands ON bands.id = musicians_bands.band_id
GROUP BY musicians.name
ORDER BY musicians.name ASC; 
```

这个查询通过连接表将音乐家和乐队关联起来，然后显示音乐家的名字以及他们所属乐队的聚合列表，并按音乐家的名字排序。它给出了以下输出：

| name | bands |
| --- | --- |
| 比尔·布鲁福德 | {ABWH,"King Crimson",Yes} |
| 大卫·吉尔莫 | {"Pink Floyd"} |
| 格雷格·莱克 | {ELP,"King Crimson"} |
| 吉思·艾默森 | {ELP} |
| 罗伯特·弗里普 | {"King Crimson"} |

这里使用的 `array_agg()` 函数将字符串值聚合到一个数组结构中。这种方法以及 `ARRAY` 数据类型是 PostgreSQL 特有的。

虽然大多数 SQL 实现都有针对聚合字符串值的解决方案，但并没有 SQL 标准函数用于聚合字符串值。

## 管理事务

虽然我们可以在单个 SQL 查询中完成很多数据操作，但有时一个更改需要多个查询。在这些情况下，如果其中一个查询失败，整个查询集必须被撤销，否则数据会被破坏。

例如，假设我们想在 `instruments` 表中插入 `'Vocals'` 作为值，但希望它是 ID #1。为此，我们首先需要将 `instruments` 表中的其他 ID 值向上移动一位，调整 `musicians` 表中的外键值，然后添加新行。查询将如下所示：

```py
UPDATE instruments SET id=id+1;
UPDATE musicians SET main_instrument=main_instrument+1;
INSERT INTO instruments(id, name) VALUES (1, 'Vocals'); 
```

在这个例子中，所有三个查询都必须成功运行才能产生我们想要的变化，而且至少前两个必须运行以避免数据损坏。如果只有第一个查询运行了，我们的数据就会损坏。

为了安全地完成这项操作，我们需要使用一个**事务**。

在 PostgreSQL 中使用事务涉及三个关键字，如下所示：

| 关键字 | 功能 |
| --- | --- |
| `BEGIN` | 开始一个事务 |
| `ROLLBACK` | 取消事务并重新开始 |
| `COMMIT` | 永久保存事务 |

要将我们的查询放入事务中，我们只需在查询之前添加 `BEGIN`，之后添加 `COMMIT`，如下所示：

```py
**BEGIN****;**
UPDATE instruments SET id=id+1;
UPDATE musicians SET main_instrument=main_instrument+1;
INSERT INTO instruments(id, name) VALUES (1, 'Vocals');
**COMMIT****;** 
```

现在，如果我们的查询中任何一个出现问题，我们可以执行一个 `ROLLBACK` 语句将数据库回滚到我们调用 `BEGIN` 时的状态。

在我们 *第十二章* 中使用的 `psycopg2` 模块等 DBAPI2 兼容模块中，事务管理通常是通过连接设置隐式处理的，或者通过连接对象方法显式处理，而不是使用 SQL 语句。

# 学习更多

这只是一个 SQL 概念和语法的快速概述；我们涵盖了您编写简单数据库应用程序所需了解的大部分内容，但还有更多需要学习。PostgreSQL 手册，可在 [`www.postgresql.org/docs/manuals`](https://www.postgresql.org/docs/manuals) 找到，是 SQL 语法和 PostgreSQL 特定功能的优秀资源。
