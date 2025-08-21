# 第二十一章：高级数据库管理

本章提供了有关 Django 中支持的每个关系数据库的额外信息，以及连接到传统数据库的注意事项和技巧。

# 一般注意事项

Django 尝试在所有数据库后端上支持尽可能多的功能。然而，并非所有的数据库后端都是一样的，Django 开发人员必须对支持哪些功能和可以安全假设的内容做出设计决策。

本文件描述了一些可能与 Django 使用相关的特性。当然，它并不打算替代特定服务器的文档或参考手册。

## 持久连接

持久连接避免了在每个请求中重新建立与数据库的连接的开销。它们由`CONN_MAX_AGE`参数控制，该参数定义了连接的最大生存期。它可以独立设置每个数据库。默认值为 0，保留了在每个请求结束时关闭数据库连接的历史行为。要启用持久连接，请将`CONN_MAX_AGE`设置为正数秒数。要获得无限的持久连接，请将其设置为`None`。

### 连接管理

Django 在首次进行数据库查询时会打开与数据库的连接。它会保持这个连接打开，并在后续请求中重用它。一旦连接超过`CONN_MAX_AGE`定义的最大寿命，或者不再可用，Django 会关闭连接。

具体来说，Django 在需要连接数据库时会自动打开一个连接，如果没有已经存在的连接，要么是因为这是第一个连接，要么是因为上一个连接已经关闭。

在每个请求开始时，如果连接已经达到最大寿命，Django 会关闭连接。如果您的数据库在一段时间后终止空闲连接，您应该将`CONN_MAX_AGE`设置为较低的值，这样 Django 就不会尝试使用已被数据库服务器终止的连接。（这个问题可能只影响非常低流量的站点。）

在每个请求结束时，如果连接已经达到最大寿命或处于不可恢复的错误状态，Django 会关闭连接。如果在处理请求时发生了任何数据库错误，Django 会检查连接是否仍然有效，如果无效则关闭连接。因此，数据库错误最多影响一个请求；如果连接变得无法使用，下一个请求将获得一个新的连接。

### 注意事项

由于每个线程都维护自己的连接，因此您的数据库必须支持至少与您的工作线程一样多的同时连接。

有时，数据库不会被大多数视图访问，例如，因为它是外部系统的数据库，或者由于缓存。在这种情况下，您应该将`CONN_MAX_AGE`设置为较低的值，甚至为`0`，因为维护一个不太可能被重用的连接是没有意义的。这将有助于保持对该数据库的同时连接数较小。

开发服务器为每个处理的请求创建一个新的线程，从而抵消了持久连接的效果。在开发过程中不要启用它们。

当 Django 建立与数据库的连接时，它会根据所使用的后端设置适当的参数。如果启用了持久连接，这个设置就不会在每个请求中重复。如果您修改了连接的隔离级别或时区等参数，您应该在每个请求结束时恢复 Django 的默认设置，或者在每个请求开始时强制设置适当的值，或者禁用持久连接。

## 编码

Django 假设所有数据库都使用 UTF-8 编码。使用其他编码可能会导致意外行为，例如数据库对 Django 中有效的数据产生值过长的错误。有关如何正确设置数据库的信息，请参阅以下特定数据库的注意事项。

# postgreSQL 注意事项

Django 支持 PostgreSQL 9.0 及更高版本。它需要使用 Psycopg2 2.0.9 或更高版本。

## 优化 postgreSQL 的配置

Django 需要其数据库连接的以下参数：

+   `client_encoding`: `'UTF8'`,

+   `default_transaction_isolation`: 默认为`'read committed'`，或者连接选项中设置的值（见此处），

+   `timezone`: 当`USE_TZ`为`True`时为`'UTC'`，否则为`TIME_ZONE`的值。

如果这些参数已经具有正确的值，Django 不会为每个新连接设置它们，这会稍微提高性能。您可以直接在`postgresql.conf`中配置它们，或者更方便地通过`ALTER ROLE`为每个数据库用户配置它们。

Django 在没有进行此优化的情况下也可以正常工作，但每个新连接都会执行一些额外的查询来设置这些参数。

## 隔离级别

与 PostgreSQL 本身一样，Django 默认使用`READ COMMITTED`隔离级别。如果需要更高的隔离级别，如`REPEATABLE READ`或`SERIALIZABLE`，请在`DATABASES`中的数据库配置的`OPTIONS`部分中设置它：

```py
import psycopg2.extensions 

DATABASES = { 
    # ... 
    'OPTIONS': { 
        'isolation_level': psycopg2.extensions.ISOLATION_LEVEL_SERIALIZABLE, 
    }, 
} 

```

在更高的隔禅级别下，您的应用程序应该准备好处理由于序列化失败而引发的异常。此选项设计用于高级用途。

## varchar 和 text 列的索引

在模型字段上指定`db_index=True`时，Django 通常会输出一个`CREATE INDEX`语句。但是，如果字段的数据库类型为`varchar`或`text`（例如，由`CharField`，`FileField`和`TextField`使用），那么 Django 将创建一个使用适当的 PostgreSQL 操作符类的额外索引。额外的索引是必要的，以正确执行使用`LIKE`操作符的查找，这在它们的 SQL 中使用`contains`和`startswith`查找类型时会发生。

# MySQL 注意事项

## 版本支持

Django 支持 MySQL 5.5 及更高版本。

Django 的`inspectdb`功能使用包含所有数据库模式详细数据的`information_schema`数据库。

Django 期望数据库支持 Unicode（UTF-8 编码）并委托给它执行事务和引用完整性的任务。重要的是要意识到，当使用 MyISAM 存储引擎时，MySQL 实际上并不执行这两个任务，详见下一节。

## 存储引擎

MySQL 有几种存储引擎。您可以在服务器配置中更改默认存储引擎。

直到 MySQL 5.5.4，默认引擎是 MyISAM。MyISAM 的主要缺点是它不支持事务或强制外键约束。另一方面，直到 MySQL 5.6.4，它是唯一支持全文索引和搜索的引擎。

自 MySQL 5.5.5 以来，默认存储引擎是 InnoDB。该引擎完全支持事务，并支持外键引用。这可能是目前最好的选择。但是，请注意，InnoDB 自增计数器在 MySQL 重新启动时会丢失，因为它不记住`AUTO_INCREMENT`值，而是将其重新创建为`max(id)+1`。这可能导致`AutoField`值的意外重用。

如果您将现有项目升级到 MySQL 5.5.5，然后添加一些表，请确保您的表使用相同的存储引擎（即 MyISAM vs. InnoDB）。特别是，如果在它们之间具有`ForeignKey`的表使用不同的存储引擎，那么在运行`migrate`时可能会看到以下错误：

```py
_mysql_exceptions.OperationalError: ( 
    1005, "Can't create table '\\db_name\\.#sql-4a8_ab' (errno: 150)" 
) 

```

## MySQL DB API 驱动程序

Python 数据库 API 在 PEP 249 中有描述。MySQL 有三个实现此 API 的知名驱动程序：

+   MySQLdb（[`pypi.python.org/pypi/MySQL-python/1.2.4`](https://pypi.python.org/pypi/MySQL-python/1.2.4)）是由 Andy Dustman 开发和支持了十多年的本地驱动程序。

+   mySQLclient ([`pypi.python.org/pypi/mysqlclient`](https://pypi.python.org/pypi/mysqlclient))是`MySQLdb`的一个分支，特别支持 Python 3，并且可以作为 MySQLdb 的替代品。在撰写本文时，这是使用 Django 与 MySQL 的推荐选择。

+   MySQL Connector/Python ([`dev.mysql.com/downloads/connector/python`](http://dev.mysql.com/downloads/connector/python))是来自 Oracle 的纯 Python 驱动程序，不需要 MySQL 客户端库或标准库之外的任何 Python 模块。

所有这些驱动程序都是线程安全的，并提供连接池。`MySQLdb`是目前唯一不支持 Python 3 的驱动程序。

除了 DB API 驱动程序，Django 还需要一个适配器来访问其 ORM 中的数据库驱动程序。Django 为 MySQLdb/mysqlclient 提供了一个适配器，而 MySQL Connector/Python 则包含了自己的适配器。

### mySQLdb

Django 需要 MySQLdb 版本 1.2.1p2 或更高版本。

如果在尝试使用 Django 时看到`ImportError: cannot import name ImmutableSet`，则您的 MySQLdb 安装可能包含一个过时的`sets.py`文件，与 Python 2.4 及更高版本中同名的内置模块发生冲突。要解决此问题，请验证您是否安装了 MySQLdb 版本 1.2.1p2 或更新版本，然后删除 MySQLdb 目录中由早期版本留下的`sets.py`文件。

MySQLdb 将日期字符串转换为 datetime 对象时存在已知问题。具体来说，值为`0000-00-00`的日期字符串对于 MySQL 是有效的，但在 MySQLdb 中会被转换为`None`。

这意味着在使用可能具有`0000-00-00`值的行的 loaddata/dumpdata 时，您应该小心，因为它们将被转换为`None`。

在撰写本文时，最新版本的 MySQLdb（1.2.4）不支持 Python 3。要在 Python 3 下使用 MySQLdb，您需要安装`mysqlclient`。

### mySQLclient

Django 需要 mysqlclient 1.3.3 或更高版本。请注意，不支持 Python 3.2。除了 Python 3.3+支持外，mysqlclient 应该与 MySQLdb 大致相同。

### mySQL connector/python

MySQL Connector/Python 可从下载页面获取。Django 适配器可在 1.1.X 及更高版本中获取。它可能不支持最新版本的 Django。

## 时区定义

如果您打算使用 Django 的时区支持，请使用`mysql_tzinfo_to_sql`将时区表加载到 MySQL 数据库中。这只需要针对您的 MySQL 服务器执行一次，而不是每个数据库。

## 创建您的数据库

您可以使用命令行工具和以下 SQL 创建您的数据库：

```py
CREATE DATABASE <dbname> CHARACTER SET utf8; 

```

这可以确保所有表和列默认使用 UTF-8。

### 校对设置

列的校对设置控制数据排序的顺序以及哪些字符串比较相等。它可以在数据库范围内设置，也可以在每个表和每个列上设置。这在 MySQL 文档中有详细说明。在所有情况下，您都可以通过直接操作数据库表来设置校对；Django 不提供在模型定义中设置这一点的方法。

默认情况下，对于 UTF-8 数据库，MySQL 将使用`utf8_general_ci`校对。这导致所有字符串相等比较以*不区分大小写*的方式进行。也就是说，"`Fred`"和"`freD`"在数据库级别被视为相等。如果在字段上有唯一约束，尝试将"`aa`"和"`AA`"插入同一列将是非法的，因为它们比较为相等（因此不唯一）。

在许多情况下，这个默认值不会有问题。但是，如果您真的想在特定列或表上进行区分大小写的比较，您将更改列或表以使用`utf8_bin`排序规则。在这种情况下要注意的主要事情是，如果您使用的是 MySQLdb 1.2.2，则 Django 中的数据库后端将为从数据库接收到的任何字符字段返回字节串（而不是 Unicode 字符串）。这与 Django *始终*返回 Unicode 字符串的正常做法有很大的不同。

由您作为开发人员来处理这样一个事实，即如果您配置表使用`utf8_bin`排序规则，您将收到字节串。Django 本身应该大部分可以顺利地处理这样的列（除了这里描述的`contrib.sessions``Session`和`contrib.admin``LogEntry`表），但是您的代码必须准备在必要时调用“django.utils.encoding.smart_text（）”，如果它真的想要处理一致的数据-Django 不会为您做这个（数据库后端层和模型填充层在内部是分开的，因此数据库层不知道它需要在这一个特定情况下进行这种转换）。

如果您使用的是 MySQLdb 1.2.1p2，Django 的标准`CharField`类将即使使用`utf8_bin`排序规则也返回 Unicode 字符串。但是，`TextField`字段将作为`array.array`实例（来自 Python 的标准`array`模块）返回。Django 对此无能为力，因为再次，当数据从数据库中读取时，所需的信息不可用。这个问题在 MySQLdb 1.2.2 中得到了解决，因此，如果您想要在`utf8_bin`排序规则下使用`TextField`，则升级到 1.2.2 版本，然后按照之前描述的处理字节串（这不应该太困难）是推荐的解决方案。

如果您决定在 MySQLdb 1.2.1p2 或 1.2.2 中使用`utf8_bin`排序规则来处理一些表，您仍应该为`django.contrib.sessions.models.Session`表（通常称为`django_session`）和`django.contrib.admin.models.LogEntry`表（通常称为`django_admin_log`）使用`utf8_general_ci`（默认值）排序规则。请注意，根据 MySQL Unicode 字符集，`utf8_general_ci`排序规则的比较速度更快，但比`utf8_unicode_ci`排序规则稍微不正确。如果这对您的应用程序是可以接受的，您应该使用`utf8_general_ci`，因为它更快。如果这是不可接受的（例如，如果您需要德语字典顺序），请使用`utf8_unicode_ci`，因为它更准确。

### 注意

模型表单集以区分大小写的方式验证唯一字段。因此，在使用不区分大小写的排序规则时，具有仅大小写不同的唯一字段值的表单集将通过验证，但在调用“save（）”时，将引发`IntegrityError`。

## 连接到数据库

连接设置按以下顺序使用：

+   `OPTIONS`

+   `NAME`，`USER`，`PASSWORD`，`HOST`，`PORT`

+   MySQL 选项文件

换句话说，如果在`OPTIONS`中设置了数据库的名称，这将优先于`NAME`，这将覆盖 MySQL 选项文件中的任何内容。以下是一个使用 MySQL 选项文件的示例配置：

```py
# settings.py 
DATABASES = { 
    'default': { 
        'ENGINE': 'django.db.backends.mysql', 
        'OPTIONS': {'read_default_file': '/path/to/my.cnf',}, 
    } 
} 

# my.cnf 
[client] 
database = NAME 
user = USER 
password = PASSWORD 
default-character-set = utf8 

```

其他一些 MySQLdb 连接选项可能会有用，例如`ssl`，`init_command`和`sql_mode`。请参阅 MySQLdb 文档以获取更多详细信息。

## 创建您的表

当 Django 生成模式时，它不指定存储引擎，因此表将使用数据库服务器配置的默认存储引擎创建。

最简单的解决方案是将数据库服务器的默认存储引擎设置为所需的引擎。

如果您使用托管服务并且无法更改服务器的默认存储引擎，则有几种选择。

+   创建表后，执行`ALTER TABLE`语句将表转换为新的存储引擎（例如 InnoDB）：

```py
        ALTER TABLE <tablename> ENGINE=INNODB; 

```

+   如果您有很多表，这可能会很麻烦。

+   另一个选项是在创建表之前使用 MySQLdb 的`init_command`选项：

```py
        'OPTIONS': { 
           'init_command': 'SET storage_engine=INNODB', 
        } 

```

这将在连接到数据库时设置默认存储引擎。创建表后，应删除此选项，因为它会向每个数据库连接添加一个仅在表创建期间需要的查询。

## 表名

即使在最新版本的 MySQL 中，也存在已知问题，可能会在特定条件下执行某些 SQL 语句时更改表名的情况。建议您尽可能使用小写表名，以避免可能由此行为引起的任何问题。Django 在自动生成模型的表名时使用小写表名，因此，如果您通过`db_table`参数覆盖表名，则主要考虑这一点。

## 保存点

Django ORM 和 MySQL（使用 InnoDB 存储引擎时）都支持数据库保存点。

如果使用 MyISAM 存储引擎，请注意，如果尝试使用事务 API 的保存点相关方法，您将收到数据库生成的错误。原因是检测 MySQL 数据库/表的存储引擎是一项昂贵的操作，因此决定不值得根据此类检测结果动态转换这些方法为无操作。

## 特定字段的注意事项

### 字符字段

如果您对字段使用`unique=True`，则存储为`VARCHAR`列类型的任何字段的`max_length`将限制为 255 个字符。这会影响`CharField`，`SlugField`和`CommaSeparatedIntegerField`。

### 时间和日期时间字段的分数秒支持

MySQL 5.6.4 及更高版本可以存储分数秒，前提是列定义包括分数指示（例如，`DATETIME(6)`）。早期版本根本不支持它们。此外，早于 1.2.5 的 MySQLdb 版本存在一个错误，也会阻止与 MySQL 一起使用分数秒。

如果数据库服务器支持，Django 不会将现有列升级以包括分数秒。如果要在现有数据库上启用它们，您需要手动更新目标数据库上的列，例如执行以下命令：

```py
ALTER TABLE `your_table` MODIFY `your_datetime_column` DATETIME(6) 

```

或在`数据迁移`中使用`RunSQL`操作。

默认情况下，使用 mysqlclient 或 MySQLdb 1.2.5 或更高版本在 MySQL 5.6.4 或更高版本上创建新的`DateTimeField`或`TimeField`列时现在支持分数秒。

### 时间戳列

如果您使用包含`TIMESTAMP`列的旧数据库，则必须将`USE_TZ = False`设置为避免数据损坏。`inspectdb`将这些列映射到`DateTimeField`，如果启用时区支持，则 MySQL 和 Django 都将尝试将值从 UTC 转换为本地时间。

### 使用 Queryset.Select_For_Update()进行行锁定

MySQL 不支持`SELECT ... FOR UPDATE`语句的`NOWAIT`选项。如果使用`select_for_update()`并且`nowait=True`，则会引发`DatabaseError`。

### 自动类型转换可能导致意外结果

在对字符串类型执行查询时，但具有整数值时，MySQL 将在执行比较之前将表中所有值的类型强制转换为整数。如果您的表包含值"`abc`"，"`def`"，并且您查询`WHERE mycolumn=0`，则两行都将匹配。同样，`WHERE mycolumn=1`将匹配值"`abc1`"。因此，在 Django 中包含的字符串类型字段在使用它进行查询之前将始终将该值转换为字符串。

如果您实现了直接继承自`Field`的自定义模型字段，正在覆盖`get_prep_value()`，或使用`extra()`或`raw()`，则应确保执行适当的类型转换。

# SQLite 注意事项

SQLite 为主要是只读或需要较小安装占用空间的应用程序提供了一个优秀的开发替代方案。然而，与所有数据库服务器一样，SQLite 具有一些特定于 SQLite 的差异，您应该注意。

## 子字符串匹配和区分大小写

对于所有 SQLite 版本，在尝试匹配某些类型的字符串时，会出现一些略微反直觉的行为。这些行为在 Querysets 中使用`iexact`或`contains`过滤器时会触发。行为分为两种情况：

1.  对于子字符串匹配，所有匹配都是不区分大小写的。也就是说，过滤器`filter（name__contains="aa"）`将匹配名称为“Aabb”的名称。

1.  对于包含 ASCII 范围之外字符的字符串，所有精确的字符串匹配都是区分大小写的，即使在查询中传递了不区分大小写的选项。因此，在这些情况下，`iexact`过滤器的行为将与精确过滤器完全相同。

这些问题的一些可能的解决方法在 sqlite.org 上有记录，但默认的 Django SQLite 后端没有使用它们，因为将它们整合起来可能会相当困难。因此，Django 暴露了默认的 SQLite 行为，您在进行不区分大小写或子字符串过滤时应该注意这一点。

## 旧的 SQLite 和 CASE 表达式

SQLite 3.6.23.1 及更早版本在处理包含`ELSE`和算术的`CASE`表达式中的查询参数时存在一个错误。

SQLite 3.6.23.1 于 2010 年 3 月发布，大多数不同平台的当前二进制发行版都包含了更新版本的 SQLite，但值得注意的是 Python 2.7 的 Windows 安装程序除外。

截至目前，Windows-Python 2.7.10 的最新版本包括 SQLite 3.6.21。您可以安装`pysqlite2`或将`sqlite3.dll`（默认安装在`C:\Python27\DLLs`中）替换为来自 sqlite.org 的更新版本以解决此问题。

## 使用更新版本的 SQLite DB-API 2.0 驱动程序

如果发现可用的话，Django 将优先使用`pysqlite2`模块而不是 Python 标准库中附带的`sqlite3`。

如果需要，这提供了升级 DB-API 2.0 接口或 SQLite 3 本身到比特定 Python 二进制发行版中包含的版本更新的能力。

## 数据库被锁定的错误

SQLite 旨在成为一个轻量级的数据库，因此无法支持高并发。`OperationalError: database is locked`错误表明您的应用程序正在经历比`sqlite`默认配置中可以处理的并发更多的情况。这个错误意味着一个线程或进程在数据库连接上有一个独占锁，另一个线程在等待锁被释放时超时了。

Python 的 SQLite 包装器具有默认的超时值，确定第二个线程在锁上等待多长时间才会超时并引发`OperationalError: database is locked`错误。

如果您遇到此错误，您可以通过以下方法解决：

+   切换到另一个数据库后端。在某一点上，SQLite 对于真实世界的应用程序来说变得太轻，这些并发错误表明您已经达到了这一点。

+   重写您的代码以减少并发并确保数据库事务的持续时间较短。

+   通过设置`timeout`数据库选项来增加默认超时值：

```py
        'OPTIONS': { # ... 'timeout': 20, # ... } 

```

这只会使 SQLite 在抛出数据库被锁定错误之前等待更长的时间；它实际上并不会真正解决这些问题。

### queryset.Select_For_Update()不支持

SQLite 不支持`SELECT ... FOR UPDATE`语法。调用它不会产生任何效果。

### 原始查询中不支持 pyformat 参数样式

对于大多数后端，原始查询（`Manager.raw()`或`cursor.execute()`）可以使用 pyformat 参数样式，其中查询中的占位符为`'%(name)s'`，参数作为字典而不是列表传递。SQLite 不支持这一点。

### 连接.queries 中未引用的参数

`sqlite3`不提供在引用和替换参数后检索 SQL 的方法。相反，在`connection.queries`中的 SQL 将使用简单的字符串插值重新构建。这可能是不正确的。在将查询复制到 SQLite shell 之前，请确保在必要的地方添加引号。

# Oracle 注意事项

Django 支持 Oracle 数据库服务器版本 11.1 及更高版本。需要版本 4.3.1 或更高版本的`cx_Oracle`（[`cx-oracle.sourceforge.net/`](http://cx-oracle.sourceforge.net/)）Python 驱动程序，尽管我们建议使用版本 5.1.3 或更高版本，因为这些版本支持 Python 3。

请注意，由于`cx_Oracle` 5.0 中存在 Unicode 损坏错误，因此不应该使用该驱动程序的该版本与 Django 一起使用；`cx_Oracle` 5.0.1 解决了此问题，因此如果您想使用更新的`cx_Oracle`，请使用版本 5.0.1。

`cx_Oracle` 5.0.1 或更高版本可以选择使用`WITH_UNICODE`环境变量进行编译。这是推荐的，但不是必需的。

为了使`python manage.py migrate`命令工作，您的 Oracle 数据库用户必须具有运行以下命令的权限：

+   `CREATE TABLE`

+   `CREATE SEQUENCE`

+   `CREATE PROCEDURE`

+   `CREATE TRIGGER`

要运行项目的测试套件，用户通常需要这些*额外*权限：

+   `CREATE USER`

+   `DROP USER`

+   `CREATE TABLESPACE`

+   `DROP TABLESPACE`

+   `CREATE SESSION WITH ADMIN OPTION`

+   `CREATE TABLE WITH ADMIN OPTION`

+   `CREATE SEQUENCE WITH ADMIN OPTION`

+   `CREATE PROCEDURE WITH ADMIN OPTION`

+   `CREATE TRIGGER WITH ADMIN OPTION`

请注意，虽然`RESOURCE`角色具有所需的`CREATE TABLE`、`CREATE SEQUENCE`、`CREATE PROCEDURE`和`CREATE TRIGGER`权限，而且授予`RESOURCE WITH ADMIN OPTION`的用户可以授予`RESOURCE`，但这样的用户不能授予单个权限（例如`CREATE TABLE`），因此`RESOURCE WITH ADMIN OPTION`通常不足以运行测试。

一些测试套件还会创建视图；要运行这些视图，用户还需要`CREATE VIEW WITH ADMIN OPTION`权限。特别是 Django 自己的测试套件需要这个权限。

所有这些权限都包含在 DBA 角色中，这适用于在私人开发人员的数据库上使用。

Oracle 数据库后端使用`SYS.DBMS_LOB`包，因此您的用户将需要对其具有执行权限。通常情况下，默认情况下所有用户都可以访问它，但如果不行，您将需要授予权限，如下所示：

```py
GRANT EXECUTE ON SYS.DBMS_LOB TO user; 

```

## 连接到数据库

要使用 Oracle 数据库的服务名称进行连接，您的`settings.py`文件应该如下所示：

```py
DATABASES = { 
    'default': { 
        'ENGINE': 'django.db.backends.oracle', 
        'NAME': 'xe', 
        'USER': 'a_user', 
        'PASSWORD': 'a_password', 
        'HOST': '', 
        'PORT': '', 
    } 
} 

```

在这种情况下，您应该将`HOST`和`PORT`都留空。但是，如果您不使用`tnsnames.ora`文件或类似的命名方法，并且希望使用 SID（在此示例中为`xe`）进行连接，那么请填写`HOST`和`PORT`如下：

```py
DATABASES = { 
    'default': { 
        'ENGINE': 'django.db.backends.oracle', 
        'NAME': 'xe', 
        'USER': 'a_user', 
        'PASSWORD': 'a_password', 
        'HOST': 'dbprod01ned.mycompany.com', 
        'PORT': '1540', 
    } 
} 

```

您应该同时提供`HOST`和`PORT`，或者将两者都留空。Django 将根据选择使用不同的连接描述符。

## 线程选项

如果您计划在多线程环境中运行 Django（例如，在任何现代操作系统上使用默认 MPM 模块的 Apache），那么您**必须**将 Oracle 数据库配置的`threaded`选项设置为 True：

```py
'OPTIONS': { 
    'threaded': True, 
}, 

```

未能这样做可能会导致崩溃和其他奇怪的行为。

## INSERT ... RETURNING INTO

默认情况下，Oracle 后端使用`RETURNING INTO`子句来高效地检索`AutoField`的值，当插入新行时。这种行为可能会导致某些不寻常的设置中出现`DatabaseError`，例如在远程表中插入，或者在具有`INSTEAD OF`触发器的视图中插入。

`RETURNING INTO`子句可以通过将数据库配置的`use_returning_into`选项设置为 False 来禁用：

```py
'OPTIONS': { 
    'use_returning_into': False, 
}, 

```

在这种情况下，Oracle 后端将使用单独的`SELECT`查询来检索`AutoField`值。

## 命名问题

Oracle 对名称长度有 30 个字符的限制。

为了适应这一点，后端将数据库标识符截断以适应，用可重复的 MD5 哈希值替换截断名称的最后四个字符。此外，后端将数据库标识符转换为全大写。

为了防止这些转换（通常仅在处理传统数据库或访问属于其他用户的表时才需要），请使用带引号的名称作为`db_table`的值：

```py
class LegacyModel(models.Model): 
    class Meta: 
        db_table = '"name_left_in_lowercase"' 

class ForeignModel(models.Model): 
    class Meta: 
        db_table = '"OTHER_USER"."NAME_ONLY_SEEMS_OVER_30"' 

```

带引号的名称也可以与 Django 的其他支持的数据库后端一起使用；但是，除了 Oracle 之外，引号没有任何效果。

在运行`migrate`时，如果将某些 Oracle 关键字用作模型字段的名称或`db_column`选项的值，则可能会遇到`ORA-06552`错误。 Django 引用所有在查询中使用的标识符，以防止大多数此类问题，但是当 Oracle 数据类型用作列名时，仍然可能发生此错误。特别要注意避免使用名称`date`，`timestamp`，`number`或`float`作为字段名称。

## NULL 和空字符串

Django 通常更喜欢使用空字符串（''“）而不是`NULL`，但是 Oracle 将两者视为相同。为了解决这个问题，Oracle 后端会忽略对具有空字符串作为可能值的字段的显式`null`选项，并生成 DDL，就好像`null=True`一样。在从数据库中获取数据时，假定这些字段中的`NULL`值实际上意味着空字符串，并且数据会被默默地转换以反映这一假设。

## Textfield 的限制

Oracle 后端将`TextField`存储为`NCLOB`列。 Oracle 对此类 LOB 列的使用施加了一些限制：

+   LOB 列不能用作主键。

+   LOB 列不能用于索引。

+   LOB 列不能在`SELECT DISTINCT`列表中使用。这意味着在包含`TextField`列的模型上尝试使用`QuerySet.distinct`方法将导致针对 Oracle 运行时出错。作为解决方法，使用`QuerySet.defer`方法与`distinct()`结合使用，以防止`TextField`列被包括在`SELECT DISTINCT`列表中。

# 使用第三方数据库后端

除了官方支持的数据库外，还有第三方提供的后端，允许您使用其他数据库与 Django 一起使用：

+   SAP SQL Anywhere

+   IBM DB2

+   Microsoft SQL Server

+   Firebird

+   ODBC

+   ADSDB

这些非官方后端支持的 Django 版本和 ORM 功能差异很大。关于这些非官方后端的具体功能以及任何支持查询，应该直接向每个第三方项目提供的支持渠道提出。

# 将 Django 与传统数据库集成

虽然 Django 最适合开发新应用程序，但完全可以将其集成到传统数据库中。Django 包括一些实用程序，以尽可能自动化这个过程。

设置好 Django 后，您将按照以下一般流程与现有数据库集成。

## 给 Django 提供您的数据库参数

您需要告诉 Django 您的数据库连接参数是什么，数据库的名称是什么。通过编辑`DATABASES`设置并为`'default'`连接分配值来完成这一点：

+   `NAME`

+   `ENGINE <DATABASE-ENGINE>`

+   `USER`

+   `PASSWORD`

+   `HOST`

+   `PORT`

## 自动生成模型

Django 带有一个名为`inspectdb`的实用程序，可以通过内省现有数据库来创建模型。您可以通过运行此命令查看输出：

```py
python manage.py inspectdb 

```

使用标准的 Unix 输出重定向将此保存为文件：

```py
python manage.py inspectdb > models.py 

```

此功能旨在作为快捷方式，而不是最终的模型生成。有关更多信息，请参阅`inspectdb`的文档。

清理模型后，将文件命名为`models.py`并将其放在包含您的应用程序的 Python 包中。然后将该应用程序添加到您的`INSTALLED_APPS`设置中。

默认情况下，`inspectdb`创建的是不受管理的模型。也就是说，在模型的`Meta`类中的`managed = False`告诉 Django 不要管理每个表的创建、修改和删除：

```py
class Person(models.Model): 
    id = models.IntegerField(primary_key=True) 
    first_name = models.CharField(max_length=70) 
    class Meta: 
       managed = False 
       db_table = 'CENSUS_PERSONS' 

```

如果你确实希望 Django 管理表的生命周期，你需要将前面的`managed`选项更改为`True`（或者简单地删除它，因为`True`是它的默认值）。

## 安装核心 Django 表

接下来，运行`migrate`命令来安装任何额外需要的数据库记录，比如管理员权限和内容类型：

```py
python manage.py migrate 

```

## 清理生成的模型

正如你所期望的那样，数据库内省并不完美，你需要对生成的模型代码进行一些轻微的清理。以下是处理生成模型的一些建议：

+   每个数据库表都转换为一个模型类（也就是说，数据库表和模型类之间是一对一的映射）。这意味着你需要将许多对多连接表的模型重构为`ManyToManyField`对象。

+   每个生成的模型都有一个属性对应每个字段，包括 id 主键字段。然而，要记住，如果一个模型没有主键，Django 会自动添加一个 id 主键字段。因此，你需要删除任何看起来像这样的行：

```py
        id = models.IntegerField(primary_key=True) 

```

+   这些行不仅是多余的，而且如果你的应用程序将向这些表中添加*新*记录，它们还会引起问题。

+   每个字段的类型（例如`CharField`、`DateField`）是通过查看数据库列类型（例如`VARCHAR`、`DATE`）来确定的。如果`inspectdb`无法将列的类型映射到模型字段类型，它将使用`TextField`，并在生成的模型中在字段旁边插入 Python 注释`'This field type is a guess.'`。留意这一点，如果需要，相应地更改字段类型。

+   如果数据库中的字段没有良好的 Django 等效项，你可以放心地将其删除。Django 模型层并不要求包含表中的每个字段。

+   如果数据库列名是 Python 保留字（比如`pass`、`class`或`for`），`inspectdb`会在属性名后面添加"`_field`"，并将`db_column`属性设置为真实字段名（例如`pass`、`class`或`for`）。

+   例如，如果一个表有一个名为`for`的`INT`列，生成的模型将有一个类似这样的字段：

```py
        for_field = models.IntegerField(db_column='for') 

```

+   `inspectdb`会在字段旁边插入 Python 注释`'Field renamed because it was a Python reserved word.'`。

+   如果你的数据库包含引用其他表的表（大多数数据库都是这样），你可能需要重新排列生成的模型的顺序，以便引用其他模型的模型被正确排序。例如，如果模型`Book`有一个指向模型`Author`的`ForeignKey`，模型`Author`应该在模型`Book`之前定义。如果需要在尚未定义的模型上创建关系，你可以使用包含模型名称的字符串，而不是模型对象本身。

+   `inspectdb`检测 PostgreSQL、MySQL 和 SQLite 的主键。也就是说，它会在适当的地方插入`primary_key=True`。对于其他数据库，你需要在每个模型中至少插入一个`primary_key=True`字段，因为 Django 模型需要有一个`primary_key=True`字段。

+   外键检测只适用于 PostgreSQL 和某些类型的 MySQL 表。在其他情况下，外键字段将被生成为`IntegerField`，假设外键列是一个`INT`列。

## 测试和调整

这些是基本步骤-从这里开始，你需要调整 Django 生成的模型，直到它们按照你的意愿工作。尝试通过 Django 数据库 API 访问数据，并尝试通过 Django 的管理站点编辑对象，并相应地编辑模型文件。

# 接下来是什么？

就是这样！

希望您喜欢阅读《精通 Django：核心》，并从这本书中学到了很多。虽然这本书将为您提供 Django 的完整参考，但没有什么能替代老实的实践-所以开始编码，祝您在 Django 职业生涯中一切顺利！

剩下的章节纯粹供您参考。它们包括附录和所有 Django 函数和字段的快速参考。
