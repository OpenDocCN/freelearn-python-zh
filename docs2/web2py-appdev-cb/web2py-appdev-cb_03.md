# 第三章。数据库抽象层

在本章中，我们将介绍以下食谱：

+   创建新模型

+   从 csv 文件创建模型

+   批量上传你的数据

+   将你的数据从一个数据库迁移到另一个数据库

+   从现有的 MySQL 和 PostgreSQL 数据库创建模型

+   通过标签高效搜索

+   从多个应用程序访问你的数据库

+   层次分类树

+   按需创建记录

+   或者，LIKE，BELONGS，以及更多关于 Google App Engine 的内容

+   用 DB 视图替换慢速虚拟字段

# 简介

**数据库抽象层（DAL）**可能是 web2py 的主要优势。DAL 向底层的 SQL 语法暴露了一个简单的**应用程序编程接口（API）**，这可能会隐藏其真正的力量。在本章的食谱中，我们提供了 DAL 的非平凡应用示例，例如构建高效按标签搜索的查询和构建层次分类树。

# 创建新模型

如前一章的食谱所示，大多数应用程序都需要数据库，构建数据库模型是应用程序设计的第一步。

## 准备工作

在这里我们假设你有一个新创建的应用程序，你将把模型放入一个名为`models/db_custom.py`的文件中。

## 如何做到这一点...

1.  首先，你需要一个数据库连接。这是由 DAL 对象创建的。例如：

    ```py
    db = DAL('sqlite://storage.sqlite')

    ```

    注意，这一行已经存在于`models/db.py`文件中，因此你可能不需要它，除非你删除了它或需要连接到不同的数据库。默认情况下，web2py 连接到存储在文件存储.sqlite 中的`sqlite`数据库。此文件位于应用程序的数据库文件夹中。如果该文件不存在，则在应用程序首次执行时由 web2py 创建。

    SQLite 速度快，并且将所有数据存储在一个单独的文件中。这意味着你的数据可以轻松地从应用程序转移到另一个应用程序。实际上，`sqlite`数据库（们）是由 web2py 与应用程序一起打包的。它提供了完整的 SQL 支持，包括翻译、连接和聚合。此外，SQLite 从 Python 2.5 及以后的版本开始就自带了，因此，它已经包含在你的 web2py 安装中了。

    SQLite 有两个缺点。一个是它不强制执行列类型，除了添加和删除列之外，没有`ALTER TABLE`。另一个缺点是任何需要写访问权限的事务都会锁定整个数据库。因此，数据库除了读取之外不能并发访问。

    这些特性使其成为开发目的和低流量网站的不错选择，但不是高流量网站的可行解决方案。

    在下面的食谱中，我们将向你展示如何连接到不同类型的数据库。

1.  一旦我们有了`db`对象，我们就可以使用`define_table`方法来定义新表。例如：

    ```py
    db.define_table('invoice',Field('name'))

    ```

    语法始终相同。第一个参数是**表名**，其后跟一个字段列表。字段构造函数接受以下参数：

    +   **字段名**

    +   **字段类型**：这可以接受以下数据类型之一的值 - `string`（默认），`text, boolean, integer, double, password, date, time, datetime, upload, blob, reference other_table, list:string, list:integer`，和 `list:reference other_table`。内部，`upload, password` 和 `list` 类型等同于 `string`，但在 web2py 级别，它们被不同地处理。

    +   `length=512:` 这是基于字符串的字段的最大长度。对于非文本字段，此值被忽略。

    +   `default=None:` 这是插入新记录时的默认值。此属性的值可以是当需要值时调用的函数（例如，在记录插入时，如果没有指定值）。

    +   `update=None:` 这与默认值相同，但仅在更新时使用该值，而不是在插入时使用。

    +   `ondelete='CASCADE':` 这映射到相应的 SQL `ON DELETE` 属性。

    +   `notnull=False:` 这指定字段值是否可以是 `NULL`（在数据库级别强制执行）。

    +   `unique=False:` 这指定字段值是否必须是唯一的（在数据库级别强制执行）。

    +   `requires=[]:` 这是一组 web2py 验证器（在 web2py 表单级别强制执行）。大多数字段类型都有默认验证器。

    +   `required=False:` 这不要与 requires 混淆，并且它告诉 web2py 在插入和更新期间必须指定此字段的值。对于 `required` 字段，默认值和更新值将被忽略。除非与 `notnull=True` 一起使用，否则即使字段是必需的，`None` 值也是可接受的。

    +   `readable=True:` 这指定字段在表单中是否可读。

    +   `writable=True:` 这指定字段在表单中是否可写。

    +   `represent=(lambda value: value):` 这是一个用于在表单和表中显示字段值的函数。

    +   `widget=SQLHTML.widgets.string.widget:` 这是一个将在表单中构建输入小部件的函数。

    +   `label="Field Name":` 这是用于表单中此字段的标签。

    +   `comment="...":` 这是在表单中添加到该字段的注释。

        `Field` 构造函数还有其他特定于上传类型字段的属性。有关更多信息，请参阅 web2py 书籍。

1.  `define_table` 方法还接受三个命名参数：

    ```py
    db.define_table('....',
    	migrate=True,
    	fake_migrate=False,
    	format='%(id)s')

    ```

    +   `migrate=True:` 这指示 web2py 在不存在时创建表，或在它不匹配模型定义时修改它。此过程伴随着元数据文件的创建。元数据文件的形式为 `databases/<hash>_<name>.table`，并将用于跟踪模型的变化，并执行自动迁移。将 `migrate=False` 设置为禁用自动迁移。

    +   `fake_migrate=False:` 有时上述元数据会损坏（或意外删除），需要重新创建。如果模型与数据库表内容匹配，则设置`fake_migrate=True`，web2py 将重新构建元数据。

    +   `format='%(id)s':` 这是一个格式化字符串，它决定了当其他表在表单（例如在选择下拉框中）中引用此表的记录时应如何表示。格式可以是一个函数，它接受一个行对象并返回一个字符串。

## 还有更多...

在所有数据库中，但 SQLite 和 Google App Engine 数据存储，如果您更改表定义，则会发出`ALTER TABLE`以确保数据库与模型匹配。在 SQLite 中，只有在添加或删除列时才会执行`ALTER TABLE`，而不是当字段类型更改时（因为 SQLite 不强制执行）。在 Google App Engine 数据存储中，没有`ALTER TABLE`的概念，可以添加列但不能删除；web2py 将忽略模型中未列出的列。

完全从模型中删除`define_table`不会导致`DROP TABLE`。该表只是直到相应的`define_table`被放回，对 web2py 不可访问。这防止了数据的意外删除。您可以使用`db.<name>.drop()`命令在 web2py 中删除表。

# 从 CSV 文件创建模型

考虑这样一个场景：您有一个 CSV 文件，您对它知之甚少。但您仍然想创建一个 Web 应用程序来访问 CSV 文件中的数据。

## 准备工作

我将假设您有一个 csv 文件在文件夹中

`/tmp/mydata.csv`

您还需要一个名为`csvstudio`的程序，您可以从[`csvstudio.googlecode.com/hg/csvstudio.py`](http://csvstudio.googlecode.com/hg/csvstudio.py)下载。

## 如何操作...

1.  第一步是查看 csv 文件：

    ```py
    python csvstudio.py -a < /tmp/mydata.csv

    ```

    +   如果文件没有损坏，并且是标准的 csv 格式，那么 csvstudio 将生成一个报告，列出 CSV 列、数据类型和数据范围。

        如果文件是非标准的 CSV 格式，或者例如是 XLS 格式，尝试在 Excel 中导入它，然后再以 CSV 格式保存。

        您还可能想尝试使用**Google Refine**来清理 CSV 文件。

1.  一旦您知道`csvstudio`可以正确读取文件，运行以下命令：

    ```py
    python csvstudio.py -w mytable -i /tmp/mydata.csv > db1.py

    ```

    +   csvstudio 创建一个名为 db1.py 的文件，其中包含一个与数据兼容的 web2py 模型。在这里，mytable 是您为表选择的名称。

1.  将此文件移动到您的应用程序的`models`文件夹中。

1.  现在您需要清理数据，以便可以在 web2py 中导入。

    ```py
    python csvstudio.py -f csv -i /tmp/mydata.csv -o /tmp/mydata2.csv	

    ```

    +   文件`mydata2.csv`现在包含与原始文件相同的数据，但列名已被清理以与生成的模型兼容。字段值已去除任何前导和尾随空格。

1.  到目前为止，您只需运行您的应用程序并调用`appadmin`。

    ```py
    http://.../app/appadmin

    ```

1.  你应该能看到你生成的模型。点击模型名称，你将在底部看到一个上传链接。上传`mydata2.csv`文件以填充你的表格。

## 还有更多...

如果你更喜欢从 shell 上传 csv 文件而不是使用`appadmin`界面，你可以这样做。

从主 web2py 文件夹内部，运行以下命令：

```py
python web2py.py -S app -M -N

```

你将得到一个 web2py shell（-S app 在应用程序上下文中打开 shell，-M 加载模型，-N 防止 cron 作业运行）。

在 shell 内部执行以下操作：

```py
 >>> f = open('/tmp/mydata2.csv','rb')
>>> db.mytable.import_from_csv_file(f)
>>> db.commit()

```

嘿，数据已经在数据库中了。当你使用 shell 时，别忘了执行`db.commit()`。

如果由于任何原因这不起作用（可能是因为 CSV 文件是非标准的，无法进行标准化），请尝试按照我们的下一个食谱操作。

# 批量上传你的数据

在这里，我们假设你有一个已知`结构`的平面文件中的数据。你想要创建一个数据库模型并将数据导入数据库。

## 准备工作

为了简化，我们假设文件位于/tmp/data.txt，具有以下结构：

```py
Clayton Troncoso|234523
Malinda Gustavson|524334
Penelope Sharpless|151555
Serena Ruggerio|234565
Lenore Marbury|234656
Amie Orduna|256456
Margery Koeppel|643124
Loraine Merkley|234555
Avis Bosserman|234523
...
Elinor Erion|212554

```

每一行都是一个以`\n`结尾的记录。字段由&mdash;分隔。第一列包含`<first name> <last name>`。第二列包含年薪值。

如同往常，我们假设你有一个名为`app`的新应用程序。

## 如何做...

1.  你首先需要做的是在你的`app`中创建一个名为`models/db1.py`的模型，包含以下数据：

    ```py
    db.define_table('employees',
    	Field('first_name'),
    	Field('last_name'),
    	Field('salary','double'))

    ```

1.  然后，你会编写一个脚本，例如：

```py
applications/app/private/importer.py

```

+   此脚本可以读取数据，解析它，并将其放入数据库中，如下所示：

    ```py
    for line in open('/tmp/data.txt','r'):
    	fullname,salary = line.strip().split('|')
    	first_name,last_name = fullname.split(' ')
    	db.employees.insert(first_name=first_name,
    		last_name=last_name,
    		salary=float(salary))
    db.commit()

    ```

+   最后，从 web2py 文件夹运行以下脚本：

```py
python web2py.py -S app -M -N -R applications/app/private/
importer.py

```

注意，导入器是一个 Python 脚本，而不是一个模块（这就是为什么我们把它放在`private`文件夹而不是`modules`文件夹中。它在我们的应用程序上下文中执行，就像是一个控制器。实际上，你可以将代码复制到一个控制器中，并通过浏览器运行它。

## 还有更多...

如果数据是干净的，前面的脚本运行良好。你可能需要在插入之前验证每个记录。这又是一个两步过程。首先，你需要向你的模型添加验证器，例如：

```py
db.define_table('employees',
	Field('first_name', requires=IS_NOT_EMPTY()),
	Field('last_name', requires=SI_NOT_EMPTY()),
	Field('salary','double', requires=IS_FLOAT_IN_RANGE(0,10**7)))

```

然后你需要调用导入时的验证器并检查错误：

```py
for line in open('/tmp/data.txt','r'):
	fullname,salary = line.strip().split('|')
	first_name,last_name = fullname.split(' ')
	r = db.employee.validate_and_insert(
		first_name=first_name,
		last_name=last_name,
		salary=float(salary))
if r.errors: print line, r.errors
	db.commit()

```

导致错误的记录将不会被插入，你可以手动处理它们。

# 将数据从一个数据库迁移到另一个数据库

因此，到目前为止，你已经构建了你的应用程序，并且你的 SQLite 数据库中有数据。但是假设你需要迁移到生产 MySQL 或 PostgreSQL 环境。

## 准备工作

假设你有一个名为`app`的应用程序，数据在`sqlite://storage.sqlite`数据库中，你想要将数据迁移到不同的数据库：

```py
mysql://username:password@hostname:port/dbname

```

## 如何做...

1.  编辑你的模型`db.py`，并替换以下内容：

    ```py
    db=DAL('sqlite://storage.sqlite')

    ```

    使用以下：

    ```py
    production=False
    URI = 'mysql://username:password@hostname:port/dbname'
    if production:
    	db=DAL(URI, pool_size=20)
    else:
    	db=DAL('sqlite://storage.sqlite')

    ```

1.  创建一个名为`applications/app/private/mover.py`的文件，包含以下数据：

    ```py
    def main():
    	other_db = DAL(URI)
    	print 'creating tables...'
    	for table in db:
    		other_db.define_table(table._tablename,*[field for field in
    			table])
    	print 'exporting data...'
    	db.export_to_csv_file(open('tmp.sql','wb'))
    	print 'importing data...'
    	other_db.import_from_csv_file(open('tmp.sql','rb'))
    	other_db.commit()
    	print 'done!'
    if __name__() == "__main__":
    	main()

    ```

1.  使用以下命令运行此文件（只运行一次，否则你会得到重复的记录）：

    ```py
    python web2py.py -S app -M -N -R applications/app/private/mover.py

    ```

1.  修改模型`db.py`，并修改以下：

    ```py
    production=False

    ```

    到以下：

    ```py
    production=True

    ```

### 还有更多...

实际上，web2py 附带以下脚本：

```py
script/cpdb.py

```

此脚本使用命令行选项执行任务和变体。阅读文件以获取更多信息。

# 从现有的 MySQL 和 PostgreSQL 数据库创建模型

通常需要从 web2py 应用程序访问现有的数据库。在某些条件下这是可能的。

## 准备工作

为了连接到现有的数据库，它必须是被支持的数据库。在撰写本文时，这包括**MySQL、PostgreSQL、MSSQL、DB2、Oracle、Informix、FireBase**和**Sybase**。您必须知道数据库类型（例如`mysql`或`postgres`），数据库名称（例如，`mydb`），以及数据库服务器运行的主机名和端口号（例如`mysql`的`127.0.0.1:3306`或`postgres`的`127.0.0.1:5432`）。您必须有一个有效的用户名和密码来访问数据库。总之，您必须知道以下 URI 字符串：

+   `mysql://username:password@127.0.0.1:3306/mydb`

+   `postgres://username:password@127.0.0.1:5432/mydb`

假设您可以连接到此数据库，您只能访问满足以下条件的表：

+   每个要访问的表都必须有一个唯一的自增整数主键（无论是否称为`id`）。对于 PostgreSQL，您也可以有复合主键（由多个字段组成），并且不一定必须是`SERIAL`类型（参见 web2py 书籍中的**键表**）。

+   记录必须通过其主键进行引用。

+   web2py 模型必须为每个要访问的表包含一个`define_table`语句，列出所有字段及其类型。

在以下内容中，我们还将假设您的系统支持使用`mysql`命令本地访问数据库（以提取 MySQL 模型），或者您的系统已安装了`psycopg2` Python 模块（以提取 PostgreSQL 模型，请参阅安装说明）。

## 如何做到这一点...

1.  首先，您需要查询数据库，并制定一个与数据库内容兼容的可能模型。这可以通过运行以下随 web2py 提供的脚本完成：

    +   要从 MySQL 数据库构建 web2py 模型，请使用：

        ```py
        python scripts/extract_mysql_models.py username:password@databasename > db1.py

        ```

    +   要从 PostgreSQL 数据库构建 web2py 模型，请使用：

    ```py
    python scripts/extract_pgsql_models.py databasename localhost 5432 username password > db1.py

    ```

    这些脚本并不完美，但它们将生成一个 db1.py 文件，描述数据库表。

1.  编辑此模型以删除您不需要访问的表。改进字段类型（例如，字符串字段可能是密码），并添加验证器。

1.  然后将此文件移动到您的应用程序的`models/`文件夹中。

1.  最后，编辑原始的`db.py`模型，并用此数据库的 URI 字符串替换它。

    +   对于 MySQL，请编写：

    ```py
    db = DAL('mysql://username:password@127.0.0.1:8000/databasename',
    	migrate_enabled=False, pool_size=20)

    ```

    +   对于 PostgreSQL，请编写：

    ```py
    db = DAL(
    	"postgres://username:password@localhost:5432/databasename", migrate_enabled=False, pool_size=10)
    	migrate = False # you can control migration per define_table

    ```

    我们禁用了所有迁移，因为表已经存在，web2py 不应该尝试创建或修改它。

不幸的是，访问现有的数据库是 web2py 中最棘手的任务之一，因为数据库不是由 web2py 创建的，web2py 需要做出一些猜测。解决这些问题的唯一方法是手动编辑模型文件，并使用对数据库内容的独立知识。

## 更多...

实际上，`extract_pgsql_models.py`还具有以下附加功能：

+   它使用 ANSI 标准`INFORMATION_SCHEMA`（这可能与其他 RDBMS 一起工作）。

+   它检测具有`id`作为其主键的键表（没有`id`）。

+   它直接连接到正在运行的数据库，因此不需要进行 SQL 转储。

+   它处理`notnull, unique`和引用约束。

+   它检测最常见的数据类型和默认值。

+   它支持 PostgreSQL 列注释（即，用于文档）。

如果您必须使用它来对抗支持 ANSI `INFORMATION_SCHEMA`的其他 RDBMS（例如，MSSQL Server），则导入并使用适当的 Python 连接器，并删除特定的`postgreSQL`查询（pg_ `tables`用于注释）。

### 注意

您不能在普通自增主键表（type='id'）和键表（primarykey=['field1',`'field2'`])之间混合引用。如果您在数据库中使用两者，您必须在 web2py 模型中将键表手动定义为自增主键（移除`id`类型，并将主键参数添加到`define_table`）。

# 高效的标签搜索

无论您是在构建社交网络、内容管理系统还是 ERP 系统，您最终都需要记录标记的能力。这个配方向您展示了一种通过标签高效搜索记录的方法。

## 准备工作

在这里，我们假设以下两个模型：

1.  包含数据的模型：

    ```py
    db.define_table('data', Field('value'))

    ```

1.  存储标签的模型：

    ```py
    db.define_table('tag', Field('record_id', db.data), Field('name'))

    ```

这里，`name`是标签名称。

## 如何做到这一点...

1.  我们想搜索列表中至少有一个标签的所有记录：

    ```py
    tags = [...]

    ```

    为了这个目的，我们创建了一个搜索函数：

    ```py
    def search_or(data=db.data, tag=db.tag, tags=[]):
    	rows = db(data.id==tag.record_id)\ (tag.name.belongs(tags)).select(
    		data.ALL,
    		orderby=data.id,
    		groupby=data.id,
    		distinct=True)
    	return rows

    ```

1.  同样，如果您想搜索具有所有标签的记录（而不是列表中的任何一个）：

    ```py
    def search_and(data=db.data,tag=db.tag,tags=[]):
    	n = len(tags):
    	rows = db(data.id==tag.record_id)\
    		(tag.name.belongs(tags)).select(
    		data.ALL,
    		orderby=data.id,
    		groupby=data.id,
    		having=data.id.count()==n)
    	return rows

    ```

注意，这两个函数适用于任何作为第一个参数传递的表。

在这两个函数中，查询涉及两个表。

```py
data.id==tag.record_id

```

web2py 将其解释为连接。

## 更多...

这个系统如果用户可以自由选择标签名称，效果很好。有时，您可能希望将标签限制在定义良好的集合中。在这种情况下，模型需要更新：

```py
db.define_table('data', Field('value'))
db.define_table('tag', Field('name', unique=True))
db.define_table('link', Field('record_id',db.data), Field('tag_id',db.
tag))

```

这里，链接表实现了数据记录和标签项之间的多对多关系。

在这种情况下，我们需要修改我们的搜索函数，因此首先我们将标签名称列表（tags）转换为标签 ID 列表，然后执行之前的查询。这可以通过使用`subquery:`来完成。

```py
def search_or(data=db.data, tag=db.tag,link=db.link,tags=[]):
	subquery = db(db.tag.name.belongs(tags)).select(db.tag.id)
	rows = db(data.id==link.record_id)\
	(link.tag_id.belongs(subquery)).select(
		data.ALL,
		orderby=data.id,
		groupby=data.id,
		distinct=True)
	return rows
def search_and(data=db.data, tag=db.tag, link=db.link, tags=[]):
	n = len(tags)
	subquery = db(db.tag.name.belongs(tags)).select(db.tag.id)
	rows = db(data.id==link.record_id)\
		(link.tag_id.belongs(subquery)).select(
			data.ALL,
			orderby=data.id,
			groupby=data.id,
			having=data.id.count()==n)
	return rows

```

我们在这里实施的技术被称为**Toxi**方法，并在以下链接中以更通用和抽象的方式描述：

[`www.pui.ch/phred/archives/2005/04/tags-database-schemas.html`](http://www.pui.ch/phred/archives/2005/04/tags-database-schemas.html).

# 从多个应用程序访问你的数据库

构建分布式应用程序的一种方法，是让多个应用程序可以访问相同的数据库。不幸的是，这不仅仅是连接到数据库的问题。实际上，不同的应用程序需要了解表内容和其他元数据，这些元数据存储在模型定义中。

有三种方法可以实现这一点，它们并不等价。这取决于应用程序是否共享文件系统，以及你希望给予两个应用程序多少自主权。

## 准备工作

我们假设你已经有两个 web2py 应用程序，一个叫做`app1`，另一个叫做`app2`，其中`app1`通过以下方式连接到数据库：

```py
db = DAL(URI)

```

在这里，URI 是一些连接字符串。无论是 SQLite 还是客户端/服务器数据库，这都无关紧要。我们还将假设`app1`使用的模型存储在`models/db1.py`中，尽管这里的名称并不重要。

现在我们希望`app2`连接到同一个数据库。

## 如何操作...

这也是一个常见的场景，你希望两个应用程序是自主的，尽管能够共享数据。**自主**意味着你希望能够独立分发每个应用程序，而无需另一个应用程序。

如果是这样的话，每个应用程序都需要自己的模型副本和自己的数据库元数据。实现这一点的唯一方法是通过代码的复制。

你必须遵循以下步骤：

1.  编辑`app2`的 URI 字符串，使其看起来与`app1`相同，但禁用迁移：

    ```py
    db = DAL(URI, migrate_enabled=False)

    ```

1.  将`app1`中的模型文件`models/d1.py`复制到`app2`中。

注意，只有`app1`能够执行迁移（如果两个都能做，情况会变得非常混乱）。如果你在`app1`中更改模型，你必须再次复制模型文件。

虽然这个解决方案打破了**不要重复自己**（DRY）模式，但它保证了每个应用程序的完全自主性，即使它们在不同的服务器上运行，也可以访问相同的数据库。

如果两个应用程序运行在同一台服务器上，你不需要复制模型文件，只需创建一个符号链接即可：

```py
ln applications/app1/models/db1.py applications/app2/models/db1.py

```

现在你只有一个模型文件。

## 还有更多...

有时候你需要一个脚本（而不是一个网络应用程序）来访问 web2py 模型。这可以通过仅访问元数据来实现，而不需要执行实际的模型文件。

这里有一个可以做到这一点的 Python 脚本（而不是 web2py 模型）：

```py
# file myscript.py
from gluon.dal import DAL
db = DAL(URI, folder='/path/to/web2py/applications/app1', auto_
import=True)
print db.tables
# add your code here

```

注意`auto_import=True`。它告诉 DAL 在指定的文件夹中查找与 URI 连接关联的元数据，并在内存中动态重建模型。以这种方式定义的模型具有正确的名称和字段类型，但它们将不具有其他属性的正确值，例如可读性、可写性、默认值、验证器等。这是因为这些属性不能在元数据中进行序列化，并且在这个场景中可能也不需要。

# 层次化分类树

任何应用程序迟早都需要一种对数据进行分类的方法，并且类别必须以树的形式存储，因为每个类别都有一个父类，可能还有子类别。没有子类别的类别是树的叶子。如果有没有父类的类别，我们创建一个虚构的根树节点，并将它们作为根的子类别附加。

主要问题是如何在数据库表中存储具有父子关系的类别，以及高效地添加节点和查询节点的祖先和后代。

这可以通过修改的先序树遍历算法来实现，如下所述。

## 如何操作...

关键技巧是将每个节点存储在其自己的记录中，带有两个整数属性，左和右，这样所有祖先的左属性都小于或等于当前节点的左属性，而右属性大于当前节点的右属性。同样，所有后代都将有一个大于或等于当前左的左属性和小于当前右的右属性。在公式中：

如果 `A.ileft<=B.ileft` 且 `A.iright>B.iright`，则 `A` 是 `B` 的父类。

注意到 `A.iright - A.ileft` 总是后代的数量。

以下是一个可能的实现：

```py
from gluon.dal import Table

class TreeProxy(object):
	skeleton = Table(None,'tree',
		Field('ileft','integer'),
		Field('iright','integer'))
	def __init__(self,table):
		self.table=table
	def ancestors(self,node):
		db = self.table._db
	return
		db(self.table.ileft<=node.ileft)(self.table.iright>node.iright)
	def descendants(self,node):
		db = self.table._db
	return
		db(self.table.ileft>=node.ileft)(self.table.iright<node.iright)
	  def add_leaf(self,parent_id=None,**fields):
		if not parent_id:
			nrecords = self.table._db(self.table).count()
			fields.update(dict(ileft=nrecords,iright=nrecords))
		else:
			node = self.table(parent_id)
			fields.update(dict(ileft=node.iright,iright=node.iright))
			node.update_record(iright=node.iright+1)
			ancestors = self.ancestors(node).select()
		for ancestor in ancestors:
			ancestor.update_record(iright=ancestor.iright+1)
			ancestors = self.ancestors(node).select()
		for ancestor in ancestors:
			ancestor.update_record(iright=ancestor.iright+1)
	return self.table.insert(**fields)

	  def del_node(self,node):
		delta = node.iright-node.ileft
		deleted = self.descendants(node).delete()
		db = self.table._db
		db(self.table.iright>node.iright).
			update(iright=self.table.iright-delta)
		del self.table[node.id]
	return deleted + 1

```

这允许我们执行以下操作：

+   定义自己的树表（mytree）和代理对象（treeproxy）：

    ```py
    treeproxy =
    	TreeProxy(db.define_table('mytree',Field('name'),Tree.skeleton))

    ```

+   插入一个新的节点：

    ```py
    id = treeproxy.add_leaf(name="root")

    ```

+   添加一些节点：

    ```py
    treeproxy.add_leaf(parent_id=id,name="child1")
    treeproxy.add_leaf(parent_id=id,name="child2")

    ```

+   搜索祖先和后代：

    ```py
    for node in treeproxy.ancestors(db.tree(id)).select():
    	print node.name
    for node in treeproxy.descendants(db.tree(id)).select():
    print node.name

    ```

+   删除一个节点及其所有后代：

    ```py
    treeproxy.del_node(db.tree(id))

    ```

# 按需创建记录

我们通常需要根据条件获取或更新记录，但记录可能不存在。如果记录不存在，我们希望创建它。在这个菜谱中，我们将展示两个可以满足此目的的实用函数：

+   `get_or_create`

+   `update_or_create`

为了使这可行，我们需要传递足够的 `field:value` 对来创建缺失的记录。

## 如何操作...

1.  这里是 `get_or_create` 的代码：

    ```py
    def get_or_create(table, **fields):
    	"""
    	Returns record from table with passed field values.
    	Creates record if it does not exist.
    	'table' is a DAL table reference, such as 'db.invoice'
    	fields are field=value pairs
    	"""
    	return table(**fields) or table.insert(**fields)

    ```

    注意如何通过表(**字段)选择与请求字段匹配的记录，如果记录不存在则返回 None。在这种情况下，将插入记录。然后，table.insert(...) 返回插入记录的引用，对于实际目的来说，就是获取刚刚插入的记录。

1.  这里是一个使用示例：

    ```py
    db.define_table('person', Field('name'))
    john = get_or_create(db.person, name="John")

    ```

1.  `update_or_create` 的代码非常相似，但我们需要两组变量&mdash; 用于 **搜索**（在更新之前）的变量和用于 **更新**的变量：

    ```py
    def update_or_create(table, fields, updatefields):
    	"""
    	Modifies record that matches 'fields' with 'updatefields'.
    	If record does not exist then create it.

    	'table' is a DAL table reference, such as 'db.person'
    	'fields' and 'updatefields' are dictionaries
    	"""
    	row = table(**fields)
    	if row:
    		row.update_record(**updatefields)
    	else:
    		fields.update(updatefields)
    		row = table.insert(**fields)
    	return row

    ```

1.  这里是一个使用示例：

    ```py
    tim = update_or_create(db.person, dict(name="tim"),
    dict(name="Tim"))

    ```

# OR、LIKE、BELONGS 以及更多在 Google App Engine 上的应用

**Google App Engine** (GAE) 的一个主要限制是无法执行使用 OR、BELONGS(IN) 和 LIKE 操作符的查询。

web2py DAL 提供了一个抽象数据库查询的系统，它不仅适用于 **关系数据库** (RDBS)，也适用于 GAE，但仍然受到前面提到的限制。这里我们展示了些解决方案。

我们创建了一个额外的 API，允许在从 GAE 存储中提取记录后在 web2py 层面上合并、过滤和排序记录。它们可以用来模拟缺失的功能，并将您的 GAE 代码也移植到 RDBS。

当前支持的 RDBS 是 SQLite、MySQL、PostgreSQL、MSSQL、DB2、Informix、Oracle、FireBird 和 Ingres。

GAE 是目前唯一支持的 NoDB。其他适配器正在开发中。

## 准备工作

在以下菜谱中，我们计划开发一个在 GAE 上运行的应用程序，并使用以下逻辑连接到数据库：

```py
if request.env.web2py_runtime_gae:
	db = DAL('google:datastore')
else:
	db = DAL('sqlite://storage.sqlite')

```

我们假设以下模型作为示例：

```py
product = db.define_table('product',
	Field('name'),
	Field('price','double'))

buyer = db.define_table('buyer',
	Field('name'))

purchase = db.define_table('purchase',
	Field('product',db.product),
	Field('buyer',db.buyer),
	Field('quantity','integer'),
	Field('order_date','date',default=request.now))

```

## 如何做到这一点...

在设置我们之前描述的 GAE 模型之后，让我们看看如何在以下部分中执行插入和更新记录、执行连接和其他操作。

### 记录插入

为了测试其余的代码，您可能想在表中插入一些记录。您可以使用 `appadmin` 或以编程方式完成此操作。以下代码在 GAE 上运行良好，但有警告，即 `insert` 方法返回的 ID 在 GAE 上不是顺序的：

```py
icecream = db.product.insert(name='Ice Cream',price=1.50)
kenny = db.buyer.insert(name='Kenny')
cartman = db.buyer.insert(name='Cartman')
db.purchase.insert(product=icecream,buyer=kenny,quantity=1,
	order_date=datetime.datetime(2009,10,10))
db.purchase.insert(product=icecream,buyer=cartman,quantity=4,
	order_date=datetime.datetime(2009,10,11))

```

### 记录更新

GAE 上的 `update` 操作与您预期的正常操作一样。两种语法都受支持：

```py
icecream.update_record(price=1.99)

```

以及：

```py
icecream.price=1.99
icecream.update_record()

```

### 连接

在关系数据库中，您可以执行以下操作：

```py
rows = db(purchase.product==product.id)
	(purchase.buyer==buyer.id).select()
for row in rows:
	print row.product.name, row.product.price,
	row.buyer.name, row.purchase.quantity

```

这会产生以下结果：

```py
Ice Cream 1.99 Kenny 1
Ice Cream 1.99 Cartman 4

```

这在 GAE 上不起作用。您必须在不使用连接的情况下执行查询，使用递归 `selects`。

```py
rows = db(purchase.id>0).select()
for row in rows:
	print row.product.name, row.product.price, row.buyer.name,
	row.quantity

```

在这里，`row.product.name` 执行递归 `selects`，并获取由 `row.product.` 引用的产品的名称。

### 逻辑 OR

在 RDBS 上，您可以使用 &mdash; 操作符在查询中实现 `OR`：

```py
rows = db((purchase.buyer==kenny)|(purchase.buyer==cartman)).select()

```

这在 GAE 上不起作用，因为不支持 `OR` 操作（在撰写本文时）。如果查询涉及相同的字段，可以使用 `IN` 操作符：

```py
rows = db(purchase.buyer.contains((kenny,cartman))).select()

```

这是一个便携且高效的解决方案。在最一般的情况下，您可能需要在 web2py 层面上而不是在数据库层面上执行 `OR` 操作。

```py
rows_kenny = db(purchase.buyer==kenny).select()
rows_cartman = db(purchase.buyer==cartman).select()
rows = rows_kenny|rows_cartman

```

在这种后一种情况下，`&mdash;` 不是在查询之间，而是在行对象之间，并且是在记录检索之后执行的。这带来了一些问题，因为原始顺序丢失了，并且由于增加了内存和资源消耗的惩罚。

### 带有 `orderby` 的 `OR`

在关系数据库中，您可以执行以下操作：

```py
rows = db((purchase.buyer==kenny)|(purchase.buyer==cartman))\
	.select(orderby=purchase.quantity)

```

但是，再次在 GAE 上，您必须在 web2py 层面上执行 `OR` 操作。因此，您还必须在 web2py 层面上进行排序：

```py
rows_kenny = db(purchase.buyer==kenny).select()
rows_cartman = db(purchase.buyer==cartman).select()
rows = (rows_kenny|rows_cartman).sort(lambda row:row.quantity)

```

`rows` 对象的 `sort` 方法接受一个行函数，并必须返回一个用于排序的表达式。它们也可以与 RDBS 一起使用以实现排序，当表达式过于复杂而无法在数据库级别实现时。

### 带有复杂 `orderby` 的 `OR`

考虑以下涉及 `OR`、`JOIN` 和排序的查询，并且仅在 RDBS 上工作：

```py
rows = db((purchase.buyer==kenny)|(purchase.buyer==cartman))\
	(purchase.buyer==buyer.id).select(orderby=buyer.name)

```

您可以使用 `sort` 方法以及 `sort` 参数中的递归 `select` 来重写它：

```py
rows = (rows_kenny|rows_cartman).sort( \
	lambda row:row.buyer.name)

```

这可以工作，但可能效率不高。您可能希望缓存 `row.buyer` 到 `buyer_names:` 的映射。

```py
buyer_names = cache.ram('buyer_names',
	lambda:dict(*[(b.id,b.name) for b in db(db.buyer).select()]),
	3600)
rows = (rows_kenny|rows_cartman).sort(
	lambda row: buyer_names.get(row.buyer,row.buyer.name))

```

在这里，`buyer_names` 是 `ids` 和 `names` 之间的映射，并且每小时（3600 秒）缓存一次。`sort` 尝试从 `buyer_names` 中选择名称，如果可能的话，否则执行递归选择。

### LIKE

在关系数据库中，例如，你可以搜索所有以字母 `C` 开头后跟任何内容（%）的记录：

```py
rows = db(buyer.name.like('C%')).select()
print rows

```

但 GAE 既不支持全文搜索，也不支持类似 SQL `LIKE` 操作符的任何内容。再一次，我们必须选择所有记录并在 web2py 层面上执行过滤。我们可以使用 `rows` 对象的 `find` 方法：

```py
rows = db(buyer.id>0).select().find(lambda
	row:row.name.startswith('C'))

```

当然，这很昂贵，不推荐用于大型表（超过几百条记录）。如果这种搜索对你的应用程序至关重要，也许你不应该使用 GAE。

### 日期和日期时间操作

对于涉及其他表达式（如日期和日期时间操作）的查询，也会出现相同的问题。考虑以下在关系数据库上工作但在 GAE 上不工作的查询：

```py
rows = db(purchase.order_date.day==11).select()

```

在 GAE 上，你必须将其重写如下：

```py
rows = db(purchase.id>0).select().find(lambda
	row:row.order_date.day==11)

```

# 用数据库视图替换慢速虚拟字段

考虑以下表：

```py
db.define_table('purchase', Field('product'),
	Field('price', 'double'),
	Field('quantity','integer'))

```

你需要添加一个字段，称为 `total price`，在检索记录时计算，定义为每个记录的价格乘以数量。

正常的做法是使用 **虚拟字段**：

```py
class MyVirtualFields(object):
	def total_price(self):
		return self.purchase.price * self.purchase.quantity
db.purchase.virtualfields.append(MyVirtualFields())

```

然后，你可以执行以下操作：

```py
for row in db(db.purchase).select():
	print row.name, row.total_price

```

这是可以的，但在 web2py 层面上计算虚拟字段可能会很慢。此外，你可能不习惯在查询中涉及虚拟字段。

这里我们提出了一种替代方案，该方案涉及为表创建一个数据库视图，该视图包括包含计算字段的列，并为 web2py 提供了访问它的方式。

## 如何做...

给定表，执行以下操作：

```py
if not db.executesql("select * from information_schema.tables where
table_name='purchase_plus' limit 1;"):
	db.executesql("create view purchase_plus as select purchase.*,
		purchase.price * purchase.quantity as total_price from purchase")
db.define_table('purchase_plus', db.purchase, Field('total_price',
	'double'),
	migrate=False)

```

现在，你可以在任何使用 `db.numbers_plus` 的地方使用 `db.purchase_plus`，除了插入操作，与 `VirtualFields` 解决方案相比，性能有所提升。

## 它是如何工作的...

以下行检查视图是否已经创建：

```py
if not db.executesql("select ...")

```

如果没有，它指示数据库创建它：

```py
db.executesql("create view ...")

```

最后，它定义了一个新的 web2py 模型，该模型映射到表：

```py
db.define_table('purchase_plus',...)

```

这个模型包括 `db.purchase` 表中的所有字段，新的字段 `total_price`，并将 `migrate=False` 设置为，这样 web2py 就不会尝试创建表（它不应该尝试创建，因为这个不是新表，而是一个视图，并且已经创建）。

## 还有更多...

注意，并非所有支持的数据库都支持视图，并且并非所有支持视图的数据库都有 `information_schema.tables`。因此，这个菜谱不能保证在所有支持的数据库上都能工作，并且会使你的应用程序不可移植。
