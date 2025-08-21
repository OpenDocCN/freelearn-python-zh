# 第二章：使用 SQLAlchemy 的关系数据库

关系数据库是几乎每个现代 Web 应用程序构建的基石。学会以表和关系的方式思考你的应用程序是一个干净、设计良好的项目的关键之一。正如你将在本章中看到的，你早期选择的数据模型将影响代码的几乎每个方面。我们将使用 SQLAlchemy，一个强大的对象关系映射器，允许我们在 Python 内部直接与数据库交互，抽象出多个数据库引擎的复杂性。

在本章中，我们将：

+   简要概述使用关系数据库的好处

+   介绍 SQLAlchemy，Python SQL 工具包和对象关系映射器

+   配置我们的 Flask 应用程序使用 SQLAlchemy

+   编写一个模型类来表示博客条目

+   学习如何从数据库保存和检索博客条目

+   执行查询-排序、过滤和聚合

+   为博客条目构建标记系统

+   使用 Alembic 创建模式迁移

# 为什么使用关系数据库？

我们应用程序的数据库远不止是我们需要保存以备将来检索的东西的简单记录。如果我们只需要保存和检索数据，我们可以轻松地使用纯文本文件。事实上，我们希望能够对我们的数据执行有趣的查询。而且，我们希望能够高效地做到这一点，而不需要重新发明轮子。虽然非关系数据库（有时被称为 NoSQL 数据库）非常受欢迎，并且在 Web 世界中有其位置，但关系数据库早就解决了过滤、排序、聚合和连接表格数据的常见问题。关系数据库允许我们以结构化的方式定义数据集，从而保持数据的一致性。使用关系数据库还赋予我们开发人员自由，可以专注于我们应用程序中重要的部分。

除了高效执行特别查询外，关系数据库服务器还会执行以下操作：

+   确保我们的数据符合模式中规定的规则

+   允许多人同时访问数据库，同时保证底层数据的一致性

+   确保数据一旦保存，即使应用程序崩溃也不会丢失

关系数据库和 SQL，与关系数据库一起使用的编程语言，是值得一整本书来讨论的话题。因为这本书致力于教你如何使用 Flask 构建应用程序，我将向你展示如何使用一个被 Python 社区广泛采用的用于处理数据库的工具，即 SQLAlchemy。

### 注意

SQLAlchemy 抽象了许多编写 SQL 查询的复杂性，但深入理解 SQL 和关系模型是无法替代的。因此，如果你是 SQL 的新手，我建议你查看在线免费提供的色彩丰富的书籍*Learn SQL the Hard Way*，*Zed Shaw*，网址为[`sql.learncodethehardway.org/`](http://sql.learncodethehardway.org/)。

# 介绍 SQLAlchemy

SQLAlchemy 是一个在 Python 中处理关系数据库非常强大的库。我们可以使用普通的 Python 对象来表示数据库表并执行查询，而不是手动编写 SQL 查询。这种方法有许多好处，如下所示：

+   你的应用程序可以完全使用 Python 开发。

+   数据库引擎之间的微小差异被抽象掉了。这使你可以像使用轻量级数据库一样做事情，例如，在本地开发和测试时使用 SQLite，然后在生产环境中切换到为高负载设计的数据库（如 PostgreSQL）。

+   数据库错误更少，因为现在在你的应用程序和数据库服务器之间有两层：Python 解释器本身（这将捕捉明显的语法错误）和 SQLAlchemy，它有明确定义的 API 和自己的错误检查层。

+   由于 SQLAlchemy 的工作单元模型有助于减少不必要的数据库往返，所以您的数据库代码可能会变得更加高效。SQLAlchemy 还有用于高效预取相关对象的设施，称为急加载。

+   **对象关系映射**（**ORM**）使您的代码更易于维护，这是一种被称为**不要重复自己**（**DRY**）的愿望。假设您向模型添加了一个列。使用 SQLAlchemy，每当您使用该模型时，该列都将可用。另一方面，如果您在整个应用程序中手写 SQL 查询，您将需要逐个更新每个查询，以确保包含新列。

+   SQLAlchemy 可以帮助您避免 SQL 注入漏洞。

+   出色的库支持：正如您将在后面的章节中看到的，有许多有用的库可以直接与您的 SQLAlchemy 模型一起工作，提供诸如维护界面和 RESTful API 之类的功能。

希望您在阅读完这个列表后感到兴奋。如果这个列表中的所有项目现在对您来说都没有意义，不要担心。当您阅读本章和后续章节时，这些好处将变得更加明显和有意义。

现在我们已经讨论了使用 SQLAlchemy 的一些好处，让我们安装它并开始编码。

### 注意

如果您想了解更多关于 SQLAlchemy 的信息，在*开源应用程序的架构*中有一整章专门讨论了它的设计，可以免费在线阅读，网址是[`aosabook.org/en/sqlalchemy.html`](http://aosabook.org/en/sqlalchemy.html)。

## 安装 SQLAlchemy

我们将使用`pip`将 SQLAlchemy 安装到博客应用的虚拟环境中。正如您在上一章中所记得的，要激活您的虚拟环境，只需切换到`source`并执行`activate`脚本：

```py
$ cd ~/projects/blog
$ source blog/bin/activate
(blog) $ pip install sqlalchemy
Downloading/unpacking sqlalchemy
…
Successfully installed sqlalchemy
Cleaning up...

```

您可以通过打开 Python 解释器并检查 SQLAlchemy 版本来检查您的安装是否成功；请注意，您的确切版本号可能会有所不同。

```py
$ python
>>> import sqlalchemy
>>> sqlalchemy.__version__
'0.9.0b2'

```

## 在我们的 Flask 应用中使用 SQLAlchemy

SQLAlchemy 在 Flask 上运行得非常好，但 Flask 的作者发布了一个名为**Flask-SQLAlchemy**的特殊 Flask 扩展，它提供了许多常见任务的辅助功能，并可以避免我们以后不得不重新发明轮子。让我们使用`pip`来安装这个扩展：

```py
(blog) $ pip install flask-sqlalchemy
…
Successfully installed flask-sqlalchemy

```

Flask 为对构建扩展感兴趣的开发人员提供了一个标准接口。随着这个框架的流行，高质量的扩展数量也在增加。如果您想查看一些更受欢迎的扩展，可以在 Flask 项目网站上找到一个精选列表，网址是[`flask.pocoo.org/extensions/`](http://flask.pocoo.org/extensions/)。

## 选择数据库引擎

SQLAlchemy 支持多种流行的数据库方言，包括 SQLite、MySQL 和 PostgreSQL。根据您想要使用的数据库，您可能需要安装一个包含数据库驱动程序的额外 Python 包。下面列出了 SQLAlchemy 支持的一些流行数据库以及相应的 pip-installable 驱动程序。一些数据库有多个驱动程序选项，所以我首先列出了最流行的一个。

| 数据库 | 驱动程序包 |
| --- | --- |
| SQLite | 不需要，自 Python 2.5 版本起已包含在 Python 标准库中 |
| MySQL | MySQL-python, PyMySQL（纯 Python），OurSQL |
| PostgreSQL | psycopg2 |
| Firebird | fdb |
| Microsoft SQL Server | pymssql, PyODBC |
| Oracle | cx-Oracle |

SQLite 与 Python 一起标准提供，并且不需要单独的服务器进程，因此非常适合快速启动。在接下来的示例中，为了简单起见，我将演示如何配置博客应用以使用 SQLite。如果您有其他数据库想法，并且希望在博客项目中使用它，请随时使用`pip`在此时安装必要的驱动程序包。

## 连接到数据库

使用您喜欢的文本编辑器，打开我们博客项目（`~/projects/blog/app/config.py`）的`config.py`模块。我们将添加一个特定于 SQLAlchemy 的设置，以指示 Flask-SQLAlchemy 如何连接到我们的数据库。以下是新的行：

```py
import os
class Configuration(object):
 APPLICATION_DIR = os.path.dirname(os.path.realpath(__file__))
    DEBUG = True
 SQLALCHEMY_DATABASE_URI = 'sqlite:///%s/blog.db' % APPLICATION_DIR

```

`SQLALCHEMY_DATABASE_URI`包括以下部分：

`dialect+driver://username:password@host:port/database`

因为 SQLite 数据库存储在本地文件中，我们需要提供的唯一信息是数据库文件的路径。另一方面，如果您想连接到本地运行的 PostgreSQL，您的 URI 可能看起来像这样：

`postgresql://postgres:secretpassword@localhost:5432/blog_db`

### 注意

如果您在连接到数据库时遇到问题，请尝试查阅 SQLAlchemy 关于数据库 URI 的文档：[`docs.sqlalchemy.org/en/rel_0_9/core/engines.html`](http://docs.sqlalchemy.org/en/rel_0_9/core/engines.html)。

现在我们已经指定了如何连接到数据库，让我们创建一个负责实际管理我们数据库连接的对象。这个对象由 Flask-SQLAlchemy 扩展提供，并且方便地命名为`SQLAlchemy`。打开`app.py`并进行以下添加：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy

from config import Configuration

app = Flask(__name__)
app.config.from_object(Configuration)
db = SQLAlchemy(app)

```

这些更改指示我们的 Flask 应用程序，进而指示 SQLAlchemy 如何与我们应用程序的数据库通信。下一步将是创建一个用于存储博客条目的表，为此，我们将创建我们的第一个模型。

# 创建 Entry 模型

**模型**是我们想要存储在数据库中的数据表的数据表示。这些模型具有称为**列**的属性，表示数据中的数据项。因此，如果我们要创建一个`Person`模型，我们可能会有用于存储名字、姓氏、出生日期、家庭地址、头发颜色等的列。由于我们有兴趣创建一个模型来表示博客条目，我们将为标题和正文内容等内容创建列。

### 注意

请注意，我们不说`People`模型或`Entries`模型 - 即使它们通常代表许多不同的对象，模型是单数。

使用 SQLAlchemy，创建模型就像定义一个类并指定分配给该类的多个属性一样简单。让我们从我们博客条目的一个非常基本的模型开始。在博客项目的`app/`目录中创建一个名为`models.py`的新文件，并输入以下代码：

```py
import datetime, re
from app import db

def slugify(s):
    return re.sub('[^\w]+', '-', s).lower()

class Entry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    slug = db.Column(db.String(100), unique=True)
    body = db.Column(db.Text)
    created_timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    modified_timestamp = db.Column(
        db.DateTime,
        default=datetime.datetime.now, 
        onupdate=datetime.datetime.now)

    def __init__(self, *args, **kwargs):
        super(Entry, self).__init__(*args, **kwargs)  # Call parent constructor.
        self.generate_slug()

    def generate_slug(self):
        self.slug = ''
        if self.title:
            self.slug = slugify(self.title)

    def __repr__(self):
        return '<Entry: %s>' % self.title
```

有很多事情要做，所以让我们从导入开始，然后逐步进行。我们首先导入标准库`datetime`和`re`模块。我们将使用`datetime`获取当前日期和时间，使用`re`进行一些字符串操作。接下来的导入语句引入了我们在`app.py`中创建的`db`对象。您可能还记得，`db`对象是`SQLAlchemy`类的一个实例，它是 Flask-SQLAlchemy 扩展的一部分。`db`对象提供了访问我们需要构建`Entry`模型的类的功能，这只是前面几行。

在`Entry`模型之前，我们定义了一个辅助函数`slugify`，我们将使用它为我们的博客条目提供一些漂亮的 URL（在第三章中使用，*模板和视图*）。`slugify`函数接受一个字符串，比如*关于 Flask 的帖子*，并使用正则表达式将可读的字符串转换为 URL，因此返回*a-post-about-flask*。

接下来是`Entry`模型。我们的`Entry`模型是一个普通的类，扩展了`db.Model`。通过扩展`db.Model`，我们的`Entry`类将继承各种我们将用于查询数据库的帮助程序。

`Entry`模型的属性是我们希望存储在数据库中的名称和数据的简单映射，并列在下面：

+   `id`：这是我们数据库表的主键。当我们创建一个新的博客条目时，数据库会自动为我们设置这个值，通常是每个新条目的自增编号。虽然我们不会明确设置这个值，但当你想要引用一个模型到另一个模型时，主键会派上用场，这一点你将在本章后面看到。

+   `title`：博客条目的标题，存储为具有最大长度为 100 的`String`列。

+   `slug`：标题的 URL 友好表示，存储为具有最大长度为 100 的`String`列。该列还指定了`unique=True`，因此没有两个条目可以共享相同的 slug。

+   `body`：帖子的实际内容，存储在`Text`列中。这与`Title`和`Slug`的`String`类型不同，因为你可以在这个字段中存储任意多的文本。

+   `created_timestamp`：博客条目创建的时间，存储在`DateTime`列中。我们指示 SQLAlchemy 在首次保存条目时自动填充这一列的当前时间。

+   `modified_timestamp`：博客条目上次更新的时间。当我们保存一个条目时，SQLAlchemy 会自动使用当前时间更新这个列。

### 注意

对于标题或事物名称等短字符串，`String`列是合适的，但当文本可能特别长时，最好使用`Text`列，就像我们为条目正文所做的那样。

我们已经重写了类的构造函数（`__init__`），这样，当创建一个新模型时，它会根据标题自动为我们设置 slug。

最后一部分是`__repr__`方法，用于生成我们的`Entry`类实例的有用表示。`__repr__`的具体含义并不重要，但允许你在调试时引用程序正在处理的对象。

最后需要添加一小段代码到 main.py，这是我们应用程序的入口点，以确保模型被导入。将以下突出显示的更改添加到 main.py 中：

```py
from app import app, db
import models
import views

if __name__ == '__main__':
    app.run()
```

## 创建 Entry 表

为了开始使用`Entry`模型，我们首先需要在我们的数据库中为它创建一个表。幸运的是，Flask-SQLAlchemy 带有一个很好的辅助程序来做这件事。在博客项目的`app`目录中创建一个名为`scripts`的新子文件夹。然后创建一个名为`create_db.py`的文件：

```py
(blog) $ cd app/
(blog) $ mkdir scripts
(blog) $ touch scripts/create_db.py

```

将以下代码添加到`create_db.py`模块中。这个函数将自动查看我们编写的所有代码，并根据我们的模型在数据库中为`Entry`模型创建一个新表：

```py
import os, sys
sys.path.append(os.getcwd())
from main import db

if __name__ == '__main__':
    db.create_all()
```

从`app/`目录内执行脚本。确保虚拟环境是激活的。如果一切顺利，你应该看不到任何输出。

```py
(blog) $ python create_db.py 
(blog) $

```

### 注意

如果在创建数据库表时遇到错误，请确保你在 app 目录中，并且在运行脚本时虚拟环境是激活的。接下来，确保你的`SQLALCHEMY_DATABASE_URI`设置中没有拼写错误。

## 使用 Entry 模型

让我们通过保存一些博客条目来尝试我们的新`Entry`模型。我们将在 Python 交互式 shell 中进行此操作。在这个阶段，让我们安装**IPython**，这是一个功能强大的 shell，具有诸如制表符补全（默认的 Python shell 没有的功能）。

```py
(blog) $ pip install ipython

```

现在检查我们是否在`app`目录中，让我们启动 shell 并创建一些条目，如下所示：

```py
(blog) $ ipython

In []: from models import *  # First things first, import our Entry model and db object.
In []: db  # What is db?
Out[]: <SQLAlchemy engine='sqlite:////home/charles/projects/blog/app/blog.db'>

```

### 注意

如果你熟悉普通的 Python shell 但不熟悉 IPython，一开始可能会有点不同。要注意的主要事情是`In[]`指的是你输入的代码，`Out[]`是你放入 shell 的命令的输出。

IPython 有一个很棒的功能，允许你打印关于对象的详细信息。这是通过输入对象的名称后跟一个问号（?）来完成的。内省`Entry`模型提供了一些信息，包括参数签名和表示该对象的字符串（称为`docstring`）的构造函数。

```py
In []: Entry?  # What is Entry and how do we create it?
Type:       _BoundDeclarativeMeta
String Form:<class 'models.Entry'>
File:       /home/charles/projects/blog/app/models.py
Docstring:  <no docstring>
Constructor information:
 Definition:Entry(self, *args, **kwargs)

```

我们可以通过将列值作为关键字参数传递来创建`Entry`对象。在前面的示例中，它使用了`**kwargs`；这是一个快捷方式，用于将`dict`对象作为定义对象的值，如下所示：

```py
In []: first_entry = Entry(title='First entry', body='This is the body of my first entry.')

```

为了保存我们的第一个条目，我们将其添加到数据库会话中。会话只是表示我们在数据库上的操作的对象。即使将其添加到会话中，它也不会立即保存到数据库中。为了将条目保存到数据库中，我们需要提交我们的会话：

```py
In []: db.session.add(first_entry)
In []: first_entry.id is None  # No primary key, the entry has not been saved.
Out[]: True
In []: db.session.commit()
In []: first_entry.id
Out[]: 1
In []: first_entry.created_timestamp
Out[]: datetime.datetime(2014, 1, 25, 9, 49, 53, 1337)

```

从前面的代码示例中可以看出，一旦我们提交了会话，将为我们的第一个条目分配一个唯一的 id，并将`created_timestamp`设置为当前时间。恭喜，您已创建了您的第一个博客条目！

尝试自己添加几个。在提交之前，您可以将多个条目对象添加到同一个会话中，因此也可以尝试一下。

### 注意

在您进行实验的任何时候，都可以随时删除`blog.db`文件，并重新运行`create_db.py`脚本，以便使用全新的数据库重新开始。

## 对现有条目进行更改

修改现有的`Entry`时，只需进行编辑，然后提交。让我们使用之前返回给我们的 id 检索我们的`Entry`，进行一些更改，然后提交。SQLAlchemy 将知道需要更新它。以下是您可能对第一个条目进行编辑的方式：

```py
In []: first_entry = Entry.query.get(1)
In []: first_entry.body = 'This is the first entry, and I have made some edits.'
In []: db.session.commit()

```

就像那样，您的更改已保存。

## 删除条目

删除条目与创建条目一样简单。我们将调用`db.session.delete`而不是调用`db.session.add`，并传入我们希望删除的`Entry`实例。

```py
In []: bad_entry = Entry(title='bad entry', body='This is a lousy entry.')
In []: db.session.add(bad_entry)
In []: db.session.commit()  # Save the bad entry to the database.
In []: db.session.delete(bad_entry)
In []: db.session.commit()  # The bad entry is now deleted from the database.

```

# 检索博客条目

虽然创建、更新和删除操作相当简单，但当我们查看检索条目的方法时，真正有趣的部分开始了。我们将从基础知识开始，然后逐渐深入到更有趣的查询。

我们将使用模型类上的特殊属性进行查询：`Entry.query`。该属性公开了各种 API，用于处理数据库中条目的集合。

让我们简单地检索`Entry`表中所有条目的列表：

```py
In []: entries = Entry.query.all()
In []: entries  # What are our entries?
Out[]: [<Entry u'First entry'>, <Entry u'Second entry'>, <Entry u'Third entry'>, <Entry u'Fourth entry'>]

```

如您所见，在此示例中，查询返回了我们创建的`Entry`实例的列表。当未指定显式排序时，条目将以数据库选择的任意顺序返回给我们。让我们指定我们希望以标题的字母顺序返回给我们条目：

```py
In []: Entry.query.order_by(Entry.title.asc()).all()
Out []:
[<Entry u'First entry'>,
 <Entry u'Fourth entry'>,
 <Entry u'Second entry'>,
 <Entry u'Third entry'>]

```

接下来是如何按照最后更新时间的逆序列出您的条目：

```py
In []: oldest_to_newest = Entry.query.order_by(Entry.modified_timestamp.desc()).all()
Out []:
[<Entry: Fourth entry>,
 <Entry: Third entry>,
 <Entry: Second entry>,
 <Entry: First entry>]

```

## 过滤条目列表

能够检索整个博客条目集合非常有用，但是如果我们想要过滤列表怎么办？我们可以始终检索整个集合，然后在 Python 中使用循环进行过滤，但那将非常低效。相反，我们将依赖数据库为我们进行过滤，并简单地指定应返回哪些条目的条件。在以下示例中，我们将指定要按标题等于`'First entry'`进行过滤的条目。

```py
In []: Entry.query.filter(Entry.title == 'First entry').all()
Out[]: [<Entry u'First entry'>]

```

如果这对您来说似乎有些神奇，那是因为它确实如此！SQLAlchemy 使用操作符重载将诸如`<Model>.<column> == <some value>`的表达式转换为称为`BinaryExpression`的抽象对象。当您准备执行查询时，这些数据结构然后被转换为 SQL。

### 注意

`BinaryExpression`只是一个表示逻辑比较的对象，并且是通过重写通常在 Python 中比较值时调用的标准方法而生成的。

为了检索单个条目，您有两个选项：`.first()`和`.one()`。它们的区别和相似之处总结在以下表中：

| 匹配行的数量 | first()行为 | one()行为 |
| --- | --- | --- |
| 1 | 返回对象 | 返回对象 |
| 0 | 返回`None` | 引发`sqlalchemy.orm.exc.NoResultFound` |
| 2+ | 返回第一个对象（基于显式排序或数据库选择的排序） | 引发`sqlalchemy.orm.exc.MultipleResultsFound` |

让我们尝试与之前相同的查询，但是，而不是调用`.all()`，我们将调用`.first()`来检索单个`Entry`实例：

```py
In []: Entry.query.filter(Entry.title == 'First entry').first()
Out[]: <Entry u'First entry'>

```

请注意，以前的`.all()`返回包含对象的列表，而`.first()`只返回对象本身。

## 特殊查找

在前面的示例中，我们测试了相等性，但还有许多其他类型的查找可能。在下表中，我们列出了一些您可能会发现有用的查找。完整列表可以在 SQLAlchemy 文档中找到。

| 示例 | 意义 |
| --- | --- |
| Entry.title == 'The title' | 标题为“The title”的条目，区分大小写。 |
| Entry.title != 'The title' | 标题不是“The title”的条目。 |
| Entry.created_timestamp < datetime.date(2014, 1, 25) | 2014 年 1 月 25 日之前创建的条目。要使用小于或等于，使用<=。 |
| Entry.created_timestamp > datetime.date(2014, 1, 25) | 2014 年 1 月 25 日之后创建的条目。要使用大于或等于，使用>=。 |
| Entry.body.contains('Python') | 正文包含单词“Python”的条目，区分大小写。 |
| Entry.title.endswith('Python') | 标题以字符串“Python”结尾的条目，区分大小写。请注意，这也将匹配以单词“CPython”结尾的标题，例如。 |
| Entry.title.startswith('Python') | 标题以字符串“Python”开头的条目，区分大小写。请注意，这也将匹配标题如“Pythonistas”。 |
| Entry.body.ilike('%python%') | 正文包含单词“python”的条目，文本中任何位置，不区分大小写。百分号“%”是通配符。 |
| Entry.title.in_(['Title one', 'Title two']) | 标题在给定列表中的条目，要么是'Title one'要么是'Title two'。 |

## 组合表达式

前面表格中列出的表达式可以使用位运算符组合，以生成任意复杂的表达式。假设我们想要检索所有博客条目中标题包含`Python`或`Flask`的条目。为了实现这一点，我们将创建两个`contains`表达式，然后使用 Python 的位`OR`运算符进行组合，这是一个管道`|`字符，不像其他许多使用双管`||`字符的语言：

```py
Entry.query.filter(Entry.title.contains('Python') | Entry.title.contains('Flask'))
```

使用位运算符，我们可以得到一些非常复杂的表达式。试着弄清楚以下示例在询问什么：

```py
Entry.query.filter(
    (Entry.title.contains('Python') | Entry.title.contains('Flask')) &
    (Entry.created_timestamp > (datetime.date.today() - datetime.timedelta(days=30)))
)
```

您可能已经猜到，此查询返回所有标题包含`Python`或`Flask`的条目，并且在过去 30 天内创建。我们使用 Python 的位`OR`和`AND`运算符来组合子表达式。对于您生成的任何查询，可以通过打印查询来查看生成的 SQL，如下所示：

```py
In []: query = Entry.query.filter(
 (Entry.title.contains('Python') | Entry.title.contains('Flask')) &
 (Entry.created_timestamp > (datetime.date.today() - datetime.timedelta(days=30)))
)
In []: print str(query)

SELECT entry.id AS entry_id, ...
FROM entry 
WHERE (
 (entry.title LIKE '%%' || :title_1 || '%%') OR (entry.title LIKE '%%' || :title_2 || '%%')
) AND entry.created_timestamp > :created_timestamp_1

```

### 否定

还有一点要讨论，那就是**否定**。如果我们想要获取所有标题中不包含`Python`或`Flask`的博客条目列表，我们该怎么做呢？SQLAlchemy 提供了两种方法来创建这些类型的表达式，一种是使用 Python 的一元否定运算符（`~`），另一种是调用`db.not_()`。以下是如何使用 SQLAlchemy 构建此查询的方法：

使用一元否定：

```py
In []: Entry.query.filter(~(Entry.title.contains('Python') | Entry.title.contains('Flask')))

```

使用`db.not_()`：

```py
In []: Entry.query.filter(db.not_(Entry.title.contains('Python') | Entry.title.contains('Flask')))

```

### 运算符优先级

并非所有操作都被 Python 解释器视为相等。这就像在数学课上学习的那样，我们学到类似*2 + 3 * 4*的表达式等于*14*而不是*20*，因为乘法运算首先发生。在 Python 中，位运算符的优先级都高于诸如相等性测试之类的东西，这意味着在构建查询表达式时，您必须注意括号。让我们看一些示例 Python 表达式，并查看相应的查询：

| 表达式 | 结果 |
| --- | --- |
| (Entry.title == 'Python' &#124; Entry.title == 'Flask') | 错误！SQLAlchemy 会抛出错误，因为首先要评估的实际上是'Python' &#124; Entry.title! |
| (Entry.title == 'Python') &#124; (Entry.title == 'Flask') | 正确。返回标题为“Python”或“Flask”的条目。 |
| ~Entry.title == 'Python' | 错误！SQLAlchemy 会将其转换为有效的 SQL 查询，但结果将没有意义。 |
| ~(Entry.title == 'Python') | 正确。返回标题不等于“Python”的条目。 |

如果您发现自己在操作符优先级方面有困难，最好在使用`==`、`!=`、`<`、`<=`、`>`和`>=`的任何比较周围加上括号。

# 构建标记系统

标签是一个轻量级的分类系统，非常适合博客。标签允许您将多个类别应用于博客文章，并允许多篇文章在其类别之外相互关联。在我的博客上，我使用标签来组织帖子，这样对于想阅读我关于 Flask 的帖子的人，只需在“Flask”标签下查找即可找到所有相关的帖子。根据我们在第一章中讨论的规范，*创建您的第一个 Flask 应用程序*，每个博客条目可以有多少个标签都可以，因此关于 Flask 的帖子可能会被标记为 Flask 和 Python。同样，每个标签（例如 Python）可以与多个条目相关联。在数据库术语中，这称为多对多关系。

为了对此进行建模，我们必须首先创建一个模型来存储标签。这个模型将存储我们使用的标签名称，因此在我们添加了一些标签之后，表可能看起来像下面这样：

| id | tag |
| --- | --- |
| 1 | Python |
| 2 | Flask |
| 3 | Django |
| 4 | random-thoughts |

让我们打开`models.py`并为`Tag`模型添加一个定义。在文件末尾添加以下类，位于`Entry`类下方：

```py
class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64))
    slug = db.Column(db.String(64), unique=True)

    def __init__(self, *args, **kwargs):
        super(Tag, self).__init__(*args, **kwargs)
        self.slug = slugify(self.name)

    def __repr__(self):
        return '<Tag %s>' % self.name
```

您以前见过所有这些。我们添加了一个主键，这将由数据库管理，并添加了一个列来存储标签的名称。`name`列被标记为唯一，因此每个标签在这个表中只会被一行表示，无论它出现在多少个博客条目中。

现在我们既有博客条目模型，也有标签模型，我们需要一个第三个模型来存储两者之间的关系。当我们希望表示博客条目被标记为特定标签时，我们将在这个表中存储一个引用。以下是数据库表级别上正在发生的事情的图示：

![构建标记系统](img/1709_02_01.jpg)

由于我们永远不会直接访问这个中间表（SQLAlchemy 会透明地处理它），我们不会为它创建一个模型，而是简单地指定一个表来存储映射。打开`models.py`并添加以下突出显示的代码：

```py
import datetime, re

from app import db

def slugify(s):
    return re.sub('[^\w]+', '-', s).lower()

entry_tags = db.Table('entry_tags',
 db.Column('tag_id', db.Integer, db.ForeignKey('tag.id')),
 db.Column('entry_id', db.Integer, db.ForeignKey('entry.id'))
)

class Entry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    slug = db.Column(db.String(100), unique=True)
    body = db.Column(db.Text)
    created_timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    modified_timestamp = db.Column(
        db.DateTime,
        default=datetime.datetime.now,
        onupdate=datetime.datetime.now)

 tags = db.relationship('Tag', secondary=entry_tags,
 backref=db.backref('entries', lazy='dynamic'))

    def __init__(self, *args, **kwargs):
        super(Entry, self).__init__(*args, **kwargs)
        self.generate_slug()

    def generate_slug(self):
        self.slug = ''
        if self.title:
            self.slug = slugify(self.title)

    def __repr__(self):
        return '<Entry %s>' % self.title

class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64))
    slug = db.Column(db.String(64), unique=True)

    def __init__(self, *args, **kwargs):
        super(Tag, self).__init__(*args, **kwargs)
        self.slug = slugify(self.name)

    def __repr__(self):
        return '<Tag %s>' % self.name
```

通过创建`entry_tags`表，我们已经建立了`Entry`和`Tag`模型之间的链接。SQLAlchemy 提供了一个高级 API 来处理这种关系，名为`db.relationship`函数。这个函数在`Entry`模型上创建了一个新属性，允许我们轻松地读取和写入给定博客条目的标签。这两行代码中有很多内容，让我们仔细看一下：

```py
tags = db.relationship('Tag', secondary=entry_tags,
    backref=db.backref('entries', lazy='dynamic'))
```

我们将`Entry`类的标签属性设置为`db.relationship`函数的返回值。前两个参数`'Tag'`和`secondary=entry_tags`指示 SQLAlchemy 我们将通过`entry_tags`表查询`Tag`模型。第三个参数创建了一个反向引用，允许我们从`Tag`模型返回到相关的博客条目列表。通过指定`lazy='dynamic'`，我们指示 SQLAlchemy，我们不希望它为我们加载所有相关的条目，而是想要一个查询对象。

## 向条目添加和删除标签

让我们使用 IPython shell 来看看这是如何工作的。关闭当前的 shell 并重新运行`scripts/create_db.py`脚本。由于我们添加了两个新表，这一步是必要的。现在重新打开 IPython：

```py
(blog) $ python scripts/create_db.py
(blog) $ ipython
In []: from models import *
In []: Tag.query.all()
Out[]: []

```

目前数据库中没有标签，所以让我们创建一些标签：

```py
In []: python = Tag(name='python')
In []: flask = Tag(name='flask')
In []: db.session.add_all([python, flask])
In []: db.session.commit()

```

现在让我们加载一些示例条目。在我的数据库中有四个：

```py
In []: Entry.query.all()
Out[]:
[<Entry Py
thon entry>,
 <Entry Flask entry>,
 <Entry More flask>,
 <Entry Django entry>]
In []: python_entry, flask_entry, more_flask, django_entry = _

```

### 注意

在 IPython 中，您可以使用下划线(`_`)来引用上一行的返回值。

要向条目添加标签，只需将它们分配给条目的`tags`属性。就是这么简单！

```py
In []: python_entry.tags = [python]
In []: flask_entry.tags = [python, flask]
In []: db.session.commit()

```

我们可以像处理普通的 Python 列表一样处理条目的标签列表，因此通常的`.append()`和`.remove()`方法也可以使用：

```py
In []: kittens = Tag(name='kittens')
In []: python_entry.tags.append(kittens)
In []: db.session.commit()
In []: python_entry.tags
Out[]: [<Tag python>, <Tag kittens>]
In []: python_entry.tags.remove(kittens)
In []: db.session.commit()
In []: python_entry.tags
Out[]: [<Tag python>]

```

## 使用 backrefs

创建`Entry`模型上的`tags`属性时，您会回忆起我们传入了`backref`参数。让我们使用 IPython 来看看后向引用是如何使用的。

```py
In []: python  # The python variable is just a tag.
Out[]: <Tag python>
In []: python.entries
Out[]: <sqlalchemy.orm.dynamic.AppenderBaseQuery at 0x332ff90>
In []: python.entries.all()
Out[]: [<Entry Flask entry>, <Entry Python entry>]

```

与`Entry.tags`引用不同，后向引用被指定为`lazy='dynamic'`。这意味着，与给出标签列表的`entry.tags`不同，我们每次访问`tag.entries`时都不会收到条目列表。为什么呢？通常，当结果集大于几个项目时，将`backref`参数视为查询更有用，可以进行过滤、排序等操作。例如，如果我们想显示最新的标记为`python`的条目会怎样？

```py
In []: python.entries.order_by(Entry.created_timestamp.desc()).first()
Out[]: <Entry Flask entry>

```

### 注意

SQLAlchemy 文档包含了可以用于 lazy 参数的各种值的优秀概述。您可以在[`docs.sqlalchemy.org/en/rel_0_9/orm/relationships.html#sqlalchemy.orm.relationship.params.lazy`](http://docs.sqlalchemy.org/en/rel_0_9/orm/relationships.html#sqlalchemy.orm.relationship.params.lazy)上找到它们。

# 对模式进行更改

本章最后要讨论的主题是如何对现有的模型定义进行修改。根据项目规范，我们希望能够保存博客条目的草稿。现在我们没有办法知道一个条目是否是草稿，所以我们需要添加一个列来存储条目的状态。不幸的是，虽然`db.create_all()`用于创建表非常完美，但它不会自动修改现有的表；为了做到这一点，我们需要使用迁移。

## 将 Flask-Migrate 添加到我们的项目中

我们将使用 Flask-Migrate 来帮助我们在更改模式时自动更新数据库。在博客虚拟环境中，使用`pip`安装 Flask-Migrate：

```py
(blog) $ pip install flask-migrate

```

### 注意

SQLAlchemy 的作者有一个名为 alembic 的项目；Flask-Migrate 使用它并直接将其与 Flask 集成，使事情变得更容易。

接下来，我们将向我们的应用程序添加一个`Migrate`助手。我们还将为我们的应用程序创建一个脚本管理器。脚本管理器允许我们在应用程序的上下文中直接从命令行执行特殊命令。我们将使用脚本管理器来执行`migrate`命令。打开`app.py`并进行以下添加：

```py
from flask import Flask
from flask.ext.migrate import Migrate, MigrateCommand
from flask.ext.script import Manager
from flask.ext.sqlalchemy import SQLAlchemy

from config import Configuration

app = Flask(__name__)
app.config.from_object(Configuration)
db = SQLAlchemy(app)
migrate = Migrate(app, db)

manager = Manager(app)
manager.add_command('db', MigrateCommand)

```

为了使用管理器，我们将在`app.py`旁边添加一个名为`manage.py`的新文件。将以下代码添加到`manage.py`中：

```py
from app import manager
from main import *

if __name__ == '__main__':
    manager.run()
```

这看起来与`main.py`非常相似，关键区别在于，我们不是调用`app.run()`，而是调用`manager.run()`。

### 注意

Django 有一个类似的，尽管是自动生成的`manage.py`文件，起着类似的功能。

## 创建初始迁移

在我们开始更改模式之前，我们需要创建其当前状态的记录。为此，请从博客的`app`目录内运行以下命令。第一个命令将在`app`文件夹内创建一个迁移目录，用于跟踪我们对模式所做的更改。第二个命令`db migrate`将创建我们当前模式的快照，以便将来的更改可以与之进行比较。

```py
(blog) $ python manage.py db init

 Creating directory /home/charles/projects/blog/app/migrations ... done
 ...
(blog) $ python manage.py db migrate
INFO  [alembic.migration] Context impl SQLiteImpl.
INFO  [alembic.migration] Will assume non-transactional DDL.
 Generating /home/charles/projects/blog/app/migrations/versions/535133f91f00_.py ... done

```

最后，我们将运行`db upgrade`来运行迁移，以指示迁移系统一切都是最新的：

```py
(blog) $ python manage.py db upgrade
INFO  [alembic.migration] Context impl SQLiteImpl.
INFO  [alembic.migration] Will assume non-transactional DDL.
INFO  [alembic.migration] Running upgrade None -> 535133f91f00, empty message

```

## 添加状态列

现在我们已经有了当前模式的快照，我们可以开始进行更改。我们将添加一个名为`status`的新列，该列将存储与特定状态对应的整数值。尽管目前只有两种状态（`PUBLIC`和`DRAFT`），但使用整数而不是布尔值使我们有可能在将来轻松添加更多状态。打开`models.py`并对`Entry`模型进行以下添加：

```py
class Entry(db.Model):
 STATUS_PUBLIC = 0
 STATUS_DRAFT = 1

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    slug = db.Column(db.String(100), unique=True)
    body = db.Column(db.Text)
 status = db.Column(db.SmallInteger, default=STATUS_PUBLIC)
    created_timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    ...
```

从命令行，我们将再次运行`db migrate`来生成迁移脚本。您可以从命令的输出中看到它找到了我们的新列！

```py
(blog) $ python manage.py db migrate
INFO  [alembic.migration] Context impl SQLiteImpl.
INFO  [alembic.migration] Will assume non-transactional DDL.
INFO  [alembic.autogenerate.compare] Detected added column 'entry.status'
 Generating /home/charl
es/projects/blog/app/migrations/versions/2c8e81936cad_.py ... done

```

因为我们在数据库中有博客条目，所以我们需要对自动生成的迁移进行小修改，以确保现有条目的状态被初始化为正确的值。为此，打开迁移文件（我的是`migrations/versions/2c8e81936cad_.py`）并更改以下行：

```py
op.add_column('entry', sa.Column('status', sa.SmallInteger(), nullable=True))
```

将`nullable=True`替换为`server_default='0'`告诉迁移脚本不要将列默认设置为 null，而是使用`0`。

```py
op.add_column('entry', sa.Column('status', sa.SmallInteger(), server_default='0'))
```

最后，运行`db upgrade`来运行迁移并创建状态列。

```py
(blog) $ python manage.py db upgrade
INFO  [alembic.migration] Context impl SQLiteImpl.
INFO  [alembic.migration] Will assume non-transactional DDL.
INFO  [alembic.migration] Running upgrade 535133f91f00 -> 2c8e81936cad, empty message

```

恭喜，您的`Entry`模型现在有了一个状态字段！

# 总结

到目前为止，您应该熟悉使用 SQLAlchemy 来处理关系数据库。我们介绍了使用关系数据库和 ORM 的好处，配置了一个 Flask 应用程序来连接到关系数据库，并创建了 SQLAlchemy 模型。所有这些都使我们能够在数据之间创建关系并执行查询。最重要的是，我们还使用了迁移工具来处理未来的数据库模式更改。

在第三章中，*模板和视图*，我们将搁置交互式解释器，开始创建视图以在 Web 浏览器中显示博客条目。我们将利用我们所有的 SQLAlchemy 知识创建有趣的博客条目列表，以及一个简单的搜索功能。我们将构建一组模板，使博客网站在视觉上更具吸引力，并学习如何使用 Jinja2 模板语言来消除重复的 HTML 编码。这将是一个有趣的章节！
