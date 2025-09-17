# 第二章：使用 SQLAlchemy 创建模型

正如我们在上一章中看到的，模型是抽象数据和提供访问数据的通用接口的一种方式。在大多数 Web 应用程序中，数据是从 **关系数据库管理系统**（**RDBMS**）存储和检索的，这是一个以行和列的表格格式存储数据的数据库，能够实现跨表的数据关系模型。一些例子包括 MySQL、Postgres、Oracle 和 MSSQL。

为了在我们的数据库上创建模型，我们将使用一个名为 **SQLAlchemy** 的 Python 包。SQLAlchemy 是最低级别的数据库 API，并在最高级别执行 **对象关系映射**。**ORM**（对象关系映射器）是一种工具，允许开发人员使用面向对象的方法存储和检索数据，并解决对象关系不匹配——当使用面向对象编程语言编写的程序使用关系数据库管理系统时，经常会遇到的一组概念和技术难题。关系型和面向对象模型差异如此之大，以至于需要额外的代码和功能才能使它们有效地协同工作。这创建了一个虚拟对象数据库，并将数据库中的大量类型转换为 Python 中的类型和对象的混合。此外，编程语言，如 Python，允许您拥有不同的对象，它们相互持有引用，并获取和设置它们的属性。像 SQLAlchemy 这样的 ORM 帮助在将它们插入传统数据库时进行转换。

为了将 SQLAlchemy 集成到我们的应用上下文中，我们将使用 **Flask SQLAlchemy**。Flask SQLAlchemy 是在 SQLAlchemy 之上提供有用默认值和 Flask 特定功能的便利层。如果您已经熟悉 SQLAlchemy，那么您可以在不使用 Flask SQLAlchemy 的情况下自由使用它。

到本章结束时，我们将拥有我们博客应用的完整数据库模式，以及与该模式交互的模型。

在本章中，我们将涵盖以下主题：

+   使用 SQLAlchemy 设计数据库表和关系

+   创建、读取、更新和删除模型

+   学习定义模型关系、约束和索引

+   创建自动数据库迁移

# 设置 SQLAlchemy

为了完成本章中的练习，您需要一个正在运行的数据库，如果您还没有的话。如果您从未安装过数据库，或者您没有偏好，那么 **SQLite** 是初学者的最佳选择，或者如果您想快速启动一个概念验证。

**SQLite** 是一个快速、无需服务器即可工作且完全包含在一个文件中的 SQL 嵌入式数据库引擎。SQLite 还在 Python 中原生支持，因此如果您选择使用 SQLite，那么在 *我们的第一个模型* 部分的练习中，将自动为您创建一个 SQLite 数据库。

# Python 包

Flask SQLAlchemy 可以与多个数据库引擎一起使用，例如 ORACLE、MSSQL、MySQL、PostgreSQL、SQLite 和 Sybase，但我们需要为这些引擎安装额外的特定包。现在，是时候通过为所有应用依赖项创建一个新的虚拟环境来引导我们的项目了。这个虚拟环境将用于我们的博客应用。输入以下代码：

```py
$ virtualenv env
```

然后，在`requirements.txt`中添加以下代码以安装包：

```py
flask-sqlalchemy
```

你还需要为所选数据库安装特定的包，这些包将作为 SQLAlchemy 的连接器。因此，在`requirements.txt`中添加特定于你的引擎的包，如下所示。SQLite 用户可以跳过此步骤：

```py
    # MySQL
    PyMySQL
    # Postgres
    psycopg2
    # MSSQL
    pyodbc
    # Oracle
    cx_Oracle
```

最后，使用以下代码激活并安装依赖项：

```py
$ source env/bin/activate
$ pip install -r requirements.txt
```

# Flask SQLAlchemy

在我们抽象数据之前，我们需要设置 Flask SQLAlchemy。SQLAlchemy 通过特殊的数据库 URI 创建其数据库连接。这是一个看起来像 URL 的字符串，包含 SQLAlchemy 连接所需的所有信息。它采用以下代码的一般形式：

```py
databasetype+driver://user:password@host:port/db_name 
```

对于你之前安装的每个驱动程序，URI 将如下所示：

```py
# SQLite connection string/uri is a path to the database file - relative or absolute.
sqlite:///database.db 
# MySQL 
mysql+pymysql://user:password@ip:port/db_name 
# Postgres 
postgresql+psycopg2://user:password@ip:port/db_name 
# MSSQL 
mssql+pyodbc://user:password@dsn_name 
# Oracle 
oracle+cx_oracle://user:password@ip:port/db_name 
```

在我们的`config.py`文件中，使用以下方式将 URI 添加到`DevConfig`文件中：

```py
class DevConfig(Config): 
  debug = True 
  SQLALCHEMY_DATABASE_URI = "YOUR URI" 
```

# 我们的第一个模型

你可能已经注意到，我们没有在我们的数据库中实际创建任何表来抽象。这是因为 SQLAlchemy 允许我们创建模型或从模型创建表。我们将在创建第一个模型之后查看这一点。

在我们的`main.py`文件中，SQLAlchemy 必须首先使用以下方式初始化我们的应用：

```py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import DevConfig

app = Flask(__name__)
app.config.from_object(DevConfig)
db = SQLAlchemy(app) 
```

SQLAlchemy 将读取我们应用的配置并自动连接到我们的数据库。让我们在`main.py`文件中创建一个`User`模型来与用户表交互，如下所示：

```py
class User(db.Model): 
  id = db.Column(db.Integer(), primary_key=True) 
  username = db.Column(db.String(255)) 
  password = db.Column(db.String(255)) 

  def __init__(self, username): 
    self.username = username 

  def __repr__(self): 
    return "<User '{}'>".format(self.username) 
```

我们取得了什么成果？我们现在有一个基于用户表且包含三个列的模型。当我们从`db.Model`继承时，与数据库的整个连接和通信将已经为我们处理。

每个是`db.Column`实例的类变量都代表数据库中的一个列。在`db.Column`实例中有一个可选的第一个参数，允许我们指定数据库中列的名称。如果没有它，SQLAlchemy 将假设变量的名称与列的名称相同。使用这个可选变量将看起来如下：

```py
username = db.Column('user_name', db.String(255))
```

`db.Column`的第二个参数告诉 SQLAlchemy 该列应该被处理为什么类型。在这本书中，我们将使用的主要类型如下：

+   `db.String`

+   `db.Text`

+   `` `db.Integer` ``

+   `db.Float`

+   `` `db.Boolean` ``

+   `db.Date`

+   `db.DateTime`

+   `db.Time`

每种类型代表的内容相当简单，如下表所示：

+   `String`和`Text`类型将 Python 字符串转换为`varchar`和`text`类型的列。

+   `Integer`和`Float`类型会将任何 Python 数字转换为正确的类型，然后再将其插入数据库中。

+   `Boolean`类型接受 Python 的`True`或`False`语句，如果数据库有`boolean`类型，则将布尔值插入数据库。如果没有`boolean`类型在数据库中，SQLAlchemy 会自动在 Python 布尔值和数据库中的 0 或 1 之间进行转换。

+   `Date`、`DateTime`和`Time`类型使用来自`datetime`原生库的同名 Python 类型，并将它们转换为数据库。

`String`、`Integer`和`Float`类型接受一个额外的参数，告诉 SQLAlchemy 我们列的长度限制。

如果你希望真正理解 SQLAlchemy 如何将你的代码转换为 SQL 查询，请将以下内容添加到`DevConfig`文件中，`SQLALCHEMY_ECHO = True`。

这将在终端中打印出创建的查询。随着你在书中进一步学习，你可能希望关闭此功能，因为每次页面加载时，终端可能会打印出数十个查询。

`primary_key`参数告诉 SQLAlchemy 该列具有**主键索引**。每个 SQLAlchemy 模型都需要一个主键才能正常工作。所有对象关系映射对象都通过会话中的身份映射与数据库行相关联，这是 SQLAlchemy 中实现的工作单元机制的核心。这就是为什么我们需要在模型中声明主键的原因。

SQLAlchemy 会假设你的表名是模型类名称的小写版本。然而，如果我们想将表命名为`user`以外的名称怎么办？为了告诉 SQLAlchemy 使用什么名称，请添加`__tablename__`类变量。

这也是你连接到数据库中已存在的表的方式。只需将表名放在以下字符串中：

```py
class User(db.Model): 
  __tablename__ = 'user_table_name' 

  id = db.Column(db.Integer(), primary_key=True) 
  username = db.Column(db.String(255)) 
  password = db.Column(db.String(255)) 
```

我们不必包含`__init__`或`__repr__`函数。如果我们不包含，那么 SQLAlchemy 将自动创建一个`__init__`函数，该函数接受列的名称和值作为关键字参数。

使用 ORM 命名表为`user`可能会导致问题，因为在 MySQL 中，`user`是一个保留字。使用 ORM 的一个优点是，你可以轻松地将你的引擎从 SQLite 迁移到 MySQL，然后到 ORACLE 等。一个非常简单的修复方法是使用前缀你的模式并使用。

# 创建用户表

使用 SQLAlchemy 进行繁重的工作，我们现在将在数据库中创建用户表。将`manage.py`更新为以下内容：

```py
from main import app, db, User 

@app.shell_context_processor
def make_shell_context():
  return dict(app=app, db=db, User=User) 
```

从现在开始，每次我们创建一个新的模型时，我们都会导入它并将其添加到返回的`dict`中。

这将允许我们在 Flask shell 中使用我们的模型，因为我们正在注入。现在运行 shell，并使用`db.create_all()`创建所有表，如下面的代码所示：

```py
    # Tell Flask where to load our shell context
    $ export FLASK_APP=manage.py
 $ flask shell
    >>> db.create_all()
```

在您的数据库中，您现在应该看到一个名为 `users` 的表，其中包含指定的列。此外，如果您正在使用 SQLite，您现在应该能在文件结构中看到一个名为 `database.db` 的文件，如下面的代码所示：

```py
$ sqlite3 database.db .tables user
```

# CRUD

在数据存储的每一种机制中，都有四种基本类型的函数：**创建**、**读取**、**更新**和**删除**（**CRUD**）。这些允许我们执行所有基本的数据操作和查看方式，这些对于我们的 Web 应用程序是必需的。为了使用这些函数，我们将使用数据库中的一个名为 **session** 的对象。会话将在本章后面进行解释，但就现在而言，可以将它们视为数据库中所有更改的存储位置。

# 创建模型

要使用我们的模型在数据库中创建新行，请将模型添加到 `session` 和 `commit` 对象中。将对象添加到会话中会标记其更改以保存。提交是将会话保存到数据库的过程，如下所示：

```py
    >>> user = User(username='fake_name')
    >>> db.session.add(user)
    >>> db.session.commit()
```

如您所见，向我们的表中添加新行很简单。

# 读取模型

在我们将数据添加到数据库后，可以使用 `Model.query` 进行数据查询。对于使用 SQLAlchemy 的人来说，这是 `db.session.query(Model)` 的简写。

对于我们的第一个示例，使用 `all()` 获取用户表的所有行作为列表，如下所示：

```py
    >>> users = User.query.all()
    >>> users
    [<User 'fake_name'>]
```

当数据库中的项目数量增加时，此查询过程会变慢。在 SQLAlchemy 中，就像在 SQL 中一样，我们有 `limit` 函数来指定我们希望处理的总行数：

```py
    >>> users = User.query.limit(10).all()
```

默认情况下，SQLAlchemy 按照主键顺序返回记录。为了控制这一点，我们有一个 `order_by` 函数，其用法如下：

```py
    # ascending
    >>> users = User.query.order_by(User.username).all()
    # descending
    >>> users = User.query.order_by(User.username.desc()).all()
```

要返回单个记录，我们使用 `first()` 而不是 `all()`，如下所示：

```py
>>> user = User.query.first()
>>> user.username
fake_name
```

要通过其主键返回一个模型，使用 `query.get()`，如下所示：

```py
>>> user = User.query.get(1)
>>> user.username
fake_name
```

所有这些函数都是可链式的，这意味着可以将它们附加在一起以修改返回的结果。那些精通 JavaScript 的您会发现以下语法很熟悉：

```py
>>> users = User.query.order_by(
            User.username.desc()
 ).limit(10).first()
```

`first()` 和 `all()` 方法返回一个值，因此结束链式调用。

此外，还有一个 Flask-SQLAlchemy 特定的方法，称为 **pagination**，可以用来代替 `first()` 或 `all()`。这是一个方便的方法，旨在启用大多数网站在显示长列表时使用的分页功能。第一个参数定义了查询应返回的页面，第二个参数定义了每页的项目数。因此，如果我们传递 `1` 和 `10` 作为参数，将返回前 10 个对象。

如果我们改为传递 `2` 和 `10`，则将返回对象 11–20，依此类推。分页方法与 `first()` 和 `all()` 方法不同，因为它返回一个分页对象而不是模型列表。例如，如果我们想获取我们博客第一页的虚构 `Post` 对象的前 10 项，我们会使用以下方法：

```py
>>> User.query.paginate(1, 10)
<flask_sqlalchemy.Pagination at 0x105118f50>
```

此对象具有几个有用的属性，如下所示：

```py
>>> page = User.query.paginate(1, 10)
# returns the entities in the page
>>> page.items
[<User 'fake_name'>]
# what page does this object represent
>>> page.page
1
# How many pages are there
>>> page.pages
1
# are there enough models to make the next or previous page
>>> page.has_prev, page.has_next
(False, False)
# return the next or previous page pagination object
# if one does not exist returns the current page
>>> page.prev(), page.next()
(<flask_sqlalchemy.Pagination at 0x10812da50>,
<flask_sqlalchemy.Pagination at 0x1081985d0>)
```

# 过滤查询

现在我们来到了 SQL 的真正威力所在——即通过一系列规则来过滤结果。为了获取满足一系列特性的模型列表，我们使用 `query.filter_by` 过滤器。`query.filter_by` 过滤器接受命名参数，这些参数代表我们在数据库的每一列中寻找的值。要获取用户名为 `fake_name` 的所有用户列表，我们会使用以下代码：

```py
    >>> users = User.query.filter_by(username='fake_name').all()
```

这个例子是针对一个值进行过滤，但可以向 `filter_by` 过滤器传递多个值。就像我们之前的函数一样，`filter_by` 是可链式的，如下所示：

```py
    >>> users = User.query.order_by(User.username.desc())
            .filter_by(username='fake_name')
            .limit(2)
            .all()
```

`query.filter_by` 语句仅在你知道你正在寻找的确切值时才有效。通过将 Python 比较语句传递给查询，可以避免这种情况，如下所示：

```py
    >>> user = User.query.filter(
            User.id > 1
        ).all()
```

这是一个简单的例子，但 `query.filter` 接受任何 Python 比较。对于常见的 Python 类型，如 `integers`、`strings` 和 `dates`，可以使用 `==` 操作符进行相等比较。如果你有一个 `integer`、`float` 或 `date` 列，也可以使用 `>`、`<`、`<=` 和 `>=` 操作符传递不等式语句。

我们还可以使用 SQLAlchemy 函数翻译复杂的 SQL 查询。例如，要使用 `IN`、`OR` 或 `NOT` SQL 比较，我们会使用以下代码：

```py
    >>> from sqlalchemy.sql.expression import not_, or_
    >>> user = User.query.filter(
        User.username.in_(['fake_name']),
        User.password == None
    ).first()
    # find all of the users with a password
    >>> user = User.query.filter(
        not_(User.password == None)
    ).first()
    # all of these methods are able to be combined
    >>> user = User.query.filter(
        or_(not_(User.password == None), User.id >= 1)
    ).first()
```

在 SQLAlchemy 中，对 `None` 的比较会被翻译为对 `NULL` 的比较。

# 更新模型

要更新已存在模型的值，将 `update` 方法应用于查询对象——即在返回模型之前，使用如 `first()` 或 `all()` 等方法，如下所示：

```py
>>> User.query.filter_by(username='fake_name').update({
 'password': 'test'
})
# The updated models have already been added to the session
>>> db.session.commit()
```

# 删除模型

如果我们希望从数据库中删除一个模型，我们会使用以下代码：

```py
>>> user = User.query.filter_by(username='fake_name').first()
>>> db.session.delete(user)
>>> db.session.commit()
```

# 模型之间的关系

SQLAlchemy 中模型之间的关系是两个或多个模型之间的链接，允许模型自动相互引用。这使得自然相关的数据，如文章的评论，可以很容易地从数据库及其相关数据中检索出来。这正是 RDBMS 中的 R 的来源，这也赋予了这种类型的数据库大量的能力。

让我们创建我们的第一个关系。我们的博客网站将需要一些博客文章。每篇博客文章将由一个用户撰写，因此将文章链接回撰写它们的用户是有意义的，这样我们就可以轻松地获取一个用户的全部文章。这是一个 **一对多** 关系的例子，如下所示：

SQLite 和 MySQL/MyISAM 引擎不强制执行关系约束。如果你在开发环境中使用 SQLite，而在生产环境中使用不同的引擎（如带有 innodb 的 MySQL），这可能会引起问题，但你可以告诉 SQLite 强制执行外键约束（这将带来性能上的惩罚）。

```py
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
```

# 一对多关系

让我们添加一个模型来表示我们网站上的博客文章：

```py
class Post(db.Model): 
  id = db.Column(db.Integer(), primary_key=True) 
  title = db.Column(db.String(255)) 
  text = db.Column(db.Text()) 
  publish_date = db.Column(db.DateTime()) 
  user_id = db.Column(db.Integer(), db.ForeignKey('user.id')) 

  def __init__(self, title): 
    self.title = title 

  def __repr__(self): 
    return "<Post '{}'>".format(self.title) 
```

注意`user_id`列。熟悉 RDBMS 的人会知道这代表一个**外键约束**。外键约束是数据库中的一个规则，它强制`user_id`的值存在于用户表中的`id`列。这是数据库中的一个检查，以确保`Post`始终引用一个存在的用户。`db.ForeignKey`的参数是用户 ID 字段的字符串表示。如果你已经决定用`__table_name__`来调用你的用户表，那么你必须更改这个字符串。这个字符串用于代替直接使用`User.id`的引用，因为在 SQLAlchemy 初始化期间，`User`对象可能还不存在。

`user_id`列本身不足以告诉 SQLAlchemy 我们有一个关系。我们必须按照以下方式修改我们的`User`模型：

```py
class User(db.Model): 
  id = db.Column(db.Integer(), primary_key=True) 
  username = db.Column(db.String(255)) 
  password = db.Column(db.String(255)) 
  posts = db.relationship( 
    'Post', 
    backref='user', 
    lazy='dynamic' 
  ) 
```

`db.relationship`函数在 SQLAlchemy 中创建了一个与`Post`模型中的`db.ForeignKey`连接的属性。第一个参数是我们引用的类的名称。我们很快就会介绍`backref`的作用，但`lazy`参数是什么？`lazy`参数控制 SQLAlchemy 如何加载我们的相关对象。`subquery`短语会在我们的`Post`对象被加载时立即加载我们的关系。这减少了查询的数量，但当返回的项目数量增加时，速度会减慢。相比之下，使用`dynamic`选项，相关对象将在访问时加载，并且可以在返回之前进行过滤。如果返回的对象数量很大或将成为很大，这是最好的选择。

我们现在可以访问`User.posts`变量，它将返回一个包含所有`user_id`字段等于我们的`User.id`的帖子的列表。现在让我们在我们的 shell 中尝试这个操作，如下所示：

```py
    >>> user = User.query.get(1)
    >>> new_post = Post('Post Title')
    >>> new_post.user_id = user.id
    >>> user.posts
    []
    >>> db.session.add(new_post)
    >>> db.session.commit()
    >>> user.posts
    [<Post 'Post Title'>]
```

注意，我们没有能够在不提交我们的数据库更改的情况下从我们的关系访问我们的帖子。

`backref`参数给了我们通过`Post.user`访问和设置我们的`User`类的能力。这是由以下代码给出的：

```py
    >>> second_post = Post('Second Title')
    >>> second_post.user = user
    >>> db.session.add(second_post)
    >>> db.session.commit()
    >>> user.posts
    [<Post 'Post Title'>, <Post 'Second Title'>]
```

因为`user.posts`是一个列表，我们也可以将我们的`Post`模型添加到列表中来自动保存，如下所示：

```py
    >>> second_post = Post('Second Title')
    >>> user.posts.append(second_post)
    >>> db.session.add(user)
    >>> db.session.commit()
    >>> user.posts
    [<Post 'Post Title'>, <Post 'Second Title'>]
```

使用`backref`选项作为动态的，我们可以将我们的关系列视为一个查询以及一个列表，如下所示：

```py
    >>> user.posts
    [<Post 'Post Title'>, <Post 'Second Title'>] >>> user.posts.order_by(Post.publish_date.desc()).all()
    [<Post 'Second Title'>, <Post 'Post Title'>]
```

在我们继续到下一个关系类型之前，让我们添加一个用于用户评论的一对多关系的另一个模型，这个模型将在后面的书中使用。我们可以使用以下代码来完成这个操作：

```py
class Post(db.Model): 
    id = db.Column(db.Integer(), primary_key=True) 
    title = db.Column(db.String(255)) 
    text = db.Column(db.Text()) 
    publish_date = db.Column(db.DateTime()) 
    comments = db.relationship( 
      'Comment', 
      backref='post', 
      lazy='dynamic' 
    ) 
    user_id = db.Column(db.Integer(), db.ForeignKey('user.id'))
    def __init__(self, title): 
        self.title = title
    def __repr__(self): 
        return "<Post '{}'>".format(self.title)

```

注意前面代码中的`__repr__`方法签名。这是一个 Python 中的内置函数，用于返回对象的字符串表示。接下来是`Comment`模型，如下所示：

```py
class Comment(db.Model): 
    id = db.Column(db.Integer(), primary_key=True) 
    name = db.Column(db.String(255)) 
    text = db.Column(db.Text()) 
    date = db.Column(db.DateTime()) 
    post_id = db.Column(db.Integer(), db.ForeignKey('post.id'))
    def __repr__(self): 
        return "<Comment '{}'>".format(self.text[:15]) 
```

# 多对多关系

如果我们有两个模型可以相互引用，但每个模型都需要引用每种类型的多于一个实例呢？在我们的例子中，我们的博客帖子需要标签以便用户可以轻松地对相似帖子进行分组。每个标签可以引用多个帖子，但每个帖子可以有多个标签。这种关系称为**多对多**关系。考虑以下例子：

```py
tags = db.Table('post_tags', 
    db.Column('post_id', db.Integer, db.ForeignKey('post.id')), 
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id')) 
) 

class Post(db.Model): 
    id = db.Column(db.Integer(), primary_key=True) 
    title = db.Column(db.String(255)) 
    text = db.Column(db.Text()) 
    publish_date = db.Column(db.DateTime()) 
    comments = db.relationship( 
      'Comment', 
      backref='post', 
      lazy='dynamic' 
    ) 
    user_id = db.Column(db.Integer(), db.ForeignKey('user.id')) 
    tags = db.relationship( 
        'Tag', 
        secondary=tags, 
        backref=db.backref('posts', lazy='dynamic') 
    ) 

    def __init__(self, title): 
        self.title = title
    def __repr__(self): 
        return "<Post '{}'>".format(self.title) 

class Tag(db.Model): 
    id = db.Column(db.Integer(), primary_key=True) 
    title = db.Column(db.String(255))

    def __init__(self, title): 
        self.title = title 

    def __repr__(self): 
        return "<Tag '{}'>".format(self.title) 
```

`db.Table`对象是对数据库的底层访问，比`db.Model`抽象级别低。`db.Model`对象位于`db.Table`之上，并提供了对表中特定行的表示。使用`db.Table`对象是因为没有必要访问表中的单个行。

`tags`变量用于表示`post_tags`表，该表包含两行：一行代表帖子的 ID，另一行代表标签的 ID。为了说明这是如何工作的，让我们来看一个例子。假设表中有以下数据：

```py
post_id   tag_id 
1         1 
1         3 
2         3 
2         4 
2         5 
3         1 
3         2 
```

SQLAlchemy 会将此翻译为以下内容：

+   一个 ID 为`1`的帖子有 ID 为`1`和`3`的标签

+   一个 ID 为`2`的帖子有 ID 为`3`、`4`和`5`的标签

+   一个 ID 为`3`的帖子有 ID 为`1`和`2`的标签

你可以像描述标签与帖子相关联一样轻松地描述这些数据。

在`db.relationship`函数设置我们的关系之前，这次它有一个次要参数。次要参数告诉 SQLAlchemy，这个关系存储在`tags`表中，如下面的代码所示：

```py
    >>> post_one = Post.query.filter_by(title='Post Title').first()
    >>> post_two = Post.query.filter_by(title='Second Title').first()
    >>> tag_one = Tag('Python')
    >>> tag_two = Tag('SQLAlchemy')
    >>> tag_three = Tag('Flask')
    >>> post_one.tags = [tag_two]
    >>> post_two.tags = [tag_one, tag_two, tag_three]
    >>> tag_two.posts
    [<Post 'Post Title'>, <Post 'Second Title'>] >>> db.session.add(post_one)
    >>> db.session.add(post_two)
    >>> db.session.commit()
```

如同在单对多关系中，主关系列只是一个列表，主要区别在于`backref`选项现在也是一个列表。因为它是列表，我们可以从`tag`对象添加帖子到标签，如下所示：

```py
    >>> tag_one.posts.append(post_one)
    [<Post 'Post Title'>, <Post 'Second Title'>] >>> post_one.tags
    [<Tag 'SQLAlchemy'>, <Tag 'Python'>]
    >>> db.session.add(tag_one)
    >>> db.session.commit()
```

# 约束和索引

使用约束被认为是一种良好的实践。这样，你可以限制某个模型属性的域，并确保数据完整性和质量。你可以使用许多类型的约束；在前面的章节中已经介绍了主键和外键约束。SQLAlchemy 支持的其它类型的约束如下所示：

+   NOT NULL（确保某个属性包含数据）

+   UNIQUE（确保数据库表中的某个属性值始终是唯一的，该表包含模型数据）

+   DEFAULT（在未提供值时为属性设置默认值）

+   CHECK（用于指定值的范围）

使用 SQLAlchemy，你可以确保你的数据域限制是明确的，并且都在同一个地方，而不是分散在应用程序代码中。

让我们通过在数据上设置一些约束来改进我们的模型。首先，我们不应接受用户模型中用户名的 NULL 值，并确保用户名始终唯一。我们使用以下代码来完成：

```py
...
class User(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    username = db.Column(db.String(255), nullable=False, unique=True)
...
```

同样的原则适用于我们的其他模型：`Post`必须始终有一个标题，`Comment`总是由某人创建，`Tag`总是有一个标题，并且这个标题值是唯一的。我们使用以下代码来设置这些约束：

```py
...
class Post(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    title = db.Column(db.String(255), nullable=False)
...
class Comment(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
...
class Tag(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    title = db.Column(db.String(255), nullable=True, unique=True)
...
```

默认值非常好；它们确保数据质量，并使你的代码更短。我们可以让 SQLAlchemy 处理评论或帖子创建的日期时间戳，以下代码：

```py
class Comment(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
...
    date = db.Column(db.DateTime(), default=datetime.datetime.now)
...

class Post(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
...
    publish_date = db.Column(db.DateTime(), default=datetime.datetime.now)
```

注意 SQLAlchemy 如何处理默认定义。这是一个强大的功能。我们传递了一个 Python 函数的引用，因此只要不需要参数（除了部分函数），我们就可以使用任何我们想要的 Python 函数。这个函数将在创建记录或更新时被调用，并且它的返回值用于列的值。当然，SQLAlchemy 也支持简单的标量值在默认定义中。

RDBMS 索引用于提高查询性能，但你应该小心使用它们，因为这会在`INSERT`、`UPDATE`和`DELETE`函数上带来额外的写入，以及存储的增加。仔细选择和配置索引超出了本书的范围，但请考虑这样一个事实：索引用于减少对某些表列的 O(N)查找，这些列可能经常被使用，或者位于具有大量行的表中，在生产中线性查找根本不可能。索引查询性能可以从对数级提高到 O(1)。这是以额外的写入和检查为代价的。

以下代码示例展示了使用 Flask SQLAlchemy 创建索引的方法：

```py
...
class User(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    username = db.Column(db.String(255), nullable=False, index=True, unique=True)
...
```

以下代码展示了使用多个列的索引的示例：

```py
db.Index('idx_col_example', User.username, User.password)
```

# SQLAlchemy 会话的便利性

现在你已经了解了 SQLAlchemy 的力量以及 SQLAlchemy 会话对象是什么，以及为什么 Web 应用在没有任何会话的情况下不应该被创建。正如之前所述，会话可以简单地描述为一个跟踪我们模型中的更改并在我们告诉它时将它们提交到数据库的对象。然而，这不仅仅是这一点。

首先，会话也是**事务**的处理者。事务是一系列在提交时刷新到数据库中的更改集合。事务提供了许多隐藏的功能。例如，当对象有关系时，事务会自动确定哪些对象应该首先保存。你可能在前一节保存标签时注意到了这一点。当我们向帖子添加标签时，会话自动知道首先保存标签，尽管我们没有将它们添加为要提交的内容。如果我们使用原始 SQL 查询和数据库连接，我们将不得不跟踪哪些行与哪些其他行相关联，以避免保存一个指向不存在对象的键外键引用。

当对象的更改保存到数据库时，事务也会自动将数据标记为过时。下次我们访问该对象时，将查询数据库以更新数据，但所有这些都在幕后发生。如果我们不使用 SQLAlchemy，我们还需要手动跟踪哪些行需要更新。如果我们想资源高效，我们只需要查询和更新这些行。

其次，会话确保数据库中同一行的引用不会出现两个不同的情况。这是通过确保所有查询都通过会话进行（`Model.query` 实际上是 `db.session.query(Model)`），并且如果该行已经在当前事务中查询过，那么将返回对该对象的指针而不是一个新的对象。如果这个检查不存在，两个代表同一行的对象可能会以不同的更改保存到数据库中。这会创建一些微妙的错误，这些错误可能不会立即被发现。

请记住，Flask SQLAlchemy 为每个请求创建一个新的会话，并在请求结束时丢弃任何未提交的更改，所以请始终记得保存你的工作。

要深入了解会话，SQLAlchemy 的创建者 Mike Bayer 在 2012 年的 PyCon Canada 上发表了一次演讲。请参阅[`www.youtube.com/watch?v=PKAdehPHOMo`](https://www.youtube.com/watch?v=PKAdehPHOMo)上的*The SQLAlchemy Session - In Depth*。

# 使用 Alembic 进行数据库迁移

网络应用的功能总是在不断变化，并且随着每个新功能的加入，我们需要更改数据库的结构。无论是添加或删除新列还是创建新表，我们的模型将在应用的生命周期中不断变化。然而，当数据库经常变化时，问题会迅速出现。当我们把更改从开发环境迁移到生产环境时，如何确保没有手动比较每个模型及其对应的表就完成了所有更改？假设你想回到你的 Git 历史记录中查看你的应用早期版本是否有你现在在生产环境中遇到的相同错误。在没有大量额外工作的前提下，你将如何将数据库更改回正确的模式？

作为程序员，我们讨厌额外的工作。幸运的是，有一个名为 **Alembic** 的工具，它可以自动从我们的 SQLAlchemy 模型的更改中创建和跟踪数据库迁移。**数据库迁移**是我们模式所有更改的记录。Alembic 允许我们将数据库升级或降级到特定的保存版本。通过几个版本进行升级或降级将执行两个选定版本之间的所有文件。Alembic 最好的地方在于，其历史文件仅仅是 Python 文件。当我们创建第一个迁移时，我们可以看到 Alembic 语法是多么简单。

Alembic 并不捕获所有可能的变化——例如，它不会记录 SQL 索引上的变化。每次迁移后，都鼓励读者回顾迁移文件并进行任何必要的修正。

我们不会直接使用 Alembic。相反，我们将使用 **Flask-Migrate**，这是一个专门为 SQLAlchemy 创建的扩展，并且与 Flask CLI 一起工作。您可以在以下代码中找到它，在 `requirements.txt` 文件中，如下所示：

```py
Flask-Migrate
```

要开始，我们不需要向我们的 `manage.py` 文件中添加任何内容，因为 Flask-Migrate 已经通过其自己的 CLI 选项扩展了 Flask CLI，如下所示：

```py
from main import app, db, User, Post, Tag, migrate

@app.shell_context_processor
def make_shell_context():
    return dict(app=app, db=db, User=User, Post=Post, Tag=Tag, migrate=migrate)
```

在我们的 `main.py` 中：

```py
import datetime

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import DevConfig

app = Flask(__name__)
app.config.from_object(DevConfig)

db = SQLAlchemy(app)
migrate = Migrate(app, db)
```

要使用我们的应用程序和 SQLAlchemy 实例初始化 `Migrate` 对象，请运行以下代码：

```py
    # Tell Flask where is our app
    $ export FLASK_APP=main.py
    $ flask db
```

要开始跟踪我们的更改，我们使用 `init` 命令，如下所示：

```py
    $ flask db init
```

这将在我们的目录中创建一个新的名为 `migrations` 的文件夹，用于存储所有历史记录。现在我们开始第一个迁移，如下所示：

```py
    $ flask db migrate -m"initial migration"

```

此命令将导致 Alembic 扫描我们的 SQLAlchemy 对象，并找到所有在此提交之前不存在的表和列。由于这是我们第一次提交，迁移文件将会相当长。请务必使用 `-m` 选项指定迁移信息，因为这是识别每次迁移所做操作的最简单方法。每个迁移文件都存储在 `migrations/versions/` 文件夹中。

要将迁移应用到数据库并更改架构，请运行以下代码：

```py
$ flask db upgrade
```

如果我们想检查所有 SQLAlchemy 生成的 DDL 代码，则使用以下代码：

```py
$ flask db upgrade --sql
```

要返回到上一个版本，使用 `history` 命令找到版本号，并将其传递给 `downgrade` 命令，如下所示：

```py
$ flask db history
<base> -> 7ded34bc4fb (head), initial migration
$ flask db downgrade 7ded34bc4fb
```

如同 Git，一个哈希标记每个迁移。这是 Alembic 的主要功能，但它只是表面层次。尝试将您的迁移与 Git 提交对齐，以便在回滚提交时更容易降级或升级。

在本书的代码中，您将在每一章找到一个初始化脚本，该脚本将创建一个 Python 虚拟环境，安装所有声明的依赖项，并初始化数据库。请查看 `init.sh` Bash 脚本。

# 摘要

现在我们已经掌握了数据控制，我们可以继续在我们的应用程序中显示数据。下一章，`第三章`，*使用模板创建视图*，将动态介绍基于我们的模型创建 HTML 以及从我们的网络界面添加模型。
