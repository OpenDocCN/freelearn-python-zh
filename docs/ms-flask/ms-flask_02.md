# 第二章：使用 SQLAlchemy 创建模型

如前所述，**模型**是一种抽象和给数据提供一个通用接口的方式。在大多数 Web 应用程序中，数据存储和检索是通过**关系数据库管理系统**（**RDBMS**）进行的，这是一个以行和列的表格格式存储数据并能够在表格之间比较数据的数据库。一些例子包括 MySQL，Postgres，Oracle 和 MSSQL。

为了在我们的数据库上创建模型，我们将使用一个名为**SQLAlchemy**的 Python 包。SQLAlchemy 在其最低级别是一个数据库 API，并在其最高级别执行**对象关系映射**（**ORM**）。ORM 是一种在不同类型的系统和数据结构之间传递和转换数据的技术。在这种情况下，它将数据库中大量类型的数据转换为 Python 中类型和对象的混合。此外，像 Python 这样的编程语言允许您拥有不同的对象，这些对象相互引用，并获取和设置它们的属性。ORM，如 SQLAlchemy，有助于将其转换为传统数据库。

为了将 SQLAlchemy 与我们的应用程序上下文联系起来，我们将使用 Flask SQLAlchemy。Flask SQLAlchemy 是 SQLAlchemy 的一个便利层，提供了有用的默认值和特定于 Flask 的函数。如果您已经熟悉 SQLAlchemy，那么您可以在没有 Flask SQLAlchemy 的情况下自由使用它。

在本章结束时，我们将拥有一个完整的博客应用程序的数据库架构，以及与该架构交互的模型。

# 设置 SQLAlchemy

为了在本章中跟进，如果您还没有运行的数据库，您将需要一个。如果您从未安装过数据库，或者您没有偏好，SQLite 是初学者的最佳选择。

**SQLite**是一种快速的 SQL，无需服务器即可工作，并且完全包含在一个文件中。此外，SQLite 在 Python 中有原生支持。如果您选择使用 SQLite，将在*我们的第一个模型*部分为您创建一个 SQLite 数据库。

## Python 包

要使用`pip`安装 Flask SQLAlchemy，请运行以下命令：

```py
$ pip install flask-sqlalchemy

```

我们还需要安装特定的数据库包，用于作为 SQLAlchemy 的连接器。SQLite 用户可以跳过此步骤：

```py
# MySQL
$ pip install PyMySQL
# Postgres
$ pip install psycopg2
# MSSQL
$ pip install pyodbc
# Oracle
$ pip install cx_Oracle

```

## Flask SQLAlchemy

在我们可以抽象化我们的数据之前，我们需要设置 Flask SQLAlchemy。SQLAlchemy 通过特殊的数据库 URI 创建其数据库连接。这是一个看起来像 URL 的字符串，包含 SQLAlchemy 连接所需的所有信息。它的一般形式如下：

```py
databasetype+driver://user:password@ip:port/db_name
```

对于您之前安装的每个驱动程序，URI 将是：

```py
# SQLite
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

您可能已经注意到，我们实际上没有在我们的数据库中创建任何表来进行抽象。这是因为 SQLAlchemy 允许我们从表中创建模型，也可以从我们的模型中创建表。这将在我们创建第一个模型后进行介绍。

在我们的`main.py`文件中，必须首先使用以下方式初始化 SQLAlchemy：

```py
from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config.from_object(DevConfig)
db = SQLAlchemy(app)

```

SQLAlchemy 将读取我们应用程序的配置，并自动连接到我们的数据库。让我们在`main.py`文件中创建一个`User`模型，以与用户表进行交互：

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

我们取得了什么成就？我们现在有一个基于用户表的模型，有三列。当我们从`db.Model`继承时，与数据库的整个连接和通信将已经为我们处理。

每个`db.Column`实例的类变量代表数据库中的一列。`db.Column`实例中有一个可选的第一个参数，允许我们指定数据库中列的名称。如果没有，SQLAlchemy 会假定变量的名称与列的名称相同。使用这个可选变量会看起来像这样：

```py
username = db.Column('user_name', db.String(255))
```

`db.Column`的第二个参数告诉 SQLAlchemy 应将该列视为什么类型。本书中我们将使用的主要类型是：

+   `db.String`

+   `db.Text`

+   `db.Integer`

+   `db.Float`

+   `db.Boolean`

+   `db.Date`

+   `db.DateTime`

+   `db.Time`

每种类型代表的含义都相当简单。`String`和`Text`类型接受 Python 字符串并将它们分别转换为`varchar`和`text`类型的列。`Integer`和`Float`类型接受任何 Python 数字并在将它们插入数据库之前将它们转换为正确的类型。布尔类型接受 Python 的`True`或`False`语句，并且如果数据库有`boolean`类型，则将布尔值插入数据库。如果数据库中没有`boolean`类型，SQLAlchemy 会自动在 Python 布尔值和数据库中的 0 或 1 之间进行转换。`Date`、`DateTime`和`Time`类型使用`datetime`本地库中同名的 Python 类型，并将它们转换为数据库中的类型。`String`、`Integer`和`Float`类型接受一个额外的参数，告诉 SQLAlchemy 我们列的长度限制。

### 注意

如果您希望真正了解 SQLAlchemy 如何将您的代码转换为 SQL 查询，请将以下内容添加到`DevConfig`文件中：

```py
SQLALCHMEY_ECHO = True
```

这将在终端上打印出创建的查询。随着您在本书中的进展，您可能希望关闭此功能，因为每次加载页面时可能会打印出数十个查询。

参数`primary_key`告诉 SQLAlchemy 该列具有**主键索引**。每个 SQLAlchemy 模型*都需要*一个主键才能正常工作。

SQLAlchemy 将假定您的表名是模型类名的小写版本。但是，如果我们希望我们的表被称为除了*users*之外的其他名称呢？要告诉 SQLAlchemy 使用什么名称，请添加`__tablename__`类变量。这也是连接到已经存在于数据库中的表的方法。只需将表的名称放在字符串中。

```py
class User(db.Model):
    __tablename__ = 'user_table_name'

    id = db.Column(db.Integer(), primary_key=True)
    username = db.Column(db.String(255))
    password = db.Column(db.String(255))
```

我们不必包含`__init__`或`__repr__`函数。如果不包含，那么 SQLAlchemy 将自动创建一个接受列的名称和值作为关键字参数的`__init__`函数。

## 创建用户表

使用 SQLAlchemy 来完成繁重的工作，我们现在将在数据库中创建用户表。更新`manage.py`为：

```py
from main import app, db, User
...
@manager.shell
def make_shell_context():
    return dict(app=app, db=db, User=User)

Style - "db","User" in first line as Code Highlight
```

### 提示

从现在开始，每当我们创建一个新模型时，导入它并将其添加到返回的`dict`中。

这将允许我们在 shell 中使用我们的模型。现在运行 shell 并使用`db.create_all()`来创建所有表：

```py
$ python manage.py shell
>>> db.create_all()

```

现在您应该在数据库中看到一个名为`users`的表以及指定的列。此外，如果您使用 SQLite，您现在应该在文件结构中看到一个名为`database.db`的文件。

# CRUD

对于数据的每种存储机制，都有四种基本类型的函数：**创建、读取、更新和删除**（**CRUD**）。这些允许我们使用的所有基本方式来操作和查看我们的 Web 应用程序所需的数据。要使用这些函数，我们将在数据库上使用一个名为**session**的对象。会话将在本章后面进行解释，但现在，将其视为我们对数据库的所有更改的存储位置。

## 创建模型

要使用我们的模型在数据库中创建新行，请将模型添加到`session`和`commit`对象中。将对象添加到会话中标记其更改以进行保存，并且提交是将会话保存到数据库中的时候：

```py
>>> user = User(username='fake_name')
>>> db.session.add(user)
>>> db.session.commit()

```

向我们的表中添加新行非常简单。

## 读取模型

在向数据库添加数据后，可以使用`Model.query`来查询数据。对于使用 SQLAlchemy 的人来说，这是`db.session.query(Model)`的简写。

对于我们的第一个示例，使用`all()`来获取数据库中的所有行作为列表。

```py
>>> users = User.query.all()
>>> users
[<User 'fake_name'>]

```

当数据库中的项目数量增加时，此查询过程变得更慢。在 SQLAlchmey 中，与 SQL 一样，我们有限制功能来指定我们希望处理的总行数。

```py
>>> users = User.query.limit(10).all()

```

默认情况下，SQLAlchemy 返回按其主键排序的记录。要控制这一点，我们有 `order_by` 函数，它的用法是：

```py
# asending
>>> users = User.query.order_by(User.username).all()
# desending
>>> users = User.query.order_by(User.username.desc()).all()

```

要返回一个模型，我们使用 `first()` 而不是 `all()`：

```py
>>> user = User.query.first()
>>> user.username
fake_name

```

要通过其主键返回一个模型，使用 `query.get()`：

```py
>>> user = User.query.get(1)
>>> user.username
fake_name

```

所有这些函数都是可链式调用的，这意味着它们可以附加到彼此以修改返回结果。精通 JavaScript 的人会发现这种语法很熟悉。

```py
>>> users = User.query.order_by(
 User.username.desc()
 ).limit(10).first()

```

`first()` 和 `all()` 方法返回一个值，因此结束了链式调用。

还有一个特定于 Flask SQLAlchemy 的方法叫做 **pagination**，可以用来代替 `first()` 或 `all()`。这是一个方便的方法，旨在启用大多数网站在显示长列表项目时使用的分页功能。第一个参数定义了查询应该返回到哪一页，第二个参数是每页的项目数。因此，如果我们传递 1 和 10 作为参数，将返回前 10 个对象。如果我们传递 2 和 10，将返回对象 11-20，依此类推。

分页方法与 `first()` 和 `all()` 方法不同，因为它返回一个分页对象而不是模型列表。例如，如果我们想要获取博客中虚构的 `Post` 对象的第一页的前 10 个项目：

```py
>>> Post.query.paginate(1, 10)
<flask_sqlalchemy.Pagination at 0x105118f50>

```

这个对象有几个有用的属性：

```py
>>> page = User.query.paginate(1, 10)
# return the models in the page
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

### 过滤查询

现在我们来到了 SQL 的真正威力，即通过一组规则过滤结果。要获取满足一组相等条件的模型列表，我们使用 `query.filter_by` 过滤器。`query.filter_by` 过滤器接受命名参数，这些参数代表我们在数据库中每一列中寻找的值。要获取所有用户名为 `fake_name` 的用户列表：

```py
>>> users = User.query.filter_by(username='fake_name').all()

```

这个例子是在一个值上进行过滤，但多个值可以传递给 `filter_by` 过滤器。就像我们之前的函数一样，`filter_by` 是可链式调用的：

```py
>>> users = User.query.order_by(User.username.desc())
 .filter_by(username='fake_name')
 .limit(2)
 .all()

```

`query.filter_by` 只有在你知道你要查找的确切值时才有效。这可以通过将 Python 比较语句传递给 `query.filter` 来避免：

```py
>>> user = User.query.filter(
 User.id > 1
 ).all()

```

这是一个简单的例子，但 `query.filter` 接受任何 Python 比较。对于常见的 Python 类型，比如 `整数`、`字符串` 和 `日期`，可以使用 `==` 运算符进行相等比较。如果有一个 `整数`、`浮点数` 或 `日期` 列，也可以使用 `>`、`<`、`<=` 和 `>=` 运算符传递不等式语句。

我们还可以使用 SQLAlchemy 函数来转换复杂的 SQL 查询。例如，使用 `IN`、`OR` 或 `NOT` SQL 比较：

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

在 SQLAlchemy 中，与 `None` 的比较会被转换为与 `NULL` 的比较。

## 更新模型

要更新已经存在的模型的值，将 `update` 方法应用到查询对象上，也就是说，在你使用 `first()` 或 `all()` 等方法返回模型之前：

```py
>>> User.query.filter_by(username='fake_name').update({
 'password': 'test'
 })
# The updated models have already been added to the session
>>> db.session.commit()

```

## 删除模型

如果我们希望从数据库中删除一个模型：

```py
>>> user = User.query.filter_by(username='fake_name').first()
>>> db.session.delete(user)
>>> db.session.commit()

```

# 模型之间的关系

SQLAlchemy 中模型之间的关系是两个或多个模型之间的链接，允许模型自动引用彼此。这允许自然相关的数据，比如 *评论到帖子*，可以轻松地从数据库中检索其相关数据。这就是关系型数据库管理系统中的 *R*，它赋予了这种类型的数据库大量的能力。

让我们创建我们的第一个关系。我们的博客网站将需要一些博客文章。每篇博客文章将由一个用户撰写，因此将博客文章链接回撰写它们的用户是很有意义的，可以轻松地获取某个用户的所有博客文章。这是一个 **一对多** 关系的例子。

## 一对多

让我们添加一个模型来代表我们网站上的博客文章：

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

请注意`user_id`列。熟悉 RDBMS 的人会知道这代表**外键约束**。外键约束是数据库中的一条规则，强制`user_id`的值存在于用户表中的`id`列中。这是数据库中的一个检查，以确保`Post`始终引用现有用户。`db.ForeignKey`的参数是`user_id`字段的字符串表示。如果决定用`__table_name__`来命名用户表，必须更改此字符串。在初始化 SQLAlchemy 时，使用此字符串而不是直接引用`User.id`，因为`User`对象可能尚不存在。

`user_id`列本身不足以告诉 SQLAlchemy 我们有一个关系。我们必须修改我们的`User`模型如下：

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

`db.relationship`函数在 SQLAlchemy 中创建一个虚拟列，与我们的`Post`模型中的`db.ForeignKey`相连接。第一个参数是我们引用的类的名称。我们很快就会介绍`backref`的作用，但`lazy`参数是什么？`lazy`参数控制 SQLAlchemy 如何加载我们的相关对象。`subquery`会在加载我们的`Post`对象时立即加载我们的关系。这减少了查询的数量，但当返回的项目数量增加时，速度会变慢。相比之下，使用`dynamic`选项，相关对象将在访问时加载，并且可以在返回之前进行筛选。如果返回的对象数量很大或将变得很大，这是最好的选择。

我们现在可以访问`User.posts`变量，它将返回所有`user_id`字段等于我们的`User.id`的帖子的列表。让我们在 shell 中尝试一下：

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

请注意，如果没有将更改提交到数据库，我们将无法访问我们的关系中的帖子。

`backref`参数使我们能够通过`Post.user`访问和设置我们的`User`类。这是由以下给出的：

```py
>>> second_post = Post('Second Title')
>>> second_post.user = user
>>> db.session.add(second_post)
>>> db.session.commit()
>>> user.posts
[<Post 'Post Title'>, <Post 'Second Title'>]

```

因为`user.posts`是一个列表，我们也可以将我们的`Post`模型添加到列表中以自动保存它：

```py
>>> second_post = Post('Second Title')
>>> user.posts.append(second_post)
>>> db.session.add(user)
>>> db.session.commit()
>>> user.posts
[<Post 'Post Title'>, <Post 'Second Title'>]

```

使用`backref`选项作为 dynamic，我们可以将我们的关系列视为查询以及列表：

```py
>>> user.posts
[<Post 'Post Title'>, <Post 'Second Title'>]
>>> user.posts.order_by(Post.publish_date.desc()).all()
[<Post 'Second Title'>, <Post 'Post Title'>]

```

在我们继续下一个关系类型之前，让我们为用户评论添加另一个模型，它具有一对多的关系，稍后将在书中使用：

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

class Comment(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(255))
    text = db.Column(db.Text())
    date = db.Column(db.DateTime())
    post_id = db.Column(db.Integer(), db.ForeignKey('post.id'))

    def __repr__(self):
        return "<Comment '{}'>".format(self.text[:15])
```

## 多对多

如果我们有两个可以相互引用的模型，但每个模型都需要引用每种类型的多个模型，该怎么办？例如，我们的博客帖子将需要标签，以便我们的用户可以轻松地将相似的帖子分组。每个标签可以指向多个帖子，但每个帖子可以有多个标签。这种类型的关系称为**多对多**关系。考虑以下示例：

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

`db.Table`对象是对数据库的低级访问，比`db.Model`的抽象更低。`db.Model`对象建立在`db.Table`之上，并提供了表中特定行的表示。使用`db.Table`对象是因为不需要访问表的单个行。

`tags`变量用于表示`post_tags`表，其中包含两行：一行表示帖子的 id，另一行表示标签的 id。为了说明这是如何工作的，如果表中有以下数据：

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

SQLAlchemy 会将其转换为：

+   id 为`1`的帖子具有 id 为`1`和`3`的标签

+   id 为`2`的帖子具有 id 为`3`、`4`和`5`的标签

+   id 为`3`的帖子具有 id 为`1`和`2`的标签

您可以将这些数据描述为与帖子相关的标签。

在`db.relationship`函数设置我们的关系之前，但这次它有 secondary 参数。secondary 参数告诉 SQLAlchemy 这个关系存储在 tags 表中。让我们看看下面的代码：

```py
>>> post_one = Post.query.filter_by(title='Post Title').first()
>>> post_two = Post.query.filter_by(title='Second Title').first()
>>> tag_one = Tag('Python')
>>> tag_two = Tag('SQLAlchemy')
>>> tag_three = Tag('Flask')
>>> post_one.tags = [tag_two]
>>> post_two.tags = [tag_one, tag_two, tag_three]
>>> tag_two.posts
[<Post 'Post Title'>, <Post 'Second Title'>]
>>> db.session.add(post_one)
>>> db.session.add(post_two)
>>> db.session.commit()

```

在一对多关系中，主关系列只是一个列表。主要区别在于`backref`选项现在也是一个列表。因为它是一个列表，我们可以从`tag`对象中向标签添加帖子，如下所示：

```py
>>> tag_one.posts.append(post_one)
[<Post 'Post Title'>, <Post 'Second Title'>]
>>> post_one.tags
[<Tag 'SQLAlchemy'>, <Tag 'Python'>]
>>> db.session.add(tag_one)
>>> db.session.commit()

```

# SQLAlchemy 会话的便利性

现在您了解了 SQLAlchemy 的强大之处，也可以理解 SQLAlchemy 会话对象是什么，以及为什么 Web 应用程序不应该没有它们。正如之前所述，会话可以简单地描述为一个跟踪我们模型更改并在我们告诉它时将它们提交到数据库的对象。但是，它比这更复杂一些。

首先，会话是**事务**的处理程序。事务是在提交时刷新到数据库的一组更改。事务提供了许多隐藏的功能。例如，当对象具有关系时，事务会自动确定哪些对象将首先保存。您可能已经注意到了，在上一节中保存标签时。当我们将标签添加到帖子中时，会话自动知道首先保存标签，尽管我们没有将其添加到提交。如果我们使用原始 SQL 查询和数据库连接，我们将不得不跟踪哪些行与其他行相关，以避免保存对不存在的对象的外键引用。

事务还会在将对象的更改保存到数据库时自动将数据标记为陈旧。当我们下次访问对象时，将向数据库发出查询以更新数据，但所有这些都是在后台进行的。如果我们不使用 SQLAlchemy，我们还需要手动跟踪需要更新的行。如果我们想要资源高效，我们只需要查询和更新那些行。

其次，会话使得不可能存在对数据库中同一行的两个不同引用。这是通过所有查询都经过会话来实现的（`Model.query`实际上是`db.session.query(Model)`），如果在此事务中已经查询了该行，则将返回指向该对象的指针，而不是一个新对象。如果没有这个检查，表示同一行的两个对象可能会以不同的更改保存到数据库中。这会产生微妙的错误，可能不会立即被发现。

请记住，Flask SQLAlchemy 为每个请求创建一个新会话，并在请求结束时丢弃未提交的任何更改，因此请记住保存您的工作。

### 注意

要深入了解会话，SQLAlchemy 的创建者 Mike Bayer 在 2012 年加拿大 PyCon 上发表了一次演讲。请参阅*SQLAlchemy 会话-深入*，链接在这里-[`www.youtube.com/watch?v=PKAdehPHOMo`](https://www.youtube.com/watch?v=PKAdehPHOMo)。

# 使用 Alembic 进行数据库迁移

Web 应用程序的功能性不断变化，随着新功能的增加，我们需要改变数据库的结构。无论是添加或删除新列，还是创建新表，我们的模型都会在应用程序的生命周期中发生变化。然而，当数据库经常发生变化时，问题很快就会出现。在将我们的更改从开发环境移动到生产环境时，如何确保您在没有手动比较每个模型及其相应表的情况下携带了每个更改？假设您希望回到 Git 历史记录中查看您的应用程序的早期版本是否存在与您现在在生产环境中遇到的相同错误。在没有大量额外工作的情况下，您将如何将数据库更改回正确的模式？

作为程序员，我们讨厌额外的工作。幸运的是，有一个名为**Alembic**的工具，它可以根据我们的 SQLAlchemy 模型的更改自动创建和跟踪数据库迁移。**数据库迁移**是我们模式的所有更改的记录。Alembic 允许我们将数据库升级或降级到特定的保存版本。通过几个版本的升级或降级将执行两个选定版本之间的所有文件。Alembic 最好的部分是它的历史文件只是 Python 文件。当我们创建我们的第一个迁移时，我们可以看到 Alembic 语法是多么简单。

### 注意

Alembic 并不捕获每一个可能的变化。例如，它不记录 SQL 索引的更改。在每次迁移之后，建议读者查看迁移文件并进行任何必要的更正。

我们不会直接使用 Alembic；相反，我们将使用**Flask-Migrate**，这是专门为 SQLAlchemy 创建的扩展，并与 Flask Script 一起使用。要使用`pip`安装它：

```py
$ pip install Flask-Migrate

```

要开始，我们需要将命令添加到我们的`manage.py`文件中，如下所示：

```py
from flask.ext.script import Manager, Server
from flask.ext.migrate import Migrate, MigrateCommand

from main import app, db, User, Post, Tag

migrate = Migrate(app, db)

manager = Manager(app)
manager.add_command("server", Server())
manager.add_command('db', MigrateCommand)

@manager.shell
def make_shell_context():
    return dict(app=app, db=db, User=User, Post=Post, Tag=Tag)

if __name__ == "__main__":
    manager.run()
```

我们使用我们的应用程序和我们的 SQLAlchemy 实例初始化了`Migrate`对象，并且通过`manage.py db`使迁移命令可调用。要查看可能的命令列表，请运行此命令：

```py
$ python manage.py db

```

要开始跟踪我们的更改，我们使用`init`命令如下：

```py
$ python manage.py db init

```

这将在我们的目录中创建一个名为`migrations`的新文件夹，其中将保存我们的所有历史记录。现在我们开始进行我们的第一个迁移：

```py
$ python manage.py  db migrate -m"initial migration"

```

这个命令将导致 Alembic 扫描我们的 SQLAlchemy 对象，并找到所有在此提交之前不存在的表和列。由于这是我们的第一个提交，迁移文件会相当长。一定要使用`-m`指定迁移消息，因为这是识别每个迁移在做什么的最简单方法。每个迁移文件都存储在`migrations/versions/`文件夹中。

要将迁移应用到您的数据库并更改模式，请运行以下命令：

```py
$ python manage.py db upgrade

```

要返回到以前的版本，使用`history`命令找到版本号，并将其传递给`downgrade`命令：

```py
$ python manage.py db history
<base> -> 7ded34bc4fb (head), initial migration
$ python manage.py db downgrade 7ded34bc4fb

```

就像 Git 一样，每个迁移都有一个哈希标记。这是 Alembic 的主要功能，但这只是表面层次。尝试将您的迁移与 Git 提交对齐，以便在还原提交时更容易降级或升级。

# 总结

现在我们已经掌握了数据控制，我们现在可以继续在我们的应用程序中显示我们的数据。下一章，第三章 *使用模板创建视图*，将动态地涵盖根据我们的模型创建基于 HTML 的视图，并从我们的 Web 界面添加模型。
