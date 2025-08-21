# 第四章：Socializer-可测试的时间线

在本章中，我们将使用代号“Socializer”构建我们的下一个应用程序。这个应用程序将为您提供一个非常典型的*时间线*信息流，其变体出现在许多知名的现代网络应用程序中。

这个应用程序将允许经过身份验证的用户关注其他用户，并被其他用户关注，并以时间顺序显示被关注用户发布的内容。除了构建基于时间线的应用程序所需的基本功能之外，我们还将使用优秀的`Blinker`库来实现其他行为，以进行进程内发布/订阅信号，这将使我们能够将应用程序解耦为更可组合、可重用的部分。

此外，Socializer 将在构建过程中考虑单元测试和功能测试，使我们能够对各种模型和视图进行严格测试，以确保其按照我们的期望进行功能。

# 开始

就像我们在上一章中所做的那样，让我们为这个应用程序创建一个全新的目录，另外创建一个虚拟环境，并安装我们将要使用的一些基本包：

```py
$ mkdir -p ~/src/socializer && cd ~/src/socializer
$ mkvirtualenv socializer
$ pip install flask flask-sqlalchemy flask-bcrypt flask-login flask-wtf blinker pytest-flask

```

我们的应用程序布局暂时将与上一章中使用的布局非常相似：

```py
├── application
│   ├── __init__.py
│   └── users
│       ├── __init__.py
│       ├── models.py
│       └── views.py
└── run.py
└── database.py

```

# 应用程序工厂

单元测试和功能测试的一个主要好处是能够在各种不同条件和配置下确保应用程序以已知和可预测的方式运行。为此，在我们的测试套件中构建所有 Flask 应用程序对象将是一个巨大的优势。然后，我们可以轻松地为这些对象提供不同的配置，并确保它们表现出我们期望的行为。

值得庆幸的是，这完全可以通过应用程序工厂模式来实现，而 Flask 对此提供了很好的支持。让我们在`application/__init__.py`模块中添加一个`create_app`方法：

```py
from flask import Flask

def create_app(config=None):
 app = Flask(__name__)

 if config is not None:
 app.config.from_object(config)

 return app

```

这个方法的作用相对简单：给定一个可选的`config`参数，构建一个 Flask 应用程序对象，可选地应用这个自定义配置，最后将新创建的 Flask 应用程序对象返回给调用者。

以前，我们只是在模块本身中实例化一个 Flask 对象，这意味着在导入此包或模块时，应用程序对象将立即可用。然而，这也意味着没有简单的方法来做以下事情：

+   将应用程序对象的构建延迟到模块导入到本地命名空间之后的某个时间。这一开始可能看起来很琐碎，但对于可以从这种惰性实例化中受益的大型应用程序来说，这是非常有用和强大的。正如我们之前提到的，应尽可能避免产生副作用的包导入。

+   替换不同的应用程序配置值，例如在运行测试时可能需要的配置值。例如，我们可能希望在运行测试套件时避免向真实用户发送电子邮件通知。

+   在同一进程中运行多个 Flask 应用程序。虽然本书中没有明确讨论这个概念，但在各种情况下这可能是有用的，比如拥有为公共 API 的不同版本提供服务的单独应用程序实例，或者为不同内容类型（JSON、XML 等）提供服务的单独应用程序对象。关于这个主题的更多信息可以从官方 Flask 在线文档的*应用程序调度*部分中获取[`flask.pocoo.org/docs/0.10/patterns/appdispatch/`](http://flask.pocoo.org/docs/0.10/patterns/appdispatch/)。

有了应用程序工厂，我们现在在何时以及如何构建我们的主应用程序对象方面有了更多的灵活性。当然，缺点（或者优点，如果你打算在同一个进程中运行多个应用程序！）是，我们不再可以访问一个准全局的`app`对象，我们可以导入到我们的模块中，以便注册路由处理程序或访问`app`对象的日志记录器。

## 应用程序上下文

Flask 的主要设计目标之一是确保您可以在同一个 Python 进程中运行多个应用程序。那么，一个应用程序如何确保被导入到模块中的`app`对象是正确的，而不是在同一个进程中运行的其他应用程序的对象？

在支持单进程/多应用程序范式的其他框架中，有时可以通过强制显式依赖注入来实现：需要`app`对象的代码应明确要求将 app 对象传递给需要它的函数或方法。从架构设计的角度来看，这听起来很棒，但如果第三方库或扩展不遵循相同的设计原则，这很快就会变得繁琐。最好的情况是，您最终将需要编写大量的样板包装函数，最坏的情况是，您最终将不得不诉诸于在模块和类中进行猴子补丁，这将最终导致比您最初预期的麻烦更多的脆弱性和不必要的复杂性。

### 注意

当然，显式依赖注入样板包装函数本身并没有什么不对。Flask 只是选择了一种不同的方法，过去曾因此受到批评，但已经证明是灵活、可测试和有弹性的。

Flask，不管好坏，都是建立在基于代理对象的替代方法之上的。这些代理对象本质上是容器对象，它们在所有线程之间共享，并且知道如何分派到在幕后绑定到特定线程的*真实*对象。

### 注意

一个常见的误解是，在多线程应用程序中，根据 WSGI 规范，每个请求将被分配一个新的线程：这根本不是事实。新请求可能会重用现有但当前未使用的线程，并且这个旧线程可能仍然存在局部作用域的变量，可能会干扰您的新请求处理。

其中一个代理对象`current_app`被创建并绑定到当前请求。这意味着，我们不再导入一个已经构建好的 Flask 应用程序对象（或者更糟糕的是，在同一个请求中创建额外的应用程序对象），而是用以下内容替换它：

```py
from flask import current_app as app

```

### 提示

当然，导入的`current_app`对象的别名是完全可选的。有时最好将其命名为`current_app`，以提醒自己它不是真正的应用程序对象，而是一个代理对象。

使用这个代理对象，我们可以规避在实现应用程序工厂模式时，在导入时没有可用的实例化 Flask 应用程序对象的问题。

### 实例化一个应用程序对象

当然，我们需要在某个时候实际创建一个应用程序对象，以便代理有东西可以代理。通常，我们希望创建对象一次，然后确保调用`run`方法以启动 Werkzeug 开发服务器。

为此，我们可以修改上一章中的`run.py`脚本，从我们的工厂实例化 app 对象，并调用新创建的实例的`run`方法，如下所示：

```py
from application import create_app

app = create_app()
app.run(debug=True)

```

现在，我们应该能够像以前一样运行这个极其简陋的应用程序：

```py
$ python run.py

```

### 提示

还可以调用 Python 解释器，以便为您导入并立即执行模块、包或脚本。这是通过`-m`标志实现的，我们之前对`run.py`的调用可以修改为更简洁的版本，如下所示：

```py
$ python –m run

```

# 单元和功能测试

实现应用程序工厂以分发 Flask 应用程序实例的主要好处之一是，我们可以更有效地测试应用程序。我们可以为不同的测试用例构建不同的应用程序实例，并确保它们尽可能地相互隔离（或者尽可能地与 Flask/Werkzeug 允许的隔离）。

Python 生态系统中测试库的主要组成部分是 unittest，它包含在标准库中，并包括了 xUnit 框架所期望的许多功能。虽然本书不会详细介绍 unittest，但一个典型的基于类的测试用例将遵循以下基本结构，假设我们仍然使用工厂模式来将应用程序配置与实例化分离：

```py
from myapp import create_app
import unittest

class AppTestCase(unittest.TestCase):

 def setUp(self):
 app = create_app()  # Could also pass custom settings.
 app.config['TESTING'] = True
 self.app = app

 # Whatever DB initialization is required

 def tearDown(self):
 # If anything needs to be cleaned up after a test.
 Pass

 def test_app_configuration(self):
 self.assertTrue(self.app.config['TESTING'])
 # Other relevant assertions

if __name__ == '__main__':
 unittest.main()

```

使用 unittest 测试格式/样式的优点如下：

+   不需要外部依赖；unittest 是 Python 标准库的一部分。

+   入门相对容易。大多数 xUnit 测试框架遵循类似的命名约定来声明测试类和测试方法，并包含几个典型断言的辅助函数，如`assertTrue`或`assertEqual`等。

然而，它并不是唯一的选择；我们将使用`pytest`和包装方便功能的相关 Flask 扩展`pytest-flask`。

除了作为一个稍微现代化和简洁的测试框架外，`pytest`相对于许多其他测试工具提供的另一个主要优势是能够为测试定义固定装置，这在它们自己的文档中描述得非常简洁，如下所示：

+   固定装置具有明确的名称，并通过声明其在测试函数、模块、类或整个项目中的使用来激活它们

+   固定装置以模块化的方式实现，因为每个固定装置名称都会触发一个固定装置函数，该函数本身可以使用其他固定装置

+   固定装置管理从简单单元到复杂功能测试的规模，允许您根据配置和组件选项对固定装置和测试进行参数化，或者在类、模块或整个测试会话范围内重用固定装置

在测试 Flask 应用程序的情况下，这意味着我们可以在`fixture`中定义对象（例如我们的应用程序对象），然后通过使用与定义的固定装置函数相同名称的参数，将该对象自动注入到测试函数中。

如果上一段文字有点难以理解，那么一个简单的例子就足以澄清问题。让我们创建以下的`conftest.py`文件，其中将包含任何测试套件范围的固定装置和辅助工具，供其他测试使用：

```py
import pytest
from application import create_app

@pytest.fixture
def app():
 app = create_app()
 return app

```

我们将在`tests/test_application.py`中创建我们的第一个测试模块，如下所示：

### 提示

请注意`tests_*`前缀对于测试文件名是重要的——它允许`pytest`自动发现哪些文件包含需要运行的测试函数和断言。如果您的 tests/folder 中的文件名没有上述前缀，那么测试运行器将放弃加载它，并将其视为包含具有测试断言的函数的文件。

```py
import flask

def test_app(app):
 assert isinstance(app, flask.Flask)

```

### 请注意

请注意`test_app`函数签名中的`app`参数与`conftest.py`中定义的`app`固定装置函数的名称相匹配，传递给`test_app`的值是`app`固定装置函数的返回值。

我们将使用安装到我们的虚拟环境中的`py.test`可执行文件来运行测试套件（当我们添加了`pytest-flask`和`pytest`库时），在包含`conftest.py`和我们的 tests/文件夹的目录中运行，输出将指示我们的测试模块已被发现并运行：

```py
$ py.test
=============== test session starts ================
platform darwin -- Python 2.7.8 -- py-1.4.26 -- pytest-2.7.0
rootdir: /path/to/socializer, inifile:
plugins: flask
collected 1 items

tests/test_application.py .

============= 1 passed in 0.02 seconds =============

```

就是这样！我们已经编写并运行了我们的第一个应用程序测试，尽管不是很有趣。如果你还不太明白发生了什么，不要担心；本章中将进行大量具体的测试，还会有更多的例子。

# 社交功能-朋友和关注者

许多现代网络应用程序允许用户*添加朋友*或*关注*其他用户，并且自己也可以被添加朋友或关注。虽然这个概念在文字上可能很简单，但有许多实现和变体，所有这些都针对它们特定的用例进行了优化。

在这种情况下，我们想要实现一个类似新闻订阅的服务，该服务会显示来自选定用户池的信息，并在每个经过身份验证的用户中显示独特的聚合时间线，以下是可能使用的三种方法类别：

+   **写入时的扇出**：每个用户的新闻订阅都存储在一个单独的逻辑容器中，旨在使读取非常简单、快速和直接，但代价是去规范化和较低的写入吞吐量。逻辑容器可以是每个用户的数据库表（尽管对于大量用户来说效率非常低），也可以是列式数据库（如 Cassandra）中的列，或者更专门的存储解决方案，如 Redis 列表，可以以原子方式向其中添加元素。

+   **读取时的扇出**：当新闻订阅需要额外的定制或处理来确定诸如可见性或相关性之类的事情时，通常最好使用读取时的扇出方法。这允许更精细地控制哪些项目将出现在动态信息中，以及以哪种顺序（假设需要比时间顺序更复杂的东西），但这会增加加载用户特定动态信息的计算时间。通过将最近的项目保存在 RAM 中（这是 Facebook™新闻订阅背后的基本方法，也是 Facebook 在世界上部署最大的 Memcache 的原因），但这会引入几层复杂性和间接性。

+   **天真的规范化**：这是方法中最不可扩展的，但实现起来最简单。对于许多小规模应用程序来说，这是最好的起点：一个包含所有用户创建的项目的帖子表（带有对创建该特定项目的用户的外键约束）和一个跟踪哪些用户正在关注谁的关注者表。可以使用各种缓存解决方案来加速请求的部分，但这会增加额外的复杂性，并且只有在必要时才能引入。

对于我们的 Socializer 应用程序，第三种方法，所谓的天真规范化，将是我们实现的方法。其他方法也是有效的，你可以根据自己的目标选择其中任何一条路线，但出于简单和阐述的目的，我们将选择需要最少工作量的方法。

有了这个想法，让我们开始实现所需的基本 SQLAlchemy 模型和关系。首先，让我们使用我们新创建的应用程序工厂来初始化和配置 Flask-SQLAlchemy 扩展，以及使用相同的混合属性方法来哈希我们的用户密码，这是我们在上一章中探讨过的方法。我们的`application/__init__.py`如下：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.bcrypt import Bcrypt

# Initialize the db extension, but without configuring
# it with an application instance.
db = SQLAlchemy()

# The same for the Bcrypt extension
flask_bcrypt = Bcrypt()

def create_app(config=None):
 app = Flask(__name__)

 if config is not None:
 app.config.from_object(config)

 # Initialize any extensions and bind blueprints to the
 # application instance here.
 db.init_app(app)
 flask_bcrypt.init_app(app)

 return app

```

由于应用程序工厂的使用，我们将扩展（`db`和`flask_bcrypt`）的实例化与它们的配置分开。前者发生在导入时，后者需要在构建 Flask 应用对象时发生。幸运的是，大多数现代的 Flask 扩展都允许发生这种确切的分离，正如我们在前面的片段中所演示的那样。

现在，我们将通过创建`application/users/__init__.py`来创建我们的用户包，然后我们将创建`application/users/models.py`，其中包含我们用于 Flask-Login 扩展的标准部分（稍后我们将使用），就像我们在上一章中所做的那样。此外，我们将为我们的关注者表和用户模型上的关联关系添加一个显式的 SQLAlchemy 映射：

```py
import datetime
from application import db, flask_bcrypt
from sqlalchemy.ext.hybrid import hybrid_property

__all__ = ['followers', 'User']

# We use the explicit SQLAlchemy mappers for declaring the
# followers table, since it does not require any of the features
# that the declarative base model brings to the table.
#
# The `follower_id` is the entry that represents a user who
# *follows* a `user_id`.
followers = db.Table(
 'followers',
 db.Column('follower_id', db.Integer, db.ForeignKey('user.id'),
 primary_key=True),
 db.Column('user_id', db.Integer, db.ForeignKey('user.id'),
 primary_key=True))

class User(db.Model):

 # The primary key for each user record.
 id = db.Column(db.Integer, primary_key=True)

 # The unique email for each user record.
 email = db.Column(db.String(255), unique=True)

 # The unique username for each record.
 username = db.Column(db.String(40), unique=True)

 # The hashed password for the user
 _password = db.Column('password', db.String(60))
 #  The date/time that the user account was created on.
 created_on = db.Column(db.DateTime,
 default=datetime.datetime.utcnow)

 followed = db.relationship('User',
 secondary=followers,
 primaryjoin=(id==followers.c.follower_id ),
 secondaryjoin=(id==followers.c.user_id),
 backref=db.backref('followers', lazy='dynamic'),
 lazy='dynamic')

 @hybrid_property
 def password(self):
 """The bcrypt'ed password of the given user."""

 return self._password

 @password.setter
 def password(self, password):
 """Bcrypt the password on assignment."""

 self._password = flask_bcrypt.generate_password_hash(
 password)

 def __repr__(self):
 return '<User %r>' % self.username

 def is_authenticated(self):
 """All our registered users are authenticated."""
 return True

 def is_active(self):
 """All our users are active."""
 return True

 def is_anonymous(self):
 """We don't have anonymous users; always False"""
 return False
 def get_id(self):
 """Get the user ID."""
 return unicode(self.id)

```

用户模型的`followed`属性是一个 SQLAlchemy 关系，它通过中间的关注者表将用户表映射到自身。由于社交连接需要隐式的多对多关系，中间表是必要的。仔细看一下`followed`属性，如下所示的代码：

```py
 followed = db.relationship('User',
 secondary=followers,
 primaryjoin=(id==followers.c.follower_id ),
 secondaryjoin=(id==followers.c.user_id),
 backref=db.backref('followers', lazy='dynamic'),
 lazy='dynamic')

```

我们可以看到，与本章和以前章节中使用的常规列定义相比，声明有些复杂。然而，`relationship`函数的每个参数都有一个非常明确的目的，如下列表所示：

+   `User`：这是目标关系类的基于字符串的名称。这也可以是映射类本身，但是那样你可能会陷入循环导入问题的泥潭。

+   `primaryjoin`：这个参数的值将被评估，然后用作主表（`user`）到关联表（`follower`）的`join`条件。

+   `secondaryjoin`：这个参数的值，类似于`primaryjoin`，在关联表（`follower`）到子表（`user`）的`join`条件中被评估并使用。由于我们的主表和子表是一样的（用户关注其他用户），这个条件几乎与`primaryjoin`参数中产生的条件相同，只是在关联表中映射的键方面有所不同。

+   `backref`：这是将插入到实例上的属性的名称，该属性将处理关系的反向方向。这意味着一旦我们有了一个用户实例，我们就可以访问`user.followers`来获取关注给定用户实例的人的列表，而不是`user.followed`属性，其中我们明确定义了当前用户正在关注的用户列表。

+   `lazy`：这是任何基于关系的属性最常被误用的属性。有各种可用的值，包括`select`、`immediate`、`joined`、`subquery`、`noload`和`dynamic`。这些确定了相关数据的加载方式或时间。对于我们的应用程序，我们选择使用`dynamic`的值，它不返回一个可迭代的集合，而是返回一个可以进一步细化和操作的`Query`对象。例如，我们可以做一些像`user.followed.filter(User.username == 'example')`这样的事情。虽然在这种特定情况下并不是非常有用，但它提供了巨大的灵活性，有时以生成效率较低的 SQL 查询为代价。

我们将设置各种属性，以确保生成的查询使用正确的列来创建自引用的多对多连接，并且只有在需要时才执行获取关注者列表的查询。关于这些特定模式的更多信息可以在官方的 SQLAlchemy 文档中找到：[`docs.sqlalchemy.org/en/latest/`](http://docs.sqlalchemy.org/en/latest/)。

现在，我们将为我们的用户模型添加一些方法，以便便于关注/取消关注其他用户。由于 SQLAlchemy 的一些内部技巧，为用户添加和移除关注者可以表达为对本地 Python 列表的操作，如下所示：

```py
def unfollow(self, user):
 """
 Unfollow the given user.

 Return `False` if the user was not already following the user.
 Otherwise, remove the user from the followed list and return
 the current object so that it may then be committed to the 
 session.
 """

 if not self.is_following(user):
 return False

 self.followed.remove(user)
 return self

def follow(self, user):
 """
 Follow the given user.
 Return `False` if the user was already following the user.
 """

 if self.is_following(user):
 return False

 self.followed.append(user)
 return self

def is_following(self, user):
 """
 Returns boolean `True` if the current user is following the
 given `user`, and `False` otherwise.
 """
 followed = self.followed.filter(followers.c.user_id == user.id)
 return followed.count() > 0

```

### 注意

实际上，您并不是在原生的 Python 列表上操作，而是在 SQLAlchemy 知道如何跟踪删除和添加的数据结构上操作，然后通过工作单元模式将这些同步到数据库。

接下来，我们将在`application/posts/models.py`的蓝图模块中创建`Post`模型。像往常一样，不要忘记创建`application/posts/__init__.py`文件，以便将文件夹声明为有效的 Python 包，否则在尝试运行应用程序时将出现一些非常令人困惑的导入错误。

目前，这个特定的模型将是一个简单的典范。以下是该项目的用户模型的当前实现：

```py
from application import db
import datetime

__all__ = ['Post']

class Post(db.Model):

 # The unique primary key for each post created.
 id = db.Column(db.Integer, primary_key=True)
 # The free-form text-based content of each post.
 content = db.Column(db.Text())

 #  The date/time that the post was created on.
 created_on = db.Column(db.DateTime(),
 default=datetime.datetime.utcnow, index=True)

 # The user ID that created this post.
 user_id = db.Column(db.Integer(), db.ForeignKey('user.id'))

 def __repr__(self):
 return '<Post %r>' % self.body

```

一旦我们定义了`Post`模型，我们现在可以为用户模型添加一个方法，该方法允许我们获取与当前实例链接的用户的新闻源。我们将该方法命名为`newsfeed`，其实现如下：

```py
def newsfeed(self):
 """
 Return all posts from users followed by the current user,
 in descending chronological order.

 """

 join_condition = followers.c.user_id == Post.user_id
 filter_condition = followers.c.follower_id == self.id
 ordering = Post.created_on.desc()

 return Post.query.join(followers,
 (join_condition)).filter(
 filter_condition).order_by(ordering)

```

### 注意

请注意，为了实现上述方法，我们必须将`Post`模型导入到`application/users/models.py`模块中。虽然这种特定的情况将正常运行，但必须始终注意可能会有一些难以诊断的潜在循环导入问题。

# 功能和集成测试

在大多数单元、功能和集成测试的处理中，通常建议在编写相应的代码之前编写测试。虽然这通常被认为是一个良好的实践，出于各种原因（主要是允许您确保正在编写的代码解决了已定义的问题），但为了简单起见，我们等到现在才涉及这个主题。

首先，让我们创建一个新的`test_settings.py`文件，它与我们现有的`settings.py`同级。这个新文件将包含我们在运行测试套件时想要使用的应用程序配置常量。最重要的是，它将包含一个指向不是我们应用程序数据库的数据库的 URI，如下所示：

```py
SQLALCHEMY_DATABASE_URI = 'sqlite:////tmp/test_app.db'
DEBUG = True
TESTING = True

```

### 注意

前面的`SQLALCHEMY_DATABASE_URI`字符串指向`/tmp/test_app.db`作为测试数据库的位置。当然，您可以选择与系统范围的`tmp`目录不同的路径。

我们还将对`conftest.py`文件进行一些添加，以添加额外的装置，用于初始化测试数据库，并确保我们有一个 SQLAlchemy 数据库会话对象可用于可能需要它的任何测试函数：

```py
import pytest
import os
from application import create_app, db as database

DB_LOCATION = '/tmp/test_app.db'

@pytest.fixture(scope='session')
def app():
 app = create_app(config='test_settings')
 return app

@pytest.fixture(scope='session')
def db(app, request):
 """Session-wide test database."""
 if os.path.exists(DB_LOCATION):
 os.unlink(DB_LOCATION)

 database.app = app
 database.create_all()

 def teardown():
 database.drop_all()
 os.unlink(DB_LOCATION)
 request.addfinalizer(teardown)
 return database

@pytest.fixture(scope='function')
def session(db, request):

 session = db.create_scoped_session()
 db.session = session

 def teardown():
 session.remove()

 request.addfinalizer(teardown)
 return session

```

### 注意

会话装置可以通过显式事务进行增强，确保在拆卸时开始并提交事务。这个（简单）实现留给读者作为一个练习。

`scope`参数指示了创建的装置对象的生命周期。在前面的例子中，我们为会话装置指定了`function`，这意味着为每个作为参数调用的测试函数创建一个新的装置对象。如果我们使用`module`作为我们的作用域值，我们将为每个包含该装置的`module`创建一个新的装置：一个装置将用于模块中的所有测试。这不应与`session`作用域值混淆，后者表示为整个测试套件运行的整个持续时间创建一个装置对象。会话范围可以在某些情况下非常有用，例如，创建数据库连接是一个非常昂贵的操作。如果我们只需要创建一次数据库连接，那么测试套件的总运行时间可能会大大缩短。

有关`py.test`装置装饰器的`scope`参数以及使用内置的`request`对象添加拆卸终结器回调函数的更多信息，可以查看在线文档：[`pytest.org/latest/contents.html`](https://pytest.org/latest/contents.html)。

我们可以编写一个简单的测试，从我们的声明性用户模型中创建一个新用户，在`tests/test_user_model.py`中：

```py
from application.users import models

def test_create_user_instance(session):
 """Create and save a user instance."""

 email = 'test@example.com'
 username = 'test_user'
 password = 'foobarbaz'

 user = models.User(email, username, password)
 session.add(user)
 session.commit()

 # We clear out the database after every run of the test suite
 # but the order of tests may affect which ID is assigned.
 # Let's not depend on magic numbers if we can avoid it.
 assert user.id is not None

 assert user.followed.count() == 0
 assert user.newsfeed().count() == 0

```

在使用`py.test`运行测试套件后，我们应该看到我们新创建的测试文件出现在列出的输出中，并且我们的测试应该无错误地运行。我们将断言我们新创建的用户应该有一个 ID（由数据库分配），并且不应该关注任何其他用户。因此，我们创建的用户的新闻源也不应该有任何元素。

让我们为用户数据模型的非平凡部分添加一些更多的测试，这将确保我们的关注/关注关系按预期工作：

```py
def test_user_relationships(session):
 """User following relationships."""

 user_1 = models.User(
 email='test1@example.com', username='test1',
 password='foobarbaz')
 user_2 = models.User(
 email='test2@example.com', username='test2',
 password='bingbarboo')

 session.add(user_1)
 session.add(user_2)

 session.commit()

 assert user_1.followed.count() == 0
 assert user_2.followed.count() == 0

 user_1.follow(user_2)

 assert user_1.is_following(user_2) is True
 assert user_2.is_following(user_1) is False
 assert user_1.followed.count() == 1

 user_1.unfollow(user_2)

 assert user_1.is_following(user_2) is False
 assert user_1.followed.count() == 0

```

# 使用 Blinker 发布/订阅事件

在任何非平凡应用程序的生命周期中，一个困难是确保代码库中存在正确的模块化水平。

存在各种方法来创建接口、对象和服务，并实现设计模式，帮助我们管理不断增加的复杂性，这是不可避免地为现实世界的应用程序所创建的。一个经常被忽视的方法是 Web 应用程序中的进程内`发布-订阅`设计模式。

通常，`发布-订阅`，或者更通俗地称为 pub/sub，是一种消息模式，其中存在两类参与者：**发布者**和**订阅者**。发布者发送消息，订阅者订阅通过主题（命名通道）或消息内容本身产生的消息的子集。

在大型分布式系统中，pub/sub 通常由一个消息总线或代理来中介，它与所有各种发布者和订阅者通信，并确保发布的消息被路由到感兴趣的订阅者。

然而，为了我们的目的，我们可以使用一些更简单的东西：使用非常简单的`Blinker`包支持的进程内发布/订阅系统，如果安装了 Flask。

## 来自 Flask 和扩展的信号

当存在`Blinker`包时，Flask 允许您订阅发布的各种信号（主题）。此外，Flask 扩展可以实现自己的自定义信号。您可以订阅应用程序中的任意数量的信号，但是信号订阅者接收消息的顺序是未定义的。

Flask 发布的一些更有趣的信号在以下列表中描述：

+   `request_started`: 这是在请求上下文创建后立即发送的，但在任何请求处理发生之前

+   `request_finished`: 这是在响应构造后发送的，但在发送回客户端之前立即发送

Flask-SQLAlchemy 扩展本身发布了以下两个信号：

+   `models_committed`: 这是在任何修改的模型实例提交到数据库后发送的

+   `before_models_committed`: 这是在模型实例提交到数据库之前发送的

Flask-Login 发布了半打信号，其中许多可以用于模块化认证问题。以下列出了一些有用的信号：

+   `user_logged_in`: 当用户登录时发送

+   `user_logged_out`: 当用户注销时发送

+   `user_unauthorized`: 当未经认证的用户尝试访问需要认证的资源时发送

## 创建自定义信号

除了订阅由 Flask 和各种 Flask 扩展发布的信号主题之外，还可以（有时非常有用！）创建自己的自定义信号，然后在自己的应用程序中使用。虽然这可能看起来像是一个绕圈子的方法，简单的函数或方法调用就足够了，但是将应用程序的各个部分中的正交关注点分离出来的能力是一个吸引人的建议。

例如，假设你有一个用户模型，其中有一个`update_password`方法，允许更改给定用户实例的密码为新的值。当密码被更改时，我们希望向用户发送一封邮件，通知他们发生了这个动作。

现在，这个功能的简单实现就是在`update_password`方法中发送邮件，这本身并不是一个坏主意。然而，想象一下，我们还有另外十几个实例需要发送邮件给用户：当他们被新用户关注时，当他们被用户取消关注时，当他们达到一定的关注者数量时，等等。

然后问题就显而易见了：我们在应用程序的各个部分混合了发送邮件给用户的逻辑和功能，这使得越来越难以理解、调试和重构。

虽然有几种方法可以管理这种复杂性，但当实现发布/订阅模式时，可以明显地看到可能的关注点的明确分离。在我们的 Flask 应用程序中使用自定义信号，我们可以创建一个添加关注者的信号，在动作发生后发布一个事件，任何数量的订阅者都可以监听该特定事件。此外，我们可以组织我们的应用程序，使得类似事件的信号订阅者（例如，发送电子邮件通知）在代码库中的同一位置。

让我们创建一个信号，每当一个用户关注另一个用户时就发布一个事件。首先，我们需要创建我们的`Namespace`信号容器对象，以便我们可以声明我们的信号主题。让我们在`application/__init__.py`模块中做这件事：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.bcrypt import Bcrypt
from blinker import Namespace

# Initialize the db extension, but without configuring
# it with an application instance.
db = SQLAlchemy()
flask_bcrypt = Bcrypt()

socializer_signals = Namespace()
user_followed = socializer_signals.signal('user-followed')

# …

```

一旦这个功能就位，我们在`User.follow()`方法中发出`user-followed`事件就很简单了，如下所示：

```py
def follow(self, user):
 """
 Follow the given user.

 Return `False` if the user was already following the user.
 """

 if self.is_following(user):
 return False
 self.followed.append(user)

 # Publish the signal event using the current model (self) as sender.
 user_followed.send(self)

 return self

```

### 注意

记得在`application/users/models.py`模块顶部添加`from the application import user_followed`导入行。

一旦发布了事件，订阅者可能会连接。让我们在`application/signal_handlers.py`中实现信号处理程序：

```py
__all__ = ['user_followed_email']

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
def user_followed_email(user, **kwargs):
 logger.debug(
 "Send an email to {user}".format(user=user.username))

from application import user_followed

def connect_handlers():
 user_followed.connect(user_followed_email)

```

最后，我们需要确保我们的信号处理程序通过将函数导入到`application/__init__.py`模块来注册：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.bcrypt import Bcrypt
from blinker import Namespace

# Initialize the db extension, but without configuring
# it with an application instance.
db = SQLAlchemy()
flask_bcrypt = Bcrypt()

socializer_signals = Namespace()
user_followed = socializer_signals.signal('user-followed')

from signal_handlers import connect_handlers
connect_handlers()

# …
# …

```

添加此功能后，每当用户关注其他用户时，我们都将在配置的日志输出中打印一条调试消息。实际向用户发送电子邮件的功能留给读者作为练习；一个好的起点是使用`Flask-Mail`扩展。

# 异常的优雅处理

无论我们多么努力，有时我们使用和编写的代码会引发异常。

通常，这些异常是在特殊情况下抛出的，但这并不减少我们应该了解应用程序的哪些部分可能引发异常，以及我们是否希望在调用点处理异常，还是简单地让它冒泡到调用堆栈的另一个帧。

对于我们当前的应用程序，有几种异常类型我们希望以一种优雅的方式处理，而不是让整个 Python 进程崩溃，导致一切戛然而止，变得丑陋不堪。

在上一章中，我们简要提到了大多数基于 Flask 和 SQLAlchemy 的应用程序（或几乎任何其他数据库抽象）中需要存在的必要异常处理，但当这些异常确实出现时，处理它们的重要性怎么强调都不为过。考虑到这一点，让我们创建一些视图、表单和模板，让我们作为新用户注册到我们的应用程序，并查看一些异常出现时处理它们的示例。

首先，让我们在`application/users/views.py`中创建基本的用户视图处理程序：

```py
from flask import Blueprint, render_template, url_for, redirect, flash, g
from flask.ext.login import login_user, logout_user

from flask.ext.wtf import Form
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, Length

from models import User
from application import db, flask_bcrypt

users = Blueprint('users', __name__, template_folder='templates')

class Login	Form(Form):
 """
Represents the basic Login form elements & validators.
 """

username = StringField('username',
 validators=[DataRequired()])
password = PasswordField('password',
 validators=[DataRequired(),Length(min=6)])

class CreateUserForm(Form):
 """
 Encapsulate the necessary information required for creating a new user.
 """

 username = StringField('username', validators=[DataRequired(), Length(min=3, max=40)])
 email = StringField('email', validators=[DataRequired(), Length(max=255)])
 password = PasswordField('password', validators=[DataRequired(),
 Length(min=8)])

 @users.route('/signup', methods=['GET', 'POST'])
 def signup():
 """
Basic user creation functionality.

 """

form = CreateUserForm()

if form.validate_on_submit():

 user = User( username=form.username.data,
 email=form.email.data,
 password=form.password.data)

 # add the user to the database
 db.session.add(user)
 db.session.commit()
 # Once we have persisted the user to the database successfully,
 # authenticate that user for the current session
login_user(user, remember=True)
return redirect(url_for('users.index'))

return render_template('users/signup.html', form=form)

@users.route('/', methods=['GET'])
def index():
return "User index page!", 200

@users.route('/login', methods=['GET', 'POST'])
def login():
 """
Basic user login functionality.

 """

if hasattr(g, 'user') and g.user.is_authenticated():
return redirect(url_for('users.index'))

form = LoginForm()

if form.validate_on_submit():

 # We use one() here instead of first()
 user = User.query.filter_by(username=form.username.data).one()
 if not user or not flask_bcrypt.check_password_hash(user.password, form.password.data):

 flash("No such user exists.")
 return render_template('users/login.html', form=form)

 login_user(user, remember=True)
 return redirect(url_for('users.index'))
 return render_template('users/login.html', form=form)

@users.route('/logout', methods=['GET'])
def logout():
logout_user()
return redirect(url_for('users.login'))

```

你会发现，登录和注销功能与我们在上一章使用 Flask-Login 扩展创建的功能非常相似。因此，我们将简单地包含这些功能和定义的路由（以及相关的 Jinja 模板），而不加评论，并专注于新的注册路由，该路由封装了创建新用户所需的逻辑。此视图利用了新的`application/users/templates/users/signup.html`视图，该视图仅包含允许用户输入其期望的用户名、电子邮件地址和密码的相关表单控件：

```py
{% extends "layout.html" %}

{% block content %}

<form action="{{ url_for('users.signup')}}" method="post">
  {{ form.hidden_tag() }}
  {{ form.id }}
  <div>{{ form.username.label }}: {{ form.username }}</div>
  {% if form.username.errors %}
  <ul class="errors">{% for error in form.username.errors %}<li>{{ error }}</li>{% endfor %}</ul>
  {% endif %}

  <div>{{ form.email.label }}: {{ form.email }}</div>
  {% if form.email.errors %}
  <ul class="errors">{% for error in form.email.errors %}<li>{{ error }}</li>{% endfor %}</ul>
  {% endif %}

  <div>{{ form.password.label }}: {{ form.password }}</div>
  {% if form.password.errors %}
  <ul class="errors">{% for error in form.password.errors %}<li>{{ error }}</li>{% endfor %}</ul>
  {% endif %}

  <div><input type="submit" value="Sign up!"></div>
</form>

{% endblock %}
```

一旦我们有了前面的模板，我们将更新我们的应用程序工厂，将用户视图绑定到应用程序对象。我们还将初始化 Flask-Login 扩展，就像我们在上一章所做的那样：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.bcrypt import Bcrypt
from blinker import Namespace
from flask.ext.login import LoginManager

# Initialize the db extension, but without configuring
# it with an application instance.
db = SQLAlchemy()
flask_bcrypt = Bcrypt()
login_manager = LoginManager()

socializer_signals = Namespace()
user_followed = socializer_signals.signal('user-followed')

from signal_handlers import *

def create_app(config=None):
app = Flask(__name__)

if config is not None:
 app.config.from_object(config)

 # Initialize any extensions and bind blueprints to the
 # application instance here.
 db.init_app(app)
 flask_bcrypt.init_app(app)
 login_manager.init_app(app)

 from application.users.views import users
 app.register_blueprint(users, url_prefix='/users')

 from application.users import models as user_models
 @login_manager.user_loader
 de fload_user(user_id):
 return user_models.User.query.get(int(user_id))

 return app

```

别忘了在我们的`application/settings.py`模块中添加一个`SECRET_KEY`配置值：

```py
SQLALCHEMY_DATABASE_URI = 'sqlite:///socializer.db'
SECRET_KEY = 'BpRvzXZ800[-t:=z1eZtx9t/,P*'

```

现在，我们应该能够运行应用程序并访问`http://localhost:5000/users/signup`，在那里我们将看到一系列用于创建新用户账户的表单输入。在成功创建新用户后，我们将自动使用 Flask-Login 扩展的`login_user()`方法进行身份验证。

然而，我们尚未考虑到的是，由于与我们的 SQLAlchemy 模型和数据库期望不匹配，用户创建失败的情况。这可能由于多种原因发生：

+   现有用户已经声明了提交的电子邮件或用户名值，这两者在我们用户模型中都被标记为唯一

+   某个字段需要数据库指定的额外验证标准，而这些标准未被满足

+   数据库不可用（例如，由于网络分区）

为了确保这些事件以尽可能优雅的方式处理，我们必须封装可能引发相关异常的代码部分，这些异常表明了这些条件之一。因此，在我们的`application/users/views.py`模块中的注册路由中，我们将修改将用户持久化到数据库的代码部分：

```py
# place with other imports…
from sqlalchemy import exc

# …

try:
 db.session.add(user)
 db.session.commit()
 except exc.IntegrityError as e:
 # A unique column constraint was violated
 current_app.exception("User unique constraint violated.")
 return render_template('users/signup.html', form=form)
 except exc.SQLAlchemyError:
 current_app.exception("Could not save new user!")
 flash("Something went wrong while creating this user!")
 return render_template('users/signup.html', form=form)

```

此外，我们将在登录路由中使用 try/except 块包装`User.query.filter_by(username=form.username.data).one()`，以确保我们处理登录表单中提交的用户名在数据库中根本不存在的情况：

```py
try:
    # We use one() here instead of first()
    user = User.query.filter_by(
           username=form.username.data).one()s
except NoResultFound:
    flash("User {username} does not exist.".format(
        username=form.username.data))
    return render_template('users/login.html', form=form)

# …
```

# 功能测试

既然我们已经创建了一些处理用户注册和登录的路由和模板，让我们利用本章早些时候获得的`py.test`知识来编写一些事后的集成测试，以确保我们的视图按预期行为。首先，让我们在`application/tests/test_user_views.py`中创建一个新的测试模块，并编写我们的第一个使用客户端固定装置的测试，以便通过内置的 Werkzeug 测试客户端模拟对应用程序的请求。这将确保已构建适当的请求上下文，以便上下文绑定对象（例如，`url_for`，`g`）可用，如下所示：

```py
def test_get_user_signup_page(client):
 """Ensure signup page is available."""
 response = client.get('/users/signup')
 assert response.status_code == 200
 assert 'Sign up!' in response.data

```

前面的测试首先向`/users/signup`路由发出请求，然后断言该路由的 HTTP 响应代码为`200`（任何成功返回`render_template()`函数的默认值）。然后它断言**注册！**按钮文本出现在返回的 HTML 中，这是一个相对安全的保证，即所讨论的页面在没有任何重大错误的情况下被渲染。

接下来，让我们添加一个成功用户注册的测试，如下所示：

```py
from flask import session, get_flashed_messages
from application.users.models import User
from application import flask_bcrypt

def test_signup_new_user(client):
 """Successfully sign up a new user."""
 data = {'username': 'test_username', 'email': 'test@example.com',
 'password': 'my test password'}

 response = client.post('/users/signup', data=data)

 # On successful creation we redirect.
 assert response.status_code == 302

 # Assert that a session was created due to successful login
 assert '_id' in session

 # Ensure that we have no stored flash messages indicating an error
 # occurred.
 assert get_flashed_messages() == []

 user = User.query.filter_by(username=data['username']).one()

 assert user.email == data['email']
 assert user.password
 assert flask_bcrypt.check_password_hash(
 user.password, data['password'])

```

如果我们立即运行测试套件，它会失败。这是由于 Flask-WTF 引入的一个微妙效果，它期望为任何提交的表单数据提供 CSRF 令牌。以下是我们修复此问题的两种方法：

+   我们可以在模拟的 POST 数据字典中手动生成 CSRF 令牌；`WTForms`库提供了实现此功能的方法

+   我们可以在`test_settings.py`模块中将`WTF_CSRF_ENABLED`配置布尔值设置为`False`，这样测试套件中发生的所有表单验证将不需要 CSRF 令牌即可被视为有效。

第一种方法的优势在于，请求/响应周期中发送的数据将紧密反映生产场景中发生的情况，缺点是我们必须为想要测试的每个表单生成（或程序化抽象）所需的 CSRF 令牌。第二种方法允许我们在测试套件中完全停止关心 CSRF 令牌，这也是一个缺点。本章中，我们将采用第二种方法所述的方式。

在前面的测试中，我们将首先创建一个包含我们希望 POST 到注册端点的模拟表单数据的字典，然后将此数据传递给`client.post('/users/signup')`方法。在新用户成功注册后，我们应该期望被重定向到不同的页面（我们也可以检查响应中*Location*头的存在和值），此外，Flask-Login 将创建一个会话 ID 来处理我们的用户会话。此外，对于我们当前的应用程序，成功的注册尝试意味着我们不应该有任何存储以供显示的闪现消息，并且应该有一个新用户记录，其中包含在 POST 中提供的数据，并且该数据应该可用并填充。

虽然大多数开发者非常热衷于测试请求的成功路径，但测试最常见的失败路径同样重要，甚至更为重要。为此，让我们为最典型的失败场景添加以下几个测试，首先是使用无效用户名的情况：

```py
import pytest
import sqlalchemy

def test_signup_invalid_user(client):
 """Try to sign up with invalid data."""

 data = {'username': 'x', 'email': 'short@example.com',
 'password': 'a great password'}

 response = client.post('/users/signup', data=data)

 # With a form error, we still return a 200 to the client since
 # browsers are not always the best at handling proper 4xx response codes.
 assert response.status_code == 200
 assert 'must be between 3 and 40 characters long.' in response.data

```

### 注意

记住，我们在`application.users.views.CreateUserForm`类中定义了用户注册的表单验证规则；用户名必须介于 3 到 40 个字符之间。

```py
def test_signup_invalid_user_missing_fields(client):
 """Try to sign up with missing email."""

 data = {'username': 'no_email', 'password': 'a great password'}
 response = client.post('/users/signup', data=data)

 assert response.status_code == 200
 assert 'This field is required' in response.data

 with pytest.raises(sqlalchemy.orm.exc.NoResultFound):
 User.query.filter_by(username=data['username']).one()

 data = {'username': 'no_password', 'email': 'test@example.com'}
 response = client.post('/users/signup', data=data)

 assert response.status_code == 200
 assert 'This field is required' in response.data

 with pytest.raises(sqlalchemy.orm.exc.NoResultFound):
 User.query.filter_by(username=data['username']).one()

```

### 注意

在前面的测试中，我们使用了`py.test`（及其他测试库）中一个经常被忽视的便利函数，即`raises(exc)`上下文管理器。这允许我们将一个函数调用包裹起来，在其中我们期望抛出异常，如果预期的异常类型（或派生类型）未被抛出，它本身将导致测试套件中的失败。

# 你的新闻动态

尽管我们已经构建了大部分支持架构，为我们的 Socializer 应用程序提供功能，但我们仍缺少拼图中更基本的一块：能够按时间顺序查看你关注的人的帖子。

为了使显示帖子所有者的信息更简单一些，让我们在我们的`Post`模型中添加一个关系定义：

```py
class Post(db.Model):
 # …
 user = db.relationship('User',
 backref=db.backref('posts', lazy='dynamic'))

```

这将允许我们使用`post.user`访问与给定帖子关联的任何用户信息，这在显示单个帖子或帖子列表的任何视图中都将非常有用。

让我们在`application/users/views.py`中为此添加一条路由：

```py
@users.route('/feed', methods=['GET'])
@login_required
def feed():
 """
 List all posts for the authenticated user; most recent first.
 """
 posts = current_user.newsfeed()
 return render_template('users/feed.html', posts=posts)

```

请注意，前面的代码片段使用了`current_user`代理（您应该将其导入到模块中），该代理由 Flask-Login 扩展提供。由于 Flask-Login 扩展在代理中存储了经过身份验证的用户对象，因此我们可以像在普通`user`对象上一样调用其方法和属性。

由于之前的 feed 端点已经运行，我们需要在`application/users/templates/users/feed.html`中提供支持模板，以便我们实际上可以渲染响应：

```py
{% extends "layout.html" %}

{% block content %}
<div class="new-post">
  <p><a href="{{url_for('posts.add')}}">New Post</a></p>
</div>

{% for post in posts %}
<div class="post">
  <span class="author">{{post.user.username}}</span>, published on <span class="date">{{post.created_on}}</span>
  <pre><code>{{post.content}}</code></pre>
</div>
{% endfor %}

{% endblock %}
```

我们需要的最后一部分是添加新帖子的视图处理程序。由于我们尚未创建`application/posts/views.py`模块，让我们来创建它。我们将需要一个`Flask-WTForm`类来处理/验证新帖子，以及一个路由处理程序来发送和处理所需的字段，所有这些都连接到一个新的蓝图上：

```py
from flask import Blueprint, render_template, url_for, redirect, flash, current_app

from flask.ext.login import login_required, current_user
from flask.ext.wtf import Form
from wtforms import StringField
from wtforms.widgets import TextArea
from wtforms.validators import DataRequired
from sqlalchemy import exc

from models import Post
from application import db

posts = Blueprint('posts', __name__, template_folder='templates')

class CreatePostForm(Form):
 """Form for creating new posts."""

 content = StringField('content', widget=TextArea(),
 validators=[DataRequired()])

@posts.route('/add', methods=['GET', 'POST'])
@login_required
def add():
 """Add a new post."""

 form = CreatePostForm()
 if form.validate_on_submit():
 user_id = current_user.id

 post = Post(user_id=user_id, content=form.content.data)
 db.session.add(post)

 try:
 db.session.commit()
 except exc.SQLAlchemyError:
 current_app.exception("Could not save new post!")
 flash("Something went wrong while creating your post!")
 else:
 return render_template('posts/add.html', form=form)

 return redirect(url_for('users.feed'))

```

相应的`application/posts/templates/posts/add.html`文件正如预期的那样相对简单，并且让人想起上一章中使用的视图模板。这里是：

```py
{% extends "layout.html" %}

{% block content %}
<form action="{{ url_for('posts.add')}}" method="post">

  {{ form.hidden_tag() }}
  {{ form.id }}

  <div class="row">
    <div>{{ form.content.label }}: {{ form.content }}</div>
    {% if form.content.errors %}
    <ul class="errors">{% for error in form.content.errors %}<li>{{ error }}</li>{% endfor %}</ul>
    {% endif %}
  </div>

  <div><input type="submit" value="Post"></div>
</form>

{% endblock %}
```

最后，我们需要通过在我们的应用程序工厂中将其绑定到我们的应用程序对象，使应用程序意识到这个新创建的帖子蓝图，位于`application/__init__.py`中：

```py
def create_app(config=None):
    app = Flask(__name__)

    # …
    from application.users.views import users
    app.register_blueprint(users, url_prefix='/users')

 from application.posts.views import posts
 app.register_blueprint(posts, url_prefix='/posts')

        # …
```

一旦上述代码就位，我们可以通过在`/users/signup`端点的 Web 界面上创建用户帐户，然后在`/posts/add`上为用户创建帖子来为这些用户生成一些测试用户和帖子。否则，我们可以创建一个小的 CLI 脚本来为我们执行此操作，我们将在下一章中学习如何实现。我们还可以编写一些测试用例来确保新闻源按预期工作。实际上，我们可以做这三件事！

# 摘要

我们通过首先介绍应用程序工厂的概念，并描述了这种方法的一些好处和权衡来开始本章。接下来，我们使用我们新创建的应用程序工厂来使用`py.test`设置我们的第一个测试套件，这需要对我们的应用程序对象的创建方式进行一些修改，以确保我们获得一个适合的实例，配置为测试场景。 

然后，我们迅速着手实现了典型 Web 应用程序背后的基本数据模型，其中包含了*社交*功能，包括关注其他用户以及被其他用户关注的能力。我们简要涉及了所谓新闻源应用程序的几种主要实现模式，并为我们自己的数据模型使用了最简单的版本。

这随后导致我们讨论和探索了发布/订阅设计模式的概念，Flask 和各种 Flask 扩展集成了`Blinker`包中的一个进程内实现。利用这些新知识，我们创建了自己的发布者和订阅者，使我们能够解决许多现代 Web 应用程序中存在的一些常见横切关注点。

对于我们的下一个项目，我们将从创建过去几章中使用的基于 HTML 的表单和视图切换到另一个非常重要的现代 Web 应用程序部分：提供一个有用的 JSON API 来进行交互。
