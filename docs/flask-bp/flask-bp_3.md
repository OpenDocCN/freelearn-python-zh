# 第三章：Snap - 代码片段共享应用程序

在本章中，我们将构建我们的第一个完全功能的、基于数据库的应用程序。这个应用程序，代号 Snap，将允许用户使用用户名和密码创建帐户。用户将被允许登录、注销、添加和列出所谓的半私密*snaps*文本，这些文本可以与其他人分享。

本章中，您应该熟悉以下至少一种关系数据库系统：PostgreSQL、MySQL 或 SQLite。此外，对 SQLAlchemy Python 库的一些了解将是一个优势，它充当这些（以及其他几个）数据库的抽象层和对象关系映射器。如果您对 SQLAlchemy 的使用不熟悉，不用担心。我们将对该库进行简要介绍，以帮助新开发人员迅速上手，并为经验丰富的开发人员提供复习。

从现在开始，在本书中，SQLite 数据库将是我们选择的关系数据库。我们列出的其他数据库系统都是基于客户端/服务器的，具有多种配置选项，可能需要根据安装的系统进行调整，而 SQLite 的默认操作模式是独立、无服务器和零配置。

我们建议您使用 SQLite 来处理这个项目和接下来的章节中的项目，但 SQLAlchemy 支持的任何主要关系数据库都可以。

# 入门

为了确保我们正确开始，让我们创建一个项目存在的文件夹和一个虚拟环境来封装我们将需要的任何依赖项：

```py
$ mkdir -p ~/src/snap && cd ~/src/snap
$ mkvirtualenv snap -i flask

```

这将在给定路径创建一个名为`snap`的文件夹，并带我们到这个新创建的文件夹。然后它将在这个环境中创建 snap 虚拟环境并安装 Flask。

### 注意

请记住，`mkvirtualenv`工具将创建虚拟环境，这将是从`pip`安装软件包的默认位置集，但`mkvirtualenv`命令不会为您创建项目文件夹。这就是为什么我们将首先运行一个命令来创建项目文件夹，然后再创建虚拟环境。虚拟环境通过激活环境后执行的`$PATH`操作完全独立于文件系统中项目文件的位置。

然后，我们将使用基本的基于蓝图的项目布局创建一个空的用户蓝图。所有文件的内容几乎与我们在上一章末尾描述的内容相同，布局应该如下所示：

```py
application
├── __init__.py
├── run.py
└── users
    ├── __init__.py
    ├── models.py
    └── views.py

```

## Flask-SQLAlchemy

一旦上述文件和文件夹被创建，我们需要安装下一个重要的一组依赖项：SQLAlchemy 和使与该库交互更类似于 Flask 的 Flask 扩展，Flask-SQLAlchemy：

```py
$ pip install flask-sqlalchemy

```

这将安装 Flask 扩展到 SQLAlchemy 以及后者的基本分发和其他几个必要的依赖项，以防它们尚未存在。

现在，如果我们使用的是除 SQLite 之外的关系数据库系统，这就是我们将在其中创建数据库实体的时刻，比如在 PostgreSQL 中，以及创建适当的用户和权限，以便我们的应用程序可以创建表并修改这些表的内容。然而，SQLite 不需要任何这些。相反，它假设任何可以访问数据库文件系统位置的用户也应该有权限修改该数据库的内容。

在本章的后面，我们将看到如何通过 SQLAlchemy 自动创建 SQLite 数据库文件。然而，为了完整起见，这里是如何在文件系统的当前文件夹中创建一个空数据库：

```py
$ sqlite3 snap.db  # hit control-D to escape out of the interactive SQL console if necessary.

```

### 注意

如前所述，我们将使用 SQLite 作为示例应用程序的数据库，并且给出的指示将假定正在使用 SQLite；二进制文件的确切名称可能在您的系统上有所不同。如果使用的不是 SQLite，您可以替换等效的命令来创建和管理您选择的数据库。

现在，我们可以开始对 Flask-SQLAlchemy 扩展进行基本配置。

### 配置 Flask-SQLAlchemy

首先，我们必须在`application/__init__.py`文件中将 Flask-SQLAlchemy 扩展注册到`application`对象中：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../snap.db'
db = SQLAlchemy(app)

```

`app.config['SQLALCHEMY_DATABASE_URI']`的值是我们之前创建的`snap.db SQLite`数据库的转义相对路径。一旦这个简单的配置就位，我们就能够通过`db.create_all()`方法自动创建 SQLite 数据库，这可以在交互式 Python shell 中调用：

```py
$  python
>>>from application import db
>>>db.create_all()

```

这是一个幂等操作，这意味着即使数据库已经存在，也不会发生任何变化。然而，如果本地数据库文件不存在，它将被创建。这也适用于添加新的数据模型：运行`db.create_all()`将它们的定义添加到数据库，确保相关表已被创建并且可访问。然而，它并不考虑已经存在于数据库中的现有模型/表定义的修改。为此，您需要使用相关工具（例如 sqlite CLI，或者迁移工具如 Alembic，我们将在后面的章节中讨论）来修改相应的表定义，以匹配您模型中已更新的定义。

### SQLAlchemy 基础知识

SQLAlchemy 首先是一个与 Python 中的关系数据库进行交互的工具包。

虽然它提供了令人难以置信的多种功能，包括各种数据库引擎的 SQL 连接处理和连接池、处理自定义数据类型的能力以及全面的 SQL 表达式 API，但大多数开发人员熟悉的功能是对象关系映射器。这个映射器允许开发人员将 Python 对象定义与他们选择的数据库中的 SQL 表连接起来，从而使他们能够灵活地控制自己应用程序中的领域模型，并且只需要最小的耦合到数据库产品和引擎特定的 SQL 特性。

虽然在本章讨论对象关系映射器的有用性（或缺乏有用性）超出了范围，但对于那些不熟悉 SQLAlchemy 的人，我们将提供使用这个工具带来的好处清单，如下所示：

+   您的领域模型是为了与最受尊敬、经过测试和部署的 Python 包之一——SQLAlchemy 进行交互而编写的。

+   由于有关使用 SQLAlchemy 的广泛文档、教程、书籍和文章，将新开发人员引入项目变得更加容易。

+   查询的验证是在模块导入时使用 SQLAlchemy 表达式语言完成的，而不是针对数据库执行每个查询字符串以确定是否存在语法错误。表达式语言是用 Python 编写的，因此可以使用您通常的一套工具和 IDE 进行验证。

+   由于实现了设计模式，如工作单元、身份映射和各种延迟加载特性，开发人员通常可以避免执行比必要更多的数据库/网络往返。考虑到典型 Web 应用程序中请求/响应周期的大部分很容易归因于各种类型的网络延迟，最小化典型响应中的数据库查询数量在多个方面都是性能上的胜利。

+   虽然许多成功的高性能应用程序可以完全建立在 ORM 上，但 SQLAlchemy 并不强制要求这样做。如果出于某种原因，更倾向于编写原始的 SQL 查询字符串或直接使用 SQLAlchemy 表达语言，那么您可以这样做，并仍然从 SQLAlchemy 本身的连接池和 Python DBAPI 抽象功能中受益。

既然我们已经给出了几个理由，说明为什么应该使用这个数据库查询和领域数据抽象层，让我们看看如何定义一个基本的数据模型。

#### 声明式映射和 Flask-SQLAlchemy

SQLAlchemy 实现了一种称为**数据映射器**的设计模式。基本上，这个数据映射器的工作是在代码中桥接数据模型的定义和操作（在我们的情况下，Python 类定义）以及数据库中这个数据模型的表示。映射器应该知道代码相关的操作（例如，对象构造、属性修改等）如何与我们选择的数据库中的 SQL 特定语句相关联，确保在我们映射的 Python 对象上执行的操作与它们关联的数据库表正确同步。

我们可以通过两种方式将 SQLAlchemy 集成到我们的应用程序中：通过使用提供表、Python 对象和数据映射一致集成的声明式映射，或者通过手动指定这些关系。此外，还可以使用所谓的 SQLAlchemy“核心”，它摒弃了基于数据域的方法，而是基于 SQL 表达语言构造，这些构造包含在 SQLAlchemy 中。

在本章（以及将来的章节）中，我们将使用声明式方法。

要使用声明式映射功能，我们需要确保我们定义的任何模型类都将继承自 Flask-SQLAlchemy 提供给我们的声明基类`Model`类（一旦我们初始化了扩展）：

```py
from application import db

class User(db.Model):
 # model attributes
 pass

```

这个`Model`类本质上是`sqlalchemy.ext.declarative.declarative_base`类的一个实例（带有一些额外的默认值和有用的功能），它为对象提供了一个元类，该元类将处理适当的映射构造。

一旦我们在适当的位置定义了我们的模型类定义，我们将通过使用`Column`对象实例来定义通过类级属性映射的相关 SQL 表的详细信息。Column 调用的第一个参数是我们想要对属性施加的类型约束（对应于数据库支持的特定模式数据类型），以及类型支持的任何可选参数，例如字段的大小。还可以提供其他参数来指示对生成的表字段定义的约束：

```py
class User(db.Model):

 id = db.Column(db.Integer, primary_key=True)
 email = db.Column(db.String(255), unique=True)
 username = db.Column(db.String(40), unique=True)

```

### 注意

如前所述，仅仅定义属性并不会自动转换为数据库中的新表和列。为此，我们需要调用`db.create_all()`来初始化表和列的定义。

我们可以轻松地创建此模型的实例，并为我们在类定义中声明的属性分配一些值：

```py
$ (snap) python
>>>from application.users.models import User
>>>new_user = User(email="me@example.com", username="me")
>>>new_user.email
'me@example.com'
>>>new_user.username
'me'

```

### 注意

您可能已经注意到，我们的用户模型没有定义`__init__`方法，但当实例化上面的示例时，我们能够将`email`和`username`参数传递给对象构造函数。这是 SQLAlchemy 声明基类的一个特性，它会自动将命名参数在对象构造时分配给它们的对象属性对应项。因此，通常不需要为数据模型定义一个具体的构造方法。

模型对象的实例化并不意味着它已经持久化到数据库中。为此，我们需要通知 SQLAlchemy 会话，我们希望添加一个新对象以进行跟踪，并将其提交到数据库中：

```py
>>>from application import db
>>>db.session.add(new_user)
>>>db.session.commit()

```

一旦对象被提交，`id`属性将获得底层数据库引擎分配给它的主键值：

```py
>>>print(new_user.id)
1

```

如果我们想修改属性的值，例如，更改特定用户的电子邮件地址，我们只需要分配新值，然后提交更改：

```py
>>>new_user.email = 'new@example.com'
>>>db.session.add(new_user)
>>>db.session.commit()
>>>print(new_user.email)
u'new@example.com'

```

此时，您可能已经注意到在任何以前的操作中都没有编写过一行 SQL，并且可能有点担心您创建的对象中嵌入的信息没有持久保存到数据库中。对数据库的粗略检查应该让您放心：

```py
$ sqlite3 snap.db
SQLite version 3.8.5 2014-08-15 22:37:57
Enter ".help" for usage hints.
sqlite> .tables
user
sqlite> .schema user
CREATE TABLE user (
 id INTEGER NOT NULL,
 email VARCHAR(255),
 username VARCHAR(40),
 PRIMARY KEY (id),
 UNIQUE (email),
 UNIQUE (username)
);
sqlite> select * from user;
1|new@example.com|me

```

### 注意

请记住，SQLite 二进制文件的确切名称可能会因您选择的操作系统而异。此外，如果您选择了除 SQLite 之外的数据库引擎来跟随这些示例，相关的命令和结果可能会大相径庭。

就是这样：SQLAlchemy 成功地在幕后管理了相关的 SQL INSERT 和 UPDATE 语句，让我们可以使用本机 Python 对象，并在准备将数据持久保存到数据库时通知会话。

当然，我们不仅限于定义类属性。在许多情况下，声明模型上的实例方法可能会证明很有用，以便我们可以执行更复杂的数据操作。例如，想象一下，我们需要获取给定用户的主键 ID，并确定它是偶数还是奇数。方法声明将如你所期望的那样：

```py
class User(db.Model):

 id = db.Column(db.Integer, primary_key=True)
 email = db.Column(db.String(255), unique=True)
 username = db.Column(db.String(40), unique=True)

def is_odd_id(self):
 return (self.id % 2 != 0)

```

实例方法调用可以像往常一样执行，但在将对象提交到会话之前，主键值将为 none：

```py
$ (snap)  python
Python 2.7.10 (default, Jul 13 2015, 23:27:37)
[GCC 4.2.1 Compatible Apple LLVM 6.1.0 (clang-602.0.53)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>fromapplication.users.models import User
>>>test = User(email='method@example.com', username='method_test')
>>>from application import db
>>>db.session.add(test)
>>>db.session.commit()
>>> test.id
2
>>>test.is_odd_id()
False

```

当然，在大多数 Web 应用程序的上下文中，前面的实现是微不足道且有些毫无意义的。然而，定义模型实例方法以编码业务逻辑的能力非常方便，我们将在本章后面看到 Flask-Login 扩展中的一些内容。

### 快照数据模型

现在我们已经探索了 SQLAlchemy 声明基础和 Flask-SQLAlchemy 扩展的基础知识，使用了一个简化的模型，我们的下一步是完善一个用户数据模型，这是几乎任何 Web 应用程序的基石。我们将在用户蓝图中创建这个模型，在一个新的`users/models.py`模块中利用我们对 SQLAlchemy 模型的知识，为用户`password`和`created_on`字段添加字段，以存储记录创建的时间。此外，我们将定义一些实例方法：

```py
import datetime
from application import db

class User(db.Model):

 # The primary key for each user record.
 id = db.Column(db.Integer, primary_key=True)

 # The unique email for each user record.
 email = db.Column(db.String(255), unique=True)

 # The unique username for each record.
 username = db.Column(db.String(40), unique=True)

 # The hashed password for the user
 password = db.Column(db.String(60))

#  The date/time that the user account was created on.
 created_on = db.Column(db.DateTime, 
 default=datetime.datetime.utcnow)

 def __repr__(self):
 return '<User {!r}>'.format(self.username)

 def is_authenticated(self):
 """All our registered users are authenticated."""
 return True

 def is_active(self):
 """All our users are active."""
 return True

 def is_anonymous(self):
 """We don)::f):lf):"""users are authenticated."""
 return False

 def get_id(self):
 """Get the user ID as a Unicode string."""
 return unicode(self.id)

```

`is_authenticated`、`is_active`、`is_anonymous`和`get_id`方法目前可能看起来是任意的，但它们是下一步所需的，即安装和设置 Flask-Login 扩展，以帮助我们管理用户身份验证系统。

## Flask-Login 和 Flask-Bcrypt 用于身份验证

我们已经多次使用其他库进行了安装扩展，我们将在当前项目的虚拟环境中安装这些扩展：

```py
$ (snap) pip install flask-login flask-bcrypt

```

第一个是一个特定于 Flask 的库，用于规范几乎每个 Web 应用程序都需要的标准用户登录过程，后者将允许我们确保我们在数据库中存储的用户密码使用行业标准算法进行哈希处理。

安装后，我们需要以通常的方式实例化和配置扩展。为此，我们将添加到`application/__init__.py`模块中：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.login import LoginManager
from flask.ext.bcrypt import Bcrypt

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../snap.db'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
flask_bcrypt = Bcrypt(app)

from application.users import models as user_models
from application.users.views import users

```

为了正确运行，Flask-Login 扩展还必须知道如何仅通过用户的 ID 从数据库中加载用户。我们必须装饰一个函数来完成这个任务，并为简单起见，我们将它插入到`application/__init__.py`模块的最后：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.login LoginManager
from flask.ext.bcrypt import Bcrypt

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../snap.db'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
flask_bcrypt = Bcrypt(app)

from application.users import models as user_models
from application.users.views import users

@login_manager.user_loader
def load_user(user_id):
 return application.user_models.query.get(int(user_id))

```

现在我们已经设置了模型和所需的方法/函数，以便 Flask-Login 可以正确运行，我们的下一步将是允许用户像几乎任何 Web 应用程序一样登录使用表单。

## Flask-WTF - 表单验证和呈现

Flask-WTF（https://flask-wtf.readthedocs.org/en/latest/）扩展包装了 WTForms 库，这是一个非常灵活的管理和验证表单的工具，并且可以在 Flask 应用程序中方便地使用。让我们现在安装它，然后我们将定义我们的第一个表单来处理用户登录：

```py
$ pip install flask-wtf

```

接下来，我们将在我们的`users/views.py`模块中定义我们的第一个表单：

```py
from flask import Blueprint

from flask.ext.wtf import Form
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, Length

users = Blueprint('users', __name__, template_folder='templates')

classLoginForm(Form):
 """
 Represents the basic Login form elements & validators.
 """

 username = StringField('username', validators=[DataRequired()])
 password = PasswordField('password', validators=[DataRequired(),
 Length(min=6)])

```

在这里，我们定义了`LoginForm`，它是`Form`的子类，具有`username`和`password`的类属性。这些属性的值分别是`StringField`和`PasswordField`，每个都有自己的验证器集，指示这两个字段的表单数据都需要非空，并且密码字段本身应至少为六个字符长才能被视为有效。

我们的`LoginForm`类将以两种不同的方式被使用，如下所示：

+   它将在我们的`login.html`模板中呈现所需的表单字段

+   它将验证我们需要完成用户成功登录所需的 POST 表单数据

为了实现第一个，我们需要在`application/templates/layout.html`中定义我们的 HTML 布局，使用 Jinja2 模板语言。请注意使用`current_user`对象代理，它通过 Flask-Login 扩展在所有 Jinja 模板中提供，这使我们能够确定正在浏览的人是否已经认证，如果是，则应该向这个人呈现略有不同的页面内容：

```py
<!doctype html>
<html>
  <head>
    <title>Snaps</title>
  </head>

  <body>
    <h1>Snaps</h1>

    {% for message in get_flashed_messages() %}
    <div class="flash">{{ message }}</div>
    {% endfor %}

    {% if not current_user.is_authenticated() %}
    <a href="{{ url_for('users.login') }}">login</a>
    {% else %}
    <a href="{{ url_for('users.logout') }}">logout</a>
    {% endif %}

    <div class="content">
    {% block content %}{% endblock %}
    </div>
  </body>
</html>
```

现在我们已经有了极其基本的布局，我们需要在`application/users/templates/users/login.html`中创建我们的`login.html`页面：

### 注意

当使用蓝图时，`application/users/templates/users/index.html`的相对复杂路径是必需的，因为默认模板加载程序搜索注册的模板路径的方式，它允许相对简单地在主应用程序模板文件夹中覆盖蓝图模板，但会增加一些额外的文件树复杂性。

```py
{% extends "layout.html" %}

{% block content %}

<form action="{{ url_for('users.login')}}" method="post">
  {{ form.hidden_tag() }}
  {{ form.id }}
  <div>{{ form.username.label }}: {{ form.username }}</div>
  {% if form.username.errors %}
  <ul class="errors">{% for error in form.username.errors %}<li>{{ error }}</li>{% endfor %}</ul>
  {% endif %}

  <div>{{ form.password.label }}: {{ form.password }}</div>
  {% if form.password.errors %}
  <ul class="errors">{% for error in form.password.errors %}<li>{{ error }}</li>{% endfor %}</ul>
  {% endif %}

  <div><input type="submit" value="Login"></div>
</form>

{% endblock %}
```

前面的代码将扩展我们之前定义的基本应用程序级`layout.html`，并插入隐藏的表单字段（Flask-WTF 提供的内置 CSRF 保护所需），表单标签，表单输入和提交按钮。我们还将显示 WTForms 返回的内联错误，以防我们提交的数据未通过相关字段的表单验证器。

> **跨站请求伪造**（**CSRF**）*是一种攻击类型，当恶意网站、电子邮件、博客、即时消息或程序导致用户的网络浏览器在用户当前已认证的受信任站点上执行不需要的操作时发生。OWASP 对 CSRF 的定义*

### 注意

防止跨站请求伪造最常见的方法是在发送给用户的每个 HTML 表单中包含一个令牌，然后可以针对已认证用户的会话中的匹配令牌进行验证。如果令牌无法验证，那么表单数据将被拒绝，因为当前认证用户可能并不是自愿提交相关表单数据。

现在我们已经创建了`login.html`模板，接下来我们可以在`application/users/views.py`中挂接一个路由视图处理程序来处理登录和表单逻辑：

```py
from flask import (Blueprint, flash, render_template, url_for, redirect, g)
from flask.ext.login import login_user, logout_user, current_user

from flask.ext.wtf import Form
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, Length

from models import User
from application import flask_bcrypt

users = Blueprint('users', __name__, template_folder='templates')

class LoginForm(Form):
 """
 Represents the basic Login form elements & validators.
 """

 username = StringField('username', 
validators=[DataRequired()])
password = PasswordField('password', 
validators=[DataRequired(),Length(min=6)])

@users.route('/login', methods=['GET', 'POST'])
def login():
 """
Basic user login functionality.

 If the user is already logged in, we
redirect the user to the default snaps index page.

 If the user is not already logged in and we have
form data that was submitted via POST request, we
call the validate_on_submit() method of the Flask-WTF
 Form object to ensure that the POST data matches what
we are expecting. If the data validates, we login the
user given the form data that was provided and then
redirect them to the default snaps index page.

 Note: Some of this may be simplified by moving the actual User
loading and password checking into a custom Flask-WTF validator
for the LoginForm, but we avoid that for the moment, here.
 """

current_user.is_authenticated():
 return redirect(url_for('snaps.listing))

 form = LoginForm()
 if form.validate_on_submit():

 user = User.query.filter_by(
 username=form.username.data).first()

 if not user:
 flash("No such user exists.")
 returnrender_template('users/login.html', form=form)

 if(not flask_bcrypt.check_password_hash(user.password,
 form.password.data)):

 flash("Invalid password.")
 returnrender_template('users/login.html', form=form)

 login_user(user, remember=True)
 flash("Success!  You're logged in.")
 returnredirect(url_for("snaps.listing"))

 return render_template('users/login.html', form=form)

@users.route('/logout', methods=['GET'])
def logout():
 logout_user()
 return redirect(url_for(('snaps.listing'))

```

### 哈希用户密码

我们将更新我们的用户模型，以确保密码在更新“密码”字段时由 Flask-Bcrypt 加密。为了实现这一点，我们将使用 SQLAlchemy 的一个功能，它类似于 Python 的@property 装饰器（以及相关的 property.setter 方法），名为混合属性。

### 注意

混合属性之所以被命名为混合属性，是因为当在类级别或实例级别调用时，它们可以提供完全不同的行为。SQLAlchemy 文档是了解它们在领域建模中可以扮演的各种角色的好地方。

我们将简单地将密码类级属性重命名为`_password`，以便我们的混合属性方法不会发生冲突。随后，我们添加了封装了密码哈希逻辑的混合属性方法，以在属性分配时使用：

### 注意

除了混合属性方法之外，我们对分配密码哈希的要求也可以通过使用 SQLAlchemy TypeDecorator 来满足，这允许我们增加现有类型（例如，String 列类型）的附加行为。

```py
import datetime
from application import db, flask_bcrypt
from sqlalchemy.ext.hybrid import hybrid_property

class User(db.Model):

 # …

 # The hashed password for the user
 _password = db.Column('password', db.String(60))

 # …
 @hybrid_property
 def password(self):
 """The bcrypt'ed password of the given user."""

return self._password

 @password.setter
 def password(self, password):
 """Bcrypt the password on assignment."""

 self._password = flask_bcrypt.generate_password_hash(
 password)

 # …

```

为了生成一个用于测试目的的用户（并验证我们的密码是否在实例构造/属性分配时被哈希），让我们加载 Python 控制台，并使用我们定义的模型和我们创建的 SQLAlchemy 数据库连接自己创建一个用户实例：

### 提示

如果您还没有，不要忘记使用`db.create_all()`来初始化数据库。

```py
>>>from application.users.models import User
>>>user = User(username='test', password='mypassword', email='test@example.com')
>>>user.password
'$2a$12$O6oHgytOVz1hrUyoknlgqeG7TiVS7M.ogRPv4YJgAJyVeUIV8ad2i'
>>>from application import db
>>>db.session.add(user)
>>>db.session.commit()

```

### 配置应用程序 SECRET_KEY

我们需要的最后一点是定义一个应用程序范围的`SECRET_KEY`，Flask-WTF 将使用它来签署用于防止 CSRF 攻击的令牌。我们将在`application/__init__.py`中的应用程序配置中添加此密钥：

```py
from flask import Flask
fromflask.ext.sqlalchemy import SQLAlchemy
fromflask.ext.login import LoginManager
fromflask.ext.bcrypt import Bcrypt

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../snap.db'
app.config['SECRET_KEY'] = "-80:,bPrVzTXp*zXZ0[9T/ZT=1ej08"
# …

```

### 注意

当然，您会想要使用您自己的唯一密钥；最简单的方法是通过`/dev/urandom`来使用您系统内核的随机数设备，对于大多数 Linux 发行版都是可用的。在 Python 中，您可以使用`os.urandom`方法来获得一个具有*n*字节熵的随机字符串。

### 连接蓝图

在我们运行应用程序之前，我们需要使用 Flask 应用程序对象注册我们新创建的用户蓝图。这需要对`application/__init__.py`进行轻微修改：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.login import LoginManager
from flask.ext.bcrypt import Bcrypt

app = Flask(__name__)

# …
from application.users.views import users
app.register_blueprint(users, url_prefix='/users')

# …

```

## 让我们运行这个东西

既然我们已经把所有小部件放在一起，让我们运行应用程序并让事情发生。我们将使用一个类似于我们在上一章中使用的`run.py`文件，它已经适应了我们的应用程序工厂的工作方式：

```py
from application import create_app

app = create_app(config='settings')
app.run(debug=True)

```

该文件被放置在`application`文件夹的同级目录下，然后以通常的方式调用：

```py
$ python run.py

```

访问`http://localhost:5000/users/login`，您应该会看到我们创建的`username`和`password`输入字段。如果您尝试输入无效字段（例如，不存在的用户名），页面将显示相关的错误消息。如果您尝试使用我们在交互提示中创建的用户凭据登录，那么您应该会看到文本：`Success! You logged in`。

## 快照的数据模型

既然我们已经创建了我们的基本用户模型、视图函数，并连接了我们的身份验证系统，让我们创建一个新的蓝图来存储我们的快照所需的模型，在`application/snaps/models.py`下。

### 提示

不要忘记创建`application/snaps/__init__.py`，否则该文件夹将无法被识别为一个包！

这个模型将与我们的用户模型非常相似，但将包含有关用户和他们的快照之间关系的附加信息。在 SQLAlchemy 中，我们将通过使用`ForeignKey`对象和`relationship`方法来描述表中记录之间的关系：

```py
import datetime
import hashlib
from application import db

class Snap(db.Model):

 # The primary key for each snap record.
 id = db.Column(db.Integer, primary_key=True)

 # The name of the file; does not need to be unique.
 name = db.Column(db.String(128))

 # The extension of the file; used for proper syntax 
 # highlighting
 extension = db.Column(db.String(12))

 # The actual content of the snap
 content = db.Column(db.Text())

 # The unique, un-guessable ID of the file
 hash_key = db.Column(db.String(40), unique=True)

 #  The date/time that the snap was created on.
 created_on = db.Column(db.DateTime, 
 default=datetime.datetime.utcnow,index=True)

 # The user this snap belongs to
 user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

 user = db.relationship('User', backref=db.backref(
 'snaps', lazy='dynamic'))

 def __init__(self, user_id, name, content, extension):
 """
 Initialize the snap object with the required attributes.
 """

 self.user_id = user_id
 self.name = name
 self.content = content
 self.extension = extension

self.created_on = datetime.datetime.utcnow()

 # This could be made more secure by combining the 
 # application SECRET_KEYin the hash as a salt.
 self.hash_key = hashlib.sha1(self.content + str(self.created_on)).hexdigest()

 def __repr__(self):
 return '<Snap {!r}>'.format(self.id)

```

这个模型大部分应该是相对熟悉的；它与我们之前为用户模式构建的模型并没有太大的不同。对于我们的快照，我们将需要一些强制属性，如下所示：

+   `user_id`：这是创建快照的用户的 ID。由于我们当前的实现将要求用户进行身份验证才能创建快照，所有生成的快照都将与发布它们的用户相关联。这也将使我们在以后轻松扩展系统，以包括用户个人资料、个人快照统计信息和删除快照的能力。

+   `created_on`：这在构造函数中设置为当前的 UTC 时间戳，并将用于按降序排序以在我们的首页上以列表形式显示它们。

+   `hash_key`：这个属性也在构造函数中设置，是快照内容与创建时间戳的加密哈希。这给了我们一个不容易猜测的唯一安全 ID，我们可以用它来在以后引用快照。

### 注意

尽管我们为前面的`hash_key`描述的条件并不保证该值是唯一的，快照哈希键的唯一性也通过数据库级别的唯一索引约束得到了强制。

+   `content`：这是快照本身的内容——模型的主要部分。

+   `extension`：这是快照的文件扩展名，这样我们就可以包含简单的语法高亮。

+   `name`：这是快照的名称，不需要是唯一的。

+   `user`：这是一个特殊属性，声明每个快照实例都与一个用户实例相关联，并允许我们访问创建快照的用户的数据。`backref`选项还指定了反向应该是可能的：也就是说，通过用户实例上的快照属性访问用户创建的所有快照。

### 使用内容敏感的默认函数更好的默认值

对前面的模型可以进行的一个改进是删除显式的`__init__`方法。最初定义它的唯一原因是确保可以从内容字段的值构造`hash_key`字段。虽然在大多数情况下，定义的显式对象构造函数已经足够好了，但 SQLAlchemy 提供了功能，允许我们根据另一个字段的内容设置一个字段的默认值。这被称为**上下文敏感的默认函数**，可以在`application/snaps/models.py`模块的顶部声明为这样：

```py
defcontent_hash(context):
 # This could be made more secure by combining the
 # application SECRET_KEY in the hash as a salt.
 content = context.current_parameters['content']
 created_on = context.current_parameters['created_on']
 return hashlib.sha1(content + str(created_on)).hexdigest()

```

一旦存在这个方法，我们就可以将`hash_key`列的默认参数定义为我们的`content_hash`内容敏感的默认值：

```py
# The unique, un-guessable ID of the file
hash_key = db.Column(db.String(40), unique=True, 
 default=content_hash)

```

## 快照视图处理程序

接下来，我们将创建所需的视图和模板，以列出和添加快照。为此，我们将在`application/snaps/views.py`中实例化一个`Blueprint`对象，并声明我们的路由处理程序：

```py
from flask import Blueprint
from flask.ext.login import login_required

from .models import Snap

snaps = Blueprint('snaps', __name__, template_folder='templates')

@snaps.route('/', methods=['GET'])
def listing():
"""List all snaps; most recent first."""

@snaps.route('/add', methods=['GET', 'POST'])
@login_required
def add():
 """Add a new snap."""

```

请注意，我们已经用`@login_required`装饰器包装了我们的`add()`路由处理程序，这将阻止未经身份验证的用户访问此端点的所有定义的 HTTP 动词（在本例中为 GET 和 POST），并返回 401。

### 注意

与其让服务器返回 HTTP 401 未经授权，不如配置 Flask-Login 将未经身份验证的用户重定向到登录页面，方法是将`login_manager.login_view`属性设置为登录页面本身的`url_for`兼容位置，而在我们的情况下将是`users.login`。

现在，让我们创建 WTForm 对象来表示一个快照，并将其放在`application/snaps/views.py`模块中：

```py
from flask.ext.wtf import Form
from wtforms import StringField
from wtforms.widgets import TextArea
from wtforms.validators import DataRequired

class SnapForm(Form):
 """Form for creating new snaps."""

 name = StringField('name', validators=[DataRequired()])
 extension = StringField('extension', 
 validators=[DataRequired()])
 content = StringField('content', widget=TextArea(),
 validators=[DataRequired()])

```

### 提示

虽然这在某种程度上是个人偏好的问题，但使用 WTForms（或任何其他类似的抽象）创建的表单可以放在模型旁边，而不是视图。或者，更进一步地，如果您有许多不同的表单与复杂的数据关系，也许将所有声明的表单放在应用程序的自己的模块中也是明智的。

我们的快照需要一个名称、一个扩展名和快照本身的内容，我们已经在前面的表单声明中封装了这些基本要求。让我们实现我们的`add()`路由处理程序：

```py
from flask import Blueprint, render_template, url_for, redirect, current_app, flash
from flask.ext.login import login_required, current_user
from sqlalchemy import exc

from .models import Snap
from application import db

# …

@snaps.route('/add', methods=['GET', 'POST'])
@login_required
def add():
 """Add a new snap."""

 form = SnapForm()

 if form.validate_on_submit():
 user_id = current_user.id

 snap = Snap(user_id=user_id, name=form.name.data,
 content=form.content.data, 
 extension=form.extension.data)
 db.session.add(snap)

try:
 db.session.commit()
 except exc.SQLAlchemyError:
 current_app.exception("Could not save new snap!")
 flash("Something went wrong while posting your snap!")

 else:
 return render_template('snaps/add.html', form=form)

 return redirect(url_for('snaps.listing'))

```

简而言之，我们将验证提交的 POST 数据，以确保它满足我们在`SnapForm`类声明中指定的验证器，然后继续使用提供的表单数据和当前认证用户的 ID 来实例化一个`Snap`对象。构建完成后，我们将将此对象添加到当前的 SQLAlchemy 会话中，然后尝试将其提交到数据库。如果发生 SQLAlchemy 异常（所有 SQLAlchemy 异常都继承自`salalchemy.exc.SQLALchemyError`），我们将记录异常到默认的应用程序日志处理程序，并设置一个闪存消息，以便提醒用户发生了意外情况。

为了完整起见，我们将在这里包括极其简单的`application/snaps/templates/snaps/add.html` Jinja 模板：

```py
{% extends "layout.html" %}

{% block content %}
<form action="{{ url_for('snaps.add')}}" method="post">

  {{ form.hidden_tag() }}
  {{ form.id }}

  <div class="row">
    <div>{{ form.name.label() }}: {{ form.name }}</div>
    {% if form.name.errors %}
    <ul class="errors">{% for error in form.name.errors %}<li>{{ error }}</li>{% endfor %}</ul>
    {% endif %}

    <div>{{ form.extension.label() }}: {{ form.extension }}</div>
    {% if form.extension.errors %}
    <ul class="errors">{% for error in form.extension.errors %}<li>{{ error }}</li>{% endfor %}</ul>
    {% endif %}
  </div>

  <div class="row">
    <div>{{ form.content.label() }}: {{ form.content }}</div>
    {% if form.content.errors %}
    <ul class="errors">{% for error in form.content.errors %}<li>{{ error }}</li>{% endfor %}</ul>
    {% endif %}
  </div>

  <div><input type="submit" value="Snap"></div>
</form>

{% endblock %}
```

完成了`add()`处理程序和相关模板后，现在是时候转向`listing()`处理程序了，这将偶然成为我们应用程序的登陆页面。列表页面将以相反的时间顺序显示最近发布的 20 个快照：

```py
@snaps.route('/', methods=['GET'])
def listing():
 """List all snaps; most recent first."""
 snaps = Snap.query.order_by(
 Snap.created_on.desc()).limit(20).all()
 return render_template('snaps/index.html', snaps=snaps)

```

`application/snaps/templates/snaps/add.html` Jinja 模板呈现了我们从数据库中查询到的快照：

```py
{% extends "layout.html" %}

{% block content %}
<div class="new-snap">
  <p><a href="{{url_for('snaps.add')}}">New Snap</a></p>
</div>

{% for snap in snaps %}
<div class="snap">
  <span class="author">{{snap.user.username}}</span>, published on <span class="date">{{snap.created_on}}</span>
  <pre><code>{{snap.content}}</code></pre>
</div>
{% endfor %}

{% endblock %}
```

接下来，我们必须确保我们创建的快照蓝图已加载到应用程序中，并通过将其添加到`application/__init__.py`模块来添加到根/URI 路径：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.login import LoginManager
from flask.ext.bcrypt import Bcrypt

# …

from application.users import models as user_models
from application.users.views import users
from application.snaps.views import snaps

app.register_blueprint(users, url_prefix='/users')
app.register_blueprint(snaps, url_prefix='')

@login_manager.user_loader
de fload_user(user_id):
 return user_models.User.query.get(int(user_id))

```

为了测试我们的新功能，我们需要将新创建的快照模型添加到我们的数据库中。我们可以通过执行我们在本章前面描述的`db.create_all()`函数来实现这一点。由于我们经常运行这个命令，让我们将其放在与我们的主应用程序包文件同级的脚本中，并将文件命名为`database.py`：

```py
from application import db
db.create_all()

```

一旦就位，我们可以简单地使用 Python 解释器执行脚本，以在我们的数据库中创建新的快照模型：

```py
$ python database.py

```

现在，我们的数据库应该已经根据我们的模型定义更新了，让我们确保应用程序按预期运行：

```py
$ python run.py

```

假设没有错误，您应该能够访问显示的 URL，并使用我们在本章早些时候创建的用户之一的凭据登录。当然，您可以通过交互式 Python 解释器创建一个新用户，然后使用这些凭据来测试应用程序的身份验证功能：

```py
$ python
>>>from application import db
>>>from application.users.models import User
>>>user = User(name='test', email='test@example.com', password='foobar')
>>>db.session.add(user)
>>>db.session.commit(user)

```

# 总结

通过阅读本章并构建 Snap 应用程序，我们已经看到了 Flask 如何通过使用扩展来增强，例如 Flask-WTF（用于 Web 表单创建和验证）、Flask-SQLAlchemy（用于与 SQLAlchemy 数据库抽象库的简单集成）、Flask-Bcrypt（用于密码哈希）和 Flask-Login（用于简单用户登录系统的标准实现要求的抽象）。虽然 Flask 本身相对简洁，但可用的扩展生态系统使得构建一个完全成熟的用户认证应用程序可以快速且相对轻松地完成。

我们探讨了上述扩展及其有用性，包括 Flask-WTF 和 Flask-SQLAlchemy，并设计了一个基于蓝图的简单应用程序，集成了上述所有组件。虽然 Snap 应用程序本身非常简单，还有很多功能需要实现，但它非常容易更新和添加其他功能。

在下一章中，我们将构建一个具有更复杂数据模型的应用程序，并包含一些在今天的 Web 应用程序中常见的社交功能。此外，它将被构建和设置为单元和功能测试，这是任何微不足道的应用程序都不应该缺少的功能。
