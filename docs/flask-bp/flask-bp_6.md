# 第六章：Hublot - Flask CLI 工具

在管理 Web 应用程序时，通常有一些任务是我们希望完成的，而不必创建整个管理 Web 界面；即使这可能相对容易地通过诸如 Flask-Admin 之类的工具来实现。许多开发人员首先转向 shell 脚本语言。Bash 几乎在大多数现代 Linux 操作系统上都是通用的，受到系统管理员的青睐，并且足够强大，可以脚本化可能需要的任何管理任务。

尽管可敬的 Bash 脚本绝对是一个选择，但编写一个基于 Python 的脚本会很好，它可以利用我们为 Web 应用程序精心制作的一些应用程序特定的数据处理。这样做，我们可以避免重复大量精力和努力，这些精力和努力是在创建、测试和部署数据模型和领域逻辑的痛苦过程中投入的，这是任何 Web 应用程序的核心。这就是 Flask-Script 的用武之地。

### 注意

在撰写本文时，Flask 尚未发布 1.0 版本，其中包括通过 Flask 作者开发的`Click`库进行集成的 CLI 脚本处理。由于 Flask/Click 集成的 API 在现在和 Flask 1.0 发布之间可能会发生重大变化，因此我们选择通过 Flask-Script 包来实现本章讨论的 CLI 工具，这已经是 Flask 的事实标准解决方案相当长的时间了。但是，通过 Click API 创建管理任务可以考虑用于任何新的 Flask 应用程序-尽管实现方式有很大不同，但基本原则是足够相似的。

除了我们可能需要一个 shell 脚本执行的不经常的任务，例如导出计算数据，向一部分用户发送电子邮件等，还有一些来自我们以前应用程序的任务可以移植到 Flask-Script CLI 命令中：

+   创建/删除我们当前的数据库模式，从而替换我们以前项目中的`database.py`

+   运行我们的 Werkzeug 开发服务器，替换以前项目中的`run.py`

此外，由于 Flask-Script 是为 Flask 应用程序编写可重用 CLI 脚本的当前事实标准解决方案，许多其他扩展发布 CLI 命令，可以集成到您的现有应用程序中。

在本章中，我们将创建一个应用程序，将从`Github` API 中提取的数据存储在本地数据库中。

### 注意

Git 是一种**分布式版本控制系统**（**DVCS**），在过去几年中变得非常流行，而且理由充分。它已经迅速成为了大量使用各种语言编写的开源项目的首选版本控制系统。

GitHub 是 Git 开源和闭源代码存储库的最知名的托管平台，还配备了一个非常完整的 API，允许根据提供的经过身份验证的凭据，以编程方式访问可用的数据和元数据（评论、拉取请求、问题等）。

为了获取这些数据，我们将创建一个简单的 Flask 扩展来封装基于 REST 的 API 查询，以获取相关数据，然后我们将使用这个扩展来创建一个 CLI 工具（通过 Flask-Script），可以手动运行或连接到基于事件或时间的调度程序，例如 cron。

然而，在我们进行任何操作之前，让我们建立一个非常简单的应用程序框架，以便我们可以开始 Flask-Script 集成。

# 开始

我们再次使用基本的基于 Blueprint 的应用程序结构，并为这个新的冒险创建一个全新的虚拟环境和目录：

```py
$ mkdir -p ~/src/hublot && cd ~/src/hublot
$ mkvirtualenv hublot
$ pip install flask flask-sqlalchemy flask-script

```

我们将开始使用的应用程序布局与我们在以前基于 Blueprint 的项目中使用的非常相似，主要区别在于`manage.py`脚本，它将是我们的 Flask-Script CLI 命令的主要入口点。还要注意缺少`run.py`和`database.py`，这是我们之前提到的，并且很快会详细解释的。

```py
├── application
│   ├── __init__.py
│   └── repositories
│       ├── __init__.py
│       └── models.py
└── manage.py

```

与我们之前的工作保持一致，我们继续使用“应用工厂”模式，允许我们在运行时实例化我们的应用，而不是在模块导入时进行，就像我们将要使用的 Flask-SQLAlchemy 扩展一样。

我们的`application/__init__.py`文件包含以下内容，您应该会非常熟悉：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy

# Initialize the db extension, but without configuring
# it with an application instance.
db = SQLAlchemy()

def create_app(config=None):
    app = Flask(__name__)

    if config is not None:
        app.config.from_object(config)

    # Initialize extensions
    db.init_app(app)

    return app
```

我们的`application/settings.py`文件包含了我们对于 Flask-SQLAlchemy 应用程序所需的基本内容：

```py
SQLALCHEMY_DATABASE_URI = 'sqlite:///../hublot.db'
```

### 注意

对于这个特定项目，我们将使用 SQLite 作为我们的首选数据库；如果您决定使用不同的数据库，请相应调整 URI。

为了简便起见，我们将引入简化的`Repository`和`Issue`模型，这些模型将包含我们想要收集的数据。这些模型将存在于`application/repositories/models.py`中：

```py
from application import db
from sqlalchemy.schema import UniqueConstraint
import datetime

class Repository(db.Model):
    """Holds the meta-information about a particular
    Github repository."""

    # The unique primary key for the local repository record.
    id = db.Column(db.Integer, primary_key=True)

    # The name of the repository.
    name = db.Column(db.String(length=255), nullable=False)

    # The github org/user that owns the repository.
    owner = db.Column(db.String(length=255), nullable=False)

    # The description (if any) of the repository.
    description = db.Column(db.Text())

    #  The date/time that the record was created on.
    created_on = db.Column(db.DateTime(), 
        default=datetime.datetime.utcnow, index=True)

    # The SQLAlchemy relation for the issues contained within this
    # repository.
    issues = db.relationship('Issue')

    __table_args__ = (UniqueConstraint('name', 'owner'), )

    def __repr__(self):
        return u'<Repository {}>'.format(self.name)
```

`Repository`模型实例将包含与`Issue`模型的一对多关系相关的给定 Git 存储库的元数据，我们将在下面定义。我们在这个`Repository`类中声明的字段在大部分情况下应该是不言自明的，唯一的例外是`__table__args__ dunder`。

### 注意

**dunder**是一个 Python 特有的新词，用于指代以两个下划线开头的任何变量或方法：*双下划线*或*dunder*。有几个内置的 dunder 方法（例如，`__init__`）和属性（例如，`__name__`），任何您声明并以两个下划线前缀的属性/方法/函数也将属于这个类别。

这个类属性允许我们能够为创建的底层 SQLAlchemy 表指定特定于表的配置。在我们的情况下，我们将用它来指定一个 UniqueConstraint 键，这个键是由名称和所有者的组合值组成的，否则通过典型的基于属性的字段定义是不可能的。

此外，我们定义了一个 issues 属性，其值是与`Issue`模型的关系；这是经典的一对多关系，访问存储库实例的 issues 属性将产生与相关存储库关联的问题列表。

### 注意

请注意，指定的关系不包括与查询性质或相关数据加载行为有关的任何参数。我们正在使用此应用程序的默认行为，这对于包含大量问题的存储库来说并不是一个好主意——在这种情况下，可能会更好地选择先前章节中使用的动态延迟加载方法。

我们在`Repository`模型中提到的`Issue`模型旨在包含与此处托管的 Git 存储库相关联的 GitHub 问题元数据。由于问题只在存储库的上下文中有意义，我们确保`repository_id`外键存在于所有问题中：

```py
class Issue(db.Model):
    """Holds the meta information regarding an issue that
    belongs to a repository."""

    # The autoincremented ID of the issue.
    id = db.Column(db.String(length=40), primary_key=True)
    # The repository ID that this issue belongs to.

    #
    # This relationship will produce a `repository` field
    # that will link back to the parent repository.
    repository_id = db.Column(db.Integer(), 
        db.ForeignKey('repository.id'))

    # The title of the issue
    title = db.Column(db.String(length=255), nullable=False)

    # The issue number
    number = db.Column(db.Integer(), nullable=False)

    state = db.Column(db.Enum('open', 'closed'), nullable=False)

    def __repr__(self):
        """Representation of this issue by number."""
        return '<Issue {}>'.format(self.number)
```

每个`Issue`模型的实例将封装关于创建的 GitHub 问题的非常有限的信息，包括问题编号、问题的状态（*关闭*或*打开*）以及问题的标题。

在以前的章节中，我们会创建一个`database.py`脚本来初始化在数据库中构建我们的 SQLAlchemy 模型。然而，在本章中，我们将使用 Flask-Script 来编写一个小的 CLI 命令，它将做同样的事情，但为我们提供一个更一致的框架来编写这些小的管理工具，并避免随着时间的推移而困扰任何非平凡应用的独立脚本文件的问题。

## manage.py 文件

按照惯例，Flask-Script 的主要入口点是一个名为`manage.py`的 Python 文件，我们将其放在`application/`包的同级目录中，就像我们在本章开头描述的项目布局一样。虽然 Flask-Script 包含了相当多的选项-配置和可定制性-我们将使用最简单的可用调用来封装我们在以前章节中使用的`database.py` Python 脚本的功能，以处理我们数据库的初始化。

我们实例化了一个`Manager`实例，它将处理我们各种命令的注册。`Manager`构造函数接受一个 Flask 应用实例作为参数，但它也（幸运地！）可以接受一个实现可调用接口并返回应用实例的函数或类：

```py
from flask.ext.script import Manager
from application import create_app, db

# Create the `manager` object with a
# callable that returns a Flask application object.
manager = Manager(app=create_app)
```

现在我们有了一个`manager`实例，我们可以使用这个实例的`command`方法来装饰我们想要转换为 CLI 命令的函数：

```py
@manager.command
def init_db():
 """Initialize SQLAlchemy database models."""

 db.create_all()

```

### 注意

请注意，默认情况下，我们用`command`方法包装的函数名称将是 CLI 调用中使用的标识符。

为了使整个过程运行起来，当我们直接调用`manage.py`文件时，我们调用管理器实例的`run`方法：

```py
if __name__ == '__main__':
    manager.run()
```

此时，我们可以通过 Python 解释器执行我们的 CLI 命令：

```py
$ python manage.py init_db

```

假设一切都按预期工作，我们应该看不到任何结果（或错误），并且我们的数据库应该被初始化为我们在模型定义中指定的表、列和索引。

让我们创建一个截然相反的命令，允许我们销毁本地数据库；在开发过程中对数据模型进行大量更改时，这有时会很方便：

```py
@manager.command
def drop_db():
 if prompt_bool(
 "Are you sure you want to lose all your data"):
 db.drop_all()

```

我们以与之前定义的`init_db`命令相同的方式调用这个新创建的`drop_db`命令：

```py
$ python manage.py drop_db

```

### 内置默认命令

除了让我们能够快速定义自己的 CLI 命令之外，Flask-Script 还包括一些默认值，这样我们就不必自己编写它们：

```py
usage: manage.py [-?] {shell,drop_db,init_db,runserver} ...

positional arguments:
 {shell,drop_db,init_db,runserver}
 shell           Runs a Python shell inside Flask application 
 context.
 drop_db
 init_db         Initialize SQLAlchemy database models.
 runserver       Runs the Flask development server i.e. 
 app.run()

optional arguments:
 -?, --help            show this help message and exit

```

### 注意

Flask-Script 会根据相关函数的`docstrings`自动生成已注册命令的帮助文本。此外，运行`manage.py`脚本而没有指定命令或使用`help`选项将显示可用顶级命令的完整列表。

如果出于任何原因，我们想要自定义默认设置，这是相对容易实现的。例如，我们需要开发服务器在 6000 端口上运行，而不是默认的 5000 端口：

```py
from flask.ext.script import Manager, prompt_bool, Server
# …

if __name__ == '__main__':
    manager.add_command('runserver', Server(port=6000))
    manager.run()
```

在这里，我们使用了定义 CLI 命令的另一种方法，即使用`manager.add_command`方法，它将一个名称和`flask.ext.script.command`的子类作为第二个参数。

同样地，我们可以覆盖默认的 shell 命令，以便我们的交互式 Python shell 包含对我们配置的 Flask-SQLAlchemy 数据库对象的引用，以及 Flask 应用对象：

```py
def _context():
    """Adds additional objects to our default shell context."""
    return dict(db=db, repositories=repositories)

if __name__ == '__main__':
    manager.add_command('runserver', Server(port=6000))
    manager.add_command('shell', Shell(make_context=_context))
    manager.run()
```

我们可以通过执行`manage.py`脚本来验证我们的`db`对象是否已经被包含，以调用交互式 shell。

```py
$ python manage.py shell

>>> type(db)
<class 'flask_sqlalchemy.SQLAlchemy'>
>>>

```

验证默认的 Flask 应用服务器是否在我们指定的端口上运行：

```py
$ python manage.py runserver
 * Running on http://127.0.0.1:6000/ (Press CTRL+C to quit)

```

Flask-Script 为默认的`runserver`和`shell`命令提供了几个配置选项，包括禁用它们的能力。您可以查阅在线文档以获取更多详细信息。

## Blueprints 中的 Flask-Script 命令

在我们应用程序级别的`manage.py`中创建临时 CLI 命令的能力既是一种祝福又是一种诅咒：祝福是因为它需要非常少的样板代码就可以运行起来，诅咒是因为它很容易变成一堆难以管理的代码混乱。

为了避免任何非平凡应用程序的不可避免的最终状态，我们将使用 Flask-Script 中子管理器的未充分利用的功能，以创建一组 CLI 命令，这些命令将存在于蓝图中，但可以通过标准的`manage.py`调用访问。这应该使我们能够将命令行界面的领域逻辑保存在与我们基于 Web 的组件的领域逻辑相同的位置。

### 子管理器

我们的第一个 Flask-Script 子管理器将包含解析 GitHub 项目 URL 的逻辑，以获取我们需要创建有效的`Repository`模型记录的组件部分：

```py
$ python manage.py repositories add "https://github.com/mitsuhiko/flask"\
 --description="Main Flask repository"

```

总体思路是，我们希望能够使用从“repositories”子管理器的“add”函数提供的位置和命名参数解析出名称、所有者和描述，从而创建一个新的`Repository`对象。

让我们开始创建一个模块，该模块将包含我们的存储库 CLI 命令，即`application/repositories/cli.py`，目前为空的`add`函数：

```py
from flask.ext.script import Manager

repository_manager = Manager(
    usage="Repository-based CLI actions.")

@repository_manager.command
def add():
    """Adds a repository to our database."""
    pass
```

请注意，我们的`repository_manager`实例是在没有应用程序实例或可返回应用程序实例的可调用对象的情况下创建的。我们将新创建的子管理器实例注册到我们的主应用程序管理器中，而不是在此处提供应用程序对象。

```py
from flask.ext.script import Manager, prompt_bool, Server, Shell
from application import create_app, db, repositories
from application.repositories.cli import repository_manager

# Create the `manager` object with a
# callable that returns a Flask application object.
manager = Manager(app=create_app)

# …
# …

if __name__ == '__main__':
    manager.add_command('runserver', Server(port=6000))
    manager.add_command('shell', Shell(make_context=_context))
 manager.add_command('repositories', repository_manager)
    manager.run()
```

这将使我们能够调用`repositories`管理器并显示可用的子命令：

```py
$ python manage.py repositories --help
usage: Repository-based CLI actions.

Repository-based CLI actions.

positional arguments:
 {add}
 add       Adds a repository to our database.

optional arguments:
 -?, --help  show this help message and exit

```

虽然这将不会产生任何结果（因为函数体是一个简单的 pass 语句），但我们可以调用我们的`add`子命令：

```py
$ python manage.py repositories add

```

### 所需和可选参数

在 Flask-Script 管理器中注册的任何命令都可以有零个或多个必需参数，以及任意默认值的可选参数。

我们的`add`命令需要一个强制参数，即要添加到我们数据库中的存储库的 URL，以及一个可选参数，即此存储库的描述。命令装饰器处理了许多最基本的情况，将命名函数参数转换为它们的 CLI 参数等效项，并将具有默认值的函数参数转换为可选的 CLI 参数。

这意味着我们可以指定以下函数声明来匹配我们之前写下的内容：

```py
@repository_manager.command
def add(url, description=None):
    """Adds a repository to our database."""

    print url, description
```

这使我们能够捕获提供给我们的 CLI 管理器的参数，并在我们的函数体中轻松地使用它们：

```py
$ python manage.py repositories add "https://github.com/mitsuhiko/flask" --description="A repository to add!"

https://github.com/mitsuhiko/flask A repository to add!

```

由于我们已经成功地编码了 CLI 工具的所需接口，让我们添加一些解析，以从 URL 中提取出我们想要的相关部分： 

```py
@repository_manager.command
def add(url, description=None):
    """Adds a repository to our database."""

 parsed = urlparse(url)

 # Ensure that our repository is hosted on github
 if parsed.netloc != 'github.com':
 print "Not from Github! Aborting."
 return 1

 try:
 _, owner, repo_name = parsed.path.split('/')
 except ValueError:
 print "Invalid Github project URL format!"
        return 1
```

### 注意

我们遵循`*nix`约定，在脚本遇到错误条件时返回一个介于 1 和 127 之间的非零值（约定是对语法错误返回 2，对其他任何类型的错误返回 1）。由于我们期望我们的脚本能够成功地将存储库对象添加到我们的数据库中，任何情况下如果这种情况没有发生，都可以被视为错误条件，因此应返回一个非零值。

现在我们正确捕获和处理 CLI 参数，让我们使用这些数据来创建我们的`Repository`对象，并将它们持久化到我们的数据库中：

```py
from flask.ext.script import Manager
from urlparse import urlparse
from application.repositories.models import Repository
from application import db
import sqlalchemy

# …

@repository_manager.command
def add(url, description=None):
    """Adds a repository to our database."""

    parsed = urlparse(url)

    # Ensure that our repository is hosted on github
    if parsed.netloc != 'github.com':
        print "Not from Github! Aborting."
        return 1

    try:
        _, owner, repo_name = parsed.path.split('/')
    except ValueError:
        print "Invalid Github project URL format!"
        return 1

 repository = Repository(name=repo_name, owner=owner)
 db.session.add(repository)

 try:
 db.session.commit()
 except sqlalchemy.exc.IntegrityError:
 print "That repository already exists!"
 return 1

 print "Created new Repository with ID: %d" % repository.id
    return 0
```

### 注意

请注意，我们已经处理了向数据库添加重复存储库（即具有相同名称和所有者的存储库）的情况。如果不捕获`IntegrityError`，CLI 命令将失败并输出指示未处理异常的堆栈跟踪。

现在运行我们新实现的 CLI 命令将产生以下结果：

```py
$ python manage.py repositories add "https://github.com/mitsuhiko/flask" --description="A repository to add!"

Created new Repository with ID: 1

```

成功创建我们的`Repository`对象可以在我们的数据库中进行验证。对于 SQLite，以下内容就足够了：

```py
$ sqlite3 hublot.db
SQLite version 3.8.5 2014-08-15 22:37:57
Enter ".help" for usage hints.

sqlite> select * from repository;

1|flask|mitsuhiko|A repository to add!|2015-07-22 04:00:36.080829

```

## Flask 扩展 - 基础知识

我们花了大量时间安装、配置和使用各种 Flask 扩展（Flask-Login、Flask-WTF、Flask-Bcrypt 等）。它们为我们提供了一个一致的接口来配置第三方库和工具，并经常集成一些使应用程序开发更加愉快的 Flask 特定功能。然而，我们还没有涉及如何构建自己的 Flask 扩展。

### 注意

我们只会查看创建有效的 Flask 扩展所需的框架，以便在项目中本地使用。如果您希望打包您的自定义扩展并在 PyPi 或 GitHub 上发布它，您将需要实现适当的`setup.py`和 setuptools 机制，以使这成为可能。您可以查看 setuptools 文档以获取更多详细信息。

### 何时应该使用扩展？

Flask 扩展通常属于以下两类之一：

+   封装第三方库提供的功能，确保当同一进程中存在多个 Flask 应用程序时，该第三方库将正常运行，并可能添加一些使与 Flask 集成更具体的便利函数/对象；例如，Flask-SQLAlchemy

+   不需要第三方库的模式和行为的编码，但确保应用程序具有一组一致的功能；例如，Flask-Login

您将在野外遇到或自己开发的大多数 Flask 扩展都属于第一类。第二类有点异常，并且通常是由在多个应用程序中观察到的常见模式抽象和精炼而来，以至于可以将其放入扩展中。

### 我们的扩展 - GitHubber

本章中我们将构建的扩展将封装`Github` API 的一个小部分，这将允许我们获取先前跟踪的给定存储库的问题列表。

### 注意

`Github` API 允许的功能比我们需要的更多，文档也很好。此外，存在几个第三方 Python 库，封装了大部分`Github` API，我们将使用其中一个。

为了简化与 GitHub 的 v3 API 的交互，我们将在本地虚拟环境中安装`github3.py` Python 包：

```py
$ pip install github3.py

```

由于我们正在在我们的 Hublot 应用程序中开发扩展，我们不打算引入自定义 Flask 扩展的单独项目的额外复杂性。然而，如果您打算发布和/或分发扩展，您将希望确保它以这样的方式结构化，以便可以通过 Python 包索引提供并通过 setuptools（或 distutils，如果您更愿意只使用标准库中包含的打包工具）进行安装。

让我们创建一个`extensions.py`模块，与`application/repositories/ package`同级，并引入任何 Flask 扩展都应包含的基本结构：

```py
class Githubber(object):
    """
    A Flask extension that wraps necessary configuration
    and functionality for interacting with the Github API
    via the `github3.py` 3rd party library.
    """

    def __init__(self, app=None):
        """
        Initialize the extension.

        Any default configurations that do not require
        the application instance should be put here.
        """

        if app:
            self.init_app(app)

    def init_app(self, app):
        """
        Initialize the extension with any application-level 
        Configuration requirements.
        """
        self.app = app
```

对于大多数扩展，这就是所需的全部。请注意，基本扩展是一个普通的 Python 对象（俗称为 POPO）定义，增加了一个`init_app`实例方法。这个方法并不是绝对必要的。如果您不打算让扩展使用 Flask 应用程序对象（例如加载配置值）或者不打算使用应用程序工厂模式，那么`init_app`是多余的，可以省略。

我们通过添加一些配置级别的检查来完善扩展，以确保我们具有`GITHUB_USERNAME`和`GITHUB_PASSWORD`以进行 API 身份验证访问。此外，我们将当前扩展对象实例存储在`app.extensions`中，这使得扩展的动态使用/加载更加简单（等等）：

```py
    def init_app(self, app):
        """
        Initialize the extension with any application-level 
        Configuration requirements.

        Also store the initialized extension and application state
        to the `app.extensions`
        """

        if not hasattr(app, 'extensions'):
            app.extensions = {}

        if app.config.get('GITHUB_USERNAME') is None:
            raise ValueError(
                "Cannot use Githubber extension without "
                "specifying the GITHUB_USERNAME.")

        if app.config.get('GITHUB_PASSWORD') is None:
            raise ValueError(
                "Cannot use Githubber extension without "
                "specifying the GITHUB_PASSWORD.")

        # Store the state of the currently configured extension in
        # `app.extensions`.
        app.extensions['githubber'] = self
        self.app = app
```

### 注意

对`Github` API 进行身份验证请求需要某种形式的身份验证。GitHub 支持其中几种方法，但最简单的方法是指定帐户的用户名和密码。一般来说，这不是你想要要求用户提供的东西：最好在这些情况下使用 OAuth 授权流程，以避免以明文形式存储用户密码。然而，对于我们相当简单的应用程序和自定义扩展，我们将放弃扩展的 OAuth 实现（我们将在后面的章节中更广泛地讨论 OAuth），并使用用户名和密码组合。

单独使用，我们创建的扩展并没有做太多事情。让我们通过添加一个装饰属性的方法来修复这个问题，该方法实例化`github3.py Github` API 客户端库：

```py
from github3 import login

class Githubber(object):
    # …
    def __init__(self, app=None):

        self._client = None
        # …

    @property
    def client(self):
        if self._client:
            return self._client

        gh_client = login(self.app.config['GITHUB_USERNAME'],
                password=self.app.config['GITHUB_PASSWORD'])

        self._client = gh_client
        return self._client
```

在前面的`client`方法中，我们实现了缓存属性模式，这将确保我们只实例化一个`github3.py`客户端，每个创建的应用程序实例只实例化一次。此外，扩展将在第一次访问时延迟加载`Github` API 客户端，这通常是一个好主意。一旦应用程序对象被初始化，这让我们可以使用扩展的客户端属性直接与`github3.py` Python 库进行交互。

现在我们已经为我们的自定义 Flask 扩展设置了基本的设置，让我们在`application/__init__.py`中的应用工厂中初始化它并配置扩展本身：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from application.extensions import Githubber

# …
hubber = Githubber()

def create_app(config=None):
    app = Flask(__name__)
    # …

    # Initialize any extensions and bind blueprints to the
    # application instance here.
    db.init_app(app)
 hubber.init_app(app)

    return app
```

注意`hubber = Githubber()`的初始化和赋值发生在工厂本身之外，但实际的`init_app(app)`方法调用和隐含的扩展配置发生在我们初始化 Flask 应用程序对象之后的工厂中。你可能已经注意到了这种分割模式（我们在之前的章节中也讨论过几次），但现在你已经通过开发自己的扩展看到了它的原因。

考虑到这一点，我们在`application/repositories/cli.py`模块中添加了一个额外的函数，以增加一些额外的 CLI 工具功能：

```py
from flask.ext.script import Manager
from urlparse import urlparse
from application.repositories.models import Repository, Issue
from application import db, hubber
import sqlalchemy

# …

@repository_manager.command
def fetch_issues(repository_id):
    """Fetch all commits for the given Repository."""

    try:
        repo = Repository.query.get(repository_id)
    except sqlalchemy.orm.exc.NoResultFound:
        print "No such repository ID!"
        return 1

    r = hubber.client.repository(repo.owner, repo.name)
    issues = []

    for issue in r.iter_issues():
        i = Issue(repository_id=repo.id, title=issue.title,
                number=issue.number, state=issue.state)

        issues.append(i)

    db.session.add_all(issues)

       print "Added {} issues!".format(len(issues))
```

从数据库中获取存储库对象（基于通过 CLI 参数指定的 ID 值），我们调用了我们的`Githubber`扩展的`client.repository()`方法，我们将其导入为`hubber`，这是在工厂序言中分配的名称。由于我们的扩展的一部分负责使用所需的凭据进行初始化，因此我们不需要在调用它的 CLI 工具中处理这个问题。

一旦我们获得了对远程 GitHub 存储库的引用，我们就通过`github3.py`提供的`iter_issues()`方法迭代注册的问题，然后创建`Issue`实例，将其持久化到 SQLAlchemy 会话中。

### 注意

对当前的`Issue`模型的一个受欢迎的改进是在`repository_id`和数字上引入一个复合索引，并使用唯一约束来确保在同一存储库上多次运行前面的命令时不会重复导入问题。

在前面的 CLI 命令中，对重复插入的异常处理也需要发生。实现留给读者作为一个（相对简单的）练习。

这些类型的 CLI 工具非常有用，可以脚本化动作和行为，这些动作和行为在典型的 Web 应用程序的当前用户请求中可能被认为成本太高。你最不希望的是你的应用程序的用户等待几秒，甚至几分钟，以完成一些你几乎无法控制的操作。相反，最好让这些事件在带外发生。实现这一目标的流行方法包括 cron 作业和作业/任务队列，例如 Celery 实现的那些（可能是事件驱动的，而不是按照 cron 作业那样定期运行），等等。

# 摘要

阅读完本章后，您应该对 Flask 扩展和基于命令行的应用程序接口（通过 Flask-Script）的内部工作方式更加熟悉。

我们首先创建了一个简单的应用程序，用于存储在 GitHub 上托管的存储库和问题的数据，然后安装和配置了我们的`manage.py`脚本，以充当 Flask-Script 默认 CLI runserver 和 shell 命令的桥梁。我们添加了`drop_db`和`init_db`全局命令，以替换我们在之前章节中使用的`database.py`脚本。完成后，我们将注意力转向在蓝图中创建子管理器的脚本，我们可以通过主`manage.py`接口脚本进行控制。

最后，我们实现了自己的 Flask 扩展，包装了一些基本配置和资源实例化的`github3.py Github` API 客户端。完成后，我们回到之前创建的子管理脚本，并添加了获取存储在 GitHub 上的给定存储库 ID 的问题列表所需的功能。

在下一章中，我们将深入研究第三方 API，我们将构建一个应用程序，该应用程序使用 OAuth 授权协议，以实现通过 Twitter 和 Facebook 进行用户帐户创建和登录。
