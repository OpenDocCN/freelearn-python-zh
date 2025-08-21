# 第六章：示例 - 构建 BugZot

在过去的几章中，我们已经讨论了许多构建企业级应用程序的技术。但是，如果我们不知道在哪里应用这些知识，那么这些知识有什么用呢？

在本章的过程中，我们将学习构建一个企业级 Web 应用程序的过程，该应用程序将用于跟踪 Omega 公司销售的各种产品的各种利益相关者报告的错误。从现在开始，我们将称之为**BugZot**的系统旨在提供此功能。

该应用程序将使用各种概念构建系统，以便在用户与系统交互的数量增加时能够轻松扩展。我们将看到如何利用各种优化的数据访问和存储技术、高度可扩展的部署和缓存技术来构建一个性能良好的应用程序，即使在高负载情况下也能表现良好。

在本章的过程中，我们将学习以下内容：

+   利用现有的 Web 框架构建企业级 Web 应用程序

+   实现优化数据库访问以加快应用程序速度

+   实施缓存技术以减少应用程序后端的负载

+   利用多线程技术增加应用程序的并发性

+   以可扩展的方式部署应用程序以供生产使用

# 技术要求

本书中的代码清单可以在[`github.com/PacktPublishing/Hands-On-Enterprise-Application-Development-with-Python`](https://github.com/PacktPublishing/Hands-On-Enterprise-Application-Development-with-Python)的`chapter06`目录下找到。

可以通过运行以下命令克隆代码示例：

```py
git clone https://github.com/PacktPublishing/Hands-On-Enterprise-Application-Development-with-Python
```

本章旨在构建一个可扩展的错误跟踪 Web 应用程序。为了实现这一目标，我们使用了许多现有的库和工具，这些库和工具是公开可用的，并经过了长时间的测试，以适应各种用例。构建和运行演示应用程序需要以下一组工具：

+   PostgreSQL 9.6 或更高版本

+   Python 3.6 或更高版本

+   Flask—Python 中的 Web 开发微框架...

# 定义需求

构建任何企业级应用程序的第一步是定义应用程序的目标。到目前为止，我们知道我们的应用程序将跟踪 Omega 公司销售的各种产品的错误。但是，我们的应用程序需要什么功能来进行错误跟踪呢？让我们来看一看，并尝试定义我们将要构建的应用程序的需求。

+   **支持多个产品**：我们的错误跟踪系统的一个基本要求是支持组织构建的多个产品的错误跟踪。考虑到组织的未来增长，这也是一个必需的功能。

+   **支持产品的多个组件**：虽然我们可以在产品级别上报告错误，但这将会太笨重，特别是考虑到大多数组织都有一个专门负责产品正交特性的团队。为了更容易地跟踪基于已提交的组件的错误，错误跟踪系统应支持基于组件的错误报告。

+   **附件支持**：很多时候，提交错误的用户，或者在错误生命周期中以任何方式参与的用户，可能希望附加显示错误效果的图像，或者可能希望附加错误的补丁，以便在合并到产品之前进行测试。这将需要错误跟踪系统提供支持，以将文件附加到错误报告中。

+   **支持评论**：一旦提交了 bug，负责解决该 bug 的用户可能需要有关该 bug 的其他信息，或者可能需要一些协作。这使得缺陷跟踪系统必须支持评论成为必须。此外，并非每条评论都可以公开。例如，如果开发人员可能已经附加到 bug 报告中以供原始提交者测试但尚未纳入主产品的补丁，开发人员可能希望保持补丁私有，以便只有特权访问的人才能看到。这也使得私人评论的功能的包含成为必要。

+   **支持多个用户角色**：组织中并非每个人对缺陷跟踪系统都具有相同级别的访问权限。例如，只有主管级别的人才能向产品添加新组件，只有员工才能看到 bug 的私人评论。这要求系统包含基于角色的访问权限作为要求。

这些是我们的缺陷跟踪系统特定的一些要求。然而，由于这些，显然还有一些其他要求需要包含在系统中。其中一些要求是：

+   **用户认证系统的要求**：系统应该提供一种根据一些简单机制对用户进行认证的机制。例如，用户应该能够通过提供他们的用户名和密码，或者电子邮件和密码组合来登录系统。

+   **用于提交新 bug 的 Web 界面**：应用程序应该提供一个简单易用的 Web 界面，用户可以用来提交新的 bug。

+   **支持 bug 生命周期**：一旦 bug 被提交到系统中，它的生命周期就从 NEW 状态开始。从那里，它可能转移到 ASSIGNED 状态，当组织中的某人接手验证和重现 bug 时。从那里，bug 可以进入各种状态。这被称为我们跟踪系统内的 bug 生命周期。我们的缺陷跟踪系统应该支持这种生命周期，并且应该如何处理当 bug 从一个状态转移到另一个状态。

因此，我们终于把我们的需求摆在了这里。当我们开始设计和定义我们的缺陷跟踪网络应用程序的构建方式时，这些需求将发挥重要作用。因此，有了需求，现在是时候开始定义我们的代码基础是什么样子了。

# 进入开发阶段

随着我们的项目结构定义并就位，现在是时候站起来开始开发我们的应用程序了。开发阶段涉及各种步骤，包括设置开发环境，开发模型，创建与模型相对应的视图，并设置服务器。

# 建立开发环境

在我们开始开发之前的第一步是建立我们的开发环境。这涉及到准备好所需的软件包，并设置环境。

# 建立数据库

我们的 Web 应用程序在管理与用户和已提交的 bug 相关的个人记录方面严重依赖数据库。对于演示应用程序，我们将选择 PostgreSQL 作为我们的数据库。要在基于 RPM 的发行版上安装它，例如 Fedora，需要执行以下命令：

```py
dnf install postgresql postgresql-server postgresql-devel
```

要在 Linux 的任何其他发行版或 Windows 或 Mac OS 等其他操作系统上安装`postgresql`，需要执行分发/操作系统的必需命令。

一旦我们安装了数据库，下一步就是初始化数据库，以便它可以用来存储我们的应用程序数据。用于设置...

# 建立虚拟环境

现在数据库已经就位，让我们设置虚拟环境，这将用于应用程序开发的目的。为了设置虚拟环境，让我们运行以下命令：

```py
virtualenv –python=python3 
```

这个命令将在我们当前的目录中设置一个虚拟环境。设置虚拟环境之后的下一步是安装应用程序开发所需的框架和其他包。

然而，在继续安装所需的包之前，让我们首先通过执行以下命令激活我们的虚拟环境：

```py
source bin/activate
```

作为一个设计决策，我们将基于 Python Flask 微框架进行 Web 应用程序开发。这个框架是一个开源框架，已经存在了相当多年，并且得到了各种插件的支持，这些插件可以很容易地与框架一起安装。该框架也是一个非常轻量级的框架，它只带有最基本的预打包模块，因此允许更小的占用空间。要安装`flask`，执行以下命令：

```py
pip install flask
```

一旦我们安装了 Flask，让我们继续设置我们将在 Web 应用程序开发中使用的其他一些必需的包，通过执行以下命令：

```py
pip install flask-sqlalchemy requests pytest flask-session
```

有了这个，我们现在已经完成了虚拟环境的设置。现在，让我们继续设置我们的代码库将是什么样子。

# 构建我们的项目

现在，我们处于一个需要决定我们的项目结构将是什么样子的阶段。项目结构非常重要，因为它决定了我们代码中不同组件之间的交互方式，以及什么地方将标志着我们应用程序的入口点。

一个结构良好的项目不仅有助于为项目提供更好的导航，而且还有助于提供代码不同部分之间的增强一致性。

所以，让我们来看看我们的代码结构将是什么样子，并理解特定目录或文件的意义：

```py
$ tree --dirsfirst├── bugzot│   ├── helpers│   │   └── __init__.py│   ├── models│   │   └── __init__.py│   ├── static│ ├── templates ...
```

# 初始化 Flask 项目

所以，我们终于进入了项目的有趣阶段，我们将从头开始构建这个项目。所以，让我们不要等太久，我们就可以看到一些行动。我们要做的第一件事是使用 Flask 设置一个基本项目并让它运行起来。为了做到这一点，让我们启动我们的代码编辑器并设置我们的初始代码库。

让我们打开文件`bugzot/application.py`并初始化我们的应用程序代码库：

```py
'''
File: application.py
Description: The file contains the application initialization
             logic that is used to serve the application.
'''
from flask import Flask, session
from flask_bcrypt import Bcrypt
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy

# Initialize our Flask application
app = Flask(__name__, instance_relative_config=True)

# Let's read the configuration
app.config.from_object('config')
app.config.from_pyfile('config.py')

# Let's setup the database
db = SQLAlchemy(app)

# Initializing the security configuration
bcrypt = Bcrypt(app)

# We will require sessions to store user activity across the application
Session(app)
```

现在我们已经完成了应用程序的非常基本的设置。让我们花一些时间来理解我们在这里做了什么。

在文件的开头，我们首先导入了我们将要构建项目的所需包。我们从`flask`包中导入`Flask`应用程序类。类似地，我们导入了代码哈希库`bcrypt`，`Flask`会话类，以及用于 Flask 的 SQLAlchemy 支持包，它提供了与 Flask 的 SQLAlchemy 集成。

一旦我们导入了所有必需的包，下一步就是初始化我们的 Flask 应用程序。为此，我们创建一个`Flask`类的实例，并将其存储在一个名为`app`的对象中。

```py
app = Flask(__name__, instance_relative_config=True)
```

在创建这个实例时，我们向类构造函数传递了两个参数。第一个参数用于表示 Flask 的应用程序名称。`__name__`提供了我们传递给构造函数的应用程序名称。第二个参数`instance_relative_config`允许我们从实例文件夹中覆盖应用程序配置。

有了这个，我们的 Flask 应用程序实例设置就完成了。接下来要做的是加载应用程序的配置，这将用于配置应用程序内部不同组件的行为，以及我们的应用程序将如何提供给用户。为了做到这一点，我们需要从配置文件中读取。以下两行实现了这一点：

```py
app.config.from_object('config')
app.config.from_pyfile('config.py')
```

第一行加载了我们项目根目录下的`config.py`文件，将其视为一个对象，并加载了它的配置。第二行负责读取实例目录下的`config.py`文件，并加载可能存在的任何配置。

一旦这些配置加载完成，它们就可以在`app.config`对象下使用。大多数 Flask 插件都配置为从`app.config`读取配置，因此减少了可能发生的混乱，如果每个插件都有不同的配置处理机制。

在我们的应用程序中加载配置后，我们现在可以继续初始化我们可能需要的其余模块。特别是，我们需要一些额外的模块来建立我们的应用程序功能。这些模块包括 SQLAlchemy 引擎，我们将使用它来构建和与我们的数据库模型交互，一个会话模块，它将被用来管理应用程序中的用户会话，以及一个`bcrypt`模块，它将被用来在整个应用程序中提供加密支持。以下代码提供了这些功能：

```py
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
Session(app)
```

从这些代码行中可以看出，要配置这些模块，我们所需要做的就是将 Flask 应用程序对象作为参数传递给各自的类构造函数，它们的配置将从那里自动获取。

现在，我们已经将应用程序初始化代码放在了适当的位置，我们需要做的下一件事是从我们的 BugZot 模块中导出所需的组件，以便可以从项目根目录调用应用程序。

为了实现这一点，我们需要做的就是将这些模块包含在模块入口点中。所以，让我们打开代码编辑器，打开`bugzot/__init__.py`，我们需要在那里获取这些对象。

```py
'''
File: __init__.py
Description: Bugzot application entrypoint file.
'''
from .application import app, bcrypt, db
```

好了，我们完成了。我们已经在 BugZot 模块中导出了所有必需的对象。现在，问题是如何启动我们的应用程序。因此，为了启动我们的应用程序并使其提供传入的请求，我们需要完成一些更多的步骤。所以，让我们打开项目根目录下的`run.py`文件，并添加以下行：

```py
'''
File: run.py
Description: Bugzot application execution point.
'''
from bugzot import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

好了，是不是很简单？我们在这里所做的就是导入我们在 BugZot 模块中创建的`flask`应用对象，并调用`app`对象的`run`方法，将应用程序将要提供给用户的`hostname`值和应用程序服务器应该绑定以监听请求的端口值传递给它。

我们现在已经准备好启动我们的应用程序服务器，并使其监听传入的请求。但是，在我们这样做之前，我们只需要完成一个步骤，即创建应用程序的配置。所以，让我们开始并创建配置。

# 创建配置

在我们启动应用程序之前，我们需要配置我们将在应用程序中使用的模块。因此，让我们首先打开代码编辑器中的`config.py`，并向其中添加以下内容，以创建我们应用程序的全局配置：

```py
'''File: config.pyDescription: Global configuration for Bugzot project'''DEBUG = FalseSECRET_KEY = 'your_application_secret_key'BCRYPT_LOG_ROUNDS = 5 # Increase this value as required for your applicationSQLALCHEMY_DATABASE_URI = "sqlite:///bugzot.db"SQLALCHEMY_ECHO = FalseSESSION_TYPE = 'filesystem'STATIC_PATH = 'bugzot/static'TEMPLATES_PATH = 'bugzot/templates'
```

有了这些，我们已经起草了全局应用程序配置。让我们尝试...

# 开发数据库模型

数据库模型构成了任何现实生活应用程序的重要部分。这是因为企业中的任何严肃应用程序肯定会处理需要在时间跨度内持久化的某种数据。

对于我们的 BugZot 也是一样的。BugZot 用于跟踪 Omega Corporation 产品中遇到的错误及其生命周期。此外，应用程序还需要记录在其上注册的用户。为了实现这一点，我们将需要多个模型，每个模型都有自己的用途。

为了开发这个应用程序，我们将所有相关的模型分组到它们自己的单独目录下，这样我们就可以清楚地知道每个模型的作用是什么。这也让我们能够保持代码库的整洁，避免开发人员在未来难以理解每个文件的作用。

因此，让我们首先开始开发管理用户账户相关信息所需的模型。

为了开始开发与用户账户相关的模型，我们首先创建一个名为`users`的目录，放在我们的模型目录下：

```py
mkdir bugzot/models/users
```

然后将其初始化为模型模块的子模块。

一旦我们完成了这一点，我们就可以开始创建我们的用户模型，其定义如下所示：

```py
'''
File: users.py
Description: The file contains the definition for the user data model
             that will be used to store the information related to the
             user accounts.
'''
from bugzot.application import db
from .roles import Role

class User(db.Model):
    """User data model for storing user account information.

    The model is responsible for storing the account information on a
    per user basis and providing access to it for authentication
    purposes.
    """

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, index=True, nullable=False)
    password = db.Column(db.String(512), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    user_role = db.Column(db.Integer, db.ForeignKey(Role.id))
    role = db.relationship("Role", lazy=False)
    joining_date = db.Column(db.DateTime, nullable=False)
    last_login = db.Column(db.DateTime, nullable=False)
    account_status= db.Column(db.Boolean, nullable=False, default=False)

    def __repr__(self):
        """User model representation."""
        return "<User {}>".format(self.username)
```

有了这个，我们刚刚创建了我们的用户模型，可以用来存储与用户相关的信息。大多数列只是提供了我们期望存储在数据库中的数据的定义。然而，这里有一些有趣的地方，让我们来看看：

```py
index=True
```

我们可以看到这个属性在用户名和电子邮件列的定义中被提及。我们将索引属性设置为 True，因为这两列经常被用来访问与特定用户相关的数据，因此可以从索引带来的优化中受益。

这里的下一个有趣的信息是与角色模型的关系映射。

```py
role = db.relationship("Role", lazy=False)
```

由于我们数据库中的每个用户都有一个与之关联的角色，我们可以从我们的用户模型到角色模型添加一个一对一的关系映射。此外，如果我们仔细看，我们设置了`lazy=False`。我们之所以要避免懒加载，有一个小原因。角色模型通常很小，而且用户模型到角色模型只有一个一对一的映射。通过避免懒加载，我们节省了一些时间，因为我们的数据库访问层不再懒加载来自角色模型的数据。现在，问题是，角色模型在哪里？

角色模型的定义可以在`bugzot/models/users/roles.py`文件中找到，但我们明确地没有在书中提供该定义，以保持章节简洁。

此外，我们需要一种机制来验证用户的电子邮件地址。我们可以通过发送包含激活链接的小邮件给用户来实现这一点，他们需要点击该链接。为此，我们还需要为每个新用户生成并存储一个激活密钥。为此，我们利用了一个名为`ActivationKey`模型的新模型，其定义可以在`bugzot/models/users/activation_key.py`文件中找到。

一旦所有这些都完成了，我们现在可以准备将这些模型从用户模型子模块中导出。为了做到这一点，让我们打开代码编辑器中的模块入口文件，并通过向`bugzot/models/users/__init__.py`文件添加以下行来导出模型：

```py
from .activation_key import ActivationKey
from .roles import Role
from .users import User
```

有了这个，我们完成了与存储用户信息相关的数据模型的定义。

我们应用程序中的下一件事是定义与产品分类相关的数据模型，用于对可以提交 bug 的产品进行分类。因此，让我们开始创建与产品分类相关的模型。

为了创建与产品相关的模型，我们首先在`bugzot/models`模块下创建一个新的子模块目录并进行初始化。接下来，我们在`bugzot/models/products/products.py`下提供产品模型的定义，如下所示：

```py
'''
File: products.py
Description: The file contains the definition for the products
             that are supported for bug filing inside the bug tracker
'''
from bugzot.application import db
from .categories import Category

class Product(db.Model):
    """Product defintion model.

    The model is used to store the information related to the products
    for which the users can file a bug.
    """

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_name = db.Column(db.String(100), nullable=False, unique=True, index=True)
    category_id = db.Column(db.Integer, db.ForeignKey(Category.id))
    category = db.relationship("Category", lazy=True)

    def __repr__(self):
        """Product model representation."""
        return "<Product {}>".format(self.product_name)
```

有了这个，我们已经完成了产品模型的定义，该模型将用于跟踪产品，用户可以在我们的应用程序中提交 bug。

在我们的产品子模块中还有一些其他模型定义，如下所示：

+   **类别**：类别模型负责存储关于特定产品所属的产品类别的信息

+   **组件**：组件模型负责存储与产品组件相关的信息，其中一个错误可以被归类

+   **版本**：版本模型负责存储与产品版本相关的信息，其中一个错误可以被分类

一旦所有这些模型被定义，它们就可以从产品的子模块中导出，以便在应用程序中使用。

以类似的方式，我们定义了与系统内错误跟踪相关的模型。我们将跳过在本章节中提及这些模型的定义，以保持章节长度合理，但是，对于好奇的人来说，这些模型的定义可以很容易地在代码库中的`bugzot/models/bugs`目录中找到。

# 迁移数据库模型

有了我们创建的数据库模型并准备好使用，下一步是将这些数据库模型迁移到我们用来运行应用程序的数据库服务器。这个过程非常简单。

要将模型迁移到数据库服务器，我们首先将它们暴露到应用程序根目录中。例如，要迁移与用户和产品相关的数据库模型，我们只需要在`bugzot/__init__.py`文件中添加以下行：

```py
from bugzot.models import ActivationKey, Category, Component, Product, Role, User, Version
```

完成后，我们只需要调用我们创建的 SQLAlchemy 数据库对象的`create_all()`方法。这可以通过添加以下...来完成

# 构建视图

一旦模型生成并准备就绪，我们需要的下一步是拥有一种机制，通过该机制我们可以与这些模型进行交互，以便访问或修改它们。我们可以通过视图的使用来实现这种功能的一种方式。

使用 Flask，构建视图是相当容易的任务。Flask Web 框架提供了多种构建视图的方法。事实上，`/ping`端点也可以被称为使用过程式风格构建的视图之一。

在示例过程中，我们现在将尝试在定义应用程序中的任何资源时遵循面向对象的方法。因此，让我们继续并开始开发一些视图。

# 开发索引视图

每当用户访问我们的应用程序时，很可能用户会登陆到应用程序的主页上。因此，我们首先构建的是索引视图。这也是我们可以了解如何在 Flask 中构建简单视图的地方之一。

因此，作为第一步，让我们通过执行以下命令在项目工作空间的视图目录中创建一个新模块，用于索引模块：

```py
mkdir bugzot/views/indextouch bugzot/views/index/__init__.py
```

有了这个，我们现在准备编写我们的第一个视图，其代码如下：

```py
'''File: index.pyDescription: The file provides the definition for the index view             which is used to render the homepage of Bugzot.'''from bugzot.application ...
```

# 获取索引视图以渲染

现在，我们已经准备好了索引视图。但是，在此视图可以提供给用户之前，我们需要为 Flask 提供有关此视图将被渲染的端点的映射。为了实现这一点，让我们打开我们的代码编辑器并打开`bugzot/__init__.py`文件，并向文件中添加以下行：

```py
from bugzot.views import IndexView
app.add_url_rule('/', view_func=IndexView.as_view('index_view'))
```

在这里，我们的重点是第二行，它负责将我们的视图与 URL 端点进行映射。我们的 Flask 应用程序的`add_url_rule()`负责提供这些映射。该方法的第一个参数是视图应该在其上呈现的 URL 路径。提供给该方法的`view_func`参数接受需要在提供的 URL 端点上呈现的视图。 

完成后，我们现在准备好提供我们的索引页面。现在我们只需要运行以下命令：

```py
python run.py
```

然后在浏览器上访问[`localhost:8000/`](http://localhost:8000/)。

# 构建用户注册视图

现在，部署并准备好使用的索引视图，让我们继续构建一个更复杂的视图，在这个视图中，我们允许用户在 BugZot 上注册。

以下代码实现了一个名为`UserRegisterView`的视图，允许用户注册到 BugZot。

```py
'''File: user_registration.pyDescription: The file contains the definition for the user registration             view allowing new users to register to the BugZot.'''from bugzot.application import app, brcypt, dbfrom bugzot.models import User, Rolefrom flask.views import MethodViewfrom datetime import datetimefrom flask import render_template, sessionclass UserRegistrationView(MethodView):    """User registration view to allow new user registration. The user ...
```

# 部署以处理并发访问

到目前为止，我们处于开发阶段，可以轻松使用 Flask 自带的开发服务器快速测试我们的更改。但是，如果您计划在生产环境中运行应用程序，这个开发服务器并不是一个好选择，我们需要更专门的东西。这是因为在生产环境中，我们将更关注应用程序的并发性，以及其安全方面，比如启用 SSL 并为一些端点提供更受限制的访问。

因此，我们需要根据我们的应用需要处理大量并发访问的事实，同时不断保持对用户的良好响应时间，来确定一些选择。

考虑到这一点，我们最终得到了以下一系列选择，它们的性质在许多生产环境中也是相当常见的：

+   **应用服务器**：Gunicorn

+   **反向代理**：Nginx

在这里，Gunicorn 将负责处理由我们的 Flask 应用程序提供的请求，而 Nginx 负责请求排队和处理静态资产的分发。

那么，首先，让我们设置 Gunicorn 以及我们将如何通过它提供应用程序。

# 设置 Gunicorn

设置 Gunicorn 的第一步是安装，这是一个相当简单的任务。我们只需要运行以下命令：

```py
pip install gunicorn
```

一旦完成了这一步，我们就可以运行 Gunicorn 了。Gunicorn 通过**WSGI**运行应用程序，WSGI 代表 Web 服务器网关接口。为了让 Gunicorn 运行我们的应用程序，我们需要在项目工作空间中创建一个名为`wsgi.py`的额外文件，内容如下：

```py
'''File: wsgi.pyDescription: WSGI interface file to run the application through WSGI interface'''from bugzot import appif __name__ == '__main__':    app.run()
```

一旦我们定义了接口文件，我们只需要运行以下命令来使 Gunicorn...

# 设置 Nginx 作为反向代理

要将 Nginx 用作我们的反向代理解决方案，我们首先需要在系统上安装它。对于基于 Fedora 的发行版，可以通过使用`dnf`或`yum`软件包管理器轻松安装，只需运行以下命令：

```py
$ sudo dnf install nginx
```

对于其他发行版，可以使用它们的软件包管理器来安装 Nginx 软件包。

安装 Nginx 软件包后，我们现在需要进行配置，以使其能够与我们的应用服务器进行通信。

要配置 Nginx 将通信代理到我们的应用服务器，创建一个名为`bugzot.conf`的文件，放在`/etc/nginx/conf.d`目录下，内容如下：

```py
server {
    listen 80;
    server_name <your_domain> www.<your_domain>;

    location / {
        include proxy_params;
        proxy_pass http://unix:<path_to_project_folder>/bugzot.sock;
    }
}
```

现在 Nginx 配置完成后，我们需要建立我们的 Gunicorn 应用服务器和 Ngnix 之间的关系。所以，让我们来做吧。

# 建立 Nginx 和 Gunicorn 之间的通信

在我们刚刚完成的 Nginx 配置中需要注意的一点是`proxy_pass`行：

```py
proxy_pass http://unix:<path_to_project_folder>/bugzot.sock
```

这行告诉 Nginx 查找一个套接字文件，通过它 Nginx 可以与应用服务器通信。我们可以告诉 Gunicorn 为我们创建这个代理文件。执行以下命令即可完成：

```py
gunicorn –bind unix:bugzot.sock -m 007 wsgi:app
```

执行此命令后，我们的 Gunicorn Web 服务器将创建一个 Unix 套接字并绑定到它。现在，剩下的就是启动我们的 Nginx Web 服务器，只需执行以下命令即可轻松实现：

```py
systemctl start nginx.service
```

一旦完成了这一步，...

# 总结

在本章中，我们获得了如何开发和托管企业级网络应用程序的实际经验。为了实现这一目标，我们首先做出了一些关于要使用哪些网络框架和数据库的技术决策。然后，我们继续定义我们的项目结构以及它在磁盘上的外观。主要目标是实现高度模块化和代码之间的耦合度较低。一旦项目结构被定义，我们就初始化了一个简单的 Flask 应用程序，并实现了一个路由来检查我们的服务器是否正常工作。然后，我们继续定义我们的模型和视图。一旦这些被定义，我们就修改了我们的应用程序，以启用提供对我们视图的访问的新路由。一旦我们的应用程序开发周期结束，我们就开始了解如何使用 Gunicorn 和 Nginx 部署应用程序以处理大量请求。

现在，当我们进入下一章时，我们将看看如何开发优化的前端，以适应我们正在开发的应用程序，并且前端如何影响用户与我们的应用程序交互时的体验。

# 问题

+   Flask 提供了哪些其他预构建的视图类？

+   我们能否在不删除关系的情况下从用户表中删除到角色表的外键约束？

+   除了 Gunicorn 之外，还有哪些用于提供应用程序的其他选项？

+   我们如何增加 Gunicorn 工作进程的数量？
