# 使用用户认证微服务扩展 TempMessenger

在上一章中，我们创建了一个基于 Web 的信使 TempMessenger，它由两个微服务组成——一个负责存储和检索消息，另一个负责提供 Web 请求。

在本章中，我们将尝试通过用户认证微服务扩展我们现有的 TempMessenger 平台。这将包括一个具有 PostgreSQL 数据库依赖项的 Nameko 服务，该服务具有创建新用户和验证现有用户的能力。

我们还将用一个更合适的 Flask 应用替换我们的 Nameko Web 服务器微服务，这将允许我们跟踪用户的 Web 会话。

有必要阅读上一章才能跟上本章的内容。

我们将涵盖以下主题：

+   创建一个 Postgres 依赖项

+   创建用户服务

+   在数据库中安全存储密码

+   验证用户

+   创建 Flask 应用

+   Web 会话

# TempMessenger 目标

让我们为我们的新的和改进的 TempMessenger 增加一些新目标：

+   用户现在可以注册该应用

+   要发送消息，用户必须登录

+   未登录的用户仍然可以阅读所有消息

如果您在任何时候想要查看本章中的所有代码，请随时在以下网址查看带有测试的完整代码：[`url.marcuspen.com/github-ppb`](http://url.marcuspen.com/github-ppb)。

# 要求

为了在本章中运行，您的本地计算机需要以下内容：

+   互联网连接

+   Docker：如果您尚未安装 Docker，请参阅官方文档：[`url.marcuspen.com/docker-install`](http://url.marcuspen.com/docker-install)

+   运行 Python 3.6 或更高版本的 virtualenv；您可以重用上一章的 virtualenv。

+   pgAdmin：请参阅官方文档以获取安装说明：[`url.marcuspen.com/pgadmin`](http://url.marcuspen.com/pgadmin)

+   运行在默认端口上的 RabbitMQ 容器：这应该是上一章第五章中的内容，*使用微服务构建 Web Messenger*。

+   运行在默认端口上的 Redis 容器：这应该是上一章第五章中的内容，*使用微服务构建 Web Messenger*。

随着我们在本章的学习过程中，所有其他要求都将被安装。

本章中的所有说明都针对 macOS 或 Debian/Ubuntu 系统；但是，我已经努力只使用多平台依赖项。

在本章中，将会有一些代码块。不同类型的代码将有它们自己的前缀，如下所示：

`$`：在您的终端中执行，始终在您的 virtualenv 中

`>>>`：在您的 Nameko/Python shell 中执行

无前缀：要在您的编辑器中使用的 Python 代码块

# 创建一个 Postgres 依赖项

以前，我们想要存储的所有数据都是临时的。消息有固定的生命周期，并且会自动过期；如果我们的应用程序发生灾难性故障，那么最坏的情况就是我们的消息会丢失，对于 TempMessenger 来说几乎没有问题！

然而，用户帐户是完全不同的问题。他们必须被存储，只要用户愿意，他们必须被安全地存储。我们还需要一个适当的模式来保持这些帐户的数据一致。我们还需要能够轻松地查询和更新数据。

因此，Redis 可能不是最佳解决方案。构建微服务的许多好处之一是，我们不会被特定的技术所束缚；仅因为我们的消息服务使用 Redis 进行存储并不意味着我们的用户服务也必须跟随...

# 启动一个 Postgres Docker 容器

首先，在终端中启动一个 Postgres Docker 容器：

```py
$ docker run --name postgres -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=users -p 5432:5432 -d postgres
```

这将启动一个带有一些基本设置的 Postgres 容器：

+   `--name`设置容器的名称

+   -e 允许我们设置环境变量：

+   `POSTGRES_PASSWORD`：用于访问数据库的密码

+   `POSTGRES_DB`：数据库的名称

+   -p 允许我们将容器上的端口`5432`暴露到本地机器上的端口`5432`

+   -d 允许我们以守护程序模式启动容器（在后台运行）

如果您正在为生产环境创建数据库，则设置更安全的密码并将其保密是非常重要的！

您可以通过执行以下操作并确保您的`postgres`容器存在来检查容器是否正在运行：

```py
$ docker ps
```

# 创建用户模型

为了在 Postgres 中存储关于我们用户的数据，我们首先需要创建一个模型，该模型将定义我们要存储的字段和数据类型。

我们首先需要安装两个新的 Python 包：SQLAlchemy 和 Psycopg。SQLAlchemy 是一个工具包和对象关系映射器，将作为我们进入 SQL 世界的入口。Psycopg 是 Python 的 PostgreSQL 数据库适配器。

首先将`sqlalchemy`（*在撰写本文时为 1.2.1 版本*）和`psycopg2`（*在撰写本文时为 2.7.4 版本*）添加到您的`base.in`文件中。从项目文件夹的根目录，在您的虚拟环境中运行：

```py
$ pip-compile requirements/base.in
$ pip-sync requirements/base.txt requirements/test.txt
```

这将向我们的要求中添加`sqlalchemy`和`psycopg2`，并确保我们的虚拟环境包与它们完全匹配。或者，如果您选择不使用 pip-tools，也可以使用`pip install`它们。

在我们的依赖文件夹中，创建一个新文件`users.py`。通常，您会为您的数据库模型有一个不同的文件，但为了简单起见，我们将它嵌入到我们的依赖中。首先，让我们定义我们的导入和我们的模型将使用的基类：

```py
from sqlalchemy import Column, Integer, Unicode 
from sqlalchemy.ext.declarative import declarative_base 

Base = declarative_base() 
```

我们首先导入`Column`，它将用于声明我们的数据库列，以及一些基本字段类型：`Integer`和`Unicode`。至于`declarative_base`，我们使用它来创建我们的`Base`类，从而我们的用户模型将继承自它。这将创建我们的模型与数据库表之间的映射。

现在让我们为我们的`users`定义一个基本模型：

```py
class User(Base): 
    __tablename__ = 'users' 

    id = Column(Integer, primary_key=True) 
    first_name = Column(Unicode(length=128)) 
    last_name = Column(Unicode(length=128)) 
    email = Column(Unicode(length=256), unique=True) 
    password = Column(Unicode(length=512)) 
```

正如您所看到的，我们的`User`类继承自我们之前定义的`Base`类。`__tablename__`设置表的名称。让我们简要地回顾一下我们定义的一些数据库列：

+   `id`：我们数据库中每个用户的唯一标识符和主键。对于简单起见，数据库模型通常将其 ID 设置为整数。

+   `first_name`和`last_name`：128 个字符的最大长度对于任何名称应该足够了。我们还使用`Unicode`作为我们的类型，以适应包含诸如中文之类的符号的名称。

+   `email`：同样，一个大的字段长度和`Unicode`来适应符号。我们还使这个字段是唯一的，这将防止创建具有相同电子邮件地址的多个用户。

+   `password`：我们不会在这里以明文存储密码；我们稍后会回到这个问题！

要了解更多关于 SQLAlchemy 的信息，请参阅[`url.marcuspen.com/sqlalchemy`](http://url.marcuspen.com/sqlalchemy)。

# 创建用户依赖项

现在我们已经定义了一个基本的用户模型，我们可以为其创建一个 Nameko 依赖项。幸运的是，`nameko-sqlalchemy`已经为我们做了一些工作，这是一个开源的 Nameko 依赖项，它将处理围绕数据库会话的所有语义，并为我们提供一些非常有用的 Pyest 固定装置进行测试。

通过将其添加到`requirements/base.in`文件中安装`nameko-sqlalchemy`（*在撰写本文时为 1.0.0 版本*），并按照之前的相同步骤安装`sqlalchemy`。

现在我们将创建一个包装器类，用于封装管理用户的所有逻辑。在`users.py`中，添加以下代码：

```py
class UserWrapper: 

    def __init__(self, session): 
        self.session = session 
```

这将是我们包装器的基础，并且将需要一个数据库会话对象，形式为`session`。稍后，我们将向这个类添加更多的方法，比如`create`和`authenticate`。为了创建我们的用户依赖项，首先让我们将以下内容添加到我们的导入中：

```py
from nameko_sqlalchemy import DatabaseSession 
```

现在让我们创建一个新的类，User Store，它将作为我们的依赖：

```py
class UserStore(DatabaseSession): 

    def __init__(self): 
        super().__init__(Base) 

    def get_dependency(self, worker_ctx): 
        database_session = super().get_dependency(worker_ctx) 
        return UserWrapper(session=database_session) 
```

解释这段代码，首先让我们谈谈`DatabaseSession`。这是一个为 Nameko 预先制作的依赖提供者，由`nameko-sqlalchemy`提供给我们，已经包括了`setup`和`get_dependency`等方法，就像上一章介绍的那样。因此，我们的`UserStore`类只是继承它来使用这个现有的功能。

`DatabaseSession`类的`__init__`方法以我们的模型的声明性基础作为它唯一的参数。在我们的`UserStore`类中，我们用我们自己的`__init__`方法覆盖了这个方法，它修改为使用我们的`Base`并执行与原始功能相同的功能，使用 Python 内置的`super`函数。

要了解更多关于 Python 的`super`方法，请参见：[`url.marcuspen.com/python-super`](http://url.marcuspen.com/python-super)。

`DatabaseSession`类中的原始`get_dependency`方法只是返回一个数据库会话；然而，我们希望我们的方法返回一个`UserWrapper`的实例，这样我们就可以轻松调用后面将要创建的`create`和`authenticate`方法。为了以一种优雅的方式覆盖它，以便我们仍然保留生成数据库会话的所有逻辑，我们再次使用`super`函数来生成`database_session`并返回我们的`UserWrapper`的实例。

# 创建用户

现在我们已经有了 Nameko 的依赖，我们可以开始为我们的`UserWrapper`添加功能。我们将从创建用户开始。将以下内容添加到`UserWrapper`类中：

```py
def create(self, **kwargs): 
    user = User(**kwargs) 
    self.session.add(user) 
    self.session.commit() 
```

这个`create`方法将创建一个新的`User`对象，将其添加到我们的数据库会话中，提交到数据库的更改，并返回用户。这里没有什么花哨的！但让我们谈谈`self.session.add`和`self.session.commit`的过程。当我们首次将用户添加到会话中时，这将把用户添加到我们本地数据库会话中的内存中，而不是将它们添加到我们的实际数据库中。新用户已经被暂存，但实际上并没有在我们的数据库中进行任何更改。这是非常有用的。假设我们想对数据库进行多次更新，多次调用数据库可能会很昂贵，所以我们首先在内存中进行所有想要的更改，然后使用单个数据库事务`commit`它们。

在前面的代码中你会注意到的另一件事是，我们使用了`**kwargs`而不是定义实际的参数来创建一个新的`User`。如果我们要更改用户模型，这样可以最小化所需的更改，因为关键字参数直接映射到字段。

# 创建用户服务

在上一章中，我们只是在同一个模块中有两个服务，这对于任何小规模项目来说都是可以的。然而，现在我们的平台开始增长，服务之间定义了新的角色，让我们开始通过将它们放在不同的模块中来拆分它们。在你的`service.py`旁边，创建一个新文件，`user_service.py`。

添加以下代码：

```py
from nameko.rpc import rpc 
from .dependencies.users import UserStore 

class UserService: 

    name = 'user_service' 
    user_store = UserStore() 

    @rpc 
    def create_user(self, first_name, last_name, email, password): 
        self.user_store.create( 
            first_name=first_name, 
            last_name=last_name, 
            email=email, 
            password=password, 
        ) 
```

如果你读过上一章，那么这里没有什么新的。我们创建了一个新的`UserService`，给它了`UserStore`依赖，并进行了一个 RPC，这只是一个对依赖的`create`方法的透传。然而，在这里我们选择定义创建用户的参数，而不像我们在依赖方法中使用`**kwargs`。这是因为我们希望 RPC 定义它与其他服务的契约。如果另一个服务发出无效调用，我们希望 RPC 尽快拒绝它，而不是浪费时间去调用依赖，或者更糟糕的是进行数据库查询。

我们已经接近可以测试这个功能的点了，但首先我们需要更新我们的`config.yaml`文件，加入我们的数据库设置。如果你使用了之前提供的命令来创建一个 Docker Postgres 容器，追加以下内容：

```py
DB_URIS: 
  user_service:Base: 
    "postgresql+psycopg2://postgres:secret@localhost/ 
    users?client_encoding=utf8" 
```

`DB_URIS`被`nameko-sqlalchemy`用于将 Nameko 服务和声明性基础对映射到 Postgres 数据库。

我们还需要在我们的 Postgres 数据库中创建表。通常情况下，您可以使用数据库迁移工具（如 Alembic）来完成这项工作。但是，为了本书的目的，我们将使用一个小的一次性 Python 脚本来为我们完成这项工作。在项目目录的根目录中，创建一个名为`setup_db.py`的新文件，其中包含以下代码：

```py
from sqlalchemy import create_engine 
from temp_messenger.dependencies.users import User 

engine = create_engine( 
    'postgresql+psycopg2://postgres:secret@localhost/' 
    'users?client_encoding=utf8' 
) 
User.metadata.create_all(engine) 
```

此代码使用我们依赖模块中的用户模型，并为我们在数据库中创建所需的表。`create_engine`是起点，因为它建立了与数据库的连接。然后我们使用我们的用户模型`metadata`（在我们的情况下包括表名和列）并调用`create_all`，它使用`engine`向数据库发出`CREATE` SQL 语句。

如果您要对用户模型进行更改并保留现有用户数据，那么学习如何使用数据库迁移工具（如 Alembic）是必不可少的，我强烈推荐这样做。

要了解有关如何使用 Alembic 的更多信息，请参阅[`url.marcuspen.com/alembic`](http://url.marcuspen.com/alembic)。

要运行，请在您的虚拟环境中的终端中执行：

```py
$ python setup_db.py
```

现在让我们使用数据库管理工具来查看我们的新表。有许多数据库管理工具，我个人最喜欢的是 Mac 上的 Postico，但是为了本书的目的，我们将使用适用于所有平台的 pgAdmin。

从[`url.marcuspen.com/pgadmin`](http://url.marcuspen.com/pgadmin)下载并安装 pgAdmin。安装完成后，打开并选择“添加新服务器”，将会弹出以下窗口：

![](img/dd0efeea-239c-4f78-8e41-92e76e614c9a.png)

在“常规”选项卡中简单地给它一个您选择的名称，然后在“连接”选项卡中，您可以填写我们的数据库详细信息，这些详细信息是我们在之前创建 Postgres Docker 截图时设置的。但是，如果您没有对此进行任何更改，您可以简单地复制前面图像中的详细信息。请记住，密码设置为`secret`。填写完毕后，点击“保存”，它应该连接到我们的数据库。

连接后，我们可以开始查看我们数据库的详细信息。要查看我们的表，您需要展开并操作菜单，就像这样：

![](img/5f1adc41-40c7-4259-924f-9d9de30657a8.png)

现在您应该能够看到我们的表，它代表了我们的用户模型：

![](img/01d0e8bb-817f-4859-b394-84dd16821d1b.png)

现在我们可以尝试使用 Nameko shell 创建一个用户。通过在项目文件夹的根目录中，在虚拟环境中执行以下命令，在终端中启动我们的新用户服务：

```py
$ nameko run temp_messenger.user_service --config config.yaml
```

在另一个终端窗口中，在您的虚拟环境中执行：

```py
$ nameko shell
```

在 Nameko shell 中，执行以下命令以创建新用户：

```py
>>> n.rpc.user_service.create_user('John', 'Doe', 'john@example.com', 'super-secret')
```

现在让我们检查 pgAdmin，看看用户是否成功创建。要刷新数据，只需按照之前的步骤显示用户表或单击“刷新”按钮即可：

![](img/f26eb87a-f620-4449-bb3d-aee5f70aaa38.png)

成功了！我们现在有一个可以创建新用户的功能性用户服务。但是，这里有一个主要问题...我们刚刚犯了软件开发人员可以犯的最严重的错误之一——以明文形式存储密码！

# 在数据库中安全地存储密码

现在是 2018 年，到目前为止，我们可能已经听过数十个关于公司泄露我们的敏感数据，包括密码，给黑客的故事。在许多情况下，泄露的密码存储的加密非常差，这意味着它们可以轻松破解。在某些情况下，密码甚至以明文形式存储！

无论如何，这种疏忽导致了数百万用户的电子邮件和密码组合泄漏。如果我们为每个在线账户使用不同的密码，这可能不是一个问题...但不幸的是，我们很懒，密码重用是相当普遍的做法。因此，减轻黑客入侵我们服务器造成的一些损害的责任落在我们开发人员身上。

2016 年 10 月，流行的视频分享平台 Dailymotion 遭遇了一次数据泄露，其中有 8500 万个帐户被盗。在这 8500 万个帐户中，有 1800 万个帐户附带了密码，但幸运的是它们是使用 Bcrypt 进行散列的。这意味着黑客需要几十年，甚至几个世纪的暴力计算才能用今天的硬件破解它们（来源：[`url.marcuspen.com/dailymotion-hack`](http://url.marcuspen.com/dailymotion-hack)）。

因此，尽管黑客成功侵入了 Dailymotion 的服务器，但通过使用散列算法（如 Bcrypt）存储密码，部分损害得到了缓解。考虑到这一点，我们现在将看看如何为我们的用户密码实现`bcrypt`散列，而不是以明文方式存储它们。

# 使用 Bcrypt

首先将`bcrypt`添加到您的`base.in`文件中，并使用与之前相同的过程安装它（*在撰写本文时为 3.1.4 版本*）。

如果您在安装 Bcrypt 时遇到问题，请参阅它们的安装说明，其中包括有关系统软件包依赖项的详细信息：[`url.marcuspen.com/pypi-bcrypt`](http://url.marcuspen.com/pypi-bcrypt)。

为了`bcrypt`创建密码的散列，它需要两样东西——您的密码和一个`salt`。`salt`只是一串随机字符。让我们看看如何在 Python 中创建`salt`：

```py
>>> from bcrypt import gensalt
>>> gensalt()
b'$2b$12$fiDoHXkWx6WMOuIfOG4Gku'
```

这是创建与 Bcrypt 兼容的`salt`的最简单方法。`$`符号代表`salt`的不同部分，我想指出第二部分：`$12`。这部分表示生成密码散列所需的工作轮次，默认为`12`。我们可以这样配置：

```py
>>> gensalt(rounds=14)
b'$2b$14$kOUKDC.05iq1ANZPgBXxYO'
```

注意这个`salt`，它已经改变成`$14`。通过增加这个值，我们也增加了创建密码的散列所需的时间。这也会增加后来检查密码尝试与散列的时间。这是有用的，因为我们试图阻止黑客在设法获取我们的数据库后对密码尝试进行暴力破解。然而，默认的轮次`12`已经足够了！现在让我们创建一个密码的散列：

```py
>>> from bcrypt import hashpw, gensalt
>>> my_password = b'super-secret'
>>> salt = gensalt()
>>> salt
b'$2b$12$YCnmXxOcs/GJVTHinSoVs.'
>>> hashpw(my_password, salt)
b'$2b$12$YCnmXxOcs/GJVTHinSoVs.43v/.RVKXQSdOhHffiGNk2nMgKweR4u'
```

在这里，我们只是使用默认数量的轮次生成了一个新的`salt`，并使用`hashpw`生成了散列。注意我们的密码的`salt`也在散列的第一部分中？这非常方便，因为这意味着我们不必单独存储`salt`，这在以后验证用户时会需要。

由于我们使用了默认数量的轮次来生成`salt`，为什么不尝试设置自己的轮次？请注意，设置的轮次越高，`hashpw`所需的时间就越长。当轮次设置为 20 时，我的机器花了将近 2 分钟来创建一个散列！

现在让我们看看如何检查密码与散列相匹配：

```py
>>> from bcrypt import hashpw, checkpw, gensalt
>>> my_password = b'super-secret'
>>> salt = gensalt()
>>> hashed_password = hashpw(my_password, salt)
>>> password_attempt = b'super-secret'
>>> checkpw(password_attempt, hashed_password)
True
```

正如你所看到的，`checkpw`接受我们正在检查的密码尝试和散列密码作为参数。当我们在我们的依赖项中实现这一点时，密码尝试将是来自 Web 请求的部分，散列密码将存储在数据库中。由于这是一个成功的尝试，`checkpw`返回`True`。让我们尝试使用一个无效的密码进行相同的操作：

```py
>>> password_attempt = b'invalid-password'
>>> checkpw(password_attempt, hashed_password)
False
```

毫不奇怪！它返回了`False`。

如果您想了解更多关于存储密码和某些方法的缺陷的信息，我建议您阅读 Dustin Boswell 的这篇简短文章：[`url.marcuspen.com/dustwell-passwords`](http://url.marcuspen.com/dustwell-passwords)。它很好地解释了黑客如何尝试使用暴力破解和彩虹表来破解密码。它还更详细地介绍了 Bcrypt。

# 散列我们的用户密码

现在我们知道如何更安全地存储密码了，让我们修改我们的`create`方法，在将密码存储到数据库之前对其进行哈希处理。首先，在我们的`users.py`依赖文件的顶部，让我们将`bcrypt`添加到我们的导入中，并添加一个新的常量：

```py
import bcrypt 

HASH_WORK_FACTOR = 15 
```

我们的新常量`HASH_WORK_FACTOR`将用于`gensalt`使用的轮次参数。我把它设置为 15，这将导致创建密码哈希和检查密码需要花费更长的时间，但会更安全。请随意设置，但请记住，增加这个值会导致我们的应用在以后创建和验证用户时需要更长的时间。

现在，在任何类之外，我们将定义一个新的辅助函数来哈希密码：

```py
def hash_password(plain_text_password): 
    salt = bcrypt.gensalt(rounds=HASH_WORK_FACTOR) 
    encoded_password = plain_text_password.encode() 

    return bcrypt.hashpw(encoded_password, salt) 
```

这个辅助函数简单地获取我们的明文密码，生成一个`salt`，并返回一个哈希密码。现在，您可能已经注意到，当使用 Bcrypt 时，我们总是必须确保我们给它的密码是字节串。正如您从前面的代码中注意到的那样，我们必须在将密码（默认为 UTF-8）传递给`hashpw`之前对其进行`.encode()`处理。Bcrypt 还将以字节串格式返回哈希密码。这将带来的问题是，我们数据库中密码的字段当前设置为 Unicode，与我们的密码不兼容。我们有两个选择：要么在存储密码之前调用`.decode()`，要么修改我们的密码字段为可以接受字节串的类型，比如`LargeBinary`。让我们选择后者，因为它更清晰，可以避免我们每次访问数据时都需要转换数据。

首先，让我们修改导入字段类型的行，包括`LargeBinary`：

```py
from sqlalchemy import Column, Integer, LargeBinary, Unicode 
```

现在我们可以更新我们的`User`模型来使用我们的新字段类型：

```py
class User(Base): 
    __tablename__ = 'users' 

    id = Column(Integer, primary_key=True) 
    first_name = Column(Unicode(length=128)) 
    last_name = Column(Unicode(length=128)) 
    email = Column(Unicode(length=256), unique=True) 
    password = Column(LargeBinary()) 
```

我们现在唯一的问题是我们现有的数据库与我们的新模式不兼容。为了解决这个问题，我们可以删除数据库表或执行迁移。在现实世界的环境中，删除整个表是绝对不可取的！如果您已经采纳了我之前的建议学习 Alembic，那么我鼓励您将您的知识付诸实践，并执行数据库迁移。但出于本书的目的，我将利用一次性的 Docker 容器并从头开始。为此，在您的项目根目录和虚拟环境内执行：

```py
$ docker rm -f postgres
$ docker run --name postgres -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=users -p 5432:5432 -d postgres
$ python setup_db.py
```

这将删除您现有的 Postgres 容器，创建一个新的容器，并运行我们之前制作的`setup_db.py`脚本。如果您检查 pgAdmin，您现在会看到密码列标题中的字段类型已从`character varying (512)`更改为`bytea`。

最后，我们现在准备更新我们的`create`方法来使用我们的新的`hash_password`函数：

```py
def create(self, **kwargs): 
    plain_text_password = kwargs['password'] 
    hashed_password = hash_password(plain_text_password) 
    kwargs.update(password=hashed_password) 

    user = User(**kwargs) 
    self.session.add(user) 
    self.session.commit() 
```

正如您在方法的前三行中所看到的：

1.  从`kwargs`中提取`plain_text_password`。

1.  调用`hash_password`来创建我们的`hashed_password`。

1.  对`kwargs`执行更新，以用哈希版本替换密码。

代码的其余部分与我们之前的版本相同。

让我们试一试。在您的虚拟环境中的终端中，启动（或重新启动）用户服务：

```py
$ nameko run temp_messenger.user_service --config config.yaml
```

在您的虚拟环境中的另一个终端窗口中，启动您的 Nameko shell：

```py
$ nameko shell
```

在您的 Nameko shell 中，执行以下操作再次添加新用户：

```py
>>> n.rpc.user_service.create_user('John', 'Doe', 'john@example.com', 'super-secret')
```

您应该注意到（取决于您设置的`HASH_WORK_FACTOR`有多大），与上次创建新用户相比，现在会有一些延迟。

现在您应该在 pgAdmin 中看到以下内容：

![](img/eaddd97d-b99e-4dfd-b087-1933233f9805.png)

# 处理重复用户

由于我们将 email 字段设置为唯一，我们的数据库已经阻止了重复的用户。但是，如果您自己尝试，您会发现返回的输出并不理想。尝试在 Nameko shell 中再次添加相同的用户。

另一个问题是，如果在创建新用户时出现任何其他错误，我们的外部服务没有很好的方式来对这些不同类型的错误做出反应，而不知道我们正在使用的数据库类型，这是我们要尽一切努力避免的。

为了解决这个问题，让我们首先在我们的`users.py`中创建两个新的异常类：

```py
class CreateUserError(Exception): 
    pass 

class UserAlreadyExists(CreateUserError): 
    pass 
```

我们还需要更新我们的导入，包括`IntegrityError`，这是 SQLAlchemy 在唯一键违规时引发的错误类型：

```py
from sqlalchemy.exc import IntegrityError 
```

同样，我们将修改我们的`create`方法，这次使用我们的两个新异常：

```py
def create(self, **kwargs): 
    plain_text_password = kwargs['password'] 
    hashed_password = hash_password(plain_text_password) 
    kwargs.update(password=hashed_password) 

    user = User(**kwargs) 
    self.session.add(user) 

    try: 
        self.session.commit() # ① 
    except IntegrityError as err: 
        self.session.rollback() # ② 
        error_message = err.args[0] # ③ 

        if 'already exists' in error_message: 
            email = kwargs['email'] 
            message = 'User already exists - {}'.format(email) 
            raise UserAlreadyExists(message) # ④ 
        else: 
            raise CreateUserError(error_message) # ⑤ 
```

我们在这里所做的是：

1.  将`self.session.commit()`包装在 try except 块中。

1.  如果发生`IntegrityError`，回滚我们的会话，这将从我们的数据库会话中删除用户-在这种情况下并不完全必要，但无论如何都是一个好的做法。

1.  提取错误消息。

1.  检查它是否包含字符串`'already exists'`。如果是，那么我们知道用户已经存在，我们引发适当的异常`UserAlreadyExists`，并给它一个包含用户电子邮件的错误消息。

1.  如果不是，那么我们有一个意外的错误，并引发更通用的错误，适合我们的服务，`CreateUserError`，并给出整个错误消息。

通过这样做，我们的外部服务现在将能够区分用户错误和意外错误。

为了测试这一点，重新启动用户服务，并尝试在 Nameko shell 中再次添加相同的用户。

# 验证用户

现在我们可以看看如何验证用户。这是一个非常简单的过程：

1.  从数据库中检索我们要验证的用户。

1.  执行`bcrypt.checkpw`，给出尝试的密码和用户的密码哈希。

1.  如果结果是`False`，则引发异常。

1.  如果是`True`，则返回用户。

# 从数据库中检索用户

从第一点开始，我们需要添加一个新的依赖方法`get`，如果存在，则返回用户的电子邮件。

首先，在`users.py`中添加一个新的异常类：

```py
class UserNotFound(Exception): 
    pass 
```

这是我们在用户找不到时会引发的。现在我们将更新我们的导入，包括以下内容：

```py
from sqlalchemy.orm.exc import NoResultFound 
```

`NoResultFound`，顾名思义，是 SQLAlchemy 在数据库中找不到请求的对象时引发的。现在我们可以为我们的`UserWrapper`类添加一个新方法：

```py
def get(self, email): 
    query = self.session.query(User) # ① 

    try: 
        user = query.filter_by(email=email).one() # ② 
    except NoResultFound: 
        message = 'User not found - {}'.format(email) 
        raise UserNotFound(message) # ③ 

    return user 
```

让我们了解一下我们在前面的代码中做了什么：

1.  为了查询我们的数据库，我们首先必须使用我们的用户模型作为参数来创建一个查询对象。

1.  一旦我们有了这个，我们可以使用`filter_by`并指定一些参数；在这种情况下，我们只想按电子邮件过滤。`filter_by`总是返回一个可迭代对象，因为可能有多个结果，但由于我们在电子邮件字段上有一个唯一的约束，所以可以安全地假设如果存在，我们只会有一个匹配。因此，我们调用`.one()`，它返回单个对象，如果过滤器为空，则引发`NoResultFound`。

1.  我们处理`NoResultFound`并引发我们自己的异常`UserNotFound`，并附上错误消息，这更适合我们的用户服务。

# 验证用户的密码

我们现在将实现一个`authenticate`方法，该方法将使用我们刚刚创建的`get`方法。

首先，让我们创建一个新的异常类，如果密码不匹配，将引发该异常：

```py
class AuthenticationError(Exception): 
    pass 
```

我们现在可以为我们的`UserWrapper`创建另一个方法来验证用户：

```py
def authenticate(self, email, password): 
    user = self.get(email) # ① 

    if not bcrypt.checkpw(password.encode(), user.password): # ② 
        message = 'Incorrect password for {}'.format(email) 
        raise AuthenticationError(message) # ③ 
```

1.  我们首先使用我们最近创建的`get`方法从数据库中检索我们要验证的用户。

1.  然后，我们使用`bcrypt.checkpw`来检查尝试的密码是否与从数据库中检索的用户对象上存储的密码匹配。我们在密码尝试上调用`.encode()`，因为我们的外部服务不会为我们执行此操作。它也不应该；这是 Bcrypt 特有的逻辑，这样的逻辑应该留在依赖项中。

1.  如果密码不正确，我们会引发`AuthenticationError`错误，并附上适当的消息。

现在剩下的就是在`user_service.py`中的`UserService`类上创建一个 RPC：

```py
@rpc 
def authenticate_user(self, email, password): 
    self.user_store.authenticate(email, password) 
```

这里没有什么特别的，只是一个简单的传递到我们刚刚创建的`user_store`依赖方法。

让我们测试一下。重新启动`user_service`，并在您的 Nameko shell 中执行以下操作：

```py
>>> n.rpc.user_service.authenticate_user('john@example.com', 'super-secret')
>>>
```

如果成功，它应该什么都不做！现在让我们尝试使用错误的密码：

```py
>>> n.rpc.user_service.authenticate_user('john@example.com', 'wrong')
Traceback (most recent call last):
...
nameko.exceptions.RemoteError: PasswordMismatch Incorrect password for john@example.com
>>>
```

就是这样！这结束了我们对用户服务的工作。现在我们将看看如何将其与我们现有的服务集成。

如果您想看一下如何为用户服务编写一些测试，您会在本章开头提到的 Github 存储库中找到它们以及所有代码。

# 拆分服务

目前，我们在同一个`service.py`模块中有我们的`MessageServer`和`WebServer`。现在是时候拆分它们了，特别是因为我们将删除`WebServer`，转而使用 Flask 服务器。在本章结束时，目标是有三个微服务共同工作，每个都有自己特定的角色：

![](img/57a9715d-61eb-4d8d-b15c-2dd353b3f03e.jpg)

上图显示了我们的服务将如何相互集成。请注意消息服务和用户服务是完全不知道彼此的。对用户服务的更改不应该需要对消息服务进行更改，反之亦然。通过拆分这些服务，我们还获得了能够在不影响其他服务的情况下部署新代码的优势。Nameko 使用 RabbitMQ 的一个额外好处是，如果一个服务短暂下线，任何工作将被排队，直到服务重新上线。我们现在将开始收获微服务架构的一些好处。

要开始这个重构，让我们在`temp_messenger`文件夹中创建一个新文件，名为`message_service.py`：

```py
from nameko.rpc import rpc 
from .dependencies.messages import MessageStore 

class MessageService: 

    name = 'message_service' 

    message_store = MessageStore() 

    @rpc 
    def get_message(self, message_id): 
        return self.message_store.get_message(message_id) 

    @rpc 
    def save_message(self, message): 
        message_id = self.message_store.save_message(message) 
        return message_id 

    @rpc 
    def get_all_messages(self): 
        messages = self.message_store.get_all_messages() 
        sorted_messages = sort_messages_by_expiry(messages) 
        return sorted_messages 

def sort_messages_by_expiry(messages, reverse=False): 
    return sorted( 
        messages, 
        key=lambda message: message['expires_in'], 
        reverse=reverse 
    ) 
```

我们在这里所做的就是从旧的`service.py`中取出`MessageService`和所有相关代码，放入我们的新的`message_service.py`模块中。

# 创建 Flask 服务器

我们现在将创建一个新的 Flask Web 服务器，它将取代我们的 Nameko Web 服务器。Flask 更适合处理 Web 请求，而且内置功能更多，同时还相当轻量级。我们将利用其中的一个功能，即会话，它将允许我们的服务器跟踪谁已登录。它还与 Jinja2 一起使用模板，这意味着我们现有的模板应该已经可以工作。

首先在我们的`base.in`文件中添加`flask`，然后使用与之前相同的过程`pip-compile`和安装（*在撰写本文时为 0.12.2 版本*）。

开始使用 Flask 非常简单；我们将从创建新的主页端点开始。在您的`temp_messenger`目录中，创建一个名为`web_server.py`的新文件，内容如下：

```py
from flask import Flask, render_template # ① 

app = Flask(__name__) # ② 

@app.route('/') # ③ 
def home(): 
    return render_template('home.html') # ④ 
```

1.  我们从`flask`中导入以下内容：

+   `Flask`：用于创建我们的 Flask 应用对象

+   `render_template`：渲染给定的模板文件

1.  创建我们的`app`，唯一的参数是从`__name__`派生的模块名称。

1.  `@app.route`允许您使用 URL 端点装饰一个函数。

有了这个，我们将能够启动我们的新 Flask Web 服务器，尽管没有功能。要测试这一点，首先导出一些环境变量：

```py
$ export FLASK_DEBUG=1
$ export FLASK_APP=temp_messenger/web_server.py
```

第一个将设置应用程序为调试模式，这是我喜欢的一个功能，因为当我们更新代码时，它将热重载，不像 Nameko 服务。第二个简单地告诉 Flask 我们的应用程序在哪里。

在启动 Flask 应用程序之前，请确保您当前没有运行旧的 Nameko Web 服务器，因为这将导致端口冲突。

在您的虚拟环境中，在项目的根目录中执行以下命令以启动服务器：

```py
$ flask run -h 0.0.0.0 -p 8000
```

这将在端口`8000`上启动 Flask 服务器，与我们以前的 Nameko web 服务器运行的端口相同。只要您的本地网络允许，甚至可以让同一网络上的其他设备导航到您的机器 IP 并使用 TempMessenger！现在在浏览器中转到`http://127.0.0.1:8000`，您应该看到以下内容（尽管没有功能）：

![](img/b97a2733-ea93-46ee-9911-6cc0bd049e39.png)

看起来与我们以前的类似，对吧？那是因为 Flask 已经使用 Jinja2 作为其默认的模板引擎，所以如果我们愿意，我们可以删除我们旧的`jinja2.py`依赖，因为它不再需要了。Flask 还会在与应用程序相同的目录中查找一个名为`templates`的文件夹，这就是它自动知道在哪里找到`home.html`的方式。

现在让我们添加从我们的消息服务中检索消息的功能。这与我们在两个 Nameko 服务之间通信时略有不同，因为 Flask 不知道如何执行 RPC。首先，让我们添加以下内容到我们的导入中：

```py
from flask.views import MethodView 
from nameko.standalone.rpc import ClusterRpcProxy 
from flask.json import jsonify 
```

我们还需要添加一些配置，以便 Flask 知道在哪里找到我们的 RabbitMQ 服务器。我们可以将其添加到我们的模块中作为一个常量，但由于我们已经在`config.yaml`中有`AMQP_URI`，所以没有必要重复！在我们的`web_server.py`模块中，在`app = Flask(__name__)`之前，添加以下内容：

```py
import yaml 
with open('config.yaml', 'r') as config_file: 
    config = yaml.load(config_file) 
```

这将从`config.yaml`中加载所有的配置变量。现在将以下类添加到`web_server.py`中：

```py
class MessageAPI(MethodView): 

    def get(self): 
        with ClusterRpcProxy(config) as rpc: 
            messages = rpc.message_service.get_all_messages() 

        return jsonify(messages) 
```

而我们的主页端点有一个基于函数的视图，这里我们有一个基于类的视图。我们定义了一个`get`方法，它将用于对这个`MessageAPI`的任何`GET`请求。请注意，这里方法的名称很重要，因为它们映射到它们各自的请求类型。如果我们要添加一个`post`方法（我们稍后会添加），那么它将映射到`MessageAPI`上的所有`POST`请求。

`ClusterRpcProxy`允许我们在 Nameko 服务之外进行 RPC。它被用作上下文管理器，并允许我们轻松调用我们的消息服务。Flask 带有一个方便的辅助函数`jsonify`，它将我们的消息列表转换为 JSON。然后简单地返回该有效负载，Flask 会为我们处理响应头和状态码。

现在让我们添加发送新消息的功能。首先，修改你的 flask 导入以包括请求：

```py
from flask import Flask, render_template, request 
```

现在在`MessageAPI`类中添加一个新的 post 方法：

```py
def post(self): # ① 
    data = request.get_json(force=True) # ② 

    try: 
        message = data['message'] # ③ 
    except KeyError: 
        return 'No message given', 400 

    with ClusterRpcProxy(config) as rpc: # ④ 
        rpc.message_service.save_message(message) 

    return '', 204 # ⑤ 
```

1.  您可能会注意到，与我们在 Nameko web 服务器中使用`post`参数获取`request`对象的方式不同，我们是从 Flask 中导入它的。在这种情况下，它是一个全局对象，为我们解析所有传入的请求数据。

1.  我们使用`get_json`，这是一个内置的 JSON 解析器，将替换我们上一章的`get_request_data`函数。我们指定`force=True`，这将强制要求请求具有有效的 JSON 数据；否则它将返回`400 Bad Request`错误代码。

1.  与我们旧的`post_message`HTTP 端点一样，我们尝试获取`data['message']`，否则返回`400`。

1.  然后我们再次使用`ClusterRpcProxy`进行 RPC 以保存消息。

1.  如果一切顺利，返回`204`。我们在这里使用`204`而不是`200`来表示，虽然请求仍然成功，但没有要返回的内容。

在这之前，我们还需要做一件事，那就是注册我们的`MessageAPI`到一个 API 端点。在我们的`web_server.py`的底部，在`MessageAPI`类之外，添加以下内容：

```py
app.add_url_rule( 
    '/messages', view_func=MessageAPI.as_view('messages') 
) 
```

这将把任何请求重定向到`/messages`到`MessageAPI`。

现在是时候重新启动我们的消息服务了。在一个新的终端窗口中，在您的虚拟环境中执行：

```py
$ nameko run temp_messenger.message_service --config config.yaml
```

由于我们现在有多个服务，这需要在不同的终端窗口中运行多个实例。如果您的 Nameko 服务在您发出请求时关闭，这可能会导致功能无限期地挂起，直到该服务再次上线。这是 Nameko 使用消息队列来消耗新任务的一个副作用；任务只是在队列中等待服务接收。

假设您的 Flask 服务器仍在运行，现在您应该能够访问我们的应用程序，以前的所有功能都在`http://127.0.0.1:8000`上！

![](img/b4105130-c7d8-4265-904c-77168f84e27c.png)

# Web 会话

现在我们通过新的 Flask 服务器恢复了旧的功能，我们可以开始添加一些新功能，比如登录和注销用户，创建新用户，并且只允许已登录的用户发送消息。所有这些都严重依赖于 Web 会话。

Web 会话允许我们通过 cookie 在不同的请求之间跟踪用户。在这些 cookie 中，我们存储可以从一个请求传递到下一个请求的信息。例如，我们可以存储用户是否经过身份验证，他们的电子邮件地址是什么，等等。这些 cookie 使用一个密钥进行加密签名，我们需要在使用 Flask 的会话之前定义它。在`config.yaml`中，添加以下内容：

```py
FLASK_SECRET_KEY: 'my-super-secret-flask-key' 
```

随意设置您自己的密钥，这只是一个例子。在类似生产环境中，这必须保持安全和安全，否则用户可以伪造自己的会话 cookie。

现在我们需要告诉我们的`app`使用这个密钥。在`app = Flask(__name__)`之后添加以下内容：

```py
app.secret_key = config['FLASK_SECRET_KEY'] 
```

完成后，Flask 现在将使用我们在`config.yaml`中的`FLASK_SECRET_KEY`来签署 cookie。

# 创建注册页面

我们将通过为新用户添加注册功能来开始这些新功能。在`web_server.py`中，添加以下新类：

```py
class SignUpView(MethodView): 

    def get(self): 
        return render_template('sign_up.html') 
```

这个新的`SignUpView`类将负责处理注册过程。我们添加了一个 get 方法，它将简单地渲染我们稍后将创建的`sign_up.html`模板。

在`web_server.py`模块的末尾，创建以下 URL 规则：

```py
app.add_url_rule( 
    '/sign_up', view_func=SignUpView.as_view('sign_up') 
) 
```

正如您可能已经知道的，这将把所有请求重定向到`/sign_up`到我们的新`SignUpView`类。

现在让我们创建我们的新模板。在`templates`文件夹中，创建一个新文件，`sign_up.html`：

```py
<!DOCTYPE html> 
<body> 
  <h1>Sign up</h1> 
  <form action="/sign_up" method="post"> 
    <input type="text" name="first_name" placeholder="First Name"> 
    <input type="text" name="last_name" placeholder="Last Name"> 
    <input type="text" name="email" placeholder="Email"> 
    <input type="password" name="password" placeholder="Password"> 
    <input type="submit" value="Submit"> 
  </form> 
  {% if error_message %} 
    <p>{{ error_message }}</p> 
  {% endif %} 
</body> 
```

这是一个基本的 HTML 表单，包括在我们的数据库中创建新用户所需的字段。`action`和`method`表单告诉它向`/sign_up`端点发出`post`请求。所有字段都是`text`字段，除了密码，它是`password`类型，这将导致用户输入被掩盖。我们还有一个 Jinja `if`语句，它将检查模板是否渲染了`error_message`。如果是，那么它将显示在段落块中。我们稍后将使用这个来向用户显示消息，比如“用户已存在”。

做出这些更改后，假设您的 Flask 服务器仍在运行，请导航到`http://127.0.0.1:8000/sign_up`，您应该看到新的注册页面：

![](img/043bf8b2-c94c-429f-8581-b4f28366ad6b.png)

这个表格目前还没有任何作用，因为我们还没有为`SignUpView`定义一个 post 方法。让我们继续创建。首先，在`web_server.py`中更新我们的导入，包括从 Nameko 导入`RemoteError`，从 Flask 导入`session`，`redirect`和`url_for`：

```py
from nameko.exceptions import RemoteError 
from flask import ( 
    Flask, 
    Redirect, 
    render_template, 
    request, 
    session, 
    url_for, 
) 
```

在您的`SignUpView`类中，添加以下`post`方法：

```py
def post(self): 
    first_name = request.form['first_name'] # ① 
    last_name = request.form['last_name'] 
    email = request.form['email'] 
    password = request.form['password'] 

    with ClusterRpcProxy(config) as cluster_rpc: 
        try: 
            cluster_rpc.user_service.create_user( # ② 
                first_name=first_name, 
                last_name=last_name, 
                email=email, 
                password=password, 
            ) 
        except RemoteError as err: # ③ 
            message = 'Unable to create user {} - {}'.format( 
                err.value 
            ) 
            app.logger.error(message) 
            return render_template( 
                'sign_up.html', error_message=message 
            ) 

    session['authenticated'] = True # ④ 
    session['email'] = email # ⑤ 

    return redirect(url_for('home')) # ⑥ 
```

这是一个相当长的方法，但它非常简单。让我们一步一步地来看：

1.  我们首先从`request.form`中检索用户的所有相关字段。

1.  然后我们使用`ClusterRpcProxy`向我们的`user_service`发出`create_user` RPC。

1.  如果发生错误，通过以下方式处理：

+   构建错误消息

+   使用 Flask 的`app.logger`将该消息记录到控制台

+   使用错误消息渲染`sign_up.html`模板

1.  如果没有错误，那么我们继续向`session`对象添加一个`True`的`authenticated`布尔值。

1.  将用户的电子邮件添加到`session`对象中。

1.  最后，我们使用`url_for`重定向用户，它将寻找名为`home`的函数端点。

在测试之前，如果您还没有运行用户服务，请在虚拟环境中的新终端中执行：

```py
nameko run temp_messenger.user_service --config config.yaml 
```

有了这个，现在您应该同时在不同的终端窗口中运行用户服务、消息服务和 Flask Web 服务器。如果没有，请使用之前的`nameko`和`flask`命令启动它们。

转到`http://127.0.0.1:8000/sign_up`，尝试创建一个新用户：

![](img/280e4a06-0dea-45cd-97b3-c29bf62fd2ec.png)

一旦您点击提交，它应该将您重定向到主页，并且您的数据库中应该有一个新用户。检查 pgAdmin 以确保它们已经被创建。

现在返回`http://127.0.0.1:8000/sign_up`，尝试再次添加相同的用户。它应该让您保持在同一个页面上并显示错误消息：

![](img/1921cfef-7c6e-4eb9-8a32-5f82bb38520c.png)

拥有注册页面是很好的，但是我们的用户需要能够在不知道 URL 的情况下导航到它！让我们对`home.html`进行一些调整，添加一个简单的注册链接。与此同时，我们还可以隐藏发送新消息的功能，除非他们已登录！在我们的`home.html`中，修改现有的`postMessage`表单如下：

```py
{% if authenticated %} 
  <form action="/messages" id="postMessage"> 
    <input type="text" name="message" placeholder="Post message"> 
    <input type="submit" value="Post"> 
  </form> 
{% else %} 
  <p><a href="/sign_up">Sign up</a></p> 
{% endif %} 
```

我们在这里所做的是将我们的表单包装在 Jinja 的`if`块中。如果用户经过身份验证，那么我们将显示`postMessage`表单；否则，我们将显示一个链接，引导用户转到注册页面。

现在我们还需要更新我们的主页端点，将`session`对象中的`authenticated`布尔值传递给模板渲染器。首先，让我们添加一个新的帮助函数，用于获取用户的认证状态。这应该位于`web_server.py`模块中任何类之外：

```py
def user_authenticated(): 
    return session.get('authenticated', False) 
```

这将尝试从`session`对象中获取`authenticated`布尔值。如果它是一个全新的`session`，那么我们不能保证`authenticated`会在那里，所以我们将其默认为`False`并返回它。

在`web_server.py`中，更新`home`端点如下：

```py
@app.route('/') 
def home(): 
    authenticated = user_authenticated() 
    return render_template( 
        'home.html', authenticated=authenticated 
    ) 
```

这将调用`user_authenticated`来获取我们用户的`authenticated`布尔值。然后我们通过传递`authenticated`来渲染模板。

我们可以做的另一个不错的调整是，只有在用户未经过身份验证时才允许其转到注册页面。为此，我们需要更新`SignUpView`中的`get`方法如下：

```py
def get(self): 
    if user_authenticated(): 
        return redirect(url_for('home')) 
    else: 
        return render_template(sign_up.html') 
```

如果我们经过身份验证，那么我们将用户重定向到`home`端点；否则，我们渲染`sign_up.html`模板。

如果您仍然打开了用于创建第一个用户的浏览器，那么如果您尝试导航到`http://127.0.0.1:8000/sign_up`，它应该将您重定向到我们网站的主页，因为您已经经过身份验证。

如果您打开一个不同的浏览器，在主页上，您应该看到我们制作的新的注册链接，发送新消息的功能应该已经消失，因为您有一个新的会话。

我们现在有一个新问题。我们已经阻止了用户从应用程序发送新消息，但是如果他们使用 Curl 或 REST 客户端，他们仍然可以发送消息。为了阻止这种情况发生，我们需要对`MessageAPI`进行一点小调整。在`MessageAPI`的 post 方法开头添加以下内容：

```py
def post(self): 
    if not user_authenticated() 
        return 'Please log in', 401 
    ... 
```

确保不要调整任何其他代码；`...`表示我们`post`方法的其余代码。这将简单地拒绝用户的请求，并使用`401`响应告诉用户登录。

# 登出用户

现在我们需要实现用户注销的功能。在`web_server.py`中，添加以下`logout`函数端点：

```py
@app.route('/logout') 
def logout(): 
    session.clear() 
    return redirect(url_for('home')) 
```

如果用户访问此端点，Flask 将清除他们的`session`对象并将其重定向到`home`端点。由于会话已清除，`authenticated`布尔值将被删除。

在`home.html`中，让我们更新我们的页面，包括用户注销的链接。为此，我们将在`postMessage`表单之后添加一个新链接：

```py
{% if authenticated %} 
  <form action="/messages" id="postMessage"> 
    <input type="text" name="message" placeholder="Post message"> 
    <input type="submit" value="Post"> 
  </form> 
  <p><a href="/logout">Logout</a></p> 
... 
```

保存后，只要我们已登录，现在我们应该在消息表单下面有一个注销链接：

![](img/1bbcc2f0-51be-485b-8af0-9abb6cb5f4e3.png)

点击注销链接后，您将被重定向回主页，您将无法再发送消息。

# 记录用户登录

我们的应用程序如果没有用户登录的能力就不完整！在我们的`web_server.py`中，创建一个新的类`LoginView`：

```py
class LoginView(MethodView): 

    def get(self): 
        if user_authenticated(): 
            return redirect(url_for('home')) 
        else: 
            return render_template('login.html') 
```

与我们的`SignUpView`中的 get 方法类似，这个方法将检查用户是否已经`authenticated`。如果是，则将重定向到`home`端点，否则，将呈现`login.html`模板。

在我们的`web_server.py`模块的末尾，添加以下 URL 规则以使用`LoginView`：

```py
app.add_url_rule( 
    '/login', view_func=LoginView.as_view('login') 
) 
```

任何对`/login`的请求现在都将被重定向到我们的`LoginView`。

现在在我们的模板文件夹中创建一个新模板`login.html`：

```py
<!DOCTYPE html> 
<body> 
  <h1>Login</h1> 
  <form action="/login" method='post'> 
    <input type="text" name="email" placeholder="Email"> 
    <input type="password" name="password" placeholder="Password"> 
    <input type="submit" value="Post"> 
  </form> 
  {% if login_error %} 
    <p>Bad log in</p> 
  {% endif %} 
</body> 
```

正如您所看到的，这与我们的`sign_up.html`模板非常相似。我们创建一个表单，但这次我们只有`email`和`password`字段。我们还有一个 Jinja 的`if`块用于错误消息。但是，这个错误消息是硬编码的，而不是从`LoginView`返回的。这是因为告诉用户登录失败的原因是不好的做法。如果是恶意用户，我们告诉他们诸如*此用户不存在*或*密码不正确*之类的东西，那么这就足以告诉他们我们数据库中存在哪些用户，他们可能会尝试暴力破解密码。

在我们的`home.html`模板中，让我们还添加一个用户登录的链接。为此，我们将在`if authenticated`块的`else`语句中添加一个新链接：

```py
{% if authenticated %} 
... 
{% else %} 
  <p><a href="/login">Login</a></p> 
  <p><a href="/sign_up">Sign up</a></p> 
{% endif %} 
```

现在我们应该能够从主页导航到登录页面：

![](img/23037fa1-de70-4735-8b4a-0894a259f1f1.png)

为了使我们的登录页面工作，我们需要在我们的`LoginView`中创建一个`post`方法。将以下内容添加到`LoginView`中：

```py
def post(self): 
    email = request.form['email'] # ① 
    password = request.form['password'] 

    with ClusterRpcProxy(config) as cluster_rpc: 
        try: 
            cluster_rpc.user_service.authenticate_user( # ② 
                email=email, 
                password=password, 
            ) 
        except RemoteError as err: # ③ 
            app.logger.error( 
                'Bad login for %s - %s', email, str(err) 
            ) 
            return render_template( 
                'login.html', login_error=True 
            ) 

    session['authenticated'] = True # ④ 
    session['email'] = email # ⑤ 

    return redirect(url_for('home')) # ⑥ 
```

您会注意到这与我们的`SignUpView`中的 post 方法非常相似。让我们简要地了解一下正在发生的事情：

1.  我们从`request.form`中检索电子邮件和密码。

1.  我们使用`ClusterRpcProxy`向`user_service`发出`authenticate_user` RPC。

1.  如果发生`RemoteError`，那么我们：

+   使用 Flask 的`app.logger`将错误记录到控制台

+   使用`login_error`设置为`True`呈现`login.html`模板

1.  如果他们成功验证，我们将在`session`对象中将`authenticated`设置为`True`。

1.  将电子邮件设置为`session`对象中的用户`email`。

1.  将用户重定向到`home`端点。

通过上述代码，我们选择将错误消息记录到只有我们可以看到的控制台，而不是将错误消息返回给用户。这使我们能够查看我们的身份验证系统是否存在任何问题，或者恶意用户是否在做坏事，同时仍然让用户知道他们提供了无效的信息。

我们的服务仍在运行，现在您应该能够测试它了！我们现在为 TempMessenger 拥有一个完全运作的身份验证系统，我们的目标已经实现。

# 在我们的消息中添加电子邮件前缀

我们的 TempMessenger 缺少的一个重要功能是问责制。我们不知道哪些用户发布了什么，对于一个匿名的消息应用来说这是可以接受的（如果这是您想要的话，那么可以跳过这一部分）。为了做到这一点，当我们存储我们的消息时，我们还希望存储发送者的电子邮件。

让我们首先重新审视`messages.py`的依赖关系。将我们`RedisClient`中的`save_message`更新为以下内容：

```py
def save_message(self, email, message): 
    message_id = uuid4().hex 
    payload = { 
        'email': email, 
        'message': message, 
    } 
    self.redis.hmset(message_id, payload) 
    self.redis.pexpire(message_id, MESSAGE_LIFETIME) 

    return message_id 
```

您会注意到的第一件事是，为了调用`save_message`，我们现在需要用户的电子邮件。

我们在这里所做的另一件事是将我们在 Redis 中存储的数据格式从字符串更改为哈希。Redis 哈希允许我们将类似字典的对象存储为值。它们还有一个额外的好处，就是可以选择以后从字典中获取哪个键，而不是获取整个对象。

在这里，我们创建了用户电子邮件和密码的字典，并使用`hmset`将其存储在 Redis 中。`hmset`没有`px`或`ex`参数，所以我们调用`pexpire`，它会在给定的毫秒数后使给定的键过期。还有一个相当于秒的`expire`。

要了解有关 Redis 哈希和其他数据类型的更多信息，请参阅：[`url.marcuspen.com/redis-data-types`](http://url.marcuspen.com/redis-data-types)。

现在我们将更新`RedisClient`中的`get_all_messages`方法如下：

```py
def get_all_messages(self): 
    return [ 
        { 
            'id': message_id, 
            'email': self.redis.hget(message_id, 'email'), 
            'message': self.redis.hget(message_id, 'message'), 
            'expires_in': self.redis.pttl(message_id), 
        } 
        for message_id in self.redis.keys() 
    ] 
```

由于数据已更改为哈希，我们还必须以不同的方式从 Redis 中检索数据，使用`hget`方法。我们还获取与每条消息对应的电子邮件。

现在我们将继续进行`message_service.py`。在`MessageService`中，将`save_message` RPC 更新为以下内容：

```py
@rpc 
def save_message(self, email, message): 
    message_id = self.message_store.save_message( 
        email, message 
    ) 
    return message_id 
```

我们所做的只是更新 RPC 的参数，包括`email`并将其传递给更新后的`message_store.save_message`。

回到我们的`web_server.py`，我们需要更新`MessageAPI`的 post 方法，在调用`MessageService`时发送用户的电子邮件：

```py
def post(self): 
    if not user_authenticated(): 
        return 'Please log in', 401 

    email = session['email'] # ① 
    data = request.get_json(force=True) 

    try: 
        message = data['message'] 
    except KeyError: 
        return 'No message given', 400 

    with ClusterRpcProxy(config) as rpc: 
        rpc.message_service.save_message(email, message) # ② 

    return '', 204 
```

我们刚刚做了两个小改动：

1.  从`session`对象中获取`email`。

1.  更新 RPC 以传递`email`。

为了在我们的页面上看到这些更改，我们还需要更新`home.html`模板。对于我们的 JavaScript 函数`updateMessages`，将其更新为以下内容：

```py
function updateMessages(messages) { 
  var $messageContainer = $('#messageContainer'); 
  var messageList = []; 
  var emptyMessages = '<p>No messages!</p>'; 

  if (messages.length === 0) { 
    $messageContainer.html(emptyMessages); 
  } else { 
    $.each(messages, function(index, value) { 
      var message = $(value.message).text() || value.message; 
      messageList.push( 
        '<p>' + value.email + ': ' + message + '</p>' 
      ); 
    }); 
    $messageContainer.html(messageList); 
  } 
} 
```

这只是一个小调整。如果你没注意到，我们已经更新了`messageList.push`以包括`email`。

在测试之前，请确保您的 Redis 存储为空，因为旧消息将以旧格式存在，这将破坏我们的应用程序。您可以通过在我们的 Redis 容器内使用`redis-cli`来做到这一点：

```py
$ docker exec -it redis /bin/bash
$ redis-cli -h redis
redis:6379> flushall
OK
redis:6379>
```

还要确保重新启动我们的消息服务，以使新更改生效。一旦你做到了，我们就可以测试这个新功能：

![](img/5069299d-f391-4c07-9735-07bb57c96d62.png)

# 总结

这就结束了我们对 TempMessenger 用户认证系统的工作。我们从本章开始使用 Python 和 Postgres 数据库，并创建了一个 Nameko 依赖项来封装它。这与上一章的 Redis 依赖项不同，因为数据是永久的，需要更多的规划。尽管如此，我们将这个逻辑隐藏起来，并简单地暴露了两个 RPC：`create_user`和`authenticate_user`。

然后，我们研究了如何在数据库中安全存储用户密码。我们探讨了一些错误的存储密码的方式，比如以明文存储密码。我们使用 Bcrypt 对我们的密码进行加密哈希，以防止在数据库受到损害时被读取。

当涉及将新的用户服务链接到我们应用程序的其他部分时，我们首先将每个服务拆分为自己的模块，以便我们可以独立部署、更新和管理它们。通过展示如何在 Web 服务器中轻松替换一个框架（Nameko）为另一个框架（Flask），我们获得了微服务架构的一些好处，而不会影响平台的其他部分。

我们探索了 Flask 框架以及如何创建基于函数和基于类的视图。我们还研究了 Flask 会话对象以及如何从一个请求到下一个存储用户数据。

作为奖励，我们修改了消息列表，还包括发送者的电子邮件地址。

我鼓励你考虑为 TempMessenger 制定新的增强功能，并相应地计划如何添加它们，确保我们的依赖逻辑不会泄漏到属于它的服务之外——这是许多人犯的错误！保持我们的服务边界定义清晰是一项艰巨的任务，有时候从更单片的方式开始，等清晰之后再将它们分离出来会有所帮助。这与我们在上一章中对`MessageService`和`WebServer`采取的方法类似。Sam Newman 的《构建微服务》（O'Reilly）很好地解释了这一点，并更详细地介绍了构建分布式系统所涉及的好处、缺点和挑战。

完成了这一章，我希望我已经让你更深入地了解了如何在实践中从微服务架构中受益。我们创建这个应用程序的过程是有意模块化的，不仅反映了微服务的模块化，而且演示了我们应该如何在对平台的影响最小的情况下添加新功能。
