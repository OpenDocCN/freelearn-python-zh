

# 第五章：连接到关系数据库

我们之前的应用程序仅使用 Python 集合来存储数据记录，而不是持久数据存储。这种设置会在 **Uvicorn** 服务器重启时导致数据丢失，因为这些集合仅在 *易失性内存*，如 **RAM** 中存储数据。从本章开始，我们将应用数据持久性以避免数据丢失，并提供一个平台来管理我们的记录，即使在服务器关闭模式下也是如此。

本章将重点介绍不同的 **对象关系映射器**（**ORMs**），这些 ORM 可以有效地使用对象和关系数据库来管理客户端数据。对象关系映射是一种技术，其中 SQL 语句用于 **创建**、**读取**、**更新** 和 **删除**（**CRUD**）在面向对象的编程方法中实现和执行。ORM 需要将所有关系或表映射到相应的实体或模型类，以避免与数据库平台的紧密耦合连接。这些模型类是用于连接到数据库的类。

除了介绍 ORM，本章还将讨论一个名为 **命令和查询责任分离**（**CQRS**）的设计模式，该模式可以帮助在域级别解决读写 ORM 事务之间的冲突。CQRS 可以帮助最小化读写 SQL 事务的运行时间，与数据建模方法相比，随着时间的推移可以提高应用程序的整体性能。

总体而言，本章的主要目标是证明 FastAPI 框架支持所有流行的 ORM，为应用程序提供后端数据库访问，它通过使用流行的关系数据库管理系统来实现，并通过使用 CQRS 设计模式对 CRUD 事务进行优化。

在本章中，我们将涵盖以下主题：

+   准备数据库连接

+   使用 *SQLAlchemy* 创建同步 CRUD 事务

+   使用 *SQLAlchemy* 实现异步 CRUD 事务

+   使用 *GINO* 进行异步 CRUD 事务

+   使用 Pony ORM 进行存储库层

+   使用 Peewee 构建存储库

+   应用 CQRS 设计模式

# 技术要求

为本章创建的应用程序原型被称为 *健身俱乐部管理系统*；它服务于会员和健身房健身运营。此原型具有管理、会员、课程管理和出勤模块，这些模块利用 `ch05a` 和 `ch05b` 项目。

# 准备数据库连接

在我们开始讨论 FastAPI 中的数据库连接之前，让我们考虑一些与应用程序相关的问题：

+   首先，从本章开始，所有应用程序原型都将使用 PostgreSQL 作为唯一的关联数据库管理系统。我们可以从 [`www.enterprisedb.com/downloads/postgres-postgresql-downloads`](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads) 下载其安装程序。

+   其次，*健身俱乐部管理系统* 原型有一个名为 `fcms` 的现有数据库，包含六个表，分别是 `signup`、`login`、`profile_members`、`profile_trainers`、`attendance_member` 和 `gym_class`。所有这些表及其元数据和关系都可以在以下图中看到：

![图 5.1 – fcms 表格](img/Figure_5.1_B17975.jpg)

图 5.1 – fcms 表格

项目文件夹中包含一个名为 `fcms_postgres.sql` 的脚本，用于安装所有这些模式。

现在我们已经安装了最新版本的 PostgreSQL 并运行了 `fcms` 脚本文件，让我们来了解 SQLAlchemy，这是 Python 场景中最广泛使用的 ORM 库。

重要提示

本章将比较和对比不同 Python ORM 的功能。在这个实验设置中，每个项目都将拥有多种数据库连接，这与每个项目只有一个数据库连接的惯例相悖。

# 使用 SQLAlchemy 创建 CRUD 事务

**SQLAlchemy** 是最受欢迎的 ORM 库，能够建立任何基于 Python 的应用程序和数据库平台之间的通信。它之所以可靠，是因为它持续更新和测试，以确保其 SQL 读写操作高效、高性能和准确。

这个 ORM 是一个模板化接口，旨在创建一个数据库无关的数据层，可以连接到任何数据库引擎。但与其他 ORM 相比，SQLAlchemy 更受数据库管理员（DBA）的欢迎，因为它可以生成优化的原生 SQL 语句。在制定其查询时，它只需要 Python 函数和表达式来执行 CRUD 操作。

在我们开始使用 SQLAlchemy 之前，请使用以下命令检查您系统中是否已安装该模块：

```py
pip list 
```

如果 SQLAlchemy 不在列表中，请使用 `pip` 命令进行安装：

```py
pip install SQLAlchemy
```

目前，开发 *健身俱乐部管理系统* 应用程序所使用的版本是 *1.4*。

## 安装数据库驱动程序

SQLAlchemy 没有所需的数据库驱动程序将无法工作。由于选择的是 PostgreSQL 数据库，因此必须安装 `psycopg2` 方言：

```py
pip install psycopg2
```

Psycopg2 是一个符合 *DB API 2.0-c*ompliant 的 PostgreSQL 驱动程序，可以进行连接池管理，并且可以与多线程 FastAPI 应用程序一起工作。这个包装器或方言对于构建我们的应用程序的同步 CRUD 事务也是必不可少的。一旦安装，我们就可以开始查看 SQLAlchemy 的数据库配置细节。所有与 SQLAlchemy 相关的代码都可以在 `ch05a` 项目中找到。

## 设置数据库连接

为了连接到任何数据库，SQLAlchemy 需要一个管理连接池和已安装方言的引擎。`create_engine()`函数来自`sqlalchemy`模块，是引擎对象的来源。但为了成功派生它，`create_engine()`需要一个配置好的数据库 URL 字符串。这个 URL 字符串包含*数据库名称*、*数据库 API 驱动程序*、*账户凭证*、数据库服务器的*IP 地址*和其*端口*。以下脚本展示了如何创建将在我们的*健身俱乐部管理系统*原型中使用的引擎：

```py
from sqlalchemy import create_engine
DB_URL =   
   "postgresql://postgres:admin2255@localhost:5433/fcms"
engine = create_engine(DB_URL)
```

`engine`是一个全局对象，必须在整个应用程序中只创建一次。它的第一个数据库连接发生在应用程序的第一个 SQL 事务之后，因为它遵循了*延迟初始化*设计模式。

此外，前一个脚本中的`engine`对于创建将被 SQLAlchemy 用于执行 CRUD 事务的 ORM 会话至关重要。

## 初始化会话工厂

SQLAlchemy 中的所有 CRUD 事务都是由*会话*驱动的。每个会话管理一组数据库“写入”和“读取”，并检查是否执行它们。例如，它维护一组已插入、更新和删除的对象，检查更改是否有效，然后与 SQLAlchemy 核心协调，如果所有事务都已验证，则将更改推进到数据库。它遵循*工作单元*设计模式的行为。SQLAlchemy 依赖于会话来保证数据的一致性和完整性。

但在我们创建会话之前，数据层需要一个绑定到派生引擎的会话工厂。ORM 从`sqlalchemy.orm`模块中有一个`sessionmaker()`指令，它需要一个`engine`对象。以下脚本展示了如何调用`sessionmaker()`：

```py
from sqlalchemy.orm import sessionmaker
engine = create_engine(DB_URL)
SessionFactory = sessionmaker(autocommit=False, 
                     autoflush=False, bind=engine)
```

除了引擎绑定之外，我们还需要将会话的`autocommit`属性设置为`False`以强制执行`commit()`和`rollback()`事务。应用程序应该是将所有更改刷新到数据库的那个，因此我们还需要将其`autoflush`功能设置为`False`。

应用程序可以通过`SessionFactory()`调用创建多个会话，但每个`APIRouter`拥有一个会话是推荐的。

## 定义 Base 类

接下来，我们需要设置`Base`类，这在将模型类映射到数据库表中至关重要。尽管 SQLAlchemy 可以在运行时创建表，但我们选择利用现有的模式定义来构建我们的原型。现在，这个`Base`类必须由模型类继承，以便在服务器启动时发生到表的映射。以下脚本展示了设置此组件是多么简单：

```py
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
```

调用`declarative_base()`函数是创建`Base`类而不是创建`registry()`来调用`generate_base()`的最简单方法，这也可以为我们提供`Base`类。

注意，所有这些配置都是原型 `/db_config/sqlalchemy_connect.py` 模块的一部分。由于它们在构建 SQLAlchemy 存储库中至关重要，因此它们被捆绑在一个模块中。但在我们实现 CRUD 事务之前，我们需要使用 `Base` 类创建模型层。

## 构建模型层

SQLAlchemy 的模型类已放置在健身俱乐部项目文件夹的 `/models/data/sqlalchemy_models.py` 文件中。如果 `BaseModel` 对 API 请求模型很重要，那么 `Base` 类在构建数据层时是必不可少的。它从配置文件导入以定义 SQLAlchemy 实体或模型。以下代码来自模块脚本，展示了我们如何在 SQLAlchemy ORM 中创建模型类定义：

```py
from sqlalchemy import Time, Boolean, Column, Integer, 
    String, Float, Date, ForeignKey
from sqlalchemy.orm import relationship
from db_config.sqlalchemy_connect import Base
class Signup(Base):
    __tablename__ = "signup"
    id = Column(Integer, primary_key=True, index=True)
    username = Column('username', String, unique=False, 
                       index=False)
    password = Column('password' ,String, unique=False, 
                       index=False)
```

`Signup` 类是 SQLAlchemy 模型的示例，因为它继承了 `Base` 类的属性。它是一个映射类，因为所有其属性都是其物理表模式对应项的列元数据的反映。该模型将 `primary_key` 属性设置为 `True`，因为 SQLAlchemy 建议每个表模式至少有一个主键。其余的 `Column` 对象映射到非主键但可以是 *唯一* 或 *索引* 的列元数据。每个模型类继承 `__tablename__` 属性，该属性设置映射表的名称。

最重要的是，我们需要确保类属性的数据类型与其在表模式中列对应的列类型相匹配。列属性必须与列对应项具有相同的名称。否则，我们需要在 `Column` 类的第一个参数中指定实际的列名，如 `Signup` 中的 `username` 和 `password` 列所示。但大多数情况下，我们必须始终确保它们相同，以避免混淆。

### 映射表关系

SQLAlchemy 强烈支持不同类型的父子或关联表关系。参与关系的模型类需要使用 `sqlalchemy.orm` 模块中的 `relationship()` 指令来利用模型类之间的一对多或一对一关系。此指令通过表模式定义中指示的外键从父类创建对子类的引用。

一个子模型类使用其外键列对象中的 `ForeignKey` 构造来将其模型类与其父类的参考键列对象链接。此指令表示该列中的值应包含在父表参考列中存储的值内。`ForeignKey` 指令适用于主键和非主键的 `Column` 对象。以下模型类定义了我们数据库模式中的一个示例列关系：

```py
class Login(Base): 
    __tablename__ = "login"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=False, index=False)
    password = Column(String, unique=False, index=False)
    date_approved = Column(Date, unique=False, index=False)
    user_type = Column(Integer, unique=False, index=False)

    trainers = relationship('Profile_Trainers', 
         back_populates="login", uselist=False)
    members = relationship('Profile_Members', 
         back_populates="login", uselist=False)
```

此 `Login` 模型根据其配置与两个子类 `Profile_Trainers` 和 `Profile_Members` 链接。这两个子模型在其 `id` 列对象中都有 `ForeignKey` 指令，如下面的模型定义所示：

```py
class Profile_Trainers(Base):
    __tablename__ = "profile_trainers"
    id = Column(Integer, ForeignKey('login.id'), 
         primary_key=True, index=True, )
    firstname = Column(String, unique=False, index=False)
    … … … … …
    … … … … …
    login = relationship('Login', 
         back_populates="trainers")
    gclass = relationship('Gym_Class', 
         back_populates="trainers")

class Profile_Members(Base): 
    __tablename__ = "profile_members"
    id = Column(Integer, ForeignKey('login.id'), 
         primary_key=True, index=True)
    firstname = Column(String, unique=False, index=False)
    lastname = Column(String, unique=False, index=False)
    age = Column(Integer, unique=False, index=False)
    … … … … … …
    … … … … … …
    trainer_id = Column(Integer, 
        ForeignKey('profile_trainers.id'), unique=False, 
        index=False)
    login = relationship('Login', back_populates="members")
    attendance = relationship('Attendance_Member', 
          back_populates="members")
    gclass = relationship('Gym_Class', 
          back_populates="members") 
```

`relationship()` 指令是创建表关系的唯一指令。我们需要指定一些其参数，例如 *子模型类的名称* 和 *反向引用指定*。`back_populates` 参数指的是相关模型类的互补属性名称。这表示在连接查询事务期间需要使用某些关系加载技术获取的行。`backref` 参数也可以用来代替 `back_populates`。

另一方面，`relationship()` 方法可以返回一个 `List` 或标量对象，具体取决于关系类型。如果是一个 *一对一* 类型，父类应该将 `useList` 参数设置为 `False` 以指示它将返回一个标量值。否则，它将从子表中选取记录列表。之前的 `Login` 类定义显示 `Profile_Trainers` 和 `Profile_Members` 与 `Login` 之间保持一对一关系，因为 `Login` 将其 `uselist` 设置为 `False`。另一方面，`Profile_Members` 和 `Attendance_Member` 之间的模型关系是一个 *一对多* 类型，因为默认情况下 `uselist` 被设置为 `True`，如下面的定义所示：

```py
class Attendance_Member(Base):
    __tablename__ = "attendance_member"
    id = Column(Integer, primary_key=True, index=True)
    member_id = Column(Integer, 
        ForeignKey('profile_members.id'), unique=False, 
        index=False)
    timeout = Column(Time, unique=False, index=False)
    timein = Column(Time, unique=False, index=False)
    date_log = Column(Date, unique=False, index=False)

    members = relationship('Profile_Members', 
             back_populates="attendance")
```

在设置模型关系时，我们还必须考虑这些相关模型类在连接查询事务期间将使用的 *关系加载类型*。我们在 `relationship()` 的 `lazy` 参数中指定此细节，默认情况下分配给 `select`。这是因为 SQLAlchemy 默认使用懒加载技术在检索连接查询。然而，您可以将其修改为使用 `joined` (`lazy="joined"`), `subquery` (`lazy="subquery"`), `select in` (`lazy="selectin"`), `raise` (`lazy="raise"`), 或 `no` (`lazy="no"`) 加载。在这些选项中，`joined` 方法更适合 *INNER JOIN* 事务。

## 实现仓库层

在 SQLAlchemy ORM 中，创建仓库层需要 *模型类* 和一个 `Session` 对象。`Session` 对象由 `SessionFactory()` 指令派生，它建立了与数据库的所有通信，并在 `commit()` 或 `rollback()` 事务之前管理所有模型对象。当涉及到查询时，`Session` 实体将记录的结果集存储在一个称为 *身份映射* 的数据结构中，该映射使用主键维护每个数据记录的唯一标识。

所有仓库事务都是 *无状态的*，这意味着在数据库执行 `commit()` 或 `rollback()` 操作后，会自动关闭加载模型对象用于插入、更新和删除事务的会话。我们从 `sqlalchemy.orm` 模块导入 `Session` 类。

### 构建 CRUD 事务

现在，我们可以开始构建健身俱乐部应用的仓库层了，因为我们已经满足了构建 CRUD 事务的要求。下面的 `SignupRepository` 类是蓝图，它将展示我们如何 *插入*、*更新*、*删除* 和 *检索* 记录到/从 `signup` 表：

```py
from typing import Dict, List, Any
from sqlalchemy.orm import Session
from models.data.sqlalchemy_models import Signup
from sqlalchemy import desc
class SignupRepository: 

    def __init__(self, sess:Session):
        self.sess:Session = sess

    def insert_signup(self, signup: Signup) -> bool: 
        try:
            self.sess.add(signup)
            self.sess.commit()
        except: 
            return False 
        return True
```

到目前为止，`insert_signup()` 是使用 SQLAlchemy 将记录持久化到 `signup` 表的最准确的方法。`Session` 有一个 `add()` 方法，我们可以调用它来将所有记录对象添加到表中，以及一个 `commit()` 事务来最终将所有新记录刷新到数据库中。`Session` 的 `flush()` 方法有时会代替 `commit()` 来执行插入并关闭 `Session`，但大多数开发者通常使用后者。请注意，`signup` 表包含所有想要获取系统访问权限的健身房会员和教练。现在，下一个脚本实现了更新记录事务：

```py
    def update_signup(self, id:int, 
           details:Dict[str, Any]) -> bool: 
       try:
             self.sess.query(Signup).
                 filter(Signup.id == id).update(details)     
             self.sess.commit() 
       except: 
           return False 
       return True
```

`update_signup()` 提供了一个简短、直接且健壮的解决方案来更新 SQLAlchemy 中的记录。另一种可能的解决方案是通过 `self.sess.query(Signup).filter(Signup.id == id).first()` 查询记录，用 `details` 字典中的新值替换检索到的对象的属性值，然后调用 `commit()`。这种方式是可以接受的，但它需要三个步骤，而不是在 `filter()` 之后调用 `update()` 方法，后者只需要一步。下一个脚本是一个删除记录事务的实现：

```py
    def delete_signup(self, id:int) -> bool: 
        try:
           signup = self.sess.query(Signup).
                  filter(Signup.id == id).delete()
           self.sess.commit()
        except: 
            return False 
        return True
```

另一方面，`delete_signup()` 遵循 `update_signup()` 的策略，在调用 `delete()` 之前先使用 `filter()`。另一种实现方式是再次使用 `sess.query()` 检索对象，并将检索到的对象作为参数传递给 `Session` 对象的 `delete(obj)`，这是一个不同的函数。始终记得调用 `commit()` 来刷新更改。现在，以下脚本展示了如何实现查询事务：

```py
    def get_all_signup(self):
        return self.sess.query(Signup).all() 
    def get_all_signup_where(self, username:str):
        return self.sess.
             query(Signup.username, Signup.password).
             filter(Signup.username == username).all() 

    def get_all_signup_sorted_desc(self):
        return self.sess.
            query(Signup.username,Signup.password).
            order_by(desc(Signup.username)).all()

    def get_signup(self, id:int): 
        return self.sess.query(Signup).
             filter(Signup.id == id).one_or_none()
```

此外，`SignupRepository` 还突出了以多种形式检索多个和单个记录。`Session` 对象有一个 `query()` 方法，它需要一个或多个 *模型类* 或 *模型列名* 作为参数。函数参数执行带有列投影的记录检索。例如，给定的 `get_all_signup()` 选择了所有带有所有列投影的注册记录。如果我们只想包括 `username` 和 `password`，我们可以将查询写成 `sess.query(Signup.username, Signup.password)`，就像在给定的 `get_all_signup_where()` 中一样。这个 `query()` 方法还展示了如何使用 `filter()` 方法以及适当的条件表达式来管理约束。过滤总是在列投影之后进行。

另一方面，`Session`对象有一个`order_by()`方法，它接受列名作为参数。它在查询事务的提取之前执行，是查询事务系列中的最后一个操作。给定的示例`get_all_signup_sorted_desc()`按`username`降序排序所有`Signup`对象。

`query()`构建器的最后一部分返回事务的结果，无论是记录列表还是单个记录。`all()`函数结束返回多个记录的查询语句，而`first()`、`scalar()`、`one()`或`one_or_none()`可以应用于结果为单行的情况。在`get_signup()`中，使用`one_or_none()`在无记录返回时引发异常。对于 SQLAlchemy 的查询事务，所有这些函数都可以关闭`Session`对象。SQLAlchemy 的存储库类位于`ch05a`文件夹的`/repository/sqlalchemy/signup.py`模块脚本文件中。

### 创建 JOIN 查询

对于 FastAPI 支持的所有 ORM，只有 SQLAlchemy 实现了具有实用性和功能的联合查询，就像我们之前实现 CRUD 事务一样。我们几乎使用了创建联合查询所需的所有方法，除了`join()`。

让我们看看`LoginMemberRepository`，它展示了如何使用 SQLAlchemy 的模型类在*一对一关系*中创建联合查询语句：

```py
class LoginMemberRepository(): 
    def __init__(self, sess:Session):
        self.sess:Session = sess

    def join_login_members(self):
        return self.sess.
           query(Login, Profile_Members).
             filter(Login.id == Profile_Members.id).all()
```

`join_login_members()`展示了创建*JOIN*查询的传统方法。此解决方案需要传递父类和子类作为查询参数，并通过`filter()`方法覆盖`ON`条件。在`query()`构建器中，父模型类必须在子类之前出现在列投影中，以便提取所需的结果。

另一种方法是使用`select_from()`函数而不是`query()`来区分父类和子类。这种方法更适合*一对一*关系。

另一方面，`MemberAttendanceRepository`展示了`Profile_Members`和`Attendance_Member`模型类之间的*一对多*关系：

```py
class MemberAttendanceRepository(): 
    def __init__(self, sess:Session):
        self.sess:Session = sess

    def join_member_attendance(self):
        return self.sess.
           query(Profile_Members, Attendance_Member).
           join(Attendance_Member).all()
    def outer_join_member(self):
         return self.sess.
            query(Profile_Members, Attendance_Member).
            outerjoin(Attendance_Member).all()
```

`join_member_attendance()`展示了在构建`Profile_Members`和`Attendance_Member`之间的*内连接*查询时`join()`方法的使用。由于`join()`自动检测并识别在开头定义的`relationship()`参数和`ForeignKey`构造，因此不再需要`filter()`来构建`ON`条件。但如果存在其他附加约束，`filter()`始终可以调用，但必须在`join()`方法之后。

`outer_join_member()`存储库方法实现了从一对多关系中的*外连接*查询。`outerjoin()`方法将提取所有映射到相应`Attendance_Member`的`Profile_Members`记录，如果没有，则返回`null`。

## 执行事务

现在，让我们将这些存储库事务应用到我们应用程序的与行政相关的 API 服务中。我们不会使用集合来存储所有记录，而是将利用 ORM 的事务来使用 PostgreSQL 管理数据。首先，我们需要导入存储库所需的必要组件，例如`SessionFactory`、存储库类和`Signup`模型类。像`Session`和其他`typing` API 这样的 API 只能作为实现类型提示的一部分。

以下脚本展示了管理员 API 服务的一部分，突出了新访问注册的插入和检索服务：

```py
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from db_config.sqlalchemy_connect import SessionFactory
from repository.sqlalchemy.signup import SignupRepository,
   LoginMemberRepository, MemberAttendanceRepository
from typing import List
router = APIRouter()
def sess_db():
    db = SessionFactory()
    try:
        yield db
    finally:
        db.close()
```

首先，我们需要通过`SessionFactory()`创建`Session`实例，这是从`sessionmaker()`派生出来的，因为存储库层依赖于会话。在我们的应用程序中，使用了一个自定义生成器`sess_db()`来打开和销毁`Session`实例。它被注入到 API 服务方法中，以告诉`Session`实例继续实例化`SignupRepository`：

```py
@router.post("/signup/add")
def add_signup(req: SignupReq, 
          sess:Session = Depends(sess_db)):
    repo:SignupRepository = SignupRepository(sess)
    signup = Signup(password= req.password, 
                 username=req.username,id=req.id)
    result = repo.insert_signup(signup)
    if result == True:
        return signup
    else: 
        return JSONResponse(content={'message':'create 
                  signup problem encountered'}, 
                status_code=500)
```

一旦实例化，存储库可以通过`insert_signup()`方法提供记录插入，该方法插入`Signup`记录。它的另一个方法是`get_all_signup()`，该方法检索所有待批准的登录账户：

```py
@router.get("/signup/list", response_model=List[SignupReq])
def list_signup(sess:Session = Depends(sess_db)):
    repo:SignupRepository = SignupRepository(sess)
    result = repo.get_all_signup()
    return result
@router.get("/signup/list/{id}", response_model=SignupReq)
def get_signup(id:int, sess:Session = Depends(create_db)): 
    repo:SignupRepository = SignupRepository(sess)
    result = repo.get_signup(id)
    return result
```

`get_signup()`和`list_signup()`服务都有一个`SignupReq`类型的`request_model`，这决定了 API 的预期输出。但正如你可能已经注意到的，`get_signup()`返回`Signup`对象，而`list_signup()`返回`Signup`记录的列表。这是如何实现的？如果`request_model`用于捕获 SQLAlchemy 查询事务的查询结果，则`BaseModel`类或请求模型必须包含一个嵌套的`Config`类，其`orm_mode`设置为`True`。这个内置配置在所有记录对象被过滤并存储在请求模型之前，为存储库使用的 SQLAlchemy 模型类型启用类型映射和验证。有关`response_model`参数的更多信息，请参阅*第一章*，*为初学者设置 FastAPI*。

我们应用程序的查询服务使用的`SignupReq`定义如下：

```py
from pydantic import BaseModel
class SignupReq(BaseModel): 
    id : int 
    username: str 
    password: str 

    class Config:
        orm_mode = True
```

脚本展示了如何使用等号（`=`）而不是典型的冒号符号（`:`）来启用`orm_mode`，这意味着`orm_mode`是配置细节，而不是类属性的一部分。

总体而言，使用 SQLAlchemy 作为存储库层是系统化和程序化的。它很容易将模型类与模式定义进行映射和同步。通过模型类建立关系既方便又可预测。尽管涉及许多 API 和指令，但它仍然是领域建模和存储库构建最广泛支持的库。它的文档([`docs.sqlalchemy.org/en/14/`](https://docs.sqlalchemy.org/en/14/))完整且信息丰富，足以指导开发者了解不同的 API 类和方法。

SQLAlchemy 受到许多人喜爱的另一个特性是它能够在应用级别生成表模式。

## 创建表

通常，SQLAlchemy 与数据库管理员已经生成的表模式一起工作。在这个项目中，ORM 设置是从设计领域模型类开始的，然后将其映射到实际的表。但 SQLAlchemy 可以为 FastAPI 平台在运行时自动创建表模式，这在项目的测试或原型阶段可能很有帮助。

`sqlalchemy`模块有一个`Table()`指令，可以使用`Column()`方法创建一个表对象，该方法我们在映射中使用过。以下是一个示例脚本，展示了 ORM 如何在应用级别创建注册表：

```py
from sqlalchemy import Table, Column, Integer, String, 
               MetaData
from db_config.sqlalchemy_connect import engine
meta = MetaData()
signup = Table(
   'signup', meta, 
   Column('id', Integer, primary_key = True, 
          nullable=False), 
   Column('username', String, unique = False, 
          nullable = False), 
   Column('password', String, unique = False, 
          nullable = False), 
)
meta.create_all(bind=engine)
```

模式定义的一部分是`MetaData()`，这是一个包含生成表所需方法的注册表。当所有模式定义都获得批准后，`MetaData()`实例的`create_all()`方法与引擎一起执行以创建表。这个过程听起来很简单，但在生产阶段的项目中，我们很少追求 SQLAlchemy 的这个 DDL 特性。

现在，让我们探索如何使用 SQLAlchemy 为异步 API 服务创建异步 CRUD 事务。

# 使用 SQLAlchemy 实现异步 CRUD 事务

从版本 1.4 开始，SQLAlchemy 支持`Session`对象。我们的`ch05b`项目展示了 SQLAlchemy 的异步方面。

## 安装符合 asyncio 规范的数据库驱动程序

在我们开始设置数据库配置之前，我们需要安装以下符合 asyncio 规范的驱动程序：`aiopg`和`asyncpg`。首先，我们需要安装`aiopg`，这是一个库，它将帮助进行任何异步访问 PostgreSQL：

```py
pip install aiopg
```

接下来，我们必须安装`asyncpg`，它通过 Python 的 AsyncIO 框架帮助构建 PostgreSQL 异步事务：

```py
pip install asyncpg
```

这个驱动程序是一个*非数据库 API 规范*的驱动程序，因为它在 AsyncIO 环境之上运行，而不是在同步数据库事务的数据库 API 规范之上。

## 设置数据库的连接

安装必要的驱动程序后，我们可以通过应用程序的`create_async_engine()`方法推导出数据库引擎，该方法创建了一个异步版本的 SQLAlchemy 的`Engine`，称为`AsyncEngine`。此方法有参数可以设置，如`future`，当设置为`True`时，可以在 CRUD 事务期间启用各种异步功能。此外，它还有一个`echo`参数，可以在运行时提供服务器日志中生成的 SQL 查询。但最重要的是数据库 URL，现在它反映了通过调用`asyncpg`协议的异步数据库访问。以下是对 PostgreSQL 数据库的异步连接的完整脚本：

```py
from sqlalchemy.ext.asyncio import create_async_engine
DB_URL = 
  "postgresql+asyncpg://postgres:admin2255@
       localhost:5433/fcms"
engine = create_async_engine(DB_URL, future=True, 
               echo=True)
```

`DB_URL`中的附加`"+asyncpg"`细节表明`psycopg2`将不再是 PostgreSQL 的核心数据库驱动程序；相反，将使用`asyncpg`。此细节使`AsyncEngine`能够利用`asyncpg`建立与数据库的连接。省略此细节将指示引擎识别`psycopg2`数据库 API 驱动程序，这将在 CRUD 事务期间引起问题。

## 创建会话工厂

与同步版本类似，`sessionmaker()`指令被用来创建带有一些新参数的会话工厂以启用`AsyncSession`。首先，将`expire_on_commit`参数设置为`False`，以便在事务期间，即使调用`commit()`后，模型实例及其属性值仍然可访问。与同步环境不同，所有实体类及其列对象在事务提交后仍然可以被其他进程访问。然后，其`class_`参数携带类名`AsyncSession`，该实体将控制 CRUD 事务。当然，`sessionmaker()`仍然需要`AsyncConnection`及其底层的异步上下文管理器。

以下脚本展示了如何使用`sessionmaker()`指令推导出会话工厂：

```py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
engine = create_async_engine(DB_URL, future=True, 
               echo=True)
AsynSessionFactory = sessionmaker(engine, 
       expire_on_commit=False, class_=AsyncSession)
```

异步 SQLAlchemy 数据库连接的完整配置可以在`/db_config/sqlalchemy_async_connect.py`模块脚本文件中找到。现在让我们创建模型层。

## 创建基类和模型层

使用`declarative_base()`创建`Base`类和使用`Base`创建模型类与同步版本中的操作相同。构建异步存储库事务的数据层不需要额外的参数。

## 构建存储库层

实现异步 CRUD 事务与实现同步事务完全不同。ORM 支持使用`AsyncConnection` API 的`execute()`方法来运行一些内置的 ORM 核心方法，即`update()`、`delete()`和`insert()`。当涉及到查询事务时，使用来自`sqlalchemy.future`模块的新`select()`指令而不是核心的`select()`方法。由于`execute()`是一个`async`方法，这要求所有仓库事务都是`async`，以便应用*Async/Await*设计模式。下面的`AttendanceRepository`使用了 SQLAlchemy 的异步类型：

```py
from typing import List, Dict, Any
from sqlalchemy import update, delete, insert
from sqlalchemy.future import select
from sqlalchemy.orm import Session
from models.data.sqlalchemy_async_models import 
            Attendance_Member
class AttendanceRepository: 

    def __init__(self, sess:Session):
        self.sess:Session = sess

    async def insert_attendance(self, attendance: 
           Attendance_Member) -> bool: 
        try:
            sql = insert(Attendance_Member).
                   values(id=attendance.id, 
                     member_id=attendance.member_id, 
                     timein=attendance.timein, 
                     timeout=attendance.timeout, 
                     date_log=attendance.date_log)
            sql.execution_options(
                   synchronize_session="fetch")
            await self.sess.execute(sql)        
        except: 
            return False 
        return True
```

前面的脚本中给出的*异步*`insert_attendance()`方法展示了在创建健身房成员的出勤日志时使用`insert()`指令。首先，我们需要将模型类名传递给`insert()`，以便会话知道要访问哪个表进行事务。之后，它发出`values()`方法来投影插入的所有列值。最后，我们需要调用`execute()`方法来运行最终的`insert()`语句，并自动提交更改，因为我们没有在配置期间关闭`sessionmaker()`的`autocommit`参数。不要忘记在运行异步方法之前调用`await`，因为这次所有操作都是在 AsyncIO 平台上运行的。

此外，您还有在运行`execute()`之前添加一些额外执行细节的选项。这些选项之一是`synchronize_session`，它告诉会话始终使用`fetch`方法同步模型属性值和数据库中的更新值。

对于`update_attendance()`和`delete_attendance()`方法，几乎采用相同的程序。我们可以通过`execute()`运行它们，无需其他操作：

```py
    async def update_attendance(self, id:int, 
           details:Dict[str, Any]) -> bool: 
       try:
           sql = update(Attendance_Member).where(
              Attendance_Member.id == id).values(**details)
           sql.execution_options(
              synchronize_session="fetch")
           await self.sess.execute(sql)

       except: 
           return False 
       return True

    async def delete_attendance(self, id:int) -> bool: 
        try:
           sql = delete(Attendance_Member).where(
                Attendance_Member.id == id)
           sql.execution_options(
                synchronize_session="fetch")
           await self.sess.execute(sql)
        except: 
            return False 
        return True
```

当涉及到查询时，仓库类包含`get_all_attendance()`方法，用于检索所有出勤记录，以及`get_attendance()`方法，通过其`id`检索特定成员的出勤日志。构建`select()`方法是一个简单且实用的任务，因为它类似于在 SQL 开发中编写原生的`SELECT`语句。首先，该方法需要知道要投影哪些列，然后如果有任何约束，它将满足这些约束。然后，它需要`execute()`方法来异步运行查询并提取`Query`对象。生成的`Query`对象有一个`scalars()`方法，我们可以调用它来检索记录列表。不要忘记通过调用`all()`方法来关闭会话。

另一方面，`check_attendance()`使用`Query`对象的`scalar()`方法来检索一条记录：特定的出勤记录。除了记录检索外，`scalar()`还会关闭会话：

```py
    async def get_all_attendance(self):
        q = await self.sess.execute(
               select(Attendance_Member))
        return q.scalars().all()

    async def get_attendance(self, id:int): 
        q = await self.sess.execute(
           select(Attendance_Member).
             where(Attendance_Member.member_id == id))
        return q.scalars().all()
    async def check_attendance(self, id:int): 
        q = await self.sess.execute(
          select(Attendance_Member).
              where(Attendance_Member.id == id))
        return q.scalar()        
```

异步 SQLAlchemy 的存储库类可以在`/repository/sqlalchemy/attendance.py`模块脚本文件中找到。现在，让我们将这些异步事务应用到我们的健身馆应用程序的出勤监控服务中。

重要提示

在`update_attendance()`函数中的`**`操作符是一个 Python 操作符重载，它将字典转换为`kwargs`。因此，`**details`的结果是`select()`指令的`values()`方法的`kwargs`参数。

## 运行 CRUD 事务

在创建`Session`实例时，AsyncIO 驱动的 SQLAlchemy 与数据库 API 兼容选项之间有两个主要区别：

+   首先，由`AsyncSessionFactory()`指令创建的`AsyncSession`需要一个异步的`with`上下文管理器，因为连接的`AsyncEngine`需要在每次`commit()`事务后关闭。在同步 ORM 版本中，关闭会话工厂不是程序的一部分。

+   其次，在其创建之后，`AsyncSession`只有在服务调用其`begin()`方法时才会开始执行所有的 CRUD 事务。主要原因在于`AsyncSession`可以被关闭，并且在事务执行后需要关闭。这就是为什么使用另一个异步上下文管理器来管理`AsyncSession`的原因。

以下代码显示了`APIRouter`脚本，它实现了使用异步`AttendanceRepository`监控健身馆会员出勤的服务：

```py
from fastapi import APIRouter
from db_config.sqlalchemy_async_connect import 
          AsynSessionFactory
from repository.sqlalchemy.attendance import 
         AttendanceRepository
from models.requests.attendance import AttendanceMemberReq
from models.data.sqlalchemy_async_models import 
         Attendance_Member
router = APIRouter()
@router.post("/attendance/add")
async def add_attendance(req:AttendanceMemberReq ):
    async with AsynSessionFactory() as sess:
        async with sess.begin():
            repo = AttendanceRepository(sess)
            attendance = Attendance_Member(id=req.id,  
                member_id=req.member_id, 
                timein=req.timein, timeout=req.timeout, 
                date_log=req.date_log)
            return await repo.insert_attendance(attendance)

@router.patch("/attendance/update")
async def update_attendance(id:int, 
                     req:AttendanceMemberReq ):
    async with AsynSessionFactory() as sess:
        async with sess.begin():
            repo = AttendanceRepository(sess)
            attendance_dict = req.dict(exclude_unset=True)
            return await repo.update_attendance(id, 
                    attendance_dict)
@router.delete("/attendance/delete/{id}")
async def delete_attendance(id:int): 
     async with AsynSessionFactory() as sess:
        async with sess.begin():
            repo = AttendanceRepository(sess)
            return await repo.delete_attencance(id)
@router.get("/attendance/list")
async def list_attendance():
     async with AsynSessionFactory() as sess:
        async with sess.begin():
            repo = AttendanceRepository(sess)
            return await repo.get_all_attendance()
```

前面的脚本显示了存储库类和`AsyncSession`实例之间没有直接的参数传递。会话必须符合两个上下文管理器，才能成为一个有效的会话。这个语法在*SQLAlchemy 1.4*下是有效的，未来随着 SQLAlchemy 的下一个版本可能会发生变化。

为异步事务创建的其他 ORM 平台更容易使用。其中之一就是**GINO**。

# 使用 GINO 进行异步事务

**GINO**，代表**GINO Is Not ORM**，是一个轻量级的异步 ORM，它运行在 SQLAlchemy Core 和 AsyncIO 环境之上。它所有的 API 都是异步准备就绪的，这样你可以构建上下文数据库连接和事务。它内置了对*JSONB*的支持，因此可以将结果转换为 JSON 对象。但有一个限制：GINO 仅支持 PostgreSQL 数据库。

在创建健身馆项目时，唯一可用的稳定 GINO 版本是 1.0.1，它需要*SQLAlchemy 1.3*。因此，安装 GINO 将自动卸载*SQLAlchemy 1.4*，从而将 GINO 存储库添加到`ch05a`项目中，以避免与 SQLAlchemy 的异步版本发生冲突。

你可以使用以下命令安装 GINO 的最新版本：

```py
pip install gino
```

## 安装数据库驱动

由于它只支持 PostgreSQL 作为 RDBMS，你只需要使用`pip`命令安装`asyncpg`。

## 建立数据库连接

除了`Gino`指令外，不需要其他 API 来打开数据库连接。我们需要实例化该类以开始构建领域层。`Gino`类可以从 ORM 的`gino`模块导入，如下面的脚本所示：

```py
from gino import Gino
db = Gino()
```

其实例类似于一个门面，它控制所有数据库事务。一旦提供了正确的 PostgreSQL 管理员凭据，它就会开始建立数据库连接。完整的 GINO 数据库连接脚本可以在`/db_config/gino_connect.py`脚本文件中找到。现在让我们构建模型层。

## 构建模型层

在结构、列元数据和`__tablename__`属性的存在方面，GINO 中的模型类定义与 SQLAlchemy 相似。唯一的区别是超类类型，因为 GINO 使用数据库引用实例的`db`中的`Model`类。以下脚本展示了如何将`Signup`领域模型映射到`signup`表：

```py
from db_config.gino_connect import db
class Signup(db.Model):
    __tablename__ = "signup"
    id = db.Column(db.Integer, primary_key=True, 
               index=True)
    username = db.Column('username',db.String, 
               unique=False, index=False)
    password = db.Column('password',db.String, 
               unique=False, index=False)
```

与 SQLAlchemy 一样，所有模型类都必须有`__tablename__`属性来指示其映射的表模式。在定义列元数据时，`db`对象有一个`Column`指令可以设置属性，如*列类型*、*主键*、*唯一*、*默认值*、*可为空*和*索引*。列类型也来自`db`引用对象，这些类型与 SQLAlchemy 相同，即`String`、`Integer`、`Date`、`Time`、`Unicode`和`Float`。

如果模型属性的名称与列名称不匹配，`Column`指令的第一个参数将注册实际列的名称并将其映射到模型属性。`username`和`password`列是映射类属性到表列名称的示例情况。

### 映射表关系

在编写本文时，GINO 默认只支持*多对一关系*。`db`引用对象有一个`ForeignKey`指令，它建立了与父模型的键关系。它只需要父表的实际引用键列和表名称来追求映射。在子模型类的`Column`对象中设置`ForeignKey`属性就足够配置以执行*左外连接*来检索父模型类的所有子记录。GINO 没有`relationship()`函数来处理有关如何检索父模型类子记录的更多细节。然而，它具有内置的加载器，可以自动确定外键并在之后执行多对一连接查询。此连接查询的完美设置是`Profile_Trainers`和`Gym_Class`模型类之间的关系配置，如下面的脚本所示：

```py
class Profile_Trainers(db.Model):
    __tablename__ = "profile_trainers"
    id = db.Column(db.Integer, db.ForeignKey('login.id'), 
              primary_key=True, index=True)
    firstname = db.Column(db.String, unique=False, 
              index=False)
    … … … … … …
    shift = db.Column(db.Integer, unique=False, 
              index=False)
class Gym_Class(db.Model): 
    __tablename__ = "gym_class"
    id = db.Column(db.Integer, primary_key=True, 
          index=True)
    member_id = db.Column(db.Integer, 
       db.ForeignKey('profile_members.id'), unique=False, 
         index=False)
    trainer_id = db.Column(db.Integer, 
      db.ForeignKey('profile_trainers.id'), unique=False,
         index=False)
    approved = db.Column(db.Integer, unique=False, 
       index=False)
```

如果我们需要构建一个能够处理*一对多*或*一对一*关系的查询，我们可能需要做出一些调整。为了使*左外连接*查询能够工作，父模型类必须定义一个`set`集合来包含在涉及一对多关系的连接查询期间的所有子记录。对于*一对一*关系，父模型只需要实例化子模型：

```py
class Login(db.Model): 
    __tablename__ = "login"
    id = db.Column(db.Integer, primary_key=True, 
               index=True)
    username = db.Column(db.String, unique=False, 
               index=False)
    … … … … … …
    def __init__(self, **kw):
        super().__init__(**kw)
        self._child = None
    @property
    def child(self):
        return self._child
    @child.setter
    def child(self, child):
        self._child = child
class Profile_Members(db.Model): 
    __tablename__ = "profile_members"
    id = db.Column(db.Integer, db.ForeignKey('login.id'), 
          primary_key=True, index=True)
    … … … … … … 
    weight = db.Column(db.Float, unique=False, index=False)
    trainer_id = db.Column(db.Integer, 
        db.ForeignKey('profile_trainers.id'), unique=False, 
            index=False)

    def __init__(self, **kw):
        super().__init__(**kw)
        self._children = set()
    @property
    def children(self):
        return self._children
    @children.setter
    def children(self, child):
        self._children.add(child)
```

这个*集合*或*子对象*必须在父的`__init__()`中实例化，以便通过 ORM 的 loader 通过*children*或*child* `@property`分别访问。使用`@property`是管理连接记录的唯一方式。

注意，loader API 的存在证明了 GINO 不支持 SQLAlchemy 所具有的自动化关系。如果我们想偏离其核心设置，就需要使用 Python 编程来添加平台不支持的一些功能，例如在`Profile_Members`和`Gym_Class`之间以及`Login`和`Profile_Members`/`Profile_Trainers`之间的一对多设置。在前面的脚本中，请注意`Profile_Members`中包含了一个构造函数和自定义的`children` Python 属性，以及`Login`中的自定义`child`属性。这是因为 GINO 只内置了一个`parent`属性。

你可以在`/models/data/gino_models.py`脚本中找到 GINO 的域模型。

重要提示

`@property`是 Python 装饰器，用于在类中实现 getter/setter。这隐藏了一个实例变量，并暴露了其*getter*和*setter*属性*字段*。使用`@property`是实现 Python 中*封装*原则的一种方式。

## 实现 CRUD 事务

让我们考虑以下`TrainerRepository`，它管理教练档案。它的`insert_trainer()`方法展示了实现插入事务的传统方式。GINO 要求其模型类调用从`db`引用对象继承的`create()`方法。所有列值都通过命名参数或作为`kwargs`的包传递给`create()`方法，在记录对象持久化之前。但 GINO 允许另一种插入选项，它使用通过向其构造函数注入列值得到的模型实例。创建的实例有一个名为`create()`的方法，它插入记录对象而不需要任何参数：

```py
from models.data.gino_models import Profile_Members, 
           Profile_Trainers, Gym_Class
from datetime import date, time
from typing import List, Dict, Any
class TrainerRepository: 

    async def insert_trainer(self, 
             details:Dict[str, Any]) -> bool: 
        try:
            await Profile_Trainers.create(**details)
        except Exception as e: 
            print(e)
            return False 
        return True
```

`update_trainer()`展示了 GINO 如何更新表记录。根据脚本，以 GINO 的方式更新表涉及以下步骤：

+   首先，它需要模型类的`get()`类方法来检索具有`id`主键的记录对象。

+   第二，提取的记录有一个名为`update()`的实例方法，它将自动使用其`kwargs`参数中指定的新数据修改映射的行。`apply()`方法将提交更改并关闭事务：

    ```py
        async def update_trainer(self, id:int, 
                     details:Dict[str, Any]) -> bool: 
           try:
                trainer = await Profile_Trainers.get(id)
                await trainer.update(**details).apply()       
           except: 
               return False 
           return True
    ```

另一个选项是使用 SQLAlchemy 的 `ModelClass.update.values(ModelClass).where(expression)` 子句，当应用于 `update_trainer()` 时，将给出以下最终语句：

```py
Profile_Trainers.update.values(**details).
     where(Profile_Trainers.id == id).gino.status()
```

它的 `delete_trainer()` 也遵循 GINO *更新* 事务的相同方法。这个事务是两步过程，最后一步需要调用提取的记录对象的 `delete()` 实例方法：

```py
    async def delete_trainer(self, id:int) -> bool: 
        try:
           trainer = await Profile_Trainers.get(id)
           await trainer.delete()        
        except: 
            return False 
        return True
```

另一方面，`TrainerRepository` 有两个方法，`get_member()` 和 `get_all_member()`，展示了 GINO 如何构建查询语句：

+   前者通过模型类的 `get()` 类方法使用其主键检索特定记录对象

+   后者使用 `query` 的 `gino` 扩展来利用 `all()` 方法，该方法检索记录：

    ```py
        async def get_all_member(self):
            return await Profile_Trainers.query.gino.all()
        async def get_member(self, id:int): 
                return await Profile_Trainers.get(id)
    ```

但在查询执行过程中将数据库行转换为模型对象的是 GINO 的内置加载器。如果我们进一步扩展 `get_all_member()` 中提出的解决方案，这将看起来像这样：

```py
query = db.select([Profile_Trainers])
q = query.execution_options(
         loader=ModelLoader(Profile_Trainers))
users = await q.gino.all()
```

在 GINO ORM 中，所有查询都利用 `ModelLoader` 将每个数据库记录加载到模型对象中：

```py
class GymClassRepository:

    async def join_classes_trainer(self):
        query = Gym_Class.join(Profile_Trainers).select()
        result = await query.gino.load(Gym_Class.
            distinct(Gym_Class.id).
                load(parent=Profile_Trainers)).all()
        return result 

    async def join_member_classes(self):
        query = Gym_Class.join(Profile_Members).select()
        result = await query.gino.load(Profile_Members.
           distinct(Profile_Members.id).
              load(add_child=Gym_Class)).all()
        return result
```

如果正常查询需要 `ModelLoader`，那么 *连接* 查询事务需要什么？GINO 没有自动支持表关系，没有 `ModelLoader`，创建 *连接* 查询是不可能的。`join_classes_trainer()` 方法实现了 `Profile_Trainers` 和 `Gym_Class` 的 *一对多* 查询。查询中的 `distinct(Gym_Class.id).load(parent=Profile_Trainers)` 子句为 `GymClass` 创建了一个 `ModelLoader`，它将合并并加载 `Profile_Trainers` 父记录到其子 `Gym_Class`。`join_member_classes()` 创建 *一对多* 连接，而 `distinct(Profile_Members.id).load(add_child=Gym_Class)` 创建一个 `ModelLoader` 来构建 `Gym_Class` 记录的集合，按照 `Profile_Members` 父记录。

另一方面，`Gym_Class` 和 `Profile_Members` 之间的 *多对一* 关系使用 `Profile_Member` 的 `load()` 函数，这是一种将 `Gym_Class` 子记录与 `Profile_Members` 匹配的不同方法。以下联合查询是 *一对多* 设置的反面，因为这里的 `Gym_Class` 记录在左侧，而配置文件在右侧：

```py
    async def join_classes_member(self):
        result = await 
          Profile_Members.load(add_child=Gym_Class)
           .query.gino.all()
```

因此，加载器在 GINO 中构建查询时扮演着重要角色，尤其是在连接操作中。尽管它使查询构建变得困难，但它仍然为许多复杂查询提供了灵活性。

所有 GINO 的存储库类都可以在 `/repository/gino/trainers.py` 脚本中找到。

## 运行 CRUD 事务

为了让我们的存储库在 `APIRouter` 模块中运行，我们需要通过将 `db` 引用对象绑定到实际数据库的 `DB_URL` 来打开数据库连接。绑定过程使用可靠的函数是理想的，因为更简单的部署形式是通过 `APIRouter` 注入完成的。以下脚本展示了如何设置这种数据库绑定：

```py
from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from db_config.gino_connect import db
from models.requests.trainers import ProfileTrainersReq
from repository.gino.trainers import TrainerRepository
async def sess_db():
    await db.set_bind(
     "postgresql+asyncpg://
       postgres:admin2255@localhost:5433/fcms")

router = APIRouter(dependencies=[Depends(sess_db)])
@router.patch("/trainer/update" )
async def update_trainer(id:int, req: ProfileTrainersReq): 
    mem_profile_dict = req.dict(exclude_unset=True)
    repo = TrainerRepository()
    result = await repo.update_trainer(id, 
           mem_profile_dict)
    if result == True: 
        return req 
    else: 
        return JSONResponse(
    content={'message':'update trainer profile problem 
         encountered'}, status_code=500)

@router.get("/trainer/list")
async def list_trainers(): 
    repo = TrainerRepository()
    return await repo.get_all_member()
```

在前面的代码中显示的 `list_trainers()` 和 `update_trainer()` REST 服务是我们 *健身俱乐部* 应用程序的一些服务，在将 `sess_db()` 注入 `APIRouter` 后将成功运行 `TrainerRepository`。GINO 在建立与 PostgreSQL 的连接时不需要太多细节，除了 `DB_URL`。始终在 URL 中指定 `asyncpg` 方言，因为它是 GINO 作为同步 ORM 支持的唯一驱动程序。

## 创建表

GINO 和 SQLAlchemy 在框架级别创建表架构的方法相同。两者都要求使用 `MetaData` 和 `Column` 指令来构建 `Table` 定义。然后，建议使用异步函数通过 `create_engine()` 方法结合我们的 `DB_URL` 来获取引擎。与 SQLAlchemy 类似，这个引擎在通过 `create_all()` 构建表时起着关键作用，但这次它使用 GINO 的 `GinoSchemaVisitor` 实例。以下脚本展示了 GINO 使用 AsyncIO 平台生成表的完整实现：

```py
from sqlalchemy import Table, Column, Integer, String, 
           MetaData, ForeignKey'
import gino
from gino.schema import GinoSchemaVisitor
metadata = MetaData()
signup = Table(
    'signup', metadata,
    Column('id', Integer, primary_key=True),
    Column('username', String),
    Column('password', String),
)
   … … … … …
async def db_create_tbl():
    engine = await gino.create_engine(DB_URL)
    await GinoSchemaVisitor(metadata).create_all(engine)
```

如 SQLAlchemy 所述，在开始时执行 DDL 事务（如模式自动生成）是可选的，因为这可能会降低 FastAPI 的性能，甚至可能导致现有数据库模式中的某些冲突。

现在，让我们探索另一个需要自定义 Python 编码的 ORM：**Pony ORM**。

# 在存储库层使用 Pony ORM

Pony ORM 依赖于 Python 语法来构建模型类和存储库事务。这个 ORM 只使用 Python 数据类型，如 `int`、`str` 和 `float`，以及类类型来实现模型定义。它使用 Python `lambda` 表达式来建立 CRUD 事务，尤其是在映射表关系时。此外，Pony 在读取记录时强烈支持记录对象的 JSON 转换。另一方面，Pony 可以缓存查询对象，这比其他方法提供了更快的性能。Pony ORM 的代码可以在 `ch05a` 项目中找到。

要使用 Pony，我们需要使用 `pip` 来安装它。这是因为它是一个第三方平台：

```py
pip install pony
```

## 安装数据库驱动程序

由于 Pony 是一个设计用于构建同步事务的 ORM，我们需要 `psycopg2` PostgreSQL 驱动程序。我们可以使用 `pip` 命令来安装它：

```py
pip install psycopg2
```

## 创建数据库的连接性

Pony 建立数据库连接的方法简单且声明式。它只需要从 `pony.orm` 模块实例化 `Database` 指令来使用正确的数据库凭据连接到数据库。以下脚本用于 *健身俱乐部* 原型：

```py
from pony.orm import  Database
db = Database("postgres", host="localhost", port="5433", 
  user="postgres", password="admin2255", database="fcms")
```

如您所见，构造函数的第一个参数是 *数据库方言*，后面跟着 `kwargs`，其中包含有关连接的所有详细信息。完整的配置可以在 `/db_config/pony_connect.py` 脚本文件中找到。现在，让我们创建 Pony 的模型类。

## 定义模型类

创建的数据库对象 `db` 是定义 Pony *实体*（一个指模型类的术语）的唯一组件。它有一个 `Entity` 属性，用于将每个模型类子类化以提供 `_table_` 属性，该属性负责 *表-实体* 映射。所有实体实例都绑定到 `db` 并映射到表。以下脚本展示了 `Signup` 类如何成为模型层的实体：

```py
from pony.orm import  Database, PrimaryKey, Required, 
         Optional, Set
from db_config.pony_connect import db
from datetime import date, time

class Signup(db.Entity):
    _table_ = "signup"
    id = PrimaryKey(int)
    username = Required(str, unique=True, max_len=100, 
         nullable=False, column='username')
    password = Required(str, unique=Fals, max_len=100, 
         nullable=False, column='password')
```

`pony.orm` 模块包含 `Required`、`Optional`、`PrimaryKey` 或 `Set` 指令，用于创建列属性。由于每个实体都必须有一个主键，因此使用 `PrimaryKey` 来定义实体的列属性。如果没有主键，Pony ORM 将隐式地为实体生成一个具有以下定义的 `id` 主键：

```py
id = PrimaryKey(int, auto=True)
```

另一方面，`Set` 指令表示实体之间的关系。所有这些指令都有一个强制属性列类型，它以 Python 语法（例如，`int`、`str`、`float`、`date` 或 `time`）或任何类类型声明列值类型。其他列属性包括 `auto`、`max_len`、`index`、`unique`、`nullable`、`default` 和 `column`。现在，让我们在模型类之间建立关系：

```py
class Login(db.Entity): 
    _table_ = "login"
    id = PrimaryKey(int)
    … … … … … …
    date_approved = Required(date)
    user_type = Required(int)

    trainers = Optional("Profile_Trainers", reverse="id")
    members = Optional("Profile_Members", reverse="id")
```

给定的 `Login` 类有两个额外的属性，`trainers` 和 `members`，它们分别作为 `Profile_Trainers` 和 `Profile_Members` 模型的引用键。反过来，这些子实体各自有指向 `Login` 模型的类属性，从而建立了一种关系。这些列属性及其引用外键关系必须与物理数据库模式相匹配。以下代码展示了 Pony 的子模型类示例：

```py
class Profile_Trainers(db.Entity):
    _table_ = "profile_trainers"
    id = PrimaryKey("Login", reverse="trainers")
    firstname = Required(str)
    … … … … … …
    tenure = Required(float)
    shift = Required(int)

members = Set("Profile_Members", 
           reverse="trainer_id")
    gclass = Set("Gym_Class", reverse="trainer_id")
class Profile_Members(db.Entity): 
    _table_ = "profile_members"
    id = PrimaryKey("Login", reverse="members")
    firstname = Required(str)
    … … … … … …
trainer_id = Required("Profile_Trainers", 
            reverse="members")
    … … … … … …
```

定义关系属性取决于两个实体之间的关系类型。如果关系类型是一对一，则应将属性定义为 *Optional(parent)*-*Required(child)* 或 *Optional(parent)*-*Optional(child)*。对于一对多，属性应定义为 *Set(parent)*-*Required(child)*。最后，对于多对一，属性必须定义为 *Set(parent)*-*Set(child)*。

`Login` 与 `Profile_Members` 之间有一个一对一的关系，这解释了为什么使用 `Optional` 属性指向 `Profile_Members` 的 `id` 键。在这个关系中，主键始终是 Pony 的引用键。

另一方面，`Profile_Trainers`模型与`Profile_Members`有一个一对一的设置，这解释了为什么前者的`trainer_id`属性使用`Required`指令指向后者的`Set`属性`members`。有时，框架需要通过指令的`reverse`参数进行反向引用。前面的代码也描述了`Profile_Members`和`Gym_Class`模型之间的相同场景，其中`Profile_Members`的`gclass`属性被声明为一个包含成员所有报名健身课程的`Set`集合。在这个关系中，引用键可以是主键，也可以是典型的类属性。以下片段显示了`Gym_Class`模型的蓝图：

```py
class Gym_Class(db.Entity): 
    _table_ = "gym_class"
    id = PrimaryKey(int)
    member_id = Required("Profile_Members", 
         reverse="gclass")
    trainer_id = Required("Profile_Trainers", 
         reverse="gclass")
    approved = Required(int)
db.generate_mapping()
```

与其他 ORM 不同，Pony 需要执行`generate_mapping()`来将所有实体映射到实际表。该方法是`db`实例的一个实例方法，必须出现在模块脚本的最后部分，如前面的片段所示，其中`Gym_Class`是最后定义的 Pony 模型类。所有 Pony 模型类都可以在`/models/data/pony_models.py`脚本文件中找到。

注意，我们可以使用**Pony ORM ER Diagram Editor**手动或数字化创建 Pony 实体，我们可以在[`editor.ponyorm.com/`](https://editor.ponyorm.com/)访问它。该编辑器可以提供免费和商业账户。现在让我们实现 CRUD 事务。

## 实现 CRUD 事务

Pony 中的 CRUD 事务是会话驱动的。但与 SQLAlchemy 不同，其存储库类不需要将`db_session`注入到存储库构造函数中。Pony 中的每个事务在没有`db_session`的情况下都不会工作。以下代码显示了一个实现管理健身会员列表所需所有事务的存储库类：

```py
from pony.orm import db_session, left_join
from models.data.pony_models import Profile_Members, 
            Gym_Class, Profile_Trainers
from datetime import date, time
from typing import List, Dict, Any
from models.requests.members import ProfileMembersReq 
class MemberRepository: 

    def insert_member(self, 
            details:Dict[str, Any]) -> bool: 
        try:
            with db_session:
                Profile_Members(**details)
        except: 
            return False 
        return True
```

在 Pony 中，插入记录意味着使用注入的记录值实例化模型类。一个例子是`insert_member()`，它通过实例化`Profile_Members`模型并注入会员详情来插入个人资料。然而，更新记录的情况则不同，如下面的脚本所示：

```py
    def update_member(self, id:int, 
               details:Dict[str, Any]) -> bool: 
       try:
          with db_session:
            profile = Profile_Members[id]
            profile.id = details["id"]
            … … … … … …
            profile.trainer_id = details["trainer_id"]
       except: 
           return False 
       return True
```

在 Pony 中更新记录，通过`update_member()`脚本实现，意味着通过`id`进行索引检索记录对象。由于 Pony 内置了对 JSON 的支持，检索到的对象会自动转换为可 JSON 化的对象。然后，那些属性的新值会覆盖原有值，因为它们必须更改。这个*UPDATE*事务同样在`db_session`的范围内，因此覆盖后自动刷新记录（s）。

另一方面，存储库类的`delete_member()`方法展示了与*UPDATE*相同的方法，只是在检索对象记录后立即调用`delete()`类方法。以下是这个操作的脚本：

```py
    def delete_member(self, id:int) -> bool: 
        try:
           with db_session: 
               Profile_Members[id].delete()
        except: 
            return False 
        return True
```

删除事务也是 `db_session` 绑定的，因此调用 `delete()` 会自动刷新表。以下代码展示了 Pony 对查询事务的实现：

```py
    def get_all_member(self):
        with db_session:
            members = Profile_Members.select()
            result = [ProfileMembersReq.from_orm(m) 
                 for m in members]
            return result

    def get_member(self, id:int): 
        with db_session:
            login = Login.get(lambda l: l.id == id)
            member = Profile_Members.get(
                lambda m: m.id == login)
            result = ProfileMembersReq.from_orm(member)
        return result
```

`get_member()` 方法使用 `get()` 类方法检索单个记录，该方法在其参数中需要一个 `lambda` 表达式。由于 `Login` 与 `Profile_Members` 之间存在一对一关系，首先，我们必须提取成员的 `Login` 记录，并使用 `login` 对象通过 `Profile_Members` 实体的 `get()` 辅助函数获取记录。这种方法也适用于具有其他实体关系的其他实体。现在，`get_all_member()` 方法使用 `select()` 方法检索结果集。如果检索操作中有约束条件，`select()` 方法也可以使用 `lambda` 表达式。

Pony 模型类具有 `get()` 和 `select()` 方法，这两个方法都返回 FastAPI 无法直接处理的 `Query` 对象。因此，我们需要一个 ORM 友好的 `Pydantic` 模型来从这些 `Query` 对象中提取最终实体。类似于 SQLAlchemy，需要一个具有嵌套 `Config` 类的 `ModelBase` 类来从 `Query` 对象中检索记录。嵌套类必须配置 `orm_mode` 为 `True`。如果涉及关系映射，请求模型还必须声明涉及关系的属性及其对应的子对象转换器。方法转换器，由 Pydantic 的 `@validator` 装饰，将被 Pony 自动调用以解释和验证 `Query` 对象为可 JSON 化的组件，如 `List` 或实体对象。以下代码展示了用于通过列表推导和 `Profile_Member dict` 对象从 `get()` 中提取记录的请求模型：

```py
from typing import List, Any
from pydantic import BaseModel, validator
class ProfileMembersReq(BaseModel): 
    id: Any
    firstname: str
    lastname: str
    age: int
    height: float
    weight: float
    membership_type: str
    trainer_id: Any

    gclass: List

    @validator('gclass', pre=True, 
         allow_reuse=True, check_fields=False)
    def gclass_set_to_list(cls, values):
        return [v.to_dict() for v in values]
    @validator('trainer_id', pre=True, 
         allow_reuse=True, check_fields=False)
    def trainer_object_to_map(cls, values):
        return values.to_dict()

    class Config:
        orm_mode = True
```

`ProfileMembersReq` 中的 `gclass_set_to_list ()` 和 `trainer_object_to_map()` 转换的存在使得数据可以填充到 `gclass` 和 `trainer_id` 属性的子对象中，分别。这些附加功能表明为什么执行 `select()` 已经可以检索 *INNER JOIN* 查询。

要构建 *LEFT JOIN* 查询事务，ORM 有一个内置指令称为 `left_join()`，它通过 Python 生成器提取带有 *LEFT JOIN* 原始对象的 `Query` 对象。以下代码展示了另一个存储库类，展示了 `left_join()` 的使用：

```py
class MemberGymClassRepository:

    def join_member_class(self): 
      with db_session: 
        generator_args = (m for m in Profile_Members 
              for g in m.gclass)
        joins = left_join(tuple_args)        
        result = [ProfileMembersReq.from_orm(m) 
              for m in joins ]
        return result
```

所有存储库类都可以在 `/repository/pony/members.py` 脚本文件中找到。

现在，使 Pony 更快的是它使用了一个 *身份映射*，其中包含从每个查询事务中检索到的所有记录对象。ORM 应用 *身份映射* 设计模式来应用其缓存机制，以使读写执行快速。它只需要内存管理和监控，以避免在复杂和大型应用程序中发生内存泄漏问题。

## 运行存储库事务

由于`db_session`已经由内部管理，因此对于`APIRouter`脚本来运行存储库事务，Pony 不需要额外的要求。存储库类直接在每个 API 中访问和实例化，以访问 CRUD 事务。

## 创建表格

如果表格尚不存在，Pony 可以通过其实体类生成这些表格。当`db`的`generate_mapping()`方法的`create_tables`参数设置为`True`时，此 DDL 事务被启用。

在语法方面，最紧凑和最简单的 ORM 是**Peewee**。

# 使用 Peewee 构建存储库

在不同的 ORM 中，Peewee 在 ORM 功能和 API 方面是最简单和最小的。该框架易于理解和使用；它不是全面的，但它具有直观的 ORM 语法。其优势在于构建和执行查询事务。

Peewee 不是为异步平台设计的，但它可以通过使用它支持的某些与异步相关的库与它们一起工作。为了使 Peewee 与异步框架 FastAPI 一起工作，我们需要至少安装*Python 3.7*。要安装 Peewee，我们需要执行以下命令：

```py
pip install peewee
```

## 安装数据库驱动程序

ORM 需要`psycopg2`作为 PostgreSQL 数据库驱动程序。我们可以使用`pip`来安装它：

```py
pip install psycopg2
```

## 创建数据库连接

为了使 Peewee 与 FastAPI 一起工作，我们必须构建一个多线程机制，其中 Peewee 可以在同一线程上处理多个请求事务，并且每个请求可以使用不同的本地线程同时执行更多操作。这个定制的多线程组件，可以使用`ContextVar`类创建，将 Peewee 连接到 FastAPI 平台。但是，为了使 Peewee 利用这些线程，我们还需要自定义其`_ConnectionState`，使用新创建的线程状态`db_state`。以下代码显示了如何从`db_state`和自定义的`_ConnectionState`派生出：

```py
from peewee import _ConnectionState
from contextvars import ContextVar
db_state_default = {"closed": None, "conn": None, 
         "ctx": None, "transactions": None}
db_state = ContextVar("db_state", 
          default=db_state_default.copy())
class PeeweeConnectionState(_ConnectionState):
    def __init__(self, **kwargs):
        super().__setattr__("_state", db_state)
        super().__init__(**kwargs)
    def __setattr__(self, name, value):
        self._state.get()[name] = value
    def __getattr__(self, name):
        return self._state.get()[name]
```

为了应用前面代码中引用的新的`db_state`和`_ConnectionState`类，即`PeeweeConnectionState`，我们需要通过`Database`类打开数据库连接。Peewee 有几种`Database`类的变体，具体取决于应用程序将选择连接到的数据库类型。由于我们将使用 PostgreSQL，因此`PostgresqlDatabase`是初始化所有必要数据库详情的正确类。建立连接后，`db`实例将有一个指向`PeeweeConnectionState`实例的`_state`属性。以下片段显示了如何使用数据库凭证连接到我们的*健身健身房*数据库的`fcms`：

```py
from peewee import PostgresqlDatabase
db = PostgresqlDatabase(
    'fcms',
    user='postgres',
    password='admin2255',
    host='localhost',
    port=5433, 
)
db._state = PeeweeConnectionState()
```

上述代码还强调，数据库连接的默认状态必须替换为可以与 FastAPI 平台一起工作的非阻塞状态。此配置可以在`/db_config/peewee_connect.py`脚本文件中找到。现在让我们构建 Peewee 的模型层。

## 创建表格和领域层

Peewee 与其他 ORM 不同，它更喜欢根据其模型类自动生成表。Peewee 推荐进行 *逆向工程*，即创建表而不是仅将它们映射到现有表。让应用程序生成表可以减少建立关系和主键的麻烦。这个 ORM 是独特的，因为它有一种“隐含”的方法来创建主键和外键。以下脚本显示了 Peewee 模型类的定义：

```py
from peewee import Model, ForeignKeyField, CharField, 
   IntegerField, FloatField, DateField, TimeField
from db_config.peewee_connect import db
class Signup(Model):
    username = CharField(unique=False, index=False)
    password = CharField(unique=False, index=False)

    class Meta:
      database = db
      db_table = 'signup'
```

我们在模型类中看不到任何主键，因为 Peewee 引擎将在其模式自动生成期间创建它们。物理外键列和模型属性将具有与模型名称相同的名称，以小写形式采用 `modelname_id` 模式。如果我们坚持为模型添加主键，将发生冲突，使 Peewee 无法正常工作。我们必须让 Peewee 从模型类创建物理表以避免这种错误。

所有模型类都从 ORM 的 `Model` 指令继承属性。它还具有如 `IntegerField`、`FloatField`、`DateField` 和 `TimeField` 等列指令，用于定义模型类的列属性。此外，每个域类都有一个嵌套的 `Meta` 类，它注册了对 `database` 和 `db_table` 的引用，它们映射到模型类。我们还可以在此设置的其他属性包括 `primary_key`、`indexes` 和 `constraints`。

在自动生成中唯一的问题是创建表关系。在自动生成之前，将子类的外键属性链接到父类不存在的主键是困难的。例如，以下 `Profile_Trainers` 模型暗示了与 `Login` 类的 *多对一* 关系，这仅由 `ForeignKeyField` 指令的 `trainer` 反向引用属性定义，而不是由 `login_id` 外键定义：

```py
class Profile_Trainers(Model):
login = ForeignKeyField(Login, 
         backref="trainers", unique=True)
    … … … … … …
    shift = IntegerField(unique=False, index=False)

    class Meta:
      database = db
      db_table = 'profile_trainers'
```

自动生成后生成的 `login_id` 列可以在以下屏幕截图中看到：

![图 5.2 – 生成的 profile_trainers 模式![图 5.2 – 生成的 profile_trainers 模式](img/Figure_5.2_B17975.jpg)

图 5.2 – 生成的 profile_trainers 模式

使用 `ForeignKeyField` 指令声明外键属性，它接受至少三个关键参数：

+   父模型名称

+   `backref` 参数，它引用子记录（如果在 *一对一* 关系中）或一组子对象（如果在 *一对多* 或 *多对一* 关系中）

+   `unique` 参数，当设置为 `True` 时表示 *一对一* 关系，否则为 `False`

定义了所有模型，包括它们的关系后，我们需要从 Peewee 的 `db` 实例调用以下方法以进行表映射：

+   使用 `connect()` 建立连接

+   使用 `create_tables()` 来根据其模型类列表进行模式生成

以下脚本显示了类定义的快照，包括调用两个 `db` 方法：

```py
class Login(Model): 
    username = CharField(unique=False, index=False)
    … … … … … …
    user_type = IntegerField(unique=False, index=False)

    class Meta:
      database = db
      db_table = 'login'

class Gym_Class(Model): 
    member = ForeignKeyField(Profile_Members, 
          backref="members")
    trainer = ForeignKeyField(Profile_Trainers, 
          backref="trainers")
    approved = IntegerField(unique=False, index=False)

    class Meta:
      database = db
      db_table = 'gym_class'
db.connect()
db.create_tables([Signup, Login, Profile_Members, 
     Profile_Trainers, Attendance_Member, Gym_Class],
           safe=True)
```

如我们所见，我们需要将 `create_tables()` 的 `safe` 参数设置为 `True`，这样 Peewee 就会在应用程序的初始服务器启动期间只执行一次模式自动生成。所有 Peewee ORM 的模型类都可以在 `/models/data/peewee_models.py` 脚本文件中找到。现在，让我们实现仓库层。

## 实现 CRUD 事务

在 Peewee ORM 中为应用程序创建异步连接并构建模型层是棘手的，但实现其仓库层是直接的。所有方法操作完全源自其模型类。例如，以下代码片段中显示的 `insert_login()` 方法展示了 `Login` 的 `create()` 静态方法如何接受记录插入的登录详情：

```py
from typing import Dict, List, Any
from models.data.peewee_models import Login, 
   Profile_Trainers, Gym_Class, Profile_Members
from datetime import date
class LoginRepository:

    def insert_login(self, id:int, user:str, passwd:str, 
          approved:date, type:int) -> bool: 
        try:
            Login.create(id=id, username=user, 
                password=passwd, date_approved=approved, 
                user_type=type)
        except Exception as e: 
           return False 
        return True
```

此方法可以被重新实现以执行批量插入，但 Peewee 有另一种通过其 `insert_many()` 类方法追求多个插入的方法。使用 `insert_many()` 需要更精确的列细节来映射多个模式值。它还需要调用 `execute()` 方法来执行所有批量插入并在之后关闭操作。

同样，`update()` 类方法在通过 `id` 主键过滤需要更新的记录后需要调用 `execute()` 方法。以下代码片段展示了这一点：

```py
    def update_login(self, id:int, 
              details:Dict[str, Any]) -> bool: 
       try:
           query = Login.update(**details).
                  where(Login.id == id)
           query.execute()
       except: 
           return False 
       return True
```

当涉及到记录删除时，`delete_login()` 展示了简单的方法——即通过使用 `delete_by_id()`。但 ORM 有另一种方法，即使用 `get()` 类方法检索记录对象——例如，`Login.get(Login.id == id)`——然后通过记录对象的 `delete_instance()` 实例方法最终删除记录。以下 `delete_login()` 事务展示了如何利用 `delete_by_id()` 类方法：

```py
    def delete_login(self, id:int) -> bool: 
        try:
           query = Login.delete_by_id(id)
        except: 
            return False 
        return True
```

以下脚本，针对 `get_all_login()` 和 `get_login()`，突出了 Peewee 如何从数据库中检索记录。Peewee 使用其 `get()` 类方法通过主键检索单个记录；在之前的代码片段中，同样的方法被应用于其 *UPDATE* 事务。同样，Peewee 使用类方法提取多个记录，但这次它使用的是 `select()` 方法。结果对象不能被 FastAPI 编码，除非它包含在 `List` 集合中，该集合将数据行序列化为可序列化为 JSON 的对象列表：

```py
    def get_all_login(self):
        return list(Login.select())

    def get_login(self, id:int): 
        return Login.get(Login.id == id)
```

另一方面，以下仓库类展示了如何使用其 `join()` 方法创建 *JOIN* 查询：

```py
from peewee import JOIN
class LoginTrainersRepository:

    def join_login_trainers(self): 
        return list(Profile_Trainers.
          select(Profile_Trainers, Login).join(Login))
class MemberGymClassesRepository:
    def outer_join_member_gym(self): 
        return list(Profile_Members.
          select(Profile_Members,Gym_Class).join(Gym_Class, 
                    join_type=JOIN.LEFT_OUTER))
```

`LoginTrainersRepository` 的 `join_login_trainers()` 方法构建了 `Profile_Trainers` 和 `Login` 对象的 *INNER JOIN* 查询。在 `Profile_Trainers` 对象的 `select()` 指令中指定的最左侧模型是父模型类型，其后是其一对一关系中的子模型类。`select()` 指令发出带有模型类类型的 `join()` 方法，这表示属于查询右侧的记录类型。*ON* 条件和外键约束是可选的，但可以通过添加 `join()` 构造的 `on` 和 `join_type` 属性来显式声明。此查询的一个示例是 `MemberGymClassesRepository` 的 `outer_join_member_gym()`，它使用 `join_type` 属性的 `LEFT_OUTER` 选项实现了 `Profile_Members` 和 `Gym_Class` 的 *LEFT OUTER JOIN*。

在 Peewee 中进行连接操作也需要使用 `list()` 集合来序列化检索到的记录。所有 Peewee 的存储库类都可以在 `/repository/peewee/login.py` 脚本中找到。

## 运行 CRUD 事务

由于 Peewee 的数据库连接是在模型层设置的，因此 `APIRouter` 或 `FastAPI` 运行 CRUD 事务时不需要额外的要求。API 服务可以轻松访问所有存储库类，而无需从 `db` 实例调用方法或指令。

到目前为止，我们已经尝试了流行的 ORM 来将关系数据库集成到 FastAPI 框架中。如果应用 ORM 对于微服务架构来说不够用，我们可以利用一些可以进一步优化 CRUD 性能的设计模式，例如 **CQRS**。

# 应用 CQRS 设计模式

CQRS 是一种微服务设计模式，负责将查询事务（*读取*）与插入、更新和删除操作（*写入*）分离。这两个组的分离减少了对这些事务的访问耦合度，从而提供了更少的流量和更快的性能，尤其是在应用程序变得复杂时。此外，此设计模式在 API 服务和存储库层之间创建了一个松散耦合的特性，如果存储库层有多个轮换和变更，这将给我们带来优势。

## 定义处理程序接口

要实现 CQRS，我们需要创建定义查询和命令事务的两个接口。以下代码显示了将识别 `Profile_Trainers` 的 *读取* 和 *写入* 事务的接口：

```py
class IQueryHandler: 
    pass 
class ICommandHandler: 
    pass
```

这里，`IQueryHandler` 和 `ICommandHandler` 是非正式接口，因为 Python 没有实际的接口定义。

## 创建命令和查询类

接下来，我们需要实现命令和查询类。命令作为执行*写入*事务的指令。它还携带执行后的结果状态。另一方面，查询指示*读取*事务从数据库检索记录并在之后包含结果。这两个组件都是具有*getter/setter*属性的序列化类。以下代码展示了`ProfileTrainerCommand`的脚本，它使用 Python 的`@property`属性来存储*INSERT*执行的状 态：

```py
from typing import Dict, Any
class ProfileTrainerCommand: 

    def __init__(self): 
        self._details:Dict[str,Any] = dict()

    @property
    def details(self):
        return self._details
    @details.setter
    def details(self, details):
        self._details = details
```

`details`属性将存储需要持久化的训练员资料记录的所有列值。

以下脚本实现了一个示例*查询*类：

```py
class ProfileTrainerListQuery: 

    def __init__(self): 
        self._records:List[Profile_Trainers] = list()

    @property
    def records(self):
        return self._records
    @records.setter
    def records(self, records):
        self._records = records
```

`ProfileTrainerListQuery`的构造函数准备了一个字典对象，该对象将在查询事务执行后包含所有检索到的记录。

## 创建命令和查询处理程序

我们将使用之前的接口来定义命令和查询处理程序。请注意，命令处理程序访问和执行存储库以执行*写入*事务，而查询处理程序处理*读取*事务。这些处理程序作为 API 服务和存储库层之间的外观。以下代码展示了`AddTrainerCommandHandler`的脚本，它管理训练员资料的*INSERT*事务：

```py
from cqrs.handlers import ICommandHandler
from repository.gino.trainers import TrainerRepository
from cqrs.commands import ProfileTrainerCommand
class AddTrainerCommandHandler(ICommandHandler): 

    def __init__(self): 
        self.repo:TrainerRepository = TrainerRepository()

    async def handle(self, 
             command:ProfileTrainerCommand) -> bool:
        result = await self.repo.
               insert_trainer(command.details)
        return result
```

处理程序依赖于`ProfileTrainerCommand`来获取对异步执行其`handle()`方法至关重要的记录值。

以下脚本展示了查询处理程序的示例实现：

```py
class ListTrainerQueryHandler(IQueryHandler): 
    def __init__(self): 
        self.repo:TrainerRepository = TrainerRepository()
        self.query:ProfileTrainerListQuery = 
             ProfileTrainerListQuery()

    async def handle(self) -> ProfileTrainerListQuery:
        data = await self.repo.get_all_member();
        self.query.records = data
        return self.query
```

查询处理程序将它们的*查询*返回给服务，而不是实际的值。`ListTrainerQueryHandler`的`handle()`方法返回`ProfileTrainerListQuery`，其中包含从*读取*事务中检索到的记录列表。这种机制是应用 CQRS 到微服务的主要目标之一。

## 访问处理程序

CQRS（Command Query Responsibility Segregation）除了管理*读取*和*写入*执行之间的摩擦外，不允许 API 服务直接与 CRUD 事务的执行进行交互。此外，它通过仅分配特定服务所需的处理程序来简化并简化 CRUD 事务的访问。

以下脚本展示了`AddTrainerCommand`如何仅直接关联到`add_trainer()`服务，以及`LisTrainerQueryHandler`如何仅直接关联到`list_trainers()`服务：

```py
from cqrs.commands import ProfileTrainerCommand
from cqrs.queries import ProfileTrainerListQuery
from cqrs.trainers.command.create_handlers import 
      AddTrainerCommandHandler
from cqrs.trainers.query.query_handlers import 
      ListTrainerQueryHandler
router = APIRouter(dependencies=[Depends(get_db)])
@router.post("/trainer/add" )
async def add_trainer(req: ProfileTrainersReq): 
    handler = AddTrainerCommandHandler()
    mem_profile = dict()
    mem_profile["id"] = req.id
    … … … … … …
    mem_profile["shift"] = req.shift
    command = ProfileTrainerCommand()
    command.details = mem_profile
    result = await handler.handle(command)
    if result == True: 
        return req 
    else: 
        return JSONResponse(content={'message':'create 
          trainer profile problem encountered'}, 
            status_code=500)
@router.get("/trainer/list")
async def list_trainers(): 
    handler = ListTrainerQueryHandler()
    query:ProfileTrainerListQuery = await handler.handle() 
    return query.records
```

我们可以通过 CQRS 在`APIRouter`中识别出频繁访问的事务。这有助于我们找到哪些事务需要性能调整和关注，这可以帮助我们在访问量增加时避免性能问题。当涉及到增强和升级时，设计模式可以帮助开发者找到优先考虑的领域，因为仓库层中的方面是分离的。通常，当业务流程需要重整时，它为应用程序提供了灵活性。所有与 CQRS 相关的脚本都可以在`/cqrs/`项目文件夹中找到。

# 摘要

对于任何应用程序，应用 ORM 总是有利有弊。它可能会因为过多的配置和组件层而使应用程序膨胀，如果管理不当，甚至可能减慢应用程序的速度。但总的来说，ORM 可以通过使用其 API 简化结构，消除不重要的重复 SQL 脚本，帮助优化查询开发。与使用`psycopg2`的`cursor`相比，ORM 可以减少软件开发的时间和成本。

在本章中，使用了四个 Python ORM 进行研究和实验，以帮助 FastAPI 创建其仓库层。首先，是*SQLAlchemy*，它提供了一种创建标准异步数据持久性和查询操作的模板化方法。然后是*GINO*，它使用 AsyncIO 环境通过其便捷的语法实现异步 CRUD 事务。还有*Pony*，它是所展示 ORM 中最 Pythonic 的，因为它使用纯 Python 代码构建其仓库事务。最后是*Peewee*，以其简洁的语法而闻名，但在异步数据库连接和 CRUD 事务的复杂组合方面较为棘手。每个 ORM 都有其优势和劣势，但所有 ORM 都提供了一个逻辑解决方案，而不是应用蛮力和原生 SQL。

如果 ORM 需要微调，我们可以通过使用与数据相关的模式，如 CQRS，来添加一些优化程度，这有助于最小化“读取”和“写入”CRUD 事务之间的摩擦。

本章强调了 FastAPI 在利用 ORM 连接到关系型数据库（如 PostgreSQL）时的灵活性。但如果我们使用如 MongoDB 这样的 NoSQL 数据库来存储信息呢？FastAPI 在执行对 MongoDB 的 CRUD 操作时，性能是否会保持相同水平？下一章将讨论将 FastAPI 集成到 MongoDB 的各种解决方案。
