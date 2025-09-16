

# 第八章：SQL 和数据建模

到目前为止，我们已经探讨了 React，它是前端技术堆栈中的一个关键库。现在，我们将探索后端开发的世界，从**结构化查询语言**（**SQL**）和数据建模开始。

SQL 和数据建模是任何后端开发人员的关键技能，在后端开发之旅中从这些技能开始将为你打下构建稳健和可扩展的 Web 应用的基础。SQL 是一种标准语言，用于管理关系数据库中的数据，这些数据库本质上用于存储结构化数据。

掌握 SQL 将帮助你编写优化的 SQL 查询，从而提高应用程序的性能并减少服务器负载。数据建模将帮助你设计一个反映应用程序将处理的数据的数据库模式。数据建模可以帮助你避免性能问题、可维护性问题以及其他在数据库工作中可能出现的常见问题。

我们将深入探讨关系数据库以及数据库表如何相互关联。我们将检查 SQL，这是许多数据库管理系统使用的标准语言，包括流行的开源**关系数据库管理系统**（**RDBMS**）**PostgreSQL**。

对关系数据库有深入的理解，并了解数据库关系如何工作，将帮助你设计可扩展和可维护的数据结构。我们将不遗余力，从设置 PostgreSQL 开始，探索 Flask 应用程序如何与 PostgreSQL 通信。

我们还将深入讨论 SQLAlchemy 通过提供一个允许使用 Python 对象与数据库进行 SQL 交互的接口，来处理各种关系数据库方言的能力。**SQLAlchemy** 是一个工业级对象关系映射器，它为与各种关系数据库方言（包括 PostgreSQL）的交互提供了一个强大的接口。

SQLAlchemy 使得编写复杂的数据库查询和管理数据库事务变得更容易。我们将检查您如何提出数据模型，并将数据从 Flask 应用程序发送到数据库。

除了数据建模和 SQL，迁移也是后端开发的一个关键方面。我们将通过 Alembic 来检查迁移，作为跟踪和更新数据库结构的一种方式。**Alembic** 是一个迁移工具，它提供了一种可靠的方式来跟踪和更新数据库结构，对于 Python 开发人员来说是一个必不可少的工具。

在本章中，我们将涵盖以下主题：

+   什么是关系数据模型？

+   探索不同的数据库关系

+   设置 PostgreSQL

+   理解 Flask 应用程序的数据库概念

+   理解 SQLAlchemy ORM 基础

+   为演讲者会议 Web 应用程序建模数据

+   从 Flask 应用程序向 PostgreSQL 数据库发送数据

+   使用 Alembic 进行迁移

# 技术要求

本章的完整代码可在 GitHub 上找到：[`github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter08`](https://github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter08)。

# 什么是关系数据模型？

**关系数据模型**是一种概念方法，用于将数据库表示为一组关系。大多数 Web 应用程序都是高度数据驱动的。开发者必须处理代码级别的数据存储，在数据结构的情况下，或者找到一种方法在关系型数据库管理系统（如 PostgreSQL 或 MySQL）中持久化存储数据。

在关系型数据库管理系统中，你可以将表格称为关系。因此，关系模型将数据表示为关系或表格的集合。进一步分解数据库结构，你将得到构成表格的行和列。然后，你有一个记录，它由行和列的组合组成。

让我们看看一个名为`customers`的假设表格，它以清晰的方式描述了典型表格的结构：

| `Id` | `firstname` | `lastname` | `email` | `phone` |
| --- | --- | --- | --- | --- |
| 1 | Joel | Doe | Joel@admin.com | 404-228-5487 |
| 2 | Edward | Spinster | Edward@admin.com | 403-268-6486 |
| 3 | Mabel | Emmanuel | Mabel@admin.com | 402-248-4484 |

表 8.1 – 展示客户信息的表格

在前面的`customers`表格中，我们有五列和三行。表格中的每一行称为一个元组。列标题如`Id`、`firstname`、`lastname`、`email`和`phone`被称为属性或字段。在数据库中，表格被创建来高效地存储数据。每个表格通常代表一个业务实体或对象，例如演讲者、场地、主题、客户、产品、订单等。

为了清晰起见，业务实体代表我们打算在应用程序业务数据模型中封装的事物，包括所有规则、关系和能够在数据库中持久化的能力。在前面的表格中，我们有一个客户业务实体，具有以下属性 – `Id`、`firstname`、`lastname`、`email`和`phone`。通常情况下，你会在你的 Web 应用程序中拥有不止一个表格。

预计你需要能够使用主键在你的数据库中的不同表格或关系之间建立联系。主键是表格中整行唯一的标识符，指代一个或多个列。如果有多个列用于主键，那么主键列的集合被称为复合键。在我们的例子中，客户表中的`Id`可以设置为主键。

此外，数据关系中的另一个概念是外键。外键指的是另一个（外部）表格中的主键。外键用于映射表格或关系之间的关系。表格关系帮助你高效地存储需要存储的数据，并在需要时更容易访问相关数据。

数据库设计中存在许多关系——*一对一*、*一对多*和*多对多*。例如，数据库中的相关表可以帮助你找出客户下的订单，了解有多少会议参与者注册了每个主题，等等。

在下一节中，我们将广泛讨论数据库中的关系以及它们在 Web 应用程序架构设计中的应用。

# 探索不同的数据库关系

在客户端-服务器模型中，数据库位于基础设施的服务器端。数据库是任何生产级 Web 应用程序收集和存储应用程序数据的核心。理解数据库中存在的关系对于组织、管理和从数据库中检索有用数据至关重要。

如前所述，数据库中存在三种类型的关系——*一对一*、*一对多*和*多对多*。我们将从深入研究一对一关系的概念开始。

## 一对一（1:1）关系

`speakers`和`speakerExtras`。

![图 8.1 – 显示一对一关系的实体关系图](img/Figure_8.1_B18554.jpg)

图 8.1 – 显示一对一关系的实体关系图

前面的图示展示了`speakers`表和`speakerExtras`表。`speakers`表详细说明了演讲者对象的基本必要信息，而`speakerExtras`表则展示了可以添加到演讲者信息中的可选信息。`speakers`表中的`speaker_id`是主键，可以用来唯一标识一个演讲者记录。

在`speakerExtras`表中，增加了一个名为`speaker_id`的列。`speakerExtras`表中的这个`speaker_id`列是一个具有唯一属性的外键。这个`speaker_id`列被用作对`speakers`表中的`speaker_id`的引用键，以在`speakerExtras`表和`speakers`表之间形成一个一对一关系。

大多数情况下，一对一关系的数据模型实际上可以合并。合并一对多关系的数据模型通常意味着将表合并为一个单一的表。这通常是在两个表有强烈的联系，且表中的数据并不复杂到需要单独的表时进行的。

然而，在某些情况下，数据库设计的要求可能需要一个单独的表来处理可能的可选信息，因此，而不是在表中留有空列，你可以创建一个不同的表来处理这些可选信息，以获得更好的数据库性能。

让我们来看看另一种数据库关系，一对多。

## 一对多（1:M）关系

数据模型中的一对多关系解释了两个表之间的关系，其中一个表中的一行可以引用另一个表中的一行或多行。一对多关系是一种父子关系，其中子记录只能有一个父记录。

大多数时候，父记录在另一个表的行中有多于一个子记录。然而，在现实场景中，我们可能会遇到一个父记录没有任何子记录的情况。没有子记录的父记录意味着子表中的外键列将为该父记录为空。

例如，考虑一个商店的数据库设计，每个客户可以有多个订单。如果一个新客户记录被创建，但还没有下订单，那么将有一个没有对应订单记录的客户记录。在这种情况下，客户记录是父记录，订单记录是子记录。

此外，在某些情况下，一对多关系也允许另一个表中的单个记录。存在一些情况，一对多关系可以被限制为功能上作为一对一关系。例如，你可以在子表的外键列上添加唯一约束，确保父表中的每个记录最多与子表中的一个记录相关联。

这将有效地将关系约束为作为一对一关系，尽管底层的数据模型是一对多关系。这种一对多关系几乎类似于一对一关系，但有一个唯一约束的细微差别。为什么这种一对多关系在数据建模中很重要？

就像其他数据库关系一样，一对多关系是关系数据库设计的基本构建块，对于组织、管理和分析复杂数据结构至关重要。一对多关系通过确保子表中的每个记录都与父表中的一个有效记录相关联，强制执行数据的引用完整性。这种方法有助于防止由孤立记录或对不存在记录的引用引起的数据不一致性和错误。

让我们来看一个一对多关系的例子。想象一下，在我们的数据库中有`customers`（客户）和`orders`（订单）表。一个客户在一段时间内可能有很多 1:M 的订单。

一个企业可能希望保留这个记录。例如，客户往往有不同的订单；特定的`orderID`不能属于多个客户。每个客户的订单都是唯一的。以下图展示了客户和订单之间的一对多关系。

![图 8.2 – 显示一对多关系的实体关系图](img/Figure_8.2_B18554.jpg)

图 8.2 – 显示一对多关系的实体关系图

在前面的图中，`orders` 表中的 `customer_id` 代表外键，是两个实体——`customers` 和 `orders` 之间的主要链接因素。如果你想了解某个特定客户有多少订单，你只需要编写一个 `JOIN` 查询，查找 `customers` 和 `orders` 表，并检查 `orders` 表中作为参考键的 `customer_id` 的出现情况。

在你的网络应用程序开发数据库设计中，你将遇到更多这种数据关系，因为它是最常用的。接下来，我们将看看另一种数据库关系，多对多。

## 多对多（M:M）关系

数据建模中的 **多对多关系** 简单来说是指一个表中的多条记录与另一个表中的多条记录相关联的数据关系。以会议数据库设计为例，你可能会有以下场景：

+   一个参与者可以注册多个会议

+   一个会议可以有许多注册的参与者

这意味着一个参与者可以参加多个会议，一个会议也有许多参与者。因此，会议参与者和会议之间存在多对多关系。

有趣的是，与一对一关系和一对多关系不同，你不能只用两个表来建模多对多关系。在多对多关系建模中，需要一个第三张连接表，其中包含两个表的主键值添加到连接表中。

让我们检查会议参与者和他们注册的会议的多对多关系的 **实体关系图**（**ER**）：

![图 8.3 – 显示多对多关系的实体关系图](img/Figure_8.3_B18554.jpg)

图 8.3 – 显示多对多关系的实体关系图

让我们分解前面的 ER 图，以便更好地理解多对多关系。我们有三张表，如上图所示——`attendees`、`enrollemnt_tbl` 和 `conf_sessions`。

`attendees` 表包含会议参与者的记录。同样，`conf_sessions` 表包含会议的记录。然后你有一个连接表 `enrollment_tbl`，它在技术上与两个表各自形成一个一对一关系。

`enrollment_tbl` 包含参与者和 `conf_sessions` 表的主键作为外键。通过 `enrollment_tbl`，我们可以查询参与者和 `conf_sessions` 表的相关记录。在这种情况下，我们可以访问特定参与者参加的所有会议。

接下来，我们将深入探讨数据库设置，使用 SQLAlchemy 和 Alembic 来处理网络应用程序的数据库部分。

# 设置 PostgreSQL、SQLAlchemy 和 Alembic

我们将首先设置后端数据库所需的数据库工具。PostgreSQL 是一个流行的免费开源关系型数据库管理系统。它与存在的其他 SQL 数据库方言类似 – 例如，MySQL、MariaDB 和 Oracle。此数据库可以用于存储任何 Web 应用的数据。PostgreSQL 具有企业级功能，使其强大、可扩展和可靠。

我们还将设置 SQLAlchemy，一个用于`读取`、`插入`、`更新`和`删除`的工具，而不是直接编写 SQL 查询。最后，在本节中，我们将设置 Alembic 以处理数据库迁移。

## 设置 PostgreSQL

要在本地机器上开始使用 PostgreSQL，请从 [`www.postgresql.org/download/`](https://www.postgresql.org/download/) 下载它并选择您的操作系统包。

![图 8.4 – 一张显示 PostgreSQL 下载页面的图表](img/Figure_8.4_B18554.jpg)

图 8.4 – 一张显示 PostgreSQL 下载页面的图表

按照说明运行安装向导以设置数据库。在 PostgreSQL 安装过程中，您将被提示输入超级用户密码。非常重要，您需要保留您输入的超级用户密码，因为这将用于登录默认的 PostgreSQL 数据库。一旦安装完成，请登录到数据库。

打开您机器上的命令终端并输入以下命令：

```py
$     psql -d postgres -U postgres
```

`psql` 从终端调用与 PostgreSQL 的连接。然后，您可以使用 `-d` 选项选择您想要访问的数据库，并使用 `-U` 选择具有数据库访问权限的用户。

注意

如果命令终端回复“`psql`不是内部或外部命令”，您可能需要将 Postgres `bin/ (C:\Program Files\PostgreSQL\14\bin)` 添加到环境变量的系统路径中。

然后，输入安装期间为超级用户创建的密码。

以下屏幕截图显示了尝试登录 PostgreSQL 时的终端。

![图 8.5 – 一张显示如何从终端访问 PostgreSQL 的屏幕截图](img/Figure_8.5_B18554.jpg)

图 8.5 – 一张显示如何从终端访问 PostgreSQL 的屏幕截图

总是创建一个与安装期间创建的超级用户不同的新用户在 Postgres 上是一个最佳实践。现在您已使用默认的 Postgres 用户登录，让我们创建另一个名为 `admin` 的用户角色，并为其分配密码 `admin123`：

```py
CREATE ROLE admin WITH LOGIN PASSWORD 'admin123';
```

我们现在将给新用户 `admin` 授予权限以创建数据库：

```py
ALTER ROLE admin CREATEDB;
```

在设置好所有这些之后，从 Postgres 用户注销并作为 `admin` 登录。要注销，请输入 `\q` 命令。

现在，以 `admin` 用户身份登录：

```py
$    psql -d postgres -U admin
```

输入管理员密码并登录。登录后，创建一个名为 `bizza` 的新数据库：

```py
$    CREATE DATABASE bizza;
```

要连接到新创建的数据库，请运行以下命令：

```py
$    \connect bizza
```

以下屏幕截图显示了如何在 PostgreSQL 中连接以创建数据库：

![图 8.6 – 显示连接到 bizza 数据库的屏幕截图](img/Figure_8.6_B18554.jpg)

图 8.6 – 显示连接到 bizza 数据库的屏幕截图

欢呼！我们已经成功设置了 PostgreSQL。现在，让我们深入了解设置 SQLAlchemy。

注意

角色名称，`admin`，和密码，`admin123`，可以是您想要的任何名称；它们不必按照建议命名。

## 设置 SQLAlchemy

为 Flask 应用程序设置 SQLAlchemy 简单直接。要在您的项目中设置 SQLAlchemy，请在终端中运行以下命令，直接在您的项目 `root` 目录内。在运行安装命令之前，请确保您的虚拟环境已激活：

```py
pip install SQLAlchemy
```

注意

SQLAlchemy 是一个 Python SQL 工具包和 ORM 库，它提供了一套高级 API，用于与关系数据库交互。

此外，我们将使用一个名为 **Flask-SQLAlchemy** 的 Flask 扩展。让我们在终端中运行以下命令来安装它：

```py
pip install flask-sqlalchemy
```

注意

Flask-SQLAlchemy 扩展为 SQLAlchemy 提供了一个包装器，使得在 Flask 应用程序中使用 SQLAlchemy 更加容易。Flask-SQLAlchemy 提供了额外的功能，例如自动会话处理、与 Flask 的应用程序上下文集成以及支持 Flask 特定的配置选项。

在本章的后续部分，我们将深入讨论 SQLAlchemy，*理解 SQLAlchemy ORM 基础*。

简而言之，让我们演示如何无缝地将 SQLAlchemy 集成到您的 Flask 应用程序中。集成 SQLAlchemy 的过程如下：

1.  创建一个新的目录，并将其命名为 `sqlalchemy`。

1.  然后，按照 *第一章* 的 *使用 Flask 设置开发环境* 部分的 Flask 项目设置进行操作。第一章。

1.  确保您的开发虚拟环境已激活。

1.  在终端中运行以下命令：

    ```py
    pip install flaskpip install flask-sqlalchemy
    ```

1.  在 `sqlalchemy` 目录中创建一个名为 `app.py` 的文件，并添加以下代码片段：

    ```py
    from flask import Flask, render_templatefrom flask_sqlalchemy import SQLAlchemyapp = Flask(__name__, template_folder='templates')app.config["SQLALCHEMY_DATABASE_URI"] =    "sqlite:///bizza.db"db = SQLAlchemy(app)@app.route('/users')def get_users():    users = User.query.all()    return render_template('users.html', users=users)class User(db.Model):__tablename__= "users"    id = db.Column(db.Integer, primary_key=True)    username = db.Column(db.String(55), unique=True)    name = db.Column(db.String(55), unique=False)    email = db.Column(db.String(100), unique=True)    def __repr__(self):        return '<User {}>'.format(self.username)
    ```

    上述代码设置了一个基本的 Flask 应用程序，该应用程序使用 SQLAlchemy 进行数据库操作。让我们解释一下上述代码中发生了什么：

    +   `from flask import Flask`: 这从 Flask 包中导入 `Flask` 类。

    +   `from flask_sqlalchemy import SQLAlchemy`: 这从 Flask-SQLAlchemy 包中导入 `SQLAlchemy` 类。

    +   `app = Flask(__name__, template_folder="templates")`: 这将创建一个新的 `Flask` 应用程序实例。`template_folder` 被添加以确保 Flask 能够找到 `templates` 目录。`App.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///bizza.db"`: 这设置了 SQLite 数据库的数据库 URI。在这种情况下，数据库文件名为 `bizza.db`，位于 Flask 应用程序文件 `app.py` 的 `instance` 目录中。我们在这个例子中使用 `SQLite`。您可以使用任何数据库方言 – MySQL、Oracle、PostgreSQL 等。

    +   `db = SQLAlchemy(app)`: 这将创建一个新的 `SQLAlchemy` 实例，并将其绑定到 Flask 应用程序。

    +   然后，我们有一个 Flask 路由处理器，它响应`/users` URL 端点。当用户访问`/users` URL 时，Flask 将执行`get_users`函数，从数据库中检索所有用户，并渲染`users.html`模板，将用户变量传递给模板以进行显示。

    +   最后，我们有包含四个列的`User`模型——`id`、`username`、`name`和`email`。每一列代表相应数据库表中的一个字段，并指定其数据类型和可选约束，如唯一性。这代表了我们在 Flask 中使用 SQLAlchemy 定义我们的模型作为 Python 类，指定它们的属性和关系，并执行数据库操作的方式。

    现在，让我们在`sqlalchemy`目录的终端中创建`users`表。

1.  在终端中输入以下命令：

    ```py
    users table with sample data with python sample_data.py. The sample_data.py is located inside the project root directory in GitHub.
    ```

1.  然后，输入`flask run`命令，并访问`http://127.0.0.1:5000/users`来查看数据库中的用户列表。

    以下是一个屏幕截图，展示了你如何使用 SQLAlchemy 定义和检索用户数据。

![图 8.7 – 展示如何使用 SQLAlchemy 进行数据库操作的屏幕截图](img/Figure_8.7_B18554.jpg)

图 8.7 – 展示如何使用 SQLAlchemy 进行数据库操作的屏幕截图

通过这种设置，我们展示了你如何定义 SQLAlchemy 模型，并使用 Python 对象在 Flask 应用程序中与数据库交互。

接下来，我们将深入了解如何设置 Alembic，这是一个数据库迁移工具。

## 设置 Alembic

`pip install alembic`或从另一个迁移包`Flask-Migrate`安装它，使用`pip install flask-migrate`。`Flask-Migrate`依赖于 Alembic 进行数据库迁移，我们将在这里将其作为数据库迁移的首选工具。

`Flask-Migrate`迁移工具允许你跟踪数据库的变化。迁移工具带有使用迁移脚本管理数据库结构和操作的能力。迁移工具 Alembic 的本质是简化迁移 SQL 脚本的自动生成。

在本章的最后部分，我们将深入讨论迁移，并提供一些实现细节。接下来，我们将讨论如何从 Flask 网络应用程序与数据库交互。

# 理解 Flask 应用程序的数据库概念

现在我们已经设置了数据库并使用终端连接到它，对一些数据库概念有一个扎实的理解对于能够设置能够收集、存储和检索用户数据的后端服务至关重要。

大多数现代网络应用都有一个数据库来存储用户数据。作为一名全栈网络开发者，你的部分责任是能够设置后端服务来收集、存储和检索用户数据。我们将在不久的将来深入探讨从 Flask 应用程序与数据库的交互，但在那之前，有一些数据库概念你需要理解。

让我们快速概述以下数据库概念。

## 关系型数据库管理系统（RDBMS）

在生产环境中与数据库工作时，您需要一个关系型数据库管理系统（RDBMS）。RDBMS 是一种软件包，允许您与数据库交互。RDBMS 软件允许您定义、操作、检索和管理数据库中的数据。

您拥有数据库功能，可以用来管理数据库中的数据和数据库本身的架构。市场上有很多不同的关系型数据库管理系统（RDBMS）版本——MySQL、PostgreSQL、Oracle、MsSQL 和 MariaDB——但我们将要使用的是称为 PostgreSQL 的。

让我们来看看另一个与数据库交互所需的相关工具，称为**数据库应用程序编程接口**（**DB-API**）。

## DB-APIs

DB-API 是一个**应用程序编程接口**（**API**），它允许编程语言或 Web 服务器框架与数据库服务器之间进行通信，使用 TCP/IP 等协议。Python 是一种我们与 Flask 框架一起使用的语言，它使用一个名为 psycopg2 的库，允许 Python 模块与 PostgreSQL 数据库服务器交互。

DB-API 充当数据库适配器，一个 Web 接口，允许 Web 服务器编程语言（在我们的例子中是 Python）在数据库上运行 SQL 查询，使用 psycopg2 作为 DB-API 的库。

对于每种编程语言或服务器框架以及我们拥有的 SQL 数据库的各种方言，都存在不同类型的 DB-API。例如，我们有 psycopg2 用于 PostgreSQL，`mysql-python`或`PyMySQL`用于 MySQL，`adodbapi`或`pymssql`用于 MS SQL Server，以及`mxODBC`或`pyodb`用于 Oracle。

PEP 248 DB-API 版本 1.0 的文档指定了 DB-API 的目标：

“这个 API 被定义为鼓励用于访问数据库的 Python 模块之间的相似性。通过这样做，我们希望实现一致性，从而使得模块更容易理解，代码在数据库之间更通用，并且从 Python 扩展数据库连接的范围。” ([`www.python.org/dev/peps/pep-0248/`](http://www.python.org/dev/peps/pep-0248/))

从本质上讲，DB-API 提供了一种低级方式来编写 SQL 查询语句，使得与不同数据库的交互始终简单且容易。

## 客户端-服务器模型交互

客户端-服务器模型交互是一种通信范式，其中客户端通过网络从服务器请求资源或服务。在这个模型中，客户端向服务器发起请求，服务器响应请求提供所需的服务或数据，从而形成一个客户端-服务器交互，这是各种应用程序和网络通信的基础。

在客户端-服务器模型交互中，浏览器充当客户端，Web 服务器充当服务器。当最终用户从服务器请求资源或网页时，请求通过浏览器（客户端端）在 Web 服务器（服务器端）上通过 HTTP 协议进行。这种相同的客户端-服务器架构也可以用于最终用户请求包含数据库数据的资源或网页。

当浏览器发起一个需要从数据库获取数据的请求时，Web 服务器接收请求并与数据库服务器建立连接。Web 服务器成为客户端，数据库服务器成为服务器，从而完成从后端基础设施的客户端-服务器模型（[`www.python.org/dev/peps/pep-0248/`](http://www.python.org/dev/peps/pep-0248/)）。

在本章的后面部分，我们将创建表格以全面了解如何与数据库交互。以下图示展示了客户端-服务器交互。

![图 8.8 – 展示客户端-服务器交互](img/Figure_8.8_B18554.jpg)

图 8.8 – 展示客户端-服务器交互的图示

在前面的图示中，客户端通过向服务器发送请求来启动交互。在基于 Web 的应用程序中，请求可以是 HTTP 请求。服务器接收请求，对其进行解释，并执行必要的操作以生成响应。

一旦响应准备就绪，服务器将其发送回客户端。客户端接收响应并对其进行处理，利用提供的信息来完成客户端-服务器交互。接下来，我们将深入了解 ORM 基础知识，了解您如何使用 Python 类与关系型数据库管理系统（RDBMS）交互。

# 理解 SQLAlchemy ORM 基础知识

SQLAlchemy 为开发者提供了完全使用 Python 代码来创建、读取、更新和删除表的能力。SQLAlchemy 是 Python SQL 工具包和 ORM，允许应用程序开发者与数据库交互，而无需编写直接的 SQL 语句。

由于 SQLAlchemy 是一个 ORM 库，Python 类被映射到表上，而这些类的实例被映射到关系数据库中的表行。然后，您就可以使用 Python 面向对象编程代码在应用程序中执行数据库 SQL **创建**、**读取**、**更新**和**删除**（**CRUD**）操作以及其他必要的操作。

SQLAlchemy 的 ORM 功能为 Python 开发者提供了利用函数调用直接生成 SQL 语句的能力。以这种方式思考数据库，开发者能够解耦对象模型和数据库模式，从而实现更灵活的数据库结构、优雅的 SQL 语句以及可以与不同类型的数据库方言交互的 Python 代码。

SQLAlchemy 不介意你使用哪种数据库系统。SQLAlchemy 使你更容易在不更改代码的情况下从一个数据库切换到另一个数据库。SQLAlchemy 无疑是一个强大的工具，它简化了 Web 开发的过程，使其更快、更高效。

由于 SQLAlchemy 的用户友好界面，开发者可以轻松创建数据库模型，并从他们的 Flask 应用程序中与之交互。此外，SQLAlchemy 旨在减少将错误引入代码库的机会，这对于大规模应用程序尤为重要。

提供了一个强大的框架来处理数据库交互，SQLAlchemy 允许开发者专注于构建应用程序的逻辑，而无需担心底层的数据库操作。这导致了一个更流畅的开发过程和更好的整体代码质量。

最后，SQLAlchemy ORM 库自带自动缓存功能。SQLAlchemy 在对象首次加载后缓存集合和对象之间的引用。这无疑提高了性能，并防止你在每次调用时都向数据库发送 SQL 查询。

在 SQLAlchemy 中，有三个主要的抽象层。这些层是引擎、方言和连接池。这个三重奏描述了你是如何选择与数据库交互的。

![图 8.9 – SQLAlchemy 的抽象层](img/Figure_8.9_B18554.jpg)

图 8.9 – SQLAlchemy 的抽象层

让我们深入挖掘 SQLAlchemy 的抽象层，以更好地理解它们在 SQLAlchemy 中的使用。

## SQLAlchemy 引擎

一个 SQLAlchemy 应用程序从创建一个引擎开始。在 SQLAlchemy 能够连接和与数据库交互之前，你需要创建一个引擎。引擎是 SQLAlchemy 中抽象层的最低层，它的工作方式与我们在使用 Psycopg2 与数据库交互时几乎相同。

让我们创建一个引擎来启动数据库连接和交互。

以下代码片段展示了使用 SQLAlchemy 的`create_engine`进行 PostgreSQL 数据库连接。SQLAlchemy 引擎连接到 PostgreSQL 数据库，并在 Python 中执行 SQL 查询：

```py
from sqlalchemy import create_engine# create the engine with the database URL
engine = create_engine
    ("postgresql://admin:admin123@localhost:5432/bizza")
# create a connection to the database
conn = engine.connect()
# execute a SQL query
result = conn.execute('SELECT * FROM speakers')
# loop through the result set and print the values
for row in result:
    print(row)
# close the result set and connection
result.close()
conn.close()
```

这就是前面代码中发生的事情：

+   从`sqlalchemy`库中导入`create_engine`函数：

    ```py
    from sqlalchemy import create_engine
    ```

+   然后，在 SQLAlchemy 中调用`create_engine()`方法并将其分配给引擎实例。接着，你传入一个连接 URL 字符串，该字符串指定了后端服务器细节，例如用于连接的数据库的`name`、`username`、`password`、`host`和`port`：

    ```py
    engine = create_engine(    'postgresql://admin:admin123@localhost:5432/bizza'    )Database name-bizzaUsername-adminPasswordadmin123
    ```

+   为当前数据库创建一个连接池，并且这个连接池在整个应用程序生命周期中只创建一次：

    ```py
    connection = engine.connect()
    ```

+   我们使用建立的连接对象执行 SQL 语句：

    ```py
    result= connection.execute('SELECT * FROM speakers')
    ```

+   最后，我们遍历查询返回的结果集并打印值。

使用`result.close()`和`conn.close()`方法关闭结果集和连接，以释放资源并防止内存泄漏是很重要的。

## SQLAlchemy 连接池

连接池是软件工程中对象池设计范式的实现，其中连接在需要时被重用而不是每次都需要创建。在 SQLAlchemy 中，`create_engine()`方法的通常使用会生成一个连接池对象，该对象在创建后会被重用，在执行数据库事务时后续连接会使用它。

这种连接池设计模式有助于提高性能并更好地管理应用程序的并发连接。连接池的默认设置可以根据需要调整，以更有效地服务于商业 Web 应用程序的最终用户。

要更新连接池，将连接池参数添加到`create_engine()`中：

```py
engine = create_engine('postgresql://user:admin123@localhost/bizza', pool_size=20, max_overflow=0)
```

以下项目符号列表显示了如何调整连接池参数以优化与 SQLAlchemy 连接池一起工作的性能。

+   使用`pool_size`，你可以设置连接池可以处理的连接数。

+   使用`max_overflow`，你可以指定连接池支持的溢出连接数。

+   使用`pool_recycle`，你可以配置连接池中连接的最大年龄（以秒为单位）。

+   使用`pool_timeout`，你可以指定应用程序在放弃从连接池获取连接之前需要等待多少秒。

## SQLAlchemy 方言

使用 SQLAlchemy 的主要好处之一是，你可以使用不同类型的 DB-API 实现（如 Psycopg2）和关系数据库，而无需更改你的 Python 代码以适应它们的内部工作方式。

因此，方言是 SQLAlchemy 中的一种系统，它使得与不同数据库的交互成为可能。你可以在开发环境中使用 SQLite，并决定在生产环境中使用 MySQL 或 PostgreSQL，而无需更改现有的代码库。

以下是一些关系数据库方言的示例：

+   PostgreSQL

+   MySQL 和 MariaDB

+   SQLite

+   Oracle

+   Microsoft SQL Server

为了使用这些数据库方言，你必须在你的应用程序中安装适当的 DB-API 驱动程序。当与 PostgreSQL 连接时，SQLAlchemy 使用 Psycopg2 作为 DB-API 规范的实现。Psycopg2 是 Python 的 PostgreSQL 适配器。Psycopg2 提供了一个简单高效的方法，使用 Python 代码与 PostgreSQL 数据库进行通信。

我们已经讨论了 SQLAlchemy 方言的工作原理，但那些方言中使用的数据类型是什么呢？SQLAlchemy 提供了多种数据类型，现在我们将探讨其中的一些。之后，我们将讨论如何将它们映射到 Python 类中。

## SQLAlchemy 数据类型 – 表与类之间的映射

SQLAlchemy 为我们可能选择在应用程序中使用的任何底层关系数据库提供了高级抽象。SQLAlchemy 对我们熟悉的常见传统数据库提供了数据类型支持。例如，日期、时间、字符串、整数和布尔值都得到了 SQLAlchemy 的良好支持。

SQLAlchemy 数据类型允许精确的数据存储和检索。数据库表中的每一列都有一个特定的数据类型，它定义了可以存储在该列中的数据类型，例如整数、字符串或日期。SQLAlchemy 提供了一系列数据类型，可用于定义数据库表中的列，包括 `Numeric`、`String`、`DateTime` 和 `Boolean`。

通过使用这些数据类型，我们可以确保我们的数据被准确且高效地存储，从而实现更快的检索和更可靠的性能。此外，表和类之间的映射很重要，因为它允许我们以面向对象的方式处理数据库表。

SQLAlchemy 提供了一个 ORM 系统，允许我们将数据库表映射到 Python 类。这意味着我们可以使用 Python 对象和方法来处理数据库数据，这使得构建和维护我们的 Web 应用程序更加容易。让我们看看 SQLAlchemy 数据类型是如何与 Python 类和实例属性映射的，分别用于表和列的创建。

本书中的 Flask 应用程序将使用这种方法从 Python 类定义表：

```py
class Speaker(Base):__tablename__ = 'speakers'
speaker_ id=Column(Integer, primary_key=True)
first_name=Column(String)
last_name=Column(String)
email=Column(String)
created_at = Column(DateTime(timezone=True),
    server_default=datetime.now)
updated_at = ColumnDateTime(timezone=True),
    default=datetime.now, onupdate=datetime.now)
```

在前面的代码片段中，我们定义了一个名为 `Speaker` 的类，它具有一些属性。让我们更深入地了解这段代码：

+   `__tablename__` 属性被设置为 `speakers`，表示 `Speaker` 类的实例应该存储在数据库中的 `speakers` 表中。

+   `speaker_id` 属性指定这是表中的 `primary_key`，并且具有 `Integer` 类型。

+   `first_name` 属性指定表中的一个列具有名称 `first_name`，类型为 `String`。

+   `last_name` 属性指定表中的一个列具有名称 `last_name`，类型为 `String`。

+   `email` 属性指定表中的一个列具有名称 `email`，类型为 `String`。

+   `created_at` 和 `updated_at` 指定了表中具有所述名称的列，其类型为 `date`。然而，方法被传递以获取当前的 `timezone`。

定义了类和实例后，我们可以利用 SQLAlchemy 的内部 API 和配置的方言，将类属性映射到数据库的相应原生结构。例如，如果我们类中有字符串数据类型，SQLAlchemy 将将其映射到 PostgreSQL 中的 `varchar` 列。

这确保了在类中使用的数据类型被正确地转换为适当的数据库列类型，从而使得 Python 代码和数据库之间的通信无缝。

为了构建一个健壮的演讲者会议 Web 应用程序，我们需要正确地建模数据。SQLAlchemy 为我们提供了一个强大的 ORM，这使得将我们的数据库表映射到类变得容易。

然后，我们可以定义这些类之间的属性和关系，这使得处理复杂的数据结构变得容易。在下一节中，我们将探讨如何使用 SQLAlchemy 为我们的演讲者会议 Web 应用程序建模数据。

# 为演讲者会议 Web 应用程序建模数据

**数据建模**是创建数据结构的概念性、逻辑性和视觉表示的过程，以便理解、分析和管理复杂的信息系统。数据建模涉及识别将在系统中表示的实体（对象、概念或事物），并定义它们的属性以及与其他实体之间的关系。

数据建模的主要目的是创建一个清晰且精确的数据表示，这些数据将由信息系统存储、处理和管理。在 Web 应用程序中，一个设计良好的数据模型可以确保 Web 应用程序具有可扩展性、效率，并满足其用户的需求。

让我们快速检查在设计会议演讲者 Web 应用程序的数据模型时需要考虑的一些最佳实践：

+   **识别实体**：首先根据系统需求识别将在 Web 应用程序中表示的实体或对象，例如用户、演讲者、演讲、日程、与会者、会议、场地和赞助商。

+   **定义关系**：确定这些实体之间的关系，例如一对一、一对多或多对多关系。例如，一个演讲者可以在会议上进行多次演讲，但一次演讲只能由一个演讲者进行。

+   **确定属性**：定义每个实体的属性或特性——例如，一个演讲者可能具有姓名、联系地址、传记或演讲主题等属性。

+   **考虑性能和可扩展性**：设计数据模型以优化性能和可扩展性，例如使用索引、缓存或反规范化以减少查询时间和提高系统响应。

+   **规划数据存储**：考虑可用的数据存储选项，例如关系数据库、NoSQL 数据库或平面文件，并根据 Web 应用程序的需求选择合适的选项。在这种情况下，选择了 PostgreSQL。

+   **数据安全**：设计数据模型以确保数据安全，例如使用加密或访问控制来保护敏感数据。例如，当用户登录时，他们输入的密码可以被散列并与存储的散列进行比较以验证其身份。

+   **考虑未来变化**：为 Web 应用程序的未来变化和更新做出规划，例如添加新的实体或属性，或修改现有关系。

你是否准备好通过建立和组织数据模型来开始开发 *Bizza* 项目的后端？*Bizza* 项目旨在创建一个以数据库驱动为主的完整栈网络应用，专注于会议演讲者。

在这个数据模型中，我们将更详细地查看数据库、表以及表之间的关系。

`bizza` 数据库有以下表和关系：

+   `users` 和 `userExtras`，用户和角色之间的多对多关系，`users` 和 `speakers` 之间的一对一关系，以及 `users` 和 `attendees` 之间的一对多关系。

+   `users` 表 —— `users` 和 `userExtras` 之间的一对一关系。

+   `users` 表 —— `users` 和 `roles` 之间的多对多关系。

+   `Speaker` 模型将与 `users` 模型相关联，以便将每个演讲者与他们的用户账户关联起来。与其他表存在以下关系——`speakers` 和 `sessions` 之间的一对多关系，`users` 和 `speakers` 之间的一对一关系，`speakers` 和 `schedules` 之间的一对多关系，以及 `speakers` 和 `presentations` 之间的一对多关系。

+   `Presentation` 模型将与 `Speaker` 模型相关联，以便将每个演示与其演讲者关联起来。与其他表存在以下关系——`presentations` 和 `speakers` 之间的多对一关系，以及 `presentations` 和 `schedules` 之间的一对多关系。

+   `Schedule` 模型将与 `Presentation` 模型相关联，以便将每个演示与其日期和时间关联起来。与其他表存在以下关系——`schedules` 和 `presentations` 之间的多对一关系，`schedules` 和 `sessions` 之间的一对多关系，`schedules` 和 `venues` 之间的多对一关系，以及 `schedules` 和 `attendees` 之间的多对多关系。

+   `attendees` 和 `schedules`。

+   `Session` 模型将与 `Presentation` 模型相关联，以便将每个会议与它的演示关联起来。与其他表存在以下关系——`sessions` 和 `schedules` 之间的多对一关系，以及 `sessions` 和 `presentations` 之间的一对一关系。

+   `venues` 和 `schedules`。

+   **赞助商模型**：此模型将包括有关每个会议赞助商的数据，例如他们的名称、标志、网站以及任何其他相关细节。

现在我们已经定义了我们的数据模型，在下一节中，我们将探讨如何通过几个简单的步骤无缝地将数据发送到 PostgreSQL 数据库。

# 从 Flask 应用程序向 PostgreSQL 数据库发送数据

与数据库交互是大多数网络应用的一个常见方面。从 Flask 应用程序向 PostgreSQL 数据库发送数据简单直接。一个网络应用如果没有存储和从数据库检索数据的能力，将是不完整的。

将数据发送到数据库的第一步是确保 Flask 应用和数据库之间存在连接。这涉及到安装所需的库，并确保您在一个虚拟环境中工作，以包含您的安装并防止其他库的干扰导致意外发生。

让我们利用在 *第一章* 的 *使用 Flask 设置开发环境* 部分中创建的 `bizza/backend` 目录。如果您还没有创建，可以创建一个。安装并激活虚拟环境。

要创建一个虚拟环境，请在终端中打开您的项目根目录，并添加以下命令：

```py
python –m venv venv
```

现在，让我们激活虚拟环境：

+   **对于 Windows 用户**：

    ```py
    Venv\Scripts\activate
    ```

+   **对于 Linux/Mac 用户**：

    ```py
    source venv/bin/activate
    ```

要从 Python 类创建数据库表，您需要安装 `flask-sqlalchemy`。我们已经安装了它。如果没有安装，请输入以下命令：

```py
pip install flask-sqlalchemy
```

在 Flask 应用模块中——即 `app.py` ——更新 `app.py` 的内容，使用以下代码在 *bizza* 数据库中创建 `users` 表。在本节中，我们只创建 `users` 表以演示如何从 Flask 应用向 PostgreSQL 发送数据。

其他模型将在 GitHub 上展示，并在书中稍后介绍，直到我们完成演讲者会议网络应用 *Bizza* 的完整实现：

```py
# Import flaskfrom flask import Flask
# Import datetime
from datetime import datetime
# Import SQLAlchemy
from flask_sqlalchemy import SQLAlchemy
# Create a Flask instance
app = Flask(__name__)
# Add the PostgreSQL database
app.config['SQLALCHEMY_DATABASE_URI'] =
    'postgresql://<db_username>:<db_password>@localhost:
    5432/<database_name>'
# Initialize the database
db = SQLAlchemy(app)
# User model definition
class User(db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True,
        nullable=False)
    email = db.Column(db.String(120), unique=True,
        nullable=False)
    password = db.Column(db.String(128), nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    roles = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True)
    is_superuser = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime,
        default=datetime.utcnow())
    updated_at = db.Column(db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow())
    def __repr__(self):
        return '<User %r>' % self.username
```

让我们深入探讨前面用于在数据库中创建 `users` 表的 Flask 应用代码片段：

+   `from flask import Flask` 从 Flask 模块导入 `Flask` 类。

+   `from flask_sqlalchemy import SQLAlchemy` 从 `flask_sqlalchemy` 模块导入 `SQLAlchemy` 类。

+   `app = Flask(__name__)` 创建了一个名为 `app` 的 `Flask` 实例。

+   `app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://<db_username>:<db_password>@localhost:5432/<database_name>'` 是 `app.config[]` 字典定义了数据库的路径，其中设置了 `db username` 和 `password`。

+   `db = SQLAlchemy(app)` 使用 Flask 应用作为参数初始化 `SQLAlchemy` 类的一个实例。

+   然后，我们开始定义 `User` 的 `model` 类，使用 `class User(db.Model):`。`User(db.Model)` 类定义了一个继承自 `db.Model` 类的 `User` 模型。

+   `__tablename__ = 'users'` 允许我们为表指定一个自定义名称——一个与 `Class User` 模型对应的表名 `users`。如果没有指定，将使用小写类名（`user`）。

+   然后，我们为表添加列。每个列都是 SQLAlchemy 的 `Column` 子类的对象：

    +   `User_id = db.Column(db.Integer, primary_key=True, nullable=False)` 定义了一个名为 `user_id` 的主键列，其数据类型为 `Integer`。

    +   `username = db.Column(db.String(50), unique=True, nullable=False)` 定义了一个名为 `username` 的列，其数据类型为 `String`，并强制要求它必须是唯一的且不可为空。

    +   `email = db.Column(db.String(120), unique=True, nullable=False)` 定义了一个名为 `email` 的列，其数据类型为 `String`，并强制要求它必须是唯一的且不可为空。

    +   `password = db.Column(db.String(256), nullable=False)` 定义了一个名为 `password` 的列，其数据类型为 `String`，并强制要求它不可为空。

    +   `first_name = db.Column(db.String(50), nullable=False)` 定义了一个名为 `first_name` 的列，其数据类型为 `String`，并强制要求它不可为空。

    +   `last_name = db.Column(db.String(50), nullable=False)` 定义了一个名为 `last_name` 的列，其数据类型为 `String`，并强制要求它不可为空。

    +   `is_active = db.Column(db.Boolean, default=True)` 定义了一个名为 `is_active` 的列，其数据类型为 `Boolean`，并设置默认值为 `True`。

    +   `is_superuser = db.Column(db.Boolean, default=False)` 定义了一个名为 `is_superuser` 的列，其数据类型为 `Boolean`，并设置默认值为 `False`。

+   `def __repr__(self)` 定义了一个返回 `User` 对象字符串表示的方法。

+   `return '<User %r>' % self.username` 返回一个包含 `User` 对象用户名的字符串。

既然我们已经定义了 `User` 类模型，现在是时候在数据库中创建 `users` 表并传入一些数据了：

1.  在项目根目录中打开一个终端，并输入 `flask` `shell` 命令。

![图 8.10 – 显示 Flask shell 的屏幕截图](img/Figure_8.10_B18554.jpg)

图 8.10 – 显示 Flask shell 的屏幕截图

1.  输入 `from app import db` 以连接到数据库对象。

1.  输入 `db.create_all()` 以创建 `users` 表。

![图 8.11 – 显示 db.create_all() 的屏幕截图](img/Figure_8.11_B18554.jpg)

图 8.11 – 显示 db.create_all() 的屏幕截图

1.  输入 `from app import User` 以获取对 `User` 模型的访问权限。

1.  使用 `flask` `shell` 终端向表中添加用户数据：

    ```py
    user1=User(username='john',first_name='John',last_name='Stewart',email='admin@admin.com',password='password')
    ```

1.  添加 `db.session.add(user1)` 和 `db.session.commit()` 以将用户添加到数据库会话中，并在数据库中提交以持久化数据。

1.  输入 `User.query.all()` 以查看带有添加信息的 `users` 表。

![图 8.12 – 显示插入数据的 Flask shell 的屏幕截图](img/Figure_8.12_B18554.jpg)

图 8.12 – 显示插入数据的 Flask shell 的屏幕截图

接下来，我们将讨论如何使用迁移添加和跟踪数据库结构的更改。

# 使用 Alembic 进行迁移

如前所述，Alembic 是一个迁移工具，它使 Flask 开发者跟踪数据库更改变得不那么复杂。由于我们预计数据模型会发生变化，我们需要一个工具来跟踪这些更改并确保它们在数据库中得到更新。

这与我们在使用 `Git` 进行源代码版本控制的方式类似。同样的方法也适用于数据库模式变更管理，我们在此保持数据库结构的增量可逆更改。在与数据库表一起工作时，您可能想要添加或删除列，从而在 Python 模型类中更改模式。

完成此操作后，您需要一个自动过程来确保您的数据库表和数据模型状态保持同步。Alembic 优雅地处理模式迁移，并确保 Python 文件中的数据模型与数据库结构相同。

让我们看看如何在 Flask 应用程序中实现迁移。我们将添加另一个模型，并使用迁移来跟踪对数据库所做的更改。在这里，我们添加了`Speaker`模型类。

如在*设置* *Alembic*部分所述，使用`pip`安装`Flask-Migrate`：

```py
pip install Flask-Migrate
```

在`app.py`内部，添加以下片段：

```py
from flask import Flaskfrom flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] =
    'sqlite:///bizza.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)
class Speaker(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    bio = db.Column(db.Text, nullable=False)
    photo = db.Column(db.String(100))
    contact_info = db.Column(db.String(100))
    user_id = db.Column(db.Integer,
        db.ForeignKey('users.user_id'), nullable=False)
    user = db.relationship('User',
        backref=db.backref('speaker', uselist=False))
    def __repr__(self):
        return f"Speaker('{self.name}', '{self.bio}',
            '{self.photo}', '{self.contact_info}')"
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
```

对`app.py`文件进行了一些更改，包括安装`flask-migrate`包，使用`app`和`db`实例作为参数创建`Migrate`类的一个实例，以及添加一个将在其包含到数据库中时被跟踪的`speaker`模型类。

## 运行迁移

根据对`app.py`中添加的`speaker`模型类的更改，让我们使用 Alembic 实现迁移：

1.  首先，在项目目录中，输入`pip install flask-migrate`命令。`flask-migrate`扩展为 Flask 应用程序提供数据库迁移支持。

1.  在 Flask 中生成初始迁移，请在项目根目录的命令终端中输入以下内容：

    ```py
    migrations in your Flask application directory.
    ```

1.  然后，一旦您初始化了迁移存储库，您可以使用`flask db migrate`命令根据对模型所做的更改生成第一个迁移。我们已将新的演讲者模型添加到`app.py`文件中。

1.  现在，使用`flask db migrate -m 'first migration message, speaker model added'`命令根据对数据库模型所做的更改生成一个新的迁移脚本。`-m`标志用于指定描述迁移中更改的消息。以下命令将创建一个包含模型中指定更改的新迁移文件：

    ```py
    flask db initflask db migrate -m "first migration speaker model"
    ```

    以下屏幕截图显示了命令的效果：

![图 8.13 – 显示迁移命令的屏幕截图](img/Figure_8.13_B18554.jpg)

图 8.13 – 显示迁移命令的屏幕截图

1.  将模式数据模型更改提交到基于迁移脚本状态的数据库中，请运行以下命令：

    ```py
    flask db upgrade
    ```

    您将得到以下输出：

![图 8.14 – 显示 flask db upgrade 命令的屏幕截图](img/Figure_8.14_B18554.jpg)

图 8.14 – 显示 flask db upgrade 命令的屏幕截图

这将应用所有挂起的迁移到您的数据库模式中。从这一点开始，您可以使用`flask db migrate`命令根据需要继续生成新的迁移。

注意

总结 Alembic 迁移所需的命令，请按照以下步骤操作：

**pip install flask-migrate**

**flask db init**

**flask db migrate -m “first migration message speakers model”**

**flask db upgrade**

Alembic 是一个用于 SQLAlchemy 的数据库迁移工具，它帮助保持数据库模式与应用程序的数据模型同步。当使用 Alembic 时，你通过一系列迁移来定义数据模型的变化，这些迁移是修改数据库模式的 Python 脚本。

这些迁移存储在迁移目录中，每个迁移都与数据库模式的一个特定版本相关联。当你运行一个迁移时，Alembic 会将数据库模式的当前状态与迁移中定义的目标状态进行比较。然后，它生成一系列 SQL 语句来修改数据库模式以匹配目标状态。

这些 SQL 语句在事务中执行，这确保了数据库模式以一致的方式被修改。使用 Alembic 管理数据库迁移，你可以确保数据库模式随着时间的推移与应用程序的数据模型保持同步。这有助于防止数据不一致和其他问题，这些问题可能在数据库模式和数据模型不同步时出现。

# 摘要

在本章中，我们广泛讨论了用于网络的 SQL 和关系数据建模。关系数据库帮助我们以关系组的形式设计数据库。我们还讨论了数据库中可能存在的关系，例如一对一、一对多和多对多关系，这使得我们能够在数据库中逻辑地分组关系并强制执行数据引用完整性。

此外，我们还探讨了如何设置 PostgreSQL。我们介绍了 SQLAlchemy 的基础知识及其相关的数据库适配器，以及它们在 Flask 应用程序开发中的使用。我们讨论了数据模型设计，以 *Bizza* 项目为例。最后，我们讨论了 Flask 应用如何与数据库和迁移进行通信，以及在 Flask 中如何跟踪数据库的变化。

在下一章中，我们将广泛讨论后端开发中的 API 以及如何使用 Flask 框架来实现 API 设计。
