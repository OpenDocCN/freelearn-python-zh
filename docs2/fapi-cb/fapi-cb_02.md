# 2

# 与数据一起工作

**数据处理**是任何 Web 应用的骨架，本章致力于掌握这一关键方面。您将开始一段在 FastAPI 中处理数据的旅程，您将学习如何使用**结构化查询语言**（**SQL**）和**NoSQL**数据库来集成、管理和优化数据存储的复杂性。我们将介绍 FastAPI 如何与强大的数据库工具结合，以创建高效和可扩展的数据管理解决方案。

从 SQL 数据库开始，您将获得实际操作经验，包括设置数据库、实现**创建、读取、更新和删除**（**CRUD**）操作，以及理解与 SQLAlchemy（Python 中流行的**对象关系映射**（**ORM**）选项）一起工作的细微差别。然后我们将转向 NoSQL 数据库，深入**MongoDB**的世界。您将学习如何将其与 FastAPI 集成，处理动态数据结构，并利用 NoSQL 解决方案的灵活性和可扩展性。

但这不仅仅是存储和检索数据。本章还关注保护敏感数据和管理数据库中的事务和并发性的最佳实践。您将探索如何保护您的数据免受漏洞的侵害，并确保应用程序数据操作的完整性和一致性。

到本章结束时，您不仅将深入了解如何在 FastAPI 中处理各种数据库系统，还将具备构建健壮和安全的 Web 应用数据模型所需的技能。无论是实现复杂查询、优化数据库性能还是确保数据安全，本章都提供了您管理应用程序数据所需的技术和知识。

在本章中，我们将介绍以下食谱：

+   设置 SQL 数据库

+   使用 SQLAlchemy 理解 CRUD 操作

+   集成 MongoDB 用于 NoSQL 数据存储

+   与数据验证和序列化一起工作

+   与文件上传和下载一起工作

+   处理异步数据操作

+   保护敏感数据及最佳实践

每个主题都旨在为您提供处理 FastAPI 中数据的必要技能和知识，确保您的应用程序不仅功能齐全，而且安全且可扩展。

# 技术要求

为了有效地运行和理解本章中的代码，请确保您已设置以下内容。如果您已经跟随了*第一章*，*FastAPI 的第一步*，您应该已经安装了一些这些内容：

+   **Python**：请确保您已在计算机上安装了 3.9 或更高版本的 Python。

+   `pip install fastapi[all]` 命令。正如我们在*第一章*，*FastAPI 的第一步*中看到的，此命令还安装了**Uvicorn**，这是一个必要的 ASGI 服务器，用于运行您的 FastAPI 应用程序。

+   **集成开发环境**（**IDE**）：应安装合适的 IDE，例如 **VS Code** 或 **PyCharm**。这些 IDE 为 Python 和 FastAPI 开发提供了出色的支持，包括语法高亮、代码补全和易于调试等功能。

+   **MongoDB**：对于本章的 NoSQL 数据库部分，您需要在本地机器上安装 MongoDB。从 [`www.mongodb.com/try/download/community`](https://www.mongodb.com/try/download/community) 下载并安装适合您操作系统的免费社区版服务器。

    通过命令行运行 Mongo Deamon 来确保 MongoDB 已正确安装：

    ```py
    C:\Program>Files\MongoDB\Server\7.0\bin. You need to open the terminal in this location to run the daemon or run:

    ```

    $ C:\Program>Files\MongoDB\Server\7.0\bin\mongod -- version

    ```py

    ```

+   **MongoDB 工具**：虽然不是必需的，但像 **MongoDB Shell** ([`www.mongodb.com/try/download/shell`](https://www.mongodb.com/try/download/shell)) 和 **MongoDB Compass GUI** ([`www.mongodb.com/try/download/compass`](https://www.mongodb.com/try/download/compass)) 这样的工具可以极大地增强您与 MongoDB 服务器的交互。它们提供了一个更用户友好的界面来管理数据库、运行查询和可视化数据结构。

本章中使用的所有代码和示例均可在 GitHub 上供参考和下载。请访问 [`github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter02`](https://github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter02) 以访问存储库。

# 设置 SQL 数据库

在数据处理的世界里，Python 的力量与 SQL 数据库的效率相结合。本食谱旨在向您介绍如何在您的应用程序中集成 SQL 数据库，这对于任何希望构建健壮和可扩展的 Web 应用程序的开发者来说是一项关键技能。

SQL 是管理和管理关系型数据库的标准语言。当与 FastAPI 结合使用时，它解锁了数据存储和检索的无限可能。

FastAPI 与 SQL 数据库的兼容性是通过 ORM 实现的。其中最受欢迎的是 **SQLAlchemy**。在本食谱中，我们将重点关注它。

## 准备工作

首先，您需要确保 FastAPI 和 SQLAlchemy 已安装到您的虚拟环境中。如果您遵循了 *第一章*，*FastAPI 的第一步* 中的步骤，那么您应该已经设置了 FastAPI。对于 SQLAlchemy，只需一个简单的 `pip` 命令即可：

```py
$ pip install sqlalchemy
```

安装完成后，下一步是配置 SQLAlchemy，使其能够与 FastAPI 一起工作。这涉及到设置数据库连接——我们将一步步进行。

## 如何操作…

现在，让我们更深入地探讨如何为您的 FastAPI 应用程序配置 SQLAlchemy。SQLAlchemy 作为您 Python 代码和数据库之间的桥梁，允许您使用 Python 类和对象而不是编写原始 SQL 查询来与数据库交互。

在安装 SQLAlchemy 后，下一步是在你的 FastAPI 应用程序中配置它。这涉及到定义你的数据库模型——在 Python 代码中表示数据库表。在 SQLAlchemy 中，模型通常使用类来定义，每个类对应于数据库中的一个表，每个类的属性对应于表中的一个列。

按照以下步骤进行操作。

1.  在当前目录下创建一个名为 `sql_example` 的新文件夹，进入该文件夹后，再创建一个名为 `database.py` 的文件。编写一个用作参考的 `base` 类：

    ```py
    from sqlalchemy.orm import DeclarativeBase
    class Base(DeclarativeBase):
        pass
    ```

    要在 SQLAlchemy 中定义一个模型，你需要创建一个从 `DeclarativeBase` 类派生的基类。这个 `Base` 类维护了你定义的类和表的目录，并且是 SQLAlchemy ORM 功能的核心。

    你可以通过阅读官方文档来了解更多信息：[`docs.sqlalchemy.org/en/13/orm/extensions/declarative/index.xhtml`](https://docs.sqlalchemy.org/en/13/orm/extensions/declarative/index.xhtml)。

1.  一旦你有了你的 `Base` 类，你就可以开始定义你的模型了。例如，如果你有一个用户表，你的模型可能看起来像这样：

    ```py
    from sqlalchemy.orm import (
        Mapped,
        mapped_column
    )
    class User(Base):
        __tablename__ = "user"
        id: Mapped[int] = mapped_column(
            primary_key=True,
        )
        name: Mapped[str]
        email: Mapped[str]
    ```

    在这个模型中，`User` 类对应于数据库中名为 `user` 的表，包含 `id`、`name` 和 `email` 列。每个 `class attribute` 指定了列的数据类型。

1.  一旦你的模型被定义，下一步就是连接到数据库并创建这些表。SQLAlchemy 使用连接字符串来定义它需要连接到的数据库的详细信息。这个连接字符串的格式取决于你使用的数据库系统。

    例如，SQLite 数据库的连接字符串可能看起来像这样：

    ```py
    DATABASE_URL = "sqlite:///./test.db"
    ```

    第一次连接到 `test.db` 数据库文件时。

    你将使用 `DATABASE_URL` 连接字符串在 SQLAlchemy 中创建一个 `Engine` 对象，该对象代表与数据库的核心接口：

    ```py
    from sqlalchemy import create_engine
    engine = create_engine(DATABASE_URL)
    ```

1.  创建好引擎后，你可以继续在数据库中创建你的表。你可以通过传递你的 `base` 类和引擎到 SQLAlchemy 的 `create_all` 方法来完成此操作：

    ```py
    Base.metadata.create_all(bind=engine)
    ```

现在你已经定义了代码中数据库的所有抽象，你可以继续设置数据库连接。

## 建立数据库连接

设置 SQL 数据库的最后一部分是建立数据库连接。这个连接允许你的应用程序与数据库通信，执行查询并检索数据。

数据库连接由会话管理。在 SQLAlchemy 中，会话代表了一个用于你的对象的 *工作区*，一个你可以添加新记录或检索现有记录的地方。每个会话都绑定到一个单独的数据库连接。

为了管理会话，我们需要创建一个 `SessionLocal` 类。这个类将被用来创建和管理与数据库交互的会话对象。以下是创建它的方法：

```py
from sqlalchemy.orm import sessionmaker
SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine
)
```

`sessionmaker` 函数创建会话的工厂。`autocommit` 和 `autoflush` 参数设置为 `False`，这意味着你必须手动提交事务并在更改刷新到数据库时管理它们。

在 `SessionLocal` 类就绪后，你可以创建一个函数，该函数将在你的 FastAPI 路由函数中使用，以获取一个新的数据库会话。我们可以在 `main.py` 模块中这样创建它：

```py
from database import SessionLocal
def get_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

在你的路由函数中，你可以使用此函数作为依赖项与数据库通信。

在 FastAPI 中，这可以通过 `Depends` 类来完成。在 `main.py` 文件中，你可以添加一个端点：

```py
from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session
from database import SessionLocal
app = FastAPI()
@app.get("/users/")
def read_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users
```

这种方法确保为每个请求创建一个新的会话，并在请求完成后关闭，这对于维护数据库事务的完整性至关重要。

你可以使用以下命令运行服务器：

```py
$ uvicorn main:app –-reload
```

如果你尝试在 `localhost:8000/users` 上调用 `GET` 端点，你会得到一个空列表，因为没有添加任何用户。

## 参见

你可以在文档页面上了解更多关于如何在 **SQLAlchemy** 中设置会话的信息：

+   *SQLAlchemy* *会话*：[`docs.sqlalchemy.org/en/20/orm/session_basics.xhtml`](https://docs.sqlalchemy.org/en/20/orm/session_basics.xhtml)

# 使用 SQLAlchemy 理解 CRUD 操作

在使用 FastAPI 设置好 SQL 数据库后，下一个关键步骤是创建数据库模型。这个过程对于你的应用程序如何与数据库交互至关重要。在 SQLAlchemy 中的 **数据库模型** 实质上是代表 SQL 数据库中表的 Python 类。它们提供了一个高级的面向对象接口，可以像处理常规 Python 对象一样操作数据库记录。

在这个菜谱中，我们将设置 **创建、读取、更新和删除** （**CRUD**） 端点以与数据库交互。

## 准备工作

在设置好模型后，你现在可以实施 CRUD 操作。这些操作构成了大多数网络应用程序的骨架，允许你与数据库交互。

## 如何操作…

对于每个操作，我们将创建一个专门的端点，以实现与数据库的交互操作。

### 创建新用户

要添加新用户，我们将使用 `POST` 请求。在 `main.py` 文件中，我们必须定义一个端点，该端点接收用户数据，在请求体中创建一个新的 `User` 实例，并将其添加到数据库中：

```py
class UserBody(BaseModel):
    name: str
    email: str
@app.post("/user")
def add_new_user(
    user: UserBody,
    db: Session = Depends(get_db)
):
    new_user = User(
        name=user.name,
        email=user.email
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user
```

在几行代码中，你已经创建了添加新用户到数据库的端点。

### 读取特定用户

要获取单个用户，我们将使用 `GET` 端点：

```py
from fastapi import HTTPException
@app.get("/user")
def get_user(
    user_id: int,
    db: Session = Depends(get_db)
    ):
    user = (
        db.query(User).filter(
            User.id == user_id
        ).first()
    )
    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    return user
```

如果用户不存在，端点将返回 `404` 响应状态。

### 更新用户

通过 API 更新记录提供了各种方法，包括`PUT`、`PATCH`或`POST`方法。尽管在理论上存在细微差别，但方法的选择通常取决于个人偏好。我倾向于使用`POST`请求，并通过添加`user_id`参数来增强`/user`端点。这简化了过程，最大限度地减少了需要大量记忆的需求。您可以在`main.py`模块中这样集成此端点：

```py
@app.post("/user/{user_id}")
def update_user(
    user_id: int,
    user: UserBody,
    db: Session = Depends(get_db),
):
    db_user = (
        db.query(User).filter(
            User.id == user_id
        ).first()
    )
    if db_user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    db_user.name = user.name
    db_user.email = user.email
    db.commit()
    db.refresh(db_user)
    return db_user
```

这就是创建数据库中更新用户记录端点所需做的所有事情。

### 删除用户

最后，要在同一`main.py`模块中删除用户，需要使用`DELETE`请求，如下所示：

```py
@app.delete("/user")
def delete_user(
    user_id: int, db: Session = Depends(get_db)
):
    db_user = (
        db.query(User).filter(
            User.id == user_id
        ).first()
    )
    if db_user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    db.delete(db_user)
    db.commit()
    return {"detail": "User deleted"}
```

这些端点涵盖了基本的 CRUD 操作，并展示了如何将 FastAPI 与 SQLAlchemy 集成以进行数据库操作。通过定义这些端点，您的应用程序可以创建、检索、更新和删除用户数据，为客户端交互提供一个完全功能的 API。

现在您已经实现了所有操作，您可以通过运行以下命令来启动服务器：

```py
$ uvicorn main:app
```

然后，打开 inreactive 文档在`http://localhost:8000/docs`，并开始通过创建、读取、更新和删除用户来尝试这些端点。

在 FastAPI 中掌握这些 CRUD 操作是构建动态和以数据驱动的 Web 应用程序的重要一步。通过了解如何将 FastAPI 端点与 SQLAlchemy 模型集成，您已经具备了开发复杂和高效应用程序的能力。

## 参见

您可以在官方文档页面上找到如何使用 SQLAlchemy 设置 ORM 以进行 CRUD 操作的清晰快速入门指南：

+   *SQLAlchemy ORM 快速入门*: [`docs.sqlalchemy.org/en/20/orm/quickstart.xhtml`](https://docs.sqlalchemy.org/en/20/orm/quickstart.xhtml)

# 集成 MongoDB 进行 NoSQL 数据存储

从 SQL 迁移到 NoSQL 数据库在数据存储和管理方面开辟了不同的范式。**NoSQL 数据库**，如 MongoDB，以其灵活性、可扩展性和处理大量非结构化数据的能力而闻名。在本食谱中，我们将探讨如何将流行的 NoSQL 数据库 MongoDB 与 FastAPI 集成。

NoSQL 数据库与传统 SQL 数据库的不同之处在于，它们通常允许更动态和灵活的数据模型。例如，MongoDB 以**二进制 JSON**（**BSON**）格式存储数据，可以轻松适应数据结构的变化。这对于需要快速开发和频繁更新数据库模式的应用程序特别有用。

## 准备工作

确保您已在您的机器上安装了 MongoDB。如果您还没有安装，您可以从[`www.mongodb.com/try/download/community`](https://www.mongodb.com/try/download/community)下载安装程序。

FastAPI 不提供用于 NoSQL 数据库的内置 ORM。然而，由于 Python 强大的库，将 MongoDB 集成到 FastAPI 中非常简单。

我们将使用`pymongo`，一个 Python 包驱动程序来与 MongoDB 交互。

首先，确保您已经在您的机器上安装并运行了 MongoDB。

然后，您可以使用 `pip` 安装 `pymongo`：

```py
$ pip install pymongo
```

在安装了 `pymongo` 之后，我们现在可以建立与 MongoDB 实例的连接并开始执行数据库操作。

## 如何做到这一点...

我们可以通过以下步骤快速将我们的应用程序连接到本地机器上运行的 Mongo DB 实例。

1.  创建一个名为 `nosql_example` 的新项目文件夹。首先，在一个 `database.py` 文件中定义连接配置：

    ```py
    From pymongo import MongoClient
    client = MongoClient()
    database = client.mydatabase
    ```

    在这个例子中，`mydatabase` 是您数据库的名称。您可以用您喜欢的名称替换它。在这里，`MongoClient` 通过连接到本地运行的 MongoDB 实例的 *默认端口* 27017 来建立连接。

1.  一旦连接建立，您就可以定义您的集合（在 SQL 数据库中相当于表）并开始与之交互。MongoDB 将数据存储在文档集合中，其中每个文档都是一个类似 JSON 的结构：

    ```py
    user_collection = database["users"]
    ```

    在这里，`user_collection` 是您 MongoDB 数据库中 `users` 集合的引用。

1.  要测试连接，您可以在 `main.py` 文件中创建一个端点，该端点将检索所有用户，应该返回一个空列表：

    ```py
    from database import user_collection
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    app = FastAPI()
    class User(BaseModel):
        name: str
        email: str
    @app.get("/users")
    def read_users() -> list[User]:
        return [user for user in user_collection.find()]
    ```

1.  现在，运行您的 `mongod` 实例。您可以从命令行执行此操作：

    ```py
    $ mongod
    ```

    如果您在 Windows 上运行，命令将是：

    ```py
    $ C:\Program>Files\MongoDB\Server\7.0\bin\mongod
    ```

就这样。为了测试它，在另一个终端窗口中，通过运行以下命令启动 FastAPI 服务器：

```py
$ uvicorn main:app
```

然后，只需打开您的浏览器到 http://localhost:8000/users；您将看到一个空列表。这意味着您的数据库连接正在正确工作。

现在连接已经建立，我们将创建一个用于添加用户和用于通过 ID 获取特定用户的端点。我们将在 `main.py` 模块中创建这两个端点。

### 创建新用户

要向集合中添加新文档，请使用 `insert_one` 方法：

```py
class UserResponse(User):
    id: str
@app.post("/user")
def create_user(user: User):
    result = user_collection.insert_one(
        user.model_dump(exclude_none=True)
    )
    user_response = UserResponse(
        id=str(result.inserted_id),
         *user.model_dump()
    )
    return user_response
```

我们刚刚创建的端点在响应中返回受影响的 `id` 号，用作其他端点的输入。

### 读取用户

要检索一个文档，您可以使用 `find_one` 方法：

```py
from bson import ObjectId
@app.get("/user")
def get_user(user_id: str):
    db_user = user_collection.find_one(
        {
            "_id": ObjectId(user_id)
            if ObjectId.is_valid(user_id)
            else None
        }
    )
    if db_user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    user_response = UserResponse(
        id=str(db_user["_id"]), **db_user
    )
    return user_response
```

如果指定的用户不存在，它将返回一个状态码为 404 的响应。

在 Mongo 中，文档的 ID 不会以纯文本形式存储，而是一个 12 字节的对象。这就是为什么在查询数据库时需要初始化一个专门的 `bson.ObjectId`，并在通过响应返回值时显式解码到 `str`。

然后，您可以使用 `uvicorn` 启动服务器：

```py
$ uvicorn main:app
```

您可以在交互式文档页面看到端点：[`localhost:8000/docs`](http://localhost:8000/docs)。确保您彻底测试每个端点及其之间的交互。

By integrating MongoDB with FastAPI, you gain the ability to handle dynamic, schemaless data structures, which is a significant advantage in many modern web applications. This recipe has equipped you with the knowledge to set up MongoDB, define models and collections, and perform CRUD operations, providing a solid foundation for building versatile and scalable applications with FastAPI and MongoDB.

## See also

You can dig into how to use the **PyMongo** Python client by reading the official documentation:

+   *PyMongo* *documentation*: [`pymongo.readthedocs.io/en/stable/`](https://pymongo.readthedocs.io/en/stable/)

# Working with data validation and serialization

Effective data validation stands as a cornerstone of robust web applications, ensuring that incoming data meets predefined criteria and remains safe for processing.

FastAPI harnesses the power of Pydantic, a Python library dedicated to data validation and serialization. By integrating Pydantic models, FastAPI streamlines the process of validating and serializing data, offering an elegant and efficient solution. This recipe shows how to utilize Pydantic models within FastAPI applications, exploring how they enable precise validation and seamless data serialization.

## Getting ready

**Pydantic models** are essentially Python classes that define the structure and validation rules of your data. They use Python’s type annotations to validate that incoming data matches the expected format. When you use a Pydantic model in your FastAPI endpoints, FastAPI automatically validates incoming request data against the model.

In this recipe, we’re going to use Pydantic’s email validator, which comes with the default `pydantic` package distribution. However, it needs to be installed in your environment. You can do this by running the following command:

```py
$ pip install pydantic[email]
```

Once the package has been installed, you are ready to start this recipe.

## How to do it…

Let’s use it in the previous project. In the `main.py` module, we’ll modify the `UserCreate` class, which is used to accept only valid `email` fields:

```py
from typing import Optional
from pydantic import BaseModel, EmailStr
class UserCreate(BaseModel):
    name: str
name is a required string and email must be a valid email address. FastAPI will automatically use this model to validate incoming data for any endpoint that expects a UserCreate object.
Let’s say you try to add a user at the `POST /user` endpoint with an invalid user information body, as shown here:

```

{

"name": "John Doe",

"email": "invalidemail.com",

}

```py

 You will get a `422` response with a message body specifying the invalid fields.
Serialization and deserialization concepts
**Serialization** is the process of converting complex data types, such as Pydantic models or database models, into simpler formats such as JSON, which can be easily transmitted over the network. **Deserialization** is the reverse process, converting incoming data into complex Python types.
FastAPI handles serialization and deserialization automatically using Pydantic models. When you return a Pydantic model from an endpoint, FastAPI serializes it to JSON. Conversely, when you accept a Pydantic model as an endpoint parameter, FastAPI deserializes the incoming JSON data into the model.
For example, the `get_user` endpoint from the NoSQL example can be improved further like so:

```

class UserResponse(User):

id: str

@app.get("/user")

def get_user(user_id: str) -> UserResponse:

db_user = user_collection.find_one(

{

"_id": ObjectId(user_id)

if ObjectId.is_valid(user_id)

else None

}

)

if db_user is None:

raise HTTPException(

status_code=404,

detail="User not found"

)

db_user["id"] = str(db_user["_id"])

User object and then serializes the returned UserResponse object back into JSON.

This automatic serialization and deserialization make working with JSON data in FastAPI straightforward and type-safe.

Advanced validation techniques

**Pydantic** offers a range of advanced validation techniques that you can leverage in FastAPI. These include custom validators and complex data types.

`@field_validator`.

For example, you could add a validator to ensure that a user’s age is within a certain range:

```py
from pydantic import BaseModel, EmailStr, field_validator
class User(BaseModel):
    name: str
    email: EmailStr
    age: int
@field_validator("age")
    def validate_age(cls, value):
        if value < 18 or value > 100:
            raise ValueError(
                "Age must be between 18 and 100"
            )
age field of the User model is between 18 and 100.
If the validation fails, a descriptive error message is automatically returned to the client.
`list`, `dict`, and custom types, allowing you to define models that closely represent your data structures.
For instance, you can have a model with a list of items:

```

class Tweet(BaseModel):

content: str

hashtags: list[str]

class User(BaseModel):

name: str

email: EmailStr

age: Optional[int]

用户模型有一个可选的 tweets 字段，它是一个 Tweet 对象的列表。

通过利用 Pydantic 的高级验证功能，您可以确保 FastAPI 应用程序处理的数据不仅格式正确，而且符合您的特定业务逻辑和约束。这为在 FastAPI 应用程序中处理数据验证和序列化提供了一种强大且灵活的方法。

参见

您可以在文档页面了解更多关于 Pydantic 验证器的潜力：

+   *Pydantic* *验证器*: [`docs.pydantic.dev/latest/concepts/validators/`](https://docs.pydantic.dev/latest/concepts/validators/)

处理文件上传和下载

在 Web 应用程序中处理文件是一个常见的需求，无论是上传用户头像、下载报告还是处理数据文件。FastAPI 提供了高效且易于实现的文件上传和下载方法。本食谱将指导您如何设置和实现 FastAPI 中的文件处理。

准备工作

让我们创建一个新的项目目录，名为`uploads_and_downloads`，其中包含一个名为`main.py`的模块和一个名为`uploads`的文件夹。这将包含应用程序侧的文件。目录结构将如下所示：

```py
uploads_and_downloads/
|─ uploads/
|─ main.py
```

我们现在可以继续创建适当的端点。

如何操作...

要在 FastAPI 中处理文件上传，您必须使用 FastAPI 中的`File`和`UploadFile`类。`UploadFile`类特别有用，因为它提供了一个异步接口，并将大文件滚存到磁盘以避免内存耗尽。

在`main.py`模块中，您可以定义如下上传文件的端点：

```py
from fastapi import FastAPI, File, UploadFile
app = FastAPI()
@app.post("/uploadfile")
async def upload_file(
    file: UploadFile = File(...)):
    return {"filename": file.filename}
```

在此示例中，`upload_file`是一个端点，它接受一个上传的文件并返回其文件名。文件以`UploadFile`对象的形式接收，然后您可以将其保存到磁盘或进一步处理。

实现文件上传

在实现文件上传时，正确处理文件数据至关重要，以确保文件保存时不会损坏。以下是如何将上传的文件保存到服务器上目录的一个示例。

创建一个新的项目文件夹`uploads_downloads`。

在`main.py`模块中创建`upload_file`端点：

```py
import shutil
from fastapi import FastAPI, File, UploadFile
app = FastAPI()
@app.post("/uploadfile")
async def upload_file(
    file: UploadFile = File(...),
):
    with open(
f"uploads/{file.filename}", "wb"
    ) as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}
```

此代码片段在`uploads`目录中以写二进制模式打开一个新文件，并使用`shutil.copyfileobj`将文件内容从`UploadFile`对象复制到新文件。

重要提示

在生产环境中，请记住适当地处理异常和错误，特别是对于较大的文件

创建一个包含一些文本的文本文件`content.txt`。

通过运行 `uvicorn main:app` 命令来启动服务器。然后，访问交互式文档；你会观察到我们刚刚为文件上传创建的端点包含一个强制字段，提示用户上传文件。通过上传文件测试端点，你会发现上传的文件位于指定的 `uploads` 文件夹中。

管理文件下载和存储

下载文件是上传的逆过程。在 FastAPI 中，你可以轻松设置一个端点来提供文件下载。`FileResponse` 类对此特别有用。它从服务器流式传输文件到客户端，这使得为大型文件提供服务变得高效。

这里是一个简单的文件下载端点：

```py
from fastapi.responses import FileResponse
@app.get(
    "/downloadfile/{filename}",
    response_class=FileResponse,
)
async def download_file(filename: str):
    if not Path(f"uploads/{filename}").exists():
        raise HTTPException(
            status_code=404,
            detail=f"file {filename} not found",
        )
    return FileResponse(
        path=f"uploads/{filename}", filename=filename
    )
```

在这个例子中，`download_file` 是一个端点，它从 `uploads` 目录提供文件供下载。在这里，`FileResponse` 会根据文件类型自动设置适当的内容类型头，并处理将文件流式传输到客户端。

文件内容将是端点的响应体。

处理文件存储是另一个关键方面，尤其是在处理大量文件或大文件大小时。通常建议将文件存储在专门的文件存储系统中，而不是直接存储在您的 Web 服务器上。可以将云存储解决方案如 **Amazon S3**、**Google Cloud Storage** 或 **Azure Blob Storage** 集成到您的 FastAPI 应用程序中，以实现可扩展和安全的文件存储。此外，考虑实施清理程序或归档策略来管理您存储的文件的生命周期。

参见

你可以在官方文档页面上了解更多关于如何管理上传文件的信息：

+   *FastAPI 请求* *文件*: [`fastapi.tiangolo.com/tutorial/request-files/`](https://fastapi.tiangolo.com/tutorial/request-files/)

处理异步数据操作

**异步编程** 是 FastAPI 的一个核心特性，它允许你开发高度高效的 Web 应用程序。它允许你的应用程序同时处理多个任务，使其特别适合 I/O 密集型操作，如数据库交互、文件处理和网络通信。

让我们深入探讨在 FastAPI 中利用异步编程进行数据操作，以增强应用程序的性能和响应能力。

准备工作

FastAPI 是基于 Starlette 和 Pydantic 构建的，它们为使用 `asyncio` 库和 `async`/`await` 语法在 Python 中编写异步代码提供了一个强大的基础。

`asyncio` 库允许你编写非阻塞代码，在等待 I/O 操作完成时可以暂停其执行，然后从上次停止的地方继续执行，而无需阻塞主执行线程。

这个示例展示了在简单、实用的例子中使用 `asyncio` 和 FastAPI 的好处。

如何操作…

让我们创建一个包含两个端点的应用程序，一个运行睡眠操作，另一个也运行睡眠操作但以异步模式运行。创建一个新的项目文件夹`async_example`，包含`main.py`模块。按照以下内容填充模块。

1.  让我们先创建 FastAPI 服务器对象类：

    ```py
    from fastapi import FastAPI
    app = FastAPI()
    ```

    2.  现在，让我们创建一个睡眠 1 秒的端点：

    ```py
    import time
    @app.get("/sync")
    def read_sync():
        time.sleep(2)
        return {
            "message": "Synchrounouns blocking endpoint"
        }
    ```

    睡眠操作代表在实际场景中从数据库获取响应的等待时间。

    3.  现在，让我们为`async def`版本创建相同的端点。睡眠操作将是来自`asyncio`模块的 sleep 函数：

    ```py
    import asyncio
    @app.get("/async")
    async def read_async():
        await asyncio.sleep(2)
        return {
            "message": 
            "Asynchronous non-blocking endpoint"
        }
    ```

现在，我们有两个端点，`GET /sync`和`GET/async`，它们除第二个包含非阻塞睡眠操作外，其他都相似。

一旦我们有了带有端点的应用程序，让我们创建一个单独的 Python 脚本来测量服务流量需求的时间。让我们称它为`timing_api_calls.py`，并通过以下步骤开始构建它。

1.  让我们定义运行服务器的函数：

    ```py
    import uvicorn
    from main import app
    def run_server():
        uvicorn.run(app, port=8000, log_level="error")
    ```

    2.  现在，让我们将服务器的开始定义为上下文管理器：

    ```py
    from contextlib import contextmanager
    from multiprocessing import Process
    @contextmanager
    def run_server_in_process():
        p = Process(target=run_server)
        p.start()
        time.sleep(2)  # Give the server a second to start
        print("Server is running in a separate process")
        yield
        p.terminate()
    ```

    3.  现在，我们可以定义一个函数，该函数向指定的路径端点发送*n*个并发请求：

    ```py
    async def make_requests_to_the_endpoint(
        n: int, path: str
    ):
        async with AsyncClient(
            base_url="http://localhost:8000"
        ) as client:
            tasks = (
                client.get(path, timeout=float("inf"))
                for _ in range(n)
            )
            await asyncio.gather(*tasks)
    ```

    4.  在这一点上，我们可以将操作组合到主函数中，为每个端点调用*n*次，并将服务所有调用的时间打印到终端：

    ```py
    async def main(n: int = 10):
        with run_server_in_process():
            begin = time.time()
            await make_requests_to_the_endpoint(n,
                                                "/sync")
            end = time.time()
            print(
                f"Time taken to make {n} requests "
                f"to sync endpoint: {end - begin} seconds"
            )
            begin = time.time()
            await make_requests_to_the_endpoint(n,
                                                "/async")
            end = time.time()
            print(
                f"Time taken to make {n} requests "
                f"to async endpoint: {end - begin}
                seconds"
            )
    ```

    5.  最后，我们可以在`asyncio`事件循环中运行主函数：

    ```py
    if __name__ == "__main__":
        asyncio.run(main())
    ```

现在我们已经构建了我们的计时脚本，让我们从命令终端按照以下方式运行它：

```py
10, your output will likely resemble the one on my machine:

```

发送到同步端点的 10 次请求所需时间：2.3172452449798584 秒

发送到异步端点的 10 次请求所需时间：2.3033862113952637 秒

```py

 It looks like there is no improvement at all with using asyncio programming.
Now, try to set the number of calls to `100`:

```

if __name__ == "__main__":

asyncio.run(main(n=100))

```py

 The output will likely be more like this:

```

发送到同步端点的 100 次请求所需时间：6.424988269805908 秒

发送到异步端点的 100 次请求所需时间：2.423431873321533 秒

```py

 This improvement is certainly noteworthy, and it’s all thanks to the use of asynchronous functions.
There’s more…
Asynchronous data operations can significantly improve the performance of your application, particularly when dealing with high-latency operations such as database access. By not blocking the main thread while waiting for these operations to complete, your application remains responsive and capable of handling other incoming requests or tasks.
If you already wrote CRUD operations synchronously, as we did in the previous recipe, *Understanding CRUD operations with SQLAlchemy*, implementing asynchronous CRUD operations in FastAPI involves modifying your standard CRUD functions so that they’re asynchronous with the `sqlalchemy[asyncio]` library. Similarly to SQL, for NoSQL, you will need to use the `motor` package, which is the asynchronous MongoDB client built on top of `pymongo`.
However, it’s crucial to use asynchronous programming judiciously. Not all parts of your application will benefit from asynchrony, and in some cases, it can introduce complexity. Here are some best practices for using asynchronous programming in FastAPI:

*   **Use Async for I/O-bound operations**: Asynchronous programming is most beneficial for I/O-bound operations (such as database access, file operations, and network requests). CPU-bound tasks that require heavy computation might not benefit as much from asynchrony.
*   **Database transactions**: When working with databases asynchronously, be mindful of transactions. Ensure that your transactions are correctly managed to maintain the integrity of your data. This often involves using context managers (async with) to handle sessions and transactions.
*   **Error handling**: Asynchronous code can make error handling trickier, especially with multiple concurrent tasks. Use try-except blocks to catch and handle exceptions appropriately.
*   `async` and `await` in your test cases as needed.

By understanding and applying these concepts, you can build applications that are not only robust but also capable of performing optimally under various load conditions. This knowledge is a valuable addition to your skillset as a modern web developer working with FastAPI.
See also
An overview of the concurrency use of the `asyncio` library in FastAPI can be found on the documentation page:

*   *FastAPI* *C**oncurrency*: [`fastapi.tiangolo.com/async/`](https://fastapi.tiangolo.com/async/)

To integrate `async`/`await` syntax with **SQLAlchemy**, you can have a look at documentation support:

*   *SQLAlchemy* *Asyncio*: [`docs.sqlalchemy.org/en/20/orm/extensions/asyncio.xhtml`](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.xhtml)

*Chapter 6*, *Integrating FastAPI with SQL Databases*, will focus on SQL database interactions. Here, you can find examples of integrating `asyncio` with `sqlalchemy`.
To integrate `asyncio` with `motor`, which is built on top of `pymongo`:

*   *Motor asynchronous* *driver*: [`motor.readthedocs.io/en/stable/`](https://motor.readthedocs.io/en/stable/)

In *Chapter 7*, *Integrating FastAPI with NoSQL Databases*, you will find examples of motor integration with FastAPI.
Securing sensitive data and best practices
In the realm of web development, the security of sensitive data is paramount.
This recipe is a checklist of best practices for securing sensitive data in your FastAPI applications.
Getting ready
First and foremost, it’s crucial to understand the types of data that need protection. *Sensitive data* can include anything from passwords and tokens to personal user details. Handling such data requires careful consideration and adherence to security best practices.
Understanding the types of data that require protection sets the foundation for implementing robust security measures, such as leveraging environment variables for sensitive configurations, a key aspect of data security in app development.
Instead of hardcoding these values in your source code, they should be stored in environment variables, which can be accessed securely within your application. This approach not only enhances security but also makes your application more flexible and easier to configure across different environments.
Another important practice is encrypting sensitive data, particularly passwords. FastAPI doesn’t handle encryption directly, but you can use libraries such as `bcrypt` or `passlib` to hash and verify passwords securely.
This recipe will provide a checklist of good practices to apply to secure sensitive data.
How to do it…
Securely handling data in FastAPI involves more than just encryption; it encompasses a range of practices that are designed to protect data throughout its life cycle in your application.
Here is a list of good practices to apply when securing your application.

*   **Validation and sanitization**: Use the Pydantic model to validate and sanitize incoming data, as shown in the *Working with data validation and serialization* recipe. Ensure the data conforms to expected formats and values, reducing the risk of injection attacks or malformed data causing issues.

    Be cautious with data that will be output to users or logs. Sensitive information should be redacted or anonymized to prevent accidental disclosure.

*   **Access control**: Implement robust access control mechanisms to ensure that users can only access the data they are entitled to. This can involve **role-based access control** (**RBAC**), permission checks and properly managing user authentication. You will discover more about this in the *Setting up* *RBAC* recipe in *Chapter 4*, *Authentication* *and Authorization*.
*   **Secure communication**: Use HTTPS to encrypt data in transit. This prevents attackers from intercepting sensitive data that’s sent to or received from your application.
*   **Database security**: Ensure that your database is securely configured. Use secure connections, avoid exposing database ports publicly, and apply the principle of least privilege to database access.
*   **Regular updates**: Keep your dependencies, including FastAPI and its underlying libraries, up to date. This helps protect your application from vulnerabilities discovered in older versions of the software.

Some of them will be covered in detail throughout this book.
There’s more…
Managing sensitive data extends beyond immediate security practices and involves considerations for data storage, transmission, and even deletion.
Here’s a checklist of more general practices so that you can secure your data, regardless of whatever code you are writing:

*   **Data storage**: Store sensitive data only when necessary. If you don’t need to store data such as credit card numbers or personal identification numbers, then don’t. When storage is necessary, ensure it is encrypted and that access is tightly controlled.
*   **Data transmission**: Be cautious when transmitting sensitive data. Use secure APIs and ensure that any external services you interact with also follow security best practices.
*   **Data retention and deletion**: Have clear policies on data retention and deletion. When data is no longer needed, ensure it is deleted securely, leaving no trace in backups or logs.
*   **Monitoring and logging**: Implement monitoring to detect unusual access patterns or potential breaches. However, be careful with what you log. Avoid logging sensitive data and ensure that logs are stored securely and are only accessible to authorized personnel.

By applying these practices, you can significantly enhance the security posture of your applications, protecting both your users and your organization from potential data breaches and ensuring compliance with data protection regulations. As a developer, understanding and implementing data security is not just a skill but a responsibility in today’s digital landscape. In the next chapter, we will learn how to build an entire RESTful API with FastAPI.

```

```py

```

```py

```
