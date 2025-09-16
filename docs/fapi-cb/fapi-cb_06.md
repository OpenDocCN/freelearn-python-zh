

# 第六章：将 FastAPI 与 SQL 数据库集成

现在，我们将开始一段旅程，充分利用 SQL 数据库在 FastAPI 应用程序中的全部潜力。本章精心设计，旨在指导你深入了解利用 **SQLAlchemy** 的细微差别，这是一个强大的 SQL 工具包和 **对象关系映射**（**ORM**）库，适用于 Python。从设置你的数据库环境到实现复杂的 **创建、读取、更新和删除**（**CRUD**）操作，以及管理复杂的关系，本章提供了一个全面蓝图，以无缝集成 SQL 数据库与 FastAPI。

通过创建一个基本的票务平台，你将实际参与配置 SQLAlchemy 与 FastAPI，创建反映你的应用程序数据结构的数据库模型，并构建高效、安全的 CRUD 操作。

此外，你还将探索使用 **Alembic** 管理数据库迁移，确保你的数据库模式与你的应用程序同步发展而无需麻烦。本章不仅涉及数据处理，还深入到优化 SQL 查询以提升性能、在数据库中保护敏感信息以及管理事务和并发，以确保数据完整性和可靠性。

到本章结束时，你将熟练地集成和管理 SQL 数据库在你的 FastAPI 应用程序中，并具备确保你的应用程序不仅高效和可扩展，而且安全的技能。无论你是从头开始构建新应用程序还是将数据库集成到现有项目中，这里涵盖的见解和技术将赋予你利用 SQL 数据库在 FastAPI 项目中全部力量的能力。

在本章中，我们将介绍以下食谱：

+   设置 SQLAlchemy

+   实现 CRUD 操作

+   与迁移一起工作

+   处理 SQL 数据库中的关系

+   优化 SQL 查询以提升性能

+   在 SQL 数据库中保护敏感数据

+   处理事务和并发

# 技术要求

为了跟随本章的所有食谱，请确保你的设置中包含以下基本要素：

+   **Python**：你的环境应安装有高于 3.9 的 Python 版本。

+   **FastAPI**：它应该安装在你的虚拟环境中，并包含所有需要的依赖项。如果你在前几章中没有这样做，你可以很容易地从你的终端完成它：

    ```py
    $ pip install fastapi[all]
    ```

本章的代码可在 GitHub 上的以下链接找到：[`github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter06`](https://github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter06)

还建议在项目根目录内为项目创建一个虚拟环境，以更好地处理依赖关系并保持项目独立。在你的虚拟环境中，你可以通过使用项目文件夹中 GitHub 仓库的 `requirements.txt` 文件一次性安装所有依赖项：

```py
$ pip install –r requirements.txt
```

由于本章的代码将使用来自 `asyncio` Python 库的 `async`/`await` 语法，你应该已经熟悉它。请随意阅读以下链接了解更多关于 `asyncio` 和 `async`/`await` 语法的信息：

+   [Python 3 库文档](https://docs.python.org/3/library/asyncio.xhtml)

+   [FastAPI 异步教程](https://fastapi.tiangolo.com/async/)

现在我们已经准备好了 一旦一切准备就绪，我们就可以开始准备我们的食谱。

# 设置 SQLAlchemy

要开始任何数据应用，你需要建立数据库连接。本食谱将帮助你设置和配置 `sqlalchemy` 包与 **SQLite** 数据库，以便你可以在应用中使用 SQL 数据库的优势。

## 准备就绪

项目将会相当大，因此我们将把应用的工作模块放在一个名为 `app` 的文件夹中，该文件夹位于我们称之为 `ticketing_system` 的根项目文件夹下。

你需要在你的环境中安装 `fastapi`、`sqlalchemy` 和 `aiosqlite` 以使用这个食谱。这个食谱旨在与版本高于 2.0.0 的 `sqlalchemy` 一起工作。你仍然可以使用版本 1；然而，需要一些适配。你可以在以下链接找到迁移指南：[SQLAlchemy 2.0 迁移指南](https://docs.sqlalchemy.org/en/20/changelog/migration_20.xhtml)。

如果你还没有使用存储库中的 `requirements.txt` 文件安装包，你可以通过运行以下命令来完成：

```py
$ pip install fastapi[all] "sqlalchemy>=2.0.0" aiosqlite
```

一旦包正确安装，你就可以按照食谱进行操作。

## 如何操作...

使用 `sqlalchemy` 设置通用 SQL 数据库连接将经过以下步骤：

1.  创建映射对象类，以匹配数据库表

1.  创建抽象层、引擎和会话以与数据库通信

1.  在服务器启动时初始化数据库连接

### 创建映射对象类

在 `app` 文件夹中，让我们创建一个名为 `database.py` 的模块，然后创建一个类对象来跟踪票务，如下所示：

```py
from sqlalchemy import Column, Float, ForeignKey, Table
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
)
class Base(DeclarativeBase):
    pass
class Ticket(Base):
    __tablename__ = "tickets"
    id: Mapped[int] = mapped_column(primary_key=True)
    price: Mapped[float] = mapped_column(nullable=True)
    show: Mapped[str | None]
    user: Mapped[str | None]
```

我们刚刚创建了一个 `Ticket` 类，它将被用来将我们的 SQL 数据库中的 `tickets` 表进行匹配。

### 创建抽象层

在 SQLAlchemy 中，*引擎* 管理数据库连接并执行 SQL 语句，而 *会话* 允许在事务性上下文中查询、插入、更新和删除数据，确保一致性和原子性。会话绑定到引擎以与数据库通信。

我们将首先创建一个返回引擎的函数。在一个名为 `db_connection.py` 的新模块中，位于 `app` 文件夹下，让我们按照以下方式编写函数：

```py
from sqlalchemy.ext.asyncio import (
    create_async_engine,
)
from sqlalchemy.orm import sessionmaker
SQLALCHEMY_DATABASE_URL = (
    "sqlite+aiosqlite:///.database.db"
)
def get_engine():
    return create_async_engine(
        SQLALCHEMY_DATABASE_URL, echo=True
    )
```

你可能已经注意到，`SQLALCHEMY_DATABASE_URL` 数据库 URL 使用了 `sqlite` 和 `aiosqlite` 模块。

这意味着我们将使用 SQLite 数据库，操作将通过支持 `asyncio` 库的 `aiosqlite` 异步库来完成。

然后，我们将使用会话创建器来指定会话将是异步的，如下所示：

```py
from sqlalchemy.ext.asyncio import (
    AsyncSession,
)
AsyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=get_engine(),
    class_=AsyncSession,
)
async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session
```

`get_db_session`函数将被用作与数据库交互的每个端点的依赖项。

### 初始化数据库连接

一旦我们有了抽象层，我们需要在服务器运行时创建我们的 FastAPI 服务器对象并启动数据库类。我们可以在`app`文件夹下的`main.py`模块中这样做：

```py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database import Base
from app.db_connection import (
    AsynSessionLocal,
    get_db_session
)
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        yield
    await engine.dispose()ispose()
app = FastAPI(lifespan=lifespan)
```

为了在启动事件中指定服务器操作，我们使用了`lifespan`参数。

我们已经准备好将我们的应用程序与数据库连接。

## 它是如何工作的…

`Ticket`数据库映射类的创建告诉我们应用程序数据库的结构，并且会话将管理事务。然后，引擎不仅会执行操作，还会将映射类与数据库进行比较，如果缺少任何表，它将创建这些表。

为了检查我们的应用程序是否与数据库通信，让我们从项目根目录的命令行启动服务器：

```py
$ uvicorn app.main:app
```

你应该在命令输出中看到消息日志，表明已创建表 tickets。此外，使用你偏好的数据库阅读器打开`.database.db`文件，你应该会看到在`database.py`模块中定义的模式下的表。

## 参见

你可以在官方文档页面上了解更多关于如何使用 SQLAlchemy 设置数据库以及如何使其与`asyncio`模块兼容的信息：

+   *如何设置 SQLAlchemy* *数据库*: [`docs.sqlalchemy.org/en/20/orm/quickstart.xhtml`](https://docs.sqlalchemy.org/en/20/orm/quickstart.xhtml)

+   *SQLAlchemy* `asyncio` *扩展* *参考*: [`docs.sqlalchemy.org/en/20/orm/extensions/asyncio.xhtml`](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.xhtml)

在这个例子中，我们通过指定以下内容使用了 SQLite 数据库：

```py
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///.database.db"
```

然而，你可以使用 SQLAlchemy 与多个 SQL 数据库交互，例如`asyncio`支持的驱动程序和数据库地址。

例如，对于 MySQL，连接字符串看起来会是这样：

```py
mysql+aiomysql://user:password@host:port/dbname[?key=value&key=value...]
```

在这种情况下，你需要在你的环境中安装`aiomysql`包。

你可以在官方文档页面查看更多信息：

+   SQLAlchemy MySQL 方言: [`docs.sqlalchemy.org/en/20/dialects/mysql.xhtml`](https://docs.sqlalchemy.org/en/20/dialects/mysql.xhtml)

+   SQLAlchemy PostgreSQL 方言: [`docs.sqlalchemy.org/en/20/dialects/postgresql.xhtml`](https://docs.sqlalchemy.org/en/20/dialects/postgresql.xhtml)

# 实现 CRUD 操作

使用 RESTful API 的 CRUD 操作可以通过 HTTP 方法（`POST`、`GET`、`PUT`和`DELETE`）实现，用于网络服务。这个配方演示了如何使用 SQLAlchemy 和`asyncio`在 SQL 数据库上异步构建 CRUD 操作以及相应的端点。

## 准备工作

在开始食谱之前，您需要有一个数据库连接、数据集中的表以及代码库中匹配的类。如果您完成了前面的食谱，应该已经准备好了。

## 如何做…

我们将首先在`app`文件夹下创建一个`operations.py`模块，按照以下步骤包含我们的数据库操作。

1.  首先，我们可以设置操作以将新票添加到数据库，如下所示：

    ```py
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.future import select
    from app.database import Ticket
    async def create_ticket(
        db_session: AsyncSession,
        show_name: str,
        user: str = None,
        price: float = None,
    ) -> int:
        ticket = Ticket(
            show=show_name,
            user=user,
            price=price,
        )
        async with db_session.begin():
            db_session.add(ticket)
            await db_session.flush()
            ticket_id = ticket.id
            await db_session.commit()
        return ticket_id
    ```

    函数在保存时会返回附加到票上的 ID。

1.  然后，让我们创建一个获取票的功能：

    ```py
    async def get_ticket(
        db_session: AsyncSession, ticket_id: int
    ) -> Ticket | None:
        query = (
            select(Ticket)
            .where(Ticket.id == ticket_id)
        )
        async with db_session as session:
            tickets = await session.execute(query)
            return tickets.scalars().first()
    ```

    如果找不到票，函数将返回一个`None`对象。

1.  然后，我们构建一个仅更新票价的操作：

    ```py
    async def update_ticket_price(
        db_session: AsyncSession,
        ticket_id: int,
        new_price: float,
    ) -> bool:
        query = (
            update(Ticket)
            .where(Ticket.id == ticket_id)
            .values(price=new_price)
        )
        async with db_session as session:
            ticket_updated = await session.execute(query)
            await session.commit()
            if ticket_updated.rowcount == 0:
                return False
            return True
    ```

    如果操作无法删除任何票，函数将返回`False`。

1.  为了完成 CRUD 操作，我们定义了一个`delete_ticket`操作：

    ```py
    async def delete_ticket(
        db_session: AsyncSession, ticket_id
    ) -> bool:
        async with db_session as session:
            tickets_removed = await session.execute(
                delete(
                    Ticket
                ).where(Ticket.id == ticket_id)
            )
            await session.commit()
            if tickets_removed.rowcount == 0:
                return False
            return True
    ```

    与更新操作类似，如果找不到要删除的票，函数将返回`False`。

1.  在定义操作后，我们可以在`main.py`模块中创建相应的端点来公开它们。

    在定义应用服务器后，让我们立即为创建操作做这件事：

    ```py
    from typing import Annotated
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.db_connection import (
        AsyncSessionLocal,
        get_engine,
        get_session
    )
    from app.operations import create_ticket
    # rest of the code 
    class TicketRequest(BaseModel):
        price: float | None
        show: str | None
        user: str | None = None
    @app.post("/ticket", response_model=dict[str, int])
    async def create_ticket_route(
        ticket: TicketRequest,
        db_session: Annotated[
            AsyncSession,
            Depends(get_db_session)
        ]
    ):
        ticket_id = await create_ticket(
            db_session,
            ticket.show,
            ticket.user,
            ticket.price,
        )
        return {"ticket_id": ticket_id}
    ```

    剩余的操作也可以以相同的方式公开。

练习

与`create_ticket`操作类似，使用相应的端点公开其他操作（获取、更新和删除）。

## 它是如何工作的…

用于与数据库交互的函数通过端点公开。这意味着外部用户将通过调用相应的端点来执行操作。

让我们验证端点是否正确工作。

按照惯例，从命令行启动服务器，运行以下命令：

```py
$ uvicorn app.main:app
```

然后，转到交互式文档链接`http://localhost:8000/docs`，您将看到您刚刚创建的端点。尝试不同的组合并查看`.database.db`数据库文件中的结果。

您刚刚使用`sqlalchemy`和`asyncio`库创建了对 SQL 数据库的 CRUD 操作。

练习

在根项目文件夹中创建一个`tests`文件夹，并编写操作函数和端点的所有单元测试。您可以参考*第五章*，*测试和调试 FastAPI 应用程序*，了解如何对 FastAPI 应用程序进行单元测试。

# 与迁移一起工作

数据库迁移让您可以版本控制数据库模式，并使其在各个环境中保持一致。它们还有助于自动化数据库更改的部署，并跟踪模式演化的历史。

食谱向您展示了如何使用**Alembic**，这是一个流行的 Python 数据库迁移管理工具。您将学习如何创建、运行和回滚迁移，以及如何将它们与您的票务系统集成。

## 准备工作

要使用这个配方，你需要在你的环境中安装`alembic`。如果你没有通过 GitHub 仓库中的`requirements.txt`文件安装它，可以在命令行中输入以下内容来安装：

```py
$ pip install alembic
```

你还需要确保你至少有一个与你要创建的数据库中的表相对应的类。如果你没有，请回到*设置 SQLAlchemy*配方并创建一个。如果你已经在运行应用程序，请删除应用程序创建的`.database.db`文件。

## 如何做到这一点…

要配置 Alembic 并管理数据库迁移，请按照以下步骤进行。

1.  第一步是设置`alembic`。在项目根目录中，在命令行中运行以下命令：

    ```py
    alembic.ini file and an alembic folder with some files inside it. The alembic.ini file is a configuration file for alembic.If you copy the project from the GitHub repository make sure to delete the existing `alembic` folder before running the `alembic` `init` command.
    ```

1.  找到`sqlalchemy.url`变量，并将数据库 URL 设置为以下：

    ```py
    sqlalchemy.url = sqlite:///.database.db
    ```

    这指定了我们正在使用 SQLite 数据库。如果你使用的是不同的数据库，请相应地调整。

1.  `alembic`目录包含一个版本文件夹和一个包含创建我们数据库迁移变量的`env.py`文件。

    打开`env.py`文件，找到`target_metadata`变量。将其值设置为我们的应用程序的元数据，如下所示：

    ```py
    from app.database import Base
    target_metadata = Base.metadata
    ```

    我们现在可以创建我们的第一个数据库迁移脚本并应用迁移。

1.  从命令行执行以下命令以创建初始迁移：

    ```py
    alembic/versions folder.
    ```

1.  确保你已经删除了现有的`.database.db`文件，然后使用以下命令执行我们的第一个迁移：

    ```py
    .database.db file with the tickets table in it.
    ```

## 它是如何工作的…

一旦我们有了我们数据库的第一个版本，让我们看看迁移是如何工作的。

假设我们想在应用程序已经部署在生产环境中时更改`database.py`模块中的表，以便在更新时不能删除任何记录。

向数据库添加一些票据，然后在代码中，让我们添加一个名为`sold`的新字段，以指示票据是否已售出：

```py
class Ticket(Base):
    __tablename__ = "tickets"
    id: Mapped[int] = mapped_column(primary_key=True)
    price: Mapped[float] = mapped_column(nullable=True)
    show: Mapped[str | None]
    user: Mapped[str | None]
    sold: Mapped[bool] = mapped_column(default=False)
```

要创建一个新的迁移，请运行以下命令：

```py
$ alembic revision –-autogenerate -m "Add sold field"
```

你将在`alembic/versions`文件夹中找到一个新脚本。

再次运行迁移命令：

```py
$ alembic upgrade head
```

打开数据库，你会看到`tickets`表模式已添加了`sold`字段，并且没有记录被删除。

你刚刚创建了一个迁移策略，它将在运行时无缝更改我们的数据库，而不会丢失任何数据。从现在开始，请记住使用迁移来跟踪数据库模式的变化。

## 参见

你可以在官方文档链接中了解更多关于如何使用 Alembic 管理数据库迁移的信息：

+   *设置* *Alembic*：[`alembic.sqlalchemy.org/en/latest/tutorial.xhtml`](https://alembic.sqlalchemy.org/en/latest/tutorial.xhtml)

+   *自动生成* *迁移*：[`alembic.sqlalchemy.org/en/latest/autogenerate.xhtml`](https://alembic.sqlalchemy.org/en/latest/autogenerate.xhtml)

# 在 SQL 数据库中处理关系

数据库关系是两个或多个表之间的关联，允许您建模复杂的数据结构并在多个表之间执行查询。在本教程中，您将学习如何为现有的票务系统应用程序实现一对一、多对一和多对多关系。您还将了解如何使用 SQLAlchemy 定义数据库模式关系并查询数据库。

## 准备工作

为了遵循本教程，您需要已经实现应用程序的核心，其中至少包含一个表。如果您已经做到了这一点，您也将准备好必要的包。我们将继续在我们的票务系统平台应用程序上工作。

## 如何操作…

我们现在将继续设置关系。我们将为每种类型的 SQL 表关系提供一个示例。

### 一对一

我们将通过创建一个新的表格来展示一对一关系，该表格包含有关票的详细信息。

一对一关系用于将特定记录的信息分组到单独的逻辑中。

话虽如此，让我们在`database.py`模块中创建表格。记录将包含有关与票关联的座位、票类型等信息，我们将使用票类型作为可能信息的标签。让我们分两步创建表格。

1.  首先，我们将向现有的`Ticket`类添加票详情引用：

    ```py
    class Ticket(Base):
        __tablename__ = "tickets"
        id: Mapped[int] = mapped_column(primary_key=True)
        price: Mapped[float] = mapped_column(
            nullable=True
        )
        show: Mapped[str | None]
        user: Mapped[str | None]
        sold: Mapped[bool] = mapped_column(default=False)
        details: Mapped["TicketDetails"] = relationship(
            back_populates="ticket"
        )
    ```

1.  然后，我们创建表格以映射票的详细信息，如下所示：

    ```py
    from sqlalchemy import ForeignKey
    class TicketDetails(Base):
        __tablename__ = "ticket_details"
        id: Mapped[int] = mapped_column(primary_key=True)
        ticket_id: Mapped[int] = mapped_column(
            ForeignKey("tickets.id")
    )
        ticket: Mapped["Ticket"] = relationship(
            back_populates="details"
        )
        seat: Mapped[str | None]
        ticket_type: Mapped[str | None]
    ```

一旦数据库类被设置以容纳新表，我们就可以按照以下步骤更新 CRUD 操作。

1.  要更新票务详情，让我们在`operations.py`模块中创建一个专用函数：

    ```py
    async def update_ticket_details(
        db_session: AsyncSession,
        ticket_id: int,
        updating_ticket_details: dict,
    ) -> bool:
        ticket_query = update(TicketDetails).where(
            TicketDetails.ticket_id == ticket_id
        )
        if updating_ticket_details != {}:
            ticket_query = ticket_query.values(
                 *updating_ticket_details
            )
            result = await db_session.execute(
                    ticket_query
                )
            await db_session.commit()
            if result.rowcount == 0:
                    return False
        return True
    ```

    如果没有记录被更新，该函数将返回`False`。

1.  接下来，修改`create_ticket`函数以考虑票的详细信息，并创建一个端点来公开我们刚刚创建的更新操作，如下所示：

    ```py
    async def create_ticket(
        db_session: AsyncSession,
        show_name: str,
        user: str = None,
        price: float = None,
    ) -> int:
        ticket = Ticket(
            show=show_name,
            user=user,
            price=price,
            details=TicketDetails(),
        )
        async with db_session.begin():
            db_session.add(ticket)
            await db_session.flush()
            ticket_id = ticket.id
            await db_session.commit()
        return ticket_id
    ```

    在本例中，每次创建票时，也会创建一个空的票详情记录，以保持数据库的一致性。

这是处理一对一关系的最小配置。我们将继续设置多对一关系。

### 多对一

票可以与活动相关联，活动可以有多个票。为了展示多对一关系，我们将创建一个`events`表，该表将与`tickets`表相关联。让我们按以下步骤进行：

让我们先在`tickets`表中创建一个列，该列将容纳`database.py`模块中`events`表的引用，如下所示：

```py
class Ticket(Base):
    __tablename__ = "tickets"
    # skip existing columns
    event_id: Mapped[int | None] = mapped_column(
        ForeignKey("events.id")
    )
    event: Mapped["Event | None"] = relationship(
        back_populates="tickets"
Event class to map the events table into the database:

```

class Event(Base):

__tablename__ = "events"

id: Mapped[int] = mapped_column(primary_key=True)

name: Mapped[str]

tickets: Mapped[list["Ticket"]] = relationship(

back_populates="event"

)

```py

 `ForeignKey`, in this case, is defined only in the `Ticket` class since the event associated can be only one.
This is all you need to create a many-to-one relationship.
Exercise
You can add to the application the operations to create an event and specify the number of tickets to create with it. Once you’ve done this, expose the operation with the corresponding endpoint.
Many to many
Let’s imagine that we have a list of sponsors that can sponsor our events. Since we can have multiple sponsors that can sponsor multiple events, this situation is best representative of a many-to-many relationship.
To work with many-to-many relationships, we need to define a class for the concerned tables and another class to track the so-called *association table*.
Let’s start by defining a column to accommodate relationships in the `Event` class:

```

class Event(Base):

__tablename__ = "events"

# 现有列

sponsors: Mapped[list["Sponsor"]] = relationship(

secondary="sponsorships",

back_populates="events",

赞助商表格：

```py
class Sponsor(Base):
    __tablename__ = "sponsors"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    events: Mapped[list["Event"]] = relationship(
        secondary="sponsorships",
        back_populates="sponsors",
    )
```

如您所注意到的，该类包含用于容纳`events`引用的列。

最后，我们可以定义一个关联表，该表将是`sponsorships`表：

```py
class Sponsorship(Base):
    __tablename__ = "sponsorships"
    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id"), primary_key=True
    )
    sponsor_id: Mapped[int] = mapped_column(
        ForeignKey("sponsors.id"), primary_key=True
    )
    amount: Mapped[float] = mapped_column(
        nullable=False, default=10
    )
```

关联表可以包含关于关系本身的信息。例如，在我们的案例中，一条有用的信息是赞助商为活动提供的金额。

这就是您为您的票务系统平台创建多对多关系所需的所有内容。

练习

为了完成您的申请，创建一个包含相关端点的操作函数，以执行以下操作：

- 向数据库添加赞助商。

- 添加带有金额的赞助。如果赞助已经存在，则用新金额替换赞助。

参见

您可以在以下官方文档页面深入了解使用 SQLAlchemy 处理关系的操作：

+   *SQLAlchemy 基本* *关系*：[`docs.sqlalchemy.org/en/20/orm/basic_relationships.xhtml`](https://docs.sqlalchemy.org/en/20/orm/basic_relationships.xhtml)

优化 SQL 查询以提升性能

在数据库管理中，优化 SQL 查询是关键，因为它提高了效率、可扩展性、成本效益、用户满意度、数据完整性、合规性和安全性。

此配方展示了如何通过改进 SQL 查询来提高应用程序的运行速度。使用更少资源和时间的查询可以提升用户满意度和应用程序容量。改进 SQL 查询是一个反复的过程，但您可以采用一些有助于您的技巧。

准备工作

确保您有一个正在运行的应用程序，该应用程序使用 SQLAlchemy 进行数据库交互，或者在整个章节中继续开发票务系统应用程序。此外，对 SQL 和数据库模式设计的基本了解可能有益。

如何操作...

改进 SQL 查询是一个涉及多个步骤的过程。与大多数优化过程一样，许多步骤都是特定于用例的，但有一些通用规则可以帮助整体优化 SQL 查询，例如以下内容：

+   避免使用*N*+1 查询

+   适度使用`JOIN`语句

+   最小化要获取的数据

我们将应用每个具有显著示例。

避免 N+1 查询

当您的应用程序执行一个查询以获取项目列表，然后遍历这些项目以获取相关数据，从而产生 N 个更多查询时，就会发生 N+1 查询问题。

假设我们想要一个端点来显示所有与相关赞助商相关的事件。第一次尝试可能是获取`events`表并遍历事件以获取`sponsors`表。这个解决方案意味着一个查询来获取事件，以及为每个事件获取赞助商的 N 个更多查询，这正是我们想要避免的。

解决方案是在查询中加载所有相关记录以检索相关赞助商。这在技术上称为*预加载*。

在 SQLAlchemy 中，这是通过使用`joinedload`选项来完成的，以便函数操作看起来像这样：

```py
async def get_events_with_sponsors(
    db_session: AsyncSession
) -> list[Event]:
    query = (
        select(Event)
        .options(
joinedload(Event.sponsors)
        )
    )
    async with db_session as session:
        result = await session.execute(query)
        events = result.scalars().all()
    return events
```

`joinedload` 方法将在查询中包含一个 `JOIN` 操作，因此不再需要执行 N 次查询来获取赞助商。

节省使用连接语句

连接表可以使查询更容易阅读。但请注意，只连接您查询所需的表。

假设我们想要获取一个按金额从高到低排序的赞助商名称列表，以获取某个活动的金额。

由于我们需要获取三个表，我们可以使用多个连接。函数看起来会是这样：

```py
async def get_event_sponsorships_with_amount(
    db_session: AsyncSession, event_id: int
):
    query = (
        select(Sponsor.name, Sponsorship.amount)
        .join(
            Sponsorship,
            Sponsorship.sponsor_id == Sponsor.id,
        )
        .join(
            Event,
            Sponsorship.event_id == Event.id
)
        .order_by(Sponsorship.amount.desc())
    )
    async with db_session as session:
        result = await session.execute(query)
        sponsor_contributions = result.fetchall()
    return sponsor_contributions
```

双重连接意味着调用我们不会使用的 `events` 表，因此将查询组织如下会更有效率：

```py
async def get_event_sponsorships_with_amount(
    db_session: AsyncSession, event_id: int
):
    query = (
select(Sponsor.name, Sponsorship.amount)
        .join(
            Sponsorship,
            Sponsorship.sponsor_id == Sponsor.id,
        )
        .where(Sponsorship.event_id == event_id)
        .order_by(Sponsorship.amount.desc())
    )
    async with db_session as session:
        result = await session.execute(query)
        sponsor_contributions = result.fetchall()
    return sponsor_contributions
```

这将返回我们所需的内容，而无需选择 `events` 表。

最小化要获取的数据

获取比所需更多的数据可能会减慢您的查询和应用程序。

使用 SQLAlchemy 的 `load_only` 函数只从数据库中加载特定的列。

想象一下，为了进行市场分析，我们被要求编写一个函数，该函数只获取具有票务 ID、用户和价格的票务列表：

```py
async def get_events_tickets_with_user_price(
    db_session: AsyncSession, event_id: int
) -> list[Ticket]:
    query = (
        select(Ticket)
        .where(Ticket.event_id == event_id)
        .options(
            load_only(
                Ticket.id, Ticket.user, Ticket.price
            )
        )
    )
    async with db_session as session:
        result = await session.execute(query)
        tickets = result.scalars().all()
    return tickets
```

我们现在尝试从该函数检索票务，如下所示：

```py
tickets = await get_events_tickets_with_user_price(
    session, event_id
)
```

您会注意到每个元素只包含 `id`、`user` 和 `price` 字段，如果您尝试访问 `show` 字段，例如，将会出现错误。在大型应用程序中，这可以减少内存使用并使响应更快。

还有更多...

SQL 查询优化不仅涉及配方中显示的内容。通常，选择特定的 SQL 数据库取决于特定的优化需求。

不同的 SQL 数据库在处理这些因素时可能具有不同的优势和劣势，这取决于它们的架构和功能。例如，一些 SQL 数据库可能支持分区、分片、复制或分布式处理，这可以提高数据的可扩展性和可用性。一些 SQL 数据库可能提供更高级的查询优化技术，如基于成本的优化、查询重写或查询缓存，这可以减少查询的执行时间和资源消耗。一些 SQL 数据库可能实现不同的存储引擎、事务模型或索引类型，这可能会影响数据操作的性能和一致性。

因此，在选择特定应用程序的 SQL 数据库时，考虑数据和查询的特征和需求，以及比较可用 SQL 数据库的功能和限制，非常重要。一种好的方法是使用真实数据集和查询对 SQL 数据库的性能进行基准测试，并测量相关指标，如吞吐量、延迟、准确性和可靠性。通过这样做，可以找到给定场景的最佳 SQL 数据库，并确定数据库设计和查询制定中可能需要改进的潜在领域。

在 SQL 数据库中保护敏感数据

敏感数据，如个人信息、财务记录或机密文件，通常存储在 SQL 数据库中，用于各种应用程序和目的。然而，这也使数据面临未经授权访问、盗窃、泄露或损坏的风险。因此，在 SQL 数据库中保护敏感数据并防止恶意攻击或意外错误至关重要。

这个食谱将展示如何将敏感数据，如信用卡信息，存储在 SQL 数据库中。

准备工作

要遵循这个食谱，你需要有一个已经设置好数据库连接的应用程序。

此外，我们还将使用`cryptography`包。如果你还没有通过`requirements.txt`文件安装它，你可以在你的环境中运行以下命令来安装：

```py
$ pip install cryptography
```

对密码学的深入了解可能有益，但并非必需。

如何做……

我们将从零开始创建一个新的表格来存储信用卡信息。其中一些信息，例如信用卡号码和**卡验证值**（**CVV**），将不会以明文形式存储在我们的数据库中，而是加密存储。由于我们需要将其恢复，我们将使用需要密钥的对称加密。让我们通过以下步骤来完成这个过程。

1.  让我们从在`database.py`模块中创建一个与数据库中的`credit_card`表相对应的类开始，如下所示：

    ```py
    class CreditCard(Base):
        __tablename__ = "credit_cards"
        id: Mapped[int] = mapped_column(primary_key=True)
        number: Mapped[str]
        expiration_date: Mapped[str]
        cvv: Mapped[str]
        card_holder_name: Mapped[str]
    ```

    2.  接下来，在`app`文件夹中，我们创建一个名为`security.py`的模块，我们将在这个模块中编写使用**Fernet 对称加密**加密和解密数据的代码，如下所示：

    ```py
    from cryptography.fernet import Fernet
    cypher_key = Fernet.generate_key()
    cypher_suite = Fernet(cypher_key)
    ```

    `cypher_suite`对象将用于定义加密和解密函数。

    值得注意的是，在生产环境中，`cypher_key`对象可以是外部提供轮换服务的一部分，也可以根据业务的安全需求在启动时创建。

    3.  在同一个模块中，我们可以创建一个加密信用卡信息的函数和一个解密它的函数，如下所示：

    ```py
    def encrypt_credit_card_info(card_info: str) -> str:
        return cypher_suite.encrypt(
            card_info.encode()
        ).decode()
    def decrypt_credit_card_info(
        encrypted_card_info: str,
    ) -> str:
        return cypher_suite.decrypt(
            encrypted_card_info.encode()
        ).decode()
    ```

    这些函数将在从数据库写入和读取时使用。

    4.  然后，我们可以在同一个`security.py`模块中编写存储操作，如下所示：

    ```py
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.database import CreditCard
    async def store_credit_card_info(
        db_session: AsyncSession,
        card_number: str,
        card_holder_name: str,
        expiration_date: str,
        cvv: str,
    ):
        encrypted_card_number = encrypt_credit_card_info(
            card_number
        )
        encrypted_cvv = encrypt_credit_card_info(cvv)
        # Store encrypted credit card information
        # in the database
        credit_card = CreditCard(
            number=encrypted_card_number,
            card_holder_name=card_holder_name,
            expiration_date=expiration_date,
            cvv=encrypted_cvv,
        )
        async with db_session.begin():
            db_session.add(credit_card)
            await db_session.flush()
            credit_card_id = credit_card.id
            await db_session.commit()
        return credit_card_id
    ```

    每次函数被等待时，信用卡信息将加密后与机密数据一起存储。

    5.  类似地，我们可以定义一个从数据库中检索加密信用卡信息的函数，如下所示：

    ```py
    async def retrieve_credit_card_info(
        db_session: AsyncSession, credit_card_id: int
    ):
        query = select(CreditCard).where(
            CreditCard.id == credit_card_id
        )
        async with db_session as session:
            result = await session.execute(query)
            credit_card = result.scalars().first()
        credit_card_number = decrypt_credit_card_info(
                credit_card.number
            ),
        cvv = decrypt_credit_card_info(credit_card.cvv)
        card_holder = credit_card.card_holder_name
        expiry = credit_card.expiration_date
        return {
            "card_number": credit_card_number,
            "card_holder_name": card_holder,
            "expiration_date": expiry,
            "cvv": cvv
        }
    ```

    我们已经开发出了代码，可以在我们的数据库中保存机密信息。

练习

我们刚刚看到了如何安全存储敏感数据的基本框架。你可以通过以下步骤自己完成这个功能：

- 为我们的加密操作编写单元测试。在`tests`文件夹中，让我们创建一个新的测试模块，名为`test_security.py`。验证信用卡信息是否安全地保存在我们的数据库中，但信用卡号码和 CVV 字段是加密的。

- 在数据库中创建端点以存储、检索和删除信用卡信息。

- 将信用卡与赞助商关联并管理相关的 CRUD 操作。

参见

我们已经使用 Fernet 对称加密来加密信用卡信息。您可以在以下链接中深入了解：

+   *Fernet 对称* *加密*：[`cryptography.io/en/latest/fernet/`](https://cryptography.io/en/latest/fernet/)

处理事务和并发

在数据库管理的领域，两个关键方面决定了应用程序的可靠性和性能：处理事务和管理并发。

事务，封装一系列数据库操作，通过确保更改作为一个单一的工作单元发生，对于维护数据一致性是基本的。另一方面，并发解决多个用户或进程同时访问共享资源的挑战。

当考虑可能同时尝试访问或修改相同数据的多个事务的场景时，事务和并发之间的关系变得明显。如果没有适当的并发控制机制，如锁定，事务可能会相互干扰，可能导致数据损坏或不一致。

这个食谱将展示如何通过模拟我们从票务平台创建的销售过程来使用 FastAPI 和 SQLAlchemy 管理事务和并发。

准备工作

您需要一个 CRUD 应用程序作为食谱的基础，或者您可以使用我们本章中一直在使用的票务系统应用程序。

如何做到这一点...

事务和并发变得重要的最显著情况是在管理更新操作时，例如在我们的应用程序的销售票中。

我们将首先创建一个函数操作，将我们的票标记为已售出并给出客户的名字。然后，我们将模拟同时发生两个销售并观察结果。为此，请按照以下步骤操作。

1.  在`operations.py`模块中，创建以下函数来出售票：

    ```py
    async def sell_ticket_to_user(
        db_session: AsyncSession, ticket_id: int, user: str
    ) -> bool:
        ticket_query = (
            update(Ticket)
            .where(
                and_(
                    Ticket.id == ticket_id,
                    Ticket.sold == False,
                )
            )
            .values(user=user, sold=True)
        )
        async with db_session as session:
            result = (
               await db_session.execute(ticket_query)
            )
            await db_session.commit()
            if result.rowcount == 0:
                return False
        return True
    ```

    查询只有在票尚未售出时才会出售票；否则，函数将返回`False`。

    2. 让我们尝试向我们的数据库添加一个票并尝试模拟两个用户同时购买同一张票。让我们将所有内容都写成单元测试的形式。

    我们首先在`tests/conftest.py`文件中定义一个固定装置，将我们的票写入数据库，如下所示：

    ```py
    @pytest.fixture
    async def add_special_ticket(db_session_test):
        ticket = Ticket(
            id=1234,
            show="Special Show",
            details=TicketDetails(),
        )
        async with db_session_test.begin():
            db_session_test.add(ticket)
            await db_session_test.commit()
    ```

    3. 我们可以通过在`tests/test_operations.py`文件中执行两个并发销售并使用两个不同的数据库会话（定义另一个作为不同的固定装置）来创建一个测试，以同时进行：

    ```py
    import asyncio
    async def test_concurrent_ticket_sales(
        add_special_ticket,
        db_session_test,
        second_session_test,
    ):
        result = await asyncio.gather(
            sell_ticket_to_user(
                db_session_test, 1234, "Jake Fake"
            ),
            sell_ticket_to_user(
                second_session_test, 1234, "John Doe"
            ),
        )
        assert result in (
            [True, False],
            [False, True],
        )  # only one of the sales should be successful
        ticket = await get_ticket(db_session_test, 1234)
        # assert that the user who bought the ticket
        # correspond to the successful sale
        if result[0]:
            assert ticket.user == "Jake Fake"
        else:
            assert ticket.user == "John Doe"
    ```

    在测试函数中，我们通过使用`asyncio.gather`函数同时运行两个协程。

    我们只是假设只有一个用户可以购买票，并且它们将匹配成功的交易。一旦我们创建了测试，我们就可以使用`pytest`执行如下：

    ```py
    $ pytest tests/test_operations.py::test_concurrent_ticket_sales
    ```

测试将成功，这意味着异步会话处理了事务冲突。

练习

您刚刚创建了一个售票操作的草稿。作为一个练习，您可以通过以下方式改进草稿：

- 在数据库中添加一个用户表

- 在票上添加用户的外键引用以使其售出

- 为数据库修改创建一个 `alembic` 迁移

- 创建一个公开 `sell_ticket_to_user` 函数的 API 端点

还有更多...

数据库系统的一个基本挑战是在保持数据一致性和完整性的同时处理来自多个用户的并发事务。不同类型的交易可能对它们如何访问和修改数据以及如何处理可能与之冲突的其他交易有不同的要求。例如，管理并发的一种常见方法是使用 *锁*，这是一种防止对数据进行未经授权或不兼容操作的机制。然而，锁也可能在性能、可用性和正确性之间引入权衡。

根据业务需求，某些事务可能需要更长时间或在不同粒度级别上获取锁，例如表级别或行级别。例如，SQLite 只允许在数据库级别上锁定，而 PostgreSQL 允许锁定到行表级别。

管理并发事务的另一个关键方面是 *隔离级别* 的概念，它定义了一个事务必须从其他并发事务的影响中隔离的程度。隔离级别确保尽管有多个用户同时访问和修改数据，事务仍能保持数据一致性。

SQL 标准定义了四个隔离级别，每个级别在并发性和数据一致性之间提供不同的权衡：

1.  **读取未提交**:

    +   此级别的交易允许脏读，这意味着一个事务可以看到其他并发事务所做的未提交更改。

    +   可重复读和幻读是可能的。

    +   此隔离级别提供了最高的并发性但数据一致性最低。

1.  **读取提交**:

    +   此级别的交易只能看到其他事务提交的更改。

    +   不允许脏读。

    +   可重复读是可能的，但幻读仍然可能发生。

    +   此级别在并发性和一致性之间取得了平衡。

1.  **可重复读**:

    +   此级别的交易在整个交易过程中看到数据的一致快照。

    +   事务开始后其他事务提交的更改不可见。

    +   防止了可重复读，但可能会发生幻读。

    +   此级别在牺牲一些并发性的情况下提供了更强的一致性。

1.  **可序列化**:

    +   此级别的交易表现得好像它们是顺序执行的——也就是说，一个接一个。

    +   它们提供了数据一致性的最高级别。

    +   防止了可重复读和幻读。

    +   此级别提供强一致性，但由于锁定增加，可能会导致并发性降低。

例如，SQLite 允许隔离，而 MySQL 和 PostgreSQL 提供所有四个事务级别。

当数据库支持时，在 SQLAlchemy 中，你可以通过在初始化时指定它作为参数来为每个引擎或连接设置隔离级别。

例如，如果你想在 PostgreSQL 的引擎级别指定隔离级别，引擎将初始化如下：

```py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
eng = create_engine(
    "postgresql+psycopg2://scott:tiger@localhost/test",
    isolation_level="REPEATABLE READ",
)
Session = sessionmaker(eng)
```

所有这些关于锁和隔离级别的选择都会影响数据库系统的架构和设计，因为并非所有 SQL 数据库都支持这些功能。因此，了解锁定策略的原则和最佳实践，以及它们与事务行为和业务逻辑的关系非常重要。

你刚刚完成了一个关于将 SQL 数据库与 FastAPI 集成的全面概述。在下一章中，我们将探讨如何将 FastAPI 应用程序与 NoSQL 数据库集成。

参见

你可以在以下链接中找到有关 SQLite 和 PostgreSQL 锁定策略的更多信息：

+   *SQLite* *锁定*: [`www.sqlite.org/lockingv3.xhtml`](https://www.sqlite.org/lockingv3.xhtml)

+   *PostgreSQL* *锁定*: [`www.postgresql.org/docs/current/explicit-locking.xhtml`](https://www.postgresql.org/docs/current/explicit-locking.xhtml)

关于单个数据库的隔离级别信息可以在相应的文档页面上找到：

+   *SQLite* *隔离*: [`www.sqlite.org/isolation.xhtml`](https://www.sqlite.org/isolation.xhtml)

+   *MySQL 隔离* *级别*: [`dev.mysql.com/doc/refman/8.0/en/innodb-transaction-isolation-levels.xhtml`](https://dev.mysql.com/doc/refman/8.0/en/innodb-transaction-isolation-levels.xhtml)

+   *PostgreSQL 隔离* *级别*: [`www.postgresql.org/docs/current/transaction-iso.xhtml`](https://www.postgresql.org/docs/current/transaction-iso.xhtml)

此外，关于如何使用 SQLAlchemy 管理隔离级别的全面指南可在以下链接找到：

+   *SQLAlchemy 会话* *事务*: [`docs.sqlalchemy.org/en/20/orm/session_transaction.xhtml`](https://docs.sqlalchemy.org/en/20/orm/session_transaction.xhtml)

```py

```
