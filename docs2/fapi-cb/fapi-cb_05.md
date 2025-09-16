# 5

# 测试和调试 FastAPI 应用程序

在我们掌握 FastAPI 的旅程中，本章我们将转向软件开发的一个关键方面，确保您应用程序的可靠性、健壮性和质量：测试和调试。随着我们深入本章，您将具备创建有效测试环境、编写和执行全面测试以及高效精确地调试 FastAPI 应用程序所需的知识和工具。

理解如何正确地进行测试和调试，不仅仅是找到错误；它关乎确保您的应用程序能够承受实际使用，在高流量下不会崩溃，并提供无缝的用户体验。通过掌握这些技能，您将能够自信地增强您的应用程序，知道每一行代码都经过仔细审查，每个潜在的瓶颈都已解决。

我们将创建一个具有最小设置的 proto 应用程序来测试食谱。

到本章结束时，您不仅将深入理解适合 FastAPI 的测试框架和调试策略，还将具备将这些技术应用于构建更健壮应用程序的实际经验。这种知识是无价的，因为它直接影响到软件的质量、维护和可扩展性。

在本章中，我们将涵盖以下食谱：

+   设置测试环境

+   编写和运行单元测试

+   测试 API 端点

+   处理日志消息

+   调试技术

+   高流量应用程序的性能测试

# 技术要求

为了深入本章内容并跟随食谱进行操作，请确保您的设置包括以下基本要素：

+   **Python**：请确保您的计算机上已安装 Python 3.7 或更高版本。

+   在您的工作环境中安装`fastapi`包。

+   `pytest` 框架，这是一个广泛用于测试 Python 代码的测试框架。

本章中使用的代码托管在 GitHub 上，地址为：[`github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter05`](https://github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter05)。

建议在项目根目录内为项目设置一个虚拟环境，以高效管理依赖项并保持项目隔离。在您的虚拟环境中，您可以使用项目文件夹中 GitHub 仓库提供的`requirements.txt`一次性安装所有依赖项：

```py
$ pip install –r requirements.txt
```

虽然不是必需的，但具备基本的 HTTP 协议知识可能会有所帮助。

# 设置测试环境

本食谱将向您展示如何设置一个针对 FastAPI 应用程序高效且有效的测试环境。到食谱结束时，您将拥有编写、运行和管理测试的坚实基础。

## 准备工作

确保您有一个正在运行的应用程序。如果没有，您可以从创建一个名为 `proto_app` 的项目文件夹开始。

如果你还没有使用 GitHub 仓库上提供的 requirements.txt 文件安装包，那么请在你的环境中使用以下命令安装测试库 `pytest` 和 `httpx`：

```py
$ pip install pytest pytest-asyncio httpx
```

在项目根目录中创建一个新的文件夹 `proto_app`，其中包含一个 `main.py` 模块，该模块包含 `app` 对象实例：

```py
from fastapi import FastAPI
app = FastAPI()
@app.get("/home")
async def read_main():
    return {"message": "Hello World"}
```

使用最小化应用程序设置，我们可以通过构建项目来容纳测试。

## 如何操作…

首先，让我们开始构建我们的项目文件夹树以容纳测试。

1.  在根目录下，让我们创建一个 `pytest.ini` 文件和一个包含测试模块 `test_main.py` 的 `tests` 文件夹。项目结构应该如下所示：

    ```py
    protoapp/
    |─ protoapp/
    │  |─ main.py
    |─ tests/
    │  |─ test_main.py
    pytest.ini contains instructions for pytest. You can write in it:

    ```

    [pytest]

    pythonpath = . protoapp

    ```py

    This will add the project root and the folder `protoapp`, containing the code, to the `PYTHONPATH` when running `pytest`.
    ```

1.  现在，在 `test_main.py` 模块中，让我们为之前创建的 `/home` 端点编写一个测试：

    ```py
    import pytest
    from httpx import ASGITransport, AsyncClient
    from protoapp.main import app
    @pytest.mark.asyncio
    async def test_read_main():
        client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        )
        response = await client.get("/home")
        assert response.status_code == 200
        assert response.json() == {
            "message": "Hello World"
        }
    $ pytest –-collect-only
    ```

    你应该得到如下输出：

    ```py
    configfile: pytest.ini
    plugins: anyio-4.2.0, asyncio-0.23.5, cov-4.1.0
    asyncio: mode=Mode.STRICT
    collected 1 item
    <Dir protoapp>
      <Dir tests>
    <Module test_main.py>
    pytest.ini
    ```

1.  使用的 `pytest` 插件

1.  目录 tests，模块 `test_main.py` 和测试 `test_read_main`，它是一个协程

1.  现在，在项目根目录的命令行终端中，运行 `pytest` 命令：

    ```py
    $ pytest
    ```

你已经设置了测试我们的原型应用程序的环境。

## 参见

该食谱展示了如何在 **FastAPI** 项目中配置 `pytest` 并使用一些良好实践。请随意深入了解 **Pytest** 的官方文档，链接如下：

+   *Pytest 配置*：[`docs.pytest.org/en/stable/reference/customize.xhtml`](https://docs.pytest.org/en/stable/reference/customize.xhtml)

+   *在 Pytest 中设置 PYTHONPATH*：[`docs.pytest.org/en/7.1.x/explanation/pythonpath.xhtml`](https://docs.pytest.org/en/7.1.x/explanation/pythonpath.xhtml)

+   *Pytest 良好实践*：[`docs.pytest.org/en/7.1.x/explanation/goodpractices.xhtml`](https://docs.pytest.org/en/7.1.x/explanation/goodpractices.xhtml)

# 编写和运行单元测试

一旦我们设置了测试环境，我们就可以专注于编写和执行 FastAPI 应用程序的测试过程。单元测试对于验证应用程序各个部分在隔离状态下的行为至关重要，确保它们按预期执行。在本食谱中，你将学习如何测试应用程序的端点。

## 准备工作

我们将使用 `pytest` 来测试 FastAPI 客户端在单元测试中的表现。由于食谱将利用大多数 **Python** 标准代码中使用的公共测试 *固定装置*，在深入食谱之前，请确保熟悉测试固定装置。如果不是这样，你始终可以参考链接中的专用文档页面：[`docs.pytest.org/en/7.1.x/reference/fixtures.xhtml`](https://docs.pytest.org/en/7.1.x/reference/fixtures.xhtml)。

## 如何操作…

我们将首先为相同的 `GET /home` 端点创建一个单元测试，但与之前的食谱不同。我们将使用 FastAPI 提供的 `TestClient` 类。

让我们为它创建一个测试夹具。由于它可能被多个测试使用，让我们在`tests`文件夹下创建一个新的`conftest.py`模块。`conftest.py`是`pytest`用来存储在测试模块间共享的公共元素的默认文件。

在`conftest.py`中，让我们编写：

```py
import pytest
from fastapi.testclient import TestClient
from protoapp.main import app
@pytest.fixture(scope="function")
def test_client(db_session_test):
    client = TestClient(app)
    yield client
```

我们现在可以利用`test_client`测试夹具为我们的端点创建一个适当的单元测试。

我们将在`test_main.py`模块中编写我们的测试：

```py
def test_read_main_client(test_client):
    response = test_client.get("/home")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
```

就这样。与之前的测试相比，这个测试更紧凑，编写起来更快，归功于 FastAPI 包提供的`TestClient`类。

现在运行`pytest`：

```py
$ pytest
```

你将在终端上看到一条消息，显示已成功收集并运行了两个测试。

## 参见

你可以在官方文档中了解更多关于 FastAPI 测试客户端的信息：

+   *FastAPI 测试* *客户端*: [`fastapi.tiangolo.com/reference/testclient/`](https://fastapi.tiangolo.com/reference/testclient/)

# 测试 API 端点

集成测试验证你的应用程序的不同部分是否按预期协同工作。它们对于确保你的系统组件正确交互至关重要，尤其是在处理外部服务、数据库或其他 API 时。

在这个配方中，我们将测试两个与 SQL 数据库交互的端点。一个将项目添加到数据库，另一个将根据 ID 读取项目。

## 准备工作

要应用这个配方，你需要你的测试环境已经为`pytest`设置好了。如果不是这种情况，请检查同一章节的配方*设置* *测试环境*。

此外，这个配方将向你展示如何使用现有端点进行集成测试。你可以用它来测试你的应用程序，或者你可以按照以下方式为我们的`protoapp`构建端点。

如果你正在使用这个配方来测试你的端点，你可以直接跳到*如何进行…*部分，并将规则应用到你的端点上。

否则，如果你还没有从`requirements.txt`中安装包，请在你的环境中安装`sqlalchemy`包：

```py
$ pip install "sqlalchemy>=2.0.0"
```

现在让我们通过以下步骤设置数据库连接。

1.  在`protoapp`文件夹下，与`main.py`模块处于同一级别，让我们创建一个包含数据库设置的`database.py`模块。让我们先创建`Base`类：

    ```py
    from sqlalchemy.orm import DeclarativeBase,
    class Base(DeclarativeBase):
        pass
    ```

    我们将使用`Base`类来定义`Item`映射类。

1.  然后，数据库`Item`映射类将如下所示：

    ```py
    from sqlalchemy.orm import (
        Mapped,
        mapped_column,
    )
    class Item(Base):
        __tablename__ = "items"
        id: Mapped[int] = mapped_column(
            primary_key=True, index=True
        )
        name: Mapped[str] = mapped_column(index=True)
        color: Mapped[str]
    ```

1.  然后，我们定义将处理会话的数据库引擎：

    ```py
    DATABASE_URL = "sqlite:///./production.db"
    engine = create_engine(DATABASE_URL)
    ```

    引擎对象将用于处理会话。

1.  然后，让我们将引擎绑定到`Base`映射类：

    ```py
    Base.metadata.create_all(bind=engine)
    ```

    现在，引擎可以将数据库表映射到我们的 Python 类。

1.  最后，在`database.py`模块中，让我们创建一个`SessionLocal`类，它将生成会话，如下所示：

    ```py
    SessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )
    ```

    `SessionLocal`是一个类，它将初始化数据库会话对象。

1.  最后，在创建端点之前，让我们创建一个数据库会话。

    由于应用程序相对较小，我们可以在同一个`main.py`中完成它：

    ```py
    from protoapp.database import SessionLocal
    def get_db_session():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    ```

    我们将使用会话与数据库进行交互。

现在我们已经设置了数据库连接，在`main.py`模块中，我们可以创建一个端点来添加项目到数据库，以及一个端点来读取它。让我们这样做。

1.  让我们首先为端点创建请求体:: 

    ```py
    from pydantic import BaseModel
    class ItemSchema(BaseModel):
        name: str
        color: str
    ```

1.  用于添加项目的端点将是：

    ```py
    from fastapi import (
        Depends,
        Request,
        HTTPException,
        status
    )
    from sqlalchemy.orm import Session
    @app.post(
    "/item",
    response_model=int,
    status_code=status.HTTP_201_CREATED
    )
    def add_item(
        item: ItemSchema,
        db_session: Session = Depends(get_db_session),
    ):
        db_item = Item(name=item.name, color=item.color)
        db_session.add(db_item)
        db_session.commit()
        db_session.refresh(db_item)
        return db_item.id
    ```

    当项目存储在数据库中时，端点将返回受影响的项 ID。

1.  现在我们有了添加项目的端点，我们可以通过创建基于 ID 检索项目的端点来继续：

    ```py
    @app.get("/item/{item_id}", response_model=ItemSchema)
    def get_item(
        item_id: int,
        db_session: Session = Depends(get_db_session),
    ):
        item_db = (
            db_session.query(Item)
            .filter(Item.id == item_id)
            .first()
        )
        if item_db is None:
            raise HTTPException(
                status_code=404, detail="Item not found"
            )
        return item_db
    ```

    如果 ID 不对应于数据库中的任何项目，端点将返回 404 状态码。

我们刚刚创建了允许我们创建集成测试的端点。

## 如何做到这一点…

一旦我们有了端点，在`tests`文件夹中，我们应该适配我们的`test_client`固定装置以使用与生产中不同的会话。

我们将把整个过程分解为两个主要动作：

+   将测试客户端适配以适应测试数据库会话

+   创建测试以模拟端点之间的交互

让我们按照以下步骤进行。

1.  首先，在之前在配方“编写和运行单元测试”中创建的`conftest.py`文件中，让我们定义一个新的引擎，该引擎将使用内存中的 SQLite 数据库并将其绑定到`Base`类映射：

    ```py
    from sqlalchemy.pool import StaticPool
    from sqlalchemy import create_engine
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)  # Bind the engine
    ```

1.  让我们为测试会话创建一个专门的会话创建器，如下所示：

    ```py
    from sqlalchemy.orm import sessionmaker
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )
    ```

1.  类似于`main.py`模块中的`get_db_session`函数，我们可以在`conftest.py`模块中创建一个固定装置来检索测试会话：

    ```py
    @pytest.fixture
    def test_db_session():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
    ```

1.  然后，我们应该修改`test_client`以使用这个会话而不是生产会话。我们可以通过覆盖返回会话的依赖项来实现，FastAPI 允许你通过调用测试客户端的方法`dependency_overrides`来轻松实现，如下所示：

    ```py
    from protoapp.main import app, get_db_session
    @pytest.fixture(scope="function")
    def test_client(test_db_session):
        client = TestClient(app)
        app.dependency_overrides[get_db_session] = (
            lambda: test_db_session
    )
        return client
    ```

    每次测试客户端需要调用会话时，固定装置将用使用内存数据库的测试会话替换它。

1.  然后，为了验证我们的应用程序与数据库的交互，我们创建了一个测试：

    +   通过`POST /item`端点将项目创建到数据库中

    +   通过使用测试会话验证项目是否正确创建在测试数据库中

    +   通过`GET /item`端点检索项目

    你可以将测试放入`test_main.py`，以下是它的样子：

    ```py
    def test_client_can_add_read_the_item_from_database(
        test_client, test_db_session
    ):
        response = test_client.get("/item/1")
        assert response.status_code == 404
        response = test_client.post(
            "/item", json={"name": "ball", "color": "red"}
        )
        assert response.status_code == 201
        # Verify the user was added to the database
        item_id = response.json()
        item = (
            test_db_session.query(Item)
            .filter(Item.id == item_id)
            .first()
        )
        assert item is not None
        response = test_client.get(f"item/{item_id}")
        assert response.status_code == 200
        assert response.json() == {
            "name": "ball",
            "color": "red",
        }
    ```

你刚刚为我们的原型应用创建了一个集成测试，请随意丰富你的应用并相应地创建更多测试。

## 参见

我们已经为测试设置了一个内存中的 SQLite 数据库。由于每个会话都与线程绑定，因此需要相应地配置引擎以避免刷新数据。

配置策略已在以下文档页面找到：

+   *SQLite 内存数据库* *配置*：[`docs.sqlalchemy.org/en/14/dialects/sqlite.xhtml#using-a-memory-database-in-multiple-threads`](https://docs.sqlalchemy.org/en/14/dialects/sqlite.xhtml#using-a-memory-database-in-multiple-threads)

# 运行测试技术

通过系统地覆盖所有端点和场景，你确保了你的 API 在各种条件下表现正确，从而为你的应用程序的功能提供信心。彻底测试 API 端点是构建可靠和健壮应用程序的基本要求。

这个配方将解释如何单独或按组运行测试以及如何检查代码的测试覆盖率。

## 准备工作

要运行这个配方，确保你已经放置了一些测试，或者你已经遵循了本章的所有前一个配方。此外，确保你在`pytest.ini`中定义了测试的 PYTHONPATH。查看配方*设置* *测试环境*了解如何操作。

## 如何做到这一点...

我们将首先查看如何通过默认分组（单独或按模块）运行测试，然后我们将介绍一种基于标记自定义测试分组的技术。

如你所知，所有单元测试都可以通过终端使用以下命令运行：

```py
$ pytest
```

然而，可以根据测试调用语法单独运行测试：

```py
$ pytest <test_module>.py::<test_name>
```

例如，如果我们想运行测试函数`test_read_main_client`，运行：

```py
$ pytest tests/test_main.py::test_read_main
```

有时测试名称变得过于复杂难以记住，或者我们有特定的需求只想运行一组特定的测试。这就是测试标记发挥作用的地方。

让我们假设我们只想运行集成测试。在我们的应用程序中，唯一的集成测试由函数`tests_client_can_add_read_the_item_from_database`表示。

我们可以通过添加特定的装饰器到函数中来应用标记：

```py
@pytest.mark.integration
def test_client_can_add_read_the_item_from_database(
    test_client, test_db_session
):
    # test content
```

然后，在`pytest.ini`配置文件中，在专用部分添加`integration`标记以注册标记：

```py
[pytest]
pythonpath = protoapp .
markers =
    integration: marks tests as integration
```

现在，你可以通过以下方式运行目标测试：

```py
$ pytest –m integration -vv
```

在输出信息中，你会看到只有标记的测试被选中并运行。你可以使用标记根据逻辑标准对应用程序的测试进行分组，例如，一个组用于**创建、读取、更新和删除**（**CRUD**）操作，一个组用于安全操作，等等。

## 检查测试覆盖率

为了确保你的端点以及代码的文本行都经过了测试，了解测试覆盖率可能很有用。

测试覆盖率是软件测试中用来衡量在特定测试套件运行时程序源代码执行程度的指标。

要与`pytest`一起使用，如果你没有使用`requirements.txt`安装包，你需要安装`pytest-cov`包：

```py
$ pip install pytest-cov
```

它的工作方式非常直接。你需要将源代码根目录（在我们的例子中是`protoapp`目录）传递给`pytest`的`--cov`参数和测试根目录（在我们的例子中是测试），如下所示：

```py
$ pytest –-cov protoapp tests
```

您将在输出中看到一个表格，列出每个模块的覆盖率百分比：

```py
Name                   Stmts   Miss  Cover
------------------------------------------
protoapp\database.py      16      0   100%
protoapp\main.py          37      4    89%
protoapp\schemas.py        8      8     0%
------------------------------------------
TOTAL                     61     12    80%
```

此外，还创建了一个名为`.coverage`的文件。这是一个包含测试覆盖率数据的二进制文件，可以使用其他工具从中生成报告。

例如，如果你运行：

```py
$ coverage html
```

它将创建一个名为`htmlcov`的文件夹，其中包含一个`index.xhtml`页面，包含覆盖率页面，您可以通过用浏览器打开它来可视化它。

## 参见

您可以在官方文档链接中了解更多有关使用 Pytest 调用单元测试的选项以及如何评估测试覆盖率。

+   *使用 Pytest 调用单元测试*：[`docs.pytest.org/en/7.1.x/how-to/usage.xhtml`](https://docs.pytest.org/en/7.1.x/how-to/usage.xhtml)

+   *Pytest* *覆盖率*：[`pytest-cov.readthedocs.io/en/latest/`](https://pytest-cov.readthedocs.io/en/latest/)

# 处理日志消息

在应用开发中有效地管理日志不仅有助于及时识别错误，还能提供有关用户交互、系统性能和潜在安全威胁的宝贵见解。它作为审计、合规和优化资源利用的关键工具，最终增强了软件的可靠性和可扩展性。

这个配方将展示如何高效地将日志记录系统集成到我们的 FastAPI 应用程序中，以监控 API 的调用。

## 准备工作

我们将使用 Python 日志生态系统的一些基本功能。

虽然这个例子很简单，但您可以参考官方文档，了解相关术语，如**日志记录器**、**处理程序**、**格式化器**和**日志级别**。请点击以下链接：

[`docs.python.org/3/howto/logging-cookbook.xhtml`](https://docs.python.org/3/howto/logging-cookbook.xhtml)。

要将日志记录集成到 FastAPI 中，请确保您有一个运行中的应用程序或使用本章中一直开发的`protoapp`。

## 如何操作...

我们希望创建一个日志记录器，将客户端的调用信息打印到终端并记录到文件中。

让我们在`protoapp`文件夹下的`logging.py`模块中创建日志记录器，按照以下步骤进行。

1.  让我们首先定义一个具有`INFO`级别值的日志记录器：

    ```py
    import logging
    client_logger = logging.getLogger("client.logger")
    logger.setLevel(logging.INFO)
    ```

    由于我们希望将消息流式传输到控制台并存储到文件中，我们需要定义两个单独的处理程序。

1.  现在，让我们定义一个处理程序，将日志消息打印到控制台。我们将使用`logging`内置包中的`StreamHandler`对象：

    ```py
    console_handler = logging.StreamHandler()
    ```

    这会将消息流式传输到控制台。

1.  让我们创建一个彩色格式化器并将其添加到我们刚刚创建的处理程序中：

    ```py
    from uvicorn.logging import ColourizedFormatter
    console_formatter = ColourizedFormatter(
        "%(levelprefix)s CLIENT CALL - %(message)s",
        use_colors=True,
    )
    console_handler.setFormatter(console_formatter)
    ```

    格式化器将以 FastAPI 使用的默认日志记录器 uvicorn 日志记录器的格式格式化日志消息。

1.  然后让我们将处理程序添加到日志记录器中：

    ```py
    client_logger.addHandler(console_handler)
    ```

    我们刚刚设置了日志记录器，以便将消息打印到控制台。

1.  让我们重复之前的*步骤 1 到 4*来创建一个将消息存储到文件并添加到我们的`client_logger`的处理程序：

    ```py
    from logging.handlers import TimedRotatingFileHandler
    file_handler = TimedRotatingFileHandler("app.log")
    file_formatter = logging.Formatter(
        "time %(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    client_logger.addHandler(file_handler)
    ```

    现在我们已经设置了日志记录器。每条消息都将被输出到控制台并存储在`app.log`文件中。

1.  一旦我们构建了我们的`client_logger`，我们就在代码中使用它来获取客户端调用的信息。

    你可以通过在`main.py`模块中添加日志记录器和专用中间件来实现这一点：

    ```py
    from protoapp.logging import client_logger
    # ... module content
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        client_logger.info(
            f"method: {request.method}, "
            f"call: {request.url.path}, "
            f"ip: {request.client.host}"
        )
        response = await call_next(request)
        return response
    ```

1.  现在启动服务器：

    ```py
    $ uvicorn protoapp.main:app
    ```

尝试调用我们定义的任何端点，你将在终端上看到我们为请求和响应定义的日志。此外，你将在由应用程序自动创建的新`app.log`文件中找到来自我们的`logger_client`的消息。

## 更多内容

定义合适的日志策略需要单独的食谱，但这超出了本书的范围。然而，当将日志记录器集成到应用程序中时，遵循一些指南是很重要的：

+   **适当使用标准日志级别**。一个经典的级别系统由 4 个级别组成：**INFO**、**WARNING**、**ERROR**、**CRITICAL**。根据应用程序的需要，你可能需要更多或更少的级别。无论如何，将每条消息放置在适当的级别。

+   **保持日志格式一致**。在整个应用程序中保持一致的日志格式。这包括一致的日期时间格式、严重级别，以及清楚地描述事件。一致的格式有助于解析日志和自动化日志分析。

+   **包含上下文信息**。在你的日志中包含相关的上下文信息（例如，用户 ID，事务 ID），以帮助追踪和调试应用程序工作流程中的问题。

+   **避免敏感信息**。永远不要记录敏感信息，如密码、API 密钥或**个人可识别信息**（**PII**）。如果必要，可以对这些细节进行掩码或哈希处理。

+   **高效日志记录**。注意日志记录对性能的影响。过度记录可能会减慢应用程序的速度，并导致日志噪声，使得找到有用的信息变得困难。在信息需求与性能影响之间取得平衡。

当然，这并不是一个详尽的列表。

## 参考内容

Python 发行版自带一个强大的内置日志记录包，您可以查看官方文档：

+   *Python* *日志记录*: [`docs.python.org/3/library/logging.xhtml`](https://docs.python.org/3/library/logging.xhtml)

此外，在**Sentry**博客上了解更多关于日志记录最佳实践和指南：

+   *日志记录* *指南*: [`blog.sentry.io/logging-in-python-a-developers-guide/`](https://blog.sentry.io/logging-in-python-a-developers-guide/)

**Sentry**是一个用于监控 Python 代码的工具。

# 调试技术

掌握调试应用程序开发对于高效识别和修复问题至关重要。这个食谱深入探讨了调试器的实际应用，利用工具和策略来定位 FastAPI 代码中的问题。

## 准备工作

要应用这个食谱，你只需要有一个正在运行的应用程序。我们可以继续使用我们的`protoapp`进行工作。

## 如何操作...

Python 发行版已经自带了一个默认的调试器，称为`pdb`。如果你使用的是**集成开发环境**（**IDE**），它通常包含一个编辑器分布调试器。无论你使用什么来调试你的代码，你必须熟悉断点的概念。

**断点**是代码中的一个点，它暂停执行并显示代码变量的状态和调用。它可以附加一个条件，如果满足条件，则激活它，否则跳过。

无论你使用的是 Python 发行版调试器`pdb`还是你的 IDE 提供的调试器，定义一个启动脚本来启动服务器可能很有用。

在项目根目录下创建一个名为`run_server.py`的文件，包含以下代码：

```py
import uvicorn
from protoapp.main import app
if __name__ == "__main__":
    uvicorn.run(app)
```

脚本导入`uvicorn`包和我们的应用`app`，并在`uvicorn`服务器上运行应用。这相当于启动命令：

```py
$ uvicorn protoapp.main:app
```

有一个脚本可以让我们有更大的灵活性来运行服务器，并在需要时将其包含到一个更广泛的 Python 程序中。

要检查是否正确设置，像运行正常的 Python 脚本一样运行脚本：

```py
$ python run_server.py
```

使用你喜欢的浏览器访问`localhost:8000/docs`并检查文档是否已正确生成。

## 使用 PDB 进行调试

PDB 调试器默认包含在任何 Python 发行版中。从 Python 3.7 以上的版本开始，你可以通过在想要暂停的代码行添加函数调用`breakpoint()`来定义一个断点，然后像平常一样运行代码。

如果你运行代码，当它到达断点行时，执行将自动切换到调试模式，你可以从终端运行调试命令。你可以通过输入 help 来找到你可以运行的命令列表：

```py
(Pdb) help
```

你可以运行列出变量、显示堆栈跟踪以检查最近的帧或定义带有条件的新断点等命令。

在这里你可以找到所有可用命令的列表：[`docs.python.org/3/library/pdb.xhtml#debugger-commands`](https://docs.python.org/3/library/pdb.xhtml#debugger-commands)。

你也可以将`pdb`作为模块调用。在这种情况下，如果程序异常退出，`pdb`将自动进入**事后**调试：

```py
$ python –m pdb run_server.py
```

这意味着`pdb`将自动重启程序，同时保留`pdb`模块的执行状态，包括断点。

当通过调用`pytest`作为模块进行测试调试时，也可以这样做：

```py
$ python –m pdb -m pytest tests
```

另一种调试策略是利用`uvicorn`服务器的重新加载功能。为此，你需要修改`run_server.py`文件，如下所示：

```py
import uvicorn
if __name__ == "__main__":
    uvicorn.run("protoapp.main:app", reload=True)
```

然后，不使用`pdb`模块运行服务器：

```py
$ python run_server.py
```

这样，你就可以在重新加载服务器功能下轻松地使用断点。 

在撰写本文时，`unvicorn`。

## 使用 VS Code 进行调试

VS Code Python 扩展自带其分布式的调试器，称为 *debugpy*。运行环境的配置可以在 `.vscode/launch.json` 文件中管理。调试我们服务器的配置文件示例如下：

```py
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger FastAPI server",
            "type": "debugpy",
            "request": "launch",
            "program": "run_server.py",
            "console": "integratedTerminal",
        },
}
```

配置指定了要使用的调试器类型（`debugpy`）、要运行的程序（我们的启动脚本 `run_server.py`），并且可以在 GUI 选项中找到。

`request` 字段指定了运行调试器的模式，可以是 *launch*（用于运行程序），或 *attach*（用于连接到已运行的实例），这对于调试运行在远程实例上的程序特别有用。

调试远程实例超出了本食谱的范围，但您可以在官方文档中找到详细说明：[`code.visualstudio.com/docs/python/debugging#_debugging-by-attaching-over-a-network-connection`](https://code.visualstudio.com/docs/python/debugging#_debugging-by-attaching-over-a-network-connection)

可以通过利用 *Test Explorer* 扩展来设置调试配置以运行单元测试。该扩展将在 `launch.json` 中查找包含 `"type": "python"` 和 `"purpose": ["debug-test"]`（或 `"request": "test"`）的配置。调试测试的配置示例如下：

```py
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug test",
            "type": "python",
            "request": "launch",
            "console": "integratedTerminal",
            "justMyCode": false,
            "stopOnEntry": true,
            "envFile": "${workspaceFolder}/.env.test",
            "purpose": ["debug-test"]
        }
    ]
}
```

您可以在 VS Code 市场扩展页面找到详细的解释：[`marketplace.visualstudio.com/items?itemName=LittleFoxTeam.vscode-python-test-adapter`](https://marketplace.visualstudio.com/items?itemName=LittleFoxTeam.vscode-python-test-adapter)。

## 使用 PyCharm 进行调试

PyCharm 通过运行/调试配置管理代码执行，这些配置是一组命名的启动属性集，详细说明了执行参数和环境。这些配置允许使用不同的设置运行脚本，例如使用不同的 Python 解释器、环境变量和输入源。

运行/调试配置有两种类型：

+   临时：自动为每次运行或调试会话生成。

+   永久：手动从模板创建或由临时配置转换而来，并保存在您的项目中，直到删除。

PyCharm 默认使用现有的永久配置或为每个会话创建一个临时配置。临时配置限制为五个，最旧的配置将被删除以为新配置腾出空间。此限制可以在设置中调整（**设置** | **高级设置** | **运行/调试** | **临时配置限制**）。图标区分永久（不透明）和临时（半透明）配置。

每个配置都可以存储在一个单独的 xml 文件中，该文件由 GUI 自动检测。

我们 FastAPI `protoapp` 的配置示例如下：

```py
<component name="ProjectRunConfigurationManager">
  <configuration default="false" name="run_server"
    type="PythonConfigurationType" factoryName="Python"
    nameIsGenerated="true">
    <module name="protoapp" />
    <option name="INTERPRETER_OPTIONS" value="" />
    <option name="PARENT_ENVS" value="true" />
    <envs>
      <env name="PYTHONUNBUFFERED" value="1" />
    </envs>
    <option name="WORKING_DIRECTORY"
      value="$PROJECT_DIR$" />
    <option name="IS_MODULE_SDK" value="true" />
    <option name="ADD_CONTENT_ROOTS" value="true" />
    <option name="ADD_SOURCE_ROOTS" value="true" />
    <option name="SCRIPT_NAME"
      value="$PROJECT_DIR$/run_server.py" />
    <option name="SHOW_COMMAND_LINE" value="false" />
    <option name="MODULE_MODE" value="false" />
    <option name="REDIRECT_INPUT" value="false" />
    <option name="INPUT_FILE" value="" />
    <method v="2" />
  </configuration>
</component>
```

您可以在专门的 PyCharm 文档页面找到如何设置的详细指南：[`www.jetbrains.com/help/pycharm/run-debug-configuration.xhtml`](https://www.jetbrains.com/help/pycharm/run-debug-configuration.xhtml)。

## 参见

你可以自由地深入研究我们刚刚在链接中解释的每个调试解决方案和概念：

+   *Python 发行版调试器*：[`docs.python.org/3/library/pdb.xhtml`](https://docs.python.org/3/library/pdb.xhtml)

+   *断点*：[`docs.python.org/3/library/functions.xhtml#breakpoint`](https://docs.python.org/3/library/functions.xhtml#breakpoint)

+   *Uvicorn 设置*：[`www.uvicorn.org/settings/`](https://www.uvicorn.org/settings/)

+   *使用 VS Code 进行调试*：[`code.visualstudio.com/docs/python/debugging`](https://code.visualstudio.com/docs/python/debugging)

+   *Debugy 调试器*：[`github.com/microsoft/debugpy/`](https://github.com/microsoft/debugpy/)

+   *使用 PyCharm 进行调试*：[`www.jetbrains.com/help/pycharm/debugging-your-first-python-application.xhtml`](https://www.jetbrains.com/help/pycharm/debugging-your-first-python-application.xhtml)

# 高流量应用程序的性能测试

性能测试对于确保你的应用程序能够处理现实世界的使用场景至关重要，尤其是在高负载下。通过系统地实施和运行性能测试，分析结果，并根据发现进行优化，你可以显著提高应用程序的响应性、稳定性和可扩展性。

该食谱将展示如何使用**Locust**框架基准测试你的应用程序的基础知识。

## 准备工作

要运行性能测试，你需要一个运行中的应用程序，我们将使用我们的`protoapp`和一个测试框架。我们将使用基于 Python 语法的**Locust**框架，它是一个测试框架。

你可以在官方文档中找到详细的解释：[`docs.locust.io/en/stable/`](https://docs.locust.io/en/stable/)。

在开始之前，请确保你已经通过运行以下命令在你的虚拟环境中安装了它：

```py
$ pip install locust
```

现在我们已经准备好设置配置文件并运行 locust 实例。

## 如何做到这一点...

当应用程序正在运行且已安装`locust`包时，我们将通过指定我们的配置来运行性能测试。

在你的项目根目录中创建一个`locustfile.py`文件。此文件将定义与测试中的应用程序交互的用户的行为。

`locustfile.py`的最小示例可以是：

```py
from locust import HttpUser, task
class ProtoappUser(HttpUser):
    host = "http://localhost:8000"
    @task
    def hello_world(self):
        self.client.get("/home")
```

配置定义了一个客户端类，其中包含服务地址和我们要测试的端点。

使用以下命令启动你的 FastAPI 服务器：

```py
$ uvicorn protoapp.main:app
```

然后在另一个终端窗口中运行 locust：

```py
$ locust
```

打开你的浏览器并导航到`http://localhost:8089`以访问应用程序的 Web 界面。

Web 界面设计直观，使得以下操作变得简单：

+   **设置并发用户**：指定在高峰使用期间同时访问服务的最大用户数。

+   **配置爬坡速率**：确定每秒添加的新用户数量以模拟增加的流量。

配置好这些参数后，点击`locustfile.py`中定义的`/home`端点。

或者，你可以使用命令行模拟流量。以下是方法：

```py
$ locust --headless --users 10 --spawn-rate 1
```

此命令以无头模式运行 Locust 以模拟：

+   10 个用户同时访问您的应用程序。

+   每秒产生 1 个用户。

在部署之前，您可以通过将其包含在 **持续集成/持续交付**（**CI/CD**）管道中，或者甚至将其纳入更大的测试流程中，来进一步扩展您的测试体验。

深入文档以测试您应用程序流量的各个方面。

您拥有所有调试和全面测试您应用程序的工具。

在下一章中，我们将构建一个与 SQL 数据库交互的综合 RESTful 应用程序。

## 参见

您可以在官方文档页面上找到更多关于 Locust 的信息：

+   *Locust 快速入门*：[`docs.locust.io/en/stable/quickstart.xhtml`](https://docs.locust.io/en/stable/quickstart.xhtml)

+   *编写 Locust 文件*：[`docs.locust.io/en/stable/writing-a-locustfile.xhtml`](https://docs.locust.io/en/stable/writing-a-locustfile.xhtml)

+   *从命令行运行 Locust*：[`docs.locust.io/en/stable/running-without-web-ui.xhtml`](https://docs.locust.io/en/stable/running-without-web-ui.xhtml)
