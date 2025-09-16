# 7

# 使用 FastAPI 构建后端

在前面的章节中，你学习了认证和授权的基本机制，现在你准备好实现它并保护一个用 FastAPI 构建的 Web API 了。在本章中，你将充分利用这些知识，创建一个简单但功能齐全的 REST API，展示二手车及其图片。

在本章中，你将了解以下操作，这些操作可以被视为在用 FastAPI 创建 REST API 时的一个松散耦合的蓝图。

本章将涵盖以下主题：

+   通过使用 Python Motor 驱动程序将 FastAPI 实例连接到 MongoDB Atlas

+   根据规范定义 Pydantic 模型，并初始化 FastAPI 应用程序

+   创建 API 路由并实现 CRUD 操作

+   使用 JWT 保护 API

+   部署到 Render

# 技术要求

本章的要求与之前定义的要求相似。你将使用以下内容进行工作：

+   Python 3.11.7 或更高版本

+   Visual Studio Code

+   MongoDB Atlas 的介绍

之后，你需要在图像托管服务**Cloudinary**（免费）和 API 托管平台**Render**（也是一个免费层账户）上创建账户。再次，你将使用 HTTPie 手动测试你将要实现的 API 端点。

让我们先了解将要开发的应用程序以及后端需要什么。

# 介绍应用程序

在有一个具体问题需要解决的情况下开始使用框架要容易得多，即使要求有些模糊。当前的任务相当简单：你需要为一家虚构的汽车销售公司创建一个用于存储和检索二手车数据的 REST API 后端。

描述车辆的这种数据结构相当简单，但一旦你深入研究细节，如发动机型号、内饰颜色、悬挂类型等，它可能会变得更加复杂。

在你的第一个简单的**创建**、**读取**、**更新**、**删除**（**CRUD**）应用程序中，你将保持资源数据有限。一辆车将由以下字段描述：

+   `品牌`：汽车品牌（福特、雷诺等），用一个*字符串*表示

+   型号或型号：例如，Fiesta 或 Clio，用一个*字符串*表示

+   `年份`：生产年份，一个限制在合理范围内的*整数*（1970–2024）

+   `Cm3`：发动机排量，与发动机功率成正比，一个范围*整数*

+   `kW`：发动机功率，以 kW 为单位的*整数*

+   `Km`：汽车行驶的公里数，一个位于数十万范围内的*整数*

+   `价格`：欧元价格

+   一个图片 URL：这是可选的

每个汽车销售网站的一个基本功能是存在图片，因此您将使用领先的图像托管和处理服务之一——**Cloudinary**——实现一个图像上传管道。稍后，您将为员工提供更多帮助，为每个汽车型号生成吸引人的文案，这将使 API 更加丰富，同时展示 FastAPI 的简洁性。

# 创建 Atlas 实例和集合

登录您的 Atlas 账户，并在名为 `cars` 的集合中创建一个名为 `carBackend` 的新数据库。您可以参考*第二章*，*使用 MongoDB 设置数据库*。在创建数据库和集合后，注意 MongoDB 连接字符串并将其保存在文本文件中，以备后用，当您创建秘密环境密钥时。

## 设置 Python 环境

在 Atlas 上创建 MongoDB 数据库并连接后，现在是时候为您设置一个新的 Python 虚拟环境并安装需求了：

1.  首先，创建一个名为 `requirements.txt` 的纯文本文件，并在其中插入以下行：

    ```py
    fastapi==0.111.0
    motor==3.4.0
    uvicorn==0.29.0
    pydantic-settings==2.2.1
    ```

1.  如果您想要能够精确地重现本书中使用的代码，则包版本控制很重要，并且您始终可以参考书中存储库中的 `requirements.txt` 文件。运行之前定义的需求文件中的 `pip` 安装命令：

    ```py
    pip install -r requirements.txt
    ```

您的环境已准备就绪。现在，凭借从*第三章*，*Python 类型提示和 Pydantic*中学到的 Python 类型提示和 Pydantic 知识，您将构建这个相对简单的汽车数据结构。

## 定义 Pydantic 模型

让我们从第一个 Pydantic 模型开始，针对单一车辆。在这里，需要提前解决的一个主要问题是如何在 Pydantic 中序列化和定义 MongoDB 的 `ObjectID` 键。虽然表示 `ObjectID` 的方法有很多种，但最简单且目前由 MongoDB 推荐的方法是将 `ObjectID` 转换为字符串。您可以参考以下文档以获取更多详细信息：[`www.mongodb.com/developer/languages/python/python-quickstart-fastapi/`](https://www.mongodb.com/developer/languages/python/python-quickstart-fastapi/)。

MongoDB 使用 `_id` 字段作为标识符。在 Python 中，由于以下划线开头的属性具有特殊含义，您不能使用原始的字段名进行模型填充。

Pydantic 别名提供了一个简单而优雅的解决方案；您可以命名字段为 `id`，但也可以将其别名为 `_id`，并将 `populate_by_name` 标志设置为 `True`，如*第三章*，*Python 类型提示和 Pydantic*中所示。

最后，您需要将 `ObjectID` 转换为字符串。为此，您将使用简单的 Python 注解和 Pydantic 的 `BeforeValidator` 模块。

1.  创建一个名为 `Chapter7` 的文件夹，并在其中创建一个 `models.py` 文件，然后开始编写导入和 `ObjectID` 类型：

    ```py
    #models.py
    from typing import Optional, Annotated, List
    from pydantic import BaseModel, ConfigDict, Field, BeforeValidator, field_validator
    PyObjectId = Annotated[str, BeforeValidator(str)]
    ```

1.  在导入之后，创建一个新的类型 `PyObjectId`，它将用于将 MongoDB 的原始 `ObjectID` 作为字符串表示，继续填充模型：

    ```py
    class CarModel(BaseModel):
        id: Optional[PyObjectId] = Field(
           alias="_id", default=None)
        brand: str = Field(...)
        make: str = Field(...)
        year: int = Field(..., gt=1970, lt=2025)
        cm3: int = Field(..., gt=0, lt=5000)
        km: int = Field(..., gt=0, lt=500000)
        price: int = Field(..., gt=0, lt=100000)
    ```

    如果你阅读了 Pydantic 的章节，这些字段应该非常熟悉；你只是在声明汽车字段，将所有字段标记为必需，并对数值量（`cm3`、`km`、`price` 和 `year`）设置一些合理的限制。

    请记住，汽车品牌数量有限，因此创建一个 **枚举** 类型的品牌名称是有可能的，也是建议的，但在这个例子中，你将保持简单。

1.  添加两个方便的字段验证器作为修饰符。你希望返回每个汽车品牌和型号的标题：

    ```py
    @field_validator("brand")
    @classmethod
    def check_brand_case(cls, v: str) -> str:
        return v.title()
    @field_validator("make")
    @classmethod
    def check_make_case(cls, v: str) -> str:
        return v.title()
    ```

1.  最后，为了完成模型，添加一个配置字典，允许它通过名称填充并允许任意类型：

    ```py
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "brand": "Ford",
                "make": "Fiesta",
                "year": 2019,
                "cm3": 1500,
                "km": 120000,
                "price": 10000,
            }
        },
    )
    ```

1.  你现在可以通过在文件末尾添加以下（临时）行来测试模型，在类定义之外运行它：

    ```py
    test_car = CarModel(
        brand="ford", make="fiesta", year=2019, cm3=1500, km=120000, price=10000
    )
    print(test_car.model_dump())
    ```

1.  运行 `models.py` 文件：

    ```py
    python models.py
    {'id': None, 'brand': 'Ford', 'make': 'Fiesta', 'year': 2019, 'cm3': 1500, 'km': 120000, 'price': 10000}.
    ```

    现在，是时候定义其他模型以更新单个实例和获取汽车列表了。`update` 模型需要只允许更改特定字段。理论上，只有 `price` 应该是可更新的，因为汽车本身是相当不可变的对象，但这个系统将允许一些模糊性和需要通过 API 手动纠正的错误数据的情况。

1.  在从 `models.py` 中删除或注释掉测试行之后，继续创建 `UpdateCarModel` 模型：

    ```py
    class UpdateCarModel(BaseModel):
        brand: Optional[str] = Field(...)
        make: Optional[str] = Field(...)
        year: Optional[int] = Field(..., gt=1970, lt=2025)
        cm3: Optional[int] = Field(..., gt=0, lt=5000)
        km: Optional[int] = Field(..., gt=0, lt=500 * 1000)
        price: Optional[int] = Field(..., gt=0, lt=100 * 1000)
    ```

    类的剩余部分与 `CarModel` 类相同，为了简洁起见将省略。

1.  最后，`ListCarsModel` 类将非常简单，因为它只需要处理 `CarModel` 类的列表：

    ```py
    class CarCollection(BaseModel):
        cars: List[CarModel]
    ```

    现在模型已经就绪，你可以进行简单的测试，看看 `ListCarsModel` 是如何工作的。

1.  创建一个名为 `test_models.py` 的新测试文件，按照顺序添加以下行以创建两个不同的汽车模型和一个列表，然后打印模型转储：

    ```py
    from models import CarCollection, CarModel
    test_car_1 = CarModel(
        brand="ford", make="fiesta", year=2019, cm3=1500, km=120000, price=10000
    )
    test_car_2 = CarModel(
        brand="fiat", make="stilo", year=2003, cm3=1600, km=320000, price=3000
    )
    car_list = CarCollection(cars=[test_car_1, test_car_2])
    print(car_list.model_dump())
    ```

    如果你用 Python 运行 `test_models.py` 文件，输出应该是以下内容：

    ```py
    {'cars': [{'id': None, 'brand': 'Ford', 'make': 'Fiesta', 'year': 2019, 'cm3': 1500, 'km': 120000, 'price': 10000}, {'id': None, 'brand': 'Fiat', 'make': 'Stilo', 'year': 2003, 'cm3': 1600, 'km': 320000, 'price': 3000}]}
    ```

模型，至少是它们的初始迭代（MongoDB 在迭代数据建模方面非常出色），现在已经完成，所以你可以在下一节开始构建你的 FastAPI 应用程序结构。

# 构建 FastAPI 应用程序

`Motor` 驱动器。最初，你将只创建一个通用的最小 FastAPI 应用程序，并逐步添加功能。

你将开始将秘密环境数据——在你的情况下，只是 MongoDB Atlas 数据库 URL——存储到 `.env` 文件中。这些值应该始终保持在存储库之外。你希望能够连接到你的 MongoDB 数据库并验证连接是否成功。

## 创建 .env 文件以保存秘密

对于应保持秘密并排除在版本控制系统之外的价值，您将使用一个环境文件（`.env`）。执行以下步骤以设置环境变量并将它们排除在版本控制之外：

1.  首先，创建一个 `.env` 文件，并在其中以以下格式放入您的秘密连接字符串，无需引号：

    ```py
    DB_URL=mongodb+srv://<USERNAME>:<PASSWORD>@cluster0.fkm24.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
    DB_NAME=carBackend
    ```

    此 `.env` 文件将后来托管其他外部服务可能需要的秘密文件，您可能在 API 开发中使用。

1.  现在，创建一个 `.gitignore` 文件，并填充基本条目：Git 不应跟踪的目录和文件。打开一个文件，命名为 `.gitignore`，并插入以下内容：

    ```py
    __pycache__/
    .env
    venv/
    ```

    网上有很多与 Python 相关的 `.gitignore` 文件示例，所以请随意查找，但这将足以满足我们的需求。

1.  现在，您可以使用以下 Git 命令将工作目录置于版本控制之下：

    ```py
    git init
    git add .
    git commit -m "initial commit"
    ```

### 使用 pydantic_settings 创建 Pydantic 配置

在接下来的步骤中，您将使用之前创建的环境变量，并将它们提供给 `pydantic_settings`——这是用于管理应用程序设置的 Pydantic 类，在*第三章**，Python 类型提示和 Pydantic* 中有介绍。这个类将在需要环境变量的任何地方轻松调用。

在完成这项准备工作后，创建一个名为 `config.py` 的文件，该文件将利用 `pydantic_settings` 包来管理您的设置，如下所示：

1.  创建一个配置文件，命名为 `config.py`，您将使用它来读取应用程序的设置。当您引入一些自动化测试或为生产环境、不同的数据库等设置不同的设置时，您将能够轻松地更改它们。将以下代码粘贴到 `config.py` 文件中：

    ```py
    from typing import Optional
    from pydantic_settings import BaseSettings, SettingsConfigDict
    class BaseConfig(BaseSettings):
        DB_URL: Optional[str]
        DB_NAME: Optional[str]
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    ```

    现在，您将使用这些配置设置来获取环境数据以连接到 MongoDB Atlas 实例。

1.  最后，您可以通过创建一个名为 `app.py` 的新 Python 文件来开始构建实际的应用程序文件。在这个文件中，首先实例化一个 FastAPI 实例，并创建一个带有简单消息的根路由：

    ```py
    from fastapi import FastAPI
    app = FastAPI()
    @app.get("/")
    async def get_root():
        return {"Message": "Root working"}
    ```

1.  您应该能够在终端中使用您选择的任何服务器运行这个裸骨应用程序：

    ```py
    uvicorn app:app
    ```

简单的根消息在 `127.0.0.1:8000` 上可用，并且应用程序正在运行。

### 连接到 Atlas

现在，是时候将其连接到 Atlas 了。为此，您将使用 FastAPI 的 **生命周期事件**，这是在应用程序启动并开始接收请求之前需要只发生一次的事件处理的新方法。生命周期事件还允许您处理在应用程序完成请求处理后应该触发的其他事件。

注意

FastAPI 网站关于此主题有出色的文档：[`fastapi.tiangolo.com/advanced/events/`](https://fastapi.tiangolo.com/advanced/events/)。

对于本章的使用案例，您将使用一个**异步上下文管理器**，这将允许您提供应用程序实例，并在应用程序启动前后触发事件。按照以下步骤操作：

1.  为了展示这是如何工作的，编辑`app.py`文件：

    ```py
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("Starting up!")
        yield
        print("Shutting down!")
    app = FastAPI(lifespan=lifespan)
    @app.get("/")
    async def get_root():
        return {"Message": "Root working!"}
    ```

    如果您使用之前显示的相同命令启动应用程序，然后使用*Ctrl+C*关闭它，您将看到`print`语句在控制台显示消息。

    `lifespan`事件异步上下文是您将通过设置连接到您的 Atlas 实例的机制。

1.  再次打开`app.py`文件，添加配置设置，更改`lifespan`函数，并引入`Motor`驱动程序：

    ```py
    from fastapi import FastAPI
    from motor import motor_asyncio
    from config import BaseConfig
    settings = BaseConfig()
    async def lifespan(app: FastAPI):
        app.client = motor_asyncio.AsyncIOMotorClient(settings.DB_URL)
        app.db = app.client[settings.DB_NAME]
        try:
            app.client.admin.command("ping")
            print("Pinged your deployment. You have successfully connected to MongoDB!")
            print("Mongo address:", settings.DB_URL)
        except Exception as e:
            print(e)
        yield
        app.client.close()
    app = FastAPI(lifespan=lifespan)
    @app.get("/")
    async def get_root():
        return {"Message": "Root working!"}
    ```

1.  如果您现在启动应用程序，您应该会收到以下类似的消息：

    ```py
    INFO:     Started server process [28228]
    INFO:     Waiting for application startup.
    Pinged your deployment. You have successfully connected to MongoDB!
    Mongo address: <your connection string>
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press
    ```

在这个设置中，您已经实现了很多内容：

+   您已创建了 FastAPI 实例，这是您 API 的骨干。

+   您已使用`pydantic_settings`设置了环境变量，因此它们是可管理和可维护的。

+   您已连接到您设置的 Atlas 集群。

+   您还“附加”了 MongoDB 数据库到应用程序，因此您将能够通过请求方便地从 API 路由器中访问它。

现在，让我们开始实现**CRUD**（创建、读取、更新和删除）操作的路线，从一个稳固且可扩展的设置开始。

# CRUD 操作

几乎每个 Web 应用程序核心的四个基本操作通常被称为**CRUD**（创建、读取、更新和删除）的缩写。这些操作使用户能够通过创建新资源、检索现有资源的一个或多个实例以及修改和删除资源来与数据交互。在这里，使用了一个更正式的 API 定义，但资源在这种情况下只是汽车。

FastAPI 与网络标准紧密相连，因此这些操作映射到特定的 HTTP 请求方法：`POST`用于创建新实例，`GET`用于读取一个或多个汽车，`PUT`用于更新，而`DELETE`用于删除资源。在您的案例中，资源由`cars`表示，它们映射到 MongoDB 文档。

### 设置 API 路由器

在应用程序准备就绪并开始服务基本根端点，设置了环境变量，并建立了与 Atlas MongoDB 数据库的连接后，您现在可以开始实现端点。

实际上，在接下来的章节中，我们将添加一个用户路由器；这将使您能够将单个汽车与特定的用户/销售人员关联起来，并允许进行一些基本的身份验证和授权。

与大多数现代 Web 框架（Express.js、Flask 等）一样，FastAPI 允许您将端点结构化和分组到 API 路由器中。`APIRouter`是一个模块，用于处理与单一类型对象或资源相关的一组操作：在您的案例中，是汽车，稍后是用户。

执行以下步骤以创建用于管理汽车的 API 路由器：

1.  在您的应用程序目录内创建一个专门的文件夹，并命名为 `/routers`。该目录将包含所有 API 路由器。在其内部，创建一个空的 `__init__.py` 文件，将文件夹转换为 Python 包。

1.  现在，创建一个名为 `/routers/cars.py` 的文件。这将是该应用程序中的第一个路由器，但如果有必要，随着应用程序的增长，您可以添加更多。按照惯例，根据它们管理的资源命名路由器。

1.  在 `/routers/cars.py` 内部，开始构建路由器：

    ```py
    from fastapi import APIRouter, Body, Request, status
    from models import CarModel
    router = APIRouter()
    ```

    `APIRouter` 的实例化与创建主 FastAPI 实例非常相似——它可以被视为一个小的 FastAPI 应用程序，它成为主应用程序的一个组成部分，以及其自动文档。

    `APIRouter` 本身没有任何功能——它需要连接到主应用程序 (`app.py`) 才能执行其任务。

1.  在继续之前，让我们修改 `app.py` 文件并将新创建的 `APIRouter` 插入其中：

    ```py
    from fastapi import FastAPI, status
    from fastapi.middleware.cors import CORSMiddleware
    from motor import motor_asyncio
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from fastapi.encoders import jsonable_encoder
    from collections import defaultdict
    from config import BaseConfig
    from routers.cars import router as cars_router
    from routers.users import router as users_router
    settings = BaseConfig()
    async def lifespan(app: FastAPI):
        app.client = motor_asyncio.AsyncIOMotorClient(settings.DB_URL)
        app.db = app.client[settings.DB_NAME]
        try:
            app.client.admin.command("ping")
            print("Pinged your deployment. You have successfully connected to MongoDB!")
        except Exception as e:
            print(e)
        yield
        app. client.close()
    app = FastAPI(lifespan=lifespan)
    app.include_router(cars_router, prefix="/cars", tags=["cars"])
    @app.get("/")
    async def get_root():
        return {"Message": "Root working!"}
    ```

您已创建了第一个 `APIRouter`，它将处理有关汽车的操作，并且您已经通过 `app.py` 文件将其连接到主 FastAPI 实例。

现在，您将通过实现各种操作的处理器来为 `APIRouter` 添加功能。

## POST 处理器

现在，与 `APIRouter` 连接后，您可以返回到 `/routers/cars.py` 文件并创建第一个端点，一个用于创建新实例的 `POST` 请求处理器：

```py
@router.post(
    "/",
    response_description="Add new car",
    response_model=CarModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def add_car(request: Request, car: CarModel = Body(...)):
    cars = request.app.db["cars"]
    document = car.model_dump(
        by_alias=True, exclude=["id"])
    inserted = await cars.insert_one(document)
    return await cars.find_one({"_id": inserted.inserted_id})
```

代码相当简单且易于理解，因为它使用了之前定义的 Pydantic 模型 (`CarModel`)，该模型足够灵活，可以通过别名作为输入和输出模型重用。

创建从模型插入文档的行使用了几个 Pydantic 功能，这些功能在 *第三章*，*Python 类型提示和 Pydantic* 中有介绍，即 *别名* 和 *排除* 字段。

现在，启动应用程序：

```py
uvicorn app:app
```

在另一个终端中，仍然位于您项目的当前工作目录内，并且虚拟环境已激活，使用 **HTTPie** 测试端点：

```py
http POST http://127.0.0.1:8000/cars/ brand="KIA" make="Ceed" year=2015 price=2000 km=100000 cm3=1500
```

您的终端应该输出以下响应：

```py
HTTP/1.1 201 Created
content-length: 109
content-type: application/json
date: Sun, 12 May 2024 15:29:45 GMT
server: uvicorn
{
    "brand": "Kia",
    "cm3": 1500,
    "id": "6640e06ad82a890d261a8a40",
    "km": 100000,
    "make": "Ceed",
    "price": 2000,
    "year": 2015
}
```

您已创建了第一个端点——您可以使用 HTTPie 或 `http://127.0.0.1:8000/docs` 上的交互式文档进一步测试它，并尝试插入一些无效数据，例如年份大于 2024 或类似的数据。

端点应该返回包含信息的 JSON，这将迅速引导您找到问题或向最终用户提供反馈。现在，您将创建用于查看数据库中汽车的 `GET` 处理器。

## 处理 GET 请求

为了查看系统中的资源——汽车，您将使用 HTTP `GET` 方法。FastAPI 充分利用 HTTP 动词语义，并紧密遵循网络标准和良好实践。

按照以下步骤操作：

1.  首先，返回整个汽车集合——如果你已经玩过`POST`端点，你可能已经插入了几辆。继续编辑`/routers/cars.py`文件，让我们添加`GET`处理器：

    ```py
    @router.get(
        "/",
        response_description="List all cars",
        response_model=CarCollection,
        response_model_by_alias=False,
    )
    async def list_cars(request: Request):
        cars = request.app.db["cars"]
        results = []
        cursor = cars.find()
        async for document in cursor:
            results.append(document)
        return CarCollection(cars=results)
    ```

1.  使用 HTTPie 测试此端点：

    ```py
    http http://127.0.0.1:8000/cars/
    ```

    运行前面的命令后，你应该会得到到目前为止插入的所有汽车，以一个漂亮的 JSON 结构呈现。函数签名和装饰器与`POST`端点相似。

1.  不使用`async` `for`（如果你不习惯，一开始可能会觉得有点反直觉），你也可以用以下方式交换空结果列表的填充：

    ```py
    return CarCollection(
        cars=await cars.find().to_list(1000)
        )
    ```

    然后，你可以使用`to_list()`方法将结果获取为一个列表。如果你希望深入了解处理游标的`Motor`文档，他们的页面可能有点枯燥但内容完整：[`motor.readthedocs.io/en/stable/api-tornado/cursors.html`](https://motor.readthedocs.io/en/stable/api-tornado/cursors.html)。

    之后，你将学习如何手动添加分页，因为集合可能会增长到数百辆汽车，你不会希望立即向用户发送数百个结果。现在，创建一个通过 ID 查找单个汽车的`GET`端点。

1.  在同一个`/routers/cars.py`文件中，添加以下`GET`处理器：

    ```py
    @router.get(
        "/{id}",
        response_description="Get a single car by ID",
        response_model=CarModel,
        response_model_by_alias=False,
    )
    async def show_car(id: str, request: Request):
        cars = request.app.db["cars"]
        try:
            id = ObjectId(id)
        except Exception:
            raise HTTPException(status_code=404, detail=f"Car {id} not found")
        if (car := await cars.find_one({"_id": ObjectId(id)})) is not None:
            return car
        raise HTTPException(status_code=404, detail=f"Car with {id} not found")
    ```

端点的逻辑包含在检查集合是否包含具有所需 ID 的汽车，并且 ID 是通过路径参数提供的这一行。Python 的 walrus 运算符（`:=`），也称为赋值表达式，使你的代码更加简洁：如果找到汽车（它不是`None`），则返回，并将真值检查的操作数——汽车实例本身——传递下去；否则，代码将继续执行并抛出 HTTP 异常。

再次，对于 HTTPie 测试命令，你需要查找一个 ID 并将其作为路径参数提供（你的 ID 值将不同于以下）：

```py
http http://127.0.0.1:8000/cars/6640e06ad82a890d261a8a40
```

你已经实现了映射到`GET` HTTP 方法的最重要两种结果列出方法：检索所有项的列表和特定单个项。其他`GET`端点可以基于 MongoDB 聚合、简单查询和过滤来检索查询，但这两个涵盖了基础。

现在，让我们通过`UPDATE`和`DELETE`方法完成 API。

## 更新和删除记录

现在，你将处理最复杂的端点——用于更新汽车实例的`PUT`方法。同样，在同一个`/routers/cars.py`文件中，在`GET`路由之后，继续编辑：

```py
async def update_car(
    id: str,
    request: Request,
    user=Depends(auth_handler.auth_wrapper),
    car: UpdateCarModel = Body(...),
):
    try:
        id = ObjectId(id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Car {id} not found")
    car = {
        k: v
        for k, v in car.model_dump(by_alias=True).items()
        if v is not None and k != "_id"
    }
```

端点函数的第一部分分析提供的用户数据，并检查哪些字段应该被更新，只需通过确认它们在提供的`UpdateCarModel` Pydantic 模型中的存在即可。如果字段在请求体中存在，其值将被传递到`update`字典中。

因此，你得到一个转换后的`car`对象，如果不为空，则将其传递给 MongoDB 的`find_one_and_update()`函数：

```py
    if len(car) >= 1:
        cars = request.app.db["cars"]
        update_result = await cars.find_one_and_update(
            {"_id": ObjectId(id)},
            {"$set": car},
            return_document=ReturnDocument.AFTER,
        )
        if update_result is not None:
            return update_result
        else:
            raise HTTPException(status_code=404, detail=f"Car {id} not found")
```

更新结果简单地执行异步更新，并利用 PyMongo 的`ReturnDocument.AFTER`在更新完成后返回更新后的文档。

最后，你还需要考虑这样一种情况：在更新时没有任何字段被设置，如果发现这种情况，就简单地返回原始文档：

```py
    if (existing_car := await cars.find_one({"_id": id})) is not None:
        return existing_car
    raise HTTPException(status_code=404, detail=f"Car {id} not found")
```

端点提供了两种`404`异常的可能性：当有字段需要更新时，以及当没有字段需要更新时。

现在，使用删除汽车的方法完成基本 CRUD 功能的实现：

```py
@router.delete("/{id}", response_description="Delete a car")
async def delete_car(
    id: str, request: Request, user=Depends(auth_handler.auth_wrapper)
):
    try:
        id = ObjectId(id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Car {id} not found")
    cars = request.app.db["cars"]
    delete_result = await cars.delete_one({"_id": id})
    if delete_result.deleted_count == 1:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    raise HTTPException(status_code=404, detail=f"Car with {id} not found")
```

这可能是最简单的端点；如果找到具有该 ID 的汽车，它将被删除，并在空（`No` `Content`）响应上返回适当的 HTTP 状态。

这就完成了基本的 CRUD 功能，但在继续之前，让我们解决另一个方面的问题，虽然它不是基本功能的一部分，但将在每个实际项目中出现：结果分页。

### 结果分页

每个与数据和用户打交道的应用程序都必须有一种适当的方式来启用和促进他们的沟通。将数百个结果强行推入浏览器并不是最佳解决方案。

使用 MongoDB 以及其他数据库进行结果分页是通过`skip`和`limit`参数实现的。

在这种情况下，你将创建一个简单的、对前端友好的分页系统，使用自定义的 Pydantic 模型，该模型将提供两个额外的 JSON 属性：当前页和`has_more`标志，以指示是否有更多结果页。

此模式将分页 UI 与箭头和页码匹配，这些箭头和页码向用户指示结果总数。

首先，注释掉现有的`GET`路由。打开`models.py`文件并添加以下模型：

```py
class CarCollectionPagination(CarCollection):
    page: int = Field(ge=1, default=1)
    has_more: bool
```

此模型继承自`CarCollection`模型并添加了两个所需的字段——当处理大型且复杂的模型时，此模式非常有用。

在`cars.py`文件中，在实例化`APIRouter`后，添加一个硬编码的常量，该常量将定义每页默认的结果数：

```py
CARS_PER_PAGE = 10
```

现在你将更新（或者更好，完全替换）`routers/cars.py`文件中的`get all`方法：

```py
@router.get(
    "/",
    response_description="List all cars, paginated",
    response_model=CarCollectionPagination,
    response_model_by_alias=False,
)
async def list_cars(
    request: Request,
    page: int = 1,
    limit: int = CARS_PER_PAGE,
):
    cars = request.app.db["cars"]
    results = []
```

函数的第一部分与上一个版本非常相似，但我们有两个新的参数：`page`和`limit`（每页的结果数）。现在，创建实际的分页：

```py
cursor = cars.find().sort("companyName").limit(limit).skip((page - 1) * limit)
    total_documents = await cars.count_documents({})
    has_more = total_documents > limit * page
    async for document in cursor:
        results.append(document)
    return CarCollectionPagination(cars=results, page=page, has_more=has_more)
```

大部分工作直接由 MongoDB 处理，使用`limit`和`skip`参数。端点需要集合中汽车的总数，以便提供有关剩余结果及其存在的信息。

此端点将像之前的端点一样工作，因此为了正确测试它，请打开 MongoDB Compass 并导入一些数据。附带的 GitHub 仓库包含一个名为`cars.csv`的文件，其中包含 1,249 辆汽车。

在导入这些数据后，你可以执行以下`GET`请求：

```py
http http://127.0.0.1:8000/cars/?page=12
```

输出应包含汽车列表，就像上一个案例一样，但也要指明页面和是否有更多结果：

```py
{
    "has_more": false,
    "page": 12
}
```

由于你已经在从数据库中提取总文档计数，你可以扩展此分页模型以包括数据库中的汽车总数或根据当前分页提供的总页数。这将是一个很好的练习，展示了如何轻松扩展和修改 FastAPI 设置。

你已经使用 FastAPI 成功创建了一个功能齐全的 REST API。现在，让我们通过提供图像上传功能来进一步增强应用程序。

# 上传图像到 Cloudinary

虽然 FastAPI 完全能够通过`StaticFiles`模块([`fastapi.tiangolo.com/tutorial/static-files/`](https://fastapi.tiangolo.com/tutorial/static-files/))提供静态文件服务，但你很少会想用你的服务器空间和带宽来存储图像或视频。

许多专业服务可以处理数字资产媒体管理，在本节中，你将学习如何与该领域的主要参与者之一——**Cloudinary**合作。

如其名所示，Cloudinary 是一个基于云的服务，为数字媒体资产和 Web 和移动应用程序提供各种解决方案。这些服务包括上传和存储图像和视频，这正是我们现在将要使用的功能。

然而，Cloudinary 和其他类似的专业服务提供了更多（图像和视频处理、过滤器、自动裁剪和格式化以及实时转换）的功能，它们可能非常适合许多媒体工作流程，尤其是非常繁重的工作流程。

要使用该服务，你首先需要通过[`cloudinary.com/users/register_free`](https://cloudinary.com/users/register_free)上的说明创建一个免费账户。

在成功注册并登录后，你将自动分配一个产品环境密钥，位于左上角。就你的用途而言，你将通过 Python API 进行交互，因为你需要能够通过 FastAPI 将图像上传到你的环境。

要开始使用 Python API，或者任何其他 API，除了这个环境密钥外，你还需要另外两块信息：*API 密钥*和*API 密钥*。这两者都可以从**设置**页面([console.cloudinary.com./settings](http://console.cloudinary.com./settings))通过选择左侧菜单中的**API 密钥**获取。

复制 API 密钥和 API 密钥，或者创建新的并复制到现有的`.``env`文件中：

```py
DB_URL=mongodb+srv://xxxxxx:xxxxxxxx@cluster0.fkm24.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
DB_NAME=carBackend
CLOUDINARY_SECRET_KEY=xxxxxxxxxxxxxxxx
CLOUDINARY_API_KEY=xxxxxxxxxx
CLOUDINARY_CLOUD_NAME=xxxxxxxx
```

环境名称映射为`CLOUDINARY_CLOUD_NAME`，而密钥和 API 密钥则由`CLOUDINARY`前缀。

你还需要修改`config.py`文件以适应新的变量：

```py
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
class BaseConfig(BaseSettings):
    DB_URL: Optional[str]
    DB_NAME: Optional[str]
    CLOUDINARY_SECRET_KEY: Optional[str]
    CLOUDINARY_API_KEY: Optional[str]
    CLOUDINARY_CLOUD_NAME: Optional[str]
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

下一步是安装`cloudinary`Python 包：

```py
pip install cloudinary
```

你还可以将其添加到你的 `requirements.txt` 文件中，此时它应该看起来像这样：

```py
fastapi==0.111.0
motor==3.4.0
uvicorn==0.29.0
httpie==3.2.2
cloudinary==1.40.0
pydantic-settings==2.2.1
```

当涉及到 JavaScript 时，Cloudinary 文档更为丰富，设置上传客户端时似乎有几个怪癖，但本质上是简单的。

### 更新模型

首先，你需要更新 `models.py` 文件以适应新的字段——一个将存储从 Cloudinary 上传的图片 URL 的字符串：

1.  打开 `models.py` 文件，并在其他字段和验证器之后仅添加一行到 `CarModel` 类中：

    ```py
    # add the picture file
        picture_url: Optional[str] = Field(None)
    ```

1.  在这一点上，你应该打开 `cars` 集合，因为你将创建一个新的、空的集合。现在，在 `cars.py` 文件中注释掉之前的 `POST` 处理器路由，并创建一个新的路由，考虑到图像上传过程。

1.  Cloudinary 提供了一个简单的实用模块，称为 `uploader`，需要导入，以及 `cloudinary` 模块本身。在现有的导入之后，添加以下行（`cars.py`）：

    ```py
    import cloudinary
    from cloudinary import uploader  # noqa: F401
    ```

    这些行导入了 `cloudinary` 和 `uploader` 包，而 `# noqa` 行防止代码检查器在保存时删除该行（因为它是从已整体导入的包中导入的）。

1.  下一步是配置你的 Cloudinary 实例，你可以为了方便在 `/routers/cars.py` 文件中这样做，尽管这也可以是应用级别的。

    为了能够读取环境变量，你需要在同一文件中再次实例化 `Settings` 类，并将变量传递给 `cloudinary` 配置对象。

1.  打开 cars 路由器并对其进行修改。现在 `/routers/cars.py` 文件的前一部分应该看起来像这样：

    ```py
    from bson import ObjectId
    from fastapi import (
        APIRouter,
        Body,
        File,
        Form,
        HTTPException,
        Request,
        UploadFile,
        status,
    )
    from fastapi.responses import Response
    from pymongo import ReturnDocument
    import cloudinary
    from cloudinary import uploader  # noqa: F401
    from config import BaseConfig
    from models import CarCollectionPagination, CarModel, UpdateCarModel
    settings = BaseConfig()
    router = APIRouter()
    CARS_PER_PAGE = 10
    cloudinary.config(
        cloud_name=settings.CLOUDINARY_CLOUD_NAME,
        api_key=settings.CLOUDINARY_API_KEY,
        api_secret=settings.CLOUDINARY_SECRET_KEY,
    )
    ```

    现在，你必须以不同的方式处理 `POST` 处理器，因为它将接受表单和一个文件（你的汽车图片），而不是 JSON。你需要接受表单数据：

    ```py
    @router.post(
        "/",
        response_description="Add new car with picture",
        response_model=CarModel,
        status_code=status.HTTP_201_CREATED,
    )
    async def add_car_with_picture(
        request: Request,
        brand: str = Form("brand"),
        make: str = Form("make"),
        year: int = Form("year"),
        cm3: int = Form("cm3"),
        km: int = Form("km"),
        price: int = Form("price"),
        picture: UploadFile = File("picture"),
    ):
    ```

    所有 `CarModel` 字段现在都映射到具有名称的表单字段，而图片被定义为 `UploadFile` 并期望一个文件。

    继续使用相同的函数，并添加上传功能：

    ```py
        cloudinary_image = cloudinary.uploader.upload(
            picture.file, crop="fill", width=800
        )
        picture_url = cloudinary_image["url"]
    ```

    处理实际上传的代码非常简单：只需调用 `uploader` 并传入接收到的文件，你可以使用许多选项、转换和过滤器。

注意

Cloudinary 文档详细介绍了可用的转换：[`cloudinary.com/documentation/transformations_intro`](https://cloudinary.com/documentation/transformations_intro).

在你的情况下，你只是裁剪图片并设置最大宽度。Cloudinary 会在图片上传后返回一个 URL，这个 URL 将成为模型的一部分，以及我们之前使用的其他数据。

最后，你可以构建一个 Pydantic 模型来表示汽车，并将其传递给 MongoDB 的 `cars` 集合：

```py
    car = CarModel(
        brand=brand,
        make=make,
        year=year,
        cm3=cm3,
        km=km,
        price=price,
        picture_url=picture_url,
    )
    cars = request.app.db["cars"]
    document = car.model_dump(by_alias=True, exclude=["id"])
    inserted = await cars.insert_one(document)
    return await cars.find_one({"_id": inserted.inserted_id})
```

你可以通过 FastAPI 在`127.0.0.1:8000/docs`上提供的交互式文档测试端点；只需选择一张图片并将其传递给根路由的`POST`处理程序中的文件字段，别忘了填写剩余的字段，否则会出现错误——就像处理 JSON 一样。

你也可以使用 HTTPie 测试路由，但首先提供一张图片并相应地命名它：

```py
http --form POST 127.0.0.1:8000/cars brand="Ford" make="Focus" year=2000 cm3=1500 price=12000 km=23000 picture="ford.jpg"
```

在准备好`Cars` API 路由器之后，现在你将创建第二个路由器来处理用户。

# 添加用户模型

你已经成功创建了一个由 Cloudinary 图像托管和处理能力驱动的 REST API，并且通过类似的过程，你可以轻松地将其他第三方服务集成到你的 API 中，使你的应用程序更加复杂和强大。

然而，没有认证，将即使是简单的 API 部署到线上也是非常危险的。例如，一个恶意用户（甚至是一个愿意恶作剧的孩子）可以轻易地用你不想显示的图片“轰炸”你的 API，并且数量足以迅速填满你的免费配额。因此，在将你的 API 提交到 GitHub 并部署——在这种情况下，到 Render.com——之前，你将添加一个用户模型和一个与*第六章*中展示的非常相似的基于 JWT 的认证方案，*认证* *和授权*。

在下一节中，你将创建一个简单的用户模型，并允许用户登录到应用程序，以便执行一些其他情况下不可用的操作——即创建、更新和删除资源（汽车）。你将从将认证逻辑抽象成一个类开始。

### 创建认证功能

在本节中，你将实现一个认证类，类似于在*第六章*中使用的，*认证和授权*，该类将抽象出认证和授权所需的功能——密码加密、JWT 编码和解码，以及用于保护路由的依赖项。请按照以下步骤操作：

1.  首先，在你的项目根目录下创建一个名为`authentication.py`的文件，并导入所需的认证模块：

    ```py
    from datetime import datetime
    import jwt
    from fastapi import HTTPException, Security
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from passlib.context import CryptContext
    ```

1.  接下来，实现一个`AuthHandler`类，它将提供所有必要的功能来散列和验证密码以及编码和解码令牌：

    ```py
    class AuthHandler:
        security = HTTPBearer()
        pwd_context = CryptContext(
            schemes=["bcrypt"], deprecated="auto"
        )
        secret = "FARMSTACKsecretString"
        def get_password_hash(self, password):
            return self.pwd_context.hash(password)
        def verify_password(
            self, plain_password, hashed_password
        ):
            return self.pwd_context.verify(
                plain_password, hashed_password
            )
        def encode_token(self, user_id, username):
            payload = {
                "exp": datetime.datetime.now(
                    datetime.timezone.utc)
                + datetime.timedelta(minutes=30),
                "iat": datetime.datetime.now(datetime.timezone.utc),
                "sub": {
                    "user_id": user_id,
                     "username": username},
            }
            return jwt.encode(payload, self.secret, algorithm="HS256")
        def decode_token(self, token):
    try:
        payload = jwt.decode(
            token, self.secret, algorithms=["HS256"]
        )
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Signature has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )
    ```

1.  最后，你将在文件末尾添加`auth_wrapper`函数，该函数将被注入到需要认证用户的 FastAPI 端点中：

    ```py
        def auth_wrapper(
            self,
            auth: HTTPAuthorizationCredentials =
                Security(security)
        ):
            return self.decode_token(auth.credentials)
    ```

    认证类几乎与在*第六章**认证和授权*中定义的相同——它提供了密码散列和验证、JWT 编码和解码的方法，以及一个方便的`auth_wrapper`方法，用作依赖注入。

1.  在 `authentication.py` 文件准备好后，添加用户模型，它与上一章中定义的模型非常相似，请注意，此模型可能更加复杂。

1.  在 `models.py` 文件中，编辑 `CarModel` 类并添加另一个字段——`user_id`。这样，您就可以将插入的汽车与特定用户关联起来，并要求每个创建操作都有效用户：

    ```py
    class CarModel(BaseModel):
        id: Optional[PyObjectId] = Field(alias="_id", default=None)
        brand: str = Field(...)
        make: str = Field(...)
        year: int = Field(..., gt=1970, lt=2025)
        cm3: int = Field(..., gt=0, lt=5000)
        km: int = Field(..., gt=0, lt=500 * 1000)
        price: int = Field(..., gt=0, lt=100000)
        user_id: str = Field(...)
        picture_url: Optional[str] = Field(None)
    ```

1.  更新汽车的模型不需要 `user_id` 字段，因为您不希望使该字段可编辑。现在，在所有汽车模型之后，让我们在同一个 `models.py` 文件中添加与用户相关的模型：

    ```py
    class UserModel(BaseModel):
        id: Optional[PyObjectId] = Field(alias="_id", default=None)
        username: str = Field(..., min_length=3, max_length=15)
        password: str = Field(...)
    class LoginModel(BaseModel):
        username: str = Field(...)
        password: str = Field(...)
    class CurrentUserModel(BaseModel):
        id: PyObjectId = Field(alias="_id", default=None)
        username: str = Field(..., min_length=3, max_length=15)
    ```

三种模型对应于您将访问用户数据的三种方式：包含所有数据的完整模型、登录和注册模型，以及应返回 `_id` 和用户名的当前用户。

### 创建用户路由器

在设置 Pydantic 模型之后，为用户创建一个新的路由器，并允许一些基本操作，例如注册、登录和基于 JWT 验证用户。

打开 `routers` 文件夹内名为 `users.py` 的文件并添加导入：

```py
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from authentication import AuthHandler
from models import CurrentUserModel, LoginModel, UserModel
router = APIRouter()
auth_handler = AuthHandler()
```

`authhandler` 类封装了所有的身份验证逻辑，您将在端点函数中看到此功能。

让我们创建注册路由：

```py
@router.post("/register", response_description="Register user")
async def register(request: Request, newUser: LoginModel = Body(...)) -> UserModel:
    users = request.app.db["users"]
    # hash the password before inserting it into MongoDB
    newUser.password = auth_handler.get_password_hash(newUser.password)
    newUser = newUser.model_dump()
    # check existing user or email 409 Conflict:
    if (
        existing_username := await users.find_one({"username": newUser["username"]})
        is not None
    ):
        raise HTTPException(
            status_code=409,
            detail=f"User with username {newUser['username']} already exists",
        )
    new_user = await users.insert_one(newUser)
    created_user = await users.find_one({"_id": new_user.inserted_id})
    return created_user
```

端点执行与*第六章*中所示的功能相同，*身份验证和授权*，但这次，您正在处理一个真实的 MongoDB 集合。登录功能也非常相似：

```py
@router.post("/login", response_description="Login user")
async def login(request: Request, loginUser: LoginModel = Body(...)) -> str:
    users = request.app.db["users"]
    user = await users.find_one({"username": loginUser.username})
    if (user is None) or (
        not auth_handler.verify_password(loginUser.password, user["password"])
    ):
        raise HTTPException(status_code=401, detail="Invalid username and/or password")
    token = auth_handler.encode_token(str(user["_id"]), user["username"])
    Wrong indentation. check and replace with:
response = JSONResponse(
    content={
        "token": token,
        "username": user["username"]
    }
)
    return response
```

如果找不到用户或密码不匹配，端点将响应 HTTP `401` 状态并抛出一个通用消息；否则，将返回用户名和令牌。

最终端点由一个 `/me` 路由组成——该路由将由前端（React）定期使用以检查现有的 JWT 及其有效性：

```py
@router.get(
    "/me",
    response_description="Logged in user data",
    response_ model=CurrentUserModel
)
async def me(
    request: Request,
    response: Response,
    user_data=Depends(auth_handler.auth_wrapper)
):
    users = request.app.db["users"]
    currentUser = await users.find_one(
        {"_id": ObjectId(user_data["user_id"])}
    )
    return currentUser
```

在完成 `users` 路由器后，将其连接到 `app.py` 文件，位于 `cars` 路由器下方：

```py
app.include_router(
    cars_router, prefix="/cars", tags=["cars"]
    )
app.include_router(
    users_router, prefix="/users", tags=["users"]
    )
```

包含管理汽车的 `APIRouter` 的 `cars.py` 文件将需要更新以考虑新添加的用户数据。创建端点现在将如下所示：

```py
@router.post(
    "/",
    response_description="Add new car with picture",
    response_model=CarModel,
    status_code=status.HTTP_201_CREATED,
)
async def add_car_with_picture(
    request: Request,
    brand: str = Form("brand"),
    make: str = Form("make"),
    year: int = Form("year"),
    cm3: int = Form("cm3"),
    km: int = Form("km"),
    price: int = Form("price"),
    picture: UploadFile = File("picture"),
    user: str =Depends(auth_handler.auth_wrapper),
):
```

用户数据通过依赖注入和 `auth_wrapper` 提供。函数的其余部分基本未更改——您只需要登录用户的 `user_id` 值：

```py
cloudinary_image = cloudinary.uploader.upload(
    picture.file, folder="FARM2", crop="fill", width=800
)
picture_url = cloudinary_image["url"]
car = CarModel(
    brand=brand,
    make=make,
    year=year,
    cm3=cm3,
    km=km,
    price=price,
    picture_url=picture_url,
    user_id=user["user_id"],
)
cars = request.app.db["cars"]
document = car.model_dump(by_alias=True, exclude=["id"])
inserted = await cars.insert_one(document)
return await cars.find_one({"_id": inserted.inserted_id})
```

API 现在相当完整；它处理各种复杂性的数据，并可以使用顶级云服务处理图像。然而，在将您的 API 部署到在线云平台供全世界查看之前，还有一件事需要做：设置 **跨源资源共享**（**CORS**）中间件。

# FastAPI 中间件和 CORS

中间件的概念在几乎每个值得信赖的 Web 框架中都很常见，FastAPI 也不例外。中间件只是一个在将请求传递给路径操作处理之前接受请求的函数，并在它们返回之前进行响应。

这个简单的概念非常强大，有众多用途——一个中间件可以检查包含认证数据（例如令牌）的特定头信息，并根据情况接受或拒绝请求，它可以用于速率限制（通常与 Redis 键值数据库一起使用），等等。

在 FastAPI 中创建中间件基于 *Starlette* 的中间件，就像 FastAPI 中大多数与网络相关的概念一样，文档提供了一些很好的示例：[`fastapi.tiangolo.com/tutorial/middleware/`](https://fastapi.tiangolo.com/tutorial/middleware/)。

在你的应用程序中，你将使用现成的中间件来启用基于 FastAPI 的后端——该后端将运行在一台机器上——与运行在不同源的前端（在你的情况下，React）进行通信。

CORS 指的是当后端和前端位于不同的源时应用的政策，默认情况下它非常严格——只允许使用相同源的系统之间共享数据（例如调用 JavaScript fetch 函数）：协议（例如 HTTP）、域名（例如 [www.packt.com](http://www.packt.com)）和端口号（例如 `3000` 或 `80`）的组合。

默认情况下，该策略阻止所有通信，所以如果你像现在这样部署你的后端，你将无法从运行在同一台机器但不同端口的 React.js 或 Next.js 应用程序中访问它。

FastAPI 解决这个任务的方案是通过中间件实现的，并且它允许细粒度的精确控制。

在你的 `app.py` 文件中，导入以下内容以导入 CORS 中间件：

```py
from fastapi.middleware.cors import CORSMiddleware
```

导入中间件后，你需要对其进行配置。在用生命周期实例化 FastAPI 实例后，添加中间件：

```py
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

这是一个通用的 CORS 设置，在生产环境中应避免使用，但对我们目的和这个示例后端来说已经足够了。方括号包含允许的方法列表（例如 `POST`、`GET` 等）、源、头信息和是否允许凭据。

你可以重新启动 Uvicorn 服务器并检查它是否像以前一样工作。现在，你将在云平台上部署后端。

# 部署到 Render.com

*Render.com* 是众多简化部署和管理 Web 应用程序、API、静态网站和其他类型软件项目的现代云平台之一。它为开发者提供了一个直观且简单的界面，以及强大的自动化工具和管道。

部署 FastAPI 实例的方法有很多：Vercel（主要作为 Next.js 背后的公司）、Fly.io、Ralway、Heroku 等等。

在这种情况下，我们将选择 Render.com，因为它提供了一个简单、快速、简化的部署流程，并提供免费层和优秀的文档。

部署过程可以分为几个步骤，您将简要地审查每一个；如果您希望快速了解，也可以访问他们的 FastAPI 专用页面：[`docs.render.com/deploy-fastapi`](https://docs.render.com/deploy-fastapi)。

这里是步骤：

1.  为您的后端设置一个 GitHub 仓库。

    再次确保您的`.gitignore`文件包含`.env`文件的条目，以及 Python 环境的`env/`目录——您不希望意外地将机密和密码提交到公共仓库，也不希望上传整个虚拟环境的内容。

    如果您还没有将后端最后的更改提交，现在就使用以下命令进行提交：

    ```py
    git add .
    git commit -m "ready for deployment"
    ```

    现在，前往[github.com](http://github.com)，使用您的凭证登录，并创建一个新的仓库。您可以随意命名；在这个例子中，我们将使用名称`FastAPIbackendCh7`。

1.  设置 Render.com 账户。

    现在，前往[render.com](http://render.com)并创建一个免费账户。您可以使用 GitHub 账户登录，然后导航到仪表板链接：[dashboard.render.com](http://dashboard.render.com)。

    定位到**新+**按钮，选择**网络服务**。在下一个提示中，选择**从 Git 仓库构建和部署**并点击**下一步**。

1.  选择 GitHub 仓库。

    在右侧菜单中选择**GitHub Configure**账户，您将被带到 GitHub，要求您安装 Render。选择您的账户，即您用于后端仓库源的那个账户，然后继续选择仓库。这将使 Render 知道要拉取哪个仓库。

1.  配置网络服务。

    这是整个过程中最重要且最复杂的步骤。Render 已经知道涉及哪个仓库，现在它必须获取部署网络服务所需的所有数据。我们将逐一检查：

    +   `main`，尤其是在我们的案例中它是唯一分支的情况下。

    +   `/`.

    +   **运行时**：您将使用 Python 3；它应该会被 Render 自动识别。

    +   **构建命令**：设置环境的命令——在您的案例中，是 Python 3 虚拟环境，因此应该是以下命令：

        ```py
        pip install -r requirements.txt
        ```

    +   `80`），命令应该是以下这样：

        ```py
        uvicorn app:app --host 0.0.0.0 --port 80
        ```

    +   `.env`文件，一个接一个：`DB_URL`和`DB_NAME`用于 MongoDB，以及三个 Cloudinary 变量。

在确认您已经输入了所有设置和变量后，您最终可以点击蓝色的**创建网络服务**按钮。

最终的设置页面将类似于以下图片。设置页面相当长，您可能需要滚动一下，但首先必须指定的是服务的名称和区域：

![](img/B22406_07_01.png)

图 7.1：Render 网络服务常规设置页面

在设置名称和区域后，您将看到您选择的仓库和要部署的分支（在您的案例中是**main**）。您可以将根目录默认留空。

![](img/B22406_07_02.png)

图 7.2：仓库和分支设置

接下来，您将指定构建和启动命令。构建命令是安装您的 Python 环境的那一个，而启动命令则是启动您的网络服务——您的 API。

![图片](img/B22406_07_03.png)

图 7.3：构建和启动命令

在开始实际的部署命令之前，最后一步是将环境变量传递给 Render.com：

![图片](img/B22406_07_04.png)

![图片](img/B22406_07_04.png)

在启动部署流程后，您需要稍等片刻——服务将需要创建一个新的 Python 环境，安装所有必需的依赖项，并启动服务。在过程完成后，您可以在页面上的 URL 上点击（在您的例子中，将是[`farm2ch7.onrender.com`](https://farm2ch7.onrender.com)，您将需要使用另一个地址）并在线检查您的 API。

您的 API 现在已经在互联网上上线，并准备好接收请求。值得一提的是，由于 FastAPI 最近越来越受欢迎，越来越多的托管服务和`.env`文件。

# 摘要

在本章中，您将一个简单的业务需求转变为一个完全功能性的 API，并部署到了互联网上。

您创建了 Pydantic 模型，并对数据结构应用了一些约束，学习了如何连接到 MongoDB Atlas 实例，并开发了一个基本但功能齐全的 CRUD 功能 FASTAPI 服务。

您已经学会了如何通过 Pydantic 来建模实体（在您的例子中，是汽车和用户），以及如何通过简单的 Pythonic FastAPI 端点，将数据无缝地流向和从您选择的数据库——MongoDB——中流动。

您通过`pydantic_settings`管理了密钥——用于连接到 MongoDB Atlas 和 Cloudinary——并精心设计了简单而灵活的模型，这些模型可以轻松地适应更多需求，进行扩展或增加更多功能。

服务现在已准备好在前端使用——最终，赋予全栈 Web 应用生命。

在下一章中，您将向这个 API 添加一个简单的用户模型，并构建一个 React 前端，该前端将消费 FastAPI 后端。
