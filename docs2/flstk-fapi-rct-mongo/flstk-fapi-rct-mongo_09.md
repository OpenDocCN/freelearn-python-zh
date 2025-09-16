# 9

# 使用 FastAPI 和 Beanie 进行第三方服务集成

在学习了构成 FARM 堆栈的工具之后，你将在本章中看到它们在一个更复杂的设置中结合使用。你将基于你对 Pydantic 和 FastAPI 的知识，了解**Beanie**，这是最受欢迎的 MongoDB **对象-文档映射器**（**ODM**）之一，以及它如何使你的代码更高效并提升你的开发者体验。

最后，你将看到当需要扩展应用程序以包含外部第三方功能时，堆栈的灵活性如何有用。在本章中，你将添加一个完全基于 AI 的销售助手，该助手将利用 OpenAI 创建吸引人的汽车描述，然后你将使用**Resend** API 服务发送自动化的电子邮件。

这些功能在现代 Web 应用程序的要求中变得越来越重要，通过本章，你将看到正确的一组工具如何使应用程序开发更高效。

本章将指导你完成以下任务：

+   安装和使用 Beanie – 一个 Python MongoDB ODM

+   了解 Beanie 的基本功能（连接、CRUD 操作和聚合）

+   使用 FastAPI 的后台任务处理长时间运行的过程，同时保持应用程序的响应性

+   从应用程序中编程发送电子邮件

+   集成 OpenAI 的 ChatGPT（或任何其他**大型语言模型**（**LLM**））

# 技术要求

本章的技术要求与我们在 FastAPI 中创建后端章节中的要求相似，增加了用于电子邮件发送功能和对 AI 集成的几个库和服务：

+   Python 3.11.7 或更高版本

+   配置了 Python 扩展的 Visual Studio Code（与*第三章*中相同）

+   MongoDB Atlas 上的账户

+   Render.com 上的账户（如果你希望部署 FastAPI 后端）

+   一个具有 API 访问权限的 OpenAI 账户，或者如果你不想部署应用程序并产生费用，可以使用免费的、本地运行的 LLM，如 Llama 2 或 Llama 3

+   Netlify 账户（免费级别）

我们强烈建议从之前账户的免费（或最便宜的）级别开始，并确保你在这些环境中感到舒适。

在解决了技术要求之后，让我们讨论你将在本章中构建的项目。

# 项目概述

在你运营一个（小型）二手车销售代理机构的情境下，要求与前面章节中的要求有些相似。你将构建一个用于显示待售汽车信息和图片的 Web 应用的后端。与前面的章节不同，现在你将使用 ODM，并且将包括电子邮件发送和 OpenAI 集成，这些将由 FastAPI 的后台任务处理。

汽车数据模型将由 Pydantic 和 Beanie 处理。应用程序将需要认证用户，而你将使用`iron-session`。

最后，你将集成一个 LLM API（在这种情况下，是 OpenAI），以帮助创建有用的汽车模型描述，列出新插入的汽车模型在营销页面上的优缺点，并在每次新汽车广告插入时向指定的收件人发送定制电子邮件。

注意

LLMs 是专门设计用于生成和理解人类语言的机器学习系统。在大型数据集上训练后，它们能够在文本摘要和生成、翻译和图像生成等任务上高效执行。在过去的几年中，LLMs 获得了流行和采用，并且随着时间的推移，它们的实施领域将只会增长。

在下一节中，你将学习如何使用 FastAPI 和 Beanie 创建后端，以及如何集成 OpenAI 和电子邮件发送功能。

# 使用 FastAPI 和 Beanie 构建后端

为了简化起见，并使应用程序尽可能具有说明性，本章中你将构建的 API 将与在 *第七章*，*使用 FastAPI 构建后端* 中构建的 API 差别不大。这样，你将能够自然地掌握使用 Motor（或 PyMongo）直接和 Beanie ODM 的方法之间的主要差异。

**对象关系映射器**（**ORMs**）和 **ODMs** 是工具，其主要目的是抽象底层数据库（无论是关系型数据库还是非关系型数据库），并简化开发过程。一些著名的 Python 示例包括 **Django ORM** 和 **SQLAlchemy**——两个经过验证和实战检验的解决方案——以及由 FastAPI 的创建者创建的 **SQLModel**，它与 FastAPI/Pydantic 世界紧密集成。

在 Python 和 MongoDB 社区中越来越受欢迎的两个现代 ODM 是 **Beanie** ([`beanie-odm.dev/`](https://beanie-odm.dev/)) 和 **Odmantic** ([`art049.github.io/odmantic/`](https://art049.github.io/odmantic/))。在这个项目中，你将使用这两个中更成熟、更老的那个——Beanie ODM。

## Beanie ODM 简介

Beanie 是 Python 最受欢迎的 MongoDB ODM 之一。ODM 是一种编程技术，允许开发人员直接与表示 NoSQL 文档的类（在我们的例子中是 Python 类）一起工作。使用 Beanie 时，每个 MongoDB 集合都映射到一个相应的文档类，这使得你可以检索或聚合数据，并执行 CRUD 操作，通过消除样板代码的必要性来节省时间。

Beanie 还优雅地处理 MongoDB 的 `ObjectId` 类型，并且由于其文档类基于 Pydantic，你可以直接使用 Pydantic 的所有强大验证和解析功能。

简而言之，Beanie 的显著特性包括以下内容：

+   异步的，基于 Motor 驱动器，非常适合性能良好的 FastAPI 应用程序

+   基于 Pydantic 并兼容 Pydantic 版本 2

+   基于模式，无缝处理 `ObjectId` 字符串转换

+   简单的 CRUD 操作，以及支持 MongoDB 强大的聚合框架

在下一节中，你将通过创建一个 Beanie 驱动的应用程序来开始学习 ODM 的某些功能。

### 创建 Beanie 应用程序

你将通过创建一个新应用程序并探索 ODM 提供的功能来学习如何使用 Beanie——连接到数据库、将集合映射到文档类，以及在文档上执行 CRUD 操作。

要开始项目并搭建 FastAPI 应用程序，请执行以下步骤：

1.  创建一个新文件夹（`chapter9`）和一个虚拟环境，使用以下命令：

    ```py
    python -m venv venv
    ```

1.  使用以下命令激活虚拟环境（适用于 Linux 或 Mac）：

    ```py
    source venv/bin/activate
    ```

    或者，对于 Windows 系统，使用以下命令：

    ```py
    venv\Scripts\activate.bat
    ```

1.  激活它，并创建一个包含以下包的初始 `requirements.txt` 文件：

    ```py
    fastapi==0.111.0
    fastapi_cors==0.0.6
    beanie==1.26.00
    bcrypt==4.0.1
    cloudinary==1.40.0
    uvicorn==0.30.1
    pydantic-settings
    PyJWT==2.8.0
    python-multipart==0.0.9
    openai==1.33.0
    resend==2.0.0
    ```

1.  通过运行以下命令安装所需的包：

    ```py
    pip install –r requirements.txt
    ```

    如果你仔细查看 `requirements.txt` 文件，你会注意到你正在安装一个新的包——`fastapi-cors`——它有助于管理 `.env` 文件，然后创建一个包含以下内容的 `.gitignore` 文件：

    ```py
    .env
    .venv
    env/
    venv/
    ```

在准备基本包和设置之后，你现在将使用 Beanie 创建模型。

## 使用 Beanie 定义模型

在搭建主要 FastAPI 应用程序之前，你将学习 Beanie 如何处理数据模型。如前所述，Beanie 的 `Document` 类代表最终将保存到 MongoDB 数据库中的文档，这些模型继承自 Beanie 的 `Document` 类，而 `Document` 类本身是基于 Pydantic 的 `BaseModel` 类。正如 Beanie 网站所述：“Beanie 中的 `Document` 类负责映射和处理集合中的数据。它继承自 Pydantic 的 `BaseModel` 类，因此遵循相同的数据类型和解析行为。” ([`beanie-odm.dev/tutorial/defining-a-document/`](https://beanie-odm.dev/tutorial/defining-a-document/))

让我们开始创建模型，同时记住该文件还将包含几个纯 Pydantic 模型，用于输入和输出的验证（并非所有模型都是基于 Beanie 的，只有映射集合中文档的模型）：

1.  在目录根目录下创建一个名为 `models.py` 的文件，并导入必要的模块：

    ```py
    from datetime import datetime
    from typing import List, Optional
    from beanie import Document, Link, PydanticObjectId
    from pydantic import BaseModel, Field
    ```

    这段代码中唯一的新导入来自 Beanie：你正在导入 `Document` 类——Beanie 用于处理数据的工具类，以及 `Link`（用于引用数据，因为你不会在汽车文档中嵌入用户数据，而是引用用户）和 `PydanticObjectId`——一个表示与 Pydantic 兼容的 `ObjectId` 字段类型。

1.  继续在 `models.py` 文件上工作并创建基本用户模型：

    ```py
    class User(Document):
        username: str = Field(min_length=3, max_length=50)
        password: str
        email: str
        created: datetime = Field(default_factory=datetime.now)
        class Settings:
            name = "user"
        class Config:
            json_schema_extra = {
                "example": {
                    "username": "John",
                    "password": "password",
                    "email": "john@mail.com",
                }
            }
    ```

    `User`模型继承自 Beanie 的`Document`类而不是 Pydantic 的`BaseModel`类，但其余部分大致相同。实际上，`Document`类基于`BaseModel`类并继承其功能——你能够使用具有默认工厂的 Pydantic 字段来创建`datetime`类型。

    然后，你使用了`Settings`类来指定将在 MongoDB 中使用的集合名称。这个类非常强大，允许在保存时设置缓存、索引、验证以及更多功能，如你可以在文档页面看到：[`beanie-odm.dev/tutorial/defining-a-document/#settings`](https://beanie-odm.dev/tutorial/defining-a-document/#settings)。

1.  继续使用相同的`models.py`文件，你现在将提供一些用于特定目的的 Pydantic 模型：注册新用户、用户登录以及提供当前用户的信息：

    ```py
    class RegisterUser(BaseModel):
        username: str
        password: str
        email: str
    class LoginUser(BaseModel):
        username: str
        password: str
    class CurrentUser(BaseModel):
        username: str
        email: str
        id: PydanticObjectId
    ```

1.  之前的代码应该感觉熟悉，因为它完全基于 Pydantic，所以定义汽车的文档模型：

    ```py
    class Car(Document):
        brand: str
        make: str
        year: int
        cm3: int
        price: float
        description: Optional[str] = None
        picture_url: Optional[str] = None
        pros: List[str] = []
        cons: List[str] = []
        date: datetime = datetime.now()
        user: Link[User] = None
        class Settings:
            name = "car"
    ```

    Beanie 文档模型包含你在整本书中使用的所有字段，以及一些新的字段：两个字符串列表，将包含每个汽车模型的优点和缺点的小文本片段——类似于*c**ompact 和易于停放。此外，汽车描述有意留空——这些字段将在稍后的后台任务中，通过 OpenAI 聊天完成提示来填充。

    这个模型的有趣之处在于`user`部分：`Link`字段类型提供了一个直接链接到用户。你可以查看文档以了解 Beanie 关系可以实现什么以及当前的限制是什么：[`beanie-odm.dev/tutorial/relations/`](https://beanie-odm.dev/tutorial/relations/)。

    Beanie 通过相应字段中的链接来管理关系，在撰写本文时，仅支持顶级字段。相关文档的链接可以是链接、可选链接以及链接列表，以及反向链接。

    反向链接是反向关系：如果一个名为`House`的对象有一个指向所有者——例如一个`Person`对象——的链接，那么该`Person`对象可以通过反向链接拥有所有房屋。

1.  最后，添加一个用于更新汽车的`UpdateCar` Pydantic 模型：

    ```py
    class UpdateCar(BaseModel):
        price: Optional[float] = None
        description: Optional[str] = None
        pros: Optional[List[str]] = None
        cons: Optional[List[str]] = None
    ```

注意，你几乎在字段上没有定义任何验证——这样做只是为了节省空间并简化模型。由于 Beanie 基于 Pydantic，它可以依赖 Pydantic 的全部功能，从而实现复杂而强大的验证。

现在已经定义了模型，你可以继续连接到 MongoDB 数据库。提前定义模型很重要，因为它们的名称将被输入到 Beanie 初始化代码中，你将在下一节中看到。

## 连接到 MongoDB 数据库

Beanie ODM 使用`pydantic-settings`及其`BasicSettings`类，以便在应用程序内部轻松访问环境变量。

该过程与在*第七章*中使用的类似，即*使用 FastAPI 构建后端*：

+   环境变量存储在`.env`文件中。

+   `pydantic-settings`用于读取环境变量并创建一个设置对象（通过`config.py`文件）。

+   这些设置，连同模型一起，用于初始化到 Atlas 的数据库连接。

要创建数据库连接并使用模型，请执行以下步骤：

1.  使用`pydantic-settings`定义配置和环境变量。由于你需要在初始化数据库连接之前获取设置，并且它们是从环境中读取的，因此请填充将包含环境变量的`.env`文件，然后通过`config.py`文件读取并将它们实例化为设置对象。

    `.env`文件应包含以下条目：

    ```py
    DB_URL=mongodb://localhost:27017/ or the Atlas address
    CLOUDINARY_SECRET_KEY=<cloudinary.secret.key>
    CLOUDINARY_API_KEY=<cloudinary.api.key>
    CLOUDINARY_CLOUD_NAME=<cloudinary.cloud.name>
    OPENAI_API_KEY=<openai.api.key>
    RESEND_API_KEY=<resend.api.key>
    ```

    你将在稍后设置 OpenAI 和 Resend API 密钥，但现在，你可以插入 MongoDB Atlas 和`config.py`的其他值。打开`config.py`文件，创建`BaseConfig`类以读取环境值并轻松覆盖这些值，基于所需的配置：

    ```py
    from typing import Optional
    from pydantic_settings import BaseSettings, SettingsConfigDict
    class BaseConfig(BaseSettings):
        DB_URL: Optional[str]
        CLOUDINARY_SECRET_KEY: Optional[str]
        CLOUDINARY_API_KEY: Optional[str]
        CLOUDINARY_CLOUD_NAME: Optional[str]
        OPENAI_API_KEY: Optional[str]
        RESEND_API_KEY: Optional[str]
        model_config = SettingsConfigDict(
            env_file=".env", extra="ignore"
        )
    ```

1.  与使用 Beanie 连接 MongoDB 数据库相比，与基于 Motor 的普通连接的差异在`database.py`文件中变得明显，你将在同一根目录中创建此文件，并用以下代码填充：

    ```py
    import motor.motor_asyncio
    from beanie import init_beanie
    from config import BaseConfig
    from models import Car, User
    settings = BaseConfig()
    async def init_db():
        client = motor.motor_asyncio.AsyncIOMotorClient(
            settings.DB_URL
        )
        await init_beanie(database=client.carAds,
            document_models=[User, Car]
        )
    ```

初始化代码被突出显示：异步`init_beanie`函数需要 Motor 客户端和文档模型。

在定义了模型并建立了数据库连接后，你现在将开始构建 FastAPI 应用程序和路由器。

## 创建 FastAPI 应用程序

所有必要的组件都已就绪，现在你已经准备好了连接到 MongoDB 数据库的连接，可以开始构建应用程序。使用新创建的`database.py`文件连接到你的 MongoDB 实例，并将其包装在生命周期上下文管理器中，以确保应用程序启动时连接，并在关闭时删除连接。

要创建主 FastAPI 应用程序文件（`app.py`），请执行以下步骤：

1.  在根目录中创建`app.py`文件，它将非常类似于在*第七章*中创建的，即*使用 FastAPI 构建后端*：

    ```py
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi_cors import CORS
    from database import init_db
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_db()
        yield
    app = FastAPI(lifespan=lifespan)
    init_db function, you imported the fastapi_cors package, which allows easier management of CORS.All you need to do now is add one line to the `.env` file to specify the allowed origins: `ALLOW_ORIGINS=*`.You can explore the documentation of this simple package here: [`pypi.org/project/fastapi-cors/`](https://pypi.org/project/fastapi-cors/).
    ```

1.  连接初始化代码嵌套在一个生命周期事件中，就像之前使用 Motor 的解决方案一样，而其余的代码只是包含你即将创建的路由和一个根端点：

    ```py
    @app.get("/", tags=["Root"])
    async def read_root() -> dict:
        return {"message": "Welcome to your beanie powered app!"}
    ```

1.  如果你已安装了 FastAPI 的较新版本（0.111 或更高版本），该版本会安装`fastapi-cli`包，你现在可以使用以下命令启动开发 FastAPI 服务器：

    ```py
    fastapi dev
    ```

    或者，你可以使用以下标准代码行：

    ```py
    uvicorn app:app --reload
    ```

之前的代码使用了新的 `fastapi-cli` 包以简化开发([`fastapi.tiangolo.com/fastapi-cli/`](https://fastapi.tiangolo.com/fastapi-cli/))。`fastapi-cors` 将提供一个名为“健康检查”的新端点。如果你尝试使用它，你会看到与 CORS 相关的环境变量（`ALLOWED_CREDENTIALS`、`ALLOWED_METHODS`、`ALLOWED_ORIGINS` 等），并且现在可以通过 `.env` 文件进行设置。

FastAPI 主应用程序现在已准备就绪，但它需要两个路由器：一个用于用户和一个用于汽车，以及认证逻辑。首先，你将处理认证类以及 `users` 路由器。

### 创建用户和认证类的 APIRouter 类

认证类将封装认证逻辑，类似于*第六章* *认证和授权*中所示，并创建管理用户的配套 **APIRouter**——注册、登录和验证。

为了简化，`authentication.py` 文件将与之前使用的文件相同。位于项目根目录的 `authentication.py` 文件包含 JWT 编码和解码逻辑、密码加密和依赖注入，如*第七章* *使用 FastAPI 构建后端*所示。

我们在此提供文件内容，以方便您使用：

```py
import datetime
import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
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
            "exp": datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(minutes=30),
            "iat": datetime.datetime.now(datetime.timezone.utc),
            "sub": {"user_id": user_id, "username": username},
        }
        return jwt.encode(payload, self.secret, algorithm="HS256")
    def decode_token(self, token):
        try:
            payload = jwt.decode(token, self.secret, algorithms=["HS256"])
            return payload["sub"]
        except jwt.ExpiredSignatureError:
            raise HTTPException(
              status_code=401,
              detail="Signature has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    def auth_wrapper(self, auth: HTTPAuthorizationCredentials = Security(security)):
        return self.decode_token(auth.credentials)
```

`user.py` 路由器将被放置在 `/routers` 文件夹中，并且它将公开三个端点：用于注册新用户、用于登录用户和用于验证用户——在头部提供 `Bearer` 令牌。最后一个路由是可选的，因为在下一章（关于 Next.js）中你不会直接使用它，因为我们选择了一个简单的基于 cookie 的解决方案。

要创建用户的 API 路由器，请执行以下步骤：

1.  创建一个 `routers/user.py` 文件并填充它以创建用户的路由器。这个路由器与 Motor 版本相似，并且它共享相同的逻辑，但在以下代码中突出了某些差异：

    ```py
    from fastapi import APIRouter, Body, Depends, HTTPException
    from fastapi.responses import JSONResponse
    from authentication import AuthHandler
    from models import CurrentUser, LoginUser, RegisterUser, User
    auth_handler = AuthHandler()
    router = APIRouter()
    @router.post(
        "/register",
        response_description="Register user",
        response_model=CurrentUser
    )
    async def register(
        newUser: RegisterUser = Body(...),
        response_model=User):
        newUser.password = auth_handler.get_password_hash(
            newUser.password)
        query = {
    "$or": [{"username": newUser.username},
        	{"email": newUser.email}]}
        existing_user = await User.find_one(query)
        if existing_user is not None:
            raise HTTPException(
                status_code=409,
                detail=f"{newUser.username} or {newUser.email}
                already exists"
            )
        user = await User(**newUser.model_dump()).save()
        return user
    ```

    路由器展示了 Beanie 的一些功能：使用 MongoDB 查询直接查询 `User` 模型（`users` 集合），如果现有用户的检查通过，则简单异步创建一个新实例。在这种情况下，你有两个条件：用户名和电子邮件必须是可用的（不在集合中）。Beanie 的查询语法非常直观：[`beanie-odm.dev/tutorial/finding-documents/`](https://beanie-odm.dev/tutorial/finding-documents/)。

1.  在 `user.py` 文件中创建登录路由：

    ```py
    @router.post("/login", response_description="Login user and return token")
    async def login(loginUser: LoginUser = Body(...)) -> str:
        user = await User.find_one(
            User.username == loginUser.username
        )
        if user and auth_handler.verify_password(
            loginUser.password, user.password):
            token = auth_handler.encode_token(
              str(user.id),
              user.username
              )
            response = JSONResponse(
                content={
                    "token": token,
                    "username": user.username})
            return response
        else:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password")
    ```

    登录功能使用 Beanie 中可用的 `find_one` MongoDB 方法。

1.  最后，添加 `/me` 路由，用于验证已登录用户。此方法使用 `get` 方法，它接受一个 `ObjectId`：

    ```py
    @router.get(
        "/me", response_description="Logged in user data", response_model=CurrentUser
    )
    async def me(
        user_data=Depends(auth_handler.auth_wrapper)
    ):
        currentUser = await User.get(user_data["user_id"])
        return currentUser
    ```

这完成了 `users.py` APIRouter，它使用了几个 Beanie 查询方法。现在，你将使用 Beanie ODM 创建 `Car` 路由器。

### Car APIRouter

与前几章所完成的内容类似，`Cars`路由器将负责执行一些 CRUD 操作。为了简单起见，您将只实现汽车实例的部分更新：您将能够更新在`UpdateCar`模型中定义的字段。由于描述和优缺点列表最初为空，它们需要能够在以后更新（通过调用 OpenAI 的 API）。

要创建`Cars`路由器，在`/routers`文件夹和`cars.py`文件中，执行以下步骤：

1.  首先，创建一个`/routers/cars.py`文件并列出初始导入（在开始实现后台任务时将添加更多）：

    ```py
    from typing import List
    import cloudinary
    from beanie import PydanticObjectId, WriteRules
    from cloudinary import uploader  # noqa: F401
    from fastapi import (APIRouter, Depends, File, Form,
        HTTPException, UploadFile, status)
    from authentication import AuthHandler
    from config import BaseConfig
    from models import Car, UpdateCar, User
    ```

    这些导入与直接使用 Motor 时使用的导入类似；主要区别是 Beanie 的导入：`PydanticObjectId`（用于处理 Pydantic 的 ObjectIds）和`WriteRules`，这将使`Car`和`User`的关系能够作为引用写入 MongoDB 数据库。

1.  继续处理文件，现在您可以实例化认证处理器(`auth_handler`)类、设置和路由器，以及 Cloudinary 配置：

    ```py
    auth_handler = AuthHandler()
    settings = BaseConfig()
    cloudinary.config(
        cloud_name=settings.CLOUDINARY_CLOUD_NAME,
        api_key=settings.CLOUDINARY_API_KEY,
        api_secret=settings.CLOUDINARY_SECRET_KEY,
    )
    router = APIRouter()
    ```

1.  在完成必要的设置和认证后，您可以创建第一条路由——`GET`处理器，在这种情况下，它只是简单地检索数据库中的所有汽车：

    ```py
    @router.get("/", response_model=List[Car])
    async def get_cars():
        return await find_all() Beanie method is asynchronous, like all Beanie methods, and it simply returns all the documents in the database. Other querying methods are .find(search query) and .first_or_none(), which are often used to check for the existence of a certain condition (such as a user with a given username or email). Finally, the to_list() method, like with Motor, returns a list of documents, but you could also use the async for construct (shown in *Chapter 4*, *Getting Started with FastAPI*) and generate a list that way.
    ```

1.  创建用于通过其 ID 获取单个汽车实例的`GET`方法：

    ```py
    @router.get("/{car_id}", response_model=Car)
    async def get_car(car_id: PydanticObjectId):
        car = await Car.get(car_id)
        if not car:
            raise HTTPException(status_code=404, detail="Car not found")
        return car
    ```

    此实现也很简单——它使用`get()`快捷方式通过`ObjectId`查询集合，这由 Beanie 优雅地处理。

1.  创建新汽车实例的方法稍微复杂一些，但并不太重。由于您正在上传图像（一个文件），您使用表单数据而不是 JSON，并且端点必须将图像上传到 Cloudinary，从 Cloudinary 获取图像 URL，然后才能将图像与其他数据一起插入 MongoDB 数据库：

    ```py
    @router.post(
        "/",
        response_description="Add new car with picture",
        response_model=Car,
        status_code=status.HTTP_201_CREATED,
    )
    async def add_car_with_picture(
        brand: str = Form("brand"),
        make: str = Form("make"),
        year: int = Form("year"),
        cm3: int = Form("cm3"),
        km: int = Form("km"),
        price: int = Form("price"),
        picture: UploadFile = File("picture"),
        user_data=Depends(auth_handler.auth_wrapper),
    ):
        cloudinary_image = cloudinary.uploader.upload(
          picture.file,
          folder="FARM2",
          crop="fill",
          width=800,
          gravity="auto" )
        picture_url = cloudinary_image["url"]
        user = await User.get(user_data["user_id"])
        car = Car(
            brand=brand,
            make=make,
            year=year,
            cm3=cm3,
            km=km,
            price=price,
            picture_url=picture_url,
            user=user,
        )
        return await car.insert(link_rule=WriteRules.WRITE)
    ```

    创建新资源的路由使用 Beanie 方法通过 ID（在请求头中的`Bearer`令牌中提供）获取用户，并使用`insert()`方法插入新汽车。

    最后，`link_rule`允许您保存销售人员的 ID（[`beanie-odm.dev/tutorial/relations/`](https://beanie-odm.dev/tutorial/relations/)）。

1.  `update`方法与 Motor 的对应方法类似，可以轻松地集成到仪表板中，用于更新或删除汽车型号广告：

    ```py
    @router.put("/{car_id}", response_model=Car)
    async def update_car(
        car_id: PydanticObjectId,
        cardata: UpdateCar):
        car = await Car.get(car_id)
        if not car:
            raise HTTPException(
                status_code=404,
                detail="Car not found")
        updated_car = {
            k: v for k, v in cardata.model_dump().items()   if v is not None}
        return await car.set(updated_car)
    ```

    再次强调，您只更新请求中提供的字段，使用 Pydantic 的`model_dump`方法来验证哪些字段实际上被提供，其他字段（在 Python 术语中为`null`或`None`）保持不变。

1.  在`delete`方法中，您只需要提供所选文档并调用`delete()`方法：

    ```py
    @router.delete("/{car_id}")
    async def delete_car(car_id: PydanticObjectId):
        car = await Car.get(car_id)
        if not car:
            raise HTTPException(status_code=404, detail="Car not found")
        await car.delete()
    ```

您现在已经完成了您的 API 路由器，并准备好实现一些更高级的功能，FastAPI 和 FARM 栈通常使这项任务变得快速且有趣。然而，在使用路由器之前，您需要将它们导入到 `app.py` 文件中。打开 `app.py` 文件并修改顶部的导入，添加路由器并将它们别名为 cars 和 users：

```py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from database import init_db
from routers import cars as cars_router
from routers import user as user_router
from fastapi_cors import CORS
```

最后，通过修改相同的 `app.py` 文件将这些功能集成到应用程序中：

```py
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
app = FastAPI(lifespan=lifespan)
CORS(app)
app.include_router(
    cars_router.router,
    prefix="/cars",
    tags=["Cars"]
)
app.include_router(
    user_router.router,
    prefix="/users",
    tags=["Users"]
)
@app.get("/", tags=["Root"])
async def read_root() -> dict:
    return {"message": "Welcome to your beanie powered app!"}
```

连接好路由器后，您将集成一个简单但实用的 AI 助手，该助手将提供有关新插入的汽车的市场信息，并自动向销售人员、客户列表或订阅者群体发送电子邮件。

## FastAPI 的后台任务

FastAPI 最有趣的功能之一是它如何处理后台任务——这些任务应该在向客户端发送响应之后异步运行。

后台任务有许多用例。任何可能需要一些时间的操作，例如等待外部 API 调用返回响应、发送电子邮件或基于端点的数据处理创建复杂文档，都是后台任务的潜在候选者。在这些所有情况下，仅仅让应用程序挂起等待结果是不良的做法，会导致糟糕的用户体验。相反，这些任务被交给后台处理，而响应则立即返回。

虽然对于简单任务非常有用，但不应将后台任务用于需要大量处理能力或/和多任务处理的进程。在这种情况下，一个更健壮的工具，如 **Celery** ([`docs.celeryq.dev/`](https://docs.celeryq.dev/)) 可能是最佳解决方案。Celery 是一个 Python 任务队列框架，可以在线程或不同的机器之间分配工作。

FastAPI 定义了一个名为 `BackgroundTasks` 的类，它继承自 **Starlette** 网络框架，它简单直观，您将在以下部分使用它将外部服务连接到您的 FastAPI 应用程序时看到。

在使用后台任务与第三方服务接口之前，创建一个非常简单的任务以供演示：

1.  在项目的根目录下创建一个名为 `background.py` 的文件，并填充以下代码：

    ```py
    from time import sleep
    def delayed_task(username: str) -> None:
        sleep(5)
        print(
            f"User just logged in: {username}"
        )
    ```

    此函数非常简单——它将在控制台上打印一条消息，等待五秒钟。

    将任务集成到端点的语法将在以下 API 路由器中展示。

1.  打开 `/routers/user.py` 文件，因为您将把这个简单的后台任务附加到 `login` 函数上。

    此函数还可以执行一些日志记录或一些更复杂且耗时的操作，这些操作会阻塞响应直到完成，但在此情况下，将使用一个简单的 `print` 语句。

1.  在文件顶部导入后台任务，并仅以以下方式修改 `login` 端点：

    ```py
    from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
    from background import delayed_task
    # code continues …
    @router.post("/login", response_description="Login user and return token")
    async def login(
        background_tasks: BackgroundTasks,
        loginUser: LoginUser = Body(...)
    ) -> str:
        user = await User.find_one(
            User.username == loginUser.username
        )
        if user and auth_handler.verify_password(
            loginUser.password, user.password
        ):
            token = auth_handler.encode_token(
                str(user.id), user.username
            )
            background_tasks.add_task(
                delayed_task,
                username=user.username
            )
            response = JSONResponse(
                content={
                    "token": token,
                    "username": user.username
                    }
                )
            return response
        else:
            raise HTTPException(
                    status_code=401,
                    detail="Invalid username or password"
        )
    fastapi dev
    ```

    你可以导航到交互式文档的地址（127.0.0.1:8000/docs）并尝试登录。

1.  如果你已经安装了 HTTPie，你可以让一个终端以开发模式运行 FastAPI 应用，打开另一个终端，并发出登录 POST 请求，确保使用你之前创建的用户正确的用户名和密码。例如，以下命令测试了用户 `tanja` 的登录：

    ```py
    http POST 127.0.0.1:8000/users/login username=tanja password=tanja123
    ```

    如果你查看第一个终端，五秒后你会看到以下信息：

    ```py
    User just logged in: tanja
    ```

你刚刚创建了一个简单但可能很有用的后台任务，并学习了语法。

在下一节中，你将创建两个后台任务，使用 OpenAI 的 API 创建新的汽车描述，并将描述和汽车数据通过电子邮件发送给已登录的用户——即插入汽车的用户。

## 将 OpenAI 集成到 FastAPI 中

在过去几年中，LLM（大型语言模型）一直是热门词汇，它们主导着网络开发的讨论，而且越来越难以找到不使用某种形式的 LLM 集成的成功应用。现代应用利用图像、文本和音频处理，这可能会给你的下一个网络应用带来优势。

在你的汽车销售和广告应用中，你将使用 OpenAI 这样的巨无霸的一个最简单功能——当前的任务是让销售人员的工作变得更容易一些，并为每辆即将上市的新车提供一条基准营销线：

1.  在获取了 OpenAI 密钥并设置环境变量后，修改 `background.py` 文件：

    ```py
    import json
    from openai import OpenAI
    from config import BaseConfig
    from models import Car
    settings = BaseConfig()
    json for decoding the OpenAI response, the openai module, as well as the config module for reading the API keys. After instantiating the settings and the OpenAI client, you will create a helper function that will generate the prompt for OpenAI.Although these tasks are handled much more elegantly with a library called LangChain—the de facto standard when working with LLMs in Python—for simplicity’s sake, you will use a simple Python `f-string` to regenerate the prompt on each request.Remember, the prompt needs to provide a text description and two arrays—one for the positive aspects and one for the negative aspects of the car.
    ```

注意

你可以轻松地将 OpenAI 替换为另一个 LLM，例如 **Google Gemini**。

1.  以下是一种创建用于生成汽车数据的提示的方法，但根据你的情况，你可能希望更加有创意或保守地使用 OpenAI 提供的描述：

    ```py
    def generate_prompt(brand: str, model: str, year: int) -> str:
        return f"""
        You are a helpful car sales assistant. Describe the {brand} {model} from {year} in a playful manner.
        Also, provide five pros and five cons of the model, but formulate the cons in a not overly negative way.
        You will respond with a JSON format consisting of the following:
        a brief description of the {brand} {model}, playful and positive, but not over the top.
        This will be called *description*. Make it at least 350 characters.
        an array of 5 brief *pros* of the car model, short and concise, maximum 12 words, slightly positive and playful
        an array of 5 brief *cons* drawbacks of the car model, short and concise, maximum 12 words, not too negative, but in a slightly negative tone
        make the *pros* sound very positive and the *cons* sound negative, but not too much
        """
    ```

1.  现在提示已经准备好生成，是时候调用 OpenAI API 了。请始终参考最新的 OpenAI API 文档（[`platform.openai.com/docs/overview`](https://platform.openai.com/docs/overview)），因为它经常发生变化。目前，在撰写本文时，以下代码演示了与 API 通信的方式，你应该将其粘贴到你的 `background.py` 文件中：

    ```py
    async def create_description(
        brand,
        make,
        year,
        picture_url):
        prompt = generate_prompt(brand, make, year)
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2,
            )
            content = response.choices[0].message.content
            car_info = json.loads(content)
            await Car.find(
                Car.brand == brand,
                Car.make == make,
                Car.year == year
            ).set(
                {
                    "description": car_info["description"],
                    "pros": car_info["pros"],
                    "cons": car_info["cons"],
                }
            )
        except Exception as e:
            print(e)
    ```

    上述代码通过聊天完成方法调用 OpenAI 客户端。你已经选择了一个模型（`gpt-4`），启动了 `messages` 数组，并设置了 `max_tokens` 和 `temperature`。再次提醒，对于所有参数设置，请参考最新的 OpenAI 文档。在这种情况下，你将令牌数量限制为 500，并将温度设置为 `0.2`（这个数量影响响应的“创意”和“保守性”）。

    在收到 OpenAI 的响应后，你将 JSON 内容（`car_info`）解析为包含所需键的 Python 字典：描述（文本）和两个字符串数组（优点和缺点）。有了这些新生成数据，你通过 Beanie 执行 MongoDB 更新，选择所有匹配品牌、型号和生产年份的汽车，并将它们的描述、优点和缺点设置为 OpenAI 返回的数据。如果发生错误，我们简单地显示错误。

1.  现在将后台任务连接到`POST`端点。打开`/routers/cars.py`文件，并在顶部导入新创建的后台函数：

    ```py
    from background import create_description
    ```

1.  其余的代码将保持不变；你只修改`POST`端点：

    ```py
    @router.post(
        "/",
        response_description="Add new car with picture",
        response_model=Car,
        status_code=status.HTTP_201_CREATED,
    )
    async def add_car_with_picture(
        background_tasks: BackgroundTasks,
        brand: str = Form("brand"),
        make: str = Form("make"),
        year: int = Form("year"),
        cm3: int = Form("cm3"),
        km: int = Form("km"),
        price: int = Form("price"),
        picture: UploadFile = File("picture"),
        user_data=Depends(auth_handler.auth_wrapper),
    ):
        cloudinary_image = cloudinary.uploader.upload(
          picture.file,
          folder="FARM2",
          crop="fill",
          width=800,
          height=600,
          gravity="auto"
        )
        picture_url = cloudinary_image["url"]
        user = await User.get(user_data["user_id"])
        car = Car(
            brand=brand,
            make=make,
            year=year,
            cm3=cm3,
            km=km,
            price=price,
            picture_url=picture_url,
            user=user,
        )
        background_tasks.add_task(
            create_description, brand=brand, make=make,
            year=year, picture_url=picture_url
        )
        return await car.insert(link_rule=WriteRules.WRITE)
    ```

这可以通过更细粒度的方式进行：你可以等待新插入的汽车生成的 ID，并仅更新那个特定实例。该函数还缺少一些基本验证，用于处理提供的汽车品牌和型号不存在的情况，或者 OpenAI 没有提供有效响应的情况。关键是端点函数立即返回响应——也就是说，在执行 MongoDB 插入后几乎立即，描述和两个数组稍后更新。

如果你尝试重新运行开发服务器并插入一辆汽车，你应该会看到新创建的文档（在 Compass 或 Atlas 中），几秒钟后，文档将更新为最初为空的字段：`description`、`pros`和`cons`。

你可以想象出许多可能被这个功能覆盖的场景：可能需要由人类审核汽车描述，然后设置广告发布（通过添加已发布的布尔变量），可能你想向所有注册用户发送电子邮件，等等。

下一节将进一步介绍这个后台工作，并展示你如何快速将电子邮件集成到你的应用程序中。

### 将电子邮件集成到 FastAPI 中

现代网络应用最常见的需求之一是发送自动化的电子邮件。今天，有众多发送电子邮件的选项，其中最受欢迎的两个选项是 Twilio 的**Mailgun**和**SendGrid**。

通过这个应用程序，你将学习如何使用名为**Resend**的相对较新的服务设置电子邮件功能。他们的以 API 为中心的方法非常适合开发者，并且易于上手。

导航到 Resend 主页([`resend.com`](https://resend.com))并创建一个免费账户。登录后，导航到`FARMstack`。密钥只会显示一次，所以请确保复制并存储在`.``env`文件中。

执行以下步骤以将 Resend 功能添加到你的应用程序中：

1.  安装`resend`包：

    ```py
    pip install resend==2.0.0
    ```

1.  安装`resend`包后，更新`background.py`文件：

    ```py
    import json
    import resend
    from openai import OpenAI
    from config import BaseConfig
    from models import Car
    settings = BaseConfig()
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resend.api_key = settings.RESEND_API_KEY
    # code continues …
    ```

1.  更新`create_description`函数，以便在从 OpenAI 返回响应后发送消息：

    ```py
    async def create_description(brand, make, year, picture_url):
        prompt = generate_prompt(brand, make, year)
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2,
            )
            content = response.choices[0].message.content
            car_info = json.loads(content)
            await Car.find(
                Car.brand == brand,
                Car.make == make,
                 Car.year == year).set(
                {
                    "description": car_info["description"],
                    "pros": car_info["pros"],
                    "cons": car_info["cons"],
                }
            )
            def generate_email():
                pros_list = "<br>".join([f"- {pro}" for pro in car_info["pros"]])
                cons_list = "<br>".join([f"- {con}" for con in car_info["cons"]])
                return f"""
                Hello,
                We have a new car for you: {brand} {make} from {year}.
                <p><img src="img/{picture_url}"/></p>
                {car_info['description']}
                <h3>Pros</h3>
                {pros_list}
                <h3>Cons</h3>
                {cons_list}
                """
            params: resend.Emails.SendParams = {
                "from":"FARM Cars <onboarding@resend.dev>",
                "to": ["youremail@gmail.com"],
                "subject": "New Car On Sale!",
                "html": generate_email(),
            }
            resend.Emails.send(params)
        except Exception as e:
            print(e)
    ```

收件人电子邮件地址应该是您在 Resend 上注册的同一电子邮件地址，因为这将是您注册和验证域名之前的唯一选项，但对于开发和测试目的来说已经足够：[`resend.com/docs/knowledge-base/`](https://resend.com/docs/knowledge-base/)。

`resend`包使发送邮件变得简单——您只需调用一次`resend.Emails.Send`函数并定义参数。在您的案例中，参数如下：

+   `to` – 收件人电子邮件列表。

+   `from` – 发件人的电子邮件地址。在这种情况下，您将保留 Resend 提供的默认地址，但稍后您将用您自己的域名地址替换它。

+   `subject` – 邮件的主题。

+   `html` – 邮件的 HTML 内容。

参数以字典的形式传递给`resend.Email.send()`函数。

该应用程序中的邮件 HTML 内容直接由 Python 中的`f-string`构建，但您始终可以求助于更复杂和高级的解决方案，例如使用**Jinja2**（对于纯 Python 解决方案，因为后端是用 Python 编写的）或使用 Resend 的 React Email（[`react.email/`](https://react.email/)）。Jinja2 可以说是最流行的 Python HTML 模板引擎，它被 Flask Web 框架所使用，而 React Email 提供基于 React 的邮件模板。

注意

请参阅*第七章*，*使用 FastAPI 构建后端*，了解如何将您的后端部署到 Render.com。程序将基本保持不变：只需跟踪环境变量，并确保添加新创建的变量（OpenAI 和 Render 密钥）。或者，您可以从本章运行后端，以便在下一章中使用它。

# 摘要

在本章中，您学习了 Beanie 的基础知识，这是一个基于 Motor 和 Pydantic 的流行 ODM 库，用于 MongoDB。您学习了如何定义模型和定义与 MongoDB 集合映射的 Beanie 文档，以及如何使用 ODM 进行查询和执行 CRUD 操作。

您构建了另一个 FastAPI 应用程序，在该应用程序中，您通过后台任务集成了第三方服务，这是 FastAPI 的一个功能，允许在后台执行慢速和长时间运行的任务，同时保持应用程序的响应性。

本章还介绍了如何将最受欢迎的 AI 服务 ChatGPT 集成到您的应用程序中，为您的最新插入实体提供智能附加数据。最后，您学习了如何实现一个简单的邮件发送解决方案，这在许多 Web 应用程序中很常见。

在下一章中，您将深入了解基于 React.js 的最受欢迎和最先进的 Web 框架：**Next.js**。您将学习 Next.js 最新版本（14）的基础知识，并发现使其与其他前端甚至全栈解决方案区别开来的最重要的特性。
