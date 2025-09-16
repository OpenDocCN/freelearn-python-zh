

# 第六章：认证和授权

认证的概念——证明用户是他们所声称的人——以及授权——确保经过认证的用户应该或不应该能够对你的 API 执行某些操作——都是非常复杂的。在本章中，您将从非常实用的角度以及 FARM 堆栈的角度来探讨认证和授权的主题。

本章将详细说明一个简单、健壮且可扩展的 FastAPI 后端设置，基于**JSON Web Token**（JWT）——可以说是近年来出现的最受欢迎和实用的认证方法。然后，您将看到如何将基于 JWT 的认证方法集成到 React 中，利用 React 的一些强大功能——即 Hooks、Context 和 React Router。

到本章结束时，您应该对 FastAPI 在后端和 React 在前端提供的认证方法有一个牢固的掌握，您将能够以细粒度和精确度对用户进行认证并控制他们在应用中的操作。

本章将涵盖以下主题：

+   用户模型及其与其他资源的关系

+   JWT 认证机制——整体概述

+   FastAPI 中的认证和授权工具

+   如何保护路由、路由器或整个应用

+   React 认证的各种解决方案

# 技术要求

要运行本章中的示例应用，您应该具备以下条件：

+   Node.js 版本 18 或更高

+   Python 3.11.7 或更高版本

要求与前面章节中的要求相同，您将安装的新包将按其使用情况进行描述。

# 理解 JSON Web Token

HTTP 是一种无状态协议，仅此一点就暗示了几个重要的后果。其中之一是，如果您想在请求之间持久化某种状态，您必须求助于一种能够记住一组数据（例如，登录的用户是谁，在之前的浏览器会话期间选择了哪些项目，或者网站偏好设置是什么）的机制。为了实现这种功能并识别当前用户，作为开发人员，您有众多选项可供选择。以下是一些最受欢迎和最现代的解决方案：

+   **基于凭证的认证**：它要求用户输入个人凭证，如用户名或电子邮件，以及密码

+   **无密码登录**：用户在创建账户后，通过电子邮件或其他通信渠道接收一个安全、时间有限的令牌进行认证，而不是使用传统的密码。安全的令牌用于会话认证，消除了输入或记住密码的需要。

+   **生物识别密码**：它利用用户的生物特征，如指纹，进行认证。

+   **社交认证**：用户利用他们现有的社交媒体账户（例如，Google、Facebook 或 LinkedIn）进行认证。这将用户的社交媒体账户与平台上的账户关联起来。

+   **经典个人凭证方法**：用户在注册时提供电子邮件并选择密码。用户还可以选择用户名作为可选项。

本章将考虑经典的个人凭证方法。当用户注册时，他们提供电子邮件并选择密码，以及可选的用户名。

## JWT 是什么？

虽然有不同方式在应用程序的不同部分之间维护用户的身份，但 JWT 可以说是连接前端应用程序（如 React、Vue.js 和 Angular）或移动应用程序与 API（在我们的案例中，是 REST API）最常见和最受欢迎的方法。JWT 仅仅是一个标准，一种结构化由看似随机的字符和数字组成的大字符串的方式，以安全的方式封装用户数据。

JWT 包含三个部分——**头部**、**负载**和**签名**。头部包含有关令牌本身的元数据——用于签名令牌的算法和令牌的类型。

负载数据是最有趣的部分。它包含以下必要的认证信息：

+   数据（声明）：用户的 ID（或用户名）

+   **签发时间字段**（**iat**）：签发令牌的日期和时间

+   令牌失效的时间：与令牌的有效期相关联

+   可选的其他字段：例如，用户名、角色等。

负载数据可以被每个人解码和阅读。您可以阅读更多关于令牌的信息，了解它们在 JWT 文档中的样子：[`jwt.io`](https://jwt.io).

最后，令牌最重要的部分是签名。签名保证了令牌所声明的信息。签名被重新生成（计算）并与原始签名进行比较——从而防止声明被修改。

例如，考虑一个声明用户名为 `John` 的 JWT。现在，如果有人试图将其更改为 `Rita`，他们还需要修改签名以匹配。然而，修改签名将使令牌无效。这种机制确保令牌的内容保持不变且真实。

因此，令牌可以完全取代认证数据——用户或电子邮件和密码组合不需要多次传输。

在接下来的章节中，您将学习如何在您的应用程序中实现基于 JWT 的认证流程。

# 带有用户和依赖项的 FastAPI 后端

如果应用程序不安全，那么网络应用程序（或移动应用程序）将没有太大用处。您可能已经听说过在认证实现中出现的微小错误，这些错误可能导致数十万甚至数百万个账户被破坏，可能暴露敏感和有价值的信息。

FastAPI 基于 OpenAPI——之前被称为 `apiKey`、`http`、`OAuth 2.0`、`openIdConnect` 等）。虽然 FastAPI 文档网站 ([`fastapi.tiangolo.com/tutorial/security/`](https://fastapi.tiangolo.com/tutorial/security/)) 提供了一个优秀且详细的教程，用于创建身份验证流程，但它基于 `OAuth 2.0` 协议，该协议使用表单数据发送凭证（用户名和密码）。

在以下章节中，你将设计一个简单的用户模型，这将使身份验证流程成为可能。然后，你将学习如何将用户数据编码成 JWT，以及如何使用令牌来访问受保护的路线。

## 身份验证的用户模型

每个身份验证流程的基础是用户模型，它必须能够存储一组最小数据，以便明确地识别用户。最常见的唯一字段是一个电子邮件地址、一个用户名，当然，还有一个主键——在 MongoDB 的情况下是一个 `ObjectId` 实例。

使用 MongoDB 模型数据与在 *第二章*，*使用 MongoDB 设置数据库* 中讨论的建模关系型数据库本质上是不同的。驱动思想是提前考虑查询，并考虑你的应用程序将要最频繁执行的查询来建模你的关系。

## 使用 FastAPI 进行身份验证和授权：教程

通过示例，使用 FastAPI 进行身份验证和授权更容易理解。在接下来的几个子章节中，你将开发一个简单但功能齐全的身份验证系统，它将包含所有必需的步骤。为了突出重要部分，同时尽可能使示例简洁，你将不会使用真实的 MongoDB 连接。相反，你将创建自己的基于 JSON 文件的基础 **数据库**，该数据库将存储用户在应用程序中注册时的数据，并有效地模拟 MongoDB 集合。首要步骤是审查你的身份验证系统。

### 审查你的身份验证系统的所有部分

以下列表提供了一个快速回顾，列出了实现 FastAPI 身份验证工作流程所需的工具和包：

+   要实现 FastAPI 的身份验证工作流程，你必须使用 FastAPI 的安全工具。在 FastAPI 中，当你需要使用 `Security()` 类声明依赖项时。其他你将需要的 FastAPI 导入是可信赖的类型——在这种情况下，你将使用 **bearer** 令牌进行授权。你可以参考 FastAPI 文档：[`fastapi.tiangolo.com/reference/security/#fastapi.security.HTTPBearer`](https://fastapi.tiangolo.com/reference/security/#fastapi.security.HTTPBearer)。

+   您还需要密码散列和比较功能，这 `passlib` 可以提供。`passlib.context` 模块包含一个主要类：`passlib.context.CryptContext`，设计用于处理与散列和比较字符串相关的许多更常见的编码任务。您的身份验证系统需要两个主要功能：在用户注册期间散列密码，以及在登录期间将散列密码与存储在您的数据库中的散列密码进行比较。

+   最后，**PyJWT** 将提供编码和解码 JWT 的功能。

### 创建模型

下一步涉及在新的虚拟环境中创建基本的 FastAPI 应用程序，激活环境，安装必要的软件包，并创建具有所需字段的合适用户模型：

1.  创建一个新的目录，使用 `cd`（更改目录）命令将其设置为工作目录，在 `/venv` 中创建一个新的 Python 环境，并激活它：

    ```py
    mkdir chapter6
    cd chapter6
    python -m venv venv
    source ./venv/bin/activate
    ```

1.  一旦新的 Python 环境激活，安装身份验证系统和应用程序所需的软件包：

    ```py
    pip install fastapi uvicorn bcrypt==4.0.1 passlib pyjwt
    ```

注意

如果您想能够精确地重现书中的代码，强烈建议您使用附带存储库中的 `/backend/requirements.txt` 文件，并使用 `pip install -r requirements.txt` 命令安装软件包。

以下是为您的身份验证系统所需的最后三个软件包：

+   `Passlib` 是一个用于 Python 的密码散列库，它支持广泛的散列算法，包括 `bcrypt`。它非常有用，因为它提供了一个统一的接口用于散列和验证密码。

+   `bcrypt` 软件包是一个 Python 模块，它提供了一个基于 Blowfish 密码散列算法的密码散列方法，您将使用此方法。请坚持使用提供的软件包版本，因为后续版本存在一些未解决的问题。

+   `PyJWT` 是用于编码和解码 JWT 的 Python 库。

1.  接下来，创建应用程序的模型。由于此应用程序将仅处理用户，因此 `models.py` 文件相当简单：

    ```py
    from pydantic import BaseModel, Field
    from typing import List
    class UserBase(BaseModel):
        id: str = Field(...)
        username: str = Field(
            ..., 
            min_length=3, 
            max_length=15)
        password: str = Field(...)
    class UserIn(BaseModel):
        username: str = Field(
            ..., 
            min_length=3,
            max_length=15)
        password: str = Field(...)
    class UserOut(BaseModel):
        id: str = Field(...)
        username: str = Field(
            ..., 
            min_length=3, 
            max_length=15)
    class UsersList(BaseModel):
        users: List[UserOut]
    ```

模型是自解释的，并且被留得尽可能明确。`UserBase` 对应于将存储在您的虚拟数据库中或 MongoDB 集合中的用户表示（请特别注意 `Object_id`）。在给定解决方案中，`id` 字段将是一个 UUID，因此您将其设置为字符串类型。

注意

为此演示目的的 Python `ObjectId()` 类。

`models.py` 文件包含两个额外的辅助 Pydantic 模型：`UserIn`，它接受用于注册或登录的用户数据（通常是用户名和密码，但可以轻松扩展以包括电子邮件或其他数据），以及 `UserOut`，它负责在应用程序中表示用户，不包括散列密码但包括 ID 和用户名。

`UsersList` 最终只是输出所有用户的列表，你将使用这个模型作为受保护路由的示例。现在，构建你的 `app.py` 文件并创建实际的应用程序。

### 创建应用程序文件

在定义了模型之后，你现在可以继续创建主要的 FastAPI 应用程序和认证类：

1.  打开一个新的 Python 文件，命名为 `app.py`。在这个文件中，创建一个最小的 FastAPI 应用程序：

    ```py
    from fastapi import FastAPI
    app = FastAPI()
    ```

    我们很快就会回到这个文件，但现在，让我们尽量让它尽可能短。现在，是时候构建你认证系统的核心了。

1.  在同一个文件夹中，创建 `authentication.py` 文件并开始构建它。有了所有这些，打开新创建的 `authentication.py` 文件，开始构建认证类。为此，你必须首先构建 `AuthHandler` 类并添加所需的导入：

    ```py
    import datetime
    import jwt
    from fastapi import HTTPException, Security
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from passlib.context import CryptContext
    class AuthHandler:
        security = HTTPBearer()
        pwd_context = CryptContext(schemes=[“bcrypt”], deprecated=”auto”)
        secret = “FARMSTACKsecretString”
    ```

现在你已经了解了所有这些导入，你可以创建一个名为 `AuthHandler` 的类，该类使用 FastAPI 的 `HTTPBearer` 作为安全依赖，并从 `passlib` 定义密码处理上下文。

### 添加安全依赖和密码处理上下文

这个过程包括多个步骤。你需要添加一个秘密字符串，理想情况下应该是随机生成的，并安全地存储在环境变量中，远离任何 `git commit`。这个秘密字符串对于哈希密码是必要的。在这里，为了简单起见，你将在这个文件中硬编码它。

因此，继续使用同一个文件，逐步编写所需的功能，如下所示：

1.  `authentication.py` 文件在 `AuthHandler` 类下：

    ```py
    def get_password_hash(self, password: str) -> str:
            return self.pwd_context.hash(password)
    ```

    这个函数只是创建给定密码的哈希值，这个结果就是你将在数据库中存储的内容。它很好地使用了你之前定义的 `passlib` 上下文。

1.  `authentication.py` 文件：

    ```py
        def verify_password(
            self,
            plain_password: str, 
            hashed_password: str) -> bool:
            return self.pwd_context.verify(
               plain_password, 
                hashed_password)
    ```

    与前面的函数类似，`verify_password` 简单地验证 `plain_password` 的哈希值是否确实等于（已经）哈希过的密码，并返回 `True` 或 `False`。

1.  `authentication.py` 文件：

    ```py
    def encode_token(self, user_id: int, username: str) -> str:
    payload = {
                “exp”: datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(minutes=30),
                “iat”: datetime.datetime.now(datetime.timezone.utc),
                “sub”: {“user_id”: user_id, “username”: username},
            }
            return jwt.encode(payload, self.secret, algorithm=”HS256”)
    ```

    你的类的 `encode_token` 方法利用 `PyJWT` 包的 `encode` 方法来创建 JWT 本身，它非常明确；有效载荷包含过期时间（非常重要——你不想 JWT 持续太久）和 *发行时间*（`iat` 部分）。它还引用了名为 `sub` 的字典，其中包含你希望编码的所有数据——在这种情况下，是用户 ID 和用户名，尽管你也可以添加角色（普通用户、管理员等）或其他数据。总结一下，JWT 编码了三份数据：

    +   过期持续时间，在这个例子中，30 分钟。

    +   令牌的发行时间，在这个例子中，设置为 `now()`。

    +   `sub` 部分是你想要包含在令牌中的数据（以字典的形式）。在这个例子中，它是用户 ID 和用户名。

1.  **解码** **令牌**

    继续构建类，因为现在需要反向功能——解码令牌的方法：

    ```py
    def decode_token(self, token: str):
        try:
            payload = jwt.decode(
                token, 
                self.secret,
                algorithms=[“HS256”])
            return payload[“sub”]
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401, 
                detail=”Signature has expired”)
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=401, 
                detail=”Invalid token”)
    Defining the dependencyFinalize your class with the dependency to be injected in the routes that will need protection:

    ```

    def auth_wrapper(

        self,

        auth: HTTPAuthorizationCredentials = Security(security)) -> dict:

        return self.decode_token(auth.credentials)

    ```py

    ```

您将使用此`auth_wrapper`作为依赖项——它将检查请求头中是否存在有效的 JWT 作为承载令牌，用于所有需要授权的路由或整个路由器。

`authorization.py` 文件是对认证/授权流程的最小实现。

在前面的步骤中，您将大多数认证和授权功能封装到一个简单紧凑的类中。创建令牌、编码和解码、以及密码哈希和验证。最后，您创建了一个简单的依赖项，用于验证用户并启用或禁用对受保护路由的访问。

构建应用程序的 FastAPI 路由器将与您在*第二章*中构建的类似，即使用 MongoDB 设置数据库。您将有两个基本端点用于注册和登录，它们将严重依赖于`AuthHandler`类。

### 创建用户的 APIRouter

在本节中，您将创建用于用户的 APIRouter，并在认证类和用字典和 UUID 实现的模拟数据库服务的帮助下实现登录和注册功能。为了实现此功能，请执行以下步骤：

1.  在应用程序的根目录下创建一个名为`routers`的文件夹，并在其中创建一个名为`users.py`的文件。将以下代码添加到`users.py`文件中：

    ```py
    import json
    import uuid
    from fastapi import APIRouter, Body, Depends, HTTPException, Request
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse
    from authentication import AuthHandler
    from models import UserBase, UserIn, UserOut, UsersList
    ```

1.  在文件开头添加导入之后，创建 APIRouter 和注册端点。注册函数使用模拟的 JSON 数据库存储用户名和哈希密码，通过使用您之前创建的`authentication.py`文件。

    ```py
    router = APIRouter()
    auth_handler = AuthHandler()
    @router.post(“/register”, response_description=”Register user”)
    async def register(request: Request, newUser: UserIn = Body(...)) -> UserBase:
        users = json.loads(open(“users.json”).read())[“users”]
        newUser.password = auth_handler.get_password_hash(newUser.password)
        if any(user[“username”] == newUser.username for user in users):
            raise HTTPException(status_code=409, detail=”Username already taken”)
        newUser = jsonable_encoder(newUser)
        newUser[“id”] = str(uuid. uuid4())
        users.append(newUser)
        with open(“users.json”, “w”) as f:
            json.dump({“users”: users}, f, indent=4)
        return newUser
    ```

    为了演示基于 JWT 的基本认证和授权系统，使用了模拟数据存储解决方案。您不是通过驱动程序连接到 MongoDB 集群，而是使用简单的 JSON 文件来存储用户及其哈希密码——这是一个类似于用于测试和脚手架目的的流行 JSON Server Node 包的解决方案。然而，所有展示的功能和逻辑都适用于真实数据库场景，并且很容易适应 MongoDB 驱动程序或 ODM，如 PyMongo、Motor 或 Beanie。

    在导入之后，包括一些您在处理真实 MongoDB 数据库时可能不需要的包，例如`uuid`，您已经实例化了 APIRouter 和自定义的`AuthHandler`类。

    `/register` 端点接受新用户的数据并在`models.py`文件中定义的`UserIn` Pydantic 类中对其进行塑形，而输出设置为`UserBase`类。这可能是您想要避免的，因为它会将哈希密码发送给新注册的用户。

    代替真实的 MongoDB 数据库，你正在读取一个名为 `users.json` 的 JSON 文件的内容——这个文件将托管一个非常简单的数据结构，将模拟你的用户 MongoDB 集合：一个包含用户数据的简单字典数组——ID、用户名和哈希密码。

    现在你有了这个“数据库”，或者用户数组，很容易遍历它们并验证是否包含一个尝试注册的用户具有相同的用户名——如果是这样，你只需用温和的 `HTTP 409` 响应代码和 `Username already` `taken` 消息将其忽略。

    如果用户名未被占用，则继续使用你的 `auth_handler` 实例，并将纯文本原始密码设置为它的哈希版本，安全地存储在数据库中。

    为了能够将用户存储为 Python 字典，请使用 `jsonable_encoder` 并向其中添加一个新的键：用作新用户 ID 的 `uuid` 字符串。

    最后，将用户（以包含 ID、用户名和哈希密码的字典形式表示）添加到你的用户列表中，将修改后的列表写入 JSON 文件，并返回用户。

1.  现在，继续使用 `users.py` 路由器，你还可以通过在文件末尾添加以下代码来创建 `login` 端点：

    ```py
    @router.post(“/login”, response_description=”Login user”)
    async def login(request: Request, loginUser: UserIn = Body(...)) -> str:
        users = json.loads(open(“users.json”).read())[“users”]
        user = next(
            (user for user in users if user[“username”] == loginUser.username), None
        )
        if (user is None) or (
            not auth_handler.verify_password(loginUser.password, user[“password”])
        ):
            raise HTTPException(status_code=401, detail=”Invalid username and/or password”)
        token = auth_handler.encode_token(str(user[“id”]), user[“username”])
        response = JSONResponse(content={“token”: token})
        return response
    ```

    此代码遵循类似的逻辑：它加载用户数据并尝试通过用户名找到登录用户（类似于查找查询）。如果用户未找到或密码验证失败，端点将引发异常。不具体说明哪个部分失败，而是告知用户整个用户名和密码组合无效，被认为是一种良好的安全实践。如果两个检查都通过，你将编码令牌并将其返回给用户。

1.  是时候连接路由器了。通过替换文件内容来编辑之前创建的 `app.py` 文件：

    ```py
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from routers.users import router as users_router
    origins = [“*”]
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=[“*”],
        allow_headers=[“*”],
    )
    app.include_router(users_router, prefix=”/users”, tags=[“users”])
    ```

    这里，你添加了 `users` 路由器。

1.  现在，在你的项目根目录下创建一个名为 `users.json` 的文件，并用一个空的 `users` 数组填充它：

    ```py
    {
        users:[]
    }
    ```

1.  保存文件并从 shell 中启动 FastAPI 应用程序：

    ```py
    uvicorn app:app --reload
    ```

1.  你应该能够执行用户注册和用户登录。使用 HTTPie 客户端尝试一下：

    ```py
    http 127.0.0.1:8000/users/register username=”marko” password=”marko123”
    ```

1.  服务器应该发送以下响应，但请注意，你的哈希和 UUID 将不同：

    ```py
    HTTP/1.1 200 OK
    content-length: 138
    content-type: application/json
    date: Sun, 07 Apr 2024 18:38:41 GMT
    server: uvicorn
    {
        “id”: “45cd212b-71eb-42b4-9d06-a74f2609764b”,
        “password”: “$2b$12$owWXcY5KgI9s6Rdfjcpx7eXaZOMWf8NaxN.SoLJ4h8O.xzFpRqEee”,
        “username”: “marko”
    }
    ```

    如果你查看 `users.json` 文件，你应该能看到类似以下的内容：

    ```py
    {
        “users”: [
            {
                “username”: “marko”,
                “password”: “$2b$12$owWXcY5KgI9s6Rdfjcpx7eXaZOMWf8NaxN.SoLJ4h8O.xzFpRqEee”,
                “id”: “45cd212b-71eb-42b4-9d06-a74f2609764b”
            }
        ]
    ```

注意

在现实世界的系统中，你甚至不希望将哈希密码发送给已登录的用户，但这个整个系统是为了演示目的而创建的，旨在尽可能具有说明性。

你已经创建了一个完整的身份验证流程（仅用于演示目的——在生产中你不会使用包含字典和 UUID 的 JSON 文件），并且你已经实现了所有必需的功能：创建用户（注册）、检查提交数据的有效性以及用户登录。最后，你通过创建一个测试用户来测试了注册功能。

### 使用 HTTPie 测试登录功能

现在，使用正确的用户/密码组合测试登录功能，然后使用错误的组合。

1.  首先，登录。在终端中，发出以下 HTTPie 命令：

    ```py
    http POST 127.0.0.1:8000/users/login username=”marko” password=”marko123”
    ```

    响应应该只是一个大字符串——你的 JWT——这个令牌的值（这里，它以字符串*eyJhbGciOiJ…*开头）应该被复制并保存以供稍后测试已验证的路由：

    ```py
    HTTP/1.1 200 OK
    content-length: 241
    content-type: application/json
    date: Sun, 07 Apr 2024 18:43:07 GMT
    server: uvicorn
    {
        “token”: 
    “eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MTI1MTcxODgsImlhdCI6MTcxMjUxNTM4OCwic3ViIjp7InVzZXJfaWQ iOiI0NWNkMjEyYi03MWViLTQyYjQtOWQwNi1hNzRmMjYwOTc2NGIiLCJ1c2VybmFtZS I6Im1hcmtvIn19.tFcJoKhTdDBDIBhCX-dCUEkCD3Fc8E-smQd2M_h5h2k”
    }
    ```

1.  尝试以下操作（注意密码是错误的）：

    ```py
    http POST 127.0.0.1:8000/users/login username=”marko” password=”marko111”
    ```

    响应将类似于以下内容：

    ```py
    HTTP/1.1 401 Unauthorized
    content-length: 45
    content-type: application/json
    date: Sun, 07 Apr 2024 18:44:34 GMT
    server: uvicorn
    {
        “detail”: “Invalid username and/or password”
    }
    ```

你已经从头开始实现了自己的 FastAPI 认证系统。现在，把它用在路由上会很好。

### 创建受保护的路由

假设你现在想要一个新端点，该端点列出你系统中所有的用户，并且你希望它只对已登录用户可见。这种方法将允许你通过利用强大的 FastAPI 依赖注入系统来保护不同路由中的任何路由：

1.  打开`users.py`文件，在末尾添加以下路由：

    ```py
    @router.get(“/list”, response_description=”List all users”)
    async def list_users(request: Request, user_data=Depends(auth_handler.auth_wrapper)):

        users = json.loads(open(“users.json”).read())[“users”]
        return UsersList(users=users)
    ```

    这条路径的关键在于`user_data`部分——如果依赖项不满足，该路径将响应异常，并显示`authentication.py`中定义的消息。

1.  尝试登录，获取从登录端点获得的 JWT（如果它还没有过期！），然后将其作为承载令牌传递：

    ```py
    http GET 127.0.0.1:8000/users/list ‘Authorization:Bearer <your Bearer Token>’     
    ```

    结果应包含你迄今为止创建的所有用户：

    ```py
    HTTP/1.1 200 OK
    content-length: 76
    content-type: application/json
    date: Sun, 07 Apr 2024 19:07:45 GMT
    server: uvicorn
    {
        “users”: [
            {
                “id”: “45cd212b-71eb-42b4-9d06-a74f2609764b”,
                “username”: “marko”
            }
        ]
    }
    ```

1.  如果你尝试修改令牌，或者让它过期，结果将是以下内容：

    ```py
    HTTP/1.1 401 Unauthorized
    content-length: 26
    content-type: application/json
    date: Sun, 07 Apr 2024 19:10:12 GMT
    server: uvicorn
    {
        “detail”: “Invalid token”
    }
    ```

在本节中，你看到了如何在 FastAPI 后端创建一个简单但高效的认证系统，创建 JWT 生成器，验证令牌，保护一些路由，并提供创建（注册）新用户和登录所需的路由。下一节将展示前端的工作方式。

# 在 React 中认证用户

在本节中，你将了解一个基本机制，这将使你能够在客户端实现简单的认证流程。一切都将围绕 JWT 以及你决定如何处理它展开。

React.js 是一个无偏见的 UI 库。它提供了多种实现用户认证和授权的方法。由于你的 FastAPI 后端实现了基于 JWT 的认证，你必须决定如何在 React 中处理 JWT。

在本章中，你将把它存储在内存中，然后存储在`localStorage`中（JavaScript 中的 HTML5 简单 Web 存储对象，允许应用程序在用户的 Web 浏览器中存储无过期日期的键值对）。本章不会涵盖基于 cookie 的解决方案，因为这种解决方案通常是最稳健和安全的，下一章将介绍此类解决方案。

这些方法各有优缺点，了解它们非常有用。认证应该始终非常认真对待，并且根据你的应用程序范围和需求，它应该始终是一个需要彻底分析的话题。

关于存储认证数据的最佳解决方案一直存在争议——在这种情况下，是 JWT。像往常一样，每种解决方案都有其优缺点。

Cookie 已经存在很长时间了——它们可以在浏览器中以键值对的形式存储数据，并且可以从浏览器和服务器中读取。它们的流行与经典的服务器端渲染网站相吻合。然而，它们只能存储非常有限的数据，并且数据结构必须非常简单。

`localstorage`和`sessionStorage`随着 HTML5 的引入而出现，是为了解决在**单页应用**（**SPAs**）中存储复杂数据结构的需求，以及其他一些需求。它们的容量大约为 10 MB，具体取决于浏览器的实现，而 cookie 的容量为 4 KB。会话存储数据在会话期间持续存在，而本地存储在浏览器中即使关闭和重新打开后也会保留，直到手动删除，这使得 SPAs 成为最令人愉悦但也最易受攻击的解决方案。两者都可以托管复杂的 JSON 数据结构。

在`localstorage`中存储 JWT 很简单，并且提供了极佳的用户和开发者体验。

大多数该领域的权威人士建议将 JWT 存储在 HTTP-only cookie 中，因为它们不能通过 JavaScript 访问，并且需要前端和后端在同一个域上运行。

这可以通过不同的方式实现，通过路由请求或使用代理。另一种流行的策略是使用所谓的刷新令牌。在此方法中，应用程序在登录时发行一个令牌，然后使用此令牌自动生成其他（刷新）令牌，从而在安全性和用户体验之间找到正确的平衡。

### Context API

在*第三章*，“Python 类型提示和 Pydantic”，你学习了如何通过`useState`钩子管理组件状态的简单部分。

假设你有一个顶级组件——甚至可能是根`App.js`组件——并且你需要将一些状态传递到 React 组件树中的深层嵌套组件。你需要将这部分数据传递给`App.js`状态组件内的组件，然后将其进一步传递到树中，直到达到真正需要这些数据的子组件。

这种模式被称为**属性钻取**——通过属性传递状态值，并且有多个不使用该状态值的组件；它们只是将其传递下去。属性钻取有几个影响，其中大多数最好是避免的：

+   重构和修改代码更困难，因为你必须始终保持状态值通信通道的完整性

+   代码的可重用性较低，因为组件需要始终提供状态值

+   需要编写的代码更多，因为组件需要接受和转发属性

React 通过**Context API**引入了一种在组件之间提供值而不需要属性钻取的方法。

### 创建一个简单的 SPA

在下一节中，你将创建一个非常简单的 SPA，允许用户注册（如果他们尚未注册），使用用户名和密码登录，如果认证成功，将看到所有注册用户的列表。UI 将紧密模仿你的后端。

注意

为了使前端功能正常且可测试，必须提供上一节中的后端，因此请确保使用以下命令运行 FastAPI 后端：

`uvicorn` `app:app --reload`

前端将通过 API 连接到正在运行的 FastAPI 后端。虽然 FastAPI 在地址`http://127.0.0.1:8000`上提供服务，但 React 前端将使用相同的 URL 进行连接，执行 GET 和 POST 请求，认证用户并列出资源。

你将了解将 JWT 存储在应用程序中的 Context API 的主要概念。以下步骤开始：

1.  创建一个新的 Vite React 项目，安装 Tailwind，并添加 Tailwind CSS，因为它简化了应用程序的样式。请参阅*第五章*，*设置 React 工作流程*，以了解如何操作。同时，删除不需要的文件和文件夹（如`App.css`）。

1.  在`/src`文件夹中创建一个新文件，并将其命名为`AuthContext.jsx`。`.jsx`扩展名是一个提醒，即上下文确实是一个 React 组件，它将包装所有需要访问上下文变量、函数、对象或数组的其他组件：

    ```py
    import {
        createContext
    } from ‘react’;
    const AuthContext = createContext();
    export const AuthProvider = ({
        children
    }) => {
        const [user, setUser] = useState(null);
        const [jwt, setJwt] = useState(null);
        const [message, setMessage] = useState(null);
        return (<AuthContext.Provider value={
            {
                user,
                jwt,
                register,
                login,
                logout,
                message,
                setMessage
            }
        } > {
                children
            } </AuthContext.Provider>)
    }
    ```

    上述代码显示了上下文创建的结构 - 你从 React 中导入了`createContext`并创建了你的第一个上下文（`AuthContext`）。在定义了一些状态变量和设置器（用于用户、`jwt`令牌和消息）之后，你返回了`AuthContext`组件和将在上下文中可用的值。语法与*第四章*，*FastAPI 入门*中检查的钩子语法略有不同，但这是一个你将多次重用的简单模板，如果你选择使用 Context API。

1.  虽然简单，但创建上下文涉及多个步骤：

    1.  首先，你需要创建将在应用程序中共享的实际上下文。

    1.  之后，应该将上下文提供给所有需要访问其值的组件。

    1.  需要访问上下文值的组件需要订阅上下文，以便能够读取，但也可以写入。

    因此，创建上下文的第一步应该是明确你需要传递给组件的确切信息类型。如果你这么想，你肯定希望使用 JWT，因为这就是这个练习的全部意义。为了展示上下文功能，你还将包括已登录的用户和将显示应用程序状态的消息。

    但是，由于上下文还可以包含并传递函数——这确实是它的最有用功能之一——你还将向上下文中添加 `register`、`login` 和 `logout` 函数。这可能在生产系统中不是你想要做的事情，但它将展示 Context API 的功能。

1.  现在，唯一剩下的事情是将函数添加到上下文中。为此，编辑现有的 `AuthContext.jsx` 文件，在声明状态变量之后，定义注册新用户的函数：

    ```py
        const register = async (username, password) => {
          try {
            const response = await fetch(‘http://127.0.0.1:8000/users/register’, {
              method: ‘POST’,
              headers: {
                ‘Content-Type’: ‘application/json’,
              },
              body: JSON.stringify({
                username,
                password
              }),
            });
            if (response.ok) {
              const data = await response.json();
              setMessage(`Registration successful: user ${data.username} created`);
            } else {
              const data = await response.json();
              setMessage(`Registration failed: ${JSON.stringify(data)}`);
            }
          } catch (error) {
            setMessage(`Registration failed: ${JSON.stringify(error)}`);
          }
        };
    ```

    这个简单的 JavaScript 函数是上下文的一部分，唯一与你的上下文交互的是设置状态消息——如果用户成功创建，消息将确认这一点。如果发生错误，消息将被设置为错误。你可能想要提供更复杂的验证逻辑和更友好的用户界面，但这对上下文的工作方式有很好的说明。

1.  现在添加与认证相关的其他函数——`login()` 函数：

    ```py
    const login = async (username, password) => {
      setJwt(null)
      const response = await     fetch(‘http://127.0.0.1:8000/users/login’, {
        method: ‘POST’,
        headers: {
          ‘Content-Type’: ‘application/json’,
        },
        body: JSON.stringify({
          username,
          password
        }),
      });
      if (response.ok) {
        const data = await response.json();
        setJwt(data.token);
        setUser({
          username
        });
        setMessage(`Login successful: token ${data.token.slice(0, 10)}..., user ${username}`);
      } else {
        const data = await response.json();
        setMessage(‘Login failed: ‘ + data.detail);
        setUser({
          username: null
        });
      }
    };
    ```

    上述代码与 `register` 函数类似——它向 FastAPI `/login` 端点发送一个 `POST` 请求，包含用户提供的用户名和密码，并在过程中清除任何现有的 JWT。如果请求成功，检索到的令牌将被设置为状态变量，并相应地设置用户名。

1.  最后一个拼图是注销用户。由于你只处理 Context API 而不是某些持久存储解决方案，代码非常简短；它只需要清除上下文变量并设置适当的消息：

    ```py
    const logout = () => {
      setUser(null);
      setJwt(null);
      setMessage(‘Logout successful’);
    };
    ```

1.  你的 `AuthContext` 几乎完成了——唯一剩下的事情是通知上下文它需要提供之前定义的函数。因此，修改 `return` 语句以包含所有内容：

    ```py
    return ( <
      AuthContext.Provider value = {
        {
          user,
          jwt,
          register,
          login,
          logout,
          message,
          setMessage
        }
      } > {
        children
      } <
      /AuthContext.Provider>
    );
    ```

1.  作为最后的润色，添加一个 `useContext` React 钩子，以简化与上下文的工作：

    ```py
    export const useAuth = () => useContext(AuthContext);
    ```

    这条简单的单行钩子现在允许你在任何可以访问上下文的组件中使用 `AuthContext` ——也就是说，任何被 `AuthContext` 包裹的组件——只需进行一些简单的 ES6 解构。现在你的 `AuthContext` 已经设置好了，你可以直接将其放入 `App.jsx` 组件中，并将其包裹在其他所有组件周围。

1.  打开 `App.jsx` 文件并编辑它：

    ```py
    import { AuthProvider } from “./AuthContext”;
    const App = () => {
      return (
        <div className=”bg-blue-200 flex flex-col justify-center items-center min-h-screen”>
          <AuthProvider>
            <h1 className=”text-2xl text-blue-800”> Simple Auth App </h1>
          </AuthProvider>{“ “}
        </div>
      );
    };
    export default App
    ```

    这个根组件不包含你之前没有见过的内容——除了导入 `AuthProvider`——这是你的自定义认证上下文组件，负责包裹组件区域和一点 Tailwind 样式。

1.  现在是定义将包裹在上下文内部的组件的部分——因为这些组件将能够消费上下文，访问上下文数据，并对其进行修改。对于更复杂的应用程序，你可能会求助于 React Router 包，但由于这将是一个非常简单的应用程序，你将把所有组件挤在一个页面上。它们并不多：

    +   上下文中的 `login()` 函数。

    +   **注册**：与登录组件类似，但用于注册新用户。

    +   **消息**：最简单的组件，仅用于显示应用程序的状态。

    +   **用户**：其状态依赖于身份验证状态的组件：如果用户已登录，他们可以看到用户列表，这意味着 JWT 存在且有效；否则，将提示用户进行登录。

1.  `Register` 组件将用于用户注册。它需要显示一个表单。在 `/src` 文件夹中创建 `Register.jsx` 文件，并创建一个包含两个字段的简单表单：

    ```py
    import { useState } from ‘react’;
    import { useAuth } from ‘./AuthContext’;
    const Register = () => {
        const [username, setUsername] = useState(‘’);
        const [password, setPassword] = useState(‘’);
        const { register } = useAuth();
        const handleSubmit = (e) => {
            e.preventDefault();
            register(username, password)
            setUsername(‘’)
            setPassword(‘’)
        };
        return (
            <div className=”m-5 p-5  border-2”>
                <form onSubmit={handleSubmit} className=’grid grid-rows-3 gap-2’>
                    <input
                        type=”text”
                        placeholder=”Username”
                        className=’p-2’
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                    />
                    <input
                        type=”password”
                        placeholder=”Password”
                        className=’p-2’
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                    />
                    <button type=”submit” className=’bg-blue-500 text-white rounded’>Register</button>
                </form>
            </div>
        );
    };
    export default Register
    ```

    你已经使用两个本地状态变量创建了一个特定于 React 的表单，这些变量负责跟踪并发送用户名和密码到你的 FastAPI 实例。`register` 函数通过 `useAuth()` 钩子从 `AuthContext` 中导入。那一行真正展示了在包装组件内部与上下文一起工作是多么容易。

    最后，`handleSubmit` 执行对 `register` 函数的调用，清除字段并阻止默认的 HTML 表单行为。

1.  创建 `Login.jsx` 文件，它与之前几乎相同（在这里你可以练习你的 React 技能并进行一些重构）。该组件有一个登录表单，将用于登录：

    ```py
    import { useState } from ‘react’;
    import { useAuth } from ‘./AuthContext’;
    const Login = () => {
        const [username, setUsername] = useState(‘’);
        const [password, setPassword] = useState(‘’);
        const { login } = useAuth();
        const handleSubmit = (e) => {
            e.preventDefault();
            login(username, password);
            setUsername(‘’);
            setPassword(‘’);
        };
        return (
            <div className=”m-5 p-5  border-2”>
                <form onSubmit={handleSubmit} className=’grid grid-rows-3 gap-2’>
                    <input
                        type=”text”
                        placeholder=”Username”
                        className=’p-2’
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                    />
                    <input
                        type=”password”
                        placeholder=”Password”
                        className=’p-2’
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                    />
                    <button type=”submit” className=’bg-blue-500 text-white rounded’>Login</button>
                </form>
            </div>
        );
    };
    export default Login
    ```

1.  在你的由 FastAPI 和 React 驱动的简单身份验证应用程序中，还有两个组件需要插入。首先，创建 `src/Message.jsx` 组件，它将用于显示状态消息：

    ```py
    import { useAuth } from “./AuthContext”
    const Message = () => {
        const { message } = useAuth()
        return (
            <div className=”p-2 my-2”>
                <p>{message}</p>
            </div>
        )
    }
    export default Message
    ```

    `Messages` 组件从上下文中读取消息状态变量并将其显示给用户。

1.  现在，你终于可以创建 `src/Users.jsx` 组件并对其进行编辑：

    ```py
    import { useEffect, useState } from ‘react’;
    import { useAuth } from ‘./AuthContext’;
    const Users = () => {
        const { jwt, logout } = useAuth();
        const [users, setUsers] = useState(null);
        useEffect(() => {
            const fetchUsers = async () => {
                const response = await fetch(‘http://127.0.0.1:8000/users/list’, {
                    headers: {
                        Authorization: `Bearer ${jwt}`,
                    },
                });
                const data = await response.json();
                setUsers(data.users);
            };
            if (jwt) {
                fetchUsers();
            }
        }, [jwt]);
        if (!jwt) return <div>Please log in to see all the users</div>;
        return (
            <div>
                {users ? (
                    <div className=’flex flex-col’>
                        <h1>The list of users</h1>
                        <ol>
                            {users.map((user) => (
                                <li key={user.id}>{user.username}</li>
                            ))}
                        </ol>
                        <button onClick={logout} className=’bg-blue-500 text-white rounded’>Logout</button>
                    </div>
                ) : (
                    <p>Loading...</p>
                )}
            </div>
        );
    };
    export default Users;
    ```

    与其他组件相比，这个组件做了一些繁重的工作。它从上下文中导入 `jwt`（以及 `logout` 函数）。这很重要，因为 `Users.jsx` 组件的输出完全取决于 JWT 的存在和有效性。

    在声明一个本地状态变量 `users` 之后，组件使用 `useEffect` React 钩子执行对 REST API 的调用，由于 `/users/list` 端点是受保护的，JWT 令牌需要存在且有效。

    如果对 `/users/list` 端点的调用成功，检索到的用户数据将被发送到 `users` 变量并显示。最后，如果上下文中没有 `jwt`，将要求用户执行登录操作并从上下文中调用 `logout` 函数。

1.  最后，为了将所有这些整合在一起，用以下代码替换 `App.jsx` 文件以导入组件，并最终完成根组件：

    ```py
    import { useState } from ‘react’;
    import { AuthProvider } from ‘./AuthContext’;
    import Register from ‘./Register’;
    import Login from ‘./Login’;
    import Users from ‘./Users’;
    import Message from ‘./Message’;
    const App = () => {
      const [showLogin, setShowLogin] = useState(true)
      return (
        <div className=’bg-blue-200 flex flex-col justify-center items-center min-h-screen’>
          <AuthProvider>
            <h1 className=’text-2xl text-blue-800’>Simple Auth App</h1>
            <Message />
            {showLogin ? <Login /> : <Register />}
            <button onClick={() => setShowLogin(!showLogin)}>{showLogin ? ‘Register’ : ‘Login’}</button>
            <hr />
            <Users />
          </AuthProvider>
        </div>
      );
    };
    export default App;
    ```

现在，你将能够测试应用程序——尝试注册、登录、输入无效数据等等。你已经创建了一个非常简单但完整的全栈身份验证解决方案。在下一节中，你将了解一些持久化登录数据的方法。

## 使用 localStorage 持久化身份验证数据

如前所述，对于持久化认证，最符合开发者需求的选项是使用 `localStorage` 或 `sessionStorage`。当涉及到存储临时、本地数据时，`localStorage` 非常有用。它被广泛用于记住购物车数据或用户在任何网站上的登录信息（在这些网站上安全性不是首要考虑因素）。与 cookies 相比，`localStorage` 具有更高的存储限制（5 MB 对 4 KB），并且不会随着每个 HTTP 请求发送。这使得它成为客户端存储的更好选择。

要使用 `localStorage`，你可以分别使用 `setItem()` 和 `getItem()` 方法来设置和获取 JSON 项。需要记住的一个重要事项是 `localStorage` 只存储字符串，因此你需要使用 `JSON.stringify()` 和 `JSON.parse()` 在 JavaScript 对象和字符串之间进行转换。

带着这些知识，尝试总结应用的要求——你希望用户能够在刷新或关闭并重新打开应用程序窗口/标签页的情况下保持登录状态，如果他们最初已经登录。用 React 术语来说，你需要一个 `useEffect` 钩子，它会运行并验证 `localStorage` 中是否存储了令牌。如果存在，你想要通过 FastAPI 的 `/me` 端点检查这个令牌，并相应地设置用户名：

1.  打开现有的 `AuthContext.jsx` 文件，在 `useState` 钩子之后定义 `useEffect` 调用：

    ```py
        export const AuthProvider = ({ children }) => {
        const [user, setUser] = useState(null);
        const [jwt, setJwt] = useState(null);
        const [message, setMessage] = useState(null);
        useEffect(() => {

            const storedJwt = localStorage.getItem(‘jwt’);
            if (storedJwt) {
                setJwt(storedJwt);
                fetch(‘http://127.0.0.1:8000/users/me’, {
                    headers: {
                        Authorization: `Bearer ${storedJwt}`,
                    },
                })
                    .then(res => res.json())
                    .then(data => {
                        if (data.username) {
                            setUser({ username: data.username });
                            setMessage(`Welcome back, ${data.username}!`);
                        }
                    })
                    .catch(() => {
                        localStorage.removeItem(‘jwt’);
                    });
            }
        }, []);
    ```

    大部分持久化逻辑都位于 `useEffect` 调用中。首先，你可以尝试从 `localStorage` 获取 `jwt` 令牌，然后使用该令牌从 `/me` 路由获取用户数据。如果找到用户名，它会被设置在上下文中，并且用户（已经）登录。如果没有找到，你将清除 `localStorage` 或在 `Users.jsx` 组件中发送一个令牌已过期的消息。

1.  `login()` 函数也必须进行修改，以便考虑到 `localStorage`。在相同的 `AuthContext.jsx` 中修改 `login()` 函数：

    ```py
    const login = async (username,
      password) => {
        setJwt(null)
        const response = await fetch(
          ‘http://127.0.0.1:8000/users/login’, {
            method: ‘POST’,
            headers: {
              ‘Content-Type’: ‘application/json’,
            },
            body: JSON.stringify({
              username,
              password
            }),
          });
        if (response.ok) {
          const data = await response
            .json();
          setJwt(data.token);
     localStorage.setItem(‘jwt’, data.token);
          setUser({
            username
          });
          setMessage(
            `Login successful: token ${data.token.slice(0, 10)}..., user ${username}`
            );
        } else {
          const data = await response
            .json();
          setMessage(‘Login failed: ‘ +
            data.detail);
          setUser({
            username: null
          });
        }
      };
    ```

    唯一的修改是将新的 JWT 设置到 `localStorage` 的 `jwt` 变量中。因此，`logout()` 函数也需要清除 `localStorage`。

1.  在相同的 `AuthContext.jsx` 文件中，修改 `logout` 函数：

    ```py
    const logout = () => {
        setUser(null);
        setJwt(‘’);
        localStorage .removeItem(‘jwt’);
        setMessage(‘Logout successful’);
    };
    ```

1.  最后，为了使你的应用程序更加明确和易于理解，打开 `Users.jsx` 组件并将其替换为以下代码：

    ```py
    import { useEffect, useState } from ‘react’;
    import { useAuth } from ‘./AuthContext’;
    const Users = () => {
        const { jwt, logout } = useAuth();
        const [users, setUsers] = useState(null);
        const [error, setError] = useState(null);
        useEffect(() => {
            const fetchUsers = async () => {
                const response = await fetch(‘http://127.0.0.1:8000/users/list’, {
                    headers: {
                        Authorization: `Bearer ${jwt}`,
                    },
                });
                const data = await response.json();
                if (!response.ok) {
                    setError(data.detail);
                }
                setUsers(data.users);
            };
            if (jwt) {
                fetchUsers();
            }
        }, [jwt]);
        if (!jwt) return <div>Please log in to see all the users</div>;
        return (
            <div>
                {users ? (
                    <div className=’flex flex-col’>
                        <h1>The list of users</h1>
                        <ol>
                            {users.map((user) => (
                                <li className=’’ key={user.id}>{user.username}</li>
                            ))}
                        </ol>
                        <button onClick={logout} className=’bg-blue-500 text-white rounded’>Logout</button>
                    </div>
                ) : (
                    <p>{error}</p>
                )}
            </div>
        );
    };
    export default Users;
    ```

现在应用程序能够持久化已登录用户，检索存储的 JWT，并恢复之前的认证状态。在尝试登录之前，请确保 FastAPI 后端在端口 `8000` 上正常运行。

尝试登录，刷新浏览器，关闭标签页，然后重新打开。

你也可以在 Chrome 或 Firefox 的开发者工具栏中的 **Application** 标签页内尝试使用令牌，看看如果你篡改它或删除它会发生什么。

## 其他认证解决方案

重要的是再次强调，例如 `localStorage`，但需要考虑到两种解决方案的具体细节。

最后，熟悉各种第三方身份验证选项也很重要。**Firebase**和**Supabase**是流行的数据库和身份验证服务，可以仅用于管理用户和验证他们。**Clerk**和**Kinde**是该领域的后起之秀，特别针对 React/Next.js/Remix.js 生态系统，而**Auth0**和**Cognito**是行业标准解决方案。几乎所有第三方身份验证系统都提供慷慨的免费或几乎免费的级别，但一旦你的应用程序增长，你不可避免地会遇到付费级别，而且费用各不相同，如果需要，替换这些服务并不容易。

# 摘要

在本章中，你看到了一个非常基础但相当有代表性的身份验证机制两种版本的实现。你学习了 FastAPI 如何启用符合标准身份验证方法的使用，并实现了最简单但有效的解决方案之一——不持久化身份验证数据和不存储`localStorage`。

你已经了解到，在定义细粒度角色和权限方面，FastAPI 是多么优雅和灵活，尤其是在 MongoDB 的帮助下，**Pydantic**作为中间人。本章专注于**JWTs**作为通信手段，因为它是当今 SPA 的主要和最受欢迎的工具，它使得服务或微服务之间具有很好的连接性。当你需要使用相同的 FastAPI 和 MongoDB 后端开发不同的应用程序时，**JWT**机制特别出色——例如，一个 React 网络应用程序和一个基于 React Native 或 Flutter 的移动应用程序。

此外，仔细考虑你的身份验证和授权策略至关重要，尤其是在从第三方系统中提取用户数据可能不可行或不切实际的情况下。这突出了制定稳健的身份验证和授权方法的重要性。

在下一章中，你将创建一个更复杂的 FastAPI 后端，通过第三方服务上传图片，并使用 MongoDB 数据库进行持久化。
