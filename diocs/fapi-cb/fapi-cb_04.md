

# 第四章：身份验证和授权

在我们的*FastAPI 食谱集*这一章中，我们将深入研究身份验证和授权的关键领域，为您构建安全网络应用程序免受未经授权访问的基础。

在我们浏览本章内容的过程中，您将开始一段实际旅程，在 FastAPI 应用程序中实施一个全面的安全模型。从用户注册和身份验证的基础到将复杂的**OAuth2**协议与**JSON Web Token**（**JWT**）集成以提高安全性，本章涵盖了所有内容。

我们将创建**软件即服务**（**SaaS**）的基本组件，帮助您学习如何实际建立用户注册系统、验证用户和高效处理会话。我们还将向您展示如何应用**基于角色的访问控制**（**RBAC**）来调整用户权限，并使用 API 密钥身份验证保护 API 端点。通过使用 GitHub 等外部登录服务进行第三方身份验证，将展示如何利用现有平台进行用户身份验证，简化用户的登录过程。

此外，您将通过实施**多因素身份验证**（**MFA**）添加一个额外的安全层，确保您的应用程序能够抵御各种攻击向量。

在本章中，我们将介绍以下食谱：

+   设置用户注册

+   使用 OAuth2 和 JWT 进行身份验证

+   设置 RBAC

+   使用第三方身份验证

+   实施 MFA

+   处理 API 密钥身份验证

+   处理会话 cookie 和注销功能

# 技术要求

要深入了解本章并跟随身份验证和授权的食谱，请确保您的设置包括以下基本要素：

+   **Python**：在您的环境中安装一个高于 3.9 版本的 Python。

+   **FastAPI**：应与所有必需的依赖项一起安装。如果您在前几章中没有这样做，您可以从终端简单地完成它：

    ```py
    $ pip install fastapi[all]
    ```

本章中使用的代码托管在 GitHub 上，地址为[`github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter04`](https://github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter04)。

在项目根目录内为项目设置虚拟环境也是推荐的，这样可以高效地管理依赖项并保持项目隔离。在您的虚拟环境中，您可以使用 GitHub 项目文件夹中提供的`requirements.txt`文件一次性安装所有依赖项：

```py
pip install –r requirements.txt
```

由于编写时交互式 Swagger 文档有限，因此掌握**Postman**或其他测试 API 的基本技能对测试我们的 API 有益。

现在我们有了这个准备，我们可以开始准备我们的食谱。

# 设置用户注册

用户注册是保护你的 FastAPI 应用程序的第一步。它涉及收集用户详细信息并安全地存储它们。以下是你可以设置基本用户注册系统的方法。配方将向你展示如何设置 FastAPI 应用的注册系统。

## 准备工作

我们将首先在 SQL 数据库中存储用户。让我们创建一个名为 `saas_app` 的项目根文件夹，其中包含代码库。

为了存储用户密码，我们将使用一个外部包来使用 **bcrypt** 算法散列纯文本。散列函数将文本字符串转换为一个独特且不可逆的输出，允许安全地存储敏感数据，如密码。更多详情请参阅 [`en.wikipedia.org/wiki/Hash_function`](https://en.wikipedia.org/wiki/Hash_function)。

如果你还没有在 `saas_app` 项目文件夹下安装来自 GitHub 仓库的 `requirements.txt` 中的包，你可以通过运行以下命令安装 `passlib` 包，其中包含 `bcrypt`：

```py
$ pip install passlib[bcrypt]
```

你还需要安装一个高于 2.0.0 版本的 `sqlalchemy`，以便跟随 GitHub 仓库中的代码：

```py
$ pip install sqlalchemy>=2.0.0
```

我们的环境现在已准备好在我们的 SaaS 中实现用户注册。

## 如何操作…

在开始实施之前，我们需要设置数据库以存储我们的用户。

我们需要设置一个 `sqlalchemy`，以便应用程序存储用户凭据。

你需要做以下事情：

+   设置一个 `User` 类来映射 SQL 数据库中的用户表。该表应包含 `id`、`username`、`email` 和 `hashed_password` 字段。

+   建立应用程序与数据库之间的连接。

首先，让我们创建一个名为 `saas_app` 的项目根文件夹。然后你可以参考 *第二章* 中的 *设置 SQL 数据库* 配方，或者从 GitHub 仓库中复制 `database.py` 和 `db_connection.py` 模块到你的根文件夹下。

在设置好数据库会话后，让我们定义一个添加用户的函数。

让我们将其变成一个名为 `operations.py` 的专用模块，在其中我们将定义所有由 API 端点使用的支持函数。

该函数将使用来自 `bcrypt` 包的密码上下文对象来散列纯文本密码。我们可以如下定义它：

```py
from passlib.context import CryptContext
pwd_context = CryptContext(
    schemes=["bcrypt"], deprecated="auto"
)
```

我们可以定义一个名为 `add_user` 的函数，根据大多数数据合规规定，将带有散列密码的新用户插入数据库中：

```py
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from models import User
def add_user(
    session: Session,
    username: str,
    password: str,
    email: str,
) -> User | None:
    hashed_password = pwd_context.hash(password)
    db_user = User(
        username=username,
        email=email,
        hashed_password=hashed_password,
    )
    session.add(db_user)
    try:
        session.commit()
        session.refresh(db_user)
    except IntegrityError:
        session.rollback()
        return
    return db_user
```

`IntegrityError` 将考虑尝试添加已存在的用户名或电子邮件的尝试。

现在，我们必须定义我们的端点，但首先，我们需要设置我们的服务器并初始化数据库连接。我们可以在 `main.py` 模块中这样做，如下所示：

```py
from contextlib import (
    asynccontextmanager,
)
from fastapi import  FastAPI
from db_connection import get_engine
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=get_engine())
    yield
app = FastAPI(
    title="Saas application", lifespan=lifespan
)
```

我们使用 `FastAPI` 对象的 `lifespan` 参数来指示服务器在启动时同步我们的数据库类 `User` 与数据库。

此外，我们还可以创建一个单独的模块，`responses.py`，以保存用于不同端点的响应类。请随意创建自己的或复制 GitHub 仓库中提供的那个。

我们现在可以编写合适的端点来在同一个 `main.py` 模块中注册用户：

```py
from typing import Annotated
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from models import Base
from db_connection import get_session
from operations import add_user
@app.post(
    "/register/user",
    status_code=status.HTTP_201_CREATED,
    response_model=ResponseCreateUser,
    responses={
        status.HTTP_409_CONFLICT: {
            "description": "The user already exists"
        }
    },
)
def register(
    user: UserCreateBody,
    session: Session = Depends(get_session),
) -> dict[str, UserCreateResponse]:
    user = add_user(
        session=session, **user.model_dump()
    )
    if not user:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "username or email already exists",
        )
    user_response = UserCreateResponse(
        username=user.username, email=user.email
    )
    return {
        "message": "user created",
        "user": user_response,
    }
```

我们刚刚实现了一个基本的机制来在我们的 SaaS 数据库中注册和存储用户。

## 它是如何工作的...

该端点将接受一个包含用户名、电子邮件和密码的 JSON 主体。

如果用户名或电子邮件已存在，将返回 `409` 响应，并且不允许创建用户。

要测试这个，在项目根目录下，运行以下命令启动服务器：

```py
$ uvicorn main:app
```

然后，使用浏览器连接到 `localhost:8000/docs` 并检查我们在 Swagger 文档中刚刚创建的端点。请随意尝试。

练习

为 `add_user` 函数和 `/register/user` 端点创建适当的测试，例如以下内容：

`def test_add_user_into_the_database(session):`

`user =` `add_user(…`

`# fill in` `the test`

`def test_endpoint_add_basic_user(client):`

`response =` `client.post(`

`"/``register/user",`

`json=`

`# continue` `the test`

你可以以对你最有利的方式安排测试。

你可以在书的 GitHub 仓库的 `Chapter04/saas_app` 文件夹中找到一个可能的测试方法。

## 参见

**bcrypt** 库允许你为你的哈希函数添加多层安全性，例如盐和额外的密钥。请随意查看 GitHub 上的源代码：

+   *Bcrypt GitHub 仓库*：[`github.com/pyca/bcrypt/`](https://github.com/pyca/bcrypt/)

此外，你还可以在以下位置找到一些有趣的示例，说明如何使用它：

+   *使用 Bcrypt 在 Python 中哈希密码*：[`www.geeksforgeeks.org/hashing-passwords-in-python-with-bcrypt/`](https://www.geeksforgeeks.org/hashing-passwords-in-python-with-bcrypt/)

# 使用 OAuth2 和 JWT 进行认证

在这个配方中，我们将集成 OAuth2 和 JWT 以在应用程序中进行安全的用户认证。这种方法通过利用令牌而不是凭据来提高安全性，符合现代认证标准。

## 准备工作

由于我们将使用特定的库来管理 JWT，请确保你已经安装了必要的依赖项。如果你还没有从 `requirements.txt` 安装包，请运行以下命令：

```py
$ pip install python-jose[cryptography]
```

此外，我们还将使用之前配方中使用的用户表，*设置用户注册*。确保在开始配方之前已经设置好。

## 如何做到这一点...

我们可以通过以下步骤设置 JWT 令牌集成。

1.  在一个名为 `security.py` 的新模块中，让我们定义用户的认证函数：

    ```py
    from sqlalchemy.orm import Session
    from models import User
    from email_validator import (
        validate_email,
        EmailNotValidError,
    )
    from operations import pwd_context
    def authenticate_user(
        session: Session,
        username_or_email: str,
        password: str,
    ) -> User | None:
        try:
            validate_email(username_or_email)
            query_filter = User.email
        except EmailNotValidError:
            query_filter = User.username
        user = (
            session.query(User)
            .filter(query_filter == username_or_email)
            .first()
        )
        if not user or not pwd_context.verify(
            password, user.hashed_password
        ):
            return
        return user
    ```

    该函数可以根据用户名或电子邮件验证输入。

1.  让我们在同一个模块（`create_access_token` 和 `decode_access_token`）中定义创建和解码访问令牌的函数。

    要创建访问令牌，我们需要指定一个密钥、用于生成它的算法以及过期时间，如下所示：

    ```py
    SECRET_KEY = "a_very_secret_key"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    ```

    然后，`create_access_token_function`如下所示：

    ```py
    from jose import jwt
    def create_access_token(data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode, SECRET_KEY, algorithm=ALGORITHM
        )
        return encoded_jwt
    ```

    要解码访问令牌，我们可以使用一个支持函数`get_user`，该函数通过用户名返回`User`对象。您可以在`operations.py`模块中自行实现，或者从 GitHub 仓库中获取。

    解码令牌的函数如下所示：

    ```py
    from jose import JWTError
    def decode_access_token(
        token: str, session: Session
    ) -> User | None:
        try:
            payload = jwt.decode(
                token, SECRET_KEY, algorithms=[ALGORITHM]
            )
            username: str = payload.get("sub")
        except JWTError:
            return
        if not username:
            return
        user = get_user(session, username)
        return user
    ```

1.  现在，我们可以继续在同一模块`security.py`中使用`APIRouter`类创建检索令牌的端点：

    ```py
    from fastapi import (
        APIRouter,
        Depends,
        HTTPException,
        status,
    )
    from fastapi.security import (
        OAuth2PasswordRequestForm,
    )
    router = APIRouter()
    class Token(BaseModel):
        access_token: str
        token_type: str
    @router.post(
        "/token",
        response_model=Token,
        responses=..., # document the responses
    )
    def get_user_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        session: Session = Depends(get_session),
    ):
        user = authenticate_user(
            session,
            form_data.username,
            form_data.password
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
            )
        access_token = create_access_token(
            data={"sub": user.username}
        )
        return {
            "access_token": access_token,
            "token_type": "bearer",
        }
    ```

1.  然后，我们现在可以为`POST /token`端点创建一个`OAuth2PasswordBearer`对象以获取访问令牌：

    ```py
    from fastapi.security import (
        OAuth2PasswordBearer,
    )
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    ```

1.  最后，我们可以创建一个返回基于令牌的凭据的`/users/me`端点：

    ```py
    @router.get(
        "/users/me",
        responses=..., # document responses
    )
    def read_user_me(
        token: str = Depends(oauth2_scheme),
        session: Session = Depends(get_session),
    ):
        user = decode_access_token(token, session)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not authorized",
            )
        return {
            "description": f"{user.username} authorized",
        }
    ```

1.  现在，让我们在`main.py`中将这些端点导入到 FastAPI 服务器中。在定义 FastAPI 对象后，让我们添加路由器，如下所示：

    ```py
    import security
    # rest of the code
    app.include_router(security.router)
    ```

我们刚刚为我们的 SaaS 定义了认证机制。

## 它是如何工作的…

现在，从项目根目录的终端运行以下代码来启动服务器：

```py
$ uvicorn main:app
```

在浏览器中转到 Swagger 文档地址（`localhost:8000/docs`），您将看到新的端点`POST /token`和`GET /users/me`。

您需要令牌来调用第二个端点，您可以通过点击锁形图标并填写凭据表单来自动在浏览器中存储它。

通过使用 JWT 和 OAuth2，您已经使您的 SaaS 应用程序更加安全，这有助于您保护敏感端点，并确保只有登录用户才能使用它们。这种安排为您提供了可靠且安全的方式来验证用户，这对于现代 Web 应用程序来说效果很好。

## 相关内容

您可以通过阅读这篇文章更好地理解 OAuth2 框架：

+   *OAuth2 简介*: [`www.digitalocean.com/community/tutorials/an-introduction-to-oauth-2`](https://www.digitalocean.com/community/tutorials/an-introduction-to-oauth-2)

您还可以查看以下 JWT 协议定义：

+   *JWT IETF* *文档*: [`datatracker.ietf.org/doc/html/rfc7519`](https://datatracker.ietf.org/doc/html/rfc7519)

# 设置 RBAC

基于组织内个体用户的角色来调节资源访问的 RBAC 是一种方法。在本食谱中，我们将实现 RBAC 在 FastAPI 应用程序中，以有效地管理用户权限。

## 准备工作

由于我们将扩展数据库以容纳角色定义，请确保在深入此之前已经完成了*设置用户注册*食谱。

要设置访问控制，我们首先需要定义一系列我们可以分配的角色。让我们按照以下步骤来做。

1.  在`module.py`模块中，我们可以定义一个新的类`Role`，并将其作为`User`模型的新字段添加，该字段将存储在用户表中：

    ```py
    from enum import Enum
    class Role(str, Enum):
        basic = "basic"
        premium = "premium"
    class User(Base):
        __tablename__ = "users"
    # existing fields
        role: Mapped[Role] = mapped_column(
            default=Role.basic
        )
    ```

1.  然后，在`operations.py`模块中，我们将修改`operations.py`中的`add_user`函数，以接受一个参数来定义用户角色；默认值将是基本角色：

    ```py
    from models import Role
    def add_user(
        session: Session,
        username: str,
        password: str,
        email: str,
        role: Role = Role.basic,
    ) -> User | None:
        hashed_password = pwd_context.hash(password)
        db_user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            role=role,
        )
        # rest of the function
    ```

1.  让我们创建一个新的模块`premium_access.py`，并通过一个新的路由器定义端点来注册高级用户，这将非常类似于注册基本用户的端点：

    ```py
    @router.post(
        "/register/premium-user",
        status_code=status.HTTP_201_CREATED,
        response_model=ResponseCreateUser,
        responses=..., # document responses
    )
    def register_premium_user(
        user: UserCreateBody,
        session: Session = Depends(get_session),
    ):
        user = add_user(
            session=session,
             *user.model_dump(),
            role=Role.premium,
        )
        if not user:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "username or email already exists",
            )
        user_response = UserCreate(
            username=user.username,
            email=user.email,
        )
        return {
            "message": "user created",
            "user": user_response,
        }
    similar to the ones used in other modules.
    ```

1.  让我们在`main.py`模块中的`app`类中添加路由器：

    ```py
    import security
    import premium_access
    # rest of the code
    app.include_router(security.router)
    app.include_router(premium_access.router)
    ```

现在我们已经拥有了在 SaaS 应用程序中实现 RBAC 的所有元素。

## 如何做到这一点...

让我们创建两个端点，一个对所有用户可访问，另一个仅对高级用户保留。让我们通过以下步骤创建端点。

1.  首先，让我们创建两个辅助函数，`get_current_user`和`get_premium_user`，分别用于检索每个案例，并作为端点的依赖项使用。

    我们可以定义一个单独的模块，称为`rbac.py`模块。让我们从导入开始：

    ```py
    from typing import Annotated
    from fastapi import (
        APIRouter,
        Depends,
        HTTPException,
        Status
    )
    from sqlalchemy.orm import Session
    from db_connection import get_session
    from models import Role
    from security import (
        decode_access_token,
        oauth2_scheme
    )
    ```

    然后，我们创建了一个将用于端点的请求模型：

    ```py
    class UserCreateResquestWithRole(BaseModel):
        username: str
        email: EmailStr
        role: Role
    ```

    然后，我们定义一个支持函数，根据令牌检索用户：

    ```py
    def get_current_user(
        token: str = Depends(oauth2_scheme),
        session: Session = Depends(get_session),
    ) -> UserCreateRequestWithRole:
        user = decode_access_token(token, session)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not authorized",
            )
        return UserCreateRequestWithRole(
            username=user.username,
            email=user.email,
            role=user.role,
        )
    ```

    然后，我们可以利用这个函数仅筛选出高级用户：

    ```py
    def get_premium_user(
        current_user: Annotated[
            get_current_user, Depends()
        ]
    ):
        if current_user.role != Role.premium:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not authorized",
            )
        return current_user
    ```

1.  现在，我们可以使用这些函数在同一个模块中的路由器来创建相应的端点。首先，我们为所有用户定义一个欢迎页面：

    ```py
    router = APIRouter()
    @router.get(
        "/welcome/all-users",
        responses=..., # document responses
    )
    def all_users_can_access(
        user: Annotated[get_current_user, Depends()]
    ):
        return {
            f"Hello {user.username}, "
            "welcome to your space"
        }
    ```

    然后，我们定义端点，仅允许高级用户访问：

    ```py
    @router.get(
        "/welcome/premium-user",
        responses={
            status.HTTP_401_UNAUTHORIZED: {
                "description": "User not authorized"
            }
        },
    )
    def only_premium_users_can_access(
        user: UserCreateResponseWithRole = Depends(
            get_premium_user
        ),
    ):
        return {
            f"Hello {user.username}, "
            "Welcome to your premium space"
        }
    ```

1.  让我们在`main.py`中添加我们创建的路由器：

    ```py
    import security
    import premium_access
    import rbac
    # rest of the module
    app.include_router(premium_access.router)
    app.include_router(rbac.router)
    # rest of the module
    ```

我们已经实现了两个基于使用角色的权限端点。

要测试我们的端点，从命令行启动服务器：

```py
$ uvicorn main:app
```

然后，从您的浏览器中，访问`http://localhost:8000/docs`上的 Swagger 页面，您可以看到刚刚创建的新端点。

一种实验方法是创建一个基本用户和一个高级用户，并使用相应的端点。在您创建了用户之后，您可以尝试使用`GET welcome/all-users`和`GET /welcome/premium-user`端点以及两个角色，并查看响应是否符合角色的预期。

在这个食谱中，您只是创建了基于用户角色的简单端点。您还可以尝试创建更多角色和端点。

## 还有更多...

应用 RBAC 的另一种方式是为令牌分配一个作用域。这个作用域可以是一个表示某些权限的字符串。因此，角色由令牌生成系统控制。在 FastAPI 中，您可以在令牌内定义作用域。您可以查看专门的文档页面以获取更多信息：[`fastapi.tiangolo.com/advanced/security/oauth2-scopes/`](https://fastapi.tiangolo.com/advanced/security/oauth2-scopes/).

# 使用第三方身份验证

将第三方身份验证集成到您的 FastAPI 应用程序中允许用户使用他们现有的社交媒体账户登录，例如 Google 或 Facebook。本食谱将指导您通过集成 GitHub 第三方登录的过程，通过简化登录过程来增强用户体验。

## 准备工作

我们将专注于集成 GitHub OAuth2 进行认证。GitHub 提供了全面的文档和一个支持良好的客户端库，简化了集成过程。

您的环境中需要`httpx`包，所以如果您还没有通过`requirements.txt`安装它，可以通过运行以下命令来完成：

```py
$ pip install httpx
```

您还需要设置一个 GitHub 账户。如果您还没有，请创建一个；您可以在官方文档中找到全面的指南，网址为[`docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github`](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github)。

然后，您需要按照以下步骤在您的账户中创建一个应用程序：

1.  从您的个人页面，点击屏幕右上角的个人资料图标，然后导航到`SaasFastAPIapp`。

1.  `http://localhost:8000/home`，这是我们稍后要创建的。

1.  `http://localhost:8000/github/auth/token`，我们稍后也将定义它。

1.  点击**注册应用程序**，应用程序将被创建，并且您将被重定向到一个列出 OAuth2 应用程序必要数据的页面。

1.  注意客户端 ID，然后点击**生成新的****客户密钥**。

1.  保存您刚刚创建的客户密钥。有了客户端 ID 和客户密钥，我们可以继续通过 GitHub 实现第三方认证。

现在，我们已经拥有了将 GitHub 第三方登录集成到我们应用程序所需的一切。

## 如何操作...

让我们从创建一个名为`third_party_login.py`的新模块开始，用于存储 GitHub 认证的辅助数据和函数。然后我们继续如下。

1.  在`third_party_login.py`模块中，您可以定义用于认证的变量：

    ```py
    GITHUB_CLIENT_ID = "your_github_client_id"
    GITHUB_CLIENT_SECRET = (
        "your_github_client_secret"
    )
    GITHUB_REDIRECT_URI = (
        "http://localhost:8000/github/auth/token"
    )
    GITHUB_AUTHORIZATION_URL = (
        "https://github.com/login/oauth/authorize"
    )
    ```

    对于`GITHUB_CLIENT_ID`和`GITHUB_CLIENT_SECRET`，请使用 OAuth 应用的值。

警告

在生产环境中，请确保不要在代码库中硬编码任何用户名或客户端 ID。

1.  然后，仍然在`third_party_login.py`模块中，让我们定义一个辅助函数`resolve_github_token`，该函数解析 GitHub 令牌并返回有关用户的信息：

    ```py
    import httpx
    from fastapi import Depends, HTTPException
    from fastapi.security import OAuth2
    from sqlalchemy.orm import Session
    from models import User, get_session
    from operations import get_user
    def resolve_github_token(
        access_token: str = Depends(OAuth2()),
        session: Session = Depends(get_session),
    ) -> User:
        user_response = httpx.get(
            "https://api.github.com/user",
            headers={"Authorization": access_token},
        ).json()
        username = user_response.get("login", " ")
        user = get_user(session, username)
        if not user:
            email = user_response.get("email", " ")
            user = get_user(session, email)
        # Process user_response
        # to log the user in or create a new account
        if not user:
            raise HTTPException(
                status_code=403, detail="Token not valid"
            )
        return user
    ```

1.  在一个名为`github_login.py`的新模块中，我们可以开始创建用于 GitHub 认证的端点。让我们创建一个新的路由器和`github_login`端点，该端点将返回前端用于将用户重定向到 GitHub 登录页面的 URL：

    ```py
    import httpx
    from fastapi import APIRouter, HTTPException, status
    from security import Token
    from third_party_login import (
        GITHUB_AUTHORIZATION_URL,
        GITHUB_CLIENT_ID,
        GITHUB_CLIENT_SECRET,
        GITHUB_REDIRECT_URI,
    )
    router = APIRouter()
    @router.get("/auth/url")
    def github_login():
        return {
            "auth_url": GITHUB_AUTHORIZATION_URL
            + f"?client_id={GITHUB_CLIENT_ID}"
        }
    ```

1.  现在，让我们在`main.py`模块中将路由器添加到服务器：

    ```py
    import github_login
    # rest of the module
    app.include_router(github_login.router)
    # rest of the module
    ```

1.  使用相同的命令`uvicorn main:app`启动服务器，并调用我们刚刚创建的端点`GET /auth/url`。您将在响应中看到一个类似的链接：[`github.com/login/oauth/authorize?client_id=your_github_client_id`](https://github.com/login/oauth/authorize?client_id=your_github_client_id)。

    此链接由 GitHub 用于认证。重定向由前端管理，不在此书的范围之内。

1.  验证登录后，您将被重定向到一个 `404` 页面。这是因为我们还没有在我们的应用程序中创建回调端点。让我们在 `github_login.py` 模块中这样做：

    ```py
    @router.get(
        "/github/auth/token",
        response_model=Token,
        responses=..., # add responses documentation
    )
    async def github_callback(code: str):
        token_response = httpx.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": GITHUB_REDIRECT_URI,
            },
            headers={"Accept": "application/json"},
        ).json()
        access_token = token_response.get("access_token")
        if not access_token:
            raise HTTPException(
                status_code=401,
                detail="User not registered",
            )
        token_type = token_response.get(
            "token_type", "bearer"
        )
        return {
            "access_token": access_token,
            "token_type": token_type,
        }
    ```

    我们刚刚创建的端点返回实际的访问令牌。

1.  如果您重新启动服务器并尝试使用由 `GET` `/auth/url` 端点提供的链接再次验证 GitHub 登录，您将收到包含类似以下内容的响应：

    ```py
    {
        "access_token": "gho_EnHbcmHdCHD1Bf2QzJ2B6gyt",
        "token_type": "bearer"
    }
    ```

1.  最后一部分是创建一个可以通过 GitHub 令牌访问的主页端点，并且可以通过解析令牌来识别用户。我们可以在 `main.py` 模块中定义它：

    ```py
    from third_party_login import resolve_github_token
    @router.get(
        "/home",
        responses=…, # add responses documentation
    )
    def homepage(
        user: UserCreateResponse = Depends(
            resolve_github_token
        ),
    ):
        return {
            "message" : f"logged in {user.username} !"
        }
    ```

您刚刚实现了一个通过 GitHub 第三方认证器进行认证的端点。

## 它是如何工作的…

首先，通过使用注册端点 `POST /register/user`，添加一个具有与您要测试的 GitHub 账户相同的用户名或电子邮件的用户。

然后，从 `GET /``auth/url` 端点提供的 GitHub URL 中检索令牌。

您将使用您的 favorite 工具中的令牌来查询 `GET /home` 端点，该端点使用 GitHub 令牌来验证权限。

在撰写本文时，我们无法使用交互式文档测试需要外部承载令牌的端点，因此请随意使用您喜欢的工具通过在头部授权中提供承载令牌来查询端点。

您也可以使用 shell 中的 `curl` 请求来完成，如下所示：

```py
$ curl --location 'http://localhost:8000/home' \
--header 'Authorization: Bearer <github-token>'
```

如果一切设置正确，您将收到以下响应：

```py
{"message":"logged in <your-username> !"}
```

您刚刚使用第三方应用程序（如 GitHub）实现了并测试了认证。其他提供者，如 Google 或 Twitter，遵循类似的程序，但有细微差别。您可以自由地实现它们。

## 参考信息

查看 GitHub 文档，它提供了如何设置 OAuth2 身份验证的指南：

+   *GitHub OAuth2* 集成：[`docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps`](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps)

您可以使用第三方授权登录，这些第三方提供类似配置。例如，您可以检查 Google 和 Twitter：

+   *Google OAuth2* 集成：[`developers.google.com/identity/protocols/oauth2`](https://developers.google.com/identity/protocols/oauth2)

+   *Twitter OAuth2* 集成：[`developer.twitter.com/en/docs/authentication/oauth-2-0`](https://developer.twitter.com/en/docs/authentication/oauth-2-0)

# 实现多因素认证（MFA）

多因素认证（MFA）通过要求用户提供两个或更多验证因素来访问资源，从而增加了一层安全性。本指南将指导您如何在 FastAPI 应用程序中添加 MFA，通过结合用户知道的东西（他们的密码）和他们拥有的东西（设备）来增强安全性。

## 准备工作

对于我们的 FastAPI 应用程序，我们将使用基于时间的**一次性密码**（**TOTP**）作为我们的多因素认证方法。TOTP 提供的是一个六到八位的数字，通常有效期为 30 秒。

首先，确保您已安装必要的软件包：

```py
$ pip install pyotp
```

**Pyotp**是一个 Python 库，实现了包括 TOTP 在内的一次性密码算法。

要使用 TOTP 认证，我们需要修改我们数据库中的用户表，以考虑用于验证密钥数的 TOTP 密钥。

让我们通过在`models.py`模块中的`User`类中添加`totp_secret`字段来修改它：

```py
class User(Base):
    # existing fields
    totp_secret: Mapped[str] = mapped_column(
        nullable=True
)
```

现在我们已经准备好实现多因素认证（MFA）。

## 如何操作...

让我们先创建两个辅助函数来生成认证器使用的 TOTP 密钥和 TOTP URI，步骤如下。

1.  我们在名为`mfa.py`的新模块中定义了这些函数：

    ```py
    import pyotp
    def generate_totp_secret():
        return pyotp.random_base32()
    def generate_totp_uri(secret, user_email):
        return pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email, issuer_name="YourAppName"
        )
    ```

    TOTP URI 也可以是二维码或链接的形式。

    我们将使用`generate_totp_secret`和`generate_totp_uri`函数来创建请求多因素认证的端点。

1.  端点将返回一个用于认证器的**TOTP URI**。为了展示机制，我们还将返回密钥数，在现实场景中，这是由认证器生成的数字：

    ```py
    from fastapi import (
        APIRouter,
        Depends,
        HTTPException,
        status,
    )
    from sqlalchemy.orm import Session
    from db_connection import get_session
    from operations import get_user
    from rbac import get_current_user
    from responses import UserCreateResponse
    router = APIRouter()
    @router.post("/user/enable-mfa")
    def enable_mfa(
        user: UserCreateResponse = Depends(
            get_current_user
        ),
        db_session: Session = Depends(get_session),
    ):
        secret = generate_totp_secret()
        db_user = get_user(db_session, user.username)
        db_user.totp_secret = secret
        db_session.add(db_user)
        db_session.commit()
        totp_uri = generate_totp_uri(secret, user.email)
        # Return the TOTP URI
        # for QR code generation in the frontend
        return {
            "totp_uri": totp_uri,
            "secret_numbers": pyotp.TOTP(secret).now(),
        }
    ```

1.  现在，我们可以创建验证密钥数的端点：

    ```py
    @app.post("/verify-totp")
    def verify_totp(
        code: str,
        username: str,
        session: Session = Depends(get_session),
    ):
        user = get_user(session, username)
        if not user.totp_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA not activated",
            )
        totp = pyotp.TOTP(user.totp_secret)
        if not totp.verify(code):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid TOTP token",
            )
        # Proceed with granting access
        # or performing the sensitive operation
        return {
            "message": "TOTP token verified successfully"
        }
    ```

如前所述，您需要在`main.py`中的`FastAPI`对象类中包含路由器，以用于所有之前的端点。

为了测试它，像往常一样，从终端启动服务器，运行以下命令：

```py
$ uvicorn main:app
```

确保您的数据库中有一个用户，转到交互式文档，并通过用户凭据调用`/user/enable-mfa`端点。您将获得包含 TOTP URI 和临时密钥数的响应，如下所示：

```py
{
  "totp_uri":
  "otpauth://totp/YourAppName:giunio%40example.com?secret=
  NBSUC4CFDUT5IEYX4IR7WKBTDTU7LN25&issuer=YourAppName",
  "secret_numbers": "853567"
}
```

记下用作`/verify-totp`端点参数的密钥数，您将获得以下响应：

```py
{
  "message": "TOTP token verified successfully"
}
```

您已经在 FastAPI 应用程序中实现了多因素认证，并通过确保即使用户的密码被泄露，攻击者仍然需要访问用户的第二个因素（运行 MFA 应用程序的设备）才能获得访问权限来增强了安全性。

## 参见

在官方文档中查看 Python One-Time Password 库：

+   *Python One-Time Password* *库*：[`pyauth.github.io/pyotp/`](https://pyauth.github.io/pyotp/)

# 处理 API 密钥认证

API 密钥认证是一种简单而有效的方法来控制对应用程序的访问。此方法涉及为需要访问您的 API 的每个用户或服务生成一个唯一的密钥，并要求在请求头中包含该密钥。

API 密钥可以通过多种方式生成，具体取决于所需的保密级别。

FastAPI 没有内置对 API 密钥认证的支持，但您可以使用依赖项或中间件轻松实现它。对于大多数用例，依赖项更灵活，因此我们将采用这种方法。

这个配方将向您展示一种基本但不够安全的方法来实现它。

## 准备工作

我们将继续开发我们的应用程序。然而，您也可以将此配方应用于从头开始的一个简单应用程序。

## 如何实现...

让我们创建一个`api_key.py`模块来存储处理 API 密钥的逻辑。该包将包含 API 列表和验证方法：

```py
from fastapi import HTTPException
from typing import Optional
VALID_API_KEYS = [
    "verysecureapikey",
    "anothersecureapi",
    "onemoresecureapi",
]
async def get_api_key(
    api_key: Optional[str]
):
    if (
        api_key not in VALID_API_KEYS
    ):
        raise HTTPException(
            status_code=403, detail="Invalid API Key"
        )
    return api_key
```

在示例中，密钥被硬编码到`VALID_API_KEYS`列表中。然而，在实际的生产场景中，密钥的管理和验证通常由专门的库或服务完成。

让我们创建一个使用 API 密钥的端点：

```py
from fastatpi import APIrouter
router = APIRouter()
@router.get("/secure-data")
async def get_secure_data(
    api_key: str = Depends(get_api_key),
):
    return {"message": "Access to secure data granted"}
```

现在，将路由器添加到`main.py`中的`FastAPI`对象类中，然后端点就准备好测试了。

通过运行以下命令启动服务器：

```py
$ uvicorn main:app
```

前往交互式文档`http://localhost:8000/docs`，并通过提供 API 密钥测试您刚刚创建的端点。

如您所见，通过向端点添加一个简单的依赖项，您可以使用 API 密钥保护您应用程序的任何端点。

## 还有更多...

我们已经开发了一个简单的模块来管理我们应用程序的 API。在生产环境中，这通常由托管平台提供的外部服务处理。然而，如果您打算实现自己的 API 管理系统，请记住 API 密钥认证的最佳实践：

+   **传输安全**：始终使用 HTTPS 以防止 API 密钥在传输过程中被拦截

+   **密钥轮换**：定期轮换 API 密钥以最小化密钥泄露的风险

+   **限制权限**：根据最小权限原则，为每个 API 密钥分配所需的最低权限

+   **监控和撤销**：监控 API 密钥的使用情况，并在检测到可疑活动时建立撤销机制

# 处理会话 cookie 和注销功能

管理用户会话并实现注销功能对于维护 Web 应用程序的安全性和用户体验至关重要。本配方展示了如何在 FastAPI 中处理会话 cookie，从用户登录时创建 cookie 到安全地终止注销会话。

## 准备工作

会话提供了一种在请求之间持久化用户数据的方式。当用户登录时，应用程序在服务器端创建一个会话，并将会话标识符发送到客户端，通常在一个**cookie**中。客户端将此标识符随每个请求发送回服务器，允许服务器检索用户的会话数据。

该配方将展示如何管理具有登录和注销功能的会话的 cookie。

## 如何实现...

FastAPI 中的 cookie 可以通过`Request`和`Response`对象类轻松管理。让我们创建一个登录和注销端点，将会话 cookie 附加到响应中，并从请求中忽略它。

让我们创建一个名为`user_session.py`的专用模块，并添加`/login`端点：

```py
from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session
from db_connection import get_session
from operations import get_user
from rbac import get_current_user
from responses import UserCreateResponse
router = APIRouter()
@router.post("/login")
async def login(
    response: Response,
    user: UserCreateResponse = Depends(
        get_current_user
    ),
    session: Session = Depends(get_session),
):
    user = get_user(session, user.username)
    response.set_cookie(
        key="fakesession", value=f"{user.id}"
    )
    return {"message": "User logged in successfully"}
```

由于我们需要验证`fakesession` cookie 已被创建，因此使用 Swagger 文档测试登录端点将不可行。

使用`uvicorn main:app`启动服务器，并使用 Postman 通过提供要登录的用户身份验证令牌来创建对`/login`端点的`Post`请求。

通过从响应部分的下拉菜单中选择**Cookies**，验证响应中是否包含`fakesession` cookie。

因此，我们可以定义一个不会在响应中返回任何会话 cookie 的注销端点：

```py
@router.post("/logout")
async def logout(
    response: Response,
    user: UserCreateResponse = Depends(
         get_current_user
    ),
):
    response.delete_cookie(
        "fakesession"
    )  # Clear session data
    return {"message": "User logged out successfully"}
```

这就是你需要管理会话的所有内容。

要测试`POST /logout`端点，使用`uvicorn`重新启动服务器。然后，在调用端点时，确保你在 HTTP 请求中提供了`fakesession` cookie 和用户的身份验证令牌。如果你之前调用了登录端点，它应该被自动存储；否则，你可以在请求的**Cookies**部分中设置它。

检查响应并确认响应中不再存在`fakesession` cookie。

## 还有更多...

除了基本配方之外，还有很多关于 cookie 可以学习。在实际环境中，你可以使用专门的库或外部服务。

无论你的选择是什么，都要将安全放在首位，并遵循以下实践来确保你的会话既安全又高效：

+   `Secure`、`HttpOnly`和`SameSite`用于防止**跨站请求伪造**（**CSRF**）和**跨站脚本**（**XSS**）攻击

+   **会话过期**：在会话存储中实现会话过期，并在 cookie 上设置最大年龄

+   **重新生成会话 ID**：在登录时重新生成会话 ID，以防止会话固定攻击

+   **监控会话**：实现机制来监控活动会话并检测异常

通过将会话管理和注销功能集成到你的 FastAPI 应用程序中，你确保了用户状态在请求之间得到安全且高效的管理。这增强了你应用程序的安全性和用户体验。请记住，遵循会话安全的最佳实践，以有效地保护用户及其数据。

在下一章中，我们将看到如何高效地调试你的 FastAPI 应用程序。

## 另请参阅

你可以在文档页面了解更多关于在 Fast 中管理 cookie 的信息：

+   *响应* *cookie*：[`fastapi.tiangolo.com/advanced/response-cookies/`](https://fastapi.tiangolo.com/advanced/response-cookies/)
