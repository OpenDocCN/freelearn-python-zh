

# 第二章：使用 Quart 创建可重用的后端

在上一章中，我们安装了开发应用程序所需的工具，这意味着我们可以开始构建后端。后端在服务器上运行，而前端在客户端的网页浏览器中运行。在我们的设置中，后端需要成为数据库和前端之间的接口，提供 API 以访问和编辑待办事项（见图 2.1）：

![图 2.1：所需设置的示意图](img/B18727_02_01.jpg)

图 2.1：所需设置的示意图

除了提供 API，后端还需要连接到数据库，管理用户会话，保护自身免受重负载和不正确使用的影响，并向用户发送电子邮件。在本章中，我们将构建具有这些功能的后端。本章结束时，我们将构建一个可重用的后端，任何 API 都可以使用它。或者，可以将这些功能分部分添加到您自己的应用程序中。

因此，在本章中，我们将涵盖以下主题：

+   创建基本的 Quart 应用程序

+   包含用户账户

+   保护应用程序

+   连接到数据库

+   发送电子邮件

# 技术要求

本章需要以下附加文件夹，并应创建它们：

```py
tozo
└── backend
    ├── src
    │   └── backend
    │       ├── blueprints
    │       ├── lib
    │       └── templates
    └── tests
        ├── blueprints
        └── lib
```

应该创建以下空文件：*backend/src/backend/__init__.py*，*backend/src/backend/blueprints/__init__.py*，*backend/src/backend/lib/__init__.py*，*backend/tests/__init__.py*，*backend/tests/blueprints/__init__.py*，和 *backend/tests/lib/__init__.py*。

要跟踪本章的开发过程，请使用配套仓库[`github.com/pgjones/tozo`](https://github.com/pgjones/tozo)，并查看标签 `r1-ch2-start` 和 `r1-ch2-end` 之间的提交。

# 创建基本的 Quart 应用程序

首先，我们可以创建一个基本的 API，它对请求做出简单的响应。我喜欢将此称为乒乓路由，因为请求是乒乓，响应是 pong。为此，我选择使用 Quart 框架。**Quart** 是一个带有扩展生态系统的小型 Web 框架，我们将使用它来添加额外的功能。

使用 Flask 作为替代方案

Quart 是非常流行的 Flask 框架的异步版本，它允许我们使用现代异步库。然而，如果您已经熟悉 Flask，您可以在本书中轻松地调整代码；有关更多信息，请参阅[`quart.palletsprojects.com/en/latest/how_to_guides/flask_migration.xhtml`](https://quart.palletsprojects.com/en/latest/how_to_guides/flask_migration.xhtml)。

要使用 Quart，我们首先需要通过在 *backend* 目录中运行以下命令来使用 `pdm` 添加它：

```py
pdm add quart
```

现在，我们可以在 *backend/src/backend/run.py* 文件中添加以下代码来创建一个 Quart 应用程序：

```py
from quart import Quart
app = Quart(__name__)
```

这允许我们添加函数，称为路由处理程序，当请求与给定的 HTTP 方法和路径匹配时，会调用这些函数并返回响应。对于我们的基本应用，我们希望对*GET /control/ping/*的请求做出响应。这是通过将以下代码添加到*backend/src/backend/run.py*中实现的：

```py
from quart import ResponseReturnValue
@app.get("/control/ping/")
async def ping() -> ResponseReturnValue:
    return {"ping": "pong"}
```

现在我们有了创建带有 ping 路由的应用的代码，我们应该设置工具，以便服务器在本地启动并处理请求。与后端工具一样，我们需要在*backend/pyproject.toml*中添加一个新的脚本名称，如下所示：

```py
[tool.pdm.scripts]
start = "quart --app src/backend/run.py run --port 5050"
```

上述代码允许在*backend*目录下运行时启动后端应用，如下所示：

```py
pdm run start
```

命令运行后，我们可以在任何目录中运行以下命令来检查 ping 路由是否工作：

```py
curl localhost:5050/control/ping/
```

或者，你可以在浏览器中输入[`localhost:5050/control/ping/`](http://localhost:5050/control/ping/)，如图 2.2 所示：

![图 2.2：在浏览器中访问时的控制 ping 路由](img/B18727_02_02.jpg)

图 2.2：在浏览器中访问时的控制 ping 路由

使用 curl

`curl` ([`curl.se/docs/manpage.xhtml`](https://curl.se/docs/manpage.xhtml)) 是一个优秀的命令行工具，用于发送 HTTP 请求。`curl`默认安装在大多数系统上，但如果你发现你没有它，你可以使用系统包管理器来安装它（`brew install curl`或`scoop install curl`）。

不带任何选项，`curl`执行一个`GET`请求，你可以使用`-X POST`选项切换到`POST`请求，或者你可以使用`--json ‘{“tool”: `curl`}’`选项发送 JSON 数据。

对于一个基本的后端来说，这已经足够了；然而，我们需要更多的功能，以及代码能够正常工作的更多确定性。我们将通过添加测试、使用蓝图、添加配置和确保一致的 JSON 错误响应来实现这一点。

## 测试 ping 路由

测试路由是否按预期工作是一种良好的实践。为此，我们可以在*backend/tests/test_run.py*中添加以下测试：

```py
from backend.run import app
async def test_control() -> None:
    test_client = app.test_client()
    response = await test_client.get("/control/ping/")
    assert (await response.get_json())["ping"] == "pong"
```

在测试代码就绪后，我们可以运行`pdm run test`来查看它是否运行并通过。

关于常见 await 错误的警告

我发现 Python 中错误地等待错误的事情很常见，而且似乎其他人也有这个问题。这个问题通常在如下代码中看到：

`await response.get_json()[“ping”]`。

这将因`couroutine cannot be indexed`错误而失败，因为`response.get_json()`返回的协程必须在索引之前等待。这个问题通过在正确的位置添加括号来解决，在这个例子中如下所示：

`(await response.get_json())[“ping”]`。

现在我们有一个工作的 ping-pong 路由，我们需要考虑如何添加更多的路由，为了清晰起见，最好使用蓝图来完成。

## 使用蓝图使代码更清晰

我们将 ping 路由处理程序添加到与应用程序相同的文件中（*backend/src/backend/run.py*），因为这是最简单的方法来启动；然而，随着我们添加更多的路由处理程序，文件将很快变得不清楚且难以更新。Quart 提供了蓝图来帮助随着应用程序变大而结构化代码。由于我们将添加更多的路由处理程序，我们将把到目前为止的内容转换为蓝图。

我们现在可以将 ping 路由处理程序移动到控制蓝图，通过向*backend/src/backend/blueprints/control.py*添加以下代码：

```py
from quart import Blueprint, ResponseReturnValue

blueprint = Blueprint("control", __name__)
@blueprint.get("/control/ping/")
async def ping() -> ResponseReturnValue:
    return {"ping": "pong"}
```

然后，我们可以通过将*backend/src/backend/run.py*更改为以下内容来将其注册到应用程序中：

```py
from quart import Quart
from backend.blueprints.control import blueprint as control_blueprint

app = Quart(__name__)
app.register_blueprint(control_blueprint)
```

现有的测试将继续工作；然而，我认为测试的位置应该覆盖它所测试的代码的位置。这使得理解测试在哪里以及测试应该测试什么变得更加容易。因此，我们需要将*backend/tests/test_run.py*移动到*backend/tests/blueprints/test_control.py*。

您现在应该有以下后端文件和结构：

```py
tozo
└── backend
    ├── pdm.lock
    ├── pyproject.toml
    ├── setup.cfg
    ├── src
    │   └── backend
    │       ├── blueprints
    │       │   ├── __init__.py
    │       │   └── control.py
    │       ├── __init__.py
    │       └── run.py
    └── tests
        ├── blueprints
        │   ├── __init__.py
        │   └── test_control.py
        └── __init__.py
```

我们将为应用程序中每个逻辑功能集合使用一个蓝图，并在整个结构中遵循此结构。我们现在可以专注于配置应用程序以在各种环境中运行。

## 配置应用程序

我们需要在多个环境中运行我们的应用程序，特别是开发、测试、CI 和生成环境。为此，我们需要在每个环境中更改一些设置；例如，数据库连接。配置允许我们更改这些设置而不必修改代码。它还允许将秘密与代码分开管理，从而更加安全。

我认为环境变量是提供配置的最佳方式，每个环境都有相同变量的不同值。我们可以指示 Quart 从带前缀的环境变量中加载配置。前缀确保只考虑相关的环境变量；默认前缀是`QUART_`，但我们将将其更改为`TOZO_`。为此，我们需要将以下更改添加到*backend/src/backend/run.py*，以便在创建应用程序后立即加载配置：

```py
app = Quart(__name__)
app.config.from_prefixed_env(prefix="TOZO")
```

应该已经存在高亮显示的代码行。

在生产中，我们将使用 Terraform 脚本来定义环境变量，而在本地，我们将从文件中加载环境变量。首先，对于开发，我们需要将以下内容添加到*backend/development.env*：

```py
TOZO_BASE_URL="localhost:5050" 
TOZO_DEBUG=true
TOZO_SECRET_KEY="secret key"
```

其次，对于测试，我们需要将以下内容添加到*backend/testing.env*：

```py
TOZO_BASE_URL="localhost:5050" 
TOZO_DEBUG=true
TOZO_SECRET_KEY="secret key" 
TOZO_TESTING=true
```

现在文件已经存在，我们可以通过将以下更改应用到*backend/pyproject.toml*来调整 PDM 脚本，以便在启动应用程序或运行测试时加载它们：

```py
[tool.pdm.scripts]
start = {cmd = "quart --app src/backend/run.py run --port   5050", env_file = "development.env"}
test = {cmd = "pytest tests/", env_file = "testing.env"}
```

这些对脚本的微小修改将确保在执行`pdm run start`和`pdm run test`命令时自动加载环境。现在，我们将查看一个经常被忽视的功能，即一致的错误响应。

## 确保错误响应是 JSON 格式

由于我们正在编写一个服务于 JSON 的后端 API，因此所有响应都应使用 JSON，包括错误响应。因此，我们不会使用 Quart 内置的错误响应，而是将使用我们自己的，这些响应会明确地产生 JSON 响应。

错误响应通常由 400-500 范围内的状态码表示。然而，仅状态码本身并不能总是传达足够的信息。例如，在注册新成员时，对于无效电子邮件地址的请求和密码强度不足的请求，都期望返回 400 状态码。因此，需要返回一个额外的代码来区分这些情况。我们可以通过在 *backend/src/backend/lib/api_error.py* 中添加以下代码来实现：

```py
class APIError(Exception):
    def __init__(self, status_code: int, code: str) -> None:
        self.status_code = status_code
        self.code = code
```

随着 `APIError` 的可用，我们现在可以通过在 *backend/src/backend/run.py* 中添加以下代码来告知 Quart 如何处理它：

```py
from quart import ResponseReturnValue
from backend.lib.api_error import APIError

@app.errorhandler(APIError)  # type: ignore
async def handle_api_error(error: APIError) -> ResponseReturnValue:
    return {"code": error.code}, error.status_code
```

我们还应该告知 Quart 如何处理任何其他意外错误，例如会导致 500 “内部服务器错误”响应的错误，如下所示：

```py
@app.errorhandler(500)
async def handle_generic_error(
    error: Exception
) -> ResponseReturnValue:
    return {"code": "INTERNAL_SERVER_ERROR"}, 500
```

我们现在已经设置了一个基本的 Quart 应用程序，以便我们能够添加我们实际应用程序所需的所有功能，首先是管理用户账户的能力。

# 包含用户账户

由于我们希望用户能够登录我们的应用程序，我们需要验证客户端是否是他们声称的那个人。之后，我们需要确保每个用户只能看到他们自己的待办事项。这通常是通过用户输入用户名和密码来实现的，然后这些密码会被与存储的版本进行核对。

我们需要验证用户对后端发出的每个请求；然而，理想情况下我们只希望用户登录一次（直到他们注销）。我们可以通过在用户登录时将信息保存到 cookie 中来实现这一点，因为浏览器会随后将 cookie 与每个请求一起发送。

当用户登录并开始会话时，我们需要将一些识别信息保存到 cookie 中；例如，他们的用户 ID。然后，我们可以在每次请求时读取 cookie 并识别是哪个用户。然而，客户端可以编辑或伪造 cookie，因此我们需要确保 cookie 中的信息没有被篡改。

我们可以通过在 cookie 中签名信息来防止篡改。签名是指使用一个密钥对数据进行加密函数处理以创建签名。然后，这个签名会与数据一起存储，允许存储的签名与重新计算版本进行核对。

Quart-Auth 是一个 Quart 扩展，它会为我们管理 cookie 和存储在其中的数据。在 *backend* 目录中运行以下命令即可安装 Quart-Auth：

```py
pdm add quart-auth
```

然后，在创建应用程序时，需要在 *backend/src/backend/run.py* 中激活 `AuthManager`，如下所示：

```py
from quart_auth import AuthManager
auth_manager = AuthManager(app)
```

虽然 Quart-Auth 为保护 cookie 提供了一套合理的默认设置，但我们的使用方式使我们能够更加安全。具体来说，我们可以利用严格的 SameSite 设置，而不是 Quart-Auth 默认的宽松设置。这是因为我们只需要对 API 路由的非导航请求进行身份验证。

SameSite

SameSite 设置确保 cookie 数据仅在来自指定域的请求中发送。这防止了其他网站使用 cookie 数据发起请求。要了解更多关于 SameSite 的信息，您可以点击以下链接：[`developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie/SameSite`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie/SameSite)。

要添加严格的 SameSite 设置，请将以下内容添加到 *backend/development.env* 和 *backend/testing.env* 文件中：

```py
TOZO_QUART_AUTH_COOKIE_SAMESITE="Strict"
```

然而，由于我们未使用 HTTPS，我们需要在开发环境中禁用安全 cookie 标志。这可以通过将以下内容添加到 *backend/development.env* 和 *backend/testing.env* 文件中来实现：

```py
TOZO_QUART_AUTH_COOKIE_SECURE=false
```

由于 Quart-Auth 管理会话，我们现在需要存储密码，确保它们足够强大，并允许无密码身份验证。

## 安全地存储密码

虽然我们现在可以管理用户的会话，但为了开始会话，用户需要通过提供电子邮件和密码来登录。虽然电子邮件可以直接存储在数据库中，但密码需要特别注意。这是因为用户经常在许多网站/服务中使用相同的密码，如果我们的应用程序发生泄露，我们可能会泄露对许多其他网站的访问权限。因此，我们不会直接存储密码，而是将密码哈希并存储。

**密码哈希**是将哈希操作应用于明文密码的结果。一个好的哈希操作应该确保生成的哈希值不能被转换回明文密码，并且每个不同的明文密码都会产生不同的哈希结果。

我喜欢使用`bcrypt`作为哈希操作，因为它同时满足这两个要求并且易于使用。在 *backend* 目录中运行以下命令即可安装`bcrypt`：

```py
pdm add bcrypt
```

安装了`bcrypt`之后，我们可以使用为每个密码生成的盐值来哈希密码，如下所示：

```py
import bcrypt
hashed = bcrypt.hashpw(password, bcrypt.gensalt(rounds=14))
```

检查提供的密码是否与哈希密码匹配是通过以下代码完成的：

```py
match = bcrypt.checkpw(password, hashed)
```

我们将在*第三章**，构建 API*中添加的登录和注册功能中使用`bcrypt`。接下来，我们需要检查密码是否足够强大。

对密码进行加盐处理

在散列密码时，最佳实践是在计算哈希之前向密码中添加`salt`。由于盐的目的是为每个存储的密码不同，因此它确保了在两个不同的实现中散列的相同密码具有不同的哈希。因此，添加盐是一个额外的安全措施，我们通过`bcrypt.gensalt`函数来实现。

## 确保密码强度

用户经常选择弱密码，这使得他们的账户容易受到攻击。为了防止这种情况，我们应该确保我们的用户选择强密码。为此，我喜欢使用`zxcvbn`，因为它给出了一个表示密码强度的分数。这是在*后端*目录中运行以下命令安装的：

```py
pdm add zxcvbn
```

然后，它被用来给出分数，如下所示：

```py
from zxcvbn import zxcvbn
score = zxcvbn(password).score
```

分数是介于 0 到 4 之间的值，我通常认为 3 或 4 的分数是好的。因此，我们将阻止使用分数较低的密码。

在下一章中添加注册和更改密码功能时，我们将使用`zxcvbn`。接下来，我们需要考虑用户在没有密码的情况下如何进行身份验证；例如，当他们忘记密码时。

## 允许无密码身份验证

有几种情况下，用户无法提供密码，但可以证明他们有权访问账户的电子邮件地址。一个典型的例子是当用户忘记他们的密码并希望重置它时。在这些情况下，我们需要向用户发送一个令牌，他们可以将其提供给我们，从而验证他们是电子邮件负责的用户。为此，令牌必须能够识别用户，恶意用户必须不能篡改令牌或创建自己的令牌。

要创建令牌，我们可以使用加密的`itsdangerous`对用户的 ID 进行签名，这也是 Quart-Auth 用于 cookie 的方法。`itsdangerous`是在*后端*目录中运行以下命令安装的：

```py
pdm add itsdangerous
```

由于这种方法没有加密签名数据，因此重要的是要记住，用户将能够阅读我们放在令牌中的任何内容。因此，我们不应在令牌中放置任何敏感信息（用户 ID 不是敏感信息）。

我们还将为我们的令牌添加时间戳；这样我们可以确保它们在特定时间段后过期。此外，由于我们希望能够在链接中使用令牌，我们需要使用`URLSafeTimedSerializer`。我们可以按照以下方式创建带有用户 ID 的令牌：

```py
from itsdangerous import URLSafeTimedSerializer
from quart import current_app
serializer = URLSafeTimedSerializer(
    current_app.secret_key, salt="salt"
)
token = serializer.dumps(user_id)
```

令牌可以按照以下方式读取和检查：

```py
from itsdangerous import BadSignature, SignatureExpired
from backend.lib.api_error import APIError
signer = URLSafeTimedSerializer(
    current_app.secret_key, salt="salt"
)
try:
    user_id = signer.loads(token, max_age=ONE_DAY)
except (SignatureExpired):
    raise APIError(403, "TOKEN_EXPIRED")
except (BadSignature):
    raise APIError(400, "TOKEN_INVALID")
else:
    # Use the user_id safely
```

由于我们使用的是定时令牌，因此在测试时我们需要控制时间。例如，如果我们想测试一个已过期的令牌，我们需要在令牌检查时创建它，使其过期。为此，我们可以使用`freezegun`，这是在*后端*目录中运行以下命令安装的：

```py
pdm add --dev freezegun
```

然后，我们可以在我们的测试中使用以下代码创建一个旧令牌：

```py
from freezegun import freeze_time
with freeze_time("2020-01-01"):
    signer = URLSafeTimedSerializer(        app.secret_key, salt="salt"    )
    token = signer.dumps(1)
```

此令牌然后可以用来测试路由处理程序对过期令牌的响应。

在下一章中，我们将添加忘记密码功能时，将使用`itsdangerous`和`freezegun`。

接下来，由于有恶意用户会尝试攻击我们的应用，我们需要保护它。

# 保护应用

在您将应用部署到生产环境后不久，用户最多会误用它，最坏的情况会攻击它。因此，从一开始就添加速率限制和请求验证进行防御是值得的。

速率限制限制了远程客户端向应用发送请求的速率。这防止了用户通过他们的请求过载应用，从而防止其他用户使用应用。

验证确保接收到的（或回复的）JSON 数据与预期的结构匹配。这很有帮助，因为它意味着如果 JSON 数据结构不正确，将显示错误消息。它还可以减轻用户发送导致错误或应用问题的结构的可能性。

## 添加速率限制

我们将使用名为 Quart-Rate-Limiter 的 Quart 扩展来强制执行速率限制，该扩展通过在`*backend*`目录中运行以下命令来安装：

```py
pdm add quart-rate-limiter
```

我们现在可以通过在`*backend/src/backend/run.py*`中添加以下代码来激活`RateLimiter`：

```py
from quart_rate_limiter import RateLimiter
rate_limiter = RateLimiter(app)
```

当激活`RateLimiter`后，应用中的任何路由都可以获得**速率限制**保护，例如，限制每分钟最多六次请求，如下所示：

```py
from datetime import timedelta

from quart_rate_limiter import rate_limit

@app.get("/")
@rate_limit(6, timedelta(minutes=1))
async def handler():
    ...
```

与其他错误一样，如果客户端超过速率限制，提供 JSON 响应很重要；我们可以在`*backend/src/backend/run.py*`中添加以下代码来实现：

```py
from quart_rate_limiter import RateLimitExceeded

@app.errorhandler(RateLimitExceeded)  # type: ignore
async def handle_rate_limit_exceeded_error(
    error: RateLimitExceeded,
) -> ResponseReturnValue:
    return {}, error.get_headers(), 429
```

现在我们能够添加速率限制，最佳实践是将它们添加到所有路由。为了确保我们这样做，让我们添加一个检查的测试。

## 确保所有路由都有速率限制

恶意攻击者通常会寻找缺少速率限制的路径作为他们可以攻击的弱点。为了减轻这种风险，我喜欢检查所有路由都有速率限制或被标记为豁免，使用`rate_exempt`装饰器。为此，我在`*tests/test_rate_limits.py*`中添加了以下代码：

```py
from quart_rate_limiter import (
    QUART_RATE_LIMITER_EXEMPT_ATTRIBUTE,
    QUART_RATE_LIMITER_LIMITS_ATTRIBUTE,
)
from backend.run import app
IGNORED_ENDPOINTS = {"static"}

def test_routes_have_rate_limits() -> None:
    for rule in app.url_map.iter_rules():
        endpoint = rule.endpoint

        exempt = getattr(
            app.view_functions[endpoint],
            QUART_RATE_LIMITER_EXEMPT_ATTRIBUTE,
            False,
        )
        if not exempt and endpoint not in IGNORED_ENDPOINTS:
            rate_limits = getattr(
                app.view_functions[endpoint],
                QUART_RATE_LIMITER_LIMITS_ATTRIBUTE,
                [],
            )
            assert rate_limits != []
```

在 Quart 应用中，规则是应用将响应的方法-路径组合。每个规则都有一个端点，指示哪个函数应该处理请求。静态端点是 Quart 添加的，因此我们在本次测试中忽略它。

此测试将检查应用中的所有路由都有速率限制或被豁免。这意味着我们还需要将`rate_exempt`装饰器添加到我们在设置基本应用时添加的控制 ping 端点。这通过在`*backend/src/backend/blueprints/control.py*`中的 ping 路由处理程序中添加突出显示的装饰器来完成，如下所示：

```py
from quart_rate_limiter import rate_exempt
@blueprint.get("/control/ping/")
@rate_exempt
async def ping() -> ResponseReturnValue:
    return {"ping": "pong"}
```

除了速率限制路由外，我们还可以通过验证请求和响应数据来保护路由。

## 添加请求和响应验证

恶意用户通常会尝试发送格式错误和无效的数据，以寻找我们代码中的错误。为了减轻这种影响，我们将使用名为 Quart-Schema 的 Quart 扩展来验证请求和响应。在 *backend* 目录中运行以下命令即可安装：

```py
pdm add "pydantic[email]" 
pdm add quart-schema
```

按照惯例，JSON（JavaScript/TypeScript）和 Python 使用不同的命名约定，前者使用 *camelCase*，后者 *snake_case*。这意味着在接收或回复时，我们需要在这两种命名约定之间进行转换。幸运的是，Quart-Schema 可以通过 `convert_casing` 选项自动为我们完成这项工作，无需额外思考。

我们可以通过向 *backend/src/backend/run.py* 添加以下代码来激活 `QuartSchema`，包括设置 `convert_casing` 选项：

```py
from quart_schema import QuartSchema
schema = QuartSchema(app, convert_casing=True) 
```

使用这种设置，我们可以使用 `dataclass` 来定义和验证路由期望接收的数据，以及验证它是否发送了正确的数据，如下所示：

```py
from quart_schema import validate_request, validate_response
@dataclass
class Todo:
    task: str
    due: Optional[datetime]
@app.post("/")
@validate_request(Todo)
@validate_response(Todo)
async def create_todo(data: Todo) -> Todo:
    ... 
    return data
```

与其他错误一样，如果客户端发送错误数据，提供包含有用信息的 JSON 响应给客户端非常重要。我们可以通过向 *backend/src/backend/run.py* 添加以下错误处理器来完成此操作：

```py
from quart_schema import RequestSchemaValidationError
@app.errorhandler(RequestSchemaValidationError)  # type: ignore
async def handle_request_validation_error(
    error: RequestSchemaValidationError,
) -> ResponseReturnValue:
    if isinstance(error.validation_error, TypeError):
        return {"errors": str(error.validation_error)}, 400
    else:
        return {"errors": error.validation_error.json()}, 400
```

检查 `validation_error` 的类型可以在响应中返回有用的信息，从而帮助纠正问题。

由于 Quart-Schema 为我们的应用添加了不受速率限制的路由，因此我们需要更改 *backend/tests/test_rate_limits.py* 中的 `IGNORED_ENDPOINTS` 行，如下所示：

```py
IGNORED_ENDPOINTS = {"static", "openapi", "redoc_ui", "swagger_ui"}
```

由于我们可以验证后端发送和接收的数据的结构，我们现在可以转向如何将数据存储在数据库中。为此，我们需要能够连接到它并执行查询。

# 连接到数据库

我们选择将应用所需的数据存储在 PostgreSQL 数据库中，我们需要连接到该数据库。为此，我喜欢使用名为 Quart-DB 的 Quart 扩展，它是一个围绕快速低级 PostgreSQL 驱动程序的优秀包装器。在 *backend* 目录中运行以下命令即可安装：

```py
pdm add quart-db
```

我们可以通过向 *backend/src/backend/run.py* 添加以下代码来激活 `QuartDB`：

```py
from quart_db import QuartDB
quart_db = QuartDB(app)
```

我们还需要配置 `QuartDB` 应连接到的数据库。这是通过添加一个 `TOZO_QUART_DB_DATABASE_URL` 环境变量来实现的，其值如下所示，其中高亮部分是可配置的：

```py
postgresql://username:password@0.0.0.0:5432/db_name
```

在开发中，我们将使用 `tozo` 作为用户名、密码和数据库名，因为它们非常明显且易于记忆。为此，请向 *backend/development.env* 添加以下内容：

```py
TOZO_QUART_DB_DATABASE_URL="postgresql://tozo:tozo@0.0.0.0:5432/tozo"
```

在测试时，我们将使用 `tozo_test` 作为用户名、密码和数据库名，以确保测试和开发数据保持分离。为此，请向 *backend/testing.env* 添加以下内容：

```py
TOZO_QUART_DB_DATABASE_URL="postgresql://tozo_test:tozo_test@0.0.0.0:5432/tozo_test"
```

在开发和测试更改后，我们需要将数据库重置到已知状态。在运行测试之前，我们也希望重置数据库以确保测试不会因为数据库处于不同状态而失败。为此，我们首先在`*backend/src/backend/run.py*`中添加以下代码以添加一个 Quart CLI 命令来重新创建数据库：

```py
import os
from subprocess import call  # nosec
from urllib.parse import urlparse 
@app.cli.command("recreate_db")
def recreate_db() -> None:
    db_url = urlparse(os.environ["TOZO_QUART_DB_DATABASE_URL"])
    call(  # nosec
        ["psql", "-U", "postgres", "-c", f"DROP DATABASE IF           EXISTS {db_url.path.removeprefix('/')}"],
    )
    call(  # nosec
        ["psql", "-U", "postgres", "-c", f"DROP USER IF EXISTS           {db_url.username}"],
    )
    call(  # nosec
        ["psql", "-U", "postgres", "-c", f"CREATE USER {db_url.       username} LOGIN PASSWORD '{db_url.password}' CREATEDB"],
    )
    call(  # nosec
        ["psql", "-U", "postgres", "-c", f"CREATE DATABASE {db_          url.path.removeprefix('/')}"],
    )
```

此命令通过`call`函数调用`psql`。前两个调用将使用`DROP DATABASE`和`DROP USER` SQL 命令删除已存在的数据库和用户。删除后，接下来的调用将使用`CREATE USER`和`CREATE DATABASE` SQL 命令创建用户和数据库。

我们现在可以利用此命令在`pdm run test`脚本中，并添加一个新的`pdm run recreate-db`脚本，以便按需重置数据库，通过在*backend/pyproject.toml*中做出以下更改：

```py
[tool.pdm.scripts]
recreate-db-base = "quart --app src/backend/run.py recreate_db"
recreate-db = {composite = ["recreate-db-base"], env_file =  "development.env"}
test = {composite = ["recreate-db-base", "pytest tests/"], env_  file = "testing.env"}
```

突出的行表示`test`脚本已被更改，而`recreate-db`和`recreate-db-base`脚本已被添加。

为了检查它是否工作，我们现在可以在*backend*目录中运行以下命令来创建开发数据库：

```py
pdm run recreate-db
```

然后，为了检查它是否工作，我们可以使用以下命令打开数据库的`psql` shell：

```py
psql –U tozo
```

前面的命令应该给出类似于*图 2.3*的输出：

![图 2.3：运行\dt 命令描述空数据库时 psql 的输出](img/B18727_02_04.jpg)

图 2.3：运行\dt 命令描述空数据库时 psql 的输出

PSQL

PSQL 是一个命令行工具，可以连接到 PostgreSQL 数据库，并允许运行查询和其他命令。这意味着您可以从命令行测试 SQL 查询并检查数据库的结构。我建议您尝试`\dt`命令，该命令列出数据库中的所有表，以及`\d tbl`命令，该命令描述名为*tbl*的表的结构。

在测试时，我们需要在 Quart 测试应用上下文中运行我们的测试，以确保建立了数据库连接。为此，我们需要在`*backend/tests/conftest.py*`中添加以下代码：

```py
from typing import AsyncGenerator

import pytest
from quart import Quart

from backend.run import app

@pytest.fixture(name="app", scope="function")
async def _app() -> AsyncGenerator[Quart, None]:
    async with app.test_app():
        yield app
```

`pytest`固定文件可以被注入到测试中，这意味着我们可以通过将其声明为参数来在测试中使用此固定文件。这意味着必须将`*backend/tests/blueprints/test_control.py*`重写如下：

```py
from quart import Quart
async def test_control(app: Quart) -> None:
    test_client = app.test_client()
    response = await test_client.get("/control/ping/")
    assert (await response.get_json())["ping"] == "pong"
```

另一个有用的功能是直接连接到数据库以在测试中使用。通过在`*backend/conftest.py*`中添加以下代码提供此固定文件：

```py
from quart_db import Connection
from backend.run import quart_db
@pytest.fixture(name="connection", scope="function")
async def _connection(app: Quart) -> AsyncGenerator[Connection, None]:
    async with quart_db.connection() as connection:
        async with connection.transaction():
            yield connection
```

在此设置完成后，我们所有的测试都可以使用应用固定文件，并对测试数据库运行测试。

除了连接到数据库外，我们还需要后端连接到邮件服务器以向用户发送邮件。

# 发送邮件

我们将向我们的应用程序用户发送电子邮件，首先是他们在注册时收到的确认邮件。如果用户忘记了密码，我们也会发送一封密码重置邮件。这些有针对性的电子邮件是事务性的，而不是营销性的，这是一个重要的区别，因为营销邮件很少通过应用程序代码发送。

对于**事务性电子邮件**，通常的目标是尽可能清晰地传达任务给用户。因此，这些电子邮件通常是基于文本的，图像最少。然而，我们应该确保电子邮件有品牌特色，并留有空间放置任何必需的法律文本。这意味着我们需要渲染电子邮件，使得事务性文本清晰，并围绕相关的品牌和文本。

## 渲染电子邮件

我们将把一封电子邮件视为由一个头部组成，我们将在这里放置品牌信息（例如，一个标志），内容部分将放置电子邮件的具体信息（例如，指向我们应用程序密码重置页面的链接），以及一个底部，其中放置任何法律信息。由于电子邮件之间只有内容会变化，我们可以考虑将头部和底部与内容分开渲染。

由于大多数电子邮件客户端支持 HTML，我们可以设计我们的电子邮件使其更具吸引力，更易于阅读。这意味着我们需要一个 HTML 头部/底部，我们可以将特定电子邮件的内容渲染到其中。这最好使用 Quart 内置的`render_template`函数来完成，该函数利用 Jinja2 来渲染模板文档。

要开始设置头部和底部，我们需要在*backend/src/backend/templates/email.xhtml*中放置以下代码：

```py
<!DOCTYPE html>
<html>
  <head>
    <title>Tozo - email</title>
    <meta http-equiv="Content-Type" content="text/html;      charset=UTF-8">
    <meta name="viewport" content="width=device-width, initial-      scale=1.0">
  </head>
  <body style="font-family: Arial, 'Helvetica Neue', Helvetica,    sans-serif; font-size: 14px; font-style: normal; margin: 0">
    <table width="100%" height="100%" cellpadding="0"       cellspacing="0" border="0">
      <tr>
        <td align="center">
          <table height="100%" cellpadding="20" cellspacing="0"            border="0" style="max-width: 540px;">
            <tr>
              <td align="left" width="540">
                {% block welcome %}
                  Hello,
                {% endblock %}
              </td>
            </tr>
            <tr>
              <td align="left" width="540">
                {% block content %}
                  Example content
                {% endblock %}
              </td>
            </tr>
            <tr>
              <td align="center" width="540">
                The Tozo team
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
```

由于电子邮件客户端只支持 HTML 和 CSS 的有限部分，我们使用表格来布局电子邮件。我们追求的布局是内容保持在视口中央的 540 像素宽度内。这应该支持大多数电子邮件客户端，同时仍然看起来不错。

突出的`block`指令在渲染时仅显示其内的内容，如图 2.4 所示。它允许任何扩展此基本电子邮件的模板替换块的内容，因此我们将以此作为所有电子邮件的基础。

![图 2.4：在浏览器中查看时渲染的电子邮件](img/Figure_2.4_NEW.jpg)

图 2.4：在浏览器中查看时渲染的电子邮件

caniemail.com

[caniemail.com](http://caniemail.com)网站是一个非常有价值的资源，用于检查现有的各种电子邮件客户端支持哪些 HTML 和 CSS 功能。我建议检查这个网站以了解添加到 HTML 电子邮件中的任何新功能。

现在我们有了漂亮的电子邮件，我们可以添加代码将它们发送到用户的电子邮件地址。

## 发送电子邮件

虽然直接从应用程序使用 SMTP 服务器发送电子邮件是可能的，但我发现使用像 Postmark ([`postmarkapp.com`](https://postmarkapp.com)) 这样的第三方服务是更好的实践。这是因为 Postmark 将确保我们的电子邮件从有助于确保低垃圾邮件评分的设置中可靠地发送，这是从新的 SMTP 服务器难以实现的。

在开发和测试中，我更喜欢不发送电子邮件，而是将它们记录下来。我发现这使开发更容易、更快（无需检查任何电子邮件收件箱）。我们可以通过从 `send_email` 函数开始，该函数通过将以下代码添加到 *backend/src/backend/lib/email.py* 中将电子邮件记录到控制台来实现：

```py
import logging
from typing import Any
from quart import render_template

log = logging.getLogger(__name__)
async def send_email(
    to: str, 
    subject: str, 
    template: str, 
    ctx: dict[str, Any], 
) -> None:
    content = await render_template(template, **ctx)
    log.info("Sending %s to %s\n%s", template, to, content)
```

我们还需要配置日志记录，这可以通过在 *backend/src/backend/run.py* 中添加以下代码通过基本设置来完成：

```py
import logging

logging.basicConfig(level=logging.INFO)
```

要使用第三方 Postmark 发送电子邮件，我们需要向他们的 API 发送 HTTP 请求。为此，我们可以在 *backend* 目录中运行以下命令来使用 `httpx`：

```py
pdm add httpx
```

然后，我们可以调整 `send_email` 函数，如果配置中可用令牌，通过更改 *backend/src/backend/lib/email.py* 中的代码通过 Postmark 发送邮件，如下所示：

```py
import logging
from typing import Any, cast

import httpx
from quart import current_app, render_template
log = logging.getLogger(__name__)

class PostmarkError(Exception):
    def __init__(self, error_code: int, message: str) -> None:
        self.error_code = error_code
        self.message = message
async def send_email(
    to: str, 
    subject: str, 
    template: str, 
    ctx: dict[str, Any], 
) -> None:
    content = await render_template(template, **ctx)
    log.info("Sending %s to %s\n%s", template, to, content)
    token = current_app.config.get("POSTMARK_TOKEN")
    if token is not None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.postmarkapp.com/email",
                json={
                    "From": "Tozo <help@tozo.dev>",
                    "To": to,
                    "Subject": subject,
                    "Tag": template,
                    "HtmlBody": content,
                },
                headers={"X-Postmark-Server-Token": token},
            )
        data = cast(dict, response.json())
        if response.status_code != 200:
            raise PostmarkError(                data["ErrorCode"], data["Message"]            )
```

`send_email` 函数现在使用 `httpx` 发送 POST 请求到 Postmark，包括作为头部的必需令牌和请求 JSON 主体中的电子邮件内容。Postmark 返回的任何错误都作为易于识别的 `PostmarkError` 抛出。我们现在可以专注于如何在测试中使用电子邮件。

## 测试电子邮件是否已发送

在测试后端的功能时，我们通常会想要检查是否已发送电子邮件。我们可以通过将以下代码添加到 *backend/tests/lib/test_email.py* 中来测试 `send_email` 函数：

```py
from pytest import LogCaptureFixture
from quart import Quart
from backend.lib.email import send_email
async def test_send_email(
    app: Quart, caplog: LogCaptureFixture
) -> None:
    async with app.app_context():
        await send_email(
            "member@tozo.dev", "Welcome", "email.xhtml", {}
        )  
    assert "Sending email.xhtml to member@tozo.dev" in caplog.text
```

`caplog` 是一个 `pytest` 修复程序，它捕获测试期间记录的所有内容。这允许我们通过查找特定文本来检查我们的电子邮件是否已记录。

现在已经设置了后端，我们已经拥有了开始开发应用程序 API 所需的一切。此阶段的文件夹结构如下：

```py
tozo
├── .github
│   └── workflows
├── backend
│   ├── src
│   │   └── backend
│   │       └── blueprints
│   │       └── lib
│   │       └── templates
│   └── tests
│       └── backend
│           └── blueprints
│           └── lib
├── frontend
│   ├── public
│   └── src
└── infrastructure
```

# 摘要

在本章中，我们使用 Quart 构建了一个后端应用程序，我们可以在其上构建特定的 API。它可以连接到数据库，管理用户会话，保护自己免受重和错误的使用，并向用户发送电子邮件。

本章中我们构建的功能在许多应用中都很常见，因此它们将对你正在尝试构建的应用程序很有用。此外，本章中构建的后端是通用的，可以适应你的特定用途。

在下一章中，我们将添加一个 API 来管理用户，允许用户登录，并支持我们在本书中构建的任务功能。

# 进一步阅读

在本书中，为了简化，我们选择只发送 HTML 电子邮件；然而，发送包含 HTML 和纯文本部分的复合电子邮件是更好的实践。你可以在 [`useplaintext.email`](https://useplaintext.email) 阅读对此的倡导。
