# 11

# 中间件和 Webhooks

在本章中，我们将深入研究 FastAPI 中间件和 Webhooks 的高级和关键方面。FastAPI 中的中间件允许你在请求和响应到达路由处理器之前以及离开之后全局地处理它们。另一方面，Webhooks 允许你的 FastAPI 应用通过发送实时数据更新与其他服务进行通信。中间件和 Webhooks 对于构建健壮、高效和可扩展的应用程序至关重要。

我们将从探索如何从头开始创建自定义 **异步服务器网关接口**（**ASGI**）中间件开始。这将让你深入理解中间件在基本层面的工作原理。

接下来，我们将开发专门用于响应修改的中间件，允许你在将响应发送回客户端之前拦截和修改它们。

我们还将介绍如何使用中间件处理 **跨源资源共享**（**CORS**）。这对于需要安全地与不同域交互的应用程序尤为重要。最后，我们将深入探讨在 FastAPI 中创建 Webhooks，展示如何有效地设置和测试它们。

到本章结束时，你将全面了解如何在 FastAPI 应用程序中实现和使用中间件和 Webhooks。这些技能将使你能够构建更动态、响应更快、更集成的网络服务。

在本章中，我们将介绍以下配方：

+   创建自定义 ASGI 中间件

+   开发用于请求修改的中间件

+   开发用于响应修改的中间件

+   使用中间件处理 CORS

+   限制来自主机的传入请求

+   实现 Webhooks

# 技术要求

到这本书的这一阶段，你应该已经对 FastAPI 的基础知识有了很好的理解，包括如何安装它以及如何运行它。

本章中使用的代码托管在 GitHub 上，地址如下：[`github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter11`](https://github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter11)。

建议在项目根目录中为项目设置一个虚拟环境，以有效地管理依赖项并保持项目隔离。

在整个章节中，我们只将使用标准的 `fastapi` 库和 `uvicorn`。你可以通过在命令行中运行以下命令，在你的虚拟环境中使用 `pip` 安装所有依赖项：

```py
$ pip install fastapi uvicorn
```

对于 *使用中间件处理 CORS* 的配方，具备一些基本的 JavaScript 和 HTML 知识将有所帮助。

# 创建自定义 ASGI 中间件

ASGI 是一个用于 Python 网络服务器和应用程序之间通信的规范，旨在支持异步功能。中间件是网络应用中的一个关键组件，它提供了一种处理请求和响应的方式。

我们已经在 *第八章* 的 *创建自定义中间件* 菜谱中看到，如何创建自定义中间件。然而，这种技术依赖于 Starlette 库中的 `BasicHTTPMiddleware` 类，这是一个高级 HTTP 中间件的实现。

在这个菜谱中，我们将学习如何从头开始创建自定义 ASGI 中间件并将其集成到 FastAPI 应用程序中。这个中间件将很简单，它只会在终端上打印日志消息信息。

与 `BasicHTTPMiddleware` 类相比，这种方法提供了对请求/响应循环的更大控制，允许进行高级自定义，并创建任何类型的具有更深层次自定义的中间件。

## 准备工作

由于我们将使用 Starlette 库来构建中间件，因此对这一库有良好的了解将是有益的，尽管不是必需的。

关于开发环境，我们将仅使用 `fastapi` 包和 `uvicorn`。确保它们已安装在你的环境中。

## 如何做到这一点...

让我们从创建一个名为 `middleware_project` 的项目根目录开始。在根目录下，创建一个名为 `middleware` 的文件夹，其中包含一个名为 `asgi_middleware.py` 的模块。让我们从声明在中间件调用期间将使用的记录器开始：

```py
import logging
logger = logging.getLogger("uvicorn")
```

然后，我们可以定义中间件类如下：

```py
from starlette.types import (
    ASGIApp, Scope, Receive, Send
)
class ASGIMiddleware:
    def __init__(
        self, app: ASGIApp, parameter: str = "default"
):
        self.app = app
        self.parameter = parameter
    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ):
        logger.info("Entering ASGI middleware")
        logger.info(
            f"The parameter is: {self.parameter}"
        )
        await self.app(scope, receive, send)
        logger.info("Exiting ASGI middleware")
```

然后，我们需要在我们的应用程序中包含中间件。在项目根目录下，创建一个包含 FastAPI 类的 `main.py` 模块来运行应用程序，如下所示：

```py
from fastapi import FastAPI
from starlette.middleware import Middleware
from middleware.asgi_middleware import ASGIMiddleware
app = FastAPI(
    title="Middleware Application",
    middleware=[
        Middleware(
            ASGIMiddleware,
            parameter="example_parameter",
        ),
    ]
)
```

这就是你在 FastAPI 应用程序中实现自定义 ASGI 中间件所需的所有内容。

## 它是如何工作的...

要看到中间件的实际效果，让我们在 `main.py` 模块中创建一个通用的端点，如下面的示例所示：

```py
@app.get("/")
async def read_root():
    return {"Hello": "Middleware World"}
```

通过在命令行中运行 `uvicorn main:app` 来启动服务器。你会看到以下消息：

```py
INFO:    Started server process [2064]
INFO:    Waiting for application startup.
INFO:    Entering ASGI middleware
INFO:    The parameter is: example_parameter
```

在消息中，你会注意到那些表示我们已经进入中间件的消息。现在尝试调用根端点。你可以通过在浏览器中打开 `http://localhost:8000/` 来实现。

仍然在终端上，这次你会注意到进入和退出中间件的两个消息：

```py
INFO:    Entering ASGI middleware
INFO:    The parameter is: example_parameter
INFO:    127.0.0.1:55750 - "GET / HTTP/1.1" 200 OK
INFO:    Exiting ASGI middleware
```

根据日志，我们两次进入了中间件，一次是在启动时，一次是在调用端点时，但我们只退出了一次中间件。

这就是为什么 ASGI 中间件拦截应用程序的每个事件，不仅包括 HTTP 请求，还包括 `lifespan` 事件，这包括启动和关闭。

在中间件中存储的事件类型信息存储在 `__call__` 方法的 `scope` 参数中。让我们在 `ASGIMiddleware.__call__` 方法中包含以下日志，以改善我们对机制的理解：

```py
class ASGIMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        parameter: str = "default",
    ):
    # method content
    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ):
        # previous logs
        logger.info(
            f"event scope: {scope.get('type')}"
        )
        await self.app(scope, receive, send)
```

如果你重新启动服务器并重新调用 `http://localhost:8000/`，你现在将看到在服务器启动时指定事件作用域类型为 `lifespan`，在端点调用后为 `http` 的日志消息。

## 还有更多...

我们刚刚看到了如何将 ASGI 中间件作为一个类来构建。然而，你也可以利用函数装饰器模式来完成它。例如，你可以这样构建相同的中间件：

```py
def asgi_middleware(
    app: ASGIApp, parameter: str = "default"
):
    @functools.wraps(app)
    async def wrapped_app(
        scope: Scope, receive: Receive, send: Send
    ):
        logger.info(
            "Entering second ASGI middleware"
        )
        logger.info(
            f"The parameter you proved is: {parameter}"
        )
        logger.info(
            f"event scope: {scope.get('type')}"
        )
        await app(scope, receive, send)
        logger.info("Exiting second ASGI middleware")
    return wrapped_app
```

这是与在 *如何做…* 子部分中之前定义的 `ASGIMiddleware` 类等效的。为了使其工作，它应该以完全相同的方式作为参数传递给 FastAPI 实例：

```py
from middleware.asgi_middleware import asgi_middleware
app = FastAPI(
    title="Middleware Application",
    middleware=[
        Middleware(
            asgi_middleware,
            parameter="example_parameter",
        ),
    ]
```

根据你的个人喜好，你可以选择你喜欢的风格。然而，在本章的其余部分，我们将继续使用中间件类风格。

## 参见

你可以在专门的文档页面了解更多关于 ASGI 规范的信息：

+   *ASGI* 文档：[`asgi.readthedocs.io/en/latest/`](https://asgi.readthedocs.io/en/latest/)

FastAPI 中的中间件类是从 Starlette 库派生出来的。你可以在 Starlette 文档页面上找到创建 ASGI 中间件的详细文档：

+   *纯 ASGI* 中间件：[`www.starlette.io/middleware/#pure-asgi-middleware`](https://www.starlette.io/middleware/#pure-asgi-middleware)

# 开发请求修改的中间件

在网络应用程序中，中间件作为处理请求的强大工具。自定义中间件可以拦截并修改这些消息，允许开发者添加或修改功能。

在这个食谱中，我们将专注于开发自定义 ASGI 中间件，在将请求体发送到客户端之前对其进行哈希处理（如果需要）。这种方法提供了动态添加或更改响应头、请求体内容和其它属性的能力。到食谱结束时，你将能够开发自定义中间件来控制每个 API 请求。

## 准备工作

在我们开始之前，请确保你已经完成了之前的食谱，*创建自定义 ASGI 中间件*，以创建特定的自定义 ASGI 中间件。我们将使用 `middleware_project` 应用程序进行工作，但这个食谱可以轻松地应用到任何应用程序上。

在创建中间件之前，在 `main.py` 模块中，让我们创建一个 `POST /send` 端点，它接受请求中的请求体内容，如下所示：

```py
import logging
logger = logging.getLogger("uvicorn")
@app.post("/send")
async def send(message: str = Body()):
    logger.info(f"Message: {message}")
    return message
```

端点将打印请求体内容到终端，并将其作为响应返回。

现在我们有了端点，我们可以创建中间件来在发送到端点之前对请求体进行哈希处理。

## 如何做…

在 `middleware` 文件夹中，让我们创建一个名为 `request_middleware.py` 的模块，它将托管我们的中间件类。让我们按照以下步骤创建中间件：

1.  以这种方式开始模块，导入所需的模块：

    ```py
    from starlette.types import (
        ASGIApp, Scope, Receive, Send, Message,
    )
    from hashlib import sha1
    ```

    我们将使用 Starlette 库中的类型来创建中间件类，并使用 `sha1` 函数来哈希请求体。

1.  由于只有某些 `HTTP` 动词接受正文（例如 `POST` 和 `PUT`，但不包括 `GET`），我们将把应应用修改的路径作为参数传递给中间件。

    创建一个名为 `HashBodyContentMiddleware` 的中间件类，如下所示：

    ```py
    class HashBodyContentMiddleWare:
        def __init__(
            self, app: ASGIApp, allowed_paths: list[str]
        ):
            self.app = app
            self.allowed_paths = allowed_paths
    ```

    我们将把路径列表传递给 `allowed_paths` 参数。

1.  定义类的 `__call__` 方法：

    ```py
        async def __call__(
            self,
            scope: Scope,
            receive: Receive,
            send: Send,
        ):
            if (
                scope["type"] != "http"
                or scope["path"]
                not in self.allowed_paths
            ):
                await self.app(scope, receive, send)
                return
    ```

    如果事件不是一个 HTTP 请求或者路径不在列表中，中间件将不会采取任何行动，并将请求传递到下一步。

1.  关于正文的信息由 `receive` 变量提供。然而，`receive` 变量是一个协程，应该将其作为协程传递给 `self.app` 对象。我们将通过在函数中创建一个新的协程来克服这一点，如下所示：

    ```py
            # continue the __call__ method content
            async def receive_with_new_body() -> Message:
                message = await receive()
                assert message["type"] == "http.request"
                body = message.get("body", b"")
                message["body"] = (
                    f'"{sha1(body).hexdigest()}"'.encode()
                )
                return message
            await self.app(
                scope,
                receive_with_new_body,
                send,
            )
    ```

    正文请求将由传递给 FastAPI 对象应用后续步骤的协程修改。

1.  现在我们需要将中间件添加到 FastAPI 实例中。我们可以在 `main.py` 模块中完成此操作。但这次我们将利用 FastAPI 实例对象的 `add_middleware` 方法，如下所示：

    ```py
    app.add_middleware(
        HashBodyContentMiddleWare,
        allowed_paths=["/send"],
    )
    ```

    现在应用程序将使请求通过我们的中间件。

这就是实现它的全部内容。为了测试中间件，让我们通过命令行运行以下命令来启动 `uvicorn` 服务器：

```py
$ uvicorn main:app
```

然后转到 `http://localhost:8000/docs` 上的交互式文档，并测试 `POST/send` 端点。例如，检查你是否可以发送如下所示的正文字符串：

```py
"hello middleware"
```

如果一切操作正确，你应该会收到如下所示的响应正文：

```py
"14bb256ec4a292037c01bdbdd3eac61f328515f3"
```

你刚刚实现了自定义 ASGI 中间件，该中间件为指定的端点哈希正文。

这是一个简单的例子，但控制请求的潜力是无限的。例如，你可以用它来引入额外的安全层，以防止跨脚本注入不受欢迎的内容。

## 参见

在 Starlette 文档页面上有关于创建用于修改请求的中间件的说明：

+   *检查或修改请求*：[`www.starlette.io/middleware/#inspecting-or-modifying-the-request`](https://www.starlette.io/middleware/#inspecting-or-modifying-the-request)

# 开发用于响应修改的中间件

除了处理请求外，Web 应用程序中的中间件也是处理响应的强大工具。自定义中间件允许我们在响应返回给 API 调用者之前拦截响应。这可以用于检查响应内容或个性化响应。在这个配方中，我们将开发自定义 ASGI 中间件，为所有响应添加自定义头。

## 准备工作

我们将创建自定义 ASGI 中间件，以修改每个 HTTP 调用的响应。在我们开始这个配方之前，请查看 *创建自定义 ASGI 中间件* 配方。此外，这个配方将补充先前的配方，*为请求修改开发中间件*。

虽然你可以将这个配方应用到你的项目中，但我们将继续在 *为请求修改开发中间件* 配方中初始化的 `middleware_project` 项目上工作。

## 如何做到这一点...

我们将在 `middleware` 文件夹中的专用模块中创建我们的中间件类。我们将把这个模块命名为 `response_middleware.py`。让我们通过以下步骤开始构建中间件。

1.  让我们开始编写我们将用于定义中间件的导入语句：

    ```py
    from typing import Sequence
    from starlette.datastructures import MutableHeaders
    from starlette.types import (
        ASGIApp, Receive, Scope, Send, Message
    )
    ```

1.  然后，我们可以开始定义 `ExtraHeadersResponseMiddleware` 中间件类，如下所示：

    ```py
    class ExtraHeadersResponseMiddleware:
        def __init__(
            self,
            app: ASGIApp,
            headers: Sequence[tuple[str, str]],
        ):
            self.app = app
            self.headers = headers
    ```

1.  我们将把头部信息列表作为参数传递给中间件。然后，`__call__` 方法将如下所示：

    ```py
        async def __call__(
            self,
            scope: Scope,
            receive: Receive,
            send: Send,
        ):
            if scope["type"] != "http":
                return await self.app(
                    scope, receive, send
                )
    ```

1.  我们将中间件限制在 HTTP 事件调用上。类似于我们在之前的配方中看到的，*为请求修改开发中间件*，我们修改发送对象，它是一个协程，并将其传递给下一个中间件，如下所示：

    ```py
            async def send_with_extra_headers(
                message: Message
            ):
                if (
                    message["type"]
                    == "http.response.start"
                ):
                    headers = MutableHeaders(
                        scope=message
                    )
                    for key, value in self.headers:
                        headers.append(key, value)
                await send(message)
            await self.app(
                scope, receive, send_with_extra_headers
            )
    ```

    响应的头部信息是从 `send_with_extra_headers` 协程对象的 `message` 参数生成的。

1.  一旦定义了中间件，我们需要将其添加到 `FastAPI` 对象实例中，使其生效。我们可以在 `main.py` 模块中添加它，如下所示：

    ```py
    app.add_middleware(
        ExtraHeadersResponseMiddleware,
        headers=(
            ("new-header", "fastapi-cookbook"),
            (
                "another-header",
                "fastapi-cookbook",
            ),
        ),
    )
    ```

    在这里，我们向响应中添加了两个头部，`new-header` 和 `another-header`。

    为了测试它，通过运行 `uvicorn main:app` 启动服务器并打开交互式文档。调用其中一个端点并检查响应中的头部信息。

    在调用 `GET /` 端点时，你会得到以下头部信息列表：

    ```py
    another-header: fastapi-cookbook
    content-length: 28
    content-type: application/json
    date: Thu,23 May 2024 09:24:41 GMT
    new-header: fastapi-cookbook
    server: uvicorn
    ```

    你将找到我们之前添加到默认头部信息中的两个头部。

你刚刚实现了修改 API 响应的中间件。

## 参见

在 Starlette 文档中，你可以找到一个创建修改响应的中间件的示例：

+   *检查或修改* *响应*：[`www.starlette.io/middleware/#inspecting-or-modifying-the-response`](https://www.starlette.io/middleware/#inspecting-or-modifying-the-response)

# 使用中间件处理 CORS

CORS 是浏览器中实现的一种安全特性，用于防止恶意网站对来自不同源的主机上的 API 发起未经授权的请求。在构建 API 时，特别是对于公共消费的 API，正确处理 CORS 至关重要，以确保合法请求得到服务，而未经授权的请求被阻止。

在这个配方中，我们将探讨如何在 FastAPI 中使用自定义中间件来处理 CORS。这种方法使我们能够深入了解 CORS 机制，并获得灵活性，以适应特定的要求。

## 准备工作

我们将把这个配方应用到 `middleware_project` 应用程序上。确保你已经运行了 FastAPI 应用程序，并且至少已经设置了 `GET /` 端点。

由于这个配方将展示如何设置 CORS 中间件来管理 CORS，你需要一个简单的 HTML 网页来调用我们的 API。

您可以自己创建一个，或者从项目的 GitHub 仓库下载 `cors_page.xhtml` 文件。该文件是一个简单的 HTML 页面，它向 `http://localhost:8000/` 上的 FastAPI 应用程序发送请求，并在同一页面上显示响应。

在开始配方之前，通过运行 `uvicorn main:app` 启动您的 FastAPI 应用程序。要查看页面，使用现代浏览器打开 `cors_page.xhtml`。然后，打开开发者控制台。在大多数浏览器中，您可以通过右键单击页面，从菜单中选择 **Inspect**，然后切换到 **Console** 标签来做到这一点。

在页面上，按 **发送 CORS 请求** 按钮。您应该在命令行上看到如下错误消息：

```py
Access to fetch at 'http://localhost:8000/' from origin 'null' has been blocked by CORS policy: Response to preflight request doesn't pass access control check: No 'Access-Control-Allow-Origin' header is present on the requested resource. If an opaque response serves your needs, set the request's mode to 'no-cors' to fetch the resource with CORS disabled.
```

这意味着调用已被 CORS 策略阻止。

让我们开始这个配方，看看如何修复它。

## 如何实现...

在 FastAPI 中，可以使用来自 Starlette 库的专用 `CORSMiddleware` 类来处理 CORS。

让我们在 `main.py` 模块中添加中间件：

```py
from fastapi.middleware.cors import CORSMiddleware
# rest of the module
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

现在，重新运行服务器，再次打开 `cors_page.xhtml`，然后尝试按 **发送 CORS 请求** 按钮。这次，您将在页面上直接看到响应消息。

`allow_origins` 参数指定了允许 CORS 的主机源。如果 `allow_origins=[*]`，则表示允许任何源。

`allow_methods` 参数指定了允许的 HTTP 方法。默认情况下，只允许 `GET`，如果 `allow_methods=[*]`，则表示允许所有方法。

然后，`allow_headers` 参数指定了允许的头部。同样，如果我们使用 `allow_headers=[*]`，则表示允许所有头部。

在生产环境中，仔细评估这些参数对于确保安全标准和使您的应用程序安全运行非常重要。

这就是实现允许客户端 CORS 的 CORS 中间件所需的所有内容。

## 相关信息

关于 CORS 的更多信息，请查看 **Mozilla** 的文档页面：

+   *CORS*: [`developer.mozilla.org/en-US/docs/Web/HTTP/CORS`](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)

您可以在文档页面上了解更多关于 CORS 中间件的功能以及 FastAPI 中的其他参数：

+   *使用* *CORSMiddleware*: [`fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware`](https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware)

您还可以查看 Starlette 的文档页面：

+   *CORSMiddleware*: [`www.starlette.io/middleware/#corsmiddleware`](https://www.starlette.io/middleware/#corsmiddleware)

# 限制来自主机的请求

在现代网络应用程序中，安全性至关重要。安全性的一个关键方面是确保您的应用程序只处理来自受信任源请求。这种做法有助于减轻诸如 **域名系统** (**DNS**) 重绑定攻击等风险，攻击者会诱使用户的浏览器与未经授权的域进行交互。

FastAPI 提供了一个名为 `TrustedHostMiddleware` 的中间件，它允许你指定哪些主机被认为是可信的。来自任何其他主机的请求将被拒绝。本指南将指导你如何设置和使用 `TrustedHostMiddleware` 类，通过仅接受来自特定主机的请求来保护你的 FastAPI 应用。

## 准备工作

我们将把这个方法应用到 `middleware_project` 应用中。该应用至少需要一个端点来测试。

## 如何实现它...

让我们将请求限制为来自本地主机的调用。在 `main.py` 中，让我们导入 `TrustedHostMiddleware` 并将其添加到 FastAPI 对象实例应用中，如下所示：

```py
from fastapi.middleware.trustedhost import (
    TrustedHostMiddleware,
)
# rest of the module
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost"],
)
```

为了测试它，让我们尝试拒绝一个调用。让我们通过在运行 `uvicorn` 时指定未定义的主机地址 `0.0.0.0` 来启动服务器。如下所示：

```py
$ uvicorn main:app --host=0.0.0.0
```

这将使我们的应用对网络可见。

要获取你的机器在本地网络中的地址，你可以在 Windows 上运行 `ipconfig`，在 Linux 或 macOS 上运行 `ip addr`。

从连接到运行我们的 FastAPI 应用的同一本地网络的另一台设备（例如智能手机）打开浏览器，输入 `http://<你的本地地址>:8000`。如果一切设置正确，你将在浏览器中看到以下消息：

```py
Invalid host header
```

当你在运行 FastAPI 服务器的机器上时，你会看到如下类似的日志消息：

```py
INFO: <client ip>:57312 - "GET / HTTP/1.1" 400 Bad Request
```

这就是你设置中间件以防止你的应用被不受欢迎的主机访问所需的所有内容。

## 相关链接

你可以在 FastAPI 文档页面上了解更多关于 `TrustedHostMiddleware` 的信息：

+   *TrustedHostMiddleware*: [`fastapi.tiangolo.com/advanced/middleware/#trustedhostmiddleware`](https://fastapi.tiangolo.com/advanced/middleware/#trustedhostmiddleware)

由于 `TrustedHostMiddleware` 定义在 Starlette 库中，你还可以在以下链接的 Starlette 文档中找到它：

+   *TrustedHostMiddleware*: [`www.starlette.io/middleware/#trustedhostmiddleware`](https://www.starlette.io/middleware/#trustedhostmiddleware)

# 实现 Webhooks

**Webhooks** 在现代网络开发中发挥着至关重要的作用，通过使不同的系统能够实时通信和响应事件。它们本质上是由一个系统中的特定事件触发的 HTTP 回调，随后将消息或有效负载发送到另一个系统。这种异步事件驱动架构允许与第三方服务无缝集成、实时通知和自动化工作流程。了解如何有效地实现 Webhooks 将使你能够构建更互动和响应式的应用。

在这个配方中，我们将看到如何在 FastAPI 中创建 webhooks。我们将创建一个 webhook，它会通知每个 API 请求的 webhook 订阅者，就像一个监控系统。到这个配方的最后，您将能够实现一个健壮的 webhook 系统，在您的 FastAPI 应用程序中促进实时通信和与其他服务的集成。

## 准备工作

要设置用于向订阅者发送请求的 webhook，我们将使用自定义 ASGI 中介件。请确保您已经遵循了*创建自定义 ASGI 中介件*的配方。我们将继续在`middleware_project` API 上工作。但是，您将找到如何实现您的 webhook 的指南，这些指南可以轻松地适应您项目的特定需求。

如果您是从头开始创建一个新项目，请确保在您的环境中安装了带有`uvicorn`的`fastapi`包。您可以使用`pip`来完成此操作：

```py
$ pip install fastapi uvicorn
```

一旦有了这些包，我们就可以开始这个配方。

## 如何做到这一点...

在我们的 API 中构建 webhook 系统，我们需要做以下几步：

1.  设置 URL 注册系统。

1.  实现 webhook 回调。

1.  记录 webhook。

让我们通过实现来了解。

### 设置 URL 注册系统

一个 webhook 调用将向注册到 webhook 的 URL 列表发送 HTTP 请求。API 将需要一个 URL 注册系统。这可以通过创建一个专门的端点来实现，该端点将 URL 存储在有状态系统中，例如数据库。但是，为了演示目的，我们将 URL 存储在应用程序状态中，这对于小型应用程序来说可能也是一个不错的选择。

让我们通过以下步骤来创建它：

1.  在`main.py`中，让我们创建一个生命周期上下文管理器来存储已注册的 URL：

    ```py
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield {"webhook_urls": []}
    ```

1.  让我们将生命周期作为一个参数传递给 FastAPI 对象，如下所示：

    ```py
    app = FastAPI(
        lifespan=lifespan,
    # rest of the parameters
    )
    ```

1.  然后，我们可以创建一个端点来注册 URL，如下所示：

    ```py
    @app.post("/register-webhook-url")
    async def add_webhook_url(
        request: Request, url: str = Body()
    ):
        if not url.startswith("http"):
            url = f"http://{url}"
        request.state.webhook_urls.append(url)
        return {"url added": url}
    ```

    端点将接受正文中的文本字符串。如果字符串中缺少`http`或`https`协议，将在存储之前将`"http://"`字符串添加到 URL 前面。

您刚刚实现了 URL 注册系统。现在，让我们继续实现 webhook 回调。

### 实现 webhook 回调

在设置注册系统之后，我们可以开始创建 webhook 的调用。如前所述，这个特定的 webhook 将在每次 API 调用时提醒订阅者。我们将利用这些信息来开发专门的中介件来处理调用。让我们按照以下步骤进行：

1.  让我们在`middleware`文件夹中创建一个新的模块，名为`webhook.py`，并定义与订阅者通信的事件：

    ```py
    from pydantic import BaseModel
    class Event(BaseModel):
        host: str
        path: str
        time: str
        body: str | None = None
    ```

1.  然后，我们定义一个协程，它将被用来向订阅者 URL 发送请求，如下所示：

    ```py
    import logging
    from httpx import AsyncClient
    client = AsyncClient()
    logger = logging.getLogger("uvicorn")
    async def send_event_to_url(
        url: str, event: Event
    ):
        logger.info(f"Sending event to {url}")
        try:
            await client.post(
                f"{url}/fastapi-webhook",
                json=event.model_dump(),
            )
        except Exception as e:
            logger.error(
                "Error sending webhook event "
                f"to {url}: {e}"
            )
    ```

    客户端向 URL 发送请求。如果请求失败，将在终端打印一条消息。

1.  我们然后定义将拦截请求的中介件。我们首先进行导入，如下所示：

    ```py
    from asyncio import create_task
    from datetime import datetime
    from fastapi import Request
    from starlette.types import (
        ASGIApp, Receive, Scope, Send,
    )
    ```

    接着，我们添加`WebhookSenderMiddleware`类，如下所示

    ```py
    class WebhookSenderMiddleWare:
        def __init__(self, app: ASGIApp):
            self.app = app
        async def __call__(
            self,
            scope: Scope,
            receive: Receive,
            send: Send,
        ):
    ```

1.  我们将仅过滤 HTTP 请求，如下所示：

    ```py
            if scope["type"] == "http":
                message = await receive()
                body = message.get("body", b"")
                request = Request(scope=scope)
    ```

1.  我们继续在`same __call__`函数中定义要传递给 webhook 订阅者的`event`对象：

    ```py
                event = Event(
                    host=request.client.host,
                    path=request.url.path,
                    time=datetime.now().isoformat(),
                    body=body,
                )
    ```

1.  然后，我们通过运行`send_event_to_url`协程来遍历 URL 调用，如下所示：

    ```py
                urls = request.state.webhook_urls
                for url in urls:
                    await create_task(
                        send_event_to_url(url, event)
                    )
    ```

1.  我们通过将修改后的`receive`函数返回给应用程序来最终化该方法：

    ```py
                async def continue_receive():
                    return message
                await self.app(
                    scope, continue_receive, send
                )
                return
            await self.app(scope, receive, send)
    ```

    我们刚刚定义了将执行调用的中间件。

1.  现在我们需要在应用程序中导入`WebhookSenderMiddleWare`中间件。我们可以在`main.py`中这样做：

    ```py
    from middleware.webhook import (
    WebhookSenderMiddleWare
    )
    # rest of the code
    app.add_middleware(WebhookSenderMiddleWare)
    ```

    应用程序现在将包括我们的中间件来处理 webhook 回调。

这就是你在 FastAPI 应用程序中实现完整 webhook 所需的所有内容。

### 记录 webhook

向 API 用户提供关于 webhook 如何工作的文档是很重要的。FastAPI 允许我们在 OpenAPI 文档中记录 webhook。

要实现这一点，你需要创建一个空体的函数，并将其声明为 webhook 端点。你可以在`main.py`中这样做：

```py
@app.webhooks.post("/fastapi-webhook")
def fastapi_webhook(event: Event):
    """_summary_
    Args:
        event (Event): Received event from webhook
        It contains information about the
        host, path, timestamp and body of the request
    """
```

你也可以通过在`middleware/webhook.py`模块中的`Event`类中添加规范来提供一个示例的正文内容，如下所示：

```py
class Event(BaseModel):
    host: str
    path: str
    time: str
    body: str | None = None
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "host": "127.0.0.1",
                    "path": "/send",
                    "time": "2024-05-22T14:24:28.847663",
                    "body": '"body content"',
}
            ]
        }
uvicorn main:app command and opening the browser at http://localhost:8000/docs, you will find the documentation for POST /fastapi-webhook in the POST register-webhook-url endpoint.
How it works…
To test the webhook, you can set up a simple server running locally on a specific port. You can create one yourself or download the `http_server.py` file from the GitHub repository. This server will run on port `8080`.
Once you have set up the server, you can run it from the command line:

```

$ python ./http_server.py

```py

 Leave the server running and make sure the FastAPI application is running on a separate terminal.
Open the interactive documentation at `http://localhost:8000/docs`. Using the `POST /register-webhook-url` endpoint, add the `"localhost:8080"` address. Make sure you specify the correct port in the URL.
Now try to call any of the endpoints to the API. The FastAPI application will make a call to the server listening at port `8080`. If you check the service terminal, you will see the messages streaming on the terminal containing the information for each call.
There’s more…
While the basic implementation of webhooks is powerful, several advanced concepts and enhancements can make your webhook system more robust, secure, and efficient. Some of the most relevant ones are as follows:

*   **Authentication**: To ensure that your API can securely communicate with a webhook endpoint, you can implement any sort of authentication, from API to OAuth.
*   **Retry mechanism**: Webhooks rely on HTTP, which is not always reliable. There may be instances where the webhook delivery fails due to network issues, server downtime, or other transient errors. Implementing a retry mechanism ensures that webhook events are eventually delivered even if the initial attempt fails.
*   **Persistent storage**: Storing webhook events in a database allows you to keep an audit trail, troubleshoot issues, and replay events if necessary. You can use SQLAlchemy, a SQL toolkit and **object-relational mapping** library for Python, to save webhook events in a relational database.
*   **WebSocket webhook**: For real-time updates, you can set up a WebSocket server in FastAPI and notify clients through WebSocket connections when webhooks are received.
*   **Rate limiting**: To prevent abuse and server overload, rate limiting can be applied to the webhook endpoint. This ensures that a single client cannot overwhelm the server with too many requests in a short period.

Webhooks are crucial for constructing interactive, event-driven applications that seamlessly integrate with third-party systems. Utilize them to their fullest potential.
See also
If you want to learn more about webhook applications, check out the **Red Hat** blog page explaining what it is and how it is used in modern applications:

*   *What is a* *webhook?*: [`www.redhat.com/en/topics/automation/what-is-a-webhook`](https://www.redhat.com/en/topics/automation/what-is-a-webhook)

You can also refer to the FastAPI documentation for information on how to document webhook endpoints in the OpenAPI documentation:

*   *OpenAPI* *Webhooks*: [`fastapi.tiangolo.com/advanced/openapi-webhooks/`](https://fastapi.tiangolo.com/advanced/openapi-webhooks/)

```
