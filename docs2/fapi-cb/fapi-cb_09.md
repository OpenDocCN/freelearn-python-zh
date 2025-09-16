# 9

# 使用 WebSocket

在现代 Web 应用程序中，实时通信变得越来越重要，它使得聊天、通知和实时更新等交互式功能成为可能。在本章中，我们将探索令人兴奋的 WebSocket 世界，以及如何在 FastAPI 应用程序中有效地利用它们。从设置 WebSocket 连接到实现聊天功能和错误处理等高级功能，本章提供了构建响应式、实时通信功能的全面指南。到本章结束时，你将具备在 FastAPI 应用程序中创建 WebSocket 并促进实时通信的技能，从而实现交互式功能和动态用户体验。

在本章中，我们将涵盖以下食谱：

+   在 FastAPI 中设置 WebSockets

+   通过 WebSocket 发送和接收消息

+   处理 WebSocket 连接和断开

+   处理 WebSocket 错误和异常

+   使用 WebSocket 实现聊天功能

+   优化 WebSocket 性能

+   使用 OAuth2 保护 WebSocket 连接

# 技术要求

为了跟随 WebSocket 食谱，确保你的设置中包含以下基本要素：

+   **Python**：在你的环境中安装一个高于 3.9 版本的 Python。

+   **FastAPI**：应安装所有必需的依赖项。如果你在前几章中没有这样做，你可以简单地从你的终端执行：

    ```py
    $ pip install fastapi[all]
    ```

本章中使用的代码托管在 GitHub 上，网址为[`github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter09`](https://github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter09)。

建议在项目根目录中为项目设置一个虚拟环境，以有效地管理依赖项并保持项目隔离。

在你的虚拟环境中，你可以通过使用 GitHub 仓库中项目文件夹提供的`requirements.txt`文件一次性安装所有依赖项：

```py
$ pip install –r requirements.txt
```

由于交互式 Swagger 文档在撰写时有限，因此基本掌握**Postman**或其他测试 API 对于测试我们的 API 是有益的。

了解**WebSockets**的工作原理可能会有所帮助，尽管这不是必需的，因为食谱会引导你完成。

对于*使用 WebSockets 实现聊天功能*食谱，我们将编写一些基本的**HTML**，包括一些**JavaScript**代码。

# 在 FastAPI 中设置 WebSocket

WebSockets 提供了一种强大的机制，可以在客户端和服务器之间建立全双工通信通道，允许实时数据交换。在本食谱中，你将学习如何在 FastAPI 应用程序中建立 WebSocket 功能连接，以实现交互式和响应式的通信。

## 准备工作

在深入研究示例之前，请确保你的环境中已安装所有必需的包。你可以从 GitHub 仓库中提供的`requirements.txt`文件安装它们，或者使用`pip`手动安装：

```py
$ pip install fastapi[all] websockets
```

由于 swagger 文档不支持 WebSocket，我们将使用外部工具来测试 WebSocket 连接，例如 Postman。

你可以在网站上找到如何安装它的说明：[`www.postman.com/downloads/`](https://www.postman.com/downloads/).

免费社区版就足以测试这些示例。

## 如何操作…

创建一个名为`chat_platform`的项目根文件夹。我们可以在其中创建包含`main.py`模块的`app`文件夹。让我们按照以下方式构建我们的简单应用程序，其中包含 WebSocket 端点。

1.  我们可以从在`main.py`模块中创建服务器开始：

    ```py
    from fastapi import FastAPI
    app = FastAPI()
    ```

1.  然后我们可以创建 WebSocket 端点以连接客户端到聊天室：

    ```py
    from fastapi import WebSocket
    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text(
            "Welcome to the chat room!"
        )
        await websocket.close()
    ```

    端点与客户端建立连接，发送欢迎消息，并关闭连接。这是 WebSocket 端点最基本的配置。

就这样。要测试它，从命令行启动服务器：

```py
ws://localhost:8000/ws and click on Connect.
In the **Response** panel, right below the URL form, you should see the list of events that happened during the connection. In particular, look for the message received by the server:

```

欢迎来到聊天室！12:37:19

```py

 That means that the WebSocket endpoint has been created and works properly.
How it works…
The `websocket` parameter in the WebSocket endpoint represents an individual WebSocket connection. By awaiting `websocket.accept()`, the server establishes the connection with the client (technically called an `websocket.send_text()` sends a message to the client. Finally, `websocket.close()` closes the connection.
The three events are listed in the **Response** panel of Postman.
Although not very useful from a practical point of view, this configuration is the bare minimum setup for a WebSocket connection. In the next recipe, we will see how to exchange messages between the client and the server through a WebSocket endpoint.
See also
You can check how to create a WebSocket endpoint on the FastAPI official documentation page:

*   *FastAPI* *WebSockets*: [`fastapi.tiangolo.com/advanced/websockets/`](https://fastapi.tiangolo.com/advanced/websockets/)

At the time of writing, the Swagger documentation does not support WebSocket endpoints. If you spin up the server and open Swagger at `http://localhost:8000/docs`, you won’t see the endpoint we have just created. A discussion is ongoing on the FastAPI GitHub repository – you can follow it at the following URL:

*   *FastAPI WebSocket Endpoints Documentation* *Discussion*: [`github.com/tiangolo/fastapi/discussions/7713`](https://github.com/tiangolo/fastapi/discussions/7713)

Sending and receiving messages over WebSockets
WebSocket connections enable bidirectional communication between clients and servers, allowing the real-time exchange of messages. This recipe will bring us one step closer to creating our chat application by showing how to enable the FastAPI application to receive messages over WebSockets and print them to the terminal output.
Getting ready
Before starting the recipe, make sure you know how to set up a **WebSocket** connection in **FastAPI**, as explained in the previous recipe. Also, you will need a tool to test WebSockets, such as **Postman**, on your machine.
How to do it…
We will enable our chatroom endpoint to receive messages from the client to print them to the standard output.
Let’s start by defining the logger. We will use the logger from the `uvicorn` package (as we did in other recipes – see, for example, *Creating custom middlewares* in *Chapter 8*, *Advanced Features and Best Practices*), which is the one used by FastAPI as well. In `main.py`, let’s write the following:

```

导入 logging

logger = logging.getLogger("uvicorn")

```py

 Now let’s modify the `ws_endpoint` function endpoint:

```

@app.websocket("/ws")

async def ws_endpoint(websocket: WebSocket):

await websocket.accept()

await websocket.send_text(

"欢迎来到聊天室！"

)

while True:

data = await websocket.receive_text()

logger.info(f"Message received: {data}")

在上一个示例中，使用了`websocket.close()`调用并使用了一个无限循环。这允许服务器端持续接收来自客户端的消息并将其打印到控制台，而不会关闭连接。在这种情况下，只有客户端可以关闭连接。

这就是你需要从客户端读取消息并将其发送到终端输出的所有内容。

当客户端调用端点时，服务器会发起连接请求。使用`websocket.receive_text()`函数，服务器打开连接并准备好接收来自客户端的消息。消息存储在`data`变量中，并打印到终端输出。然后服务器向客户端发送确认消息。

让我们来测试一下。通过命令行运行`uvicorn app.main:app`启动服务器，然后打开 Postman。然后按照以下步骤操作。

1.  创建一个新的 WebSocket 请求，并连接到`ws://localhost:8000/ws`地址。

    一旦建立连接，你将在终端输出中看到以下消息：

    ```py
    Hello FastAPI application.On the output terminal you will the following message:

    ```

    INFO: Message received: Hello FastAPI application

    ```py

    While in the messages section of the client request you will see the new message:

    ```

    消息已接收！14:46:20

    ```py

    ```

    2.  然后，你可以通过点击 WebSocket **URL**字段右侧的**断开连接**按钮从客户端关闭连接。

通过使服务器能够接收来自客户端的消息，你刚刚通过 WebSocket 在客户端和服务器之间实现了双向通信。

参见

实际上，`Fastapi.WebSocket`实例是来自`send_json`或`receive_json`方法的`starlette.WebSocket`类。

更多信息请查看官方 Starlette 文档页面：

+   *Starlette* *WebSockets*: [`www.starlette.io/websockets/`](https://www.starlette.io/websockets/)

处理 WebSocket 连接和断开连接

当客户端与**FastAPI**服务器建立 WebSocket 连接时，适当地处理这些连接的生命周期至关重要。这包括接受传入的连接、维护活跃的连接以及优雅地处理断开连接，以确保客户端和服务器之间的通信顺畅。在这个配方中，我们将探讨如何有效地管理 WebSocket 连接并优雅地处理断开连接。

准备工作

要遵循这个配方，你需要有**Postman**或其他任何工具来测试 WebSocket 连接。此外，你需要在你的应用程序中已经实现了一个 WebSocket 端点。如果还没有，请检查前面的两个配方。

如何做到这一点...

我们将看到如何管理以下两种情况：

+   客户端端断开连接

+   服务器端断开连接

让我们详细查看这些情况中的每一个。

客户端端断开连接

你可能在*通过 WebSockets 发送和接收消息*的配方中注意到，如果连接从客户端（例如，从 Postman）在服务器控制台关闭，则会抛出一个未被捕获的`WebSocketDisconnect`异常。

这是因为客户端的断开连接应该在`try-except`块中适当处理。

让我们调整端点以考虑这一点。在`main.py`模块中，我们按照以下方式修改`/ws`端点：

```py
from fastapi.websockets import WebSocketDisconnect
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text(
        "Welcome to the chat room!"
    )
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Message received: {data}")
    except WebSocketDisconnect:
        logger.warning(
            "Connection closed by the client"
/ws, and then disconnect, you won’t see the error propagation anymore.
Server-side disconnection
In this situation, the connection is closed by the server. Suppose the server will close the connection based on a specific message such as the `"disconnect"` text string, for example.
Let’s implement it in the `/``ws` endpoint:

```

@app.websocket("/ws")

async def ws_endpoint(websocket: WebSocket):

await websocket.accept()

await websocket.send_text(

"欢迎来到聊天室！"

)

while True:

data = await websocket.receive_text()

logger.info(f"收到消息: {data}")

if data == "disconnect":

logger.warn("断开连接...")

await websocket.close()

将数据字符串内容传递给 websocket.close 方法并退出 while 循环。

如果你运行服务器，尝试连接到 WebSocket `/ws` 端点，并发送`"disconnect"`字符串作为消息，服务器将关闭连接。

你已经看到了如何管理 WebSocket 端点的断开和连接握手，然而，我们仍然需要为每个端点管理正确的状态码和消息。让我们在下面的配方中查看这一点。

处理 WebSocket 错误和异常

WebSocket 连接容易受到在连接生命周期中可能发生的各种错误和异常的影响。常见问题包括连接失败、消息解析错误和意外的断开连接。正确处理错误并与客户端正确通信对于维护响应和健壮的基于 WebSocket 的应用程序至关重要。在这个菜谱中，我们将探讨如何在 FastAPI 应用程序中有效地处理 WebSocket 错误和异常。

准备工作

该菜谱将展示如何管理特定端点可能发生的 WebSocket 错误。我们将展示如何改进在 *处理 WebSocket 连接和断开连接* 菜谱中定义的 `/ws` 端点。

如何实现...

在前一个示例中，`/ws` 端点编码的方式在服务器关闭连接时返回相同的响应代码和消息。就像 HTTP 响应一样，FastAPI 允许你个性化响应，向客户端返回更有意义的消息。

让我们看看如何实现。你可以使用以下类似解决方案：

```py
from fastapi import status
@app.websocket("/ws")
async def chatroom(websocket: WebSocket):
    if not websocket.headers.get("Authorization"):
        return await websocket.close()
    await websocket.accept()
    await websocket.send_text(
        "Welcome to the chat room!"
    )
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Message received: {data}")
            if data == "disconnect":
                logger.warn("Disconnecting...")
                return await websocket.close(
                    code=status.WS_1000_NORMAL_CLOSURE,
                    reason="Disconnecting...",
                )
    except WebSocketDisconnect:
        logger.warn("Connection closed by the client")
```

我们已经指定了 `websocket.close` 方法的状态码和原因，这些将被传输给客户端。

如果我们现在启动服务器并从客户端发送断开连接的消息，你将在响应窗口中看到断开连接的日志消息，如下所示：

```py
Disconnected from localhost:8000/ws 14:09:08
1000 Normal Closure:  Disconnecting...
```

这就是你优雅地断开 WebSocket 连接所需的所有内容。

替代方案

类似于如何为 HTTP 请求渲染 `HTTPException` 实例（参见 *第一章*，*使用 FastAPI 的第一步*）中的 *处理错误和异常* 菜谱，FastAPI 还允许使用 `WebSocketException` 来处理 WebSocket 连接，它将自动渲染为响应。

为了更好地理解，假设我们想要在客户端写入不允许的内容时断开连接——例如，`"bad message"` 文本字符串。让我们修改聊天室端点：

```py
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text(
        "Welcome to the chat room!"
    )
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Message received: {data}")
            if data == "disconnect":
                logger.warn("Disconnecting...")
                return await websocket.close(
                    code=status.WS_1000_NORMAL_CLOSURE,
                    reason="Disconnecting...",
                )
            if "bad message" in data:
                raise WebSocketException(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason="Inappropriate message"
                )
    except WebSocketDisconnect:
        logger.warn("Connection closed by the client")
```

如果你启动服务器并尝试发送包含 `"bad message"` 字符串的消息，客户端将会断开连接。此外，在你的 WebSocket 连接的 Postman 的 *响应* 面板部分，你将看到以下日志消息：

```py
Disconnected from localhost:8000/ws 14:51:40
1008 Policy Violation: Inappropriate message
```

你刚刚看到了如何通过抛出适当的异常来将 WebSocket 错误传达给客户端。你可以使用这种策略来处理在运行应用程序时可能出现的各种错误，并将它们正确地传达给 API 用户。

参见

与 HTTP 相比，WebSocket 是一种相对较新的协议，因此它仍在随着时间的推移而发展。尽管状态码不像 HTTP 那样被广泛使用，但你可以在以下链接中找到 WebSocket 代码的定义：

+   *WebSocket 关闭代码编号* 注册：[`www.iana.org/assignments/websocket/websocket.xml#close-code-number`](https://www.iana.org/assignments/websocket/websocket.xml#close-code-number)

您还可以在以下页面找到浏览器 WebSocket 事件的兼容性列表：

+   *WebSocket* 关闭事件：[`developer.mozilla.org/en-US/docs/Web/API/CloseEvent`](https://developer.mozilla.org/en-US/docs/Web/API/CloseEvent)

此外，FastAPI 中的 `WebSocketException` 类在官方文档链接中有文档说明：

+   *FastAPI WebSocketException API* 文档：[`fastapi.tiangolo.com/reference/exceptions/#fastapi.WebSocketException`](https://fastapi.tiangolo.com/reference/exceptions/#fastapi.WebSocketException)

使用 WebSockets 实现聊天功能

实时聊天功能是许多现代网络应用程序的常见功能，使用户能够即时相互沟通。在本配方中，我们将探讨如何在 FastAPI 应用程序中使用 WebSockets 实现聊天功能。

通过利用 WebSockets，我们将在服务器和多个客户端之间创建双向通信通道，允许实时发送和接收消息。

准备工作

要遵循配方，您需要对 WebSockets 有良好的理解，并知道如何使用 FastAPI 构建 WebSocket 端点。

此外，具备一些基本的 HTML 和 JavaScript 知识可以帮助创建简单的网页，用于应用程序。我们将使用的配方是聊天应用程序的基础。

还将使用 `jinja2` 包为 HTML 页面应用基本模板。请确保它在您的环境中。如果您没有使用 `requirements.txt` 安装包，请使用 `pip` 安装 `jinja2`：

```py
$ pip install jinja2
```

安装完成后，我们就可以开始配方了。

如何操作…

要构建应用程序，我们需要构建三个核心组件——WebSocket 连接管理器、WebSocket 端点和聊天 HTML 页面：

1.  让我们从构建连接管理器开始。连接管理器的角色是跟踪打开的 WebSocket 连接并向活跃的连接广播消息。让我们在 `app` 文件夹下的一个专用 `ws_manager.py` 模块中定义 `ConnectionManager` 类：

    ```py
    import asyncio
    from fastapi import WebSocket
    class ConnectionManager:
        def __init__(self):
            self.active_connections: list[WebSocket] = []
        async def connect(self, websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
        def disconnect(self, websocket: WebSocket):
            self.active_connections.remove(websocket)
        async def send_personal_message(
            self, message: dict, websocket: WebSocket
        ):
            await websocket.send_json(message)
        async def broadcast(
            self, message: json, exclude: WebSocket = None
        ):
            tasks = [
                connection.send_json(message)
                for connection in self.active_connections
                if connection != exclude
            ]
            await asyncio.gather(*tasks)
    ```

    `async def connect` 方法将负责握手并将 WebSocket 添加到活跃列表中。`def disconnect` 方法将从活跃连接列表中删除 WebSocket。`async def send_personal_message` 方法将向特定的 WebSocket 发送消息。最后，`async def broadcast` 将向所有活跃连接发送消息，除非指定了排除的连接。

    连接管理器将在聊天 WebSocket 端点中使用。

    2. 在一个名为 `chat.py` 的单独模块中创建 WebSocket 端点。让我们初始化连接管理器：

    ```py
    from app.ws_manager import ConnectionManager
    conn_manager = ConnectionManager()
    ```

    然后我们定义路由器：

    ```py
    from fastapi import APIRouter
    router = APIRouter()
    ```

    最后，我们可以定义 WebSocket 端点：

    ```py
    from fastapi import WebSocket, WebSocketDisconnect
    @router.websocket("/chatroom/{username}")
    async def chatroom_endpoint(
        websocket: WebSocket, username: str
    ):
        await conn_manager.connect(websocket)
        await conn_manager.broadcast(
            f"{username} joined the chat",
            exclude=websocket,
        )
        try:
            while True:
                data = await websocket.receive_text()
                await conn_manager.broadcast(
                    {"sender": username, "message": data},
                    exclude=websocket,
                )
                await conn_manager.send_personal_message(
                    {"sender": "You", "message": data},
                    websocket,
                )
        except WebSocketDisconnect:
            conn_manager.disconnect(websocket)
            await connection_manager.broadcast(
                {
                    "sender": "system",
                    "message": f"Client #{username} "
                    "left the chat",
                }
            )
    ```

    3.  当新客户端加入聊天后，连接管理器会向所有聊天参与者发送消息，通知他们新成员的到来。端点使用 `username` 路径参数检索客户端的名称。别忘了在 `main.py` 文件中将路由器添加到 FastAPI 对象中：

    ```py
    from app.chat import router as chat_router
    # rest of the code
    app = FastAPI()
    chatroom.xhtml should be stored in a templates folder in the project root. We will keep the page simple with the JavaScript tag embedded.The HTML part will look like this:

    ```

    <!doctype html>

    <html>

    <head>

    <title>聊天</title>

    </head>

    <body>

    <h1>WebSocket 聊天</h1>

    <h2>您的 ID: <span id="ws-id"></span></h2>

    <form action="" onsubmit="sendMessage(event)">

    <input

    type="text"

    id="messageText"

    autocomplete="off"

    />

    <button>发送</button>

    </form>

    <ul id="messages"></ul>

    <script>

    <!—js 脚本内容 -->

    <script/>

    </body>

    </html>

    ```py

    The `<script>` tag will contain the Javascript code that will connect to the WebSocket `/chatroom/{username}` endpoint with the client name as a parameter, send the message from the client page, receive messages from the server, and render the message text on the page in the messages list section.You can find an example in the GitHub repository, in the `templates/chatroom.xhtml` file. Feel free to make your own version or download it.
    ```

    4.  总结来说，我们需要构建返回 HTML 页面的端点。我们可以在同一个 `chat.py` 模块中构建它：

    ```py
    from fastapi.responses import HTMLResponse
    from fastapi.templating import Jinja2Templates
    from app.ws_manager import ConnectionManager
    conn_manager = ConnectionManager()
    templates = Jinja2Templates(directory="templates")
    @router.get("/chatroom/{username}")
    async def chatroom_page_endpoint(
        request: Request, username: str
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="chatroom.xhtml",
            context={"username": username},
        )
    ```

端点将作为路径参数接收客户端的用户名，该用户名将在聊天对话中显示。

你已经在你的 FastAPI 应用程序中设置了一个基本的聊天室，使用了 WebSockets 协议。你只需要使用 `uvicorn app.main:app` 启动服务器，并通过浏览器连接到 `http://localhost:8000/chatroom/your-username`。然后，从另一个页面，使用不同的用户名连接到相同的地址，并在两个浏览器之间开始交换消息。

它是如何工作的…

当连接到 `GET /chatroom/{username}` 端点地址（`http://localhost:8000/chatroom/{username}`）时，服务器将使用用户名来渲染定制的 HTML 页面。

HTML 将包含连接到 `/chatroom` WebSocket 端点的代码，并为每个用户创建一个新的 WebSocket 连接。

端点将使用 `ConnectionManager()` 连接管理器对象通过 HTML 页面在所有客户端之间交换消息。

相关内容

我们使用了 Jinja2 模板库的基本功能。然而，通过查看文档，你可以自由发挥创造力，发现这个包的潜力：

+   *Jinja2* 文档：[`jinja.palletsprojects.com/en/3.1.x/`](https://jinja.palletsprojects.com/en/3.1.x/)

优化 WebSocket 性能

WebSocket 连接为客户端和服务器之间的实时通信提供了强大的机制。为了确保 WebSocket 应用程序的最佳性能和可扩展性，实施有效的优化技术和测量方法至关重要。在本配方中，我们将了解如何基准测试 WebSocket 端点以测试支持的连接数，并提出实际的建议和技术来优化 FastAPI 应用程序中的 WebSocket 性能。

准备中

除了了解如何设置 WebSocket 端点外，我们还将使用“使用 WebSockets 实现聊天功能”的配方来基准测试支持的流量。你还可以通过将此策略应用于你的应用程序来遵循该配方。

无论您将其应用于应用程序还是聊天功能，在端点执行期间打印一些消息日志都可能很有用。

例如，对于 WebSocket `/chatroom/{username}` 端点，你可以在每次消息广播后添加日志，如下所示：

```py
import logging
logger = logging.getLogger("uvicorn")
@router.websocket("/chatroom/{username}")
async def chatroom_endpoint(
    websocket: WebSocket, username: str
):
    await conn_manager.connect(websocket)
    await conn_manager.broadcast(
        # method's parameters
    )
    logger.info(f"{username} joined the chat")
    try:
        while True:
            data = await websocket.receive_text()
            await conn_manager.broadcast(
                # method's parameters
            )
            await conn_manager.send_personal_message(
                # method's parameters
            )
            logger.info(
                f"{username} says: {data}"
            )
    except WebSocketDisconnect:
        conn_manager.disconnect(websocket)
        await conn_manager.broadcast(
            # method's paramters
        )
        logger.info(f"{username} left the chat")
```

我们现在准备好创建一个基准脚本以测试我们的聊天功能。

如何做到这一点...

让我们在根目录下创建一个脚本文件，并将其命名为 `benchmark_websocket.py`。一个典型的基准脚本应该执行以下任务：

+   定义一个运行 FastAPI 服务器的函数

+   定义另一个函数来连接 *n* 个 WebSocket 端点的客户端并交换一定数量的消息

+   通过在单独的进程中运行服务器并运行客户端来总结前面的步骤

创建脚本的步骤如下：

1.  让我们先定义一个函数来运行我们的服务器：

    ```py
    import uvicorn
    from app.main import app
    def run_server():
        uvicorn.run(app)
    ```

    `run_server` 函数是替代我们通常在终端中运行的命令行 `uvicorn app.main:app` 命令的另一种方法。

    2.  现在，让我们定义一个函数来创建一定数量的客户端，这些客户端将连接到 WebSocket 端点并交换一些消息：

    ```py
    import asyncio
    from websockets import connect
    async def connect_client(
        n: int, n_messages: int = 3
    ):
        async with connect(
            f"ws://localhost:8000/chatroom/user{n}",
        ) as client:
            for _ in range(n_messages):
                await client.send(
                    f"Hello World from user{n}"
                )
                await asyncio.sleep(n * 0.1)
            await asyncio.sleep(2)
    ```

    为了模拟并发连接模式，我们使用一个 `async def` 函数。这将使我们能够在高负载下评估服务器的性能。

    此外，我们在消息之间添加了一些异步睡眠时间 (`asyncio.sleep`) 来模拟聊天客户端的人类行为。

    3.  然后，我们可以将所有前面的函数在一个整体的 `async def main` 函数中执行，如下所示：

    ```py
    import multiprocessing
    async def main(n_clients: int = 10):
        p = multiprocessing.Process(target=run_server)
        p.start()
        await asyncio.sleep(1)
        connections = [
            connect_client(n) for n in range(n_clients)
        ]
        await asyncio.gather(*connections)
        await asyncio.sleep(1)
        p.terminate()
    ```

    该函数创建一个进程来启动服务器，启动它，等待一段时间以完成启动，并同时运行所有客户端以调用服务器。

    4.  最后，为了使其运行，如果作为脚本运行，我们需要将其传递给事件循环。我们可以这样做：

    ```py
    if __name__ == "__main__":
        asyncio.run(main())
    ```

要运行脚本，只需在命令行中将其作为 Python 脚本运行：

```py
n_clients, you will probably see all the messages flowing on the terminal. However, by increasing n_clients, depending on your machine, at some point, the script will not be able to run anymore and you will see socket connection errors popping up. That means that you passed the limit to support new connections with your endpoint.
What we did is the core of a basic script to benchmark. You can further expand the script based on your needs by adding timing or parametrization to have a broader view of your application’s capabilities.
You can also do the same by using dedicated test frameworks, similar to what we did in the *Performance testing for* *high traffic* *applications* recipe in *Chapter 5*, *Testing and Debugging FastAPI Applications*, for HTTP traffic.
There’s more…
Benchmarking your WebSocket is only the first step to optimize your application performance. Here is a checklist of actions that you can take to improve your application performance and reduce errors:

*   `TestClient` also supports WebSocket connections, so use it to ensure that the behavior of the endpoint is the one expected and does not change during the development process.
*   `try/except` blocks to handle specific error conditions. Also, when possible, use `async for` over `while True` when managing message exchanges. This will automatically capture and treat disconnection errors.
*   **Use connection pool managers**: Connection pool managers improve performance and code maintainability when handling multiple clients, such as in chat applications.

See also
You can see more on unit testing WebSockets with FastAPI in the official documentation:

*   *Testing WebSockets in* *FastAPI*: [`fastapi.tiangolo.com/advanced/testing-websockets/`](https://fastapi.tiangolo.com/advanced/testing-websockets/)

Securing WebSocket connections with OAuth2
Securing WebSocket connections is paramount to safeguarding the privacy and security of user interactions in real-time applications. By implementing authentication and access control mechanisms, developers can mitigate risks associated with unauthorized access, eavesdropping, and data tampering. In this recipe, we will see how to create a secure WebSocket connection endpoint with OAuth2 token authorization in your FastAPI applications.
Getting ready
To follow the recipe, you should already know how to set up a basic WebSocket endpoint – explained in the *Setting up WebSockets in FastAPI* recipe in this chapter.
Furthermore, we are going to use **OAuth2** with a password and a bearer token. We will apply the same strategy we used to secure HTTP endpoints in the *Securing your API with OAuth2* recipe in *Chapter 3*, *Building RESTful APIs with FastAPI*. Feel free to have a look before starting the recipe.
Before starting the recipe, let’s create a simple WebSocket endpoint, `/secured-ws`, in the `main.py` module:

```

@app.websocket("/secured-ws")

async def secured_websocket(

websocket: WebSocket,

username: str

):

await websocket.accept()

await websocket.send_text(f"欢迎 {username}！")

async for data in websocket.iter_text():

await websocket.send_text(

您写道：{data}

)

```py

 The endpoint will accept any connection with a parameter to specify the username. Then it will send a welcome message to the client and return each message received to the client.
The endpoint is insecure since it does not have any protection and can be easily reached. Let’s dive into the recipe to see how to protect it with OAuth2 authentication.
How to do it…
At the time of writing, there is no support for the `OAuth2PasswordBearer` class for WebSocket in FastAPI. This means that checking the bearer token in the headers for WebSocket is not as straightforward as it is for HTTP calls. However, we can create a WebSocket-specific class that is derived from the one used by HTTP to achieve the same functionality as follows.

1.  Let’s do it in a dedicated module under the `app` folder called `ws_password_bearer.py`:

    ```

    from fastapi import (

    WebSocket,

    WebSocketException,

    status,

    )

    from fastapi.security import OAuth2PasswordBearer

    class OAuth2WebSocketPasswordBearer(

    OAuth2PasswordBearer

    ):

    async def __call__(

    self, websocket: WebSocket

    ) -> str:

    authorization: str = websocket.headers.get(

    "authorization"

    )

    if not authorization:

    raise WebSocketException(

    code=status.HTTP_401_UNAUTHORIZED,

    reason="未认证",

    )

    scheme, param = authorization.split()

    if scheme.lower() != "bearer":

    raise WebSocketException(

    code=status.HTTP_403_FORBIDDEN,

    reason=(

    "无效的认证 "

    "credentials"

    ),

    )

    return param

    ```py

    We will use it to create a `get_username_from_token` function to retrieve the username from the token. You can create the function in a dedicated module – `security.py`.

     2.  Let’s define the `oauth2_scheme_for_ws` object:

    ```

    from app.ws_password_bearer import (

    OAuth2WebSocketPasswordBearer,

    )

    oauth2_scheme_for_ws = OAuth2WebSocketPasswordBearer(

    tokenUrl="/token"

    )

    ```py

     3.  The `tokenUrl` argument specifies the callback endpoint to call to retrieve the token. This endpoint should be built according to the token resolution you use. After that, we can create a function that retrieves the username from the token:

    ```

    def get_username_from_token(

    token: str = Depends(oauth2_scheme_for_ws),

    ) -> str:

    user = fake_token_resolver(token)

    if not user:

    raise WebSocketException(

    code=status.HTTP_401_UNAUTHORIZED,

    reason=(

    "无效的认证凭证"

    )

    )

    return user.username

    ```py

    The purpose of the `fake_token_resolver` function is to simulate the process of resolving a token. This function can be found in the `security.py` file in the GitHub repository of the chapter. Furthermore, the example contains only two users, `johndoe` and `janedoe`, who can be used later for testing. Also, the `security.py` module from the GitHub repository contains the `POST /token` endpoint to be used to retrieve the token.

    However, it is important to mention that this function does not provide any actual security and it is only used for example purposes. In a production environment, it is recommended to use a `/secured-ws`, in the `main.py` module:

    ```

    from import Annotated

    从 fastapi 导入 Depends

    from app.security 导入 get_username_from_token

    @app.websocket("/secured-ws")

    async def secured_websocket(

    websocket: WebSocket,

    username: Annotated[

    get_username_from_token, Depends()

    ]

    ):

    # 端点其余部分

    ```py

This is all you need to build a secured WebSocket endpoint.
To test it, spin up the server from the terminal by running the following:

```

$ uvicorn app.main:app

```py

 When attempting to connect to the WebSocket endpoint using Postman or another tool to the address `ws://localhost:8000/secured-ws`, an authorization error will occur, and the connection will be rejected before the handshake.
To allow the connection, we need to retrieve the token and pass it through the headers of the WebSocket request in `tokenized` string to the username. For example, for `johndoe`, the token would be `tokenizedjohndoe`.
Let’s pass it through the header. In Postman, you can pass the bearer token to the WebSocket request in the `Authorization` and value that will be `bearer tokenizedjohndoe`.
Now, if you try to connect, it should connect and you will be able to exchange messages with the endpoint.
You have just secured a WebSocket endpoint in FastAPI. By implementing OAuth2 authorization, you can enhance the security posture of your FastAPI applications and safeguard WebSocket communication against potential threats and vulnerabilities.
Exercise
Try to build a secure chat functionality where users need to log in to participate in the chat.
Tips: The endpoint that returns the HTML page should check for the bearer token in the cookies. If the cookie is not found or the bearer token is not valid, it should redirect the client to a login page that puts the token in the browser’s cookies.
You can use the `response.RedirectResponse` class from the `fastapi` package to handle redirections. The usage is quite straightforward and you can have a look at the documentation page at the link:
[`fastapi.tiangolo.com/advanced/custom-response/#redirectresponse`](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).
See also
Integrating `OAuth2PasswordBearer`-like class is a current topic of interest, and it is expected to evolve quickly over time. You can follow the ongoing discussion in the FastAPI GitHub repository:

*   *OAuth2PasswordBearer with WebSocket* *Discussion*: [`github.com/tiangolo/fastapi/discussions/8983`](https://github.com/tiangolo/fastapi/discussions/8983)

```

```py

```

```py

```
