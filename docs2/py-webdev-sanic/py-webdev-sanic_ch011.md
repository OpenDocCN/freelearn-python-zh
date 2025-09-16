# 10 使用 Sanic 实现常见用例

我们已经学会了如何使用 Sanic，我们也学到了一些良好的实践和习惯。现在，让我们开始构建一些有趣的东西。当你开始一个新的项目时，从这里开始是非常诱人的。毕竟，你头脑中关于要构建的想法就是实现，对吧？你有一个最终的想法（比如聊天机器人）。所以，你坐下来开始构建机器人。

但本章放在书的末尾的原因是因为你显然不能从这里开始。只有在你获得了 HTTP 和 Sanic 的知识，并在过程中提升我们的技术技能之后，我们才能深入研究实现细节。本章的目标是查看一些你可能需要构建的实际功能。随着管道问题的解决，现在我们已经有了对 HTTP 和 Sanic 的稳固基础和理解，我们可以开始构建一些真实世界的用例。

在考虑本章要包含哪些主题时，我想到了一些我知道 Sanic 被用于的常见用例。选择一些经常出现的实现，然后尝试一起构建是有意义的。显然，这本书的空间有限，所以我们不会深入细节。相反，我们将查看一些实现细节，讨论一些考虑因素，并描述你可能会采取的一般方法来解决这个问题。我希望向你展示我在处理类似这样的项目时的思考过程。

就像上一章一样，仓库中将有大量代码，但书中并没有。这仅仅是因为它并不都与对话相关，但我想要确保你有完整的工作示例来作为你自己的项目的起点。为了获得完整的理解，你真的应该在线跟随源代码阅读本章。我会指出具体的设计决策，并包括一些特别值得提到的精选片段。

那么，我们将要构建什么？列表包括：

+   同步 WebSocket 数据流

+   渐进式 Web 应用后端

+   GraphQL API

+   聊天机器人

## 技术要求

由于本章建立在前面章节的基础上，你应该已经满足了所有的技术需求。我们将开始看到一些额外的第三方包的使用，所以请确保你有`pip`。

如果你想要提前跳过以确保你的环境已经设置好，以下是我们将要使用的 pip 包：

```py
$ pip instal aioredis ariande databases[postgresql] nextcord
```

此外，如果你还记得，在第二章中我们讨论了使用工厂模式。因为我们现在开始构建可能成为*真实世界*应用基础的东西，我觉得在这里使用一个可扩展的工厂模式会更好。因此，在这本书的剩余部分，你将看到越来越多的我们已经建立并使用的工厂模式的用法。

## WebSocket 数据流

在本书的早期，我们在第五章的“Websockets for two-way communication”部分探讨了 websockets。如果您还没有阅读该部分，我鼓励您现在去阅读。现在，我们将对我们的 websocket 实现进行扩展，创建一个水平可扩展的 websocket feed。这里代码的基本前提将与该部分相同，这就是为什么在继续到这里的示例之前，您应该了解我们在那里构建的内容。

我们将要构建的 feed 的目的是在多个浏览器之间共享发生的事件。基于第五章的示例，我们将添加一个第三方代理，这将使我们能够运行多个应用程序实例。这意味着我们可以水平扩展我们的应用程序。之前的实现存在一个问题，即它在内存中存储客户端信息。由于没有机制在多个应用程序实例之间共享状态或广播消息，因此一个 websocket 连接无法保证能够将消息推送到每个其他客户端。最多它只能将消息推送到恰好被路由并连接到同一应用程序实例的客户端。最终，这使得无法通过多个工作者扩展应用程序。

目标现在将是创建所谓的**pubsub**。这是一个意味着“发布和订阅”的术语，因为该模式依赖于多个源订阅中央代理。当这些源中的任何一个向代理发布消息时，所有已订阅的其他源都会接收到该消息。术语 pubsub 是对代理和源之间这种推送和拉取的简单描述。我们在构建 feed 时将使用这个概念。

在我看来，处理 pubsub 的最简单方法是使用 Redis，它有一个非常简单的内置 pubsub 机制。想法很简单：每个应用程序实例都将是一个源。在启动时，应用程序实例将订阅 Redis 实例上的特定频道。通过建立这种连接，现在它现在具有从该代理在特定频道上推送和拉取消息的能力。通过将此推送到第三方服务，所有我们的应用程序都将能够通过 pubsub 的推送和拉取访问相同的信息。

在第五章的 websockets 示例中，当收到消息时，服务器会将该消息推送到连接到同一应用程序实例的其他客户端。我们还将做类似的事情。浏览器客户端将使用多个 web 服务器之一打开一个 websocket 连接，该连接将保持与客户端的连接。这同样会被保存在内存中。当客户端实例有传入消息时，它不会直接将消息分发到其他客户端，而是将消息推送到 pubsub 代理。然后，所有其他实例都会接收到该消息，因为它们都订阅了同一个代理，并且可以将消息推送到任何连接到该应用程序实例的 websocket 客户端。这样，我们可以构建一个分布式 websocket 流。

要开始，我们将使用`docker-compose`启动一个 Redis 服务以及我们的开发应用程序。请查看仓库中的详细信息，了解如何完成此操作：____。我们将假设你有一个可用的 Redis 实例并且正在运行。

1.  我们首先创建一个 websocket 处理程序并将其附加到一个蓝图上。

    ```py
    from sanic import Blueprint
    from sanic.log import logger
    from .channel import Channel
    bp = Blueprint("Feed", url_prefix="/feed")
    @bp.websocket("/<channel_name>")
    async def feed(request, ws, channel_name):
        logger.info("Incoming WS request")
        channel, is_existing = await Channel.get(
            request.app.ctx.pubsub, request.app.ctx.redis, channel_name
        )
        if not is_existing:
            request.app.add_task(channel.receiver())
        client = await channel.register(ws)
        try:
            await client.receiver()
        finally:
            await channel.unregister(client)
    ```

    这是我们在这个示例中 Sanic 集成的全部内容。我们定义了一个 websocket 端点。该端点要求我们通过访问`channel_name`来访问一个源，这个`channel_name`意味着一个唯一的监听位置。这可以是用户名或聊天室股票行情等。重点是`channel_name`意味着代表你应用程序中的一个位置，人们将希望作为源持续从你的应用程序中检索信息。例如，这也可以用来构建一种共享编辑应用程序，其中多个用户可以同时更改同一资源。在这个示例中，处理程序通过获取一个`Channel`对象来工作。如果它创建了一个新的`Channel`，那么我们将向后台发送一个`receiver`任务，该任务负责监听我们的 pubsub 代理。处理程序中的下一件事是将我们的当前 websocket 连接注册到通道上，然后创建另一个`receiver`。第二个`client.receiver`的目的就是监听 websocket 连接，并将传入的消息推送到 pubsub 代理。

1.  让我们快速看一下`Client`对象。

    ```py
    from dataclasses import dataclass, field
    from uuid import UUID, uuid4
    from aioredis import Redis
    from sanic.server.websockets.impl import WebsocketImplProtocol
    @dataclass
    class Client:
        protocol: WebsocketImplProtocol
        redis: Redis
        channel_name: str
        uid: UUID = field(default_factory=uuid4)
        def __hash__(self) -> int:
            return self.uid.int
        async def receiver(self):
            while True:
                message = await self.protocol.recv()
                if not message:
                    break
                await self.redis.publish(self.channel_name, message)
        async def shutdown(self):
            await self.protocol.close()
    ```

    正如刚才所说的，这个对象的目的是在有消息时监听当前的 websocket 连接并将消息发送到 pubsub 代理。这是通过`publish`方法实现的。

1.  我们现在将查看`Channel`对象。这个类比`Client`类要长一些，所以我们将分部分查看其代码。打开 GitHub 仓库查看完整的类定义可能会有所帮助。

    ```py
    class ChannelCache(dict):
        ...
    class Channel:
        cache = ChannelCache()
        def __init__(self, pubsub: PubSub, redis: Redis, name: str) -> None:
            self.pubsub = pubsub
            self.redis = redis
            self.name = name
            self.clients: Set[Client] = set()
            self.lock = Lock()
        @classmethod
        async def get(cls, pubsub: PubSub, redis: Redis, name: str) -> Tuple[Channel, bool]:
            is_existing = False
            if name in cls.cache:
                channel = cls.cache[name]
                await channel.acquire_lock()
                is_existing = True
            else:
                channel = cls(pubsub=pubsub, redis=redis, name=name)
                await channel.acquire_lock()
                cls.cache[name] = channel
                await pubsub.subscribe(name)
            return channel, is_existing
    ```

每个应用程序实例都会在内存中创建并缓存一个频道。这意味着对于每个请求加入特定频道的传入请求的应用程序实例，只有一个该频道的实例被创建。即使我们有十个（10）应用程序实例，我们也不关心我们是否有十个（10）个频道的实例。我们关心的是在任何单个应用程序实例上，永远不会超过一个`Channel`订阅单个 Redis pubsub 频道。在同一个应用程序实例上有多个频道可能会变得混乱并导致内存泄漏。因此，我们还想添加一个机制，在频道不再需要时清理缓存。我们可以这样做：

```py
 async def destroy(self) -> None:
        if not self.lock.locked():
            logger.debug(f"Destroying Channel {self.name}")
            await self.pubsub.reset()
            del self.__class__.cache[self.name]
        else:
            logger.debug(f"Abort destroying Channel {self.name}. It is locked")
```

我们在这里使用`Lock`的原因是试图避免多个请求尝试销毁频道实例的竞态条件。

如果你记得上面提到的，在频道创建（或从缓存中检索）之后，我们在 Channel 实例上注册了 websocket 连接，看起来像这样：

```py
 async def register(self, protocol: WebsocketImplProtocol) -> Client:
        client = Client(protocol=protocol, redis=self.redis, channel_name=self.name)
        self.clients.add(client)
        await self.publish(f"Client {client.uid} has joined")
        return client
```

我们简单地创建`Client`对象，将其添加到需要在此实例上接收消息的已知客户端中，并发送一条消息让其他客户端知道有人刚刚加入。发布消息的方法看起来就像这样：

```py
 async def publish(self, message: str) -> None:
        logger.debug(f"Sending message: {message}")
        await self.redis.publish(self.name, message)
```

一旦客户端已注册，它还需要有注销的能力。注销的方法如下：

```py
 async def unregister(self, client: Client) -> None:
        if client in self.clients:
            await client.shutdown()
            self.clients.remove(client)
            await self.publish(f"Client {client.uid} has left")
        if not self.clients:
            self.lock.release()
            await self.destroy()
```

在这里，我们将当前客户端从`Channel`上的已知客户端中移除。如果不再有客户端监听此频道，那么我们可以关闭它并自行清理。

这是一个非常简单的模式，提供了巨大的灵活性。事实上，在我提供支持和帮助人们使用 Sanic 应用程序的过程中，我多次提供了使用与此类似模式构建应用程序的帮助。使用这个，你可以创建一些真正令人难以置信的前端应用程序。我知道我做到了。说到这里，在我们接下来的章节中，我们将开始探讨 Sanic 与在浏览器中运行的前端 Web 应用程序之间的相互作用。

## 支持渐进式 Web 应用

构建 Web API 的许多用例是为了支持**渐进式 Web 应用**（PWA，也称为单页应用，或 SPA）。像许多其他 Web 开发者一样，真正吸引我从事 Web 开发的原因是为了在浏览器中构建一个可用的应用程序或网站。让我们说实话，我们中很少有人会编写`curl`命令来使用我们最喜欢的 API。Web API 的真正力量在于它能够支持其他事物。

一个 PWA 需要什么才能运行？当你构建一个 PWA 时，最终产品是一堆静态文件。好吧，所以我们将这些文件放入一个名为`./public`的目录中，然后我们提供这些文件：

```py
app.static("/", "./public")
```

好了，我们现在正在运行一个 PWA。我们已经完成了。

好吧，不是那么快。能够提供静态内容很重要，但这不是你需要考虑的唯一因素。让我们看看构建 PWA 时需要考虑的一些因素。

### 处理子域名和 CORS

在第七章中，我们花了很多时间从安全的角度研究 CORS。我敢打赌，要求 CORS 保护的最大理由是需要为 PWA 提供服务。这类应用在互联网上无处不在，通常需要应对。这种情况通常发生的原因是，PWA 的前端和后端通常位于不同的子域名上。这通常是因为它们运行在不同的服务器上。静态内容可能由 CDN 提供，而后端位于 VPS 或 PAAS 提供（有关 Sanic 部署选项的更多信息，请参阅第八章）。

CORS 是一个大话题。它也是容易出错的东西。幸运的是，有一个简单的方法可以使用 Sanic Extensions 来实现这一点，这是一个由 Sanic 团队开发和维护的包，用于向 Sanic 添加一些额外功能。Sanic Extensions 专注于所有更具有意见和特定用例实现的特性，这些特性不适合核心项目。CORS 就是其中之一。

那么，我们如何开始呢？

```py
$ pip install sanic[ext]
```

或者

```py
$ pip install sanic sanic-ext
```

就这些。只需将`sanic-ext`包安装到你的环境中，你就可以获得开箱即用的 CORS 保护。截至版本 21.12，如果你环境中已有`sanic-ext`，Sanic 将为你自动实例化它。

我们现在唯一需要做的就是进行配置。通常，要开始进行 CORS 配置，我们需要设置允许的源：

```py
app.config.CORS_ORIGINS = "http://foobar.com,http://bar.com"
```

好吧，等等，你说，“我不能只从 Sanic 提供前端资源并避免 CORS，因为前后端都在同一台服务器上吗？”是的。如果这种方法对你有效，那就去做吧。让我们看看这可能是什么样子（从开发的角度来看）。

### 运行开发服务器

当你决定希望前端和后端应用都在同一台服务器上运行时会发生什么？或者，当你想使用上面显示的`app.static`方法来提供你的项目文件时？在本地构建这可能非常棘手。

这种情况之所以如此，是因为在构建前端应用时，你需要一个前端服务器。大多数框架都有某种构建要求。也就是说，你输入一些代码，保存后，然后像`webpack`或`rollup`这样的包会编译你的 JS，并通过本地开发 Web 服务器为你提供服务。你的前端开发服务器可能运行在 5555 端口上，所以你访问`http://localhost:5555`。

但是，你希望从前端应用程序访问本地运行的后端以填充内容。后端运行在 `http://localhost:7777`。哎呀，你看到这里的问题了吗？我们又回到了 CORS 的问题。只要你的前端应用程序是由与后端不同的服务器运行的，你将继续遇到 CORS 问题。

最终，我们试图让单个服务器同时运行后端和前端。由于我们谈论的是本地开发，我们还想为我们的 Python 文件和 JavaScript 文件提供自动重新加载功能。我们还需要触发 JavaScript 的重建，最后我们需要从这个位置提供所有服务。

幸运的是，Sanic 可以为我们做所有这些。现在让我们使用 Sanic 作为前端项目的本地开发服务器。

这将适用于你想要的任何前端工具，因为我们本质上将在 Python 中调用这些工具。我选择的前端开发框架是 Svelte，但你可以随意尝试使用 React、Vue 或其他许多替代方案。我不会带你设置前端项目的步骤，因为这里并不重要。想象一下你已经完成了。如果你想跟随代码，请参阅 GitHub 仓库：___。

为了实现我们的目标，我们将设置 Sanic 服务器，为前端应用程序的构建目录添加自动重新加载功能。对于使用 `rollup`（一个流行的 JS 构建工具）的 Svelte 项目，这是一个 `./public` 目录。

1.  我们首先声明静态文件的位置，并使用 `static` 来提供服务：

    ```py
    app = Sanic("MainApp")
    app.config.FRONTEND_DIR = Path(__file__).parent / "my-svelte-project"
    app.static("/", app.config.FRONTEND_DIR / "public")
    ```

1.  当我们运行 Sanic 时，请确保将此目录添加到自动重新加载器中，如下所示：

    ```py
    sanic server:app -d -p7777 -R ./my-svelte-project/src
    ```

1.  我们接下来想要做的是定义一个自定义信号。我们将稍后使用它，所以现在它只需要定义它。它只需要存在，这样我们就可以稍后等待事件。

    ```py
    @app.signal("watchdog.file.reload")
    async def file_reloaded():
    ...
    ```

1.  现在，我们准备构建一个将检查重新加载的文件并决定是否需要触发 `rollup` 构建过程的程序。我们将分两部分来看。首先，我们创建一个启动监听器，该监听器检查文件扩展名以确定服务器启动是由任何 `.svelte` 或 `.js` 文件扩展名的重新加载触发的。

    ```py
    @app.before_server_start
    async def check_reloads(app, _):
        do_rebuild = False
        if reloaded := app.config.get("RELOADED_FILES"):
            reloaded = reloaded.split(",")
            do_rebuild = any(
                ext in ("svelte", "js")
                for filename in reloaded
                if (ext := filename.rsplit(".", 1)[-1])
            )
    ```

    截至 21.12 版本，触发重新加载的文件被存储在 `SANIC_RELOADED_FILES` 环境变量中。由于任何以 SANIC_ 前缀开始的任何环境变量都被加载到我们的 `app.config` 中，因此如果存在，我们可以简单地读取该值并检查文件扩展名。假设需要重建，我们接下来想要触发对我们的 shell 的子进程调用以运行构建命令：

    ```py
     if do_rebuild:
            rebuild = await create_subprocess_shell(
                "yarn run build",
                stdout=PIPE,
                stderr=PIPE,
                cwd=app.config.FRONTEND_DIR,
            )
            while True:
                message = await rebuild.stdout.readline()
                if not message:
                    break
                output = message.decode("ascii").rstrip()
                logger.info(f"[reload] {output}")
            await app.dispatch("watchdog.file.reload")
    ```

    最后，当所有这些都完成时，我们将触发我们之前创建的定制事件。

1.  到目前为止，自动重新加载和自动重建都按预期工作。我们现在唯一缺少的是触发网页刷新的能力。这可以通过一个名为`livereload.js`的工具来实现。您可以通过搜索并安装 JavaScript 来访问 livereload.js。本质上，它将创建一个到端口 35729 的服务器的 websocket 连接。然后，从这个 websocket 中，您可以发送消息提示浏览器刷新。为了从 Sanic 中这样做，我们将运行嵌套应用程序。添加第二个应用程序定义：

    ```py
    livereload = Sanic("livereload")
    livereload.static("/livereload.js", app.config.FRONTEND_DIR / "livereload.js")
    ```

1.  我们还需要声明几个更多的常量。这些主要是为了运行 livereload 需要从服务器发送的两种类型的信息：

    ```py
    INDEX_HTML = app.config.FRONTEND_DIR / "public" / "index.html"
    HELLO = {
        "command": "hello",
        "protocols": [
            "http://livereload.com/protocols/official-7",
        ],
        "serverName": app.name,
    }
    RELOAD = {"command": "reload", "path": str(INDEX_HTML)}
    ```

1.  接下来，设置必要的监听器以运行嵌套服务器：

    ```py
    @app.before_server_start
    async def start(app, _):
        app.ctx.livereload_server = await livereload.create_server(
            port=35729, return_asyncio_server=True
        )
        app.add_task(runner(livereload, app.ctx.livereload_server))
    @app.before_server_stop
    async def stop(app, _):
        await app.ctx.livereload_server.close()
    ```

    上面代码中使用的`runner`任务应该看起来像这样：

    ```py
    async def runner(app, app_server):
        app.is_running = True
        try:
            app.signalize()
            app.finalize()
            await app_server.serve_forever()
        finally:
            app.is_running = False
            app.is_stopping = True
    ```

1.  是时候添加 websocket 处理器了：

    ```py
    @livereload.websocket("/livereload")
    async def livereload_handler(request, ws):
        global app
        logger.info("Connected")
        msg = await ws.recv()
        logger.info(msg)
        await ws.send(ujson.dumps(HELLO))
        while True:
            await app.event("watchdog.file.reload")
            await ws.send(ujson.dumps(RELOAD))
    ```

如您所见，处理器接受来自 livereload 的初始消息，然后发送一个`HELLO`消息回。之后，我们将运行一个循环并等待我们创建的自定义信号被触发。当它被触发时，我们发送 RELOAD 消息，这将触发浏览器刷新网页。

哇！现在我们已经在 Sanic 内部运行了一个完整的 JavaScript 开发环境。这对于那些希望从同一位置提供前端和后端内容的 PWA 来说是一个完美的设置。

既然我们已经在谈论前端内容，接下来我们将探讨前端开发者另一个重要的话题：GraphQL

## GraphQL

在 2015 年，Facebook 公开发布了一个旨在与传统 Web API 竞争并颠覆 RESTful Web 应用程序概念的项目。这个项目就是我们今天所知道的 GraphQL。这本书到目前为止都假设我们正在使用将 HTTP 方法与深思熟虑的路径相结合的传统方法来构建端点。在这种方法中，Web 服务器负责作为客户端和数据源（即数据库）之间的接口。GraphQL 的概念摒弃了所有这些，并允许客户端直接请求它想要接收的信息。有一个单一的端点（通常是`/graphql`）和一个单一的 HTTP 方法（通常是`POST`）。单一的路线定义旨在用于检索数据和在应用程序中引起状态变化。所有这些都在通过该单一端点发送的查询体中完成。GraphQL 旨在彻底改变我们构建 Web 的方式，并成为未来标准的实践。至少，这是许多人所说的将要发生的事情。

实际上并没有发生这种情况。在撰写本文时，GraphQL 的流行似乎已经达到顶峰，现在正在下降。尽管如此，我确实相信 GraphQL 在 Web 应用程序世界中填补了一个必要的空白，并且它将继续作为替代实现存在多年（只是不是作为替代品）。因此，我们确实需要了解如何将其与 Sanic 集成，以便在需要部署这些服务器的情况下。

在我们回答“为什么要使用 GraphQL？”这个问题之前，我们必须了解它是什么。正如其名称所暗示的，**GraphQL** 是一种查询语言。查询是一种类似于 JSON 的请求，用于以特定格式提供信息。一个希望从 Web 服务器接收信息的客户端可能会发送一个包含查询的 `POST` 请求，如下所示：

```py
{
  countries (limit: 3, offset:2) {
    name
    region
    continent
    capital {
      name
      district
    }
    languages {
      language
      isofficial
      percentage
    }
  }
}
```

作为回报，服务器将去获取它所需的数据，并编译一个符合该描述的返回 JSON 文档：

```py
{
  "data": {
    "countries": [
      {
        "name": "Netherlands Antilles",
        "region": "Caribbean",
        "continent": "North America",
        "capital": {
          "name": "Willemstad",
          "district": "Curaçao"
        },
        "languages": [
          {
            "language": "Papiamento",
            "isofficial": true,
            "percentage": 86.19999694824219
          },
          {
            "language": "English",
            "isofficial": false,
            "percentage": 7.800000190734863
          },
          {
            "language": "Dutch",
            "isofficial": true,
            "percentage": 0
          }
        ]
      },
      ...
    ]
  }
}
```

如你所见，这成为了一个非常强大的客户端工具，因为它可以将可能需要多个网络调用的信息捆绑成一个单一的操作。它还允许客户端（例如 PWA）以它需要的格式具体检索它所需的数据。

### 我为什么要使用 GraphQL？

我认为 GraphQL 是前端开发者的最佳拍档，但对于后端开发者来说却是噩梦。确实，使用 GraphQL 的 Web 应用程序通常比它们的替代品向 Web 服务器发出更少的 HTTP 请求。同样确实的是，前端开发者使用 GraphQL 操作来自 Web 服务器获取的响应会更加容易，因为他们可以成为数据结构架构的设计师。

GraphQL 提供了一种非常简单的数据检索方法。因为它是一个强类型规范，这使得可以拥有使生成查询过程非常优雅的工具。例如，许多 GraphQL 实现都附带了一个开箱即用的 Web UI，可用于开发。这些 UI 通常包括导航模式的能力，可以看到可以执行的所有类型的查询，以及可以检索的信息。见图 __ 中的示例。

插入图片

图 ___. 显示“模式”选项卡的 GraphQL UI 示例，该选项卡显示了所有可用的信息

在使用这些工具时，你可以玩得很开心，以构建你想要的确切信息。简单来说：GraphQL 使用简单，易于实现。当你开始构建临时的自定义查询时，它还具有非常令人满意的“酷”感。

除了在后台是个噩梦。尽管从客户端的角度来看简化了很多，但现在的网络服务器需要处理更高级别的复杂性。因此，当有人告诉我他们想要构建一个 GraphQL 应用程序时，我通常会问他们：为什么？如果他们是为了构建一个面向公众的 API，那么这可能是很棒的。GitHub 是一个很好的例子，它提供了一个面向公众的 GraphQL API，使用起来非常愉快。查询 GitHub API 简单直观。然而，如果他们是为了自己的内部用途构建 API，那么必须考虑一系列权衡。

GraphQL 并非在总体上比 REST 更容易或更简单。相反，它几乎将复杂性完全转移到网络服务器上。这可能是可以接受的，但这是一个你必须考虑的权衡。我通常发现后端复杂性的总体增加超过了实施带来的任何好处。

我知道这听起来可能像我不喜欢 GraphQL。这并不是真的。我认为 GraphQL 是一个很好的概念，并且我认为有一些非常出色的工具（包括 Python 世界中的工具）可以帮助构建这些应用程序。如果你想将 GraphQL 包含在你的 Sanic 应用程序中，我强烈推荐使用像 `Ariadne` ([`ariadnegraphql.org/`](https://ariadnegraphql.org/)) 和 `Strawberry` ([`strawberry.rocks/`](https://strawberry.rocks/)) 这样的工具。即使有了这些工具，在我看来，构建一个好的 GraphQL 应用程序仍然比较困难，而且还有一些陷阱等着吞噬你。当我们探讨如何构建 Sanic GraphQL 应用程序时，我会尝试指出这些问题，以便我们可以绕过它们。

### 将 GraphQL 添加到 Sanic

我为这一部分构建了一个小的 GraphQL 应用程序。当然，所有的代码都存放在这本书的 GitHub 仓库中：____。我强烈建议你在阅读时将代码准备好。坦白说，整个代码过于复杂和冗长，无法全部在这里展示。因此，我们将一般性地讨论它，并将具体细节指回仓库。为了方便起见，我在代码库本身中也添加了一些注释和进一步的讨论点。

当我们在第九章的 *To ORM or Not to ORM, that is the question* 节中讨论数据库访问时，我们讨论了是否应该实现 ORM。讨论的是你是否应该使用工具来帮助你构建 SQL 查询，还是自己构建它们。双方都有非常好的论点：支持 ORM 与反对 ORM。我选择了一种相对混合的方法，手动构建 SQL 查询，然后构建一个轻量级的实用工具来将数据填充到可用的模型中。

在这里也可以提出一个类似的问题：我应该自己构建还是使用一个包？我的回答是，你应该绝对使用一个包。我看不出有任何理由要尝试自己构建一个自定义实现。Python 中有几个不错的选择；我个人的偏好是 Ariadne。我特别喜欢这个包采用的 schema-first 方法。使用它允许我在 `.gql` 文件中定义我的应用程序的 GraphQL 部分，因此使我的 IDE 能够添加语法高亮和其他语言特定便利。

让我们开始：

1.  由于我们在示例中使用了 Ariadne，所以我们首先将其安装到我们的虚拟环境中：

    ```py
    $ pip install ariadne
    ```

1.  要启动 Ariadne 的“hello world”应用程序并不需要太多：

    ```py
    from ariadne import QueryType, graphql, make_executable_schema
    from ariadne.constants import PLAYGROUND_HTML
    from graphql.type import GraphQLResolveInfo
    from sanic import Request, Sanic, html, json
    app = Sanic(__name__)
    query = QueryType()
    type_defs = """
        type Query {
            hello: String!
        }
    """
    @query.field("hello")
    async def resolve_hello(_, info: GraphQLResolveInfo):
        user_agent = info.context.headers.get("user-agent", "guest")
        return "Hello, %s!" % user_agent
    @app.post("/graphql")
    async def graphql_handler(request: Request):
        success, result = await graphql(
            request.app.ctx.schema,
            request.json,
            context_value=request,
            debug=app.debug,
        )
        status_code = 200 if success else 400
        return json(result, status=status_code)
    @app.get("/graphql")
    async def graphql_playground(request: Request):
        return html(PLAYGROUND_HTML)
    @app.before_server_start
    async def setup_graphql(app, _):
        app.ctx.schema = make_executable_schema(type_defs, query)
    ```

如您所见，有两个端点：

+   一个显示 GraphQL 查询构建器的 `GET` 请求

+   一个 `POST` 请求，是 GraphQL 后端的入口

从这个简单的起点开始，你可以根据你的心愿构建 Sanic 和 Ariadne。让我们看看你可能采取的一种潜在策略。

1.  放弃上述内容，我们开始创建一个结构与我们之前看到的非常相似的程序。创建 `./blueprints/graphql/query.py` 并放置您的根级 GraphQL 对象。

    ```py
    from ariadne import QueryType
    query = QueryType()
    ```

1.  现在，我们在我们的 GraphQL 蓝图实例内部创建所需的两个端点：

    ```py
    from sanic import Blueprint, Request, html, json
    from sanic.views import HTTPMethodView
    from ariadne.constants import PLAYGROUND_HTML
    bp = Blueprint("GraphQL", url_prefix="/graphql")
    class GraphQLView(HTTPMethodView, attach=bp, uri=""):
        async def get(self, request: Request):
            return html(PLAYGROUND_HTML)
        async def post(self, request: Request):
            success, result = await graphql(
                request.app.ctx.schema,
                request.json,
                context_value=request,
                debug=request.app.debug,
            )
            status_code = 200 if success else 400
            return json(result, status=status_code)
    ```

    如您所见，这几乎与之前的简单版本相同。

1.  在这个相同的蓝图实例上，我们将放置所有我们的启动逻辑。这使所有内容都位于一个方便的位置，并允许我们一次性将其附加到我们的应用程序实例上。

    ```py
    from ariadne import graphql, make_executable_schema
    from world.common.dao.integrator import RootIntegrator
    from world.blueprints.cities.integrator import CityIntegrator
    from world.blueprints.countries.integrator import CountryIntegrator
    from world.blueprints.languages.integrator import LanguageIntegrator
    @bp.before_server_start
    async def setup_graphql(app, _):
        integrator = RootIntegrator.create(
            CityIntegrator,
            CountryIntegrator,
            LanguageIntegrator,
            query=query,
        )
        integrator.load()
        integrator.attach_resolvers()
        defs = integrator.generate_query_defs()
        additional = integrator.generate_additional_schemas()
        app.ctx.schema = make_executable_schema(defs, query, *additional)
    ```

你可能想知道，什么是集成器，所有这些代码都在做什么。这就是我将要向您推荐查看特定细节的仓库的地方，但我们将在这里解释这个概念。

在我的应用程序示例中，`Integrator` 是一个存在于领域内的对象，它是设置 Ariadne 可使用的 GraphQL 模式的通道。

在 GitHub 仓库中，您将看到最简单的集成器是为 `languages` 模块设计的。它看起来像这样：

```py
from world.common.dao.integrator import BaseIntegrator
class LanguageIntegrator(BaseIntegrator):
    name = "language"
```

旁边有一个名为 `schema.gql` 的文件：

```py
type Language {
    countrycode: String
    language: String
    isofficial: Boolean
    percentage: Float
}
```

然后，`RootIntegrator` 的任务是整合所有不同的领域，并使用动态生成的模式和如上所示的硬编码模式为 Ariadne 生成模式。

我们还需要为我们的 GraphQL 查询创建一个起始点。一个查询可能看起来像这样：

```py
 async def query_country(
        self, _, info: GraphQLResolveInfo, *, name: str
    ) -> Country:
        executor = CountryExecutor(info.context.app.ctx.postgres)
        return await executor.get_country_by_name(name=name)
```

用户创建一个查询，然后我们从数据库中获取它。这里的 Executor 与 `hikingapp` 中的工作方式完全相同。请参阅第 ___ 章。因此，有了这样的查询，我们现在可以将 GraphQL 查询转换为对象。

```py
{
  country(name: "Israel") {
    name
    region
    continent
    capital {
      name
      district
    }
    languages {
      language
      isofficial
      percentage
    }
  }
}
```

利用 GraphQL 的力量，我们的响应应该是这样的：

```py
{
  "data": {
    "country": {
      "name": "Israel",
      "region": "Middle East",
      "continent": "Asia",
      "capital": {
        "name": "Jerusalem",
        "district": "Jerusalem"
      },
      "languages": [
        {
          "language": "Hebrew",
          "isofficial": true,
          "percentage": 63.099998474121094
        },
        {
          "language": "Arabic",
          "isofficial": true,
          "percentage": 18
        },
        {
          "language": "Russian",
          "isofficial": false,
          "percentage": 8.899999618530273
        }
      ]
    }
  }
}
```

Ariadne（以及其他 GraphQL 实现）的工作方式是，你定义一个强类型模式。有了对这个模式的了解，你可能会得到嵌套对象。例如，上面的 `Country` 模式可能看起来像这样：

```py
type Country {
    code: String
    name: String
    continent: String
    region: String
    capital: City
    languages: [Language]
}
```

`Country`类型有一个名为`capital`的字段，它是一个`City`类型。由于这不是一个简单可以序列化为 JSON 的标量值，我们需要告诉 Ariadne 如何翻译或解析这个字段。根据 GitHub 中的示例，我们需要像这样查询我们的数据库：

```py
class CountryIntegrator(BaseIntegrator):
    name = "country"
    async def resolve_capital(
        self,
        country: Country,
        info: GraphQLResolveInfo
    ) -> City:
        executor = CityExecutor(info.context.app.ctx.postgres)
        return await executor.get_city_by_id(country.capital)
```

这就是我们在不同对象之间跟随路径的方法。然后，Ariadne 的任务是将所有这些不同的查询和解析器拼凑在一起，生成一个最终要返回的对象。这是 GraphQL 的力量。

你可能也注意到了一个缺陷。因为每个解析器都旨在独立操作，并将单个字段转换为值，所以你很容易从数据库中检索过量的数据。这尤其在你有一个所有对象都解析到同一数据库实例的数组时更为明显。这被称为“n+1”问题。虽然这不是 GraphQL 特有的问题，但许多 GraphQL 系统的设计使其特别容易受到这个问题的影响。如果你忽略这个问题，当响应单个请求时，你的服务器可能会反复从数据库请求相同的信息，尽管它应该已经有了这些信息。

许多应用程序都存在这个问题。它们依赖的数据库查询比实际需要的要多得多。所有这些过度检索累积起来，降低了 Web 应用程序的性能和效率。虽然你在开发任何应用程序时都应该意识到这个问题，并且要有意识地去处理，但我认为在 GraphQL 实现中，你必须特别计划这一点，因为 GraphQL 依赖于简化的解析器。因此，我在构建这类应用程序时能提供的最重要的建议就是考虑基于内存、基于请求的缓存。也就是说，在请求实例上缓存对象可能会节省大量的 SQL 查询。

我鼓励你花些时间审查 GitHub 仓库中的其余代码。有一些有用的模式可以在实际应用中使用。由于它们不一定与 Sanic 或 Sanic 中的 GraphQL 实现直接相关，我们在这里暂时停止讨论，转而讨论另一个 Sanic 的流行用例：聊天机器人。

## 构建一个 Discord 机器人（在另一个服务中运行 Sanic）

在 2021 年初的某个时候，我被 Sanic 社区的一些人说服，我们需要迁移我们的主要讨论和社区建设工具。我们有一个相对未被充分利用的聊天应用程序，还有一个主要用于较长风格支持问题的社区论坛。Discord 比其他选项提供的对话更加亲密。当有人建议我使用 Discord 时，我有点犹豫是否要在我的工具箱中添加另一个应用程序。尽管如此，我们还是继续了。如果这本书的读者中有 Discord 的粉丝，那么你不需要我向你解释它的好处。对于其他人来说，Discord 是一个非常易于使用且引人入胜的平台，它真正促进了对我们这个网络角落有益的讨论。

随着我对这个平台了解的越来越多，最让我印象深刻的是聊天机器人无处不在。有一个我之前不知道的、与构建机器人相关的惊人亚文化。这些机器人中的绝大多数都是使用 SDKs 构建的，这些 SDKs 是围绕与 Discord API 接口所需的大部分客户端 HTTP 交互的开放源代码项目。在这个基础上建立起了整个生态系统和框架，以帮助开发者构建引人入胜的机器人。

自然地，人们经常问的下一个问题是：我该如何将 Sanic 集成到我的机器人应用程序中？我们将尝试做到这一点。

但首先，我想指出的是，虽然我们将要构建的示例使用的是 Discord，但其中的原则与在 Discord 上运行这一点毫无关联。我们即将要做的是运行一些`asyncio`进程，并重用这个循环来运行 Sanic。这意味着你实际上可以使用这种完全相同的技术来运行嵌套的 Sanic 应用程序。我们将在下一节中看到这会是什么样子。

### 构建简单的 Discord 机器人

我并不是 Discord 的专家。基于这个平台，有一个完整的开发领域，我并不假装自己是这方面的权威。我们的目标是集成一个与 Sanic 的机器人应用程序。为此，我们将使用`nextcord`搭建一个基本的 Discord 机器人。如果你对`nextcord`不熟悉，截至本书编写时，它是对已废弃的`discord.py`项目的活跃维护分支。如果你对那个也不熟悉，不用担心。简单的解释是，这些是用于在 Discord 上构建机器人应用程序的框架。类似于 Sanic 提供 HTTP 通信的工具，这些框架提供了与 Discord 通信的工具。

让我们花一分钟时间来考虑一下他们文档中的基本“Hello World”应用程序：

```py
import nextcord
client = nextcord.Client()
@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')
client.run('your token here')
```

老实说，这看起来与我们构建的 Sanic 并没有太大的不同。它从应用程序实例开始。然后，有装饰器包装处理程序。最后我们看到的是`client.run`。

这是我们要构建的关键。这个`run`方法将创建一个循环，并在应用程序关闭之前运行它。我们现在的任务是运行 Sanic 在这个应用程序内部。这意味着我们**不会**使用 Sanic cli 来启动我们的应用程序。相反，我们将使用以下方式运行应用程序：

```py
$ python bot.py
```

让我们开始吧。

1.  首先，从他们的文档中复制最小的机器人示例到`bot.py`。你可以在这里获取代码：[`nextcord.readthedocs.io/en/latest/quickstart.html`](https://nextcord.readthedocs.io/en/latest/quickstart.html)

1.  创建一个简单的 Sanic 应用程序作为概念验证。

    ```py
    from sanic import Sanic, Request, json
    app = Sanic(__name__)
    @app.get("/")
    async def handler(request: Request):
        await request.app.ctx.general.send("Someone sent a message")
        return json({"foo": "bar"})
    @app.before_server_start
    async def before_server_start(app, _):
        await app.ctx.general.send("Wadsworth, reporting for duty")
    ```

    目前还没有什么特别的事情发生。我们有一个单独的处理程序，在服务器启动之前在监听器中发送消息。此外，我们还有一个单独的处理程序，当路由端点被击中时，也会触发向我们的 Discord 服务器发送消息。

1.  为了将此与 Discord 机器人集成，我们将使用`on_ready`事件来运行我们的 Sanic 服务器。

    ```py
    from server import app
    @client.event
    async def on_ready():
        app.config.GENERAL_CHANNEL_ID = 906651165649928245
        app.ctx.wadsworth = client
        app.ctx.general = client.get_channel(app.config.GENERAL_CHANNEL_ID)

        if not app.is_running:
            app_server = await app.create_server(port=9999, return_asyncio_server=True)
            app.ctx.app_server = app_server
            client.loop.create_task(runner(app_server))
    ```

    > **重要通知**
    > 
    > 为了简化，我只是从`server`导入`app`。这是因为这是一个超级简单的实现。实际上，如果我要构建一个合适的应用程序，我**不会**使用这种模式。相反，我会使用本书中反复讨论的工厂模式，并从可调用对象构建我的应用程序。这是为了帮助导入管理和避免传递全局作用域变量。

    这里发生了一些我们需要讨论的事情。首先，如前所述，这是告诉`nextcord`在应用程序启动并连接到 Discord 且因此“就绪”时运行此处理器的语法。但是，根据他们的文档，此事件可能会被触发多次。尝试多次运行 Sanic 将是一个错误，因为它将无法正确绑定到套接字。为了避免这种情况，我们查看`app.is_running`标志以确定是否应该再次运行它。接下来会发生的事情是，我们将手动创建一个 Sanic 服务器。之后——这部分非常重要——我们将该应用程序服务器实例传递给一个**新**的任务。为什么？因为如果我们从这个当前任务运行 Sanic，它将无限期地阻塞，Discord 机器人永远不会真正运行。由于我们希望它们同时运行，因此从另一个`asyncio`任务运行 Sanic 是至关重要的。

1.  接下来，我们需要创建那个`runner`操作。这里的任务是运行创建的服务器。这意味着我们需要手动触发所有监听事件。这也意味着我们需要执行一些连接关闭操作。因为我们操作的水平比正常情况低得多，所以你需要更加亲自动手。

    ```py
    async def runner(app_server: AsyncioServer):
        app.is_running = True
        try:
            await app_server.startup()
            await app_server.before_start()
            await app_server.after_start()
            await app_server.serve_forever()
        finally:
            app.is_running = False
            app.is_stopping = True
            await app_server.before_stop()
            await app_server.close()
            for connection in app_server.connections:
                connection.close_if_idle()
            await app_server.after_stop()
            app.is_stopping = False
    ```

这里的工作看起来很简单。它启动应用程序，运行一些监听事件，然后将持续监听，直到应用程序关闭。在完全退出之前，我们需要在`finally`块中运行一些清理操作。

一旦您实现了所有这些，您就可以像我们之前所说的那样通过执行 bot.py 脚本来运行它。现在，您应该会在启动应用程序生命周期期间由 Sanic 触发的 Discord 服务器上看到这条消息。

<<<< 图片 >>>>>

接下来，您应该能够点击您的单个端点并看到另一条消息：

<<<< 图片 >>>>>

由于我们不是使用运行 Sanic 的标准方法，我不太推荐这种方法。首先，很容易搞错调用顺序，要么遗漏一些关键事件，要么不恰当地处理像关闭这样的操作。诚然，上面的关闭机制是不完整的。首先，它不包括对现有连接的优雅关闭处理。

这引出了下一个问题：我们是否可以在 Sanic 内部运行 Discord 机器人，而不是在 Discord 机器人内部运行 Sanic？是的，这正是我们接下来要做的。

### 在 Sanic 中运行 Discord 机器人

在我们开始之前，让我们考虑一下`client.run`正在做什么。它执行运行其服务所需的任何内部实例化，包括连接到 Discord 服务器。然后，它进入一个循环，异步接收和发送消息到 Discord 服务器。这听起来非常类似于 Sanic 服务器所做的事情。因此，我们可以做与我们刚才所做完全相同的事情，只是顺序相反。

1.  取出我们刚刚构建的代码，并从机器人中移除`on_ready`事件。

1.  添加一个启动时间监听器，在新的后台任务中启动机器人。

    ```py
    @app.before_server_start
    async def startup_wadsworth(app, _):
        app.ctx.wadsworth = client
        app.add_task(client.start(app.config.DISCORD_TOKEN))
        while True:
            if client.is_ready():
                app.ctx.general = client.get_channel(app.config.GENERAL_CHANNEL_ID)
                await app.ctx.general.send("Wadsworth, reporting for duty")
                break
            await asyncio.sleep(0.1)
    ```

    在这个监听器中，我们也在做与上一个示例中相同的事情。我们设置了`app.ctx.wadsworth`和`app.ctx.general`，以便它们在构建过程中易于访问和使用。此外，我们希望在 Wadsworth 上线并准备好工作的时候发送一条消息。是的，我们可以像之前一样使用`on_ready`从机器人中完成这个操作，但我们也可以从 Sanic 中完成这个操作。在上面的代码中，我们创建了一个循环来检查机器人的状态。一旦它准备好了，我们将发送消息并关闭循环。

1.  我们接下来需要确保正确关闭机器人连接。我们将在关闭监听器中完成这个操作。

    ```py
    @app.before_server_stop
    async def shutdown(app, _):
        await client.close()
    ```

现在，您已经具备了从 Sanic 运行机器人的全部能力。这应该表现得与之前完全一样，但您现在可以使用 Sanic CLI 运行应用程序的全部功能，正如我们在本书的其余部分所做的那样。现在就启动它吧：

```py
$ sanic server:app -p 7777 --debug --workers=2
```

这种嵌套其他`asyncio`应用程序的模式不仅适用于同时运行 Discord 机器人和 Sanic，其适用范围更广。它还允许我们在同一进程中运行多个 Sanic 应用程序，尽管它们在不同的端口上。这正是我们接下来要做的。

## 嵌套 Sanic 应用程序：在 Sanic 内部运行 Sanic 以创建 HTTP 代理

在 Sanic 内运行 Sanic 似乎有点像俄罗斯套娃。虽然这最初可能看起来像是一个惊人的思想实验，但它确实有一些实际应用。这种运行两个 Sanic 实例的最明显例子是创建你自己的 HTTP 到 HTTPS 代理。这正是我们现在要做的。或者，至少是某种程度上的。

我想添加的一个注意事项是，这个示例将使用 **自签名证书**。这意味着它不适合生产使用。你应该查看第七章中名为 ___ 的部分，以了解如何使用 TLS 正确保护你的应用程序的详细信息。

首先，我们将创建两个服务器。为了简单起见，一个将是 server.py（你的主要应用程序在 443 端口上运行 HTTPS），另一个将是 redirect.py（在 80 端口上运行的 HTTP 到 HTTPS 代理）。

1.  我们将首先创建我们的自签名证书。如果你在 Windows 机器上，你可能需要查找如何在你的操作系统上完成这个操作。

    ```py
    $ openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365
    ```

1.  接下来，我们在 server.py 中使用简单的工厂模式开始构建我们的 Sanic 应用程序。这个构建的代码可以在 ___ 找到。

    ```py
    from sanic import Sanic
    from wadsworth.blueprints.view import bp
    from wadsworth.blueprints.info.view import bp as info_view
    from wadsworth.applications.redirect import attach_redirect_app
    def create_app():
        app = Sanic("MainApp")
        app.config.SERVER_NAME = "localhost:8443"
        app.blueprint(bp)
        app.blueprint(info_view)
        attach_redirect_app(app)
        return app
    ```

    > **提示**
    > 
    > 我首先想指出的是 `SERVER_NAME` 的使用。这是一个在 Sanic 中默认未设置的配置值。这通常是你应该在所有应用程序中使用的东西。这是一个在 Sanic 背景中在几个位置使用的有用值。在我们的示例中，我们想使用它来帮助我们使用 app.url_for 生成稍后路线的 URL。该值应该是你的应用程序的域名，加上端口（如果它不是使用标准 80 或 443）。你不应该包括 http:// 或 https://。

    `attach_redirect_app` 是什么？这是一个另一个应用程序工厂。但它的运作方式会有所不同，因为它还将负责将 resitect 应用程序嵌套在 `MainApp` 内。最后一点值得指出的是，有一个 Blueprint Group bp，我们将把所有的 Blueprints 都附加到它上面。不过，`info_view` 将是独立的。更多细节稍后揭晓。

1.  我们开始第二个工厂模式：`attach_redirect_app` 在 `redirect.py`。

    ```py
    def attach_redirect_app(main_app: Sanic):
        redirect_app = Sanic("RedirectApp")
        redirect_app.blueprint(info_view)
        redirect_app.blueprint(redirect_view)
        redirect_app.ctx.main_app = main_app
    ```

    我们将附加两个视图：我们刚刚附加到 `MainApp` 的相同 `info_view`，以及将执行重定向逻辑的 `redirect_view`。我们将在完成这里的工厂和 `redirect.py` 中的服务器后查看它。此外，请注意，我们将 `main_app` 附加到 `redirect_app.ctx` 以便稍后检索。正如我们所学的，通过 ctx 传递对象是处理需要在应用程序中引用的对象的首选方法。

1.  接下来，我们将向 `MainApp` 添加一些监听器。这将在 `attach_redirect_app` 工厂内部发生。有些软件架构师可能不喜欢我将逻辑关注点嵌套在一起，但我们将忽略批评者并继续这样做，因为我们追求的是必须紧密耦合的逻辑，这将使我们未来更容易调试和更新。

    ```py
    def attach_redirect_app(main_app: Sanic):
        ...
        @main_app.before_server_start
        async def startup_redirect_app(main: Sanic, _):
            app_server = await redirect_app.create_server(
                port=8080, return_asyncio_server=True
            )
            if not app_server:
                raise ServerError("Failed to create redirect server")
            main_app.ctx.redirect = app_server
            main_app.add_task(runner(redirect_app, app_server))
    ```

    在这里，我们深入到 Sanic 服务器的较低级别操作。我们基本上需要模仿 Sanic CLI 和 app.run 所做的操作，但要在现有循环的范围内进行。当你运行一个 Sanic 服务器实例时，它将阻塞进程直到关闭。但我们需要运行两个服务器。因此，`RedirectApp` 服务器需要作为一个后台任务运行。我们通过使用 add_task 推迟运行服务器的工作来实现这一点。我们将在完成工厂后回到 runner。

1.  `RedirectApp` 也需要关闭。因此，我们向 MainApp 添加另一个监听器来完成这项工作。

    ```py
    def attach_redirect_app(main_app: Sanic):
        ...
        @main_app.after_server_stop
        async def shutdown_redirect_app(main: Sanic, _):
            await main.ctx.redirect.before_stop()
            await main.ctx.redirect.close()
            for connection in main.ctx.redirect.connections:
                connection.close_if_idle()
            await main.ctx.redirect.after_stop()
            redirect_app.is_stopping = False
    This includes all of the major elements you need for turning down Sanic. It is a little bit basic and if you are implementing this in the real world, you might want to take a look into how Sanic server performs a graceful shutdown to close out any existing requests.
    We now turn to runner, the function that we passed off to be run in a background task to run the RedirectApp.
    async def runner(app: Sanic, app_server: AsyncioServer):
        app.is_running = True
        try:
            app.signalize()
            app.finalize()
            ErrorHandler.finalize(app.error_handler)
            app_server.init = True
            await app_server.before_start()
            await app_server.after_start()
            await app_server.serve_forever()
        finally:
            app.is_running = False
            app.is_stopping = True
    ```

    再次强调，我们正在完成 Sanic 在幕后启动服务器的一些高级步骤。它确实在 `create_server` 之前运行 `before_start`。影响很小。由于我们的 `RedirectApp` 甚至没有使用任何事件监听器，我们可以没有 `before_start` 和 `after_start`（以及关闭事件）。

1.  现在转到应用程序的重要部分：重定向视图。

    ```py
    from sanic import Blueprint, Request, response
    from sanic.constants import HTTP_METHODS
    bp = Blueprint("Redirect")
    @bp.route("/<path:path>", methods=HTTP_METHODS)
    async def proxy(request: Request, path: str):
        return response.redirect(
            request.app.url_for(
                "Redirect.proxy",
                path=path,
                _server=request.app.ctx.main_app.config.SERVER_NAME,
                _external=True,
                _scheme="https",
            ),
            status=301,
        )
    ```

    此路由将非常全面。它基本上将接受所有未匹配的端点，无论使用什么 HTTP 方法。这是通过使用路径参数类型并将 `HTTP_METHODS` 常量传递给路由定义来实现的。任务是重定向到 https 版本的精确相同的请求。你可以这样做几种方式。例如，以下方法有效：

    ```py
    f"https://{request.app.ctx.main_app.config.SERVER_NAME}{request.path}"
    ```

    然而，对我来说和我的大脑，我喜欢使用 `url_for`。如果你更喜欢替代方案：你做你的。重定向函数是一个方便的方法，用于生成适当的重定向响应。由于我们的用例需要从 http 重定向到 https，我们使用 301 重定向来表示这是一个永久（而不是临时）的重定向。让我们看看它是如何工作的。

1.  要运行我们的应用程序，我们需要使用我们生成的 TLS 证书。

    ```py
    $ sanic wadsworth.applications.server:create_app \
        --factory --workers=2 --port=8443 \
        --cert=./wadsworth/certs/cert.pem \
        --key=./wadsworth/certs/key.pem
    ```

    我们再次使用 CLI 运行应用程序。请确保使用 `--factory`，因为我们正在传递一个可调用对象。同时，我们告诉 Sanic 它可以在哪里找到为 TLS 加密生成的证书和密钥。

1.  一旦运行起来，我们将进入终端使用 `curl` 进行测试。首先，我们将确保两个应用程序都是可访问的：

    ```py
    $ curl http://localhost:8080/info
    {"server":"RedirectApp"}
    That looks right.
    $ curl -k https://localhost:8443/info   
    {"server":"MainApp"}
    ```

    这看起来也是正确的。请注意，我在 curl 命令中包含了`-k`。这是因为我们创建的自签名证书。由于它不是来自官方受信任的证书颁发机构，`curl`将不会自动发出请求，直到你明确告诉它证书是好的。关于这一点，真正有趣的是`/info`端点**并没有**被定义两次。如果你查看源代码，你会看到它是一个蓝图，已经应用于两个应用程序。非常方便。

1.  现在我们来到了最后的测试：重定向。

    ```py
    $ curl -kiL http://localhost:8080/v1/hello/Adam      
    HTTP/1.1 301 Moved Permanently
    Location: https://localhost:8443/v1/hello/Adam
    content-length: 0
    connection: keep-alive
    content-type: text/html; charset=utf-8
    HTTP/1.1 200 OK
    content-length: 16
    connection: keep-alive
    content-type: application/json
    {"hello":"Adam"}
    ```

请确保注意我们正在访问 8080 端口，这是`RedirectApp`。我们再次使用`-k`来告诉 curl 不要担心证书验证。我们还使用`-L`来告诉`curl`跟随任何重定向。最后，我们添加`-i`来输出完整的 HTTP 响应，这样我们就可以看到发生了什么。

如您从上面的响应中可以看到，我们生成了一个适当的 301 重定向，并将用户引导到了 https 版本，它用我的名字亲切地问候了我。

就这样：一个简单的 HTTP 到 HTTPS 重定向应用程序，在 Sanic 内部运行 Sanic。

## 摘要

我喜欢构建 Web 应用程序的机会，可以构建解决问题的方案。例如，在本章的早期，我们遇到了从 Sanic 运行 JavaScript 开发服务器的问题。如果你让五位不同的开发者来解决这个问题，你可能会得到五种不同的解决方案。我相信，在某种程度上，构建 Web 应用程序是一种艺术形式。也就是说，它不是一个必须以唯一一种*明显*的方式解决的问题的严格领域。相反，什么是明显的，只能根据你构建的独特环境和参数来确定。

当然，我们在这里构建的只是 Sanic 可能性的冰山一角。显示的选择既有一些流行的用例，也有一些可能不太直接的用例。我希望你能从中汲取一些想法和模式，并加以有效利用。通过阅读这本书并内化本章中的示例，我希望我已经帮助激发了你在构建应用程序方面的创造性思维。

如果我们将本章的所有想法整合成一个单一的应用程序，你将得到一个由 Sanic 驱动的 PWA，使用分布式 WebSocket 流和 GraphQL API，同时还运行一个 Discord 机器人。我的观点是，在应用程序中实现功能不能在真空中完成。在决定如何构建某物时，你必须考虑你的架构的其他部分。本章旨在帮助您了解我在解决这些问题时的思考过程。

随着我们接近本书的结尾，我们最后需要做的是将我们所知的大量内容整合成一个可部署的应用程序。这就是我们在第十一章要做的：构建一个完全功能、适用于生产的 Sanic 应用程序。
