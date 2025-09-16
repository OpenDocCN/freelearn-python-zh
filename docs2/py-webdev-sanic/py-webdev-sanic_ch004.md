# 3 路由和接收 HTTP 请求

在*第一章，Sanic 和异步框架的介绍*中，我们查看了一个原始 HTTP 请求以了解它包含的信息。在本章中，我们将更仔细地查看包含 HTTP 方法和 URI 路径的第一行。正如我们所学的，Web 框架最基本的功能是将原始 HTTP 请求转换为可执行的处理程序。在我们看到如何实现之前，记住原始请求的样子是好的：

```py
POST /path/to/endpoint HTTP/1.1
Host: localhost:7777
User-Agent: curl/7.76.1
Accept: */*
Content-Length: 14
Content-Type: application/json
{"foo": "bar"}
```

观察请求，我们看到以下内容：

+   第一行（有时称为*起始行*）包含三个子部分：**HTTP 方法**、**请求目标**和**HTTP**协议

+   第二部分包含零个或多个以`key: value`形式出现的 HTTP 头信息，每对之间由换行符分隔

+   然后，我们有一个空白行将头部与正文分开

+   最后，我们还有可选的正文

具体的规范由 RFC 7230，3\. [`datatracker.ietf.org/doc/html/rfc7230#section-3`](https://datatracker.ietf.org/doc/html/rfc7230#section-3) 覆盖

本书的一个目标是通过学习策略来设计易于消费的 API 端点，同时考虑到我们正在构建的应用程序的需求和限制。目标是理解服务器与传入的 Web 请求的第一交互，以及如何围绕这一点设计我们的应用程序。我们将学习：请求的结构；Sanic 为我们做出的选择以及它留下的选择；以及将 HTTP 请求转换为可执行代码所涉及的其他问题。记住，本书的目的不仅仅是学习如何使用一个花哨的新工具，还要提升 Web 开发和知识技能。为了成为更了解的开发者，我们不仅寻求理解*如何*使用 Sanic 构建，还要理解*为什么*我们可能以特定方式构建某些内容。通过理解一些机制，我们将学会提出更好的问题并做出更好的决策。这并不意味着我们需要成为 HTTP 协议和规范的专家。然而，通过熟悉 Sanic 对原始请求的处理，我们最终将拥有构建 Web 应用程序的更强大的工具集。

特别是，我们将涵盖以下主题：

+   理解 HTTP 方法

+   路径、斜杠及其重要性

+   高级路径参数

+   API 版本控制

+   虚拟主机

+   提供静态内容

## 技术要求

除了我们之前所构建的内容之外，在本章中，你应该拥有以下工具以便能够跟随示例进行学习：

+   Docker Compose

+   Curl

+   您可以在 GitHub 上找到本章的源代码：[`github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/03`](https://github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/03)

## 理解 HTTP 方法

如果你之前构建过任何类型的网站，你可能对**HTTP 方法**的概念有所了解；或者至少对基本的`GET`和`POST`方法有所了解。然而，你知道有九种标准的 HTTP 方法吗？在本节中，我们将了解这些不同的方法以及我们**可以**如何利用它们。

就像 IP 地址或网络域名是互联网上的**地点**一样，HTTP 方法是在互联网上的**动作**。它们是网络语言中的动词集合。这些 HTTP 方法有一个共同的理解和意义。Web 应用程序通常会在类似的使用场景中使用这些方法。这并不意味着你必须遵循这些约定，或者如果你的应用程序偏离了标准就会出错。我们应该学习这些规则，以便我们知道何时可能适合打破它们。这些标准存在是为了创建一个共同的语言，让 Web 开发者和消费者可以使用它来沟通：

| 方法 | 描述 | 有主体 | 安全 | Sanic 支持 |
| --- | --- | --- | --- | --- |
| `CONNECT` | 打开双向通信，如到资源的隧道 | 否 | 是 | 否 |
| `DELETE` | 删除资源 | 否（通常） | 否 | 是 |
| `GET` | 获取资源 | 否 | 是 | 是 |
| `HEAD` | 仅获取资源的元数据 | 否 | 是 | 是 |
| `OPTIONS` | 请求允许的通信选项 | 否 | 是 | 是 |
| `PATCH` | 部分修改资源 | 是 | 否 | 是 |
| `POST` | 向服务器发送数据 | 是 | 否 | 是 |
| `PUT` | 创建新资源或如果存在则完全更新 | 是 | 否 | 是 |
| `TRACE` | 执行用于调试的消息回环 | 否 | 是 | 否 |

表 3.1 - HTTP 方法概述

当我们谈论一个方法是否**安全**时，我们的意思是它不应该改变状态。这并不是说`GET`方法不能有副作用。当然，它可以。例如，有人点击端点会触发日志或某种资源计数器。这些在技术上可能是行业所说的副作用。“*这里的重要区别是用户没有请求这些副作用，因此不能对它们负责。”* RFC 2616，9.1.1 ([`datatracker.ietf.org/doc/html/rfc2616#section-9`](https://datatracker.ietf.org/doc/html/rfc2616#section-9))。这意味着从用户访问资源的角度来看，确定一个端点是否**安全**是一个意图问题。如果用户意图检索个人资料信息，则是安全的。如果用户意图更新个人资料信息，则是不安全的。

尽管尝试坚持 *表 3.1* 中的方法描述当然很有帮助，但无疑你将遇到不符合这些类别的用例。当这种情况发生时，我鼓励你重新审视你的应用程序设计。有时问题可以通过新的端点路径来解决。有时我们需要创建自己的定义。这是可以的。然而，我警告不要将 *安全* 方法改为 *不安全*。使用 `GET` 请求进行状态更改被认为是不良的做法，是 *新手错误*。

在决定我们的 HTTP 方法之后，我们将进入下一节学习如何实现它们并将它们附加到路由上。

### 在路由处理器中使用 HTTP 方法

我们终于准备好学习和了解框架是什么了！如果你之前使用过 Flask，这会看起来很熟悉。如果没有，我们即将要做的是创建一个路由定义，它是一组指令，告诉 Sanic 将任何传入的 HTTP 请求发送到我们的路由处理器。路由定义必须有两个部分：一个 URI 路径和一个或多个 HTTP 方法。

仅匹配 URI 路径是不够的。HTTP 方法也被 Sanic 用于将你的传入请求路由到正确的处理器。即使我们实现最基本的路由定义形式，这两部分都必须存在。让我们看看最简单的用例，看看 Sanic 会做出什么默认选择：

```py
@app.route("/my/stuff")
async def stuff_handler(...):
    ...
```

在这个例子中，我们在 `/my/stuff` 上定义了一个路由。通常我们会用可选的 `methods` 参数注入 `route()` 调用，以告诉它我们希望该处理器响应哪些 HTTP 方法。我们没有在这里这样做，所以它将默认为 `GET`。我们有告诉路由它应该处理其他 HTTP 方法的选项：

```py
@app.route("/my/stuff", methods=["GET", "HEAD"])
async def stuff_handler(...):
    return text("Hello")
```

> **重要提示**
> 
> 我们将在本章稍后讨论 `HEAD` 方法。但重要的是要知道，`HEAD` 请求不应该有任何响应体。Sanic 会为我们强制执行这一点。尽管技术上这个端点正在响应文本 `Hello`，但 Sanic 会从响应中移除体，只发送元数据。

现在我们已经设置了一个具有多个方法的单个端点，我们可以用这两种方法来访问它。

首先，使用 `GET` 请求（应该注意的是，当使用 `curl` 时，如果你没有指定方法，它将默认为 `GET`）：

```py
$ curl localhost:7777/my/stuff -i
HTTP/1.1 200 OK
content-length: 5
connection: keep-alive
content-type: text/plain; charset=utf-8
Hello
Then, with a HEAD request.
$ curl localhost:7777/my/stuff -i --head
HTTP/1.1 200 OK
content-length: 5
connection: keep-alive
content-type: text/plain; charset=utf-8
```

为了方便起见，Sanic 为其支持的所有 HTTP 方法在应用实例和任何 Blueprint 实例上提供了快捷装饰器：

```py
@app.get("/")
def get_handler(...):
    ...
@app.post("/")
def post_handler(...):
    ...
@app.put("/")
def put_handler(...):
    ...
@app.patch("/")
def patch_handler(...):
    ...
@app.delete("/")
def delete_handler(...):
    ...
@app.head("/")
def head_handler(...):
    ...
@app.options("/")
def options_handler(...):
    ...
```

这些装饰器也可以堆叠。我们之前看到的最后一个例子也可以这样写：

```py
@app.head("/my/stuff")
@app.get("/my/stuff")
async def stuff_handler(...):
    return text("Hello")
```

关于 HTTP 方法还有一点需要了解，那就是你可以访问 HTTP 请求对象上的传入方法。如果你在同一个处理器上处理不同类型的 HTTP 方法，但需要以不同的方式处理它们，这非常有帮助。以下是一个例子，我们通过查看 HTTP 方法来改变处理器的行为

```py
from sanic.response import text, empty
from sanic.constants import HTTPMethod
@app.options("/do/stuff")
@app.post("/do/stuff")
async def stuff_handler(request: Request):
    if request.method == HTTPMethod.OPTIONS:
        return empty()
    else:
        return text("Hello")
```

在继续到高级方法路由之前，我们应该提到一些 Sanic 语法。这里所有的示例都使用装饰器语法来定义路由。这无疑是实现这一目标最常见的方式，因为它很方便。然而，还有一个替代方案。所有的路由定义都可以转换为如下所示的功能定义：

```py
@app.get("/foo")
async def handler_1(request: Request):
    ...
async def handler_2(request: Request):
    ...
app.add_route(handler_2, "/bar")
```

在某些情况下，这可能是一个更吸引人的模式来使用。当我们在本章后面遇到基于类的视图时，我们还会再次看到它。

### 高级方法路由

Sanic 默认不支持 `CONNECT` 和 `TRACE` 这两种标准 HTTP 方法。但让我们想象一下，如果你想构建一个 HTTP 代理或需要在其路由处理程序中提供 `CONNECT` 方法的其他系统。尽管 Sanic 默认不允许这样做，但你有两种潜在的方法：

首先，我们可以创建一个中间件，它会寻找 `CONNECT` 并劫持请求以提供自定义响应。这种从中间件响应的 *技巧* 是一个允许你在处理程序接管之前停止请求/响应生命周期的功能，否则会失败并显示 `404 Not Found`：

```py
async def connect_handler(request: Request):
    return text("connecting...")
@app.on_request
async def method_hijack(request: Request):
    if request.method == "CONNECT":
        return await connect_handler(request)
```

你可以看到，这种方法的潜在缺点是我们需要实现自己的路由系统，如果我们想要将不同的端点发送到不同的处理程序。

第二种方法可能是告诉 Sanic 路由器 `CONNECT` 是一个有效的 HTTP 方法。一旦我们这样做，我们就可以将其添加到正常的请求处理程序中：

```py
app.router.ALLOWED_METHODS = [*app.router.ALLOWED_METHODS, "CONNECT"]
@app.route("/", methods=["CONNECT"])
async def connect_handler(request: Request):
    return text("connecting...")
```

对于这种策略的一个重要考虑是，你需要在注册新处理程序之前尽可能早地重新定义 `app.router.ALLOWED_METHODS`。因此，它最好直接在 `app = Sanic(...)` 之后进行。

这种策略提供的一个附带好处是能够创建自己的 HTTP 方法生态系统，并使用自己的定义。如果你打算让你的 API 供公众使用，这可能并不一定可取。然而，对于你自己的目的来说，这可能是有用的、实用的，或者只是纯粹的乐趣。虽然有九种标准方法，但可能性是无限的。你想要创建自己的动词吗？你当然可以这样做。

`ATTACK /path/to/the/dragon HTTP/1.1`

### 方法安全性和请求体

正如我们所学的，通常有两种类型的 HTTP 方法：**安全** 和 **不安全**。不安全的方法是 `POST`、`PUT`、`PATCH` 和 `DELETE`。这些方法通常被理解为它们是改变状态的。也就是说，通过点击这些端点，用户意图以某种方式改变或修改资源。

相反的是安全方法：`GET`、`HEAD` 和 `OPTIONS`。这些端点的目的是从应用程序请求信息，而不是改变状态。

被认为是一种良好的实践来遵循这一做法。如果一个端点将在服务器上做出更改，请不要使用 `GET`。

与这种划分相对应的是请求体的概念。让我们再次回顾一下原始的 HTTP 请求：

```py
POST /path/to/endpoint HTTP/1.1
Host: localhost:7777
User-Agent: curl/7.76.1
Accept: */*
Content-Length: 14
Content-Type: application/json
{"foo": "bar"}
```

HTTP 请求可以可选地包含一个消息体。在上面的例子中，请求体是最后一行：`{"foo": "bar"}`。

需要注意的是，Sanic 只会花费时间读取`POST`、`PUT`和`PATCH`请求的消息体。如果是一个使用任何其他 HTTP 方法的 HTTP 请求，它将在读取头部后停止读取 HTTP 消息。这是一个性能优化，因为我们通常不期望在*安全*的 HTTP 请求中存在消息体。

你可能已经注意到这个列表中没有包括`DELETE`。为什么？一般来说，HTTP 规范说可能存在请求体（[`datatracker.ietf.org/doc/html/rfc7231#section-4.3.5`](https://datatracker.ietf.org/doc/html/rfc7231#section-4.3.5)）。Sanic 假设除非你告诉它，否则它不会存在。为此，我们只需设置`ignore_body=False`：

```py
@app.delete("/", ignore_body=False)
async delete_something(request: Request):
    await delete_something_using_request(request.body)
```

如果我们没有设置`ignore_body=False`，并且我们在`DELETE`请求中发送一个消息体，Sanic 将在日志中发出警告，让我们知道 HTTP 消息的一部分没有被消费。如果你打算使用`DELETE`方法，你应该注意这一点，因为 Sanic 做出了这样的假设。还应该注意的是，如果你习惯于接收带有消息体的 GET 请求，你也需要使用`ignore_body=False`。然而，我希望你有一个非常好的理由来做这件事，因为这将违反大多数网络标准。

从这个例子中我们可以得到的一个有用的启示是，开箱即用，以下两个端点**并不相等**。

```py
@app.route("/one", methods=["GET"])
async def one(request: Request):
    return text("one")
@app.get("/two")
async def two(request: Request):
    return text("two")
```

`/one`和`/two`的行为将相似。然而，如果没有进一步的定制，第一个请求将花费时间尝试读取可能不存在的请求体，而第二个则假设不存在消息体。虽然性能差异可能很小，但通常更倾向于使用`@app.get("/two")`而不是`@app.route("/one", methods=["GET"])`。这两个端点之所以不同，是因为它们对`ignore_body`的默认值不同。

> **重要提示**
> 
> 如果你正在构建一个 GraphQL 应用程序，那么通常即使对于信息请求，端点也会使用`POST`。这是因为将消息体放在`POST`请求中通常比放在`GET`请求中更容易被接受。然而，值得一提的是，如果我们真的想的话，我们可以通过设置`ignore_body=False`从`GET`请求中消费消息体。

当决定使用哪种方法时，另一个需要考虑的因素是**幂等性**。简而言之，幂等性意味着你可以重复执行相同的操作，每次的结果都应该是相同的。被认为是幂等的 HTTP 方法有：`GET`、`HEAD`、`PUT`、`DELETE`、`OPTIONS`和`TRACE`。在设计你的 API 时请记住这一点。

### RESTful API 设计

HTTP 方法通常用于 **RESTful API 设计**。关于构建 RESTful API 的文献已经非常丰富，因此我们不会深入探讨它是什么，而是更多地关注我们如何实际实现它。然而，我们首先应该快速回顾一下基本前提。

Web API 端点有一个目标。这个目标是指用户想要获取信息或通过添加或更改来操作的东西。基于共同的理解，HTTP 方法告诉服务器您希望如何与该目标交互。该 **目标** 通常被称为 **资源**，在这里我们可以互换使用这些术语。

为了理解这个概念，我喜欢回想我小时候玩的冒险电脑游戏。我的冒险角色会偶然发现一个物品：比如说一个橡胶鸡。当我点击那个物品时，会出现一个菜单，显示不同的动词，告诉我我可以对这个物品做什么：捡起、查看、使用、交谈等等。有一个目标（橡胶鸡），以及方法（动词或动作）。

将这些与我们上面定义的 HTTP 方法结合起来，让我们看看一个具体的例子。在我们的假设情况下，我们将构建一个用于管理喜欢冒险电脑游戏的人们的社交媒体平台的 API。用户需要能够创建个人资料、查看其他个人资料以及更新自己的个人资料。我们可能会设计以下端点：

| **方法** | **URI 路径** | **描述** |
| --- | --- | --- |
| `GET` | `/profiles` | 所有成员个人资料的列表 |
| `POST` | `/profiles` | 创建新的个人资料 |
| `GET` | `/profiles/<username>` | 获取单个用户的个人资料 |
| `PUT` | `/profiles/<username>` | 删除旧的个人资料并用完整的个人资料替换 |
| `PATCH` | `/profiles/<username>` | 仅对个人资料的一部分进行更改 |
| `DELETE` | `/profiles/<username>` | 删除个人资料——但为什么有人会想删除他们的冒险游戏玩家个人资料呢？ |

表 3.2 - 示例 HTTP 方法与端点

在我们继续之前，如果您对 Sanic 中的路由工作方式不熟悉（以及 `<username>` 语法意味着什么），您可以在用户指南中获取更多信息：[`sanicframework.org/en/guide/basics/routing.html`](https://sanicframework.org/en/guide/basics/routing.html)，我们也会在本章的“从路径中提取信息”部分更详细地探讨它。您可以自由地跳过并稍后回来。

如您所见，实际上只有两个 URI 路径：`/profiles` 和 `/profiles/<username>`。然而，通过使用 HTTP 方法，我们已经能够定义与我们的 API 的六种不同交互！个人资料蓝图可能是什么样的呢？

```py
from sanic import Blueprint, Request
bp = Blueprint("MemberProfiles", url_prefix="/profile")
@bp.get("")
async def fetch_all_profiles(request: Request):
    ...
@bp.post("")
async def create_new_profile(request: Request):
    ...
@bp.get("/<username>")
async def fetch_single_profile(request: Request, username: str):
    ...
@bp.put("/<username>")
async def replace_profile(request: Request, username: str):
    ...
@bp.patch("/<username>")
async def update_profile(request: Request, username: str):
    ...
@bp.delete("/<username>")
async def delete_profile(request: Request, username: str):
    ...
```

使用 HTTP 方法来定义我们的用例似乎很有帮助，并且有映射它们的装饰器似乎很方便。但是，似乎有很多样板代码和重复。接下来，我们将探讨基于类的视图以及我们如何简化我们的代码。

### 使用基于类的视图简化你的端点

之前的例子暴露了仅使用函数和装饰器来设计 API 的弱点。当我们想要为 `/profile/<user_id:uuid>` 添加端点处理程序时会发生什么？或者当我们想要对现有端点进行其他更改时？我们现在有多个地方可以做出相同的更改，这导致我们无法在所有路由定义之间保持一致性，这是违反了 **DRY**（**不要重复自己**）原则的，可能会导致错误。因此，从长远来看，维护这些端点可能比必要的更困难。

这就是使用 **基于类的视图**（**CBV**）的一个非常有说服力的原因。这个模式将给我们提供将前两个端点和最后四个端点链接在一起的机会，使它们更容易管理。它们被分组在一起是因为它们共享相同的 URI 路径。而不是独立的函数，每个 HTTP 方法将是一个类上的功能方法。而且，这个类将被分配一个公共的 URI 路径。一点代码应该能让你更容易理解：

```py
from sanic import Blueprint, Request, HttpMethodView
bp = Blueprint("MemberProfiles", url_prefix="/profile")
class AllProfilesView(HttpMethodView):
    async def get(request: Request):
        """same as fetch_all_profiles() from before"""
    async def post(request: Request):
        """same as create_new_profile() from before"""
class SingleProfileView(HttpMethodView):
    async def get(request: Request, username: str):
        """same as fetch_single_profile() from before"""
    async def put(request: Request, username: str):
        """same as replace_profile() from before"""
    async def patch(request: Request, username: str):
        """same as update_profile() from before"""
    async def delete(request: Request, username: str):
        """same as delete_profile() from before"""
app.add_route(AllProfilesView.as_view(), "")
app.add_route(SingleProfileView.as_view(), "/<username>")
```

> **重要提示**
> 
> 在本书的后面，我们可能会看到更多使用自定义装饰器来添加共享功能的情况。值得一提的是，我们也可以很容易地将它们添加到 CBV 中，我强烈建议你花点时间查阅用户指南，看看它是如何工作的：[`sanicframework.org/en/guide/advanced/class-based-views.html#decorators`](https://sanicframework.org/en/guide/advanced/class-based-views.html#decorators)
> 
> 在向 CBV 方法添加装饰器时，需要注意的一点是实例方法的 `self` 参数。你可能需要调整你的装饰器，或者使用 `staticmethod` 来使其按预期工作。上面提到的文档解释了如何做到这一点。

之前，我们看到了如何使用 `add_route` 作为将单个函数作为处理程序附加到路由定义的替代方法。它看起来是这样的：

```py
async def handler(request: Request):
...
app.add_route(handler, "/path")
```

这种模式是将 CBV 附加到 Sanic 或 Blueprint 实例的主要方式之一。需要注意的是，你需要使用类方法 `as_view()` 来调用它。在我们之前的例子中，我们看到了它的样子：

```py
app.add_route(SingleProfileView.as_view(), "/<username>")
```

这也可以通过在声明时附加 CBV 来实现。这个选项只在你已经有一个已知的 Blueprint 或 Application 实例时才有效。我们将重写 `SingleProfileView` 以利用这种替代语法。

```py
class SingleProfileView(
HttpMethodView,
attach=app,
uri="/<username>"
):
async def get(request: Request, username: str):
        """same as fetch_single_profile() from before"""
    async def put(request: Request, username: str):
        """same as replace_profile() from before"""
    async def patch(request: Request, username: str):
        """same as update_profile() from before"""
    async def delete(request: Request, username: str):
        """same as delete_profile() from before"""
```

你应该如何决定使用哪一个？我个人觉得第二个版本更容易、更简洁。但最大的缺点是，你不能懒加载 CBV 并稍后附加，因为它需要提前知道。

### 对 OPTIONS 和 HEAD 的全面支持

通常，在所有端点上支持`OPTIONS`和`HEAD`方法是一种最佳实践，只要这是合适的。这可能会变得繁琐，包括大量的重复模板代码。仅使用标准路由定义来实现这一点，就需要大量的代码重复，如下所示。下面，我们看到我们需要两个路由定义，而实际上只需要一个。现在想象一下，如果每个端点都需要有`OPTIONS`和`HEAD`会怎样！

```py
@app.get("/path/to/something")
async def do_something(request: Request):
    ...
@app.post("/path/to/something")
async def do_something(request: Request):
    ...
@app.options("/path/to/something")
async def do_something_options(request: Request):
    ...
@app.head("/path/to/something")
async def do_something_head(request: Request):
    ...
```

我们可以利用 Sanic 的路由器来添加处理程序，为每个路由添加处理这些请求的处理程序。想法将是遍历我们应用程序中定义的所有路由，并在需要时动态添加`OPTIONS`和`HEAD`的处理程序。在*第七章*的后面，我们将使用这种策略来创建我们的自定义 CORS 策略。然而，现在我们只需要记住，我们希望能够使用以下 HTTP 方法之一来处理对有效端点的任何请求：

```py
async def options_handler(request: Request):
    ...
async def head_handler(request: Request):
    ...
@app.before_server_start
def add_info_handlers(app: Sanic, _):
    app.router.reset()
    for group in app.router.groups.values():
        if "OPTIONS" not in group.methods:
            app.add_route(
                handler=options_handler,
                uri=group.uri,
                methods=["OPTIONS"],
                strict_slashes=group.strict,
            )
    app.router.finalize()
```

让我们仔细看看这段代码。

首先，我们创建路由处理程序：当端点被击中时将执行工作的函数。现在，它们不做任何事情。如果您想知道这个端点*可能*做什么，请跳转到*设置有效的 CORS 策略*中的 CORS 讨论，该讨论位于*第七章*。

```py
async def options_handler(request: Request):
    ...
async def head_handler(request: Request):
    ...
```

在我们注册了所有端点之后，下一部分需要完成。在*第十一章*中，我们通过在工厂内部运行代码来完成这个任务。您可以随意提前查看那里的示例，以便能够将其与我们的当前实现进行比较。

在我们的当前示例中，我们没有工厂，而是在事件监听器内部添加路由。通常情况下，这是不可能的，因为我们不能在应用程序运行后更改我们的路由。当 Sanic 应用程序启动时，它内部首先做的事情之一就是调用`app.router.finalize()`。但是，它不会让我们调用这个方法两次。因此，我们需要在所有动态路由生成完成后运行`app.router.reset()`，添加我们的路由，并最终在所有动态路由生成完成后调用`app.router.finalize()`。您可以在可能动态添加路由的任何地方使用这种相同的策略。这是一个好主意吗？一般来说，我会说动态添加路由不是一个好主意。端点的变化可能会导致不可预测性，或者在分布式应用程序中出现奇怪的错误。然而，在这个例子中，通过动态路由生成获得的收益是巨大的，风险非常低。

Sanic 路由器为我们提供了一些不同的属性，我们可以遍历它们来查看注册了哪些路由。最常用于公共消费的两个属性是`app.router.routes`和`app.router.groups`。了解它们是什么以及它们之间的区别是有帮助的。我们将暂时暂停对`OPTIONS`和`HEAD`的讨论，来看看这两个属性是如何实现的：

```py
@app.before_server_start
def display(app: Sanic, _):
    for route in app.router.routes:
        print(route)
    for group in app.router.groups.values():
        print(group)
@app.patch("/two")
@app.post("/two")
def two_groups(request: Request):
    return text("index")
@app.route("/one", methods=["PATCH", "POST"])
def one_groups(request: Request):
    return text("index")
```

首先，要注意的是，其中一个是生成`Route`对象，另一个是生成`RouteGroup`对象。第二个明显的启示是，一个是列表，另一个是字典。但`Route`和`RouteGroup`是什么？

在我们的控制台，我们会看到有三个`Route`对象，但只有两个`RouteGroup`对象。这是因为 Sanic 将看起来相似的路径组合在一起，以便更有效地匹配它们。`Route`是一个单独的定义。每次我们调用`@app.route`时，我们都在创建一个新的`Route`。在这里，我们可以看到它们是根据 URI 路径进行分组的：

```py
<Route: name=__main__.two_groups path=two>
<Route: name=__main__.two_groups path=two>
<Route: name=__main__.one_groups path=one>
<RouteGroup: path=two len=2>
<RouteGroup: path=one len=1>
```

回到我们对自动化的讨论，我们将使用`app.router.groups`。这是因为我们想知道哪些方法被分配给了特定的路径，哪些没有被分配。最快的方法是查看 Sanic 为我们提供的组。我们只需要检查该组是否已经包含了一个处理 HTTP 方法的处理程序（这样我们就不会覆盖已经存在的内容），然后调用`add_route`。

```py
for group in app.router.groups.values():
    if "OPTIONS" not in group.methods:
        app.add_route(
            handler=options_handler,
            uri=group.uri,
            methods=["OPTIONS"],
            strict_slashes=group.strict,
        )
    if "GET" in group.methods and "HEAD" not in group.methods:
        app.add_route(
            handler=head_handler,
            uri=group.uri,
            methods=["HEAD"],
            strict_slashes=group.strict,
        )
```

尽管我们现在不会查看`options_handler`，但我们可以更仔细地看看`head_handler`。根据 RFC 2616 的定义，`HEAD`请求与`GET`请求相同：“*HEAD 方法与 GET 方法相同，除了服务器在响应中不得返回消息体。*”（[`www.w3.org/Protocols/rfc2616/rfc2616-sec9.html#sec9.4`](https://www.w3.org/Protocols/rfc2616/rfc2616-sec9.html#sec9.4))。

在 Sanic 中实现这一点相当简单。实际上我们想要做的是检索同一端点的`GET`处理程序的响应，但只返回元数据，*而不是*请求体。我们将使用`functools.partial`将`GET`处理程序传递给我们的`head_handler`。然后，它只需要运行`get_handler`并返回响应。正如我们在本章前面看到的，Sanic 会在发送响应到客户端之前为我们处理移除体（body）的工作：

```py
from functools import partial
for group in app.router.groups.values():
    if "GET" in group.methods and "HEAD" not in group.methods:
        get_route = group.methods_index["GET"]
        app.add_route(
            handler=partial(
                head_handler,
                get_handler=get_route.handler
            ),
            uri=group.uri,
            methods=["HEAD"],
            strict_slashes=group.strict,
            name=f"{get_route.name}_head",
        )
async def head_handler(request: Request, get_handler, *args, **kwargs):
    return await get_handler(request: Request, *args, **kwargs)
```

> **重要提示**
> 
> 在上面的例子中，我们在`add_route`方法中添加了`name=f"{get_route.name}_head"`。这是因为 Sanic 中的所有路由都有一个“名称”。如果你没有手动提供，那么 Sanic 会尝试使用`handler.__name__`为你生成一个名称。在这种情况下，我们传递了一个`partial`函数作为路由处理程序，而 Sanic 不知道如何为它生成名称，因为 Python 中的`partial`函数没有`__name__`属性。

现在我们已经了解了如何利用 HTTP 方法来获取优势，我们将探讨路由中的下一个重要领域：**路径**。

## 路径、斜杠以及它们的重要性

在石器的时代，当互联网被发明出来时，如果你导航到一个 URL，你实际上是在接收一个存在于某台电脑上的文件。如果你请求`/path/to/something.html`，服务器会在`/path/to`目录中寻找一个名为`something.html`的文件。如果该文件存在，它会发送给你。

虽然这种情况仍然存在，但许多应用的时代确实已经改变。互联网在很大程度上仍然基于这个前提，但通常发送的是生成的文档而不是静态文档。尽管如此，仍然保持这种心理模型在脑海中是有帮助的。认为你的 API 路径应该指向某种资源将帮助你避免某些 API 设计缺陷。例如：

```py
/path/to/create_something  << BAD
/path/to/something         << GOOD
```

你的 URI 路径应该使用名词，而不是动词。如果我们想执行一个动作并告诉服务器做什么，我们应该像我们学的那样操作 HTTP 方法，而不是 URI 路径。走这条路——相信我，我试过——会导致一些看起来很混乱的应用程序。很可能有一天早上醒来，你会看到一堆零散和不连贯的路径，然后问自己：我做了什么？然而，可能会有时间和地点适合这样做，所以我们很快会重新讨论这个问题。

知道我们的路径应该包含名词，接下来的明显问题是它们应该是单数还是复数。我认为在互联网上关于这里什么是对的并没有一个单一的共识。许多人总是使用复数；许多人总是使用单数；还有一些人决定混合使用。虽然这个决定本身可能看起来很小，但它仍然很重要，需要建立一致性。选择一个系统并保持一致性本身比实际的决定更重要。

在这个问题解决之后，我将给出我的观点。使用复数名词。为什么？它使得路径的嵌套非常优雅，这可以很好地转化为 Blueprints 的嵌套：

```py
/users      << to get all users
/users/123  << to get user ID 123
```

我确实鼓励你在觉得合理的情况下使用单数名词。但如果你这样做，你必须无处不在。只要你在选择上保持一致和逻辑，你的 API 就会显得很精致。混合单数和复数路径会让你的 API 显得杂乱无章和业余。这里有一个非常好的资源，解释了如何*一致地*打破我刚刚提出的两个规则（使用名词，使用复数）：[`restfulapi.net/resource-naming/`](https://restfulapi.net/resource-naming/)。再次强调，对于我们来说，不仅学习*规则*或做某事的*正确方式*很重要和有帮助，而且学习何时打破它们，或何时制定我们自己的规则也很重要。有时遵循标准是有意义的，有时则不然。这就是我们从仅仅能够制作网络应用的人，变成知道如何设计和构建网络应用的人的过程。这种区别就是专业知识。

在设计路径时，也鼓励优先使用连字符（`-`）而不是空格、大写字母或下划线。这增加了 API 的人阅读性。考虑一下这些之间的区别：

```py
/users/AdamHopkins        << BAD
/users/adam_hopkins       << BAD
/users/adam%20hopkins     << BAD
/users/adam-hopkins       << GOOD
```

大多数人都会认为最后一个选项是最容易阅读的。

### 严格的斜杠

由于传统的范式，其中端点等同于服务器的文件结构，路径中的尾部斜杠获得了特定的含义。人们普遍认为，带有和没有尾部斜杠的路径是不同的，不能互换。

如果您导航到`/characters`，您可能会期望收到我们虚构社交媒体应用中所有字符的列表。然而，`/characters/`在技术上意味着*显示* `characters` *目录中的所有内容*。因为这可能令人困惑，您被鼓励避免使用尾部斜杠。

另一方面，人们普遍认为这些*确实是*同一回事。事实上，许多浏览器（和网站）都把它们视为相同。我将向您展示您如何自己测试这一点：

打开您的网络浏览器并访问：[`sanic.readthedocs.io/en/stable/`](https://sanic.readthedocs.io/en/stable/)

现在打开第二个标签页并访问：[`sanic.readthedocs.io/en/stable`](https://sanic.readthedocs.io/en/stable)

它是同一页。事实上，似乎这个网络服务器打破了刚才提到的规则，并更喜欢没有尾部斜杠。那么，这让我们处于什么位置，我们应该实现什么？这完全取决于您自己决定，让我们看看我们如何在 Sanic 中控制它。

如果您什么都不做，Sanic 会为您删除尾部斜杠。然而，Sanic 确实提供了通过设置`strict_slashes`参数来控制尾部斜杠是否有意义的选项。考虑一个带有和没有尾部斜杠，以及带有和没有`strict_slashes`的应用程序设置：

```py
@app.route("/characters")
@app.route("/characters/")
@app.route("/characters", strict_slashes=True)
@app.route("/characters/", strict_slashes=True)
async def handler(request: Request):
    ...
```

上述定义将失败。为什么？当 Sanic 在路径定义中看到尾部斜杠时，它会将其删除，*除非* `strict_slashes=True`。因此，第一条和第二条路由被认为是相同的。此外，第三条路由也是相同的，因此导致冲突。

虽然普遍认为规则是尾部斜杠*应该*有含义，但对于仅是路径一部分的尾部斜杠来说，情况并非如此。RFC 2616，第 3.2.3 节指出，空路径（`""`）与单个斜杠路径（`"/"`）是同一回事。（[`datatracker.ietf.org/doc/html/rfc2616#section-3.2.3`](https://datatracker.ietf.org/doc/html/rfc2616#section-3.2.3)）

我整理了一个关于 Sanic 如何处理尾部斜杠可能场景的深入讨论。如果您正在考虑使用它，我建议您查看这里：[`community.sanicframework.org/t/route-paths-how-do-they-work/825.`](https://community.sanicframework.org/t/route-paths-how-do-they-work/825.)

如果让我发表意见，我会说不要使用它们。允许`/characters`和`/characters/`具有相同含义会更加宽容。因此，我个人会这样定义上述路由：

```py
@app.route("/characters")
async def handler(request: Request):
    ...
```

### 从路径中提取信息

在本节中，我们需要考虑的最后一件事情是从我们的请求中提取可用的信息。我们经常查看的第一个地方是 URI 路径。Sanic 提供了一种简单的语法来从路径中提取参数：

```py
@app.get("/characters/<name>")
async def profile(request: Request, name: str):
    print text(f"Hello {name}")
```

我们已经声明路径的第二部分包含一个变量。Sanic 路由器会提取这个变量，并将其作为参数注入到我们的处理器中。需要注意的是，如果我们不做其他操作，这种注入将是一个 `str` 类型的值。

Sanic 还提供了一个简单的机制来转换类型。假设我们想从消息源中检索一条消息，在数据库中查询它，并返回该消息。在这种情况下，我们的数据库调用需要 `message_id` 是一个 `int`。

```py
@app.get("/messages/<message_id:int>")
async def message_details(request: Request, message_id: int):
    ...
```

这个路由定义将告诉 Sanic 在注入之前将第二部分转换为 `int` 类型。同样重要的是要注意，如果值是无法转换为 `int` 的类型，它将引发一个 `404 Not Found` 错误。因此，参数类型不仅仅是类型转换，它还涉及到路由处理。

您可以参考下一节和用户指南，了解所有允许的参数类型。[`sanicframework.org/en/guide/basics/routing.html#path-parameters`](https://sanicframework.org/en/guide/basics/routing.html#path-parameters)

除了从路径本身提取信息外，我们可能还想查找用户数据的两个其他地方是查询参数和请求体。查询参数是 URL 中 `?` 后的部分：

`/characters?count=10&name=george`

我们应该如何决定信息应该通过路径、查询参数，还是作为表单或 JSON 体的部分传递？最佳实践规定，信息应该按照以下方式访问：

+   **路径参数**：描述我们正在寻找的资源的信息

+   **查询参数**：可用于过滤、搜索或排序响应的信息

+   **请求体**：所有其他内容

在应用程序开发初期就养成一个良好的习惯，了解不同可用的信息来源。第四章深入探讨了通过查询参数和请求体传递数据。当然，HTTP 路径本身也非常有价值。我们刚刚看到了精心设计路径可能有多么重要。接下来，我们将更深入地探讨从 HTTP 路径中提取数据。

## 高级路径参数

在最后一节中，我们学习了从动态 URL 路径中提取信息的基本知识，这些信息可以用于编码。这确实是所有 Web 框架的基本功能。许多框架也允许您指定路径参数应该是什么。我们了解到 `/messages/<message_id:int>` 会匹配 `/messages/123` 但不会匹配 `/messages/abc`。我们还了解了 Sanic 在将匹配路径段转换为整数方面提供的便利。

但是，对于更复杂的类型怎么办？或者如果我们需要在将匹配的值用于我们的应用程序之前修改它怎么办？在本节中，我们将探讨一些有助于实现这些目标的实用模式。

### 自定义参数匹配

默认情况下，Sanic 提供了八种可匹配的路径参数类型：

+   `str`：匹配任何有效的字符串

+   `slug`：匹配标准路径别名

+   `int`：匹配任何整数

+   `float`：匹配任何数字

+   `alpha`：仅匹配字母字符

+   `path`：匹配任何可展开的路径

+   `ymd`：匹配 `YYYY-MM-DD`

+   `uuid`：匹配一个 `UUID`

这些中的每一个都提供了一个与匹配参数相对应的类型。例如，如果你有这个路径：`/report/<report_date:ymd>`，你的处理程序中的 `date` 对象将是一个 `datetime.date` 实例：

```py
from datetime import date
@app.get("/report/<report_date:ymd>")
async def get_report(request: Request, report_date: date):
    assert isinstance(report_date, date)
```

这是一个非常有用的模式，因为它为我们完成了两件事。首先，它确保传入的请求是正确的格式。一个请求为 `/report/20210101` 将收到一个 `404 Not Found` 响应。其次，当我们开始在处理程序中处理那个 `report_date` 实例时，它已经被转换成了一个可用的数据类型：`date`。

当我们需要对标准类型之外的类型进行路由时会发生什么？Sanic 当然允许我们通过定义一个路径段的自定义正则表达式来实现第一部分。让我们想象一下，我们有一个想要匹配有效 IPv4 地址的端点：`/ip/1.2.3.4`。

这里最简单的方法是找到一个相关的正则表达式并将其添加到我们的路径段定义中：

```py
IP_ADDRESS_PATTERN = (
    r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
)
@app.get(f"/<ip:{IP_ADDRESS_PATTERN}>")
async def get_ip_details(request: Request, ip: str):
    return text(f"type={type(ip)} {ip=}")
```

现在，当我们访问我们的端点时，我们应该有一个有效的匹配：

```py
$ curl localhost:7777/1.2.3.4  
type=<class 'str'> ip='1.2.3.4'
```

使用正则表达式匹配还允许我们在有限数量的选项之间定义一个狭窄的端点：

```py
@app.get("/icecream/<flavor:vanilla|chocolate>")
async def get_flavor(request: Request, flavor: str):
    return text(f"You chose {flavor}")
```

现在，我们有了基于我们两个可用选择的路由：

```py
$ curl localhost:7777/icecream/mint
️ 404 — Not Found
==================
Requested URL /icecream/mint not found
$ curl localhost:7777/icecream/vanilla
You chose vanilla
```

虽然正则表达式匹配在某些时候非常有帮助，但问题在于输出仍然是一个 `str`。回到我们最初的 IPv4 示例，如果我们想要一个 `ipaddress.IPv4Address` 类的实例来工作，我们需要手动将匹配的值转换为 `ipaddress.IPv4Address`。

虽然如果你只有一个或两个处理程序，这可能看起来不是什么大问题，但如果你有十几个端点需要动态 IP 地址作为路径参数，这可能会变得繁琐。Sanic 的解决方案是自定义模式匹配。我们可以告诉 Sanic 我们想要创建自己的参数类型。为此，我们需要三样东西：

+   一个简短的描述符，我们将用它来命名我们的类型

+   一个函数，它将返回我们想要的值或在没有匹配时引发 `ValueError`

+   一个回退正则表达式，它也会匹配我们的值

在 IP 地址示例中：

1.  我们将参数命名为 `ipv4`

1.  我们可以使用标准库的 `ipaddress.ip_address` 构造函数

1.  我们已经从之前有了回退正则表达式。我们可以继续注册自定义参数类型：

    ```py
    import ipaddress
    app.router.register_pattern(
        "ipv4",
        ipaddress.ip_address,
        IP_ADDRESS_PATTERN,
    )
    @app.get("/<ip:ipv4>")
    async def get_ip_details(request: Request, ip: ipaddress.IPv4Address):
        return text(f"type={type(ip)} {ip=}")
    ```

现在，我们在处理程序中有一个更可用的对象（`ipaddress.IPv4Address`），并且我们还有一个非常容易重用的路径参数（`<ip:ipv4>`）。

那么，关于我们的第二个例子，冰淇淋口味呢？如果我们想要一个`Enum`或其他自定义模型，而不是`str`类型，那会怎么样？不幸的是，Python 标准库中没有用于解析冰淇淋口味的函数（也许有人应该构建一个），因此我们需要创建自己的：

1.  首先，我们将使用`Enum`创建我们的模型。为什么是`Enum`？这是一个保持我们的代码整洁和一致性的绝佳工具。如果我们设置的环境正确——因为我们已经在*第二章*中注意到了使用好工具——我们有一个单一的地方可以维护我们的口味，并使用代码补全：

    ```py
    from enum import Enum, auto
    class Flavor(Enum):
        VANILLA = auto()
        CHOCOLATE = auto()
    ```

1.  接下来，我们需要一个正则表达式，我们可以在稍后的路由定义中使用它来匹配传入的请求：

    ```py
    flavor_pattern = "|".join(
        f.lower() for f in Flavor.__members__.keys()
    )
    ```

    结果模式应该是：`vanilla|chocolate`。

1.  我们还需要创建一个函数，它将充当我们的解析器。其任务是返回我们的目标类型或引发`ValueError`：

    ```py
    def parse_flavor(flavor: str) -> Flavor:
    try:
        return Flavor[flavor.upper()]
    except KeyError:
        raise ValueError(f"Invalid ice cream flavor: {flavor}")
    ```

1.  我们现在可以继续将这个模式注册到 Sanic 中。就像之前的 IP 地址示例一样，我们有我们的参数类型名称，一个用于检查匹配的函数，以及一个回退正则表达式。

    ```py
    app.router.register_pattern(
    "ice_cream_flavor",
    parse_flavor,
    flavor_pattern,
    )
    ```

1.  现在我们已经注册了我们的模式，我们可以继续在所有的冰淇淋端点中使用它：

    ```py
    @app.get("/icecream/<flavor:ice_cream_flavor>")
    async def get_flavor(request: Request, flavor: Flavor):
    return text(f"You chose {flavor}")
    ```

    当我们现在访问端点时，我们应该有一个`Enum`实例，但仍然只接受匹配我们定义的两个口味之一的请求。美味！

    ```py
    $ curl localhost:7777/icecream/mint
    404 — Not Found
    ===============
    Requested URL /icecream/mint not found
    $ curl localhost:7777/icecream/vanilla
    You chose Flavor.VANILLA
    ```

这个例子中的关键是有一个好的解析函数。在我们的例子中，我们知道如果将不良口味输入到`Enum`构造函数中，它将引发`KeyError`。这是一个问题。如果我们的应用程序无法匹配`mint`，它将抛出`KeyError`，并且应用程序将响应`500 内部服务器错误`。这不是我们想要的。通过捕获异常并将其转换为`ValueError`，Sanic 能够理解这是预期的，并且它应该响应`404 未找到`。

### 修改匹配的参数值

正如我们所学的，使用路径参数类型在构建我们的 API 以响应预期请求和忽略不良路径方面非常有帮助。尽可能具体，这是最佳实践，以便我们的端点能够获取正确的数据。我们刚刚探讨了如何使用参数类型将匹配的值重铸为更有用的数据类型。但是，如果我们不关心改变值的`type`，而是实际值本身呢？

回到我们的角色配置文件应用示例，假设我们有一些包含**短横线**的 URL。如果你不熟悉短横线，它基本上是一个使用小写字母和短横线来在 URL 路径中创建人类友好内容的字符串。我们之前已经看到了一个例子：`/users/adam-hopkins`。

在我们的假设应用中，我们需要构建一个端点，该端点返回关于角色实例的详细信息。

1.  首先，我们将创建一个模型来描述字符对象的外观。

    ```py
    @dataclass
    class Character:
        name: str
        super_powers: List[str]
        favorite_foods: List[str]
    ```

1.  我们希望能够返回关于我们角色的具体细节。例如，端点 `/characters/george/name` 应该返回 `George`。因此，我们的下一个任务是定义我们的路由：

    ```py
    @app.get("/characters/<name:alpha>/<attr:slug>")
    async def character_property(request: Request, name: str, attr: str):
        character = await get_character(name)
        return json(getattr(character, attr))
    ```

1.  这是一个相当简单的路由。它搜索角色，然后返回请求的属性。让我们看看它在实际操作中的表现：

    ```py
    $ curl localhost:7777/characters/george/name
    "George"
    ```

1.  现在，让我们尝试获取乔治的超级能力。

    ![图片](img/file0.png)

    哎呀，发生了什么事？我们试图访问的属性是 `Character.super_powers`。但是，我们的端点接受缩写（因为它们对人们来说更容易阅读）。因此，我们需要转换属性。就像在前一节中，我们可以在处理程序内部**可以**转换我们的值一样，这会使解决方案的扩展变得更加困难。我们**可以**在我们的处理程序内部运行 `attr.replace("-", "_")`，也许这是一个可行的解决方案。它确实在处理程序中增加了额外的代码。幸运的是，我们还有另一个选择。这是一个很好的中间件用例，我们需要将所有缩写（例如 `this-is-a-slug`）转换为蛇形小写（例如 `this_is_snake_case`），以便将来可以编程使用。通过转换缩写，我们可以查找 `super_powers` 而不是 `super-powers`。

1.  让我们创建这个中间件：

    ```py
    @app.on_request
    def convert_slugs(request: Request):
        request.match_info = {
            key: value.replace("-", "_") 
            for key, value in request.match_info.items()
        }
    ```

    这将修改在路由处理程序执行之前 `Request` 实例。对我们这个用例来说，这意味着所有匹配的值都将从缩写形式转换为蛇形小写。请注意，我们在这个函数中**没有**返回任何内容。如果我们这样做，Sanic 会认为我们正在尝试通过提前返回来终止请求/响应周期。这不是我们的意图。我们只想修改 `Request`。

1.  让我们再次测试这个端点：

    ```py
    $ curl localhost:7777/characters/george/super-powers
    ["whistling","hand stands"]
    ```

1.  虽然中间件不是解决这个问题的唯一方法。Sanic 使用信号来分发应用程序可以监听的事件。而不是上面的中间件，我们可以使用信号做类似的事情，如下所示：

    ```py
    @app.signal("http.routing.after")
    def convert_slugs(request: Request, route: Route, handler, kwargs):
        request.match_info = {
            key: value.replace("-", "_") 
            for key, value in kwargs.items()
        }
    ```

如您所见，这是一个非常相似的实现。也许对我们作为开发者来说最大的区别是，信号为我们提供了更多的工作参数。尽管如此，坦白说，`route`、`handler` 和 `kwargs` 都是可以在 `Request` 实例中访问的属性。中间件和信号在第六章中进行了更深入的讨论。现在，只需知道这些是在路由处理程序之外改变请求/响应周期的两种方法。稍后我们将了解更多关于它们之间的区别以及何时可能更倾向于选择其中一种。

## API 版本控制

在 *第二章* *组织项目* 中，我们讨论了如何使用 Blueprints 实现 API 版本控制。如果您还记得，这只是一个在 Blueprint 定义中添加关键字值的问题。

给定下面的 Blueprint 定义，我们得到 URL 路径：`/v1/characters`：

```py
bp = Blueprint("characters", version=1, url_prefix="/characters")
@bp.get("")
async def get_all_characters(...):
    ...
```

`version`关键字参数在路由级别也是可用的。如果版本在多个地方定义（例如，在路由和蓝图上），则优先考虑范围最窄的。让我们看看版本可以在哪些不同的地方定义的例子，并看看结果是什么。我们将在路由级别、蓝图级别和蓝图组级别定义它：

```py
bp = Blueprint("Characters")
bp_v2 = Blueprint("CharactersV2", version=2)
group = Blueprint.group(bp, bp_v2, version=3)
@bp.get("", version=1)
async def version_1(...):
    ...
@bp_v2.get("")
async def version_2(...):
    ...
@bp.get("")
async def version_3(...):
    ...
app.blueprint(group, url_prefix="/characters")
```

我们现在有以下路由。仔细看看例子，看看我们是如何操纵蓝图和`version`参数来控制每个路径交付的处理程序的：

+   `/v1/characters <Route: name=main.Characters.version_1 path=v1/characters>`

+   `/v3/characters <Route: name=main.Characters.version_3 path=v3/characters>`

+   `/v2/characters <Route: name=main.CharactersV2.version_2 path=v2/characters>`

向端点路径添加版本相当简单。但为什么我们要这样做呢？这是一个好的做法，因为它使你的 API 对用户来说既灵活又一致稳定。通过允许端点进行版本控制，你保持了对其做出更改的能力，同时仍然允许旧请求不被拒绝。随着时间的推移，当你过渡你的 API 以添加、删除或增强功能时，这会带来极大的好处。

即使你的 API 的唯一消费者是你的自己的网站，仍然是一个好的做法来对 API 进行版本控制，这样你就有了一条更容易的升级路径，而不会导致应用程序退化。

使用版本“锁定”功能是一种常见的做法。这是一种创建所谓的 API 合约的形式。将 API 合约视为开发者对 API 将继续工作的承诺。换句话说，一旦你将 API 投入使用——尤其是如果你发布了文档——你就向用户承诺 API 将继续按原样运行。你可以自由地添加新功能，但任何不向后兼容的破坏性更改都违反了该合约。因此，当你确实需要添加破坏性更改时，版本可能是你工具箱中实现目标的好方法。

这里有一个例子。我们正在构建我们的角色资料数据库。我们 API 的第一个版本提供了一个创建新资料的端点，它看起来可能像这样：

```py
@bp.post("")
async def create_character_profile(request: Request):
    async create_character(name=request.json["name"], ...)
    ...
```

此端点是建立在假设传入的 JSON 体相对简单的基础上的，如下所示：

```py
{
    "name": "Alice"
}
```

当我们想要处理更复杂的使用案例时会发生什么？

```py
{
    "meta": {
        "pseudonuym": "The Fantastic Coder",
        "real_name": "Alice"
    },
    "superpowers": [
        {
            "skill": "Blazing fast typing skills"
        }
    ]
}
```

如果我们将太多逻辑放入其中，我们的路由处理程序可能会变得复杂、混乱，总体上难以维护。作为一个一般做法，我喜欢保持我的路由处理程序非常简洁。如果我看到我的代码在视图处理程序内部接近 50 行代码，我知道可能需要进行一些重构。理想情况下，我喜欢将它们保持在 20 行或更少。

我们可以保持代码整洁的一种方法是将这些用例分开。API 的版本 1 仍然可以使用更简单的数据结构创建角色，而版本 2 具有更复杂结构的处理能力。

### 我是否应该让所有路由都提升版本？

你可能想知道为什么你想要

通常情况下，你可能需要在单个端点增加版本，但不是所有端点都需要。这引发了一个问题：在未更改的端点上我应该使用哪个版本？最终，这将成为只能由应用程序决定的唯一问题。记住 API 的使用方式可能会有所帮助。

非常常见的是，当 API 结构有完全的断裂或某些重大的重构时，API 会提升版本。这可能会伴随着新的技术栈、新的 API 结构或设计模式。一个例子是 GitHub 将他们的 API 从 v3 更改为 v4。他们 API 的旧版本（v3）是 RESTful 的，类似于我们在本章前面讨论的。新版本（v4）基于 GraphQL（关于 GraphQL 的更多信息请见第十章）。这是 API 的完全重新设计。因为 v3 和 v4 完全不兼容，所以他们改变了版本号。

在 GitHub 的情况下，所有端点都需要更改，因为它实际上是一个全新的 API。然而，这种剧烈的变化并不是版本更改的唯一催化剂。如果我们只更改 API 的一小部分兼容性，而保持其余部分不变呢？

有些人可能会觉得在所有端点上实施新版本号是有意义的。实现这一目标的一种方法是在端点上添加多个路由定义：

```py
v1 = Blueprint("v1", version=1)
v2 = Blueprint("v2", version=2)
@v1.route(...)
@v2.route(...)
async def unchanged_route(...):
    ...
```

这种方法的缺点是可能会变得非常繁琐，难以维护。如果你需要在更改版本时向每个处理器添加新的路由定义，你可能会从一开始就放弃添加版本。请考虑这一点。

嵌套蓝图怎么样？有没有一个在启动时动态添加路由的功能？你能想到解决方案吗？在这本书的前面我们已经看到了各种工具和策略，可能会对我们有所帮助。现在放下这本书，打开你的电脑上的代码编辑器，尝试一下。我鼓励你尝试版本和嵌套，看看哪些是可能的，哪些是不可能的。

记住`app.router.routes`和`app.router.groups`吗？尝试将单个处理器添加到多个 Blueprints 中。或者尝试将相同的 Blueprints 添加到不同的组中。我挑战你找出一个模式，让相同的处理器在不同的版本上运行，而无需像上面示例中那样进行多次定义。从这一点开始，看看你能想出什么，不要像上面那样重复路由定义：

```py
v1 = Blueprint("v1", version=1)
v2 = Blueprint("v2", version=2)
@v1.route(...)
async def unchanged_route(...):
    ...
```

这里有一个实用的代码片段，你可以在开发过程中使用，以查看哪些路径被定义：

```py
from sanic.log import logger
@app.before_server_start
def display(app: Sanic, _):
    routes = sorted(app.router.routes, key=lambda route: route.uri)
    for route in routes:
        logger.debug(f"{route.uri} [{route.name}]")
```

回到我们的问题：我的所有路由都应该更新版本吗？有些人会说是的，但当只有一条路由发生变化时，更新所有路由的版本似乎人为地复杂。无论如何，如果这样做有意义，可以同时更新所有内容。

如果我们只想更新正在变化的路由，这又会带来另一个问题。我们应该更新到哪个版本呢？很多人会告诉你版本应该**只**是整数：`v1`、`v2`、`v99`等。我觉得这很受限制，而且它确实让以下一组端点感觉很不自然：

+   `/v1/characters`

+   `/v1/characters/puppets`

+   `/v1/characters/super_heroes`

+   `/v1/characters/historical`

+   `/v2/characters`

虽然我不否认这种方法，但它似乎**应该**为所有路由都有一个 `v2`，即使它们没有变化。我们正在努力避免这种情况。为什么不使用像语义版本控制那样的次版本号呢？拥有单个 `/v1.1` 端点似乎比单个 `/v2` 更自然和易于接受。再次强调，这将是根据您的应用程序需求以及将消费您的 API 的用户类型来决定的问题。如果您决定语义版本控制风格适合您的应用程序需求，您可以通过使用浮点数作为版本参数来添加它，如下所示：

```py
@bp.post("", version=1.1)
async def create_character_profile_enhanced(request: Request):
    async create_character_enhanced(data=request.json)
```

> **重要提示**
> 
> 语义版本控制是软件开发中的一个重要概念，但超出了这里的范围。简而言之，这个概念是通过声明由点连接的主版本号、次版本号和修订号来创建一个版本。例如：1.2.3。一般来说，语义版本控制表明主版本号的增加对应于向后不兼容的更改，次版本号对应于新功能，修订号对应于错误修复。如果您不熟悉它，我建议花些时间阅读相关的文档，因为它在软件开发中得到广泛应用：[`semver.org/`](https://semver.org/)
> 
> **提示**
> 
> 如果您打算让第三方集成您的 API，强烈建议您在端点中使用版本。如果 API 只打算由您的应用程序使用，这可能就不那么重要了。尽管如此，它可能仍然是一个有用的模式。因此，我建议对于新项目使用 `version=1`，对于替换现有 API 的项目使用 `version=2`，即使遗留应用程序没有版本方案。

### 版本前缀

在 Sanic 中使用版本的标准方式是 `version=<int>` 或 `version=<float>`。版本将**始终**插入到您的路径的最开始。无论您嵌套多深以及有多少层 `url_prefix`，都无关紧要。即使是深度嵌套的路由定义也可以只有一个版本，并且它将是路径中的第一个部分：`/v1/some/deeply/nested/path/to/handler`。

然而，当你试图在你的应用程序上构建多层时，这确实会带来一个问题。如果你想有一些 HTML 页面和一个 API，并基于它们的路径将它们分开，你会怎么做？考虑以下我们可能希望在应用程序中拥有的路径：

+   `/page/profile.html`

+   `/api/v1/characters/<name>`

注意，版本化的 API 路由是以`/api`开头的吗？由于 Sanic*总是*将版本放在路径的其他部分之前，因此仅使用 URI 和 Blueprint URI 前缀是无法控制的。然而，Sanic 在所有可以使用`version`的地方提供了一个`version_prefix`参数。默认值是`/v`，但请随意根据需要更新它。在下面的例子中，我们可以将整个 API 设计嵌套在一个蓝图组中，以自动将`/api`添加到每个端点的前面：

```py
group = Blueprint.group(bp1, bp2, bp3, version_prefix="/api/v")
```

> **提示**
> 
> 这里也有相同的路径参数。例如，你可以这样做：`version_prefix=/<section>/v`。只是确保你记住，`section`现在将作为每个路由处理器的注入关键字参数。

你现在应该已经很好地掌握了如何以及何时使用版本。它们是使你的 API 更加专业和可维护的强大工具，因为它们允许更灵活的开发模式。接下来，我们将探讨另一个创建应用程序代码灵活性和可重用性的工具：虚拟主机。

## 虚拟主机

一些应用程序可以从多个域名访问。这带来了只有一个应用程序部署来管理的优势，但能够服务多个域名。在我们的例子中，我们将想象我们已经完成了计算机冒险游戏社交媒体网站。API 确实是件了不起的事情。

事实上，这太令人难以置信了，Alice 和 Bob 都向我们提出了成为分销商和*白标*我们的应用程序，或为他们的社交媒体网站重用 API 的机会。在互联网世界中，这是一种相当常见的做法，一旦提供商构建了一个应用程序，其他提供商只需将他们的域名指向同一个应用程序，并像拥有自己的那样运营。为了实现这一点，我们需要有独特的 URL。

+   `mine.com`

+   `alice.com`

+   `bob.com`

所有这些域名都将设置其 DNS 记录指向我们的应用程序。这可以在应用程序内部不进行任何进一步更改的情况下工作。但如果我们需要知道哪个域名正在处理请求，并且为每个域名执行一些不同的操作呢？这些信息应该在我们的请求头中可用。这应该仅仅是一个检查头的问题：

```py
@bp.route("")
async def do_something(request: Request):
    if request.headers["host"] == "alice.com":
        await do_something_for_alice(request)
    elif request.headers["host"] == "bob.com":
        await do_something_for_bob(request)
    else:
        await do_something_for_me(request)
```

这个例子可能看起来很小很简单，但你可能可以想象复杂性如何增加。记住我之前提到我喜欢将每个处理器的代码行数保持到最小？这确实是一个你可以想象处理器可能会变得非常长的用例。

实际上，我们在这个端点所做的是基于主机的路由。根据传入请求的主机，我们将端点路由到不同的位置。

Sanic 已经为我们做了这件事。我们只需要将逻辑拆分成单独的路由处理器，并为每个处理器提供一个 `host` 参数。这样就能实现我们需要的路由，同时将其从我们的响应处理器中分离出来。

```py
@bp.route("", host="alice.com")
async def do_something_for_alice(request: Request)::
    await do_something_for_alice(request: Request)
@bp.route("", host="bob.com")
async def do_something_for_bob(request: Request):
    await do_something_for_bob(request: Request)
@bp.route("", host="mine.com")
async def do_something_for_me(request: Request):
    await do_something_for_me(request: Request)
```

如果你发现自己处于这种情况，你不需要为每个端点定义 `host`。只需要为那些你希望有基于主机路由的端点定义。遵循这个模式，我们可以在多个域名之间重用同一个应用程序，并且仍然有一些端点能够区分它们，而其他端点则对多个域名正在访问它的事实一无所知。

有一个重要的事情需要记住：如果你创建了一个具有主机级别路由的端点，那么该路径上的所有路由也必须具有它。例如，你不能这样做。注意第三个路由没有定义 `host` 参数。

以下示例将不会工作，并在启动时引发异常：

```py
@bp.route("", host="alice.com")
async def do_something_for_alice(request: Request)::
    await do_something_for_alice(request: Request)
@bp.route("", host="bob.com")
async def do_something_for_bob(request: Request):
    await do_something_for_bob(request: Request)
@bp.route("")
async def do_something_for_me(request: Request):
    await do_something_for_me(request: Request)
```

为了解决这个问题，确保所有可以组合在一起的路由都有一个 `host` 值。这样它们就可以被区分开来。如果其中之一有 `host`，那么它们都需要有。

我们已经一般性地讨论了在将网络请求路由到我们的响应处理器时需要考虑的所有因素。但是，我们还没有探讨 Sanic 如何将请求传递到静态内容（即你希望在 web 服务器上发送的实际文件，如图片和样式表）。接下来，我们将讨论一些使用和不使用 Sanic 的选项。

## 服务器静态内容

到目前为止，我们本章的所有讨论都是关于为响应动态生成内容。然而，我们确实讨论了在目录结构中传递文件是一个有效的用例，Sanic 支持这种情况。这是因为大多数网络应用程序都需要为一些静态内容提供服务。最常见的情况是为浏览器渲染的 JavaScript 文件、图片和样式表。现在，我们将深入了解静态内容，看看它是如何工作的，并且我们可以提供这种类型的内容。在了解 Sanic 是如何做到这一点之后，我们将看到另一种非常常见的模式，即使用代理在 Sanic 之外提供内容。

### 从 Sanic 服务器静态内容

我们的 `app` 实例上有一个名为 `app.static()` 的方法。该方法需要两个参数：

+   我们应用程序的 URI 路径

+   一个路径告诉 Sanic 它可以从哪里访问该资源

第二个参数可以是单个文件，也可以是目录。如果是目录，目录中的所有内容都将像我们在本章开头讨论的老式网络服务器一样可访问。

如果你计划为所有网络资源提供服务，这将非常有帮助。如果你的文件夹结构如下所示？

```py
.
├── server.py
└── assets
    ├── index.html
    ├── css
    │   └── styles.css
    ├── img
    │   └── logo.png
    └── js
        └── bundle.js
```

我们可以使用 Sanic 来提供所有这些资源，并使它们像这样可访问：

`app.static("/static", "./assets")`

这些资源现在可以访问：

`$ curl localhost:7777/static/css/styles.css`

### 使用 Nginx 提供静态内容

现在我们已经看到了如何使用 Sanic 提供静态文件，一个很好的下一个问题是：你应该这样做吗？

Sanic 在创建大多数 Web API 所需的动态端点方面非常快。它甚至在提供静态内容方面做得相当不错，将所有端点逻辑保持在单个应用程序中，甚至允许操作这些端点或重命名文件。正如我们在 *第一章*，*Sanic 和异步框架简介* 中讨论的那样，Sanic 应用程序也旨在快速构建。

然而，有一种可能更快的方法来提供静态内容。对于一个旨在通过你的 API 请求数据的浏览器单页应用程序，你最大的障碍之一将是减少首次页面渲染的时间。这意味着你必须尽可能快地将所有 JS、CSS、图片或其他文件打包到浏览器中，以减少渲染延迟。

因此，你可能想要考虑在 Sanic 前使用像 Nginx 这样的代理层。代理的目的将是（1）将任何请求发送到 API 通过 Sanic，并且（2）自己处理提供静态内容。如果你打算提供大量的静态内容，你可能想要考虑这个选项。Nginx 内置了一个缓存引擎，能够比任何 Python 应用程序更快地提供静态内容。

*第八章*，*运行服务器* 讨论了在决定是否使用 Nginx 和 Docker 等工具时需要考虑的部署策略。现在，我们将使用 Docker Compose 快速轻松地启动 Nginx。

1.  我们需要创建我们的 `docker-compose.yml` 清单：

    ```py
    version: "3"
    services:
      client:
        image: nginx:alpine
        ports:
          - 8888:80
        volumes:
          - ./nginx/default.conf:/etc/nginx/conf.d/default.conf
          - ./static:/var/www
    ```

    如果你不太熟悉 Docker Compose 或如何安装和运行它，你应该能够在网上找到大量的教程和信息。我们示例中的简单设置将需要你在 `docker-compose.yml` 文件中将 `./static` 的路径设置为你的静态资源所在的任何目录。

    > **提示**
    > 
    > 这是一个故意设计的非常简单的实现。你应该确保一个真实的 Nginx 部署包括像 TLS 加密和代理密钥这样的功能。查看用户指南以获取更多详细信息和使用说明。[`sanicframework.org/en/guide/deployment/nginx.html#nginx-configuration`](https://sanicframework.org/en/guide/deployment/nginx.html#nginx-configuration)

1.  接下来，我们将创建控制 Nginx 所需的 `.nginx/default.conf` 文件。

    ```py
    upstream example.com {
        keepalive 100;
        server 1.2.3.4:8000;
    }
    server {
        server_name example.com;
        root /var/www;
        location / {
            try_files $uri @sanic;
        }
        location @sanic {
            proxy_pass http://$server_name;
            proxy_set_header Host $host;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
        location ~* \.(jpg|jpeg|png|gif|ico|css|js|txt)$ {
            expires max;
            log_not_found off;
            access_log off;
        }
    }
    ```

    我们使用以下命令启动它：`$ docker-compose up`这里最重要的更改是服务器地址。你应该将`1.2.3.4:8000`更改为你的应用程序可以访问的任何地址和端口。请记住，这**不会**是`127.0.0.1`或`localhost`。由于 Nginx 将在 Docker 容器内运行，该本地地址将指向容器本身，而不是你的计算机的本地网络地址。相反，出于开发目的，你应该考虑将其设置为你的本地 IP 地址。

1.  第 3 步，你需要确保 Sanic 知道要在该网络地址上提供服务。你还记得我们在第二章中是如何说我们要运行 Sanic 的吗？就像这样：

    ```py
    $ sanic server:app -p 7777 --debug --workers=2
    ```

    在这个例子中，我们将将其更改为：

    ```py
    $ sanic server:app -H 0.0.0.0 -p 7777 --debug --workers=2
    ```

    我的本地 IP 地址是`192.168.1.7`，因此我将我的 Nginx 配置中的`upstream`块设置为：`server 192.168.1.7:7777;`。

1.  第 4 步，你现在应该能够访问`./static`目录中的任何静态文件。我有一个名为`foo.txt`的文件。我使用`curl`的`-i`标志来查看头部信息。重要的头部信息是`Expires`和`Cache-Control`。这些帮助你的浏览器缓存文件而不是重新请求它。

    ```py
    $ curl localhost:8888/foo.txt -i
    HTTP/1.1 200 OK
    Server: nginx/1.21.0
    Date: Tue, 15 Jun 2021 18:42:20 GMT
    Content-Type: text/plain
    Content-Length: 9
    Last-Modified: Tue, 15 Jun 2021 18:39:01 GMT
    Connection: keep-alive
    ETag: "60c8f3c5-9"
    Expires: Thu, 31 Dec 2037 23:55:55 GMT
    Cache-Control: max-age=315360000
    Accept-Ranges: bytes
    hello...
    ```

如果你尝试向一个不存在的文件发送请求，Nginx 会将该路由发送到你的 Sanic 应用程序。当涉及到代理和 Nginx 时，这种设置只是冰山一角。然而，这种策略对于 Python 网络应用来说是非常常见的。如前所述，当我们讨论第八章中的部署选项时，我们将更深入地探讨这个话题。

### 流式传输静态内容

还值得重申的是，Sanic 服务器是构建并旨在作为前端服务器。这意味着它可以在没有代理服务器的情况下作为你的入口点，包括提供静态内容。是否要代理的决定——至少与交付静态文件相关——可能是一个关于流量多少以及你的应用程序可能需要交付多少文件的问题。

另一个需要考虑的重要因素是，你的应用程序是否需要流式传输文件。流式传输将在第五章中深入讨论。让我们创建一个简单的网页来流式传输视频，看看它可能是什么样子。

1.  首先是 HTML。将其存储为`index.html`。

    ```py
    <html>
        <head>
            <title>Sample Stream</title>
        </head>
        <body>
            <video width="1280" height="720" controls>
                <source src="/mp4" type="video/mp4" />
            </video>
        </body>
    </html>
    ```

1.  接下来，找到一个你想要流式传输的`mp4`文件。它可以任何视频文件。如果你没有，你可以从像这样的网站免费下载一个示例文件：[`samplelib.com/sample-mp4.html`](https://samplelib.com/sample-mp4.html)。

1.  现在，我们将创建一个小的 Sanic 应用程序来流式传输该视频。

    ```py
    from sanic import Sanic, response
    @app.route("/mp4")
    async def handler_file_stream(request: Request):
        return await response.file_stream("/path/to/sample.mp4")
    app.static("/index.html", "/path/to/index.html")
    @app.route("/")
    def redirect(request: Request):
        return response.redirect("/index.html")
    ```

1.  正常运行服务器并在你的网页浏览器中访问它：[`localhost:7777`](http://localhost:7777)

你应该注意到根 URI（`/`）将你重定向到了`/index.html`。使用`app.static`，应用程序告诉 Sanic 它应该接受对`/index.html`的任何请求，并从服务器上位于`/path/to/index.html`的静态内容中返回。这应该是你上面提供的内容。希望你有播放按钮，现在你可以将视频流到你的浏览器中。享受吧！

## 摘要

本章涵盖了将 HTTP 请求转换为可用内容的大量内容。在 Web 框架的核心是其将原始请求转换为可执行处理器的功能。我们已经了解了 Sanic 是如何做到这一点的，以及我们如何使用 HTTP 方法、良好的 API 设计原则、路径、路径参数提取和静态内容来构建有用的应用程序。正如我们在本书前面所学，一点初步规划就能走得很远。在编写大量代码之前，考虑 HTTP 提供的工具以及 Sanic 如何让我们利用这些功能是非常有帮助的。

如果我们在第二章中很好地设置了目录，那么轻松地镜像那种结构并将蓝图嵌套以匹配我们预期的 API 设计应该会非常容易。

本章有一些关键要点。你应该有目的地、深思熟虑地设计你的 API 端点路径——使用名词——指向预期的目标或资源。然后，应该使用 HTTP 方法作为动词，告诉你的应用程序和用户对那个目标或资源*做什么*。最后，你应该从那些路径中提取有用的信息，以便在处理程序中使用。

我们主要关注原始 HTTP 请求的第一行：HTTP 方法和 URI 路径。在下一章中，我们将深入探讨从请求中提取更多数据，包括头部和请求体。
