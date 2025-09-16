# 6 在请求处理器之外操作

在 Sanic 中，应用程序开发的基本构建块是请求处理器，有时也称为“路由处理器”。这些术语可以互换使用，意思相同。这是当请求被路由到你的应用程序进行处理和响应时 Sanic 运行的函数。这是业务逻辑和 HTTP 逻辑结合的地方，允许开发者规定如何将响应发送回客户端。这是学习如何使用 Sanic 构建时的明显起点。

然而，仅请求处理器本身并不能提供足够的强大功能来创建一个精致的应用程序体验。为了构建一个精致且专业的应用程序，我们必须跳出处理器，看看 Sanic 还能提供哪些其他工具。现在是时候考虑 HTTP 请求/响应周期不再局限于单个函数了。我们将扩大我们的范围，使得响应请求不再仅仅是处理器的责任，而是整个应用程序的责任。当我们瞥见中间件时，我们已经尝到了这种味道。

在本章中，我们将介绍以下主题：

+   利用 ctx

+   使用中间件修改请求和响应

+   利用信号进行工作内通信

+   掌握 HTTP 连接

+   实现异常处理

+   背景任务处理

当然，并非所有项目都需要这些功能，但当它们被用在正确的地方时，它们可以非常强大。您是否曾经在家中的 DIY 项目中工作，但并没有找到合适的工具？当您需要一字螺丝刀，但只有平头螺丝刀时，这可能会非常令人沮丧且效率低下。没有合适的工具会使任务变得更难，有时也会降低您能完成的工作质量。

想象一下我们在本章中探索的功能就像工具。你可能听说过这样一句俗语：“*如果你手里拿着锤子，那么每个问题看起来都像钉子*。”幸运的是，我们有一系列工具，我们的工作现在就是学习如何使用它们。我们即将探索 Sanic 工具包，看看我们能解决哪些问题。

## 技术要求

在本章中，您应该拥有与之前章节相同的工具，以便能够跟随示例（IDE、现代 Python 和 curl）。

您可以在 GitHub 上找到本章的源代码：[`github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/06`](https://github.com/PacktPublishing/Web-Development-with-Sanic/tree/main/chapters/06)。

## 利用 ctx

在我们开始工具包之前，还有一个概念我们必须熟悉。它在 Sanic 中相当普遍，你会在很多地方看到它。我说的就是：`ctx`。那是什么？

它代表*上下文*。这些`ctx`对象可以在多个地方找到，不利用它们来构建是不切实际的。它们允许将状态从应用程序的一个位置传递到另一个位置。它们是为了开发者自己的使用而存在的，你应该自由地按照自己的意愿使用它们。也就是说，`ctx`对象是你的，你可以添加信息而不用担心名称冲突或以其他方式影响 Sanic 的操作。

最常见的例子是数据库连接对象。你只创建一次，但你想在许多地方访问它。这是怎么做到的？

```py
@app.before_server_start
async def setup_db(app, loop):
    app.ctx.db = await setup_my_db()
```

现在，你可以在任何可以访问应用实例的地方访问数据库实例。例如，你可以在某个函数内部访问它：

```py
from sanic import Sanic
async def some_function_somewhere():
    app = Sanic.get_app()
    await app.ctx.db.execute(...)
Or, perhaps you need it in your route handler:
bp = Blueprint("auth")
@bp.post("/login")
async def login(request: Request):
    session_id = await request.app.ctx.db.execute(...)
    ...
```

这里列出了所有具有`ctx`对象的位置：

| **对象** | **描述** | **示例** |
| --- | --- | --- |
| Sanic | 在你的工作实例整个生命周期内可用。它是针对工作实例特定的，这意味着如果你运行多个工作实例，它们将*不会*保持同步。最适合用于连接管理，或其他需要在应用实例整个生命周期内可用的东西。 | `app.ctx` |
| 蓝图 | 在蓝图实例存在期间可用。这可能有助于你有一些特定的数据需要在整个工作实例生命周期内可用，但你又想控制它对特定蓝图附加内容的访问。 | `bp.ctx` |
| 请求 | 在单个 HTTP 请求期间可用。对于在中间件中添加详细信息并在处理程序或其他中间件中使其可用很有帮助。常见用途包括会话 ID 和用户实例。 | `request.ctx` |
| ConnInfo | 在整个 HTTP 连接期间（可能包括多个请求）可用。如果你使用代理，请特别小心。通常不应用于敏感信息。 | `request.conn_info.ctx` |
| 路由 | 在路由和信号实例上可用。这是 Sanic 实际上在 ctx 对象上存储一些细节的唯一例外。 | `request.route.ctx` |

表 6.1 - 使用`ctx`对象的 Sanic 特性

我们将经常回到`ctx`对象。它们在 Sanic 中是一个非常重要的概念，允许传递任意的数据和对象。它们并不完全相同，你可能会发现自己比其他任何对象更频繁地使用`app.ctx`和`request.ctx`。

现在我们有了这个基本构建块，我们将看到这些对象是如何在应用程序中传递的。在下一节关于中间件的部分，我们将看到请求对象——因此也是`request.ctx`——如何在应用程序的多个地方被访问。

## 使用中间件修改请求和响应

如果你一直跟随这本书到现在，中间件的概念应该是熟悉的。这是你应该熟悉的第一件工具。

中间件是可以在路由处理器前后运行的代码片段。中间件有两种类型：请求和响应。

### 请求中间件

请求中间件按照声明的顺序执行，在路由处理器之前。

```py
@app.on_request
async def one(request):
    print("one")
@app.on_request
async def two(request):
    print("two")
@app.get("/")
async def handler(request):
    print("three")
    return text("done")
```

当我们尝试达到这个终点时，我们应该在终端看到以下内容：

```py
one
two
three
(sanic.access)[INFO][127.0.0.1:47194]: GET http://localhost:7777/  200 4
```

但是，这只能讲述故事的一部分。有时我们可能需要添加一些额外的逻辑来仅针对我们应用程序的**部分**。让我们假设我们正在构建一个电子商务应用程序。像其他在线商店一样，我们需要构建一个购物车来存放将要购买的产品。为了我们的示例，我们将想象当用户登录时，我们在数据库中创建购物车，并将其引用存储在 cookie 中。类似于这样：

```py
@app.post("/login")
async def login(request):
    user = await do_some_fancy_login_stuff(request)
    cart = await generate_shopping_cart(request)
    response = text(f"Hello {user.name}")
    response.cookies["cart"] = cart.uid
    return response
```

不要过于纠结于这里的细节。重点是，在每次后续请求中，都将会有一个名为 cart 的 cookie，我们可以用它从我们的数据库中获取数据。

现在，假设我们希望我们的`/cart`路径上的所有端点都能访问购物车。我们可能有添加项目、删除项目、更改数量等端点。然而，我们始终需要访问购物车。而不是在每个处理器中重复逻辑，我们可以在蓝图上做一次。将中间件添加到单个蓝图上的所有路由看起来和功能与应用范围中间件相似。

```py
bp = Blueprint("ShoppingCart", url_prefix="/cart")
@bp.on_request
async def fetch_cart(request):
    cart_id = request.cookies.get("cart")
    request.ctx.cart = await fetch_shopping_cart(cart_id)
@bp.get("/")
async def get_cart(request):
    print(request.ctx.cart)
    ...
```

正如我们所期望的，每个附加到`ShoppingCart`蓝图上的端点在运行处理器之前都会获取购物车。我相信你可以看到这种模式的价值。当你能够识别一组需要类似功能的路由时，有时最好将其提取到中间件中。这也是指出这一点的好时机，即这也适用于蓝图组。我们可以将中间件更改为这样，并产生相同的影响：

```py
group = Blueprint.group(bp)
@group.on_request
async def fetch_cart(request):
    cart_id = request.cookies.get("cart")
    request.ctx.cart = await fetch_shopping_cart(cart_id)
```

正如我们所期望的，属于该蓝图组的端点现在可以访问购物车。

知道我们可以在应用范围和蓝图特定范围内执行中间件，这引发了一个有趣的问题：它们应用的顺序是什么？无论它们声明的顺序如何，所有应用范围的中间件都将**始终**在蓝图特定中间件之前运行。为了说明这一点，我们将使用一个混合两种类型的示例。

```py
bp = Blueprint("Six", url_prefix="/six")
@app.on_request
async def one(request):
    request.ctx.numbers = []
    request.ctx.numbers.append(1)
@bp.on_request
async def two(request):
    request.ctx.numbers.append(2)
@app.on_request
async def three(request):
    request.ctx.numbers.append(3)
@bp.on_request
async def four(request):
    request.ctx.numbers.append(4)
@app.on_request
async def five(request):
    request.ctx.numbers.append(5)
@bp.on_request
async def six(request):
    request.ctx.numbers.append(6)
@app.get("/")
async def app_handler(request):
    return json(request.ctx.numbers)
@bp.get("/")
async def bp_handler(request):
    return json(request.ctx.numbers)
app.blueprint(bp)
```

正如这个示例所示，我们通过交替声明应用和蓝图中间件来穿插它们：首先是应用，然后是蓝图等。虽然代码按顺序列出函数（1，2，3，4，5，6），但我们的输出不会。你应该能够预测我们的端点将如何响应，应用编号将附加在蓝图编号之前。确实如此：

```py
$ curl localhost:7777     
[1,3,5]
$ curl localhost:7777/six
[1,3,5,2,4,6]
```

还有一点很有帮助的是指出，由于中间件只是传递`Request`对象，后续中间件可以访问早期中间件所做的任何更改。在这个例子中，我们在一个中间件中创建了数字列表，然后它对所有中间件都是可用的。

### 响应中间件

在 HTTP 生命周期的另一边，我们有响应中间件。请求中间件的规则同样适用：

+   它是根据声明的顺序执行的，*尽管它是反向顺序！*

+   响应中间件可以是应用范围内的或蓝图特定的

+   所有应用范围内的中间件将在任何蓝图特定中间件之前运行

在最后一节中，我们使用中间件从 1 计数到 6。我们将使用完全相同的代码（顺序很重要！），但将请求改为响应：

```py
bp = Blueprint("Six", url_prefix="/six")
@app.on_response
async def one(request, response):
    request.ctx.numbers = []
    request.ctx.numbers.append(1)
@bp.on_response
async def two(request, response):
    request.ctx.numbers.append(2)
@app.on_response
async def three(request, response):
    request.ctx.numbers.append(3)
@bp.on_response
async def four(request, response):
    request.ctx.numbers.append(4)
@app.on_response
async def five(request, response):
    request.ctx.numbers.append(5)
@bp.on_response
async def six(request, response):
    request.ctx.numbers.append(6)
@app.get("/")
async def app_handler(request):
    return json(request.ctx.numbers)
@bp.get("/")
async def bp_handler(request):
    return json(request.ctx.numbers)
```

现在，当我们访问我们的端点时，我们将看到不同的顺序：

```py
$ curl localhost:7777
500 — Internal Server Error
===========================
'types.SimpleNamespace' object has no attribute 'numbers'
AttributeError: 'types.SimpleNamespace' object has no attribute 'numbers' while handling path /
Traceback of __main__ (most recent call last):
  AttributeError: 'types.SimpleNamespace' object has no attribute 'numbers'
    File /path/to/sanic/app.py, line 777, in handle_request
    response = await response
    File /path/to/server.py, line 48, in app_handler
    return json(request.ctx.numbers)
```

哎呀，发生了什么事？嗯，因为我们直到响应中间件才定义了我们的`ctx.numbers`容器，所以在处理程序内部不可用。让我们快速修改一下。我们将在请求中间件内部创建该对象。为了我们的示例，我们还将从请求处理程序返回 None，而不是从最后一个中间件创建我们的响应。在这个例子中，最后一个响应中间件将是第一个声明的蓝图响应中间件。

```py
@bp.on_response
async def complete(request, response):
    return json(request.ctx.numbers)
@app.on_request
async def zero(request):
request.ctx.numbers = []
@app.on_response
async def one(request, response):
    request.ctx.numbers.append(1)
@bp.on_response
async def two(request, response):
    request.ctx.numbers.append(2)
@app.on_response
async def three(request, response):
    request.ctx.numbers.append(3)
@bp.on_response
async def four(request, response):
    request.ctx.numbers.append(4)
@app.on_response
async def five(request, response):
    request.ctx.numbers.append(5)
@bp.on_response
async def six(request, response):
    request.ctx.numbers.append(6)
@bp.get("/")
async def bp_handler(request):
    request.ctx.numbers = []
    return json("blah blah blah")
```

仔细看看上面的内容。我们仍然有应用和蓝图中间件的混合。我们在处理程序内部创建了数字容器。此外，重要的是要注意，我们正在使用与请求中间件相同的顺序，后者产生了：1, 3, 5, 2, 4, 6。这里的更改只是为了向我们展示响应中间件是如何反转其顺序的。你能猜出我们的数字将按什么顺序排列吗？让我们检查一下：

```py
$ curl localhost:7777/six
[5,3,1,6,4,2]
```

首先，所有应用范围内的响应中间件都会运行（按声明的反向顺序）。其次，所有蓝图特定的中间件都会运行（按声明的反向顺序）。当你创建响应中间件时，如果它们相互关联，请记住这个区别。

虽然请求中间件的常见用例是为请求对象添加一些数据以供进一步处理，但这对于响应中间件来说并不实用。我们上面的例子有点奇怪且不实用。那么响应中间件有什么用呢？最常见的情况可能是设置头和 cookie。

这里有一个简单（并且非常常见）的用例：

```py
@app.on_response
async def add_correlation_id(request: Request, response: HTTPResponse):
    header_name = request.app.config.REQUEST_ID_HEADER
    response.headers[header_name] = request.id
```

你为什么要这样做呢？许多 Web API 使用所谓的*关联 ID*来帮助识别单个请求。这对于日志记录目的很有帮助，对于跟踪请求在堆栈中通过各种系统时的流动，以及对于消费你的 API 的客户来说，跟踪正在发生的事情也很有帮助。Sanic 遵循这个原则，并将自动为你设置`request.id`。这个值将是来自传入请求头部的传入关联 ID，或者为每个请求生成一个唯一值。默认情况下，Sanic 将为这个值生成一个 UUID。你通常不需要担心这个问题，除非你想要使用除了 UUID 之外的东西来关联 Web 请求。如果你对如何覆盖 Sanic 生成这些值的逻辑感兴趣，请查看*第十一章*，*一个完整的真实世界示例*。

回到我们上面的例子，我们看到我们只是简单地获取那个值并将其附加到我们的响应头中。我们现在可以看到它在行动：

```py
$ curl localhost:7777 -i 
HTTP/1.1 200 OK
X-Request-ID: 1e3f9c46-1b92-4d33-80ce-cca532e2b93c
content-length: 9
connection: keep-alive
content-type: text/plain; charset=utf-8
Hello, world.
```

这段小代码是我强烈建议你添加到所有你的应用程序中的。当你与请求 ID 记录结合使用时，它非常有用。这也是我们将在*第十一章*中添加到我们的应用程序中的内容。

### 使用中间件提前（或延迟）响应

当我们探索上一节中的响应中间件排序示例时，你是否注意到了我们的响应中发生了一些奇怪的事情？你是否看到了这个：

```py
@bp.on_response
async def complete(request, response):
    return json(request.ctx.numbers)
...
@bp.get("/")
async def bp_handler(request):
    request.ctx.numbers = []
    return json("blah blah blah")
```

我们从处理器得到了一个无意义的响应，但它并没有被返回。这是因为在我们的中间件中我们返回了一个`HTTPResponse`对象。无论何时你从中间件返回一个值——无论是请求还是响应——Sanic 都会假设你正在尝试结束 HTTP 生命周期并立即返回。因此，你*永远*不应该从中间件返回以下任何内容：

+   不是`HTTPResponse`对象

+   不打算中断 HTTP 生命周期

然而，这个规则不适用于`None`值。如果你只是想停止中间件的执行，仍然可以使用`return None`。

```py
@app.on_request
async def check_for_politeness(request: Request):
    if "please" in request.headers:
        return None
    return text("You must say please")
```

让我们看看现在会发生什么：

```py
$ curl localhost:7777/show-me-the-money                
You must say please
$ curl localhost:7777/show-me-the-money -H "Please: With a cherry on top"
```

在第二个请求中，由于它有正确的头，因此它被允许继续进行。因此，我们可以看到从中间件返回`None`也是可以接受的。如果你熟悉在 Python 循环中使用`continue`，它大致有相同的影响：停止执行，并进入下一步。

> **重要提示**
> 
> 尽管我们正在寻找请求头部的`please`值，但我们能够传递`Please`并且它仍然可以工作，因为头总是不区分大小写的。

### 中间件和流式响应

你还应该知道关于中间件的一个额外的*陷阱*。记得我们简单地说中间件基本上是在路由处理器前后封装吗？这并不完全正确。

真诚地说，中间件封装了响应的生成。由于这*通常*发生在处理器的返回语句中，这就是我们为什么采取简单方法的原因。

如果我们回顾第五章的示例并使用我们的流处理程序，这一点很容易看出。这就是我们开始的地方：

```py
@app.get("/")
async def handler(request: Request):
    resp = await request.respond()
    for _ in range(4):
        await resp.send(b"Now I'm free, free-falling")
        await asyncio.sleep(1)
    await resp.eof()
```

让我们添加一些打印语句和一些中间件，以便我们可以检查执行顺序。

```py
@app.get("/")
async def handler(request: Request):
    print("before respond()")
    resp = await request.respond()
    print("after respond()")
    for _ in range(4):
        print("sending")
        await resp.send(b"Now I'm free, free-falling")
        await asyncio.sleep(1)
    print("cleanup")
    await resp.eof()
    print("done")
@app.on_request
async def req_middleware(request):
    print("request middleware")
@app.on_response
async def resp_middleware(request, response):
    print("response middleware")
```

现在，我们将访问端点，并查看我们的终端日志：

```py
request middleware
before respond()
response middleware
after respond()
sending
(sanic.access)[INFO][127.0.0.1:49480]: GET http://localhost:7777/  200 26
sending
sending
sending
cleanup
done
```

正如我们所预期的那样，请求中间件首先运行，然后我们开始路由处理程序。但是，响应中间件在我们调用 `request.respond()` 之后立即运行。对于大多数响应中间件的用例（如添加头信息），这通常不会造成问题。然而，如果你绝对必须在路由处理程序完成后执行一些代码，那么这就会成为一个问题。在这种情况下，你的解决方案是使用信号，我们将在本章后面探讨。

但首先，我们将探讨信号，有时它们可以很好地替代中间件。虽然中间件本质上是一个工具，它允许我们在路由处理器的限制之外扩展业务逻辑，并在不同的端点之间共享它，但我们会了解到信号更像是允许我们在 Sanic 中插入代码的断点。

## 利用信号进行工作内通信

通常，Sanic 试图让开发者能够扩展其功能以创建自定义解决方案。这也是为什么在与 Sanic 接口时，有多个选项可以注入自定义类来接管、更改或以其他方式扩展其功能。例如，你知道你可以替换其 HTTP 协议，从而将 Sanic 实质上变成一个 FTP 服务器（或任何其他基于 TCP 的协议）吗？或者，你可能想扩展路由功能？

这类定制相当高级。我们不会在本书中涵盖它们，因为对于大多数用例来说，这就像是用手锤在墙上钉钉子一样。

Sanic 团队引入信号作为一种在更用户友好的格式中扩展平台功能的方法。非常有意地，设置信号处理程序看起来和感觉就像是一个路由处理程序：

```py
@app.signal("http.lifecycle.begin")
async def connection_begin(conn_info):
    print("Hello from http.lifecycle.begin")
```

你可能会问：这究竟是什么，我该如何使用它？在这个例子中，我们了解到 `http.lifecycle.begin` 是一个事件名称。当 Sanic 向客户端打开 HTTP 连接时，它会派发这个信号。然后 Sanic 会检查是否有任何处理程序正在等待它，并运行它们。因此，我们所做的就是设置一个处理程序来附加到该事件。在本章中，我们将更深入地探讨预定义的事件。但首先，让我们更仔细地检查信号的结构和操作。

### 信号定义

所有信号都通过其事件名称定义，该名称由三个部分组成。我们刚刚看到了一个名为 `http.lifecycle.begin` 的信号事件。显然，这三个部分是 `http`、`lifecycle` 和 `begin`。一个事件将 *仅* 有三个部分。

这很重要，因为尽管 Sanic 默认提供了一组信号，但它也允许我们在过程中创建自己的信号。因此，我们需要遵循这个模式。将第一部分视为命名空间，中间部分视为引用，最后部分视为动作。有点像这样：

```py
namespace.reference.action
```

以这种方式思考有助于我概念化它们。我喜欢把它们想象成路由。事实上，它们确实是！在底层，Sanic 以与路由处理器相同的方式处理信号处理器，因为它们继承自相同的基类。

如果信号本质上是一个路由，那么这意味着它也可以查找动态路径参数吗？是的！看看这个：

```py
@app.signal("http.lifecycle.<foo>")
async def handler(**kwargs):
    print("Hello!!!")
```

现在去访问应用中的任何路由，我们应该在我们的终端中看到以下内容：

```py
[DEBUG] Dispatching signal: http.lifecycle.begin
Hello!!!
[DEBUG] Dispatching signal: http.lifecycle.read_head
Hello!!!
[DEBUG] Dispatching signal: http.lifecycle.request
Hello!!!
[DEBUG] Dispatching signal: http.lifecycle.handle
Hello!!!
request middleware
response middleware
[DEBUG] Dispatching signal: http.lifecycle.response
Hello!!!
[INFO][127.0.0.1:39580]: GET http://localhost:7777/  200 20
[DEBUG] Dispatching signal: http.lifecycle.send
Hello!!!
[DEBUG] Dispatching signal: http.lifecycle.complete
Hello!!!
```

在继续查看可用的信号之前，还有一件事我们需要注意：条件。`app.signal()` 方法接受一个名为 condition 的关键字参数，可以帮助限制匹配的事件。只有与相同条件分发的事件才会被执行。

我们将查看一个具体的例子。

1.  首先添加一些请求中间件。

    ```py
    @app.on_request
    async def req_middleware(request):
        print("request middleware")
    ```

1.  然后添加一个信号来连接到我们的中间件（这将在稍后看到，它是一个内置的）。

    ```py
    @app.signal("http.middleware.before")
    async def handler(**kwargs):
        print("Hello!!!")
    ```

1.  现在，让我们在访问端点后查看我们的终端：

    ```py
    [DEBUG] Dispatching signal: http.middleware.before
    request middleware
    ```

    嗯嗯，我们看到信号已被分发，我们的中间件也运行了，但我们的信号处理器没有。为什么？`http.middleware.*` 事件是特殊的，因为它们只有在满足特定**条件**时才会匹配。因此，我们需要修改我们的信号定义以包含所需条件。

1.  将你的信号修改为添加条件，如下所示：

    ```py
    @app.signal("http.middleware.before", condition={"attach_to": "request"})
    async def handler(**kwargs):
        print("Hello!!!")
    ```

1.  再次访问端点。现在我们应该看到预期的文本。

    ```py
    [DEBUG] Dispatching signal: http.middleware.before
    Hello!!!
    request middleware
    ```

条件是你可以添加到自定义信号分发中的内容（继续阅读以了解更多关于*自定义信号*部分的内容）。它看起来像这样：

```py
app.dispatch("custom.signal.event", condition={"foo": "bar"})
```

大多数信号用例不需要这种方法。然而，如果你发现需要对信号分发进行更多控制，这可能正是你需要的工具。让我们把注意力转回到 Sanic 的内置信号，看看我们还能将信号附加到哪些其他事件上。

### 使用内置信号

有许多内置的信号我们可以使用。查看下面的表格，并在书中标记这一页。我强烈建议你在尝试解决问题时经常回到这个表格，看看你的选项。虽然这本书中我们提出的实现和使用可能很小，但你的任务是学习这个过程，这样你就可以更有效地解决你自己的应用需求。

首先，这些信号与路由相关。它们将在每个请求上执行。

| **事件名称** | **参数** | **描述** |
| --- | --- | --- |
| `http.routing.before` | `request` | 当 Sanic 准备解析传入路径到路由时 |
| `http.routing.after` | `request, route, kwargs, handler` | 在找到路由之后立即 |

表 6.2 - 可用的内置路由信号

其次，我们有与请求/响应生命周期特别相关的信号。

| **事件名称** | **参数** | **描述** |
| --- | --- | --- |
| `http.lifecycle.begin` | `conn_info` | 当建立 HTTP 连接时 |
| `http.lifecycle.read_head` | `head` | 在读取 HTTP 头部信息之后，但在解析之前 |
| `http.lifecycle.request` | `request` | 在创建请求对象之后立即 |
| `http.lifecycle.handle` | `request` | 在 Sanic 开始处理请求之前 |
| `http.lifecycle.read_body` | `body` | 每次从请求体中读取字节时 |
| `http.lifecycle.exception` | `request, exception` | 在路由处理程序或中间件中抛出异常时 |
| `http.lifecycle.response` | `request, response` | 在发送响应之前 |
| `http.lifecycle.send` | `data` | 每次将数据发送到 HTTP 传输时 |
| `http.lifecycle.complete` | `conn_info` | 当 HTTP 连接关闭时 |

表 6.3 - 可用的内置请求/响应生命周期信号

第三，我们有围绕每个中间件处理程序的事件。这些信号可能不是你经常使用的。相反，它们主要存在以供 Sanic 插件开发者受益。

| **事件名称** | **参数** | **条件** | **描述** |
| --- | --- | --- | --- |
| `http.middleware.before` | `request, response` | `{"attach_to": "request"} 或 {"attach_to": "response"}` | 在每个中间件运行之前 |
| `http.middleware.after` | `request, response` | `{"attach_to": "request"} 或 {"attach_to": "response"}` | 每个中间件运行之后 |

表 6.4 - 可用的内置中间件信号

最后，我们有服务器事件。这些信号与监听器事件一一对应。虽然你可以将它们作为信号调用，但描述中已指示，每个都有方便的装饰器。

| **事件名称** | **参数** | **描述** |
| --- | --- | --- |
| `server.init.before` | `app, loop` | 在服务器启动之前（相当于 `app.before_server_start`） |
| `server.init.after` | `app, loop` | 服务器启动后（相当于 `app.after_server_start`） |
| `server.shutdown.before` | `app, loop` | 在服务器关闭之前（相当于 `app.before_server_stop`） |
| `server.shutdown.after` | `app, loop` | 在服务器关闭之后（相当于 `app.after_server_stop`） |

表 6.5 - 可用的内置服务器生命周期信号

我想分享一个例子，说明信号的力量。我为 Sanic 用户提供了很多支持。如果你花过时间查看社区资源（无论是论坛还是 Discord 服务器），你很可能见过我帮助开发者解决问题。我真的喜欢参与开源软件的这一方面。

有一次，有人向我求助，他们遇到了中间件的问题。目标是使用响应中间件在服务器发送响应时记录有用的信息。问题是，当中间件中引发异常时，它将停止其他中间件的运行。因此，这个人无法记录每个响应。在其他响应中间件中引发异常的请求从未到达记录器。解决方案——你可能已经猜到了——是使用信号。特别是，`http.lifecycle.response`事件在这个用例中工作得非常完美。

为了说明这一点，这里有一些代码：

1.  设置两个中间件，一个用于日志记录，另一个用于引发异常。记住，它们需要按照你希望它们运行的顺序相反：

    ```py
    @app.on_response
    async def log_response(request, response):
        logger.info("some information for your logs")
    @app.on_response
    async def something_bad_happens_here(request, response):
        raise InvalidUsage("Uh oh")
    ```

1.  当我们访问任何端点时，`log_response`将*永远不会*被运行。

1.  为了解决这个问题，将`log_response`从中间件改为信号（只需更改装饰器即可）：

    ```py
    @app.signal("http.lifecycle.response")
    async def log_response(request, response):
        logger.info("some information for your logs")
    ```

1.  现在，当我们访问端点并遇到异常时，我们仍然会得到预期的日志：

    ```py
    [ERROR] Exception occurred in one of response middleware handlers
    Traceback (most recent call last):
      File "/home/adam/Projects/Sanic/sanic/sanic/request.py", line 183, in respond
        response = await self.app._run_response_middleware(
      File "_run_response_middleware", line 22, in _run_response_middleware
        from ssl import Purpose, SSLContext, create_default_context
      File "/tmp/p.py", line 23, in something_bad_happens_here
        raise InvalidUsage("Uh oh")
    sanic.exceptions.InvalidUsage: Uh oh
    [DEBUG] Dispatching signal: http.lifecycle.response
    [INFO] some information for your logs
    [INFO][127.0.0.1:40466]: GET http://localhost:7777/  200 3
    ```

我们还可以使用这个完全相同的信号来解决我们早期遇到的一个问题。记得当我们检查响应中间件并使用流处理程序得到一些令人惊讶的结果时吗？如果你回到本章的早期部分，我们会注意到响应中间件实际上是在响应对象创建时被调用的，而不是在处理程序完成后。我们可以在歌词流完之后使用`http.lifecycle.response`来包装。

```py
@app.signal("http.lifecycle.response")
async def http_lifecycle_response(request, response):
    print("Finally... the route handler is over")
```

这可能又是一个你放下书本进行探索的好时机。回到那个早期的流处理程序示例，并尝试一些这些信号。看看它们接收到的参数，并思考你如何使用它们。当然，了解它们发送的顺序也同样重要。

完成这些后，我们将探讨创建自定义信号和事件。

### 自定义信号

到目前为止，我们特别关注内置信号。它们是 Sanic 信号提供的狭义实现。虽然将它们视为允许我们在 Sanic 本身中插入功能的断点是有帮助的，但事实上，还有一个更通用的概念在发挥作用。

信号允许应用程序内部通信。因为它们可以作为后台任务异步发送，所以这可以成为应用程序的一部分通知另一部分发生了某事的方便方法。这引入了信号的重要概念之一：它们可以以内联或任务的形式发送。

到目前为止，我们看到的每个内置信号示例都是内联的。也就是说，Sanic 会在信号完成之前停止处理请求。这就是我们能够在生命周期中添加功能的同时保持一致流程的方式。

这可能并不总是期望的。事实上，很多时候当你想使用自定义信号实现自己的解决方案时，让它们作为后台任务运行，这给了应用程序在执行其他任务的同时继续响应请求的能力。

以记录为例。想象一下，我们回到了我们的示例，我们正在构建一个电子商务应用程序。我们想增强我们的访问日志，包括有关已认证用户（如果有）和他们购物车中物品数量的信息。让我们将我们之前的中间件示例转换为信号：

1.  我们需要创建一个信号，将用户和购物车信息拉到我们的请求对象上。同样，我们只需要更改第一行。

    ```py
    @app.signal("http.lifecycle.handle")
    async def fetch_user_and_cart(request):
        cart_id = request.cookies.get("cart")
        session_id = request.cookies.get("session")
        request.ctx.cart = await fetch_shopping_cart(cart_id)
        request.ctx.user = await fetch_user(session_id)
    ```

1.  为了我们的示例，我们想快速组合一些模型和类似这样的假 getter：

    ```py
    @dataclass
    class Cart:
        items: List[str]
    @dataclass
    class User:
        name: str
    async def fetch_shopping_cart(cart_id):
        return Cart(["chocolate bar", "gummy bears"])
    async def fetch_user(session_id):
        return User("Adam")
    ```

1.  这将足以让我们的示例运行起来，但我们想看到它。现在，我们将添加一个路由处理程序，它只是输出我们的`request.ctx`：

    ```py
    @app.get("/")
    async def route_handler(request: Request):
        return json(request.ctx.__dict__)
    ```

1.  我们现在应该看到我们的假用户和购物车如预期那样可用：

    ```py
    $ curl localhost:7777 -H 'Cookie: cart=123&session_id=456'
    {
      "cart": {
        "items": [
          "chocolate bar",
          "gummy bears"
        ]
      },
      "user": {
        "name": "Adam"
      }
    }
    ```

1.  由于我们想使用自己的访问日志，我们应该关闭 Sanic 的访问日志。在第二章中，我们决定将所有示例都这样运行：

    ```py
    $ sanic server:app -p 7777 --debug --workers=2
    ```

    我们现在将改变这一点。添加`--no-access-logs`：

    ```py
    $ sanic server:app -p 7777 --debug --workers=2 --no-access-logs
    ```

1.  现在，我们将添加我们自己的请求记录器。但是，为了说明我们想要表达的观点，我们将手动让我们的信号响应时间变长：

    ```py
    @app.signal("http.lifecycle.handle")
    async def access_log(request):
        await asyncio.sleep(3)
        name = request.ctx.user.name
        count = len(request.ctx.cart.items)
        logger.info(f"Request from {name}, who has a cart with {count} items")
    ```

1.  当你访问端点时，你将在日志中看到以下内容。你也应该体验到在日志出现之前，以及在你收到响应之前会有延迟。

    ```py
    [DEBUG] Dispatching signal: http.lifecycle.request
    [DEBUG] Dispatching signal: http.lifecycle.handle
    [INFO] Request from Adam, who has a cart with 2 items
    ```

1.  为了解决这个问题，我们将为我们的记录器创建一个自定义信号，并从`fetch_user_and_cart`中分发事件。让我们进行以下更改：

    ```py
    @app.signal("http.lifecycle.request")
    async def fetch_user_and_cart(request):
        cart_id = request.cookies.get("cart")
        session_id = request.cookies.get("session")
        request.ctx.cart = await fetch_shopping_cart(cart_id)
        request.ctx.user = await fetch_user(session_id)
        await request.app.dispatch(
            "olives.request.incoming",
            context={"request": request},
            inline=True,
        )
    @app.signal("olives.request.incoming")
    async def access_log(request):
        await asyncio.sleep(3)
        name = request.ctx.user.name
        count = len(request.ctx.cart.items)
        logger.info(f"Request from {name}, who has a cart with {count} items")
    ```

1.  这次当我们访问端点时，有两件事你需要注意。首先，你的响应应该几乎立即返回。我们之前遇到的延迟响应应该消失了。其次，访问日志中的延迟应该保持。

我们在这里有效地做的是将日志记录中的任何 I/O 等待时间从请求周期中移除。为了做到这一点，我们创建了一个自定义信号。这个信号被称作`olives.request.incoming`。这并没有什么特别之处。它是完全任意的。唯一的要求，正如我们讨论的那样，是它有三个部分。

要执行信号，我们只需要用相同名称调用`app.dispatch`：

```py
await app.dispatch("one.two.three")
```

因为我们想在`access_log`中访问请求对象，所以我们使用了可选参数`context`来传递对象。

那么，为什么`http.lifecycle.handle`信号延迟了响应，而`olives.request.incoming`没有？因为前者是**内联**执行的，而后者是作为后台任务执行的。在底层，Sanic 使用`inline=True`调用 dispatch。现在就添加到自定义调度中，看看它如何影响响应。再次强调，日志和响应现在都延迟了。你应该在想要你的应用程序在调度时暂停，直到所有附加的信号都运行完毕时使用这个。如果这个顺序不重要，如果你省略它，你会获得更好的性能。

dispatch 还接受一些可能对你有帮助的参数。以下是函数签名：

```py
def dispatch(
    event: str,
    *,
    condition: Optional[Dict[str, str]] = None,
    context: Optional[Dict[str, Any]] = None,
    fail_not_found: bool = True,
    inline: bool = False,
    reverse: bool = False,
):
```

这个函数接受的参数如下：

+   `condition`：用作中间件信号，以控制额外的匹配（我们看到了`http.middleware.*`信号的使用）

+   `context`：应传递给信号的参数

+   `fail_not_found`：如果你调度了一个不存在的事件，会发生什么？应该抛出异常还是静默失败？

+   `inline`：是否在任务中运行，如之前讨论的那样

+   `reverse`：当事件上有多个信号时，它们应该按什么顺序运行？

### 等待事件

分派信号事件的最后一个有用之处在于，它们也可以像 asyncio 事件一样使用，以阻塞直到它们被调度。这种用例与调度不同。当你调度一个信号时，你正在导致其他操作发生，通常是在后台任务中。当你想要暂停现有任务直到该事件发生时，你应该等待信号事件。这意味着它将阻塞当前存在的任务，无论是后台任务还是正在处理的实际请求。

最简单的方法是使用一个在应用程序中持续运行的超级简单的循环来展示这一点。

1.  按照以下方式设置你的循环。注意我们使用`app.event`和我们的事件名称。为了简单起见，我们使用了一个内置的信号事件，但它也可以是自定义的。为了使其工作，我们只需要一个与相同名称注册的 app.signal。

    ```py
    async def wait_for_event(app: Sanic):
        while True:
            print("> waiting")
            await app.event("http.lifecycle.request")
            print("> event found")
    @app.after_server_start
    async def after_server_start(app, loop):
        app.add_task(wait_for_event(app))
    ```

1.  现在，当我们访问我们的端点时，我们应该在日志中看到以下内容：

    ```py
    > waiting
    [INFO] Starting worker [165193]
    [DEBUG] Dispatching signal: http.lifecycle.request
    > event found
    > waiting
    ```

这可能是一个有用的工具，特别是如果你的应用程序使用 websockets。例如，你可能想跟踪打开套接字的数量。随时可以回到 websockets 示例，看看你是否可以将一些事件和信号集成到你的实现中。

另一个有用的用例是，在你响应之前，你的端点需要发生多个事件。你希望将一些工作推迟到信号，但最终它确实需要在响应前完成。

我们可以这样做。设置以下处理程序和信号。

```py
@app.signal("registration.email.send")
async def send_registration_email(email, request):
    await asyncio.sleep(3)
    await request.app.dispatch("registration.email.done")
@app.post("/register")
async def handle_registration(request):
    await do_registration()
    await request.app.dispatch(
        "registration.email.send",
        context={
            "email": "alice@bob.co",
            "request": request,
        },
    )
    await do_something_else_while_email_is_sent()
    print("Waiting for email send to complete")
    await request.app.event("registration.email.done")
    print("Done.")
    return text("Registration email sent")
```

现在我们查看终端时，应该看到以下内容：

```py
do_registration
Sending email
do_something_else_while_email_is_sent
Waiting for email send to complete
Done.
```

由于我们知道发送电子邮件将是一个昂贵的操作，我们将它发送到后台，同时继续处理请求。通过使用 app.event，我们能够等待`registration.email.done`事件被分发，然后回复电子邮件实际上已经发送。

在这个例子中，你应该注意的一点是，实际上并没有信号附加到`registration.email.done`上。默认情况下，Sanic 会抱怨并抛出异常。如果你想使用这种模式，你有三种选择。

1.  注册一个信号。

    ```py
    @app.signal("registration.email.done")
    async def noop():
        ...
    ```

1.  由于我们实际上不需要执行任何操作，所以我们实际上不需要一个处理器：

    ```py
    app.add_signal(None, "registration.email.done")
    ```

1.  告诉 Sanic 在发生分发时自动创建所有事件，无论是否有注册的信号：

    ```py
    app.config.EVENT_AUTOREGISTER = True
    ```

既然我们已经知道有几种方式可以在 HTTP 生命周期内控制业务逻辑的执行，我们将接下来探索我们可以利用新发现工具做的一些其他事情。

## 掌握 HTTP 连接

在第四章的早期，我们讨论了 HTTP 生命周期代表了客户端和服务器之间的对话。客户端请求信息，服务器响应。特别是，我们将它比作双向通信的视频聊天。让我们深入这个类比，以扩展我们对 HTTP 和 Sanic 的理解。

而不是将 HTTP 请求视为视频聊天，最好是将其视为一个单独的对话，或者更好，一个单一的问题和答案。类似于这样：

**客户**: 嗨，我的会话 ID 是 123456，我的购物车 ID 是 987654。你能告诉我我可以购买的其他商品吗？

**服务器**: 嗨亚当，你的购物车中已经有了纯橄榄油和特级初榨橄榄油。你可以添加：香醋或红葡萄酒醋。

Sanic 是一个“高性能”的 Web 框架，因为它能够同时与多个客户端进行这些对话。当它在为一位客户端获取结果时，它可以开始与其他客户端的对话：

**客户 1**: 你们销售哪些产品？

**客户 2**: 一桶橄榄油的价格是多少？

**客户 3**: 生活的意义是什么？

由于服务器能够同时对应多个视频聊天会话，因此它变得更加高效地响应。但是，当一个客户端有多个问题时会发生什么？为每个“对话”开始和结束视频聊天将会既耗时又昂贵。

*开始视频聊天*

**客户**: 这里是我的凭证，我可以登录吗？

**服务器**: 嗨亚当，很高兴再次见到你，这是你的会话 ID：123456。再见。

*结束视频聊天*

*开始视频聊天*

**客户**: 嗨，我的会话 ID 是 123456。我可以更新我的个人资料信息吗？

**服务器**: 哎呀，无效请求。看起来你没有发送正确的数据。再见。

*结束视频聊天*

每次视频聊天开始和停止时，我们都在浪费时间和资源。HTTP/1.1 通过引入持久连接来试图解决这个问题。这是通过 Keep-Alive 头部实现的。我们不需要担心客户端或服务器如何具体处理这个头部。Sanic 会相应地处理。

我们需要理解的是，它确实存在，并且包含一个超时时间。这意味着如果在这个超时期间有另一个请求到来，Sanic 不会关闭与客户端的连接。

*开始视频聊天*

**客户端**: 这是我的凭证，我可以登录吗？

**服务器**: 嗨，亚当，很高兴再次见到你，这是你的会话 ID：123456。

**服务器**: *等待中…*

**服务器**: *等待中…*

**服务器**: *等待中…*

**服务器**: 再见。

*停止视频聊天*

我们现在在单个视频聊天中创建了一个效率，允许进行多个对话。

我们需要考虑的两个实际问题： (1) 服务器应该等待多长时间？以及 (2) 我们能否使连接更高效？

### Sanic 中的 Keep-Alive

Sanic 默认会保持 HTTP 连接活跃。这使得操作更高效，正如我们之前看到的。然而，可能存在一些情况下这并不理想。也许你*永远*不想保持连接开启。如果你知道你的应用程序永远不会处理每个客户端超过一个请求，那么可能使用宝贵的内存来保持一个永远不会被重用的连接是浪费的。要关闭它，只需在你的应用程序实例上设置一个配置值：

```py
app.config.KEEP_ALIVE = False
```

如你所能猜到的，即使是最基本的 Web 应用程序也不会落入这个类别。因此，尽管我们有关闭 keep-alive 的能力，但你可能不应该这么做。

你更有可能想要更改的是超时时间。默认情况下，Sanic 将保持连接开启五秒钟。这可能看起来不是很长，但对于大多数用例来说应该足够长，而且不会造成浪费。然而，这仅仅是 Sanic 做的一个完全猜测。你更有可能了解并理解你应用程序的需求，你应该可以自由地调整这个数字以满足你的需求。如何？再次，通过一个简单的配置值：

```py
app.config.KEEP_ALIVE_TIMEOUT = 60
```

为了给你一些背景信息，这里是从 Sanic 用户指南中摘录的一段内容，它提供了一些关于其他系统如何操作的见解：

*“Apache httpd 服务器默认 keepalive 超时 = 5 秒*

*Nginx 服务器默认 keepalive 超时 = 75 秒*

*Nginx 性能调整指南使用 keepalive = 15 秒*

*IE (5-9) 客户端硬 keepalive 限制 = 60 秒*

*Firefox 客户端硬 keepalive 限制 = 115 秒*

*Opera 11 客户端硬 keepalive 限制 = 120 秒*

*Chrome 13+ 客户端 keepalive 限制 > 300+ 秒"*

[`sanicframework.org/en/guide/deployment/configuration.html#keep-alive-timeout`](https://sanicframework.org/en/guide/deployment/configuration.html#keep-alive-timeout)

你如何知道是否应该增加超时时间？如果你正在构建一个单页应用程序，其中你的 API 旨在为 JS 前端提供动力，那么浏览器可能会发出很多请求。这通常是这些前端应用程序的工作方式。如果你预期用户会点击按钮、浏览一些内容，然后再次点击，这尤其正确。我首先想到的是一种网络门户类型的应用程序，其中单个用户可能需要在分钟内进行数十次调用，但这些调用可能被一些浏览时间间隔所分隔。在这种情况下，将超时时间增加到反映预期使用可能是有意义的。

这并不意味着你应该过度增加它。首先，如我们上面所看到的，浏览器通常有一个它们将保持连接打开的最大限制。其次，连接长度过长可能会造成浪费，并损害你的内存性能。你追求的是一个平衡。没有一种完美的答案，所以你可能需要通过实验来找出什么有效。

### 按连接缓存数据

如果你正在考虑如何利用这些工具来满足你的应用程序需求，你可能已经注意到了可以创建的潜在效率。回到本章的开头，有一个表格列出了 Sanic 中你可以使用的所有上下文（`ctx`）对象。其中之一是针对连接的特定对象。

这意味着你不仅能够创建有状态的请求，还可以将状态添加到单个连接中。我们的简单示例将是一个计数器。

1.  首先在建立连接时创建一个计数器。我们将为此使用一个信号：

    ```py
    from itertools import count
    @app.signal("http.lifecycle.begin")
    async def setup_counter(conn_info):
        conn_info.ctx._counter = count()
    ```

1.  接下来，我们将使用中间件在每次请求时增加计数器：

    ```py
    @app.on_request
    async def increment(request):
        request.conn_info.ctx.count = next(request.conn_info.ctx._counter)
    ```

1.  然后，我们将将其输出到我们的请求体中，以便我们可以看到它看起来像什么：

    ```py
    @app.get("/")
    async def handler(request):
        return json({"request_number": request.conn_info.ctx.count})
    ```

1.  现在，我们将使用 curl 发出多个请求。为此，我们只需多次给出 URL：

    ```py
    $ curl localhost:7777 localhost:7777
    {"request_number":0}
    {"request_number":1}
    ```

这当然是一个简单的例子，我们可以很容易地从 Sanic 中获取这些信息：

```py
@app.get("/")
async def handler(request):
    return json(
        {
            "request_number": request.conn_info.ctx.count,
            "sanic_count": request.protocol.state["requests_count"],
        },
    )
```

如果你有某些可能获取成本高昂的数据，但又希望它对所有请求都可用，这将非常有用。回到我们之前的角色扮演模型，这就像你的服务器在视频聊天开始时获取了一些详细信息。现在，每当客户端提出问题，服务器已经将详细信息放在缓存中。

> **重要提示**
> 
> 这确实有一个警告。如果你的应用程序通过代理暴露，可能是连接池。也就是说，代理可能会从不同的客户端接收请求并将它们捆绑在一个连接中。想象一下，如果你的视频聊天会话不是在某个人的私人住宅中，而是在一个大大学宿舍的大厅里。任何人都可以走到单个视频聊天会话前提问。你可能无法保证每次都是同一个人。因此，在你将任何敏感细节暴露在这个对象上之前，你必须知道它是安全的。你的最佳实践可能就是将敏感细节保留在`request.ctx`上。

### 像专业人士一样处理异常

在一个理想的世界里，我们的应用程序永远不会失败，用户永远不会提交错误信息。所有端点将始终返回`200 OK`响应。这当然是纯粹的幻想，没有任何网络应用程序如果不解决失败的可能性就不能完整。在现实生活中，我们的代码会有错误，会有未处理的边缘情况，用户会发送错误数据并滥用应用程序。简而言之：我们的应用程序会失败。因此，我们必须时刻考虑这一点。

当然，Sanic 为我们提供了一些默认的处理方式。它包括几种不同的异常处理器样式（HTML、JSON 和文本），并且可以在生产环境和开发环境中使用。它当然是中立的，因此对于相当规模的应用程序来说可能是不够充分的。我们将在后面的*回退错误处理*部分更多地讨论回退错误处理。正如我们刚刚学到的，在应用程序中处理异常对于网络应用程序的质量（以及最终的安全性）至关重要。现在我们将学习如何在 Sanic 中做到这一点。

## 实施适当的异常处理

在我们探讨如何使用 Sanic 处理异常之前，重要的是要考虑，如果未能妥善处理这个问题，可能会变成一个安全问题。显而易见的方式可能是不小心泄露敏感信息。这被称为*泄露*。这种情况发生在异常被抛出（可能是用户的错误或故意为之）并且你的应用程序报告回显露出有关应用程序构建方式或存储数据的细节时。

在一个现实世界的最坏情况下，我曾经在一个网络应用程序中有一个被遗忘的旧端点，它已经不再工作。没有人再使用它，我简单地忘记了它的存在，甚至不知道它仍然处于活跃状态。问题是这个端点没有适当的异常处理，错误直接在发生时报告。这意味着即使是“*无法使用用户名 ABC 和密码 EFG 连接到数据库 XYZ*”这样的消息也会直接流向访问端点的人。哎呀。

因此，尽管我们直到第七章才讨论一般的安全问题，但它确实扩展到了当前对异常处理的探索。这里有两个主要问题：提供带有回溯或其他实现细节的异常消息，以及错误地使用 400 系列响应。

### 不良异常消息

在开发过程中，尽可能多地了解您的请求信息非常有帮助。这就是为什么在您的响应中包含异常消息和回溯是可取的。当您以调试模式构建应用程序时，您将获得所有这些详细信息。但请确保在生产环境中将其关闭！就像我希望我的应用程序始终只提供`200 OK`响应一样，我希望我永远不会遇到一个意外泄露调试信息的网站。这种情况在野外确实存在，所以请小心不要犯这个错误。

更常见的是在响应时没有正确考虑错误的内容。在编写将发送给最终用户的消息时，请记住，您不希望无意中泄露实现细节。

### 错误使用状态

与不良异常密切相关的是泄露关于您应用程序信息的异常。想象一下，如果您的银行网站有一个端点是：`/accounts/id/123456789`。他们做了尽职调查并正确保护了这个端点，以确保只有您才能访问它。这没问题。但是，如果有人无法访问它会发生什么？当我尝试访问您的银行账户时会发生什么？显然我会得到 401 未授权，因为这不是我的账户。然而，一旦你这样做，银行现在就承认 123456789 是一个合法的账户号码。因此，我**强烈**建议您使用下面的图表并将其牢记在心。

| **状态** | **描述** | **Sanic 异常** | **何时使用** |
| --- | --- | --- | --- |
| 400 | 错误请求 | `InvalidUsage` | 当任何用户提交意外形式的数据或他们以其他方式做了您的应用程序不打算处理的事情时 |
| 401 | 未授权 | `Unauthorized` | 当未知用户尚未认证时。换句话说，您不知道用户是谁。 |
| 403 | 禁止访问 | `Forbidden` | 当已知用户没有权限在**已知**资源上执行某些操作时 |
| 404 | 未找到 | `NotFound` | 当任何用户尝试访问隐藏资源时 |
|  |  |  |  |

表 6.6 - Sanic 对常见 400 系列 HTTP 响应的异常

最大的失败可能是人们无意中通过 401 或 403 暴露了隐藏资源的存在。您的银行本应发送给我一个 404，并引导我到一个“页面未找到”的响应。这并不是说您应该总是优先考虑 404。但从安全角度来看，考虑谁可以访问信息，以及他们应该或不应该知道什么，对您是有利的。然后，您可以决定哪种错误响应是合适的。

### 通过抛出异常来响应

在 Sanic 中处理异常最方便的事情之一是它相对容易上手。记住，我们只是在编写一个 Python 脚本，你应该像对待其他任何东西一样对待它。当事情出错时，你应该怎么做？抛出异常！这里有一个例子。

1.  创建一个简单的处理器，我们在这里将忽略返回值，因为我们不需要它来证明我们的观点。发挥你的想象力，看看 `...` 之外可能是什么。

    ```py
    @app.post("/cart)
    async def add_to_cart(request):
        if "name" not in request.json:
            raise InvalidUsage("You forgot to send a product name")
        ...
    ```

1.  接下来，我们将向端点提交一些 JSON 数据，但不包括名称属性。请确保使用 `-i` 选项，这样我们就可以检查响应头。

    ```py
    $ curl localhost:7777/cart -X POST -d '{}' -i
    HTTP/1.1 400 Bad Request
    content-length: 83
    connection: keep-alive
    content-type: text/plain; charset=utf-8
    400 — Bad Request
    =================
    You forgot to send a product name
    ```

注意我们收到了一个 400 响应，但实际上并没有从处理器返回响应。这是因为如果你从 `sanic.exceptions` 中抛出任何异常，它们*可以*用来返回适当的状态码。此外，你会发现该模块中的许多异常（如 `InvalidUsage`）都有一个默认的 `status_code`。这就是为什么当你抛出 `InvalidUsage` 时，Sanic 会以 400 响应。当然，你可以通过传递不同的值来覆盖状态码。让我们看看这将如何工作：

1.  设置此端点并更改 `status_code` 为 400 以外的值：

    ```py
    @app.post("/coffee")
    async def teapot(request):
        raise InvalidUsage("Hmm...", status_code=418)
    ```

1.  现在，让我们访问它：

    ```py
    $ curl localhost:777/coffee -X POST -i      
    HTTP/1.1 418 I'm a teapot
    content-length: 58
    connection: keep-alive
    content-type: text/plain; charset=utf-8
    418 — I'm a teapot
    ==================
    Hmm...
    ```

如您所见，我们向异常传递了 418 状态码。Sanic 接受了这个代码，并将其正确转换为适当的 HTTP 响应：`418 我是一把茶壶`。是的，这是一个真实的 HTTP 响应。你不信？在 RFC 7168 § 2.3.3 中查找。[`datatracker.ietf.org/doc/html/rfc7168#section-2.3.3`](https://datatracker.ietf.org/doc/html/rfc7168#section-2.3.3)

这里是所有内置异常及其相关响应码的参考：

| **异常** | **状态** |
| --- | --- |
| `HeaderNotFound` | 400 错误请求 |
| `InvalidUsage` | 400 错误请求 |
| `Unauthorized` | 401 未授权 |
| `Forbidden` | 403 禁止 |
| `FileNotFound` | 404 文件未找到 |
| `NotFound` | 404 文件未找到 |
| `MethodNotSupported` | 405 方法不允许 |
| `RequestTimeout` | 408 请求超时 |
| `PayloadTooLarge` | 413 请求实体过大 |
| `ContentRangeError` | 416 请求范围不满足 |
| `InvalidRangeType` | 416 请求范围不满足 |
| `HeaderExpectationFailed` | 417 期望失败 |
| `ServerError` | 500 内部服务器错误 |
| `URLBuildError` | 500 内部服务器错误 |
| `ServiceUnavailable` | 503 服务不可用 |

表 6.4 带内置 HTTP 响应的 Sanic 异常

因此，使用这些状态码是一个非常好的实践。一个明显的例子可能是当你正在数据库中查找不存在的东西时：

```py
@app.get("/product/<product_id:uuid>")
async def product_details(request, product_id):
    try:
        product = await Product.query(product_id=product_id)
    except DoesNotExist:
        raise NotFound("No product found")
```

使用 Sanic 异常可能是获得适当响应给用户的解决方案之一。

我们当然可以更进一步。我们可以创建自己的自定义异常，这些异常从 Sanic 异常中继承，以利用相同的特性。

1.  创建一个继承现有 Sanic 异常之一的异常：

    ```py
    from sanic.exceptions import InvalidUsage
    class MinQuantityError(InvalidUsage):
        ...
    ```

1.  在适当的时候提升它：

    ```py
    @app.post("/cart")
    async def add_to_cart(request):
        if request.json["qty"] < 5:
            raise MinQuantityError(
                "Sorry, you must purchase at least 5 of this item"
            )
    ```

1.  当我们有一个不良请求（少于 5 项）时，可以看到错误：

    ```py
    $ curl localhost:777/cart -X POST -d '{"qty": 1}' -i
    HTTP/1.1 400 Bad Request
    content-length: 98
    connection: keep-alive
    content-type: text/plain; charset=utf-8
    400 — Bad Request
    =================
    Sorry, you must purchase at least 5 of this item
    ```

鼓励使用和重用继承自 `SanicException` 的异常。这不仅是一种良好的实践，因为它提供了一种一致且干净的机制来组织你的代码，而且它使得提供适当的 HTTP 响应变得容易。

到目前为止，在本书中，当我们客户端遇到异常（如最后一个例子所示）时，我们已经收到了一个很好的错误文本表示。在下一节中，我们将了解其他类型的异常输出，以及我们如何控制它。

### 回退处理

坦白说：格式化异常是平凡的。毫无疑问，使用我们迄今为止学到的技能，我们可以构建我们自己的异常处理器集合。我们知道如何使用模板、捕获异常和返回带有错误状态的 HTTP 响应。但是创建这些需要时间和大量的样板代码。

这就是为什么 Sanic 提供了三种（3）不同的异常处理器：HTML、JSON 和纯文本。在本书的大部分例子中，我们只使用了纯文本处理器，因为这更适合在书中展示信息。让我们回到我们引发 `NotFound` 错误的例子，看看它使用三种不同类型的处理器可能是什么样子。

#### HTML

1.  设置我们的端点以引发异常：

    ```py
    @app.get("/product/<product_name:slug>")
    async def product_details(request, product_name):
        raise NotFound("No product found")
    ```

1.  告诉 Sanic 使用 HTML 格式化。我们将在第八章深入探讨配置。现在，我们只需在我们的 Sanic 实例之后设置该值：

    ```py
    app = Sanic(__name__)
    app.config.FALLBACK_ERROR_FORMAT = "html"
    ```

1.  打开一个网页浏览器并访问我们的端点。你应该会看到类似这样的内容：

![图 6.1 - 示例 404 页面，显示 Sanic 中默认的 404 未找到 HTML 页面看起来是什么样子](img/file7.png)

图 6.1 - 示例 404 页面，显示 Sanic 中默认的 404 未找到 HTML 页面看起来是什么样子

#### JSON

1.  使用之前的相同设置，但将回退格式更改为 `json`。

    ```py
    app.config.FALLBACK_ERROR_FORMAT = "html"
    ```

1.  这次我们将使用 curl 访问端点：

    ```py
    $ curl localhost:7777/product/missing-product
    {
      "description": "Not Found",
      "status": 404,
      "message": "No product found"
    }
    ```

与之前示例中看到的格式良好的 HTML 不同，我们的异常已被格式化为 JSON。如果你的端点将——例如——被一个 JavaScript 浏览器应用程序使用，这更为合适。

#### 文本

1.  再次使用相同的设置，我们将回退格式更改为 `text`。

    ```py
    app.config.FALLBACK_ERROR_FORMAT = "text"
    ```

1.  我们将再次使用 curl 来访问端点：

    ```py
    $ curl localhost:7777/product/missing-product
    404 — Not Found
    ===============
    No product found
    ```

如你所见，有三个方便的格式化器适用于我们的异常，可能在不同情况下都适用。

#### 自动

前三个示例使用了 `FALLBACK_ERROR_FORMAT` 来展示有三种内置的错误格式。还有一个设置 FALLBACK_ERROR_FORMAT 的第四个选项：`auto`。它看起来是这样的。

```py
app.config.FALLBACK_ERROR_FORMAT = "auto"
```

当格式设置为`auto`时，Sanic 将查看路由处理程序和传入的请求，以确定最合适的处理程序。例如，如果一个路由处理程序始终使用`text()`响应对象，那么 Sanic 将假设你希望异常也以`text`格式格式化。同样适用于`html()`和`json()`响应。

当处于`auto`模式时，Sanic 甚至会比这更进一步。它会分析传入的请求，查看头部信息，以确保它认为正确的内容与客户端所说的想要接收的内容相匹配。

#### 每个路由的手动覆盖

我们最后一个选择是在路由定义中的单个路由上设置错误格式。这允许我们具体指定，并在需要时偏离回退选项。

1.  考虑我们设置回退为`html`的例子。

    ```py
    app.config.FALLBACK_ERROR_FORMAT = "html"
    ```

1.  现在，让我们将本节开头的路由定义更改为以下具有特定定义的`error_format`：

    ```py
    @app.get("/product/<product_name:slug>", error_format="text")
    async def product_details(request, product_name):
        raise NotFound("No product found")
    ```

1.  如您可能已经猜到的，我们不会看到一个格式化的 HTML 页面，而会看到之前提到的纯文本。

    ```py
    $ curl localhost:7777/product/missing-product
    404 — Not Found
    ===============
    No product found
    ```

### 捕获异常

虽然 Sanic 方便地为我们处理了很多异常，但不用说，它无法预知应用中可能出现的每一个错误。因此，我们需要考虑如何处理来自 Sanic 之外的异常。或者，更确切地说，如何处理不是通过 Sanic 的异常手动抛出的异常，使用 Sanic 方便添加响应代码的异常。

回到我们的电子商务示例，让我们想象我们正在使用第三方供应商来处理我们的信用卡交易。他们方便地为我们提供了一个我们可以用来处理信用卡的模块。当出现问题的时候，他们的模块将抛出`CreditCardError`。我们现在的工作是确保我们的应用程序准备好处理这个错误。

然而，在我们这样做之前，让我们看看为什么这很重要。

1.  想象一下，这是我们终点：

    ```py
    @app.post("/cart/complete")
    async def complete_transaction(request):
        ...
        await submit_payment(...)
        ...
    ```

1.  现在，我们访问端点，如果出现错误，我们会得到以下响应：

    ```py
    $ curl localhost:7777/cart/complete -X POST
    500 — Internal Server Error
    ============================
    The server encountered an internal error and cannot complete your request.
    ```

这不是一条很有帮助的信息。然而，如果我们查看我们的日志，我们可能会看到以下内容：

```py
[ERROR] Exception occurred while handling uri: 'http://localhost:7777/cart/complete'
Traceback (most recent call last):
  File "handle_request", line 83, in handle_request
    """
  File "/path/to/server.py", line 19, in complete_transaction
    await submit_payment(...)
  File "/path/to/server.py", line 13, in submit_payment
    raise CreditCardError("Expiration date must be in format: MMYY")
CreditCardError: Expiration date must be in format: MMYY
[INFO][127.0.0.1:58334]: POST http://localhost:7777/cart/complete  500 144
```

这个错误看起来对我们的用户可能更有帮助。

当然，一个解决方案可能是捕获异常并返回我们想要的响应：

```py
@app.post("/cart/complete")
async def complete_transaction(request):
    ...
    try:
        await submit_payment(...)
    except CreditCardError as e:
        return text(str(e), status=400)
    ...
```

然而，这种模式并不理想。当我们需要在应用程序的各个位置捕获每个潜在的异常并将它们转换为响应时，这将需要大量的额外代码。这也会使我们的代码变成一个巨大的 try/except 块混乱，使阅读和维护变得更困难。简而言之，这会违反我们在本书早期确立的一些开发原则。

一个更好的解决方案是添加一个应用程序级别的异常处理器。这告诉 Sanic，每当这个异常冒泡时，它应该捕获它并以某种方式响应。它看起来非常像路由处理器：

```py
@app.exception(CreditCardError)
async def handle_credit_card_errors(request, exception):
    return text(str(exception), status=400)
```

Sanic 现已将此注册为异常处理器，并在 `CreditCardError` 被抛出时使用它。当然，这个处理器非常简单，但你可以想象它可以用作：额外的日志记录、提供请求上下文、凌晨 3 点向你的 DevOps 团队发送紧急警报通知等等。

> **提示**
> 
> 错误处理器不仅限于你的应用程序实例。就像其他常规路由处理器一样，它们可以注册在你的 Blueprint 实例上，以便为应用程序的特定子集定制错误处理。

异常处理是应用程序开发中极其重要的一个部分。它是业余应用程序和专业应用程序之间的一个直接区分因素。我们现在知道如何使用异常不仅为用户提供有用的消息，还可以提供适当的 HTTP 响应代码。我们现在转向另一个主题（后台处理），这可以帮助将你的应用程序提升到下一个层次。

## 后台处理

在大多数应用程序的开发过程中，开发者或用户开始注意到应用程序感觉有点慢。有些操作似乎需要很长时间，这损害了应用程序其他部分的可用性。这可能是因为计算成本高昂，也可能是因为网络操作需要连接到另一个系统。

让我们假设你处于这种场景。你已经构建了一个优秀的应用程序和一个端点，允许用户通过点击按钮生成 PDF 报告，显示各种复杂的数据和图表。问题是，为了检索所有数据并处理这些数字似乎需要二十（20）秒。对于 HTTP 请求来说，这是一段很长的时间！在花费时间尽可能提高报告生成器的性能之后，你最终得出结论，它已经尽可能快地运行。你能做什么？

将其推送到后台。

当我们说“后台处理”时，我们真正指的是一种允许当前请求完成而不需要最终完成所需工作的解决方案。在这个例子中，这意味着在报告生成实际完成之前，完成启动报告生成的请求。无论何时何地，我都建议将工作推送到后台。在本章的“等待事件”部分中，我们看到了在后台发送注册电子邮件的用例。确实，如前所述的信号的使用是一种后台处理形式。然而，这并不是 Sanic 提供的唯一工具。

### 向循环中添加任务

如您可能已经知道，`asyncio` 库的一个基石是任务。它们本质上是在循环上运行异步工作的处理单元。如果任务或任务循环的概念对您来说仍然陌生，那么在继续之前，在互联网上做一些研究可能是个好主意。

在典型场景中，您可以通过获取事件循环并调用`create_task`来生成任务，如下所示：

```py
import asyncio
async def something():
...
async def main():
loop = asyncio.get_running_loop()
loop.create_task(something())
```

这可能对您来说并不陌生，但它的作用是在当前任务之外启动`something`。

Sanic 提供了一个简单的接口来创建任务，如下所示：

```py
async def something():
...
app.add_task(something)
```

这可能是最简单的后台处理形式，并且是一个您应该习惯使用的模式。为什么使用这个而不是`create_task`？有三个原因：

+   这更容易，因为您不需要获取循环。

+   它可以在循环开始之前在全局范围内使用。

+   它可以调用或不调用，也可以带或不带应用程序实例作为参数。

为了说明灵活性，将上一个例子与以下例子进行对比：

```py
from sanic import Sanic
from my_app import something
app = Sanic(“MyAwesomeApp”)
app.add_task(something(app))
```

> **提示**
> 
> 如果任务没有像第一个例子那样被调用，Sanic 将检查该函数是否期望`app`实例作为参数，并将其注入。

Asyncio 任务非常有帮助，但有时您需要一个更健壮的解决方案。让我们看看我们的其他选项。

### 与外部服务集成

如果您的应用程序有工作要做，但由于某种原因超出了您的 API 范围，您可能需要转向现成的解决方案。这通常是以另一种在别处运行的服务的形式出现的。现在，您的 Web API 的工作就是向该服务提供工作。

在 Python 世界中，这类工作的经典框架是 Celery。当然，这并不是唯一的选择，但鉴于这本书不是关于决定使用什么的，我们将以 Celery 为例，因为它被广泛使用且为人所知。简而言之，Celery 是一个平台，它的工作进程从队列中读取消息。某些客户端负责将工作推送到队列，当工作进程接收到消息时，它将执行该工作。

为了 Celery 能够运行，它在某台机器上运行一个进程。它有一组已知的操作可以执行（也称为“任务”）。要启动一个任务，外部客户端需要通过代理连接到它，并发送运行任务的指令。一个基本的实现可能看起来像这样。

1.  我们设置了一个客户端，使其能够与进程通信。一个常见的放置位置是在`application.ctx`上，以便在应用程序的任何地方使用。

    ```py
    from celery import Celery
    @app.before_server_start 
    def setup_celery(app, _):
        app.ctx.celery = Celery(...)
    ```

1.  要使用它，我们只需从路由处理程序中调用客户端，将一些工作推送到 Celery。

    ```py
    @app.post("/start_task") 
    async def start_task(request): 
        task = request.app.ctx.celery.send_task(
            "execute_slow_stuff", 
            kwargs=request.json
        )
        return text(f"Started task with {task.id=}", status=202)
    ```

这里要指出的重要一点是，我们正在使用`202 已接受`状态来告诉 whoever made the request，该操作已被接受处理。没有保证它会完成，或者将会完成。

在检查 Celery 之后，你可能认为它对你的需求来说有点过度。但是，`app.add_task`似乎还不够。接下来，我们将看看你如何开发自己的进程内队列系统。

### 设计进程内任务队列

有时，对于你的需求来说，显而易见的黄金比例解决方案是构建一个完全局限于 Sanic 的东西。如果你只需要担心一个服务而不是多个服务，这将更容易管理。你可能仍然希望保留“工作者”和“任务队列”的概念，而不需要像 Celery 这样的服务实现所需的额外开销。所以，让我们构建一些东西，希望这能成为你在应用程序中构建更令人惊叹的东西的起点。在我们开始之前，你可以在 GitHub 仓库中查看最终的代码产品：___。

在我们继续前进之前，让我们将名称从“任务队列”更改为“作业队列”。我们不希望因为像 asyncio 任务这样的例子而混淆自己。在本节剩余部分，单词“任务”将指代 asyncio 任务。

首先，我们将开发一套针对我们的作业队列的需求。

+   应该有一个或多个能够执行作业（在请求/响应周期之外）的“工作者”。

+   它们应该按照先入先出的顺序执行作业。

+   作业的完成顺序并不重要（例如，作业 A 在作业 B 之前开始，但哪个先完成并不重要）。

+   我们应该能够检查作业的状态。

我们实现这一点的策略是构建一个框架，其中我们有一个“工作者”，它本身就是一个后台任务。它的任务是查找通用队列中的作业并执行它们。这个概念与 Celery 非常相似，但我们将在我们的 Sanic 应用程序中使用 asyncio 任务来处理它。我们将通过查看源代码来完成这一点，但不会全部展示。与本次讨论无关的实现细节将在此省略。有关完整详情，请参阅 GitHub 仓库中的源代码：____。

1.  首先，让我们设置一个非常简单的应用程序，它只有一个蓝图。

    ```py
    from sanic import Sanic
    from job.blueprint import bp
    app = Sanic(__name__)
    app.config.NUM_TASK_WORKERS = 3
    app.blueprint(bp)
    ```

1.  那个蓝图将是我们将附加一些监听器和端点的地方。

    ```py
    from sanic import Blueprint
    from job.startup import (
    setup_task_executor,
    setup_job_fetch,
    register_operations,
    )
    from job.view import JobListView, JobDetailView
    bp = Blueprint("JobQueue", url_prefix="/job")
    bp.after_server_start(setup_job_fetch)
    bp.after_server_start(setup_task_executor)
    bp.after_server_start(register_operations)
    bp.add_route(JobListView.as_view(), "")
    bp.add_route(JobDetailView.as_view(), "/<uid:uuid>")
    ```

    如你所见，我们需要运行三个监听器：`setup_job fetch`、`setup_task_executor`和`register_operations`。我们还有两个视图：一个是列表视图，另一个是详情视图。让我们逐一查看这些项目，看看它们是什么。

1.  由于我们想要存储任务的状态，我们需要某种类型的数据存储。为了使事情尽可能简单，我创建了一个基于文件的数据库，名为`FileBackend`。

    ```py
    async def setup_job_fetch(app, _):
    app.ctx.jobs = FileBackend("./db")
    ```

1.  这个作业管理系统将根据我们的作业队列的功能驱动，该队列将使用`asyncio.Queue`实现。因此，我们接下来需要设置我们的队列和工作者。

    ```py
    async def setup_task_executor(app, _):
    app.ctx.queue = asyncio.Queue(maxsize=64)
    for x in range(app.config.NUM_TASK_WORKERS):
    name = f"Worker-{x}"
    print(f"Starting up executor: {name}")
    app.add_task(worker(name, app.ctx.queue, app.ctx.jobs))
    ```

    在创建我们的队列之后，我们创建一个或多个后台任务。正如你所见，我们只是使用 Sanic 的`add_task`方法从`worker`函数创建一个任务。我们将在稍后看到这个函数。

1.  我们需要的最后一个监听器将设置一个对象，该对象将用于存储我们所有的潜在操作。

    ```py
    async def register_operations(app, _):
    app.ctx.registry = OperationRegistry(Hello)
    ```

    为了提醒你，`Operation`将是我们希望在后台运行的东西。在这个例子中，我们有一个操作：`Hello`。在查看操作之前，让我们看看两个视图。

1.  列表视图将有一个 POST 调用，该调用负责将新的作业推入队列。你也可以想象这将是列出所有现有作业（当然，分页）的端点的一个合适位置。首先，它需要从请求中获取一些数据：

    ```py
    class JobListView(HTTPMethodView):
    async def post(self, request):
    operation = request.json.get("operation")
    kwargs = request.json.get("kwargs", {})
    if not operation:
    raise InvalidUsage("Missing operation")
    ```

    在这里，我们执行一些非常简单的数据验证。在现实世界的场景中，你可能想要做更多的事情来确保请求的 JSON 符合你的预期。

1.  在验证数据后，我们可以将有关作业的信息推送到队列。

    ```py
    uid = uuid.uuid4()
    await request.app.ctx.queue.put(
    {
    "operation": operation,
    "uid": uid,
    "kwargs": kwargs,
    }
    )
    return json({"uid": str(uid)}, status=202)
    ```

    我们创建了一个 UUID。这个唯一标识符将用于在数据库中存储作业，并在以后检索有关它的信息。此外，重要的是指出，我们使用`202 Accepted`响应，因为它是最合适的形式。

1.  详细视图非常简单。使用唯一标识符，我们只需在数据库中查找并返回它。

    ```py
    class JobDetailView(HTTPMethodView):
    async def get(self, request, uid: uuid.UUID):
    data = await request.app.ctx.jobs.fetch(uid)
    return json(data)
    ```

1.  回到我们的`Hello`操作，我们现在来构建它：

    ```py
    import asyncio
    from .base import Operation
    class Hello(Operation):
    async def run(self, name="world"):
    message = f"Hello, {name}"
    print(message)
    await asyncio.sleep(10)
    print("Done.")
    return message
    ```

    如你所见，它是一个简单的对象，有一个`run`方法。当运行`Job`时，该方法将由工作者调用。

1.  工作者实际上只是一个异步函数。它的任务将运行一个永无止境的循环。在这个循环中，它将等待队列中有作业。

    ```py
    async def worker(name, queue, backend):
    while True:
    job = await queue.get()
    if not job:
    break
    size = queue.qsize()
    print(f"[{name}] Running {job}. {size} in queue.")
    ```

1.  一旦它有了如何运行作业的信息，它需要创建一个作业实例，并执行它。

    ```py
    job_instance = await Job.create(job, backend)
    async with job_instance as operation:
    await job_instance.execute(operation)
    ```

关于这个解决方案的几点补充：它最大的缺点是没有恢复机制。如果你的应用程序崩溃或重启，将无法继续处理已经开始的作业。在真正的任务管理过程中，这通常是一个重要的功能。因此，在 GitHub 仓库中，除了构建此解决方案所使用的源代码外，你还会找到“子进程”任务队列的源代码。我不会带你一步步构建它，因为它在很大程度上是一个类似的练习，有很多相同的代码。然而，它与这个解决方案在两个重要方面有所不同：它确实有恢复和重启未完成作业的能力，并且它不是在异步任务中运行，而是利用 Sanic 的进程管理监听器通过多进程技术创建子进程。在你继续学习和通过这本书的工作时，请花些时间查看那里的源代码。

## 摘要

在我看来，作为应用开发者，你可以迈出的最大一步之一是制定策略来抽象问题的解决方案，并在多个地方重用该解决方案。如果你听说过 DRY（不要重复自己）原则，这就是我的意思。应用程序很少是“完整”的。我们开发它们、维护它们并修改它们。如果我们有太多的重复代码，或者代码与单一用例过于紧密耦合，那么修改它或将其适应不同用例就会变得更加困难。学会概括我们的解决方案可以减轻这个问题。

在 Sanic 中，这意味着将逻辑从路由处理程序中提取出来。最好我们能够最小化单个处理程序中的代码量，并将这些代码放置在其他位置，以便其他端点可以重用。你是否注意到了在 *设计进程内任务队列* 的最终示例中，路由处理程序并没有超过十几行？虽然确切的长度并不重要，但保持这些代码简洁且简短，并将你的逻辑放在其他地方是有帮助的。

也许本章最大的收获之一应该是通常没有一种唯一的方法来完成某事。我们经常可以使用这些方法的混合来达到我们的目标。那么，应用开发者的任务就是查看工具箱，并决定在特定情况下哪种工具最适合。

因此，作为一个 Sanic 开发者，你应该学习如何制定策略来在路由处理程序之外响应网络请求。在本章中，我们学习了使用中间件、内置和自定义信号、连接管理、异常处理和后台处理等工具来帮助你完成这项任务。再次强调，将这些视为你的工具箱中的核心工具。需要拧紧螺丝吗？拿出你的中间件。需要在木头上钻孔吗？是时候从架子上拿走钻头了。你对 Sanic 中这些基本构建块（如中间件）越熟悉，你对如何构建专业级应用程序的理解就会越深入。

现在是你的任务，通过玩转这些工具并在成为更好的开发者道路上内化它们。

我们已经触及了与安全相关问题的表面。在下一章中，我们将更深入地探讨如何保护我们的 Sanic 应用程序。
