## 第十五章：15

Web 服务的功能方法

我们将暂时离开探索性数据分析的主题，转向 Web 服务器和 Web 服务。Web 服务器在某种程度上是一系列函数的级联。我们可以将许多功能设计模式应用于呈现 Web 内容的问题。我们的目标是探讨我们可以如何接近表示状态转移（REST）。我们希望使用功能设计模式构建 RESTful Web 服务。

我们不需要再发明另一个 Python Web 框架。我们也不想从可用的框架中选择。Python 中有许多 Web 框架，每个框架都有其独特的一组特性和优势。

本章的目的是提出一些可以应用于大多数可用框架的原则。这将使我们能够利用功能设计模式来呈现 Web 内容。

当我们查看极大型或复杂的数据集时，我们可能需要一个支持子集或搜索的 Web 服务。我们也可能需要一个可以以各种格式下载子集的网站。在这种情况下，我们可能需要使用功能设计来创建支持这些更复杂要求的 RESTful Web 服务。

交互式 Web 应用程序通常依赖于有状态的会话来使网站更容易使用。用户的会话信息会通过 HTML 表单提供的数据、从数据库中检索的数据或从先前交互的缓存中恢复的数据进行更新。由于有状态的数据必须作为每个事务的一部分进行检索，它更像是一个输入参数或结果值。这可能导致即使在存在 cookies 和数据库更新的情况下，也会出现功能式编程。

在本章中，我们将探讨几个主题：

+   HTTP 请求和响应模型的一般概念。

+   Python 应用程序使用的 Web 服务器网关接口（WSGI）标准。

+   利用 WSGI，在可能将 Web 服务定义为函数的地方。这与 HTTP 无状态服务器的理念相符。

+   我们还将探讨授权客户端应用程序使用 Web 服务的方法。

### 15.1 HTTP 请求-响应模型

HTTP 协议几乎是无状态的：用户代理（或浏览器）发起请求，服务器提供响应。对于不涉及 cookies 的服务，客户端应用程序可以采用功能视图的协议。我们可以使用`http.client`或`urllib.request`模块构建客户端。HTTP 用户代理可以像以下函数一样实现：

```py
import urllib.request 

def urllib_get(url: str) -> tuple[int, str]: 
    with urllib.request.urlopen(url) as response: 
        body_bytes = response.read() 
        encoding = response.headers.get_content_charset("utf-8") 
        return response.status, body_bytes.decode(encoding)
```

类似于 wget 或 curl 这样的程序会使用作为命令行参数提供的 URL 进行此类处理。浏览器会在用户点击和指向时执行此操作；URL 通常来自用户的操作，通常是点击链接文本或图像。

注意，一个页面的编码通常在响应的两个不同位置进行描述。HTTP 头通常会命名正在使用的编码。在这个例子中，当头信息不完整时，会提供默认的 `"utf-8"` 编码。此外，HTML 内容也可以提供编码信息。具体来说，一个 `<meta charset="utf-8">` 标签可以声明一个编码。理想情况下，它与头中注明的编码相同。或者，一个 `<meta http-equiv...>` 标签可以提供编码。

虽然 HTTP 处理是无状态的，但用户体验（UX）设计的实际考虑导致了一些需要保持状态的具体实现细节。为了使人类用户感到舒适，服务器必须知道他们做了什么，并保留事务状态。这是通过使客户端软件（浏览器或移动应用程序）跟踪 cookie 来实现的。为了使 cookie 起作用，响应头提供了 cookie 数据，后续请求必须将保存的 cookie 返回给服务器。

HTTP 响应将包括一个状态码。在某些情况下，这个状态码将需要用户代理采取额外的操作。300-399 范围内的许多状态码表示请求的资源已移动。然后应用程序或浏览器需要从 `Location` 头中保存详细信息并请求新的 URL。401 状态码表示需要认证；用户代理必须使用包含访问服务器凭证的 `Authorization` 头进行另一个请求。`urllib` 库实现处理这种有状态客户端处理。`http.client` 库类似，但它不会自动遵循 3xx 重定向状态码。

看到协议的另一边，一个静态内容服务器可以是无状态的。我们可以使用 `http.server` 库来做这件事，如下所示：

```py
from http.server import HTTPServer, SimpleHTTPRequestHandler 
from typing import NoReturn 

def server_demo() -> NoReturn: 
    httpd = HTTPServer( 
          (’localhost’, 8080), 
          SimpleHTTPRequestHandler 
    ) 
    print(f"Serving on http://localhost:8080...") 
    while True: 
        httpd.handle_request() 
    httpd.shutdown()
```

我们创建了一个 `server` 对象，并将其分配给 `httpd` 变量。我们提供了地址 `localhost` 和端口号 `8080`。作为接受请求的一部分，HTTP 协议将分配另一个端口；这用于创建处理程序类的实例。在一个端口上监听但在其他端口上执行工作允许服务器并发处理多个请求。

在这个例子中，我们提供了 `SimpleHTTPRequestHandler` 类作为每个请求的实例化类。这个类必须实现一个最小接口，该接口将发送头信息，然后将响应体的内容发送给客户端。这个特定的类将从本地目录中提供文件。如果我们想自定义它，我们可以创建一个子类，该子类实现了如 `do_GET()` 和 `do_POST()` 等方法来改变行为。

`HTTPServer` 类有一个 `serve_forever()` 方法，可以避免编写显式的 `while` 语句。我们在这里展示了 `while` 语句，以明确指出，如果需要停止服务器，通常必须使用中断信号来崩溃服务器。

本例使用端口号 8080，这个端口号不需要提升权限。Web 服务器通常使用端口号 80 和 443，这些端口号需要提升权限。通常，最好使用像 NGINX 或 Apache httpd 这样的服务器来管理特权端口。

#### 15.1.1 通过 cookies 注入状态

cookies 的添加改变了客户端和服务器之间的整体关系，使其成为有状态的。有趣的是，这并不涉及对 HTTP 的任何更改。状态信息是通过请求和回复的头信息进行通信的。服务器将在响应头中向用户代理发送 cookies。用户代理将保存 cookies 并在请求头中回复它们。

用户代理或浏览器需要保留作为响应一部分提供的 cookie 值缓存，并在后续请求中包含适当的 cookies。Web 服务器将在请求头中查找 cookies，并在响应头中提供更新的 cookies。这种效果是使 Web 服务器无状态；状态变化仅发生在客户端。因为服务器将 cookies 视为请求中的附加参数，并在响应中提供额外的详细信息，这塑造了我们对于响应请求的功能的看法。

cookies 可以包含任何适合 4,096 字节的内容。它们通常被加密，以避免将 Web 服务器细节暴露给客户端计算机上运行的其它应用程序。传输大的 cookies 可能会很慢，应该避免。最佳实践是将会话信息保存在数据库中，并在 cookie 中只提供数据库键。这使得会话持久化，并允许会话处理由任何可用的 Web 服务器处理，从而实现服务器之间的负载均衡。

会话的概念是 Web 应用程序软件的一个特性，而不是 HTTP。会话通常通过 cookie 实现，以保留会话信息。当发起初始请求时，没有可用的 cookie，将创建一个新的会话 cookie。每个后续请求都将包括 cookie 的值。登录用户将在他们的会话 cookie 中包含额外的详细信息。会话可以持续到服务器愿意接受 cookie 的时间；cookie 可以是永久有效的，或者几分钟后过期。

RESTful 风格的 Web 服务不依赖于会话或 cookies。每个 REST 请求都是独特的。在许多情况下，每个请求都会提供一个`Authorization`头，以提供认证和授权的凭据。这通常意味着必须有一个单独面向客户端的应用程序来创建令人愉悦的用户体验，这通常涉及到会话。常见的架构是一个前端应用程序，可能是一个移动应用程序或基于浏览器的网站，用于提供对支持 RESTful Web 服务的视图。

我们将在本章中关注 RESTful Web 服务。RESTful 方法非常适合无状态的函数式设计模式。

无会话 REST 过程的后果之一是每个 REST 请求都是单独认证的。这通常意味着 REST 服务也必须使用安全套接字层（SSL）协议。`HTTPS`方案是用于从客户端到服务器安全传输凭证所必需的。

#### 15.1.2 考虑具有功能设计的服务器

HTTP 背后的一个核心思想是服务器的响应是请求的函数。从概念上讲，网络服务应该有一个顶层实现，可以概括如下：

```py
response = httpd(request)
```

虽然这是 HTTP 的本质，但它缺少许多重要细节。首先，HTTP 请求不是一个简单的、单一的数据结构。它包含一些必需的部分和一些可选的部分。一个请求可能包含头部信息、一个方法（例如，`GET`、`POST`、`PUT`、`PATCH`等）、一个 URL，并且可能有附件。URL 包含几个可选部分，包括路径、查询字符串和片段标识符。附件可能包括来自 HTML 表单的输入或上传的文件，或者两者都有。

其次，响应同样有三个部分。它有一个状态码、头部信息和响应体。我们简单的`httpd()`函数模型没有涵盖这些额外的细节。

我们需要扩展这种简单观点，以便更准确地分解网络处理为有用的函数。

#### 15.1.3 深入探讨功能视图

HTTP 响应和请求都有与主体分开的头部。请求还可以包含一些附加的表单数据或其他上传。因此，我们可以更有用地将 Web 服务器视为这样：

```py
headers, content = httpd( 
    headers, request, [attachments, either forms or uploads] 
)
```

请求头部可能包括 cookie 值，这可以被视为添加了更多的参数。此外，Web 服务器通常依赖于其运行的操作系统环境。这些操作系统环境数据可以被视为作为请求一部分提供的更多参数。

多用途互联网邮件扩展（MIME）类型定义了网络服务可能返回的内容类型。MIME 描述了一个大但相对定义良好的内容范围。这可以包括纯文本、HTML、JSON、XML 或任何网站可能提供的大量非文本媒体。

HTTP 请求处理的一些常见特性是我们希望重用的。这种可重用元素的想法导致了从简单到复杂的各种网络服务框架的创建。功能设计允许我们重用函数的方式表明，功能方法有助于构建网络服务。

我们将通过检查我们如何创建服务响应的各种元素管道来研究网络服务的功能设计。我们将通过嵌套请求处理的函数来实现这一点，这样内部元素就可以免受外部元素提供的通用开销的影响。这也允许外部元素充当过滤器：无效的请求可以产生错误响应，从而使内部函数能够专注于应用程序处理。

#### 15.1.4 嵌套服务

我们可以将网络请求处理看作是一系列分层上下文。基础可能包括会话管理：检查请求以确定这是现有会话中的另一个请求还是新会话。在这个基础上，另一层可以提供用于表单处理的令牌，这些令牌可以检测跨站请求伪造（CSRF）。在这些之上，可能还有一层处理会话内的用户身份验证。

对之前解释的功能的概念性视图可能如下所示：

```py
response = content( 
    authentication( 
        csrf( 
            session(headers, request, forms) 
        ) 
    ) 
)
```

这里的想法是每个函数都可以建立在先前函数的结果之上。每个函数要么丰富请求，要么拒绝它，因为它无效。例如，`session()` 函数可以使用头部信息来确定这是一个现有会话还是一个新会话。`csrf()` 函数将检查表单输入以确保使用了正确的令牌。CSRF 处理需要有效的会话。`authentication()` 函数可以为缺少有效凭证的会话返回错误响应；当存在有效凭证时，它可以丰富请求，添加用户信息。

`content()` 函数无需担心会话、伪造和非认证用户。它可以专注于解析路径，以确定应提供哪种类型的内容。在更复杂的应用中，`content()` 函数可能包括从路径元素到确定适当内容的函数的相当复杂的映射。

这种嵌套的函数视图存在一个深刻的问题。函数栈被定义为按照特定顺序使用。`csrf()` 函数必须首先执行，以便为 `authentication()` 函数提供有用的信息。然而，我们可以想象一个高安全场景，其中在检查 CSRF 令牌之前必须先进行身份验证。我们不希望为每种可能的网络架构定义独特的函数。

虽然每个上下文都必须有一个独特的焦点，但有一个单一的、统一的请求和响应处理视图会更有帮助。这允许独立构建各个部分。一个有用的网站将是多个不同函数的组合。

使用标准化接口，我们可以组合函数以实现所需功能。这将符合函数式编程的目标，即编写简洁且表达力强的程序来提供网络内容。WSGI 标准提供了一种统一的方式来构建复杂服务，作为部分的组合。

### 15.2 WSGI 标准

Web 服务器网关接口（WSGI）定义了创建对网络请求响应的标准接口。这是大多数基于 Python 的网络服务器的通用框架。以下链接提供了大量信息：[`wsgi.readthedocs.org/en/latest/`](http://wsgi.readthedocs.org/en/latest/)。

关于 WSGI 的一些重要背景信息可以在以下链接中找到：[`www.python.org/dev/peps/pep-0333/`](https://www.python.org/dev/peps/pep-0333/)。

Python 库的 `wsgiref` 包包含 WSGI 的参考实现。每个 WSGI 应用程序都具有相同的接口，如下所示：

```py
def some_app(environ, start_response): 
    # compute the status, headers, and content of the response 
    start_response(status, headers) 
    return content
```

`environ` 参数是一个字典，它包含请求的所有参数，以单一、统一的结构。头部、请求方法、路径以及表单或文件上传的任何附件都将包含在环境字典中。除了这些之外，还提供了 OS 级别的上下文，以及一些属于 WSGI 请求处理的项。

`start_response` 参数是一个必须使用的函数，用于发送响应的状态和头部信息。负责构建响应的 WSGI 服务器部分将使用提供的 `start_response()` 函数，并将响应文档作为返回值构建。

从 WSGI 应用程序返回的响应是一个字符串或类似字符串的文件包装器的序列，这些序列将被返回给用户代理。如果使用 HTML 模板工具，则序列可能只有一个项目。在某些情况下，例如使用 Jinja2 模板构建 HTML 内容，模板可以延迟作为文本块的序列进行渲染。这允许服务器在向用户代理下载的同时混合模板填充。

`wsgiref` 包没有一组完整的类型定义。这通常不是问题。例如，在 `werkzeug` 包中，`werkzeug.wsgi` 模块包含有用的类型定义。由于 `werkzeug` 包通常与 Flask 一起安装，因此对于我们的目的来说非常方便。

`werkzeug.wsgi` 模块包含一个具有多个有用类型提示的存根文件。这些提示不是工作应用程序的一部分；它们仅由 mypy 工具使用。我们可以研究以下 `werkzeug.wsgi` 的 WSGI 应用程序类型提示：

```py
from sys import _OptExcInfo 
from typing import Any, Callable, Dict, Iterable, Protocol 

class StartResponse(Protocol): 
    def __call__( 
        self, status: str, headers: list[tuple[str, str]], exc_info: "_OptExcInfo" | None = ... 
    ) -> Callable[[bytes], Any]: ... 

WSGIEnvironment = Dict[str, Any] 
WSGIApplication = Callable[[WSGIEnvironment, StartResponse], Iterable[bytes]]
```

`WSGIEnvironment` 类型提示定义了一个没有对值有有用边界的字典。很难列举出 WSGI 标准定义的所有可能类型的值。与其使用详尽的复杂定义，似乎更好的方法是使用 `Any`。

`StartResponse` 类型提示是提供给 WSGI 应用的 `start_response()` 函数的签名。这被定义为 `Protocol` 以显示存在一个可选的第三个参数，用于异常信息。

整个 WSGI 应用程序 `WSGIApplication` 需要环境和 `start_response()` 函数。结果是字节的可迭代集合。

这些提示背后的想法是允许我们定义一个应用程序如下：

```py
from typing import TYPE_CHECKING 

if TYPE_CHECKING: 
    from _typeshed.wsgi import ( 
        WSGIApplication, WSGIEnvironment, StartResponse 
    ) 

def static_text_app( 
    environ: "WSGIEnvironment", 
    start_response: "StartResponse" 
) -> Iterable[bytes]: 
    ...
```

我们包括了一个条件 `import` 来提供类型提示，仅在运行 mypy 工具时使用。在不在使用 mypy 工具的情况下，类型提示作为字符串提供。这种额外的说明可以帮助解释一个复杂函数集合的设计，这些函数响应 Web 请求。

每个 WSGI 应用程序都需要设计成函数的集合。这个集合可以看作是嵌套函数或变换链。链中的每个应用程序要么返回一个错误，要么将请求传递给另一个应用程序，该应用程序将确定最终结果。

通常，URL 路径用于确定将使用哪些许多替代应用程序中的哪一个。这会导致一个 WSGI 应用程序的树，这些应用程序可能共享公共组件。

这里有一个非常简单的路由应用程序，它接受 URL 路径的第一个元素，并使用它来定位提供内容的另一个 WSGI 应用程序：

```py
from wsgiref.simple_server import demo_app 

SCRIPT_MAP: dict[str, "WSGIApplication"] = { 
    "demo": demo_app, 
    "static": static_text_app, 
    "index.html": welcome_app, 
    "": welcome_app, 
} 

def routing( 
        environ: "WSGIEnvironment", 
        start_response: "StartResponse" 
) -> Iterable[bytes]: 
    top_level = wsgiref.util.shift_path_info(environ) 
    if top_level: 
        app = SCRIPT_MAP.get(top_level, welcome_app) 
    else: 
        app = welcome_app 
    content = app(environ, start_response) 
    return content
```

此应用程序将使用`wsgiref.util.shift_path_info()`函数调整环境。更改是对请求路径的头部/尾部分割，可在`environ[’PATH_INFO’]`字典中找到。路径的头部，直到第一个`"/"`，将被分配给环境中的`SCRIPT_NAME`项；`PATH_INFO`项将被更新以包含路径的尾部。返回值也将是路径的头部，与`environ[’SCRIPT_NAME’]`相同的值。在没有路径可解析的情况下，返回值是`None`，并且不进行环境更新。

`routing()`函数使用路径上的第一个项目在`SCRIPT_MAP`字典中定位应用程序。如果请求的路径不符合映射，我们使用`welcome_app`作为默认值。这似乎比 HTTP `404 NOT FOUND`错误要好一些。

这个 WSGI 应用程序是一个函数，它从多个其他 WSGI 函数中选择。请注意，路由函数不返回一个函数；它将修改后的环境提供给选定的 WSGI 应用程序。这是从函数到函数传递工作的典型设计模式。

从这里，我们可以看到框架如何泛化路径匹配过程，使用正则表达式。我们可以想象配置`routing()`函数，使用一系列正则表达式和 WSGI 应用程序，而不是从字符串到 WSGI 应用程序的映射。增强的`routing()`函数将评估每个正则表达式以寻找匹配项。在匹配的情况下，可以使用任何`match.groups()`函数在调用请求的应用程序之前更新环境。

#### 15.2.1 在 WSGI 处理期间引发异常

WSGI 应用程序的一个核心特征是链中的每个阶段都负责过滤请求。其理念是在处理过程中尽早拒绝错误的请求。当构建一系列独立的 WSGI 应用程序时，每个阶段有两个基本选择：

+   评估`start_response()`函数以启动带有错误状态的回复

+   或者将带有扩展环境的请求传递给下一个阶段

考虑一个提供小型文本文件的 WSGI 应用程序。文件可能不存在，或者请求可能指向文件目录。我们可以定义一个提供静态内容的 WSGI 应用程序如下：

```py
def headers(content: bytes) -> list[tuple[str, str]]: 
    return [ 
        ("Content-Type", ’text/plain;charset="utf-8"’), 
        ("Content-Length", str(len(content))), 
    ] 

def static_text_app( 
    environ: "WSGIEnvironment", 
    start_response: "StartResponse" 
) -> Iterable[bytes]: 
    log = environ[’wsgi.errors’] 
    try: 
        static_path = Path.cwd() / environ[’PATH_INFO’][1:] 
        with static_path.open() as static_file: 
            print(f"{static_path=}", file=log) 
            content = static_file.read().encode("utf-8") 
            start_response(’200 OK’, headers(content)) 
            return [content] 
    except IsADirectoryError as exc: 
        return index_app(environ, start_response) 
    except FileNotFoundError as exc: 
        print(f"{static_path=} {exc=}", file=log) 
        message = f"Not Found {environ[’PATH_INFO’]}".encode("utf-8") 
        start_response(’404 NOT FOUND’, headers(message)) 
        return [message]
```

此应用程序从当前工作目录和请求 URL 中提供的路径元素创建一个`Path`对象。路径信息是 WSGI 环境的一部分，在具有`’PATH_INFO’`键的项中。由于路径的解析方式，它将有一个前导的”/”，我们通过使用`environ[’PATH_INFO’][1:]`来丢弃它。

此应用程序尝试以文本文件的形式打开请求的路径。存在两个常见问题，这两个问题都作为异常处理：

+   如果文件是一个目录，我们将请求路由到不同的 WSGI 应用程序，即`index_app`，以展示目录内容

+   如果文件根本找不到，我们将返回 HTTP `404` NOT FOUND 响应

此 WSGI 应用程序引发的任何其他异常都不会被捕获。调用此应用程序的应用程序应该设计有某种通用的错误响应能力。如果应用程序没有处理异常，将使用通用的 WSGI 失败响应。

我们的处理涉及操作的严格顺序。我们必须读取整个文件，以便我们可以创建适当的 HTTP Content-Length 头。

此小型应用程序展示了 WSGI 的响应或转发请求到另一个形成响应的应用程序的想法。这种立即响应或转发设计模式使得构建多阶段管道成为可能。每个阶段要么拒绝请求，要么完全处理它，要么将其传递给其他应用程序。

这些管道通常被称为中间件，因为它们位于基础服务器（如 NGINX）和最终 Web 应用程序或 RESTful API 之间。想法是使用中间件为每个请求执行一系列常见的过滤器或映射。

#### 15.2.2 实用型 Web 应用程序

WSGI 标准的意图不是定义一个完整的 Web 框架；意图是定义一组最小标准，允许灵活的 Web 相关处理互操作性。这个最小标准与函数式编程概念很好地匹配。

Web 应用程序框架专注于开发者的需求。它应该提供许多简化以提供 Web 服务。基础接口必须与 WSGI 兼容，以便可以在各种环境中使用。然而，开发者的观点将与最小的 WSGI 定义有所不同。

Web 服务器，如 Apache httpd 或 NGINX，有适配器，可以从 Web 服务器提供 WSGI 兼容的接口到 Python 应用程序。有关 WSGI 实现的更多信息，请访问[`wiki.python.org/moin/WSGIImplementations`](https://wiki.python.org/moin/WSGIImplementations)。

将我们的应用程序嵌入到更大的服务器中，可以使我们实现关注点的整洁分离。我们可以使用 Apache httpd 或 NGINX 来提供静态内容，例如`.css`、`.js`和图像文件。然而，对于 HTML 页面，像 NGINX 这样的服务器可以使用`uwsgi`模块将请求传递给一组 Python 进程。这使 Python 专注于处理网页内容的有趣且复杂的 HTML 部分。

下载静态内容需要很少的定制。通常没有特定于应用程序的处理。这最好在一个单独的服务中处理，该服务可以优化以执行此固定任务。

动态内容的处理（通常是网页的 HTML 内容）是 Python 相关有趣工作的发生地。这项工作可以分离到优化运行这种更复杂的应用特定计算的服务器上。

将静态内容与动态内容分离以提供优化的下载意味着我们必须创建一个单独的媒体服务器，或者定义我们的网站具有两组路径。对于较小的网站，单独的`/media`路径效果很好。对于较大的网站，则需要不同的媒体服务器。

WSGI 定义的一个重要后果是`environ`字典通常会更新额外的配置参数。通过这种方式，一些 WSGI 应用程序可以作为网关，从 cookie、头部、配置文件或数据库中提取信息来丰富环境。

### 15.3 将网络服务定义为函数

我们将研究一个 RESTful 网络服务，它可以切割和分割数据源，并提供 JSON、XML 或 CSV 文件的下载。

直接使用 WSGI 进行此类应用程序不是最优的，因为我们需要为传统网站处理的全部细节创建大量的“样板”处理。更有效的方法是使用更复杂的网络服务器，如 Flask、Django、Bottle 或这里列出的任何框架：[`wiki.python.org/moin/WebFrameworks`](https://wiki.python.org/moin/WebFrameworks)。这些服务器处理传统情况更完整，使我们作为开发者能够专注于页面或网站的独特功能。

我们将使用一个包含四个数据对序列的简单数据集：Anscombe 四重奏。我们在第三章、函数、迭代器和生成器中探讨了读取和解析这些数据的方法。这是一个小的数据集，但它可以用来展示 RESTful 网络服务的原则。

我们将把我们的应用程序分为两层：一个网络层，它将提供可见的 RESTful 网络服务，和一个数据服务层，它将管理底层数据。我们将首先查看网络层，因为这为数据服务层必须运行的环境提供了上下文。

请求必须包含以下两块信息：

+   所需的数据序列。想法是通过过滤和提取所需子集来切割可用的信息池。

+   用户需要的输出格式。这包括常见的序列化格式，如 HTML、CSV、JSON 和 XML。

系列选择通常是通过请求路径完成的。我们可以请求`/anscombe/I`或`/anscombe/II`来选择四重奏中的特定系列。路径设计很重要，这似乎是识别数据的正确方式。

以下两个基本思想有助于定义路径：

+   一个 URL 定义了一个资源

+   没有充分的理由让 URL 发生变化

在这种情况下，`I`或`II`的数据集选择器不依赖于发布日期或某些组织批准状态，或其他外部因素。这种设计似乎创建出永恒且绝对的 URL。

另一方面，输出格式不是 URL 的一部分。它仅仅是一个序列化格式，而不是数据本身。一个选择是在 HTTP `Accept`头中命名格式。在某些情况下，为了使浏览器使用起来更方便，可以使用查询字符串来指定输出格式。一种方法是通过查询来指定序列化格式。我们可以在路径末尾使用`?form=json`、`?format=json`，甚至`?output_serialization=json`来指定输出序列化格式应为 JSON。HTTP `Accept`头是首选的，但仅使用浏览器进行实验可能比较困难。

我们可以使用一个浏览器友好的 URL，其形式如下：

```py
http://localhost:8080/anscombe/III?form=csv
```

这将请求以 CSV 格式下载第三系列。

OpenAPI 规范提供了一种定义 URL 家族和预期结果的方法。这个规范是有帮助的，因为它作为网络服务器预期行为的清晰、正式合同。OpenAPI 规范最有帮助的是有一个具体的路径、参数和响应列表。一个好的规范将包括示例，有助于编写服务器的验收测试套件。

通常，OpenAPI 规范由网络服务器提供，以帮助客户端正确使用可用的服务。建议使用像`"/openapi.yml"`或`"/openapi.json"`这样的 URL 来提供关于网络应用程序所需的信息。

#### 15.3.1 Flask 应用程序处理

我们将使用 Flask 框架，因为它提供了一个易于扩展的 Web 服务过程。它支持基于函数的设计，将请求路径映射到构建响应的视图函数。该框架还利用了装饰器，与函数式编程概念相匹配。

为了将所有配置和 URL 路由绑定在一起，使用一个总的`Flask`实例作为容器。我们的应用程序将是`Flask`类的一个实例。作为一种简化，每个视图函数都是单独定义的，并通过将 URL 映射到函数的路由表绑定到`Flask`实例。这个路由表是通过装饰器构建的。

应用程序的核心是这个视图函数集合。通常，每个视图函数需要做三件事：

1.  验证请求。

1.  执行请求的状态更改或数据访问。

1.  准备响应。

理想情况下，视图函数不做任何其他事情。

这里是初始的 `Flask` 对象，它将包含路由及其函数：

```py
from flask import Flask 

app = Flask(__name__)
```

我们已创建 `Flask` 实例并将其分配给 `app` 变量。作为一个方便的默认值，我们使用了模块的名称，`__name__`，作为应用程序的名称。这通常足够。对于复杂的应用程序，可能更好的是提供一个不特定于 Python 模块或包名称的名称。

大多数应用程序都需要提供配置参数。在这种情况下，源数据是一个可能更改的可配置值。

对于较大的应用程序，通常有必要定位整个配置文件。对于这个小型应用程序，我们将提供配置值作为字面量：

```py
from pathlib import Path 

app.config[’FILE_PATH’] = Path.cwd() / "Anscombe.txt"
```

大多数视图函数应该是相对较小的、专注于其他应用层功能的函数。对于这个应用程序，网络表示依赖于数据服务层来获取和格式化数据。这导致以下三个步骤的函数：

1.  验证各种输入。这包括验证路径、任何查询参数、表单输入数据、上传的文件、头部值，甚至是 cookie 值。

1.  如果该方法涉及状态更改，如 `POST`、`PUT`、`PATCH` 或 `DELETE`，则执行状态更改操作。这些操作通常会返回一个指向将显示更改结果的路径的“重定向”响应。如果该方法涉及 `GET` 请求，则收集所需数据。

1.  准备响应。

第 2 步的重要之处在于所有数据操作都与 RESTful 网络应用程序分离。网络表示建立在数据访问和操作的基础之上。网络应用程序被设计为一个视图或对底层结构的展示。

我们将查看网络应用程序的两个 URL 路径。第一个路径将提供 Anscombe 集合中可用系列索引。`view` 函数可以定义为以下内容：

```py
from flask import request, abort, make_response, Response 

@app.route("/anscombe/") 
def index_view() -> Response: 
    # 1\. Validate 
    response_format = format() 
    # 2\. Get data 
    data = get_series_map(app.config[’FILE_PATH’]) 
    index_listofdicts = [{"Series": k} for k in data.keys()] 
    # 3\. Prepare Response 
    try: 
        content_bytes = serialize(response_format, index_listofdicts, document_tag="Index", row_tag="Series") 
        response = make_response(content_bytes, 200, {"Content-Type": response_format}) 
        return response 
    except KeyError: 
        abort(404, f"Unknown {response_format=}")
```

此函数具有 Flask 的 `@app.route` 装饰器。这表明哪些 URL 应由此 `view` 函数处理。这里有许多选项和替代方案可用。当请求与可用路由之一匹配时，将评估 `view` 函数。

`format()` 函数的定义将在稍后展示。它通过查找两个地方来定位用户期望的格式：URL 中的查询字符串，即 `?` 后面，以及 `Accept` 头部。如果查询字符串值无效，将创建一个 404 响应。

`get_series_map()` 函数是数据服务层的一个基本功能。这将定位 Anscombe 系列数据，并将 `Series` 名称映射到系列数据。

索引信息以列表-of-dict 结构的形式存在。这种结构可以不经过太多复杂地转换为 JSON、CSV 和 HTML。创建 XML 则要困难一些。困难之处在于 Python 列表和字典对象没有特定的类名，这使得提供 XML 标签变得有些尴尬。

数据准备分为两部分进行。首先，索引信息以所需格式序列化。其次，使用字节、HTTP 状态码 200 和`Content-Type`头部的特定值构建一个 Flask `Response`对象。

`abort()`函数停止进程并返回带有给定代码和原因信息的错误响应。对于 RESTful Web 服务，添加一个将结果转换为 JSON 的小型辅助函数很有帮助。在数据验证和准备期间使用`abort()`函数使得在请求的第一个问题时结束处理变得容易。

`format()`函数定义如下：

```py
def format() -> str: 
    if arg := request.args.get(’form’): 
        try: 
            return { 
                ’xml’: ’application/xml’, 
                ’html’: ’text/html’, 
                ’json’: ’application/json’, 
                ’csv’: ’text/csv’, 
            }[arg] 
        except KeyError: 
            abort(404, "Unknown ?form=") 
    else: 
        return request.accept_mimetypes.best or "text/html"
```

此函数从`request`对象的两个属性中查找输入：

+   `args`将包含 URL 中“？”之后出现的参数值

+   `accept_mimetypes`将包含从`Accept`头部解析出的值，允许应用程序定位满足客户端期望的响应

`request`对象是带有正在进行的 Web 请求详细信息的线程局部存储。它被用作全局变量，使得一些函数看起来有些笨拙。像`request`这样的全局变量往往会掩盖此函数的实际参数。使用显式参数还需要提供底层类型信息，这不过是视觉上的杂乱。

定义提供系列数据的`series_view()`函数如下：

```py
@app.route("/anscombe/<series_id>") 
def series_view(series_id: str, form: str | None = None) -> Response: 
    # 1\. Validate 
    response_format = format() 
    # 2\. Get data (and validate some more) 
    data = get_series_map(app.config[’FILE_PATH’]) 
    try: 
        dataset = anscombe_filter(series_id, data)._as_listofdicts() 
    except KeyError: 
        abort(404, "Unknown Series") 
    # 3\. Prepare Response 
    try: 
        content_bytes = serialize(response_format, dataset, document_tag="Series", row_tag="Pair") 
        response = make_response( 
            content_bytes, 200, {"Content-Type": response_format} 
        ) 
        return response 
    except KeyError: 
        abort(404, f"Unknown {response_format=}")
```

此函数结构与之前的`index_view()`函数类似。请求得到验证，数据获取，并准备响应。与之前的函数一样，工作被委托给另外两个数据访问函数：`get_series_map()`和`anscombe_filter()`。这些函数与 Web 应用程序分开，可能是命令行应用程序的一部分。

这两个函数都依赖于底层的数据访问层。我们将在下一节中查看这些函数。

#### 15.3.2 数据访问层

`get_series_map()`函数与第三章使用生成器函数清理原始数据部分中显示的示例类似，函数、迭代器和生成器。在本节中，我们将包括一些重要的更改。我们将从以下两个`NamedTuple`定义开始：

```py
from Chapter03.ch03_ex4 import ( 
    series, head_split_fixed, row_iter) 
from collections.abc import Callable, Iterable 
from typing import NamedTuple, Any, cast 

class Pair(NamedTuple): 
    x: float 
    y: float 

    @classmethod 
    def create(cls: type["Pair"], source: Iterable[str]) -> "Pair": 
        return Pair(*map(float, source)) 

class Series(NamedTuple): 
    series: str 
    data: list[Pair] 

    @classmethod 
    def create(cls: type["Series"], name: str, source: Iterable[tuple[str, str]]) -> "Series": 
        return Series(name, list(map(Pair.create, source))) 

    def _as_listofdicts(self) -> list[dict[str, Any]]: 
        return [p._asdict() for p in self.data]
```

我们定义了一个名为`Pair`的命名元组，并提供了一个`@classmethod`来构建`Pair`的实例。此定义将自动提供一个`_asdict()`方法，该方法返回一个形式为`dict[str, Any]`的字典，包含属性名称和值。这对于序列化很有帮助。

同样，我们定义了一个名为 `Series` 的命名元组。`create()` 方法可以从值列表的可迭代源构建一个元组。自动提供的 `_asdict()` 方法对于序列化可能很有帮助。然而，对于这个应用程序，我们将使用 `_as_listofdicts` 方法来创建可以序列化的字典列表。

从系列名称到 `Series` 对象的映射函数具有以下定义：

```py
from pathlib import Path 

def get_series_map(source_path: Path) -> dict[str, Series]: 
    with source_path.open() as source: 
        raw_data = list(head_split_fixed(row_iter(source))) 
        series_iter = ( 
            Series.create(id_str, series(id_num, raw_data)) 
            for id_num, id_str in enumerate( 
                [’I’, ’II’, ’III’, ’IV’]) 
        ) 
        mapping = { 
            series.series: series 
            for series in series_iter 
        } 
    return mapping
```

`get_series_map()` 函数打开本地数据文件，并将 `row_iter()` 函数应用于文件的每一行。这会将行解析为单独的项目行。使用 `head_split_fixed()` 函数从文件中移除标题。结果是元组列表结构，被分配给变量 `raw_data`。

从 `raw_data` 结构中，使用 `Series.create()` 方法将文件中的值序列转换为由单个 `Pair` 实例组成的 `Series` 对象。最后一步是使用字典推导式收集单个 `Series` 实例到一个从系列名称到 `Series` 对象的单个映射中。

由于 `get_series_map()` 函数的输出是一个映射，我们可以像以下示例那样通过名称选择特定的系列：

```py
>>> source = Path.cwd() / "Anscombe.txt" 
>>> get_series_map(source)[’I’] 
Series(series=’I’, data=[Pair(x=10.0, y=8.04), Pair(x=8.0, y=6.95), ...])
```

给定一个键，例如，`‘I’`，该系列是一个包含 `Pair` 对象的列表，这些对象具有系列中每个项目的 `x`、`y` 值。

##### 应用过滤器

在这个应用程序中，我们使用一个非常简单的过滤器。整个过滤器过程体现在以下函数中：

```py
def anscombe_filter( 
    set_id: str, raw_data_map: dict[str, Series] 
) -> Series: 
    return raw_data_map[set_id]
```

我们将这个简单的表达式转换为函数有三个原因：

+   函数表示法与其他 Flask 应用程序的各个部分略微更一致，并且比下标表达式更灵活

+   我们可以轻松扩展过滤功能以执行更多操作

+   我们可以为这个函数包含单独的单元测试

虽然简单的 lambda 函数可以工作，但测试起来可能不太方便。

对于错误处理，我们实际上什么都没做。我们专注于有时被称为“快乐路径”的理想事件序列。在这个函数中出现的任何问题都会抛出异常。WSGI 包装函数应该捕获所有异常，并返回适当的状态消息和错误响应内容。

例如，`set_id` 方法可能以某种方式出错。而不是过分关注它可能出错的所有方式，我们将允许 Python 抛出异常。实际上，这个函数遵循 Grace Hopper 海军上将的建议，即寻求宽恕比请求许可更好。这个建议在代码中体现为避免请求许可：没有试图验证参数有效性的预备 `if` 语句。只有宽恕处理：将抛出异常，并通过评估 Flask 的 `abort()` 函数来处理。

##### 序列化结果

序列化是将 Python 数据转换为字节流的过程，适合传输。每种格式最好通过一个简单的函数来描述，该函数仅序列化该格式。然后，顶层通用序列化器可以从一系列特定序列化器中选择。

序列化器的一般类型提示如下：

```py
from collections.abc import Callable 
from typing import Any, TypeAlias 

Serializer: TypeAlias = Callable[[list[dict[str, Any]]], bytes]
```

这个定义避免了具体的`Series`定义。它使用了一个更一般的`list[dict[str, Any]]`类型提示。这可以应用于`Series`的数据以及其他类似序列标签的项目。

从 MIME 类型到序列化器函数的映射将导致以下映射对象：

```py
SERIALIZERS: dict[str, Serializer] = {
'application/xml': serialize_xml,
'text/html': serialize_html,
'application/json': serialize_json,
'text/csv': serialize_csv,
}

```

这个变量将在引用的四个函数定义之后定义。我们在这里提供它作为上下文，展示序列化设计的发展方向。

顶层`serialize()`函数可以定义如下：

```py
def serialize( 
    format: str | None, 
    data: list[dict[str, Any]], 
    **kwargs: str 
) -> bytes: 
    """Relies on global SERIALIZERS, set separately""" 
    if format is None: 
        format = "text/html" 
    function = SERIALIZERS.get( 
        format.lower(), 
        serialize_html 
    ) 
    return function(data, **kwargs)
```

总体上的`serialize()`函数在`SERIALIZERS`字典中定位一个特定的序列化器。这个特定的函数符合`Serializer`类型提示。该函数将`Series`对象转换为字节，可以下载到 Web 客户端应用程序。

`serialize()`函数不执行任何数据转换。它将 MIME 类型字符串映射到一个执行转换的函数。

我们将查看一些单独的序列化器。在 Python 处理中创建字符串相对常见。然后我们可以将这些字符串编码为字节。为了避免重复编码操作，我们将定义一个装饰器来组合序列化和字节编码。以下是我们可以使用的装饰器：

```py
from collections.abc import Callable 
from typing import TypeVar, ParamSpec 
from functools import wraps 

T = TypeVar("T") 
P = ParamSpec("P") 

def to_bytes( 
    function: Callable[P, str] 
) -> Callable[P, bytes]: 
    @wraps(function) 
    def decorated(*args: P.args, **kwargs: P.kwargs) -> bytes: 
        text = function(*args, **kwargs) 
        return text.encode("utf-8") 
    return decorated
```

我们创建了一个名为`@to_bytes`的小型装饰器。这个装饰器将评估给定的函数，然后使用 UTF-8 编码结果以获取字节。请注意，装饰器将装饰函数的返回类型从`str`更改为`bytes`。我们使用了`ParamSpec`提示来收集装饰函数声明的参数。这确保了像 mypy 这样的工具可以将装饰函数的参数规范与基础函数相匹配。

我们将展示如何使用 JSON 和 CSV 序列化器来实现这一点。HTML 和 XML 序列化涉及更多的编程，但并没有显著的复杂性。

##### 使用 JSON 或 CSV 格式序列化数据

JSON 和 CSV 序列化器是相似的，因为两者都依赖于 Python 的库进行序列化。这些库本质上是命令式的，因此函数体是语句的序列。

下面是 JSON 序列化器的示例：

```py
import json 

@to_bytes 
def serialize_json(data: list[dict[str, Any]], **kwargs: str) -> str: 
    text = json.dumps(data, sort_keys=True) 
    return text
```

我们创建了一个字典列表结构，并使用`json.dumps()`函数创建了一个字符串表示形式。JSON 模块需要一个具体化的列表对象；我们不能提供一个惰性生成器函数。`sort_keys=True`参数值对于单元测试很有帮助，因为顺序被明确地说明了，并且可以用来匹配预期的结果。然而，它对于应用程序不是必需的，并且代表了一点点开销。

下面是 CSV 序列化器的示例：

```py
import csv 
import io 

@to_bytes 
def serialize_csv(data: list[dict[str, Any]], **kwargs: str) -> str: 
    buffer = io.StringIO() 
    wtr = csv.DictWriter(buffer, sorted(data[0].keys())) 
    wtr.writeheader() 
    wtr.writerows(data) 
    return buffer.getvalue()
```

`csv` 模块的读取器和写入器是命令式和函数式元素的混合。我们必须创建写入器，并且必须按照严格的顺序正确创建标题。此函数的客户端可以使用 `Pair` 命名元组的 `_fields` 属性来确定写入器的列标题。

writer 对象的 `writerows()` 方法将接受一个惰性生成器函数。此函数的客户端可以使用 `NamedTuple` 对象的 `_asdict()` 方法返回一个适合与 CSV 写入器一起使用的字典。

##### 使用 XML 和 HTML 序列化数据

将数据序列化为 XML 的目标是创建一个看起来像这样的文档：

```py
<?xml version="1.0" encoding="UTF-8"?> 
<Series> 
<Pair><x>2</x><y>3</y></Pair> 
<Pair><x>5</x><y>7</y></Pair> 
</Series>
```

此 XML 文档不包括对正式 XML 架构定义 (XSD) 的引用。然而，它被设计成与上面显示的命名元组定义并行。

生成此类文档的一种方法是通过创建模板并填写字段。这可以使用 Jinja 或 Mako 等包来完成。有许多复杂的模板工具可以创建 XML 或 HTML 页面。其中许多包括在模板中嵌入对对象序列（如字典列表）的迭代的能力，而无需在初始化序列化的函数中执行。访问 [`wiki.python.org/moin/Templating`](https://wiki.python.org/moin/Templating) 获取替代方案列表。

在这里，一个更复杂的序列化库可能会有所帮助。有许多可供选择。访问 [`wiki.python.org/moin/PythonXml`](https://wiki.python.org/moin/PythonXml) 获取替代方案列表。

现代 HTML 基于 XML。因此，可以通过将实际值填充到模板中来构建类似于 XML 文档的 HTML 文档。HTML 文档通常比 XML 文档有更多的开销。额外的复杂性源于在 HTML 中，文档被期望提供一个包含大量上下文信息的完整网页。

我们省略了创建 HTML 或 XML 的细节，将其留给读者作为练习。

### 15.4 跟踪使用

RESTful API 需要用于安全连接。这意味着服务器必须使用 SSL，并且连接将通过 HTTPS 协议进行。其想法是管理“前端”或客户端应用程序使用的 SSL 证书。在许多网络服务环境中，移动应用程序和基于 JavaScript 的交互式前端将拥有允许访问后端的证书。

除了 SSL 之外，另一个常见的做法是在每个事务中要求一个 API 密钥。API 密钥可以用来验证访问。它也可以用来授权特定功能。最重要的是，它对于跟踪实际使用至关重要。跟踪使用的一个后果是，如果在一个给定的时间段内过度使用 API 密钥，可能会限制请求。

商业模式的变体很多。例如，API 密钥的使用可能是一个可计费事件，并且将产生费用。对于其他业务，流量必须达到某个阈值，然后才需要支付。

重要的是要确保 API 使用的不可否认性。当执行交易以进行状态更改时，可以使用 API 密钥来识别发出请求的应用程序。这反过来意味着创建可以充当用户身份验证凭据的 API 密钥。密钥必须难以伪造，并且相对容易验证。

创建 API 密钥的一种方法是用加密随机数生成一个难以预测的密钥字符串。可以使用`secrets`模块生成唯一的 API 密钥值。以下是一个生成唯一密钥的示例，该密钥可以分配给客户端以跟踪活动：

```py
>>> import secrets 
>>> secrets.token_urlsafe(24) 
’NLHirCPVf-S7aSAiaAJo3JECYk9dSeyq’
```

在随机字节上使用 64 进制编码来创建一串字符。使用三的倍数作为长度将避免在 64 进制编码中出现任何尾随的`=`符号。我们使用了 URL 安全的 64 进制编码，这意味着结果字符串中不会包含`/`或`+`字符。这意味着密钥可以用作 URL 的一部分，或者可以在标题中提供。

使用更复杂的方法生成令牌不会导致更随机的数据。使用`secrets`模块确保很难伪造分配给其他用户的密钥。

`secrets`模块作为单元和集成测试的一部分使用时，因其难以使用而闻名。为了生成高质量、安全的数据，它避免了像`random`模块那样有显式种子。由于可重复的单元测试用例不能依赖于`secrets`模块的可重复结果，因此在测试时应使用模拟对象。这一结果的后果是创建一个便于测试的设计。

随着 API 密钥的生成，它们需要发送给创建应用程序的用户，并保存在 API 服务的一部分数据库中。

如果请求中包含数据库中的密钥，则关联的用户负责该请求。如果 API 请求不包含已知的密钥，则请求可以拒绝，并返回`401 UNAUTHORIZED`响应。

这个小型数据库可以是一个文本文件，服务器在加载时将其映射到授权权限的 API 密钥。该文件可以在启动时读取，并检查修改时间以确定服务器缓存的版本是否仍然是最新的。当有新的密钥可用时，文件将被更新，服务器将重新读取该文件。

有关 API 密钥的更多信息，请参阅[`swagger.io/docs/specification/2-0/authentication/api-keys/`](https://swagger.io/docs/specification/2-0/authentication/api-keys/)。

对有效 API 密钥的基本检查如此常见，以至于 Flask 提供了一个装饰器来识别此功能。使用`@app.before_app_request`标记一个将在每个视图函数之前调用的函数。这个函数可以在允许任何处理之前确定 API 密钥的有效性。

这个 API 密钥检查通常被绕过了一些路径。例如，如果服务将下载其 OpenAPI 规范，则路径应在不考虑是否存在`API-Key`头的情况下处理。这通常意味着一个特殊情况检查，以查看`request.path`是否为`openapi.json`或其他规范常见名称之一。

同样，服务器可能需要根据 CORS 头的存在来响应请求。有关更多信息，请参阅[`www.w3.org/TR/cors/#http-cors-protocol`](https://www.w3.org/TR/cors/#http-cors-protocol)。这可能会通过添加另一组异常使`before_app_request()`函数变得更加复杂。

好消息是，只有两个例外需要在每个请求中都包含`API-Key`头。一个是处理 OpenAPI 规范，另一个是 CORS 预请求。这不太可能改变，几个`if`语句就足够了。

### 15.5 摘要

在本章中，我们探讨了如何将函数式设计应用于基于 REST 的 Web 服务的内容服务问题。我们探讨了 WSGI 标准如何导致整体应用在一定程度上具有函数性。我们还探讨了如何通过从请求中提取元素以供我们的应用程序函数使用，将更函数化的设计嵌入到 WSGI 上下文中。

对于简单的服务，问题通常分解为三个不同的操作：获取数据、搜索或过滤，然后序列化结果。我们通过三个函数来解决这个问题：`raw_data()`、`anscombe_filter()`和`serialize()`。我们将这些函数包装在一个简单的 WSGI 兼容应用程序中，以将 Web 服务与提取和过滤数据的实际处理分离。

我们还探讨了 Web 服务函数如何专注于“快乐路径”并假设所有输入都是有效的。如果输入无效，普通的 Python 异常处理将引发异常。WSGI 包装函数将捕获错误并返回适当的状态码和错误内容。

我们还没有研究与上传数据或从表单中接受数据以更新持久数据存储相关的更复杂问题。这些问题并不比获取数据和序列化结果更复杂。

对于简单的查询和数据共享，一个小型网络服务应用程序可能会有所帮助。我们可以应用函数式设计模式，并确保网站代码简洁且易于理解。对于更复杂的网络应用程序，我们应该考虑使用一个能够正确处理细节的框架。

在下一章中，我们将探讨一个更完整的函数式编程示例。这是一个案例研究，它将一些统计措施应用于样本数据，以确定数据是否可能是随机的，或者可能包含一些有趣的关系。

### 15.6 练习

本章的练习基于 Packt Publishing 在 GitHub 上提供的代码。请参阅[`github.com/PacktPublishing/Functional-Python-Programming-3rd-Edition`](https://github.com/PacktPublishing/Functional-Python-Programming-3rd-Edition)。

在某些情况下，读者可能会注意到 GitHub 上提供的代码包括一些练习的部分解决方案。这些作为提示，允许读者探索替代解决方案。

在许多情况下，练习将需要单元测试用例来确认它们确实解决了问题。这些通常与 GitHub 仓库中已提供的单元测试用例相同。读者应将书籍中的示例函数名称替换为自己的解决方案以确认其工作。

#### 15.6.1 WSGI 应用程序：welcome

在本章的 The WSGI standard 部分，描述了一个路由应用程序。它展示了三个应用程序路由，包括以`/demo`开头的路径和一个针对`/index.html`路径的特殊情况。

通过 WSGI 创建应用程序可能具有挑战性。构建一个函数`welcome_app()`，该函数显示一个包含演示应用程序和静态下载应用程序链接的 HTML 页面。

为此应用程序编写的单元测试应使用模拟的`StartResponse`函数和一个模拟的环境。

#### 15.6.2 WSGI 应用程序：demo

在本章的 The WSGI standard 部分，描述了一个路由应用程序。它展示了三个应用程序路由，包括以`/demo`开头的路径和一个针对`/index.html`路径的特殊情况。

构建一个函数`demo_app()`，以执行一些可能有用的活动。这里的意图是有一个路径可以响应 HTTP `POST`请求来完成一些工作，在日志文件中创建一个条目。结果必须是一个重定向（状态码 303，通常）到使用`static_text_app()`下载日志文件的 URL。这种行为被称为 Post/Redirect/Get，当导航回上一个页面时，可以提供良好的用户体验。有关此设计模式的更多详细信息，请参阅[`www.geeksforgeeks.org/post-redirect-get-prg-design-pattern/`](https://www.geeksforgeeks.org/post-redirect-get-prg-design-pattern/)。

下面是演示应用程序可能实现的有用工作的两个示例：

+   一个`GET`请求可以显示一个带有表单的 HTML 页面。表单上的提交按钮可以将`POST`请求发送到执行某种计算的函数。

+   一个`POST`请求可以执行`doctest.testfile()`来运行单元测试套件并收集结果日志。

#### 15.6.3 使用 XML 序列化数据

在本章的 Serializing data with XML and HTML 部分，我们描述了使用 Flask 构建的 RESTful API 的两个附加功能。

在那些示例中扩展响应，将结果数据序列化为 XML，除了 CSV 和 JSON。添加 XML 序列化的一个替代方案是下载并安装一个库，该库可以序列化 `Series` 和 `Pair` 对象。另一个选择是编写一个可以与 `list[dict[str, Any]]` 对象一起工作的函数。添加 XML 序列化格式还需要添加测试用例来确认响应具有预期的格式和内容。

#### 15.6.4 使用 HTML 序列化数据

在本章的使用 XML 和 HTML 序列化数据部分，我们描述了使用 Flask 构建的 RESTful API 的两个附加功能。

在那些示例中扩展响应，将结果数据序列化为 HTML，除了 CSV 和 JSON。HTML 序列化可能比 XML 序列化更复杂，因为数据在 HTML 展示中有很多开销。而不是 `Pair` 对象的表示，通常的做法是包含一个完整的 HTML 表格结构，它反映了 CSV 的行和列。添加 HTML 序列化格式还需要添加测试用例来确认响应具有预期的格式和内容。

### 加入我们的社区 Discord 空间

加入我们的 Python Discord 工作空间，讨论并了解更多关于这本书的信息：[`packt.link/dHrHU`](https://packt.link/dHrHU)

![图片](img/file1.png)
