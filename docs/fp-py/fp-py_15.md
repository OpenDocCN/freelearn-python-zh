# 第十五章：面向 Web 服务的功能性方法

我们将远离探索性数据分析，而是仔细研究 Web 服务器和 Web 服务。在某种程度上，这些都是一系列函数。我们可以将许多函数设计模式应用于呈现 Web 内容的问题上。我们的目标是探索我们可以使用**表述状态转移**（**REST**）的方式。我们希望使用函数设计模式构建 RESTful Web 服务。

我们不需要再发明另一个 Python Web 框架；有很多框架可供选择。我们将避免创建一个庞大的通用解决方案。

我们不想在可用的框架中进行选择。每个框架都有不同的特性和优势。

我们将提出一些可以应用于大多数可用框架的原则。我们应该能够利用功能设计模式来呈现 Web 内容。这将使我们能够构建具有功能设计优势的基于 Web 的应用程序。

例如，当我们查看极大的数据集或极复杂的数据集时，我们可能需要一个支持子集或搜索的 Web 服务。我们可能需要一个能够以各种格式下载子集的网站。在这种情况下，我们可能需要使用功能设计来创建支持这些更复杂要求的 RESTful Web 服务。

最复杂的 Web 应用程序通常具有使网站更易于使用的有状态会话。会话信息通过 HTML 表单提供的数据更新，或者从数据库中获取，或者从以前的交互的缓存中获取。虽然整体交互涉及状态更改，但应用程序编程可以在很大程度上是功能性的。一些应用程序函数在使用请求数据、缓存数据和数据库对象时可能是非严格的。

为了避免特定 Web 框架的细节，我们将专注于**Web 服务器网关接口**（**WSGI**）设计模式。这将使我们能够实现一个简单的 Web 服务器。以下链接提供了大量信息：

[`wsgi.readthedocs.org/en/latest/`](http://wsgi.readthedocs.org/en/latest/)

有关 WSGI 的一些重要背景信息可以在以下链接找到：

[`www.python.org/dev/peps/pep-0333/`](https://www.python.org/dev/peps/pep-0333/)

我们将从 HTTP 协议开始。然后，我们可以考虑诸如 Apache httpd 之类的服务器来实现此协议，并了解`mod_wsgi`如何成为基本服务器的合理扩展。有了这些背景，我们可以看看 WSGI 的功能性质以及如何利用功能设计来实现复杂的 Web 搜索和检索工具。

# HTTP 请求-响应模型

基本的 HTTP 协议理想上是无状态的。用户代理或客户端可以从功能性的角度看待协议。我们可以使用`http.client`或`urllib`库构建客户端。HTTP 用户代理基本上执行类似于以下内容的操作：

```py
import urllib.request
with urllib.request.urlopen(""http://slott-softwarearchitect.blogspot.com"") as response:
 **print(response.read())

```

像**wget**或**curl**这样的程序在命令行上执行此操作；URL 是从参数中获取的。浏览器响应用户的指向和点击执行此操作；URL 是从用户的操作中获取的，特别是点击链接文本或图像的操作。

然而，互联网协议的实际考虑导致了一些有状态的实现细节。一些 HTTP 状态代码表明用户代理需要额外的操作。

3xx 范围内的许多状态代码表示所请求的资源已经移动。然后，用户代理需要根据`Location`头部中发送的信息请求新的位置。401 状态代码表示需要进行身份验证；用户代理可以响应一个包含访问服务器的凭据的授权头部。`urllib`库的实现处理这种有状态的开销。`http.client`库不会自动遐射 3xx 重定向状态代码。

用户代理处理 3xx 和 401 代码的技术并不是深度有状态的。可以使用简单的递归。如果状态不表示重定向，那么它是基本情况，函数有一个结果。如果需要重定向，可以使用重定向地址递归调用函数。

在协议的另一端，静态内容服务器也应该是无状态的。HTTP 协议有两个层次：TCP/IP 套接字机制和依赖于较低级别套接字的更高级别的 HTTP 结构。较低级别的细节由`scoketserver`库处理。Python 的`http.server`库是提供更高级别实现的库之一。

我们可以使用`http.server`库如下：

```py
from http.server import HTTPServer, SimpleHTTPRequestHandler
running = True
httpd = HTTPServer(('localhost',8080), SimpleHTTPRequestHandler)
while running:
 **httpd.handle_request()
httpd.shutdown()

```

我们创建了一个服务器对象，并将其分配给`httpd`变量。我们提供了地址和端口号，以便监听连接请求。TCP/IP 协议将在一个单独的端口上生成一个连接。HTTP 协议将从这个其他端口读取请求并创建一个处理程序的实例。

在这个例子中，我们提供了`SimpleHTTPRequestHandler`作为每个请求实例化的类。这个类必须实现一个最小的接口，它将发送头部，然后将响应的主体发送给客户端。这个特定的类将从本地目录中提供文件。如果我们希望自定义这个，我们可以创建一个子类，实现`do_GET()`和`do_POST()`等方法来改变行为。

通常，我们使用`serve_forever()`方法而不是编写自己的循环。我们在这里展示循环是为了澄清服务器通常必须崩溃。如果我们想要礼貌地关闭服务器，我们将需要一些方法来改变`shutdown`变量的值。例如，*Ctrl + C*信号通常用于这个目的。

## 通过 cookie 注入状态

添加 cookie 改变了客户端和服务器之间的整体关系，使其变得有状态。有趣的是，这并没有改变 HTTP 协议本身。状态信息通过请求和回复的头部进行通信。用户代理将在请求头中发送与主机和路径匹配的 cookie。服务器将在响应头中向用户代理发送 cookie。

因此，用户代理或浏览器必须保留 cookie 值的缓存，并在每个请求中包含适当的 cookie。Web 服务器必须接受请求头中的 cookie，并在响应头中发送 cookie。Web 服务器不需要缓存 cookie。服务器仅仅将 cookie 作为请求中的附加参数和响应中的附加细节。

虽然 cookie 原则上可以包含几乎任何内容，但是 cookie 的使用已经迅速发展为仅包含会话状态对象的标识符。服务器可以使用 cookie 信息来定位某种持久存储中的会话状态。这意味着服务器还可以根据用户代理请求更新会话状态。这也意味着服务器可以丢弃旧的会话。

“会话”的概念存在于 HTTP 协议之外。它通常被定义为具有相同会话 cookie 的一系列请求。当进行初始请求时，没有 cookie 可用，会创建一个新的会话。随后的每个请求都将包括该 cookie。该 cookie 将标识服务器上的会话状态对象；该对象将具有服务器提供一致的 Web 内容所需的信息。

然而，REST 方法对 Web 服务不依赖于 cookie。每个 REST 请求都是独立的，不适用于整体会话框架。这使得它比使用 cookie 简化用户交互的交互式站点不那么“用户友好”。

这也意味着每个单独的 REST 请求原则上是单独进行身份验证的。在许多情况下，服务器会生成一个简单的令牌，以避免客户端在每个请求中发送更复杂的凭据。这导致 REST 流量使用**安全套接字层**（**SSL**）协议进行安全处理；然后使用`https`方案而不是`http`。在本章中，我们将统称这两种方案为 HTTP。

## 考虑具有功能设计的服务器

HTTP 的一个核心理念是守护程序的响应是请求的函数。从概念上讲，一个 Web 服务应该有一个可以总结如下的顶层实现：

```py
response = httpd(request)

```

然而，这是不切实际的。事实证明，HTTP 请求并不是一个简单的、整体的数据结构。它实际上有一些必需的部分和一些可选的部分。一个请求可能有头部，有一个方法和一个路径，还可能有附件。附件可能包括表单或上传的文件或两者都有。

让事情变得更加复杂的是，浏览器的表单数据可以作为一个查询字符串发送到`GET`请求的路径中。或者，它可以作为`POST`请求的附件发送。虽然存在混淆的可能性，但大多数 Web 应用程序框架将创建 HTML 表单标签，通过`<form>`标签中的"`method=POST`"语句提供它们的数据；然后表单数据将成为一个附件。

## 更深入地观察功能视图

HTTP 响应和请求都有头部和正文。请求可以有一些附加的表单数据。因此，我们可以将 Web 服务器看作是这样的：

```py
headers, content = httpd(headers, request, [uploads])

```

请求头可能包括 cookie 值，这可以被视为添加更多参数。此外，Web 服务器通常依赖于其运行的操作系统环境。这个操作系统环境数据可以被视为作为请求的一部分提供的更多参数。

内容有一个大而相当明确定义的范围。**多用途互联网邮件扩展**（**MIME**）类型定义了 Web 服务可能返回的内容类型。这可以包括纯文本、HTML、JSON、XML，或者网站可能提供的各种非文本媒体。

当我们更仔细地观察构建对 HTTP 请求的响应所需的处理时，我们会看到一些我们想要重用的共同特征。可重用元素的这一理念导致了从简单到复杂的 Web 服务框架的创建。功能设计允许我们重用函数的方式表明，功能方法似乎非常适合构建 Web 服务。

我们将通过嵌套请求处理的各种元素来创建服务响应的管道，来研究 Web 服务的功能设计。我们将通过嵌套请求处理的各种元素来创建服务响应的管道，这样内部元素就可以摆脱外部元素提供的通用开销。这也允许外部元素充当过滤器：无效的请求可以产生错误响应，从而使内部函数可以专注于应用程序处理。

## 嵌套服务

我们可以将 Web 请求处理视为许多嵌套上下文。例如，外部上下文可能涵盖会话管理：检查请求以确定这是现有会话中的另一个请求还是新会话。内部上下文可能提供用于表单处理的令牌，可以检测**跨站点请求伪造**（**CSRF**）。另一个上下文可能处理会话中的用户身份验证。

先前解释的函数的概念视图大致如下：

```py
response= content(authentication(csrf(session(headers, request, [forms]))))

```

这里的想法是每个函数都可以建立在前一个函数的结果之上。每个函数要么丰富请求，要么拒绝请求，因为它是无效的。例如，`session`函数可以使用标头来确定这是一个现有会话还是一个新会话。`csrf`函数将检查表单输入，以确保使用了正确的令牌。CSRF 处理需要一个有效的会话。`authentication`函数可以为缺乏有效凭据的会话返回错误响应；当存在有效凭据时，它可以丰富请求的用户信息。

`content`函数不必担心会话、伪造和非经过身份验证的用户。它可以专注于解析路径，以确定应提供什么类型的内容。在更复杂的应用程序中，`content`函数可能包括从路径元素到确定适当内容的函数的相当复杂的映射。

然而，嵌套函数视图仍然不太对。问题在于每个嵌套上下文可能还需要调整响应，而不是或者除了调整请求之外。

我们真的希望更像这样：

```py
def session(headers, request, forms):
 **pre-process: determine session
 **content= csrf(headers, request, forms)
 **post-processes the content
 **return the content
def csrf(headers, request, forms):
 **pre-process: validate csrf tokens
 **content=  authenticate(headers, request, forms)
 **post-processes the content
 **return the content

```

这个概念指向了通过一系列嵌套的函数来创建丰富输入或丰富输出或两者的功能设计。通过一点巧妙，我们应该能够定义一个简单的标准接口，各种函数可以使用。一旦我们标准化了接口，我们就可以以不同的方式组合函数并添加功能。我们应该能够满足我们的函数式编程目标，编写简洁而富有表现力的程序，提供 Web 内容。

# WSGI 标准

**Web 服务器网关接口**（**WSGI**）为创建对 Web 请求的响应定义了一个相对简单的标准化设计模式。Python 库的`wsgiref`包包括了 WSGI 的一个参考实现。

每个 WSGI“应用程序”都具有相同的接口：

```py
def some_app(environ, start_response):
 **return content

```

`environ`是一个包含请求参数的字典，具有统一的结构。标头、请求方法、路径、表单或文件上传的任何附件都将在环境中。除此之外，还提供了操作系统级别的上下文以及一些属于 WSGI 请求处理的项目。

`start_response`是一个必须用于发送响应状态和标头的函数。负责构建响应的 WSGI 服务器的部分将使用`start_response`函数来发送标头和状态，以及构建响应文本。对于某些应用程序，可能需要使用高阶函数包装此函数，以便向响应添加额外的标头。

返回值是一个字符串序列或类似字符串的文件包装器，将返回给用户代理。如果使用 HTML 模板工具，则序列可能只有一个项目。在某些情况下，比如**Jinja2**模板，模板可以作为文本块序列进行延迟渲染，将模板填充与向用户代理下载交错进行。

由于它们的嵌套方式，WSGI 应用程序也可以被视为一个链。每个应用程序要么返回错误，要么将请求交给另一个应用程序来确定结果。

这是一个非常简单的路由应用程序：

```py
SCRIPT_MAP = {
 **""demo"": demo_app,
 **""static"": static_app,
 **"""": welcome_app,
}
def routing(environ, start_response):
 **top_level= wsgiref.util.shift_path_info(environ)
 **app= SCRIPT_MAP.get(top_level, SCRIPT_MAP[''])
 **content= app(environ, start_response)
 **return content

```

此应用程序将使用`wsgiref.util.shift_path_info()`函数来调整环境。这将对请求路径中的项目进行“头/尾拆分”，可在`environ['PATH_INFO']`字典中找到。路径的头部——直到第一个“拆分`”——将被移动到环境中的`SCRIPT_NAME`项目中；`PATH_INFO`项目将被更新为路径的尾部。返回值也将是路径的头部。在没有要解析的路径的情况下，返回值是`None`，不会进行环境更新。

`routing()`函数使用路径上的第一项来定位`SCRIPT_MAP`字典中的应用程序。我们使用`SCRIPT_MAP['']`字典作为默认值，以防所请求的路径不符合映射。这似乎比 HTTP `404 NOT FOUND`错误好一点。

这个 WSGI 应用程序是一个选择多个其他函数的函数。它是一个高阶函数，因为它评估数据结构中定义的函数。

很容易看出，一个框架可以使用正则表达式来概括路径匹配过程。我们可以想象使用一系列正则表达式（REs）和 WSGI 应用程序来配置`routing()`函数，而不是从字符串到 WSGI 应用程序的映射。增强的`routing()`函数应用程序将评估每个 RE 以寻找匹配项。在匹配的情况下，可以使用任何`match.groups()`函数来在调用请求的应用程序之前更新环境。

## 在 WSGI 处理过程中抛出异常

WSGI 应用程序的一个中心特点是，沿着链的每个阶段都负责过滤请求。其想法是尽可能早地拒绝有错误的请求。Python 的异常处理使得这变得特别简单。

我们可以定义一个 WSGI 应用程序，提供静态内容如下：

```py
def static_app(environ, start_response):
 **try:
 **with open(CONTENT_HOME+environ['PATH_INFO']) as static:
 **content= static.read().encode(""utf-8"")
 **headers= [
 **(""Content-Type"",'text/plain; charset=""utf-8""'),(""Content-Length"",str(len(content))),]
 **start_response('200 OK', headers)
 **return [content]
 **except IsADirectoryError as e:
 **return index_app(environ, start_response)
 **except FileNotFoundError as e:
 **start_response('404 NOT FOUND', [])
 **return([repr(e).encode(""utf-8"")])

```

在这种情况下，我们只是尝试打开所请求的路径作为文本文件。我们无法打开给定文件的两个常见原因，这两种情况都作为异常处理：

+   如果文件是一个目录，我们将使用不同的应用程序来呈现目录内容

+   如果文件根本找不到，我们将返回一个 HTTP 404 NOT FOUND 响应

此 WSGI 应用程序引发的任何其他异常都不会被捕获。调用此应用程序的应用程序应设计有一些通用的错误响应能力。如果它不处理异常，将使用通用的 WSGI 失败响应。

### 注意

我们的处理涉及严格的操作顺序。我们必须读取整个文件，以便我们可以创建一个适当的 HTTP `Content-Length`头。

此外，我们必须以字节形式提供内容。这意味着 Python 字符串必须被正确编码，并且我们必须向用户代理提供编码信息。甚至错误消息`repr(e)`在下载之前也要被正确编码。

## 务实的 WSGI 应用程序

WSGI 标准的目的不是定义一个完整的 Web 框架；目的是定义一组最低限度的标准，允许 Web 相关处理的灵活互操作。一个框架可以采用与内部架构完全不同的方法来提供 Web 服务。但是，它的最外层接口应与 WSGI 兼容，以便可以在各种上下文中使用。

诸如**Apache httpd**和**Nginx**之类的 Web 服务器有适配器，它们提供了从 Web 服务器到 Python 应用程序的 WSGI 兼容接口。有关 WSGI 实现的更多信息，请访问

[`wiki.python.org/moin/WSGIImplementations`](https://wiki.python.org/moin/WSGIImplementations)。

将我们的应用程序嵌入到一个更大的服务器中，可以让我们有一个整洁的关注分离。我们可以使用 Apache httpd 来提供完全静态的内容，比如.css、.js 和图像文件。但是对于 HTML 页面，我们可以使用 Apache 的`mod_wsgi`接口将请求转交给一个单独的 Python 进程，该进程只处理网页内容的有趣部分。

这意味着我们必须要么创建一个单独的媒体服务器，要么定义我们的网站有两组路径。如果我们采取第二种方法，一些路径将有完全静态的内容，可以由 Apache httpd 处理。其他路径将有动态内容，将由 Python 处理。

在使用 WSGI 函数时，重要的是要注意我们不能以任何方式修改或扩展 WSGI 接口。例如，提供一个附加参数，其中包含定义处理链的函数序列，似乎是一个好主意。每个阶段都会从列表中弹出第一个项目作为处理的下一步。这样的附加参数可能是函数设计的典型，但接口的改变违背了 WSGI 的目的。

WSGI 定义的一个后果是配置要么使用全局变量，要么使用请求环境，要么使用一个函数，该函数从缓存中获取一些全局配置对象。使用模块级全局变量适用于小例子。对于更复杂的应用程序，可能需要一个配置缓存。可能还有必要有一个 WSGI 应用程序，它仅仅更新`environ`字典中的配置参数，并将控制权传递给另一个 WSGI 应用程序。

# 将 web 服务定义为函数

我们将研究一个 RESTful web 服务，它可以“切割和切块”数据源，并提供 JSON、XML 或 CSV 文件的下载。我们将提供一个整体的 WSGI 兼容包装器，但是应用程序的“真正工作”的函数不会被狭窄地限制在 WSGI 中。

我们将使用一个简单的数据集，其中包括四个子集合：安斯康姆四重奏。我们在第三章“函数、迭代器和生成器”中讨论了读取和解析这些数据的方法。这是一个小数据集，但可以用来展示 RESTful web 服务的原则。

我们将把我们的应用程序分成两个层次：一个是 web 层，它将是一个简单的 WSGI 应用程序，另一个是其余的处理，它将是更典型的函数式编程。我们首先看看 web 层，这样我们就可以专注于提供有意义的结果的函数式方法。

我们需要向 web 服务提供两个信息：

+   我们想要的四重奏——这是一个“切割和切块”的操作。在这个例子中，它主要是一个“切片”。

+   我们想要的输出格式。

数据选择通常通过请求路径完成。我们可以请求`/anscombe/I/`或`/anscombe/II/`来从四重奏中选择特定的数据集。这个想法是 URL 定义了一个资源，而且没有好的理由让 URL 发生变化。在这种情况下，数据集选择器不依赖于日期，或者一些组织批准状态或其他外部因素。URL 是永恒和绝对的。

输出格式不是 URL 的一部分。它只是一个序列化格式，而不是数据本身。在某些情况下，格式是通过 HTTP“接受”头请求的。这在浏览器中很难使用，但在使用 RESTful API 的应用程序中很容易使用。从浏览器中提取数据时，通常使用查询字符串来指定输出格式。我们将在路径的末尾使用`?form=json`方法来指定 JSON 输出格式。

我们可以使用的 URL 看起来像这样：

```py
http://localhost:8080/anscombe/III/?form=csv

```

这将请求第三个数据集的 CSV 下载。

## 创建 WSGI 应用程序

首先，我们将使用一个简单的 URL 模式匹配表达式来定义我们应用程序中唯一的路由。在一个更大或更复杂的应用程序中，我们可能会有多个这样的模式：

```py
import re
path_pat= re.compile(r""^/anscombe/(?P<dataset>.*?)/?$"")

```

这种模式允许我们在路径的顶层定义一个整体的 WSGI 意义上的“脚本”。在这种情况下，脚本是“anscombe”。我们将路径的下一个级别作为要从 Anscombe Quartet 中选择的数据集。数据集值应该是`I`、`II`、`III`或`IV`中的一个。

我们对选择条件使用了一个命名参数。在许多情况下，RESTful API 使用以下语法进行描述：

```py
/anscombe/{dataset}/

```

我们将这种理想化的模式转化为一个适当的正则表达式，并在路径中保留了数据集选择器的名称。

这是演示这种模式如何工作的单元测试的一种类型：

```py
test_pattern= """"""
>>> m1= path_pat.match(""/anscombe/I"")
>>> m1.groupdict()
{'dataset': 'I'}
>>> m2= path_pat.match(""/anscombe/II/"")
>>> m2.groupdict()
{'dataset': 'II'}
>>> m3= path_pat.match(""/anscombe/"")
>>> m3.groupdict()
{'dataset': ''}
""""""

```

我们可以使用以下命令将三个先前提到的示例包含在整个 doctest 中：

```py
__test__ = {
 **""test_pattern"": test_pattern,
}

```

这将确保我们的路由按预期工作。能够从 WSGI 应用程序的其余部分单独测试这一点非常重要。测试完整的 Web 服务器意味着启动服务器进程，然后尝试使用浏览器或测试工具（如 Postman 或 Selenium）进行连接。访问[`www.getpostman.com`](http://www.getpostman.com)或[`www.seleniumhq.org`](http://www.seleniumhq.org)以获取有关 Postman 和 Selenium 用法的更多信息。我们更喜欢单独测试每个功能。

以下是整个 WSGI 应用程序，其中突出显示了两行命令：

```py
import traceback
import urllib
def anscombe_app(environ, start_response):
 **log= environ['wsgi.errors']
 **try:
 **match= path_pat.match(environ['PATH_INFO'])
 **set_id= match.group('dataset').upper()
 **query= urllib.parse.parse_qs(environ['QUERY_STRING'])
 **print(environ['PATH_INFO'], environ['QUERY_STRING'],match.groupdict(), file=log)
 **log.flush()
 **dataset= anscombe_filter(set_id, raw_data())
 **content, mime= serialize(query['form'][0], set_id, dataset)
 **headers= [
 **('Content-Type', mime),('Content-Length', str(len(content))),        ]
 **start_response(""200 OK"", headers)
 **return [content]
 **except Exception as e:
 **traceback.print_exc(file=log)
 **tb= traceback.format_exc()
 **page= error_page.substitute(title=""Error"", message=repr(e), traceback=tb)
 **content= page.encode(""utf-8"")
 **headers = [
 **('Content-Type', ""text/html""),('Content-Length', str(len(content))),]
 **start_response(""404 NOT FOUND"", headers)
 **return [content]

```

此应用程序将从请求中提取两个信息：`PATH_INFO`和`QUERY_STRING`方法。`PATH_INFO`请求将定义要提取的集合。`QUERY_STRING`请求将指定输出格式。

应用程序处理分为三个函数。`raw_data()`函数从文件中读取原始数据。结果是一个带有`Pair`对象列表的字典。`anscombe_filter()`函数接受选择字符串和原始数据的字典，并返回一个`Pair`对象的列表。然后，将成对的列表通过`serialize()`函数序列化为字节。序列化器应该生成字节，然后可以与适当的头部打包并返回。

我们选择生成一个 HTTP`Content-Length`头。这并不是必需的，但对于大型下载来说是礼貌的。因为我们决定发出这个头部，我们被迫实现序列化的结果，以便我们可以计算字节数。

如果我们选择省略`Content-Length`头部，我们可以大幅改变此应用程序的结构。每个序列化器可以更改为生成器函数，该函数将按照生成的顺序产生字节。对于大型数据集，这可能是一个有用的优化。但是，对于观看下载的用户来说，这可能并不那么愉快，因为浏览器无法显示下载的完成进度。

所有错误都被视为`404 NOT FOUND`错误。这可能会产生误导，因为可能会出现许多个别问题。更复杂的错误处理将提供更多的`try:/except:`块，以提供更多信息反馈。

出于调试目的，我们在生成的网页中提供了一个 Python 堆栈跟踪。在调试的上下文之外，这是一个非常糟糕的主意。来自 API 的反馈应该足够修复请求，什么都不多。堆栈跟踪为潜在的恶意用户提供了太多信息。

## 获取原始数据

`raw_data()`函数在很大程度上是从第三章*函数，迭代器和生成器*中复制的。我们包含了一些重要的更改。以下是我们用于此应用程序的内容：

```py
from Chapter_3.ch03_ex5 import series, head_map_filter, row_iter, Pair
def raw_data():
 **""""""
 **>>> raw_data()['I'] #doctest: +ELLIPSIS
 **(Pair(x=10.0, y=8.04), Pair(x=8.0, y=6.95), ...
 **""""""
 **with open(""Anscombe.txt"") as source:
 **data = tuple(head_map_filter(row_iter(source)))
 **mapping = dict((id_str, tuple(series(id_num,data)))
 **for id_num, id_str in enumerate(['I', 'II', 'III', 'IV'])
 **)
 **return mapping

```

我们打开了本地数据文件，并应用了一个简单的`row_iter()`函数，以将文件的每一行解析为一个单独的行。我们应用了`head_map_filter()`函数来从文件中删除标题。结果创建了一个包含所有数据的元组结构。

我们通过从源数据中选择特定系列，将元组转换为更有用的`dict()`函数。每个系列将是一对列。对于系列`"I`,`"`，它是列 0 和 1。对于系列`"II`,`"`，它是列 2 和 3。

我们使用`dict()`函数与生成器表达式保持一致，与`list()`和`tuple()`函数一样。虽然这并非必要，但有时看到这三种数据结构及其使用生成器表达式的相似之处是有帮助的。

`series()`函数为数据集中的每个*x*，*y*对创建了单独的`Pair`对象。回顾一下，我们可以看到修改这个函数后的输出值，使得生成的`namedtuple`类是这个函数的参数，而不是函数的隐式特性。我们更希望看到`series(id_num,Pair,data)`方法，以查看`Pair`对象是如何创建的。这个扩展需要重写第三章中的一些示例，*函数、迭代器和生成器*。我们将把这留给读者作为练习。

这里的重要变化是，我们展示了正式的`doctest`测试用例。正如我们之前指出的，作为一个整体，Web 应用程序很难测试。必须启动 Web 服务器，然后必须使用 Web 客户端来运行测试用例。然后必须通过阅读 Web 日志来解决问题，这可能很困难，除非显示完整的回溯。最好尽可能多地使用普通的`doctest`和`unittest`测试技术来调试 Web 应用程序。

## 应用过滤器

在这个应用程序中，我们使用了一个非常简单的过滤器。整个过滤过程体现在下面的函数中：

```py
def anscombe_filter(set_id, raw_data):
 **""""""
 **>>> anscombe_filter(""II"", raw_data()) #doctest: +ELLIPSIS
 **(Pair(x=10.0, y=9.14), Pair(x=8.0, y=8.14), Pair(x=13.0, y=8.74), ...
 **""""""
 **return raw_data[set_id]

```

我们将这个微不足道的表达式转换成一个函数有三个原因：

+   函数表示法略微更一致，比下标表达式更灵活

+   我们可以很容易地扩展过滤功能

+   我们可以在此函数的文档字符串中包含单独的单元测试

虽然简单的 lambda 可以工作，但测试起来可能不太方便。

对于错误处理，我们什么也没做。我们专注于有时被称为“快乐路径”的内容：理想的事件序列。在这个函数中出现的任何问题都将引发异常。WSGI 包装函数应该捕获所有异常并返回适当的状态消息和错误响应内容。

例如，`set_id`方法可能在某些方面是错误的。与其过分关注它可能出错的所有方式，我们宁愿让 Python 抛出异常。事实上，这个函数遵循了 Python I 的建议，“最好是寻求宽恕，而不是征求许可”。这个建议在代码中体现为避免“征求许可”：没有寻求将参数限定为有效的准备性`if`语句。只有“宽恕”处理：异常将被引发并在 WSGI 包装函数中处理。这个基本建议适用于前面的原始数据和我们现在将看到的序列化。

## 序列化结果

序列化是将 Python 数据转换为适合传输的字节流的过程。每种格式最好由一个简单的函数来描述，该函数只序列化这一种格式。然后，顶层通用序列化程序可以从特定序列化程序列表中进行选择。序列化程序的选择导致以下一系列函数：

```py
serializers = {
 **'xml': ('application/xml', serialize_xml),
 **'html': ('text/html', serialize_html),
 **'json': ('application/json', serialize_json),
 **'csv': ('text/csv', serialize_csv),
}
def serialize(format, title, data):
 **""""""json/xml/csv/html serialization.
 **>>> data = [Pair(2,3), Pair(5,7)]
 **>>> serialize(""json"", ""test"", data)
 **(b'[{""x"": 2, ""y"": 3}, {""x"": 5, ""y"": 7}]', 'application/json')
 **""""""
 **mime, function = serializers.get(format.lower(), ('text/html', serialize_html))
 **return function(title, data), mime

```

整体`serialize()`函数找到必须在响应中使用的特定序列化程序和特定 MIME 类型。然后调用其中一个特定的序列化程序。我们还在这里展示了一个`doctest`测试用例。我们没有耐心测试每个序列化程序，因为显示一个工作似乎就足够了。

我们将分别查看序列化器。我们将看到序列化器分为两组：产生字符串的序列化器和产生字节的序列化器。产生字符串的序列化器将需要将字符串编码为字节。产生字节的序列化器不需要进一步处理。

对于生成字符串的序列化器，我们需要使用标准的转换为字节的函数组合。我们可以使用装饰器进行函数组合。以下是我们如何将转换为字节标准化：

```py
from functools import wraps
def to_bytes(function):
 **@wraps(function)
 **def decorated(*args, **kw):
 **text= function(*args, **kw)
 **return text.encode(""utf-8"")
 **return decorated

```

我们创建了一个名为`@to_bytes`的小装饰器。这将评估给定的函数，然后使用 UTF-8 对结果进行编码以获得字节。我们将展示如何将其与 JSON、CSV 和 HTML 序列化器一起使用。XML 序列化器直接产生字节，不需要与此额外函数组合。

我们还可以在`serializers`映射的初始化中进行函数组合。我们可以装饰函数定义的引用，而不是装饰函数对象的引用。

```py
serializers = {
 **'xml': ('application/xml', serialize_xml),
 **'html': ('text/html', to_bytes(serialize_html)),
 **'json': ('application/json', to_bytes(serialize_json)),
 **'csv': ('text/csv', to_bytes(serialize_csv)),
}

```

虽然这是可能的，但这似乎并不有用。产生字符串和产生字节的序列化器之间的区别并不是配置的重要部分。

## 将数据序列化为 JSON 或 CSV 格式

JSON 和 CSV 序列化器是类似的函数，因为两者都依赖于 Python 的库进行序列化。这些库本质上是命令式的，因此函数体是严格的语句序列。

这是 JSON 序列化器：

```py
import json
@to_bytes
def serialize_json(series, data):
 **""""""
 **>>> data = [Pair(2,3), Pair(5,7)]
 **>>> serialize_json(""test"", data)
 **b'[{""x"": 2, ""y"": 3}, {""x"": 5, ""y"": 7}]'
 **""""""
 **obj= [dict(x=r.x, y=r.y) for r in data]
 **text= json.dumps(obj, sort_keys=True)
 **return text

```

我们创建了一个字典结构的列表，并使用`json.dumps()`函数创建了一个字符串表示。JSON 模块需要一个具体化的`list`对象；我们不能提供一个惰性生成器函数。`sort_keys=True`参数值对于单元测试是必不可少的。但对于应用程序并不是必需的，而且代表了一些额外的开销。

这是 CSV 序列化器：

```py
import csv, io
@to_bytes
def serialize_csv(series, data):
 **""""""

 **>>> data = [Pair(2,3), Pair(5,7)]
 **>>> serialize_csv(""test"", data)
 **b'x,y\\r\\n2,3\\r\\n5,7\\r\\n'
 **""""""
 **buffer= io.StringIO()
 **wtr= csv.DictWriter(buffer, Pair._fields)
 **wtr.writeheader()
 **wtr.writerows(r._asdict() for r in data)
 **return buffer.getvalue()

```

CSV 模块的读取器和写入器是命令式和函数式元素的混合。我们必须创建写入器，并严格按顺序创建标题。我们使用了`Pair`命名元组的`_fields`属性来确定写入器的列标题。

写入器的`writerows()`方法将接受一个惰性生成器函数。在这种情况下，我们使用了每个`Pair`对象的`_asdict()`方法返回适用于 CSV 写入器的字典。

## 将数据序列化为 XML

我们将使用内置库来看一种 XML 序列化的方法。这将从单个标签构建文档。一个常见的替代方法是使用 Python 内省来检查和映射 Python 对象和类名到 XML 标签和属性。

这是我们的 XML 序列化：

```py
import xml.etree.ElementTree as XML
def serialize_xml(series, data):
 **""""""
 **>>> data = [Pair(2,3), Pair(5,7)]
 **>>> serialize_xml(""test"", data)
 **b'<series name=""test""><row><x>2</x><y>3</y></row><row><x>5</x><y>7</y></row></series>'
 **""""""
 **doc= XML.Element(""series"", name=series)
 **for row in data:
 **row_xml= XML.SubElement(doc, ""row"")
 **x= XML.SubElement(row_xml, ""x"")
 **x.text= str(row.x)
 **y= XML.SubElement(row_xml, ""y"")
 **y.text= str(row.y)
 **return XML.tostring(doc, encoding='utf-8')

```

我们创建了一个顶级元素`<series>`，并将`<row>`子元素放在该顶级元素下面。在每个`<row>`子元素中，我们创建了`<x>`和`<y>`标签，并为每个标签分配了文本内容。

使用 ElementTree 库构建 XML 文档的接口往往是非常命令式的。这使得它不适合于否则功能设计。除了命令式风格之外，注意我们没有创建 DTD 或 XSD。我们没有为标签正确分配命名空间。我们还省略了通常是 XML 文档中的第一项的`<?xml version=""1.0""?>`处理指令。

更复杂的序列化库将是有帮助的。有许多选择。访问[`wiki.python.org/moin/PythonXml`](https://wiki.python.org/moin/PythonXml)获取备选列表。

## 将数据序列化为 HTML

在我们最后一个序列化示例中，我们将看到创建 HTML 文档的复杂性。复杂性的原因是在 HTML 中，我们需要提供一个带有一些上下文信息的整个网页。以下是解决这个 HTML 问题的一种方法：

```py
import string
data_page = string.Template(""""""<html><head><title>Series ${title}</title></head><body><h1>Series ${title}</h1><table><thead><tr><td>x</td><td>y</td></tr></thead><tbody>${rows}</tbody></table></body></html>"""""")
@to_bytes
def serialize_html(series, data):
 **"""""">>> data = [Pair(2,3), Pair(5,7)]>>> serialize_html(""test"", data) #doctest: +ELLIPSISb'<html>...<tr><td>2</td><td>3</td></tr>\\n<tr><td>5</td><td>7</td></tr>...""""""
 **text= data_page.substitute(title=series,rows=""\n"".join(
 **""<tr><td>{0.x}</td><td>{0.y}</td></tr>"".format(row)
 **for row in data)
 **)
 **return text

```

我们的序列化函数有两个部分。第一部分是一个`string.Template()`函数，其中包含了基本的 HTML 页面。它有两个占位符，可以将数据插入模板中。`${title}`方法显示了标题信息可以插入的位置，`${rows}`方法显示了数据行可以插入的位置。

该函数使用简单的格式字符串创建单独的数据行。然后将它们连接成一个较长的字符串，然后替换到模板中。

虽然对于像前面的例子这样简单的情况来说是可行的，但对于更复杂的结果集来说并不理想。有许多更复杂的模板工具可以创建 HTML 页面。其中一些包括在模板中嵌入循环的能力，与初始化序列化的功能分开。访问[`wiki.python.org/moin/Templating`](https://wiki.python.org/moin/Templating)获取备选列表。

# 跟踪使用情况

许多公开可用的 API 需要使用"API 密钥"。API 的供应商要求您注册并提供电子邮件地址或其他联系信息。作为交换，他们提供一个激活 API 的 API 密钥。

API 密钥用于验证访问。它也可以用于授权特定功能。最后，它还用于跟踪使用情况。这可能包括在给定时间段内过于频繁地使用 API 密钥时限制请求。

商业模式的变化是多种多样的。例如，使用 API 密钥是一个计费事件，会产生费用。对于其他企业来说，流量必须达到一定阈值才需要付款。

重要的是对 API 的使用进行不可否认。这反过来意味着创建可以作为用户身份验证凭据的 API 密钥。密钥必须难以伪造，相对容易验证。

创建 API 密钥的一种简单方法是使用加密随机数来生成难以预测的密钥字符串。像下面这样的一个小函数应该足够好：

```py
import random
rng= random.SystemRandom()
import base64
def make_key_1(rng=rng, size=1):
 **key_bytes= bytes(rng.randrange(0,256) for i in range(18*size))
 **key_string= base64.urlsafe_b64encode(key_bytes)
 **return key_string

```

我们使用了`random.SystemRandom`类作为我们安全随机数生成器的类。这将使用`os.urandom()`字节来初始化生成器，确保了一个可靠的不可预测的种子值。我们单独创建了这个对象，以便每次请求密钥时都可以重复使用。最佳做法是使用单个随机种子从生成器获取多个密钥。

给定一些随机字节，我们使用了 base 64 编码来创建一系列字符。在初始随机字节序列中使用三的倍数，可以避免在 base 64 编码中出现任何尾随的"`=`"符号。我们使用了 URL 安全的 base 64 编码，这不会在结果字符串中包含"`/`"或"`+`"字符，如果作为 URL 或查询字符串的一部分使用可能会引起混淆。

### 注意

更复杂的方法不会导致更多的随机数据。使用`random.SystemRandom`可以确保没有人可以伪造分配给另一个用户的密钥。我们使用了*18×8*个随机位，给我们大量的随机密钥。

有多少随机密钥？看一下以下命令及其输出：

```py
>>> 2**(18*8)
22300745198530623141535718272648361505980416

```

成功伪造其他人的密钥的几率很小。

另一种选择是使用`uuid.uuid4()`来创建一个随机的**通用唯一标识符**（**UUID**）。这将是一个 36 个字符的字符串，其中包含 32 个十六进制数字和四个"-"标点符号。随机 UUID 也难以伪造。包含用户名或主机 IP 地址等数据的 UUID 是一个坏主意，因为这会编码信息，可以被解码并用于伪造密钥。使用加密随机数生成器的原因是避免编码任何信息。

RESTful Web 服务器然后将需要一个带有有效密钥和可能一些客户联系信息的小型数据库。如果 API 请求包括数据库中的密钥，相关用户将负责该请求。如果 API 请求不包括已知密钥，则可以用简单的`401 未经授权`响应拒绝该请求。由于密钥本身是一个 24 个字符的字符串，数据库将非常小，并且可以很容易地缓存在内存中。

普通的日志抓取可能足以显示给定密钥的使用情况。更复杂的应用程序可能会将 API 请求记录在单独的日志文件或数据库中，以简化分析。

# 总结

在本章中，我们探讨了如何将功能设计应用于使用基于 REST 的 Web 服务提供内容的问题。我们看了一下 WSGI 标准导致了总体上有点功能性的应用程序的方式。我们还看了一下如何通过从请求中提取元素来将更功能性的设计嵌入到 WSGI 上下文中，以供我们的应用程序函数使用。

对于简单的服务，问题通常可以分解为三个不同的操作：获取数据，搜索或过滤，然后序列化结果。我们用三个函数解决了这个问题：`raw_data()`，`anscombe_filter()`和`serialize()`。我们将这些函数封装在一个简单的 WSGI 兼容应用程序中，以将 Web 服务与围绕提取和过滤数据的“真实”处理分离。

我们还看了 Web 服务函数可以专注于“快乐路径”，并假设所有输入都是有效的方式。如果输入无效，普通的 Python 异常处理将引发异常。WSGI 包装函数将捕获错误并返回适当的状态代码和错误内容。

我们避免了与上传数据或接受来自表单的数据以更新持久数据存储相关的更复杂的问题。这些问题与获取数据和序列化结果并没有显著的复杂性。它们已经以更好的方式得到解决。

对于简单的查询和数据共享，小型 Web 服务应用程序可能会有所帮助。我们可以应用功能设计模式，并确保网站代码简洁而富有表现力。对于更复杂的 Web 应用程序，我们应考虑使用一个能够正确处理细节的框架。

在下一章中，我们将看一些可用于我们的优化技术。我们将扩展来自第十章*Functools 模块*的`@lru_cache`装饰器。我们还将研究一些其他优化技术，这些技术在第六章*递归和归约*中提出。
