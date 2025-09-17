

# 第二章：发现 Quart

**Quart** 是在 2017 年作为流行的 **Flask** 框架的进化而开始的。Quart 与 Flask 具有许多相同的设计决策，因此很多关于 Flask 的建议也适用于 Quart。本书将专注于 Quart，以便我们支持异步操作并探索诸如 WebSocket 和 HTTP/2 支持等功能。

Quart 和 Flask 不是唯一的 Python 框架。在网络上提供服务的项目有着悠久的历史，例如 **Bottle**、**cherrypy** 和 **Django**。所有这些工具都在网络上被使用，它们都拥有一个类似的目标：为 Python 社区提供构建 Web 应用程序的简单工具。

小型框架，如 Quart 和 Bottle，通常被称为微框架；然而，这个术语可能有些误导。这并不意味着你只能创建微应用。使用这些工具，你可以构建任何大小应用。前缀“微”意味着框架试图做出尽可能少的决策。它让你可以自由组织你的应用程序代码，并使用你想要的任何库。

微框架充当粘合代码，将请求传递到你的系统中，并发送响应。它不会对你的项目强制执行任何特定的范式。

这种哲学的一个典型例子是当你需要与 SQL 数据库交互时。Django 等框架是“电池组”式的，提供了构建 Web 应用程序所需的一切，包括一个 **对象关系映射器**（**ORM**）来绑定对象与数据库查询结果。

如果你想在 Django 中使用 SQLAlchemy 等替代 ORM 来利用其一些优秀功能，你将选择一条艰难的道路，因为这将涉及重写你希望利用的 Django 库中的大量代码，因为 Django 与其自带的 ORM 集成非常紧密。对于某些应用程序来说，这可能是个好事，但并不一定适用于构建微服务。

另一方面，Quart 没有内置的库来与你的数据交互，这让你可以自由选择自己的库。框架只会尝试确保它有足够的钩子，以便外部库可以扩展并提供各种功能。换句话说，在 Quart 中使用 ORM，并确保你正确处理 SQL 会话和事务，主要就是将 SQLAlchemy 等包添加到你的项目中。如果你不喜欢某个特定库的集成方式，你可以自由地使用另一个库或自己构建集成。Quart 也可以使用更常见的 Flask 扩展，尽管这样做存在性能风险，因为这些扩展不太可能是异步的，可能会阻塞你的应用程序的工作。

当然，这并不是万能的解决方案。在您的选择上完全自由也意味着更容易做出糟糕的决定，构建一个依赖于有缺陷的库或设计不佳的应用程序。但不必担心！本章将确保您了解 Quart 提供的内容，以及如何组织代码来构建微服务。

本章涵盖以下主题：

+   确保我们已安装 Python

+   Quart 如何处理请求

+   Quart 的内置功能

+   微服务骨架

本章的目标是提供您构建 Quart 微服务所需的所有信息。通过这样做，不可避免地会重复一些您可以在 Quart 官方文档中找到的信息，但专注于提供构建微服务时有趣细节和任何相关内容。Quart 和 Flask 都有良好的在线文档。

确保您查看 Quart 和 Flask 的文档，分别列出如下：

+   [`pgjones.gitlab.io/quart/index.html`](https://pgjones.gitlab.io/quart/index.html)

+   [`flask.palletsprojects.com/`](https://flask.palletsprojects.com/)

这两者都应作为本章的绝佳补充。源代码位于 [`gitlab.com/pgjones/quart`](https://gitlab.com/pgjones/quart)。

这一点值得注意，因为当您需要了解软件如何工作时，源代码总是终极真理。

# 确保我们已安装 Python

在我们开始深入研究其功能之前，我们应该确保我们已经安装并配置了 Python！

您可能会在网上看到一些提及 *Python 版本 2* 的文档或帖子。从 Python 2 到 Python 3 的过渡期很长，如果这本书几年前就写成，我们可能会讨论每个版本的优点。然而，Python 3 已经能够满足大多数人所需的所有功能，Python 2 在 2020 年停止由核心 Python 团队支持。本书使用最新的 Python 3.9 稳定版本来展示所有代码示例，但它们很可能在 Python 3.7 或更高版本上也能运行，因为这是 Quart 所需的最小版本。

如果您的计算机没有至少 Python 3.7，您可以从 Python 的官方网站下载新版本，那里提供了安装说明：[`www.python.org/downloads/`](https://www.python.org/downloads/)。

如果你在这个书中所有的代码示例都在虚拟环境中运行，你会发现这会更容易。或者使用 virtualenv ([`docs.python.org/3/library/venv.html`](https://docs.python.org/3/library/venv.html))。虚拟环境是 Python 保持每个项目独立的方式，这意味着你可以安装 Quart 和任何其他你需要的库；它只会影响你当前正在工作的应用程序。其他应用程序和项目可以有不同的库，或者同一库的不同版本，而不会相互干扰。使用 virtualenv 还意味着你可以轻松地在其他地方重新创建项目依赖，这在我们在后面的章节中部署微服务时将非常有用。

一些代码编辑器，如 PyCharm 或 Visual Studio，可能会为你管理虚拟环境。本书中的每个代码示例都在终端中运行，因此我们将使用终端来创建我们的 virtualenv。这也显示了比在网页或日志文件中查看程序输出更详细的工作方式，并且在未来修复任何问题时将非常有帮助。

在终端中，例如 macOS 终端应用程序或 Windows Subsystem for Linux，切换到你想要工作的目录，并运行以下命令：

```py
python -m venv my-venv 
```

根据你安装 Python 的方式，你可能需要使用 `python3` 来创建虚拟环境。

这将在当前目录中创建一个名为 `my-venv` 的新虚拟环境。如果你愿意，可以给它指定另一个路径，但重要的是要记住它的位置。要使用虚拟环境，你必须激活它：

```py
source my-venv/bin/activate 
```

在本书中的大多数命令行示例中，我们假设你正在 Linux 上运行，因为大多数在线服务都使用 Linux，所以熟悉它是很好的。这意味着大多数命令在 macOS 或使用 Windows Subsystem for Linux 的 Windows 上也能正常工作。在这些系统上运行 Docker 容器也是可能的，我们将在讨论部署微服务时再详细介绍容器。

现在，让我们安装 Quart，以便我们可以运行示例代码：

```py
pip install quart 
```

要在不关闭终端的情况下停止使用虚拟环境，你可以输入 `deactivate`。不过，现在让我们保持 virtualenv 激活状态，看看 Quart 将如何工作。

# Quart 处理请求的方式

框架的入口点是 `quart.app` 模块中的 `Quart` 类。运行 Quart 应用程序意味着运行这个类的单个实例，它将负责处理传入的 **异步服务器网关接口**（**ASGI**）和 **Web 服务器网关接口**（**WSGI**）请求，将它们分发给正确的代码，然后返回响应。记住，在 *第一章*，*理解微服务* 中，我们讨论了 ASGI 和 WSGI，以及它们如何定义 Web 服务器和 Python 应用程序之间的接口。

Quart 类提供了一个`route`方法，可以用来装饰您的函数。以这种方式装饰函数后，它成为一个视图，并注册到路由系统中。

当一个请求到达时，它将指向一个特定的端点——通常是一个网址（例如[`duckduckgo.com/?q=quart`](https://duckduckgo.com/?q=quart)）或地址的一部分，例如`/api`。路由系统是 Quart 如何将端点与视图连接起来的方式——即运行以处理请求的代码部分。

这是一个功能齐全的 Quart 应用程序的非常基础的示例：

```py
# quart_basic.py
from quart import Quart
app = Quart(__name__)
@app.route("/api")
def my_microservice():
    return {"Hello": "World!"}
if __name__ == "__main__":
    app.run() 
```

所有代码示例都可以在 GitHub 上找到，地址为[`github.com/PacktPublishing/Python-Microservices-Development-2nd-Edition/tree/main/CodeSamples`](https://github.com/PacktPublishing/Python-Microservices-Development-2nd-Edition/tree/main/CodeSample)。

我们看到我们的函数返回一个字典，Quart 知道这应该被编码为 JSON 对象以进行传输。然而，只有查询`/api`端点才会返回值。其他任何端点都会返回 404 错误，表示它找不到您请求的资源，因为我们没有告诉它任何信息！

`__name__`变量，当您运行单个 Python 模块时其值将是`__main__`，是应用程序包的名称。Quart 使用该名称创建一个新的日志记录器来格式化所有日志消息，并找到文件在磁盘上的位置。Quart 将使用该目录作为辅助程序的根目录，例如与您的应用程序关联的配置，以及确定`static`和`templates`目录的默认位置，我们将在后面讨论。

如果您在终端中运行该模块，`Quart`应用程序将运行其自己的开发 Web 服务器，并开始监听端口`5000`上的传入连接。这里，我们假设您仍然处于之前创建的虚拟环境中，并且上面的代码在名为`quart_basic.py`的文件中：

```py
$ python quart_basic.py 
 * Serving Quart app 'quart_basic'
 * Environment: production
 * Please use an ASGI server (e.g. Hypercorn) directly in production
 * Debug mode: False
 * Running on http://localhost:5000 (CTRL + C to quit)
[2020-12-10 14:05:18,948] Running on http://localhost:5000 (CTRL + C to quit) 
```

使用浏览器或`curl`命令访问`http://localhost:5000/api`将返回一个包含正确头部的有效 JSON 响应：

```py
$ curl -v http://localhost:5000/api 
*   Trying localhost...
...
< HTTP/1.1 200
< content-type: application/json
< content-length: 18
< date: Wed, 02 Dec 2020 20:29:19 GMT
< server: hypercorn-h11
<
* Connection #0 to host localhost left intact
{"Hello":"World!"}* Closing connection 0 
```

本书将大量使用`curl`命令。如果您在 Linux 或 macOS 下，它应该已经预安装；请参阅[`curl.haxx.se/`](https://curl.haxx.se/)。

如果您不是在测试计算机上开发应用程序，您可能需要调整一些设置，例如它应该使用哪些 IP 地址来监听连接。当我们讨论部署微服务时，我们将介绍一些更好的更改其配置的方法，但现在，可以将`app.run`行更改为使用不同的`host`和`port`：

```py
app.run(host="0.0.0.0", port=8000) 
```

虽然许多 Web 框架明确地将`request`对象传递给您的代码，但 Quart 提供了一个全局的`request`变量，它指向它为传入的 HTTP 请求构建的当前`request`对象。

这个设计决策使得简单视图的代码非常简洁。就像我们的例子一样，如果你不需要查看请求内容来回复，就没有必要保留它。只要你的视图返回客户端应该得到的内容，并且 Quart 可以序列化它，一切就会如你所愿发生。对于其他视图，它们只需导入该变量并使用它即可。

`request`变量是全局的，但它对每个传入的请求都是唯一的，并且是线程安全的。让我们在这里添加一些`print`方法调用，以便我们可以看到底层发生了什么。我们还将显式地使用`jsonify`创建一个`Response`对象，而不是让 Quart 为我们做这件事，这样我们就可以检查它：

```py
# quart_details.py
from quart import Quart, request, jsonify
app = Quart(__name__)
@app.route("/api", provide_automatic_options=False)
async def my_microservice():
    print(dir(request))
    response = jsonify({"Hello": "World!"})
    print(response)
    print(await response.get_data())
    return response
if __name__ == "__main__":
    print(app.url_map)
    app.run() 
```

在另一个终端中与`curl`命令一起运行新版本，你会得到很多详细信息，包括以下内容：

```py
$ python quart_details.py 
QuartMap([<QuartRule '/api' (HEAD, GET, OPTIONS) -> my_microservice>,
 <QuartRule '/static/<filename>' (HEAD, GET, OPTIONS) -> static>])
Running on http://localhost:5000 (CTRL + C to quit)

[… '_load_field_storage', '_load_form_data', '_load_json_data', '_send_push_promise', 'accept_charsets', 'accept_encodings', 'accept_languages', 'accept_mimetypes', 'access_control_request_headers', 'access_control_request_method', 'access_route', 'args', 'authorization', 'base_url', 'blueprint', 'body', 'body_class', 'body_timeout', 'cache_control', 'charset', 'content_encoding', 'content_length', 'content_md5', 'content_type', 'cookies', 'data', 'date', 'dict_storage_class', 'encoding_errors', 'endpoint', 'files', 'form', 'full_path', 'get_data', 'get_json', 'headers', 'host', 'host_url', 'http_version', 'if_match', 'if_modified_since', 'if_none_match', 'if_range', 'if_unmodified_since', 'is_json', 'is_secure', 'json', 'list_storage_class', 'max_forwards', 'method', 'mimetype', 'mimetype_params', 'on_json_loading_failed', 'origin', 'parameter_storage_class', 'path', 'pragma', 'query_string', 'range', 'referrer', 'remote_addr', 'root_path', 'routing_exception', 'scheme', 'scope', 'send_push_promise', 'url', 'url_charset', 'url_root', 'url_rule', 'values', 'view_args']
Response(200)
b'{"Hello":"World!"}' 
```

让我们探索这里发生了什么：

+   `路由`: 当服务启动时，Quart 创建`QuartMap`对象，我们在这里可以看到它对端点和相关视图的了解。

+   `Request`: Quart 创建一个`Request`对象，`my_microservice`向我们展示它是一个对`/api`的`GET`请求。

+   `dir()`显示了一个类中哪些方法和变量，例如`get_data()`用于检索随请求发送的任何数据。

+   `Response`: 要发送回客户端的`Response`对象；在这种情况下，是`curl`。它有一个 HTTP 响应代码`200`，表示一切正常，并且其数据是我们告诉它发送的'Hello world'字典。

## 路由

路由发生在`app.url_map`中，这是一个`QuartMap`类的实例，它使用一个名为`Werkzeug`的库。该类使用正则表达式来确定由`@app.route`装饰的函数是否与传入的请求匹配。路由只查看你在路由调用中提供的路径，以查看它是否与客户端的请求匹配。

默认情况下，映射器将只接受在声明的路由上的`GET`、`OPTIONS`和`HEAD`方法。使用不支持的方法向有效的端点发送 HTTP 请求将返回一个`405 Method Not Allowed`响应，并在`allow`头中附带支持的方法列表：

```py
$ curl -v -XDELETE  http://localhost:5000/api
**   Trying 127.0.0.1...
* TCP_NODELAY set
* Connected to localhost (127.0.0.1) port 5000 (#0)
> DELETE /api HTTP/1.1
> Host: localhost:5000
> User-Agent: curl/7.64.1
> Accept: */*
>
< HTTP/1.1 405
< content-type: text/html
< allow: GET, OPTIONS, HEAD
< content-length: 137
< date: Wed, 02 Dec 2020 21:14:36 GMT
< server: hypercorn-h11
<
<!doctype html>
<title>405 Method Not Allowed</title>
<h1>Method Not Allowed</h1>
Specified method is invalid for this resource
* Connection #0 to host 127.0.0.1 left intact
    * Closing connection 0 
```

如果你想要支持特定的方法，允许你向端点`POST`或`DELETE`一些数据，你可以通过`methods`参数将它们传递给`route`装饰器，如下所示：

```py
@app.route('/api', methods=['POST', 'DELETE', 'GET']) 
def my_microservice(): 
    return {'Hello': 'World!'} 
```

注意，由于请求处理器自动管理，所有规则中都会隐式添加`OPTIONS`和`HEAD`方法。你可以通过将`provide_automatic_options=False`参数传递给`route`函数来禁用此行为。这在你想要在`OPTIONS`被调用时向响应添加自定义头时很有用，例如在处理**跨源资源共享**（**CORS**）时，你需要添加几个`Access-Control-Allow-*`头。

关于`HTTP`请求方法的更多信息，一个很好的资源是 Mozilla 开发者网络：[`developer.mozilla.org/en-US/docs/Web/HTTP/Methods`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods).

### 变量和转换器

对于 API 的一个常见要求是能够指定我们想要请求的确切数据。例如，如果你有一个系统中每个人都有一个唯一的数字来识别他们，你可能会想创建一个处理发送到`/person/N`端点的所有请求的函数，这样`/person/3`只处理 ID 号为`3`的人，而`/person/412`只影响 ID 为`412`的人。

你可以使用`route`中的变量来做这件事，使用`<VARIABLE_NAME>`语法。这种表示法相当标准（`Bottle`也使用它），允许你用动态值描述端点。如果我们创建一个`route`，例如`/person/<person_id>`，那么，当 Quart 调用你的函数时，它会将 URL 中找到的值转换为具有相同名称的函数参数：

```py
@app.route('/person/<person_id>') 
def person(person_id): 
    return {'Hello': person_id}

$ curl localhost:5000/person/3 
{"Hello": "3"} 
```

如果你有几个匹配相同 URL 的路由，映射器会使用一组特定的规则来确定调用哪个。`Quart`和`Flask`都使用`Werkzeug`来组织它们的路由；这是从 Werkzeug 的路由模块中摘取的实现描述：

1.  没有参数的规则优先考虑性能。这是因为我们期望它们匹配得更快，一些常见的规则通常没有参数（索引页面等）。

1.  更复杂的规则优先，因此第二个参数是权重的负数。

1.  最后，我们按实际权重排序。

因此，Werkzeug 的规则有用于排序的权重，而在 Quart 中这些权重既没有被使用也没有被显示。所以，这归结为首先选择变量较多的视图，然后按出现顺序选择其他视图，当 Python 导入不同的模块时。一个经验法则是确保你的应用程序中声明的每个路由都是唯一的，否则追踪哪个被选中会让你头疼。

这也意味着我们的新路由不会对发送到`/person`、`/person/3/help`或任何其他变体的查询做出响应——只对`/person/`后面跟一些字符集的查询做出响应。然而，字符包括字母和标点符号，而且我们已经决定`/api/apiperson_id`是一个数字！这就是转换器有用的地方。

我们可以告诉`route`一个变量具有特定的类型。由于`/api/apiperson_id`是一个整数，我们可以使用`<int:person_id>`，就像前面的例子一样，这样我们的代码只有在给出数字时才响应，而不是给出名称时。你还可以看到，与字符串`"3"`不同，`person_id`是一个没有引号的数字：

```py
@app.route('/person/<int:person_id>') 
def person(person_id): 
    return {'Hello': person_id} 
```

```py
$ curl localhost:5000/person/3 
{ 
  "Hello": 3 
} 
$ curl localhost:5000/person/simon
<!doctype html>
<title>404 Not Found</title>
<h1>Not Found</h1>
Nothing matches the given URI 
```

如果我们有两个路由，一个用于 `/person/<int:person_id>`，另一个用于 `/person/<person_id>`（具有不同的函数名！），那么需要整数的更具体的一个将获取所有在正确位置有数字的请求，而另一个函数将获取剩余的请求。

内置转换器包括 `string`（默认为 Unicode 字符串）、`int`、`float`、`path`、`any` 和 `uuid`。

路径转换器类似于默认转换器，但包括正斜杠，因此对 URL 的请求，如 `/api/some/path/like/this`，将匹配路由 `/api/<path:my_path>`，函数将获得一个名为 `my_path` 的参数，其中包含 `some/path/like/this`。如果你熟悉正则表达式，它类似于匹配 `[^/].*?`。

`int` 和 `float` 用于整数和浮点数——十进制数。`any` 转换器允许你组合多个值。一开始使用可能会有些困惑，但如果你需要将几个特定的字符串路由到同一位置，它可能很有用。路由 `/<any(about, help, contact):page_name>` 将匹配对 `/about`、`/help` 或 `/contact` 的请求，并且所选的哪一个将包含在传递给函数的 `page_name` 变量中。

`uuid` 转换器匹配 UUID 字符串，例如从 Python 的 `uuid` 模块中获得的字符串，提供唯一的标识符。所有这些转换器在实际操作中的示例也包含在本章的 GitHub 代码示例中。

创建自定义转换器相当简单。例如，如果你想匹配用户 ID 和用户名，你可以创建一个查找数据库并将整数转换为用户名的转换器。为此，你需要创建一个从 `BaseConverter` 类派生的类，该类实现两个方法：`to_python()` 方法将值转换为视图的 Python 对象，以及 `to_url()` 方法用于反向转换（由 `url_for()` 使用，将在下一节中描述）：

```py
# quart_converter.py
from quart import Quart, request 
from werkzeug.routing import BaseConverter, ValidationError
_USERS = {"1": "Alice", "2": "Bob"}
_IDS = {val: user_id for user_id, val in _USERS.items()}
class RegisteredUser(BaseConverter):
    def to_python(self, value):
        if value in _USERS:
            return _USERS[value]
        raise ValidationError()
    def to_url(self, value):
        return _IDS[value]
app = Quart(__name__)
app.url_map.converters["registered"] = RegisteredUser
@app.route("/api/person/<registered:name>")
def person(name):
    return {"Hello": name}
if __name__ == "__main__":
    app.run() 
```

如果转换失败，将引发 `ValidationError` 方法，映射器将考虑该 `route` 简单地不匹配该请求。让我们尝试几个调用，看看它在实际中是如何工作的：

```py
$ curl localhost:5000/api/person/1 
{ 
  "Hello hey": "Alice" 
}

$ curl localhost:5000/api/person/2 
{ 
  "Hello hey": "Bob" 
}

$ curl localhost:5000/api/person/3 

<!doctype html>
<title>404 Not Found</title>
<h1>Not Found</h1>
Nothing matches the given URI 
```

注意，上面的只是一个展示转换器强大功能的示例——一个以这种方式处理个人信息的 API 可能会向恶意人员泄露大量信息。当代码演变时，更改所有路由也可能很痛苦，因此最好只在必要时使用这种技术。

路由的最佳实践是尽可能保持其静态和直接。这一点在移动所有端点需要更改所有连接到它们的软件时尤其正确！通常，在端点的 URL 中包含一个版本号是一个好主意，这样就可以立即清楚地知道，例如 `/v1/person` 和 `/v2/person` 之间的行为将有所不同。

### `url_for` 函数

Quart 的路由系统的最后一个有趣特性是`url_for()`函数。给定任何视图，它将返回其实际 URL。以下是一个使用 Python 交互式使用的示例：

```py
>>> from quart_converter import app 
>>> from quart import url_for 
>>> import asyncio
>>> async def run_url_for():
...     async with app.test_request_context("/", method="GET"):
...         print(url_for('person', name='Alice')) 
... 
>>> loop = asyncio.get_event_loop()
>>> loop.run_until_complete(run_url_for())
/api/person/1 
```

之前的示例使用了**读取-评估-打印循环**（**REPL**），您可以通过直接运行 Python 可执行文件来获取它。那里还有一些额外的代码来设置异步程序，因为在这里，Quart 不会为我们做这件事。

当您想在模板中显示某些视图的 URL 时，`url_for`功能非常有用——这取决于执行上下文。您不必硬编码一些链接，只需将函数名指向`url_for`即可获取它。

## 请求

当收到请求时，Quart 调用视图并使用请求上下文来确保每个请求都有一个隔离的环境，特定于该请求。我们在上面的代码中看到了一个例子，我们使用辅助方法`test_request_context()`进行测试。换句话说，当你在视图中访问全局请求对象时，你可以保证它是针对你特定请求的处理。

如我们之前在调用`dir(request)`时看到的，当涉及到获取有关正在发生的事情的信息时，`Request`对象包含许多方法，例如请求的计算机地址、请求类型以及其他信息，如授权头。请随意使用示例代码作为起点，实验一些这些请求方法。

在以下示例中，客户端发送的 HTTP 基本认证请求在发送到服务器时始终被转换为 base64 形式。Quart 将检测 Basic 前缀，并将其解析为`request.authorization`属性中的`username`和`password`字段：

```py
# quart_auth.py
from quart import Quart, request
app = Quart(__name__)
@app.route("/")
def auth():
    print("Quart's Authorization information")
    print(request.authorization)
    return ""
if __name__ == "__main__":
    app.run() 
```

```py
$ python quart_auth.py 
* Running on http://localhost:5000/ (Press CTRL+C to quit) 
Quart's Authorization information
{'username': 'alice', 'password': 'password'} 
[2020-12-03 18:34:50,387] 127.0.0.1:55615 GET / 1.1 200 0 3066
$ curl http://localhost:5000/ --user alice:password 
```

这种行为使得在`request`对象之上实现可插拔的认证系统变得容易。其他常见的请求元素，如 cookies 和 files，都可以通过其他属性访问，正如我们将在整本书中了解到的那样。

## 响应

在许多之前的示例中，我们只是返回了一个 Python 字典，并让 Quart 为我们生成客户端可以理解的响应。有时，我们调用`jsonify()`以确保结果是 JSON 对象。

有其他方法可以为我们的 Web 应用程序生成响应，还有一些其他值会自动转换为正确的对象。我们可以返回以下任何一个，Quart 都会做正确的事：

+   `Response()`：手动创建一个`Response`对象。

+   `str`：字符串将被编码为响应中的 text/html 对象。这对于 HTML 页面特别有用。

+   `dict`：字典将被`jsonify()`编码为 application/json。

+   可以返回一个生成器或异步生成器对象，以便将数据流式传输到客户端。

+   一个 `(response, status)` 元组：如果响应匹配前面提到的数据类型之一，它将被转换为 `response` 对象，状态将是使用的 HTTP 响应代码。

+   一个 `(response, status, headers)` 元组：响应将被转换，并且 `response` 对象将使用提供的字典作为头部信息添加到响应中。

在大多数情况下，一个微服务将返回一些其他软件将解释并选择如何显示的数据，因此如果我们想返回一个列表或其他可以序列化为 JSON 的对象，我们将返回 Python 字典或使用 `jsonify()`。

这里有一个使用 YAML 的示例，YAML 是另一种流行的数据表示方式：`yamlify()` 函数将返回一个 `(response, status, headers)` 元组，该元组将被 Quart 转换成一个合适的 `Response` 对象：

```py
# yamlify.py
from quart import Quart
import yaml  # requires PyYAML
app = Quart(__name__)
def yamlify(data, status=200, headers=None):
    _headers = {"Content-Type": "application/x-yaml"}
    if headers is not None:
        _headers.update(headers)
    return yaml.safe_dump(data), status, _headers
@app.route("/api")
def my_microservice():
    return yamlify(["Hello", "YAML", "World!"])
if __name__ == "__main__":
    app.run() 
```

Quart 处理请求的方式可以总结如下：

1.  当应用程序启动时，任何使用 `@app.route()` 装饰的函数都被注册为一个视图并存储在 `app.url_map` 中。

1.  根据其端点和方法，将调用分配到正确的视图中。

1.  在一个局部、隔离的执行上下文中创建了一个 `Request` 对象。

1.  一个 `Response` 对象封装了要发送的内容。

这四个步骤基本上就是你开始使用 Quart 构建应用程序所需了解的全部内容。下一节将总结 Quart 提供的最重要内置功能，以及这种请求-响应机制。

# Quart 的内置功能

上一节让我们对 Quart 处理请求的方式有了很好的理解，这对您开始使用 Quart 已经足够了。还有更多有用的辅助工具。在本节中，我们将发现以下主要工具：

+   `session` 对象：基于 Cookie 的数据

+   **全局变量**：在 `request` 上下文中存储数据

+   **信号**：发送和拦截事件

+   **扩展和中间件**：添加功能

+   **模板**：构建基于文本的内容

+   **配置**：在 `config` 文件中分组你的运行选项

+   **蓝图**：在命名空间中组织你的代码

+   **错误处理和调试**：处理应用程序中的错误

## `session` 对象

与 `request` 对象类似，Quart 创建了一个 `session` 对象，它是 `request` 上下文唯一的。它是一个类似于字典的对象，Quart 将其序列化成用户端的 cookie。会话映射中包含的数据将被转换为 JSON 映射，然后使用 `zlib` 进行压缩以减小其大小，最后使用 base64 进行编码。

当 `session` 被序列化时，**itsdangerous** ([`pythonhosted.org/itsdangerous/`](https://pythonhosted.org/itsdangerous/)) 库使用在应用程序中定义的 `secret_key` 值对内容进行签名。签名使用 **HMAC** ([`en.wikipedia.org/wiki/Hash-based_message_authentication_code`](https://en.wikipedia.org/wiki/Hash-based_message_authentication_code)) 和 SHA1。

这个签名作为数据的后缀添加，确保客户端无法篡改存储在 cookie 中的数据，除非他们知道用于签名会话值的秘密密钥。请注意，数据本身并未加密。Quart 允许您自定义要使用的签名算法，但在您需要将数据存储在 cookie 中时，HMAC + SHA1 已经足够好了。

然而，当您构建不生成 HTML 的微服务时，您很少依赖 cookie，因为它们是特定于网络浏览器的。但是，为每个用户保持一个易失性的键值存储的想法对于加快一些服务器端工作可以非常有用。例如，如果您需要在每次用户连接时执行一些数据库查找以获取与用户相关的信息，那么在服务器端将此信息缓存在一个类似 `session` 的对象中，并根据其身份验证详细信息检索值是非常有意义的。

## 全局变量

如本章前面所述，Quart 提供了一种机制来存储特定于某个 `request` 上下文的全局变量。这用于 `request` 和 `session`，但也可以用来存储任何自定义对象。

`quart.g` 变量包含所有全局变量，您可以在其上设置任何属性。在 Quart 中，可以使用 `@app.before_request` 装饰器指向一个函数，该函数将在每次请求之前被应用程序调用，在将请求分派到视图之前。

在 Quart 中，使用 `before_request` 来设置全局值是一种典型模式。这样，在请求上下文中调用的所有函数都可以与名为 `g` 的特殊全局变量交互并获取数据。在下面的示例中，我们复制了客户端在执行 HTTP Basic Authentication 时提供的 `username`，并将其存储在 `user` 属性中：

```py
# globals.py
from quart import Quart, g, request
app = Quart(__name__)
@app.before_request
def authenticate():
    if request.authorization:
        g.user = request.authorization["username"]
    else:
        g.user = "Anonymous"
@app.route("/api")
def my_microservice():
    return {"Hello": g.user}
if __name__ == "__main__":
    app.run() 
```

当客户端请求 `/api` 视图时，`authenticate` 函数将根据提供的头信息设置 `g.user`：

```py
$ curl http://localhost:5000/api 
{ 
  "Hello": "Anonymous" 
} 
$ curl http://localhost:5000/api --user alice:password 
{ 
  "Hello": "alice" 
} 
```

任何您可能想到的特定于 `request` 上下文的数据，并且在整个代码中可以有用地共享，都可以添加到 `quart.g` 中。

## 信号

有时在应用程序中，我们希望在组件没有直接连接的情况下，从一个地方向另一个地方发送消息。我们可以发送此类消息的一种方式是使用信号。Quart 与 `Blinker` ([`pythonhosted.org/blinker/`](https://pythonhosted.org/blinker/)) 集成，这是一个信号库，允许您将函数订阅到事件。

事件是 `AsyncNamedSignal` 类的实例，该类基于 `blinker.base.NamedSignal` 类。它使用一个唯一的标签创建，Quart 在 0.13 版本中创建了 10 个这样的实例。Quart 在请求处理的关键时刻触发信号。由于 `Quart` 和 `Flask` 使用相同的系统，我们可以参考以下完整列表：[`flask.pocoo.org/docs/latest/api/#core-signals-list`](http://flask.pocoo.org/docs/latest/api/#core-signals-list)。

通过调用信号的`connect`方法来注册特定事件。当某些代码调用信号的`send`方法时，会触发信号。`send`方法接受额外的参数，以便将数据传递给所有已注册的函数。

在以下示例中，我们将`finished`函数注册到`request_finished`信号。该函数将接收`response`对象：

```py
# signals.py
from quart import Quart, g, request_finished
from quart.signals import signals_available
app = Quart(__name__)
def finished(sender, response, **extra):
    print("About to send a Response")
    print(response)
request_finished.connect(finished)
@app.route("/api")
async def my_microservice():
    return {"Hello": "World"}
if __name__ == "__main__":
    app.run() 
```

`signal`功能由`Blinker`提供，当你安装`Quart`时，`Blinker`作为依赖项默认安装。

Quart 实现的一些信号在微服务中可能没有用，例如当框架渲染模板时发生的信号。然而，有一些有趣的信号在 Quart 的整个`request`生命周期中被触发，可以用来记录正在发生的事情。例如，当框架在处理异常之前发生异常时，会触发`got_request_exception`信号。这就是**Sentry**([`sentry.io`](https://sentry.io))的 Python 客户端如何将自己挂钩以记录异常的方式。

当你想通过事件触发一些功能并解耦代码时，在你的应用程序中实现自定义信号可能也很有趣。例如，如果你的微服务生成 PDF 报告，并且你想对报告进行加密签名，你可以触发一个`report_ready`信号，并让签名者注册该事件。

信号实现的一个重要方面是，注册的函数不会按任何特定顺序调用，因此如果被调用的函数之间存在依赖关系，这可能会导致问题。如果你需要执行更复杂或耗时的操作，那么考虑使用`queue`，如**RabbitMQ**([`www.rabbitmq.com/`](https://www.rabbitmq.com/))或由云平台如 Amazon Simple Queue Service 或 Google PubSub 提供的`queue`，将消息发送到另一个服务。这些消息队列提供了比基本信号更多的选项，并允许两个组件轻松通信，甚至不必在相同的计算机上。我们将在*第六章*，*与其他服务交互*中介绍消息队列的示例。

## 扩展和中间件

Quart 扩展只是 Python 项目，一旦安装，就提供名为`quart_something`的包或模块。它们在需要执行如身份验证或发送电子邮件等操作时，可以避免重新发明轮子。

因为`Quart`可以支持一些`Flask`可用的扩展，你通常可以在 Flask 的扩展列表中找到一些有用的东西：在 Python 包索引[`pypi.org/`](https://pypi.org/)中搜索`Framework::Flask`。要使用`Flask`扩展，你必须首先导入一个`patch`模块以确保它能够正常工作。例如，要导入 Flask 的`login`扩展，请使用以下命令：

```py
import quart.flask_patch
import flask_login 
```

已知与 Quart 兼容的 Flask 扩展的最新列表可以在以下地址找到。这是在寻找你的微服务需要的额外功能时的一个好起点：[`pgjones.gitlab.io/quart/how_to_guides/flask_extensions.html`](http://pgjones.gitlab.io/quart/how_to_guides/flask_extensions.html)。

扩展 Quart 的另一种机制是使用 ASGI 或 WSGI 中间件。这些通过围绕端点包装自身来扩展应用程序，并改变进出数据。

在下面的示例中，中间件模拟了一个 `X-Forwarded-For` 标头，这样 Quart 应用程序就会认为它位于一个代理（如 `nginx`）后面。在测试环境中，当你想确保应用程序在尝试获取远程 IP 地址时表现正常时，这很有用，因为 `remote_addr` 属性将获取代理的 IP 地址，而不是真实客户端的 IP 地址。在这个例子中，我们必须创建一个新的 `Headers` 对象，因为现有的一个是不可变的：

```py
# middleware.py
from quart import Quart, request
from werkzeug.datastructures import Headers
class XFFMiddleware:
    def __init__(self, app, real_ip="10.1.1.1"):
        self.app = app
        self.real_ip = real_ip
    async def __call__(self, scope, receive, send):
        if "headers" in scope and "HTTP_X_FORWARDED_FOR" not in scope["headers"]:
            new_headers = scope["headers"].raw_items() + [
                (
                    b"X-Forwarded-For",
                    f"{self.real_ip}, 10.3.4.5, 127.0.0.1".encode(),
                )
            ]
            scope["headers"] = Headers(new_headers)
        return await self.app(scope, receive, send)
app = Quart(__name__)
app.asgi_app = XFFMiddleware(app.asgi_app)
@app.route("/api")
def my_microservice():
    if "X-Forwarded-For" in request.headers:
        ips = [ip.strip() for ip in request.headers["X-Forwarded-For"].split(",")]
        ip = ips[0]
    else:
        ip = request.remote_addr
    return {"Hello": ip}
if __name__ == "__main__":
    app.run() 
```

注意，我们在这里使用 `app.asgi_app` 来包装 ASGI 应用程序。`app.asgi_app` 是应用程序存储的地方，以便人们可以以这种方式包装它。`send` 和 `receive` 参数是通过它们我们可以通信的通道。值得记住的是，如果中间件向客户端返回响应，那么 `Quart` 应用程序将永远不会看到该请求！

在大多数情况下，我们不需要编写自己的中间件，只需包含一个扩展来添加其他人已经制作的功能就足够了。

## 模板

如我们之前所看到的示例，发送回 JSON 或 YAML 文档是足够简单的。同样，大多数微服务产生的是机器可读数据，如果人类需要阅读它，前端必须正确地格式化它，例如在网页上使用 JavaScript。然而，在某些情况下，我们可能需要创建具有某些布局的文档，无论是 HTML 页面、PDF 报告还是电子邮件。

对于任何基于文本的内容，Quart 集成了名为 **Jinja** 的模板引擎（[`jinja.palletsprojects.com/`](https://jinja.palletsprojects.com/)）。你经常会看到示例展示 Jinja 被用来创建 HTML 文档，但它可以与任何基于文本的文档一起使用。配置管理工具，如 Ansible，使用 Jinja 从模板创建配置文件，以便计算机的设置可以自动保持最新。

大多数情况下，Quart 会使用 Jinja 来生成 HTML 文档、电子邮件消息或其他面向人类的通信内容——例如短信或与 Slack 或 Discord 等工具上的人交谈的机器人。Quart 提供了如 `render_template` 这样的辅助工具，通过选择一个 Jinja 模板并给出一些数据来生成响应。

例如，如果你的微服务发送电子邮件而不是依赖于标准库的电子邮件包来生成电子邮件内容，这可能会很繁琐，那么你可以使用 Jinja。以下示例电子邮件模板应该保存为 `email_template.j2`，以便后续的代码示例能够正常工作：

```py
Date: {{date}} 
From: {{from}} 
Subject: {{subject}} 
To: {{to}} 
Content-Type: text/plain 

Hello {{name}}, 

We have received your payment! 

Below is the list of items we will deliver for lunch: 

{% for item in items %}- {{item['name']}} ({{item['price']}} Euros) 
{% endfor %} 

Thank you for your business! 

-- 
My Fictional Burger Place 
```

Jinja 使用双括号来标记将被值替换的变量。变量可以是执行时传递给 Jinja 的任何内容。你还可以直接在模板中使用 Python 的 `if` 和 `for` 块，使用 `{% for x in y % }... {% endfor %}` 和 `{% if x %}...{% endif %}` 标记。

以下是一个使用电子邮件模板生成完全有效的 `RFC 822` 消息的 Python 脚本，你可以通过 SMTP 发送它：

```py
# email_render.py
from datetime import datetime
from jinja2 import Template
from email.utils import format_datetime
def render_email(**data):
    with open("email_template.j2") as f:
        template = Template(f.read())
    return template.render(**data)
data = {
    "date": format_datetime(datetime.now()),
    "to": "bob@example.com",
    "from": "shopping@example-shop.com",
    "subject": "Your Burger order",
    "name": "Bob",
    "items": [
        {"name": "Cheeseburger", "price": 4.5},
        {"name": "Fries", "price": 2.0},
        {"name": "Root Beer", "price": 3.0},
    ],
}
print(render_email(**data)) 
```

`render_email` 函数使用 `Template` 类来生成电子邮件，使用提供的数据。

Jinja 是一个强大的工具，并附带了许多在这里描述会占用太多空间的特性。如果你需要在你的微服务中进行一些模板化工作，它是一个不错的选择，也存在于 Quart 中。查看以下链接以获取 Jinja 特性的完整文档：[`jinja.palletsprojects.com/`](https://jinja.palletsprojects.com/)。

## 配置

在构建应用程序时，你需要公开运行它们所需的选项，例如连接到数据库所需的信息、要使用的联系电子邮件地址或任何特定于部署的变量。

Quart 在其配置方法上使用了一种类似于 Django 的机制。`Quart` 对象包含一个名为 `config` 的对象，其中包含一些内置变量，并且可以在你启动 `Quart` 应用程序时通过你的配置对象进行更新。例如，你可以在 Python 格式的文件中定义一个 `Config` 类，如下所示：

```py
# prod_settings.py
class Config:
    DEBUG = False
    SQLURI = "postgres://username:xxx@localhost/db" 
```

然后，你可以使用 `app.config.from_object` 从你的 `app` 对象中加载它：

```py
>>> from quart import Quart
>>> import pprint
>>> pp = pprint.PrettyPrinter(indent=4)
>>> app = Quart(__name__) 
>>> app.config.from_object('prod_settings.Config') 
>>> pp.pprint(app.config) 
{   'APPLICATION_ROOT': None,
    'BODY_TIMEOUT': 60,
    'DEBUG': False,
    'ENV': 'production',
    'JSONIFY_MIMETYPE': 'application/json',
    'JSONIFY_PRETTYPRINT_REGULAR': False,
    'JSON_AS_ASCII': True,
    'JSON_SORT_KEYS': True,
    'MAX_CONTENT_LENGTH': 16777216,
    'PERMANENT_SESSION_LIFETIME': datetime.timedelta(days=31),
    'PREFER_SECURE_URLS': False,
    'PROPAGATE_EXCEPTIONS': None,
    'RESPONSE_TIMEOUT': 60,
    'SECRET_KEY': None,
    'SEND_FILE_MAX_AGE_DEFAULT': datetime.timedelta(seconds=43200),
    'SERVER_NAME': None,
    'SESSION_COOKIE_DOMAIN': None,
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_NAME': 'session',
    'SESSION_COOKIE_PATH': None,
    'SESSION_COOKIE_SAMESITE': None,
    'SESSION_COOKIE_SECURE': False,
    'SESSION_REFRESH_EACH_REQUEST': True,
    'SQLURI': 'postgres://username:xxx@localhost/db',
    'TEMPLATES_AUTO_RELOAD': None,
    'TESTING': False,
    'TRAP_HTTP_EXCEPTIONS': False} 
```

然而，当使用 Python 模块作为配置文件时，存在两个显著的缺点。首先，由于这些配置模块是 Python 文件，因此很容易在其中添加代码以及简单的值。这样做的话，你将不得不像对待其他应用程序代码一样对待这些模块；这可能会是一种复杂的方式来确保它始终产生正确的值，尤其是在使用模板生成配置的情况下！通常，当应用程序部署时，配置是独立于代码进行管理的。

其次，如果另一个团队负责管理你的应用程序的配置文件，他们需要编辑 Python 代码来完成这项工作。虽然这通常是可以接受的，但它增加了引入一些问题的可能性，因为它假设其他人熟悉 Python 以及你的应用程序的结构。通常，确保只需要更改配置的人不需要了解代码的工作方式是一种良好的实践。

由于 Quart 通过 `app.config` 暴露其配置，因此从 JSON、YAML 或其他流行的基于文本的配置格式中加载附加选项相当简单。以下所有示例都是等效的：

```py
>>> from quart import Quart
>>> import yaml
>>> from pathlib import Path 
>>> app = Quart(__name__)
>>> print(Path("prod_settings.json").read_text())
{
    "DEBUG": false,
    "SQLURI":"postgres://username:xxx@localhost/db"
} 
>>> app.config.from_json("prod_settings.json")
>>> app.config["SQLURI"]
'postgres://username:xxx@localhost/db'
>>> print(Path("prod_settings.yml").read_text())
---
DEBUG: False
SQLURI: "postgres://username:xxx@localhost/db"
>>> app.config.from_file("prod_settings.yml", yaml.safe_load) 
```

你可以为 `from_file` 提供一个用于理解数据的函数，例如 `yaml.safe_load`、`toml.load` 和 `json.load`。如果你更喜欢带有 `[sections]` 和 `name = value` 的 INI 格式，那么存在许多扩展来帮助，标准库的 `ConfigParser` 也非常直接。

## 蓝图

当你编写具有多个端点的微服务时，你将结束于许多不同的装饰函数——记住那些是带有装饰器的函数，例如 `@app.route`。组织代码的第一个逻辑步骤是每个端点有一个模块，当你创建应用实例时，确保它们被导入，这样 Quart 就可以注册视图。

例如，如果你的微服务管理一个公司的员工数据库，你可以有一个端点用于与所有员工交互，另一个端点用于与团队交互。你可以将你的应用程序组织成这三个模块：

+   `app.py`: 包含 `Quart` 应用对象，并运行应用

+   `employees.py`: 用于提供所有与员工相关的视图

+   `teams.py`: 用于提供所有与团队相关的视图

从那里，员工和团队可以被视为应用的子集，可能有一些特定的实用程序和配置。这是构建任何 Python 应用程序的标准方式。

蓝图通过提供一种将视图分组到命名空间中的方法，进一步扩展了这种逻辑，使得在单独的文件中使用这种结构，并为其提供一些特殊的框架支持。你可以创建一个类似于 `Quart` 应用对象的 `Blueprint` 对象，然后使用它来安排一些视图。初始化过程可以通过 `app.register_blueprint` 注册蓝图，以确保蓝图定义的所有视图都是应用的一部分。员工蓝图的一个可能实现如下：

```py
# blueprints.py
from quart import Blueprint
teams = Blueprint("teams", __name__)
_DEVS = ["Alice", "Bob"]
_OPS = ["Charles"]
_TEAMS = {1: _DEVS, 2: _OPS}
@teams.route("/teams")
def get_all():
    return _TEAMS
@teams.route("/teams/<int:team_id>")
def get_team(team_id):
    return _TEAMS[team_id] 
```

主要模块（`app.py`）可以导入此文件，并使用 `app.register_blueprint(teams)` 注册其蓝图。当你想在另一个应用程序或同一应用程序中多次重用一组通用的视图时，这种机制也非常有趣——可以想象一个场景，例如，库存管理区域和销售区域可能都需要查看当前库存水平的能力。

## 错误处理

当你的应用程序出现问题时，能够控制客户端将接收到的响应是很重要的。在 HTML 网络应用中，当你遇到 `404`（资源未找到）或 `5xx`（服务器错误）时，通常会得到特定的 HTML 页面，这就是 Quart 默认的工作方式。但在构建微服务时，你需要对应该发送回客户端的内容有更多的控制——这就是自定义错误处理器有用的地方。

另一个重要功能是在发生意外错误时调试你的代码；Quart 自带一个内置的调试器，可以在你的应用程序以调试模式运行时激活。

### 自定义错误处理器

当你的代码没有处理异常时，Quart 会返回一个 HTTP `500`响应，不提供任何特定信息，如跟踪信息。生成通用错误是一个安全的默认行为，以避免在错误信息的正文中向用户泄露任何私人信息。默认的`500`响应是一个简单的 HTML 页面以及正确的状态码：

```py
$ curl http://localhost:5000/api 
<!doctype html>
<title>500 Internal Server Error</title>
<h1>Internal Server Error</h1>
Server got itself in trouble 
```

当使用 JSON 实现微服务时，一个好的做法是确保发送给客户端的每个响应，包括任何异常，都是 JSON 格式的。你的微服务的消费者期望每个响应都是可机器解析的。告诉客户端你遇到了错误，并让它设置好处理该消息并展示给人类，比给客户端一些他们不理解的东西并让它抛出自己错误要好得多。

Quart 允许你通过几个函数自定义应用程序错误处理。第一个是`@app.errorhandler`装饰器，它的工作方式类似于`@app.route`。但与提供端点不同，装饰器将一个函数链接到一个特定的错误代码。

在以下示例中，我们使用它来连接一个函数，当 Quart 返回`500`服务器响应（任何代码异常）时，该函数将返回 JSON 格式的错误：

```py
# error_handler.py
from quart import Quart
app = Quart(__name__)
@app.errorhandler(500)
def error_handling(error):
    return {"Error": str(error)}, 500
@app.route("/api")
def my_microservice():
    raise TypeError("Some Exception")
if __name__ == "__main__":
    app.run() 
```

无论代码抛出什么异常，Quart 都会调用这个错误视图。然而，如果你的应用程序返回 HTTP `404`或任何其他`4xx`或`5xx`响应，你将回到 Quart 发送的默认 HTML 响应。为了确保你的应用程序为每个`4xx`和`5xx`响应发送 JSON，我们需要将这个函数注册到每个错误代码。

```py
error_handling function to every error using app.register_error_handler, which is similar to the @app.errorhandler decorator:
```

```py
# catch_all_errors.py
from quart import Quart, jsonify, abort
from werkzeug.exceptions import HTTPException, default_exceptions
def jsonify_errors(app):
    def error_handling(error):
        if isinstance(error, HTTPException):
            result = {
                "code": error.code,
                "description": error.description,
                "message": str(error),
            }
        else:
            description = abort.mapping[ error.code].description
            result = {"code":  error.code, "description": description, "message": str(error)}
        resp = jsonify(result)
        resp.status_code = result["code"]
        return resp
    for code in default_exceptions.keys():
        app.register_error_handler(code, error_handling)
    return app
app = Quart(__name__)
app = jsonify_errors(app)
@app.route("/api")
def my_microservice():
   raise TypeError("Some Exception")
if __name__ == "__main__":
    app.run() 
```

`jsonify_errors`函数修改了一个`Quart`应用程序实例，并为可能发生的每个`4xx`和`5xx`错误设置了自定义的 JSON 错误处理器。

# 微服务骨架

到目前为止，在本章中，我们探讨了 Quart 的工作原理，以及它提供的几乎所有内置功能——我们将在整本书中使用这些功能。我们还没有涉及的一个主题是如何组织项目中的代码，以及如何实例化你的`Quart`应用程序。到目前为止的每个示例都使用了一个 Python 模块和`app.run()`调用来运行服务。

将所有内容放在一个模块中是可能的，但除非你的代码只有几行，否则会带来很多麻烦。由于我们希望发布和部署代码，最好将其放在 Python 包中，这样我们就可以使用标准的打包工具，如`pip`和`setuptools`。

将视图组织到蓝图，并为每个蓝图创建一个模块也是一个好主意。这让我们能更好地跟踪每段代码的作用，并在可能的情况下重用代码。

最后，可以从代码中删除 `run()` 调用，因为 Quart 提供了一个通用的运行命令，该命令通过 `QUART_APP` 环境变量的信息查找应用程序。使用该运行器提供了额外的选项，例如，可以在不进入设置的情况下配置用于运行应用程序的主机和端口。

GitHub 上的微服务项目是为本书创建的，是一个通用的 Quart 项目，您可以用它来启动微服务。它实现了一个简单的布局，这对于构建微服务非常有效。您可以安装并运行它，然后对其进行修改。项目可以在 [`github.com/PythonMicroservices/microservice-skeleton`](https://github.com/PythonMicroservices/microservice-skeleton) 找到。

`microservice` 项目骨架包含以下结构：

+   `setup.py`: Distutils 的设置文件，用于安装和发布项目。

+   `Makefile`: 包含一些有用的目标，用于构建、构建和运行项目。

+   `settings.yml`: 在 YAML 文件中的应用默认设置。

+   `requirements.txt`: 根据 `pip freeze` 生成的 `pip` 格式的项目依赖。

+   `myservices/`: 实际的包

    +   `__init__.py`

    +   `app.py`: 包含应用程序本身的模块

    +   `views/`: 包含按蓝图组织视图的目录

        +   `__init__.py`

        +   `home.py`: 为主蓝图，它服务于根端点

    +   `tests/:` 包含所有测试的目录

        +   `__init__.py`

        +   `test_home.py`: 对主蓝图视图的测试

在以下代码中，`app.py` 文件使用名为 `create_app` 的辅助函数实例化 `Quart` 应用程序，以注册蓝图并更新设置：

```py
import os
from myservice.views import blueprints
from quart import Quart
_HERE = os.path.dirname(__file__)
_SETTINGS = os.path.join(_HERE, "settings.ini")
def create_app(name=__name__, blueprints=None, settings=None):
    app = Quart(name)
    # load configuration
    settings = os.environ.get("QUART_SETTINGS", settings)
    if settings is not None:
        app.config.from_pyfile(settings)
    # register blueprints
    if blueprints is not None:
        for bp in blueprints:
            app.register_blueprint(bp)
    return app
app = create_app(blueprints=blueprints, settings=_SETTINGS) 
```

`home.py` 视图使用蓝图创建了一个简单的路由，不返回任何内容：

```py
from quart import Blueprint
home = Blueprint("home", __name__)
@home.route("/")
def index():
    """Home view.
    This view will return an empty JSON mapping.
    """
    return {} 
```

此示例应用程序可以通过 Quart 内置的命令行运行，使用包名：

```py
$ QUART_APP=myservice quart run
 * Serving Quart app 'myservice.app'
 * Environment: production
 * Please use an ASGI server (e.g. Hypercorn) directly in production
 * Debug mode: False
 * Running on http://localhost:5000 (CTRL + C to quit)
[2020-12-06 20:17:28,203] Running on http://127.0.0.1:5000 (CTRL + C to quit) 
```

从那里开始，为您的微服务构建 JSON 视图包括向微服务/views 添加模块及其相应的测试。

# 摘要

本章为我们提供了 Quart 框架的详细概述以及如何使用它来构建微服务。需要记住的主要事项是：

+   Quart 在 ASGI 协议周围包装了一个简单的请求-响应机制，这使得您几乎可以用纯 Python 编写应用程序。

+   Quart 很容易扩展，如果需要，可以使用 Flask 扩展。

+   Quart 内置了一些有用的功能：蓝图、全局变量、信号、模板引擎和错误处理器。

+   微服务项目是一个 Quart 骨架，本书将使用它来编写微服务。

下一章将重点介绍开发方法：如何持续编码、测试和记录您的微服务。
