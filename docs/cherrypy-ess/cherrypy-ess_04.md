# 第四章。CherryPy 深入

第三章介绍了 CherryPy 的常见方面，而没有过多深入细节。在本章中，我们将通过解释关键特性，如如何运行多个 HTTP 服务器、使用额外的 URI 分派器、使用内置工具并开发新的工具、提供静态内容以及最后如何 CherryPy 和 WSGI 交互，深入探讨使 CherryPy 成为网络开发者如此强大的库的原因。本章内容密集，但将为您提供一个良好的基础，让您在使用产品时更加轻松高效。

# HTTP 兼容性

CherryPy 正在稳步发展，尽可能地遵守 HTTP 规范——首先支持旧的 HTTP/1.0，然后逐渐过渡到完全支持 RFC 2616 中定义的 HTTP/1.1。据说 CherryPy 对 HTTP/1.1 的兼容性是有条件的，因为它实现了规范中的所有 *必须* 和 *要求* 级别，但没有实现所有 *应该* 级别。因此，CherryPy 支持以下 HTTP/1.1 的特性：

+   如果客户端声称支持 HTTP/1.1，则必须在任何使用该协议版本的请求中发送 `Host` 头字段。如果没有这样做，CherryPy 将立即停止请求处理，并返回 `400` 错误代码消息（RFC 2616 的第 14.23 节）。

+   CherryPy 在所有配置中生成 `Date` 头字段（RFC 2616 的第 14.18 节）。

+   CherryPy 可以处理客户端支持的 `Continue` 响应状态码（`100`）。

+   CherryPy 内置的 HTTP 服务器支持 HTTP/1.1 中的默认持久连接，通过使用 `Connection: Keep-Alive` 头部。请注意，如果选择的 HTTP 服务器不支持此功能，更改 HTTP 服务器（更多详情请参阅第十章
cherrypy.server.quickstart()

```

如您所见，我们调用了服务器对象的 `quickstart()` 方法，这将实例化内置 HTTP 服务器并在其自己的线程中启动它。

现在想象一下，我们有一个希望在多个网络接口上运行的应用程序；我们应该这样做：

```py
from cherrypy import _cpwsgi
# Create a server on interface 1102.168.0.12 port 100100
s1 = _cpwsgi.CPWSGIServer()
s1.bind_addr = ('1102.168.0.12', 100100)
# Create a server on interface 1102.168.0.27 port 4700
s2 = _cpwsgi.CPWSGIServer()
s2.bind_addr = ('1102.168.0.27', 4700)
# Inform CherryPy which servers to start and use
cherrypy.server.httpservers = {s1: ('1102.168.0.12', 100100),
s2: ('1102.168.0.27', 4700)}
cherrypy.server.start()

```

如您所见，我们首先创建了内置 HTTP 服务器的两个实例，并为每个实例设置了套接字应该监听传入请求的绑定地址。

然后，我们将这些服务器附加到 CherryPy 的 HTTP 服务器池中，并调用 `start()` 方法，这将使每个服务器在其接口上启动。

注意，我们并没有调用 `cherrypy.config.update`，因为这将会更新所有服务器共享的全局配置设置。然而，这实际上并不是一个问题，因为内置服务器的每个实例都有与配置键匹配的属性。因此：

```py
s1.socket_port = 100100
s1.socket_host = '1102.168.0.12'
s1.socket_file = ''
s1.socket_queue_size = 5
s1.socket_timeout = 10
s1.protocol_version = 'HTTP/1.1'
s1.reverse_dns = False
s1.thread_pool = 10
s1.max_request_header_size = 500 * 1024
s1.max_request_body_size = 100 * 1024 * 1024
s1.ssl_certificate = None
s1.ssl_private_key = None

```

如您所见，您可以直接设置服务器实例的设置，避免使用全局配置。这种技术还允许应用程序同时通过 HTTP 和 HTTPS 提供服务，正如我们将在第十章（Chapter 10 中看到的，默认情况下，CherryPy 将 URI 映射到具有`exposed`属性设置为`True`的 Python 可调用对象。随着时间的推移，CherryPy 社区希望更加灵活，并且会欣赏其他调度器的解决方案。这就是为什么 CherryPy 3 提供了另外三个内置调度器，并提供了编写和使用您自己的调度器的一种简单方法。

+   其中一个是设置为允许按 HTTP 方法开发应用程序。（GET、POST、PUT 等。）

+   第二个是基于一个流行的第三方包 Routes，由 Ben Bangert 从 Ruby on Rails 的原始 Ruby 实现中开发而来。

+   第三个调度器是一个虚拟主机调度器，它允许根据请求的域名而不是 URI 路径进行调度。

## HTTP 方法调度器

在某些应用中，URI 与服务器在资源上执行的操作是独立的。例如，看看下面的 URI：

```py
http://somehost.com/album/delete/12

```

如你所见，URI 包含了客户端希望执行的操作。使用默认的 CherryPy 调度器，这会映射到类似以下的内容：

```py
album.delete(12)

```

虽然这样做是可以的，但你可能希望从 URI 本身中移除该操作，使其更加独立，这样它看起来就会像：

```py
http://somehost.com/album/12

```

你可能会立即想知道服务器应该如何知道要执行哪个操作。这个信息由 HTTP 请求本身携带，多亏了 HTTP 方法：

```py
DELETE /album/12 HTTP/1.1

```

处理此类请求的页面处理器看起来如下：

```py
class Album:
exposed = True
def GET(self, id):
....
def POST(self, title, description):
....
def PUT(self, id, title, description):
....
def DELETE(self, id):
....

```

当使用 HTTP 方法分配器时，被调用的页面处理器将是`album.DELETE(12)`。

如果你查看之前的类定义，你会看到方法没有携带`exposed`属性，而是类本身设置了该属性。这个原因来自于分配器实现的方式。

当一个请求到达服务器时，CherryPy 会寻找最佳匹配的页面处理器。当使用 HTTP 方法分配器时，处理器实际上是 URI 所指向的资源的概念性表示，在我们的例子中是`album`类的实例。然后分配器检查该类是否有与请求使用的 HTTP 方法名称匹配的方法。如果有，分配器会使用剩余的参数调用它。否则，它会立即发送 HTTP 错误代码`405 方法不允许`来通知客户端它不能使用 HTTP 方法，因此不能在该特定资源上执行该操作。

例如，如果我们没有在`Album`类中对`DELETE`进行定义，那么在之前使用的请求中会返回这样的错误代码。

然而，无论如何，CherryPy 都会自动将`Allow` HTTP 头添加到响应中，以通知客户端它可以对资源使用哪些方法。

### 注意

注意，在这种情况下，CherryPy 不会像使用 URI 到对象分配器那样寻找`index`或`default`页面处理器。这来自于仅基于 URI 分配与 URI+HTTP 方法分配之间的基本区别。第六章将更详细地讨论这一点。

要启用 HTTP 方法分配器，你必须将`request.dispatch`键设置为针对目标路径的该分配器的实例。

例如，如果我们整个应用程序都是使用那种技术构建的，我们会使用：

```py
{'/' : {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}

```

HTTP 方法分配器通常用于遵循 REST 原则的应用程序中，我们将在第六章中看到。

## Routes 分配器

无论是在 URI 到对象或 HTTP 方法分配器中，我们都没有明确声明与页面处理器关联的 URI；相反，我们将找到最佳对应的责任留给了 CherryPy 引擎。许多开发者更喜欢明确的方法，并决定 URI 应该如何映射到页面处理器。

因此，当使用 Routes 分配器时，你必须连接一个匹配 URI 并关联特定页面处理器的模式。

让我们回顾一个例子：

```py
import cherrypy
class Root:
def index(self):
return "Not much to say"
def hello(self, name):
return "Hello %s" % name
if __name__ == '__main__':
root = Root()
# Create an instance of the dispatcher
d = cherrypy.dispatch.RoutesDispatcher()
# connect a route that will be handled by the 'index' handler
d.connect('default_route', '', controller=root)
# connect a route to the 'hello' handler
# this will match URIs such as '/say/hello/there'
# but not '/hello/there'
d.connect('some_other', 'say/:action/:name',
controller=root, action='hello')
# set the dispatcher
conf = {'/': {'request.dispatch': d}}
cherrypy.quickstart(root, '/', config=conf)

```

### 注意

当使用 Routes 分配器处理器时，你不需要有`exposed`属性。

路由分配器的`connect`方法定义为：

```py
connect(name, route, controller, **kwargs)

```

下面是`connect`方法的参数：

+   `name` 参数是连接到路由的唯一名称。

+   `route` 是匹配 URI 的模式。

+   `controller` 是包含页面处理程序的实例。

+   `**kwargs` 允许你为路由传递额外的有效参数。

请参阅官方 Routes 文档以了解该包的工作方式。

默认情况下，CherryPy 路由调度器不会将 Routes 映射返回的 `action` 和 `controller` 值传递给与任何路由匹配的 URI。这些在 CherryPy 应用程序中不一定有用。然而，如果你需要它们，你可以将 Routes 调度器构造函数的 `fetch_result` 参数设置为 `True`。然后这两个值都将传递给页面处理程序，但在此情况下，你必须将 `controller` 和 `action` 参数添加到所有页面处理程序中。

## 虚拟主机调度器

可能会发生这样的情况，你需要在单个 CherryPy 服务器内托管不同的网络应用程序，每个应用程序服务一个特定的域名。CherryPy 提供了一种简单的方法来实现这一点，如下面的示例所示：

```py
import cherrypy
class Site:
def index(self):
return "Hello, world"
index.exposed = True
class Forum:
def __init__(self, name):
self.name = name
def index(self):
return "Welcome on the %s forum" % self.name
index.exposed = True
if __name__ == '__main__':
site = Site()
site.cars = Forum('Cars')
site.music = Forum('My Music')
hostmap = {'www.ilovecars.com': '/cars',
'www.mymusic.com': '/music',}
cherrypy.config.update({'server.socket_port': 80})
conf = {'/': {'request.dispatch': cherrypy.dispatch.VirtualHost(**hostmap)}}
cherrypy.tree.mount(site, config=conf)
cherrypy.server.quickstart()
cherrypy.engine.start()

```

首先，正如你所看到的，我们只是创建了一个应用程序树。接下来，我们定义 `hostmap` 字典，它将通知 `VirtualHost` 调度器如何根据域名来服务请求。因此，来自 [www.mymusic.com](http://www.mymusic.com) 的请求将由位于 `/music` 前缀的应用程序提供服务。接下来，我们告诉 CherryPy 我们将使用 `VirtualHost` 调度器，我们最后像往常一样挂载网站应用程序并启动服务器。

注意，此示例需要你编辑你机器上的 `hosts` 文件以添加以下两个域名：

```py
127.0.0.1 www.ilovecars.com
127.0.0.1 www.mymusic.com

```

它将自动将请求重定向到这些域名，而不是在互联网上查找它们。完成此示例后，你应该从 `hosts` 文件中删除这些行。

# 将钩子插入到 CherryPy 的核心引擎

CherryPy 最强大的方面之一是其核心如何以非常精细的粒度让你修改其正常行为。实际上，CherryPy 提供了一种称为钩子的机制来定制核心引擎。

**钩子**是 Python 可调用项在请求处理过程中的特定点应用的入口点。CherryPy 提供以下入口点：

| 插入点 | 描述 |
| --- | --- |
| `on_start_resource` | 在进程开始时调用。 |
| `before_request_body` | 在 CherryPy 尝试读取请求体之前调用。它允许一个工具通过在工具中将 `process_request_body` 属性设置为 `False` 来通知 CherryPy 是否应该执行此操作。 |
| `before_handler` | 在页面处理程序被调用之前调用。例如，一个工具可以将处理程序设置为 `None` 来通知 CherryPy 它不应该处理页面处理程序。 |
| `before_finalize` | 无论页面处理程序是否被调用，在 CherryPy 开始处理响应之前调用。 |
| `on_end_resource` | 当资源处理结束时调用。 |
| `before_error_response after_error_response` | 当 CherryPy 引擎捕获到错误时调用，以便应用程序恢复并决定下一步操作。 |
| `on_end_request` | 在整体处理结束时调用，在客户端连接关闭后立即调用。这允许您释放资源。 |

下图显示了 CherryPy 在处理请求时遵循的全局流程。黑色线条和箭头表示正常流程，而灰色线条表示发生错误时的路径。

![Hook into CherryPy's Core Engine](img/1848_04_01.jpg)

在这些钩子点之一附加回调是通过调用以下内容完成的：

```py
cherrypy.request.hooks.attach(point, callback, failsafe=None,
priority=None, **kwargs)

```

第一个参数是钩子点的名称，如前表所示。第二个参数是将被应用的 Python 可调用对象。第三个参数指示 CherryPy，即使另一个回调在处理此钩子点时可能失败，CherryPy 也必须运行此可调用对象。最后一个参数必须是一个介于 0 到 100 之间的值，以指示每个回调的权重并提供一种对它们进行排序的方法。较低的值将首先运行。

`failsafe` 参数非常有用，因为它为应用程序提供了一种灵活地恢复可能发生的问题的方法。确实，一些回调可能会失败，但不会影响请求处理链的整个流程。

### 注意

注意，您可以在给定的钩子点上附加所需的任何数量的回调。回调可以在应用程序运行时即时附加。然而，附加的回调越多，该钩子点的处理速度就会越慢。

钩子机制相当接近 CherryPy 2 中曾经被称为过滤器的东西。然而，随着时间的推移，人们观察到它们过于底层，并且大多数时候让用户感到不舒服。这就是为什么开发者直接使用它们的频率仍然很低。相反，它们通过一个名为 tools 的更高层接口来应用。

# CherryPy 工具箱

工具接口是由 Robert Brewer 在重构 CherryPy 时设计的。目标是提供一套现成的工具，通过友好且灵活的 API 实现常见任务。在 CherryPy 中，内置工具提供了一个单一接口，用于通过钩子机制调用我们在第三章中审查的 CherryPy 库。

正如我们在第三章中看到的，工具可以以三种不同的方式使用：

+   从配置设置

+   作为 Python 装饰器或通过页面处理器的特殊`_cp_config`属性

+   作为可以在任何函数内部应用的 Python 可调用对象

由于这种灵活性，工具可以设置为全局路径及其子集，或者设置为特定的页面处理器。现在让我们回顾 CherryPy 提供的内置工具。

## 基本身份验证工具

**目的：** 此工具的目的是为您的应用程序提供基本认证（RFC 2617）。

**参数：**

| 名称 | 默认 | 描述 |
| --- | --- | --- |
| `realm` | N/A (在此情况下，N/A 表示参数必须由开发者提供，因为它没有默认值。) | 定义领域值的字符串。 |
| `users` | N/A | 形式为 username:password 的字典或返回此类字典的 Python 可调用对象。 |
| `encrypt` | None | 用于加密客户端返回的密码并与用户字典中提供的加密密码进行比较的 Python 可调用对象。如果为 None，则使用 MD5 散列。 |

**示例：**

```py
import sha
import cherrypy
class Root:
@cherrypy.expose
def index(self):
return """<html>
<head></head>
<body>
<a href="admin">Admin area</a>
</body>
</html>
"""
class Admin:
@cherrypy.expose
def index(self):
return "This is a private area"
if __name__ == '__main__':
def get_users():
# 'test': 'test'
return {'test': 'a104a8fe5ccb110ba61c4c0873d3101e10871082fbbd3'}
def encrypt_pwd(token):
return sha.new(token).hexdigest()
conf = {'/admin': {'tools.basic_auth.on': True,
'tools.basic_auth.realm': 'Some site',
'tools.basic_auth.users': get_users,
'tools.basic_auth.encrypt': encrypt_pwd}}
root = Root()
root.admin = Admin()
cherrypy.quickstart(root, '/', config=conf)

```

`get_users`函数返回一个硬编码的字典，但它也可以从数据库或其他地方获取值。请注意，基本认证方案实际上并不安全，因为密码只是编码的，如果有人捕获它，可以即时解码。然而，由于安全套接字层加密了包含的数据，这种方案通常在 SSL 上使用，因为它是最容易实施的。

## 缓存工具

**目的：** 此工具的目的是提供 CherryPy 生成内容的内存缓存。

**参数：**

| 名称 | 默认 | 描述 |
| --- | --- | --- |
| `invalid_methods` | ("POST", "PUT", "DELETE") | 不应缓存的 HTTP 方法字符串元组。这些方法还将使任何缓存的资源副本失效（删除）。 |
| `cache_class` | MemoryCache | 用于缓存的类对象。 |

一个全面的示例超出了本书的范围，但如果您对这个工具感兴趣，您应该首先查看 CherryPy 测试套件，并访问 CherryPy 用户邮件列表。

## 解码工具

**目的：** 此工具的目的是解码传入的请求参数。

**参数：**

| 名称 | 默认 | 描述 |
| --- | --- | --- |
| `encoding` | None | 应使用什么编码来解码传入的内容？如果为 None，则查找`Content-Type`头，如果找不到合适的字符集，则使用`default_encoding`。 |
| `default_encoding` | "UTF-8" | 默认编码，当未提供或找到时将使用此编码。 |

**示例：**

```py
import cherrypy
from cherrypy import tools
class Root:
@cherrypy.expose
def index(self):
return """<html>
<head></head>
<body>
<form action="hello" method="post">
<input type="text" name="name" value="" />
</form>
</body>
</html>
"""
@cherrypy.expose
@tools.decode(encoding='ISO-88510-1')
def hello(self, name):
return "Hello %s" % (name, )
if __name__ == '__main__':
cherrypy.quickstart(Root(), '/')

```

在此情况下，当 HTML 表单发送到服务器时，CherryPy 会尝试使用我们设置的编码来解码传入的数据。如果您查看`name`参数的类型，您会看到当使用解码工具时它是*Unicode*，而没有工具时它是一个*字符串*。

## 摘要认证工具

**目的：** 此工具的目的是提供 RFC 2617 中定义的摘要认证。

**参数：**

| 名称 | 默认 | 描述 |
| --- | --- | --- |
| `realm` | N/A | 定义领域值的字符串。 |
| `users` | N/A | 形式为—username:password 的字典或返回此类字典的 Python 可调用对象。 |

**示例：**

```py
import cherrypy
class Root:
@cherrypy.expose
def index(self):
return """<html>
<head></head>
<body>
<a href="admin">Admin area</a>
</body>
</html>
"""
class Admin:
@cherrypy.expose
def index(self):
return "This is a private area"
if __name__ == '__main__':
def get_users():
return {'test': 'test'}
conf = {'/admin': {'tools.digest_auth.on': True,
'tools.digest_auth.realm': 'Some site',
'tools.digest_auth.users': get_users}}
root = Root()
root.admin = Admin()
cherrypy.quickstart(root, '/', config=conf)

```

注意，摘要工具不提供传递加密密码的方法。这是因为摘要方案定义了不将密码以明文形式发送到网络上。它的工作方式如下：

1.  1. 客户端请求访问资源。服务器返回`401`错误代码，表示它使用摘要方案。服务器为此交换提供令牌。

1.  2. 客户端根据令牌、用户名和密码创建一条新消息，并通过 MD5 算法生成哈希值。

1.  3. 当服务器收到来自客户端的新消息时，它尝试生成相同的值。如果它们都匹配，则允许认证。

正如你所见，密码永远不会以明文形式在网络上传输。已经进行了讨论，以决定如何使摘要工具进化，以避免需要以明文形式存储密码。一种方法是将摘要令牌的中间步骤之一（步骤 1）存储起来，并将此值与客户端发送的值进行比较。这超出了本书的范围，但你可以在 CherryPy 邮件列表中获取更多信息。

## 编码工具

**目的：** 此工具的目的是以定义的编码编码响应内容。

**参数：**

| 名称 | 默认值 | 描述 |
| --- | --- | --- |
| `encoding` | None | 要使用什么编码来编码响应？如果为 None，它将查找`Content-Type`头，并在可能的情况下设置合适的字符集。 |
| `errors` | "strict" | 定义工具在无法编码字符时必须如何反应。 |

**示例：**

```py
import cherrypy
from cherrypy import tools
class Root:
@cherrypy.expose
def index(self):
return """<html>
<head></head>
<body>
<form action="hello" method="post">
<input type="text" name="name" value="" />
</form>
</body>
</html>
"""
@cherrypy.expose
@tools.encode(encoding='ISO-88510-15')
def hello(self, name):
return "Hello %s" % name
if __name__ == '__main__':
cherrypy.quickstart(Root(), '/')

```

## 错误重定向工具

**目的：** 此工具的目的是修改 CherryPy 默认错误处理器。

**参数：**

| 名称 | 默认值 | 描述 |
| --- | --- | --- |
| `url` | '' | 应重定向到的 URL。 |
| `internal` | `True` | 当`True`时，重定向对客户端是隐藏的，并且仅在请求的上下文中发生。如果`False`，CherryPy 会通知客户端客户端应自行向提供的 URL 发出重定向。 |

## Etag 工具

**目的：** 此工具的目的是验证用户代理发送的**实体标签（Etag**），并根据 RFC 2616 第 14.24 节定义生成相应的响应。Etags 是缓存 HTTP 响应并减轻任何相关方负担的一种方式。

**参数：**

| 名称 | 默认值 | 描述 |
| --- | --- | --- |
| `autotags` | False | 当`True`时，工具将根据响应体设置生成一个`etag`值。 |

**示例：**

```py
import cherrypy
from cherrypy import tools
class Root:
@cherrypy.expose
def index(self):
return """<html>
<head></head>
<body>
<form action="hello" method="post">
<input type="text" name="name" value="" />
</form>
</body>
</html>
"""
@cherrypy.expose
def hello(self, name):
return "Hello %s" % name
if __name__ == '__main__':
conf = {'/': {'tools.etags.on': True,
'tools.etags.autotags': True}}
cherrypy.quickstart(Root(), '/', config=conf)

```

在上一个示例中，我们为整个应用程序设置了`etags`工具。在第一次请求`index`页面处理器时，该工具将生成一个`etag`值并将其插入到响应头中。在下一次请求该 URI 时，客户端将包含最后接收到的`etag`。工具将比较它们，如果它们匹配，则响应将为`304 Not Modified`，通知客户端它可以安全地使用其资源副本。

注意，如果您需要以不同的方式计算`etag`值，最佳做法是将`autotags`参数设置为`False`，这是默认值，然后从您的页面处理器中自行添加`Etag`头到响应头中。

## Gzip 工具

**目的：** 此工具的目的是对响应体进行内容编码。

**参数：**

| 名称 | 默认值 | 描述 |
| --- | --- | --- |
| `compress_level` | `10` | 要达到的压缩级别。越低，速度越快。 |
| `mime_types` | `['text/html', 'text/plain']` | 可以压缩的 MIME 类型列表。 |

**示例：**

```py
import cherrypy
from cherrypy import tools
class Root:
@cherrypy.expose
@tools.gzip()
def index(self):
return "this will be compressed"
if __name__ == '__main__':
cherrypy.quickstart(Root(), '/')

```

注意，当响应通过其`stream`属性流式传输时不应使用`gzip`工具。实际上，在这种情况下，CherryPy 在有任何内容要发送时就开始发送主体，例如页面处理器产生内容时，而不是返回它。

## 忽略头工具

**目的：** 此工具的目的是在 CherryPy 处理之前从 HTTP 请求中删除指定的头。

**参数：**

| 名称 | 默认值 | 描述 |
| --- | --- | --- |
| `ignore_headers` | `headers=('Range',)` | 要忽略的头名称元组。 |

**示例：**

```py
import cherrypy
from cherrypy import tools
class Root:
@cherrypy.expose
@tools.ignore_headers(headers=('Accept-Language',))
def index(self):
return "Accept-Language: %s" \
% cherrypy.request.headers.get('Accept-Language',
'none provided')
@cherrypy.expose
def other(self):
return "Accept-Language: %s" % cherrypy.request.headers.get('Accept-Language')
if __name__ == '__main__':
cherrypy.quickstart(Root(), '/')

```

如果您访问[`localhost:8080/`](http://localhost:8080/)，无论客户端是否确实设置了该头，您都将得到以下信息：

```py
Accept-Language: none provided

```

如果您导航到[`localhost:8080/other`](http://localhost:8080/other)，您将得到以下信息：

```py
Accept-Language: en-us,en;q=0.5

```

## 日志头工具

**目的：** 此工具的目的是在服务器上发生错误时将请求头输出到错误日志文件。此工具默认禁用。

**参数：** 无

**示例：**

```py
import cherrypy
from cherrypy import tools
class Root:
@cherrypy.expose
def index(self):
raise StandardError, "Some sensible error message here"
if __name__ == '__main__':
cherrypy.config.update({'global': {'tools.log_headers.on':
True}})
cherrypy.quickstart(Root(), '/')

```

当您访问[`localhost:8080`](http://localhost:8080)时，将引发错误，错误日志将显示请求头。请注意，在这种情况下，此工具是通过`cherrypy.config.update()`方法在 Web 服务器级别设置的，但它也可以按路径级别应用。

## 日志堆栈跟踪工具

**目的：** 此工具的目的是在发生异常时将错误的堆栈跟踪输出到错误日志文件。此工具默认启用。

**参数：** 无

**示例：**

```py
import cherrypy
from cherrypy import tools
class Root:
@cherrypy.expose
def index(self):
raise StandardError, "Some sensible error message here"
if __name__ == '__main__':
# This tool is applied globally to the CherryPy process
# by using the global cherrypy.config.update method.
cherrypy.config.update({'global': {'tools.log_tracebacks.on':
False}})
cherrypy.quickstart(Root(), '/')

```

## 代理工具

**目的：** 此工具的目的是更改请求的基本 URL。当在 Apache 等服务器后面运行应用程序时，这特别有用。

**参数：**

| 名称 | 默认值 | 描述 |
| --- | --- | --- |
| `base` | None | 如果设置且`local`为空，这将是从`cherrypy.request.base`可用的新的基本 URL。 |
| `local` | 'X-Forwarded-Host' | 查找本地主机设置的头部，例如前端 Web 服务器设置的。 |
| `remote` | 'X-Forwarded-For' | 查找原始客户端 IP 地址的头部。 |
| `scheme` | 'X-Forwarded-Proto' | 查找原始方案使用的头部：例如*http*或*https*。 |

当未设置基本 URL 时，该工具将从请求头部获取的值构建新的基本 URI，基于其他参数。

**示例：**

```py
import cherrypy
from cherrypy import tools
class Root:
@cherrypy.expose
def index(self):
return "Base URL: %s %s " % (cherrypy.request.base,
cherrypy.url(''))
@cherrypy.expose
def other(self):
raise cherrypy.HTTPRedirect(cherrypy.url(''))
if __name__ == '__main__':
conf = {'global': {'tools.proxy.on': True,
'tools.proxy.base': 'http://someapp.net/blog',
'tools.proxy.local': ''}}
cherrypy.config.update(conf)
cherrypy.quickstart(Root(), '/')

```

当导航到[`localhost:8080`](http://localhost:8080)时，你会看到以下消息：

```py
Base URL: http://someapp.net/blog http://someapp.net/blog/

```

如果你导航到[`localhost:8080/other`](http://localhost:8080/other)，你将被重定向到[`someapp.net/blog/`](http://someapp.net/blog/)，这表明代理工具以透明的方式确保 CherryPy 库的行为与您提供的设置保持一致。

在此工具后面使用另一个服务器使用示例，请参阅第十章。

## Referer 工具

**目的：** 此工具的目的是允许根据模式过滤请求。在匹配模式后，可以拒绝或接受请求。

**参数：**

| 名称 | 默认值 | 描述 |
| --- | --- | --- |
| `pattern` | N/A | 正则表达式模式。 |
| `accept` | True | 如果为`True`，任何匹配的引用将允许请求继续。否则，任何匹配的引用将导致请求被拒绝。 |
| `accept_missing` | False | 是否允许没有引用的请求。 |
| `error` | 403 | 拒绝时返回给用户的 HTTP 错误代码。 |
| `message` | 'Forbidden Referer header.' | 拒绝时返回给用户的消息。 |

**示例：**

```py
import cherrypy
from cherrypy import tools
class Root:
@cherrypy.expose
def index(self):
return cherrypy.request.headers.get('Referer')
if __name__ == '__main__':
conf = {'/': {'tools.referer.on': True,
'tools.referer.pattern': 'http://[^/]*dodgy\.com',
'tools.referer.accept': False}}
cherrypy.quickstart(Root(), '/', config=conf)

```

在此示例中，我们将拒绝所有来自`dodgy.com`域名及其子域的请求。

## 响应头部工具

**目的：** 此工具的目的是允许一次性为所有或许多页面处理器设置一些常见的头部信息。

**参数：**

| 名称 | 默认值 | 描述 |
| --- | --- | --- |
| `headers` | None | 列表：元组（头部，值） |

**示例：**

```py
import cherrypy
from cherrypy import tools
class Root:
@cherrypy.expose
def index(self):
return "Some text"
@cherrypy.expose
def other(self):
return "Some other text"
if __name__ == '__main__':
conf = {'/': {'tools.response_headers.on': True,
'tools.response_headers.headers': [('Content-Type',
'text/plain')]}}
cherrypy.quickstart(Root(), '/', config=conf)

```

在此示例中，该工具为所有页面处理器设置`Content-Type`为`text/plain`。

## 尾部斜杠工具

**目的：** 此工具的目的是提供一种灵活的方式来处理请求的尾部斜杠。此工具默认启用。

**参数：**

| 名称 | 默认值 | 描述 |
| --- | --- | --- |
| `missing` | True | 如果页面处理器是索引，如果`missing`参数为`True`，并且请求遗漏了尾部斜杠，CherryPy 将自动向带有尾部斜杠的 URI 发出重定向。 |
| `extra` | False | 如果页面处理器不是索引，如果 `extra` 参数设置为 `True`，并且 URI 有尾部斜杠，CherryPy 将向没有尾部斜杠的 URI 发出重定向。 |

**示例：**

```py
import cherrypy
from cherrypy import tools
class Root:
@cherrypy.expose
def index(self):
return "This should have been redirected to add the trailing
slash"
@cherrypy.expose
def nothing(self):
return "This should have NOT been redirected"
nothing._cp_config = {'tools.trailing_slash.on': False}
@cherrypy.expose
def extra(self):
return "This should have been redirected to remove the
trailing slash"
extra._cp_config = {'tools.trailing_slash.on': True,
'tools.trailing_slash.missing': False,
'tools.trailing_slash.extra': True}
if __name__ == '__main__':
cherrypy.quickstart(Root(), '/')

```

要了解这个工具，请导航到以下 URL：

[`localhost:8080`](http://localhost:8080)

[`localhost:8080/nothing`](http://localhost:8080/nothing)

[`localhost:8080/nothing/`](http://localhost:8080/nothing/)

[`localhost:8080/extra/`](http://localhost:8080/extra/)

## XML-RPC 工具

**目的：** 这个工具的目的是将 CherryPy 转换为 XML-RPC 服务器，并使页面处理器成为 XML-RPC 可调用对象。

**参数：** 无

**示例：**

```py
import cherrypy
from cherrypy import _cptools
class Root:
@cherrypy.expose
def index(self):
return "Regular web page handler"
class XMLRPCApp(_cptools.XMLRPCController):
@cherrypy.expose
def echo(self, message):
return message
if __name__ == '__main__':
root = Root()
root.xmlrpc = XMLRPCApp()
cherrypy.quickstart(root, '/')

```

`XMLRPCController` 是一个辅助类，应该用来代替直接使用 XML-RPC 工具。

你可以按照以下方式测试你的 XML-RPC 处理器：

```py
>>> import xmlrpclib
>>> s = xmlrpclib.ServerProxy('http://localhost:8080/xmlrpc')
>>> s.echo('test')
'test'

```

## 工具箱

CherryPy 工具必须属于一个由 CherryPy 引擎管理的工具箱。工具箱有自己的命名空间，以避免名称冲突。尽管没有阻止你使用默认的工具箱，但你也可以创建自己的工具箱，如下所示：

```py
from cherrypy._cptools import Toolbox,
mytb = Toolbox('mytb')
mytb.xml_parse = Tool('before_handler', xmlparse)
conf = {'/': {'mytb.xml_parse.on': True,
'mytb.xml_parse.engine': 'amara'}}

```

## 创建一个工具

现在我们已经审查了 CherryPy 一起提供的工具箱，我们将解释如何编写一个工具。在决定创建一个工具之前，你应该问自己一些问题，例如：

+   应该在 CherryPy 级别处理添加的功能吗？

+   在请求处理的哪个级别应该应用这个功能？

+   你是否会修改 CherryPy 的默认行为？

这些问题只是确保你想要添加的功能处于正确的级别。工具有时看起来像是一个模式，你可以在此基础上设计你的应用程序。

我们将创建一个工具，该工具将读取并解析请求体中包含的 XML，并将其解析为页面处理器参数。为此，我们将使用 ElementTree 库。（ElementTree 由 Fredrik Lundh 维护，Amara 由 Uche Ogbuji 维护。）

工具可以通过继承 `Tool` 类或通过该类的实例来创建，如下面的示例所示。实例化 `Tool` 类是最常见的情况，也是我们将要讨论的情况。

类构造函数声明如下：

```py
Tool(point, callable, name=None, priority=50)

```

+   `point` 参数是一个字符串，指示此工具应附加到哪个钩点。

+   `callable` 参数是一个 Python 可调用对象，将被应用。

+   `name` 参数定义了工具在工具箱中的名称。如果没有提供，它将使用在工具箱中持有工具实例的属性的名称（参考我们的示例）。

+   `priority` 设置了当多个工具附加到相同的钩点时，工具的顺序。

一旦创建了工具的实例，你可以按照以下方式将其附加到内置工具箱：

```py
cherrypy.tools.mytool = Tool('on_start_resource', mycallable)

```

这个工具将像任何其他内置工具一样，对您的应用程序可用。

在创建工具时，你可以为你的可调用对象提供两个属性，这些属性将在初始化工具时使用。它们如下所示：

+   `failsafe:` 如果`True`，则表示即使在工具轮到之前发生错误，工具也会运行。默认为`False`。

+   `priority:` 此工具相对于同一钩点上的其他工具的相对顺序。默认为`50`。

因此，你可以这样写：

```py
def mycallable(...):
CherryPytools, creating....
mycallable.failsafe = True
mycallable.priority = 30
cherrypy.tools.mytool = Tool('on_start_resource', mycallable)

```

CherryPy 为将在`before_handler`钩点处应用的工具提供了一个快捷方式，换句话说，就是在页面处理器被调用之前。这应该是非内置工具最常见的用例之一。

```py
cherrypy.tools.mytool = Tool('before_handler', mycallable)

```

这相当于以下内容：

```py
cherrypy.tools.mytool = HandlerTool(mycallable)

```

`HandlerTool`类提供了一个额外的功能，它允许你的可调用对象通过`HandlerTool`类的`handler(*args, **kwargs)`方法本身作为一个页面处理器应用。因此：

```py
class Root:
other = cherrypy.tools.mytool.handler()

```

这可以在不重复代码的情况下，为应用程序的不同区域提供相同的处理器。

现在让我们看一个更详细的示例：

```py
import cherrypy
from cherrypy import tools
CherryPytools, creatingfrom cherrypy import Tool
from xml.parsers.expat import ExpatError
from xml.sax._exceptions import SAXParseException
def xmlparse(engine='elementtree', valid_content_types=['text/xml',
'application/xml'], param_name='doc'):
# Transform the XML document contained in the request body into
# an instance of the chosen XML engine.
# Get the mime type of the entity sent by the user-agent
ct = cherrypy.request.headers.get('Content-Type', None)
# if it is not a mime type we can handle
# then let's inform the user-agent
if ct not in valid_content_types:
raise cherrypy.HTTPError(415, 'Unsupported Media Type')
# CherryPy will set the request.body with a file object
# where to read the content from
if hasattr(cherrypy.request.body, 'read'):
content = cherrypy.request.body.read()
doc = content
try:
if engine == 'elementtree':
from elementtree import ElementTree as ETX
doc = ETX.fromstring(content)
elif engine == 'amara':
import amara
doc = amara.parse(content)
except (ExpatError, SAXParseException):
raise cherrypy.HTTPError(400, 'XML document not
well-formed')
# inject the parsed document instance into
# the request parameters as if it had been
# a regular URL encoded value
cherrypy.request.params[param_name] = doc
# Create a new Tool and attach it to the default CherryPy toolbox
tools.xml_parse = Tool('before_handler', xmlparse)
class Root:
@cherrypy.expose
@tools.xml_parse()
def echoet(self, doc):
return doc.find('.//message').text
@cherrypy.expose
@tools.xml_parse(engine='amara', param_name='d')
def echoamara(self, d):
return unicode(d.root.message)
if __name__ == '__main__':
cherrypy.quickstart(Root(), '/')

```

### 注意

为了测试这个工具，你需要 ElementTree 或 Amara，或者两者都需要。你可以通过`easy_install`命令安装它们。

我们的 XML 工具将读取 HTTP 正文内容，并通过指定的 XML 工具包进行解析。然后，它将解析的文档注入到请求参数中，以便新的文档实例作为常规参数传递给页面处理器。

启动前面的示例，然后在 Python 解释器中运行：

```py
>>> s = '<root><message>Hello!<message></root>'
>>> headers = {'Content-Type': 'application/xml'}
>>> import httplib
>>> conn = httplib.HTTPConnection("localhost:8080")
>>> conn.request("POST", "/echoet", s, headers)
>>> r1 = conn.getresponse()
>>> print r1.status, r1.reason
200 OK
>>> r1.read()
'Hello!'
>>> conn.request("POST", "/echoamara", s, headers)
>>> r1 = conn.getresponse()
>>> print r1.status, r1.reason
200 OK
>>> r1.read()
'Hello!'
>>> conn.request("POST", "/echoamara", s)
>>> r1 = conn.getresponse()
>>> print r1.status, r1.reason
415 Unsupported Media Type
>>> conn.close()

```

如你所见，CherryPy 3 提供的工具界面功能强大、灵活，同时非常直观且易于重用。然而，在使用工具之前，始终要仔细思考你的需求。它们应该用于适合 HTTP 请求/响应模型的底层操作。

# 静态资源服务

CherryPy 提供了两个简单的工具来服务单个文件或整个目录。在任一情况下，CherryPy 都会通过自动检查请求中的`If-Modified-Since`和`If-Unmodified-Since`头来处理你的静态资源的 HTTP 缓存方面，如果存在，则直接返回`304 Not Modified`响应。

## 使用 Staticfile 工具服务单个文件

`staticfile`工具可以用来服务单个文件。

**参数：**

| 名称 | 默认值 | 描述 |
| --- | --- | --- |
| `filename` | N/A | 物理文件的绝对或相对路径。 |
| `root` | None | 如果文件名是相对的，你必须提供文件的根目录。 |
| `match` | "" | 用于检查 URI 路径是否匹配特定模式的正则表达式。 |
| `content_types` | None | 形如`ext: mime type`的字典。 |

**示例：**

为了这个目的，让我们假设我们有以下目录结构：

```py
application \
myapp.py
design1.css

```

`design1.css`设置如下：

```py
body {
background-color: #86da12;
}

```

`myapp.py`模块将定义如下：

```py
import cherrypy
class MyApp:
@cherrypy.expose
def index(self):
return """<html>
<head>
<title>My application</title>
<link rel="stylesheet" href="css/style.css" type="text/css"></link>
</head>
<html>
<body>
Hello to you.
static resource servingsingle file, Staticfile tool used</body>
</html>"""
if __name__ == '__main__':
import os.path
current_dir = os.path.dirname(os.path.abspath(__file__))
cherrypy.config.update({'environment': 'production',
'log.screen': True})
conf = {'/': {'tools.staticfile.root': current_dir},
'/css/style.css': {'tools.staticfile.on': True,
'tools.staticfile.filename':
'design1.css'}}
cherrypy.quickstart(MyApp(), '/my', config=conf)

```

必须考虑以下几点：

+   根目录可以全局设置整个应用程序，这样你就不必为每个 URI 路径定义它。

+   当使用 `staticfile` 工具时，URI 和物理资源不需要有相同的名称。实际上，它们在命名上可以完全不相关，就像前面的示例一样。

+   注意，尽管应用程序挂载在 `/my` 前缀上，这意味着对 CSS 文件的请求将是 `/my/css/style.css`（注意这是这种情况，因为链接元素中提供的路径是在 `href` 属性中的相对路径，而不是绝对路径：它不以 `/` 开头），我们的配置设置不包括该前缀。正如我们在第三章中看到的，这是因为配置设置与应用程序挂载的位置无关。

## Using the Staticdir Tool to Serve a Complete Directory

`staticdir` 工具可以用来服务一个完整的目录。

**参数：**

| Name | Default | 描述 |
| --- | --- | --- |
| `dir` | N/A | 物理目录的绝对或相对路径。 |
| `root` | None | 如果 `dir` 是相对路径，你必须提供文件的根目录。 |
| `match` | "" | 匹配文件的正则表达式模式。 |
| `content_types` | None | 形式为 ext: mime type 的字典。 |
| `index` | "" | 如果 URI 指向的不是文件而是目录，你可以指定要服务的物理索引文件名。 |

**示例：**

考虑新的目录布局。

```py
application \
myapp.py
data \
design1.css
some.js
feeds \
app.rss
app.atom

```

通过静态目录工具处理该结构将类似于：

```py
import cherrypy
class MyApp:
@cherrypy.expose
def index(self):
return """<html>
<head>
<title>My application</title>
<link rel="stylesheet" href="static/css/design1.css"
type="text/css"></link>
<script type="application/javascript"
src="img/some.js"></script>
</head>
<html>
<body>
<a href="feed/app.rss">RSS 2.0 feed</a>
<a href="feed/app.atom">Atom 1.0 feed</a>
</body>
</html>"""
static resource servingdirectory, Staticdir tool usedif __name__ == '__main__':
import os.path
current_dir = os.path.dirname(os.path.abspath(__file__))
cherrypy.config.update({'environment': 'production',
'log.screen': True})
conf = {'/': {'tools.staticdir.root': current_dir},
'/static/css': {'tools.gzip.on': True,
'tools.gzip.mime_types':['text/css'],
'tools.staticdir.on': True,
'tools.staticdir.dir': 'data'},
'/static/scripts': {'tools.gzip.on': True,
'tools.gzip.mime_types':
['application/javascript'],
'tools.staticdir.on': True,
'tools.staticdir.dir': 'data'},
'/feed': {'tools.staticdir.on': True,
'tools.staticdir.dir': 'feeds',
'tools.staticdir.content_types':
{'rss':'application/xml',
'atom': 'application/atom+xml'}}}
cherrypy.quickstart(MyApp(), '/', config=conf)

```

在这个示例中，你会注意到 CSS 和 JavaScript 文件的 URI 路径与其物理对应物完全匹配。同时仔细看看我们是如何根据文件扩展名定义资源的适当 `Content-Type` 的。当 CherryPy 无法自行确定要使用的正确 MIME 类型时，这很有用。最后，看看我们是如何将静态目录工具与 `gzip` 工具混合使用，以便在服务之前压缩我们的静态内容。

### 注意

你可能会觉得 CherryPy 需要绝对路径来与不同的静态工具一起工作有些限制。但考虑到 CherryPy 无法控制应用程序的部署方式和它将驻留的位置。因此，提供这些信息的责任在于部署者。然而，请记住，绝对路径可以通过 `root` 属性或直接在 `filename` 或 `dir` 中提供。

## Bypassing Static Tools to Serve Static Content

有时你可能想重用 CherryPy 的内部功能来服务内容，但又不直接使用静态工具。这可以通过从你的页面处理程序中调用 `serve_file` 函数来实现。实际上，这个函数也是由内置工具调用的。考虑以下示例：

```py
import os.path
import cherrypy
from cherrypy.lib.static import serve_file
class Root:
@cherrypy.expose
def feed(self, name):
accepts = cherrypy.request.headers.elements('Accept')
for accept in accepts:
if accept.value == 'application/atom+xml':
return serve_file(os.path.join(current_dir, 'feeds',
'%s.atom' % name),
content_type='application/atom+xml')
# Not Atom accepted? Well then send RSS instead...
return serve_file(os.path.join(current_dir, 'feeds',
'%s.rss' % name),
content_type='application/xml')
if __name__ == '__main__':
current_dir = os.path.dirname(os.path.abspath(__file__))
cherrypy.config.update({'environment': 'production',
'log.screen': True})
cherrypy.quickstart(Root(), '/')

```

在这里，我们定义了一个页面处理程序，当被调用时，将检查用户代理首选的源内容表示形式——可能是 RSS 或 Atom。

# WSGI 支持

**Web 服务器网关接口**（**WSGI**）由 Phillip J. Eby 编写的**Python 增强提案**（**PEP-333**）定义，旨在在 Web 服务器和 Web 应用程序之间提供一个松散耦合的桥梁。

WSGI 定义了以下三个组件：

+   服务器或网关

+   中间件

+   应用程序或框架

下图显示了 WSGI 及其层：

![WSGI 支持](img/1848_04_02.jpg)

WSGI 的目标是允许组件能够以尽可能少的 API 开销随意插入和运行。这允许代码重用常见的功能，如会话、身份验证、URL 分发、记录等。事实上，由于 API 最小化和不干扰，支持 WSGI 规范的框架或库将能够处理这些组件。

直到 CherryPy 3.0，由于 CherryPy 的内部设计和认为 WSGI 不一定能提高产品的质量，CherryPy 对 WSGI 的支持并不受欢迎。当 Robert Brewer 承担项目的重构工作时，他基于 Christian Wyglendowski 所做的工作改进了 WSGI 支持，使其成为 CherryPy 中的第一公民，并因此满足了社区的需求。

### 注意

注意，CherryPy 工具和 WSGI 中间件在设计上不同，但在功能上没有区别。它们旨在以不同的方式提供相同的功能。CherryPy 工具主要在 CherryPy 中有意义，因此在该环境中进行了优化。CherryPy 工具和 WSGI 中间件可以在单个应用程序中共存。

## **在 CherryPy WSGI 服务器中托管 WSGI 应用程序**

让我们看看如何在 WSGI 环境中使用 CherryPy 的例子：

```py
import cherrypy
from paste.translogger import TransLogger
WSGIWSGI application, hostingdef application(environ, start_response):
status = '200 OK'
response_headers = [('Content-type', 'text/plain')]
start_response(status, response_headers)
return ['Hello world!\n']
if __name__ == '__main__':
cherrypy.tree.graft(TransLogger(application), script_name='/')
cherrypy.server.quickstart()
cherrypy.engine.start()

```

让我们解释一下我们做了什么：

1.  1. 首先，我们创建一个遵守 WSGI 规范的 WSGI 应用程序，因此是一个遵守 WSGI 应用程序签名的 Python 可调用对象。`environ`参数包含从服务器到应用程序处理过程中正交传播的值。中间件可以通过添加新值或转换现有值来修改此字典。`start_response`参数是由外部层（一个中间件或最终是 WSGI 服务器）提供的 Python 可调用对象，用于执行响应处理。然后，我们的 WSGI 应用程序返回一个可迭代对象，它将被外部层消费。

1.  2. 然后，我们将应用程序封装到 paste 包提供的中间件中。Paste 是由 Ian Bicking 创建和维护的一套常见的 WSGI 中间件。在我们的例子中，我们使用`TransLogger`中间件来启用对传入请求的记录。WSGI 定义了中间件，使其能够像服务器一样封装 WSGI 应用程序，并作为托管 WSGI 服务器的应用程序。

1.  3. 最后，我们通过`cherrypy.tree.graft()`方法将 WSGI 应用程序嫁接到 CherryPy 树中，并启动 CherryPy 服务器和引擎。

由于内置的 CherryPy 服务器是一个 WSGI 服务器，它可以无障碍地处理 WSGI 应用程序。然而，请注意，CherryPy 的许多方面，如工具和配置设置，将不会应用于托管 WSGI 应用程序。你需要使用中间件来执行如`paste.transLogger`之类的操作。或者，你可以像以下这样使用`wsgiapp`工具：

```py
import cherrypy
from paste.translogger import TransLogger
def application(environ, start_response):
status = '200 OK'
response_headers = [('Content-type', 'text/plain')]
start_response(status, response_headers)
return ['Hello world!\n']
class Root:
pass
if __name__ == '__main__':
app = TransLogger(application)
conf = {'/': {'tools.wsgiapp.on': True,
'tools.wsgiapp.app': app,
'tools.gzip.on': True}}
cherrypy.tree.mount(Root(), '/', config=conf)
cherrypy.server.quickstart()
cherrypy.engine.start()

```

在这个例子中，我们使用`wsgiapp`工具封装 WSGI 应用程序。请注意，我们可以像对待常规页面处理器一样对 WSGI 应用程序应用工具。

## **在第三方 WSGI 服务器中托管 CherryPy WSGI 应用程序**

在这个例子中，我们将像传统那样编写 CherryPy 应用程序，并在一个不同于内置的 WSGI 服务器中托管它。实际上，我们将使用`wsgiref`包提供的默认 WSGI 服务器。

### 注意

`wsgiref`包是一组 WSGI 辅助工具，自 Python 2.5 起已成为 Python 标准库的一部分。否则，你可以通过`easy_install wsgiref`来获取它。

```py
import cherrypy
from cherrypy import tools
from wsgiref.simple_server import make_server
from flup.middleware.gzip import GzipMiddleware
class Root:
@cherrypy.expose
@tools.response_headers(headers=[('Content-Language', 'en-GB')])
def index(self):
return "Hello world!"
if __name__ == '__main__':
wsgi_app = cherrypy.Application(Root(), script_name="/")
cherrypy.engine.start(blocking=False)
httpd = make_server('localhost', 8080, GzipMiddleware(wsgi_app))
print "HTTP Serving HTTP on http://localhost:8080/"
httpd.serve_forever()

```

让我们解释这个例子：

1.  1. 首先，我们创建一个常规的 CherryPy 应用程序。注意我们在这个上下文中仍然可以安全地使用 CherryPy 工具。

1.  2. 然后，我们通过`cherrypy.Application`辅助工具从它创建一个 WSGI 应用程序。这返回一个由 CherryPy 应用程序组成的 WSGI 有效的可调用对象。

1.  3. 接下来，我们以非阻塞模式启动 CherryPy 引擎，因为我们仍然需要 CherryPy 来处理请求并将请求调度到正确的页面处理器。

1.  4. 然后，我们创建一个 WSGI 服务器实例，托管我们的 WSGI 应用程序，该应用程序被 gzip 中间件封装，该中间件压缩响应体。这个中间件由`flup`包提供，它是另一个 WSGI 中间件集。 (Flup 由 Allan Saddi 维护。)

总结来说，CherryPy 3 对 WSGI 的支持水平非常出色，同时足够灵活，以便在需要时你可以使用两种设计中的最佳方案。CherryPy 可以被视为一个全面且一致的 WSGI 实现。此外，CherryPy 拥有目前最全面和最快的 WSGI 服务器，如果你需要 WSGI 支持，没有理由相信你应该放弃这个库。你可以在[`wsgi.org`](http://wsgi.org)获取更多关于 WSGI 的信息。

# 概述

在本章中，我们回顾了 CherryPy 库的关键点，希望这能打开你的思路，了解如何充分利用其功能。虽然 CherryPy 是一个小型的包，但它提供了一套扩展且一致的特性集，旨在使你的生活更轻松。然而，CherryPy 的一些方面超出了本书的范围，获取更详细信息的最佳方式是访问用户和开发者公开邮件列表。

现在你已经对库有了良好的背景知识，我们将继续通过开发一个简单的照片博客应用程序来使用它。
