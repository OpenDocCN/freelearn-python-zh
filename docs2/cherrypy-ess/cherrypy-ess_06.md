# 第六章. 网络服务

在第五章中，我们定义了数据访问层和我们的应用程序将要操作的对象。在本章中，我们将解释如何通过使用网络服务作为 API 来访问和操作我们定义的对象，从而阐述我们的照片博客应用程序。我们将介绍基于 REST 原则、Atom 发布协议的网络服务概念，并解释如何使用 CherryPy 实现它们。到本章结束时，你应该了解网络服务如何增强和扩展你的网络应用程序的能力，同时为第三方应用程序提供一个简单的入口点。

# 传统网络开发

大多数网络应用程序使用相同的基 URI 来处理资源的提供和资源的操作。例如，以下内容很常见：

| URI | 请求体 | HTTP 方法 | 操作 |
| --- | --- | --- | --- |
| `/album/` | N/A | GET | 获取所有专辑 |
| `/album/?id=12` | N/A | GET | 获取 ID 为 12 的专辑 |
| `/album/edit?id=12` | N/A | GET | 返回一个表单以对资源执行操作 |
| `/album/create` | title=Friends | POST | 创建专辑 |
| `/album/delete` | id=12 | POST | 删除 ID 为 12 的专辑 |
| `/album/update` | id=12&title=Family | POST | 更新 ID 为 12 的专辑 |

在 CherryPy 托管的应用程序中，这可以翻译为：

```py
class Album:
@cherrypy.expose
def index(self, id=None):
# returns all albums as HTML or the one
# requested by the id parameter if provided
@cherrypy.expose
def edit(self, id=None):
# returns an HTML page with a form to perform
# an action on a resource (create, update, delete)
@cherrypy.expose
def create(self, title):
# create an album with a title
# returns an HTML page stating the success
@cherrypy.expose
def update(self, id, title):
# update an album with a title
# returns an HTML page stating the success
@cherrypy.expose
def delete(self, id):
# delete the album with the given id
# returns an HTML page stating the success

```

虽然这种方法是有效的，但当需要向不同类型的用户代理（浏览器、机器人、服务等）开放时，它并不是最佳选择。例如，假设我们决定提供一个肥客户端应用程序来操作专辑。在这种情况下，页面处理器返回的 HTML 页面将毫无用处；XML 或 JSON 数据将更相关。我们可能还希望将我们应用程序的一部分作为服务提供给第三方应用程序。

一个显著的例子是 flickr 提供的服务，([`www.flickr.com/`](http://www.flickr.com/))一个在线照片管理应用程序，它允许用户在许多上下文中查询 flickr 服务（[`www.flickr.com/services/api/`](http://www.flickr.com/services/api/)），如获取当前照片、活动、博客文章、评论等，以不同的格式。多亏了这些网络服务，大量第三方应用程序得以扩展，从而从网络应用程序或甚至从肥客户端应用程序中扩展 flickr 用户的体验。

## 关注点分离

之前的设计示例的问题在于缺乏**关注点分离**。正如 Tim Bray 关于网络的看法（请参阅[`www.tbray.org/ongoing/When/200x/2006/03/26/On-REST`](http://www.tbray.org/ongoing/When/200x/2006/03/26/On-REST)以获取更多详细信息）：

*系统中有很多东西，通过 URI 来标识*。

*系统中对资源有两种操作：那些可以改变其状态的和那些不能的*。

从第一个陈述中，我们给任何可以通过系统传递的事物命名；我们称之为资源。资源的例子可以是图片、诗歌、篮球比赛的结果、澳大利亚的温度等。我们还了解到，每个资源都应该以非歧义的方式被识别。从 Tim 的第二点陈述中，我们意识到在设计上应该逻辑上分离——只读操作和可以更改资源的操作。

这些区分的一个重要推论是我们希望让客户端通知服务器它希望接收的内容类型。在我们的例子中，我们的页面处理程序仅返回 HTML 页面，而检查客户端可以处理的内容并发送资源最佳表示将更加灵活。

网络应用程序开发者应考虑以下原则：

+   任何事物都是资源。

+   资源有一个或多个标识符，但一个标识符只能指向一个资源。

+   资源有一个或多个客户端可以请求的表示形式。

+   资源操作分为改变资源状态和不改变资源状态的那些。

基于这些元素，我们可以重新定义我们的设计如下：

```py
class Album:
@cherrypy.expose
def index(self):
# returns all albums as HTML
@cherrypy.expose
def default(self, id):
# returns the album specified or raise a NotFound
@cherrypy.expose
def edit(self, id=None):
# returns an HTML page with a form to perform
# an action on a resource (create, update, delete)
class AlbumManager:
@cherrypy.expose
def create(self, title):
# create an album with a title
# returns an XML/JSon/XHTML document
# representing the resource
@cherrypy.expose
def update(self, id, title):
# update an album with a title
# returns an XML/JSon/XHTML document
# representing the resource
@cherrypy.expose
def delete(self, id):
# delete the album with the given id
# returns nothing

```

通过这样做，我们允许任何类型的用户代理通过请求公开的`AlbumManager`处理程序来操作资源。浏览器仍然会从`Album`页面处理程序获取专辑的 HTML 表示。你可能会争辩说，浏览器不知道如何处理从`AlbumManager`页面处理程序返回的 XML 或 JSON 数据。这里缺失的信息是，HTML 表单的提交及其响应的处理将由一些客户端脚本代码通过 JavaScript 执行，该代码能够相应地处理 XML 或 JSON 数据块。我们将在第七章（ch07.html "第七章。表示层"）中更详细地介绍这项技术。

上述定义的原则是今天所说的**网络服务**的基础。网络服务是网络应用程序提供的 API，以便异构用户代理可以通过 HTML 以外的格式与应用程序交互。通过 REST、SOAP、XML-RPC、Atom 等方式可以创建不同的网络服务。为了本书的目的，我们将回顾 REST 和 Atom 发布协议作为照片博客应用程序的网络服务。

# REST

**表示性状态转移**（**REST**）是 Roy T. Fielding 在 2000 年他的论文《架构风格和网络软件架构设计》中描述的分布式超媒体系统的架构风格。

REST 基于以下元素：

+   *资源:* 资源是任何事物的抽象概念。例如，它可以是图片、博客条目、两种货币之间的当前汇率、体育结果、数学方程式等。

+   *资源标识符:* 允许分布式系统的组件以独特的方式识别资源。

+   *表示:* 资源的一个表示仅仅是数据。

+   *表示元数据:* 关于表示本身的信息。

+   *资源元数据:* 关于资源的信息。

+   *控制数据:* 系统中组件之间传递的消息信息。

REST 还建议每个流动的消息应该是**无状态的**，这意味着它应该包含足够的信息供系统中的下一个组件处理，因此不应依赖于之前的或后续的消息。每个消息都是自包含的。这是通过使用资源元数据和表示元数据来实现的。

这些是描述 REST 的元素，但它们并不绑定到任何底层协议。最常用的 REST 用例可以在 Web 中找到，并使用 HTTP 协议实现。尽管如此，REST 可以在其他环境和其他协议中使用。

HTTP 是实施 REST 的好候选，以下是一些原因：

+   它是网络的基础，这是一个分布式超媒体系统。

+   它是无状态的。

+   每个请求都可以包含足够的信息，可以独立于系统中的其余部分进行处理。

+   HTTP 使用的`Content-Type`和`Accept`头部提供了通过不同表示形式来表示单个资源的手段。

+   URI 是强大且常见的资源标识符。

# 统一资源标识符

REST 是关于在网络上命名资源并提供对这些资源执行操作的统一机制。这就是为什么 REST 告诉我们资源至少由一个标识符来识别。当基于 HTTP 协议实现 REST 架构时，这些标识符被定义为**统一资源标识符**（**URI**）。

URI 集合的两个常见子集是：

+   **统一资源定位符**（**URL**），例如：[`www.cherrypy.org/`](http://www.cherrypy.org/)

+   **统一资源名称**（**URN**），例如：

    ```py
    urn:isbn:0-201-71088-9
    urn:uuid:13e8cf26-2a25-11db-8693-000ae4ea7d46

    ```

URL 的有趣之处在于它们包含足够的信息来定位网络上的资源。因此，在给定的 URL 中，我们知道要定位资源，我们需要使用与 HTTP 方案关联的 HTTP 协议，该协议托管在主机[www.cherrypy.org](http://www.cherrypy.org)上的路径`/`。（然而，请注意，并非 Web 社区中的每个人都认为这种能力的多路复用是 URL 的积极方面，但这次讨论超出了本书的范围。）

# HTTP 方法

如果 URI 提供了命名资源的方式，HTTP 方法提供了我们可以对这些资源进行操作的手段。让我们回顾 HTTP 1.1 中最常见的方法（也称为动词）。

| HTTP 方法 | 允许幂等 | 操作 |
| --- | --- | --- |
| `HEAD` | 是 | 获取资源元数据。响应与 GET 相同，但无主体。 |
| `GET` | 是 | 获取资源元数据和内容。 |
| `POST` | 否 | 请求服务器使用请求体中的数据创建一个新的资源。 |
| `PUT` | 是 | 请求服务器用请求体中包含的资源替换现有的资源。服务器不能将包含的资源应用于未由该 URI 标识的资源。 |
| `DELETE` | 是 | 请求服务器删除由该 URI 标识的资源。 |
| `OPTIONS` | 是 | 请求服务器返回有关能力的信息，无论是全局的还是特定于资源的。 |

表格的幂等列表示使用该特定 HTTP 方法的请求是否会有与两个连续相同调用相同的副作用。

默认情况下，CherryPy 处理程序反映请求-URI 的路径，处理程序与 URI 的一个元素匹配，但正如我们所看到的，CherryPy 的分发器可以被更改，使其不是在 URI 中查找处理程序，而是从请求元数据（如使用的 HTTP 方法）中查找。

让我们回顾一个应用于照片博客应用程序的例子：

```py
import cherrypy
from cherrypy.lib.cptools import accept
from models import Photoblog, Album
from lib.config import conf
from lib.tools import find_acceptable_within
class AlbumRESTService(object):
exposed = True
def GET(self, album_id):
best = accept(['application/xml', 'application/atom+xml',
'text/json', 'text/x-json'])
album = Album.fetch(album_id)
if not album:
raise cherrypy.NotFound()
if best in ['application/xml','application/atom+xml']:
cherrypy.response.headers['Content-Type'] =
'application/atom+xml'
entry = album.to_atom_entry()
return entry.xml()
if best in ['application/json', 'text/x-json', 'text/json']:
cherrypy.response.headers['Content-Type'] =
'application/json'
return album.to_json()
raise cherrypy.HTTPError(400, 'Bad Request')
def POST(self, title, segment, author, description, content,
blog_id):
photoblog = Photoblog.fetch(blog_id)
if not photoblog:
raise cherrypy.NotFound()
album = Album()
album.create(photoblog, title, segment, author, description,
content)
cherrypy.response.status = '201 Created'
cherrypy.response.headers['Location'] = '%s/album/%d' %
(conf.app.base_url, album.ID)
def PUT(self, album_id, title, segment, author, description,
content):
album = Album.fetch(album_id)
if not album:
raise cherrypy.NotFound()
album.update(title, segment, author, description, content)
def DELETE(self, album_id):
album = Album.fetch(album_id)
if album:
album.delete()
cherrypy.response.status = '204 No Content'

```

让我们解释在这个上下文中每个 HTTP 方法的作用。

+   `GET:` 这返回请求资源的表示形式，取决于`Accept`头。我们的应用程序允许`application/xml, application/atom+xml, text/json`或`text/x-json`。我们使用一个名为`accept`的函数，它返回找到的可接受头或立即引发一个`cherrypy.HTTPError (406, 'Not Acceptable')`错误，通知用户代理我们的应用程序无法处理其请求。然后我们验证资源是否仍然存在；如果不存在，我们引发一个`cherrypy.NotFound`错误，这是`cherrypy.HTTPError(404, 'Not Found')`的快捷方式。一旦我们检查了先决条件，我们就返回资源的请求表示。

    注意，这相当于默认分发器的`index()`方法。但请记住，当使用方法分发器时，没有`default()`方法的等效方法。

+   `POST:` HTTP `POST` 方法允许用户代理创建一个新的资源。第一步是检查将要处理该资源的照片博客是否存在。然后我们创建资源，并返回状态码`201 Created`以及`Location`头，指示检索新创建资源的 URI。

+   `PUT:` HTTP `PUT` 方法允许用户代理用请求体中提供的一个资源替换资源。这通常被认为是一个更新操作。尽管 RFC 2616 没有禁止`PUT`也创建一个新的资源，但我们将不会在我们的应用程序中以这种方式使用它，我们将在后面解释。

+   `DELETE:` `DELETE` 方法请求服务器删除资源。对此方法的响应可以是`200 OK`或`204 No Content`。后者通知用户代理它不应更改其当前状态，因为响应没有主体。

`POST`和`PUT`之间的（缺乏）差异长期以来一直是网络开发者讨论的来源。有些人认为有两个方法是有误导性的。让我们尝试理解它们为什么是不同的，为什么我们需要两者。

`POST` 请求：

```py
POST /album HTTP/1.1
Host: localhost:8080
Content-Length: 77
Content-Type: application/x-www-form-urlencoded
blog_id=1&description=Family&author=sylvain&title=My+family&content=&
segment=

```

`POST` 响应：

```py
HTTP/1.1 201 Created
Content-Length: 0
Location: http://localhost:8080/album/12
Allow: DELETE, GET, HEAD, POST, PUT
Date: Sun, 21 Jan 2007 16:30:43 GMT
Server: CherryPy/3.0.0
Connection: close

```

`PUT`请求：

```py
PUT /album/12 HTTP/1.1
Host: localhost:8080
Content-Length: 69
Content-Type: application/x-www-form-urlencoded
description=Family&author=sylvain&title=Your+family&content=&segment=

```

`PUT`响应：

```py
HTTP/1.1 200 OK
Date: Sun, 21 Jan 2007 16:37:12 GMT
Content-Length: 0
Allow: DELETE, GET, HEAD, POST, PUT
Server: CherryPy/3.0.0
Connection: close

```

初看，两个请求似乎相当相似，但实际上它们有一个非常重要的区别，那就是请求的 URI。

可以将数据`POST`到 URI，其中可能或可能不会创建资源，而在`PUT`的情况下，URI 本身就是资源之一，发送的内容是资源的新表示。在这种情况下，如果资源在该 URI 上尚不存在，服务器可以创建它，如果它已经实现这样做的话；否则，服务器可以返回一个 HTTP 错误消息，表明它没有满足请求。简而言之，客户端将数据`POST`到进程，但将请求 URI 标识的资源的新表示`PUT`。

问题的一个根本原因是许多 Web 应用程序仅依赖于`POST`方法来实现对资源的任何操作，无论是创建、更新还是删除。这尤其是因为这些应用程序通常只提供 HTML 表单，这些表单只支持`GET`和`POST`来执行这些操作。

考虑到越来越多的 Web 应用程序利用关注点分离并通过 JavaScript 或外部服务通过客户端代码处理提交，`PUT`和`DELETE`方法的使用很可能会增加，尽管在某些环境中可能会成为问题，因为防火墙策略禁止`PUT`和`DELETE`请求。

# 整合

我们的博客应用程序将为以下实体提供 REST 接口：专辑、电影和条目。由于它们携带的信息、它们之间的关系以及它们的设计，我们可以提供与实体本身无关的相同接口。因此，我们重构了`Album`类并创建了一个`Resource`类，该类将集中实现每个操作。每个实体服务接口只需将信息传递给`Resource`类，让它处理繁重的工作。因此，我们避免了代码的重复。

```py
import cherrypy
from cherrypy.lib.cptools import accept
from models import Photoblog
from lib import conf
from lib.tools import find_acceptable_within
class Resource(object):
def handle_GET(self, obj_id):
best = accept(['application/xml', 'application/atom+xml',
'text/json', 'text/x-json',
'application/json'])
obj = self.__source_class.fetch(obj_id)
if not obj:
raise cherrypy.NotFound()
if best in ['application/xml', 'application/atom+xml']:
cherrypy.response.headers['Content-Type'] = 'application/atom+xml'
entry = obj.to_atom_entry()
return entry.xml()
if best in ['text/json', 'text/x-json', 'application/json']:
cherrypy.response.headers['Content-Type'] =
'application/json'
return obj.to_json()
raise cherrypy.HTTPError(400, 'Bad Request')
def handle_POST(self container_cls, container_id,
location_scheme, *args, **kwargs):
container = container_cls.fetch(container_id)
if not container:
raise cherrypy.NotFound()
obj = self.__source_class()
obj.create(container, *args, **kwargs)
cherrypy.response.status = '201 Created'
cherrypy.response.headers['Location'] = location_scheme %
(conf.app.base_url, obj.ID)
def handle_PUT(cls, source_cls, obj_id, *args, **kwargs):
obj = self.__source_class.fetch(obj_id)
if not obj:
raise cherrypy.NotFound()
obj.update(obj, *args, **kwargs)
def handle_DELETE(cls, source_cls, obj_id):
obj = self.__source_class.fetch(obj_id)
if obj:
obj.delete(obj)
cherrypy.response.status = '204 No Content'

```

然后，让我们重新定义我们的`AlbumRESTService`类以利用`Resource`类：

```py
from models import Photoblog, Album
from _resource import Resource
class AlbumRESTService(Resource):
exposed = True
# The entity class that will be used by the Resource class
_source_class = Album
def GET(self, album_id):
return self.handle_GET(album_id)
def POST(self, title, segment, author, description, content,
blog_id):
self.handle_POST(Photoblog, blog_id, '%s/album/%d',
title, segment, author, description,content)
def PUT(self, album_id, title, segment, author, description,
content):
self.handle_PUT(album_id,
title, segment, author, description, content)
def DELETE(self, album_id):
self.handle_DELETE(album_id)

```

我们现在有一个将处理专辑资源的 RESTful 接口。电影和照片实体将以相同的方式进行管理。这意味着我们的应用程序现在将支持以下请求：

```py
POST http://somehost.net/service/rest/album/
GET http://somehost.net/service/rest/album/12
PUT http://somehost.net/service/rest/album/12
DELETE http://somehost.net/service/rest/album/12

```

在这些调用中的每一个，URI 都是资源的唯一标识符或名称，而 HTTP 方法是执行在该资源上的操作。

# 通过 CherryPy 的 REST 接口

到目前为止，我们已经描述了我们的照片博客应用程序将支持的服务，但没有详细说明如何通过 CherryPy 实现。

正如我们在前面的章节中看到的，HTTP REST 依赖于 HTTP 方法来通知 Web 应用用户代理希望执行的操作类型。为了通过 CherryPy 实现我们的照片博客应用中的 REST，我们将使用 HTTP 方法分发器，正如在第四章中回顾的那样，来处理对上述服务类的传入请求，大致如下：

```py
rest_service = Service()
rest_service.album = AlbumRESTService()
conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
cherrypy.tree.mount(rest_service, '/service/rest', config=conf)

```

这意味着针对 URI 路径如`/service/rest/album/`的请求将在 REST 精神下执行。

REST 是一个相当常见的术语，但构建真正的 RESTful 应用可能是一项艰巨的任务。困难在于定义与应用资源相关的一个合理且有意义的服务 URI 集。换句话说，困难的部分在于 API 的设计。本节应该已经向您介绍了 REST 背后的原则，但围绕 REST 开发大型系统的架构需要高度理解所处理资源、它们的命名约定以及它们之间的关系。

# Atom 发布协议

在前面的章节中，我们介绍了 REST 并展示了它如何作为 Web 应用的服务使用。在本节中，我们将介绍**Atom 发布协议**（**APP**），在撰写本书时，它正在成为新的 IETF 标准。这意味着本节的一些方面可能在您阅读时可能已经不再是最新的。

APP 作为一种基于 HTTP 的应用层协议，起源于 Atom 社区，允许发布和编辑 Web 资源。APP 服务器和客户端之间的消息单元基于 RFC 4287 中定义的 Atom XML 文档格式。

虽然 APP 没有指定为 REST 原则的实现，但该协议遵循相同的思想，使其具有 RESTful 特性。因此，前一部分的许多原则也适用于这里；但首先让我们概述一下 Atom XML 文档格式。

# Atom XML 文档格式

Atom XML 文档格式通过两个顶级元素描述了一组信息：

+   源：一个源由以下内容组成：

    +   元数据（有时被称为源的头）

    +   零个或多个条目

+   条目：一个条目由以下内容组成：

    +   元数据

    +   一些内容

以下是一个符合 RFC4287 的 Atom 1.0 源文档示例：

```py
<?xml version="1.0" encoding="utf-8"?>
<feed >
<title>Photoblog feed</title>
<published>2006-08-13T10:57:18Z</published>
<updated>2006-08-13T11:18:01Z</updated>
<link rel="self" href="http://host/blog/feed/album/" type="application/atom+xml" />
<author>
<name>Sylvain Hellegouarch</name>
</author>
<id>urn:uuid:13e8cf26-2a25-11db-8693-000ae4ea7d46</id>
<entry>
<title>This is my family album</title>
<id>urn:uuid:25cd2014-2ab3-11db-902d-000ae4ea7d46</id>
<link rel="self" href="http://host/blog/feed/album/12"
type="application/atom+xml" />
<link rel="alternate" href="http://host/blog/album/12"
type="text/html" />
<updated>2006-08-13T11:18:01Z</updated>
<content type="text">Some content</content>
</entry>
</feed>

```

Web 应用可以为订阅提供 Atom 文档，从而为用户代理提供一种将自己同步到应用开发者选择提供的信息的方式。

我们的摄影博客应用将提供以下实体的 Atom 源：

+   照片博客：每个博客的条目将链接到博客的相册条目。

+   相册：每个博客的条目将链接到相册的电影条目。

+   电影：每个条目将关联到一张电影照片。

我们不会解释 Atom 文档的每个元素，但会回顾一些最常见的元素。

+   `id`、`title`和`updated`是任何源或条目中的强制元素。

    +   `id`必须是 RFC 3987 中定义的 IRI，作为 URI 的补充

    +   `updated`必须遵循 RFC 3339。RFC 4287 表示，该元素只有在修改具有语义意义时才需要更新。

+   `author`在 Atom 源中是强制性的，无论是在`feed`元素、`entry`元素还是在两者中。然而，如果条目没有提供，则条目可以继承源`author`元素。

+   `link`不是强制的，但推荐使用，并且非常有用，可以提供以下信息：

    +   使用`rel="self"`指定与条目或源关联的资源 URI

    +   使用`rel="alternate"`指定资源替代表示的 URI，并指定资源的媒体类型

    +   使用`rel="related"`来指定相关资源的 URI

+   `content`最多只能出现一次。一个条目的内容可以是内联在条目中的文本、转义 HTML 或 XHTML，或者通过`src`属性引用，提供实际内容的 URI。

因此，对于电影源，我们将有：

```py
<?xml version="1.0" encoding="UTF-8"?>
<feed >
<id>urn:uuid:8ed4ae87-2ac9-11db-b2c4-000ae4ea7d46</id>
<title>Film of my holiday</title>
<updated>2006-08-13T13:50:49Z</updated>
<author>
<name>Sylvain Hellegouarch</name>
</author>
<entry>
APPAtom XML-document<id>urn:uuid:41548439-c12d-48b5-baec-a72b1bf8576f</id>
<published>2006-08-13T13:45:38Z</published>
<updated>2006-08-13T13:50:49Z</updated>
<title>At the beach</title>
<link rel="self" href="http://host/feed/photo/at-the-beach"
type="application/atom+xml"/>
<link rel="alternate" href="http://host/photo/at-the-beach"
type="text/html" />
<content src="img/IMAGE001.png"
type="image/png" />
</entry>
</feed>

```

Atom 格式在博客环境中被广泛使用，以允许用户订阅它。然而，由于其灵活性和可扩展性，Atom 格式现在被用于不同的环境中，如发布、存档和导出内容。

# APP 实现

在照片博客应用程序中提供**Atom 发布协议**（APP）实现的目的是介绍该协议，并提供两个不同的服务，以展示关注点分离的好处。由于 APP 尚未成为标准，并且由于在撰写本书时它正处于相当多的讨论中，因此有可能在我们阅读本节时，我们的实现可能不再符合标准。然而，风险最小，因为当前协议草案的版本，即 13，就其主要特性而言似乎足够稳定。

Atom 发布协议定义了一组操作，这些操作通过 HTTP 及其机制以及 Atom XML 文档格式作为消息单元，在 APP 服务和用户代理之间进行。

APP 首先定义一个服务文档，它为用户代理提供 APP 服务所提供的不同集合的 URI。其形式为：

```py
<?xml version="1.0" encoding="UTF-8"?>
<service  xmlns:atom=
"http://www.w3.org/2005/Atom">
<workspace>
<collection href="http://host/service/atompub/album/">
<atom:title>Friends Albums</atom:title>
<categories fixed="yes">
<atom:category term="friends" />
</categories>
</collection>
<collection href="http://host/service/atompub/film/">
<atom:title>Films</atom:title>
<accept>image/png,image/jpeg</accept>
</collection>
</workspace>
</service>

```

一旦用户代理获取了该服务文档，它就知道有两个`集合`可用。第一个`集合`通知用户代理它将只接受具有与定义匹配类别的 Atom 文档。第二个`集合`将只接受`image/png`或`image/jpeg`MIME 类型的数据。

**集合**是 APP 所指成员的容器。创建成员的操作是在集合上完成的，而检索、更新和删除操作是在该成员本身上完成的，而不是在集合上。

集合表示为 Atom 源，其中条目被称为成员。对 Atom 条目的关键补充是使用具有`rel`属性设置为`edit`的 Atom 链接来描述成员资源。通过将此属性设置为该值，我们表明链接元素的`href`属性引用的是可以检索、编辑和删除的成员资源的 URL。包含此类链接元素的 Atom 条目称为集合的**成员**。

APP 通过以下表格中描述的 HTTP 方法指定如何对集合的成员或集合本身执行基本的 CRUD 操作。

| 操作 | HTTP 方法 | 状态码 | 返回内容 |
| --- | --- | --- | --- |
| 获取 | `GET` | `200` | 代表资源的 Atom 条目 |
| 创建 | `POST` | `201` | 代表资源的 Atom 条目，通过 Location 和 Content-Location 头部的 URI 表示新创建的资源 |
| 更新 | `PUT` | `200` | 代表资源的 Atom 条目 |
| 删除 | `DELETE` | `200` | 无 |

在创建或更新资源时，APP 服务器可以自由修改资源的一部分，例如其`id`、其`updated`值等。因此，用户代理不应依赖于其资源版本，而应始终与服务器同步。

尽管集合的成员是 Atom 条目，但不必通过提交 Atom 条目来创建新成员。APP 支持任何媒体类型，只要它通过`app:collection`元素的`app:accept`元素允许即可。该元素接受一个以逗号分隔的媒体类型列表，指定客户端集合将处理 POST 请求的内容类型。

如果您将 PNG 图像`POST`到接受它的集合，服务器将创建至少两个资源。

+   成员资源，可以看作是图像的元数据

+   媒体资源

记住，APP 服务器对发送的内容拥有完全控制权，因此可以想象 APP 服务器在存储之前将 PNG 内容转换为 JPEG。客户端不能假设发送的内容或资源会被复制，就像服务器所做的那样。在任何情况下，服务器在创建成功时返回成员资源（请参阅 APP 规范以获取详细示例），这正是 APP 如此强大的原因，因为无论服务器声称处理哪种类型的资源，APP 都确保会以 Atom 条目的形式生成元数据。

除了定义一个接口来操作集合内的成员外，APP 还提供了当集合变得过大时的分页支持。这允许用户代理请求集合中给定范围的成员。我们不会解释此功能，但如果您对此功能感兴趣，可以查看 APP 规范。

此外，由于照片博客应用将尽可能遵循 REST 原则来实现 APP，我们邀请您参考 REST 部分，以获取关于 APP 如何使用 REST 原则的更具体细节。

在本节中，我们简要介绍了原子发布协议（Atom Publishing Protocol），这是一种基于 Atom XML 文档格式的协议，允许发布异构数据类型。尽管它还不是官方标准，但 APP 已经引起了许多组织的兴趣，并且很可能你会在越来越多的应用中找到它。

# 摘要

本章向您介绍了网络服务（web services）的概念，它定义了通过常见的网络协议（如 HTTP）提供 API 的想法。通过提供这样的 API，您的网络应用变得更加灵活、强大和可扩展。尽管网络服务不是必需的功能，并不是每个网络应用都会提供它们。我们的照片博客应用，在其展示一些常见现代网络技术的精神下，将它们用作示例而不是强制性的功能。然而，通过审查我们的照片博客应用的代码，您将了解网络服务的一些有趣的好处，这可能会为您自己的应用提供灵感。
