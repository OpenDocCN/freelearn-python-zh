# 构建 RESTful API

**表征状态转移**（**REST**）是一种用于实现 Web 服务的架构风格。它由 Roy Fielding 在 2000 年的博士论文中定义。REST 旨在实现系统之间统一和预定义操作的标准。这些系统可以是客户端浏览器、移动应用程序、运行并行工作进程的服务器——你名之。通过使用 HTTP 方法，REST 是平台和编程语言无关的，并且解耦客户端和服务器，以便更容易地进行开发。这通常用于需要从服务器拉取或更新用户信息的 Web**单页应用程序**（**SPAs**）。REST 还用于为外部开发者提供一个通用的接口来访问用户数据。例如，Facebook 和 Twitter 在其应用程序程序接口（或 API）中使用 REST。

你可以在[`www.ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm`](https://www.ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm)查看 Roy Fielding 关于 REST 的原始论文。

在本章中，你将学习以下主题：

+   HTTP 协议：请求、响应、方法、头部和 URI 格式

+   如何构建 REST 服务

+   如何使用 JWT 保护 REST 服务

# 什么是 REST？

在深入了解 REST 之前，由于它是一种系统间通信的风格，让我们首先快速了解一下它所使用的实际协议，这本书的整个基础。

# HTTP

**超文本传输协议**（**HTTP**）是一种属于第 7 层（应用层）的请求-响应协议。这一层与应用程序本身进行交互。属于第 7 层的其他协议还包括**简单邮件传输协议**（**SMTP**）、**网络文件系统**（**NFS**）、**文件传输协议**（**FTP**）等。

HTTP 被设计为供客户端（用户代理）从服务器请求资源使用。这些资源可以是 HTML 文件或任何其他内容，如 JSON、XML 或媒体文件。这些资源请求由网络使用**统一资源定位符**（**URLs**）标识。

URL 是一种特定的 URI 类型，由以下元素组成：

```py
<scheme>://<authority>/<path>/<query><fragment>
```

之前的 `<authority>` 部分：

```py
<userinfo>@<host>:<port>
```

以下是我们应用程序的一个示例 URL：

```py
http://someserver.com:5000/blog/user/user1?title=sometitle#1
```

让我们分别列出这个的元素：

| **方案** | **HTTP** |
| --- | --- |
| `authority.host` | `someserver.com` |
| `authority.port` | `5000` |
| `path` | `blog/user/user1` |
| `query` | `title=sometitle` |
| `fragment` | `1` |

接下来，我们将快速查看一个用户代理向服务器发送的 HTTP 请求消息。这是一个来自 Mozilla 浏览器的 GET 请求，如下面的代码中高亮显示的文本所示：

```py
GET /blog/user/user1 HTTP/1.1
Host: someserver.com
Accept: image/gif, image/jpeg, */*
Accept-Language: en-us
Accept-Encoding: gzip, deflate
User-Agent: Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)
Cookie: cookie1=somevalue; cookie2=othervalue; session:dsofksdfok439349i3sdkfoskfoskfosdkfo
(blank line)
```

因此，一个 HTTP 请求由以下部分组成：

+   **请求行**：进一步由`<Request method> <Request URI> <HTTP version>`组成

+   **请求头**: 包含客户端接受的信息、用户代理、cookies 以及甚至基本的认证凭证

+   **一个空白行**: 将头部与主体部分分开

+   **请求体**: 可选

接受的 HTTP 请求方法有`GET`、`HEAD`、`POST`、`PUT`、`DELETE`、`CONNECT`、`OPTIONS`、`TRACE`和`PATCH`。REST 规范将使用它们来识别应用程序类型操作。

HTTP 响应请求看起来如下：

```py
HTTP/1.0 200 OK
Content-Type: application/json
Content-Length: 1330
Server: Werkzeug/0.14.1 Python/2.7.10
Date: Thu, 19 Jul 2018 11:14:16 GMT
{ "author": "user1" ... }
```

它由以下元素组成：

+   **状态行**: 响应的状态

+   **响应头**: 包含有关内容类型、长度、服务器类型（在我们的例子中，是 Flask 的开发服务器本身）、日期以及是否可以发送 set-cookie 操作的信息

+   一个空白行

+   **响应体**: 在我们的例子中，这是一个 JSON 响应，可能是 REST 服务响应

状态响应代码对 REST 也非常重要。它们分为以下类别：

+   **信息性**: 1XX

+   **成功**: 2XX

+   **重定向**: 3XX

+   **客户端错误**: 4XX

+   **服务器错误**: 5XX

有关状态响应代码的更多详细信息，请参阅 RFC2616 在[`www.w3.org/Protocols/rfc2616/rfc2616-sec10.html`](https://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html)。

# REST 定义和最佳实践

在深入了解 REST 之前，让我们看看一个例子。有一个客户端——在这种情况下，是一个网页浏览器——和一个服务器，客户端通过 HTTP 向服务器发送请求以获取一些模型，如下所示：

![图片](img/dec57efd-0837-4883-bd2d-9971b1a10913.png)

服务器将随后响应一个包含所有模型的文档，如下所示：

![图片](img/6ea3f8b0-ab4c-4648-a50b-5b69e62d2552.png)

客户端可以通过以下`PUT` HTTP 请求修改服务器上的数据：

![图片](img/f8282865-f224-433a-b137-67f923f75a0a.png)

然后，服务器将响应它已修改数据。这是一个非常简化的例子，但它将作为 REST 定义的背景。

与严格的规范不同，REST 定义了一套对通信的约束，以定义一种可以以多种方式实施的方法。这些约束源于与其他通信协议（如远程过程调用**RPC**或简单对象访问协议**SOAP**）多年来的试验和错误。这些协议因为其严格性、冗长性以及难以用于创建 API 的事实而被废弃。这些问题被识别出来，REST 的约束被创建出来，以防止这些问题再次发生。

REST 提供了以下指导性约束：

+   **客户端和服务器之间的关注点分离**: 只要 API 不改变，客户端和服务器应该能够独立地演变或改变。

+   **无状态：**处理请求所需的所有必要信息都存储在请求本身或客户端中。服务器无状态的例子是 Flask 中的`session`对象。`session`对象不在服务器上存储其信息，而是在客户端的 cookie 中存储。cookie 与每个请求一起发送，以便服务器解析并确定是否在 cookie 中存储了请求资源的必要数据，而不是服务器为每个用户存储会话信息。 |

+   **统一接口：**这个约束有许多不同的部分，如下所述： |

    +   接口是基于资源构建的，在我们的案例中是模型。 |

    +   服务器发送的数据不是服务器中的实际数据，而是一种表示。例如，每个请求都发送数据的 JSON 抽象，而不是实际的数据库。 |

    +   服务器发送的数据足以允许客户端修改服务器上的数据。在先前的示例中，传递给客户端的 ID 扮演了这一角色。 |

    +   API 提供的每个资源都必须以相同的方式进行表示和访问。例如，一个资源不能以 XML 表示，而另一个以 JSON 表示。 |

+   **分层系统：**负载均衡器、代理、缓存以及其他服务器和服务可以在客户端和服务器之间执行，只要最终结果与它们不存在时相同。这提高了性能、可扩展性和可用性。 |

+   **缓存性：**客户端可以缓存响应，因此服务器必须定义响应是否可缓存。这可以提高性能。 |

当一个系统遵守所有这些约束时，它被认为是 RESTful 系统。最常见的形式是由 HTTP 和 JSON 构建的。每个资源都位于自己的 URL 路径上，并使用不同的 HTTP 请求方法进行修改。通常，这具有以下形式： |

| **HTTP 方法** | **URL** | **操作** |
| --- | --- | --- |
| `GET` | `http://host/resource` | 获取所有资源表示 |
| `GET` | `http://host/resource/1` | 通过 ID 为`1`获取资源 |
| `POST` | `http://host/resource` | 从`POST`请求中的表单数据创建新的资源 |
| `PUT` | `http://host/resource/1` | 修改 ID 为`1`的资源现有数据 |
| `DELETE` | `http://host/resource/1` | 删除 ID 为`1`的资源 |

例如，对第二个`GET`请求的响应可能如下所示： |

```py
HTTP/1.0 200 OK
Content-Type: application/json
Content-Length: 1330
Server: Werkzeug/0.14.1 Python/2.7.10
Date: Thu, 19 Jul 2018 11:14:16 GMT

{ "id": 100, "title": "Some blog post title" }
```

在 RESTful API 中，返回正确的 HTTP 状态代码与响应数据一起，也非常重要，以便通知客户端服务器上实际发生了什么，而无需客户端解析返回的消息。以下是 RESTful API 中使用的 HTTP 代码列表及其含义： |

| **HTTP 代码** | **名称** | **含义** |
| --- | --- | --- |
| `200` | OK | HTTP 的默认代码。请求成功，并返回了数据。 |
| `201` | 已创建 | 请求成功，并在服务器上创建了一个新资源。 |
| `204` | 无内容 | 请求成功，但响应没有返回内容。 |
| `400` | 错误请求 | 请求被拒绝，因为客户端存在某些感知到的错误——要么是请求格式不正确，要么是缺少所需的数据。 |
| `401` | 未授权 | 请求被拒绝，因为客户端未认证，在再次请求此资源之前应该进行认证。 |
| `403` | 禁止 | 请求被拒绝，因为客户端没有权限访问此资源。这与`401`代码不同，`401`代码假设用户未认证。`403`代码表示无论认证与否，资源都不可访问。 |
| `404` | 未找到 | 请求的资源不存在。 |
| `405` | 方法不允许 | 请求被拒绝，因为 HTTP 方法对于 URL 不可用。 |
| `500` | 内部服务器错误 | 当 Web 服务器遇到意外条件，阻止其满足客户端请求时，会以这个状态码响应。 |
| `501` | 未实现 | 当不支持处理请求所需的功能时显示此错误。当服务器不识别请求方法时，这是适当的响应。 |
| `502` | 网关错误 | 当服务器作为网关或代理并从上游服务器收到无效响应时。 |
| `503` | 服务不可用 | 目前无法处理请求，因为服务器暂时过载或维护。 |
| `504` | 网关超时 | 未从上游服务器及时收到响应。 |

# 设置 RESTful Flask API

在我们的应用程序中，我们将创建一个 RESTful 接口来访问我们数据库中的博客文章数据。数据表示将作为 JSON 发送。数据将通过使用前面表格中的通用形式检索和修改，但 URI 将是`/api/posts`。

如果您还没有下载并访问本章提供的示例代码，也没有查看 API 的 Flask URL 映射，那么在应用程序的根目录中可以找到一个简单的方法，如下面的代码所示：

```py
$ # Initialise the virtual environment and database with test data
$ ./init.sh
$ # Activate the python virtual environment
$ source venv/bin/activate
$ export FLASK_APP=main.py
$ echo app.url_map | flask shell | grep api ..
 <Rule '/auth/api' (POST, OPTIONS) -> auth.api>,
 <Rule '/api/post' (HEAD, GET, PUT, POST, OPTIONS, DELETE) -> postapi>,
 <Rule '/api/post/<post_id>' (HEAD, GET, PUT, POST, OPTIONS, DELETE) 
 -> postapi>,
```

我们将为 API 实现一个认证端点以及创建博客文章 CRUD API 所需的端点。

我们可以使用标准的 Flask 视图来创建 API，但 Flask 扩展**Flask Restful**可以使任务变得更加简单，并将帮助我们遵守完整的 REST 兼容性（RESTful）。

要将此新依赖项包含到我们的应用程序中，您可以在`requirements.txt`文件中找到以下内容：

```py
... 
Flask-Restful ...
```

我们将创建一个新的模块用于 API。应用程序结构如下面的代码所示：

```py
./
  main.py
  config.py
  ...
  webapp/
    blog/
    main/
    auth/
    api/
 __init__.py      blog/
 controlers.py
          fields.py
 parsers.py
    templates/
    static/
```

再次强调，我们的想法是构建一个易于扩展的应用程序。这次，对于每个添加的应用程序模块——比如博客、共享照片，等等——我们将在`api/`模块内部创建一个新的模块，其中定义了所有 API 逻辑。另一种方法可能是将 REST API 包含在每个模块中。

就像所有其他模块一样，`api/__init__.py`中有一个`create_module`函数，它处理主工厂函数`create_app`的初始化。`PostApi`类也将使用`Api`对象的`add_resource()`方法定义其路由。

这可以在提供的代码文件`api/__init__.py`中看到，如下所示：

```py
from flask_restful import Api
from .blog.controllers import PostApi

rest_api = Api()

def create_module(app, **kwargs):
    rest_api.add_resource(
        PostApi,
        '/api/post',
        '/api/post/<int:post_id>',
    )
    rest_api.init_app(app)
```

这也可以在`__init__.py`文件中的`create_app`函数中看到，如下所示：

```py
...
def create_app(object_name):
...
    from api import create_module as api_create_module
    ...
    api_create_module(app)

    return app
```

我们`Post` API 的控制逻辑和视图存储在名为`api/blog`的新文件夹中的`controllers.py`文件中。在`controllers.py`内部，我们将创建 API 本身，如下所示：

```py
from flask_restful import Resource 

class PostApi(Resource): 
  ...
```

在 Flask Restful 中，每个 REST 资源都定义为继承自`Resource`对象的一个类。这与第四章中展示的`MethodView`对象类似，*使用蓝图创建控制器*，任何继承自`Resource`对象的类都使用以 HTTP 方法命名的函数定义其逻辑。例如，当`GET` HTTP 方法击中`PostApi`类时，将执行`get`方法。

# JWT 认证

为了解决我们的认证问题，可以使用`Flask-Login`并检查登录的 cookie 数据。然而，这要求希望使用我们 API 的开发者通过网页界面登录他们的程序。我们也可以让开发者将登录数据与每个请求一起发送，但良好的设计实践是在绝对必要时才发送敏感信息。相反，我们的 API 将提供一个`auth/api`端点，允许他们发送登录凭证并获取一个访问令牌。

对于认证机制，我们将使用**JSON Web Token**（**JWT**）在用户登录时为我们的 API 消费者创建访问令牌。JWT 令牌声明了哪个用户已登录，从而节省服务器对数据库进行认证的另一个调用。此令牌内部编码了过期日期，不允许在过期后使用该令牌。这意味着即使令牌被恶意用户窃取，它也只有在客户端需要重新认证之前的一段时间内才有用。一如既往，请确保使用 HTTPS 来加密所有客户端-服务器连接。

为了利用这个特性，我们将使用另一个 Flask 扩展——Flask-JWT-extended。你将在`requirements.txt`文件中找到其依赖项的声明，如下所示：

```py
...
flask-jwt-extended
...
```

扩展的初始化将在`auth`模块中进行。

查看以下`auth/__init__.py`文件：

```py
from flask_jwt_extended import JWTManager
...
jwt = JWTManager()
...
def create_module(app, **kwargs):
    ...
    jwt.init_app(app)
    ...
```

接下来，我们使用以下辅助函数在同一个文件中认证用户：

```py

def authenticate(username, password):
    from .models import User
    user = User.query.filter_by(username=username).first()
    if not user:
        return None
    # Do the passwords match
    if not user.check_password(password):
        return None
    return user
```

登录端点的定义可以在`auth/controllers.py`中找到，如下面的代码所示：

```py
@auth_blueprint.route('/api', methods=['POST'])
def api():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    username = request.json.get('username', None)
 password = request.json.get('password', None)
    if not username:
        return jsonify({"msg": "Missing username parameter"}), 400
    if not password:
        return jsonify({"msg": "Missing password parameter"}), 400
 user = authenticate(username, password)
    if not user:
        return jsonify({"msg": "Bad username or password"}), 401
    # Identity can be any data that is json serializable
    access_token = create_access_token(identity=user.id)
    return jsonify(access_token=access_token), 200
```

首先，我们验证请求是否包含 JSON 正文。为此，我们使用 Flask 中的`request.is_json`函数。接下来，我们使用`request.json.get`从 JSON 正文中提取用户名和密码。然后我们使用之前的帮助函数`authenticate`检查用户的凭证。最后，我们使用用户名作为我们的身份返回 JWT 访问令牌。

我们 API 的用户必须将从这个资源收到的令牌传递给任何需要用户凭证的方法。为了测试此代码，将使用一个名为**curl**的工具。Curl 是 Bash 中包含的一个命令行工具，它允许创建和操作 HTTP 请求。要测试它，使用`curl`实用程序首先登录，如下面的代码所示：

```py
$ curl -H "Content-Type: application/json" -d '{"username":"user1","password":"password"}' http://localhost:5000/auth/api {
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyOGZjMDNkOC0xY2MyLTQwZDQtODJlMS0xMGQ0Mjc2YTk1ZjciLCJleHAiOjE1MzIwMTg4NDMsImZyZXNoIjpmYWxzZSwiaWF0IjoxNTMyMDE3OTQzLCJ0eXBlIjoiYWNjZXNzIiwibmJmIjoxNTMyMDE3OTQzLCJpZGVudGl0eSI6InVzZXIxIn0.Cs-ANWq0I2M2XMrZpQof-_cX0gsKE7U4UG1t1rB0UoY"
}
```

然后，我们使用`-H`标志发送请求头，表明内容正文是 JSON，并使用`-d`标志发送请求正文数据。接下来，我们可以使用令牌来访问受 API 保护的资源，如下所示：

```py
$ export ACCESS="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyOGZjMDNkOC0xY2MyLTQwZDQtODJlMS0xMGQ0Mjc2YTk1ZjciLCJleHAiOjE1MzIwMTg4NDMsImZyZXNoIjpmYWxzZSwiaWF0IjoxNTMyMDE3OTQzLCJ0eXBlIjoiYWNjZXNzIiwibmJmIjoxNTMyMDE3OTQzLCJpZGVudGl0eSI6InVzZXIxIn0.Cs-ANWq0I2M2XMrZpQof-_cX0gsKE7U4UG1t1rB0UoY"
$ curl -H "Authorization: Bearer $ACCESS" http://localhost:5000/api/post
```

注意访问令牌是如何在请求头中按照`Authorization: Bearer <TOKEN>`约定发送的。如果我们尝试在没有令牌的情况下访问相同的资源，我们会得到以下结果：

```py
$ curl -v http://localhost:5000/api/post
...
< HTTP/1.0 401 UNAUTHORIZED
...
{
  "msg": "Missing Authorization Header"
}
```

如预期，我们得到了 HTTP `401`状态码。为了保护 API 端点，我们只需使用`flask-jwt-extended`装饰器`@jwt_required`，并且为了获取用户名，我们使用`get_jwt_identity()`函数。

`flask-jwt-extended`装饰器提供了一些额外的功能，例如令牌过期、刷新令牌端点的功能以及许多配置选项。您可以在[`flask-jwt-extended.readthedocs.io/en/latest/`](http://flask-jwt-extended.readthedocs.io/en/latest/)了解更多信息。

# GET 请求

对于我们的某些`GET`、`PUT`和`DELETE`请求，我们的 API 将需要修改的帖子的 ID。

发送到客户端的数据必须是`Post`对象的表示，那么我们的`Post`对象将如何转换呢？Flask Restful 提供了一种通过`fields`对象和`marshal_with`函数装饰器将任何对象转换为 JSON 的方法。

# 输出格式化

输出格式是通过创建一个表示基本类型的`field`对象的字典来定义的。字段的键定义了字段将尝试转换的属性。通过将字典传递给`marshal_with`装饰器，`get`方法尝试返回的任何对象都将首先使用该字典进行转换。这也适用于对象列表。让我们看看实现此 API 端点的一种简单方法。以下示例代码考虑了分页，但我们将在稍后向您展示它是如何工作的。

看看以下`api/blog/controllers.py`文件：

```py
import datetime

from flask import abort
from flask_restful import Resource, fields, marshal_with
from flask_jwt_extended import jwt_required, get_jwt_identity
from webapp.blog.models import db, Post, Tag
from webapp.auth.models import User
...

post_fields = {
 'id': fields.Integer(),
    'author': fields.String(attribute=lambda x: x.user.username),
    'title': fields.String(),
    'text': HTMLField(),
    'tags': fields.List(fields.Nested(nested_tag_fields)),
    'publish_date': fields.DateTime(dt_format='iso8601')
}

class PostApi(Resource):
    @marshal_with(post_fields)
    @jwt_required
    def get(self, post_id=None):
        if post_id:
            post = Post.query.get(post_id)
            if not post:
                abort(404)
            return post
        else:
            posts = Post.query.all()
            return posts
```

在浏览器中重新加载 API 时，每个`Post`对象将以 JSON 格式显示。注意字段声明中的`HTMLField`。问题是 API 不应该从帖子创建表单中的 WYSIWYG 编辑器返回 HTML。如前所述，服务器不应该关心 UI，HTML 纯粹是输出规范。为了解决这个问题，我们需要一个自定义字段对象，它将从字符串中删除 HTML。在`api/blog/`文件夹中的新文件`fields.py`中，我们有以下内容：

```py
try:
 # Try python3
    from html.parser import HTMLParser
except Exception as e:
 # Nop python2
    from HTMLParser import HTMLParser

from flask_restful import fields

class HTMLStripper(HTMLParser):
    fed = list()

    def __init__(self):
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = HTMLStripper()
    s.feed(html)
    return s.get_data()

class HTMLField(fields.Raw):
    def format(self, value):
        return strip_tags(str(value))
```

异常块是为了考虑 Python2 和 Python3 的兼容性，因为 Python3 中`HTMLParser`模块的标准库已经改变。我们现在有一个`strip_tags`函数，它将返回任何被清理了 HTML 标签的字符串。通过从`fields.Raw`类继承并使用`strip_tags`函数传递值，定义了一个新的字段类型，称为`HTMLfield`。如果页面被重新加载，所有的 HTML 都会消失，只剩下文本。

Flask Restful 提供了许多默认字段，如下所示列表：

+   `fields.String`：使用`str()`转换值。

+   `fields.FormattedString`：传递带有括号中变量名的 Python 格式化字符串。

+   `fields.Url`：这提供了与 Flask `url_for`函数相同的功能。

+   `fields.DateTime`：将 Python `date`或`datetime`对象转换为字符串。`format`关键字参数指定字符串应该是`ISO8601`日期还是`RFC822`日期。

+   `fields.Float`：将值转换为浮点数的字符串表示。

+   `fields.Integer`：将值转换为整数的字符串表示。

+   `fields.Nested`：允许通过另一个字段对象字典表示嵌套对象。

+   `fields.List`：与 MongoEngine API 非常相似，这个字段接受另一个字段类型作为参数，并尝试将值列表转换为字段类型的 JSON 列表。

+   `fields.Boolean`：将值转换为`boolean`参数的字符串表示。

返回的数据中增加了两个字段：作者和标签。评论将被省略，因为它们应该包含在其自己的资源下。

`author`字段使用`field`类的属性关键字参数。这允许表示对象的任何属性，而不仅仅是基本级别的属性。因为标签的多对多关系返回一个对象列表，所以不能使用相同的解决方案来处理标签。现在可以使用`NestedField`类型在`ListField`内部和另一个字段字典返回一个标签字典列表。这给 API 的最终用户带来了额外的便利，使他们能够像有标签 API 一样轻松地进行查询。

# 请求参数

当向资源的根发送 `GET` 请求时，我们的 API 当前发送数据库中的所有 `Post` 对象。如果对象数量较少或使用 API 的人数较少，这是可以接受的。然而，如果任一数量增加，API 将对数据库造成大量压力。与网络界面类似，API 也应该分页。

为了实现这一点，我们的 API 需要接受一个名为 `page` 的 `GET` 查询字符串参数，该参数指定要加载的页面。Flask Restful 提供了一种方法来获取请求数据并解析它。如果缺少必要的参数，或者类型不匹配，Flask Restful 将自动创建一个 JSON 错误消息。在 `api/blog/` 文件夹中的新文件 `parsers.py` 中，你可以找到以下代码：

```py
...
from flask_restful import reqparse 
...
post_get_parser = reqparse.RequestParser() 
post_get_parser.add_argument( 
  'page', 
  type=int, 
  location=['args', 'headers'], 
  required=False,
) 
```

当请求没有 post ID 键时，以下代码是我们应该在 `PostApi` 类中拥有的代码：

```py
from .parsers import post_get_parser 
...
class PostApi(Resource): 
  @marshal_with(post_fields)
  @jwt_required 
  def get(self, post_id=None):
    if post_id: 
      .. 
      return post
    else:
      args = post_get_parser.parse_args() 
      page = args['page'] or 1 
      ...
      posts = Post.query.order_by( 
        Post.publish_date.desc() 
      ).paginate(page, current_app.config.get('POSTS_PER_PAGE', 10)) 
      ...
      return posts.items 
```

在前面的示例中，`RequestParser` 在查询字符串或请求头中查找 `page` 变量，并返回该页面的 `Post` 对象。同样，我们使用与网络视图页面版本相同的值来配置页面大小。我们使用 `current_app` Flask 代理来获取我们的配置中的任何值。

使用 `RequestParser` 创建解析器对象后，可以使用 `add_argument` 方法添加参数。`add_argument` 的第一个参数是要解析的参数的键，但 `add_argument` 还接受许多关键字参数，如下所示列表所示：

+   `action`: 解析器在成功解析值后对值执行的操作。两个可用的选项是 `store` 和 `append`。`store` 选项将解析值添加到返回的字典中。`append` 选项将解析值添加到字典中列表的末尾。

+   `case_sensitive`: 这是一个布尔参数，用于允许或禁止键不区分大小写。

+   `choices`: 这类似于 MongoEngine，是参数允许值的列表。

+   `default`: 如果请求中缺少参数，则生成的值。

+   `dest`: 这是添加解析值到返回数据中的键。

+   `help`: 这是当验证失败时返回给用户的消息。

+   `ignore`: 这是一个布尔参数，用于允许或禁止类型转换失败。

+   `location`: 这表示查找数据的位置。可用的位置如下：

    +   `args` 用于在 `GET` 查询字符串中查找

    +   `headers` 用于在 HTTP 请求头中查找

    +   `form` 用于在 HTTP `POST` 数据中查找

    +   `cookies` 用于在 HTTP 钩子中查找

    +   `json` 用于查找任何发送的 JSON

    +   `files` 用于在 `POST` 文件数据中查找

+   `required`: 这是一个布尔参数，用于确定该参数是否为可选的。

+   `store_missing`: 这是一个布尔参数，用于确定如果参数不在请求中，是否应该存储默认值。

+   `type`: 这是将传递的值转换为的 Python 类型。

使用 Flask Restful 解析器，向 API 添加新参数非常简单。例如，让我们添加一个用户参数，允许我们搜索所有由用户创建的帖子。首先，在`api/blog/parsers.py`文件中，我们有以下内容：

```py
post_get_parser = reqparse.RequestParser()
post_get_parser.add_argument('page', type=int, location=['args', 'headers'])
post_get_parser.add_argument('user', type=str, location=['args', 'headers'])
```

然后，在`api/blog/controllers.py`文件中，我们有以下内容：

```py
class PostApi(Resource):
    @marshal_with(post_fields)
    @jwt_required
    def get(self, post_id=None):
        if post_id:
            ...
            return post
        else:
            args = post_get_parser.parse_args()
            page = args['page'] or 1

            if args['user']:
 user = User.query.filter_by(username=args['user']).first()
 if not user:
 abort(404)

                posts = user.posts.order_by(
                    Post.publish_date.desc()
         ).paginate(page, current_app.config.get('POSTS_PER_PAGE', 10))
            else:
                posts = Post.query.order_by(
                    Post.publish_date.desc()
        ).paginate(page, current_app.config.get('POSTS_PER_PAGE', 10))
          return posts.items
```

当从`Resource`中调用 Flask 的`abort`函数时，Flask Restful 将自动创建一个错误消息，并带有状态码返回。

为了简化测试 API，我们使用`curl`，但你可以自由使用任何其他可用的工具与 HTTP API 交互。在从我们的认证端点请求访问令牌后，请求`post`并使用`id=1`，如下所示：

```py
$ curl -H "Authorization: Bearer $ACCESS" "http://localhost:5000/api/post/1"
```

或者你可以这样请求所有帖子：

```py
$ curl -H "Authorization: Bearer $ACCESS" "http://localhost:5000/api/post"
```

注意，响应只获取第一页，正如预期的那样。现在让我们请求第二页，如下所示：

```py
$ curl -H "Authorization: Bearer $ACCESS" "http://localhost:5000/api/post?page=2"
```

最后，你可以这样请求特定用户的帖子：

```py
$ curl -H "Authorization: Bearer $ACCESS" "http://localhost:5000/api/post?user=user1"
```

# POST 请求

REST 中的`POST`方法用于资源创建，尽管这不是一个幂等的方法。使用我们对 Flask Restful 解析器的新知识，我们可以处理`POST`端点。首先，我们需要一个解析器，它将接受标题、正文文本和标签列表。在`parser.py`文件中找到以下内容：

```py
post_post_parser = reqparse.RequestParser()
post_post_parser.add_argument(
    'title',
    type=str,
    required=True,
    help="Title is required",
    location=('json', 'values')
)
post_post_parser.add_argument(
    'text',
    type=str,
    required=True,
    help="Body text is required",
    location=('json', 'values')
)
post_post_parser.add_argument(
    'tags',
    type=str,
    action='append',
    location=('json', 'values')
)
```

接下来，我们创建了一个名为`add_tags_to_post`的辅助函数，用于向帖子添加标签。如果标签不存在，它将它们添加到数据库中。我们将在`POST`和`PUT`请求中使用它——这里没有新内容，只是一个简单的 SQLAlchemy 辅助函数，帮助我们使代码更简洁。

接下来，`PostApi`类需要一个`post`方法来处理传入的请求。`post`方法将使用给定的标题和正文文本值。如果存在`tags`键，则将标签添加到帖子中，如果传递的标签不存在，则会创建新的标签，如下面的代码所示：

```py
import datetime 
from .parsers import ( 
  post_get_parser, 
  post_post_parser 
) 
from webapp.models import db, User, Post, Tag 
class PostApi(Resource): 
  ... 
  @jwt_required
  def post(self, post_id=None):  
      args = post_post_parser.parse_args(strict=True) 
      new_post = Post(args['title'])
      new_post.user_id = get_jwt_identity()
      new_post.text = args['text'] 
      if args['tags']:
        add_tags_to_post(post, args['tags']) 
      db.session.add(new_post) 
      db.session.commit()
      return {'id': new_post.id}, 201
```

在`return`语句中，如果返回了一个元组，第二个参数被视为状态码。还有一个第三个值，通过传递一个字典作为额外的头部值。此外，请注意我们使用的`get_jwt_identity`函数，用于从 JWT 令牌中获取用户 ID。这是在登录阶段设置的，我们使用用户 ID 来设置 JWT 身份。

要传递`POST`变量，使用`d`标志，如下所示：

```py
$ curl -X POST -H "Authorization: Bearer $ACCESS" -H "Content-Type: application/json" -d '{"title":"Text Title", "text":"Some text"}' "http://localhost:5000/api/post" {
    "id": 310
}
```

应返回新创建帖子的 ID。如果你去浏览器，你应该能看到我们使用你用来生成认证令牌的用户创建的新帖子。

# PUT 请求

如本章开头表格中所示，`PUT`请求用于更改现有资源的值。就像`post`方法一样，我们应该做的第一件事是在`parsers.py`中创建一个新的解析器，如下所示：

```py
post_put_parser = reqparse.RequestParser()
post_put_parser.add_argument(
    'title',
    type=str,
    location=('json', 'values')
)
post_put_parser.add_argument(
    'text',
    type=str,
    location=('json', 'values')
)
post_put_parser.add_argument(
    'tags',
    type=str,
    action='append',
    location=('json', 'values')
)
```

`put`方法的逻辑与`post`方法非常相似。主要区别是每个更改都是可选的，任何不提供`post_id`的请求都会被拒绝，如下面的代码所示：

```py
...
def add_tags_to_post(post, tags_list):
    for item in tags_list:
        tag = Tag.query.filter_by(title=item).first()

        # Add the tag if it exists. If not, make a new tag
        if tag:
            post.tags.append(tag)
        else:
            new_tag = Tag(item)
            post.tags.append(new_tag)
...

    @jwt_required
    def put(self, post_id=None):
        if not post_id:
            abort(400)
        post = Post.query.get(post_id)
        if not post:
            abort(404)
        args = post_put_parser.parse_args(strict=True)
        if get_jwt_identity() != post.user_id:
            abort(403)
        if args['title']:
            post.title = args['title']
        if args['text']:
            post.text = args['text']
        if args['tags']:
            print("Tags %s" % args['tags'])
            add_tags_to_post(post, args['tags'])

        db.session.merge(post)
        db.session.commit()
        return {'id': post.id}, 201
```

还要注意的是，就像我们对待网页视图控制器一样，我们拒绝任何非博客文章创作者本人提出的更改博客文章的请求。

为了测试这个方法，`curl` 也可以使用 `-X` 标志创建 `PUT` 请求，如下所示：

```py
$ curl -X PUT -H "Authorization: Bearer $ACCESS" -H "Content-Type: application/json" \
 -d '{"title": "Modified From REST", "text": "this is from REST", "tags": ["tag1","tag2"]}' \
http://localhost:5000/api/post/5
```

# 删除请求

最后，在下面的代码中，我们有 `DELETE` 请求，这是四种支持的方法中最简单的一个。与 `delete` 方法的主要区别是它不返回任何内容，这是 `DELETE` 请求的接受标准：

```py
@jwt_required
def delete(self, post_id=None):
    if post_id:
        abort(400)
    post = Post.query.get(post_id)
    if not post:
        abort(404)
    if get_jwt_identity() != post.user_id:
        abort(401)
    db.session.delete(post)
    db.session.commit()
    return "", 204
```

再次提醒，我们可以使用以下方法进行测试：

```py
$ curl -X DELETE -H "Authorization: Bearer $ACCESS"
http://localhost:5000/api/post/102
```

如果一切删除成功，你应该收到 `204` 状态码，并且不应该有任何内容显示。

在我们完全离开 REST 之前，有一个最后的挑战要测试你对 Flask Restful 的理解。尝试创建一个 API，不仅可以从 `http://localhost:5000/api/comments` 进行修改，而且允许开发者通过使用 URL 格式 `http://localhost:5000/api/post/<int:post_id>/comments` 仅修改特定文章的评论。

# 摘要

我们的 `Post` API 现在已经是一个完整的特性。如果开发者想的话，他们可以使用这个 API 创建桌面或移动应用程序，而无需使用 HTML 抓取，这是一个非常漫长且繁琐的过程。给那些希望将你的网站作为平台使用的开发者提供这样的能力，将增加你网站的知名度，因为他们实际上会通过他们的应用程序或网站给你提供免费广告。

在下一章中，我们将使用流行的程序 Celery 来异步运行程序和任务，与我们的应用程序一起使用。
