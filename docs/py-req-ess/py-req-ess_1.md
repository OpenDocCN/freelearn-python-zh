# 第一章：使用 Requests 与网络交互

在这些现代日子里，从网络服务中读取数据和获取信息往往是一项至关重要的任务。每个人都知道**应用程序编程接口**（**API**）是如何让 Facebook 将“赞”按钮的使用推广到整个网络，并在社交通信领域占据主导地位的。它具有自己的特色，能够影响商业发展、产品开发和供应链管理。在这个阶段，学习一种有效处理 API 和打开网络 URL 的方法是当务之急。这将极大地影响许多网络开发过程。

# HTTP 请求简介

每当我们的网页浏览器试图与网页服务器进行通信时，它都是通过使用**超文本传输协议**（**HTTP**）来完成的，该协议作为一个请求-响应协议。在这个过程中，我们向网页服务器发送一个请求，并期待得到一个回应。以从网站下载 PDF 文件为例。我们发送一个请求说“给我这个特定的文件”，然后我们从网页服务器得到一个响应，其中包含“这里是文件，接着是文件本身”。我们发送的 HTTP 请求可能包含许多有趣的信息。让我们深入挖掘它。

这里是经过我的设备发送的 HTTP 请求的原始信息。通过查看以下示例，我们可以掌握请求的重要部分：

```py
* Connected to google.com (74.125.236.35) port 80 (#0)
> GET / HTTP/1.1
> User-Agent: curl/7.35.0
> Host: google.com
> Accept: */*
>
< HTTP/1.1 302 Found
< Cache-Control: private
< Content-Type: text/html; charset=UTF-8
< Location: http://www.google.co.in/?gfe_rd=cr&ei=_qMUVKLCIa3M8gewuoCYBQ
< Content-Length: 261
< Date: Sat, 13 Sep 2014 20:07:26 GMT
* Server GFE/2.0 is not blacklisted
< Server: GFE/2.0
< Alternate-Protocol: 80:quic,p=0.002
```

现在，我们将向服务器发送一个请求。让我们利用 HTTP 请求的这些部分：

+   **方法**：前一个示例中的 `GET / http /1.1` 是一个大小写敏感的 HTTP 方法。以下是一些 HTTP 请求方法：

    +   `GET`：此操作通过给定的 URI 从指定的服务器获取信息。

    +   `HEAD`: 这个功能与 GET 类似，但不同之处在于，它只返回状态行和头部部分。

    +   `POST`：这可以将我们希望处理的数据提交到服务器。

    +   `PUT`：当我们打算创建一个新的 URL 时，这个操作会创建或覆盖目标资源的所有当前表示。

    +   `DELETE`：此操作将删除由给定`Request-URI`描述的所有资源。

    +   `OPTIONS`: 这指定了请求/响应周期的通信选项。它允许客户端提及与资源相关联的不同选项。

+   **请求 URI**：统一资源标识符（URI）具有识别资源名称的能力。在先前的例子中，主机名是`请求 URI`。

+   **请求头字段**：如果我们想添加更多关于请求的信息，我们可以使用请求头字段。它们是冒号分隔的键值对。一些`request-headers`的值包括：

    +   `Accept-Charset`: 这用于指示响应可接受的字符集。

    +   `Authorization`: 这包含用户代理的认证信息的凭证值。

    +   `主机`: 这标识了用户请求的资源所对应的互联网主机和端口号，使用用户提供的原始 URI。

    +   `User-agent`: 它容纳了关于发起请求的用户代理的信息。这可以用于统计目的，例如追踪协议违规行为。

# Python 模块

有些广泛使用的 Python 模块可以帮助打开 URL。让我们来看看它们：

+   `httplib2`: 这是一个功能全面的 HTTP 客户端库。它支持许多其他 HTTP 库中未提供的特性。它支持诸如缓存、持久连接、压缩、重定向以及多种认证等功能。

+   `urllib2`：这是一个在复杂世界中广泛使用的模块，用于获取 HTTP URL。它定义了帮助进行 URL 操作的功能和类，例如基本和摘要认证、重定向、cookies 等。

+   `Requests`：这是一个 Apache2 许可的 Python 编写的 HTTP 库，拥有许多功能，从而提高了生产力。

# 请求与 urllib2

让我们比较`urllib2`和`Requests`；`urllib2.urlopen()`可以用来打开一个 URL（可以是字符串或请求对象），但在与网络交互时，还有很多其他事情可能会成为负担。在这个时候，一个简单的 HTTP 库，它具有使网络交互变得顺畅的能力，正是当务之急，而 Requests 就是其中之一。

以下是一个使用 `urllib2` 和 `Requests` 获取网络服务数据的示例，它清晰地展示了使用 `Requests` 的工作是多么简单：

以下代码给出了`urllib2`的一个示例：

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib2

gh_url = 'https://api.github.com'

req = urllib2.Request(gh_url)

password_manager = urllib2.HTTPPasswordMgrWithDefaultRealm()
password_manager.add_password(None, gh_url, 'user', 'pass')

auth_manager = urllib2.HTTPBasicAuthHandler(password_manager)
opener = urllib2.build_opener(auth_manager)

urllib2.install_opener(opener)

handler = urllib2.urlopen(req)

print handler.getcode()
print handler.headers.getheader('content-type')

# ------

# 'application/json'

```

使用 `Requests` 实现的相同示例：

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests

r = requests.get('https://api.github.com', auth=('user', 'pass'))

print r.status_code
print r.headers['content-type']

# ------

# 'application/json'

```

这些示例可以在[`gist.github.com/kennethreitz/973705`](https://gist.github.com/kennethreitz/973705)找到。

在这个初始阶段，示例可能看起来相当复杂。不要深入到示例的细节中。只需看看`requests`库带来的美丽之处，它使我们能够用非常少的代码登录 GitHub。使用`requests`的代码似乎比`urllib2`示例简单且高效得多。这将有助于我们在各种事情上提高生产力。

# 请求的本质

与 `HTTP/1.0` 类似，`HTTP/1.1` 具有很多优点和新增功能，例如多次重用连接以减少相当大的开销、持久连接机制等。幸运的是，`requests` 库就是基于它构建的，这让我们能够与网络进行平滑和无缝的交互。我们无需手动将查询字符串添加到我们的 URL 中，也不必对 POST 数据进行编码。持久连接和 HTTP 连接池是完全自动的，由嵌入在 `requests` 中的 `urllib3` 提供。使用 `requests`，我们得到了一种无需再次考虑编码参数的方法，无论它是 GET 还是 POST。

在 URL 中无需手动添加查询字符串，同样也不需要添加诸如连接池保持活动状态、带有 cookie 持久性的会话、基本/摘要认证、浏览器风格的 SSL 验证、连接超时、多部分文件上传等功能。

# 提出一个简单的请求

现在我们来创建第一个获取网页的请求，这个过程非常简单。它包括导入`requests`模块，然后使用`get`方法获取网页。让我们来看一个例子：

```py
>>> import requests
>>> r =  requests.get('http://google.com')

```

哇哦！我们完成了。

在前面的例子中，我们使用 `requests.get` 获取了 `google` 网页，并将其保存在变量 `r` 中，该变量最终变成了 `response` 对象。`response` 对象 `r` 包含了关于响应的大量信息，例如头部信息、内容、编码类型、状态码、URL 信息以及许多更复杂的细节。

同样地，我们可以使用所有 HTTP 请求方法，如 GET、POST、PUT、DELETE、HEAD，与 `requests` 一起使用。

现在我们来学习如何在 URL 中传递参数。我们可以使用`params`关键字将参数添加到请求中。

以下为传递参数所使用的语法：

```py
parameters = {'key1': 'value1', 'key2': 'value2'}
r = requests.get('url', params=parameters)
```

为了对此有一个清晰的了解，让我们通过登录 GitHub，使用以下代码中的`requests`来获取 GitHub 用户详细信息：

```py
>>> r = requests.get('https://api.github.com/user', auth=('myemailid.mail.com', 'password'))
>>> r.status_code
200
>>> r.url
u'https://api.github.com/user'
>>> r.request
<PreparedRequest [GET]>

```

我们使用了`auth`元组，它支持基本/摘要/自定义认证，用于登录 GitHub 并获取用户详情。`r.status_code`的结果表明我们已成功获取用户详情，并且我们已经访问了 URL，以及请求的类型。

# 响应内容

响应内容是在我们发送请求时，服务器返回到我们控制台的信息。

在与网络交互时，解码服务器的响应是必要的。在开发应用程序时，我们可能会遇到许多需要处理原始数据、JSON 格式或甚至二进制响应的情况。为此，`requests`库具有自动解码服务器内容的能力。Requests 可以顺畅地解码许多 Unicode 字符集。此外，Requests 还会根据响应的编码进行有根据的猜测。这基本上是通过考虑头部信息来实现的。

如果我们访问 `r.content` 的值，它将返回一个原始字符串格式的响应内容。如果我们访问 `r.text`，Requests 库将使用 `r.encoding` 对响应（`r.content` 的值）进行编码，并返回一个新的编码字符串。在这种情况下，如果 `r.encoding` 的值为 `None`，Requests 将使用 `r.apparent_encoding` 来假设编码类型，`r.apparent_encoding` 是由 `chardet` 库提供的。

我们可以通过以下方式访问服务器的响应内容：

```py
>>> import requests
>>> r = requests.get('https://google.com')
>>> r.content
'<!doctype html><html itemscope="" itemtype="http://schema.org/WebPage" …..'
>>> type(r.content)
<type 'str'>
>>> r.text
u'<!doctype html><html itemscope=""\ itemtype="http://schema.org/WebPage" lang="en-IN"><head><meta content="........
>>> type(r.text)
<type 'unicode'>

```

在前面的行中，我们尝试使用 `requests.get()` 获取 `google` 首页，并将其赋值给变量 `r`。这里的 `r` 变量实际上是一个请求对象，我们可以使用 `r.content` 访问原始内容，以及使用 `r.text` 获取编码后的响应内容。

如果我们想找出 Requests 正在使用的编码，或者如果我们想更改编码，我们可以使用属性`r.encoding`，如下面的示例所示：

```py
>>> r.encoding
'ISO-8859-1'
>>> r.encoding = 'utf-8'

```

在代码的第一行，我们正在尝试访问 Requests 所使用的编码类型。结果是`'ISO-8859-1'`。在下一行，我希望将编码更改为`'utf-8'`。因此，我将编码类型赋值给了`r.encoding`。如果我们像第二行那样更改编码，Requests 通常会使用已分配的`r.encoding`的最新值。所以从那时起，每次我们调用`r.text`时，它都会使用相同的编码。

例如，如果`r.encoding`的值为`None`，Requests 通常会使用`r.apparent_encoding`的值。以下示例解释了这种情况：

```py
>>> r.encoding = None
>>> r.apparent_encoding
'ascii'

```

通常，显式编码的值由`chardet`库指定。如果我们充满热情地尝试将新的编码类型设置为`r.apparent_encoding`，Requests 将引发一个`AttributeError`，因为其值不能被更改。

```py
>>> r.apparent_encoding = 'ISO-8859-1'
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
AttributeError: can't set attribute

```

请求足够高效，可以自定义编码。以我们创建了自己的编码并已在`codecs`模块中注册为例，我们可以轻松地使用我们的自定义编解码器；这是因为`r.encoding`的值和 Requests 会处理解码工作。

# 不同类型的请求内容

Requests 具有处理不同类型的请求内容的功能，例如二进制响应内容、JSON 响应内容和原始响应内容。为了清晰地展示不同类型的响应内容，我们列出了详细信息。这里使用的示例是用 Python 2.7.x 开发的。

## 自定义标题

我们可以在请求中发送自定义的头部信息。为此，我们只需创建一个包含我们头部信息的字典，并在`get`或`post`方法中传递头部参数。在这个字典中，键是头部信息的名称，而值则是，嗯，这对的值。让我们将一个 HTTP 头部信息传递给一个请求：

```py
>>> import json
>>> url = 'https://api.github.com/some/endpoint'
>>>  payload = {'some': 'data'}
>>> headers = {'Content-Type': 'application/json'}
>>> r = requests.post(url, data=json.dumps(payload), headers=headers)

```

此示例取自[`docs.python-requests.org/en/latest/user/quickstart/#custom-headers`](http://docs.python-requests.org/en/latest/user/quickstart/#custom-headers)中的请求文档。

在这个例子中，我们向请求发送了一个带有值`application/json`的头部`content-type`作为参数。

同样地，我们可以发送带有自定义头部的请求。比如说，我们有必要发送一个带有授权头部且值为某个令牌的请求。我们可以创建一个字典，其键为`'Authorization'`，值为一个看起来如下所示的令牌：

```py
>>> url = 'some url'
>>>  header = {'Authorization' : 'some token'}
>>> r.request.post(url, headers=headers)

```

## 发送表单编码的数据

我们可以使用 Requests 发送类似 HTML 表单的表单编码数据。将一个简单的字典传递给 data 参数即可完成此操作。当发起请求时，数据字典将自动转换为表单编码。

```py
>>> payload = {'key1': 'value1', 'key2': 'value2'}
>>> r = request.post("some_url/post", data=payload)
>>> print(r.text)
{
 …
 "form": {
 "key2": "value2",
 "key1": "value1"
 },
 …
}

```

在前面的例子中，我们尝试发送了表单编码的数据。而在处理非表单编码的数据时，我们应该在字典的位置发送一个字符串。

## 发布多部分编码的文件

我们倾向于通过 POST 方法上传多部分数据，如图片或文件。在 `requests` 库中，我们可以使用 `files` 参数来实现，它是一个包含 `'name'` 和 `file-like-objects` 值的字典。同时，我们也可以将其指定为 `'name'`，值可以是 `'filename'` 或 `fileobj`，就像以下这种方式：

```py
{'name' : file-like-objects} or
{'name': ('filename',  fileobj)}

```

示例如下：

```py
>>> url = 'some api endpoint'
>>> files = {'file': open('plan.csv', 'rb')}
>>> r = requests.post(url, files=files)

We can access the response using 'r.text'.
>>>  r.text
{
 …
 "files": {
 "file": "< some data … >"
 },
 ….
}

```

在前一个例子中，我们没有指定内容类型或头部信息。除此之外，我们还有能力为上传的文件设置名称：

```py
>>> url = 'some url'
>>> files = {'file': ('plan.csv', open('plan.csv', 'rb'), 'application/csv', {'Expires': '0'})}
>>> r = requests.post(url, files)
>>> r.text
{
 …
 "files"
 "file": "< data...>"
 },
 …
}

```

我们也可以以下这种方式发送字符串作为文件接收：

```py
>>> url = 'some url'
>>> files = {'file' : ('plan.csv', 'some, strings, to, send')}
>>> r.text
{
 …
 "files": {
 "file": "some, strings, to, send"
 },
 …
}

```

# 查看内置响应状态码

状态码有助于让我们知道请求发送后的结果。为了了解这一点，我们可以使用`status_code`：

```py
>>> r = requests.get('http://google.com')
>>> r.status_code
200

```

为了使处理`状态码`变得更加容易，Requests 模块内置了一个状态码查找对象，它作为一个便捷的参考。我们必须将`requests.codes.ok`与`r.status_code`进行比较以实现这一点。如果结果为`True`，则表示是`200`状态码，如果为`False`，则不是。我们还可以将`r.status.code`与`requests.codes.ok`、`requests.code.all_good`进行比较，以使查找工作得以进行。

```py
>>> r = requests.get('http://google.com')
>>> r.status_code == requests.codes.ok
True

```

现在，让我们尝试检查一个不存在的 URL。

```py
>>> r = requests.get('http://google.com/404')
>>> r.status_code == requests.codes.ok
False

```

我们有处理不良`请求`的能力，例如 4XX 和 5XX 类型的错误，通过通知错误代码来实现。这可以通过使用`Response.raise_for_status()`来完成。

让我们先通过发送一个错误请求来尝试一下：

```py
>>> bad_request = requests.get('http://google.com/404')
>>> bad_request.status_code
404
>>>bad_request.raise_for_status()
---------------------------------------------------------------------------
HTTPError                              Traceback (most recent call last)
----> bad_request..raise_for_status()

File "requests/models.py",  in raise_for_status(self)
 771
 772         if http_error_msg:
--> 773             raise HTTPError(http_error_msg, response=self)
 774
 775     def close(self):

HTTPError: 404 Client Error: Not Found

```

现在如果我们尝试一个有效的 URL，我们不会得到任何响应，这是成功的标志：

```py
>>> bad_request = requests.get('http://google.com')
>>> bad_request.status_code
200
>>> bad_request.raise_for_status()
>>>

```

### 小贴士

**下载示例代码**

您可以从[`www.packtpub.com`](http://www.packtpub.com)下载您购买的所有 Packt Publishing 书籍的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

# 查看响应头

服务器响应头帮助我们了解原始服务器处理请求所使用的软件。我们可以通过`r.headers`访问服务器响应头：

```py
>>> r = requests.get('http://google.com')
>>> r.headers
CaseInsensitiveDict({'alternate-protocol': '80:quic', 'x-xss-protection': '1; mode=block', 'transfer-encoding': 'chunked', 'set-cookie': 'PREF=ID=3c5de2786273fce1:FF=0:TM=1410378309:LM=1410378309:S=DirRRD4dRAxp2Q_3; …..

```

**请求评论**（**RFC**）7230 表示 HTTP 头部名称不区分大小写。这使我们能够使用大写和小写字母访问头部。

```py
>>> r.headers['Content-Type']
'text/html; charset=ISO-8859-1'

>>>  r.headers.get('content-type')
'text/html; charset=ISO-8859-1'

```

# 使用 Requests 访问 Cookies

我们可以访问响应中的 cookie，如果存在的话：

```py
>>> url = 'http://somewebsite/some/cookie/setting/url'
>>> r = requests.get(url)

>>> r.cookies['some_cookie_name']
'some_cookie_value'

```

我们可以发送自己的 cookie，如下例所示：

```py
>>> url = 'http://httpbin.org/cookies'
>>> cookies = dict(cookies_are='working')

>>> r = requests.get(url, cookies=cookies)
>>> r.text
'{"cookies": {"cookies_are": "working"}}'

```

# 使用请求历史跟踪请求的重定向

有时我们访问的 URL 可能已经被移动，或者可能会被重定向到其他位置。我们可以使用 Requests 来追踪它们。响应对象的 history 属性可以用来追踪重定向。Requests 除了 HEAD 之外，可以用每个动词完成位置重定向。`Response.history`列表包含了为了完成请求而生成的 Requests 对象。

```py
>>> r = requests.get('http:google.com')
>>> r.url
u'http://www.google.co.in/?gfe_rd=cr&ei=rgMSVOjiFKnV8ge37YGgCA'
>>> r.status_code
200
>>> r.history
(<Response [302]>,)

```

在前面的例子中，当我们尝试向`'www.google.com'`发送请求时，我们得到了`r.history`的值为`302`，这意味着 URL 已经被重定向到了其他位置。`r.url`在这里显示了这一重定向的证明，包括重定向的 URL。

如果我们不想让 Requests 处理重定向，或者我们在使用 POST、GET、PUT、PATCH、OPTIONS 或 DELETE 时，我们可以将`allow_redirects=False,`的值设置为 False，这样重定向处理就会被禁用。

```py
>>> r = requests.get('http://google.com', allow_redirects=False)
>>> r.url
u'http://google.com/'
>> r.status_code
302
>>> r.history
[ ]

```

在前面的例子中，我们使用了参数 `allow_redirects=False,`，这导致在 URL 中没有任何重定向，`r.url` 的值保持不变，而 `r.history` 被设置为空。

如果我们使用头部来访问 URL，我们可以简化重定向过程。

```py
>>> r = requests.head('http://google.com', allow_redirects=True)
>>> r.url
u'http://www.google.co.in/?gfe_rd=cr&ei=RggSVMbIKajV8gfxzID4Ag'
>>> r.history
(<Response [302]>,)

```

在这个例子中，我们尝试使用带有启用参数`allow_redirects`的 head 方法访问 URL，结果导致 URL 被重定向。

# 使用超时来控制高效使用

考虑这样一个案例，我们正在尝试访问一个耗时过多的响应。如果我们不想让进程继续进行，并在超过特定时间后抛出异常，我们可以使用参数`timeout`。

当我们使用`timeout`参数时，我们是在告诉 Requests 在经过特定的时间段后不要等待响应。如果我们使用`timeout`，这并不等同于在整个响应下载上定义一个时间限制。如果在指定的`timeout`秒内底层套接字上没有确认任何字节，抛出一个异常是一个好的实践。

```py
>>> requests.get('http://google.com', timeout=0.03)
---------------------------------------------------------------------------
Timeout                                   Traceback (most recent call last)
…….
……..
Timeout: HTTPConnectionPool(host='google.com', port=80): Read timed\ out. (read timeout=0.03)

```

在本例中，我们已将`timeout`值指定为`0.03`，超时时间已超过，从而带来了响应，因此导致了`timeout`异常。超时可能发生在两种不同的情况下：

+   在尝试连接到位于远程位置的服务器时，请求超时了。

+   如果服务器在分配的时间内没有发送完整的响应，请求将超时。

# 错误和异常

在发送请求和获取响应的过程中，如果出现问题，将会引发不同类型的错误和异常。其中一些如下：

+   `HTTPError`: 当存在无效的 HTTP 响应时，Requests 将引发一个`HTTPError`异常

+   `ConnectionError`：如果存在网络问题，例如连接被拒绝和 DNS 故障，Requests 将引发`ConnectionError`异常

+   `Timeout`: 如果请求超时，将抛出此异常

+   `TooManyRedirects`: 如果请求超过了配置的最大重定向次数，则会引发此类异常

其他一些出现的异常类型包括`Missing schema Exception`、`InvalidURL`、`ChunkedEncodingError`、`ContentDecodingError`等等。

此示例取自[`docs.python-requests.org/en/latest/user/quickstart/#errors-and-exceptions`](http://docs.python-requests.org/en/latest/user/quickstart/#errors-and-exceptions)提供的请求文档。

# 摘要

在本章中，我们介绍了一些基本主题。我们学习了为什么 Requests 比`urllib2`更好，如何发起一个简单的请求，不同类型的响应内容，如何为我们的 Requests 添加自定义头信息，处理表单编码数据，使用状态码查找，定位请求的重定向位置以及关于超时的问题。

在下一章中，我们将深入学习 Requests 的高级概念，这将帮助我们根据需求灵活地使用 Requests 库。
