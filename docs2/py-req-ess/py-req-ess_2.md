# 第二章 深入挖掘请求

在本章中，我们将探讨 Requests 模块的高级主题。Requests 模块中还有许多更多功能，使得与网络的交互变得轻而易举。让我们更深入地了解使用 Requests 模块的不同方法，这有助于我们理解其使用的便捷性。

简而言之，我们将涵盖以下主题：

+   使用 Session 对象在请求间持久化参数

+   揭示请求和响应的结构

+   使用预定义请求

+   使用 Requests 验证 SSL 证书

+   主体内容工作流程

+   使用生成器发送分块编码的请求

+   使用事件钩子获取请求方法参数

+   遍历流式 API

+   使用链接头描述 API

+   传输适配器

# 使用 Session 对象在请求间持久化参数

Requests 模块包含一个`session`对象，该对象具有在请求之间持久化设置的能力。使用这个`session`对象，我们可以持久化 cookies，我们可以创建准备好的请求，我们可以使用 keep-alive 功能，并且可以做更多的事情。Session 对象包含了 Requests API 的所有方法，如`GET`、`POST`、`PUT`、`DELETE`等等。在使用 Session 对象的所有功能之前，让我们了解一下如何使用会话并在请求之间持久化 cookies。

让我们使用会话方法来获取资源。

```py
>>> import requests
>>> session = requests.Session()
>>> response = requests.get("https://google.co.in", cookies={"new-cookie-identifier": "1234abcd"})

```

在前面的例子中，我们使用`requests`创建了一个`session`对象，并使用其`get`方法来访问网络资源。

在前一个示例中我们设置的`cookie`值可以通过`response.request.headers`来访问。

```py
>>> response.request.headers
CaseInsensitiveDict({'Cookie': 'new-cookie-identifier=1234abcd', 'Accept-Encoding': 'gzip, deflate, compress', 'Accept': '*/*', 'User-Agent': 'python-requests/2.2.1 CPython/2.7.5+ Linux/3.13.0-43-generic'})
>>> response.request.headers['Cookie']
'new-cookie-identifier=1234abcd'

```

使用 `session` 对象，我们可以指定一些属性的默认值，这些值需要通过 GET、POST、PUT 等方式发送到服务器。我们可以通过在 `Session` 对象上指定 `headers`、`auth` 等属性的值来实现这一点。

```py
>>> session.params = {"key1": "value", "key2": "value2"}
>>> session.auth = ('username', 'password')
>>> session.headers.update({'foo': 'bar'})

```

在前面的示例中，我们使用`session`对象为属性设置了一些默认值——`params`、`auth`和`headers`。如果我们想在后续请求中覆盖它们，可以像以下示例中那样操作：

```py
>>> session.get('http://mysite.com/new/url', headers={'foo': 'new-bar'})

```

# 揭示请求和响应的结构

请求对象是用户在尝试与网络资源交互时创建的。它将以准备好的请求形式发送到服务器，并且包含一些可选的参数。让我们来仔细看看这些参数：

+   `方法`: 这是与网络服务交互所使用的 HTTP 方法。例如：GET、POST、PUT。

+   `URL`: 需要发送请求的网页地址。

+   `headers`: 请求中要发送的头部信息字典。

+   `files`: 在处理分片上传时可以使用。这是一个文件字典，键为文件名，值为文件对象。

+   `data`：这是要附加到`request.json`的正文。这里有两种情况会出现：

    +   如果提供了`json`，则头部中的`content-type`会被更改为`application/json`，在此点，`json`作为请求的主体。

    +   在第二种情况下，如果同时提供了`json`和`data`，则`data`会被静默忽略。

+   `params`：一个字典，包含要附加到 URL 的 URL 参数。

+   `auth`: 这用于我们在需要指定请求的认证时。它是一个包含用户名和密码的元组。

+   `cookies`：一个字典或饼干罐，可以添加到请求中。

+   `hooks`: 回调钩子的字典。

响应对象包含服务器对 HTTP 请求的响应。它是在 Requests 从服务器收到响应后生成的。它包含了服务器返回的所有信息，同时也存储了我们最初创建的请求对象。

每当我们使用 `requests` 向服务器发起调用时，在此背景下会发生两个主要的事务，具体如下：

+   我们正在构建一个请求对象，该对象将被发送到服务器以请求资源

+   由`requests`模块生成一个响应对象

现在，让我们来看一个从 Python 官方网站获取资源的例子。

```py
>>> response = requests.get('https://python.org')

```

在上一行代码中，创建了一个`requests`对象，并将发送到`'https://python.org'`。因此获得的 Requests 对象将被存储在`response.request`变量中。我们可以以下这种方式访问发送到服务器的请求对象的头部信息：

```py
>>> response.request.headers
CaseInsensitiveDict({'Accept-Encoding': 'gzip, deflate, compress', 'Accept': '*/*', 'User-Agent': 'python-requests/2.2.1 CPython/2.7.5+ Linux/3.13.0-43-generic'})

```

服务器返回的标题可以通过其 'headers' 属性进行访问，如下例所示：

```py
>>> response.headers
CaseInsensitiveDict({'content-length': '45950', 'via': '1.1 varnish', 'x-cache': 'HIT', 'accept-ranges': 'bytes', 'strict-transport-security': 'max-age=63072000; includeSubDomains', 'vary': 'Cookie', 'server': 'nginx', 'age': '557','content-type': 'text/html; charset=utf-8', 'public-key-pins': 'max-age=600; includeSubDomains; ..)

```

`response` 对象包含不同的属性，如 `_content`、`status_code`、`headers`、`url`、`history`、`encoding`、`reason`、`cookies`、`elapsed`、`request`。

```py
>>> response.status_code
200
>>> response.url
u'https://www.python.org/'
>>> response.elapsed
datetime.timedelta(0, 1, 904954)
>>> response.reason
'OK'

```

# 使用准备好的 Requests

我们发送给服务器的每个请求默认都会转换为`PreparedRequest`。从 API 调用或会话调用中接收到的`Response`对象的`request`属性实际上就是使用的`PreparedRequest`。

可能存在需要发送请求并额外添加不同参数的情况。参数可以是`cookies`、`files`、`auth`、`timeout`等等。我们可以通过使用会话和预请求的组合来高效地处理这个额外步骤。让我们来看一个例子：

```py
>>> from requests import Request, Session
>>> header = {}
>>> request = Request('get', 'some_url', headers=header)

```

我们在之前的例子中尝试发送一个带有头部的`get`请求。现在，假设我们打算使用相同的方法、URL 和头部信息发送请求，但想要添加一些额外的参数。在这种情况下，我们可以使用会话方法来接收完整的会话级别状态，以便访问最初发送请求的参数。这可以通过使用`session`对象来实现。

```py
>>> from requests import Request, Session
>>> session = Session()
>>> request1 = Request('GET', 'some_url', headers=header)

```

现在，让我们使用`session`对象来准备一个请求，以获取`session`级别的状态值：

```py
>>> prepare = session.prepare_request(request1)

```

我们现在可以发送带有更多参数的请求对象 `request`，如下所示：

```py
>>> response = session.send(prepare, stream=True, verify=True)
200

```

哇！节省了大量时间！

`prepare` 方法使用提供的参数准备完整的请求。在之前的示例中使用了 `prepare_request` 方法。还有一些其他方法，如 `prepare_auth`、`prepare_body`、`prepare_cookies`、`prepare_headers`、`prepare_hooks`、`prepare_method`、`prepare_url`，这些方法用于创建单个属性。

# 使用 Requests 验证 SSL 证书

Requests 提供了验证 HTTPS 请求的 SSL 证书的功能。我们可以使用 `verify` 参数来检查主机的 SSL 证书是否已验证。

让我们考虑一个没有 SSL 证书的网站。我们将向其发送带有参数`verify`的 GET 请求。

发送请求的语法如下：

```py
requests.get('no ssl certificate site', verify=True)
```

由于该网站没有 SSL 证书，将会导致类似以下错误：

```py
requests.exceptions.ConnectionError: ('Connection aborted.', error(111, 'Connection refused'))

```

让我们验证一个已认证网站的 SSL 证书。考虑以下示例：

```py
>>> requests.get('https://python.org', verify=True)
<Response [200]>

```

在前面的例子中，结果是`200`，因为提到的网站是 SSL 认证的。

如果我们不想通过请求验证 SSL 证书，那么我们可以将参数设置为`verify=False`。默认情况下，`verify`的值将变为`True`。

# 主体内容工作流程

以一个在我们发起请求时正在下载连续数据流的例子来说明。在这种情况下，客户端必须持续监听服务器，直到接收到完整的数据。考虑先访问响应内容，然后再担心体（body）的情况。在上面的两种情况下，我们可以使用参数`stream`。让我们来看一个例子：

```py
>>> requests.get("https://pypi.python.org/packages/source/F/Flask/Flask-0.10.1.tar.gz", stream=True)

```

如果我们使用参数 `stream=True,` 发起请求，连接将保持开启状态，并且只会下载响应的头信息。这使我们能够通过指定条件，如数据字节数，在需要时随时获取内容。

语法如下：

```py
if int(request.headers['content_length']) < TOO_LONG:
content = r.content
```

通过设置参数 `stream=True` 并将响应作为类似文件的 `response.raw` 对象访问，如果我们使用 `iter_content` 方法，我们可以遍历 `response.data`。这将避免一次性读取较大的响应。

语法如下：

```py
iter_content(chunk_size=size in bytes, decode_unicode=False)
```

同样地，我们可以使用`iter_lines`方法遍历内容，该方法将逐行遍历响应数据。

语法如下：

```py
iter_lines(chunk_size = size in bytes, decode_unicode=None, delimitter=None)
```

### 注意

在使用`stream`参数时需要注意的重要事项是，当它被设置为`True`时，并不会释放连接，除非所有数据都被消耗或者执行了`response.close`。

## Keep-alive 功能

由于`urllib3`支持对同一套接字连接进行多次请求的重用，我们可以使用一个套接字发送多个请求，并通过`Requests`库中的 keep-alive 功能接收响应。

在一个会话中，它变成了自动的。会话中提出的每个请求默认都会自动使用适当的连接。在读取完所有来自主体的数据后，所使用的连接将被释放。

## 流式上传

一个类似文件的大尺寸对象可以使用`Requests`库进行流式传输和上传。我们所需做的只是将流的内容作为值提供给`request`调用中的`data`属性，如下面的几行所示。

语法如下：

```py
with open('massive-body', 'rb') as file:
    requests.post('http://example.com/some/stream/url',
                  data=file)
```

# 使用生成器发送分块编码的请求

分块传输编码是 HTTP 请求中传输数据的一种机制。通过这个机制，数据被分成一系列的数据块进行发送。请求支持分块传输编码，适用于出站和入站请求。为了发送一个分块编码的请求，我们需要提供一个用于你主体的生成器。

使用方法如下所示：

```py
>>> def generator():
...     yield "Hello "
...     yield "World!"
...
>>> requests.post('http://example.com/some/chunked/url/path',
 data=generator())

```

# 使用事件钩子获取请求方法参数

我们可以使用钩子来修改请求过程信号事件的处理部分。例如，有一个名为`response`的钩子，它包含了从请求生成的响应。它是一个可以作为请求参数传递的字典。其语法如下：

```py
hooks = {hook_name: callback_function, … }
```

`callback_function` 参数可能返回也可能不返回值。当它返回值时，假设它是用来替换传入的数据。如果回调函数不返回任何值，则对数据没有任何影响。

这里是一个回调函数的示例：

```py
>>> def print_attributes(request, *args, **kwargs):
...     print(request.url)
...     print(request .status_code)
...     print(request .headers)

```

如果在执行`callback_function`函数时出现错误，你将在标准输出中收到一个警告信息。

现在我们使用前面的`callback_function`来打印一些请求的属性：

```py
>>> requests.get('https://www.python.org/',
 hooks=dict(response=print_attributes))
https://www.python.org/
200
CaseInsensitiveDict({'content-type': 'text/html; ...})
<Response [200]>

```

# 遍历流式 API

流式 API 通常会保持请求开启，使我们能够实时收集流数据。在处理连续数据流时，为确保不遗漏任何消息，我们可以借助 Requests 库中的`iter_lines()`方法。`iter_lines()`会逐行遍历响应数据。这可以通过在发送请求时将参数 stream 设置为`True`来实现。

### 注意事项

最好记住，调用 `iter_lines()` 函数并不总是安全的，因为它可能会导致接收到的数据丢失。

考虑以下例子，它来自[`docs.python-requests.org/en/latest/user/advanced/#streaming-requests`](http://docs.python-requests.org/en/latest/user/advanced/#streaming-requests):

```py
>>> import json
>>> import requests
>>> r = requests.get('http://httpbin.org/stream/4', stream=True)
>>> for line in r.iter_lines():
...     if line:
...         print(json.loads(line) )

```

在前面的例子中，响应包含了一系列数据。借助`iter_lines()`函数，我们尝试通过遍历每一行来打印这些数据。

## 编码

如 HTTP 协议（RFC 7230）中所述，应用程序可以请求服务器以编码格式返回 HTTP 响应。编码过程将响应内容转换为可理解格式，使其易于访问。当 HTTP 头无法返回编码类型时，Requests 将尝试借助`chardet`来假设编码。

如果我们访问请求的响应头，它确实包含`content-type`键。让我们看看响应头的`content-type`：

```py
>>> re = requests.get('http://google.com')
>>> re.headers['content-type']
 'text/html; charset=ISO-8859-1'

```

在前面的例子中，内容类型包含 `'text/html; charset=ISO-8859-1'`。这种情况发生在 Requests 库发现`charset`值为`None`且`'content-type'`值为`'Text'`时。

它遵循 RFC 7230 协议，在这种情况下将`charset`的值更改为`ISO-8859-1`。如果我们处理的是像`'utf-8'`这样的不同编码类型，我们可以通过将属性设置为`Response.encoding`来显式指定编码。

## HTTP 动词

请求支持使用以下表中定义的完整范围的 HTTP 动词。在使用这些动词的大多数情况下，`'url'` 是必须传递的唯一参数。

| 方法 | 描述 |
| --- | --- |
| GET | GET 方法请求获取指定资源的表示形式。除了检索数据外，使用此方法不会产生其他效果。定义如下 `requests.get(url, **kwargs)` |
| POST | POST 方法用于创建新的资源。提交的 `data` 将由服务器处理到指定的资源。定义如下 `requests.post(url, data=None, json=None, **kwargs)` |
| PUT | 此方法上传指定 URI 的表示。如果 URI 不指向任何资源，服务器可以使用给定的`data`创建一个新的对象，或者修改现有的资源。定义如下`requests.put(url, data=None, **kwargs)` |
| 删除 | 这很容易理解。它用于删除指定的资源。定义如下 `requests.delete(url, **kwargs)` |
| 头部 | 此动词用于检索响应头中写入的元信息，而无需获取响应体。定义如下 `requests.head(url, **kwargs)` |
| 选项 | 选项是一个 HTTP 方法，它返回服务器为指定 URL 支持的 HTTP 方法。定义如下 `requests.options(url, **kwargs)` |
| PATCH | 此方法用于对资源应用部分修改。定义如下 `requests.patch(url, data=None, **kwargs)` |

# 使用链接头描述 API

以访问一个信息分布在不同页面的资源为例。如果我们需要访问资源的下一页，我们可以利用链接头。链接头包含了请求资源的元数据，即在我们这个例子中的下一页信息。

```py
>>> url = "https://api.github.com/search/code?q=addClass+user:mozilla&page=1&per_page=4"
>>> response = requests.head(url=url)
>>> response.headers['link']
'<https://api.github.com/search/code?q=addClass+user%3Amozilla&page=2&per_page=4>; rel="next", <https://api.github.com/search/code?q=addClass+user%3Amozilla&page=250&per_page=4>; rel="last"

```

在前面的例子中，我们在 URL 中指定了想要访问第一页，并且该页应包含四条记录。Requests 会自动解析链接头并更新关于下一页的信息。当我们尝试访问链接头时，它显示了包含页码和每页记录数的输出值。

# 传输适配器

它用于为 Requests 会话提供一个接口，以便连接到 HTTP 和 HTTPS。这将帮助我们模拟网络服务以满足我们的需求。借助传输适配器，我们可以根据我们选择的 HTTP 服务来配置请求。Requests 包含一个名为**HTTPAdapter**的传输适配器。

考虑以下示例：

```py
>>> session = requests.Session()
>>> adapter = requests.adapters.HTTPAdapter(max_retries=6)
>>> session.mount("http://google.co.in", adapter)

```

在这个例子中，我们创建了一个请求会话，在这个会话中，每次请求在连接失败时只会重试六次。

# 摘要

在本章中，我们学习了创建会话以及根据不同标准使用会话。我们还深入探讨了 HTTP 动词和代理的使用。我们了解了流式请求、处理 SSL 证书验证和流式响应。此外，我们还了解了如何使用预定义请求、链接头和分块编码请求。

在下一章中，我们将学习各种认证类型以及如何使用 Requests 库来应用它们。
