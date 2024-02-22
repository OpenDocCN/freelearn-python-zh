# 第十一章：Web 开发

在本章中，我们将介绍以下配方：

+   处理 JSON - 如何解析和编写 JSON 对象

+   解析 URL - 如何解析 URL 的路径、查询和其他部分

+   消费 HTTP - 如何从 HTTP 端点读取数据

+   提交表单到 HTTP - 如何将 HTML 表单提交到 HTTP 端点

+   构建 HTML - 如何生成带有适当转义的 HTML

+   提供 HTTP - 在 HTTP 上提供动态内容

+   提供静态文件 - 如何通过 HTTP 提供静态文件

+   Web 应用程序中的错误 - 如何报告 Web 应用程序中的错误

+   处理表单和文件 - 解析从 HTML 表单和上传的文件接收到的数据

+   REST API - 提供基本的 REST/JSON API

+   处理 cookies - 如何处理 cookies 以识别返回用户

# 介绍

HTTP 协议，更一般地说，Web 技术集，被认为是创建分布式系统的一种有效和健壮的方式，可以利用一种广泛和可靠的方式来实现进程间通信，具有可用的技术和缓存、错误传播、可重复请求的范例，以及在服务可能失败而不影响整体系统状态的情况下的最佳实践。

Python 有许多非常好的和可靠的 Web 框架，从全栈解决方案，如 Django 和 TurboGears，到更精细调整的框架，如 Pyramid 和 Flask。然而，对于许多情况来说，标准库可能已经提供了您需要实现基于 HTTP 的软件的工具，而无需依赖外部库和框架。

在本章中，我们将介绍标准库提供的一些常见配方和工具，这些工具在 HTTP 和基于 Web 的应用程序的上下文中非常方便。

# 处理 JSON

在使用基于 Web 的解决方案时，最常见的需求之一是解析和处理 JSON。Python 内置支持 XML 和 HTML，还支持 JSON 编码和解码。

JSON 编码器也可以被专门化以处理非标准类型，如日期。

# 如何做...

对于这个配方，需要执行以下步骤：

1.  `JSONEncoder`和`JSONDecoder`类可以被专门化以实现自定义的编码和解码行为：

```py
import json
import datetime
import decimal
import types

class CustomJSONEncoder(json.JSONEncoder):
    """JSON Encoder with support for additional types.

    Supports dates, times, decimals, generators and
    any custom class that implements __json__ method.
    """
    def default(self, obj):
        if hasattr(obj, '__json__') and callable(obj.__json__):
            return obj.__json__()
        elif isinstance(obj, (datetime.datetime, datetime.time)):
            return obj.replace(microsecond=0).isoformat()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, types.GeneratorType):
            return list(obj)
        else:
            return super().default(obj)
```

1.  然后，我们可以将我们的自定义编码器传递给`json.dumps`，以根据我们的规则对 JSON 输出进行编码：

```py
jsonstr = json.dumps({'s': 'Hello World',
                    'dt': datetime.datetime.utcnow(),
                    't': datetime.datetime.utcnow().time(),
                    'g': (i for i in range(5)),
                    'd': datetime.date.today(),
                    'dct': {
                        's': 'SubDict',
                        'dt': datetime.datetime.utcnow()
                    }}, 
                    cls=CustomJSONEncoder)

>>> print(jsonstr)
{"t": "10:53:53", 
 "s": "Hello World", 
 "d": "2018-06-29", 
 "dt": "2018-06-29T10:53:53", 
 "dct": {"dt": "2018-06-29T10:53:53", "s": "SubDict"}, 
 "g": [0, 1, 2, 3, 4]}
```

1.  只要提供了`__json__`方法，我们也可以对任何自定义类进行编码：

```py
class Person:
    def __init__(self, name, surname):
        self.name = name
        self.surname = surname

    def __json__(self):
        return {
            'name': self.name,
            'surname': self.surname
        }
```

1.  结果将是一个包含提供数据的 JSON 对象：

```py
>>> print(json.dumps({'person': Person('Simone', 'Marzola')}, 
                     cls=CustomJSONEncoder))
{"person": {"name": "Simone", "surname": "Marzola"}}
```

1.  加载回编码值将导致纯字符串被解码，因为它们不是 JSON 类型：

```py
>>> print(json.loads(jsonstr))
{'g': [0, 1, 2, 3, 4], 
 'd': '2018-06-29', 
 's': 'Hello World', 
 'dct': {'s': 'SubDict', 'dt': '2018-06-29T10:56:30'}, 
 't': '10:56:30', 
 'dt': '2018-06-29T10:56:30'}
```

1.  如果我们还想解析回日期，我们可以尝试专门化`JSONDecoder`来猜测字符串是否包含 ISO 8601 格式的日期，并尝试解析它：

```py
class CustomJSONDecoder(json.JSONDecoder):
    """Custom JSON Decoder that tries to decode additional types.

    Decoder tries to guess dates, times and datetimes in ISO format.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, object_hook=self.parse_object
        )

    def parse_object(self, values):
        for k, v in values.items():
            if not isinstance(v, str):
                continue

            if len(v) == 10 and v.count('-') == 2:
                # Probably contains a date
                try:
                    values[k] = datetime.datetime.strptime(v, '%Y-
                    %m-%d').date()
                except:
                    pass
            elif len(v) == 8 and v.count(':') == 2:
                # Probably contains a time
                try:
                    values[k] = datetime.datetime.strptime(v, 
                    '%H:%M:%S').time()
                except:
                    pass
            elif (len(v) == 19 and v.count('-') == 2 and 
                v.count('T') == 1 and v.count(':') == 2):
                # Probably contains a datetime
                try:
                    values[k] = datetime.datetime.strptime(v, '%Y-
                    %m-%dT%H:%M:%S')
                except:
                    pass
        return values
```

1.  回到以前的数据应该导致预期的类型：

```py
>>> jsondoc = json.loads(jsonstr, cls=CustomJSONDecoder)
>>> print(jsondoc)
{'g': [0, 1, 2, 3, 4], 
 'd': datetime.date(2018, 6, 29), 
 's': 'Hello World', 
 'dct': {'s': 'SubDict', 'dt': datetime.datetime(2018, 6, 29, 10, 56, 30)},
 't': datetime.time(10, 56, 30), 
 'dt': datetime.datetime(2018, 6, 29, 10, 56, 30)}
```

# 它是如何工作的...

要生成 Python 对象的 JSON 表示，使用`json.dumps`方法。该方法接受一个额外的参数`cls`，可以提供自定义编码器类：

```py
json.dumps({'key': 'value', cls=CustomJSONEncoder)
```

每当需要编码编码器不知道如何编码的对象时，提供的类的`default`方法将被调用。

我们的`CustomJSONEncoder`类提供了一个`default`方法，用于处理编码日期、时间、生成器、小数和任何提供`__json__`方法的自定义类：

```py
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__json__') and callable(obj.__json__):
            return obj.__json__()
        elif isinstance(obj, (datetime.datetime, datetime.time)):
            return obj.replace(microsecond=0).isoformat()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, types.GeneratorType):
            return list(obj)
        else:
            return super().default(obj)
```

这是通过依次检查编码对象的属性来完成的。请记住，编码器知道如何编码的对象不会被提供给`default`方法；只有编码器不知道如何处理的对象才会传递给`default`方法。

因此，我们只需要检查我们想要支持的对象，而不是标准对象。

我们的第一个检查是验证提供的对象是否有`__json__`方法：

```py
if hasattr(obj, '__json__') and callable(obj.__json__):
    return obj.__json__()
```

对于具有`__json__`属性的任何对象，该属性是可调用的，我们将依赖调用它来检索对象的 JSON 表示。`__json__`方法所需做的就是返回任何 JSON 编码器知道如何编码的对象，通常是一个`dict`，其中对象的属性将被存储。

对于日期的情况，我们将使用简化的 ISO 8601 格式对其进行编码：

```py
elif isinstance(obj, (datetime.datetime, datetime.time)):
    return obj.replace(microsecond=0).isoformat()
elif isinstance(obj, datetime.date):
    return obj.isoformat()
```

这通常允许来自客户端的轻松解析，例如 JavaScript 解释器可能需要从提供的数据中构建`date`对象。

`Decimal`只是为了方便转换为浮点数。这在大多数情况下足够了，并且与任何 JSON 解码器完全兼容，无需任何额外的机制。当然，我们可以返回更复杂的对象，例如字典，以保留固定的精度：

```py
elif isinstance(obj, decimal.Decimal):
    return float(obj)
```

最后，生成器被消耗，并从中返回包含的值的列表。这通常是您所期望的，表示生成器逻辑本身将需要不合理的努力来保证跨语言的兼容性：

```py
elif isinstance(obj, types.GeneratorType):
    return list(obj)
```

对于我们不知道如何处理的任何对象，我们只需让父对象实现`default`方法并继续：

```py
else:
    return super().default(obj)
```

这将只是抱怨对象不可 JSON 序列化，并通知开发人员我们不知道如何处理它。

自定义解码器支持的工作方式略有不同。

虽然编码器将接收它知道的对象和它不知道的对象（因为 Python 对象比 JSON 对象更丰富），但很容易看出它只能请求对它不知道的对象进行额外的指导，并对它知道如何处理的对象以标准方式进行处理。

解码器只接收有效的 JSON 对象；否则，提供的字符串根本就不是有效的 JSON。

它如何知道提供的字符串必须解码为普通字符串，还是应该要求额外的指导？

它不能，因此它要求对任何单个解码的对象进行指导。

这就是为什么解码器基于一个`object_hook`可调用，它将接收每个单独解码的 JSON 对象，并可以检查它以执行其他转换，或者如果正常解码是正确的，它可以让它继续。

在我们的实现中，我们对解码器进行了子类化，并提供了一个基于本地类方法`parse_object`的默认`object_hook`参数：

```py
class CustomJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, object_hook=self.parse_object
        )
```

然后，`parse_object`方法将接收到解码 JSON（顶级或嵌套的）中找到的任何 JSON 对象；因此，它将接收到一堆字典，可以以任何需要的方式检查它们，并编辑它们的内容以执行 JSON 解码器本身执行的其他转换：

```py
def parse_object(self, values):
    for k, v in values.items():
        if not isinstance(v, str):
            continue

        if len(v) == 10 and v.count('-') == 2:
            # Probably contains a date
            try:
                values[k] = datetime.datetime.strptime(v, '%Y-%m-
                %d').date()
            except:
                pass
        elif len(v) == 8 and v.count(':') == 2:
            # Probably contains a time
            try:
                values[k] = datetime.datetime.strptime(v, 
                '%H:%M:%S').time()
            except:
                pass
        elif (len(v) == 19 and v.count('-') == 2 and 
            v.count('T') == 1 and v.count(':') == 2):
            # Probably contains a datetime
            try:
                values[k] = datetime.datetime.strptime(v, '%Y-%m-
                %dT%H:%M:%S')
            except:
                pass
    return values
```

接收到的参数实际上是一个完整的 JSON 对象，因此它永远不会是单个字段；它总是一个对象（因此，一个完整的 Python 字典，具有多个键值）。

看看以下对象：

```py
{'g': [0, 1, 2, 3, 4], 
 'd': '2018-06-29', 
 's': 'Hello World', 
```

您不会收到一个`g`键，但您将收到整个 Python 字典。这意味着如果您的 JSON 文档没有嵌套的 JSON 对象，您的`object_hook`将被调用一次，并且不会再有其他调用。

因此，我们的`parse_object`方法提供的自定义`object_hook`会迭代解码后的 JSON 对象的所有属性：

```py
for k, v in values.items():
    if not isinstance(v, str):
        continue
```

由于 JSON 中的日期和时间通常以 ISO 8601 格式的字符串表示，因此它会忽略一切不是字符串的内容。

我们对数字、列表和字典的转换非常满意（如果您期望日期被放在列表中，可能需要转到列表），因此如果值不是字符串，我们就跳过它。

当值是字符串时，我们检查其属性，如果我们猜测它可能是日期，我们尝试将其解析为日期。

我们可以考虑日期的正确定义：由两个破折号分隔的三个值，后跟由两个冒号分隔的三个值，中间有一个"T"来分隔两个值：

```py
elif (len(v) == 19 and v.count('-') == 2 and 
      v.count('T') == 1 and v.count(':') == 2):
    # Probably contains a datetime
```

如果匹配该定义，我们实际上会尝试将其解码为 Python 的`datetime`对象，并在解码后的 JSON 对象中替换该值：

```py
# Probably contains a datetime
try:
    values[k] = datetime.datetime.strptime(v, '%Y-%m-%dT%H:%M:%S')
except:
    pass
```

# 还有更多...

您可能已经注意到，将 Python 编码为 JSON 是相当合理和健壮的，但返回的过程中充满了问题。

JSON 不是一种非常表达性的语言；它不提供任何用于自定义类型的机制，因此您有一种标准方法可以向解码器提供关于您期望将某些内容解码为的类型的提示。

虽然我们可以*猜测*像`2017-01-01T13:21:17`这样的东西是一个日期，但我们根本没有任何保证。也许最初它实际上是一些文本，碰巧包含可以解码为日期的内容，但从未打算成为 Python 中的`datetime`对象。

因此，通常只在受限环境中实现自定义解码是安全的。如果您知道并控制将接收数据的源，通常可以安全地提供自定义解码。您可能希望通过使用自定义属性来扩展 JSON，这些属性可能会指导解码器（例如具有告诉您它是日期还是字符串的`__type__`键），但在开放的网络世界中，通常不明智地尝试猜测人们发送给您的内容，因为网络非常多样化。

有一些扩展的标准 JSON 版本试图解决解码数据中的这种歧义，例如 JSON-LD 和 JSON Schema，它们允许您在 JSON 中表示更复杂的实体。

如果有必要，您应该依赖这些标准，以避免重新发明轮子的风险，并面对您的解决方案已经由现有标准解决的限制。

# 解析 URL

在处理基于 Web 的软件时，经常需要了解链接、协议和路径。

您可能会倾向于依赖正则表达式或字符串拆分来解析 URL，但是如果考虑到 URL 可能包含的所有奇特之处（例如凭据或特定协议等），它可能并不像您期望的那样容易。

Python 提供了`urllib`和`cgi`模块中的实用工具，当您想要考虑 URL 可能具有的所有可能不同的格式时，这些工具可以使生活更轻松。

依靠它们可以使生活更轻松，使您的软件更健壮。

# 如何做...

`urllib.parse`模块具有多种工具可用于解析 URL。最常用的解决方案是依赖于`urllib.parse.urlparse`，它可以处理最常见的 URL 类型：

```py
import urllib.parse

def parse_url(url):
    """Parses an URL of the most widespread format.

    This takes for granted there is a single set of parameters
    for the whole path.
    """
    parts = urllib.parse.urlparse(url)
    parsed = vars(parts)
    parsed['query'] = urllib.parse.parse_qs(parts.query)
    return parsed
```

可以在命令行上调用前面的代码片段，如下所示：

```py
>>> url = 'http://user:pwd@host.com:80/path/subpath?arg1=val1&arg2=val2#fragment'
>>> result = parse_url(url)
>>> print(result)
OrderedDict([('scheme', 'http'),
             ('netloc', 'user:pwd@host.com:80'),
             ('path', '/path/subpath'),
             ('params', ''),
             ('query', {'arg1': ['val1'], 'arg2': ['val2']}),
             ('fragment', 'fragment')])
```

返回的`OrderedDict`包含组成我们的 URL 的所有部分，并且对于查询参数，它们已经被解析。

# 还有更多...

如今，URI 还支持在每个路径段中提供参数。这在实践中很少使用，但如果您的代码预期接收此类 URI，则不应依赖于`urllib.parse.urlparse`，因为它尝试从 URL 中解析参数，而这对于这些 URI 来说并不受支持：

```py
>>> url = 'http://user:pwd@host.com:80/root;para1/subpath;para2?arg1=val1#fragment'
>>> result = urllib.parse.urlparse(url)
>>> print(result)
ParseResult(scheme='http', netloc='user:pwd@host.com:80', 
            path='/root;para1/subpath', 
            params='para2', 
            query='arg1=val1', 
            fragment='fragment')
```

您可能已经注意到，路径的最后一部分的参数在`params`中被正确解析，但是第一部分的参数保留在`path`中。

在这种情况下，您可能希望依赖于`urllib.parse.urlsplit`，它不会解析参数，而会将它们保留下来供您解析。因此，您可以自行拆分 URL 段和参数：

```py
>>> parsed = urllib.parse.urlsplit(url)
>>> print(parsed)
SplitResult(scheme='http', netloc='user:pwd@host.com:80', 
            path='/root;para1/subpath;para2', 
            query='arg1=val1', 
            fragment='fragment')
```

请注意，在这种情况下，所有参数都保留在“路径”中，然后您可以自行拆分它们。

# HTTP 消费

您可能正在与基于 HTTP REST API 的第三方服务进行交互，或者可能正在从第三方获取内容或仅下载软件需要的文件。这并不重要。如今，几乎不可能编写一个应用程序并忽略 HTTP；您迟早都会面对它。人们期望各种应用程序都支持 HTTP。如果您正在编写图像查看器，他们可能希望能够将指向图像的 URL 传递给它并看到图像出现。

虽然它们从来没有真正用户友好和明显，但 Python 标准库一直有与 HTTP 交互的方式，并且这些方式可以直接使用。

# 如何做到这一点...

此处的步骤如下：

1.  `urllib.request`模块提供了提交 HTTP 请求所需的机制。它的轻量级包装可以解决大多数 HTTP 使用需求：

```py
import urllib.request
import urllib.parse
import json

def http_request(url, query=None, method=None, headers={}, data=None):
    """Perform an HTTP request and return the associated response."""
    parts = vars(urllib.parse.urlparse(url))
    if query:
        parts['query'] = urllib.parse.urlencode(query)

    url = urllib.parse.ParseResult(**parts).geturl()
    r = urllib.request.Request(url=url, method=method, 
                            headers=headers,
                            data=data)
    with urllib.request.urlopen(r) as resp:
        msg, resp = resp.info(), resp.read()

    if msg.get_content_type() == 'application/json':
        resp = json.loads(resp.decode('utf-8'))

    return msg, resp
```

1.  我们可以使用我们的`http_request`函数执行请求以获取文件：

```py
>>> msg, resp = http_request('https://httpbin.org/bytes/16')
>>> print(msg.get_content_type(), resp)
application/octet-stream b'k\xe3\x05\x06=\x17\x1a9%#\xd0\xae\xd8\xdc\xf9>'
```

1.  我们还可以使用它与基于 JSON 的 API 进行交互：

```py
>>> msg, resp = http_request('https://httpbin.org/get', query={
...     'a': 'Hello',
...     'b': 'World'
... })
>>> print(msg.get_content_type(), resp)
application/json
{'url': 'https://httpbin.org/get?a=Hello&b=World', 
 'headers': {'Accept-Encoding': 'identity', 
             'User-Agent': 'Python-urllib/3.5', 
             'Connection': 'close', 
             'Host': 'httpbin.org'}, 
 'args': {'a': 'Hello', 'b': 'World'}, 
 'origin': '127.19.102.123'}
```

1.  它还可以用于提交或上传数据到端点：

```py
>>> msg, resp = http_request('https://httpbin.org/post', method='POST',
...                          data='This is my posted data!'.encode('ascii'),
...                          headers={'Content-Type': 'text/plain'})
>>> print(msg.get_content_type(), resp)
application/json 
{'data': 'This is my posted data!', 
 'json': None, 
 'form': {}, 
 'args': {}, 
 'files': {}, 
 'headers': {'User-Agent': 'Python-urllib/3.5', 
             'Connection': 'close', 
             'Content-Type': 'text/plain', 
             'Host': 'httpbin.org', 
             'Accept-Encoding': 'identity', 
             'Content-Length': '23'}, 
 'url': 'https://httpbin.org/post', 
 'origin': '127.19.102.123'}
```

# 它是如何工作的...

`http_request`方法负责创建`urllib.request.Request`实例，通过网络发送它并获取响应。

向指定的 URL 发送请求，其中附加了查询参数。

函数的第一件事是解析 URL，以便能够替换其中的部分。这样做是为了能够用提供的部分替换/追加查询参数：

```py
parts = vars(urllib.parse.urlparse(url))
if query:
    parts['query'] = urllib.parse.urlencode(query)
```

`urllib.parse.urlencode`将接受一个参数字典，例如`{'a': 5, 'b': 7}`，并将返回带有`urlencode`参数的字符串：`'b=7&a=5'`。

然后，将生成的查询字符串放入`url`的解析部分中，以替换当前存在的查询参数。

然后，从现在包括正确查询参数的所有部分构建`url`：

```py
url = urllib.parse.ParseResult(**parts).geturl()
```

一旦准备好带有编码查询的`url`，它就会构建一个请求，代理指定的方法、标头和请求的主体：

```py
r = urllib.request.Request(url=url, method=method, headers=headers,
                           data=data)
```

在进行普通的`GET`请求时，这些将是默认的，但能够指定它们允许我们执行更高级的请求，例如`POST`，或者在我们的请求中提供特殊的标头。

然后打开请求并读取响应：

```py
with urllib.request.urlopen(r) as resp:
    msg, resp = resp.info(), resp.read()
```

响应以`urllib.response.addinfourl`对象的形式返回，其中包括两个相关部分：响应的主体和一个`http.client.HTTPMessage`，我们可以从中获取所有响应信息，如标头、URL 等。

通过像读取文件一样读取响应来检索主体，而通过`info()`方法检索`HTTPMessage`。

通过检索的信息，我们可以检查响应是否为 JSON 响应，在这种情况下，我们将其解码回字典，以便我们可以浏览响应而不仅仅是接收纯字节：

```py
if msg.get_content_type() == 'application/json':
    resp = json.loads(resp.decode('utf-8'))
```

对于所有响应，我们返回消息和主体。如果不需要，调用者可以忽略消息：

```py
return msg, resp
```

# 还有更多...

对于简单的情况来说，进行 HTTP 请求可能非常简单，但对于更复杂的情况来说可能非常复杂。完美地处理 HTTP 协议可能是一项漫长而复杂的工作，特别是因为协议规范本身并不总是清楚地规定事物应该如何工作，很多都来自于对现有的网络服务器和客户端工作方式的经验。

因此，如果您的需求超出了仅仅获取简单端点的范围，您可能希望依赖于第三方库来执行 HTTP 请求，例如几乎适用于所有 Python 环境的 requests 库。

# 向 HTTP 提交表单

有时您必须与 HTML 表单交互或上传文件。这通常需要处理`multipart/form-data`编码。

表单可以混合文件和文本数据，并且表单中可以有多个不同的字段。因此，它需要一种方式来在同一个请求中表示多个字段，其中一些字段可以是二进制文件。

这就是为什么在多部分中编码数据可能会变得棘手，但是可以使用标准库工具来制定一个基本的食谱，以便在大多数情况下都能正常工作。

# 如何做到这一点...

以下是此食谱的步骤：

1.  `multipart`本身需要跟踪我们想要编码的所有字段和文件，然后执行编码本身。

1.  我们将依赖`io.BytesIO`来存储所有生成的字节：

```py
import io
import mimetypes
import uuid

class MultiPartForm:
    def __init__(self):
        self.fields = {}
        self.files = []

    def __setitem__(self, name, value):
        self.fields[name] = value

    def add_file(self, field, filename, data, mimetype=None):
        if mimetype is None:
            mimetype = (mimetypes.guess_type(filename)[0] or
                        'application/octet-stream')
        self.files.append((field, filename, mimetype, data))

    def _generate_bytes(self, boundary):
        buffer = io.BytesIO()
        for field, value in self.fields.items():
            buffer.write(b'--' + boundary + b'\r\n')
            buffer.write('Content-Disposition: form-data; '
                        'name="{}"\r\n'.format(field).encode('utf-8'))
            buffer.write(b'\r\n')
            buffer.write(value.encode('utf-8'))
            buffer.write(b'\r\n')
        for field, filename, f_content_type, body in self.files:
            buffer.write(b'--' + boundary + b'\r\n')
            buffer.write('Content-Disposition: file; '
                        'name="{}"; filename="{}"\r\n'.format(
                            field, filename
                        ).encode('utf-8'))
            buffer.write('Content-Type: {}\r\n'.format(
                f_content_type
            ).encode('utf-8'))
            buffer.write(b'\r\n')
            buffer.write(body)
            buffer.write(b'\r\n')
        buffer.write(b'--' + boundary + b'--\r\n')
        return buffer.getvalue()

    def encode(self):
        boundary = uuid.uuid4().hex.encode('ascii')
        while boundary in self._generate_bytes(boundary=b'NOBOUNDARY'):
            boundary = uuid.uuid4().hex.encode('ascii')

        content_type = 'multipart/form-data; boundary={}'.format(
            boundary.decode('ascii')
        )
        return content_type, self._generate_bytes(boundary)
```

1.  然后我们可以提供并编码我们的`form`数据：

```py
>>> form = MultiPartForm()
>>> form['name'] = 'value'
>>> form.add_file('file1', 'somefile.txt', b'Some Content', 'text/plain')
>>> content_type, form_body = form.encode()
>>> print(content_type, '\n\n', form_body.decode('ascii'))
multipart/form-data; boundary=6c5109dfa19a450695013d4eecac2b0b 

--6c5109dfa19a450695013d4eecac2b0b
Content-Disposition: form-data; name="name"

value
--6c5109dfa19a450695013d4eecac2b0b
Content-Disposition: file; name="file1"; filename="somefile.txt"
Content-Type: text/plain

Some Content
--6c5109dfa19a450695013d4eecac2b0b--
```

1.  使用我们先前食谱中的`http_request`方法，我们可以通过 HTTP 提交任何`form`：

```py
>>> _, resp = http_request('https://httpbin.org/post', method='POST',
                           data=form_body, 
                           headers={'Content-Type': content_type})
>>> print(resp)
{'headers': {
    'Accept-Encoding': 'identity', 
    'Content-Type': 'multipart/form-data; boundary=6c5109dfa19a450695013d4eecac2b0b', 
    'User-Agent': 'Python-urllib/3.5', 
    'Content-Length': '272', 
    'Connection': 'close', 
    'Host': 'httpbin.org'
 }, 
 'json': None,
 'url': 'https://httpbin.org/post', 
 'data': '', 
 'args': {}, 
 'form': {'name': 'value'}, 
 'origin': '127.69.102.121', 
 'files': {'file1': 'Some Content'}}
```

正如你所看到的，`httpbin`正确接收了我们的`file1`和我们的`name`字段，并对两者进行了处理。

# 工作原理...

`multipart`实际上是基于在单个主体内编码多个请求。每个部分都由一个**boundary**分隔，而在边界内则是该部分的数据。

每个部分都可以提供数据和元数据，例如所提供数据的内容类型。

这样接收者就可以知道所包含的数据是二进制、文本还是其他类型。例如，指定`form`的`surname`字段值的部分将如下所示：

```py
Content-Disposition: form-data; name="surname"

MySurname
```

提供上传文件数据的部分将如下所示：

```py
Content-Disposition: file; name="file1"; filename="somefile.txt"
Content-Type: text/plain

Some Content
```

我们的`MultiPartForm`允许我们通过字典语法存储纯`form`字段：

```py
def __setitem__(self, name, value):
    self.fields[name] = value
```

我们可以在命令行上调用它，如下所示：

```py
>>> form['name'] = 'value'
```

并通过`add_file`方法提供文件：

```py
def add_file(self, field, filename, data, mimetype=None):
    if mimetype is None:
        mimetype = (mimetypes.guess_type(filename)[0] or
                    'application/octet-stream')
    self.files.append((field, filename, mimetype, data))
```

我们可以在命令行上调用这个方法，如下所示：

```py
>>> form.add_file('file1', 'somefile.txt', b'Some Content', 'text/plain')
```

这些只是在稍后调用`_generate_bytes`时才会使用的字典和列表，用于记录想要的字段和文件。

所有的辛苦工作都是由`_generate_bytes`完成的，它会遍历所有这些字段和文件，并为每一个创建一个部分：

```py
for field, value in self.fields.items():
    buffer.write(b'--' + boundary + b'\r\n')
    buffer.write('Content-Disposition: form-data; '
                'name="{}"\r\n'.format(field).encode('utf-8'))
    buffer.write(b'\r\n')
    buffer.write(value.encode('utf-8'))
    buffer.write(b'\r\n')
```

由于边界必须分隔每个部分，非常重要的是要验证边界是否不包含在数据本身中，否则接收者可能会在遇到它时错误地认为部分已经结束。

这就是为什么我们的`MultiPartForm`类会生成一个`boundary`，检查它是否包含在多部分响应中，如果是，则生成一个新的，直到找到一个不包含在数据中的`boundary`：

```py
boundary = uuid.uuid4().hex.encode('ascii')
while boundary in self._generate_bytes(boundary=b'NOBOUNDARY'):
    boundary = uuid.uuid4().hex.encode('ascii')
```

一旦我们找到了一个有效的`boundary`，我们就可以使用它来生成多部分内容，并将其返回给调用者，同时提供必须使用的内容类型（因为内容类型为接收者提供了关于要检查的`boundary`的提示）：

```py
content_type = 'multipart/form-data; boundary={}'.format(
    boundary.decode('ascii')
)
return content_type, self._generate_bytes(boundary)
```

# 还有更多...

多部分编码并不是一个简单的主题；例如，在多部分主体中对名称的编码并不是一个简单的话题。

多年来，关于在多部分内容中对字段名称和文件名称进行正确编码的方式已经多次更改和讨论。

从历史上看，在这些字段中只依赖于纯 ASCII 名称是安全的，因此，如果您想确保您提交的数据的服务器能够正确接收您的数据，您可能希望坚持使用简单的文件名和字段，不涉及 Unicode 字符。

多年来，提出了多种其他编码这些字段和文件名的方法。UTF-8 是 HTML5 的官方支持的后备之一。建议的食谱依赖于 UTF-8 来编码文件名和字段，以便与使用纯 ASCII 名称的情况兼容，但仍然可以在服务器支持它们时依赖于 Unicode 字符。

# 构建 HTML

每当您构建网页、电子邮件或报告时，您可能会依赖用实际值替换 HTML 模板中的占位符，以便向用户显示所需的内容。

我们已经在第二章中看到了*文本管理*，如何实现一个最小的简单模板引擎，但它并不特定于 HTML。

在处理 HTML 时，特别重要的是要注意对用户提供的值进行转义，因为这可能导致页面损坏甚至 XSS 攻击。

显然，您不希望您的用户因为您在网站上注册时使用姓氏`"<script>alert('You are hacked!')</script>"`而对您生气。

出于这个原因，Python 标准库提供了可以用于正确准备内容以插入 HTML 的转义工具。

# 如何做...

结合`string.Formatter`和`cgi`模块，可以创建一个负责为我们进行转义的格式化程序：

```py
import string
import cgi

class HTMLFormatter(string.Formatter):
    def get_field(self, field_name, args, kwargs):
        val, key = super().get_field(field_name, args, kwargs)
        if hasattr(val, '__html__'):
            val = val.__html__()
        elif isinstance(val, str):
            val = cgi.escape(val)
        return val, key

class Markup:
    def __init__(self, v):
        self.v = v
    def __str__(self):
        return self.v
    def __html__(self):
        return str(self)
```

然后我们可以在需要时使用`HTMLFormatter`和`Markup`类，同时保留注入原始`html`的能力：

```py
>>> html = HTMLFormatter().format('Hello {name}, you are {title}', 
                                  name='<strong>Name</strong>',
                                  title=Markup('<em>a developer</em>'))
>>> print(html)
Hello &lt;strong&gt;Name&lt;/strong&gt;, you are <em>a developer</em>
```

我们还可以轻松地将此配方与有关文本模板引擎的配方相结合，以实现一个具有转义功能的极简 HTML 模板引擎。

# 它是如何工作的...

每当`HTMLFormatter`需要替换格式字符串中的值时，它将检查检索到的值是否具有`__html__`方法：

```py
if hasattr(val, '__html__'):
    val = val.__html__()
```

如果存在该方法，则预计返回值的 HTML 表示。并且预计是一个完全有效和转义的 HTML。

否则，预计值将是需要转义的字符串：

```py
elif isinstance(val, str):
    val = cgi.escape(val)
```

这样，我们提供给`HTMLFormatter`的任何值都会默认进行转义：

```py
>>> html = HTMLFormatter().format('Hello {name}', 
                                  name='<strong>Name</strong>')
>>> print(html)
Hello &lt;strong&gt;Name&lt;/strong&gt;
```

如果我们想要避免转义，我们可以依赖`Markup`对象，它可以包装一个字符串，使其原样传递而不进行任何转义：

```py
>>> html = HTMLFormatter().format('Hello {name}', 
                                  name=Markup('<strong>Name</strong>'))
>>> print(html)
Hello <strong>Name</strong>
```

这是因为我们的`Markup`对象实现了一个`__html__`方法，该方法返回原样的字符串。由于我们的`HTMLFormatter`忽略了任何具有`__html__`方法的值，因此我们的字符串将无需任何形式的转义而通过。

虽然`Markup`允许我们根据需要禁用转义，但是当我们知道实际上需要 HTML 时，我们可以将 HTML 方法应用于任何其他对象。需要在网页中表示的任何对象都可以提供一个`__html__`方法，并将根据它自动转换为 HTML。

例如，您可以向您的`User`类添加`__html__`，并且每当您想要将用户放在网页中时，您只需要提供`User`实例本身。

# 提供 HTTP

通过 HTTP 进行交互是分布式应用程序或完全分离的软件之间最常见的通信手段之一，也是所有现有 Web 应用程序和基于 Web 的工具的基础。

虽然 Python 有数十个出色的 Web 框架可以满足大多数不同的需求，但标准库本身具有您可能需要实现基本 Web 应用程序的所有基础。

# 如何做...

Python 有一个方便的协议名为 WSGI 来实现基于 HTTP 的应用程序。对于更高级的需求，可能需要一个 Web 框架；对于非常简单的需求，Python 本身内置的`wsgiref`实现可以满足我们的需求：

```py
import re
import inspect
from wsgiref.headers import Headers
from wsgiref.simple_server import make_server
from wsgiref.util import request_uri
from urllib.parse import parse_qs

class WSGIApplication:
    def __init__(self):
        self.routes = []

    def route(self, path):
        def _route_decorator(f):
            self.routes.append((re.compile(path), f))
            return f
        return _route_decorator

    def serve(self):
        httpd = make_server('', 8000, self)
        print("Serving on port 8000...")
        httpd.serve_forever()

    def _not_found(self, environ, resp):
        resp.status = '404 Not Found'
        return b"""<h1>Not Found</h1>"""

    def __call__(self, environ, start_response):
        request = Request(environ)

        routed_action = self._not_found
        for regex, action in self.routes:
            match = regex.fullmatch(request.path)
            if match:
                routed_action = action
                request.urlargs = match.groupdict()
                break

        resp = Response()

        if inspect.isclass(routed_action):
            routed_action = routed_action()
        body = routed_action(request, resp)

        resp.send(start_response)
        return [body]

class Response:
    def __init__(self):
        self.status = '200 OK'
        self.headers = Headers([
            ('Content-Type', 'text/html; charset=utf-8')
        ])

    def send(self, start_response):
        start_response(self.status, self.headers.items())

class Request:
    def __init__(self, environ):
        self.environ = environ
        self.urlargs = {}

    @property
    def path(self):
        return self.environ['PATH_INFO']

    @property
    def query(self):
        return parse_qs(self.environ['QUERY_STRING'])
```

然后我们可以创建一个`WSGIApplication`并向其注册任意数量的路由：

```py
app = WSGIApplication()

@app.route('/')
def index(request, resp):
    return b'Hello World, <a href="/link">Click here</a>'

@app.route('/link')
def link(request, resp):
    return (b'You clicked the link! '
            b'Try <a href="/args?a=1&b=2">Some arguments</a>')

@app.route('/args')
def args(request, resp):
    return (b'You provided %b<br/>'
            b'Try <a href="/name/HelloWorld">URL Arguments</a>' % 
            repr(request.query).encode('utf-8'))

@app.route('/name/(?P<first_name>\\w+)')
def name(request, resp):
    return (b'Your name: %b' % request.urlargs['first_name'].encode('utf-8'))
```

一旦准备就绪，我们只需要提供应用程序：

```py
app.serve()
```

如果一切正常，通过将浏览器指向`http://localhost:8000`，您应该会看到一个 Hello World 文本和一个链接，引导您到进一步提供查询参数，URL 参数并在各种 URL 上提供服务的页面。

# 它是如何工作的...

`WSGIApplication`创建一个负责提供 Web 应用程序本身（`self`）的 WSGI 服务器：

```py
def serve(self):
    httpd = make_server('', 8000, self)
    print("Serving on port 8000...")
    httpd.serve_forever()
```

在每个请求上，服务器都会调用`WSGIApplication.__call__`来检索该请求的响应。

`WSGIApplication.__call__`扫描所有注册的路由（每个路由可以使用`app.route(path)`注册，其中`path`是正则表达式）。当正则表达式与当前 URL 路径匹配时，将调用注册的函数以生成该路由的响应：

```py
def __call__(self, environ, start_response):
    request = Request(environ)

    routed_action = self._not_found
    for regex, action in self.routes:
        match = regex.fullmatch(request.path)
        if match:
            routed_action = action
            request.urlargs = match.groupdict()
            break
```

一旦找到与路径匹配的函数，就会调用该函数以获取响应主体，然后将生成的主体返回给服务器：

```py
resp = Response()
body = routed_action(request, resp)

resp.send(start_response)
return [body]
```

在返回主体之前，将调用`Response.send`通过`start_response`可调用发送响应 HTTP 标头和状态。

`Response`和`Request`对象用于保留当前请求的环境（以及从 URL 解析的任何附加参数）、响应的标头和状态。这样，处理请求的操作可以接收它们并检查请求或在发送之前添加/删除响应的标头。

# 还有更多...

虽然基本的基于 HTTP 的应用程序可以使用提供的`WSGIApplication`实现，但完整功能的应用程序还有很多缺失或不完整的地方。

在涉及更复杂的 Web 应用程序时，通常需要缓存、会话、身份验证、授权、管理数据库连接、事务和管理等部分，并且大多数 Python Web 框架都可以轻松为您提供这些部分。

实现完整的 Web 框架不在本书的范围之内，当 Python 环境中有许多出色的 Web 框架可用时，您可能应该尽量避免重复造轮子。

Python 拥有广泛的 Web 框架，涵盖了从用于快速开发的全栈框架（如 Django）到面向 API 的微框架（如 Flask）以及灵活的解决方案（如 Pyramid 和 TurboGears），其中所需的部分可以根据需要启用、禁用或替换，从全栈解决方案到微框架。

# 提供静态文件

有时在处理基于 JavaScript 的应用程序或静态网站时，有必要能够直接从磁盘上提供目录的内容。

Python 标准库提供了一个现成的 HTTP 服务器，用于处理请求，并将它们映射到目录中的文件，因此我们可以快速地编写自己的 HTTP 服务器来编写网站，而无需安装任何其他工具。

# 如何做...

`http.server`模块提供了实现负责提供目录内容的 HTTP 服务器所需的大部分内容：

```py
import os.path
import socketserver
from http.server import SimpleHTTPRequestHandler, HTTPServer

def serve_directory(path, port=8000):
    class ConfiguredHandler(HTTPDirectoryRequestHandler):
        SERVED_DIRECTORY = path
    httpd = ThreadingHTTPServer(("", port), ConfiguredHandler)
    print("serving on port", port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()

class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    pass

class HTTPDirectoryRequestHandler(SimpleHTTPRequestHandler):
    SERVED_DIRECTORY = '.'

    def translate_path(self, path):
        path = super().translate_path(path)
        relpath = os.path.relpath(path)
        return os.path.join(self.SERVED_DIRECTORY, relpath)
```

然后`serve_directory`可以针对任何路径启动，以在`http://localhost:8000`上提供该路径的内容：

```py
serve_directory('/tmp')
```

将浏览器指向`http://localhost:8000`应该列出`/tmp`目录的内容，并允许您浏览它并查看任何文件的内容。

# 工作原理...

`ThreadingHTTPServer`将`HTTPServer`与`ThreadingMixin`结合在一起，这允许您一次提供多个请求。

这在提供静态网站时尤其重要，因为浏览器经常保持连接时间比需要的更长，当一次只提供一个请求时，您可能无法获取您的 CSS 或 JavaScript 文件，直到浏览器关闭前一个连接。

对于每个请求，`HTTPServer`将其转发到指定的处理程序进行处理。`SimpleHTTPRequestHandler`能够提供请求，将其映射到磁盘上的本地文件，但在大多数 Python 版本中，它只能从当前目录提供服务。

为了能够从任何目录提供请求，我们提供了一个自定义的`translate_path`方法，它替换了相对于`SERVED_DIRECTORY`类变量的标准实现产生的路径。

然后`serve_directory`将所有内容放在一起，并将`HTTPServer`与定制的请求处理程序结合在一起，以创建一个能够处理提供路径的请求的服务器。

# 还有更多...

在较新的 Python 版本中，关于`http.server`模块已经发生了很多变化。最新版本 Python 3.7 已经提供了`ThreadingHTTPServer`类，并且现在可以配置特定目录由`SimpleHTTPRequestHandler`提供服务，因此无需自定义`translate_path`方法来提供特定目录的服务。

# Web 应用程序中的错误

通常，当 Python WSGI Web 应用程序崩溃时，您会在终端中获得一个回溯，浏览器中的路径为空。

这并不是很容易调试发生了什么，除非您明确检查终端，否则很容易错过页面没有显示出来的情况，因为它实际上崩溃了。

幸运的是，Python 标准库为 Web 应用程序提供了一些基本的调试工具，使得可以将崩溃报告到浏览器中，这样您就可以在不离开浏览器的情况下查看并修复它们。

# 如何做...

`cgitb`模块提供了将异常及其回溯格式化为 HTML 的工具，因此我们可以利用它来实现一个 WSGI 中间件，该中间件可以包装任何 Web 应用程序，以在浏览器中提供更好的错误报告：

```py
import cgitb
import sys

class ErrorMiddleware:
    """Wrap a WSGI application to display errors in the browser"""
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        app_iter = None
        try:
            app_iter = self.app(environ, start_response)
            for item in app_iter:
                yield item
        except:
            try:
                start_response('500 INTERNAL SERVER ERROR', [
                    ('Content-Type', 'text/html; charset=utf-8'),
                    ('X-XSS-Protection', '0'),
                ])
            except Exception:
                # There has been output but an error occurred later on. 
                # In that situation we can do nothing fancy anymore, 
                # better log something into the error log and fallback.
                environ['wsgi.errors'].write(
                    'Debugging middleware caught exception in streamed '
                    'response after response headers were already sent.\n'
                )
            else:
                yield cgitb.html(sys.exc_info()).encode('utf-8')
        finally:
            if hasattr(app_iter, 'close'):
                app_iter.close()
```

`ErrorMiddleware`可以用于包装任何 WSGI 应用程序，以便在出现错误时将错误显示在 Web 浏览器中。

例如，我们可以从上一个示例中重新获取我们的`WSGIApplication`，添加一个将导致崩溃的路由，并提供包装后的应用程序以查看错误如何报告到 Web 浏览器中：

```py
from web_06 import WSGIApplication
from wsgiref.simple_server import make_server

app = WSGIApplication()

@app.route('/crash')
def crash(req, resp):
    raise RuntimeError('This is a crash!')

app = ErrorMiddleware(app)

httpd = make_server('', 8000, app)
print("Serving on port 8000...")
httpd.serve_forever()
```

一旦将浏览器指向`http://localhost:8000/crash`，您应该看到触发异常的精美格式的回溯。

# 工作原理...

`ErrorMiddleware`接收原始应用程序并替换请求处理。

所有 HTTP 请求都将被`ErrorMiddleware`接收，然后将其代理到应用程序，返回应用程序提供的结果响应。

如果在消耗应用程序响应时出现异常，它将停止标准流程，而不是进一步消耗应用程序的响应，它将格式化异常并将其作为响应发送回浏览器。

这是因为`ErrorMiddleware.__call__`实际上调用了包装的应用程序并迭代了任何提供的结果：

```py
def __call__(self, environ, start_response):
    app_iter = None
    try:
        app_iter = self.app(environ, start_response)
        for item in app_iter:
            yield item
    ...
```

这种方法适用于返回正常响应的应用程序和返回生成器作为响应的应用程序。

如果在调用应用程序或消耗响应时出现错误，则会捕获错误并尝试使用新的`start_response`来通知服务器错误到浏览器：

```py
except:
    try:
        start_response('500 INTERNAL SERVER ERROR', [
            ('Content-Type', 'text/html; charset=utf-8'),
            ('X-XSS-Protection', '0'),
        ])
```

如果`start_response`失败，这意味着被包装的应用程序已经调用了`start_response`，因此不可能再更改响应状态码或标头。

在这种情况下，由于我们无法再提供精美格式的响应，我们只能退回到在终端上提供错误：

```py
except Exception:
    # There has been output but an error occurred later on. 
    # In that situation we can do nothing fancy anymore, 
    # better log something into the error log and fallback.
    environ['wsgi.errors'].write(
        'Debugging middleware caught exception in streamed '
        'response after response headers were already sent.\n'
    )
```

如果`start_response`成功，我们将停止返回应用程序响应的内容，而是返回错误和回溯，由`cgitb`精美格式化：

```py
else:
    yield cgitb.html(sys.exc_info()).encode('utf-8')
```

在这两种情况下，如果它提供了`close`方法，我们将关闭应用程序响应。这样，如果它是一个需要关闭的文件或任何源，我们就可以避免泄漏它：

```py
finally:
    if hasattr(app_iter, 'close'):
        app_iter.close()
```

# 还有更多...

Python 标准库之外还提供了更完整的 Web 应用程序错误报告解决方案。如果您有进一步的需求或希望通过电子邮件或通过 Sentry 等云错误报告解决方案通知错误，您可能需要提供一个错误报告 WSGI 库。

来自 Flask 的`Werkzeug`调试器，来自 Pylons 项目的`WebError`库，以及来自 TurboGears 项目的`Backlash`库可能是这个目的最常见的解决方案。

您可能还想检查您的 Web 框架是否提供了一些高级的错误报告配置，因为其中许多提供了这些功能，依赖于这些库或其他工具。

# 处理表单和文件

在提交表单和上传文件时，它们通常以`multipart/form-data`编码发送。

我们已经看到如何创建以`multipart/form-data`编码的数据，并将其提交到端点，但是如何处理以这种格式接收的数据呢？

# 如何做...

标准库中的`cgi.FieldStorage`类已经提供了解析多部分数据并以易于处理的方式发送回数据所需的所有机制。

我们将创建一个简单的 Web 应用程序（基于`WSGIApplication`），以展示如何使用`cgi.FieldStorage`来解析上传的文件并将其显示给用户：

```py
import cgi

from web_06 import WSGIApplication
import base64

app = WSGIApplication()

@app.route('/')
def index(req, resp):
    return (
        b'<form action="/upload" method="post" enctype="multipart/form-
           data">'
        b'  <input type="file" name="uploadedfile"/>'
        b'  <input type="submit" value="Upload">'
        b'</form>'
    )

@app.route('/upload')
def upload(req, resp):
    form = cgi.FieldStorage(fp=req.environ['wsgi.input'], 
                            environ=req.environ)
    if 'uploadedfile' not in form:
        return b'Nothing uploaded'

    uploadedfile = form['uploadedfile']
    if uploadedfile.type.startswith('image'):
        # User uploaded an image, show it
        return b'<img src="data:%b;base64,%b"/>' % (
            uploadedfile.type.encode('ascii'),
            base64.b64encode(uploadedfile.file.read())
        )
    elif uploadedfile.type.startswith('text'):
        return uploadedfile.file.read()
    else:
        return b'You uploaded %b' % uploadedfile.filename.encode('utf-8')

app.serve()
```

# 工作原理...

该应用程序公开了两个网页。一个位于网站的根目录（通过`index`函数），只显示一个带有上传字段的简单表单。

另一个`upload`函数，接收上传的文件，如果是图片或文本文件，则显示出来。在其他情况下，它将只显示上传文件的名称。

处理多部分格式上传的唯一要求是创建一个`cgi.FieldStorage`：

```py
form = cgi.FieldStorage(fp=req.environ['wsgi.input'], 
                        environ=req.environ)
```

`POST`请求的整个主体始终在`environ`请求中可用，使用`wsgi.input`键。

这提供了一个类似文件的对象，可以读取以消耗已发布的数据。确保在创建`FieldStorage`后将其保存，如果需要多次使用它，因为一旦从`wsgi.input`中消耗了数据，它就变得不可访问。

`cgi.FieldStorage`提供了类似字典的接口，因此我们可以通过检查`uploadedfile`条目是否存在来检查是否上传了文件：

```py
if 'uploadedfile' not in form:
    return b'Nothing uploaded'
```

这是因为在我们的表单中，我们提供了`uploadedfile`作为字段的名称：

```py
b'  <input type="file" name="uploadedfile"/>'
```

该特定字段将可以通过`form['uploadedfile']`访问。

因为它是一个文件，它将返回一个对象，通过该对象我们可以检查上传文件的 MIME 类型，以确定它是否是一张图片：

```py
if uploadedfile.type.startswith('image'):
```

如果它是一张图片，我们可以读取它的内容，将其编码为`base64`，这样它就可以被`img`标签显示出来：

```py
base64.b64encode(uploadedfile.file.read())
```

`filename`属性仅在上传文件是无法识别的格式时使用，这样我们至少可以打印出上传文件的名称：

```py
return b'You uploaded %b' % uploadedfile.filename.encode('utf-8')
```

# REST API

REST 与 JSON 已成为基于 Web 的应用程序之间的跨应用程序通信技术的事实标准。

这是一个非常有效的协议，而且它的定义可以被每个人理解，这使得它很快就变得流行起来。

与其他更复杂的通信协议相比，快速的 REST 实现可以相对快速地推出。

由于 Python 标准库提供了我们构建基于 WSGI 的应用程序所需的基础，因此很容易扩展我们现有的配方以支持基于 REST 的请求分发。

# 如何做...

我们将使用我们之前的配方中的`WSGIApplication`，但是不是为根注册一个函数，而是注册一个能够根据请求方法进行分发的特定类。

1.  我们想要实现的所有 REST 类都必须继承自单个`RestController`实现：

```py
class RestController:
    def __call__(self, req, resp):
        method = req.environ['REQUEST_METHOD']
        action = getattr(self, method, self._not_found)
        return action(req, resp)

    def _not_found(self, environ, resp):
        resp.status = '404 Not Found'
        return b'{}'  # Provide an empty JSON document
```

1.  然后我们可以子类化`RestController`来实现所有特定的`GET`、`POST`、`DELETE`和`PUT`方法，并在特定路由上注册资源：

```py
import json
from web_06 import WSGIApplication

app = WSGIApplication()

@app.route('/resources/?(?P<id>\\w*)')
class ResourcesRestController(RestController):
    RESOURCES = {}

    def GET(self, req, resp):
        resource_id = req.urlargs['id']
        if not resource_id:
            # Whole catalog requested
            return json.dumps(self.RESOURCES).encode('utf-8')

        if resource_id not in self.RESOURCES:
            return self._not_found(req, resp)

        return json.dumps(self.RESOURCES[resource_id]).encode('utf-8')

    def POST(self, req, resp):
        content_length = int(req.environ['CONTENT_LENGTH'])
        data = req.environ['wsgi.input'].read(content_length).decode('utf-8')

        resource = json.loads(data)
        resource['id'] = str(len(self.RESOURCES)+1)
        self.RESOURCES[resource['id']] = resource
        return json.dumps(resource).encode('utf-8')

    def DELETE(self, req, resp):
        resource_id = req.urlargs['id']
        if not resource_id:
            return self._not_found(req, resp)
        self.RESOURCES.pop(resource_id, None)

        req.status = '204 No Content'
        return b''
```

这已经提供了基本功能，允许我们从内存目录中添加、删除和列出资源。

1.  为了测试这一点，我们可以在后台线程中启动服务器，并使用我们之前的配方中的`http_request`函数：

```py
import threading
threading.Thread(target=app.serve, daemon=True).start()

from web_03 import http_request
```

1.  然后我们可以创建一个新的资源：

```py
>>> _, resp = http_request('http://localhost:8000/resources', method='POST', 
                           data=json.dumps({'name': 'Mario',
                                            'surname': 'Mario'}).encode('utf-8'))
>>> print('NEW RESOURCE: ', resp)
NEW RESOURCE:  b'{"surname": "Mario", "id": "1", "name": "Mario"}'
```

1.  这里我们列出它们全部：

```py
>>> _, resp = http_request('http://localhost:8000/resources')
>>> print('ALL RESOURCES: ', resp)
ALL RESOURCES:  b'{"1": {"surname": "Mario", "id": "1", "name": "Mario"}}'
```

1.  添加第二个：

```py
>>> http_request('http://localhost:8000/resources', method='POST', 
                 data=json.dumps({'name': 'Luigi',
                                  'surname': 'Mario'}).encode('utf-8'))
```

1.  接下来，我们看到现在列出了两个资源：

```py
>>> _, resp = http_request('http://localhost:8000/resources')
>>> print('ALL RESOURCES: ', resp)
ALL RESOURCES:  b'{"1": {"surname": "Mario", "id": "1", "name": "Mario"}, 
                   "2": {"surname": "Mario", "id": "2", "name": "Luigi"}}'
```

1.  然后我们可以从目录中请求特定的资源：

```py
>>> _, resp = http_request('http://localhost:8000/resources/1')
>>> print('RESOURCES #1: ', resp)
RESOURCES #1:  b'{"surname": "Mario", "id": "1", "name": "Mario"}'
```

1.  我们还可以删除特定的资源：

```py
>>> http_request('http://localhost:8000/resources/2', method='DELETE')
```

1.  然后查看它是否已被删除：

```py
>>> _, resp = http_request('http://localhost:8000/resources')
>>> print('ALL RESOURCES', resp)
ALL RESOURCES b'{"1": {"surname": "Mario", "id": "1", "name": "Mario"}}'
```

这应该允许我们为大多数简单情况提供 REST 接口，依赖于 Python 标准库中已经可用的内容。

# 工作原理...

大部分工作由`RestController.__call__`完成：

```py
class RestController:
    def __call__(self, req, resp):
        method = req.environ['REQUEST_METHOD']
        action = getattr(self, method, self._not_found)
        return action(req, resp)
```

每当调用`RestController`的子类时，它将查看 HTTP 请求方法，并查找一个命名类似于 HTTP 方法的实例方法。

如果有的话，将调用该方法，并返回方法本身提供的响应。如果没有，则调用`self._not_found`，它将只响应 404 错误。

这依赖于`WSGIApplication.__call__`对类而不是函数的支持。

当`WSGIApplication.__call__`通过`app.route`找到与路由关联的对象是一个类时，它将始终创建它的一个实例，然后它将调用该实例：

```py
if inspect.isclass(routed_action):
    routed_action = routed_action()
body = routed_action(request, resp)
```

如果`routed_action`是`RestController`的子类，那么将会发生的是`routed_action = routed_action()`将用其实例替换类，然后`routed_action(request, resp)`将调用`RestController.__call__`方法来实际处理请求。

然后，`RestController.__call__`方法可以根据 HTTP 方法将请求转发到正确的实例方法。

请注意，由于 REST 资源是通过在 URL 中提供资源标识符来识别的，因此分配给`RestController`的路由必须具有一个`id`参数和一个可选的`/`：

```py
@app.route('/resources/?(?P<id>\\w*)')
```

否则，您将无法区分对整个`GET`资源目录`/resources`的请求和对特定`GET`资源`/resources/3`的请求。

缺少`id`参数正是我们的`GET`方法决定何时返回整个目录的内容或不返回的方式：

```py
def GET(self, req, resp):
    resource_id = req.urlargs['id']
    if not resource_id:
        # Whole catalog requested
        return json.dumps(self.RESOURCES).encode('utf-8')
```

对于接收请求体中的数据的方法，例如`POST`，`PUT`和`PATCH`，您将不得不从`req.environ['wsgi.input']`读取请求体。

在这种情况下，重要的是提供要读取的字节数，因为连接可能永远不会关闭，否则读取可能会永远阻塞。

`Content-Length`头部可用于知道输入的长度：

```py
def POST(self, req, resp):
    content_length = int(req.environ['CONTENT_LENGTH'])
    data = req.environ['wsgi.input'].read(content_length).decode('utf-8')
```

# 处理 cookie

在 Web 应用程序中，cookie 经常用于在浏览器中存储数据。最常见的用例是用户识别。

我们将实现一个非常简单且不安全的基于 cookie 的身份识别系统，以展示如何使用它们。

# 如何做...

`http.cookies.SimpleCookie`类提供了解析和生成 cookie 所需的所有设施。

1.  我们可以依赖它来创建一个将设置 cookie 的 Web 应用程序端点：

```py
from web_06 import WSGIApplication

app = WSGIApplication()

import time
from http.cookies import SimpleCookie

@app.route('/identity')
def identity(req, resp):
    identity = int(time.time())

    cookie = SimpleCookie()
    cookie['identity'] = 'USER: {}'.format(identity)

    for set_cookie in cookie.values():
        resp.headers.add_header('Set-Cookie', set_cookie.OutputString())
    return b'Go back to <a href="/">index</a> to check your identity'
```

1.  我们可以使用它来创建一个解析 cookie 并告诉我们当前用户是谁的 cookie：

```py
@app.route('/')
def index(req, resp):
    if 'HTTP_COOKIE' in req.environ:
        cookies = SimpleCookie(req.environ['HTTP_COOKIE'])
        if 'identity' in cookies:
            return b'Welcome back, %b' % cookies['identity'].value.encode('utf-8')
    return b'Visit <a href="/identity">/identity</a> to get an identity'
```

1.  一旦启动应用程序，您可以将浏览器指向`http://localhost:8000`，然后您应该看到 Web 应用程序抱怨您缺少身份：

```py
app.serve()
```

点击建议的链接后，您应该得到一个，返回到索引页面，它应该通过 cookie 识别您。

# 它是如何工作的...

`SimpleCookie`类表示一个或多个值的 cookie。

每个值都可以像字典一样设置到 cookie 中：

```py
cookie = SimpleCookie()
cookie['identity'] = 'USER: {}'.format(identity)
```

如果 cookie`morsel`必须接受更多选项，那么可以使用字典语法进行设置：

```py
cookie['identity']['Path'] = '/'
```

每个 cookie 可以包含多个值，每个值都应该使用`Set-Cookie` HTTP 头进行设置。

迭代 cookie 将检索构成 cookie 的所有键/值对，然后在它们上调用`OutputString()`将返回编码为`Set-Cookie`头部所期望的 cookie 值，以及所有其他属性：

```py
for set_cookie in cookie.values():
    resp.headers.add_header('Set-Cookie', set_cookie.OutputString())
```

实际上，一旦设置了 cookie，调用`OutputString()`将会将您发送回浏览器的字符串：

```py
>>> cookie = SimpleCookie()
>>> cookie['somevalue'] = 42
>>> cookie['somevalue']['Path'] = '/'
>>> cookie['somevalue'].OutputString()
'somevalue=42; Path=/'
```

读取 cookie 与从`environ['HTTP_COOKIE']`值构建 cookie 一样简单，如果它可用的话：

```py
cookies = SimpleCookie(req.environ['HTTP_COOKIE'])
```

一旦 cookie 被解析，其中存储的值可以通过字典语法访问：

```py
cookies['identity']
```

# 还有更多...

在处理 cookie 时，您应该注意的一个特定条件是它们的生命周期。

Cookie 可以有一个`Expires`属性，它将说明它们应该在哪个日期死亡（浏览器将丢弃它们），实际上，这就是您删除 cookie 的方式。使用过去日期的`Expires`日期再次设置 cookie 将删除它。

但是 cookie 也可以有一个`Max-Age`属性，它规定它们应该保留多长时间，或者可以创建为会话 cookie，当浏览器窗口关闭时它们将消失。

因此，如果您遇到 cookie 随机消失或未正确加载回来的问题，请始终检查这些属性，因为 cookie 可能刚刚被浏览器删除。
