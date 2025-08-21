# 附录 F. 请求和响应对象

Django 使用请求和响应对象通过系统传递状态。

当请求页面时，Django 会创建一个包含有关请求的元数据的`HttpRequest`对象。然后 Django 加载适当的视图，将`HttpRequest`作为第一个参数传递给视图函数。每个视图负责返回一个`HttpResponse`对象。

本文档解释了`django.http`模块中定义的`HttpRequest`和`HttpResponse`对象的 API。

# HttpRequest 对象

## 属性

除非以下另有说明，否则应将所有属性视为只读。`session`是一个值得注意的例外。

**HttpRequest.scheme**

表示请求的方案（通常是`http`或`https`）的字符串。

**HttpRequest.body**

原始的 HTTP 请求正文作为字节字符串。这对于以不同于常规 HTML 表单的方式处理数据很有用：二进制图像，XML 有效负载等。要处理常规表单数据，请使用`HttpRequest.POST`。

您还可以使用类似文件的接口从 HttpRequest 中读取。请参阅`HttpRequest.read()`。

**HttpRequest.path**

表示所请求页面的完整路径的字符串，不包括域名。

示例：`/music/bands/the_beatles/`

**HttpRequest.path_info**

在某些 Web 服务器配置下，主机名后的 URL 部分被分成脚本前缀部分和路径信息部分。`path_info`属性始终包含路径信息部分的路径，无论使用的是哪个 Web 服务器。使用这个而不是`path`可以使您的代码更容易在测试和部署服务器之间移动。

例如，如果您的应用程序的`WSGIScriptAlias`设置为`/minfo`，那么`path`可能是`/minfo/music/bands/the_beatles/`，而`path_info`将是`/music/bands/the_beatles/`。

**HttpRequest.method**

表示请求中使用的 HTTP 方法的字符串。这是保证大写的。例如：

```py
if request.method == 'GET':
     do_something() elif request.method == 'POST':
     do_something_else() 

```

**HttpRequest.encoding**

表示用于解码表单提交数据的当前编码的字符串（或`None`，表示使用`DEFAULT_CHARSET`设置）。您可以写入此属性以更改访问表单数据时使用的编码。

任何后续的属性访问（例如从`GET`或`POST`读取）将使用新的`encoding`值。如果您知道表单数据不是使用`DEFAULT_CHARSET`编码的，则这很有用。

**HttpRequest.GET**

包含所有给定的 HTTP `GET`参数的类似字典的对象。请参阅下面的`QueryDict`文档。

**HttpRequest.POST**

包含所有给定的 HTTP `POST`参数的类似字典的对象，前提是请求包含表单数据。请参阅下面的`QueryDict`文档。

如果您需要访问请求中发布的原始或非表单数据，请通过`HttpRequest.body`属性访问。

可能会通过`POST`以空的`POST`字典形式进行请求-例如，通过`POST` HTTP 方法请求表单，但不包括表单数据。因此，您不应该使用`if request.POST`来检查是否使用了`POST`方法；而是使用`if request.method == 'POST'`（见上文）。

注意：`POST`不包括文件上传信息。请参阅`FILES`。

**HttpRequest.COOKIES**

包含所有 cookie 的标准 Python 字典。键和值都是字符串。

**HttpRequest.FILES**

包含所有上传文件的类似字典的对象。`FILES`中的每个键都是`<input type="file" name="" />`中的`name`。`FILES`中的每个值都是`UploadedFile`。

请注意，如果请求方法是`POST`，并且`<form>`发布到请求的`enctype="multipart/form-data"`，则`FILES`将只包含数据。否则，`FILES`将是一个空的类似字典的对象。

**HttpRequest.META**

包含所有可用的 HTTP 标头的标准 Python 字典。可用的标头取决于客户端和服务器，但这里有一些示例：

+   `CONTENT_LENGTH`：请求正文的长度（作为字符串）

+   `CONTENT_TYPE`: 请求主体的 MIME 类型

+   `HTTP_ACCEPT_ENCODING`: 响应的可接受编码

+   `HTTP_ACCEPT_LANGUAGE`: 响应的可接受语言

+   `HTTP_HOST`: 客户端发送的 HTTP Host 标头

+   `HTTP_REFERER`: 引用页面（如果有）

+   `HTTP_USER_AGENT`: 客户端的用户代理字符串

+   `QUERY_STRING`: 查询字符串，作为单个（未解析的）字符串

+   `REMOTE_ADDR`: 客户端的 IP 地址

+   `REMOTE_HOST`: 客户端的主机名

+   `REMOTE_USER`: 由 Web 服务器认证的用户（如果有）

+   `REQUEST_METHOD`: 诸如"`GET`"或"`POST`"的字符串

+   `SERVER_NAME`: 服务器的主机名

+   `SERVER_PORT`: 服务器的端口（作为字符串）

除了`CONTENT_LENGTH`和`CONTENT_TYPE`之外，请求中的任何 HTTP 标头都会通过将所有字符转换为大写字母，将任何连字符替换为下划线，并在名称前添加`HTTP_`前缀来转换为`META`键。因此，例如，名为`X-Bender`的标头将被映射到`META`键`HTTP_X_BENDER`。

**HttpRequest.user**

表示当前已登录用户的`AUTH_USER_MODEL`类型的对象。如果用户当前未登录，`user`将被设置为`django.contrib.auth.models.AnonymousUser`的实例。您可以使用`is_authenticated()`来区分它们，如下所示：

```py
if request.user.is_authenticated():
     # Do something for logged-in users. else:
     # Do something for anonymous users. 

```

只有在您的 Django 安装已激活`AuthenticationMiddleware`时，`user`才可用。

**HttpRequest.session**

一个可读写的类似字典的对象，表示当前会话。只有在您的 Django 安装已激活会话支持时才可用。

**HttpRequest.urlconf**

Django 本身未定义，但如果其他代码（例如自定义中间件类）设置了它，它将被读取。当存在时，它将被用作当前请求的根 URLconf，覆盖`ROOT_URLCONF`设置。

**HttpRequest.resolver_match**

表示已解析 URL 的`ResolverMatch`的实例。此属性仅在 URL 解析发生后设置，这意味着它在所有视图中都可用，但在执行 URL 解析之前执行的中间件方法中不可用（例如`process_request`，您可以改用`process_view`）。

## 方法

**HttpRequest.get_host()**

使用`HTTP_X_FORWARDED_HOST`（如果启用了`USE_X_FORWARDED_HOST`）和`HTTP_HOST`标头的信息返回请求的原始主机，按顺序。如果它们没有提供值，该方法将使用`SERVER_NAME`和`SERVER_PORT`的组合，详见 PEP 3333。

示例：`127.0.0.1:8000`

**注意**

当主机位于多个代理后面时，`get_host()`方法会失败。一个解决方案是使用中间件来重写代理标头，就像以下示例中的那样：

```py
class MultipleProxyMiddleware(object):
     FORWARDED_FOR_FIELDS = [
         'HTTP_X_FORWARDED_FOR',
         'HTTP_X_FORWARDED_HOST',
         'HTTP_X_FORWARDED_SERVER',
     ]
     def process_request(self, request):
         """
         Rewrites the proxy headers so that only the most
         recent proxy is used.
         """
         for field in self.FORWARDED_FOR_FIELDS:
             if field in request.META:
                 if ',' in request.META[field]:
                     parts = request.META[field].split(',')
                     request.META[field] = parts[-1].strip() 

```

此中间件应该放置在依赖于`get_host()`值的任何其他中间件之前，例如`CommonMiddleware`或`CsrfViewMiddleware`。

**HttpRequest.get_full_path()**

返回`path`，以及附加的查询字符串（如果适用）。

示例：`/music/bands/the_beatles/?print=true`

**HttpRequest.build_absolute_uri(location)**

返回`location`的绝对 URI 形式。如果未提供位置，则位置将设置为`request.get_full_path()`。

如果位置已经是绝对 URI，则不会被改变。否则，将使用此请求中可用的服务器变量构建绝对 URI。

示例：`http://example.com/music/bands/the_beatles/?print=true`

**HttpRequest.get_signed_cookie()**

返回已签名 cookie 的值，如果签名不再有效，则引发`django.core.signing.BadSignature`异常。如果提供`default`参数，异常将被抑制，而将返回该默认值。

可选的`salt`参数可用于提供额外的保护，防止对您的秘密密钥进行暴力攻击。如果提供了`salt`参数，将根据附加到 cookie 值的签名时间戳检查`max_age`参数，以确保 cookie 的年龄不超过`max_age`秒。

例如：

```py
>>> request.get_signed_cookie('name') 
'Tony' 
>>> request.get_signed_cookie('name', salt='name-salt') 
'Tony' # assuming cookie was set using the same salt 
>>> request.get_signed_cookie('non-existing-cookie') 
...
KeyError: 'non-existing-cookie' 
>>> request.get_signed_cookie('non-existing-cookie', False) 
False 
>>> request.get_signed_cookie('cookie-that-was-tampered-with') 
... 
BadSignature: ... 
>>> request.get_signed_cookie('name', max_age=60)
...
SignatureExpired: Signature age 1677.3839159 > 60 seconds 
>>> request.get_signed_cookie('name', False, max_age=60) 
False

```

**HttpRequest.is_secure()**

如果请求是安全的，则返回`True`；也就是说，如果是通过 HTTPS 发出的请求。

**HttpRequest.is_ajax()**

通过检查`HTTP_X_REQUESTED_WITH`标头中的字符串"`XMLHttpRequest`"，如果请求是通过`XMLHttpRequest`发出的，则返回`True`。大多数现代 JavaScript 库都会发送此标头。如果您自己编写`XMLHttpRequest`调用（在浏览器端），如果要使`is_ajax()`起作用，您将不得不手动设置此标头。

如果响应因是否通过 AJAX 请求而有所不同，并且您正在使用类似 Django 的`cache middleware`的某种缓存形式，则应该使用`vary_on_headers('HTTP_X_REQUESTED_WITH')`装饰视图，以便正确缓存响应。

**HttpRequest.read(size=None)**

**HttpRequest.readline()**

**HttpRequest.readlines()**

**HttpRequest.xreadlines()**

**HttpRequest.__iter__()**

实现从`HttpRequest`实例读取的类似文件的接口的方法。这使得可以以流式方式消耗传入的请求。一个常见的用例是使用迭代解析器处理大型 XML 有效负载，而不必在内存中构造整个 XML 树。

根据这个标准接口，可以直接将`HttpRequest`实例传递给诸如`ElementTree`之类的 XML 解析器：

```py
import xml.etree.ElementTree as ET 
for element in ET.iterparse(request):
     process(element) 

```

# QueryDict 对象

在`HttpRequest`对象中，`GET`和`POST`属性是`django.http.QueryDict`的实例，这是一个类似字典的类，专门用于处理同一键的多个值。这是必要的，因为一些 HTML 表单元素，特别是`<select multiple>`，会传递同一键的多个值。

在正常的请求/响应周期中，`request.POST`和`request.GET`中的`QueryDict`将是不可变的。要获得可变版本，您需要使用`.copy()`。

## 方法

`QueryDict`实现了所有标准字典方法，因为它是字典的子类，但有以下例外。

**QueryDict.__init__()**

基于`query_string`实例化一个`QueryDict`对象。

```py
>>> QueryDict('a=1&a=2&c=3') 
<QueryDict: {'a': ['1', '2'], 'c': ['3']}>

```

如果未传递`query_string`，则生成的`QueryDict`将为空（它将没有键或值）。

您遇到的大多数`QueryDict`，特别是`request.POST`和`request.GET`中的那些，将是不可变的。如果您自己实例化一个，可以通过在其`__init__()`中传递`mutable=True`来使其可变。

用于设置键和值的字符串将从`encoding`转换为 Unicode。如果未设置编码，则默认为`DEFAULT_CHARSET`。

**QueryDict.__getitem__(key)**

返回给定键的值。如果键有多个值，`__getitem__()`将返回最后一个值。如果键不存在，则引发`django.utils.datastructures.MultiValueDictKeyError`。

**QueryDict.__setitem__(key, value)**

将给定的键设置为`[value]`（一个 Python 列表，其单个元素为`value`）。请注意，像其他具有副作用的字典函数一样，只能在可变的`QueryDict`上调用（例如通过`copy()`创建的`QueryDict`）。

**QueryDict.__contains__(key)**

如果设置了给定的键，则返回`True`。这使您可以执行例如`if "foo" in request.GET`。

**QueryDict.get(key, default)**

使用与上面的`__getitem__()`相同的逻辑，具有返回默认值的钩子，如果键不存在。

**QueryDict.setdefault(key, default)**

与标准字典的`setdefault()`方法一样，只是它在内部使用`__setitem__()`。

**QueryDict.update(other_dict)**

接受`QueryDict`或标准字典。就像标准字典的`update()`方法一样，只是它将项目附加到当前字典项，而不是替换它们。例如：

```py
>>> q = QueryDict('a=1', mutable=True) 
>>> q.update({'a': '2'}) 
>>> q.getlist('a')
['1', '2'] 
>>> q['a'] # returns the last 
['2']

```

**QueryDict.items()**

与标准字典`items()`方法类似，只是这使用与`__getitem__()`相同的最后值逻辑。例如：

```py
>>> q = QueryDict('a=1&a=2&a=3') 
>>> q.items() 
[('a', '3')]

```

**QueryDict.iteritems()**

与标准字典`iteritems()`方法类似。与`QueryDict.items()`一样，这使用与`QueryDict.__getitem__()`相同的最后值逻辑。

**QueryDict.iterlists()**

与`QueryDict.iteritems()`类似，只是它包括字典的每个成员的所有值作为列表。

**QueryDict.values()**

与标准字典`values()`方法类似，只是这使用与`__getitem__()`相同的最后值逻辑。例如：

```py
>>> q = QueryDict('a=1&a=2&a=3') 
>>> q.values() 
['3']

```

**QueryDict.itervalues()**

与`QueryDict.values()`类似，只是一个迭代器。

此外，`QueryDict`有以下方法：

**QueryDict.copy()**

返回对象的副本，使用 Python 标准库中的`copy.deepcopy()`。即使原始对象不是可变的，此副本也将是可变的。

**QueryDict.getlist(key, default)**

返回请求的键的数据，作为 Python 列表。如果键不存在且未提供默认值，则返回空列表。除非默认值不是列表，否则保证返回某种列表。

**QueryDict.setlist(key, list)**

将给定的键设置为`list_`（与`__setitem__()`不同）。

**QueryDict.appendlist(key, item)**

将项目附加到与键关联的内部列表。

**QueryDict.setlistdefault(key, default_list)**

与`setdefault`类似，只是它接受值的列表而不是单个值。

**QueryDict.lists()**

与`items()`类似，只是它包括字典的每个成员的所有值作为列表。例如：

```py
>>> q = QueryDict('a=1&a=2&a=3') 
>>> q.lists() 
[('a', ['1', '2', '3'])]

```

**QueryDict.pop(key)**

返回给定键的值列表并从字典中删除它们。如果键不存在，则引发`KeyError`。例如：

```py
>>> q = QueryDict('a=1&a=2&a=3', mutable=True) 
>>> q.pop('a') 
['1', '2', '3']

```

**QueryDict.popitem()**

删除字典的任意成员（因为没有排序的概念），并返回一个包含键和键的所有值的列表的两个值元组。在空字典上调用时引发`KeyError`。例如：

```py
>>> q = QueryDict('a=1&a=2&a=3', mutable=True) 
>>> q.popitem() 
('a', ['1', '2', '3'])

```

**QueryDict.dict()**

返回`QueryDict`的`dict`表示。对于`QueryDict`中的每个（key，list）对，`dict`将有（key，item），其中 item 是列表的一个元素，使用与`QueryDict.__getitem__()`相同的逻辑：

```py
>>> q = QueryDict('a=1&a=3&a=5') 
>>> q.dict() 
{'a': '5'}

```

**QueryDict.urlencode([safe])**

返回查询字符串格式的数据字符串。例如：

```py
>>> q = QueryDict('a=2&b=3&b=5') 
>>> q.urlencode() 
'a=2&b=3&b=5'

```

可选地，urlencode 可以传递不需要编码的字符。例如：

```py
>>> q = QueryDict(mutable=True) 
>>> q['next'] = '/a&b/' 
>>> q.urlencode(safe='/') 
'next=/a%26b/'

```

# HttpResponse 对象

与 Django 自动创建的`HttpRequest`对象相反，`HttpResponse`对象是您的责任。您编写的每个视图都负责实例化，填充和返回`HttpResponse`。

`HttpResponse`类位于`django.http`模块中。

## 用法

**传递字符串**

典型用法是将页面内容作为字符串传递给`HttpResponse`构造函数：

```py
>>> from django.http import HttpResponse 
>>> response = HttpResponse("Here's the text of the Web page.") 
>>> response = HttpResponse("Text only, please.",
   content_type="text/plain")

```

但是，如果您想逐步添加内容，可以将`response`用作类似文件的对象：

```py
>>> response = HttpResponse() 
>>> response.write("<p>Here's the text of the Web page.</p>") 
>>> response.write("<p>Here's another paragraph.</p>")

```

**传递迭代器**

最后，您可以传递`HttpResponse`一个迭代器而不是字符串。`HttpResponse`将立即消耗迭代器，将其内容存储为字符串，并丢弃它。

如果您需要响应从迭代器流式传输到客户端，则必须使用`StreamingHttpResponse`类。

**设置头字段**

要设置或删除响应中的头字段，请将其视为字典：

```py
>>> response = HttpResponse() 
>>> response['Age'] = 120 
>>> del response['Age']

```

请注意，与字典不同，如果头字段不存在，`del`不会引发`KeyError`。

为了设置`Cache-Control`和`Vary`头字段，建议使用`django.utils.cache`中的`patch_cache_control()`和`patch_vary_headers()`方法，因为这些字段可以有多个逗号分隔的值。修补方法确保其他值，例如中间件添加的值，不会被移除。

HTTP 头字段不能包含换行符。尝试设置包含换行符（CR 或 LF）的头字段将引发`BadHeaderError`。

**告诉浏览器将响应视为文件附件**

要告诉浏览器将响应视为文件附件，请使用`content_type`参数并设置`Content-Disposition`标头。例如，这是如何返回 Microsoft Excel 电子表格的方式：

```py
>>> response = HttpResponse
  (my_data, content_type='application/vnd.ms-excel') 
>>> response['Content-Disposition'] = 'attachment; filename="foo.xls"'

```

`Content-Disposition`标头与 Django 无关，但很容易忘记语法，因此我们在这里包含了它。

## 属性

**HttpResponse.content**

表示内容的字节串，如果需要，从 Unicode 对象编码而来。

**HttpResponse.charset**

表示响应将被编码的字符集的字符串。如果在`HttpResponse`实例化时未给出，则将从`content_type`中提取，如果不成功，则将使用`DEFAULT_CHARSET`设置。

**HttpResponse.status_code**

响应的 HTTP 状态码。

**HttpResponse.reason_phrase**

响应的 HTTP 原因短语。

**HttpResponse.streaming**

这总是`False`。

此属性存在，以便中间件可以将流式响应与常规响应区分对待。

**HttpResponse.closed**

如果响应已关闭，则为`True`。

## 方法

**HttpResponse.__init__()**

```py
HttpResponse.__init__(content='', 
  content_type=None, status=200, reason=None, charset=None) 

```

使用给定的页面内容和内容类型实例化`HttpResponse`对象。`content`应该是迭代器或字符串。如果它是迭代器，它应该返回字符串，并且这些字符串将被连接在一起形成响应的内容。如果它不是迭代器或字符串，在访问时将被转换为字符串。有四个参数：

+   `content_type`是可选地由字符集编码完成的 MIME 类型，并用于填充 HTTP `Content-Type`标头。如果未指定，它将由`DEFAULT_CONTENT_TYPE`和`DEFAULT_CHARSET`设置形成，默认为：text/html; charset=utf-8。

+   `status`是响应的 HTTP 状态码。

+   `reason`是 HTTP 响应短语。如果未提供，将使用默认短语。

+   `charset`是响应将被编码的字符集。如果未给出，将从`content_type`中提取，如果不成功，则将使用`DEFAULT_CHARSET`设置。

**HttpResponse.__setitem__(header, value)**

将给定的标头名称设置为给定的值。`header`和`value`都应该是字符串。

**HttpResponse.__delitem__(header)**

删除具有给定名称的标头。如果标头不存在，则静默失败。不区分大小写。

**HttpResponse.__getitem__(header)**

返回给定标头名称的值。不区分大小写。

**HttpResponse.has_header(header)**

基于对具有给定名称的标头进行不区分大小写检查，返回`True`或`False`。

**HttpResponse.setdefault(header, value)**

除非已经设置了标头，否则设置标头。

**HttpResponse.set_cookie()**

```py
HttpResponse.set_cookie(key, value='', 
  max_age=None, expires=None, path='/', 
  domain=None, secure=None, httponly=False) 

```

设置 cookie。参数与 Python 标准库中的`Morsel` cookie 对象中的参数相同。

+   `max_age`应该是秒数，或者`None`（默认），如果 cookie 只应该持续客户端浏览器会话的时间。如果未指定`expires`，将进行计算。

+   `expires`应该是格式为`"Wdy, DD-Mon-YY HH:MM:SS GMT"`的字符串，或者是 UTC 中的`datetime.datetime`对象。如果`expires`是`datetime`对象，则将计算`max_age`。

+   如果要设置跨域 cookie，请使用`domain`。例如，`domain=".lawrence.com"`将设置一个可由 www.lawrence.com、blogs.lawrence.com 和 calendars.lawrence.com 读取的 cookie。否则，cookie 只能由设置它的域读取。

+   如果要防止客户端 JavaScript 访问 cookie，请使用`httponly=True`。

`HTTPOnly`是包含在 Set-Cookie HTTP 响应标头中的标志。它不是 RFC 2109 标准的一部分，并且并非所有浏览器都一致地遵守。但是，当它被遵守时，它可以是减轻客户端脚本访问受保护的 cookie 数据的有用方式。

**HttpResponse.set_signed_cookie()**

与`set_cookie()`类似，但在设置之前对 cookie 进行加密签名。与`HttpRequest.get_signed_cookie()`一起使用。您可以使用可选的`salt`参数来增加密钥强度，但您需要记住将其传递给相应的`HttpRequest.get_signed_cookie()`调用。

**HttpResponse.delete_cookie()**

删除具有给定键的 cookie。如果键不存在，则静默失败。

由于 cookie 的工作方式，`path`和`domain`应该与您在`set_cookie()`中使用的值相同-否则可能无法删除 cookie。

**HttpResponse.write(content)**

**HttpResponse.flush()**

**HttpResponse.tell()**

这些方法实现了与`HttpResponse`类似的文件接口。它们与相应的 Python 文件方法的工作方式相同。

**HttpResponse.getvalue()**

返回`HttpResponse.content`的值。此方法使`HttpResponse`实例成为类似流的对象。

**HttpResponse.writable()**

始终为`True`。此方法使`HttpResponse`实例成为类似流的对象。

**HttpResponse.writelines(lines)**

将一系列行写入响应。不会添加行分隔符。此方法使`HttpResponse`实例成为类似流的对象。

## HttpResponse 子类

Django 包括许多处理不同类型 HTTP 响应的`HttpResponse`子类。与`HttpResponse`一样，这些子类位于`django.http`中。

**HttpResponseRedirect**

构造函数的第一个参数是必需的-重定向的路径。这可以是一个完全合格的 URL（例如，[`www.yahoo.com/search/`](http://www.yahoo.com/search/)）或者没有域的绝对路径（例如，`/search/`）。查看`HttpResponse`以获取其他可选的构造函数参数。请注意，这会返回一个 HTTP 状态码 302。

**HttpResponsePermanentRedirect**

与`HttpResponseRedirect`类似，但返回永久重定向（HTTP 状态码 301）而不是找到重定向（状态码 302）。

**HttpResponseNotModified**

构造函数不接受任何参数，也不应向此响应添加任何内容。使用此方法指定自上次用户请求以来页面未被修改（状态码 304）。

**HttpResponseBadRequest**

行为与`HttpResponse`相同，但使用 400 状态码。

**HttpResponseNotFound**

行为与`HttpResponse`相同，但使用 404 状态码。

**HttpResponseForbidden**

行为与`HttpResponse`相同，但使用 403 状态码。

**HttpResponseNotAllowed**

与`HttpResponse`类似，但使用 405 状态码。构造函数的第一个参数是必需的：允许的方法列表（例如，`['GET', 'POST']`）。

**HttpResponseGone**

行为与`HttpResponse`相同，但使用 410 状态码。

**HttpResponseServerError**

行为与`HttpResponse`相同，但使用 500 状态码。

如果`HttpResponse`的自定义子类实现了`render`方法，Django 将把它视为模拟`SimpleTemplateResponse`，并且`render`方法本身必须返回一个有效的响应对象。

# JsonResponse 对象

```py
class JsonResponse(data, encoder=DjangoJSONEncoder, safe=True, **kwargs) 

```

帮助创建 JSON 编码响应的`HttpResponse`子类。它继承了大部分行为，但有一些不同之处：

+   其默认的`Content-Type`头设置为`application/json`。

+   第一个参数`data`应该是一个`dict`实例。如果将`safe`参数设置为`False`（见下文），则可以是任何可 JSON 序列化的对象。

+   `encoder`默认为`django.core.serializers.json.DjangoJSONEncoder`，将用于序列化数据。

`safe`布尔参数默认为`True`。如果设置为`False`，则可以传递任何对象进行序列化（否则只允许`dict`实例）。如果`safe`为`True`，并且将非`dict`对象作为第一个参数传递，将引发`TypeError`。

## 用法

典型的用法可能如下：

```py
>>> from django.http import JsonResponse >>> response = JsonResponse({'foo': 'bar'}) >>> response.content '{"foo": "bar"}'

```

**序列化非字典对象**

为了序列化除`dict`之外的对象，您必须将`safe`参数设置为`False`：

```py
response = JsonResponse([1, 2, 3], safe=False) 

```

如果不传递`safe=False`，将引发`TypeError`。

**更改默认的 JSON 编码器**

如果需要使用不同的 JSON 编码器类，可以将`encoder`参数传递给构造方法：

```py
response = JsonResponse(data, encoder=MyJSONEncoder) 

```

# StreamingHttpResponse 对象

`StreamingHttpResponse`类用于从 Django 向浏览器流式传输响应。如果生成响应需要太长时间或使用太多内存，你可能会想要这样做。例如，用于生成大型 CSV 文件非常有用。

## 性能考虑

Django 设计用于短暂的请求。流式响应将会绑定一个工作进程，直到响应完成。这可能导致性能不佳。

一般来说，你应该在请求-响应周期之外执行昂贵的任务，而不是使用流式响应。

`StreamingHttpResponse`不是`HttpResponse`的子类，因为它具有稍微不同的 API。但是，它几乎是相同的，具有以下显着的区别：

+   它应该给出一个产生字符串作为内容的迭代器。

+   除了迭代响应对象本身，你无法访问它的内容。这只能在响应返回给客户端时发生。

+   它没有`content`属性。相反，它有一个`streaming_content`属性。

+   你不能使用类似文件的对象的`tell()`或`write()`方法。这样做会引发异常。

`StreamingHttpResponse`应该只在绝对需要在将数据传输给客户端之前不迭代整个内容的情况下使用。因为无法访问内容，许多中间件无法正常工作。例如，对于流式响应，无法生成`ETag`和`Content-Length`标头。

## 属性

`StreamingHttpResponse`具有以下属性：

+   * `*.streaming_content.` 一个表示内容的字符串的迭代器。

+   * `*.status_code.` 响应的 HTTP 状态码。

+   * `*.reason_phrase.` 响应的 HTTP 原因短语。

+   * `*.streaming.` 这总是`True`。

# FileResponse 对象

`FileResponse`是针对二进制文件进行了优化的`StreamingHttpResponse`的子类。如果 wsgi 服务器提供了`wsgi.file_wrapper`，它将使用它，否则它会以小块流式传输文件。

`FileResponse`期望以二进制模式打开的文件，如下所示：

```py
>>> from django.http import FileResponse 
>>> response = FileResponse(open('myfile.png', 'rb'))

```

# 错误视图

Django 默认提供了一些视图来处理 HTTP 错误。要使用自定义视图覆盖这些视图，请参阅自定义错误视图。

## 404（页面未找到）视图

`defaults.page_not_found(request, template_name='404.html')`

当你在视图中引发`Http404`时，Django 会加载一个专门处理 404 错误的特殊视图。默认情况下，它是视图`django.views.defaults.page_not_found()`，它要么生成一个非常简单的未找到消息，要么加载和呈现模板`404.html`（如果你在根模板目录中创建了它）。

默认的 404 视图将向模板传递一个变量：`request_path`，这是导致错误的 URL。

关于 404 视图有三件事需要注意：

+   如果 Django 在检查 URLconf 中的每个正则表达式后找不到匹配项，也会调用 404 视图。

+   404 视图传递一个`RequestContext`，并且将可以访问由你的模板上下文处理器提供的变量（例如`MEDIA_URL`）。

+   如果`DEBUG`设置为`True`（在你的设置模块中），那么你的 404 视图将永远不会被使用，而且你的 URLconf 将被显示出来，带有一些调试信息。

## 500（服务器错误）视图

`defaults.server_error(request, template_name='500.html')`

同样地，Django 在视图代码运行时出现运行时错误的情况下执行特殊行为。如果视图导致异常，Django 将默认调用视图`django.views.defaults.server_error`，它要么生成一个非常简单的服务器错误消息，要么加载和呈现模板`500.html`（如果你在根模板目录中创建了它）。

默认的 500 视图不会向`500.html`模板传递任何变量，并且使用空的`Context`进行呈现，以减少额外错误的可能性。

如果`DEBUG`设置为`True`（在您的设置模块中），则永远不会使用您的 500 视图，而是显示回溯信息，附带一些调试信息。

## 403（HTTP Forbidden）视图

`defaults.permission_denied(request, template_name='403.html')`

与 404 和 500 视图一样，Django 还有一个视图来处理 403 Forbidden 错误。如果视图导致 403 异常，那么 Django 将默认调用视图`django.views.defaults.permission_denied`。

此视图加载并呈现根模板目录中的模板`403.html`，如果该文件不存在，则根据 RFC 2616（HTTP 1.1 规范）提供文本 403 Forbidden。

`django.views.defaults.permission_denied`由`PermissionDenied`异常触发。要在视图中拒绝访问，可以使用以下代码：

```py
from django.core.exceptions import PermissionDenied

def edit(request, pk):
     if not request.user.is_staff:
         raise PermissionDenied
     # ... 

```

## 400（错误请求）视图

`defaults.bad_request(request, template_name='400.html')`

当 Django 中引发`SuspiciousOperation`时，可能会由 Django 的某个组件处理（例如重置会话数据）。如果没有特别处理，Django 将认为当前请求是'bad request'而不是服务器错误。

`django.views.defaults.bad_request`，在其他方面与`server_error`视图非常相似，但返回状态码 400，表示错误条件是客户端操作的结果。

当`DEBUG`为`False`时，也只有`bad_request`视图才会被使用。

# 自定义错误视图

Django 中的默认错误视图应该适用于大多数 Web 应用程序，但如果需要任何自定义行为，可以轻松地覆盖它们。只需在 URLconf 中指定处理程序（在其他任何地方设置它们都不会起作用）。

`page_not_found()`视图被`handler404`覆盖：

```py
handler404 = 'mysite.views.my_custom_page_not_found_view' 

```

`server_error()`视图被`handler500`覆盖：

```py
handler500 = 'mysite.views.my_custom_error_view' 

```

`permission_denied()`视图被`handler403`覆盖：

```py
handler403 = 'mysite.views.my_custom_permission_denied_view' 

```

`bad_request()`视图被`handler400`覆盖：

```py
handler400 = 'mysite.views.my_custom_bad_request_view' 

```
