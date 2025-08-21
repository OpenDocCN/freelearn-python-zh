# 第十七章：Django 中间件

中间件是 Django 请求/响应处理的钩子框架。它是一个轻量级的、低级别的插件系统，用于全局修改 Django 的输入或输出。

每个中间件组件负责执行一些特定的功能。例如，Django 包括一个中间件组件`AuthenticationMiddleware`，它使用会话将用户与请求关联起来。

本文档解释了中间件的工作原理，如何激活中间件以及如何编写自己的中间件。Django 附带了一些内置的中间件，您可以直接使用。请参见本章后面的*可用中间件*。

# 激活中间件

要激活中间件组件，请将其添加到 Django 设置中的`MIDDLEWARE_CLASSES`列表中。

在`MIDDLEWARE_CLASSES`中，每个中间件组件都由一个字符串表示：中间件类名的完整 Python 路径。例如，这是由`django-admin startproject`创建的默认值：

```py
MIDDLEWARE_CLASSES = [ 
    'django.contrib.sessions.middleware.SessionMiddleware', 
    'django.middleware.common.CommonMiddleware', 
    'django.middleware.csrf.CsrfViewMiddleware', 
    'django.contrib.auth.middleware.AuthenticationMiddleware', 
    'django.contrib.messages.middleware.MessageMiddleware', 
    'django.middleware.clickjacking.XFrameOptionsMiddleware', 
] 

```

Django 安装不需要任何中间件-如果你愿意的话，`MIDDLEWARE_CLASSES`可以为空，但强烈建议至少使用`CommonMiddleware`。

`MIDDLEWARE_CLASSES`中的顺序很重要，因为一个中间件可能依赖于其他中间件。例如，`AuthenticationMiddleware`将认证用户存储在会话中；因此，它必须在`SessionMiddleware`之后运行。有关 Django 中间件类的常见提示的*中间件排序*，请参见本章后面。

# 钩子和应用顺序

在请求阶段，在调用视图之前，Django 按照在`MIDDLEWARE_CLASSES`中定义的顺序应用中间件，从上到下。有两个钩子可用：

+   `process_request()`

+   `process_view()`

在响应阶段，在调用视图之后，中间件按照从下到上的顺序应用。有三个钩子可用：

+   `process_exception()`

+   `process_template_response()`

+   `process_response()`

如果您愿意，您也可以将其视为洋葱：每个中间件类都是包装视图的一层。

下面描述了每个钩子的行为。

# 编写自己的中间件

编写自己的中间件很容易。每个中间件组件都是一个单独的 Python 类，定义了以下一个或多个方法：

## process_request

方法：`process_request(request)`

+   `request`是一个`HttpRequest`对象。

+   `process_request()`在 Django 决定执行哪个视图之前，对每个请求都会调用。

它应该返回`None`或者一个`HttpResponse`对象。如果返回`None`，Django 将继续处理此请求，执行任何其他`process_request()`中间件，然后执行`process_view()`中间件，最后执行适当的视图。

如果返回一个`HttpResponse`对象，Django 将不再调用任何其他请求、视图或异常中间件，或者适当的视图；它将对该`HttpResponse`应用响应中间件，并返回结果。

## process_view

方法：`process_view(request, view_func, view_args, view_kwargs)`

+   `request`是一个`HttpRequest`对象。

+   `view_func`是 Django 即将使用的 Python 函数。（它是实际的函数对象，而不是函数名作为字符串。）

+   `view_args`是将传递给视图的位置参数列表。

+   `view_kwargs`是将传递给视图的关键字参数字典。

+   `view_args`和`view_kwargs`都不包括第一个视图参数（`request`）。

`process_view()`在 Django 调用视图之前调用。它应该返回`None`或者一个`HttpResponse`对象。如果返回`None`，Django 将继续处理此请求，执行任何其他`process_view()`中间件，然后执行适当的视图。

如果返回一个`HttpResponse`对象，Django 将不再调用任何其他视图或异常中间件，或者适当的视图；它将对该`HttpResponse`应用响应中间件，并返回结果。

### 注意

在 `process_request` 或 `process_view` 中从中间件访问 `request.POST` 将阻止任何在中间件之后运行的视图能够修改请求的上传处理程序，并且通常应该避免这样做。

`CsrfViewMiddleware` 类可以被视为一个例外，因为它提供了 `csrf_exempt()` 和 `csrf_protect()` 装饰器，允许视图明确控制 CSRF 验证应该在何时发生。

## process_template_response

方法：`process_template_response(request, response)`

+   `request` 是一个 `HttpRequest` 对象。

+   `response` 是由 Django 视图或中间件返回的 `TemplateResponse` 对象（或等效对象）。

如果响应实例具有 `render()` 方法，表示它是 `TemplateResponse` 或等效对象，则会在视图执行完成后立即调用 `process_template_response()`。

它必须返回一个实现 `render` 方法的响应对象。它可以通过更改 `response.template_name` 和 `response.context_data` 来修改给定的 `response`，也可以创建并返回全新的 `TemplateResponse` 或等效对象。

您不需要显式渲染响应-一旦调用了所有模板响应中间件，响应将自动渲染。

在响应阶段中，中间件按照相反的顺序运行，其中包括 `process_template_response()`。

## process_response

方法：`process_response(request, response)`

+   `request` 是一个 `HttpRequest` 对象。

+   `response` 是由 Django 视图或中间件返回的 `HttpResponse` 或 `StreamingHttpResponse` 对象。

在将响应返回给浏览器之前，将调用 `process_response()`。它必须返回一个 `HttpResponse` 或 `StreamingHttpResponse` 对象。它可以修改给定的 `response`，也可以创建并返回全新的 `HttpResponse` 或 `StreamingHttpResponse`。

与 `process_request()` 和 `process_view()` 方法不同，`process_response()` 方法始终会被调用，即使同一中间件类的 `process_request()` 和 `process_view()` 方法被跳过（因为之前的中间件方法返回了一个 `HttpResponse`）。特别是，这意味着您的 `process_response()` 方法不能依赖于在 `process_request()` 中进行的设置。

最后，在响应阶段，中间件按照从下到上的顺序应用。这意味着在 `MIDDLEWARE_CLASSES` 的末尾定义的类将首先运行。

### 处理流式响应

与 `HttpResponse` 不同，`StreamingHttpResponse` 没有 `content` 属性。因此，中间件不能再假定所有响应都有 `content` 属性。如果它们需要访问内容，它们必须测试流式响应并相应地调整其行为：

```py
if response.streaming: 
    response.streaming_content =  wrap_streaming_content(response.streaming_content) 
else: 
    response.content = alter_content(response.content) 

```

`streaming_content` 应被假定为太大而无法在内存中保存。响应中间件可以将其包装在一个新的生成器中，但不得消耗它。包装通常实现如下：

```py
def wrap_streaming_content(content): 
    for chunk in content: 
        yield alter_content(chunk) 

```

## process_exception

方法：`process_exception(request, exception)`

+   `request` 是一个 `HttpRequest` 对象。

+   `exception` 是由视图函数引发的 `Exception` 对象。

当视图引发异常时，Django 调用 `process_exception()`。`process_exception()` 应该返回 `None` 或一个 `HttpResponse` 对象。如果它返回一个 `HttpResponse` 对象，模板响应和响应中间件将被应用，并将生成的响应返回给浏览器。否则，将启用默认的异常处理。

同样，在响应阶段中，中间件按照相反的顺序运行，其中包括 `process_exception`。如果异常中间件返回一个响应，那么该中间件上面的中间件类将根本不会被调用。

## __init__

大多数中间件类不需要初始化器，因为中间件类本质上是 `process_*` 方法的占位符。如果您需要一些全局状态，可以使用 `__init__` 进行设置。但是，请记住一些注意事项：

1.  Django 在不带任何参数的情况下初始化您的中间件，因此您不能将 `__init__` 定义为需要任何参数。

1.  与每个请求调用一次的 `process_*` 方法不同，`__init__` 仅在 Web 服务器响应第一个请求时调用一次。

### 将中间件标记为未使用

有时在运行时确定是否应使用某个中间件是有用的。在这些情况下，您的中间件的 `__init__` 方法可能会引发 `django.core.exceptions.MiddlewareNotUsed`。Django 将从中间件流程中删除该中间件，并在 `DEBUG` 设置为 `True` 时，将在 `django.request` 记录器中记录调试消息。

## 其他指南

+   中间件类不必是任何东西的子类。

+   中间件类可以存在于 Python 路径的任何位置。Django 关心的是 `MIDDLEWARE_CLASSES` 设置包含其路径。

+   随时查看 Django 提供的中间件示例。

+   如果您编写了一个您认为对其他人有用的中间件组件，请为社区做出贡献！让我们知道，我们将考虑将其添加到 Django 中。

# 可用的中间件

## 缓存中间件

`django.middleware.cache.UpdateCacheMiddleware`; 和 `django.middleware.cache.FetchFromCacheMiddleware`

启用站点范围的缓存。如果启用了这些选项，则每个由 Django 提供动力的页面将根据 `CACHE_MIDDLEWARE_SECONDS` 设置的定义缓存。请参阅缓存文档。

## 常见中间件

`django.middleware.common.CommonMiddleware`

为完美主义者添加了一些便利：

+   禁止访问 `DISALLOWED_USER_AGENTS` 设置中的用户代理，该设置应该是编译的正则表达式对象的列表。

+   基于 `APPEND_SLASH` 和 `PREPEND_WWW` 设置执行 URL 重写。

+   如果 `APPEND_SLASH` 为 `True`，并且初始 URL 不以斜杠结尾，并且在 URLconf 中找不到，则将通过在末尾添加斜杠来形成新的 URL。如果在 URLconf 中找到此新 URL，则 Django 将重定向请求到此新 URL。否则，将像往常一样处理初始 URL。

+   例如，如果您没有 `foo.com/bar` 的有效 URL 模式，但是有 `foo.com/bar/` 的有效模式，则将重定向到 `foo.com/bar/`。

+   如果 `PREPEND_WWW` 为 `True`，则缺少前导 `www.` 的 URL 将重定向到具有前导 `www.` 的相同 URL。

+   这两个选项都旨在规范化 URL。哲学是每个 URL 应该存在于一个且仅一个位置。从技术上讲，URL `foo.com/bar` 与 `foo.com/bar/` 是不同的-搜索引擎索引器将其视为单独的 URL-因此最佳做法是规范化 URL。

+   根据 `USE_ETAGS` 设置处理 ETags。如果 `USE_ETAGS` 设置为 `True`，Django 将通过对页面内容进行 MD5 哈希来计算每个请求的 ETag，并在适当时负责发送 `Not Modified` 响应。

+   `CommonMiddleware.response_redirect_class.` 默认为 `HttpResponsePermanentRedirect`。子类 `CommonMiddleware` 并覆盖属性以自定义中间件发出的重定向。

+   `django.middleware.common.BrokenLinkEmailsMiddleware.` 将损坏的链接通知邮件发送给 `MANAGERS.`

## GZip 中间件

`django.middleware.gzip.GZipMiddleware`

### 注意

安全研究人员最近披露，当网站使用压缩技术（包括 `GZipMiddleware`）时，该网站会暴露于许多可能的攻击。这些方法可以用来破坏 Django 的 CSRF 保护，等等。在您的网站上使用 `GZipMiddleware` 之前，您应该非常仔细地考虑您是否受到这些攻击的影响。如果您对自己是否受影响有任何疑问，您应该避免使用 `GZipMiddleware`。有关更多详细信息，请参阅 `breachattack.com`。

为了理解 GZip 压缩的浏览器压缩内容（所有现代浏览器）。

此中间件应放置在需要读取或写入响应正文的任何其他中间件之前，以便在之后进行压缩。

如果以下任何条件为真，则不会压缩内容：

+   内容主体长度小于 200 字节。

+   响应已设置了`Content-Encoding`头。

+   请求（浏览器）未发送包含`gzip`的`Accept-Encoding`头。

您可以使用`gzip_page()`装饰器将 GZip 压缩应用于单个视图。

## 有条件的 GET 中间件

`django.middleware.http.ConditionalGetMiddleware`

处理有条件的 GET 操作。如果响应具有`ETag`或`Last-Modified`头，并且请求具有`If-None-Match`或`If-Modified-Since`，则响应将被`HttpResponseNotModified`替换。

还设置了`Date`和`Content-Length`响应头。

## 区域中间件

`django.middleware.locale.LocaleMiddleware`

基于请求数据启用语言选择。它为每个用户定制内容。请参阅国际化文档。

`LocaleMiddleware.response_redirect_class`

默认为`HttpResponseRedirect`。子类化`LocaleMiddleware`并覆盖属性以自定义中间件发出的重定向。

## 消息中间件

`django.contrib.messages.middleware.MessageMiddleware`

启用基于 cookie 和会话的消息支持。请参阅消息文档。

## 安全中间件

### 注意

如果您的部署情况允许，通常最好让您的前端 Web 服务器执行`SecurityMiddleware`提供的功能。这样，如果有一些不是由 Django 提供服务的请求（如静态媒体或用户上传的文件），它们将具有与请求到您的 Django 应用程序相同的保护。

`django.middleware.security.SecurityMiddleware`为请求/响应周期提供了几个安全增强功能。`SecurityMiddleware`通过向浏览器传递特殊头来实现这一点。每个头都可以通过设置独立启用或禁用。

### HTTP 严格传输安全

设置：

+   `SECURE_HSTS_INCLUDE_SUBDOMAINS`

+   `SECURE_HSTS_SECONDS`

对于应该只能通过 HTTPS 访问的网站，您可以通过设置`Strict-Transport-Security`头，指示现代浏览器拒绝通过不安全的连接连接到您的域名（在一定时间内）。这减少了您对一些 SSL 剥离中间人（MITM）攻击的风险。

如果将`SECURE_HSTS_SECONDS`设置为非零整数值，则`SecurityMiddleware`将在所有 HTTPS 响应上为您设置此头。

启用 HSTS 时，最好首先使用一个小值进行测试，例如，`SECURE_HSTS_SECONDS = 3600`表示一小时。每次 Web 浏览器从您的站点看到 HSTS 头时，它将拒绝在给定时间内与您的域进行非安全（使用 HTTP）通信。

一旦确认您的站点上的所有资产都安全提供服务（即，HSTS 没有破坏任何内容），最好增加此值，以便偶尔访问者受到保护（31536000 秒，即 1 年，是常见的）。

此外，如果将`SECURE_HSTS_INCLUDE_SUBDOMAINS`设置为`True`，`SecurityMiddleware`将在`Strict-Transport-Security`头中添加`includeSubDomains`标记。这是建议的（假设所有子域都仅使用 HTTPS 提供服务），否则您的站点可能仍然会通过不安全的连接对子域进行攻击。

### 注意

HSTS 策略适用于整个域，而不仅仅是您设置头的响应的 URL。因此，只有在整个域通过 HTTPS 提供服务时才应该使用它。

正确尊重 HSTS 头的浏览器将拒绝允许用户绕过警告并连接到具有过期、自签名或其他无效 SSL 证书的站点。如果使用 HSTS，请确保您的证书状况良好并保持良好！

### X-content-type-options: nosniff

设置：

+   `SECURE_CONTENT_TYPE_NOSNIFF`

一些浏览器会尝试猜测它们获取的资产的内容类型，覆盖`Content-Type`头。虽然这可以帮助显示配置不正确的服务器的站点，但也可能带来安全风险。

如果您的网站提供用户上传的文件，恶意用户可能会上传一个特制的文件，当您期望它是无害的时，浏览器会将其解释为 HTML 或 Javascript。

为了防止浏览器猜测内容类型并强制它始终使用`Content-Type`头中提供的类型，您可以传递`X-Content-Type-Options: nosniff`头。如果`SECURE_CONTENT_TYPE_NOSNIFF`设置为`True`，`SecurityMiddleware`将对所有响应执行此操作。

请注意，在大多数部署情况下，Django 不涉及提供用户上传的文件，这个设置对您没有帮助。例如，如果您的`MEDIA_URL`是由您的前端 Web 服务器（nginx，Apache 等）直接提供的，那么您需要在那里设置这个头部。

另一方面，如果您正在使用 Django 执行诸如要求授权才能下载文件之类的操作，并且无法使用您的 Web 服务器设置头部，那么这个设置将很有用。

### X-XSS 保护

设置：

+   `SECURE_BROWSER_XSS_FILTER`

一些浏览器有能力阻止看起来像 XSS 攻击的内容。它们通过查找页面的 GET 或 POST 参数中的 Javascript 内容来工作。如果服务器的响应中重放了 Javascript，则页面将被阻止渲染，并显示错误页面。

`X-XSS-Protection header`用于控制 XSS 过滤器的操作。

为了在浏览器中启用 XSS 过滤器，并强制它始终阻止疑似的 XSS 攻击，您可以传递`X-XSS-Protection: 1; mode=block`头。如果`SECURE_BROWSER_XSS_FILTER`设置为`True`，`SecurityMiddleware`将对所有响应执行此操作。

### 注意

浏览器 XSS 过滤器是一种有用的防御措施，但不能完全依赖它。它无法检测所有的 XSS 攻击，也不是所有的浏览器都支持该头部。确保您仍在验证和所有输入，以防止 XSS 攻击。

### SSL 重定向

设置：

+   `SECURE_REDIRECT_EXEMPT`

+   `SECURE_SSL_HOST`

+   `SECURE_SSL_REDIRECT`

如果您的网站同时提供 HTTP 和 HTTPS 连接，大多数用户最终将默认使用不安全的连接。为了最佳安全性，您应该将所有 HTTP 连接重定向到 HTTPS。

如果将`SECURE_SSL_REDIRECT`设置为 True，`SecurityMiddleware`将永久（HTTP 301）将所有 HTTP 连接重定向到 HTTPS。

出于性能原因，最好在 Django 之外进行这些重定向，在前端负载均衡器或反向代理服务器（如 nginx）中。`SECURE_SSL_REDIRECT`适用于这种情况下无法选择的部署情况。

如果`SECURE_SSL_HOST`设置有值，所有重定向将发送到该主机，而不是最初请求的主机。

如果您的网站上有一些页面应该通过 HTTP 可用，并且不重定向到 HTTPS，您可以在`SECURE_REDIRECT_EXEMPT`设置中列出正则表达式来匹配这些 URL。

如果您部署在负载均衡器或反向代理服务器后，并且 Django 似乎无法确定请求实际上已经安全，您可能需要设置`SECURE_PROXY_SSL_HEADER`设置。

## 会话中间件

`django.contrib.sessions.middleware.SessionMiddleware`

启用会话支持。有关更多信息，请参见第十五章，“Django 会话”。

## 站点中间件

`django.contrib.sites.middleware.CurrentSiteMiddleware`

为每个传入的`HttpRequest`对象添加代表当前站点的`site`属性。有关更多信息，请参见站点文档（[`docs.djangoproject.com/en/1.8/ref/contrib/sites/`](https://docs.djangoproject.com/en/1.8/ref/contrib/sites/)）。

## 身份验证中间件

`django.contrib.auth.middleware`提供了三个用于身份验证的中间件：

+   `*.AuthenticationMiddleware.` 向每个传入的`HttpRequest`对象添加代表当前登录用户的`user`属性。

+   `*.RemoteUserMiddleware.` 用于利用 Web 服务器提供的身份验证。

+   `*.SessionAuthenticationMiddleware.` 允许在用户密码更改时使用户会话失效。此中间件必须出现在`MIDDLEWARE_CLASSES`中`*.AuthenticationMiddleware`之后。

有关 Django 中用户身份验证的更多信息，请参见第十一章，“Django 中的用户身份验证”。

## CSRF 保护中间件

`django.middleware.csrf.CsrfViewMiddleware`

通过向 POST 表单添加隐藏的表单字段并检查请求的正确值来防止跨站点请求伪造（CSRF）。有关 CSRF 保护的更多信息，请参见第十九章，“Django 中的安全性”。

## X-Frame-options 中间件

`django.middleware.clickjacking.XFrameOptionsMiddleware`

通过 X-Frame-Options 标头进行简单的点击劫持保护。

# 中间件排序

*表 17.1*提供了有关各种 Django 中间件类的排序的一些提示：

| **类** | **注释** |
| --- | --- |
| UpdateCacheMiddleware | 在修改`Vary`标头的中间件之前（`SessionMiddleware`，`GZipMiddleware`，`LocaleMiddleware`）。 |
| GZipMiddleware | 在可能更改或使用响应正文的任何中间件之前。在`UpdateCacheMiddleware`之后：修改`Vary`标头。 |
| ConditionalGetMiddleware | 在`CommonMiddleware`之前：当`USE_ETAGS`=`True`时使用其`Etag`标头。 |
| SessionMiddleware | 在`UpdateCacheMiddleware`之后：修改`Vary`标头。 |
| LocaleMiddleware | 在顶部之一，之后是`SessionMiddleware`（使用会话数据）和`CacheMiddleware`（修改`Vary`标头）。 |
| CommonMiddleware | 在可能更改响应的任何中间件之前（它计算`ETags`）。在`GZipMiddleware`之后，因此它不会在经过 gzip 处理的内容上计算`ETag`标头。靠近顶部：当`APPEND_SLASH`或`PREPEND_WWW`设置为`True`时进行重定向。 |
| CsrfViewMiddleware | 在假定已处理 CSRF 攻击的任何视图中间件之前。 |
| AuthenticationMiddleware | 在`SessionMiddleware`之后：使用会话存储。 |
| MessageMiddleware | 在`SessionMiddleware`之后：可以使用基于会话的存储。 |
| FetchFromCacheMiddleware | 在修改`Vary`标头的任何中间件之后：该标头用于选择缓存哈希键的值。 |
| FlatpageFallbackMiddleware | 应该靠近底部，因为它是一种最后一招的中间件。 |
| RedirectFallbackMiddleware | 应该靠近底部，因为它是一种最后一招的中间件。 |

表 17.1：中间件类的排序

# 接下来是什么？

在下一章中，我们将研究 Django 中的国际化。
