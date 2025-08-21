# 第十五章：Django 会话

想象一下，如果您每次导航到另一个页面都必须重新登录到网站，或者您最喜欢的网站忘记了所有的设置，您每次访问时都必须重新输入？

现代网站如果没有一种方式来记住您是谁以及您在网站上的先前活动，就无法提供我们习惯的可用性和便利性。HTTP 是*无状态*的设计-在一次请求和下一次请求之间没有持久性，服务器无法判断连续的请求是否来自同一个人。

这种状态的缺乏是通过*会话*来管理的，这是您的浏览器和 Web 服务器之间的一种半永久的双向通信。当您访问现代网站时，在大多数情况下，Web 服务器将使用*匿名会话*来跟踪与您的访问相关的数据。会话被称为匿名，因为 Web 服务器只能记录您的操作，而不能记录您是谁。

我们都经历过这种情况，当我们在以后返回到电子商务网站时，发现我们放在购物车中的物品仍然在那里，尽管没有提供任何个人信息。会话通常使用经常受到诟病但很少被理解的*cookie*来持久化。与所有其他 Web 框架一样，Django 也使用 cookie，但以更聪明和安全的方式，您将看到。

Django 完全支持匿名会话。会话框架允许您在每个站点访问者的基础上存储和检索任意数据。它在服务器端存储数据并抽象了发送和接收 cookie。Cookie 包含会话 ID-而不是数据本身（除非您使用基于 cookie 的后端）；这是一种比其他框架更安全的实现 cookie 的方式。

# 启用会话

会话是通过中间件实现的。要启用会话功能，请编辑`MIDDLEWARE_CLASSES`设置，并确保其中包含`'django.contrib.sessions.middleware.SessionMiddleware'`。由`django-admin startproject`创建的默认`settings.py`已激活`SessionMiddleware`。

如果您不想使用会话，您也可以从`MIDDLEWARE_CLASSES`中删除`SessionMiddleware`行，并从`INSTALLED_APPS`中删除`'django.contrib.sessions'`。这将节省一点开销。

# 配置会话引擎

默认情况下，Django 将会话存储在数据库中（使用模型`django.contrib.sessions.models.Session`）。虽然这很方便，但在某些设置中，将会话数据存储在其他地方可能更快，因此可以配置 Django 将会话数据存储在文件系统或缓存中。

## 使用基于数据库的会话

如果您想使用基于数据库的会话，您需要将`'django.contrib.sessions'`添加到您的`INSTALLED_APPS`设置中。一旦配置了安装，运行`manage.py migrate`来安装存储会话数据的单个数据库表。

## 使用缓存会话

为了获得更好的性能，您可能希望使用基于缓存的会话后端。要使用 Django 的缓存系统存储会话数据，您首先需要确保已配置了缓存；有关详细信息，请参阅缓存文档。

### 注意

只有在使用 Memcached 缓存后端时，才应该使用基于缓存的会话。本地内存缓存后端不会保留数据足够长时间，因此直接使用文件或数据库会话而不是通过文件或数据库缓存后端发送所有内容将更快。此外，本地内存缓存后端不是多进程安全的，因此在生产环境中可能不是一个好选择。

如果在`CACHES`中定义了多个缓存，Django 将使用默认缓存。要使用另一个缓存，将`SESSION_CACHE_ALIAS`设置为该缓存的名称。配置好缓存后，您有两种选择来存储缓存中的数据：

+   将`SESSION_ENGINE`设置为`"django.contrib.sessions.backends.cache"`以使用简单的缓存会话存储。会话数据将直接存储在缓存中。但是，会话数据可能不是持久的：如果缓存填满或缓存服务器重新启动，缓存数据可能会被驱逐。

+   对于持久的缓存数据，将`SESSION_ENGINE`设置为`"django.contrib.sessions.backends.cached_db"`。这使用了一个写入缓存-每次写入缓存时也会写入数据库。会话读取仅在数据不在缓存中时才使用数据库。

这两种会话存储都非常快，但简单缓存更快，因为它忽略了持久性。在大多数情况下，`cached_db`后端将足够快，但如果您需要最后一点性能，并且愿意让会话数据不时被清除，那么`cache`后端适合您。如果您使用`cached_db`会话后端，还需要遵循使用基于数据库的会话的配置说明。

## 使用基于文件的会话

要使用基于文件的会话，请将`SESSION_ENGINE`设置为`"django.contrib.sessions.backends.file"`。您可能还想设置`SESSION_FILE_PATH`设置（默认为`tempfile.gettempdir()`的输出，很可能是`/tmp`）以控制 Django 存储会话文件的位置。请确保您的 Web 服务器有权限读取和写入此位置。

## 使用基于 cookie 的会话

要使用基于 cookie 的会话，请将`SESSION_ENGINE`设置为`"django.contrib.sessions.backends.signed_cookies"`。会话数据将使用 Django 的加密签名工具和`SECRET_KEY`设置进行存储。

建议将`SESSION_COOKIE_HTTPONLY`设置为`True`，以防止 JavaScript 访问存储的数据。

### 注意

**如果`SECRET_KEY`不保密，并且您使用`PickleSerializer`，这可能导致任意远程代码执行。**

拥有`SECRET_KEY`的攻击者不仅可以生成被您的站点信任的伪造会话数据，还可以远程执行任意代码，因为数据使用 pickle 进行序列化。如果您使用基于 cookie 的会话，请特别注意始终保持您的秘钥完全保密，以防止任何可能远程访问的系统。

### 注意

**会话数据已签名但未加密**

在使用 cookie 后端时，会话数据可以被客户端读取。使用 MAC（消息认证码）来保护数据免受客户端的更改，因此当被篡改时会使会话数据无效。如果存储 cookie 的客户端（例如，您的用户浏览器）无法存储所有会话 cookie 并丢弃数据，也会发生相同的无效。即使 Django 压缩了数据，仍然完全有可能超过每个 cookie 的常见限制 4096 字节。

### 注意

**没有新鲜度保证**

还要注意，虽然 MAC 可以保证数据的真实性（即它是由您的站点生成的，而不是其他人），以及数据的完整性（即它是否完整且正确），但它无法保证新鲜度，也就是说，您被发送回客户端的是您最后发送的内容。这意味着对于某些会话数据的使用，cookie 后端可能会使您容易受到重放攻击。与其他会话后端不同，其他会话后端会在用户注销时保留每个会话的服务器端记录并使其无效，而基于 cookie 的会话在用户注销时不会被无效。因此，如果攻击者窃取了用户的 cookie，他们可以使用该 cookie 以该用户的身份登录，即使用户已注销。只有当 cookie 的年龄大于您的`SESSION_COOKIE_AGE`时，才会检测到 cookie 已过期。

最后，假设上述警告没有阻止您使用基于 cookie 的会话：cookie 的大小也会影响站点的速度。

# 在视图中使用会话

当激活`SessionMiddleware`时，每个`HttpRequest`对象-任何 Django 视图函数的第一个参数-都将有一个`session`属性，这是一个类似字典的对象。您可以在视图的任何时候读取它并写入`request.session`。您可以多次编辑它。

所有会话对象都继承自基类`backends.base.SessionBase`。它具有以下标准字典方法：

+   `__getitem__(key)`

+   `__setitem__(key, value)`

+   `__delitem__(key)`

+   `__contains__(key)`

+   `get(key, default=None)`

+   `pop(key)`

+   `keys()`

+   `items()`

+   `setdefault()`

+   `clear()`

它还具有这些方法：

## flush()

从会话中删除当前会话数据并删除会话 cookie。如果您希望确保无法再次从用户的浏览器访问以前的会话数据（例如，`django.contrib.auth.logout()`函数调用它）。

## set_test_cookie()

设置一个测试 cookie 以确定用户的浏览器是否支持 cookie。由于 cookie 的工作方式，您将无法在用户的下一个页面请求之前测试这一点。有关更多信息，请参见下面的*设置测试 cookie*。

## test_cookie_worked()

返回`True`或`False`，取决于用户的浏览器是否接受了测试 cookie。由于 cookie 的工作方式，您将不得不在先前的单独页面请求上调用`set_test_cookie()`。有关更多信息，请参见下面的*设置测试 cookie*。

## delete_test_cookie()

删除测试 cookie。使用此方法进行清理。

## set_expiry(value)

设置会话的过期时间。您可以传递许多不同的值：

+   如果`value`是一个整数，会话将在多少秒的不活动后过期。例如，调用`request.session.set_expiry(300)`会使会话在 5 分钟后过期。

+   如果`value`是`datetime`或`timedelta`对象，则会话将在特定日期/时间过期。请注意，只有在使用`PickleSerializer`时，`datetime`和`timedelta`值才能被序列化。

+   如果`value`是`0`，用户的会话 cookie 将在用户的 Web 浏览器关闭时过期。

+   如果`value`是`None`，会话将恢复使用全局会话过期策略。

阅读会话不被视为过期目的的活动。会话的过期是根据会话上次修改的时间计算的。

## get_expiry_age()

返回直到此会话过期的秒数。对于没有自定义过期时间（或者设置为在浏览器关闭时过期）的会话，这将等于`SESSION_COOKIE_AGE`。此函数接受两个可选的关键字参数：

+   `modification`：会话的最后修改，作为`datetime`对象。默认为当前时间

+   `expiry`：会话的过期信息，作为`datetime`对象，一个`int`（以秒为单位），或`None`。默认为通过`set_expiry()`存储在会话中的值，如果有的话，或`None`

## get_expiry_date()

返回此会话将过期的日期。对于没有自定义过期时间（或者设置为在浏览器关闭时过期）的会话，这将等于从现在开始`SESSION_COOKIE_AGE`秒的日期。此函数接受与`get_expiry_age()`相同的关键字参数。

## get_expire_at_browser_close()

返回`True`或`False`，取决于用户的会话 cookie 是否在用户的 Web 浏览器关闭时过期。

## clear_expired()

从会话存储中删除过期的会话。这个类方法由`clearsessions`调用。

## cycle_key()

在保留当前会话数据的同时创建一个新的会话密钥。`django.contrib.auth.login()`调用此方法以减轻会话固定。

# 会话对象指南

+   在`request.session`上使用普通的 Python 字符串作为字典键。这更多是一种约定而不是一条硬性规定。

+   以下划线开头的会话字典键是由 Django 内部使用的保留字。

不要用新对象覆盖`request.session`，也不要访问或设置其属性。像使用 Python 字典一样使用它。

# 会话序列化

在 1.6 版本之前，Django 默认使用`pickle`对会话数据进行序列化后存储在后端。如果您使用签名的 cookie 会话后端并且`SECRET_KEY`被攻击者知晓（Django 本身没有固有的漏洞会导致泄漏），攻击者可以在其会话中插入一个字符串，该字符串在反序列化时在服务器上执行任意代码。这种技术简单易行，并且在互联网上很容易获得。

尽管 cookie 会话存储对 cookie 存储的数据进行签名以防篡改，但`SECRET_KEY`泄漏会立即升级为远程代码执行漏洞。可以通过使用 JSON 而不是`pickle`对会话数据进行序列化来减轻此攻击。为了方便这一点，Django 1.5.3 引入了一个新的设置`SESSION_SERIALIZER`，用于自定义会话序列化格式。为了向后兼容，Django 1.5.x 中此设置默认使用`django.contrib.sessions.serializers.PickleSerializer`，但为了加强安全性，从 Django 1.6 开始默认使用`django.contrib.sessions.serializers.JSONSerializer`。

即使在自定义序列化器中描述的注意事项中，我们强烈建议坚持使用 JSON 序列化*特别是如果您使用 cookie 后端*。

## 捆绑的序列化器

### 序列化器.JSONSerializer

从`django.core.signing`的 JSON 序列化器周围的包装器。只能序列化基本数据类型。此外，由于 JSON 仅支持字符串键，请注意在`request.session`中使用非字符串键将无法按预期工作：

```py
>>> # initial assignment 
>>> request.session[0] = 'bar' 
>>> # subsequent requests following serialization & deserialization 
>>> # of session data 
>>> request.session[0]  # KeyError 
>>> request.session['0'] 
'bar' 

```

请参阅自定义序列化器部分，了解 JSON 序列化的限制详情。

### 序列化器.PickleSerializer

支持任意 Python 对象，但如上所述，如果`SECRET_KEY`被攻击者知晓，可能会导致远程代码执行漏洞。

## 编写自己的序列化器

请注意，与`PickleSerializer`不同，`JSONSerializer`无法处理任意 Python 数据类型。通常情况下，方便性和安全性之间存在权衡。如果您希望在 JSON 支持的会话中存储更高级的数据类型，包括`datetime`和`Decimal`，则需要编写自定义序列化器（或在将这些值存储在`request.session`之前将其转换为 JSON 可序列化对象）。

虽然序列化这些值相当简单（`django.core.serializers.json.DateTimeAwareJSONEncoder`可能会有所帮助），但编写一个可靠地获取与输入相同内容的解码器更加脆弱。例如，您可能会冒返回实际上是字符串的`datetime`的风险，只是碰巧与`datetime`选择的相同格式相匹配）。

您的序列化器类必须实现两个方法，`dumps(self, obj)`和`loads(self, data)`，分别用于序列化和反序列化会话数据字典。

# 设置测试 cookie

作为便利，Django 提供了一种简单的方法来测试用户的浏览器是否接受 cookie。只需在视图中调用`request.session`的`set_test_cookie()`方法，并在随后的视图中调用`test_cookie_worked()`，而不是在同一视图调用中。

`set_test_cookie()`和`test_cookie_worked()`之间的这种尴尬分离是由于 cookie 的工作方式。当您设置一个 cookie 时，实际上无法确定浏览器是否接受它，直到浏览器的下一个请求。在验证测试 cookie 有效后，请使用`delete_test_cookie()`进行清理是一个良好的做法。

以下是典型的用法示例：

```py
def login(request): 
    if request.method == 'POST': 
        if request.session.test_cookie_worked(): 
            request.session.delete_test_cookie() 
            return HttpResponse("You're logged in.") 
        else: 
            return HttpResponse("Please enable cookies and try again.") 
    request.session.set_test_cookie() 
    return render_to_response('foo/login_form.html') 

```

# 在视图之外使用会话

本节中的示例直接从`django.contrib.sessions.backends.db`后端导入`SessionStore`对象。在您自己的代码中，您应该考虑从`SESSION_ENGINE`指定的会话引擎中导入`SessionStore`，如下所示：

```py
>>> from importlib import import_module 
>>> from django.conf import settings 
>>> SessionStore = import_module(settings.SESSION_ENGINE).SessionStore 

```

API 可用于在视图之外操作会话数据：

```py
>>> from django.contrib.sessions.backends.db import SessionStore 
>>> s = SessionStore() 
>>> # stored as seconds since epoch since datetimes are not serializable in JSON. 
>>> s['last_login'] = 1376587691 
>>> s.save() 
>>> s.session_key 
'2b1189a188b44ad18c35e113ac6ceead' 

>>> s = SessionStore(session_key='2b1189a188b44ad18c35e113ac6ceead') 
>>> s['last_login'] 
1376587691 

```

为了减轻会话固定攻击，不存在的会话密钥将被重新生成：

```py
>>> from django.contrib.sessions.backends.db import SessionStore 
>>> s = SessionStore(session_key='no-such-session-here') 
>>> s.save() 
>>> s.session_key 
'ff882814010ccbc3c870523934fee5a2' 

```

如果您使用`django.contrib.sessions.backends.db`后端，每个会话只是一个普通的 Django 模型。`Session`模型在`django/contrib/sessions/models.py`中定义。因为它是一个普通模型，您可以使用普通的 Django 数据库 API 访问会话：

```py
>>> from django.contrib.sessions.models import Session 
>>> s = Session.objects.get(pk='2b1189a188b44ad18c35e113ac6ceead') 
>>> s.expire_date 
datetime.datetime(2005, 8, 20, 13, 35, 12) 
Note that you'll need to call get_decoded() to get the session dictionary. This is necessary because the dictionary is stored in an encoded format: 
>>> s.session_data 
'KGRwMQpTJ19hdXRoX3VzZXJfaWQnCnAyCkkxCnMuMTExY2ZjODI2Yj...' 
>>> s.get_decoded() 
{'user_id': 42} 

```

# 会话保存时

默认情况下，只有在会话已被修改时（即其字典值已被分配或删除）Django 才会保存到会话数据库：

```py
# Session is modified. 
request.session['foo'] = 'bar' 

# Session is modified. 
del request.session['foo'] 

# Session is modified. 
request.session['foo'] = {} 

# Gotcha: Session is NOT modified, because this alters 
# request.session['foo'] instead of request.session. 
request.session['foo']['bar'] = 'baz' 

```

在上面示例的最后一种情况中，我们可以通过在会话对象上设置`modified`属性来明确告诉会话对象已被修改：

```py
request.session.modified = True 

```

要更改此默认行为，请将`SESSION_SAVE_EVERY_REQUEST`设置为`True`。当设置为`True`时，Django 将在每个请求上将会话保存到数据库。请注意，只有在创建或修改会话时才会发送会话 cookie。如果`SESSION_SAVE_EVERY_REQUEST`为`True`，则会在每个请求上发送会话 cookie。类似地，会话 cookie 的`expires`部分在每次发送会话 cookie 时都会更新。如果响应的状态码为 500，则不会保存会话。

# 浏览器长度会话与持久会话

您可以通过`SESSION_EXPIRE_AT_BROWSER_CLOSE`设置来控制会话框架是使用浏览器长度会话还是持久会话。默认情况下，`SESSION_EXPIRE_AT_BROWSER_CLOSE`设置为`False`，这意味着会话 cookie 将在用户的浏览器中存储，直到`SESSION_COOKIE_AGE`。如果您不希望用户每次打开浏览器时都需要登录，请使用此设置。

如果`SESSION_EXPIRE_AT_BROWSER_CLOSE`设置为`True`，Django 将使用浏览器长度的 cookie-即当用户关闭浏览器时立即过期的 cookie。

### 注意

一些浏览器（例如 Chrome）提供设置，允许用户在关闭和重新打开浏览器后继续浏览会话。在某些情况下，这可能会干扰`SESSION_EXPIRE_AT_BROWSER_CLOSE`设置，并阻止会话在关闭浏览器时过期。请在测试启用了`SESSION_EXPIRE_AT_BROWSER_CLOSE`设置的 Django 应用程序时注意这一点。

# 清除会话存储

当用户在您的网站上创建新会话时，会话数据可能会在会话存储中累积。Django 不提供自动清除过期会话。因此，您需要定期清除过期会话。Django 为此提供了一个清理管理命令：`clearsessions`。建议定期调用此命令，例如作为每日 cron 作业。

请注意，缓存后端不会受到此问题的影响，因为缓存会自动删除过时数据。Cookie 后端也不会受到影响，因为会话数据是由用户的浏览器存储的。

# 接下来是什么

接下来，我们将继续研究更高级的 Django 主题，通过检查 Django 的缓存后端。
