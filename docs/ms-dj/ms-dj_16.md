# 第十六章：Django 的缓存框架

动态网站的一个基本权衡是，它们是动态的。每当用户请求一个页面时，Web 服务器都会进行各种计算，从数据库查询到模板渲染到业务逻辑再到创建用户所看到的页面。从处理开销的角度来看，这比标准的从文件系统中读取文件的服务器安排要昂贵得多。

对于大多数 Web 应用程序来说，这种开销并不是什么大问题。大多数 Web 应用程序不是 www.washingtonpost.com 或 www.slashdot.org；它们只是一些流量一般的中小型站点。但对于中高流量的站点来说，尽量减少开销是至关重要的。

这就是缓存的作用。缓存某些东西就是保存昂贵计算的结果，这样你就不必在下一次执行计算。下面是一些伪代码，解释了这在动态生成的网页上是如何工作的：

```py
given a URL, try finding that page in the cache 
if the page is in the cache: 
    return the cached page 
else: 
    generate the page 
    save the generated page in the cache (for next time) 
    return the generated page 

```

Django 自带一个强大的缓存系统，可以让你保存动态页面，这样它们就不必为每个请求重新计算。为了方便起见，Django 提供了不同级别的缓存粒度：你可以缓存特定视图的输出，也可以只缓存难以生成的部分，或者缓存整个站点。

Django 也可以很好地与下游缓存（如 Squid，更多信息请访问 [`www.squid-cache.org/`](http://www.squid-cache.org/)）和基于浏览器的缓存一起使用。这些是你无法直接控制的缓存类型，但你可以通过 HTTP 头提供关于你的站点应该缓存哪些部分以及如何缓存的提示。

# 设置缓存

缓存系统需要进行一些设置。主要是告诉它你的缓存数据应该存放在哪里；是在数据库中、在文件系统中还是直接在内存中。这是一个影响缓存性能的重要决定。

你的缓存偏好设置在设置文件的 `CACHES` 设置中。

## Memcached

Django 原生支持的最快、最高效的缓存类型是 Memcached（更多信息请访问 [`memcached.org/`](http://memcached.org/)），它是一个完全基于内存的缓存服务器，最初是为了处理 LiveJournal.com 上的高负载而开发的，并且后来由 Danga Interactive 开源。它被 Facebook 和 Wikipedia 等网站使用，以减少数据库访问并显著提高站点性能。

Memcached 作为守护进程运行，并被分配了指定的内存量。它所做的就是提供一个快速的接口，用于在缓存中添加、检索和删除数据。所有数据都直接存储在内存中，因此没有数据库或文件系统使用的开销。

安装完 Memcached 本身后，你需要安装一个 Memcached 绑定。有几个 Python Memcached 绑定可用；最常见的两个是 python-memcached（ftp://ftp.tummy.com/pub/python-memcached/）和 pylibmc（[`sendapatch.se/projects/pylibmc/`](http://sendapatch.se/projects/pylibmc/)）。要在 Django 中使用 Memcached：

+   将 `BACKEND` 设置为 `django.core.cache.backends.memcached.MemcachedCache` 或 `django.core.cache.backends.memcached.PyLibMCCache`（取决于你选择的 memcached 绑定）

+   将 `LOCATION` 设置为 `ip:port` 值，其中 `ip` 是 Memcached 守护进程的 IP 地址，`port` 是 Memcached 运行的端口，或者设置为 `unix:path` 值，其中 `path` 是 Memcached Unix socket 文件的路径。

在这个例子中，Memcached 在本地主机（`127.0.0.1`）的端口 11211 上运行，使用 `python-memcached` 绑定：

```py
CACHES = { 
    'default': { 
        'BACKEND': 'django.core.cache.backends.memcached.MemcachedCache', 
        'LOCATION': '127.0.0.1:11211', 
    } 
} 

```

在这个例子中，Memcached 可以通过本地的 Unix socket 文件 `/tmp/memcached.sock` 使用 `python-memcached` 绑定来访问：

```py
CACHES = { 
    'default': { 
        'BACKEND': 'django.core.cache.backends.memcached.MemcachedCache', 
        'LOCATION': 'unix:/tmp/memcached.sock', 
    } 
} 

```

Memcached 的一个优秀特性是它能够在多台服务器上共享缓存。这意味着您可以在多台机器上运行 Memcached 守护程序，并且程序将把这组机器视为*单个*缓存，而无需在每台机器上复制缓存值。要利用这个特性，在`LOCATION`中包含所有服务器地址，可以用分号分隔或作为列表。

在这个例子中，缓存是在 IP 地址`172.19.26.240`和`172.19.26.242`上运行的 Memcached 实例之间共享的，端口都是 11211：

```py
CACHES = { 
    'default': { 
        'BACKEND': 'django.core.cache.backends.memcached.MemcachedCache', 
        'LOCATION': [ 
            '172.19.26.240:11211', 
            '172.19.26.242:11211', 
        ] 
    } 
} 

```

在下面的例子中，缓存是在 IP 地址`172.19.26.240`（端口 11211）、`172.19.26.242`（端口 11212）和`172.19.26.244`（端口 11213）上运行的 Memcached 实例之间共享的：

```py
CACHES = { 
    'default': { 
        'BACKEND': 'django.core.cache.backends.memcached.MemcachedCache', 
        'LOCATION': [ 
            '172.19.26.240:11211', 
            '172.19.26.242:11212', 
            '172.19.26.244:11213', 
        ] 
    } 
} 

```

关于 Memcached 的最后一点是，基于内存的缓存有一个缺点：因为缓存数据存储在内存中，如果服务器崩溃，数据将丢失。

显然，内存并不适用于永久数据存储，因此不要仅依赖基于内存的缓存作为您唯一的数据存储。毫无疑问，Django 缓存后端都不应该用于永久存储-它们都是用于缓存而不是存储的解决方案-但我们在这里指出这一点是因为基于内存的缓存特别是临时的。

## 数据库缓存

Django 可以将其缓存数据存储在您的数据库中。如果您有一个快速、索引良好的数据库服务器，这将效果最佳。要将数据库表用作缓存后端：

+   将`BACKEND`设置为`django.core.cache.backends.db.DatabaseCache`

+   将`LOCATION`设置为`tablename`，即数据库表的名称。这个名称可以是任何你想要的，只要它是一个有效的表名，而且在你的数据库中还没有被使用。

在这个例子中，缓存表的名称是`my_cache_table`：

```py
CACHES = { 
    'default': { 
        'BACKEND': 'django.core.cache.backends.db.DatabaseCache', 
        'LOCATION': 'my_cache_table', 
    } 
} 

```

### 创建缓存表

在使用数据库缓存之前，您必须使用这个命令创建缓存表：

```py
python manage.py createcachetable 

```

这将在您的数据库中创建一个符合 Django 数据库缓存系统期望的正确格式的表。表的名称取自`LOCATION`。如果您使用多个数据库缓存，`createcachetable`会为每个缓存创建一个表。如果您使用多个数据库，`createcachetable`会观察数据库路由器的`allow_migrate()`方法（见下文）。与`migrate`一样，`createcachetable`不会触及现有表。它只会创建缺失的表。

### 多个数据库

如果您在使用多个数据库进行数据库缓存，还需要为数据库缓存表设置路由指令。对于路由的目的，数据库缓存表显示为一个名为`CacheEntry`的模型，在名为`django_cache`的应用程序中。这个模型不会出现在模型缓存中，但模型的详细信息可以用于路由目的。

例如，以下路由器将所有缓存读取操作定向到`cache_replica`，并将所有写操作定向到`cache_primary`。缓存表只会同步到`cache_primary`：

```py
class CacheRouter(object): 
    """A router to control all database cache operations""" 

    def db_for_read(self, model, **hints): 
        # All cache read operations go to the replica 
        if model._meta.app_label in ('django_cache',): 
            return 'cache_replica' 
        return None 

    def db_for_write(self, model, **hints): 
        # All cache write operations go to primary 
        if model._meta.app_label in ('django_cache',): 
            return 'cache_primary' 
        return None 

    def allow_migrate(self, db, model): 
        # Only install the cache model on primary 
        if model._meta.app_label in ('django_cache',): 
            return db == 'cache_primary' 
        return None 

```

如果您没有为数据库缓存模型指定路由指令，缓存后端将使用`default`数据库。当然，如果您不使用数据库缓存后端，您就不需要担心为数据库缓存模型提供路由指令。

## 文件系统缓存

基于文件的后端将每个缓存值序列化并存储为单独的文件。要使用此后端，将`BACKEND`设置为`'django.core.cache.backends.filebased.FileBasedCache'`，并将`LOCATION`设置为适当的目录。

例如，要将缓存数据存储在`/var/tmp/django_cache`中，使用以下设置：

```py
CACHES = { 
    'default': { 
        'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache', 
        'LOCATION': '/var/tmp/django_cache', 
    } 
} 

```

如果您在 Windows 上，将驱动器号放在路径的开头，就像这样：

```py
CACHES = { 
    'default': { 
        'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache', 
        'LOCATION': 'c:/foo/bar', 
    } 
} 

```

目录路径应该是绝对的-也就是说，它应该从文件系统的根目录开始。设置末尾是否加斜杠并不重要。确保此设置指向的目录存在，并且可以被运行您的网页服务器的系统用户读取和写入。继续上面的例子，如果您的服务器以用户`apache`运行，请确保目录`/var/tmp/django_cache`存在，并且可以被用户`apache`读取和写入。

## 本地内存缓存

如果在设置文件中未指定其他缓存，则这是默认缓存。如果您想要内存缓存的速度优势，但又没有运行 Memcached 的能力，请考虑使用本地内存缓存后端。要使用它，请将`BACKEND`设置为`django.core.cache.backends.locmem.LocMemCache`。例如：

```py
CACHES = { 
    'default': { 
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache', 
        'LOCATION': 'unique-snowflake' 
    } 
} 

```

缓存`LOCATION`用于标识单个内存存储。如果您只有一个`locmem`缓存，可以省略`LOCATION`；但是，如果您有多个本地内存缓存，您将需要为其中至少一个分配一个名称，以便将它们分开。

请注意，每个进程将拥有自己的私有缓存实例，这意味着不可能进行跨进程缓存。这显然也意味着本地内存缓存不是特别内存高效，因此在生产环境中可能不是一个好选择。但对于开发来说是不错的选择。

## 虚拟缓存（用于开发）

最后，Django 附带了一个虚拟缓存，它实际上不缓存-它只是实现了缓存接口而不执行任何操作。如果您的生产站点在各个地方都使用了重度缓存，但在开发/测试环境中不想缓存并且不想改变代码以特殊处理后者，这将非常有用。要激活虚拟缓存，请将`BACKEND`设置如下：

```py
CACHES = { 
    'default': { 
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache', 
    } 
} 

```

## 使用自定义缓存后端

尽管 Django 默认支持多种缓存后端，但有时您可能希望使用自定义的缓存后端。要在 Django 中使用外部缓存后端，请将 Python 导入路径作为`CACHES`设置的`BACKEND`，如下所示：

```py
CACHES = { 
    'default': { 
        'BACKEND': 'path.to.backend', 
    } 
} 

```

如果您正在构建自己的后端，可以使用标准缓存后端作为参考实现。您可以在 Django 源代码的`django/core/cache/backends/`目录中找到这些代码。

### 注意

除非有一个真正令人信服的理由，比如不支持它们的主机，否则您应该坚持使用 Django 提供的缓存后端。它们经过了充分测试，易于使用。

## 缓存参数

每个缓存后端都可以提供额外的参数来控制缓存行为。这些参数作为`CACHES`设置中的额外键提供。有效参数如下：

+   `TIMEOUT`：用于缓存的默认超时时间（以秒为单位）。此参数默认为 300 秒（5 分钟）。您可以将`TIMEOUT`设置为`None`，以便默认情况下缓存键永不过期。值为`0`会导致键立即过期（实际上不缓存）。

+   `OPTIONS`：应传递给缓存后端的任何选项。有效选项的列表将随着每个后端的不同而变化，并且由第三方库支持的缓存后端将直接将它们的选项传递给底层缓存库。

+   实现自己的清除策略的缓存后端（即`locmem`，`filesystem`和`database`后端）将遵守以下选项：

+   `MAX_ENTRIES`：在旧值被删除之前缓存中允许的最大条目数。此参数默认为`300`。

+   `CULL_FREQUENCY`：当达到`MAX_ENTRIES`时被删除的条目比例。实际比例是`1 / CULL_FREQUENCY`，因此将`CULL_FREQUENCY`设置为`2`，以在达到`MAX_ENTRIES`时删除一半的条目。此参数应为整数，默认为`3`。

+   `CULL_FREQUENCY`的值为`0`意味着当达到`MAX_ENTRIES`时整个缓存将被清除。在某些后端（特别是`database`）上，这样做会使清除*更*快，但会增加缓存未命中的次数。

+   `KEY_PREFIX`：一个字符串，将自动包含（默认情况下是前置）到 Django 服务器使用的所有缓存键中。

+   `VERSION`：Django 服务器生成的缓存键的默认版本号。

+   `KEY_FUNCTION`：包含一个点路径到一个函数的字符串，该函数定义如何将前缀、版本和键组合成最终的缓存键。

在这个例子中，文件系统后端被配置为超时 60 秒，并且最大容量为 1000 个项目：

```py
CACHES = { 
    'default': { 
        'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache', 
        'LOCATION': '/var/tmp/django_cache', 
        'TIMEOUT': 60, 
        'OPTIONS': {'MAX_ENTRIES': 1000} 
    } 
} 

```

# 每个站点的缓存

设置缓存后，使用缓存的最简单方法是缓存整个站点。您需要将`'django.middleware.cache.UpdateCacheMiddleware'`和`'django.middleware.cache.FetchFromCacheMiddleware'`添加到您的`MIDDLEWARE_CLASSES`设置中，就像这个例子中一样：

```py
MIDDLEWARE_CLASSES = [ 
    'django.middleware.cache.UpdateCacheMiddleware', 
    'django.middleware.common.CommonMiddleware', 
    'django.middleware.cache.FetchFromCacheMiddleware', 
] 

```

### 注意

不，这不是一个打字错误：更新中间件必须在列表中首先出现，获取中间件必须在最后出现。细节有点模糊，但是如果您想要完整的故事，请参阅下一章中的 MIDDLEWARE_CLASSES 顺序。

然后，将以下必需的设置添加到您的 Django 设置文件中：

+   `CACHE_MIDDLEWARE_ALIAS`：用于存储的缓存别名。

+   `CACHE_MIDDLEWARE_SECONDS`：每个页面应该被缓存的秒数。

+   `CACHE_MIDDLEWARE_KEY_PREFIX`-：如果缓存跨多个使用相同 Django 安装的站点共享，则将其设置为站点的名称，或者是对此 Django 实例唯一的其他字符串，以防止键冲突。如果您不在乎，可以使用空字符串。

`FetchFromCacheMiddleware`使用`status 200`缓存`GET`和`HEAD`响应，其中请求和响应头允许。对于具有不同查询参数的相同 URL 的请求的响应被认为是唯一的页面，并且被单独缓存。此中间件期望`HEAD`请求以与相应的`GET`请求相同的响应头进行响应；在这种情况下，它可以为`HEAD`请求返回缓存的`GET`响应。此外，`UpdateCacheMiddleware`自动在每个`HttpResponse`中设置一些头：

+   将`Last-Modified`头设置为请求新的（未缓存）页面时的当前日期/时间。

+   将`Expires`头设置为当前日期/时间加上定义的`CACHE_MIDDLEWARE_SECONDS`。

+   将`Cache-Control`头设置为页面的最大年龄-同样，从`CACHE_MIDDLEWARE_SECONDS`设置。

如果视图设置了自己的缓存到期时间（即它在其`max-age`部分中有一个），则页面将被缓存直到到期时间，而不是`CACHE_MIDDLEWARE_SECONDS`。 

`Cache-Control`头）那么页面将被缓存直到到期时间，而不是`CACHE_MIDDLEWARE_SECONDS`。使用`django.views.decorators.cache`中的装饰器，您可以轻松地设置视图的到期时间（使用`cache_control`装饰器）或禁用视图的缓存（使用`never_cache`装饰器）。有关这些装饰器的更多信息，请参阅使用其他标头部分。

如果`USE_I18N`设置为`True`，则生成的缓存键将包括活动语言的名称。这样，您可以轻松地缓存多语言站点，而无需自己创建缓存键。

当`USE_L10N`设置为`True`时，缓存键还包括活动语言，当`USE_TZ`设置为`True`时，还包括当前时区。

# 每个视图的缓存

使用缓存框架的更细粒度的方法是通过缓存单个视图的输出。`django.views.decorators.cache`定义了一个`cache_page`装饰器，它将自动为您缓存视图的响应。使用起来很容易：

```py
from django.views.decorators.cache import cache_page 

@cache_page(60 * 15) 
def my_view(request): 
    ... 

```

`cache_page`接受一个参数：缓存超时时间，以秒为单位。在上面的例子中，`my_view()`视图的结果将被缓存 15 分钟。（请注意，我已经将其写成`60 * 15`，以便阅读。`60 * 15`将被计算为`900`-也就是说，15 分钟乘以 60 秒每分钟。）

每个视图的缓存，就像每个站点的缓存一样，是基于 URL 的。如果多个 URL 指向同一个视图，每个 URL 将被单独缓存。继续`my_view`的例子，如果您的 URLconf 如下所示：

```py
urlpatterns = [ 
    url(r'^foo/([0-9]{1,2})/$', my_view), 
] 

```

然后对`/foo/1/`和`/foo/23/`的请求将被分别缓存，正如你可能期望的那样。但一旦请求了特定的 URL（例如`/foo/23/`），随后对该 URL 的请求将使用缓存。

`cache_page`还可以接受一个可选的关键字参数`cache`，它指示装饰器在缓存视图结果时使用特定的缓存（来自你的`CACHES`设置）。

默认情况下，将使用`default`缓存，但你可以指定任何你想要的缓存：

```py
@cache_page(60 * 15, cache="special_cache") 
def my_view(request): 
    ... 

```

你也可以在每个视图的基础上覆盖缓存前缀。`cache_page`接受一个可选的关键字参数`key_prefix`，它的工作方式与中间件的`CACHE_MIDDLEWARE_KEY_PREFIX`设置相同。可以像这样使用：

```py
@cache_page(60 * 15, key_prefix="site1") 
def my_view(request): 
    ... 

```

`key_prefix`和`cache`参数可以一起指定。`key_prefix`参数和在`CACHES`下指定的`KEY_PREFIX`将被连接起来。

## 在 URLconf 中指定每个视图的缓存

前一节中的示例已经硬编码了视图被缓存的事实，因为`cache_page`会直接修改`my_view`函数。这种方法将你的视图与缓存系统耦合在一起，这对于几个原因来说都不理想。例如，你可能希望在另一个没有缓存的站点上重用视图函数，或者你可能希望将视图分发给可能希望在没有被缓存的情况下使用它们的人。

解决这些问题的方法是在 URLconf 中指定每个视图的缓存，而不是在视图函数旁边。这样做很容易：只需在 URLconf 中引用视图函数时用`cache_page`包装视图函数即可。

这是之前的旧 URLconf：

```py
urlpatterns = [ 
    url(r'^foo/([0-9]{1,2})/$', my_view), 
] 

```

这里是相同的内容，`my_view`被包裹在`cache_page`中：

```py
from django.views.decorators.cache import cache_page 

urlpatterns = [ 
    url(r'^foo/([0-9]{1,2})/$', cache_page(60 * 15)(my_view)), 
] 

```

# 模板片段缓存

如果你想要更多的控制，你也可以使用`cache`模板标签来缓存模板片段。为了让你的模板可以访问这个标签，放置

在模板顶部附近使用`{% load cache %}`。`{% cache %}`模板标签会缓存给定时间内的块内容。

它至少需要两个参数：缓存超时（以秒为单位）和要给缓存片段的名称。名称将被直接使用，不要使用变量。

例如：

```py
{% load cache %} 
{% cache 500 sidebar %} 
    .. sidebar .. 
{% endcache %} 

```

有时你可能希望根据片段内部出现的一些动态数据来缓存多个副本的片段。

例如，你可能希望为站点的每个用户使用前面示例中使用的侧边栏的单独缓存副本。通过向`{% cache %}`模板标签传递额外的参数来唯一标识缓存片段来实现这一点：

```py
{% load cache %} 
{% cache 500 sidebar request.user.username %} 
    .. sidebar for logged in user .. 
{% endcache %} 

```

指定多个参数来标识片段是完全可以的。只需向`{% cache %}`传递所需的参数即可。如果`USE_I18N`设置为`True`，则每个站点的中间件缓存将遵循活动语言。

对于`cache`模板标签，你可以使用模板中可用的翻译特定变量之一来实现相同的结果：

```py
{% load i18n %} 
{% load cache %} 

{% get_current_language as LANGUAGE_CODE %} 

{% cache 600 welcome LANGUAGE_CODE %} 
    {% trans "Welcome to example.com" %} 
{% endcache %} 

```

缓存超时可以是一个模板变量，只要模板变量解析为整数值即可。

例如，如果模板变量`my_timeout`设置为值`600`，那么以下两个示例是等价的：

```py
{% cache 600 sidebar %} ... {% endcache %} 
{% cache my_timeout sidebar %} ... {% endcache %} 

```

这个功能在模板中避免重复很有用。你可以在一个地方设置超时，然后只需重用该值。默认情况下，缓存标签将尝试使用名为`template_fragments`的缓存。如果没有这样的缓存存在，它将退回到使用默认缓存。你可以选择一个备用的缓存后端来与`using`关键字参数一起使用，这必须是标签的最后一个参数。

```py
{% cache 300 local-thing ...  using="localcache" %} 

```

指定未配置的缓存名称被认为是一个错误。

如果你想获取用于缓存片段的缓存键，你可以使用`make_template_fragment_key`。`fragment_name`与`cache`模板标签的第二个参数相同；`vary_on`是传递给标签的所有额外参数的列表。这个函数对于使缓存项无效或覆盖缓存项可能很有用，例如：

```py
>>> from django.core.cache import cache 
>>> from django.core.cache.utils import make_template_fragment_key 
# cache key for {% cache 500 sidebar username %} 
>>> key = make_template_fragment_key('sidebar', [username]) 
>>> cache.delete(key) # invalidates cached template fragment 

```

# 低级别缓存 API

有时，缓存整个渲染页面并不会带来太多好处，实际上，这种方式过度。例如，您的站点可能包括一个视图，其结果取决于几个昂贵的查询，这些查询的结果在不同的时间间隔内发生变化。在这种情况下，使用每个站点或每个视图缓存策略提供的全页缓存并不理想，因为您不希望缓存整个结果（因为某些数据经常更改），但仍然希望缓存很少更改的结果。

对于这样的情况，Django 公开了一个简单的低级缓存 API。您可以使用此 API 以任何您喜欢的粒度存储对象。您可以缓存任何可以安全进行 pickle 的 Python 对象：字符串，字典，模型对象列表等（大多数常见的 Python 对象都可以进行 pickle；有关 pickling 的更多信息，请参阅 Python 文档）。

## 访问缓存

您可以通过类似字典的对象`django.core.cache.caches`访问`CACHES`设置中配置的缓存。在同一线程中对同一别名的重复请求将返回相同的对象。

```py
>>> from django.core.cache import caches 
>>> cache1 = caches['myalias'] 
>>> cache2 = caches['myalias'] 
>>> cache1 is cache2 
True 

```

如果命名键不存在，则将引发`InvalidCacheBackendError`。为了提供线程安全性，将为每个线程返回缓存后端的不同实例。

作为快捷方式，默认缓存可用为`django.core.cache.cache`：

```py
>>> from django.core.cache import cache 

```

此对象等同于`caches['default']`。

## 基本用法

基本接口是`set（key，value，timeout）`和`get（key）`：

```py
>>> cache.set('my_key', 'hello, world!', 30) 
>>> cache.get('my_key') 
'hello, world!' 

```

`timeout`参数是可选的，默认为`CACHES`设置中适当后端的`timeout`参数（如上所述）。这是值应在缓存中存储的秒数。将`None`传递给`timeout`将永远缓存该值。`timeout`为`0`将不会缓存该值。如果对象在缓存中不存在，则`cache.get（）`将返回`None`：

```py
# Wait 30 seconds for 'my_key' to expire... 

>>> cache.get('my_key') 
None 

```

我们建议不要将文字值`None`存储在缓存中，因为您无法区分存储的`None`值和由返回值`None`表示的缓存未命中。`cache.get（）`可以接受`default`参数。这指定如果对象在缓存中不存在时要返回的值：

```py
>>> cache.get('my_key', 'has expired') 
'has expired' 

```

要仅在键不存在时添加键，请使用`add（）`方法。它接受与`set（）`相同的参数，但如果指定的键已经存在，则不会尝试更新缓存：

```py
>>> cache.set('add_key', 'Initial value') 
>>> cache.add('add_key', 'New value') 
>>> cache.get('add_key') 
'Initial value' 

```

如果您需要知道`add（）`是否将值存储在缓存中，可以检查返回值。如果存储了该值，则返回`True`，否则返回`False`。还有一个`get_many（）`接口，只会命中一次缓存。`get_many（）`返回一个包含实际存在于缓存中的所有您请求的键的字典（并且尚未过期）：

```py
>>> cache.set('a', 1) 
>>> cache.set('b', 2) 
>>> cache.set('c', 3) 
>>> cache.get_many(['a', 'b', 'c']) 
{'a': 1, 'b': 2, 'c': 3} 

```

要更有效地设置多个值，请使用`set_many（）`传递键值对的字典：

```py
>>> cache.set_many({'a': 1, 'b': 2, 'c': 3}) 
>>> cache.get_many(['a', 'b', 'c']) 
{'a': 1, 'b': 2, 'c': 3} 

```

与`cache.set（）`类似，`set_many（）`接受一个可选的`timeout`参数。您可以使用`delete（）`显式删除键。这是清除特定对象的缓存的简单方法：

```py
>>> cache.delete('a') 

```

如果要一次清除一堆键，`delete_many（）`可以接受要清除的键的列表：

```py
>>> cache.delete_many(['a', 'b', 'c']) 

```

最后，如果要删除缓存中的所有键，请使用`cache.clear（）`。请注意；`clear（）`将从缓存中删除所有内容，而不仅仅是应用程序设置的键。

```py
>>> cache.clear() 

```

您还可以使用`incr（）`或`decr（）`方法来增加或减少已经存在的键。默认情况下，现有的缓存值将增加或减少 1。可以通过向增量/减量调用提供参数来指定其他增量/减量值。

如果您尝试增加或减少不存在的缓存键，则会引发`ValueError`。

```py
>>> cache.set('num', 1) 
>>> cache.incr('num') 
2 
>>> cache.incr('num', 10) 
12 
>>> cache.decr('num') 
11 
>>> cache.decr('num', 5) 
6 

```

如果缓存后端实现了`close（）`，则可以使用`close（）`关闭与缓存的连接。

```py
>>> cache.close() 

```

请注意，对于不实现`close`方法的缓存，`close（）`是一个空操作。

## 缓存键前缀

如果您在服务器之间共享缓存实例，或在生产和开发环境之间共享缓存实例，那么一个服务器缓存的数据可能会被另一个服务器使用。如果缓存数据在服务器之间的格式不同，这可能会导致一些非常难以诊断的问题。

为了防止这种情况发生，Django 提供了为服务器中使用的所有缓存键添加前缀的功能。当保存或检索特定缓存键时，Django 将自动使用`KEY_PREFIX`缓存设置的值作为缓存键的前缀。通过确保每个 Django 实例具有不同的`KEY_PREFIX`，您可以确保缓存值不会发生冲突。

## 缓存版本

当您更改使用缓存值的运行代码时，您可能需要清除任何现有的缓存值。这样做的最简单方法是刷新整个缓存，但这可能会导致仍然有效和有用的缓存值的丢失。Django 提供了一种更好的方法来定位单个缓存值。

Django 的缓存框架具有系统范围的版本标识符，使用`VERSION`缓存设置指定。此设置的值将自动与缓存前缀和用户提供的缓存键结合，以获取最终的缓存键。

默认情况下，任何键请求都将自动包括站点默认的缓存键版本。但是，原始缓存函数都包括一个`version`参数，因此您可以指定要设置或获取的特定缓存键版本。例如：

```py
# Set version 2 of a cache key 
>>> cache.set('my_key', 'hello world!', version=2) 
# Get the default version (assuming version=1) 
>>> cache.get('my_key') 
None 
# Get version 2 of the same key 
>>> cache.get('my_key', version=2) 
'hello world!' 

```

特定键的版本可以使用`incr_version()`和`decr_version()`方法进行增加和减少。这使得特定键可以升级到新版本，而不影响其他键。继续我们之前的例子：

```py
# Increment the version of 'my_key' 
>>> cache.incr_version('my_key') 
# The default version still isn't available 
>>> cache.get('my_key') 
None 
# Version 2 isn't available, either 
>>> cache.get('my_key', version=2) 
None 
# But version 3 *is* available 
>>> cache.get('my_key', version=3) 
'hello world!' 

```

## 缓存键转换

如前两节所述，用户提供的缓存键不会直接使用-它与缓存前缀和键版本结合以提供最终的缓存键。默认情况下，这三个部分使用冒号连接以生成最终字符串：

```py
def make_key(key, key_prefix, version): 
    return ':'.join([key_prefix, str(version), key]) 

```

如果您想以不同的方式组合部分，或对最终键应用其他处理（例如，对键部分进行哈希摘要），可以提供自定义键函数。`KEY_FUNCTION`缓存设置指定了与上面`make_key()`原型匹配的函数的点路径。如果提供了此自定义键函数，它将被用于替代默认的键组合函数。

## 缓存键警告

Memcached，最常用的生产缓存后端，不允许缓存键超过 250 个字符或包含空格或控制字符，使用这样的键将导致异常。为了鼓励可移植的缓存代码并最小化不愉快的惊喜，其他内置缓存后端在使用可能导致在 memcached 上出错的键时会发出警告（`django.core.cache.backends.base.CacheKeyWarning`）。

如果您正在使用可以接受更广泛键范围的生产后端（自定义后端或非 memcached 内置后端之一），并且希望在没有警告的情况下使用此更广泛范围，您可以在一个`INSTALLED_APPS`的`management`模块中使用以下代码来消除`CacheKeyWarning`：

```py
import warnings 

from django.core.cache import CacheKeyWarning 

warnings.simplefilter("ignore", CacheKeyWarning) 

```

如果您想为内置后端之一提供自定义键验证逻辑，可以对其进行子类化，仅覆盖`validate_key`方法，并按照使用自定义缓存后端的说明进行操作。

例如，要为`locmem`后端执行此操作，请将此代码放入一个模块中：

```py
from django.core.cache.backends.locmem import LocMemCache 

class CustomLocMemCache(LocMemCache): 
    def validate_key(self, key): 
        # Custom validation, raising exceptions or warnings as needed. 
        # ... 

```

...并在`CACHES`设置的`BACKEND`部分使用此类的点 Python 路径。

# 下游缓存

到目前为止，本章重点介绍了缓存自己的数据。但是，与 Web 开发相关的另一种缓存也很重要：下游缓存执行的缓存。这些是在请求到达您的网站之前就为用户缓存页面的系统。以下是一些下游缓存的示例：

+   您的 ISP 可能会缓存某些页面，因此，如果您从`http://example.com/`请求页面，则您的 ISP 将向您发送页面，而无需直接访问`example.com`。`example.com`的维护者对此缓存一无所知；ISP 位于`example.com`和您的 Web 浏览器之间，透明地处理所有缓存。

+   您的 Django 网站可能位于*代理缓存*之后，例如 Squid Web 代理缓存（有关更多信息，请访问[`www.squid-cache.org/`](http://www.squid-cache.org/)），该缓存可提高页面性能。在这种情况下，每个请求首先将由代理处理，只有在需要时才会传递给您的应用程序。

+   您的 Web 浏览器也会缓存页面。如果网页发送适当的头，则您的浏览器将对该页面的后续请求使用本地缓存副本，而无需再次联系网页以查看其是否已更改。

下游缓存是一个不错的效率提升，但也存在危险：许多网页的内容基于认证和一系列其他变量而异，盲目保存页面的缓存系统可能向随后访问这些页面的访问者公开不正确或敏感的数据。

例如，假设您运营一个 Web 电子邮件系统，收件箱页面的内容显然取决于哪个用户已登录。如果 ISP 盲目缓存您的站点，那么通过该 ISP 首次登录的用户将使其特定于用户的收件箱页面缓存供站点的后续访问者使用。这不好。

幸运的是，HTTP 提供了解决这个问题的方法。存在许多 HTTP 头，用于指示下游缓存根据指定的变量延迟其缓存内容，并告诉缓存机制不要缓存特定页面。我们将在接下来的部分中查看其中一些头。

# 使用 vary 头

`Vary`头定义了缓存机制在构建其缓存键时应考虑哪些请求头。例如，如果网页的内容取决于用户的语言首选项，则称该页面取决于语言。默认情况下，Django 的缓存系统使用请求的完全限定 URL 创建其缓存键，例如`http://www.example.com/stories/2005/?order_by=author`。

这意味着对该 URL 的每个请求都将使用相同的缓存版本，而不考虑用户代理的差异，例如 cookie 或语言首选项。但是，如果此页面根据请求头的某些差异（例如 cookie、语言或用户代理）生成不同的内容，则需要使用`Vary`头来告诉缓存机制页面输出取决于这些内容。

要在 Django 中执行此操作，请使用方便的`django.views.decorators.vary.vary_on_headers()`视图装饰器，如下所示：

```py
from django.views.decorators.vary import vary_on_headers 

@vary_on_headers('User-Agent') 
def my_view(request): 
    # ... 

```

在这种情况下，缓存机制（例如 Django 自己的缓存中间件）将为每个唯一的用户代理缓存页面的单独版本。使用`vary_on_headers`装饰器而不是手动设置`Vary`头（使用类似`response['Vary'] = 'user-agent'`的东西）的优势在于，装饰器会添加到`Vary`头（如果已经存在），而不是从头开始设置它，并可能覆盖已经存在的任何内容。您可以将多个头传递给`vary_on_headers()`：

```py
@vary_on_headers('User-Agent', 'Cookie') 
def my_view(request): 
    # ... 

```

这告诉下游缓存在两者上变化，这意味着每个用户代理和 cookie 的组合都将获得自己的缓存值。例如，具有用户代理`Mozilla`和 cookie 值`foo=bar`的请求将被视为与具有用户代理`Mozilla`和 cookie 值`foo=ham`的请求不同。因为在 cookie 上变化是如此常见，所以有一个`django.views.decorators.vary.vary_on_cookie()`装饰器。这两个视图是等效的。

```py
@vary_on_cookie 
def my_view(request): 
    # ... 

@vary_on_headers('Cookie') 
def my_view(request): 
    # ... 

```

您传递给`vary_on_headers`的标头不区分大小写；`User-Agent`与`user-agent`是相同的。您还可以直接使用辅助函数`django.utils.cache.patch_vary_headers()`。此函数设置或添加到`Vary`标头。例如：

```py
from django.utils.cache import patch_vary_headers 

def my_view(request): 
    # ... 
    response = render_to_response('template_name', context) 
    patch_vary_headers(response, ['Cookie']) 
    return response 

```

`patch_vary_headers`将`HttpResponse`实例作为其第一个参数，并将不区分大小写的标头名称列表/元组作为其第二个参数。有关`Vary`标头的更多信息，请参阅官方 Vary 规范（有关更多信息，请访问[`www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.44`](http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.44)）。

# 控制缓存：使用其他标头

缓存的其他问题是数据的隐私和数据应该存储在缓存级联中的哪个位置的问题。用户通常面临两种缓存：自己的浏览器缓存（私有缓存）和其提供者的缓存（公共缓存）。

公共缓存由多个用户使用，并由其他人控制。这会带来敏感数据的问题-您不希望您的银行账号存储在公共缓存中。因此，Web 应用程序需要一种告诉缓存哪些数据是私有的，哪些是公共的方法。

解决方案是指示页面的缓存应该是私有的。在 Django 中，使用`cache_control`视图装饰器。例如：

```py
from django.views.decorators.cache import cache_control 

@cache_control(private=True) 
def my_view(request): 
    # ... 

```

此装饰器负责在后台发送适当的 HTTP 标头。请注意，缓存控制设置`private`和`public`是互斥的。装饰器确保如果应该设置`private`，则删除公共指令（反之亦然）。

两个指令的一个示例用法是提供公共和私有条目的博客网站。公共条目可以在任何共享缓存上缓存。以下代码使用`django.utils.cache.patch_cache_control()`，手动修改缓存控制标头的方法（它由`cache_control`装饰器内部调用）：

```py
from django.views.decorators.cache import patch_cache_control 
from django.views.decorators.vary import vary_on_cookie 

@vary_on_cookie 
def list_blog_entries_view(request): 
    if request.user.is_anonymous(): 
        response = render_only_public_entries() 
        patch_cache_control(response, public=True) 
    else: 
        response = render_private_and_public_entries(request.user) 
        patch_cache_control(response, private=True) 

    return response 

```

还有其他控制缓存参数的方法。例如，HTTP 允许应用程序执行以下操作：

+   定义页面应缓存的最长时间。

+   指定缓存是否应该始终检查更新版本，仅在没有更改时提供缓存内容。（某些缓存可能会在服务器页面更改时提供缓存内容，仅因为缓存副本尚未过期。）

在 Django 中，使用`cache_control`视图装饰器来指定这些缓存参数。在此示例中，`cache_control`告诉缓存在每次访问时重新验证缓存，并将缓存版本存储最多 3600 秒：

```py
from django.views.decorators.cache import cache_control 

@cache_control(must_revalidate=True, max_age=3600) 
def my_view(request): 
    # ... 

```

`cache_control()`中的任何有效的`Cache-Control` HTTP 指令在`cache_control()`中都是有效的。以下是完整列表：

+   `public=True`

+   `private=True`

+   `no_cache=True`

+   `no_transform=True`

+   `must_revalidate=True`

+   `proxy_revalidate=True`

+   `max_age=num_seconds`

+   `s_maxage=num_seconds`

有关 Cache-Control HTTP 指令的解释，请参阅 Cache-Control 规范（有关更多信息，请访问[`www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.9`](http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.9)）。 （请注意，缓存中间件已经使用`CACHE_MIDDLEWARE_SECONDS`设置的值设置了缓存标头的`max-age`。如果您在`cache_control`装饰器中使用自定义的`max_age`，装饰器将优先，并且标头值将被正确合并。）

如果要使用标头完全禁用缓存，`django.views.decorators.cache.never_cache`是一个视图装饰器，它添加标头以确保响应不会被浏览器或其他缓存缓存。例如：

```py
from django.views.decorators.cache import never_cache 

@never_cache 
def myview(request): 
    # ... 

```

# 接下来是什么？

在下一章中，我们将看一下 Django 的中间件。
