# 第四章：在前 10 部电影中进行缓存

在本章中，我们将使用我们的用户投票的票数来构建 MyMDB 中前 10 部电影的列表。为了确保这个受欢迎的页面保持快速加载，我们将看看帮助我们优化网站的工具。最后，我们将看看 Django 的缓存 API 以及如何使用它来优化我们的项目。

在本章中，我们将做以下事情：

+   使用聚合查询创建一个前 10 部电影列表

+   了解 Django 的工具来衡量优化

+   使用 Django 的缓存 API 来缓存昂贵操作的结果

让我们从制作我们的前 10 部电影列表页面开始。

# 创建前 10 部电影列表

为了构建我们的前 10 部电影列表，我们将首先创建一个新的`MovieManager`方法，然后在新的视图和模板中使用它。我们还将更新基本模板中的顶部标题，以便从每个页面轻松访问列表。

# 创建 MovieManager.top_movies()

我们的`MovieManager`类需要能够返回一个由我们的用户投票选出的最受欢迎电影的`QuerySet`对象。我们使用了一个天真的受欢迎度公式，即![](img/e7400a23-0fe5-4725-8751-68f43e1455d2.png)票数减去![](img/933b0552-c3d0-4633-bbf3-87a483749b81.png)票数的总和。就像在第二章*将用户添加到 MyMDB*中一样，我们将使用`QuerySet.annotate()`方法来进行聚合查询以计算投票数。

让我们将我们的新方法添加到`django/core/models.py`：

```py
from django.db.models.aggregates import (
    Sum
)

class MovieManager(models.Manager):

    # other methods omitted

    def top_movies(self, limit=10):
        qs = self.get_queryset()
        qs = qs.annotate(
            vote_sum=Sum('vote__value'))
        qs = qs.exclude(
            vote_sum=None)
        qs = qs.order_by('-vote_sum')
        qs = qs[:limit]
        return qs
```

我们按照它们的票数总和（降序）对结果进行排序，以获得我们的前 10 部电影列表。然而，我们面临的问题是，一些电影没有投票，因此它们的`vote_sum`值将为`NULL`。不幸的是，`NULL`将首先被 Postgres 排序。我们将通过添加一个约束来解决这个问题，即没有投票的电影，根据定义，不会成为前 10 部电影之一。我们使用`QuerySet.exclude`（与`QuerySet.filter`相反）来删除没有投票的电影。

这是我们第一次看到一个`QuerySet`对象被切片。除非提供步长，否则`QuerySet`对象不会被切片评估（例如，`qs [10:20:2]`会使`QuerySet`对象立即被评估并返回第 10、12、14、16 和 18 行）。

现在我们有了一个合适的`Movie`模型实例的`QuerySet`对象，我们可以在视图中使用`QuerySet`对象。

# 创建 TopMovies 视图

由于我们的`TopMovies`视图需要显示一个列表，我们可以像以前一样使用 Django 的`ListView`。让我们更新`django/core/views.py`：

```py
from django.views.generic import ListView
from core.models import Movie

class TopMovies(ListView):
    template_name = 'core/top_movies_list.html'
    queryset = Movie.objects.top_movies(
        limit=10)
```

与以前的`ListView`类不同，我们需要指定一个`template_name`属性。否则，`ListView`将尝试使用`core/movie_list.html`，这是`MovieList`视图使用的。

接下来，让我们创建我们的模板。

# 创建 top_movies_list.html 模板

我们的前 10 部电影页面不需要分页，所以模板非常简单。让我们创建`django/core/templates/core/top_movies_list.html`：

```py
{% extends "base.html" %}

{% block title %}
  Top 10 Movies
{% endblock %}

{% block main %}
  <h1 >Top 10 Movies</h1 >
  <ol >
    {% for movie in object_list %}
      <li >
        <a href="{% url "core:MovieDetail" pk=movie.id %}" >
          {{ movie }}
        </a >
      </li >
    {% endfor %}
  </ol >
{% endblock %}
```

扩展`base.html`，我们将重新定义两个模板`block`标签。新的`title`模板`block`有我们的新标题。`main`模板`block`列出了`object_list`中的电影，包括每部电影的链接。

最后，让我们更新`django/templates/base.html`，以包括一个链接到我们的前 10 部电影页面：

```py
{# rest of template omitted #}
<div class="mymdb-masthead">
  <div class="container">
    <nav class="nav">
       {# skipping other nav items #}
       <a
          class="nav-link"
          href="{% url 'core:TopMovies' %}"
        >
        Top 10 Movies
       </a>
       {# skipping other nav items #}
      </nav>
   </div>
</div>
{# rest of template omitted #}
```

现在，让我们在我们的 URLConf 中添加一个`path()`对象，这样 Django 就可以将请求路由到我们的`TopMovies`视图。

# 添加到 TopMovies 的路径

像往常一样，我们需要添加一个`path()`来帮助 Django 将请求路由到我们的视图。让我们更新`django/core/urls.py`：

```py
from django.urls import path

from . import views

app_name = 'core'
urlpatterns = [
    path('movies',
         views.MovieList.as_view(),
         name='MovieList'),
    path('movies/top',
         views.TopMovies.as_view(),
         name="TopMovies"),
    # other paths omitted
 ]
```

有了这个，我们就完成了。现在我们在 MyMDB 上有了一个前 10 部电影页面。

然而，浏览所有的投票意味着扫描项目中最大的表。让我们看看如何优化我们的项目。

# 优化 Django 项目

如何优化 Django 项目没有单一正确答案，因为不同的项目有不同的约束。要成功，重要的是要清楚你要优化什么，以及在硬数据中使用什么，而不是直觉。

清楚地了解我们要进行优化的内容很重要，因为优化通常涉及权衡。您可能希望进行优化的一些约束条件如下：

+   响应时间

+   Web 服务器内存

+   Web 服务器 CPU

+   数据库内存

一旦您知道要进行优化的内容，您将需要一种方法来测量当前性能和优化代码的性能。优化代码通常比未优化代码更复杂。在承担复杂性之前，您应始终确认优化是否有效。

Django 只是 Python，因此您可以使用 Python 分析器来测量性能。这是一种有用但复杂的技术。讨论 Python 分析的细节超出了本书的范围。然而，重要的是要记住 Python 分析是我们可以使用的有用工具。

让我们看看一些特定于 Django 的测量性能的方法。

# 使用 Django 调试工具栏

Django 调试工具栏是一个第三方包，可以在浏览器中提供大量有用的调试信息。工具栏由一系列面板组成。每个面板提供不同的信息集。

一些最有用的面板（默认情况下启用）如下：

+   请求面板：它显示与请求相关的信息，包括处理请求的视图、接收到的参数（从路径中解析出来）、cookie、会话数据以及请求中的 GET/POST 数据。

+   SQL 面板：显示进行了多少查询，它们的执行时间线以及在查询上运行`EXPLAIN`的按钮。数据驱动的 Web 应用程序通常会因其数据库查询而变慢。

+   模板面板：显示已呈现的模板及其上下文。

+   日志面板：它显示视图产生的任何日志消息。我们将在下一节讨论更多关于日志记录的内容。

配置文件面板是一个高级面板，默认情况下不启用。该面板在您的视图上运行分析器并显示结果。该面板带有一些注意事项，这些注意事项在 Django 调试工具栏在线文档中有解释（[`django-debug-toolbar.readthedocs.io/en/stable/panels.html#profiling`](https://django-debug-toolbar.readthedocs.io/en/stable/panels.html#profiling)）。

Django 调试工具栏在开发中很有用，但不应在生产中运行。默认情况下，只有在`DEBUG = True`时才能工作（这是您在生产中绝对不能使用的设置）。

# 使用日志记录

Django 使用 Python 的内置日志系统，您可以使用`settings.LOGGING`进行配置。它使用`DictConfig`进行配置，如 Python 文档中所述。

作为一个复习，这是 Python 的日志系统的工作原理。该系统由*记录器*组成，它们从我们的代码接收*消息*和*日志级别*（例如`DEBUG`和`INFO`）。如果记录器被配置为不过滤掉该日志级别（或更高级别）的消息，它将创建一个*日志记录*，并将其传递给所有其*处理程序*。处理程序将检查它是否与处理程序的日志级别匹配，然后它将格式化日志记录（使用*格式化程序*）并发出消息。不同的处理程序将以不同的方式发出消息。`StreamHandler`将写入流（默认为`sys.stderr`），`SysLogHandler`写入`SysLog`，`SMTPHandler`发送电子邮件。

通过记录操作所需的时间，您可以对需要进行优化的内容有一个有意义的了解。使用正确的日志级别和处理程序，您可以在生产中测量资源消耗。

# 应用性能管理

应用性能管理（APM）是指作为应用服务器一部分运行并跟踪执行操作的服务。跟踪结果被发送到报告服务器，该服务器将所有跟踪结果合并，并可以为您提供对生产服务器性能的代码行级洞察。这对于大型和复杂的部署可能有所帮助，但对于较小、较简单的 Web 应用程序可能过于复杂。

# 本节的快速回顾

在本节中，我们回顾了在实际开始优化之前知道要优化什么的重要性。我们还看了一些工具，帮助我们衡量我们的优化是否成功。

接下来，我们将看看如何使用 Django 的缓存 API 解决一些常见的性能问题。

# 使用 Django 的缓存 API

Django 提供了一个开箱即用的缓存 API。在`settings.py`中，您可以配置一个或多个缓存。缓存可用于存储整个站点、单个页面的响应、模板片段或任何可 pickle 的对象。Django 提供了一个可以配置多种后端的单一 API。

在本节中，我们将执行以下功能：

+   查看 Django 缓存 API 的不同后端

+   使用 Django 缓存页面

+   使用 Django 缓存模板片段

+   使用 Django 缓存`QuerySet`

我们不会研究*下游*缓存，例如**内容交付网络**（**CDN**）或代理缓存。这些不是 Django 特有的，有各种各样的选择。一般来说，这些类型的缓存将依赖于 Django 已发送的相同`VARY`标头。

接下来，让我们看看如何配置缓存 API 的后端。

# 检查 Django 缓存后端之间的权衡

不同的后端可能适用于不同的情况。但是，缓存的黄金法则是它们必须比它们缓存的源*更快*，否则您会使应用程序变慢。决定哪个后端适合哪个任务最好是通过对项目进行仪器化来完成的，如前一节所讨论的。不同的后端有不同的权衡。

# 检查 Memcached 的权衡

**Memcached**是最受欢迎的缓存后端，但仍然存在需要评估的权衡。Memcached 是一个用于小数据的内存键值存储，可以由多个客户端（例如 Django 进程）使用一个或多个 Memcached 主机进行共享。但是，Memcached 不适合缓存大块数据（默认情况下为 1 MB 的数据）。另外，由于 Memcached 全部在内存中，如果进程重新启动，则整个缓存将被清除。另一方面，Memcached 因为快速和简单而保持受欢迎。

Django 带有两个 Memcached 后端，取决于您想要使用的`Memcached`库：

+   `django.core.cache.backends.memcached.MemcachedCache`

+   `django.core.cache.backends.memcached.PyLibMCCache`

您还必须安装适当的库（`python-memcached`或`pylibmc`）。要将您的 Memcached 服务器的地址设置为`LOCATION`，请将其设置为格式为`address:PORT`的列表（例如，`['memcached.example.com:11211',]`）。示例配置在本节末尾列出。

在*开发*和*测试*中使用 Memcached 可能不会很有用，除非您有相反的证据（例如，您需要复制一个复杂的错误）。

Memcached 在生产环境中很受欢迎，因为它快速且易于设置。它通过让所有 Django 进程连接到相同的主机来避免数据重复。但是，它使用大量内存（并且在可用内存用尽时会迅速且不良地降级）。另外，注意运行另一个服务的操作成本是很重要的。

以下是使用`memcached`的示例配置：

```py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.memcached.PyLibMCCache',
        'LOCATION':  [
            '127.0.0.1:11211',
        ],
    }
}
```

# 检查虚拟缓存的权衡

**虚拟缓存**（`django.core.cache.backends.dummy.DummyCache`）将检查密钥是否有效，但否则不执行任何操作。

当您想确保您确实看到代码更改的结果而不是缓存时，此缓存在*开发*和*测试*中可能很有用。

不要在*生产*中使用此缓存，因为它没有效果。

以下是一个虚拟缓存的示例配置：

```py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
    }
}
```

# 检查本地内存缓存的权衡

**本地内存缓存**（`django.core.cache.backends.locmem.LocMemCache`）使用 Python 字典作为全局内存缓存。如果要使用多个单独的本地内存缓存，请在`LOCATION`中给出每个唯一的字符串。它被称为本地缓存，因为它是每个进程的本地缓存。如果您正在启动多个进程（就像在生产中一样），那么不同进程处理请求时可能会多次缓存相同的值。这种低效可能更简单，因为它不需要另一个服务。

这是一个在*开发*和*测试*中使用的有用缓存，以确认您的代码是否正确缓存。

您可能想在*生产*中使用这个，但要记住不同进程缓存相同数据的潜在低效性。

以下是本地内存缓存的示例配置：

```py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'defaultcache',

    },
    'otherCache': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'othercache',
    }
}
```

# 检查基于文件的缓存权衡

Django 的**基于文件的缓存**（`django.core.cache.backends.filebased.FileBasedCache`）使用指定的`LOCATION`目录中的压缩文件来缓存数据。使用文件可能看起来很奇怪；缓存不应该是*快速*的，而文件是*慢*的吗？答案再次取决于您要缓存的内容。例如，对外部 API 的网络请求可能比本地磁盘慢。请记住，每个服务器都将有一个单独的磁盘，因此如果您运行一个集群，数据将会有一些重复。

除非内存受限，否则您可能不想在*开发*或*测试*中使用这个。

您可能想在生产中缓存特别大或请求速度慢的资源。请记住，您应该给服务器进程写入`LOCATION`目录的权限。此外，请确保为缓存给服务器提供足够的磁盘空间。

以下是使用基于文件的缓存的示例配置：

```py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache',
        'LOCATION': os.path.join(BASE_DIR, '../file_cache'),
    }
}
```

# 检查数据库缓存权衡

**数据库缓存**后端（`django.core.cache.backends.db.DatabaseCache`）使用数据库表（在`LOCATION`中命名）来存储缓存。显然，如果您的数据库速度很快，这将效果最佳。根据情况，即使在缓存数据库查询结果时，这也可能有所帮助，如果查询复杂但单行查找很快。这有其优势，因为缓存不像内存缓存那样是短暂的，可以很容易地在进程和服务器之间共享（如 Memcached）。

数据库缓存表不是由迁移管理的，而是由`manage.py`命令管理，如下所示：

```py
$ cd django
$ python manage.py createcachetable
```

除非您想在*开发*或*测试*中复制您的生产环境，否则您可能不想使用这个。

如果您的测试证明它是合适的，您可能想在*生产*中使用这个。请记住考虑增加的数据库负载对性能的影响。

以下是使用数据库缓存的示例配置：

```py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.db.DatabaseCache',
        'LOCATION': 'django_cache_table',
    }
}
```

# 配置本地内存缓存

在我们的情况下，我们将使用一个具有非常低超时的本地内存缓存。这意味着我们在编写代码时大多数请求将跳过缓存（旧值（如果有）将已过期），但如果我们快速点击刷新，我们将能够确认我们的缓存正在工作。

让我们更新`django/config/settings.py`以使用本地内存缓存：

```py
 CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'default-locmemcache',
        'TIMEOUT': 5, # 5 seconds
    }
 }
```

尽管我们可以有多个配置不同的缓存，但默认缓存的名称应为`'default'`。

`Timeout`是值在被清除（移除/忽略）之前在缓存中保留的时间（以秒为单位）。如果`Timeout`为`None`，则该值将被视为永不过期。

现在我们已经配置了缓存，让我们缓存`MovieList`页面。

# 缓存电影列表页面

我们将假设`MovieList`页面对我们来说非常受欢迎且昂贵。为了降低提供这些请求的成本，我们将使用 Django 来缓存整个页面。

Django 提供了装饰器（函数）`django.views.decorators.cache.cache_page`，它可以用来缓存单个页面。这是一个装饰器而不是一个 mixin，可能看起来有点奇怪。当 Django 最初发布时，它没有 **基于类的视图**（**CBVs**），只有 **基于函数的视图**（**FBVs**）。随着 Django 的成熟，很多代码切换到使用 CBVs，但仍然有一些功能实现为 FBV 装饰器。

在 CBVs 中，有几种不同的使用函数装饰器的方式。我们的方法是构建我们自己的 mixin。CBVs 的很多功能来自于能够将新行为混入到现有类中的能力。了解如何做到这一点是一项有用的技能。

# 创建我们的第一个 mixin – CachePageVaryOnCookieMixin

让我们在 `django/core/mixins.py` 中创建一个新的类：

```py
from django.core.cache import caches
from django.views.decorators.cache import (
    cache_page)

class CachePageVaryOnCookieMixin:
    """
    Mixin caching a single page.

    Subclasses can provide these attributes:

    `cache_name` - name of cache to use.
    `timeout` - cache timeout for this
    page. When not provided, the default
    cache timeout is used. 
    """
    cache_name = 'default'

    @classmethod
    def get_timeout(cls):
        if hasattr(cls, 'timeout'):
            return cls.timeout
        cache = caches[cls.cache_name]
        return cache.default_timeout

    @classmethod
    def as_view(cls, *args, **kwargs):
        view = super().as_view(
            *args, **kwargs)
        view = vary_on_cookie(view)
        view = cache_page(
            timeout=cls.get_timeout(),
            cache=cls.cache_name,
        )(view)
        return view
```

我们的新 mixin 覆盖了我们在 URLConfs 中使用的 `as_view()` 类方法，并使用 `vary_on_cookie()` 和 `cache_page()` 装饰器装饰视图。这实际上就像我们在 `as_view()` 方法上使用我们的函数装饰器一样。

让我们先看看 `cache_page()` 装饰器。`cache_page()` 需要一个 `timeout` 参数，并且可以选择接受一个 `cache` 参数。`timeout` 是缓存页面应该过期并且必须重新缓存之前的时间（以秒为单位）。我们的默认超时值是我们正在使用的缓存的默认值。子类化 `CachePageVaryOnCookieMixin` 的类可以提供一个新的 `timeout` 属性，就像我们的 `MovieList` 类提供了一个 `model` 属性一样。`cache` 参数期望所需缓存的字符串名称。我们的 mixin 被设置为使用 `default` 缓存，但通过引用一个类属性，这也可以被子类更改。

当缓存一个页面，比如 `MoveList`，我们必须记住，对于不同的用户，生成的页面是不同的。在我们的情况下，`MovieList` 的头对已登录用户（显示 *注销* 链接）和已注销用户（显示 *登录* 和 *注册* 链接）是不同的。Django 再次为我们提供了 `vary_on_cookie()` 装饰器。

`vary_on_cookie()` 装饰器将一个 `VARY cookie` 头添加到响应中。`VARY` 头被缓存（包括下游缓存和 Django 的缓存）用来告诉它们有关该资源的变体。`VARY cookie` 告诉缓存，每个不同的 cookie/URL 对都是不同的资源，应该分别缓存。这意味着已登录用户和已注销用户将看到不同的页面，因为它们将有不同的 cookie。

这对我们的命中率（缓存被 *命中* 而不是重新生成资源的比例）有重要影响。命中率低的缓存将几乎没有效果，因为大多数请求将 *未命中* 缓存，并导致处理请求。

在我们的情况下，我们还使用 cookie 进行 CSRF 保护。虽然会话 cookie 可能会降低命中率一点，具体取决于情况（查看用户的活动以确认），但 CSRF cookie 几乎是致命的。CSRF cookie 的性质是经常变化，以便攻击者无法预测。如果那个不断变化的值与许多请求一起发送，那么很少能被缓存。幸运的是，我们可以将我们的 CSRF 值从 cookie 移出，并将其存储在服务器端会话中，只需通过 `settings.py` 进行更改。

为您的应用程序决定正确的 CSRF 策略可能是复杂的。例如，AJAX 应用程序将希望通过标头添加 CSRF 令牌。对于大多数站点，默认的 Django 配置（使用 cookie）是可以的。如果您需要更改它，值得查看 Django 的 CSRF 保护文档（[`docs.djangoproject.com/en/2.0/ref/csrf/`](https://docs.djangoproject.com/en/2.0/ref/csrf/)）。

在 `django/conf/settings.py` 中，添加以下代码：

```py
CSRF_USE_SESSIONS = True
```

现在，Django 不会将 CSRF 令牌发送到 cookie 中，而是将其存储在用户的会话中（存储在服务器上）。

如果用户已经有 CSRF cookie，它们将被忽略；但是，它仍然会对命中率产生抑制作用。在生产环境中，您可能希望考虑添加一些代码来删除这些 CSRF cookie。

现在我们有了一种轻松混合缓存行为的方法，让我们在`MovieList`视图中使用它。

# 使用 CachePageVaryOnCookieMixin 与 MovieList

让我们在`django/core/views.py`中更新我们的视图：

```py
from django.views.generic import ListView
from core.mixins import (
    VaryCacheOnCookieMixin)

class MovieList(VaryCacheOnCookieMixin, ListView):
    model = Movie
    paginate_by = 10

    def get_context_data(self, **kwargs):
        # omitted due to no change
```

现在，当`MovieList`收到路由请求时，`cache_page`将检查它是否已被缓存。如果已经被缓存，Django 将返回缓存的响应，而不做任何其他工作。如果没有被缓存，我们常规的`MovieList`视图将创建一个新的响应。新的响应将添加一个`VARY cookie`头，然后被缓存。

接下来，让我们尝试在模板中缓存我们的前 10 部电影列表的一部分。

# 使用`{% cache %}`缓存模板片段

有时，页面加载缓慢是因为我们模板的某个部分很慢。在本节中，我们将看看如何通过缓存模板的片段来解决这个问题。例如，如果您使用的标签需要很长时间才能解析（比如，因为它发出了网络请求），那么它将减慢使用该标签的任何页面。如果无法优化标签本身，将模板中的结果缓存可能就足够了。

通过编辑`django/core/templates/core/top_movies.html`来缓存我们渲染的前 10 部电影列表：

```py
{% extends "base.html" %}
{% load cache %}

{% block title %}
  Top 10 Movies
{% endblock %}

{% block main %}
  <h1 >Top 10 Movies</h1 >
  {% cache 300 top10 %}
  <ol >
    {% for movie in object_list %}
      <li >
        <a href="{% url "core:MovieDetail" pk=movie.id %}" >
          {{ movie }}
        </a >
      </li >
    {% endfor %}
  </ol >
  {% endcache %}
{% endblock %}
```

这个块向我们介绍了`{% load %}`标签和`{% cache %}`标签。

`{% load %}`标签用于加载标签和过滤器的库，并使它们可用于模板中使用。一个库可以提供一个或多个标签和/或过滤器。例如，`{% load humanize %}`加载标签和过滤器，使值看起来更人性化。在我们的情况下，`{% load cache %}`只提供了`{% cache %}`标签。

`{% cache 300 top10 %}`将在提供的秒数下缓存标签的主体，并使用提供的键。第二个参数必须是一个硬编码的字符串（而不是一个变量），但如果片段需要有变体，我们可以提供更多的参数（例如，`{% cache 300 mykey request.user.id %}`为每个用户缓存一个单独的片段）。该标签将使用`default`缓存，除非最后一个参数是`using='cachename'`，在这种情况下，将使用命名缓存。

使用`{% cache %}`进行缓存发生在不同的级别，而不是使用`cache_page`和`vary_on_cookie`。视图中的所有代码仍将被执行。视图中的任何缓慢代码仍将减慢我们的速度。缓存模板片段只解决了我们模板代码中一个非常特定的缓慢片段的问题。

由于`QuerySets`是懒惰的，通过将我们的`for`循环放在`{% cache %}`中，我们避免了评估`QuerySet`。如果我们想缓存一个值以避免查询它，如果我们在视图中这样做，我们的代码会更清晰。

接下来，让我们看看如何使用 Django 的缓存 API 缓存对象。

# 使用对象的缓存 API

Django 的缓存 API 最精细的用法是存储与 Python 的`pickle`序列化模块兼容的对象。我们将在这里看到的`cache.get()`/`cache.set()`方法在`cache_page()`装饰器和`{% cache %}`标签内部使用。在本节中，我们将使用这些方法来缓存`Movie.objects.top_movies()`返回的`QuerySet`。

方便的是，`QuerySet`对象是可 pickle 的。当`QuerySets`被 pickled 时，它将立即被评估，并且生成的模型将存储在`QuerySet`的内置缓存中。在 unpickling 一个`QuerySet`时，我们可以迭代它而不会引起新的查询。如果`QuerySet`有`select_related`或`prefetch_related`，那些查询将在 pickling 时执行，而在 unpickling 时不会重新运行。

让我们从`top_movies_list.html`中删除`{% cache %}`标签，而是更新`django/core/views.py`：

```py
import django
from django.core.cache import cache
from django.views.generic import ListView

from core.models import Movie

class TopMovies(ListView):
    template_name = 'core/top_movies_list.html'

    def get_queryset(self):
        limit = 10
        key = 'top_movies_%s' % limit
        cached_qs = cache.get(key)
        if cached_qs:
            same_django = cached_qs._django_version == django.get_version()
            if same_django:
                return cached_qs
        qs = Movie.objects.top_movies(
            limit=limit)
        cache.set(key, qs)
        return qs
```

我们的新`TopMovies`视图重写了`get_queryset`方法，并在使用`MovieManger.top_movies()`之前检查缓存。对`QuerySet`对象进行 pickling 确实有一个警告——不能保证在不同的 Django 版本中兼容，因此在继续之前应该检查所使用的版本。

`TopMovies`还展示了一种访问默认缓存的不同方式，而不是`VaryOnCookieCache`使用的方式。在这里，我们导入并使用`django.core.cache.cache`，它是`django.core.cache.caches['default']`的代理。

在使用低级 API 进行缓存时，记住一致的键的重要性是很重要的。在大型代码库中，很容易在不同的键下存储相同的数据，导致效率低下。将缓存代码放入管理器或实用程序模块中可能很方便。

# 总结

在本章中，我们创建了一个 Top 10 电影视图，审查了用于检测 Django 代码的工具，并介绍了如何使用 Django 的缓存 API。Django 和 Django 社区提供了帮助您发现在哪里优化代码的工具，包括使用分析器、Django 调试工具栏和日志记录。Django 的缓存 API 通过`cache_page`缓存整个页面，通过模板标签`{% cache %}`缓存模板片段，以及通过`cache.set`/`cache.get`缓存任何可 picklable 对象，为我们提供了丰富的 API。

接下来，我们将使用 Docker 部署 MyMDB。
