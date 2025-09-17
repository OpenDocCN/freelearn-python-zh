# 有用的 Flask 扩展

正如我们在整本书中看到的那样，Flask 被设计得尽可能小，同时仍然提供创建 Web 应用程序所需的灵活性和工具。然而，有很多功能是许多 Web 应用程序共有的，这意味着许多应用程序将需要执行相同任务的代码。为了解决这个问题，并避免重复造轮子，人们为 Flask 创建了扩展，我们在整本书中已经看到了许多 Flask 扩展。本章将重点介绍一些更有用的 Flask 扩展，它们的内容不足以单独成章，但将为您节省大量时间和挫折。

在本章中，你将学习以下内容：

+   开发一个具有出色后端性能指标的调试工具栏

+   使用 Redis 或 memcached 进行页面缓存

+   创建一个具有所有模型 CRUD 功能的管理后台

+   启用国际化（i18n），并将您的网站翻译成多种语言

+   容易发送电子邮件

# Flask CLI

在 第一章 “入门”中，我们介绍了一些基本功能，并学习了如何使用 Flask CLI。现在，我们将看到如何充分利用这一功能。

在 Flask CLI 中，你可以创建自定义命令，在应用程序上下文中运行。Flask CLI 本身使用 **Click**，这是一个由 Flask 的创建者开发的库，用于创建具有复杂参数的命令行工具。

要了解有关 Click 的更多详细信息，请查看文档，可在 [`click.pocoo.org`](http://click.pocoo.org) 找到。

我们的目标是创建一组命令，帮助我们管理和部署我们的 Flask 应用程序。首先要解决的问题是我们将在哪里以及如何创建这些命令行函数。由于我们的 CLI 是一个应用程序的全局实用工具，我们将将其放置在 `webapp/cli.py`：

```py
import logging
import click
from .auth.models import User, db

log = logging.getLogger(__name__)

def register(app):
 @app.cli.command('create-user')
 @click.argument('username')
 @click.argument('password')
    def create_user(username, password):
        user= User()
        user.username = username
        user.set_password(password)
        try:
            db.session.add(user)
            db.session.commit()
            click.echo('User {0} Added.'.format(username))
        except Exception as e:
            log.error("Fail to add new user: %s Error: %s" 
            % (username, e))
            db.session.rollback()
...
```

我们将在 `register` 函数内部开发所有我们的函数，这样我们就不需要从主模块导入我们的 Flask 应用程序。这样做会导致循环依赖导入。接下来，请注意以下我们使用的装饰器：

+   `@app.cli.command` 注册了我们的函数有一个新的命令行命令；如果没有传递参数，那么 `Click` 将假设函数的名称。

+   `@click.argument` 添加了一个命令行参数；在我们的情况下，用于用户名和密码（用于创建用户凭据）。参数是位置命令行选项。

我们在 `main.py` 中注册了所有的命令行函数。注意以下片段中突出显示的文本，其中我们调用了之前创建的 `register` 方法：

```py
import os
from webapp import create_app
from webapp.cli import register

env = os.environ.get('WEBAPP_ENV', 'dev')
app = create_app('config.%sConfig' % env.capitalize())
register(app)

if __name__ == '__main__':
    app.run()
```

从命令行界面（CLI）开始，让我们尝试我们刚刚创建的命令，如下所示：

```py
# First we need to export our FLASK_APP env var
$ export FLASK_APP=main.py
$ flask create-user user10 password
User user10 Added.
$ flask run
 * Serving Flask app "main"
2018-08-12 20:25:43,031:INFO:werkzeug: * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

接下来，你可以打开你的网络浏览器，并使用新创建的 `user10` 凭据登录我们的博客。

提供的代码还包括一个 `list-users` 命令，但它的实现现在对你来说应该很直观，这里不再进行额外解释。让我们关注一个简单且实用的函数，用于显示我们应用程序的所有路由：

```py
@app.cli.command('list-routes')
def list_routes():
    for url in app.url_map.iter_rules():
        click.echo("%s %s %s" % (url.rule, url.methods, url.endpoint))
```

`list-routes` 命令列出了在 `app` 对象上注册的所有路由及其关联的 URL。这在调试 Flask 扩展时非常有用，因为它使得检查其蓝图注册是否工作变得非常简单。

# Flask Debug Toolbar

**Flask Debug Toolbar** 是一个 Flask 扩展，通过将调试工具添加到你的应用程序的网页视图中来帮助开发。它为你提供了关于视图渲染代码的瓶颈以及渲染视图所需的 SQLAlchemy 查询次数等信息。

和往常一样，我们将使用 `pip` 安装 Flask Debug Toolbar 并将其添加到我们的 `requirements.txt` 文件中：

```py
$ source venv/bin/activate
(venv) $ pip install -r requirements
```

接下来，我们需要将 Flask Debug Toolbar 添加到 `webapp/__init__.py` 文件中。由于在本章中我们将大量修改此文件，以下是到目前为止文件的开始部分，以及初始化 Flask Debug Toolbar 的代码：

```py
...
from flask_debugtoolbar import DebugToolbarExtension 

...
debug_toolbar = DebugToolbarExtension()
...
def create_app(config):
...
    debug_toolbar.init_app(app)
...
```

这就是将 Flask Debug Toolbar 启用并运行所需的所有内容。如果你的应用配置中的 `DEBUG` 变量设置为 `true`，则工具栏将显示。如果 `DEBUG` 没有设置为 `true`，则工具栏不会注入到页面中：

![图片](img/21d3d10e-70e0-49a7-81ad-5d2b1fcd2ad3.png)

在屏幕的右侧，你会看到工具栏。每个部分都是一个链接，它将在页面上显示一个值表。要获取渲染视图时调用的所有函数的列表，请点击 Profiler 旁边的复选框以启用它，然后重新加载页面并点击 Profiler。这个视图可以让你轻松快速地诊断你的应用程序中哪些部分运行最慢，或者被调用得最多。

默认情况下，Flask Debug Toolbar 会拦截 `HTTP 302 重定向` 请求。要禁用此功能，请将以下内容添加到你的配置中：

```py
class DevConfig(Config): 
    DEBUG = True 
    DEBUG_TB_INTERCEPT_REDIRECTS = False 
```

此外，如果你正在使用 Flask-MongoEngine，你可以通过覆盖要渲染的面板并添加以下 MongoEngine 的自定义面板来查看渲染页面时所做的所有查询：

```py
class DevConfig(Config): 
    DEBUG = True 
    DEBUG_TB_PANELS = [
        'flask_debugtoolbar.panels.versions.VersionDebugPanel', 
        'flask_debugtoolbar.panels.timer.TimerDebugPanel', 
        'flask_debugtoolbar.panels.headers.HeaderDebugPanel', 
        'flask_debugtoolbar.panels.
         request_vars.RequestVarsDebugPanel',        
         'flask_debugtoolbar.panels.config_vars.
         ConfigVarsDebugPanel ',         
         'flask_debugtoolbar.panels.template.
         TemplateDebugPanel',        'flask_debugtoolbar.panels.
         logger.LoggingPanel',        'flask_debugtoolbar.panels.
         route_list.RouteListDebugPanel'        
        'flask_debugtoolbar.panels.profiler.
         ProfilerDebugPanel',        'flask_mongoengine.panels.
         MongoDebugPanel' 
    ] 
    DEBUG_TB_INTERCEPT_REDIRECTS = False 
```

这将在工具栏中添加一个与默认 SQLAlchemy 面板非常相似的面板。

# Flask 缓存

在 第七章 “使用 Flask 与 NoSQL”，我们了解到页面加载时间是决定你的 Web 应用成功或失败的最重要因素之一。尽管我们的页面不经常更改，而且新帖子也不会经常发布，但我们仍然每次用户浏览器请求页面时都会渲染模板并查询数据库。

Flask 缓存通过允许我们存储视图函数的结果并返回存储的结果，而不是再次渲染模板来解决此问题。首先，我们需要在我们的虚拟环境中安装 Flask 缓存。这已经在运行 `init.sh` bash 脚本时完成。`init.sh` 脚本将首先安装 `requirements.txt` 中声明的所有依赖项：

```py
...
Flask-Caching
...
```

接下来，在 `webapp/__init__.py` 中初始化它，如下所示：

```py
from flask_caching import Cache 
...
cache = Cache()
... 
def create_app(config):
...
    cache.init_app(app)
...
```

在我们开始缓存视图之前，我们需要告诉 Flask 缓存我们希望如何存储新函数的结果：

```py
class DevConfig(Config): 

    CACHE_TYPE = 'simple'
```

`simple` 选项告诉 Flask 缓存将结果存储在内存中的 Python 字典中，这对于 Flask 应用程序的大多数情况是足够的。我们将在本节后面介绍更多类型的缓存后端。

# 缓存视图和函数

为了缓存视图函数的结果，只需将装饰器添加到任何函数中：

```py
...
from .. import cache
...

@blog_blueprint.route('/')
@blog_blueprint.route('/<int:page>')
@cache.cached(timeout=60)
def home(page=1):
    posts = 
    Post.query.order_by(Post.publish_date.desc()).paginate(page, 
    current_app.config['POSTS_PER_PAGE'], False)
    recent, top_tags = sidebar_data()

    return render_template(
        'home.html',
        posts=posts,
        recent=recent,
        top_tags=top_tags
    )
```

`timeout` 参数指定缓存结果应该持续多少秒，然后函数应该再次运行并存储。为了确认视图实际上正在被缓存，请检查调试工具栏中的 SQLAlchemy 部分。我们还可以通过激活分析器并比较前后时间来查看缓存对页面加载时间的影响。在作者的顶级笔记本电脑上，主要博客页面渲染需要 34 毫秒，主要是由于对数据库进行的八次不同查询。但是，在激活缓存后，这减少到 0.08 毫秒。这是速度提高了 462.5 倍！

视图函数不是唯一可以缓存的东西。为了缓存任何 Python 函数，只需将类似的装饰器添加到函数定义中，如下所示：

```py
@cache.cached(timeout=7200, key_prefix='sidebar_data') 
def sidebar_data(): 
    recent = Post.query.order_by( 
        Post.publish_date.desc() 
    ).limit(5).all() 

    top_tags = db.session.query( 
        Tag, func.count(tags.c.post_id).label('total') 
    ).join( 
        tags 
    ).group_by( 
        Tag 
    ).order_by('total DESC').limit(5).all() 

    return recent, top_tags 
```

`key_prefix` 关键字参数对于 Flask 缓存正确存储非视图函数的结果是必要的。对于每个缓存的函数，这需要是唯一的，否则函数的结果将相互覆盖。此外，请注意，此函数的超时设置为两小时，而不是之前的 60 秒。这是因为此函数的结果不太可能像视图函数那样发生变化，如果数据已过时，这并不是一个很大的问题。

# 缓存带参数的函数

然而，正常的缓存装饰器不考虑函数参数。如果我们使用正常的缓存装饰器缓存了一个带有参数的函数，它将为每个参数集返回相同的结果。为了解决这个问题，我们使用 `memoize` 函数：

```py
...
from .. import db, cache
...

class User(db.Model):
... 
    @cache.memoize(60)
    def has_role(self, name):
        for role in self.roles:
            if role.name == name:
                return True
        return False
```

`Memoize`存储传递给函数的参数以及结果。在上面的例子中，`memoize`被用来存储`verify_auth_token`方法的返回结果，该方法被多次调用，并且每次都会查询数据库。这个方法可以安全地进行 memoization，因为它每次在传入相同的令牌时都会返回相同的结果。唯一的例外是，如果在函数存储的 60 秒内用户对象被删除，但这非常不可能。

请务必小心，不要对依赖于全局作用域变量或不断变化数据的函数进行`memoize`或缓存。这可能会导致一些非常微妙的错误，在最坏的情况下，甚至会导致数据竞争。最适合进行 memoization 的函数被称为纯函数。**纯函数**是指当传入相同的参数时，将产生相同结果的函数。无论函数运行多少次，结果都不会改变。纯函数也没有任何*副作用*，这意味着它们不会改变全局作用域变量。这也意味着纯函数不能执行任何 I/O 操作。虽然`verify_auth_token`函数不是纯函数，因为它执行数据库 I/O，但这是可以接受的，因为，如前所述，底层数据发生变化的可能性非常小。

在我们开发应用程序时，我们不希望缓存视图函数，因为结果会不断变化。为了解决这个问题，将`CACHE_TYPE`变量设置为`null`，在生产配置中，将`CACHE_TYPE`变量设置为简单，这样当应用程序部署时，一切都会按预期工作：

```py
class ProdConfig(Config): 

    CACHE_TYPE = 'simple'

class DevConfig(Config): 

    CACHE_TYPE = 'null' 
```

# 使用查询字符串缓存路由

一些路由，如我们的`home`和`post`路由，通过 URL 传递参数并返回特定于这些参数的内容。如果这些路由被缓存，我们会遇到问题，因为第一个渲染的路由将返回所有请求，无论 URL 参数如何。这个问题的解决方案相当简单。缓存方法中的`key_prefix`关键字参数可以是字符串或函数，它将被执行以动态生成键。

这意味着可以创建一个函数，该函数会根据 URL 参数创建一个相关的键，这样每个请求只有在之前调用过该特定参数组合时才会返回缓存的页面。在`blog/controllers.py`文件中，找到以下函数：

```py
def make_cache_key(*args, **kwargs):
    path = request.path
    args = str(hash(frozenset(request.args.items())))
    messages = str(hash(frozenset(get_flashed_messages())))
    return (path + args + messages).encode('utf-8')
```

我们使用此函数通过混合 URL 路径、参数和 Flask 消息来创建缓存键。这将防止用户登出时消息不显示。我们将在主页视图和按 ID 显示帖子时使用这种类型的缓存键生成。

现在，每个单独的帖子页面将被缓存 10 分钟。

# 使用 Redis 作为缓存后端

如果传递给缓存函数的视图函数数量或唯一参数的数量太大，以至于内存无法处理，你可以为缓存使用不同的后端。正如在 第七章 中提到的 *使用 NoSQL 与 Flask*，Redis 可以作为缓存的后端。要实现这个功能，需要做的只是将以下配置变量添加到 `ProdConfig` 类中，如下所示：

```py
class ProdConfig(Config): 
    ... 
    CACHE_TYPE = 'redis' 
    CACHE_REDIS_HOST = 'localhost' 
    CACHE_REDIS_PORT = '6379' 
    CACHE_REDIS_PASSWORD = 'password' 
    CACHE_REDIS_DB = '0' 
```

如果你用你自己的数据替换了变量的值，Flask Cache 将会自动创建一个连接到你的 `redis` 数据库，并使用它来存储函数的结果。所需做的只是安装 Python 的 `redis` 库。在执行了 `init.sh` 脚本之后，这个库就已经安装好了，我们执行这个脚本是为了设置本章的工作环境。你可以在 `requirements.txt` 中找到这个库：**

```py
...
redis
...
```

如果你想测试你的 Redis 缓存，我们准备了一个包含 RabbitMQ 和 Redis 的 Docker composer 文件。要启动它，只需在 CLI 上执行以下命令：

```py
# Start dockers for RMQ and Redis in the background
$ docker-compose up -d Creating rabbitmq ... doneCreating redis ... done # Check the currently active containers
$ docker container list
CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES
3266cbdee1d7 redis "docker-entrypoint.s…" 43 seconds ago Up 58 seconds 0.0.0.0:6379->6379/tcp redis
64a99718442c rabbitmq:3-management "docker-entrypoint.s…" 43 seconds ago Up 58 seconds 4369/tcp, 5671/tcp, 0.0.0.0:5672->5672/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:15672->15672/tcp rabbitmq
```

记得使用以下生产配置来测试你的应用程序：

```py
$ export WEBAPP_ENV=prod
$ export FLASK_APP=main.py
$ flask run
```

# 使用 memcached 作为缓存后端

就像 Redis 后端一样，memcached 后端提供了一种存储结果的替代方式，如果存储限制变得过于限制性。与 Redis 相比，memcached 是设计用来缓存对象以供后续使用并减少数据库负载的。Redis 和 memcached 都服务于相同的目的，选择哪一个取决于个人偏好。要使用 memcached，我们需要使用以下命令安装它的 Python 库：

```py
$ pip install memcache
```

连接到你的 memcached 服务器的过程在配置对象中处理，就像 Redis 设置一样：

```py
class ProdConfig(Config): 
    ... 
    CACHE_TYPE = 'memcached' 
    CACHE_KEY_PREFIX = 'flask_cache' 
    CACHE_MEMCACHED_SERVERS = ['localhost:11211'] 
```

# Flask Assets

网络应用程序的另一个瓶颈是需要下载页面上的 CSS 和 JavaScript 库的 HTTP 请求的数量。额外的文件只能在页面 HTML 加载和解析之后下载。为了解决这个问题，许多现代浏览器会一次性下载许多这些库，但浏览器可以发起的并发请求数量是有限的。

服务器上可以执行几个操作来减少下载这些文件所需的时间。开发者用来解决这个问题的主要技术是将所有的 JavaScript 库合并成一个文件，所有的 CSS 库合并成另一个文件，同时从生成的文件中移除所有的空白和回车符（也称为 **minification**）。这减少了多个 HTTP 请求的开销，并且可以将文件大小减少高达 30%。另一种技术是告诉浏览器使用特殊的 HTTP 头部信息本地缓存文件，这样文件只有在发生变化时才会再次加载。这些操作手动执行可能会很繁琐，因为它们需要在每次部署到服务器后执行。

幸运的是，Flask Assets 实现了所有讨论的技术。Flask Assets 通过提供一个文件列表和连接它们的方式工作，然后在模板中添加一个特殊的控制块，代替正常的链接和脚本标签。然后，Flask Assets 将添加一个指向新生成文件的`link`或`script`标签。要开始使用，需要安装 Flask Assets。我们还需要安装`cssmin`和`jsmin`——你可以在`requirements.txt`中找到这些依赖项。

现在，需要创建要连接的文件集合，即所谓的包。在`ewebapp/__init__.py`中，我们有以下内容：

```py
...
from flask_assets import Environment, Bundle 
...
assets_env = Environment() 

main_css = Bundle( 
    'css/bootstrap.css', 
    filters='cssmin', 
    output='css/common.css' 
) 

main_js = Bundle( 
    'js/jquery.js', 
    'js/bootstrap.js', 
    filters='jsmin', 
    output='js/common.js' 
) 
```

每个`Bundle`对象接受无限数量的文件作为位置参数来定义要打包的文件，一个`filters`关键字参数来定义要发送文件通过的过滤器，以及一个`output`参数，它定义了结果将保存到`static`文件夹中的文件名。

`filters`关键字可以是一个值或一个列表。要获取所有可用的过滤器列表，包括自动的 Less 和 CSS 编译器，请参阅[`webassets.readthedocs.org/en/latest/`](http://webassets.readthedocs.org/en/latest/)上的文档。

虽然确实，由于我们的站点样式较少，CSS 包中只有一个文件，但将文件放入包中仍然是一个好主意，有两个原因。首先，在我们开发期间，我们可以使用未压缩版本的库，这使得调试更容易。当应用部署到生产环境时，库会自动压缩。其次，这些库将带有缓存头信息发送到浏览器，而在 HTML 中正常链接它们时则不会。

在 Flask Assets 可以测试之前，需要做三个更改。首先，在`_init_.py`格式中，需要注册扩展和包：

```py
from .extensions import ( 
    bcrypt, 
    oid, 
    login_manager, 
    principals, 
    rest_api, 
    celery, 
    debug_toolbar, 
    cache, 
    assets_env, 
    main_js, 
    main_css 
) 

def create_app(object_name): 
    ... 
    assets_env.init_app(app) 

    assets_env.register("main_js", main_js) 
    assets_env.register("main_css", main_css) 
```

接下来，`DevConfig`类需要一个额外的变量来告诉 Flask Assets 在开发期间不要编译库：

```py
class DevConfig(Config): 
    DEBUG = True 
    DEBUG_TB_INTERCEPT_REDIRECTS = False 
    ASSETS_DEBUG = True
```

最后，两个`base.html`文件中的链接和脚本标签都需要替换为 Flask Assets 的控制块。文件中已经有以下内容：

```py
<link rel="stylesheet" 
 href=https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootst
 rap.min.css>
```

用以下代码替换前面的片段：

```py
{% assets "main_css" %} 
<link rel="stylesheet" type="text/css" href="{{ ASSET_URL }}" 
 /> 
{% endassets %} 
```

同样，在`base.html`文件中找到以下内容：

```py
<script 
 src="img/> .min.js"></script><script 
 src="img/> ap.min.js"></script>
```

再次，用以下代码替换前面的代码：

```py
{% assets "main_js" %} 
<script src="img/{{ ASSET_URL }}"></script> 
{% endassets %} 
```

现在，如果你重新加载页面，所有的 CSS 和 JavaScript 都将由 Flask Assets 处理。

# Flask Admin

在第六章，“保护您的应用”，我们创建了一个接口，允许用户创建和编辑博客文章，而无需使用 CLI。这对于展示章节中提到的安全措施是足够的，但仍然没有通过接口删除文章或为文章分配标签的方法。我们也没有为普通用户隐藏的删除或编辑评论的方法。我们的应用需要的是一个功能齐全的管理员界面，类似于 WordPress 界面。这对于应用来说是一个如此常见的需求，以至于出现了一个名为 Flask Admin 的 Flask 扩展，旨在帮助开发者轻松创建管理员界面。再次，我们可以在`requirements.txt`的依赖列表中找到 Flask Admin。

由于我们将创建一个包含表单、视图和模板的完整管理员界面，Flask Admin 是我们应用中新的模块的好候选者。首先，看看我们新的应用结构：

```py
./
  webapp/
    admin/
 __init__.py
 forms.py
 controllers.py
    api/
    auth/
    blog/
    templates/
      admin/
 ...      auth/
      blog/
      ...
 ...
```

如同往常，我们需要在我们的`webapp/admin/__init__.py`文件中创建`create_module`函数：

```py
...
from flask_admin import Admin 
...
admin = Admin()

def create_module(app, **kwargs):
    admin.init_app(app)
    ....

```

然后，在主`webapp/__init__.py`文件中调用`create_module`函数：

```py

def create_app(object_name): 
    ...
    from .admin import create_module as admin_create_module
    ...
    admin_create_module(app)
```

Flask Admin 通过在`admin`对象上注册定义一个或多个路由的视图类来工作。Flask Admin 主要有三种类型的视图：`ModelView`、`FileAdmin`和`BaseView`。接下来，我们将看到如何使用这些视图并对其进行自定义。

最后，我们向管理员界面添加一个导航栏选项，并且只将其渲染给具有管理员角色的用户。因此，在`templates/navbar.html`文件中，插入以下内容：

```py
{% if current_user.is_authenticated and current_user.has_role('admin') %}
<li class="nav-item">
    <a class="nav-link" href="{{url_for('admin.index')}}">
    Admin<span class="sr-only">(current)</span></a>
</li>
{% endif %}
```

# 创建基本管理页面

`BaseView`类允许将正常的 Flask 页面添加到您的`admin`界面中。这通常是 Flask Admin 设置中最少使用的视图类型，但如果您希望包含类似使用 JavaScript 图表库的自定义报告，您可以使用基础视图单独完成。正如预期的那样，我们将在`admin/controllers.py`文件中定义我们的视图：

```py
from flask.ext.admin import BaseView, expose 

class CustomView(BaseView): 
    @expose('/')
    @login_required
    @has_role('admin')
    def index(self): 
        return self.render('admin/custom.html') 

    @expose('/second_page')
    @login_required
    @has_role('admin')
    def second_page(self):
        return self.render('admin/second_page.html') 
```

在`BaseView`的子类中，如果它们一起定义，可以一次性注册多个视图。然而，请注意，每个`BaseView`的子类至少需要在`/`路径上有一个公开的方法。此外，除`/`路径内的方法之外的其他方法将不会出现在管理员界面的导航中，并将需要链接到该类中的其他页面。`expose`和`self.render`函数与正常 Flask API 中的对应函数工作方式完全相同。

要让您的模板继承 Flask Admin 的默认样式，我们在模板目录中创建一个新的文件夹，命名为`admin`，包含一个名为`custom.html`的文件，并添加以下 Jinja 代码：

```py
{% extends 'admin/master.html' %} 
{% block body %} 
    This is the custom view! 
    <a href="{{ url_for('.second_page') }}">Link</a> 
{% endblock %} 
```

要查看此模板，需要在`admin`对象上注册`CustomView`的一个实例。这将在`create_module`函数中完成，遵循与 API 模块相同的结构和逻辑：

```py
...
from .controllers import CustomView 
...
def create_module(object_name):
    ,,,
    admin.add_view(CustomView(name='Custom'))
```

`name` 关键字参数指定在 `admin` 接口顶部的导航栏上使用的标签应读取为“自定义”。在你将 `CustomView` 注册到 `admin` 对象之后，你的 `admin` 接口现在应该在导航栏中有一个第二个链接，如下面的截图所示：

![图片](img/53cf3e18-3657-4f5f-a025-99e502d0e002.png)

# 创建数据库管理页面

Flask Admin 的主要优势在于你可以通过提供 SQLAlchemy 或 MongoEngine 模型给 Flask Admin 来自动创建数据的管理页面。创建这些页面非常简单；在 `admin.py` 文件中，你只需要编写以下代码：

```py
from flask_admin.contrib.sqla import ModelView 
# or, if you use MongoEngine 
# from flask_admin.contrib.mongoengine import ModelView 

class CustomModelView(ModelView): 
    pass
```

然后，在 `admin/__init__.py` 文件中，注册数据库 `session` 对象以及你希望使用的模型类，如下所示：

```py
from flask_admin import Admin
from .controllers import CustomView, CustomModelView 
from webapp.blog.models import db, Reminder, Post, Comment, Tag
from webapp.auth.models import User, Role 

admin = Admin()
def create_module(app, **kwargs): 
    admin.init_app(app)
    admin.add_view(CustomView(name='Custom'))
    models = [User, Role, Comment, Tag, Reminder]

    for model in models: 
       admin.add_view(CustomModelView(model, db.session, 
       category='models'))
...
```

`category` 关键字告诉 Flask Admin 将具有相同类别值的所有视图放入导航栏上的同一个下拉菜单中。如果你现在打开浏览器，你会看到一个名为“模型”的新下拉菜单，其中包含指向数据库中所有表的管理页面的链接，如下所示：

![图片](img/dd891965-d771-4133-b666-3a74a4965b4e.png)

为每个模型生成的接口提供了很多功能。可以创建新的帖子，并且可以批量删除现有的帖子。所有字段都可以通过这个接口设置，包括作为可搜索下拉菜单实现的关系字段。`date` 和 `datetime` 字段甚至有自定义的 JavaScript 输入和下拉日历菜单。总的来说，这是对在 第六章 中创建的手动接口的一个巨大改进，*保护你的应用*。

# 增强 `post` 页面的管理功能

虽然这个接口在质量上有了巨大的提升，但仍然缺少一些功能。我们不再有原始接口中可用的 WYSIWYG 编辑器，但通过启用一些更强大的 Flask Admin 功能，这个页面可以得到改进。

要将 WYSIWYG 编辑器重新添加到 `post` 创建页面，我们需要一个新的 `WTForms` 字段，因为 Flask Admin 使用 Flask WTF 构建其表单。我们还需要用这个新字段类型覆盖 `post` 编辑和创建页面中的 `textarea` 字段。首先需要做的是在 `admin/forms.py` 文件中创建新的字段类型，使用 `textarea` 字段作为基础，如下所示：

```py
from wtforms import ( 
    widgets, 
    TextAreaField
) 

class CKTextAreaWidget(widgets.TextArea):
    def __call__(self, field, **kwargs):
        kwargs.setdefault('class_', 'ckeditor') 
        return super(CKTextAreaWidget, self).__call__(field, 
         **kwargs)

class CKTextAreaField(TextAreaField): 
    widget = CKTextAreaWidget() 
```

在这段代码中，我们创建了一个新的字段类型，`CKTextAreaField`，它为 `textarea` 添加了一个小部件。这个小部件所做的只是给 HTML 标签添加一个类。现在，要将这个字段添加到 `Post` 管理页面，`Post` 需要自己的 `ModelView`：

```py
from webapp.forms import CKTextAreaField 

class PostView(CustomModelView):
    form_overrides = dict(text=CKTextAreaField)
    column_searchable_list = ('text', 'title')
    column_filters = ('publish_date',)

    create_template = 'admin/post_edit.html'
    edit_template = 'admin/post_edit.html'
```

这段代码中有几个新内容。首先，`form_overrides` 类变量告诉 Flask Admin 用这个新字段类型覆盖名称文本的字段类型。`column_searchable_list` 函数定义了哪些列可以通过文本进行搜索。添加这个功能将允许 Flask Admin 在概览页面上包含一个搜索字段，我们可以通过这个字段搜索定义的字段值。接下来，`column_filters` 类变量告诉 Flask Admin 在该模型的概览页面上创建一个 `filters` 接口。`filters` 接口允许通过向显示的行添加条件来过滤非文本列。使用前面的代码可以实现的示例是创建一个过滤器，显示所有 `publish_date` 值大于 2015 年 1 月 1 日的行。

最后，`create_template` 和 `edit_template` 类变量允许你为 Flask Admin 定义自定义模板。对于我们将要使用的自定义模板，我们需要在 `admin` 文件夹中创建一个新的文件，名为 `post_edit.html`。在这个模板中，我们将包含与 第六章 中 *保护你的应用* 所使用的相同的 JavaScript 库，如下所示：

```py
{% extends 'admin/model/edit.html' %} 
{% block tail %} 
    {{ super() }} 
    <script 
        src="img/ckeditor.js"> 
    </script> 
{% endblock %} 
```

最后，要将我们新创建的自定义视图添加到 Flask-Admin 中，我们需要将其添加到 `admin/__init__.py` 文件中的 `create_module` 函数中：

```py
def create_module(app, **kwargs):
    ...
    admin.add_view(PostView(Post, db.session, category='Models'))
    ...    
```

继承模板的尾部位于文件末尾。一旦模板创建完成，你的 `post` 编辑和创建页面应该看起来像这样：

![图片](img/bd0e778c-7096-4441-97c8-4fad385505b4.png)

# 创建文件系统管理页面

大多数 `admin` 接口都覆盖的一个常见功能是能够从网络访问服务器的文件系统。幸运的是，Flask Admin 通过 `FileAdmin` 类包含了这个功能：

```py
class CustomFileAdmin(FileAdmin):
    pass
```

现在，只需将新类导入到你的 `admin/__init__.py` 文件中，并传入你希望从网络访问的路径：

```py
admin.add_view(CustomFileAdmin(app.static_folder,'/static/',name='Static Files'))
```

# 保护 Flask Admin

目前，整个 `admin` 界面对全世界都是可访问的——让我们来修复这个问题。`CustomView` 中的路由可以像任何其他路由一样进行保护，如下所示：

```py
class CustomView(BaseView): 
    @expose('/') 
    @login_required 
    @has_role('admin') 
    def index(self): 
        return self.render('admin/custom.html') 

    @expose('/second_page') 
    @login_required 
    @has_role('admin') 
    def second_page(self): 
        return self.render('admin/second_page.html') 
```

要保护 `ModeView` 和 `FileAdmin` 子类，它们需要定义一个名为 `is_accessible` 的方法，该方法要么返回 `true`，要么返回 `false`：

```py
class CustomModelView(ModelView): 
    def is_accessible(self): 
        return current_user.is_authenticated and 
               current_user.has_role('admin') 

class CustomFileAdmin(FileAdmin): 
    def is_accessible(self): 
        return current_user.is_authenticated and 
               current_user.has_role('admin') 
```

由于我们在 第六章 中正确设置了认证，*保护你的应用*，这个任务变得非常简单。

# Flask-Babel

在本节中，我们将探讨一种为我们的博客启用国际化的方法。这是构建支持多语言的全局网站的一个基本功能。我们将再次使用 Flask-Babel 扩展，它是由 Flask 的作者创建的。像往常一样，我们将确保这个依赖项存在于我们的 `requirements.txt` 文件中：

```py
...
Flask-Babel
...
```

Flask-Babel 使用 Babel Python 库进行国际化（i18n）和本地化，并添加了一些实用工具和 Flask 集成。要使用 Flask-Babel，首先我们需要在 `babel/babel.cfg` 文件中配置 Babel：

```py
[python: webapp/**.py]
[jinja2: webapp/templates/**.html]
encoding = utf-8
extensions=jinja2.ext.autoescape,jinja2.ext.with_
```

我们将 Babel 配置为仅在 `webapp` 目录中查找要翻译的 Python 文件，并从 `webapp/templates` 目录中的 `Jinja2` 模板中提取文本。

然后，我们需要在 `webapp/translations` 上创建一个翻译目录，其中将包含我们支持的所有语言的翻译。

Babel 附带一个名为 `pybabel` 的命令行实用程序。我们将使用它来设置我们博客将支持的所有语言，以及触发提取过程、更新和编译。首先，要创建一种新语言，输入以下命令：

```py
$ pybabel init -i ./babel/messages.pot -d ./webapp/translations -l pt 
```

葡萄牙语，或 `pt`，已经在提供的支持代码中初始化，但你可以尝试创建一种新语言。只需将 `pt` 改为其他语言。之后，你可以检查 `webapp/translations`，应该会看到 Babel 已经创建了一个包含我们语言代码的新目录。此目录包含一个 `messages.po` 文件，我们将在此文件中编写提取文本所需的翻译，以及 `messages.mo` 的编译版本。

接下来，为了触发 Babel 在我们的应用程序中搜索要翻译的文本，使用以下命令：

```py
$ pybabel extract -v -F ./babel/babel.cfg -o ./babel/messages.pot .
```

这将更新 `messages.pot` 主文件，其中包含所有需要翻译的文本。然后，我们告诉 Babel 使用以下命令更新所有支持语言的 `messages.po` 文件：

```py
$ pybabel update -i ./babel/messages.pot -d webapp/translations
```

现在，`messages.po` 文件将包含类似以下内容：

```py
# Portuguese translations for PROJECT.
# Copyright (C) 2018 ORGANIZATION
# This file is distributed under the same license as the PROJECT project.
# FIRST AUTHOR <EMAIL@ADDRESS>, 2018.
#
msgid ""
msgstr ""
"Project-Id-Version: PROJECT VERSION\n"
...

#: webapp/templates/head.html:5
msgid "Welcome to this Blog"
msgstr ""

#: webapp/templates/macros.html:57
msgid "Read More"
msgstr ""

...
```

在这里，翻译者需要使用从 `msgid` 翻译的文本更新 `msgstr`。从英语翻译到某种目标语言。完成此操作后，我们将告诉 Babel 编译 `messages.po` 文件，并使用以下命令生成更新的 `messages.mo` 文件：

```py
$ pybabel compile -d ./webapp/translations
```

Babel 如何识别我们应用程序中要翻译的文本？很简单——`Jinja2` 已经为 Babel 准备好了，所以在我们的模板中，我们只需输入以下内容：

```py
<h1>{{_('Some text to translate')}}</h1>
```

`_('text')` 是 `gettext` 函数的别名，如果存在翻译，则返回字符串的翻译，对于可能变为复数的文本使用 `ngettext`。

对于 Flask 集成，我们将创建一个名为 `webapp/babel` 的新模块。这是我们初始化扩展的地方。为此，在 `babel/__init__.py` 文件中添加以下内容：

```py
from flask import has_request_context, session
from flask_babel import Babel

babel = Babel()
...
def create_module(app, **kwargs):
    babel.init_app(app)
    from .controllers import babel_blueprint
    app.register_blueprint(babel_blueprint)
```

然后，我们需要定义一个函数，该函数返回当前的区域代码给 Flask-Babel。最佳位置是在 `babel/__init__.py` 文件中添加它：

```py
...
@babel.localeselector
def get_locale():
    if has_request_context():
        locale = session.get('locale')
        if locale:
            return locale
        session['locale'] = 'en'
        return session['locale']
...
```

我们将使用会话来保存当前选定的区域设置，如果没有设置，我们将回退到英语。我们的函数用 `@babel.localeselector` 装饰器装饰，以在 Flask-Babel 上注册我们的函数。

接下来，我们需要定义一个端点，可以通过调用它来切换当前选定的语言。此端点将会将会话区域设置为新的语言并重定向到主页。通过在 `babel/controllers.py` 文件中添加以下代码来完成此操作：

```py
from flask import Blueprint, session, redirect, url_for

babel_blueprint = Blueprint(
    'babel',
    __name__,
    url_prefix="/babel"
)

@babel_blueprint.route('/<string:locale>')
def index(locale):
    session['locale'] = locale
    return redirect(url_for('blog.home'))
```

最后，我们将为我们的用户提供一种更改当前语言的方式。这将在导航栏中完成。为此，将以下内容添加到 `templates/navbar.html` 文件中：

```py
...
<ul class="navbar-nav ml-auto">
    <li class="nav-item dropdown">
        <a class="nav-link dropdown-toggle" href="#" 
        id="navbarDropdown" role="button" data-toggle="dropdown">
            Lang
        </a>
        <div class="dropdown-menu">
            <a class="dropdown-item" href="{{url_for('babel.index', 
            locale='en')}}">en</a>
            <a class="dropdown-item" href="{{url_for('babel.index', 
            locale='pt')}}">pt</a>
        </div>
    </li>
...
</ul>
```

新的导航栏选项将带我们到带有选定语言的 Babel 索引端点。任何我们想要支持的新语言都应该添加到这里。最后，我们只需在我们的主 `__init__.py` 文件上调用 Babel 的 `create_module` 函数即可。

```py
def create_app():
...
    from babel import create_module as babel_create_module
...
    babel_create_module(app)
```

就这样。我们现在已经设置了所有必要的配置，以支持我们博客应用程序上的任何语言。

![图片](img/0264c6b3-d3e0-4420-829d-b6a38b377432.png)

# Flask Mail

本章将要介绍的最后一个 Flask 扩展是 Flask Mail，它允许你从 Flask 的配置中连接和配置你的 SMTP 客户端。Flask Mail 还将有助于简化 第十二章，*测试 Flask 应用程序* 中的应用程序测试。第一步是使用 `pip` 安装 Flask Mail。你应该在本章中已经完成了这个步骤，在我们的 `init.sh` 脚本中，所以让我们检查我们的依赖文件，以确保以下内容：

```py
...
Flask-Mail
...
```

`flask_mail` 将通过读取 `app` 对象中的配置变量来连接我们选择的 SMTP 服务器，因此我们需要将这些值添加到我们的 `config` 对象中：

```py
class DevConfig(Config): 

    MAIL_SERVER = 'localhost' 
    MAIL_PORT = 25 
    MAIL_USERNAME = 'username' 
    MAIL_PASSWORD = 'password' 
```

最后，在 `_init_.py` 中初始化 `mail` 对象：

```py
...
from flask_mail import Mail
...
mail = Mail()

def create_app(object_name): 
...
    mail.init_app(app)
...
```

要了解 Flask Mail 如何简化我们的电子邮件代码，请考虑以下内容——这个代码片段是我们创建在 第九章，*使用 Celery 创建异步任务* 中的 Remind 任务，但使用 Flask Mail 而不是标准库的 SMTP 模块：

```py
from flask_mail import Message
from .. import celery, mail

@celery.task(
    bind=True,
    ignore_result=True,
    default_retry_delay=300,
    max_retries=5
)
def remind(self, pk):
    logs.info("Remind worker %d" % pk)
    reminder = Reminder.query.get(pk)
    msg = Message(body="Text %s" % str(reminder.text), 
    recipients=[reminder.email], subject="Your reminder")
    try:
        mail.send(msg)
        logs.info("Email sent to %s" % reminder.email)
        return
    except Exception as e:
        logs.error(e)
        self.retry(exc=e)
```

# 摘要

本章中的任务使我们能够显著扩展我们应用程序的功能。我们现在拥有了一个功能齐全的管理员界面，浏览器中的一个有用的调试工具，两个可以大大加快页面加载时间的工具，以及一个使发送电子邮件不那么头疼的实用工具。

如本章开头所述，Flask 是一个基础框架，允许你挑选和选择你需要的功能。因此，重要的是要记住，在你的应用程序中并不需要包含所有这些扩展。如果你是唯一一个在应用程序上工作的内容创作者，CLI 可能就是你所需要的，因为添加这些功能会占用开发时间（当它们不可避免地出现问题时，还会占用维护时间）。这个警告是在本章末尾给出的，因为许多 Flask 应用程序变得难以管理的主要原因之一就是它们包含了太多的扩展，测试和维护所有这些扩展变成了一项非常庞大的任务。

在下一章中，你将学习扩展的内部工作原理，以及如何创建你自己的扩展。
