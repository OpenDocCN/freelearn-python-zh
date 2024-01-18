# 使用 Django 创建在线视频游戏商店

我出生在 70 年代末，这意味着我在视频游戏产业诞生时长大。我的第一款视频游戏主机是 Atari 2600，正是因为这款特定的视频游戏主机，我决定要成为一名程序员并制作视频游戏。然而，我从未在游戏行业找到工作，但我仍然喜欢玩视频游戏，在业余时间里，我尝试开发自己的游戏。

直到今天，我仍然在互联网上四处转悠，尤其是在 eBay 上，购买旧的视频游戏，以重温我美好的童年回忆，当时全家，我的父母和姐姐，都喜欢一起玩 Atari 2600 游戏。

由于我对复古视频游戏很感兴趣，我们将开发一个复古视频游戏在线商店；这将是一个很好的方式来开发有趣的东西，同时也学到很多关于流行的 Django web 框架的网页开发知识。

在本章中，我们将涵盖以下内容：

+   设置环境

+   创建一个 Django 项目

+   创建 Django 应用程序

+   探索 Django 管理界面

+   学习如何创建应用程序模型并使用 Django ORM 执行查询

另外，作为额外的内容，我们将使用**npm**（**Node Package Manager**）来下载客户端依赖项。我们还将介绍如何使用任务运行器 Gulp 创建简单的任务。

为了让我们的应用程序更漂亮，我们将使用 Bootstrap。

所以，让我们开始吧！

# 设置开发环境

像往常一样，我们将开始为开发设置环境。在第四章中，*汇率和货币转换工具*，你已经了解了`pipenv`，所以在本章和接下来的章节中，我们将使用`pipenv`来创建我们的虚拟环境和管理我们的依赖项。

首先，我们要创建一个目录，用来存放我们的项目。在你的工作目录中，创建一个名为`django-project`的目录，如下所示：

```py
mkdir django-project && cd django-project
```

现在我们可以运行`pipenv`来创建我们的虚拟环境：

```py
pipenv --three
```

如果你已经在其他位置安装了 Python 3，你可以使用参数`--python`并指定 Python 可执行文件的路径。如果一切顺利，你应该会看到如下输出：

![](img/ade4540c-d7e8-469a-ac30-b921844e9030.png)

现在我们可以使用`pipenv`命令行激活我们的虚拟环境：

```py
pipenv shell
```

太棒了！我们现在要添加的唯一依赖项是 Django。

在撰写本书时，Django 2.0 已经发布。与之前相比，它有很多很好的功能。你可以在[`docs.djangoproject.com/en/2.0/releases/2.0/`](https://docs.djangoproject.com/en/2.0/releases/2.0/)上查看新功能列表。

让我们在我们的虚拟环境中安装 Django：

```py
pipenv install django
```

Django 2.0 已经停止支持 Python 2.0，所以如果你计划使用 Python 2 开发应用程序，你应该安装 Django 1.11.x 或更低版本。我强烈建议你使用 Python 3 开始一个新项目。Python 2 将在几年后停止维护，并且新的包将为 Python 3 创建。Python 2 的流行包将迁移到 Python 3。

在我看来，Django 2 最好的新功能是新的路由语法，因为现在不需要编写正则表达式。像下面这样写更加清晰和可读：

```py
path('user/<int:id>/', views.get_user_by_id)
```

以前的语法更多地依赖于正则表达式：

```py
url('^user/?P<id>[0-9]/$', views.get_user_by_id)
```

这样会简单得多。我在 Django 2.0 中真正喜欢的另一个功能是他们稍微改进了管理 UI，并使其响应式；这是一个很棒的功能，因为我曾经在小手机屏幕上使用非响应式网站时，创建新用户（当你在外出时无法访问桌面）会很痛苦。

# 安装 Node.js

在 Web 开发方面，几乎不可能远离 Node.js。 Node.js 是一个于 2009 年发布的项目。它是一个 JavaScript 运行时，允许我们在服务器端运行 JavaScript。如果我们使用 Django 和 Python 开发网站，为什么要关心 Node.js 呢？原因是 Node.js 生态系统有几个工具，将帮助我们以简单的方式管理客户端依赖关系。我们将使用其中一个工具，即 npm。

将 npm 视为 JavaScript 世界的`pip`。然而，npm 有更多功能。我们将使用的功能之一是 npm 脚本。

所以，让我们继续安装 Node.js。通常，开发人员需要转到 Node.js 网站并从那里下载，但我发现使用一个名为 NVM 的工具更简单，它允许我们轻松安装和切换不同版本的 Node.js。

要在我们的环境中安装 NVM，您可以按照[`github.com/creationix/nvm`](https://github.com/creationix/nvm)上的说明进行操作。

我们正在介绍在 Unix/Linux 和 macOS 系统上安装 NVM。如果您使用 Windows，有一个使用 Go 语言开发的 Windows 版本，可以在[`github.com/coreybutler/nvm-windows`](https://github.com/coreybutler/nvm-windows)找到。

安装 NVM 后，您可以使用以下命令安装最新版本的 Node.js：

```py
nvm install node
```

您可以使用以下命令验证安装是否正确：

```py
node --version
```

在编写本书时，最新的 Node.js 版本是 v8.8.1。

您还可以在终端上输入`npm`，您应该看到类似于以下输出的输出：

![](img/36ec4dc3-0d99-4c0e-8604-c7a67979e542.png)

# 创建一个新的 Django 项目

要创建一个新的 Django 项目，请运行以下命令：

```py
django-admin startproject gamestore
```

请注意，`django-admin`创建了一个名为`gamestore`的目录，其中包含一些样板代码。我们将在稍后查看 Django 创建的文件，但首先，我们将创建我们的第一个 Django 应用程序。在 Django 世界中，您有项目和应用程序，根据 Django 文档，项目描述了 Web 应用程序本身，应用程序是一个提供某种功能的 Python 包；这些应用程序包含自己的一组路由、视图、静态文件，并且可以在不同的 Django 项目中重复使用。

如果您完全不理解，不要担心；随着我们的进展，您会学到更多。

说了这么多，让我们创建项目的初始应用程序。运行`cd gamestore`，一旦进入`gamestore`目录，执行以下命令：

```py
python-admin startapp main
```

如果列出`gamestore`目录的内容，您应该会看到一个名为`main`的新目录；那是我们将要创建的 Django 应用程序的目录。

在不写任何代码的情况下，您已经拥有一个完全功能的 Web 应用程序。要运行应用程序并查看结果，请运行以下命令：

```py
python manage.py runserver
```

您应该看到以下输出：

```py
Performing system checks...

System check identified no issues (0 silenced).

You have 14 unapplied migration(s). Your project may not work properly until you apply the migrations for app(s): admin, auth, contenttypes, sessions.
Run 'python manage.py migrate' to apply them.

December 20, 2017 - 09:27:48
Django version 2.0, using settings 'gamestore.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```

打开您喜欢的 Web 浏览器，转到`http://127.0.0.1:8000`，您将看到以下页面：

![](img/91704c14-da95-48eb-9d06-64970b558e88.png)

当我们第一次启动应用程序时，需要注意的一点是以下警告：

```py
You have 14 unapplied migration(s). Your project may not work properly until you apply the migrations for app(s): admin, auth, contenttypes, sessions.
Run 'python manage.py migrate' to apply them.
```

这意味着 Django 项目默认注册的应用程序`admin`、`auth`、`contenttypes`和`sessions`有尚未应用到该项目的迁移（数据库更改）。我们可以使用以下命令运行这些迁移：

```py
➜ python manage.py migrate
Operations to perform:
 Apply all migrations: admin, auth, contenttypes, sessions
Running migrations:
 Applying contenttypes.0001_initial... OK
 Applying auth.0001_initial... OK
 Applying admin.0001_initial... OK
 Applying admin.0002_logentry_remove_auto_add... OK
 Applying contenttypes.0002_remove_content_type_name... OK
 Applying auth.0002_alter_permission_name_max_length... OK
 Applying auth.0003_alter_user_email_max_length... OK
 Applying auth.0004_alter_user_username_opts... OK
 Applying auth.0005_alter_user_last_login_null... OK
 Applying auth.0006_require_contenttypes_0002... OK
 Applying auth.0007_alter_validators_add_error_messages... OK
 Applying auth.0008_alter_user_username_max_length... OK
 Applying auth.0009_alter_user_last_name_max_length... OK
 Applying sessions.0001_initial... OK
```

在这里，Django 在 SQLite 数据库中创建了所有表，您将在应用程序的`root`目录中找到 SQLite 数据库文件。

`db.sqlite3`文件是包含我们应用程序表的数据库文件。选择 SQLite 只是为了使本章的应用程序更简单。Django 支持大量数据库；最受欢迎的数据库，如 Postgres、Oracle，甚至 MSSQL 都受支持。

如果再次运行`runserver`命令，就不应该有任何迁移警告了：

```py
→ python manage.py runserver
Performing system checks...

System check identified no issues (0 silenced).
December 20, 2017 - 09:50:49
Django version 2.0, using settings 'gamestore.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```

现在我们只需要做一件事来结束这一部分；我们需要创建一个管理员用户，这样我们就可以登录到 Django 管理界面并管理我们的 Web 应用程序。

与 Django 中的其他一切一样，这非常简单。只需运行以下命令：

```py
python manage.py createsuperuser
```

你将被要求输入用户名和电子邮件，并设置密码，这就是你设置管理员帐户所需要做的一切。

在接下来的部分，我们将更仔细地查看 Django 为我们创建的文件。

# 探索 Django 项目的结构

如果你看一下 Django 的网站，它说*Django：完美主义者的网络框架，有截止日期*，我完全同意这个说法。到目前为止，我们还没有写任何代码，我们已经有了一个正在运行的网站。只需几个命令，我们就可以创建一个具有相同目录结构和样板代码的新项目。让我们开始开发。

我们可以设置一个新的数据库并创建一个超级用户，而且，Django 还带有一个非常好用和有用的管理界面，你可以在其中查看我们的数据和用户。

在这一部分，我们将探索 Django 在启动新项目时为我们创建的代码，以便我们熟悉结构。让我们继续添加项目的其他组件。

如果你查看项目的根目录，你会发现一个名为`db.sqlite3`的文件，另一个名为`manage.py`的文件，最后，还有一个与项目同名的目录，在我们的例子中是`gamestore`。`db.sqlite3`文件，顾名思义，是数据库文件；这个文件是在项目的根文件夹中创建的，因为我们正在使用 SQLite。你可以直接从命令行探索这个文件；我们很快会演示如何做到这一点。

第二个文件是`manage.py`。这个文件是由`django-admin`在每个 Django 项目中自动创建的。它基本上做的事情和`django-admin`一样，再加上两件额外的事情；它会将`DJANGO_SETTINGS_MODULE`设置为指向项目的设置文件，并将项目的包放在`sys.path`上。如果你执行`manage.py`而没有任何参数，你可以看到所有可用命令的帮助。

如你所见，`manage.py`有许多选项，比如管理密码，创建超级用户，管理数据库，创建和执行数据库迁移，启动新应用和项目，以及一个非常重要的选项`runserver`，正如其名字所示，它将为你启动 Django 开发服务器。

现在我们已经了解了`manage.py`以及如何执行它的命令，我们将退一步，学习如何检查我们刚刚创建的数据库。做到这一点的命令是`dbshell`；让我们试一试：

```py
python manage.py dbshell
```

# 深入 SQLite

你应该进入 SQLite3 命令提示符：

```py
SQLite version 3.16.2 2017-01-06 16:32:41
Enter ".help" for usage hints.
sqlite>
```

如果你想获取数据库的所有表的列表，可以使用命令`.tables`：

```py
sqlite> .tables
auth_group auth_user_user_permissions
auth_group_permissions django_admin_log
auth_permission django_content_type
auth_user django_migrations
auth_user_groups django_session
```

在这里你可以看到，我们通过`migrate`命令创建的所有表。

要查看每个表的结构，可以使用命令`.schema`，我们可以使用选项`--indent`，这样输出将以更可读的方式显示：

```py
sqlite> .schema --indent auth_group
CREATE TABLE IF NOT EXISTS "auth_group"(
 "id" integer NOT NULL PRIMARY KEY AUTOINCREMENT,
 "name" varchar(80) NOT NULL UNIQUE
 );
```

这些是我在使用 SQLite3 数据库时最常用的命令，但命令行界面提供了各种命令。你可以使用`.help`命令获取所有可用命令的列表。

当创建原型、概念验证项目或者创建非常小的项目时，SQLite3 数据库非常有用。如果我们的项目不属于这些类别中的任何一种，我建议使用其他 SQL 数据库，比如 MySQL、Postgres 和 Oracle。还有非 SQL 数据库，比如 MongoDB。使用 Django，你可以毫无问题地使用这些数据库；如果你使用 Django 的 ORM（对象关系模型），大部分时间你可以在不同的数据库之间切换，应用程序仍然可以完美地工作。

# 查看项目的包目录

接下来，让我们看看项目的包目录。在那里，你会找到一堆文件。你会看到的第一个文件是`settings.py`，这是一个非常重要的文件，因为你将在这里放置我们应用程序的所有设置。在这个设置文件中，你可以指定将使用哪些应用程序和数据库，你还可以告诉 Django 在哪里搜索静态文件和模板、中间件等。

然后你有`urls.py`；这个文件是你指定应用程序可用的 URL 的地方。你可以在项目级别设置 URL，也可以为每个 Django 应用程序设置 URL。如果你检查这个`urls.py`文件的内容，你不会找到太多细节。基本上，你会看到一些解释如何添加新的 URL 的文本，但 Django 已经定义了（开箱即用）一个 URL 到 Django 管理站点：

```py
  from django.contrib import admin
  from django.urls import path

  urlpatterns = [
      path('admin/', admin.site.urls),
  ]
```

我们将逐步介绍如何向项目添加新的 URL，但无论如何我们都可以解释这个文件；还记得我提到过在 Django 中可以有不同的应用吗？所以`django.contrib.admin`也是一个应用，而一个应用有自己的一组 URL、视图、模板。那么它在这里做什么？当我们导入 admin 应用然后定义一个名为`urlpatterns`的列表时，在这个列表中我们使用一个名为 path 的函数，第一个参数是 URL，第二个参数可以是一个将要执行的视图。但在这种情况下，它传递了`admin.site`应用的 URL，这意味着`admin/`将是基本 URL，而`admin.site.urls`中定义的所有 URL 将在其下创建。

例如，如果在`admin.site.url`中，我定义了两个 URL，`users/`和`groups/`，当我有`path('admin/', admin.site.urls)`时，我实际上将创建两个 URL：

+   `admin/users/`

+   `admin/groups/`

最后，我们有`wsgi.py`，这是 Django 在创建新项目时为我们创建的一个简单的 WSGI 配置。

现在我们对 Django 项目的结构有了一些了解，是时候创建我们项目的第一个应用了。

# 创建项目的主要应用

在这一部分，我们将创建我们的第一个 Django 应用程序。一个 Django 项目可以包含多个应用程序。将项目拆分为应用程序是一个很好的做法，原因有很多；最明显的是你可以在不同的项目中重用相同的应用程序。将项目拆分为多个应用程序的另一个原因是它强制实现关注点的分离。你的项目将更有组织，更容易理解，我们的同事会感谢你，因为这样维护起来会更容易。

让我们继续运行`startapp`命令，并且，如前所示，你可以使用`django-admin`命令或者使用`manager.py`。由于我们使用`django-admin`命令创建了项目，现在是一个很好的机会来测试`manager.py`命令。要创建一个新的 Django 应用程序，请运行以下命令：

```py
python manager.py startapp main
```

在这里，我们将创建一个名为`main`的应用程序。不要担心没有显示任何输出，Django 会悄悄地创建项目和应用程序。如果你现在列出目录内容，你会看到一个名为`main`的目录，而在`main`目录中你会找到一些文件；我们将在添加更改时解释每个文件。

所以，我们想要做的第一件事是为我们的应用程序添加一个登陆页面。为此，我们需要做三件事：

+   首先，我们添加一个新的 URL，告诉 Django 当我们网站的用户浏览到根目录时，它应该转到站点`/`并显示一些内容

+   第二步是添加一个视图，当用户浏览到站点的根目录``/``时将执行该视图

+   最后一步是添加一个包含我们希望向用户显示的内容的 HTML 模板

说到这一点，我们需要在`main`应用程序目录中包含一个名为`urls.py`的新文件。首先，我们添加一些导入：

```py
from django.urls import path
from . import views
```

在前面的代码中，我们从`django.urls`中导入了 path 函数。path 函数将返回一个要包含在`urlpatterns`列表中的元素，我们还在同一目录中导入了 views 文件；我们想要导入这个视图，因为我们将在那里定义在访问特定路由时将执行的函数：

```py
  urlpatterns = [
      path(r'', views.index, name='index'),
  ]
```

然后我们使用 path 函数来定义一个新的路由。函数 path 的第一个参数是一个包含我们希望在应用程序中提供的 URL 模式的字符串。这个模式可能包含尖括号（例如`<int:user_id>`）来捕获 URL 上传递的参数，但是在这一点上，我们不打算使用它；我们只是想为应用程序的根添加一个 URL，所以我们添加一个空字符串`''`。第二个参数是将要执行的函数，可选地，您可以添加关键字参数`name`，它设置 URL 的名称。我们很快就会看到为什么这很有用。

第二部分是在`views.py`文件中定义名为`index`的函数，如下所示：

```py
  from django.shortcuts import render

  def index(request):
      return render(request, 'main/index.html', {})
```

由于此时没有太多事情要做，我们首先从`django.shortcuts`中导入 render 函数。Django 有自己的模板引擎，内置在框架中，可以将默认模板引擎更改为您喜欢的其他模板引擎（例如 Jinja2，这是 Python 生态系统中最受欢迎的模板引擎之一），但是为了简单起见，我们将使用默认引擎。`render`函数获取请求对象、模板和上下文对象；后者是一个包含要在模板中显示的数据的对象。

我们需要做的下一件事是添加一个模板，该模板将包含我们希望在用户浏览我们的应用程序时显示的内容。现在，大多数 Web 应用程序的页面包含永远不会改变的部分，例如顶部菜单栏或页面页脚，这些部分可以放入一个单独的模板中，可以被其他模板重用。幸运的是，Django 模板引擎具有这个功能。事实上，我们不仅可以在模板中注入子模板，还可以有一个基本模板，其中包含将在所有页面之间共享的 HTML。说到这一点，我们将在`gamestore/templates`目录中创建一个名为`base.html`的文件，其中包含以下内容：

```py
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, 
        initial-scale=1">
    <meta name="description" content="">
    <meta name="author" content="">
    <link rel="icon" href="../../favicon.ico">

    <title>Vintage video games store</title>

    {% load staticfiles %}
    <link href="{% static 'styles/site.css' %}" rel='stylesheet'>
    <link href="{% static 'styles/bootstrap.min.css' %}" 
       rel='stylesheet'>
    <link href="{% static 'styles/font-awesome.min.css' %}"
          rel='stylesheet'>
  </head>

  <body>

    <nav class="navbar navbar-inverse navbar-fixed-top">
      <div class="container">
        <div class="navbar-header">
          <button type="button" class="navbar-toggle 
             collapsed" data-toggle="collapse" data-target="#navbar"
             aria-expanded="false" aria-controls="navbar">
            <span class="sr-only">Toggle navigation</span>
            <span class="icon-bar"></span>
            <span class="icon-bar"></span>
            <span class="icon-bar"></span>
          </button>
          <a class="navbar-brand" href="/">Vintage video
         games store</a>
        </div>
        <div id="navbar" class="collapse navbar-collapse">
          <ul class="nav navbar-nav">
            <li>
              <a href="/">
                <i class="fa fa-home" aria-hidden="true"></i> HOME
              </a>
            </li>
            {% if user.is_authenticated%}
            <li>
              <a href="/cart/">
                <i class="fa fa-shopping-cart" 
                   aria-hidden="true"></i> CART
              </a>
            </li>
            {% endif %}
          </ul>          
        </div><!--/.nav-collapse -->
      </div>
    </nav>

    <div class="container">

      <div class="starter-template">
        {% if messages %}
          {% for message in messages %}
            <div class="alert alert-info" role="alert">
              {{message}}
            </div>
          {% endfor %}
        {% endif %}

        {% block 'content' %}
        {% endblock %}
      </div>
    </div>
  </body>
</html>
```

我们不打算逐个讨论所有 HTML 部分，只讨论 Django 模板引擎的特定语法部分：

```py
  {% load static %}
  <link href="{% static 'styles/site.css' %}" rel='stylesheet'>
  <link href="{% static 'styles/bootstrap.min.css' %}" 
          rel='stylesheet'>
  <link href="{% static 'styles/font-awesome.min.css' %}" 
         rel='stylesheet'>
```

这里需要注意的第一件事是`{% load static %}`，它将告诉 Django 的模板引擎我们要加载静态模板标签。静态模板标签用于链接静态文件。这些文件可以是图像、JavaScript 或样式表文件。你可能会问，Django 是如何找到这些文件的呢，答案很简单：通过魔法！不，开玩笑；静态模板标签将在`settings.py`文件中的`STATIC_ROOT`变量指定的目录中查找文件；在我们的情况下，我们定义了`STATIC_ROOT = '/static/'`，所以当使用标签`{% static 'styles/site.css' %}`时，链接`/static/styles/site.css`将被返回。

你可能会想，为什么不只写`/static/styles/site.css`而不使用标签？这样做的原因是，标签为我们提供了更多的灵活性，以便在需要更新我们提供静态文件的路径时进行更改。想象一种情况，你有一个包含数百个模板的大型应用程序，在所有这些模板中，你都硬编码了`/static/`，然后决定更改该路径（而且你没有团队）。你需要更改每个文件来执行此更改。如果你使用静态标签，你只需将文件移动到不同的位置，标签就会更改`STATIC_ROOT`变量在设置文件中的值。

我们在这个模板中使用的另一个标签是`block`标签：

```py
{% block 'content' %}
{% endblock %}
```

`block`标签非常简单；它定义了基本模板中可以被子模板用来在该区域注入内容的区域。当我们创建下一个模板文件时，我们将看到这是如何工作的。

第三部分是添加模板。`index`函数将呈现存储在`main/index.html`的模板，这意味着它将留在`main/templates/main/`目录中。让我们继续创建文件夹`main/templates`，然后`main/templates/main`：

```py
mkdir main/templates && mkdir main/templates/main
```

在`main/templates/main/`目录中创建一个名为`index.html`的文件，内容如下：

```py
{% extends 'base.html' %}

{% block 'content' %}
   <h1>Welcome to the gamestore!</h1>
{% endblock %}
```

正如你在这里看到的，我们首先扩展了基本模板，这意味着`base.html`文件的所有内容将被 Django 模板引擎用来构建 HTML，当用户浏览到`/`时，将提供给浏览器。现在，我们还使用了`block`标签；在这种情况下，它意味着引擎将在`base.html`文件中搜索名为`'content'`的块标签，如果找到，引擎将在`'content'`块中插入`h1 html`标签。

这一切都是关于代码的可重用性和可维护性，因为你不需要在我们应用程序的每个单个模板中插入菜单标记和加载 JavaScript 和 CSS 文件的标记；你只需要在基本模板中插入它们并在这里使用`block`标签。内容会改变。使用基本模板的第二个原因是，再次想象一种情况，你需要改变一些东西——比如我们在`base.html`文件中定义的顶部菜单，因为菜单只在`base.html`文件中定义。要执行更改，你只需要在`base.html`中更改标记，所有其他模板将继承更改。

我们几乎准备好运行我们的代码并查看应用程序目前的外观了，但首先，我们需要安装一些客户端依赖项。

# 安装客户端依赖项

现在我们已经安装了 NodeJS，我们可以安装项目的客户端依赖项。由于本章的重点是 Django 和 Python，我们不想花太多时间来设计我们的应用程序并浏览庞大的 CSS 文件。然而，我们希望我们的应用程序看起来很棒，因此我们将安装两样东西：Bootstrap 和 Font Awesome。

Bootstrap 是一个非常著名的工具包，已经存在多年了。它有一套非常好的组件、网格系统和插件，将帮助我们使我们的应用程序在用户在桌面上浏览应用程序或者甚至移动设备上浏览应用程序时看起来很棒。

Font Awesome 是另一个存在已久的项目，它是一个字体和图标框架。

要安装这些依赖项，我们可以直接运行 npm 的安装命令。然而，我们要做得更好。类似于`pipenv`，它为我们的 Python 依赖项创建一个文件，`npm`也有类似的东西。这个文件叫做`package.json`，它不仅包含了项目的依赖项，还包含了关于包的脚本和元信息。

让我们继续将`package.json`文件添加到`gamestore/`目录中，内容如下：

```py
    {
      "name": "gamestore",
      "version": "1.0.0",
      "description": "Retro game store website",
      "dependencies": {
         "bootstrap": "³.3.7",
        "font-awesome": "⁴.7.0"
      }
    }
```

太棒了！保存文件，并在终端上运行以下命令：

```py
npm install
```

如果一切顺利，您应该会看到一条消息，说明已安装了两个软件包。

如果列出`gamestore`目录的内容，您将看到`npm`创建了一个名为`node_modules`的新目录，`npm`安装了 Bootstrap 和 Font Awesome。

为简单起见，我们将只复制我们需要的 CSS 文件和字体到`static`文件夹。 但是，在构建应用程序时，我建议使用诸如`webpack`之类的工具，它将捆绑所有我们的客户端依赖项，并设置`webpack`开发服务器来为您的 Django 应用程序提供文件。 由于我们想专注于 Python 和 Django，我们可以继续手动复制文件。

首先，我们可以按以下方式创建 CSS 文件的目录：

```py
mkdir static && mkdir static/styles
```

然后我们需要复制 bootstrap 文件。 首先是最小化的 CSS 文件：

```py
cp node_modules/bootstrap/dist/css/bootstrap.min.css static/styles/
```

接下来，我们需要复制 Font Awesome 文件，从最小化的 CSS 开始：

```py
cp node_modules/font-awesome/css/font-awesome.min.css static/styles/
```

和字体：

```py
cp -r node_modules/font-awesome/fonts/ static/
```

我们将添加另一个 CSS 文件，其中将包含我们可能添加到应用程序中的一些自定义 CSS，以赋予应用程序个性化的外观。 在`gamestore/static/styles`目录中添加一个名为`site.css`的文件，内容如下：

```py
  .nav.navbar-nav .fa-home,
  .nav.navbar-nav .fa-shopping-cart {
     font-size: 1.5em;
   }

   .starter-template {
      padding: 70px 15px;
   }

   h2.panel-title {
      font-size: 25px;
   }
```

我们需要做一些事情来第一次运行我们的应用程序； 首先，我们需要将我们创建的主应用程序添加到`gamestore/gamestore`目录中的`settings.py`文件的`INSTALLED_APPS`列表中。 它应如下所示：

```py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'main',
]
```

在同一设置文件中，您将找到列表`TEMPLATES`：

```py
TEMPLATES = [
    {
        'BACKEND': 
 'django.templates.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.templates.context_processors.debug',
                'django.templates.context_processors.request',
                'django.contrib.auth.context_processors.auth',

 'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```

`DIRS`的值是一个空列表。 我们需要将其更改为：

```py
'DIRS': [
    os.path.join(BASE_DIR, 'templates')
]
```

这将告诉 Django 在`templates`目录中搜索模板。

然后，在`settings.py`文件的末尾添加以下行：

```py
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static'), ]
```

这将告诉 Django 在`gamestore/static`目录中搜索静态文件。

现在我们需要告诉 Django 注册我们在`main`应用程序中定义的 URL。 因此，让我们继续打开`gamestore/gamestore`目录中的文件`urls.py`。 我们需要在`urlpatterns`列表中包含`"main.urls"`。 更改后，`urls.py`文件应如下所示：

```py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('main.urls'))
]
```

请注意，我们还需要导入`django.urls`模块的`include`函数。

太好了！ 现在我们已经准备好使用我们的应用程序中的所有客户端依赖项，并且可以第一次启动应用程序以查看我们迄今为止实施的更改。 打开终端，并使用`runserver`命令启动 Django 的开发服务器，如下所示：

```py
python manage.py runserver
```

浏览到`http://localhost:8000`； 您应该看到一个页面，类似于以下截图所示的页面：

![](img/b2034c76-688d-456c-9e99-41b3b0429427.png)

# 添加登录和注销视图

每个在线商店都需要某种用户管理。 我们应用的用户应该能够创建帐户，更改其帐户详细信息，显然登录到我们的应用程序，以便他们可以下订单，还可以从应用程序注销。

我们将开始添加登录和注销功能。 好消息是，在 Django 中实现这一点非常容易。

首先，我们需要在我们的登录页面上添加一个 Django 表单。 Django 有一个内置的身份验证表单； 但是，我们想要自定义它，所以我们将创建另一个类，该类继承自 Django 内置的`AuthenticationForm`并添加我们的更改。

在`gamestore/main/`中创建一个名为`forms.py`的文件，内容如下：

```py
from django import forms
from django.contrib.auth.forms import AuthenticationForm

class AuthenticationForm(AuthenticationForm):
    username = forms.CharField(
        max_length=50,
        widget=forms.TextInput({
            'class': 'form-control',
            'placeholder': 'User name'
  })
    )

    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput({
            'class': 'form-control',
            'placeholder': 'Password'
  })
    )
```

这个类非常简单。 首先，我们从`django`模块导入`forms`和从`django.contrib.auth.forms`导入`AuthenticationForm`，然后我们创建另一个类，也称为`AuthenticationForm`，它继承自 Django 的`AuthenticationForm`。 然后我们定义两个属性，用户名和密码。 我们将用户名定义为`CharField`的一个实例，并在其构造函数中传递一些关键字参数。 它们是：

+   `max_length`，顾名思义，限制字符串的大小为`50`个字符。

+   我们还使用了`widget`参数，指定了如何在页面上呈现此属性。在这种情况下，我们希望将其呈现为输入文本元素，因此我们传递了一个`TextInput`实例。可以向`widget`传递一些选项；在我们的情况下，这里我们传递了`'class'`，这是 CSS 类和占位符。

当模板引擎在页面上呈现此属性时，所有这些选项都将被使用。

我们在这里定义的第二个属性是密码。我们还将其定义为`CharField`，而不是传递`max_length`，这次我们将标签设置为`'Password'`。我们将`widget`设置为`PasswordInput`，这样模板引擎将在页面上将字段呈现为类型等于密码的输入，并且最后，我们为此字段类和占位符定义了相同的设置。

现在我们可以开始注册新的登录和注销 URL。打开文件`gamestore/main/urls.py`。首先，我们将添加一些`import`语句：

```py
from django.contrib.auth.views import login
from django.contrib.auth.views import logout
from .forms import AuthenticationForm
```

在`import`语句之后，我们可以开始注册身份验证 URL。在`urlpattens`列表的末尾，添加以下代码：

```py
  path(r'accounts/login/', login, {
      'template_name': 'login.html',
      'authentication_form': AuthenticationForm
  }, name='login'),
```

因此，在这里我们创建了一个新的 URL，`'accounts/login'`，当请求这个 URL 时，视图函数`login`将被执行。路径函数的第三个参数是一个带有一些选项的字典，`template_name`指定了浏览到底层 URL 时将呈现在页面上的模板。我们还使用`AuthenticationForm`值定义了`authetication_form`。最后，我们将关键字参数`name`设置为`login`；为这个 URL 命名在需要创建此 URL 的链接时非常有帮助，也提高了可维护性，因为 URL 本身的更改不会要求模板的更改，因为模板通过名称引用 URL。

现在登录已经就位，让我们添加注销 URL：

```py
  path(r'accounts/logout/', logout, {
      'next_page': '/'
  }, name='logout'),
```

与登录 URL 类似，在注销 URL 中，我们使用路径函数首先传递 URL 本身(`accounts/logout`)；我们传递了从 Django 内置认证视图中导入的函数 logout，并且作为一个选项，我们将`next_page`设置为`/`。这意味着当用户注销时，我们将用户重定向到应用程序的根页面。最后，我们还将 URL 命名为 logout。

很好。现在是时候添加模板了。我们要添加的第一个模板是登录模板。在`gamestore/templates/`下创建一个名为`login.html`的文件，内容如下：

```py
{% extends 'base.html' %}

{% block 'content' %}

<div>
  <form action="." method="post" class="form-signin">

    {% csrf_token %}

    <h2 class="form-signin-heading">Login</h2>
    <label for="inputUsername" class="sr-only">User name</label>
    {{form.username}}
    <label for="inputPassword" class="sr-only">Password</label>
    {{form.password}}
    <input class="btn btn-lg btn-primary btn-block" 
        type="Submit" value="Login">
  </form>
  <div class='signin-errors-container'>
    {% if form.non_field_errors %}
    <ul class='form-errors'>
      {% for error in form.non_field_errors %}
        <li>{{ error }}</li>
      {% endfor %}
    </ul>
    {% endif %}
  </div>
</div>

{% endblock %}
```

在这个模板中，我们还扩展了基本模板，并且我们添加了登录模板的内容，其中包含在基本模板中定义的内容块。

首先，我们创建一个`form`标签，并将方法设置为`POST`。然后，我们添加`csrf_token`标签。我们添加此标签的原因是为了防止跨站点请求攻击，其中恶意站点代表当前登录用户执行请求到我们的站点。

如果您想了解更多关于这种类型的攻击，您可以访问网站[`www.owasp.org/index.php/Cross-Site_Request_Forgery_(CSRF)`](https://www.owasp.org/index.php/Cross-Site_Request_Forgery_(CSRF))。

在跨站点请求伪造标记之后，我们添加了我们需要的两个字段：用户名和密码。

然后我们有以下标记：

```py
  <div class='signin-errors-container'>
    {% if form.non_field_errors %}
    <ul class='form-errors'>
      {% for error in form.non_field_errors %}
      <li>{{ error }}</li>
      {% endfor %}
    </ul>
    {% endif %}
  </div>
```

这是我们将显示可能的身份验证错误的地方。表单对象有一个名为`non_field_error`的属性，其中包含与字段验证无关的错误。例如，如果您的用户输入了错误的用户名或密码，那么错误将被添加到`non_field_error`列表中。

我们创建一个`ul`元素（无序列表）并循环遍历`non_field_errors`列表，添加带有错误文本的`li`元素（列表项）。

我们现在已经放置了登录，并且只需要将其包含到页面中-更具体地说，是到`base.html`模板。但是，首先，我们需要创建一个小的部分模板，它将在页面上显示登录和注销链接。继续添加一个名为`_loginpartial.html`的文件到`gamestore/templates`目录，其中包含以下内容：

```py
  {% if user.is_authenticated %}
  <form id="logoutForm" action="{% url 'logout' %}" method="post"
       class="navbar-right">
      {% csrf_token %}
    <ul class="nav navbar-nav navbar-right">
      <li><span class="navbar-brand">Logged as: 
            {{ user.username }}</span></li>
      <li><a href="javascript:document.getElementById('
           logoutForm').submit()">Log off</a></li>
    </ul>

  </form>

  {% else %}

  <ul class="nav navbar-nav navbar-right">
      <li><a href="{% url 'login' %}">Log in</a></li>
  </ul>

  {% endif %}
```

这个部分模板将根据用户是否经过身份验证而呈现两种不同的内容。如果用户已经过身份验证，它将呈现注销表单。请注意，表单的操作使用了命名 URL；我们没有将其设置为`/accounts/logout`，而是设置为`{% url 'logout' %}`。Django 的 URL 标记将使用 URL 名称替换 URL。同样，我们需要添加`csrf_token`标记以防止跨站点请求伪造攻击，最后，我们定义了一个无序列表，其中有两个项目；第一项将显示文本`Logged as:`和用户的用户名，列表中的第二项将显示注销按钮。

请注意，我们在列表项元素中添加了一个锚标签，并且`href`属性中有一些 JavaScript 代码。该代码非常简单；它使用`getElementById`函数获取表单，然后调用表单的提交函数将请求提交到服务器的`/accounts/logout`。

这只是对实现的偏好；您可以轻松地跳过此 JavaScript 代码并添加提交按钮。它会产生相同的效果。

如果用户未经过身份验证，我们只显示`登录`链接。`登录`链接还使用 URL 标记，该标记将使用 URL 替换名称`login`。

太棒了！让我们将登录部分模板添加到基本模板中。打开`gamestore/templates`中的`base.html`文件，并找到无序列表，如下所示：

```py
  <ul class="nav navbar-nav">
    <li>
      <a href="/">
        <i class="fa fa-home" aria-hidden="true"></i> HOME
      </a>
    </li>    
  </ul>
```

我们将使用`include`标签添加`_loginpartial.html`模板：

```py
  {% include '_loginpartial.html' %}
```

include 标签将在标记中的此位置注入`_loginpartial.html`模板的内容。

最后一步是添加一些样式，使登录页面看起来像应用程序的其余部分一样好看。打开`gamestore/static/styles`目录中的`site.css`文件，并包含以下内容：

```py
    /* Signin page */
    /* Styling extracted from http://getbootstrap.com/examples/signin/  
    */

    .form-signin {
        max-width: 330px;
        padding: 15px;
        margin: 0 auto;
    }

    .form-signin input[type="email"] {
        margin-bottom: -1px;
    }

    .form-signin input[type="email"] border-top {
        left-radius: 0;
       right-radius: 0;
    }

    .form-signin input[type="password"] {
        margin-bottom: 10px;
    }

    .form-signin input[type="password"] border-top {
        left-radius: 0;
        right-radius: 0;
    }

    .form-signin .form-signin-heading {
      margin-bottom: 10px;
    }

    .form-signin .checkbox {
      font-weight: normal;
    }

    .form-signin .form-control {
      position: relative;
      height: auto;
      -webkit-box-sizing: border-box;
      -moz-box-sizing: border-box;
      box-sizing: border-box;
      padding: 10px;
      font-size: 16px;
    }

    .form-signin .form-control:focus {
      z-index: 2;
    }

    .signin-errors-container .form-errors {
      padding: 0;
      display: flex;
      flex-direction: column;
      list-style: none;
      align-items: center;
      color: red;
    }

    .signin-errors-container .form-errors li {
      max-width: 350px;
     }
```

# 测试登录/注销表单

在尝试此操作之前，让我们打开`gamestore/gamestore`目录中的`settings.py`文件，并在文件末尾添加以下设置：

```py
LOGIN_REDIRECT_URL = '/'
```

这将告诉 Django，在登录后，用户将被重定向到“/”。

现在我们准备测试登录和注销功能，尽管您可能在数据库中没有任何用户。但是，我们在设置 Django 项目时创建了超级用户，所以继续尝试使用该用户登录。运行命令`runserver`再次启动 Django 开发服务器：

```py
python manage.py runserver
```

浏览到`http://localhost:8000`，请注意您现在在页面的右上角有登录链接：

![](img/6d74e76d-f481-4eec-9dc9-d8ad6b061fa5.png)

如果您点击，您将被重定向到`/accounts/login`，并且将呈现我们创建的登录页面模板：

![](img/3025a9a4-9ae4-4eed-b361-54290b3e87a9.png)

首先，尝试输入错误的密码或用户名，以便我们可以验证错误消息是否正确显示：

![](img/ff28669a-804e-448a-9452-632421a24126.png)

太棒了！它有效！

现在使用超级用户登录，如果一切正常，您应该被重定向到应用程序根 URL。它说，以您的用户名登录，然后就会有一个注销链接。试一试，点击注销链接：

![](img/6490f579-a5ca-4bcb-9106-699f74da2029.png)

# 创建新用户

现在我们能够登录和注销我们的应用程序，我们需要添加另一个页面，以便用户可以在我们的应用程序上创建帐户并下订单。

在创建新帐户时，我们希望强制执行一些规则。规则是：

+   用户名字段是必需的，并且必须对我们的应用程序是唯一的

+   邮箱字段是必需的，并且必须在我们的应用程序中是唯一的

+   名字和姓氏都是必需的

+   两个密码字段都是必需的，并且它们必须匹配

如果这些规则中有任何一个没有被遵循，我们将不会创建用户账户，并且应该向用户返回一个错误。

说到这里，让我们添加一个小的辅助函数，用于验证字段是否具有数据库中已存在的值。打开`gamestore/main`目录下的`forms.py`文件。首先，我们需要导入 User 模型：

```py
from django.contrib.auth.models import User
```

然后，添加`validate_unique_user`函数：

```py
def validate_unique_user(error_message, **criteria):
    existent_user = User.objects.filter(**criteria)

    if existent_user:
        raise forms.ValidationError(error_message)
```

这个函数获取一个错误消息和关键字参数，这些参数将被用作搜索与特定值匹配的项目的条件。我们创建一个名为`existent_user`的变量，并通过传递条件来过滤用户模型。如果变量`existent_user`的值与`None`不同，这意味着我们找到了一个符合我们条件的用户。然后，我们使用传递给函数的错误消息引发一个`ValidationError`异常。

很好。现在我们可以开始添加一个包含用户在创建账户时需要填写的所有字段的表单。在`gamestore/main`目录下的`forms.py`文件中，添加以下类：

```py
class SignupForm(forms.Form):
    username = forms.CharField(
       max_length=10,
       widget=forms.TextInput({
           'class': 'form-control',
           'placeholder': 'First name'
  })
    )

    first_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput({
            'class': 'form-control',
            'placeholder': 'First name'
  })
    )

    last_name = forms.CharField(
        max_length=200,
        widget=forms.TextInput({
            'class': 'form-control',
            'placeholder': 'Last name'
  })
    )

    email = forms.CharField(
        max_length=200,
        widget=forms.TextInput({
            'class': 'form-control',
            'placeholder': 'Email'
  })
    )

    password = forms.CharField(
        min_length=6,
        max_length=10,
        widget=forms.PasswordInput({
           'class': 'form-control',
           'placeholder': 'Password'
  })
    )

    repeat_password = forms.CharField(
        min_length=6,
        max_length=10,
        widget=forms.PasswordInput({
            'class': 'form-control',
            'placeholder': 'Repeat password'
  })
    )
```

因此，我们首先创建一个名为`SignupForm`的类，它将继承自`Form`，我们为创建新账户所需的每个字段定义一个属性，然后添加一个用户名、名字和姓氏、一个电子邮件，然后两个密码字段。请注意，在密码字段中，我们将密码的最小和最大长度分别设置为`6`和`10`。

在同一个类`SignupForm`中，让我们添加一个名为`clean_username`的方法：

```py
  def clean_username(self):
      username = self.cleaned_data['username']

      validate_unique_user(
         error_message='* Username already in use',
          username=username)

      return username
```

这个方法的名称中的前缀`clean`将使 Django 在解析字段的发布数据时自动调用此方法；在这种情况下，它将在解析字段用户名时执行。

所以，我们获取用户名的值，然后调用`validate_unique_user`方法，传递一个默认的错误消息和一个关键字参数用户名，这将被用作过滤条件。

我们需要验证唯一性的另一个字段是电子邮件 ID，因此让我们实现`clean_email`方法，如下所示：

```py
  def clean_email(self):
      email = self.cleaned_data['email']

      validate_unique_user(
         error_message='* Email already in use',
         email=email)

      return email
```

这基本上与清理用户名相同。首先，我们从请求中获取电子邮件并将其传递给`validate_unique_user`函数。第一个参数是错误消息，第二个参数是将用作过滤条件的电子邮件。

我们为创建账户页面定义的另一个规则是密码和（重复）密码字段必须匹配，否则将向用户显示错误。因此，让我们添加相同的并实现`clean`方法，但这次我们要验证`repeat_password`字段而不是`password`。这样做的原因是，如果我们实现一个`clean_password`函数，在那时`repeat_password`在`cleaned_data`字典中还不可用，因为数据的解析顺序与它们在类中定义的顺序相同。因此，为了确保我们将有两个值，我们实现`clean_repeat_password`：

```py
    def clean_repeat_password(self):
      password1 = self.cleaned_data['password']
      password2 = self.cleaned_data['repeat_password']

      if password1 != password2:
         raise forms.ValidationError('* Passwords did not match')

     return password1
```

很好。所以这里我们首先定义了两个变量；`password1`，它是`password`字段的请求值，`password2`，它是`repeat_password`字段的请求值。之后，我们只是比较这些值是否不同；如果它们不同，我们引发一个`ValidationError`异常，其中包含错误消息，通知用户密码不匹配，账户将不会被创建。

# 创建用户创建的视图

有了表单和验证，我们现在可以添加处理创建新账户请求的视图。打开`gamestore/main`目录下的`views.py`文件，并首先添加一些`import`语句：

```py
from django.views.decorators.csrf import csrf_protect
from .forms import SignupForm
from django.contrib.auth.models import User
```

因为我们将收到来自`POST`请求的数据，所以最好添加跨站点请求伪造检查，因此我们需要导入`csrf_protect`装饰器。

我们还导入了刚刚创建的`SignupForm`，这样我们就可以将其传递给视图或用它来解析请求数据。最后，我们导入了`User`模型。

所以，让我们创建`signup`函数：

```py
@csrf_protect def signup(request):

    if request.method == 'POST':

        form = SignupForm(request.POST)

        if form.is_valid():
            user = User.objects.create_user(
                username=form.cleaned_data['username'],
                first_name=form.cleaned_data['first_name'],
                last_name=form.cleaned_data['last_name'],
                email=form.cleaned_data['email'],
                password=form.cleaned_data['password']
            )
            user.save()

            return render(request, 
 'main/create_account_success.html', {})

    else:
        form = SignupForm()

 return render(request, 'main/signup.html', {'form': form})
```

我们首先用`csrf_protect`装饰器装饰`signup`函数。函数首先检查请求的 HTTP 方法是否等于`POST`；在这种情况下，它将创建一个`SignupForm`的实例，将`POST`数据作为参数传递。然后我们在表单上调用`is_valid()`函数，如果表单有效，它将返回 true；否则返回 false。如果表单有效，我们将创建一个新用户并调用`save`函数，最后我们渲染`create_account_success.html`。

如果请求的`HTTP`方法是`GET`，我们所做的唯一事情就是创建一个没有参数的`SignupForm`实例。之后，我们调用`render`函数，将`request`对象作为第一个参数传递，然后是我们要渲染的模板，最后一个参数是`SignupForm`的实例。

我们将很快创建这个函数中引用的两个模板，但首先，我们需要在`gamestore/main`的`url.py`文件中创建一个新的 URL：

```py
path(r'accounts/signup/', views.signup, name='signup'),
```

这个新的 URL 可以直接添加到`urlpatterns`列表的末尾。

我们还需要创建模板。我们从`signup`模板开始；在`gamestore/main/templates/main`中创建一个名为`signup.html`的文件，内容如下：

```py
{% extends "base.html" %}

{% block "content" %}

    <div class="account-details-container">
        <h1>Signup</h1>

        <form action="{% url 'signup' %}" method="POST">
          {% csrf_token %}

          {{ form }}

          <button class="btn btn-primary">Save</button>

        </form>
    </div>

{% endblock %}
```

这个模板与我们之前创建的模板非常相似，它扩展了基本模板并向基本模板的内容块注入了一些数据。我们添加了一个带有标题文本的`h1`标签和一个动作设置为`{% url 'signup' %}`的表单，`url`标签将其更改为`/accounts/signup`，并将方法设置为`POST`。

和通常一样，在表单中，我们使用`csrf_token`标签，它将与`views`文件中的`signup`函数中的`@csrf_protect`装饰器一起工作，以防止跨站请求伪造。

然后我们只需调用`{{ form }}`，它将在这个区域中渲染整个表单，然后在字段后面添加一个提交表单的按钮。

最后，我们创建一个模板，用于显示账户已成功创建的信息。在`gamestore/main/templates/main`目录下添加一个名为`create_account_success.html`的文件，内容如下：

```py
{% extends 'base.html' %}

{% block 'content' %}

    <div class='create-account-msg-container'>

        <div class='circle'>
          <i class="fa fa-thumbs-o-up" aria-hidden="true"></i>
        </div>

        <h3>Your account have been successfully created!</h3>

        <a href="{% url 'login' %}">Click here to login</a>

    </div>

{% endblock %}
```

太棒了！为了使它看起来更好，我们将在`gamestore/static`目录中的`site.css`文件中包含一些 CSS 代码。在文件末尾添加如下内容：

```py
/* Account created page */
.create-account-msg-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 100px;
}

.create-account-msg-container .circle {
    width: 200px;
    height: 200px;
    border: solid 3px;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: 30px;
    border-radius: 50%;
}

.create-account-msg-container .fa-thumbs-o-up {
    font-size: 9em;
}

.create-account-msg-container a {
    font-size: 1.5em;
}

/* Sign up page */

.account-details-container #id_password,
.account-details-container #id_repeat_password {
    width:200px;
}

.account-details-container {
    max-width: 400px;
    padding: 15px;
    margin: 0 auto;
}

.account-details-container .btn.btn-primary {
    margin-top:20px;
}

.account-details-container label {
    margin-top: 20px;
}

.account-details-container .errorlist {
    padding-left: 10px;
    display: inline-block;
    list-style: none;
    color: red;
}
```

这就是创建用户页面的全部内容；让我们试试吧！再次启动 Django 开发服务器，并浏览到`http://localhost:8000/accounts/signup`，您应该会看到创建用户表单，如下所示：

![](img/ca34bdb6-e35e-4b9b-9425-9652ba21ef52.png)

填写所有字段后，您应该被重定向到一个确认页面，如下所示：

![](img/82ffe42c-8485-4150-8cb8-ec480a29ddd4.png)

自己进行一些测试！尝试添加无效的密码，以验证我们实现的验证是否正常工作。

# 创建游戏数据模型

好了，我们可以登录到我们的应用程序，我们可以创建新用户，我们还添加了前台模板，目前是空白的，但我们将解决这个问题。我们已经到了本章的核心；我们将开始添加代表商店中可以购买的物品的模型。

我们将在网站上拥有的游戏模型的要求是：

+   商店将销售不同游戏平台的游戏

+   首页将有一个列出精选游戏的部分

+   商店的用户应该能够转到游戏详情页面并查看有关游戏的更多信息

+   游戏应该可以通过不同的标准进行发现，例如开发者、发布商、发布日期等。

+   商店的管理员应该能够使用 Django 管理界面更改产品详情。

+   产品的图片可以更改，如果找不到，应该显示默认图片

话虽如此，让我们开始添加我们的第一个模型类。在`gamestore/main/`中打开文件`models.py`，并添加以下代码：

```py
class GamePlatform(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name
```

在这里，我们添加了`GamePlatform`类，它将代表商店中可用的游戏平台。这个类非常简单；我们只需创建一个从`Model`类继承的类，并且我们只定义了一个名为`name`的属性。`name`属性被定义为最大长度为 100 个字符的`CharField`。Django 提供了各种各样的数据类型；你可以在[`docs.djangoproject.com/en/2.0/ref/models/fields/`](https://docs.djangoproject.com/en/2.0/ref/models/fields/)上看到完整的列表。

然后我们重写了`__str__`方法。这个方法将决定`GamePlatform`的实例在被打印出来时如何显示。我重写这个方法的原因是我想在 Django 管理界面的`GamePlatform`列表中显示`GamePlatform`的名称。

我们要添加的第二个模型类是`Game`模型。在同一个文件中，添加以下代码：

```py
class Game(models.Model):
    class Meta:
        ordering = ['-promoted', 'name']

    name = models.CharField(max_length=100)

    release_year = models.IntegerField(null=True)

    developer = models.CharField(max_length=100)

    published_by = models.CharField(max_length=100)

    image = models.ImageField(
        upload_to='images/',
  default='images/placeholder.png',
  max_length=100
    )

    gameplatform = models.ForeignKey(GamePlatform,
                                     null=False,
                                     on_delete=models.CASCADE)

    highlighted = models.BooleanField(default=False)
```

与我们之前创建的模型类一样，`Game`类也继承自`Model`，我们根据规格定义了所有需要的字段。这里有一些新的需要注意的地方；`release_year`属性被定义为整数字段，并且我们设置了`null=True`属性，这意味着这个字段不是必需的。

另一个使用不同类型的属性是图片属性，它被定义为`ImageField`，这将允许我们为应用程序的管理员提供更改游戏图片的可能性。这种类型继承自`FileField`，在 Django 管理界面中，该字段将被呈现为文件选择器。`ImageFile`参数`upload_to`指定了图片将被存储的位置，默认是游戏没有图片时将呈现的默认图片。我们在这里指定的最后一个参数是`max_length`，这是图片路径的最大长度。

然后，我们定义了一个`ForeignKey`。如果你不知道它是什么，外键基本上是一个标识另一个表中行的字段。在我们的例子中，这里我们希望游戏平台与多个游戏相关联。我们传递给主键定义的一些关键字参数；首先我们传递了外键类型，`null`参数设置为`False`，这意味着这个字段是必需的，最后我们将删除规则设置为`CASCADE`，所以如果应用程序的管理员删除了一个游戏平台，该操作将级联并删除与该特定游戏平台相关联的所有游戏。

我们定义的最后一个属性是`highlighted`属性。你还记得我们的一个要求是能够突出一些产品，并且让它们出现在更显眼的区域，以便用户能够轻松找到它们吗？这个属性就是做这个的。它是一个布尔类型的属性，其默认值设置为`False`。

另一个细节，我留到最后的是：你有没有注意到我们的模型类里有一个名为`Meta`的类？这是我们可以添加关于模型的元信息的方式。在这个例子中，我们设置了一个名为`ordering`的属性，其值是一个字符串数组，其中每个项代表`Game`模型的一个属性，所以我们首先有`-highlighted`，横杠符号表示降序排列，然后我们还有名称，它将以升序排列出现。

让我们继续向类中添加更多代码：

```py
    objects = GameManager()

    def __str__(self):
      return f'{self.gameplatform.name} - {self.name}'
```

在这里，我们有两件事。首先，我们分配了一个名为`GameManager`的类的实例，我稍后会详细介绍，我们还定义了特殊方法`__str__`，它定义了当打印`Game`对象的实例时，它将显示游戏平台和一个符号破折号，后跟名称本身的名称。

在`Game`类的定义之前，让我们添加另一个名为`GameManager`的类：

```py
class GameManager(models.Manager):

    def get_highlighted(self):
        return self.filter(highlighted=True)

    def get_not_highlighted(self):
        return self.filter(highlighted=False)

    def get_by_platform(self, platform):
        return self.filter(gameplatform__name__iexact=platform)
```

在我们深入了解这个实现的细节之前，我只想说几句关于 Django 中的`Manager`对象。`Manager`是 Django 中数据库和模型类之间的接口。默认情况下，每个模型类都有一个`Manager`，可以通过属性对象访问，那么为什么要定义自己的 manager 呢？我为这个`models`类实现了一个`Manager`的原因是我想把所有关于数据库操作的代码都留在模型内部，因为这样可以使代码更清晰、更易于测试。

所以，在这里我定义了另一个类`GameManager`，它继承自`Manager`，到目前为止我们定义了三个方法——`get_highlighted`，它获取所有标记为`True`的游戏，`get_not_highlighted`，它获取所有标记为`False`的游戏，`get_by_platform`，它获取给定游戏平台的所有游戏。

关于这个类中的前两个方法：我本可以只使用过滤函数并传递一个参数，其中`highlighted`等于`True`或`False`，但正如我之前提到的，将所有这些方法放在管理器内部会更清晰。

现在我们准备创建数据库。在终端中运行以下命令：

```py
python manage.py makemigrations
```

这个命令将创建一个包含我们刚刚在模型中实现的更改的迁移文件。当创建迁移时，我们可以运行`migrate`命令，然后将更改应用到数据库：

```py
python manage.py migrate
```

太棒了！接下来，我们将创建一个模型来存储游戏的价格。

# 创建价格列表数据模型

我们希望在我们的应用程序中拥有的另一个功能是能够更改产品的价格，以及知道价格是何时添加的，最重要的是，它是何时最近更新的。为了实现这一点，我们将在`models.py`文件中的`gamestore/main/`目录中创建另一个模型类，称为`PriceList`，使用以下代码：

```py
class PriceList(models.Model):
    added_at = models.DateTimeField(auto_now_add=True)

    last_updated = models.DateTimeField(auto_now=True)

    price_per_unit = models.DecimalField(max_digits=9,
                                         decimal_places=2, 
 default=0)

    game = models.OneToOneField(
        Game,
        on_delete=models.CASCADE,
        primary_key=True)

    def __str__(self):
        return self.game.name
```

正如你在这里看到的，你有两个日期时间字段。第一个是`added_at`，它有一个属性`auto_now_add`等于`True`。它的作用是让 Django 在我们将这个价格添加到表中时自动添加当前日期。`last_update`字段是用另一个参数定义的，`auto_now`等于`True`；这告诉 Django 在每次更新发生时设置当前日期。

然后，我们有一个名为`price_per_unit`的价格字段，它被定义为一个最大为`9`位数和`2`位小数的`DecimalField`。这个字段不是必需的，它将始终默认为`0`。

接下来，我们创建一个`OneToOneField`来创建`PriceList`和`Game`对象之间的链接。我们定义当游戏被删除时，`PriceList`表中的相关行也将被删除，并将此字段定义为主键。

最后，我们重写`__str__`方法，使其返回游戏的名称。这在使用 Django 管理界面更新价格时会很有帮助。

现在我们可以再次生成迁移文件：

```py
python manage.py makemigrations
```

使用以下命令应用更改：

```py
python manage.py migrate
```

太棒了！现在我们准备开始添加视图和模板，以在页面上显示我们的游戏。

# 创建游戏列表和详细页面

创建了游戏和价格的模型之后，我们已经到达了本节的有趣部分，即创建将在页面上显示游戏的视图和模板。让我们开始吧！

所以，我们在`main/templates/main`中创建了一个名为`index.html`的模板，但我们没有在上面显示任何内容。为了使该页面更有趣，我们将添加两件事：

1.  页面顶部的一个部分，将显示我们想要突出显示的游戏。它可以是新到店的游戏，非常受欢迎的游戏，或者某个目前价格很好的游戏。

1.  在突出显示游戏的部分之后，我们将列出所有其他游戏。

我们要添加的第一个模板是一个部分视图，用于列出游戏。这个部分视图将被共享到我们想要显示游戏列表的所有模板中。这个部分视图将接收两个参数：`gameslist` 和 `highlight_games`。让我们继续添加一个名为 `games-list.html` 的文件，放在 `gamestore/main/templates/main/` 中，内容如下：

```py
{% load staticfiles %}
{% load humanize %}

<div class='game-container'>
    {% for game in gameslist %}
    {% if game.highlighted and highlight_games %}
      <div class='item-box highlighted'>
    {% else %}
      <div class='item-box'>
    {% endif %}
      <div class='item-image'>
      <img src="{% static game.image.url %}"></img>
    </div>
      <div class='item-info'>
        <h3>{{game.name}}</h3>
        <p>Release year: {{game.release_year}}</p>
        <p>Developer: {{game.developer}}</p>
        <p>Publisher: {{game.published_by}}</p>
        {% if game.pricelist.price_per_unit %}
          <p class='price'>
            Price: 
          ${{game.pricelist.price_per_unit|floatformat:2|intcomma}}
          </p>
        {% else %}
        <p class='price'>Price: Not available</p>
        {% endif %}
      </div>
     <a href="/cart/add/{{game.id}}" class="add-to-cart btn
 btn-primary">
       <i class="fa fa-shopping-cart" aria-hidden="true"></i>
       Add to cart
     </a>
   </div>
   {% endfor %}
</div>
```

这里需要注意的一点是，我们在页面顶部添加了 `{% load humanize %}`；这是 Django 框架内置的一组模板过滤器，我们将使用它们来正确格式化游戏价格。为了使用这些过滤器，我们需要编辑 `gamestore/gamestore` 目录中的 `settings.py` 文件，并将 `django.contrib.humanize` 添加到 `INSTALLED_APPS` 设置中。

这段代码将创建一个容器，其中包含游戏图片、详细信息和一个添加到购物车的按钮，类似于以下内容：

![](img/a90a8c0b-9daf-4a99-adb3-1a56365aec9d.png)

现在我们想要修改 `gamestore/main/templates/main` 下的 `index.html`。我们可以用以下代码替换 `index.html` 文件的整个内容：

```py
{% extends 'base.html' %}

{% block 'content' %}
  {% if highlighted_games_list %}
    <div class='panel panel-success'>
      <div class='panel-heading'>
        <h2 class='panel-title'><i class="fa fa-gamepad"
  aria-hidden="true"></i>Highlighted games</h2>
      </div>
      <div class='panel-body'>
        {% include 'main/games-list.html' with 
         gameslist=highlighted_games_list highlight_games=False%}
        {% if show_more_link_highlighted %}
        <p>
          <a href='/games-list/highlighted/'>See more items</a>
        </p>
        {% endif %}
      </div>
    </div>
  {% endif %}

  {% if games_list %}
    {% include 'main/games-list.html' with gameslist=games_list 
     highlight_games=False%}
    {% if show_more_link_games %}
      <p>
        <a href='/games-list/all/'>See all items</a>
      </p>
    {% endif %}
  {% endif %}

{% endblock %}
```

太棒了！有趣的代码是：

```py
   {% include 'main/games-list.html' with 
     gameslist=highlighted_games_list 
       highlight_games=False%}
```

正如你所看到的，我们正在包含部分视图并传递两个参数：`gameslist` 和 `highlight_games`。`gameslist` 显然是我们希望部分视图渲染的游戏列表，而 `highlight_games` 将在我们想要以不同颜色显示推广游戏时使用，以便它们可以很容易地被识别出来。在首页，`highlight_games` 参数没有被使用，但是当我们创建一个视图来列出所有游戏，不管它是否被推广，改变推广游戏的颜色可能会很有趣。

在推广游戏部分下面，我们有一个列出未推广游戏的部分，它也使用了部分视图 `games-list.html`。

前端的最后一步是包含相关的 CSS 代码，所以让我们编辑 `gamestore/static/styles/` 下的 `site.css` 文件，并添加以下代码：

```py
.game-container {
    margin-top: 10px;
    display:flex;
    flex-direction: row;
    flex-wrap: wrap;
}

.game-container .item-box {
    flex-grow: 0;
    align-self: auto;
    width:339px;
    margin: 0px 10px 20px 10px;
    border: 1px solid #aba5a5;
    padding: 10px;
    background-color: #F0F0F0;
}

.game-container .item-box .add-to-cart {
    margin-top: 15px;
    float: right;
}

.game-container .item-box.highlighted {
    background-color:#d7e7f5;
}

.game-container .item-box .item-image {
    float: left;
}

.game-container .item-box .item-info {
    float: left;
    margin-left: 15px;
    width:100%;
    max-width:170px;
}

.game-container .item-box .item-info p {
    margin: 0 0 3px;
}

.game-container .item-box .item-info p.price {
    font-weight: bold;
    margin-top: 20px;
    text-transform: uppercase;
    font-size: 0.9em;
}

.game-container .item-box .item-info h3 {
    max-width: 150px;
    word-wrap: break-word;
    margin: 0px 0px 10px 0px;
}
```

现在我们需要修改 `index` 视图，所以编辑 `gamestore/main/` 中的 `views.py` 文件，并对 `index` 函数进行以下更改：

```py
def index(request):
    max_highlighted_games = 3
  max_game_list = 9    highlighted_games_list = Game.objects.get_highlighted()
    games_list = Game.objects.get_not_highlighted()

    show_more_link_promoted = highlighted_games_list.count() > 
    max_highlighted_games
    show_more_link_games = games_list.count() > max_game_list

    context = {
        'highlighted_games_list': 
         highlighted_games_list[:max_highlighted_games],
        'games_list': games_list[:max_game_list],
        'show_more_link_games': show_more_link_games,
        'show_more_link_promoted': show_more_link_promoted
    }

    return render(request, 'main/index.html', context)
```

在这里，我们首先定义了我们想要显示每个游戏类别的项目数量；对于推广游戏，将显示三款游戏，而非推广类别将最多显示九款游戏。

然后，我们获取推广和非推广游戏，并创建两个变量 `show_more_link_promoted` 和 `show_more_link_games`，如果数据库中的游戏数量超过我们之前定义的最大数量，它们将被设置为 `True`。

我们创建一个包含我们想要在模板中呈现的所有数据的上下文变量，最后，我们调用 `render` 函数，并将 `request` 传递给我们想要呈现的模板，以及上下文。

因为我们使用了 `Game` 模型，我们需要导入它：

```py
from .models import Game
```

现在我们准备在页面上看到结果了，但首先，我们需要创建一些游戏。为此，我们首先需要在管理员中注册模型。要做到这一点，编辑 `admin.py` 文件，并包含以下代码：

```py
    from django.contrib import admin

    from .models import GamePlatform
    from .models import Game
    from .models import PriceList

    admin.autodiscover()

    admin.site.register(GamePlatform)
    admin.site.register(Game)
    admin.site.register(PriceList)
```

在 Django 管理站点中注册模型将允许我们添加、编辑和删除游戏、游戏平台和价格列表中的项目。因为我们将向游戏添加图片，我们需要配置 Django 应该保存我们通过管理站点上传的图片的位置。因此，让我们继续打开 `gamestore/gamestore` 目录中的 `settings.py` 文件，并在 `STATIC_DIRS` 设置的下面添加这一行：

```py
MEDIA_ROOT  = os.path.join(BASE_DIR, 'static'</span>)
```

现在，启动网站：

```py
python manage.py runserver
```

浏览到`http://localhost:8000/admin`，并使用我们创建的超级用户帐户登录。您应该会在页面上看到列出的模型：

![](img/67a5e1f4-ea27-49db-baa6-d720c8b35235.png)

如果您首先点击“游戏”平台，您将看到一个空列表。点击页面右上方的“游戏平台”行上的 ADD 按钮，将显示以下表单：

![](img/dd1567e5-3c9c-4428-b68d-5fb26bd0bef2.png)

只需输入您喜欢的任何名称，然后单击“保存”按钮以保存更改。

在添加游戏之前，我们需要找到一个默认图像，并将其放置在`gamestore/static/images/`。图像的名称应为`placeholder.png`。

我们构建的布局将更适合尺寸为 130x180 的图像。为了简化，当我创建原型时，我不想花太多时间寻找完美的图像，我会去网站[`placeholder.com/`](https://placeholder.com/)。在这里，您可以构建任何尺寸的占位图像。为了获得我们应用程序的正确尺寸，您可以直接转到[`via.placeholder.com/130x180`](http://via.placeholder.com/130x180)。

当您放置默认图像后，您可以开始添加游戏，方法与添加游戏平台相同，只需重复该过程多次以添加一些设置为推广的游戏。

添加游戏后，再次访问网站，您应该会在首页上看到游戏列表，如下所示：

![](img/8e40b7bf-7333-4410-926c-c4b2bc5af4a3.png)

在我的项目中，我添加了四个推广游戏。请注意，因为我们在第一页上只显示了三个推广游戏，所以我们呈现了“查看更多项目”链接。

# 添加列表游戏视图

由于我们没有在第一页上显示所有项目，因此我们需要构建页面，如果用户点击“查看更多项目”链接，将显示所有项目。这应该相当简单，因为我们已经有一个列出游戏的部分视图。

让我们在`main`应用的`url.py`文件中创建另外两个 URL，并将它们添加到`urlpatterns`列表中：

```py
    path(r'games-list/highlighted/', views.show_highlighted_games),
    path(r'games-list/all/', views.show_all_games),
```

完美！现在我们需要添加一个模板来列出所有游戏。在`gamestore/main/templates/main`下创建一个名为`all_games.html`的文件，内容如下：

```py
{% extends 'base.html' %}

{% block 'content' %}

 <h2>Highlighted games</h2>
 <hr/>

 {% if games %}
   {% include 'main/games-list.html' with gameslist=games
        highlight_promoted=False%}
   {% else %}
   <div class='empty-game-list'>
   <h3>There's no promoted games available at the moment</h3>
  </div>
 {% endif %}

 {% endblock %}
```

在同一文件夹中再添加一个名为`highlighted.html`的文件：

```py
{% extends 'base.html' %}

{% block 'content' %}

<h2>All games</h2>
<hr/>

{% if games %}
  {% include 'main/games-list.html' with gameslist=games
    highlight_games=True%}
  {% else %}
  <div class='empty-game-list'>
    <h3>There's no promoted games available at the moment</h3>
  </div>
{% endif %}

{% endblock %}
```

这里没有我们以前没有见过的东西。这个模板将接收一个游戏列表，并将其传递给`games-list.html`部分视图，该视图将为我们渲染游戏。这里有一个`if`语句，检查列表中是否有游戏。如果列表为空，它将显示消息，说明目前没有可用的游戏。否则，它将呈现内容。

现在最后一件事是添加视图。打开`gamestore/main/`下的`views.py`文件，并添加以下两个函数：

```py
def show_all_games(request):
    games = Game.objects.all()

    context = {'games': games}

    return render(request, 'main/all_games.html', context)

def show_highlighted_games(request):
    games = Game.objects.get_highlighted()

    context = {'games': games}

    return render(request, 'main/highlighted.html', context)
```

这些功能非常相似；一个获取所有游戏的列表，另一个获取仅推广游戏的列表

让我们再次打开应用程序。由于数据库中有更多的推广项目，让我们点击页面上突出显示游戏部分的“查看更多项目”链接。您应该会进入以下页面：

![](img/28d8b456-325c-42cc-86eb-94cfb4fb8e20.png)

完美！它的工作就像预期的那样。

接下来，我们将为按钮添加功能，以便将这些项目添加到购物车中。

# 创建购物车模型

看起来现在我们有一个正在运行的应用程序，我们可以显示我们的游戏，但这里有一个大问题。你能猜到是什么吗？好吧，这个问题并不难，我在本节的标题中已经给出了答案。无论如何，我们的用户无法购买游戏，我们需要实现一个购物车，这样我们就可以开始让我们的用户开心了！

现在，您可以以许多种方式在应用程序上实现购物车，但我们将简单地将购物车项目保存在数据库中，而不是基于用户会话进行实现。

购物车的要求如下：

+   用户可以添加任意数量的商品

+   用户应该能够更改购物车中的商品；例如，他们应该能够更改商品的数量

+   应该可以删除商品

+   应该有一个清空购物车的选项

+   所有数据都应该经过验证

+   如果拥有该购物车的用户被删除，购物车及其商品也应该被删除

说到这里，打开`gamestore/main`目录下的`models.py`文件，让我们添加我们的第一个类：

```py
class ShoppingCartManager(models.Manager):

    def get_by_id(self, id):
        return self.get(pk=id)

    def get_by_user(self, user):
        return self.get(user_id=user.id)

    def create_cart(self, user):
        new_cart = self.create(user=user)
        return new_cart
```

和我们为`Game`对象创建自定义`Manager`一样，我们也将为`ShoppingCart`创建一个`Manager`。我们将添加三个方法。第一个是`get_by_id`，顾名思义，根据 ID 检索购物车。第二个方法是`get_by_user`，它接收`django.contrib.auth.models.User`的实例作为参数，并将返回给定用户实例的购物车。最后一个方法是`create_cart`；当用户创建账户时将调用此方法。

现在我们有了需要的方法的管理器，让我们添加`ShoppingCart`类：

```py
class ShoppingCart(models.Model):
    user = models.ForeignKey(User,
                             null=False,
                             on_delete=models.CASCADE)

    objects = ShoppingCartManager()

    def __str__(self):
        return f'{self.user.username}\'s shopping cart'
```

这个类非常简单。和以往一样，我们从`Model`继承，并为类型`User`定义一个外键。这个外键是必需的，如果用户被删除，购物车也会被删除。

在外键之后，我们将我们自定义的`Manager`分配给对象的属性，并且我们还实现了特殊方法`__str__`，这样在 Django 管理界面中购物车会以更好的方式显示。

接下来，让我们为`ShoppingCartItem`模型添加一个管理类，如下所示：

```py
class ShoppingCartItemManager(models.Manager):

    def get_items(self, cart):
        return self.filter(cart_id=cart.id)
```

在这里，我们只定义了一个方法，名为`get_items`，它接收一个购物车对象，并返回底层购物车的商品列表。在`Manager`类之后，我们可以创建模型：

```py
class ShoppingCartItem(models.Model):
    quantity = models.IntegerField(null=False)

    price_per_unit = models.DecimalField(max_digits=9,
                                         decimal_places=2,
                                         default=0)

    cart = models.ForeignKey(ShoppingCart,
                             null=False,
                             on_delete=models.CASCADE)
    game = models.ForeignKey(Game,
                             null=False,
                             on_delete=models.CASCADE)

    objects = ShoppingCartItemManager()
```

我们首先定义了两个属性：数量，这是一个整数值，和每件商品的价格，这是一个十进制值。在这个模型中我们也有`price_per_item`，因为当用户将商品添加到购物车时，如果管理员更改了产品的价格，我们不希望已经添加到购物车的商品的价格发生变化。价格应该与用户首次将产品添加到购物车时的价格相同。

如果用户完全删除商品并重新添加，新的价格应该得到反映。在这两个属性之后，我们定义了两个外键，一个是类型`ShoppingCart`，另一个是`Game`。

最后，我们将`ShoppingCartItemManager`设置为对象的属性。

我们还需要导入 User 模型：

```py
from django.contrib.auth.models import User
```

在我们尝试验证一切是否正常工作之前，我们应该创建并应用迁移。在终端上运行以下命令：

```py
python manage.py makemigrations
```

和以前一样，我们需要运行迁移命令来将迁移应用到数据库：

```py
python manage.py migrate
```

# 创建购物车表单

我们现在已经有了模型。让我们添加一个新的表单，用于在页面上显示购物车数据进行编辑。打开`gamestore/main/`目录下的`forms.py`文件，在文件末尾添加以下代码：

```py
    ShoppingCartFormSet = inlineformset_factory(
      ShoppingCart,
      ShoppingCartItem,
      fields=('quantity', 'price_per_unit'),
      extra=0,
      widgets={
          'quantity': forms.TextInput({
             'class': 'form-control quantity',
          }),
          'price_per_unit': forms.HiddenInput()
      }
    )
```

在这里，我们使用`inlineformset_factory`函数创建一个内联`formset`。内联`formset`适用于当我们想通过外键与相关对象一起工作时。在我们这里非常方便；我们有一个与`ShoppingCartItem`相关的`ShoppingCart`模型。

因此，我们向`inlineformset_factory`函数传递了一些参数。首先是父模型（`ShoppingCart`），然后是模型（`ShoppingCartItems`）。因为在购物车中我们只想编辑数量并从购物车中移除商品，所以我们添加了一个包含我们想要在页面上呈现的`ShoppingCartItem`字段的元组——在这种情况下是`quantity`和`price_per_unit`。下一个参数`extra`指定表单是否应在表单上呈现任何空的额外行；在我们的情况下，我们不需要这样做，因为我们不希望将额外的商品添加到购物车视图中。

在最后一个参数`widgets`中，我们可以指定表单中字段的呈现方式。数量字段将呈现为文本输入，我们不希望`price_per_unit`可见，所以我们将其定义为隐藏输入，这样当我们将表单提交到服务器时，它会被发送回服务器。

最后，在同一个文件中，让我们添加一些必要的导入：

```py
from django.forms import inlineformset_factory
from .models import ShoppingCartItem
from .models import ShoppingCart
```

打开`views.py`文件，让我们添加一个基于类的视图。首先，我们需要添加一些导入语句：

```py
from django.views.generic.edit import UpdateView
from django.http import HttpResponseRedirect
from django.urls import reverse_lazy
from django.db.models import Sum, F, DecimalField

from .models import ShoppingCart
from .models import ShoppingCartItem
from .forms import ShoppingCartFormSet
```

然后，我们可以创建如下的类：

```py
class ShoppingCartEditView(UpdateView):
    model = ShoppingCart
    form_class = ShoppingCartFormSet
    template_name = 'main/cart.html'    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        items = ShoppingCartItem.objects.get_items(self.object)

        context['is_cart_empty'] = (items.count() == 0)

        order = items.aggregate(
            total_order=Sum(F('price_per_unit') * F('quantity'),
                            output_field=DecimalField())
        )

        context['total_order'] = order['total_order']

        return context

    def get_object(self):
        try:
            return ShoppingCart.objects.get_by_user(self.request.user)
        except ShoppingCart.DoesNotExist:
            new_cart = ShoppingCart.objects.create_cart(self.request.user)
            new_cart.save()
            return new_cart

 def form_valid(self, form):
        form.save()
        return HttpResponseRedirect(reverse_lazy('user-cart'))
```

这与我们迄今为止创建的视图略有不同，因为这是一个从`UpdateView`继承的基于类的视图。实际上，在 Django 中，视图是可调用对象，当使用类而不是函数时，我们可以利用继承和混合。在我们的情况下，我们使用`UpdateView`，因为它是一个用于显示将编辑现有对象的表单的视图。

这个类视图首先定义了一些属性，比如模型，这是我们将在表单中编辑的模型。`form_class`是用于编辑数据的表单。最后，我们有将用于呈现表单的模板。

我们重写了`get_context_data`，因为我们在表单上下文中包含了一些额外的数据。因此，首先我们调用基类上的`get_context_data`来构建上下文，然后我们获取当前购物车的商品列表，以便确定购物车是否为空。我们将这个值设置为上下文项`is_cart_empty`，可以从模板中访问。

之后，我们想要计算当前购物车中商品的总价值。为此，我们需要首先通过（价格*数量）来计算每件商品的总价，然后对结果进行求和。在 Django 中，可以对`QuerySet`的值进行聚合；我们已经有了包含购物车中商品列表的`QuerySet`，所以我们只需要使用`aggregate`函数。在我们的情况下，我们向`aggregate`函数传递了两个参数。首先，我们得到字段`price_per_unit`乘以数量的总和，并将结果存储在一个名为`total_order`的属性中。`aggregate`函数的第二个参数定义了输出数据类型，我们希望它是一个十进制值。

当我们得到聚合的结果时，我们在上下文字典中创建了一个名为`total_order`的新项，并将结果赋给它。最后，我们返回上下文。

我们还重写了`get_object`方法。在这个方法中，我们尝试获取请求用户的购物车。如果购物车不存在，将引发一个`ShoppingCart.DoesNotExist`异常。在这种情况下，我们为用户创建一个购物车并返回它。

最后，我们还实现了`form_valid`方法，它只保存表单并将用户重定向回购物车页面。

# 创建购物车视图

现在是时候创建购物车视图了。这个视图将呈现我们刚刚创建的表单，用户应该能够更改购物车中每件商品的数量，以及移除商品。如果购物车为空，我们应该显示一条消息，说明购物车是空的。

在添加视图之前，让我们继续打开`gamestore/main/`中的`urls.py`文件，并添加以下 URL：

```py
 path(r'cart/', views.ShoppingCartEditView.as_view(), name='user-
  cart'),
```

在这里，我们定义了一个新的 URL，`'cart/'`，当访问时，它将执行基于类的视图`ShoppingCartEditView`。我们还为 URL 定义了一个名称，以简化操作。

我们将在`gamestore/main/templates/main`中创建一个名为`cart.html`的新文件，内容如下：

```py
{% extends 'base.html' %}

{% block 'content' %}

{% load humanize %}

<div class='cart-details'>

<h3>{{ shoppingcart}}</h3>

{% if is_cart_empty %}

<h2>Your shopping cart is empty</h2>

{% else %}

<form action='' method='POST'>

  {% csrf_token %}

  {{ form.management_form }}

 <button class='btn btn-success'>
  <i class="fa fa-refresh" aria-hidden="true"></i>
     Updated cart
</button>
  <hr/>
  <table class="table table-striped">
  <thead>
    <tr>
      <th scope="col">Game</th>
      <th scope="col">Quantity</th>
      <th scope="col">Price per unit</th>
      <th scope="col">Options</th>
    </tr>
  </thead>
  <tbody>
   {% for item_form in form %}
   <tr>
     <td>{{item_form.instance.game.name}}</td>
     <td class=
  "{% if item_form.quantity.errors %}has-errors{% endif%}">
     {{item_form.quantity}}
   </td>
   <td>${{item_form.instance.price_per_unit|
            floatformat:2|intcomma}}</td>
   <td>{{item_form.DELETE}} Remove item</td>
   {% for hidden in item_form.hidden_fields %}
     {{ hidden }}
   {% endfor %}
  </tr>
  {% endfor %}
  <tbody>
 </table>
 </form>
<hr/>
<div class='footer'>
  <p class='total-value'>Total of your order:
     ${{total_order|floatformat:2|intcomma}}</p>
  <button class='btn btn-primary'>
     <i class="fa fa-check" aria-hidden="true"></i>
        SEND ORDER
  </button>
</div>
  {% endif %}
</div>
{% endblock %}
```

模板非常简单；我们只需循环遍历表单并渲染每一个。这里需要注意的一点是我们在模板开头加载了`humanize`。

`humanize`是一组模板过滤器，我们可以在模板中使用它来格式化数据。

我们使用`humanize`中的`intcomma`过滤器来格式化购物车中所有商品的总和。`intcomma`过滤器将把整数或浮点值转换为字符串，并在每三位数字后添加一个逗号。

您可以在新视图上尝试它。但是，购物车将是空的，不会显示任何数据。接下来，我们将添加包含商品的功能。

# 将商品添加到购物车

我们即将完成购物车。现在我们将实现一个视图，将商品包含在购物车中。

我们需要做的第一件事是创建一个新的 URL。打开`gamestore/main/`目录中的`url.py`文件，并将此 URL 添加到`urlpatterns`列表中：

```py
   path(r'cart/add/<int:game_id>/', views.add_to_cart),
```

完美。在此 URL 中，我们可以传递游戏 ID，并且它将执行一个名为`add_to_cart`的视图。让我们添加这个新视图。在`gamestore/main`中打开`views.py`文件。首先，我们添加导入语句，如下所示：

```py
from decimal import Decimal
from django.shortcuts import get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
```

现在，我们需要一种方法来知道特定商品是否已经添加到购物车中，因此我们转到`gametore/main`中的`models.py`，并向`ShoppingCartItemManager`类添加一个新方法：

```py
def get_existing_item(self, cart, game):
    try:
        return self.get(cart_id=cart.id,
                        game_id=game.id)
    except ShoppingCartItem.DoesNotExist:
        return None
```

`get_existing_item`使用`cart id`和`game id`作为条件搜索`ShoppingCartItem`对象。如果在购物车中找不到该商品，则返回`None`；否则，它将返回购物车商品。

现在我们将视图添加到`views.py`文件中：

```py
@login_required def add_to_cart(request, game_id):
    game = get_object_or_404(Game, pk=game_id)
    cart = ShoppingCart.objects.get_by_user(request.user)

    existing_item = ShoppingCartItem.objects.get_existing_item(cart, 
    game)

    if existing_item is None:

        price = (Decimal(0)
            if not hasattr(game, 'pricelist')
            else game.pricelist.price_per_unit)

        new_item = ShoppingCartItem(
            game=game,
            quantity=1,
            price_per_unit=price,
            cart=cart
        )
        new_item.save()
    else:
        existing_item.quantity = F('quantity') + 1
  existing_item.save()

        messages.add_message(
             request,
             messages.INFO,
             f'The game {game.name} has been added to your cart.')

        return HttpResponseRedirect(reverse_lazy('user-cart'))
```

此函数获取请求和游戏 ID，然后我们开始获取游戏和当前用户的购物车。然后我们将购物车和游戏传递给我们刚刚创建的`get_existing`函数。如果我们在购物车中没有特定的商品，我们就创建一个新的`ShoppingCartItem`；否则，我们只是更新数量并保存。

我们还添加了一条消息，通知用户该商品已添加到购物车中。

最后，我们将用户重定向到购物车页面。

最后一步，让我们打开`gamestore/static/styles`中的`site.css`文件，并为我们的购物车视图添加样式：

```py
.cart-details h3 {
    margin-bottom: 40px;
}

.cart-details .table tbody tr td:nth-child(2) {
    width: 10%;
}

.cart-details .table tbody tr td:nth-child(3) {
    width: 25%;
}

.cart-details .table tbody tr td:nth-child(4) {
    width: 20%;
}

.has-errors input:focus {
    border-color: red;
    box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(255,0,0,1);
    webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(255,0,0,1);
}

.has-errors input {
    color: red;
    border-color: red;
}

.cart-details .footer {
    display:flex;
    justify-content: space-between;
}

.cart-details .footer .total-value {
    font-size: 1.4em;
    font-weight: bold;
    margin-left: 10px;
}
```

在尝试这个之前，我们需要在顶部菜单中添加到购物车视图的链接。在`gamestore/templates`中打开`base.html`文件，找到我们包含`_loginpartial.html`文件的位置，并在其之前包含以下代码：

```py
{% if user.is_authenticated%}
<li>
  <a href="/cart/">
    <i class="fa fa-shopping-cart"
  aria-hidden="true"></i> CART
  </a>
</li>
{% endif %}
```

现在我们应该准备好测试它了。转到第一页，尝试向购物车中添加一些游戏。您应该会被重定向到购物车页面：

![](img/88194946-e679-46f9-a81a-b3b782e05424.png)

# 总结

这是一个漫长的旅程，在本章中我们涵盖了很多内容。在本章中，您已经看到使用 Django 构建应用是多么容易。这个框架真的很符合“完美主义者的截止日期”这句话。

您已经学会了如何创建一个新的 Django 项目和应用程序，并简要介绍了 Django 在我们启动新项目时为我们生成的样板代码。我们学会了如何创建模型并使用迁移来对数据库应用更改。

Django 表单也是本章我们涵盖的一个主题，您应该能够为您的项目创建复杂的表单。

作为奖励，我们学会了如何安装和使用**NodeJS 版本管理器**（**NVM**）来安装 Node.js，以便使用 npm 安装项目依赖项。

在第五章中，*使用微服务构建 Web Messenger*，我们将扩展此应用程序，并创建将处理商店库存的服务。
