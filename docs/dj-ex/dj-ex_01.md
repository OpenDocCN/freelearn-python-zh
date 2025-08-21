# 第一章：创建一个博客应用

在本书中，你将学习如何创建完整的，可用于生产环境的 Django 项目。如果你还没有安装 Django，你将在本章的第一部分学习如何安装。本章将会涉及如何使用 Django 创建一个简单的博客应用。本章的目的是对框架如何工作有一个基本概念，理解不同组件之间如何交互，并教你使用基本功能创建 Django 项目。本章会引导你创建一个完整项目，但不会阐述所有细节。不同框架组件的细节会在本书接下来的章节中介绍。

本章将会涉及以下知识点：

- 安装 Django，并创建第一个项目
- 设计模型（model），并生成模型迁移（model migration）
- 为模型创建一个管理站点
- 使用`QuerySet`和管理器（manager）
- 创建视图（view），模板（template）和 URL
- 为列表视图中添加分页
- 使用基于类的视图

## 1.1 安装 Django

如果你已经安装了 Django，可以略过本节，直接跳到`创建第一个项目`。Django 是一个 Python 包，可以在任何 Python 环境中安装。如果你还没有安装 Django，这是为本地开发安装 Django 的快速指南。

Django 可以在 Python 2.7 或 3 版本中工作。本书的例子中，我们使用 Python 3。如果你使用 Linux 或 Mac OS X，你可能已经安装了 Python。如果不确定是否安装，你可以在终端里输入`python`。如果看到类似下面的输出，表示已经安装了 Python：

```py
Python 3.5.2 (v3.5.2:4def2a2901a5, Jun 26 2016, 10:47:25)
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

如果你安装的 Python 版本低于 3，或者没有安装，请从[官方网站](http://www.python.org/download/)下载并安装。

> **译者注：**如果你的电脑上同时安装了 Python 2 和 Python 3，需要输入`python3`，而不是`python`。

因为将会使用 Python 3，所以不需要安装数据库。该 Python 版本内置了 SQLite 数据库。SQLite 是一个轻量级数据库，可以在 Django 开发中使用。如果计划在生产环境部署应用，应该使用更高级的数据库，比如 PostgreSQL，MySQL，或者 Oracle。在[这里](https://docs.djangoproject.com/en/1.11/topics/install/#database-installation)获取更多关于如何在 Django 中使用数据库的信息。

### 1.1.1 创建独立的 Python 环境

推荐你使用`virtualenv`创建独立的 Python 环境，这样就可以为不同项目使用不同的包版本，比在系统范围内安装 Python 包更实用。使用`virtualenv`的另一个好处是，安装 Python 包时不需要管理员权限。在终端运行下面的命令来安装`virtualenv`：

```py
pip install virtualenv
```

> **译者注**：如果电脑上同时安装了 Python 2 和 Python 3，需要使用`pip3`。

安装`virtualenv`后，使用下面的命令创建一个独立的 Python 环境：

```py
virtualenv my_env
```

这将会创建一个包括 Python 环境的`my_env/`目录。当虚拟环境激活时，安装的所有 Python 库都会在`my_env/lib/python3.5/site-packages`目录中。

如果电脑上同时安装了 Python 2 和 Python 3，你需要告诉`virtualenv`使用后者。使用以下命令定位 Python 3 的安装路径，然后使用该路径创建虚拟环境：

```py
zenx$ which python3
/Library/Frameworks/Python.framework/Versions/3.5/bin/python3
zenx$ virtualenv my_env -p
/Library/Frameworks/Python.framework/Versions/3.5/bin/python3
```

运行下面的命令激活虚拟环境：

```py
source my_env/bin/activate
```

终端的提示中，括号内是激活的虚拟环境的名称，比如：

```py
(my_env)laptop:~ zenx$
```

你可以使用`deactive`命令随时停用虚拟环境。

你可以在[这里](https://virtualenv.pypa.io/en/latest/)找到更多关于`virtualenv`的信息。

在`virtualenv`之上，你可以使用`virtualenvwrapper`。该工具进行了一些封装，更容易创建和管理虚拟环境。你可以在[这里](http://virtualenvwrapper.readthedocs.io/en/latest/)下载。

### 1.1.2 使用 pip 安装 Django

推荐使用`pip`安装 Django。Python 3.5 中已经安装了`pip`。在终端运行以下命令安装 Django：

```py
pip install Django
```

Django 将会安装在虚拟环境的`site-packages`目录中。

检查一下 Django 是否安装成功。在终端中运行`python`，然后导入 Django，检查版本：

```py
>>> import django
>>> django.VERSION
(1, 11, 0, 'final', 1)
```

如果得到类似以上的输出，表示 Django 已经安装成功。

有多种方式可以安装 Django，访问[这里](https://docs.djangoproject.com/en/1.11/topics/install/)查看完成的安装指南。

## 1.2 创建第一个项目

我们的第一个 Django 项目是一个完整的博客网站。Django 提供了一个命令，可以很容易创建一个初始的项目文件结构。在终端运行以下命令：

```py
django-admin startproject mysite
```

这会创建一个名为`mysite`的 Django 项目。

让我们看一下生成的项目结构：

```py
mysite/
  manage.py
  mysite/
    __init__.py
    settings.py
    urls.py
    wsgi.py
```

以下是这些文件的基本介绍：

- `manage.py`：用于与项目交互的命令行工具。它对`django-admin.py`工具进行了简单的封装。你不需要编辑该文件。
- `mysite/`：你的项目目录，由以下文件组成：
 * `__init__.py`：一个空文件，告诉 Python，把`mysite`目录当做一个 Python 模块。
 * `settings.py`：用于设置和配置你的项目。包括初始的默认设置。
 * `urls.py`：放置 URL 模式（pattern）的地方。这里定义的每个 URL 对应一个视图。
 * `wsgi.py`：配置你的项目，让它作为一个 WSGI 应用运行。

生成的`settings.py`文件中包括：使用 SQLite 数据库的基本配置，以及默认添加到项目中的 Django 应用。我们需要为这些初始应用在数据库中创建表。

打开终端，运行以下命令：

```py
cd mysite
python manage.py migrate
```

你会看到以类似这样结尾的输出：

```py
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
  Applying sessions.0001_initial... OK
```

初始应用的数据库表已经创建成功。一会你会学习`migrate`管理命令。

### 1.2.1 运行开发服务器

Django 自带一个轻量级的 web 服务器，可以快速运行你的代码，不需要花时间配置生产服务器。当你运行 Django 的开发服务器时，它会一直监测代码的变化，自动重新载入，不需要修改代码后，手动重启服务器。但是，有些操作它可能无法监测，比如在项目中添加新文件，这种情况下，你需要手动重启服务器。

从项目的根目录下输入以下命令，启动开发服务器：

```py
python manage.py runserver
```

你会看到类似这样的输出：

```py
Performing system checks...

System check identified no issues (0 silenced).
April 21, 2017 - 08:01:00
Django version 1.11, using settings 'kaoshao.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.

```

在浏览器中打开`http://127.0.0.1:8000/`。你应该可以看到一个页面，告诉你项目已经成功运行，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE1.1.png) 

你可以让 Django 在自定义的 host 和端口运行开发服务器，或者载入另一个配置文件来运行项目。例如，你可以这样运行`manage.py`命令：

```py
python manage.py runserver 127.0.0.1:8001 --settings=mysite.settings
```

这可以用来处理多个环境需要不同的配置。记住，该服务器只能用于开发，不适合生产环境使用。要在生产环境发布 Django，你需要使用真正的 web 服务器（比如 Apache，Gunicorn，或者 uWSGI）作为 Web Server Gateway Interface（WSGI）。你可以在[这里](https://docs.djangoproject.com/en/1.11/howto/deployment/wsgi/)找到更多关于如何使用不同的 web 服务器发布 Django 的信息。

### 1.2.2 项目设置

让我们打开`settings.py`文件，看下项目的配置。该文件中有很多 Django 的配置，但这只是所有 Django 配置中的一部分。你可以在[这里](https://docs.djangoproject.com/en/1.11/ref/settings/)查看所有配置和它们的默认值。

以下配置非常值得一看：

- `DEBUG`：一个布尔值，用于开启或关闭项目的调试模式。如果设置为`True`，当应用抛出一个未捕获的异常时，Django 会显示详细的错误页面。当你部署到生产环境时，记得设置为`False`。在生产环境部署站点时，永远不要启用`DEBUG`，否则会暴露项目的敏感数据。
- `ALLOWED_HOSTS`：当开启调试模式，或者运行测试时，它不会起作用。一旦你准备把站点部署到生产环境，并设置`DEBUG`为`False`，就需要将你的域或 host 添加到该设置中，以便它可以提供该 Django 站点。
- `INSTALLED_APPS`：你需要在所有项目中编辑该设置。该设置告诉 Django，该站点激活了哪些应用。默认情况下，Django 包括以下应用：
 * `django.contrib.admin`：管理站点。
 * `django.contrib.auth`：权限框架。
 * `django.contrib.contenttypes`：内容类型的框架。
 * `django.contrib.sessions`：会话框架。
 * `django.contrib.messages`：消息框架。
 * `django.contrib.staticfiles`：管理静态文件的框架。
- `MIDDLEWARE`：一个包括被执行的中间件的元组。
- `ROOT_URLCONF`：指明哪个 Python 模块定义了应用的根 URL 模式。
- `DATABASES`：一个字典，其中包括了所有在项目中使用的数据库的设置。必须有一个`default`数据库。默认使用 SQLite3 数据库。
- `LANGUAGE_CODE`：定义该 Django 站点的默认语言编码。

> **译者注：**Django 1.9 和之前的版本是`MIDDLEWARE`，之后的版本修改为`MIDDLEWARE`。

不用担心现在不能理解这些配置的意思。在以后的章节中你会逐渐熟悉 Django 配置。

### 1.2.3 项目和应用

在本书中，你会一次次看到术语项目（project）和应用（application）。在 Django 中，一个项目认为是一个具有一些设置的 Django 安装；一个应用是一组模型，视图，模板和 URLs。应用与框架交互，提供一些特定的功能，而且可能在多个项目中复用。你可以认为项目是你的站点，其中包括多个应用，比如博客，wiki，或者论坛，它们可以在其它项目中使用。

### 1.2.4 创建一个应用

现在，让我们创建第一个 Django 应用。我们会从头开始创建一个博客应用。在项目的根目录下，执行以下命令：

```py
python manage.py startapp blog
```

这将会创建应用的基本架构，如下所示：

```py
blog/
	__init__.py
   admin.py
   apps.py
   migrations/
       __init__.py
   models.py
   tests.py
   views.py
```

以下是这些文件：

- `admin.py`：用于注册模型，把它们包括进 Django 管理站点。是否使用 Django 管理站点是可选的。
- `apps.py`：用于放置应用配置（application configuration），可以配置应用的某些属性。
- `migrations`：该目录会包含应用的数据库迁移。迁移允许 Django 追踪模型的变化，并同步数据库。
- `models.py`：应用的数据模型。所有 Django 应用必须有一个`models.py`文件，但该文件可以为空。
- `tests.py`：用于添加应用的测试。
- `views.py`：用于存放应用的逻辑。每个视图接收一个 HTTP 请求，然后处理请求，并返回响应。

> **译者注：**从 Django 1.9 开始，`startapp`命令会创建`apps.py`文件。

## 1.3 设计博客的数据架构

我们将会开始定义博客的初始数据模型。一个模型是一个 Python 类，并继承自`django.db.models.Model`，其中每个属性表示数据库的一个字段。Django 会为`models.py`中定义的每个模型创建一张数据库表。创建模型后，Django 会提供一个实用的 API 进行数据库查询。

首先，我们定义一个`Post`模型，在`blog`应用的`models.py`文件中添加以下代码：

```py
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class Post(models.Model):
    STATUS_CHOICES = (
        ('draft', 'Draft'),
        ('published', 'Published'),
    )

    title = models.CharField(max_length=250)
    slug = models.SlugField(max_length=250, 
                            unique_for_date='publish')
    author = models.ForeignKey(User, 
                               related_name='blog_posts')
    body = models.TextField()
    publish = models.DateTimeField(default=timezone.now)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=10,
                              choices=STATUS_CHOICES,
                              default='draft')

    class Meta:
        ordering = ('-publish',)

    def __str__(self):
        return self.title
```

这是博客帖子的基础模型。让我们看看该模型定义的字段：

- `title`：这个字段是帖子的标题。该字段的类型是`CharField`，在 SQL 数据库中会转换为`VARCHAR`。
- `slug`：这个字段会在 URLs 中使用。一个别名（slug）是一个短标签，只包括字母，数字，下划线或连字符。我们将使用`slug`字段为 blog 的帖子创建漂亮的，搜索引擎友好的 URLs。我们为该字段添加了`unique_for_date`参数，所以我们可以使用帖子的日期和别名来构造帖子的 URLs。Django 不允许多个帖子有相同的别名和日期。
- `author`：这个字段是一个`ForeignKey`。该字段定义了多对一的关系。我们告诉 Django，一篇帖子由一个用户编写，一个用户可以编写多篇帖子。对于该字段，Django 使用关联模型的主键，在数据库中创建一个外键。在这里，我们关联了 Django 权限系统的`User`模型。我们使用`related_name`属性，指定了从`User`到`Post`的反向关系名。之后我们会学习更多这方面的内容。
- `body`：帖子的正文。该字段是`TextField`，在 SQL 数据中会转换为`TEXT`。
	 `publish`：帖子发布的时间。	我们使用 Django 中`timezone`的`now`方法作为默认值。这只是一个时区感知的`datetime.now`。（**译者注：**根据不同时区，返回该时区的当前时间）
- `created`：帖子创建的时间。我们使用`auto_now_add`，因此创建对象时，时间会自动保存。
- `updated`：帖子最后被修改的时间。我们使用`auto_now`，因此保存对象时，时间会自动更新。
- `status`：该字段表示帖子的状态。我们使用`choices`参数，因此该字段的值只能是给定选项中的一个。

正如你所看到的，Django 内置了很多不同类型的字段，可以用来定义你的模型。你可以在[这里](https://docs.djangoproject.com/en/1.11/ref/models/fields/)找到所有字段类型。

模型中的`Meta`类包含元数据。我们告诉 Django，查询数据库时，默认排序是`publish`字段的降序排列。我们使用负号前缀表示降序排列。

`__str__()`方法是对象的默认可读表示。Django 会在很多地方（比如管理站点）使用它。

> 如果你是从 Python 2.X 中迁移过来的，请注意在 Python 3 中所有字符串天生就是 Unicode 编码，因此我们只使用`__str__()`方法。`__unicode__()`方法被废弃了。

因为我们要处理时间，所以将会安装`pytz`模块。该模块为 Python 提供了时区定义，同时 SQLite 也需要它操作时间。打开终端，使用以下命令安装`pytz`。

```py
pip install pytz
```

Django 内置支持时区感知。在项目的`settings.py`文件中，通过`USE_TZ`设置，启用或禁用时区支持。使用`startproject`管理命令创建新项目时，该设置为`True`。

### 1.3.1 激活应用

为了让 Django 保持追踪应用，并且可以为它的模型创建数据库，我们需要激活应用。编辑`settings.py`文件，在`INSTALLED_APPS`设置中添加`blog`。如下所示：

```py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'blog',
]
```

现在，Django 知道在该项目中，我们的应用已经激活，并且可以自省（instrospect）它的模型。

### 1.3.2 创建并应用数据库迁移

让我们在数据库中创建模型的数据库表。Django 自带一个迁移系统，可以追踪模型变化，并同步到数据库中。`migrate`命令会应用迁移到`INSTALLED_APPS`中列出的所有应用；它根据当前模型和迁移来同步数据库。

首先，我们需要为刚创建的新模型创建一个数据库迁移。在项目的根目录下输入以下命令：

```py
python manage.py makemigrations blog
```

你会得到类似以下的输出：

```py
Migrations for 'blog':
  0001_initial.py:
    - Create model Post
```

Django 在`blog`应用的`migrations`目录中创建了`0001_initial.py`文件。你可以打开该文件，查看数据库迁移生成的内容。

让我们看看 SQL 代码，Django 会在数据库中执行它们，为我们的模型创建数据库表。`sqlmigrate`命令接收一个数据库迁移名称，并返回 SQL 语句，但不会执行。运行以下命令检查数据：

```py
python manage.py sqlmigrate blog 0001
```

输入看起来是这样：

```py
BEGIN;
CREATE TABLE "blog_post" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "title" varchar(250) NOT NULL, "slug" varchar(250) NOT NULL, "body" text NOT NULL, "publish" datetime NOT NULL, "created" datetime NOT NULL, "updated" datetime NOT NULL, "status" varchar(10) NOT NULL, "author_id" integer NOT NULL REFERENCES "auth_user" ("id"));
CREATE INDEX "blog_post_slug_b95473f2" ON "blog_post" ("slug");
CREATE INDEX "blog_post_author_id_dd7a8485" ON "blog_post" ("author_id");
COMMIT;
```

使用不同的数据库，输出会有不同。上面是为 SQLite 生成的输出。正如你所看见的，Django 通过组合应用名和模型名的小写字母生成表名（blog_post），但你可以在模型的`Meta`类中使用`db_table`属性指定表明。Django 会自动为每个模型创建一个主键，但你同样可以在某个模型字段中指定`primary_key=True`，来指定主键。

让我们使用新模型同步数据库。运行以下命令，应用已经存在的数据库迁移：

```py
python manage.py migrate
```

你会得到以下面这行结尾的输出：

```py
  Applying blog.0001_initial... OK
```

我们刚才为`INSTALLED_APPS`中列出的所有应用（包括`blog`应用）进行了数据库迁移。应用了迁移后，数据库会反应模型的当前状态。

如果添加，移除或修改了已存在模型的字段，或者添加了新模型，你需要使用`makemigrations`命令创建一个新的数据库迁移。数据库迁移将会允许 Django 保持追踪模型的变化。然后，你需要使用`migrate`命令应用该迁移，保持数据库与模型同步。

## 1.4 为模型创建管理站点

现在，我们已经定义了`Post`模型，我们将会创建一个简单的管理站点，来管理博客帖子。Django 内置了管理界面，非常适合编辑内容。Django 管理站点通过读取模型的元数据，并为编辑内容提供可用于生产环境的界面，进行动态构建。你可以开箱即用，或者配置如何显示模型。

记住，`django.contrib.admin`已经包括在我们项目的`INSTALLED_APPS`设置中，所以我们不需要再添加。

### 1.4.1 创建超级用户

首先，我们需要创建一个用户来管理这个站点。运行以下命令：

```py
python manage.py createsuperuser
```

你会看到以下输出。输入你的用户名，e-mail 和密码：

```py
Username (leave blank to use 'admin'): admin
Email address: admin@admin.com
Password: ********
Password (again): ********
Superuser created successfully.
```

### 1.4.2 Django 管理站点

现在，使用`python manage.py runserver`命令启动开发服务器，并在浏览器中打开`http://127.0.0.1:8000/admin/`。你会看到如下所示的管理员登录界面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE1.2.png)

使用上一步创建的超级用户登录。你会看到管理站点的首页，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE1.3.png)

这里的`Group`和`User`模型是 Django 权限框架的一部分，位于`django.contrib.auth`中。如果你点击`Users`，会看到你之前创建的用户。你的`blog`应用的`Post`模型与`User`模型关联在一起。记住，这种关系由`author`字段定义。

### 1.4.3 添加模型到管理站点

让我们添加 blog 模型到管理站点。编辑`blog`应用的`admin.py`文件，如下所示：

```py
from django.contrib import admin
from .models import Post

admin.site.register(Post)
```

现在，在浏览器中重新载入管理站点。你会看到`Post`模型，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE1.4.png)

这很容易吧？当你在 Django 管理站点注册模型时，你会得到一个用户友好的界面，该界面通过内省你的模型产生，允许你非常方便的排列，编辑，创建和删除对象。

点击`Post`右边的`Add`链接来添加一篇新的帖子。你会看到 Django 为模型动态生成的表单，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE1.5.png)

Django 为每种字段类型使用不同的表单控件。即使是复杂的字段（比如`DateTimeField`），也会使用类似`JavaScript`的日期选择器显示一个简单的界面。

填写表单后，点击`Save`按钮。你会被重定向到帖子列表页面，其中显示一条成功消息和刚刚创建的帖子，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE1.6.png)

### 1.4.4 自定义模型显示方式

现在，我们看下如何自定义管理站点。编辑`blog`应用的`admin.py`文件，修改为：

```py
from django.contrib import admin
from .models import Post

class PostAdmin(admin.ModelAdmin):
    list_display = ('title', 'slug', 'author', 'publish', 'status')

admin.site.register(Post, PostAdmin)
```

我们告诉 Django 管理站点，使用从`ModelAdmin`继承的自定义类注册模型到管理站点。在这个类中，我们可以包括如何在管理站点中显示模型的信息，以及如何与它们交互。`list_display`属性允许你设置想在管理对象列表页中显示的模型字段。

让我们使用更多选项自定义管理模型，如下所示：

```py
class PostAdmin(admin.ModelAdmin):
    list_display = ('title', 'slug', 'author', 'publish', 'status')
    list_filter = ('status', 'created', 'publish', 'author')
    search_fields = ('title', 'body')
    prepopulated_fields = {'slug': ('title', )}
    raw_id_fields = ('author', )
    date_hierarchy = 'publish'
    ordering = ['status', 'publish']
```

回到浏览器，重新载入帖子类别页，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE1.7.png)

你可以看到，帖子列表页中显示的字段就是在`list_display`属性中指定的字段。现在，帖子列表页包括一个右边栏，可以通过`list_filter`属性中包括的字段来过滤结果。页面上出现了一个搜索栏。这是因为我们使用`search_fields`属性定义了可搜索的字段列表。在搜索栏下面，有一个通过日期进行快速导航的栏。这是通过定义`date_hierarchy`属性定义的。你还可以看到，帖子默认按`Status`和`Publish`列排序。这是因为使用`ordering`属性指定了默认排序。

现在点击`Add post`链接，你会看到有些不同了。当你为新帖子输入标题时，会自动填写`slug`字段。我们通过`prepopulated_fields`属性已经告诉了 Django，用`title`字段的输入预填充`slug`字段。同样，`author`字段显示为搜索控件，当你有成千上万的用户时，比下拉框更人性化，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE1.8.png)

通过几行代码，我们已经在管理站点中自定义了模型的显示方式。还有很多自定义和扩展 Django 管理站点的方式。本书后面的章节会涉及这个特性。

## 1.5 使用 QuerySet 和管理器

现在，你已经有了一个功能完整的管理站点来管理博客的内容，是时候学习如何从数据库中检索对象，并与之交互了。Django 自带一个强大的数据库抽象 API，可以很容易的创建，检索，更新和删除对象。Django 的 ORM（Object-relational Mapper）兼容 MySQL，PostgreSQL，SQLite 和 Oracle。记住，你可以在项目的`settings.py`文件中编辑`DATABASES`设置，来定义项目的数据库。Django 可以同时使用多个数据库，你可以用任何你喜欢的方式，甚至编写数据库路由来处理数据。

一旦创建了数据模型，Django 就提供了一个自由的 API 来与之交互。你可以在[这里](https://docs.djangoproject.com/en/1.11/ref/models/)找到数据模型的官方文档。

### 1.5.1 创建对象

打开终端，运行以下命令来打开 Python 终端：

```py
python manage.py shell
```

然后输入以下行：

```py
>>> from django.contrib.auth.models import User
>>> from blog.models import Post
>>> user = User.objects.get(username='admin')
>>> post = Post(title='One more post',
                slug='one-more-post',
                body='Post body.',
                author=user)
>>> post.save()
```

> **译者注：**书中的代码不是创建一个`Post`实例，而是直接使用`create()`在数据库中创建对象。这个地方应该是作者的笔误。

让我们分析这段代码做了什么。首先，我们检索`username`为`admin`的用户对象：

```py
user = User.objects.get(username='admin')
```

`get()`方法允许你冲数据库中检索单个对象。注意，该方法期望一个匹配查询的结果。如果数据库没有返回结果，该方法会抛出`DoesNotExist`异常；如果数据库返回多个结果，将会抛出`MultipleObjectsReturned`异常。这两个异常都是执行查询的模型类的属性。

然后，我们使用`title`，`slug`和`body`创建了一个`Post`实例，并设置之前返回的`user`作为帖子的作者：

```py
post = Post(title='Another post', slug='another-post', body='Post body.', author=user)
```

> 该对象在内存中，而不会存储在数据库中。

最后，我们使用`save()`方法保存`Post`对象到数据库中：

```py
post.save()

```

这个操作会在底层执行一个`INSERT`语句。我们已经知道如何先在内存创建一个对象，然后存储到数据库中，但也可以使用`create()`方法直接在数据库中创建对象：

```py
Post.objects.create(title='One more post', 
                    slug='one-more-post',
                    body='Post body.', 
                    author=user)
```

### 1.5.2 更新对象

现在，修改帖子的标题，并再次保存对象：

```py
>>> post.title = 'New title'
>>> post.save()
```

此时，`save()`方法会执行`UPDATE`语句。

> 直到调用`save()`方法，你对对象的修改才会存到数据库中。

### 1.5.3 检索对象

Django 的 ORM 是基于`QuerySet`的。一个`QuerySet`是来自数据库的对象集合，它可以有数个过滤器来限制结果。你已经知道如何使用`get()`方法从数据库检索单个对象。正如你所看到的，我们使用`Post.objects.get()`访问该方法。每个 Django 模型最少有一个管理器（manager），默认管理器叫做`objects`。你通过使用模型管理器获得一个`QuerySet`对象。要从表中检索所有对象，只需要在默认的`objects`管理器上使用`all()`方法，比如：

```py
>>> all_posts = Post.objects.all()
```

这是如何创建一个返回数据库中所有对象的`QuerySet`。注意，该`QuerySet`还没有执行。Django 的`QuerySet`是懒惰的；只有当强制它们执行时才会执行。这种行为让`QuerySet`变得很高效。如果没有没有把`QuerySet`赋值给变量，而是直接在 Python 终端输写，`QuerySet`的 SQL 语句会执行，因为我们强制它输出结果：

```py
>>> Post.objects.all()
```

#### 1.5.3.1 使用`filter()`方法

你可以使用管理器的`filter()`方法过滤一个`QuerySet`。例如，我们使用下面的`QuerySet`检索所有 2015 年发布的帖子：

```py
Post.objects.filter(publish__year=2015)
```

你也可以过滤多个字段。例如，我们可以检索 2015 年发布的，作者的`username`是`amdin`的帖子：

```py
Post.objects.filter(publish__year=2015, author__username='admin')
```

这等价于链接多个过滤器，来创建`QuerySet`：

```py
Post.objects.filter(publish__year=2015)\
            .filter(author__username='admin')
```

> 通过两个下划线（publish\_\_year)，我们使用字段查找方法构造了查询，但我们也可以使用两个下划线访问相关模型的字段（author\_\_username）。

#### 1.5.3.2 使用`exclude()`

你可以使用管理器的`exclude()`方法从`QuerySet`中排除某些结果。例如，我们可以检索所有 2017 年发布的，标题不是以`Why`开头的帖子：

```py
Post.objects.filter(publish__year=2017)\
            .exclude(title__startswith='Why')
```

#### 1.5.3.3 使用`order_by()`

你可以使用管理器的`order_by()`方法对不同字段进行排序。例如，你可以检索所有对象，根据它们的标题排序：

```py
Post.objects.order_by('title')
```

默认是升序排列。通过负号前缀指定降序排列，比如：

```py
Post.objects.order_by('-title')
```

### 1.5.4 删除对象

如果想要删除对象，可以这样操作：

```py
post = Post.objects.get(id=1)
post.delete()
```

> 注意，删除对象会删除所有依赖关系。

### 1.5.5 什么时候执行 QuerySet

你可以连接任意多个过滤器到`QuerySet`，在`QuerySet`执行之前，不会涉及到数据库。`QuerySet`只在以下几种情况被执行：

- 你第一次迭代它们
- 当你对它们进行切片操作。比如：`Post.objects.all()[:3]`
- 当你对它们进行`pickle`或缓存
- 当你对它们调用`repr()`或`len()`
- 当你显示对它们调用`list()`
- 当你在语句中测试，比如`bool()`，`or`，`and`或者`if`

### 1.5.6 创建模型管理器

正如我们之前提到的，`objects`是每个模型的默认管理器，它检索数据库中的所有对象。但我们也可以为模型自定义管理器。接下来，我们会创建一个自定义管理器，用于检索所有状态为`published`的帖子。

为模型添加管理器有两种方式：添加额外的管理器方法或者修改初始的管理器`QuerySet`。前者类似`Post.objects.my_manager()`，后者类似`Post.my_manager.all()`。我们的管理器允许我们使用`Post.published`来检索帖子。

编辑`blog`应用中的`models.py`文件，添加自定义管理器：

```py
class PublishedManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset()\
                      .filter(status='published')


class Post(models.Model):
    # ...
    objects = models.Manager()
    published = PublishedManager()
```

`get_queryset()`是返回被执行的`QuerySet`的方法。我们使用它在最终的`QuerySet`中包含了自定义的过滤器。我们已经自定义了管理器，并添加到`Post`模型中；现在可以用它来执行查询。例如，我们可以检索所有标题以`Who`开头，并且已经发布的帖子：

```py
Post.published.filter(title__startswith='Who')
```

> **译者注：**这里修改了`models.py`文件，因此需要在终端再次导入`Post`：`from blog.models import Post`。

## 1.6 构建列表和详情视图

现在，你已经了解了如何使用 ORM，可以随时构建博客应用的视图了。一个 Django 视图就是一个 Python 函数，它接收一个 web 请求，并返回一个 web 响应。视图中的所有逻辑返回期望的响应。

首先，我们会创建应用视图，然后定义每个视图的 URL 模式，最后创建 HTML 模板渲染视图产生的数据。每个视图渲染一个的模板，同时把变量传递给模板，并返回一个具有渲染输出的 HTTP 响应。

### 1.6.1 创建列表和详情视图

让我们从创建显示所有帖子的列表视图开始。编辑`blog`应用的`views.py`文件，如下所示：

```py
from django.shortcuts import render, get_object_or_404
from .models import Post

def post_list(request):
    posts = Post.published.all()
    return render(request,
                  'blog/post/list.html',
                  {'posts': posts})
```

你刚创建了第一个 Django 视图。`post_list`视图接收`request`对象作为唯一的参数。记住，该参数是所有视图都必需的。在这个视图中，我们使用之前创建的`published`管理器检索所有状态为`published`的帖子。

最后，我们使用 Django 提供的快捷方法`render()`，渲染指定模板的帖子列表。该函数接收`request`对象作为参数，通过模板路径和变量来渲染指定的模板。它返回一个带有渲染后文本（通常是 HTML 代码）的`HttpResponse`对象。`render()`快捷方法考虑了请求上下文，因此由模板上下文处理器（template context processor）设置的任何变量都可以由给定的模板访问。模板上下文处理器是可调用的，它们把变量设置到上下文中。你将会在第三章中学习如何使用它们。

让我们创建第二个视图，用于显示单个帖子。添加以下函数到`views.py`文件中：

```py
def post_detail(request, year, month, day, post):
    post = get_object_or_404(Post, slug=post,
                                   status='published',
                                   publish__year=year,
                                   publish__month=month,
                                   publish__day=day)
    return render(request,
                  'blog/post/detail.html',
                  {'post': post})
```

这是帖子的详情视图。该视图接收`year`，`month`，`day`和`post`作为参数，用于检索指定别名和日期的已发布的帖子。注意，当我们创建`Post`模型时，添加了`unique_for_date`参数到`slug`字段。这就确保了指定日期和别名时，只会检索到一个帖子。在详情视图中，我们使用`get_object_or_404()`快捷方法检索期望的帖子。该函数检索匹配给定参数的对象，如果没有找到对象，就会引发 HTTP 404（Not found）异常。最后，我们使用模板，调用`render()`快捷方法渲染检索出来的帖子。

### 1.6.2 为视图添加 URL 模式

一个 URL 模式由一个 Python 正则表达式，一个视图和一个项目范围内的名字组成。Django 遍历每个 URL 模式，并在匹配到第一个请求的 URL 时停止。然后，Django 导入匹配 URL 模式的视图，传递`HttpRequest`类实例和关键字或位置参数，并执行视图。

如果你以前没有使用过正则表达式，可以在[这里](https://docs.python.org/3/howto/regex.html )了解。

在`blog`应用的目录下新建一个`urls.py`文件，添加以下代码：

```py
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.post_list, name='post_list'),
    url(r'^(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/'\
        r'(?P<post>[-\w]+)/$',
        views.post_detail,
        name='post_detail'),
]
```

第一个 URL 模式不带任何参数，映射到`post_list`视图。第二个模式带以下四个参数，映射到`post_detail`视图。让我们看看 URL 模式的正则表达式：

- year：需要四个数字
- month：需要两个数字，在前面补零。
- day：需要两个数字，在前面补零。
- post：可以由单词和连字符组成。

> 最好为每个应用创建一个`urls.py`文件，这可以让应用在其它项目中复用。

现在你需要在项目的主 URL 模式中包含`blog`应用的 URL 模式。编辑项目目录中的`urls.py`文件，如下所示：

```py
from django.conf.urls import url, include
from django.contrib import admin

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^blog/', include('blog.urls',
                           namespace='blog',
                           app_name='blog'))
]
```

这样你就可以让 Django 包括 URL 模式，该模式在`blog/`路径下的`urls.py`文件中定义。你指定它们的命名空间为`blog`，这样你可以很容易的引用该 URLs 组。

### 1.6.3 模型的标准 URLs

你可以使用上一节定义的`post_detail` URL，为`Post`对象构建标准的 URL。Django 的惯例是在模型中添加`get_absolute_url()`方法，该方法返回对象的标准 URL。对于这个方法，我们会使用`reverse()`方法，它允许你通过它们的名字，以及传递参数构造 URLs。编辑`models.py`文件，添加以下代码：

```py
from django.core.urlresolvers import reverse
   Class Post(models.Model):
       # ...
       def get_absolute_url(self):
           return reverse('blog:post_detail',
                          args=[self.publish.year,
                                self.publish.strftime('%m'),
                                self.publish.strftime('%d'),
                                self.slug])

```

注意，我们使用`strftime()`函数构造使用零开头的月份和日期。我们将会在模板中使用`get_absolute_url()`方法。

## 1.7 为视图创建模板

我们已经为应用创建了视图和 URL 模式。现在该添加模板来显示用户界面友好的帖子了。

在你的`blog`应用目录中创建以下目录和文件：

```py
templates/
    blog/
        base.html
        post/
            list.html
            detail.html
```

这就是模板的文件结构。`base.html`文件将会包括网站的主 HTML 结构，它把内容分为主内容区域和一个侧边栏。`list.html`和`detail.html`文件继承自`base.html`文件，分别用于渲染帖子列表视图和详情视图。

Django 有一个强大的模板语言，允许你指定如何显示数据。它基于模板标签——`{% tag %}`，模板变量——`{{ variable }}`，和可作用于变量的模板过滤器——`{{ variable|filter }}`。你可以在[这里](https://docs.djangoproject.com/en/1.11/ref/templates/builtins/)查看所有内置模板标签和过滤器。

让我们编辑`base.html`文件，添加以下代码：

```py
{% load staticfiles %}
<!DOCTYPE html>
<html>
<head>
	<title>{% block title %}{% endblock %}</title>
   <link href="{% static "css/blog.css" %}" rel="stylesheet">
</head>
<body>
	<div id="content">
		{% block content %}
		{% endblock %}
	</div>
	<div id="sidebar">
		<h2>My blog</h2>
		<p>This is my blog.</p>
	</div>
</body>
</html>
```

`{% load staticfiles %}`告诉 Django 加载`staticfiles`模板标签，它是`django.contrib.staticfiles`应用提供的。加载之后，你可以在该模板中使用`{% static %}`模板过滤器。通过该模板过滤器，你可以包括静态文件（比如`blog.css`，在 blog 应用的`static/`目录下可以找到这个例子的代码）。拷贝这个目录到你项目的相同位置，来使用这些静态文件。

你可以看到，有两个`{% block %}`标签。它们告诉 Django，我们希望在这个区域定义一个块。从这个模板继承的模板，可以用内容填充这些块。我们定义了一个`title`块和一个`content`块。

让我们编辑`post/list.html`文件，如下所示：

```py
{% extends "blog/base.html" %}

{% block title %}My Blog{% endblock %}

{% block content %}
	<h1>My Blog</h1>
	{% for post in posts %}
		<h2>
			<a href="{{ post.get_absolute_url }}">
				{{ post.title }}
			</a>
		</h2>
		<p class="date">
			Published {{ post.publish }} by {{ post.author }}
		</p>
		{{ post.body|truncatewords:30|linebreaks }}
	{% endfor %}
{% endblock %}
```

使用`{% extends %}`模板标签告诉 Django 从`blog/base.html`模板继承。接着，我们填充基类模板的`title`和`content`块。我们迭代帖子，并显示它们的标题，日期，作者和正文，其中包括一个标题链接到帖子的标准 URL。在帖子的正文中，我们使用了两个模板过滤器：`truncatewords`从内容中截取指定的单词数，`linebreaks`把输出转换为 HTML 换行符。你可以连接任意多个模板过滤器；每个过滤器作用于上一个过滤器产生的输出。

打开终端，执行`python manage.py runserver`启动开发服务器。在浏览器打开`http://127.0.0.1:8000/blog/`，就能看到运行结果。注意，你需要一些状态为`Published`的帖子才能看到。如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE1.9.png)

接着，让我们编辑`post/detail.html`文件，添加以下代码：

```py
{% extends "blog/base.html" %}

{% block title %}{{ post.title }}{% endblock %}

{% block content %}
	<h1>{{ post.title }}</h1>
	<p class="date">
		Published {{ post.publish }} by {{ post.author }}
	</p>
	{{ post.body|linebreaks }}
{% endblock %}
```

返回浏览器，点击某条帖子的标题跳转到详情视图，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE1.10.png)

观察一下 URL，类似这样：`/blog/2017/04/25/who-was-django-reinhardt/`。我们为帖子创建了一个对搜索引擎友好的 URL。

## 1.8 添加分页

当你开始往博客中添加内容，你会发现需要把帖子分页。Django 内置了一个分页类，可以很容易的管理分页内容。

编辑`blog`应用的`views.py`文件，导入分页类，并修改`post_list`视图：

```py
from django.core.paginator import Paginator, EmptyPage,\
                                  PageNotAnInteger
                                  
def post_list(request):
	object_list = Post.published.all()
	paginator = Paginator(object_list, 3) # 3 posts in each page
	page = request.GET.get('page')
	try:
		posts = paginator.page(page)
	except PageNotAnInteger:
		# If page is not an integer deliver the first page
		posts = paginator.page(1)
	except EmptyPage:
		# If page is out of range deliver last page of results
       posts = paginator.page(paginator.num_pages)
   return render(request,
                 'blog/post/list.html',
					{'page': page, 'posts': posts})
```

分页是这样工作的：

1. 用每页想要显示的对象数量初始化`Paginator`类。
2. 获得`GET`中的`page`参数，表示当前页码。
3. 调用`Paginator`类的`page()`方法，获得想要显示页的对象。
4. 如果`page`参数不是整数，则检索第一页的结果。如果这个参数大于最大页码，则检索最后一页。
5. 把页码和检索出的对象传递给模板。

现在，我们需要创建显示页码的模板，让它可以在任何使用分页的模板中使用。在`blog`应用的`templates`目录中，创建`pagination.html`文件，并添加以下代码：

```py
<div class="pagination">
	<span class="step-links">
		{% if page.has_previous %}
			<a href="?page={{ page.previous_page_number }}">Previous</a>
		{% endif %}
		<span class="current">
			Page {{ page.number }} of {{ page.paginator.num_pages }}.
		</span>
		{% if page.has_next %}
			<a href="?page={{ page.next_page_number }}">Next</a>
			{% endif %}
	</span>
</div>
```

这个分页模板需要一个`Page`对象，用于渲染上一个和下一个链接，并显示当前页和总页数。让我们回到`blog/post/list.html`模板，将`pagination.html`模板包括在`{% content %}`块的底部，如下所示：

```py
{% block content %}
	...
	{% include "pagination.html" with page=posts %}
{% endblock %}
```

因为我们传递给模板的`Page`对象叫做`posts`，所以我们把分页模板包含在帖子列表模板中，并指定参数进行正确的渲染。通过这种方法，你可以在不同模型的分页视图中重用分页模板。

在浏览器中打开`http://127.0.0.1:8000/blog/`，你会在帖子列表底部看到分页，并且可以通过页码导航：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE1.11.png)

## 1.9 使用基于类的视图

视图接收一个 web 请求，返回一个 web 响应，并且可以被调用，所以可以把视图定义为类方法。Django 为此提供了基础视图类。它们都是继承自`View`类，可以处理 HTTP 方法调度和其它功能。这是创建视图的一个替代方法。

我们使用 Django 提供的通用`ListView`，把`post_list`视图修改为基于类的视图。这个基础视图允许你列出任何类型的对象。

编辑`blog`应用的`views.py`文件，添加以下代码：

```py
from django.views.generic import ListView

class PostListView(ListView):
	queryset = Post.published.all()
	context_object_name = 'posts'
	paginate_by = 3
	template_name = 'blog/post/list.html'
```

这个基于类的视图与之前的`post_list`视图类似，它做了以下操作：

- 使用特定的`QuerySet`代替检索所有对象。我们可以指定`model=Post`，然后 Django 会为我们创建通用的`Post.objects.all()`这个`QuerySet`，来代替定义一个`queryset`属性。
- 为查询结果使用上下文变量`posts`。如果不指定`context_object_name`，默认变量是`object_list`。
- 对结果进行分页，每页显示三个对象。
- 使用自定义模板渲染页面。如果没有设置默认模板，`ListView`会使用`blog/post_list.html`。

打开`blog`应用的`urls.py`文件，注释之前的`post_list` URL 模式，使用`PostListView`类添加新的 URL 模式：

```py
urlpatterns = [
	# post views
	# url(r'^$', views.post_list, name='post_list'),
	url(r'^$', views.PostListView.as_view(), name='post_list'),
	url(r'^(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/'\
	    r'(?P<post>[-\w]+)/$',
	    views.post_detail,
	    name='post_detail'),
]
```

为了保证分页正常工作，我们需要传递正确的`page`对象给模板。Django 的`ListView`使用`page_obj`变量传递选中页，因此你需要编辑`list.html`模板，使用正确的变量包括页码：

```py
{% include "pagination.html" with page=page_obj %}
```

在浏览器打开`http://127.0.0.1:8000/blog/`，检查是不是跟之前使用的`post_list`视图一致。这是一个基于类视图的简单示例，使用了 Django 提供的通过类。你会在第十章和后续章节学习更多基于类的视图。

## 1.10 总结

在这章中，通过创建一个基本的博客应用，我们学习了 Django 框架的基础知识。你设计了数据模型，并进行了数据库迁移。你创建了视图，模板和博客的 URLs，以及对象分页。

在下一章，你将会学习如何完善博客应用，包括评论系统，标签功能，并且允许用户通过 e-mail 分享帖子。