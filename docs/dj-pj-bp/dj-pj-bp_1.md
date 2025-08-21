# 第一章：Blueblog-博客平台

我们将从一个简单的 Django 博客平台开始。近年来，Django 已经成为 Web 框架中的明星领导者之一。当大多数人决定开始使用 Web 框架时，他们的搜索结果要么是**Ruby on Rails**（**RoR**），要么是 Django。两者都是成熟、稳定且被广泛使用的。似乎使用其中一个的决定主要取决于你熟悉哪种编程语言。Ruby 程序员选择 RoR，Python 程序员选择 Django。在功能方面，两者都可以用来实现相同的结果，尽管它们对待事物的方式有所不同。

如今最受欢迎的博客平台之一是 Medium，被许多知名博客作者广泛使用。它的流行源于其优雅的主题和简单易用的界面。我将带你创建一个类似的 Django 应用程序，其中包含大多数博客平台没有的一些惊喜功能。这将让你体验到即将到来的东西，并展示 Django 有多么多才多艺。

在开始任何软件开发项目之前，最好先大致规划一下我们想要实现的目标。以下是我们的博客平台将具有的功能列表：

+   用户应该能够注册账户并创建他们的博客

+   用户应该能够调整他们博客的设置

+   用户应该有一个简单的界面来创建和编辑博客文章

+   用户应该能够在平台上的其他博客上分享他们的博客文章

我知道这似乎是很多工作，但 Django 带有一些`contrib`包，可以大大加快我们的工作速度。

# contrib 包

`contrib`包是 Django 的一部分，其中包含一些非常有用的应用程序，Django 开发人员决定应该随 Django 一起发布。这些包含的应用程序提供了令人印象深刻的功能集，包括我们将在此应用程序中使用的一些功能：

+   管理是一个功能齐全的 CMS，可用于管理 Django 站点的内容。管理应用程序是 Django 流行的重要原因。我们将使用此功能为网站管理员提供界面，以便在我们的应用程序中进行数据的审查和管理

+   Auth 提供用户注册和身份验证，而无需我们做任何工作。我们将使用此模块允许用户在我们的应用程序中注册、登录和管理他们的个人资料

### 注意

`contrib`模块中还有很多好东西。我建议你查看完整列表[`docs.djangoproject.com/en/stable/ref/contrib/#contrib-packages`](https://docs.djangoproject.com/en/stable/ref/contrib/#contrib-packages)。

我通常在所有我的 Django 项目中至少使用三个`contrib`包。它们提供了通常需要的功能，如用户注册和管理，并使你能够专注于项目的核心部分，为你提供一个坚实的基础来构建。

# 设置我们的开发环境

对于这第一章，我将详细介绍如何设置开发环境。对于后面的章节，我只会提供最少的说明。有关我如何设置开发环境以及原因的更多详细信息，请参阅附录，*开发环境设置详细信息和调试技术*。

让我们从为我们的项目创建目录结构开始，设置虚拟环境并配置一些基本的 Django 设置，这些设置需要在每个项目中设置。让我们称我们的博客平台为 BlueBlog。

### 注意

有关即将看到的步骤的详细说明，请参阅附录，*开发环境设置详细信息和调试技术*。如果您对我们为什么要做某事或特定命令的作用感到不确定，请参考该文档。

要开始一个新项目，您需要首先打开您的终端程序。在 Mac OS X 中，它是内置终端。在 Linux 中，终端根据每个发行版单独命名，但您不应该有找到它的麻烦；尝试在程序列表中搜索单词终端，应该会显示相关内容。在 Windows 中，终端程序称为命令行。您需要根据您的操作系统启动相关程序。

### 注意

如果您使用 Windows 操作系统，您需要稍微修改书中显示的命令。请参考附录中的*在 Windows 上开发*部分，了解详情。

打开您操作系统的相关终端程序，并通过以下命令创建我们项目的目录结构；使用以下命令`cd`（进入）到根项目目录：

```py
> mkdir –p blueblog
> cd blueblog

```

接下来让我们创建虚拟环境，安装 Django，并启动我们的项目：

```py
> pyvenv blueblogEnv
> source blueblogEnv/bin/activate
> pip install django
> django-admin.py startproject blueblog src

```

搞定这些之后，我们就可以开始开发我们的博客平台了。

## 数据库设置

在您喜欢的编辑器中打开`$PROJECT_DIR/src/blueblog/settings.py`中的设置，并确保`DATABASES`设置变量与以下内容匹配：

```py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
```

为了初始化数据库文件，请运行以下命令：

```py
> cd src
> python manage.py migrate

```

## 静态文件设置

设置开发环境的最后一步是配置`staticfiles` `contrib`应用程序。staticfiles 应用程序提供了许多功能，使得管理项目的静态文件（css、图片、JavaScript）变得容易。虽然我们的使用将是最小化的，但您应该仔细查看 Django 文档中关于 staticfiles 的详细信息，因为它在大多数真实世界的 Django 项目中被广泛使用。您可以在[`docs.djangoproject.com/en/stable/howto/static-files/`](https://docs.djangoproject.com/en/stable/howto/static-files/)找到文档。

为了设置 staticfiles 应用程序，我们必须在`settings.py`文件中配置一些设置。首先确保`django.contrib.staticfiles`已添加到`INSTALLED_APPS`中。Django 应该默认已经做了这个。

接下来，将`STATIC_URL`设置为您希望静态文件从中提供的任何 URL。我通常将其保留为默认值`/static/`。这是 Django 在您使用静态模板标签获取静态文件路径时将放入您的模板中的 URL。

## 一个基础模板

接下来让我们设置一个基础模板，所有应用程序中的其他模板都将从中继承。我喜欢将项目源文件夹中多个应用程序使用的模板放在名为 templates 的目录中。为了设置这一点，在设置文件的`TEMPLATES`配置字典的`DIRS`数组中添加`os.path.join(BASE_DIR, 'templates')`，然后在`$PROJECT_ROOT/src`中创建一个名为 templates 的目录。接下来，使用您喜欢的文本编辑器，在新文件夹中创建一个名为`base.html`的文件，内容如下：

```py
<html>
<head>
    <title>BlueBlog</title>
</head>
<body>
    {% block content %}
    {% endblock %}
</body>
</html>
```

与 Python 类继承自其他类一样，Django 模板也可以继承自其他模板。就像 Python 类的函数可以被子类覆盖一样，Django 模板也可以定义子模板可以覆盖的块。我们的`base.html`模板提供了一个供继承模板覆盖的块，称为**content**。

使用模板继承的原因是代码重用。我们应该将我们希望在网站的每个页面上可见的 HTML，如标题、页脚、版权声明、元标记等，放在基础模板中。然后，任何继承自它的模板将自动获得所有这些常见的 HTML，我们只需要覆盖我们想要自定义的块的 HTML 代码。你将看到这种在本书中的项目中使用创建和覆盖基础模板中的块的原则。

# 用户帐户

数据库设置完成后，让我们开始创建我们的应用程序。如果你记得的话，我们功能列表中的第一件事是允许用户在我们的网站上注册帐户。正如我之前提到的，我们将使用 Django contrib 包中的 auth 包来提供用户帐户功能。

为了使用 auth 包，我们需要在设置文件（位于`$PROJECT_ROOT/src/blueblog/settings.py`）中的`INSTALLED_APPS`列表中添加它。在设置文件中，找到定义`INSTALLED_APPS`的行，并确保字符串`django.contrib.auth`是列表的一部分。默认情况下应该是这样的，但如果不是，请手动添加。

你会看到 Django 默认情况下包含了 auth 包和其他一些 contrib 应用程序到列表中。一个新的 Django 项目默认包含这些应用程序，因为几乎所有的 Django 项目最终都会使用它们。

### 注意

如果需要将 auth 应用程序添加到列表中，请记住使用引号括起应用程序名称。

我们还需要确保`MIDDLEWARE_CLASSES`列表包含`django.contrib.sessions.middleware.SessionMiddleware`、`django.contrib.auth.middleware.AuthenticationMiddleware`和`django.contrib.auth.middleware.SessionAuthenticationMiddleware`。这些中间件类让我们在视图中访问已登录的用户，并确保如果我更改了我的帐户密码，我将从先前登录的所有其他设备中注销。

随着你对各种 contrib 应用程序及其用途的了解越来越多，你可以开始删除你知道在项目中不需要的任何应用程序。现在，让我们添加允许用户在我们的应用程序中注册的 URL、视图和模板。

## 用户帐户应用程序

为了创建与用户帐户相关的各种视图、URL 和模板，我们将开始一个新的应用程序。要这样做，在命令行中输入以下内容：

```py
> python manage.py startapp accounts

```

这将在`src`文件夹内创建一个新的`accounts`文件夹。我们将在这个文件夹内的文件中添加处理用户帐户的代码。为了让 Django 知道我们想要在项目中使用这个应用程序，将应用程序名称（accounts）添加到`INSTALLED_APPS`设置变量中；确保用引号括起来。

## 帐户注册

我们将要处理的第一个功能是用户注册。让我们从在`accounts/views.py`中编写注册视图的代码开始。确保`views.py`的内容与这里显示的内容匹配：

```py
from django.contrib.auth.forms import UserCreationForm
from django.core.urlresolvers import reverse
from django.views.generic import CreateView

class UserRegistrationView(CreateView):
    form_class = UserCreationForm
    template_name = 'user_registration.html'

    def get_success_url(self):
        return reverse('home')
```

我将在稍后解释这段代码的每一行都做了什么。但首先，我希望你能达到一个状态，可以注册一个新用户并亲自看看流程是如何工作的。接下来，我们将为这个视图创建模板。为了创建模板，你首先需要在`accounts`文件夹内创建一个名为`templates`的新文件夹。文件夹的名称很重要，因为 Django 会自动在具有该名称的文件夹中搜索模板。要创建这个文件夹，只需输入以下命令：

```py
> mkdir accounts/templates

```

接下来，在`templates`文件夹内创建一个名为`user_registration.html`的新文件，并输入下面显示的代码：

```py
{% extends "base.html" %}

{% block content %}
<h1>Create New User</h1>
<form action="" method="post">{% csrf_token %}
    {{ form.as_p }}
    <input type="submit" value="Create Account" />
</form>
{% endblock %}
```

最后，删除`blueblog/urls.py`中的现有代码，并替换为以下内容：

```py
from django.conf.urls import include
from django.conf.urls import url
from django.contrib import admin
from django.views.generic import TemplateView
from accounts.views import UserRegistrationView

urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url(r'^$', TemplateView.as_view(template_name='base.html'), name='home'),
    url(r'^new-user/$', UserRegistrationView.as_view(), name='user_registration'),
]
```

这就是我们在项目中需要的所有代码来实现用户注册！让我们进行一个快速演示。通过输入以下命令来运行开发服务器：

```py
> python manage.py runser
ver

```

在浏览器中，访问`http://127.0.0.1:8000/new-user/`，您将看到一个用户注册表单。填写表单并点击提交。成功注册后，您将被带到一个空白页面。如果有错误，表单将再次显示，并显示适当的错误消息。让我们验证一下我们的新账户是否确实在数据库中创建了。

在下一步中，我们将需要一个管理员账户。Django auth contrib 应用程序可以为用户账户分配权限。具有最高权限级别的用户被称为**超级用户**。超级用户账户可以自由地管理应用程序并执行任何管理员操作。要创建超级用户账户，请运行以下命令：

```py
> python manage.py createsuperuser

```

### 注意

由于您已经在终端中运行了`runserver`命令，您需要先按下终端中的*Ctrl* + *C*来退出。然后您可以在同一个终端中运行`createsuperuser`命令。运行`createsuperuser`命令后，您需要再次启动`runserver`命令来浏览网站。

如果您想保持`runserver`命令运行，并在新的终端窗口中运行`createsuperuser`命令，您需要确保通过运行与我们创建新项目时相同的`source blueblogEnv/bin/activate`命令来激活此应用程序的虚拟环境。

创建完账户后，访问`http://127.0.0.1:8000/admin/`并使用管理员账户登录。您将看到一个名为**Users**的链接。点击该链接，您应该会看到我们应用程序中注册的用户列表。其中将包括您刚刚创建的用户。

恭喜！在大多数其他框架中，要实现一个可用的用户注册功能，需要付出更多的努力。Django 以其一应俱全的方式，使我们能够以最少的努力实现相同的功能。

接下来，我将解释您编写的每行代码的作用。

### 通用视图

以下是用户注册视图的代码：

```py
class UserRegistrationView(CreateView):
    form_class = UserCreationForm
    template_name = 'user_registration.html'

    def get_success_url(self):
        return reverse('home')
```

我们的视图对于做了这么多工作来说非常简短。这是因为我们使用了 Django 最有用的功能之一，即通用视图，而不是从头开始编写处理所有工作的代码。通用视图是 Django 提供的基类，提供了许多 Web 应用程序通常需要的功能。通用视图的强大之处在于能够轻松地对其进行大量定制。

### 注意

您可以在[`docs.djangoproject.com/en/stable/topics/class-based-views/`](https://docs.djangoproject.com/en/stable/topics/class-based-views/)上的文档中阅读更多关于 Django 通用视图的信息。

在这里，我们使用了`CreateView`通用视图。这个通用视图可以使用模板显示`ModelForm`，并在提交时，如果表单数据无效，可以重新显示页面并显示错误，或者调用表单的`save`方法并将用户重定向到可配置的 URL。`CreateView`可以以多种方式进行配置。

如果您希望从某个 Django 模型自动生成`ModelForm`，只需将`model`属性设置为`model`类，表单将自动从模型的字段生成。如果您希望表单只显示模型的某些字段，请使用`fields`属性列出您想要的字段，就像使用`ModelForm`时所做的那样。

在我们的情况下，我们不是自动生成`ModelForm`，而是提供了我们自己的`UserCreationForm`。我们通过在视图上设置`form_class`属性来实现这一点。这个表单是 auth contrib 应用的一部分，它提供了字段和一个`save`方法，可以用来创建一个新用户。随着我们在后面的章节中开始开发更复杂的应用程序，您会发现这种从 Django 提供的小型可重用部分组合解决方案的主题是 Django Web 应用程序开发中的常见做法，我认为这是框架中最好的特性之一。

最后，我们定义了一个`get_success_url`函数，它执行简单的反向 URL 并返回生成的 URL。`CreateView`调用此函数以获取在提交有效表单并成功保存时将用户重定向到的 URL。为了快速启动并运行某些东西，我们省略了一个真正的成功页面，只是将用户重定向到一个空白页面。我们以后会修复这个问题。

### 模板和 URL

模板扩展了我们之前创建的基本模板，简单地使用`CreateView`传递给它的表单，使用`form.as_p`方法显示表单，您可能在之前的简单 Django 项目中见过。

`urls.py`文件更有趣一些。您应该熟悉其中的大部分内容，我们包含管理站点 URL 的部分以及我们为视图分配 URL 的部分。我想在这里解释一下`TemplateView`的用法。

像`CreateView`一样，`TemplateView`是 Django 提供给我们的另一个通用视图。顾名思义，这个视图可以向用户呈现和显示模板。它有许多自定义选项。最重要的是`template_name`，它告诉它要呈现和显示给用户的模板是哪一个。

我们本可以创建另一个视图类，它是`TemplateView`的子类，并通过设置属性和覆盖函数来自定义它，就像我们为注册视图所做的那样。但我想向您展示 Django 中使用通用视图的另一种方法。如果您只需要自定义通用视图的一些基本参数；在这种情况下，我们只想设置视图的`template_name`参数，您可以将值作为函数关键字参数传递给类的`as_view`方法，这样只需要传递`key=value`对。在`urls.py`文件中包含它时。在这里，我们传递模板名称，当用户访问它的 URL 时，视图呈现的模板。由于我们只需要一个占位符 URL 来重定向用户，我们只需使用空白的`base.html`模板。

### 提示

通过传递键/值对来自定义通用视图的技术只有在您有兴趣自定义非常基本的属性时才有意义，就像我们在这里做的那样。如果您想要更复杂的自定义，我建议您子类化视图，否则您将很快得到难以维护的混乱代码。

## 登录和注销

注册完成后，让我们编写代码为用户提供登录和注销的功能。首先，用户需要一种方式从站点上的任何页面转到登录和注册页面。为此，我们需要在我们的模板中添加页眉链接。这是展示模板继承如何可以在我们的模板中导致更清洁和更少代码的绝佳机会。

在我们的`base.html`文件的`body`标签后面添加以下行：

```py
{% block header %}
<ul>
    <li><a href="">Login</a></li>
    <li><a href="">Logout</a></li>
    <li><a href="{% url "user_registration"%}">Register Account</a></li>
</ul>
{% endblock %}
```

如果您现在打开我们站点的主页（在`http://127.0.0.1:8000/`），您应该看到我们之前空白页面上的三个链接。它应该类似于以下截图：

![登录和注销](img/00698_01_01.jpg)

单击**注册账户**链接。您将看到我们之前的注册表单，以及相同的三个链接。请注意我们只将这些链接添加到`base.html`模板中。但由于用户注册模板扩展了基本模板，所以它在我们的努力下获得了这些链接。这就是模板继承真正发挥作用的地方。

您可能已经注意到登录/注销链接的`href`为空。让我们从登录部分开始。

### 登录视图

让我们先定义 URL。在`blueblog/urls.py`中从 auth 应用程序导入登录视图：

```py
from django.contrib.auth.views import login
```

接下来，将其添加到`urlpatterns`列表中：

```py
url(r'^login/$', login, {'template_name': 'login.html'}, name='login'),
```

然后，在`accounts/templates`中创建一个名为`login.html`的新文件。输入以下内容：

```py
{% extends "base.html" %}

{% block content %}
<h1>Login</h1>
<form action="{% url "login" %}" method="post">{% csrf_token %}
    {{ form.as_p }}

    <input type="hidden" name="next" value="{{ next }}" />
    <input type="submit" value="Submit" />
</form>
{% endblock %}
```

最后，打开`blueblog/settings.py`并在文件末尾添加以下行：

```py
LOGIN_REDIRECT_URL = '/'
```

让我们回顾一下我们在这里所做的事情。首先，请注意，我们没有创建自己的代码来处理登录功能，而是使用了 auth 应用程序提供的视图。我们使用`from django.contrib.auth.views import login`导入它。接下来，我们将其与登录/URL 关联起来。如果您还记得用户注册部分，我们将模板名称作为关键字参数传递给`as_view()`函数中的主页视图。这种方法用于基于类的视图。对于旧式的视图函数，我们可以将一个字典传递给`url`函数，作为关键字参数传递给视图。在这里，我们使用了我们在`login.html`中创建的模板。

如果您查看登录视图的文档（[`docs.djangoproject.com/en/stable/topics/auth/default/#django.contrib.auth.views.login`](https://docs.djangoproject.com/en/stable/topics/auth/default/#django.contrib.auth.views.login)），您会发现成功登录后，它会将用户重定向到`settings.LOGIN_REDIRECT_URL`。默认情况下，此设置的值为`/accounts/profile/`。由于我们没有定义这样的 URL，我们将更改设置以指向我们的主页 URL。

接下来，让我们定义登出视图。

### 登出视图

在`blueblog/urls.py`中使用`from django.contrib.auth.views import logout`导入登出视图，并将以下内容添加到`urlpatterns`列表中：

```py
url(r'^logout/$', logout, {'next_page': '/login/'}, name='logout'),
```

就是这样。登出视图不需要模板；它只需要配置一个 URL，以在登出后将用户重定向到该 URL。我们只需将用户重定向回登录页面。

### 导航链接

在添加了登录/登出视图之后，我们需要让之前在导航菜单中添加的链接带用户到这些视图。将`templates/base.html`中的链接列表更改为以下内容：

```py
<ul>
    {% if request.user.is_authenticated %}
    <li><a href="{% url "logout" %}">Logout</a></li>
    {% else %}
    <li><a href="{% url "login" %}">Login</a></li>
    <li><a href="{% url "user_registration"%}">Register Account</a></li>
    {% endif %}
</ul>
```

如果用户尚未登录，这将向用户显示**登录和注册账户**链接。如果他们已经登录，我们使用`request.user.is_authenticated`函数进行检查，只会显示**登出**链接。您可以自行测试所有这些链接，并查看需要多少代码才能使我们网站的一个重要功能运行。这一切都是因为 Django 提供的 contrib 应用程序。

## 博客

用户注册已经完成，让我们开始处理应用程序的博客部分。我们将为博客创建一个新应用程序，在控制台中输入以下内容：

```py
> python manage.py startapp blog
> mkdir blog/templates

```

将博客应用程序添加到`settings.py`文件中的`INSTALLED_APPS`列表中。应用程序创建并安装后，让我们开始使用我们将使用的模型。

## 模型

在`blog/models.py`中，输入下面显示的代码：

```py
from django.contrib.auth.models import User
from django.db import models

class Blog(models.Model):
    owner = models.ForeignKey(User, editable=False)
    title = models.CharField(max_length=500)

    slug = models.CharField(max_length=500, editable=False)

class BlogPost(models.Model):
    blog = models.ForeignKey(Blog)
    title = models.CharField(max_length=500)
    body = models.TextField()

    is_published = models.BooleanField(default=False)

    slug = models.SlugField(max_length=500, editable=False)
```

在输入此代码后，运行以下命令为这些模型创建数据库表：

```py
> python manage.py makemigrations blog
> python manage.py migrate blog

```

这将创建支持我们新模型所需的数据库表。模型非常基本。您可能以前没有使用过的一个字段类型是**SlugField**。Slug 是用于唯一标识某物的一段文本。在我们的情况下，我们使用两个 slug 字段来标识我们的博客和博客文章。由于这些字段是不可编辑的，我们将不得不编写代码为它们赋一些值。我们稍后会研究这个问题。

## 创建博客视图

让我们创建一个视图，用户可以在其中设置他的博客。让我们创建一个用户将用来创建新博客的表单。创建一个新文件`blog/forms.py`，并输入以下内容：

```py
from django import forms

from blog.models import Blog

class BlogForm(forms.ModelForm):
    class Meta:
        model = Blog

        fields = [
                 'title'
                 ]
```

这将创建一个模型表单，允许仅对我们的`Blog`模型的**标题**字段进行编辑。让我们创建一个模板和视图来配合这个表单。

创建一个名为`blog/templates/blog_settings.html`的文件，并输入以下 HTML 代码：

```py
{% extends "base.html" %}

{% block content %}
<h1>Blog Settings</h1>
<form action="{% url "new-blog" %}" method="post">{% csrf_token %}
    {{ form.as_p }}

    <input type="submit" value="Submit" />
</form>
{% endblock %}
```

您可能已经注意到，我在博客设置命名的 URL 上使用了`url`标签，但尚未创建该 URL 模式。在创建视图后，我们将这样做，但请记住名称，确保我们的 URL 得到相同的名称。

### 注意

创建视图、模板和 URL 的顺序没有固定的规定。你可以自行决定哪种方式更适合你。

在你的`blog/views.py`文件中，添加以下代码来创建视图：

```py
from django.core.urlresolvers import reverse
from django.http.response import HttpResponseRedirect
from django.utils.text import slugify
from django.views.generic import CreateView

from blog.forms import BlogForm

class NewBlogView(CreateView):
    form_class = BlogForm
    template_name = 'blog_settings.html'

    def form_valid(self, form):
        blog_obj = form.save(commit=False)
        blog_obj.owner = self.request.user
        blog_obj.slug = slugify(blog_obj.title)

        blog_obj.save()
        return HttpResponseRedirect(reverse('home'))
```

修改`blueblog/urls.py`。在文件顶部添加`from blog.views import NewBlogView`，并将其添加到`urlpatterns`列表中：

```py
url(r'^blog/new/$', NewBlogView.as_view(), name='new-blog'),
```

作为最后一步，我们需要一些方式让用户访问我们的新视图。将`base.html`中的标题块更改为以下内容：

```py
{% block header %}
<ul>
    {% if request.user.is_authenticated %}
    <li><a href="{% url "new-blog" %}">Create New Blog</a></li>
    <li><a href="{% url "logout" %}">Logout</a></li>
    {% else %}
    <li><a href="{% url "login" %}">Login</a></li>
    <li><a href="{% url "user_registration"%}">Register Account</a></li>
    {% endif %}
</ul>
{% endblock %}
```

要测试我们的最新功能，打开`http://127.0.0.1:8000`上的主页，然后点击**创建新博客**链接。它将呈现一个表单，您可以在其中输入博客标题并保存您的新博客。页面应该类似于以下截图：

![创建博客视图](img/00698_01_02.jpg)

我们添加的大部分代码都很基本。有趣的部分是`NewBlogView`。让我们看看它是如何工作的。首先，注意我们是从`CreateView`通用视图中继承的。创建视图允许我们轻松地显示和处理一个将创建给定模型的新对象的表单。要配置它，我们可以设置视图的`model`和`fields`属性，然后创建视图将使用它们生成模型表单，或者我们可以手动创建模型表单并将其分配给视图，就像我们在这里做的那样。

我们还配置了用于显示表单的模板。然后我们定义`form_valid`函数，当表单提交有效数据时，创建视图将调用该函数。在我们的实现中，我们调用模型表单的`save`方法，并将`commit`关键字参数设置为`False`。这告诉表单使用传递的数据创建我们模型的新对象，但不保存创建的对象到数据库。然后我们将新博客对象的所有者设置为登录的用户，并将其 slug 设置为用户输入的标题的 slugified 版本。slugify 是 Django 提供的众多实用函数之一。一旦我们根据我们的要求修改了博客对象，我们保存它并从`form_valid`函数返回`HttpResponseRedirect`。这个响应返回给浏览器，然后将用户带到主页。

到目前为止，我们的主页只是一个带有导航栏的空白页面。但它有一个严重的问题。首先通过导航栏中的链接创建一个新的博客。成功创建新博客后，我们将被重定向回主页，再次看到一个链接来创建另一个博客。但这不是我们想要的行为。理想情况下，我们的用户应该限制为每个帐户一个博客。

让我们来解决这个问题。首先，我们将限制博客创建视图，只允许用户在没有博客的情况下创建博客。在`blog/views.py`中导入`HttpResponseForbidden`和`Blog`模型：

```py
from django.http.response import HttpResponseForbidden
from blog.models import Blog
```

在`NewBlogView`类中添加一个`dispatch`方法，其中包含以下代码：

```py
def dispatch(self, request, *args, **kwargs):
    user = request.user
    if Blog.objects.filter(owner=user).exists():
        return HttpResponseForbidden ('You can not create more than one blogs per account')
    else:
        return super(NewBlogView, self).dispatch(request, *args, **kwargs)
```

`dispatch`方法是要在通用视图上覆盖的最有用的方法之一。当视图 URL 被访问时，它是第一个被调用的方法，并根据请求类型决定是否调用视图类上的`get`或`post`方法来处理请求。因此，如果您想要在所有请求类型（GET、POST、HEAD、PUT 等）上运行一些代码，`dispatch`是要覆盖的最佳方法。

在这种情况下，我们确保用户没有与其帐户关联的博客对象。如果有，我们将使用`HttpResponseForbidden`响应类返回`Not Allowed`响应。试一下。如果您之前已经创建了博客，现在甚至不能访问新的博客页面，而应该看到一个错误。

最后一件事。在注销后尝试访问 URL`http://127.0.0.1:8000/blog/new/`。注意您将收到`AnonymousUser`对象不可迭代的错误。这是因为即使您没有以注册用户的身份登录，视图的代码仍然假定您是。此外，您应该无法在未登录的情况下访问新博客页面。为了解决这个问题，首先将这两个导入行放在`blog/views.py`的顶部：

```py
from django.utils.decorators import method_decorator
from django.contrib.auth.decorators import login_required
```

然后更改 dispatch 方法的定义行以匹配以下内容：

```py
@method_decorator(login_required)
def dispatch(self, request, *args, **kwargs):
```

如果您现在尝试在未登录的情况下访问页面，您应该会看到`Page not found (404)`的 Django 错误页面。如果您查看该页面的 URL，您将看到 Django 正在尝试提供`/accounts/login/`的 URL。这是`login_required`装饰器的默认行为。为了解决这个问题，我们需要更改设置文件中`LOGIN_URL`变量的值。将其放在`blueblog/settings.py`中：

```py
LOGIN_URL = '/login/'
```

现在尝试访问`http://localhost:8000/blog/new/`，您将被重定向到登录页面。如果输入正确的用户名/密码组合，您将登录并被带到您之前尝试访问的页面，**创建新博客**页面。这个功能是免费提供给我们的，因为我们使用了 Django 的内置登录视图。

我们将在后面的章节中讨论`method_decorator`和`login_required`装饰器。如果您现在想要更多关于这些的信息，请查看 Django 文档中它们的文档。它在解释这两者方面做得非常出色。

您可以在[`docs.djangoproject.com/en/stable/topics/auth/default/#the-login-required-decorator`](https://docs.djangoproject.com/en/stable/topics/auth/default/#the-login-required-decorator)找到`login_required`的文档。对于`method_decorator`，您可以查看[`docs.djangoproject.com/en/stable/topics/class-based-views/intro/#decorating-the-class`](https://docs.djangoproject.com/en/stable/topics/class-based-views/intro/#decorating-the-class)。

## 主页

现在是时候为我们的用户创建一个合适的主页，而不是显示一个空白页面和一些导航链接。此外，当**创建新博客**链接导致错误页面时，向用户显示它似乎非常不专业。让我们通过创建一个包含一些智能的主页视图来解决所有这些问题。我们将在博客应用程序中放置我们的主页视图的代码。从技术上讲，它可以放在任何地方，但我个人喜欢将这样的视图放在项目的主要应用程序（在这种情况下是博客）或创建一个新的应用程序来放置这样的常见视图。在您的`blog/views.py`文件中，从`django.views.generic`中导入`TemplateView`通用视图，并放入以下视图的代码：

```py
class HomeView(TemplateView):
    template_name = 'home.html'

    def get_context_data(self, **kwargs):
        ctx = super(HomeView, self).get_context_data(**kwargs)

        if self.request.user.is_authenticated():
            ctx['has_blog'] = Blog.objects.filter(owner=self.request.user).exists()

        return ctx
```

通过在`blueblog/urls.py`中导入它`from blog.views import HomeView`，并将现有的根 URL 配置从`url(r'^$', TemplateView.as_view(template_name='base.html'), name='home'),`更改为`url(r'^$', HomeView.as_view(), name='home'),`，将此新视图绑定到主页 URL。

由于不再需要`TemplateView`类，您可以从导入中将其删除。您应该已经对我们在这里做什么有了一个很好的想法。唯一新的东西是`TemplateView`及其`get_context_data`方法。`TemplateView`是 Django 内置的另一个通用视图。我们通过提供模板文件名来配置它，并且视图通过将我们的`get_context_data`函数返回的字典作为上下文传递给模板来呈现该模板。在这里，如果用户有与其帐户关联的现有博客，我们将`has_blog`上下文变量设置为`True`。

我们的观点已经完成，我们需要对`base.html`模板进行一些更改，并添加一个新的`home.html`模板。对于`base.html`模板，更改头部块中的代码以匹配：

```py
{% block header %}
<ul>
    {% if request.user.is_authenticated %}
    {% block logged_in_nav %}{% endblock %}
    <li><a href="{% url "logout" %}">Logout</a></li>
    {% else %}
    <li><a href="{% url "login" %}">Login</a></li>
    <li><a href="{% url "user_registration"%}">Register Account</a></li>
    {% endif %}
</ul>
{% endblock %}
```

我们已经删除了**创建新博客**链接，并用另一个名为`logged_in_nav`的块进行了替换。这个想法是每个从基本模板继承的页面都可以在这里添加导航链接，以显示给已登录的用户。最后，创建一个名为`blog/templates/home.html`的新文件，并添加以下代码：

```py
{% extends "base.html" %}

{% block logged_in_nav %}
{% if not has_blog %}
<li><a href="{% url "new-blog" %}">Create New Blog</a></li>
{% else %}
<li><a href="">Edit Blog Settings</a></li>
{% endif %}
{% endblock %}
```

就像我们讨论的那样，主页模板覆盖了`logged_in_nav`块，以添加一个链接来创建一个新的博客（如果用户没有现有的博客），或者编辑现有博客的设置。您可以通过访问主页来测试我们所有的更改，看看已经创建了博客的用户和没有博客的新用户。您会看到只有在用户还没有创建博客时，才会显示创建新博客的链接。

接下来，让我们来处理设置视图。

## 博客设置视图

将视图的代码放在`blog/views.py`中：

```py
class UpdateBlogView(UpdateView):
    form_class = BlogForm
    template_name = 'blog_settings.html'
    success_url = '/'
    model = Blog

    @method_decorator(login_required)
    def dispatch(self, request, *args, **kwargs):
        return super(UpdateBlogView, self).dispatch(request, *args, **kwargs)
```

您需要从`django.views.generic`中导入`UpdateView`。还要更新同一文件中`HomeView`的`get_context_data`方法，使其与此匹配。

```py
def get_context_data(self, **kwargs):
    ctx = super(HomeView, self).get_context_data(**kwargs)

    if self.request.user.is_authenticated():
        if Blog.objects.filter(owner=self.request.user).exists():
            ctx['has_blog'] = True
            ctx['blog'] = Blog.objects.get(owner=self.request.user)
    return ctx
```

将`blog/templates/blog_settings.html`更改为以下内容：

```py
{% extends "base.html" %}

{% block content %}
<h1>Blog Settings</h1>
<form action="" method="post">{% csrf_token %}
    {{ form.as_p }}

    <input type="submit" value="Submit" />
</form>
{% endblock %}
```

我们唯一做的改变是删除了之前在表单动作中明确定义的 URL。这样，表单将始终提交到提供它的 URL。这一点很重要，我们以后会看到。

按照以下代码更新`blog/templates/home.html`：

```py
{% extends "base.html" %}

{% block logged_in_nav %}
{% if not has_blog %}
<li><a href="{% url "new-blog" %}">Create New Blog</a></li>
{% else %}
<li><a href="{% url "update-blog" pk=blog.pk %}">Edit Blog Settings</a></li>
{% endif %}
{% endblock %}
```

最后，在`blueblog/urls.py`中导入`UpdateBlogView`，并将以下内容添加到`urlpatterns`。

```py
url(r'^blog/(?P<pk>\d+)/update/$', UpdateBlogView.as_view(), name='update-blog'),
```

就是这样。使用您在上一节中用来创建博客的用户访问主页，这次您会看到一个链接来编辑您的博客，而不是创建一个新的。在这里要看的有趣的地方是`UpdateView`子类；`UpdateBlogView`。我们只定义了表单类、模板名称、成功的 URL 和模型，就得到了一个完整的可工作的更新视图。通过配置这些东西，并且我们的 URL 设置使得我们要编辑的对象的主键作为关键字参数`pk`传递给我们的视图，`UpdateView`会显示一个与我们要编辑的模型实例相关联的表单。在主页视图中，我们将用户的博客添加到上下文中，并在主页模板中使用它来生成一个用于更新视图的 URL。

在表单中，我们需要更改表单的动作属性，以便在提交时，它会发布到当前页面。由于我们在创建和更新视图中使用相同的模板，我们需要表单提交到渲染自身的任何 URL。正如您将在即将到来的项目中看到的那样，在 Django 中使用相同模板与类似视图是一种常见的做法。而 Django 通用视图的结构使这更容易实现。

## 创建和编辑博客文章

让我们创建用户可以使用来创建和编辑博客文章的视图。让我们从创建新博客文章开始。我们之前已经创建了模型，所以让我们从我们将使用的表单和模板开始。在`blog/forms.py`中，创建这个表单：

```py
class BlogPostForm(forms.ModelForm):
    class Meta:
        model = BlogPost

        fields = [
                 'title',
                 'body'
                 ]
```

您还需要导入`BlogPost`模型。对于模板，创建一个名为`blog/templates/blog_post.html`的新文件，并添加以下内容：

```py
{% extends "base.html" %}

{% block content %}
<h1>Create New Blog Post</h1>
<form action="" method="post">{% csrf_token %}
    {{ form.as_p }}

    <input type="submit" value="Submit" />
</form>
{% endblock %}
```

在`blog/views.py`中，导入`BlogPostForm`和`BlogPost`模型，然后创建`NewBlogPostView`：

```py
class NewBlogPostView(CreateView):
    form_class = BlogPostForm
    template_name = 'blog_post.html'

    @method_decorator(login_required)
    def dispatch(self, request, *args, **kwargs):
        return super(NewBlogPostView, self).dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        blog_post_obj = form.save(commit=False)
        blog_post_obj.blog = Blog.objects.get(owner=self.request.user)
        blog_post_obj.slug = slugify(blog_post_obj.title)
        blog_post_obj.is_published = True

        blog_post_obj.save()

        return HttpResponseRedirect(reverse('home'))
```

在`blueblog/urls.py`中，导入前面的视图，并添加以下 URL 模式：

```py
url(r'blog/post/new/$', NewBlogPostView.as_view(), name='new-blog-post'),
```

最后，将主页模板`blog/template/home.html`更改为链接到我们的新页面：

```py
{% extends "base.html" %}

{% block logged_in_nav %}
    {% if not has_blog %}
    <li><a href="{% url "new-blog" %}">Create New Blog</a></li>
    {% else %}
    <li><a href="{% url "update-blog" pk=blog.pk %}">Edit Blog Settings</a></li>
    <li><a href="{% url "new-blog-post" %}">Create New Blog Post</a></li>
    {% endif %}
{% endblock %}
```

到目前为止，所有这些代码对你来说应该都很熟悉。我们使用了模型表单和通用视图来获得我们需要的功能，而我们需要做的只是配置一些东西。我们没有写一行代码来创建相关的表单字段，验证用户输入，并处理各种错误和成功的情况。

您可以通过在主页上导航中使用**创建新博客文章**链接来测试我们的新视图。

## 编辑博客文章

与之前对`Blog`模型所做的一样，我们将使用相同的模板为博客文章创建一个编辑视图。但首先，我们需要为用户添加一种查看他的博客文章并链接到编辑页面的方式。为了保持简单，让我们将此列表添加到我们的主页视图中。在**HomeView**中，编辑`get_context_data`方法以匹配以下内容：

```py
def get_context_data(self, **kwargs):
    ctx = super(HomeView, self).get_context_data(**kwargs)

    if self.request.user.is_authenticated():
        if Blog.objects.filter(owner=self.request.user).exists():
            ctx['has_blog'] = True
            blog = Blog.objects.get(owner=self.request.user)

            ctx['blog'] = blog
            ctx['blog_posts'] = BlogPost.objects.filter(blog=blog)

    return ctx
```

在`blog/templates/home.html`的末尾，在`logged_in_nav`块结束后，添加以下代码来覆盖内容块并显示博客文章：

```py
{% block content %}
<h1>Blog Posts</h1>
<ul>
    {% for post in blog_posts %}
    <li>{{ post.title }} | <a href="">Edit Post</a></li>
    {% endfor %}
</ul>
{% endblock %}
```

如果您现在访问主页，您将看到用户发布的帖子列表。让我们创建编辑帖子的功能。在`blog/views.py`中创建以下视图：

```py
class UpdateBlogPostView(UpdateView):
    form_class = BlogPostForm
    template_name = 'blog_post.html'
    success_url = '/'
    model = BlogPost

    @method_decorator(login_required)
    def dispatch(self, request, *args, **kwargs):
        return super(UpdateBlogPostView, self).dispatch(request, *args, **kwargs)
```

将此视图导入到您的`blueblog/urls.py`文件中，并添加以下模式：

```py
url(r'blog/post/(?P<pk>\d+)/update/$', UpdateBlogPostView.as_view(), name='update-blog-post'),
```

编辑我们之前在主页模板中创建的博客文章列表，以添加编辑帖子的 URL：

```py
{% for post in blog_posts %}
    <li>{{ post.title }} | <a href="{% url "update-blog-post" pk=post.pk %}">Edit Post</a></li>
{% endfor %}
```

如果您现在打开主页，您会看到可以单击**编辑帖子**链接，并且它会带您到博客文章的编辑页面。我们需要修复的最后一件事是编辑博客文章页面的标题。您可能已经注意到，即使在编辑时，标题也会显示**创建新博客文章**。为了解决这个问题，请将`blog/templates/blog_post.html`中的`h1`标签替换为以下内容：

```py
<h1>{% if object %}Edit{% else %}Create{% endif %} Blog Post</h1>
```

`UpdateView`通过模板传递给模板的上下文包括一个名为`object`的变量。这是用户当前正在编辑的实例。我们在模板中检查此变量的存在。如果找到它，我们知道正在编辑现有的博客文章。如果没有，我们知道正在创建新的博客文章。我们检测到这一点并相应地设置标题。

## 查看博客文章

要添加一个显示博客文章的视图，请在`blog/views.py`中添加以下视图类：

```py
class BlogPostDetailsView(DetailView):
    model = BlogPost
    template_name = 'blog_post_details.html'
```

记得从`django.views.generic`中导入`DetailView`通用视图。接下来，使用以下代码创建`blog/templates/blog_post_details.html`模板：

```py
{% extends "base.html" %}

{% block content %}
<h1>{{ object.title }}</h1>
<p>{{ object.body }}</p>
{% endblock %}
```

导入详细视图，并将以下 URL 模式添加到`urls.py`文件中：

```py
url(r'blog/post/(?P<pk>\d+)/$', BlogPostDetailsView.as_view(), name='blog-post-details'),
```

最后，在主页模板中更改博客文章列表，以从帖子标题链接到帖子详细页面：

```py
{% for post in blog_posts %}
    <li><a href="{% url "blog-post-details" pk=post.pk %}">{{ post.title }}</a> | <a href="{% url "update-blog-post" pk=post.pk %}">Edit Post</a></li>
{% endfor %}
```

在主页上，博客文章标题现在应该链接到详细页面。

# 多个用户

到目前为止，我们只使用了一个用户账户，并使我们的网站适用于该用户。让我们进入令人兴奋的部分，并将帖子分享到其他用户的博客中。但是，一旦多个用户加入到混合中，我们在继续之前应该看一下一件事。

## 安全性

为了展示我们应用程序中完全缺乏安全性，让我们创建一个新的用户账户。使用页眉链接注销并注册一个新账户。接下来，用该用户登录。您应该会进入主页，并且在列表中不应该看到任何博客文章。

现在，在 URL`http://127.0.0.1:8000/blog/post/1/update/`中输入。您应该会在编辑视图中看到我们从第一个用户创建的博客文章。更改博客文章的标题或正文，然后单击保存。您将被重定向回主页，并且似乎保存成功了。重新登录到第一个账户，您会看到博客文章的标题已更新。这是一个严重的安全漏洞，必须修复，否则任何用户都可以无限制地编辑其他用户的博客文章。

我们再次解决这个问题的简单方式展示了 Django 框架的强大和简单。将以下方法添加到`UpdateBlogPostView`类中：

```py
def get_queryset(self):
    queryset = super(UpdateBlogPostView, self).get_queryset()
    return queryset.filter(blog__owner=self.request.user)
```

就是这样！再次尝试打开`http://127.0.0.1:8000/blog/post/1/update/`。这次，您不会再看到允许您编辑另一个用户的博客文章，而是会看到一个 404 页面。

在查看`UpdateView`通用视图的工作方式后，可以理解这小段代码的作用。通用视图调用许多小方法，每个方法都有特定的工作。以下是`UpdateView`类定义的一些方法的列表：

+   `get_object`

+   `get_queryset`

+   `get_context_object_name`

+   `get_context_data`

+   `get_slug_field`

拥有这些小方法的好处是，为了改变子类的功能，我们可以只覆盖其中一个并实现我们的目的，就像我们在这里所做的那样。阅读 Django 文档，了解通用视图使用的这些方法和许多其他方法的作用。

对于我们的情况，`get_queryset`方法，正如其名称所示，获取在其中搜索要编辑的对象的查询集。我们从超级方法中获取默认的`queryset`（它只返回`self.model.objects.all()`），并返回一个进一步过滤的版本，只包括当前登录用户拥有的博客文章。您应该熟悉关系过滤器。如果这些对您来说是新的，请阅读 Django 教程，熟悉模型查询集过滤的基础知识。

现在如果您尝试访问其他人的博客文章，您会看到 404 的原因是，当`CreateView`尝试获取要编辑的对象时，它收到的查询集只包括当前登录用户拥有的博客文章。由于我们试图编辑其他人的博客文章，它不包括在该查询集中。找不到要编辑的对象，`CreateView`返回 404。

## 分享博客文章

博客文章分享功能允许用户选择要与其博客文章分享的另一个用户的博客。这将允许用户通过在更受欢迎的作家的博客上分享其内容来获得更多读者，读者将能够在一个地方阅读更相关的内容，而不需要发现更多的博客。

使分享成为可能的第一步是在`BlogPost`模型上添加一个字段，指示帖子与哪些博客共享。将此字段添加到`blog/models.py`中的`BlogPost`模型：

```py
shared_to = models.ManyToManyField(Blog, related_name='shared_posts')
```

我们只是添加了一个基本的 Django 多对多关系字段。如果您想复习一下多对多字段提供的功能，我建议您再次查看 Django 教程，特别是处理 M2M 关系的部分。

关于新字段需要注意的一点是，我们必须明确指定`related_name`。您可能知道，每当您使用任何关系字段（`ForeignKey`，`OneToMany`，`ManyToMany`）将一个模型与另一个模型关联时，Django 会自动向另一个模型添加一个属性，以便轻松访问链接的模型。

在添加`shared_to`字段之前，`BlogPost`模型已经有一个指向`Blog`模型的`ForeignKey`。如果您查看了`Blog`模型上可用的属性（使用 shell），您会发现一个`blogpost_set`属性，这是一个管理器对象，允许访问引用该`Blog`的`BlogPost`模型。如果我们尝试添加`ManyToMany`字段而没有`related_name`，Django 会抱怨，因为新的关系还会尝试添加一个反向关系，也称为`blogpost_set`。因此，我们需要给反向关系取另一个名字。

定义了 M2M 关系后，您现在可以通过在`Blog`模型上使用`shared_posts`属性的`all()`方法来访问与博客模型共享的博客文章。稍后我们将看到一个例子。

定义新字段后，运行以下命令迁移您的数据库以创建新的关系：

```py
> python manage.py makemigrations blog
> python manage.py migrate blog

```

接下来，让我们创建一个视图，允许用户选择要与其博客文章分享的博客。**将**此添加到`blog/views.py`：

```py
class ShareBlogPostView(TemplateView):
    template_name = 'share_blog_post.html'

    @method_decorator(login_required)
    def dispatch(self, request, *args, **kwargs):
        return super(ShareBlogPostView, self).dispatch(request, *args, **kwargs)

    def get_context_data(self, pk, **kwargs):
        blog_post = BlogPost.objects.get(pk=pk)
        currently_shared_with = blog_post.shared_to.all()
        currently_shared_with_ids = map(lambda x: x.pk, currently_shared_with)
        exclude_from_can_share_list = [blog_post.blog.pk] + list(currently_shared_with_ids)

        can_be_shared_with = Blog.objects.exclude(pk__in=exclude_from_can_share_list)

        return {
            'post': blog_post,
            'is_shared_with': currently_shared_with,
            'can_be_shared_with': can_be_shared_with
        }
```

这个视图是模板视图的子类。到目前为止，您应该对它的工作原理有一个很好的了解。这里要看的重要部分是`get_context_data`方法内的代码。首先，我们使用从解析的 URL 模式中收集的关键字参数中传递的`id`获取博客文章对象。接下来，我们获取此帖子已经与之共享的所有博客对象的列表。我们这样做是因为我们不希望混淆用户，允许他们分享已经与之共享的博客的帖子。

下一行代码使用 Python 内置的`map`方法处理帖子共享的博客的查询集。`map`是在 Python 中处理任何类型的列表（或类似列表的对象）时最有用的方法之一。它的第一个参数是一个接受一个参数并返回一个参数的函数，第二个参数是一个列表。然后，`map`在输入列表中的每个元素上调用给定的函数，并收集结果到最终返回的列表中。在这里，我们使用`lambda`来提取此帖子已经共享的博客对象的 ID。

最后，我们可以获取可以与此帖子共享的博客对象列表。我们使用`exclude`方法来排除已经共享帖子的博客对象。我们将这些传递给模板上下文。接下来，让我们看看您需要在`blog/templates/share_blog_post.html`中创建的模板：

```py
{% extends "base.html" %}

{% block content %}
{% if can_be_shared_with %}
<h2>Share {{ post.title }}</h2>
<ul>
    {% for blog in can_be_shared_with %}
    <li><a href="{% url "share-post-with-blog" post_pk=post.pk blog_pk=blog.pk %}">{{ blog.title }}</a></li>
    {% endfor %}
</ul>
{% endif %}

{% if is_shared_with %}
<h2>Stop sharing with:</h2>
<ul>
    {% for blog in is_shared_with %}
    <li><a href="{% url "stop-sharing-post-with-blog" post_pk=post.pk blog_pk=blog.pk %}">{{ blog.title }}</a></li>
    {% endfor %}
</ul>
{% endif %}
{% endblock %}
```

这个模板中没有什么特别的。让我们继续讨论这些模板所指的两个 URL 和视图，因为没有这些，我们无法呈现这个模板。首先，让我们看看`SharepostWithBlog`，您需要在`blog/views.py`中创建它。您还需要在文件顶部添加此导入行：

```py
from django.views.generic import View
```

视图的代码如下：

```py
class SharePostWithBlog(View):
    @method_decorator(login_required)
    def dispatch(self, request, *args, **kwargs):
        return super(SharePostWithBlog, self).dispatch(request, *args, **kwargs)

    def get(self, request, post_pk, blog_pk):
        blog_post = BlogPost.objects.get(pk=post_pk)
        if blog_post.blog.owner != request.user:
            return HttpResponseForbidden('You can only share posts that you created')

        blog = Blog.objects.get(pk=blog_pk)
        blog_post.shared_to.add(blog)

        return HttpResponseRedirect(reverse('home'))
```

将其导入到`blueblog/urls.py`中，并使用以下 URL 模式添加它：

```py
url(r'blog/post/(?P<pk>\d+)/share/$', SharePostWithBlog.as_view(), name='share-blog-post-with-blog'),
```

与我们以前的所有视图不同，这个视图不太适合 Django 提供的任何通用视图中。但是 Django 有一个基本的通用视图，使我们的生活比创建处理请求的函数更容易。

当您需要完全自定义处理请求时，可以使用`View`通用视图。与所有通用视图一样，它具有一个`dispatch`方法，您可以重写以在进一步处理请求之前拦截请求。在这里，我们确保用户在允许他们继续之前已登录。

在`View`子类中，您创建与您想要处理的请求类型相同名称的方法。在这里，我们创建一个`get`方法，因为我们只关心处理`GET`请求。`View`类负责在客户端使用正确的请求方法时调用我们的方法。在我们的 get 方法中，我们正在进行基本检查，以查看用户是否拥有博客帖子。如果是，我们将博客添加到`BlogPost`模型的`shared_to ManyToMany`关系中。

我们需要创建的最后一个视图是允许用户删除他们已经共享的博客帖子。该视图的代码如下所示：

```py
class StopSharingPostWithBlog(View):
    @method_decorator(login_required)
    def dispatch(self, request, *args, **kwargs):
        return super(StopSharingPostWithBlog, self).dispatch(request, *args, **kwargs)

    def get(self, request, post_pk, blog_pk):
        blog_post = BlogPost.objects.get(pk=post_pk)
        if blog_post.blog.owner != request.user:
            return HttpResponseForbidden('You can only stop sharing posts that you created')

        blog = Blog.objects.get(pk=blog_pk)
        blog_post.shared_to.remove(blog)

        return HttpResponseRedirect(reverse('home'))
```

与`SharePostWithBlog`视图一样，这个视图是`View`通用视图的子类。代码几乎与先前的视图完全相同。唯一的区别是在先前的视图中，我们使用了`blog_post.shared_to.add`，而在这个视图中，我们使用了`blog_post.shared_to.remove`方法。

最后，将这两个视图导入到`blueblog/urls.py`中，并添加以下模式：

```py
url(r'blog/post/(?P<post_pk>\d+)/share/to/(?P<blog_pk>\d+)/$', SharePostWithBlog.as_view(), name='share-post-with-blog'),
    url(r'blog/post/(?P<post_pk>\d+)/stop/share/to/(?P<blog_pk>\d+)/$', StopSharingPostWithBlog.as_view(), name='stop-sharing-post-with-blog'),
```

为了在首页显示一个链接到分享此帖子页面，编辑`home.html`模板，将`content`块内的整个代码更改为以下内容：

```py
{% if blog_posts %}
<h2>Blog Posts</h2>
<ul>
    {% for post in blog_posts %}
    <li>
        <a href="{% url "blog-post-details" pk=post.pk %}">{{ post.title }}</a> |
        <a href="{% url "update-blog-post" pk=post.pk %}">Edit Post</a> |
        <a href="{% url "share-blog-post" pk=post.pk %}">Share Post</a>
    </li>
    {% endfor %}
</ul>
{% endif %}
```

就是这样。现在当您访问主页时，每篇博客帖子旁边应该有一个**分享帖子**链接。单击它后，您将看到第二个页面，其中包含在其他用户博客上分享博客帖子的链接。单击该链接应该分享您的帖子，并在同一页面上显示相应的删除链接。当然，为了测试这一点，您应该创建第二个用户帐户，并使用该帐户添加一个博客。

我们应该做的最后一件事是修改**HomeView**的`get_context_data`方法，以便在博客帖子列表中也包括共享的帖子：

```py
def get_context_data(self, **kwargs):
    ctx = super(HomeView, self).get_context_data(**kwargs)

    if self.request.user.is_authenticated():
            if Blog.objects.filter(owner=self.request.user).exists():
            ctx['has_blog'] = True
            blog = Blog.objects.get(owner=self.request.user)

            ctx['blog'] = blog
            ctx['blog_posts'] = BlogPost.objects.filter(blog=blog)
            ctx['shared_posts'] = blog.shared_posts.all()

    return ctx
```

将其添加到`blog/templates/home.html`模板的`content`块的底部：

```py
{% if shared_posts %}
<h2>Shared Blog Posts</h2>
<ul>
    {% for post in shared_posts %}
    <li>
        <a href="{% url "blog-post-details" pk=post.pk %}">{{ post.title }}</a>
    </li>
    {% endfor %}
</ul>
{% endif %}
{% endblock %}
```

就是这样，我们的第一个应用程序已经完成了！如果你现在打开主页，你应该会看到每篇博客文章旁边有一个**分享帖子**链接。点击这个链接应该会打开另一个页面，你可以在那里选择与哪个博客分享这篇文章。为了测试它，你应该使用我们之前创建的另一个账户创建另一个博客，当时我们正在查看我们应用程序的安全性。一旦你配置了另一个博客，你的分享博客文章页面应该看起来类似于这样：

![分享博客文章](img/00698_01_03.jpg)

点击另一个博客的标题应该会分享这篇文章并带你回到主页。如果你再次点击同一篇文章的**分享帖子**链接，你现在应该会看到一个标题，上面写着**停止与...分享**，以及你与之分享这篇文章的博客的名称。

如果你现在登录另一个账户，你应该会看到这篇文章现在已经分享到那里，并列在**分享的博客文章**部分下面。

# 总结

在本章中，我们看到了如何启动我们的应用程序并正确设置它，以便我们可以快速开发东西。我们研究了使用模板继承来实现代码重用，并为我们的网站提供导航栏等共同元素。以下是我们迄今为止涵盖的主题列表：

+   使用 sqlite3 数据库的基本项目布局和设置

+   简单的 Django 表单和模型表单用法

+   Django 贡献应用程序

+   使用`django.contrib.auth`为应用程序添加用户注册和身份验证

+   模板继承

+   用于编辑和显示数据库对象的通用视图

+   数据库迁移

我们将会在本书的其余章节中运用我们在这里学到的教训。
