# 第四章：创建一个社交网站

在上一章中，你学习了如何创建站点地图和订阅，并且为博客应用构建了一个搜索引擎。在这一章中，你会开发一个社交应用。你会为用户创建登录，登出，编辑和修改密码的功能。你会学习如何为用户创建自定义的个人资料，并在网站中添加社交认证。

本章会涉及以下知识点：

- 使用认证框架
- 创建用户注册视图
- 用自定义个人资料模型扩展`User`模型
- 用`python-social-auth`添加社交认证

让我们从创建新项目开始。

## 4.1 创建一个社交网站项目

我们将会创建一个社交应用，让用户可以分享他们在 Internet 上发现的图片。我们需要为该项目构建以下元素：

- 一个认证系统，用于用户注册，登录，编辑个人资料，修改或重置密码
- 一个关注系统，允许用户互相关注
- 显示分享的图片，并实现一个书签工具，让用户可以分享任何网站的图片
- 每个用户的活动信息，让用户可以看到他关注的用户上传的内容

本章讨论第一点。

### 4.1.1 启动社交网站项目

打开终端，使用以下命令为项目创建一个虚拟环境，并激活：

```py
mkdir env
virtualenv env/bookmarks
source env/bookmarks/bin/activate
```

终端会如下显示你激活的虚拟环境：

```py
(bookmarks)laptop:~ zenx$
```

使用以下命令，在虚拟环境中安装 Django：

```py
pip install Django
```

执行以下命令创建一个新项目：

```py
django-admin startproject bookmarks
```

创建初始项目结构之后，使用以下命令进入项目目录，并创建一个`account`的新应用：

```py
cd bookmarks/
django-admin startapp account
```

通过把该应用添加到`settings.py`文件的`INSTALLED_APPS`中，来激活它。把它放在`INSTALLED_APPS`列表的最前面：

```py
INSTALLED_APPS = (
	'account',
	# ...
)
```

执行下面的命令，同步`INSTALLED_APPS`设置中默认应用的模型到数据库中：

```py
python manage.py migrate
```

接下来，我们用`authentication`框架在项目中构建一个认证系统。

## 4.2 使用 Django 认证框架

Django 内置一个认证框架，可以处理用户认证，会话，权限和用户组。该认证系统包括常见的用户操作视图，比如登录，登出，修改密码和重置密码。

认证框架位于`django.contrib.auth`中，并且被其它 Django `contrib`包使用。记住，你已经在第一章中使用过认证框架，为博客应用创建了一个超级用户，以便访问管理站点。

当你使用`startproject`命令创建新 Django 项目时，认证框架已经包括在项目的默认设置中。它由`django.contrib.auth`应用和以下两个中间件（middleware）类组成（这两个中间类位于项目的`MIDDLEWARE_CLASSES`设置中）：

- `AuthenticationMiddleware`：使用会话管理用户和请求
- `SessionMiddleware`：跨请求处理当前会话

一个中间件是一个带有方法的类，在解析请求或响应时，这些方法在全局中执行。你会在本书的好几个地方使用中间件类。你会在第十三章学习如何创建自定义的中间件。

该认证框架还包括以下模块：

- `User`：一个有基础字典的用户模型；主要字段有：`username`，`password`，`email`，`first_name`，`last_name`和`is_active`。
- `Group`：一个用于对用户分类的组模型。
- `Permission`：执行特定操作的标识。

该框架还包括默认的认证视图和表单，我们之后会学习。

### 4.2.1 创建登录视图

我们从使用 Django 认证框架允许用户登录网站开始。我们的视图要执行以下操作来登录用户：

1. 通过提交表单获得用户名和密码。
2. 对比数据库中的数据，来验证用户。
3. 检查用户是否激活。
4. 用户登录，并开始一个认证的会话（authenticated session）。

首先，我们将创建一个登录表单。在`account`应用目录中创建`forms.py`文件，添加以下代码：

```py
from django import forms

class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)
```

该表单用于在数据库用验证用户。注意，我们使用`PasswordInput`组件来渲染包括`type="password"`属性的 HTML `input`元素。编辑`account`应用的`views.py`文件，添加以下代码：

```py
from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth import authenticate, login
from .forms import LoginForm

def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            user = authenticate(username=cd['username'],
                                password=cd['password'])
            if user is not None:
                if user.is_active:
                    login(request, user)
                    return HttpResponse('Authenticated successfully')
                else:
                    return HttpResponse('Disabled account')
            else:
                return HttpResponse('Invalid login')
    else:
        form = LoginForm()
    return render(request, 'account/login.html', {'form': form})
```

这是我们在视图中所做的基本登录操作：当使用`GET`请求调用`user_login`视图时，我们使用`form = LoginForm()`实例化一个新的登录表单，用于在模板中显示。当用户通过`POST`提交表单时，我们执行以下操作：

1. 使用`form = LoginForm(request.POST)`实例化带有提交的数据的表单。
2. 检查表单是否有效。如果无效，则在模板中显示表单错误（例如，用户没有填写某个字段）。
3. 如果提交的数据有效，我们使用`authenticate()`方法，在数据库中验证用户。该方法接收`username`和`password`参数，如果用户验证成功，则返回`User`对象，否则返回`None`。如果用户没有通过验证，我们返回一个原始的`HttpResponse`，显示一条消息。
4. 如果用户验证成功，我们通过`is_active`属性检查用户是否激活。这是 Django `User`模型的属性。如果用户没有激活，我们返回一个`HttpResponse`显示信息。
5. 如果是激活的用户，我们在网站登录用户。我们调用`login()`方法，把用户设置在 session 中，并返回一条成功消息。

> 注意`authenticate`和`login`之间的区别：`authenticate()`方法检查用户的认证信息，如果正确，则返回`User`对象；`login()`在当前 session 中设置用户。

现在，你需要为该视图创建 URL 模式。在`account`应用目录中创建`urls.py`文件，并添加以下代码：

```py
from django.conf.urls import url
from . import views

urlpatterns = [
    # post views
    url(r'^login/$', views.user_login, name='login'),
]
```

编辑`bookmarks`项目目录中的`urls.py`文件，在其中包括`account`应用的 URL 模式：

```py
from django.conf.urls import url, include
from django.contrib import admin

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^account/', include('account.urls')),
]
```

现在可以通过 URL 访问登录视图了。是时候为该视图创建一个模板了。因为该项目还没有模板，所以你可以创建一个基础模板，在登录模板中扩展它。在`account`应用目录中创建以下文件和目录：

```py
templates/
	account/
		login.html
	base.html
```

编辑`base.html`文件，添加以下代码：

```py
{% load staticfiles %}
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}{% endblock  %}</title>
    <link href="{% static "css/base.css" %}" rel="stylesheet">
</head>
<body>
    <div id="header">
        <span class="logo">Bookmarks</span>
    </div>
    <div id="content">
        {% block content %}
        {% endblock  %}
    </div>
</body>
</html>
```

这是网址的基础模板。跟之前的项目一样，我们在主模板中包括 CSS 样式。该基础模板定义了`title`和`content`区域，可以被从它扩展的模板填充内容。

让我们为登录表单创建模板。打开`account/login.html`模板，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Log-in{% endblock  %}

{% block content %}
    <h1>Log-in</h1>
    <p>Please, user the following form to log-in</p>
    <form action="." method="post">
        {{ form.as_p }}
        {% csrf_token %}
        <p><input type="submit" value="Log-in"></p>
    </form>
{% endblock  %}
```

该模板包括了在视图中实例化的表单。因为我们的表单会通过`POST`提交，所以我们使用`{% csrf_token %}`模板标签进行 CSRF 保护。你在第二章学习了 CSRF 保护。

现在数据库中还没有用户。首先，你需要创建一个超级用户，访问管理站点来管理其他用户。打开命令行，执行`python manage.py createsuperuser`。填写必需的用户名，邮箱和密码。然后使用`python manage.py runserver`启动开发服务器，并在浏览器中打开`http://127.0.0.1:8000/admin/`。使用你刚创建的用户登录管理站点。你会看到 Django 管理站点中包括了 Django 认证框架的`User`和`Group`模型，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.1.png)	

通过管理站点创建一个新用户，并在浏览器中打开`http://127.0.0.1:8000/account/login/`。你会看到包括登录表单的模板：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.2.png)

现在，提交表单时不填其中一个字段。这时，你会看到表单是无效的，并显示错误信息，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.3.png)

如果你输入一个不存在的用户，或者错误的密码，你会看到一条`Invalid login`消息。

如果你输入有效的认证信息，会看到一条`Authenticated successfully`消息，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.4.png)

### 4.2.2 使用 Django 认证视图

Django 在认证框架中包括了几个表单和视图，你可以直接使用。你已经创建的登录视图对于理解 Django 中的用户认证过程是一个很好的练习。然而，你在绝大部分情况下可以使用默认的 Django 认证视图。

Django 提供了以下视图处理认证：

- `login`：操作一个登录表单，并登录用户
- `logout`：登出一个用户
- `logout_then_login`：登出一个用户，并重定向用户到登录页面

Django 提供以下视图处理修改密码：

- `password_change`：操作一个修改用户密码的表单
- `password_change_done`：修改密码后，显示成功页面

Django 还提供以下视图用于重置密码：

- `password_reset`：允许用户重置密码。它生成一个带令牌的一次性链接，并发送到用户的电子邮箱中。
- `password_reset_done`：告诉用户，重置密码的邮件已经发送到他的邮箱中。
- `password_reset_confirm`：让用户设置新密码。
- `password_reset_complete`：用户重置密码后，显示成功页面。

创建一个带用户账户的网站时，这里列出的视图会节省你很多时间。你可以覆盖这些视图使用的默认值，比如需要渲染的模板的位置，或者视图使用的表单。

你可以在[这里](https://docs.djangoproject.com/en/1.11/topics/auth/default/#module-django.contrib.auth.views)获得更多关于内置的认证视图的信息。

### 4.2.3 登录和登出视图

编辑`account`应用的`urls.py`文件，如下所示：

```py
from django.conf.urls import url
from django.contrib.auth.views import login, logout, logout_then_login
from . import views

urlpatterns = [
    # previous login view
    # url(r'^login/$', views.user_login, name='login'),

    # login / logout urls
    url(r'^login/$', login, name='login'),
    url(r'^logout/$', logout, name='logout'),
    url(r'^logout-then-login/$', logout_then_login, name='logout_then_login'),
]
```

> **译者注：**Django 新版本中，URL 模式使用方式跟旧版本不一样。

我们注释了之前为`user_login`视图创建的 URL 模式，使用了 Django 认证框架的`login`视图。

在`account`应用的`templates`目录中创建一个`registration`目录。这是 Django 认证视图的默认路径，它期望你的认证模板在这个路径下。在新创建的目录中创建`login.html`文件，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Log-in{% endblock  %}

{% block content %}
    <h1>Log-in</h1>
    {% if form.errors %}
        <p>
            Your username and password didn't match.
            Please try again.
        </p>
    {% else %}
        <p>Please, user the following form to log-in</p>
    {% endif %}

    <div class="login-form">
        <form action="{% url 'login' %}" method="post">
            {{ form.as_p }}
            {% csrf_token %}
            <input type="hidden" name="next" value="{{ next }}" />
            <p><input type="submit" value="Log-in"></p>
        </form>
    </div>
{% endblock  %}
```

这个`login`模板跟我们之前创建那个很像。Django 默认使用`django.contrib.auth.forms`中的`AuthenticationForm`。该表单尝试验证用户，如果登录不成功，则抛出一个验证错误。这种情况下，如果认证信息出错，我们可以在模板中使用`{% if form.errors %}`查找错误。注意，我们添加了一个隐藏的 HTML `<input>`元素，用于提交名为`next`的变量的值。当你在请求中传递一个`next`参数时（比如，`http://127.0.0.1:8000/account/login/?next=/account/`），这个变量首次被登录视图设置。

`next`参数必须是一个 URL。如果指定了这个参数，Django 登录视图会在用户登录后，重定义到给定的 URL。

现在，在`registration`模板目录中创建一个`logged_out.html`模板，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Logged out{% endblock  %}

{% block content %}
    <h1>Logged out</h1>
    <p>You have been successfully logged out. You can <a href="{% url "login" %}">log-in again></a>.</p>
{% endblock  %}
```

用户登出之后，Django 会显示这个模板。

为登录和登出视图添加 URL 模式和模板后，网站已经可以使用 Django 认证视图登录了。

> 注意，我们在`urlconf`中包含的`logout_then_login`视图不需要任何模板，因为它重定义到了登录视图。

现在我们开始创建一个新的视图，当用户登录账号时，用于显示用户的仪表盘。打开`account`应用的`views.py`文件，添加以下代码：

```py
from django.contrib.auth.decorators import login_required

@login_required
def dashboard(request):
    return render(request,
                  'account/dashboard.html',
                  {'section': 'dashboard'})
```

我们用认证框架的`login_required`装饰器装饰视图。该装饰器检查当前用户是否认证。如果是认证用户，它会执行被装饰的视图。如果不是认证用户，它会重定向用户到登录 URL，并在登录 URL 中带上一个名为`next`的`GET`参数，该参数是用户试图访问的 URL。通过这样的做法，当用户成功登录后，登录视图会重定向用户到用户登录之前试图访问的页面。记住，我们在登录模板的表单中添加了一个隐藏的`<input>`元素就是为了这个目的。

我们还定义了一个`section`变量。我们用这个变量跟踪用户正在查看网站的哪一部分（section）。多个视图可能对应相同的部分。这是定义每个视图对应的 section 的简便方式。

现在，你需要为仪表盘视图创建一个模板。在`templates/account/`目录下创建`dashboard.html`文件，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Dashboard{% endblock %}

{% block content %}
    <h1>Dashboard</h1>  
    <p>Welcome to your dashboard.</p>
{% endblock  %}
```

接着，在`account`应用的`urls.py`文件中，为该视图添加 URL 模式：

```py
urlpatterns = [
	# ...
	url(r'^$', views.dashboard, name='dashboard'),
 ]
```

编辑项目的`settings.py`文件，添加以下代码：

```py
from django.core.urlresolvers import reverse_lazy

LOGIN_REDIRECT_URL = reverse_lazy('dashboard')
LOGIN_URL = reverse_lazy('login')
LOGOUT_URL = reverse_lazy('logout')
```

这些设置是：

- `LOGIN_REDIRECT_URL`：告诉 Django，如果`contrib.auth.views.login`视图没有获得`next`参数时，登录后重定向到哪个 URL
- `LOGIN_URL`：重定向用户登录的 URL（比如使用`login_required`装饰器）
- `LOGOUT_URL`：重定向用户登出的 URL

  我们使用`reverse_lazy()`，通过 URL 的名字动态创建 URL。`reverse_lazy()`函数跟`reverse()`函数一样逆向 URL。当你需要在项目 URL 配置加载之前逆向 URL 时，可以使用`reverse_lazy()`。

让我们总结一下，到现在为止，我们做了哪些工作：

- 你在项目中添加了内置的 Django 认证登录和登出视图
- 你为这两个视图创建了自定义模板，并定义了一个简单的视图，让用户登录后重定向到这个视图
- 最后，你配置了设置，让 Django 默认使用这些 URL

现在，我们需要把登录和登出链接到基础模板中，把所有功能串起来。

要做到这点，我们需要确定，无论当前用户是否登录，都能显示适当的链接。通过认证中间件，当前用户被设置在`HttpRequest`对象中。你可以通过`request.user`访问。即使用户没有认证，你也可以找到一个用户对象。一个未认证的用户在`request`中是一个`AnonymousUser`的实例。调用`request.user.is_authenticated()`是检测当前用户是否认证最好的方式。

编辑`base.html`文件，修改 ID 为`header`的`<div>`，如下所示：

```py
<div id="header">
	<span class="logo">Bookmarks</span>
	{% if request.user.is_authenticated %}
		<ul class="menu">
			<li {% if section == "dashboard" %}class="selected"{% endif %}>
				<a href="{% url "dashboard" %}">My dashboard</a>
			</li>
			<li {% if section == "images" %}class="selected"{% endif %}>
				<a href="#">Images</a>
			</li>
			<li {% if section == "people" %}class="selected"{% endif %}>
				<a href="#">People</a>
			</li>
		</ul>
	{% endif %}
	
	<span class="user">
		{% if request.user.is_authenticated %}
			Hello {{ request.user.first_name }},
			<a href="{% url "logout" %}">Logout</a>
		{% else %}
			<a href="{% url "login" %}">Log-in</a>
		{% endif %}
	</span>
</div>
```

正如你所看到的，我们只为认证的用户显示网站的菜单。我们还检查当前的 section，通过 CSS 为相应的`<li>`项添加`selected`类属性来高亮显示菜单中的当前 section。我们还显示用户的姓，如果是认证过的用户，还显示一个登出链接，否则显示登录链接。

现在，在浏览器中打开`http://127.0.0.1:8000/account/login`。你会看到登录页面。输入有效的用户名和密码，点击`Log-in`按钮，你会看到这样的页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.5.png)

因为`My dashboard`有`selected`属性，所以你会看到它是高亮显示的。因为是认证过的用户，所以用户的姓显示在头部的右边。点击`Logout`链接，你会看到下面的页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.6.png)

在这个页面中，用户已经登出，所以你不能再看到网站的菜单。现在头部右边显示`Log-in`链接。

如果你看到的是 Django 管理站点的登出页面，而不是你自己的登出页面，检查项目的`INSTALLED_APPS`设置，确保`django.contrib.admin`在`account`应用之后。这两个模板位于同样的相对路径中，Django 目录加载器会使用第一个。

### 4.2.4 修改密码视图

用户登录我们的网站后，我们需要用户可以修改他们的密码。我们通过集成 Django 认证视图来修改密码。打开`account`应用的`urls.py`文件，添加以下 URL 模式：

```py
from django.contrib.auth.views import password_change
from django.contrib.auth.views import password_change_done

# change password urls
urlpatterns = [
    url(r'^password-change/$', password_change, name='password_change'),
    url(r'^password_change/done/$', password_change_done, name='password_change_done'),
]
```

`password_change`视图会处理修改密码表单，`password_change_done`会在用户成功修改密码后显示一条成功消息。让我们为每个视图创建一个模板。

在`account`应用的`templates/registration/`目录中创建`password_change_form.html`文件，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Change you password{% endblock  %}

{% block content %}
    <h1>Change you password</h1>
    <p>Use the form below to change your password.</p>
    <form action="." method="post">
        {{ form.as_p }}
        <p><input type="submit" value="Change"></p>
        {% csrf_token %}
    </form>
{% endblock %}
```

该模板包括修改密码的表单。在同一个目录下创建`password_change_done.html`文件，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Password changed{% endblock %}

{% block content %}
    <h1>Password changed</h1>
    <p>Your password has been successfully changed.</p>
{% endblock %}
```

该模板只包括一条用户成功修改密码后显示的成功消息。

在浏览器中打开`http://127.0.0.1:8000/account/password-change/`。如果用户没有登录，浏览器会重定向到登录页面。当你认证成功后，你会看到下面的修改密码页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.7.png)

在表单中填写当前密码和新密码，点击`Change`按钮。你会看到以下成功页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.8.png)

登出后，使用新密码再次登录，确定所有功能都能正常工作。

### 4.2.5 重置密码视图

在`account`应用的`urls.py`文件中，为重置密码添加以下 URL 模式：

```py
from django.contrib.auth.views import password_reset
from django.contrib.auth.views import password_reset_done
from django.contrib.auth.views import password_rest_confirm
from django.contrib.auth.views import password_reset_complete

# restore password urls
url(r'^password-reset/$', password_reset, name='password_reset'),
url(r'^password-reset/done/$', password_reset_done, name='password_reset_done'),
url(r'^password-reset/confirm/(?P<uidb64>[-\w]+)/(?P<token>[-\w]+)/$', password_reset_confirm, name='password_reset_confirm'),
url(r'^password-reset/complete/$', password_reset_complete, name='password_reset_complete'),
```

在`account`应用的`templates/registration/`目录中创建`password_reset_form.html`文件，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Reset your password{% endblock %}

{% block content %}
    <h1>Forgotten your password?</h1>
    <p>Enter your e-mail address to obtain a new password.</p>
    <form action="." method="post">
        {{ form.as_p }}
        <p><input type="submit" value="Send e-mail"></p>
        {% csrf_token %}
    </form>
{% endblock %}
```

在同一个目录下创建`password_reset_email.html`文件，添加以下代码：

```py
Someon asked for password reset for email {{ email }}. Fllow the link below:
{{ protocol }}://{{ domain }}/{% url "password_reset_form" uidb64=uid token=token %}
Your usernmae, in case you've forgotten: {{ user.get_username }}
```

这个模板用于渲染发送给用户重置密码的邮件。

在同一个目录下创建`password_reset_done.html`文件，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Reset your password{% endblock %}

{% block content %}
    <h1>Reset your password</h1>
    <p>We've emailed you instructions for setting your password.</p>
    <p>If you don't receive an email, please make sure you've entered the address you registered with.</p>
{% endblock %}
```

创建另一个模板文件`password_reset_confirm.html`，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Reset your password{% endblock %}

{% block content %}
    <h1>Reset your password</h1>
    {% if validlink %}
        <p>Please enter your new password twice:</p>
        <form action="." method="post">
            {{ formo.as_p }}
            {% csrf_token %}
            <p><input type="submit" value="Change my password" /></p>
        </form>
    {% else %}
        <p>The password reset link was invalid, possible because it has already been used. 
            Please request a new password reset.</p>
    {% endif %}
{% endblock  %}
```

我们检查提供的链接是否有效。Django 重置页面视图设置该变量，并把它放在这个模板的上下文中。如果链接有效，我们显示重置密码表单。

创建另一个`password_reset_complete.html`文件，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Password reset{% endblock %}

{% block content %}
    <h1>Password set</h1>
    <p>Your password has been set. You can <a href="{% url "login" %}">log in now</a></p>
{% endblock %}
```

最后，编辑`account`应用的`registration/login.html`模板，在`<form>`元素后面添加以下代码：

```py
<p><a href="{% url "password_reset" %}">Forgotten your password?</a></p>
```

现在，在浏览器中打开`htth://127.0.0.1:8000/account/login/`，点击`Forgotten your password?`链接，你会看到以下链接：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.9.png)

此时，你需要在项目的`settings.py`文件中添加`SMTP`配置，让 Django 可以发送邮件。我们已经在第二章学习了如何添加邮件设置。但是在开发期间，你可以让 Django 在标准输出中写邮件，代替通过 SMTP 服务发送邮件。Django 提供了一个邮件后台，可以把邮件输出到控制台。编辑项目的`settings.py`文件，添加下面这一行代码：

```py
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
```

`EMAIL_BACKEND`设置指定用于发送邮件的类。

回到浏览器，输入已有用户的邮箱地址，点击`Send a e-mail`按钮。你会看到以下页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.10.png)

看一眼正在运行开发服务器的控制台，你会看到生成的邮件：

```py
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Subject: Password reset on 127.0.0.1:8000
From: webmaster@localhost
To: lakerszhy@gmail.com
Date: Tue, 02 May 2017 03:50:20 -0000
Message-ID: <20170502035020.7440.93778@bogon>

Someon asked for password reset for email lakerszhy@gmail.com. Fllow the link below:
http://127.0.0.1:8000/account/password-reset/confirm/Mg/4lp-4b14906c833231658e9f/
Your usernmae, in case you've forgotten: antonio
```

邮件使用我们之间创建的`password_reset_email.html`模板渲染。重置密码的 URL 包括一个 Django 动态生成的令牌。在浏览器中打开连接，会看到以下页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.11.png)

设置新密码的页面对应`password_reset_confirm.html`模板。填写新密码并点击`Change my password`按钮。Django 会创建一个新的加密密码，并保存到数据库中。你会看到一个成功页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.12.png)

现在你可以使用新密码再次登录。每个用于设置新密码的令牌只能使用一次。如果你再次打开收到的链接，会看到一条令牌无效的消息。

你已经在项目中集成了 Django 认证框架的视图。这些视图适用于大部分场景。如果需要不同的行为，你可以创建自己的视图。

## 4.3 用户注册和用户资料

现在，已存在的用户可以登录，登出和修改密码，如果用户忘记密码，可以重置密码。现在，我们需要创建视图，用于游客创建账户。

### 4.3.1 用户注册

让我们创建一个简单的视图，允许用户在我们的网站注册。首先，我们必须创建一个表单，让用户输入用户名，姓名和密码。编辑`account`应用中的`forms.py`文件，添加以下代码：

```py
from django.contrib.auth.models import User

class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(label='Password', widget=forms.PasswordInput)
    password2 = forms.CharField(label='Repeat Password', widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ('username', 'first_name', 'email')

    def clean_password2(self):
        cd = self.cleaned_data
        if cd['password'] != cd['password2']:
            raise forms.ValidationError("Passwords don't match.")
        return cd['password2']
```

我们为`User`模型创建了一个模型表单。在表单中，我们只包括了模型的`username`，`first_name`，`email`字段。这些字段会根据相应的模型字段验证。例如，如果用户选择了一个已存在的用户名，会得到一个验证错误。我们添加了两个额外字段：`password`和`password2`，用来设置密码和确认密码。我们定义了`clean_password2()`方法，检查两次输入的密码是否一致，如果不一致，则让表单无效。当我们调用表单的`is_valid()`方法验证时，这个检查会执行。你可以为任何表单字段提供`clean_<fieldname>()`方法，清理特定字段的值或抛出表单验证错误。表单还包括一个通用的`clean()`方法验证整个表单，验证相互依赖的字段时非常有用。

Django 还在`django.contrib.auth.forms`中提供了`UserCreationForm`表单供你使用，这个表单跟我们刚创建的表单类似。

编辑`account`应用中的`views.py`文件，添加以下代码：

```py
from .forms import LoginForm, UserRegistrationForm

def register(request):
    if request.method == 'POST':
        user_form = UserRegistrationForm(request.POST)
        if user_form.is_valid():
            # Create a new user object but avoid saving it yet
            new_user = user_form.save(commit=False)
            # Set the chosen password
            new_user.set_password(user_form.cleaned_data['password'])
            # Save the User object
            new_user.save()
            return render(request, 'account/register_done.html', {'new_user': new_user})
    else:
        user_form = UserRegistrationForm()
    return render(request, 'account/register.html', {'user_form': user_form})
```

这个创建用户账户的视图非常简单。为了安全，我们使用`User`模型的`set_password()`方法处理加密保存，来代替保存用户输入的原始密码。

现在编辑`account`应用的`urls.py`文件，添加以下 URL 模式：

```py
url(r'^register/$', views.register, name='register')
```

最后，我们在`account/`模板目录中创建`register.html`文件，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Create an account{% endblock %}

{% block content %}
    <h1>Create an account</h1>
    <p>Please, sign up using the following form:</p>
    <form action="." method="post">
        {{ user_form.as_p }}
        {% csrf_token %}
        <p><input type="submit" value="Create my account"></p>
    </form>
{% endblock %}
```

在同一个目录中添加`register_done.html`模板文件，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Welcome{% endblock %}

{% block content %}
    <h1>Welcome {{ new_user.first_name }}!</h1>
    <p>Your account has been successfully created. Now you can <a href="{% url "login" %}">log in</a>.</p>
{% endblock %}
```

现在，在浏览器中打开`http://127.0.0.1:8000/account/register/`，你会看到刚创建的注册页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.13.png)

为新用户填写信息，点击`Create my account`按钮。如果所有字段都有效，则会创建用户，你会看到下面的成功消息：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.14.png)

点击`log in`链接，输入你的用户名和密码验证能否访问你的账户。

现在，你还可以在登录模板中添加注册链接。编辑`registration/login.html`模板，把这行代码：

```py
<p>Please, user the following form to log-in</p>
```

替换为：

```py
<p>Please, user the following form to log-in. 
If you don't have an account <a href="{% url "register" %}">register here</a></p>
```

我们可以通过登录页面访问注册页面了。

### 4.3.2 扩展 User 模型

当你必须处理用户账户时，你会发现 Django 认证框架的`User`模型适用于常见情况。但是`User`模型有非常基础的字段。你可能希望扩展`User`模型包含额外的数据。最好的方式是创建一个包括所有额外字段的个人资料模型，并且与 Django 的`User`模型是一对一的关系。

编辑`account`应用的`models.py`文件，添加以下代码：

```py
from django.db import models
from django.conf import settings

class Profile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL)
    date_of_birth = models.DateField(blank=True, null=True)
    photo = models.ImageField(upload_to='users/%Y/%m/%d', blank=True)

    def __str__(self):
        return 'Pofile for User {}'.format(self.user.username)
```

> 为了让代码保持通用性，请使用`get_user_model()`方法检索用户模型。同时，定义模型和用户模型之间的关系时，使用`AUTH_USER_MODEL`设置引用用户模型，而不是直接引用该用户模型。

一对一的`user`字段允许我们用用户关联个人资料。`photo`字段是一个`ImageField`字段。你需要安装 PIL（Python Imaging Library）或 Pillow（PIL 的一个分支）Python 包来管理图片。在终端中执行以下命令安装 Pillow：

```py
pip install Pillow
```

为了在 Django 开发服务器中提供多媒体文件上传功能，需要在项目的`settings.py`文件中添加以下设置：

```py
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media/')
```

`MEDIA_URL`是用户上传的多媒体文件的基 URL，`MEDIA_ROOT`是多媒体文件的本地路径。我们根据项目路径动态构建该路径，让代码更通用。

现在，编辑`bookmarks`项目的主`urls.py`文件，如下所示修改代码：

```py
from django.conf import settings
from django.conf.urls.static import static

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

这样，Django 开发服务器将在开发过程中负责多媒体文件服务。

`static()`帮助函数只适用于开发环境，不适合生产环境。永远不要在生产环境使用 Django 为静态文件提供服务。

打开终端执行以下命令，为新模型创建数据库迁移：

```py
python manage.py makemigrations
```

你会得到这样的输出：

```py
Migrations for 'account':
  account/migrations/0001_initial.py
    - Create model Profile
```

接着使用以下命令同步数据库：

```py
python manage.py migrate
```

你会看到包括下面这一样的输出：

```py
Applying account.0001_initial... OK
```

编辑`account`应用的`admin.py`文件，在管理站点注册`Profile`模型，如下所示：

```py
from .models import Profile

class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'date_of_birth', 'photo')

admin.site.register(Profile, ProfileAdmin)
```

使用`python manage.py runserver`命令运行开发服务器。现在，你会在项目的管理站点看到`Profile`模型，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.15.png)

现在，我们将让用户在网站上编辑个人资料。在`account`应用的`forms.py`文件中添加以下模型表单：

```py
from .models import Profile

class UserEditForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email')

class ProfileEditForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ('date_of_birth', 'photo')
```

这些表单的作用是：

- `UserEditForm`：允许用户编辑存在内置的`User`模型中的姓，名和邮箱。
- `ProfileEditForm`：允许用户编辑存在自定义的`Profile`模型中的额外数据。用户可以编辑出生日期，并上传一张图片。

编辑`account`应用的`views.py`文件，导入`Profile`模型：

```py
from .models import Profile
```

在`register`视图的`new_user.save()`下面添加以下代码：

```py
# Create the user profile
profile = Profile.objects.create(user=new_user)
```

当用户在我们网站注册时，我们会创建一个空的个人资料关联到用户。你需要使用管理站点手动为之前创建的用户创建`Profile`对象。

现在我们让用户可以编辑个人资料。添加以下代码到同一个文件中：

```py
from .forms import LoginForm, UserRegistrationForm, UserEditForm, ProfileEditForm

@login_required
def edit(request):
    if request.method == 'POST':
        user_form = UserEditForm(instance=request.user, data=request.POST)
        profile_form = ProfileEditForm(instance=request.user.profile, 
                                       data=request.POST,
                                       files=request.FILES)
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
    else:
        user_form = UserEditForm(instance=request.user)
        profile_form = ProfileEditForm(instance=request.user.profile)

    return render(request, 'account/edit.html', {'user_form': user_form, 'profile_form': profile_form})
```

我们使用了`login_required`装饰器，因为用户必须认证后才能编辑个人资料。在这里，我们使用了两个模型表单：`UserEditForm`存储内置的`User`模型数据，`ProfileEditForm`存储额外的个人数据。我们检查两个表单的`is_valid()`方法返回 True 来验证提交的数据。在这里，我们保持两个表单，用来更新数据库中相应的对象。

在`account`应用的`urls.py`文件中添加以下 URL 模式：

```py
url(r'^edit/$', views.edit, name='edit')
```

最后，在`templates/account/`目录中，为该视图创建`edit.html`模板，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Edit your account{% endblock %}

{% block content %}
    <h1>Edit your account</h1>
    <p>You can edit your account using the following form:</p>
    <form action="." method='post' enctype="multipart/form-data">
        {{ user_form.as_p }}
        {{ profile_form.as_p }}
        {% csrf_token %}
        <p><input type="submit" value="Save changes"></p>
    </form>
{% endblock %}
```

> 我们在表单中包括了`enctype="multipart/form-data"`，来启用文件上传。我们使用一个 HTML 表单提交`user_form`和`profile_form`两个表单。

注册一个新用户，并在浏览器中打开`http://127.0.0.1:8000/account/edit/`，你会看到以下界面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.16.png)

现在你可以编辑仪表盘页面，来包括编辑个人资料和修改密码的页面链接。打开`account/dashboard.html`模板，把这一行代码：

```py
<p>Welcome to your dashboard.</p>
```

替换为：

```py
<p>
	Welcome to your dashboard. 
	You can <a href="{% url "edit" %}">edit your profiles</a> 
	or <a href="{% url "password_change" %}">change your password</a>. 
</p>
```

用户现在可以通过仪表盘访问编辑个人资料的表单。

#### 4.3.2.1 使用自定义 User 模型

Django 还提供了方式，可以用自定义模型代替整个`User`模型。你的用户类应从 Django 的`AbstractUser`类继承，它作为一个抽象模型，提供了默认用户的完整实现。你可以在[这里](https://docs.djangoproject.com/en/1.11/topics/auth/customizing/#substituting-a-custom-user-model)阅读更多关于这个模型的信息。

使用自定义用户模型会有更多的灵活性，但它也可能给一些需要与`User`模型交互的可插拔应用应用的集成带来一定的困难。

### 4.3.3 使用消息框架

处理用户动作时，你可能想要通知用户动作的结果。Django 内置一个消息框架，允许你显示一次性提示。该消息框架位于`django.contrib.message`中，当你用`python manage.py startproject`创建新项目时，它默认包括在`settings.py`的`INSTALLED_APPS`列表中。你注意到，设置文件的`MIDDLEWARE_CLASSES`设置列表中，包括一个名为`django.contrib.message.middleware.MessageMiddleware`的中间件。该消息框架提供了一种简单的方式来给用户添加消息。消息存储在数据库中，并会在用户下次请求时显示。你可以通过导入消息模块，使用简单的快捷方式添加新消息，来在视图中使用消息框架，如下所示：

```py
from django.contrib import message
message.error(request, 'Something went wrong')
```

你可以使用`add_message()`方法，或者以下任何一个快捷方法创建新消息：

- `success()`：动作执行成功后显示成功消息
- `info()`：信息消息
- `waring()`：还没有失败，但很可能马上失败
- `error()`：一个不成功的操作，或某些事情失败
- `debug()`：调试信息，会在生产环境移除或忽略

让我们显示消息给用户。因为消息框架对项目来说是全局的，所以我们可以在基础模板中显示消息给用户。打开`base.html`模板，在 id 为 header 和 content 的`<div>`元素之间添加以下代码：

```py
{% if messages %}
	<ul class="messages">
		{% for message in messages %}
			<li class="{{ message.tags }}">
				{{ message|safe }}
				<a href="#" class="close">✖</a>
			</li>
		{% endfor %}
	</ul>
{% endif %}
```

消息框架包括一个上下文处理器（context processor），它会添加`messages`变量到请求上下文中。因此，你可以在模板使用该变量显示当前消息。

现在，让我们修改`edit`视图来使用消息框架。编辑`account`应用的`views.py`文件，如下修改`edit`视图：

```py
from django.contrib import messages

@login_required
def edit(request):
    if request.method == 'POST':
    # ...
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Profile updated successfully')
        else:
            messages.error(request, 'Error updating your profile')
    else:
        user_form = UserEditForm(instance=request.user)
        # ...
```

当用户成功更新个人资料后，我们添加一条成功消息。如果任何一个表单无效，我们添加一条错误消息。

在浏览器中打开`http://127.0.0.1:8000/account/edit/`，并编辑你的个人资料。当个人资料更新成功后，你会看到以下消息：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.17.png)

当表单无效时，你会看到以下消息：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE4.18.png)

## 4.4 创建自定义认证后台

Django 允许你针对不同来源进行身份验证。`AUTHENTICATION_BACKENDS`设置包括了项目的认证后台列表。默认情况下，该设置为：

```py
('django.contrib.auth.backends.ModelBackend',)
```

默认的`ModelBackend`使用`django.contrib.auth`的`User`模型，验证数据库中的用户。这适用于大部分项目。但是你可以创建自定义的后台，来验证其它来源的用户，比如一个 LDAP 目录或者其它系统。

你可以在[这里](https://docs.djangoproject.com/en/1.11/topics/auth/customizing/#other-authentication-sources)阅读更多关于自定义认证的信息。

一旦你使用`django.contrib.auth`中的`authenticate()`函数，Django 会一个接一个尝试`AUTHENTICATION_BACKENDS`中定义的每一个后台来验证用户，直到其中一个验证成功。只有所有后台都验证失败，才不会在站点中验证通过。

Django 提供了一种简单的方式来定义自己的认证后台。一个认证后台是提供了以下两个方法的类：

- `authenticate()`：接收用户信息作为参数，如果用户认证成功，则返回 True，否则返回 False。
- `get_user()`：接收用户 ID 作为参数，并返回一个`User`对象。

创建一个自定义认证后台跟编写一个实现这两个方法的 Python 类一样简单。我们会创建一个认证后台，让用户使用邮箱地址代替用户名验证。

在`account`应用目录中创建一个`authentication.py`文件，添加以下代码：

```py
from django.contrib.auth.models import User

class EmailAuthBackend:
    """
    Authenticates using e-mail account.
    """
    def authenticate(self, username=None, password=None):
        try:
            user = User.objects.get(email=username)
            if user.check_password(password):
                return user
            return None
        except User.DoesNotExist:
            retur None

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
```

这是一个很简单的认证后台。`authenticate()`方法接收`username`和`password`作为可选参数。我们可以使用不同的参数，但我们使用`username`和`password`确保后台可以立即在认证框架中工作。上面的代码完成以下工作：

- `authenticate()`：我们尝试使用给定的邮箱地址检索用户，并用`User`模型内置的`check_password()`方法检查密码。该方法会处理密码哈希化，并比较给定的密码和数据库中存储的密码。
- `get_user()`：我们通过`user_id`参数获得一个用户。在用户会话期间，Django 使用认证用户的后台来检索`User`对象。

编辑项目的`settings.py`，添加以下设置：

```py
AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',
    'account.authentication.EmailAuthBackend',
)
```

我们保留了默认的`ModelBackend`，使用用户名和密码认证，并包括了自己的基于邮箱地址的认证后台。现在，在浏览器中打开`http://127.0.0.1/8000/account/login/`。记住，Django 会试图使用每一个后台验证用户，所以你现在可以使用用户名或邮箱账号登录。

> `AUTHENTICATION_ BACKENDS`设置中列出的后端顺序很重要。如果同样的信息对于多个后台都有效，Django 会在第一个成功验证用户的后台停止。

## 4.5 为网站添加社交认证

你可能还想为网站添加社交认证，比如使用 Facebook，Twitter 或 Google 服务认证。`Python-socail-auth`是一个 Python 模块，可以简化添加社交认证过程。通过这个模块，你可以让用户使用其他服务的账户登录你的网站。

> **译者注：**从 2016 年 12 月 3 日开始，这个模块迁移到了[Python Social Auth](https://github.com/python-social-auth)。原书的内容已经过时，所以就不翻译了。

## 4.6 总结

在本章中，你学习了如何在网站中构建认证系统和创建自定义用户资料。你还在网站中添加了社交认证。

下一章中，你会学习如何创建一个图片书签系统，生成图片的缩略图，以及创建 AJAX 视图。













