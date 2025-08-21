# 第二章：将用户添加到 MyMDB

在上一章中，我们启动了我们的项目并创建了我们的`core`应用程序和我们的`core`模型（`Movie`和`Person`）。在本章中，我们将在此基础上做以下事情：

+   让用户注册、登录和退出

+   让已登录用户对电影进行投票

+   根据投票为每部电影评分

+   使用投票来推荐前 10 部电影。

让我们从管理用户开始这一章。

# 创建`user`应用程序

在本节中，您将创建一个名为`user`的新 Django 应用程序，将其注册到您的项目中，并使其管理用户。

在第一章 *构建 MyMDB* 的开头，您了解到 Django 项目由许多 Django 应用程序组成（例如我们现有的`core`应用程序）。Django 应用程序应提供明确定义和紧密范围的行为。将用户管理添加到我们的`core`应用程序中违反了这一原则。让一个 Django 应用程序承担太多责任会使测试和重用变得更加困难。例如，我们将在本书中的整个过程中重用我们在这个`user` Django 应用程序中编写的代码。

# 创建一个新的 Django 应用程序

在我们创建`core`应用程序时所做的一样，我们将使用`manage.py`来生成我们的`user`应用程序：

```py
$ cd django
$ python manage.py startapp user
$ cd user
$ ls
__init__.py     admin.py        apps.py         migrations      models.py       tests.py        views.py
```

接下来，我们将通过编辑我们的`django/config/settings.py`文件并更新`INSTALLED_APPS`属性来将其注册到我们的 Django 项目中：

```py
INSTALLED_APPS = [
    'user',  # must come before admin
    'core',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]
```

出于我们将在*登录和退出*部分讨论的原因，我们需要将`user`放在`admin`应用程序之前。通常，将我们的应用程序放在内置应用程序之上是一个好主意。

我们的`user`应用程序现在是我们项目的一部分。通常，我们现在会继续为我们的应用程序创建和定义模型。但是，由于 Django 内置的`auth`应用程序，我们已经有了一个可以使用的用户模型。

如果我们想使用自定义用户模型，那么我们可以通过更新`settings.py`并将`AUTH_USER_MODEL`设置为模型的字符串 python 路径来注册它（例如，`AUTH_USER_MODEL=myuserapp.models.MyUserModel`）。

接下来，我们将创建我们的用户注册视图。

# 创建用户注册视图

我们的`RegisterView`类将负责让用户注册我们的网站。如果它收到一个`GET`请求，那么它将向用户显示`UserCreationFrom`；如果它收到一个`POST`请求，它将验证数据并创建用户。`UserCreationForm`由`auth`应用程序提供，并提供了一种收集和验证注册用户所需数据的方式；此外，如果数据有效，它还能保存一个新的用户模型。

让我们将我们的视图添加到`django/user/views.py`中：

```py
from django.contrib.auth.forms import (
    UserCreationForm,
)
from django.urls import (
    reverse_lazy,
)
from django.views.generic import (
    CreateView,
)

class RegisterView(CreateView):
    template_name = 'user/register.html'
    form_class = UserCreationForm
    success_url = reverse_lazy(
        'core:MovieList')
```

让我们逐行查看我们的代码：

+   `class RegisterView(CreateView):`：我们的视图扩展了`CreateView`，因此不必定义如何处理`GET`和`POST`请求，我们将在接下来的步骤中讨论。

+   `template_name = 'user/register.html'`：这是一个我们将创建的模板。它的上下文将与我们以前看到的有些不同；它不会有`object`或`object_list`变量，但会有一个`form`变量，它是`form_class`属性中设置的类的实例。

+   `form_class = UserCreationForm`：这是这个`CreateView`应该使用的表单类。更简单的模型可以只说`model = MyModel`，但是用户稍微复杂一些，因为密码需要输入两次然后进行哈希处理。我们将在第三章 *海报、头像和安全* 中讨论 Django 如何存储密码。

+   `success_url = reverse_lazy('core:MovieList')`：当模型创建成功时，这是您需要重定向到的 URL。这实际上是一个可选参数；如果模型有一个名为`model.get_absolute_url()`的方法，那么将使用该方法，我们就不需要提供`success_url`。

`CreateView`的行为分布在许多基类和 mixin 中，它们通过方法相互作用，作为我们可以重写以改变行为的挂钩。让我们来看看一些最关键的点。

如果`CreateView`收到`GET`请求，它将呈现表单的模板。 `CreateView`的祖先之一是`FormMixin`，它重写了`get_context_data()`来调用`get_form()`并将表单实例添加到我们模板的上下文中。 渲染的模板作为响应的主体由`render_to_response`返回。

如果`CreateView`收到`POST`请求，它还将使用`get_form()`来获取表单实例。 表单将被*绑定*到请求中的`POST`数据。 绑定的表单可以验证其绑定的数据。 `CreateView`然后将调用`form.is_valid()`，并根据需要调用`form_valid()`或`form_invalid()`。 `form_valid()`将调用`form.save()`（将数据保存到数据库）然后返回一个 302 响应，将浏览器重定向到`success_url`。 `form_invalid()`方法将使用包含错误消息的表单重新呈现模板，供用户修复并重新提交。

我们还第一次看到了`reverse_lazy()`。 它是`reverse()`的延迟版本。 延迟函数是返回值直到使用时才解析的函数。 我们不能使用`reverse()`，因为视图类在构建完整的 URLConfs 集时进行评估，所以如果我们需要在视图的*类*级别使用`reverse()`，我们必须使用`reverse_lazy()`。 值直到视图返回其第一个响应才会解析。

接下来，让我们为我们的视图创建模板。

# 创建 RegisterView 模板

在编写带有 Django 表单的模板时，我们必须记住 Django 不提供`<form>`或`<button type='submit>`标签，只提供表单主体的内容。 这让我们有可能在同一个`<form>`中包含多个 Django 表单。 有了这个想法，让我们将我们的模板添加到`django/user/templates/user/register.html`中：

```py
{% extends "base.html" %}

{% block main %}
  <h1>Register for MyMDB</h1>
  <form method="post">
    {{ form.as_p}}
    {% csrf_token %}
    <button
        type="submit"
        class="btn btn-primary">
      Register
    </button>
  </form>
{% endblock %}
```

与我们之前的模板一样，我们扩展`base.html`并将我们的代码放在现有`block`之一中（在这种情况下是`main`）。 让我们更仔细地看看表单是如何呈现的。

当表单呈现时，它分为两部分，首先是一个可选的`<ul class='errorlist'>`标签，用于一般错误消息（如果有的话），然后每个字段分为四个基本部分：

+   一个带有字段名称的`<label>`标签

+   一个`<ul class="errorlist">`标签，显示用户先前表单提交的错误；只有在该字段有错误时才会呈现

+   一个`<input>`（或`<select>`）标签来接受输入

+   一个`<span class="helptext">`标签，用于字段的帮助文本

`Form`带有以下三个实用方法来呈现表单：

+   `as_table()`: 每个字段都包裹在一个`<tr>`标签中，标签中包含一个`<th>`标签和一个包裹在`<td>`标签中的小部件。 不提供包含的`<table>`标签。

+   `as_ul`: 整个字段（标签和帮助文本小部件）都包裹在一个`<li>`标签中。 不提供包含的`<ul>`标签。

+   `as_p`: 整个字段（标签和帮助文本小部件）都包裹在一个`<p>`标签中。

对于相同的表单，不提供包含`<table>`和`<ul>`标签，也不提供`<form>`标签，以便在必要时更容易一起输出多个表单。

如果您想对表单呈现进行精细的控制，`Form`实例是可迭代的，在每次迭代中产生一个`Field`，或者可以按名称查找为`form["fieldName"]`。

在我们的示例中，我们使用`as_p()`方法，因为我们不需要精细的布局控制。

这个模板也是我们第一次看到`csrf_token`标签。 CSRF 是 Web 应用程序中常见的漏洞，我们将在第三章中更多地讨论它，*海报、头像和安全性*。 Django 自动检查所有`POST`和`PUT`请求是否有有效的`csrfmiddlewaretoken`和标头。 缺少这个的请求甚至不会到达视图，而是会得到一个`403 Forbidden`的响应。

现在我们有了模板，让我们在我们的 URLConf 中为我们的视图添加一个`path()`对象。

# 添加到 RegisterView 的路径

我们的`user`应用程序没有`urls.py`文件，所以我们需要创建`django/user/urls.py`文件：

```py
from django.urls import path

from user import views

app_name = 'user'
urlpatterns = [
    path('register',
         views.RegisterView.as_view(),
         name='register'),
]
```

接下来，我们需要在`django/config/urls.py`的根 URLConf 中`include()`此 URLConf：

```py
from django.urls import path, include
from django.contrib import admin

import core.urls
import user.urls

urlpatterns = [
    path('admin/', admin.site.urls),
    path('user/', include(
        user.urls, namespace='user')),
    path('', include(
        core.urls, namespace='core')),
]
```

由于 URLConf 只会搜索直到找到*第一个*匹配的`path`，因此我们总是希望将没有前缀或最广泛的 URLConfs 的`path`放在最后，以免意外阻止其他视图。

# 登录和登出

Django 的`auth`应用程序提供了用于登录和注销的视图。将此添加到我们的项目将是一个两步过程：

1.  在`user` URLConf 中注册视图

1.  为视图添加模板

# 更新用户 URLConf

Django 的`auth`应用程序提供了许多视图，以帮助简化用户管理和身份验证，包括登录/注销、更改密码和重置忘记的密码。一个功能齐全的生产应用程序应该为用户提供所有三个功能。在我们的情况下，我们将限制自己只提供登录和注销。

让我们更新`django/user/urls.py`以使用`auth`的登录和注销视图：

```py
from django.urls import path
from django.contrib.auth import views as auth_views

from user import views

app_name = 'user'
urlpatterns = [
    path('register',
         views.RegisterView.as_view(),
         name='register'),
    path('login/',
         auth_views.LoginView.as_view(),
         name='login'),
    path('logout/',
         auth_views.LogoutView.as_view(),
         name='logout'),
]
```

如果您提供了登录/注销、更改密码和重置密码，则可以使用`auth`的 URLConf，如下面的代码片段所示：

```py
from django.contrib.auth import urls
app_name = 'user'
urlpatterns = [
    path('', include(urls)),
]
```

现在，让我们添加模板。

# 创建一个 LoginView 模板

首先，在`django/user/templates/registration/login.html`中为登录页面添加模板：

```py
{% extends "base.html" %}

{% block title %}
Login - {{ block.super }}
{% endblock %}

{% block main %}
  <form method="post">
    {% csrf_token %}
    {{ form.as_p }}
    <button
        class="btn btn-primary">
      Log In
    </button>
  </form>
{% endblock %}
```

前面的代码看起来与`user/register.html`非常相似。

但是，当用户登录时应该发生什么？

# 成功的登录重定向

在`RegisterView`中，我们能够指定成功后将用户重定向到何处，因为我们创建了视图。`LoginView`类将按照以下步骤决定将用户重定向到何处：

1.  如果`POST`参数`next`是一个有效的 URL，并指向托管此应用程序的服务器，则使用`POST`参数`next`。`path()`名称不可用。

1.  如果`next`是一个有效的 URL，并指向托管此应用程序的服务器，则使用`GET`参数`next`。`path()`名称不可用。

1.  `LOGIN_REDIRECT_URL`设置默认为`'/accounts/profile/'`。`path()`名称*可用*。

在我们的情况下，我们希望将所有用户重定向到电影列表，所以让我们更新`django/config/settings.py`以设置`LOGIN_REDIRECT_URL`：

```py
LOGIN_REDIRECT_URL = 'core:MovieList'
```

但是，如果有情况需要将用户重定向到特定页面，我们可以使用`next`参数将其专门重定向到特定页面。例如，如果用户尝试在登录之前执行操作，我们将他们所在的页面传递给`LoginView`作为`next`参数，以便在登录后将他们重定向回所在的页面。

现在，当用户登录时，他们将被重定向到我们的电影列表视图。接下来，让我们为注销视图创建一个模板。

# 创建一个 LogoutView 模板

`LogoutView`类的行为有些奇怪。如果它收到一个`GET`请求，它将注销用户，然后尝试呈现`registration/logged_out.html`。`GET`请求修改用户状态是不寻常的，因此值得记住这个视图有点不同。

`LogoutView`类还有另一个问题。如果您没有提供`registration/logged_out.html`模板，并且已安装`admin`应用程序，则 Django *可能*会使用`admin`的模板，因为`admin`应用程序确实有该模板（退出`admin`应用程序，您会看到它）。

Django 将模板名称解析为文件的方式是一个三步过程，一旦找到文件，就会停止，如下所示：

1.  Django 遍历`settings.TEMPLATES`中`DIRS`列表中的目录。

1.  如果`APP_DIRS`为`True`，则它将遍历`INSTALLED_APPS`中列出的应用程序，直到找到匹配项。如果`admin`在`INSTALLED_APPS`列表中出现在`user`之前，那么它将首先匹配。如果`user`在前面，`user`将首先匹配。

1.  引发`TemplateDoesNotExist`异常。

这就是为什么我们把`user`放在已安装应用程序列表的第一位，并添加了一个警告未来开发人员不要改变顺序的注释。

我们现在已经完成了我们的`user`应用程序。让我们回顾一下我们取得了什么成就。

# 快速回顾本节

我们创建了一个`user`应用来封装用户管理。在我们的`user`应用中，我们利用了 Django 的`auth`应用提供的许多功能，包括`UserCreationForm`、`LoginView`和`LogoutView`类。我们还了解了 Django 提供的一些新的通用视图，并结合`UserCreationForm`类使用`CreateView`来创建`RegisterView`类。

现在我们有了用户，让我们允许他们对我们的电影进行投票。

# 让用户对电影进行投票

像 IMDB 这样的社区网站的一部分的乐趣就是能够对我们喜欢和讨厌的电影进行投票。在 MyMDB 中，用户将能够为电影投票，要么是![](img/0c9c7943-1c53-4ac8-9020-ef1441b7b361.png)，要么是![](img/0c525625-fa6c-4afe-a5a6-49f32529c098.png)。一部电影将有一个分数，即![](img/0c9c7943-1c53-4ac8-9020-ef1441b7b361.png)的数量减去![](img/0c525625-fa6c-4afe-a5a6-49f32529c098.png)的数量。

让我们从投票的最重要部分开始：`Vote`模型。

# 创建 Vote 模型

在 MyMDB 中，每个用户可以对每部电影投一次票。投票可以是正面的—![](img/99b1dc6e-8da4-459c-ac70-7ea8c550c455.png)—或者是负面的—![](img/184747c1-7c19-4599-8fd8-d8319dab8d5c.png)。

让我们更新我们的`django/core/models.py`文件来拥有我们的`Vote`模型：

```py
class Vote(models.Model):
    UP = 1
    DOWN = -1
    VALUE_CHOICES = (
        (UP, "",),
        (DOWN, "",),
    )

    value = models.SmallIntegerField(
        choices=VALUE_CHOICES,
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE
    )
    movie = models.ForeignKey(
        Movie,
        on_delete=models.CASCADE,
    )
    voted_on = models.DateTimeField(
        auto_now=True
    )

    class Meta:
        unique_together = ('user', 'movie')
```

这个模型有以下四个字段：

+   `value`，必须是`1`或`-1`。

+   `user`是一个`ForeignKey`，它通过`settings.AUTH_USER_MODEL`引用`User`模型。Django 建议您永远不要直接引用`django.contrib.auth.models.User`，而是使用`settings.AUTH_USER_MODEL`或`django.contrib.auth.get_user_model()`。

+   `movie`是一个引用`Movie`模型的`ForeignKey`。

+   `voted_on`是一个带有`auto_now`启用的`DateTimeField`。`auto_now`参数使模型在每次保存模型时更新字段为当前日期时间。

`unique_together`属性的`Meta`在表上创建了一个唯一约束。唯一约束将防止两行具有相同的`user`和`movie`值，强制执行我们每个用户每部电影一次投票的规则。

让我们为我们的模型创建一个迁移，使用`manage.py`：

```py
$ python manage.py makemigrations core
Migrations for 'core':
  core/migrations/0003_auto_20171003_1955.py
    - Create model Vote
    - Alter field rating on movie
    - Add field movie to vote
    - Add field user to vote
    - Alter unique_together for vote (1 constraint(s))
```

然后，让我们运行我们的迁移：

```py
$ python manage.py migrate core
Operations to perform:
  Apply all migrations: core
Running migrations:
  Applying core.0003_auto_20171003_1955... OK
```

现在我们已经设置好了我们的模型和表，让我们创建一个表单来验证投票。

# 创建 VoteForm

Django 的表单 API 非常强大，让我们可以创建几乎任何类型的表单。如果我们想创建一个任意的表单，我们可以创建一个扩展`django.forms.Form`的类，并向其中添加我们想要的字段。然而，如果我们想构建一个代表模型的表单，Django 为我们提供了一个快捷方式，即`django.forms.ModelForm`。

我们想要的表单类型取决于表单将被放置的位置以及它将如何被使用。在我们的情况下，我们想要一个可以放在`MovieDetail`页面上的表单，并让它给用户以下两个单选按钮：![](img/05b14743-dedd-4122-97df-cc15869422be.png)和![](img/73102249-cbaf-442e-a8f8-7ba208bb4348.png)。

让我们来看看可能的最简单的`VoteForm`：

```py
from django import forms

from core.models import Vote

class VoteForm(forms.ModelForm):
    class Meta:
        model = Vote
        fields = (
            'value', 'user', 'movie',)
```

Django 将使用`value`、`user`和`movie`字段从`Vote`模型生成一个表单。`user`和`movie`将是使用`<select>`下拉列表选择正确值的`ModelChoiceField`，而`value`是一个使用`<select>`下拉小部件的`ChoiceField`，这不是我们默认想要的。

`VoteForm`将需要`user`和`movie`。由于我们将使用`VoteForm`来保存新的投票，我们不能消除这些字段。然而，让用户代表其他用户投票将会创建一个漏洞。让我们自定义我们的表单来防止这种情况发生：

```py
from django import forms
from django.contrib.auth import get_user_model

from core.models import Vote, Movie

class VoteForm(forms.ModelForm):

    user = forms.ModelChoiceField(
        widget=forms.HiddenInput,
        queryset=get_user_model().
            objects.all(),
        disabled=True,
    )
    movie = forms.ModelChoiceField(
        widget=forms.HiddenInput,
        queryset=Movie.objects.all(),
        disabled=True
    )
    value = forms.ChoiceField(
        label='Vote',
        widget=forms.RadioSelect,
        choices=Vote.VALUE_CHOICES,
    )

    class Meta:
        model = Vote
        fields = (
            'value', 'user', 'movie',)
```

在前面的表单中，我们已经自定义了字段。

让我们仔细看一下`user`字段：

+   `user = forms.ModelChoiceField(`: `ModelChoiceField`接受另一个模型作为该字段的值。通过提供有效选项的`QuerySet`实例来验证模型的选择。

+   `queryset=get_user_model().objects.all(),`：定义此字段的有效选择的`QuerySet`。在我们的情况下，任何用户都可以投票。

+   `widget=forms.HiddenInput,`: `HiddenInput`小部件呈现为`<input type='hidden'>`HTML 元素，这意味着用户不会被任何 UI 分散注意力。

+   `disabled=True,`: `disabled`参数告诉表单忽略此字段的任何提供的数据，只使用代码中最初提供的值。这可以防止用户代表其他用户投票。

`movie`字段与`user`基本相同，但`queryset`属性查询`Movie`模型实例。

值字段以不同的方式进行了定制：

+   `value = forms.ChoiceField(`: `ChoiceField`用于表示可以从有限集合中具有单个值的字段。默认情况下，它由下拉列表小部件表示。

+   `label='Vote',`: `label`属性让我们自定义此字段使用的标签。虽然`value`在我们的代码中有意义，但我们希望用户认为他们的投票是`![](img/05b14743-dedd-4122-97df-cc15869422be.png)/![](img/73102249-cbaf-442e-a8f8-7ba208bb4348.png)`。

+   `widget=forms.RadioSelect,`: 下拉列表隐藏选项，直到用户点击下拉列表。但我们的值是我们希望始终可见的有效行动呼叫。使用`RadioSelect`小部件，Django 将每个选择呈现为`<input type='radio'>`标签，并带有适当的`<label>`标签和`name`值，以便更容易进行投票。

+   `choices=Vote.VALUE_CHOICES,`: `ChoiceField`必须告知有效选择；方便的是，它使用与模型字段的`choices`参数相同的格式，因此我们可以重用模型中使用的`Vote.VALUE_CHOICES`元组。

我们新定制的表单将显示为标签`vote`和两个单选按钮。

现在我们有了表单，让我们将投票添加到`MovieDetail`视图，并创建知道如何处理投票的视图。

# 创建投票视图

在这一部分，我们将更新`MovieDetail`视图，让用户投票并记录投票到数据库中。为了处理用户的投票，我们将创建以下两个视图：

+   `CreateVote`，这将是一个`CreateView`，如果用户尚未为电影投票

+   `UpdateVote`，这将是一个`UpdateView`，如果用户已经投票但正在更改他们的投票

让我们从更新`MovieDetail`开始，为电影提供投票的 UI。

# 将 VoteForm 添加到 MovieDetail

我们的`MovieDetail.get_context_data`方法现在会更加复杂。它将需要获取用户对电影的投票，实例化表单，并知道将投票提交到哪个 URL（`create_vote`或`update_vote`）。

我们首先需要一种方法来检查用户模型是否对给定的`Movie`模型实例有相关的`Vote`模型实例。为此，我们将创建一个带有自定义方法的`VoteManager`类。我们的方法将具有特殊行为 - 如果没有匹配的`Vote`模型实例，它将返回一个*未保存*的空白`Vote`对象。这将使我们更容易使用正确的`movie`和`user`值实例化我们的`VoteForm`。

这是我们的新`VoteManager`：

```py
class VoteManager(models.Manager):

    def get_vote_or_unsaved_blank_vote(self, movie, user):
        try:
            return Vote.objects.get(
                movie=movie,
                user=user)
        except Vote.DoesNotExist:
            return Vote(
                movie=movie,
                user=user)

class Vote(models.Model):
    # constants and field omitted

    objects = VoteManager()

    class Meta:
        unique_together = ('user', 'movie')
```

`VoteManager`与我们以前的`Manager`非常相似。

我们以前没有遇到的一件事是使用构造函数实例化模型（例如，`Vote(movie=movie, user=user)`）而不是使用其管理器的`create()`方法。使用构造函数在内存中创建一个新模型，但*不*在数据库中创建。未保存的模型本身是完全可用的（通常可用所有方法和管理器方法），但除了依赖关系的任何内容。未保存的模型没有`id`，因此在调用其`save()`方法保存之前，无法使用`RelatedManager`或`QuerySet`查找它。

现在我们已经拥有了`MovieDetail`所需的一切，让我们来更新它：

```py
class MovieDetail(DetailView):
    queryset = (
        Movie.objects
           .all_with_related_persons())

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        if self.request.user.is_authenticated:
            vote = Vote.objects.get_vote_or_unsaved_blank_vote(
                movie=self.object,
                user=self.request.user
            )
                    if vote.id:
                vote_form_url = reverse(
                    'core:UpdateVote',
                    kwargs={
                        'movie_id': vote.movie.id,
                        'pk': vote.id})
            else:
                vote_form_url = (
                    reverse(
                        'core:CreateVote',
                        kwargs={
                            'movie_id': self.object.id}
                    )
                )
            vote_form = VoteForm(instance=vote)
            ctx['vote_form'] = vote_form
            ctx['vote_form_url'] = \
                vote_form_url
        return ctx
```

我们在上述代码中引入了两个新元素，`self.request`和使用实例化表单。

视图通过它们的`request`属性访问它们正在处理的请求。此外，`Request`有一个`user`属性，它让我们访问发出请求的用户。我们使用这个来检查用户是否已经验证，因为只有已验证的用户才能投票。

`ModelForms`可以使用它们所代表的模型的实例进行实例化。当我们使用一个实例实例化`ModelForm`并渲染它时，字段将具有实例的值。一个常见任务的一个很好的快捷方式是在这个表单中显示这个模型的值。

我们还将引用两个我们还没有创建的`path`；我们马上就会创建。首先，让我们通过更新`movie_detail.html`模板的侧边栏块来完成我们的`MovieDetail`更新：

```py
{% block sidebar %}
 {# rating div omitted #}
  <div>
    {% if vote_form %}
      <form
          method="post"
          action="{{ vote_form_url }}" >
        {% csrf_token %}
        {{ vote_form.as_p }}
        <button
            class="btn btn-primary" >
          Vote
        </button >
      </form >
    {% else %}
      <p >Log in to vote for this
        movie</p >
    {% endif %}
  </div >
{% endblock %}
```

在设计这个过程中，我们再次遵循模板应该具有尽可能少的逻辑的原则。

接下来，让我们添加我们的`CreateVote`视图。

# 创建`CreateVote`视图

`CreateVote`视图将负责使用`VoteForm`验证投票数据，然后创建正确的`Vote`模型实例。然而，我们不会为投票创建一个模板。如果有问题，我们将把用户重定向到`MovieDetail`视图。

这是我们应该在`django/core/views.py`文件中拥有的`CreateVote`视图：

```py
from django.contrib.auth.mixins import (
    LoginRequiredMixin, )
from django.shortcuts import redirect
from django.urls import reverse
from django.views.generic import (
    CreateView, )

from core.forms import VoteForm

class CreateVote(LoginRequiredMixin, CreateView):
    form_class = VoteForm

    def get_initial(self):
        initial = super().get_initial()
        initial['user'] = self.request.user.id
        initial['movie'] = self.kwargs[
            'movie_id']
        return initial

    def get_success_url(self):
        movie_id = self.object.movie.id
        return reverse(
            'core:MovieDetail',
            kwargs={
                'pk': movie_id})

    def render_to_response(self, context, **response_kwargs):
        movie_id = context['object'].id
        movie_detail_url = reverse(
            'core:MovieDetail',
            kwargs={'pk': movie_id})
        return redirect(
            to=movie_detail_url)
```

在前面的代码中，我们引入了四个与`RegisterView`类不同的新概念——`get_initial()`、`render_to_response()`、`redirect()`和`LoginRequiredMixin`。它们如下：

+   `get_initial()`用于在表单从请求中获取`data`值之前，使用`initial`值预填充表单。这对于`VoteForm`很重要，因为我们已经禁用了`movie`和`user`。`Form`会忽略分配给禁用字段的`data`。即使用户在表单中发送了不同的`movie`值或`user`值，它也会被禁用字段忽略，而我们的`initial`值将被使用。

+   `render_to_response()`被`CreateView`调用以返回一个包含渲染模板的响应给客户端。在我们的情况下，我们不会返回一个包含模板的响应，而是一个 HTTP 重定向到`MovieDetail`。这种方法有一个严重的缺点——我们会丢失与表单相关的任何错误。然而，由于我们的用户只有两种输入选择，我们也无法提供太多错误消息。

+   `redirect()`来自 Django 的`django.shortcuts`包。它提供了常见操作的快捷方式，包括创建一个 HTTP 重定向响应到给定的 URL。

+   `LoginRequiredMixin`是一个可以添加到任何`View`中的 mixin，它将检查请求是否由已验证用户发出。如果用户没有登录，他们将被重定向到登录页面。

Django 的默认登录页面设置为`/accounts/profile/`，所以让我们通过编辑`settings.py`文件并添加一个新的设置来改变这一点：

```py
LOGIN_REDIRECT_URL = 'user:login'
```

现在我们有一个视图，它将创建一个`Vote`模型实例，并在成功或失败时将用户重定向回相关的`MovieDetail`视图。

接下来，让我们添加一个视图，让用户更新他们的`Vote`模型实例。

# 创建`UpdateVote`视图

`UpdateVote`视图要简单得多，因为`UpdateView`（就像`DetailView`）负责查找投票，尽管我们仍然必须关注`Vote`的篡改。

让我们更新我们的`django/core/views.py`文件：

```py
from django.contrib.auth.mixins import (
    LoginRequiredMixin, )
from django.core.exceptions import (
    PermissionDenied)
from django.shortcuts import redirect
from django.urls import reverse
from django.views.generic import (
    UpdateView, )

from core.forms import VoteForm

class UpdateVote(LoginRequiredMixin, UpdateView):
    form_class = VoteForm
    queryset = Vote.objects.all()

    def get_object(self, queryset=None):
        vote = super().get_object(
            queryset)
        user = self.request.user
        if vote.user != user:
            raise PermissionDenied(
                'cannot change another '
                'users vote')
        return vote

    def get_success_url(self):
        movie_id = self.object.movie.id
        return reverse(
            'core:MovieDetail',
            kwargs={'pk': movie_id})

    def render_to_response(self, context, **response_kwargs):
        movie_id = context['object'].id
        movie_detail_url = reverse(
            'core:MovieDetail',
            kwargs={'pk': movie_id})
        return redirect(
            to=movie_detail_url)

```

我们的`UpdateVote`视图在`get_object()`方法中检查检索到的`Vote`是否是已登录用户在其中的投票。我们添加了这个检查来防止投票篡改。我们的用户界面不会让用户错误地这样做。如果`Vote`不是由已登录用户投出的，那么`UpdateVote`会抛出一个`PermissionDenied`异常，Django 会处理并返回一个`403 Forbidden`响应。

最后一步将是在`core` URLConf 中注册我们的新视图。

# 在`core/urls.py`中添加视图

我们现在创建了两个新视图，但是，和往常一样，除非它们在 URLConf 中列出，否则用户无法访问它们。让我们编辑`core/urls.py`：

```py
urlpatterns = [
    # previous paths omitted
    path('movie/<int:movie_id>/vote',
         views.CreateVote.as_view(),
         name='CreateVote'),
    path('movie/<int:movie_id>/vote/<int:pk>',
         views.UpdateVote.as_view(),
         name='UpdateVote'),
]
```

# 本节的快速回顾

在本节中，我们看到了如何构建基本和高度定制的表单来接受和验证用户输入。我们还讨论了一些简化处理表单常见任务的内置视图。

接下来，我们将展示如何开始使用我们的用户、投票来对每部电影进行排名并提供一个前 10 名的列表。

# 计算电影得分

在这一部分，我们将使用 Django 的聚合查询 API 来计算每部电影的得分。Django 通过将功能内置到其`QuerySet`对象中，使编写与数据库无关的聚合查询变得容易。

让我们首先添加一个计算`MovieManager`得分的方法。

# 使用 MovieManager 来计算电影得分

我们的`MovieManager`类负责构建与`Movie`相关的`QuerySet`对象。我们现在需要一个新的方法，该方法检索电影（理想情况下仍与相关人员相关）并根据其收到的投票总和标记每部电影的得分（我们可以简单地对所有的`1`和`-1`求和）。

让我们看看如何使用 Django 的`QuerySet.annotate()` API 来做到这一点：

```py
from django.db.models.aggregates import (
    Sum
)

class MovieManager(models.Manager):

    def all_with_related_persons(self):
        qs = self.get_queryset()
        qs = qs.select_related(
            'director')
        qs = qs.prefetch_related(
            'writers', 'actors')
        return qs

    def all_with_related_persons_and_score(self):
        qs = self.all_with_related_persons()
        qs = qs.annotate(score=Sum('vote__value'))
        return qs
```

在`all_with_related_persons_and_score`中，我们调用`all_with_related_persons`并获得一个我们可以进一步使用`annotate()`调用修改的`QuerySet`。

`annotate`将我们的常规 SQL 查询转换为聚合查询，将提供的聚合操作的结果添加到一个名为`score`的新属性中。Django 将大多数常见的 SQL 聚合函数抽象为类表示，包括`Sum`、`Count`和`Average`（以及更多）。

新的`score`属性可用于我们从`QuerySet`中`get()`出来的任何实例，以及我们想要在我们的新`QuerySet`上调用的任何方法（例如，`qs.filter(score__gt=5)`将返回一个具有`score`属性大于 5 的电影的`QuerySet`）。

我们的新方法仍然返回一个懒惰的`QuerySet`，这意味着我们的下一步是更新`MovieDetail`及其模板。

# 更新 MovieDetail 和模板

现在我们可以查询带有得分的电影，让我们更改`MovieDetail`使用的`QuerySet`：

```py
 class MovieDetail(DetailView):
    queryset = Movie.objects.all_with_related_persons_and_score() 
    def get_context_data(self, **kwargs):
        # body omitted for brevity
```

现在，当`MovieDetail`在其查询集上使用`get()`时，该`Movie`将具有一个得分属性。让我们在我们的`movie_detail.html`模板中使用它：

```py
{% block sidebar %}
  {# movie rating div omitted #}
  <div >
    <h2 >
      Score: {{ object.score|default_if_none:"TBD" }}
    </h2 >
  </div>
  {# voting form div omitted #}
{% endblock %}
```

我们可以安全地引用`score`属性，因为`MovieDetail`的`QuerySet`。然而，我们不能保证得分不会是`None`（例如，如果`Movie`没有投票）。为了防止空白得分，我们使用`default_if_none`过滤器来提供一个要打印的值。

我们现在有一个可以计算所有电影得分的`MovieManager`方法，但是当您在`MovieDetail`中使用它时，这意味着它只会为正在显示的`Movie`计算得分。

# 总结

在本章中，我们向我们的系统添加了用户，让他们注册、登录（和退出登录），并对我们的电影进行投票。我们学会了如何使用聚合查询来高效地计算数据库中这些投票的结果。

接下来，我们将让用户上传与我们的`Movie`和`People`模型相关的图片，并讨论安全考虑。
