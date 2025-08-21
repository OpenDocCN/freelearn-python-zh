# 第六章：开始 Answerly

我们将构建的第二个项目是一个名为 Answerly 的 Stack Overflow 克隆。 注册 Answerly 的用户将能够提问和回答问题。 提问者还将能够接受答案以标记它们为有用。

在本章中，我们将做以下事情：

+   创建我们的新 Django 项目 Answerly，一个 Stack Overflow 克隆

+   为 Answerly 创建模型（`Question`和`Answer`）

+   让用户注册

+   创建表单，视图和模板，让用户与我们的模型进行交互

+   运行我们的代码

该项目的代码可在[`github.com/tomaratyn/Answerly`](https://github.com/tomaratyn/Answerly)上找到。

本章不会深入讨论已在第一章中涵盖的主题，尽管它将涉及许多相同的要点。 相反，本章将重点放在更进一步并引入新视图和第三方库上。

让我们开始我们的项目！

# 创建 Answerly Django 项目

首先，让我们为我们的项目创建一个目录：

```py
$ mkdir answerly
$ cd answerly
```

我们未来的所有命令和路径都将相对于这个项目目录。 一个 Django 项目由多个 Django 应用程序组成。

我们将使用`pip`安装 Django，Python 的首选软件包管理器。 我们还将在`requirements.txt`文件中跟踪我们安装的软件包：

```py
django<2.1
psycopg2<2.8
```

现在，让我们安装软件包：

```py
$ pip install -r requirements.txt
```

接下来，让我们使用`django-admin`生成实际的 Django 项目：

```py
$ django-admin startproject config
$ mv config django
```

默认情况下，Django 创建一个将使用 SQLite 的项目，但这对于生产来说是不可用的； 因此，我们将遵循在开发和生产中使用相同数据库的最佳实践。

让我们打开`django/config/settings.py`并更新它以使用我们的 Postgres 服务器。 找到以`DATABASES`开头的`settings.py`中的行； 要使用 Postgres，请将`DATABASES`的值更改为以下代码：

```py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'answerly',
        'USER': 'answerly',
        'PASSWORD': 'development',
        'HOST': '127.0.0.1',
        'PORT': '5432',
    }
}
```

现在我们已经开始并配置了我们的项目，我们可以创建并安装我们将作为项目一部分制作的两个 Django 应用程序：

```py
$ cd django
$ python manage.py startapp user
$ python manage.py startapp qanda
```

Django 项目由应用程序组成。 Django 应用程序是所有功能和代码所在的地方。 模型，表单和模板都属于 Django 应用程序。 应用程序，就像其他 Python 模块一样，应该有一个明确定义的范围。 在我们的情况下，我们有两个应用程序，每个应用程序都有不同的角色。 `qanda`应用程序将负责我们应用程序的问题和答案功能。 `user`应用程序将负责我们应用程序的用户管理。 它们每个都将依赖其他应用程序和 Django 的核心功能以有效地工作。

现在，让我们通过更新`django/config/settings.py`在我们的项目中安装我们的应用程序：

```py
INSTALLED_APPS = [
    'user',
    'qanda',

    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]
```

既然 Django 知道我们的应用程序，让我们从`qanda`的模型开始安装。

# 创建 Answerly 模型

Django 在创建数据驱动的应用程序方面特别有帮助。 模型代表应用程序中的数据，通常是这些应用程序的核心。 Django 通过*fat models, thin views, dumb templates*的最佳实践鼓励这一点。 这些建议鼓励我们将业务逻辑放在我们的模型中，而不是我们的视图中。

让我们从`Question`模型开始构建我们的`qanda`模型。

# 创建 Question 模型

我们将在`django/qanda/models.py`中创建我们的`Question`模型：

```py
from django.conf import settings
from django.db import models
from django.urls.base import reverse

class Question(models.Model):
    title = models.CharField(max_length=140)
    question = models.TextField()
    user = models.ForeignKey(to=settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('questions:question_detail', kwargs={'pk': self.id})

    def can_accept_answers(self, user):
        return user == self.user
```

`Question`模型，像所有 Django 模型一样，派生自`django.db.models.Model`。 它具有以下四个字段，这些字段将成为`questions_question`表中的列：

+   `title`：一个字符字段，将成为最多 140 个字符的`VARCHAR`列。

+   `question`：这是问题的主体。 由于我们无法预测这将有多长，我们使用`TextField`，它将成为`TEXT`列。`TEXT`列没有大小限制。

+   `user`：这将创建一个外键到项目配置的用户模型。 在我们的情况下，我们将使用 Django 提供的默认`django.contrib.auth.models.User`。 但是，建议我们尽量避免硬编码这一点。

+   `created`：这将自动设置为创建`Question`模型的日期和时间。

`Question`还实现了 Django 模型上常见的两种方法（`__str__`和`get_absolute_url`）：

+   `__str__()`：这告诉 Python 如何将我们的模型转换为字符串。这在管理后端、我们自己的模板和调试中非常有用。

+   `get_absolute_url()`：这是一个常见的实现方法，让模型返回查看此模型的 URL 路径。并非所有模型都需要此方法。Django 的内置视图，如`CreateView`，将使用此方法在创建模型后将用户重定向到视图。

最后，在“fat models”的精神下，我们还有`can_accept_answers()`。谁可以接受对`Question`的`Answer`的决定取决于`Question`。目前，只有提问问题的用户可以接受答案。

现在我们有了`Question`，自然需要`Answer`。

# 创建`Answer`模型

我们将在`django/questions/models.py`文件中创建`Answer`模型，如下所示：

```py
from django.conf import settings
from django.db import models

class Question(model.Models):
    # skipped

class Answer(models.Model):
    answer = models.TextField()
    user = models.ForeignKey(to=settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    question = models.ForeignKey(to=Question,
                                 on_delete=models.CASCADE)
    accepted = models.BooleanField(default=False)

    class Meta:
        ordering = ('-created', )
```

`Answer`模型有五个字段和一个`Meta`类。让我们先看看这些字段：

+   `answer`：这是用户答案的无限文本字段。`answer`将成为一个`TEXT`列。

+   `user`：这将创建一个到我们项目配置为使用的用户模型的外键。用户模型将获得一个名为`answer_set`的新`RelatedManager`，它将能够查询用户的所有`Answer`。

+   `question`：这将创建一个到我们的`Question`模型的外键。`Question`还将获得一个名为`answer_set`的新`RelatedManager`，它将能够查询所有`Question`的`Answer`。

+   `created`：这将设置为创建`Answer`的日期和时间。

+   `accepted`：这是一个默认设置为`False`的布尔值。我们将用它来标记已接受的答案。

模型的`Meta`类让我们为我们的模型和表设置元数据。对于`Answer`，我们使用`ordering`选项来确保所有查询都将按`created`的降序排序。通过这种方式，我们确保最新的答案将首先列出，默认情况下。

现在我们有了`Question`和`Answer`模型，我们需要创建迁移以在数据库中创建它们的表。

# 创建迁移

Django 自带一个内置的迁移库。这是 Django“一揽子”哲学的一部分。迁移提供了一种管理我们需要对模式进行的更改的方法。每当我们对模型进行更改时，我们可以使用 Django 生成一个迁移，其中包含了如何创建或更改模式以适应新模型定义的指令。要对数据库进行更改，我们将应用模式。

与我们在项目上执行的许多操作一样，我们将使用 Django 为我们的项目提供的`manage.py`脚本：

```py
$ python manage.py makemigrations
 Migrations for 'qanda':
  qanda/migrations/0001_initial.py
    - Create model Answer
    - Create model Question
    - Add field question to answer
    - Add field user to answer
$ python manage.py migrate
Operations to perform:
  Apply all migrations: admin, auth, contenttypes, qanda, sessions
Running migrations:
  Applying qanda.0001_initial... OK
```

现在我们已经创建了迁移并应用了它们，让我们为我们的项目设置一个基础模板，以便我们的代码能够正常工作。

# 添加基础模板

在创建视图之前，让我们创建一个基础模板。Django 的模板语言允许模板相互继承。基础模板是所有其他项目模板都将扩展的模板。这将给我们整个项目一个共同的外观和感觉。

由于项目由多个应用程序组成，它们都将使用相同的基础模板，因此基础模板属于项目，而不属于任何特定的应用程序。这是一个罕见的例外，违反了一切都在应用程序中的规则。

要添加一个项目范围的模板目录，请更新`django/config/settings.py`。检查`TEMPLATES`设置并将其更新为以下内容：

```py
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            os.path.join(BASE_DIR, 'templates')
        ],
        'APP_DIRS': True,
        'OPTIONS': {
                # skipping rest of options.
        },
    },
]
```

特别是，`django.template.backends.django.DjangoTemplates`设置的`DIRS`选项设置了一个项目范围的模板目录，将被搜索。`'APP_DIRS': True`意味着每个安装的应用程序的`templates`目录也将被搜索。为了让 Django 搜索`django/templates`，我们必须将`os.path.join(BASE_DIR, 'templates')`添加到`DIRS`列表中。

# 创建 base.html

Django 自带了自己的模板语言，名为 Django 模板语言。Django 模板是文本文件，使用字典（称为上下文）进行渲染以查找值。模板还可以包括标签（使用`{% tag argument %}`语法）。模板可以使用`{{ variableName }}`语法从其上下文中打印值。值可以发送到过滤器进行调整，然后显示（例如，`{{ user.username | uppercase }}`将打印用户的用户名，所有字符都是大写）。最后，`{# ignored #}`语法可以注释掉多行文本。

我们将在`django/templates/base.html`中创建我们的基本模板：

```py
{% load static %}
<!DOCTYPE html>
<html lang="en" >
<head >
  <meta charset="UTF-8" >
  <title >{% block title %}Answerly{% endblock %}</title >
  <link
      href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.2/css/bootstrap.min.css"
      rel="stylesheet">
  <link
      href="https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"
      rel="stylesheet">
  <link rel="stylesheet" href="{% static "base.css" %}" >
</head >
<body >
<nav class="navbar navbar-expand-lg  bg-light" >
  <div class="container" >
    <a class="navbar-brand" href="/" >Answerly</a >
    <ul class="navbar-nav" >
    </ul >
  </div >
</nav >
<div class="container" >
  {% block body %}{% endblock %}
</div >
</body >
</html >
```

我们不会详细介绍这个 HTML，但值得回顾涉及的 Django 模板标签：

+   `{% load static %}`：`load`让我们加载默认情况下不可用的模板标签库。在这种情况下，我们加载了静态库，它提供了`static`标签。该库和标签并不总是共享它们的名称。这是由`django.contrib.static`应用程序提供的 Django。

+   `{% block title %}Answerly{% endblock %}`：块让我们定义模板在扩展此模板时可以覆盖的区域。

+   `{% static 'base.css' %}`：`static`标签（从前面加载的`static`库中加载）使用`STATIC_URL`设置来创建对静态文件的引用。在这种情况下，它将返回`/static/base.css`。只要文件在`settings.STATICFILES_DIRS`列出的目录中，并且 Django 处于调试模式，Django 就会为我们提供该文件。对于生产环境，请参阅第九章，*部署 Answerly*。

这就足够我们的`base.html`文件开始了。我们将在*更新 base.html 导航*部分中稍后更新`base.html`中的导航。

接下来，让我们配置 Django 知道如何找到我们的`base.css`文件，通过配置静态文件。

# 配置静态文件

接下来，让我们在`django/config/settings.py`中配置一个项目范围的静态文件目录：

```py
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]
```

这将告诉 Django，在调试模式下应该提供`django/static/`中的任何文件。对于生产环境，请参阅第九章，*部署 Answerly*。

让我们在`django/static/base.css`中放一些基本的 CSS：

```py
nav.navbar {
  margin-bottom: 1em;
}
```

现在我们已经创建了基础，让我们创建`AskQuestionView`。

# 让用户发布问题

现在我们将创建一个视图，让用户发布他们需要回答的问题。

Django 遵循**模型-视图-模板**（**MVT**）模式，将模型、控制和表示逻辑分开，并鼓励可重用性。模型代表我们将在数据库中存储的数据。视图负责处理请求并返回响应。视图不应该包含 HTML。模板负责响应的主体和定义 HTML。这种责任的分离已被证明使编写代码变得容易。

为了让用户发布问题，我们将执行以下步骤：

1.  创建一个处理问题的表单

1.  创建一个使用 Django 表单创建问题的视图

1.  创建一个在 HTML 中渲染表单的模板

1.  在视图中添加一个`path`

首先，让我们创建`QuestionForm`类。

# 提问表单

Django 表单有两个目的。它们使得渲染表单主体以接收用户输入变得容易。它们还验证用户输入。当一个表单被实例化时，它可以通过`intial`参数给出初始值，并且通过`data`参数给出要验证的数据。提供了数据的表单被称为绑定的。

Django 的许多强大之处在于将模型、表单和视图轻松地结合在一起构建功能。

我们将在`django/qanda/forms.py`中创建我们的表单：

```py
from django import forms
from django.contrib.auth import get_user_model

from qanda.models import Question

class QuestionForm(forms.ModelForm):
    user = forms.ModelChoiceField(
        widget=forms.HiddenInput,
        queryset=get_user_model().objects.all(),
        disabled=True,
    )

    class Meta:
        model = Question
        fields = ['title', 'question', 'user', ]
```

`ModelForm`使得从 Django 模型创建表单更容易。我们使用`QuestionForm`的内部`Meta`类来指定表单的模型和字段。

通过添加一个`user`字段，我们能够覆盖 Django 如何呈现`user`字段。我们告诉 Django 使用`HiddenInput`小部件，它将把字段呈现为`<input type='hidden'>`。`queryset`参数让我们限制有效值的用户（在我们的情况下，所有用户都是有效的）。最后，`disabled`参数表示我们将忽略由`data`（即来自请求的）提供的任何值，并依赖于我们提供给表单的`initial`值。

现在我们知道如何呈现和验证问题表单，让我们创建我们的视图。

# 创建 AskQuestionView

我们将在`django/qanda/views.py`中创建我们的`AskQuestionView`类：

```py
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import CreateView

from qanda.forms import QuestionForm
from qanda.models import Question

class AskQuestionView(LoginRequiredMixin, CreateView):
    form_class = QuestionForm
    template_name = 'qanda/ask.html'

    def get_initial(self):
        return {
            'user': self.request.user.id
        }

    def form_valid(self, form):
        action = self.request.POST.get('action')
        if action == 'SAVE':
            # save and redirect as usual.
            return super().form_valid(form)
        elif action == 'PREVIEW':
            preview = Question(
                question=form.cleaned_data['question'],
                title=form.cleaned_data['title'])
            ctx = self.get_context_data(preview=preview)
            return self.render_to_response(context=ctx)
        return HttpResponseBadRequest()
```

`AskQuestionView`派生自`CreateView`并使用`LoginRequiredMixin`。`LoginRequiredMixin`确保任何未登录用户发出的请求都将被重定向到登录页面。`CreateView`知道如何为`GET`请求呈现模板，并在`POST`请求上验证表单。如果表单有效，`CreateView`将调用`form_valid`。如果表单无效，`CreateView`将重新呈现模板。

我们的`form_valid`方法覆盖了原始的`CreateView`方法，以支持保存和预览模式。当我们想要保存时，我们将调用原始的`form_valid`方法。原始方法保存新问题并返回一个 HTTP 响应，将用户重定向到新问题（使用`Question.get_absolute_url()`）。当我们想要预览问题时，我们将在我们模板的上下文中重新呈现我们的模板，其中包含新的`preview`变量。

当我们的视图实例化表单时，它将把`get_initial()`的结果作为`initial`参数传递，并将`POST`数据作为`data`参数传递。

现在我们有了我们的视图，让我们创建`ask.html`。

# 创建 ask.html

让我们在`django/qanda/ask.html`中创建我们的模板：

```py
{% extends "base.html" %}

{% load markdownify %}
{% load crispy_forms_tags %}

{% block title %} Ask a question {% endblock %}

{% block body %}
  <div class="col-md-12" >
    <h1 >Ask a question</h1 >
    {% if preview %}
      <div class="card question-preview" >
        <div class="card-header" >
          Question Preview
        </div >
        <div class="card-body" >
          <h1 class="card-title" >{{ preview.title }}</h1>
          {{ preview.question |  markdownify }}
        </div >
      </div >
    {% endif %}

    <form method="post" >
      {{ form | crispy }}
      {% csrf_token %}
      <button class="btn btn-primary" type="submit" name="action"
              value="PREVIEW" >
        Preview
      </button >
      <button class="btn btn-primary" type="submit" name="action"
              value="SAVE" >
        Ask!
      </button >
    </form >
  </div >
{% endblock %}
```

此模板使用我们的`base.html`模板，并将所有 HTML 放在那里定义的`blocks`中。当我们呈现模板时，Django 会呈现`base.html`，然后用在`ask.html`中定义的内容填充块的值。

`ask.html`还加载了两个第三方标签库，`markdownify`和`crispy_forms_tags`。`markdownify`提供了用于预览卡正文的`markdownify`过滤器（`{{preview.question | markdownify}}`）。`crispy_forms_tags`库提供了`crispy`过滤器，它应用 Bootstrap 4 CSS 类以帮助 Django 表单呈现得很好。

这些库中的每一个都需要安装和配置，我们将在接下来的部分中进行（*安装和配置 Markdownify*和*安装和配置 Django Crispy Forms*）。

以下是`ask.html`向我们展示的一些新标记：

+   `{% if preview %}`：这演示了如何在 Django 模板语言中使用`if`语句。我们只想在我们的上下文中有一个`preview`变量时才呈现`Question`的预览。

+   `{% csrf_token %}`：此标记将预期的 CSRF 令牌添加到我们的表单中。 CSRF 令牌有助于保护我们免受恶意脚本试图代表一个无辜但已登录的用户提交数据的攻击；有关更多信息，请参阅第三章，*海报、头像和安全性*。在 Django 中，CSRF 令牌是不可选的，缺少 CSRF 令牌的`POST`请求将不会被处理。

让我们更仔细地看看那些第三方库，从 Markdownify 开始。

# 安装和配置 Markdownify

Markdownify 是由 R Moelker 和 Erwin Matijsen 创建的 Django 应用程序，可在**Python Package Index**（**PyPI**）上找到，并根据 MIT 许可证（一种流行的开源许可证）进行许可。Markdownify 提供了 Django 模板过滤器`markdownify`，它将 Markdown 转换为 HTML。

Markdownify 通过使用**python-markdown**包将 Markdown 转换为 HTML 来工作。然后，Marodwnify 使用 Mozilla 的`bleach`库来清理结果 HTML，以防止跨站脚本（**XSS**）攻击。然后将结果返回到模板进行输出。

要安装 Markdownify，让我们将其添加到我们的`requirements.txt`文件中：

```py
django-markdownify==0.2.2
```

然后，运行`pip`进行安装：

```py
$ pip install -r requirements.txt
```

现在，我们需要在`django/config/settings.py`中将`markdownify`添加到我们的`INSTALLED_APPS`列表中。

最后一步是配置 Markdownify，让它知道要对哪些 HTML 标签进行白名单。将以下设置添加到`settings.py`中：

```py
MARKDOWNIFY_STRIP = False
MARKDOWNIFY_WHITELIST_TAGS = [
    'a', 'blockquote', 'code', 'em', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
    'h7', 'li', 'ol', 'p', 'strong', 'ul',
]
```

这将使我们的用户可以使用所有文本、列表和标题标签。将`MARKDOWNIFY_STRIP`设置为`False`告诉 Markdownify 对其他 HTML 标签进行 HTML 编码（而不是剥离）。

现在我们已经配置了 Markdownify，让我们安装和配置 Django Crispy Forms。

# 安装和配置 Django Crispy Forms

Django Crispy Forms 是 PyPI 上可用的第三方 Django 应用程序。Miguel Araujo 是开发负责人。它是根据 MIT 许可证许可的。Django Crispy Forms 是最受欢迎的 Django 库之一，因为它使得渲染漂亮（清晰）的表单变得如此容易。

在 Django 中遇到的问题之一是，当 Django 渲染字段时，它会呈现为这样：

```py
<label for="id_title">Title:</label>
<input 
      type="text" name="title" maxlength="140" required id="id_title" />
```

然而，为了漂亮地设计该表单，例如使用 Bootstrap 4，我们希望呈现类似于这样的内容：

```py
<div class="form-group"> 
<label for="id_title" class="form-control-label  requiredField">
   Title
</label> 
<input type="text" name="title" maxlength="140" 
  class="textinput textInput form-control" required="" id="id_title">  
</div>
```

遗憾的是，Django 没有提供钩子，让我们轻松地将字段包装在具有类`form-group`的`div`中，或者添加 CSS 类，如`form-control`或`form-control-label`。

Django Crispy Forms 通过其`crispy`过滤器解决了这个问题。如果我们通过执行`{{ form | crispy}}`将一个表单发送到它，Django Crispy Forms 将正确地转换表单的 HTML 和 CSS，以适应各种 CSS 框架（包括 Zurb Foundation，Bootstrap 3 和 Bootstrap 4）。您可以通过更高级的使用 Django Crispy Forms 进一步自定义表单的渲染，但在本章中我们不会这样做。

要安装 Django Crispy Forms，让我们将其添加到我们的`requirements.txt`并使用`pip`进行安装：

```py
$ echo "django-crispy-forms==1.7.0" >> requirements.txt
$ pip install -r requirements.txt
```

现在，我们需要通过编辑`django/config/settings.py`并将`'crispy_forms'`添加到我们的`INSTALLED_APPS`列表中，将其安装为我们项目中的 Django 应用程序。

接下来，我们需要配置我们的项目，以便 Django Crispy Forms 知道使用 Bootstrap 4 模板包。更新`django/config/settings.py`以进行新的配置：

```py
CRISPY_TEMPLATE_PACK = 'bootstrap4'
```

现在我们已经安装了模板所依赖的所有库，我们可以配置 Django 将请求路由到我们的`AskQuestionView`。

# 将请求路由到 AskQuestionView

Django 使用 URLConf 路由请求。这是一个`path()`对象的列表，用于匹配请求的路径。第一个匹配的`path()`的视图将处理请求。URLConf 可以包含另一个 URLConf。项目的设置定义了其根 URLConf（在我们的情况下是`django/config/urls.py`）。

在根 URLConf 中为项目中所有视图的所有`path()`对象定义可以变得混乱，并使应用程序不太可重用。通常方便的做法是在每个应用程序中放置一个 URLConf（通常在`urls.py`文件中）。然后，根 URLConf 可以使用`include()`函数来包含其他应用程序的 URLConfs 以路由请求。

让我们在`django/qanda/urls.py`中为我们的`qanda`应用程序创建一个 URLConf：

```py
from django.urls.conf import path

from qanda import views

app_name = 'qanda'
urlpatterns = [
    path('ask', views.AskQuestionView.as_view(), name='ask'),
]
```

路径至少有两个组件：

+   首先，是定义匹配路径的字符串。这可能有命名参数，将传递给视图。稍后我们将在*将请求路由到 QuestionDetail 视图*部分看到一个例子。

+   其次，是一个接受请求并返回响应的可调用对象。如果您的视图是一个函数（也称为**基于函数的视图**（**FBV**）），那么您可以直接传递对函数的引用。如果您使用的是**基于类的视图**（**CBV**），那么您可以使用其`as_view()`类方法来返回所需的可调用对象。

+   可选的`name`参数，我们可以在视图或模板中引用这个`path()`对象（例如，就像`Question`模型在其`get_absolute_url()`方法中所做的那样）。

强烈建议为所有的`path()`对象命名。

现在，让我们更新我们的根 URLConf 以包括`qanda`的 URLConf：

```py
from django.contrib import admin
from django.urls import path, include

import qanda.urls

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include(qanda.urls, namespace='qanda')),
]
```

这意味着对`answerly.example.com/ask`的请求将路由到我们的`AskQuestionView`。

# 本节的快速回顾

在本节中，我们执行了以下操作：

+   创建了我们的第一个表单，`QuestionForm`

+   创建了使用`QuestionForm`创建`Question`的`AskQuestionView`

+   创建了一个模板来渲染`AskQuestionView`和`QuestionForm`

+   安装和配置了为我们的模板提供过滤器的第三方库

现在，让我们允许我们的用户使用`QuestionDetailView`类查看问题。

# 创建 QuestionDetailView

`QuestionDetailView`必须提供相当多的功能。它必须能够执行以下操作：

+   显示问题

+   显示所有答案

+   让用户发布额外的答案

+   让提问者接受答案

+   让提问者拒绝先前接受的答案

尽管`QuestionDetailView`不会处理任何表单，但它必须显示许多表单，导致一个复杂的模板。这种复杂性将给我们一个机会来注意如何将模板分割成单独的子模板，以使我们的代码更易读。

# 创建答案表单

我们需要制作两个表单，以使`QuestionDetailView`按照前一节的描述工作：

+   `AnswerForm`：供用户发布他们的答案

+   `AnswerAcceptanceForm`：供问题的提问者接受或拒绝答案

# 创建 AnswerForm

`AnswerForm`将需要引用一个`Question`模型实例和一个用户，因为这两者都是创建`Answer`模型实例所必需的。

让我们将我们的`AnswerForm`添加到`django/qanda/forms.py`中：

```py
from django import forms
from django.contrib.auth import get_user_model

from qanda.models import Answers

class AnswerForm(forms.ModelForm):
    user = forms.ModelChoiceField(
        widget=forms.HiddenInput,
        queryset=get_user_model().objects.all(),
        disabled=True,
    )
    question = forms.ModelChoiceField(
        widget=forms.HiddenInput,
        queryset=Question.objects.all(),
        disabled=True,
    )

    class Meta:
        model = Answer
        fields = ['answer', 'user', 'question', ]
```

`AnswerForm`类看起来很像`QuestionForm`类，尽管字段的命名略有不同。它使用了与`QuestionForm`相同的技术，防止用户篡改与`Answer`相关联的`Question`，就像`QuestionForm`用于防止篡改`Question`的用户一样。

接下来，我们将创建一个接受`Answer`的表单。

# 创建 AnswerAcceptanceForm

如果`accepted`字段为`True`，则`Answer`被接受。我们将使用一个简单的表单来编辑这个字段：

```py
class AnswerAcceptanceForm(forms.ModelForm):
    accepted = forms.BooleanField(
        widget=forms.HiddenInput,
        required=False,
    )

    class Meta:
        model = Answer
        fields = ['accepted', ]
```

使用`BooleanField`会有一个小问题。如果我们希望`BooleanField`接受`False`值以及`True`值，我们必须设置`required=False`。否则，`BooleanField`在接收到`False`值时会感到困惑，认为它实际上没有收到值。

我们使用了一个隐藏的输入，因为我们不希望用户勾选复选框然后再点击提交。相反，对于每个答案，我们将生成一个接受表单和一个拒绝表单，用户只需点击一次即可提交。

接下来，让我们编写`QuestionDetailView`类。

# 创建 QuestionDetailView

现在我们有了要使用的表单，我们可以在`django/qanda/views.py`中创建`QuestionDetailView`：

```py
from django.views.generic import DetailView

from qanda.forms import AnswerForm, AnswerAcceptanceForm
from qanda.models import Question

class QuestionDetailView(DetailView):
    model = Question

    ACCEPT_FORM = AnswerAcceptanceForm(initial={'accepted': True})
    REJECT_FORM = AnswerAcceptanceForm(initial={'accepted': False})

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx.update({
            'answer_form': AnswerForm(initial={
                'user': self.request.user.id,
                'question': self.object.id,
            })
        })
        if self.object.can_accept_answers(self.request.user):
            ctx.update({
                'accept_form': self.ACCEPT_FORM,
                'reject_form': self.REJECT_FORM,
            })
        return ctx
```

`QuestionDetailView`让 Django 的`DetailView`完成大部分工作。`DetailView`从`Question`的默认管理器（`Question.objects`）中获取一个`Question`的`QuerySet`。然后，`DetailView`使用`QuerySet`根据 URL 路径中收到的`pk`获取一个`Question`。`DetailView`还根据我们的应用程序和模型名称（`appname/modelname_detail.html`）知道要渲染哪个模板。

我们唯一需要自定义`DetailView`行为的地方是`get_context_data（）`。`get_context_data（）`提供用于呈现模板的上下文。在我们的情况下，我们使用该方法将要呈现的表单添加到上下文中。

接下来，让我们为`QuestionDetailView`创建模板。

# 创建 question_detail.html

我们的`QuestionDetailView`模板将与我们以前的模板略有不同。

以下是我们将放入`django/qanda/templates/qanda/question_detail.html`中的内容：

```py
{% extends "base.html" %}

{% block title %}{{ question.title }} - {{ block.super }}{% endblock %}

{% block body %}
  {% include "qanda/common/display_question.html" %}
  {% include "qanda/common/list_answers.html" %}
  {% if user.is_authenticated %}
    {% include "qanda/common/question_post_answer.html" %}
  {% else %}
    <div >Login to post answers.</div >
  {% endif %}
{% endblock %}
```

前面的模板似乎并没有做任何事情。相反，我们使用`{% include %}`标签将其他模板包含在此模板中，以使我们的代码组织更简单。`{% include %}`将当前上下文传递给新模板，呈现它，并将其插入到指定位置。

让我们依次查看这些子模板，从`dispaly_question.html`开始。

# 创建 display_question.html 通用模板

我们已经将显示问题的 HTML 放入了自己的子模板中。然后其他模板可以包含此模板，以呈现问题。

让我们在`django/qanda/templates/qanda/common/display_question.html`中创建它：

```py
{% load markdownify %}
<div class="question" >
  <div class="meta col-sm-12" >
    <h1 >{{ question.title }}</h1 >
    Asked by {{ question.user }} on {{ question.created }}
  </div >
  <div class="body col-sm-12" >
    {{ question.question|markdownify }}
  </div >
</div >
```

HTML 本身非常简单，在这里没有新标签。我们重用了之前配置的`markdownify`标签和库。

接下来，让我们看一下答案列表模板。

# 创建 list_answers.html

答案列表模板必须列出问题的所有答案，并渲染答案是否被接受。如果用户可以接受（或拒绝）答案，那么这些表单也会被呈现。

让我们在`django/qanda/templates/qanda/view_questions/question_answers.html`中创建模板：

```py
{% load markdownify %}
<h3 >Answers</h3 >
<ul class="list-unstyled answers" >
  {% for answer in question.answer_set.all %}
    <li class="answer row" >
      <div class="col-sm-3 col-md-2 text-center" >
        {% if answer.accepted %}
          <span class="badge badge-pill badge-success" >Accepted</span >
        {% endif %}
        {% if answer.accepted and reject_form %}
          <form method="post"
                action="{% url "qanda:update_answer_acceptance" pk=answer.id %}" >
            {% csrf_token %}
            {{ reject_form }}
            <button type="submit" class="btn btn-link" >
              <i class="fa fa-times" aria-hidden="true" ></i>
              Reject
            </button >
          </form >
        {% elif accept_form %}
          <form method="post"
                action="{% url "qanda:update_answer_acceptance" pk=answer.id %}" >
            {% csrf_token %}
            {{ accept_form }}
            <button type="submit" class="btn btn-link" title="Accept answer" >
              <i class="fa fa-check-circle" aria-hidden="true"></i >
              Accept
            </button >
          </form >
        {% endif %}
      </div >
      <div class="col-sm-9 col-md-10" >
        <div class="body" >{{ answer.answer|markdownify }}</div >
        <div class="meta font-weight-light" >
          Answered by {{ answer.user }} on {{ answer.created }}
        </div >
      </div >
    </li >
  {% empty %}
    <li class="answer" >No answers yet!</li >
  {% endfor %}
</ul >
```

关于这个模板有两件事需要注意：

+   模板中有一个罕见的逻辑，`{% if answer.accepted and reject_form %}`。通常，模板应该是简单的，避免了解业务逻辑。然而，避免这种情况会创建一个更复杂的视图。这是我们必须始终根据具体情况评估的权衡。

+   `{% empty %}`标签与我们的`{% for answer in question.answer_set.all %}`循环有关。`{% empty %}`在列表为空的情况下使用，就像 Python 的`for ... else`语法一样。

接下来，让我们看一下发布答案模板。

# 创建 post_answer.html 模板

在接下来要创建的模板中，用户可以发布和预览他们的答案。

让我们在`django/qanda/templates/qanda/common/post_answer.html`中创建我们的下一个模板：

```py
{% load crispy_forms_tags %}

<div class="col-sm-12" >
  <h3 >Post your answer</h3 >
  <form method="post"
        action="{% url "qanda:answer_question" pk=question.id %}" >
    {{ answer_form | crispy }}
    {% csrf_token %}
    <button class="btn btn-primary" type="submit" name="action"
            value="PREVIEW" >Preview
    </button >
    <button class="btn btn-primary" type="submit" name="action"
            value="SAVE" >Answer
    </button >
  </form >
</div >
```

这个模板非常简单，使用`crispy`过滤器对`answer_form`进行渲染。

现在我们所有的子模板都完成了，让我们创建一个`path`来将请求路由到`QuestionDetailView`。

# 将请求路由到 QuestionDetail 视图

为了能够将请求路由到我们的`QuestionDetailView`，我们需要将其添加到`django/qanda/urls.py`中的 URLConf：

```py
    path('q/<int:pk>', views.QuestionDetailView.as_view(),
         name='question_detail'),
```

在上述代码中，我们看到`path`使用了一个名为`pk`的参数，它必须是一个整数。这将传递给`QuestionDetailView`并在`kwargs`字典中可用。`DetailView`将依赖于此参数的存在来知道要检索哪个`Question`。

接下来，我们将创建一些我们在模板中引用的与表单相关的视图。让我们从`CreateAnswerView`类开始。

# 创建 CreateAnswerView

`CreateAnswerView`类将用于为`Question`模型实例创建和预览`Answer`模型实例。

让我们在`django/qanda/views.py`中创建它：

```py
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import CreateView

from qanda.forms import AnswerForm

class CreateAnswerView(LoginRequiredMixin, CreateView):
    form_class = AnswerForm
    template_name = 'qanda/create_answer.html'

    def get_initial(self):
        return {
            'question': self.get_question().id,
            'user': self.request.user.id,
        }

    def get_context_data(self, **kwargs):
        return super().get_context_data(question=self.get_question(),
                                        **kwargs)

    def get_success_url(self):
        return self.object.question.get_absolute_url()

    def form_valid(self, form):
        action = self.request.POST.get('action')
        if action == 'SAVE':
            # save and redirect as usual.
            return super().form_valid(form)
        elif action == 'PREVIEW':
            ctx = self.get_context_data(preview=form.cleaned_data['answer'])
            return self.render_to_response(context=ctx)
        return HttpResponseBadRequest()

    def get_question(self):
        return Question.objects.get(pk=self.kwargs['pk'])
```

`CreateAnswerView`类遵循与`AskQuestionView`类类似的模式：

+   这是一个`CreateView`

+   它受`LoginRequiredMixin`保护

+   它使用`get_initial（）`为其表单提供初始参数，以便恶意用户无法篡改与答案相关的问题或用户

+   它使用`form_valid（）`来执行预览或保存操作

主要的区别是我们需要在 `CreateAnswerView` 中添加一个 `get_question()` 方法来检索我们要回答的问题。`kwargs['pk']` 将由我们将创建的 `path` 填充（就像我们为 `QuestionDetailView` 做的那样）。

接下来，让我们创建模板。

# 创建 create_answer.html

这个模板将能够利用我们已经创建的常见模板元素，使渲染问题和答案表单更容易。

让我们在 `django/qanda/templates/qanda/create_answer.html` 中创建它：

```py
{% extends "base.html" %}
{% load markdownify %}

{% block body %}
  {% include 'qanda/common/display_question.html' %}
  {% if preview %}
    <div class="card question-preview" >
      <div class="card-header" >
        Answer Preview
      </div >
      <div class="card-body" >
        {{ preview|markdownify }}
      </div >
    </div >
  {% endif %}
  {% include 'qanda/common/post_answer.html' with answer_form=form %}
{% endblock %}
```

前面的模板介绍了 `{% include %}` 的新用法。当我们使用 `with` 参数时，我们可以传递一系列新名称，这些值应该在子模板的上下文中具有。在我们的情况下，我们只会将 `answer_form` 添加到 `post_answer.html` 的上下文中。其余的上下文仍然被传递给 `{% include %}`。如果我们在 `{% include %}` 的最后一个参数中添加 `only`，我们可以阻止其余的上下文被传递。

# 将请求路由到 CreateAnswerView

最后一步是通过在 `qanda/urls.py` 的 `urlpatterns` 列表中添加一个新的 `path` 来将 `CreateAnswerView` 连接到 `qanda` URLConf 中：

```py
   path('q/<int:pk>/answer', views.CreateAnswerView.as_view(),
         name='answer_question'),
```

接下来，我们将创建一个视图来处理 `AnswerAcceptanceForm`。

# 创建 UpdateAnswerAcceptanceView

我们在 `list_answers.html` 模板中使用的 `accept_form` 和 `reject_form` 变量需要一个视图来处理它们的表单提交。让我们将其添加到 `django/qanda/views.py` 中：

```py
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import UpdateView

from qanda.forms import AnswerAcceptanceForm
from qanda.models import Answer

class UpdateAnswerAcceptance(LoginRequiredMixin, UpdateView):
    form_class = AnswerAcceptanceForm
    queryset = Answer.objects.all()

    def get_success_url(self):
        return self.object.question.get_absolute_url()

    def form_invalid(self, form):
        return HttpResponseRedirect(
            redirect_to=self.object.question.get_absolute_url())
```

`UpdateView` 的工作方式类似于 `DetailView`（因为它在单个模型上工作）和 `CreateView`（因为它处理一个表单）。`CreateView` 和 `UpdateView` 共享一个共同的祖先：`ModelFormMixin`。`ModelFormMixin` 为我们提供了我们过去经常使用的钩子：`form_valid()`、`get_success_url()` 和 `form_invalid()`。

由于这个表单的简单性，我们将通过将用户重定向到问题来响应无效的表单。

接下来，让我们将其添加到我们的 URLConf 中的 `django/qanda/urls.py` 文件中：

```py
   path('a/<int:pk>/accept', views.UpdateAnswerAcceptance.as_view(),
         name='update_answer_acceptance'),
```

记得在你的 `path()` 对象的第一个参数中有一个名为 `pk` 的参数，这样 `UpdateView` 就可以检索到正确的 `Answer`。

接下来，让我们创建一个每日问题列表。

# 创建每日问题页面

为了帮助人们找到问题，我们将创建每天问题的列表。

Django 提供了创建年度、月度、周度和每日归档视图的视图。在我们的情况下，我们将使用 `DailyArchiveView`，但它们基本上都是一样的。它们从 URL 的路径中获取一个日期，并在该期间搜索所有相关内容。

让我们使用 Django 的 `DailyArchiveView` 来构建一个每日问题列表。

# 创建 DailyQuestionList 视图

让我们将我们的 `DailyQuestionList` 视图添加到 `django/qanda/views.py` 中：

```py
from django.views.generic import DayArchiveView

from qanda.models import Question

class DailyQuestionList(DayArchiveView):
    queryset = Question.objects.all()
    date_field = 'created'
    month_format = '%m'
    allow_empty = True
```

`DailyQuestionList` 不需要覆盖 `DayArchiveView` 的任何方法，只需让 Django 做这项工作。让我们看看它是如何做到的。

`DayArchiveView` 期望在 URL 的路径中获取一个日期、月份和年份。我们可以使用 `day_format`、`month_format` 和 `year_format` 来指定这些的格式。在我们的情况下，我们将期望的格式更改为 `'%m'`，这样月份就会被解析为一个数字，而不是默认的 `'%b'`，这是月份的简称。这些格式与 Python 的标准 `datetime.datetime.strftime` 相同。一旦 `DayArchiveView` 有了日期，它就会使用该日期来过滤提供的 `queryset`，使用在 `date_field` 属性中命名的字段。`queryset` 按日期排序。如果 `allow_empty` 为 `True`，那么结果将被渲染，否则将抛出 404 异常，对于没有要列出的项目的日期。为了渲染模板，对象列表被传递到模板中，就像 `ListView` 一样。默认模板假定遵循 `appname/modelname_archive_day.html` 的格式。

接下来，让我们为这个视图创建模板。

# 创建每日问题列表模板

让我们将我们的模板添加到 `django/qanda/templates/qanda/question_archive_day.html` 中：

```py
{% extends "base.html" %}

{% block title %} Questions on {{ day }} {% endblock %}

{% block body %}
  <div class="col-sm-12" >
    <h1 >Highest Voted Questions of {{ day }}</h1 >
    <ul >
      {% for question in object_list %}
        <li >
          {{ question.votes }}
          <a href="{{ question.get_absolute_url }}" >
            {{ question }}
          </a >
          by
            {{ question.user }}
          on {{ question.created }}
        </li >
      {% empty %}
        <li>Hmm... Everyone thinks they know everything today.</li>
      {% endfor %}
    </ul >
    <div>
      {% if previous_day %}
        <a href="{% url "qanda:daily_questions" year=previous_day.year month=previous_day.month day=previous_day.day %}" >
           << Previous Day
        </a >
      {% endif %}
      {% if next_day %}
        <a href="{% url "qanda:daily_questions" year=next_day.year month=next_day.month day=next_day.day %}" >
          Next Day >>
        </a >
      {% endif %}
    </div >
  </div >
{% endblock %}
```

问题列表就像人们所期望的那样，即一个带有 `{% for %}` 循环创建 `<li>` 标签和链接的 `<ul>` 标签。

`DailyArchiveView`（以及所有日期存档视图）的一个便利之处是它们提供其模板的上下文，包括下一个和上一个日期。这些日期让我们在日期之间创建一种分页。

# 将请求路由到 DailyQuestionLists

最后，我们将创建一个`path`到我们的`DailyQuestionList`视图，以便我们可以将请求路由到它：

```py
    path('daily/<int:year>/<int:month>/<int:day>/',
         views.DailyQuestionList.as_view(),
         name='daily_questions'),
```

接下来，让我们创建一个视图来代表*今天*的问题。

# 获取今天的问题列表

拥有每日存档是很好的，但我们希望提供一种方便的方式来访问今天的存档。我们将使用`RedirectView`来始终将用户重定向到今天日期的`DailyQuestionList`。

让我们将其添加到`django/qanda/views.py`中：

```py
class TodaysQuestionList(RedirectView):
    def get_redirect_url(self, *args, **kwargs):
        today = timezone.now()
        return reverse(
            'questions:daily_questions',
            kwargs={
                'day': today.day,
                'month': today.month,
                'year': today.year,
            }
        )
```

`RedirectView`是一个简单的视图，返回 301 或 302 重定向响应。我们使用 Django 的`django.util.timezone`根据 Django 的配置获取今天的日期。默认情况下，Django 使用**协调世界时**（**UTC**）进行配置。由于时区的复杂性，通常最简单的方法是在 UTC 中跟踪所有内容，然后在客户端上调整显示。

我们现在已经为我们的初始`qanda`应用程序创建了所有的视图，让用户提问和回答问题。提问者还可以接受问题的答案。

接下来，让我们让用户实际上可以使用`user`应用程序登录、注销和注册。

# 创建用户应用程序

正如我们之前提到的，Django 应用程序应该有一个明确的范围。为此，我们将创建一个单独的 Django 应用程序来管理用户，我们将其称为`user`。我们不应该将我们的用户管理代码放在`qanda`或者`user`应用程序中的`Question`模型。

让我们使用`manage.py`创建应用：

```py
$ python manage.py startapp user
```

然后，将其添加到`django/config/settings.py`的`INSTALLED_APPS`列表中：

```py
INSTALLED_APPS = [
    'user',
    'qanda',

    'markdownify',
    'crispy_forms',

    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]
```

特别重要的是要将`user`应用程序*放在*`admin`应用程序之前，因为它们都将定义登录模板。先到达的应用程序将首先解析其登录模板。我们不希望我们的用户被重定向到管理员应用程序。

接下来，让我们在`django/user/urls.py`中为我们的`user`应用程序创建一个 URLConf：

```py
from django.urls import path

import user.views

app_name = 'user'
urlpatterns = [
]
```

现在，我们将在`django/config/urls.py`中的主 URLConf 中包含`user`应用程序的 URLConf：

```py
from django.contrib import admin
from django.urls import path, include

import qanda.urls
import user.urls

urlpatterns = [
    path('admin/', admin.site.urls),
    path('user/', include(user.urls, namespace='user')),
    path('', include(qanda.urls, namespace='questions')),
]
```

现在我们已经配置了我们的应用程序，我们可以添加我们的登录和注销视图。

# 使用 Django 的 LoginView 和 LogoutView

为了提供登录和注销功能，我们将使用`django.contrib.auth`应用提供的视图。让我们更新`django/users/urls.py`来引用它们：

```py
from django.urls import path

import user.views

app_name = 'user'
urlpatterns = [
    path('login', LoginView.as_view(), name='login'),
    path('logout', LogoutView.as_view(), name='logout'),
]
```

这些视图负责登录和注销用户。然而，登录视图需要一个模板来渲染得漂亮。`LoginView`期望它在`registration/login.html`名称下。

我们将模板放在`django/user/templates/registration/login.html`中：

```py
{% extends "base.html" %}
{% load crispy_forms_tags %}

{% block title %} Login - {{ block.super }} {% endblock %}

{% block body %}
  <h1>Login</h1>
  <form method="post" class="col-sm-6">
    {% csrf_token %}
    {{ form|crispy }}
    <button type="submit" class="btn btn-primary">Login</button>
  </form>
{% endblock %}
```

`LogoutView`不需要一个模板。

现在，我们需要通知我们 Django 项目的`settings.py`关于登录视图的位置以及用户登录和注销时应执行的功能。让我们在`django/config/settings.py`中添加一些设置：

```py
LOGIN_URL = 'user:login'
LOGIN_REDIRECT_URL = 'questions:index'
LOGOUT_REDIRECT_URL = 'questions:index'
```

这样，`LoginRequiredMixin`就可以知道我们需要将未经身份验证的用户重定向到哪个视图。我们还通知了`django.contrib.auth`的`LoginView`和`LogoutView`在用户登录和注销时分别将用户重定向到哪里。

接下来，让我们为用户提供一种注册网站的方式。

# 创建 RegisterView

Django 不提供用户注册视图，但如果我们使用`django.conrib.auth.models.User`作为用户模型，它确实提供了一个`UserCreationForm`。由于我们使用`django.conrib.auth.models.User`，我们可以为我们的注册视图使用一个简单的`CreateView`：

```py
from django.contrib.auth.forms import UserCreationForm
from django.views.generic.edit import CreateView

class RegisterView(CreateView):
    template_name = 'user/register.html'
    form_class = UserCreationForm
```

现在，我们只需要在`django/user/templates/register.html`中创建一个模板：

```py
{% extends "base.html" %}
{% load crispy_forms_tags %}
{% block body %}
  <div class="col-sm-12">
    <h1 >Register for MyQA</h1 >
    <form method="post" >
      {% csrf_token %}
      {{ form | crispy }}
      <button type="submit" class="btn btn-primary" >
        Register
      </button >
    </form >
  </div >
{% endblock %}
```

同样，我们的模板遵循了一个熟悉的模式，类似于我们在过去的视图中看到的。我们使用我们的基本模板、块和 Django Crispy Form 来快速简单地创建我们的页面。

最后，我们可以在`user` URLConf 的`urlpatterns`列表中添加一个`path`到该视图：

```py
path('register', user.views.RegisterView.as_view(), name='register'),
```

# 更新 base.html 导航

现在我们已经创建了所有的视图，我们可以更新我们基础模板的`<nav>`来列出所有我们的 URL：

```py
{% load static %}
<!DOCTYPE html>
<html lang="en" >
<head >
{# skipping unchanged head contents #}
</head >
<body >
<nav class="navbar navbar-expand-lg  bg-light" >
  <div class="container" >
    <a class="navbar-brand" href="/" >Answerly</a >
    <ul class="navbar-nav" >
      <li class="nav-item" >
        <a class="nav-link" href="{% url "qanda:ask" %}" >Ask</a >
      </li >
      <li class="nav-item" >
        <a
            class="nav-link"
            href="{% url "qanda:index" %}" >
          Today's  Questions
        </a >
      </li >
      {% if user.is_authenticated %}
        <li class="nav-item" >
          <a class="nav-link" href="{% url "user:logout" %}" >Logout</a >
        </li >
      {% else %}
        <li class="nav-item" >
          <a class="nav-link" href="{% url "user:login" %}" >Login</a >
        </li >
        <li class="nav-item" >
          <a class="nav-link" href="{% url "user:register" %}" >Register</a >
        </li >
      {% endif %}
    </ul >
  </div >
</nav >
<div class="container" >
  {% block body %}{% endblock %}
</div >
</body >
</html >
```

太好了！现在我们的用户可以随时访问我们网站上最重要的页面。

# 运行开发服务器

最后，我们可以使用以下命令访问我们的开发服务器：

```py
$ cd django
$ python manage.py runserver
```

现在我们可以在浏览器中打开网站 [`localhost:8000/`](http://localhost::8000/)。

# 总结

在本章中，我们开始了 Answerly 项目。Answerly 由两个应用程序（`user`和`qanda`）组成，通过 PyPI 安装了两个第三方应用程序（Markdownify 和 Django Crispy Forms），以及一些 Django 内置应用程序（`django.contrib.auth`被直接使用）。

已登录用户现在可以提问，回答问题，并接受答案。我们还可以看到每天投票最高的问题。

接下来，我们将通过使用 ElasticSearch 添加搜索功能，帮助用户更轻松地发现问题。
