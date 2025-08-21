# 第三章：海报、头像和安全性

电影是一种视觉媒体，所以电影数据库至少应该有图片。让用户上传文件可能会带来很大的安全隐患；因此，在本章中，我们将一起讨论这两个主题。

在本章中，我们将做以下事情：

+   为每部电影添加一个允许用户上传图像的文件上传功能

+   检查**开放式 Web 应用安全项目**（**OWASP**）风险前 10 名清单

我们将在进行文件上传时检查安全性的影响。此外，我们将看看 Django 在哪些方面可以帮助我们，在哪些方面我们必须做出谨慎的设计决策。

让我们从向 MyMDB 添加文件上传开始。

# 将文件上传到我们的应用程序

在本节中，我们将创建一个模型，用于表示和管理用户上传到我们网站的文件；然后，我们将构建一个表单和视图来验证和处理这些上传。

# 配置文件上传设置

在我们开始实现文件上传之前，我们需要了解文件上传取决于一些必须在生产和开发中不同的设置。这些设置会影响文件的存储和提供方式。

Django 有两组文件设置：`STATIC_*`和`MEDIA_*`。**静态文件**是我们项目的一部分，由我们开发的文件（例如 CSS 和 JavaScript）。**媒体文件**是用户上传到我们系统的文件。媒体文件不应该受信任，绝对*不*应该被执行。

我们需要在我们的`django/conf/settings.py`中设置两个新的设置：

```py
MEDIA_URL = '/uploaded/'
MEDIA_ROOT = os.path.join(BASE_DIR, '../media_root')
```

`MEDIA_URL`是将提供上传文件的 URL。在开发中，这个值并不太重要，只要它不与我们的视图之一的 URL 冲突即可。在生产中，上传的文件应该从与提供我们应用程序的域名（而不是子域名）不同的域名提供。一个用户的浏览器如果被欺骗执行了来自与我们应用程序相同的域名（或子域名）的文件，那么它将信任该文件的 cookie（包括用户的会话 ID）。所有浏览器的默认策略称为**同源策略**。我们将在第五章 *使用 Docker 部署*中再次讨论这个问题。

`MEDIA_ROOT`是 Django 应该保存代码的目录路径。我们希望确保这个目录不在我们的代码目录下，这样它就不会意外地被检入版本控制，也不会意外地被授予任何慷慨的权限（例如执行权限），我们授予我们的代码库。

在生产中，我们还有其他设置需要配置，比如限制请求体的大小，但这些将作为第五章 *使用 Docker 部署*的一部分来完成。

接下来，让我们创建`media_root`目录：

```py
$ mkdir media_root
$ ls
django                 media_root              requirements.dev.txt
```

太好了！接下来，让我们创建我们的`MovieImage`模型。

# 创建 MovieImage 模型

我们的`MovieImage`模型将使用一个名为`ImageField`的新字段来保存文件，并*尝试*验证文件是否为图像。尽管`ImageField`确实尝试验证字段，但这并不足以阻止一个恶意用户制作一个故意恶意的文件（但会帮助一个意外点击了`.zip`而不是`.png`的用户）。Django 使用`Pillow`库来进行此验证；因此，让我们将`Pillow`添加到我们的要求文件`requirements.dev.txt`中：

```py
Pillow<4.4.0
```

然后，使用`pip`安装我们的依赖项：

```py
$ pip install -r requirements.dev.txt
```

现在，我们可以创建我们的模型：

```py
from uuid import uuid4

from django.conf import settings
from django.db import models

def movie_directory_path_with_uuid(
        instance, filename):
    return '{}/{}'.format(
        instance.movie_id, uuid4())

class MovieImage(models.Model):
    image = models.ImageField(
        upload_to=movie_directory_path_with_uuid)
    uploaded = models.DateTimeField(
        auto_now_add=True)
    movie = models.ForeignKey(
        'Movie', on_delete=models.CASCADE)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE)
```

`ImageField`是`FileField`的一个专门版本，它使用`Pillow`来确认文件是否为图像。`ImageField`和`FileField`与 Django 的文件存储 API 一起工作，该 API 提供了一种存储和检索文件以及读写文件的方式。默认情况下，Django 使用`FileSystemStorage`，它实现了存储 API 以在本地文件系统上存储数据。这对于开发来说已经足够了，但我们将在第五章中探讨替代方案，*使用 Docker 部署*。

我们使用了`ImageField`的`upload_to`参数来指定一个函数来生成上传文件的名称。我们不希望用户能够指定系统中文件的名称，因为他们可能会选择滥用我们用户的信任并让我们看起来很糟糕的名称。我们使用一个函数来将给定电影的所有图片存储在同一个目录中，并使用`uuid4`为每个文件生成一个通用唯一名称（这也避免了名称冲突和处理文件互相覆盖）。

我们还记录了谁上传了文件，这样如果我们发现了一个坏文件，我们就有线索可以找到其他坏文件。

现在让我们进行迁移并应用它：

```py
$ python manage.py makemigrations core
Migrations for 'core':
  core/migrations/0004_movieimage.py
    - Create model MovieImage
$ python manage.py migrate core
Operations to perform:
  Apply all migrations: core
Running migrations:
  Applying core.0004_movieimage... OK
```

接下来，让我们为我们的`MovieImage`模型构建一个表单，并在我们的`MovieDetail`视图中使用它。

# 创建和使用 MovieImageForm

我们的表单将与我们的`VoteForm`非常相似，它将隐藏和禁用`movie`和`user`字段，这些字段对于我们的模型是必要的，但是从客户端信任是危险的。让我们将它添加到`django/core/forms.py`中：

```py
from django import forms

from core.models import MovieImage

class MovieImageForm(forms.ModelForm):

    movie = forms.ModelChoiceField(
        widget=forms.HiddenInput,
        queryset=Movie.objects.all(),
        disabled=True
    )

    user = forms.ModelChoiceField(
        widget=forms.HiddenInput,
        queryset=get_user_model().
            objects.all(),
        disabled=True,
    )

    class Meta:
        model = MovieImage
        fields = ('image', 'user', 'movie')
```

我们不会用自定义字段或小部件覆盖`image`字段，因为`ModelForm`类将自动提供正确的`<input type="file">`。

现在，我们可以在`MovieDetail`视图中使用它：

```py
from django.views.generic import DetailView

from core.forms import (VoteForm, 
    MovieImageForm,)
from core.models import Movie

class MovieDetail(DetailView):
    queryset = Movie.objects.all_with_related_persons_and_score()

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx['image_form'] = self.movie_image_form()
        if self.request.user.is_authenticated:
            # omitting VoteForm code.
        return ctx

 def movie_image_form(self):
        if self.request.user.is_authenticated:
            return MovieImageForm()
        return None
```

这次，我们的代码更简单，因为用户*只能*上传新图片，不支持其他操作，这样我们可以始终提供一个空表单。然而，使用这种方法，我们仍然不显示错误消息。丢失错误消息不应被视为最佳实践。

接下来，我们将更新我们的模板以使用我们的新表单和上传的图片。

# 更新`movie_detail.html`以显示和上传图片

我们将需要对`movie_detail.html`模板进行两次更新。首先，我们需要更新我们的`main`模板块，以显示图片列表。其次，我们需要更新我们的`sidebar`模板块，以包含我们的上传表单。

首先让我们更新我们的`main`块：

```py
{% block main %}
  <div class="col" >
    <h1 >{{ object }}</h1 >
    <p class="lead" >
      {{ object.plot }}
    </p >
  </div >
  <ul class="movie-image list-inline" >
    {% for i in object.movieimage_set.all %}
      <li class="list-inline-item" >
          <img src="img/{{ i.image.url }}" >
      </li >
    {% endfor %}
  </ul >
  <p >Directed
    by {{ object.director }}</p >
 {# writers and actors html omitted #}
{% end block %}
```

我们在前面的代码中使用了`image`字段的`url`属性，它返回了`MEDIA_URL`设置与计算出的文件名连接在一起，这样我们的`img`标签就可以正确显示图片。

在`sidebar`块中，我们将添加一个上传新图片的表单：

```py
{% block sidebar %}
  {# rating div omitted #}
  {% if image_form %}
    <div >
      <h2 >Upload New Image</h2 >
      <form method="post"
            enctype="multipart/form-data"
            action="{% url 'core:MovieImageUpload' movie_id=object.id %}" >
        {% csrf_token %}
        {{ image_form.as_p }}
        <p >
          <button
              class="btn btn-primary" >
            Upload
          </button >
        </p >
      </form >
    </div >
  {% endif %}
  {# score and voting divs omitted #}
{% endblock %}
```

这与我们之前的表单非常相似。但是，我们*必须*记得在我们的`form`标签中包含`enctype`属性，以便上传的文件能够正确附加到请求中。

现在我们完成了我们的模板，我们可以创建我们的`MovieImageUpload`视图来保存我们上传的文件。

# 编写 MovieImageUpload 视图

我们倒数第二步将是在`django/core/views.py`中添加一个视图来处理上传的文件：

```py
from django.contrib.auth.mixins import (
    LoginRequiredMixin) 
from django.views.generic import CreateView

from core.forms import MovieImageForm

class MovieImageUpload(LoginRequiredMixin, CreateView):
    form_class = MovieImageForm

    def get_initial(self):
        initial = super().get_initial()
        initial['user'] = self.request.user.id
        initial['movie'] = self.kwargs['movie_id']
        return initial

    def render_to_response(self, context, **response_kwargs):
        movie_id = self.kwargs['movie_id']
        movie_detail_url = reverse(
            'core:MovieDetail',
            kwargs={'pk': movie_id})
        return redirect(
            to=movie_detail_url)

    def get_success_url(self):
        movie_id = self.kwargs['movie_id']
        movie_detail_url = reverse(
            'core:MovieDetail',
            kwargs={'pk': movie_id})
        return movie_detail_url
```

我们的视图再次将所有验证和保存模型的工作委托给`CreateView`和我们的表单。我们从请求的`user`属性中检索`user.id`属性（因为`LoginRequiredMixin`类的存在，我们可以确定用户已登录），并从 URL 中获取电影 ID，然后将它们作为初始参数传递给表单，因为`MovieImageForm`的`user`和`movie`字段是禁用的（因此它们会忽略请求体中的值）。保存和重命名文件的工作都由 Django 的`ImageField`完成。

最后，我们可以更新我们的项目，将请求路由到我们的`MovieImageUpload`视图并提供我们上传的文件。

# 将请求路由到视图和文件

在这一部分，我们将更新`core`的`URLConf`，将请求路由到我们的新`MovieImageUpload`视图，并看看我们如何在开发中提供我们上传的图片。我们将看看如何在生产中提供上传的图片第五章，*使用 Docker 部署*。

为了将请求路由到我们的`MovieImageUpload`视图，我们将更新`django/core/urls.py`：

```py
from django.urls import path

from . import views

app_name = 'core'
urlpatterns = [
    # omitted existing paths
    path('movie/<int:movie_id>/image/upload',
         views.MovieImageUpload.as_view(),
         name='MovieImageUpload'),
    # omitted existing paths
]
```

我们像往常一样添加我们的`path()`函数，并确保我们记得它需要一个名为`movie_id`的参数。

现在，Django 将知道如何路由到我们的视图，但它不知道如何提供上传的文件。

在开发中为了提供上传的文件，我们将更新`django/config/urls.py`：

```py
from django.conf import settings
from django.conf.urls.static import (
    static, )
from django.contrib import admin
from django.urls import path, include

import core.urls
import user.urls

MEDIA_FILE_PATHS = static(
    settings.MEDIA_URL,
    document_root=settings.MEDIA_ROOT)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('user/', include(
        user.urls, namespace='user')),
    path('', include(
        core.urls, namespace='core')),
] + MEDIA_FILE_PATHS
```

Django 提供了`static()`函数，它将返回一个包含单个`path`对象的列表，该对象将路由以`MEDIA_URL`开头的任何请求到`document_root`内的文件。这将为我们在开发中提供一种服务上传的图像文件的方法。这个功能不适合生产环境，如果`settings.DEBUG`为`False`，`static()`将返回一个空列表。

现在我们已经看到了 Django 核心功能的大部分，让我们讨论它如何与**开放 Web 应用程序安全项目**（**OWASP**）的十大最关键安全风险（OWASP Top 10）列表相关。

# OWASP Top 10

OWASP 是一个专注于通过为 Web 应用程序提供公正的实用安全建议来使*安全可见*的非营利慈善组织。OWASP 的所有材料都是免费和开源的。自 2010 年以来，OWASP 征求信息安全专业人员的数据，并用它来开发 Web 应用程序安全中最关键的十大安全风险的列表（OWASP Top 10）。尽管这个列表并不声称列举所有问题（它只是前十名），但它是基于安全专业人员在野外进行渗透测试和对全球公司的生产或开发中的真实代码进行代码审计时所看到的情况。

Django 被开发为尽可能地减少和避免这些风险，并在可能的情况下，为开发人员提供工具来最小化风险。

让我们列举 2013 年的 OWASP Top 10（撰写时的最新版本，2017 RC1 已被拒绝），并看看 Django 如何帮助我们减轻每个风险。

# A1 注入

自 OWASP Top 10 创建以来，这一直是头号问题。**注入**意味着用户能够注入由我们的系统或我们使用的系统执行的代码。例如，SQL 注入漏洞让攻击者在我们的数据库中执行任意 SQL 代码，这可能导致他们绕过我们几乎所有的控制和安全措施（例如，让他们作为管理员用户进行身份验证；SQL 注入漏洞可能导致 shell 访问）。对于这个问题，特别是对于 SQL 注入，最好的解决方案是使用参数化查询。

Django 通过提供`QuerySet`类来保护我们免受 SQL 注入的侵害。`QuerySet`确保它发送的所有查询都是参数化的，以便数据库能够区分我们的 SQL 代码和查询中的值。使用参数化查询将防止 SQL 注入。

然而，Django 允许使用`QuerySet.raw()`和`QuerySet.extra()`进行原始 SQL 查询。这两种方法都支持参数化查询，但开发人员必须确保他们**永远不要**使用来自用户的值通过字符串格式化（例如`str.format`）放入 SQL 查询，而是**始终**使用参数。

# A2 破坏身份验证和会话管理

**破坏身份验证**和**会话管理**指的是攻击者能够身份验证为另一个用户或接管另一个用户的会话的风险。

Django 在这里以几种方式保护我们，如下：

+   Django 的`auth`应用程序始终对密码进行哈希和盐处理，因此即使数据库被破坏，用户密码也无法被合理地破解。

+   Django 支持多种*慢速*哈希算法（例如 Argon2 和 Bcrypt），这使得暴力攻击变得不切实际。这些算法并不是默认提供的（Django 默认使用`PBDKDF2`），因为它们依赖于第三方库，但可以使用`PASSWORD_HASHERS`设置进行配置。

+   Django 会话 ID 默认情况下不会在 URL 中公开，并且登录后会更改会话 ID。

然而，Django 的加密功能始终以`settings.SECRET_KEY`字符串为种子。将`SECRET_KEY`的生产值检入版本控制应被视为安全问题。该值不应以明文形式共享，我们将在第五章 *使用 Docker 部署*中讨论。

# A3 跨站脚本攻击

**跨站脚本攻击**（**XSS**）是指攻击者能够让 Web 应用显示攻击者创建的 HTML 或 JavaScript，而不是开发者创建的 HTML 或 JavaScript。这种攻击非常强大，因为如果攻击者可以执行任意 JavaScript，那么他们可以发送请求，这些请求看起来与用户的真实请求无法区分。

Django 默认情况下会对模板中的所有变量进行 HTML 编码保护。

然而，Django 确实提供了将文本标记为安全的实用程序，这将导致值不被编码。这些应该谨慎使用，并充分了解如果滥用会造成严重安全后果。

# A4 不安全的直接对象引用

**不安全的直接对象引用**是指我们在资源引用中不安全地暴露实现细节，而没有保护资源免受非法访问/利用。例如，我们电影详细页面的`<img>`标签的`src`属性中的路径直接映射到文件系统中的文件。如果用户操纵 URL，他们可能访问他们本不应访问的图片，从而利用漏洞。或者，使用在 URL 中向用户公开的自动递增主键可以让恶意用户遍历数据库中的所有项目。这种风险的影响高度取决于暴露的资源。

Django 通过不将路由路径与视图耦合来帮助我们。我们可以根据主键进行模型查找，但并不是必须这样做，我们可以向我们的模型添加额外的字段（例如`UUIDField`）来将表的主键与 URL 中使用的 ID 解耦。在第三部分的 Mail Ape 项目中，我们将看到如何使用`UUIDField`类作为模型的主键。

# A5 安全配置错误

**安全配置错误**指的是当适当的安全机制被不当部署时所产生的风险。这种风险处于开发和运营的边界，并需要两个团队合作。例如，如果我们在生产环境中以`DEBUG`设置为`True`运行我们的 Django 应用，我们将面临在没有任何错误的情况下向公众暴露过多信息的风险。

Django 通过合理的默认设置以及 Django 项目网站上的技术和主题指南来帮助我们。Django 社区也很有帮助——他们在邮件列表和在线博客上发布信息，尽管在线博客文章应该持怀疑态度，直到你验证了它们的声明。

# A6 敏感数据暴露

**敏感数据暴露**是指敏感数据可能在没有适当授权的情况下被访问的风险。这种风险不仅仅是攻击者劫持用户会话，还包括备份存储方式、加密密钥轮换方式，以及最重要的是哪些数据实际上被视为*敏感*。这些问题的答案是项目/业务特定的。

Django 可以通过配置为仅通过 HTTPS 提供页面来帮助减少来自攻击者使用网络嗅探的意外暴露风险。

然而，Django 并不直接提供加密，也不管理密钥轮换、日志、备份和数据库本身。有许多因素会影响这种风险，这些因素超出了 Django 的范围。

# A7 缺少功能级别的访问控制

虽然 A6 指的是数据被暴露，但缺少功能级别的访问控制是指功能受到不充分保护的风险。考虑我们的`UpdateVote`视图——如果我们忘记了`LoginRequiredMixin`类，那么任何人都可以发送 HTTP 请求并更改我们用户的投票。

Django 的`auth`应用程序提供了许多有用的功能来减轻这些问题，包括超出本项目范围的权限系统，以及混合和实用程序，使使用这些权限变得简单（例如，`LoginRequiredMixin`和`PermissionRequiredMixin`）。

然而，我们需要适当地使用 Django 的工具来完成手头的工作。

# A8 跨站点请求伪造（CSRF）

**CSRF**（发音为*see surf*）是 OWASP 十大中技术上最复杂的风险。CSRF 依赖于一个事实，即每当浏览器从服务器请求任何资源时，它都会自动发送与该域关联的所有 cookie。恶意攻击者可能会欺骗我们已登录的用户之一，让其查看第三方网站上的页面（例如`malicious.example.org`），例如，带有指向我们网站的 URL 的`img`标签的`src`属性（例如，`mymdb.example.com`）。当用户的浏览器看到`src`时，它将向该 URL 发出`GET`请求，并发送与我们网站相关的所有 cookie（包括会话 ID）。

风险在于，如果我们的 Web 应用程序收到`GET`请求，它将进行用户未打算的修改。减轻此风险的方法是确保进行任何进行修改的操作（例如，`UpdateVote`）都具有唯一且不可预测的值（CSRF 令牌），只有我们的系统知道，这确认了用户有意使用我们的应用程序执行此操作。

Django 在很大程度上帮助我们减轻这种风险。Django 提供了`csrf_token`标签，使向表单添加 CSRF 令牌变得容易。Django 负责添加匹配的 cookie（用于验证令牌），并确保任何使用的动词不是`GET`、`HEAD`、`OPTIONS`或`TRACE`的请求都有有效的 CSRF 令牌进行处理。Django 进一步通过使其所有的通用编辑视图（`EditView`、`CreateView`、`DeleteView`和`FormView`）仅在`POST`上执行修改操作，而不是在`GET`上，来帮助我们做正确的事情。

然而，Django 不能拯救我们免受自身的伤害。如果我们决定禁用此功能或编写具有`GET`副作用的视图，Django 无法帮助我们。

# A9 使用已知漏洞的组件

一条链只有其最薄弱的一环那么强，有时，项目可能在其依赖的框架和库中存在漏洞。

Django 项目有一个安全团队，接受安全问题的机密报告，并有安全披露政策，以使社区了解影响其项目的问题。一般来说，Django 发布后会在首次发布后的 16 个月内获得支持（包括安全更新），但**长期支持**（**LTS**）发布将获得 3 年的支持（下一个 LTS 发布将是 Django 2.2）。

然而，Django 不会自动更新自身，也不会强制我们运行最新版本。每个部署都必须自行管理这一点。

# A10 未经验证的重定向和转发

如果我们的网站可以自动将用户重定向/转发到第三方网站，那么我们的网站就有可能被用来欺骗用户被转发到恶意网站。

Django 通过确保`LoginView`的`next`参数只会转发用户的 URL，这些 URL 是我们项目的一部分，来保护我们。

然而，Django 不能保护我们免受自身的伤害。我们必须确保我们从不使用用户提供的未经验证的数据作为 HTTP 重定向或转发的基础。

# 总结

在本节中，我们已更新我们的应用程序，以便用户上传与电影相关的图像，并审查了 OWASP 十大。我们介绍了 Django 如何保护我们，以及我们需要保护自己的地方。

接下来，我们将构建一个前十名电影列表，并看看如何使用缓存来避免每次扫描整个数据库。
