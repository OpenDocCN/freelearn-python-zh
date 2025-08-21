# 第三章：表单和视图

在本章中，我们将涵盖以下主题：

+   创建一个带有 CRUDL 功能的应用程序

+   保存模型实例的作者

+   上传图片

+   使用自定义模板创建表单布局

+   使用 django-crispy-forms 创建表单布局

+   使用表单集

+   过滤对象列表

+   管理分页列表

+   组合基于类的视图

+   提供 Open Graph 和 Twitter Card 数据

+   提供 schema.org 词汇

+   生成 PDF 文档

+   使用 Haystack 和 Whoosh 实现多语言搜索

+   使用 Elasticsearch DSL 实现多语言搜索

# 介绍

虽然数据库结构在模型中定义，但视图提供了必要的端点，以向用户显示内容或让他们输入新的和更新的数据。在本章中，我们将重点关注用于管理表单、列表视图和生成 HTML 以外的替代输出的视图。在最简单的示例中，我们将把 URL 规则和模板的创建留给您。

# 技术要求

要使用本章的代码，您将需要最新稳定版本的 Python、MySQL 或 PostgreSQL 数据库，以及带有虚拟环境的 Django 项目。一些教程将需要特定的 Python 依赖项。此外，为了生成 PDF 文档，您将需要`cairo`、`pango`、`gdk-pixbuf`和`libffi`库。对于搜索，您将需要一个 Elasticsearch 服务器。您将在相应的教程中获得更多关于它们的详细信息。

本章中的大多数模板将使用 Bootstrap 4 CSS 框架，以获得更美观的外观和感觉。

您可以在 GitHub 存储库的`ch03`目录中找到本章的所有代码：[`github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition`](https://github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition)。

# 创建一个带有 CRUDL 功能的应用程序

在计算机科学中，**CRUDL**首字母缩写代表**创建**、**读取**、**更新**、**删除**和**列表**功能。许多具有交互功能的 Django 项目将需要您实现所有这些功能来管理网站上的数据。在本教程中，我们将看到如何为这些基本功能创建 URL 和视图。

# 准备工作

让我们创建一个名为`ideas`的新应用程序，并将其放入设置中的`INSTALLED_APPS`中。在该应用程序中创建以下`Idea`模型，并在该模型内部创建`IdeaTranslations`模型以进行翻译：

```py
# myproject/apps/idea/models.py import uuid

from django.db import models
from django.urls import reverse
from django.conf import settings
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.model_fields import TranslatedField
from myproject.apps.core.models import (
    CreationModificationDateBase, UrlBase
)

RATING_CHOICES = (
    (1, "★☆☆☆☆"), 
    (2, "★★☆☆☆"), 
    (3, "★★★☆☆"), 
    (4, "★★★★☆"),
    (5, "★★★★★"),
)

class Idea(CreationModificationDateBase, UrlBase):
    uuid = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False
    )
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        verbose_name=_("Author"),
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        related_name="authored_ideas",
    )
    title = models.CharField(_("Title"), max_length=200)
    content = models.TextField(_("Content"))

    categories = models.ManyToManyField(
        "categories.Category",
        verbose_name=_("Categories"),
        related_name="category_ideas",
    )
    rating = models.PositiveIntegerField(
        _("Rating"), choices=RATING_CHOICES, blank=True, null=True
    )
    translated_title = TranslatedField("title")
    translated_content = TranslatedField("content")

    class Meta:
        verbose_name = _("Idea")
        verbose_name_plural = _("Ideas")

    def __str__(self):
        return self.title

    def get_url_path(self):
        return reverse("ideas:idea_detail", kwargs={"pk": self.pk})

class IdeaTranslations(models.Model):
    idea = models.ForeignKey(
        Idea,
        verbose_name=_("Idea"),
        on_delete=models.CASCADE,
        related_name="translations",
    )
    language = models.CharField(_("Language"), max_length=7)

    title = models.CharField(_("Title"), max_length=200)
    content = models.TextField(_("Content"))

    class Meta:
        verbose_name = _("Idea Translations")
        verbose_name_plural = _("Idea Translations")
        ordering = ["language"]
        unique_together = [["idea", "language"]]

    def __str__(self):
        return self.title
```

我们在这里使用了上一章的几个概念：我们从模型混合继承，并利用了模型翻译表。在*使用模型混合*和*使用模型翻译表*教程中了解更多。我们将在本章的所有教程中使用`ideas`应用程序和这些模型。

此外，创建一个类似的`categories`应用程序，其中包括`Category`和`CategoryTranslations`模型：

```py
# myproject/apps/categories/models.py
from django.db import models
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.model_fields import TranslatedField

class Category(models.Model):
    title = models.CharField(_("Title"), max_length=200)

    translated_title = TranslatedField("title")

    class Meta:
        verbose_name = _("Category")
        verbose_name_plural = _("Categories")

    def __str__(self):
        return self.title

class CategoryTranslations(models.Model):
    category = models.ForeignKey(
        Category,
        verbose_name=_("Category"),
        on_delete=models.CASCADE,
        related_name="translations",
    )
    language = models.CharField(_("Language"), max_length=7)

    title = models.CharField(_("Title"), max_length=200)

    class Meta:
        verbose_name = _("Category Translations")
        verbose_name_plural = _("Category Translations")
        ordering = ["language"]
        unique_together = [["category", "language"]]

    def __str__(self):
        return self.title
```

# 如何做...

Django 中的 CRUDL 功能包括表单、视图和 URL 规则。让我们创建它们：

1.  在`ideas`应用程序中添加一个新的`forms.py`文件，其中包含用于添加和更改`Idea`模型实例的模型表单：

```py
# myprojects/apps/ideas/forms.py from django import forms
from .models import Idea

class IdeaForm(forms.ModelForm):
    class Meta:
        model = Idea
        fields = "__all__"
```

1.  在`ideas`应用程序中添加一个新的`views.py`文件，其中包含操作`Idea`模型的视图：

```py
# myproject/apps/ideas/views.py from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, DetailView

from .forms import IdeaForm
from .models import Idea

class IdeaList(ListView):
    model = Idea

class IdeaDetail(DetailView):
    model = Idea
    context_object_name = "idea"

@login_required
def add_or_change_idea(request, pk=None):
    idea = None
    if pk:
        idea = get_object_or_404(Idea, pk=pk)

    if request.method == "POST":
        form = IdeaForm(
            data=request.POST, 
            files=request.FILES, 
            instance=idea
        )

        if form.is_valid():
            idea = form.save()
            return redirect("ideas:idea_detail", pk=idea.pk)
    else:
        form = IdeaForm(instance=idea)

    context = {"idea": idea, "form": form}
    return render(request, "ideas/idea_form.html", context)

@login_required
def delete_idea(request, pk):
    idea = get_object_or_404(Idea, pk=pk)
    if request.method == "POST":
        idea.delete()
        return redirect("ideas:idea_list")
    context = {"idea": idea}
    return render(request, "ideas/idea_deleting_confirmation.html", context)
```

1.  在`ideas`应用程序中创建`urls.py`文件，其中包含 URL 规则：

```py
# myproject/apps/ideas/urls.py from django.urls import path

from .views import (
    IdeaList,
    IdeaDetail,
    add_or_change_idea,
    delete_idea,
)

urlpatterns = [
    path("", IdeaList.as_view(), name="idea_list"),
    path("add/", add_or_change_idea, name="add_idea"),
    path("<uuid:pk>/", IdeaDetail.as_view(), name="idea_detail"),
    path("<uuid:pk>/change/", add_or_change_idea,  
     name="change_idea"),
    path("<uuid:pk>/delete/", delete_idea, name="delete_idea"),
]
```

1.  现在，让我们将这些 URL 规则插入到项目的 URL 配置中。我们还将包括 Django 贡献的`auth`应用程序中的帐户 URL 规则，以便我们的`@login_required`装饰器正常工作：

```py
# myproject/urls.py from django.contrib import admin
from django.conf.urls.i18n import i18n_patterns
from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect

urlpatterns = i18n_patterns(
    path("", lambda request: redirect("ideas:idea_list")),
    path("admin/", admin.site.urls),
    path("accounts/", include("django.contrib.auth.urls")),
 path("ideas/", include(("myproject.apps.ideas.urls", "ideas"), 
     namespace="ideas")),
)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static("/media/", document_root=settings.MEDIA_ROOT)
```

1.  现在您应该能够创建以下模板：

+   `registration/login.html`中带有登录表单

+   `ideas/idea_list.html`中包含一个想法列表

+   `ideas/idea_detail.html`中包含有关想法的详细信息

+   `ideas/idea_form.html`中包含添加或更改想法的表单

+   `ideas/idea_deleting_confirmation.html`中包含一个空表单，用于确认删除想法

在模板中，您可以通过命名空间和路径名称来访问`ideas`应用程序的 URL，如下所示：

```py
{% load i18n %}
<a href="{% url 'ideas:change_idea' pk=idea.pk %}">{% trans "Change this idea" %}</a>
<a href="{% url 'ideas:add_idea' %}">{% trans "Add idea" %}</a>
```

如果您遇到困难或想节省时间，请查看本书的代码文件中相应的模板，您可以在[`github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition/tree/master/ch03/myproject_virtualenv/src/django-myproject/myproject/templates/ideas`](https://github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition/tree/master/ch03/myproject_virtualenv/src/django-myproject/myproject/templates/ideas)找到。

# 它是如何工作的...

在这个示例中，我们使用 UUID 字段作为`Idea`模型的主键。有了这个 ID，每个想法都有一个不可预测的唯一 URL。或者，您也可以使用 slug 字段用于 URL，但是您必须确保每个 slug 都被填充并且在整个网站中是唯一的。

出于安全原因，不建议使用默认的增量 ID 用于 URL：用户可以找出数据库中有多少项，并尝试访问下一个或上一个项目，尽管他们可能没有权限这样做。

在我们的示例中，我们使用基于类的通用视图来列出和阅读想法，并使用基于函数的视图来创建、更新和删除它们。更改数据库中记录的视图需要经过身份验证的用户，使用`@login_required`装饰器。对于所有 CRUDL 功能，使用基于类的视图或基于函数的视图也是完全可以的。

成功添加或更改想法后，用户将被重定向到详细视图。删除想法后，用户将被重定向到列表视图。

# 还有更多...

此外，您可以使用 Django 消息框架在每次成功添加、更改或删除后在页面顶部显示成功消息。

您可以在官方文档中阅读有关它们的信息：[`docs.djangoproject.com/en/2.2/ref/contrib/messages/`](https://docs.djangoproject.com/en/2.2/ref/contrib/messages/)。

# 另请参阅

+   *在第二章*的*使用模型混合*食谱中，模型和数据库结构

+   *在第二章*的*使用模型翻译表*食谱中，模型和数据库结构

+   *保存模型实例的作者*食谱

+   *在第四章*的*安排 base.html 模板*食谱中，模板和 JavaScript

# 保存模型实例的作者

每个 Django 视图的第一个参数是`HttpRequest`对象，按照惯例命名为`request`。它包含有关从浏览器或其他客户端发送的请求的元数据，包括当前语言代码、用户数据、cookie 和会话等项目。默认情况下，视图使用的表单接受 GET 或 POST 数据、文件、初始数据和其他参数；但是，它们本身并没有访问`HttpRequest`对象的能力。在某些情况下，将`HttpRequest`附加到表单中是有用的，特别是当您想要根据其他请求数据过滤表单字段的选择或处理保存诸如当前用户或 IP 之类的内容时。

在这个示例中，我们将看到一个表单的例子，其中，对于添加或更改的想法，当前用户将被保存为作者。

# 准备工作

我们将在前一个示例中进行扩展。

# 如何做...

要完成此食谱，请执行以下两个步骤：

1.  修改`IdeaForm`模型表单如下：

```py
# myprojects/apps/ideas/forms.py from django import forms
from .models import Idea

class IdeaForm(forms.ModelForm):
    class Meta:
        model = Idea
        exclude = ["author"]

 def __init__(self, request, *args, **kwargs):
 self.request = request
 super().__init__(*args, **kwargs)

 def save(self, commit=True):
 instance = super().save(commit=False)
 instance.author = self.request.user
 if commit:
 instance.save()
            self.save_m2m()
 return instance
```

1.  修改视图以添加或更改想法：

```py
# myproject/apps/ideas/views.py from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404

from .forms import IdeaForm
from .models import Idea

@login_required
def add_or_change_idea(request, pk=None):
    idea = None
    if pk:
        idea = get_object_or_404(Idea, pk=pk)

    if request.method == "POST":
        form = IdeaForm(request, data=request.POST, 
         files=request.FILES, instance=idea)

        if form.is_valid():
            idea = form.save()
            return redirect("ideas:idea_detail", pk=idea.pk)
    else:
        form = IdeaForm(request, instance=idea)

    context = {"idea": idea, "form": form}
    return render(request, "ideas/idea_form.html", context)
```

# 它是如何工作的...

让我们来看看这个表单。首先，我们从表单中排除`author`字段，因为我们希望以编程方式处理它。我们重写`__init__()`方法，接受`HttpRequest`作为第一个参数，并将其存储在表单中。模型表单的`save()`方法处理模型的保存。`commit`参数告诉模型表单立即保存实例，否则创建并填充实例，但尚未保存。在我们的情况下，我们获取实例而不保存它，然后从当前用户分配作者。最后，如果`commit`为`True`，我们保存实例。我们将调用动态添加的`save_m2m()`方法来保存多对多关系，例如类别。

在视图中，我们只需将`request`变量作为第一个参数传递给表单。

# 另请参阅

+   *使用 CRUDL 功能创建应用程序*食谱

+   *上传图像*食谱

# 上传图像

在这个食谱中，我们将看一下处理图像上传的最简单方法。我们将在`Idea`模型中添加一个`picture`字段，并为不同目的创建不同尺寸的图像版本。

# 准备工作

对于具有图像版本的图像，我们将需要`Pillow`和`django-imagekit`库。让我们在虚拟环境中使用`pip`安装它们（并将它们包含在`requirements/_base.txt`中）：

```py
(env)$ pip install Pillow
(env)$ pip install django-imagekit==4.0.2
```

然后，在设置中将`"imagekit"`添加到`INSTALLED_APPS`。

# 如何做...

执行以下步骤完成食谱：

1.  修改`Idea`模型以添加`picture`字段和图像版本规格：

```py
# myproject/apps/ideas/models.py
import contextlib
import os

from imagekit.models import ImageSpecField
from pilkit.processors import ResizeToFill

from django.db import models
from django.utils.translation import gettext_lazy as _
from django.utils.timezone import now as timezone_now

from myproject.apps.core.models import (CreationModificationDateBase, UrlBase)

def upload_to(instance, filename):
 now = timezone_now()
 base, extension = os.path.splitext(filename)
 extension = extension.lower()
 return f"ideas/{now:%Y/%m}/{instance.pk}{extension}"

class Idea(CreationModificationDateBase, UrlBase):
    # attributes and fields…
    picture = models.ImageField(
        _("Picture"), upload_to=upload_to
    )
    picture_social = ImageSpecField(
        source="picture",
        processors=[ResizeToFill(1024, 512)],
        format="JPEG",
        options={"quality": 100},
    )
    picture_large = ImageSpecField(
        source="picture", 
        processors=[ResizeToFill(800, 400)], 
        format="PNG"
    )
    picture_thumbnail = ImageSpecField(
        source="picture", 
        processors=[ResizeToFill(728, 250)], 
        format="PNG"
    )
    # other fields, properties, and  methods…

 def delete(self, *args, **kwargs):
 from django.core.files.storage import default_storage
 if self.picture:
 with contextlib.suppress(FileNotFoundError):
 default_storage.delete(
 self.picture_social.path
 )
 default_storage.delete(
 self.picture_large.path
 )
 default_storage.delete(
 self.picture_thumbnail.path
 )
 self.picture.delete()
 super().delete(*args, **kwargs)
```

1.  在`forms.py`中为`Idea`模型创建一个模型表单`IdeaForm`，就像我们在之前的食谱中所做的那样。

1.  在添加或更改想法的视图中，确保将`request.FILES`与`request.POST`一起发布到表单中：

```py
# myproject/apps/ideas/views.py from django.contrib.auth.decorators import login_required
from django.shortcuts import (render, redirect, get_object_or_404)
from django.conf import settings

from .forms import IdeaForm
from .models import Idea

@login_required
def add_or_change_idea(request, pk=None):
    idea = None
    if pk:
        idea = get_object_or_404(Idea, pk=pk)
    if request.method == "POST":
        form = IdeaForm(
            request, 
 data=request.POST, 
 files=request.FILES, 
            instance=idea,
        )

        if form.is_valid():
            idea = form.save()
            return redirect("ideas:idea_detail", pk=idea.pk)
    else:
        form = IdeaForm(request, instance=idea)

    context = {"idea": idea, "form": form}
    return render(request, "ideas/idea_form.html", context)
```

1.  在模板中，确保将编码类型设置为`"multipart/form-data"`，如下所示：

```py
<form action="{{ request.path }}" method="post" enctype="multipart/form-data">{% csrf_token %}
{{ form.as_p }}
<button type="submit">{% trans "Save" %}</button>
</form>
```

如果您正在使用`django-crispy-form`，如*使用 django-crispy-forms 创建表单布局*食谱中所述，`enctype`属性将自动添加到表单中。

# 它是如何工作的...

Django 模型表单是从模型动态创建的。它们提供了模型中指定的字段，因此您不需要在表单中手动重新定义它们。在前面的示例中，我们为`Idea`模型创建了一个模型表单。当我们保存表单时，表单知道如何将每个字段保存在数据库中，以及如何上传文件并将其保存在媒体目录中。

在我们的示例中，`upload_to()`函数用于将图像保存到特定目录，并定义其名称，以便不会与其他模型实例的文件名冲突。每个文件将保存在类似`ideas/2020/01/0422c6fe-b725-4576-8703-e2a9d9270986.jpg`的路径下，其中包括上传的年份和月份以及`Idea`实例的主键。

一些文件系统（如 FAT32 和 NTFS）每个目录可用的文件数量有限；因此，将它们按上传日期、字母顺序或其他标准划分为目录是一个好习惯。

我们使用`django-imagekit`中的`ImageSpecField`创建了三个图像版本：

+   `picture_social`用于社交分享。

+   `picture_large`用于详细视图。

+   `picture_thumbnail`用于列表视图。

图像版本未在数据库中链接，而只是保存在默认文件存储中，路径为`CACHE/images/ideas/2020/01/0422c6fe-b725-4576-8703-e2a9d9270986/`。

在模板中，您可以使用原始图像或特定图像版本，如下所示：

```py
<img src="img/strong>" alt="" />
<img src="img/strong>" alt="" />
```

在`Idea`模型定义的末尾，我们重写`delete()`方法，以便在删除`Idea`实例之前删除图像版本和磁盘上的图片。

# 另请参阅

+   *使用 django-crispy-forms 创建表单布局*食谱

+   第四章*，模板和 JavaScript*中的*安排 base.html 模板*食谱

+   *在第四章*的*提供响应式图片*食谱中

# 使用自定义模板创建表单布局

在早期版本的 Django 中，所有表单渲染都是在 Python 代码中处理的，但自从 Django 1.11 以来，引入了基于模板的表单小部件渲染。在这个食谱中，我们将研究如何使用自定义模板来处理表单小部件。我们将使用 Django 管理表单来说明自定义小部件模板如何提高字段的可用性。

# 准备工作

让我们创建`Idea`模型及其翻译的默认 Django 管理：

```py
# myproject/apps/ideas/admin.py from django import forms
from django.contrib import admin
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.admin import LanguageChoicesForm

from .models import Idea, IdeaTranslations

class IdeaTranslationsForm(LanguageChoicesForm):
    class Meta:
        model = IdeaTranslations
        fields = "__all__"

class IdeaTranslationsInline(admin.StackedInline):
    form = IdeaTranslationsForm
    model = IdeaTranslations
    extra = 0

@admin.register(Idea)
class IdeaAdmin(admin.ModelAdmin):
 inlines = [IdeaTranslationsInline]

 fieldsets = [
 (_("Author and Category"), {"fields": ["author", "categories"]}),
 (_("Title and Content"), {"fields": ["title", "content", 
         "picture"]}),
 (_("Ratings"), {"fields": ["rating"]}),
 ]
```

如果您访问想法的管理表单，它将如下所示：

![](img/d063e19c-516f-4966-bad8-b91d5a734cda.png)

# 如何做到...

要完成这个食谱，请按照以下步骤进行：

1.  通过将`"django.forms"`添加到`INSTALLED_APPS`，在模板配置中将`APP_DIRS`标志设置为`True`，并使用`"TemplatesSetting"`表单渲染器，确保模板系统能够找到自定义模板：

```py
# myproject/settings/_base.py
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
 "django.forms",
    # other apps…
]

TEMPLATES = [
    {
        "BACKEND": 
        "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "myproject", "templates")],
 "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",

                "django.contrib.messages.context_processors
                 .messages",
                "django.template.context_processors.media",
                "django.template.context_processors.static",
                "myproject.apps.core.context_processors
                .website_url",
            ]
        },
    }
]

FORM_RENDERER = "django.forms.renderers.TemplatesSetting"
```

1.  编辑`admin.py`文件如下：

```py
# myproject/apps/ideas/admin.py from django import forms
from django.contrib import admin
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.admin import LanguageChoicesForm

from myproject.apps.categories.models import Category
from .models import Idea, IdeaTranslations

class IdeaTranslationsForm(LanguageChoicesForm):
    class Meta:
        model = IdeaTranslations
        fields = "__all__"

class IdeaTranslationsInline(admin.StackedInline):
    form = IdeaTranslationsForm
    model = IdeaTranslations
    extra = 0

class IdeaForm(forms.ModelForm):
 categories = forms.ModelMultipleChoiceField(
 label=_("Categories"),
 queryset=Category.objects.all(),
 widget=forms.CheckboxSelectMultiple(),
 required=True,
 )

 class Meta:
 model = Idea
 fields = "__all__"

    def __init__(self, *args, **kwargs):
 super().__init__(*args, **kwargs)

 self.fields[
 "picture"
        ].widget.template_name = "core/widgets/image.html"

@admin.register(Idea)
class IdeaAdmin(admin.ModelAdmin):
 form = IdeaForm
    inlines = [IdeaTranslationsInline]

    fieldsets = [
        (_("Author and Category"), {"fields": ["author", 
         "categories"]}),
        (_("Title and Content"), {"fields": ["title", "content", 
         "picture"]}),
        (_("Ratings"), {"fields": ["rating"]}),
    ]
```

1.  最后，为您的图片字段创建一个模板：

```py
{# core/widgets/image.html #} {% load i18n %}

<div style="margin-left: 160px; padding-left: 10px;">
    {% if widget.is_initial %}
        <a href="{{ widget.value.url }}">
 <img src="img/{{ widget.value.url }}" width="624" 
             height="auto" alt="" />
 </a>
        {% if not widget.required %}<br />
            {{ widget.clear_checkbox_label }}:
            <input type="checkbox" name="{{ widget.checkbox_name 
             }}" id="{{ widget.checkbox_id }}">
        {% endif %}<br />
        {{ widget.input_text }}:
    {% endif %}
    <input type="{{ widget.type }}" name="{{ widget.name }}"{% 
     include "django/forms/widgets/attrs.html" %}>
</div>
<div class="help">
 {% trans "Available formats are JPG, GIF, and PNG." %}
 {% trans "Minimal size is 800 x 800 px." %}
</div>
```

# 它是如何工作的...

如果您现在查看想法的管理表单，您会看到类似这样的东西：

![](img/50557bdd-bdfe-4d29-acb7-9e64af5eb808.png)

这里有两个变化：

+   现在类别选择使用的是一个带有多个复选框的小部件。

+   现在图片字段使用特定模板呈现，显示图像预览和帮助文本，显示首选文件类型和尺寸。

我们在这里做的是覆盖了 idea 的模型表单，并修改了类别的小部件和图片字段的模板。

Django 中的默认表单渲染器是`"django.forms.renderers.DjangoTemplates"`，它只在应用程序目录中搜索模板。我们将其更改为`"django.forms.renderers.TemplatesSetting"`，以便在`DIRS`路径下的模板中也进行查找。

# 另请参阅

+   *在第二章*的*使用模型翻译表*食谱中

+   *上传图片*食谱

+   *使用 django-crispy-forms 创建表单布局*食谱

# 使用 django-crispy-forms 创建表单布局

`django-crispy-forms` Django 应用程序允许您使用以下 CSS 框架之一构建、自定义和重用表单：Uni-Form、Bootstrap 3、Bootstrap 4 或 Foundation。使用`django-crispy-forms`有点类似于 Django 贡献的管理中的字段集；但是，它更先进和可定制。您可以在 Python 代码中定义表单布局，而不必担心每个字段在 HTML 中的呈现方式。此外，如果您需要添加特定的 HTML 属性或包装，您也可以轻松实现。`django-crispy-forms`使用的所有标记都位于可以根据特定需求进行覆盖的模板中。

在这个食谱中，我们将使用 Bootstrap 4 创建一个漂亮的布局，用于添加或编辑想法的前端表单，这是一个用于开发响应式、移动优先网站项目的流行前端框架。

# 准备工作

我们将从本章中创建的`ideas`应用程序开始。接下来，我们将依次执行以下任务：

1.  确保您已经为您的站点创建了一个`base.html`模板。在第四章*的*安排 base.html 模板*食谱中了解更多。

1.  集成 Bootstrap 4 前端框架的 CSS 和 JS 文件

从[`getbootstrap.com/docs/4.3/getting-started/introduction/`](https://getbootstrap.com/docs/4.3/getting-started/introduction/)中获取到`base.html`模板。

1.  在您的虚拟环境中使用`pip`安装`django-crispy-forms`（并将其包含在`requirements/_base.txt`中）：

```py
(env)$ pip install django-crispy-forms
```

1.  确保在设置中将`"crispy_forms"`添加到`INSTALLED_APPS`中，并将`"bootstrap4"`设置为此项目中要使用的模板包：

```py
# myproject/settings/_base.py
INSTALLED_APPS = (
    # ...
    "crispy_forms",
    "ideas",
)
# ...
CRISPY_TEMPLATE_PACK = "bootstrap4"
```

# 如何做到...

按照以下步骤进行：

1.  让我们修改想法的模型表单：

```py
# myproject/apps/ideas/forms.py from django import forms
from django.utils.translation import ugettext_lazy as _
from django.conf import settings
from django.db import models

from crispy_forms import bootstrap, helper, layout

from .models import Idea

class IdeaForm(forms.ModelForm):
    class Meta:
        model = Idea
        exclude = ["author"]

    def __init__(self, request, *args, **kwargs):
        self.request = request
        super().__init__(*args, **kwargs)

 self.fields["categories"].widget = 
         forms.CheckboxSelectMultiple()

 title_field = layout.Field(
            "title", css_class="input-block-level"
        )
 content_field = layout.Field(
            "content", css_class="input-block-level", rows="3"
        )
 main_fieldset = layout.Fieldset(
            _("Main data"), title_field, content_field
        )

 picture_field = layout.Field(
            "picture", css_class="input-block-level"
        )
 format_html = layout.HTML(
 """{% include "ideas/includes
                /picture_guidelines.html" %}"""
        )

 picture_fieldset = layout.Fieldset(
 _("Picture"),
 picture_field,
 format_html,
 title=_("Image upload"),
 css_id="picture_fieldset",
 )

 categories_field = layout.Field(
            "categories", css_class="input-block-level"
        )
 categories_fieldset = layout.Fieldset(
 _("Categories"), categories_field,
            css_id="categories_fieldset"
        )

 submit_button = layout.Submit("save", _("Save"))
 actions = bootstrap.FormActions(submit_button)

 self.helper = helper.FormHelper()
 self.helper.form_action = self.request.path
 self.helper.form_method = "POST"
        self.helper.layout = layout.Layout(
 main_fieldset,
 picture_fieldset,
 categories_fieldset,
 actions,
 )

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.author = self.request.user
        if commit:
            instance.save()
            self.save_m2m()
        return instance
```

1.  然后，让我们创建`picture_guidelines.html`模板，内容如下：

```py
{# ideas/includes/picture_guidelines.html #} {% load i18n %}
<p class="form-text text-muted">
    {% trans "Available formats are JPG, GIF, and PNG." %}
    {% trans "Minimal size is 800 × 800 px." %}
</p>
```

1.  最后，让我们更新想法表单的模板：

```py
{# ideas/idea_form.html #} {% extends "base.html" %}
{% load i18n crispy_forms_tags static %}

{% block content %}
    <a href="{% url "ideas:idea_list" %}">{% trans "List of 
     ideas" %}</a>
    <h1>
        {% if idea %}
            {% blocktrans trimmed with 
             title=idea.translated_title %}
                Change Idea "{{ title }}
            {% endblocktrans %}
        {% else %}
            {% trans "Add Idea" %}
        {% endif %}
```

```py
    </h1>
    {% crispy form %}
{% endblock %}
```

# 它是如何工作的...

在想法的模型表单中，我们创建了一个包含主要字段集、图片字段集、类别字段集和提交按钮的表单助手布局。每个字段集都包含字段。任何字段集、字段或按钮都可以具有附加参数，这些参数成为字段的属性，例如`rows="3"`或`placeholder=_("Please enter a title")`。对于 HTML 的`class`和`id`属性，有特定的参数，`css_class`和`css_id`。

idea 表单页面将类似于以下内容：

![](img/188283f1-62f5-40af-b4b8-5a1adc05995b.png)

就像在上一个配方中一样，我们修改了类别字段的小部件，并为图片字段添加了额外的帮助文本。

# 还有更多...

对于基本用法，给定的示例已经足够了。但是，如果您需要项目中表单的特定标记，您仍然可以覆盖和修改`django-crispy-forms`应用程序的模板，因为 Python 文件中没有硬编码的标记，而是通过模板呈现所有生成的标记。只需将`django-crispy-forms`应用程序中的模板复制到项目的模板目录中，并根据需要进行更改。

# 另请参阅

+   *使用 CRUDL 功能创建应用程序*配方

+   *使用自定义模板创建表单布局*配方

+   *过滤对象列表*配方

+   *管理分页列表*配方

+   *组成基于类的视图*配方

+   第四章*，模板和 JavaScript*中的*安排 base.html 模板*配方

# 使用 formsets

除了普通或模型表单外，Django 还有一个表单集的概念。这些是相同类型的表单集，允许我们一次创建或更改多个实例。Django 表单集可以通过 JavaScript 进行增强，这使我们能够动态地将它们添加到页面中。这正是我们将在本配方中要做的。我们将扩展想法的表单，以允许在同一页上为不同语言添加翻译。

# 准备工作

让我们继续在上一个配方*使用 django-crispy-forms 创建表单布局*中继续工作`IdeaForm`。

# 如何做...

按照以下步骤进行：

1.  让我们修改`IdeaForm`的表单布局：

```py
# myproject/apps/ideas/forms.py from django import forms
from django.utils.translation import ugettext_lazy as _
from django.conf import settings
from django.db import models

from crispy_forms import bootstrap, helper, layout

from .models import Idea, IdeaTranslations

class IdeaForm(forms.ModelForm):
    class Meta:
        model = Idea
        exclude = ["author"]

    def __init__(self, request, *args, **kwargs):
        self.request = request
        super().__init__(*args, **kwargs)

        self.fields["categories"].widget = 
         forms.CheckboxSelectMultiple()

        title_field = layout.Field(
            "title", css_class="input-block-level"
        )
        content_field = layout.Field(
            "content", css_class="input-block-level", rows="3"
        )
        main_fieldset = layout.Fieldset(
            _("Main data"), title_field, content_field
        )

        picture_field = layout.Field(
            "picture", css_class="input-block-level"
        )
        format_html = layout.HTML(
            """{% include "ideas/includes
                /picture_guidelines.html" %}"""
        )

        picture_fieldset = layout.Fieldset(
            _("Picture"),
            picture_field,
            format_html,
            title=_("Image upload"),
            css_id="picture_fieldset",
        )

        categories_field = layout.Field(
            "categories", css_class="input-block-level"
        )
        categories_fieldset = layout.Fieldset(
            _("Categories"), categories_field,
            css_id="categories_fieldset"
        )

        inline_translations = layout.HTML(
 """{% include "ideas/forms/translations.html" %}"""
        )

        submit_button = layout.Submit("save", _("Save"))
        actions = bootstrap.FormActions(submit_button)

        self.helper = helper.FormHelper()
        self.helper.form_action = self.request.path
        self.helper.form_method = "POST"
        self.helper.layout = layout.Layout(
            main_fieldset,
            inline_translations,
            picture_fieldset,
            categories_fieldset,
            actions,
        )

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.author = self.request.user
        if commit:
            instance.save()
            self.save_m2m()
        return instance
```

1.  然后，在同一个文件的末尾添加`IdeaTranslationsForm`：

```py
class IdeaTranslationsForm(forms.ModelForm):
 language = forms.ChoiceField(
 label=_("Language"),
 choices=settings.LANGUAGES_EXCEPT_THE_DEFAULT,
 required=True,
 )

 class Meta:
 model = IdeaTranslations
 exclude = ["idea"]

 def __init__(self, request, *args, **kwargs):
 self.request = request
 super().__init__(*args, **kwargs)

 id_field = layout.Field("id")
 language_field = layout.Field(
            "language", css_class="input-block-level"
        )
 title_field = layout.Field(
            "title", css_class="input-block-level"
        )
 content_field = layout.Field(
            "content", css_class="input-block-level", rows="3"
        )
 delete_field = layout.Field("DELETE")
 main_fieldset = layout.Fieldset(
 _("Main data"),
 id_field,
 language_field,
 title_field,
 content_field,
 delete_field,
 )

 self.helper = helper.FormHelper()
 self.helper.form_tag = False
        self.helper.disable_csrf = True
        self.helper.layout = layout.Layout(main_fieldset)
```

1.  修改视图以添加或更改想法，如下所示：

```py
# myproject/apps/ideas/views.py from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.forms import modelformset_factory
from django.conf import settings

from .forms import IdeaForm, IdeaTranslationsForm
from .models import Idea, IdeaTranslations

@login_required
def add_or_change_idea(request, pk=None):
    idea = None
    if pk:
        idea = get_object_or_404(Idea, pk=pk)
    IdeaTranslationsFormSet = modelformset_factory(
 IdeaTranslations, form=IdeaTranslationsForm, 
 extra=0, can_delete=True
    )
    if request.method == "POST":
        form = IdeaForm(request, data=request.POST, 
         files=request.FILES, instance=idea)
        translations_formset = IdeaTranslationsFormSet(
 queryset=IdeaTranslations.objects.filter(idea=idea),
 data=request.POST,
 files=request.FILES,
 prefix="translations",
 form_kwargs={"request": request},
 )
        if form.is_valid() and translations_formset.is_valid():
            idea = form.save()
 translations = translations_formset.save(
 commit=False
            )
 for translation in translations:
 translation.idea = idea
 translation.save()
 translations_formset.save_m2m()
 for translation in 
             translations_formset.deleted_objects:
 translation.delete()
            return redirect("ideas:idea_detail", pk=idea.pk)
    else:
        form = IdeaForm(request, instance=idea)
 translations_formset = IdeaTranslationsFormSet(
 queryset=IdeaTranslations.objects.filter(idea=idea),
 prefix="translations",
 form_kwargs={"request": request},
 )

    context = {
        "idea": idea, 
        "form": form, 
 "translations_formset": translations_formset
    }
    return render(request, "ideas/idea_form.html", context)
```

1.  然后，让我们编辑`idea_form.html`模板，并在末尾添加对`inlines.js`脚本文件的引用：

```py
{# ideas/idea_form.html #}
{% extends "base.html" %}
{% load i18n crispy_forms_tags static %}

{% block content %}
    <a href="{% url "ideas:idea_list" %}">{% trans "List of 
     ideas" %}</a>
    <h1>
        {% if idea %}
            {% blocktrans trimmed with 
             title=idea.translated_title %}
                Change Idea "{{ title }}"
            {% endblocktrans %}
        {% else %}
            {% trans "Add Idea" %}
        {% endif %}
    </h1>
    {% crispy form %}
{% endblock %}

{% block js %}
 <script src="img/inlines.js' %}"></script>
{% endblock %}
```

1.  为翻译 formsets 创建模板：

```py
{# ideas/forms/translations.html #} {% load i18n crispy_forms_tags %}
<section id="translations_section" class="formset my-3">
    {{ translations_formset.management_form }}
    <h3>{% trans "Translations" %}</h3>
    <div class="formset-forms">
        {% for formset_form in translations_formset %}
            <div class="formset-form">
                {% crispy formset_form %}
            </div>
        {% endfor %}
    </div>
    <button type="button" class="btn btn-primary btn-sm 
     add-inline-form">{% trans "Add translations to another 
     language" %}</button>
    <div class="empty-form d-none">
        {% crispy translations_formset.empty_form %}
    </div>
</section>
```

1.  最后但并非最不重要的是，添加 JavaScript 来操作 formsets：

```py
/* site/js/inlines.js */ window.WIDGET_INIT_REGISTER = window.WIDGET_INIT_REGISTER || [];

$(function () {
    function reinit_widgets($formset_form) {
        $(window.WIDGET_INIT_REGISTER).each(function (index, func) 
        {
            func($formset_form);
        });
    }

    function set_index_for_fields($formset_form, index) {
        $formset_form.find(':input').each(function () {
            var $field = $(this);
            if ($field.attr("id")) {
                $field.attr(
                    "id",
                    $field.attr("id").replace(/-__prefix__-/, 
                     "-" + index + "-")
                );
            }
            if ($field.attr("name")) {
                $field.attr(
                    "name",
                    $field.attr("name").replace(
                        /-__prefix__-/, "-" + index + "-"
                    )
                );
            }
        });
        $formset_form.find('label').each(function () {
            var $field = $(this);
            if ($field.attr("for")) {
                $field.attr(
                    "for",
                    $field.attr("for").replace(
                        /-__prefix__-/, "-" + index + "-"
                    )
                );
            }
        });
        $formset_form.find('div').each(function () {
            var $field = $(this);
            if ($field.attr("id")) {
                $field.attr(
                    "id",
                    $field.attr("id").replace(
                        /-__prefix__-/, "-" + index + "-"
                    )
                );
            }
        });
    }

    function add_delete_button($formset_form) {
        $formset_form.find('input:checkbox[id$=DELETE]')
         .each(function () {
            var $checkbox = $(this);
            var $deleteLink = $(
                '<button class="delete btn btn-sm 
                  btn-danger mb-3">Remove</button>'
            );
            $formset_form.append($deleteLink);
            $checkbox.closest('.form-group').hide();
        });

    }

    $('.add-inline-form').click(function (e) {
        e.preventDefault();
        var $formset = $(this).closest('.formset');
        var $total_forms = $formset.find('[id$="TOTAL_FORMS"]');
        var $new_form = $formset.find('.empty-form')
        .clone(true).attr("id", null);
        $new_form.removeClass('empty-form d-none')
        .addClass('formset-form');
        set_index_for_fields($new_form, 
         parseInt($total_forms.val(), 10));
        $formset.find('.formset-forms').append($new_form);
        add_delete_button($new_form);
        $total_forms.val(parseInt($total_forms.val(), 10) + 1);
        reinit_widgets($new_form);
    });
    $('.formset-form').each(function () {
        $formset_form = $(this);
        add_delete_button($formset_form);
        reinit_widgets($formset_form);
    });
    $(document).on('click', '.delete', function (e) {
        e.preventDefault();
        var $formset = $(this).closest('.formset-form');
        var $checkbox = 
        $formset.find('input:checkbox[id$=DELETE]');
        $checkbox.attr("checked", "checked");
        $formset.hide();
    });
});
```

# 它是如何工作的...

您可能已经从 Django 模型管理中了解了 formsets。在那里，formsets 用于具有对父模型的外键的子模型的 inlines 机制。

在这个配方中，我们使用`django-crispy-forms`向 idea 表单添加了 formsets。结果将如下所示：

![](img/9685ebfb-8112-4d54-888d-0326cd58da3a.png)

正如您所看到的，我们可以将 formsets 插入到表单的末尾，也可以在其中任何位置插入，只要有意义。在我们的示例中，将翻译列出在可翻译字段之后是有意义的。

翻译表单的表单布局与`IdeaForm`的布局一样，但另外还有`id`和`DELETE`字段，这对于识别每个模型实例和从列表中删除它们是必要的。`DELETE`字段实际上是一个复选框，如果选中，将从数据库中删除相应的项目。此外，翻译的表单助手具有`form_tag=False`，它不生成`<form>`标签，以及`disable_csrf=True`，它不包括 CSRF 令牌，因为我们已经在父表单`IdeaForm`中定义了这些内容。

在视图中，如果请求是通过 POST 方法发送的，并且表单和表单集都有效，则我们保存表单并创建相应的翻译实例，但首先不保存它们。这是通过`commit=False`属性完成的。对于每个翻译实例，我们分配想法，然后将翻译保存到数据库中。最后，我们检查表单集中是否有任何标记为删除的表单，并将其从数据库中删除。

在`translations.html`模板中，我们渲染表单集中的每个表单，然后添加一个额外的隐藏空表单，JavaScript 将使用它来动态生成表单集的新表单。

每个表单集表单都有所有字段的前缀。例如，第一个表单集表单的`title`字段将具有 HTML 字段名称`"translations-0-title"`，同一表单集表单的`DELETE`字段将具有 HTML 字段名称`"translations-0-DELETE"`。空表单具有一个单词`"__prefix__"`，而不是索引，例如`"translations-__prefix__-title"`。这在 Django 级别进行了抽象，但是在使用 JavaScript 操纵表单集表单时需要了解这一点。

`inlines.js` JavaScript 执行了一些操作：

+   对于每个现有的表单集表单，它初始化其 JavaScript 驱动的小部件（您可以使用工具提示、日期或颜色选择器、地图等），并创建一个删除按钮，该按钮显示在`DELETE`复选框的位置。

+   当单击删除按钮时，它会检查`DELETE`复选框并将表单集表单隐藏在用户视野之外。

+   当单击添加按钮时，它会克隆空表单，并用下一个可用索引替换`"__prefix__"`，将新表单添加到列表中，并初始化 JavaScript 驱动的小部件。

# 还有更多...

JavaScript 使用一个数组`window.WIDGET_INIT_REGISTER`，其中包含应调用以初始化具有给定表单集表单的小部件的函数。要在另一个 JavaScript 文件中注册新函数，可以执行以下操作：

```py
/* site/js/main.js */ function apply_tooltips($formset_form) {
    $formset_form.find('[data-toggle="tooltip"]').tooltip();
}

/* register widget initialization for a formset form */
window.WIDGET_INIT_REGISTER = window.WIDGET_INIT_REGISTER || [];
window.WIDGET_INIT_REGISTER.push(apply_tooltips);
```

这将为标记中具有`data-toggle="tooltip"`和`title`属性的表单集表单中的所有出现应用工具提示功能，就像这个例子中一样：

```py
<button data-toggle="tooltip" title="{% trans 'Remove this translation' %}">{% trans "Remove" %}</button>
```

# 另请参阅

+   使用 django-crispy-forms 创建表单布局的配方

+   第四章*，模板和 JavaScript*中的*安排 base.html 模板*配方

# 过滤对象列表

在 Web 开发中，除了具有表单的视图之外，还典型地具有对象列表视图和详细视图。列表视图可以简单地列出按字母顺序或创建日期排序的对象；然而，对于大量数据来说，这并不是非常用户友好的。为了获得最佳的可访问性和便利性，您应该能够按所有可能的类别对内容进行筛选。在本配方中，我们将看到用于按任意数量的类别筛选列表视图的模式。

我们将要创建的是一个可以按作者、类别或评分进行筛选的想法列表视图。它将类似于以下内容，并应用了 Bootstrap 4：

![](img/0dc5a7c9-2254-4018-a314-e97c6be30a32.png)

# 准备工作

对于筛选示例，我们将使用具有与作者和类别相关的`Idea`模型。还可以按评分进行筛选，这是具有选择的`PositiveIntegerField`。让我们使用先前配方中创建的模型的 ideas 应用。

# 如何做...

要完成这个配方，请按照以下步骤操作：

1.  创建`IdeaFilterForm`，其中包含所有可能的类别以进行过滤：

```py
# myproject/apps/ideas/forms.py from django import forms
from django.utils.translation import ugettext_lazy as _
from django.db import models
from django.contrib.auth import get_user_model

from myproject.apps.categories.models import Category

from .models import RATING_CHOICES

User = get_user_model()

class IdeaFilterForm(forms.Form):
    author = forms.ModelChoiceField(
        label=_("Author"),
        required=False,
        queryset=User.objects.annotate(
            idea_count=models.Count("authored_ideas")
        ).filter(idea_count__gt=0),
    )
    category = forms.ModelChoiceField(
        label=_("Category"),
        required=False,
        queryset=Category.objects.annotate(
            idea_count=models.Count("category_ideas")
        ).filter(idea_count__gt=0),
    )
    rating = forms.ChoiceField(
        label=_("Rating"), required=False, choices=RATING_CHOICES
    )
```

1.  创建`idea_list`视图以列出经过筛选的想法：

```py
# myproject/apps/ideas/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings

from .forms import IdeaFilterForm
from .models import Idea, RATING_CHOICES

PAGE_SIZE = getattr(settings, "PAGE_SIZE", 24)

def idea_list(request):
    qs = Idea.objects.order_by("title")
    form = IdeaFilterForm(data=request.GET)

    facets = {
        "selected": {},
        "categories": {
            "authors": form.fields["author"].queryset,
            "categories": form.fields["category"].queryset,
            "ratings": RATING_CHOICES,
        },
    }

    if form.is_valid():
        filters = (
            # query parameter, filter parameter
            ("author", "author"),
            ("category", "categories"),
            ("rating", "rating"),
        )
        qs = filter_facets(facets, qs, form, filters)

    context = {"form": form, "facets": facets, "object_list": qs}
    return render(request, "ideas/idea_list.html", context)
```

1.  在同一文件中，添加辅助函数`filter_facets()`：

```py
def filter_facets(facets, qs, form, filters):
    for query_param, filter_param in filters:
        value = form.cleaned_data[query_param]
        if value:
            selected_value = value
            if query_param == "rating":
                rating = int(value)
                selected_value = (rating, 
                 dict(RATING_CHOICES)[rating])
            facets["selected"][query_param] = selected_value
            filter_args = {filter_param: value}
            qs = qs.filter(**filter_args).distinct()
    return qs
```

1.  如果尚未这样做，请创建`base.html`模板。您可以根据第四章*，模板和 JavaScript*中的*安排 base.html 模板*配方中提供的示例进行操作。

1.  创建`idea_list.html`模板，内容如下：

```py
{# ideas/idea_list.html #}
{% extends "base.html" %}
{% load i18n utility_tags %}

{% block sidebar %}
    {% include "ideas/includes/filters.html" %}
{% endblock %}

{% block main %}
    <h1>{% trans "Ideas" %}</h1>
    {% if object_list %}
        {% for idea in object_list %}
            <a href="{{ idea.get_url_path }}" class="d-block my-3">
                <div class="card">
                  <img src="img/{{ idea.picture_thumbnail.url }}" 
                   alt="" />
                  <div class="card-body">
                    <p class="card-text">{{ idea.translated_title 
                     }}</p>
                  </div>
                </div>
            </a>
        {% endfor %}
    {% else %}
        <p>{% trans "There are no ideas yet." %}</p>
    {% endif %}
    <a href="{% url 'ideas:add_idea' %}" class="btn btn-primary">
     {% trans "Add idea" %}</a>
{% endblock %}
```

1.  然后，让我们创建过滤器的模板。此模板使用了在第五章*，自定义模板过滤器和标记*中描述的`{% modify_query %}`模板标记，以生成过滤器的 URL：

```py
{# ideas/includes/filters.html #} {% load i18n utility_tags %}
<div class="filters panel-group" id="accordion">
    {% with title=_('Author') selected=facets.selected.author %}
        <div class="panel panel-default my-3">
            {% include "misc/includes/filter_heading.html" with 
             title=title %}
            <div id="collapse-{{ title|slugify }}"
                 class="panel-collapse{% if not selected %} 
                  collapse{% endif %}">
                <div class="panel-body"><div class="list-group">
                    {% include "misc/includes/filter_all.html" with 
                     param="author" %}
                    {% for cat in facets.categories.authors %}
                        <a class="list-group-item
                          {% if selected == cat %}
                          active{% endif %}"
                           href="{% modify_query "page" 
                            author=cat.pk %}">
                            {{ cat }}</a>
                    {% endfor %}
                </div></div>
            </div>
        </div>
    {% endwith %}
    {% with title=_('Category') selected=facets.selected
      .category %}
        <div class="panel panel-default my-3">
            {% include "misc/includes/filter_heading.html" with 
               title=title %}
            <div id="collapse-{{ title|slugify }}"
                 class="panel-collapse{% if not selected %} 
                  collapse{% endif %}">
                <div class="panel-body"><div class="list-group">
                    {% include "misc/includes/filter_all.html" with 
                      param="category" %}
                    {% for cat in facets.categories.categories %}
                        <a class="list-group-item
                          {% if selected == cat %}
                          active{% endif %}"
                           href="{% modify_query "page" 
                            category=cat.pk %}">
                            {{ cat }}</a>
                    {% endfor %}
                </div></div>
            </div>
        </div>
    {% endwith %}
    {% with title=_('Rating') selected=facets.selected.rating %}
        <div class="panel panel-default my-3">
            {% include "misc/includes/filter_heading.html" with 
              title=title %}
            <div id="collapse-{{ title|slugify }}"
                 class="panel-collapse{% if not selected %} 
                  collapse{% endif %}">
                <div class="panel-body"><div class="list-group">
                    {% include "misc/includes/filter_all.html" with 
                     param="rating" %}
                    {% for r_val, r_display in 
                      facets.categories.ratings %}
                        <a class="list-group-item
                          {% if selected.0 == r_val %}
                          active{% endif %}"
                           href="{% modify_query "page" 
                            rating=r_val %}">
                            {{ r_display }}</a>
                    {% endfor %}
                </div></div>
            </div>
        </div>
    {% endwith %}
</div>
```

1.  每个类别将遵循过滤器侧边栏中的通用模式，因此我们可以创建和包含具有共同部分的模板。首先，我们有过滤器标题，对应于`misc/includes/filter_heading.html`，如下所示：

```py
{# misc/includes/filter_heading.html #} {% load i18n %}
<div class="panel-heading">
    <h6 class="panel-title">
        <a data-toggle="collapse" data-parent="#accordion"
           href="#collapse-{{ title|slugify }}">
            {% blocktrans trimmed %}
                Filter by {{ title }}
            {% endblocktrans %}
        </a>
    </h6>
</div>
```

1.  然后，每个过滤器将包含一个重置该类别过滤的链接，在这里表示为`misc/includes/filter_all.html`。此模板还使用了`{% modify_query %}`模板标记，在第五章*，自定义模板过滤器和标记*中描述了这个模板标记：

```py
{# misc/includes/filter_all.html #} {% load i18n utility_tags %}
<a class="list-group-item {% if not selected %}active{% endif %}"
   href="{% modify_query "page" param %}">
    {% trans "All" %}
</a>
```

1.  需要将想法列表添加到`ideas`应用的 URL 中：

```py
# myproject/apps/ideas/urls.py from django.urls import path

from .views import idea_list

urlpatterns = [
    path("", idea_list, name="idea_list"),
    # other paths…
]
```

# 它是如何工作的...

我们正在使用传递给模板上下文的`facets`字典来了解我们有哪些过滤器以及选择了哪些过滤器。要深入了解，`facets`字典包括两个部分：`categories`字典和`selected`字典。`categories`字典包含所有可过滤类别的 QuerySets 或选择。`selected`字典包含每个类别的当前选定值。在`IdeaFilterForm`中，我们确保只列出至少有一个想法的类别和作者。

在视图中，我们检查表单中的查询参数是否有效，然后根据所选类别过滤对象的 QuerySet。此外，我们将选定的值设置为将传递给模板的`facets`字典。

在模板中，对于`facets`字典中的每个分类，我们列出所有类别，并将当前选定的类别标记为活动状态。如果没有为给定类别选择任何内容，我们将默认的“全部”链接标记为活动状态。

# 另请参阅

+   *管理分页列表*配方

+   *基于类的视图的组合*配方

+   *安排 base.html 模板*配方在第四章*，模板和 JavaScript*

+   在第五章*，自定义模板过滤器和标记*中描述的*创建一个模板标记来修改请求查询参数*配方

# 管理分页列表

如果您有动态更改的对象列表或其数量大于 24 个左右，您可能需要分页以提供良好的用户体验。分页不提供完整的 QuerySet，而是提供数据集中特定数量的项目，这对应于一页的适当大小。我们还显示链接，允许用户访问组成完整数据集的其他页面。Django 有用于管理分页数据的类，我们将看到如何在这个配方中使用它们。

# 准备工作

让我们从*过滤对象列表*配方开始`ideas`应用的模型、表单和视图。

# 如何做...

要将分页添加到想法的列表视图中，请按照以下步骤操作：

1.  从 Django 中导入必要的分页类到`views.py`文件中。我们将在过滤后的`idea_list`视图中添加分页管理。此外，我们将通过将`page`分配给`object_list`键，稍微修改上下文字典：

```py
# myproject/apps/ideas/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.core.paginator import (EmptyPage, PageNotAnInteger, Paginator)

from .forms import IdeaFilterForm
from .models import Idea, RATING_CHOICES

PAGE_SIZE = getattr(settings, "PAGE_SIZE", 24)

def idea_list(request):
    qs = Idea.objects.order_by("title")
    form = IdeaFilterForm(data=request.GET)

    facets = {
        "selected": {},
        "categories": {
            "authors": form.fields["author"].queryset,
            "categories": form.fields["category"].queryset,
            "ratings": RATING_CHOICES,
        },
    }

    if form.is_valid():
        filters = (
            # query parameter, filter parameter
            ("author", "author"),
            ("category", "categories"),
            ("rating", "rating"),
        )
        qs = filter_facets(facets, qs, form, filters)

 paginator = Paginator(qs, PAGE_SIZE)
 page_number = request.GET.get("page")
 try:
 page = paginator.page(page_number)
 except PageNotAnInteger:
 # If page is not an integer, show first page.
 page = paginator.page(1)
 except EmptyPage:
 # If page is out of range, show last existing page.
 page = paginator.page(paginator.num_pages)

    context = {
        "form": form,
        "facets": facets, 
        "object_list": page,
    }
    return render(request, "ideas/idea_list.html", context)
```

1.  修改`idea_list.html`模板如下：

```py
{# ideas/idea_list.html #}
{% extends "base.html" %}
{% load i18n utility_tags %}

{% block sidebar %}
    {% include "ideas/includes/filters.html" %}
{% endblock %}

{% block main %}
    <h1>{% trans "Ideas" %}</h1>
    {% if object_list %}
        {% for idea in object_list %}
            <a href="{{ idea.get_url_path }}" class="d-block my-3">
                <div class="card">
                  <img src="img/{{ idea.picture_thumbnail.url }}" 
                   alt="" />
                  <div class="card-body">
                    <p class="card-text">{{ idea.translated_title 
                     }}</p>
                  </div>
                </div>
            </a>
        {% endfor %}
        {% include "misc/includes/pagination.html" %}
    {% else %}
        <p>{% trans "There are no ideas yet." %}</p>
    {% endif %}
    <a href="{% url 'ideas:add_idea' %}" class="btn btn-primary">
     {% trans "Add idea" %}</a>
{% endblock %}
```

1.  创建分页小部件模板：

```py
{# misc/includes/pagination.html #} {% load i18n utility_tags %}
{% if object_list.has_other_pages %}
    <nav aria-label="{% trans 'Page navigation' %}">

        <ul class="pagination">
            {% if object_list.has_previous %}
                <li class="page-item"><a class="page-link" href="{% 
          modify_query page=object_list.previous_page_number %}">
                    {% trans "Previous" %}</a></li>
            {% else %}
                <li class="page-item disabled"><span class="page-
                 link">{% trans "Previous" %}</span></li>
            {% endif %}

            {% for page_number in object_list.paginator
             .page_range %}
                {% if page_number == object_list.number %}
                    <li class="page-item active">
                        <span class="page-link">{{ page_number }}
                            <span class="sr-only">{% trans 
                             "(current)" %}</span>
                        </span>
                    </li>
                {% else %}
                    <li class="page-item">
                        <a class="page-link" href="{% modify_query 
                         page=page_number %}">
                            {{ page_number }}</a>
                    </li>
                {% endif %}
            {% endfor %}

            {% if object_list.has_next %}
                <li class="page-item"><a class="page-link" href="{% 
             modify_query page=object_list.next_page_number %}">
                    {% trans "Next" %}</a></li>
            {% else %}
                <li class="page-item disabled"><span class="page-
                 link">{% trans "Next" %}</span></li>
            {% endif %}
        </ul>
    </nav>
{% endif %}
```

# 它是如何工作的...

当您在浏览器中查看结果时，您将看到分页控件，类似于以下内容：

![](img/b53de7b4-5c07-4550-a0b0-c22c96e06655.png)

我们如何实现这一点？当 QuerySet 被过滤掉时，我们将创建一个分页器对象，传递 QuerySet 和我们想要每页显示的最大项目数，这里是 24。然后，我们将从查询参数`page`中读取当前页码。下一步是从分页器中检索当前页对象。如果页码不是整数，我们获取第一页。如果页码超过可能的页数，就会检索到最后一页。页面对象具有分页小部件中所需的方法和属性，如前面截图中所示。此外，页面对象的行为类似于 QuerySet，因此我们可以遍历它并从页面的一部分获取项目。

模板中标记的片段创建了一个分页小部件，其中包含 Bootstrap 4 前端框架的标记。只有在当前页面多于一个时，我们才显示分页控件。我们有到上一页和下一页的链接，以及小部件中所有页面编号的列表。当前页码被标记为活动状态。为了生成链接的 URL，我们使用`{% modify_query %}`模板标签，稍后将在第五章*，自定义模板过滤器和标签*的*创建一个模板标签以修改请求查询参数*方法中进行描述。

# 另请参阅

+   *过滤对象列表*的方法

+   *组合基于类的视图*的方法

+   *创建一个模板标签以修改请求查询参数*的方法在第五章*，自定义模板过滤器和标签*

# 组合基于类的视图

Django 视图是可调用的，接受请求并返回响应。除了基于函数的视图之外，Django 还提供了一种将视图定义为类的替代方法。当您想要创建可重用的模块化视图或组合通用混合视图时，这种方法非常有用。在这个方法中，我们将之前显示的基于函数的`idea_list`视图转换为基于类的`IdeaListView`视图。

# 准备工作

创建与前面的*过滤对象列表*和*管理分页列表*类似的模型、表单和模板。

# 如何做...

按照以下步骤执行该方法：

1.  我们的基于类的视图`IdeaListView`将继承 Django 的`View`类并重写`get()`方法：

```py
# myproject/apps/ideas/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.core.paginator import (EmptyPage, PageNotAnInteger, Paginator)
from django.views.generic import View

from .forms import IdeaFilterForm
from .models import Idea, RATING_CHOICES

PAGE_SIZE = getattr(settings, "PAGE_SIZE", 24)

class IdeaListView(View):
    form_class = IdeaFilterForm
    template_name = "ideas/idea_list.html"

    def get(self, request, *args, **kwargs):
        form = self.form_class(data=request.GET)
        qs, facets = self.get_queryset_and_facets(form)
        page = self.get_page(request, qs)
        context = {"form": form, "facets": facets, 
         "object_list": page}
        return render(request, self.template_name, context)

    def get_queryset_and_facets(self, form):
        qs = Idea.objects.order_by("title")
        facets = {
            "selected": {},
            "categories": {
                "authors": form.fields["author"].queryset,
                "categories": form.fields["category"].queryset,
                "ratings": RATING_CHOICES,
            },
        }
        if form.is_valid():
            filters = (
                # query parameter, filter parameter
                ("author", "author"),
                ("category", "categories"),
                ("rating", "rating"),
            )
            qs = self.filter_facets(facets, qs, form, filters)
        return qs, facets

    @staticmethod
    def filter_facets(facets, qs, form, filters):
        for query_param, filter_param in filters:
            value = form.cleaned_data[query_param]
            if value:
                selected_value = value
                if query_param == "rating":
                    rating = int(value)
                    selected_value = (rating,  
                     dict(RATING_CHOICES)[rating])
                facets["selected"][query_param] = selected_value
                filter_args = {filter_param: value}
                qs = qs.filter(**filter_args).distinct()
        return qs

    def get_page(self, request, qs):
        paginator = Paginator(qs, PAGE_SIZE)
        page_number = request.GET.get("page")
        try:
            page = paginator.page(page_number)
        except PageNotAnInteger:
            page = paginator.page(1)
        except EmptyPage:
            page = paginator.page(paginator.num_pages)
        return page
```

1.  我们需要在 URL 配置中创建一个 URL 规则，使用基于类的视图。您可能之前已经为基于函数的`idea_list`视图添加了一个规则，这将是类似的。要在 URL 规则中包含基于类的视图，使用`as_view()`方法如下：

```py
# myproject/apps/ideas/urls.py from django.urls import path

from .views import IdeaListView

urlpatterns = [
path("", IdeaListView.as_view(), name="idea_list"),
    # other paths…
]
```

# 它是如何工作的...

以下是`get()`方法中发生的事情，该方法用于处理 HTTP GET 请求：

+   首先，我们创建`form`对象，将`request.GET`类似字典的对象传递给它。`request.GET`对象包含使用 GET 方法传递的所有查询变量。

+   然后，将`form`对象传递给`get_queryset_and_facets()`方法，该方法通过包含两个元素的元组返回相关值：QuerySet 和`facets`字典。

+   将当前请求对象和检索到的 QuerySet 传递给`get_page()`方法，该方法返回当前页对象。

+   最后，我们创建一个`context`字典并呈现响应。

如果需要支持，我们还可以提供一个`post()`方法，该方法用于处理 HTTP POST 请求。

# 还有更多...

正如你所看到的，`get()`和`get_page()`方法在很大程度上是通用的，因此我们可以在`core`应用程序中使用这些方法创建一个通用的`FilterableListView`类。然后，在任何需要可过滤列表的应用程序中，我们可以创建一个基于类的视图，该视图扩展了`FilterableListView`以处理这种情况。这个扩展类只需定义`form_class`和`template_name`属性以及`get_queryset_and_facets()`方法。这种模块化和可扩展性代表了基于类的视图工作的两个关键优点。

# 另请参阅

+   *过滤对象列表*的步骤

+   *管理分页列表*的步骤

# 提供 Open Graph 和 Twitter Card 数据

如果您希望网站的内容在社交网络上分享，您至少应该实现 Open Graph 和 Twitter Card 元标记。这些元标记定义了网页在 Facebook 或 Twitter 动态中的呈现方式：将显示什么标题和描述，将设置什么图片，以及 URL 是关于什么的。在这个步骤中，我们将为`idea_detail.html`模板准备社交分享。

# 准备工作

让我们继续使用之前步骤中的`ideas`应用。

# 如何操作...

按照以下步骤完成步骤：

1.  确保已创建包含图片字段和图片版本规格的`Idea`模型。有关更多信息，请参阅*使用 CRUDL 功能创建应用*和*上传图片*的步骤。

1.  确保为 ideas 准备好详细视图。有关如何操作，请参阅*使用 CRUDL 功能创建应用*的步骤。

1.  将详细视图插入 URL 配置中。如何操作在*使用 CRUDL 功能创建应用*的步骤中有描述。

1.  在特定环境的设置中，定义`WEBSITE_URL`和`MEDIA_URL`作为媒体文件的完整 URL，就像这个例子中一样：

```py
# myproject/settings/dev.py from ._base import *

DEBUG = True
WEBSITE_URL = "http://127.0.0.1:8000" # without trailing slash
MEDIA_URL = f"{WEBSITE_URL}/media/"

```

1.  在`core`应用中，创建一个上下文处理器，从设置中返回`WEBSITE_URL`变量：

```py
# myproject/apps/core/context_processors.py from django.conf import settings

def website_url(request):
    return {
        "WEBSITE_URL": settings.WEBSITE_URL,
    }
```

1.  在设置中插入上下文处理器：

```py
# myproject/settings/_base.py
TEMPLATES = [
    {
        "BACKEND": 
        "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "myproject", "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors
                 .messages",
                "django.template.context_processors.media",
                "django.template.context_processors.static",
                "myproject.apps.core.context_processors
                .website_url",
            ]
        },
    }
]
```

1.  创建包含以下内容的`idea_detail.html`模板：

```py
{# ideas/idea_detail.html #} {% extends "base.html" %}
{% load i18n %}

{% block meta_tags %}
 <meta property="og:type" content="website" />
 <meta property="og:url" content="{{ WEBSITE_URL }}
     {{ request.path }}" />
 <meta property="og:title" content="{{ idea.translated_title }}" 
     />
 {% if idea.picture_social %}
 <meta property="og:image" content=
         "{{ idea.picture_social.url }}" />
 <!-- Next tags are optional but recommended -->
        <meta property="og:image:width" content=
         "{{ idea.picture_social.width }}" />
 <meta property="og:image:height" content=
         "{{ idea.picture_social.height }}" />
 {% endif %}
 <meta property="og:description" content=
     "{{ idea.translated_content }}" />
 <meta property="og:site_name" content="MyProject" />
 <meta property="og:locale" content="{{ LANGUAGE_CODE }}" />

 <meta name="twitter:card" content="summary_large_image">
 <meta name="twitter:site" content="@DjangoTricks">
 <meta name="twitter:creator" content="@archatas">
 <meta name="twitter:url" content="{{ WEBSITE_URL }}
     {{ request.path }}">
 <meta name="twitter:title" content=
     "{{ idea.translated_title }}">
 <meta name="twitter:description" content=
     "{{ idea.translated_content }}">
 {% if idea.picture_social %}
 <meta name="twitter:image" content=
         "{{ idea.picture_social.url }}">
 {% endif %}
{% endblock %}

{% block content %}
    <a href="{% url "ideas:idea_list" %}">
     {% trans "List of ideas" %}</a>
    <h1>
        {% blocktrans trimmed with title=idea.translated_title %}
            Idea "{{ title }}"
        {% endblocktrans %}
    </h1>
    <img src="img/{{ idea.picture_large.url }}" alt="" />
    {{ idea.translated_content|linebreaks|urlize }}
    <p>
        {% for category in idea.categories.all %}
            <span class="badge badge-pill badge-info">
             {{ category.translated_title }}</span>
        {% endfor %}
    </p>
    <a href="{% url 'ideas:change_idea' pk=idea.pk %}" 
     class="btn btn-primary">{% trans "Change this idea" %}</a>
    <a href="{% url 'ideas:delete_idea' pk=idea.pk %}" 
     class="btn btn-danger">{% trans "Delete this idea" %}</a>
{% endblock %} 
```

# 它是如何工作的...

Open Graph 标签是具有以`og:`开头的特殊名称的元标记，Twitter 卡片标签是具有以`twitter:`开头的特殊名称的元标记。这些元标记定义了当前页面的 URL、标题、描述和图片，站点名称、作者和区域设置。在这里提供完整的 URL 是很重要的；仅提供路径是不够的。

我们使用了`picture_social`图片版本，其在社交网络上具有最佳尺寸：1024×512 像素。

您可以在[`developers.facebook.com/tools/debug/sharing/`](https://developers.facebook.com/tools/debug/sharing/)上验证您的 Open Graph 实现。

Twitter 卡片实现可以在[`cards-dev.twitter.com/validator`](https://cards-dev.twitter.com/validator)上进行验证。

# 另请参阅

+   *使用 CRUDL 功能创建应用*的步骤

+   *上传图片*的步骤

+   *提供 schema.org 词汇*的步骤

# 提供 schema.org 词汇

对于**搜索引擎优化**（**SEO**）来说，拥有语义标记是很重要的。但为了进一步提高搜索引擎排名，根据 schema.org 词汇提供结构化数据是很有益的。许多来自 Google、Microsoft、Pinterest、Yandex 等的应用程序使用 schema.org 结构，以创建丰富的可扩展体验，比如在搜索结果中为事件、电影、作者等创建特殊的一致外观卡片。

有几种编码，包括 RDFa、Microdata 和 JSON-LD，可以用来创建 schema.org 词汇。在这个步骤中，我们将以 JSON-LD 格式为`Idea`模型准备结构化数据，这是 Google 首选和推荐的格式。

# 准备工作

让我们将`django-json-ld`包安装到项目的虚拟环境中（并将其包含在`requirements/_base.txt`中）：

```py
(env)$ pip install django-json-ld==0.0.4
```

在设置中的`INSTALLED_APPS`下放置`"django_json_ld"`：

```py
# myproject/settings/_base.py
INSTALLED_APPS = [
    # other apps…
 "django_json_ld",
]
```

# 如何操作...

按照以下步骤完成步骤：

1.  在`Idea`模型中添加包含以下内容的`structured_data`属性：

```py
# myproject/apps/ideas/models.py
from django.db import models
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.models import ( CreationModificationDateBase, UrlBase )

class Idea(CreationModificationDateBase, UrlBase):
    # attributes, fields, properties, and methods…

 @property
    def structured_data(self):
 from django.utils.translation import get_language

 lang_code = get_language()
 data = {
 "@type": "CreativeWork",
 "name": self.translated_title,
 "description": self.translated_content,
 "inLanguage": lang_code,
 }
 if self.author:
 data["author"] = {
 "@type": "Person",
 "name": self.author.get_full_name() or 
                 self.author.username,
 }
 if self.picture:
 data["image"] = self.picture_social.url
 return data
```

1.  修改`idea_detail.html`模板：

```py
{# ideas/idea_detail.html #} {% extends "base.html" %}
{% load i18n json_ld %}

{% block meta_tags %}
    {# Open Graph and Twitter Card meta tags here… #}

    {% render_json_ld idea.structured_data %}
{% endblock %}

{% block content %}
    <a href="{% url "ideas:idea_list" %}">
     {% trans "List of ideas" %}</a>
    <h1>
        {% blocktrans trimmed with title=idea.translated_title %}
            Idea "{{ title }}"
        {% endblocktrans %}
    </h1>
    <img src="img/{{ idea.picture_large.url }}" alt="" />
    {{ idea.translated_content|linebreaks|urlize }}
    <p>
        {% for category in idea.categories.all %}
            <span class="badge badge-pill badge-info">
             {{ category.translated_title }}</span>
        {% endfor %}
    </p>
    <a href="{% url 'ideas:change_idea' pk=idea.pk %}" 
     class="btn btn-primary">{% trans "Change this idea" %}</a>
    <a href="{% url 'ideas:delete_idea' pk=idea.pk %}" 
     class="btn btn-danger">{% trans "Delete this idea" %}</a>
{% endblock %}
```

# 它是如何工作的...

`{% render_json_ld %}`模板标签将呈现类似于以下内容的脚本标签：

```py
<script type=application/ld+json>{"@type": "CreativeWork", "author": {"@type": "Person", "name": "admin"}, "description": "Lots of African countries have not enough water. Dig a water channel throughout Africa to provide water to people who have no access to it.", "image": "http://127.0.0.1:8000/media/CACHE/images/ideas/2019/09/b919eec5-c077-41f0-afb4-35f221ab550c_bOFBDgv/9caa5e61fc832f65ff6382f3d482807a.jpg", "inLanguage": "en", "name": "Dig a water channel throughout Africa"}</script>
```

`structured_data`属性返回一个嵌套字典，根据 schema.org 词汇，这些词汇被大多数流行的搜索引擎所理解。

您可以通过查看官方文档[`schema.org/docs/schemas.html`](https://schema.org/docs/schemas.html)来决定要应用于模型的词汇。

# 另请参阅

+   第二章*，模型和数据库结构*中的*创建一个模型 mixin 来处理元标签*配方

+   *使用 CRUDL 功能创建应用*配方

+   *上传图片*配方

+   *提供 Open Graph 和 Twitter Card 数据*配方

# 生成 PDF 文档

Django 视图允许您创建的不仅仅是 HTML 页面。您可以创建任何类型的文件。例如，在第四章*，模板和 JavaScript*中的*暴露设置*配方中，我们的视图提供其输出作为 JavaScript 文件而不是 HTML。您还可以创建 PDF 文档，用于发票、门票、收据、预订确认等。在这个配方中，我们将向您展示如何为数据库中的每个想法生成手册以打印。我们将使用**WeasyPrint**库将 HTML 模板制作成 PDF 文档。

# 准备工作

WeasyPrint 依赖于您需要在计算机上安装的几个库。在 macOS 上，您可以使用 Homebrew 使用此命令安装它们：

```py
$ brew install python3 cairo pango gdk-pixbuf libffi
```

然后，您可以在项目的虚拟环境中安装 WeasyPrint 本身。还要将其包含在`requirements/_base.txt`中：

```py
(env)$ pip install WeasyPrint==48
```

对于其他操作系统，请查看[`weasyprint.readthedocs.io/en/latest/install.html`](https://weasyprint.readthedocs.io/en/latest/install.html)上的安装说明。

此外，我们将使用`django-qr-code`生成链接回网站以便快速访问的**QR 码**。让我们也在虚拟环境中安装它（并将其包含在`requirements/_base.txt`中）：

```py
(env)$ pip install django-qr-code==1.0.0
```

在设置中将`"qr_code"`添加到`INSTALLED_APPS`：

```py
# myproject/settings/_base.py
INSTALLED_APPS = [    
    # Django apps…
    "qr_code",
]
```

# 如何做...

按照以下步骤完成配方：

1.  创建将生成 PDF 文档的视图：

```py
# myproject/apps/ideas/views.py
from django.shortcuts import get_object_or_404
from .models import Idea

def idea_handout_pdf(request, pk):
    from django.template.loader import render_to_string
    from django.utils.timezone import now as timezone_now
    from django.utils.text import slugify
    from django.http import HttpResponse

    from weasyprint import HTML
    from weasyprint.fonts import FontConfiguration

    idea = get_object_or_404(Idea, pk=pk)
    context = {"idea": idea}
    html = render_to_string(
        "ideas/idea_handout_pdf.html", context
    )

    response = HttpResponse(content_type="application/pdf")
    response[
        "Content-Disposition"
    ] = "inline; filename={date}-{name}-handout.pdf".format(
        date=timezone_now().strftime("%Y-%m-%d"),
        name=slugify(idea.translated_title),
    )

    font_config = FontConfiguration()
    HTML(string=html).write_pdf(
        response, font_config=font_config
    )

    return response
```

1.  将此视图插入 URL 配置：

```py
# myproject/apps/ideas/urls.py from django.urls import path

from .views import idea_handout_pdf

urlpatterns = [
    # URL configurations…
    path(
 "<uuid:pk>/handout/",
 idea_handout_pdf,
 name="idea_handout",
 ),
]
```

1.  为 PDF 文档创建模板：

```py
{# ideas/idea_handout_pdf.html #} {% extends "base_pdf.html" %}
{% load i18n qr_code %}

{% block content %}
    <h1 class="h3">{% trans "Handout" %}</h1>
    <h2 class="h1">{{ idea.translated_title }}</h2>
    <img src="img/{{ idea.picture_large.url }}" alt="" 
     class="img-responsive w-100" />
    <div class="my-3">{{ idea.translated_content|linebreaks|
     urlize }}</div>
    <p>
        {% for category in idea.categories.all %}
            <span class="badge badge-pill badge-info">
             {{ category.translated_title }}</span>
        {% endfor %}
    </p>
    <h4>{% trans "See more information online:" %}</h4>
    {% qr_from_text idea.get_url size=20 border=0 as svg_code %}
    <img alt="" src="img/>     {{ svg_code|urlencode }}" />
    <p class="mt-3 text-break">{{ idea.get_url }}</p>
{% endblock %}
```

1.  还要创建`base_pdf.html`模板：

```py
{# base_pdf.html #} <!doctype html>
{% load i18n static %}
<html lang="en">
<head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, 
     initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet"      
     href="https://stackpath.bootstrapcdn.com
      /bootstrap/4.3.1/css/bootstrap.min.css"
          integrity="sha384-
           ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY
           /iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

    <title>{% trans "Hello, World!" %}</title>

    <style>
    @page {
        size: "A4";
        margin: 2.5cm 1.5cm 3.5cm 1.5cm;
    }
    footer {
        position: fixed;
        bottom: -2.5cm;
        width: 100%;
        text-align: center;
        font-size: 10pt;
    }
    footer img {
        height: 1.5cm;
    }
    </style>

    {% block meta_tags %}{% endblock %}
</head>
<body>
    <main class="container">
        {% block content %}
        {% endblock %}
    </main>
    <footer>
        <img alt="" src="img/>         {# url-encoded SVG logo goes here #}" />
        <br />
        {% trans "Printed from MyProject" %}
    </footer>
</body>
</html>
```

# 它是如何工作的...

WeasyPrint 生成准备打印的像素完美的文档。我们可以向演示会的观众提供的手册示例看起来类似于这样：

![](img/fa351e86-8e13-4efb-8820-54e1db58e70f.png)

文档的布局是在标记和 CSS 中定义的。WeasyPrint 有自己的渲染引擎。在官方文档中阅读更多关于支持功能的信息：[`weasyprint.readthedocs.io/en/latest/features.html`](https://weasyprint.readthedocs.io/en/latest/features.html)。

您可以使用 SVG 图像，这些图像将保存为矢量图形，而不是位图，因此在打印时会更清晰。内联 SVG 尚不受支持，但您可以在那里使用带有数据源或外部 URL 的`<img>`标签。在我们的示例中，我们使用 SVG 图像作为 QR 码和页脚中的徽标。

让我们来看一下视图的代码。我们使用所选想法作为`html`字符串渲染`idea_handout_pdf.html`模板。然后，我们创建一个 PDF 内容类型的`HttpResponse`对象，文件名由当前日期和 slugified 想法标题组成。然后，我们创建 WeasyPrint 的 HTML 对象与 HTML 内容，并将其写入响应，就像我们写入文件一样。此外，我们使用`FontConfiguration`对象，它允许我们在布局中附加和使用来自 CSS 配置的网络字体。最后，我们返回响应对象。

# 另请参阅

+   *使用 CRUDL 功能创建应用*配方

+   *上传图片*配方

+   JavaScript 中的*暴露设置*配方在第四章*，模板和 JavaScript*中

# 使用 Haystack 和 Whoosh 实现多语言搜索

内容驱动网站的主要功能之一是全文搜索。Haystack 是一个模块化的搜索 API，支持 Solr、Elasticsearch、Whoosh 和 Xapian 搜索引擎。对于项目中每个需要在搜索中找到的模型，您需要定义一个索引，该索引将从模型中读取文本信息并将其放入后端。在本食谱中，您将学习如何为多语言网站使用 Haystack 和基于 Python 的 Whoosh 搜索引擎设置搜索。

# 准备工作

我们将使用先前定义的`categories`和`ideas`应用程序。

确保在您的虚拟环境中安装了`django-haystack`和`Whoosh`（并将它们包含在`requirements/_base.txt`中）：

```py
(env)$ pip install django-haystack==2.8.1
(env)$ pip install Whoosh==2.7.4
```

# 如何操作...

让我们通过执行以下步骤来设置 Haystack 和 Whoosh 的多语言搜索：

1.  创建一个包含`MultilingualWhooshEngine`和我们想法的搜索索引的`search`应用程序。搜索引擎将位于`multilingual_whoosh_backend.py`文件中：

```py
# myproject/apps/search/multilingual_whoosh_backend.py from django.conf import settings
from django.utils import translation
from haystack.backends.whoosh_backend import (
    WhooshSearchBackend,
    WhooshSearchQuery,
    WhooshEngine,
)
from haystack import connections
from haystack.constants import DEFAULT_ALIAS

class MultilingualWhooshSearchBackend(WhooshSearchBackend):
    def update(self, index, iterable, commit=True, 
     language_specific=False):
        if not language_specific and self.connection_alias == 
         "default":
            current_language = (translation.get_language() or 
             settings.LANGUAGE_CODE)[
                :2
            ]
            for lang_code, lang_name in settings.LANGUAGES:
                lang_code_underscored = lang_code.replace("-", "_")
                using = f"default_{lang_code_underscored}"
                translation.activate(lang_code)
                backend = connections[using].get_backend()
                backend.update(index, iterable, commit, 
                 language_specific=True)
            translation.activate(current_language)
        elif language_specific:
            super().update(index, iterable, commit)

class MultilingualWhooshSearchQuery(WhooshSearchQuery):
    def __init__(self, using=DEFAULT_ALIAS):
        lang_code_underscored =   
        translation.get_language().replace("-", "_")
        using = f"default_{lang_code_underscored}"
        super().__init__(using=using)

class MultilingualWhooshEngine(WhooshEngine):
    backend = MultilingualWhooshSearchBackend
    query = MultilingualWhooshSearchQuery
```

1.  让我们创建搜索索引，如下所示：

```py
# myproject/apps/search/search_indexes.py from haystack import indexes

from myproject.apps.ideas.models import Idea

class IdeaIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True)

    def get_model(self):
        return Idea

    def index_queryset(self, using=None):
        """
        Used when the entire index for model is updated.
        """
        return self.get_model().objects.all()

    def prepare_text(self, idea):
        """
        Called for each language / backend
        """
        fields = [
            idea.translated_title, idea.translated_content
        ]
        fields += [
            category.translated_title 
            for category in idea.categories.all()
        ]
        return "\n".join(fields)
```

1.  配置设置以使用`MultilingualWhooshEngine`：

```py
# myproject/settings/_base.py import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

#…

INSTALLED_APPS = [
    # contributed
    # …
    # third-party
    # …
    "haystack",
    # local
    "myproject.apps.core",
    "myproject.apps.categories",
    "myproject.apps.ideas",
    "myproject.apps.search",
]

LANGUAGE_CODE = "en"

# All official languages of European Union
LANGUAGES = [
    ("bg", "Bulgarian"),
    ("hr", "Croatian"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("nl", "Dutch"),
    ("en", "English"),
    ("et", "Estonian"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("de", "German"),
    ("el", "Greek"),
    ("hu", "Hungarian"),
    ("ga", "Irish"),
    ("it", "Italian"),
    ("lv", "Latvian"),
    ("lt", "Lithuanian"),
    ("mt", "Maltese"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("sk", "Slovak"),
    ("sl", "Slovene"),
    ("es", "Spanish"),
    ("sv", "Swedish"),
]

HAYSTACK_CONNECTIONS = {}
for lang_code, lang_name in LANGUAGES:
 lang_code_underscored = lang_code.replace("-", "_")
 HAYSTACK_CONNECTIONS[f"default_{lang_code_underscored}"] = {
 "ENGINE":   
 "myproject.apps.search.multilingual_whoosh_backend
  .MultilingualWhooshEngine",
 "PATH": os.path.join(BASE_DIR, "tmp", 
  f"whoosh_index_{lang_code_underscored}"),
 }
 lang_code_underscored = LANGUAGE_CODE.replace("-", "_")
 HAYSTACK_CONNECTIONS["default"] = HAYSTACK_CONNECTIONS[
 f"default_{lang_code_underscored}"
]
```

1.  添加 URL 规则的路径：

```py
# myproject/urls.py from django.contrib import admin
from django.conf.urls.i18n import i18n_patterns
from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect

urlpatterns = i18n_patterns(
    path("", lambda request: redirect("ideas:idea_list")),
    path("admin/", admin.site.urls),
    path("accounts/", include("django.contrib.auth.urls")),
    path("ideas/", include(("myproject.apps.ideas.urls", "ideas"), 
    namespace="ideas")),
    path("search/", include("haystack.urls")),
)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static("/media/", document_root=settings.MEDIA_ROOT)
```

1.  我们需要一个搜索表单和搜索结果的模板，如下所示：

```py
{# search/search.html #}
{% extends "base.html" %}
{% load i18n %}

{% block sidebar %}
    <form method="get" action="{{ request.path }}">
        <div class="well clearfix">
            {{ form.as_p }}
            <p class="pull-right">
                <button type="submit" class="btn btn-primary">
                 {% trans "Search" %}</button>
            </p>
        </div>
    </form>
{% endblock %}

{% block main %}
    {% if query %}
        <h1>{% trans "Search Results" %}</h1>

        {% for result in page.object_list %}
            {% with idea=result.object %}
                <a href="{{ idea.get_url_path }}" 
                 class="d-block my-3">
                    <div class="card">
                      <img src="img/{{ idea.picture_thumbnail.url }}" 
                       alt="" />
                      <div class="card-body">
                        <p class="card-text">
                         {{ idea.translated_title }}</p>
                      </div>
                    </div>
                </a>
            {% endwith %}
        {% empty %}
            <p>{% trans "No results found." %}</p>
        {% endfor %}

        {% include "misc/includes/pagination.html" with 
         object_list=page %}
    {% endif %}
{% endblock %}
```

1.  在`misc/includes/pagination.html`中添加一个分页模板，就像在*管理分页列表*食谱中一样。

1.  调用`rebuild_index`管理命令来对数据库数据进行索引并准备全文搜索的使用：

```py
(env)$ python manage.py rebuild_index --noinput
```

# 工作原理...

`MultilingualWhooshEngine`指定了两个自定义属性：

+   `backend`指向`MultilingualWhooshSearchBackend`，它确保项目将为`LANGUAGES`设置中给定的每种语言进行索引，并将其放在`HAYSTACK_CONNECTIONS`中定义的相关 Haystack 索引位置下。

+   `query`引用了`MultilingualWhooshSearchQuery`，其责任是确保在搜索关键字时，将使用特定于当前语言的 Haystack 连接。

每个索引都有一个`text`字段，用于存储模型特定语言的全文。索引的模型由`get_model()`方法确定，`index_queryset()`方法定义要索引的 QuerySet，`prepare_text()`方法中定义要在其中搜索的内容为换行分隔的字符串。

对于模板，我们已经使用了 Bootstrap 4 的一些元素，使用了表单的开箱即用的渲染功能。可以使用类似本章前面解释的*使用 django-crispy-forms 创建表单布局*的方法来增强这一点。

最终的搜索页面将在侧边栏中有一个表单，在主列中有搜索结果，并且看起来类似于以下内容：

![](img/675a5960-e419-4368-977e-e4cc65304211.png)

定期更新搜索索引的最简单方法是调用`rebuild_index`管理命令，也许可以通过每晚的 cron 作业来实现。要了解更多信息，请查看第十三章*维护*中的*设置定期任务的 cron 作业*食谱。

# 另请参阅

+   *使用 django-crispy-forms 创建表单布局*食谱

+   *管理分页列表*食谱

+   第十三章*维护*中的*设置定期任务的 cron 作业*食谱

# 使用 Elasticsearch DSL 实现多语言搜索

Haystack 与 Whoosh 是一个良好的稳定搜索机制，只需要一些 Python 模块，但为了获得更好的性能，我们建议使用 Elasticsearch。在本食谱中，我们将向您展示如何为多语言搜索使用它。

# 准备工作

首先，让我们安装 Elasticsearch 服务器。在 macOS 上，您可以使用 Homebrew 来完成：

```py
$ brew install elasticsearch
```

在撰写本文时，Homebrew 上的最新稳定版本的 Elasticsearch 是 6.8.2。

在您的虚拟环境中安装`django-elasticsearch-dsl`（并将其包含在`requirements/_base.txt`中）：

```py
(env)$ pip install django-elasticsearch-dsl==6.4.1
```

请注意，安装匹配的`django-elasticsearch-dsl`版本非常重要。否则，当尝试连接到 Elasticsearch 服务器或构建索引时，将会出现错误。您可以在[`github.com/sabricot/django-elasticsearch-dsl`](https://github.com/sabricot/django-elasticsearch-dsl)上查看版本兼容性表。

# 如何做...

让我们通过执行以下步骤设置多语言搜索与 Elasticsearch DSL：

1.  修改设置文件，并将`"django_elasticsearch_dsl"`添加到`INSTALLED_APPS`，并将`ELASTICSEARCH_DSL`设置如下：

```py
# myproject/settings/_base.py 
INSTALLED_APPS = [
    # other apps…
    "django_elasticsearch_dsl",
]

ELASTICSEARCH_DSL={
 'default': {
 'hosts': 'localhost:9200'
    },
}
```

1.  在`ideas`应用程序中，创建一个`documents.py`文件，其中包含`IdeaDocument`用于 idea 搜索索引，如下所示：

```py
# myproject/apps/ideas/documents.py
from django.conf import settings
from django.utils.translation import get_language, activate
from django.db import models

from django_elasticsearch_dsl import fields
from django_elasticsearch_dsl.documents import (
    Document,
    model_field_class_to_field_class,
)
from django_elasticsearch_dsl.registries import registry

from myproject.apps.categories.models import Category
from .models import Idea

def _get_url_path(instance, language):
    current_language = get_language()
    activate(language)
    url_path = instance.get_url_path()
    activate(current_language)
    return url_path

@registry.register_document
class IdeaDocument(Document):
    author = fields.NestedField(
        properties={
            "first_name": fields.StringField(),
            "last_name": fields.StringField(),
            "username": fields.StringField(),
            "pk": fields.IntegerField(),
        },
        include_in_root=True,
    )
    title_bg = fields.StringField()
    title_hr = fields.StringField()
    # other title_* fields for each language in the LANGUAGES 
      setting…
    content_bg = fields.StringField()
    content_hr = fields.StringField()
    # other content_* fields for each language in the LANGUAGES 
      setting…

    picture_thumbnail_url = fields.StringField()

    categories = fields.NestedField(
        properties=dict(
            pk=fields.IntegerField(),
            title_bg=fields.StringField(),
            title_hr=fields.StringField(),
            # other title_* definitions for each language in the 
              LANGUAGES setting…
        ),
        include_in_root=True,
    )

    url_path_bg = fields.StringField()
    url_path_hr = fields.StringField()
    # other url_path_* fields for each language in the LANGUAGES 
      setting…

    class Index:
        name = "ideas"
        settings = {"number_of_shards": 1, "number_of_replicas": 0}

    class Django:
        model = Idea
        # The fields of the model you want to be indexed in 
          Elasticsearch
        fields = ["uuid", "rating"]
        related_models = [Category]

    def get_instances_from_related(self, related_instance):
        if isinstance(related_instance, Category):
            category = related_instance
            return category.category_ideas.all()
```

1.  向`IdeaDocument`添加`prepare_*`方法以准备索引的数据：

```py
    def prepare(self, instance):
        lang_code_underscored = settings.LANGUAGE_CODE.replace
         ("-", "_")
        setattr(instance, f"title_{lang_code_underscored}", 
         instance.title)
        setattr(instance, f"content_{lang_code_underscored}", 
         instance.content)
        setattr(
            instance,
            f"url_path_{lang_code_underscored}",
            _get_url_path(instance=instance, 
              language=settings.LANGUAGE_CODE),
        )
        for lang_code, lang_name in 
         settings.LANGUAGES_EXCEPT_THE_DEFAULT:
            lang_code_underscored = lang_code.replace("-", "_")
            setattr(instance, f"title_{lang_code_underscored}", 
             "")
            setattr(instance, f"content_{lang_code_underscored}", 
             "")
            translations = instance.translations.filter(language=
             lang_code).first()
            if translations:
                setattr(instance, f"title_{lang_code_underscored}", 
                 translations.title)
                setattr(
                    instance, f"content_{lang_code_underscored}", 
                     translations.content
                )
            setattr(
                instance,
                f"url_path_{lang_code_underscored}",
                _get_url_path(instance=instance, 
                  language=lang_code),
            )
        data = super().prepare(instance=instance)
        return data

    def prepare_picture_thumbnail_url(self, instance):
        if not instance.picture:
            return ""
        return instance.picture_thumbnail.url

    def prepare_author(self, instance):
        author = instance.author
        if not author:
            return []
        author_dict = {
            "pk": author.pk,
            "first_name": author.first_name,
            "last_name": author.last_name,
            "username": author.username,
        }
        return [author_dict]

    def prepare_categories(self, instance):
        categories = []
        for category in instance.categories.all():
            category_dict = {"pk": category.pk}
            lang_code_underscored = 
             settings.LANGUAGE_CODE.replace("-", "_")
            category_dict[f"title_{lang_code_underscored}"] = 
             category.title
            for lang_code, lang_name in 
             settings.LANGUAGES_EXCEPT_THE_DEFAULT:
                lang_code_underscored = lang_code.replace("-", "_")
                category_dict[f"title_{lang_code_underscored}"] = 
                 ""
                translations = 
                 category.translations.filter(language=
                  lang_code).first()
                if translations:
                    category_dict[f"title_{lang_code_underscored}"] 
                   = translations.title
            categories.append(category_dict)
        return categories
```

1.  向`IdeaDocument`添加一些属性和方法，以从索引文档中返回翻译内容：

```py
    @property
    def translated_title(self):
        lang_code_underscored = get_language().replace("-", "_")
        return getattr(self, f"title_{lang_code_underscored}", "")

    @property
    def translated_content(self):
        lang_code_underscored = get_language().replace("-", "_")
        return getattr(self, f"content_{lang_code_underscored}", 
         "")

    def get_url_path(self):
        lang_code_underscored = get_language().replace("-", "_")
        return getattr(self, f"url_path_{lang_code_underscored}", 
         "")

    def get_categories(self):
        lang_code_underscored = get_language().replace("-", "_")
        return [
            dict(
                translated_title=category_dict[f"title_{lang_
                 code_underscored}"],
                **category_dict,
            )
            for category_dict in self.categories
        ] 
```

1.  在`documents.py`文件中还有一件事要做，那就是对`UUIDField`映射进行修补，因为默认情况下，Django Elasticsearch DSL 尚不支持它。为此，请在导入部分之后插入此行：

```py
model_field_class_to_field_class[models.UUIDField] = fields.TextField
```

1.  在`ideas`应用程序的`forms.py`中创建`IdeaSearchForm`：

```py
# myproject/apps/ideas/forms.py from django import forms
from django.utils.translation import ugettext_lazy as _

from crispy_forms import helper, layout

class IdeaSearchForm(forms.Form):
    q = forms.CharField(label=_("Search for"), required=False)

    def __init__(self, request, *args, **kwargs):
        self.request = request
        super().__init__(*args, **kwargs)

        self.helper = helper.FormHelper()
        self.helper.form_action = self.request.path
        self.helper.form_method = "GET"
        self.helper.layout = layout.Layout(
            layout.Field("q", css_class="input-block-level"),
            layout.Submit("search", _("Search")),
        )
```

1.  添加用于使用 Elasticsearch 搜索的视图：

```py
# myproject/apps/ideas/views.py from django.shortcuts import render
from django.conf import settings
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.utils.functional import LazyObject

from .forms import IdeaSearchForm

PAGE_SIZE = getattr(settings, "PAGE_SIZE", 24)

class SearchResults(LazyObject):
    def __init__(self, search_object):
        self._wrapped = search_object

    def __len__(self):
        return self._wrapped.count()

    def __getitem__(self, index):
        search_results = self._wrapped[index]
        if isinstance(index, slice):
            search_results = list(search_results)
        return search_results

def search_with_elasticsearch(request):
    from .documents import IdeaDocument
    from elasticsearch_dsl.query import Q

    form = IdeaSearchForm(request, data=request.GET)

    search = IdeaDocument.search()

    if form.is_valid():
        value = form.cleaned_data["q"]
        lang_code_underscored = request.LANGUAGE_CODE.replace("-", 
          "_")
        search = search.query(
            Q("match_phrase", **{f"title_{
             lang_code_underscored}": 
             value})
            | Q("match_phrase", **{f"content_{
               lang_code_underscored}": value})
            | Q(
                "nested",
                path="categories",
                query=Q(
                    "match_phrase",
                    **{f"categories__title_{
                     lang_code_underscored}": value},
                ),
            )
        )
    search_results = SearchResults(search)

    paginator = Paginator(search_results, PAGE_SIZE)
    page_number = request.GET.get("page")
    try:
        page = paginator.page(page_number)
    except PageNotAnInteger:
        # If page is not an integer, show first page.
        page = paginator.page(1)
    except EmptyPage:
        # If page is out of range, show last existing page.
        page = paginator.page(paginator.num_pages)

    context = {"form": form, "object_list": page}
    return render(request, "ideas/idea_search.html", context)
```

1.  创建一个`idea_search.html`模板，用于搜索表单和搜索结果：

```py
{# ideas/idea_search.html #}
{% extends "base.html" %}
{% load i18n crispy_forms_tags %}

{% block sidebar %}
    {% crispy form %}
{% endblock %}

{% block main %}
    <h1>{% trans "Search Results" %}</h1>
    {% if object_list %}
        {% for idea in object_list %}
            <a href="{{ idea.get_url_path }}" class="d-block my-3">
                <div class="card">
                  <img src="img/{{ idea.picture_thumbnail_url }}" 
                    alt="" />
                  <div class="card-body">
                    <p class="card-text">{{ idea.translated_title 
                      }}</p>
                  </div>
                </div>
            </a>
        {% endfor %}
        {% include "misc/includes/pagination.html" %}
    {% else %}
        <p>{% trans "No ideas found." %}</p>
    {% endif %}
{% endblock %}
```

1.  在`misc/includes/pagination.html`中添加一个分页模板，就像*管理分页列表*配方中一样。

1.  调用`search_index --rebuild`管理命令来索引数据库数据并准备使用全文搜索：

```py
(env)$ python manage.py search_index --rebuild
```

# 它是如何工作的...

Django Elasticsearch DSL 文档类似于模型表单。在那里，您定义要保存到索引的模型字段，以便稍后用于搜索查询。在我们的`IdeaDocument`示例中，我们保存 UUID、评分、作者、类别、标题、内容和 URL 路径以及所有语言和图片缩略图 URL。`Index`类定义了此文档的 Elasticsearch 索引的设置。`Django`类定义了从哪里填充索引字段。有一个`related_models`设置，告诉在哪个模型更改后也更新此索引。在我们的情况下，它是一个`Category`模型。请注意，使用`django-elasticsearch-dsl`，只要保存模型，索引就会自动更新。这是使用信号完成的。

`get_instances_from_related()`方法告诉如何在更改`Category`实例时检索`Idea`模型实例。

`IdeaDocument`的`prepare()`和`prepare_*()`方法告诉从哪里获取数据以及如何保存特定字段的数据。例如，我们从`IdeaTranslations`模型的`title`字段中读取`title_lt`的数据，其中`language`字段等于`"lt"`。

`IdeaDocument`的最后属性和方法用于从当前活动语言的索引中检索信息。

然后，我们有一个带有搜索表单的视图。表单中有一个名为`q`的查询字段。当提交时，我们在当前语言的标题、内容或类别标题字段中搜索查询的单词。然后，我们用惰性评估的`SearchResults`类包装搜索结果，以便我们可以将其与默认的 Django 分页器一起使用。

视图的模板将在侧边栏中包含搜索表单，在主列中包含搜索结果，并且看起来会像这样：

![](img/e41b1ce7-bc57-4f78-9cb0-74cb7b297949.png)

# 另请参阅

+   *创建具有 CRUDL 功能的应用程序*配方

+   *使用 Haystack 和 Whoosh 实现多语言搜索*配方

+   *使用 django-crispy-forms 创建表单布局*配方

+   *管理分页列表*配方
