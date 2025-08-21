# 第二章：模型和数据库结构

在本章中，我们将涵盖以下主题：

+   使用模型 mixin

+   创建一个具有 URL 相关方法的模型 mixin

+   创建一个模型 mixin 来处理创建和修改日期

+   创建一个处理元标签的模型 mixin

+   创建一个处理通用关系的模型 mixin

+   处理多语言字段

+   使用模型翻译表

+   避免循环依赖

+   添加数据库约束

+   使用迁移

+   将外键更改为多对多字段

# 介绍

当您开始一个新的应用程序时，您要做的第一件事是创建代表您的数据库结构的模型。我们假设您已经创建了 Django 应用程序，或者至少已经阅读并理解了官方的 Django 教程。在本章中，您将看到一些有趣的技术，这些技术将使您的数据库结构在项目中的不同应用程序中保持一致。然后，您将学习如何处理数据库中数据的国际化。之后，您将学习如何避免模型中的循环依赖以及如何设置数据库约束。在本章的最后，您将学习如何使用迁移来在开发过程中更改数据库结构。

# 技术要求

要使用本书中的代码，您需要最新稳定版本的 Python、MySQL 或 PostgreSQL 数据库以及一个带有虚拟环境的 Django 项目。

您可以在 GitHub 存储库的`ch02`目录中找到本章的所有代码：[`github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition`](https://github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition)。

# 使用模型 mixin

在面向对象的语言中，比如 Python，一个 mixin 类可以被视为一个带有实现特性的接口。当一个模型扩展一个 mixin 时，它实现了接口并包含了所有的字段、属性、属性和方法。Django 模型中的 mixin 可以在您想要多次在不同模型中重用通用功能时使用。Django 中的模型 mixin 是抽象基本模型类。我们将在接下来的几个示例中探讨它们。

# 准备工作

首先，您需要创建可重用的 mixin。将模型 mixin 保存在`myproject.apps.core`应用程序中是一个很好的地方。如果您创建了一个可重用的应用程序，将模型 mixin 保存在可重用的应用程序本身中，可能是在一个`base.py`文件中。

# 如何做...

打开任何您想要在其中使用 mixin 的 Django 应用程序的`models.py`文件，并键入以下代码：

```py
# myproject/apps/ideas/models.py
from django.db import models
from django.urls import reverse
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.models import (
    CreationModificationDateBase,
    MetaTagsBase,
    UrlBase,
)

class Idea(CreationModificationDateBase, MetaTagsBase, UrlBase):
    title = models.CharField(
        _("Title"),
        max_length=200,
    )
    content = models.TextField(
        _("Content"),
    )
    # other fields…

    class Meta:
        verbose_name = _("Idea")
        verbose_name_plural = _("Ideas")

    def __str__(self):
        return self.title

    def get_url_path(self):
        return reverse("idea_details", kwargs={
            "idea_id": str(self.pk),
        })
```

# 它是如何工作的...

Django 的模型继承支持三种类型的继承：抽象基类、多表继承和代理模型。模型 mixin 是抽象模型类，我们通过使用一个指定字段、属性和方法的抽象`Meta`类来定义它们。当您创建一个模型，比如在前面的示例中所示的`Idea`，它继承了`CreationModificationDateMixin`、`MetaTagsMixin`和`UrlMixin`的所有特性。这些抽象类的所有字段都保存在与扩展模型的字段相同的数据库表中。在接下来的示例中，您将学习如何定义您自己的模型 mixin。

# 还有更多...

在普通的 Python 类继承中，如果有多个基类，并且它们都实现了一个特定的方法，并且您在子类的实例上调用该方法，只有第一个父类的方法会被调用，就像下面的例子一样：

```py
>>> class A(object):
... def test(self):
...     print("A.test() called")
... 

>>> class B(object):
... def test(self):
...     print("B.test() called")
... 

>>> class C(object):
... def test(self):
...     print("C.test() called")
... 

>>> class D(A, B, C):
... def test(self):
...     super().test()
...     print("D.test() called")

>>> d = D()
>>> d.test()
A.test() called
D.test() called
```

这与 Django 模型基类相同；然而，有一个特殊的例外。

Django 框架对元类进行了一些魔术，调用了每个基类的`save()`和`delete()`方法。

这意味着您可以自信地对特定字段进行预保存、后保存、预删除和后删除操作，这些字段是通过覆盖 mixin 中的`save()`和`delete()`方法来定义的。

要了解更多关于不同类型的模型继承，请参阅官方 Django 文档，网址为[`docs.djangoproject.com/en/2.2/topics/db/models/#model-inheritance`](https://docs.djangoproject.com/en/2.2/topics/db/models/#model-inheritance)。

# 另请参阅

+   *创建一个具有与 URL 相关方法的模型 mixin*配方

+   *创建一个模型 mixin 来处理创建和修改日期*配方

+   *创建一个模型 mixin 来处理 meta 标签*配方

# 创建一个具有与 URL 相关方法的模型 mixin

对于每个具有自己独特详细页面的模型，定义`get_absolute_url()`方法是一个良好的做法。这个方法可以在模板中使用，也可以在 Django 管理站点中用于预览保存的对象。但是，`get_absolute_url()`是模棱两可的，因为它返回 URL 路径而不是完整的 URL。

在这个配方中，我们将看看如何创建一个模型 mixin，为模型特定的 URL 提供简化的支持。这个 mixin 将使您能够做到以下几点：

+   允许您在模型中定义 URL 路径或完整 URL

+   根据您定义的路径自动生成其他 URL

+   在幕后定义`get_absolute_url()`方法

# 准备工作

如果尚未这样做，请创建`myproject.apps.core`应用程序，您将在其中存储您的模型 mixin。然后，在 core 包中创建一个`models.py`文件。或者，如果您创建了一个可重用的应用程序，请将 mixin 放在该应用程序的`base.py`文件中。

# 如何做...

逐步执行以下步骤：

1.  将以下内容添加到`core`应用程序的`models.py`文件中：

```py
# myproject/apps/core/models.py from urllib.parse import urlparse, urlunparse
from django.conf import settings
from django.db import models

class UrlBase(models.Model):
    """
    A replacement for get_absolute_url()
    Models extending this mixin should have either get_url or 
     get_url_path implemented.
    """
    class Meta:
        abstract = True

    def get_url(self):
        if hasattr(self.get_url_path, "dont_recurse"):
            raise NotImplementedError
        try:
            path = self.get_url_path()
        except NotImplementedError:
            raise
        return settings.WEBSITE_URL + path
    get_url.dont_recurse = True

    def get_url_path(self):
        if hasattr(self.get_url, "dont_recurse"):
            raise NotImplementedError
        try:
            url = self.get_url()
        except NotImplementedError:
            raise
        bits = urlparse(url)
        return urlunparse(("", "") + bits[2:])
    get_url_path.dont_recurse = True

    def get_absolute_url(self):
        return self.get_url()
```

1.  将`WEBSITE_URL`设置添加到`dev`、`test`、`staging`和`production`设置中，不带斜杠。例如，对于开发环境，如下所示：

```py
# myproject/settings/dev.py
from ._base import *

DEBUG = True
WEBSITE_URL = "http://127.0.0.1:8000"  # without trailing slash
```

1.  要在您的应用程序中使用 mixin，从`core`应用程序导入 mixin，在您的模型类中继承 mixin，并定义`get_url_path()`方法，如下所示：

```py
# myproject/apps/ideas/models.py
from django.db import models
from django.urls import reverse
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.models import UrlBase

class Idea(UrlBase):
    # fields, attributes, properties and methods…

    def get_url_path(self):
        return reverse("idea_details", kwargs={
            "idea_id": str(self.pk),
        })
```

# 它是如何工作的...

`UrlBase`类是一个抽象模型，具有三种方法，如下所示：

+   `get_url()`检索对象的完整 URL。

+   `get_url_path()`检索对象的绝对路径。

+   `get_absolute_url()`模仿`get_url_path()`方法。

`get_url()`和`get_url_path()`方法预计会在扩展模型类中被覆盖，例如`Idea`。您可以定义`get_url()`，`get_url_path()`将会将其剥离为路径。或者，您可以定义`get_url_path()`，`get_url()`将在路径的开头添加网站 URL。

一个经验法则是始终覆盖`get_url_path()`方法。

在模板中，当您需要链接到同一网站上的对象时，请使用`get_url_path()`，如下所示：

```py
<a href="{{ idea.get_url_path }}">{{ idea.title }}</a>
```

在外部通信中使用`get_url()`进行链接，例如在电子邮件、RSS 订阅或 API 中；例如如下：

```py
<a href="{{  idea.get_url }}">{{ idea.title }}</a>
```

默认的`get_absolute_url()`方法将在 Django 模型管理中用于“查看网站”功能，并且也可能被一些第三方 Django 应用程序使用。

# 还有更多...

一般来说，不要在 URL 中使用递增的主键，因为将它们暴露给最终用户是不安全的：项目的总数将可见，并且只需更改 URL 路径就可以轻松浏览不同的项目。

只有当它们是**通用唯一标识符**（**UUIDs**）或生成的随机字符串时，您才可以在详细页面的 URL 中使用主键。否则，请创建并使用 slug 字段，如下所示：

```py
class Idea(UrlBase):
    slug = models.SlugField(_("Slug for URLs"), max_length=50)
```

# 另请参阅

+   *使用模型 mixin*配方

+   *创建一个模型 mixin 来处理创建和修改日期*配方

+   *创建一个模型 mixin 来处理 meta 标签*配方

+   *创建一个模型 mixin 来处理通用关系*配方

+   *为开发、测试、暂存和生产环境配置设置*配方，在第一章*，使用 Django 3.0 入门*

# 创建一个模型 mixin 来处理创建和修改日期

在您的模型中包含创建和修改模型实例的时间戳是很常见的。在这个示例中，您将学习如何创建一个简单的模型 mixin，为您的模型保存创建和修改的日期和时间。使用这样的 mixin 将确保所有模型使用相同的时间戳字段名称，并具有相同的行为。

# 准备工作

如果还没有这样做，请创建`myproject.apps.core`包来保存您的 mixin。然后，在核心包中创建`models.py`文件。

# 如何做...

打开`myprojects.apps.core`包中的`models.py`文件，并在其中插入以下内容：

```py
# myproject/apps/core/models.py
from django.db import models
from django.utils.translation import gettext_lazy as _

class CreationModificationDateBase(models.Model):
    """
    Abstract base class with a creation and modification date and time
    """

    created = models.DateTimeField(
        _("Creation Date and Time"),
        auto_now_add=True,
    )

    modified = models.DateTimeField(
        _("Modification Date and Time"),
        auto_now=True,
    )

    class Meta:
        abstract = True
```

# 它是如何工作的...

`CreationModificationDateMixin`类是一个抽象模型，这意味着扩展模型类将在同一个数据库表中创建所有字段，也就是说，不会有使表更复杂的一对一关系。

这个 mixin 有两个日期时间字段，`created`和`modified`。使用`auto_now_add`和`auto_now`属性，时间戳将在保存模型实例时自动保存。字段将自动获得`editable=False`属性，因此在管理表单中将被隐藏。如果在设置中将`USE_TZ`设置为`True`（这是默认和推荐的），将使用时区感知的时间戳。否则，将使用时区无关的时间戳。时区感知的时间戳保存在数据库中的**协调世界时**（**UTC**）时区，并在读取或写入时将其转换为项目的默认时区。时区无关的时间戳保存在数据库中项目的本地时区；一般来说，它们不实用，因为它们使得时区之间的时间管理更加复杂。

要使用这个 mixin，我们只需要导入它并扩展我们的模型，如下所示：

```py
# myproject/apps/ideas/models.py
from django.db import models

from myproject.apps.core.models import CreationModificationDateBase

class Idea(CreationModificationDateBase):
    # other fields, attributes, properties, and methods…
```

# 另请参阅

+   *使用模型 mixin*示例

+   *创建一个处理 meta 标签的模型 mixin*示例

+   *创建一个处理通用关系的模型 mixin*示例

# 创建一个处理 meta 标签的模型 mixin

当您为搜索引擎优化您的网站时，不仅需要为每个页面使用语义标记，还需要包含适当的 meta 标签。为了最大的灵活性，有必要定义特定于在您的网站上拥有自己详细页面的对象的常见 meta 标签的内容。在这个示例中，我们将看看如何为与关键字、描述、作者和版权 meta 标签相关的字段和方法创建模型 mixin。

# 准备工作

如前面的示例中所述，确保您的 mixin 中有`myproject.apps.core`包。另外，在该包下创建一个目录结构`templates/utils/includes/`，并在其中创建一个`meta.html`文件来存储基本的 meta 标签标记。

# 如何做...

让我们创建我们的模型 mixin：

1.  确保在设置中将`"myproject.apps.core"`添加到`INSTALLED_APPS`中，因为我们希望为此模块考虑`templates`目录。

1.  将以下基本的 meta 标签标记添加到`meta_field.html`中：

```py
{# templates/core/includes/meta_field.html #}
<meta name="{{ name }}" content="{{ content }}" />
```

1.  打开您喜欢的编辑器中的核心包中的`models.py`文件，并添加以下内容：

```py
# myproject/apps/core/models.py from django.conf import settings
from django.db import models
from django.utils.translation import gettext_lazy as _
from django.utils.safestring import mark_safe
from django.template.loader import render_to_string

class MetaTagsBase(models.Model):
    """
    Abstract base class for generating meta tags
    """
    meta_keywords = models.CharField(
        _("Keywords"),
        max_length=255,
        blank=True,
        help_text=_("Separate keywords with commas."),
    )
    meta_description = models.CharField(
        _("Description"),
        max_length=255,
        blank=True,
    )
    meta_author = models.CharField(
        _("Author"),
        max_length=255,
        blank=True,
    )
    meta_copyright = models.CharField(
        _("Copyright"),
        max_length=255,
        blank=True,
    )

    class Meta:
        abstract = True

    def get_meta_field(self, name, content):
        tag = ""
        if name and content:
            tag = render_to_string("core/includes/meta_field.html", 
            {
                "name": name,
                "content": content,
            })
        return mark_safe(tag)

    def get_meta_keywords(self):
        return self.get_meta_field("keywords", self.meta_keywords)

    def get_meta_description(self):
        return self.get_meta_field("description", 
         self.meta_description)

    def get_meta_author(self):
        return self.get_meta_field("author", self.meta_author)

    def get_meta_copyright(self):
        return self.get_meta_field("copyright", 
         self.meta_copyright)

    def get_meta_tags(self):
        return mark_safe("\n".join((
            self.get_meta_keywords(),
            self.get_meta_description(),
            self.get_meta_author(),
            self.get_meta_copyright(),
        )))
```

# 它是如何工作...

这个 mixin 为扩展自它的模型添加了四个字段：`meta_keywords`，`meta_description`，`meta_author`和`meta_copyright`。还添加了相应的`get_*()`方法，用于呈现相关的 meta 标签。其中每个方法都将名称和适当的字段内容传递给核心的`get_meta_field()`方法，该方法使用此输入返回基于`meta_field.html`模板的呈现标记。最后，提供了一个快捷的`get_meta_tags()`方法，用于一次生成所有可用元数据的组合标记。

如果您在模型中使用这个 mixin，比如在本章开头的*使用模型 mixin*配方中展示的`Idea`中，您可以将以下内容放在`detail`页面模板的`HEAD`部分，以一次性渲染所有的元标记：

```py
{% block meta_tags %}
{{ block.super }}
{{ idea.get_meta_tags }}
{% endblock %}
```

在这里，一个`meta_tags`块已经在父模板中定义，这个片段展示了子模板如何重新定义块，首先将父模板的内容作为`block.super`，然后用`idea`对象的附加标签扩展它。您也可以通过类似以下的方式只渲染特定的元标记：`{{ idea.get_meta_description }}`。

从`models.py`代码中，您可能已经注意到，渲染的元标记被标记为安全-也就是说，它们没有被转义，我们不需要使用`safe`模板过滤器。只有来自数据库的值被转义，以确保最终的 HTML 格式正确。当我们为`meta_field.html`模板调用`render_to_string()`时，`meta_keywords`和其他字段中的数据库数据将自动转义，因为该模板在其内容中没有指定`{% autoescape off %}`。

# 另请参阅

+   *使用模型 mixin*配方

+   *创建一个处理创建和修改日期的模型 mixin*配方

+   *创建处理通用关系的模型 mixin*配方

+   在第四章*，模板和 JavaScript*中*安排 base.html 模板*配方

# 创建一个处理通用关系的模型 mixin

除了常规的数据库关系，比如外键关系或多对多关系，Django 还有一种将模型与任何其他模型的实例相关联的机制。这个概念被称为通用关系。对于每个通用关系，我们保存相关模型的内容类型以及该模型实例的 ID。

在这个配方中，我们将看看如何在模型 mixin 中抽象通用关系的创建。

# 准备工作

为了使这个配方工作，您需要安装`contenttypes`应用程序。它应该默认在设置中的`INSTALLED_APPS`列表中，如下所示：

```py
# myproject/settings/_base.py

INSTALLED_APPS = [
    # contributed
    "django.contrib.admin",
    "django.contrib.auth",
 "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # third-party
    # ...
    # local
    "myproject.apps.core",
    "myproject.apps.categories",
    "myproject.apps.ideas",
]
```

再次确保您已经为模型 mixin 创建了`myproject.apps.core`应用程序。

# 如何做...

要创建和使用通用关系的 mixin，请按照以下步骤进行：

1.  在文本编辑器中打开核心包中的`models.py`文件，并在那里插入以下内容：

```py
# myproject/apps/core/models.py from django.db import models
from django.utils.translation import gettext_lazy as _
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import GenericForeignKey
from django.core.exceptions import FieldError

def object_relation_base_factory(
        prefix=None,
        prefix_verbose=None,
        add_related_name=False,
        limit_content_type_choices_to=None,
        is_required=False):
    """
    Returns a mixin class for generic foreign keys using
    "Content type - object ID" with dynamic field names.
    This function is just a class generator.

    Parameters:
    prefix:           a prefix, which is added in front of
                      the fields
    prefix_verbose:   a verbose name of the prefix, used to
                      generate a title for the field column
                      of the content object in the Admin
    add_related_name: a boolean value indicating, that a
                      related name for the generated content
                      type foreign key should be added. This
                      value should be true, if you use more
                      than one ObjectRelationBase in your
                      model.

    The model fields are created using this naming scheme:
        <<prefix>>_content_type
        <<prefix>>_object_id
        <<prefix>>_content_object
    """
    p = ""
    if prefix:
        p = f"{prefix}_"

    prefix_verbose = prefix_verbose or _("Related object")
    limit_content_type_choices_to = limit_content_type_choices_to 
     or {}

    content_type_field = f"{p}content_type"
    object_id_field = f"{p}object_id"
    content_object_field = f"{p}content_object"

    class TheClass(models.Model):
 class Meta:
 abstract = True

    if add_related_name:
        if not prefix:
            raise FieldError("if add_related_name is set to "
                             "True, a prefix must be given")
        related_name = prefix
    else:
        related_name = None

    optional = not is_required

    ct_verbose_name = _(f"{prefix_verbose}'s type (model)")

    content_type = models.ForeignKey(
        ContentType,
        verbose_name=ct_verbose_name,
        related_name=related_name,
        blank=optional,
        null=optional,
        help_text=_("Please select the type (model) "
                    "for the relation, you want to build."),
        limit_choices_to=limit_content_type_choices_to,
        on_delete=models.CASCADE)

    fk_verbose_name = prefix_verbose

    object_id = models.CharField(
        fk_verbose_name,
        blank=optional,
        null=False,
        help_text=_("Please enter the ID of the related object."),
        max_length=255,
        default="")  # for migrations

    content_object = GenericForeignKey(
        ct_field=content_type_field,
        fk_field=object_id_field)

    TheClass.add_to_class(content_type_field, content_type)
    TheClass.add_to_class(object_id_field, object_id)
    TheClass.add_to_class(content_object_field, content_object)

    return TheClass
```

1.  以下代码片段是如何在您的应用中使用两个通用关系的示例（将此代码放在`ideas/models.py`中）：

```py
# myproject/apps/ideas/models.py from django.db import models
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.models import (
    object_relation_base_factory as generic_relation,
)

FavoriteObjectBase = generic_relation(
    is_required=True,
)

OwnerBase = generic_relation(
    prefix="owner",
    prefix_verbose=_("Owner"),
    is_required=True,
    add_related_name=True,
    limit_content_type_choices_to={
        "model__in": (
            "user",
            "group",
        )
    }
)

class Like(FavoriteObjectBase, OwnerBase):
    class Meta:
        verbose_name = _("Like")
        verbose_name_plural = _("Likes")

    def __str__(self):
        return _("{owner} likes {object}").format(
            owner=self.owner_content_object,
            object=self.content_object
        )
```

# 它是如何工作的...

正如您所看到的，这个片段比之前的更复杂。

`object_relation_base_factory`函数，我们已经给它起了别名`generic_relation`，在我们的导入中，它本身不是一个 mixin；它是一个生成模型 mixin 的函数-也就是说，一个抽象模型类来扩展。动态创建的 mixin 添加了`content_type`和`object_id`字段以及指向相关实例的`content_object`通用外键。

为什么我们不能只定义一个具有这三个属性的简单模型 mixin？动态生成的抽象类允许我们为每个字段名称添加前缀；因此，我们可以在同一个模型中拥有多个通用关系。例如，之前展示的`Like`模型将为喜欢的对象添加`content_type`、`object_id`和`content_object`字段，以及为喜欢对象的用户或组添加`owner_content_type`、`owner_object_id`和`owner_content_object`。

`object_relation_base_factory`函数，我们已经给它起了别名

对于`generic_relation`的简称，通过`limit_content_type_choices_to`参数添加了限制内容类型选择的可能性。前面的示例将`owner_content_type`的选择限制为`User`和`Group`模型的内容类型。

# 另请参阅

+   *创建一个具有 URL 相关方法的模型 mixin*配方

+   处理创建和修改日期的模型混合的配方

+   处理处理元标签的模型混合的配方

+   在第四章的*实现“喜欢”小部件*配方中，模板和 JavaScript

# 处理多语言字段

Django 使用国际化机制来翻译代码和模板中的冗长字符串。但是开发人员可以决定如何在模型中实现多语言内容。我们将向您展示如何直接在项目中实现多语言模型的几种方法。第一种方法是在模型中使用特定语言字段。

这种方法具有以下特点：

+   在模型中定义多语言字段很简单。

+   在数据库查询中使用多语言字段很简单。

+   您可以使用贡献的管理来编辑具有多语言字段的模型，无需额外修改。

+   如果需要，您可以轻松地在同一模板中显示对象的所有翻译。

+   在设置中更改语言数量后，您需要为所有多语言模型创建和运行迁移。

# 准备工作

您是否已经创建了本章前面配方中使用的`myproject.apps.core`包？现在，您需要在`core`应用程序中创建一个新的`model_fields.py`文件，用于自定义模型字段。

# 如何做...

执行以下步骤来定义多语言字符字段和多语言文本字段：

1.  打开`model_fields.py`文件，并创建基本多语言字段，如下所示：

```py
# myproject/apps/core/model_fields.py from django.conf import settings
from django.db import models
from django.utils.translation import get_language
from django.utils import translation

class MultilingualField(models.Field):
    SUPPORTED_FIELD_TYPES = [models.CharField, models.TextField]

    def __init__(self, verbose_name=None, **kwargs):
        self.localized_field_model = None
        for model in MultilingualField.SUPPORTED_FIELD_TYPES:
            if issubclass(self.__class__, model):
                self.localized_field_model = model
        self._blank = kwargs.get("blank", False)
        self._editable = kwargs.get("editable", True)
        super().__init__(verbose_name, **kwargs)

    @staticmethod
    def localized_field_name(name, lang_code):
        lang_code_safe = lang_code.replace("-", "_")
        return f"{name}_{lang_code_safe}"

    def get_localized_field(self, lang_code, lang_name):
        _blank = (self._blank
                  if lang_code == settings.LANGUAGE_CODE
                  else True)
        localized_field = self.localized_field_model(
            f"{self.verbose_name} ({lang_name})",
            name=self.name,
            primary_key=self.primary_key,
            max_length=self.max_length,
            unique=self.unique,
            blank=_blank,
            null=False, # we ignore the null argument!
            db_index=self.db_index,
            default=self.default or "",
            editable=self._editable,
            serialize=self.serialize,
            choices=self.choices,
            help_text=self.help_text,
            db_column=None,
            db_tablespace=self.db_tablespace)
        return localized_field

    def contribute_to_class(self, cls, name,
                            private_only=False,
                            virtual_only=False):
        def translated_value(self):
            language = get_language()
            val = self.__dict__.get(
                MultilingualField.localized_field_name(
                        name, language))
            if not val:
                val = self.__dict__.get(
                    MultilingualField.localized_field_name(
                            name, settings.LANGUAGE_CODE))
            return val

        # generate language-specific fields dynamically
        if not cls._meta.abstract:
            if self.localized_field_model:
                for lang_code, lang_name in settings.LANGUAGES:
                    localized_field = self.get_localized_field(
                        lang_code, lang_name)
                    localized_field.contribute_to_class(
                            cls,
                            MultilingualField.localized_field_name(
                                    name, lang_code))

                setattr(cls, name, property(translated_value))
            else:
                super().contribute_to_class(
                    cls, name, private_only, virtual_only)
```

1.  在同一文件中，为字符和文本字段表单子类化基本字段，如下所示：

```py
class MultilingualCharField(models.CharField, MultilingualField):
    pass

class MultilingualTextField(models.TextField, MultilingualField):
    pass
```

1.  在核心应用中创建一个`admin.py`文件，并添加以下内容：

```py
# myproject/apps/core/admin.py
from django.conf import settings

def get_multilingual_field_names(field_name):
    lang_code_underscored = settings.LANGUAGE_CODE.replace("-", 
     "_")
    field_names = [f"{field_name}_{lang_code_underscored}"]
    for lang_code, lang_name in settings.LANGUAGES:
        if lang_code != settings.LANGUAGE_CODE:
            lang_code_underscored = lang_code.replace("-", "_")
            field_names.append(
                f"{field_name}_{lang_code_underscored}"
            )
    return field_names
```

现在，我们将考虑如何在应用程序中使用多语言字段的示例，如下所示：

1.  首先，在项目的设置中设置多种语言。假设我们的网站将支持欧盟所有官方语言，英语是默认语言：

```py
# myproject/settings/_base.py LANGUAGE_CODE = "en"

# All official languages of European Union
LANGUAGES = [
    ("bg", "Bulgarian"),    ("hr", "Croatian"),
    ("cs", "Czech"),        ("da", "Danish"),
    ("nl", "Dutch"),        ("en", "English"),
    ("et", "Estonian"),     ("fi", "Finnish"),
    ("fr", "French"),       ("de", "German"),
    ("el", "Greek"),        ("hu", "Hungarian"),
    ("ga", "Irish"),        ("it", "Italian"),
    ("lv", "Latvian"),      ("lt", "Lithuanian"),
    ("mt", "Maltese"),      ("pl", "Polish"),
    ("pt", "Portuguese"),   ("ro", "Romanian"),
    ("sk", "Slovak"),       ("sl", "Slovene"),
    ("es", "Spanish"),      ("sv", "Swedish"),
]
```

1.  然后，打开`myproject.apps.ideas`应用的`models.py`文件，并为`Idea`模型创建多语言字段，如下所示：

```py
# myproject/apps/ideas/models.py
from django.db import models
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.model_fields import (
    MultilingualCharField,
    MultilingualTextField,
)

class Idea(models.Model):
    title = MultilingualCharField(
        _("Title"),
        max_length=200,
    )
    content = MultilingualTextField(
        _("Content"),
    )

    class Meta:
        verbose_name = _("Idea")
        verbose_name_plural = _("Ideas")

    def __str__(self):
        return self.title
```

1.  为`ideas`应用创建一个`admin.py`文件：

```py
# myproject/apps/ideas/admin.py
from django.contrib import admin
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.admin import get_multilingual_field_names

from .models import Idea

@admin.register(Idea)
class IdeaAdmin(admin.ModelAdmin):
    fieldsets = [
        (_("Title and Content"), {
            "fields": get_multilingual_field_names("title") +
                      get_multilingual_field_names("content")
        }),
    ]
```

# 它是如何工作的...

`Idea`的示例将生成一个类似以下的模型：

```py
class Idea(models.Model):
    title_bg = models.CharField(
        _("Title (Bulgarian)"),
        max_length=200,
    )
    title_hr = models.CharField(
        _("Title (Croatian)"),
        max_length=200,
    )
    # titles for other languages…
    title_sv = models.CharField(
        _("Title (Swedish)"),
        max_length=200,
    )

    content_bg = MultilingualTextField(
        _("Content (Bulgarian)"),
    )
    content_hr = MultilingualTextField(
        _("Content (Croatian)"),
    )
    # content for other languages…
    content_sv = MultilingualTextField(
        _("Content (Swedish)"),
    )

    class Meta:
        verbose_name = _("Idea")
        verbose_name_plural = _("Ideas")

    def __str__(self):
        return self.title
```

如果有带有破折号的语言代码，比如瑞士德语的“de-ch”，那么这些语言的字段将被下划线替换，比如`title_de_ch`和`content_de_ch`。

除了生成的特定语言字段之外，还将有两个属性 - `title` 和 `content` - 它们将返回当前活动语言中对应的字段。如果没有可用的本地化字段内容，它们将回退到默认语言。

`MultilingualCharField`和`MultilingualTextField`字段将根据您的`LANGUAGES`设置动态地处理模型字段。它们将覆盖`contribute_to_class()`方法，该方法在 Django 框架创建模型类时使用。多语言字段动态地为项目的每种语言添加字符或文本字段。您需要创建数据库迁移以在数据库中添加适当的字段。此外，创建属性以返回当前活动语言的翻译值或默认情况下的主语言。

在管理中，`get_multilingual_field_names()` 将返回一个特定语言字段名称的列表，从`LANGUAGES`设置中的一个默认语言开始，然后继续使用其他语言。

以下是您可能在模板和视图中使用多语言字段的几个示例。

如果在模板中有以下代码，它将显示当前活动语言的文本，比如立陶宛语，如果翻译不存在，将回退到英语：

```py
<h1>{{ idea.title }}</h1>
<div>{{ idea.content|urlize|linebreaks }}</div>
```

如果您希望将您的`QuerySet`按翻译后的标题排序，可以定义如下：

```py
>>> lang_code = input("Enter language code: ")
>>> lang_code_underscored = lang_code.replace("-", "_")
>>> qs = Idea.objects.order_by(f"title_{lang_code_underscored}")
```

# 另请参阅

+   *使用模型翻译表*配方

+   *使用迁移*配方

+   第六章，模型管理

# 使用模型翻译表

在处理数据库中的多语言内容时，第二种方法涉及为每个多语言模型使用模型翻译表。

这种方法的特点如下：

+   您可以使用贡献的管理来编辑翻译，就像内联一样。

+   更改设置中的语言数量后，不需要进行迁移或其他进一步的操作。

+   您可以轻松地在模板中显示当前语言的翻译，但在同一页上显示特定语言的多个翻译会更困难。

+   您必须了解并使用本配方中描述的特定模式来创建模型翻译。

+   使用这种方法进行数据库查询并不那么简单，但是，正如您将看到的，这仍然是可能的。

# 准备工作

我们将从`myprojects.apps.core`应用程序开始。

# 如何做...

执行以下步骤来准备多语言模型：

1.  在`core`应用程序中，创建带有以下内容的`model_fields.py`：

```py
# myproject/apps/core/model_fields.py
from django.conf import settings
from django.utils.translation import get_language
from django.utils import translation

class TranslatedField(object):
    def __init__(self, field_name):
        self.field_name = field_name

    def __get__(self, instance, owner):
        lang_code = translation.get_language()
        if lang_code == settings.LANGUAGE_CODE:
            # The fields of the default language are in the main
               model
            return getattr(instance, self.field_name)
        else:
            # The fields of the other languages are in the
               translation
            # model, but falls back to the main model
            translations = instance.translations.filter(
                language=lang_code,
            ).first() or instance
            return getattr(translations, self.field_name)
```

1.  将以下内容添加到`core`应用程序的`admin.py`文件中：

```py
# myproject/apps/core/admin.py
from django import forms
from django.conf import settings
from django.utils.translation import gettext_lazy as _

class LanguageChoicesForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        LANGUAGES_EXCEPT_THE_DEFAULT = [
            (lang_code, lang_name)
            for lang_code, lang_name in settings.LANGUAGES
            if lang_code != settings.LANGUAGE_CODE
        ]
        super().__init__(*args, **kwargs)
        self.fields["language"] = forms.ChoiceField(
            label=_("Language"),
            choices=LANGUAGES_EXCEPT_THE_DEFAULT, 
            required=True,
        )
```

现在让我们实现多语言模型：

1.  首先，在项目的设置中设置多种语言。假设我们的网站将支持欧盟所有官方语言，英语是默认语言：

```py
# myproject/settings/_base.py
LANGUAGE_CODE = "en"

# All official languages of European Union
LANGUAGES = [
    ("bg", "Bulgarian"),    ("hr", "Croatian"),
    ("cs", "Czech"),        ("da", "Danish"),
    ("nl", "Dutch"),        ("en", "English"),
    ("et", "Estonian"),     ("fi", "Finnish"),
    ("fr", "French"),       ("de", "German"),
    ("el", "Greek"),        ("hu", "Hungarian"),
    ("ga", "Irish"),        ("it", "Italian"),
    ("lv", "Latvian"),      ("lt", "Lithuanian"),
    ("mt", "Maltese"),      ("pl", "Polish"),
    ("pt", "Portuguese"),   ("ro", "Romanian"),
    ("sk", "Slovak"),       ("sl", "Slovene"),
    ("es", "Spanish"),      ("sv", "Swedish"),
]
```

1.  然后，让我们创建`Idea`和`IdeaTranslations`模型：

```py
# myproject/apps/ideas/models.py
from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.model_fields import TranslatedField

class Idea(models.Model):
    title = models.CharField(
        _("Title"),
        max_length=200,
    )
    content = models.TextField(
        _("Content"),
    )
    translated_title = TranslatedField("title")
    translated_content = TranslatedField("content")

    class Meta:
        verbose_name = _("Idea")
        verbose_name_plural = _("Ideas")

    def __str__(self):
        return self.title

class IdeaTranslations(models.Model):
    idea = models.ForeignKey(
        Idea,
        verbose_name=_("Idea"),
        on_delete=models.CASCADE,
        related_name="translations",
    )
    language = models.CharField(_("Language"), max_length=7)

    title = models.CharField(
        _("Title"),
        max_length=200,
    )
    content = models.TextField(
        _("Content"),
    )

    class Meta:
        verbose_name = _("Idea Translations")
        verbose_name_plural = _("Idea Translations")
        ordering = ["language"]
        unique_together = [["idea", "language"]]

    def __str__(self):
        return self.title
```

1.  最后，创建`ideas`应用程序的`admin.py`如下：

```py
# myproject/apps/ideas/admin.py
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
        (_("Title and Content"), {
            "fields": ["title", "content"]
        }),
    ]
```

# 工作原理...

我们将默认语言的特定于语言的字段保留在`Idea`模型本身中。每种语言的翻译都在`IdeaTranslations`模型中，该模型将作为内联翻译列在管理中列出。`IdeaTranslations`模型没有模型的语言选择，这是有原因的——我们不希望每次添加新语言或删除某种语言时都创建迁移。相反，语言选择设置在管理表单中，还要确保默认语言被跳过或在列表中不可选择。语言选择使用`LanguageChoicesForm`类进行限制。

要获取当前语言中的特定字段，您将使用定义为`TranslatedField`的字段。在模板中，看起来像这样：

```py
<h1>{{ idea.translated_title }}</h1>
<div>{{ idea.translated_content|urlize|linebreaks }}</div>
```

要按特定语言的翻译标题对项目进行排序，您将使用`annotate()`方法如下：

```py
>>> from django.conf import settings
>>> from django.db import models
>>> lang_code = input("Enter language code: ")

>>> if lang_code == settings.LANGUAGE_CODE:
...     qs = Idea.objects.annotate(
...         title_translation=models.F("title"),
...         content_translation=models.F("content"),
...     )
... else:
...     qs = Idea.objects.filter(
...         translations__language=lang_code,
...     ).annotate(
...         title_translation=models.F("translations__title"),
...         content_translation=models.F("translations__content"),
...     )

>>> qs = qs.order_by("title_translation")

>>> for idea in qs:
...     print(idea.title_translation)
```

在这个例子中，我们在 Django shell 中提示输入语言代码。如果语言是默认语言，我们将`title`和`content`存储为`Idea`模型的`title_translation`和`content_translation`。如果选择了其他语言，我们将从选择的语言中读取`title`和`content`作为`IdeaTranslations`模型的`title_translation`和`content_translation`。

之后，我们可以通过`title_translation`或`content_translation`筛选或排序`QuerySet`。

# 另请参阅

+   *处理多语言字段*配方

+   第六章，模型管理

# 避免循环依赖

在开发 Django 模型时，非常重要的是要避免循环依赖，特别是在`models.py`文件中。循环依赖是指不同 Python 模块之间的相互导入。您不应该从不同的`models.py`文件中交叉导入，因为这会导致严重的稳定性问题。相反，如果存在相互依赖，您应该使用本配方中描述的操作。

# 准备工作

让我们使用`categories`和`ideas`应用程序来说明如何处理交叉依赖。

# 如何做...

在处理使用其他应用程序模型的模型时，请遵循以下实践：

1.  对于来自其他应用程序的模型的外键和多对多关系，请使用`"<app_label>.<model>"`声明，而不是导入模型。在 Django 中，这适用于`ForeignKey`，`OneToOneField`和`ManyToManyField`，例如：

```py
# myproject/apps/ideas/models.py from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _

class Idea(models.Model):
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        verbose_name=_("Author"),
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
    )
    category = models.ForeignKey(
        "categories.Category",
        verbose_name=_("Category"),
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
    )
    # other fields, attributes, properties and methods…
```

这里，`settings.AUTH_USER_MODEL`是一个具有值如`"auth.User"`的设置：

1.  如果您需要在方法中访问另一个应用程序的模型，请在方法内部导入该模型，而不是在模块级别导入，例如：

```py
# myproject/apps/categories/models.py
from django.db import models
from django.utils.translation import gettext_lazy as _

class Category(models.Model):
    # fields, attributes, properties, and methods…

    def get_ideas_without_this_category(self):
        from myproject.apps.ideas.models import Idea
        return Idea.objects.exclude(category=self)
```

1.  如果您使用模型继承，例如用于模型混合，将基类保留在单独的应用程序中，并将它们放在`INSTALLED_APPS`中将使用它们的其他应用程序之前，如下所示：

```py
# myproject/settings/_base.py

INSTALLED_APPS = [
    # contributed
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # third-party
    # ...
    # local
    "myproject.apps.core",
    "myproject.apps.categories",
    "myproject.apps.ideas",
]
```

在这里，`ideas`应用程序将如下使用`core`应用程序的模型混合：

```py
# myproject/apps/ideas/models.py
from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.models import (
 CreationModificationDateBase,
 MetaTagsBase,
 UrlBase,
)

class Idea(CreationModificationDateBase, MetaTagsBase, UrlBase):
    # fields, attributes, properties, and methods…
```

# 另请参阅

+   第一章*中的*为开发、测试、暂存和生产环境配置设置*示例，Django 3.0 入门

+   第一章**中的*尊重 Python 文件的导入顺序*示例，Django 3.0 入门**

+   *使用模型混合*示例

+   *将外键更改为多对多字段*示例

# 添加数据库约束

为了更好地保证数据库的完整性，通常会定义数据库约束，告诉某些字段绑定到其他数据库表的字段，使某些字段唯一或非空。对于高级数据库约束，例如使字段在满足条件时唯一或为某些字段的值设置特定条件，Django 有特殊的类：`UniqueConstraint`和`CheckConstraint`。在这个示例中，您将看到如何使用它们的实际示例。

# 准备工作

让我们从`ideas`应用程序和将至少具有`title`和`author`字段的`Idea`模型开始。

# 如何做...

在`Idea`模型的`Meta`类中设置数据库约束如下：

```py
# myproject/apps/ideas/models.py
from django.db import models
from django.utils.translation import gettext_lazy as _

class Idea(models.Model):
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        verbose_name=_("Author"),
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        related_name="authored_ideas",
    )
    title = models.CharField(
        _("Title"),
        max_length=200,
    )

    class Meta:
        verbose_name = _("Idea")
        verbose_name_plural = _("Ideas")
        constraints = [
 models.UniqueConstraint(
 fields=["title"],
 condition=~models.Q(author=None),
 name="unique_titles_for_each_author",
 ),
 models.CheckConstraint(
 check=models.Q(
 title__iregex=r"^\S.*\S$"
 # starts with non-whitespace,
 # ends with non-whitespace,
 # anything in the middle
 ),
 name="title_has_no_leading_and_trailing_whitespaces",
 )
 ]
```

# 它是如何工作的...

我们在数据库中定义了两个约束。

第一个`UniqueConstraint`告诉标题对于每个作者是唯一的。如果作者未设置，则标题可以重复。要检查作者是否已设置，我们使用否定查找：`~models.Q(author=None)`。请注意，在 Django 中，查找的`~`运算符等同于 QuerySet 的`exclude()`方法，因此这些 QuerySets 是等价的：

```py
ideas_with_authors = Idea.objects.exclude(author=None)
ideas_with_authors2 = Idea.objects.filter(~models.Q(author=None))
```

第二个约束条件`CheckConstraint`检查标题是否不以空格开头和结尾。为此，我们使用正则表达式查找。

# 还有更多...

数据库约束不会影响表单验证。如果保存条目到数据库时任何数据不符合其条件，它们只会引发`django.db.utils.IntegrityError`。

如果您希望在表单中验证数据，您必须自己实现验证，例如在模型的`clean()`方法中。对于`Idea`模型，这将如下所示：

```py
# myproject/apps/ideas/models.py from django.db import models
from django.conf import settings
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

class Idea(models.Model):
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        verbose_name=_("Author"),
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        related_name="authored_ideas2",
    )
    title = models.CharField(
        _("Title"),
        max_length=200,
    )

    # other fields and attributes…

    class Meta:
        verbose_name = _("Idea")
        verbose_name_plural = _("Ideas")
        constraints = [
            models.UniqueConstraint(
                fields=["title"],
                condition=~models.Q(author=None),
                name="unique_titles_for_each_author2",
            ),
            models.CheckConstraint(
                check=models.Q(
                    title__iregex=r"^\S.*\S$"
                    # starts with non-whitespace,
                    # ends with non-whitespace,
                    # anything in the middle
                ),
                name="title_has_no_leading_and_trailing_whitespaces2",
            )
        ]

 def clean(self):
 import re
 if self.author and Idea.objects.exclude(pk=self.pk).filter(
 author=self.author,
 title=self.title,
 ).exists():
 raise ValidationError(
 _("Each idea of the same user should have a unique title.")
 )
 if not re.match(r"^\S.*\S$", self.title):
 raise ValidationError(
 _("The title cannot start or end with a whitespace.")
 )

    # other properties and methods…
```

# 另请参阅

+   第三章*中的*表单和视图*

+   第十章*中的*使用数据库查询表达式*示例，花里胡哨

# 使用迁移

在敏捷软件开发中，项目的要求会随着时间的推移而不断更新和更新。随着开发的进行，您将不得不沿途执行数据库架构更改。使用 Django 迁移，您不必手动更改数据库表和字段，因为大部分工作都是自动完成的，使用命令行界面。

# 准备工作

在命令行工具中激活您的虚拟环境，并将活动目录更改为您的项目目录。

# 如何做...

要创建数据库迁移，请查看以下步骤：

1.  当您在新的`categories`或`ideas`应用程序中创建模型时，您必须创建一个初始迁移，该迁移将为您的应用程序创建数据库表。这可以通过使用以下命令来完成：

```py
(env)$ python manage.py makemigrations ideas
```

1.  第一次要为项目创建所有表时，请运行以下命令：

```py
(env)$ python manage.py migrate
```

当您想要执行所有应用程序的新迁移时，请运行此命令。

1.  如果要执行特定应用程序的迁移，请运行以下命令：

```py
(env)$ python manage.py migrate ideas
```

1.  如果对数据库模式进行了一些更改，则必须为该模式创建一个迁移。例如，如果我们向 idea 模型添加一个新的 subtitle 字段，可以使用以下命令创建迁移：

```py
(env)$ python manage.py makemigrations --name=subtitle_added ideas
```

然而，`--name=subtitle_added`字段可以被跳过，因为在大多数情况下，Django 会生成相当自解释的默认名称。

1.  有时，您可能需要批量添加或更改现有模式中的数据，这可以通过数据迁移而不是模式迁移来完成。要创建修改数据库表中数据的数据迁移，可以使用以下命令：

```py
(env)$ python manage.py makemigrations --name=populate_subtitle \
> --empty ideas
```

`--empty`参数告诉 Django 创建一个骨架数据迁移，您必须在应用之前修改它以执行必要的数据操作。对于数据迁移，建议设置名称。

1.  要列出所有可用的已应用和未应用的迁移，请运行以下命令：

```py
(env)$ python manage.py showmigrations
```

已应用的迁移将以[X]前缀列出。未应用的迁移将以[ ]前缀列出。

1.  要列出特定应用程序的所有可用迁移，请运行相同的命令，但传递应用程序名称，如下所示：

```py
(env)$ python manage.py showmigrations ideas
```

# 它是如何工作的...

Django 迁移是数据库迁移机制的指令文件。这些指令文件告诉我们要创建或删除哪些数据库表，要添加或删除哪些字段，以及要插入、更新或删除哪些数据。它们还定义了哪些迁移依赖于其他迁移。

Django 有两种类型的迁移。一种是模式迁移，另一种是数据迁移。当您添加新模型、添加或删除字段时，应创建模式迁移。当您想要向数据库填充一些值或大量删除数据库中的值时，应使用数据迁移。数据迁移应该通过命令行工具中的命令创建，然后在迁移文件中编码。

每个应用程序的迁移都保存在它们的`migrations`目录中。第一个迁移通常称为`0001_initial.py`，在我们的示例应用程序中，其他迁移将被称为`0002_subtitle_added.py`和`0003_populate_subtitle.py`。每个迁移都有一个自动递增的数字前缀。对于执行的每个迁移，都会在`django_migrations`数据库表中保存一个条目。

可以通过指定要迁移的迁移编号来来回迁移，如下命令所示：

```py
(env)$ python manage.py migrate ideas 0002

```

要取消应用程序的所有迁移，包括初始迁移，请运行以下命令：

```py
(env)$ python manage.py migrate ideas zero
```

取消迁移需要每个迁移都有前向和后向操作。理想情况下，后向操作应该恢复前向操作所做的更改。然而，在某些情况下，这样的更改是无法恢复的，例如当前向操作从模式中删除了一个列时，因为它将破坏数据。在这种情况下，后向操作可能会恢复模式，但数据将永远丢失，或者根本没有后向操作。

在测试了前向和后向迁移过程并确保它们在其他开发和公共网站环境中能够正常工作之前，不要将您的迁移提交到版本控制中。

# 还有更多...

在官方的*How To*指南中了解更多关于编写数据库迁移的信息，网址为[`docs.djangoproject.com/en/2.2/howto/writing-migrations/`](https://docs.djangoproject.com/en/2.2/howto/writing-migrations/)​。

# 另请参阅

+   第一章*中的*使用虚拟环境*配方

+   在第一章*中的*使用 Django、Gunicorn、Nginx 和 PostgreSQL 的 Docker 容器*食谱，使用 Django 3.0 入门

+   在第一章*中的*使用 pip 处理项目依赖关系*食谱，使用 Django 3.0 入门

+   在第一章*中的*在您的项目中包含外部依赖项*食谱，使用 Django 3.0 入门

+   *将外键更改为多对多字段*食谱

# 将外键更改为多对多字段

这个食谱是如何将多对一关系更改为多对多关系的实际示例，同时保留已经存在的数据。在这种情况下，我们将同时使用模式迁移和数据迁移。

# 准备就绪

假设您有`Idea`模型，其中有一个指向`Category`模型的外键。

1.  让我们在`categories`应用程序中定义`Category`模型，如下所示：

```py
# myproject/apps/categories/models.py
from django.db import models
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.model_fields import MultilingualCharField

class Category(models.Model):
    title = MultilingualCharField(
        _("Title"),
        max_length=200,
    )

    class Meta:
        verbose_name = _("Category")
        verbose_name_plural = _("Categories")

    def __str__(self):
        return self.title
```

1.  让我们在`ideas`应用程序中定义`Idea`模型，如下所示：

```py
# myproject/apps/ideas/models.py from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.model_fields import (
    MultilingualCharField,
    MultilingualTextField,
)

class Idea(models.Model):
    title = MultilingualCharField(
        _("Title"),
        max_length=200,
    )
    content = MultilingualTextField(
        _("Content"),
    )
 category = models.ForeignKey(
        "categories.Category",
        verbose_name=_("Category"),
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
        related_name="category_ideas",
    ) 
    class Meta:
        verbose_name = _("Idea")
        verbose_name_plural = _("Ideas")

    def __str__(self):
        return self.title
```

1.  通过使用以下命令创建和执行初始迁移：

```py
(env)$ python manage.py makemigrations categories
(env)$ python manage.py makemigrations ideas
(env)$ python manage.py migrate
```

# 如何做...

以下步骤将向您展示如何从外键关系切换到多对多关系，同时保留已经存在的数据：

1.  添加一个名为`categories`的新多对多字段，如下所示：

```py
# myproject/apps/ideas/models.py from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.model_fields import (
    MultilingualCharField,
    MultilingualTextField,
)

class Idea(models.Model):
    title = MultilingualCharField(
        _("Title"),
        max_length=200,
    )
    content = MultilingualTextField(
        _("Content"),
    )
    category = models.ForeignKey(
        "categories.Category",
        verbose_name=_("Category"),
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
        related_name="category_ideas",
    )
    categories = models.ManyToManyField(
 "categories.Category",
 verbose_name=_("Categories"),
 blank=True,
 related_name="ideas",
 )

    class Meta:
        verbose_name = _("Idea")
        verbose_name_plural = _("Ideas")

    def __str__(self):
        return self.title
```

1.  创建并运行模式迁移，以向数据库添加新的关系，如下面的代码片段所示：

```py
(env)$ python manage.py makemigrations ideas
(env)$ python manage.py migrate ideas
```

1.  创建一个数据迁移，将类别从外键复制到多对多字段，如下所示：

```py
(env)$ python manage.py makemigrations --empty \
> --name=copy_categories ideas
```

1.  打开新创建的迁移文件（`0003_copy_categories.py`），并定义前向迁移指令，如下面的代码片段所示：

```py
# myproject/apps/ideas/migrations/0003_copy_categories.py from django.db import migrations

def copy_categories(apps, schema_editor):
 Idea = apps.get_model("ideas", "Idea")
 for idea in Idea.objects.all():
 if idea.category:
 idea.categories.add(idea.category)

class Migration(migrations.Migration):

    dependencies = [
        ('ideas', '0002_idea_categories'),
    ]

    operations = [
        migrations.RunPython(copy_categories),
    ]
```

1.  运行新的数据迁移，如下所示：

```py
(env)$ python manage.py migrate ideas
```

1.  在`models.py`文件中删除外键`category`字段，只留下新的`categories`多对多字段，如下所示：

```py
# myproject/apps/ideas/models.py from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _

from myproject.apps.core.model_fields import (
    MultilingualCharField,
    MultilingualTextField,
)

class Idea(models.Model):
    title = MultilingualCharField(
        _("Title"),
        max_length=200,
    )
    content = MultilingualTextField(
        _("Content"),
    )

    categories = models.ManyToManyField(
 "categories.Category",
 verbose_name=_("Categories"),
 blank=True,
 related_name="ideas",
 )

    class Meta:
        verbose_name = _("Idea")
        verbose_name_plural = _("Ideas")

    def __str__(self):
        return self.title
```

1.  创建并运行模式迁移，以从数据库表中删除`Categories`字段，如下所示：

```py
(env)$ python manage.py makemigrations ideas
(env)$ python manage.py migrate ideas
```

# 它是如何工作的...

首先，我们向`Idea`模型添加一个新的多对多字段，并生成一个迁移以相应地更新数据库。然后，我们创建一个数据迁移，将现有关系从外键`category`复制到新的多对多`categories`。最后，我们从模型中删除外键字段，并再次更新数据库。

# 还有更多...

我们的数据迁移目前只包括前向操作，将外键中的类别复制到多对多字段中

将类别键作为新类别关系中的第一个相关项目。虽然我们在这里没有详细说明，在实际情况下最好也包括反向操作。这可以通过将第一个相关项目复制回`category`外键来实现。不幸的是，任何具有多个类别的`Idea`对象都将丢失额外数据。

# 另请参阅

+   *使用迁移*食谱

+   *处理多语言字段*食谱

+   *使用模型翻译表*食谱

+   *避免循环依赖*食谱
