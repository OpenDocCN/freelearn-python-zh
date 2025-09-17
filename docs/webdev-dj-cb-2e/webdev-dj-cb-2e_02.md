# 第二章：数据库结构

本章将涵盖以下主题：

+   使用模型混入

+   创建一个包含与 URL 相关方法的模型混入

+   创建一个用于处理创建和修改日期的模型混入

+   创建一个用于处理元标签的模型混入

+   创建一个用于处理通用关系的模型混入

+   处理多语言字段

+   使用迁移

+   从 South 迁移切换到 Django 迁移

+   将外键更改为多对多字段

# 简介

当你启动一个新应用时，首先要做的是创建代表你的数据库结构的模型。我们假设你之前已经创建了 Django 应用，或者至少你已经阅读并理解了官方的 Django 教程。在本章中，我们将看到一些使你的项目中的不同应用保持数据库结构一致性的有趣技术。然后，我们将看到如何创建自定义模型字段，以便在数据库中处理数据的国际化。本章结束时，我们将看到如何使用迁移在开发过程中更改你的数据库结构。

# 使用模型混入

在面向对象的语言，如 Python 中，混入类可以被视为一个具有实现功能的接口。当一个模型扩展混入时，它实现了该接口，并包括所有其字段、属性和方法。在 Django 模型中，当你想要在不同模型中多次重用通用功能时，可以使用混入。

## 准备工作

首先，你需要创建可重用的混入。本章后面将给出一些典型的混入示例。一个存放模型混入的好地方是在`utils`模块中。

### 小贴士

如果你创建了一个将与他人共享的可重用应用，请将模型混入放在可重用应用中，例如在`base.py`文件中。

## 如何操作...

打开你想要使用混入的任何 Django 应用的`models.py`文件，并输入以下代码：

```py
# demo_app/models.py
# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
from django.db import models
from django.utils.translation import ugettext_lazy as _
from django.utils.encoding import python_2_unicode_compatible
from utils.models import UrlMixin
from utils.models import CreationModificationMixin
from utils.models import MetaTagsMixin

@python_2_unicode_compatible
class Idea(UrlMixin, CreationModificationMixin, MetaTagsMixin):
  title = models.CharField(_("Title"), max_length=200)
  content = models.TextField(_("Content"))

  class Meta:
    verbose_name = _("Idea")
    verbose_name_plural = _("Ideas")

  def __str__(self):
    return self.title
```

## 它是如何工作的...

Django 模型继承支持三种继承类型：抽象基类、多表继承和代理模型。模型混入是具有指定字段、属性和方法的抽象模型类。当你创建一个如`Idea`这样的模型时，如前例所示，它会继承来自`UrlMixin`、`CreationModificationMixin`和`MetaTagsMixin`的所有特性。所有抽象类的字段都保存在与扩展模型相同的数据库表中。在接下来的食谱中，你将学习如何定义你的模型混入。

注意，我们为`Idea`模型使用了`@python_2_unicode_compatible`装饰器。你可能还记得在第一章中“使你的代码兼容 Python 2.7 和 Python 3”食谱中的内容，它的目的是使`__str__()`方法与 Unicode 兼容，适用于以下两个 Python 版本：2.7 和 3。

## 更多内容...

要了解不同类型的模型继承类型，请参考在[`docs.djangoproject.com/en/1.8/topics/db/models/#model-inheritance`](https://docs.djangoproject.com/en/1.8/topics/db/models/#model-inheritance)提供的官方 Django 文档。

## 参见

+   在 第一章 的 *使代码兼容 Python 2.7 和 Python 3* 菜谱中，*开始使用 Django 1.8*

+   *创建一个具有 URL 相关方法的模型混入* 菜谱

+   *创建一个处理创建和修改日期的模型混入* 菜谱

+   *创建一个处理元标签的模型混入* 菜谱

# 创建一个具有 URL 相关方法的模型混入

对于每个有自己的页面的模型，定义 `get_absolute_url()` 方法是一个好的实践。这个方法可以在模板中使用，也可以在 Django 管理站点中预览保存的对象。然而，`get_absolute_url()` 是模糊的，因为它返回的是 URL 路径而不是完整的 URL。在这个菜谱中，我们将看到如何创建一个模型混入，允许你默认定义 URL 路径或完整 URL，并自动生成另一个，并处理正在设置的 `get_absolute_url()` 方法。

## 准备工作

如果你还没有这样做，创建 `utils` 包以保存你的混入。然后，在 `utils` 包中创建 `models.py` 文件（或者，如果你创建了一个可重用的应用，将混入放在你的应用中的 `base.py` 文件中）。

## 如何操作...

依次执行以下步骤：

1.  将以下内容添加到你的 `utils` 包的 `models.py` 文件中：

    ```py
    # utils/models.py
    # -*- coding: UTF-8 -*-
    from __future__ import unicode_literals
    import urlparse
    from django.db import models
    from django.contrib.sites.models import Site
    from django.conf import settings

    class UrlMixin(models.Model):
        """
        A replacement for get_absolute_url()
        Models extending this mixin should have 
        either get_url or get_url_path implemented.
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
            website_url = getattr(
                settings, "DEFAULT_WEBSITE_URL",
     "http://127.0.0.1:8000"
            )
            return website_url + path
        get_url.dont_recurse = True

        def get_url_path(self):
            if hasattr(self.get_url, "dont_recurse"):
                raise NotImplementedError
            try:
                url = self.get_url()
            except NotImplementedError:
                raise
            bits = urlparse.urlparse(url)
            return urlparse.urlunparse(("", "") + bits[2:])
        get_url_path.dont_recurse = True

        def get_absolute_url(self):
            return self.get_url_path()
    ```

1.  要在你的应用中使用混入，从 `utils` 包中导入它，在你的模型类中继承混入，并定义 `get_url_path()` 方法如下：

    ```py
    # demo_app/models.py
    # -*- coding: UTF-8 -*-
    from __future__ import unicode_literals
    from django.db import models
    from django.utils.translation import ugettext_lazy as _
    from django.core.urlresolvers import reverse
    from django.utils.encoding import \
        python_2_unicode_compatible

    from utils.models import UrlMixin

    @python_2_unicode_compatible
    class Idea(UrlMixin):
        title = models.CharField(_("Title"), max_length=200)

        # …

        get_url_path(self):
            return reverse("idea_details", kwargs={
                "idea_id": str(self.pk),
            })
    ```

1.  如果你在这个代码的预发布或生产环境中进行检查，或者运行一个与默认 IP 或端口不同的本地服务器，请在你的本地设置中设置 `DEFAULT_WEBSITE_URL`（不带尾随斜杠），如下所示：

    ```py
    # settings.py
    # …
    DEFAULT_WEBSITE_URL = "http://www.example.com"
    ```

## 它是如何工作的…

`UrlMixin` 类是一个具有三个方法：`get_url()`、`get_url_path()` 和 `get_absolute_url()` 的抽象模型。期望在扩展模型类（例如，`Idea`）中覆盖 `get_url()` 或 `get_url_path()` 方法。你可以定义 `get_url()`，这是对象的完整 URL，然后 `get_url_path()` 将将其剥离为路径。你也可以定义 `get_url_path()`，这是对象的绝对路径，然后 `get_url()` 将在路径的开头添加网站 URL。`get_absolute_url()` 方法将模仿 `get_url_path()` 方法。

### 小贴士

常规做法是始终覆盖 `get_url_path()` 方法。

在模板中，当你需要同一网站中对象的链接时，使用 `<a href="{{ idea.get_url_path }}">{{ idea.title }}</a>`。对于电子邮件、RSS 源或 API 中的链接，使用 `<a href="{{ idea.get_url }}">{{ idea.title }}</a>`。

默认的`get_absolute_url()`方法将在 Django 模型管理中用于*网站视图*功能，也可能被一些第三方 Django 应用程序使用。

## 相关内容

+   *使用模型混入器*配方

+   *创建一个处理创建和修改日期的模型混入器*配方

+   *创建一个处理元标签的模型混入器*配方

+   *创建一个处理通用关系的模型混入器*配方

# 创建一个处理创建和修改日期的模型混入器

在模型实例的创建和修改中包含时间戳是一种常见的行为。在这个配方中，我们将看到如何创建一个简单的模型混入器，用于保存模型的创建和修改日期和时间。使用这样的混入器将确保所有模型使用相同的字段名来存储时间戳，并且具有相同的行为。

## 准备工作

如果你还没有这样做，创建`utils`包以保存你的混入器。然后，在`utils`包中创建`models.py`文件。

## 如何做到这一点...

打开你的`utils`包中的`models.py`文件，并在其中插入以下内容：

```py
# utils/models.py
# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
from django.db import models
from django.utils.translation import ugettext_lazy as _
from django.utils.timezone import now as timezone_now

class CreationModificationDateMixin(models.Model):
  """
  Abstract base class with a creation and modification
  date and time
  """

  created = models.DateTimeField(
    _("creation date and time"),
    editable=False,
  )

  modified = models.DateTimeField(
    _("modification date and time"),
    null=True,
    editable=False,
  )

  def save(self, *args, **kwargs):
    if not self.pk:
      self.created = timezone_now()
    else:
      # To ensure that we have a creation data always,
      # we add this one
    if not self.created:
      self.created = timezone_now()

      self.modified = timezone_now()

      super(CreationModificationDateMixin, self).\
      save(*args, **kwargs)
    save.alters_data = True

  class Meta:
    abstract = True
```

## 它是如何工作的...

`CreationModificationDateMixin`类是一个抽象模型，这意味着扩展模型类将在同一个数据库表中创建所有字段，也就是说，不会有导致表难以处理的一对一关系。这个混入器有两个日期时间字段和一个在保存扩展模型时将被调用的`save()`方法。`save()`方法检查模型是否有主键，这是新未保存实例的情况。在这种情况下，它将创建日期设置为当前日期和时间。如果存在主键，则将修改日期设置为当前日期和时间。

作为替代，除了`save()`方法之外，你还可以为创建和修改字段使用`auto_now_add`和`auto_now`属性，这将自动添加创建和修改时间戳。

## 相关内容

+   *使用模型混入器*配方

+   *创建一个处理元标签的模型混入器*配方

+   *创建一个处理通用关系的模型混入器*配方

# 创建一个处理元标签的模型混入器

如果你想要优化你的网站以适应搜索引擎，你不仅需要为每个页面设置语义标记，还需要设置适当的元标签。为了获得最大的灵活性，你需要有一种方法来为每个对象定义特定的元标签，每个对象在你的网站上都有自己的页面。在这个配方中，我们将看到如何创建与元标签相关的字段和方法模型混入器。

## 准备工作

如前几个配方所示，确保你有用于混入器的`utils`包。在你的首选编辑器中打开此包的`models.py`文件。

## 如何做到这一点...

将以下内容放入`models.py`文件中：

```py
# utils/models.py
# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
from django.db import models
from django.utils.translation import ugettext_lazy as _
from django.template.defaultfilters import escape
from django.utils.safestring import mark_safe

class MetaTagsMixin(models.Model):
  """
  Abstract base class for meta tags in the <head> section
  """
  meta_keywords = models.CharField(
    _("Keywords"),
    max_length=255,
    blank=True,
    help_text=_("Separate keywords by comma."),
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

    def get_meta_keywords(self):
      tag = ""
      if self.meta_keywords:
        tag = '<meta name="keywords" content="%s" />\n' %\
          escape(self.meta_keywords)
      return mark_safe(tag)

    def get_meta_description(self):
      tag = ""
      if self.meta_description:
        tag = '<meta name="description" content="%s" />\n' %\
          escape(self.meta_description)
      return mark_safe(tag)

    def get_meta_author(self):
      tag = ""
      if self.meta_author:
        tag = '<meta name="author" content="%s" />\n' %\
          escape(self.meta_author)
      return mark_safe(tag)

    def get_meta_copyright(self):
      tag = ""
      if self.meta_copyright:
        tag = '<meta name="copyright" content="%s" />\n' %\
          escape(self.meta_copyright)
      return mark_safe(tag)

    def get_meta_tags(self):
      return mark_safe("".join((
        self.get_meta_keywords(),
        self.get_meta_description(),
        self.get_meta_author(),
        self.get_meta_copyright(),
      )))
```

## 它是如何工作的...

此混入为其扩展的模型添加了四个字段：`meta_keywords`、`meta_description`、`meta_author`和`meta_copyright`。还添加了在 HTML 中渲染元标签的方法。

如果您在如`Idea`这样的模型中使用此混入，该模型在本章的第一个菜谱中展示，那么您可以在您的详情页模板的`HEAD`部分放入以下内容以渲染所有元标签：

```py
{{ idea.get_meta_tags }}
```

您还可以使用以下行来渲染特定的元标签：

```py
{{ idea.get_meta_description }}
```

如您从代码片段中注意到的，渲染的元标签被标记为安全，即它们没有被转义，我们不需要使用安全模板过滤器。只有来自数据库的值被转义，以确保最终的 HTML 格式正确。

## 参见

+   *使用模型混入* 菜谱

+   *创建一个处理创建和修改日期的模型混入* 菜谱

+   *创建一个处理通用关系的模型混入* 菜谱

# 创建一个处理通用关系的模型混入

除了外键关系或多对多关系等常规数据库关系之外，Django 还有一个将模型与任何其他模型的实例相关联的机制。这个概念被称为通用关系。对于每个通用关系，都有一个保存的相关模型的类型以及该模型实例的 ID。

在这个菜谱中，我们将看到如何在模型混入中泛化通用关系的创建。

## 准备工作

为了使这个菜谱工作，您需要安装`contenttypes`应用。默认情况下，它应该在`INSTALLED_APPS`目录中，如下所示：

```py
# settings.py
INSTALLED_APPS = (
    # …
    "django.contrib.contenttypes",
)
```

再次确保您已经为您模型混入创建了`utils`包。

## 如何做到...

1.  在文本编辑器中打开`utils`包中的`models.py`文件，并插入以下内容：

    ```py
    # utils/models.py
    # -*- coding: UTF-8 -*-
    from __future__ import unicode_literals
    from django.db import models
    from django.utils.translation import ugettext_lazy as _
    from django.contrib.contenttypes.models import ContentType
    from django.contrib.contenttypes import generic
    from django.core.exceptions import FieldError

    def object_relation_mixin_factory(
      prefix=None,
      prefix_verbose=None,
      add_related_name=False,
      limit_content_type_choices_to={},
      limit_object_choices_to={},
      is_required=False,
    ):
      """
        returns a mixin class for generic foreign keys using
        "Content type - object Id" with dynamic field names.
        This function is just a class generator

        Parameters:
        prefix : a prefix, which is added in front of the fields
        prefix_verbose :    a verbose name of the prefix, used to
                            generate a title for the field column
                            of the content object in the Admin.
        add_related_name :  a boolean value indicating, that a
                            related name for the generated content
                            type foreign key should be added. This
                            value should be true, if you use more
                            than one ObjectRelationMixin in your model.

        The model fields are created like this:

        <<prefix>>_content_type :   Field name for the "content type"
        <<prefix>>_object_id :      Field name for the "object Id"
        <<prefix>>_content_object : Field name for the "content object"

        """
        p = ""
        if prefix:
          p = "%s_" % prefix

        content_type_field = "%scontent_type" % p
        object_id_field = "%sobject_id" % p
        content_object_field = "%scontent_object" % p

        class TheClass(models.Model):
          class Meta:
            abstract = True

        if add_related_name:
          if not prefix:
            raise FieldError("if add_related_name is set to True,"
              "a prefix must be given")
            related_name = prefix
        else:
          related_name = None

        optional = not is_required

        ct_verbose_name = (
          _("%s's type (model)") % prefix_verbose
          if prefix_verbose
          else _("Related object's type (model)")
        )

        content_type = models.ForeignKey(
          ContentType,
          verbose_name=ct_verbose_name,
          related_name=related_name,
          blank=optional,
          null=optional,
          help_text=_("Please select the type (model) for the relation, you want to build."),
          limit_choices_to=limit_content_type_choices_to,
        )

        fk_verbose_name = (prefix_verbose or _("Related object"))

        object_id = models.CharField(
          fk_verbose_name,
          blank=optional,
          null=False,
          help_text=_("Please enter the ID of the related object."),
          max_length=255,
          default="",  # for south migrations
        )
        object_id.limit_choices_to = limit_object_choices_to
        # can be retrieved by 
        # MyModel._meta.get_field("object_id").limit_choices_to

        content_object = generic.GenericForeignKey(
          ct_field=content_type_field,
          fk_field=object_id_field,
        )

        TheClass.add_to_class(content_type_field, content_type)
        TheClass.add_to_class(object_id_field, object_id)
        TheClass.add_to_class(content_object_field, content_object)

        return TheClass
    ```

1.  以下是如何在您的应用中使用两个通用关系的一个示例（将此代码放入`demo_app/models.py`），如下所示：

    ```py
    # demo_app/models.py
    # -*- coding: UTF-8 -*-
    from __future__ import nicode_literals
    from django.db import models
    from utils.models import object_relation_mixin_factory
    from django.utils.encoding import python_2_unicode_compatible

    FavoriteObjectMixin = object_relation_mixin_factory(
        is_required=True,
    )

    OwnerMixin = object_relation_mixin_factory(
        prefix="owner",
        prefix_verbose=_("Owner"),
        add_related_name=True,
        limit_content_type_choices_to={
            'model__in': ('user', 'institution')
        },
        is_required=True,
    )

    @python_2_unicode_compatible
    class Like(FavoriteObjectMixin, OwnerMixin):
        class Meta:
            verbose_name = _("Like")
            verbose_name_plural = _("Likes")

        def __str__(self):
            return _("%(owner)s likes %(obj)s") % {
                "owner": self.owner_content_object,
                "obj": self.content_object,
            }
    ```

## 它是如何工作的...

如您所见，这个片段比之前的更复杂。`object_relation_mixin_factory`对象本身不是一个混入；它是一个生成模型混入的函数，即一个可以从中扩展的抽象模型类。动态创建的混入添加了`content_type`和`object_id`字段以及指向相关实例的`content_object`通用外键。

为什么我们不能只定义一个包含这三个属性的简单模型混入呢？一个动态生成的抽象类允许我们为每个字段名设置前缀；因此，我们可以在同一个模型中拥有多个通用关系。例如，之前展示的`Like`模型将为喜欢的对象添加`content_type`、`object_id`和`content_object`字段，以及为喜欢该对象的（用户或机构）添加`owner_content_type`、`owner_object_id`和`owner_content_object`字段。

`object_relation_mixin_factory()` 函数通过 `limit_content_type_choices_to` 参数添加了限制内容类型选择的可能性。前面的例子仅将 `owner_content_type` 的选择限制为 `User` 和 `Institution` 模型的内容类型。此外，还有一个 `limit_object_choices_to` 参数，可以用于自定义表单验证，仅将通用关系限制为特定对象，例如具有发布状态的对象。

## 参见

+   *创建一个包含与 URL 相关方法的模型混入器* 的配方

+   *创建一个用于处理创建和修改日期的模型混入器* 的配方

+   *创建一个用于处理元标签的模型混入器* 的配方

+   在第四章 *模板和 JavaScript* 中实现 *实现 Like 小部件* 的配方

# 处理多语言字段

Django 使用国际化机制来翻译代码和模板中的冗长字符串。然而，开发者需要决定如何在模型中实现多语言内容。有几个第三方模块可以处理可翻译的模型字段；然而，我更喜欢在本配方中向您介绍的这个简单解决方案。

您将了解的方法的优点如下：

+   在数据库中定义多语言字段非常直接

+   在数据库查询中使用多语言字段非常简单

+   您可以使用贡献的行政功能编辑具有多语言字段的模型，而无需额外修改

+   如果需要，您可以在同一模板中轻松显示一个对象的全部翻译

+   您可以使用数据库迁移来添加或删除语言

## 准备工作

您已经创建了 `utils` 包吗？您现在需要为那里的自定义模型字段创建一个新的 `fields.py` 文件。

## 如何操作…

执行以下步骤以定义多语言字符字段和多语言文本字段：

1.  打开 `fields.py` 文件，并按照以下方式创建多语言字符字段：

    ```py
    # utils/fields.py
    # -*- coding: UTF-8 -*-
    from __future__ import unicode_literals
    from django.conf import settings
    from django.db import models
    from django.utils.translation import get_language
    from django.utils.translation import string_concat

    class MultilingualCharField(models.CharField):

      def __init__(self, verbose_name=None, **kwargs):

        self._blank = kwargs.get("blank", False)
        self._editable = kwargs.get("editable", True)

        super(MultilingualCharField, self).\
          __init__(verbose_name, **kwargs)

      def contribute_to_class(self, cls, name,
        virtual_only=False):
        # generate language specific fields dynamically
        if not cls._meta.abstract:
          for lang_code, lang_name in settings.LANGUAGES:
            if lang_code == settings.LANGUAGE_CODE:
              _blank = self._blank
            else:
              _blank = True

            localized_field = models.CharField(
              string_concat(self.verbose_name, 
                " (%s)" % lang_code),
                  name=self.name,
                    primary_key=self.primary_key,
                    max_length=self.max_length,
                    unique=self.unique,
                    blank=_blank,
                    null=False,
                    # we ignore the null argument!
                    db_index=self.db_index,
                    rel=self.rel,
                    default=self.default or "",
                    editable=self._editable,
                    serialize=self.serialize,
                    choices=self.choices,
                    help_text=self.help_text,
                    db_column=None,
                    db_tablespace=self.db_tablespace
            )
            localized_field.contribute_to_class(
              cls,
              "%s_%s" % (name, lang_code),
            )

            def translated_value(self):
              language = get_language()
              val = self.__dict__["%s_%s" % (name, language)]
              if not val:
                val = self.__dict__["%s_%s" % \
                  (name, settings.LANGUAGE_CODE)]
                  return val

          setattr(cls, name, property(translated_value))
    ```

1.  在同一文件中，添加一个类似的多语言文本字段。以下代码中突出显示了不同的部分：

    ```py
    class MultilingualTextField(models.TextField):

      def __init__(self, verbose_name=None, **kwargs):

        self._blank = kwargs.get("blank", False)
        self._editable = kwargs.get("editable", True)

        super(MultilingualTextField, self).\
          __init__(verbose_name, **kwargs)

        def contribute_to_class(self, cls, name, virtual_only=False):
          # generate language specific fields dynamically
          if not cls._meta.abstract:
            for lang_code, lang_name in settings.LANGUAGES:
              if lang_code == settings.LANGUAGE_CODE:
                _blank = self._blank
              else:
                _blank = True

                localized_field = models.TextField(
                  string_concat(self.verbose_name, 
                    " (%s)" % lang_code),
                  name=self.name,
                  primary_key=self.primary_key,
                  max_length=self.max_length,
                  unique=self.unique,
                  blank=_blank,
                  null=False,
                  # we ignore the null argument!
                  db_index=self.db_index,
                  rel=self.rel,
                  default=self.default or "",
                  editable=self._editable,
                  serialize=self.serialize,
                  choices=self.choices,
                  help_text=self.help_text,
                  db_column=None,
                  db_tablespace=self.db_tablespace
                )
                localized_field.contribute_to_class(
                  cls,
                    "%s_%s" % (name, lang_code),
                )

            def translated_value(self):
              language = get_language()
              val = self.__dict__["%s_%s" % (name, language)]
              if not val:
                val = self.__dict__["%s_%s" % \
                  (name, settings.LANGUAGE_CODE)]
                return val

            setattr(cls, name, property(translated_value))
    ```

现在，我们将考虑一个如何在您的应用程序中使用多语言字段的示例，如下所示：

1.  首先，在您的设置中设置多种语言：

    ```py
    # myproject/settings.py
    # -*- coding: UTF-8 -*-
    # …
    LANGUAGE_CODE = "en"

    LANGUAGES = (
        ("en", "English"),
        ("de", "Deutsch"),
        ("fr", "Français"),
        ("lt", "Lietuvi kalba"),
    )
    ```

1.  然后，按照以下方式为您的模型创建多语言字段：

    ```py
    # demo_app/models.py
    # -*- coding: UTF-8 -*-
    from __future__ import unicode_literals
    from django.db import models
    from django.utils.translation import ugettext_lazy as _
    from django.utils.encoding import \
        python_2_unicode_compatible

    from utils.fields import MultilingualCharField
    from utils.fields import MultilingualTextField

    @python_2_unicode_compatible
    class Idea(models.Model):
     title = MultilingualCharField(
     _("Title"),
     max_length=200,
     )
     description = MultilingualTextField(
     _("Description"),
     blank=True,
     )

        class Meta:
            verbose_name = _("Idea")
            verbose_name_plural = _("Ideas")

        def __str__(self):
            return self.title
    ```

## 它是如何工作的…

`Idea` 的示例将创建一个类似于以下模型的模型：

```py
class Idea(models.Model):
  title_en = models.CharField(
    _("Title (en)"),
    max_length=200,
  )
  title_de = models.CharField(
    _("Title (de)"),
    max_length=200,
    blank=True,
  )
  title_fr = models.CharField(
    _("Title (fr)"),
    max_length=200,
    blank=True,
  )
  title_lt = models.CharField(
    _("Title (lt)"),
    max_length=200,
    blank=True,
  )
  description_en = models.TextField(
    _("Description (en)"),
    blank=True,
  )
  description_de = models.TextField(
    _("Description (de)"),
    blank=True,
  )
  description_fr = models.TextField(
    _("Description (fr)"),
    blank=True,
  )
  description_lt = models.TextField(
    _("Description (lt)"),
    blank=True,
  )
```

此外，还将有两个属性：`title` 和 `description`，它们将返回当前活动语言中的标题和描述。

`MultilingualCharField`和`MultilingualTextField`字段会根据你的`LANGUAGES`设置动态处理模型字段。它们将覆盖 Django 框架创建模型类时使用的`contribute_to_class()`方法。多语言字段会为项目的每种语言动态添加字符或文本字段。此外，还会创建属性以返回当前活动语言或默认的主语言的翻译值。

例如，你可以在模板中有以下内容：

```py
<h1>{{ idea.title }}</h1>
<div>{{ idea.description|urlize|linebreaks }}</div>
```

这将根据当前选定的语言显示英文、德语、法语或立陶宛语文本。然而，如果翻译不存在，它将回退到英文。

这里是另一个例子。如果你想在视图中按翻译标题对`QuerySet`进行排序，你可以定义如下：

```py
qs = Idea.objects.order_by("title_%s" % request.LANGUAGE_CODE)
```

# 使用迁移

并非一旦创建了你的数据库结构，它就不会在未来改变。随着开发的迭代进行，你可以在开发过程中获取业务需求更新，并且你将需要在过程中执行数据库模式更改。使用 Django 迁移，你不需要手动更改数据库表和字段，因为大部分操作都是通过命令行界面自动完成的。

## 准备工作

在命令行工具中激活你的虚拟环境。

## 如何操作…

要创建数据库迁移，请查看以下步骤：

1.  当你在新的`demo_app`应用中创建模型时，你需要创建一个初始迁移，这将为你应用创建数据库表。这可以通过以下命令完成：

    ```py
    (myproject_env)$ python manage.py makemigrations demo_app

    ```

1.  第一次创建项目所有表时，运行以下命令：

    ```py
    (myproject_env)$ python manage.py migrate

    ```

    它执行所有没有数据库迁移的应用的常规数据库同步，并且除了这个之外，它还会迁移所有设置了迁移的应用。此外，当你想要执行所有应用的新的迁移时，也要运行此命令。

1.  如果你想要执行特定应用的迁移，运行以下命令：

    ```py
    (myproject_env)$ python manage.py migrate demo_app

    ```

1.  如果你修改了数据库模式，你必须为该模式创建一个迁移。例如，如果我们向`Idea`模型添加一个新的子标题字段，我们可以使用以下命令创建迁移：

    ```py
    (myproject_env)$ python manage.py makemigrations --name \
    subtitle_added demo_app

    ```

1.  要创建一个修改数据库表中数据的迁移，我们可以使用以下命令：

    ```py
    (myproject_env)$ python manage.py makemigrations --empty \ --name populate_subtitle demo_app

    ```

    这将创建一个数据迁移框架，你需要修改并添加数据操作到它之前应用。

1.  要列出所有可用已应用和未应用的迁移，运行以下命令：

    ```py
    (myproject_env)$ python manage.py migrate --list

    ```

    已应用的迁移将带有`[X]`前缀。

1.  要列出特定应用的可用迁移，运行以下命令：

    ```py
    (myproject_env)$ python manage.py migrate --list demo_app

    ```

## 它是如何工作的…

Django 迁移是数据库迁移机制的指令文件。指令文件告诉我们哪些数据库表需要创建或删除；哪些字段需要添加或删除；以及哪些数据需要插入、更新或删除。

Django 中有两种类型的迁移。一种是模式迁移，另一种是数据迁移。在添加新模型或添加或删除字段时应该创建模式迁移。当你想向数据库中填充一些值或从数据库中大量删除值时，应该使用数据迁移。数据迁移应该使用命令行工具中的命令创建，然后在迁移文件中编程。每个应用的迁移都保存在它们的`migrations`目录中。第一个迁移通常被称为`0001_initial.py`，我们示例应用中的其他迁移将被称为`0002_subtitle_added.py`和`0003_populate_subtitle.py`。每个迁移都有一个自动递增的数字前缀。对于每个执行的迁移，都会在`django_migrations`数据库表中保存一个条目。

可以通过指定我们想要迁移到的迁移编号来回迁移，如下所示：

```py
(myproject_env)$ python manage.py migrate demo_app 0002

```

如果你想要撤销特定应用的全部迁移，可以使用以下命令：

```py
(myproject_env)$ python manage.py migrate demo_app zero

```

### 小贴士

在测试了前向和反向迁移过程并且确定它们将在其他开发和公共网站环境中良好工作之前，不要将迁移提交到版本控制。

## 参见

+   第一章中的*使用 pip 处理项目依赖和将外部依赖包含在你的项目中*食谱，*Django 1.8 入门*

+   *将外键更改为多对多字段*食谱

# 从 South 迁移切换到 Django 迁移

如果你像我一样，自从 Django 核心功能中存在数据库迁移之前（即，在 Django 1.7 之前）就开始使用 Django，那么你很可能之前已经使用过第三方 South 迁移。在这个食谱中，你将学习如何将你的项目从 South 迁移切换到 Django 迁移。

## 准备工作

确保所有应用及其 South 迁移都是最新的。

## 如何操作…

执行以下步骤：

1.  将所有应用迁移到最新的 South 迁移，如下所示：

    ```py
    (myproject_env)$ python manage.py migrate

    ```

1.  在设置中将`south`从`INSTALLED_APPS`中移除。

1.  对于每个有 South 迁移的应用，删除迁移文件，只留下`migrations`目录。

1.  使用以下命令创建新的迁移文件：

    ```py
    (my_project)$ python manage.py makemigrations

    ```

1.  由于数据库模式已经正确设置，可以伪造初始 Django 迁移：

    ```py
    (my_project)$ python manage.py migrate --fake-initial

    ```

1.  如果你的应用中存在任何循环外键（即，不同应用中的两个模型通过外键或多对多关系相互指向），请分别对这些应用应用假初始迁移：

    ```py
    (my_project)$ python manage.py migrate --fake-initial demo_app

    ```

## 工作原理…

在切换到处理数据库模式更改的新方式时，数据库中没有冲突，因为 South 迁移历史保存在 `south_migrationhistory` 数据库表中；而 Django 迁移历史保存在 `django_migrations` 数据库表中。唯一的问题是具有不同语法的迁移文件，因此需要将 South 迁移完全替换为 Django 迁移。

因此，首先，我们删除 South 迁移文件。然后，`makemigrations` 命令识别空的 `migrations` 目录并为每个应用创建新的初始 Django 迁移。一旦这些迁移被伪造，就可以创建并应用进一步的 Django 迁移。

## 参见

+   *使用迁移* 配方

+   *将外键更改为多对多字段* 的配方

# 将外键更改为多对多字段

这个配方是一个实际示例，说明如何在保留已存在数据的情况下将多对一关系更改为多对多关系。我们将为此情况使用模式和数据迁移。

## 准备工作

假设您有一个 `Idea` 模型，其中有一个指向 `Category` 模型的外键，如下所示：

```py
# demo_app/models.py
# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
from django.db import models
from django.utils.translation import ugettext_lazy as _
from django.utils.encoding import python_2_unicode_compatible

@python_2_unicode_compatible
class Category(models.Model):
    title = models.CharField(_("Title"), max_length=200)

    def __str__(self):
        return self.title

@python_2_unicode_compatible
class Idea(models.Model):
    title = model.CharField(_("Title"), max_length=200)
 category = models.ForeignKey(Category,
 verbose_name=_("Category"), null=True, blank=True)

    def __str__(self):
        return self.title
```

应使用以下命令创建和执行初始迁移：

```py
(myproject_env)$ python manage.py makemigrations demo_app
(myproject_env)$ python manage.py migrate demo_app

```

## 如何做到这一点…

以下步骤将指导您如何从外键关系到多对多关系进行切换，同时保留已存在的数据：

1.  添加一个名为 `categories` 的新多对多字段，如下所示：

    ```py
    # demo_app/models.py
    @python_2_unicode_compatible
    class Idea(models.Model):
        title = model.CharField(_("Title"), max_length=200)
        category = models.ForeignKey(Category,
            verbose_name=_("Category"),
            null=True,
            blank=True,
        )
     categories = models.ManyToManyField(Category, 
     verbose_name=_("Categories"),
     blank=True, 
     related_name="ideas",
     )

    ```

1.  为了将新字段添加到数据库中，请创建并运行一个模式迁移，如下所示：

    ```py
    (myproject_env)$ python manage.py makemigrations demo_app \
    --name categories_added
    (myproject_env)$ python manage.py migrate demo_app

    ```

1.  创建一个数据迁移，将类别从外键复制到多对多字段，如下所示：

    ```py
    (myproject_env)$ python manage.py makemigrations --empty \
    --name copy_categories demo_app

    ```

1.  打开新创建的迁移文件 (`demo_app/migrations/0003_copy_categories.py`) 并定义正向迁移指令，如下所示：

    ```py
    # demo_app/migrations/0003_copy_categories.py
    # -*- coding: utf-8 -*-
    from __future__ import unicode_literals
    from django.db import models, migrations

    def copy_categories(apps, schema_editor):
        Idea = apps.get_model("demo_app", "Idea")
        for idea in Idea.objects.all():
            if idea.category:
                idea.categories.add(idea.category)

    class Migration(migrations.Migration):

        dependencies = [
            ('demo_app', '0002_categories_added'),
        ]

        operations = [
            migrations.RunPython(copy_categories),
        ]
    ```

1.  运行以下数据迁移：

    ```py
    (myproject_env)$ python manage.py migrate demo_app

    ```

1.  在 `models.py` 文件中删除外键字段 `category`：

    ```py
    # demo_app/models.py
    @python_2_unicode_compatible
    class Idea(models.Model):
        title = model.CharField(_("Title"), max_length=200)
        categories = models.ManyToManyField(Category,
            verbose_name=_("Categories"),
            blank=True,
            related_name="ideas",
        )
    ```

1.  创建并运行一个模式迁移，以从数据库表中删除 `categories` 字段，如下所示：

    ```py
    (myproject_env)$ python manage.py schemamigration \
    --name delete_category demo_app
    (myproject_env)$ python manage.py migrate demo_app

    ```

## 它是如何工作的…

首先，我们在 `Idea` 模型中添加一个新的多对多字段。然后，我们将现有关系从外键关系到多对多关系复制。最后，我们移除外键关系。

## 参见

+   *使用迁移* 配方

+   *从 South 迁移切换到 Django 迁移* 配方
