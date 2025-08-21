# 第五章：自定义模板过滤器和标签

在本章中，我们将涵盖以下配方：

+   遵循自己的模板过滤器和标签的约定

+   创建一个模板过滤器以显示自发布以来经过了多少天

+   创建一个模板过滤器来提取第一个媒体对象

+   创建一个模板过滤器以使 URL 更加人性化

+   创建一个模板标签以包含模板（如果存在）

+   创建一个模板标签以在模板中加载 QuerySet

+   创建一个模板标签以将内容解析为模板

+   创建模板标签以修改请求查询参数

# 介绍

Django 具有功能丰富的模板系统，包括模板继承、更改值表示的过滤器和用于表现逻辑的标签等功能。此外，Django 允许您向应用程序添加自定义模板过滤器和标签。自定义过滤器或标签应位于您的应用程序中的`templatetags` Python 包下的模板标签库文件中。然后可以使用`{% load %}`模板标签在任何模板中加载您的模板标签库。在本章中，我们将创建几个有用的过滤器和标签，以便更多地控制模板编辑者。 

# 技术要求

要使用本章的代码，您将需要最新稳定版本的 Python 3，MySQL 或 PostgreSQL 数据库，以及带有虚拟环境的 Django 项目。

您可以在 GitHub 存储库的`ch05`目录中找到本章的所有代码：[`github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition`](https://github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition)。

# 遵循自己的模板过滤器和标签的约定

如果没有遵循指南，自定义模板过滤器和标签可能会令人困惑和不一致。拥有方便灵活的模板过滤器和标签对于模板编辑者来说非常重要。在本篇中，我们将看一些增强 Django 模板系统功能时应该使用的约定：

1.  当页面的逻辑更适合于视图、上下文处理器或模型方法时，不要创建或使用自定义模板过滤器或标签。当您的内容是特定于上下文的，例如对象列表或对象详细视图时，在视图中加载对象。如果您需要在几乎每个页面上显示一些内容，请创建上下文处理器。当您需要获取与模板上下文无关的对象的一些属性时，请使用模型的自定义方法而不是模板过滤器。

1.  使用`_tags`后缀命名模板标签库。当您的模板标签库与您的应用程序命名不同时，您可以避免模糊的包导入问题。

1.  在新创建的库中，将过滤器与标签分开，例如使用注释，如下面的代码所示：

```py
# myproject/apps/core/templatetags/utility_tags.py from django import template 

register = template.Library()

""" TAGS """

# Your tags go here…

""" FILTERS """

# Your filters go here…
```

1.  在创建高级自定义模板标签时，确保其语法易于记忆，包括以下可以跟随标签名称的构造：

+   `for [app_name.model_name]`：包括此构造以使用特定模型。

+   `using [template_name]`：包括此构造以使用模板作为模板标签的输出。

+   `limit [count]`：包括此构造以将结果限制为特定数量。

+   `as [context_variable]`：包括此构造以将结果存储在可以多次重用的上下文变量中。

1.  尽量避免在模板标签中定义多个按位置定义的值，除非它们是不言自明的。否则，这可能会使模板开发人员感到困惑。

1.  尽可能使可解析的参数多。没有引号的字符串应被视为需要解析的上下文变量，或者作为提醒模板标签组件结构的简短单词。

# 创建一个模板过滤器以显示自发布以来经过了多少天

在谈论创建或修改日期时，方便阅读更加人性化的时间差异，例如，博客条目是 3 天前发布的，新闻文章是今天发布的，用户上次登录是昨天。在这个示例中，我们将创建一个名为`date_since`的模板过滤器，它将根据天、周、月或年将日期转换为人性化的时间差异。

# 准备工作

如果尚未完成，请创建`core`应用程序，并将其放置在设置中的`INSTALLED_APPS`中。然后，在此应用程序中创建一个`templatetags` Python 包（Python 包是带有空的`__init__.py`文件的目录）。

# 如何做...

创建一个`utility_tags.py`文件，其中包含以下内容：

```py
# myproject/apps/core/templatetags/utility_tags.py from datetime import datetime
from django import template
from django.utils import timezone
from django.utils.translation import ugettext_lazy as _

register = template.Library()

""" FILTERS """

DAYS_PER_YEAR = 365
DAYS_PER_MONTH = 30
DAYS_PER_WEEK = 7

@register.filter(is_safe=True)
def date_since(specific_date):
    """
    Returns a human-friendly difference between today and past_date
    (adapted from https://www.djangosnippets.org/snippets/116/)
    """
    today = timezone.now().date()
    if isinstance(specific_date, datetime):
        specific_date = specific_date.date()
    diff = today - specific_date
    diff_years = int(diff.days / DAYS_PER_YEAR)
    diff_months = int(diff.days / DAYS_PER_MONTH)
    diff_weeks = int(diff.days / DAYS_PER_WEEK)
    diff_map = [
        ("year", "years", diff_years,),
        ("month", "months", diff_months,),
        ("week", "weeks", diff_weeks,),
        ("day", "days", diff.days,),
    ]
    for parts in diff_map:
        (interval, intervals, count,) = parts
        if count > 1:
            return _(f"{count} {intervals} ago")
        elif count == 1:
            return _("yesterday") \
                if interval == "day" \
                else _(f"last {interval}")
    if diff.days == 0:
        return _("today")
    else:
        # Date is in the future; return formatted date.
        return f"{specific_date:%B %d, %Y}"

```

# 它是如何工作的...

在模板中使用此过滤器，如下所示的代码将呈现类似于昨天、上周或 5 个月前的内容：

```py
{% load utility_tags %}
{{ object.published|date_since }}
```

您可以将此过滤器应用于`date`和`datetime`类型的值。

每个模板标签库都有一个`template.Library`类型的注册表，其中收集了过滤器和标签。 Django 过滤器是由`@register.filter`装饰器注册的函数。在这种情况下，我们传递了`is_safe=True`参数，以指示我们的过滤器不会引入任何不安全的 HTML 标记。

默认情况下，模板系统中的过滤器将与函数或其他可调用对象的名称相同。如果需要，可以通过将名称传递给装饰器来为过滤器设置不同的名称，如下所示：

```py
@register.filter(name="humanized_date_since", is_safe=True)
def date_since(value):
    # …
```

过滤器本身相当不言自明。首先读取当前日期。如果过滤器的给定值是`datetime`类型，则提取其`date`。然后，根据`DAYS_PER_YEAR`、`DAYS_PER_MONTH`、`DAYS_PER_WEEK`或天数间隔计算今天和提取值之间的差异。根据计数，返回不同的字符串结果，如果值在未来，则返回格式化日期。

# 还有更多...

如果需要，我们也可以覆盖其他时间段，例如 20 分钟前、5 小时前，甚至是 10 年前。为此，我们将在现有的`diff_map`集合中添加更多的间隔，并且为了显示时间差异，我们需要对`datetime`值进行操作，而不是`date`值。

# 另请参阅

+   提取第一个媒体对象的模板过滤器的方法

+   创建一个模板过滤器以使 URL 更加人性化的方法

# 创建一个模板过滤器来提取第一个媒体对象

想象一下，您正在开发一个博客概述页面，对于每篇文章，您希望从内容中显示图像、音乐或视频，这些内容来自内容。在这种情况下，您需要从帖子模型的字段中存储的 HTML 内容中提取`<figure>`、`<img>`、`<object>`、`<embed>`、`<video>`、`<audio>`和`<iframe>`标签。在这个示例中，我们将看到如何使用`first_media`过滤器来执行此操作。

# 准备工作

我们将从`core`应用程序开始，在设置中应设置为`INSTALLED_APPS`，并且应该包含此应用程序中的`templatetags`包。

# 如何做...

在`utility_tags.py`文件中，添加以下内容：

```py
# myproject/apps/core/templatetags/utility_tags.py import re
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

""" FILTERS """

MEDIA_CLOSED_TAGS = "|".join([
    "figure", "object", "video", "audio", "iframe"])
MEDIA_SINGLE_TAGS = "|".join(["img", "embed"])
MEDIA_TAGS_REGEX = re.compile(
    r"<(?P<tag>" + MEDIA_CLOSED_TAGS + ")[\S\s]+?</(?P=tag)>|" +
    r"<(" + MEDIA_SINGLE_TAGS + ")[^>]+>",
    re.MULTILINE)

@register.filter
def first_media(content):
    """
    Returns the chunk of media-related markup from the html content
    """
    tag_match = MEDIA_TAGS_REGEX.search(content)
    media_tag = ""
    if tag_match:
        media_tag = tag_match.group()
    return mark_safe(media_tag)
```

# 它是如何工作的...

如果数据库中的 HTML 内容有效，并且将以下代码放入模板中，则将从对象的内容字段中检索媒体标签；否则，如果未找到媒体，则将返回空字符串：

```py
{% load utility_tags %}
{{ object.content|first_media }} 
```

正则表达式是搜索或替换文本模式的强大功能。首先，我们定义了所有支持的媒体标签名称的列表，将它们分成具有开放和关闭标签（`MEDIA_CLOSED_TAGS`）和自关闭标签（`MEDIA_SINGLE_TAGS`）的组。从这些列表中，我们生成了编译后的正则表达式`MEDIA_TAGS_REGEX`。在这种情况下，我们搜索所有可能的媒体标签，允许它们跨越多行出现。

让我们看看这个正则表达式是如何工作的，如下所示：

+   交替模式由管道（`|`）符号分隔。

+   模式中有两组——首先是那些具有开放和关闭普通标签（`<figure>`，`<object>`，`<video>`，`<audio>`，`<iframe>`和`<picture>`）的标签，然后是最后一个模式，用于所谓的自关闭

或空标签（`<img>`和`<embed>`）。

+   对于可能是多行的普通标签，我们将使用`[\S\s]+?`模式，该模式至少匹配任何符号一次；但是，我们尽可能少地执行这个操作，直到找到它后面的字符串。

+   因此，`<figure[\S\s]+?</figure>`搜索`<figure>`标签的开始以及它后面的所有内容，直到找到`</figure>`标签的闭合。

+   类似地，对于自关闭标签的`[^>]+`模式，我们搜索除右尖括号（可能更为人所知的是大于号符号，即`>`）之外的任何符号，至少一次，尽可能多次，直到遇到指示标签关闭的尖括号。

`re.MULTILINE`标志确保可以找到匹配项，即使它们跨越内容中的多行。然后，在过滤器中，我们使用这个正则表达式模式进行搜索。默认情况下，在 Django 中，任何过滤器的结果都会显示为`<`，`>`和`&`符号转义为`&lt;`，`&gt;`和`&amp;`实体。然而，在这种情况下，我们使用`mark_safe()`函数来指示结果是安全的并且已准备好用于 HTML，以便任何内容都将被呈现而不进行转义。因为原始内容是用户输入，所以我们这样做，而不是在注册过滤器时传递`is_safe=True`，因为我们需要明确证明标记是安全的。

# 还有更多...

如果您对正则表达式感兴趣，可以在官方 Python 文档中了解更多信息[`docs.python.org/3/library/re.html`](https://docs.python.org/3/library/re.html)。

# 另请参阅

+   *创建一个模板过滤器以显示发布后经过多少天*食谱

+   *创建一个模板过滤器以使 URL 更加人性化*食谱

# 创建一个模板过滤器以使 URL 更加人性化

Web 用户通常在地址字段中以不带协议（`http://`）或斜杠（`/`）的方式识别 URL，并且以类似的方式输入 URL。在这个食谱中，我们将创建一个`humanize_url`过滤器，用于以更短的格式向用户呈现 URL，截断非常长的地址，类似于 Twitter 在推文中对链接所做的操作。

# 准备工作

与之前的食谱类似，我们将从`core`应用程序开始，在设置中应该设置`INSTALLED_APPS`，其中包含应用程序中的`templatetags`包。

# 如何做...

在`core`应用程序的`utility_tags.py`模板库的`FILTERS`部分中，让我们添加`humanize_url`过滤器并注册它，如下所示：

```py
# myproject/apps/core/templatetags/utility_tags.py import re
from django import template

register = template.Library()

""" FILTERS """

@register.filter
def humanize_url(url, letter_count=40):
    """
    Returns a shortened human-readable URL
    """
    letter_count = int(letter_count)
    re_start = re.compile(r"^https?://")
    re_end = re.compile(r"/$")
    url = re_end.sub("", re_start.sub("", url))
    if len(url) > letter_count:
        url = f"{url[:letter_count - 1]}…"
    return url
```

# 工作原理...

我们可以在任何模板中使用`humanize_url`过滤器，如下所示：

```py
{% load utility_tags %}
<a href="{{ object.website }}" target="_blank">
    {{ object.website|humanize_url }}
</a>
<a href="{{ object.website }}" target="_blank">
    {{ object.website|humanize_url:30 }}
</a>
```

该过滤器使用正则表达式来删除前导协议和尾部斜杠，将 URL 缩短到给定的字母数量（默认为 40），并在截断后添加省略号，如果完整的 URL 不符合指定的字母数量。例如，对于`https://docs.djangoproject.com/en/3.0/howto/custom-template-tags/`的 URL，40 个字符的人性化版本将是`docs.djangoproject.com/en/3.0/howto/cus…`。

# 另请参阅

+   *创建一个模板过滤器以显示发布后经过多少天*食谱

+   *创建一个模板过滤器以提取第一个媒体对象*食谱

+   *创建一个模板标签以包含模板（如果存在）*食谱

# 创建一个模板标签以包含模板（如果存在）

Django 提供了`{% include %}`模板标签，允许一个模板呈现和包含另一个模板。但是，如果您尝试包含文件系统中不存在的模板，则此模板标签会引发错误。在此食谱中，我们将创建一个`{% try_to_include %}`模板标签，如果存在，则包含另一个模板，并通过渲染为空字符串来静默失败。

# 准备工作

我们将从已安装并准备好自定义模板标签的`core`应用程序开始。

# 如何做...

执行以下步骤创建`{% try_to_include %}`模板标签：

1.  首先，让我们创建解析模板标签参数的函数，如下所示：

```py
# myproject/apps/core/templatetags/utility_tags.py from django import template
from django.template.loader import get_template

register = template.Library()

""" TAGS """

@register.tag
def try_to_include(parser, token):
    """
    Usage: {% try_to_include "some_template.html" %}

    This will fail silently if the template doesn't exist.
    If it does exist, it will be rendered with the current context.
    """
    try:
        tag_name, template_name = token.split_contents()
    except ValueError:
        tag_name = token.contents.split()[0]
        raise template.TemplateSyntaxError(
            f"{tag_name} tag requires a single argument")
    return IncludeNode(template_name)
```

1.  然后，我们需要在同一文件中创建一个自定义的`IncludeNode`类，该类从基本的`template.Node`扩展。让我们在`try_to_include()`函数之前插入它，如下所示：

```py
class IncludeNode(template.Node):
    def __init__(self, template_name):
        self.template_name = template.Variable(template_name)

    def render(self, context):
        try:
            # Loading the template and rendering it
            included_template = self.template_name.resolve(context)
            if isinstance(included_template, str):
                included_template = get_template(included_template)
            rendered_template = included_template.render(
                context.flatten()
            )
        except (template.TemplateDoesNotExist,
                template.VariableDoesNotExist,
                AttributeError):
            rendered_template = ""
        return rendered_template

@register.tag
def try_to_include(parser, token):
    # …
```

# 它是如何工作的...

高级自定义模板标签由两部分组成：

+   解析模板标签参数的函数

+   负责模板标签逻辑和输出的`Node`类

`{% try_to_include %}`模板标签期望一个参数——即`template_name`。因此，在`try_to_include()`函数中，我们尝试将令牌的拆分内容仅分配给`tag_name`变量（即`try_to_include`）和`template_name`变量。如果这不起作用，将引发`TemplateSyntaxError`。该函数返回`IncludeNode`对象，该对象获取`template_name`字段并将其存储在模板`Variable`对象中以供以后使用。

在`IncludeNode`的`render()`方法中，我们解析`template_name`变量。如果上下文变量被传递给模板标签，则其值将在此处用于`template_name`。如果引用的字符串被传递给模板标签，那么引号内的内容将用于`included_template`，而与上下文变量对应的字符串将被解析为其相应的字符串等效。

最后，我们将尝试加载模板，使用解析的`included_template`字符串，并在当前模板上下文中呈现它。如果这不起作用，则返回空字符串。

至少有两种情况可以使用此模板标签：

+   在包含路径在模型中定义的模板时，如下所示：

```py
{% load utility_tags %}
{% try_to_include object.template_path %}
```

+   在模板上下文变量的范围中的某个地方使用`{% with %}`模板标签定义路径的模板。当您需要为 Django CMS 中模板的占位符创建自定义布局时，这是非常有用的：

```py
{# templates/cms/start_page.html #} {% load cms_tags %}
{% with editorial_content_template_path=
"cms/plugins/editorial_content/start_page.html" %}
    {% placeholder "main_content" %}
{% endwith %}
```

稍后，占位符可以使用`editorial_content`插件填充，然后读取`editorial_content_template_path`上下文变量，如果可用，则可以安全地包含模板：

```py
{# templates/cms/plugins/editorial_content.html #}
{% load utility_tags %}
{% if editorial_content_template_path %}
    {% try_to_include editorial_content_template_path %}
{% else %}
    <div>
        <!-- Some default presentation of
        editorial content plugin -->
    </div>
{% endif %}
```

# 还有更多...

您可以在任何组合中使用`{% try_to_include %}`标签和默认的`{% include %}`标签来包含扩展其他模板的模板。这对于大型网络平台非常有益，其中您有不同类型的列表，其中复杂的项目与小部件具有相同的结构，但具有不同的数据来源。

例如，在艺术家列表模板中，您可以包含`artist_item`模板，如下所示：

```py
{% load utility_tags %}
{% for object in object_list %}
    {% try_to_include "artists/includes/artist_item.html" %}
{% endfor %}
```

此模板将从项目基础扩展，如下所示：

```py
{# templates/artists/includes/artist_item.html #} {% extends "utils/includes/item_base.html" %}
{% block item_title %}
    {{ object.first_name }} {{ object.last_name }}
{% endblock %}
```

项目基础定义了任何项目的标记，并包括`Like`小部件，如下所示：

```py
{# templates/utils/includes/item_base.html #} {% load likes_tags %}
<h3>{% block item_title %}{% endblock %}</h3>
{% if request.user.is_authenticated %}
    {% like_widget for object %}
{% endif %}
```

# 另请参阅

+   *在第四章*中实现`Like`小部件的食谱，模板和 JavaScript

+   *创建一个模板标签以在模板中加载 QuerySet*食谱

+   *创建一个将内容解析为模板的模板标签*食谱

+   *创建模板标签以修改请求查询参数*食谱

# 创建一个模板标签以在模板中加载 QuerySet

通常，应在视图中定义应显示在网页上的内容。如果要在每个页面上显示内容，逻辑上应创建上下文处理器以使其全局可用。另一种情况是当您需要在某些页面上显示其他内容，例如最新新闻或随机引用，例如起始页面或对象的详细页面。在这种情况下，您可以使用自定义 `{% load_objects %}` 模板标签加载必要的内容，我们将在本教程中实现。

# 准备工作

我们将再次从 `core` 应用程序开始，该应用程序应已安装并准备好用于自定义模板标签。

此外，为了说明这个概念，让我们创建一个带有 `Article` 模型的 `news` 应用程序，如下所示：

```py
# myproject/apps/news/models.py from django.db import models
from django.urls import reverse
from django.utils.translation import ugettext_lazy as _

from myproject.apps.core.models import CreationModificationDateBase, UrlBase

class ArticleManager(models.Manager):
 def random_published(self):
 return self.filter(
 publishing_status=self.model.PUBLISHING_STATUS_PUBLISHED,
 ).order_by("?")

class Article(CreationModificationDateBase, UrlBase):
    PUBLISHING_STATUS_DRAFT, PUBLISHING_STATUS_PUBLISHED = "d", "p"
    PUBLISHING_STATUS_CHOICES = (
        (PUBLISHING_STATUS_DRAFT, _("Draft")),
        (PUBLISHING_STATUS_PUBLISHED, _("Published")),
    )
    title = models.CharField(_("Title"), max_length=200)
    slug = models.SlugField(_("Slug"), max_length=200)
    content = models.TextField(_("Content"))
    publishing_status = models.CharField(
        _("Publishing status"),
        max_length=1,
        choices=PUBLISHING_STATUS_CHOICES,
        default=PUBLISHING_STATUS_DRAFT,
    )

 custom_manager = ArticleManager()

    class Meta:
        verbose_name = _("Article")
        verbose_name_plural = _("Articles")

    def __str__(self):
        return self.title

    def get_url_path(self):
        return reverse("news:article_detail", kwargs={"slug": self.slug})
```

在这里，有趣的部分是 `Article` 模型的 `custom_manager`。该管理器可用于列出随机发布的文章。

使用上一章的示例，您可以完成具有 URL 配置、视图、模板和管理设置的应用程序。然后，使用管理表单向数据库添加一些文章。

# 如何做...

高级自定义模板标签由解析传递给标签的参数的函数和呈现标签输出或修改模板上下文的 `Node` 类组成。执行以下步骤创建 `{% load_objects %}` 模板标签：

1.  首先，让我们创建处理模板标签参数解析的函数，如下所示：

```py
# myproject/apps/core/templatetags/utility_tags.py from django import template
from django.apps import apps

register = template.Library()

""" TAGS """

@register.tag
def load_objects(parser, token):
    """
    Gets a queryset of objects of the model specified by app and
    model names

    Usage:
        {% load_objects [<manager>.]<method>
                        from <app_name>.<model_name>
                        [limit <amount>]
                        as <var_name> %}

    Examples:
        {% load_objects latest_published from people.Person
                        limit 3 as people %}
        {% load_objects site_objects.all from news.Article
                        as articles %}
        {% load_objects site_objects.all from news.Article
                        limit 3 as articles %}
    """
    limit_count = None
    try:
        (tag_name, manager_method,
         str_from, app_model,
         str_limit, limit_count,
         str_as, var_name) = token.split_contents()
    except ValueError:
        try:
            (tag_name, manager_method,
             str_from, app_model,
             str_as, var_name) = token.split_contents()
        except ValueError:
            tag_name = token.contents.split()[0]
            raise template.TemplateSyntaxError(
                f"{tag_name} tag requires the following syntax: "
                f"{{% {tag_name} [<manager>.]<method> from "
                "<app_name>.<model_name> [limit <amount>] "
                "as <var_name> %}")
    try:
        app_name, model_name = app_model.split(".")
    except ValueError:
        raise template.TemplateSyntaxError(
            "load_objects tag requires application name "
            "and model name, separated by a dot")
    model = apps.get_model(app_name, model_name)
    return ObjectsNode(
        model, manager_method, limit_count, var_name
    )
```

1.  然后，我们将在同一文件中创建自定义 `ObjectsNode` 类，扩展自 `template.Node` 基类。让我们在 `load_objects()` 函数之前插入它，如下面的代码所示：

```py
class ObjectsNode(template.Node):
    def __init__(self, model, manager_method, limit, var_name):
        self.model = model
        self.manager_method = manager_method
        self.limit = template.Variable(limit) if limit else None
        self.var_name = var_name

    def render(self, context):
        if "." in self.manager_method:
            manager, method = self.manager_method.split(".")
        else:
            manager = "_default_manager"
            method = self.manager_method

        model_manager = getattr(self.model, manager)
        fallback_method = self.model._default_manager.none
        qs = getattr(model_manager, method, fallback_method)()
        limit = None
        if self.limit:
            try:
                limit = self.limit.resolve(context)
            except template.VariableDoesNotExist:
                limit = None
        context[self.var_name] = qs[:limit] if limit else qs
        return ""

@register.tag
def load_objects(parser, token):
    # …
```

# 它是如何工作的...

`{% load_objects %}` 模板标签加载由管理器方法定义的指定应用程序和模型的 QuerySet，将结果限制为指定的计数，并将结果保存到给定的上下文变量中。

以下代码是如何使用我们刚刚创建的模板标签的简单示例。它将在任何模板中加载所有新闻文章，使用以下代码片段：

```py
{% load utility_tags %}
{% load_objects all from news.Article as all_articles %}
<ul>
    {% for article in all_articles %}
        <li><a href="{{ article.get_url_path }}">
         {{ article.title }}</a></li>
    {% endfor %}
</ul>
```

这是使用 `Article` 模型的默认 `objects` 管理器的 `all()` 方法，并且它将按照模型的 `Meta` 类中定义的 `ordering` 属性对文章进行排序。

接下来是一个示例，使用自定义管理器和自定义方法从数据库中查询对象。管理器是为模型提供数据库查询操作的接口。

每个模型至少有一个默认的名为 `objects` 的管理器。对于我们的 `Article` 模型，我们添加了一个名为 `custom_manager` 的额外管理器，其中包含一个名为 `random_published()` 的方法。以下是我们如何在 `{% load_objects %}` 模板标签中使用它来加载一个随机发布的文章：

```py
{% load utility_tags %}
{% load_objects custom_manager.random_published from news.Article limit 1 as random_published_articles %}
<ul>
    {% for article in random_published_articles %}
        <li><a href="{{ article.get_url_path }}">
         {{ article.title }}</a></li>
    {% endfor %}
</ul>
```

让我们来看一下 `{% load_objects %}` 模板标签的代码。在解析函数中，标签有两种允许的形式——带有或不带有 `limit`。字符串被解析，如果识别格式，则模板标签的组件将传递给 `ObjectsNode` 类。

在 `Node` 类的 `render()` 方法中，我们检查管理器的名称及其方法的名称。如果未指定管理器，则将使用 `_default_manager`。这是 Django 注入的任何模型的自动属性，并指向第一个可用的 `models.Manager()` 实例。在大多数情况下，`_default_manager` 将是 `objects` 管理器。之后，我们将调用管理器的方法，并在方法不存在时回退到空的 QuerySet。如果定义了 `limit`，我们将解析其值并相应地限制 QuerySet。最后，我们将将结果的 QuerySet 存储在上下文变量中，如 `var_name` 所给出的那样。

# 另请参阅

+   *在 Chapter 2*，模型和数据库结构中创建一个带有 URL 相关方法的模型混合的食谱

+   在 Chapter 2*，Models and Database Structure*中的*创建模型混合以处理创建和修改日期*配方

+   在 Chapter 2*，Models and Database Structure*中的*创建一个模板标签以包含模板（如果存在）*配方

+   在 Chapter 2*，Models and Database Structure*中的*创建一个模板标签以将内容解析为模板*配方

+   创建模板标签以修改请求查询参数的配方

# 创建一个模板标签以将内容解析为模板

在这个配方中，我们将创建`{% parse %}`模板标签，它将允许您将模板片段放入数据库。当您想要为经过身份验证和未经身份验证的用户提供不同的内容，当您想要包含个性化的称谓，或者当您不想在数据库中硬编码媒体路径时，这将非常有价值。

# 准备工作

像往常一样，我们将从`core`应用程序开始，该应用程序应该已经安装并准备好用于自定义模板标签。

# 如何做...

高级自定义模板标签由一个解析传递给标签的参数的函数和一个`Node`类组成，该类渲染标签的输出或修改模板上下文。执行以下步骤来创建`{% parse %}`模板标签：

1.  首先，让我们创建解析模板标签参数的函数，如下所示：

```py
# myproject/apps/core/templatetags/utility_tags.py
from django import template

register = template.Library()

""" TAGS """

@register.tag
def parse(parser, token):
    """
    Parses a value as a template and prints or saves to a variable

    Usage:
        {% parse <template_value> [as <variable>] %}

    Examples:
        {% parse object.description %}
        {% parse header as header %}
        {% parse "{{ MEDIA_URL }}js/" as js_url %}
    """
    bits = token.split_contents()
    tag_name = bits.pop(0)
    try:
        template_value = bits.pop(0)
        var_name = None
        if len(bits) >= 2:
            str_as, var_name = bits[:2]
    except ValueError:
        raise template.TemplateSyntaxError(
            f"{tag_name} tag requires the following syntax: "
            f"{{% {tag_name} <template_value> [as <variable>] %}}")
    return ParseNode(template_value, var_name)
```

1.  然后，我们将在同一文件中创建自定义的`ParseNode`类，该类从基本的`template.Node`扩展，如下面的代码所示（将其放在`parse()`函数之前）：

```py
class ParseNode(template.Node):
    def __init__(self, template_value, var_name):
        self.template_value = template.Variable(template_value)
        self.var_name = var_name

    def render(self, context):
        template_value = self.template_value.resolve(context)
        t = template.Template(template_value)
        context_vars = {}
        for d in list(context):
            for var, val in d.items():
                context_vars[var] = val
        req_context = template.RequestContext(
            context["request"], context_vars
        )
        result = t.render(req_context)
        if self.var_name:
            context[self.var_name] = result
            result = ""
        return result

@register.tag
def parse(parser, token):
    # …
```

# 它是如何工作的...

`{% parse %}`模板标签允许您将值解析为模板并立即渲染它，或将其存储在上下文变量中。

如果我们有一个带有描述字段的对象，该字段可以包含模板变量或逻辑，我们可以使用以下代码解析和渲染它：

```py
{% load utility_tags %}
{% parse object.description %}
```

还可以使用引号字符串定义要解析的值，如下面的代码所示：

```py
{% load static utility_tags %}
{% get_static_prefix as STATIC_URL %}
{% parse "{{ STATIC_URL }}site/img/" as image_directory %}
<img src="img/{{ image_directory }}logo.svg" alt="Logo" />
```

让我们来看一下`{% parse %}`模板标签的代码。解析函数逐位检查模板标签的参数。首先，我们期望解析名称和模板值。如果仍然有更多的位于令牌中，我们期望可选的`as`单词后跟上上下文变量名的组合。模板值和可选变量名被传递给`ParseNode`类。

该类的`render()`方法首先解析模板变量的值，并将其创建为模板对象。然后复制`context_vars`并生成请求上下文，模板进行渲染。如果定义了变量名，则将结果存储在其中并渲染一个空字符串；否则，立即显示渲染的模板。

# 另请参阅

+   在 Chapter 2*，Models and Database Structure*中的*创建一个模板标签以包含模板（如果存在）*配方

+   在模板中加载查询集的*创建模板标签*配方

+   在*创建模板标签以修改请求查询参数*配方中

# 创建模板标签以修改请求查询参数

Django 有一个方便灵活的系统，可以通过向 URL 配置文件添加正则表达式规则来创建规范和干净的 URL。然而，缺乏内置技术来管理查询参数。诸如搜索或可过滤对象列表的视图需要接受查询参数，以通过另一个参数深入筛选结果或转到另一页。在这个配方中，我们将创建`{% modify_query %}`、`{% add_to_query %}`和`{% remove_from_query %}`模板标签，让您可以添加、更改或删除当前查询的参数。

# 准备工作

再次，我们从`core`应用程序开始，该应用程序应该在`INSTALLED_APPS`中设置，其中包含`templatetags`包。

还要确保在`OPTIONS`下的`TEMPLATES`设置中将`request`上下文处理器添加到`context_processors`列表中。

```py
# myproject/settings/_base.py
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "myproject", "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
 "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "django.template.context_processors.media",
                "django.template.context_processors.static",
                "myproject.apps.core.context_processors.website_url",
            ]
        },
    }
]
```

# 如何做...

对于这些模板标签，我们将使用`@simple_tag`装饰器来解析组件，并要求您只需定义呈现函数，如下所示：

1.  首先，让我们添加一个辅助方法来组合每个标签输出的查询字符串：

```py
# myproject/apps/core/templatetags/utility_tags.py from urllib.parse import urlencode

from django import template
from django.utils.encoding import force_str
from django.utils.safestring import mark_safe

register = template.Library()

""" TAGS """

def construct_query_string(context, query_params):
    # empty values will be removed
    query_string = context["request"].path
    if len(query_params):
        encoded_params = urlencode([
            (key, force_str(value))
            for (key, value) in query_params if value
        ]).replace("&", "&amp;")
        query_string += f"?{encoded_params}"
    return mark_safe(query_string)
```

1.  然后，我们将创建`{% modify_query %}`模板标签：

```py
@register.simple_tag(takes_context=True)
def modify_query(context, *params_to_remove, **params_to_change):
    """Renders a link with modified current query parameters"""
    query_params = []
    for key, value_list in context["request"].GET.lists():
        if not key in params_to_remove:
            # don't add key-value pairs for params_to_remove
            if key in params_to_change:
                # update values for keys in params_to_change
                query_params.append((key, params_to_change[key]))
                params_to_change.pop(key)
            else:
                # leave existing parameters as they were
                # if not mentioned in the params_to_change
                for value in value_list:
                    query_params.append((key, value))
                    # attach new params
    for key, value in params_to_change.items():
        query_params.append((key, value))
    return construct_query_string(context, query_params)
```

1.  接下来，让我们创建`{% add_to_query %}`模板标签：

```py
@register.simple_tag(takes_context=True)
def add_to_query(context, *params_to_remove, **params_to_add):
    """Renders a link with modified current query parameters"""
    query_params = []
    # go through current query params..
    for key, value_list in context["request"].GET.lists():
        if key not in params_to_remove:
            # don't add key-value pairs which already
            # exist in the query
            if (key in params_to_add
                    and params_to_add[key] in value_list):
                params_to_add.pop(key)
            for value in value_list:
                query_params.append((key, value))
    # add the rest key-value pairs
    for key, value in params_to_add.items():
        query_params.append((key, value))
    return construct_query_string(context, query_params)
```

1.  最后，让我们创建`{% remove_from_query %}`模板标签：

```py
@register.simple_tag(takes_context=True)
def remove_from_query(context, *args, **kwargs):
    """Renders a link with modified current query parameters"""
    query_params = []
    # go through current query params..
    for key, value_list in context["request"].GET.lists():
        # skip keys mentioned in the args
        if key not in args:
            for value in value_list:
                # skip key-value pairs mentioned in kwargs
                if not (key in kwargs and
                        str(value) == str(kwargs[key])):
                    query_params.append((key, value))
    return construct_query_string(context, query_params)
```

# 工作原理...

所有三个创建的模板标签的行为都类似。首先，它们从`request.GET`字典样的`QueryDict`对象中读取当前查询参数，然后将其转换为新的（键，值）`query_params`元组列表。然后，根据位置参数和关键字参数更新值。最后，通过首先定义的辅助方法形成新的查询字符串。在此过程中，所有空格和特殊字符都被 URL 编码，并且连接查询参数的和号被转义。将此新的查询字符串返回到模板。

要了解有关`QueryDict`对象的更多信息，请参阅官方 Django 文档

在[`docs.djangoproject.com/en/3.0/ref/request-response/#querydict-objects`](https://docs.djangoproject.com/en/3.0/ref/request-response/#querydict-objects)。

让我们看一个示例，演示了`{% modify_query %}`模板标签的用法。模板标签中的位置参数定义要删除哪些查询参数，关键字参数定义要在当前查询中更新哪些查询参数。如果当前 URL 是`http://127.0.0.1:8000/artists/?category=fine-art&page=5`，我们可以使用以下模板标签呈现一个转到下一页的链接：

```py
{% load utility_tags %}
<a href="{% modify_query page=6 %}">6</a>
```

使用前述模板标签呈现的输出如下代码段所示：

```py
<a href="/artists/?category=fine-art&amp;page=6">6</a>
```

我们还可以使用以下示例来呈现一个重置分页并转到另一个类别`sculpture`的链接，如下所示：

```py
{% load utility_tags %}
<a href="{% modify_query "page" category="sculpture" %}">
    Sculpture
</a>
```

因此，使用前述模板标签呈现的输出将如下代码段所示：

```py
<a href="/artists/?category=sculpture">
    Sculpture
</a>
```

使用`{% add_to_query %}`模板标签，您可以逐步添加具有相同名称的参数。例如，如果当前 URL 是`http://127.0.0.1:8000/artists/?category=fine-art`，您可以使用以下代码段添加另一个类别`Sculpture`：

```py
{% load utility_tags %}
<a href="{% add_to_query category="sculpture" %}">
    + Sculpture
</a> 
```

这将在模板中呈现，如下代码段所示：

```py
<a href="/artists/?category=fine-art&amp;category=sculpture">
    + Sculpture
</a>
```

最后，借助`{% remove_from_query %}`模板标签的帮助，您可以逐步删除具有相同名称的参数。例如，如果当前 URL 是`http://127.0.0.1:8000/artists/?category=fine-art&category=sculpture`，您可以使用以下代码段删除`Sculpture`类别：

```py
{% load utility_tags %}
<a href="{% remove_from_query category="sculpture" %}">
    - Sculpture
</a>
```

这将在模板中呈现如下：

```py
<a href="/artists/?category=fine-art">
    - Sculpture
</a>
```

# 另请参阅

+   第三章*中的*对象列表过滤器*配方，表单和视图

+   *创建一个模板标签来包含模板（如果存在）*配方

+   *创建一个模板标签来在模板中加载 QuerySet*配方

+   *创建一个模板标签来解析内容作为模板*配方
