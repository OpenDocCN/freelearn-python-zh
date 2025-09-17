# 第五章：自定义模板过滤器和标签

在本章中，我们将涵盖以下主题：

+   遵循你自己的模板过滤器或标签的约定

+   创建一个模板过滤器以显示自文章发布以来过去了多少天

+   创建一个模板过滤器以提取第一个媒体对象

+   创建一个模板过滤器以使 URL 人性化

+   创建一个模板标签以包含一个模板（如果存在）

+   创建一个模板标签以在模板中加载 QuerySet

+   创建一个模板标签以将内容解析为模板

+   创建一个模板标签以修改请求查询参数

# 简介

如你所知，Django 有一个功能丰富的模板系统，具有模板继承、用于更改值表示的过滤器以及用于表现逻辑的标签等功能。此外，Django 允许你向你的应用程序添加自己的模板过滤器和标签。自定义过滤器或标签应位于应用程序`templatetags` Python 包下的模板标签库文件中。然后，你可以使用`{% load %}`模板标签在任何模板中加载你的模板标签库。在本章中，我们将创建几个有用的过滤器和标签，这将赋予模板编辑器更多的控制权。

要查看本章的模板标签在实际中的应用，请创建一个虚拟环境，将本章提供的代码提取到其中，运行开发服务器，并在浏览器中访问`http://127.0.0.1:8000/en/`。

# 遵循你自己的模板过滤器或标签的约定

如果你没有持续遵循的指导方针，自定义模板过滤器或标签可能会变得一团糟。模板过滤器或标签应该尽可能地为模板编辑器提供服务。它们应该既方便又灵活。在本食谱中，我们将探讨一些在增强 Django 模板系统功能时应遵循的约定。

## 如何做到这一点...

在扩展 Django 模板系统时，遵循以下约定：

1.  当页面的逻辑更适合视图、上下文处理器或模型方法时，不要创建或使用自定义模板过滤器或标签。当你的内容是上下文特定的，例如对象列表或对象详情视图，请在视图中加载对象。如果你需要在每个页面上显示一些内容，请创建一个上下文处理器。当你需要获取与模板上下文无关的对象属性时，使用模型的自定义方法而不是模板过滤器。

1.  使用`_tags`后缀命名模板标签库。如果你的应用程序名称与你的模板标签库不同，你可以避免模糊的包导入问题。

1.  在新创建的库中，将过滤器与标签分开，例如，使用以下代码中的注释进行说明：

    ```py
    # utils/templatetags/utility_tags.py
    # -*- coding: UTF-8 -*-
    from __future__ import unicode_literals
    from django import template
    register = template.Library()

    ### FILTERS ###
    # .. your filters go here..

    ### TAGS ###
    # .. your tags go here..
    ```

1.  在创建高级自定义模板标签时，确保它们的语法易于记忆，包括以下结构：

    +   `for [app_name.model_name]`：包含此结构以使用特定的模型

    +   `using [template_name]`：包含此结构以使用模板输出模板标签

    +   `limit [count]`：包含此结构以将结果限制在特定数量

    +   `as [context_variable]`：包含此结构以将结果保存到可以多次重用的上下文变量

1.  尽量避免在模板标签中定义多个位置值，除非它们是自我解释的。否则，这可能会让模板开发者感到困惑。

1.  尽可能多地创建可解析的参数。没有引号的单个字符串应被视为需要解析的上下文变量或提醒你模板标签组件结构的简短单词。

# 创建一个模板过滤器以显示自帖子发布以来已过去多少天

并非所有人都会跟踪日期，在谈论前沿信息的创建或修改日期时；对我们中的许多人来说，读取时间差更方便。例如，博客条目是三天前发布的，新闻文章是今天发布的，用户最后一次登录是昨天。在这个菜谱中，我们将创建一个名为 `days_since` 的模板过滤器，它将日期转换为人性化的时间差。

## 准备工作

如果你还没有这样做，请创建一个 `utils` 应用程序并将其放在设置中的 `INSTALLED_APPS` 下。然后，在这个应用程序中创建一个 `templatetags` Python 包（Python 包是包含空 `__init__.py` 文件的目录）。

## 如何操作...

创建一个包含以下内容的 `utility_tags.py` 文件：

```py
# utils/templatetags/utility_tags.py
# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
from datetime import datetime
from django import template
from django.utils.translation import ugettext_lazy as _
from django.utils.timezone import now as tz_now
register = template.Library()

### FILTERS ###

@register.filter
def days_since(value):
    """ Returns number of days between today and value."""

    today = tz_now().date()
    if isinstance(value, datetime.datetime):
        value = value.date()
    diff = today - value
    if diff.days > 1:
        return _("%s days ago") % diff.days
    elif diff.days == 1:
        return _("yesterday")
    elif diff.days == 0:
        return _("today")
    else:
        # Date is in the future; return formatted date.
        return value.strftime("%B %d, %Y")
```

## 它是如何工作的...

如果你像以下代码所示在模板中使用此过滤器，它将渲染类似 *昨天* 或 *5 天前* 的内容：

```py
{% load utility_tags %}
{{ object.published|days_since }}
```

你可以将此过滤器应用于 `date` 和 `datetime` 类型的值。

每个模板标签库都有一个注册表，其中收集了过滤器和标签。Django 过滤器是由 `@register.filter` 装饰器注册的函数。默认情况下，模板系统中的过滤器将命名为与函数或其它可调用对象相同的名称。如果你想，你可以通过将名称传递给装饰器来为过滤器设置不同的名称，如下所示：

```py
@register.filter(name="humanized_days_since")
def days_since(value):
    ...
```

过滤器本身相当直观。最初，读取当前日期。如果过滤器的给定值是 `datetime` 类型，则提取 `date`。然后，计算今天与提取值之间的差异。根据天数，返回不同的字符串结果。

## 更多...

此过滤器也很容易扩展以显示时间差异，例如 *刚刚*、*7 分钟前* 和 *3 小时前*。只需对 `datetime` 值而不是日期值进行操作。

## 参见

+   *创建一个用于提取第一个媒体对象的模板过滤器* 菜谱

+   *创建一个用于人性化 URL 的模板过滤器* 菜谱

# 创建一个用于提取第一个媒体对象的模板过滤器

假设你正在开发一个博客概览页面，并且对于每篇帖子，你想要在该页面上显示从内容中获取的图片、音乐或视频。在这种情况下，你需要从帖子的 HTML 内容中提取`<figure>`、`<img>`、`<object>`、`<embed>`、`<video>`、`<audio>`和`<iframe>`标签。在这个配方中，我们将看到如何使用正则表达式在`first_media`过滤器中执行此操作。

## 准备工作

我们将从`utils`应用开始，这个应用应该在设置中的`INSTALLED_APPS`中设置，并且在这个应用中设置`templatetags`包。

## 如何做到这一点...

在`utility_tags.py`文件中，添加以下内容：

```py
# utils/templatetags/utility_tags.py
# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
import re
from django import template
from django.utils.safestring import mark_safe
register = template.Library()

### FILTERS ###

media_tags_regex = re.compile(
    r"<figure[\S\s]+?</figure>|"
    r"<object[\S\s]+?</object>|"
    r"<video[\S\s]+?</video>|"
    r"<audio[\S\s]+?</audio>|"
    r"<iframe[\S\s]+?</iframe>|"
    r"<(img|embed)[^>]+>",
    re.MULTILINE
)

@register.filter
def first_media(content):
    """ Returns the first image or flash file from the html
 content """
    m = media_tags_regex.search(content)
    media_tag = ""
    if m:
        media_tag = m.group()
    return mark_safe(media_tag)
```

## 它是如何工作的...

如果数据库中的 HTML 内容有效，当你将以下代码放入模板中时，它将从对象的`content`字段中检索媒体标签；如果没有找到媒体，将返回一个空字符串。

```py
{% load utility_tags %}
{{ object.content|first_media }}
```

正则表达式是搜索/替换文本模式的一个强大功能。首先，我们将定义编译后的正则表达式为`media_file_regex`。在我们的例子中，我们将搜索所有可能出现在多行中的媒体标签。

### 小贴士

Python 字符串可以不使用加号（`+`）进行连接。

让我们看看这个正则表达式是如何工作的，如下所示：

+   交替模式由竖线（`|`）符号分隔。

+   对于可能的多行标签，我们将使用`[\S\s]+?`模式，该模式至少匹配一次任何符号；然而，尽可能少地匹配，直到我们找到它后面的字符串。因此，`<figure[\S\s]+?</figure>`搜索一个`<figure>`标签以及它之后的所有内容，直到找到关闭的`</figure>`标签。

+   类似地，使用`[^>]+`模式，我们搜索至少一次且尽可能多次的任何符号，除了大于（`>`）符号。

`re.MULTILINE`标志确保搜索将在多行中发生。然后，在过滤器中，我们将对这个正则表达式模式进行搜索。默认情况下，过滤器的结果将显示为`<`、`>`和`&`符号，它们被转义为`&lt;`、`&gt;`和`&amp;`实体。然而，我们使用`mark_safe()`函数将结果标记为安全且 HTML 就绪，以便在模板中显示而不进行转义。

## 更多...

如果你对正则表达式感兴趣，你可以在官方 Python 文档中了解更多信息，链接为[`docs.python.org/2/library/re.html`](https://docs.python.org/2/library/re.html)。

## 参见

+   *创建一个用于显示自帖子发布以来已过去多少天的模板过滤器*配方

+   *创建一个用于使 URL 人性化的模板过滤器*配方

# 创建一个用于使 URL 人性化的模板过滤器

通常，普通网络用户在地址字段中输入 URL 时没有协议和尾随斜杠。在这个配方中，我们将创建一个`humanize_url`过滤器，用于以更短的形式向用户展示 URL，截断非常长的地址，类似于 Twitter 对推文中的链接所做的那样。

## 准备工作

与之前的食谱类似，我们将从 `utils` 应用程序开始，该应用程序应在设置中的 `INSTALLED_APPS` 中设置，并包含 `templatetags` 包。

## 如何操作...

在 `utils` 应用中的 `utility_tags.py` 模板库的 `FILTERS` 部分，让我们添加一个 `humanize_url` 过滤器并将其注册，如下面的代码所示：

```py
# utils/templatetags/utility_tags.py
# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
import re
from django import template
register = template.Library()

### FILTERS ###

@register.filter
def humanize_url(url, letter_count):
    """ Returns a shortened human-readable URL """
    letter_count = int(letter_count)
    re_start = re.compile(r"^https?://")
    re_end = re.compile(r"/$")
    url = re_end.sub("", re_start.sub("", url))
    if len(url) > letter_count:
        url = "%s…" % url[:letter_count - 1]
    return url
```

## 它是如何工作的...

我们可以在任何模板中使用 `humanize_url` 过滤器，如下所示：

```py
{% load utility_tags %}
<a href="{{ object.website }}" target="_blank">
    {{ object.website|humanize_url:30 }}
</a>
```

该过滤器使用正则表达式删除前缀协议和尾随斜杠，将 URL 缩短到指定的字母数，如果 URL 不适合指定的字母数，则在末尾添加省略号。

## 参见

+   *创建一个模板过滤器以显示自帖子发布以来已过去多少天* 的食谱

+   *创建一个模板过滤器以提取第一个媒体对象* 的食谱

+   *创建一个模板标签以包含存在的模板* 的食谱

# 创建一个模板标签以包含存在的模板

Django 有 `{% include %}` 模板标签，它渲染并包含另一个模板。然而，在某些情况下存在一个问题，如果模板不存在，则会引发错误。在这个食谱中，我们将看到如何创建一个 `{% try_to_include %}` 模板标签，该标签包含另一个模板，如果不存在该模板，则静默失败。

## 准备工作

我们将再次从已安装并准备好自定义模板标签的 `utils` 应用程序开始。

## 如何操作...

高级自定义模板标签由两部分组成：解析模板标签参数的函数以及负责模板标签逻辑和输出的 `Node` 类。按照以下步骤创建 `{% try_to_include %}` 模板标签：

1.  首先，让我们创建一个解析模板标签参数的函数，如下所示：

    ```py
    # utils/templatetags/utility_tags.py
    # -*- coding: UTF-8 -*-
    from __future__ import unicode_literals
    from django import template
    from django.template.loader import get_template
    register = template.Library()

    ### TAGS ###

    @register.tag
    def try_to_include(parser, token):
      """Usage: {% try_to_include "sometemplate.html" %}
      This will fail silently if the template doesn't exist.
      If it does exist, it will be rendered with the current
      context."""
      try:
        tag_name, template_name = token.split_contents()
      except ValueError:
        raise template.TemplateSyntaxError, \
          "%r tag requires a single argument" % \
          token.contents.split()[0]
      return IncludeNode(template_name)
    ```

1.  然后，我们需要在同一个文件中的 `Node` 类，如下所示：

    ```py
    class IncludeNode(template.Node):
      def __init__(self, template_name):
        self.template_name = template_name

      def render(self, context):
        try:
          # Loading the template and rendering it
          template_name = template.resolve_variable(
            self. template_name, context)
          included_template = get_template(
            template_name
          ).render(context)
        except template.TemplateDoesNotExist:
          included_template = ""
        return included_template
    ```

## 它是如何工作的...

`{% try_to_include %}` 模板标签期望一个参数，即 `template_name`。因此，在 `try_to_include()` 函数中，我们尝试将标记的分隔内容仅分配给 `tag_name` 变量（即 `try_to_include`）和 `template_name` 变量。如果这不起作用，则引发模板语法错误。该函数返回 `IncludeNode` 对象，该对象获取 `template_name` 字段以供以后使用。

在 `IncludeNode` 的 `render()` 方法中，我们解析 `template_name` 变量。如果向模板标签传递了上下文变量，则其值将在这里用于 `template_name`。如果向模板标签传递了引号字符串，则引号内的内容将用于 `template_name`。

最后，我们将尝试加载模板并使用当前模板上下文进行渲染。如果不起作用，则返回空字符串。

至少有两种情况我们可以使用这个模板标签：

+   当在模型中定义路径时包含模板，如下所示：

    ```py
    {% load utility_tags %}
    {% try_to_include object.template_path %}
    ```

+   它用于在模板上下文变量作用域中定义路径的模板中包含模板时。这在需要为 Django CMS 中的占位符创建自定义布局的插件时特别有用：

    ```py
    {# templates/cms/start_page.html #}
    {% with editorial_content_template_path="cms/plugins/editorial_content/start_page.html" %}
        {% placeholder "main_content" %}
    {% endwith %}

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

## 更多...

你可以使用`{% try_to_include %}`标签以及默认的`{% include %}`标签来包含扩展其他模板的模板。这对于大型门户来说是有益的，在这些门户中，你有不同种类的列表，其中复杂的项目与小部件具有相同的结构，但数据来源不同。

例如，在艺术家列表模板中，你可以包含艺术家项目模板，如下所示：

```py
{% load utility_tags %}
{% for object in object_list %}
    {% try_to_include "artists/includes/artist_item.html" %}
{% endfor %}
```

此模板将从项目基类扩展，如下所示：

```py
{# templates/artists/includes/artist_item.html #}
{% extends "utils/includes/item_base.html" %}

{% block item_title %}
    {{ object.first_name }} {{ object.last_name }}
{% endblock %}
```

项目基类定义了任何项目的标记，并包括一个 Like 小部件，如下所示：

```py
{# templates/utils/includes/item_base.html #}
{% load likes_tags %}

<h3>{% block item_title %}{% endblock %}</h3>
{% if request.user.is_authenticated %}
    {% like_widget for object %}
{% endif %}
```

## 参见

+   在第七章的*为 Django CMS 创建模板*菜谱中，*Django CMS*

+   在第七章的*编写自己的 CMS 插件*菜谱中，*Django CMS*

+   在第四章的*实现 Like 小部件*菜谱中，*模板和 JavaScript*

+   *在模板中创建一个加载 QuerySet 的模板标签*菜谱

+   *创建一个将内容解析为模板的模板标签*菜谱

+   *创建一个修改请求查询参数的模板标签*菜谱

# 在模板中创建一个加载 QuerySet 的模板标签

通常，应该在视图中定义要在网页上显示的内容。如果这是要在每个页面上显示的内容，那么创建一个上下文处理器是合理的。另一种情况是，你需要在某些页面上显示额外的内容，例如最新新闻或随机引言；例如，对象的起始页面或详细信息页面。在这种情况下，你可以使用`{% get_objects %}`模板标签加载必要的内容，我们将在本菜谱中实现它。

## 准备工作

再次，我们将从应该安装并准备好自定义模板标签的`utils`应用开始。

## 如何做到...

一个高级自定义模板标签由一个解析传递给标签的参数的函数和一个`Node`类组成，该类渲染标签的输出或修改模板上下文。执行以下步骤以创建`{% get_objects %}`模板标签：

1.  首先，让我们创建一个解析模板标签参数的函数，如下所示：

    ```py
    # utils/templatetags/utility_tags.py
    # -*- coding: UTF-8 -*-
    from __future__ import unicode_literals
    from django.db import models
    from django import template
    register = template.Library()

    ### TAGS ###

    @register.tag
    def get_objects(parser, token):
        """
        Gets a queryset of objects of the model specified
        by app and model names
        Usage:
            {% get_objects [<manager>.]<method> from
    <app_name>.<model_name> [limit <amount>] as
            <var_name> %}
        Example:
            {% get_objects latest_published from people.Person
    limit 3 as people %}
            {% get_objects site_objects.all from news.Article
            limit 3 as articles %}
            {% get_objects site_objects.all from news.Article
             as articles %}
        """
        amount = None
        try:
            tag_name, manager_method, str_from, appmodel, \
            str_limit, amount, str_as, var_name = \
                token.split_contents()
        except ValueError:
            try:
                tag_name, manager_method, str_from, appmodel, \
                str_as, var_name = token.split_contents()
            except ValueError:
                raise template.TemplateSyntaxError, \
                    "get_objects tag requires a following "\
                    "syntax: "\
                    "{% get_objects [<manager>.]<method> "\
                    "from <app_ name>.<model_name> "\
                    "[limit <amount>] as <var_name> %}"
        try:
            app_name, model_name = appmodel.split(".")
        except ValueError:
            raise template.TemplateSyntaxError, \
                "get_objects tag requires application name "\
                "and model name separated by a dot"
        model = models.get_model(app_name, model_name)
        return ObjectsNode(
            model, manager_method, amount, var_name
        )
    ```

1.  然后，我们将在同一文件中创建`Node`类，如下面的代码所示：

    ```py
    class ObjectsNode(template.Node):
        def __init__(
            self, model, manager_method, amount, var_name
        ):
            self.model = model
            self.manager_method = manager_method
            self.amount = amount
            self.var_name = var_name

        def render(self, context):
            if "." in self.manager_method:
                manager, method = \
                    self.manager_method.split(".")
            else:
                manager = "_default_manager"
                method = self.manager_method

            qs = getattr(
                getattr(self.model, manager),
                method,
                self.model._default_manager.none,
            )()
            if self.amount:
                amount = template.resolve_variable(
                    self.amount, context
                )
                context[self.var_name] = qs[:amount]
            else:
                context[self.var_name] = qs
            return ""
    ```

## 它是如何工作的...

`{% get_objects %}`模板标签从指定的应用和模型的方法中加载定义的 QuerySet，将结果限制到指定的数量，并将结果保存到上下文变量中。

以下代码是使用我们刚刚创建的模板标签的最简单示例。它将在任何模板中使用以下片段加载所有新闻文章：

```py
{% load utility_tags %}
{% get_objects all from news.Article as all_articles %}
{% for article in all_articles %}
    <a href="{{ article.get_url_path }}">{{ article.title }}</a>
{% endfor %}
```

这是在使用`Article`模型的默认`objects`管理器的`all()`方法，并且它将根据模型`Meta`类中定义的`ordering`属性对文章进行排序。

创建一个需要自定义管理器和自定义方法来从数据库查询对象的更高级的示例。管理器是一个提供数据库查询操作给模型的接口。每个模型默认至少有一个名为`objects`的管理器。作为一个例子，让我们创建一个具有草稿或发布状态以及允许选择随机发布艺术家的`custom_manager`的`Artist`模型：

```py
# artists/models.py
# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
from django.db import models
from django.utils.translation import ugettext_lazy as _

STATUS_CHOICES = (
    ("draft", _("Draft"),
    ("published", _("Published"),
)
class ArtistManager(models.Manager):
    def random_published(self):
        return self.filter(status="published").order_by("?")

class Artist(models.Model):
    # ...
    status = models.CharField(_("Status"), max_length=20, 
        choices=STATUS_CHOICES)
    custom_manager =  ArtistManager()
```

要加载一个随机发布的艺术家，您可以将以下片段添加到任何模板中：

```py
{% load utility_tags %}
{% get_objects custom_manager.random_published from artists.Artist limit 1 as random_artists %}
{% for artist in random_artists %}
    {{ artist.first_name }} {{ artist.last_name }}
{% endfor %}
```

让我们看看`{% get_objects %}`模板标签的代码。在解析函数中，有两种预期的格式之一；带有限制和不带限制。字符串将被解析，模型将被识别，然后模板标签的组件传递给`ObjectNode`类。

在`Node`类的`render()`方法中，我们将检查管理器的名称及其方法名称。如果没有定义，将使用`_default_manager`，这是 Django 注入的任何模型的一个自动属性，指向第一个可用的`models.Manager()`实例。在大多数情况下，`_default_manager`将与`objects`相同。之后，我们将调用管理器的方法，如果方法不存在，则回退到空的`QuerySet`。如果定义了限制，我们将解析其值并限制`QuerySet`。最后，我们将`QuerySet`保存到上下文变量中。

## 参见

+   *创建一个模板标签以包含一个模板（如果存在）*的配方

+   *创建一个模板标签以将内容解析为模板*的配方

+   *创建一个模板标签以修改请求查询参数*的配方

# 创建一个模板标签以将内容解析为模板

在这个配方中，我们将创建一个`{% parse %}`模板标签，这将允许您将模板片段放入数据库。当您想为认证用户和非认证用户提供不同的内容，或者想包含个性化的问候语或不想在数据库中硬编码媒体路径时，这非常有用。

## 准备工作

如同往常，我们将从应该安装并准备好自定义模板标签的`utils`应用开始。

## 如何操作...

一个高级的自定义模板标签由一个解析传递给标签的参数的函数和一个渲染标签输出或修改模板上下文的`Node`类组成。按照以下步骤创建它们：

1.  首先，让我们创建一个解析模板标签参数的函数，如下所示：

    ```py
    # utils/templatetags/utility_tags.py
    # -*- coding: UTF-8 -*-
    from __future__ import unicode_literals
    from django import template
    register = template.Library()

    ### TAGS ###

    @register.tag
    def parse(parser, token):
        """
        Parses the value as a template and prints it or
        saves to a variable
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
            if len(bits) == 2:
                bits.pop(0)  # remove the word "as"
                var_name = bits.pop(0)
        except ValueError:
            raise template.TemplateSyntaxError, \
                "parse tag requires a following syntax: "\
                "{% parse <template_value> [as <variable>] %}"

        return ParseNode(template_value, var_name)
    ```

1.  然后，我们将在同一文件中创建`Node`类，如下所示：

    ```py
    class ParseNode(template.Node):
        def __init__(self, template_value, var_name):
            self.template_value = template_value
            self.var_name = var_name

        def render(self, context):
            template_value = template.resolve_variable(
                self.template_value, context)
            t = template.Template(template_value)
            context_vars = {}
            for d in list(context):
                for var, val in d.items():
                    context_vars[var] = val
            result = t.render(template.RequestContext(
                context["request"], context_vars))
            if self.var_name:
                context[self.var_name] = result
                return ""
            return result
    ```

## 它是如何工作的...

`{% parse %}`模板标签允许您将值解析为模板并立即渲染或将其保存为上下文变量。

如果我们有一个包含描述字段的对象，该字段可以包含模板变量或逻辑，我们可以使用以下代码进行解析和渲染：

```py
{% load utility_tags %}
{% parse object.description %}
```

也可以定义一个值以便使用引号字符串进行解析，如下面的代码所示：

```py
{% load utility_tags %}
{% parse "{{ STATIC_URL }}site/img/" as img_path %}
<img src="img/{{ img_path }}someimage.png" alt="" />
```

让我们看看`{% parse %}`模板标签的代码。解析函数逐个检查模板标签的参数。首先，我们期望`parse`名称，然后是模板值，最后我们期望可选的`as`词后跟上下文变量名称。模板值和变量名称传递给`ParseNode`类。该类的`render()`方法首先解析模板变量的值，并从中创建一个模板对象。然后，它使用所有上下文变量渲染模板。如果变量名称已定义，结果被保存到它；否则，结果立即显示。

## 参见

+   *创建一个模板标签以包含模板（如果存在）*菜谱

+   *在模板中创建加载 QuerySet 的模板标签*菜谱

+   *创建一个模板标签来修改请求查询参数*菜谱

# 创建一个模板标签来修改请求查询参数

Django 通过向 URL 配置文件添加正则表达式规则，提供了一个方便且灵活的系统来创建规范和干净的 URL。然而，在管理查询参数方面，缺乏内置机制。例如，搜索或可筛选对象列表视图需要接受查询参数，以便通过另一个参数深入筛选结果或转到另一页。在这个菜谱中，我们将创建`{% modify_query %}`、`{% add_to_query %}`和`{% remove_from_query %}`模板标签，这些标签允许您添加、更改或删除当前查询的参数。

## 准备工作

再次强调，我们从`utils`应用开始，它应该在`INSTALLED_APPS`中设置，并包含`templatetags`包。

此外，请确保您已为`TEMPLATE_CONTEXT_PROCESSORS`设置配置了`request`上下文处理器，如下所示：

```py
# conf/base.py or settings.py
TEMPLATE_CONTEXT_PROCESSORS = (
    "django.contrib.auth.context_processors.auth",
    "django.core.context_processors.debug",
    "django.core.context_processors.i18n",
    "django.core.context_processors.media",
    "django.core.context_processors.static",
    "django.core.context_processors.tz",
    "django.contrib.messages.context_processors.messages",
 "django.core.context_processors.request",
)
```

## 如何去做...

对于这些模板标签，我们将使用`simple_tag`装饰器来解析组件，并要求您只需定义渲染函数，如下所示：

1.  首先，我们将创建`{% modify_query %}`模板标签：

    ```py
    # utils/templatetags/utility_tags.py
    # -*- coding: UTF-8 -*-
    from __future__ import unicode_literals
    import urllib
    from django import template
    from django.utils.encoding import force_str
    register = template.Library()

    ### TAGS ###

    @register.simple_tag(takes_context=True)
    def modify_query(
        context, *params_to_remove, **params_to_change
    ):
        """ Renders a link with modified current query
        parameters """
        query_params = []
        for key, value_list in \
            context["request"].GET._iterlists():
            if not key in params_to_remove:
                # don't add key-value pairs for
                # params_to_change
                if key in params_to_change:
                    query_params.append(
                        (key, params_to_change[key])
                    )
                    params_to_change.pop(key)
                else:
                    # leave existing parameters as they were
                    # if not mentioned in the params_to_change
                    for value in value_list:
                        query_params.append((key, value))
        # attach new params
        for key, value in params_to_change.items():
            query_params.append((key, value))
        query_string = context["request"].path
        if len(query_params):
            query_string += "?%s" % urllib.urlencode([
                (key, force_str(value))
                for (key, value) in query_params if value
            ]).replace("&", "&amp;")
        return query_string
    ```

1.  然后，让我们创建`{% add_to_query %}`模板标签：

    ```py
    @register.simple_tag(takes_context=True)
    def add_to_query(
        context, *params_to_remove, **params_to_add
    ):
        """ Renders a link with modified current query
        parameters """
        query_params = []
        # go through current query params..
        for key, value_list in \
            context["request"].GET._iterlists():
            if not key in params_to_remove:
                # don't add key-value pairs which already
                # exist in the query
                if key in params_to_add and \
                unicode(params_to_add[key]) in value_list:
                    params_to_add.pop(key)
                for value in value_list:
                    query_params.append((key, value))
        # add the rest key-value pairs
        for key, value in params_to_add.items():
            query_params.append((key, value))
        # empty values will be removed
        query_string = context["request"].path
        if len(query_params):
            query_string += "?%s" % urllib.urlencode([
                (key, force_str(value))
                for (key, value) in query_params if value
            ]).replace("&", "&amp;")
        return query_string
    ```

1.  最后，让我们创建`{% remove_from_query %}`模板标签：

    ```py
    @register.simple_tag(takes_context=True)
    def remove_from_query(context, *args, **kwargs):
        """ Renders a link with modified current query
        parameters """
        query_params = []
        # go through current query params..
        for key, value_list in \
            context["request"].GET._iterlists():
            # skip keys mentioned in the args
            if not key in args:
                for value in value_list:
                    # skip key-value pairs mentioned in kwargs
                    if not (key in kwargs and
                      unicode(value) == unicode(kwargs[key])):
                        query_params.append((key, value))
        # empty values will be removed
        query_string = context["request"].path
        if len(query_params):
            query_string = "?%s" % urllib.urlencode([
                (key, force_str(value))
                for (key, value) in query_params if value
            ]).replace("&", "&amp;")
        return query_string
    ```

## 它是如何工作的...

所有的三个创建的模板标签表现相似。首先，它们从`request.GET`字典样式的`QueryDict`对象中读取当前查询参数到一个新的键值`query_params`元组列表中。然后，根据位置参数和关键字参数更新值。最后，形成新的查询字符串，所有空格和特殊字符都进行 URL 编码，并且连接查询参数的与号被转义。这个新的查询字符串被返回到模板中。

### 小贴士

要了解更多关于 `QueryDict` 对象的信息，请参阅官方 Django 文档，[`docs.djangoproject.com/en/1.8/ref/request-response/#querydict-objects`](https://docs.djangoproject.com/en/1.8/ref/request-response/#querydict-objects)。

让我们看看如何使用 `{% modify_query %}` 模板标签的示例。模板标签中的位置参数定义了要删除哪些查询参数，而关键字参数定义了在当前查询中要修改哪些查询参数。如果当前 URL 是 `http://127.0.0.1:8000/artists/?category=fine-art&page=5`，我们可以使用以下模板标签来渲染一个链接，该链接跳转到下一页：

```py
{% load utility_tags %}
<a href="{% modify_query page=6 %}">6</a>
```

以下片段是使用前面的模板标签渲染的输出：

```py
<a href="/artists/?category=fine-art&amp;page=6">6</a>
```

我们也可以使用以下示例来渲染一个链接，该链接重置分页并跳转到另一个分类，*雕塑*，如下所示：

```py
{% load utility_tags i18n %}
<a href="{% modify_query "page" category="sculpture" %}">{% trans "Sculpture" %}</a>
```

以下片段是使用前面的模板标签渲染的输出：

```py
<a href="/artists/?category=sculpture">Sculpture</a>
```

使用 `{% add_to_query %}` 模板标签，您可以逐步添加具有相同名称的参数。例如，如果当前 URL 是 `http://127.0.0.1:8000/artists/?category=fine-art`，您可以使用以下链接添加另一个分类，*雕塑*：

```py
{% load utility_tags i18n %}
<a href="{% add_to_query "page" category="sculpture" %}">{% trans "Sculpture" %}</a>
```

这将在模板中渲染成如下片段：

```py
<a href="/artists/?category=fine-art&amp;category=sculpture">Sculpture</a>
```

最后，借助 `{% remove_from_query %}` 模板标签，您可以逐步删除具有相同名称的参数。例如，如果当前 URL 是 `http://127.0.0.1:8000/artists/?category=fine-art&category=sculpture`，您可以使用以下链接帮助删除 *雕塑* 分类：

```py
{% load utility_tags i18n %}
<a href="{% remove_from_query "page" category="sculpture" %}"><span class="glyphicon glyphicon-remove"></span> {% trans "Sculpture" %}</a>
```

这将在模板中渲染如下：

```py
<a href="/artists/?category=fine-art"><span class="glyphicon glyphicon-remove"></span> Sculpture</a>
```

## 参见

+   第三章中的 *过滤对象列表* 配方，*表单和视图*

+   *创建一个模板标签以包含存在的模板* 的配方

+   *创建一个模板标签以在模板中加载 QuerySet* 的配方

+   *创建一个模板标签以将内容解析为模板* 的配方
