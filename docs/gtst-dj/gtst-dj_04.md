# 第四章：使用模板

正如我们在第一章中所看到的，我们解释了 MVC 和 MVT 模型，模板是允许我们生成返回给客户端的 HTML 代码的文件。在我们的视图中，HTML 代码不与 Python 代码混合。

Django 自带其自己的模板系统。然而，由于 Django 是模块化的，可以使用不同的模板系统。这个系统由一个语言组成，将用于制作我们的动态模板。

在本章中，我们将学习如何做以下事情：

+   将数据发送到模板

+   在模板中显示数据

+   在模板中显示对象列表

+   在 Django 中使用过滤器处理链

+   有效使用 URL

+   创建基础模板以扩展其他模板

+   在我们的模板中插入静态文件

# 在模板中显示 Hello world！

我们将创建我们应用程序的第一个模板。为此，我们必须首先编辑`settings.py`文件，以定义将包含我们模板的文件夹。我们将首先将项目文件夹定义为`PROJECT_ROOT`，以简化迁移到另一个系统：

```py
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIRS = (os.path.join(PROJECT_ROOT, '../TasksManager/templates')
  # Put strings here, like "/home/html/django_templates" or "C:/www/django/templates".
  # Always use forward slashes, even on Windows.
  # Don't forget to use absolute paths, not relative paths.
)
```

现在 Django 知道在哪里查找模板，我们将创建应用程序的第一个模板。为此，请使用文件浏览器，并在`TasksManager/templates/en/public/`文件夹中添加`index.html`文件。我们不需要创建`__init__.py`文件，因为这些文件不包含任何 Python 文件。

以下是`index.html`文件的内容：

```py
<html>
  <head>
    <title>
      Hello World Title
    </title>
  </head>
  <body>
    <h1>
      Hello World Django
    </h1>
    <article>
      Hello world !
    </article>
  </body>
</html>
```

尽管模板是正确的，但我们需要更改视图以指示其使用。我们将使用以下内容修改`index.py`文件：

```py
from django.shortcuts import render
# View for index page. 
def page(request):
  return render(request, 'en/public/index.html')
```

如果我们测试这个页面，我们会注意到模板已经被视图考虑进去了。

# 从视图向模板注入数据

在改进我们的模板之前，我们必须将变量发送到模板。数据的注入是基于这些变量，因为模板将执行某些操作。事实上，正如我们在 MVC 模式的解释中所看到的，控制器必须将变量发送到模板以便显示它们。

有几个函数可以将变量发送到模板。两个主要函数是`render()`和`render_to_response()`。`render()`函数与`render_to_response()`非常相似。主要区别在于，如果我们使用`render`，我们不需要指定`context_instance = RequestContext(request)`以发送当前上下文。这是稍后在本书中使用 CSRF 中间件的上下文。

我们将改变我们的视图，以在我们的模板中注入变量。这些变量将对使用模板语言非常有用。以下是我们修改后的视图：

```py
from django.shortcuts import render
"""
View for index page. 
"""

def page(request):
  my_variable = "Hello World !"
  years_old = 15
  array_city_capitale = [ "Paris", "London", "Washington" ]
  return render(request, 'en/public/index.html', { "my_var":my_variable, "years":years_old, "array_city":array_city_capitale })
```

# 创建动态模板

Django 自带完整的模板语言。这意味着我们将使用模板标签，这将允许我们在模板中具有更多的灵活性，并显示变量，执行循环，并设置过滤器。

HTML 和模板语言在模板中混合在一起；然而，模板语言非常简单，与 HTML 代码相比只是少数。网页设计师可以轻松修改模板文件。

# 在模板中集成变量

在我们的控制器中，我们发送了一个名为`my_var`的变量。我们可以以以下方式在`<span>`标签中显示它。在我们的模板标签的`<article>`标签中添加以下行：

```py
<span> {{my_var}} </ span> 
```

因此，因为我们的变量包含`string = "Hello World!"`，将生成以下 HTML 代码：

```py
<span> Hello World! </span>
```

我们将学习如何为变量或函数创建条件，以便在以下示例中过滤变量中的数据。

## 条件语句

语言模板还允许条件结构。请注意，对于显示变量，使用双大括号`{{}}`，但一旦我们有一个作为条件或循环的操作，我们将使用`{%%}`。

我们的控制器发送一个可以定义年龄的`years`变量。条件结构的一个示例是，当您可以更改控制器中变量的值以观察更改时。在我们的`<article>`标签中添加以下代码：

```py
<span>
  {% if years <10 %}
    You are a children
  {% elif years < 18 %}
    You are a teenager
  {% else %}
    You are an adult!
  {% endif %}
</span>
```

在我们的情况下，当我们将值`15`发送到生成的模板时，使用的代码如下：

```py
<span> You are a teenager </span>
```

## 在模板中循环

循环允许您阅读表或数据字典的元素。在我们的控制器中，我们发送了一个名为`array_city`的数据表，其中包含城市的名称。要以列表形式查看所有这些城市的名称，我们可以在模板中编写以下内容：

```py
<ul>
  {% for city in array_city %}
    <li>
      {{ city }}
    </li>
  {% endfor %}
</ul>
```

此循环将遍历`array_city`表，并将每个元素放入我们在`<li>`标签中显示的`city`变量中。使用我们的示例数据，此代码将生成以下 HTML 代码：

```py
<ul>
  <li>Paris</li>
  <li>London</li>
  <li>Washington</li>
</ul>
```

# 使用过滤器

过滤器是在将数据发送到模板之前修改数据的有效方法。我们将在以下部分中查看一些过滤器的示例，以更好地理解它们。

## 大写和小写过滤器

小写过滤器将转换为小写字母，而大写过滤器将转换为大写字母。在接下来的部分中给出的示例中包含`my_hello`变量，其值为`Hello World!`

### 小写过滤器

小写过滤器的代码如下：

```py
<span> {{ my_hello | lower }} </span>
```

此代码生成以下 HTML 代码：

```py
<span> hello </span>
```

### 大写过滤器

大写过滤器的代码如下：

```py
<span> {{ my_hello | upper }} </span>
```

此代码生成以下 HTML 代码：

```py
<span> HELLO </span>
```

## capfirst 过滤器

capfirst 过滤器将第一个字母转换为大写。具有`myvar = "hello"`变量的示例如下：

```py
<span>{{ my_hello | capfirst }}</span>
```

此代码生成以下 HTML 代码：

```py
<span> Hello </span>
```

## 复数过滤器

复数过滤器可以轻松处理复数形式。通常，开发人员由于时间不足而选择简单的解决方案。解决方案是显示频道：*您的购物车中有 2 个产品*。

Django 简化了这种类型的字符串。如果变量表示复数值，复数过滤器将在单词末尾添加后缀，如下所示：

```py
You have {{ product }} nb_products {{ nb_products | pluralize }} in our cart.
```

如果`nb_products`为`1`和`2`，则此频道将显示以下三个频道：

```py
You have 1 product in our cart.
You have 2 products in our cart.
I received {{ nb_diaries }} {{ nb_diaries|pluralize : "y , ies "}}.
```

如果`nb_diaries`为`1`和`2`，则上述代码将显示以下两个链：

```py
I received one diary.
I received two diaries.
```

在上一个示例中，我们首次使用了带参数的过滤器。要为过滤器设置参数，必须使用以下语法：

```py
{{ variable | filter:"parameters" }}
```

此过滤器有助于提高您网站的质量。当网站显示正确的句子时，它看起来更专业。

## 转义和安全以避免 XSS 过滤器

XSS 过滤器用于转义 HTML 字符。此过滤器有助于防止 XSS 攻击。这些攻击是基于黑客注入客户端脚本的。以下是 XSS 攻击的逐步描述：

+   攻击者找到一个表单，以便内容将显示在另一个页面上，例如商业网站的评论字段。

+   黑客编写 JavaScript 代码以使用此表单中的标记进行黑客攻击。提交表单后，JavaScript 代码将存储在数据库中。

+   受害者查看页面评论，JavaScript 运行。

风险比简单的`alert()`方法更重要，以显示消息。使用这种类型的漏洞，黑客可以窃取会话 ID，将用户重定向到伪造的网站，编辑页面等。

更具体地说，过滤器更改以下字符：

+   `<` 被转换为 `&lt;`

+   `>` 被转换为 `&gt;`

+   `'` 被转换为 `'`

+   `"` 被转换为 `&quot;`

+   `&` 被转换为 `&amp;`

我们可以使用`{% autoescape %} tag`自动转义块的内容，该标签带有 on 或 off 参数。默认情况下，autoescape 是启用的，但请注意，在较旧版本的 Django 中，autoescape 未启用。

当启用 autoescape 时，如果我们想将一个变量定义为可信任的变量，我们可以使用 safe 过滤器对其进行过滤。以下示例显示了不同的可能场景：

```py
<div>
  {% autoescape on %}
  <div>
    <p>{{ variable1 }}</p>
    <p>
      <span>
        {{ variable2|safe }}
      </span>
      {% endautoescape %}
      {% autoescape off %}
    </p>
  </div>
    <span>{{ variable3 }}</span>
    <span>{{ variable4|escape }}</span>
  {% endautoescape %}
  <span>{{ variable5 }}</span>
</div>
```

在这个例子中：

+   `variable1`被`autoescape`转义

+   `variable2`没有被转义，因为它被过滤为安全的

+   `variable3`没有被转义，因为`autoescape`被定义为关闭

+   `variable4`被转义，因为它已经使用转义过滤器进行了过滤

+   `variable5`被转义，因为`autoescape`是关闭的

## linebreaks 过滤器

linebreaks 过滤器允许您将换行符转换为 HTML 标记。一个单独的换行符被转换为`<br />`标记。一个换行符后跟一个空格将变成一个段落分隔，`</p>`：

```py
<span>{{ text|linebreaks }}</span>
```

## truncatechars 过滤器

truncatechars 过滤器允许您从一定长度截断字符串。如果超过这个数字，字符串将被截断，Django 会添加字符串“`...`”。

包含“欢迎来到 Django”的变量的示例如下：

```py
{{ text|truncatechars:14 }}
```

这段代码输出如下：

```py
"Welcome in ..."
```

# 创建 DRY URL

在学习什么是 DRY 链接之前，我们首先会提醒您 HTML 链接是什么。每天，当我们上网时，我们通过点击链接来改变页面或网站。这些链接被重定向到 URL。以下是一个指向[google.com](http://google.com)的示例链接：

```py
<a href="http://www.google.com">Google link !</a>
```

我们将在我们的应用程序中创建第二个页面，以创建第一个有效的链接。在`urls.py`文件中添加以下行：

```py
url(r'^connection$', 'TasksManager.views.connection.page'),
```

然后，创建一个对应于前面 URL 的视图：

```py
from django.shortcuts import render
# View for connection page. 
def page(request):
  return render(request, 'en/public/connection.html')
```

我们将为新视图创建第二个模板。让我们复制第一个模板，并将副本命名为`connection.html`，并修改`Connection`中的`Hello world`。我们可以注意到这个模板不符合 DRY 哲学。这是正常的；我们将在下一节学习如何在不同模板之间共享代码。

我们将在我们的第一个`index.html`模板中创建一个 HTML 链接。这个链接将引导用户到我们的第二个视图。我们的`<article>`标签变成了：

```py
<article>
  Hello world !
  <br />
  <a href="connection">Connection</a>
</article>
```

现在，让我们用开发服务器测试我们的网站，并打开浏览器到我们网站的 URL。通过测试网站，我们可以检查链接是否正常工作。这是一个好事，因为现在你能够用 Django 制作一个静态网站，而且这个框架包含一个方便的工具来管理 URL。

Django 永远不会在`href`属性中写入链接。事实上，通过正确地填写我们的`urls.py`文件，我们可以引用 URL 的名称和地址。

为了做到这一点，我们需要改变包含以下 URL 的`urls.py`文件：

```py
url(r'^$', 'TasksManager.views.index.page', name="public_index"),
url(r'^connection/$', 'TasksManager.views.connection.page', name="public_connection"),
```

给我们的每个 URL 添加 name 属性可以让我们使用 URL 的名称来创建链接。修改您的`index.html`模板以创建 DRY 链接：

```py
<a href="{% url 'public_connection' %}">Connection</a>
```

再次测试新网站；请注意，链接仍然有效。但是目前，这个功能对我们来说是没有用的。如果 Google 决定改进以网站名称结尾的 URL 的索引，您将不得不更改所有的 URL。要在 Django 中做到这一点，您只需要更改第二个 URL 如下：

```py
url(r'^connection-TasksManager$', 'TasksManager.views.connection.page', name="public_connection"),
```

如果我们再次测试我们的网站，我们可以看到更改已经正确完成，并且`urls.py`文件中的更改对网站的所有页面都有效。当您需要使用参数化 URL 时，您必须使用以下语法将参数集成到 URL 中：

```py
{% url "url_name" param %}
{% url "url_name" param1, param2 %}
```

# 扩展模板

模板的传承允许您定义一个超级模板和一个从超级模板继承的子模板。在超级模板中，可以定义子模板可以填充的块。这种方法允许我们通过在超级模板中应用通用代码到多个模板来遵循 DRY 哲学。我们将使用一个例子，`index.html`模板将扩展`base.html`模板。

以下是我们必须在`template`文件夹中创建的`base.html`模板代码：

```py
<html>
  <head>
    <title>
      % block title_html %}{% endblock %}
    </title>
  </head>
  <body>
    <h1>
      Tasks Manager - {% block h1 %}{% endblock %}
    </h1>
    <article>
      {% block article_content %}{% endblock %}
    </article>
  </body>
</html>
```

在前面的代码中，我们定义了子模板可以覆盖的三个区域：`title_html`、`h1`和`article_content`。以下是`index.html`模板代码：

```py
{% extends "base.html" %}
{% block title_html %}
  Hello World Title
{% endblock %}
{% block h1 %}
  {{ bloc.super }}Hello World Django
{% endblock %}
{% block article_content %}
  Hello world !
{% endblock %}
```

在这个模板中，我们首先使用了 extends 标签，它扩展了`base.html`模板。然后，block 和 endblock 标签允许我们重新定义`base.html`模板中的内容。我们可以以相同的方式更改我们的`connection.html`模板，这样`base.html`的更改就可以在两个模板上进行。

可以定义尽可能多的块。我们还可以创建超级模板，以创建更复杂的架构。

# 在模板中使用静态文件

诸如 JavaScript 文件、CSS 或图像之类的静态文件对于获得人体工程学网站至关重要。这些文件通常存储在一个文件夹中，但在开发或生产中修改此文件夹可能会很有用。

根据 URL，Django 允许我们定义一个包含静态文件的文件夹，并在需要时轻松修改其位置。

要设置 Django 查找静态文件的路径，我们必须通过添加或更改以下行来更改我们的`settings.py`文件：

```py
STATIC_URL = '/static/'
STATICFILES_DIRS = (
    os.path.join(PROJECT_ROOT, '../TasksManager/static/'),
)
```

我们将为我们未来的静态文件定义一个合适的架构。选择早期一致的架构非常重要，因为它使应用程序支持以及包括其他开发人员变得更容易。我们的静态文件架构如下：

```py
static/
  images/
  javascript/
    lib/
  css/
  pdf/
```

我们为每种静态文件创建一个文件夹，并为 JavaScript 库定义一个`lib`文件夹，如 jQuery，我们将在本书中使用。例如，我们更改了我们的`base.html`文件。我们将添加一个 CSS 文件来管理我们页面的样式。为了做到这一点，我们必须在`</title>`和`</head>`之间添加以下行：

```py
<link href="{% static "css/style.css" %}" rel="stylesheet" type="text/css" />
```

在我们的静态模板中使用标签，我们还必须通过在使用静态标签之前放置以下行来加载系统：

```py
{% load staticfiles %}
```

我们将在`/static/css`文件夹中创建`style.css`文件。这样，浏览器在开发过程中不会生成错误。

# 摘要

在本章中，我们学习了如何创建模板并将数据发送到模板，以及如何在模板中使用条件、循环和过滤器。我们还讨论了如何为灵活的 URL 结构创建 DRY URLs，扩展模板以满足 DRY 哲学，以及如何使用静态文件。

在下一章中，我们将学习如何结构化我们的数据以保存在数据库中。
