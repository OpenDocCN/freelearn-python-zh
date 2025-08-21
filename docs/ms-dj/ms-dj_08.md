# 第八章：高级模板

尽管你与 Django 的模板语言的大部分交互将是作为模板作者的角色，但你可能想要自定义和扩展模板引擎-要么使其执行一些它尚未执行的操作，要么以其他方式使你的工作更轻松。

本章深入探讨了 Django 模板系统的内部。它涵盖了如果你计划扩展系统或者只是对它的工作方式感到好奇，你需要了解的内容。它还涵盖了自动转义功能，这是一项安全措施，随着你继续使用 Django，你肯定会注意到它。

# 模板语言回顾

首先，让我们快速回顾一些在第三章*模板*中引入的术语：

+   **模板**是一个文本文档，或者是一个普通的 Python 字符串，使用 Django 模板语言进行标记。模板可以包含模板标签和变量。

+   **模板标签**是模板中的一个符号，它执行某些操作。这个定义是故意模糊的。例如，模板标签可以生成内容，充当控制结构（`if`语句或`for`循环），从数据库中获取内容，或者启用对其他模板标签的访问。

模板标签用`{%`和`%}`括起来：

```py
        {% if is_logged_in %} 
            Thanks for logging in! 
        {% else %} 
            Please log in. 
        {% endif %} 

```

+   **变量**是模板中输出值的符号。

+   变量标签用`{{`和`}}`括起来：

+   **上下文**是传递给模板的`name->value`映射（类似于 Python 字典）。

+   模板通过用上下文中的值替换变量“洞”并执行所有模板标签来**渲染**上下文。

有关这些术语的基础知识的更多细节，请参考第三章*模板*。本章的其余部分讨论了扩展模板引擎的方法。不过，首先让我们简要地看一下第三章*模板*中省略的一些内部内容，以简化。

# RequestContext 和上下文处理器

在渲染模板时，你需要一个上下文。这可以是`django.template.Context`的一个实例，但 Django 也带有一个子类`django.template.RequestContext`，它的行为略有不同。

`RequestContext`默认情况下向您的模板上下文添加了一堆变量-诸如`HttpRequest`对象或有关当前登录用户的信息。

`render()`快捷方式会创建一个`RequestContext`，除非显式传递了不同的上下文实例。例如，考虑这两个视图：

```py
from django.template import loader, Context 

def view_1(request): 
    # ... 
    t = loader.get_template('template1.html') 
    c = Context({ 
        'app': 'My app', 
        'user': request.user, 
        'ip_address': request.META['REMOTE_ADDR'], 
        'message': 'I am view 1.' 
    }) 
    return t.render(c) 

def view_2(request): 
    # ... 
    t = loader.get_template('template2.html') 
    c = Context({ 
        'app': 'My app', 
        'user': request.user, 
        'ip_address': request.META['REMOTE_ADDR'], 
        'message': 'I am the second view.' 
    }) 
    return t.render(c) 

```

（请注意，在这些示例中，我们故意没有使用`render()`的快捷方式-我们手动加载模板，构建上下文对象并渲染模板。我们为了清晰起见，详细说明了所有步骤。）

每个视图都传递相同的三个变量-`app`，`user`和`ip_address`-到它的模板。如果我们能够消除这种冗余，那不是很好吗？`RequestContext`和上下文处理器被创建来解决这个问题。上下文处理器允许您指定一些变量，这些变量在每个上下文中自动设置-而无需在每个`render()`调用中指定这些变量。

问题在于，当你渲染模板时，你必须使用`RequestContext`而不是`Context`。使用上下文处理器的最低级别方法是创建一些处理器并将它们传递给`RequestContext`。以下是如何使用上下文处理器编写上面的示例：

```py
from django.template import loader, RequestContext 

def custom_proc(request): 
    # A context processor that provides 'app', 'user' and 'ip_address'. 
    return { 
        'app': 'My app', 
        'user': request.user, 
        'ip_address': request.META['REMOTE_ADDR'] 
    } 

def view_1(request): 
    # ... 
    t = loader.get_template('template1.html') 
    c = RequestContext(request,  
                       {'message': 'I am view 1.'},   
                       processors=[custom_proc]) 
    return t.render(c) 

def view_2(request): 
    # ... 
    t = loader.get_template('template2.html') 
    c = RequestContext(request,  
                       {'message': 'I am the second view.'},   
                       processors=[custom_proc]) 
    return t.render(c) 

```

让我们逐步了解这段代码：

+   首先，我们定义一个函数`custom_proc`。这是一个上下文处理器-它接受一个`HttpRequest`对象，并返回一个要在模板上下文中使用的变量字典。就是这样。

+   我们已将两个视图函数更改为使用`RequestContext`而不是`Context`。上下文构造方式有两个不同之处。首先，`RequestContext`要求第一个参数是一个`HttpRequest`对象-首先传递到视图函数中的对象（`request`）。其次，`RequestContext`需要一个可选的`processors`参数，它是要使用的上下文处理器函数的列表或元组。在这里，我们传入`custom_proc`，我们上面定义的自定义处理器。

+   每个视图不再必须在其上下文构造中包含`app`，`user`或`ip_address`，因为这些由`custom_proc`提供。

+   每个视图仍然具有灵活性，可以引入任何可能需要的自定义模板变量。在此示例中，`message`模板变量在每个视图中设置不同。

在第三章*模板*中，我介绍了`render()`快捷方式，它使您无需调用`loader.get_template()`，然后创建一个`Context`，然后在模板上调用`render()`方法。

为了演示上下文处理器的较低级别工作，上面的示例没有使用`render()`。但是，使用`render()`与上下文处理器是可能的，也是更好的。可以使用`context_instance`参数来实现这一点，如下所示：

```py
from django.shortcuts import render 
from django.template import RequestContext 

def custom_proc(request): 
    # A context processor that provides 'app', 'user' and 'ip_address'. 
    return { 
        'app': 'My app', 
        'user': request.user, 
        'ip_address': request.META['REMOTE_ADDR'] 
    } 

def view_1(request): 
    # ... 
    return render(request, 'template1.html', 
                  {'message': 'I am view 1.'}, 
                  context_instance=RequestContext( 
                  request, processors=[custom_proc] 
                  ) 
    ) 

def view_2(request): 
    # ... 
    return render(request, 'template2.html',                  {'message': 'I am the second view.'}, 
                  context_instance=RequestContext( 
                  request, processors=[custom_proc] 
                  ) 
) 

```

在这里，我们已将每个视图的模板渲染代码简化为单个（包装）行。这是一个改进，但是，评估这段代码的简洁性时，我们必须承认我们现在几乎过度使用了另一端的频谱。我们消除了数据中的冗余（我们的模板变量），但增加了代码中的冗余（在`processors`调用中）。

如果您必须一直输入`processors`，使用上下文处理器并不能节省太多输入。因此，Django 提供了全局上下文处理器的支持。`context_processors`设置（在您的`settings.py`中）指定应始终应用于`RequestContext`的上下文处理器。这样可以避免每次使用`RequestContext`时都需要指定`processors`。

默认情况下，`context_processors`设置如下：

```py
'context_processors': [ 
            'django.template.context_processors.debug', 
            'django.template.context_processors.request', 
            'django.contrib.auth.context_processors.auth', 
'django.contrib.messages.context_processors.messages', 
        ], 

```

此设置是一个可调用对象的列表，其接口与上面的`custom_proc`函数相同-接受请求对象作为其参数，并返回要合并到上下文中的项目的字典。请注意，`context_processors`中的值被指定为**字符串**，这意味着处理器必须在 Python 路径的某个位置（因此您可以从设置中引用它们）。

每个处理器都按顺序应用。也就是说，如果一个处理器向上下文添加一个变量，并且第二个处理器使用相同的名称添加一个变量，则第二个处理器将覆盖第一个处理器。Django 提供了许多简单的上下文处理器，包括默认启用的处理器：

## auth

`django.contrib.auth.context_processors.auth`

如果启用了此处理器，则每个`RequestContext`都将包含这些变量：

+   `user`：表示当前登录用户的`auth.User`实例（或`AnonymousUser`实例，如果客户端未登录）。

+   `perms`：表示当前登录用户具有的权限的`django.contrib.auth.context_processors.PermWrapper`实例。

## DEBUG

`django.template.context_processors.debug`

如果启用了此处理器，则每个`RequestContext`都将包含这两个变量-但仅当您的`DEBUG`设置为`True`并且请求的 IP 地址（`request.META['REMOTE_ADDR']`）在`INTERNAL_IPS`设置中时：

+   `debug`-`True`：您可以在模板中使用此选项来测试是否处于`DEBUG`模式。

+   `sql_queries`：一个`{'sql': ..., 'time': ...}`字典的列表，表示请求期间发生的每个 SQL 查询及其所花费的时间。列表按查询顺序生成，并在访问时惰性生成。

## i18n

`django.template.context_processors.i18n`

如果启用了此处理器，则每个`RequestContext`都将包含这两个变量：

+   `LANGUAGES`：`LANGUAGES`设置的值。

+   `LANGUAGE_CODE`：`request.LANGUAGE_CODE`，如果存在的话。否则，为`LANGUAGE_CODE`设置的值。

## 媒体

`django.template.context_processors.media`

如果启用了此处理器，每个`RequestContext`都将包含一个名为`MEDIA_URL`的变量，该变量提供`MEDIA_URL`设置的值。

## 静态

`django.template.context_processors.static`

如果启用了此处理器，每个`RequestContext`都将包含一个名为`STATIC_URL`的变量，该变量提供`STATIC_URL`设置的值。

## csrf

`django.template.context_processors.csrf`

此处理器添加了一个`csrf_token`模板标记所需的令牌，以防止跨站点请求伪造（请参见第十九章，“Django 中的安全性”）。

## 请求

`django.template.context_processors.request`

如果启用了此处理器，每个`RequestContext`都将包含一个名为`request`的变量，该变量是当前的`HttpRequest`。

## 消息

`django.contrib.messages.context_processors.messages`

如果启用了此处理器，每个`RequestContext`都将包含这两个变量：

+   `messages`：已通过消息框架设置的消息（作为字符串）的列表。

+   `DEFAULT_MESSAGE_LEVELS`：消息级别名称与其数值的映射。

# 编写自己的上下文处理器指南

上下文处理器具有非常简单的接口：它只是一个接受一个`HttpRequest`对象的 Python 函数，并返回一个添加到模板上下文中的字典。每个上下文处理器必须返回一个字典。以下是一些编写自己上下文处理器的提示：

+   使每个上下文处理器负责尽可能小的功能子集。使用多个处理器很容易，因此最好将功能拆分为将来重用的逻辑片段。

+   请记住，`TEMPLATE_CONTEXT_PROCESSORS`中的任何上下文处理器都将在由该设置文件提供动力的每个模板中可用，因此请尝试选择与模板可能独立使用的变量名不太可能发生冲突的变量名。由于变量名区分大小写，因此最好使用所有大写字母来表示处理器提供的变量。

+   自定义上下文处理器可以存在于代码库中的任何位置。Django 关心的是您的自定义上下文处理器是否由`TEMPLATES`设置中的`'context_processors'`选项指向，或者如果直接使用`Engine`，则由`Engine`的`context_processors`参数指向。话虽如此，惯例是将它们保存在应用程序或项目中名为`context_processors.py`的文件中。

# 自动 HTML 转义

在从模板生成 HTML 时，总是存在一个变量包含影响生成的 HTML 的字符的风险。例如，考虑这个模板片段：

```py
Hello, {{ name }}. 

```

起初，这似乎是一种无害的显示用户姓名的方式，但请考虑如果用户将他的名字输入为这样会发生什么：

```py
<script>alert('hello')</script> 

```

使用这个名称值，模板将被渲染为：

```py
Hello, <script>alert('hello')</script> 

```

……这意味着浏览器将弹出一个 JavaScript 警报框！同样，如果名称包含`'<'`符号，会怎么样？

```py
<b>username 

```

这将导致渲染的模板如下：

```py
Hello, <b>username 

```

……这将导致网页的其余部分变粗！显然，不应盲目信任用户提交的数据并直接插入到您的网页中，因为恶意用户可能利用这种漏洞做出潜在的坏事。

这种安全漏洞称为跨站脚本（XSS）攻击。（有关安全性的更多信息，请参见第十九章，“Django 中的安全性”）。为了避免这个问题，您有两个选择：

+   首先，您可以确保通过`escape`过滤器运行每个不受信任的变量，该过滤器将潜在有害的 HTML 字符转换为无害的字符。这是 Django 最初几年的默认解决方案，但问题在于它把责任放在了*您*，开发者/模板作者身上，确保您转义了所有内容。很容易忘记转义数据。

+   其次，您可以利用 Django 的自动 HTML 转义。本节的其余部分将描述自动转义的工作原理。

+   在 Django 中，默认情况下，每个模板都会自动转义每个变量标签的输出。具体来说，这五个字符会被转义：

+   `<` 被转换为 `&lt;`

+   `>` 被转换为 `&gt;`

+   `'`（单引号）被转换为`'`

+   `"`（双引号）被转换为`&quot;`

+   `&` 被转换为 `&amp;`

再次强调，这种行为默认情况下是开启的。如果您使用 Django 的模板系统，您就受到了保护。

## 如何关闭它

如果您不希望数据在每个站点、每个模板级别或每个变量级别自动转义，可以通过多种方式关闭它。为什么要关闭它？因为有时模板变量包含您希望呈现为原始 HTML 的数据，这种情况下您不希望它们的内容被转义。

例如，您可能会在数据库中存储一大段受信任的 HTML，并希望直接将其嵌入到模板中。或者，您可能正在使用 Django 的模板系统来生成非 HTML 文本-例如电子邮件消息。

## 对于单个变量

要为单个变量禁用自动转义，请使用`safe`过滤器：

```py
This will be escaped: {{ data }} 
This will not be escaped: {{ data|safe }} 

```

将*safe*视为*免受进一步转义*或*可以安全解释为 HTML*的简写。在这个例子中，如果`data`包含`<b>`，输出将是：

```py
This will be escaped: &lt;b&gt; 
This will not be escaped: <b> 

```

## 对于模板块

要控制模板的自动转义，可以将模板（或模板的特定部分）包装在`autoescape`标签中，如下所示：

```py
{% autoescape off %} 
    Hello {{ name }} 
{% endautoescape %} 

```

`autoescape`标签接受`on`或`off`作为参数。有时，您可能希望在本来被禁用自动转义的情况下强制进行自动转义。以下是一个示例模板：

```py
Auto-escaping is on by default. Hello {{ name }} 

{% autoescape off %} 
    This will not be auto-escaped: {{ data }}. 

    Nor this: {{ other_data }} 
    {% autoescape on %} 
        Auto-escaping applies again: {{ name }} 
    {% endautoescape %} 
{% endautoescape %} 

```

自动转义标签会将其效果传递给扩展当前模板以及通过`include`标签包含的模板，就像所有块标签一样。例如：

```py
# base.html 

{% autoescape off %} 
<h1>{% block title %}{% endblock %}</h1> 
{% block content %} 
{% endblock %} 
{% endautoescape %} 

# child.html 

{% extends "base.html" %} 
{% block title %}This & that{% endblock %} 
{% block content %}{{ greeting }}{% endblock %} 

```

因为基础模板中关闭了自动转义，所以在子模板中也会关闭自动转义，当`greeting`变量包含字符串`<b>Hello!</b>`时，将会产生以下渲染的 HTML：

```py
<h1>This & that</h1> 
<b>Hello!</b> 

```

一般来说，模板作者不需要太担心自动转义。Python 端的开发人员（编写视图和自定义过滤器的人）需要考虑数据不应该被转义的情况，并适当标记数据，以便在模板中正常工作。

如果您正在创建一个可能在您不确定自动转义是否启用的情况下使用的模板，那么请为任何需要转义的变量添加`escape`过滤器。当自动转义开启时，`escape`过滤器不会导致数据双重转义-`escape`过滤器不会影响自动转义的变量。

## 在过滤器参数中自动转义字符串文字

正如我们之前提到的，过滤器参数可以是字符串：

```py
{{ data|default:"This is a string literal." }} 

```

所有字符串文字都会被插入到模板中，而不会进行任何自动转义-它们的行为就好像它们都通过了`safe`过滤器。背后的原因是模板作者控制着字符串文字的内容，因此他们可以确保在编写模板时正确地转义文本。

这意味着您应该这样写

```py
{{ data|default:"3 &lt; 2" }} 

```

...而不是

```py
{{ data|default:"3 < 2" }} <== Bad! Don't do this. 

```

这不会影响来自变量本身的数据。变量的内容仍然会在必要时自动转义，因为它们超出了模板作者的控制。

# 模板加载内部

通常，您会将模板存储在文件系统中，而不是自己使用低级别的`Template` API。将模板保存在指定为模板目录的目录中。 Django 根据您的模板加载设置在许多地方搜索模板目录（请参阅下面的*Loader 类型*），但指定模板目录的最基本方法是使用`DIRS`选项。

## DIRS 选项

通过在设置文件中的`TEMPLATES`设置中使用`DIRS`选项或在`Engine`的`dirs`参数中使用`DIRS`选项，告诉 Django 您的模板目录是什么。这应设置为包含完整路径的字符串列表，以包含模板目录：

```py
TEMPLATES = [ 
    { 
        'BACKEND': 'django.template.backends.django.DjangoTemplates', 
        'DIRS': [ 
            '/home/html/templates/lawrence.com', 
            '/home/html/templates/default', 
        ], 
    }, 
] 

```

您的模板可以放在任何您想要的地方，只要目录和模板对 Web 服务器可读。它们可以具有任何您想要的扩展名，例如`.html`或`.txt`，或者它们可以根本没有扩展名。请注意，这些路径应使用 Unix 样式的正斜杠，即使在 Windows 上也是如此。

## 加载程序类型

默认情况下，Django 使用基于文件系统的模板加载程序，但 Django 还配备了其他几个模板加载程序，它们知道如何从其他来源加载模板；其中最常用的应用程序加载程序将在下面进行描述。

### 文件系统加载程序

`filesystem.Loader`从文件系统加载模板，根据`DIRS <TEMPLATES-DIRS>`。此加载程序默认启用。但是，直到您将`DIRS <TEMPLATES-DIRS>`设置为非空列表之前，它才能找到任何模板：

```py
TEMPLATES = [{ 
    'BACKEND': 'django.template.backends.django.DjangoTemplates', 
    'DIRS': [os.path.join(BASE_DIR, 'templates')], 
}] 

```

### 应用程序目录加载程序

`app_directories.Loader`从文件系统加载 Django 应用程序的模板。对于`INSTALLED_APPS`中的每个应用程序，加载程序都会查找`templates`子目录。如果目录存在，Django 将在其中查找模板。这意味着您可以将模板与各个应用程序一起存储。这也使得很容易使用默认模板分发 Django 应用程序。例如，对于此设置：

```py
INSTALLED_APPS = ['myproject.reviews', 'myproject.music'] 

```

`get_template('foo.html')`将按照这些顺序在这些目录中查找`foo.html`：

+   `/path/to/myproject/reviews/templates/`

+   `/path/to/myproject/music/templates/`

并使用它找到的第一个。

**INSTALLED_APPS 的顺序很重要！**

例如，如果您想要自定义 Django 管理界面，您可能会选择使用自己的`myproject.reviews`中的`admin/base_site.html`覆盖标准的`admin/base_site.html`模板，而不是使用`django.contrib.admin`。

然后，您必须确保`myproject.reviews`在`INSTALLED_APPS`中出现在`django.contrib.admin`之前，否则将首先加载`django.contrib.admin`，并且您的将被忽略。

请注意，加载程序在首次运行时执行优化：它缓存了具有`templates`子目录的`INSTALLED_APPS`包的列表。

您只需将`APP_DIRS`设置为`True`即可启用此加载程序：

```py
TEMPLATES = [{ 
    'BACKEND': 'django.template.backends.django.DjangoTemplates', 
    'APP_DIRS': True, 
}] 

```

### 其他加载程序

其余的模板加载程序是：

+   `django.template.loaders.eggs.Loader`

+   `django.template.loaders.cached.Loader`

+   `django.template.loaders.locmem.Loader`

这些加载程序默认情况下是禁用的，但是您可以通过在`TEMPLATES`设置中的`DjangoTemplates`后端中添加`loaders`选项或将`loaders`参数传递给`Engine`来激活它们。有关这些高级加载程序的详细信息，以及构建自己的自定义加载程序，可以在 Django 项目网站上找到。

# 扩展模板系统

现在您对模板系统的内部工作有了更多了解，让我们看看如何使用自定义代码扩展系统。大多数模板定制以自定义模板标签和/或过滤器的形式出现。尽管 Django 模板语言带有许多内置标签和过滤器，但您可能会组装自己的标签和过滤器库，以满足自己的需求。幸运的是，定义自己的功能非常容易。

## 代码布局

自定义模板标签和过滤器必须位于 Django 应用程序中。如果它们与现有应用程序相关，将它们捆绑在那里是有意义的；否则，您应该创建一个新的应用程序来保存它们。该应用程序应该包含一个`templatetags`目录，与`models.py`、`views.py`等文件处于同一级别。如果这个目录还不存在，请创建它-不要忘记`__init__.py`文件，以确保该目录被视为 Python 包。

添加此模块后，您需要在使用模板中的标签或过滤器之前重新启动服务器。您的自定义标签和过滤器将位于`templatetags`目录中的一个模块中。

模块文件的名称是您以后将用来加载标签的名称，因此要小心选择一个不会与另一个应用程序中的自定义标签和过滤器冲突的名称。

例如，如果您的自定义标签/过滤器在名为`review_extras.py`的文件中，您的应用程序布局可能如下所示：

```py
reviews/ 
    __init__.py 
    models.py 
    templatetags/ 
        __init__.py 
        review_extras.py 
    views.py 

```

在您的模板中，您将使用以下内容：

```py
{% load review_extras %} 

```

包含自定义标签的应用程序必须在`INSTALLED_APPS`中，以便`{% load %}`标签能够工作。

### 注意

**幕后**

要获取大量示例，请阅读 Django 默认过滤器和标签的源代码。它们分别位于`django/template/defaultfilters.py`和`django/template/defaulttags.py`中。有关`load`标签的更多信息，请阅读其文档。

## 创建模板库

无论您是编写自定义标签还是过滤器，首先要做的是创建一个**模板库**-这是 Django 可以连接到的一小部分基础设施。

创建模板库是一个两步过程：

+   首先，决定哪个 Django 应用程序应该包含模板库。如果您通过`manage.py startapp`创建了一个应用程序，您可以将其放在那里，或者您可以创建另一个仅用于模板库的应用程序。我们建议选择后者，因为您的过滤器可能对将来的项目有用。无论您选择哪种路线，请确保将应用程序添加到您的`INSTALLED_APPS`设置中。我马上会解释这一点。

+   其次，在适当的 Django 应用程序包中创建一个`templatetags`目录。它应该与`models.py`、`views.py`等文件处于同一级别。例如：

```py
        books/
        __init__.py
        models.py
        templatetags/
        views.py
```

在`templatetags`目录中创建两个空文件：一个`__init__.py`文件（表示这是一个包含 Python 代码的包）和一个包含自定义标签/过滤器定义的文件。后者的文件名是您以后将用来加载标签的名称。例如，如果您的自定义标签/过滤器在名为`review_extras.py`的文件中，您可以在模板中写入以下内容：

```py
{% load review_extras %} 

```

`{% load %}`标签查看您的`INSTALLED_APPS`设置，并且只允许加载已安装的 Django 应用程序中的模板库。这是一个安全功能；它允许您在单台计算机上托管许多模板库的 Python 代码，而不会为每个 Django 安装启用对所有模板库的访问。

如果您编写的模板库与任何特定的模型/视图无关，那么拥有一个仅包含`templatetags`包的 Django 应用程序包是有效的和非常正常的。

在`templatetags`包中放置多少模块都没有限制。只需记住，`{% load %}`语句将加载给定 Python 模块名称的标签/过滤器，而不是应用程序的名称。

创建了该 Python 模块后，您只需根据您是编写过滤器还是标签来编写一些 Python 代码。要成为有效的标签库，模块必须包含一个名为`register`的模块级变量，它是`template.Library`的实例。

这是所有标签和过滤器注册的数据结构。因此，在您的模块顶部附近，插入以下内容：

```py
from django import template 
register = template.Library() 

```

# 自定义模板标签和过滤器

Django 的模板语言配备了各种内置标签和过滤器，旨在满足应用程序的呈现逻辑需求。尽管如此，您可能会发现自己需要的功能不在核心模板原语集中。

您可以通过使用 Python 定义自定义标签和过滤器来扩展模板引擎，然后使用`{% load %}`标签将其提供给模板。

## 编写自定义模板过滤器

自定义过滤器只是接受一个或两个参数的 Python 函数：

+   变量的值（输入）-不一定是一个字符串。

+   参数的值-这可以有一个默认值，或者完全省略。

例如，在过滤器`{{ var|foo:"bar" }}`中，过滤器`foo`将接收变量`var`和参数`"bar"`。由于模板语言不提供异常处理，从模板过滤器引发的任何异常都将暴露为服务器错误。

因此，如果有一个合理的回退值可以返回，过滤函数应该避免引发异常。在模板中表示明显错误的输入情况下，引发异常可能仍然比隐藏错误的静默失败更好。这是一个示例过滤器定义：

```py
def cut(value, arg): 
    """Removes all values of arg from the given string""" 
    return value.replace(arg, '') 

```

以下是该过滤器的使用示例：

```py
{{ somevariable|cut:"0" }} 

```

大多数过滤器不带参数。在这种情况下，只需在函数中省略参数。例如：

```py
def lower(value): # Only one argument. 
    """Converts a string into all lowercase""" 
    return value.lower() 

```

### 注册自定义过滤器

编写完过滤器定义后，您需要将其注册到您的`Library`实例中，以使其可用于 Django 的模板语言：

```py
register.filter('cut', cut) 
register.filter('lower', lower) 

```

`Library.filter()`方法接受两个参数：

1.  过滤器的名称-一个字符串。

1.  编译函数-一个 Python 函数（而不是函数的名称作为字符串）。

您可以将`register.filter()`用作装饰器：

```py
@register.filter(name='cut') 
def cut(value, arg): 
    return value.replace(arg, '') 

@register.filter 
def lower(value): 
    return value.lower() 

```

如果省略`name`参数，就像上面的第二个示例一样，Django 将使用函数的名称作为过滤器名称。最后，`register.filter()`还接受三个关键字参数，`is_safe`，`needs_autoescape`和`expects_localtime`。这些参数在下面的过滤器和自动转义以及过滤器和时区中进行了描述。

### 期望字符串的模板过滤器

如果您正在编写一个模板过滤器，只期望第一个参数是字符串，您应该使用装饰器`stringfilter`。这将在将对象传递给您的函数之前将其转换为其字符串值：

```py
from django import template 
from django.template.defaultfilters import stringfilter 

register = template.Library() 

@register.filter 
@stringfilter 
def lower(value): 
    return value.lower() 

```

这样，您就可以将一个整数传递给这个过滤器，它不会引起`AttributeError`（因为整数没有`lower()`方法）。

### 过滤器和自动转义

在编写自定义过滤器时，要考虑过滤器将如何与 Django 的自动转义行为交互。请注意，在模板代码中可以传递三种类型的字符串：

+   **原始字符串**是本机 Python `str`或`unicode`类型。在输出时，如果自动转义生效，它们会被转义并保持不变，否则。

+   **安全字符串**是在输出时已标记为免受进一步转义的字符串。任何必要的转义已经完成。它们通常用于包含原始 HTML 的输出，该 HTML 旨在在客户端上按原样解释。

+   在内部，这些字符串的类型是`SafeBytes`或`SafeText`。它们共享一个名为`SafeData`的基类，因此您可以使用类似的代码对它们进行测试：

+   如果`value`是`SafeData`的实例：

```py
        # Do something with the "safe" string.
        ...
```

+   **标记为“需要转义”的字符串**在输出时始终会被转义，无论它们是否在`autoescape`块中。但是，这些字符串只会被转义一次，即使自动转义适用。

在内部，这些字符串的类型是`EscapeBytes`或`EscapeText`。通常，您不必担心这些问题；它们存在是为了实现`escape`过滤器。

模板过滤器代码分为两种情况：

1.  您的过滤器不会在结果中引入任何 HTML 不安全的字符（`<`，`>`，`'`，`"`或`&`），这些字符在结果中本来就存在；或

1.  或者，您的过滤器代码可以手动处理任何必要的转义。当您将新的 HTML 标记引入结果时，这是必要的。

在第一种情况下，您可以让 Django 为您处理所有自动转义处理。您只需要在注册过滤器函数时将`is_safe`标志设置为`True`，如下所示：

```py
@register.filter(is_safe=True)
def myfilter(value):
    return value

```

这个标志告诉 Django，如果将安全字符串传递到您的过滤器中，则结果仍将是安全的，如果传递了不安全的字符串，则 Django 将自动转义它（如果需要的话）。您可以将其视为意味着“此过滤器是安全的-它不会引入任何不安全的 HTML 可能性。”

`is_safe`之所以必要是因为有很多普通的字符串操作会将`SafeData`对象转换回普通的`str`或`unicode`对象，而不是尝试捕获它们所有，这将非常困难，Django 会在过滤器完成后修复损坏。

例如，假设您有一个过滤器，它将字符串`xx`添加到任何输入的末尾。由于这不会向结果引入危险的 HTML 字符（除了已经存在的字符），因此应该使用`is_safe`标记过滤器：

```py
@register.filter(is_safe=True) 
def add_xx(value): 
    return '%sxx' % value 

```

当在启用自动转义的模板中使用此过滤器时，Django 将在输入未标记为安全时转义输出。默认情况下，`is_safe`为`False`，并且您可以在任何不需要的过滤器中省略它。在决定您的过滤器是否确实将安全字符串保持为安全时要小心。如果您删除字符，可能会无意中在结果中留下不平衡的 HTML 标记或实体。

例如，从输入中删除`>`可能会将`<a>`变为`<a`，这需要在输出时进行转义，以避免引起问题。同样，删除分号（`;`）可能会将`&amp;`变为`&amp`，这不再是一个有效的实体，因此需要进一步转义。大多数情况下不会有这么棘手，但是在审查代码时要注意任何类似的问题。

标记过滤器`is_safe`将强制过滤器的返回值为字符串。如果您的过滤器应返回布尔值或其他非字符串值，则将其标记为`is_safe`可能会产生意想不到的后果（例如将布尔值`False`转换为字符串`False`）。

在第二种情况下，您希望标记输出为安全，以免进一步转义您的 HTML 标记，因此您需要自己处理输入。要将输出标记为安全字符串，请使用`django.utils.safestring.mark_safe()`。

不过要小心。您需要做的不仅仅是标记输出为安全。您需要确保它确实是安全的，您的操作取决于自动转义是否生效。

这个想法是编写可以在模板中运行的过滤器，无论自动转义是打开还是关闭，以便为模板作者简化事情。

为了使您的过滤器知道当前的自动转义状态，请在注册过滤器函数时将`needs_autoescape`标志设置为`True`。（如果您不指定此标志，它将默认为`False`）。这个标志告诉 Django，您的过滤器函数希望传递一个额外的关键字参数，称为`autoescape`，如果自动转义生效，则为`True`，否则为`False`。

例如，让我们编写一个过滤器，强调字符串的第一个字符：

```py
from django import template 
from django.utils.html import conditional_escape 
from django.utils.safestring import mark_safe 

register = template.Library() 

@register.filter(needs_autoescape=True) 
def initial_letter_filter(text, autoescape=None): 
    first, other = text[0], text[1:] 
    if autoescape: 
        esc = conditional_escape 
    else: 
        esc = lambda x: x 
    result = '<strong>%s</strong>%s' % (esc(first), esc(other)) 
    return mark_safe(result) 

```

`needs_autoescape`标志和`autoescape`关键字参数意味着我们的函数将知道在调用过滤器时是否自动转义。我们使用`autoescape`来决定输入数据是否需要通过`django.utils.html.conditional_escape`传递。 （在后一种情况下，我们只使用身份函数作为“转义”函数。）

`conditional_escape()`函数类似于`escape()`，只是它只转义**不是**`SafeData`实例的输入。如果将`SafeData`实例传递给`conditional_escape()`，则数据将保持不变。

最后，在上面的例子中，我们记得将结果标记为安全，以便我们的 HTML 直接插入模板而不需要进一步转义。在这种情况下，不需要担心 `is_safe` 标志（尽管包含它也不会有什么坏处）。每当您手动处理自动转义问题并返回安全字符串时，`is_safe` 标志也不会改变任何东西。

### 过滤器和时区

如果您编写一个在 `datetime` 对象上操作的自定义过滤器，通常会将其注册为 `expects_localtime` 标志设置为 `True`：

```py
@register.filter(expects_localtime=True) 
def businesshours(value): 
    try: 
        return 9 <= value.hour < 17 
    except AttributeError: 
        return '' 

```

当设置了此标志时，如果您的过滤器的第一个参数是时区感知的日期时间，则 Django 会根据模板中的时区转换规则在适当时将其转换为当前时区后传递给您的过滤器。

### 注意

**在重用内置过滤器时避免 XSS 漏洞**

在重用 Django 的内置过滤器时要小心。您需要向过滤器传递 `autoescape=True` 以获得正确的自动转义行为，并避免跨站脚本漏洞。例如，如果您想编写一个名为 `urlize_and_linebreaks` 的自定义过滤器，该过滤器结合了 `urlize` 和 `linebreaksbr` 过滤器，那么过滤器将如下所示：

`from django.template.defaultfilters import linebreaksbr, urlize` `@register.filter` `def urlize_and_linebreaks(text):` `return linebreaksbr(` `urlize(text, autoescape=True),autoescape=True)` `然后：` `{{ comment|urlize_and_linebreaks }}` `等同于：` `{{ comment|urlize|linebreaksbr }}`

## 编写自定义模板标签

标签比过滤器更复杂，因为标签可以做任何事情。Django 提供了许多快捷方式，使编写大多数类型的标签更容易。首先我们将探讨这些快捷方式，然后解释如何为那些快捷方式不够强大的情况下从头编写标签。

### 简单标签

许多模板标签需要一些参数-字符串或模板变量-并且在仅基于输入参数和一些外部信息进行一些处理后返回结果。

例如，`current_time` 标签可能接受一个格式字符串，并根据格式化返回时间字符串。为了简化这些类型的标签的创建，Django 提供了一个辅助函数 `simple_tag`。这个函数是 `django.template.Library` 的一个方法，它接受一个接受任意数量参数的函数，将其包装在一个 `render` 函数和其他必要的部分中，并将其注册到模板系统中。

我们的 `current_time` 函数可以这样编写：

```py
import datetime 
from django import template 

register = template.Library() 

@register.simple_tag 
def current_time(format_string): 
    return datetime.datetime.now().strftime(format_string) 

```

关于 `simple_tag` 辅助函数的一些注意事项：

+   在我们的函数被调用时，已经检查了所需数量的参数等，所以我们不需要再做这些。

+   参数（如果有）周围的引号已经被剥离，所以我们只收到一个普通字符串。

+   如果参数是模板变量，则我们的函数会传递变量的当前值，而不是变量本身。

如果您的模板标签需要访问当前上下文，可以在注册标签时使用 `takes_context` 参数：

```py
@register.simple_tag(takes_context=True) 
def current_time(context, format_string): 
    timezone = context['timezone'] 
    return your_get_current_time_method(timezone, format_string) 

```

请注意，第一个参数必须称为 `context`。有关 `takes_context` 选项的工作原理的更多信息，请参阅包含标签部分。如果您需要重命名标签，可以为其提供自定义名称：

```py
register.simple_tag(lambda x: x-1, name='minusone') 

@register.simple_tag(name='minustwo') 
def some_function(value): 
    return value-2 

```

`simple_tag` 函数可以接受任意数量的位置参数或关键字参数。例如：

```py
@register.simple_tag 
def my_tag(a, b, *args, **kwargs): 
    warning = kwargs['warning'] 
    profile = kwargs['profile'] 
    ... 
    return ... 

```

然后在模板中，可以传递任意数量的参数，用空格分隔，到模板标签。就像在 Python 中一样，关键字参数的值使用等号（“=`”）设置，并且必须在位置参数之后提供。例如：

```py
{% my_tag 123 "abcd" book.title warning=message|lower profile=user.profile %} 

```

### 包含标签

另一种常见的模板标签类型是通过呈现另一个模板来显示一些数据的类型。例如，Django 的管理界面使用自定义模板标签来显示“添加/更改”表单页面底部的按钮。这些按钮始终看起来相同，但链接目标会根据正在编辑的对象而变化-因此它们是使用填充了当前对象详细信息的小模板的完美案例。（在管理界面的情况下，这是`submit_row`标签。）

这些类型的标签被称为包含标签。编写包含标签最好通过示例来演示。让我们编写一个为给定的`Author`对象生成书籍列表的标签。我们将像这样使用该标签：

```py
{% books_for_author author %} 

```

结果将会是这样的：

```py
<ul> 
    <li>The Cat In The Hat</li> 
    <li>Hop On Pop</li> 
    <li>Green Eggs And Ham</li> 
</ul> 

```

首先，我们定义一个接受参数并生成结果数据字典的函数。请注意，我们只需要返回一个字典，而不是更复杂的内容。这将用作模板片段的上下文：

```py
def books_for_author(author): 
    books = Book.objects.filter(authors__id=author.id) 
    return {'books': books} 

```

接下来，我们创建用于呈现标签输出的模板。根据我们的示例，模板非常简单：

```py
<ul> 
{% for book in books %}<li>{{ book.title }}</li> 
{% endfor %} 
</ul> 

```

最后，我们通过在`Library`对象上调用`inclusion_tag()`方法来创建和注册包含标签。根据我们的示例，如果前面的模板在模板加载器搜索的目录中的名为`book_snippet.html`的文件中，我们可以像这样注册标签：

```py
# Here, register is a django.template.Library instance, as before 
@register.inclusion_tag('book_snippet.html') 
def show_reviews(review): 
    ... 

```

或者，可以在首次创建函数时使用`django.template.Template`实例注册包含标签：

```py
from django.template.loader import get_template 
t = get_template('book_snippet.html') 
register.inclusion_tag(t)(show_reviews) 

```

有时，你的包含标签可能需要大量的参数，这使得模板作者很难传递所有参数并记住它们的顺序。为了解决这个问题，Django 为包含标签提供了一个`takes_context`选项。如果在创建包含标签时指定了`takes_context`，则该标签将不需要必需的参数，而底层的 Python 函数将有一个参数：调用标签时的模板上下文。例如，假设你正在编写一个包含标签，它将始终在包含`home_link`和`home_title`变量指向主页的上下文中使用。下面是 Python 函数的样子：

```py
@register.inclusion_tag('link.html', takes_context=True) 
def jump_link(context): 
    return { 
        'link': context['home_link'], 
        'title': context['home_title'], 
    } 

```

（请注意，函数的第一个参数必须称为`context`。）模板`link.html`可能包含以下内容：

```py
Jump directly to <a href="{{ link }}">{{ title }}</a>. 

```

然后，每当你想要使用该自定义标签时，加载它的库并在没有任何参数的情况下调用它，就像这样：

```py
{% jump_link %} 

```

请注意，当使用`takes_context=True`时，无需向模板标签传递参数。它会自动访问上下文。`takes_context`参数默认为`False`。当设置为`True`时，标签将传递上下文对象，就像这个例子一样。这是这种情况和之前的`inclusion_tag`示例之间的唯一区别。像`simple_tag`一样，`inclusion_tag`函数也可以接受任意数量的位置或关键字参数。

### 分配标签

为了简化设置上下文变量的标签创建，Django 提供了一个辅助函数`assignment_tag`。这个函数的工作方式与`simple_tag()`相同，只是它将标签的结果存储在指定的上下文变量中，而不是直接输出它。因此，我们之前的`current_time`函数可以这样编写：

```py
@register.assignment_tag 
def get_current_time(format_string): 
    return datetime.datetime.now().strftime(format_string) 

```

然后，你可以使用`as`参数将结果存储在模板变量中，并在适当的位置输出它：

```py
{% get_current_time "%Y-%m-%d %I:%M %p" as the_time %} 
<p>The time is {{ the_time }}.</p> 

```

# 高级自定义模板标签

有时，创建自定义模板标签的基本功能不够。别担心，Django 让你完全访问所需的内部部分，从头开始构建模板标签。

## 快速概述

模板系统以两步过程工作：编译和渲染。要定义自定义模板标签，您需要指定编译如何工作以及渲染如何工作。当 Django 编译模板时，它将原始模板文本分割为节点。每个节点都是`django.template.Node`的一个实例，并且具有`render()`方法。编译的模板就是`Node`对象的列表。

当您在编译的模板对象上调用`render()`时，模板会在其节点列表中的每个`Node`上调用`render()`，并提供给定的上下文。结果都被连接在一起形成模板的输出。因此，要定义一个自定义模板标签，您需要指定原始模板标签如何转换为`Node`（编译函数），以及节点的`render()`方法的作用。

## 编写编译函数

对于模板解析器遇到的每个模板标签，它都会调用一个 Python 函数，该函数具有标签内容和解析器对象本身。此函数负责根据标签的内容返回一个基于`Node`的实例。例如，让我们编写一个我们简单模板标签`{% current_time %}`的完整实现，它显示当前日期/时间，根据标签中给定的参数以`strftime()`语法格式化。在任何其他事情之前，决定标签语法是一个好主意。在我们的情况下，让我们说标签应该像这样使用：

```py
<p>The time is {% current_time "%Y-%m-%d %I:%M %p" %}.</p> 

```

此函数的解析器应该抓取参数并创建一个`Node`对象：

```py
from django import template 

def do_current_time(parser, token): 
    try: 

      tag_name, format_string = token.split_contents() 

    except ValueError: 

      raise template.TemplateSyntaxError("%r tag requires a single  argument" % token.contents.split()[0]) 

   if not (format_string[0] == format_string[-1] and format_string[0]  in ('"', "'")): 
        raise template.TemplateSyntaxError("%r tag's argument should  be in quotes" % tag_name) 
   return CurrentTimeNode(format_string[1:-1]) 

```

**注意：**

+   `parser`是模板解析器对象。在这个例子中我们不需要它。

+   `token.contents`是标签的原始内容的字符串。在我们的例子中，它是`'current_time "%Y-%m-%d %I:%M %p"'`。

+   `token.split_contents()`方法将参数在空格上分开，同时保持引号括起的字符串在一起。更直接的`token.contents.split()`不会那么健壮，因为它会简单地在所有空格上分割，包括引号括起的字符串中的空格。始终使用`token.split_contents()`是一个好主意。

+   此函数负责为任何语法错误引发`django.template.TemplateSyntaxError`，并提供有用的消息。

+   `TemplateSyntaxError`异常使用`tag_name`变量。不要在错误消息中硬编码标签的名称，因为这会将标签的名称与您的函数耦合在一起。`token.contents.split()[0]`将始终是您的标签的名称-即使标签没有参数。

+   该函数返回一个`CurrentTimeNode`，其中包含有关此标签的所有节点需要知道的信息。在这种情况下，它只传递参数`"%Y-%m-%d %I:%M %p"`。模板标签中的前导和尾随引号在`format_string[1:-1]`中被移除。

+   解析是非常低级的。Django 开发人员尝试使用诸如 EBNF 语法之类的技术在此解析系统之上编写小型框架，但这些实验使模板引擎变得太慢。它是低级的，因为这是最快的。

## 编写渲染器

编写自定义标签的第二步是定义一个具有`render()`方法的`Node`子类。继续上面的例子，我们需要定义`CurrentTimeNode`：

```py
import datetime 
from django import template 

class CurrentTimeNode(template.Node): 
    def __init__(self, format_string): 
        self.format_string = format_string 

    def render(self, context): 
        return datetime.datetime.now().strftime(self.format_string) 

```

**注意：**

+   `__init__()`从`do_current_time()`获取`format_string`。始终通过`__init__()`向`Node`传递任何选项/参数/参数。

+   `render()`方法是实际工作发生的地方。

+   `render()`通常应该在生产环境中静默失败，特别是在`DEBUG`和`TEMPLATE_DEBUG`为`False`的情况下。然而，在某些情况下，特别是如果`TEMPLATE_DEBUG`为`True`，此方法可能会引发异常以便更容易进行调试。例如，如果几个核心标签接收到错误数量或类型的参数，它们会引发`django.template.TemplateSyntaxError`。

最终，编译和渲染的解耦导致了一个高效的模板系统，因为一个模板可以渲染多个上下文而不必多次解析。

## 自动转义注意事项

模板标签的输出**不会**自动通过自动转义过滤器运行。但是，在编写模板标签时，仍然有一些事项需要牢记。如果模板的`render()`函数将结果存储在上下文变量中（而不是以字符串返回结果），则应在适当时调用`mark_safe()`。最终呈现变量时，它将受到当时生效的自动转义设置的影响，因此需要将应该免受进一步转义的内容标记为这样。

此外，如果模板标签为执行某些子呈现创建新的上下文，请将自动转义属性设置为当前上下文的值。`Context`类的`__init__`方法接受一个名为`autoescape`的参数，您可以用于此目的。例如：

```py
from django.template import Context 

def render(self, context): 
    # ... 
    new_context = Context({'var': obj}, autoescape=context.autoescape) 
    # ... Do something with new_context ... 

```

这不是一个非常常见的情况，但如果您自己呈现模板，则会很有用。例如：

```py
def render(self, context): 
    t = context.template.engine.get_template('small_fragment.html') 
    return t.render(Context({'var': obj}, autoescape=context.autoescape)) 

```

如果在此示例中忽略了将当前`context.autoescape`值传递给我们的新`Context`，则结果将始终自动转义，这可能不是在模板标签用于内部时所期望的行为。

`{% autoescape off %}`块。

## 线程安全考虑

一旦解析了节点，就可以调用其`render`方法任意次数。由于 Django 有时在多线程环境中运行，单个节点可能会同时响应两个独立请求的不同上下文进行呈现。

因此，确保模板标签是线程安全的非常重要。为确保模板标签是线程安全的，不应在节点本身上存储状态信息。例如，Django 提供了内置的`cycle`模板标签，每次呈现时在给定字符串列表中循环：

```py
{% for o in some_list %} 
    <tr class="{% cycle 'row1' 'row2' %}> 
        ... 
    </tr> 
{% endfor %} 

```

`CycleNode`的一个天真的实现可能如下所示：

```py
import itertools 
from django import template 

class CycleNode(template.Node): 
    def __init__(self, cyclevars): 
        self.cycle_iter = itertools.cycle(cyclevars) 

    def render(self, context): 
        return next(self.cycle_iter) 

Thread 1 performs its first loop iteration, `CycleNode.render()` returns 'row1'Thread 2 performs its first loop iteration, `CycleNode.render()` returns 'row2'Thread 1 performs its second loop iteration, `CycleNode.render()` returns 'row1'Thread 2 performs its second loop iteration, `CycleNode.render()` returns 'row2'
```

CycleNode 正在迭代，但它是全局迭代的。就线程 1 和线程 2 而言，它总是返回相同的值。这显然不是我们想要的！

为了解决这个问题，Django 提供了一个`render_context`，它与当前正在呈现的模板的`context`相关联。`render_context`的行为类似于 Python 字典，并且应该用于在`render`方法的调用之间存储`Node`状态。让我们重构我们的`CycleNode`实现以使用`render_context`：

```py
class CycleNode(template.Node): 
    def __init__(self, cyclevars): 
        self.cyclevars = cyclevars 

    def render(self, context): 
        if self not in context.render_context: 
            context.render_context[self] =  itertools.cycle(self.cyclevars) 
        cycle_iter = context.render_context[self] 
        return next(cycle_iter) 

```

请注意，将全局信息存储为`Node`生命周期内不会更改的属性是完全安全的。

在`CycleNode`的情况下，`cyclevars`参数在`Node`实例化后不会改变，因此我们不需要将其放入`render_context`中。但是，特定于当前正在呈现的模板的状态信息，例如`CycleNode`的当前迭代，应存储在`render_context`中。

## 注册标签

最后，按照上面“编写自定义模板过滤器”的说明，使用模块的`Library`实例注册标签。例如：

```py
register.tag('current_time', do_current_time) 

```

`tag()`方法接受两个参数：

+   模板标签的名称-一个字符串。如果不写，将使用编译函数的名称。

+   编译函数-一个 Python 函数（而不是函数的名称作为字符串）。

与过滤器注册一样，也可以将其用作装饰器：

```py
@register.tag(name="current_time") 
def do_current_time(parser, token): 
    ... 

@register.tag 
def shout(parser, token): 
    ... 

```

如果省略`name`参数，就像上面的第二个示例一样，Django 将使用函数的名称作为标签名称。

## 将模板变量传递给标签

尽管可以使用`token.split_contents()`将任意数量的参数传递给模板标签，但这些参数都会被解包为字符串文字。为了将动态内容（模板变量）作为参数传递给模板标签，需要进行更多的工作。

虽然前面的示例已将当前时间格式化为字符串并返回字符串，但假设您想要传递来自对象的`DateTimeField`并使模板标签格式化该日期时间：

```py
<p>This post was last updated at {% format_time blog_entry.date_updated "%Y-%m-%d %I:%M %p" %}.</p> 

```

最初，`token.split_contents()`将返回三个值：

1.  标签名称`format_time`。

1.  字符串`'blog_entry.date_updated'`（不包括周围的引号）。

1.  格式化字符串`'"%Y-%m-%d %I:%M %p"'`。`split_contents()`的返回值将包括字符串字面量的前导和尾随引号。

现在您的标签应该开始看起来像这样：

```py
from django import template 

def do_format_time(parser, token): 
    try: 
        # split_contents() knows not to split quoted strings. 
        tag_name, date_to_be_formatted, format_string =    
        token.split_contents() 
    except ValueError: 
        raise template.TemplateSyntaxError("%r tag requires exactly  
          two arguments" % token.contents.split()[0]) 
    if not (format_string[0] == format_string[-1] and   
          format_string[0] in ('"', "'")): 
        raise template.TemplateSyntaxError("%r tag's argument should  
          be in quotes" % tag_name) 
    return FormatTimeNode(date_to_be_formatted, format_string[1:-1]) 

```

您还需要更改渲染器以检索`blog_entry`对象的`date_updated`属性的实际内容。这可以通过在`django.template`中使用`Variable()`类来实现。

要使用`Variable`类，只需使用要解析的变量的名称对其进行实例化，然后调用`variable.resolve(context)`。例如：

```py
class FormatTimeNode(template.Node): 
    def __init__(self, date_to_be_formatted, format_string): 
        self.date_to_be_formatted =   
          template.Variable(date_to_be_formatted) 
        self.format_string = format_string 

    def render(self, context): 
        try: 
            actual_date = self.date_to_be_formatted.resolve(context) 
            return actual_date.strftime(self.format_string) 
        except template.VariableDoesNotExist: 
            return '' 

```

如果无法在页面的当前上下文中解析传递给它的字符串，变量解析将抛出`VariableDoesNotExist`异常。

## 在上下文中设置一个变量

上述示例只是简单地输出一个值。通常，如果您的模板标签设置模板变量而不是输出值，那么它会更灵活。这样，模板作者可以重用模板标签创建的值。要在上下文中设置一个变量，只需在`render()`方法中对上下文对象进行字典赋值。这是一个更新后的`CurrentTimeNode`版本，它设置了一个模板变量`current_time`而不是输出它：

```py
import datetime 
from django import template 

class CurrentTimeNode2(template.Node): 
    def __init__(self, format_string): 
        self.format_string = format_string 
    def render(self, context): 
        context['current_time'] = 
 datetime.datetime.now().strftime(self.format_string)
 return ''

```

请注意，`render()`返回空字符串。`render()`应始终返回字符串输出。如果模板标签所做的只是设置一个变量，`render()`应返回空字符串。以下是如何使用标签的新版本：

```py
{% current_time "%Y-%M-%d %I:%M %p" %} 
<p>The time is {{ current_time }}.</p> 

```

### 上下文中的变量范围

上下文中设置的任何变量只能在分配它的模板的相同`block`中使用。这种行为是有意的；它为变量提供了一个作用域，使它们不会与其他块中的上下文发生冲突。

但是，`CurrentTimeNode2`存在一个问题：变量名`current_time`是硬编码的。这意味着您需要确保您的模板不使用

`{{ current_time }}`在其他任何地方，因为`{% current_time %}`将盲目地覆盖该变量的值。

更清晰的解决方案是让模板标签指定输出变量的名称，如下所示：

```py
{% current_time "%Y-%M-%d %I:%M %p" as my_current_time %} 
<p>The current time is {{ my_current_time }}.</p> 

```

为此，您需要重构编译函数和`Node`类，如下所示：

```py
import re 

class CurrentTimeNode3(template.Node): 
    def __init__(self, format_string, var_name): 
        self.format_string = format_string 
        self.var_name = var_name 
    def render(self, context): 
        context[self.var_name] =    
          datetime.datetime.now().strftime(self.format_string) 
        return '' 

def do_current_time(parser, token): 
    # This version uses a regular expression to parse tag contents. 
    try: 
        # Splitting by None == splitting by spaces. 
        tag_name, arg = token.contents.split(None, 1) 
    except ValueError: 
        raise template.TemplateSyntaxError("%r tag requires arguments"    
          % token.contents.split()[0]) 
    m = re.search(r'(.*?) as (\w+)', arg) 
    if not m: 
        raise template.TemplateSyntaxError
          ("%r tag had invalid arguments"% tag_name) 
    format_string, var_name = m.groups() 
    if not (format_string[0] == format_string[-1] and format_string[0]   
       in ('"', "'")): 
        raise template.TemplateSyntaxError("%r tag's argument should be  
            in quotes" % tag_name) 
    return CurrentTimeNode3(format_string[1:-1], var_name) 

```

这里的区别在于`do_current_time()`获取格式字符串和变量名，并将两者都传递给`CurrentTimeNode3`。最后，如果您只需要为自定义上下文更新模板标签使用简单的语法，您可能希望考虑使用我们上面介绍的赋值标签快捷方式。

## 解析直到另一个块标签

模板标签可以协同工作。例如，标准的`{% comment %}`标签隐藏直到`{% endcomment %}`。要创建这样一个模板标签，可以在编译函数中使用`parser.parse()`。以下是一个简化的示例

`{% comment %}`标签可能被实现：

```py
def do_comment(parser, token): 
    nodelist = parser.parse(('endcomment',)) 
    parser.delete_first_token() 
    return CommentNode() 

class CommentNode(template.Node): 
    def render(self, context): 
        return '' 

```

### 注意

`{% comment %}`的实际实现略有不同，它允许在`{% comment %}`和`{% endcomment %}`之间出现损坏的模板标签。它通过调用`parser.skip_past('endcomment')`而不是`parser.parse(('endcomment',))`，然后是`parser.delete_first_token()`来实现这一点，从而避免生成节点列表。

`parser.parse()`接受一个块标签名称的元组''直到解析''。它返回`django.template.NodeList`的一个实例，这是解析器在遇到元组中命名的任何标签之前''遇到''的所有`Node`对象的列表。在上面的示例中的"`nodelist = parser.parse(('endcomment',))`"中，`nodelist`是`{% comment %}`和`{% endcomment %}`之间的所有节点的列表，不包括

`{% comment %}`和`{% endcomment %}`本身。

在调用`parser.parse()`之后，解析器尚未“消耗”

`{% endcomment %}`标签，所以代码需要显式调用`parser.delete_first_token()`。`CommentNode.render()`只是返回一个空字符串。`{% comment %}`和`{% endcomment %}`之间的任何内容都会被忽略。

## 解析直到另一个块标签，并保存内容

在前面的例子中，`do_comment()`丢弃了`{% comment %}`和`{% endcomment %}`之间的所有内容

`{% comment %}`和`{% endcomment %}`。而不是这样做，可以对块标签之间的代码进行操作。例如，这里有一个自定义模板标签`{% upper %}`，它会将其自身和之间的所有内容都大写

`{% endupper %}`。用法：

```py
{% upper %}This will appear in uppercase, {{ your_name }}.{% endupper %} 

```

与前面的例子一样，我们将使用`parser.parse()`。但是这次，我们将将结果的`nodelist`传递给`Node`：

```py
def do_upper(parser, token): 
    nodelist = parser.parse(('endupper',)) 
    parser.delete_first_token() 
    return UpperNode(nodelist) 

class UpperNode(template.Node): 
    def __init__(self, nodelist): 
        self.nodelist = nodelist 
    def render(self, context): 
        output = self.nodelist.render(context) 
        return output.upper() 

```

这里唯一的新概念是`UpperNode.render()`中的`self.nodelist.render(context)`。有关复杂渲染的更多示例，请参阅`django/template/defaulttags.py`中的`{% for %}`和`django/template/smartif.py`中的`{% if %}`的源代码。

# 接下来是什么

继续本节关于高级主题的主题，下一章涵盖了 Django 模型的高级用法。
