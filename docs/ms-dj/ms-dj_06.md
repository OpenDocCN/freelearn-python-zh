# 第六章：表单

HTML 表单是交互式网站的支柱，从谷歌的单个搜索框的简单性到无处不在的博客评论提交表单到复杂的自定义数据输入界面。

本章涵盖了如何使用 Django 访问用户提交的表单数据，对其进行验证并执行某些操作。在此过程中，我们将涵盖`HttpRequest`和`Form`对象。

# 从请求对象获取数据

我在第二章中介绍了`HttpRequest`对象，*视图和 URLconfs*，当时我们首次涵盖了视图函数，但那时我对它们没有太多可说的。回想一下，每个视图函数都以`HttpRequest`对象作为其第一个参数，就像我们的`hello()`视图一样：

```py
from django.http import HttpResponse 

def hello(request): 
    return HttpResponse("Hello world") 

```

`HttpRequest`对象，比如这里的变量`request`，有许多有趣的属性和方法，您应该熟悉它们，以便了解可能发生的情况。您可以使用这些属性来获取有关当前请求的信息（即加载 Django 站点上当前页面的用户/网络浏览器）在执行视图函数时。

## 关于 URL 的信息

`HttpRequest`对象包含有关当前请求的 URL 的几个信息（*表 6.1*）。

| 属性/方法 | 描述 | 示例 |
| --- | --- | --- |
| `request.path` | 完整路径，不包括域名，但包括前导斜杠。 | `"/hello/"` |
| `request.get_host()` | 主机（即俗称的“域名”）。 | `"127.0.0.1:8000"`或`"www.example.com"` |
| `request.get_full_path()` | `path`，加上查询字符串（如果有的话）。 | `"/hello/?print=true"` |
| `request.is_secure()` | 如果请求是通过 HTTPS 进行的，则为`True`。否则为`False`。 | `True`或`False` |

表 6.1：HttpRequest 方法和属性

始终使用这些属性/方法，而不是在视图中硬编码 URL。这样可以使代码更灵活，可以在其他地方重用。一个简单的例子：

```py
# BAD! 
def current_url_view_bad(request): 
    return HttpResponse("Welcome to the page at /current/") 

# GOOD 
def current_url_view_good(request): 
    return HttpResponse("Welcome to the page at %s" % request.path) 

```

## 请求对象的其他信息

`request.META`是一个 Python 字典，包含给定请求的所有可用 HTTP 标头-包括用户的 IP 地址和用户代理（通常是 Web 浏览器的名称和版本）。请注意，可用标头的完整列表取决于用户发送了哪些标头以及您的 Web 服务器设置了哪些标头。该字典中一些常用的键是：

+   `HTTP_REFERER`：引用的 URL，如果有的话。（请注意`REFERER`的拼写错误）。

+   `HTTP_USER_AGENT`：用户的浏览器的用户代理字符串，如果有的话。它看起来像这样：`"Mozilla/5.0 (X11; U; Linux i686; fr-FR; rv:1.8.1.17) Gecko/20080829 Firefox/2.0.0.17"`。

+   `REMOTE_ADDR`：客户端的 IP 地址，例如`"12.345.67.89"`。（如果请求通过任何代理，则这可能是一个逗号分隔的 IP 地址列表，例如`"12.345.67.89,23.456.78.90"`）。

请注意，因为`request.META`只是一个基本的 Python 字典，如果您尝试访问一个不存在的键，您将得到一个`KeyError`异常。（因为 HTTP 标头是外部数据-即它们是由您的用户的浏览器提交的-所以不应该信任它们，您应该始终设计您的应用程序，以便在特定标头为空或不存在时优雅地失败。）您应该使用`try`/`except`子句或`get()`方法来处理未定义键的情况：

```py
# BAD! 
def ua_display_bad(request): 
    ua = request.META['HTTP_USER_AGENT']  # Might raise KeyError! 
    return HttpResponse("Your browser is %s" % ua) 

# GOOD (VERSION 1) 
def ua_display_good1(request): 
    try: 
        ua = request.META['HTTP_USER_AGENT'] 
    except KeyError: 
        ua = 'unknown' 
    return HttpResponse("Your browser is %s" % ua) 

# GOOD (VERSION 2) 
def ua_display_good2(request): 
    ua = request.META.get('HTTP_USER_AGENT', 'unknown') 
    return HttpResponse("Your browser is %s" % ua) 

```

我鼓励您编写一个小视图，显示所有`request.META`数据，以便了解其中的内容。以下是该视图的样子：

```py
def display_meta(request): 
    values = request.META.items() 
    values.sort() 
    html = [] 
    for k, v in values: 
      html.append('<tr><td>%s</td><td>%s</td></tr>' % (k, v)) 
    return HttpResponse('<table>%s</table>' % '\n'.join(html)) 

```

查看请求对象包含的信息的另一种好方法是仔细查看 Django 错误页面，当您使系统崩溃时-那里有大量有用的信息，包括所有 HTTP 标头和其他请求对象（例如`request.path`）。

## 有关提交数据的信息

关于请求的基本元数据之外，`HttpRequest`对象有两个属性，包含用户提交的信息：`request.GET`和`request.POST`。这两个都是类似字典的对象，可以访问`GET`和`POST`数据。

`POST`数据通常是从 HTML `<form>`提交的，而`GET`数据可以来自页面 URL 中的`<form>`或查询字符串。

### 注意

**类似字典的对象**

当我们说`request.GET`和`request.POST`是*类似字典*的对象时，我们的意思是它们的行为类似于标准的 Python 字典，但在技术上并不是字典。例如，`request.GET`和`request.POST`都有`get()`、`keys()`和`values()`方法，您可以通过`for key in request.GET`来遍历键。那么为什么要区分呢？因为`request.GET`和`request.POST`都有标准字典没有的额外方法。我们将在短时间内介绍这些方法。您可能遇到过类似的术语*类似文件*的对象-具有一些基本方法（如`read()`）的 Python 对象，让它们可以充当"真实"文件对象的替代品。

# 一个简单的表单处理示例

继续图书、作者和出版商的示例，让我们创建一个简单的视图，让用户通过标题搜索我们的图书数据库。通常，开发表单有两个部分：HTML 用户界面和处理提交数据的后端视图代码。第一部分很容易；让我们设置一个显示搜索表单的视图：

```py

from django.shortcuts import render 

def search_form(request): 
    return render(request, 'search_form.html') 

```

正如您在第三章中学到的，这个视图可以存在于 Python 路径的任何位置。为了论证，将其放在`books/views.py`中。相应的模板`search_form.html`可能如下所示：

```py
<html> 
<head> 
    <title>Search</title> 
</head> 
<body> 
    <form action="/search/" method="get"> 
        <input type="text" name="q"> 
        <input type="submit" value="Search"> 
    </form> 
</body> 
</html> 

```

将此文件保存到您在第三章中创建的`mysite/templates`目录中，*模板*，或者您可以创建一个新的文件夹`books/templates`。只需确保您的设置文件中的`'APP_DIRS'`设置为`True`。`urls.py`中的 URL 模式可能如下所示：

```py
from books import views 

urlpatterns = [ 
    # ... 
    url(r'^search-form/$', views.search_form), 
    # ... 
] 

```

（请注意，我们直接导入`views`模块，而不是像`from mysite.views import search_form`这样的方式，因为前者更简洁。我们将在第七章中更详细地介绍这种导入方法，*高级视图和 URLconfs*）。现在，如果您运行开发服务器并访问`http://127.0.0.1:8000/search-form/`，您将看到搜索界面。足够简单。不过，尝试提交表单，您将收到 Django 404 错误。表单指向 URL`/search/`，但尚未实现。让我们用第二个视图函数来修复这个问题：

```py
# urls.py 

urlpatterns = [ 
    # ... 
    url(r'^search-form/$', views.search_form), 
    url(r'^search/$', views.search), 
    # ... 
] 

# books/views.py 

from django.http import HttpResponse 

# ... 

def search(request): 
    if 'q' in request.GET: 
        message = 'You searched for: %r' % request.GET['q'] 
    else: 
        message = 'You submitted an empty form.' 
    return HttpResponse(message) 

```

目前，这只是显示用户的搜索词，这样我们可以确保数据被正确提交到 Django，并且您可以感受搜索词是如何在系统中流动的。简而言之：

+   HTML `<form>`定义了一个变量`q`。当提交时，`q`的值通过`GET`（`method="get"`）发送到 URL`/search/`。

+   处理 URL`/search/`（`search()`）的 Django 视图可以访问`request.GET`中的`q`值。

这里要指出的一个重要事情是，我们明确检查`request.GET`中是否存在`'q'`。正如我在前面的`request.META`部分中指出的，您不应信任用户提交的任何内容，甚至不应假设他们首先提交了任何内容。如果我们没有添加这个检查，任何空表单的提交都会在视图中引发`KeyError`：

```py
# BAD! 
def bad_search(request): 
    # The following line will raise KeyError if 'q' hasn't 
    # been submitted! 
    message = 'You searched for: %r' % request.GET['q'] 
    return HttpResponse(message) 

```

## 查询字符串参数

因为`GET`数据是通过查询字符串传递的（例如，`/search/?q=django`），您可以使用`request.GET`来访问查询字符串变量。在第二章中，*视图和 URLconfs*，介绍了 Django 的 URLconf 系统，我将 Django 的美观 URL 与更传统的 PHP/Java URL 进行了比较，例如`/time/plus?hours=3`，并说我会在第六章中向您展示如何做后者。现在您知道如何在视图中访问查询字符串参数（例如在这个示例中的`hours=3`）-使用`request.GET`。

`POST`数据的工作方式与`GET`数据相同-只需使用`request.POST`而不是`request.GET`。`GET`和`POST`之间有什么区别？当提交表单的行为只是获取数据时使用`GET`。当提交表单的行为会产生一些副作用-更改数据、发送电子邮件或其他超出简单数据*显示*的操作时使用`POST`。在我们的图书搜索示例中，我们使用`GET`，因为查询不会改变服务器上的任何数据。（如果您想了解更多关于`GET`和`POST`的信息，请参阅 http://www.w3.org/2001/tag/doc/whenToUseGet.html 网站。）现在我们已经验证了`request.GET`是否被正确传递，让我们将用户的搜索查询连接到我们的图书数据库中（同样是在`views.py`中）：

```py
from django.http import HttpResponse 
from django.shortcuts import render 
from books.models import Book 

def search(request): 
    if 'q' in request.GET and request.GET['q']: 
        q = request.GET['q'] 
        books = Book.objects.filter(title__icontains=q) 
        return render(request, 'search_results.html', 
                      {'books': books, 'query': q}) 
    else: 
        return HttpResponse('Please submit a search term.') 

```

关于我们在这里所做的一些说明：

+   除了检查`'q'`是否存在于`request.GET`中，我们还确保在将其传递给数据库查询之前，`request.GET['q']`是一个非空值。

+   我们使用`Book.objects.filter(title__icontains=q)`来查询我们的图书表，找到标题包含给定提交的所有书籍。`icontains`是一种查找类型（如第四章和附录 B 中所解释的那样），该语句可以粗略地翻译为“获取标题包含`q`的书籍，而不区分大小写。”

+   这是一个非常简单的图书搜索方法。我们不建议在大型生产数据库上使用简单的`icontains`查询，因为它可能会很慢。（在现实世界中，您可能希望使用某种自定义搜索系统。搜索网络以获取*开源全文搜索*的可能性。）

+   我们将`books`，一个`Book`对象的列表，传递给模板。`search_results.html`文件可能包括类似以下内容：

```py
         <html> 
          <head> 
              <title>Book Search</title> 
          </head> 
          <body> 
            <p>You searched for: <strong>{{ query }}</strong></p> 

            {% if books %} 
                <p>Found {{ books|length }}
                    book{{ books|pluralize }}.</p> 
                <ul> 
                    {% for book in books %} 
                    <li>{{ book.title }}</li> 
                    {% endfor %} 
                </ul> 
            {% else %} 
                <p>No books matched your search criteria.</p> 
            {% endif %} 

          </body> 
        </html> 

```

注意使用`pluralize`模板过滤器，根据找到的书籍数量输出“s”。

# 改进我们简单的表单处理示例

与以前的章节一样，我向您展示了可能起作用的最简单的方法。现在我将指出一些问题，并向您展示如何改进它。首先，我们的`search()`视图对空查询的处理很差-我们只显示一个**请提交搜索词。**消息，要求用户点击浏览器的返回按钮。

这是可怕的，不专业的，如果您真的在实际中实现了这样的东西，您的 Django 权限将被撤销。更好的方法是重新显示表单，并在其前面显示一个错误，这样用户可以立即重试。最简单的方法是再次渲染模板，就像这样：

```py
from django.http import HttpResponse 
from django.shortcuts import render 
from books.models import Book 

def search_form(request): 
    return render(request, 'search_form.html') 

def search(request): 
    if 'q' in request.GET and request.GET['q']: 
        q = request.GET['q'] 
        books = Book.objects.filter(title__icontains=q) 
        return render(request, 'search_results.html', 
                      {'books': books, 'query': q}) 
    else: 
 return render
           (request, 'search_form.html', {'error': True})

```

（请注意，我在这里包括了`search_form()`，这样您就可以在一个地方看到两个视图。）在这里，我们改进了`search()`，如果查询为空，就重新渲染`search_form.html`模板。因为我们需要在该模板中显示错误消息，所以我们传递了一个模板变量。现在我们可以编辑`search_form.html`来检查`error`变量：

```py
<html> 
<head> 
    <title>Search</title> 
</head> 
<body> 
 {% if error %} 
 <p style="color: red;">Please submit a search term.</p> 
 {% endif %} 
    <form action="/search/" method="get"> 
        <input type="text" name="q"> 
        <input type="submit" value="Search"> 
    </form> 
</body> 
</html> 

```

我们仍然可以从我们原始的视图`search_form()`中使用这个模板，因为`search_form()`不会将`error`传递给模板，所以在这种情况下不会显示错误消息。有了这个改变，这是一个更好的应用程序，但现在问题是：是否真的需要一个专门的`search_form()`视图？

目前，对 URL`/search/`（没有任何`GET`参数）的请求将显示空表单（但带有错误）。只要我们在没有`GET`参数的情况下访问`/search/`，就可以删除`search_form()`视图及其相关的 URLpattern，同时将`search()`更改为在有人访问`/search/`时隐藏错误消息：

```py
def search(request): 
    error = False 
    if 'q' in request.GET: 
        q = request.GET['q'] 
if not q: 
 error = True 
 else: 
            books = Book.objects.filter(title__icontains=q) 
            return render(request, 'search_results.html', 
                          {'books': books, 'query': q}) 
 return render(request, 'search_form.html', 
 {'error': error})

```

在这个更新的视图中，如果用户在没有`GET`参数的情况下访问`/search/`，他们将看到没有错误消息的搜索表单。如果用户提交了一个空值的`'q'`，他们将看到带有错误消息的搜索表单。最后，如果用户提交了一个非空值的`'q'`，他们将看到搜索结果。

我们可以对此应用进行最后一次改进，以消除一些冗余。现在我们已经将两个视图和 URL 合并为一个，并且`/search/`处理搜索表单显示和结果显示，`search_form.html`中的 HTML`<form>`不必硬编码 URL。而不是这样：

```py
<form action="/search/" method="get"> 

```

可以更改为这样：

```py
<form action="" method="get"> 

```

`action=""` 表示*将表单提交到与当前页面相同的 URL*。有了这个改变，如果您将`search()`视图连接到另一个 URL，您就不必记得更改`action`。

# 简单验证

我们的搜索示例仍然相当简单，特别是在数据验证方面；我们只是检查确保搜索查询不为空。许多 HTML 表单包括比确保值非空更复杂的验证级别。我们都在网站上看到过错误消息：

+   *请输入一个有效的电子邮件地址。'foo'不是一个电子邮件地址。*

+   *请输入一个有效的五位数字的美国邮政编码。'123'不是一个邮政编码。*

+   *请输入格式为 YYYY-MM-DD 的有效日期。*

+   *请输入至少 8 个字符长且至少包含一个数字的密码。*

让我们调整我们的`search()`视图，以验证搜索词是否少于或等于 20 个字符长。（举个例子，假设超过这个长度可能会使查询变得太慢。）我们该如何做到这一点？

最简单的方法是直接在视图中嵌入逻辑，如下所示：

```py
def search(request): 
    error = False 
    if 'q' in request.GET: 
        q = request.GET['q'] 
        if not q: 
            error = True 
 elif len(q) > 20: 
 error = True 
        else: 
            books = Book.objects.filter(title__icontains=q) 
            return render(request, 'search_results.html', 
                          {'books': books, 'query': q}) 
    return render(request, 'search_form.html', 
        {'error': error}) 

```

现在，如果您尝试提交一个超过 20 个字符长的搜索查询，它将不允许您进行搜索；您将收到一个错误消息。但是`search_form.html`中的错误消息目前说：“请提交搜索词”。-所以我们必须更改它以适应两种情况：

```py
<html> 
<head> 
    <title>Search</title> 
</head> 
<body> 
    {% if error %} 
 <p style="color: red;"> 
 Please submit a search term 20 characters or shorter. 
 </p> 
    {% endif %} 

    <form action="/search/" method="get"> 
        <input type="text" name="q"> 
        <input type="submit" value="Search"> 
    </form> 
</body> 
</html> 

```

这里有一些不好的地方。我们的一刀切错误消息可能会令人困惑。为什么空表单提交的错误消息要提及 20 个字符的限制？

错误消息应该是具体的、明确的，不应该令人困惑。问题在于我们使用了一个简单的布尔值`error`，而我们应该使用一个错误消息字符串列表。以下是我们可能如何修复它：

```py
def search(request): 
    errors = [] 
    if 'q' in request.GET: 
        q = request.GET['q'] 
        if not q: 
 errors.append('Enter a search term.') 
        elif len(q) > 20: 
 errors.append('Please enter at most 20 characters.') 
        else: 
            books = Book.objects.filter(title__icontains=q) 
            return render(request, 'search_results.html', 
                          {'books': books, 'query': q}) 
    return render(request, 'search_form.html', 
                  {'errors': errors}) 

```

然后，我们需要对`search_form.html`模板进行小的调整，以反映它现在传递了一个`errors`列表，而不是一个`error`布尔值：

```py
<html> 
<head> 
    <title>Search</title> 
</head> 
<body> 
    {% if errors %} 
 <ul> 
 {% for error in errors %} 
 <li>{{ error }}</li> 
 {% endfor %} 
 </ul> 
    {% endif %} 
    <form action="/search/" method="get"> 
        <input type="text" name="q"> 
        <input type="submit" value="Search"> 
    </form> 
</body> 
</html> 

```

# 创建联系表单

尽管我们多次迭代了图书搜索表单示例并对其进行了良好的改进，但它仍然基本上很简单：只有一个字段`'q'`。随着表单变得更加复杂，我们必须一遍又一遍地重复前面的步骤，为我们使用的每个表单字段重复这些步骤。这引入了很多废料和很多人为错误的机会。幸运的是，Django 的开发人员考虑到了这一点，并在 Django 中构建了一个处理表单和验证相关任务的更高级别库。

## 您的第一个表单类

Django 带有一个表单库，称为`django.forms`，它处理了本章中我们探讨的许多问题-从 HTML 表单显示到验证。让我们深入研究并使用 Django 表单框架重新设计我们的联系表单应用程序。

使用表单框架的主要方法是为您处理的每个 HTML `<form>`定义一个`Form`类。在我们的情况下，我们只有一个`<form>`，所以我们将有一个`Form`类。这个类可以放在任何您想要的地方，包括直接放在您的`views.py`文件中，但社区约定是将`Form`类放在一个名为`forms.py`的单独文件中。

在与您的`mysite/views.py`相同的目录中创建此文件，并输入以下内容：

```py
from django import forms 

class ContactForm(forms.Form): 
    subject = forms.CharField() 
    email = forms.EmailField(required=False) 
    message = forms.CharField() 

```

这是非常直观的，类似于 Django 的模型语法。表单中的每个字段都由`Field`类的一种类型表示-这里只使用`CharField`和`EmailField`作为`Form`类的属性。默认情况下，每个字段都是必需的，因此要使`email`可选，我们指定`required=False`。让我们进入 Python 交互解释器，看看这个类能做什么。它能做的第一件事是将自己显示为 HTML：

```py
>>> from mysite.forms import ContactForm 
>>> f = ContactForm() 
>>> print(f) 
<tr><th><label for="id_subject">Subject:</label></th><td><input type="text" name="subject" id="id_subject" /></td></tr> 
<tr><th><label for="id_email">Email:</label></th><td><input type="text" name="email" id="id_email" /></td></tr> 
<tr><th><label for="id_message">Message:</label></th><td><input type="text" name="message" id="id_message" /></td></tr> 

```

Django 为每个字段添加了标签，以及用于辅助功能的`<label>`标签。其目的是使默认行为尽可能优化。此默认输出采用 HTML `<table>`格式，但还有其他几种内置输出：

```py
>>> print(f.as_ul()) 
<li><label for="id_subject">Subject:</label> <input type="text" name="subject" id="id_subject" /></li> 
<li><label for="id_email">Email:</label> <input type="text" name="email" id="id_email" /></li> 
<li><label for="id_message">Message:</label> <input type="text" name="message" id="id_message" /></li> 

>>> print(f.as_p()) 
<p><label for="id_subject">Subject:</label> <input type="text" name="subject" id="id_subject" /></p> 
<p><label for="id_email">Email:</label> <input type="text" name="email" id="id_email" /></p> 
<p><label for="id_message">Message:</label> <input type="text" name="message" id="id_message" /></p> 

```

请注意，输出中不包括开放和关闭的`<table>`、`<ul>`和`<form>`标签，因此您可以根据需要添加任何额外的行和自定义。这些方法只是常见情况下的快捷方式，即“显示整个表单”。您还可以显示特定字段的 HTML：

```py
>>> print(f['subject']) 
<input id="id_subject" name="subject" type="text" /> 
>>> print f['message'] 
<input id="id_message" name="message" type="text" /> 

```

`Form`对象的第二个功能是验证数据。要验证数据，请创建一个新的`Form`对象，并将数据字典传递给它，将字段名称映射到数据：

```py
>>> f = ContactForm({'subject': 'Hello', 'email': 'adrian@example.com', 'message': 'Nice site!'}) 

```

一旦您将数据与`Form`实例关联起来，就创建了一个**绑定**表单：

```py
>>> f.is_bound 
True 

```

对任何绑定的`Form`调用`is_valid()`方法，以了解其数据是否有效。我们已为每个字段传递了有效值，因此整个`Form`都是有效的：

```py
>>> f.is_valid() 
True 

```

如果我们不传递`email`字段，它仍然有效，因为我们已经为该字段指定了`required=False`：

```py
>>> f = ContactForm({'subject': 'Hello', 'message': 'Nice site!'}) 
>>> f.is_valid() 
True 

```

但是，如果我们省略`subject`或`message`中的任何一个，`Form`将不再有效：

```py
>>> f = ContactForm({'subject': 'Hello'}) 
>>> f.is_valid() 
False 
>>> f = ContactForm({'subject': 'Hello', 'message': ''}) 
>>> f.is_valid() 
False 

```

您可以深入了解特定字段的错误消息：

```py
>>> f = ContactForm({'subject': 'Hello', 'message': ''}) 
>>> f['message'].errors 
['This field is required.'] 
>>> f['subject'].errors 
[] 
>>> f['email'].errors 
[] 

```

每个绑定的`Form`实例都有一个`errors`属性，该属性为您提供了一个将字段名称映射到错误消息列表的字典：

```py
>>> f = ContactForm({'subject': 'Hello', 'message': ''}) 
>>> f.errors 
{'message': ['This field is required.']} 

```

最后，对于数据已被发现有效的`Form`实例，将提供`cleaned_data`属性。这是提交的数据的“清理”。Django 的表单框架不仅验证数据；它通过将值转换为适当的 Python 类型来清理数据：

```py
>>> f = ContactForm({'subject': 'Hello', 'email': 'adrian@example.com', 
'message': 'Nice site!'}) 
>>> f.is_valid() True 
>>> f.cleaned_data 
{'message': 'Nice site!', 'email': 'adrian@example.com', 'subject': 
'Hello'} 

```

我们的联系表单只处理字符串，这些字符串被“清理”为字符串对象-但是，如果我们使用`IntegerField`或`DateField`，表单框架将确保`cleaned_data`使用适当的 Python 整数或`datetime.date`对象来表示给定字段。

# 将表单对象与视图绑定

除非我们有一种方法将其显示给用户，否则我们的联系表单对我们来说没有太大用处。为此，我们首先需要更新我们的`mysite/views`：

```py
# views.py 

from django.shortcuts import render 
from mysite.forms import ContactForm 
from django.http import HttpResponseRedirect 
from django.core.mail import send_mail 

# ... 

def contact(request): 
    if request.method == 'POST': 
        form = ContactForm(request.POST) 
        if form.is_valid(): 
            cd = form.cleaned_data 
            send_mail( 
                cd['subject'], 
                cd['message'], 
                cd.get('email', 'noreply@example.com'), 
                ['siteowner@example.com'], 
            ) 
            return HttpResponseRedirect('/contact/thanks/') 
    else: 
        form = ContactForm() 
    return render(request, 'contact_form.html', {'form': form}) 

```

接下来，我们必须创建我们的联系表单（保存到`mysite/templates`）：

```py
# contact_form.html 

<html> 
<head> 
    <title>Contact us</title> 
</head> 
<body> 
    <h1>Contact us</h1> 

    {% if form.errors %} 
        <p style="color: red;"> 
            Please correct the error{{ form.errors|pluralize }} below. 
        </p> 
    {% endif %} 

    <form action="" method="post"> 
        <table> 
            {{ form.as_table }} 
        </table> 
        {% csrf_token %} 
        <input type="submit" value="Submit"> 
    </form> 
</body> 
</html> 

```

最后，我们需要更改我们的`urls.py`，以便在`/contact/`处显示我们的联系表单：

```py
 # ... 
from mysite.views import hello, current_datetime, hours_ahead, contact 

 urlpatterns = [ 

     # ... 

     url(r'^contact/$', contact), 
] 

```

由于我们正在创建一个`POST`表单（可能会导致修改数据的效果），我们需要担心跨站点请求伪造。幸运的是，您不必太担心，因为 Django 带有一个非常易于使用的系统来防止它。简而言之，所有针对内部 URL 的`POST`表单都应使用`{% csrf_token %}`模板标记。更多细节

`{% csrf_token %}`可以在第十九章*Django 中的安全性*中找到。

尝试在本地运行此代码。加载表单，提交表单时没有填写任何字段，使用无效的电子邮件地址提交表单，最后使用有效数据提交表单。（当调用`send_mail()`时，除非您配置了邮件服务器，否则会收到`ConnectionRefusedError`。）

# 更改字段呈现方式

当您在本地呈现此表单时，您可能首先注意到的是`message`字段显示为`<input type="text">`，而应该是`<textarea>`。我们可以通过设置字段的小部件来解决这个问题：

```py
from django import forms 

class ContactForm(forms.Form): 
    subject = forms.CharField() 
    email = forms.EmailField(required=False) 
    message = forms.CharField(widget=forms.Textarea)

```

表单框架将每个字段的呈现逻辑分离为一组小部件。每种字段类型都有一个默认小部件，但您可以轻松地覆盖默认值，或者提供自定义小部件。将`Field`类视为**验证逻辑**，而小部件表示**呈现逻辑**。

# 设置最大长度

最常见的验证需求之一是检查字段的大小。为了好玩，我们应该改进我们的`ContactForm`以将`subject`限制为 100 个字符。要做到这一点，只需向`CharField`提供`max_length`，如下所示：

```py
from django import forms 

class ContactForm(forms.Form): 
    subject = forms.CharField(max_length=100) 
    email = forms.EmailField(required=False) 
    message = forms.CharField(widget=forms.Textarea) 

```

还可以使用可选的`min_length`参数。

# 设置初始值

作为对这个表单的改进，让我们为`subject`字段添加一个初始值：`I love your site!`（一点点建议的力量不会有害）。为此，我们可以在创建`Form`实例时使用`initial`参数：

```py
def contact(request): 
    if request.method == 'POST': 
        form = ContactForm(request.POST) 
        if form.is_valid(): 
            cd = form.cleaned_data 
            send_mail( 
                cd['subject'], 
                cd['message'], 
                cd.get('email', 'noreply@example.com'), 
['siteowner@example.com'], 
            ) 
            return HttpResponseRedirect('/contact/thanks/') 
    else: 
        form = ContactForm( 
            initial={'subject': 'I love your site!'} 
        ) 
    return render(request, 'contact_form.html', {'form':form}) 

```

现在，`subject`字段将显示为预填充了这种陈述。请注意，传递初始数据和绑定表单的数据之间存在差异。最大的区别在于，如果你只是传递初始数据，那么表单将是未绑定的，这意味着它不会有任何错误消息。

# 自定义验证规则

想象一下，我们已经推出了我们的反馈表单，电子邮件已经开始涌入。只有一个问题：一些提交的消息只有一两个单词，这对我们来说不够长。我们决定采用一个新的验证策略：请至少四个单词。

有许多方法可以将自定义验证集成到 Django 表单中。如果我们的规则是我们将一遍又一遍地重用的，我们可以创建一个自定义字段类型。大多数自定义验证都是一次性的事务，可以直接绑定到`Form`类。我们想要在`message`字段上进行额外的验证，因此我们在`Form`类中添加了一个`clean_message()`方法：

```py
from django import forms 

class ContactForm(forms.Form): 
    subject = forms.CharField(max_length=100) 
    email = forms.EmailField(required=False) 
    message = forms.CharField(widget=forms.Textarea) 

    def clean_message(self): 
 message = self.cleaned_data['message'] 
 num_words = len(message.split()) 
 if num_words < 4: 
 raise forms.ValidationError("Not enough words!") 
 return message

```

Django 的表单系统会自动查找任何以`clean_`开头并以字段名称结尾的方法。如果存在这样的方法，它将在验证期间被调用。具体来说，`clean_message()`方法将在给定字段的默认验证逻辑之后被调用（在本例中，是必需的`CharField`的验证逻辑）。

因为字段数据已经部分处理，我们从`self.cleaned_data`中提取它。此外，我们不必担心检查该值是否存在且非空；这是默认验证器完成的。我们天真地使用`len()`和`split()`的组合来计算单词的数量。如果用户输入的单词太少，我们会引发一个`forms.ValidationError`。

附加到此异常的字符串将显示为错误列表中的一项。重要的是我们明确地在方法的最后返回字段的清理值。这允许我们在自定义验证方法中修改值（或将其转换为不同的 Python 类型）。如果我们忘记了返回语句，那么将返回`None`，并且原始值将丢失。

# 指定标签

默认情况下，Django 自动生成的表单 HTML 上的标签是通过用空格替换下划线并大写第一个字母来创建的-因此`email`字段的标签是"`Email`"。（听起来熟悉吗？这是 Django 模型用于计算字段默认`verbose_name`值的相同简单算法。我们在第四章中介绍过这一点，*模型*）。但是，与 Django 的模型一样，我们可以自定义给定字段的标签。只需使用`label`，如下所示：

```py
class ContactForm(forms.Form): 
    subject = forms.CharField(max_length=100) 
 email = forms.EmailField(required=False,
        label='Your e-mail address') 
    message = forms.CharField(widget=forms.Textarea)
```

# 自定义表单设计

我们的`contact_form.html`模板使用`{{ form.as_table }}`来显示表单，但我们可以以其他方式显示表单，以便更精细地控制显示。自定义表单的呈现方式最快的方法是使用 CSS。

错误列表，特别是可以通过一些视觉增强，并且自动生成的错误列表使用`<ul class="errorlist">`，这样你就可以用 CSS 来定位它们。以下 CSS 确实让我们的错误更加突出：

```py
<style type="text/css"> 
    ul.errorlist { 
        margin: 0; 
        padding: 0; 
    } 
    .errorlist li { 
        background-color: red; 
        color: white; 
        display: block; 
        font-size: 10px; 
        margin: 0 0 3px; 
        padding: 4px 5px; 
    } 
</style> 

```

虽然为我们生成表单的 HTML 很方便，但在许多情况下，您可能希望覆盖默认的呈现方式。`{{ form.as_table }}`和其他方法在开发应用程序时是有用的快捷方式，但表单的显示方式可以被覆盖，主要是在模板本身内部，您可能会发现自己这样做。

每个字段的小部件（`<input type="text">`，`<select>`，`<textarea>`等）可以通过在模板中访问`{{ form.fieldname }}`来单独呈现，并且与字段相关的任何错误都可以作为`{{ form.fieldname.errors }}`获得。

考虑到这一点，我们可以使用以下模板代码为我们的联系表单构建一个自定义模板：

```py
<html> 
<head> 
    <title>Contact us</title> 
</head> 
<body> 
    <h1>Contact us</h1> 

    {% if form.errors %} 
        <p style="color: red;"> 
            Please correct the error{{ form.errors|pluralize }} below. 
        </p> 
    {% endif %} 

    <form action="" method="post"> 
        <div class="field"> 
            {{ form.subject.errors }} 
            <label for="id_subject">Subject:</label> 
            {{ form.subject }} 
        </div> 
        <div class="field"> 
            {{ form.email.errors }} 
            <label for="id_email">Your e-mail address:</label> 
            {{ form.email }} 
        </div> 
        <div class="field"> 
            {{ form.message.errors }} 
            <label for="id_message">Message:</label> 
            {{ form.message }} 
        </div> 
        {% csrf_token %} 
        <input type="submit" value="Submit"> 
    </form> 
</body> 
</html> 

```

如果存在错误，`{{ form.message.errors }}`会显示一个`<ul class="errorlist">`，如果字段有效（或表单未绑定），则显示一个空字符串。我们还可以将`form.message.errors`视为布尔值，甚至可以将其作为列表进行迭代。例如：

```py
<div class="field{% if form.message.errors %} errors{% endif %}"> 
    {% if form.message.errors %} 
        <ul> 
        {% for error in form.message.errors %} 
            <li><strong>{{ error }}</strong></li> 
        {% endfor %} 
        </ul> 
    {% endif %} 
    <label for="id_message">Message:</label> 
    {{ form.message }} 
</div> 

```

在验证错误的情况下，这将在包含的`<div>`中添加一个“errors”类，并在无序列表中显示错误列表。

# 接下来呢？

本章结束了本书的介绍性材料-所谓的*核心课程* 本书的下一部分，第七章，*高级视图和 URLconfs*，到第十三章，*部署 Django*，将更详细地介绍高级 Django 用法，包括如何部署 Django 应用程序（第十三章，*部署 Django*）。在这七章之后，你应该已经了解足够的知识来开始编写自己的 Django 项目。本书中的其余材料将帮助您填补需要的空白。我们将从第七章开始，*高级视图和 URLconfs*，通过回顾并更仔细地查看视图和 URLconfs（首次介绍于第二章，*视图和 URLconfs*）。
