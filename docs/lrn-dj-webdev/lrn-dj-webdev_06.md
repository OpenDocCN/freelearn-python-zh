# 第六章：使用 AJAX 增强用户界面

**AJAX**的到来是 Web 2.0 历史上的一个重要里程碑。AJAX 是一组技术，使开发人员能够构建交互式、功能丰富的 Web 应用程序。在 AJAX 本身出现之前，这些技术多年前就已经存在。然而，AJAX 的出现代表了 Web 从需要在数据交换时刷新的静态页面向动态、响应迅速和交互式用户界面的转变。

由于我们的项目是一个 Web 2.0 应用程序，它应该更加注重用户体验。我们的应用程序的成功取决于用户在上面发布和分享内容。因此，我们的应用程序的用户界面是我们的主要关注点之一。本章将通过引入 AJAX 功能来改进我们的应用程序界面，使其更加用户友好和交互性。

在本章中，您将学习以下主题：

+   AJAX 及其优势

+   在 Django 中使用 AJAX

+   如何使用开源 jQuery 框架

+   实现推文的搜索

+   在不加载单独页面的情况下编辑推文

+   提交推文时自动完成标签

# AJAX 及其优势

AJAX 代表**异步 JavaScript 和 XML**，包括以下技术：

+   用于结构化和样式信息的 HTML 和 CSS

+   JavaScript 用于动态访问和操作信息

+   一个由现代浏览器提供的对象，用于在不重新加载当前网页的情况下与服务器交换数据

+   在客户端和服务器之间传输数据的格式

有时会使用 XML，但它可以是 HTML、纯文本或基于 JavaScript 的格式 JSON。

AJAX 技术使您能够在不重新加载整个页面的情况下在客户端和服务器之间交换数据。通过使用 AJAX，Web 开发人员能够增加 Web 页面的交互性和可用性。

在正确的地方实现 AJAX 时，它提供了以下优势：

+   **更好的用户体验**：通过 AJAX，用户可以在不刷新页面的情况下完成很多操作，这使得 Web 应用程序更接近常规桌面应用程序

+   **更好的性能**：通过与服务器交换所需的数据，AJAX 节省了带宽并提高了应用程序的速度

有许多使用 AJAX 的 Web 应用程序的例子。谷歌地图和 Gmail 可能是最突出的两个例子。事实上，这两个应用程序在推广 AJAX 的使用方面起到了重要作用，因为它们取得了成功。Gmail 与其他网络邮件服务的区别在于其用户界面，它使用户能够在不等待页面在每个操作后重新加载的情况下交互式地管理他们的电子邮件。这创造了更好的用户体验，使 Gmail 感觉更像是一个响应迅速且功能丰富的应用程序，而不是一个简单的网站。

本章将解释如何在 Django 中使用 AJAX，以使我们的应用程序更具响应性和用户友好性。我们将实现当今 Web 应用程序中发现的三种最常见的 AJAX 功能。但在此之前，我们将了解使用 AJAX 框架的好处，而不是使用原始 JavaScript 函数。

# 在 Django 中使用 AJAX 框架

由于我们已经在项目中使用了 Bootstrap，因此我们无需为 AJAX 和 jQuery 单独配置它。

使用 AJAX 框架的许多优点：

+   JavaScript 的实现因浏览器而异。一些浏览器提供更完整和功能丰富的实现，而其他浏览器包含不完整或不符合标准的实现。

没有 AJAX 框架，开发人员必须跟踪浏览器对他们使用的 JavaScript 功能的支持，并必须解决一些浏览器对 JavaScript 实现的限制。

另一方面，当使用 AJAX 框架时，框架会为我们处理这一点；它抽象了对 JavaScript 实现的访问，并处理了不同浏览器之间的差异和怪癖。这样，我们可以专注于开发功能，而不必担心浏览器的差异和限制。

+   标准的 JavaScript 函数和类集合对于完整的 Web 应用程序开发有些不足。各种常见任务需要许多行代码，即使它们可以包装在简单的函数中。

因此，即使您决定不使用 AJAX 框架，您也会发现自己在编写一个函数库，该函数库封装了 JavaScript 功能并使其更易于使用。然而，既然已经有许多优秀的开源库可用，为什么要重新发明轮子呢？

今天市场上可用的 AJAX 框架范围从提供服务器端和客户端组件的综合解决方案到简化使用 JavaScript 的轻量级客户端库。鉴于我们已经在服务器端使用 Django，我们只需要一个客户端框架。除此之外，该框架应该易于与 Django 集成，而不需要任何额外的依赖。最后，最好选择一个轻量级和快速的框架。有许多优秀的框架符合我们的要求，例如**Prototype**，**Yahoo! UI Library**和**jQuery**。

但是，对于我们的应用程序，我将选择 jQuery，因为它是这三种框架中最轻量级的。它还拥有一个非常活跃的开发社区和广泛的插件范围。如果您已经有其他框架的经验，可以在本章中继续使用它。的确，您将不得不将本章中的 JavaScript 代码适应到您的框架中，但是无论您选择哪种框架，服务器端的 Django 代码都将保持不变。

### 注意

您还需要导入 Bootstrap 和 jQuery。因此，在我们的 Django 项目中使用 AJAX 功能不需要特定的安装或导入。

# 使用开源的 jQuery 框架

在我们开始在项目中实现 AJAX 增强功能之前，让我们快速介绍一下 jQuery 框架。

## jQuery JavaScript 框架

jQuery 是一个 JavaScript 函数库，它简化了与 HTML 文档的交互并对其进行操作。该库旨在减少编写代码和实现跨浏览器兼容性所需的时间和精力，同时充分利用 JavaScript 提供的功能来构建交互式和响应式的 Web 应用程序。

使用 jQuery 的一般工作流程包括以下两个步骤：

1.  选择要处理的 HTML 元素或一组元素。

1.  将 jQuery 方法应用于所选组。

### 元素选择器

jQuery 提供了一种简单的选择元素的方法：通过将 CSS 选择器字符串传递给名为`$()`的函数。以下是一些示例，说明了此函数的用法：

+   如果您想选择页面上的所有锚（`<a>`）元素，可以使用`$("a")`函数调用

+   如果您想选择具有`.title` CSS 类的锚元素，请使用

```py
$("a.title")
```

+   要选择 ID 为`#nav`的元素，可以使用`$("#nav")`

+   要选择`#nav`内部的所有列表项（`<li>`）元素，请使用`$("#nav li")`

`$()`函数构造并返回一个 jQuery 对象。之后，您可以在此对象上调用方法以与所选的 HTML 元素交互。

### jQuery 方法

jQuery 提供了各种方法来操作 HTML 文档。您可以隐藏或显示元素，将事件处理程序附加到事件，修改 CSS 属性，操作页面结构，最重要的是执行 AJAX 请求。

为了调试，我们选择 Chrome 浏览器作为我们的首选浏览器。 Chrome 是最先进的 JavaScript 调试器之一，以其 Chrome 开发者工具的形式。要启动它，请在键盘上按下*Ctrl*+*Shift*+*J*。

要尝试本节中概述的方法，请启动开发服务器并导航到用户配置文件页面（`http://127.0.0.1:8000/user/ratan/`）。通过按下键盘上的*Ctrl*+*Shift*+*J*打开 Chrome 开发者工具（按*F12*），并尝试选择元素并操作它们。

### 隐藏和显示元素

让我们从简单的事情开始。要在页面上隐藏一个元素，请在其上调用`hide()`方法。要再次显示它，请调用`show()`方法。例如，尝试在您的应用程序的 Bootstrap 中称为`navbar`的导航菜单上尝试这个：

```py
>>> $(".navbar").hide()
>>> $(".navbar").show() 

```

您还可以在隐藏和显示元素时对元素进行动画处理。尝试使用`fadeOut()`、`fadeIn()`、`slideUp()`或`slideDown()`方法来查看这两种动画效果中的两种。

当然，如果一次选择多个元素，这些方法（就像所有其他 jQuery 方法一样）也会起作用。例如，如果打开用户配置文件并在 Chrome 开发人员工具控制台中输入以下方法调用，则所有推文都将消失：

```py
>>> $('.well').slideUp()

```

### 访问 CSS 属性和 HTML 属性

接下来，我们将学习如何更改元素的 CSS 属性。jQuery 提供了一个名为`css()`的方法来执行 CSS 操作。如果您以字符串形式传递 CSS 属性名称调用此方法，它将返回此属性的值：

```py
>>> $(".navbar").css("display")

```

这样的结果如下：

```py
block

```

如果向此方法传递第二个参数，它将将所选元素的指定 CSS 属性设置为附加参数：

```py
>>> $(".navbar").css("font-size", "0.8em")

```

这样的结果如下：

```py
<div id="nav" style="font-size: 0.8em;">

```

实际上，您可以操纵任何 HTML 属性，而不仅仅是 CSS 属性。要这样做，请使用`attr()`方法，它的工作方式与`css()`方法类似。使用属性名称调用它会返回属性值，而使用属性名称或值对调用它会将属性设置为传递的值：

```py
>>> $("input").attr("size", "48")

```

这将导致以下结果：

```py
<input type="hidden" name="csrfmiddlewaretoken" value="xxx" size="48">
<input id="id_country" name="country" type="hidden" value="Global" size="48">
<input type="submit" value="post" size="48">

```

这将一次性将页面上所有输入元素的大小更改为`48`。

除此之外，还有一些快捷方法可以获取和设置常用的属性，例如`val()`，当不带参数调用时返回输入字段的值，并在传递一个参数时将该值设置为参数。还有控制元素内部 HTML 代码的`html()`方法。

最后，有两种方法可以用来附加或分离 CSS 类到一个元素：它们是`addClass()`和`removeClass()`方法。还提供了第三种方法来切换 CSS 类，称为`toggleClass()`方法。所有这些类方法都将要更改的类的名称作为参数。

### 操作 HTML 文档

现在您已经熟悉了如何操作 HTML 元素，让我们看看如何添加新元素或删除现有元素。要在元素之前插入 HTML 代码，请使用`before()`方法，要在元素之后插入代码，请使用`after()`方法。请注意 jQuery 方法的命名方式非常直观，易于记忆！

让我们通过在用户页面上的标签列表周围插入括号来测试这些方法。

打开您的用户页面，并在 Chrome 开发者工具控制台中输入以下内容：

```py
>>> $(".well span").before("<strong>(</strong>")
>>> $(".well span").after("<strong>)</strong>")

```

您可以向`before()`或`after()`方法传递任何您想要的字符串。该字符串可以包含纯文本、一个 HTML 元素或更多。这些方法提供了一种非常灵活的方式来动态添加 HTML 元素到 HTML 文档中。

如果要删除一个元素，请使用`remove()`方法。例如：

```py
$("#navbar").remove()

```

这种方法不仅隐藏了元素，还将其从文档树中完全删除。如果在使用`remove()`方法后尝试重新选择元素，您将得到一个空集：

```py
>>> $("#nav")

```

这样的结果如下：

```py
[]

```

当然，这只是从当前页面实例中删除元素。如果重新加载页面，元素将再次出现。

### 遍历文档树

尽管 CSS 选择器提供了一种非常强大的选择元素的方式，但有时您希望从特定元素开始遍历文档树。

对此，jQuery 提供了几种方法。`parent()`方法返回当前选定元素的父元素。`children()`方法返回所选元素的所有直接子元素。最后，`find()`方法返回当前选定元素的所有后代元素。所有这些方法都接受一个可选的 CSS 选择器字符串，以限制结果为与选择器匹配的元素。例如，`$(".column").find("span")`返回类 column 的所有`<span>`后代。

如果要访问一组中的单个元素，请使用`get()`方法，该方法将元素的索引作为参数。例如，`$("span").get(0)`方法返回所选组中的第一个`<span>`元素。

### 处理事件

接下来我们将学习事件处理程序。事件处理程序是在特定事件发生时调用的 JavaScript 函数，例如，当单击按钮或提交表单时。jQuery 提供了一系列方法来将处理程序附加到事件上；在我们的应用程序中特别感兴趣的事件是鼠标点击和表单提交。要处理单击元素的事件，我们选择该元素并在其上调用`click()`方法。该方法将事件处理程序函数作为参数。让我们在 Chrome 开发者控制台中尝试一下。

打开应用程序的用户个人资料页面，并在推文后插入一个按钮：

```py
>>> $(".well span").after("<button id=\"test-button\">Click me!</button>")

```

### 注意

请注意，我们必须转义传递给`after()`方法的字符串中的引号。

如果您尝试单击此按钮，将不会发生任何事情，因此让我们为其附加一个事件处理程序：

```py
>>> $("#test-button").click(function () { alert("You clicked me!"); })

```

现在，当您点击按钮时，将出现一个消息框。这是如何工作的？

我们传递给`click()`方法的参数可能看起来有点复杂，因此让我们再次检查一下：

```py
function () { alert("You clicked me!"); }

```

这似乎是一个函数声明，但没有函数名。事实上，这个构造在 JavaScript 术语中创建了所谓的匿名函数，当您需要即时创建一个函数并将其作为参数传递给另一个函数时使用。我们本可以避免使用匿名函数，并将事件处理程序声明为常规函数：

```py
>>> function handler() { alert("You clicked me!"); }
>>> $("#test-button").click(handler)

```

前面的代码实现了相同的效果，但第一个更简洁、紧凑。我强烈建议您熟悉 JavaScript 中的匿名函数（如果您还没有），因为我相信您在使用一段时间后会欣赏这种构造并发现它更易读。

处理表单提交与处理鼠标点击非常相似。首先选择表单，然后在其上调用`submit()`方法，然后将处理程序作为参数传递。在后面的部分中，我们将在项目中添加 AJAX 功能时多次使用这种方法。

### 发送 AJAX 请求

在完成本节之前，让我们谈一下 AJAX 请求。jQuery 提供了许多发送 AJAX 请求到服务器的方法。例如，`load()`方法接受一个 URL，并将该 URL 的页面加载到所选元素中。还有发送 GET 或 POST 请求以及接收结果的方法。在实现项目中的 AJAX 功能时，我们将更深入地研究这些方法。

### 接下来呢？

这就结束了我们对 jQuery 的快速介绍。本节提供的信息足以继续本章，一旦您完成本章，您将能够自己实现许多有趣的 AJAX 功能。但是，请记住，这个 jQuery 介绍只是冰山一角。如果您想全面了解 jQuery 框架，我强烈建议您阅读 Packt Publishing 的*Learning jQuery*，因为它更详细地介绍了 jQuery。您可以在[`www.packtpub.com/jQuery`](http://www.packtpub.com/jQuery)了解更多关于这本书的信息。

# 实现推文搜索

我们将通过实现实时搜索来引入 AJAX 到我们的应用程序中。这个功能背后的想法很简单：当用户在文本字段中输入一些关键词并点击搜索时，一个脚本在后台工作，获取搜索结果并在同一个页面上呈现它们。搜索页面不会重新加载，从而节省带宽，并提供更好、更具响应性的用户体验。

在我们开始实现这个功能之前，我们需要牢记一个重要的规则，即在使用 AJAX 时编写应用程序，确保它在没有 AJAX 支持的浏览器和没有启用 JavaScript 的用户中也能正常工作。如果你这样做，你就确保每个人都能使用你的应用程序。

## 实现搜索

因此，在我们使用 AJAX 之前，让我们编写一个简单的视图，通过标题搜索书签。首先，我们需要创建一个搜索表单，所以打开`tweets/forms.py`文件，并添加以下类：

```py
class SearchForm(forms.Form):
query = forms.CharField(label='Enter a keyword to search for',
widget=forms.TextInput(attrs={'size': 32, 'class':'form-control'}))
```

正如你所看到的，这是一个非常简单的表单类，只有一个文本字段。用户将使用这个字段输入搜索关键词。接下来，让我们创建一个视图来进行搜索。打开`tweets/views.py`文件，并输入以下代码：

```py
class Search(View):
  """Search all tweets with query /search/?query=<query> URL"""
  def get(self, request):
    form = SearchForm()
    params = dict()
    params["search"] = form
  return render(request, 'search.html', params)

  def post(self, request):
    form = SearchForm(request.POST)
    if form.is_valid():
    query = form.cleaned_data['query']
    tweets = Tweet.objects.filter(text__icontains=query)
    context = Context({"query": query, "tweets": tweets})
    return_str = render_to_string('partials/_tweet_search.html', context)
  return HttpResponse(json.dumps(return_str), content_type="application/json")
  else:
    HttpResponseRedirect("/search")
```

除了一些方法调用，这个视图应该非常容易理解。如果你看一下`get`请求，它非常简单，因为它准备搜索表单，然后呈现它。

`post()`方法是所有魔法发生的地方。当我们呈现搜索结果时，它只是一个带有搜索表单的布局呈现，也就是说，如果你看一下我们创建的名为`search.html`的新文件，你会看到以下内容：

```py
{% extends "base.html" %}
{% load staticfiles %}
{% block content %}

<div class="row clearfix">
  <div class="col-md-6 col-md-offset-3 column">
    <form id="search-form" action="" method="post">{% csrf_token %}
      <div class="input-group input-group-sm">
      {{ search.query.errors }}
      {{ search.query }}
        <span class="input-group-btn">
          <button class="btn btn-search" type="submit">search</button>
        </span>
      </div><!-- /input-group -->
    </form>
  </div>
  <div class="col-md-12 column tweets">
  </div>
</div>
{% endblock %}
{% block js %}
  <script src="img/search.js' %}"></script>
{% endblock %}
```

如果你仔细观察，你会看到一个名为`{% block js %}`的新部分的包含。这里使用的概念与`{% block content %}`块相同，也就是说，这里声明的内容将在`base.html`文件中呈现。进一步看，再看修改后的`base.html`文件，我们可以看到以下内容：

```py
{% load staticfiles %}
  <html>
    <head>
      <link href="{% static 'css/bootstrap.min.css' %}"
        rel="stylesheet" media="screen">
        {% block css %}
        {% endblock %}
    </head>
    <body>
      <nav class="navbar navbar-default" role="navigation">
        <a class="navbar-brand" href="#">MyTweets</a>
        <p class="navbar-text navbar-right">User Profile Page</p>
      </nav>
      <div class="container">
        {% block content %}
        {% endblock %}
      </div>
      <nav class="navbar navbar-default navbar-fixed-bottom" role="navigation">
        <p class="navbar-text navbar-right">Footer </p>
      </nav>
      <script src="img/jquery-2.1.1.min.js' %}"></script>
      <script src="img/bootstrap.min.js' %}"></script>
      <script src="img/base.js' %}"></script>
        {% block js %}
        {% endblock %}
    </body>
  </html>
```

上述代码清楚地显示了两个新的内容块，如下所示：

```py
{% block css %}
  {% endblock %}
  {% block js %}
{% endblock %}
```

它们用于包含相应的文件类型，并将文件类型与基础一起呈现，因此使用简单的规则，每页只声明一个 CSS 和 JavaScript 文件，从而使项目的维护变得更加简单。我们将在本书的后面使用调用**assets pipeline**的概念来实现这一点。

现在，回到我们的 AJAX 搜索功能，你会发现这个`search.html`文件与`tweet.html`文件类似。

对于搜索功能，我们将创建一个新的 URL，需要将其附加到以下的`urls.py`文件中：

```py
url(r'^search/$', Search.as_view()),
urls.py
from django.conf.urls import patterns, include, url
from django.contrib import admin
from tweet.views import Index, Profile, PostTweet, HashTagCloud, Search

admin.autodiscover()

urlpatterns = patterns('',
url(r'^$', Index.as_view()),
url(r'^user/(\w+)/$', Profile.as_view()),
url(r'^admin/', include(admin.site.urls)),
url(r'^user/(\w+)/post/$', PostTweet.as_view()),
url(r'^hashTag/(\w+)/$', HashTagCloud.as_view()),
url(r'^search/$', Search.as_view()),
)
```

在`search.html`文件中，我们定义了`search.js`方法；让我们创建这个 JavaScript 文件，它实际上发出了 AJAX 请求：

```py
search.js

$('#search-form').submit(function(e){
$.post('/search/', $(this).serialize(), function(data){
$('.tweets').html(data);
});
e.preventDefault();
});
```

当表单提交时，这段 JavaScript 代码会被触发，它会向`/search`用户发出一个 AJAX post 请求，带有序列化的表单数据，并获得响应。然后，根据获得的响应，它将数据附加到具有类别 tweets 的元素。

如果我们在浏览器中打开用户搜索，它会看起来像下面的截图：

![实现搜索](img/image00303.jpeg)

现在，等等！当这个表单提交时会发生什么？

AJAX 请求发送到搜索类的`post()`方法，如下所示：

```py
def post(self, request):
  form = SearchForm(request.POST)
  if form.is_valid():
    query = form.cleaned_data['query']
    tweets = Tweet.objects.filter(text__icontains=query)
    context = Context({"query": query, "tweets": tweets})
    return_str = render_to_string('partials/_tweet_search.html', context)
  return HttpResponse(json.dumps(return_str), content_type="application/json")
  else:
    HttpResponseRedirect("/search")
```

我们从`request.POST`方法中提取表单验证；如果表单有效，就从表单对象中提取查询。

然后，`tweets = Tweet.objects.filter(text__icontains===query)`方法搜索给定查询项的子字符串匹配。

搜索是使用`Tweets.objects`模块中的`filter`方法进行的。你可以把它看作是 Django 模型中`SELECT`语句的等价物。它接收搜索条件作为参数，并返回搜索结果。每个参数的名称必须遵循以下命名约定：

```py
field__operator
```

请注意，`field` 和 `operator` 变量之间用两个下划线分隔：field 是我们想要搜索的字段的名称，operator 是我们想要使用的查找方法。以下是常用操作符的列表：

+   `exact`: 参数的值与字段的精确匹配

+   `contains`: 该字段包含参数的值

+   `startswith`: 该字段以参数的值开头

+   `lt`: 该字段小于参数的值

+   `gt`: 该字段大于参数的值

此外，还有前三个操作符的不区分大小写版本：`iexact`、`icontains` 和 `istartswith`，也可以包括在列表中。

我们现在正在做的一件完全不同的事情是：

```py
context = Context({"query": query, "tweets": tweets})
return_str = render_to_string('partials/_tweet_search.html', context)
return HttpResponse(json.dumps(return_str), content_type="application/json")
```

我们的目标是在不重新加载或刷新搜索页面的情况下加载搜索结果。如果是这样，我们之前的渲染方法将如何帮助我们？它不能。我们需要一些方法，可以帮助我们在不重新加载页面的情况下将数据发送到浏览器。

我们广泛使用网页开发中称为**partials**的概念。它们通常是在服务器端生成的小段 HTML 代码片段，以 JSON 格式呈现，然后通过 JavaScript 添加到现有 DOM 中。

为了实现这个方法，我们首先会在现有模板文件夹中创建一个名为 partials 的文件夹，以及一个名为 `_tweet_search.html` 的文件，内容如下：

```py
{% for tweet in tweets %}
  <div class="well">
    <span>{{ tweet.text }}</span>
  </div>
{% endfor %}
{% if not tweets %}
  <div class="well">
    <span> No Tweet found.</span>
  </div>
{% endif %}
```

该代码将在一个良好的框中渲染整个推文对象，或者如果找不到推文对象，它将在框中渲染 `未找到推文`。

前面的概念是在视图中将一个 partial 渲染为字符串，如果我们需要为渲染传递任何参数，我们需要在调用从 partials 生成字符串的地方首先传递它们。要为 partials 传递参数，我们需要创建一个上下文对象，然后传递我们的参数：

```py
context = Context({"query": query, "tweets": tweets})
return_str = render_to_string('partials/_tweet_search.html', context)
```

首先，我们将创建包含 `query`（稍后将使用）和 `tweets` 参数的上下文，并使用 `render_to_string()` 函数。然后，我们可以使用 JSON 将字符串转储到 `HttpResponse()` 函数，如下所示：

```py
return HttpResponse(json.dumps(return_str), content_type="application/json")
```

导入列表如下：

```py
from django.views.generic import View
from django.shortcuts import render
from user_profile.models import User
from models import Tweet, HashTag
from tweet.forms import TweetForm, SearchForm
from django.http import HttpResponseRedirect
from django.template.loader import render_to_string
from django.template import Context
from django.http import HttpResponse
import json
```

就是这样！我们完成了一个基于 AJAX 的推文搜索。搜索 `django` 列出了我们创建的两条推文，如下截图所示：

![实现搜索](img/image00304.jpeg)

继续使用搜索引擎，并且我相信你会更加喜欢 Django。

现在我们有了一个功能性的（尽管非常基本的）搜索页面。搜索功能本身将在后面的章节中得到改进，但对我们来说现在重要的是将 AJAX 引入搜索表单，以便在幕后获取结果并呈现给用户，而无需重新加载页面。由于我们的模块化代码，这个任务将比看起来要简单得多。

# 实现实时搜索推文

在上一节中进行了简单的搜索，现在我们将实现实时搜索，技术上是相同的，但唯一的区别是搜索表单将随着每次按键而提交，并且结果将实时加载。

要实现实时搜索，我们需要做以下两件事：

+   我们需要拦截并处理提交搜索表单的事件。这可以使用 jQuery 的 `submit()` 方法来完成。

+   我们需要使用 AJAX 在后台加载搜索结果，并将它们插入页面中。

jQuery 提供了一个名为 `load()` 的方法，用于从服务器检索页面并将其内容插入到所选元素中。在其最简单的形式中，该函数将远程页面的 URL 作为参数。

我们将在标签上实现实时搜索，也就是说，我们将创建一个新页面，与我们刚刚创建的搜索页面相同，但这将用于标签，并且我们将使用实时标签建议（标签自动完成）。在开始之前，我们需要相同的 Twitter `typeahead` JavaScript 库。

从 [`twitter.github.io/typeahead.js/`](http://twitter.github.io/typeahead.js/) 下载这个库的最新版本。

在本章中，我们下载了版本为 10.05 的库。下载并保存到当前的 JavaScript 文件夹中。

首先，让我们稍微修改我们的搜索视图，以便在接收到名为 AJAX 的额外 GET 变量时，仅返回搜索结果而不是搜索页面的其余部分。我们这样做是为了使客户端的 JavaScript 代码能够轻松地检索搜索结果，而不需要搜索页面的其余部分的 HTML 格式。这可以通过在请求时简单地使用 `bookmark_list.html` 模板而不是 `search.html` 模板来实现。

GET 包含关键的 AJAX 参数。打开 `bookmarks/views.py` 文件并修改 `search_page` 参数（朝文件末尾），使其如下所示：

```py
def search_page(request):
  [...]
  variables = RequestContext(request, {
    'form': form,
    'bookmarks': bookmarks,
    'show_results': show_results,
    'show_tags': True,
    'show_user': True
  })
  if request.GET.has_key('AJAX'):):):
    return render_to_response('bookmark_list.html', variables)
  else:
    return render_to_response('search.html', variables)
```

接下来，在 `site_media` 目录中创建一个名为 `search.js` 的文件，并将其链接到 `templates/search.html` 文件，如下所示：

```py
{% extends "base.html" %}
  {% block external %}
    <script type="text/javascript" src="img/search.js">
    </script>
  {% endblock %}
{% block title %}Search Bookmarks{% endblock %}
{% block head %}Search Bookmarks{% endblock %}
[...]
```

现在是有趣的部分！让我们创建一个函数，加载搜索结果并将它们插入相应的 `div` 标签中。在 `site_media/search.js` 文件中写入以下代码：

```py
function search_submit() {
  var query = $("#id_query").val();
  $("#search-results").load(
    "/search/?AJAX&query=" + encodeURIComponent(query)
  );
return false;
}
```

让我们逐行浏览这个函数：

+   该函数首先使用 `val()` 方法从文本字段中获取查询字符串。

+   我们使用 `load()` 方法从 `search_page` 视图获取搜索结果，并将搜索结果插入到 `#search-results` div 中。首先对查询调用 `encodeURIComponent` 参数构造请求 URL，它的工作方式与我们在 Django 模板中使用的 `urlencode` 过滤器完全相同。调用这个函数很重要，以确保即使用户在文本字段中输入特殊字符，如 `&`，构造的 URL 仍然有效。在转义查询后，我们将其与 `/search/?AJAX&query=` 参数连接起来。这个 URL 调用 `search_page` 视图，并将 GET 变量的 AJAX 参数和查询传递给它。视图返回搜索结果，`load()` 方法便将结果加载到 `#search-results` div 中。

+   我们从函数中返回 `False`，告诉浏览器在调用处理程序后不要提交表单。如果我们在函数中不返回 `False`，浏览器将继续像往常一样提交表单，而我们不希望这样。

还有一个小细节：在何处以及何时应该将 `search_submit` 参数附加到搜索表单的提交事件上？在编写 JavaScript 时的一个经验法则是，在文档完成加载之前，我们不能操作文档树中的元素。因此，我们的函数必须在搜索页面加载完成后立即调用。幸运的是，jQuery 提供了一种在 HTML 文档加载时执行函数的方法。让我们通过将以下代码附加到 `site_media/search.js` 文件来利用它：

```py
$(document).ready(function () {
  $("#search-form").submit(search_submit);
});
```

`$(document)` 函数选择当前页面的文档元素。请注意，`document` 变量周围没有引号；它是浏览器提供的变量，而不是字符串。

`ready()` 方法接受一个函数，并在所选元素完成加载后立即执行它。因此，实际上，我们告诉 jQuery 在 HTML 文档加载完成后立即执行传递的函数。我们将一个匿名函数传递给 `ready()` 方法，这个函数简单地将 `search_submit` 参数绑定到 `#search-form` 表单的提交事件上。

就是这样。我们用不到十五行的代码实现了实时搜索。要测试新功能，转到 `http://127.0.0.1:8000/search/`，提交查询，并注意结果如何在不重新加载页面的情况下显示。

本节涵盖的信息可以应用于任何需要在后台处理而无需重新加载页面的表单。例如，您可以创建一个带有预览按钮的评论表单，该按钮在同一页面上加载预览而无需重新加载。在下一节中，我们将增强用户页面，使用户可以在原地编辑书签而无需离开用户页面。

# 在不加载单独页面的情况下原地编辑推文

编辑发布的内容是网站上非常常见的任务。通常通过在内容旁边提供一个“编辑”链接来实现。当用户点击链接时，该链接会将用户带到另一个页面上的一个表单，用户可以在那里编辑内容。用户提交表单后，会被重定向回内容页面。

另一方面，想象一下，您可以在不离开内容页面的情况下编辑内容。当您点击“编辑”按钮时，内容会被一个表单替换。当您提交表单时，它会消失，更新后的内容会出现在原来的位置。所有操作都在同一个页面上进行；使用 JavaScript 和 AJAX 来完成表单的渲染和提交。这样的工作流程会更直观和响应更快吗？

上述描述的技术称为**原地编辑**。它现在在 Web 应用程序中变得更加普遍。我们将通过让用户在用户页面上原地编辑书签来实现此功能。

由于我们的应用程序尚不支持编辑书签，我们将首先实现这一点，然后修改编辑过程以在原地工作。

## 实现书签编辑

我们已经拥有大部分需要实现书签编辑的部分。如果您回忆一下前一章，我们在 `bookmarks/views.py` 文件中实现了 `bookmark_save_page` 视图，以便如果用户尝试多次保存相同的 URL，则更新相同的书签而不是创建副本。这得益于数据模型提供的 `get_or_create()` 方法，这个小细节极大地简化了书签编辑的实现。我们需要做的是：

+   我们将要编辑的书签的 URL 作为名为 URL 的 GET 变量传递给 `bookmark_save_page` 视图。

+   我们修改 `bookmark_save_page` 视图，以便在接收到 GET 变量时填充书签表单的字段。该表单将填充与传递的 URL 对应的书签的数据。

当填充的表单被提交时，书签将被更新，就像我们之前解释的那样，因为它看起来好像用户又提交了相同的 URL。

在我们实现上述描述的技术之前，让我们通过将保存书签的部分移动到一个单独的函数中来减少 `bookmark_save_page` 视图的大小。我们将称此函数为 `_bookmark_save`。名称开头的下划线告诉 Python 在导入视图模块时不要导入此函数。该函数期望请求和有效的表单对象作为参数；它将根据表单数据保存书签并返回该书签。

打开 `bookmarks/views.py` 文件并创建以下函数；如果愿意，可以从 `bookmark_save_page` 视图中复制并粘贴代码，因为我们除了最后的 `return` 语句外不会对其进行任何更改：

```py
def _bookmark_save(request, form):
  # Create or get link.
  link, dummy = \
  Link.objects.get_or_create(url=form.clean_data['url'])
  # Create or get bookmark.
  bookmark, created = Bookmark.objects.get_or_create(
    user=request.user,
    link=link
  )
  # Update bookmark title.
  bookmark.title = form.clean_data['title']
  # If the bookmark is being updated, clear old tag list.
  if not created:
    bookmark.tag_set.clear()
    # Create new tag list.
    tag_names = form.clean_data['tags'].split()
    for tag_name in tag_names:
      tag, dummy = Tag.objects.get_or_create(name=tag_name)
      bookmark.tag_set.add(tag)
      # Save bookmark to database and return it.
      bookmark.save()
    return bookmark
    Now in the same file, replace the code that you removed from bookmark_save_page
    with a call to _bookmark_save :
      @login_required
      def bookmark_save_page(request):
        if request.method == 'POST':
          form = BookmarkSaveForm(request.POST)
        if form.is_valid():
          bookmark = _bookmark_save(request, form)
          return HttpResponseRedirect(
            '/user/%s/' % request.user.username
          )
        else:
          form = BookmarkSaveForm()
          variables = RequestContext(request, {
            'form': form
          })
        return render_to_response('bookmark_save.html', variables)
```

`bookmark_save_page` 视图中的当前逻辑如下：

[伪代码]

```py
if there is POST data:
  Validate and save bookmark.
  Redirect to user page.
else:
  Create an empty form.
Render page.
```

要实现书签编辑，我们需要稍微修改逻辑，如下所示：

[伪代码]

```py
if there is POST data:
  Validate and save bookmark.
  Redirect to user page.
  else if there is a URL in GET data:
    Create a form an populate it with the URL's bookmark.
  else:
    Create an empty form.
Render page.
```

让我们将上述伪代码翻译成 Python。修改 `bookmarks/views.py` 文件中的 `bookmark_save_page` 视图，使其看起来像以下代码（新代码已突出显示）：

```py
from django.core.exceptions import ObjectDoesNotExist
@login_required
def bookmark_save_page(request):
  if request.method == 'POST':
    form = BookmarkSaveForm(request.POST)
      if form.is_valid():
        bookmark = _bookmark_save(request, form)
        return HttpResponseRedirect(
          '/user/%s/' % request.user.username)
        elif request.GET.has_key('url'):):):
          url = request.GET['url']
          title = ''
          tags = ''
        try:
          link = Link.objects.get(url=url)
          bookmark = Bookmark.objects.get(
            link=link,
            user=request.user
          )
        title = bookmark.title
        tags = ' '.join(
          tag.name for tag in bookmark.tag_set.all()
        )
        except ObjectDoesNotExist:
          pass
        form = BookmarkSaveForm({
          'url': url,
          'title': title,
          'tags': tags
        })
        else:
          form = BookmarkSaveForm()
          variables = RequestContext(request, {
            'form': form
          })
        return render_to_response('bookmark_save.html', variables)
```

代码的这一新部分首先检查是否存在名为 URL 的 GET 变量。如果是这样，它将加载此 URL 的相应`Link`和`Bookmark`对象，并将所有数据绑定到书签保存表单。您可能会想知道为什么我们在 try-except 结构中加载`Link`和`Bookmark`对象，并默默地忽略异常。

确实，如果没有找到请求的 URL 的书签，引发 HTTP 404 异常是完全有效的。然而，我们的代码选择在这种情况下只填充 URL 字段，留下标题和标签字段为空。

现在，在用户页面的每个书签旁边添加**编辑**链接。打开`templates/bookmark_list.html`文件并插入突出显示的代码：

```py
{% if bookmarks %}
  <ul class="bookmarks">
    {% for bookmark in bookmarks %}
      <li>
        <a href="{{ bookmark.link.url }}" class="title">
        {{ bookmark.title|escape }}</a>
        {% if show_edit %}
          <a href="/save/?url={{ bookmark.link.url|urlencode }}"
          class="edit">[edit]</a>
        {% endif %}
      <br />
      {% if show_tags %}
        Tags:
          {% if bookmark.tag_set.all %}
            <ul class="tags">
              {% for tag in bookmark.tag_set.all %}
                <li><a href="/tag/{{ tag.name|urlencode }}/">
              {{ tag.name|escape }}</a></li>
              {% endfor %}
            </ul>
      {% else %}
        None.
      {% endif %}
      <br />
[...]
```

注意我们是如何通过将书签的 URL 附加到`/save/?url= {{ bookmark.link.url|urlencode }}`来构建编辑链接的。

此外，由于我们只想在用户页面上显示编辑链接，模板只在`show_edit`标志设置为`True`时呈现这些链接。否则，让用户编辑其他人的链接是没有意义的。现在打开`bookmarks/views.py`文件，并在`user_page`标志的模板变量中添加`show_edit`标志：

```py
def user_page(request, username):
  user = get_object_or_404(User, username=username)
  bookmarks = user.bookmark_set.order_by('-id')
  variables = RequestContext(request, {
    'bookmarks': bookmarks,
    'username': username,
    'show_tags': True,
    'show_edit': username == request.user.username,
  })
return render_to_response('user_page.html', variables)
```

`username == request.user.username`表达式仅在用户查看自己的页面时评估为`True`，这正是我们想要的。

最后，我建议您稍微减小编辑链接的字体大小。打开`site_media/style.css`文件并将以下内容附加到其末尾：

```py
ul.bookmarks .edit {
  font-size: 70%;
}
```

我们完成了！在继续之前，随意导航到您的用户页面并尝试编辑书签。

## 实现书签的原地编辑

现在我们已经实现了书签编辑，让我们转向令人兴奋的部分：使用 AJAX 添加原地编辑！

我们的方法是：

+   我们将拦截点击编辑链接的事件，并使用 AJAX 从服务器加载书签编辑表单。然后我们将用编辑表单替换页面上的书签。

+   当用户提交编辑表单时，我们将拦截提交事件，并使用 AJAX 将更新后的书签发送到服务器。

+   服务器保存书签并返回新书签的 HTML 表示。然后我们将用服务器返回的标记替换页面上的编辑表单。

我们将使用与实时搜索非常相似的方法来实现前面的过程。首先，我们将修改`bookmark_save_page`视图，以便在 GET 变量称为 AJAX 存在时响应 AJAX 请求。接下来，我们将编写 JavaScript 代码，从视图中检索编辑表单，当用户提交此表单时，将书签数据发送回服务器。

由于我们希望从`bookmark_save_page`视图返回一个编辑表单的标记给 AJAX 脚本，让我们稍微重构一下我们的模板。在模板中创建一个名为`bookmark_save_form.html`的文件，并将书签保存表单从`bookmark_save.html`文件移动到这个新文件中：

```py
<form id="save-form" method="post" action="/save/">
  {{ form.as_p }}
  <input type="submit" value="save" />
</form>
```

请注意，我们还更改了表单的 action 属性为`/save/`并为其赋予了一个 ID。这对于表单在用户页面以及书签提交页面上的工作是必要的。

接下来，在`bookmark_save.html`文件中包含这个新模板：

```py
{%extends "base.html" %}
{%block title %}Save Bookmark{% endblock %}
{%block head %}Save Bookmark{% endblock %}
{%block content %}
{%include 'bookmark_save_form.html' %}
{%endblock %}
```

*好*，现在我们将表单放在一个单独的模板中。让我们更新`bookmark_save_page`视图，以处理正常和 AJAX 请求。打开`bookmarks/views.py`文件并更新视图，使其看起来像下面修改后的样子（用新加粗的行）：

```py
def bookmark_save_page(request):
  AJAX = request.GET.has_key('AJAX')))
  if request.method == 'POST':
    form = BookmarkSaveForm(request.POST)
    if form.is_valid():
      bookmark = _bookmark_save(form)
        if AJAX:
          variables = RequestContext(request, {
            'bookmarks': [bookmark],
            'show_edit': True,
            'show_tags': True
        })
      return render_to_response('bookmark_list.html', variables)
      else:
        return HttpResponseRedirect(
          '/user/%s/' % request.user.username
        )
      else:
        if AJAX:
          return HttpResponse('failure')
          elif request.GET.has_key('url'):
            url = request.GET['url']
            title = ''
            tags = ''
        try:
          link = Link.objects.get(url=url)
          bookmark = Bookmark.objects.get(link=link, user=request.user)
          title = bookmark.title
          tags = ' '.join(tag.name for tag in bookmark.tag_set.all())
        except:::
          pass
          form = BookmarkSaveForm({
            'url': url,
            'title': title,
            'tags': tags
          })
        else:
          form = BookmarkSaveForm()
          variables = RequestContext(request, {
            'form': form
          })
          if AJAX:
            return render_to_response(
              'bookmark_save_form.html',
              variables
            )
            else:
              return render_to_response(
                'bookmark_save.html',
                variables
              )
```

让我们分别检查每个突出显示的部分：

```py
AJAX = request.GET.has_key('AJAX')
```

在方法的开头，我们将检查是否存在名为 AJAX 的 GET 变量。我们将结果存储在名为 AJAX 的变量中。稍后在方法中，我们可以使用这个变量来检查我们是否正在处理 AJAX 请求：

```py
if condition:
  if form.is_valid():
    bookmark = _bookmark_save(form)
    if AJAX:
      variables = RequestContext(request, {
        'bookmarks': [bookmark],
         'show_edit': True,
         'show_tags': True
      })
    return render_to_response('bookmark_list.html', variables)
    else:
      return HttpResponseRedirect('/user/%s/' % request.user.username)
    else:
      if AJAX:
        return HttpResponse('failure')
```

如果我们收到 POST 请求，我们检查提交的表单是否有效。如果有效，我们保存书签。接下来，我们检查这是否是一个 AJAX 请求。如果是，我们使用`bookmark_list.html`模板呈现保存的书签，并将其返回给请求脚本。否则，这是一个正常的表单提交，因此我们将用户重定向到他们的用户页面。另一方面，如果表单无效，我们只会像处理 AJAX 请求一样返回字符串`'failure'`，我们将通过在 JavaScript 中显示错误对话框来响应。如果是正常请求，则无需执行任何操作，因为页面将重新加载，并且表单将显示输入中的任何错误：

```py
if AJAX:
  return render_to_response('bookmark_save_form.html', variables)
  else:
    return render_to_response('bookmark_save.html', variables)
```

这在方法的末尾进行检查。如果没有 POST 数据，即执行到这一点，这意味着我们应该呈现一个表单并返回它。如果是 AJAX 请求，则使用`bookmark_save_form.html`模板，否则将其保存为 HTML 文件。

我们的视图现在已准备好为 AJAX 请求提供服务，也可以处理正常的页面请求。让我们编写 JavaScript 代码，以利用更新后的视图。在`site_media`文件夹中创建一个名为`bookmark_edit.js`的新文件。但是，在向其中添加任何代码之前，让我们将`bookmark_edit.js`文件链接到`user_page.html`模板。打开`user_page.html`文件，并进行以下修改：

```py
{% extends "base.html" %}
  {% block external %}
    <script type="text/javascript" src="img/bookmark_edit.js">
    </script>
  {% endblock %}
  {% block title %}{{ username }}{% endblock %}
  {% block head %}Bookmarks for {{ username }}{% endblock %}
  {% block content %}
    {% include 'bookmark_list.html' %}
  {% endblock %}
```

我们需要在`bookmark_edit.js`文件中编写两个函数：

+   `bookmark_edit`：此函数处理编辑链接的点击。它从服务器加载编辑表单，并用此表单替换书签。

+   `bookmark_save`：此函数处理编辑表单的提交。它将表单数据发送到服务器，并用服务器返回的书签 HTML 替换表单。

让我们从第一个函数开始。打开`site_media/bookmark_edit.js`文件，并在其中编写以下代码：

```py
function bookmark_edit() {
  var item = $(this).parent();
  var url = item.find(".title").attr("href");
  item.load("/save/?AJAX&url=" + escape(url), null, function () {
    $("#save-form").submit(bookmark_save);
  });
  return false;
}
```

因为这个函数处理编辑链接上的点击事件，所以`this`变量指的是编辑链接本身。将其包装在 jQuery `$()`函数中并调用`parent()`函数返回编辑链接的父元素，即书签的`<li>`元素（在 Firebug 控制台中尝试一下，看看自己是否能看到相同的结果）。

在获取书签的`<li>`元素的引用之后，我们获取书签的标题的引用，并使用`attr()`方法从中提取书签的 URL。

接下来，我们使用`load()`方法将编辑表单放置在书签的 HTML 文件中。这次，我们在 URL 之外调用`load()`方法时，还使用了两个额外的参数。`load()`函数接受两个可选参数，如下所示：

+   如果我们发送 POST 请求，则它接受键或值对的对象。由于我们使用 GET 请求从服务器端视图获取编辑表单，因此对于此参数，我们传递 null。

+   它接受一个函数，当 jQuery 完成将 URL 加载到所选元素时调用该函数。我们传递的函数将`bookmark_save()`方法（接下来我们将要编写的方法）附加到刚刚检索到的表单上。

最后，该函数返回`False`，告诉浏览器不要跟随编辑链接。现在我们需要使用`$(document).ready()`将`bookmark_edit()`函数附加到单击编辑链接的事件上：

```py
$(document).ready(function () {
  $("ul.bookmarks .edit").click(bookmark_edit);
});
```

如果您在编写此函数后尝试在用户页面中编辑书签，则应该会出现编辑表单，但是您还应该在 Firebug 控制台中收到 JavaScript 错误消息，因为`bookmark_save()`函数未定义，所以让我们来编写它：

```py
function bookmark_save() {
  var item = $(this).parent();
  var data = {
    url: item.find("#id_url").val(),
    title: item.find("#id_title").val(),
    tags: item.find("#id_tags").val()
  };
  $.post("/save/?AJAX", data, function (result) {
    if (result != "failure") {
      item.before($("li", result).get(0));
      item.remove();
      $("ul.bookmarks .edit").click(bookmark_edit);
    }
    else {
      alert("Failed to validate bookmark before saving.");
    }
  });
  return false;
}
```

在这里，`this`变量指的是编辑表单，因为我们处理提交表单的事件。该函数首先通过检索对表单的父元素（再次是书签的`<li>`元素）的引用来开始。接下来，该函数使用每个表单字段的 ID 和`val()`方法从表单中检索更新的数据。

然后它使用一个名为`$.post()`的方法将数据发送回服务器。最后，它返回`False`以防止浏览器提交表单。

您可能已经猜到，`$.post()`函数是一个发送 POST 请求到服务器的 jQuery 方法。它有三个参数，如下：

+   POST 请求目标的 URL。

+   表示 POST 数据的键/值对对象。

+   当请求完成时调用的函数。服务器响应作为字符串参数传递给此函数。

值得一提的是，jQuery 提供了一个名为`$.get()`的方法，用于向服务器发送 GET 请求。它接受与`$.post()`函数相同类型的参数。我们使用`$.post()`方法将更新的书签数据发送到`bookmark_save_page`视图。正如前面几段讨论的那样，如果视图成功保存书签，则返回更新的书签 HTML。否则，它返回`failure`字符串。

因此，我们检查服务器返回的结果是否是“失败”。如果请求成功，我们使用`before()`方法在旧书签之前插入新书签，并使用`remove()`方法从 HTML 文档中删除旧书签。另一方面，如果请求失败，我们会显示一个显示失败的警报框。

在我们完成本节之前还有一些小事情。为什么我们插入`$("li",result).get(0)`方法而不是结果本身？如果您检查`bookmark_save_page`视图，您会看到它使用`bookmark_list.html`模板来构建书签的 HTML。然而，`bookmark_list.html`模板返回包装在`<ul>`标签中的书签`<li>`元素。基本上，`$("li", result).get(0)`方法告诉 jQuery 从结果中提取第一个`<li>`元素，这就是我们想要的元素。正如您从前面的片段中看到的，您可以使用 jQuery `$()`函数通过将该字符串作为函数的第二个参数传递来选择 HTML 字符串中的元素。

`bookmark_submit`模板是从`bookmark_edit`模板中的事件附加的，因此我们不需要在`$(document).ready()`方法中做任何事情。

最后，在将更新的书签加载到页面后，我们再次调用`$("ul.bookmarks.edit").click(bookmark_edit)`方法，将`bookmark_edit`模板附加到新加载的编辑链接上。如果不这样做并尝试两次编辑书签，第二次点击编辑链接将带您到一个单独的表单页面。

当您完成编写 JavaScript 代码后，打开浏览器并转到您的用户页面，尝试使用新功能。编辑书签，保存它们，并注意到如何在页面上立即反映出更改而无需重新加载。

现在您已经完成了这一部分，应该对就地编辑的实现有很好的理解。还有许多其他情况下，这个功能可以很有用，例如，可以用来在同一页上编辑文章或评论，而不必跳转到位于不同 URL 上的表单进行编辑。

在下一节中，我们将实现一个帮助用户在提交书签时输入标签的第三个常见的 AJAX 功能。

# 在提交推文时自动完成标签

我们将在本章中要实现的最后一个 AJAX 增强功能是标签的自动完成。自动完成的概念是在 Google 发布其 Suggest 搜索界面时进入 Web 应用程序的。Suggest 通过根据用户到目前为止输入的内容，在搜索输入字段下方显示最受欢迎的搜索查询。这也类似于集成开发环境中的代码编辑器根据您的输入提供代码完成建议。这个功能通过让用户输入他们想要的单词的几个字符，然后让他们从列表中选择而不必完全输入来节省时间。

我们将通过在提交书签时提供建议来实现此功能，但我们不打算从头开始编写此功能，而是要使用 jQuery 插件来实现它。jQuery 拥有一个不断增长的大型插件列表，提供各种功能。安装插件与安装 jQuery 本身没有什么不同。您下载一个（或多个）文件并将它们链接到您的模板，然后编写几行 JavaScript 代码来激活插件。

您可以通过导航到[`docs.jquery.com/Plugins`](http://docs.jquery.com/Plugins)来浏览可用的 jQuery 插件列表。在列表中搜索 autocomplete 插件并下载它，或者您可以直接从[`bassistance.de/jquery-plugins/jquery-plugin-autocomplete/`](http://bassistance.de/jquery-plugins/jquery-plugin-autocomplete/)获取它。

您将收到一个包含许多文件的 zip 存档文件。将以下文件（可以在`jquery/autocomplete/scroll`目录中找到）提取到`site_media`目录中：

+   **jquery.autocomplete.css**

+   **dimensions.js**

+   **jquery.bgiframe.min.js**

+   **jquery.autocomplete.js**

由于我们希望在书签提交页面上提供自动完成功能，请在`site_media`文件夹中创建一个名为`tag_autocomplete.js`的空文件。然后打开`templates/bookmark_save.html`文件，并将所有前述文件链接到它：

```py
{% extends "base.html" %}
  {% block external %}
  <link rel="stylesheet"
  href="/site_media/jquery.autocomplete.css" type="text/css" />
  <script type="text/javascript"
  src="img/dimensions.js"> </script>
  <script type="text/javascript"
  src="img/jquery.bgiframe.min.js"> </script>
  <script type="text/javascript"
  src="img/jquery.autocomplete.js"> </script>
  <script type="text/javascript"
  src="img/tag_autocomplete.js"> </script>
  {% endblock %}
  {% block title %}Save Bookmark{% endblock %}
  {% block head %}Save Bookmark{% endblock %}
[...]
```

我们现在已经完成了插件的安装。如果你阅读它的文档，你会发现这个插件是通过在选定的输入元素上调用一个名为`autocomplete()`的方法来激活的。`autocomplete()`函数接受以下参数：

+   **服务器端 URL**：对于这一点，插件向这个 URL 发送一个 GET 请求，其中包含到目前为止已经输入的内容，并期望服务器返回一组建议。

+   **可用于指定各种选项的对象**：我们感兴趣的选项有很多。这个选项有一个布尔变量，告诉插件输入字段用于输入多个值（记住我们使用同一个文本字段输入所有标签），以及用于告诉插件哪个字符串分隔多个条目的多个分隔符。在我们的情况下，它是一个单个空格字符。

因此，在激活插件之前，我们需要编写一个视图，接收用户输入并返回一组建议。打开`bookmarks/views.py`文件，并将以下内容追加到文件末尾：

```py
def AJAX_tag_autocomplete(request):
  if request.GET.has_key('q'):):):
    tags = \
    Tag.objects.filter(name__istartswith=request.GET['q'])[:10]
  return HttpResponse('\n'.join(tag.name for tag in tags))
return HttpResponse()
```

`autocomplete()`插件将用户输入发送到名为`q`的 GET 变量。因此，我们可以验证该变量是否存在，并构建一个以该变量值开头的标签列表。这是使用我们在本章前面学到的`filter()`方法和`istartswith`运算符完成的。我们只取前十个结果，以避免给用户带来过多的建议，并减少带宽和性能成本。最后，我们将建议连接成一个由换行符分隔的单个字符串，将字符串包装成一个`HttpResponse`对象，并返回它。

有了建议视图准备好后，在`urls.py`文件中为插件添加一个 URL 条目，如下所示：

```py
urlpatterns = patterns('',
  # AJAX
  (r'^AJAX/tag/autocomplete/$', AJAX_tag_autocomplete),
)
```

现在在`site_media/tag_autocomplete.js`文件中输入以下代码，激活标签输入字段上的插件：

```py
$(document).ready(function () {
  $("#id_tags").autocomplete(
    '/AJAX/tag/autocomplete/',
    {multiple: true, multipleSeparator: ' '}
  );
});
```

该代码将一个匿名函数传递给`$(document).ready()`方法。这个函数在标签输入字段上调用`autocomplete()`函数，并传递了我们之前讨论过的参数。

这几行代码就是我们实现标签自动完成所需要的全部内容。要测试新功能，请导航到`http://127.0.0.1:8000/save/`的书签提交表单，并尝试在标签字段中输入一个或两个字符。根据数据库中可用的标签，应该会出现建议。

有了这个功能，我们完成了这一章。我们涵盖了很多材料，并学习了许多令人兴奋的技术和技巧。阅读完本章后，你应该能够想到并实现许多其他用户界面的增强功能，比如在用户页面上删除书签的能力，或者通过标签实时浏览书签等等。

下一章将转向一个不同的主题：我们将让用户对他们最喜欢的书签进行投票和评论，我们的应用程序的首页将不再像现在这样空荡荡了！

# 总结

呼，这是一个很长的章节，但希望你从中学到了很多！我们从学习 jQuery 框架和如何将其整合到我们的 Django 项目开始了这一章。之后，我们在我们的书签应用程序中实现了三个令人兴奋的功能：实时搜索、就地编辑和自动完成。

下一章将是另一个令人兴奋的章节。我们将让用户提交书签到首页并为他们最喜欢的书签投票。我们还将让用户对书签进行评论。所以，请继续阅读！
