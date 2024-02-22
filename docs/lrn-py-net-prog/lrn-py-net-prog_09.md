# 第九章：网络应用

在第二章 *HTTP 和网络工作*中，我们探讨了 HTTP 协议——万维网主要使用的协议，并学习了如何使用 Python 作为 HTTP 客户端。在第三章 *API 实践*中，我们扩展了这一点，并研究了消费 Web API 的方法。在本章中，我们将把重点转向，看看如何使用 Python 构建应用程序，以响应 HTTP 请求。

在本章中，我们将涵盖以下内容：

+   Python web frameworks

+   一个 Python 网络应用

+   托管 Python 和 WSGI

我应该提前指出，托管现代 Web 应用是一个非常庞大的主题，完整的处理远远超出了本书的范围，我们的重点是将 Python 代码应用于网络问题。诸如数据库访问、选择和配置负载均衡器和反向代理、容器化以及保持整个系统运行所需的系统管理技术等主题在这里不会涉及。然而，有许多在线资源可以为您提供一个起点，我们将尽量在相关的地方提及尽可能多的资源。

话虽如此，上述列出的技术并不是创建和提供基于 Python 的 Web 应用程序的要求，它们只是在服务达到规模时所需的。正如我们将看到的，对于易于管理的小规模应用程序托管也有选择。

# Web 服务器中包含什么？

要了解如何使用 Python 来响应 HTTP 请求，我们需要了解一些通常需要发生的事情，以便响应请求，以及已经存在的工具和模式。

基本的 HTTP 请求和响应可能如下所示：

![Web 服务器中包含什么？](img/6008OS_09_01.jpg)

在这里，我们的 Web 客户端向服务器发送 HTTP 请求，其中 Web 服务器程序解释请求，创建适当的 HTTP 响应，并将其发送回来。在这种情况下，响应主体只是从中读取的 HTML 文件的内容，响应头由 Web 服务器程序添加。

Web 服务器负责响应客户端请求的整个过程。它需要执行的基本步骤是：

![Web 服务器中包含什么？](img/6008OS_09_02.jpg)

首先，Web 服务器程序需要接受客户端的 TCP 连接尝试。然后，它通过 TCP 连接从客户端接收 HTTP 请求。服务器需要在生成 HTTP 响应时保持 TCP 连接打开，并使用连接将响应发送回客户端。服务器在此之后对连接的处理取决于所使用的 HTTP 版本以及请求中可能的 Connection 头的值（有关完整细节，请参阅 RFC [`tools.ietf.org/html/rfc7230#section-6.3`](http://tools.ietf.org/html/rfc7230#section-6.3)）。

一旦 Web 服务器收到请求，它会解析请求，然后生成响应。当请求的 URL 映射到服务器上的有效资源时，服务器将使用该 URL 处的资源进行响应。资源可以是磁盘上的文件（所谓的**静态内容**），如前面的基本 HTTP 请求和响应的图表所示，它可以是一个 HTTP 重定向，或者它可以是一个动态生成的 HTML 页面。如果出现问题，或者 URL 无效，则响应将包含`4xx`或`5xx`范围内的状态代码。准备好响应后，服务器通过 TCP 连接将其发送回客户端。

在 Web 的早期，几乎所有请求的资源都是从磁盘读取的静态文件，Web 服务器可以用一种语言编写，并且可以轻松处理前面图像中显示的所有四个步骤。然而，随着越来越多的动态内容的需求，例如购物篮和数据库驱动的资源，如博客、维基和社交媒体，很快就发现将这些功能硬编码到 Web 服务器本身是不切实际的。相反，Web 服务器内置了设施，允许调用外部代码作为页面生成过程的一部分。

因此，Web 服务器可以用快速的语言（如 C 语言）编写，并处理低级别的 TCP 连接、请求的初始解析和验证以及处理静态内容，但在需要动态响应时，可以调用外部代码来处理页面生成任务。

这个外部代码是我们在谈论 Web 应用程序时通常指的内容。因此，响应过程的职责可以分为以下几个部分，如下图所示：

![Web 服务器中有什么？](img/6008OS_09_03.jpg)

Web 应用程序可以用 Web 服务器能够调用的任何语言编写，提供了很大的灵活性，并允许使用更高级别的语言。这可以大大减少开发新 Web 服务所需的时间。如今有很多语言可以用来编写 Web 应用程序，Python 也不例外。

# Python 和 Web

使用本书中讨论的一些技术，特别是第八章中讨论的技术，可以使用 Python 编写一个完整的 Web 服务器，处理我们在前一节中列出的处理 HTTP 请求的四个步骤。已经有几个正在积极开发的 Web 服务器纯粹用 Python 编写，包括 Gunicorn ([`gunicorn.org`](http://gunicorn.org))和 CherryPy ([`www.cherrypy.org`](http://www.cherrypy.org))。标准库 http.server 模块中甚至有一个非常基本的 HTTP 服务器。

编写一个完整的 HTTP 服务器并不是一项微不足道的任务，详细的处理远远超出了本书的范围。由于已经准备好部署的优秀 Web 服务器的普及，这也不是一个非常常见的需求。如果你确实有这个需求，我建议你首先查看前面提到的 Web 服务器的源代码，更详细地查看第八章中列出的框架，*客户端和服务器应用程序*，并阅读相关 RFC 中的完整 HTTP 规范。您可能还想阅读 WSGI 规范，稍后在 WSGI 部分讨论，以便允许服务器充当其他 Python Web 应用程序的主机。

更强的要求是构建一个 Web 服务应用程序来生成一些动态内容，并快速运行起来。在这种情况下，Python 为我们提供了一些优秀的选择，以 Web 框架的形式。

## Web 框架

Web 框架是位于 Web 服务器和我们的 Python 代码之间的一层，它提供了抽象和简化的 API，用于执行解释 HTTP 请求和生成响应的许多常见操作。理想情况下，它还应该结构良好，引导我们采用经过良好测试的 Web 开发模式。Python Web 应用程序的框架通常是用 Python 编写的，并且可以被视为 Web 应用程序的一部分。

框架提供的基本服务包括：

+   HTTP 请求和响应的抽象

+   URL 空间的管理（路由）

+   Python 代码和标记（模板）的分离

今天有许多 Python 网络框架在使用，以下是一些流行的框架列表，排名不分先后：

+   Django（[www.djangoproject.com](http://www.djangoproject.com)）

+   CherryPy（[www.cherrypy.org](http://www.cherrypy.org)）

+   Flask（[flask.pocoo.org](http://flask.pocoo.org)）

+   Tornado（[www.tornadoweb.org](http://www.tornadoweb.org)）

+   TurboGears（[www.turbogears.org](http://www.turbogears.org)）

+   金字塔（[www.pylonsproject.org](http://www.pylonsproject.org)）

### 注意

维护有关框架的最新列表[`wiki.python.org/moin/WebFrameworks`](https://wiki.python.org/moin/WebFrameworks)和[`docs.python-guide.org/en/latest/scenarios/web/#frameworks`](http://docs.python-guide.org/en/latest/scenarios/web/#frameworks)。

有这么多框架是因为可以采用许多方法来执行它们执行的任务，并且对于它们甚至应该执行的任务有许多不同的观点。

一些框架提供了快速构建简单 Web 应用程序所需的最低功能。这些通常被称为**微框架**，这里最受欢迎的是 Armin Ronacher 的出色的 Flask。尽管它们可能不包括一些重量级框架的功能，但它们通常做得非常好，并提供了钩子，以便轻松扩展更复杂的任务。这允许完全定制的 Web 应用程序开发方法。

其他框架采用更多的电池包含方式，为现代 Web 应用程序的所有常见需求提供支持。这里的主要竞争者是 Django，它包括从模板到表单管理和数据库抽象，甚至完整的开箱即用的基于 Web 的数据库管理界面的所有内容。TurboGears 通过将核心微框架与其他功能的几个已建立的软件包集成来提供类似的功能。

还有其他框架提供支持具有事件驱动架构的 Web 应用程序的功能，例如 Tornado 和 CherryPy。这两者还具有自己内置的生产质量 Web 服务器。

选择一个框架可能是一个棘手的决定，没有正确的答案。我们将快速浏览今天最流行的框架之一，以了解框架可以提供的服务，然后讨论如何选择一个框架的方法。

# Flask-微框架

为了体验使用 Python Web 框架的感觉，我们将使用 Flask 编写一个小应用程序。我们选择 Flask，因为它提供了一个精简的接口，为我们提供了所需的功能，同时让我们编写代码。而且，它不需要任何重要的预配置，我们需要做的就是安装它，就像这样：

```py
**>>> pip install flask**
**Downloading/unpacking flask**

```

Flask 也可以从项目的主页[`flask.pocoo.org`](http://flask.pocoo.org)下载。请注意，要在 Python 3 下运行 Flask，您将需要 Python 3.3 或更高版本。

现在创建一个项目目录，并在目录中创建一个名为`tinyflaskapp.py`的文本文件。我们的应用程序将允许我们浏览 Python 内置函数的文档字符串。将其输入到`tinyflaskapp.py`中：

```py
from flask import Flask, abort
app = Flask(__name__)
app.debug = True

objs = __builtins__.__dict__.items()
docstrings = {name.lower(): obj.__doc__ for name, obj in objs if
              name[0].islower() and hasattr(obj, '__name__')}

@app.route('/')
def index():
    link_template = '<a href="/functions/{}">{}</a></br>'
    links = []
    for func in sorted(docstrings):
        link = link_template.format(func, func)
        links.append(link)
    links_output = '\n'.join(links)
    return '<h1>Python builtins docstrings</h1>\n' + links_output

@app.route('/functions/<func_name>')
def show_docstring(func_name):
    func_name = func_name.lower()
    if func_name in docstrings:
        output = '<h1>{}</h1>\n'.format(func_name)
        output += '<pre>{}</pre>'.format(docstrings[func_name])
        return output
    else:
        abort(404)

if __name__ == '__main__':
    app.run()
```

此代码可以在本书本章的源代码下载中找到，位于`1-init`文件夹中。

Flask 包含一个开发 Web 服务器，因此要尝试我们的应用程序，我们只需要运行以下命令：

```py
**$ python3.4 tinyflaskapp.py**
 *** Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)**
 *** Restarting with stat**

```

我们可以看到 Flask 服务器告诉我们它正在侦听的 IP 地址和端口。现在在 Web 浏览器中连接到它显示的 URL（在前面的示例中，这是`http://127.0.0.1:5000/`），您应该会看到一个列出 Python 内置函数的页面。单击其中一个应该显示一个显示函数名称及其文档字符串的页面。

如果您想在另一个接口或端口上运行服务器，可以更改`app.run()`调用，例如更改为`app.run(host='0.0.0.0', port=5001)`。

让我们来看看我们的代码。从顶部开始，我们通过创建一个 `Flask` 实例来创建我们的 Flask 应用，这里给出了我们主要模块的名称。然后我们将调试模式设置为活动状态，在浏览器中出现问题时提供良好的回溯，并且还设置开发服务器以自动重新加载代码更改而无需重新启动。请注意，调试模式永远不应该在生产应用中保持活动状态！这是因为调试器具有交互元素，允许在服务器上执行代码。默认情况下，调试是关闭的，因此当我们将应用投入生产时，我们只需要删除 `app.config.debug` 行即可。

接下来，我们将内置的函数对象从全局变量中过滤出来，并提取它们的文档字符串以备后用。现在我们有了应用程序的主要部分，我们遇到了 Flask 的第一个超能力：URL 路由。一个 Flask 应用的核心是一组函数，通常称为**视图**，它们处理我们 URL 空间的各个部分的请求——`index()` 和 `show_docstring()` 就是这样的函数。您会看到这两个函数都是由 Flask 装饰器函数 `app.route()` 预先处理的。这告诉 Flask 装饰的函数应该处理我们 URL 空间的哪些部分。也就是说，当一个请求带有与 `app.route()` 装饰器中的模式匹配的 URL 时，将调用具有匹配装饰器的函数来处理请求。视图函数必须返回 Flask 可以返回给客户端的响应，但稍后会详细介绍。

我们的 `index()` 函数的 URL 模式只是站点根目录 `'/'`，这意味着只有对根目录的请求才会由 `index()` 处理。

在 `index()` 中，我们只需将输出的 HTML 编译为字符串——首先是我们链接到函数页面的列表，然后是一个标题——然后我们返回字符串。Flask 获取字符串并创建响应，使用字符串作为响应主体，并添加一些 HTTP 头。特别是对于 `str` 返回值，它将 `Content-Type` 设置为 `text/html`。

`show_docstrings()` 视图也做了类似的事情——它在 HTML 标题标签中返回我们正在查看的内置函数的名称，以及包含在 `<pre>` 标签中的文档字符串（以保留换行和空格）。

有趣的部分是 `app.route('/functions/<func_name>')` 的调用。在这里，我们声明我们的函数页面将位于 `functions` 目录中，并使用 `<func_name>` 段捕获请求的函数名称。Flask 捕获 URL 的尖括号部分，并使其可用于我们的视图。我们通过为 `show_docstring()` 声明 `func_name` 参数将其引入视图命名空间。

在视图中，我们通过查看名称是否出现在 `docstrings` 字典中来检查提供的名称是否有效。如果有效，我们构建并返回相应的 HTML。如果无效，我们通过调用 Flask 的 `abort()` 函数向客户端返回 `404 Not Found` 响应。此函数会引发一个 Flask `HTTPException`，如果我们的应用程序没有处理，Flask 将生成一个错误页面并将其返回给客户端，同时返回相应的状态码（在本例中为 404）。这是在遇到错误请求时快速失败的好方法。

## 模板

从我们之前的视图中可以看出，即使在调皮地省略了通常的 HTML 正式性，比如 `<DOCTYPE>` 和 `<html>` 标签以节省复杂性，但在 Python 代码中构建 HTML 仍然很笨拙。很难对整个页面有所感觉，对于没有 Python 知识的设计师来说，无法进行页面设计。此外，将呈现代码的生成与应用逻辑混合在一起会使两者都更难测试。

几乎所有的 Web 框架都通过使用模板习语来解决这个问题。由于大部分 HTML 是静态的，问题就出现了：为什么还要将它保留在应用程序代码中呢？有了模板，我们可以将 HTML 完全提取到单独的文件中。然后这些文件包括 HTML 代码，包括一些特殊的占位符和逻辑标记，以允许动态元素被插入。

Flask 使用了 Armin Ronacher 的另一个作品*Jinja2*模板引擎来完成这项任务。让我们来适应我们的应用程序来使用模板。在你的项目文件夹中，创建一个名为`templates`的文件夹。在里面，创建三个新的文本文件，`base.html`，`index.html`和`docstring.html`。填写它们如下：

`base.html`文件将是这样的：

```py
<!DOCTYPE html>
<html>
<head>
    <title>Python Builtins Docstrings</title>
</head>
<body>
{% block body %}{% endblock %}
</body>
</html>
```

`index.html`文件将是这样的：

```py
{% extends "base.html" %}
{% block body %}
    <h1>Python Builtins Docstrings</h1>
    <div>
    {% for func in funcs %}
        <div class="menuitem link">
            <a href="/functions/{{ func }}">{{ func }}</a>
        </div>
    {% endfor %}
    </table>
{% endblock %}
```

`docstring.html`文件将是这样的：

```py
{% extends 'base.html' %}
{% block body %}
    <h1>{{ func_name }}</h1>
    <pre>{{ doc }}</pre>
    <p><a href="/">Home</a></p>
{% endblock %}
```

在`tinyflaskapp.py`顶部的`from flask import...`行中添加`render_template`，然后修改你的视图如下：

```py
@app.route('/')
def index():
    return render_template('index.html', funcs=sorted(docstrings))

@app.route('/functions/<func_name>')
def show_docstring(func_name):
    func_name = func_name.lower()
    if func_name in docstrings:
        return render_template('docstring.html',
                               func_name=func_name,
                               doc=docstrings[func_name])
    else:
        abort(404)
```

这段代码可以在本章源代码的`2-templates`文件夹中找到。

注意到视图变得简单得多，HTML 现在更加可读了吗？我们的视图不再手动组合返回字符串，而是简单地调用`render_template()`并返回结果。

那么`render_template()`做了什么呢？它会在`templates`文件夹中查找作为第一个参数提供的文件，读取它，运行文件中的任何处理指令，然后将处理后的 HTML 作为字符串返回。提供给`render_template()`的任何关键字参数都会传递给模板，并在其处理指令中可用。

看看这些模板，我们可以看到它们大部分是 HTML，但是包含一些额外的指令供 Flask 使用，包含在`{{ }}`和`{% %}`标签中。`{{ }}`指令简单地将命名变量的值替换到 HTML 的相应位置。所以例如`docstrings.html`中的`{{ func_name }}`会将我们传递给`render_template()`的`func_name`的值替换进去。

`{% %}`指令包含逻辑和流程控制。例如，`index.html`中的`{% for func in funcs %}`指令循环遍历`funcs`中的值，并重复包含的 HTML 对于每个值。

最后，你可能已经注意到模板允许**继承**。这是由`{% block %}`和`{% extends %}`指令提供的。在`base.html`中，我们声明了一些共享的样板 HTML，然后在`<body>`标签中我们只有一个`{% block body %}`指令。在`index.html`和`docstring.html`中，我们不包括样板 HTML；相反我们`extend`了`base.html`，这意味着这些模板将填充在`base.html`中声明的`block`指令。在`index.html`和`docstring.html`中，我们声明了一个`body block`，Flask 将其内容插入到`base.html`中的 HTML 中，替换匹配的`{% block body %}`。继承允许共享代码的重用，并且可以级联到任意级别。

在 Jinja2 模板指令中还有更多的功能可用；在[`jinja.pocoo.org/docs/dev/templates/`](http://jinja.pocoo.org/docs/dev/templates)查看模板设计者文档以获取完整列表。

## 其他模板引擎

Jinja2 显然不是唯一存在的模板包；你可以在[`wiki.python.org/moin/Templating`](https://wiki.python.org/moin/Templating)找到一个维护的 Python 模板引擎列表。

像框架一样，存在不同的引擎是因为对于什么是一个好的引擎有不同的哲学观念。有些人认为逻辑和表现应该绝对分开，模板中不应该有流程控制和表达式，只提供值替换机制。其他人则采取相反的方式，允许在模板标记中使用完整的 Python 表达式。而其他一些引擎则采取中间路线的方式，比如 Jinja2。还有一些引擎使用完全不同的方案，比如基于 XML 的模板或者通过特殊的 HTML 标签属性声明逻辑。

没有“正确”的方法；最好尝试一些方法，看看哪种对你最有效。然而，如果一个框架有自己的引擎，比如 Django，或者与现有引擎紧密集成，比如 Flask，通常最好使用它们提供的内容，如果可以的话，你通常会更顺利。

## 添加一些样式

目前，我们的页面看起来有点单调。让我们添加一些样式。我们将通过包含一个静态 CSS 文档来实现这一点，但是相同的方法也可以用于包含图像和其他静态内容。本节的代码可以在本章源代码的`3-style`文件夹中找到。

首先，在你的项目文件夹中创建一个新的`static`文件夹，在其中创建一个名为`style.css`的新文本文件。将以下内容保存到其中：

```py
body        { font-family: Sans-Serif; background: white; }
h1          { color: #38b; }
pre         { margin: 0px; font-size: 1.2em; }
.menuitem   { float: left; margin: 1px 1px 0px 0px; }
.link       { width: 100px; padding: 5px 25px; background: #eee; }
.link a      { text-decoration: none; color: #555; }
.link a:hover { font-weight: bold; color: #38b; }
```

接下来，更新你的`base.html`文件的`<head>`部分，使其看起来像这样：

```py
<head>
    <title>Python Builtins Docstrings</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}"/>
</head>
```

注意前面代码中的第三行和第四行——即`<link>`标签——应该在你的代码中是一行。再次在浏览器中尝试你的 Web 应用程序，注意它看起来（希望）更加现代化。

在这里，我们只是在`base.html`中的基本 HTML 中添加了一个样式表，添加了一个指向我们的`static/style.css`文件的`<link>`标签。我们使用 Flask 的`url_for()`函数来实现这一点。`url_for()`函数返回我们 URL 空间中命名部分的路径。在这种情况下，它是特殊的`static`文件夹，默认情况下 Flask 会在我们的 Web 应用程序的根目录中查找。`url_for()`还可以用于获取我们视图函数的路径，例如，`url_for('index')`会返回`/`。

你可以将图像和其他资源放在`static`文件夹中，并以相同的方式引用它们。

## 关于安全性的说明

如果你是新手网页编程，我强烈建议你了解网页应用程序中两种常见的安全漏洞。这两种漏洞都相当容易避免，但如果不加以解决，可能会产生严重后果。

### XSS

第一个是**跨站脚本**（**XSS**）。这是攻击者将恶意脚本代码注入到网站的 HTML 中，导致用户的浏览器在不知情的情况下以该网站的安全上下文执行操作。一个典型的向量是用户提交的信息在没有适当的净化或转义的情况下重新显示给用户。

例如，一个方法是诱使用户访问包含精心制作的`GET`参数的 URL。正如我们在第二章中看到的，*HTTP 和* *与 Web 的工作*，这些参数可以被 Web 服务器用来生成页面，有时它们的内容被包含在响应页面的 HTML 中。如果服务器在显示时不小心用 HTML 转义代码替换 URL 参数中的特殊字符，攻击者可以将可执行代码（例如 Javascript）放入 URL 参数中，并在访问该 URL 时实际执行它。如果他们能够诱使受害者访问该 URL，那么该代码将在用户的浏览器中执行，使攻击者有可能执行用户可以执行的任何操作。

基本的 XSS 预防措施是确保从 Web 应用程序外部接收的任何输入在返回给客户端时都得到适当的转义。Flask 在这方面非常有帮助，因为它默认激活了 Jinja2 的自动转义功能，这意味着我们通过模板渲染的任何内容都会自动受到保护。然而，并非所有的框架都具有这个功能，有些框架需要手动设置。此外，这仅适用于用户生成的内容不能包含标记的情况。在允许用户生成内容中包含一些标记的维基等情况下，你需要更加小心——请参阅本章的`5-search`文件夹中的源代码下载以获取示例。你应该始终确保查看你的框架文档。

### CSRF

第二种攻击形式是**跨站请求伪造**（**CSRF**）。在这种攻击中，网站被欺骗以在用户的安全上下文中执行操作，而用户并不知情或同意。这通常是由 XSS 攻击引发的，导致用户的浏览器在用户登录的情况下在目标站点上执行操作。需要注意的是，这可能会影响网站，即使用户并没有在主动浏览它们；网站通常只有在用户明确注销时才清除 cookie 身份验证令牌，因此从网站和浏览器的角度来看，即使用户停止浏览网站后来自浏览器的任何请求，如果他们没有注销，都将被视为用户仍然登录。

防止 CSRF 攻击的一种技术是使潜在可滥用的操作（例如提交表单）需要一个仅服务器和客户端知道的一次性令牌值。CRSF 攻击通常采取预先组合的 HTTP 请求的形式，模仿用户提交表单或类似操作。然而，如果每次服务器向客户端发送一个表单时都包含一个不同的一次性令牌值，那么攻击者就无法在预先组合的请求中包含这个值，因此攻击尝试可以被检测并拒绝。这种技术对 XSS 发起的攻击和攻击者窃听浏览会话的 HTTP 流量的攻击效果较差。前者很难完全防范，最好的解决方案是确保首先不存在 XSS 漏洞。后者可以通过使用 HTTPS 而不是 HTTP 来减轻。有关更多信息，请参阅下面链接的 OWASP 页面。

不同的框架对提供基于一次性令牌的 CSRF 保护有不同的方法。Flask 没有内置此功能，但很容易添加一些内容，例如：

```py
@app.before_request
def csrf_protect():
    if request.method == "POST":
        token = session.pop('_csrf_token', None)
        if not token or token != request.form.get('_csrf_token'):
            abort(403)

def generate_csrf_token():
    if '_csrf_token' not in session:
        session['_csrf_token'] = some_random_string()
    return session['_csrf_token']

app.jinja_env.globals['csrf_token'] = generate_csrf_token
```

然后在带有表单的模板中，只需执行以下操作：

```py
<form method="post" action="<whatever>">
    <input name="_csrf_token" type="hidden" value="{{ csrf_token() }}">
```

这是来自 Flask 网站的：[`flask.pocoo.org/snippets/3/`](http://flask.pocoo.org/snippets/3/)。虽然这包含了一些我们还没有涵盖的 Flask 功能，包括会话和`@app.before_request()`装饰器，你只需要在你的应用程序中包含上面的代码，并确保在每个表单中包含一个`_` `csrf_token`隐藏输入。另一种方法是使用 Flask-WTF 插件，它提供了与`WTForms`包的集成，该包具有内置的 CSRF 保护。

另一方面，Django 具有内置的保护，但您需要启用并使用它。其他框架各不相同。始终检查您选择的框架的文档。

### 注意

关于 XSS 和 CSRF 的更多信息，请参阅 Flask 和 Django 网站：

+   [`flask.pocoo.org/docs/latest/security/`](http://flask.pocoo.org/docs/latest/security/)

+   [`docs.djangoproject.com/en/1.7/topics/security/`](https://docs.djangoproject.com/en/1.7/topics/security/)

同样在 OWASP 网站上，有一个包含各种与计算机安全相关信息的存储库：

+   [`www.owasp.org/index.php/XSS`](https://www.owasp.org/index.php/XSS)

+   [`www.owasp.org/index.php/CSRF`](https://www.owasp.org/index.php/CSRF)

## 结束框架

这就是我们在 Flask 中的涉足的尽头。在本章的可下载源代码中，有一些进一步适应我们应用程序的示例，特别是表单提交、访问请求中的表单值和会话。Flask 教程详细介绍了其中许多元素，非常值得一看[`flask.pocoo.org/docs/0.10/tutorial/`](http://flask.pocoo.org/docs/0.10/tutorial/)。

这就是一个非常基本的 Python web 应用程序的样子。显然，有很多种方式可以编写相同的应用程序，就像有很多框架一样，那么你该如何选择一个框架呢？

首先，明确你的应用程序的目标是有帮助的。你是否需要数据库交互？如果是的话，像 Django 这样的更集成的解决方案可能更快开始。你是否需要基于网络的数据输入或管理界面？同样，如果是的话，Django 已经内置了这个功能。

接下来你可以看看你的环境。你的组织中是否已经有了一些首选的包，用于你可能想要执行的操作，比如数据库访问或单元测试？如果有，是否有任何框架已经在使用这些？如果没有，那么微框架可能是一个更好的选择，插入你所需的包。你是否有首选的操作系统或网络服务器用于托管，哪些框架支持这些？你的托管是否在 Python 版本、数据库技术或类似方面限制了你？另外，如果你有网页设计师，你是否有时间让他们熟悉复杂的模板语言，还是必须保持简单？

这些问题的答案可以帮助你缩小选择范围。然后，研究这些框架，询问正在使用它们的人，并尝试一些看起来可能的选择，将会让你达到你需要去的地方。

话虽如此，对于一个需要用户表单提交和数据库访问的一般网络应用程序，Django 是一个不错的选择。它真的是“电池已包含”，它的数据库模型很优雅，它的开箱即用的数据库管理和数据输入界面非常强大，可以节省大量时间。对于像 API 这样的简单应用程序，Flask 也是一个很好的选择，如果需要数据库访问，可以与 SQLAlchemy 一起使用。

正如我之前提到的，没有正确的答案，但通过探索现有的选择，看看框架采用的不同方法，可以学到很多东西。

当然，一旦我们有了我们的网络应用程序，我们需要一种托管它的方式。我们现在将看一些选项。

# 托管 Python 网络应用程序

正如我们在本章开头讨论的那样，为了运行 Python 网络应用程序，我们需要一个网络服务器来托管它。今天存在许多网络服务器，你很可能已经听说过几个。流行的例子有 Apache、nginx（发音为*engine-x*）、lhttpd（发音为*lighty*）和微软的**Internet Information Services**（**IIS**）。

关于网络服务器和它们可以使用的各种机制，有很多术语。我们将简要介绍一下网络应用程序的历史，以帮助解释其中一些概念。

## CGI

在 Web 的早期，网络服务器主要只需要向客户端发送 HTML 页面，或偶尔的图像文件。就像之前的 HTTP 请求旅程图中一样，这些静态资源会存在于服务器的硬盘上，网络服务器的主要任务是接受来自客户端的套接字连接，将请求的 URL 映射到本地文件，并将文件作为 HTTP 响应通过套接字发送回去。

然而，随着对动态内容的需求的增加，网络服务器被赋予了通过调用外部程序和脚本来生成页面的能力，这就是我们今天所说的网络应用程序。网络应用程序最初采用脚本或编译后的可执行文件的形式，它们与常规静态内容一样存在于已发布的 Web 树的磁盘上。网络服务器将被配置，以便当客户端请求这些网络应用程序文件时，网络服务器不仅仅是读取文件并返回它，而是启动一个新的操作系统进程并执行文件，将结果作为请求的 HTML 网页返回。

如果我们更新我们之前图像中的 HTTP 请求的旅程，我们的请求的旅程现在看起来会是这样的：

![CGI](img/6008OS_09_04.jpg)

显然，Web 服务器和 Web 应用程序之间需要一种协议来传递它们之间的 HTTP 请求和返回的 HTML 页面。最早的机制被称为**通用网关接口**（**CGI**）。Web 服务器会将请求分解为环境变量，并在调用处理程序时将其添加到环境中，并通过标准输入将请求的主体（如果有的话）传递给程序。然后，程序会简单地将其生成的 HTTP 响应传输到其标准输出，Web 服务器会捕获并返回给客户端。

然而，由于性能问题，CGI 在今天逐渐不受青睐，如果可能的话，应该避免编写 Python CGI 应用程序。

## 为了更美好的世界而回收利用

CGI 可以工作，但主要缺点是必须为每个请求启动一个新进程。从操作系统资源的角度来看，启动进程是昂贵的，因此这种方法非常低效。已经开发出了替代方案。

两种方法变得常见。第一种是使 Web 服务器在启动时启动和维护多个进程，准备接受新连接——这种技术称为**预分叉**。使用这种技术，仍然存在一对一的进程-客户端关系，但是当新客户端连接时，进程已经创建，从而提高了响应时间。此外，可以重复使用进程，而不是在每次连接时重新创建。

除此之外，Web 服务器被制作成可扩展的，并且创建了与不同语言的绑定，以便 Web 应用程序可以嵌入到 Web 服务器进程中。最常见的例子是 Apache Web 服务器的各种语言模块，用于诸如 PHP 和 Perl 之类的语言。

通过预分叉和 Web 应用程序嵌入，我们的请求的旅程可能如下所示：

![为了更美好的世界而回收利用](img/6008OS_09_05.jpg)

在这里，请求由语言绑定代码转换，我们的 Web 应用程序看到的请求取决于绑定本身的设计。这种管理 Web 应用程序的方法对于一般的 Web 负载效果相当不错，今天仍然是托管 Web 应用程序的一种流行方式。现代浏览器通常也提供多线程变体，其中每个进程可以使用多个线程处理请求，每个客户端连接使用一个线程，进一步提高效率。

解决 CGI 性能问题的第二种方法是将 Web 应用程序进程的管理完全交给一个单独的系统。这个单独的系统会预先分叉并维护运行 Web 应用程序代码的进程池。与 Web 服务器预分叉一样，这些进程可以为每个客户端连接重复使用。开发了新的协议，允许 Web 服务器将请求传递给外部进程，其中最值得注意的是 FastCGI 和 SCGI。在这种情况下，我们的旅程将是：

![为了更美好的世界而回收利用](img/6008OS_09_06.jpg)

同样，请求如何转换并呈现给 Web 应用程序取决于所使用的协议。

尽管在实践中这可能更复杂一些，但它比在预分叉的 Web 服务器进程中嵌入应用程序代码具有优势。主要是，Web 应用程序进程池可以独立于 Web 服务器进程池进行管理，从而更有效地调整两者。

## 事件驱动服务器

然而，Web 客户端数量继续增长，服务器需要能够处理非常大量的同时客户端连接，这些数字使用多进程方法证明是有问题的。这促使了事件驱动 Web 服务器的发展，例如*nginx*和*lighttpd*，它们可以在单个进程中处理许多数千个同时连接。这些服务器还利用预分叉，保持与机器中 CPU 核心数量一致的一些事件驱动进程，从而确保服务器的资源得到充分利用，同时也获得事件驱动架构的好处。

## WSGI

Python Web 应用程序最初是针对这些早期集成协议编写的：CGI，FastCGI 和现在基本上已经废弃的`mod_python` Apache 模块。然而，这证明是麻烦的，因为 Python Web 应用程序与它们编写的协议或服务器绑定在一起。将它们移动到不同的服务器或协议需要对应用程序代码进行一些重新工作。

这个问题通过 PEP 333 得到解决，它定义了**Web 服务网关接口**（**WSGI**）协议。这为 Web 服务器调用 Web 应用程序代码建立了一个类似于 CGI 的通用调用约定。当 Web 服务器和 Web 应用程序都支持 WSGI 时，服务器和应用程序可以轻松交换。WSGI 支持已添加到许多现代 Web 服务器中，现在是在 Web 上托管 Python 应用程序的主要方法。它在 PEP 3333 中更新为 Python 3。

我们之前讨论的许多 Web 框架在幕后支持 WSGI 与其托管 Web 服务器进行通信，包括 Flask 和 Django。这是使用这样的框架的另一个重要好处-您可以免费获得完整的 WSGI 兼容性。

Web 服务器可以使用 WSGI 托管 Web 应用程序的两种方法。首先，它可以直接支持托管 WSGI 应用程序。纯 Python 服务器，如 Gunicorn，遵循这种方法，它们使得提供 Python Web 应用程序非常容易。这正变得越来越受欢迎。

第二种方法是非 Python 服务器使用适配器插件，例如 Apache 的`mod_wsgi`，或者 nginx 的`mod_wsgi`插件。

WSGI 革命的例外是事件驱动服务器。WSGI 不包括允许 Web 应用程序将控制权传递回调用进程的机制，因此对于使用阻塞 IO 风格 WSGI Web 应用程序来说，使用事件驱动服务器没有好处，因为一旦应用程序阻塞，例如，对于数据库访问，它将阻塞整个 Web 服务器进程。

因此，大多数事件驱动框架包括一个生产就绪的 Web 服务器-使 Web 应用程序本身成为事件驱动，并将其嵌入到 Web 服务器进程中是托管它的唯一方法。要使用这些框架托管 Web 应用程序，请查看框架的文档。

# 实际托管

那么这在实践中是如何工作的呢？就像我们在 Flask 中看到的那样，许多框架都配备了自己内置的开发 Web 服务器。然而，这些不建议在生产环境中使用，因为它们通常不是为了在重视安全性和可伸缩性的环境中使用而设计的。

目前，使用 Gunicorn 服务器可能是托管 Python Web 应用程序的生产质量服务器的最快方法。使用我们之前的 Flask 应用程序，我们可以通过几个步骤将其启动和运行。首先我们安装 Gunicorn：

```py
**$ pip install gunicorn**

```

接下来，我们需要稍微修改我们的 Flask 应用程序，以便在 Gunicorn 下正确使用`__builtins__`。在您的`tinyflaskapp.py`文件中，找到以下行：

```py
objs = __builtins__.__dict__.items()
```

将其更改为：

```py
objs = __builtins__.items()
```

现在我们可以运行 Gunicorn。在 Flask 应用程序项目文件夹中，运行以下命令：

```py
**$ gunicorn --bind 0.0.0.0:5000 tinyflaskapp:app**

```

这将启动 Gunicorn Web 服务器，在所有可用接口上监听端口 5000，并为我们的 Flask 应用提供服务。如果我们现在通过`http://127.0.0.1:5000`在 Web 浏览器中访问它，我们应该看到我们的文档索引页面。有关如何使 Gunicorn 成为守护进程的说明，以便它在后台运行，并且随系统自动启动和停止，可以在文档页面上找到，网址为[`gunicorn-docs.readthedocs.org/en/latest/deploy.html#monitoring`](http://gunicorn-docs.readthedocs.org/en/latest/deploy.html#monitoring)。

Gunicorn 使用了之前描述的预分叉进程模型。您可以使用`-w`命令行选项设置进程数（Gunicorn 称它们为工作进程）。文档的“设计”部分包含有关确定要使用的最佳工作进程数量的详细信息，尽管一个好的起点是`(2 x $num_cores) + 1`，其中`$num_cores`是可供 Gunicorn 使用的 CPU 核心数量。

Gunicorn 提供了两种标准的工作类型：同步和异步。同步类型提供严格的每个客户端连接一个工作进程的行为，异步类型使用 eventlet（有关此库的详细信息和安装说明，请参见第八章，“客户端和服务器应用程序”）来提供基于事件的工作进程，可以处理多个连接。只有在使用反向代理时，才建议使用同步类型，因为使用同步类型直接提供给互联网会使您的应用程序容易受到拒绝服务攻击的影响（有关更多详细信息，请参见文档的设计部分）。如果不使用反向代理，则应改用异步类型。工作类型是使用`-k`选项在命令行上设置的。

进一步提高性能并扩展的一种有效方法是使用快速的事件驱动 Web 服务器，例如 nginx，作为我们 Gunicorn 实例前面的**反向代理**。反向代理充当传入 Web 请求的第一行服务器。它直接响应任何它确定是错误的请求，并且还可以配置为提供静态内容以替代我们的 Gunicorn 实例。但是，它还配置为将需要动态内容的任何请求转发到我们的 Gunicorn 实例，以便我们的 Python Web 应用程序可以处理它们。通过这种方式，我们可以获得 nginx 处理大部分 Web 流量的性能优势，而 Gunicorn 和我们的 Web 应用程序可以专注于提供动态页面。

### 注意

有关配置此反向代理配置的详细说明可以在 Gunicorn 页面上找到，网址为[`gunicorn-docs.readthedocs.org/en/latest/deploy.html#nginx-configuration`](http://gunicorn-docs.readthedocs.org/en/latest/deploy.html#nginx-configuration)。

如果您更喜欢使用 Apache，那么另一种有效的托管方法是使用带有`mod_wsgi`模块的 Apache。这需要一些更多的配置，完整的说明可以在以下网址找到：[`code.google.com/p/modwsgi/`](https://code.google.com/p/modwsgi/)。`mod_wsgi`默认在嵌入模式下运行应用程序，其中 Web 应用程序托管在每个 Apache 进程中，这导致了类似于前面的预分叉示例的设置。或者它提供了一个守护程序模式，其中`mod_wsgi`管理一个外部于 Apache 的进程池，类似于之前的 FastCGI 示例。实际上，守护程序模式是出于稳定性和内存性能的考虑而推荐的。有关此配置的说明，请参阅`mod_wsgi`快速配置文档，网址为：[`code.google.com/p/modwsgi/wiki/QuickConfigurationGuide`](https://code.google.com/p/modwsgi/wiki/QuickConfigurationGuide)。

# 总结

我们已经快速浏览了将 Python 应用程序放在 Web 上的过程。我们概述了 Web 应用程序架构及其与 Web 服务器的关系。我们看了看 Python Web 框架的实用性，注意到它们为我们提供了工具和结构，可以更快地编写更好的 Web 应用程序，并帮助我们将我们的应用程序与 Web 服务器集成起来。

我们在 Flask Web 框架中编写了一个小应用程序，看到了它如何帮助我们优雅地管理我们的 URL 空间，以及模板引擎如何帮助我们清晰地管理应用逻辑和 HTML 的分离。我们还强调了一些常见的潜在安全漏洞——XSS 和 CSRF——并介绍了一些基本的缓解技术。

最后，我们讨论了 Web 托管架构以及可以用于将 Python Web 应用程序部署到 Web 的各种方法。特别是，WSGI 是 Web 服务器/ Web 应用程序交互的标准协议，Gunicorn 可用于快速部署，并与 nginx 反向代理一起扩展。Apache 与 mod_wsgi 也是一种有效的托管方法。

在这本书中，我们涵盖了很多内容，还有很多探索工作要做。我们希望这本书让你对可能性有所了解，并且渴望发现更多，希望这只是你在使用 Python 进行网络编程冒险的开始。
