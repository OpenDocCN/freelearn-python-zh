# 使用蓝图创建控制器

**模型-视图-控制器**（**MVC**）方程式的最后一部分是控制器。我们已经在 `main.py` 文件中看到了视图函数的基本用法。现在，我们将介绍更复杂和强大的版本，并将我们的不同视图函数转变为统一的整体。我们还将讨论 Flask 处理 HTTP 请求生命周期的内部机制以及定义 Flask 视图的先进方法。

# 会话和全局变量

**会话**是 Flask 在请求之间存储信息的方式；为此，Flask 将使用之前设置的 `SECRET_KEY` 配置来应用 HMAC-SHA1 默认加密方法。因此，用户可以读取他们的会话 cookie，但不能修改它。Flask 还设置了一个默认的会话生命周期，默认为 31 天，以防止中继攻击；这可以通过使用配置键的 `PERMANENT_SESSION_LIFETIME` 配置键来更改。

在当今的现代化网络应用程序中，安全性至关重要；请仔细阅读 Flask 的文档，其中涵盖了各种攻击方法：[`flask.pocoo.org/docs/security/`](http://flask.pocoo.org/docs/security/).

Flask 会话对象是一种特殊的 Python 字典，但您可以使用它就像使用普通的 Python 字典一样，如下所示：

```py
from flask import session
...
session['page_loads'] = session.get('page_loads', 0) + 1
...
```

**全局**是一个线程安全的命名空间存储，用于在请求上下文中保持数据。在每个请求的开始时，创建一个新的全局对象，并在请求结束时销毁该对象。这是放置用户对象或任何需要在视图、模板或请求上下文中调用的 Python 函数之间共享的数据的正确位置。这是无需传递任何数据即可完成的。

使用 `g`（全局）非常简单，为了在请求上下文中设置一个键：

```py
from flask import g
....
# Set some key with some value on a request context
g.some_key = "some_value"
# Get a key
v = g.some_key
# Get and remove a key
v = g.pop('some_key', "default_if_not_present")
```

# 请求设置和清理

当你的 **WSGI**（**Web 服务器网关接口**）处理请求时，Flask 会创建一个包含请求本身所有信息的请求上下文对象。此对象被推入一个包含其他重要信息的堆栈中，例如 Flask 的 `app`、`g`、`session` 和闪存消息。

请求对象对任何正在处理请求的功能、视图或模板都是可用的；这无需传递请求对象本身。`request` 包含诸如 HTTP 头部、URI 参数、URL 路径、WSGI 环境等信息。

有关 Flask 请求对象的更详细信息，请参阅：[`flask.pocoo.org/docs/api/#incoming-request-data`](http://flask.pocoo.org/docs/api/#incoming-request-data).

我们可以通过在请求创建时实现自己的钩子来轻松地向请求上下文添加更多信息。为此，我们可以使用 Flask 的装饰器函数 `@app.before_request` 和 `g` 对象。`@app.before_request` 函数在每次创建新请求之前执行。例如，以下代码为页面加载次数保持一个全局计数器：

```py
import random
from flask import session, g

@app.before_request
def before_request():
    session['page_loads'] = session.get('page_loads', 0) + 1
    g.random_key = random.randrange(1, 10)
```

可以用 `@app.before_request` 装饰器装饰多个函数，它们都会在请求视图函数执行之前执行。还有一个装饰器，`@app.teardown_request`，它在每个请求结束后被调用。

初始化本章提供的示例代码，并观察 `g`、`session` 和 `request` 的数据如何变化。还要注意由 WTForm 设置的 `csrf_token`，以保护我们的表单。

# 错误页面

将浏览器的默认错误页面显示给最终用户会让人感到震惊，因为用户会失去你应用的所有上下文，他们必须点击后退按钮才能返回你的网站。当使用 Flask 的 `abort()` 函数返回错误时，要显示你自己的模板，请使用 `errorhandler` 装饰器函数：

```py
@app.errorhandler(404) 
def page_not_found(error): 
    return render_template('404.html'), 404 
```

`errorhandler` 也可以用来将内部服务器错误和 HTTP 500 状态码转换为用户友好的错误页面。`app.errorhandler()` 函数可以接受一个或多个 HTTP 状态码来定义它将操作哪个代码。通过返回一个元组而不是仅仅一个 HTML 字符串，你可以定义 `Response` 对象的 HTTP 状态码。默认情况下，这设置为 `200`。`recommend` 方法在 第六章，*保护你的应用* 中有介绍。

# 基于类的视图

在大多数 Flask 应用中，视图是通过函数处理的。然而，当许多视图共享共同的功能或者你的代码中有可以拆分成单独函数的部分时，将视图实现为类以利用继承会很有用。

例如，如果我们有渲染模板的视图，我们可以创建一个通用的视图类，以使我们的代码保持 *DRY*：

```py
from flask.views import View 

class GenericView(View): 
    def __init__(self, template): 
        self.template = template 
        super(GenericView, self).__init__() 

    def dispatch_request(self): 
        return render_template(self.template) 

app.add_url_rule( 
    '/', view_func=GenericView.as_view( 
        'home', template='home.html' 
    ) 
)
```

关于这段代码，首先要注意的是我们视图类中的 `dispatch_request()` 函数。这是我们的视图中充当正常视图函数并返回 HTML 字符串的函数。`app.add_url_rule()` 函数模仿了 `app.route()` 函数，因为它将路由绑定到函数调用。第一个参数定义了函数的路由，`view_func` 参数定义了处理路由的函数。`View.as_view()` 方法传递给 `view_func` 参数，因为它将 `View` 类转换成视图函数。第一个参数定义了视图函数的名称，这样 `url_for()` 等函数就可以路由到它。其余参数传递给 `View` 类的 `__init__` 函数。

与正常的视图函数一样，除了 `GET` 方法之外的其他 HTTP 方法必须明确允许 `View` 类。要允许其他方法，必须添加一个包含命名方法列表的类变量：

```py
class GenericView(View): 
    methods = ['GET', 'POST'] 
    ... 
    def dispatch_request(self): 
        if request.method == 'GET': 
            return render_template(self.template) 
        elif request.method == 'POST': 
            ... 
```

这可以是一个非常强大的方法。以渲染来自数据库表的表格列表的网页为例；它们几乎相同，因此是通用方法的良好候选者。尽管执行起来不是一件简单的事情，但实现它所花费的时间可以在未来为你节省时间。使用基于类的视图的初始骨架可能是这样的：

```py
from flask.views import View

class GenericListView(View):

    def __init__(self, model, list_template='generic_list.html'):
        self.model = model
        self.list_template = list_template
        self.columns = self.model.__mapper__.columns.keys()
        # Call super python3 style
        super(GenericListView, self).__init__()

    def render_template(self, context):
        return render_template(self.list_template, **context)

    def get_objects(self):
        return self.model.query.all()

    def dispatch_request(self):
        context = {'objects': self.get_objects(),
                   'columns': self.columns}
        return self.render_template(context)

app.add_url_rule(
    '/generic_posts', view_func=GenericListView.as_view(
        'generic_posts', model=Post)
    )

app.add_url_rule(
    '/generic_users', view_func=GenericListView.as_view(
        'generic_users', model=User)
)

app.add_url_rule(
    '/generic_comments', view_func=GenericListView.as_view(
        'generic_comments', model=Comment)
)
```

有一些有趣的事情需要注意。首先，在类构造函数中，我们使用 SQLAlchemy 模型列初始化`columns`类属性；我们正在利用 SQLAlchemy 的模型自省能力来实现我们的通用模板。因此，列名将被传递到我们的通用模板中，这样我们就可以为任何我们抛给它的模型正确渲染一个格式良好的表格列表。

这是一个使用单个类视图处理所有模型列表视图的简单示例。

这就是模板的样式：

```py
{% extends "base.html" %}
{% block body %}

<div class="table-responsive">
    <table class="table table-bordered table-hover">
    {% for obj in objects %}
        <tr>
        {% for col in columns %}
        <td>
        {{col}} {{ obj[col] }}
        </td>
        {% endfor %}
        </tr>
    {% endfor %}
    </table>
</div>

{% endblock %}
```

你可以通过运行本章提供的示例代码，然后直接访问声明的 URL 来访问这些视图：

+   `http://localhost:5000/generic_users`

+   `http://localhost:5000/generic_posts`

+   `http://localhost:5000/generic_comments`

你可能已经注意到我们的表格视图缺少表列标题。作为一个练习，我挑战你来实现它；你可以简单地渲染提供的`columns`类属性，或者更好的方法是使用标签/列映射来显示更友好的列名。

# 方法类视图

通常，当函数处理多个 HTTP 方法时，由于代码中嵌套在`if`语句中的大段代码，代码可能会变得难以阅读，如下所示：

```py
@app.route('/user', methods=['GET', 'POST', 'PUT', 'DELETE']) 
def users(): 
    if request.method == 'GET': 
        ... 
    elif request.method == 'POST': 
        ... 
    elif request.method == 'PUT': 
        ... 
    elif request.method == 'DELETE': 
        ... 
```

这可以通过`MethodView`类来解决。`MethodView`允许每个方法由不同的类方法处理，以分离关注点：

```py
from flask.views import MethodView 

class UserView(MethodView): 
    def get(self): 
        ... 
    def post(self): 
        ... 
    def put(self): 
        ... 
    def delete(self): 
        ... 

app.add_url_rule( 
    '/user', 
    view_func=UserView.as_view('user') 
) 
```

# 蓝图

在 Flask 中，**蓝图**是扩展现有 Flask 应用的一种方法。它们提供了一种将具有共同功能的一组视图组合起来的方式，并允许开发者将应用分解为不同的组件。在我们的架构中，蓝图将充当我们的*控制器*。

视图被注册到蓝图上；可以为它定义一个单独的模板和静态文件夹，当它包含所有所需的内容时，它可以在主 Flask 应用上注册以添加蓝图的内容。蓝图在功能上类似于 Flask 应用对象，但实际上不是一个自包含的应用。这就是 Flask 扩展提供视图函数的方式。为了了解蓝图是什么，这里有一个非常简单的例子：

```py
from flask import Blueprint 
example = Blueprint( 
    'example', 
    __name__, 
    template_folder='templates/example', 
    static_folder='static/example', 
    url_prefix="/example" 
) 

@example.route('/') 
def home(): 
    return render_template('home.html') 
```

蓝图需要两个必需的参数，蓝图名称和包名称，这些名称在 Flask 内部使用，传递`__name__`给它就足够了。

其他参数是可选的，定义了蓝图将查找文件的位置。因为指定了`templates_folder`，蓝图将不会在默认模板文件夹中查找，并且路由将渲染`templates/example/home.html`而不是`templates/home.html`。`url_prefix`选项自动将提供的 URI 添加到蓝图中的每个路由的开头。所以，主页视图的 URL 实际上是`/example/`。

`url_for()`函数现在必须告诉请求的路由在哪个蓝图：

```py
{{ url_for('example.home') }} 
```

此外，`url_for()`函数现在必须告诉视图是否是从同一蓝图内部渲染的：

```py
{{ url_for('.home') }} 
```

`url_for()`函数还会在指定的`static`文件夹中查找静态文件。

使用以下方法将蓝图添加到我们的应用中：

```py
app.register_blueprint(example) 
```

让我们将当前的应用转换为使用蓝图的应用。我们首先需要定义我们的蓝图，然后再定义所有路由：

```py
blog_blueprint = Blueprint( 
    'blog', 
    __name__, 
    template_folder='templates/blog', 
    url_prefix="/blog" 
) 
```

现在，因为已经定义了`templates`文件夹，我们需要将所有的模板移动到`templates`文件夹下的一个名为`blog`的子文件夹中。接下来，所有我们的路由都需要将`@app.route`更改为`@blog_blueprint.route`，并且任何类视图的分配现在都需要注册到`blog_blueprint`。记住，模板中的`url_for()`函数调用也将需要更改，前面要加上一个点来表示该路由位于同一个蓝图下。

在文件末尾，在`if __name__ == '__main__':`语句之前，添加以下内容：

```py
app.register_blueprint(blog_blueprint)
```

现在，所有内容都回到了我们的应用中，这些内容注册在蓝图下。因为我们的基础应用不再有任何视图，让我们在基础 URL 上添加一个重定向：

```py
@app.route('/') 
def index(): 
    return redirect(url_for('blog.home')) 
```

为什么是`blog`而不是`blog_blueprint`？因为`blog`是蓝图的名字，而名字是 Flask 在内部用于路由的。`blog_blueprint`是 Python 文件中变量的名字。

# 摘要

在本章中，我们向您介绍了一些 Flask 的强大功能；我们看到了如何使用会话在请求之间存储用户数据，以及如何使用全局变量在请求上下文中保持数据。我们向您介绍了请求上下文的概念，并开始向您展示一些新功能，这些功能将使我们能够轻松地将我们的应用程序扩展到任何规模，使用蓝图和方法类视图。

我们现在让我们的应用在蓝图内部运行，但这给我们带来了什么？假设我们想要在我们的网站上添加一个照片分享功能，我们能够将所有的视图函数组合到一个包含自己的模板、静态文件夹和 URL 前缀的蓝图里，而无需担心会破坏网站其他部分的功能。

在下一章中，通过升级我们的文件和代码结构，蓝图将被进一步强化，将它们分离到不同的文件中。
