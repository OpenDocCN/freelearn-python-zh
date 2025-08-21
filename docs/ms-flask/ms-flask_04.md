# 第四章：使用蓝图创建控制器

**模型视图控制器**（**MVC**）方程的最后一部分是控制器。我们已经在`main.py`文件中看到了视图函数的基本用法。现在，我们将介绍更复杂和强大的版本，并将我们零散的视图函数转化为统一的整体。我们还将讨论 Flask 如何处理 HTTP 请求的生命周期以及定义 Flask 视图的高级方法。

# 请求设置、拆卸和应用全局

在某些情况下，需要跨所有视图函数访问特定于请求的变量，并且还需要从模板中访问。为了实现这一点，我们可以使用 Flask 的装饰器函数`@app.before_request`和对象`g`。函数`@app.before_request`在每次发出新请求之前执行。Flask 对象`g`是每个特定请求需要保留的任何数据的线程安全存储。在请求结束时，对象被销毁，并在新请求开始时生成一个新对象。例如，以下代码检查 Flask `session`变量是否包含已登录用户的条目；如果存在，它将`User`对象添加到`g`中：

```py
from flask import g, session, abort, render_template

@app.before_request
def before_request():
    if ‘user_id’ in session:
        g.user = User.query.get(session[‘user_id’])

@app.route(‘/restricted’)
def admin():
    if g.user is None:
        abort(403)
    return render_template(‘admin.html’)
```

多个函数可以使用`@app.before_request`进行装饰，并且它们都将在请求的视图函数执行之前执行。还存在一个名为`@app.teardown_request`的装饰器，它在每个请求结束后调用。请记住，这种处理用户登录的方法只是一个示例，不安全。推荐的方法在第六章 *保护您的应用*中有介绍。

# 错误页面

向最终用户显示浏览器的默认错误页面会让用户失去应用的所有上下文，他们必须点击*返回*按钮才能返回到您的站点。要在使用 Flask 的`abort()`函数返回错误时显示自己的模板，可以使用`errorhandler`装饰器函数：

```py
@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404
```

`errorhandler`还可用于将内部服务器错误和 HTTP 500 代码转换为用户友好的错误页面。`app.errorhandler()`函数可以接受一个或多个 HTTP 状态码，以定义它将处理哪个代码。返回元组而不仅仅是 HTML 字符串允许您定义`Response`对象的 HTTP 状态代码。默认情况下，这被设置为`200`。

# 基于类的视图

在大多数 Flask 应用中，视图由函数处理。但是，当许多视图共享公共功能或有代码片段可以拆分为单独的函数时，将视图实现为类以利用继承将非常有用。

例如，如果我们有渲染模板的视图，我们可以创建一个通用的视图类，以保持我们的代码*DRY*：

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

关于此代码的第一件事是我们视图类中的`dispatch_request()`函数。这是我们视图中充当普通视图函数并返回 HTML 字符串的函数。`app.add_url_rule()`函数模仿`app.route()`函数，因为它将路由与函数调用绑定在一起。第一个参数定义了函数的路由，`view_func`参数定义了处理路由的函数。`View.as_view()`方法传递给`view_func`参数，因为它将`View`类转换为视图函数。第一个参数定义了视图函数的名称，因此诸如`url_for()`之类的函数可以路由到它。其余参数传递给`View`类的`__init__`函数。

与普通的视图函数一样，除了`GET`之外的 HTTP 方法必须明确允许`View`类。要允许其他方法，必须添加一个包含命名方法列表的类变量：

```py
class GenericView(View):
    methods = ['GET', 'POST']
    …
    def dispatch_request(self):
        if request.method == ‘GET’:
            return render_template(self.template)
        elif request.method == ‘POST’:
            …
```

## 方法类视图

通常，当函数处理多个 HTTP 方法时，由于大量代码嵌套在`if`语句中，代码可能变得难以阅读：

```py
@app.route('/user', methods=['GET', 'POST', 'PUT', 'DELETE'])
def users():
    if request.method == 'GET':
        …
    elif request.method == 'POST':
        …
    elif request.method == 'PUT':
        …
    elif request.method == 'DELETE':
        …
```

这可以通过`MethodView`类来解决。`MethodView`允许每个方法由不同的类方法处理以分离关注点：

```py
from flask.views import MethodView

class UserView(MethodView):
    def get(self):
        …
    def post(self):
        …
    def put(self):
        …
    def delete(self):
        …

app.add_url_rule(
    '/user',
    view_func=UserView.as_view('user')
)
```

# 蓝图

在 Flask 中，**蓝图**是扩展现有 Flask 应用程序的一种方法。蓝图提供了一种将具有共同功能的视图组合在一起的方式，并允许开发人员将其应用程序分解为不同的组件。在我们的架构中，蓝图将充当我们的*控制器*。

视图被注册到蓝图中；可以为其定义一个单独的模板和静态文件夹，并且当它具有所有所需的内容时，可以在主 Flask 应用程序上注册蓝图内容。蓝图的行为很像 Flask 应用程序对象，但实际上并不是一个独立的应用程序。这就是 Flask 扩展提供视图函数的方式。为了了解蓝图是什么，这里有一个非常简单的例子：

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

蓝图需要两个必需参数——蓝图的名称和包的名称——这些参数在 Flask 内部使用；将`__name__`传递给它就足够了。

其他参数是可选的，并定义蓝图将在哪里查找文件。因为指定了`templates_folder`，蓝图将不会在默认模板文件夹中查找，并且路由将呈现`templates/example/home.html`而不是`templates/home.html`。`url_prefix`选项会自动将提供的 URI 添加到蓝图中的每个路由的开头。因此，主页视图的 URL 实际上是`/example/`。

`url_for()`函数现在必须告知所请求的路由位于哪个蓝图中：

```py
{{ url_for('example.home') }}
```

此外，`url_for()`函数现在必须告知视图是否在同一个蓝图中呈现：

```py
{{ url_for('.home') }}
```

`url_for()`函数还将在指定的静态文件夹中查找静态文件。

要将蓝图添加到我们的应用程序中：

```py
app.register_blueprint(example)
```

让我们将我们当前的应用程序转换为使用蓝图的应用程序。我们首先需要在所有路由之前定义我们的蓝图：

```py
blog_blueprint = Blueprint(
    'blog',
    __name__,
    template_folder='templates/blog',
    url_prefix="/blog"
)
```

现在，因为模板文件夹已经定义，我们需要将所有模板移到模板文件夹的子文件夹中，命名为 blog。接下来，我们所有的路由需要将`@app.route`改为`@blog_blueprint.route`，并且任何类视图分配现在需要注册到`blog_blueprint`。记住，模板中的`url_for()`函数调用也需要更改为在路由前加上一个句点以指示该路由在同一个蓝图中。

在文件末尾，在`if __name__ == '__main__':`语句之前，添加以下内容：

```py
app.register_blueprint(blog_blueprint)
```

现在我们所有的内容都回到了应用程序中，该应用程序在蓝图下注册。因为我们的基本应用程序不再具有任何视图，让我们在基本 URL 上添加一个重定向：

```py
@app.route('/')
def index():
    return redirect(url_for('blog.home'))
```

为什么是 blog 而不是`blog_blueprint`？因为 blog 是蓝图的名称，而名称是 Flask 在内部用于路由的。`blog_blueprint`是 Python 文件中的变量名称。

# 总结

我们现在的应用程序在一个蓝图中运行，但这给了我们什么？假设我们想要在我们的网站上添加一个照片分享功能；我们可以将所有视图函数分组到一个蓝图中，该蓝图具有自己的模板、静态文件夹和 URL 前缀，而不会担心破坏网站其余部分的功能。在下一章中，通过升级我们的文件和代码结构，蓝图将变得更加强大，通过将它们分离成不同的文件。
