# 第四章：视图和 URL

在本章中，我们将讨论以下主题：

+   基于类和基于函数的视图

+   混合

+   装饰器

+   常见的视图模式

+   设计 URL

# 从顶部看视图

在 Django 中，视图被定义为一个可调用的函数，它接受一个请求并返回一个响应。它通常是一个带有特殊类方法（如`as_view()`）的函数或类。

在这两种情况下，我们创建一个普通的 Python 函数，它以`HTTPRequest`作为第一个参数，并返回一个`HTTPResponse`。`URLConf`也可以向该函数传递其他参数。这些参数可以从 URL 的部分捕获或设置为默认值。

一个简单的视图如下所示：

```py
# In views.py
from django.http import HttpResponse

def hello_fn(request, name="World"):
    return HttpResponse("Hello {}!".format(name))
```

我们的两行视图函数非常简单易懂。我们目前没有对`request`参数执行任何操作。我们可以检查请求以更好地理解调用视图的上下文，例如通过查看`GET`/`POST`参数、URI 路径或 HTTP 头部（如`REMOTE_ADDR`）。

它在`URLConf`中对应的行如下：

```py
# In urls.py
    url(r'^hello-fn/(?P<name>\w+)/$', views.hello_fn),
    url(r'^hello-fn/$', views.hello_fn),
```

我们正在重用相同的视图函数来支持两个 URL 模式。第一个模式需要一个名称参数。第二个模式不从 URL 中获取任何参数，视图函数将在这种情况下使用`World`的默认名称。

## 视图变得更加优雅

基于类的视图是在 Django 1.4 中引入的。以下是将先前的视图重写为功能等效的基于类的视图的样子：

```py
from django.views.generic import View
class HelloView(View):
    def get(self, request, name="World"):
        return HttpResponse("Hello {}!".format(name))
```

同样，相应的`URLConf`将有两行，如下命令所示：

```py
# In urls.py
    url(r'^hello-cl/(?P<name>\w+)/$', views.HelloView.as_view()),
    url(r'^hello-cl/$', views.HelloView.as_view()),
```

这个`view`类与我们之前的视图函数之间有一些有趣的区别。最明显的区别是我们需要定义一个类。接下来，我们明确地定义我们只处理`GET`请求。之前的视图函数对于`GET`、`POST`或任何其他 HTTP 动词都会给出相同的响应，如下所示，使用 Django shell 中的测试客户端的命令：

```py
>>> from django.test import Client
>>> c = Client()

>>> c.get("http://0.0.0.0:8000/hello-fn/").content
b'Hello World!'

>>> c.post("http://0.0.0.0:8000/hello-fn/").content
b'Hello World!'

>>> c.get("http://0.0.0.0:8000/hello-cl/").content
b'Hello World!'

>>> c.post("http://0.0.0.0:8000/hello-cl/").content
b''

```

从安全性和可维护性的角度来看，明确是好的。

使用类的优势在于需要自定义视图时会变得很明显。比如，您需要更改问候语和默认名称。然后，您可以编写一个通用视图类来适应任何类型的问候，并派生您的特定问候类如下：

```py
class GreetView(View):
    greeting = "Hello {}!"
    default_name = "World"
    def get(self, request, **kwargs):
        name = kwargs.pop("name", self.default_name)
        return HttpResponse(self.greeting.format(name))

class SuperVillainView(GreetView):
    greeting = "We are the future, {}. Not them. "
    default_name = "my friend"
```

现在，`URLConf`将引用派生类：

```py
# In urls.py
    url(r'^hello-su/(?P<name>\w+)/$', views.SuperVillainView.as_view()),
    url(r'^hello-su/$', views.SuperVillainView.as_view()),
```

虽然以类似的方式自定义视图函数并非不可能，但您需要添加几个带有默认值的关键字参数。这可能很快变得难以管理。这正是通用视图从视图函数迁移到基于类的视图的原因。

### 注意

**Django Unchained**

在寻找优秀的 Django 开发人员花了 2 周后，史蒂夫开始打破常规。注意到最近黑客马拉松的巨大成功，他和哈特在 S.H.I.M 组织了一个 Django Unchained 比赛。规则很简单——每天构建一个 Web 应用程序。它可以很简单，但你不能跳过一天或打破链条。谁创建了最长的链条，谁就赢了。

获胜者——布拉德·扎尼真是个惊喜。作为一个传统的设计师，几乎没有任何编程背景，他曾经参加了为期一周的 Django 培训，只是为了好玩。他设法创建了一个由 21 个 Django 站点组成的不间断链条，大部分是从零开始。

第二天，史蒂夫在他的办公室安排了一个 10 点的会议。虽然布拉德不知道，但这将是他的招聘面试。在预定的时间，有轻轻的敲门声，一个二十多岁的瘦削有胡须的男人走了进来。

当他们交谈时，布拉德毫不掩饰他不是程序员这一事实。事实上，他根本不需要假装。透过他那副厚框眼镜，透过他那宁静的蓝色眼睛，他解释说他的秘诀非常简单——获得灵感，然后专注。

他过去每天都以一个简单的线框开始。然后，他会使用 Twitter bootstrap 模板创建一个空的 Django 项目。他发现 Django 的基于类的通用视图是以几乎没有代码创建视图的绝佳方式。有时，他会从 Django-braces 中使用一个或两个 mixin。他还喜欢通过管理界面在移动中添加数据。

他最喜欢的项目是 Labyrinth——一个伪装成棒球论坛的蜜罐。他甚至设法诱捕了一些搜寻易受攻击站点的监视机器人。当史蒂夫解释了 SuperBook 项目时，他非常乐意接受这个提议。创建一个星际社交网络的想法真的让他着迷。

通过更多的挖掘，史蒂夫能够在 S.H.I.M 中找到半打像布拉德这样有趣的个人资料。他得知，他应该首先在组织内部搜索，而不是寻找外部。

# 基于类的通用视图

基于类的通用视图通常以面向对象的方式实现（模板方法模式）以实现更好的重用。我讨厌术语*通用视图*。我宁愿称它们为*库存视图*。就像库存照片一样，您可以在稍微调整的情况下用于许多常见需求。

通用视图是因为 Django 开发人员觉得他们在每个项目中都在重新创建相同类型的视图。几乎每个项目都需要显示对象列表（`ListView`），对象的详细信息（`DetailView`）或用于创建对象的表单（`CreateView`）的页面。为了遵循 DRY 原则，这些可重用的视图与 Django 捆绑在一起。

Django 1.7 中通用视图的方便表格如下：

| 类型 | 类名 | 描述 |
| --- | --- | --- |
| 基类 | `View` | 这是所有视图的父类。它执行分发和健全性检查。 |
| 基类 | `TemplateView` | 这呈现模板。它将`URLConf`关键字暴露到上下文中。 |
| 基类 | `RedirectView` | 这在任何`GET`请求上重定向。 |
| 列表 | `ListView` | 这呈现任何可迭代的项目，例如`queryset`。 |
| 详细 | `DetailView` | 这根据`URLConf`中的`pk`或`slug`呈现项目。 |
| 编辑 | `FormView` | 这呈现并处理表单。 |
| 编辑 | `CreateView` | 这呈现并处理用于创建新对象的表单。 |
| 编辑 | `UpdateView` | 这呈现并处理用于更新对象的表单。 |
| 编辑 | `DeleteView` | 这呈现并处理用于删除对象的表单。 |
| 日期 | `ArchiveIndexView` | 这呈现具有日期字段的对象列表，最新的对象排在第一位。 |
| 日期 | `YearArchiveView` | 这在`URLConf`中给出的`year`上呈现对象列表。 |
| 日期 | `MonthArchiveView` | 这在`year`和`month`上呈现对象列表。 |
| 日期 | `WeekArchiveView` | 这在`year`和`week`号上呈现对象列表。 |
| 日期 | `DayArchiveView` | 这在`year`，`month`和`day`上呈现对象列表。 |
| 日期 | `TodayArchiveView` | 这在今天的日期上呈现对象列表。 |
| 日期 | `DateDetailView` | 这根据其`pk`或`slug`在`year`，`month`和`day`上呈现对象。 |

我们没有提到诸如`BaseDetailView`之类的基类或`SingleObjectMixin`之类的混合类。它们被设计为父类。在大多数情况下，您不会直接使用它们。

大多数人混淆了基于类的视图和基于类的通用视图。它们的名称相似，但它们并不是相同的东西。这导致了一些有趣的误解，如下所示：

+   **Django 捆绑的唯一通用视图**：幸运的是，这是错误的。提供的基于类的通用视图中没有特殊的魔法。

您可以自由地编写自己的通用基于类的视图集。您还可以使用第三方库，比如`django-vanilla-views`（[`django-vanilla-views.org/`](http://django-vanilla-views.org/)），它具有标准通用视图的更简单的实现。请记住，使用自定义通用视图可能会使您的代码对他人来说变得陌生。

+   **基于类的视图必须始终派生自通用视图**：同样，通用视图类并没有什么神奇之处。虽然 90%的时间，您会发现像`View`这样的通用类非常适合用作基类，但您可以自由地自己实现类似的功能。

# 视图混入

混入是类基视图中 DRY 代码的本质。与模型混入一样，视图混入利用 Python 的多重继承来轻松重用功能块。它们通常是 Python 3 中没有父类的类（或者在 Python 2 中从`object`派生，因为它们是新式类）。

混入在明确定义的位置拦截视图的处理。例如，大多数通用视图使用`get_context_data`来设置上下文字典。这是插入额外上下文的好地方，比如一个`feed`变量，指向用户可以查看的所有帖子，如下命令所示：

```py
class FeedMixin(object):
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["feed"] = models.Post.objects.viewable_posts(self.request.user)
        return context
```

`get_context_data`方法首先通过调用所有基类中的同名方法来填充上下文。接下来，它使用`feed`变量更新上下文字典。

现在，可以很容易地使用这个混入来通过将其包含在基类列表中来添加用户的 feed。比如，如果 SuperBook 需要一个典型的社交网络主页，其中包括一个创建新帖子的表单，然后是您的 feed，那么可以使用这个混入如下：

```py
class MyFeed(FeedMixin, generic.CreateView):
    model = models.Post
    template_name = "myfeed.html"
    success_url = reverse_lazy("my_feed")
```

一个写得很好的混入几乎没有要求。它应该灵活，以便在大多数情况下都能派上用场。在前面的例子中，`FeedMixin`将覆盖派生类中的`feed`上下文变量。如果父类使用`feed`作为上下文变量，那么它可能会受到包含此混入的影响。因此，使上下文变量可定制会更有用（这留给您作为练习）。

混入能够与其他类结合是它们最大的优势和劣势。使用错误的组合可能导致奇怪的结果。因此，在使用混入之前，您需要检查混入和其他类的源代码，以确保没有方法或上下文变量冲突。

## 混入的顺序

您可能已经遇到了包含几个混入的代码，如下所示：

```py
class ComplexView(MyMixin, YourMixin, AccessMixin, DetailView):
```

确定列出基类的顺序可能会变得非常棘手。就像 Django 中的大多数事情一样，通常适用 Python 的正常规则。Python 的**方法解析顺序**（**MRO**）决定了它们应该如何排列。

简而言之，混入首先出现，基类最后出现。父类越专业，它就越向左移动。在实践中，这是您需要记住的唯一规则。

要理解为什么这样做，请考虑以下简单的例子：

```py
class A:
    def do(self):
        print("A")

class B:
    def do(self):
        print("B")

class BA(B, A):
    pass

class AB(A, B):
    pass

BA().do() # Prints B
AB().do() # Prints A
```

正如您所期望的，如果在基类列表中提到`B`在`A`之前，那么将调用`B`的方法，反之亦然。

现在想象`A`是一个基类，比如`CreateView`，`B`是一个混入，比如`FeedMixin`。混入是对基类基本功能的增强。因此，混入代码应该首先执行，然后根据需要调用基本方法。因此，正确的顺序是`BA`（混入在前，基类在后）。

调用基类的顺序可以通过检查类的`__mro__`属性来确定：

```py
>>> AB.__mro__
 (__main__.AB, __main__.A, __main__.B, object)

```

因此，如果`AB`调用`super()`，首先会调用`A`；然后，`A`的`super()`将调用`B`，依此类推。

### 提示

Python 的 MRO 通常遵循深度优先，从左到右的顺序来选择类层次结构中的方法。更多细节可以在[`www.python.org/download/releases/2.3/mro/`](http://www.python.org/download/releases/2.3/mro/)找到。

# 装饰器

在类视图之前，装饰器是改变基于函数的视图行为的唯一方法。作为函数的包装器，它们不能改变视图的内部工作，因此有效地将它们视为黑匣子。

装饰器是一个接受函数并返回装饰函数的函数。感到困惑？有一些语法糖可以帮助你。使用注解符号`@`，如下面的`login_required`装饰器示例所示：

```py
@login_required
def simple_view(request):
    return HttpResponse()
```

以下代码与上面完全相同：

```py
def simple_view(request):
    return HttpResponse()

simple_view = login_required(simple_view)
```

由于`login_required`包装了视图，所以包装函数首先获得控制权。如果用户未登录，则重定向到`settings.LOGIN_URL`。否则，它执行`simple_view`，就好像它不存在一样。

装饰器不如 mixin 灵活。但它们更简单。在 Django 中，您可以同时使用装饰器和 mixin。实际上，许多 mixin 都是用装饰器实现的。

# 视图模式

让我们看一些在设计视图中看到的常见设计模式。

## 模式 - 受控访问视图

**问题**：页面需要根据用户是否已登录、是否为工作人员或任何其他条件有条件地访问。

**解决方案**：使用 mixin 或装饰器来控制对视图的访问。

### 问题详情

大多数网站有一些只有在登录后才能访问的页面。其他一些页面对匿名或公共访问者开放。如果匿名访问者尝试访问需要登录用户的页面，则可能会被路由到登录页面。理想情况下，登录后，他们应该被路由回到他们最初希望看到的页面。

同样，有些页面只能由某些用户组看到。例如，Django 的管理界面只对工作人员可访问。如果非工作人员用户尝试访问管理页面，他们将被路由到登录页面。

最后，有些页面只有在满足某些条件时才能访问。例如，只有帖子的创建者才能编辑帖子。其他任何人访问此页面都应该看到**权限被拒绝**的错误。

### 解决方案详情

有两种方法可以控制对视图的访问：

1.  通过在基于函数的视图或基于类的视图上使用装饰器：

```py
@login_required(MyView.as_view())
```

1.  通过 mixin 重写类视图的`dispatch`方法：

```py
from django.utils.decorators import method_decorator

class LoginRequiredMixin:
    @method_decorator(login_required)
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)
```

我们这里真的不需要装饰器。推荐更明确的形式如下：

```py
class LoginRequiredMixin:

    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated():
            raise PermissionDenied
        return super().dispatch(request, *args, **kwargs)
```

当引发`PermissionDenied`异常时，Django 会在您的根目录中显示`403.html`模板，或者在其缺失时显示标准的“403 Forbidden”页面。

当然，对于真实项目，您需要一个更健壮和可定制的 mixin 集。`django-braces`包（[`github.com/brack3t/django-braces`](https://github.com/brack3t/django-braces)）有一套出色的 mixin，特别是用于控制对视图的访问。

以下是使用它们来控制登录和匿名视图的示例：

```py
from braces.views import LoginRequiredMixin, AnonymousRequiredMixin

class UserProfileView(LoginRequiredMixin, DetailView):
    # This view will be seen only if you are logged-in
    pass  

class LoginFormView(AnonymousRequiredMixin, FormView):
    # This view will NOT be seen if you are loggedin
    authenticated_redirect_url = "/feed"
```

Django 中的工作人员是在用户模型中设置了`is_staff`标志的用户。同样，您可以使用一个名为`UserPassesTestMixin`的 django-braces mixin，如下所示：

```py
from braces.views import UserPassesTestMixin

class SomeStaffView(UserPassesTestMixin, TemplateView):
    def test_func(self, user):
        return user.is_staff
```

您还可以创建 mixin 来执行特定的检查，比如对象是否正在被其作者编辑（通过与登录用户进行比较）：

```py
class CheckOwnerMixin:

    # To be used with classes derived from SingleObjectMixin
    def get_object(self, queryset=None):
        obj = super().get_object(queryset)
        if not obj.owner == self.request.user:
            raise PermissionDenied
        return obj
```

## 模式 - 上下文增强器

**问题**：基于通用视图的几个视图需要相同的上下文变量。

**解决方案**：创建一个设置共享上下文变量的 mixin。

### 问题详情

Django 模板只能显示存在于其上下文字典中的变量。然而，站点需要在多个页面中具有相同的信息。例如，侧边栏显示您的动态中最近的帖子可能需要在多个视图中使用。

然而，如果我们使用通用的基于类的视图，通常会有一组与特定模型相关的有限上下文变量。在每个视图中设置相同的上下文变量并不符合 DRY 原则。

### 解决方案详情

大多数通用的基于类的视图都是从`ContextMixin`派生的。它提供了`get_context_data`方法，大多数类都会重写这个方法，以添加他们自己的上下文变量。在重写这个方法时，作为最佳实践，您需要首先调用超类的`get_context_data`，然后添加或覆盖您的上下文变量。

我们可以将这个抽象成一个 mixin 的形式，就像我们之前看到的那样：

```py
class FeedMixin(object):

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["feed"] = models.Post.objects.viewable_posts(self.request.user)
        return context
```

我们可以将这个 mixin 添加到我们的视图中，并在我们的模板中使用添加的上下文变量。请注意，我们正在使用第三章中定义的模型管理器，*模型*，来过滤帖子。

一个更一般的解决方案是使用`django-braces`中的`StaticContextMixin`来处理静态上下文变量。例如，我们可以添加一个额外的上下文变量`latest_profile`，其中包含最新加入站点的用户：

```py
class CtxView(StaticContextMixin, generic.TemplateView):
    template_name = "ctx.html"
    static_context = {"latest_profile": Profile.objects.latest('pk')}
```

在这里，静态上下文意味着任何从一个请求到另一个请求都没有改变的东西。在这种意义上，您也可以提到`QuerySets`。然而，我们的`feed`上下文变量需要`self.request.user`来检索用户可查看的帖子。因此，在这里不能将其包括为静态上下文。

## 模式 - 服务

**问题**：您网站的信息经常被其他应用程序抓取和处理。

**解决方案**：创建轻量级服务，以机器友好的格式返回数据，如 JSON 或 XML。

### 问题细节

我们经常忘记网站不仅仅是人类使用的。网站流量的很大一部分来自其他程序，如爬虫、机器人或抓取器。有时，您需要自己编写这样的程序来从另一个网站提取信息。

通常，为人类消费而设计的页面对机械提取来说很麻烦。HTML 页面中的信息被标记包围，需要进行大量的清理。有时，信息会分散，需要进行大量的数据整理和转换。

在这种情况下，机器接口将是理想的。您不仅可以减少提取信息的麻烦，还可以实现混搭。如果应用程序的功能以机器友好的方式暴露，其功能的持久性将大大增加。

### 解决方案细节

**面向服务的架构**（**SOA**）已经推广了服务的概念。服务是向其他应用程序公开的一个独特的功能块。例如，Twitter 提供了一个返回最新公共状态的服务。

一个服务必须遵循一定的基本原则：

+   **无状态性**：这避免了通过外部化状态信息来避免内部状态

+   **松耦合**：这样可以减少依赖和假设的最小数量

+   **可组合的**：这应该很容易重用并与其他服务组合

在 Django 中，您可以创建一个基本的服务，而无需任何第三方包。您可以返回 JSON 格式的序列化数据，而不是返回 HTML。这种形式的服务通常被称为 Web 应用程序编程接口（API）。

例如，我们可以创建一个简单的服务，返回 SuperBook 中最近的五篇公共帖子：

```py
class PublicPostJSONView(generic.View):

    def get(self, request, *args, **kwargs):
        msgs = models.Post.objects.public_posts().values(
            "posted_by_id", "message")[:5]
        return HttpResponse(list(msgs), content_type="application/json")
```

为了更可重用的实现，您可以使用`django-braces`中的`JSONResponseMixin`类，使用其`render_json_response`方法返回 JSON：

```py
from braces.views import JSONResponseMixin

class PublicPostJSONView(JSONResponseMixin, generic.View):

    def get(self, request, *args, **kwargs):
        msgs = models.Post.objects.public_posts().values(
            "posted_by_id", "message")[:5]
        return self.render_json_response(list(msgs))
```

如果我们尝试检索这个视图，我们将得到一个 JSON 字符串，而不是 HTML 响应：

```py
>>> from django.test import Client
>>> Client().get("http://0.0.0.0:8000/public/").content
b'{"posted_by_id": 23, "message": "Hello!"},
 {"posted_by_id": 13, "message": "Feeling happy"},
 ...

```

请注意，我们不能直接将`QuerySet`方法传递给 JSON 响应。它必须是一个列表、字典或任何其他基本的 Python 内置数据类型，被 JSON 序列化器识别。

当然，如果您需要构建比这个简单 API 更复杂的东西，您将需要使用诸如 Django REST 框架之类的包。 Django REST 框架负责序列化（和反序列化）`QuerySets`，身份验证，生成可在 Web 上浏览的 API，以及许多其他必要功能，以创建一个强大而完整的 API。

# 设计 URL

Django 拥有最灵活的 Web 框架之一。基本上，没有暗示的 URL 方案。您可以使用适当的正则表达式明确定义任何 URL 方案。

然而，正如超级英雄们喜欢说的那样——“伴随着伟大的力量而来的是巨大的责任。”您不能再随意设计 URL。

URL 曾经很丑陋，因为人们认为用户会忽略它们。在 90 年代，门户网站流行时，普遍的假设是您的用户将通过前门，也就是主页进入。他们将通过点击链接导航到网站的其他页面。

搜索引擎已经改变了这一切。根据 2013 年的一份研究报告，近一半（47％）的访问来源于搜索引擎。这意味着您网站中的任何页面，根据搜索相关性和受欢迎程度，都可能成为用户看到的第一个页面。任何 URL 都可能是前门。

更重要的是，浏览 101 教会了我们安全。我们警告初学者不要在网上点击蓝色链接。先读 URL。这真的是您银行的 URL 还是一个试图钓取您登录详细信息的网站？

如今，URL 已经成为用户界面的一部分。它们被看到，复制，分享，甚至编辑。让它们看起来好看且一目了然。不再有眼睛的疼痛，比如：

`http://example.com/gallery/default.asp?sid=9DF4BC0280DF12D3ACB60090271E26A8&command=commntform`

短而有意义的 URL 不仅受到用户的欣赏，也受到搜索引擎的欢迎。长且与内容相关性较低的 URL 会对您的网站搜索引擎排名产生不利影响。

最后，正如“酷炫的 URI 不会改变”所暗示的，您应该尽量保持 URL 结构随时间的稳定。即使您的网站完全重新设计，您的旧链接仍应该有效。Django 可以轻松确保如此。

在我们深入了解设计 URL 的细节之前，我们需要了解 URL 的结构。

## URL 解剖

从技术上讲，URL 属于更一般的标识符家族，称为**统一资源标识符（URI**）。因此，URL 的结构与 URI 相同。

URI 由几个部分组成：

*URI = 方案 + 网络位置 + 路径 + 查询 + 片段*

例如，可以使用`urlparse`函数在 Python 中解构 URI（`http://dev.example.com:80/gallery/videos?id=217#comments`）：

```py
>>> from urllib.parse import urlparse
>>> urlparse("http://dev.example.com:80/gallery/videos?id=217#comments")
ParseResult(scheme='http', netloc='dev.example.com:80', path='/gallery/videos', params='', query='id=217', fragment='comments')

```

URI 的各个部分可以以图形方式表示如下：

![URL 解剖

尽管 Django 文档更喜欢使用术语 URLs，但更准确地说，您大部分时间都在使用 URI。在本书中，我们将这些术语互换使用。

Django URL 模式主要涉及 URI 的“路径”部分。所有其他部分都被隐藏起来。

### urls.py 中发生了什么？

通常有助于将`urls.py`视为项目的入口点。当我研究 Django 项目时，这通常是我打开的第一个文件。基本上，`urls.py`包含整个项目的根 URL 配置或`URLConf`。

它将是从`patterns`返回的 Python 列表，分配给名为`urlpatterns`的全局变量。每个传入的 URL 都会与顺序中的每个模式进行匹配。在第一次匹配时，搜索停止，并且请求被发送到相应的视图。

这里，是从[Python.org](http://Python.org)的`urls.py`中的一个摘录，最近在 Django 中重新编写：

```py
urlpatterns = patterns(
    '',
    # Homepage
    url(r'^$', views.IndexView.as_view(), name='home'),
    # About
    url(r'^about/$',
        TemplateView.as_view(template_name="python/about.html"),
        name='about'),
    # Blog URLs
    url(r'^blogs/', include('blogs.urls', namespace='blog')),
    # Job archive
    url(r'^jobs/(?P<pk>\d+)/$',
        views.JobArchive.as_view(),
        name='job_archive'),
    # Admin
    url(r'^admin/', include(admin.site.urls)),
)
```

这里需要注意的一些有趣的事情如下：

+   `patterns`函数的第一个参数是前缀。对于根`URLConf`，通常为空。其余参数都是 URL 模式。

+   每个 URL 模式都是使用`url`函数创建的，该函数需要五个参数。大多数模式有三个参数：正则表达式模式，视图可调用和视图的名称。

+   `about`模式通过直接实例化`TemplateView`来定义视图。有些人讨厌这种风格，因为它提到了实现，从而违反了关注点的分离。

+   博客 URL 在其他地方提到，特别是在 blogs 应用程序的`urls.py`中。一般来说，将应用程序的 URL 模式分离成自己的文件是一个很好的做法。

+   `jobs`模式是这里唯一的一个命名正则表达式的例子。

在未来的 Django 版本中，`urlpatterns`应该是一个 URL 模式对象的普通列表，而不是`patterns`函数的参数。这对于有很多模式的站点来说很棒，因为`urlpatterns`作为一个函数只能接受最多 255 个参数。

如果你是 Python 正则表达式的新手，你可能会觉得模式语法有点神秘。让我们试着揭开它的神秘面纱。

### URL 模式语法

URL 正则表达式模式有时看起来像一团令人困惑的标点符号。然而，像 Django 中的大多数东西一样，它只是普通的 Python。

通过了解 URL 模式的两个功能，可以很容易地理解它：匹配以某种形式出现的 URL，并从 URL 中提取有趣的部分。

第一部分很容易。如果你需要匹配一个路径，比如`/jobs/1234`，那么只需使用"`^jobs/\d+`"模式（这里`\d`代表从 0 到 9 的单个数字）。忽略前导斜杠，因为它会被吞掉。

第二部分很有趣，因为在我们的例子中，有两种提取作业 ID（即`1234`）的方法，这是视图所需的。

最简单的方法是在要捕获的每组值周围放括号。每个值将作为位置参数传递给视图。例如，"`^jobs/(\d+)`"模式将把值"`1234`"作为第二个参数（第一个是请求）发送给视图。

位置参数的问题在于很容易混淆顺序。因此，我们有基于名称的参数，其中每个捕获的值都可以被命名。我们的例子现在看起来像"`^jobs/(?P<pk>\d+)/`"。这意味着视图将被调用，关键字参数`pk`等于"`1234`"。

如果你有一个基于类的视图，你可以在`self.args`中访问你的位置参数，在`self.kwargs`中访问基于名称的参数。许多通用视图期望它们的参数仅作为基于名称的参数，例如`self.kwargs["slug"]`。

#### 记忆法-父母质疑粉色动作人物

我承认基于名称的参数的语法很难记住。我经常使用一个简单的记忆法作为记忆助手。短语“Parents Question Pink Action-figures”代表括号、问号、（字母）P 和尖括号的首字母。

把它们放在一起，你会得到`(?P<`。你可以输入模式的名称，然后自己找出剩下的部分。

这是一个很方便的技巧，而且很容易记住。想象一下一个愤怒的父母拿着一个粉色的浩克动作人物。

另一个提示是使用在线正则表达式生成器，比如[`pythex.org/`](http://pythex.org/)或[`www.debuggex.com/`](https://www.debuggex.com/)来制作和测试你的正则表达式。

### 名称和命名空间

总是给你的模式命名。这有助于将你的代码与确切的 URL 路径解耦。例如，在以前的`URLConf`中，如果你想重定向到`about`页面，可能会诱人地使用`redirect("/about")`。相反，使用`redirect("about")`，因为它使用名称而不是路径。

以下是一些反向查找的更多示例：

```py
>>> from django.core.urlresolvers import reverse
>>> print(reverse("home"))
"/"
>>> print(reverse("job_archive", kwargs={"pk":"1234"}))
"jobs/1234/"

```

名称必须是唯一的。如果两个模式有相同的名称，它们将无法工作。因此，一些 Django 包用于向模式名称添加前缀。例如，一个名为 blog 的应用程序可能必须将其编辑视图称为'`blog-edit`'，因为'`edit`'是一个常见的名称，可能会与另一个应用程序发生冲突。

命名空间是为了解决这类问题而创建的。在命名空间中使用的模式名称必须在该命名空间内是唯一的，而不是整个项目。建议您为每个应用程序都分配一个命名空间。例如，我们可以通过在根`URLconf`中包含此行来创建一个“blog”命名空间，其中只包括博客的 URL：

```py
url(r'^blog/', include('blog.urls', namespace='blog')),
```

现在博客应用程序可以使用模式名称，比如“`edit`”或其他任何名称，只要它们在该应用程序内是唯一的。在引用命名空间内的名称时，您需要在名称之前提到命名空间，然后是“`:`”。在我们的例子中，它将是“`blog:edit`”。

正如 Python 之禅所说 - “命名空间是一个非常棒的想法 - 让我们做更多这样的事情。”如果这样做可以使您的模式名称更清晰，您可以创建嵌套的命名空间，比如“`blog:comment:edit`”。我强烈建议您在项目中使用命名空间。

### 模式顺序

按照 Django 处理它们的方式，即自上而下，对您的模式进行排序以利用它们。一个很好的经验法则是将所有特殊情况放在顶部。更广泛的模式可以在更下面提到。最广泛的 - 如果存在的话，可以放在最后。

例如，您的博客文章的路径可以是任何有效的字符集，但您可能希望单独处理关于页面。正确的模式顺序应该如下：

```py
urlpatterns = patterns(
    '',
    url(r'^about/$', AboutView.as_view(), name='about'),
    url(r'^(?P<slug>\w+)/$', ArticleView.as_view(), name='article'),
)  
```

如果我们颠倒顺序，那么特殊情况`AboutView`将永远不会被调用。

### URL 模式样式

一致地设计网站的 URL 很容易被忽视。设计良好的 URL 不仅可以合理地组织您的网站，还可以让用户猜测路径变得容易。设计不良的 URL 甚至可能构成安全风险：比如，在 URL 模式中使用数据库 ID（它以单调递增的整数序列出现）可能会增加信息窃取或网站剥离的风险。

让我们来看一些在设计 URL 时遵循的常见样式。

#### 百货商店 URL

有些网站的布局就像百货商店。有一个食品区，里面有一个水果通道，通道里有不同种类的苹果摆在一起。

在 URL 的情况下，这意味着您将按以下层次结构找到这些页面：

```py
http://site.com/ <section> / <sub-section> / <item>
```

这种布局的美妙之处在于很容易向上爬到父级部分。一旦删除斜杠后面的部分，您就会上升一个级别。

例如，您可以为文章部分创建一个类似的结构，如下所示：

```py
# project's main urls.py
urlpatterns = patterns(
    '',
    url(r'^articles/$', include(articles.urls), namespace="articles"),
)

# articles/urls.py
urlpatterns = patterns(
    '',
    url(r'^$', ArticlesIndex.as_view(), name='index'),
    url(r'^(?P<slug>\w+)/$', ArticleView.as_view(), name='article'),
)
```

注意“`index`”模式，它将在用户从特定文章上升时显示文章索引。

#### RESTful URL

2000 年，Roy Fielding 在他的博士论文中引入了**表现状态转移**（**REST**）这个术语。强烈建议阅读他的论文（[`www.ics.uci.edu/~fielding/pubs/dissertation/top.htm`](http://www.ics.uci.edu/~fielding/pubs/dissertation/top.htm)）以更好地理解 Web 本身的架构。它可以帮助你编写更好的 Web 应用程序，不违反架构的核心约束。

其中一个关键的见解是 URI 是资源的标识符。资源可以是任何东西，比如一篇文章，一个用户，或者一组资源，比如事件。一般来说，资源是名词。

Web 为您提供了一些基本的 HTTP 动词来操作资源：`GET`，`POST`，`PUT`，`PATCH`和`DELETE`。请注意，这些不是 URL 本身的一部分。因此，如果您在 URL 中使用动词来操作资源，这是一个不好的做法。

例如，以下 URL 被认为是不好的：

`http://site.com/articles/submit/`

相反，你应该删除动词，并使用 POST 操作到这个 URL：

`http://site.com/articles/`

### 提示

**最佳实践**

如果 HTTP 动词可以使用，就不要在 URL 中使用动词。

请注意，在 URL 中使用动词并不是错误的。您网站的搜索 URL 可以使用动词“`search`”，因为它不符合 REST 的一个资源：

`http://site.com/search/?q=needle`

RESTful URL 对于设计 CRUD 接口非常有用。创建、读取、更新和删除数据库操作与 HTTP 动词之间几乎是一对一的映射。

请注意，RESTful URL 风格是部门商店 URL 风格的补充。大多数网站混合使用这两种风格。它们被分开以便更清晰地理解。

### 提示

**下载示例代码**

您可以从[`www.packtpub.com`](http://www.packtpub.com)的帐户中下载您购买的所有 Packt 图书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，将文件直接发送到您的电子邮件。拉取请求和错误报告可以发送到[`github.com/DjangoPatternsBook/superbook`](https://github.com/DjangoPatternsBook/superbook)的 SuperBook 项目。

# 总结

在 Django 中，视图是 MVC 架构中非常强大的部分。随着时间的推移，基于类的视图已被证明比传统的基于函数的视图更灵活和可重用。混合是这种可重用性的最好例子。

Django 拥有非常灵活的 URL 分发系统。设计良好的 URL 需要考虑几个方面。设计良好的 URL 也受到用户的赞赏。

在下一章中，我们将看一下 Django 的模板语言以及如何最好地利用它。
