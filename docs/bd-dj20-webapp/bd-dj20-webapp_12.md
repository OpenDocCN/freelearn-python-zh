# 第十二章：构建 API

现在 Mail Ape 可以向我们的订阅者发送电子邮件了，让我们让用户更容易地使用 API 与 Mail Ape 集成。在本章中，我们将构建一个 RESTful JSON API，让用户可以创建邮件列表并将订阅者添加到邮件列表中。为了简化创建我们的 API，我们将使用 Django REST 框架（DRF）。最后，我们将使用 curl 在命令行上访问我们的 API。

在本章中，我们将做以下事情：

+   总结 DRF 的核心概念

+   创建`Serializer`，定义如何解析和序列化`MailingList`和`Subscriber`模型

+   创建权限类以限制 API 对`MailingList`所有者的用户

+   使用 Django REST 框架的基于类的视图来创建我们 API 的视图

+   使用 curl 通过 HTTP 访问我们的 API

+   在单元测试中测试我们的 API

让我们从 DRF 开始这一章。

# 从 Django REST 框架开始

我们将首先安装 DRF，然后审查其配置。在审查 DRF 配置时，我们将了解使其有用的功能和概念。

# 安装 Django REST 框架

让我们首先将 DRF 添加到我们的`requirements.txt`文件中：

```py
djangorestframework<3.8
```

接下来，我们可以使用`pip`进行安装：

```py
$ pip install -r requirements.txt
```

现在我们已经安装了库，让我们在`django/mailinglist/settings.py`文件中的`INSTALLED_APPS`列表中添加 DRF：

```py
INSTALLED_APPS = [
# previously unchanged list
    'rest_framework',
]
```

# 配置 Django REST 框架

DRF 通过其视图类高度可配置。但是，我们可以使用`settings.py`文件中的 DRF 设置来避免在所有 DRF 视图中重复相同的常见设置。

DRF 的所有功能都源自 DRF 处理视图的方式。DRF 提供了丰富的视图集合，扩展了`APIView`（它又扩展了 Django 的`View`类）。让我们看看 APIView 的生命周期和相关设置。

DRF 视图的生命周期执行以下操作：

1.  **在 DRF 请求对象中包装 Django 的请求对象**：DRF 有一个专门的`Request`类，它包装了 Django 的`Request`类，将在下面的部分中讨论。

1.  **执行内容协商**：查找请求解析器和响应渲染器。

1.  **执行身份验证**：检查与请求相关联的凭据。

1.  **检查权限**：检查与请求相关联的用户是否可以访问此视图。

1.  **检查节流**：检查最近是否有太多请求由此用户发出。

1.  **执行视图处理程序**：执行与视图相关的操作（例如创建资源、查询数据库等）。

1.  **渲染响应**：将响应呈现为正确的内容类型。

DRF 的自定义`Request`类与 Django 的`Request`类非常相似，只是它可以配置为解析器。DRF 视图根据视图的设置和请求的内容类型在内容协商期间找到正确的解析器。解析后的内容可以像 Django 请求与`POST`表单提交一样作为`request.data`可用。

DRF 视图还使用一个专门的`Response`类，它使用渲染而不是 Django 模板。渲染器是在内容协商步骤中选择的。

大部分前面的步骤都是使用可配置的类来执行的。通过在项目的`settings.py`中创建一个名为`REST_FRAMEWORK`的字典，可以配置 DRF。让我们回顾一些最重要的设置：

+   `DEFAULT_PARSER_CLASSES`：默认支持 JSON、表单和多部分表单。其他解析器（例如 YAML 和 MessageBuffer）可作为第三方社区包提供。

+   `DEFAULT_AUTHENTICATION_CLASSES`：默认支持基于会话的身份验证和 HTTP 基本身份验证。会话身份验证可以使在应用的前端使用 API 更容易。DRF 附带了一个令牌身份验证类。OAuth（1 和 2）支持可通过第三方社区包获得。

+   `DEFAULT_PERMISSION_CLASSES`: 默认情况下允许任何用户执行任何操作（包括更新和删除操作）。DRF 附带了一组更严格的权限，列在文档中（[`www.django-rest-framework.org/api-guide/permissions/#api-reference`](https://www.django-rest-framework.org/api-guide/permissions/#api-reference)）。我们稍后还将看一下如何在本章后面创建自定义权限类。

+   `DEFAULT_THROTTLE_CLASSES`/`DEFAULT_THROTTLE_RATES`: 默认情况下为空（未限制）。DRF 提供了一个简单的节流方案，让我们可以在匿名请求和用户请求之间设置不同的速率。

+   `DEFAULT_RENDERER_CLASSES`: 这默认为 JSON 和*browsable*模板渲染器。可浏览的模板渲染器为视图和测试视图提供了一个简单的用户界面，适合开发。

我们将配置我们的 DRF 更加严格，即使在开发中也是如此。让我们在`django/config/settings.py`中更新以下新设置`dict`：

```py
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_THROTTLE_CLASSES': (
        'rest_framework.throttling.UserRateThrottle',
        'rest_framework.throttling.AnonRateThrottle',
    ),
    'DEFAULT_THROTTLE_RATES': {
        'user': '60/minute',
        'anon': '30/minute',
    },
}
```

这个配置默认将 API 限制为经过身份验证的用户，并对他们的请求设置了节流。经过身份验证的用户在被节流之前可以每分钟发出 60 个请求。未经身份验证的用户可以每分钟发出 30 个请求。DRF 接受`second`、`minute`、`hour`或`day`的节流周期。

接下来，让我们来看一下 DRF 的`Serializer`。

# 创建 Django REST Framework 序列化器

当 DRF 解析器解析请求的主体时，解析器基本上会返回一个 Python 字典。但是，在我们可以对数据执行任何操作之前，我们需要确认数据是否有效。在以前的 Django 视图中，我们会使用 Django 表单。在 DRF 中，我们使用`Serializer`类。

DRF 的`Serializer`类与 Django 表单类非常相似。两者都涉及接收验证数据和准备模型输出。但是，`Serializer`类不知道如何呈现其数据，而 Django 表单知道。请记住，在 DRF 视图中，渲染器负责将结果呈现为 JSON 或请求协商的任何其他格式。

就像 Django 表单一样，`Serializer`可以被创建来处理任意数据或基于 Django 模型。此外，`Serializer`由一组字段组成，我们可以用来控制序列化。当`Serializer`与模型相关联时，Django REST 框架知道为哪个模型`Field`使用哪个序列化器`Field`，类似于`ModelForm`的工作方式。

让我们在`django/mailinglist/serializers.py`中为我们的`MailingList`模型创建一个`Serializer`：

```py
from django.contrib.auth import get_user_model
from rest_framework import serializers

from mailinglist.models import MailingLIst

class MailingListSerializer(serializers.HyperlinkedModelSerializer):
    owner = serializers.PrimaryKeyRelatedField(
        queryset=get_user_model().objects.all())

    class Meta:
        model = MailingList
        fields = ('url', 'id', 'name', 'subscriber_set')
        read_only_fields = ('subscriber_set', )
        extra_kwargs = {
            'url': {'view_name': 'mailinglist:api-mailing-list-detail'},
            'subscriber_set': {'view_name': 'mailinglist:api-subscriber-detail'},
        }
```

这似乎与我们编写`ModelForm`的方式非常相似；让我们仔细看一下：

+   `HyperlinkedModelSerializer`: 这是显示到任何相关模型的超链接的`Serializer`类，因此当它显示`MailingList`的相关`Subscriber`模型实例时，它将显示一个链接（URL）到该实例的详细视图。

+   `owner = serializers.PrimaryKeyRelatedField(...)`: 这改变了序列化模型的`owner`字段。`PrimaryKeyRelatedField`返回相关对象的主键。当相关模型没有序列化器或相关 API 视图时（比如 Mail Ape 中的用户模型），这是有用的。

+   `model = MailingList`: 告诉我们的`Serializer`它正在序列化哪个模型

+   `fields = ('url', 'id', ...)`: 这列出了要序列化的模型字段。`HyperlinkedModelSerializer`包括一个额外的字段`url`，它是序列化模型详细视图的 URL。就像 Django 的`ModelForm`一样，`ModelSerializer`类（例如`HyperlinkedModelSerializer`）为每个模型字段有一组默认的序列化器字段。在我们的情况下，我们决定覆盖`owner`的表示方式（参考关于`owner`属性的前一点）。

+   `read_only_fields = ('subscriber_set', )`: 这简明地列出了哪些字段不可修改。在我们的情况下，这可以防止用户篡改`Subscriber`所在的邮件列表。

+   `extra_kwargs`: 这个字典让我们为每个字段的构造函数提供额外的参数，而不覆盖整个字段。通常是为了提供`view_name`参数，这是查找视图的 URL 所需的。

+   'url': {'view_name': '...'},: 这提供了`MailingList` API 详细视图的名称。

+   'subscriber_set': {'view_name': '...'},: 这提供了`Subscriber` API 详细视图的名称。

实际上有两种标记`Serializer`字段为只读的方法。一种是使用`read_only_fields`属性，就像前面的代码示例中那样。另一种是将`read_only=True`作为`Field`类构造函数的参数传递（例如，`email = serializers.EmailField(max_length=240, read_only=True)`）。

接下来，我们将为我们的`Subscriber`模型创建两个`Serializer`。我们的两个订阅者将有一个区别：`Subscriber.email`是否可编辑。当他们创建`Subscriber`时，我们需要让用户写入`Subscriber.email`。但是，我们不希望他们在创建用户后能够更改电子邮件。

首先，让我们在`django/mailinglist/serialiers.py`中为`Subscription`模型创建一个`Serializer`：

```py
from rest_framework import serializers

from mailinglist.models import Subscriber

class SubscriberSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Subscriber
        fields = ('url', 'id', 'email', 'confirmed', 'mailing_list')
        extra_kwargs = {
            'url': {'view_name': 'mailinglist:api-subscriber-detail'},
            'mailing_list': {'view_name': 'mailinglist:api-mailing-list-detail'},
        }
```

`SubscriberSerializer`与我们的`MailingListSerializer`类似。我们使用了许多相同的元素：

+   子类化`serializers.HyperlinkedModelSerializer`

+   使用内部`Meta`类的`model`属性声明相关模型

+   使用内部`Meta`类的`fields`属性声明相关模型的字段

+   使用`extra_kwargs`字典和`view_name`键提供相关模型的详细视图名称。

对于我们的下一个`Serializer`类，我们将创建一个与`SubscriberSerializer`类似的类，但将`email`字段设置为只读；让我们将其添加到`django/mailinglist/serialiers.py`中：

```py
from rest_framework import serializers

from mailinglist.models import Subscriber

class ReadOnlyEmailSubscriberSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Subscriber
        fields = ('url', 'id', 'email', 'confirmed', 'mailing_list')
        read_only_fields = ('email', 'mailing_list',)
        extra_kwargs = {
            'url': {'view_name': 'mailinglist:api-subscriber-detail'},
            'mailing_list': {'view_name': 'mailinglist:api-mailing-list-detail'},
        }
```

这个`Serializer`让我们更新`Subscriber`是否`confirmed`，但不会让`Subscriber`的`email`字段发生变化。

现在我们已经创建了一些`Serializer`，我们可以看到它们与 Django 内置的`ModelForm`有多么相似。接下来，让我们创建一个`Permission`类，以防止用户访问彼此的`MailingList`和`Subscriber`模型实例。

# API 权限

在本节中，我们将创建一个权限类，Django REST 框架将使用它来检查用户是否可以对`MailingList`或`Subscriber`执行操作。这将执行与我们在第十章中创建的`UserCanUseMailingList`混合类非常相似的角色，开始 Mail Ape。

让我们在`django/mailinglist/permissions.py`中创建我们的`CanUseMailingList`类：

```py
from rest_framework.permissions import BasePermission

from mailinglist.models import Subscriber, MailingList

class CanUseMailingList(BasePermission):

    message = 'User does not have access to this resource.'

    def has_object_permission(self, request, view, obj):
        user = request.user
        if isinstance(obj, Subscriber):
            return obj.mailing_list.user_can_use_mailing_list(user)
        elif isinstance(obj, MailingList):
            return obj.user_can_use_mailing_list(user)
        return False
```

让我们更仔细地看一下我们的`CanUseMailingList`类中引入的一些新元素：

+   `BasePermission`: 提供权限类的基本约定，实现`has_permission()`和`has_object_permission()`方法，始终返回`True`

+   `message`: 这是`403`响应体的消息

+   `def has_object_permission(...)`: 检查请求的用户是否是相关`MailingList`的所有者

`CanUseMailingList`类不覆盖`BasePermission.has_permission(self, request, view)`，因为我们系统中的权限都是在对象级别而不是视图或模型级别。

如果您需要更动态的权限系统，您可能希望使用 Django 的内置权限系统（[`docs.djangoproject.com/en/2.0/topics/auth/default/#permissions-and-authorization`](https://docs.djangoproject.com/en/2.0/topics/auth/default/#permissions-and-authorization)）或 Django Guardian（[`github.com/django-guardian/django-guardian`](https://github.com/django-guardian/django-guardian)）。

现在我们有了`Serializer`和权限类，我们将编写我们的 API 视图。

# 创建我们的 API 视图

在本节中，我们将创建定义 Mail Ape 的 RESTful API 的实际视图。Django REST 框架提供了一系列基于类的视图，这些视图类似于 Django 的一系列基于类的视图。DRF 通用视图与 Django 通用视图的主要区别之一是它们如何将多个操作组合在一个单一的视图类中。例如，DRF 提供了`ListCreateAPIView`类，但 Django 只提供了`ListView`类和`CreateView`类。DRF 提供了`ListCreateAPIView`类，因为在`/api/v1/mailinglists`上的资源预期将提供`MailingList`模型实例的列表和创建端点。

Django REST 框架还提供了一套函数装饰器（[`www.django-rest-framework.org/api-guide/views/#function-based-views`](http://www.django-rest-framework.org/api-guide/views/#function-based-views)），这样你也可以使用基于函数的视图。

通过创建我们的 API 来学习更多关于 DRF 视图的知识，首先从`MailingList` API 视图开始。

# 创建 MailingList API 视图

Mail Ape 将提供一个 API 来创建、读取、更新和删除`MailingList`。为了支持这些操作，我们将创建以下两个视图：

+   一个扩展了`ListCreateAPIView`的`MailingListCreateListView`

+   一个扩展了`RetrieveUpdateDestroyAPIView`的`MailingListRetrieveUpdateDestroyView`

# 通过 API 列出邮件列表

为了支持获取用户的`MailingList`模型实例列表和创建新的`MailingList`模型实例，我们将在`django/mailinglist/views.py`中创建`MailingListCreateListView`类：

```py
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from mailinglist.permissions import CanUseMailingList
from mailinglist.serializers import MailingListSerializer

class MailingListCreateListView(generics.ListCreateAPIView):
    permission_classes = (IsAuthenticated, CanUseMailingList)
    serializer_class = MailingListSerializer

    def get_queryset(self):
        return self.request.user.mailinglist_set.all()

    def get_serializer(self, *args, **kwargs):
        if kwargs.get('data', None):
            data = kwargs.get('data', None)
            owner = {
                'owner': self.request.user.id,
            }
            data.update(owner)
        return super().get_serializer(*args, **kwargs)
```

让我们详细查看我们的`MailingListCreateListView`类：

+   `ListCreateAPIView`：这是我们扩展的 DRF 通用视图。它通过`get_queryset()`方法返回的序列化内容响应`GET`请求。当它收到`POST`请求时，它将创建并返回一个`MailingList`模型实例。

+   `permission_classes`：这是一组权限类，按顺序调用。如果`IsAuthenticated`失败，那么`IsOwnerPermission`将不会被调用。

+   `serializer_class = MailingListSerializer`：这是该视图使用的序列化器。

+   `def get_queryset(self)`: 用于获取要序列化和返回的模型的`QuerySet`。

+   `def get_serializer(...)`: 用于获取序列化器实例。在我们的情况下，我们正在用当前登录的用户覆盖（如果有的话）从请求中收到的 owner。通过这样做，我们确保用户不能创建属于其他用户的邮件列表。这与我们可能如何在 Django 表单视图中覆盖`get_initial()`非常相似（例如，参考第十章中的`CreateMessageView`类，*开始 Mail Ape*）。

既然我们有了我们的视图，让我们在`django/mailinglist/urls.py`中添加以下代码：

```py
   path('api/v1/mailing-list', views.MailingListCreateListView.as_view(),
         name='api-mailing-list-list'),
```

现在，我们可以通过向`/mailinglist/api/v1/mailing-list`发送请求来创建和列出`MailingList`模型实例。

# 通过 API 编辑邮件列表

接下来，让我们通过在`django/mailinglist/views.py`中添加一个新视图来查看、更新和删除单个`MailingList`模型实例。

```py
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from mailinglist.permissions import CanUseMailingList
from mailinglist.serializers import MailingListSerializer
from mailinglist.models import MailingList

class MailingListRetrieveUpdateDestroyView(
    generics.RetrieveUpdateDestroyAPIView):

    permission_classes = (IsAuthenticated, CanUseMailingList)
    serializer_class = MailingListSerializer
    queryset = MailingList.objects.all()
```

`MailingListRetrieveUpdateDestroyView`看起来与我们之前的视图非常相似，但是扩展了`RetrieveUpdateDestroyAPIView`类。像 Django 内置的`DetailView`一样，`RetrieveUpdateDestroyAPIView`期望它将在请求路径中接收到`MailingList`模型实例的`pk`。`RetrieveUpdateDestroyAPIView`知道如何处理各种 HTTP 方法：

+   在`GET`请求中，它检索由`pk`参数标识的模型

+   在`PUT`请求中，它用收到的参数覆盖`pk`标识的模型的所有字段

+   在`PATCH`请求中，仅覆盖请求中收到的字段

+   在`DELETE`请求中，它删除由`pk`标识的模型

任何更新（无论是通过`PUT`还是`PATCH`）都由`MailingListSerializer`进行验证。

另一个区别是，我们为视图定义了一个`queryset`属性（`MailingList.objects.all()`），而不是一个`get_queryset()`方法。我们不需要动态限制我们的`QuerySet`，因为`CanUseMailingList`类将保护我们免受用户编辑/查看他们没有权限访问的`MailingLists`。

就像以前一样，现在我们需要将我们的视图连接到我们应用的 URLConf 中的`django/mailinglist/urls.py`，使用以下代码：

```py
   path('api/v1/mailinglist/<uuid:pk>',
         views.MailingListRetrieveUpdateDetroyView.as_view(),
         name='api-mailing-list-detail'),
```

请注意，我们从请求的路径中解析出`<uuid:pk>`参数，就像我们在一些 Django 的常规视图中对单个模型实例进行操作一样。

现在我们有了我们的`MailingList` API，让我们也允许我们的用户通过 API 管理`Subscriber`。

# 创建订阅者 API

在这一部分，我们将创建一个 API 来管理`Subscriber`模型实例。这个 API 将由两个视图支持：

+   `SubscriberListCreateView`用于列出和创建`Subscriber`模型实例

+   `SubscriberRetrieveUpdateDestroyView`用于检索、更新和删除`Subscriber`模型实例

# 列出和创建订阅者 API

`Subscriber`模型实例与`MailingList`模型实例有一个有趣的区别，即`Subscriber`模型实例与用户没有直接关联。要获取`Subscriber`模型实例的列表，我们需要知道应该查询哪个`MailingList`模型实例。`Subscriber`模型实例的创建面临同样的问题，因此这两个操作都必须接收相关的`MailingList`的`pk`来执行。

让我们从在`django/mailinglist/views.py`中创建我们的`SubscriberListCreateView`开始。

```py
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from mailinglist.permissions import CanUseMailingList
from mailinglist.serializers import SubscriberSerializer
from mailinglist.models import MailingList, Subscriber

class SubscriberListCreateView(generics.ListCreateAPIView):
    permission_classes = (IsAuthenticated, CanUseMailingList)
    serializer_class = SubscriberSerializer

    def get_queryset(self):
        mailing_list_pk = self.kwargs['mailing_list_pk']
        mailing_list = get_object_or_404(MailingList, id=mailing_list_pk)
        return mailing_list.subscriber_set.all()

    def get_serializer(self, *args, **kwargs):
        if kwargs.get('data'):
            data = kwargs.get('data')
            mailing_list = {
                'mailing_list': reverse(
                    'mailinglist:api-mailing-list-detail',
                    kwargs={'pk': self.kwargs['mailing_list_pk']})
            }
            data.update(mailing_list)
        return super().get_serializer(*args, **kwargs)
```

我们的`SubscriberListCreateView`类与我们的`MailingListCreateListView`类有很多共同之处，包括相同的基类和`permission_classes`属性。让我们更仔细地看看一些区别：

+   `serializer_class`: 使用`SubscriberSerializer`。

+   `get_queryset()`: 在返回所有相关的`Subscriber`模型实例的`QuerySet`之前，检查 URL 中标识的相关`MailingList`模型实例是否存在。

+   `get_serializer()`: 确保新的`Subscriber`与 URL 中的`MailingList`相关联。我们使用`reverse()`函数来识别相关的`MailingList`模型实例，因为`SubscriberSerializer`类继承自`HyperlinkedModelSerializer`类。`HyperlinkedModelSerializer`希望相关模型通过超链接或路径（而不是`pk`）来识别。

接下来，我们将在`django/mailinglist/urls.py`的 URLConf 中为我们的`SubscriberListCreateView`类添加一个`path()`对象：

```py
   path('api/v1/mailinglist/<uuid:mailing_list_pk>/subscribers',
         views.SubscriberListCreateView.as_view(),
         name='api-subscriber-list'),
```

在为我们的`SubscriberListCreateView`类添加一个`path()`对象时，我们需要确保有一个`mailing_list_pk`参数。这让`SubscriberListCreateView`知道要操作哪些`Subscriber`模型实例。

我们的用户现在可以通过我们的 RESTful API 向他们的`MailingList`添加`Subscriber`。向我们的 API 添加用户将触发确认电子邮件，因为`Subscriber.save()`将由我们的`SubscriberSerializer`调用。我们的 API 不需要知道如何发送电子邮件，因为我们的*fat model*是`Subscriber`行为的专家。

然而，这个 API 在 Mail Ape 中存在潜在的错误。我们当前的 API 允许我们添加一个已经确认的`Subscriber`。然而，我们的`Subscriber.save()`方法将向所有新的`Subscriber`模型实例的电子邮件地址发送确认电子邮件。这可能导致我们向已经确认的`Subscriber`发送垃圾邮件。为了解决这个 bug，让我们在`django/mailinglist/models.py`中更新`Subscriber.save`：

```py
class Subscriber(models.Model):
    # skipping unchanged attributes and methods

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        is_new = self._state.adding or force_insert
        super().save(force_insert=force_insert, force_update=force_update,
                     using=using, update_fields=update_fields)
        if is_new and not self.confirmed:
            self.send_confirmation_email()
```

现在，我们只有在保存新的*且*未确认的`Subscriber`模型实例时才调用`self.send_confirmation_email()`。

太棒了！现在，让我们创建一个视图来检索、更新和删除`Subscriber`模型实例。

# 通过 API 更新订阅者

现在，我们已经为 Subscriber 模型实例创建了列表 API 操作，我们可以创建一个 API 视图来检索、更新和删除单个`Subscriber`模型实例。

让我们将我们的视图添加到`django/mailinglist/views.py`中：

```py
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from mailinglist.permissions import CanUseMailingList
from mailinglist.serializers import ReadOnlyEmailSubscriberSerializer
from mailinglist.models import Subscriber

class SubscriberRetrieveUpdateDestroyView(
    generics.RetrieveUpdateDestroyAPIView):

    permission_classes = (IsAuthenticated, CanUseMailingList)
    serializer_class = ReadOnlyEmailSubscriberSerializer
    queryset = Subscriber.objects.all()
```

我们的`SubscriberRetrieveUpdateDestroyView`与我们的`MailingListRetrieveUpdateDestroyView`视图非常相似。两者都继承自相同的`RetrieveUpdateDestroyAPIView`类，以响应 HTTP 请求并使用相同的`permission_classes`列表提供核心行为。但是，`SubscriberRetrieveUpdateDestroyView`有两个不同之处：

+   `serializer_class = ReadOnlyEmailSubscriberSerializer`：这是一个不同的`Serializer`。在更新的情况下，我们不希望用户能够更改电子邮件地址。

+   `queryset = Subscriber.objects.all()`：这是所有`Subscribers`的`QuerySet`。我们不需要限制`QuerySet`，因为`CanUseMailingList`将防止未经授权的访问。

接下来，让我们确保我们可以通过将其添加到`django/mailinglist/urls.py`中的`urlpatterns`列表来路由到它：

```py
   path('api/v1/subscriber/<uuid:pk>',
         views.SubscriberRetrieveUpdateDestroyView.as_view(),
         name='api-subscriber-detail'),
```

现在我们有了我们的观点，让我们尝试在命令行上与它进行交互。

# 运行我们的 API

在本节中，我们将在命令行上运行 Mail Ape，并使用`curl`在命令行上与我们的 API 进行交互，`curl`是一个用于与服务器交互的流行命令行工具。在本节中，我们将执行以下功能：

+   在命令行上创建用户

+   在命令行上创建邮件列表

+   在命令行上获取`MailingList`列表

+   在命令行上创建`Subscriber`

+   在命令行上获取`Subscriber`列表

让我们首先使用 Django `manage.py shell`命令创建我们的用户：

```py
$ cd django
$ python manage.py shell
Python 3.6.3 (default) 
Type 'copyright', 'credits' or 'license' for more information
IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.
In [1]: from django.contrib.auth import get_user_model

In [2]: user = get_user_model().objects.create_user(username='user', password='secret')
In [3]: user.id
2
```

如果您已经使用 Web 界面注册了用户，可以使用该用户。此外，在生产中永远不要使用`secret`作为您的密码。

现在我们有了一个可以在命令行上使用的用户，让我们启动本地 Django 服务器：

```py
$ cd django
$ python manage.py runserver
```

现在我们的服务器正在运行，我们可以打开另一个 shell 并获取我们用户的`MailingList`列表：

```py
$ curl "http://localhost:8000/mailinglist/api/v1/mailing-list" \
     -u 'user:secret'
[]
```

让我们仔细看看我们的命令：

+   `curl`：这是我们正在使用的工具。

+   `"http://... api/v1/mailing-list"`：这是我们发送请求的 URL。

+   `-u 'user:secret'`：这是基本的身份验证凭据。`curl`会正确地对这些进行编码。

+   `[]`：这是服务器返回的空 JSON 列表。在我们的情况下，`user`还没有任何`MailingList`。

我们得到了一个 JSON 响应，因为 Django REST 框架默认配置为使用 JSON 渲染。

要为我们的用户创建一个`MailingList`，我们需要发送这样的`POST`请求：

```py
$ curl -X "POST" "http://localhost:8000/mailinglist/api/v1/mailing-list" \
     -H 'Content-Type: application/json; charset=utf-8' \
     -u 'user:secret' \
     -d $'{
  "name": "New List"
}'
{"url":"http://localhost:8000/mailinglist/api/v1/mailinglist/cd983e25-c6c8-48fa-9afa-1fd5627de9f1","id":"cd983e25-c6c8-48fa-9afa-1fd5627de9f1","name":"New List","owner":2,"subscriber_set":[]}
```

这是一个更长的命令，结果也更长。让我们来看看每个新参数：

+   `-H 'Content-Type: application/json; charset=utf-8' \`：这添加了一个新的 HTTP `Content-Type`头，告诉服务器将正文解析为 JSON。

+   `-d $'{ ... }'`：这指定了请求的正文。在我们的情况下，我们正在发送一个 JSON 对象，其中包含新邮件列表的名称。

+   `"url":"http://...cd983e25-c6c8-48fa-9afa-1fd5627de9f1"`：这是新`MailingLIst`的完整详细信息的 URL。

+   `"name":"New List"`：这显示了我们请求的新列表的名称。

+   `"owner":2`：这显示了列表所有者的 ID。这与我们之前创建的用户的 ID 匹配，并包含在此请求中（使用`-u`）。

+   `"subscriber_set":[]`：这显示了此邮件列表中没有订阅者。

现在我们可以重复我们最初的请求来列出`MailingList`，并检查我们的新`MailingList`是否包含在内：

```py
$ curl "http://localhost:8000/mailinglist/api/v1/mailing-list" \
     -u 'user:secret'
[{"url":"http://localhost:8000/mailinglist/api/v1/mailinglist/cd983e25-c6c8-48fa-9afa-1fd5627de9f1","id":"cd983e25-c6c8-48fa-9afa-1fd5627de9f1","name":"New List","owner":2,"subscriber_set":[]}]
```

看到我们可以在开发中运行我们的服务器和 API 是很好的，但我们不想总是依赖手动测试。让我们看看如何自动化测试我们的 API。

如果您想测试创建订阅者，请确保您的 Celery 代理（例如 Redis）正在运行，并且您有一个工作程序来消耗任务以获得完整的体验。

# 测试您的 API

API 通过让用户自动化他们与我们服务的交互来为我们的用户提供价值。当然，DRF 也帮助我们自动化测试我们的代码。

DRF 为我们讨论的所有常见 Django 工具提供了替代品第八章，*测试 Answerly*：

+   Django 的`RequestFactory`类的`APIRequestFactory`

+   Django 的`Client`类的`APIClient`

+   Django 的`TestCase`类的`APITestCase`

`APIRequestFactory`和`APIClient`使得更容易发送格式化为我们的 API 的请求。例如，它们提供了一种简单的方法来为不依赖于基于会话的认证的请求设置凭据。否则，这两个类的作用与它们的默认 Django 等效类相同。

`APITestCase`类简单地扩展了 Django 的`TestCase`类，并用`APIClient`替换了 Django 的`Client`。

让我们看一个例子，我们可以添加到`django/mailinglist/tests.py`中：

```py
class ListMailingListsWithAPITestCase(APITestCase):

    def setUp(self):
        password = 'password'
        username = 'unit test'
        self.user = get_user_model().objects.create_user(
            username=username,
            password=password
        )
        cred_bytes = '{}:{}'.format(username, password).encode('utf-8')
        self.basic_auth = base64.b64encode(cred_bytes).decode('utf-8')

    def test_listing_all_my_mailing_lists(self):
        mailing_lists = [
            MailingList.objects.create(
                name='unit test {}'.format(i),
                owner=self.user)
            for i in range(3)
        ]

        self.client.credentials(
            HTTP_AUTHORIZATION='Basic {}'.format(self.basic_auth))

        response = self.client.get('/mailinglist/api/v1/mailing-list')

        self.assertEqual(200, response.status_code)
        parsed = json.loads(response.content)
        self.assertEqual(3, len(parsed))

        content = str(response.content)
        for ml in mailing_lists:
            self.assertIn(str(ml.id), content)
            self.assertIn(ml.name, content)
```

让我们更仔细地看一下在我们的`ListMailingListsWithAPITestCase`类中引入的新代码：

+   `class ListMailingListsWithAPITestCase(APITestCase)`: 这使得`APITestCase`成为我们的父类。`APITestCase`类基本上是一个`TestCase`类，只是用`APIClient`对象代替了常规的 Django `Client`对象分配给`client`属性。我们将使用这个类来测试我们的视图。

+   `base64.b64encode(...)`: 这对我们的用户名和密码进行了 base64 编码。我们将使用这个来提供一个 HTTP 基本认证头。我们必须使用`base64.b64encode()`而不是`base64.base64()`，因为后者会引入空格来视觉上分隔长字符串。此外，我们需要对我们的字符串进行`encode`/`decode`，因为`b64encode()`操作`byte`对象。

+   `client.credentials()`: 这让我们设置一个认证头，以便将来由这个`client`对象发送所有的请求。在我们的情况下，我们发送了一个 HTTP 基本认证头。

+   `json.loads(response.content)`: 这解析了响应内容体并返回一个 Python 列表。

+   `self.assertEqual(3, len(parsed))`: 这确认了解析列表中的项目数量是正确的。

如果我们使用`self.client`发送第二个请求，我们不需要重新认证，因为`client.credentials()`会记住它接收到的内容，并继续将其传递给所有请求。我们可以通过调用`client.credentials()`来清除凭据。

现在，我们知道如何测试我们的 API 代码了！

# 摘要

在本章中，我们介绍了如何使用 Django REST 框架为我们的 Django 项目创建 RESTful API。我们看到 Django REST 框架使用了与 Django 表单和 Django 通用视图类似的原则。我们还使用了 Django REST 框架中的一些核心类，我们使用了`ModelSerializer`来构建基于 Django 模型的`Serializer`，并使用了`ListCreateAPIView`来创建一个可以列出和创建 Django 模型的视图。我们使用了`RetrieveUpdateDestroyAPIView`来管理基于其主键的 Django 模型实例。

接下来，我们将使用亚马逊网络服务将我们的代码部署到互联网上。
