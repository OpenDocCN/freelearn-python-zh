# 第十二章：构建 API

在上一章中，你构建了一个学生注册和课程报名系统。你创建了显示课程内容的视图，并学习了如何使用 Django 的缓存框架。在本章中，你会学习以下知识点：

- 构建一个 RESTful API
- 为 API 视图处理认证和权限
- 创建 API 视图集和路由

## 12.1 构建 RESTful API

你可能想要创建一个接口，让其它服务可以与你的 web 应用交互。通过构建一个 API，你可以允许第三方以编程方式使用信息和操作你的应用。

你可以通过很多方式构建 API，但最好是遵循 REST 原则。REST 架构是表述性状态传递（`Representational State Transfer`）的缩写。RESTful API 是基于资源的。你的模型代表资源，HTTP 方法（比如 GET，POST，PUT 或 DELETE）用于检索，创建，更新或者删除对象。HTTP 响应代码也可以在这个上下文中使用。返回的不同 HTTP 响应代码表示 HTTP 请求的结果，比如 2XX 响应代码表示成功，4XX 表示错误等等。

RESTful API 最常用的交互数据的格式是 JSON 和 XML。我们将为项目构建一个 JSON 序列化的 REST API。我们的 API 会提供以下功能：

- 检索主题
- 检索可用的课程
- 检索课程内容
- 报名参加课程

我们可以通过 Django 创建自定义视图，从头开始构建 API。但是有很多第三方模块可以简化创建 API，其中最流行的是`Django Rest Framework`。

### 12.1.1 安装 Django Rest Framework

`Django Rest Framework`可以很容易的为项目构建 REST API。你可以在[这里](http://www.django-rest-framework.org/)查看所有文档。

打开终端，使用以下命令安装框架：

```py
pip install djangorestframework
```

编辑`educa`项目的`settings.py`文件，在`INSTALLED_APPS`设置中添加`rest_framework`：

```py
INSTALLED_APPS = [
	# ...
	'rest_framework',
]
```

然后在`settings.py`文件中添加以下代码：

```py
REST_FRAMEWORK = {
    'DEFAULT_PREMISSION_CLASSES': [
        'rest_framework.permissions.DjangoModelPermissionsOrAnonReadOnly'
    ]
}
```

你可以使用`REST_FRAMEWORK`设置为 API 提供一个特定配置。REST Framework 提供了大量设置来配置默认行为。`DEFAULT_PREMISSION_CLASSES`设置指定读取，创建，更新或者删除对象的默认权限。我们设置`DjangoModelPermissionsOrAnonReadOnly`是唯一的默认权限类。这个类依赖 Django 的权限系统，允许用户创建，更新或删除对象，同时为匿名用户提供只读访问。之后你会学习更多关于权限的内容。

你可以访问[这里](http://www.django-rest-framework.org/api-guide/settings/)查看完整的 REST Framework 可用设置列表。

### 12.1.2 定义序列化器

设置 REST Framework 之后，我们需要指定如何序列化我们的数据。输出数据必须序列化为指定格式，输入数据会反序列化处理。框架为单个类构建序列化器提供了以下类：

- `Serializer`：为普通 Python 类实例提供序列化
- `ModelSerializer`：为模型实例提供序列化
- `HyperlinkedModelSerializer`：与`ModelSerializer`一样，但使用链接而不是主键表示对象关系

让我们构建第一个序列化器。在`courses`应用目录中创建以下文件结构：

```py
api/
	__init__.py
	serializers.py
```

我们会在`api`目录中构建所有 API 功能，保持良好的文件结构。编辑`serializers.py`文件，并添加以下代码：

```py
from rest_framework import serializers
from ..models import Subject

class SubjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subject
        fields = ('id', 'title', 'slug')
```

这是`Subject`模型的序列化器。序列化器的定义类似于 Django 的`Form`和`ModelForm`类。`Meta`类允许你指定序列化的模型和序列化中包括的字段。如果没有设置`fields`属性，则会包括所有模型字段。

让我们试试序列化器。打开终端执行`python manage.py shell`命令，然后执行以下代码：

```py
>>> from courses.models import Subject
>>> from courses.api.serializers import SubjectSerializer
>>> subject = Subject.objects.latest('id')
>>> serializer = SubjectSerializer(subject)
>>> serializer.data
```

在这个例子中，我们获得一个`Subject`对象，创建一个`SubjectSerializer`实例，然后访问序列化的数据。你会看到以下输出：

```py
{'id': 4, 'slug': 'mathematics', 'title': 'Mathematics'}
```

正如你所看到的，模型数据转换为 Python 的原生数据类型。

### 12.1.3 理解解析器和渲染器

在 HTTP 响应中返回序列化的数据之前，需要把它渲染为特定格式。同样的，当你获得 HTTP 请求时，在你操作它之前，需要解析传入的数据并反序列化数据。REST Framework 包括渲染器和解析器来处理这些操作。

让我们看看如何解析收到的数据。给定一个 JSON 字符串输入，你可以使用 REST Framework 提供的`JSONParser`类转换为 Python 对象。在 Python 终端中执行以下代码：

```py
from io import BytesIO
from rest_framework.parsers import JSONParser
data = b'{"id":4,"title":"Music","slug":"music"}'
JSONParser().parse(BytesIO(data))
```

你会看到以下输出：

```py
{'id': 4, 'title': 'Music', 'slug': 'music'}
```

REST Framework 还包括`Renderer`类，允许你格式化 API 响应。框架通过内容协商决定使用哪个渲染器。它检查请求的`Accept`头，决定响应期望的内容类型。根据情况，渲染器由 URL 格式后缀确定。例如，触发`JSONRenderer`的访问会返回 JSON 响应。

回到终端执行以下代码，从上一个序列化器例子中渲染`serializer`对象：

```py
>>> from rest_framework.renderers import JSONRenderer
>>> JSONRenderer().render(serializer.data)
```

你会看到以下输出：

```py
b'{"id":4,"title":"Mathematics","slug":"mathematics"}'
```

我们使用`JSONRenderer`渲染序列化的数据位 JSON。默认情况下，REST Framework 使用两个不同的渲染器：`JSONRenderer`和`BrowsableAPIRenderer`。后者提供一个 web 接口，可以很容易的浏览你的 API。你可以在`REST_FRAMEWORK`设置的`DEFAULT_RENDERER_CLASSES`选项中修改默认的渲染器类。

你可以查看更多关于[渲染器](http://www.django-rest-framework.org/api-guide/renderers/)和[解析器](http://www.django-rest-framework.org/api-guide/parsers/)的信息。

### 12.1.4 构建列表和详情视图

REST Framework 自带一组构建 API 的通用视图和 mixins。它们提供了检索，创建，更新或删除模型对象的功能。你可以在[这里](http://www.django-rest-framework.org/api-guide/generic-views/)查看 REST Framework 提供的所有通用的 mixins 和视图。

让我们创建检索`Subject`对象的列表和详情视图。在`courses/api/`目录中创建`views.py`文件，并添加以下代码：

```py
from rest_framework import generics
from ..models import Subject
from .serializers import SubjectSerializer

class SubjectListView(generics.ListAPIView):
    queryset = Subject.objects.all()
    serializer_class = SubjectSerializer

class SubjectDetailView(generics.RetrieveAPIView):
    queryset = Subject.objects.all()
    serializer_class = SubjectSerializer
```

在这段代码中，我们使用了 REST Framework 的通用`ListAPIView`和`RetrieveAPIView`。我们在详情视图中包括一个`pk` URL 参数，来检索给定主键的对象。两个视图都包括以下属性：

- `queryset`：用于检索对象的基础`QuerySet`。
- `serializer_class`：序列化对象的类。

让我们为视图添加 URL 模式。在`courses/api/`目录中创建`urls.py`文件，并添加以下代码：

```py
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^subjects/$', views.SubjectListView.as_view(), name='subject_list'),
    url(r'^subjects/(?P<pk>\d+)/$', views.SubjectDetailView.as_view(), name='subject_detail'),
]
```

编辑`educa`项目的主`urls.py`文件，并引入 API 模式：

```py
urlpatterns = [
    # ...
    url(r'^api/', include('courses.api.urls', namespace='api')),
]
```

我们为 API 的 URL 使用`api`命名空间。使用`python manage.py runserver`启动开发服务器。打开终端，并使用`curl`获取`http://127.0.0.1:8000/api/subjects/`：

```py
bogon:educa lakerszhy$ curl http://127.0.0.1:8000/api/subjects/
```

你会看到类似以下的响应：

```py
[{"id":4,"title":"Mathematics","slug":"mathematics"},
{"id":3,"title":"Music","slug":"music"},
{"id":2,"title":"Physics","slug":"physics"},
{"id":1,"title":"Programming","slug":"programming"}]
```

HTTP 响应包括 JSON 格式的`Subject`对象列表。如果你的操作系统没有安装`curl`，请在[这里](https://curl.haxx.se/dlwiz/)下载。除了`curl`，你还可以使用其它工具发送自定义 HTTP 请求，比如浏览器扩展`Postman`，你可以在[这里](https://www.getpostman.com/)下载`Postman`。

在浏览器中打开`http://127.0.0.1:8000/api/subjects/`。你会看到 REST Framework 的可浏览 API，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE12.1.png)

这个 HTML 界面由`BrowsableAPIRenderer`渲染器提供。你还可以在 URL 中包括`id`来访问一个`Subject`对象的 API 详情视图。在浏览器中打开`http://127.0.0.1:8000/api/subjects/1/`。你会看到单个`Subject`对象以 JSON 格式渲染。

### 12.1.5 创建嵌套的序列化器

我们将为`Course`模型创建一个序列化器。编辑`api/serializers.py`文件，并添加以下代码：

```py
from ..models import Course

class CourseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Course
        fields = ('id', 'subject', 'title', 'slug', 
            'overview', 'created', 'owner', 'modules')
```

让我们看看一个`Course`对象是如何被序列化的。在终端执行`python manage.py shell`，然后执行以下代码：

```py
>>> from rest_framework.renderers import JSONRenderer
>>> from courses.models import Course
>>> from courses.api.serializers import CourseSerializer
>>> course = Course.objects.latest('id')
>>> serializer = CourseSerializer(course)
>>> JSONRenderer().render(serializer.data)
```

你获得的 JSON 对象包括我们在`CourseSerializer`中指定的字段。你会看到`modules`管理器的关联对象被序列化为主键列表，如下所示：

```py
"modules": [17, 18, 19, 20, 21, 22]
```

我们想包括每个单元的更多信息，所以我们需要序列化`Module`对象，并且嵌套它们。修改`api/serializers.py`文件中的上一段代码，如下所示：

```py
from ..models import Course, Module

class ModuleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Module
        fields = ('order', 'title', 'description')

class CourseSerializer(serializers.ModelSerializer):
    modules = ModuleSerializer(many=True, read_only=True)
    
    class Meta:
        model = Course
        fields = ('id', 'subject', 'title', 'slug', 
            'overview', 'created', 'owner', 'modules')
```

我们定义了`ModuleSerializer`，为`Module`模型提供了序列化。然后我们添加`modules`属性到`CourseSerializer`来嵌套`ModuleSerializer`序列化器。我们设置`many=True`表示正在序列化的是多个对象。`read_only`参数表示该字段是可读的，并且不应该包括在任何输入中来创建或更新对象。

打开终端，并再创建一个`CourseSerializer`实例。使用`JSONRenderer`渲染序列化器的`data`属性。这次，单元列表被嵌套的`ModuleSerializer`序列化器序列化，如下所示：

```py
"modules": [
    {
        "order": 0,
        "title": "Django overview",
        "description": "A brief overview about the Web Framework."
    }, 
    {
        "order": 1,
        "title": "Installing Django",
        "description": "How to install Django."
    },
    ...
```

你可以在[这里](http://www.django-rest-framework.org/api-guide/serializers/)阅读更多关于序列化器的信息。

### 12.1.6 构建自定义视图

REST Framework 提供了一个`APIView`类，可以在 Django 的`View`类之上构建 API 功能。`APIView`类与`View`类不同，它使用 REST Framework 的自定义`Request`和`Response`对象，并且处理`APIException`异常返回相应的 HTTP 响应。它还包括一个内置的认证和授权系统来管理视图的访问。

我们将为用户创建课程报名的视图。编辑`api/views.py`文件，并添加以下代码：

```py
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from ..models import Course

class CourseEnrollView(APIView):
    def post(self, request, pk, format=None):
        course = get_object_or_404(Course, pk=pk)
        course.students.add(request.user)
        return Response({'enrolled': True})
```

`CourseEnrollView`视图处理用户报名参加课程。上面的代码完成以下任务：

- 我们创建了一个继承自`APIView`的自定义视图。
- 我们为 POST 操作定义了`post()`方法。这个视图不允许其它 HTTP 方法。
- 我们期望 URL 参数`pk`包含课程 ID。我们用给定的`pk`参数检索课程，如果没有找到则抛出 404 异常。
- 我们添加当前对象到`Course`对象的多对多关系`students`中，并返回成功的响应。

编辑`api/urls.py`文件，并为`CourseEnrollView`视图添加 URL 模式：

```py
url(r'^courses/(?P<pk>\d+)/enroll/$', views.CourseEnrollView.as_view(), name='course_enroll'),
```

理论上，我们现在可以执行一个 POST 请求，为当前用户报名参加一个课程。但是，我们需要识别用户，并阻止未认证用户访问这个视图。让我们看看 API 认证和权限是如何工作的。

### 12.1.7 处理认证

REST Framework 提供了识别执行请求用户的认证类。如果认证成功，框架会在`request.user`中设置认证的`User`对象。否则设置为 Django 的`AnonymousUser`实例。

REST Framework 提供以下认证后台：

- `BasicAuthentication`：HTTP 基础认证。客户端用 Base64 在`Authorization` HTTP 头中发送用户和密码。你可以在[这里](https://en.wikipedia.org/wiki/Basic_access_authentication)进一步学习。
- `TokenAuthentication`：基于令牌的认证。一个`Token`模型用于存储用户令牌。用户在`Authorization` HTTP 头中包括用于认证的令牌。
- `SessionAuthentication`：使用 Django 的会话后台用于认证。当执行从你的网站前端到 API 的 AJAX 请求时，这个后台非常有用。

你可以通过继承 REST Framework 提供的`BaseAuthentication`类，并覆写`authenticate()`方法来构建自定义认证后台。

你可以基于单个视图设置认证，或者用`DEFAULT_AUTHENTICATION_CLASSES`设置为全局认证。

> 认证只识别执行请求的用户。它不会允许或阻止访问视图。你必须使用权限来显示访问视图。

你可以在[这里](http://www.django-rest-framework.org/api-guide/authentication/)查看所有关于认证的信息。

让我们添加`BasicAuthentication`到我们的视图。编辑`courses`应用的`api/views.py`文件，并添加`authentication_classes`属性到`CourseEnrollView`：

```py
from rest_framework.authentication import BasicAuthentication

class CourseEnrollView(APIView):
    authentication_classes = (BasicAuthentication, )
    # ...
```

用户将通过设置在 HTTP 请求中的`Authorization`头的证书识别。

### 12.1.8 添加权限到视图

REST Framework 包括一个权限系统，用于限制视图的访问。REST Framework 的一些内置权限是：

- `AllowAny`：不限制访问，不管用户是否认证。
- `IsAuthenticated`：只允许认证的用户访问。
- `IsAuthenticatedOrReadOnly`：认证用户可以完全访问。匿名用户只允许执行读取方法，比如 GET，HEAD 或 OPTIONS。
- `DjangoModelPermissions`：捆绑到`django.contrib.auth`的权限。视图需要一个`queryset`属性。只有分配了模型权限的认证用户才能获得权限。
- `DjangoObjectPermissions`：基于单个对象的 Django 权限。

如果用户被拒绝访问，他们通常会获得以下某个 HTTP 错误代码：

- `HTTP 401`：未认证
- `HTTP 403`：没有权限

你可以在[这里](http://www.django-rest-framework.org/api-guide/permissions/)阅读更多关于权限的信息。

编辑`courses`应用的`api/views.py`文件，并在`CourseEnrollView`中添加`permission_classes`属性：

```py
from rest_framework.authentication import BasicAuthentication
from rest_framework.permissions import IsAuthenticated

class CourseEnrollView(APIView):
    authentication_classes = (BasicAuthentication, )
    permission_classes = (IsAuthenticated, )
    # ..
```

我们引入了`IsAuthenticated`权限。这会阻止匿名用户访问这个视图。现在我们可以执行 POST 请求到新的 API 方法。

确保开发服务器正在运行。打开终端并执行以下命令：

```py
curl -i -X POST http://127.0.0.1:8000/api/courses/1/enroll/
```

你会获得以下响应：

```py
HTTP/1.0 401 UNAUTHORIZED
...
{"detail": "Authentication credentials were not provided."}
```

因为我们是未认证用户，所以如期获得`401` HTTP 代码。让我们用其中一个用户进行基础认证。执行以下命令：

```py
curl -i -X POST -u student:password http://127.0.0.1:8000/api/courses/1/enroll/
```

用已存在用户凭证替换`student:password`。你会获得以下响应：

```py
HTTP/1.0 200 OK
...
{"enrolled": true}
```

你可以访问管理站点，检查用户是否报名参加课程。

### 12.1.9 创建视图集和路由

`ViewSets`允许你定义你的 API 交互，并让 REST Framework 用`Router`对象动态构建 URL。通过视图集，你可以避免多个视图的重复逻辑。视图集包括典型的创建，检索，更新，删除操作，分别是`list()`，`create()`，`retrieve()`，`update()`，`partial_update()`和`destroy()`。

让我们为`Course`模型创建一个视图集。编辑`api/views.py`文件，并添加以下代码：

```py
from rest_framework import viewsets
from .serializers import CourseSerializer

class CourseViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Course.objects.all()
    serializer_class = CourseSerializer
```

我们从`ReadOnlyModelViewSet`继承，它提供了只读操作`list()`和`retrieve()`，用于列出对象或检索单个对象。编辑`api/urls.py`文件，并为我们的视图集创建一个路由：

```py
from django.conf.urls import url, include
from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register('courses', views.CourseViewSet)

urlpatterns = [
    # ...
    url(r'^', include(router.urls)),
]
```

我们创建了一个`DefaultRouter`对象，并用`courses`前缀注册我们的视图集。路由负责为我们的视图集自动生成 URL。

在浏览器中打开`http://127.0.0.1:8000/api/`。你会看到路由在它的基础 URL 中列出所有视图集，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE12.2.png)

你现在可以访问`http://127.0.0.1:8000/api/courses/`检索课程列表。

你可以在[这里](http://www.django-rest-framework.org/api-guide/viewsets/)进一步学习视图集。你还可以在这里查看更多关于[路由](http://www.django-rest-framework.org/api-guide/routers/)的信息。

### 12.1.10 添加额外操作到视图集

你可以添加额外操作到视图集中。让我们把之前的`CourseEnrollView`视图为一个自定义视图集操作。编辑`api/views.py`文件，并修改`CourseViewSet`类：

```py
from rest_framework.decorators import detail_route

class CourseViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Course.objects.all()
    serializer_class = CourseSerializer

    @detail_route(
        methods=['post'],
        authentication_classes=[BasicAuthentication],
        permission_classes=[IsAuthenticated]
    )
    def enroll(self, request, *args, **kwargs):
        course = self.get_object()
        course.students.add(request.user)
        return Response({'enrolled': True})
```

我们添加了一个自定义的`enroll()`方法，它代表这个视图集的一个额外操作。上面的代码执行以下任务：

- 我们使用框架的`detail_route`装饰器，指定这是在单个对象上执行的操作。
- 装饰器允许我们为操作添加自定义属性。我们指定这个视图只允许 POST 方法，并设置了认证和权限类。
- 我们使用`self.get_object()`检索`Courses`对象。
- 我们把当前用户添加到`students`多对多关系中，并返回一个自定义的成功响应。

编辑`api/urls.py`文件，移除以下 URL，因为我们不再需要它：

```py
url(r'^courses/(?P<pk>\d+)/enroll/$', views.CourseEnrollView.as_view(), name='course_enroll'),
```

然后编辑`api/views.py`文件，移除`CourseEnrollView`类。

现在，报名参加课程的 URL 由路由自动生成。因为它使用操作名`enroll`，所以 URL 保持不变。

### 12.1.11 创建自定义权限

我们希望学生可以访问它们报名的课程内容。只有报名的学生才可以访问课程内容。最好的实现方式是使用一个自定义权限类。Django 提供的`BasePermission`类允许你定义以下方法：

- `has_permission()`：视图级别的权限检查
- `has_object_permission()`：实例级别的权限检查

如果获得访问权限，这些方法返回`True`，否则返回`False`。在`courses/api/`目录中创建`permissions.py`文件，并添加以下代码：

```py
from rest_framework.permissions import BasePermission

class IsEnrolled(BasePermission):
    def has_object_permission(self, request, view, obj):
        return obj.students.filter(id=request.user.id).exists()
```

我们从`BasePermission`类继承，并覆写`has_object_permission()`。我们检查执行请求的用户是否存在`Course`对象的`students`关系中。我们下一步会使用`IsEnrolled`权限。

### 12.1.12 序列化课程内容

我们需要序列化课程内容。`Content`模型包括一个通用外键，允许我们访问关联对象的不同内容模型。但是，我们在上一章为所有内容模型添加了通用的`render()`方法。我们可以使用这个方法为 API 提供渲染后的内容。

编辑`courses`应用的`api/serializers.py`文件，并添加以下代码：

```py
from ..models import Content

class ItemRelatedField(serializers.RelatedField):
    def to_representation(self, value):
        return value.render()

class ContentSerializer(serializers.ModelSerializer):
    item = ItemRelatedField(read_only=True)

    class Meta:
        model = Content
        fields = ('order', 'item')
```

在这段代码中，通过继承 REST Framework 提供的`RelatedField`序列化器字段和覆写`to_representation()`方法，我们定义了一个自定义字段。我们为`Content`模型定义了`ContentSerializer`序列化器，并用自定义字段作为`item`通用外键。

我们需要一个包括内容的`Module`模型的替换序列化器，以及一个扩展的`Course`序列化器。编辑`api/serializers.py`文件，并添加以下代码：

```py
class ModuleWithContentsSerializer(serializers.ModelSerializer):
    contents = ContentSerializer(many=True)

    class Meta:
        model = Module
        fields = ('order', 'title', 'description', 'contents')

class CourseWithContentsSerializer(serializers.ModelSerializer):
    modules = ModuleWithContentsSerializer(many=True)

    class Meta:
        model = Course
        fields = ('id', 'subject', 'title', 'slug', 'overview', 
            'created', 'owner', 'modules')
```

让我们创建一个模仿`retrieve()`操作，但是包括课程内容的视图。编辑`api/views.py`文件，并在`CourseViewSet`类中添加以下方法：

```py
from .permissions import IsEnrolled
from .serializers import CourseWithContentsSerializer

class CourseViewSet(viewsets.ReadOnlyModelViewSet):
    # ...
    @detail_route(
        methods=['get'],
        serializer_class=CourseWithContentsSerializer,
        authentication_classes=[BasicAuthentication],
        permission_classes=[IsAuthenticated, IsEnrolled]
    )
    def contents(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)
```

这个方法执行以下任务：

- 我们使用`detail_route`装饰器指定该操作在单个对象上执行。
- 我们指定该操作只允许 GET 方法。
- 我们使用新的`CourseWithContentsSerializer`序列化器类，它包括渲染的课程内容。
- 我们使用`IsAuthenticated`和自定义的`IsEnrolled`权限。这样可以确保只有报名的用户可以访问课程内容。
- 我们使用存在的`retrieve()`操作返回课程对象。

在浏览器中打开`http://127.0.0.1:8000/api/courses/1/contents/`。如果你用正确证书访问视图，你会看到课程的每个单元，包括渲染后的课程内容的 HTML，如下所示：

```py
{
   "order": 0,
   "title": "Installing Django",
   "description": "",
   "contents": [
        {
        "order": 0,
        "item": "<p>Take a look at the following video for installing Django:</p>\n"
        }, 
        {
        "order": 1,
        "item": "\n<iframe width=\"480\" height=\"360\" src=\"http://www.youtube.com/embed/bgV39DlmZ2U?wmode=opaque\" frameborder=\"0\" allowfullscreen></iframe>\n\n"
        } 
    ]
    }
```

你已经构建了一个简单的 API，允许其它服务通过编程方式访问`course`应用。REST Framework 还允许你用`ModelViewSet`视图集管理创建和编辑对象。我们已经学习了 Django Rest Framework 的主要部分，但你仍可以在[这里](http://www.django-rest-framework.org/)进一步学习它的特性。

## 12.2 总结

在这一章中，你创建了一个 RESTful API，可以让其它服务与你的 web 应用交互。

额外的第十三章可以在[这里](https://www.packtpub.com/sites/default/files/downloads/Django_By_Example_GoingLive.pdf)下载。它教你如何使用`uWSGI`和`NGINX`构建一个生产环境。你还会学习如何实现一个自定义的中间件和创建自定义的管理命令。

你已经到达了本书的结尾。恭喜你！你已经学会了用 Django 构建一个成功的 web 应用所需要的技巧。本书指导你完成开发实际项目，以及将 Django 与其它技术结合。现在你已经准备好创建自己的 Django 项目，不管是一个简单的原型还是一个大型的 web 应用。

祝你下一次 Django 冒险活动好运！