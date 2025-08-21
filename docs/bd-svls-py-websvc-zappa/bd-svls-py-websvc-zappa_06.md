# 第六章：使用 Zappa 构建 Django REST API

在本章中，我们将使用 Django Rest Framework 创建一个 RESTful API。它将基于一个简单的 RESTful API，具有**CRUD**（**创建**，**检索**，**更新**和**删除**）操作。我们可以考虑之前开发的**ImageGallery**应用程序与 REST API 扩展。在这里，我们将为`PhotoAlbum`创建一个 API，用户可以通过 REST API 界面创建新相册以及图片。

本章我们将涵盖以下主题：

+   安装和配置 Django REST 框架

+   设计 REST API

+   使用 Zappa 构建、测试和部署 Django 应用程序

# 技术要求

在继续之前，有一些技术先决条件需要满足。这些先决条件是设置和配置开发环境所必需的。以下是所需软件的列表：

+   Ubuntu 16.04/Mac/Windows

+   Python 3.6

+   Pipenv 工具

+   Django

+   Django Rest Framework

+   Django Rest Framework JWT

+   Django 存储

+   Django Imagekit

+   Boto3

+   Zappa

我们将在虚拟环境中安装这些软件包。在下一节中，我们将看到有关安装过程的详细信息。

# 安装和配置 Django REST 框架

我们已经在第五章的*设置虚拟环境*部分详细介绍了虚拟环境设置过程。您可以按照这些说明配置 pipenv 工具并为本章创建一个新的虚拟环境。让我们转到下一节，使用 pipenv 工具安装所需的软件包。

# 安装所需的软件包

我们将使用 Django REST 框架开发 REST API，因此我们需要使用`pipenv install <package_name>`命令安装以下软件包：

+   `django`

+   `djangorestframework`

+   `djangorestframework-jwt`

+   `django-storages`

+   `django-imagekit`

+   `boto3`

+   `zappa`

您可以通过在空格分隔的其他软件包之后提及其他软件包来一次安装多个软件包，例如`pipenv install <package_one> <package_two> ...`。

安装这些软件包后，我们可以继续实施，并且将有以下提到的`Pipfile`：

文件—`Pipfile`：

```py
[[source]]

url = "https://pypi.python.org/simple"
verify_ssl = true
name = "pypi"

[dev-packages]

[packages]

django = "*"
djangorestframework = "*"
django-storages = "*"
django-imagekit = "*"
"boto3" = "*"
zappa = "*"

[requires]

python_version = "3.6"

```

Pipenv 在`Pipfile.lock`文件中维护版本及其 git 哈希。所以我们不需要担心。

我们已经完成了配置开发环境，现在是时候实施 REST API 了。请继续关注下一节，我们将使用 Django Rest Framework 设计 REST API。

# 设计 REST API

我们将为我们的 ImageGallery 应用程序设计 REST API。我们使用 Django 的管理界面开发了这个应用程序。现在我们将通过 RESTful API 界面扩展 ImageGallery 应用程序的现有实现。在实施解决方案之前，让我们简要介绍一下 Django REST 框架。

# 什么是 Django Rest Framework？

Django Rest Framework 是一个开源库，旨在以乐观的方式实现 REST API。它遵循 Django 设计模式，使用不同的术语。您可以在其文档网站([`www.django-rest-framework.org/#quickstart`](http://www.django-rest-framework.org/#quickstart))找到快速入门教程。

Django Rest Framework 是强大的，支持 ORM 和非 ORM 数据源。它内置支持可浏览的 API 客户端([`restframework.herokuapp.com/`](https://restframework.herokuapp.com/))和许多其他功能。

建议在生产环境中不要使用 Web Browsable API 界面。您可以通过在`settings.py`中设置渲染类来禁用它。

```py
settings.py file.
```

文件—`settings.py`：

```py
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': (
        'rest_framework.renderers.JSONRenderer',
    )
}
```

# 集成 REST 框架

要集成 Django REST Framework，您可以简单地使用 pipenv 包装工具进行安装，就像在之前设置虚拟环境的部分中提到的那样。安装完成后，您可以继续在`INSTALLED_APPS`设置中添加`rest_framework`。看一下这段代码：

```py
INSTALLED_APPS = (
    ...
    'rest_framework',
)
```

如果您想要在登录和注销视图以及 Web 浏览 API 一起使用，那么您可以在根`urls.py`文件中添加以下 URL 模式：

```py
urlpatterns = [
    ...
    url(r'^api-auth/', include('rest_framework.urls'))
]
```

就是这样！现在我们已经成功集成了 Django REST Framework，我们可以继续创建 REST API。在创建 REST API 之前，我们需要实现身份验证和授权层，以便我们的每个 REST API 都能免受未经授权的访问。

让我们在下一节看看如何使我们的 REST API 安全。敬请关注。

# 实施身份验证和授权

身份验证和授权是设计 REST API 时必须考虑的重要部分。借助这些层，我们可以防止未经授权的访问我们的应用程序。有许多类型的实现模式可用，但我们将使用**JWT**（**JSON Web Token**）。在[`en.wikipedia.org/wiki/JSON_Web_Token`](https://en.wikipedia.org/wiki/JSON_Web_Token)上了解更多信息。JWT 对于实现分布式微服务架构非常有用，并且不依赖于集中式服务器数据库来验证令牌的真实性。

有许多 Python 库可用于实现 JWT 令牌机制。在我们的情况下，我们希望使用`django-rest-framework-jwt`库（[`getblimp.github.io/django-rest-framework-jwt/`](https://getblimp.github.io/django-rest-framework-jwt/)），因为它提供了对 Django Rest Framework 的支持。

我假设您在之前描述的*虚拟环境*部分设置环境时已经安装了这个库。让我们看看下一节应该如何配置`django-rest-framework-jwt`库。

# 配置 django-rest-framework-jwt

安装完成后，您需要在`settings.py`中添加一些与权限和身份验证相关的预定义类，如下面的代码片段所示。

文件—`settings.py`：

```py
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': (
        'rest_framework.renderers.JSONRenderer',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_jwt.authentication.JSONWebTokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ),
} 
```

现在我们需要根据用户凭据添加获取令牌的 URL。在根`urls.py`中，我们将添加以下语句：

```py
from django.urls import path
from rest_framework_jwt.views import obtain_jwt_token
#...

urlpatterns = [
    '',
    # ...

    path(r'api-token-auth/', obtain_jwt_token),
]
```

`api-token-auth` API 将在成功验证后返回一个 JWT 令牌，例如：

```py
$ curl -X POST -d "username=admin&password=password123" http://localhost:8000/api-token-auth/

{"token":"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6ImFiZHVsd2FoaWQiLCJleHAiOjE1MjYwNDUwNjgsImVtYWlsIjoiYWJkdWx3YWhpZDI0QGdtYWlsLmNvbSJ9.Iw0ZTtdZpsQqrKIkf2VKoWw91txYp9DLkBYMS9OPoCU"}
```

这个令牌可以通过添加授权标头和令牌来授权所有其他受保护的 API，如下所示：

```py
$ curl -H "Authorization: JWT <your_token>" http://localhost:8000/protected-url/
```

还有其他用例，您可能需要对已发行的令牌执行许多操作。为此，您需要阅读`django-rest-framework-jwt`的文档（[`getblimp.github.io/django-rest-framework-jwt/`](https://getblimp.github.io/django-rest-framework-jwt/)）。

现在让我们开始为我们的 ImageGallery 应用程序实现 API。

# 实施序列化器

Django Rest Framework 设计了一个类似于 Django 表单模块的序列化器模块，用于实现 JSON 表示层。序列化器负责对数据进行序列化和反序列化；您可以在这里看到有关数据序列化的详细解释（[`www.django-rest-framework.org/tutorial/1-serialization/#creating-a-serializer-class`](http://www.django-rest-framework.org/tutorial/1-serialization/#creating-a-serializer-class)）。

序列化程序模块有许多有用的类，例如`Serializer`、`ModelSerializer`、`HyperlinkedModelSerializer`等([`www.django-rest-framework.org/api-guide/serializers/`](http://www.django-rest-framework.org/api-guide/serializers/))。每个类都具有类似的操作，但具有扩展功能。`Serializer`类用于设计类似于 Django 表单表示的自定义数据表示，`ModelSerializer`用于表示与 Django 的`ModelFrom`类似的模型类数据。`HyperlinkedModelSerializer`通过超链接表示扩展了`ModelSerializer`的表示，并使用主键来关联相关数据。

我们需要创建一个使用`ModelSerializer`的序列化程序类。看一下这段代码。

文件—`gallery`/`serializers.py`：

```py
from rest_framework import serializers
from gallery.models import PhotoAlbum, Photo

class PhotoSerializer(serializers.ModelSerializer):

    class Meta:
        model = Photo
        fields = ('id', 'image', 'created_at', 'updated_at')

class PhotoAlbumSerializer(serializers.ModelSerializer):

    class Meta:
        model = PhotoAlbum
        fields = ('id', 'name', 'photos', 'created_at', 'updated_at')
        depth = 1
```

在这里，我们创建了`PhotoSerializer`和`PhotoAlbumSerializer`类，使用`ModelSerializer`类。这些序列化程序与模型类相关联；因此，数据表示将基于模型结构。

让我们继续下一节，我们将创建视图。

# 实现 viewsets

```py
Photo and PhotoAlbum models.
```

文件—`gallery`/`views.py`：

```py
from rest_framework import viewsets
from gallery.models import Photo, PhotoAlbum
from gallery.serializers import PhotoSerializer, PhotoAlbumSerializer

class PhotoViewset(viewsets.ModelViewSet):

    queryset = Photo.objects.all()
    serializer_class = PhotoSerializer

    def get_queryset(self, *args, **kwargs):
        if 'album_id' not in self.kwargs:
            raise APIException('required album_id')
        elif 'album_id' in self.kwargs and \
                not Photo.objects.filter(album__id=self.kwargs['album_id']).exists():
                                            raise NotFound('Album not found')
        return Photo.objects.filter(album__id=self.kwargs['album_id'])

    def perform_create(self, serializer):
        serializer.save(album_id=int(self.kwargs['album_id']))

class PhotoAlbumViewset(viewsets.ModelViewSet):

    queryset = PhotoAlbum.objects.all()
    serializer_class = PhotoAlbumSerializer
```

在这里，您可以看到我们已经创建了与`Photo`和`PhotoAlbum`模型相关的两个不同的 viewsets 类。`PhotoAlbum`模型与`Photo`模型有一对多的关系。因此，我们将编写一个嵌套 API，例如`albums/(?P<album_id>[0-9]+)/photos`。为了根据`album_id`返回相关的照片记录，我们重写了`get_queryset`方法，以便根据给定的`album_id`过滤`queryset`。

类似地，我们重写了`perform_create`方法，以在创建新记录时设置关联的`album_id`。我们将在即将到来的部分中提供完整的演示。

让我们看一下 URL 配置，我们在那里配置了嵌套 API 模式。

# 配置 URL 路由

Django REST Framework 提供了一个`router`模块来配置标准的 URL 配置。它自动添加了与所述 viewsets 相关的所有必需的 URL 支持。在这里阅读更多关于`routers`的信息：[`www.django-rest-framework.org/api-guide/routers/`](http://www.django-rest-framework.org/api-guide/routers/)。以下是与我们的路由配置相关的代码片段。

文件—`gallery`/`urls.py`：

```py
from django.urls import path, include
from rest_framework import routers
from gallery.views import PhotoAlbumViewset, PhotoViewset

router = routers.DefaultRouter()
router.register('albums', PhotoAlbumViewset)
router.register('albums/(?P<album_id>[0-9]+)/photos', PhotoViewset)

urlpatterns = [
    path(r'', include(router.urls)),
]
```

在这里，我们创建了一个默认路由器，并注册了带有 URL 前缀的 viewsets。路由器将自动确定 viewsets，并生成所需的 API URL。

```py
urls.py file.
```

文件—`imageGalleryProject`/`urls.py`：

```py
from django.contrib import admin
from django.urls import path, include
from rest_framework_jwt.views import obtain_jwt_token

urlpatterns = [
    path('admin/', admin.site.urls),
    path(r'', include('gallery.urls')),
    path(r'api-token-auth/', obtain_jwt_token),
]
```

一旦您包含了`gallery.urls`模式，它将在应用程序级别可用。我们已经完成了实现，现在是时候看演示了。让我们继续下一节，我们将探索 Zappa 配置，以及在 AWS Lambda 上的执行和部署过程。

# 使用 Zappa 构建、测试和部署 Django 应用程序

Django 提供了一个轻量级的部署 Web 服务器，运行在本地机器的 8000 端口上。您可以在进入生产环境之前对应用程序进行调试和测试。在这里阅读更多关于它的信息([`docs.djangoproject.com/en/2.0/ref/django-admin/#runserver`](https://docs.djangoproject.com/en/2.0/ref/django-admin/#runserver))。

让我们继续下一节，我们将探索应用程序演示和在 AWS Lambda 上的部署。

# 在本地环境中执行

```py
python manage.py runserver command:
```

```py
$ python manage.py runserver
Performing system checks...
System check identified no issues (0 silenced).

May 14, 2018 - 10:04:25
Django version 2.0.5, using settings 'imageGalleryProject.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```

现在是时候看一下您的 API 的执行情况了。我们将使用 Postman，一个 API 客户端工具，来测试 REST API。您可以从[`www.getpostman.com/`](https://www.getpostman.com/)下载 Postman 应用程序。让我们在接下来的部分中看到所有 API 的执行情况。

# API 身份验证

在访问资源 API 之前，我们需要对用户进行身份验证并获取 JWT 访问令牌。让我们使用`api-token-auth`API 来获取访问令牌。我们将使用`curl`命令行工具来执行 API。以下是`curl`命令的执行：

```py
$ curl -H "Content-Type: application/json" -X POST -d '{"username":"abdulwahid", "password":"abdul123#"}' http://localhost:8000/api-token-auth/
{"token":"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6ImFiZHVsd2FoaWQiLCJleHAiOjE1Mjk1NjYxOTgsImVtYWlsIjoiYWJkdWx3YWhpZDI0QGdtYWlsLmNvbSJ9.QypghhspJrNsp-v_XxlZeQFi_Wsujqh27EjlJtOaY_4"}
```

在这里，我们收到了 JWT 令牌作为用户身份验证的响应。现在我们将使用这个令牌作为授权标头来访问其他 API 资源。

# 在 API "/albums/"上的 GET 请求

此 API 将列出`PhotoAlbum`模型的所有记录。让我们尝试使用 cRUL 命令以`GET`请求方法访问`/album/` API，如下所示：

```py
$ curl -i http://localhost:8000/albums/ 
HTTP/1.1 401 Unauthorized
Date: Thu, 21 Jun 2018 07:33:07 GMT
Server: WSGIServer/0.2 CPython/3.6.5
Content-Type: application/json
WWW-Authenticate: JWT realm="api"
Allow: GET, POST, HEAD, OPTIONS
X-Frame-Options: SAMEORIGIN
Content-Length: 58
Vary: Cookie

{"detail":"Authentication credentials were not provided."}
```

在这里，我们从服务器收到了 401 未经授权的错误，消息是未提供身份验证凭据。这就是我们使用 JWT 令牌身份验证机制保护所有 API 的方式。

现在，如果我们只是使用从身份验证 API 获取的访问令牌添加授权标头，我们将从服务器获取记录。以下是成功的 API 访问授权标头的 cURL 执行：

```py
$ curl -i -H "Authorization: JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6ImFiZHVsd2FoaWQiLCJleHAiOjE1Mjk1NjY4NjUsImVtYWlsIjoiYWJkdWx3YWhpZDI0QGdtYWlsLmNvbSJ9.Dnbwuf3Mu2kcfk8KrbC-ql94lfHzK0z_5TgCPl5CeaM" http://localhost:8000/albums/
HTTP/1.1 200 OK
Date: Thu, 21 Jun 2018 07:40:14 GMT
Server: WSGIServer/0.2 CPython/3.6.5
Content-Type: application/json
Allow: GET, POST, HEAD, OPTIONS
X-Frame-Options: SAMEORIGIN
Content-Length: 598

[
    {
        "created_at": "2018-03-17T22:39:08.513389Z",
        "id": 1,
        "name": "Screenshot",
        "photos": [
            {
                "album": 1,
                "created_at": "2018-03-17T22:47:03.775033Z",
                "id": 5,
                "image": "https://chapter-5.s3-ap-south-1.amazonaws.com/media/Screenshot/AWS_Lambda_Home_Page.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIXNW3FK64BZR3DLA%2F20180621%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20180621T073958Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=721acd5b023e13132f606a3f72bd672bad95a0dcb24572099c4cb49cdc34df71",
                "updated_at": "2018-03-17T22:47:18.298215Z"
            }
        ],
        "updated_at": "2018-03-17T22:47:17.328637Z"
    }
]
```

正如您所看到的，我们通过提供授权标头从`"/albums/"` API 获取了数据。在这里，我们可以使用`| python -m json.tool`以 JSON 可读格式打印返回响应。

# 在 API "/albums/<album_id>/photos/"上的 POST 请求

现在我们可以向现有记录添加更多照片。以下是 cRUL 命令执行的日志片段，我们正在将图像文件上传到现有相册：

```py
$ curl -i -H "Content-Type: multipart/form-data" -H "Authorization: JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6ImFiZHVsd2FoaWQiLCJleHAiOjE1Mjk1NzE5ODEsImVtYWlsIjoiYWJkdWx3YWhpZDI0QGdtYWlsLmNvbSJ9.3CHaV4uI-4xwbzAVdBA4ooHtaCdUrVn97uR_G8MBM0I" -X POST -F "image=@/home/abdulw/Pictures/serverless.png" http://localhost:8000/albums/1/photos/ HTTP/1.1 201 Created
Date: Thu, 21 Jun 2018 09:01:44 GMT
Server: WSGIServer/0.2 CPython/3.6.5
Content-Type: application/json
Allow: GET, POST, HEAD, OPTIONS
X-Frame-Options: SAMEORIGIN
Content-Length: 450

{
    "created_at": "2018-06-21T09:02:27.918719Z",
    "id": 7,
    "image": "https://chapter-5.s3-ap-south-1.amazonaws.com/media/Screenshot/serverless.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJA3LNVLKPTEOWH5A%2F20180621%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20180621T090228Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=4e28ef5daa6e1887344514d9953f17df743e747c32b532cde12b840241fa13f0",
    "updated_at": "2018-06-21T09:02:27.918876Z"
}
```

现在，您可以看到图像已上传到 AWS S3 存储，并且我们已经配置了 AWS S3 和 CloudFront，因此我们获得了 CDN 链接。让我们再次查看所有记录的列表：

```py
$ curl -H "Authorization: JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6ImFiZHVsd2FoaWQiLCJleHAiOjE1Mjk1NzIzNTYsImVtYWlsIjoiYWJkdWx3YWhpZDI0QGdtYWlsLmNvbSJ9.m2w1THn5Nrpy0dCi8k0bPdeo67OHNYEKO-yTX5Wnuig" http://localhost:8000/albums/ | python -m json.tool

[
    {
        "created_at": "2018-03-17T22:39:08.513389Z",
        "id": 1,
        "name": "Screenshot",
        "photos": [
            {
                "album": 1,
                "created_at": "2018-03-17T22:47:03.775033Z",
                "id": 5,
                "image": "https://chapter-5.s3-ap-south-1.amazonaws.com/media/Screenshot/AWS_Lambda_Home_Page.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJA3LNVLKPTEOWH5A%2F20180621%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20180621T090753Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=832abe952870228c2ae22aaece81c05dc1414a2e9a78394d441674634a6d2bbf",
                "updated_at": "2018-03-17T22:47:18.298215Z"
            },
            {
                "album": 1,
                "created_at": "2018-06-21T09:01:44.354167Z",
                "id": 6,
                "image": "https://chapter-5.s3-ap-south-1.amazonaws.com/media/Screenshot/serverless.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJA3LNVLKPTEOWH5A%2F20180621%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20180621T090753Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=90a00ad79f141c919d8e65474325534461cf837f462cb52a840afb3863b72013",
                "updated_at": "2018-06-21T09:01:44.354397Z"
            },
            {
                "album": 1,
                "created_at": "2018-06-21T09:02:27.918719Z",
                "id": 7,
                "image": "https://chapter-5.s3-ap-south-1.amazonaws.com/media/Screenshot/serverless.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJA3LNVLKPTEOWH5A%2F20180621%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20180621T090753Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=90a00ad79f141c919d8e65474325534461cf837f462cb52a840afb3863b72013",
                "updated_at": "2018-06-21T09:02:27.918876Z"
            }
        ],
        "updated_at": "2018-03-17T22:47:17.328637Z"
    }
]

```

现在我们的应用程序已根据我们的要求实施。我们可以继续使用 Zappa 在 AWS Lambda 上部署应用程序。现在让我们转向下一节来配置 Zappa。

# 配置 Zappa

```py
zappa_settings.json file:
```

```py
{
    "dev": {
        "aws_region": "ap-south-1",
        "django_settings": "imageGalleryProject.settings",
        "profile_name": "default",
        "project_name": "imagegallerypro",
        "runtime": "python3.6",
        "s3_bucket": "chapter-5",
        "remote_env": "s3://important-credentials-bucket/environments.json"
    }
}
```

在这里，我们根据要求定义了配置。由于密钥定义了每个配置，我们可以看到它的用法：

+   `aws_region`：Lambda 将上传的 AWS 区域。

+   `django_settings`：Django 设置文件的导入路径。

+   `profile_name`：在`~/.aws/credentials`文件中定义的 AWS CLI 配置文件。

+   `project_name`：上传 Lambda 函数的项目名称。

+   `runtime`：Python 运行时解释器。

+   `s3_bucket`：创建 Amazon S3 存储桶并上传部署包。

+   `remote_env`：设置 Amazon S3 位置上传的 JSON 文件中提到的所有键值对的环境变量。

借助这些配置信息，我们将继续部署。

# 构建和部署

一旦我们完成配置，就可以进行部署。Zappa 提供了两个不同的命令来执行部署，例如`zappa deploy <stage_name>`和`zappa update <stage_name>`。最初，我们将使用`zappa deploy <stage_name>`命令，因为这是我们第一次部署此 Lambda 应用程序。

如果您已经部署了应用程序并希望重新部署，那么您将使用`zappa update <stage_name>`命令。在上一章中，我们详细讨论了 Zappa 的部署过程，因此您可以参考它。

以下是我们部署过程的日志片段：

```py
$ zappa update dev
(python-dateutil 2.7.3 (/home/abdulw/.local/share/virtualenvs/imageGalleryProject-4c9zDR_T/lib/python3.6/site-packages), Requirement.parse('python-dateutil==2.6.1'), {'zappa'})
Calling update for stage dev..
Downloading and installing dependencies..
 - pillow==5.1.0: Downloading
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.95M/1.95M [00:00<00:00, 7.73MB/s]
 - sqlite==python36: Using precompiled lambda package
Packaging project as zip.
Uploading imagegallerypro-dev-1529573380.zip (20.2MiB)..
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21.2M/21.2M [00:06<00:00, 2.14MB/s]
Updating Lambda function code..
Updating Lambda function configuration..
Uploading imagegallerypro-dev-template-1529573545.json (1.6KiB)..
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.65K/1.65K [00:00<00:00, 28.9KB/s]
Deploying API Gateway..
Scheduling..
Unscheduled imagegallerypro-dev-zappa-keep-warm-handler.keep_warm_callback.
Scheduled imagegallerypro-dev-zappa-keep-warm-handler.keep_warm_callback with expression rate(4 minutes)!
Your updated Zappa deployment is live!: https://cfsla2gds0.execute-api.ap-south-1.amazonaws.com/dev
https://cfsla2gds0.execute-api.ap-south-1.amazonaws.com/dev.
```

让我们转到下一节，我们将在部署的应用程序上执行一些操作。

# 在生产环境中执行

一旦您成功部署了应用程序，您将获得托管应用程序链接。这个链接就是通过将 AWS API 网关与 Zappa 的 AWS Lambda 配置生成的链接。

现在您可以在生产环境中使用应用程序。身份验证 API 的屏幕截图在下一节中。

# 身份验证 API

正如我们在本地环境中看到的身份验证执行一样，在生产环境中也是一样的。以下是部署在 AWS Lambda 上的身份验证 API 执行的日志片段：

```py
$ curl -H "Content-Type: application/json" -X POST -d '{"username":"abdulwahid", "password":"abdul123#"}' https://cfsla2gds0.execute-api.ap-south-1.amazonaws.com/dev/api-token-auth/
{"token":"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6ImFiZHVsd2FoaWQiLCJleHAiOjE1Mjk1NzQyOTMsImVtYWlsIjoiYWJkdWx3YWhpZDI0QGdtYWlsLmNvbSJ9.pHuHaJpjlESwdQxXMiqGOuy2_lpVW1X26RiB9NN8rhI"}
```

正如您在这里所看到的，功能不会对任何事物产生影响，因为应用程序正在无服务器环境中运行。让我们看看另一个 API。

# 对“/albums/”API 的 GET 请求

通过身份验证 API 获得的访问令牌，您有资格访问所有受保护的 API。以下是`/albums/`API 的`GET`请求的屏幕截图：

```py
$ curl -H "Authorization: JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6ImFiZHVsd2FoaWQiLCJleHAiOjE1Mjk1NzQ4MzgsImVtYWlsIjoiYWJkdWx3YWhpZDI0QGdtYWlsLmNvbSJ9.55NucqsavdgxcmNNs6_hbJMCw42mWPyylaVvuiP5KwI" https://cfsla2gds0.execute-api.ap-south-1.amazonaws.com/dev/albums/ | python -m json.tool

[
    {
        "created_at": "2018-03-17T22:39:08.513389Z",
        "id": 1,
        "name": "Screenshot",
        "photos": [
            {
                "album": 1,
                "created_at": "2018-03-17T22:47:03.775033Z",
                "id": 5,
                "image": "https://chapter-5.s3-ap-south-1.amazonaws.com/media/Screenshot/AWS_Lambda_Home_Page.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJA3LNVLKPTEOWH5A%2F20180621%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20180621T094957Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=0377bc8750b115b6bff2cd5acc024c6375f5fedc6de35275ea1392375041adc0",
                "updated_at": "2018-03-17T22:47:18.298215Z"
            }
        ],
        "updated_at": "2018-03-17T22:47:17.328637Z"
    }
]
```

就是这样。我们已经完成了无服务器环境的部署。希望对您有所帮助。

# 总结

在本章中，我们学习了如何在 Django REST 框架中开发 REST API。我们介绍了使用 JWT 身份验证机制保护 API 的过程。最后，我们使用 Zappa 在无服务器环境中部署了应用程序。

在下一章中，我们将使用非常轻量级的 Python 框架开发基于高性能 API 的应用程序。我们还将探索更多 Zappa 配置选项，以建立缓存机制。敬请关注，发现 Zappa 世界中更多的宝藏。

# 问题

1.  什么是 Django Rest 框架？

1.  Django-storage 有什么用？
