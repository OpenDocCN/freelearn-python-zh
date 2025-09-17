# 使用 Django 改进我们的 API 并为其添加认证

在本章中，我们将使用我们在上一章中开始使用的 PostgreSQL 10.5 数据库来改进 Django RESTful API。我们将使用 Django REST 框架中包含的许多功能来向 API 添加新功能，并将添加与认证相关的安全功能。我们将执行以下操作：

+   在模型中添加唯一约束

+   使用`PATCH`方法更新资源的单个字段

+   利用分页功能

+   自定义分页类

+   理解认证、权限和限制

+   向模型添加与安全相关的数据

+   为对象级权限创建一个自定义权限类

+   持久化发起请求的用户并配置权限策略

+   在迁移中为新的必填字段设置默认值

+   使用必要的认证来组合请求

+   使用认证凭据浏览 API

# 在模型中添加唯一约束

我们的 API 有一些重要的问题需要我们迅速解决。目前，我们可以创建具有相同描述的许多 ESRB 评级。我们不应该能够这样做，因此，我们将对`EsrbRating`模型进行必要的更改，以在`description`字段上添加唯一约束。我们还将为`Game`和`Player`模型的`name`字段添加唯一约束。这样，我们将学习必要的步骤来更改多个模型的约束，并通过迁移反映底层数据库模式的变化。

确保您退出 Django 开发服务器。请记住，您只需在运行它的终端或命令提示符窗口中按*Ctrl* + *C*即可。 ... 

# 使用 PATCH 方法更新资源的单个字段

由于使用了基于类的通用视图，我们的 API 能够更新现有资源的单个字段，因此，我们为`PATCH`方法提供了一个实现。例如，我们可以使用`PATCH`方法来更新一个现有的游戏，并将它的`played_once`和`played_times`字段的值设置为`True`和`1`。我们不希望使用`PUT`方法，因为这个方法旨在替换整个游戏。请记住，`PATCH`方法旨在对现有游戏应用一个增量，因此，它是仅更改`played_once`和`played_times`字段值的适当方法。

现在，我们将组合并发送一个 HTTP `PATCH`请求来更新一个现有的游戏，特别是更新`played_once`和`played_times`字段的值，并将它们设置为`True`和`10`。确保将`2`替换为配置中现有游戏的`id`。示例的代码文件包含在`restful_python_2_07_01`文件夹中，在`Django01/cmd/cmd703.txt`文件中：

```py
    http PATCH ":8000/games/2/" played_once=true played_times=10  
```

以下是对应的`curl`命令。示例的代码文件包含在`restful_python_2_07_01`文件夹中，在`Django01/cmd/cmd704.txt`文件中：

```py
 curl -iX PATCH -H "Content-Type: application/json" -d '{"played_once":"true", "played_times": 10}' "localhost:8000/games/2/"

```

前面的命令将组合并发送一个包含指定 JSON 键值对的 HTTP `PATCH`请求。请求在`/games/`之后有一个数字，因此它将匹配`'^games/(?P<pk>[0-9]+)/$'`并运行`views.GameDetail`基于类的视图的`patch`方法。请记住，`patch`方法是在`RetrieveUpdateDestroyAPIView`超类中定义的，并最终调用在`mixins.UpdateModelMixin`中定义的`update`方法。如果更新`played_once`和`played_times`字段值的`Game`实例有效，并且它已成功持久化到数据库中，则对方法的调用将返回`200 OK`状态码，并将最近更新的`Game`序列化为 JSON 格式放在响应体中。

以下行显示了示例响应：

```py
    HTTP/1.1 200 OK
    Allow: GET, PUT, PATCH, DELETE, HEAD, OPTIONS
    Content-Length: 204
    Content-Type: application/json
    Date: Fri, 26 Oct 2018 16:40:51 GMT
    Server: WSGIServer/0.2 CPython/3.7.1
    Vary: Accept, Cookie
    X-Frame-Options: SAMEORIGIN

    {
        "esrb_rating": "AO (Adults Only)",
        "name": "Mutant Football League: Dynasty Edition",
        "played_once": true,
        "played_times": 10,
        "release_date": "2018-10-20T03:02:00.776594Z",
        "url": "http://localhost:8000/games/2/"
    }

```

# 利用分页

我们的数据库为每个持久化我们定义的模型的表都有几行。然而，在我们开始在现实生产环境中使用我们的 API 之后，我们将有数千个玩家得分、玩家和游戏——尽管 ESRB 评级仍然数量很少。我们绝对必须准备我们的 API 以处理大量结果集。幸运的是，我们可以利用 Django REST framework 中可用的分页功能，使其容易指定我们希望如何将大量结果集拆分为单个数据页。

首先，我们将编写命令来组合和发送 HTTP `POST`请求以创建 10 个属于我们创建的 ESRB 评级之一（`T (Teen)`）的游戏。这样，...

# 自定义分页类

我们使用的`rest_framework.pagination.LimitOffsetPagination`类声明了一个`max_limit`类属性，默认值为`None`。此属性允许我们指定可以使用`limit`查询参数指定的最大允许限制。默认设置下，没有限制，我们将能够处理指定`limit`查询参数值为`1000000`的请求。

我们绝对不希望我们的 API 能够通过单个请求生成包含一百万个玩家得分或单个玩家的响应。不幸的是，没有配置设置允许我们更改类分配给`max_limit`类属性的值。因此，我们被迫创建 Django REST Framework 提供的`limit`/`offset`分页风格的定制版本。

在`games_service/games`文件夹内创建一个名为`max_limit_pagination.py`的新 Python 文件，并输入以下代码，该代码声明了新的`MaxLimitPagination`类。示例的代码文件包含在`restful_python_2_07_03`文件夹中，位于`Django01/games-service/games/max_limit_pagination.py`文件中：

```py
from rest_framework.pagination import LimitOffsetPagination 

class MaxLimitPagination(LimitOffsetPagination): 
    max_limit = 8 
```

前面的行将`MaxLimitPagination`类声明为`rest_framework.pagination.LimitOffsetPagination`超类的子类，并覆盖了为`max_limit`类属性指定的值，将其设置为`8`。

在`games_service/games_service`文件夹中打开`settings.py`文件，并将指定`REST_FRAMEWORK`字典中`DEFAULT_PAGINATION_CLASS`键值的行替换为高亮行。以下行显示了名为`REST_FRAMEWORK`的新字典声明。示例的代码文件包含在`restful_python_2_07_03`文件夹中，在`Django01/games-service/games/settings.py`文件中：

```py
REST_FRAMEWORK = { 
    'DEFAULT_PAGINATION_CLASS': 
 'games.max_limit_pagination.MaxLimitPagination',    'PAGE_SIZE': 4 
} 
```

现在通用视图将使用最近声明的`games.pagination.MaxLimitPagination`类，该类提供了一个基于`limit`/`offset`的样式，最大`limit`值等于`8`。如果一个请求指定了一个大于`8`的`limit`值，该类将使用最大限制值，即`8`，并且我们永远不会在一个分页响应中返回超过`8`个条目。

现在，我们将编写一个命令来组成并发送一个 HTTP 请求以检索游戏的第一个页面，具体来说，是一个将`limit`值设置为`20`的`/games/`的 HTTP `GET`方法。示例的代码文件包含在`restful_python_2_07_03`文件夹中，在`Django01/cmd/cmd719.txt`文件中：

```py
    http GET ":8000/games/?limit=20"
```

以下是对应的`curl`命令。示例的代码文件包含在`restful_python_2_07_03`文件夹中，在`Django01/cmd/cmd720.txt`文件中：

```py
    curl -iX GET "localhost:8000/games/?limit=20"
```

结果将使用一个等于`8`的极限值，而不是指示的`20`，因为我们正在使用我们的自定义分页类。结果将在`results`键中提供包含 10 个游戏资源的第一个集合，在`count`键中提供查询的总游戏数，并在`next`和`previous`键中提供下一页和上一页的链接。在这种情况下，结果集是第一页，因此，`next`键中下一页的链接是`http://localhost:8000/games/?limit=8&offset=8`。我们将在响应头中收到`200 OK`状态码，并在`results`数组中收到前八个游戏。以下行显示了头信息和输出第一行：

```py
    HTTP/1.1 200 OK
    Allow: GET, POST, HEAD, OPTIONS
    Content-Length: 1542
    Content-Type: application/json
    Date: Fri, 26 Oct 2018 21:25:06 GMT
    Server: WSGIServer/0.2 CPython/3.7.1
    Vary: Accept, Cookie
    X-Frame-Options: SAMEORIGIN

    {
        "count": 12,
        "next": "http://localhost:8000/games/?limit=8&offset=8",
        "previous": null,
        "results": 
            {

```

配置最大限制以避免生成巨大的响应是一个好习惯。

打开一个网页浏览器并输入`http://localhost:8000/games/`。如果你使用另一台计算机或设备运行浏览器，请将`localhost`替换为运行 Django 开发服务器的计算机的 IP 地址。可浏览 API 将组成并发送一个到`/games/`的 HTTP `GET`请求，并将显示其执行结果，即头信息和 JSON 游戏列表。因为我们已经配置了分页，所以渲染的网页将包括与我们使用的基分页类关联的默认分页模板，并在网页右上角显示可用的页码。

以下截图显示了在网页浏览器中输入 URL 后渲染的网页，包括资源描述游戏列表和三个页面：

![图片

# 理解身份验证、权限和节流

我们当前的 API 版本处理所有传入请求，无需任何类型的身份验证。Django REST 框架允许我们轻松使用不同的身份验证方案来识别发起请求的用户或签名请求的令牌。然后，我们可以使用这些凭据来应用权限和速率限制策略，以确定请求是否必须被允许。在生产环境中，我们可以将身份验证方案与运行在 HTTPS 下的 API 结合使用。在我们的开发配置中，我们将继续在 HTTP 下使用 API，但这仅适用于开发。

如同其他配置发生的情况一样，...

# 将安全相关数据添加到模型中

我们将把一个游戏与创建者或所有者关联起来。只有经过身份验证的用户才能创建新的游戏。只有游戏的创建者才能更新或删除它。未经身份验证的所有请求将只能对游戏有只读访问权限。

打开`games_service/games`文件夹中的`models.py`文件。将声明`Game`类的代码替换为以下代码。代码列表中的新行和编辑行被突出显示。示例的代码文件包含在`restful_python_2_07_04`文件夹中的`Django01/games-service/games/models.py`文件中：

```py
class Game(models.Model): 
    created = models.DateTimeField(auto_now_add=True) 
    name = models.CharField(max_length=200, unique=True) 
    esrb_rating = models.ForeignKey( 
        EsrbRating,  
        related_name='games',  
        on_delete=models.CASCADE) 
    release_date = models.DateTimeField() 
    played_once = models.BooleanField(default=False) 
    played_times = models.IntegerField(default=0) 
    owner = models.ForeignKey( 
        'auth.User',  
        related_name='games', 
        on_delete=models.CASCADE) 

    class Meta: 
        ordering = ('name',) 

    def __str__(self): 
        return self.name
```

新版本的`Game`模型声明了一个新的`owner`字段，该字段使用`django.db.models.ForeignKey`类提供与`auth.User`模型的许多对一关系，具体来说，是与`django.contrib.auth.User`模型。这个`User`模型代表 Django 身份验证系统中的用户。为`related_name`参数指定的`'games'`值创建了一个从`User`模型到`Game`模型的反向关系。这个值表示用于将`User`对象关联回`Game`对象的名称。这样，我们将能够访问特定用户拥有的所有游戏。每次我们删除一个用户时，我们希望删除该用户拥有的所有游戏，因此，我们为`on_delete`参数指定了`models.CASCADE`值。

现在，我们将运行`manage.py`中的`createsuperuser`子命令来创建 Django 的超级用户，我们将使用它来轻松地验证我们的请求。我们稍后会创建更多用户：

```py
    python manage.py createsuperuser
```

命令将要求你输入想要用于超级用户的用户名。输入所需的用户名并按*Enter*键。在这个例子中，我们将使用`your_games_super_user`作为用户名。你将看到类似以下的一行：

```py
    Username (leave blank to use 'xxxxxxxx'):
```

然后，命令将要求你输入电子邮件地址：

```py
    Email address: 
```

输入一个电子邮件地址，例如`your_games_super_user@example.com`，并按*Enter*键。

最后，命令将要求你输入新超级用户的密码：

```py
    Password:
```

输入你想要的密码并按*Enter*键。在示例中，我们将使用`WCS3qn!a4ybX#`作为密码。

命令将要求你再次输入密码：

```py
    Password (again):
```

输入并按 *Enter*。如果输入的两个密码匹配，将创建超级用户：

```py
    Superuser created successfully.
```

打开 `games_service/games` 文件夹中的 `serializers.py` 文件。在声明导入的最后一行之后，在 `GameCategorySerializer` 类声明之前添加以下代码。示例的代码文件包含在 `restful_python_2_07_04` 文件夹中，在 `Django01/games-service/games/serializers.py` 文件中：

```py
from django.contrib.auth.models import User 

class UserGameSerializer(serializers.HyperlinkedModelSerializer): 
    class Meta: 
        model = Game 
        fields = ( 
            'url', 
            'name') 

class UserSerializer(serializers.HyperlinkedModelSerializer): 
    games = UserGameSerializer(many=True, read_only=True) 

    class Meta: 
        model = User 
        fields = ( 
            'url',  
            'id', 
            'username', 
            'games') 
```

`UserGameSerializer` 类是 `HyperlinkedModelSerializer` 超类的子类。我们使用这个新的序列化器类来序列化与用户相关的游戏。我们只想包含 URL 和游戏名称，因此，代码指定了 `'url'` 和 `'name'` 作为在 `Meta` 内部类中定义的字段元组的成员。我们不希望使用 `GameSerializer` 序列化器类来序列化与用户相关的游戏，因为我们想序列化更少的字段，因此，我们创建了 `UserGameSerializer` 类。

`UserSerializer` 类是 `HyperlinkedModelSerializer` 超类的子类。这个序列化器类与 `django.contrib.auth.models.User` 模型相关。`UserSerializer` 类声明了一个 `games` 属性，它是一个之前解释过的 `UserGameSerializer` 的实例，其中 `many` 和 `read_only` 都设置为 `True`，因为它是一个一对多关系，并且是只读的。我们使用 `games` 名称，我们在将 `owner` 字段作为 `models.ForeignKey` 实例添加到 `Game` 模型时指定的 `related_name` 字符串值。这样，`games` 字段将为我们提供每个属于用户的游戏的 URL 和名称数组。

我们将在 `game`s_service/games 文件夹中的 `serializers.py` 文件进行更多修改。我们将向现有的 `GameSerializer` 类添加一个 `owner` 字段。以下行显示了 `GameSerializer` 类的新代码。新和编辑的行被突出显示。示例的代码文件包含在 `restful_python_2_07_04` 文件夹中，在 `Django01/games-service/games/serializers.py` 文件中：

```py
class GameSerializer(serializers.HyperlinkedModelSerializer): 
    # We want to display the game ESRB rating description instead of 
    #

its id 
    esrb_rating = serializers.SlugRelatedField( 
        queryset=EsrbRating.objects.all(),  
        slug_field='description') 
    # We want to display the user name that is the owner 
    owner = serializers.ReadOnlyField(source='owner.username') 

    class Meta: 
        model = Game 
        fields = ( 
            'url', 
            'esrb_rating', 
            'name', 
            'release_date', 
            'played_once', 
            'played_times', 
            'owner') 
```

现在，`GameSerializer` 类声明了一个 `owner` 属性，它是一个 `serializers.ReadOnlyField` 类的实例，其中 `source` 等于 `'owner.username'`。这样，我们将序列化相关 `django.contrib.auth.User` 中 `owner` 字段持有的 `username` 字段的值。我们使用 `ReadOnlyField` 类，因为当认证用户创建游戏时，所有者会自动填充，因此，在游戏创建后不可能更改所有者。这样，`owner` 字段将为我们提供创建游戏的用户名。此外，我们还向在 `Meta` 内部类中声明的 `fields` 字符串元组中添加了 `'owner'`。

# 创建一个用于对象级权限的自定义权限类

在`games_service/games`文件夹内创建一个名为`customized_permissions.py`的新 Python 文件，并输入以下声明新`IsOwnerOrReadOnly`类的代码。示例代码文件包含在`restful_python_2_07_04`文件夹中，位于`Django01/games-service/games/customized_permissions.py`文件中：

```py
from rest_framework import permissions 

class IsOwnerOrReadOnly(permissions.BasePermission): 
    def has_object_permission(self, request, view, obj): 
        if request.method in permissions.SAFE_METHODS: 
            return True 
        else: 
            return obj.owner == request.user 
```

`rest_framework.permissions.BasePermission`类是所有权限类应该继承的基础类。...

# 持久化发起请求的用户并配置权限策略

我们希望能够列出所有用户并检索单个用户的详细信息。我们将创建`rest_framework.generics`模块中声明的两个以下通用类视图的子类：

+   `ListAPIView`：实现了`get`方法，用于检索`queryset`的列表

+   `RetrieveAPIView`：实现了`get`方法以检索模型实例

在`games_service/games`文件夹中打开`views.py`文件。在声明导入的最后一行之后，在`GameCategoryList`类声明之前添加以下代码。示例代码文件包含在`restful_python_2_07_04`文件夹中，位于`Django01/games-service/games/views.py`文件中：

```py
from django.contrib.auth.models import User 
from rest_framework import permissions 
from games.serializers import UserSerializer 
from games.customized_permissions import IsOwnerOrReadOnly 

class UserList(generics.ListAPIView): 
    queryset = User.objects.all() 
    serializer_class = UserSerializer 
    name = 'user-list' 

class UserDetail(generics.RetrieveAPIView): 
    queryset = User.objects.all() 
    serializer_class = UserSerializer 
    name = 'user-detail'
```

继续编辑`games_service/games`文件夹中的`views.py`文件。将以下高亮显示的行添加到`views.py`文件中声明的`ApiRoot`类中。这样，我们就能通过可浏览 API 导航到与用户相关的视图。示例代码文件包含在`restful_python_2_07_04`文件夹中，位于`Django01/games-service/games/views.py`文件中：

```py
class ApiRoot(generics.GenericAPIView): 
    name = 'api-root' 
    def get(self, request, *args, **kwargs): 
        return Response({ 
 'users': reverse(UserList.name, request=request),            'players': reverse(PlayerList.name, request=request), 
            'esrb-ratings': reverse(EsrbRatingList.name, request=request), 
            'games': reverse(GameList.name, request=request), 
            'scores': reverse(PlayerScoreList.name, request=request) 
            }) 
```

继续编辑`games_service/games`文件夹中的`views.py`文件。将以下高亮显示的行添加到`GameList`类视图以覆盖从`rest_framework.mixins.CreateModelMixin`超类继承的`perform_create`方法。记住，`generics.ListCreateAPIView`类继承自`CreateModelMixin`类和其他类。新方法中的代码将在将新的`Game`实例持久化到数据库之前填充`owner`。此外，新代码覆盖了`permission_classes`类属性的值，以配置基于类的视图的权限策略。示例代码文件包含在`restful_python_2_07_04`文件夹中，位于`Django01/games-service/games/views.py`文件中：

```py
class GameList(generics.ListCreateAPIView): 
    queryset = Game.objects.all() 
    serializer_class = GameSerializer 
    name = 'game-list' 
    permission_classes = ( 
        permissions.IsAuthenticatedOrReadOnly, 
        IsOwnerOrReadOnly) 

    def perform_create(self, serializer): 
        serializer.save(owner=self.request.user)
```

覆盖的`perform_create`方法的代码通过为`serializer.save`方法的调用设置`owner`参数的值，将额外的`owner`字段传递给`create`方法。代码将`owner`属性设置为`self.request.user`的值，即与请求关联的用户。这样，每次持久化新的游戏时，它都会将请求关联的用户保存为其所有者。

在`games_service/games`文件夹中的`views.py`文件中继续编辑。将以下高亮行添加到`GameDetail`类视图以覆盖`permission_classes`类属性的值，以配置基于类的视图的权限策略。示例代码文件包含在`restful_python_2_07_04`文件夹中的`Django01/games-service/games/views.py`文件中：

```py
class GameDetail(generics.RetrieveUpdateDestroyAPIView): 
    queryset = Game.objects.all() 
    serializer_class = GameSerializer 
    name = 'game-detail' 
    permission_classes = ( 
        permissions.IsAuthenticatedOrReadOnly, 
        IsOwnerOrReadOnly) 
```

我们在`permission_classes`元组中为`GameList`和`GameDetail`类都包含了`IsAuthenticatedOrReadOnly`类和之前创建的`IsOwnerOrReadOnly`权限类。

打开`games_service/games`文件夹中的`urls.py`文件。将以下元素添加到`urlpatterns`字符串列表中。新字符串定义了指定请求中必须匹配的正则表达式的 URL 模式，以在`views.py`文件中运行之前创建的基于类的视图的特定方法：`UserList`和`UserDetail`。示例代码文件包含在`restful_python_2_07_04`文件夹中的`Django01/games-service/games/serializers.py`文件中：

```py
    url(r'^users/$', 
        views.UserList.as_view(), 
        name=views.UserList.name), 
    url(r'^users/(?P<pk>[0-9]+)/$', 
        views.UserDetail.as_view(), 
        name=views.UserDetail.name),
```

现在打开`games_service`文件夹中的`urls.py`文件，特别是`games_service/urls.py`文件。该文件定义了根 URL 配置，我们希望包含 URL 模式以允许可浏览 API 显示登录和注销视图。以下行显示了添加了高亮的新代码。示例代码文件包含在`restful_python_2_07_04`文件夹中的`Django01/games-service/games/serializers.py`文件中：

```py
from django.conf.urls import url, include 

urlpatterns = [ 
    url(r'^', include('games.urls')), 
 url(r'^api-auth/', include('rest_framework.urls')), ] 
```

新增行添加了在`rest_framework.urls`模块中定义的 URL 模式，并将它们关联到`^api-auth/`模式。可浏览 API 使用`api-auth/`作为所有与用户登录和注销相关的视图的前缀

# 在迁移中为新的必填字段设置默认值

我们在我们的数据库中持续了很多游戏，并为那些是必填字段的游戏添加了一个新的`owner`字段。我们不希望删除所有现有的游戏，因此，我们将利用 Django 的一些特性，这些特性使我们能够轻松地在底层数据库中做出更改，而不会丢失现有数据。

现在我们需要检索我们创建的超级用户的`id`，以便将其用作现有游戏的默认所有者。Django 将允许我们轻松地更新现有游戏，为它们设置所有者用户。

运行以下命令以从`auth_user`表中检索与`username`匹配`'superuser'`的行的`id`。替换`your_games_super_user ...`

# 使用必要的认证来组合请求

现在，我们将编写一个命令来组合并发送一个不需要认证凭据的 HTTP `POST`请求以创建一个新的游戏。示例代码文件包含在`restful_python_2_07_04`文件夹中的`Django01/cmd/cmd721.txt`文件中：

```py
http POST ":8000/games/" name='Super Mario Odyssey' esrb_rating='T (Teen)' release_date='2017-10-27T01:00:00.776594Z'
```

以下是对应的 `curl` 命令。示例代码文件包含在 `restful_python_2_07_04` 文件夹中的 `Django01/cmd/cmd722.txt` 文件中：

```py
curl -iX POST -H "Content-Type: application/json" -d '{"name":"Super Mario Odyssey", "esrb_rating":"T (Teen)", "release_date": "2017-10-27T01:00:00.776594Z"}' 
"localhost:8000/games/"
```

我们将在响应头中收到一个 `403 Forbidden` 状态码，并在 JSON 体的详细消息中指出我们没有提供认证凭据。以下是一些示例响应行：

```py
    HTTP/1.1 403 Forbidden
    Allow: GET, POST, HEAD, OPTIONS
    Content-Length: 58
    Content-Type: application/json
    Date: Sat, 27 Oct 2018 15:03:53 GMT
    Server: WSGIServer/0.2 CPython/3.7.1
    Vary: Accept, Cookie
    X-Frame-Options: SAMEORIGIN

    {
        "detail": "Authentication credentials were not provided."
    }

```

如果我们想要创建一个新的游戏，即向 `/games/` 发送一个 `POST` 请求，我们需要通过使用 HTTP 认证来提供认证凭据。现在我们将编写并发送一个带有认证凭据的 HTTP 请求来创建一个新的游戏，即使用超级用户名称和他们的密码。请记住将 `your_games_super_user` 替换为你为超级用户使用的名称，将 `WCS3qn!a4ybX#` 替换为你为该用户配置的密码。示例代码文件包含在 `restful_python_2_07_04` 文件夹中的 `Django01/cmd/cmd723.txt` 文件中：

```py
http -a your_games_super_user:'WCS3qn!a4ybX#' POST ":8000/games/" name='Super Mario Odyssey' esrb_rating='T (Teen)' release_date='2017-10-27T01:00:00.776594Z'

```

以下是对应的 `curl` 命令。示例代码文件包含在 `restful_python_2_07_04` 文件夹中的 `Django01/cmd/cmd724.txt` 文件中：

```py
curl --user your_games_super_user:'password' -iX POST -H "Content-Type: application/json" -d '{"name":"Super Mario Odyssey", "esrb_rating":"T (Teen)", "release_date": "2017-10-27T01:00:00.776594Z"}' "localhost:8000/games/"  
```

如果以 `your_games_super_user` 命名的用户作为其所有者的新 `Game` 在数据库中成功持久化，则函数将返回一个 HTTP `201 Created` 状态码，并在响应体中将最近持久化的 `Game` 序列化为 JSON。以下是一些示例响应行，其中包含 JSON 响应中的新 `Game` 对象：

```py
    HTTP/1.1 201 Created
    Allow: GET, POST, HEAD, OPTIONS
    Content-Length: 209
    Content-Type: application/json
    Date: Sat, 27 Oct 2018 15:17:40 GMT
    Location: http://localhost:8000/games/13/
    Server: WSGIServer/0.2 CPython/3.7.1
    Vary: Accept, Cookie
    X-Frame-Options: SAMEORIGIN

    {
        "esrb_rating": "T (Teen)",
        "name": "Super Mario Odyssey",
        "owner": "your_games_super_user",
        "played_once": false,
        "played_times": 0,
        "release_date": "2017-10-27T01:00:00.776594Z",
        "url": "http://localhost:8000/games/13/"
    }

```

现在我们将使用认证凭据来编写并发送一个 HTTP `PATCH` 请求，以更新之前创建的游戏的 `played_once` 和 `played_times` 字段值。然而，在这种情况下，我们将使用在 Django 中创建的另一个用户来认证请求。请记住将 `gaston-hillar` 替换为你为用户使用的名称，将 `FG$gI⁷⁶q#yA3v` 替换为他们的密码。此外，将 `13` 替换为你配置中为之前创建的游戏生成的 `id`。示例代码文件包含在 `restful_python_2_07_04` 文件夹中的 `Django01/cmd/cmd725.txt` 文件中：

```py
http -a 'gaston-hillar':'FG$gI⁷⁶q#yA3v' PATCH ":8000/games/13/" played_once=true played_times=15

```

以下是对应的 `curl` 命令。示例代码文件包含在 `restful_python_2_07_04` 文件夹中的 `Django01/cmd/cmd726.txt` 文件中：

```py
curl --user 'gaston-hillar':'FG$gI⁷⁶q#yA3v' -iX PATCH -H "Content-Type: application/json" -d '{"played_once": "true", "played_times": 15}' 
"localhost:8000/games/13/"

```

我们将在响应头中收到一个 `403 Forbidden` 状态码，并在 JSON 体的详细消息中指出我们没有权限执行该操作。我们想要更新的游戏的拥有者是 `your_games_super_user`，而此请求的认证凭据使用了一个不同的用户。因此，操作被 `IsOwnerOrReadOnly` 类中的 `has_object_permission` 方法拒绝。以下是一些示例响应行：

```py
    HTTP/1.1 403 Forbidden
    Allow: GET, PUT, PATCH, DELETE, HEAD, OPTIONS
    Content-Length: 63
    Content-Type: application/json
    Date: Sat, 27 Oct 2018 15:23:45 GMT
    Server: WSGIServer/0.2 CPython/3.7.1
    Vary: Accept, Cookie
    X-Frame-Options: SAMEORIGIN

    {
        "detail": "You do not have permission to perform this action."
    }

```

如果我们使用相同的身份验证凭据，通过 `GET` 方法发送一个 HTTP 请求来获取该资源，我们就能检索到指定用户不拥有的游戏。请求将成功，因为 `GET` 是安全方法之一，并且非所有者用户被允许读取游戏。请记住将 `gaston-hillar` 替换为你为用户使用的名称，将 `FG$gI⁷⁶q#yA3v` 替换为他们的密码。此外，将 `13` 替换为你配置中为之前创建的游戏生成的 ID。示例代码文件包含在 `restful_python_2_07_04` 文件夹中的 `Django01/cmd/cmd727.txt` 文件中：

```py
    http -a 'gaston-hillar':'FG$gI⁷⁶q#yA3v' GET ":8000/games/13/"
```

以下是对应的 `curl` 命令。示例代码文件包含在 `restful_python_2_07_04` 文件夹中的 `Django01/cmd/cmd728.txt` 文件中：

```py
 curl --user 'gaston-hillar':'FG$gI⁷⁶q#yA3v' -iX GET "localhost:8000/games/13/"

```

# 使用身份验证凭据浏览 API

打开一个网络浏览器并输入 `http://localhost:8000/`。如果你使用另一台计算机或设备运行浏览器，请将 `localhost` 替换为运行 Django 开发服务器的计算机的 IP 地址。可浏览 API 将组成并发送一个 `GET` 请求到 `/`，并显示其执行的结果，即 API 根。你将注意到右上角有一个“登录”超链接。

点击“登录”，浏览器将显示 Django REST 框架的登录页面。在用户名字段中输入 `gaston-hillar`，在密码字段中输入 `FG$gI⁷⁶q#yA3v`，然后点击“登录”。现在，你将作为 `gaston-hillar` 登录，并且你将通过可浏览 API 组成和发送的所有请求 ...

# 测试你的知识

让我们看看你是否能正确回答以下问题：

1.  以下哪一行定义了一个名为 `title` 的字段，该字段将在模型中生成一个唯一约束？

    1.  `title = django.db.models.CharField(max_length=250, unique=True)`

    1.  `title = django.db.models.UniqueCharField(max_length=250)`

    1.  `title = django.db.models.CharField(max_length=250, options=django.db.models.unique_constraint)`

1.  以下哪一行定义了一个名为 `title` 的字段，该字段在模型中不会生成唯一约束？

    1.  `title = django.db.models.CharField(max_length=250, unique=False)`

    1.  `title = django.db.models.NonUniqueCharField(max_length=250)`

    1.  `title = django.db.models.CharField(max_length=250, options=django.db.models.allow_duplicates)`

1.  以下 `REST_FRAMEWORK` 字典中哪个设置的键指定了一个全局设置，该设置将使用默认的分页类为通用视图提供分页响应？

    1.  `DEFAULT_PAGINATED_RESPONSE_PARSER`

    1.  `DEFAULT_PAGINATION_CLASS`

    1.  `DEFAULT_PAGINATED_RESPONSE_CLASS`

1.  以下哪个分页类在 Django REST 框架中提供了基于限制/偏移量的样式？

    1.  `rest_framework.pagination.LimitOffsetPaging`

    1.  `rest_framework.styles.LimitOffsetPagination`

    1.  `rest_framework.pagination.LimitOffsetPagination`

1.  `rest_framework.authentication.BasicAuthentication` 类：

    1.  与 Django 的会话框架一起用于认证

    1.  提供基于用户名和密码的 HTTP 基本认证

    1.  提供基于简单令牌的认证

1.  `rest_framework.authentication.SessionAuthentication`类：

    1.  与 Django 的会话框架一起用于认证

    1.  提供基于用户名和密码的 HTTP 基本认证

    1.  提供基于简单令牌的认证

1.  在`REST_FRAMEWORK`字典中，以下哪个设置的键指定了一个全局设置，该设置是一个字符串元组，表示我们想要用于认证的类？

    1.  `DEFAULT_AUTH_CLASSES`

    1.  `AUTHENTICATION_CLASSES`

    1.  `DEFAULT_AUTHENTICATION_CLASSES`

# 摘要

在本章中，我们从多个方面改进了 RESTful API。我们向模型中添加了唯一约束并更新了数据库，使得使用`PATCH`方法更新单个字段变得容易，并利用了分页。

然后，我们开始处理认证、权限和速率限制。我们向模型中添加了与安全相关的数据，并更新了数据库。我们在不同的代码片段中进行了许多更改，以实现特定的安全目标，并利用了 Django REST Framework 的认证和权限功能。

现在我们已经构建了一个改进且复杂的 API，它考虑了认证并使用了权限策略，我们将使用框架中包含的额外抽象，添加...
