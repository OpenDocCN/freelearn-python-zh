# 第八章：使用 Django 2.1 节流、过滤、测试和部署 API

在本章中，我们将使用 Django 2.1 和 Django REST Framework 中包含的附加功能来改进我们的 RESTful API。我们还将编写、执行和改进单元测试，并学习一些与部署相关的内容。我们将查看以下内容：

+   使用`requirements.txt`文件安装包以与过滤器、节流和测试一起工作

+   理解过滤、搜索和排序类

+   为视图配置过滤、搜索和排序

+   执行 HTTP 请求以测试过滤、搜索和排序功能

+   在可浏览 API 中进行过滤、搜索和排序

+   理解节流类和目标

+   配置节流策略

+   执行 HTTP 请求以测试节流策略

+   使用`pytest`设置单元测试

+   编写第一轮单元测试

+   使用`pytest`运行单元测试

+   提高测试覆盖率

+   在云上运行 Django RESTful API

# 使用`requirements.txt`文件安装包以与过滤器、节流和测试一起工作

确保您退出 Django 开发服务器。您只需在运行它的终端或命令提示符窗口中按*Ctrl* + *C*即可。

现在，我们将安装许多附加包以使用过滤功能，并能够轻松运行测试以及测量它们的代码覆盖率。确保您已激活我们在上一章中创建的虚拟环境，命名为`Django01`。在激活虚拟环境后，是时候运行许多命令了，这些命令对 macOS、Linux 和 Windows 都是相同的。

现在，我们将编辑现有的`requirements.txt`文件，以指定我们的应用程序所需的附加包...

# 理解过滤、搜索和排序类

在上一章中，我们利用 Django REST Framework 中可用的分页功能来指定我们希望将大型结果集分割成单独的数据页面的方式。然而，我们始终以整个`queryset`作为结果集进行工作；也就是说，我们没有应用任何过滤。

Django REST Framework 使得为已编码的视图自定义过滤、搜索和排序功能变得容易。

在`games_service/games_service`文件夹中打开`settings.py`文件。在声明名为`REST_FRAMEWORK`的字典的第一行之后添加以下突出显示的行，以添加新的`'DEFAULT_FILTER_BACKENDS'`设置键。不要删除新突出显示行之后的行。我们不显示它们以避免重复代码。示例的代码文件包含在`restful_python_2_08_01`文件夹中，位于`Django01/games-service/games_service/settings.py`文件中：

```py
REST_FRAMEWORK = { 
    'DEFAULT_FILTER_BACKENDS': ( 
        'django_filters.rest_framework.DjangoFilterBackend', 
        'rest_framework.filters.SearchFilter', 
        'rest_framework.filters.OrderingFilter'),
```

`'DEFAULT_FILTER_BACKENDS'`设置键的值指定了一个全局设置，它是一个字符串值的元组，表示我们想要用于过滤后端的默认类。我们将使用以下三个类：

| 模块 | 类名 | 所有者 |
| --- | --- | --- |
| `django_filters.rest_framework` | `DjangoFilterBackend` | Django 过滤器 |
| `rest_framework.filters` | `SearchFilter` | Django REST 框架 |
| `rest_framework.filters` | `OrderingFilter` | Django REST 框架 |

`DjangoFilterBackend` 类通过最近安装的 `django-filer` 包提供字段过滤功能。我们可以指定我们想要能够过滤的字段集合，或者创建一个具有更多自定义设置的 `django_filters.rest_framework.FilterSet` 类并将其与所需的视图关联。

`SearchFilter` 类提供基于单个查询参数的搜索功能，基于 Django 管理员的搜索功能。我们可以指定我们想要包含在搜索中的字段集合，客户端将能够通过在这些字段上执行单个查询来过滤项目。当我们想要使请求能够通过单个查询在多个字段上搜索时，这很有用。

`OrderingFilter` 类允许请求的客户端通过单个查询参数控制结果的排序方式。我们可以指定哪些字段可以进行排序。

注意，我们还可以通过将之前列出的任何类包含在一个元组中并将其分配给所需通用视图的 `filter_backends` 类属性来配置过滤后端。然而，在这种情况下，我们将使用所有基于类的视图的默认配置。

每当我们设计 RESTful API 时，我们必须确保我们以合理优化的方式提供所需的功能，并使用可用的资源。因此，我们必须小心确保我们配置的字段在过滤、搜索和排序功能中可用。我们在这些功能中做出的配置将对 Django 集成 ORM 在数据库上生成和执行的查询产生影响。我们必须确保我们有适当的数据库优化，考虑到将要执行的查询。

请保持在 `games_service/games_service` 文件夹中的 `settings.py` 文件。在声明字典 `INSTALLED_APPS` 的第一行之后添加以下突出显示的行，以将 `'django_filters'` 添加为新安装的应用程序到 Django 项目中。

不要删除新突出显示行之后出现的行。我们不显示它们以避免重复代码。示例代码文件包含在 `restful_python_2_08_01` 文件夹中的 `Django01/games-service/games_service/settings.py` 文件：

```py
INSTALLED_APPS = [ 
    # Django Filters 
    'django_filters', 
```

# 配置视图的过滤、搜索和排序

打开 `games_service/games` 文件夹中的 `views.py` 文件。在声明 `UserList` 类之前，在声明导入的最后一行之后添加以下代码。示例代码文件包含在 `restful_python_2_08_01` 文件夹中的 `Django01/games-service/games/views.py` 文件：

```py
from rest_framework import filters 
from django_filters import AllValuesFilter, DateTimeFilter, NumberFilter 
from django_filters.rest_framework import FilterSet 
```

继续编辑 `games_service/games` 文件夹中的 `views.py` 文件。将以下高亮行添加到 `views.py` 文件中声明的 `EsrbRatingList` 类。不要删除此类中未显示的现有行 ...

# 执行 HTTP 请求以测试过滤、搜索和排序

现在，我们可以启动 Django 的开发服务器来组合并发送 HTTP 请求。根据您的需求，执行以下两个命令之一以访问连接到您的局域网的其他设备或计算机上的 API：

```py
    python manage.py runserver
    python manage.py runserver 0.0.0.0:8000

```

在我们运行之前的任何命令之后，开发服务器将在端口 `8000` 上开始监听。

现在，我们将编写一个命令来组合并发送一个 HTTP `GET` 请求，以检索所有描述匹配 `T (Teen)` 的 ESRB 评级。示例代码文件包含在 `restful_python_2_08_01` 文件夹中，位于 `Django01/cmd/cmd801.txt` 文件中：

```py
    http ":8000/esrb-ratings/?description=T+(Teen)"
```

以下是对应的 `curl` 命令。示例代码文件包含在 `restful_python_2_08_01` 文件夹中，位于 `Django01/cmd/cmd802.txt` 文件中：

```py
    curl -iX GET "localhost:8000/esrb-ratings/?description=T+(Teen)"
```

以下行显示了一个与过滤中指定的描述匹配的单个 ESRB 评级的示例响应。以下行仅显示 JSON 主体，不包含头部信息：

```py
    {
        "count": 1,
        "next": null,
        "previous": null,
        "results": [
            {
                "description": "T (Teen)",
                "games": [
                    "http://localhost:8000/games/4/",
                    "http://localhost:8000/games/3/",
                    "http://localhost:8000/games/6/",
                    "http://localhost:8000/games/12/",
                    "http://localhost:8000/games/7/",
                    "http://localhost:8000/games/13/",
                    "http://localhost:8000/games/9/",
                    "http://localhost:8000/games/11/",
                    "http://localhost:8000/games/5/",
                    "http://localhost:8000/games/8/",
                    "http://localhost:8000/games/10/"
                ],
                "id": 2,
                "url": "http://localhost:8000/esrb-ratings/2/"
            }
        ]
    }

```

现在，我们将编写一个命令来组合并发送一个 HTTP `GET` 请求，以检索所有相关 ESRB 评级为 `1` 且 `played_times` 字段值等于 `10` 的游戏。我们希望按 `release_date` 降序排序结果，因此我们在 `ordering` 的值中指定 `-release_date`。字段名前的连字符（`-`）指定使用降序排序功能，而不是默认的升序排序。请确保将 `1` 替换为描述为 `AO (Adults Only)` 的 ESRB 评级的 `id` 值。示例代码文件包含在 `restful_python_2_08_01` 文件夹中，位于 `Django01/cmd/cmd803.txt` 文件中：

```py
http ":8000/games/?esrb_rating=1&played_times=10&ordering=-release_date"

```

以下是对应的 `curl` 命令。示例代码文件包含在 `restful_python_2_08_01` 文件夹中，位于 `Django01/cmd/cmd804.txt` 文件中：

```py
curl -iX GET 
"localhost:8000/games/?esrb_rating=1&played_times=10&ordering=-release_date"

```

以下行显示了一个与过滤中指定的标准匹配的单个游戏的示例响应。以下行仅显示 JSON 主体，不包含头部信息：

```py
    {
        "count": 1,
        "next": null,
        "previous": null,
        "results": [
            {
                "esrb_rating": "AO (Adults Only)",
                "name": "Mutant Football League: Dynasty Edition",
                "owner": "your_games_super_user",
                "played_once": true,
                "played_times": 10,
                "release_date": "2018-10-20T03:02:00.776594Z",
                "url": "http://localhost:8000/games/2/"
            }
        ]
    }

```

在 `GameList` 类中，我们将 `'esrb_rating'` 指定为 `filterset_fields` 字符串元组中的一个字符串。因此，我们必须在过滤中使用 ESRB 评级的 `id`。

现在，我们将运行一个命令，该命令将编写并发送一个使用与注册得分相关的游戏名称的过滤器来组合和发送 HTTP `GET` 请求。`PlayerScoreFilter` 类为我们提供了在 `game_name` 中的相关游戏名称的过滤器。我们将该过滤器与另一个与注册得分相关的玩家名称的过滤器结合起来。`PlayerScoreFilter` 类为我们提供了一种在 `player_name` 中过滤相关玩家名称的方法。必须满足准则中指定的两个条件，因此，过滤器使用 `AND` 运算符组合。示例的代码文件包含在 `restful_python_2_08_01` 文件夹中，位于 `Django01/cmd/cmd805.txt` 文件：

```py
http ":8000/player-
scores/?player_name=Enzo+Scocco&game_name=Battlefield+V"

```

以下是对应的 `curl` 命令。示例的代码文件包含在 `restful_python_2_08_01` 文件夹中，位于 `Django01/cmd/cmd806.txt` 文件：

```py
curl -iX GET "localhost:8000/player-
scores/?player_name=Enzo+Scocco&game_name=Battlefield+V"

```

以下行显示了与过滤器中指定的条件匹配的得分的示例响应。以下行仅显示没有标题的 JSON 正文：

```py
    {
        "count": 1,
        "next": null,
        "previous": null,
        "results": [
            {
                "game": "Battlefield V",
                "id": 3,
                "player": "Enzo Scocco",
                "score": 43200,
                "score_date": "2019-01-01T03:02:00.776594Z",
                "url": "http://localhost:8000/player-scores/3/"
            }
        ]
    }

```

我们将编写并发送一个 HTTP `GET` 请求来检索所有符合以下条件的得分，按 `score` 降序排序：

+   `score` 值介于 17,000 和 45,000 之间

+   `score_date` 值介于 2019-01-01 和 2019-01-31 之间

以下命令将编写并发送之前解释的 HTTP `GET` 请求。示例的代码文件包含在 `restful_python_2_08_01` 文件夹中，位于 `Django01/cmd/cmd807.txt` 文件：

```py
http ":8000/player-scores/?from_score_date=2019-01-01&to_score_date=2019-01-
31&min_score=17000&max_score=45000&ordering=-score"

```

以下是对应的 `curl` 命令。示例的代码文件包含在 `restful_python_2_08_01` 文件夹中，位于 `Django01/cmd/cmd808.txt` 文件：

```py
curl -iX GET "localhost:8000/player-scores/?from_score_date=2019-01-01&to_score_date=2019-01-
31&min_score=17000&max_score=45000&ordering=-score"

```

以下行显示了与过滤器中指定的条件匹配的三款游戏的示例响应。以下行仅显示没有标题的 JSON 正文：

```py
    {
        "count": 3,
        "next": null,
        "previous": null,
        "results": [
            {
                "game": "Battlefield V",
                "id": 3,
                "player": "Enzo Scocco",
                "score": 43200,
                "score_date": "2019-01-01T03:02:00.776594Z",
                "url": "http://localhost:8000/player-scores/3/"
            },
            {
                "game": "Battlefield V",
                "id": 1,
                "player": "Gaston Hillar",
                "score": 17500,
                "score_date": "2019-01-01T03:02:00.776594Z",
                "url": "http://localhost:8000/player-scores/1/"
            },
            {
                "game": "Mutant Football League: Dynasty Edition",
                "id": 4,
                "player": "Enzo Scocco",
                "score": 17420,
                "score_date": "2019-01-01T05:02:00.776594Z",
                "url": "http://localhost:8000/player-scores/4/"
            }
        ]
    }

```

在之前的请求中，没有响应包含超过一页的内容。如果响应需要超过一页，`previous` 和 `next` 键的值将显示包含过滤器、搜索、排序和分页组合的 URL。Django 将所有功能组合起来构建适当的 URL。

我们将编写并发送一个 HTTP 请求来检索所有名称以 `'S'` 开头的游戏。我们将使用我们配置的搜索功能，将搜索行为限制在 `name` 字段的以 `'S'` 开头匹配上。示例的代码文件包含在 `restful_python_2_08_01` 文件夹中，位于 `Django01/cmd/cmd809.txt` 文件：

```py
    http ":8000/games/?search=H"
```

以下是对应的 `curl` 命令。示例的代码文件包含在 `restful_python_2_08_01` 文件夹中，位于 `Django01/cmd/cmd810.txt` 文件：

```py
    curl -iX GET "localhost:8000/games/?search=H"
```

以下行显示了与指定搜索条件匹配的两个游戏的示例响应；即，那些名称以 `'H'` 开头的游戏。以下行仅显示没有标题的 JSON 正文：

```py
    {
        "count": 2,
        "next": null,
        "previous": null,
        "results": [
            {
                "esrb_rating": "T (Teen)",
                "name": "Heavy Fire: Red Shadow",
                "owner": "your_games_super_user",
                "played_once": false,
                "played_times": 0,
                "release_date": "2018-06-21T03:02:00.776594Z",
                "url": "http://localhost:8000/games/3/"
            },
            {
                "esrb_rating": "T (Teen)",
                "name": "Honor and Duty: D-Day",
                "owner": "your_games_super_user",
                "played_once": false,
                "played_times": 0,
                "release_date": "2018-06-21T03:02:00.776594Z",
                "url": "http://localhost:8000/games/6/"
            }
        ]
    }

```

到目前为止，我们一直在使用默认的搜索和排序查询参数：`'search'`和`'ordering'`。我们只需在`games_service/games_service`文件夹中的`settings.py`文件中的`SEARCH_PARAM`和`ORDERING_PARAM`设置中指定所需的名称作为字符串。

# 在可浏览 API 中进行过滤、搜索和排序

我们可以利用可浏览 API 通过网页浏览器轻松测试过滤、搜索和排序功能。打开网页浏览器并输入`http://localhost:8000/player-scores/`。如果您使用另一台计算机或设备运行浏览器，请将`localhost`替换为运行 Django 开发服务器的计算机的 IP 地址。

可浏览 API 将组合并发送一个 HTTP `GET`请求到`/player-scores/`，并将显示其执行结果；即，头部信息和 JSON 玩家得分列表。您会注意到在 OPTIONS 按钮的左侧有一个新的“过滤器”按钮。

点击“过滤器”，可浏览 API 将显示“过滤器”对话框，...

# 理解节流类和目标

到目前为止，我们还没有对我们的 API 使用设置任何限制，因此，认证用户和未认证用户都可以随意组合和发送他们想要的请求。我们只是利用了 Django REST Framework 中可用的分页功能来指定我们希望如何将大型结果集拆分为单个数据页。然而，任何用户都可以无限制地组合和发送数千个请求进行处理。

显然，在云平台上部署封装在微服务中的此类 API 不是一个好主意。任何用户对 API 的错误使用都可能导致微服务消耗大量资源，并且云平台的账单将反映这种情况。

我们将使用 Django REST Framework 中可用的节流功能来配置以下基于未认证或认证用户请求的 API 使用全局限制。我们将定义以下配置：

+   **未认证用户**：他们每小时最多可以运行`5`次请求

+   **认证用户**：他们每小时最多可以运行`20`次请求

此外，我们希望将每小时对 ESRB 评分相关视图的请求限制为最多 25 次，无论用户是否已认证。

Django REST Framework 提供了三个节流类（如下表所示），位于 `rest_framework.throttling` 模块中。所有这些类都是 `SimpleRateThrottle` 超类的子类，而 `SimpleRateThrottle` 是 `BaseThrottle` 超类的子类。这些类允许我们设置每个周期内允许的最大请求数量，该数量将基于不同的机制来确定先前请求信息以指定范围。节流前的请求信息存储在缓存中，并且这些类覆盖了 `get_cache_key` 方法，该方法确定范围：

| 节流类名称 | 描述 |
| --- | --- |
| `AnonRateThrottle` | 这个类限制了匿名用户可以发起的请求速率。请求的 IP 地址是唯一的缓存键。因此，请注意，来自同一 IP 地址的所有请求将累积总请求数量。 |
| `UserRateThrottle` | 这个类限制了特定用户可以发起的请求速率。对于认证用户，认证用户的 `id` 是唯一的缓存键。对于匿名用户，请求的 IP 地址是唯一的缓存键。 |
| `ScopedRateThrottle` | 这个类限制了使用 `throttle_scope` 属性指定的值标识的 API 特定部分的请求速率。当我们需要以不同的速率限制对 API 特定部分的访问时，此类非常有用。 |

# 配置节流策略

我们将使用三种节流类的组合来实现我们之前解释的目标。确保您已退出 Django 开发服务器。请记住，您只需在运行 Django 开发服务器的终端或命令提示符窗口中按 *Ctrl* + *C* 即可。

在 `games_service/games_service` 文件夹中打开 `settings.py` 文件。在声明名为 `REST_FRAMEWORK` 的字典的第一行之后添加以下突出显示的行，以添加新的 `'DEFAULT_THROTTLE_CLASSES'` 和 `'DEFAULT_THROTTLE_RATES'` 设置键。不要删除新突出显示行之后出现的行。我们不显示它们以避免重复代码。示例代码文件包含在 `restful_python_2_08_02` 文件夹中，...

# 提高测试覆盖率

现在，我们将编写额外的测试函数以提高测试覆盖率。具体来说，我们将编写与基于玩家类的视图相关的单元测试：`PlayerList` 和 `PlayerDetail`。保持 `games_service/games` 文件夹中的 `tests.py` 文件。在声明新函数和新测试函数的最后一行之后添加以下代码。示例代码文件包含在 `restful_python_2_08_03` 文件夹中，在 `Django01/games-service/games/tests.py` 文件中：

```py
def create_player(client, name, gender): 
    url = reverse('player-list') 
    player_data = {'name': name, 'gender': gender} 
    player_response = client.post(url, player_data, format='json') 
    return player_response 

@pytest.mark.django_db 
def test_create_and_retrieve_player(client): 
    """ 
    Ensure we can create a new Player and then retrieve it 
    """ 
    new_player_name = 'Will.i.am' 
    new_player_gender = Player.MALE 
    response = create_player(client, new_player_name, new_player_gender) 
    assert response.status_code == status.HTTP_201_CREATED 
    assert Player.objects.count() == 1 
    assert Player.objects.get().name == new_player_name 
```

代码声明了`create_player`函数，该函数接收新玩家所需的`name`和`gender`作为参数。该方法构建 URL 和数据字典，以向与`player-list`视图名称关联的视图发送 HTTP `POST`方法，并返回此请求生成的响应。代码使用接收到的`client`来访问允许我们轻松组合和发送 HTTP 请求进行测试的`APIClient`实例。许多测试函数将调用`create_player`函数来创建玩家，然后向 API 发送其他 HTTP 请求。

`test_create_and_retrieve_player`测试函数测试我们是否可以创建一个新的`Player`对象然后检索它。该方法调用之前解释的`create_player`函数，然后使用`assert`检查以下预期结果：

+   响应的`status_code`是 HTTP `201 Created`（`status.HTTP_201_CREATED`）

+   从数据库中检索到的`Player`对象总数是`1`

+   从数据库中检索到的`Player`对象的`name`属性与我们创建对象时指定的描述相匹配

+   从数据库中检索到的`Player`对象的`gender`属性与我们创建对象时指定的描述相匹配

保持位于`games_service/games`文件夹中的`tests.py`文件。在最后一行之后添加以下代码以声明新的测试函数。示例代码文件包含在`restful_python_2_08_03`文件夹中，在`Django01/games-service/games/tests.py`文件中：

```py
@pytest.mark.django_db 
def test_create_duplicated_player(client): 
    """ 
    Ensure we can create a new Player and we cannot create a duplicate 
    """ 
    url = reverse('player-list') 
    new_player_name = 'Fergie' 
    new_player_gender = Player.FEMALE 
    post_response1 = create_player(client, new_player_name, new_player_gender) 
    assert post_response1.status_code == status.HTTP_201_CREATED 
    post_response2 = create_player(client, new_player_name, new_player_gender) 
    assert post_response2.status_code == status.HTTP_400_BAD_REQUEST 

@pytest.mark.django_db 
def test_retrieve_players_list(client): 
    """ 
    Ensure we can retrieve a player 
    """ 
    new_player_name = 'Vanessa Perry' 
    new_player_gender = Player.FEMALE 
    create_player(client, new_player_name, new_player_gender) 
    url = reverse('player-list') 
    get_response = client.get(url, format='json') 
    assert get_response.status_code == status.HTTP_200_OK 
    assert get_response.data['count'] == 1 
    assert get_response.data['results'][0]['name'] == new_player_name 
    assert get_response.data['results'][0]['gender'] == new_player_gender
```

代码声明了以下以`test_`前缀开始的测试函数：

+   `test_create_duplicated_player`：这个测试函数测试了唯一约束是否使我们能够创建两个具有相同名称的玩家。当我们第二次使用重复的玩家名称组合并发送 HTTP `POST`请求时，我们应该收到 HTTP `400 Bad Request`状态码（`status.HTTP_400_BAD_REQUEST`）。

+   `test_retrieve_player_list`：这个测试函数测试我们是否可以通过 HTTP `GET`请求通过`id`检索特定的玩家。

我们刚刚编写了一些与玩家相关的测试来提高测试覆盖率。然而，我们绝对应该编写更多的测试来覆盖我们 API 中包含的所有功能。

现在，我们将使用`pytest`命令再次运行测试。确保你在激活了虚拟环境的终端或命令提示符窗口中运行以下命令，并且你位于包含`manage.py`文件的`games_service`文件夹中：

```py
    pytest -v
```

以下行显示了示例输出：

```py
    ============================== test session starts 
    ==============================
    platform darwin -- Python 3.6.6, pytest-3.9.3, py-1.7.0, pluggy-
    0.8.0 -- /Users/gaston/HillarPythonREST2/Django01/bin/python3
    cachedir: .pytest_cache
    Django settings: games_service.settings (from ini file)
    rootdir: /Users/gaston/HillarPythonREST2/Django01/games_service, 
    inifile: pytest.ini
    plugins: django-3.4.3, cov-2.6.0
    collected 8 items 

    games/tests.py::test_create_and_retrieve_esrb_rating PASSED               
    [ 12%]
    games/tests.py::test_create_duplicated_esrb_rating PASSED                 
    [ 25%]
    games/tests.py::test_retrieve_esrb_ratings_list PASSED                    
    [ 37%]
    games/tests.py::test_update_game_category PASSED                          
    [ 50%]
    games/tests.py::test_filter_esrb_rating_by_description PASSED             
    [ 62%]
    games/tests.py::test_create_and_retrieve_player PASSED                    
    [ 75%]
    games/tests.py::test_create_duplicated_player PASSED                      
    [ 87%]
    games/tests.py::test_retrieve_players_list PASSED                         
    [100%]

    =========================== 8 passed in 1.48 seconds 
    ============================

```

提供的输出详细说明了`pytest`执行了`8`个测试，并且所有测试都通过了。可以使用`pytest`的固定功能来减少之前编写的函数中的样板代码。然而，我们的重点是使函数易于理解。然后，你可以将代码作为基准，通过充分利用 Pytest 固定功能和`pytest-django`提供的附加功能来改进它。

我们刚刚创建了一些单元测试来了解我们如何编写它们。然而，当然，编写更多的测试来提供对 API 中包含的所有功能和执行场景的适当覆盖是必要的。

# 执行 HTTP 请求以测试限流策略

启动 Django 的开发服务器以组合和发送 HTTP 请求。根据您的需求执行以下两个命令之一：

```py
    python manage.py runserver
    python manage.py runserver 0.0.0.0:8000
```

现在，我们将编写多次组合和发送 HTTP 请求的命令。为了做到这一点，我们将学习如何通过以下任何一种选项与`http`和`curl`命令结合来实现这一目标。根据您的需求选择最合适的一个。不要忘记，您将需要在您选择的任何选项中激活虚拟环境，以便在您使用`http`命令时运行命令：

+   macOS：带有 Bash shell 的终端。

+   Linux：带有 Bash shell 的终端。

+   Windows：...

# 使用 pytest 设置单元测试

在`games_service`文件夹内创建一个新的`pytest.ini`文件（与包含`manage.py`文件的同一文件夹）。以下行显示了指定 Pytest 所需配置的代码。示例的代码文件包含在`restful_python_2_08_02`文件夹中，在`Django01/game_service/manage.py`文件中：

```py
[pytest] 
DJANGO_SETTINGS_MODULE = games_service.settings 
python_files = tests.py test_*.py *_tests.py 
```

配置变量`DJANGO_SETTINGS_MODULE`指定了在执行测试时，我们希望使用位于`games_service/games_service`文件夹中的`settings.py`文件作为 Django 的设置模块。

配置变量`python_files`指示`pytest`将使用哪些过滤器来查找具有测试函数的模块。

# 编写第一轮单元测试

现在，我们将编写第一轮单元测试。具体来说，我们将编写与 ESRB 评级类视图相关的单元测试：`EsrbRatingList`和`EsrbRatingDetail`。

打开位于`games_service/games`文件夹中的`tests.py`文件。用以下行替换现有的代码，这些行声明了许多`import`语句和两个函数。示例的代码文件包含在`restful_python_2_08_02`文件夹中，在`Django01/games-service/games/tests.py`文件中：

```py
import pytest 
from django.urls import reverse 
from django.utils.http import urlencode 
from rest_framework import status 
from games import views 
from games.models import EsrbRating 

def create_esrb_rating(client, description): 
 url = reverse(views.EsrbRatingList.name) ...
```

# 使用 pytest 运行单元测试

现在，运行以下命令以创建测试数据库，运行所有迁移，并使用 `pytest`，结合 `pytest-django` 插件，发现并执行我们创建的所有测试。测试运行器将执行 `tests.py` 文件中以 `test_` 前缀开始的全部方法，并将显示结果。确保你在激活了虚拟环境的终端或命令提示符窗口中运行此命令，并且你位于包含 `manage.py` 文件的 `games_service` 文件夹内：

```py
    pytest -v
```

在通过 `pytest` 在 API 上运行请求时，测试不会更改我们一直在使用的数据库。

测试运行器将执行 `tests.py` 中定义的所有以 `test_` 前缀开始的函数，并将显示结果。我们使用 `-v` 选项指示 `pytest` 以详细模式打印测试函数名称和状态。

以下行显示了示例输出：

```py
    ============================== test session starts 
    ==============================
    platform darwin -- Python 3.6.6, pytest-3.9.3, py-1.7.0, pluggy-
    0.8.0 -- /Users/gaston/HillarPythonREST2/Django01/bin/python3
    cachedir: .pytest_cache
    Django settings: games_service.settings (from ini file)
    rootdir: /Users/gaston/HillarPythonREST2/Django01/games_service, 
    inifile: pytest.ini
    plugins: django-3.4.3, cov-2.6.0
    collected 5 items 

    games/tests.py::test_create_and_retrieve_esrb_rating PASSED               
    [ 20%]
    games/tests.py::test_create_duplicated_esrb_rating PASSED                 
    [ 40%]
    games/tests.py::test_retrieve_esrb_ratings_list PASSED                    
    [ 60%]
    games/tests.py::test_update_game_category PASSED                          
    [ 80%]
    games/tests.py::test_filter_esrb_rating_by_description PASSED             
    [100%]

    =========================== 5 passed in 1.68 seconds 
    ============================

```

输出提供了详细信息，表明测试运行器执行了 `5` 个测试，并且所有测试都通过了。

# 在云端运行 Django RESTful API

与 Django 和 Django REST 框架相关的一个最大的缺点是每个 HTTP 请求都是阻塞的。因此，每当 Django 服务器收到一个 HTTP 请求时，它不会开始处理传入队列中的任何其他 HTTP 请求，直到服务器收到第一个 HTTP 请求的响应。 

然而，RESTful Web 服务的一个巨大优势是它们是无状态的；也就是说，它们不应该在任何服务器上保持客户端状态。我们的 API 是一个无状态 RESTful Web 服务的良好示例。因此，我们可以让 API 在尽可能多的服务器上运行，以实现我们的可扩展性目标。显然，我们必须考虑到我们可以轻松地转换数据库服务器 ...

# 测试你的知识

让我们看看你是否能正确回答以下问题：

1.  以下哪个由 `pytest-django` 插件提供的 fixture 允许我们访问 `APIClient` 实例，这使得我们能够轻松地编写和发送 HTTP 请求进行测试？

    1.  `client`

    1.  `api_client`

    1.  `http`

1.  以下哪个在 `pytest-django` 中声明的装饰器表示测试函数需要与测试数据库一起工作？

    1.  `@pytest.django.db`

    1.  `@pytest.mark.django_db`

    1.  `@pytest.mark.db`

1.  `ScopedRateThrottle` 类：

    1.  限制特定用户可以发起的请求数量

    1.  限制与 `throttle_scope` 属性分配的值标识的 API 特定部分的请求数量

    1.  限制匿名用户可以发起的请求数量

1.  `UserRateThrottle` 类：

    1.  限制特定用户可以发起的请求数量

    1.  限制与 `throttle_scope` 属性分配的值标识的 API 特定部分的请求数量

    1.  限制匿名用户可以发起的请求数量

1.  `DjangoFilterBackend` 类：

    1.  提供基于单个查询参数的搜索功能，并基于 Django 管理员的搜索功能

    1.  允许客户端通过单个查询参数控制结果的排序

    1.  提供字段过滤功能

1.  The `SearchFilter` class:

    1.  提供基于单个查询参数的搜索功能，并基于 Django 管理员的搜索功能

    1.  允许客户端通过单个查询参数控制结果的排序

    1.  提供字段过滤功能

1.  以下哪个类属性指定了我们想要用于基于类的视图的`FilterSet`子类？

    1.  `filters_class`

    1.  `filtering_class`

    1.  `filterset_class`

# 摘要

在本章中，我们利用了 Django REST Framework 中包含的许多功能来定义节流策略。我们使用了类的过滤、搜索和排序，使得在 HTTP 请求中配置过滤器、搜索查询和期望的结果排序变得容易。我们使用了可浏览的 API 功能来测试我们 API 中包含的新特性。

我们编写了第一轮单元测试，并设置了必要的配置以使用流行的现代`pytest` Python 单元测试框架与 Django REST Framework。然后，我们编写了额外的单元测试以改进测试覆盖率。最后，我们理解了许多关于云部署和可扩展性的考虑因素。

现在我们已经使用 Django REST Framework 构建了一个复杂的 API ...
