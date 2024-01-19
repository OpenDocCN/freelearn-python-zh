# 使 Scraper 成为一个真正的服务

在本章中，我们将涵盖：

+   创建和配置 Elastic Cloud 试用账户

+   使用 curl 访问 Elastic Cloud 集群

+   使用 Python 连接 Elastic Cloud 集群

+   使用 Python API 执行 Elasticsearch 查询

+   使用 Elasticsearch 查询具有特定技能的工作

+   修改 API 以按技能搜索工作

+   将配置存储在环境中

为 ECS 创建 AWS IAM 用户和密钥对

+   配置 Docker 以与 ECR 进行身份验证

+   将容器推送到 ECR

+   创建 ECS 集群

+   创建任务来运行我们的容器

+   在 AWS 中启动和访问容器

# 介绍

在本章中，我们将首先添加一个功能，使用 Elasticsearch 搜索工作列表，并扩展 API 以实现此功能。然后将 Elasticsearch 功能移至 Elastic Cloud，这是将我们的基于云的 Scraper 云化的第一步。然后，我们将将我们的 Docker 容器移至 Amazon Elastic Container Repository（ECR），最后在 Amazon Elastic Container Service（ECS）中运行我们的容器（和 Scraper）。

# 创建和配置 Elastic Cloud 试用账户

在这个示例中，我们将创建和配置一个 Elastic Cloud 试用账户，以便我们可以将 Elasticsearch 作为托管服务使用。Elastic Cloud 是 Elasticsearch 创建者提供的云服务，提供了完全托管的 Elasticsearch 实现。

虽然我们已经研究了将 Elasticsearch 放入 Docker 容器中，但在 AWS 中实际运行带有 Elasticsearch 的容器非常困难，因为存在许多内存要求和其他系统配置，这些配置在 ECS 中很难实现。因此，对于云解决方案，我们将使用 Elastic Cloud。

# 如何做

我们将按照以下步骤进行：

1.  打开浏览器，转到[`www.elastic.co/cloud/as-a-service/signup`](https://www.elastic.co/cloud/as-a-service/signup)。您将看到一个类似以下内容的页面：

![](img/db8027f8-ba93-421c-a026-fb2cedd5ddcb.png)Elastic Cloud 注册页面

1.  输入您的电子邮件并点击“开始免费试用”按钮。当邮件到达时，请进行验证。您将被带到一个页面来创建您的集群：

![](img/c08434c1-5481-4c1f-8582-674152731121.png)集群创建页面

1.  在其他示例中，我将使用 AWS（而不是 Google）在俄勒冈州（us-west-2）地区，所以我将为这个集群选择这两个选项。您可以选择适合您的云和地区。您可以将其他选项保持不变，然后只需按“创建”。然后您将看到您的用户名和密码。记下来。以下屏幕截图给出了它如何显示用户名和密码：

![](img/c8d98099-d766-43d2-9825-62dc4ddd9b9c.png)Elastic Cloud 账户的凭据信息我们不会在任何示例中使用 Cloud ID。

1.  接下来，您将看到您的端点。对我们来说，Elasticsearch URL 很重要：

![](img/054dab3f-ed34-43a7-9cab-099198e4bc8a.png)

1.  就是这样 - 你已经准备好了（至少可以使用 14 天）！

# 使用 curl 访问 Elastic Cloud 集群

Elasticsearch 基本上是通过 REST API 访问的。Elastic Cloud 也是一样的，实际上是相同的 API。我们只需要知道如何正确构建 URL 以进行连接。让我们来看看。

# 如何做

我们将按照以下步骤进行：

1.  当您注册 Elastic Cloud 时，您会获得各种端点和变量，例如用户名和密码。URL 类似于以下内容：

```py
https://<account-id>.us-west-2.aws.found.io:9243
```

根据云和地区，域名的其余部分以及端口可能会有所不同。

1.  我们将使用以下 URL 的略微变体来与 Elastic Cloud 进行通信和身份验证：

```py
https://<username>:<password>@<account-id>.us-west-2.aws.found.io:9243
```

1.  目前，我的 URL 是（在您阅读此内容时将被禁用）：

```py
https://elastic:tduhdExunhEWPjSuH73O6yLS@d7c72d3327076cc4daf5528103c46a27.us-west-2.aws.found.io:9243
```

1.  可以使用 curl 检查基本身份验证和连接：

```py
$ curl https://elastic:tduhdExunhEWPjSuH73O6yLS@7dc72d3327076cc4daf5528103c46a27.us-west-2.aws.found.io:9243
{
  "name": "instance-0000000001",
  "cluster_name": "7dc72d3327076cc4daf5528103c46a27",
  "cluster_uuid": "g9UMPEo-QRaZdIlgmOA7hg",
  "version": {
    "number": "6.1.1",
    "build_hash": "bd92e7f",
    "build_date": "2017-12-17T20:23:25.338Z",
    "build_snapshot": false,
    "lucene_version": "7.1.0",
    "minimum_wire_compatibility_version": "5.6.0",
    "minimum_index_compatibility_version": "5.0.0"
  },
  "tagline": "You Know, for Search"
}
Michaels-iMac-2:pems michaelheydt$
```

然后我们可以开始交谈了！

# 使用 Python 连接 Elastic Cloud 集群

现在让我们看看如何使用 Elasticsearch Python 库连接到 Elastic Cloud。

# 准备工作

此示例的代码位于`11/01/elasticcloud_starwars.py`脚本中。此脚本将从 swapi.co API/网站中获取 Star Wars 角色数据，并将其放入 Elastic Cloud 中。

# 如何做

我们按照以下步骤进行：

1.  将文件作为 Python 脚本执行：

```py
$ python elasticcloud_starwars.py
```

1.  这将循环遍历最多 20 个字符，并将它们放入`sw`索引中，文档类型为`people`。代码很简单（用您的 URL 替换 URL）：

```py
from elasticsearch import Elasticsearch
import requests
import json

if __name__ == '__main__':
    es = Elasticsearch(
        [
            "https://elastic:tduhdExunhEWPjSuH73O6yLS@d7c72d3327076cc4daf5528103c46a27.us-west-2.aws.found.io:9243"
  ])

i = 1 while i<20:
    r = requests.get('http://swapi.co/api/people/' + str(i))
    if r.status_code is not 200:
 print("Got a " + str(r.status_code) + " so stopping")
 break  j = json.loads(r.content)
 print(i, j)
 #es.index(index='sw', doc_type='people', id=i, body=json.loads(r.content))
  i = i + 1
```

1.  连接是使用 URL 进行的，用户名和密码添加到其中。数据是使用 GET 请求从 swapi.co 中提取的，然后使用 Elasticsearch 对象上的`.index()`调用。您将看到类似以下的输出：

```py
1 Luke Skywalker
2 C-3PO
3 R2-D2
4 Darth Vader
5 Leia Organa
6 Owen Lars
7 Beru Whitesun lars
8 R5-D4
9 Biggs Darklighter
10 Obi-Wan Kenobi
11 Anakin Skywalker
12 Wilhuff Tarkin
13 Chewbacca
14 Han Solo
15 Greedo
16 Jabba Desilijic Tiure
Got a 404 so stopping
```

# 还有更多...

当您注册 Elastic Cloud 时，您还会获得一个指向 Kibana 的 URL。Kibana 是 Elasticsearch 的强大图形前端：

1.  在浏览器中打开 URL。您将看到一个登录页面：

![](img/8cba8c2e-8870-4c2c-a721-4a1787387ad8.png)Kibana 登录页面

1.  输入您的用户名和密码，然后您将进入主仪表板：

![](img/6aff28f8-f90a-4018-a809-990fceae344a.png)创建索引模式

我们被要求为我们的应用程序创建一个索引模式：sw 创建的一个索引。在索引模式文本框中，输入`sw*`，然后按下下一步。

1.  我们将被要求选择时间过滤器字段名称。选择 I don't want to use the Time Filter，然后按下 Create Index Pattern 按钮。几秒钟后，您将看到创建的索引的确认：

![](img/951c8ef3-95f8-454a-b76c-7315b6f68b47.png)创建的索引

1.  现在点击 Discover 菜单项，您将进入交互式数据浏览器，在那里您将看到我们刚刚输入的数据：

![](img/3c49c59b-7f1d-4f48-b3bc-081f342a95b8.png)添加到我们的索引的数据

在这里，您可以浏览数据，看看 Elasticsearch 如何有效地存储和组织这些数据。

# 使用 Python API 执行 Elasticsearch 查询

现在让我们看看如何使用 Elasticsearch Python 库搜索 Elasticsearch。我们将在 Star Wars 索引上执行简单的搜索。

# 准备工作

确保在示例中修改连接 URL 为您的 URL。

# 如何做

搜索的代码在`11/02/search_starwars_by_haircolor.py`脚本中，只需执行该脚本即可运行。这是一个相当简单的搜索，用于查找头发颜色为`blond`的角色：

1.  代码的主要部分是：

```py
es = Elasticsearch(
    [
        "https://elastic:tduhdExunhEWPjSuH73O6yLS@7dc72d3327076cc4daf5528103c46a27.us-west-2.aws.found.io:9243"
  ])

search_definition = {
    "query":{
        "match": {
            "hair_color": "blond"
  }
    }
}

result = es.search(index="sw", doc_type="people", body=search_definition)
print(json.dumps(result, indent=4))
```

1.  通过构建表达 Elasticsearch DSL 查询的字典来执行搜索。在这种情况下，我们的查询要求所有文档的`"hair_color"`属性为`"blond"`。然后将此对象作为`.search`方法的 body 参数传递。此方法的结果是描述找到的内容（或未找到的内容）的字典。在这种情况下：

```py
{
  "took": 2,
  "timed_out": false,
  "_shards": {
    "total": 5,
    "successful": 5,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": 2,
    "max_score": 1.3112576,
    "hits": [
      {
        "_index": "sw",
        "_type": "people",
        "_id": "1",
        "_score": 1.3112576,
        "_source": {
          "name": "Luke Skywalker",
          "height": "172",
          "mass": "77",
          "hair_color": "blond",
          "skin_color": "fair",
          "eye_color": "blue",
          "birth_year": "19BBY",
          "gender": "male",
          "homeworld": "https://swapi.co/api/planets/1/",
          "films": [
            "https://swapi.co/api/films/2/",
            "https://swapi.co/api/films/6/",
            "https://swapi.co/api/films/3/",
            "https://swapi.co/api/films/1/",
            "https://swapi.co/api/films/7/"
          ],
          "species": [
            "https://swapi.co/api/species/1/"
          ],
          "vehicles": [
            "https://swapi.co/api/vehicles/14/",
            "https://swapi.co/api/vehicles/30/"
          ],
          "starships": [
            "https://swapi.co/api/starships/12/",
            "https://swapi.co/api/starships/22/"
          ],
          "created": "2014-12-09T13:50:51.644000Z",
          "edited": "2014-12-20T21:17:56.891000Z",
          "url": "https://swapi.co/api/people/1/"
        }
      },
      {
        "_index": "sw",
        "_type": "people",
        "_id": "11",
        "_score": 0.80259144,
        "_source": {
          "name": "Anakin Skywalker",
          "height": "188",
          "mass": "84",
          "hair_color": "blond",
          "skin_color": "fair",
          "eye_color": "blue",
          "birth_year": "41.9BBY",
          "gender": "male",
          "homeworld": "https://swapi.co/api/planets/1/",
          "films": [
            "https://swapi.co/api/films/5/",
            "https://swapi.co/api/films/4/",
            "https://swapi.co/api/films/6/"
          ],
          "species": [
            "https://swapi.co/api/species/1/"
          ],
          "vehicles": [
            "https://swapi.co/api/vehicles/44/",
            "https://swapi.co/api/vehicles/46/"
          ],
          "starships": [
            "https://swapi.co/api/starships/59/",
            "https://swapi.co/api/starships/65/",
            "https://swapi.co/api/starships/39/"
          ],
          "created": "2014-12-10T16:20:44.310000Z",
          "edited": "2014-12-20T21:17:50.327000Z",
          "url": "https://swapi.co/api/people/11/"
        }
      }
    ]
  }
}
```

结果为我们提供了有关搜索执行的一些元数据，然后是`hits`属性中的结果。每个命中都会返回实际文档以及索引名称、文档类型、文档 ID 和分数。分数是文档与搜索查询相关性的 lucene 计算。虽然此查询使用属性与值的精确匹配，但您可以看到这两个文档仍然具有不同的分数。我不确定为什么在这种情况下，但搜索也可以不太精确，并基于各种内置启发式来查找“类似”某个句子的项目，也就是说，例如当您在 Google 搜索框中输入文本时。

# 还有更多...

Elasticsearch 搜索 DSL 和搜索引擎本身非常强大和富有表现力。我们只会在下一个配方中查看这个例子和另一个例子，所以我们不会详细介绍。要了解更多关于 DSL 的信息，您可以从官方文档开始[`www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html`](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html)。

# 使用 Elasticsearch 查询具有特定技能的工作

在这个配方中，我们回到使用我们创建的爬虫从 StackOverflow 中爬取和存储工作列表到 Elasticsearch。然后，我们扩展这个功能，查询 Elasticsearch 以找到包含一个或多个指定技能的工作列表。

# 准备工作

我们将使用一个本地 Elastic Cloud 引擎而不是本地 Elasticsearch 引擎。如果您愿意，您可以更改。现在，我们将在一个本地运行的 Python 脚本中执行此过程，而不是在容器内或在 API 后面执行。

# 如何做到这一点

我们按照以下步骤进行：

1.  该配方的代码位于`11/03/search_jobs_by_skills.py`文件中。

```py
from sojobs.scraping import get_job_listing_info
from elasticsearch import Elasticsearch
import json

if __name__ == "__main__":

    es = Elasticsearch()

    job_ids = ["122517", "163854", "138222", "164641"]

    for job_id in job_ids:
        if not es.exists(index='joblistings', doc_type='job-listing', id=job_id):
            listing = get_job_listing_info(job_id)
            es.index(index='joblistings', doc_type='job-listing', id=job_id, body=listing)

    search_definition = {
        "query": {
            "match": {
                "JSON.skills": {
                    "query": "c#"   }
            }
        }
    }

    result = es.search(index="joblistings", doc_type="job-listing", body=search_definition)
    print(json.dumps(result, indent=4))
```

这段代码的第一部分定义了四个工作列表，如果它们尚不可用，则将它们放入 Elasticsearch 中。它遍历了这个工作的 ID，如果尚未可用，则检索它们并将它们放入 Elasticsearch 中。

其余部分定义了要针对 Elasticsearch 执行的查询，并遵循相同的模式来执行搜索。唯一的区别在于搜索条件的定义。最终，我们希望将一系列工作技能与工作列表中的技能进行匹配。

这个查询只是将单个技能与我们的工作列表文档中的技能字段进行匹配。示例指定我们要匹配目标文档中的 JSON.skills 属性。这些文档中的技能就在文档的根部下面，所以在这个语法中我们用 JSON 作为前缀。

Elasticsearch 中的这个属性是一个数组，我们的查询值将匹配该属性数组中的任何一个值为`"c#"`的文档。

1.  在 Elasticsearch 中只使用这四个文档运行此搜索将产生以下结果（这里的输出只显示结果，而不是返回的四个文档的完整内容）：

```py
{
  "took": 4,
  "timed_out": false,
  "_shards": {
    "total": 5,
    "successful": 5,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": 2,
    "max_score": 1.031828,
    "hits": [

```

放入 Elasticsearch 的每个工作都有 C#作为技能（我随机选择了这些文档，所以这有点巧合）。

1.  这些搜索的结果返回了每个被识别的文档的全部内容。如果我们不希望每次命中都返回整个文档，我们可以更改查询以实现这一点。让我们修改查询，只返回命中的 ID。将`search_definition`变量更改为以下内容：

```py
search_definition = {
    "query": {
        "match": {
            "JSON.skills": {
                "query": "c# sql"
  }
        }
    },
    "_source": ["ID"]
}
```

1.  包括`"_source"`属性告诉 Elasticsearch 在结果中返回指定的文档属性。执行此查询将产生以下输出：

```py
{
  "took": 4,
  "timed_out": false,
  "_shards": {
    "total": 5,
    "successful": 5,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": 2,
    "max_score": 1.031828,
    "hits": [
      {
        "_index": "joblistings",
        "_type": "job-listing",
        "_id": "164641",
        "_score": 1.031828,
        "_source": {
          "ID": "164641"
        }
      },
      {
        "_index": "joblistings",
        "_type": "job-listing",
        "_id": "122517",
        "_score": 0.9092852,
        "_source": {
          "ID": "122517"
        }
      }
    ]
  }
}
```

现在，每个命中只返回文档的 ID 属性。如果有很多命中，这将有助于控制结果的大小。

1.  让我们来到这个配方的最终目标，识别具有多种技能的文档。这实际上是对`search_defintion`进行了一个非常简单的更改：

```py
search_definition={
  "query": {
    "match": {
      "JSON.skills": {
        "query": "c# sql",
        "operator": "AND"
      }
    }
  },
  "_source": [
    "ID"
  ]
}
```

这说明我们只想要包含`"c#"`和`"sql"`两个技能的文档。然后运行脚本的结果如下：

```py
{
  "took": 4,
  "timed_out": false,
  "_shards": {
    "total": 5,
    "successful": 5,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": 2,
    "max_score": 1.031828,
    "hits": [
      {
        "_index": "joblistings",
        "_type": "job-listing",
        "_id": "164641",
        "_score": 1.031828,
        "_source": {
          "ID": "164641"
        }
      },
      {
        "_index": "joblistings",
        "_type": "job-listing",
        "_id": "122517",
        "_score": 0.9092852,
        "_source": {
          "ID": "122517"
        }
      }
    ]
  }
}
```

结果集现在减少到两个命中，如果您检查，这些是唯一具有这些技能值的两个。

# 修改 API 以按技能搜索工作

在这个配方中，我们将修改我们现有的 API，添加一个方法来搜索具有一组技能的工作。

# 如何做到这一点

我们将扩展 API 代码。 我们将对 API 的实现进行两个基本更改。 第一个是我们将为搜索功能添加一个额外的 Flask-RESTful API 实现，第二个是我们将 Elasticsearch 和我们自己的微服务的地址都可通过环境变量进行配置。

API 实现在`11/04_scraper_api.py`中。 默认情况下，该实现尝试连接到本地系统上的 Elasticsearch。 如果您正在使用 Elastic Cloud，请确保更改 URL（并确保索引中有文档）：

1.  可以通过简单执行脚本来启动 API：

```py
$ python scraper_api.py
Starting the job listing API ...
 * Running on http://0.0.0.0:8080/ (Press CTRL+C to quit)
 * Restarting with stat
Starting the job listing API ...
 * Debugger is active!
 * Debugger pin code: 449-370-213
```

1.  要进行搜索请求，我们可以向`/joblistings/search`端点进行 POST，以`"skills=<用空格分隔的技能>"`的形式传递数据。 以下是使用 C#和 SQL 进行作业搜索的示例：

```py
$ curl localhost:8080/joblistings/search -d "skills=c# sql"
{
  "took": 4,
  "timed_out": false,
  "_shards": {
    "total": 5,
    "successful": 5,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": 2,
    "max_score": 1.031828,
    "hits": [
      {
        "_index": "joblistings",
        "_type": "job-listing",
        "_id": "164641",
        "_score": 1.031828,
        "_source": {
          "ID": "164641"
        }
      },
      {
        "_index": "joblistings",
        "_type": "job-listing",
        "_id": "122517",
        "_score": 0.9092852,
        "_source": {
          "ID": "122517"
        }
      }
    ]
  }
}
```

我们得到了在上一个食谱中看到的结果。 现在我们已经通过互联网实现了我们的搜索功能！

# 工作原理

这通过添加另一个 Flask-RESTful 类实现来实现：

```py
class JobSearch(Resource):
    def post(self):
        skills = request.form['skills']
        print("Request for jobs with the following skills: " + skills)

        host = 'localhost'
  if os.environ.get('ES_HOST'):
            host = os.environ.get('ES_HOST')
        print("ElasticSearch host: " + host)

        es = Elasticsearch(hosts=[host])
        search_definition = {
            "query": {
                "match": {
                    "JSON.skills": {
                        "query": skills,
                        "operator": "AND"
  }
                }
            },
            "_source": ["ID"]
        }

        try:
            result = es.search(index="joblistings", doc_type="job-listing", body=search_definition)
            print(result)
            return result

        except:
            return sys.exc_info()[0]

api.add_resource(JobSearch, '/', '/joblistings/search')
```

这个类实现了一个 post 方法，作为映射到`/joblistings/search`的资源。 进行 POST 操作的原因是我们传递了一个由多个单词组成的字符串。 虽然这可以在 GET 操作中进行 URL 编码，但 POST 允许我们将其作为键值传递。 虽然我们只有一个键，即 skills，但未来扩展到其他键以支持其他搜索参数可以简单地添加。

# 还有更多...

从 API 实现中执行搜索的决定是应该在系统发展时考虑的。 这是我的观点，仅仅是我的观点（但我认为其他人会同意），就像 API 调用实际的爬取微服务一样，它也应该调用一个处理搜索的微服务（然后该微服务将与 Elasticsearch 进行接口）。 这也适用于存储从爬取微服务返回的文档，以及访问 Elasticsearch 以检查缓存文档。 但出于我们在这里的目的，我们将尽量保持简单。

# 在环境中存储配置

这个食谱指出了在上一个食谱中对 API 代码进行的更改，以支持**12-Factor**应用程序的一个*因素*。 12-Factor 应用程序被定义为设计为软件即服务运行的应用程序。 我们已经在这个方向上移动了一段时间的爬虫，将其分解为可以独立运行的组件，作为脚本或容器运行，并且很快我们将看到，作为云中的组件。 您可以在[`12factor.net/`](https://12factor.net/)上了解有关 12-Factor 应用程序的所有信息。

Factor-3 指出我们应该通过环境变量将配置传递给我们的应用程序。 虽然我们绝对不希望硬编码诸如外部服务的 URL 之类的东西，但使用配置文件也不是最佳实践。 在部署到各种环境（如容器或云）时，配置文件通常会固定在镜像中，并且无法根据应用程序动态部署到不同环境而随需求更改。

修复此问题的最佳方法是始终查找环境变量中的配置设置，这些设置可以根据应用程序的运行方式而改变。 大多数用于运行 12-Factor 应用程序的工具允许根据环境决定应用程序应该在何处以及如何运行来设置环境变量。

# 如何做到这一点

在我们的工作列表实现中，我们使用以下代码来确定 Elasticsearch 的主机：

```py
host = 'localhost'
if os.environ.get('ES_HOST'):
    host = os.environ.get('ES_HOST')
print("ElasticSearch host: " + host)

es = Elasticsearch(hosts=[host])
```

这是一个简单直接的操作，但对于使我们的应用程序在不同环境中具有极高的可移植性非常重要。 默认情况下使用 localhost，但让我们使用`ES_HOST`环境变量定义不同的主机。

技能搜索的实现也进行了类似的更改，以允许我们更改我们的爬虫微服务的本地主机的默认值：

```py
CONFIG = {'AMQP_URI': "amqp://guest:guest@localhost"}
if os.environ.get('JOBS_AMQP_URL'):
    CONFIG['AMQP_URI'] = os.environ.get('JOBS_AMQP_URL')
print("AMQP_URI: " + CONFIG["AMQP_URI"])

with ClusterRpcProxy(CONFIG) as rpc:
```

我们将在接下来的教程中看到 Factor-3 的使用，当我们将这段代码移到 AWS 的弹性容器服务时。

# 创建用于 ECS 的 AWS IAM 用户和密钥对

在这个教程中，我们将创建一个身份和访问管理（IAM）用户账户，以允许我们访问 AWS 弹性容器服务（ECS）。我们需要这个，因为我们将把我们的爬虫和 API 打包到 Docker 容器中（我们已经做过了），但现在我们将把这些容器移到 AWS ECS 并在那里运行它们，使我们的爬虫成为一个真正的云服务。

# 准备就绪

这假设你已经创建了一个 AWS 账户，我们在之前的章节中使用过它，当我们查看 SQS 和 S3 时。你不需要另一个账户，但我们需要创建一个非根用户，该用户具有使用 ECS 的权限。

# 操作步骤

有关如何创建具有 ECS 权限和密钥对的 IAM 用户的说明可以在[`docs.aws.amazon.com/AmazonECS/latest/developerguide/get-set-up-for-amazon-ecs.html`](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/get-set-up-for-amazon-ecs.html)找到。

这个页面上有很多说明，比如设置 VPC 和安全组。现在只关注创建用户、分配权限和创建密钥对。

我想要强调的一件事是你创建的 IAM 账户的权限。在[`docs.aws.amazon.com/AmazonECS/latest/developerguide/instance_IAM_role.html`](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/instance_IAM_role.html)上有关于如何做这个的详细说明。我曾经见过这样的操作没有做好。只需确保当你检查刚刚创建的用户的权限时，以下权限已经被分配：

![](img/f67ea806-ca40-49f7-a2f1-fd561116c372.png)AWS IAM 凭证

我直接将这些附加到我用于 ECS 的账户上，而不是通过组。如果没有分配这些，当推送容器到 ECR 时会出现加密的身份验证错误。

还有一件事：我们需要访问密钥 ID 和相关的密钥。这将在创建用户时呈现给你。如果你没有记录下来，你可以在用户账户页面的安全凭证选项卡中创建另一个：

![](img/9fa87343-5f4c-439a-abc8-f79339dfcd9b.png)

请注意，无法获取已存在的访问密钥 ID 的密钥。你需要创建另一个。

# 配置 Docker 以便与 ECR 进行身份验证

在这个教程中，我们将配置 Docker 以便能够将我们的容器推送到弹性容器仓库（ECR）。

# 准备就绪

Docker 的一个关键元素是 Docker 容器仓库。我们之前使用 Docker Hub 来拉取容器。但我们也可以将我们的容器推送到 Docker Hub，或者任何兼容 Docker 的容器仓库，比如 ECR。但这并不是没有问题的。docker CLI 并不自然地知道如何与 ECR 进行身份验证，所以我们需要做一些额外的工作来让它能够工作。

确保安装了 AWS 命令行工具。这些工具是必需的，用于让 Docker 能够与 ECR 进行身份验证。在[`docs.aws.amazon.com/cli/latest/userguide/installing.html`](https://docs.aws.amazon.com/cli/latest/userguide/installing.html)上有很好的说明。安装验证通过后，你需要配置 CLI 以使用前面教程中创建的账户。这可以通过`aws configure`命令来完成，该命令会提示你输入四个项目：

```py
$ aws configure
AWS Access Key ID [None]: AKIA---------QKCVQAA
AWS Secret Access Key [None]: KEuSaLgn4dpyXe-------------VmEKdhV
Default region name [None]: us-west-2
Default output format [None]: json
```

将密钥替换为之前检索到的密钥，并设置默认区域和数据类型。

# 操作步骤

我们按照以下步骤进行教程：

1.  执行以下命令。这将返回一个命令，用于对接 Docker 和 ECR 进行身份验证：

```py
$ aws ecr get-login --no-include-email --region us-west-2 docker login -u AWS -p eyJwYXlsb2FkIjoiN3BZVWY4Q2JoZkFwYUNKOUp6c1BkRy80VmRYN0Y2LzQ0Y2pVNFJKZTA5alBrUEdSMHlNUk9TMytsTFVURGtxb3Q5VTZqV0xxNmRCVHJnL1FIb2lGbEF0dVZhNFpEOUkxb1FxUTNwcUluaVhqS1FCZmU2WTRLNlQrbjE4VHdiOEpqbmtwWjJJek8xRlR2Y2Y5S3NGRlQrbDZhcktUNXZJbjNkb1czVGQ2TXZPUlg5cE5Ea2w4S29vamt6SE10Ym8rOW5mLzBvVkRRSDlaY3hqRG45d0FzNVA5Z1BPVUU5OVFrTEZGeENPUHJRZmlTeHFqaEVPcGo3ZVAxL3pCNnFTdjVXUEozaUNtV0I0b1lFNEcyVzA4M2hKQmpESUFTV1VMZ1B0MFI2YUlHSHJxTlRvTGZOR1R5clJ2VUZKcnFWZGptMkZlR0ppK3I5emFrdGFKeDJBNVRCUzBzZDZaOG1yeW1Nd0dBVi81NDZDeU1XYVliby9reWtaNUNuZE8zVXFHdHFKSnJmQVRKakhlVU1jTXQ1RjE0Tk83OWR0ckNnYmZmUHdtS1hXOVh6MklWUG5VUlJsekRaUjRMMVFKT2NjNlE0NWFaNkR2enlDRWw1SzVwOEcvK3lSMXFPYzdKUWpxaUErdDZyaCtDNXJCWHlJQndKRm5mcUJhaVhBMVhNMFNocmlNd0FUTXFjZ0NtZTEyUGhOMmM2c0pNTU5hZ0JMNEhXSkwyNXZpQzMyOVI2MytBUWhPNkVaajVMdG9iMVRreFFjbjNGamVNdThPM0ppZnM5WGxPSVJsOHlsUUh0LzFlQ2ZYelQ1cVFOU2g1NjFiVWZtOXNhNFRRWlhZUlNLVVFrd3JFK09EUXh3NUVnTXFTbS9FRm1PbHkxdEpncXNzVFljeUE4Y1VYczFnOFBHL2VwVGtVTG1ReFYwa0p5MzdxUmlIdHU1OWdjMDRmZWFSVGdSekhQcXl0WExzdFpXcTVCeVRZTnhMeVVpZW0yN3JkQWhmaStpUHpMTXV1NGZJa3JjdmlBZFF3dGwrdEVORTNZSVBhUnZJMFN0Q1djN2J2blI2Njg3OEhQZHJKdXlYaTN0czhDYlBXNExOamVCRm8waUt0SktCckJjN0tUZzJEY1d4NlN4b1Vkc2ErdnN4V0N5NWFzeWdMUlBHYVdoNzFwOVhFZWpPZTczNE80Z0l5RklBU0pHR3o1SVRzYVkwbFB6ajNEYW9QMVhOT3dhcDYwcC9Gb0pQMG1ITjNsb202eW1EaDA0WEoxWnZ0K0lkMFJ4bE9lVUt3bzRFZFVMaHJ2enBMOUR4SGI5WFFCMEdNWjFJRlI0MitSb3NMaDVQa0g1RHh1bDJZU0pQMXc0UnVoNUpzUm5rcmF3dHZzSG5PSGd2YVZTeWl5bFR0cFlQY1haVk51NE5iWnkxSzQwOG5XTVhiMFBNQzJ5OHJuNlpVTDA9IiwiZGF0YWtleSI6IkFRRUJBSGo2bGM0WElKdy83bG4wSGMwMERNZWs2R0V4SENiWTRSSXBUTUNJNThJblV3QUFBSDR3ZkFZSktvWklodmNOQVFjR29HOHdiUUlCQURCb0Jna3Foa2lHOXcwQkJ3RXdIZ1lKWUlaSUFXVURCQUV1TUJFRURQdTFQVXQwRDFkN3c3Rys3Z0lCRUlBN21Xay9EZnNOM3R5MS9iRFdRYlZtZjdOOURST2xhQWFFbTBFQVFndy9JYlBjTzhLc0RlNDBCLzhOVnR0YmlFK1FXSDBCaTZmemtCbzNxTkE9IiwidmVyc2lvbiI6IjIiLCJ0eXBlIjoiREFUQV9LRVkiLCJleHBpcmF0aW9uIjoxNTE1NjA2NzM0fQ== https://270157190882.dkr.ecr.us-west-2.amazonaws.com
```

这个输出是一个命令，你需要执行它来让你的 docker CLI 与 ECR 进行身份验证！这个密钥只在几个小时内有效（我相信是十二小时）。你可以从`docker login`开始的位置复制所有内容，一直到密钥末尾的 URL。

1.  在 Mac（和 Linux）上，我通常简化为以下步骤：

```py
$(aws ecr get-login --no-include-email --region us-west-2)
WARNING! Using --password via the CLI is insecure. Use --password-stdin.
Login Succeeded
```

更容易。在这一点上，我们可以使用 docker 命令将容器推送到 ECR。

这是我见过的一些问题的地方。我发现密钥末尾的 URL 可能仍然是根用户，而不是您为 ECR 创建的用户（此登录必须是该用户）。如果是这种情况，后续命令将出现奇怪的身份验证问题。解决方法是删除所有 AWS CLI 配置文件并重新配置。这种解决方法并不总是有效。有时候，我不得不使用一个全新的系统/虚拟机，通过 AWS CLI 安装/配置，然后生成这个密钥才能使其工作。

# 将容器推送到 ECR

在这个食谱中，我们将重建我们的 API 和微服务容器，并将它们推送到 ECR。我们还将 RabbitMQ 容器推送到 ECR。

# 准备就绪

请耐心等待，因为这可能会变得棘手。除了我们的容器镜像之外，我们还需要将 RabbitMQ 容器推送到 ECR。ECS 无法与 Docker Hub 通信，也无法拉取该镜像。这将非常方便，但同时也可能是一个安全问题。

从家庭互联网连接推送这些容器到 ECR 可能需要很长时间。我在 EC2 中创建了一个与我的 ECR 相同地区的 Linux 镜像，从 github 上拉取了代码，在那台 EC2 系统上构建了容器，然后推送到 ECR。如果不是几秒钟的话，推送只需要几分钟。

首先，让我们在本地系统上重建我们的 API 和微服务容器。我已经在`11/05`食谱文件夹中包含了 Python 文件、两个 docker 文件和微服务的配置文件。

让我们从构建 API 容器开始：

```py
$ docker build ../.. -f Dockerfile-api -t scraper-rest-api:latest
```

这个 docker 文件与之前的 API Docker 文件类似，只是修改了从`11/05`文件夹复制文件的部分。

```py
FROM python:3
WORKDIR /usr/src/app

RUN pip install Flask-RESTful Elasticsearch Nameko
COPY 11/11/scraper_api.py .

CMD ["python", "scraper_api.py"]
```

然后构建 scraper 微服务的容器：

```py
$ docker build ../.. -f Dockerfile-microservice -t scraper-microservice:latest
```

这个 Dockerfile 与微服务的 Dockerfile 略有不同。它的内容如下：

```py
FROM python:3
WORKDIR /usr/src/app

RUN pip install nameko BeautifulSoup4 nltk lxml
RUN python -m nltk.downloader punkt -d /usr/share/nltk_data all

COPY 11/05/scraper_microservice.py .
COPY modules/sojobs sojobs

CMD ["python", "-u", "scraper_microservice.py"]
```

现在我们准备好配置 ECR 来存储我们的容器，供 ECS 使用。

我们现在使用 python 而不是“nameko run”命令来运行微服务。这是由于 ECS 中容器启动顺序的问题。如果 RabbitMQ 服务器尚未运行，“nameko run”命令的性能不佳，而在 ECS 中无法保证 RabbitMQ 服务器已经运行。因此，我们使用 python 启动。因此，该实现具有一个启动，基本上是复制“nameko run”的代码，并用 while 循环和异常处理程序包装它，直到容器停止。

# 如何操作

我们按照以下步骤进行：

1.  登录到我们为 ECS 创建的帐户后，我们可以访问弹性容器仓库。这项服务可以保存我们的容器供 ECS 使用。有许多 AWS CLI 命令可以用来处理 ECR。让我们从列出现有仓库的以下命令开始：

```py
$ aws ecr describe-repositories
{
    "repositories": []
}
```

1.  现在我们还没有任何仓库，让我们创建一些。我们将创建三个仓库，分别用于不同的容器：scraper-rest-api、scraper-microservice，以及一个 RabbitMQ 容器，我们将其命名为`rabbitmq`。每个仓库都映射到一个容器，但可以有多个标签（每个最多有 1,000 个不同的版本/标签）。让我们创建这三个仓库：

```py
$ aws ecr create-repository --repository-name scraper-rest-api
{
  "repository": {
    "repositoryArn": "arn:aws:ecr:us-west-2:414704166289:repository/scraper-rest-api",
    "repositoryUri": "414704166289.dkr.ecr.us-west-2.amazonaws.com/scraper-rest-api",
    "repositoryName": "scraper-rest-api",
    "registryId": "414704166289",
    "createdAt": 1515632756.0
  }
}

05 $ aws ecr create-repository --repository-name scraper-microservice
{
  "repository": {
    "repositoryArn": "arn:aws:ecr:us-west-2:414704166289:repository/scraper-microservice",
    "registryId": "414704166289",
    "repositoryName": "scraper-microservice",
    "repositoryUri": "414704166289.dkr.ecr.us-west-2.amazonaws.com/scraper-microservice",
    "createdAt": 1515632772.0
  }
}

05 $ aws ecr create-repository --repository-name rabbitmq
{
  "repository": {
    "repositoryArn": "arn:aws:ecr:us-west-2:414704166289:repository/rabbitmq",
    "repositoryName": "rabbitmq",
    "registryId": "414704166289",
    "createdAt": 1515632780.0,
    "repositoryUri": "414704166289.dkr.ecr.us-west-2.amazonaws.com/rabbitmq"
  }
}
```

注意返回的数据。我们需要在接下来的步骤中使用每个仓库的 URL。

1.  我们需要*标记*我们的本地容器镜像，以便它们的 docker 知道当我们*推送*它们时，它们应该去我们 ECR 中的特定仓库。此时，您的 docker 中应该有以下镜像：

```py
$ docker images
REPOSITORY           TAG          IMAGE ID     CREATED        SIZE
scraper-rest-api     latest       b82653e11635 29 seconds ago 717MB
scraper-microservice latest       efe19d7b5279 11 minutes ago 4.16GB
rabbitmq             3-management 6cb6e2f951a8 2 weeks ago    151MB
python               3            c1e459c00dc3 3 weeks ago    692MB
```

1.  使用`<image-id> <ECR-repository-uri>` docker tag 进行标记。让我们标记所有三个（我们不需要对 python 镜像进行操作）：

```py
$ docker tag b8 414704166289.dkr.ecr.us-west-2.amazonaws.com/scraper-rest-api

$ docker tag ef 414704166289.dkr.ecr.us-west-2.amazonaws.com/scraper-microservice

$ docker tag 6c 414704166289.dkr.ecr.us-west-2.amazonaws.com/rabbitmq
```

1.  现在的 docker 镜像列表中显示了标记的镜像以及原始镜像：

```py
$ docker images
REPOSITORY TAG IMAGE ID CREATED SIZE
414704166289.dkr.ecr.us-west-2.amazonaws.com/scraper-rest-api latest b82653e11635 4 minutes ago 717MB
scraper-rest-api latest b82653e11635 4 minutes ago 717MB
414704166289.dkr.ecr.us-west-2.amazonaws.com/scraper-microservice latest efe19d7b5279 15 minutes ago 4.16GB
scraper-microservice latest efe19d7b5279 15 minutes ago 4.16GB
414704166289.dkr.ecr.us-west-2.amazonaws.com/rabbitmq latest 6cb6e2f951a8 2 weeks ago 151MB
rabbitmq 3-management 6cb6e2f951a8 2 weeks ago 151MB
python 3 c1e459c00dc3 3 weeks ago 692MB
```

1.  现在我们最终将镜像推送到 ECR：

```py
$ docker push 414704166289.dkr.ecr.us-west-2.amazonaws.com/scraper-rest-api
The push refers to repository [414704166289.dkr.ecr.us-west-2.amazonaws.com/scraper-rest-api]
7117db0da9a9: Pushed
8eb1be67ed26: Pushed
5fcc76c4c6c0: Pushed
6dce5c484bde: Pushed
057c34df1f1a: Pushed
3d358bf2f209: Pushed
0870b36b7599: Pushed
8fe6d5dcea45: Pushed
06b8d020c11b: Pushed
b9914afd042f: Pushed
4bcdffd70da2: Pushed
latest: digest: sha256:2fa2ccc0f4141a1473386d3592b751527eaccb37f035aa08ed0c4b6d7abc9139 size: 2634

$ docker push 414704166289.dkr.ecr.us-west-2.amazonaws.com/scraper-microservice
The push refers to repository [414704166289.dkr.ecr.us-west-2.amazonaws.com/scraper-microservice]
3765fccaf6a6: Pushed
4bde7a8212e1: Pushed
d0aa245987b4: Pushed
5657283a8f79: Pushed
4f33694fe63a: Pushed
5fcc76c4c6c0: Pushed
6dce5c484bde: Pushed
057c34df1f1a: Pushed
3d358bf2f209: Pushed
0870b36b7599: Pushed
8fe6d5dcea45: Pushed
06b8d020c11b: Pushed
b9914afd042f: Pushed
4bcdffd70da2: Pushed
latest: digest: sha256:02c1089689fff7175603c86d6ef8dc21ff6aaffadf45735ef754f606f2cf6182 size: 3262

$ docker push 414704166289.dkr.ecr.us-west-2.amazonaws.com/rabbitmq
The push refers to repository [414704166289.dkr.ecr.us-west-2.amazonaws.com/rabbitmq]
e38187f05202: Pushed
ea37471972cd: Pushed
2f1d47e88a53: Pushed
e8c84964de08: Pushed
d0537ac3fb13: Pushed
9f345d60d035: Pushed
b45610229549: Pushed
773afacc96cc: Pushed
5eb8d21fccbb: Pushed
10699a5bd960: Pushed
27be686b9e1f: Pushed
96bfbdb03e1c: Pushed
1709335ba200: Pushed
2ec5c0a4cb57: Pushed
latest: digest: sha256:74308ef1dabc1a0b9615f756d80f5faf388f4fb038660ae42f437be45866b65e size: 3245
```

1.  现在检查镜像是否已经到达仓库。以下是`scraper-rest-api`的情况：

```py
$ aws ecr list-images --repository-name scraper-rest-api
{
  "imageIds": [
    {
      "imageTag": "latest",
      "imageDigest": "sha256:2fa2ccc0f4141a1473386d3592b751527eaccb37f035aa08ed0c4b6d7abc9139"
    }
  ]
}
```

现在我们的容器已经存储在 ECR 中，我们可以继续创建一个集群来运行我们的容器。

# 创建一个 ECS 集群

弹性容器服务（ECS）是 AWS 在云中运行 Docker 容器的服务。使用 ECS 有很多强大的功能（和细节）。我们将看一个简单的部署，它在单个 EC2 虚拟机上运行我们的容器。我们的目标是将我们的爬虫放到云中。关于使用 ECS 扩展爬虫的详细信息将在另一个时间（和书籍）中介绍。

# 如何做到

我们首先使用 AWS CLI 创建一个 ECR 集群。然后我们将在集群中创建一个 EC2 虚拟机来运行我们的容器。

我在`11/06`文件夹中包含了一个 shell 文件，名为`create-cluster-complete.sh`，它可以一次运行所有这些命令。

有许多步骤需要进行配置，但它们都相当简单。让我们一起走过它们：

1.  以下创建了一个名为 scraper-cluster 的 ECR 集群：

```py
$ aws ecs create-cluster --cluster-name scraper-cluster
{
  "cluster": {
    "clusterName": "scraper-cluster",
    "registeredContainerInstancesCount": 0,
    "clusterArn": "arn:aws:ecs:us-west-2:414704166289:cluster/scraper-cluster",
    "status": "ACTIVE",
    "activeServicesCount": 0,
    "pendingTasksCount": 0,
    "runningTasksCount": 0
  }
}
```

哇，这太容易了！嗯，还有一些细节要处理。在这一点上，我们没有任何 EC2 实例来运行容器。我们还需要设置密钥对、安全组、IAM 策略，哎呀！看起来很多，但我们将很快、很容易地完成它。

1.  创建一个密钥对。每个 EC2 实例都需要一个密钥对来启动，并且需要远程连接到实例（如果您想要的话）。以下是创建一个密钥对，将其放入本地文件，然后与 AWS 确认它已创建：

```py
$ aws ec2 create-key-pair --key-name ScraperClusterKP --query 'KeyMaterial' --output text > ScraperClusterKP.pem

$ aws ec2 describe-key-pairs --key-name ScraperClusterKP
{
  "KeyPairs": [
    {
      "KeyFingerprint": "4a:8a:22:fa:53:a7:87:df:c5:17:d9:4f:b1:df:4e:22:48:90:27:2d",
      "KeyName": "ScraperClusterKP"
    }
  ]
}
```

1.  现在我们创建安全组。安全组允许我们从互联网打开端口到集群实例，因此允许我们访问运行在我们的容器中的应用程序。我们将创建一个安全组，其中包括端口 22（ssh）和 80（http），以及 RabbitMQ 的两个端口（5672 和 15672）被打开。我们需要打开 80 端口以与 REST API 进行通信（我们将在下一个步骤中将 80 映射到 8080 容器）。我们不需要打开 15672 和 5672 端口，但它们有助于通过允许您从 AWS 外部连接到 RabbitMQ 来调试该过程。以下四个命令创建了安全组和该组中的规则：

```py
$ aws  ec2  create-security-group  --group-name  ScraperClusterSG  --description  "Scraper Cluster SG”
{
  "GroupId": "sg-5e724022"
} 
$ aws ec2 authorize-security-group-ingress --group-name ScraperClusterSG --protocol tcp --port 22 --cidr 0.0.0.0/0

$ aws ec2 authorize-security-group-ingress --group-name ScraperClusterSG --protocol tcp --port 80 --cidr 0.0.0.0/0

$ aws ec2 authorize-security-group-ingress --group-name ScraperClusterSG --protocol tcp --port 5672 --cidr 0.0.0.0/0

$ aws ec2 authorize-security-group-ingress --group-name ScraperClusterSG --protocol tcp --port 15672 --cidr 0.0.0.0/0
```

您可以使用 aws ec2 describe-security-groups --group-names ScraperClusterSG 命令确认安全组的内容。这将输出该组的 JSON 表示。

1.  要将 EC2 实例启动到 ECS 集群中，需要放置一个 IAM 策略，以允许它进行连接。它还需要具有与 ECR 相关的各种能力，例如拉取容器。这些定义在配方目录中包含的两个文件`ecsPolicy.json`和`rolePolicy.json`中。以下命令将这些策略注册到 IAM（输出被省略）：

```py
$ aws iam create-role --role-name ecsRole --assume-role-policy-document file://ecsPolicy.json

$ aws  iam  put-role-policy  --role-name  ecsRole  --policy-name  ecsRolePolicy  --policy-document  file://rolePolicy.json

$ aws iam create-instance-profile --instance-profile-name ecsRole 
$ aws iam add-role-to-instance-profile --instance-profile-name ecsRole --role-name ecsRole
```

在启动实例之前，我们需要做一件事。我们需要有一个文件将用户数据传递给实例，告诉实例连接到哪个集群。如果我们不这样做，它将连接到名为`default`而不是`scraper-cluster`的集群。这个文件是`userData.txt`在配方目录中。这里没有真正的操作，因为我提供了这个文件。

1.  现在我们在集群中启动一个实例。我们需要使用一个经过优化的 ECS AMI 或创建一个带有 ECS 容器代理的 AMI。我们将使用一个带有此代理的预构建 AMI。以下是启动实例的步骤：

```py
$ aws ec2 run-instances --image-id ami-c9c87cb1 --count 1 --instance-type m4.large --key-name ScraperClusterKP --iam-instance-profile "Name= ecsRole" --security-groups ScraperClusterSG --user-data file://userdata.txt
```

这将输出描述您的实例的一些 JSON。

1.  几分钟后，您可以检查此实例是否在容器中运行：

```py
$ aws ecs list-container-instances --cluster scraper-cluster
{
  "containerInstanceArns": [
    "arn:aws:ecs:us-west-2:414704166289:container-instance/263d9416-305f-46ff-a344-9e7076ca352a"
  ]
}
```

太棒了！现在我们需要定义要在容器实例上运行的任务。

这是一个 m4.large 实例。它比适用于免费层的 t2.micro 大一点。因此，如果您想保持成本低廉，请确保不要让它长时间运行。

# 创建一个运行我们的容器的任务

在这个步骤中，我们将创建一个 ECS 任务。任务告诉 ECR 集群管理器要运行哪些容器。任务是对要在 ECR 中运行的容器以及每个容器所需的参数的描述。任务描述会让我们联想到我们使用 Docker Compose 所做的事情。

# 准备工作

任务定义可以使用 GUI 构建，也可以通过提交任务定义 JSON 文件来启动。我们将使用后一种技术，并检查文件`td.json`的结构，该文件描述了如何一起运行我们的容器。此文件位于`11/07`配方文件夹中。

# 操作步骤

以下命令将任务注册到 ECS：

```py
$ aws ecs register-task-definition --cli-input-json file://td.json
{
  "taskDefinition": {
    "volumes": [

    ],
    "family": "scraper",
    "memory": "4096",
    "placementConstraints": [

    ]
  ],
  "cpu": "1024",
  "containerDefinitions": [
    {
      "name": "rabbitmq",
      "cpu": 0,
      "volumesFrom": [

      ],
      "mountPoints": [

      ],
      "portMappings": [
        {
          "hostPort": 15672,
          "protocol": "tcp",
          "containerPort": 15672
        },
        {
          "hostPort": 5672,
          "protocol": "tcp",
          "containerPort": 5672
        }
      ],
      "environment": [

      ],
      "image": "414704166289.dkr.ecr.us-west-2.amazonaws.com/rabbitmq",
      "memory": 256,
      "essential": true
    },
    {
      "name": "scraper-microservice",
      "cpu": 0,
      "essential": true,
      "volumesFrom": [

      ],
      "mountPoints": [

      ],
      "portMappings": [

      ],
      "environment": [
        {
          "name": "AMQP_URI",
          "value": "pyamqp://guest:guest@rabbitmq"
        }
      ],
      "image": "414704166289.dkr.ecr.us-west-2.amazonaws.com/scraper-microservice",
      "memory": 256,
      "links": [
        "rabbitmq"
      ]
    },
    {
      "name": "api",
      "cpu": 0,
      "essential": true,
      "volumesFrom": [

      ],
      "mountPoints": [

      ],
      "portMappings": [
        {
          "hostPort": 80,
          "protocol": "tcp",
          "containerPort": 8080
        }
      ],
      "environment": [
        {
          "name": "AMQP_URI",
          "value": "pyamqp://guest:guest@rabbitmq"
        },
        {
          "name": "ES_HOST",
          "value": "https://elastic:tduhdExunhEWPjSuH73O6yLS@7dc72d3327076cc4daf5528103c46a27.us-west-2.aws.found.io:9243"
        }
      ],
      "image": "414704166289.dkr.ecr.us-west-2.amazonaws.com/scraper-rest-api",
      "memory": 128,
      "links": [
        "rabbitmq"
      ]
    }
  ],
  "requiresCompatibilities": [
    "EC2"
  ],
  "status": "ACTIVE",
  "taskDefinitionArn": "arn:aws:ecs:us-west-2:414704166289:task-definition/scraper:7",
  "requiresAttributes": [
    {
      "name": "com.amazonaws.ecs.capability.ecr-auth"
    }
  ],
  "revision": 7,
  "compatibilities": [
    "EC2"
  ]
}
```

输出是由 ECS 填写的任务定义，并确认接收到任务定义。

# 它是如何工作的

任务定义由两个主要部分组成。第一部分提供有关整体任务的一些一般信息，例如为整个容器允许多少内存和 CPU。然后它包括一个定义我们将运行的三个容器的部分。

文件以定义整体设置的几行开头：

```py
{
    "family": "scraper-as-a-service",
  "requiresCompatibilities": [
        "EC2"
  ],
  "cpu": "1024",
  "memory": "4096",
  "volumes": [], 
```

任务的实际名称由``"family"``属性定义。我们声明我们的容器需要 EC2（任务可以在没有 EC2 的情况下运行-我们的任务需要它）。然后我们声明我们希望将整个任务限制为指定的 CPU 和内存量，并且我们不附加任何卷。

现在让我们来看一下定义容器的部分。它以以下内容开始：

```py
"containerDefinitions": [
```

现在让我们逐个检查每个容器的定义。以下是`rabbitmq`容器的定义：

```py
{
    "name": "rabbitmq",
  "image": "414704166289.dkr.ecr.us-west-2.amazonaws.com/rabbitmq",   "cpu": 0,
  "memory": 256,
  "portMappings": [
        {
            "containerPort": 15672,
  "hostPort": 15672,
  "protocol": "tcp"
  },
  {
            "containerPort": 5672,
  "hostPort": 5672,
  "protocol": "tcp"
  }
    ],
  "essential": true },
```

第一行定义了容器的名称，此名称还参与 API 和 scraper 容器通过 DNS 解析此容器的名称。图像标签定义了要为容器拉取的 ECR 存储库 URI。

确保将此容器和其他两个容器的图像 URL 更改为您的存储库的图像 URL。

接下来是定义允许为此容器分配的最大 CPU（0 表示无限）和内存。端口映射定义了容器主机（我们在集群中创建的 EC2 实例）和容器之间的映射。我们映射了两个 RabbitMQ 端口。

基本标签表示此容器必须保持运行。如果失败，整个任务将被停止。

接下来定义的容器是 scraper 微服务：

```py
{
    "name": "scraper-microservice",
  "image": "414704166289.dkr.ecr.us-west-2.amazonaws.com/scraper-microservice",
  "cpu": 0,
  "memory": 256,
  "essential": true,
  "environment": [
        {
            "name": "AMQP_URI",
  "value": "pyamqp://guest:guest@rabbitmq"
  }
    ],
  "links": [
        "rabbitmq"
  ]
},
```

这与具有环境变量和链接定义的不同。环境变量是`rabbitmq`容器的 URL。ECS 将确保在此容器中将环境变量设置为此值（实现 Factor-3）。虽然这与我们在本地使用 docker compose 运行时的 URL 相同，但如果`rabbitmq`容器的名称不同或在另一个集群上，它可能是不同的 URL。

链接设置需要一点解释。链接是 Docker 的一个已弃用功能，但在 ECS 中仍在使用。在 ECS 中，它们是必需的，以便容器解析同一集群网络中其他容器的 DNS 名称。这告诉 ECS，当此容器尝试解析`rabbitmq`主机名（如环境变量中定义的那样）时，它应返回分配给该容器的 IP 地址。

文件的其余部分定义了 API 容器：

```py
{
  "name": "api",
  "image": "414704166289.dkr.ecr.us-west-2.amazonaws.com/scraper-rest-api",
  "cpu": 0,
  "memory": 128,
  "essential": true,
  "portMappings": [
    {
      "containerPort": 8080,
      "hostPort": 80,
      "protocol": "tcp"
    }
  ],
  "environment": [
    {
      "name": "AMQP_URI",
      "value": "pyamqp://guest:guest@rabbitmq"
    },
    {
      "name": "ES_HOST",
      "value": "https://elastic:tduhdExunhEWPjSuH73O6yLS@7dc72d3327076cc4daf5528103c46a27.us-west-2.aws.found.io:9243"
    }
  ],
  "links": [
    "rabbitmq"
  ]
}
    ]
}
```

在此定义中，我们定义了端口映射以允许 HTTP 进入容器，并设置了 API 用于与 Elastic Cloud 和`rabbitmq`服务器通信的环境变量（该服务器将请求传递给`scraper-microservice`容器）。这还定义了对`rabbitmq`的链接，因为也需要解析。

# 在 AWS 中启动和访问容器

在此配方中，我们将通过告知 ECS 运行我们的任务定义来将我们的 scraper 作为服务启动。然后，我们将通过发出 curl 来检查它是否正在运行，以获取作业列表的内容。

# 准备工作

在运行任务之前，我们需要做一件事。ECS 中的任务经历多次修订。每次您使用相同名称（“family”）注册任务定义时，ECS 都会定义一个新的修订号。您可以运行任何修订版本。

要运行最新的版本，我们需要列出该 family 的任务定义，并找到最新的修订号。以下列出了集群中的所有任务定义。此时我们只有一个：

```py
$ aws ecs list-task-definitions
{
  "taskDefinitionArns": [
    "arn:aws:ecs:us-west-2:414704166289:task-definition/scraper-as-a-service:17"
  ]
}
```

请注意我的修订号是 17。虽然这是我当前唯一注册的此任务的版本，但我已经注册（和注销）了 16 个之前的修订版本。

# 如何做

我们按照以下步骤进行：

1.  现在我们可以运行我们的任务。我们可以使用以下命令来完成这个操作：

```py
$ aws  ecs  run-task  --cluster  scraper-cluster  --task-definition scraper-as-a-service:17  --count  1
{
  "tasks": [
    {
      "taskArn": "arn:aws:ecs:us-west-2:414704166289:task/00d7b868-1b99-4b54-9f2a-0d5d0ae75197",
      "version": 1,
      "group": "family:scraper-as-a-service",
      "containerInstanceArn": "arn:aws:ecs:us-west-2:414704166289:container-instance/5959fd63-7fd6-4f0e-92aa-ea136dabd762",
      "taskDefinitionArn": "arn:aws:ecs:us-west-2:414704166289:task-definition/scraper-as-a-service:17",
      "containers": [
        {
          "name": "rabbitmq",
          "containerArn": "arn:aws:ecs:us-west-2:414704166289:container/4b14d4d5-422c-4ffa-a64c-476a983ec43b",
          "lastStatus": "PENDING",
          "taskArn": "arn:aws:ecs:us-west-2:414704166289:task/00d7b868-1b99-4b54-9f2a-0d5d0ae75197",
          "networkInterfaces": [

          ]
        },
        {
          "name": "scraper-microservice",
          "containerArn": "arn:aws:ecs:us-west-2:414704166289:container/511b39d2-5104-4962-a859-86fdd46568a9",
          "lastStatus": "PENDING",
          "taskArn": "arn:aws:ecs:us-west-2:414704166289:task/00d7b868-1b99-4b54-9f2a-0d5d0ae75197",
          "networkInterfaces": [

          ]
        },
        {
          "name": "api",
          "containerArn": "arn:aws:ecs:us-west-2:414704166289:container/0e660af7-e2e8-4707-b04b-b8df18bc335b",
          "lastStatus": "PENDING",
          "taskArn": "arn:aws:ecs:us-west-2:414704166289:task/00d7b868-1b99-4b54-9f2a-0d5d0ae75197",
          "networkInterfaces": [

          ]
        }
      ],
      "launchType": "EC2",
      "overrides": {
        "containerOverrides": [
          {
            "name": "rabbitmq"
          },
          {
            "name": "scraper-microservice"
          },
          {
            "name": "api"
          }
        ]
      },
      "lastStatus": "PENDING",
      "createdAt": 1515739041.287,
      "clusterArn": "arn:aws:ecs:us-west-2:414704166289:cluster/scraper-cluster",
      "memory": "4096",
      "cpu": "1024",
      "desiredStatus": "RUNNING",
      "attachments": [

      ]
    }
  ],
  "failures": [

  ]
} 
```

输出给我们提供了任务的当前状态。第一次运行时，它需要一些时间来启动，因为容器正在复制到 EC2 实例上。造成延迟的主要原因是带有所有 NLTK 数据的`scraper-microservice`容器。

1.  您可以使用以下命令检查任务的状态：

```py
$ aws  ecs  describe-tasks  --cluster  scraper-cluster  --task 00d7b868-1b99-4b54-9f2a-0d5d0ae75197
```

您需要更改任务 GUID 以匹配从运行任务的输出的`"taskArn"`属性中获取的 GUID。当所有容器都在运行时，我们就可以测试 API 了。

1.  调用我们的服务，我们需要找到集群实例的 IP 地址或 DNS 名称。您可以从我们创建集群时的输出中获取这些信息，也可以通过门户或以下命令获取。首先，描述集群实例：

```py
$ aws ecs list-container-instances --cluster scraper-cluster
{
  "containerInstanceArns": [
    "arn:aws:ecs:us-west-2:414704166289:container-instance/5959fd63-7fd6-4f0e-92aa-ea136dabd762"
  ]
}
```

1.  使用我们 EC2 实例的 GUID，我们可以查询其信息并使用以下命令获取 EC2 实例 ID：

```py
$ aws ecs describe-container-instances --cluster scraper-cluster --container-instances 5959fd63-7fd6-4f0e-92aa-ea136dabd762 | grep "ec2InstanceId"
            "ec2InstanceId": "i-08614daf41a9ab8a2",
```

1.  有了那个实例 ID，我们可以获取 DNS 名称：

```py
$ aws ec2 describe-instances --instance-ids i-08614daf41a9ab8a2 | grep "PublicDnsName"
                    "PublicDnsName": "ec2-52-27-26-220.us-west-2.compute.amazonaws.com",
                                        "PublicDnsName": "ec2-52-27-26-220.us-west-2.compute.amazonaws.com"
                                "PublicDnsName": "ec2-52-27-26-220.us-west-2.compute.amazonaws.com"
```

1.  有了那个 DNS 名称，我们可以使用 curl 来获取作业列表：

```py
$ curl ec2-52-27-26-220.us-west-2.compute.amazonaws.com/joblisting/122517 | head -n 6
```

然后我们得到了以下熟悉的结果！

```py
{
  "ID": "122517",
  "JSON": {
    "@context": "http://schema.org",
    "@type": "JobPosting",
    "title": "SpaceX Enterprise Software Engineer, Full Stack",
```

我们的爬虫现在正在云端运行！

# 还有更多...

我们的爬虫正在一个`m4.large`实例上运行，所以我们想要关闭它，以免超出免费使用额度。这是一个两步过程。首先，需要终止集群中的 EC2 实例，然后删除集群。请注意，删除集群不会终止 EC2 实例。

我们可以使用以下命令终止 EC2 实例（以及我们刚刚从集群询问中获取的实例 ID）：

```py
$ aws ec2 terminate-instances --instance-ids i-08614daf41a9ab8a2
{
  "TerminatingInstances": [
    {
      "CurrentState": {
        "Name": "shutting-down",
        "Code": 32
      },
      "PreviousState": {
        "Name": "running",
        "Code": 16
      },
      "InstanceId": "i-08614daf41a9ab8a2"
    }
  ]
}
```

集群可以使用以下命令删除：

```py
$ aws ecs delete-cluster --cluster scraper-cluster
{
  "cluster": {
    "activeServicesCount": 0,
    "pendingTasksCount": 0,
    "clusterArn": "arn:aws:ecs:us-west-2:414704166289:cluster/scraper-cluster",
    "runningTasksCount": 0,
    "clusterName": "scraper-cluster",
    "registeredContainerInstancesCount": 0,
    "status": "INACTIVE"
  }
}
```
