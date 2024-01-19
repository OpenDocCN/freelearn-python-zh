# 创建一个简单的数据 API

在本章中，我们将涵盖：

+   使用 Flask-RESTful 创建 REST API

+   将 REST API 与抓取代码集成

+   添加一个用于查找工作列表技能的 API

+   将数据存储在 Elasticsearch 中作为抓取请求的结果

+   在抓取之前检查 Elasticsearch 中的列表

# 介绍

我们现在已经达到了学习抓取的一个激动人心的转折点。从现在开始，我们将学习使用几个 API、微服务和容器工具将抓取器作为服务运行，所有这些都将允许在本地或云中运行抓取器，并通过标准化的 REST API 访问抓取器。

我们将在本章中开始这个新的旅程，使用 Flask-RESTful 创建一个简单的 REST API，最终我们将使用它来对服务进行页面抓取请求。我们将把这个 API 连接到一个 Python 模块中实现的抓取器功能，该模块重用了在第七章中讨论的从 StackOverflow 工作中抓取的概念，*文本整理和分析*。

最后几个食谱将重点介绍将 Elasticsearch 用作这些结果的缓存，存储我们从抓取器中检索的文档，然后首先在缓存中查找它们。我们将在第十一章中进一步研究 ElasticCache 的更复杂用法，比如使用给定技能集进行工作搜索，*使抓取器成为真正的服务*。

# 使用 Flask-RESTful 创建 REST API

我们从使用 Flask-RESTful 创建一个简单的 REST API 开始。这个初始 API 将由一个单一的方法组成，让调用者传递一个整数值，并返回一个 JSON 块。在这个食谱中，参数及其值以及返回值在这个时候并不重要，因为我们首先要简单地使用 Flask-RESTful 来运行一个 API。

# 准备工作

Flask 是一个 Web 微框架，可以让创建简单的 Web 应用功能变得非常容易。Flask-RESTful 是 Flask 的一个扩展，可以让创建 REST API 同样简单。您可以在`flask.pocoo.org`上获取 Flask 并了解更多信息。Flask-RESTful 可以在`https://flask-restful.readthedocs.io/en/latest/`上了解。可以使用`pip install flask`将 Flask 安装到您的 Python 环境中。Flask-RESTful 也可以使用`pip install flask-restful`进行安装。

本书中其余的食谱将在章节目录的子文件夹中。这是因为这些食谱中的大多数要么需要多个文件来操作，要么使用相同的文件名（即：`apy.py`）。

# 如何做

初始 API 实现在`09/01/api.py`中。API 本身和 API 的逻辑都在这个单一文件`api.py`中实现。API 可以以两种方式运行，第一种方式是简单地将文件作为 Python 脚本执行。

然后可以使用以下命令启动 API：

```py
python api.py
```

运行时，您将首先看到类似以下的输出：

```py
Starting the job listing API
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
Starting the job listing API
 * Debugger is active!
 * Debugger pin code: 362-310-034
```

该程序在`127.0.0.1:5000`上公开了一个 REST API，我们可以使用`GET`请求到路径`/joblisting/<joblistingid>`来请求工作列表。我们可以使用 curl 尝试一下：

```py
curl localhost:5000/joblisting/1
```

此命令的结果将如下：

```py
{
 "YouRequestedJobWithId": "1"
}
```

就像这样，我们有一个正在运行的 REST API。现在让我们看看它是如何实现的。

# 它是如何工作的

实际上并没有太多的代码，这就是 Flask-RESTful 的美妙之处。代码以导入`flask`和`flask_restful`开始。

```py
from flask import Flask
from flask_restful import Resource, Api
```

接下来是用于设置 Flask-RESTful 的初始配置的代码：

```py
app = Flask(__name__)
api = Api(app)
```

接下来是一个代表我们 API 实现的类的定义：

```py
class JobListing(Resource):
    def get(self, job_listing_id):
        print("Request for job listing with id: " + job_listing_id)
        return {'YouRequestedJobWithId': job_listing_id}
```

Flask-RESTful 将映射 HTTP 请求到这个类的方法。具体来说，按照惯例，`GET`请求将映射到名为`get`的成员函数。将 URL 的值映射到函数的`jobListingId`参数。然后，该函数返回一个 Python 字典，Flask-RESTful 将其转换为 JSON。

下一行代码告诉 Flask-RESTful 如何将 URL 的部分映射到我们的类：

```py
api.add_resource(JobListing, '/', '/joblisting/<string:job_listing_id>')
```

这定义了以`/joblisting`开头的路径的 URL 将映射到我们的`JobListing`类，并且 URL 的下一部分表示要传递给`get`方法的`jobListingId`参数的字符串。由于在此映射中未定义其他动词，因此假定使用 GET HTTP 动词。

最后，我们有一段代码，指定了当文件作为脚本运行时，我们只需执行`app.run()`（在这种情况下传递一个参数以便获得调试输出）。

```py
if __name__ == '__main__':
    print("Starting the job listing API")
    app.run(debug=True)
```

然后，Flask-RESTful 找到我们的类并设置映射，开始在`127.0.0.1:5000`（默认值）上监听，并将请求转发到我们的类和方法。

# 还有更多...

Flask-RESTful 的默认运行端口是`5000`。可以使用`app.run()`的替代形式来更改。对于我们的食谱，将其保留在 5000 上就可以了。最终，您会在类似容器的东西中运行此服务，并在前面使用诸如 NGINX 之类的反向代理，并执行公共端口映射到内部服务端口。

# 将 REST API 与抓取代码集成

在这个食谱中，我们将把我们为从 StackOverflow 获取干净的工作列表编写的代码与我们的 API 集成。这将导致一个可重用的 API，可以用来执行按需抓取，而客户端无需了解抓取过程。基本上，我们将创建一个*作为服务的抓取器*，这是我们在本书的其余食谱中将花费大量时间的概念。

# 准备工作

这个过程的第一部分是将我们在第七章中编写的现有代码创建为一个模块，以便我们可以重用它。我们将在本书的其余部分中的几个食谱中重用这段代码。在将其与 API 集成之前，让我们简要地检查一下这个模块的结构和内容。

该模块的代码位于项目的模块文件夹中的`sojobs`（用于 StackOverflow 职位）模块中。

![](img/c192716d-ac40-476d-936a-36091d78b2fb.png)sojobs 文件夹

在大多数情况下，这些文件是从第七章中使用的文件复制而来，即*文本整理和分析*。可重用的主要文件是`scraping.py`，其中包含几个函数，用于方便抓取。在这个食谱中，我们将使用的函数是`get_job_listing_info`：

```py
def get_job_listing(job_listing_id):
    print("Got a request for a job listing with id: " + job_listing_id)

    req = requests.get("https://stackoverflow.com/jobs/" + job_listing_id)
    content = req.text

    bs = BeautifulSoup(content, "lxml")
    script_tag = bs.find("script", {"type": "application/ld+json"})

    job_listing_contents = json.loads(script_tag.contents[0])
    desc_bs = BeautifulSoup(job_listing_contents["description"], "lxml")
    just_text = desc_bs.find_all(text=True)

    joined = ' '.join(just_text)
    tokens = word_tokenize(joined)

    stop_list = stopwords.words('english')
    with_no_stops = [word for word in tokens if word.lower() not in stop_list]
    two_grammed = tech_2grams(with_no_stops)
    cleaned = remove_punctuation(two_grammed)

    result = {
        "ID": job_listing_id,
        "JSON": job_listing_contents,
        "TextOnly": just_text,
        "CleanedWords": cleaned
    }

    return json.dumps(result)
```

回到第七章中的代码，您可以看到这段代码是我们在那些食谱中创建的重用代码。不同之处在于，这个函数不是读取单个本地的`.html`文件，而是传递了一个工作列表的标识符，然后构造了该工作列表的 URL，使用 requests 读取内容，执行了几项分析，然后返回结果。

请注意，该函数返回一个 Python 字典，其中包含请求的工作 ID、原始 HTML、列表的文本和清理后的单词列表。该 API 将这些结果聚合返回给调用者，其中包括`ID`，因此很容易知道请求的工作，以及我们执行各种清理的所有其他结果。因此，我们已经创建了一个增值服务，用于工作列表，而不仅仅是获取原始 HTML。

确保你的 PYTHONPATH 环境变量指向模块目录，或者你已经设置好你的 Python IDE 以在这个目录中找到模块。否则，你将会得到找不到这个模块的错误。

# 如何做

我们按以下步骤进行食谱：

1.  这个食谱的 API 代码在`09/02/api.py`中。这扩展了上一个食谱中的代码，以调用`sojobs`模块中的这个函数。服务的代码如下：

```py
from flask import Flask
from flask_restful import Resource, Api
from sojobs.scraping import get_job_listing_info

app = Flask(__name__)
api = Api(app)

class JobListing(Resource):
    def get(self, job_listing_id):
        print("Request for job listing with id: " + job_listing_id)
        listing = get_job_listing_info(job_listing_id)
        print("Got the following listing as a response: " + listing)
        return listing

api.add_resource(JobListing, '/', '/joblisting/<string:job_listing_id>')

if __name__ == '__main__':
    print("Starting the job listing API")
    app.run(debug=True)
```

请注意，主要的区别是从模块导入函数，并调用函数并从结果返回数据。

1.  通过执行带有 Python `api.py`的脚本来运行服务。然后我们可以使用`curl`测试 API。以下请求我们之前检查过的 SpaceX 工作列表。

```py
curl localhost:5000/joblisting/122517
```

1.  这导致了相当多的输出。以下是部分响应的开头：

```py
"{\"ID\": \"122517\", \"JSON\": {\"@context\": \"http://schema.org\", \"@type\": \"JobPosting\", \"title\": \"SpaceX Enterprise Software Engineer, Full Stack\", \"skills\": [\"c#\", \"sql\", \"javascript\", \"asp.net\", \"angularjs\"], \"description\": \"<h2>About this job</h2>\\r\\n<p><span>Location options: <strong>Paid relocation</strong></span><br/><span>Job type: <strong>Permanent</strong></span><br/><span>Experience level: <strong>Mid-Level, Senior</strong></span><br/><span>Role: <strong>Full Stack Developer</strong></span><br/><span>Industry: <strong>Aerospace, Information Technology, Web Development</strong></span><br/><span>Company size: <strong>1k-5k people</strong></span><br/><span>Company type: <strong>Private</strong></span><br/></p><br/><br/><h2>Technologies</h2> <p>c#, sql, javascr
```

# 添加一个 API 来查找工作列表的技能

在这个食谱中，我们向我们的 API 添加了一个额外的操作，允许我们请求与工作列表相关的技能。这演示了一种能够检索数据的子集而不是整个列表内容的方法。虽然我们只对技能做了这个操作，但这个概念可以很容易地扩展到任何其他数据的子集，比如工作的位置、标题，或者几乎任何对 API 用户有意义的其他内容。

# 准备工作

我们要做的第一件事是向`sojobs`模块添加一个爬取函数。这个函数将被命名为`get_job_listing_skills`。以下是这个函数的代码：

```py
def get_job_listing_skills(job_listing_id):
    print("Got a request for a job listing skills with id: " + job_listing_id)

    req = requests.get("https://stackoverflow.com/jobs/" + job_listing_id)
    content = req.text

    bs = BeautifulSoup(content, "lxml")
    script_tag = bs.find("script", {"type": "application/ld+json"})

    job_listing_contents = json.loads(script_tag.contents[0])
    skills = job_listing_contents['skills']

    return json.dumps(skills)
```

这个函数检索工作列表，提取 StackOverflow 提供的 JSON，然后只返回 JSON 的`skills`属性。

现在，让我们看看如何添加一个方法来调用 REST API。

# 如何做

我们按以下步骤进行食谱：

1.  这个食谱的 API 代码在`09/03/api.py`中。这个脚本添加了一个额外的类`JobListingSkills`，具体实现如下：

```py
class JobListingSkills(Resource):
    def get(self, job_listing_id):
        print("Request for job listing's skills with id: " + job_listing_id)
        skills = get_job_listing_skills(job_listing_id)
        print("Got the following skills as a response: " + skills)
        return skills
```

这个实现与上一个食谱类似，只是调用了获取技能的新函数。

1.  我们仍然需要添加一个语句来告诉 Flask-RESTful 如何将 URL 映射到这个类的`get`方法。因为我们实际上是在检索整个工作列表的子属性，我们将扩展我们的 URL 方案，包括一个额外的段代表整体工作列表资源的子属性。

```py
api.add_resource(JobListingSkills, '/', '/joblisting/<string:job_listing_id>/skills')
```

1.  现在我们可以使用以下 curl 仅检索技能：

```py
curl localhost:5000/joblisting/122517/skills
```

这给我们带来了以下结果：

```py
"[\"c#\", \"sql\", \"javascript\", \"asp.net\", \"angularjs\"]"
```

# 将数据存储在 Elasticsearch 中作为爬取请求的结果

在这个食谱中，我们扩展了我们的 API，将我们从爬虫那里收到的数据保存到 Elasticsearch 中。我们稍后会使用这个（在下一个食谱中）来通过使用 Elasticsearch 中的内容来优化请求，以便我们不会重复爬取已经爬取过的工作列表。因此，我们可以与 StackOverflow 的服务器友好相处。

# 准备工作

确保你的 Elasticsearch 在本地运行，因为代码将访问`localhost:9200`上的 Elasticsearch。有一个很好的快速入门可用于 [`www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html`](https://www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html)，或者你可以在 第十章 中查看 Docker Elasticsearch 食谱，*使用 Docker 创建爬虫微服务*，如果你想在 Docker 中运行它。

安装后，你可以使用以下`curl`检查正确的安装：

```py
curl 127.0.0.1:9200?pretty
```

如果安装正确，你将得到类似以下的输出：

```py
{
 "name": "KHhxNlz",
 "cluster_name": "elasticsearch",
 "cluster_uuid": "fA1qyp78TB623C8IKXgT4g",
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
```

您还需要安装 elasticsearch-py。它可以在[`www.elastic.co/guide/en/elasticsearch/client/python-api/current/index.html`](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/index.html)找到，但可以使用`pip install elasticsearch`快速安装。

# 如何做到的

我们将对我们的 API 代码进行一些小的更改。之前的代码已经复制到`09/04/api.py`中，并进行了一些修改。

1.  首先，我们为 elasticsearch-py 添加了一个导入：

```py
from elasticsearch import Elasticsearch
```

1.  现在我们对`JobListing`类的`get`方法进行了快速修改（我在 JobListingSkills 中也做了同样的修改，但出于简洁起见，这里省略了）：

```py
class JobListing(Resource):
    def get(self, job_listing_id):
        print("Request for job listing with id: " + job_listing_id)
        listing = get_job_listing_info(job_listing_id)

        es = Elasticsearch()
        es.index(index='joblistings', doc_type='job-listing', id=job_listing_id, body=listing)

        print("Got the following listing as a response: " + listing)
        return listing
```

1.  这两行新代码创建了一个`Elasticsearch`对象，然后将结果文档插入到 ElasticSearch 中。在第一次调用 API 之前，我们可以通过以下 curl 看到没有内容，也没有`'joblistings'`索引：

```py
curl localhost:9200/joblistings
```

1.  考虑到我们刚刚安装了 Elasticsearch，这将导致以下错误。

```py
{"error":{"root_cause":[{"type":"index_not_found_exception","reason":"no such index","resource.type":"index_or_alias","resource.id":"joblistings","index_uuid":"_na_","index":"joblistings"}],"type":"index_not_found_exception","reason":"no such index","resource.type":"index_or_alias","resource.id":"joblistings","index_uuid":"_na_","index":"joblistings"},"status":404}
```

1.  现在通过`python api.py`启动 API。然后发出`curl`以获取作业列表（`curl localhost:5000/joblisting/122517`）。这将导致类似于之前的配方的输出。现在的区别是这个文档将存储在 Elasticsearch 中。

1.  现在重新发出先前的 curl 以获取索引：

```py
curl localhost:9200/joblistings
```

1.  现在你会得到以下结果（只显示前几行）：

```py
{
 "joblistings": {
  "aliases": {},
  "mappings": {
   "job-listing": {
     "properties": {
       "CleanedWords" {
         "type": "text",
         "fields": {
           "keyword": {
           "type": "keyword",
           "ignore_above": 256
          }
        }
       },
     "ID": {
       "type": "text",
       "fields": {
         "keyword": {
         "type": "keyword",
         "ignore_above": 256
        }
      }
    },
```

已经创建了一个名为`joblistings`的索引，这个结果展示了 Elasticsearch 通过检查文档识别出的索引结构。

虽然 Elasticsearch 是无模式的，但它会检查提交的文档并根据所找到的内容构建索引。

1.  我们刚刚存储的特定文档可以通过以下 curl 检索：

```py
curl localhost:9200/joblistings/job-listing/122517
```

1.  这将给我们以下结果（同样，只显示内容的开头）：

```py
{
 "_index": "joblistings",
 "_type": "job-listing",
 "_id": "122517",
 "_version": 1,
 "found": true,
 "_source": {
  "ID": "122517",
  "JSON": {
   "@context": "http://schema.org",
   "@type": "JobPosting",
   "title": "SpaceX Enterprise Software Engineer, Full Stack",
   "skills": [
    "c#",
    "sql",
    "javascript",
    "asp.net",
    "angularjs"
  ],
  "description": "<h2>About this job</h2>\r\n<p><span>Location options: <strong>Paid relocation</strong></span><br/><span>Job type: <strong>Permanent</strong></span><br/><span>Experience level: <strong>Mid-Level,
```

就像这样，只用两行代码，我们就将文档存储在了 Elasticsearch 数据库中。现在让我们简要地看一下这是如何工作的。

# 它是如何工作的

使用以下行执行了文档的存储：

```py
es.index(index='joblistings', doc_type='job-listing', id=job_listing_id, body=listing)
```

让我们检查每个参数相对于存储这个文档的作用。

`index`参数指定我们要将文档存储在其中的 Elasticsearch 索引。它的名称是`joblistings`。这也成为用于检索文档的 URL 的第一部分。

每个 Elasticsearch 索引也可以有多个文档“类型”，这些类型是逻辑上的文档集合，可以表示索引内不同类型的文档。我们使用了`'job-listing'`，这个值也构成了用于检索特定文档的 URL 的第二部分。

Elasticsearch 不要求为每个文档指定标识符，但如果我们提供一个，我们可以查找特定的文档而不必进行搜索。我们将使用文档 ID 作为作业列表 ID。

最后一个参数`body`指定文档的实际内容。这段代码只是传递了从爬虫接收到的结果。

# 还有更多...

让我们简要地看一下 Elasticsearch 通过查看文档检索的结果为我们做了什么。

首先，我们可以在结果的前几行看到索引、文档类型和 ID：

```py
{
 "_index": "joblistings",
 "_type": "job-listing",
 "_id": "122517",
```

当使用这三个值进行查询时，文档的检索非常高效。

每个文档也存储了一个版本，这种情况下是 1。

```py
    "_version": 1,
```

如果我们使用相同的代码进行相同的查询，那么这个文档将再次存储，具有相同的索引、文档类型和 ID，因此版本将增加。相信我，再次对 API 进行 curl，你会看到这个版本增加到 2。

现在检查``"JSON"``属性的前几个属性的内容。我们将 API 返回的结果的此属性分配为嵌入在 HTML 中的 StackOverflow 作业描述的 JSON。

```py
 "JSON": {
  "@context": "http://schema.org",
  "@type": "JobPosting",
  "title": "SpaceX Enterprise Software Engineer, Full Stack",
  "skills": [
   "c#",
   "sql",
   "javascript",
   "asp.net",
   "angularjs"
  ],
```

这就是像 StackOverflow 这样的网站给我们提供结构化数据的美妙之处，使用 Elasticsearch 等工具，我们可以得到结构良好的数据。我们可以并且将利用这一点，只需很少量的代码就可以产生很大的效果。我们可以轻松地使用 Elasticsearch 执行查询，以识别基于特定技能（我们将在即将到来的示例中执行此操作）、行业、工作福利和其他属性的工作列表。

我们的 API 的结果还返回了一个名为`CleanedWords`的属性，这是我们的几个 NLP 过程提取高价值词语和术语的结果。以下是最终存储在 Elasticsearch 中的值的摘录：

```py
 "CleanedWords": [
  "job",
  "Location",
  "options",
  "Paid relocation",
  "Job",
  "type",
  "Permanent",
  "Experience",
  "level",
```

而且，我们将能够使用这些来执行丰富的查询，帮助我们根据这些特定词语找到特定的匹配项。

# 在爬取之前检查 Elasticsearch 中是否存在列表

现在让我们通过检查是否已经存储了工作列表来利用 Elasticsearch 作为缓存，因此不需要再次访问 StackOverflow。我们扩展 API 以执行对工作列表的爬取，首先搜索 Elasticsearch，如果结果在那里找到，我们返回该数据。因此，我们通过将 Elasticsearch 作为工作列表缓存来优化这个过程。

# 如何做

我们按照以下步骤进行：

这个示例的代码在`09/05/api.py`中。`JobListing`类现在有以下实现：

```py
class JobListing(Resource):
    def get(self, job_listing_id):
        print("Request for job listing with id: " + job_listing_id)

        es = Elasticsearch()
        if (es.exists(index='joblistings', doc_type='job-listing', id=job_listing_id)):
            print('Found the document in ElasticSearch')
            doc =  es.get(index='joblistings', doc_type='job-listing', id=job_listing_id)
            return doc['_source']

        listing = get_job_listing_info(job_listing_id)
        es.index(index='joblistings', doc_type='job-listing', id=job_listing_id, body=listing)

        print("Got the following listing as a response: " + listing)
        return listing
```

在调用爬虫代码之前，API 会检查文档是否已经存在于 Elasticsearch 中。这是通过名为`exists`的方法执行的，我们将要获取的索引、文档类型和 ID 传递给它。

如果返回 true，则使用 Elasticsearch 对象的`get`方法检索文档，该方法也具有相同的参数。这将返回一个表示 Elasticsearch 文档的 Python 字典，而不是我们存储的实际数据。实际的数据/文档是通过访问字典的`'_source'`键来引用的。

# 还有更多...

`JobListingSkills` API 实现遵循了稍微不同的模式。以下是它的代码：

```py
class JobListingSkills(Resource):
    def get(self, job_listing_id):
        print("Request for job listing's skills with id: " + job_listing_id)

        es = Elasticsearch()
        if (es.exists(index='joblistings', doc_type='job-listing', id=job_listing_id)):
            print('Found the document in ElasticSearch')
            doc =  es.get(index='joblistings', doc_type='job-listing', id=job_listing_id)
            return doc['_source']['JSON']['skills']

        skills = get_job_listing_skills(job_listing_id)

        print("Got the following skills as a response: " + skills)
        return skills
```

这个实现仅在检查文档是否已经存在于 ElasticSearch 时使用 ElasticSearch。它不会尝试保存从爬虫中新检索到的文档。这是因为`get_job_listing`爬虫的结果只是技能列表，而不是整个文档。因此，这个实现可以使用缓存，但不会添加新数据。这是设计决策之一，即对爬取方法进行不同的设计，返回的只是被爬取文档的子集。

对此的一个潜在解决方案是，将这个 API 方法调用`get_job_listing_info`，然后保存文档，最后只返回特定的子集（在这种情况下是技能）。再次强调，这最终是围绕 sojobs 模块的用户需要哪些类型的方法的设计考虑。出于这些初始示例的目的，考虑到在该级别有两个不同的函数返回不同的数据集更好。
