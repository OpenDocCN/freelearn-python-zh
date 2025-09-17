# 第五章. 使用 Requests 与社交媒体交互

在这个当代世界中，我们的生活与社交媒体的互动和协作紧密相连。网络上的信息非常宝贵，并且被大量资源所利用。例如，世界上的热门新闻可以通过 Twitter 标签轻松找到，这可以通过与 Twitter API 的交互来实现。

通过使用自然语言处理，我们可以通过抓取账户的 Facebook 状态来分类一个人的情绪。所有这些都可以通过使用 Requests 和相关的 API 轻松完成。如果我们要频繁地调用 API，Requests 是一个完美的模块，因为它几乎支持所有功能，如缓存、重定向、代理等等。

我们在本章中将涵盖以下主题：

+   与 Twitter 互动

+   与 Facebook 互动

+   与 Reddit 互动

# API 简介

在深入细节之前，让我们快速了解一下**应用程序编程接口**（**API**）究竟是什么。

网络 API 是一套规则和规范。它帮助我们与不同的软件进行通信。API 有不同类型，而本例中讨论的是 REST API。**表征状态转移**（**REST**）是一种包含构建可扩展网络服务指南的架构。遵循这些指南并符合 REST 约束的 API 被称为**RESTful API**。简而言之，约束包括：

+   客户端-服务器

+   无状态

+   可缓存

+   分层系统

+   统一接口

+   按需编码

Google Maps API、Twitter API 和 GitHub API 是各种 RESTful API 的示例。

让我们更深入地了解 API。以获取带有“worldtoday”标签的所有 Twitter 推文为例，这包括认证过程、向不同 URL 发送请求并接收响应，以及处理不同的方法。所有这些过程和步骤都将由 Twitter 的 API 指定。通过遵循这些步骤，我们可以与网络顺利协作。

## Twitter API 入门

要开始使用 Twitter API，我们首先需要获取一个 API 密钥。这是一个在调用 API 时由计算机程序传递的代码。API 密钥的基本目的是它能够唯一地识别它试图与之交互的程序。它还通过其令牌在我们进行身份验证的过程中为我们提供服务。

下一步涉及创建一个认证请求的过程，这将使我们能够访问 Twitter 账户。一旦我们成功认证，我们将可以自由地处理推文、关注者、趋势、搜索等内容。让我们来了解一下需要遵循的步骤。

### 注意

请注意，在所有示例中，我们将使用 Twitter API 1.1 版本。

## 获取 API 密钥

获取 API 密钥非常简单。您需要遵循以下章节中规定的步骤：

1.  首先，您需要使用您的 Twitter 凭证登录到页面[`apps.twitter.com/`](https://apps.twitter.com/)。

1.  点击**创建新应用**按钮。

1.  现在，您需要填写以下字段以设置新的应用程序：

    +   **名称**：指定您的应用程序名称。这用于归因于推文的来源以及在面向用户的授权屏幕中。

    +   **描述**：输入您应用的简短描述。当用户面对授权界面时，将显示此描述。

    +   **网站**: 指定您的完整网站 URL。一个完整的 URL 包括 http://或 https://，并且末尾不会带有斜杠（例如：`http://example.com`或`http://www.example.com`）。

    +   **回调 URL**：此字段回答了问题——在成功认证后我们应该返回哪里。

    +   **开发者协议**：仔细阅读**开发者协议**，然后勾选**是，我同意**。

1.  现在，通过点击**创建您的 Twitter 应用**，将为我们创建一个包含之前指定详情的新应用。

1.  成功创建后，我们将被重定向到一个页面，其中默认选中了**详情**标签页。现在，请选择**密钥和访问令牌**标签页。我们应该点击**创建我的访问令牌**按钮来生成我们的访问令牌。

1.  最后，记下**消费者密钥（API 密钥）**、**消费者密钥（API 密钥）**、**访问令牌**和**访问令牌密钥**。

## 创建一个认证请求

如果我们还记得第三章的主题，我们学习了使用 `requests` 进行不同类型的身份验证，例如基本身份验证、摘要身份验证和 OAuth 身份验证。现在是时候将这些知识应用到实际中了！

现在，我们将使用 OAuth1 认证来获取访问 Twitter API 的权限。在获取密钥的第一步中，我们获得了消费者密钥、消费者密钥密钥、访问令牌和访问令牌密钥，现在我们应该使用它们来验证我们的应用程序。以下命令展示了我们如何完成这个过程：

```py
>>> import requests
>>> from requests_oauthlib import OAuth1
>>> CONSUMER_KEY = 'YOUR_APP_CONSUMER_KEY'
>>> CONSUMER_SECRET = 'YOUR_APP_CONSUMER_SECRET'
>>> ACCESS_TOKEN = 'YOUR_APP_ACCESS_TOKEN'
>>> ACCESS_TOKEN_SECRET = 'YOUR_APP_ACCESS_TOKEN_SECRET'

>>> auth = OAuth1(CONSUMER_KEY, CONSUMER_SECRET,
...               ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

```

在前面的行中，我们已经将我们的密钥和令牌发送到 API，并完成了身份验证，并将它们存储在变量`auth`中。现在，我们可以使用这个变量进行各种与 API 的交互。让我们开始与 Twitter API 进行交互。

### 注意事项

请记住，在此之后展示的所有推特互动示例都将使用上一节中获得的“auth”值。

## 获取你喜欢的推文

首先，让我们获取认证用户的几个喜欢的推文。为此，我们应该向 Twitter API 发送请求以访问喜欢的推文。可以通过指定参数通过`资源 URL`发送请求。获取喜欢的列表的`资源 URL`看起来像这样：

`https://api.twitter.com/1.1/favorites/list.json`

我们还可以向 URL 发送一些可选参数，如`user_id`、`screen_name`、`count`、`since_id`、`max_id`、`include_identities`，以满足我们的需求。现在让我们获取一条喜欢的推文。

```py
>>> favorite_tweet = requests.get('https://api.twitter.com/1.1/favorites/list.json?count=1', auth=auth)
>>> favorite_tweet.json()
[{u'contributors': None, u'truncated': False, u'text': u'India has spent $74 mil to reach Mars. Less than the budget of the film \u201cGravity,\u201d $100 million.\n\n#respect\n#ISRO\n#Mangalyaan', u'in_reply_to_status_id': None, …}]

```

在第一步中，我们向资源 URL 发送了一个带有参数`count`和认证`auth`的`get`请求。在下一步中，我们访问了以 JSON 格式返回的响应，其中包含了我最喜欢的推文，就这么简单。

由于我们在请求中指定了计数参数为`1`，我们偶然看到了一条喜欢的推文的结果。默认情况下，如果我们没有指定可选参数`count`，请求将返回`20`条最近的喜欢的推文。

## 执行简单搜索

我们现在将使用 Twitter 的 API 进行搜索。为此，我们将利用 Twitter 的`Search API`。搜索的基本 URL 结构具有以下语法：

`https://api.twitter.com/1.1/search/tweets.json?q=%40twitterapi`

它还增加了额外的参数，如`结果类型`、`地理位置`、`语言`、`在结果集中迭代`。

```py
>>> search_results = requests.get('https://api.twitter.com/1.1/search/tweets.json?q=%40python', auth=auth)
>>> search_results.json().keys()
[u'search_metadata', u'statuses']
>>> search_results.json()["search_metadata"]
{u'count': 15, u'completed_in': 0.022, u'max_id_str': u'529975076746043392', u'since_id_str': u'0', u'next_results': u'?max_id=527378999857532927&q=%40python&include_entities=1', u'refresh_url': u'?since_id=529975076746043392&q=%40python&include_entities=1', u'since_id': 0, u'query': u'%40python', u'max_id': 529975076746043392}

```

在前面的例子中，我们尝试搜索包含单词`python`的推文。

## 访问关注者列表

让我们访问指定用户的关注者。默认情况下，当我们查询关注者列表时，它返回最近的`20`位关注用户。资源 URL 看起来像这样：

`https://api.twitter.com/1.1/followers/list.json`

它返回指定用户关注的用户对象的带光标集合：

```py
>>> followers = requests.get('https://api.twitter.com/1.1/followers/list.json', auth=auth)
>>> followers.json().keys()
[u'previous_cursor', u'previous_cursor_str', u'next_cursor', u'users', u'next_cursor_str']
>>> followers.json()["users"]
[{u'follow_request_sent': False, u'profile_use_background_image': True, u'profile_text_color': u'333333'... }]

```

## 转发

被转发过的推文称为**转发推文**。要访问由认证用户创建的最新转发推文，我们将使用以下网址：

`https://api.twitter.com/1.1/statuses/retweets_of_me.json`

可以与其一起发送的可选参数有 `count`、`since_id`、`max_id`、`trim_user`、`include_entities`、`include_user_entities`

```py
>>> retweets = requests.get('https://api.twitter.com/1.1/statuses/retweets_of_me.json', auth=auth)
>>> len(retweets.json())
16
>>> retweets.json()[0]
{u'contributors': None, u'text': u'I\u2019m now available to take on new #python #django #freelance projects. Reply for more details!', {u'screen_name': u'vabasu', ...}}

```

## 访问可用趋势

Twitter 的热门话题是由标签驱动的特定时间内的主题。以获取 Twitter 中可用趋势的位置为例。为此，我们将使用以下网址：

`https://api.twitter.com/1.1/trends/available.json`

资源 URL 的响应是一个以编码形式表示的位置数组：

```py
>>> available_trends = requests.get('https://api.twitter.com/1.1/trends/available.json', auth=auth)
>>> len(available_trends.json())
467
>>> available_trends.json()[10]
{u'name': u'Blackpool', u'countryCode': u'GB', u'url': u'http://where.yahooapis.com/v1/place/12903', u'country': u'United Kingdom', u'parentid': 23424975, u'placeType': {u'code': 7, u'name': u'Town'}, u'woeid': 12903}

```

在前面的代码行中，我们搜索了`available_trends`的位置。然后，我们了解到拥有`available_trends`的位置数量是`467`。后来，我们尝试访问第十个位置的数据，结果返回了一个包含位置信息的响应，该信息是用**woeid**编码的。这是一个称为**Where on Earth ID**的唯一标识符。

## 更新用户状态

为了更新认证用户的当前状态，这通常被称为发推文，我们遵循以下程序。

对于每次更新尝试，更新文本将与认证用户的最近推文进行比较。任何可能导致重复的尝试都将被阻止，从而导致`403 错误`。因此，用户不能连续两次提交相同的状态。

```py
>>> requests.post('https://api.twitter.com/1.1/statuses/update.json?status=This%20is%20a%20Tweet', auth=auth)

```

# 与 Facebook 互动

Facebook API 平台帮助我们这样的第三方开发者创建自己的应用程序和服务，以便访问 Facebook 上的数据。

让我们使用 Facebook API 来绘制 Facebook 数据。Facebook 提供了两种类型的 API；即 Graph API 和 Ads API。Graph API 是一个 RESTful JSON API，通过它可以访问 Facebook 的不同资源，如状态、点赞、页面、照片等。Ads API 主要处理管理对广告活动、受众等访问的权限。

在本章中，我们将使用 Facebook Graph API 与 Facebook 进行交互。它以节点和边的方式命名，表示其表示方式。节点代表*事物*，这意味着一个用户、一张照片、一个页面；而边则代表事物之间的连接；即页面的照片、照片的评论。

### 注意事项

本节中的所有示例都将使用 Graph API 版本 2.2

## 开始使用 Facebook API

要开始使用 Facebook API，我们需要一个被称为访问令牌的不透明字符串，该字符串由 Facebook 用于识别用户、应用或页面。其后是获取密钥的步骤。我们将几乎向 API 发送所有请求到 `graph.facebook.com`，除了视频上传相关的内容。发送请求的流程是通过使用以下方式中的节点唯一标识符来进行的：

```py
GET graph.facebook.com/{node-id}
```

同样地，我们可以这样进行 POST 操作：

```py
POST graph.facebook.com/{node-id}
```

## 获取一个密钥

Facebook API 的令牌是可移植的，可以从移动客户端、网页浏览器或服务器进行调用。

有四种不同类型的访问令牌：

+   **用户访问令牌**：这是最常用的一种访问令牌，需要用户的授权。此令牌可用于访问用户信息和在用户的动态时间轴上发布数据。

+   **应用访问令牌**：当在应用级别处理时，这个令牌就会出现。这个令牌并不能帮助获取用户的访问权限，但它可以用来读取流。

+   **页面访问令牌**：此令牌可用于访问和管理 Facebook 页面。

+   **客户端令牌**：此令牌可以嵌入到应用程序中以获取对应用级 API 的访问权限。

在本教程中，我们将使用应用访问令牌，该令牌由应用 ID 和应用密钥组成，以获取对资源的访问权限。

按照以下步骤获取应用访问令牌：

1.  使用位于[`developers.facebook.com/developer-console/`](https://developers.facebook.com/developer-console/)的 Facebook 开发者控制台创建一个应用程序。请注意，我们应该登录到[`developers.facebook.com`](http://developers.facebook.com)，以便我们能够获得创建应用程序的权限。

1.  一旦我们完成了应用程序的创建，我们就可以在我们的[`developers.facebook.com`](http://developers.facebook.com)账户的应用程序页面上获取 App Id 和 App Secret 的访问权限。

就这些；获取密钥就这么简单。我们不需要创建任何认证请求来发送消息，与 Twitter 上的情况不同。App ID 和 App Secret 就足以赋予我们访问资源的权限。

## 获取用户资料

我们可以使用 API URL `https://graph.facebook.com/me` 通过 GET 请求访问已登录网站的人的当前用户资料。在通过 requests 使用任何 Graph API 调用时，我们需要传递之前获得的访问令牌作为参数。

首先，我们需要导入 requests 模块，然后我们必须将访问令牌存储到一个变量中。这个过程按照以下方式进行：

```py
>>> import requests
>>> ACCESS_TOKEN = '231288990034554xxxxxxxxxxxxxxx'

```

在下一步，我们应该以以下方式发送所需的图形 API 调用：

```py
>>> me = requests.get("https://graph.facebook.com/me", params={'access_token': ACCESS_TOKEN})

```

现在，我们有一个名为 `me` 的 `requests.Response` 对象。`me.text` 返回一个 JSON 响应字符串。要访问检索到的用户配置文件中的各种元素（例如，`id`、`name`、`last_name`、`hometown`、`work`），我们需要将 `json` `response` 字符串转换为 `json object` 字符串。我们可以通过调用方法 `me.json()` 来实现这一点。`me.json.keys()` 返回字典中的所有键：

```py
>>> me.json().keys()
[u'website', u'last_name', u'relationship_status', u'locale', u'hometown', u'quotes', u'favorite_teams', u'favorite_athletes', u'timezone', u'education', u'id', u'first_name', u'verified', u'political', u'languages', u'religion', u'location', u'username', u'link', u'name', u'gender', u'work', u'updated_time', u'interested_in']

```

用户的`id`是一个唯一的数字，用于在 Facebook 上识别用户。我们可以通过以下方式从用户资料中获取当前资料 ID。在后续的示例中，我们将使用此 ID 来检索当前用户的友人、动态和相册。

```py
>>> me.json()['id']
u'10203783798823031'
>>> me.json()['name']
u'Bala Subrahmanyam Varanasi'

```

## 获取朋友列表

让我们收集特定用户的好友列表。为了实现这一点，我们应该向`https://graph.facebook.com/<user-id>/friends`发起 API 调用，并将`user-id`替换为用户的 ID 值。

现在，让我们获取在前一个示例中检索到的用户 ID 的朋友列表：

```py
>>> friends = requests.get("https://graph.facebook.com/10203783798823031/friends", params={'access_token': ACCESS_TOKEN})
>>> friends.json().keys()
[u'paging', u'data']

```

API 调用的响应包含一个 JSON 对象字符串。朋友的信息存储在`response json`对象的`data`属性中，这是一个包含朋友 ID 和名称作为键的朋友对象列表。

```py
>>> len(friends.json()['data'])
32
>>> friends.json().keys()
[u'paging', u'data']
>>> friends.json()['data'][0].keys()
[u'name', u'id']

```

## 获取推送内容

为了检索包括当前用户或他人发布在当前用户个人资料中的状态更新和链接的帖子流，我们应该在请求中使用 feed 参数。

```py
>>> feed = requests.get("https://graph.facebook.com/10203783798823031/feed", params={'access_token': ACCESS_TOKEN})
>>> feed.json().keys()
[u'paging', u'data']
>>> len(feed.json()["data"])
24
>>> feed.json()["data"][0].keys()
[u'from', u'privacy', u'actions', u'updated_time', u'likes', u'created_time', u'type', u'id', u'status_type']

```

在前面的例子中，我们发送了一个请求以获取具有用户 ID `10203783798823031`的特定用户的动态。

## 检索专辑

让我们访问当前登录用户创建的相册。这可以通过以下方式实现：

```py
>>> albums = requests.get("https://graph.facebook.com/10203783798823031/albums", params={'access_token': ACCESS_TOKEN})
>>> albums.json().keys()
[u'paging', u'data']
>>> len(albums.json()["data"])
13
>>> albums.json()["data"][0].keys()
[u'count', u'from', u'name', u'privacy', u'cover_photo', u'updated_time', u'link', u'created_time', u'can_upload', u'type', u'id']
>>> albums.json()["data"][0]["name"]
u'Timeline Photos'

```

在前面的例子中，我们向图 API 发送了一个请求，以获取具有`user-id` `10203783798823031`的用户的专辑。然后我们尝试通过 JSON 访问响应数据。

# 与 Reddit 互动

Reddit 是一个流行的社交网络、娱乐和新闻网站，注册会员可以提交内容，例如文本帖子或直接链接。它允许注册用户对提交的内容进行“赞同”或“反对”的投票，以在网站页面上对帖子进行排名。每个内容条目都按兴趣领域分类，称为 **SUBREDDITS**。

在本节中，我们将直接访问 reddit API，使用 Python 的 requests 库。我们将涵盖以下主题：对 reddit API 的基本概述、获取与我们自己的 reddit 账户相关的数据，以及使用搜索 API 检索链接。

## 开始使用 reddit API

Reddit API 由四个重要的部分组成，在开始与之交互之前，我们需要熟悉这四个部分。这四个部分是：

1.  **列表**: Reddit 中的端点被称为列表。它们包含诸如 `after`/`before`、`limit`、`count`、`show` 等参数。

1.  **modhashes**：这是一个用于防止**跨站请求伪造**（**CSRF**）攻击的令牌。我们可以通过使用`GET /api/me.json`来获取我们的 modhash。

1.  **fullnames**: 全名是一个事物的类型和其唯一 ID 的组合，它构成了 Reddit 上全局唯一 ID 的紧凑编码。

1.  **账户**: 这涉及到用户的账户。使用它我们可以注册、登录、设置强制 HTTPS、更新账户、更新电子邮件等等。

## 注册新账户

在 reddit 上注册新账户很简单。首先，我们需要访问 reddit 网站——[`www.reddit.com/`](https://www.reddit.com/)，然后点击右上角的**登录或创建账户**链接，就会出现注册表单。注册表单包括：

+   **用户名**：用于唯一标识 Reddit 社区成员

+   **电子邮件**：用于直接与用户沟通的可选字段

+   **密码**：登录 Reddit 平台的加密密码

+   **验证密码**：此字段应与密码字段相同

+   **验证码**: 此字段用于检查尝试登录的用户是真人还是可编程的机器人

让我们创建一个新账户，使用我们选择的用户名和密码。目前，请将电子邮件字段留空。我们将在下一节中添加它。

在以下示例中，我假设我们之前创建的用户名和密码分别是`OUR_USERNAME`和`OUR_PASSWORD`。

## 修改账户信息

现在，让我们在我们的账户资料中添加一封电子邮件，这是我们在上一个部分创建账户时故意未完成的。

1.  让我们从创建一个会话对象开始这个过程，它允许我们在所有请求中维护某些参数和 cookie。

    ```py
    >>> import requests
    >>> client = requests.session()
    >>> client.headers = {'User-Agent': 'Reddit API - update profile'}

    ```

1.  让我们创建一个具有`'user'`、`'passwd'`和`'api type'`属性的`DATA`属性。

    ```py
    >>> DATA = {'user': 'OUR_USERNAME', 'passwd': 'OUR_PASSWORD', 'api type': 'json'}

    ```

1.  我们可以通过向 URL 发起一个`post`请求调用来访问我们的 Reddit 账户——[`ssl.reddit.com/api/login`](https://ssl.reddit.com/api/login)，其中登录凭证存储在`DATA`属性中。

    ```py
    >>> response = client.post('https://ssl.reddit.com/api/login', data=DATA)

    ```

1.  Reddit API 对上述帖子请求的响应将被存储在 `response` 变量中。`response` 对象包含 `data` 和 `errors` 信息，如下例所示：

    ```py
    >>> print response.json()
    {u'json': {u'errors': [], u'data': {u'need_https': False, u'modhash': u'v4k68gabo0aba80a7fda463b5a5548120a04ffb43490f54072', u'cookie': u'32381424,2014-11-09T13:53:30,998c473d93cfeb7abcd31ac457c33935a54caaa7'}}}

    ```

1.  我们需要将前一个响应中获得的`modhash`值发送，以执行更新调用以更改我们的`email`。现在，让我们调用以下示例中的 reddit 更新 API：

    ```py
    >>> modhash = response.json()['json']['data']['modhash']
    >>> update_params = {"api_type": "json", "curpass": "OUR_PASSWORD",
    ...                  "dest": "www.reddit.com", "email": "user@example.com",
    ...                  "verpass": "OUR_PASSWORD", "verify": True, 'uh': modhash}
    >>> r = client.post('http://www.reddit.com/api/update', data=update_params)

    ```

1.  更新调用响应存储在 `r` 中。如果没有错误，则 `status_code` 将为 `200`，`errors` 属性的值将是一个空列表，如下例所示：

    ```py
    >>> print r.status_code
    200
    >>> r.text
    u'{"json": {"errors": []}}'

    ```

1.  现在，让我们通过获取当前认证用户的详细信息来检查`email`字段是否已设置。如果`has_mail`属性为`True`，那么我们可以假设电子邮件已成功更新。

    ```py
    >>> me = client.get('http://www.reddit.com/api/me.json')
    >>> me.json()['data']['has_mail']
    True

    ```

## 执行简单搜索

我们可以使用 Reddit 的搜索 API 来搜索整个网站或特定子版块。在本节中，我们将探讨如何发起一个搜索 API 请求。按照以下步骤进行，以发起一个搜索请求。

要进行搜索 API 调用，我们需要向`http://www.reddit.com/search.json` URL 发送一个带有搜索查询参数`q`的 GET 请求。

```py
>>> search = requests.get('http://www.reddit.com/search.json', params={'q': 'python'})
>>> search.json().keys()
[u'kind', u'data']
>>> search.json()['data']['children'][0]['data'].keys()
[u'domain', u'author', u'media', u'score', u'approved_by', u'name', u'created', u'url', u'author_flair_text', u'title' ... ]

```

搜索响应存储在`search`变量中，它是一个`requests.Response`对象。搜索结果存储在`data`属性的`children`属性中。我们可以像以下示例中那样访问搜索结果中的`title`、`author`、`score`或其他项目：

```py
>>> search.json()['data']['children'][0]['data']['title']
u'If you could change something in Python what would it be?'
>>> search.json()['data']['children'][0]['data']['author']
u'yasoob_python'
>>> search.json()['data']['children'][0]['data']['score']
146

```

## 搜索 subreddits

在 reddit 的子版块中通过标题和描述进行搜索与在 reddit 中进行搜索相同。为此，我们需要向`http://www.reddit.com/search.json` URL 发送一个带有搜索查询参数`q`的 GET 请求。

```py
>>> subreddit_search = requests.get('http://www.reddit.com/subreddits/search.json', params={'q': 'python'})

```

搜索响应存储在`search`变量中，它是一个`requests.Response`对象。搜索结果存储在`data`属性中。

```py
>>> subreddit_search.json()['data']['children'][0]['data']['title']
u'Python'

```

# 摘要

本章旨在指导您使用 Python 和 requests 库与一些最受欢迎的社交媒体平台进行交互。我们首先学习了在现实世界中 API 的定义和重要性。然后，我们与一些最受欢迎的社会化媒体网站，如 Twitter、Facebook 和 Reddit 进行了交互。每个关于社交网络的章节都将通过一组有限的示例提供实际操作经验。

在下一章，我们将逐步学习使用 requests 和 BeautifulSoup 库进行网络爬取。
