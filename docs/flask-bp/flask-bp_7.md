# 第七章：Dinnerly - 食谱分享

在本章中，我们将探讨所谓的社交登录的现代方法，其中我们允许用户使用来自另一个网络应用程序的派生凭证对我们的应用程序进行身份验证。目前，支持这种机制的最广泛的第三方应用程序是 Twitter 和 Facebook。

虽然存在其他几种广泛的网络应用程序支持这种集成类型（例如 LinkedIn、Dropbox、Foursquare、Google 和 GitHub 等），但您潜在用户的大多数将至少拥有 Twitter 或 Facebook 中的一个帐户，这两个是当今主要的社交网络。

为此，我们将添加、配置和部署 Flask-OAuthlib 扩展。该扩展抽象出了通常在处理基于 OAuth 的授权流程时经常遇到的一些困难和障碍（我们将很快解释），并包括功能以快速设置所需的默认值来协商提供者/消费者/资源所有者令牌交换。作为奖励，该扩展将为我们提供与用户代表的这些远程服务的经过身份验证的 API 进行交互的能力。

# 首先是 OAuth

让我们先把这个搞清楚：OAuth 可能有点难以理解。更加火上浇油的是，OAuth 框架/协议在过去几年中经历了一次重大修订。第 2 版于 2012 年发布，但由于各种因素，仍有一些网络应用程序继续实施 OAuth v1 协议。

### 注意

OAuth 2.0 与 OAuth 1.0 不兼容。此外，OAuth 2.0 更像是授权框架规范，而不是正式的协议规范。现代网络应用程序中大多数 OAuth 2.0 实现是不可互操作的。

为了简单起见，我们将概述 OAuth 2.0 授权框架的一般术语、词汇和功能。第 2 版是两个规范中更简单的一个，这是有道理的：后者的设计目标之一是使客户端实现更简单，更不容易出错。大部分术语在两个版本中是相似的，如果不是完全相同的。

虽然由于 Flask-OAuthlib 扩展和处理真正繁重工作的底层 Python 包，OAuth 授权交换的复杂性大部分将被我们抽象化，但对于网络应用程序和典型实现的 OAuth 授权框架（特别是最常见的授权授予流程）的一定水平的了解将是有益的。

## 为什么使用 OAuth？

适当的在线个人安全的一个重大错误是在不同服务之间重复使用访问凭证。如果您用于一个应用的凭证被泄露，这将使您面临各种安全问题。现在，您可能会在使用相同一组凭证的所有应用程序上受到影响，唯一的后期修复方法是去到处更改您的凭证。

比在不同服务之间重复使用凭证更糟糕的是，用户自愿将他们的凭证交给第三方服务，比如 Twitter，以便其他服务，比如 Foursquare，可以代表用户向 Twitter 发出请求（例如，在他们的 Twitter 时间轴上发布签到）。虽然不是立即明显，但这种方法的问题之一是凭证必须以明文形式存储。

出于各种原因，这种情况并不理想，其中一些原因是您作为应用程序开发人员无法控制的。

OAuth 在框架的 1 版和 2 版中都试图通过创建 API 访问委托的开放标准来解决跨应用程序共享凭据的问题。OAuth 最初设计的主要目标是确保应用程序 A 的用户可以代表其委托应用程序 B 访问，并确保应用程序 B 永远不会拥有可能危害应用程序 A 用户帐户的凭据。

### 注意

虽然拥有委托凭据的应用程序可以滥用这些凭据来执行一些不良操作，但根凭据从未被共享，因此帐户所有者可以简单地使被滥用的委托凭据无效。如果根帐户凭据简单地被提供给第三方应用程序，那么后者可以通过更改所有主要身份验证信息（用户名、电子邮件、密码等）来完全控制帐户，从而有效地劫持帐户。

## 术语

关于 OAuth 的使用和实施的大部分混乱源于对用于描述基本授权流的基本词汇和术语的误解。更糟糕的是，有几个流行的 Web 应用程序已经实施了 OAuth（以某种形式），并决定使用自己的词汇来代替官方 RFC 中已经决定的词汇。

### 注意

RFC，或称为请求评论，是来自**互联网工程任务组**（**IETF**）的一份文件或一组文件的备忘录式出版物，IETF 是管理大部分互联网建立在其上的开放标准的主要机构。RFC 通常由一个数字代码表示，该代码在 IETF 中唯一标识它们。例如，OAuth 2.0 授权框架 RFC 编号为 6749，可以在 IETF 网站上完整找到。

为了帮助减轻一些混乱，以下是 OAuth 实施中大多数基本组件的简化描述：

+   消费者：这是代表用户发出请求的应用程序。在我们的特定情况下，Dinnerly 应用程序被视为消费者。令人困惑的是，官方的 OAuth 规范是指客户端而不是消费者。更令人困惑的是，一些应用程序同时使用消费者和客户端术语。通常，消费者由必须保存在应用程序配置中的密钥和秘钥表示，并且必须受到良好的保护。如果恶意实体获得了您的消费者密钥和秘钥，他们就可以在向第三方提供商发出授权请求时假装成您的应用程序。

+   **提供者**：这是消费者代表用户试图访问的第三方服务。在我们的情况下，Twitter 和 Facebook 是我们将用于应用程序登录的提供者。其他提供者的例子可能包括 GitHub、LinkedIn、Google 以及任何其他提供基于授权流的 OAuth 授权的服务。

+   **资源所有者**：这是有能力同意委托资源访问的实体。在大多数情况下，资源所有者是所涉及应用程序的最终用户（例如，Twitter 和 Dinnerly）。

+   **访问令牌**：这是客户端代表用户向提供者发出请求以访问受保护资源的凭据。令牌可以与特定的权限范围相关联，限制其可以访问的资源。此外，访问令牌可能会在由提供者确定的一定时间后过期；此时需要使用刷新令牌来获取新的有效访问令牌。

+   **授权服务器**：这是负责在资源所有者同意委托他们的访问权限后向消费者应用程序发放访问令牌的服务器（通常由 URI 端点表示）。

+   **流程类型**：OAuth 2.0 框架提供了几种不同的授权流程概述。有些最适合于没有网络浏览器的命令行应用程序，有些更适合于原生移动应用程序，还有一些是为连接具有非常有限访问能力的设备而创建的（例如，如果您想将 Twitter 帐户特权委托给您的联网烤面包机）。我们最感兴趣的授权流程，不出所料，是为基本基于网络浏览器的访问而设计的。

有了上述词汇表，您现在应该能够理解官方 OAuth 2.0 RFC 中列出的官方抽象协议流程：

```py
 +--------+                               +---------------+
 |        |--(A)- Authorization Request ->|   Resource    |
 |        |                               |     Owner     |
 |        |<-(B)-- Authorization Grant ---|               |
 |        |                               +---------------+
 |        |
 |        |                               +---------------+
 |        |--(C)-- Authorization Grant -->| Authorization |
 | Client |                               |     Server    |
 |        |<-(D)----- Access Token -------|               |
 |        |                               +---------------+
 |        |
 |        |                               +---------------+
 |        |--(E)----- Access Token ------>|    Resource   |
 |        |                               |     Server    |
 |        |<-(F)--- Protected Resource ---|               |
 +--------+                               +---------------+

```

以下是从 RFC 6749 中列出的流程图中列出的步骤的描述，并且为了我们的目的更加相关：

1.  客户端（或消费者）请求资源所有者授予授权。这通常是用户被重定向到远程提供者的登录屏幕的地方，比如 Twitter，在那里解释了客户端应用程序希望访问您控制的受保护资源。同意后，我们进入下一步。

1.  客户端从资源所有者（用户）那里收到授权凭证，这是代表资源所有者对提供者实施的特定类型授权流程的授权的临时凭证。对于大多数 Web 应用程序来说，这通常是授权代码授予流程。

1.  一旦客户端收到授权凭证，它会将其发送到授权服务器，以代表资源所有者请求认证令牌。

1.  授权服务器验证授权凭证并对发出请求的客户端进行身份验证。在满足这两个要求后，服务器将有效的认证令牌返回给客户端，然后客户端可以使用该令牌代表用户向提供者发出经过认证的请求。

## 那么 OAuth 1.0 有什么问题呢？

理论上：没有太多问题。实际上：对于消费者来说，正确实施起来有些困难，而且极易出错。

在实施和使用 OAuth 1.0 提供程序时的主要困难围绕着消费者应用程序未能正确执行所需的加密请求签名。参数和参数必须从查询字符串中收集，还必须从请求正文和各种 OAuth 参数（例如，`oauth_nonce`，`oauth_signature_method`，`oauth_timestamp`等）中收集，然后进行 URL 编码（意味着非 URL 安全值被特殊编码以确保它们被正确传输）。一旦键/值对已被编码，它们必须按键的字典顺序进行排序（记住，编码后的键而不是原始键值），然后使用典型的 URL 参数分隔符将它们连接成一个字符串。此外，要提交请求的 HTTP 动词（例如，`GET`或`POST`）必须预先添加到我们刚刚创建的字符串中，然后跟随请求将被发送到的 URL。最后，签名密钥必须由消费者秘钥和 OAuth 令牌秘钥构建，然后传递给 HMAC-SHA1 哈希算法的实现，以及我们之前构建的有效载荷。

假设您已经全部正确理解了这些（很容易出现简单错误，比如按字母顺序而不是按字典顺序对密钥进行排序），那么请求才会被视为有效。此外，在发生签名错误的情况下，没有简单的方法确定错误发生的位置。

OAuth 1.0 需要这种相当复杂的过程的原因之一是，该协议的设计目标是它应该跨不安全的协议（如 HTTP）运行，但仍确保请求在传输过程中没有被恶意方修改。

尽管 OAuth 2.0 并不被普遍认为是 OAuth 1.0 的值得继任者，但它通过简单要求所有通信都在 HTTPS 上进行，大大简化了实现。

## 三步授权

在 OAuth 框架的所谓三步授权流程中，应用程序（`consumer`）代表用户（`resource owner`）发出请求，以访问远程服务（`provider`）上的资源。

### 注意

还存在一个两步授权流程，主要用于应用程序之间的访问，资源所有者不需要同意委托访问受保护资源。例如，Twitter 实现了两步和三步授权流程，但前者在资源访问和强制 API 速率限制方面没有与后者相同的访问范围。

这就是 Flask-Social 将允许我们为 Twitter 和 Facebook 实现的功能，我们选择的两个提供者，我们的应用程序将作为消费者。最终结果将是我们的 Dinnerly 应用程序将拥有这两个提供者的访问令牌，这将允许我们代表我们的用户（资源所有者）进行经过身份验证的 API 请求，这对于实现任何跨社交网络发布功能是必要的。

# 设置应用程序

再次，让我们为我们的项目设置一个基本的文件夹，以及相关的虚拟环境，以隔离我们的应用程序依赖关系：

```py
$ mkdir –p ~/src/dinnerly
$ mkvirtualenv dinnerly
$ cd ~/src/dinnerly

```

创建后，让我们安装我们需要的基本包，包括 Flask 本身以及 Flask-OAuthlib 扩展，我们值得信赖的朋友 Flask-SQLAlchemy 和我们在之前章节中使用过的 Flask-Login：

```py
$ pip install flask flask-oauthlib flask-sqlalchemy flask-login flask-wtf

```

我们将利用我们在过去章节中表现良好的 Blueprint 应用程序结构，以确保坚实的基础。现在，我们将有一个单一的用户 Blueprint，其中将处理 OAuth 处理：

```py
-run.py
-application
 ├── __init__.py
 └── users
     ├── __init__.py
     ├── models.py
    └── views.py

```

一旦建立了非常基本的文件夹和文件结构，让我们使用应用程序工厂来创建我们的主应用程序对象。现在，我们要做的只是在`application/__init__.py`中实例化一个非常简单的应用程序，其中包含一个 Flask-SQLAlchemy 数据库连接：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy

# Deferred initialization of the db extension
db = SQLAlchemy()

def create_app(config=None):
    app = Flask(__name__, static_folder=None)

    if config is not None:
        app.config.from_object(config)

    db.init_app(app)
    return app
```

为了确保我们实际上可以运行应用程序并创建数据库，让我们使用简单的`run.py`和`database.py`脚本，将它们放在`application`文件夹的同级目录。`run.py`的内容与我们在之前章节中使用的内容类似：

```py
from application import create_app

app = create_app(config='settings')
app.run(debug=True)
```

### 注意

在本章的后面，我们将探讨运行 Dinnerly 应用程序的替代方法，其中大部分更适合生产部署。在`app.run()`上调用的 Werkzeug 开发服务器非常不适合除了本地开发之外的任何其他用途。

我们的`database.py`同样简单明了：

```py
from application import db, create_app
app = create_app(config='settings')
db.app = app

db.create_all()
```

这将允许我们根据我们的模型定义在数据库中创建相关的模式，但我们还没有声明模型；现在运行脚本基本上不会有任何操作。这没关系！在这变得有用之前我们还有很多工作要做。

## 声明我们的模型

与大多数应用程序一样，我们首先声明我们的数据模型和它们需要的任何关系。当然，我们需要一个`User`模型，它将是 OAuth 授权和令牌交换的核心。

正如您可能还记得我们对 OAuth 术语和基本的三步授权授予流程的简要概述，访问令牌是允许客户端（我们的 Dinnerly 应用程序）查询远程服务提供商（例如 Twitter 或 Facebook）资源的东西。由于我们需要这些令牌来向列出的服务提供商发出请求，我们希望将它们存储在某个地方，以便我们可以在没有用户为每个操作重新进行身份验证的情况下使用它们；这将非常繁琐。

我们的`User`模型将与我们以前使用过的`User`模型非常相似（尽管我们删除了一些属性以简化事情），我们将把它放在`application/users/models.py`的明显位置：

```py
import datetime
from application import db

class User(db.Model):

    # The primary key for each user record.
    id = db.Column(db.Integer, primary_key=True)

    # The username for a user. Might not be
    username = db.Column(db.String(40))

    #  The date/time that the user account was created on.
    created_on = db.Column(db.DateTime,
        default=datetime.datetime.utcnow)

    def __repr__(self):
        return '<User {!r}>'.format(self.username)
```

### 注意

请注意，我们没有包括有关密码的任何内容。由于此应用程序的意图是要求使用 Facebook 或 Twitter 创建帐户并登录，我们放弃了典型的用户名/密码凭据组合，而是将身份验证委托给这些第三方服务之一。

为了帮助我们的用户会话管理，我们将重用我们在之前章节中探讨过的 Flask-Login 扩展。以防您忘记，扩展的基本要求之一是在用于表示经过身份验证的用户的任何模型上声明四种方法：`is_authenticated`，`is_active`，`is_anonymous`和`get_id`。让我们将这些方法的最基本版本附加到我们已经声明的`User`模型中：

```py
class User(db.Model):

   # …

    def is_authenticated(self):
        """All our registered users are authenticated."""
        return True

    def is_active(self):
        """All our users are active."""
        return True

    def is_anonymous(self):
        """All users are not in an anonymous state."""
        return False

    def get_id(self):
        """Get the user ID as a Unicode string."""
        return unicode(self.id)
```

现在，您可能已经注意到`User`模型上没有声明的 Twitter 或 Facebook 访问令牌属性。当然，添加这些属性是一个选择，但我们将使用稍微不同的方法，这需要更多的前期复杂性，并且将允许添加更多提供程序而不会过度污染我们的`User`模型。

我们的方法将集中在创建用户与各种提供程序类型之间的多个一对一数据关系的想法上，这些关系将由它们自己的模型表示。让我们在`application/users/models.py`中添加我们的第一个提供程序模型到存储：

```py
class TwitterConnection(db.Model):

    # The primary key for each connection record.
    id = db.Column(db.Integer, primary_key=True)

    # Our relationship to the User that this
    # connection belongs to.
    user_id = db.Column(db.Integer(),
        db.ForeignKey('user.id'), nullable=False, unique=True)

    # The twitter screen name of the connected account.
    screen_name = db.Column(db.String(), nullable=False)

    # The Twitter ID of the connected account
    twitter_user_id = db.Column(db.Integer(), nullable=False)

    # The OAuth token
    oauth_token = db.Column(db.String(), nullable=False)

    # The OAuth token secret
    oauth_token_secret = db.Column(db.String(), nullable=False)
```

前面的模型通过`user_id`属性声明了与`User`模型的外键关系，除了主键之外的其他字段存储了进行身份验证请求所需的 OAuth 令牌和密钥，以代表用户访问 Twitter API。此外，我们还存储了 Twitter 的`screen_name`和`twitter_user_id`，以便将此值用作相关用户的用户名。保留 Twitter 用户 ID 有助于我们将 Twitter 上的用户与本地 Dinnerly 用户匹配（因为`screen_name`可以更改，但 ID 是不可变的）。

一旦`TwitterConnection`模型被定义，让我们将关系添加到`User`模型中，以便我们可以通过`twitter`属性访问相关的凭据：

```py
Class User(db.Model):
  # …

  twitter = db.relationship("TwitterConnection", uselist=False,
    backref="user")
```

这在`User`和`TwitterConnection`之间建立了一个非常简单的一对一关系。`uselist=False`参数确保配置的属性将引用标量值，而不是列表，这将是一对多关系的默认值。

因此，一旦我们获得了用户对象实例，我们就可以通过`user.twitter`访问相关的`TwitterConnection`模型数据。如果没有附加凭据，那么这将返回`None`；如果有附加凭据，我们可以像预期的那样访问子属性：`user.twitter.oauth_token`，`user.twitter.screen_name`等。

让我们为等效的`FacebookConnection`模型做同样的事情，它具有类似的属性。与`TwitterConnection`模型的区别在于 Facebook OAuth 只需要一个令牌（而不是组合令牌和密钥），我们可以选择存储 Facebook 特定的 ID 和名称（而在其他模型中，我们存储了 Twitter 的`screen_name`）：

```py
class FacebookConnection(db.Model):

    # The primary key for each connection record.
    id = db.Column(db.Integer, primary_key=True)

    # Our relationship to the User that this
    # connection belongs to.
    user_id = db.Column(db.Integer(),
        db.ForeignKey('user.id'), nullable=False)

    # The numeric Facebook ID of the user that this
    # connection belongs to.
    facebook_id = db.Column(db.Integer(), nullable=False)

    # The OAuth token
    access_token = db.Column(db.String(), nullable=False)

    # The name of the user on Facebook that this
    # connection belongs to.
    name = db.Column(db.String())
```

一旦我们建立了这个模型，我们就会想要像之前为`TwitterConnection`模型一样，将这种关系引入到我们的`User`模型中：

```py
class User(db.Model):

       # …

    facebook = db.relationship("FacebookConnection", 
        uselist=False, backref="user")
```

`user`实例的前述`facebook`属性的功能和用法与我们之前定义的`twitter`属性完全相同。

## 在我们的视图中处理 OAuth

有了我们基本的用户和 OAuth 连接模型，让我们开始构建所需的 Flask-OAuthlib 对象来处理授权授予流程。第一步是以我们应用程序工厂的通常方式初始化扩展。在此期间，让我们也初始化 Flask-Login 扩展，我们将用它来管理已登录用户的认证会话：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from flask_oauthlib.client import OAuth
 from flask.ext.login import LoginManager

# Deferred initialization of our extensions
db = SQLAlchemy()
oauth = OAuth()
login_manager = LoginManager()

def create_app(config=None):
    app = Flask(__name__, static_folder=None)

    if config is not None:
        app.config.from_object(config)

    db.init_app(app)
 oauth.init_app(app)
 login_manager.init_app(app)

    return app
```

现在我们有了一个`oauth`对象可供我们使用，我们可以为每个服务提供商实例化单独的 OAuth 远程应用程序客户端。让我们将它们放在我们的`application/users/views.py 模块`中：

```py
from flask.ext.login import login_user, current_user
from application import oauth

twitter = oauth.remote_app(
    'twitter',
    consumer_key='<consumer key>',
    consumer_secret='<consumer secret>',
    base_url='https://api.twitter.com/1.1/',
    request_token_url='https://api.twitter.com/oauth/request_token',
    access_token_url='https://api.twitter.com/oauth/access_token',
    authorize_url='https://api.twitter.com/oauth/authenticate')

facebook = oauth.remote_app(
    'facebook',
    consumer_key='<facebook app id>',
    consumer_secret='<facebook app secret>',
    request_token_params={'scope': 'email,publish_actions'},
    base_url='https://graph.facebook.com',
    request_token_url=None,
    access_token_url='/oauth/access_token',
    access_token_method='GET',
    authorize_url='https://www.facebook.com/dialog/oauth')
```

现在，在实例化这些 OAuth 对象时似乎有很多事情要做，但其中大部分只是告诉通用的 OAuth 连接库各种三方 OAuth 授权授予流程的服务提供商 URI 端点在哪里。然而，有一些参数值需要您自己填写：消费者密钥（对于 Twitter）和应用程序密钥（对于 Facebook）。要获得这些值，您必须在相应的服务上注册一个新的 OAuth 客户端应用程序，您可以在这里这样做：

+   Twitter: [`apps.twitter.com/app/new`](https://apps.twitter.com/app/new)，然后转到**Keys**和**Access Tokens**选项卡以获取消费者密钥和消费者密钥。

+   Facebook: [`developers.facebook.com/apps/`](https://developers.facebook.com/apps/)，同意服务条款并注册您的帐户进行应用程序开发。然后，选择要添加的网站类型应用程序，并按照说明生成所需的应用程序 ID 和应用程序密钥。

在 Facebook 的情况下，我们通过`request_token_params`参数的`scope`键的`publish_actions`值请求了发布到相关用户的墙上的权限。这对我们来说已经足够了，但如果您想与 Facebook API 互动不仅仅是推送状态更新，您需要请求正确的权限集。Facebook 文档中有关于第三方应用程序开发者如何使用权限范围值执行不同操作的额外信息和指南。

一旦您获得了所需的密钥和密钥，就将它们插入到前述`oauth`远程应用程序客户端配置中留下的占位符中。

现在，我们需要让我们的应用程序处理授权流程的各个部分，这些部分需要用户从服务提供商那里请求授予令牌。我们还需要让我们的应用程序处理回调路由，服务提供商将在流程完成时重定向到这些路由，并携带各种 OAuth 令牌和密钥，以便我们可以将这些值持久化到我们的数据库中。

让我们创建一个用户 Blueprint 来对`application/users/views.py`中的各种路由进行命名空间处理，同时，我们还可以从 Flask 和 Flask-Login 中导入一些实用程序来帮助我们的集成：

```py
from flask import Blueprint, redirect, url_for, request
from flask.ext.login import login_user, current_user

from application.users.models import (
    User, TwitterConnection, FacebookConnection)
from application import oauth, db, login_manager
import sqlalchemy

users = Blueprint('users', __name__, template_folder='templates')
```

根据 Flask-Login 的要求，我们需要定义一个`user_loader`函数，它将通过 ID 从我们的数据库中获取用户：

```py
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
```

以非常相似的方式，Flask-OAuthlib 要求我们定义一个方法（每个服务一个）作为令牌获取器；而 Flask-Login 需要`user_loader`通过 ID 从数据库中获取用户。OAuthlib 需要一个函数来获取当前登录用户的 OAuth 令牌。如果当前没有用户登录，则该方法应返回`None`，表示我们可能需要开始授权授予流程来获取所需的令牌：

```py
@twitter.tokengetter
def get_twitter_token():
    """Fetch Twitter token from currently logged
    in user."""
    if (current_user.is_authenticated() and
            current_user.twitter):
        return (current_user.twitter.oauth_token,
                current_user.twitter.oauth_token_secret)
    return None

@facebook.tokengetter
def get_facebook_token():
    """Fetch Facebook token from currently logged
    in user."""
    if (current_user.is_authenticated() and
            current_user.facebook):
        return (current_user.facebook.oauth_token, )
    return None
```

### 注意

请注意，我们使用了 Flask-Login 提供的`current_user`代理对象来访问当前经过身份验证的用户的对象，然后我们调用了在本章前面定义的`User`模型中的`is_authenticated`方法。

接下来，我们需要定义路由和处理程序来启动三方授权授予。我们的第一个用户蓝图路由将处理使用 Twitter 作为第三方提供商的尝试登录：

```py
@users.route('/login/twitter')
def login_twitter():
    """Kick-off the Twitter authorization flow if
    not currently authenticated."""

    if current_user.is_authenticated():
        return redirect(url_for('recipes.index'))
    return twitter.authorize(
        callback=url_for('.twitter_authorized',
            _external=True))
```

前面的路由首先确定当前用户是否已经经过身份验证，并在他们已经经过身份验证时将其重定向到主`recipes.index`路由处理程序。

### 注意

我们已经为`recipes.index`路由设置了一些重定向，但我们还没有定义。如果您打算在我们设置这些之前测试应用程序的这一部分，您将不得不在蓝图路由中添加一个存根页面，或者将其更改为其他内容。

如果用户尚未经过身份验证，我们通过`twitter.authorize`方法调用来启动授权授予。这将启动 OAuth 流程，并在授权成功完成后（假设用户同意允许我们的应用程序访问他们的第三方受保护资源），Twitter 将调用 GET 请求到我们提供的回调 URL 作为第一个参数。这个请求将包含 OAuth 令牌和他们认为有用的任何其他信息（如`screen_name`）在查询参数中，然后由我们来处理请求，提取出我们需要的信息。

为此，我们定义了一个`twitter_authorized`路由处理程序，其唯一目的是提取出 OAuth 令牌和密钥，以便我们可以将它们持久化到我们的数据库中，然后使用 Flask-Login 的`login_user`函数为我们的 Dinnerly 应用程序创建一个经过身份验证的用户会话：

```py
@users.route('/login/twitter-authorized')
def twitter_authorized():
  resp = twitter.authorized_response()

  try:
    user = db.session.query(User).join(
      TwitterConnection).filter(
        TwitterConnection.oauth_token == 
          resp['oauth_token']).one()
    except sqlalchemy.orm.exc.NoResultFound:
      credential = TwitterConnection(
        twitter_user_id=int(resp['user_id']),
        screen_name=resp['screen_name'],
        oauth_token=resp['oauth_token'],
        oauth_token_secret=resp['oauth_token_secret'])

        user = User(username=resp['screen_name'])
        user.twitter = credential

        db.session.add(user)
        db.session.commit()
        db.session.refresh(user)

  login_user(user)
  return redirect(url_for('recipes.index'))
```

在前面的路由处理程序中，我们首先尝试从授权流中提取 OAuth 数据，这些数据可以通过`twitter.authorized_response()`提供给我们。

### 注意

如果用户决定拒绝授权请求，那么`twitter.authorized_response()`将返回`None`。处理这种错误情况留给读者作为一个练习。

提示：闪存消息和重定向到描述发生情况的页面可能是一个很好的开始！

一旦从授权流的 OAuth 数据响应中提取出 OAuth 令牌，我们就会检查数据库，看看是否已经存在具有此令牌的用户。如果是这种情况，那么用户已经在 Dinnerly 上创建了一个帐户，并且只希望重新验证身份。（也许是因为他们正在使用不同的浏览器，因此他们没有之前生成的会话 cookie 可用。）

如果我们系统中没有用户被分配了 OAuth 令牌，那么我们将使用我们刚刚收到的数据创建一个新的`User`记录。一旦这个记录被持久化到 SQLAlchemy 会话中，我们就使用 Flask-Login 的`login_user`函数将他们登录。

虽然我们在这里专注于路由处理程序和 Twitter OAuth 授权授予流程，但 Facebook 的流程非常相似。我们的用户蓝图附加了另外两个路由，这些路由将处理希望使用 Facebook 作为第三方服务提供商的登录：

```py
@users.route('/login/facebook')
def login_facebook():
    """Kick-off the Facebook authorization flow if
    not currently authenticated."""

    if current_user.is_authenticated():
        return redirect(url_for('recipes.index'))
    return facebook.authorize(
        callback=url_for('.facebook_authorized',
            _external=True))
```

然后，我们定义了`facebook_authorized`处理程序，它将以与`twitter_authorized`路由处理程序非常相似的方式通过查询参数接收 OAuth 令牌参数：

```py
@users.route('/login/facebook-authorized')
def facebook_authorized():
  """Handle the authorization grant & save the token."""

  resp = facebook.authorized_response()
  me = facebook.get('/me')

  try:
    user = db.session.query(User).join(
      FacebookConnection).filter(
        TwitterConnection.oauth_token ==
          resp['access_token']).one()
    except sqlalchemy.orm.exc.NoResultFound:
      credential = FacebookConnection(
        name=me.data['name'],
        facebook_id=me.data['id'],
        access_token=resp['access_token'])

        user = User(username=resp['screen_name'])
        user.twitter = credential

        db.session.add(user)
        db.session.commit()
        db.session.refresh(user)

  login_user(user)
  return redirect(url_for('recipes.index'))
```

这个处理程序与我们之前为 Twitter 定义的处理程序之间的一个不容忽视的区别是调用`facebook.get('/me')`方法。一旦我们执行了授权授予交换，facebook OAuth 对象就能够代表用户对 Facebook API 进行经过身份验证的请求。我们将利用这一新发现的能力来查询有关委托授权凭据的用户的一些基本细节，例如该用户的 Facebook ID 和姓名。一旦获得，我们将存储这些信息以及新创建用户的 OAuth 凭据。

## 创建食谱

现在我们已经允许用户使用 Twitter 或 Facebook 在 Dinnerly 上创建经过身份验证的帐户，我们需要在这些社交网络上创建一些值得分享的东西！我们将通过`application/recipes/models.py`模块创建一个非常简单的`Recipe`模型：

```py
import datetime
from application import db

class Recipe(db.Model):

    # The unique primary key for each recipe created.
    id = db.Column(db.Integer, primary_key=True)

    # The title of the recipe.
    title = db.Column(db.String())

    # The ingredients for the recipe.
    # For the sake of simplicity, we'll assume ingredients
    # are in a comma-separated string.
    ingredients = db.Column(db.Text())

    # The instructions for each recipe.
    instructions = db.Column(db.Text())

    #  The date/time that the post was created on.
    created_on = db.Column(db.DateTime(),
        default=datetime.datetime.utcnow,
        index=True)

    # The user ID that created this recipe.
    user_id = db.Column(db.Integer(), db.ForeignKey('user.id'))

    # User-Recipe is a one-to-many relationship.
    user = db.relationship('User',
            backref=db.backref('recipes'))
```

我们刚刚定义的`Recipe`模型并没有什么特别之处；它有一个标题、配料和说明。每个食谱都归属于一个用户，我们已经创建了必要的基于关系的字段和我们模型中的`ForeignKey`条目，以便我们的数据以通常的关系数据库方式正确链接在一起。有一些字段用于存储任何食谱中你所期望的典型内容：`title`、`ingredients`和`instructions`。由于 Dinnerly 的目的是在各种社交网络上分享食谱片段，我们应该添加一个方法来帮助生成食谱的简短摘要，并将其限制在 140 个字符以下（以满足 Twitter API 的要求）：

```py
def summarize(self, character_count=136):
    """
    Generate a summary for posting to social media.
    """

    if len(self.title) <= character_count:
        return self.title

    short = self.title[:character_count].rsplit(' ', 1)[0]
    return short + '...'
```

前面定义的`summarize`方法将返回`Recipe`的标题，如果标题包含的字符少于 140 个。如果包含的字符超过 140 个，我们将使用空格作为分隔符将字符串拆分成列表，使用`rsplit`（它从字符串的末尾而不是`str.split`所做的开头开始），然后附加省略号。

### 注意

我们刚刚定义的`summarize`方法只能可靠地处理 ASCII 文本。存在一些 Unicode 字符，可能与 ASCII 字符集中的空格相似，但我们的方法不会正确地在这些字符上拆分。

## 将食谱发布到 Twitter 和 Facebook

在发布新食谱时，我们希望自动将摘要发布到已连接到该用户的服务。当然，有许多方法可以实现这一点：

+   在我们尚未定义的食谱视图处理程序中，我们可以在成功创建/提交`Recipe`对象实例后调用相应的 OAuth 连接对象方法。

+   用户可能需要访问特定的 URI（或提交具体数据的表单），这将触发跨发布。

+   当`Recipe`对象提交到数据库时，我们可以监听 SQLAlchemy 发出的`after_insert`事件，并将我们的摘要推送到连接的社交网络上。

由于前两个选项相对简单，有点无聊，并且到目前为止我们在这本书中还没有探讨过 SQLAlchemy 事件，所以第三个选项是我们将要实现的。

### SQLAlchemy 事件

SQLAlchemy 的一个不太为人所知的特性是事件 API，它发布了几个核心和 ORM 级别的钩子，允许我们附加和执行任意代码。

### 注意

事件系统在精神上（如果不是在实现上）与我们在前一章中看到的 Blinker 分发系统非常相似。我们不是创建、发布和消费基于 blinker 的信号，而是简单地监听 SQLAlchemy 子系统发布的事件。

大多数应用程序永远不需要实现对已发布事件的处理程序。它们通常是 SQLAlchemy 的插件和扩展的范围，允许开发人员增强其应用程序的功能，而无需编写大量的样板连接器/适配器/接口逻辑来与这些插件或扩展进行交互。

我们感兴趣的 SQLAlchemy 事件被归类为 ORM 事件。即使在这个受限的事件范围内（还有大量其他已发布的核心事件，我们甚至不会在这里讨论），仍然有相当多的事件。大多数开发人员通常感兴趣的是映射器级别的事件：

+   `before_insert`：在发出与该实例对应的`INSERT`语句之前，此函数接收一个对象实例

+   `after_insert`：在发出与该实例对应的`INSERT`语句之后，此函数接收一个对象实例

+   `before_update`：在发出与该实例对应的`UPDATE`语句之前，此函数接收一个对象实例

+   `after_update`：在发出与该实例对应的`UPDATE`语句之后，此函数接收一个对象实例

+   `before_delete`：在发出与该实例对应的`DELETE`语句之前，此函数接收一个对象实例

+   `after_delete`：在发出与该实例对应的`DELETE`语句之后，此函数接收一个对象实例

每个命名事件都会与 SQLAlchemy 的`Mapper`对象一起发出（该对象定义了`class`属性与数据库列的对应关系），将被用于执行查询的连接对象，以及被操作的目标对象实例。

通常，开发人员会使用原始连接对象来执行简单的 SQL 语句（例如，增加计数器，向日志表添加一行等）。然而，我们将使用`after_insert`事件来将我们的食谱摘要发布到 Twitter 和 Facebook。

为了从组织的角度简化事情，让我们将 Twitter 和 Facebook 的 OAuth 客户端对象实例化移到它们自己的模块中，即`application/users/services.py`中：

```py
from application import oauth

twitter = oauth.remote_app(
    'twitter',
    consumer_key='<consumer key>',
    consumer_secret='<consumer secret>',
    base_url='https://api.twitter.com/1/',
    request_token_url='https://api.twitter.com/oauth/request_token',
    access_token_url='https://api.twitter.com/oauth/access_token',
    authorize_url='https://api.twitter.com/oauth/authenticate',
    access_token_method='GET')

facebook = oauth.remote_app(
    'facebook',
    consumer_key='<consumer key>',
    consumer_secret='<consumer secret>',
    request_token_params={'scope': 'email,publish_actions'},
    base_url='https://graph.facebook.com',
    request_token_url=None,
    access_token_url='/oauth/access_token',
    access_token_method='GET',
    authorize_url='https://www.facebook.com/dialog/oauth')
```

将此功能移动到一个单独的模块中，我们可以避免一些更糟糕的循环导入可能性。现在，在`application/recipes/models.py`模块中，我们将添加以下函数，当发出`after_insert`事件并由`listens_for`装饰器标识时将被调用：

```py
from application.users.services import twitter, facebook
from sqlalchemy import event

@event.listens_for(Recipe, 'after_insert')
def listen_for_recipe_insert(mapper, connection, target):
    """Listens for after_insert event from SQLAlchemy
    for Recipe model instances."""

    summary = target.summarize()

    if target.user.twitter:
        twitter_response = twitter.post(
            'statuses/update.json',
            data={'status': summary})
        if twitter_response.status != 200:
            raise ValueError("Could not publish to Twitter.")

    if target.user.facebook:
        fb_response = facebook.post('/me/feed', data={
            'message': summary
        })
        if fb_response.status != 200:
            raise ValueError("Could not publish to Facebook.")
```

我们的监听函数只需要一个目标（被操作的食谱实例）。我们通过之前编写的`Recipe.summarize()`方法获得食谱摘要，然后使用 OAuth 客户端对象的`post`方法（考虑到每个服务的不同端点 URI 和预期的负载格式）来创建用户已连接到的任何服务的状态更新。

### 提示

我们在这里定义的函数的错误处理代码有些低效；每个 API 可能返回不同的 HTTP 错误代码，很可能一个服务可能会接受帖子，而另一个服务可能会因为某种尚未知的原因而拒绝它。处理与多个远程第三方 API 交互时可能出现的各种故障模式是复杂的，可能是一本书的主题。

## 寻找共同的朋友

大多数现代的社交型网络应用程序的一个非常典型的特性是能够在你已经熟悉的应用程序上找到其他社交网络上的用户。这有助于您为应用程序实现任何类型的友谊/关注者模型。没有人喜欢在新平台上没有朋友，所以为什么不与您在其他地方已经交过的朋友联系呢？

通过找到用户在 Twitter 上正在关注的账户和当前存在于 Dinnerly 应用程序中的用户的交集，这相对容易实现。

### 注意

两个集合 A 和 B 的交集 C 是存在于 A 和 B 中的共同元素的集合，没有其他元素。

如果您还不了解数学集合的基本概念以及可以对其执行的操作，那么应该在您的阅读列表中加入一个关于天真集合论的入门课程。

我们首先添加一个路由处理程序，经过身份验证的用户可以查询该处理程序，以查找他们在`application/users.views.py`模块中的共同朋友列表。

```py
from flask import abort, render_template
from flask.ext.login import login_required

# …

@users.route('/twitter/find-friends')
@login_required
def twitter_find_friends():
    """Find common friends."""

    if not current_user.twitter:
        abort(403)

    twitter_user_id = current_user.twitter.twitter_user_id

    # This will only query 5000 Twitter user IDs.
    # If your users have more friends than that,
    # you will need to handle the returned cursor
    # values to iterate over all of them.
    response = twitter.get(
        'friends/ids?user_id={}'.format(twitter_user_id))

    friends = response.json().get('ids', list())
    friends = [int(f) for f in friends]

    common_friends = User.query.filter(
        User.twitter_user_id.in_(friends))

    return render_template('users/friends.html',
        friends=common_friends)
```

### 注意

在前面的方法中，我们使用了简单的`abort()`调用，但是没有阻止您创建模板，这些模板会呈现附加信息，以帮助最终用户理解为什么某个操作失败了。

前面的视图函数使用了我们可靠的 Flask-Login 扩展中的`login_required`装饰器进行包装，以确保对此路由的任何请求都是由经过身份验证的用户发出的。未经身份验证的用户由于某种明显的原因无法在 Dinnerly 上找到共同的朋友。

然后，我们确保经过身份验证的用户已连接了一组 Twitter OAuth 凭据，并取出`twitter_user_id`值，以便我们可以正确构建 Twitter API 请求，该请求要求用户的 ID 或`screen_name`。

### 提示

虽然`screen_name`可能比长数字标识符更容易调试和推理，但请记住，一个人随时可以在 Twitter 上更新`screen_name`。如果您想依赖这个值，您需要编写一些代码来验证并在远程服务上更改时更新本地存储的`screen_name`值。

一旦对远程服务上账户关注的人的 Twitter ID 进行了`GET`请求，我们解析这个结果并构建一个整数列表，然后将其传递给 User-mapped 类上的 SQLAlchemy 查询。现在我们已经获得了一个用户列表，我们可以将这些传递给我们的视图（我们不会提供实现，这留给读者作为练习）。

当然，找到共同的朋友只是方程的一半。一旦我们在 Twitter 上找到了我们的朋友，下一步就是在 Dinnerly 上也关注他们。为此，我们需要向我们的应用程序添加一个（最小的！）社交组件，类似于我们在上一章中实现的内容。

这将需要添加一些与数据库相关的实体，我们可以使用更新/添加相关模型的常规程序，然后重新创建数据库模式，但我们将利用这个机会来探索一种更正式的跟踪模式相关变化的方法。

# 插曲 - 数据库迁移

在应用程序开发的世界中，我们使用各种工具来跟踪和记录随时间变化的代码相关变化。一般来说，这些都属于版本控制系统的范畴，有很多选择：Git、Mercurial、Subversion、Perforce、Darcs 等。每个系统的功能略有不同，但它们都有一个共同的目标，即保存代码库的时间点快照（或代码库的部分，取决于所使用的工具），以便以后可以重新创建它。

Web 应用程序的一个方面通常难以捕捉和跟踪是数据库的当前状态。过去，我们通过存储整个 SQL 快照以及应用程序代码来解决这个问题，并指示开发人员删除并重新创建他们的数据库。对此的下一级改进将是创建一些小型基于 SQL 的脚本，应按特定顺序逐渐构建底层模式，以便在需要修改时，将另一个小型基于 SQL 的脚本添加到列表中。

虽然后一种方法非常灵活（它几乎可以适用于任何依赖关系数据库的应用程序），但是稍微抽象化，可以利用我们已经使用的 SQLAlchemy 对象关系模型的功能，这将是有益的。

## Alembic

这样的抽象已经存在，它叫做 Alembic。这个库由 SQLAlchemy 的相同作者编写，允许我们创建和管理对应于我们的 SQLAlchemy 数据模型所需的模式修改的变更集。

和我们在本书中讨论过的大多数库一样，Flask-Alembic 也被封装成了一个 Flask 扩展。让我们在当前的虚拟环境中安装它：

```py
$ pip install flask-alembic

```

由于大多数 Flask-Alembic 的功能可以和应该通过 CLI 脚本来控制，所以该软件包包括了启用 Flask-Script 命令的钩子。因此，让我们也安装这个功能：

```py
$ pip install flask-script

```

我们将创建我们的`manage.py` Python 脚本来控制我们的 CLI 命令，作为我们`application/包`的兄弟，并确保它包含用于集成 Flask-Alembic 的 db 钩子：

```py
from flask.ext.script import Manager, Shell, Server
from application import create_app, db
from flask_alembic.cli.script import manager as alembic_manager

# Create the `manager` object with a
# callable that returns a Flask application object.
manager = Manager(app=create_app)

def _context():
    """Adds additional objects to our default shell context."""
    return dict(db=db)

if __name__ == '__main__':
 manager.add_command('db', alembic_manager)
    manager.add_command('runserver', Server(port=6000))
    manager.add_command('shell', Shell(make_context=_context))
    manager.run()
```

现在我们已经安装了这两个扩展，我们需要配置 Flask-Alembic 扩展，以便它了解我们的应用对象。我们将在应用程序工厂函数中以通常的方式来做这个：

```py
# …
from flask.ext.alembic import Alembic

# …
# Intialize the Alembic extension
alembic = Alembic()

def create_app(config=None):
    app = Flask(__name__, static_folder=None)

    if config is not None:
        app.config.from_object(config)

    import application.users.models
    import application.recipes.models
       # …
 alembic.init_app(app)

    from application.users.views import users
    app.register_blueprint(users, url_prefix='/users')

    return app
```

让我们捕获当前数据库模式，这个模式是由我们在应用程序中定义的 SQLAlchemy 模型描述的：

```py
$ python manage.py db revision 'Initial schema.'

```

这将在`migrations/文件夹`中创建两个新文件（在第一次运行此命令时创建），其中一个文件将以一堆随机字符开头，后跟`_initial_schema.py`。

### 注意

看起来随机的字符实际上并不那么随机：它们是基于哈希的标识符，可以帮助迁移系统在多个开发人员同时为应用程序的不同部分工作迁移时以更可预测的方式运行，这在当今是相当典型的。

另一个文件`script.py.mako`是 Alembic 在调用命令时将使用的模板，用于生成这些自动修订摘要。这个脚本可以根据您的需要进行编辑，但不要删除任何模板`${foo}`变量！

生成的迁移文件包括两个函数定义：`upgrade()`和`downgrade()`。当 Alembic 获取当前数据库修订版（此时为`None`）并尝试将其带到目标（通常是最新）修订版时，将运行升级函数。`downgrade()`函数也是如此，但是方向相反。拥有这两个函数对于回滚类型的情况非常方便，当在包含不同迁移集的代码分支之间切换时，以及其他一些边缘情况。许多开发人员忽略了生成和测试降级迁移，然后在项目的生命周期的后期非常后悔。

根据您使用的关系数据库，您的确切迁移可能会有所不同，但它应该看起来类似于这样：

```py
"""Initial schema.

Revision ID: cd5ee4319a3
Revises:
Create Date: 2015-10-30 23:54:00.990549

"""

# revision identifiers, used by Alembic.
revision = 'cd5ee4319a3'
down_revision = None
branch_labels = ('default',)
depends_on = None

from alembic import op
import sqlalchemy as sa

def upgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=40), nullable=True),
    sa.Column('created_on', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('facebook_connection',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('facebook_id', sa.Integer(), nullable=False),
    sa.Column('access_token', sa.String(), nullable=False),
    sa.Column('name', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id')
    )
    op.create_table('recipe',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('title', sa.String(), nullable=True),
    sa.Column('ingredients', sa.Text(), nullable=True),
    sa.Column('instructions', sa.Text(), nullable=True),
    sa.Column('created_on', sa.DateTime(), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        op.f('ix_recipe_created_on'), 'recipe',
        ['created_on'], unique=False)
    op.create_table('twitter_connection',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('screen_name', sa.String(), nullable=False),
    sa.Column('twitter_user_id', sa.Integer(), nullable=False),
    sa.Column('oauth_token', sa.String(), nullable=False),
    sa.Column('oauth_token_secret', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id')
    )
    ### end Alembic commands ###

def downgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('twitter_connection')
    op.drop_index(
        op.f('ix_recipe_created_on'), table_name='recipe')
    op.drop_table('recipe')
    op.drop_table('facebook_connection')
    op.drop_table('user')
    ### end Alembic commands ###
```

现在，在这个脚本中有很多事情要做，或者至少看起来是这样。`upgrade()`函数中正在发生的是创建与我们在应用程序中定义的模型元数据和属于它们的字段相对应的表。通过比较当前模型定义和当前活动数据库模式，Alembic 能够推断出需要生成什么，并输出所需的命令列表来同步它们。

如果您熟悉关系数据库术语（列、主键、约束等），那么大多数语法元素应该相对容易理解，您可以在 Alembic 操作参考中阅读它们的含义：[`alembic.readthedocs.org/en/latest/ops.html`](http://alembic.readthedocs.org/en/latest/ops.html)

生成了初始模式迁移后，现在是应用它的时候了：

```py
$ python manage.py db upgrade

```

这将向您在 Flask-SQLAlchemy 配置中配置的关系型数据库管理系统发出必要的 SQL（基于生成的迁移）。

# 摘要

在这个相当冗长且内容丰富的章节之后，您应该会对 OAuth 及与 OAuth 相关的实现和一般术语感到更加放心，此外，数据库迁移的实用性，特别是由 Alembic 生成的与应用程序模型中声明的表和约束元数据同步的迁移风格。

本章从深入探讨 OAuth 授权授予流程和术语开始，考虑到 OAuth 的复杂性，这并不是一件小事！一旦我们建立了一定的知识基础，我们就实现了一个应用程序，利用 Flask-OAuthlib 为用户提供了创建账户并使用 Twitter 和 Facebook 等第三方服务进行登录的能力。

在完善示例应用程序的数据处理部分之后，我们转向了 Alembic，即 SQLAlchemy 数据迁移工具包，以将我们模型中的更改与我们的关系型数据库同步。

在本章开始的项目对于大多数具有社交意识的网络应用程序来说是一个很好的起点。我们强烈建议您利用本章和前几章学到的知识来创建一个现代、经过高度测试的、功能齐全的网络应用程序。
