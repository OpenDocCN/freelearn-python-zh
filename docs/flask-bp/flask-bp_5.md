# 第五章：Shutterbug，照片流 API

在本章中，我们将构建一个（主要是）基于 JSON 的 API，允许我们查看按时间顺序倒序排列的已添加照片列表——由于 Instagram 和类似的照片分享应用程序，这在近年来变得非常流行。为简单起见，我们将放弃许多这些应用程序通常围绕的社交方面；但是，我们鼓励您将前几章的知识与本章的信息相结合，构建这样的应用程序。

Shutterbug，我们即将开始的最小 API 应用程序，将允许用户通过经过身份验证的基于 JSON 的 API 上传他们选择的照片。

此外，我们将使用 Flask（实际上是 Werkzeug）的较少为人所知的功能之一，创建一个自定义中间件，允许我们拦截传入请求并修改全局应用程序环境，用于非常简单的 API 版本控制。

# 开始

和前几章一样，让我们为这个应用程序创建一个全新的目录和虚拟环境：

```py
$ mkdir -p ~/src/shutterbug && cd ~/src/shutterbug
$ mkvirtualenv shutterbug
$ pip install flask flask-sqlalchemy pytest-flask flask-bcrypt

```

创建以下应用程序布局以开始：

```py
├── application/
│   ├── __init__.py
│   └── resources
│       ├── __init__.py
│       └── photos.py
├── conftest.py
├── database.py
├── run.py
├── settings.py
└── tests/
```

### 注意

这里呈现的应用程序布局与我们在前几章中使用的典型基于 Blueprint 的结构不同；我们将使用典型 Flask-RESTful 应用程序建议的布局，这也适合 Shutterbug 应用程序的简单性。

# 应用程序工厂

在本章中，我们将再次使用应用程序工厂模式；让我们将我们的骨架`create_app`方法添加到`application/__init__.py`模块中，并包括我们的 Flask-SQLAlchemy 数据库初始化：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.bcrypt import Bcrypt

# Initialize the db extension, but without configuring
# it with an application instance.
db = SQLAlchemy()
flask_bcrypt = Bcrypt()

def create_app(config=None):
    app = Flask(__name__)

    if config is not None:
        app.config.from_object(config)

    db.init_app(app)
    flask_bcrypt.init_app(app)

    return app
```

让我们包含我们的基本`run.py`：

```py
from application import create_app

app = create_app()
app.run()
```

这应该使我们能够使用内置的 Werkzeug 应用程序服务器运行应用程序，代码如下：

```py
$ python run.py

```

# 插曲——Werkzeug

我们在本书的过程中已经几次谈到了 Werkzeug，但我们并没有真正解释它是什么，为什么我们使用它，或者它为什么有用。要理解 Werkzeug，我们首先需要知道它存在的原因。为此，我们需要了解 Python Web 服务器网关接口规范的起源，通常缩写为 WSGI。

如今，选择 Python Web 应用程序框架相对来说是一个相对简单的偏好问题：大多数开发人员根据以前的经验、必要性（例如，设计为异步请求处理的 Tornado）或其他可量化或不可量化的标准选择框架。

然而，几年前，应用程序框架的选择影响了您可以使用的 Web 服务器。由于当时所有 Python Web 应用程序框架以稍微不同的方式实现了它们自己的 HTTP 请求处理，它们通常只与 Web 服务器的子集兼容。开发人员厌倦了这种有点不方便的现状，提出了通过一个共同规范 WSGI 统一 Web 服务器与 Python 应用程序的交互的提案。

一旦建立了 WSGI 规范，所有主要框架都采用了它。此外，还创建了一些所谓的*实用*工具；它们的唯一目的是将官方 WSGI 规范与更健壮的中间 API 进行桥接，这有助于开发现代 Web 应用程序。此外，这些实用程序库可以作为更完整和健壮的应用程序框架的基础。

您现在可能已经猜到，Werkzeug 是这些 WSGI 实用程序库之一。当与模板语言 Jinja 和一些方便的默认配置、路由和其他基本 Web 应用程序必需品结合使用时，我们就有了 Flask。

Flask 是我们在本书中主要处理的内容，但是从 Werkzeug 中抽象出来的大部分工作都包含在其中。虽然它很大程度上不被注意到，但是可以直接与它交互，以拦截和修改请求的部分，然后 Flask 有机会处理它。在本章中，当我们为 JSON API 请求实现自定义 Werkzeug 中间件时，我们将探索其中的一些可能性。

# 使用 Flask-RESTful 创建简单的 API

使用 Flask 的一个巨大乐趣是它提供了看似无限的可扩展性和可组合性。由于它是一个相当薄的层，位于 Werkzeug 和 Jinja 之上，因此在约束方面对开发人员的要求并不多。

由于这种灵活性，我们可以利用 Flask-RESTful 等扩展，使得创建基于 JSON 的 API 变得轻松愉快。首先，让我们安装这个包：

```py
$ pip install flask-restful

```

接下来，让我们以通常的方式在我们的应用工厂中初始化这个扩展：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.bcrypt import Bcrypt
from flask.ext.restful import Api

# ………
api = Api()

def create_app(config=None):
    app = Flask(__name__)

    if config is not None:
        app.config.from_object(config)

    db.init_app(app)
    flask_bcrypt.init_app(app)

 api.init_app(app)

    return app
```

Flask-RESTful 扩展的主要构建块是资源的概念。资源在本质上是一个带有一些非常有用的默认设置的`Flask`方法视图，用于内容类型协商。如果直到现在你还没有遇到过 Flask 中`MethodView`的概念，不要担心！它们非常简单，并且通过允许您在类上定义方法，直接映射到基本的 HTTP 动词：`GET`、`PUT`、`POST`、`PATCH`和`DELETE`，为您提供了一个相对简单的接口来分离 RESTful 资源。Flask-RESTful 资源又扩展了`MethodView`类，因此允许使用相同的基于动词的路由处理风格。

更具体地说，这意味着 Flask-RESTful API 名词可以以以下方式编写。我们将首先将我们的照片资源视图处理程序添加到`application/resources/photos.py`中：

```py
class SinglePhoto(Resource):

    def get(self, photo_id):
        """Handling of GET requests."""
        pass

    def delete(self, photo_id):
        """Handling of DELETE requests."""
        pass

class ListPhoto(Resource):

    def get(self):
        """Handling of GET requests."""
        pass

    def post(self):
        """Handling of POST requests."""
        pass
```

### 注意

在前面的两个`Resource`子类中，我们定义了可以处理的 HTTP 动词的一个子集；我们并不需要为所有可能的动词定义处理程序。例如，如果我们的应用程序接收到一个 PATCH 请求到前面的资源中的一个，Flask 会返回 HTTP/1.1 405 Method Not Allowed。

然后，我们将这些视图处理程序导入到我们的应用工厂中，在`application/__init__.py`中，以便将这两个类绑定到我们的 Flask-RESTful API 对象：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.restful import Api
from flask.ext.bcrypt import Bcrypt

# Initialize the db extension, but without configuring
# it with an application instance.
db = SQLAlchemy()
api = Api()
flask_bcrypt = Bcrypt()

def create_app(config=None):
    app = Flask(__name__)

    if config is not None:
        app.config.from_object(config)

    db.init_app(app)
    flask_bcrypt.init_app(app)

 from .resources.photos import SinglePhoto, ListPhoto
 api.add_resource(ListPhoto, '/photos')
 api.add_resource(SinglePhoto, '/photos/<int:photo_id>')

    api.init_app(app)

    return app
```

### 注意

请注意，在调用`api.init_app(app)`之前，我们已经将资源绑定到了 API 对象。如果我们在绑定资源之前初始化，路由将不存在于 Flask 应用程序对象上。

我们可以通过启动交互式 Python 会话并检查 Flask 应用程序的`url_map`属性来确认我们定义的路由是否映射到应用程序对象。

### 提示

从应用程序文件夹的父文件夹开始会话，以便正确设置`PYTHONPATH`：

```py
In [1]: from application import create_app
In [2]: app = create_app()
In [3]: app.url_map
Out[3]:
Map([<Rule '/photos' (HEAD, POST, OPTIONS, GET) -> listphoto>,
 <Rule '/photos/<photo_id>' (HEAD, DELETE, OPTIONS, GET) -> singlephoto>,
 <Rule '/static/<filename>' (HEAD, OPTIONS, GET) -> static>])

```

前面的输出列出了一个 Werkzeug `Map`对象，其中包含三个`Rule`对象，每个对象列出了一个 URI，对该 URI 有效的 HTTP 动词，以及一个标准化标识符（视图处理程序可以是函数，也可以是`MethodView`子类，还有其他几个选项），指示将调用哪个视图处理程序。

### 注意

Flask 将自动处理所有已定义端点的 HEAD 和 OPTIONS 动词，并为静态文件处理添加一个默认的`/static/<filename>`路由。如果需要，可以通过在应用程序工厂中对`Flask`对象初始化设置`static_folder`参数为`None`来禁用此默认静态路由：

```py
 app = Flask(__name__, static_folder=None)

```

让我们对我们的骨架用户视图资源处理程序做同样的事情，我们将在`application/resources/users.py`中声明：

```py
from flask.ext.restful import Resource

class SingleUser(Resource):

    def get(self, user_id):
        """Handling of GET requests."""
        pass

class CreateUser(Resource):

    def post(self):
        """Handling of POST requests."""
        pass
```

### 注意

请注意，我们本可以将`post`方法处理程序放在`SingleUser`资源定义中，但相反，我们将其拆分为自己的资源。这并非绝对必要，但会使我们的应用程序更容易跟踪，并且只会花费我们额外的几行代码。

与我们在照片视图中所做的类似，我们将把它们添加到我们的 Flask-RESTful API 对象中的应用工厂中：

```py
def create_app(config=None):

    # …

    from .resources.photos import SinglePhoto, ListPhoto
    from .resources.users import SingleUser, CreateUser

    api.add_resource(ListPhoto, '/photos')
    api.add_resource(SinglePhoto, '/photos/<int:photo_id>')
    api.add_resource(SingleUser, '/users/<int:user_id>')
    api.add_resource(CreateUser, '/users')

    api.init_app(app)
    return app
```

## 使用混合属性改进密码处理

我们的`User`模型将与我们在上一章中使用的模型非常相似，并且将使用类属性`getter`/`setter`来处理`password`属性。这将确保无论我们是在对象创建时设置值还是手动设置已创建对象的属性，都能一致地应用 Bcrypt 密钥派生函数到原始用户密码。

这包括使用 SQLAlchemy 的`hybrid_property`描述符，它允许我们定义在类级别访问时（例如`User.password`，我们希望返回用户模型的密码字段的 SQL 表达式）与实例级别访问时（例如`User().password`，我们希望返回用户对象的实际加密密码字符串而不是 SQL 表达式）行为不同的属性。

我们将把密码类属性定义为`_password`，这将确保我们避免任何不愉快的属性/方法名称冲突，以便我们可以正确地定义混合的`getter`和`setter`方法。

由于我们的应用在数据建模方面相对简单，我们可以在`application/models.py`中使用单个模块来处理我们的模型：

```py
from application import db, flask_bcrypt
from sqlalchemy.ext.hybrid import hybrid_property

import datetime

class User(db.Model):
    """SQLAlchemy User model."""

    # The primary key for each user record.
    id = db.Column(db.Integer, primary_key=True)

    # The unique email for each user record.
    email = db.Column(db.String(255), unique=True, nullable=False)

    # The unique username for each record.
    username = db.Column(db.String(40), unique=True, nullable=False)

 # The bcrypt'ed user password
 _password = db.Column('password', db.String(60), nullable=False)

    #  The date/time that the user account was created on.
    created_on = db.Column(db.DateTime,
       default=datetime.datetime.utcnow)

    def __repr__(self):
        return '<User %r>' % self.username

 @hybrid_property
 def password(self):
 """The bcrypt'ed password of the given user."""

 return self._password

 @password.setter
 def password(self, password):
 """Bcrypt the password on assignment."""

        self._password = flask_bcrypt.generate_password_hash(password)
```

在同一个模块中，我们可以声明我们的`Photo`模型，它将负责维护与图像相关的所有元数据，但不包括图像本身：

```py
class Photo(db.Model):
    """SQLAlchemy Photo model."""

    # The unique primary key for each photo created.
    id = db.Column(db.Integer, primary_key=True)

    # The free-form text-based comment of each photo.
    comment = db.Column(db.Text())

    # Path to photo on local disk
    path = db.Column(db.String(255), nullable=False)

    #  The date/time that the photo was created on.
    created_on = db.Column(db.DateTime(),
        default=datetime.datetime.utcnow, index=True)

    # The user ID that created this photo.
    user_id = db.Column(db.Integer(), db.ForeignKey('user.id'))

    # The attribute reference for accessing photos posted by this user.
    user = db.relationship('User', backref=db.backref('photos',
        lazy='dynamic'))

    def __repr__(self):
        return '<Photo %r>' % self.comment
```

## API 身份验证

对于大多数应用程序和 API，身份验证和授权的概念对于非平凡操作至关重要：

+   **身份验证**：这断言所提供的凭据的真实性，并确保它们属于已知实体；简单来说，这意味着确保提供给应用程序的用户名和密码属于有效用户。一旦验证，应用程序就会假定使用这些凭据执行的请求是代表给定用户执行的。

+   **授权**：这是经过身份验证的实体在应用程序范围内的可允许操作。在大多数情况下，授权预设了已经进行了预先身份验证步骤。实体可能已经经过身份验证，但没有被授权访问某些资源：如果您在 ATM 机上输入您的卡和 PIN 码（因此进行了身份验证），您可以查看自己的账户，但尝试查看另一个人的账户将会（希望！）导致拒绝，因为您没有被授权访问那些信息。

对于 Shutterbug，我们只关心身份验证。如果我们要添加各种功能，比如能够创建可以访问共享照片池的私人用户组，那么就需要系统化的授权来确定哪些用户可以访问哪些资源的子集。

### 身份验证协议

许多开发人员可能已经熟悉了几种身份验证协议：通常的标识符/密码组合是现有大多数网络应用程序的标准，而 OAuth 是许多现代 API 的标准（例如 Twitter、Facebook、GitHub 等）。对于我们自己的应用程序，我们将使用非常简单的 HTTP 基本身份验证协议。

虽然 HTTP 基本身份验证并不是最灵活也不是最安全的（实际上它根本不提供任何加密），但对于简单的应用程序、演示和原型 API 来说，实施这种协议是合理的。在 Twitter 早期，这实际上是您可以使用的唯一方法来验证其 API！此外，在通过 HTTPS 传输数据时，我们应该在任何生产级环境中这样做，我们可以确保包含用户标识和密码的明文请求受到加密，以防止任何可能监听的恶意第三方。

HTTP 基本身份验证的实现并不是过于复杂的，但绝对是我们可以转嫁给扩展的东西。让我们继续将 Flask-HTTPAuth 安装到我们的环境中，这包括创建扩展的实例：

```py
$ pip install flask-httpauth

```

并在我们的`application/__init__.py`中设置扩展：

```py
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.restful import Api
from flask.ext.bcrypt import Bcrypt
from flask.ext.httpauth import HTTPBasicAuth

# …

api = Api()
flask_bcrypt = Bcrypt()
auth = HTTPBasicAuth()

def create_app(config=None):
    # …

 import authentication

    api.add_resource(ListPhoto, '/photos')
    api.add_resource(SinglePhoto, '/photos/<int:photo_id>')

    # …
```

Flask-HTTPAuth 包括各种装饰器来声明处理程序/回调，以执行身份验证过程的各个部分。我们将实现一个可以最大程度控制身份验证方式的处理程序，并将其放在`application/authentication.py`中的新模块中。除了验证凭据外，我们还将在成功验证时将 SQLAlchemy 用户对象附加到 Flask 上下文本地`g`，以便我们可以在请求处理和响应生成的其他部分中利用这些数据：

```py
import sqlalchemy
from . import auth, flask_bcrypt
from .models import User
from flask import g

@auth.verify_password
def verify_password(username, password):
    """Verify a username/hashed password tuple."""

    try:
        user = User.query.filter_by(username=username).one()
    except sqlalchemy.orm.exc.NoResultFound:
        # We found no username that matched
        return False

    # Perform password hash comparison in time-constant manner.
    verified = flask_bcrypt.check_password_hash(user.password,
        password)

 if verified is True:
 g.current_user = user

    return verified
```

`auth.verify_password`装饰器允许我们指定一个接受用户名和密码的函数，这两者都从发送请求的 Authorization 头中提取出来。然后，我们将使用这些信息来查询具有相同用户名的用户的数据库，并在成功找到一个用户后，我们将确保提供的密码散列到与我们为该用户存储的相同值。如果密码不匹配或用户名不存在，我们将返回 False，Flask-HTTPAuth 将向请求客户端返回 401 未经授权的标头。

现在，要实际使用 HTTP 基本身份验证，我们需要将`auth.login_required`装饰器添加到需要身份验证的视图处理程序中。我们知道除了创建新用户之外，所有用户操作都需要经过身份验证的请求，所以让我们实现这一点：

```py
from flask.ext.restful import Resource
from application import auth

class SingleUser(Resource):

 method_decorators = [auth.login_required]

    def get(self, user_id):
        """Handling of GET requests."""
        pass

    # …
```

### 注意

由于 Resource 对象的方法的 self 参数指的是 Resource 实例而不是方法，我们不能在视图的各个方法上使用常规视图装饰器。相反，我们必须使用`method_decorators`类属性，它将按顺序应用已声明的函数到已调用的视图方法上，以处理请求。

## 获取用户

现在我们已经弄清楚了应用程序的身份验证部分，让我们实现 API 端点以创建新用户和获取现有用户数据。我们可以如下完善`SingleUser`资源类的`get()`方法：

```py
from flask.ext.restful import abort

# …

def get(self, user_id):
    """Handling of GET requests."""

    if g.current_user.id != user_id:
        # A user may only access their own user data.
        abort(403, message="You have insufficient permissions"
            " to access this resource.")

    # We could simply use the `current_user`,
    # but the SQLAlchemy identity map makes this a virtual
    # no-op and alos allows for future expansion
    # when users may access information of other users
    try:
        user = User.query.filter(User.id == user_id).one()
    except sqlalchemy.orm.exc.NoResultFound:
        abort(404, message="No such user exists!")

    data = dict(
        id=user.id,
        username=user.username,
        email=user.email,
        created_on=user.created_on)

    return data, 200
```

在前面的方法中发生了很多新的事情，让我们来分解一下。首先，我们将检查请求中指定的`user_id`（例如，`GET /users/1`）是否与当前经过身份验证的用户相同：

```py
if g.current_user.id != user_id:
        # A user may only access their own user data.
        abort(403, message="You have insufficient permissions"
            " to access this resource.")
```

虽然目前这可能看起来有些多余，但它在允许将来更简单地修改授权方案的同时，还扮演了遵循更符合 RESTful 方法的双重角色。在这里，资源是由其 URI 唯一指定的，部分由用户对象的唯一主键标识符构成。

经过授权检查后，我们将通过查询传递为命名 URI 参数的`user_id`参数，从数据库中提取相关用户：

```py
try:
    user = User.query.filter(User.id == user_id).one()
except sqlalchemy.orm.exc.NoResultFound:
    abort(404, message="No such user exists!")
```

如果找不到这样的用户，那么我们将使用 HTTP 404 Not Found 中止当前请求，并指定消息以使非 20x 响应的原因更清晰。

最后，我们将构建一个用户数据的字典，作为响应返回。我们显然不希望返回散列密码或其他敏感信息，因此我们将明确指定我们希望在响应中序列化的字段：

```py
data = dict(id=user.id, username=user.username, email=user.email,
            created_on=user.created_on)

    return data, 200
```

由于 Flask-RESTful，我们不需要显式地将我们的字典转换为 JSON 字符串：响应表示默认为`application/json`。然而，有一个小问题：Flask-RESTful 使用的默认 JSON 编码器不知道如何将 Python `datetime`对象转换为它们的 RFC822 字符串表示。这可以通过指定`application/json` MIME 类型表示处理程序并确保我们使用`flask.json`编码器而不是 Python 标准库中的默认`json`模块来解决。

我们可以在`application/__init__.py`模块中添加以下内容：

```py
from flask import Flask, json, make_response
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.restful import Api
from flask.ext.bcrypt import Bcrypt
from flask.ext.httpauth import HTTPBasicAuth

# …

db = SQLAlchemy()
# …

@api.representation('application/json')
def output_json(data, code, headers=None):
    resp = make_response(json.dumps(data), code)
    resp.headers.extend(headers or {})
    return resp
```

### 创建新用户

从 API 中获取现有用户的类比当然是创建新用户。而典型的 Web 应用程序通过填写各种表单字段来完成这一过程，通过我们的 API 创建新用户需要将信息通过 POST 请求提交到服务器进行验证，然后将新用户插入数据库。这些步骤的实现应该放在我们的`CreateUser`资源的`post()`方法中：

```py
class CreateUser(Resource):

    def post(self):
        """Create a new user."""

        data = request.json
        user = User(**data)

        db.session.add(user)

        try:
            db.session.commit()
        except sqlalchemy.exc.IntegrityError:
            abort(409, message="User already exists!")

        data = dict(id=user.id, username=user.username, email=user.email, created_on=user.created_on)

        return data, 201, {'Location': url_for( 'singleuser', user_id=user.id, _external=True)}
```

### 注意

如果请求的内容类型设置为`application/json`，则`request.json`文件将填充 POST 数据。

在前面的方法实现中没有什么太意外的：我们从`request.json`中获取了 POST 数据，创建了一个`User`对象（非常不安全！您可以在本章稍后看到更好的替代方法），尝试将其添加到数据库中并捕获异常，如果同一用户名或电子邮件地址的用户已经存在，然后序列化一个 HTTP 201 Created 响应，其中包含新创建用户的 URI 的`Location`头。

#### 输入验证

虽然 Flask 包含一个相对简单的方式来通过`flask.request`代理对象访问 POST 的数据，但它不包含任何功能来验证数据是否按我们期望的格式进行格式化。这没关系！Flask 试图尽可能地与数据存储和操作无关，将这些工作留给开发人员。幸运的是，Flask-RESTful 包括`reqparse`模块，可以用于数据验证，其使用在精神上与用于 CLI 参数解析的流行`argparse`库非常相似。

我们将在`application/resources/users.py`模块中设置我们的新用户数据解析器/验证器，并声明我们的字段及其类型以及在 POST 数据中是否为有效请求所需的字段：

```py
from flask.ext.restful import Resource, abort, reqparse, url_for

# …

new_user_parser = reqparse.RequestParser()
new_user_parser.add_argument('username', type=str, required=True)
new_user_parser.add_argument('email', type=str, required=True)
new_user_parser.add_argument('password', type=str, required=True)
```

现在我们在模块中设置了`new_user_parser`，我们可以修改`CreateUser.post()`方法来使用它：

```py
def post(self):
    """Handling of POST requests."""

    data = new_user_parser.parse_args(strict=True)
    user = User(**data)

    db.session.add(user)

    # …
```

`new_user_parser.parse_args(strict=True)`的调用将尝试匹配我们之前通过`add_argument`定义的字段的声明类型和要求，并且在请求中存在任何字段未通过验证或者有额外字段没有明确考虑到的情况下，将内部调用`abort()`并返回 HTTP 400 错误（感谢`strict=True`选项）。

使用`reqparse`来验证 POST 的数据可能比我们之前直接赋值更加繁琐，但是安全性更高。通过直接赋值技术，恶意用户可能会发送任意数据，希望覆盖他们不应该访问的字段。例如，我们的数据库可能包含内部字段`subscription_exipires_on datetime`，一个恶意用户可能会提交一个包含这个字段值设置为遥远未来的 POST 请求。这绝对是我们想要避免的事情！

### API 测试

让我们应用一些我们在之前章节中学到的关于使用`pytest`进行功能和集成测试的知识。

我们的第一步（在必要的 pip 安装`pytest-flask`之后）是像我们在之前的章节中所做的那样添加一个`conftest.py`文件，它是我们`application/`文件夹的同级文件夹。

```py
import pytest
import os
from application import create_app, db as database

DB_LOCATION = '/tmp/test_shutterbug.db'

@pytest.fixture(scope='session')
def app():
    app = create_app(config='test_settings')
    return app

@pytest.fixture(scope='function')
def db(app, request):
    """Session-wide test database."""
    if os.path.exists(DB_LOCATION):
        os.unlink(DB_LOCATION)

    database.app = app
    database.create_all()

    def teardown():
        database.drop_all()
        os.unlink(DB_LOCATION)

    request.addfinalizer(teardown)
    return database

@pytest.fixture(scope='function')
def session(db, request):

    session = db.create_scoped_session()
    db.session = session

    def teardown():
        session.remove()

    request.addfinalizer(teardown)
    return session
```

前面的`conftest.py`文件包含了我们编写 API 测试所需的基本测试装置；这里不应该有任何意外。然后我们将添加我们的`test_settings.py`文件，它是新创建的`conftest.py`的同级文件，并填充它与我们想要在测试运行中使用的应用程序配置值：

```py
SQLALCHEMY_DATABASE_URI = 'sqlite:////tmp/test_shutterbug.db'
SECRET_KEY = b"\x98\x9e\xbaP'D\x03\xf5\x91u5G\x1f"
DEBUG = True
UPLOAD_FOLDER = '/tmp/'
TESTING = True

```

一旦这些都就位，我们就可以开始在`tests/test_users.py`中编写我们的测试函数和断言。我们的第一个测试将确保我们可以通过 API 创建一个新用户，并且新创建的资源的 URI 将在`Location`标头中返回给我们：

```py
from application.models import User
from flask import json
import base64

def test_create_new_user(db, session, client):
    """Attempt to create a basic user."""

    data = {'username': 'you', 'email': 'you@example.com',
            'password': 'foobar'}

    response = client.post('/users', data=data)
    assert response.status_code == 201
    assert 'Location' in response.headers

    user = User.query.filter(User.username == data['username']).one()

    assert '/users/{}'.format(user.id) in response.headers['Location']
```

一旦我们确定可以创建用户，下一个逻辑步骤是测试如果客户端尝试使用无效或缺少的参数创建用户，则会返回错误：

```py
def test_create_invalid_user(db, session, client):
    """Try to create a user with invalid/missing information."""

    data = {'email': 'you@example.com'}
    response = client.post('/users', data=data)

    assert response.status_code == 400
    assert 'message' in response.json
    assert 'username' in response.json['message']
```

作为对我们的 HTTP 基本身份验证实现的健全性检查，让我们还添加一个测试来获取单个用户记录，这需要对请求进行身份验证：

```py
def test_get_single_user_authenticated(db, session, client):
    """Attempt to fetch a user."""

    data = {'username': 'authed', 'email': 'authed@example.com',
            'password': 'foobar'}
    user = User(**data)
    session.add(user)
    session.commit()

    creds = base64.b64encode(
        b'{0}:{1}'.format(
            user.username, data['password'])).decode('utf-8')

    response = client.get('/users/{}'.format(user.id),
        headers={'Authorization': 'Basic ' + creds})

    assert response.status_code == 200
    assert json.loads(response.get_data())['id'] == user.id
```

未经身份验证的请求获取单个用户记录的相关测试如下：

```py
def test_get_single_user_unauthenticated(db, session, client):
    data = {'username': 'authed', 'email': 'authed@example.com',
            'password': 'foobar'}
    user = User(**data)
    session.add(user)
    session.commit()

    response = client.get('/users/{}'.format(user.id))
    assert response.status_code == 401
```

我们还可以测试我们非常简单的授权实现是否按预期运行（回想一下，我们只允许经过身份验证的用户查看自己的信息，而不是系统中其他任何用户的信息。）通过创建两个用户并尝试通过经过身份验证的请求访问彼此的数据来进行测试：

```py
def test_get_single_user_unauthorized(db, session, client):

    alice_data = {'username': 'alice', 'email': 'alice@example.com',
            'password': 'foobar'}
    bob_data = {'username': 'bob', 'email': 'bob@example.com',
            'password': 'foobar'}
    alice = User(**alice_data)
    bob = User(**bob_data)

    session.add(alice)
    session.add(bob)

    session.commit()

    alice_creds = base64.b64encode(b'{0}:{1}'.format(
        alice.username, alice_data['password'])).decode('utf-8')

    bob_creds = base64.b64encode(b'{0}:{1}'.format(
        bob.username, bob_data['password'])).decode('utf-8')

    response = client.get('/users/{}'.format(alice.id),
        headers={'Authorization': 'Basic ' + bob_creds})

    assert response.status_code == 403

    response = client.get('/users/{}'.format(bob.id),
        headers={'Authorization': 'Basic ' + alice_creds})

    assert response.status_code == 403
```

## 插曲 - Werkzeug 中间件

对于某些任务，我们有时需要在将请求路由到处理程序函数或方法之前修改传入请求数据和/或环境的能力。在许多情况下，实现这一点的最简单方法是使用`before_request`装饰器注册一个函数；这通常用于在`g`对象上设置`request-global`值或创建数据库连接。

虽然这应该足够涵盖大部分最常见的用例，但有时在 Flask 应用程序对象下方（构造请求代理对象时）但在 HTTP 服务器上方更方便。为此，我们有中间件的概念。此外，一个正确编写的中间件将在其他兼容的 WSGI 实现中可移植；除了应用程序特定的怪癖外，没有什么能阻止您在我们当前的 Flask 应用程序中使用最初为 Django 应用程序编写的中间件。

中间件相对简单：它们本质上是任何可调用的东西（类、实例、函数或方法，可以以类似于函数的方式调用），以便返回正确的响应格式，以便链中的其他中间件可以正确调用。

对于我们当前基于 API 的应用程序有用的中间件的一个例子是，它允许我们从请求 URI 中提取可选的版本号，并将此信息存储在环境中，以便在请求处理过程中的各个点使用。例如，对`/v0.1a/users/2`的请求将被路由到`/users/2`的处理程序，并且`v0.1a`将通过`request.environ['API_VERSION']`在 Flask 应用程序本身中可访问。

在`application/middlewares.py`中的新模块中，我们可以实现如下：

```py
import re

version_pattern = re.compile(r"/v(?P<version>[0-9a-z\-\+\.]+)", re.IGNORECASE)

class VersionedAPIMiddleware(object):
    """

    The line wrapping here is a bit off, but it's not critical.

    """

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')

        match = version_pattern.match(path)

        if match:
            environ['API_VERSION'] = match.group(1)
            environ['PATH_INFO'] = re.sub(version_pattern, '', path,
                count=1)
        else:
            environ['API_VERSION'] = None

        return self.app(environ, start_response)
```

我们将在工厂中将此中间件绑定到应用程序对象：

```py
# …

from .middlewares import VersionedAPIMiddleware

# …
def create_app(config=None):
    app = Flask(__name__, static_folder=None)
 app.wsgi_app = VersionedAPIMiddleware(app.wsgi_app)

    # …

    api.init_app(app)
    return app
```

### 注意

在添加多个 WSGI 中间件时，它们的顺序有时很重要。在添加可能修改 WSGI 环境的中间件时，请务必记住这一点。

一旦绑定，中间件将在 Flask 接收请求之前插入请求处理，即使我们明确实例化了一个 Flask 应用程序对象。在应用程序中访问`API_VERSION`值只是简单地查询绑定到请求环境的键：

```py
from flask import request
# …
# …
if request.environ['API_VERSION'] > 2:
    # Handle this differently
else:
    # Handle it normally
```

API 版本号的解析也可以扩展到检查 HTTP 头（自定义或其他），除了我们在此提供的基于 URL 的版本提取；可以为任一方便性提出论点。

### 回到 Shutterbug - 上传照片

现在我们有了一个最小但功能齐全的 API 来创建和获取用户，我们需要一个类似的 API 来上传照片。首先，我们将使用与之前相同的资源模式，另外定义一个`RequestParser`实例来验证有关照片的用户提交数据：

```py
from flask.ext.restful import Resource, reqparse
from flask import current_app, request, g, url_for
from application import auth, db, models
import uuid
import os
import werkzeug

new_photo_parser = reqparse.RequestParser()
new_photo_parser.add_argument('comment', type=str,
    required=False)
new_photo_parser.add_argument('photo',
    type=werkzeug.datastructures.FileStorage,
    required=True, location='files')

class UploadPhoto(Resource):

    method_decorators = [auth.login_required]

    def post(self):
        """Adds a new photo via form-encoded POST data."""

        data = new_photo_parser.parse_args(strict=True)

        # Save our file to the filesystem first
        f = request.files['photo']

        extension = os.path.splitext(f.filename)[1]
        name = werkzeug.utils.secure_filename(
            str(uuid.uuid4()) + extension)
        path = os.path.join(
            current_app.config['UPLOAD_FOLDER'], name)

        f.save(path)

        data['user_id'] = g.current_user.id
        data['path'] = path

        # Get rid of the binary data that was sent; we've already
        # saved this to disk.
        del data['photo']

        # Add a new Photo entry to the database once we have
        # successfully saved the file to the filesystem above.
        photo = models.Photo(**data)
        db.session.add(photo)
        db.session.commit()

        data = dict(id=photo.id,
            path=photo.path, comment=photo.comment,
            created_on=photo.created_on)

        return data, 201, {'Location': url_for('singlephoto',
            photo_id=photo.id, _external=True)}
```

请注意，在前面的`UploadPhoto`资源中，我们正在访问`request.files`以提取通过 POST 发送到端点的二进制数据。然后，我们解析出扩展名，生成一个唯一的随机字符串作为文件名，最后将文件保存到我们在应用程序配置中配置的已知`UPLOAD_FOLDER`中。

### 注意

请注意，我们使用`werkzeug.utils.secure_filename`函数来净化上传图像的扩展名，以确保它不容易受到路径遍历或其他基于文件系统的利用的影响，这在处理用户上传的二进制数据时很常见。

在接受将持久化到文件系统的不受信任数据时，应该执行许多其他验证和净化步骤（例如，确保文件的 MIME 类型与实际上传的扩展名和二进制数据匹配，限制图像的大小/尺寸），但出于简洁起见，我们省略了它们。数据验证技术和最佳实践本身就可以填满一整本书。

我们最终将图像持久化到的本地文件系统路径与可能陪伴照片上传的可选评论一起添加到我们的照片 SQLAlchemy 记录中。然后将整个记录添加到会话中，并提交到数据库，然后在标头中返回新创建的资产的位置的 201 响应。在这里，我们避免处理一些简单的错误条件，以便我们可以专注于所呈现的核心概念，并将它们的实现留给读者作为练习。

在尝试任何新的照片上传功能之前，请确保将资源绑定到我们应用程序工厂中的 API 对象：

```py
def create_app(config=None):
    # …

 from .resources.photos import (SinglePhoto, ListPhoto,
 UploadPhoto)
 # …

    api.add_resource(ListPhoto, '/photos')
 api.add_resource(UploadPhoto, '/photos')
    api.add_resource(SinglePhoto, '/photos/<int:photo_id>')
    api.add_resource(SingleUser, '/users/<int:user_id>')
    api.add_resource(CreateUser, '/users')

    # …
```

#### 分布式系统中的文件上传

我们已经大大简化了现代 Web 应用程序中文件上传的处理。当然，简单通常有一些缺点。

其中最明显的是，在前面的实现中，我们受限于单个应用服务器。如果存在多个应用服务器，则确保上传的文件在这些多个服务器之间保持同步将成为一个重大的运营问题。虽然有许多解决这个特定问题的解决方案（例如，分布式文件系统协议，如 NFS，将资产上传到远程存储，如 Amazon 的**简单存储服务**（**S3**）等），但它们都需要额外的思考和考虑来评估它们的利弊以及对应用程序结构的重大更改。

### 测试照片上传

由于我们正在进行一些测试，让我们通过在`tests/test_photos.py`中编写一些简单的测试来保持这个过程。首先，让我们尝试使用未经身份验证的请求上传一些二进制数据：

```py
import io
import base64
from application.models import User, Photo

def test_unauthenticated_form_upload_of_simulated_file(session, client):
    """Ensure that we can't upload a file via un-authed form POST."""

    data = dict(
        file=(io.BytesIO(b'A test file.'), 'test.png'))

    response = client.post('/photos', data=data)
    assert response.status_code == 401
```

然后，让我们通过正确验证的请求来检查明显的成功路径：

```py
def test_authenticated_form_upload_of_simulated_file(session, client):
    """Upload photo via POST data with authenticated user."""

    password = 'foobar'
    user = User(username='you', email='you@example.com',
        password=password)

    session.add(user)

    data = dict(
        photo=(io.BytesIO(b'A test file.'), 'test.png'))

    creds = base64.b64encode(
        b'{0}:{1}'.format(user.username, password)).decode('utf-8')

    response = client.post('/photos', data=data,
        headers={'Authorization': 'Basic ' + creds})

    assert response.status_code == 201
    assert 'Location' in response.headers

    photos = Photo.query.all()
    assert len(photos) == 1

    assert ('/photos/{}'.format(photos[0].id) in
        response.headers['Location'])
```

最后，让我们确保在提交（可选）评论时，它被持久化到数据库中：

```py
def test_upload_photo_with_comment(session, client):
    """Adds a photo with a comment."""

    password = 'foobar'
    user = User(username='you', email='you@example.com',
    password=password)

    session.add(user)

    data = dict(
        photo=(io.BytesIO(b'A photo with a comment.'),
        'new_photo.png'),
        comment='What an inspiring photo!')

    creds = base64.b64encode(
        b'{0}:{1}'.format(
            user.username, password)).decode('utf-8')

    response = client.post('/photos', data=data,
        headers={'Authorization': 'Basic ' + creds})

    assert response.status_code == 201
    assert 'Location' in response.headers

    photos = Photo.query.all()
    assert len(photos) == 1

    photo = photos[0]
    assert photo.comment == data['comment']
```

## 获取用户的照片

除了上传照片的能力之外，Shutterbug 应用程序的核心在于能够以逆向时间顺序获取经过认证用户上传的照片列表。为此，我们将完善`application/resources/photos.py`中的`ListPhoto`资源。由于我们希望能够对返回的照片列表进行分页，我们还将创建一个新的`RequestParser`实例来处理常见的页面/限制查询参数。此外，我们将使用 Flask-RESTful 的编组功能来序列化从 SQLAlchemy 返回的`Photo`对象，以便将它们转换为 JSON 并发送到请求的客户端。

### 注意

**编组**是 Web 应用程序（以及大多数其他类型的应用程序！）经常做的事情，即使你可能从未听说过这个词。简单地说，你将数据转换成更适合传输的格式，比如 Python 字典或列表，然后将其转换为 JSON 格式，并通过 HTTP 传输给发出请求的客户端。

```py
from flask.ext.restful import Resource, reqparse, fields, marshal
photos_parser = reqparse.RequestParser()
photos_parser.add_argument('page', type=int, required=False,
        default=1, location='args')
photos_parser.add_argument('limit', type=int, required=False,
        default=10, location='args')

photo_fields = {
    'path': fields.String,
    'comment': fields.String,
    'created_on': fields.DateTime(dt_format='rfc822'),
}

class ListPhoto(Resource):

    method_decorators = [auth.login_required]

    def get(self):
        """Get reverse chronological list of photos for the
        currently authenticated user."""

        data = photos_parser.parse_args(strict=True)
        offset = (data['page'] - 1) * data['limit']
        photos = g.current_user.photos.order_by(
            models.Photo.created_on.desc()).limit(
            data['limit']).offset(offset)

        return marshal(list(photos), photo_fields), 200
```

请注意，在前面的`ListPhoto.get()`处理程序中，我们根据请求参数提供的页面和限制计算了一个偏移值。页面和限制与我们的数据集大小无关，并且易于理解，适用于消费 API 的客户端。SQLAlchemy（以及大多数数据库 API）只理解偏移和限制。转换公式是众所周知的，并适用于任何排序的数据集。

# 摘要

本章的开始有些不同于之前的章节。我们的目标是创建一个基于 JSON 的 API，而不是一个典型的生成 HTML 并消费提交的 HTML 表单数据的 Web 应用程序。

我们首先稍微偏离一下，解释了 Werkzeug 的存在和用处，然后使用名为 Flask-RESTful 的 Flask 扩展创建了一个基本的 API。接下来，我们确保我们的 API 可以通过要求身份验证来保护，并解释了身份验证和授权之间微妙但根本的区别。

然后，我们看了如何实现 API 的验证规则，以确保客户端可以创建有效的资源（例如新用户、上传照片等）。我们使用`py.test`框架实现了几个功能和集成级别的单元测试。

我们通过实现最重要的功能——照片上传，完成了本章。我们确保这个功能按预期运行，并实现了照片的逆向时间顺序视图，这对 API 的消费者来说是必要的，以便向用户显示上传的图片。在此过程中，我们讨论了 Werkzeug 中间件的概念，这是一种强大但经常被忽视的方式，可以在 Flask 处理请求之前审查和（可能）修改请求。

在下一章中，我们将探讨使用和创建命令行工具，这将允许我们通过 CLI 接口和管理我们的 Web 应用程序。
