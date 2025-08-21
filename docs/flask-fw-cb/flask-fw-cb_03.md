# 第三章：Flask 中的数据模型

这一章将会覆盖任何应用中最重要的部分：和数据库的交互。本章中将介绍如何用 Flask 连接数据库系统，定义模型，查询数据。

本章将包含下面小节：

*   创建一个 SQLAlchemy DB 实例
*   创建一个基本的商品模型
*   创建一个关系类别模型
*   使用 Alembic 和 Flask-Migrate 实现数据库迁移（migration）
*   用 Redis 建立模型数据索引
*   使用非关系型数据库 MongoDB

## 介绍

Flask 被设计的足够灵活来支持任何数据库。最简单的方式是直接使用 sqlite3，sqlite3 它提供了 DB-API2.0 接口，不提供 ORM。sqlite3 使用 SQL 语句和数据库对话。这种方法不推荐用来构建大型应用，因为最终维护应用会变成一个噩梦。同样，用这种方法实际上是不存在模型的，所有的事情在视图函数中进行，比如在视图函数中编写查询语句去和数据库交互。

本章我们将使用广泛使用的 SQLAlchemy 为 Flask 应用创建一个 ORM 层。同时学习如何编写一个使用 NoSQL 数据库的 Flask 应用。

###### 提示

ORM 的指的是对象关系映射（Object Relation Mapping/Modeling），抽象的表明了我们的应用数据如何存储，如何处理。强大的 ORM 使得设计和查询业务逻辑非常简单和简洁。

## 创建一个 SQLAlchemy DB 实例

SQLAlchemy 是一个 Python SQL 工具集，它提供了一个 ORM，可以灵活高效的处理 SQL，并且通过它能够感受到 Python 的面向对象特性。

#### 准备

Flask-SQLAlchemy 是一个扩展，为 Flask 提供了 SQLAlchemy 接口。
安装 Flask-SQLAlchemy:

```py
$ pip install flask-sqlalchemy 
```

使用 Flask-SQLAlchemy 首先要做的是设置应用配置参数，告诉 SQLAlchemy 数据库的位置：

```py
app.config['SQLALCHENY_DATABASE_URI'] = os.environ('DATABASE_URI') 
```

SQLALCHEMY_DATABASE_URI 是数据库协议的组合，需要认证，需要数据库的名字。用 SQLite 举例，它看起来像这样：

```py
sqlite:tmp/test.db 
```

用 PostgreSQL 举例，看起来像这样：

```py
postgresql://yourusername:yourpassword@localhost/yournewdb. 
```

这个扩展提供了一个叫做 Model 的类，它用来为我们的应用定义模型。了解更多数据库 URLS 参见
`http://docs.sqlalchemy.org/en/rel_0_9/core/engines.html#database-urls`。

###### 提示

除了 SQLite，都需要安装独立的数据库。比如，如果需要使用 PostgreSQL,需要安装 psycopg2.

#### 怎么做

用一个小应用进行演示。下面的小节也一直会使用这个应用。这里，我们将会看到如何创建一个 db 实例，和一些基本的 DB 命令。文件结构看起来像这样：

```py
flask_catalog/
    - run.py
    my_app/
        - __init__.py 
```

首先从 flask_app/run.py 开始，这已经在书里见到很多次了：

```py
from my_app import app
app.run(debug=True) 
```

然后是应用配置文件，`flask_app/my_app/__init__.py`:

```py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:tmp/test.sqlite'
db = SQLAlchemy(app) 
```

###### 译者注

原书为 from flask.ext.sqlalchemy import SQLAlchemy，现已不建议这么使用。
原书为 sqlite:tmp/test.db，改为 sqlite:tmp/test.sqlite。

我们配置 SQLALCHEMY_DATABASE_URI 为一个特定的路径。然后我们创建了一个 SQLAlchemy 对象叫做 db。从名字可以看出，这个对象将会处理所有和 ORM 相关的活动。之前提到过，db 对象有一个名为 Model 的类，它提供了在 Flask 创建模型的基础。任何类都可以继承 Model 去创建模型，模型将作为数据库表。

现在如果在浏览器打开`http://127.0.0.1:5000`，我们看不到任何东西。因为应用里就没有东西。

#### 更多

有时你可能需要一个单独的 SQLAlchemy db 实例可以被多个应用使用或者动态的创建应用。在这些情况下，我们不会讲一个 db 实例绑定在一个单独的应用上。这里我们必须和应用上下文一起工作以达到预期的结果。
这种情况下，使用 SQLAlchemy 注册方式将有所不同：

```py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    db.init_app(app)
    return app 
```

###### 提示

上面的方法也可以用来初始化其他 Flask 扩展，而且这在实际应用中是很通常的做法。

现在，所有使用全局 db 实例的操作都需要一个 Flask 上下文了：

```py
Flask application context
>>> from my_app import create_app
>>> app = create_app()
>>> app.test_request_context().push()
>>> # Do whatever needs to be done
>>> app.test_request_context().pop()
Or we can use context manager
with app():
    # We have flask application context now till we are inside the with block 
```

#### 其他

*   下面章节将扩展当前的应用为一个完整的应用，以帮助我们更好的理解 ORM

## 创建一个基本的商品模型

这一小节，我们将创建一个应用帮助我们在网站目录中显示商品。它也可以用来向目录中添加商品也可以根据需要删除他们。从前面章节可以看到，也可以使用非持久化的存储，但是现在我们将数据存储在数据库里做持久化存储。

#### 怎么做

文件夹目录看起来像这样：

```py
flask_catalog/
    - run.py
    my_app/
        – __init__.py
        catalog/
            - __init__.py
            - views.py
            - models.py 
```

首先，修改应用配置文件，`flask_catalog/my_app/__init__.py`：

```py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:tmp/test.sqlite'
db = SQLAlchemy(app)

from my_app.catalog.views import catalog
app.register_blueprint(catalog)

db.create_all() 
```

最后一句 db.create_all()，告诉应用在特定的数据库里创建所有的表。所以，当应用运行的时候，如果表不存在的话，所有的表将会创建。现在是时候去在`flask_catalog/my_app/catalog/models.py`中创建模型了：

```py
from my_app import db

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    price = db.Column(db.Float)

    def __init__(self, name, price):
        self.name = name
        self.price = price

    def __repr__(self):
        return '<Product %d>' % self.id 
```

这个文件中，我们创建了一个叫做 Product 的模型，它有三个字段，id，name，price。id 是一个自增长的字段，它会存储记录的 ID 并且做为主键。name 是一个字符串类型的字段，price 是浮点类型的。

现在为视图添加一个新的文件，`flask_catalog/my_app/catalog/views.py`。在这个文件里我们有许多视图方法来处理商品模型和应用：

```py
from flask import request, jsonify, Blueprint
from my_app import app, db
from my_app.catalog.models import Product

catalog = Blueprint('catalog', __name__)

@catalog.route('/')
@catalog.route('/home')
def home():
    return "Welocme to the Catolog Home." 
```

这个方法处理了主页看起来像什么样子。大多数情况下会在应用里使用模板进行渲染。我们稍后会继续讨论这个问题，现在看下下面的代码：

```py
@catalog.route('/product/<id>')
def product(id):
    product = Product.query.get_or_404(id)
    return 'Product - %s, $%s' % (product.name, product.price) 
```

这个方法控制了当用户用商品特定 ID 进行搜索时的输出。我们用 ID 过滤商品，当商品被找到的时候返回它的信息；如果没有找到，产生一个 404 错误。看下面的代码：

```py
@catalog.route('/products')
def products():
    products = Product.query.all()
    res = {}
    for product in products:
        res[product.id] = {
            'name': product.name,
            'price': str(product.price)
    }
    return jsonify(res) 
```

这个方法以 JSON 格式返回了所有商品信息。看下面代码：

```py
@catalog.route('/product-create', methods=['POST',])
def create_product():
    name = request.form.get('name')
    price = request.form.get('price')
    product = Product(name, price)
    db.session.add(product)
    db.session.commit()
    return "Product created" 
```

这个方法控制了数据库中商品的创建。我们首先从 request 请求中获取信息，然后用这些信息创建一个 Product 实例。然后将这个 Product 实例添加到数据库会话中（session），然后提交保存这条记录到数据库。

#### 原理

首先，数据库是空的没有任何商品。这可以通过在浏览器打开`http://127.0.0.1:5000/products`进行确认。页面上仅仅会显示一个{}。
现在，要去创建一个商品。为此我们需要发送一个 POST 请求，POST 请求可以很容易的通过使用 Python request 库实现：

```py
>>> import requests
>>> requests.post('http://127.0.0.1:5000/product-create', data={'name': 'iPhone 5S', 'price': '549.0'}) 
```

想要确认商品是否在数据库里了，可以在浏览器里再一次输入`http://127.0.0.1:5000/products`。这次，它会以 JSON 的形式输出商品信息。

#### 其他

*   在下一小节，创建一个关系类别模型，中将会演示表之间的关系

## 创建一个关系类别模型

前一小节，我们创建了一个简单的商品模型。但是，在实际情况下，应用会更加复杂，各个表之间有各种各样的关系（relationships）。这些关系可以是一对一的，一对多的，多对一的，或者是多对多的。这一小节，我们将用例子去理解他们中的一些。

#### 怎么做

假设我们每个商品类别可以有多个商品，但是每个商品至少有一个类别。让我们修改之前的一些代码，同时对模型和视图做出修改。在模型中我们增加了一个 Category 模型，在视图中，我们增加了新的方法去处理类别相关的调用。

首先修改 models.py，增加 Category 模型，并且修改 Product 模型：

```py
from my_app import db

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    price = db.Column(db.Float)
    category_id = db.Column(db.Integer, db.ForeignKey('category.id'))
    category = db.relationship('Category', backref=db.backref('products', lazy='dynamic'))

    def __init__(self, name, price, category):
        self.name = name
        self.price = price
        self.category = category

    def __repr__(self):
        return '<Product %d>' % self.id 
```

在前面的 Product 模型中，注意新增加的两个字段 category_id 和 category。category_id 是 Category 模型的外键，category 代表关系表。从他们的定义可以看出一个是关系，另一个使用这个关系在数据库中存储外键值。这是一个从商品到类别的简单多对一的关系。同时，注意 category 字段中的 backref 参数；这个参数允许我们可以在视图里编写 category.prodycts 从 Category 模型获取商品。从另一端看这一个一对多的关系。考虑下面代码：

```py
calss Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '<Category %d>' % self.id 
```

前面的代码是 Category 模型，它仅仅只有一个 name 字段。
现在修改 views.py 来适应模型的改变：

```py
from my_app.catalog.models import Product, Category

@catalog.route('/products')
def products():
    products = Product.query.all()
    res = {}
    for product in products:
        res[product.id] = {
            'name': product.name,
            'price': product.price,
            'category': product.category.name
        }
    return jsonify(res) 
```

这里，我们只做了一个修改，在返回商品 JSON 信息的时候添加了 category 信息。看下面的代码：

```py
@catalog.route('/product-create', methods=['POST',])
def create_product():
    name = request.form.get('name')
    price = request.form.get('price')
    categ_name = request.form.get('category')
    category = Category.query.filter_by(name=categ_name).first()
    if not category:
        category = Category(categ_name)
    product = Product(name, price, category)
    db.session.add(product)
    db.session.commit()
    return "Product created." 
```

看一下是如何在创建商品之前查找类别的。首先，使用请求中的类别名在已经存在的类别里进行搜索。如果找到了，就使用它进行商品的创建。否则，就创建一个新的类别。看下面的代码：

```py
@catalog.route('/category-create', methods=['POST',])
def create_category():
    name = request.form.get('name')
    category = Category(name)
    db.session.add(category)
    db.session.commit()
    return 'Category created.' 
```

前面的是一个非常简单的使用请求里的 name 来创建类别的方法。看下面的代码：

```py
@catalog.route('/categories')
def categories():
    categories = Category.query.all()
    res = {}
    for category in categoires:
        res[category.id] = {
            'name': category.name
        }
        for product in category.products:

            res[category.id]['products'] = {
                'id': product.id,
                'name': product.name,
                'price': product.price
            }
    return jsonify(res) 
```

上面代码有点复杂。首先从数据库里获取到所有的类别信息，然后遍历每个类别，获取所有的商品信息，然后用 JSON 的形式返回。

###### 译者注

上面代码是存在问题的，它的原意是想列出每个类别下所有的商品，但是结果只能列出一个。可以改为：

```py
@catalog.route('/categories')
def categories():
    categories = Category.query.all()
    res = {}
    for category in categories:
        res[category.id] = {
            'name': category.name
        }
        res[category.id]['products'] = []
        for product in category.products:
            p = {
                'id': product.id,
                'name': product.name,
                'price': product.price
            }
            res[category.id]['products'].append(p)
    return jsonify(res) 
```

## 使用 Alembic 和 Flask-Migrate 进行数据库迁移（migration）

现在我们想要在 Product 模型中添加一个新的叫做 company 的字段。一种方式是去通过使用 db.drop_all()和 db.create_all()删除数据库然后新建一个。但是，这种方法不能用于生产中。我们希望迁移我们的数据库到最新的模型，并保持所有数据完整。
为此，我们使用 Alembic，这是一个基于 Python 的工具来管理数据库迁移和使用 SQLAlchemy 作为底层引擎。Alembic 在很大程度上提供了自动迁移，但有一些限制(当然，我们不能期望任何工具是完美的)。我们使用一个叫做 Flask-Migrate 的扩展来简化迁移的过程。

#### 准备

先安装 Flask-Migrate

```py
$ pip install Flask-Migrate 
```

这个将会安装 Flask-Script 和 Alembic 还有其他一些依赖。Flask-Script 使得 Flask-Migrate 提供一些简单使用的命令行参数，这些参数对用户而言是一个高级别的抽象，并且隐藏了所有复杂的特性（如果需要的话事实上也不是很难的去自定义）。

#### 怎么做

为了能够迁移，需要稍微去修改一下 app 定义，`my_app/__init__.py`看起来像这样：

```py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:tmp/test.sqlite'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

manager = Manager(app)
manager.add_command('db', MigrateCommand)

# import my_app.catalog.views
from my_app.catalog.views import catalog
app.register_blueprint(catalog)

db.create_all() 
```

同时，需要对 run.py 做些小改动：

```py
from my_app import manager
manager.run() 
```

run.py 的改动是因为我们需要使用 Flask script manager 的方式去启动应用。script manager 同样提供了额外的命令行参数。在这个例子中我们将使用 db 做为命令行参数。
如果我们把 run.py 当做脚本运行时，给它传递–help 参数，终端这时候将会展示所有的选项，看起来像下面这样：

![](img/151c44f70cf908c87a3696e791f75118.png)

现在，运行这个应用，可以使用：

```py
$ python run.py runserver 
```

初始化迁移，需要运行 init 命令：

```py
$ python run.py db init 
```

之后当我们对模型做了更改，需要运行 migrate 命令：

```py
$ python run.py db migrate 
```

为了将更改反映到数据库上，需要运行 upgrade 命令：

```py
$ python run.py db upgrade 
```

#### 原理

现在，修改商品模型，添加一个新的字段 company：

```py
class Product(db.Model):
    # ...
    # Same product model as last recipe
    # ...
    company = db.Column(db.String(100)) 
```

migrate 的结果看起来像这样：

```py
$ python run.py db migrate
INFO [alembic.migration] Context impl SQLiteImpl.
INFO [alembic.migration] Will assume non-transactional DDL.
INFO [alembic.autogenerate.compare] Detected added column 
    'product.company' Generating <path/to/application>/
    flask_catalog/migrations/versions/2c08f71f9253_.py ... done 
```

前面的代码，我们看到 Alembic 将新的模型和数据库进行比较，然后检测到 product 新增了 company 一列（由 Product 模型创建）。

相似的，upgrade 的的输出将看起来像这样：

```py
$ python run.py db upgrade
INFO [alembic.migration] Context impl SQLiteImpl.
INFO [alembic.migration] Will assume non-transactional DDL.
INFO [alembic.migration] Running upgrade None -> 2c08f71f9253, empty message 
```

这里，Alembic 用之前检测到的迁移来升级数据库。可以看到输出中有一个 16 进制数。这代表了执行迁移的版本。
Alembic 内部使用它来追踪数据库表的更改。

## 用 Redis 建立模型数据索引

也许有些特性要去实现，但是不想对他们做持久化存储。所以，我们可以将他们存储在内存里保持一段时间，然后隐藏他们，举个例子，在网站上向访问者展示访问过的商品列表。

#### 准备

我们将使用 Redis 来做到这些，使用下面命令安装 Redis:

```py
$ pip install redis 
```

确保你的 Redis 服务器在运行，以便链接。安装和运行 Redis 服务器，参见：`http://redis.io/topics/quickstart`

然后我们需要和 Redis 连接。在`my_app/__init__.py`中添加下面代码可以做到这些：

```py
from redis import Redis
redis = Redis() 
```

我们可以在应用中需要用到 Redis 的地方构造 redis，比如定义 app 的地方，或者在视图文件里。最好是在应用
文件中做，因为这样连接将在整个应用中保持打开，仅仅通过导入 redis 就能够在任何需要的地方使用，。

#### 怎么做

我们将在 Redis 中设置一个集合来存储最近浏览过的商品。当我们浏览商品的时候会填充它。该记录将在 10 分钟后过期。对 views.py 做如下修改：

```py
from my_app import redis

@catalog.route('/product/<id>')
def product(id):
    product = Product.query.get_or_404(id)
    product_key = 'product-%s' % product.id
    redis.set(product_key, product.name)
    redis.expire(product_key, 600)
    return 'Product - %s, $%s' % (product.name, product.price) 
```

###### 提示

好的习惯是从配置文件获取 expire 时间，600。可以在`my_app/__init__.py`中设置该值，然后从这里获取。

在前面的方法中，注意 redis 对象的 set()和 expire()方法。首先，我们在 Redis 中使用 product_key 设置商品的 ID。然后，我们设置过期时间为 600 秒。

现在我们将查找缓存中还存活的键。然后获取和这些键相匹配的商品，之后返回他们：

```py
@catalog.route('/recent-products')
def recent_products():
    keys_alive = redis.keys('product-*')
    products = [redis.get(k).decode("utf-8") for k in keys_alive]
    return jsonify({'products': products}) 
```

###### 译者注

我运行代码的时候，因为 redis.get(k)获取到的字符串是 unicode 类型的，如果没有 decode(“utf-8”)，jsonify 解析会出错，所以这里加上解码。

#### 原理

当用户访问一个商品的时候就会有一条记录被存储，这条记录将保持 600 秒。下面的 10 分钟这个商品将被列在最近商品中，除非再一次被访问，然后再一次设置为 10 分钟。

## 使用非关系型数据库 MongoDB

有时，我们正在构建的应用程序中使用的数据可能根本不是结构化的，也可以是半结构化的，也可以是其模式随时间变化的数据。在这种情况下，我们将避免使用 RDBMS，因为它增加了痛苦，并且难以理解和维护。这时应该使用 NoSQL 数据库。
同时，在当前流行的开发环境下，由于开发速度快，不一定能够第一次设计出完美的结构（scheam）。NoSQL 提供了修改结构的灵活性，而不需要太多麻烦。
在生产环境中，数据库通常在短时间内增长到一个巨大的规模。这极大地影响了整个系统的性能。垂直和水平缩放技术（Vertical-and horizontal-scaling）也是可用的，但是非常昂贵。这些情况下，可以考虑使用 NoSQL 数据库，因为它就是为了这个目的而被设计的。NoSQL 数据库能够在大型集群上运行，并处理大量生成的高速数据，这使得它们在使用传统 RDBMS 处理伸缩性（scaling）问题时是一个不错的选择。
这里将使用 MongoDB 来演示 Flask 如何集成 NoSQL。

#### 准备

Flask 有许多 MongoDB 的扩展。我们将使用 Flask-MongoEngine，因为它提供了一个非常好的抽象，让我们很容易理解。通过下面命令安装它：

```py
$ pip install flask-mongoengine 
```

记住去开启 MongoDB 服务器，以便连接。更多安装和运行 MongoDB 的信息，可以参见`http://docs.mongodb.org/manual/installation/`

#### 怎么做

下面使用 MongoDB 重写我们的应用。首先修改`my_app/__init__.py`:

```py
from flask import Flask
from flask_mongoengine import MongoEngine
from redis import Redis

app = Flask(__name__)
app.config['MONGODB_SETTINGS']  = {'DB': 'my_catalog'}
app.debug = True
db = MongoEngine(app)

redis = Redis()

from my_app.catalog.views import catalog
app.register_blueprint(catalog) 
```

###### 提示

现在我们使用 MONGODB_SETTINGS 的配置而不是通常 SQLAlchemy-centric 的配置。这里，我们只需指定数据库的名字就可以使用。首先，我们需要在 MongoDB 中创建数据库，使用下面命令：

```py
>>> mongo
MongoDB shell version: 2.6.4
> use my_catalog
switched to db my_catalog 
```

接下来，我们将使用 MongoDB 字段创建一个 Product 模型。修改`flask_catalog/my_app/catalog/models.py`:

```py
import datetime
from my_app import db

class Product(db.Document):
    created_at = db.DateTimeField(default=datetime.datetime.now, required=True)
    key = db.StringField(max_length=255, required=True)
    name = db.StringField(max_length=255, required=True)
    price = db.DecimalField()

    def __repr__(self):
        return '<Product %r>' % self.id 
```

#### 其他

注意创建模型的 MongoDB 字段和前面小节使用的 SQLAlchemy 是相似的。这里取消了 ID 字段，我们设置了 created_at，这个字段将会存储记录创建的时间戳。

接下来是视图文件，`flask_catalog/my_app/catalog/views.py`:

```py
from decimal import Decimal
from flask import request, Blueprint, jsonify
from my_app.catalog.models import Product

catalog = Blueprint('catalog', __name__)

@catalog.route('/')
@catalog.route('/home')
def home():
    return "Welcome to the Catalog Home."

@catalog.route('/product/<key>')
def product(key):
    product = Product.objects.get_or_404(key=key)
    return 'Product - %s, $%s' % (product.name, product.price)

@catalog.route('/products')
def products():
    products = Product.objects.all()
    res = {}
    for product in products:
        res[product.key] = {
            'name': product.name,
            'price': str(product.price)
        }
    return jsonify(res)

@catalog.route('/product-create', methods=['POST',])
def create_product():
    name = request.form.get('name')
    key = request.form.get('key')
    price = request.form.get('price')
    product = Product(
        name=name,
        key=key,
        price=Decimal(price)
    )
    product.save()
    return 'Product created.' 
```

你会发现这非常类似于基于 SQLAlchemy 模型创建的视图。仅仅存在一些差异，而且是很容易理解的。