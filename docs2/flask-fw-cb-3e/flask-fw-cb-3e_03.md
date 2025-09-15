

# 第三章：Flask 中的数据建模

本章涵盖了任何应用程序最重要的方面之一，即与数据库系统的交互。在本章中，您将了解 Flask 如何连接到数据库系统，定义模型，以及查询数据库以检索和传输数据。Flask 被设计得足够灵活，以支持任何数据库。最简单的方法是使用直接的`SQLite3`包，这是一个*DB-API 2.0*接口，并不提供实际的**对象关系映射**（**ORM**）。在这里，我们需要编写 SQL 查询来与数据库通信。这种方法不推荐用于大型项目，因为它最终可能成为维护应用程序的噩梦。此外，使用这种方法，模型几乎不存在，所有操作都在视图函数中完成，将数据库查询写入视图函数不是一种好的做法。

在本章中，我们将讨论如何使用 SQLAlchemy 为我们的 Flask 应用程序创建 ORM 层，这对于任何大小的应用程序都是推荐和广泛使用的。此外，我们还将简要了解如何使用 NoSQL 数据库系统编写 Flask 应用程序。

信息

ORM 表示我们的应用程序数据模型在概念层面上如何存储和处理数据。一个强大的 ORM 使得设计和查询业务逻辑变得简单且流畅。

在本章中，我们将介绍以下菜谱：

+   创建一个 SQLAlchemy 数据库实例

+   创建基本的产品模型

+   创建关系型类别模型

+   使用 Alembic 和 Flask-Migrate 迁移数据库

+   使用 Redis 索引模型数据

+   选择 MongoDB 的 NoSQL 方式

# 创建一个 SQLAlchemy 数据库实例

SQLAlchemy 是一个 Python SQL 工具包，它提供了 ORM，它结合了 SQL 的灵活性和强大功能以及 Python 面向对象的特性。在本例中，我们将了解如何创建一个 SQLAlchemy 数据库实例，该实例可用于执行未来菜谱中将要涵盖的任何数据库操作。

## 准备工作

Flask-SQLAlchemy 是提供 Flask 的 SQLAlchemy 接口的扩展。此扩展可以通过以下方式简单地使用`pip`安装：

```py
$ pip install flask-sqlalchemy
```

在使用 Flask-SQLAlchemy 时，首先要记住的是应用程序配置参数，它告诉 SQLAlchemy 要使用数据库的位置：

```py
app.config['SQLALCHEMY_DATABASE_URI'] =
  os.environ('DATABASE_URI')
```

`SQLALCHEMY_DATABASE_URI`是数据库协议、所需的任何身份验证以及数据库名称的组合。在 SQLite 的情况下，它可能看起来如下所示：

```py
sqlite:////tmp/test.db
```

在 PostgreSQL 的情况下，它看起来如下所示：

```py
postgresql://yourusername:yourpassword@localhost/yournewdb
```

此扩展提供了一个名为`Model`的类，它帮助我们为应用程序定义模型。有关数据库 URL 的更多信息，请参阅[`docs.sqlalchemy.org/en/14/core/engines.html#database-urls`](https://docs.sqlalchemy.org/en/14/core/engines.html#database-urls)。

小贴士

SQLite 数据库 URI 是操作系统特定的，这意味着 URI 对于 Unix/macOS/Linux 和 Windows 会有所不同。请参阅[`docs.sqlalchemy.org/en/14/core/engines.html#sqlite`](https://docs.sqlalchemy.org/en/14/core/engines.html#sqlite)以获取更多详细信息。

信息

对于 SQLite 以外的所有数据库系统，都需要单独的库。例如，要使用 PostgreSQL，你需要`psycopg2`。

## 如何做到这一点...

让我们在本食谱中创建一个小型应用程序，以了解 Flask 的基本数据库连接。我们将在接下来的几个食谱中构建这个应用程序。在这里，我们只需看看如何创建一个`db`实例并验证其存在。文件的结构看起来如下：

```py
    flask_catalog/
        run.py
        my_app/
            __init__.py
```

首先，我们从`flask_app/run.py`开始。这是我们在这本书的多个食谱中之前读到的常规`run`文件：

```py
from my_app import app
app.run(debug=True)
```

然后，我们配置我们的应用程序配置文件，`flask_app/my_app/__init__.py`:

```py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] =
  'sqlite:////tmp/test.db'
db = SQLAlchemy(app)
with app.app_context():
    db.create_all()
```

在这里，我们首先配置我们的应用程序，将`SQLALCHEMY_DATABASE_URI`指向一个特定位置。然后，我们创建一个名为`db`的`SQLAlchemy`对象。正如其名所示，这个对象将处理我们所有的 ORM 相关活动。如前所述，这个对象有一个名为`Model`的类，它为 Flask 中的模型创建提供了基础。任何类都可以通过继承`Model`类来创建模型，这些模型将作为数据库表。

现在，如果我们在一个浏览器中打开`http://127.0.0.1:5000` URL，我们将什么也看不到。这是因为我们刚刚为这个应用程序配置了数据库连接，浏览器上没有东西可以显示。然而，你总是可以前往`app.config`中指定的数据库位置，以查看新创建的`test.db`文件。

## 更多...

有时，你可能希望单个 SQLAlchemy `db`实例在多个应用程序中使用，或者动态创建应用程序。在这种情况下，将`db`实例绑定到单个应用程序不是首选的。在这里，你必须与应用程序上下文一起工作以实现预期的结果。

在这种情况下，注册应用程序与 SQLAlchemy 的方式如下：

```py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()
def create_app():
    app = Flask(__name__)
    db.init_app(app)
    return app
```

小贴士

在初始化应用程序时，可以使用任何 Flask 扩展采取这种方法，这在处理实际应用程序时非常常见。

现在，所有之前通过`db`实例在全局范围内可能进行的操作都需要始终在 Flask 应用程序上下文中进行。

Flask 应用程序上下文如下：

```py
>>> from my_app import create_app
>>> app = create_app()
>>> app.test_request_context().push()
>>> # Do whatever needs to be done
>>> app.test_request_context().pop()
```

或者，你可以使用上下文管理器，如下所示：

```py
with app():
# We have flask application context now till we are inside the with block
```

## 参见

接下来的几个食谱将扩展当前应用程序以使其成为一个完整的应用程序，这将帮助我们更好地理解 ORM 层。

# 创建基本的产品模型

在这个菜谱中，我们将创建一个应用程序，帮助我们存储要在网站目录部分显示的产品。应该能够添加产品到目录，并在需要时删除它们。正如你在上一章中看到的，这也可以使用非持久存储来完成。然而，在这里，我们将数据存储在数据库中以实现持久存储。

## 如何做到这一点…

新的目录布局如下所示：

```py
    flask_catalog/
        run.py
        my_app/
            __init__.py
            catalog/
                __init__.py
                views.py
                models.py
```

首先，从修改应用程序配置文件`flask_catalog/my_app/__init__.py`开始：

```py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] =
    'sqlite:////tmp/test.db'
db = SQLAlchemy(app)
from my_app.catalog.views import catalog
app.register_blueprint(catalog)
with app.app_context():
    db.create_all()
```

文件中的最后一个语句是`db.create_all()`，它告诉应用程序创建指定数据库中的所有表。因此，一旦应用程序运行，如果表中还没有数据，所有表都将被创建。由于你现在不在应用程序请求中，请使用`with app.app_context():`手动创建上下文。

现在是时候创建位于`flask_catalog/my_app/catalog/models.py`中的模型了：

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

在这个文件中，我们创建了一个名为`Product`的模型，它有三个字段，即`id`、`name`和`price`。`id`是数据库中自动生成的字段，它将存储记录的 ID，是主键。`name`是`string`类型的字段，而`price`是`float`类型的字段。

现在，为视图添加一个新文件，该文件位于`lask_catalog/my_app/catalog/views.py`。在这个文件中，我们有多个视图方法，它们控制我们如何处理产品模型和整个 Web 应用程序。

```py
from flask import request, jsonify, Blueprint
from my_app import db
from my_app.catalog.models import Product
catalog = Blueprint('catalog', __name__)
@catalog.route('/')
@catalog.route('/home')
def home():
    return "Welcome to the Catalog Home."
```

上述方法处理主页或应用程序着陆页的外观或对用户的响应。你可能会想在你的应用程序中使用模板来渲染它。我们将在下一章中介绍。

看看下面的代码：

```py
@catalog.route('/product/<id>')
def product(id):
    product = Product.query.get_or_404(id)
    return 'Product - %s, $%s' % (product.name,
      product.price)
```

上述方法控制当用户使用其 ID 查找特定产品时显示的输出。我们使用 ID 过滤产品，如果找到产品，则返回其信息，否则以`404`错误终止。

考虑以下代码：

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

上述方法以 JSON 格式返回数据库中所有产品的列表。如果没有找到产品，它将简单地返回一个空的 JSON：`{}`。

考虑以下代码：

```py
@catalog.route('/product-create', methods=['POST',])
def create_product():
    name = request.form.get('name')
    price = request.form.get('price')
    product = Product(name, price)
    db.session.add(product)
    db.session.commit()
    return 'Product created.'
```

上述方法控制数据库中产品的创建。我们首先从`request`对象中获取信息，然后根据这些信息创建一个`Product`实例。

然后，我们将这个`Product`实例添加到数据库会话中，最后使用`commit`将记录保存到数据库中。

## 它是如何工作的...

在开始时，数据库是空的，没有产品。这可以通过在浏览器中打开`http://127.0.0.1:5000/products`来确认。这将导致一个空的 JSON 响应，或`{}`。

现在，首先，我们想要创建一个产品。为此，我们需要发送一个`POST`请求，这可以通过使用`requests`库从 Python 提示符轻松发送：

```py
>>> import requests
>>> requests.post('http://127.0.0.1:5000/product-create', data={'name': 'iPhone 5S', 'price': '549.0'})
```

为了确认产品是否现在在数据库中，我们可以在浏览器中再次打开 `http://127.0.0.1:5000/products`。这次，它将显示产品详情的 JSON 输出，看起来像这样：

```py
{
  "1": {
    "name": "iPhone 5S",
    "price": "549."
  }
}
```

# 创建一个关系型分类模型

在我们之前的菜谱中，我们创建了一个简单的产品模型，它有几个字段。然而，在实践中，应用程序要复杂得多，并且它们之间的表之间存在各种关系。这些关系可以是一对一、一对多、多对一或多对多。在这个菜谱中，我们将通过一个例子来尝试理解这些关系之一。

## 如何做到这一点...

假设我们想要有产品分类，其中每个分类可以拥有多个产品，但每个产品只能有一个分类。让我们通过修改上一道菜谱中的应用程序中的某些文件来实现这一点。我们将对模型和视图进行修改。在模型中，我们将添加一个 `Category` 模型，在视图中，我们将添加处理分类相关调用的新方法，并修改现有方法以适应新添加的功能。

首先，修改 `models.py` 文件以添加 `Category` 模型并对 `Product` 模型进行一些修改：

```py
from my_app import db
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    price = db.Column(db.Float)
    category_id = db.Column(db.Integer,
      db.ForeignKey('category.id'))
    category = db.relationship(
        'Category', backref=db.backref('products',
          lazy='dynamic')
    )
    def __init__(self, name, price, category):
        self.name = name
        self.price = price
        self.category = category
    def __repr__(self):
        return '<Product %d>' % self.id
```

在先前的 `Product` 模型中，检查新添加的 `category_id` 和 `category` 字段。`category_id` 是到 `Category` 模型的外键，而 `category` 代表关系表。从定义本身可以看出，其中一个是关系，另一个使用这个关系在数据库中存储外键值。这是一个从 `product` 到 `category` 的简单多对一关系。注意 `category` 字段中的 `backref` 参数；这个参数允许我们通过在视图中写入 `category.products` 这样简单的语句来从 `Category` 模型访问产品。这就像从另一端的一个一对多关系。

重要信息

仅在模型中添加字段并不会立即反映到数据库中。你可能需要删除整个数据库然后重新运行应用程序，或者运行迁移，这将在下一道菜谱中介绍，即 *使用 Alembic 和 Flask-Migrate 迁移数据库*。

对于 SQLite，你可以简单地删除在初始化应用程序时创建的数据库文件。

创建一个只有一个名为 `name` 字段的 `Category` 模型：

```py
class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return '<Category %d>' % self.id
```

现在，修改 `views.py` 以适应模型中的更改。首先在 `products()` 方法中进行第一次更改：

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

在这里，我们只做了一项修改，即在产品的 JSON 数据中发送 `category` 名称，该数据在向先前的端点发出请求时生成并作为响应返回。

修改 `create_product()` 方法，在创建产品之前先查找分类：

```py
@catalog.route('/product-create', methods=['POST',])
def create_product():
    name = request.form.get('name')
    price = request.form.get('price')
    categ_name = request.form.get('category')
    category =
      Category.query.filter_by(name=categ_name).first()
    if not category:
        category = Category(categ_name)
    product = Product(name, price, category)
    db.session.add(product)
    db.session.commit()
    return 'Product created.'
```

在这里，我们首先将在请求中的类别名称中搜索现有的类别。如果找到现有类别，我们将使用它在产品创建中；如果没有找到，我们将创建一个新的类别。

创建一个新的方法，`create_category()`，用于处理类别的创建：

```py
@catalog.route('/category-create', methods=['POST',])
def create_category():
    name = request.form.get('name')
    category = Category(name)
    db.session.add(category)
    db.session.commit()
    return 'Category created.'
```

上述代码是一个相对简单的方法，用于使用请求中提供的名称创建一个类别。

创建一个新的方法，`categories()`，用于处理所有类别及其对应产品的列表：

```py
@catalog.route('/categories')
def categories():
    categories = Category.query.all()
    res = {}
    for category in categories:
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

上述方法做了一些稍微复杂的事情。在这里，我们从数据库中检索了所有类别，然后对每个类别，我们检索了所有产品，然后以 JSON 格式返回所有数据。

## 工作原理...

本食谱的工作原理与上一个食谱，*创建基本* *产品模型*，非常相似。

要创建一个具有类别的产品，向 `/product-create` 端点发送一个 `POST` 请求：

```py
>>> import requests
>>> requests.post('http://127.0.0.1:5000/product-create', data={'name': 'iPhone 5S', 'price': '549.0', 'category': 'Phones'})
```

要查看从数据库中检索到的数据现在看起来如何，请在您的浏览器中打开 `http://127.0.0.1:5000/categories`：

```py
{
  "1": {
    "name": "Phones",
    "products": {
      "id": 1,
      "name": "iPhone 5S",
      "price": 549.0
    }
  }
}
```

## 参见

参考创建基本产品模型的食谱，以了解本食谱的上下文以及本食谱如何为浏览器工作，因为其工作原理与上一个食谱非常相似。

# 使用 Alembic 和 Flask-Migrate 迁移数据库

更新数据库模式是所有应用程序的重要用例，因为它涉及添加或删除表和/或列，或更改列类型。一种方法是通过 `db.drop_all()` 和 `db.create_all()` 删除数据库，然后创建一个新的数据库。然而，这种方法不能用于生产环境或预发布环境。我们希望将数据库迁移到与最新更新的模型匹配，同时保留所有数据完整。

对于此，我们拥有 **Alembic**，这是一个基于 Python 的数据库迁移管理工具，它使用 SQLAlchemy 作为底层引擎。Alembic 在很大程度上提供了自动迁移，但也存在一些限制（当然，我们不可能期望任何工具都能无缝运行）。作为锦上添花的部分，我们还有一个名为 **Flask-Migrate** 的 Flask 扩展，它进一步简化了迁移过程。在本食谱中，我们将介绍使用 Alembic 和 Flask-Migrate 进行数据库迁移的基本技术。

## 准备工作

首先，运行以下命令来安装 `Flask-Migrate`：

```py
$ pip install Flask-Migrate
```

这也将安装 Alembic 以及其他许多依赖项。

## 如何操作...

要启用迁移，我们需要稍微修改我们的应用程序定义。让我们了解如果我们为我们的 `catalog` 应用程序进行相同的修改，这样的配置将如何出现：

以下代码行显示了 `my_app/__init__.py` 的外观：

```py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] =
  'sqlite:////tmp/test.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)
from my_app.catalog.views import catalog
app.register_blueprint(catalog)
with app.app_context():
    db.create_all()
```

如果我们在以脚本方式运行 `flask` 命令时传递 `--help`，终端将显示所有可用选项，如下面的截图所示：

![图 3.1 – 数据库迁移选项](img/B19111_03_1.jpg)

图 3.1 – 数据库迁移选项

要初始化迁移，运行 `init` 命令：

```py
$ flask db init
```

重要信息

要使迁移命令生效，Flask 应用程序应该是可定位的；否则，您将收到以下错误：

`错误：无法定位 Flask 应用程序。使用'flask --app'选项，'FLASK_APP'环境变量，或在当前目录中的'wsgi.py'或'app.py'文件。`

在我们的情况下，只需将 Flask 应用程序导出到环境变量中：

`export FLASK_APP="my_app.__init__.py"`

或者，简单地使用以下命令：

`export FLASK_APP=my_app`

一旦对模型进行了更改，请调用`migrate`命令：

```py
$ flask db migrate
```

要使更改反映在数据库中，请调用`upgrade`命令：

```py
$ flask db upgrade
```

## 它是如何工作的...

现在，假设我们修改了`product`表的模型，添加了一个名为

`company`，如下所示：

```py
class Product(db.Model):
    # Same Product model as last recipe
    # ...
    company = db.Column(db.String(100))
```

`migrate`的结果将类似于以下片段：

```py
$ flask db migrate
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
INFO  [alembic.autogenerate.compare] Detected added column 'product.company'
Generating
<path/to/application>/flask_catalog/migrations/versions/2c08f71f9253_.py
... done
```

在前面的代码中，我们可以看到 Alembic 将新模型与数据库表进行比较，并检测到`product`表（由`Product`模型创建）中新增的`company`列。

类似地，`upgrade`的输出将类似于以下片段：

```py
$ flask db upgrade
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade None -> 2c08f71f9253, empty message
```

在这里，Alembic 对之前检测到的迁移进行数据库升级。我们可以看到前一个输出中的十六进制代码。这代表了迁移执行的修订版本。这是 Alembic 内部用于跟踪数据库表更改的。

## 参见

请参考*创建基本产品模型*配方，了解此配方中关于`product`的目录模型的上下文。

# 使用 Redis 索引模型数据

可能有一些我们想要实现但不想为它们提供持久存储的功能。在这种情况下，将这些功能暂时存储在类似缓存存储中是一个很好的方法 - 例如，当我们想在网站上向访客显示最近查看的产品列表时。在这个配方中，我们将了解如何使用 Redis 作为有效的缓存来存储非持久数据，这些数据可以以高速访问。

## 准备工作

我们将使用 Redis 来完成此操作，可以使用以下命令安装：

```py
$ pip install redis
```

确保您运行 Redis 服务器以建立连接。有关安装和运行 Redis 服务器的信息，请参阅[`redis.io/topics/quickstart`](http://redis.io/topics/quickstart)。

然后，我们需要保持与 Redis 的连接打开。这可以通过在`my_app/__init__.py`中添加以下代码行来实现：

```py
from redis import Redis
redis = Redis()
```

我们可以在应用程序文件中这样做，在那里我们将定义应用程序，或者在视图文件中，我们将使用它。建议您在应用程序文件中这样做，因为这样，连接将贯穿整个应用程序，并且只需导入所需的`redis`对象即可使用。

## 如何操作...

我们将在 Redis 中维护一个`set`，它将存储最近访问过的产品。每当访问产品时，它将被填充。条目将在 10 分钟后过期。此更改在`views.py`中：

```py
@catalog.route('/product/<id>')
def product(id):
    product = Product.query.get_or_404(id)
    product_key = 'product-%s' % product.id
    redis.set(product_key, product.name)
    redis.expire(product_key, 600)
    return 'Product - %s, $%s' % (product.name,
      product.price)
```

在前面的方法中，注意`redis`对象上的`set()`和`expire()`方法。首先，使用 Redis 存储中的`product_key`值设置产品 ID。然后，将键的`expire`时间设置为`600`秒。

小贴士

最好从配置值中获取`expire`时间——即`600`。这可以在`my_app/__init__.py`中的应用程序对象上设置，然后从那里获取。

现在，我们将查找缓存中仍然存在的键，然后获取与这些键对应的产品并返回它们：

```py
@catalog.route('/recent-products')
def recent_products():
    keys_alive = redis.keys('product-*')
    products = [redis.get(k).decode('utf-8') for k in
      keys_alive]
    return jsonify({'products': products})
```

## 它是如何工作的...

每当用户访问产品时，都会向存储中添加一个条目，并且该条目将保留 600 秒（10 分钟）。现在，除非再次访问，否则此产品将在接下来的 10 分钟内列在最近产品列表中，这将会再次将时间重置为 10 分钟。

要测试这一点，向您的数据库添加一些产品：

```py
>>> requests.post('http://127.0.0.1:5000/product-create', data={'name': 'iPhone 5S', 'price': '549.0', 'category': 'Phones'})
>>> requests.post('http://127.0.0.1:5000/product-create', data={'name': 'iPhone 13', 'price': '799.0', 'category': 'Phones'})
>>> requests.post('http://127.0.0.1:5000/product-create', data={'name': 'iPad Pro', 'price': '999.0', 'category': 'Tablets'})
>>> requests.post('http://127.0.0.1:5000/product-create', data={'name': 'iPhone 5S', 'price': '549.0', 'category': 'Phones'})
```

然后，通过在浏览器中打开产品 URL 来简单地访问一些产品——例如，`http://127.0.0.1:5000/product/1`和`http://127.0.0.1:5000/product/3`。

现在，在浏览器中打开`http://127.0.0.1:5000/recent-products`以查看最近产品的列表：

```py
{
  "products": [
    "iPad Pro",
    "iPhone 5S"
  ]
}
```

# 选择 MongoDB 的 NoSQL 方式

有时，我们在构建的应用程序中要使用的数据可能根本不是结构化的；它可能是半结构化的，或者可能有一些数据其模式随时间频繁变化。在这种情况下，我们会避免使用 RDBMS，因为它增加了痛苦，并且难以扩展和维护。对于这种情况，最好使用 NoSQL 数据库。

此外，由于当前流行的开发环境中的快速和快速开发，设计完美的模式并不总是可能的。NoSQL 提供了修改模式而无需太多麻烦的灵活性。

在生产环境中，数据库通常会在一段时间内增长到巨大的规模。这极大地影响了整个系统的性能。有垂直和水平扩展技术可用，但有时它们可能非常昂贵。在这种情况下，可以考虑使用 NoSQL 数据库，因为它从头开始就是为了类似的目的而设计的。NoSQL 数据库能够在大型多个集群上运行并处理以高速生成的大量数据，这使得它们在处理传统 RDBMS 的扩展问题时成为一个不错的选择。

在这个菜谱中，我们将使用 MongoDB 来学习如何将 NoSQL 与 Flask 集成。

## 准备工作

有许多扩展可用于使用 Flask 与 MongoDB。我们将使用`Flask-MongoEngine`，因为它提供了良好的抽象级别，这使得它易于理解。可以使用以下命令进行安装：

```py
$ pip install flask-mongoengine
```

记得运行 MongoDB 服务器以建立连接。有关安装和运行 MongoDB 的更多详细信息，请参阅[`docs.mongodb.org/manual/installation/`](http://docs.mongodb.org/manual/installation/)。

## 如何操作...

首先，使用命令行在 MongoDB 中手动创建一个数据库。让我们把这个数据库命名为 `my_catalog`：

```py
>>> mongosh
Current Mongosh Log ID:  62fa8dtfd435df654150997b
Connecting to:  mongodb://127.0.0.1:27017/?directConnection
=true&serverSelectionTimeoutMS=2000&appName=mongosh+1.5.4
Using MongoDB:    6.0.0
Using Mongosh:    1.5.4
test> use my_catalog
switched to db my_catalog
```

以下是我们使用 MongoDB 重写的目录应用程序。第一个变化出现在我们的配置文件 `my_app/__init__.py` 中：

```py
from flask import Flask
from flask_mongoengine import MongoEngine
app = Flask(__name__)
app.config['MONGODB_SETTINGS'] = {'DB': 'my_catalog'}
app.debug = True
db = MongoEngine(app)
from my_app.catalog.views import catalog
app.register_blueprint(catalog)
```

信息

注意，我们不再使用通常以 SQLAlchemy 为中心的设置，我们现在有 `MONGODB_SETTINGS`。在这里，我们只需指定要使用的数据库名称，在我们的例子中是 `my_catalog`。

接下来，我们将使用 MongoDB 字段创建一个 `Product` 模型。这通常在模型文件 `my_app/catalog/models.py` 中完成：

```py
import datetime
from my_app import db
class Product(db.Document):
    created_at = db.DateTimeField(
        default=datetime.datetime.now, required=True
    )
    key = db.StringField(max_length=255, required=True)
    name = db.StringField(max_length=255, required=True)
    price = db.DecimalField()
    def __repr__(self):
        return '<Product %r>' % self.id
```

重要信息

现在是查看用于创建前面模型的 MongoDB 字段及其与之前配方中使用的 SQLAlchemy 字段相似性的好时机。在这里，我们有一个 `key` 字段而不是 `ID` 字段，它存储将用于唯一标识记录的唯一标识符。此外，请注意在创建模型时 `Product` 继承的类。在 SQLAlchemy 的情况下，它是 `db.Model`，而在 MongoDB 的情况下，它是 `db.Document`。这符合这些数据库系统的工作方式。SQLAlchemy 与传统的 RDBMS 一起工作，但 MongoDB 是一个 NoSQL 文档数据库系统。

以下是我们文件，即 `my_app/catalog/views.py`：

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
    return 'Product - %s, $%s' % (product.name,
      product.price)
@catalog.route('/products')
def products():
    products = Product.objects.all()
    res = {}
    for product in products:
        res[product.key] = {
            'name': product.name,
            'price': str(product.price),
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

你会注意到它与为基于 SQLAlchemy 的模型创建的视图非常相似。只是从 MongoEngine 扩展调用的方法有一些细微的差别，这些应该很容易理解。

## 它是如何工作的...

首先，通过使用 `/product-create` 端点将产品添加到数据库中：

```py
>>> res = requests.post('http://127.0.0.1:5000/product-create', data={'key': 'iphone-5s', 'name': 'iPhone 5S', 'price': '549.0'})
```

现在，通过在浏览器中访问 `http://127.0.0.1:5000/products` 端点来验证产品添加。以下是将得到的 JSON 值：

```py
{
  "iphone-5s": {
    "name": "iPhone 5S",
    "price": "549.00"
  }
}
```

## 参见

参考创建基本产品模型的配方，了解此应用程序的结构。
