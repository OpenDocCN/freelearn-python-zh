# 7

# RESTful API 构建

**应用程序编程接口**（**API**）可以概括为开发者与应用程序的接口。正如最终用户有一个可见的前端用户界面，他们可以通过它与应用程序交互和交流一样，开发者也需要一个与之交互的接口。**表示状态转换**（**REST**）不是一个协议或标准。它只是一个软件架构风格或一系列为编写应用程序而定义的建议，其目的是简化应用程序内部和外部的接口。当以符合 REST 定义的方式编写网络服务 API 时，它们被称为 RESTful API。保持 RESTful 可以使 API 与内部应用程序细节解耦。这导致易于扩展并保持简单。统一的接口确保每个请求都有文档记录。

信息

关于 REST 或简单的对象访问协议(SOAP)哪个更好，这是一个有争议的话题。这实际上是一个主观问题，因为它取决于需要做什么。每种方法都有自己的优点，应根据应用程序的需求进行选择。

REST 调用用于将 API 分割成逻辑资源，这些资源可以通过 HTTP 请求访问和操作，其中每个请求都包含以下方法之一——`GET`、`POST`、`PUT`、`PATCH`和`DELETE`（可能有更多，但这些都是最常用的）。这些方法中的每一个都有其特定的含义。REST 的一个关键隐含原则是资源的逻辑分组应该是易于理解的，因此可以提供简单性和可移植性。

我们有一个名为`product`的资源，正如我们在本书中迄今为止所使用的。现在，让我们看看我们如何逻辑地将我们的 API 调用映射到资源分割：

+   `GET /products/1`: 这将获取 ID 为`1`的产品

+   `GET /products`: 这将获取产品列表

+   `POST /products`: 这将创建一个新的产品

+   `PUT /products/1`: 这将替换或重新创建 ID 为`1`的产品

+   `PATCH /products/1`: 这将部分更新 ID 为`1`的产品

+   `DELETE /products/1`: 这将删除 ID 为`1`的产品

在本章中，我们将介绍以下菜谱：

+   创建基于类的 REST 接口

+   创建基于扩展的 REST 接口

+   创建完整的 RESTful API

# 创建基于类的 REST 接口

我们在*第四章*的*编写基于类的视图*菜谱中看到了如何在 Flask 中使用可插拔视图的概念，*与视图一起工作*。在这个菜谱中，我们现在将看到如何使用相同的方法来创建视图，这些视图将为我们的应用程序提供 REST 接口。

## 准备就绪

让我们看看一个简单的视图，它将处理对`Product`模型的 REST 风格调用。

## 如何做到这一点...

我们只需修改`views.py`中的产品处理视图，以扩展`MethodView`类：

```py
import json
from flask.views import MethodView
class ProductView(MethodView):
    def get(self, id=None, page=1):
        if not id:
            products = Product.query.paginate(page,
              10).items
            res = {}
            for product in products:
                res[product.id] = {
                    'name': product.name,
                    'price': product.price,
                    'category': product.category.name
                }
        else:
            product =
              Product.query.filter_by(id=id).first()
            if not product:
                abort(404)
                res = json.dumps({
                    'name': product.name,
                    'price': product.price,
                    'category': product.category.name
                })
        return res
```

紧接着的 `get()` 方法会查找产品并发送回 JSON 结果。同样，我们也可以编写 `post()`、`put()` 和 `delete()` 方法：

```py
def post(self):
    # Create a new product.
    # Return the ID/object of the newly created product.
    return
def put(self, id):
    # Update the product corresponding provided id.
    # Return the JSON corresponding updated product.
    return
def delete(self, id):
    # Delete the product corresponding provided id.
    # Return success or error message.
    return
```

许多人会质疑为什么这里没有路由。要包含路由，我们必须做以下事情：

```py
product_view =  ProductView.as_view('product_view')
app.add_url_rule('/products/', view_func=product_view,
    methods=['GET', 'POST'])
app.add_url_rule('/products/<int:id>',
    view_func=product_view,
    methods=['GET', 'PUT', 'DELETE'])
```

这里第一条语句将类内部转换为实际的可用于路由系统的视图函数。接下来的两个语句是与可以进行的调用相对应的 URL 规则。

## 它是如何工作的...

`MethodView` 类识别了发送请求中使用的 HTTP 方法类型，并将名称转换为小写。然后，它将此与类中定义的方法进行匹配，并调用匹配的方法。因此，如果我们向 `ProductView` 发起 `GET` 调用，它将自动映射到 `get()` 方法并相应处理。

# 创建基于扩展的 REST 接口

在之前的菜谱 *创建基于类的 REST 接口* 中，我们看到了如何使用可插拔视图创建 REST 接口。在这个菜谱中，我们将使用一个名为 **Flask-RESTful** 的扩展，它是基于我们在上一个菜谱中使用的相同可插拔视图编写的，但它通过自己处理许多细微差别，使我们开发者能够专注于实际的 API 开发。它也独立于 **对象关系映射**（**ORM**），因此我们想要使用的 ORM 上没有附加条件。

## 准备工作

首先，我们将从安装扩展开始：

```py
$ pip install flask-restful
```

我们将修改上一个菜谱中的目录应用程序，使用此扩展添加 REST 接口。

## 如何做...

如往常一样，从 `my_app/__init__.py` 中应用程序配置的更改开始，它看起来像以下几行代码：

```py
from flask_restful import Api
api = Api(app)
```

在这里，`app` 是我们的 Flask 应用程序对象/实例。

接下来，在 `views.py` 文件中创建 API。在这里，我们只是尝试了解如何安排 API 的框架。实际的方法和处理器将在 *创建完整的 RESTful API* 菜谱中介绍：

```py
from flask_restful import Resource
from my_app import api
class ProductApi(Resource):
    def get(self, id=None):
        # Return product data
        return 'This is a GET response'
    def post(self):
        # Create a new product
        return 'This is a POST response'
    def put(self, id):
        # Update the product with given id
        return 'This is a PUT response'
    def delete(self, id):
        # Delete the product with given id
        return 'This is a DELETE response'
```

前面的 API 结构是自我解释的。考虑以下代码：

```py
api.add_resource(
    ProductApi,
    '/api/product',
    '/api/product/<int:id>'
)
```

在这里，我们为 `ProductApi` 创建了路由，并且可以根据需要指定多个路由。

## 它是如何工作的...

我们将使用 `requests` 库在 Python shell 中查看这个 REST 接口是如何工作的。

信息

`requests` 是一个非常流行的 Python 库，它使得 HTTP 请求的渲染变得非常简单。只需运行 `$ pip install` `requests` 命令即可安装。

命令将显示以下信息：

```py
>>> import requests
>>> res = requests.get('http://127.0.0.1:5000/api/product')
>>> res.json()
'This is a GET response'
>>> res = requests.post('http://127.0.0.1:5000/api/product')
>>> res.json()
'This is a POST response'
>>> res = requests.put('http://127.0.0.1:5000/api/product/1')
>>> res.json()
'This is a PUT response'
>>> res = requests.delete('http://127.0.0.1:5000/api/product/1')
>>> res.json()
'This is a DELETE response'
```

在前面的片段中，我们看到所有我们的请求都正确地路由到了相应的方法；这从收到的响应中可以明显看出。

## 参见

参考以下菜谱，*创建完整的 RESTful API*，以查看本菜谱中的 API 框架如何变得生动。

# 创建完整的 RESTful API

在这个菜谱中，我们将把在最后一个菜谱 *创建基于扩展的 REST 接口* 中创建的 API 结构转换为完整的 RESTful API。

## 准备工作

我们将基于最后一个菜谱中的 API 骨架来创建一个完全独立的 SQLAlchemy RESTful API。虽然我们将使用 SQLAlchemy 作为演示目的的 ORM，但这个菜谱可以用类似的方式为任何 ORM 或底层数据库编写。

## 如何做到这一点...

以下代码行是`Product`模型的完整 RESTful API。这些代码片段将放入`views.py`文件中。

从导入开始并添加`parser`：

```py
import json
from flask_restful import Resource, reqparse
parser = reqparse.RequestParser()
parser.add_argument('name', type=str)
parser.add_argument('price', type=float)
parser.add_argument('category', type=dict)
```

在前面的代码片段中，我们为`POST`和`PUT`请求中预期的参数创建了`parser`。请求期望每个参数都有一个值。如果任何参数缺少值，则使用`None`作为值。

按照以下代码块所示编写方法来获取产品：

```py
class ProductApi(Resource):
    def get(self, id=None, page=1):
        if not id:
            products = Product.query.paginate(page=page,
              per_page=10).items
        else:
            products = [Product.query.get(id)]
        if not products:
            abort(404)
        res = {}
        for product in products:
            res[product.id] = {
                'name': product.name,
                'price': product.price,
                'category': product.category.name
            }
        return json.dumps(res)
```

前面的`get()`方法对应于`GET`请求，如果没有传递`id`，则返回分页的产品列表；否则，返回相应的产品。

创建以下方法来添加一个新的产品：

```py
    def post(self):
        args = parser.parse_args()
        name = args['name']
        price = args['price']
        categ_name = args['category']['name']
        category =
          Category.query.filter_by(name=categ_name).first()
        if not category:
            category = Category(categ_name)
        product = Product(name, price, category)
        db.session.add(product)
        db.session.commit()
        res = {}
        res[product.id] = {
            'name': product.name,
            'price': product.price,
            'category': product.category.name,
        }
        return json.dumps(res)
```

前面的`post()`方法将通过发送`POST`请求来创建一个新的产品。

编写以下方法来更新或本质上替换现有的产品记录：

```py
    def put(self, id):
        args = parser.parse_args()
        name = args['name']
        price = args['price']
        categ_name = args['category']['name']
        category =
          Category.query.filter_by(name=categ_name).first()
        Product.query.filter_by(id=id).update({
            'name': name,
            'price': price,
            'category_id': category.id,
        })
        db.session.commit()
        product = Product.query.get_or_404(id)
        res = {}
        res[product.id] = {
            'name': product.name,
            'price': product.price,
            'category': product.category.name,
        }
        return json.dumps(res)
```

在前面的代码中，我们使用`PUT`请求更新了一个现有的产品。在这里，即使我们打算更改其中的一些参数，我们也应该提供所有参数。这是因为`PUT`被定义成传统的工作方式。如果我们想要一个只传递我们打算更新的参数的请求，那么我们应该使用`PATCH`请求。我敦促你自己尝试一下。

使用以下方法删除产品：

```py
    def delete(self, id):
        product = Product.query.filter_by(id=id)
        product.delete()
        db.session.commit()
        return json.dumps({'response': 'Success'})
```

最后，但同样重要的是，我们有`DELETE`请求，它将简单地删除与传递的`id`匹配的产品。

以下是我们 API 可以容纳的所有可能路由的定义：

```py
api.add_resource(
    ProductApi,
    '/api/product',
    '/api/product/<int:id>',
    '/api/product/<int:id>/<int:page>'
)
```

## 它是如何工作的...

为了测试和查看它是如何工作的，我们可以通过 Python shell 使用`requests`库发送多个请求：

```py
>>> import requests
>>> import json
>>> res = requests.get('http://127.0.0.1:5000/api/product')
>>> res.json()
{'message': 'The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.'}
```

我们发送了一个`GET`请求来获取产品列表，但没有任何记录。现在让我们创建一个新的产品：

```py
>>> d = {'name': u'iPhone', 'price': 549.00, 'category':
...    {'name':'Phones'}}
>>> res = requests.post('http://127.0.0.1:5000/api/product', data=json.
...    dumps(d), headers={'Content-Type': 'application/json'})
>>> res.json()
'{"1": {"name": "iPhone", "price": 549.0, "category": "Phones"}}'
```

我们发送了一个`POST`请求来创建一个带有一些数据的产品。注意请求中的`headers`参数。在 Flask-RESTful 中发送的每个`POST`请求都应该有这个头。现在，我们应该再次查找产品列表：

```py
>>> res = requests.get('http://127.0.0.1:5000/api/product')
>>> res.json()
'{"1": {"name": "iPhone", "price": 549.0, "category": "Phones"}}'
```

如果我们再次通过`GET`请求查找产品，我们可以看到现在数据库中有一个新创建的产品。

我将把它留给你去尝试独立地整合其他 API 请求。

重要

RESTful API 的一个重要方面是使用基于令牌的认证，以允许只有有限的认证用户能够使用和调用 API。我敦促你自己去探索这一点。我们在*第六章*中介绍了用户认证的基础，这将为这个概念提供基础。
