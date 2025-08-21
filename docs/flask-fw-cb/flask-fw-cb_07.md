# 第七章：构建 RESTful API

API，即应用编程接口，可以概括为应用对开发者的接口。就像用户有一个可以和应用沟通的可视化界面一样，开发者同样需要一个接口和应用交互。REST，即表现层状态转移，它不是一个协议或者标准。它只是一种软件架构风格，或者是为编写应用程序定义的一组约束，旨在简化应用程序内外接口。当 web 服务 API 遵循了 REST 风格进行编写时，它们就可以称为 RESTful API。RESTful 使得 API 和应用内部细节分离。这使得扩展很容易，并且使得事情变得简单。统一接口确保每个请求都得文档化。

###### 提示

关于 REST 和 SOAP 哪个好存在一个争论。它实际上是一个主观问题，因为它取决于需要做什么。每个都有它自己的好处，应该根据应用程序的需要来进行选择。

这一章，我们将包含下面小节：

*   创建一个基于类的 REST 接口
*   创建一个基于扩展的 REST 接口
*   创建一个 SQLAlchemy-independent REST API
*   一个完整的 REST API 例子

## 介绍

从名字可以看出，表现层状态转移（REST）意味着可以分离 API 到逻辑资源，这些资源可以通过使用 HTTP 请求获得和操作，一个 HTTP 请求由 GET，POST，PUT，PATCH，DELETE 中的一个（还有其他 HTTP 方法，但这些是最常使用的）。这些方法中的每一个都有一个特定的意义。REST 的关键隐含原则之一是资源的逻辑分组应该是简单容易理解的，提供简单性和可移植性。
这本书到这里，我们一直在使用一个资源叫做 Product。让我们来看看怎么讲 API 调用映射到资源分离上：

*   GET /products/1:获取 ID 为 1 的商品
*   GET /products:获取商品列表
*   POST /products:创建一个新商品
*   PUT /products/1:更新 ID 为 1 的商品
*   PATCH /products/1:部分更新 ID 为 1 的商品
*   DELETE /products/1:删除 ID 为 1 的商品

## 创建一个基于类的 REST 接口

在第四章里我们看到了在 Flask 里如何使用基于类的视图。我们将使用相同的概念去创建视图，为我们应用提供 REST 接口。

#### 准备

让我们写一个简单的视图来处理 Product 模型的 REST 接口。

#### 怎么做

需要简单的修改商品视图，来继承 MethodView 类：

```py
from flask.views import MethodView

class ProductView(MethodView):

    def get(self, id=None, page=1):
        if not id:
            products = Product.query.paginate(page, 10).items
            res = {}
            for product in products:
                res[product.id] = {
                    'name': product.name,
                    'price': product.price,
                    'category': product.category.name
                }
            # 译者注 加上这一句，否则会报错
            res = json.dumps(res)
        else:
            product = Product.query.filter_by(id=id).first()
            if not product:
                abort(404)
            res = json.dumps({
                'name': product.name,
                'price': product.price,
                'category': product.category.name
            })
        return res 
```

get()方法搜索 product，然后返回 JSON 结果。
可以用同样的方式完成 post(),put(),delete()方法：

```py
def post(self):
    # Create a new product.
    # Return the ID/object of newly created product.
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

很多人会想为什么我们没在这里写路由。为了包含路由，我们得像下面这样做：

```py
product_view = ProductView.as_view('product_view')
app.add_url_rule('/products/', view_func=product_view, methods=['GET', 'POST'])
app.add_url_rule('/products/<int:id>', view_func=product_view, methods=['GET', 'PUT', 'DELETE']) 
```

第一句首先转换类为实际的视图函数，这样才可以用在路由系统中。后面两句是 URL 规则和其对应的请求方法。

###### 译者注

测试时如果遇到/products/路由已经注册，原因可能是第四章已经定义了一个/products/视图函数，注释掉即可，或者修改这里的路由名称。

#### 原理

MethodView 类定义了请求中的 HTTP 方法，并将名字转为小写。请求到来时，HTTP 方法匹配上类中定义的方法，就会调用相应的方法。所以，如果对 ProductView 进行一个 GET 调用，它将自动的匹配上 get()方法。

#### 更多

我们还可以使用一个叫做 Flask-Classy 的扩展（`https://pythonhosted.or/Flask-Classy`）。这将在很大程度上自动处理类和路由，并使生活更加美好。我们不会在这里讨论这些，但它是一个值得研究的扩展。

## 创建基于扩展的 REST 接口

前面一节中，我们看到如何使用热插拔的视图创建一个 REST 接口。这里我们将使用一个 Flask 扩展叫做 Flask-Restless。Flask-Restless 是完全为了构建 REST 接口而开发的。它提供了一个简单的为使用 SQLAlchemy 创建的数据模型构建 RESTful APIs 的方法。这些生成的 api 以 JSON 格式发送和接收消息。

#### 准备

首先，需安装 Flask-Restless 扩展：

```py
$ pip install Flask-Restless 
```

我们借用第四章的程序构建我们的应用，以此来包含 RESTful API 接口。

###### 提示

如果 views 和 handlers 的概念不是很清楚，建议在继续阅读之前，先去阅读第四章。

#### 怎么做

通过使用 Flask-Restless 是非常容易向一个 SQLAlchemy 模型新增 RESTful API 接口的。首先，需向应用新增扩展提供的 REST API 管理器，然后通过使用 app 对象创建一个实例：

```py
from flask_restless import APIManager
manager = APIManager(app, flask_sqlalchemy_db=db) 
```

之后，我们需要通过使用 manager 实例使能模型里的 API 创建。为此，需向 views.py 新增下面代码：

```py
from my_app import manager

manager.create_api(Product, methods=['GET', 'POST', 'DELETE'])
manager.create_api(Category, methods=['GET', 'POST', 'DELETE']) 
```

这将在 Product 和 Category 模型里创建 GET，POST，DELETE 这些 RESTful API。通常，如果 methods 参数缺失的话，只支持 GET 方法。

#### 原理

为了测试和理解这些是如何工作的，我们通过使用 Python requests 库发送一些请求:

```py
>>> import requests
>>> import json
>>> res = requests.get("http://127.0.0.1:5000/api/category")
>>> res.json()
{u'total_pages': 0, u'objects': [], u'num_results': 0, u'page': 1} 
```

###### 译者注

res.json()可能会从出错，可使用 res.text

我们发送了一个 GET 请求去获取类别列表，但是现在没有记录。来看一下商品：

```py
>>> res = requests.get('http://127.0.0.1:5000/api/product')
>>> res.json()
{u'total_pages': 0, u'objects': [], u'num_results': 0, u'page': 1} 
```

我们发送了一个 GET 请求去获取商品列表，但是没有记录。现在让我们创建一个商品：

```py
>>> d = {'name': u'iPhone', 'price': 549.00, 'category':{'name':'Phones'}}
>>> res = requests.post('http://127.0.0.1:5000/api/product', data=json.dumps(d), headers={'Content-Type': 'application/json'})
>>> res.json()
{u'category': {u'id': 1, u'name': u'Phones'}, u'name': u'iPhone', 
u'company': u'', u'price': 549.0, u'category_id': 1, u'id': 2, u'image_path': u''} 
```

我们发送了一个 POST 请求去创建一个商品。注意看请求里的 headers 参数。每个发给 Flask-Restless 的 POST 请求都应该包含这个头。现在，我们再一次搜索商品列表：

```py
>>> res = requests.get('http://127.0.0.1:5000/api/product')
>>> res.json()
{u'total_pages': 1, u'objects': [{u'category': {u'id': 1, u'name': u'Phones'}, u'name': u'iPhone', u'company': u'', u'price': 549.0, u'category_id': 1, u'id': 1, u'image_path': u''}], u'num_results': 1, u'page': 1} 
```

我们可以看到新创建的商品已经在数据库中了。
同样需要注意的是，查询结果默认已经分好页了，这是优秀的 API 的标识之一。

#### 更多

自动创建 RESTful API 接口非常的酷，但是每个应用都需要一些自定义，验证，处理业务的逻辑。
这使得使用 preprocessors 和 postprocessors 成为可能。从名字可以看出，preprocessors 会在请求被处理前运行，postprocessors 会在请求处理完，发送给应用前运行。它们被定义在 create_api()中，做为请求类型（GET，POST 等）映射，并且作为前处理程序或后处理程序的方法列表，用于处理指定的请求：

```py
manager.create_api(
    Product,
    methods=['GET', 'POST', 'DELETE'],
    preprocessors={
        'GET_SINGLE': ['a_preprocessor_for_single_get'],
        'GET_MANY': ['another_preprocessor_for_many_get'],
        'POST': ['a_preprocessor_for_post']
    },
    postprocessors={
        'DELETE': ['a_postprocessor_for_delete']
    }
) 
```

单个或多个记录都可以调用 GET，PUT，PATCH 方法；但是它们各有两个变体（variants）。举个例子，前面的代码里，对于 GET 请求有 GET_SINGLE 和 GET_MANY。preprocessors 和 postprocessors 对于各自请求接收不同的参数，然后执行它们，并且没有返回值。参见`https://flask-restless.readthedocs.org/en/latest/`了解更多细节。

###### 译者注

对 preprocessor 和 postprocessors 的理解，参见`http://flask-restless.readthedocs.io/en/stable/customizing.html#request-preprocessors-and-postprocessors`

## 创建一个 SQLAlchemy-independent REST API

在前一小节中，我们看到了如何使用依赖于 SQLAlchemy 的扩展创建一个 REST API 接口。现在我们将使用一个名为 Flask-Restful 的扩展，它是在 Flask 可插拔视图上编写的，并且独立于 ORM。

#### 准备

首先，安装扩展:

```py
$ pip install Flask-Restful 
```

我们将修改前面的商品目录应用，通过使用这个扩展增加一个 REST 接口。

#### 怎么做

通常，首先要修改应用的配置，看起来像这样：

```py
from flask_restful import Api   

api = Api(app) 
```

这里，app 是我们应用的对象/实例。
接下来，在 views.py 里创建 API。在这里，我们将尝试理解 API 的框架，更详细的实现在下一小节里：

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

前面的 API 结构是很容易理解的。看下面代码：

```py
api.add_resource(ProductApi, '/api/product', '/api/product/<int:id>') 
```

这里，我们为 ProductApi 创建路由，我们可以根据需要指定多条路由。

#### 原理

我们将使用 Python requests 库在看这些是如何工作的，就像前一小节那样：

```py
>>> import requests
>>> res = requests.get('http://127.0.0.1:5000/api/product')
>>> res.json()
u'This is a GET response'
>>> res = requests.post('http://127.0.0.1:5000/api/product')
>u'This is a POST response'
>>> res = requests.put('http://127.0.0.1:5000/api/product/1')
u'This is a PUT response'
>>> res = requests.delete('http://127.0.0.1:5000/api/product/1')
u'This is a DELETE response' 
```

在前面一小段代码中，我们看到了我们的请求被相应的方法处理了；从回复中可以确认这一点。

#### 其他

*   确保在继续向下阅读之前先阅读完这一小节

## 一个完整的 REST API 例子

这一小节，我们将上一小节的 API 框架改写为一个完整的 RESTful API 接口。

#### 准备

我们将使用上一小节的 API 框架作为基础，来创建一个完整的 SQLAlchemy-independent RESTful API。尽管我们使用 SQLAlchemy 作为 ORM 来进行演示，这一小节可以使用任何 ORM 或者底层数据库进行编写。

#### 怎么做

下面的代码是 Product 模型完整的 RESTful API 接口。views.py 看起来像这样：

```py
from flask_restful import reqparse

parser = reqparse.RequestParser()
parser.add_argument('name', type=str)
parser.add_argument('price', type=float)
parser.add_argument('category', type=dict) 
```

前面的一小段代码，我们为希望在 POST，PUT 请求中解析出来的参数创建了 parser。请求期待每个参数不是空值。如果任何参数的值是缺失的，则将使用 None 做为值。看下面代码：

```py
class ProductApi(Resource):

    def get(self, id=None, page=1):
        if not id:
            products = Product.query.paginate(page, 10).items
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

前面的 get 方法对应于 GET 请求，如果没有传递 id，将返回商品分好页的商品列表；否则，返回匹配的商品。看下面 POST 请求代码：

```py
def post(self):
    args = parser.parse_args()
    name = args['name']
    price = args['price']
    categ_name = args['category']['name']
    category = Category.query.filter_by(name=categ_name).first()
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

前面 post()方法将在 POST 请求时创建一个新的商品。看下面代码：

```py
def put(self, id):
    args = parser.parse_args()
    name = args['name']
    price = args['price']
    categ_name = args['category']['name']
    category = Category.query.filter_by(name=categ_name).first()
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

前面代码，通过 PUT 请求更新了一个已经存在的商品。这里，我们应该提供所有的参数，即使我们仅仅想更新一部分。这是因为 PUT 被定义的工作方式就是这样。如果我们想要一个请求只传递那些我们想要更新的参数，这应该使用 PATCH 请求。看下面代码：

```py
def delete(self, id):
    product = Product.query.filter_by(id=id)
    product.delete()
    db.session.commit()
    return json.dumps({'response': 'Success'}) 
```

最后同样重要的是，DELETE 请求将删除匹配上 id 的商品。看下面代码：

```py
api.add_resource(
    ProductApi,
    '/api/product',
    '/api/product/<int:id>',
    '/api/product/<int:id>/<int:page>'
) 
```

上一句代码是我们的 API 可以容纳的所有 URL 的定义。

###### 提示

REST API 的一个重要方面是基于令牌的身份验证，它只允许有限和经过身份验证的用户能够使用和调用 API。这将留给你自己探索。我们在第六章 Flask 认证中介绍的用户身份验证的基础知识，将作为此概念的基础。

