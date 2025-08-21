# 第七章：构建在线商店

在上一章中，你创建了关注系统和用户活动流。你还学习了 Django 信号是如何工作的，并在项目中集成了 Redis，用于计算图片的浏览次数。在这一章中，你会学习如何构建一个基本的在线商店。你会创建商品目录（catalog），并用 Django 会话（session）实现购物车。你还会学习如果创建自定义上下文管理器，以及用 Celery 启动异步任务。

在这一章中，你会学习：

- 创建商品目录
- 使用 Django 会话创建购物车
- 管理客户订单
- 用 Celery 给客户发送异步通知

## 7.1 创建在线商店项目

我们将创建一个新的 Django 项目来构建在线商店。用户可以通过商品目录浏览，并把商品添加到购物车中。最后，客户结账并下单。本章将会覆盖在线商店以下几个功能：

- 创建商品目录模型，把它们添加到管理站点，并创建一个基础视图，用于显示目录
- 使用 Django 会话构建购物车系统，允许用户浏览网站时保留选定的商品
- 创建用于下单的表单和功能
- 用户下单后，发送一封异步确认邮件给用户

首先，我们为新项目创建虚机环境，并用以下命令激活虚拟环境：

```py
mkdiv env
virtualenv env/myshop
source env/myshop/bin/activate
```

使用以下命令在虚拟环境中安装 Django：

```py
pip install Django
```

打开终端，执行以下命令，创建`myshop`项目，以及`shop`应用：

```py
django-admin startproject myshop
cd myshop
django-admin startapp shop
```

编辑项目的`settings.py`文件，在`INSTALLED_APPS`设置中添加`shop`应用：

```py
INSTALLED_APPS = (
	# ...
	'shop',
)
```

现在项目中的应用已经激活。让我们为商品目录定义模型。

### 7.1.1 创建商品目录模型

商店的目录由属于不同类别的商品组成。每个商品有名字，可选的描述，可选的图片，价格，以及有效的库存。编辑你刚创建的`shop`应用的`models.py`文件，添加以下代码：

```py
from django.db import models

class Category(models.Model):
    name = models.CharField(max_length=200, db_index=True)
    slug = models.SlugField(max_length=200, db_index=True, unique=True)

    class Meta:
        ordering = ('name', )
        verbose_name = 'category'
        verbose_name_plural = 'categories'
    
    def __str__(self):
        return self.name

class Product(models.Model):
    category = models.ForeignKey(Category, related_name='products')
    name = models.CharField(max_length=200, db_index=True)
    slug = models.SlugField(max_length=200, db_index=True)
    image = models.ImageField(upload_to='products/%Y/%m/%d', blank=True)
    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.PositiveIntegerField()
    available = models.BooleanField(default=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ('name', )
        index_together = (('id', 'slug'), )

    def __str__(self):
        return self.name
```

这是我们的`Category`和`Product`模型。`Category`模型由`name`字段和唯一的`slug`字段组成。`Product`模型包括以下字段：

- `category`：这是指向`Catetory`模型的`ForeignKey`。这是一个多对一的关系：一个商品属于一个目录，而一个目录包括多个商品。
- `name`：这是商品的名称。
- `slug`：这是商品的别名，用于构建友好的 URL。
- `image`：这是一张可选的商品图片。
- `description`：这是商品的可选描述。
- `price`：这是`DecimalField`。这个字段用 Python 的`decimal.Decimal`类型存储固定精度的十进制数。使用`max_digits`属性设置最大的位数（包括小数位），使用`decimal_places`属性设置小数位。
- `stock`：这个`PositiveIntegerField`存储商品的库存。
- `available`：这个布尔值表示商品是否有效。这允许我们在目录中启用或禁用商品。
- `created`：对象创建时存储该字段。
- `updated`：对象最后更新时存储该字段。

对于`price`字段，我们使用`DecimalField`代替`FloatField`，来避免四舍五入的问题。

> 总是使用`DecimalField`存储货币值。在 Python 内部，`FloatField`使用`float`类型，而`DecimalField`使用`Decimal`类型。使用`Decimal`类型可以避免`float`的四舍五入问题。

在`Product`模型的`Meta`类中，我们用`index_together`元选项为`id`和`slug`字段指定共同索引。这是因为我们计划通过`id`和`slug`来查询商品。两个字段共同索引可以提升用这两个字段查询的性能。

因为我们要在模型中处理图片，打开终端，用以下命令安装`Pillow`：

```py
pip install Pillow
```

现在，执行以下命令，创建项目的初始数据库迁移：

```py
python manage.py makemigrations
```

你会看到以下输出：

```py
Migrations for 'shop':
  shop/migrations/0001_initial.py
    - Create model Category
    - Create model Product
    - Alter index_together for product (1 constraint(s))
```

执行以下命令同步数据：

```py
python manage.py migrate
```

你会看到包括这一行的输出：

```py
Applying shop.0001_initial... OK
```

现在数据库与模型已经同步了。

### 7.1.2 在管理站点注册目录模型

让我们把模型添加到管理站点，从而可以方便的管理目录和商品。编辑`shop`应用的`admin.py`文件，添加以下代码：

```py
from django.contrib import admin
from .models import Category, Product

class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'slug')
    prepopulated_fields = {'slug': ('name', )}
admin.site.register(Category, CategoryAdmin)

class ProductAdmin(admin.ModelAdmin):
    list_display = ('name', 'slug', 'price', 'stock', 'available', 'created', 'updated')
    list_filter = ('available', 'created', 'updated')
    list_editable = ('price', 'stock', 'available')
    prepopulated_fields = {'slug': ('name', )}
admin.site.register(Product, ProductAdmin)
```

记住，我们使用`prepopulated_fields`属性指定用其它字段的值自动填充的字段。正如你前面看到的，这样可以很容易的生成别名。我们在`ProductAdmin`类中使用`list_editable`属性设置可以在管理站点的列表显示页面编辑的字段。这样可以一次编辑多行。`list_editable`属性中的所有字段都必须列在`list_display`属性中，因为只有显示的字段才可以编辑。

现在使用以下命令为网站创建超级用户：

```py
python manage.py createsuperuser
```

执行`python manage.py runserver`命令启动开服务器。在浏览器中打开`http://127.0.0.1:8000/admin/shop/product/add/`，然后用刚创建的用户登录。使用管理界面添加一个新的目录和商品。管理页面的商品修改列表页面看起来是这样的：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE7.1.png)

### 7.1.3 构建目录视图

为了显示商品目录，我们需要创建一个视图列出所有商品，或者通过制定的目录过滤商品。编辑`shop`应用的`views.py`文件，添加以下代码：

```py
from django.shortcuts import render, get_object_or_404
from .models import Category, Product

def product_list(request, category_slug=None):
    category = None
    categories = Category.objects.all()
    products = Product.objects.filter(available=True)
    if category_slug:
        category = get_object_or_404(Category, slug=category_slug)
        products = products.filter(category=category)
    return render(request, 
                  'shop/product/list.html', 
                  {'category': category, 
                  'categories': categories, 
                  'products': products})
```

我们用`available=True`过滤`QuerySet`，只检索有效地商品。我们用可选的`category_slug`参数，过滤指定目录的商品。

我们还需要一个查询和显示单个商品的视图。添加以下代码到`views.py`文件中：

```py
def product_detail(request, id, slug):
    product = get_object_or_404(Product, id=id, slug=slug, available=True)
    return render(request,
                'shop/product/detail.html',
                {'product': product})
```

`product_detail`视图接收`id`和`slug`参数来查询`Product`实例。我们可以只使用 ID 获得该实例，因为 ID 是唯一性的属性。但是我们会在 URL 中包括别名，为商品构建搜索引擎友好的 URL。

创建商品列表和详情视图后，我们需要为它们定义 URL 模式。在`shop`应用目录中创建`urls.py`文件，添加以下代码：

```py
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.product_list, name='product_list'),
    url(r'^(?P<category_slug>[-\w]+)/$', views.product_list, name='product_list_by_category'),
    url(r'^(?P<id>\d+)/(?P<slug>[-\w]+)/$', views.product_detail, name='product_detail'),
]
```

这些是商品目录的 URL 模式。我们为`product_list`视图定义了两个不同的 URL 模式：`product_list`模式不带任何参数调用`product_list`视图；`product_list_by_category`模式给视图提供`category_slug`参数，用于过滤指定目录的商品。我们添加了`product_detail`模式，传递`id`和`slug`参数给视图，用于检索特定商品。

编辑`myshop`项目的`urls.py`文件，如下所示：

```py
from django.conf.urls import url, include
from django.contrib import admin

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^', include('shop.urls', namespace='shop')),
]
```

我们在项目的主 URL 模式中引入了`shop`应用的 URL，并指定命名空间为`shop`。

现在编辑`shop`应用的`models.py`文件，导入`reverse()`函数，并为`Category`和`Product`模型添加`get_absolute_url()`方法，如下所示：

```py
from django.core.urlresolvers import reverse
# ...
class Category(models.Model):
	# ...
	def get_absolute_url(self):
        return reverse('shop:product_list_by_category', args=[self.slug])
        
class Product(models.Model):
	# ...
	def get_absolute_url(self):
        return reverse('shop:product_detail', args=[self.id, self.slug])
```

你已经知道，`get_absolute_url()`是检索指定对象 URL 的约定成俗的方法。我们在这里使用之前在`urls.py`文件中定义的 URL 模式。

### 7.1.4 创建目录模板

现在我们需要为商品列表和详情视图创建模板。在`shop`应用目录中创建以下目录和文件结构：

```py
templates/
	shop/
		base.html
		product/
			list.html
			detail.html
```

我们需要定义一个基础模板，并在商品列表和详情模板中继承它。编辑`shop/base.html`模板，添加以下代码：

```py
{% load static %}
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>{% block title %}My shop{% endblock %}</title>
    <link href="{% static "css/base.css" %}" rel="stylesheet">
</head>
<body>
    <div id="header">
        <a href="/" class="logo">My shop</a>
    </div>
    <div id="subheader">
        <div class="cart">
            Your cart is empty.
        </div>
    </div>
    <div id="content">
        {% block content %}
        {% endblock %}
    </div>
</body>
</html>
```

这是商店的基础模板。为了引入模板使用的 CSS 样式表和图片，你需要拷贝本章实例中的静态文件，它们位于`shop`应用的`static/`目录。把它们拷贝到你的项目中的相同位置。

编辑`shop/product/list.html`模板，添加以下代码：

```py
{% extends "shop/base.html" %}
{% load static %}

{% block title %}
    {% if category %}{{ category.name }}{% else %}Products{% endif %}
{% endblock title %}

{% block content %}
    <div id="sidebar">
        <h3>Categories</h3>
        <ul>
            <li {% if not category %}class="selected"{% endif %}>
                <a href="{% url "shop:product_list" %}">All</a>
            </li>
            {% for c in categories %}
                <li {% if category.slug == c.slug %}class="selected"{% endif %}>
                    <a href="{{ c.get_absolute_url }}">{{ c.name }}</a>
                </li>
            {% endfor %}
        </ul>
    </div>
    <div id="main" class="product-list">
        <h1>{% if catetory %}{{ category.name }}{% else %}Products{% endif %}</h1>
        {% for product in products %}
            <div class="item">
                <a href="{{ product.get_absolute_url }}">
                    <img src="{% if product.image %}{{ product.image.url }}{% else %}{% static "img/no_image.png" %}{% endif %}">
                </a>
                <a href="{{ product.get_absolute_url }}">{{ product.name }}</a><br/>
                ${{ product.price }}
            </div>
        {% endfor %}
    </div>
{% endblock content %}
```

这是商品列表目录。它继承自`shop/base.html`目录，用`categories`上下文变量在侧边栏显示所有目录，用`products`显示当前页商品。用同一个模板列出所有有效商品和通过目录过滤的所有商品。因为`Product`模型的`image`字段可以为空，所以如果商品没有图片时，我们需要提供一张默认图片。图片位于静态文件目录，相对路径为`img/no_image.png`。

因为我们用`ImageField`存储商品图片，所以需要开发服务器管理上传的图片文件。编辑`myshop`的`settings.py`文件，添加以下设置：

```py
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media/')
```

`MEDIA_URL`是管理用户上传的多媒体文件的基础 URL。`MEDIA_ROOT`是这些文件的本地路径，我们在前面添加`BASE_DIR`变量，动态生成该路径。

要让 Django 管理通过开发服务器上传的多媒体文件，需要编辑`myshop`项目的`urls.py`文件，如下所示：

```py
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # ...
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

记住，我们只在开发阶段这么做。在生产环境，你不应该用 Django 管理静态文件。

使用管理站点添加一些商品，然后在浏览器中打开`http://127.0.0.1:8000/`。你会看到商品列表页面，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE7.2.png)

如果你用管理站点创建了一个商品，但是没有上传图片，则会显示默认图片：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE7.3.png)

让我们编辑商品详情模板。编辑`shop/product/detail.html`模板，添加以下代码：

```py
{% extends "shop/base.html" %}
{% load static %}

{% block titie %}
    {% if category %}{{ category.title }}{% else %}Products{% endif %}
{% endblock titie %}

{% block content %}
    <div class="product-detail">
        <img src="{% if product.image %}{{ product.image.url }}{% else %} {% static "img/no_image.png" %}{% endif %}">
        <h1>{{ product.name }}</h1>
        <h2><a href="{{ product.category.get_absolute_url }}">{{ product.category }}</a></h2>
        <p class="price">${{ product.price }}</p>
        {{ product.description|linebreaks }}
    </div>
{% endblock content %}
```

我们在关联的目录对象上调用`get_absolute_url()`方法，来显示属于同一个目录的有效商品。现在在浏览器中打开`http://127.0.0.1/8000/`，点击某个商品查看详情页面，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE7.4.png)

我们现在已经创建了一个基本的商品目录。

## 7.2 构建购物车

创建商品目录之后，下一步是创建购物车，让用户选择他们希望购买的商品。当用户浏览网站时，购物车允许用户选择并暂时存储他们想要的商品，直到最后下单。购物车存储在会话中，所以在用户访问期间可以保存购物车里的商品。

我们将使用 Django 的会话框架保存购物车。购物车会一直保存在会话中，直到完成购物或者用户结账离开。我们还需要为购物车和它的商品创建额外的 Django 模型。

### 7.2.1 使用 Django 会话

Django 提供了一个会话框架，支持匿名和用户会话。会话框架允许你为每个访问者存储任意数据。会话数据保存在服务端，cookies 包括会话 ID，除非你使用基于 cookie 的会话引擎。会话中间件负责发送和接收 cookies。默认的会话引擎在数据库中存储会话数据，但是接下来你会看到，可以选择不同的会话引擎。要使用会话，你必须确保项目的`MIDDLEWARE_CLASSES`设置中包括`django.contrib.sessions.middleware.SessionMiddleware`。这个中间件负责管理会话，当你用`startproject`命令创建新项目时，会默认添加这个中间件。

会话中间件让当前会话在`request`对象中生效。你可以使用`request.session`访问当前会话，与使用 Python 字典类似的存储和检索会话数据。会话字典默认接收所有可以序列化为 JSON 的 Python 对象。你可以这样在会话中设置变量：

```py
request.session['foo'] = 'bar'
```

查询一个会话的键：

```py
request.session.get('foo')
```

删除存储在会话中的键：

```py
del request.session['foo']
```

正如你所看到的，我们把`request.session`当做标准的 Python 字典。

> 当用户登录到网站时，他们的匿名会话丢失，并未认证用户创建新的会话。如果你在匿名会话中存储了数据，并想在用户登录后保留，你需要旧的会话数据拷贝到新的会话中。

### 7.2.2 会话设置

你可以使用几种设置为项目配置会话。其中最重要的是`SESSION_ENGINE`。该设置允许你设置会话存储的位置。默认情况下，Django 使用`django.contrib.sessions`应用的`Session`模型，把会话存储在数据库中。

Django 为存储会话数据提供了以下选项：

- `Database sessions`：会话数据存储在数据库中。这是默认的会话引擎。
- `File-based sessions`：会话数据存储在文件系统中。
- `Cached sessions`：会话数据存储在缓存后台。你可以使用`CACHES`设置指定婚车后台。在缓存系统中存储会话数据的性能最好。
- `Cached database sessions`：会话数据存储在连续写入的缓存（write-through cache）和数据库中。只有在缓存中没有数据时才读取数据库。
- `Cookie-based sessions`：会话数据存储于发送到浏览器的 cookies。

> 使用`cache-based`会话引擎有更好的性能。Django 支持 Memcached，以及其它支持 Redis 的第三方缓存后台和缓存系统。

你可以只是用其它设置自定义会话。以下是一些重要的会话相关设置：

- `SESSION_COOKIE_AGE`：这是会话 cookies 的持续时间（单位是秒）。默认值是 1209600（两周）。
- `SESSION_COOKIE_DOMAIN`：会话 cookies 使用的域。设置为`.mydomain.com`可以启用跨域 cookies。
- `SESSION_EXPIRE_AT_BROWSER_CLOSE`：当浏览器关闭后，表示会话是否过期的一个布尔值。
- `SESSION_SAVE_EVERY_REQUEST`：如果这个布尔值为`True`，则会在每次请求时把会话保存到数据库中。会话的过期时间也会每次更新。

你可以在[这里](https://docs.djangoproject.com/en/1.11/ref/settings/#sessions)查看所有会话设置。

### 7.2.3 会话过期

你可以使用`SESSTION_EXPIRE_AT_BROWSER_CLOSE`设置选择`browser-length`会话或者持久会话。默认值为`False`，强制把会话的有效期设置为`SESSION_COOKIE_AGE`的值。如果设置`SESSTION_EXPIRE_AT_BROWSER_CLOSE`为`True`，当用户关闭浏览器后，会话会过期，而`SESSION_COOKIE_AGE`不会起任何作用。

你可以使用`request.session`的`set_expiry()`方法覆写当前会话的有效期。

### 7.2.4 在会话中存储购物车

我们需要创建一个简单的可以序列号为 JSON 的结构体，在会话中存储购物车商品。购物车的每一件商品必须包括以下数据：

- `Product`实例的`id`
- 选择该商品的数量
- 该商品的单价

因为商品价格可能变化，所以当商品添加到购物车时，我们把商品的价格和商品本身同事存入购物车。这样的话，即使之后商品的价格发生变化，用户看到的还是添加到购物车时的价格。

现在你需要创建购物车，并与会话关联起来。购物车必须这样工作：

- 需要购物车时，我们检查是否设置了自定义会话键。如果会话中没有设置购物车，则创建一个新的购物车，并保存在购物车会话键中。
- 对于连续的请求，我们执行相同的检查，并从购物车会话键中取出购物车的商品。我们从会话中检索购物车商品，并从数据库中检索它们关联的`Product`对象。

编辑项目`settings.py`文件，添加以下设置：

```py
CART_SESSION_ID = 'cart'
```

我们在用户会话用这个键存储购物车。因为每个访客的 Django 会话是独立的，所以我们可以为所有会话使用同一个购物车会话键。

让我们创建一个管理购物车的应用。打开终端，执行以下命令创建一个新应用：

```py
python manage.py startapp cart
```

然后编辑项目的`settings.py`文件，把`cart`添加到`INSTALLED_APPS`：

```py
INSTALLED_APPS = (
	# ...
	'cart',
)
```

在`cart`应用目录中创建`cart.py`文件，并添加以下代码：

```py
from decimal import Decimal
from django.conf import settings
from shop.models import Product

class Cart:
    def __init__(self, request):
        self.session = request.session
        cart = self.session.get(settings.CART_SESSION_ID)
        if not cart:
            # save an empty cart in the session
            cart = self.session[settings.CART_SESSION_ID] = {}
        self.cart = cart
```

这个`Cart`类用于管理购物车。我们要求用`request`对象初始化购物车。我们用`self.session = request.session`存储当前会话，以便在`Cart`类的其它方法中可以访问。首先，我们用`self.session.get(settings.CART_SESSION_ID)`尝试从当前会话中获得购物车。如果当前会话中没有购物车，通过在会话中设置一个空字典来设置一个空的购物车。我们希望购物车字典用商品 ID 做为键，一个带数量和价格的字典作为值。这样可以保证一个商品不会在购物车中添加多次；同时还可以简化访问购物车的数据。

让我们创建一个方法，用于向购物车中添加商品，或者更新商品数量。在`Cart`类中添加`add()`和`save()`方法：

```py
def add(self, product, quantity=1, update_quantity=False):
    product_id = str(product.id)
    if product_id not in self.cart:
        self.cart[product_id] = {
            'quantity': 0,
            'price': str(product.price)
        }
    if update_quantity:
        self.cart[product_id]['quantity'] = quantity
    else:
        self.cart[product_id]['quantity'] += quantity
    self.save()

def save(self):
    # update the session cart
    self.session[settings.CART_SESSION_ID] = self.cart
    # mark the sessions as "modified" to make sure it is saved
    self.session.modified = True
```

`add()`方法接收以下参数：

- `product`：在购物车中添加或更新的`Product`实例。
- `quantity`：可选的商品数量。默认为 1.
- `update_quantity`：一个布尔值，表示使用给定的数量更新数量（`True`），或者把新数量加到已有的数量上（`False`）。

我们用商品`id`作为购物车内容字典的键。因为 Django 使用 JSON 序列号会话数据，而 JSON 只允许字符串类型的键名，所以我们把商品`id`转换为字符串。商品`id`是键，保存的值是带商品`quantity`和`price`的字典。为了序列号，我们把商品价格转换为字符串。最后，我们调用`save()`方法在会话中保存购物车。

`save()`方法在会话中保存购物车的所有修改，并使用`session.modified = True`标记会话已修改。这告诉 Django，会话已经修改，需要保存。

我们还需要一个方法从购物车中移除商品。在`Cart`类中添加以下方法：

```py
def remove(self, product):
    product_id = str(product.id)
    if product_id in self.cart:
        del self.cart[product_id]
        self.save()
```

`remove()`方法从购物车字典中移除指定商品，并调用`save()`方法更新会话中的购物车。

我们将需要迭代购物车中的商品，并访问关联的`Product`实例。因为需要在类中定义`__iter__()`方法。在`Cart`类中添加以下方法：

```py
def __iter__(self):
    product_ids = self.cart.keys()
    # get the product objects and add them to the cart
    products = Product.objects.filter(id__in=product_ids)
    for product in products:
        self.cart[str(product.id)]['product'] = product

    for item in self.cart.values():
        item['price'] = Decimal(item['price'])
        item['total_price'] = item['price'] * item['quantity']
        yield item
```

在`__iter__()`方法中，我们检索购物车中的`Product`实例，并把它们包括在购物车商品中。最后，我们迭代购物车商品，把`price`转换回`Decimal`类型，并为每一项添加`total_price`属性。现在我们可以在购物车中方便的迭代商品。

我们还需要返回购物车中商品总数量。当在一个对象上调用`len()`函数时，Python 会调用`__len__()`方法返回对象的长度。我们定义一个`__len__()`方法，返回购物车中所有商品的总数量。在`Cart`类中添加`__len__()`方法：

```py
def __len__(self):
    return sum(item['quantity'] for item in self.cart.values())
```

我们返回购物车中所有商品数量的总和。

添加以下方法，计算购物车中所有商品的总价：

```py
def get_total_price(self):
	return sum(Decimal(item['price']) * item['quantity'] for item in self.cart.values())
```

最后，添加一个清空购物车会话的方法：

```py
def clear(self):
    del self.session[settings.CART_SESSION_ID]
    self.session.modified = True
```

我们的`Cart`类已经可以管理购物车了。

### 7.2.5 创建购物车视图

现在我们已经创建了`Cart`类来管理购物车，我们需要创建添加，更新和移除购物车商品的视图。我们需要创建以下视图：

- 一个添加或更新购物车商品的视图，可以处理当前和新的数量
- 一个从购物车中移除商品的视图
- 一个显示购物车商品和总数的视图

#### 7.2.5.1 添加商品到购物车

要添加商品到购物车中，我们需要一个用户可以选择数量的表单。在`cart`应用目录中创建`forms.py`文件，并添加以下代码：

```py
from django import forms

PRODUCT_QUANTITY_CHOICES = [(i, str(i)) for i in range(1, 21)]

class CartAddProductForm(forms.Form):
    quantity = forms.TypedChoiceField(choices=PRODUCT_QUANTITY_CHOICES, coerce=int)
    update = forms.BooleanField(required=False, initial=False, widget=forms.HiddenInput)
```

我们用这个表单向购物车中添加商品。`CartAddProductForm`类包括以下两个字段：

- `quantity`：允许用户选择 1-20 之间的数量。我们使用带`coerce=int`的`TypedChoiceField`字段把输入的值转换为整数。
- `update`：允许你指定把数量累加到购物车中已存在的商品数量上（`False`），还是用给定的数量更新已存在商品数量（`True`）。我们为该字段使用`HiddenInput`组件，因为我们不想让用户看见它。

让我们创建向购物车添加商品的视图。编辑`cart`应用的`views.py`文件，并添加以下代码：

```py
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST
from shop.models import Product
from .cart import Cart
from .forms import CartAddProductForm

@require_POST
def cart_add(request, product_id):
    cart = Cart(request)
    product = get_object_or_404(Product, id=product_id)
    form = CartAddProductForm(request.POST)
    if form.is_valid():
        cd = form.cleaned_data
        cart.add(product=product, quantity=cd['quantity'], update_quantity=cd['update'])
    return redirect('cart:cart_detail')
```

这个视图用于向购物车中添加商品或者更新已有商品的数量。因为这个视图会修改数据，所以我们只允许 POST 请求。视图接收商品 ID 作为参数。我们用给定的商品 ID 检索`Product`实例，并验证`CartAddProductForm`。如果表单有效，则添加或更新购物车中的商品。该视图重定向到`cart_detail` URL，它会显示购物车中的内容。之后我们会创建`cart_detail`视图。

我们还需要一个从购物车中移除商品的视图。在`cart`应用的`views.py`文件中添加以下代码：

```py
def cart_remove(request, product_id):
    cart = Cart(request)
    product = get_object_or_404(Product, id=product.id)
    cart.remove(product)
    return redirect('cart:cart_detail')
```

`cart_remove`视图接收商品 ID 作为参数。我们用给定的商品 ID 检索`Product`实例，并从购物车中移除该商品。接着我们重定向到`cart_detail` URL。

最后，我们需要一个显示购物车和其中的商品的视图。在`views.py`文件中添加以下代码：

```py
def cart_detail(request):
    cart = Cart(request)
    return render(request, 'cart/detail.html', {'cart': cart})
```

`cart_detail`视图获得当前购物车，并显示它。

我们已经创建了以下视图：向购物车中添加商品，更新数量，从购物车中移除商品，已经显示购物车。让我们为这些视图添加 URL。在`cart`应用目录中创建`urls.py`文件，并添加以下 URL 模式：

```py
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.cart_detail, name='cart_detail'),
    url(r'^add/(?P<product_id>\d+)/$', views.cart_add, name='cart_add'),
    url(r'^remove/(?P<product_id>\d+)/$', views.cart_remove, name='cart_remove'),
]
```

最后，编辑`myshop`项目的主`urls.py`文件，引入`cart`的 URL 模式：

```py
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^cart/', include('cart.urls', namespace='cart')),
    url(r'^', include('shop.urls', namespace='shop')),
]
```

确保在`shop.urls`模式之前引入这个 URL 模式，因为它比前者更有限定性。

#### 7.2.5.2 构建显示购物车的模板

`cart_add`和`cart_remove`视图不需要渲染任何模板，但是我们需要为`cart_detail`视图创建显示购物车和总数量的模板。

在`cart`应用目录中创建以下文件结构：

```py
templates/
	cart/
		detail.html
```

编辑`cart/detail.html`目录，并添加以下代码：

```py
{% extends "shop/base.html" %}
{% load static %}

{% block title %}
    Your shopping cart
{% endblock title %}

{% block content %}
    <h1>Your shopping cart</h1>
    <table class="cart">
        <thead>
            <tr>
                <th>Image</th>
                <th>Product</th>
                <th>Quantity</th>
                <th>Remove</th>
                <th>Unit price</th>
                <th>Price</th>
            </tr>
        </thead>
        <tbody>
            {% for item in cart %}
                {% with product=item.product %}
                    <tr>
                        <td>
                            <a href="{{ prouct.get_absolute_url }}">
                                <img src="{% if product.image %}{{ product.image.url}}{% else %}{% static "img/no_image.png" %}{% endif %}">
                            </a>
                        </td>
                        <td>{{ product.name }}</td>
                        <td>{{ item.quantity }}</td>
                        <td><a href="{% url "cart:cart_remove" product.id %}">Remove</a></td>
                        <td class="num">${{ item.price }}</td>
                        <td class="num">${{ item.total_price }}</td>
                    </tr>
                {% endwith %}
            {% endfor %}
            <tr class="total">
                <td>Total</td>
                <td colspan="4"></td>
                <td class="num">${{ cart.get_total_price }}</td>
            </tr>
        </tbody>
    </table>
    <p class="text-right">
        <a href="{% url "shop:product_list" %}" class="button light">Continue shopping</a>
        <a href="#" class="button">Checkout</a>
    </p>
{% endblock content %}
```

这个模板用于显示购物车的内容。它包括一个当前购物车中商品的表格。用户通过提交表单到`cart_add`视图，来修改选中商品的数量。我们为每个商品提供了`Remove`链接，用户可以从购物车移除商品。

#### 7.2.5.3 添加商品到购物车

现在我们需要在商品详情页面添加`Add to cart`按钮。编辑`shop`应用的`views.py`文件，修改`product_detail`视图，如下所示：

```py
from cart.forms import CartAddProductForm

def product_detail(request, id, slug):
    product = get_object_or_404(Product, id=id, slug=slug, available=True)
    cart_product_form = CartAddProductForm()
    return render(request,
                'shop/product/detail.html',
                {'product': product,
                'cart_product_form': cart_product_form})
```

编辑`shop`应用的`shop/product/detail.html`模板，在商品价格之后添加表单，如下所示：

```py
<p class="price">${{ product.price }}</p>
<form action="{% url "cart:cart_add" product.id %}" method="post">
    {{ cart_product_form }}
    {% csrf_token %}
    <input type="submit" value="Add to cart">
</form>
```

使用`python manage.py runserver`命令启动开发服务器。在浏览器中打开`127.0.0.1/8000/`，然后导航到商品详情页面。它现在包括一个选择数量的表单，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE7.5.png)

选择数量，然后点击`Add to cart`按钮。表单通过 POST 提交到`cart_add`视图。该视图把商品添加到会话中的购物车，包括当前价格和选择的数量。然后重定义到购物车详情页面，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE7.6.png)

#### 7.2.5.4 在购物车中更新商品数量

当用户查看购物车时，他们可能希望在下单前修改商品数量。我们接下来实现在购物车详情页面修改数量。

编辑`cart`应用的`views.py`文件，如下修改`cart_detail`视图：

```py
def cart_detail(request):
    cart = Cart(request)
    for item in cart:
        item['update_quantity_form'] = CartAddProductForm(
            initial={'quantity': item['quantity'], 'update': True})
    return render(request, 'cart/detail.html', {'cart': cart})
```

我们为购物车中的每个商品创建了一个`CartAddProductForm`实例，允许用户修改商品数量。我们用当前商品数量初始化表单，并设置`update`字段为`True`。因此，当我们把表单提交到`cart_add`视图时，会用新数量了代替当前数量。

现在编辑`cart`应用的`cart/detail.html`模板，找到这一行代码：

```py
<td>{{ item.quantity }}</td>
```

把这行代码替换为：

```py
<td>
	<form action="{% url "cart:cart_add" product.id %}" method="post">
	    {{ item.update_quantity_form.quantity }}
	    {{ item.update_quantity_form.update }}
	    <input type="submit" value="Update">
	    {% csrf_token %}
	</form>
</td>
```

在浏览器中打开`http://127.0.0.1:8000/cart/`。你会看到购物车中每个商品都有一个修改数量的表单，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE7.7.png)

修改商品数量，然后点击`Update`按钮，测试一下新功能。

### 7.2.6 为当前购物车创建上下文处理器

你可能已经注意到了，我们的网站头部还是显示`Your cart is emtpy`。当我们开始向购物车中添加商品，我们将看到它替换成购物车中商品的总数量和总价钱。因为这是需要在所有页面显示，所以我们将创建一个上下文处理器（context processor），将当前购物车包含在请求上下文中，而不管已经处理的视图。

#### 7.2.6.1 上下文处理器

上下文处理器是一个 Python 函数，它将`request`对象作为参数，并返回一个添加到请求上下文中的字典。当你需要让某些东西在所有模板都可用时，它会派上用场。

默认情况下，当你使用`startproject`命令创建新项目时，项目中会包括以下模板上下文处理器，它们位于`TEMPLATES`设置的`context_processors`选项中：

- `django.template.context_processors.debug`：在上下文中设置`debug`布尔值和`sql_queries`变量，表示请求中执行的 SQL 查询列表
- `django.template.context_processors.request`：在上下文中设置`request`变量
- `django.contrib.auth.context_processors.auth`：在请求中设置`user`变量
- `django.contrib.messages.context_processors.messages`：在上下文中设置`message`变量，其中包括所有已经用消息框架发送的消息。

Django 还启用了`django.template.context_processors.csrf`来避免跨站点请求伪造攻击。这个上下文处理器不在设置中，但它总是启用的，并且为了安全不能关闭。

你可以在[这里](https://docs.djangoproject.com/en/1.11/ref/templates/api/#built-in-template-context-processors)查看所有内置的上下文处理器列表。

#### 7.2.6.2 在请求上下文中设置购物车

让我们创建一个上下文处理器，把当前购物车添加到模板的请求上下文中。我们可以在所有模板中访问购物车。

在`cart`应用目录中创建`context_processors.py`文件。上下文处理器可以位于代码的任何地方，但是在这里创建他们将保持代码组织良好。在文件中添加以下代码：

```py
from .cart import Cart

def cart(request):
    return {'cart': Cart(request)}
```

正如你所看到的，上下文处理器是一个函数，它将`request`对象作为参数，并返回一个对象的字典，这些对象可用于所有使用`RequestContext`渲染的模板。在我们的上下文处理器中，我们用`request`对象实例化购物车，模板可以通过`cart`变量名访问它。

编辑项目的`settings.py`文件，在`TEMPLATES`设置的`context_processors`选项中添加`cart.context_processors.cart`，如下所示：

```py
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'cart.context_processors.cart',
            ],
        },
    },
]
```

每次使用`RequestContext`渲染模板时，会执行你的上下文处理器。`cart`变量会设置在模板的上下文中。

> 上下文处理器会在所有使用`RequestContext`的请求中执行。如果你想访问数据库的话，可能希望创建一个自定义模板标签来代替上下文处理器。

现在编辑`shop`应用的`shop/base.html`模板，找到以下代码：

```py
<div class="cart">
	Your cart is empty.
</div>
```

用下面的代码替换上面的代码：

```py
<div class="cart">
	{% with total_items=cart|length %}
	    {% if cart|length > 0 %}
	        Your cart:
	        <a href="{% url "cart:cart_detail" %}">
	            {{ total_items }} item{{ total_items|pluralize }},
	            ${{ cart.get_total_price }}
	        </a>
	    {% else %}
	        Your cart is empty.
	    {% endif %}
	{% endwith %}
</div>
```

使用`python manage.py runserver`重启开发服务器。在浏览器中打开`http://127.0.0.1:8000/`，并添加一些商品到购物车中。在网站头部，你会看到当前购物车总数量和总价钱，如下所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE7.8.png)

## 7.3 注册用户订单

当购物车结账后，你需要在数据库中保存订单。订单包括用户信息和他们购买的商品。

使用以下命令创建一个新应用，来管理用户订单：

```py
python manage.py startapp orders
```

编辑项目的`settings.py`文件，在`INSTALLED_APPS`设置中添加`orders`：

```py
INSTALLED_APPS = [
	# ...
	'orders',
]
```

你已经激活了新应用。

### 7.3.1 创建订单模型

你需要创建一个模型存储订单详情，以及一个模型存储购买的商品，包括价格和数量。编辑`orders`应用的`models.py`文件，添加以下代码：

```py
from django.db import models
from shop.models import Product

class Order(models.Model):
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    email = models.EmailField()
    address = models.CharField(max_length=250)
    postal_code = models.CharField(max_length=20)
    city = models.CharField(max_length=100)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    paid = models.BooleanField(default=False)

    class Meta:
        ordering = ('-created', )

    def __str__(self):
        return 'Order {}'.format(self.id)

    def get_total_cost(self):
        return sum(item.get_cost() for item in self.items.all())

class OrderItem(models.Model):
    order = models.ForeignKey(Order, related_name='items')
    product = models.ForeignKey(Product, related_name='order_items')
    price = models.DecimalField(max_digits=10, decimal_places=2)
    quantity = models.PositiveIntegerField(default=1)

    def __str__(self):
        return '{}'.format(self.id)

    def get_cost(self):
        return self.price * self.quantity
```

`Order`模型包括几个用户信息字段和一个默认值为`False`的`paid`布尔字段。之后，我们将用这个字段区分已支付和未支付的订单。我们还定义了`get_total_cost()`方法，获得这个订单中购买商品的总价钱。

`OrderItem`模型允许我们存储商品，数量和每个商品的支付价格。我们用`get_cost()`返回商品价钱。

运行以下命令，为`orders`应用创建初始数据库迁移：

```py
python manage.py makemigrations
```

你会看到以下输出：

```py
Migrations for 'orders':
  orders/migrations/0001_initial.py
    - Create model Order
    - Create model OrderItem
```

运行以下命令让新的迁移生效：

```py
python manage.py migrate
```

现在你的订单模型已经同步到数据库中。

### 7.3.2 在管理站点引入订单模型

让我们在管理站点添加订单模型。编辑`orders`应用的`admin.py`文件，添加以下代码：

```py
from django.contrib import admin
from .models import Order, OrderItem

class OrderItemInline(admin.TabularInline):
    model = OrderItem
    raw_id_fields = ['product']

class OrderAdmin(admin.ModelAdmin):
    list_display = ['id', 'first_name', 'last_name', 'email', 
        'address', 'postal_code', 'city', 'paid', 'created', 'updated']
    list_filter = ['paid', 'created', 'updated']
    inlines = [OrderItemInline]

admin.site.register(Order, OrderAdmin)
```

我们为`OrderItem`模型使用`ModeInline`，把它作为内联模型引入`OrderAdmin`类。内联可以包含一个模型，与父模型在同一个编辑页面显示。

使用`python manage.py runserver`命令启动开发服务器，然后在浏览器中打开`http:127.0.0.1/8000/admin/order/add/`。你会看到以下界面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE7.9.png)

### 7.3.3 创建用户订单

当用户最终下单时，我们需要使用刚创建的订单模型来保存购物车中的商品。创建一个新订单的工作流程是这样的：

1. 向用户显示一个填写数据的订单表单。
2. 用用户输入的数据创建一个新的`Order`实例，然后为购物车中的每件商品创建关联的`OrderItem`实例。
3. 清空购物车中所有内容，然后重定向到成功页面。

首先，我们需要一个输入订单详情的表单。在`orders`应用目录中创建`forms.py`文件，并添加以下代码：

```py
from django import forms
from .models import Order

class OrderCreateForm(forms.ModelForm):
    class Meta:
        model = Order
        fields = ['first_name', 'last_name', 'email', 
            'address', 'postal_code', 'city']
```

这是我们用于创建新`Order`对象的表单。现在我们需要一个视图处理表单和创建新表单。编辑`orders`应用的`views.py`文件，并添加以下代码：

```py
from django.shortcuts import render
from .models import OrderItem
from .forms import OrderCreateForm
from cart.cart import Cart

def order_create(request):
    cart = Cart(request)
    if request.method == 'POST':
        form = OrderCreateForm(request.POST)
        if form.is_valid():
            order = form.save()
            for item in cart:
                OrderItem.objects.create(order=order, product=item['product'], 
                    price=item['price'], quantity=item['quantity'])
            # clear the cart
            cart.clear()
            return render(request, 'orders/order/created.html', {'order': order})
    else:
        form = OrderCreateForm()
    return render(request, 'orders/order/create.html', {'cart': cart, 'form': form})
```

在`order_create`视图中，我们用`cart = Cart(request)`从会话中获得当前购物车。根据请求的方法，我们执行以下任务：

- `GET`请求：实例化`OrderCreateForm`表单，并渲染`orders/order/create.html`模板。
- `POST`请求：验证提交的数据。如果数据有效，则使用`order = form.save()`创建一个新的`Order`实例。然后我们会将它保存到数据库中，并存储在`order`变量中。创建`order`之后，我们会迭代购物车中的商品，并为每个商品创建`OrderItem`。最后，我们会清空购物车的内容。

现在，在`orders`应用目录中创建`urls.py`文件，并添加以下代码：

```py
from django.conf.urls import url
from .import views

urlpatterns = [
    url(r'^create/$', views.order_create, name='order_create'),
]
```

这是`order_create`视图的 URL 模式。编辑`myshop`项目的`urls.py`文件，并引入以下模式。记住，把它放在`shop.urls`模式之前：

```py
url(r'^orders/', include('orders.urls', namespace='orders')),
```

编辑`cart`应用的`cart/detail.html`模板，找到这行代码：

```py
<a href="#" class="button">Checkout</a>
```

把这样代码替换为以下代码：

```py
<a href="{% url "orders:order_create" %}" class="button">Checkout</a>
```

现在用户可以从购物车详情页面导航到订单表单。我们还需要为下单定义模板。在`orders`应用目录中创建以下文件结构：

```py
templates/
	orders/
		order/
			create.html
			created.html
```

编辑`orders/order/create.html`模板，并添加以下代码：

```py
{% extends "shop/base.html" %}

{% block title %}
    Checkout
{% endblock title %}

{% block content %}
    <h1>Checkout</h1>

    <div class="order-info">
        <h3>Your order</h3>
        <ul>
            {% for item in cart %}
                <li>
                    {{ item.quantity }}x {{ item.product.name }}
                    <span>${{ item.total_price }}</span>
                </li>
            {% endfor %}
        </ul>
        <p>Total: ${{ cart.get_total_price }}</p>
    </div>

    <form action="." method="post" class="order-form">
        {{ form.as_p }}
        <p><input type="submit" value="Place order"></p>
        {% csrf_token %}
    </form>
{% endblock content %}
```

这个模板显示购物车中的商品，包括总数量和下单的表单。

编辑`orders/order/created.html`模板，并添加以下代码：

```py
{% extends "shop/base.html" %}

{% block title %}
    Thank you
{% endblock title %}

{% block content %}
    <h1>Thank you</h1>
    <p>Your order has been successfully completed. 
        Your order number is <stong>{{ order.id }}</stong>
    </p>
{% endblock content %}
```

成功创建订单后，我们渲染这个模板。启动开发服务器，并在浏览器中打开`http://127.0.0.1:8000/`。在购物车中添加一些商品，然后跳转到结账界面。如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE7.10.png)

用有效的数据填写表单，然后点击`Place order`按钮。订单会被创建，你将看到成功页面，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE7.11.png)

## 7.4 使用 Celery 启动异步任务

你在视图中执行的所有操作都会影响响应时间。在很多场景中，你可能希望尽快给用户返回响应，并让服务器执行一些异步处理。对于费时处理，或者失败后可能需要重试策略的处理尤其重要。例如，一个视频分享平台允许用户上传视频，但转码上传的视频需要很长的时间。网站可能给用户返回一个响应，告诉用户马上开始转码，然后开始异步转码。另一个例子是给用户发送邮件。如果网站在视图中发送邮件通知，SMTP 连接可能失败，或者减慢响应时间。启动异步任务避免阻塞操作是必不可少的。

Celery 是一个可以处理大量消息的分布式任务队列。它既可以实时处理，也支持任务调度。使用 Celery 不仅可以很容易的创建异步任务，还可以尽快执行任务，但也可以在一个指定时间执行任务。

你可以在[这里](http://celery.readthedocs.org/en/latest/)查看 Celery 文档。

### 7.4.1 安装 Celery

让我们安装 Celery，并在项目中集成它。使用以下`pip`命令安装 Celery：

```py
pip install celery
```

Celery 必须有一个消息代理（message broker）处理外部请求。代理负责发送消息给 Celery 的`worker`，`worker`收到消息后处理任务。让我们安装一个消息代理。

### 7.4.2 安装 RabbitMQ

Celery 有几个消息代理可供选择，包括键值对存储（比如 Redis），或者一个实际的消息系统（比如 RabbitMQ）。我们将用 RabbitMQ 配置 Celery，因为它是 Celery 的推荐消息 worker。

如果你使用的是 Linux，可以在终端执行以下命令安装 RabbitMQ：

```py
apt-get install rabbitmq
```

如果你需要在 Max OS X 或者 Windows 上安装 RabbitMQ，你可以在[这里](https://www.rabbitmq.com/download.html)找到独立的版本。

安装后，在终端执行以下命令启动 RabbitMQ：

```py
rabbitmq-server
```

你会看到以这一行结尾的输出：

```py
Starting broker... completed with 10 plugins.
```

### 7.4.3 在项目中添加 Celery

你需要为 Celery 实例提供一个配置。在`myshop`中创建`celery.py`文件，该文件会包括项目的 Celery 配置，并添加以下代码：

```py
import os
from celery import Celery
from django.conf import settings

# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myshop.settings')

app = Celery('myshop')

app.config_from_object('django.conf:settings')
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)
```

在这段代码中，我们为 Celery 命令行程序设置`DJANGO_SETINGS_MODULE`变量。然后用`app = Celery('myshop.)`创建了一个应用实例。我们用`config_from_object()`方法从项目设置中加载所有自定义设置。最后我们告诉 Celery，为`INSTALLED_APPS`设置中列出的应用自动查找异步任务。Celery 会在每个应用目录中查找`tasks.py`文件，并加载其中定义的异步任务。

你需要在项目的`__init__.py`文件中导入`celery`模块，确保 Django 启动时会加载 Celery。编辑`myshop/__init__.py`文件，并添加以下代码：

```py
from .celery import app as celery_app
```

现在你可以开始为应用编写异步任务了。

> `CELERY_ALWAYS_EAGER`设置允许你以同步方式在本地执行任务，而不是将其发送到队列。这对于运行单元测试，或者在不运行 Celery 的情况下，运行本地环境中的项目时非常有用。

### 7.4.4 在应用中添加异步任务

当用户下单后，我们将创建一个异步任务，给用户发送一封邮件通知。

一般的做法是在应用目录的`tasks`模块中包括应用的异步任务。在`orders`应用目录中创建`tasks.py`文件。Celery 会在这里查找异步任务。在其中添加以下代码：

```py
from celery import task
from django.core.mail import send_mail
from .models import Order

@task
def order_created(order_id):
    order = Order.objects.get(id=order_id)
    subject = 'Order nr. {}'.format(order.id)
    message = 'Dear {},\n\nYou have successfully placed an order.\
    	Your order id is {}.'.format(order.first_name, order.id)
    mail_sent = send_mail(subject, message, 'admin@myshop.com', [order.email])
    return mail_sent
```

我们使用`task`装饰器定义了`order_created`任务。正如你所看到的，Celery 任务就是一个用`task`装饰的 Python 函数。我们的`task`函数接收`order_id`作为参数。推荐只传递 ID 给任务函数，并在任务执行时查询对象。我们用 Django 提供的`send_mail()`函数，当用户下单后发送邮件通知。如果你不想配置邮件选项，你可以在项目的`settings.py`文件中添加以下设置，让 Django 在控制台输出邮件：

```py
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
```

> 异步任务不仅可以用于费时的操作，还可以用于可能失败的操作，这些操作不会执行很长时间，但它们可能会连接失败，或者需要重试策略。

现在我们需要在`order_create`视图中添加任务。打开`orders`应用的`views.py`文件，并导入任务：

```py
from .tasks import order_created
```

然后在清空购物车之后调用`order_created`异步任务：

```py
# clear the cart
cart.clear()
# launch asynchronous task
order_created.delay(order.id)
```

我们调用任务的`delay()`方法异步执行任务。任务会被添加到队列中，`worker`会尽快执行。

打开另一个终端，并使用以下命令启动 Celery 的`worker`：

```py
celery -A myshop worker -l info
```

> **译者注：**必须在`myshop`项目目录下执行上面的命令。

现在 Celery 的`worker`已经运行，准备好处理任务了。确保 Django 开发服务器也在运行。在浏览器中打开`http://127.0.0.1/8000/`，添加一些商品到购物车中，然后完成订单。在终端，你已经启动了 Celery `worker`，你会看到类似这样的输出：

```py
[2017-05-11 06:40:27,416: INFO/MainProcess] Received task: orders.tasks.order_created[4d6f667b-7cc7-4310-82fc-8323810fae54]
[2017-05-11 06:40:27,825: INFO/PoolWorker-3] Task orders.tasks.order_created[4d6f667b-7cc7-4310-82fc-8323810fae54] succeeded in 0.12212000600266038s: 1
```

任务已经执行，你会收到一封订单的邮件通知。

### 7.4.5 监控 Celery

你可能希望监控已经执行的异步任务。Flower 是一个基于网页的监控 Celery 工具。你可以使用`pip install flower`安装 Flower。

安装后，你可以在项目目录下执行以下命令启动 Flower：

```py
celery -A myshop flower
```

在浏览器中打开`http://127.0.0.1:5555/dashboard`。你会看到活动的 Celery `worker`和异步任务统计：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE7.12.png)

你可以在[这里](http://flower.readthedocs.org/en/latest/)查看 Flower 的文档。

## 7.5 总结

在这章中，你创建了一个基础的在线商店应用。你创建了商品目录，并用会话构建了购物车。你实现了自定义上下文处理器，让模板可以访问购物车，并创建了下单的表单。你还学习了如何使用 Celery 启动异步任务。

在下一章中，你会学习在商店中集成支付网关（payment gateway），在管理站点添加用户操作，导出 CVS 格式数据，以及动态生成 PDF 文件。