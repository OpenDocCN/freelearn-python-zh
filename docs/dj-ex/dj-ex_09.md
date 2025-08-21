# 第九章：扩展你的商店

上一章中，你学习了如何在商店中集成支付网关。你完成了支付通知，学习了如何生成 CSV 和 PDF 文件。在这一章中，你会在商店中添加优惠券系统。你将学习如何处理国际化和本地化，并构建一个推荐引擎。

本章会覆盖以下知识点：

- 创建优惠券系统实现折扣
- 在项目中添加国际化
- 使用 Rosetta 管理翻译
- 使用 django-parler 翻译模型
- 构建一个商品推荐系统

## 9.1 创建优惠券系统

很多在线商店会给顾客发放优惠券，在购买商品时可以兑换折扣。在线优惠券通常是一组发放给用户的代码，这个代码在某个时间段内有效。这个代码可以兑换一次或多次。

我们将为我们的商品创建一个优惠券系统。顾客在某个时间段内输入我们的优惠券才有效。优惠券没有使用次数限制，可以抵扣购物车的总金额。对于这个功能，我们需要创建一个模型，用于存储优惠券码，有效时间和折扣金额。

使用以下命令在`myshop`项目中添加一个新应用：

```py
python manage.py startapp coupons
```

编辑`myshop`的`settings.py`文件，把应用添加到`INSTALLED_APPS`中：

```py
INSTALLED_APPS = (
	# ...
	'coupons',
)
```

现在新应用已经在我们的 Django 项目中激活了。

### 9.1.1 构建优惠券模型

让我们从创建`Coupon`模型开始。编辑`coupons`应用的`models.py`文件，并添加以下代码：

```py
from django.db import models
from django.core.validators import MinValueValidator
from django.core.validators import MaxValueValidator

class Coupon(models.Model):
    code = models.CharField(max_length=50, unique=True)
    valid_from = models.DateTimeField()
    valid_to = models.DateTimeField()
    discount = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(100)])
    active = models.BooleanField()

    def __str__(self):
        return self.code
```

这是存储优惠券的模型。`Coupon`模型包括以下字段：

- `code`：用户必须输入优惠券码才能使用优惠券。
- `valid_from`：优惠券开始生效的时间。
- `valid_to`：优惠券过期的时间。
- `discount`：折扣率（这是一个百分比，所以范围是 0 到 100）。我们使用验证器限制这个字段的最小值和最大值。
- `active`：表示优惠券是否有效的布尔值。

执行以下命令，生成`coupon`应用的初始数据库迁移：

```py
python manage.py makemigrations
```

输出会包括以下行：

```py
Migrations for 'coupons':
  coupons/migrations/0001_initial.py
    - Create model Coupon
```

然后执行以下命令，让数据库迁移生效：

```py
python manage.py migrate
```

你会看到包括这一行的输出：

```py
Applying coupons.0001_initial... OK
```

现在迁移已经应用到数据库中了。让我们把`Coupon`模型添加到管理站点。编辑`coupons`应用的`admin.py`文件，并添加以下代码：

```py
from django.contrib import admin
from .models import Coupon

class CouponAdmin(admin.ModelAdmin):
    list_display = ['code', 'valid_from', 'valid_to', 'discount', 'active']
    list_filter = ['active', 'valid_from', 'valid_to']
    search_fields = ['code']
admin.site.register(Coupon, CouponAdmin)
```

现在`Coupon`模型已经在管理站点注册。执行`python manage.py runserver`命令启动开发服务器，然后在浏览器中打开`http://127.0.0.1:8000/admin/coupons/coupon/add/`。你会看下图中的表单：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE9.1.png)

填写表单，创建一个当天可用的优惠券，并勾选`Active`，然后点击`Save`按钮。

### 9.1.2 在购物车中使用优惠券

我们可以存储新的优惠券，并且可以查询已存在的优惠券。现在我们需要让顾客可以使用优惠券。考虑一下应该怎么实现这个功能。使用优惠券的流程是这样的：

1. 用户添加商品到购物车中。
2. 用户在购物车详情页面的表单中输入优惠券码。
3. 当用户输入了优惠券码，并提交了表单，我们用这个优惠券码查找当前有效地一张优惠券。我们必须检查这张优惠券码匹配用户输入的优惠券码，`active`属性为`True`，以及当前时间在`valid_from`和`valid_to`之间。
4. 如果找到了优惠券，我们把它保存在用户会话中，显示包括折扣的购物车，然后更新总金额。
5. 当用户下单时，我们把优惠券保存到指定的订单中。

在`coupons`应用目录中创建`forms.py`文件，并添加以下代码：

```py
from django import forms

class CouponApplyForm(forms.Form):
    code = forms.CharField()
```

我们用这个表单让用户输入优惠券码。编辑`coupons`应用的`views.py`文件，并添加以下代码：

```py
from django.shortcuts import render, redirect
from django.utils import timezone
from django.views.decorators.http import require_POST
from .models import Coupon
from .forms import CouponApplyForm

@require_POST
def coupon_apply(request):
    now = timezone.now()
    form = CouponApplyForm(request.POST)
    if form.is_valid():
        code = form.cleaned_data['code']
        try:
            coupon = Coupon.objects.get(code__iexact=code, 
                valid_from__lte=now, 
                valid_to__gte=now, 
                active=True)
            request.session['coupon_id'] = coupon.id
        except Coupon.DoesNotExist:
            request.session['coupon_id'] = None
    return redirect('cart:cart_detail')
```

`coupon_apply`视图验证优惠券，并把它存储在用户会话中。我们用`require_POST`装饰器装饰这个视图，只允许 POST 请求。在视图中，我们执行以下任务：

1. 我们用提交的数据实例化`CouponApplyForm`表单，并检查表单是否有效。
2. 如果表单有效，我们从表单的`cleaned_data`字典中获得用户输入的优惠券码。我们用给定的优惠券码查询`Coupon`对象。我们用`iexact`字段执行大小写不敏感的精确查询。优惠券必须是有效的（`active=True`），并且在当前时间是有效地。我们用 Django 的`timezone.now()`函数获得当前时区的时间，我们把它与`valid_from`和`valid_to`字段比较，对这两个字段分别执行`lte`（小于等于）和`gte`（大于等于）字段查询。
3. 我们在用户会话中存储优惠券 ID。
4. 我们重定向用户到`cart_detail` URL，显示使用了优惠券的购物车。

我们需要一个`coupon_apply`视图的 URL 模式。在`coupons`应用目录中添加`urls.py`文件，并添加以下代码：

```py
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^apply/$', views.coupon_apply, name='apply'),
]
```

然后编辑`myshop`项目的主`urls.py`文件，添加`coupons`的 URL 模式：

```py
url(r'^coupons/', include('coupons.urls', namespace='coupons')),
```

记住，把这个模式放到`shop.urls`模式之前。

现在编辑`cart`应用的`cart.py`文件，添加以下导入：

```py
from coupons.models import Coupon
```

在`Cart`类的`__init__()`方法最后添加以下代码，从当前会话中初始化优惠券：

```py
# store current applied coupon
self.coupon_id = self.session.get('coupon_id')
```

这行代码中，我们尝试从当前会话中获得`coupon_id`会话键，并把它存储到`Cart`对象中。在`Cart`对象中添加以下方法：

```py
@property
def coupon(self):
    if self.coupon_id:
        return Coupon.objects.get(id=self.coupon_id)
    return None
    
def get_discount(self):
	if self.coupon:
		return (self.coupon.discount / Decimal('100') * self.get_total_price())
	return Decimal('0')

def get_total_price_after_discount(self):
	return self.get_total_price() - self.get_discount()
```

这些方法分别是：

- `coupon()`：我们定义这个方法为`property`。如果`cart`中包括`coupon_id`属性，则返回给定`id`的`Coupon`对象。
- `get_discount()`：如果`cart`包括`coupon`，则查询它的折扣率，并返回从购物车总金额中扣除的金额。
- `get_total_price_after_discount()`：减去`get_discount()`方法返回的金额后，购物车的总金额。

现在`Cart`类已经准备好处理当前会话中的优惠券，并且可以减去相应的折扣。

让我们在购物车详情视图中引入优惠券系统。编辑`cart`应用的`views.py`，在文件顶部添加以下导入：

```py
from coupons.forms import CouponApplyForm
```

接着编辑`cart_detail`视图，并添加新表单：

```py
def cart_detail(request):
    cart = Cart(request)
    for item in cart:
        item['update_quantity_form'] = CartAddProductForm(
            initial={'quantity': item['quantity'], 'update': True})
    coupon_apply_form = CouponApplyForm()
    return render(request, 'cart/detail.html', 
        {'cart': cart, 'coupon_apply_form': coupon_apply_form})
```

编辑`cart`应用的`cart/detail.html`目录，找到以下代码：

```py
<tr class="total">
	<td>Total</td>
	<td colspan="4"></td>
	<td class="num">${{ cart.get_total_price }}</td>
</tr>
```

替换为下面的代码：

```py
{% if cart.coupon %}
	<tr class="subtotal">
	    <td>Subtotal</td>
	    <td colspan="4"></td>
	    <td class="num">${{ cart.get_total_price }}</td>
	</tr>
	<tr>
	    <td>
	        "{{ cart.coupon.code }}" coupon
	        ({{ cart.coupon.discount }}% off)
	    </td>
	    <td colspan="4"></td>
	    <td class="num neg">
	        - ${{ cart.get_discount|floatformat:"2" }}
	    </td>
	</tr>
{% endif %}
<tr class="total">
	<td>Total</td>
	<td colspan="4"></td>
	<td class="num">
	    ${{ cart.get_total_price_after_discount|floatformat:"2" }}
	</td>
</tr>
```

这段代码显示一个可选的优惠券和它的折扣率。如果购物车包括一张优惠券，我们在第一行显示购物车总金额为`Subtotal`。然后在第二行显示购物车使用的当前优惠券。最后，我们调用`cart`对象的`cart.get_total_price_after_discount()`方法，显示折扣之后的总金额。

在同一个文件的`</table>`标签之后添加以下代码：

```py
<p>Apply a coupon:</p>
<form action="{% url "coupons:apply" %}" method="post">
    {{ coupon_apply_form }}
    <input type="submit" value="Apply">
    {% csrf_token %}
</form>
```

这会显示输入优惠券码的表单，并在当前购物车中使用。

在浏览器中打开`http://127.0.0.1:8000/`，添加一个商品到购物车中，然后使用表单中输入的优惠券码。你会看到购物车显示优惠券折扣，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE9.2.png)

让我们把优惠券添加到购物流程的下一步。编辑`orders`应用的`orders/order/create.html`模板，找到以下代码：

```py
<ul>
	{% for item in cart %}
	    <li>
	        {{ item.quantity }}x {{ item.product.name }}
	        <span>${{ item.total_price }}</span>
	    </li>
	{% endfor %}
</ul>
```

替换为以下代码：

```py
<ul>
    {% for item in cart %}
        <li>
            {{ item.quantity }}x {{ item.product.name }}
            <span>${{ item.total_price }}</span>
        </li>
    {% endfor %}
    {% if cart.coupon %}
        <li>
            "{{ cart.coupon.code }}" ({{ cart.coupon.discount }}% off)
            <span>- ${{ cart.get_discount|floatformat:"2" }}</span>
        </li>
    {% endif %}
</ul>
```

如果有优惠券的话，订单汇总已经使用了优惠券。现在找到这行代码：

```py
<p>Total: ${{ cart.get_total_price }}</p>
```

替换为下面这一行：

```py
<p>Total: ${{ cart.get_total_price_after_discount|floatformat:"2" }}</p>
```

这样，总价格是使用优惠券之后的价格。

在浏览器中打开中`http://127.0.0.1:8000/orders/create/`。你会看到订单汇总包括了使用的优惠券：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE9.3.png)

现在用户可以在购物车中使用优惠券了。但是当用户结账时，我们还需要在创建的订单中存储优惠券信息。

### 9.1.3 在订单中使用优惠券

我们将存储每个订单使用的优惠券。首先，我们需要修改`Order`模型来存储关联的`Coupon`对象（如果存在的话）。

编辑`orders`应用的`models.py`文件，并添加以下导入：

```py
from decimal import Decimal
from django.core.validators import MinValueValidator
from django.core.validators import MaxValueValidator
from coupons.models import Coupon
```

然后在`Order`模型中添加以下字段：

```py
coupon = models.ForeignKey(Coupon, related_name='orders', null=True, blank=True)
discount = models.IntegerField(default=0, 
        validators=[MinValueValidator(0), MaxValueValidator(100)])
```

这些字段允许我们存储一个可选的订单使用的优惠券和优惠券的折扣。折扣存储在关联的`Coupon`对象中，但我们在`Order`模型中包括它，以便优惠券被修改或删除后还能保存。

因为修改了`Order`模型，所以我们需要创建一个数据库迁移。在命令行中执行以下命令：

```py
python manage.py makemigrations
```

你会看到类似这样的输出：

```py
Migrations for 'orders':
  orders/migrations/0002_auto_20170515_0731.py
    - Add field coupon to order
    - Add field discount to order
```

执行以下命令同步数据库迁移：

```py
python manage.py migrate orders
```

你会看到新的数据库迁移已经生效，现在`Order`模型的字段修改已经同步到数据库中。

回到`models.py`文件，修改`Order`模型的`get_total_cost()`方法：

```py
def get_total_cost(self):
	total_cost = sum(item.get_cost() for item in self.items.all())
	return total_cost - total_cost * self.discount / Decimal('100')
```

如果存在优惠券，`Order`模型的`get_total_cost()`方法会计算优惠券的折扣。

编辑`orders`应用的`views.py`文件，修改其中的`order_create`视图，当创建新订单时，保存关联的优惠券。找到这一行代码：

```py
order = form.save()
```

替换为以下代码：

```py
order = form.save(commit=False)
if cart.coupon:
    order.coupon = cart.coupon
    order.discount = cart.coupon.discount
order.save()
```

在新代码中，我们用`OrderCreateForm`表单的`save()`方法创建了一个`Order`对象，并用`commit=False`避免保存到数据库中。如果购物车中包括优惠券，则存储使用的关联优惠券和折扣。然后我们把`order`对象存储到数据库中。

执行`python manage.py runserver`命令启动开发服务器，并使用`./ngrok http 8000`命令启动 Ngrok。

在浏览器中打开 Ngrok 提供的 URL，并使用你创建的优惠券完成一次购物。当你成功完成一次购物，你可以访问`http://127.0.0.1:8000/admin/orders/order/`，检查订单是否包括优惠券和折扣，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE9.4.png)

你还可以修改管理订单详情模板和订单 PDF 账单，用跟购物车同样的方式显示使用的优惠券。

接下来，我们要为项目添加国际化。

## 9.2 添加国际化和本地化

Django 提供了完整的国际化和本地化支持。它允许你把应用翻译为多种语言，它会处理特定区域日期，时间，数字和时区。让我们弄清楚国际化和本地化的区别。国际化（通常缩写为 i18n）是让软件适用于潜在的不同语言和地区的过程，让软件不会硬编码为特定语言和地区。本地化（缩写为 l10n）是实际翻译软件和适应特定地区的过程。使用 Django 自己的国际化框架，它本身被翻译为超过 50 中语言。

### 9.2.1 使用 Django 国际化

国际化框架让你很容易的在 Python 代码和模板中编辑需要翻译的字符串。它依赖 GNU gettext 工具集生成和管理信息文件。一个信息文件是一个代表一种语言的普通文本文件。它包括你应用中的部分或全部需要翻译的字符串，以及相应的单种语言的翻译。信息文件的扩展名是`.po`。

一旦完成翻译，信息文件就会被编译，用来快速访问翻译后的字符串。被编译的翻译文件扩展名是`.mo`。

#### 9.2.1.1 国际化和本地化设置

Django 为国际化提供了一些设置。以下是最相关的设置：

- `USE_I18N`：指定 Django 的翻译系统是否可用的布尔值。默认为`True`。
- `USE_L10N`：表示本地格式是否可用的布尔值。可用时，用本地格式表示日期和数字。默认为`False`。
- `USE_TZ`：指定日期和时间是否时区感知的布尔值。当你用`startproject`创建项目时，该值设置为`True`。
- `LANGUAGE_CODE`：项目的默认语言代码。它使用标准的语言 ID 格式，比如`en-us`表示美式英语，`en-gb`表示英式英语。这个设置需要`USE_I18N`设为`True`才生效。你可以在[这里](http://www.i18nguy.com/unicode/language-identifiers.html)查看有效地语言 ID 列表。
- `LANGUAGES`：一个包括项目可用语言的元组。它由包括语言代码和语言名称的双元组构成。你可以在`django.conf.global_settings`中查看可用语言列表。当你选择你的网站将使用哪些语言时，你可以设置`LANGUAGES`为这个列表的一个子集。
- `LOCALE_PATHS`：Django 查找项目中包括翻译的信息文件的目录列表。
- `TIME_ZONE`：表示项目时区的字符串。当你使用`startproject`命令创建新项目时，它设置为`UTC`。你可以设置它为任何时区，比如`Europe/Madrid`。

这是一些可用的国际化和本地化设置。你可以在[这里](https://docs.djangoproject.com/en/1.11/ref/settings/#globalization-i18n-l10n)查看完整列表。

#### 9.2.1.2 国际化管理命令

使用`manage.py`或者`django-admin`工具管理翻译时，Django 包括以下命令：

- `makemessages`：它在源代码树上运行，查找所有标记为需要翻译的字符串，并在`locale`目录中创建或更新`.po`信息文件。每种语言创建一个`.po`文件。
- `compilemessages`：编译存在的`.po`信息文件为`.mo`文件，用于检索翻译。

你需要`gettext`工具集创建，更新和编译信息文件。大部分 Linux 发行版都包括了`gettext`工具集。如果你使用的是 Mac OS X，最简单的方式是用`brew install gettext`命令安装。你可能还需要用`brew link gettext --force`强制链接到它。对于 Windows 安装，请参考[这里](https://docs.djangoproject.com/en/1.11/topics/i18n/translation/#gettext-on-windows)的步骤。

#### 9.2.1.3 如果在 Django 项目中添加翻译

让我们看下国际化我们项目的流程。我们需要完成以下工作：

1. 我们标记 Python 代码和目录中需要编译的字符串。
2. 我们运行`makemessages`命令创建或更新信息文件，其中包括了代码中所有需要翻译的字符串。
3. 我们翻译信息文件中的字符串，然后用`compilemessages`管理命令编辑它们。

#### 9.2.1.4 Django 如何决定当前语言

Django 自带一个中间件，它基于请求的数据决定当前语言。位于`django.middleware.locale.LocaleMiddleware`的`LocaleMiddleware`中间件执行以下任务：

1. 如果你使用`i18_patterns`，也就是你使用翻译后的 URL 模式，它会在被请求的 URL 中查找语言前缀，来决定当前语言。
2. 如果没有找到语言前缀，它会在当前用户会话中查询`LANGUAGE_SESSION_KEY`。
3. 如果没有在会话中设置语言，它会查找带当前语言的 cookie。这个自定义的 cookie 名由`LANGUAGE_COOKIE_NAME`设置提供。默认情况下，该 cookie 名为`django-language`。
4. 如果没有找到 cookie，它会查询请求的`Accept-Language`头。
5. 如果`Accept-Language`头没有指定语言，Django 会使用`LANGUAGE_CODE`设置中定义的语言。

默认情况下，Django 会使用`LANGUAGE_CODE`设置中定义的语言，除非你使用`LocaleMiddleware`。以上描述的过程只适用于使用这个中间件。

### 9.2.2 为国际化我们的项目做准备

让我们为我们的项目使用不同语言。我们将创建商店的英语和西拔牙语版本。编辑项目的`settings.py`文件，在`LANGUAGE_CODE`设置之后添加`LANGUAGES`设置：

```py
LANGUAGES = (
    ('en', 'English'),
    ('es', 'Spanish'),
)
```

`LANGUAGES`设置中包括两个元组，每个元组包括语言代码和名称。语言代码可以指定地区，比如`en-us`或`en-gb`，也可以通用，比如`en`。在这个设置中，我们指定我们的应用只对英语和西班牙可用。如果我们没有定义`LANGUAGES`设置，则网站对于 Django 的所有翻译语言都可用。

如下修改`LANGUAGE_CODE`设置：

```py
LANGUAGE_CODE = 'en'
```

在`MIDDLEWARE`设置中添加`django.middleware.locale.LocaleMiddleware`。确保这个中间件在`SessionMiddleware`之后，因为`LocaleMiddleware`需要使用会话数据。它还需要在`CommonMiddleware`之前，因为后者需要一个激活的语言解析请求的 URL。`MIDDLEWARE`设置看起来是这样的：

```py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',
    'django.middleware.common.CommonMiddleware',
    # ...
]
```

> 中间件的顺序很重要，因为每个中间件都依赖前面其它中间件执行后的数据集。中间件按`MIDDLEWARE`中出现的顺序应用在请求上，并且反序应用在响应上。

在项目主目录中穿件以下目录结构，与`manage.py`同级：

```py
locale/
	en/
	es/
```

`locale`目录是应用的信息文件存储的目录。再次编辑`settings.py`文件，在其中添加以下设置：

```py
LOCALE_PATHS = (
    os.path.join(BASE_DIR, 'locale/'),
)
```

`LOCALE_PATHS`设置指定了 Django 查找翻译文件的目录。最先出现的路径优先级最高。

当你在项目目录中使用`makemessages`命令时，信息文件会在我们创建的`locale/`路径中生成。但是，对于包括`locale/`目录的应用来说，信息文件会在这个应用的`locale/`目录中生成。

### 9.2.3 翻译 Python 代码

要翻译 Python 代码中的字面量，你可以用`django.utils.translation`中的`gettext()`函数标记要翻译的字符串。这个函数翻译信息并返回一个字符串。惯例是把这个函数导入为短别名`_`。

你可以在[这里](https://docs.djangoproject.com/en/1.11/topics/i18n/translation/)查看所有关于翻译的文档。

#### 9.2.3.1 标准翻译

以下代码展示了如何标记一个需要翻译的字符串：

```py
from django.utils.translation import gettext as _
output = _('Text to be translated.')
```

#### 9.2.3.2 惰性翻译

Django 的所有翻译函数都包括惰性（lazy）版本，它们的后缀都是`_lazy()`。使用惰性函数时，当值被访问时翻译字符串，而不是惰性函数被调用时翻译（这就是为什么它们被惰性翻译）。当标记为翻译的字符串位于加载模式时执行的路径中，这些惰性翻译函数非常方便。

> 使用`gettext_lazy()`代替`gettext()`时，当值被访问时翻译字符串，而不是翻译函数调用时翻译。Django 为所有翻译函数提供了惰性版本。

#### 9.2.3.3 带变量的翻译

标记为翻译的字符串的字符串可以包括占位符来在翻译中引入变量。以下代码是翻译带占位符的字符串：

```py
from django.utils.translation import gettext as _
month = _('April')
day = '14'
output = _('Today is %(month)s %(day)s') % {'month': month, 'day': day}
```

通过使用占位符，你可以重新排列文本变量。比如，上个例子中的英文防疫可能是`Today is April 14`，而西拔牙语翻译时`Hoy es 14 de Abiril`。当需要翻译的字符串中包括一个以上的参数时，总是使用字符串插值代替位置插值。这样就可以重新排列站位文本。

#### 9.2.3.4 翻译中的复数形式

对于复数形式，你可以使用`ngettext()`和`ngettext_lazy()`。这些函数根据一个表示对象数量的参数翻译单数和复数形式。下面的代码展示了如何使用它们：

```py
output = ngettext('there is %(count)d product',
					'there are %(count)d products',
					count) % {'count': count}
```

现在你已经学会了翻译 Python 代码中字面量的基础，是时候翻译我们的项目了。

#### 9.2.3.5 翻译你的代码

编辑项目的`settings.py`文件，导入`gettext_lazy()`函数，并如下修改`LANGUAGES`设置来翻译语言名称：

```py
from django.utils.translation import gettext_lazy as _

LANGUAGES = (
    ('en', _('English')),
    ('es', _('Spanish')),
)
```

我们在这里使用`gettext_lazy()`函数代替`gettext()`，来避免循环导入，所以当语言名称被访问时翻译它们。

打开终端，并在项目目录中执行以下命令：

```py
django-admin makemessages --all
```

你会看到以下输出：

```py
processing locale en
processing locale es
```

看一眼`locale/`目录，你会看这样的文件结构：

```py
en/
	LC_MESSAGES/
		django.po
es/
	LC_MESSAGES/
		django.po
```

为每种语言创建了一个`.po`信息文件。用文本编辑器打开`es/LC_MESSAGES/django.po`文件。在文件结尾，你会看到以下内容：

```py
#: myshop/settings.py:122
msgid "English"
msgstr ""

#: myshop/settings.py:123
msgid "Spanish"
msgstr ""
```

每个需要翻译的字符串前面都有一条注释，显示它位于的文件和行数。每个翻译包括两个字符串：

- `msgid`：源代码中需要翻译的字符串。
- `msgstr`：对应语言的翻译，默认为空。你需要在这里输入给定字符串的实际翻译。

为给的`msgid`字符串填入`msgstr`翻译：

```py
#: myshop/settings.py:122
msgid "English"
msgstr "Inglés"

#: myshop/settings.py:123
msgid "Spanish"
msgstr "Español"
```

保存修改的信息文件，打开终端，执行以下命令：

```py
django-admin compilemessages
```

如果一切顺利，你会看到类似这样的输出：

```py
processing file django.po in /Users/lakerszhy/Documents/GitHub/Django-By-Example/code/Chapter 9/myshop/locale/en/LC_MESSAGES
processing file django.po in /Users/lakerszhy/Documents/GitHub/Django-By-Example/code/Chapter 9/myshop/locale/es/LC_MESSAGES
```

输出告诉你信息文件已经编译。再看一眼`myshop`项目的`locale`目录，你会看到以下文件：

```py
en/
	LC_MESSAGES/
		django.mo
		django.po
es/
	LC_MESSAGES/
		django.mo
		django.po
```

你会看到为每种语言生成了一个编译后的`.mo`信息文件。

我们已经翻译了语言名本身。现在让我们翻译在网站中显示的模型字段名。编辑`orders`应用的`models.py`文件，为`Order`模型字段添加需要翻译的名称标记：

```py
from django.utils.translation import gettext_lazy as _

class Order(models.Model):
    first_name = models.CharField(_('first name'), max_length=50)
    last_name = models.CharField(_('last name'), max_length=50)
    email = models.EmailField(_('email'))
    address = models.CharField(_('address'), max_length=250)
    postal_code = models.CharField(_('postal code'), max_length=20)
    city = models.CharField(_('city'), max_length=100)
    # ...
```

我们为用户下单时显示的字段添加了名称，分别是`first_name`，`last_name`，`email`，`address`，`postal_code`和`city`。记住，你也可以使用`verbose_name`属性为字段命名。

在`orders`应用中创建以下目录结构：

```py
locale/
	en/
	es/
```

通过创建`locale/`目录，这个应用中需要翻译的字符串会存储在这个目录的信息文件中，而不是主信息文件。通过这种方式，你可以为每个应用生成独立的翻译文件。

在项目目录打开终端，执行以下命令：

```py
django-admi makemessages --all
```

你会看到以下输出：

```py
processing locale en
processing locale es
```

用文本编辑器开大`es/LC_MESSAGES/django.po`文件。你会看到`Order`模型需要翻译的字符串。为给的`msgid`字符串填入`msgstr`翻译：

```py
#: orders/models.py:10
msgid "first name"
msgstr "nombre"

#: orders/models.py:11
msgid "last name"
msgstr "apellidos"

#: orders/models.py:12
msgid "email"
msgstr "e-mail"

#: orders/models.py:13
msgid "address"
msgstr "dirección"

#: orders/models.py:14
msgid "postal code"
msgstr "código postal"

#: orders/models.py:15
msgid "city"
msgstr "ciudad"
```

填完之后保存文件。

除了文本编辑器，你还可以使用 Poedit 编辑翻译。Poedit 是一个编辑翻译的软件，它使用`gettext`。它有 Linux，Windows 和 Mac OS X 版本。你可以在[这里](http://poedit.net/)下载。

让我们再翻译项目中的表单。`orders`应用的`OrderCreateForm`不需要翻译，因为它是一个`ModelForm`，它的表单字段标签使用了`Order`模型字段的`verbose_name`属性。我们将翻译`cart`和`coupons`应用的表单。

编辑`cart`应用中的`forms.py`文件，为`CartAddProductForm`的`quantity`字段添加一个`lable`属性，然后标记为需要翻译：

```py
from django import forms
from django.utils.translation import gettext_lazy as _

PRODUCT_QUANTITY_CHOICES = [(i, str(i)) for i in range(1, 21)]

class CartAddProductForm(forms.Form):
    quantity = forms.TypedChoiceField(
        choices=PRODUCT_QUANTITY_CHOICES, 
        coerce=int,
        label=_('Quantity'))
    update = forms.BooleanField(
        required=False, 
        initial=False, 
        widget=forms.HiddenInput)
```

编辑`coupons`应用的`forms.py`文件，如下翻译`CouponApplyForm`表单：

```py
from django import forms
from django.utils.translation import gettext_lazy as _

class CouponApplyForm(forms.Form):
    code = forms.CharField(label=_('Coupon'))
```

我们为`code`字段添加了`label`属性，并标记为需要翻译。

### 9.2.4 翻译模板

Django 为翻译模板中的字符串提供了`{% trans %}`和`{% blocktrans %}`模板标签。要使用翻译模板标签，你必须在模板开头添加`{% load i18n %}`加载它们。

#### 9.2.4.1 模板标签{% trans %}

`{% trans %}`模板标签允许你标记需要翻译的字符串，常量或者变量。在内部，Django 在给定的文本上执行`gettext()`。以下是在模板中标记需要翻译的字符串：

```py
{% trans "Text to be translated" %}
```

你可以使用`as`在变量中存储翻译后的内容，然后就能在整个模板中使用这个变量。下面这个例子在`greeting`变量中存储翻译后的文本：

```py
{% trans "Hello!" as greeting %}
<h1>{{ greeting }}</h1>
```

`{% trans %}`标签对简单的翻译字符串很有用，但它不能处理包括变量的翻译内容。

#### 9.2.4.2 模板标签{% blocktrans %}

`{% blocktrans %}`模板标签允许你标记包括字面量的内容和使用占位符的变量内容。下面这个例子展示了如何使用`{% blocktrans %}`标签标记一个包括`name`变量的翻译内容：

```py
{% blocktrans %}Hello {{ name }}!{% endblocktrans %}
```

你可以使用`with`引入模板表达式，比如访问对象属性，或者对变量使用模板过滤器。这时，你必须总是使用占位符。你不能在`blocktrans`块中访问表达式或者对象属性。下面的例子展示了如何使用`with`，其中引入了一个对象属性，并使用`capfirst`过滤器：

```py
{% blocktrans with name=user.name|capfirst %}
	Hello {{ name }}!
{% endblocktrans %}
```

> 当需要翻译的字符串中包括变量内容时，使用`{% blocktrans %}`代替`{% trans %}`。

#### 9.2.4.3 翻译商店的模板

编辑`shop`应用的`shop/base.html`模板。在模板开头加载`i18n`标签，并标记需要翻译的字符串：

```py
{% load i18n %}
{% load static %}
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>
        {% block title %}{% trans "My shop" %}{% endblock %}
    </title>
    <link href="{% static "css/base.css" %}" rel="stylesheet">
</head>
<body>
    <div id="header">
        <a href="/" class="logo">{% trans "My shop" %}</a>
    </div>
    <div id="subheader">
        <div class="cart">
            {% with total_items=cart|length %}
                {% if cart|length > 0 %}
                    {% trans "Your cart:" %}
                    <a href="{% url "cart:cart_detail" %}">
                        {% blocktrans with total_items_plural=otal_items|pluralize total_price=cart.get_total_price %}
                            {{ total_items }} item{{ total_items_plural }},
                            ${{ total_price }}
                        {% endblocktrans %}
                    </a>
                {% else %}
                    {% trans "Your cart is empty." %}
                {% endif %}
            {% endwith %}
        </div>
    </div>
    <div id="content">
        {% block content %}
        {% endblock %}
    </div>
</body>
</html>
```

记住，我们用`{% blocktrans %}`标签显示购物车汇总。之前购物车汇总是这样的：

```py
{{ total_items }} item{{ total_items|pluralize }},
${{ cart.get_total_price }}
```

我们利用`{% blocktrans with ... %}`为`total_items|pluralize`（在这里使用模板标签）和`cart.get_total_price`（在这里访问对象方法）使用占位符，结果是：

```py
{% blocktrans with total_items_plural=otal_items|pluralize total_price=cart.get_total_price %}
	{{ total_items }} item{{ total_items_plural }},
   ${{ total_price }}
{% endblocktrans %}
```

接着编辑`shop`应用的`shop/product/detai.html`模板，在`{% extends %}`标签（它必须总是第一个标签）之后加载`i18n`标签：

```py
{% load i18n %}
```

然后找到这一行：

```py
<input type="submit" value="Add to cart">
```

替换为：

```py
<input type="submit" value="{% trans "Add to cart" %}">
```

现在翻译`orders`应用的模板。编辑`orders`应用的`orders/order/create.html`模板，如下标记需要翻译的文本：

```py
{% extends "shop/base.html" %}
{% load i18n %}

{% block title %}
    {% trans "Checkout" %}
{% endblock title %}

{% block content %}
    <h1>{% trans "Checkout" %}</h1>

    <div class="order-info">
        <h3>{% trans "Your order" %}</h3>
        <ul>
            {% for item in cart %}
                <li>
                    {{ item.quantity }}x {{ item.product.name }}
                    <span>${{ item.total_price }}</span>
                </li>
            {% endfor %}
            {% if cart.coupon %}
                <li>
                    {% blocktrans with code=cart.coupon.code discount=cart.coupon.discount %}
                        "{{ code }}" ({{ discount }}% off)
                    {% endblocktrans %}
                    <span>- ${{ cart.get_discount|floatformat:"2" }}</span>
                </li>
            {% endif %}
        </ul>
        <p>{% trans "Total" %}: ${{ cart.get_total_price_after_discount|floatformat:"2" }}</p>
    </div>

    <form action="." method="post" class="order-form">
        {{ form.as_p }}
        <p><input type="submit" value="{% trans "Place order" %}"></p>
        {% csrf_token %}
    </form>
{% endblock content %}
```

在本章示例代码中查看以下文件是如何标记需要翻译的字符串：

- `shop`应用：`shop/product/list.hmtl`模板
- `orders`应用：`orders/order/created.html`模板
- `cart`应用：`cart/detail.html`模板

让我们更新信息文件来引入新的翻译字符串。打开终端，执行以下命令：

```py
django-admin makemessages --all
```

`.po`翻译文件位于`myshop`项目的`locale`目录中，`orders`应用现在包括了所有我们标记过的翻译字符串。

编辑项目和`orders`应用的`.po`翻译文件，并填写西班牙语翻译。你可以参考本章示例代码的`.po`文件。

从项目目录中打开终端，并执行以下命令：

```py
cd orders/
django-admin compilemessages
cd ../
```

我们已经编译了`orders`应用的翻译文件。

执行以下命令，在项目的信息文件中包括没有`locale`目录的应用的翻译。

```py
django-admin compilemessage
```

### 9.2.5 使用 Rosetta 的翻译界面

Rosetta 是一个第三方应用，让你可以用跟 Django 管理站点一样的界面编辑翻译。Rosetta 可以很容易的编辑`.po`文件，并且它会更新编译后的编译文件。让我们把它添加到项目中。

使用`pip`命令安装 Rosetta：

```py
pip install django-rosetta
```

然后把`rosetta`添加到项目`settings.py`文件中的`INSTALLED_APP`设置中：

你需要把 Rosetta 的 URL 添加到主 URL 配置中。编辑项目的`urls.py`文件，并添加下 URL 模式：

```py
url(r'^rosetta/', include('rosetta.urls')),
```

确保把它放在`shop.urls`模式之后，避免错误的匹配。

在浏览器中打开`http://127.0.0.1:8000/admin/`，并用超级用户登录。然后导航到`http://127.0.0.1:8000/rosetta/`。你会看到已经存在的语言列表，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE9.5.png)

点击`Filter`中的`All`显示所有可用的信息文件，包括属于`orders`应用的信息文件。在`Spanish`中点击`Myshop`链接来编辑西班牙语翻译。你会看到一个需要翻译的字符串列表，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE9.6.png)

你可以在`Spanish`列中输入翻译。`Occurrences`列显示每个需要翻译的字符串所在的文件和行数。

包括占位符的翻译是这样的：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE9.7.png)

Rosetta 用不同的颜色显示占位符。当你翻译内容时，确保不要翻译占位符。比如这一行字符串：

```py
%(total_items)s item%(total_items_plural)s, $%(total_price)s
```

翻译为西班牙语后是这样的：

```py
%(total_items)s producto%(total_items_plural)s, $%(total_price)s
```

你可以参考本章的示例代码，用同样的西班牙语翻译你的项目。

当你完成翻译后，点击`Save and translate next block`按钮，把翻译保存到`.po`文件。保存翻译时，Rosseta 会编译信息文件，所以不需要执行`compilemessages`命令。但是 Rosetta 需要写入`locale`目录的权限来写入信息文件。确保这些目录有合理的权利。

如果你希望其他用户也可以编辑翻译，在浏览器中打开`http://127.0.0.1:8000/admin/auth/group/add/`，并创建一个`translators`组。然后访问`http://127.0.0.1:8000/admin/auth/user/`，编辑你要授予翻译权限的用户。编辑用户时，在`Premissions`中，把`translators`组添加到每个用户的`Chosen Groups`中。Resetta 只对超级用户和属于`translators`组的用户可用。

你可以在[这里](http://django-rosetta.readthedocs.io/en/latest/)阅读 Rosetta 的文档。

> 当你在生产环境添加新翻译时，如果你的 Django 运行在一个真实的 web 服务器上，你必须在执行`compilemessages`命令或者用 Rosetta 保存翻译之后重启服务器，才能让修改生效。

### 9.2.6 不明确的翻译

你可能已经注意到了，在 Rosetta 中有一个`Fuzzy`列。这不是 Rosetta 的特征，而是有`gettext`提供的。如果启用了翻译的`fuzzy`标记，那么它就不会包括在编译后的信息文件中。这个标记用于需要翻译者修改的翻译字符串。当用新的翻译字符串更新`.po`文件中，可能有些翻译字符串自动标记为`fuzzy`。当`gettext`发现某些`msgid`变动不大时，它会匹配为旧的翻译，并标记为`fuzzy`，以便复核。翻译者应该复核不明确的翻译，然后移除`fuzzy`标记，并在此编译信息文件。

### 9.2.7 URL 模式的国际化

Django 为 URL 提供了国际化功能。它包括两种主要的国际化 URL 特性：

- **URL 模式中的语言前缀：**把语言前缀添加到 URL 中，在不同的基础 URL 下提供每种语言的版本
- **翻译后的 URL 模式：**标记需要翻译的 URL 模式，因此同一个 URL 对于每种语言是不同的

翻译 URL 的其中一个原因是为搜索引擎优化你的网站。通过在模式中添加语言前缀，你就可以为每种语言提供索引 URL，而不是为所有语言提供一个索引 URL。此外，通过翻译 URL 为不同语言，你可以为搜索引擎提供对每种语言排名更好的 URL。

#### 9.2.7.1 添加语言前缀到 URL 模式中

Django 允许你在 URL 模式中添加语言前缀。例如，网站的英语版本可以`/en/`起始路径下，而西班牙语版本在`/es/`下。

要在 URL 模式中使用语言，你需要确保`settings.py`文件的`MIDDLEWARE`设置中包括`django.middleware.locale.LocaleMiddleware`。Django 将用它从请求 URL 中识别当前语言。

让我们在 URL 模式中添加语言前缀。编辑`myshop`项目的`urls.py`文件，添加以下导入：

```py
from django.conf.urls.i18n import i18n_patterns
```

然后添加`i18n_patterns()`，如下所示：

```py
urlpatterns = i18n_patterns(
    url(r'^admin/', admin.site.urls),
    url(r'^cart/', include('cart.urls', namespace='cart')),
    url(r'^orders/', include('orders.urls', namespace='orders')),
    url(r'^paypal/', include('paypal.standard.ipn.urls')),
    url(r'^payment/', include('payment.urls', namespace='payment')),
    url(r'^coupons/', include('coupons.urls', namespace='coupons')),
    url(r'^rosetta/', include('rosetta.urls')),
    url(r'^', include('shop.urls', namespace='shop')),
)
```

你可以在`patterns()`和`i18n_patterns()`中结合 URL 模式，这样有些模式包括语言前缀，有些不包括。但是，最好只使用翻译后的 URL，避免不小心把翻译后的 URL 匹配到没有翻译的 URL 模式。

启动开发服务器，并在浏览器中打开`http://127.0.0.1:8000/`。因为你使用了`LocaleMiddleware`中间件，所以 Django 会执行`Django 如何决定当前语言`中描述的步骤，决定当前的语言，然后重定义到包括语言前缀的同一个 URL。看一下眼浏览器中的 URL，它应该是`http://127.0.0.1:8000/en/`。如果浏览器的`Accept-Language`头是西班牙语或者英语，则当前语言是它们之一；否则当前语言是设置中定义的默认`LANGUAGE_CODE`(英语）。

#### 9.2.7.2 翻译 URL 模式

Django 支持 URL 模式中有翻译后的字符串。对应单个 URL 模式，你可以为每种语言使用不同的翻译。你可以标记需要翻译的 URL 模式，方式与标记字面量一样，使用`gettext_lazy()`函数。

编辑`myshop`项目的主`urls.py`文件，把翻译字符串添加到`cart`，`orders`，`payment`和`coupons`应用的 URL 模式的正则表达式中：

```py
urlpatterns = i18n_patterns(
    url(r'^admin/', admin.site.urls),
    url(_(r'^cart/'), include('cart.urls', namespace='cart')),
    url(_(r'^orders/'), include('orders.urls', namespace='orders')),
    url(r'^paypal/', include('paypal.standard.ipn.urls')),
    url(_(r'^payment/'), include('payment.urls', namespace='payment')),
    url(_(r'^coupons/'), include('coupons.urls', namespace='coupons')),
    url(r'^rosetta/', include('rosetta.urls')),
    url(r'^', include('shop.urls', namespace='shop')),
)
```

编辑`orders`应用的`urls.py`文件，编辑需要翻译的 URL 模式：

```py
from django.utils.translation import gettext_lazy as _

urlpatterns = [
    url(_(r'^create/$'), views.order_create, name='order_create'),
    # ..
]
```

编辑`payment`应用的`urls.py`文件，如下修改代码：

```py
from django.utils.translation import gettext as _

urlpatterns = [
    url(_(r'^process/$'), views.payment_process, name='process'),
    url(_(r'^done/$'), views.payment_done, name='done'),
    url(_(r'^canceled/$'), views.payment_canceled, name='canceled'),
]
```

我们不要翻译`shop`应用的 URL 模式，因为它们由变量构建，不包括任何字面量。

打开终端，执行以下命令更新信息文件：

```py
django-admin makemessages --all
```

确保开发服务器正在运行。在浏览器中打开`http://127.0.0.1:8000/en/rosetta/`，然后点击`Spanish`中的`Myshop`链接。你可以使用`Display`过滤器只显示没有翻译的字符串。在 URL 翻译中，一定要保留正则表达式中的特殊字符。翻译 URL 是一个精细的任务；如果你修改了正则表达式，就会破坏 URL。

### 9.2.8 允许用户切换语言

因为我们现在提供了多种语言，所以我们应该让用户可以切换网站的语言。我们会在网站中添加一个语言选择器。语言选择器用链接显示可用的语言列表。

编辑`shop/base.html`模板，找到以下代码：

```py
<div id="header">
    <a href="/" class="logo">{% trans "My shop" %}</a>
</div>
```

替换为以下代码：

```py
<div id="header">
    <a href="/" class="logo">{% trans "My shop" %}</a>

    {% get_current_language as LANGUAGE_CODE %}
    {% get_available_languages as LANGUAGES %}
    {% get_language_info_list for LANGUAGES as languages %}
    <div class="languages">
        <p>{% trans "Languages" %}:</p>
        <ul class="languages">
            {% for language in languages %}
                <li>
                    <a href="/{{ language.code }}" {% if language.code == LANGUAGE_CODE %} class="selected"{% endif %}>
                        {{ language.name_local }}
                    </a>
                </li>
            {% endfor %}
        </ul>
    </div>
</div>
```

我们是这样构建语言选择器的：

1. 我们首先用`{% load i18n %}`加载国际化标签。
2. 我们用`{% get_current_language %}`标签查询当前语言。
3. 我们用`{% get_available_languages %}`模板标签获得`LANGUAGES`设置中定义的语言。
4. 我们用`{% get_language_info_list %}`标签提供访问语言属性的便捷方式。
5. 我们构建 HTML 列表显示所有可用的语言，并在当前激活语言上添加`selected`类属性。

我们用`i18n`提供的模板标签，根据项目设置提供可用的语言。现在打开`http://127.0.0.1:8000/`。你会看到网站右上角有语言选择器，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE9.8.png)

用户现在可以很容易的切换语言。

### 9.2.9 用 django-parler 翻译模型

Django 没有为翻译模型提供好的解决方案。你必须实现自己的解决方案来管理不同语言的内容，或者使用第三方模块翻译模型。有一些第三方应用允许你翻译模型字段。每种采用不同的方法存储和访问翻译。其中一个是`django-parler`。这个模块提供了一种非常高效的翻译模型的方式，并且它和 Django 管理站点集成的非常好。

`django-parler`为每个模型生成包括翻译的独立的数据库表。这张表包括所有翻译后的字段，以及一个翻译所属的原对象的外键。因为每行存储单个语言的内容，所以它还包括一个语言字段。

#### 9.2.9.1 安装 django-parler

使用`pip`命令安装`django-parler`：

```py
pip install django-parler
```

然后编辑项目的`settings.py`文件，把`parler`添加到`INSTALLED_APPS`设置中。并在设置文件中添加以下代码：

```py
PARLER_LANGUAGES = {
    None: (
        {'code': 'en', },
        {'code': 'es', },
    ),
    'default': {
        'fallback': 'en',
        'hide_untranslated': False,
    }
}
```

这个设置定义了 django-parler 的可用语言`en`和`es`。我们指定默认语言是`en`，并且指定 django-parler 不隐藏没有翻译的内容。

#### 9.2.9.2 翻译模型字段

让我们为商品目录添加翻译。django-parler 提供了一个`TranslatableModel`模型类和一个`TranslatedFields`包装器（wrapper）来翻译模型字段。编辑`shop`应用的`models.py`文件，添加以下导入：

```py
from parler.models import TranslatableModel, TranslatedFields
```

然后修改`Category`模型，让`name`和`slug`字段可翻译。我们现在还保留非翻译字段：

```py
class Category(TranslatableModel):
    name = models.CharField(max_length=200, db_index=True)
    slug = models.SlugField(max_length=200, db_index=True, unique=True)

    translations = TranslatedFields(
        name = models.CharField(max_length=200, db_index=True),
        slug = models.SlugField(max_length=200, db_index=True, unique=True)
    )
```

现在`Category`模型继承自`TranslatableModel`，而不是`models.Model`。并且`name`和`slug`字段都包括在`TranslatedFields`包装器中。

编辑`Product`模型，为`name`，`slug`和`description`字段添加翻译。同样保留非翻译字段：

```py
class Product(TranslatableModel):
    name = models.CharField(max_length=200, db_index=True)
    slug = models.SlugField(max_length=200, db_index=True)
    description = models.TextField(blank=True)
    translations = TranslatedFields(
        name = models.CharField(max_length=200, db_index=True),
        slug = models.SlugField(max_length=200, db_index=True),
        description = models.TextField(blank=True)
    )
    category = models.ForeignKey(Category, related_name='products')
    image = models.ImageField(upload_to='products/%Y/%m/%d', blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.PositiveIntegerField()
    available = models.BooleanField(default=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
```

django-parler 为每个`TranslatableModel`模型生成另一个模型。图 9.9 中，你可以看到`Product`模型的字段和生成的`ProductTranslation`模型：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE9.9.png)

django-parler 生成的`ProductTranslation`模型包括`name`，`slug`和`description`可翻译字段，一个`language_code`字段，以及指向`Product`对象的外键`master`字段。从`Product`到`ProductTranslation`是一对多的关系。每个`Product`对象会为每种语言生成一个`ProductTranslation`对象。

因为 Django 为翻译使用了单独的数据库表，所以有些 Django 特性不能使用了。一个翻译后的字段不能用作默认的排序。你可以在查询中用翻译后的字段过滤，但你不能再`ordering`元选项中包括翻译后的字段。编辑`shop`应用的`models.py`文件，注释`Category`类中`Meta`类的`ordering`属性：

```py
class Meta:
    # ordering = ('name', )
    verbose_name = 'category'
    verbose_name_plural = 'categories'
```

我们还必须注释`Product`类中`Meta`类的`index_together`属性，因为当前 django-parler 版本不提供验证它的支持：

```py
class Meta:
    ordering = ('-created', )
    # index_together = (('id', 'slug'), )
```

你可以在[这里](http://django-parler.readthedocs.org/en/latest/compatibility.html)阅读更多关于 django-parler 和 Django 兼容性的信息。

#### 9.2.9.3 创建自定义数据库迁移

当你为翻译创建了新模型，你需要执行`makemigrations`命令为模型生成数据库迁移，然后同步到数据库中。但是当你将已存在字段变为可翻译后，你的数据库中可能已经存在数据了。我们将把当前数据迁移到新的翻译模型中。因此，我们添加了翻译后的字段，但暂时保留了原来的字段。

为已存在字段添加翻译的流程是这样的：

1. 我们为新的可翻译模型字段创建数据库迁移，并保留原来的字段。
2. 我们构建一个自定义数据库迁移，从已存在字段中拷贝数据到翻译模型中。
3. 我们从原来的模型中移除已存在的字段。

执行以下命令，为添加到`Category`和`Product`模型中的翻译字段创建数据库迁移：

```py
python manage.py makemigrations shop --name "add_translation_model"
```

你会看到以下输出：

```py
Migrations for 'shop':
  shop/migrations/0002_add_translation_model.py
    - Create model CategoryTranslation
    - Create model ProductTranslation
    - Change Meta options on category
    - Alter index_together for product (0 constraint(s))
    - Add field master to producttranslation
    - Add field master to categorytranslation
    - Alter unique_together for producttranslation (1 constraint(s))
    - Alter unique_together for categorytranslation (1 constraint(s))
```

现在我们需要创建一个自定义数据库迁移，把已存在的数据拷贝到新的翻译模型中。使用以下命令创建一个空的数据库迁移：

```py
python manage.py makemigrations --empty shop --name "migrate_translatable_fields"
```

你会看到以下输出：

```py
Migrations for 'shop':
  shop/migrations/0003_migrate_translatable_fields.py
```

编辑`shop/migrations/0003_migrate_translatable_fields.py`文件，并添加以下代码：

```py
# -*- coding: utf-8 -*-
# Generated by Django 1.11.1 on 2017-05-17 01:18
from __future__ import unicode_literals
from django.db import models, migrations
from django.apps import apps
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist

translatable_models = {
    'Category': ['name', 'slug'],
    'Product': ['name', 'slug', 'description'],
}

def forwards_func(apps, schema_editor):
    for model, fields in translatable_models.items():
        Model = apps.get_model('shop', model)
        ModelTranslation = apps.get_model('shop', '{}Translation'.format(model))

        for obj in Model.objects.all():
            translation_fields = {field: getattr(obj, field) for field in fields}
            translation = ModelTranslation.objects.create(
                master_id=obj.pk,
                language_code=settings.LANGUAGE_CODE,
                **translation_fields
            )

def backwards_func(apps, shcema_editor):
    for model, fields in translatable_models.items():
        Model = apps.get_model('shop', model)
        ModelTranslation = apps.get_model('shop', '{}Translation'.format(model))

        for obj in Model.objects.all():
            translation = _get_translation(obj, ModelTranslation)
            for field in fields:
                setattr(obj, field, getattr(translation, field))
            obj.save()

def _get_translation(obj, MyModelTranslation):
    translation = MyModelTranslation.objects.filter(master_id=obj.pk)
    try:
        # Try default translation
        return translation.get(language_code=settings.LANGUAGE_CODE)
    except ObjectDoesNotExist:
        # Hope there is a single translation
        return translations.get()

class Migration(migrations.Migration):

    dependencies = [
        ('shop', '0002_add_translation_model'),
    ]

    operations = [
        migrations.RunPython(forwards_func, backwards_func)
    ]
```

这个迁移包括`forwards_func()`和`backwards_func()`函数，其中包含要执行数据库同步和反转的代码。

迁移流程是这样的：

1. 我们在`translatable_models`字典中定义模型和可翻译的字段。
2. 要同步迁移，我们用`app.get_model()`迭代包括翻译的模型，来获得模型和它可翻译的模型类。
3. 我们迭代数据库中所有存在的对象，并为项目设置中定义的`LANGUAGE_CODE`创建一个翻译对象。我们包括了一个指向原对象的`ForeignKey`，以及从原字段中拷贝的每个可翻译字段。

`backwards_func()`函数执行相反的操作，它查询默认的翻译对象，并把可翻译字段的值拷贝回原对象。

我们已经创建了一个数据库迁移来添加翻译字段，以及一个从已存在字段拷贝内容到新翻译模型的迁移。

最后，我们需要删除不再需要的原字段。编辑`shop`应用的`models.py`文件，移除`Category`模型的`name`和`slug`字段。现在`Category`模型字段是这样的：

```py
class Category(TranslatableModel):
    translations = TranslatedFields(
        name = models.CharField(max_length=200, db_index=True),
        slug = models.SlugField(max_length=200, db_index=True, unique=True)
    )
```

移除`Product`模型的`name`，`slug`和`description`字段。它现在是这样的：

```py
class Product(TranslatableModel):
    translations = TranslatedFields(
        name = models.CharField(max_length=200, db_index=True),
        slug = models.SlugField(max_length=200, db_index=True),
        description = models.TextField(blank=True)
    )
    category = models.ForeignKey(Category, related_name='products')
    image = models.ImageField(upload_to='products/%Y/%m/%d', blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.PositiveIntegerField()
    available = models.BooleanField(default=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
```

现在我们需要创建最后一个迁移，让修改生效。但是，如果我们尝试执行`manage.py`工具，我们会看到一个错误，因为我们还没有让管理站点适配可翻译模型。让我们先修改管理站点。

#### 9.2.9.4 在管理站点集成翻译

Django 管理站点可以很好的跟 django-parler 集成。django-parler 包括一个`TranslatableAdmin`类，它覆写了 Django 提供的`ModelAdmin`类，来管理模型翻译。

编辑`shop`应用的`admin.py`文件，添加以下导入：

```py
from parler.admin import TranslatableAdmin
```

修改`CategoryAdmin`和`ProductAdmin`类，让它们从`TranslatableAdmin`继承。django-parler 还不知道`prepopulated_fields`属性，但它支持相同功能的`get_ prepopulated_fields()`方法。让我们相应的修改，如下所示：

```py
from django.contrib import admin
from parler.admin import TranslatableAdmin
from .models import Category, Product

class CategoryAdmin(TranslatableAdmin):
    list_display = ('name', 'slug')

    def get_prepopulated_fields(self, request, obj=None):
        return {'slug': ('name', )}

admin.site.register(Category, CategoryAdmin)

class ProductAdmin(TranslatableAdmin):
    list_display = ('name', 'slug', 'price', 'stock', 'available', 'created', 'updated')
    list_filter = ('available', 'created', 'updated')
    list_editable = ('price', 'stock', 'available')

    def get_prepopulated_fields(self, request, obj=None):
        return {'slug': ('name', )}
        
admin.site.register(Product, ProductAdmin)
```

我们已经让管理站点可以与新的翻译模型一起工作了。现在可以同步模型修改到数据库中。

#### 9.2.9.5 为模型翻译同步数据库迁移

适配管理站点之前，我们已经从模型中移除了旧的字段。现在我们需要为这个修改创建一个迁移。打开终端执行以下命令：

```py
python manage.py makemigrations shop --name "remove_untranslated_fields"
```

你会看到以下输出：

```py
Migrations for 'shop':
  shop/migrations/0004_remove_untranslated_fields.py
    - Change Meta options on product
    - Remove field name from category
    - Remove field slug from category
    - Remove field description from product
    - Remove field name from product
    - Remove field slug from product
```

通过这次迁移，我们移除了原字段，保留了可翻译字段。

总结一下，我们已经创建了以下迁移：

1. 添加可翻译字段到模型中
2. 从原字段迁移已存在字段到可翻译字段
3. 从模型中移除原字段

执行以下命令，同步我们创建的三个迁移：

```py
python manage.py migrate shop
```

你会看到包括以下行的输出：

```py
Applying shop.0002_add_translation_model... OK
Applying shop.0003_migrate_translatable_fields... OK
Applying shop.0004_remove_untranslated_fields... OK
```

现在模型已经跟数据库同步了。让我们翻译一个对象。

使用`python manage.py runserver`启动开发服务器，然后在浏览器中打开`http://127.0.0.1:8000/en/admin/shop/category/add/`。你会看到`Add category`页面包括两个标签页，一个英语和一个西班牙语翻译：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE9.10.png)

现在你可以添加一个翻译，然后点击`Save`按钮。确保切换标签页之前保存修改，否则输入的信息会丢失。

#### 9.2.9.6 为翻译适配视图

我们必须让`shop`的视图适配翻译的`QuerySet`。在命令行中执行`python manage.py shell`，看一眼如何检索和查询翻译字段。要获得当前语言的字段内容，你只需要与访问普遍模型字段一样访问该字段：

```py
>>> from shop.models import Product
>>> product = Product.objects.first()
>>> product.name
'Black tea'
```

当你访问翻译后的字段时，它们已经被当前语言处理了。你可以在对象上设置另一个当前语言，来访问指定的翻译：

```py
>>> product.set_current_language('es')
>>> product.name
'Té negro'
>>> product.get_current_language()
'es'
```

当使用`filter()`执行`QeurySet`时，你可以在相关的翻译对象上用`translations__`语法过滤：

```py
>>> Product.objects.filter(translations__name='Black tea')
[<Product: Black tea>]
```

你也可以用`language()`管理器为对象检索指定语言：

```py
>>> Product.objects.language('es').all()
[<Product: Té negro>, <Product: Té en polvo>, <Product: Té rojo>, <Product: Té verde>]
```

正如你所看到的，访问和查询翻译字段非常简单。

让我们适配商品目录视图。编辑`shop`应用的`views.py`文件，在`product_list`视图中找到这一行代码：

```py
category = get_object_or_404(Category, slug=category_slug)
```

替换为以下代码：

```py
language = request.LANGUAGE_CODE
category = get_object_or_404(
    Category, 
    translations__language_code=language,
    translations__slug=category_slug)
```

接着编辑`product_detail`视图，找到这一行代码：

```py
product = get_object_or_404(Product, id=id, slug=slug, available=True)
```

替换为以下代码：

```py
language = request.LANGUAGE_CODE
product = get_object_or_404(
    Product, 
    id=id, 
    translations__language_code=language,
    translations__slug=slug, 
    available=True)
```

现在`product_list`和`product_detail`视图已经适配了用翻译字段检索对象。启动开发服务器，并在浏览器中打开`http://127.0.0.1:8000/es/`。你会看到商品列表页面，所有商品都已经翻译为西班牙语：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE9.11.png)

现在用`slug`字段构建的每个商品的 URL 已经翻译为当前语言。例如，一个商品的西班牙语 URL 是`http://127.0.0.1:8000/es/1/te-negro/`，而英语的 URL 是`http://127.0.0.1:8000/en/1/black-tea/`。如果你导航到一个商品的详情页面，你会看到翻译后的 URL 和选中语言的内容，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE9.12.png)

如果你想进一步了解 django-parler，你可以在[这里](http://django-parler.readthedocs.org/en/latest/)找到所有文档。

你已经学习了如何翻译 Python 代码，模板，URL 模式和模型字段。要完成国际化和本地化过程，我们还需要显示本地化格式的日期，时间和数组。

### 9.2.10 格式的本地化

根据用户的地区，你可能希望以不同格式显示日期，时间和数字。修改项目的`settings.py`文件中的`USE_L10N`设置为`True`，可以启动本地化格式。

启用`USE_L10N`后，当 Django 在模板中输出值时，会尝试使用地区特定格式。你可以看到，你的英文版网站中的十进制用点号分隔小数部分，而不在西班牙版本中显示为逗号。这是因为 Django 为`es`地区指定了地区格式。你可以在[这里](https://github.com/django/django/blob/stable/1.11.x/django/conf/locale/es/formats.py)查看西班牙格式配置。

通常你会设置`USE_L10N`为`True`，让 Django 为每个地区应用本地化格式。但是，有些情况下你可能不想使用地区化的值。当输出必须提供机器可读的 JavaScript 或 JSON 时，这一点尤其重要。

Django 提供了`{% localize %}`模板标签，运行你在模板块中开启或关闭本地化。这让你可以控制格式的本地化。要使用这个模板标签，你必须加载`l10n`标签。下面这个例子展示了如何在模板中开启或关闭本地化：

```py
{% load l10n %}

{% localize on %}
	{{ value }}
{% endlocalize %}

{% localize off %}
	{{ value }}
{% endlocalize %}
```

Django 还提供了`localize`和`unlocalize`模板过滤器，强制或避免本地化一个值，如下所示：

```py
{{ value|localize }}
{{ value|unlocalize }}
```

你还可以创建自定义格式过滤器来指定本地格式。你可以在[这里](https://docs.djangoproject.com/en/1.11/topics/i18n/formatting/)查看更多关于格式本地化的信息。

### 9.2.11 用 django-localflavor 验证表单字段

django-localflavor 是一个第三方模板，其中包含一组特定用途的功能，比如每个国家特定的表单字段或模型字段。验证本地区域，本地电话号码，身份证，社会安全号码等非常有用。这个包由一系列以 ISO 3166 国家代码命名的模块组成。

用以下命令安装 django-localflavor：

```py
pip install django-localflavor
```

编辑项目的`settings.py`文件，把`localflavor`添加到`INSTALLED_APPS`设置中。

我们会添加一个美国（U.S）邮政编码字段，所以创建新订单时需要一个有效的美国邮政编码。

编辑`orders`应用的`forms.py`文件，如下修改：

```py
from django import forms
from .models import Order
from localflavor.us.forms import USZipCodeField

class OrderCreateForm(forms.ModelForm):
    postal_code = USZipCodeField()
    class Meta:
        model = Order
        fields = ['first_name', 'last_name', 'email', 
            'address', 'postal_code', 'city']
```

我们从`localflavor`的`us`包中导入了`USZipCodeField`字段，并把它用于`OrderCreateForm`表单的`postal_code`字段。在浏览器中打开`http://127.0.0.1:8000/en/orders/create/`，尝试输入一个 3 个字母的邮政编码。你会看`USZipCodeField`抛出的验证错误：

```py
Enter a zip code in the format XXXXX or XXXXX-XXXX.
```

这只是一个简单的例子，说明如何在你的项目中使用`localflavor`的自定义字段进行验证。`localflavor`提供的本地组件对于让你的应用适应特定国家非常有用。你可以在[这里](https://django-localflavor.readthedocs.org/en/latest/)阅读`django-localflavor`文档，查看每个国家所有可用的本地组件。

接下来，我们将在商店中构建一个推荐引擎。

## 9.3 构建推荐引擎

推荐引擎是一个预测用户对商品的偏好或评价的系统。系统根据用户行为和对用户的了解选择商品。如今，很多在线服务都使用推荐系统。它们帮助用户从大量的可用数据中选择用户可能感兴趣的内容。提供良好的建议可以增强用户参与度。电子商务网站还可以通过推荐相关产品提高销量。

我们将创建一个简单，但强大的推荐引擎，来推测用户通常会一起购买的商品。我们将根据历史销售确定通常一起购买的商品，来推荐商品。我们将在两个不同的场景推荐补充商品：

- **商品详情页面：**我们将显示一个通常与给定商品一起购买的商品列表。它会这样显示：**购买了这个商品的用户还买了 X，Y，Z。**我们需要一个数据结构，存储每个商品与显示的商品一起购买的次数。
- **购物车详情页面：**根据用户添加到购物车中的商品，我们将推荐通常与这些商品一起购买的商品。这种情况下，我们计算的获得相关商品的分数必须汇总。

我们将使用 Redis 存储一起购买的商品。记住，你已经在第六章中使用了 Redis。如果你还没有安装 Redis，请参考第六章。

### 9.3.1 根据之前的购买推荐商品

现在，我们将根据用户已经添加到购物车中的商品来推荐商品。我们将在 Redis 中为网站中每个出售的商品存储一个键。商品键会包括一个带评分的 Redis 有序集。每次完成一笔新的购买，我们为每个一起购买的商品的评分加 1。

当一个订单支付成功后，我们为购买的每个商品存储一个键，其中包括属于同一个订单的商品有序集。这个有序集让我们可以为一起购买的商品评分。

编辑项目的`settings.py`文件，编辑以下设置：

```py
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 1
```

这是建立一个 Redis 服务器连接必须的设置。在`shop`应用目录中创建一个`recommender.py`文件，添加以下代码：

```py
import redis
from django.conf import settings
from .models import Product

# connect to Redis
r = redis.StrictRedis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB
)

class Recommender:
    def get_product_key(self, id):
        return 'product:{}:purchased_with'.format(id)

    def products_bought(self, products):
        product_ids = [p.id for p in products]
        for product_id in product_ids:
            for with_id in product_ids:
                # get the other products bought with each product
                if product_id != with_id:
                    # increment score for product purchased together
                    r.zincrby(self.get_product_key(product_id), with_id, amount=1)
```

`Recommender`类允许我们存储购买的商品，以及为给定的商品检索商品推荐。`get_product_key()`方法接收一个`Product`对象的 ID，然后为存储相关商品的有序集构建 Redis 键，看起来是这样的：`product:[id]:purchased_with`。

`product_bought()`方法接收一个一起购买（也就是属于同一个订单）的`Product`对象列表。我们在这个方法中执行以下任务：

1. 我们获得给定的`Product`对象的商品 ID。
2. 我们迭代商品 ID。对于每个 ID，我们迭代商品 ID，并跳过同一个商品，这样我们获得了与每个商品一起购买的商品。
3. 我们用`get_product_id()`方法获得每个购买的商品的 Redis 商品键。对于一个 ID 是 33 的商品，这个方法返回的键是`product:33:purchased_with`。这个键用于包括与这个商品一起购买的商品 ID 的有序集。
4. 我们将 ID 包含在有序集中的商品评分加 1。这个评分表示其它商品与给定商品一起购买的次数。

因此这个方法可以保存一起购买的商品，并对它们评分。现在，我们需要一个方法检索与给定的商品列表一起购买的商品。在`Recommender`类中添加`suggest_products_for()`方法：

```py
def suggest_products_for(self, products, max_results=6):
    product_ids = [p.id for p in products]
    if len(products) == 1:
        # only 1 product
        suggestions = r.zrange(
            self.get_product_key(product_ids[0]),
            0, -1, desc=True
        )[:max_results]
    else:
        # generate a temporary key
        flat_ids = ''.join([str(id) for id in product_ids])
        tmp_key = 'tmp_{}'.format(flat_ids)
        # multiple products, combine scores of all products
        # store the resulting sored set in a temporary key
        keys = [self.get_product_key(id) for id in product_ids]
        r.zunionstore(tmp_key, keys)
        # remove ids for the products the recommendation is for 
        r.zrem(tmp_key, *product_ids)
        # get the product ids by their score, descendant sort
        suggestions = r.zrange(tmp_key, 0, -1, desc=True)[:max_results]
        # remove the temporary key
        r.delete(tmp_key)
    suggested_products_ids = [int(id) for id in suggestions]

    # get suggested products and sort by order of appearance
    suggested_products = list(Product.objects.filter(id__in=suggested_products_ids))
    suggested_products.sort(key=lambda x: suggested_products_ids.index(x.id))
    return suggested_products
```

`suggest_products_for()`方法接收以下参数：

- `products`：要获得推荐商品的商品列表。它可以包括一个或多个商品。
- `max_results`：一个整数，表示返回的推荐商品的最大数量。

在这个方法中，我们执行以下操作：

1. 我们获得给定商品对象的商品 ID。
2. 如果只给定了一个商品，我们检索与该商品一起购买的商品 ID，并按它们一起购买的总次数排序。我们用 Redis 的`ZRANGE`命令进行排序。我们限制结果数量为`max_results`参数指定的数量（默认是 6）。
3. 如果给定的商品多余 1 个，我们用商品 ID 生成一个临时的 Redis 键。
4. 我们组合每个给定商品的有序集中包括的商品，并求和所有评分。通过 Redis 的`ZUNIONSTORE`命令实现这个操作。`ZUNIONSTORE`命令用给定的键执行有序集的并集，并在新的 Redis 键中存储元素的评分总和。你可以在[这里](https://redis.io/commands/ZUNIONSTORE)阅读更多关于这个命令的信息。我们在一个临时键中存储评分和。
5. 因为我们正在汇总评分，所以我们得到的有可能是正在获得推荐商品的商品。我们用`ZREM`命令从生成的有序集中移除它们。
6. 我们从临时键中检索商品 ID，并用`ZRANGE`命令根据评分排序。我们限制结果数量为`max_results`参数指定的数量。然后我们移除临时键。
7. 最后，我们用给定的 ID 获得`Product`对象，并按 ID 同样的顺序进行排序。

为了更实用，让我们再添加一个清除推荐的方法。在`Recommender`类中添加以下方法：

```py
def clear_purchases(self):
    for id in Product.objects.values_list('id', flat=True):
        r.delete(self.get_product_key(id))
```

让我们试试推荐引擎。确保数据库中包括几个`Product`对象，并在终端使用以下命令初始化 Redis 服务：

```py
src/redis-server
```

打开另一个终端，执行`python manage.py shell`，输入下面代码检索商品：

```py
from shop.models import Product
black_tea = Product.objects.get(translations__name='Black tea')
red_tea = Product.objects.get(translations__name='Red tea')
green_tea = Product.objects.get(translations__name='Green tea')
tea_powder = Product.objects.get(translations__name='Tea powder')
```

然后添加一些测试购买到推荐引擎中：

```py
from shop.recommender import Recommender
r = Recommender()
r.products_bought([black_tea, red_tea])
r.products_bought([black_tea, green_tea])
r.products_bought([red_tea, black_tea, tea_powder])
r.products_bought([green_tea, tea_powder])
r.products_bought([black_tea, tea_powder])
r.products_bought([red_tea, green_tea])
```

我们已经存储了以下评分：

```py
black_tea: red_tea (2), tea_powder (2), green_tea (1)
red_tea: black_tea (2), tea_powder (1), green_tea (1)
green_tea: black_tea (1), tea_powder (1), red_tea(1)
tea_powder: black_tea (2), red_tea (1), green_tea (1)
```

让我们看一眼单个商品的推荐商品：

```py
>>> r.suggest_products_for([black_tea])
[<Product: Tea powder>, <Product: Red tea>, <Product: Green tea>]
>>> r.suggest_products_for([red_tea])
[<Product: Black tea>, <Product: Tea powder>, <Product: Green tea>]
>>> r.suggest_products_for([green_tea])
[<Product: Black tea>, <Product: Tea powder>, <Product: Red tea>]
>>> r.suggest_products_for([tea_powder])
[<Product: Black tea>, <Product: Red tea>, <Product: Green tea>]
```

正如你所看到的，推荐商品的顺序基于它们的评分排序。让我们用多个商品的评分总和获得推荐商品：

```py
>>> r.suggest_products_for([black_tea, red_tea])
[<Product: Tea powder>, <Product: Green tea>]
>>> r.suggest_products_for([green_tea, red_tea])
[<Product: Black tea>, <Product: Tea powder>]
>>> r.suggest_products_for([tea_powder, black_tea])
[<Product: Red tea>, <Product: Green tea>]
```

你可以看到，推荐商品的顺序与评分总和匹配。例如，`black_tea`和`red_tea`的推荐商品是`tea_powder(2+1)`和`green_tea(1+1)`。

我们已经确认推荐算法如期工作了。让我们为网站的商品显示推荐。

编辑`shop`应用的`views.py`文件，并添加以下导入：

```py
from .recommender import Recommender
```

在`product_detail()`视图的`render()`函数之前添加以下代码：

```py
r = Recommender()
recommended_products = r.suggest_products_for([product], 4)
```

我们最多获得 4 个推荐商品。现在`product_detail`视图如下所示：

```py
from .recommender import Recommender

def product_detail(request, id, slug):
    language = request.LANGUAGE_CODE
    product = get_object_or_404(
        Product, 
        id=id, 
        translations__language_code=language,
        translations__slug=slug, 
        available=True)
    cart_product_form = CartAddProductForm()
    r = Recommender()
    recommended_products = r.suggest_products_for([product], 4)
    return render(
        request,
        'shop/product/detail.html',
        {
            'product': product,
            'cart_product_form': cart_product_form,
            'recommended_products': recommended_products
        }
    )
```

现在编辑`shop`应用的`shop/product/detail.html`模板，在`{{ product.description|linebreaks }}`之后添加以下代码：

```py
{% if recommended_products %}
    <div class="recommendations">
        <h3>{% trans "People who bought this also bought" %}</h3>
        {% for p in recommended_products %}
            <div class="item">
                <a href="{{ p.get_absolute_url }}">
                    <img src="{% if p.image %}{{ p.image.url }}{% else %}{% static "img/no_image.png" %}{% endif %}">
                </a>
                <p><a href="{{ p.get_absolute_url }}">{{ p.name }}</a></p>
            </div>
        {% endfor %}
    </div>
{% endif %}
```

使用`python manage.py runserver`启动开发服务器，并在浏览器中打开`http://127.0.0.1:8000/en/`。点击任何一个商品显示详情页面。你会看到商品下面的推荐商品，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE9.13.png)

接下来我们在购物车中包括商品推荐。基于用户添加到购物车中的商品生成推荐商品。编辑`cart`应用的`views.py`文件，添加以下导入：

```py
from shop.recommender import Recommender
```

然后编辑`cart_detail`视图，如下所示：

```py
def cart_detail(request):
    cart = Cart(request)
    for item in cart:
        item['update_quantity_form'] = CartAddProductForm(
            initial={'quantity': item['quantity'], 'update': True})
    coupon_apply_form = CouponApplyForm()

    r = Recommender()
    cart_products = [item['product'] for item in cart]
    recommended_products = r.suggest_products_for(cart_products, max_results=4)
    return render(
        request, 'cart/detail.html', 
        {
            'cart': cart, 
            'coupon_apply_form': coupon_apply_form,
            'recommended_products': recommended_products
        }
    )
```

编辑`cart`应用的`cart/detail.html`模板，在`</table>`标签之后添加以下代码：

```py
{% if recommended_products %}
    <div class="recommendations cart">
        <h3>{% trans "People who bought this also bought" %}</h3>
        {% for p in recommended_products %}
            <div class="item">
                <a href="{{ p.get_absolute_url }}">
                    <img src="{% if p.image %}{{ p.image.url }}{% else %}{% static "img/no_image.png" %}{% endif %}">
                </a>
                <p><a href="{{ p.get_absolute_url }}">{{ p.name }}</a></p>
            </div>
        {% endfor %}
    </div>
{% endif %}
```

在浏览器中打开`http://127.0.0.1:8000/en/`，并添加一些商品到购物车中。当你导航到`http://127.0.0.1:8000/en/cart/`，你会看到购物车中商品合计的推荐商品，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE9.14.png)

恭喜你！你已经用 Django 和 Redis 构建了一个完整的推荐引擎。

## 9.4 总结

在本章中，你使用会话创建了优惠券系统。你学习了如何进行国际化和本地化。你还用 Redis 构建了一个推荐引擎。

在下一章中，你会开始一个新的项目。你会通过 Django 使用基于类的视图构建一个在线学习平台，你还会创建一个自定义的内容管理系统。