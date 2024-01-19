# 订单微服务

在本章中，我们将扩展我们在[第7章](a8e0af3b-67d9-4649-986b-041d136af0e8.xhtml)中实现的Web应用程序，*使用Django创建在线视频游戏商店*。我不知道您是否注意到，在该项目中有一些重要的东西缺失。首先是提交订单的能力。就目前而言，用户可以浏览产品并将商品添加到购物车；但是，没有办法发送订单并完成购买。

另一个缺失的项目是我们应用程序的用户能够查看已发送的所有订单以及其订单历史的页面。

说到这里，我们将创建一个名为*order*的微服务，它将处理网站上的所有订单相关事务。它将接收订单，更新订单等等。

在本章中，您将学到：

+   创建微服务的基础知识

+   如何使用Django REST框架创建RESTful API

+   如何使用服务并将其与其他应用程序集成

+   如何编写测试

+   如何在AWS上部署应用程序

+   如何使用Gunicorn在HTTP代理`nginx`后运行我们的Web应用程序

所以，让我们开始吧！

# 设置环境

就像之前的所有章节一样，我们将从设置我们需要在其上开发服务的环境开始这一章。让我们首先创建我们的工作目录：

```py
mkdir microservices && cd microservices
```

然后，我们使用`pipenv`创建我们的虚拟环境：

```py
pipenv --python ~/Install/Python3.6/bin/python3.6
```

如果您不知道如何使用`pipenv`，在[第4章](2223dee0-d5de-417e-9ca9-6bf4a6038cb6.xhtml)的*设置环境*部分，*汇率和货币转换工具*中，有一个非常好的介绍，介绍了如何开始使用`pipenv`。

创建虚拟环境后，我们需要安装项目依赖项。对于这个项目，我们将安装Django和Django REST框架：

```py
pipenv install django djangorestframework requests python-dateutil
```

我们使用Django和Django REST框架而不是像Flask这样的简单框架的原因是，这个项目的主要目的是提供关注点的分离，创建一个将处理在前一章中开发的在线游戏商店中的订单的微服务。我们不仅希望提供供Web应用程序消费的API。最好有一个简单的网站，以便我们可以列出订单，查看每个订单的详细信息，并执行更新，如更改订单状态。

正如您在上一章中看到的，Django已经拥有一个非常强大和灵活的管理界面，我们可以自定义以向用户提供这种功能，而无需花费太多时间开发Web应用程序。

安装依赖项后，您的`Pipfile`应如下所示：

```py
[[source]]

verify_ssl = true
name = "pypi"
url = "https://pypi.python.org/simple"

[packages]

django = "*"
djangorestframework = "*"

[dev-packages]

[requires]

python_version = "3.6"
```

完美！现在，我们可以开始一个新的Django项目。我们将使用`django-admin`工具创建项目。让我们继续创建一个名为`order`的项目：

```py
django-admin startproject order
```

创建项目后，我们将创建一个Django应用程序。对于这个项目，我们将只创建一个名为`main`的应用程序。首先，我们将更改目录到服务目录：

```py
cd order
```

然后，我们再次使用`django-admin`工具创建一个应用程序：

```py
django-admin startapp main
```

创建Django应用程序后，您的项目结构应该类似于以下结构：

```py
.
├── main
│   ├── admin.py
│   ├── apps.py
│   ├── __init__.py
│   ├── migrations
│   │   └── __init__.py
│   ├── models.py
│   ├── tests.py
│   └── views.py
├── manage.py
└── order
    ├── __init__.py
    ├── settings.py
    ├── urls.py
    └── wsgi.py
```

接下来，我们将开始创建我们服务的模型。

# 创建服务模型

在订单服务的第一部分，我们将创建一个模型，用于存储来自在线视频游戏商店的订单数据。让我们打开主应用程序目录中的`models.py`文件，并开始添加模型：

```py
class OrderCustomer(models.Model):
    customer_id = models.IntegerField()
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
```

我们将创建一个名为`OrderCustomer`的类，它继承自`Model`，并定义三个属性；`customer_id`，它将对应于在线游戏商店中的客户ID，客户的`name`，最后是`email`。

然后，我们将创建存储订单信息的模型：

```py
class Order(models.Model):

    ORDER_STATUS = (
        (1, 'Received'),
        (2, 'Processing'),
        (3, 'Payment complete'),
        (4, 'Shipping'),
        (5, 'Completed'),
        (6, 'Cancelled'),
    )

    order_customer = models.ForeignKey(
        OrderCustomer, 
        on_delete=models.CASCADE
    )    
    total = models.DecimalField(
        max_digits=9,
        decimal_places=2,
        default=0
    )
    created_at = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    status = models.IntegerField(choices=ORDER_STATUS, default='1')   
```

`Order`类继承自`Model`，我们通过添加一个包含应用中订单状态的元组来开始这个类。我们还定义了一个外键`order_customer`，它将创建`OrderCustomer`和`Order`之间的关系。然后是定义其他字段的时间，从`total`开始，它是该订单的总购买价值。然后有两个日期时间字段；`created_at`，这是顾客提交订单的日期，`last_update`，这是在我们想知道订单何时有状态更新时将要使用的字段。

当将`auto_now_add`添加到`DateTimeField`时，Django使用`django.utils.timezone.now`函数，该函数将返回带有时区信息的当前`datetime`对象。DateField使用`datetime.date.today()`，它不包含时区信息。

我们要创建的最后一个模型是`OrderItems`。这将保存属于订单的项目。我们将像这样定义它：

```py
class OrderItems(models.Model):
    class Meta:
        verbose_name_plural = 'Order items'

    product_id = models.IntegerField()
    name = models.CharField(max_length=200)
    quantity = models.IntegerField()
    price_per_unit = models.DecimalField(
        max_digits=9,
        decimal_places=2,
        default=0 
    )
    order = models.ForeignKey(
        Order, on_delete=models.CASCADE, related_name='items')
```

在这里，我们还定义了一个`Meta`类，以便我们可以为模型设置一些元数据。在这种情况下，我们将`verbose_name_plural`设置为`Order items`，以便在Django管理界面中正确拼写。然后，我们定义了`product_id`，`name`，`quantity`和`price_per_unit`，它们指的是在线视频游戏商店中的`Game`模型。

最后，我们有项目数量和外键`Order`。

现在，我们需要编辑`microservices/order/order`目录中的`settings.py`文件，并将主应用程序添加到`INSTALLED_APPS`中。它应该是这样的：

```py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'main',
]
```

唯一剩下的就是创建和应用数据库迁移。首先，我们运行`makemigrations`命令：

```py
python manage.py makemigrations
```

然后`迁移`将更改应用到数据库：

```py
python manage.py migrate
```

# 创建模型的管理器

为了使我们的应用程序更易读，不要在端点中充斥着大量业务逻辑，我们将为我们的模型类创建管理器。如果您遵循了上一章，您应该对此非常熟悉。简而言之，管理器是为Django模型提供查询操作的接口。

默认情况下，Django为每个模型添加一个管理器；它存储在名为objects的属性上。Django添加到模型的默认管理器有时是足够的，不需要创建自定义管理器，但是将所有与数据库相关的代码保持在模型内是一个好习惯。这将使我们的代码更一致、可读，并且更易于测试和维护。

在我们的情况下，我们感兴趣创建的唯一模型是名为Order的自定义模型管理器，但在我们开始实现订单管理器之前，我们需要创建一些辅助类。我们需要创建的第一个类是一个将定义在执行数据库查询时可能发生的自定义异常的类。当然，我们可以使用标准库中已经定义的异常，但是在应用程序的上下文中创建有意义的异常总是一个好习惯。

我们要创建的三个异常是`InvalidArgumentError`，`OrderAlreadyCompletedError`和`OrderCancellationError`。

当将无效参数传递给我们将在管理器中定义的函数时，将引发异常`InvalidArgumentError`，因此让我们继续在主应用程序目录中创建一个名为`exceptions.py`的文件，并包含以下内容：

```py
class InvalidArgumentError(Exception):
    def __init__(self, argument_name):
        message = f'The argument {argument_name} is invalid'
        super().__init__(message)
```

在这里，我们定义了一个名为`InvalidArgumentError`的类，它继承自`Exception`，并且我们在其中唯一要做的事情是重写构造函数并接收一个名为`argument_name`的参数。通过这个参数，我们可以指定引发异常的原因。

我们还将自定义异常消息，最后，我们将在超类上调用构造函数。

我们还将创建一个异常，当我们尝试取消状态为已取消的订单时，将引发异常，以及当我们尝试将订单的状态设置为已完成时，订单已经完成时：

```py
class OrderAlreadyCompletedError(Exception):
    def __init__(self, order):
        message = f'The order with ID: {order.id} is already  
        completed.'
  super().__init__(message)

class OrderAlreadyCancelledError(Exception):
    def __init__(self, order):
        message = f'The order with ID: {order.id} is already  
        cancelled.'
  super().__init__(message)
```

然后，我们将添加另外两个自定义异常：

```py
class OrderCancellationError(Exception):
    pass     class OrderNotFoundError(Exception):
    pass
```

这两个类并没有做太多事情。它们只是从`Exception`继承。我们将为每个异常配置和自定义消息，并将其传递给超类初始化程序。自定义异常类的价值在于它将提高我们应用程序的可读性和可维护性。

太好了！在开始管理之前，我们只需要添加一件事。我们将在模型管理器中创建函数，该函数将返回按状态过滤的数据。正如您所看到的，在`Order`模型的定义中，我们定义了状态如下：

```py
ORDER_STATUS = (
    (1, 'Received'),
    (2, 'Processing'),
    (3, 'Payment complete'),
    (4, 'Shipping'),
    (5, 'Completed'),
    (6, 'Cancelled'),
)
```

这意味着，如果我们想要获取所有状态为`Completed`的订单，我们需要编写类似以下行的内容：

```py
  Order.objects.filter(status=5)
```

这段代码只有一个问题，你能猜到是什么吗？如果你猜到了*魔法*数字`5`，那你绝对是对的！想象一下，如果我们的同事需要维护这段代码，并且只看到那里的数字`5`，并不知道5实际上代表什么，他们会有多沮丧。因此，我们将创建一个枚举，以便用来表示不同的状态。让我们在`main`应用程序目录中创建一个名为`status.py`的文件，并添加以下枚举：

```py
from enum import Enum, auto

class Status(Enum):
    Received = auto()
    Processing = auto()
    Payment_Complete = auto()
    Shipping = auto()
    Completed = auto()
    Cancelled = auto()
```

因此，现在，当我们需要获取所有状态为`Completed`的订单时，我们可以这样做：

```py
Order.objects.filter(Status.Received.value)
```

好多了！

现在，让我们为其创建模型管理器。在邮件应用程序目录中创建一个名为`managers.py`的文件，我们可以开始添加一些导入：

```py
from datetime import datetime
from django.db.models import Manager, Q

from .status import Status

from .exceptions import InvalidArgumentError
from .exceptions import OrderAlreadyCompletedError
from .exceptions import OrderCancellationError

from . import models
```

然后，我们定义`OrderManager`类和第一个名为`set_status`的方法：

```py
class OrderManager(Manager):

    def set_status(self, order, status):
        if status is None or not isinstance(status, Status):
            raise InvalidArgumentError('status')

        if order is None or not isinstance(order, models.Order):
            raise InvalidArgumentError('order')

        if order.status is Status.Completed.value:
            raise OrderAlreadyCompletedError()

        order.status = status.value
        order.save()
```

这种方法需要两个参数，订单和状态。`order`是`Order`类型的对象，状态是我们之前创建的`Status`枚举的一个项目。

我们通过验证参数并引发相应的异常来开始这种方法。首先，我们验证字段是否具有值并且是正确的类型。如果验证失败，它将引发`InvalidArgumentError`。然后，我们检查我们正在为其设置状态的订单是否已经完成；在这种情况下，我们无法再更改它，因此我们引发`OrderAlreadyCompletedError`。如果所有参数都有效，我们设置订单的状态并保存。

在我们的应用程序中，我们希望能够取消尚未处理的订单；换句话说，我们只允许在状态为`Received`时取消订单。`cancel_order`方法应该如下所示：

```py
def cancel_order(self, order):
    if order is None or not isinstance(order, models.Order):
        raise InvalidArgumentError('order')

    if order.status != Status.Received.value:
        raise OrderCancellationError()

    self.set_status(order, Status.Cancelled)
```

这种方法只获取`order`参数，首先，我们需要检查订单对象是否有效，并在无效时引发`InvalidArgumentError`。然后，我们检查订单的状态是否为`not Received`。在这种情况下，我们引发`OrderCancellationError`异常。否则，我们继续调用`set_status`方法，传递`Status.Cancelled`作为参数。

我们还需要获取给定客户的所有订单列表：

```py
def get_all_orders_by_customer(self, customer_id):
    try:
        return self.filter(
            order_customer_id=customer_id).order_by(
            'status', '-created_at')
    except ValueError:
        raise InvalidArgumentError('customer_id')
```

`get_all_orders_by_customer`方法将`customer_id`作为参数。然后，我们使用filter函数来按`customer_id`过滤订单，同时按状态排序；仍在处理中的订单将位于QuerySet的顶部。

如果`customer_id`无效，例如，如果我们传递的是字符串而不是整数，则会引发`ValueError`异常。我们捕获此异常并引发我们的自定义异常`InvalidArgumentError`。

我们在线视频游戏商店的财务部门要求获取特定用户的所有完整和不完整订单列表，因此让我们为其添加一些方法：

```py
def get_customer_incomplete_orders(self, customer_id):
    try:
        return self.filter(
            ~Q(status=Status.Completed.value),
            order_customer_id=customer_id).order_by('status')
    except ValueError:
        raise InvalidArgumentError('customer_id')

def get_customer_completed_orders(self, customer_id):
    try:
        return self.filter(
            status=Status.Completed.value,
            order_customer_id=customer_id)
    except ValueError:
        raise InvalidArgumentError('customer_id')
```

第一个方法`get_customer_incomplete_orders`获取一个名为`customer_id`的参数。就像之前的方法一样；我们将捕获`ValueError`异常，以防`customer_id`无效，并引发`InvalidArgumentError`。这种方法的有趣之处在于过滤器。在这里，我们使用`Q()`对象，它封装了一个`Python`对象形式的SQL表达式。

在这里，我们有`~Q(status=Status.Completed.value)`，这是`not`运算符，等同于状态不是`Status.Complete`。我们还过滤`order_customer_id`以检查它是否等于方法的`customer_id`参数，最后，我们按状态对QuerySet进行排序。

`get_customer_completed_orders`基本上是一样的，但这次我们过滤状态等于`Status.Completed`的订单。

`Q()`对象允许我们编写更复杂的查询，利用`|`（或）和`&`（与）运算符。

接下来，负责订单生命周期的每个部门都希望有一种简单的方式来获取处于特定阶段的订单；例如，负责发货游戏的工作人员希望获取所有状态等于“支付完成”的订单列表，以便将这些订单发货给客户。因此，我们需要添加一个方法来实现这一点：

```py
def get_orders_by_status(self, status):
    if status is None or not isinstance(status, Status):
        raise InvalidArgumentError('status')

    return self.filter(status=status.value)
```

这是一个非常简单的方法；在这里，我们将状态作为参数。我们检查状态是否有效；如果无效，我们引发`InvalidArgumentError`。否则，我们继续并按状态过滤订单。

我们财务部门的另一个要求是获取特定日期范围内的订单列表：

```py
def get_orders_by_period(self, start_date, end_date):
    if start_date is None or not isinstance(start_date, datetime):
        raise InvalidArgumentError('start_date')

    if end_date is None or not isinstance(end_date, datetime):
        raise InvalidArgumentError('end_date')

    result = self.filter(created_at__range=[start_date, end_date])
    return result
```

在这里，我们得到两个参数`start_date`和`end_date`。与所有其他方法一样，我们首先检查这些参数是否有效；在这种情况下，参数不能是`None`，并且必须是`Datetime`对象的实例。如果任何字段无效，将引发`InvalidArgumentError`。当参数有效时，我们使用`created_at`字段过滤订单，还使用了特殊的语法`created_at__range`，这意味着我们将传递一个日期范围，并将其用作过滤器。在这里，我们传递`start_date`和`end_date`。

可能有一个有趣的方法可以实现，并且可以为我们应用程序的管理员增加价值。这里的想法是添加一个方法，当调用时，自动将订单更改为下一个状态：

```py
def set_next_status(self, order):
    if order is None or not isinstance(order, models.Order):
        raise InvalidArgumentError('order')

    if order.status is Status.Completed.value:
        raise OrderAlreadyCompletedError()

    order.status += 1
    order.save()
```

这个方法只接受一个参数，即订单。我们检查订单是否有效，如果无效，我们引发`InvalidArgumentError`。我们还希望确保一旦订单达到“已完成”状态，就不能再更改。因此，我们检查订单是否处于“已完成”状态，然后引发`OrderAlreadyCompleted`异常。最后，我们将当前状态加1并保存对象。

现在，我们可以更改我们的`Order`模型，使其使用我们刚刚创建的`OrderManager`。打开主应用程序目录中的model.py文件，在`Order`类的末尾添加以下行：

```py
objects = OrderManager()
```

现在，我们可以通过`Order.objects`访问我们在`OrderManager`中定义的所有方法。

接下来，我们将为我们的模型管理器方法添加测试。

# 学习测试

到目前为止，在本书中，我们还没有涵盖如何创建测试。现在是一个很好的时机，所以我们将为模型管理器中创建的方法创建测试。

我们为什么需要测试？对这个问题的简短回答是，测试将使我们知道方法或函数是否做了正确的事情。另一个原因（也是我认为最重要的原因之一）是，测试在进行代码更改时给我们更多的信心。

Django在开箱即用的情况下提供了出色的工具来创建单元测试和集成测试，并结合像Selenium这样的框架，可以基本上测试我们应用的所有部分。

说了这些，让我们创建我们的第一个测试。当创建一个新的Django应用程序时，Django会在`app`目录中创建一个名为`test.py`的文件。您可以在其中编写您的测试，或者如果您更喜欢通过将测试分成多个文件来使项目更有组织性，您可以删除该文件并创建一个名为`tests`的目录，并将所有测试文件放在其中。由于我们只打算为Order模型管理器创建测试，我们将把所有测试都放在Django为我们创建的`tests.py`文件中。

# 创建测试文件

打开`test.py`文件，让我们首先添加一些导入：

```py
from dateutil.relativedelta import relativedelta

from django.test import TestCase
from django.utils import timezone

from .models import OrderCustomer, Order
from .status import Status

from .exceptions import OrderAlreadyCompletedError
from .exceptions import OrderCancellationError
from .exceptions import InvalidArgumentError
```

很好！我们首先导入相对增量函数，这样我们就可以轻松进行日期操作，比如向日期添加天数或月数。这在测试按一定时间段获取订单的方法时将非常有帮助。

现在，我们导入一些与Django相关的内容。首先是`TestCase`类，它是`unittest.TestCase`的子类。由于我们将编写与数据库交互的测试，最好使用`django.tests.TestCase`而不是`unittest.TestCase`。Django的`TestCase`实现将确保您的测试在事务中运行，以提供隔离。这样，当运行测试时，由于测试套件中另一个测试创建的数据，我们将不会有不可预测的结果。

我们还导入了一些我们将在测试中使用的模型类，`Order`，`OrderCustomer`模型，以及在测试更改订单状态的方法时使用的Status类。

在为应用程序编写测试时，我们不仅要测试*好*的情况，还要测试当出现问题时，当错误的参数传递给正在测试的函数和方法时。因此，我们导入我们自定义的错误类，以确保在正确的情况下引发正确的异常。

现在我们已经导入了必要的内容，是时候创建类和方法来为我们的测试设置数据了：

```py
class OrderModelTestCase(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.customer_001 = OrderCustomer.objects.create(
            customer_id=1,
            email='customer_001@test.com'
        )

        Order.objects.create(order_customer=cls.customer_001)

        Order.objects.create(order_customer=cls.customer_001,
                             status=Status.Completed.value)

        cls.customer_002 = OrderCustomer.objects.create(
            customer_id=1,
            email='customer_002@test.com'
        )

        Order.objects.create(order_customer=cls.customer_002)
```

在这里，我们创建了一个名为`OrderModelTestCase`的类，继承自`django.test.TestCase`。然后，我们定义了`setUpTestData`方法，这个方法将负责设置每个测试将使用的数据。

在这里，我们创建了两个用户；第一个用户有两个订单，其中一个订单状态设置为`Completed`。第二个用户只有一个订单。

# 测试取消订单功能

我们要测试的第一个方法是`cancel_orders`方法。顾名思义，它将取消一个订单。在这个方法中，有一些我们想要测试的东西：

+   第一个测试非常直接；我们只想测试是否可以取消订单，将其状态设置为`Cancelled`。

+   第二个测试是不应该取消尚未收到的订单；换句话说，只有当前状态设置为`Received`的订单才能被取消。

+   我们需要测试如果将无效参数传递给`cancel_order`方法时是否会引发正确的异常。

说了这些，让我们添加我们的测试：

```py
def test_cancel_order(self):
    order = Order.objects.get(pk=1)

    self.assertIsNotNone(order)
    self.assertEqual(Status.Received.value, order.status)

    Order.objects.cancel_order(order)

    self.assertEqual(Status.Cancelled.value, order.status)

def test_cancel_completed_order(self):
    order = Order.objects.get(pk=2)

    self.assertIsNotNone(order)
    self.assertEqual(Status.Completed.value, order.status)

    with self.assertRaises(OrderCancellationError):
        Order.objects.cancel_order(order)

def test_cancel_order_with_invalid_argument(self):
    with self.assertRaises(InvalidArgumentError):
        Order.objects.cancel_order({'id': 1})
```

第一个测试`test_cancel_order`，首先获取ID为1的订单。我们使用`assertIsNotNone`函数断言返回的值不是`None`，同时使用`assertEqual`函数确保订单的状态是`Received`。

然后，我们从订单模型管理器中调用`cancel_order`方法传递订单，最后，我们再次使用`assertEqual`函数来验证订单的状态是否确实更改为`Cancelled`。

第二个测试`test_cancel_complated_order`从获取ID等于`2`的订单开始；请记住，我们已将此订单设置为`Completed`状态。然后，我们做与上一个测试相同的事情；验证订单不等于`None`，并验证状态设置为`Complete`。最后，我们使用`assertRaises`函数测试，如果我们尝试取消已取消的订单，将引发正确的异常；在这种情况下，将引发`OrderCancellationError`类型的异常。

最后，我们有`test_cancel_order_with_invalid_argument`函数，它将测试如果我们向`cancel_order`函数传递无效参数，是否会引发正确的异常。

# 测试获取所有订单的功能

现在，我们将为`get_all_orders_by_customer`方法添加测试。对于这个方法，我们需要测试：

+   当给定顾客ID时，返回正确数量的订单

+   当向方法传递无效参数时引发正确的异常

```py
def test_get_all_orders_by_customer(self):
    orders = Order.objects.get_all_orders_by_customer(customer_id=1)

    self.assertEqual(2, len(orders),
                     msg='It should have returned 2 orders.')

def test_get_all_order_by_customer_with_invalid_id(self):
    with self.assertRaises(InvalidArgumentError):
        Order.objects.get_all_orders_by_customer('o')
```

`get_all_orders_by_customer`方法的测试非常简单。在第一个测试中，我们获取ID为`1`的顾客的订单，并测试返回的项目数量是否等于`2`。

在第二个测试中，我们断言调用`get_all_orders_by_customer`时使用无效参数，实际上会引发`InvalidArgumentError`类型的异常。在这种情况下，测试将成功通过。

# 获取顾客的不完整订单

`get_customer_incomplete_orders`方法返回给定顾客ID的状态与`Completed`不同的所有订单。对于这个测试，我们需要验证：

+   该方法返回正确数量的项目，以及返回的项目是否没有状态等于`Completed`

+   我们将测试当向该方法传递无效值时是否引发异常

```py
def test_get_customer_incomplete_orders(self):
    orders = Order.objects.get_customer_incomplete_orders(customer_id=1)

    self.assertEqual(1, len(orders))
    self.assertEqual(Status.Received.value, orders[0].status)

def test_get_customer_incomplete_orders_with_invalid_id(self):
    with self.assertRaises(InvalidArgumentError):
        Order.objects.get_customer_incomplete_orders('o')
```

测试`test_get_customer_incomplete_orders`从调用`get_customer_incomplete_orders`函数开始，并将顾客ID设置为`1`作为参数传递。然后，我们验证返回的项目数量是否正确；在这种情况下，只有一个不完整的订单，所以应该是`1`。最后，我们检查返回的项目是否实际上具有与`Completed`不同的状态。

另一个测试与之前测试异常的测试完全相同，只是调用该方法并断言已引发了正确的异常。

# 获取顾客的已完成订单

接下来，我们将测试`get_customer_completed_order`。这个方法，正如其名称所示，返回给定顾客的所有状态为`Completed`的订单。在这里，我们将测试与`get_customer_incompleted_orders`相同的场景：

```py
def test_get_customer_completed_orders(self):
    orders = Order.objects.get_customer_completed_orders(customer_id=1)

    self.assertEqual(1, len(orders))
    self.assertEqual(Status.Completed.value, orders[0].status)

def test_get_customer_completed_orders_with_invalid_id(self):
    with self.assertRaises(InvalidArgumentError):
        Order.objects.get_customer_completed_orders('o')
```

首先，我们调用`get_customer_completed_orders`，传递顾客ID等于`1`，然后验证返回的项目数量是否等于`1`。最后，我们验证返回的项目是否实际上具有状态设置为`Completed`。

# 按状态获取订单

`get_order_by_status`函数根据状态返回订单列表。这里有两种情况需要测试：

+   如果该方法返回给定状态的正确数量的订单

+   当向方法传递无效参数时引发正确的异常

```py
def test_get_order_by_status(self):
    order = Order.objects.get_orders_by_status(Status.Received)

    self.assertEqual(2, len(order),
                     msg=('There should be only 2 orders '
                          'with status=Received.'))

    self.assertEqual('customer_001@test.com',
                     order[0].order_customer.email)

def test_get_order_by_status_with_invalid_status(self):
    with self.assertRaises(InvalidArgumentError):
        Order.objects.get_orders_by_status(1)
```

很简单。我们首先调用`get_orders_by_status`进行第一项测试，将`Status.Received`作为参数传递。然后，我们验证只有两个订单被返回。

对于第二个测试，对于`get_order_by_status`方法，与之前的异常测试一样，运行该方法，传递无效参数，然后验证是否引发了`InvalidArgumentError`类型的异常。

# 按期获取订单

现在，我们将测试`get_order_by_period`方法，该方法在给定初始日期和结束日期时返回订单列表。对于这个方法，我们将执行以下测试：

+   调用该方法，传递参数，应返回在该期间创建的订单

+   调用方法，传递我们知道没有创建任何订单的有效日期，这应该返回一个空结果

+   测试在调用方法时是否会引发异常，传递无效的开始日期

+   测试在调用方法时是否会引发异常，传递无效的结束日期

```py
def test_get_orders_by_period(self):

    date_from = timezone.now() - relativedelta(days=1)
    date_to = date_from + relativedelta(days=2)

    orders = Order.objects.get_orders_by_period(date_from, date_to)

    self.assertEqual(3, len(orders))

    date_from = timezone.now() + relativedelta(days=3)
    date_to = date_from + relativedelta(months=1)

    orders = Order.objects.get_orders_by_period(date_from, date_to)

    self.assertEqual(0, len(orders))

def test_get_orders_by_period_with_invalid_start_date(self):
    start_date = timezone.now()

    with self.assertRaises(InvalidArgumentError):
        Order.objects.get_orders_by_period(start_date, None)

def test_get_orders_by_period_with_invalid_end_date(self):
    end_date = timezone.now()

    with self.assertRaises(InvalidArgumentError):
        Order.objects.get_orders_by_period(None, end_date)
```

我们通过创建`date_from`方法来开始这个方法，它是当前日期减去一天。在这里，我们使用`python-dateutil`包的`relativedelta`方法执行日期操作。然后，我们定义`date_to`，它是当前日期加两天。

现在我们有了我们的时间段，我们可以将这些值作为参数传递给`get_orders_by_period`方法。在我们的情况下，我们设置了三个订单，全部都是用当前日期创建的，因此这个方法调用应该返回确切的三个订单。

然后，我们定义了一个我们知道不会有任何订单的不同时间段。`date_from`函数定义为当前日期加三天，因此`date_from`是当前日期加`1`个月。

再次调用该方法，传递`date_from`和`date_to`的新值不应该返回任何订单。

`get_orders_by_period`的最后两个测试与我们之前实现的异常测试相同。

# 设置订单的下一个状态

我们将要创建的`Order`模型管理器的下一个方法是`set_next_status`方法。`set_next_status`方法只是一个方便使用的方法，它将设置订单的下一个状态。如果你记得，我们创建的`Status`枚举意味着枚举中的每个项目都设置为`auto()`，这意味着枚举中的项目将获得一个数字顺序号作为值。

当我们将订单保存在数据库中并将其状态设置为，例如`Status.Processing`时，数据库中状态字段的值将为`2`。

该功能只是将`1`添加到当前订单的状态，因此它转到下一个状态项目，除非状态是`Completed`；那是订单生命周期的最后状态。

现在我们已经刷新了关于这种方法如何工作的记忆，是时候为它创建测试了，我们将不得不执行以下测试：

+   当调用`set_next_status`时，订单会获得下一个状态

+   测试在调用`set_next_status`并传递状态为`Completed`的订单时是否会引发异常

+   测试在传递无效订单作为参数时是否会引发异常

```py
def test_set_next_status(self):
    order = Order.objects.get(pk=1)

    self.assertTrue(order is not None,
                    msg='The order is None.')

    self.assertEqual(Status.Received.value, order.status,
                     msg='The status should have been 
                     Status.Received.')

    Order.objects.set_next_status(order)

    self.assertEqual(Status.Processing.value, order.status,
                     msg='The status should have been 
                     Status.Processing.')

def test_set_next_status_on_completed_order(self):
    order = Order.objects.get(pk=2)

    with self.assertRaises(OrderAlreadyCompletedError):
        Order.objects.set_next_status(order)

def test_set_next_status_on_invalid_order(self):
    with self.assertRaises(InvalidArgumentError):
        Order.objects.set_next_status({'order': 1})
```

第一个测试`test_set_next_status`开始通过获取ID等于`1`的订单。然后，它断言订单对象不等于none，并且我们还断言订单状态的值为`Received`。然后，我们调用`set_next_status`方法，将订单作为参数传递。然后，我们再次断言以确保状态已经改变。如果订单的状态等于`2`，也就是`Status`枚举中的`Processing`，则测试将通过。

另外两个测试与订单测试非常相似，我们断言异常，但值得一提的是测试`test_set_next_status_on_completed_order`断言，如果我们尝试在状态等于`Status.Completed`的订单上调用`set_next_status`，那么将引发`OrderAlreadyCompletedError`类型的异常。

# 设置订单的状态

最后，我们将实现`Order`模型管理器的最后测试。我们将为`set_status`方法创建测试。`set_status`方法确实做了它的名字所暗示的事情；它将为给定的订单设置状态。我们需要执行以下测试：

+   设置状态并验证订单的状态是否真的已经改变

+   在已经完成的订单中设置状态；它应该引发`OrderAlreadyCompletedError`类型的异常

+   在已经取消的订单中设置状态；它应该引发`OrderAlreadyCancelledError`类型的异常

+   使用无效订单调用`set_status`方法；它应该引发`InvalidArgumentError`类型的异常

+   使用无效状态调用`set_status`方法；它应该引发`InvalidArgumentError`类型的异常

```py
def test_set_status(self):
    order = Order.objects.get(pk=1)

    Order.objects.set_status(order, Status.Processing)

    self.assertEqual(Status.Processing.value, order.status)

def test_set_status_on_completed_order(self):
    order = Order.objects.get(pk=2)

    with self.assertRaises(OrderAlreadyCompletedError):
        Order.objects.set_status(order, Status.Processing)

def test_set_status_on_cancelled_order(self):
    order = Order.objects.get(pk=1)
    Order.objects.cancel_order(order)

    with self.assertRaises(OrderAlreadyCancelledError):
        Order.objects.set_status(order, Status.Processing)

def test_set_status_with_invalid_order(self):
    with self.assertRaises(InvalidArgumentError):
        Order.objects.set_status(None, Status.Processing)

def test_set_status_with_invalid_status(self):
    order = Order.objects.get(pk=1)

    with self.assertRaises(InvalidArgumentError):
        Order.objects.set_status(order, {'status': 1})
```

我们不会遍历所有测试，因为它们测试异常的方式类似于我们之前实现的测试，但是值得浏览第一个测试。在`test_set_status`测试中，它将获取ID等于`1`的订单，正如我们在`setUpTestData`中定义的那样，其状态等于`Status.Received`。我们调用`set_status`方法，传递订单和新状态作为参数，在这种情况下是`Status.Processing`。设置新状态后，我们只需调用`assertEquals`来确保订单的状态实际上已更改为`Status.Processing`。

# 创建订单模型序列化程序

现在我们已经有了一切我们需要开始创建API端点。在这一部分，我们将为`Order`管理器中实现的每个方法创建端点。

对于其中一些端点，我们将使用Django REST框架。使用Django REST框架的优势在于该框架包含了许多开箱即用的功能。它具有不同的身份验证方法，对对象的序列化非常强大，我最喜欢的是它将为您提供一个Web界面，您可以在其中浏览API，还包含了大量的基类和混合类，当您需要创建基于类的视图时。

所以，让我们马上开始吧！

在这一点上，我们需要做的第一件事是为我们模型的实体创建序列化程序类，`Order`，`OrderCustomer`和`OrderItem`。

继续在主`app`目录中创建一个名为`serializers.py`的文件，并让我们从添加一些导入语句开始：

```py
import functools

from rest_framework import serializers

from .models import Order, OrderItems, OrderCustomer
```

我们首先从标准库中导入`functools`模块；然后，我们从`rest_framework`模块中导入序列化程序。我们将使用它来创建我们的模型序列化程序。最后，我们将导入我们将用来创建`序列化程序`的模型，`Order`，`OrderItems`和`OrderCustomer`。

我们要创建的第一个序列化程序是`OrderCustomerSerializer`：

```py
class OrderCustomerSerializer(serializers.ModelSerializer):
    class Meta:
        model = OrderCustomer
        fields = ('customer_id', 'email', 'name', )
```

`OrderCustomerSerializer`继承自`ModelSerializer`，它非常简单；它只是定义了一些类元数据。我们将设置模型，`OrderCustomer`，还有将包含一个包含我们要序列化的字段的元组的属性字段。

然后，我们创建`OrderItemSerializer`：

```py
class OrderItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = OrderItems
        fields = ('name', 'price_per_unit', 'product_id', 'quantity', )
```

`OrderItemSerializer`与`OrderCustomerSerializer`非常相似。该类也继承自`ModelSerializer`，并定义了一些元数据属性。第一个是模型，我们将其设置为`OrderItems`，然后是包含我们要序列化的每个模型字段的元组的字段。

我们要创建的最后一个序列化程序是`OrderSerializer`，所以让我们从定义一个名为`OrderSerializer`的类开始：

```py
class OrderSerializer(serializers.ModelSerializer):
    items = OrderItemSerializer(many=True)
    order_customer = OrderCustomerSerializer()
    status = serializers.SerializerMethodField()
```

首先，我们定义两个属性。`items`属性设置为`OrderItemSerializer`，这意味着当我们需要序列化JSON数据时，它将使用该序列化程序，当我们想要添加新订单时。`items`属性指的是订单包含的商品。在这里，我们只使用一个关键字参数`(many=True)`。这将告诉你，序列化程序的items将是一个数组。

状态字段有点特殊；如果你还记得`Order`模型中的状态字段，它被定义为`ChoiceField`。当我们在数据库中保存订单时，该字段将存储值`1`，如果订单状态为`Received`，则存储值`2`，如果状态为`Processing`，依此类推。当我们的API的消费者调用端点获取订单时，他们将对状态的名称感兴趣，而不是数字。

因此，解决这个问题的方法是将字段定义为`SerializeMethodField`，然后我们将创建一个名为`get_status`的函数，它将返回订单状态的显示名称。我们很快就会看到`get_status`方法的实现是什么样子的。

我们还定义了`order_customer`属性，它设置为`OrderCustomerSerializer`，这意味着在尝试添加新订单时，框架将使用`OrderCustomerSerializer`类来反序列化我们发送的JSON对象。

然后，我们定义一个`Meta`类，以便我们可以向序列化器类添加一些元数据信息：

```py
    class Meta:
        depth = 1
        model = Order
        fields = ('items', 'total', 'order_customer',
                  'created_at', 'id', 'status', )
```

第一个属性`depth`指定了在序列化之前应该遍历的关系深度。在这种情况下，它设置为`1`，因为在获取订单对象时，我们还希望获取有关客户和商品的信息。与其他序列化器一样，我们将模型设置为`Order`，并且fields属性指定了哪些字段将被序列化和反序列化。

然后，我们实现`get_status`方法：

```py
    def get_status(self, obj):
        return obj.get_status_display()
```

这是一个将为`ChoiceField`状态获取显示值的方法。这将覆盖默认行为，并返回`get_status_display()`函数的结果。

`_created_order_item`方法只是一个辅助方法，我们将使用它来创建和准备订单项对象，然后执行批量插入操作：

```py
    def _create_order_item(self, item, order):
        item['order'] = order
        return OrderItems(**item)
```

在这里，我们将获得两个参数。第一个参数将是一个包含有关`OrderItem`的数据的字典，以及一个类型为`Order`的对象的`order`参数。首先，我们更新传递给第一个参数的字典，添加`order`对象，然后我们调用`OrderItem`构造函数，将商品作为`item`字典的参数传递。

我马上就会向你展示它的用途。现在我们已经到了这个序列化器的核心，我们将实现`create`方法，这将是一个在每次调用序列化器的`save`方法时自动调用的方法：

```py
def create(self, validated_data):
    validated_customer = validated_data.pop('order_customer')
    validated_items = validated_data.pop('items')

    customer = OrderCustomer.objects.create(**validated_customer)

    validated_data['order_customer'] = customer
    order = Order.objects.create(**validated_data)

    mapped_items = map(
        functools.partial(
        self._create_order_item, order=order), validated_items
    )

    OrderItems.objects.bulk_create(mapped_items)

    return order
```

因此，当调用`save`方法时，`create`方法将被自动调用，并将`validated_data`作为参数。`validated_date`是经过验证的、反序列化的订单数据。它看起来类似于以下数据：

```py
{
    "items": [
        {
            "name": "Prod 001",
            "price_per_unit": 10,
            "product_id": 1,
            "quantity": 2
        },
        {
            "name": "Prod 002",
            "price_per_unit": 12,
            "product_id": 2,
            "quantity": 2
        }
    ],
    "order_customer": {
        "customer_id": 14,
        "email": "test@test.com",
        "name": "Test User"
    },
    "order_id": 1,
    "status": 4,
    "total": "190.00"
}
```

正如你所看到的，在这个JSON中，我们一次性传递了所有信息。这里，我们有`order`，`items`属性，它是订单项的列表，以及`order_customer`，其中包含提交订单的客户的信息。

由于我们必须分别创建这些对象，我们首先弹出`order_customer`和`items`，所以我们有三个不同的对象。第一个`validated_customer`将只包含与下订单的人相关的数据。`validated_items`对象将只包含订单每个商品相关的数据，最后，`validated_data`对象将只包含订单本身的数据。

拆分数据后，我们现在可以开始添加对象。我们首先创建一个`OrderCustomer`：

```py
customer = OrderCustomer.objects.create(**validated_customer)
```

然后，我们可以创建订单。`Order`有一个外键字段叫做`order_customer`，它是与特定订单相关联的客户。我们需要在`validated_data`字典中创建一个名为`order_customer`的新项目，并将其值设置为我们刚刚创建的客户：

```py
validated_data['order_customer'] = customer
order = Order.objects.create(**validated_data)
```

最后，我们将添加`OrderItems`。现在，要添加订单项，我们需要做一些事情。`validated_items`变量是属于底层订单的商品列表，我们首先需要为每个商品设置订单，并为列表中的每个商品创建一个`OrderItem`对象。

执行此操作的不同方式。例如，您可以分两部分进行；首先遍历项目列表并设置订单属性，然后再次遍历列表并创建`OrderItem`对象。然而，那样并不那么优雅，是吗？

这里更好的方法是利用Python是一种多范式编程语言的事实，我们可以以更加函数式的方式解决这个问题：

```py
mapped_items = map(
    functools.partial(
        self._create_order_item, order=order), validated_items
)

OrderItems.objects.bulk_create(mapped_items)
```

在这里，我们利用了内置函数map之一。`map`函数将应用我指定的作为第一个参数的函数到作为第二个参数传递的可迭代对象上，然后返回一个包含结果的可迭代对象。

我们将作为map的第一个参数传递的函数称为`partial`，来自`functools`模块。`partial`函数是一个高阶函数，意味着它将返回另一个函数（第一个参数中的函数），并将参数和关键字参数添加到其签名中。在前面的代码中，它将返回`self._create_order_item`，第一个参数将是可迭代的`validated_items`中的一个项目。第二个参数是我们之前创建的订单。

之后，`mapped_items`的值应该包含一个`OrderItem`对象的列表，唯一剩下的事情就是调用`bulk_create`，它将为我们插入列表中的所有项目。

接下来，我们将创建视图。

# 创建视图

在创建视图之前，我们将创建一些辅助类和函数，这些类和函数将使视图中的代码更简单、更清晰。继续创建一个名为`view_helper.py`的文件，在主应用程序目录中，像往常一样，让我们从包含导入语句开始：

```py
from rest_framework import generics, status
from rest_framework.response import Response

from django.http import HttpResponse

from .exceptions import InvalidArgumentError
from .exceptions import OrderAlreadyCancelledError
from .exceptions import OrderAlreadyCompletedError

from .serializers import OrderSerializer
```

在这里，我们从Django REST Framework导入了一些东西，主要是通用的，其中包含了我们将用来创建自定义视图的通用视图类的定义。状态包含了所有HTTP状态码，在向客户端发送响应时非常有用。然后，我们导入了`Response`类，它将允许我们向客户端发送内容，可以以不同的内容类型呈现，例如JSON和XML。

然后，我们从Django中导入`HttpResponse`，以及在rest框架中的`Response`的等价物。

我们还导入了我们之前实现的所有自定义异常，这样我们就可以正确处理数据，并在出现问题时向客户端发送有用的错误消息。

最后，我们导入`OrderSerializer`，我们将用它来进行序列化、反序列化和验证模型。

我们将要创建的第一个类是`OrderListAPIBaseView`类，它将作为返回内容列表给客户端的所有视图的基类：

```py
class OrderListAPIBaseView(generics.ListAPIView):
    serializer_class = OrderSerializer
    lookup_field = ''

    def get_queryset(self, lookup_field_id):
        pass

    def list(self, request, *args, **kwargs):
        try:
            result = self.get_queryset(kwargs.get(self.lookup_field, None))
        except Exception as err:
            return Response(err, status=status.HTTP_400_BAD_REQUEST)

        serializer = OrderSerializer(result, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
```

`OrderListAPIBaseView`继承自通用的`ListAPIView`，它为我们提供了get和list方法，我们可以重写这些方法以添加满足我们要求的功能。

该类首先定义了两个属性；`serializer_class`，设置为`OrderSerializer`，以及`lookup_field`，在这种情况下我们将其设置为空字符串。我们将在子类中重写这个值。然后，我们定义了`get_queryset`方法，这也将在子类中被重写。

最后，我们实现了列表方法，它将首先运行`get_queryset`方法来获取将返回给用户的数据。如果发生错误，它将返回一个状态为`400`（`BAD REQUEST`）的响应，否则，它将使用`OrderSerializer`来序列化数据。`result`参数是`get_queryset`方法返回的`QuerySet`结果，`many`关键字参数告诉序列化器我们将序列化一个项目列表。

当数据被正确序列化时，我们将发送一个状态为`200`（OK）的响应，其中包含查询的结果。

这个基类的想法是，所有子类只需要实现`get_queryset`方法，这将使视图类保持简洁整洁。

现在，我们将添加一个函数，它将帮助我们执行`POST`请求的方法。让我们继续添加一个名为`set_status_handler`的函数：

```py
def set_status_handler(set_status_delegate):
    try:
        set_status_delegate()
    except (
            InvalidArgumentError,
            OrderAlreadyCancelledError,
            OrderAlreadyCompletedError) as err:
        return HttpResponse(err, status=status.HTTP_400_BAD_REQUEST)

    return HttpResponse(status=status.HTTP_204_NO_CONTENT)
```

这个函数非常简单；它只会将一个函数作为参数。运行该函数；如果发生异常之一，它将向客户端返回一个`400`（`BAD REQUEST`）响应，否则，它将返回一个`204`（`NO CONTENT`）响应。

# 添加视图

现在，是时候开始添加视图了！打开主`app`目录中的`views.py`文件，让我们添加一些导入语句：

```py
from django.http import HttpResponse
from django.shortcuts import get_object_or_404

from rest_framework import generics, status
from rest_framework.response import Response

from .models import Order
from .status import Status
from .view_helper import OrderListAPIBaseView
from .view_helper import set_status_handler
from .serializers import OrderSerializer
```

首先，我们将从`django.http`模块导入`HttpReponse`，并从`django.shortcuts`模块导入`get_object_or_404`。后者只是一个帮助函数，它将获取一个对象，如果找不到它，将返回状态码为`440`（`NOT FOUND`）的响应。

然后，我们导入generics以创建通用视图和状态，并从`rest_framework`中导入`Response`类。

最后，我们导入一些模型、帮助方法和函数，以及我们将在视图中使用的序列化器。

我们应该准备开始创建视图了。让我们创建一个视图，它将获取给定客户的所有订单：

```py
class OrdersByCustomerView(OrderListAPIBaseView):
    lookup_field = 'customer_id'

    def get_queryset(self, customer_id):
        return Order.objects.get_all_orders_by_customer(customer_id)
```

很好！因此，我们创建了一个从基类（`OrderListAPIBaseView`）继承的类，我们在`view_helpers.py`中创建了这个基类，由于我们已经实现了列表方法，因此我们在这里需要实现的唯一方法是`get_queryset`。`get_queryset`方法以`customer_id`作为参数，并简单地调用我们在`Order`模型管理器中创建的`get_all_orders_by_customer`，传递`customer_id`。

我们还定义了`lookup_field`的值，它将用于获取传递给基类列表方法的`kwargs`的关键字参数的值。

让我们再添加两个视图来获取未完成和完成的订单：

```py
class IncompleteOrdersByCustomerView(OrderListAPIBaseView):
    lookup_field = 'customer_id'

    def get_queryset(self, customer_id):
        return Order.objects.get_customer_incomplete_orders(
            customer_id
        )

class CompletedOrdersByCustomerView(OrderListAPIBaseView):
    lookup_field = 'customer_id'

    def get_queryset(self, customer_id):
        return Order.objects.get_customer_completed_orders(
            customer_id
        )
```

与我们实现的第一个视图基本相同，我们定义了`lookup_field`并重写了`get_queryset`以调用`Order`模型管理器中的适当方法。

现在，我们将添加一个视图，当给定特定状态时，将获取订单列表：

```py
class OrderByStatusView(OrderListAPIBaseView):
    lookup_field = 'status_id'

    def get_queryset(self, status_id):
        return Order.objects.get_orders_by_status(
            Status(status_id)
        )
```

正如你在这里所看到的，我们将`lookup_field`定义为`status_id`，并重写`get_queryset`以调用`get_orders_by_status`，传递状态值。

在这里，我们使用`Status`（`status_id`），因此我们传递`Enum`项而不仅仅是ID。

到目前为止，我们实现的所有视图都只接受`GET`请求，并且将返回订单列表。现在，我们将实现一个支持`POST`请求的视图，以便能够接收新订单：

```py
class CreateOrderView(generics.CreateAPIView):

    def post(self, request, *arg, **args):
        serializer = OrderSerializer(data=request.data)

        if serializer.is_valid():
            order = serializer.save()
            return Response(
                {'order_id': order.id},
                status=status.HTTP_201_CREATED)

        return Response(status=status.HTTP_400_BAD_REQUEST)
```

现在，这个类与我们创建的前几个类有些不同，基类是通用的。`CreateAPIView`为我们提供了一个`post`方法，因此我们重写该方法以添加我们需要的逻辑。首先，我们获取请求的数据并将其作为参数传递给`OrderSerializer`类；它将对数据进行反序列化并将其设置为序列化器变量。然后，我们调用`is_valid()`方法，它将验证接收到的数据。如果请求的数据无效，我们返回一个`400`响应（`BAD REQUEST`），否则，我们继续调用`save()`方法。这个方法将在序列化器上内部调用`create`方法，并且它将创建新订单以及新订单的客户和订单项目。如果一切顺利，我们将返回一个`202`响应（`CREATED`），并附上新创建订单的ID。

现在，我们将创建三个函数，用于处理订单取消、设置下一个订单状态，以及最后，设置特定订单的状态：

```py
def cancel_order(request, order_id):
    order = get_object_or_404(Order, order_id=order_id)

    return set_status_handler(
        lambda: Order.objects.cancel_order(order)
    )

def set_next_status(request, order_id):
    order = get_object_or_404(Order, order_id=order_id)

    return set_status_handler(
        lambda: Order.objects.set_next_status(order)
    )

def set_status(request, order_id, status_id):
    order = get_object_or_404(Order, order_id=order_id)

    try:
        status = Status(status_id)
    except ValueError:
        return HttpResponse(
            'The status value is invalid.',
            status=status.HTTP_400_BAD_REQUEST)

    return set_status_handler(
        lambda: Order.objects.set_status(order, status)
    )
```

正如您所看到的，我们在这里没有使用Django REST框架的基于类的视图。我们只是使用常规函数。第一个函数`cancel_order`接收两个参数——请求和`order_id`。我们首先使用快捷函数`get_object_or_404`。`get_object_or_404`函数会在无法找到与第二个参数中传递的条件匹配的对象时返回`404`响应（`未找到`）。否则，它将返回该对象。

然后，我们使用了我们在`view_helpers.py`文件中实现的辅助函数`set_status_handler`。这个函数接收另一个函数作为参数。因此，我们传递了一个将执行我们想要的`Order`模型管理器中的方法的`lambda`函数。在这种情况下，当执行`lambda`函数时，它将执行我们在`Order`模型管理器中定义的`cancel_order`方法，传递我们想要取消的订单。

`set_next_status`函数非常类似，但是我们将在`lambda`函数内部调用`cancel_order`，而不是调用`set_next_status`，传递我们想要设置为下一个状态的订单。

`set_status`函数包含了一些更多的逻辑，但它也很简单。这个函数将接收两个参数，`order_id`和`status_id`。首先，我们获取订单对象，然后使用`status_id`查找状态。如果状态不存在，将引发`ValueError`异常，然后我们返回`400`响应（`错误请求`）。否则，我们调用`set_status_handle`，传递一个`lambda`函数，该函数将执行`set_status`函数，传递订单和状态对象。

# 设置服务URL

现在我们已经将所有视图放在了适当的位置，是时候开始设置我们的订单服务用户可以调用以获取和修改订单的URL了。让我们继续打开主`app`目录中的`urls.py`文件；首先，我们需要导入所有要使用的视图类和函数：

```py
from .views import (
    cancel_order,
    set_next_status,
    set_status,
    OrdersByCustomerView,
    IncompleteOrdersByCustomerView,
    CompletedOrdersByCustomerView,
    OrderByStatusView,
    CreateOrderView,
)
```

太好了！现在，我们可以开始添加URL：

```py
urlpatterns = [
    path(
        r'order/add/',
        CreateOrderView.as_view()
    ),
    path(
        r'customer/<int:customer_id>/orders/get/',
        OrdersByCustomerView.as_view()
    ),
    path(
        r'customer/<int:customer_id>/orders/incomplete/get/',
        IncompleteOrdersByCustomerView.as_view()
    ),
    path(
        r'customer/<int:customer_id>/orders/complete/get/',
        CompletedOrdersByCustomerView.as_view()
    ),
    path(
        r'order/<int:order_id>/cancel',
        cancel_order
    ),
    path(
        r'order/status/<int:status_id>/get/',
        OrderByStatusView.as_view()
    ),
    path(
        r'order/<int:order_id>/status/<int:status_id>/set/',
        set_status
    ),
    path(
        r'order/<int:order_id>/status/next/',
        set_next_status
    ),
]
```

要添加新的URL，我们需要使用`path`函数来传递第一个参数，即`URL`。第二个参数是在发送请求到由第一个参数指定的`URL`时将执行的函数。我们创建的每个URL都必须添加到`urlspatterns`列表中。请注意，Django 2简化了如何向URL添加参数。以前，您需要使用正则表达式；现在，您只需遵循`<type:param>`的表示法。

在我们尝试这个之前，我们必须打开`urls.py`文件，但这次是在订单目录中，因为我们需要包括我们刚刚创建的URL。

`urls.py`文件应该类似于这样：

```py
"""order URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1\. Add an import: from my_app import views
    2\. Add a URL to urlpatterns: path('', views.home, name='home')
Class-based views
    1\. Add an import: from other_app.views import Home
    2\. Add a URL to urlpatterns: path('', Home.as_view(), name='home')
Including another URLconf
    1\. Import the include() function: from django.urls import include, path
    2\. Add a URL to urlpatterns: path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
]
```

现在，我们希望我们在主应用程序中定义的所有URL都位于`/api/`下。为了实现这一点，我们唯一需要做的就是创建一个新的路由，并包括来自主应用程序的URL。在`urlpatterns`列表中添加以下代码：

```py
path('api/', include('main.urls')),
```

并且不要忘记导入`include`函数：

```py
from django.urls import include
```

将订单服务部署到AWS时，它不会是公共的；但是作为额外的安全措施，我们将为此服务启用令牌身份验证。

要调用服务的API，我们必须发送身份验证令牌。让我们继续启用它。在`order`目录中打开`settings.py`文件，并添加以下内容：

```py
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework.authentication.TokenAuthentication',
    )
}
```

您可以将其放在`INSTALLED_APPS`之后。

`DEFAULT_PERMISSION_CLASSES`函数定义了全局权限策略。在这里，我们将其设置为`rest_framework.permissions.IsAuthenticated`，这意味着它将拒绝任何未经授权的用户访问。

`DEFAULT_AUTHENTICATION_CLASSES`函数指定了全局身份验证模式。在这种情况下，我们将使用令牌身份验证。

然后，在`INSTALLED_APPS`中，我们需要包括`rest_framework.authtoken`。您的`INSTALLED_APPS`应该如下所示：

```py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'main',
    'rest_framework',
    'rest_framework.authtoken',
]
```

太好了！保存文件，并在终端上运行以下命令：

```py
python manage.py migrate
```

Django REST框架具有开箱即用的视图，因此用户可以调用并获取令牌。但是，为简单起见，我们将创建一个可以访问API的用户。然后，我们可以手动创建一个身份验证令牌，该令牌可用于对订单服务API进行请求。

让我们继续创建这个用户。使用以下命令启动服务：

```py
python manage.py runserver
```

然后浏览到`https://localhost:8000/admin`。

在身份验证和授权选项卡下，您将看到`Users`模型。单击添加并创建一个用户名为`api_user`的用户。创建用户后，返回管理首页，在`AUTH TOKEN`下，单击添加。在下拉菜单中选择`api_user`，然后单击保存。您应该会看到以下页面：

![](assets/edccbabd-f4f7-4bf9-9e8a-930fe3e07c42.png)

复制密钥，让我们创建一个小脚本，只需添加一个订单，以便我们可以测试API。

创建一个名为`send_order.py`的文件；它可以放在任何您想要的地方，只要您已激活虚拟环境，因为我们将使用requests包将订单发送到订单服务。将以下内容添加到`send_order.py`文件中：

```py
import json
import sys
import argparse
from http import HTTPStatus

import requests

def setUpData(order_id):
    data = {
        "items": [
            {
                "name": "Prod 001",
                "price_per_unit": 10,
                "product_id": 1,
                "quantity": 2
            },
            {
                "name": "Prod 002",
                "price_per_unit": 12,
                "product_id": 2,
                "quantity": 2
            }
        ],
        "order_customer": {
            "customer_id": 14,
            "email": "test@test.com",
            "name": "Test User"
        },
        "order_id": order_id,
        "status": 1,
        "total": "190.00"
    }

    return data

def send_order(data):

    response = requests.put(
        'http://127.0.0.1:8000/api/order/add/',
        data=json.dumps(data))

    if response.status_code == HTTPStatus.NO_CONTENT:
        print('Ops! Something went wrong!')
        sys.exit(1)

    print('Request was successfull')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Create a order for test')

    parser.add_argument('--orderid',
                        dest='order_id',
                        required=True,
                        help='Specify the the order id')

    args = parser.parse_args()

    data = setUpData(args.order_id)
    send_order(data)
```

太棒了！现在，我们可以启动开发服务器：

```py
python manage.py runserver
```

在另一个窗口中，我们将运行刚刚创建的脚本：

```py
python send_order.py --orderid 10
```

您可以看到以下结果：

![](assets/2a95e41c-4f91-4a4e-ab80-64223855e413.png)

什么？这里出了些问题，你能猜到是什么吗？请注意在我运行Django开发服务器的终端中打印的屏幕截图中的日志消息：

```py
[21/Jan/2018 09:30:37] "PUT /api/order/add/ HTTP/1.1" 401 58
```

好的，这里说服务器已收到了对`/api/order/add/`的PUT请求，这里需要注意的一件事是代码`401`表示`未经授权`。这意味着我们在`settings.py`文件中添加的设置运行正常。要调用API，我们需要进行身份验证，而我们正在使用令牌身份验证。

要为用户创建一个令牌，我们需要在Django管理UI中登录。在那里，我们将找到如下所示的`AUTH TOKEN`部分：

![](assets/2d36948d-637f-4279-8ae5-2ab99d1abc0a.png)

单击右侧的绿色加号。然后，您可以选择要为其创建令牌的用户，当您准备好时，单击保存。之后，您将看到已创建的令牌列表：

![](assets/20e4a65d-5b1a-4455-9ed6-8aad6945c683.png)

该密钥是您要在请求的**HEADER**中发送的密钥。

现在我们有了一个令牌，我们可以修改`send_order.py`脚本，并将令牌信息添加到请求中，因此在`send_order`函数的顶部添加以下代码：

```py
token = '744cf4f8bd628e62f248444a478ce06681cb8089'

headers = {
    'Authorization': f'Token {token}',
    'Content-type': 'application/json'
}
```

令牌变量是我们为用户`api_user`创建的令牌。要获取令牌，只需登录到Django管理UI，在`AUTH TOKEN`下，您将看到已创建的令牌。只需删除我在此处添加的令牌，并用在您的应用程序上为`api_user`生成的令牌替换它。

然后，我们需要在请求中发送头。更改以下代码：

```py
response = requests.put(
    'http://127.0.0.1:8000/api/order/add/',
    data=json.dumps(data))
```

将其替换为：

```py
response = requests.put(
    'http://127.0.0.1:8000/api/order/add/',
    headers=headers,
    data=json.dumps(data))
```

现在，我们可以转到终端并再次运行我们的代码。您应该看到类似于以下屏幕截图中显示的输出：

![](assets/b834857f-c578-4561-bf20-7c10a898da7a.png)

请注意，现在我们得到了以下日志消息：

```py
[21/Jan/2018 09:49:40] "PUT /api/order/add/ HTTP/1.1" 201 0
```

这意味着身份验证正常工作。继续花时间探索Django管理UI，并验证我们现在在数据库中创建了一个客户和一个订单以及一些商品。

让我们尝试一些其他端点，看看它们是否按预期工作。例如，我们可以获取刚刚创建的客户的所有订单。

您可以使用任何工具对端点进行小型测试。有一些非常方便的浏览器插件可以安装，或者如果您像我一样喜欢在终端上做所有事情，您可以使用cURL。或者，如果您想尝试使用Python构建一些东西，可以安装`httpie`包，使用pip命令`pip install httpie --upgrade --user`在本地目录下`./local/bin`安装`httpie`。所以，不要忘记将此目录添加到您的PATH中。我喜欢使用`httpie`而不是cURL，因为`httpie`显示了一个漂亮和格式化的JSON输出，这样我就可以更好地查看从端点返回的响应。

所以，让我们尝试我们创建的第一个`GET`端点：

```py
  http http://127.0.0.1:8000/api/customer/1/orders/get/ 'Authorization: Token 744cf4f8bd628e62f248444a478ce06681cb8089'
```

然后你应该看到以下输出：

```py
HTTP/1.1 200 OK
Allow: GET, HEAD, OPTIONS
Content-Length: 270
Content-Type: application/json
Date: Sun, 21 Jan 2018 10:03:00 GMT
Server: WSGIServer/0.2 CPython/3.6.2
Vary: Accept
X-Frame-Options: SAMEORIGIN

[
 {
 "items": [
 {
 "name": "Prod 001",
 "price_per_unit": 10,
 "product_id": 1,
 "quantity": 2
 },
 {
 "name": "Prod 002",
 "price_per_unit": 12,
 "product_id": 2,
 "quantity": 2
 }
 ],
 "order_customer": {
 "customer_id": 14,
 "email": "test@test.com",
 "name": "Test User"
 },
 "order_id": 10,
 "status": 1,
 "total": "190.00"
 }
]
```

完美！正如预期的那样。继续尝试其他端点！

接下来，我们要回到在线视频游戏商店并发送订单。

# 与在线游戏商店的集成

现在我们的服务已经运行起来了，我们准备完成[第7章](a8e0af3b-67d9-4649-986b-041d136af0e8.xhtml)中的Django在线视频游戏商店项目。我们不打算进行太多更改，但有两个改进我们要做：

+   目前，在在线视频游戏商店中，无法提交订单。我们网站的用户只能将商品添加到购物车中，查看和编辑购物车中的商品。我们将完成这个实现，并创建一个视图，以便我们可以提交订单。

+   我们将实现另一个视图，可以在其中查看订单历史记录。

所以，让我们开始吧！

我们要做的第一个变化是为我们在服务订单中创建的`api_user`添加身份验证令牌。我们还想要添加订单服务的基本URL，这样我们就可以更容易地构建我们需要执行请求的URL。在`gamestore`目录中的`settings.py`文件中添加这两个常量变量：

```py
ORDER_SERVICE_AUTHTOKEN = '744cf4f8bd628e62f248444a478ce06681cb8089'
ORDER_SERVICE_BASEURL = 'http://127.0.0.1:8001'
```

这段代码放在哪里都可以，但也许最好是放在文件的末尾。

我们接下来要做的变化是添加一个名为`OrderItem`的`namedtuple`，只是为了帮助我们准备订单数据，使其与订单服务期望的格式兼容。在`gamestore/main`目录中的`models.py`文件中添加`import`：

```py
from collections import namedtuple
```

模型文件的另一个变化是，我们将在`ShoppingCartManager`类中添加一个名为`empty`的新方法，这样当调用它时，它将删除所有购物车中的商品。在`ShoppingCartManager`类中添加以下方法：

```py
def empty(self, cart):
    cart_items = ShoppingCartItem.objects.filter(
        cart__id=cart.id
    )

    for item in cart_items:
        item.delete()
```

在文件末尾，让我们创建`namedtuple`：

```py
OrderItem = namedtuple('OrderItem', 
                         'name price_per_unit product_id quantity')
```

接下来，我们要更改`cart.html`模板。找到`send order`按钮：

```py
<button class='btn btn-primary'>
  <i class="fa fa-check" aria-hidden="true"></i>
  &nbsp;SEND ORDER
</button>
```

用以下内容替换它：

```py
<form action="/cart/send">
  {% csrf_token %}
  <button class='btn btn-primary'>
    <i class="fa fa-check" aria-hidden="true"></i>
    &nbsp;SEND ORDER
  </button>
</form>
```

很好！我们刚刚在按钮周围创建了一个表单，并在表单中添加了跨站点请求伪造令牌，这样当我们点击按钮时，它将发送一个请求到`cart/send`。

让我们添加新的URL。在主`app`目录中打开`urls.py`文件，然后添加两个新的URL：

```py
path(r'cart/send', views.send_cart),
path(r'my-orders/', views.my_orders),
```

您可以将这两个URL定义放在`/cart/`URL的定义之后。

打开`views.py`文件并添加一些新的导入：

```py
import json
import requests
from http import HTTPStatus
from django.core.serializers.json import DjangoJSONEncoder
from gamestore import settings
```

然后，我们添加一个函数，将帮助我们将订单数据序列化为要发送到订单服务的格式：

```py
def _prepare_order_data(cart):

    cart_items = ShoppingCartItem.objects.values_list(
        'game__name',
        'price_per_unit',
        'game__id',
        'quantity').filter(cart__id=cart.id)

    order = cart_items.aggregate(
        total_order=Sum(F('price_per_unit') * F('quantity'),
                        output_field=DecimalField(decimal_places=2))
    )

    order_items = [OrderItem(*x)._asdict() for x in cart_items]

    order_customer = {
        'customer_id': cart.user.id,
        'email': cart.user.email,
        'name': f'{cart.user.first_name} {cart.user.last_name}'
    }

    order_dict = {
        'items': order_items,
        'order_customer': order_customer,
        'total': order['total_order']
    }

    return json.dumps(order_dict, cls=DjangoJSONEncoder)
```

现在，我们还有两个视图要添加，第一个是`send_order`：

```py
@login_required
def send_cart(request):
    cart = ShoppingCart.objects.get(user_id=request.user.id)

    data = _prepare_order_data(cart)

    headers = {
        'Authorization': f'Token {settings.ORDER_SERVICE_AUTHTOKEN}',
        'Content-type': 'application/json'
    }

    service_url = f'{settings.ORDER_SERVICE_BASEURL}/api/order/add/'

    response = requests.post(
        service_url,
        headers=headers,
        data=data)

    if HTTPStatus(response.status_code) is HTTPStatus.CREATED:
        request_data = json.loads(response.text)
        ShoppingCart.objects.empty(cart)
        messages.add_message(
            request,
            messages.INFO,
            ('We received your order!'
             'ORDER ID: {}').format(request_data['order_id']))
    else:
        messages.add_message(
            request,
            messages.ERROR,
            ('Unfortunately, we could not receive your order.'
             ' Try again later.'))

    return HttpResponseRedirect(reverse_lazy('user-cart'))
```

接下来是`my_orders`视图，这将是返回订单历史记录的新视图：

```py
@login_required
def my_orders(request):
    headers = {
        'Authorization': f'Token {settings.ORDER_SERVICE_AUTHTOKEN}',
        'Content-type': 'application/json'
    }

    get_order_endpoint = f'/api/customer/{request.user.id}/orders/get/'
    service_url = f'{settings.ORDER_SERVICE_BASEURL}{get_order_endpoint}'

    response = requests.get(
        service_url,
        headers=headers
    )

    if HTTPStatus(response.status_code) is HTTPStatus.OK:
        request_data = json.loads(response.text)
        context = {'orders': request_data}
    else:
        messages.add_message(
            request,
            messages.ERROR,
            ('Unfortunately, we could not retrieve your orders.'
             ' Try again later.'))
        context = {'orders': []}

    return render(request, 'main/my-orders.html', context)
```

我们需要创建`my-orders.html`文件，这将是由`my_orders`视图呈现的模板。在`main/templates/main/`目录中创建一个名为`my-orders.html`的新文件，内容如下：

```py
{% extends 'base.html' %}

{% block 'content' %}

<h3>Order history</h3>

{% for order in orders %}

<div class="order-container">
  <div><strong>Order ID:</strong> {{order.id}}</div>
  <div><strong>Create date:</strong> {{ order.created_at }}</div>
  <div><strong>Status:</strong> <span class="label label-success">{{order.status}}</span></div>
  <div class="table-container">
    <table class="table table-striped">
      <thead>
        <tr>
          <th>Product name</th>
          <th>Quantity</th>
          <th>Price per unit</th>
        </tr>
      </thead>
      <tbody>
        {% for item in order.items %}
        <tr>
          <td>{{item.name}}</td><td>{{item.quantity}}</td>  
          <td>${{item.price_per_unit}}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  <div><strong>Total amount:</strong>{{order.total}}</div>
  <hr/>
</div>
{% endfor %}
{% endblock %}
```

这个模板非常基础；它只是循环订单，然后循环商品并构建一个包含商品信息的HTML表格。

我们需要在`site.css`中做一些更改，这是在线视频游戏商店的自定义样式。打开`static/styles`文件夹中的`site.css`文件，让我们做一些修改。首先，找到以下代码，如下所示：

```py
.nav.navbar-nav .fa-home,
.nav.navbar-nav .fa-shopping-cart {
    font-size: 1.5em;
}
```

用以下内容替换它：

```py
.nav.navbar-nav .fa-home,
.nav.navbar-nav .fa-shopping-cart,
.nav.navbar-nav .fa-truck {
    font-size: 1.5em;
}
```

在此文件的末尾，我们可以添加特定于订单历史页面的样式：

```py
.order-container {
    border: 1px solid #000;
    margin: 20px;
    padding: 10px;
}
```

现在，我们将添加一个菜单选项，该选项将是指向新的`my orders`页面的链接。在应用程序`root`目录中的`templates`目录中打开`base.html`文件，并找到菜单选项`CART`：

```py
<li>
  <a href="/cart/">
    <i class="fa fa-shopping-cart" aria-hidden="true"></i> CART
  </a>
</li>
```

在`</li>`标签结束后，添加以下代码：

```py
<li>
  <a href="/my-orders/">
    <i class="fa fa-truck" aria-hidden="true"></i> ORDERS
  </a>
</li>
```

最后，我们要做的最后一项更改是改进我们在UI中显示的错误消息的布局。找到`base.html`文件末尾的此代码：

```py
{% if messages %}
  {% for message in messages %}    
    {{message}}
    </div>
  {% endfor %}
{% endif %}
```

用以下代码替换它：

```py
{% if messages %}
  {% for message in messages %}
    {% if message.tags == 'error' %}
      <div class="alert alert-danger" role="alert">
    {% else %}
      <div class="alert alert-info" role="alert">
    {% endif %}
    {{message}}
    </div>
  {% endfor %}
{% endif %}
```

# 测试集成

我们已经准备就绪。现在，我们需要启动网站和服务，以便验证一切是否正常工作。

需要记住的一件事是，为了测试，我们需要在不同的端口上运行Django应用程序。我们可以使用默认端口`800`运行网站（游戏在线商店），对于订单服务，我们可以使用端口`8001`。

打开两个终端；在一个终端中，我们将启动在线视频游戏商店：

```py
python manage.py runserver
```

然后，在第二个终端上，我们将启动订单服务：

```py
python manage.py runserver 127.0.0.1:8001
```

太棒了！打开浏览器，转到`http://localhost:8000`并使用我们的凭据登录。登录后，您会注意到一些不同之处。现在，顶部菜单中有一个名为`ORDERS`的新选项。它应该是空的，所以继续向购物车中添加一些项目。完成后，转到购物车视图并单击发送订单按钮。

如果一切顺利，您应该在页面顶部看到通知，如下所示：

![](assets/23172db2-3235-40a2-8e5d-7fa0e02142bd.png)

太棒了！它正如预期的那样工作。请注意，在将订单发送到订单服务后，购物车也被清空了。

现在，点击顶部菜单上的`ORDERS`选项，您应该看到我们刚刚提交的订单：

![](assets/9a3ff13e-c15e-4cce-8cfa-7d66c14e0787.png)

# 部署到AWS

现在，是时候向世界展示我们迄今为止所做的工作了。

我们将在Amazon Web服务的EC2实例上部署gamestore Django应用程序和订单服务。

本节不涉及配置虚拟私有云、安全组、路由表和EC2实例。Packt有许多关于这个主题的优秀书籍和视频可供参考。

相反，我们将假设您已经设置好了环境，并专注于：

+   部署应用程序

+   安装所有必要的依赖项

+   安装和使用`gunicorn`

+   安装和配置`nginx`

我的AWS设置非常简单，但绝对适用于更复杂的设置。现在，我有一个VPC，一个子网和两个EC2实例（`gamestore`和order-service）。请参阅以下截图：

![](assets/e36b74d3-ac74-4979-851d-878b9b9f5b7b.png)

我们可以从`gamestore`应用程序开始；通过ssh连接到您希望部署游戏在线应用程序的EC2实例。请记住，要在这些实例中的一个中进行`ssh`，您需要拥有`.pem`文件：

```py
ssh -i gamestore-keys.pem ec2-user@35.176.16.157
```

我们将首先更新在该计算机上安装的任何软件包；虽然不是必需的，但这是一个很好的做法，因为其中一些软件包可能具有安全修复和性能改进，您可能希望在安装时拥有这些改进。Amazon Linux使用`yum`软件包管理器，因此我们运行以下命令：

```py
sudo yum update
```

只需对任何需要更新的软件包回答“是”`y`。

这些EC2实例默认未安装Python，因此我们也需要安装它：

```py
sudo yum install python36.x86_64 python36-pip.noarch python36- setuptools.noarch
```

我们还需要安装`nginx`：

```py
sudo yum install nginx
```

然后，我们安装我们的项目依赖项：

```py
sudo pip-3.6 install django requests pillow gunicorn
```

完美！现在，我们可以复制我们的应用程序，退出此实例，并从我们的本地机器上运行以下命令：

```py
scp -R -i gamestore-keys.pem ./gamestore ec2-user@35.176.16.157:~/gamestore
```

此命令将递归地将本地机器上`gamestore`目录中的所有文件复制到EC2实例中我们的主目录中。

# 修改settings.py文件

这里有一件事情我们需要改变。在`settings.py`文件中，有一个名为`ALLOWED_HOSTS`的列表，在我们创建Django项目时为空。我们需要添加我们将部署应用程序的EC2的IP地址；在我的情况下，它将是：

```py
ALLOWED_HOSTS=["35.176.16.157"]
```

我们还需要更改文件末尾定义的`ORDER_SERVICE_BASEURL`。它需要是我们将部署到订单服务的实例的地址。在我的情况下，IP是`35.176.194.15`，所以我的变量看起来像这样：

```py
ORDER_SERVICE_BASEURL = "http://35.176.194.15"
```

我们将创建一个文件夹来保存应用程序，因为将应用程序运行在`ec2-user`文件夹中不是一个好主意。因此，我们将在`root`目录中创建一个名为`app`的新文件夹，并将`gamestore`目录复制到新创建的目录中：

```py
sudo mkdir /app && sudo cp -R ./gamestore /app/
```

我们还需要设置该目录的当前权限。当安装`nginx`时，它还会创建一个`nginx`用户和一个组。因此，让我们更改整个文件夹的所有权：

```py
cd / && sudo chown -R nginx:nginx ./gamestore
```

最后，我们将设置`nginx`，编辑`/etc/nginx/nginx.conf`文件，在`service`下添加以下配置：

```py
location / {
  proxy_pass http://127.0.0.1:8000;
  proxy_set_header Host $host;
  proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}

location /static {
  root /app/gamestore;
}
```

我们需要重新启动`nginx`服务，以便服务反映我们刚刚做的更改：

```py
sudo service nginx restart
```

最后，我们转到`application`文件夹：

```py
cd /app/gamestore
```

使用`gunicorn`启动应用程序。我们将以`nginx`用户的身份启动应用程序：

```py
sudo gunicorn -u nginx gamestore.wsgi
```

现在，我们可以浏览到该网站。您不需要指定端口`8000`，因为`nginx`将把从端口`80`进来的请求路由到`127.0.0.1:8000`。

# 部署订单服务

部署订单服务与`gamestore`项目基本相同，唯一的区别是我们将安装不同的Python依赖项并将应用程序部署到不同的目录。所以，让我们开始吧。

您几乎可以重复直到安装`nginx`步骤的所有步骤。还要确保您从现在开始使用另一个EC2实例的弹性IP地址。

安装`nginx`后，我们可以安装订单服务的依赖项：

```py
sudo pip-3.6 install django djangorestframework requests
```

现在我们可以复制项目文件。转到您拥有服务目录的目录，并运行此命令：

```py
scp -R -i order-service-keys.pem ./order ec2-user@35.176.194.15:~/gamestore
```

与`gamestore`一样，我们还需要编辑`settings.py`文件并添加我们的EC2实例弹性IP：

```py
ALLOWED_HOSTS=["35.176.194.15"]
```

我们还将在`root`目录中创建一个文件夹，以便项目不会留在`ec2-user`的主目录中：

```py
sudo mkdir /srv && sudo cp -R ./order /srv/
```

让我们也更改整个目录的所有者：

```py
cd / && sudo chown -R nginx:nginx ./order
```

让我们编辑`/etc/nginx/nginx.conf`文件，在`service`下添加以下配置：

```py
location / {
  proxy_pass http://127.0.0.1:8000;
  proxy_set_header Host $host;
  proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

这次，我们不需要配置静态文件夹，因为订单服务没有像图像、模板、JS或CSS文件这样的东西。

重新启动`nginx`服务：

```py
sudo service nginx restart
```

转到服务的目录：

```py
cd /srv/order
```

并使用`gunicorn`启动应用程序。我们将以`nginx`用户的身份启动应用程序：

```py
sudo gunicorn -u nginx order.wsgi
```

最后，我们可以浏览到`gamestore`部署的地址，您应该看到网站正在运行。

浏览到该网站，您将看到第一页。所有产品都在加载，登录和注销部分也正常工作。这是我的系统的截图：

![](assets/9660d0a6-ccf9-4d45-981a-f541742d1c64.png)

如果您浏览使用订单服务的视图，例如订单部分，您可以验证一切是否正常运行，如果您在网站上下了任何订单，您应该在这里看到订单列表，如下面的截图所示：

![](assets/b57770dd-21a8-4748-81ae-5c8326529984.png)

# 总结

在本章中，我们涵盖了许多主题；我们已经构建了订单服务，负责接收我们在上一章开发的网络应用程序的订单。订单服务还提供其他功能，例如能够更新订单状态并使用不同的标准提供订单信息。

这个微服务是我们在上一章开发的网络应用程序的延伸，接下来的章节中，我们将通过添加无服务器函数来进一步扩展它，以便在成功接收订单时通知我们应用程序的用户，以及当订单状态变更为已发货时通知他们。
