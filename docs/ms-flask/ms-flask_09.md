# 第九章：使用 Celery 创建异步任务

在创建 Web 应用程序时，保持请求处理时间在 50 毫秒左右以下是至关重要的。由于大部分响应时间都被等待用户连接所占据，额外的处理时间可能会挂起服务器。应该避免服务器上的任何额外处理。然而，在 Web 应用程序中，有几个操作可能需要花费超过几秒钟的时间，特别是涉及复杂的数据库操作或图像处理时。为了保护用户体验，将使用名为 Celery 的任务队列将这些操作移出 Flask 进程。

# Celery 是什么？

**Celery**是用 Python 编写的异步任务队列。Celery 通过 Python 多进程库*并发*运行任务，这些任务是用户定义的函数。Celery 接收消息，告诉它从**代理**开始任务，通常称为消息队列，如下图所示：

![Celery 是什么？](img/B03929_09_01.jpg)

**消息队列**是一个专门设计用于在生产者进程和消费者进程之间发送数据的系统。**生产者进程**是创建要发送到队列中的消息的任何程序，**消费者进程**是从队列中取出消息的任何程序。从生产者发送的消息存储在**先进先出**（**FIFO**）队列中，最旧的项目首先被检索。消息存储直到消费者接收消息，之后消息被删除。消息队列提供实时消息传递，而不依赖于轮询，即持续检查进程状态的过程。当消息从生产者发送时，消费者正在其连接到消息队列上*监听*新消息；消费者不会不断地联系队列。这种差异就像**AJAX**和**WebSockets**之间的差异；AJAX 需要与服务器保持不断的联系，而 WebSockets 只是一个持续的流。

可以用传统数据库替换消息队列。Celery 甚至内置了对 SQLAlchemy 的支持以实现这一点。然而，强烈不建议使用数据库作为 Celery 的代理。使用数据库代替消息队列需要消费者不断地轮询数据库以获取更新。此外，由于 Celery 使用多进程进行并发处理，大量读取的连接数量会迅速增加。在中等负载下，使用数据库需要生产者同时向数据库进行大量写入，而消费者正在读取。数据库不能有太多的连接同时进行读取、写入和更新相同的数据。当这种情况发生时，表通常会被锁定，所有其他连接都在等待每次写入完成后才能读取数据，反之亦然。更糟糕的是，这可能导致竞争条件，即并发事件更改和读取相同的资源，并且每个并发操作都使用过时版本的数据。特定于 Celery，这可能导致相同的操作针对相同的消息多次运行。

也可以使用消息队列作为代理和数据库来存储任务的结果。在前面的图表中，消息队列用于发送任务请求和任务结果。

然而，使用数据库存储任务的最终结果允许最终产品无限期地存储，而消息队列将在生产者接收数据后立即丢弃数据，如下图所示：

![Celery 是什么？](img/B03929_09_02.jpg)

这个数据库通常是一个键值 NoSQL 存储，以帮助处理负载。如果您计划对先前运行的任务进行分析，这将非常有用；否则，最好只使用消息队列。

甚至有一个选项可以完全丢弃任务的结果，而不返回任务的结果。这样做的缺点是生产者无法知道任务是否成功，但在较小的项目中通常足够。

对于我们的堆栈，我们将使用**RabbitMQ**作为消息代理。RabbitMQ 在所有主要操作系统上运行，并且非常简单设置和运行。Celery 还支持 RabbitMQ，无需任何额外的库，并且是 Celery 文档中推荐的消息队列。

### 注意

在撰写本文时，尚无法在 Python 3 中使用 RabbitMQ 与 Celery。您可以使用 Redis 代替 RabbitMQ。唯一的区别将是连接字符串。有关更多信息，请参见[`docs.celeryproject.org/en/latest/getting-started/brokers/redis.html`](http://docs.celeryproject.org/en/latest/getting-started/brokers/redis.html)。

# 设置 Celery 和 RabbitMQ

要使用`pip`安装 Celery，请运行以下命令：

```py
$ pip install Celery

```

我们还需要一个 Flask 扩展来帮助处理初始化 Celery：

```py
$ pip install Flask-Celery-Helper

```

Flask 文档指出，Flask 对 Celery 的扩展是不必要的。但是，在使用应用程序工厂组织应用程序时，使 Celery 服务器能够与 Flask 的应用程序上下文一起工作是很重要的。因此，我们将使用**Flask-Celery-Helper**来完成大部分工作。

接下来，需要安装 RabbitMQ。RabbitMQ 不是用 Python 编写的；因此，每个操作系统的安装说明都将不同。幸运的是，RabbitMQ 在[`www.rabbitmq.com/download.html`](https://www.rabbitmq.com/download.html)上为每个操作系统维护了详细的说明列表。

安装 RabbitMQ 后，打开终端窗口并运行以下命令：

```py
$ rabbitmq-server

```

这将启动一个带有用户名为 guest 和密码为 guest 的 RabbitMQ 服务器。默认情况下，RabbitMQ 只接受本地主机上的连接，因此这种设置对开发来说是可以的。

# 在 Celery 中创建任务

如前所述，Celery 任务只是执行一些操作的用户定义函数。但在编写任何任务之前，需要创建我们的 Celery 对象。这是 Celery 服务器将导入以处理运行和调度所有任务的对象。

至少，Celery 需要一个配置变量才能运行：与消息代理的连接。连接被定义为 URL，就像 SQLAlchemy 连接一样。后端，用于存储我们任务结果的地方，也被定义为 URL，如下面的代码所示：

```py
class DevConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///../database.db'
    CELERY_BROKER_URL = "amqp://guest:guest@localhost:5672//"
    CELERY_BACKEND = "amqp://guest:guest@localhost:5672//"
In the extensions.py file, the Celery class from Flask-Celery-Helper will be initialized:
```

```py
from flask.ext.celery import Celery
celery = Celery()
```

因此，为了使我们的 Celery 进程能够与数据库和任何其他 Flask 扩展一起工作，它需要在我们的应用程序上下文中工作。为了做到这一点，Celery 需要为每个进程创建我们应用程序的新实例。与大多数 Celery 应用程序不同，我们需要一个 Celery 工厂来创建应用程序实例并在其上注册我们的 Celery 实例。在顶级目录中的一个新文件中，与`manage.py`位于同一位置，命名为`celery_runner.py`，添加以下内容：

```py
import os
from webapp import create_app
from celery import Celery
from webapp.tasks import log

def make_celery(app):
    celery = Celery(
        app.import_name,
        broker=app.config['CELERY_BROKER_URL'],
        backend=app.config['CELERY_BACKEND_URL']
    )
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask

    return celery

env = os.environ.get('WEBAPP_ENV', 'dev')
flask_app = create_app(
    'webapp.config.%sConfig' % env.capitalize()
)
celery = make_celery(flask_app)
```

`make_celery`函数的作用是在 Python 的`with`块中包装对每个 Celery 任务的每次调用。这确保了对任何 Flask 扩展的每次调用都可以正常工作，因为它正在与我们的应用程序一起工作。还要确保不要将 Flask 应用程序实例命名为`app`，因为 Celery 会尝试导入任何名为`app`或`celery`的对象作为 Celery 应用程序实例。因此，将您的 Flask 对象命名为`app`将导致 Celery 尝试将其用作 Celery 对象。

现在，我们可以编写我们的第一个任务。这将是一个简单的任务，只是返回传递给它的任何字符串。在应用程序目录中的一个新文件中命名为`tasks.py`，添加以下内容：

```py
from webapp.extensions import celeryfrom webapp.extensions import celery
@celery.task()
def log(msg):
    return msg
```

现在，谜题的最后一部分是在新的终端窗口中运行 Celery 进程，称为**worker**。再次强调，这是将监听我们的消息代理以启动新任务的进程：

```py
$ celery worker -A celery_runner --loglevel=info

```

`loglevel`标志存在的原因是，您可以在终端窗口中看到任务已收到的确认以及其输出的可用性。

现在，我们可以向 Celery 工作进程发送命令。打开 `manage.py` shell 并导入 `log` 任务：

```py
>>> from webapp.tasks import log
>>> log("Message")
Message
>>> result = log.delay("Message")

```

该函数可以像调用其他函数一样调用；这样做将在当前进程中执行该函数。但是，在任务上调用 `delay` 方法将向工作进程发送消息，以使用给定的参数执行该函数。

在运行 Celery 工作进程的终端窗口中，您应该看到类似以下内容：

```py
Task tasks.log succeeded in 0.0005873600021s: 'Message'

```

对于任何异步任务，`ready` 方法可用于判断任务是否成功完成。如果为真，则可以使用 `get` 方法来检索任务的结果。

```py
>>> result.ready()
True
>>> result.get()
"Message"

```

`get` 方法会导致当前进程等待，直到 `ready` 函数返回 `True` 以检索结果。因此，在调用任务后立即调用 `get` 实质上使任务同步。因此，任务实际上很少返回值给生产者。绝大多数任务执行某些操作然后退出。

当在 Celery 工作进程上运行任务时，可以通过 `state` 属性访问任务的状态。这允许更细粒度地了解任务在工作进程中当前正在执行的操作。可用的状态如下：

+   `FAILURE`：任务失败，所有重试也失败

+   `PENDING`：任务尚未被工作进程接收

+   `RECEIVED`：任务已被工作进程接收，但尚未处理

+   `RETRY`：任务失败，正在等待重试

+   `REVOKED`：任务已停止

+   `STARTED`：工作进程已开始处理任务

+   `SUCCESS`：任务成功完成

在 Celery 中，如果任务失败，则任务可以使用 `retry` 方法重新调用自身，如下所示：

```py
@celery.task(bind=True)
def task(self, param):
    try:
        some_code
    except Exception, e:
        self.retry(exc=e)
```

装饰器函数中的 `bind` 参数告诉 Celery 将任务对象的引用作为函数的第一个参数传递。使用 `self` 参数，可以调用 `retry` 方法，该方法将使用相同的参数重新运行任务。可以将其他参数传递给函数装饰器，以更改任务的行为：

+   `max_retries`：这是任务在被声明为失败之前可以重试的最大次数。

+   `default_retry_delay`：这是在再次运行任务之前等待的时间（以秒为单位）。如果您预期导致任务失败的条件是短暂的，例如网络错误，那么最好将其保持在大约一分钟左右。

+   `rate_limit`：这指定在给定间隔内允许运行此任务的唯一调用总数。如果值是整数，则是每秒允许运行此任务的总数。该值也可以是形式为 *x/m* 的字符串，表示每分钟 *x* 个任务，或形式为 *x/h* 的字符串，表示每小时 *x* 个任务。例如，传入 *5/m* 将只允许每分钟调用此任务五次。

+   `time_limit`：如果指定，任务将在运行时间超过此秒数时被终止。

+   `ignore_result`：如果不使用任务的返回值，则不要将其发送回。

最好为每个任务指定所有这些内容，以避免任务不会运行的任何机会。

# 运行 Celery 任务

`delay` 方法是 `apply_async` 方法的简写版本，格式如下所示：

```py
task.apply_async(
    args=[1, 2],
    kwargs={'kwarg1': '1', 'kwarg2': '2'}
)
```

但是，`args` 关键字可以是隐式的：

```py
apply_async([1, 2], kwargs={'kwarg1': '1', 'kwarg2': '2'})
```

调用 `apply_async` 允许您在任务调用中定义一些额外的功能，这些功能在 `delay` 方法中无法指定。首先，`countdown` 选项指定工作进程在接收到任务后等待运行任务的时间（以秒为单位）：

```py
>>> from webapp.tasks import log
>>> log.apply_async(["Message"], countdown=600)

```

`countdown` 不能保证任务将在 `600` 秒后运行。`countdown` 只表示任务在 *x* 秒后准备处理。如果所有工作进程都忙于处理其他任务，则任务将不会立即运行。

`apply_async` 提供的另一个关键字参数是 `eta` 参数。`eta` 通过一个指定任务应该运行的确切时间的 Python `datetime` 对象传递。同样，`eta` 不可靠。

```py
>>> import datetime
>>> from webapp.tasks import log
# Run the task one hour from now
>>> eta = datetime.datetime.now() + datetime.timedelta(hours=1)
>>> log.apply_async(["Message"], eta=eta)

```

## Celery 工作流

Celery 提供了许多方法来将多个依赖任务分组在一起，或者并行执行多个任务。这些方法受到函数式编程语言中的语言特性的很大影响。然而，要理解这是如何工作的，我们首先需要了解签名。考虑以下任务：

```py
@celery.task()
def multiply(x, y):
    return x * y
```

让我们看看一个**签名**的实际操作以理解它。打开 `manage.py` shell：

```py
>>> from celery import signature
>>> from webapp.tasks import multiply
# Takes the same keyword args as apply_async
>>> signature('webapp.tasks.multiply', args=(4, 4) , countdown=10)
webapp.tasks.multiply(4, 4)
# same as above
>>> from webapp.tasks import multiply
>>> multiply.subtask((4, 4), countdown=10)
webapp.tasks.multiply(4, 4)
# shorthand for above, like delay in that it doesn't take
# apply_async's keyword args
>>> multiply.s(4, 4)
webapp.tasks.multiply(4, 4)
>>> multiply.s(4, 4)()
16
>>> multiply.s(4, 4).delay()

```

调用任务的签名，有时称为任务的**子任务**，会创建一个可以传递给其他函数以执行的函数。执行签名，就像示例中倒数第三行那样，会在当前进程中执行函数，而不是在工作进程中执行。

### 部分

任务签名的第一个应用是函数式编程风格的部分。**部分**是最初接受许多参数的函数；然而，对原始函数应用操作以返回一个新函数，因此前 *n* 个参数始终相同。一个例子是一个不是任务的 `multiply` 函数：

```py
>>> new_multiply = multiply(2)
>>> new_multiply(5)
10
# The first function is unaffected
>>> multiply(2, 2)
4

```

这是一个虚构的 API，但这与 Celery 版本非常接近：

```py
>>> partial = multiply.s(4)
>>> partial.delay(4)

```

工作窗口中的输出应该显示 **16**。基本上，我们创建了一个新函数，保存到部分中，它将始终将其输入乘以四。

### 回调

一旦任务完成，根据前一个任务的输出运行另一个任务是非常常见的。为了实现这一点，`apply_async` 函数有一个 `link` 方法：

```py
>>> multiply.apply_async((4, 4), link=log.s())

```

工作器输出应该显示 `multiply` 任务和 `log` 任务都返回 **16**。

如果您有一个不需要输入的函数，或者您的回调不需要原始方法的结果，则必须使用 `si` 方法将任务签名标记为不可变：

```py
>>> multiply.apply_async((4, 4), link=log.si("Message"))

```

**回调**可以用来解决现实世界的问题。如果我们想要在每次任务创建新用户时发送欢迎电子邮件，那么我们可以通过以下调用产生该效果：

```py
>>> create_user.apply_async(("John Doe", password), link=welcome.s())

```

部分和回调可以结合产生一些强大的效果：

```py
>>> multiply.apply_async((4, 4), link=multiply.s(4))

```

重要的是要注意，如果保存了此调用并在其上调用了 `get` 方法，则结果将是 **16** 而不是 **64**。这是因为 `get` 方法不会返回回调方法的结果。这将在以后的方法中解决。

### 组

`group` 函数接受一个签名列表，并创建一个可调用函数来并行执行所有签名，然后返回所有结果的列表：

```py
>>> from celery import group
>>> sig = group(multiply.s(i, i+5) for i in range(10))
>>> result = sig.delay()
>>> result.get()
[0, 6, 14, 24, 36, 50, 66, 84, 104, 126]

```

### 链

`chain` 函数接受任务签名，并将每个结果的值传递给链中的下一个值，返回一个结果如下：

```py
>>> from celery import chain
>>> sig = chain(multiply.s(10, 10), multiply.s(4), multiply.s(20))
# same as above
>>> sig = (multiply.s(10, 10) | multiply.s(4) | multiply.s(20))
>>> result = sig.delay()
>>> result.get()
8000

```

链和部分可以进一步发展。链可以用于在使用部分时创建新函数，并且链可以嵌套如下：

```py
# combining partials in chains
>>> func = (multiply.s(10) | multiply.s(2))
>>> result = func.delay(16)
>>> result.get()
200
# chains can be nested
>>> func = (
 multiply.s(10) | multiply.s(2) | (multiply.s(4) | multiply.s(5))
)
>>> result = func.delay(16)
>>> result.get()
800

```

### 和弦

`chord` 函数创建一个签名，将执行一组签名，并将最终结果传递给回调：

```py
>>> from celery import chord
>>> sig = chord(
 group(multiply.s(i, i+5) for i in range(10)),
 log.s()
)
>>> result = sig.delay()
>>> result.get()
[0, 6, 14, 24, 36, 50, 66, 84, 104, 126]

```

就像链接参数一样，回调不会随着 `get` 方法返回。

使用 `chain` 语法与组和回调自动创建一个和弦签名：

```py
# same as above
>>> sig = (group(multiply.s(i, i+5) for i in range(10)) | log.s())
>>> result = sig.delay()
>>> result.get()
[0, 6, 14, 24, 36, 50, 66, 84, 104, 126]

```

### 定期运行任务

Celery 还有能力定期调用任务。对于熟悉 ***nix** 操作系统的人来说，这个系统很像命令行实用程序 `cron`，但它的额外好处是在我们的源代码中定义，而不是在某个系统文件中。因此，当我们的代码准备发布到 第十三章 *部署 Flask 应用* 中时，更新将更容易。此外，所有任务都在应用上下文中运行，而由 `cron` 调用的 Python 脚本则不会。

要添加定期任务，请将以下内容添加到 `DevConfig` 配置对象中：

```py
import datetime
…

CELERYBEAT_SCHEDULE = {
    'log-every-30-seconds': {
        'task': 'webapp.tasks.log',
        'schedule': datetime.timedelta(seconds=30),
        'args': ("Message",)
    },
}
```

此`configuration`变量定义了`log`任务应该每 30 秒运行一次，并将`args`元组作为参数传递。任何`timedelta`对象都可以用来定义运行任务的间隔。

要运行周期性任务，需要另一个名为`beat`工作程序的专门工作程序。在另一个终端窗口中，运行以下命令：

```py
$ celery -A celery_runner beat

```

如果您现在观看主要的`Celery`工作程序中的终端输出，您应该每 30 秒看到一个日志事件。

如果您的任务需要以更具体的间隔运行，例如，每周二在 6 月的凌晨 3 点和下午 5 点？对于非常具体的间隔，有`Celery` `crontab`对象。

为了说明`crontab`对象如何表示间隔，以下是一些示例：

```py
>>> from celery.schedules import crontab
# Every midnight
>>> crontab(minute=0, hour=0)
# Once a 5AM, then 10AM, then 3PM, then 8PM
>>> crontab(minute=0, hour=[5, 10, 15, 20])
# Every half hour
>>> crontab(minute='*/30')
# Every Monday at even numbered hours and 1AM
>>> crontab(day_of_week=1, hour ='*/2, 1')

```

该对象具有以下参数：

+   `分钟`

+   小时

+   `星期几`

+   `每月的日期`

+   `月份`

这些参数中的每一个都可以接受各种输入。使用纯整数时，它们的操作方式与`timedelta`对象类似，但它们也可以接受字符串和列表。当传递一个列表时，任务将在列表中的每个时刻执行。当传递一个形式为**/x*的字符串时，任务将在模运算返回零的每个时刻执行。此外，这两种形式可以组合成逗号分隔的整数和除法的字符串。

# 监控 Celery

当我们的代码被推送到服务器时，我们的`Celery`工作程序将不会在终端窗口中运行，它将作为后台任务运行。因此，Celery 提供了许多命令行参数来监视您的`Celery`工作程序和任务的状态。这些命令采用以下形式：

```py
$ celery –A celery_runner <command>

```

查看工作程序状态的主要任务如下：

+   `状态`：这会打印运行的工作程序以及它们是否正常运行

+   `结果`：当传递一个任务 id 时，这显示任务的返回值和最终状态

+   `清除`：使用此命令，代理中的所有消息将被删除

+   `检查活动`：这将列出所有活动任务

+   `检查已安排`：这将列出所有已使用`eta`参数安排的任务

+   `检查已注册`：这将列出所有等待处理的任务

+   `检查统计`：这将返回一个字典，其中包含有关当前运行的工作程序和代理的统计信息

## 使用 Flower 进行基于 Web 的监控

**Flower**是一个基于 Web 的实时管理工具，用于 Celery。在 Flower 中，可以监视所有活动的，排队的和已完成的任务。Flower 还提供了关于每个图表在队列中停留的时间以及执行所需的时间和每个任务的参数的图表和统计信息。

要安装 Flower，请使用以下`pip`：

```py
$ pip install flower

```

要运行它，只需将`flower`视为`Celery`命令，如下所示：

```py
$ celery flower -A celery_runner --loglevel=info

```

现在，打开浏览器到`http://localhost:5555`。最好在任务运行时熟悉界面，因此转到命令行并输入以下内容：

```py
>>> sig = chord(
 group(multiply.s(i, i+5) for i in xrange(10000)),
 log.s()
)
>>> sig.delay()

```

您的工作程序现在将开始处理 10,000 个任务。在任务运行时浏览不同的页面，看看 Flower 在工作程序真正忙碌时如何与其交互。

# 创建一个提醒应用

让我们来看一些 Celery 中的真实例子。假设我们网站上的另一页现在需要一个提醒功能。用户可以创建提醒，将在指定时间发送电子邮件到指定位置。我们需要一个模型，一个任务，以及一种在每次创建模型时自动调用我们的任务的方法。

让我们从以下基本的 SQLAlchemy 模型开始：

```py
class Reminder(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    date = db.Column(db.DateTime())
    email = db.Column(db.String())
    text = db.Column(db.Text())

    def __repr__(self):
        return "<Reminder '{}'>".format(self.text[:20])
```

现在我们需要一个任务，将发送电子邮件到模型中的位置。在我们的`tasks.py`文件中，添加以下任务：

```py
import smtplib
from email.mime.text import MIMEText

@celery.task(
    bind=True,
    ignore_result=True,
    default_retry_delay=300,
    max_retries=5
)
def remind(self, pk):
    reminder = Reminder.query.get(pk)
    msg = MIMEText(reminder.text)

    msg['Subject'] = "Your reminder"
    msg['From'] = your_email
    msg['To'] = reminder.email

    try:
        smtp_server = smtplib.SMTP('localhost')
        smtp_server.starttls()
        smtp_server.login(user, password)
        smtp_server.sendmail(
            your_email, 
            [reminder.email],
            msg.as_string()
        )
        smtp_server.close()

        return
    except Exception, e:
        self.retry(exc=e)
```

请注意，我们的任务接受的是主键而不是模型。这是对抗竞争条件的一种保护，因为传递的模型可能在工作程序最终处理它时已经过时。您还需要用自己的登录信息替换占位符电子邮件和登录。

当用户创建提醒模型时，我们如何调用我们的任务？我们将使用一个名为`events`的 SQLAlchemy 功能。SQLAlchemy 允许我们在我们的模型上注册回调，当对我们的模型进行特定更改时将被调用。我们的任务将使用`after_insert`事件，在新数据输入到数据库后被调用，无论模型是全新的还是正在更新。

我们需要在`tasks.py`中的回调：

```py
def on_reminder_save(mapper, connect, self):
    remind.apply_async(args=(self.id,), eta=self.date)
```

现在，在`__init__.py`中，我们将在我们的模型上注册我们的回调：

```py
from sqlalchemy import event
from .tasks import on_reminder_save

def create_app(object_name):
    app = Flask(__name__)
    app.config.from_object(object_name)

    db.init_app(app)
    event.listen(Reminder, 'after_insert', on_reminder_save)
    …
```

现在，每当模型被保存时，都会注册一个任务，该任务将向我们的用户发送一封电子邮件。

# 创建每周摘要

假设我们的博客有很多不使用 RSS 而更喜欢邮件列表的人，这是大量的用户。我们需要一种方法，在每周末结束时创建一个新帖子列表，以增加我们网站的流量。为了解决这个问题，我们将创建一个摘要任务，该任务将由一个 beat worker 在每个星期六的上午 10 点调用。

首先，在`tasks.py`中，让我们创建我们的任务如下：

```py
@celery.task(
    bind=True,
    ignore_result=True,
    default_retry_delay=300,
    max_retries=5
)
def digest(self):
    # find the start and end of this week
    year, week = datetime.datetime.now().isocalendar()[0:2]
    date = datetime.date(year, 1, 1)
    if (date.weekday() > 3):
        date = date + datetime.timedelta(days=7 - date.weekday())
    else:
        date = date - datetime.timedelta(days=date.weekday())
    delta = datetime.timedelta(days=(week - 1) * 7)
    start, end = date + delta, date + delta + datetime.timedelta(days=6)

    posts = Post.query.filter(
        Post.publish_date >= start,
        Post.publish_date <= end
    ).all()

    if (len(posts) == 0):
        return

    msg = MIMEText(
        render_template("digest.html", posts=posts),
        'html'
    )

    msg['Subject'] = "Weekly Digest"
    msg['From'] = your_email

    try:
        smtp_server = smtplib.SMTP('localhost')
        smtp_server.starttls()
        smtp_server.login(user, password)
        smtp_server.sendmail(
            your_email,
            [recipients],
            msg.as_string()
        )
        smtp_server.close()

        return
    except Exception, e:
        self.retry(exc=e)
```

我们还需要在`config.py`的配置对象中添加一个周期性计划来管理我们的任务：

```py
CELERYBEAT_SCHEDULE = {
    'weekly-digest': {
        'task': 'tasks.digest',
        'schedule': crontab(day_of_week=6, hour='10')
    },
}
```

最后，我们需要我们的电子邮件模板。不幸的是，电子邮件客户端中的 HTML 已经非常过时。每个电子邮件客户端都有不同的渲染错误和怪癖，找到它们的唯一方法就是在所有客户端中打开您的电子邮件。许多电子邮件客户端甚至不支持 CSS，而那些支持的也只支持很少的选择器和属性。为了弥补这一点，我们不得不使用 10 年前的网页开发方法，也就是使用带有内联样式的表进行设计。这是我们的`digest.html`：

```py
"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html >
    <head>
        <meta http-equiv="Content-Type"
              content="text/html; charset=UTF-8" />
        <meta name="viewport"
              content="width=device-width, initial-scale=1.0"/>
        <title>Weekly Digest</title>
    </head>
    <body>
        <table align="center"
               border="0"
               cellpadding="0"
               cellspacing="0"
               width="500px">
            <tr>
                <td style="font-size: 32px;
                           font-family: Helvetica, sans-serif;
                           color: #444;
                           text-align: center;
                           line-height: 1.65">
                    Weekly Digest
                </td>
            </tr>
            {% for post in posts %}
                <tr>
                    <td style="font-size: 24px;
                               font-family: sans-serif;
                               color: #444;
                               text-align: center;
                               line-height: 1.65">
                        {{ post.title }}
                    </td>
                </tr>
                <tr>
                    <td style="font-size: 14px;
                               font-family: serif;
                               color: #444;
                               line-height:1.65">
                        {{ post.text | truncate(500) | safe }}
                    </td>
                </tr>
                <tr>
                    <td style="font-size: 12px;
                               font-family: serif;
                               color: blue;
                               margin-bottom: 20px">
                        <a href="{{ url_for('.post', post_id=post.id) }}">Read More</a>
                    </td>
                </tr>
            {% endfor %}
        </table>
    </body>
</html>
```

现在，每周末，我们的摘要任务将被调用，并且会向我们邮件列表中的所有用户发送一封电子邮件。

# 总结

Celery 是一个非常强大的任务队列，允许程序员将较慢的任务的处理推迟到另一个进程中。现在您了解了如何将复杂的任务移出 Flask 进程，我们将看一下一系列简化 Flask 应用程序中一些常见任务的 Flask 扩展。
