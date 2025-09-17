# 第九章：使用 Celery 创建异步任务

在创建 Web 应用时，保持处理请求所需的时间低于或大约 50 毫秒至关重要。在每秒请求率中等到高的 Web 应用或 Web 服务中，响应时间变得更加重要。想象一下请求就像需要至少以与流速相同的速度处理的液体流，否则就会溢出。任何可以避免的服务器上的额外处理都应该避免。然而，在 Web 应用中，对需要超过几秒钟的操作的要求相当常见，尤其是在涉及复杂的数据库操作或图像处理时。

在构建一个能够水平扩展的应用时，应该能够将所有重型处理过程从 Web 服务器层解耦，并将它们耦合到一个可以独立扩展自己的工作层。

为了保护我们的用户体验和网站可靠性，将使用名为 Celery 的任务队列将这些操作从 Flask 进程中移出。

在本章中，我们将涵盖以下主题：

+   使用 Docker 运行 RabbitMQ 和 Redis

+   Celery 和 Flask 集成

+   学习识别应该在 Web 服务器外部运行的进程

+   从简单的异步任务到复杂的流程创建和调用多种类型的任务

+   将 Celery 作为带有 beats 的调度器使用

# 什么是 Celery？

**Celery** 是一个用 Python 编写的异步任务队列。Celery 通过 Python 的多进程库并发运行多个任务，这些任务是由用户定义的函数。Celery 从一个称为 **broker** 的消息队列接收消息，以启动任务，通常称为消息队列，如下面的图所示：

![图片](img/bcdd964a-f516-4b59-b8cd-3e4392304658.png)

**消息队列** 是一个专门设计用来在生产者进程和消费者进程之间发送数据的系统。**生产者进程** 是任何创建要发送到队列的消息的程序，而 **消费者进程** 是任何从队列中取出消息的程序。从生产者发送的消息存储在一个 **先进先出**（**FIFO**）队列中，其中最早的项目首先被检索。消息存储直到消费者接收消息，之后消息被删除。消息队列提供实时消息，不依赖于轮询，这意味着持续检查进程的状态。当消息从生产者发送时，消费者正在监听它们与消息队列的连接以获取新消息；消费者不是不断联系队列。这种差异类似于 **AJAX** 和 **WebSockets** 之间的差异，AJAX 需要持续与服务器保持联系，而 WebSockets 只是一个双向的持续通信流。

可以用传统的数据库替换消息队列。Celery 甚至内置了对 SQLAlchemy 的支持，以允许这样做。然而，使用数据库作为 Celery 的代理被高度不建议。用数据库代替消息队列需要消费者不断轮询数据库以获取更新。此外，由于 Celery 使用多进程进行并发，进行大量读取的连接数量会迅速增加。在中等负载下，使用数据库需要生产者在消费者读取的同时对数据库进行大量写入。

还可以使用消息队列作为代理和数据库来存储任务的结果。在上面的图中，消息队列用于发送任务请求和任务结果。然而，使用数据库存储任务最终结果允许最终产品无限期地存储，而消息队列将在生产者接收数据后立即丢弃数据，如下面的图所示：

![图片](img/03a68834-f6ae-4b8f-aa49-36634765da94.png)

这个数据库通常是一个键/值 NoSQL 存储，有助于处理负载。如果你计划对之前运行的任务进行数据分析，这很有用，但否则，坚持使用消息队列会更安全。

甚至有一个选项可以完全丢弃任务的结果，并且根本不返回结果。这的缺点是生产者无法知道任务是否成功，但在较小的项目中，这通常是可接受的。

对于我们的堆栈，我们将使用 RabbitMQ 作为消息代理。RabbitMQ 运行在所有主要操作系统上，并且非常简单易设置和运行。Celery 也支持 RabbitMQ，无需任何额外库，并且在 Celery 文档中被推荐为消息队列。

在撰写本文时，无法在 Python 3 中使用 RabbitMQ 与 Celery 结合。然而，你可以使用 Redis 来替代 RabbitMQ。唯一的区别将是连接字符串。更多信息，请参阅[`docs.celeryproject.org/en/latest/getting-started/brokers/redis.html`](http://docs.celeryproject.org/en/latest/getting-started/brokers/redis.html)。

# 设置 Celery 和 RabbitMQ

要在我们的`virtualenv`上安装 Celery，我们需要将其添加到我们的`requirements.txt`文件中：

```py
... 
Celery...
```

如往常一样，使用提供的`init.sh`脚本，或者使用这里解释的步骤在 Python 虚拟环境中创建和安装所有依赖项。

我们还需要一个 Flask 扩展来帮助处理 Celery 的初始化：

```py
$ pip install Flask-Celery-Helper
```

Flask 文档指出，Flask 扩展对于 Celery 是不必要的。然而，当你的应用使用应用工厂组织时，使 Celery 服务器与 Flask 的应用上下文协同工作是很重要的。因此，我们将使用`Flask-Celery-Helper`来完成这项繁重的工作。

接下来，RabbitMQ 需要启动并运行。为了轻松完成此操作，我们将使用 Docker 容器。请确保您已安装并正确设置了 Docker；如果没有，请查看第一章，*入门*，以获取说明。首先，我们需要一个非常简单的 Dockerfile：

```py
FROM rabbitmq:3-management

ENV RABBITMQ_ERLANG_COOKIE "SWQOKODSQALRPCLNMEQG"
ENV RABBITMQ_DEFAULT_USER "rabbitmq"
ENV RABBITMQ_DEFAULT_PASS "rabbitmq"
ENV RABBITMQ_DEFAULT_VHOST "/"
```

构建和运行带有管理界面的 RabbitMQ Docker 镜像只需要这些步骤。我们使用的是 Docker Hub 上的镜像，您可以在[`hub.docker.com/_/rabbitmq/`](https://hub.docker.com/_/rabbitmq/)下载。访问 Hub 页面以获取更多配置细节。

接下来，让我们构建我们的镜像，执行以下命令：

```py
$ docker build -t blog-rmq .
```

`-t` 标志用于给我们的镜像添加一个友好的名称；在这种情况下，`blog-rmq`。然后使用以下命令在后台运行新创建的镜像：

```py
$ docker run -d -p 15672:15672 -p 5672:5672 blog-rmq
```

`-d` 标志用于在后台（守护进程）运行容器。`-p` 标志用于容器和我们的主机/桌面之间的端口映射。

让我们检查它是否正常运行：

```py
$ docker ps
CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES
6eb2ab1da516 blog-rmq "docker-entrypoint.s…" 13 minutes ago Up 14 minutes 4369/tcp, 5671/tcp, 0.0.0.0:5672->5672/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:15672->15672/tcp xenodochial_kepler
```

让我们来看看 RabbitMQ 的管理界面。在您的浏览器中，导航到 `http://localhost:15672` 并使用 Dockerfile 上配置的凭据登录。在这种情况下，我们的用户名是 `rabbitmq`，密码也是 `rabbitmq`。

如果您需要更多信息，RabbitMQ 在 [`www.rabbitmq.com/download.html`](https://www.rabbitmq.com/download.html) 维护了针对每个操作系统的详细安装和配置说明列表。

安装 RabbitMQ 后，打开一个终端窗口并运行以下命令：

```py
$ rabbitmq-server
```

# 在 Celery 中创建任务

如前所述，Celery 任务只是执行某些操作的用户定义函数。但在编写任何任务之前，我们的 Celery 对象需要被创建。这是 Celery 服务器将导入以处理运行和调度所有任务的对象。

至少，Celery 需要一个配置变量来运行，那就是连接到消息代理。连接的定义方式与 SQLAlchemy 连接相同；也就是说，作为一个 URL。存储我们任务结果的后端也被定义为 URL，如下面的代码所示：

```py
class DevConfig(Config): 
    DEBUG = True 
    SQLALCHEMY_DATABASE_URI = 'sqlite:///../database.db' 
    CELERY_BROKER_URL = "amqp://rabbitmq:rabbitmq@localhost//" 
    CELERY_RESULT_BACKEND = "amqp://rabbitmq:rabitmq@localhost//" 
```

在 `__init__.py` 文件中，将初始化来自 `Flask-Celery-Helper` 的 `Celery` 类：

```py
from flask_celery import Celery
... 
celery = Celery()
...
def create_app(object_name):
...
    celery.init_app(app)
...
```

因此，为了使我们的 Celery 进程能够与数据库和任何其他 Flask 扩展一起工作，它需要在我们的应用程序上下文中运行。为此，Celery 需要为每个进程创建我们应用程序的新实例。像大多数 Celery 应用程序一样，我们需要一个 Celery 工厂来创建应用程序实例并在其上注册我们的 Celery 实例。在一个名为 `celery_runner.py` 的新文件中，位于顶级目录——与 `manage.py` 所在的位置相同——我们有以下内容：

```py
import os
from webapp import create_app
from celery import Celery

def make_celery(app):
    celery = Celery(
        app.import_name,
        broker=app.config['CELERY_BROKER_URL'],
        backend=app.config['CELERY_RESULT_BACKEND']
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
flask_app = create_app('config.%sConfig' % env.capitalize())

celery = make_celery(flask_app)
```

`make_celery`函数将每个 Celery 任务的调用都包装在 Python 的`with`块中。这确保了每次调用任何 Flask 扩展都将像与我们的应用程序一起工作一样工作。同时，确保不要将 Flask 应用程序实例命名为`app`，因为 Celery 试图将任何名为`app`或`celery`的对象导入为 Celery 应用程序实例。所以将你的 Flask 对象命名为`app`会导致 Celery 试图将其用作 Celery 对象。

现在我们可以编写我们的第一个任务了。它将是一个简单的任务；一个只返回传递给它的任何字符串的任务。我们在博客模块目录中有一个新文件，名为`tasks.py`。在这个文件中，找到以下内容：

```py
from .. import celery

@celery.task() 
def log(msg): 
    return msg
```

现在，最后一部分是要在一个新的终端窗口中运行 Celery 进程，这被称为**工作进程**。同样，这是将监听我们的消息代理以接收启动新任务的命令的过程：

```py
$ celery worker -A celery_runner --loglevel=info
```

`loglevel`标志在那里，所以你会在终端窗口中看到任务已被接收，并且其输出可用的确认信息。

现在，我们可以向我们的 Celery 工作进程发送命令。打开 Flask shell 会话，如下所示：

```py
$ export FLASK_APP=main.py $ flask shell    
>>> from webapp.blog.tasks import log
>>> log("Message")
Message
>>> result = log.delay("Message")
```

函数可以被调用，就像它是任何其他函数一样，这样做将在当前进程中执行该函数。然而，在任务上调用`delay`方法将向工作进程发送消息，以使用给定的参数执行该函数。

在运行 Celery 工作进程的终端窗口中，你应该看到如下类似的内容：

```py
Task webapp.blog.tasks.log succeeded in 0.0005873600021s: 'Message'
```

与任何异步任务一样，可以使用`ready`方法来判断任务是否成功完成。如果为`True`，则可以使用`get`方法按如下方式检索任务的结果：

```py
>>> result.ready() True >>> result.get() "Message"
```

`get`方法会导致当前进程等待直到`ready`函数返回`True`以检索结果。因此，在调用任务后立即调用`get`实际上会使任务变为同步。正因为如此，任务实际上返回值给生产者的情况相当罕见。绝大多数任务执行一些操作然后退出。

当在 Celery 工作进程中运行任务时，可以通过`state`属性访问任务的状态。这允许更细致地了解任务在工作进程中的当前操作。可用的状态如下：

+   `FAILURE`：任务失败，所有重试都失败了。

+   `PENDING`：任务尚未被工作进程接收。

+   `RECEIVED`：任务已被工作进程接收，但尚未处理。

+   `RETRY`：任务失败，正在等待重试。

+   `REVOKED`：任务被停止。

+   `STARTED`：工作进程已经开始处理任务。

+   `SUCCESS`：任务成功完成。

在 Celery 中，如果任务失败，则任务可以使用`retry`方法调用自己，如下所示：

```py
@celery.task(bind=True) 
def task(self, param): 
  try: 
    some_code 
  except Exception, e: 
    self.retry(exc=e) 
```

装饰器函数中的 `bind` 参数告诉 Celery 将任务对象的引用作为函数的第一个参数传递。使用 `self` 参数，可以调用 `retry` 方法，这将使用相同的参数重新运行任务。还可以将几个其他参数传递给函数装饰器以改变任务的行为：

+   `max_retries`：这是任务在被视为失败之前可以重试的最大次数。

+   `default_retry_delay`：这是在再次运行任务之前等待的时间（以秒为单位）。如果你预计导致任务失败的条件是暂时的，例如网络错误，那么将其保持在约一分钟左右是个好主意。

+   `rate_limit`：这指定了在给定间隔内允许运行此任务的总唯一调用次数。如果值是整数，则表示每秒允许此任务运行的调用次数总和。该值也可以是形式为 *x/m* 的字符串，表示每分钟 *x* 个任务，或 *x/h*，表示每小时 *x* 个任务。例如，传入 *5/m* 将只允许此任务每分钟被调用五次。

+   `time_limit`：如果指定了此参数，则如果任务运行时间超过指定的秒数，则将其终止。

+   `ignore_result`：如果任务没有使用返回值，则不要将其发送回去。

对于每个任务指定所有这些参数是一个好主意，以避免任何任务可能不会运行的机会。

# 运行 Celery 任务

`delay` 方法是 `apply_async` 方法的简写版本，其调用格式如下：

```py
task.apply_async( 
  args=[1, 2], 
  kwargs={'kwarg1': '1', 'kwarg2': '2'} 
) 
```

然而，`args` 关键字可以是隐式的，如下所示：

```py
apply_async([1, 2], kwargs={'kwarg1': '1', 'kwarg2': '2'}) 
```

调用 `apply_async` 允许你在任务调用中定义一些在 `delay` 方法中无法指定的额外功能。首先，`countdown` 选项指定了工作员在接收到任务后应该等待多长时间（以秒为单位）再运行它：

```py
>>> from webapp.blog.tasks import log
>>> log.apply_async(["Message"], countdown=600)
```

`countdown` 并不保证任务将在 `600` 秒后运行。`countdown` 选项仅表示任务在 *x* 秒后将可供处理。如果所有工作进程都忙于其他任务，则它不会立即运行。

`apply_async` 提供的另一个关键字参数是 `eta` 参数。`eta` 通过 Python `datetime` 对象传递，指定任务应该运行的确切时间。同样，`eta` 不可靠：

```py
>>> import datetime
>>> from webapp.blog.tasks import log
# Run the task one hour from now
>>> eta = datetime.datetime.now() + datetime.timedelta(hours=1)
>>> log.apply_async(["Message"], eta=eta)

```

# Celery 工作流

Celery 提供了许多方法来将多个依赖任务组合在一起，或者并行执行多个任务。这些方法在很大程度上受到了函数式编程语言中发现的语言特性的影响。然而，要理解这是如何工作的，我们首先需要了解签名。考虑以下任务：

```py
@celery.task() 
def multiply(x, y): 
    return x * y 
```

让我们看看 **签名** 的实际应用来理解它。打开 Flask shell 并输入以下内容：

```py
# Export FLASK_APP if you haven't already
$ export FLASK_APP=main.py
$ flask shell
>>> from celery import signature
>>> from webapp.blog.tasks import multiply
# Takes the same keyword args as apply_async
>>> signature('webapp.tasks.multiply', args=(4, 4), countdown=10) webapp.tasks.multiply(4, 4) # same as above
>>> from webapp.blog.tasks import multiply
>>> multiply.subtask((4, 4), countdown=10) webapp.tasks.multiply(4, 4) # shorthand for above, like delay in that it doesn't take
# apply_async's keyword args
>>> multiply.s(4, 4) webapp.blog.tasks.multiply(4, 4) >>> multiply.s(4, 4)() 16 >>> multiply.s(4, 4).delay() 
```

调用一个任务的签名（有时称为**子任务**）创建一个可以传递给其他函数执行的函数。执行签名，就像前面示例中的倒数第三行，将在当前进程中执行函数，而不是在工作者进程中。

# 偏函数

任务签名首次应用是功能编程风格的偏函数。**偏函数**是函数，原本接受多个参数，但通过对一个原始函数应用操作来返回一个新的函数，因此前**n**个参数总是相同的。考虑以下示例，我们有一个`multiply`函数，它不是一个任务：

```py
>>> new_multiply = multiply(2)
>>> new_multiply(5) 10 # The first function is unaffected
>>> multiply(2, 2) 4

```

这是一个虚构的 API，但非常接近 Celery 版本：

```py
>>> partial = multiply.s(4)
>>> partial.delay(4)
```

工作者窗口中的输出应显示`16`。基本上，我们创建了一个新的函数，保存到偏函数中，该函数将始终将其输入乘以四。

# 回调

一旦任务完成，根据前一个任务的输出运行另一个任务是非常常见的。为了实现这一点，`apply_async`函数有一个`link`方法，用法如下：

```py
>>> multiply.apply_async((4, 4), link=log.s())
```

工作者的输出应显示`multiply`任务和`log`任务都返回了`16`。

如果你有一个不接受输入的函数，或者你的回调不需要原始方法的返回结果，那么任务签名必须使用`si`方法标记为不可变：

```py
>>> multiply.apply_async((4, 4), link=log.si("Message"))
```

**回调**可以用来解决实际问题。如果我们想在每次任务创建新用户时发送欢迎邮件，那么我们可以通过以下调用产生这种效果：

```py
>>> create_user.apply_async(("John Doe", password), link=welcome.s())
```

偏函数和回调可以组合起来产生一些强大的效果：

```py
>>> multiply.apply_async((4, 4), link=multiply.s(4))
```

重要的是要注意，如果这个调用被保存，并且对其调用`get`方法，结果将是`16`，而不是`64`。这是因为`get`方法不会为回调方法返回结果。这个问题将在后续方法中得到解决。

# 组

`group`函数接受一个签名列表，创建一个可调用的函数来并行执行所有签名，然后返回所有结果，如下所示：

```py
>>> from celery import group
>>> sig = group(multiply.s(i, i+5) for i in range(10))
>>> result = sig.delay()
>>> result.get() [0, 6, 14, 24, 36, 50, 66, 84, 104, 126]
```

# 链

`chain`函数接受任务签名并将每个结果的值传递给链中的下一个值，返回一个结果，如下所示：

```py
>>> from celery import chain
>>> sig = chain(multiply.s(10, 10), multiply.s(4), multiply.s(20))
# same as above
>>> sig = (multiply.s(10, 10) | multiply.s(4) | multiply.s(20))
>>> result = sig.delay()
>>> result.get() 8000

```

链和偏函数可以进一步扩展。链可以在使用偏函数时创建新函数，并且链可以嵌套，如下所示：

```py
# combining partials in chains 
>>> func = (multiply.s(10) | multiply.s(2)) 
>>> result = func.delay(16) 
>>> result.get() 320 # chains can be nested 
>>> func = ( multiply.s(10) | multiply.s(2) | (multiply.s(4) | multiply.s(5)) ) 
>>> result = func.delay(16) 
>>> result.get() 6400

```

# 和弦

`chord`函数创建一个签名，将执行一组签名并将最终结果传递给回调：

```py
>>> from celery import chord
>>> sig = chord(
        group(multiply.s(i, i+5) for i in range(10)),
        log.s()
)
>>> result = sig.delay()
>>> result.get() [0, 6, 14, 24, 36, 50, 66, 84, 104, 126]

```

就像链接参数一样，回调不会被`get`方法返回。

使用组和回调的`chain`语法会自动创建一个和弦签名：

```py
# same as above
>>> sig = (group(multiply.s(i, i+5) for i in range(10)) | log.s())
>>> result = sig.delay()
>>> result.get() [0, 6, 14, 24, 36, 50, 66, 84, 104, 126]

```

# 定期运行任务

Celery 还具有定期调用任务的能力。对于那些熟悉 *******nix** 操作系统的人来说，这个系统很像命令行工具 `cron`，但它有一个额外的优点，即它在我们的源代码中定义，而不是在某个系统文件中定义。因此，当我们的代码准备好发布到生产环境时——我们将在这个第十三章达到这个阶段——*部署 Flask 应用程序*，这将更容易更新我们的代码。此外，所有任务都在应用程序上下文中运行，而由 `cron` 调用的 Python 脚本则不是。

要添加周期性任务，请将以下内容添加到 `DevConfig` 配置对象中：

```py
import datetime 
... 
CELERYBEAT_SCHEDULE = { 
    'log-every-30-seconds': { 
        'task': 'webapp.blog.tasks.log', 
        'schedule': datetime.timedelta(seconds=30), 
        'args': ("Message",) 
    }, 
} 
```

这个`配置`变量定义了`log`任务应该每 30 秒运行一次，并将`args`元组作为参数传递。任何`timedelta`对象都可以用来定义运行任务的间隔。

要运行周期性任务，需要一个名为`beat`的专用工作者。在另一个终端窗口中，运行以下命令：

```py
$ celery -A celery_runner beat
```

如果你现在查看主 `Celery` 工作者的控制台输出，你应该现在每 30 秒看到一个日志事件。

如果你的任务需要在更具体的间隔上运行；比如说，例如，在六月的每个星期二凌晨 3 点和下午 5 点？对于非常具体的间隔，有 Celery 的 `crontab` 对象。

为了说明 `crontab` 对象如何表示间隔，考虑以下示例：

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

该对象有以下参数：

+   `minute`

+   `hour`

+   `day_of_week`

+   `day_of_month`

+   `month_of_year`

这些参数可以接受各种输入。使用纯整数时，它们的工作方式与 `timedelta` 对象类似，但也可以接受字符串和列表。当传递一个列表时，任务将在列表中的每个时刻执行。当传递一个形式为 **/x* 的字符串时，任务将在模运算返回零的每个时刻执行。这两种形式也可以组合成一个由逗号分隔的整数和除法字符串。

# 监控 Celery

当我们的代码推送到服务器时，我们的 `Celery` 工作者不会在终端窗口中运行——而是作为后台任务运行。因此，Celery 提供了许多命令行参数来监控你的 `Celery` 工作者和任务的状态。这些命令的形式如下：

```py
$ celery -A celery_runner <command>
```

查看您工作者状态的 主要任务如下：

+   `status`: 这将打印正在运行的工作者以及它们是否处于活动状态。

+   `result`: 当传递一个任务 ID 时，这将显示任务的返回值和最终状态。

+   `purge`: 使用这个，将删除代理中的所有消息。

+   `inspect active`: 这列出了所有活动任务。

+   `inspect scheduled`: 这列出了所有带有 `eta` 参数已安排的任务。

+   `inspect registered`: 这列出了所有等待处理的任务。

+   `inspect stats`: 这将返回一个包含当前运行中的工作者和代理的静态信息的字典。

# 基于 Web 的监控使用 Flower

**Flower**是一个基于 Web 的 Celery 实时管理工具。在 Flower 中，可以监控所有活跃的、排队的和完成的任务。Flower 还提供了关于每个任务在队列中等待了多久以及执行了多久，以及每个任务的参数的图表和统计数据。

要安装`flower`，使用以下`pip`命令：

```py
$ pip install flower
```

要运行它，只需将`flower`当作 Celery 命令来运行，如下所示：

```py
$ celery flower -A celery_runner --loglevel=info
```

现在，打开你的浏览器到`http://localhost:5555`。最好在任务运行时熟悉界面，所以请在命令行中输入以下内容：

```py
>>> export FLASK_APP=manage.py
>>> flask shell
>>> from webapp.blog.tasks import *
>>> from celery import chord, group
>>> sig = chord(  group(multiply.s(i, i+5) for i in xrange(10000)),  log.s() )
>>> sig.delay()
```

你的 worker 进程现在将开始处理 10,000 个任务。在任务运行时浏览不同的页面，看看`flower`如何与你的 worker 交互，如图所示：

![图片](img/dd0af794-6345-440f-a8b6-f02b99026662.png)

# 创建提醒应用

让我们来看看 Celery 的一些实际应用示例。假设我们网站上的另一个页面现在需要一个提醒功能。用户可以创建提醒，在指定时间将电子邮件发送到指定的位置。我们需要一个模型、一个任务以及每次创建模型时自动调用我们的任务的方法。

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

现在，我们需要一个任务，该任务将向模型中的位置发送电子邮件。在我们的`blog/tasks.py`文件中查找以下任务：

```py
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
    msg['From'] = current_app.config['SMTP_FROM']
    msg['To'] = reminder.email
    try:
        smtp_server = smtplib.SMTP(current_app.config['SMTP_SERVER'])
        smtp_server.starttls()
        smtp_server.login(current_app.config['SMTP_USER'], 
        current_app.config['SMTP_PASSWORD'])
        smtp_server.sendmail("", [reminder.email], msg.as_string())
        smtp_server.close()
        return
    except Exception as e:
        self.retry(exc=e)
```

注意，我们的任务接受一个主键，而不是一个模型。这是为了防止竞争条件，因为传递的模型在 worker 最终处理它时可能已经过时。你还需要将占位符电子邮件和登录详情替换成你自己的登录信息。

当用户创建一个提醒模型时，我们如何调用我们的任务？我们将使用一个名为`events`的 SQLAlchemy 功能。SQLAlchemy 允许我们在模型上注册回调，当我们的模型发生特定变化时会被调用。我们的任务将使用`after_insert`事件，该事件在将新数据输入数据库后调用，无论模型是全新的还是正在更新。

在`blog/tasks.py`中，我们需要一个回调：

```py
def on_reminder_save(mapper, connect, self): 
    remind.apply_async(args=(self.id,), eta=self.date) 
```

现在，在`blog/__init__.py`中，我们将在我们的模型上注册回调：

```py
from sqlalchemy import event
from .models import db, Reminder
from .tasks import on_reminder_save

def create_module(app, **kwargs):
    event.listen(Reminder, 'after_insert', on_reminder_save)
    from .controllers import blog_blueprint
    app.register_blueprint(blog_blueprint)
```

现在，每次模型被保存时，都会注册一个任务，该任务将向我们的用户发送电子邮件。

# 创建每周摘要

我们的博客有很多不使用 RSS 的人，他们更喜欢使用邮件列表。我们需要一种方法在每周结束时创建一篇新帖子的列表，以增加我们网站的流量。为了解决这个问题，我们将创建一个摘要任务，该任务将在每周六上午 10 点由一个 beat worker 调用。

首先，在`blog/tasks.py`中，让我们创建以下任务：

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
        date = date + datetime.timedelta(7 - date.weekday())
    else:
        date = date - datetime.timedelta(date.weekday())
    delta = datetime.timedelta(days=(week - 1) * 7)
    start, end = date + delta, date + delta + 
    datetime.timedelta(days=6)

    posts = Post.query.filter(
        Post.publish_date >= start,
        Post.publish_date <= end
    ).all()

    if (len(posts) == 0):
        return

    msg = MIMEText(render_template("digest.html", posts=posts), 'html')

    msg['Subject'] = "Weekly Digest"
    msg['From'] = current_app.config['SMTP_FROM']

    try:
        smtp_server = smtplib.SMTP(current_app.config['SMTP_SERVER'])
        smtp_server.starttls()
        smtp_server.login(current_app.config['SMTP_USER'], 
        current_app.config['SMTP_PASSWORD'])
        smtp_server.sendmail("", [""], msg.as_string())
        smtp_server.close()

        return
    except Exception as e:
        self.retry(exc=e)
```

我们还需要在我们的`config.py`配置对象中添加一个周期性调度来管理我们的任务：

```py
from celery.schedules import crontab
...
CELERYBEAT_SCHEDULE = { 'weekly-digest': { 'task': 'blog.tasks.digest', 'schedule': crontab(day_of_week=6, hour='10') }, }
```

我们还需要配置我们的 SMTP 服务器，以便我们能够发送电子邮件。这可以通过使用 Gmail 或您的公司电子邮件凭证来完成。将您选择的账户信息添加到 `config.py` 中的配置对象：

```py
...
SMTP_SERVER = "smtp.gmail.com"
SMTP_USER = "sometestemail@gmail.com"
SMTP_PASSWORD = "password"
SMTP_FROM = "from@flask.com"
...
```

最后，我们需要我们的电子邮件模板。不幸的是，电子邮件客户端中的 HTML 已经非常过时。每个电子邮件客户端都有不同的渲染错误和怪癖，唯一的办法是打开所有客户端中的电子邮件。许多电子邮件客户端甚至不支持 CSS，而那些支持 CSS 的客户端支持的选择器和属性也非常有限。为了弥补这一点，我们不得不使用 10 年前的网络开发方法；也就是说，使用内联样式设计表格。下面是我们的 `digest.html` 文件：

```py
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" 
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
          <a href="{{ url_for('.post', post_id=post.id) }}">Read 
             More</a> 
        </td> 
      </tr> 
      {% endfor %} 
    </table> 
  </body> 
</html> 
```

现在，在每周的末尾，我们的摘要任务将被调用，并将向所有在我们的邮件列表中的用户发送电子邮件。

# 摘要

Celery 是一个非常强大的任务队列，允许程序员将较慢的任务的处理推迟到另一个进程。现在你已经了解了如何将复杂任务从 Flask 进程中移出，我们将查看一系列简化 Flask 应用中常见任务的 Flask 扩展。

在下一章中，你将学习如何利用一些优秀的社区构建的 Flask 扩展来提高性能、调试，甚至快速创建一个管理后台。
