# 第十一章：发送电子邮件的任务

现在我们有了我们的模型和视图，我们需要让 Mail Ape 发送电子邮件。我们将让 Mail Ape 发送两种类型的电子邮件，订阅者确认电子邮件和邮件列表消息。我们将通过创建一个名为`SubscriberMessage`的新模型来跟踪邮件列表消息的成功发送，以跟踪是否成功将消息发送给存储在`Subscriber`模型实例中的地址。由于向许多`Subscriber`模型实例发送电子邮件可能需要很长时间，我们将使用 Celery 在常规 Django 请求/响应周期之外作为任务发送电子邮件。

在本章中，我们将做以下事情：

+   使用 Django 的模板系统生成我们电子邮件的 HTML 主体

+   使用 Django 发送包含 HTML 和纯文本的电子邮件

+   使用 Celery 执行异步任务

+   防止我们的代码在测试期间发送实际电子邮件

让我们首先创建一些我们将用于发送动态电子邮件的常见资源。

# 创建电子邮件的常见资源

在本节中，我们将创建一个基本的 HTML 电子邮件模板和一个用于呈现电子邮件模板的`Context`对象。我们希望为我们的电子邮件创建一个基本的 HTML 模板，以避免重复使用样板 HTML。我们还希望确保我们发送的每封电子邮件都包含一个退订链接，以成为良好的电子邮件用户。我们的`EmailTemplateContext`类将始终提供我们的模板需要的常见变量。

让我们首先创建一个基本的 HTML 电子邮件模板。

# 创建基本的 HTML 电子邮件模板

我们将在`django/mailinglist/templates/mailinglist/email/base.html`中创建我们的基本电子邮件 HTML 模板：

```py
<!DOCTYPE html>
<html lang="en" >
<head >
<body >
{% block body %}
{% endblock %}

Click <a href="{{ unsubscription_link }}">here</a> to unsubscribe from this
mailing list.
Sent with Mail Ape .
</body >
</html >
```

前面的模板看起来像是`base.html`的一个更简单的版本，只有一个块。电子邮件模板可以扩展`email/base.html`并覆盖主体块，以避免样板 HTML。尽管文件名相同（`base.html`），Django 不会混淆两者。模板是通过它们的模板路径标识的，不仅仅是文件名。

我们的基本模板还期望`unsubscription_link`变量始终存在。这将允许用户取消订阅，如果他们不想继续接收电子邮件。

为了确保我们的模板始终具有`unsubscription_link`变量，我们将创建一个`Context`来确保始终提供它。

# 创建 EmailTemplateContext

正如我们之前讨论过的（参见第一章，*构建 MyMDB*），要呈现模板，我们需要为 Django 提供一个`Context`对象，其中包含模板引用的变量。在编写基于类的视图时，我们只需要在`get_context_data()`方法中提供一个字典，Django 会为我们处理一切。然而，当我们想要自己呈现模板时，我们将不得不自己实例化`Context`类。为了确保我们所有的电子邮件模板呈现代码提供相同的最小信息，我们将创建一个自定义模板`Context`。

让我们在`django/mailinglist/emails.py`中创建我们的`EmailTemplateContext`类：

```py
from django.conf import settings

from django.template import Context

class EmailTemplateContext(Context):

    @staticmethod
    def make_link(path):
        return settings.MAILING_LIST_LINK_DOMAIN + path

    def __init__(self, subscriber, dict_=None, **kwargs):
        if dict_ is None:
            dict_ = {}
        email_ctx = self.common_context(subscriber)
        email_ctx.update(dict_)
        super().__init__(email_ctx, **kwargs)

    def common_context(self, subscriber):
        subscriber_pk_kwargs = {'pk': subscriber.id}
        unsubscribe_path = reverse('mailinglist:unsubscribe',
                                   kwargs=subscriber_pk_kwargs)
        return {
            'subscriber': subscriber,
            'mailing_list': subscriber.mailing_list,
            'unsubscribe_link': self.make_link(unsubscribe_path),
        }
```

我们的`EmailTemplateContext`由以下三种方法组成：

+   `make_link()`: 这将 URL 的路径与我们项目的`MAILING_LIST_LINK_DOMAIN`设置连接起来。`make_link`是必要的，因为 Django 的`reverse()`函数不包括域。Django 项目可以托管在多个不同的域上。我们将在*配置电子邮件设置*部分更多地讨论`MAILING_LIST_LINK_DOMAIN`的值。

+   `__init__()`: 这覆盖了`Context.__init__(...)`方法，给了我们一个机会将`common_context()`方法的结果添加到`dict_`参数的值中。我们要小心让参数接收到的数据覆盖我们在`common_context`中生成的数据。

+   `common_context()`: 这返回一个字典，提供我们希望所有`EmailTemplateContext`对象可用的变量。我们始终希望有`subscriber`、`mailing_list`和`unsubscribtion_link`可用。

我们将在下一节中使用这两个资源，我们将向新的`Subscriber`模型实例发送确认电子邮件。

# 发送确认电子邮件

在本节中，我们将向新的`Subscriber`发送电子邮件，让他们确认对`MailingList`的订阅。

在本节中，我们将：

1.  将 Django 的电子邮件配置设置添加到我们的`settings.py`

1.  编写一个函数来使用 Django 的`send_mail()`函数发送电子邮件

1.  创建和渲染电子邮件正文的 HTML 和文本模板

1.  更新`Subscriber.save()`以在创建新的`Subscriber`时发送电子邮件

让我们从更新配置开始，使用我们邮件服务器的设置。

# 配置电子邮件设置

为了能够发送电子邮件，我们需要配置 Django 与**简单邮件传输协议**（**SMTP**）服务器进行通信。在开发和学习过程中，您可能可以使用与您的电子邮件客户端相同的 SMTP 服务器。对于发送大量生产电子邮件，使用这样的服务器可能违反您的电子邮件提供商的服务条款，并可能导致帐户被暂停。请注意您使用的帐户。

让我们在`django/config/settings.py`中更新我们的设置：

```py
EMAIL_HOST = 'smtp.example.com'
EMAIL_HOST_USER = 'username'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_PASSWORD')

MAILING_LIST_FROM_EMAIL = 'noreply@example.com'
MAILING_LIST_LINK_DOMAIN = 'http://localhost:8000'
```

在上面的代码示例中，我使用了很多`example.com`的实例，您应该将其替换为您的 SMTP 主机和域的正确域。让我们更仔细地看一下设置：

+   `EMAIL_HOST`: 这是我们正在使用的 SMTP 服务器的地址。

+   `EMAIL_HOST_USER`: 用于对 SMTP 服务器进行身份验证的用户名。

+   `EMAIL_PORT`: 连接到 SMTP 服务器的端口。

+   `EMAIL_USE_TLS`: 这是可选的，默认为`False`。如果您要通过 TLS 连接到 SMTP 服务器，请使用它。如果您使用 SSL，则使用`EMAIL_USE_SSL`设置。SSL 和 TLS 设置是互斥的。

+   `EMAIL_HOST_PASSWORD`: 主机的密码。在我们的情况下，我们将期望密码在环境变量中。

+   `MAILING_LIST_FROM_EMAIL`: 这是我们使用的自定义设置，用于设置我们发送的电子邮件的`FROM`标头。

+   `MAILING_LIST_LINK_DOMAIN`: 这是所有电子邮件模板链接的前缀域。我们在`EmailTemplateContext`类中看到了这个设置的使用。

接下来，让我们编写我们的创建函数来发送确认电子邮件。

# 创建发送电子邮件确认函数

现在，我们将创建一个实际创建并发送确认电子邮件给我们的`Subscriber`的函数。`email`模块将包含所有我们与电子邮件相关的代码（我们已经在那里创建了`EmailTemplateContext`类）。

我们的`send_confirmation_email()`函数将需要执行以下操作：

1.  为渲染电子邮件正文创建一个`Context`

1.  为电子邮件创建主题

1.  渲染 HTML 和文本电子邮件正文

1.  使用`send_mail()`函数发送电子邮件

让我们在`django/mailinglist/emails.py`中创建该函数：

```py
from django.conf import settings
from django.core.mail import send_mail
from django.template import engines, Context
from django.urls import reverse

CONFIRM_SUBSCRIPTION_HTML = 'mailinglist/email/confirmation.html'

CONFIRM_SUBSCRIPTION_TXT = 'mailinglist/email/confirmation.txt'

class EmailTemplateContext(Context):
    # skipped unchanged class

def send_confirmation_email(subscriber):
    mailing_list = subscriber.mailing_list
    confirmation_link = EmailTemplateContext.make_link(
        reverse('mailinglist:confirm_subscription',
                kwargs={'pk': subscriber.id}))
    context = EmailTemplateContext(
        subscriber,
        {'confirmation_link': confirmation_link}
    )
    subject = 'Confirming subscription to {}'.format(mailing_list.name)

    dt_engine = engines['django'].engine
    text_body_template = dt_engine.get_template(CONFIRM_SUBSCRIPTION_TXT)
    text_body = text_body_template.render(context=context)
    html_body_template = dt_engine.get_template(CONFIRM_SUBSCRIPTION_HTML)
    html_body = html_body_template.render(context=context)

    send_mail(
        subject=subject,
        message=text_body,
        from_email=settings.MAILING_LIST_FROM_EMAIL,
        recipient_list=(subscriber.email,),
        html_message=html_body)
```

让我们更仔细地看一下我们的代码：

+   `EmailTemplateContext()`: 这实例化了我们之前创建的`Context`类。我们为其提供了一个`Subscriber`实例和一个包含确认链接的`dict`。`confirmation_link`变量将被我们的模板使用，我们将在接下来的两个部分中创建。

+   `engines['django'].engine`: 这引用了 Django 模板引擎。引擎知道如何使用`settings.py`中`TEMPLATES`设置中的配置设置来查找`Template`。

+   `dt_engine.get_template()`: 这将返回一个模板对象。我们将模板的名称作为参数提供给`get_template()`方法。

+   `text_body_template.render()`: 这将模板（使用之前创建的上下文）渲染为字符串。

最后，我们使用`send_email()`函数发送电子邮件。`send_email()`函数接受以下参数：

+   `subject=subject`: 电子邮件消息的主题。

+   `message=text_body`: 电子邮件的文本版本。

+   `from_email=settings.MAILING_LIST_FROM_EMAIL`：发件人的电子邮件地址。如果我们不提供`from_email`参数，那么 Django 将使用`DEFAULT_FROM_EMAIL`设置。

+   `recipient_list=(subscriber.email,)`：收件人电子邮件地址的列表（或元组）。这必须是一个集合，即使您只发送给一个收件人。如果包括多个收件人，他们将能够看到彼此。

+   `html_message=html_body`：电子邮件的 HTML 版本。这个参数是可选的，因为我们不必提供 HTML 正文。如果我们提供 HTML 正文，那么 Django 将发送包含 HTML 和文本正文的电子邮件。电子邮件客户端将选择显示电子邮件的 HTML 或纯文本版本。

现在我们已经有了发送电子邮件的代码，让我们制作我们的电子邮件正文模板。

# 创建 HTML 确认电子邮件模板

让我们制作 HTML 订阅电子邮件确认模板。我们将在`django/mailinglist/templates/mailinglist/email_templates/confirmation.html`中创建模板：

```py
{% extends "mailinglist/email_templates/email_base.html" %}

{% block body %}
  <h1>Confirming subscription to {{ mailing_list }}</h1 >
  <p>Someone (hopefully you) just subscribed to {{ mailinglist }}.</p >
  <p>To confirm your subscription click <a href="{{ confirmation_link }}">here</a>.</p >
  <p>If you don't confirm, you won't hear from {{ mailinglist }} ever again.</p >
  <p>Thanks,</p >
  <p>Your friendly internet Mail Ape !</p>
{% endblock %}
```

我们的模板看起来就像一个 HTML 网页模板，但它将用于电子邮件。就像一个普通的 Django 模板一样，我们正在扩展一个基本模板并填写一个块。在我们的情况下，我们正在扩展的模板是我们在本章开始时创建的`email/base.html`模板。另外，请注意我们如何使用我们在`send_confirmation_email()`函数中提供的变量（例如`confirmation_link`）和我们的`EmailTemplateContext`（例如`mailing_list`）。

电子邮件可以包含 HTML，但并非总是由 Web 浏览器呈现。值得注意的是，一些版本的 Microsoft Outlook 使用 Microsoft Word HTML 渲染器来渲染电子邮件。即使是在运行在浏览器中的 Gmail 也会在呈现之前操纵它收到的 HTML。请小心在真实的电子邮件客户端中测试复杂的布局。

接下来，让我们创建这个模板的纯文本版本。

# 创建文本确认电子邮件模板

现在，我们将创建确认电子邮件模板的纯文本版本；让我们在`django/mailinglist/templates/mailinglist/email_templates/confirm_subscription.txt`中创建它：

```py
Hello {{subscriber.email}},

Someone (hopefully you) just subscribed to {{ mailinglist }}.

To confirm your subscription go to {{confirmation_link}}.

If you don't confirm you won't hear from {{ mailinglist }} ever again.

Thanks,

Your friendly internet Mail Ape !
```

在上述情况下，我们既不使用 HTML 也不扩展任何基本模板。

然而，我们仍在引用我们在`send_confirmation_email()`中提供的变量（例如`confirmation_link`）函数和我们的`EmailTemplateContext`类（例如`mailing_list`）。

现在我们已经有了发送电子邮件所需的所有代码，让我们在创建新的`Subscriber`模型实例时发送它们。

# 在新的 Subscriber 创建时发送

作为最后一步，我们将向用户发送确认电子邮件；我们需要调用我们的`send_confirmation_email`函数。基于 fat models 的理念，我们将从我们的`Subscriber`模型而不是视图中调用我们的`send_confirmation_email`函数。在我们的情况下，当保存新的`Subscriber`模型实例时，我们将发送电子邮件。

让我们更新我们的`Subscriber`模型，在保存新的`Subscriber`时发送确认电子邮件。为了添加这种新行为，我们需要编辑`django/mailinglist/models.py`：

```py
from django.db import models
from mailinglist import emails

class Subscriber(models.Model):
    # skipping unchanged model body

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        is_new = self._state.adding or force_insert
        super().save(force_insert=force_insert, force_update=force_update,
                     using=using, update_fields=update_fields)
        if is_new:
            self.send_confirmation_email()

    def send_confirmation_email(self):        
           emails.send_confirmation_email(self)
```

在创建模型时添加新行为的最佳方法是重写模型的`save()`方法。在重写`save()`时，非常重要的是我们仍然调用超类的`save()`方法，以确保模型保存。我们的新保存方法有三个作用：

+   检查当前模型是否为新模型

+   调用超类的`save()`方法

+   如果模型是新的，则发送确认电子邮件

要检查当前模型实例是否是新的，我们检查`_state`属性。`_state`属性是`ModelState`类的一个实例。通常，以下划线（`_`）开头的属性被认为是私有的，并且可能会在 Django 的不同版本中发生变化。但是，`ModelState`类在 Django 的官方文档中有描述，所以我们可以更放心地使用它（尽管我们应该密切关注未来版本的变化）。如果`self._state.adding`为`True`，那么`save()`方法将会将这个模型实例插入为新行。如果`self._state.adding`为`True`，那么`save()`方法将会更新现有行。

我们还将`emails.send_confirmation_email()`的调用包装在`Subscriber`方法中。如果我们想要重新发送确认电子邮件，这将非常有用。任何想要重新发送确认电子邮件的代码都不需要知道`emails`模块。模型是所有操作的专家。这是 fat model 哲学的核心。

# 本节的快速回顾

在本节中，我们学习了更多关于 Django 模板系统以及如何发送电子邮件。我们学会了如何渲染模板，而不是使用 Django 的内置视图来直接使用 Django 模板引擎为我们渲染它。我们使用了 Django 的最佳实践，创建了一个服务模块来隔离所有我们的电子邮件代码。最后，我们还使用了`send_email()`来发送一封带有文本和 HTML 正文的电子邮件。

接下来，让我们在向用户返回响应后使用 Celery 发送这些电子邮件。

# 使用 Celery 发送电子邮件

随着我们构建越来越复杂的应用程序，我们经常希望执行操作，而不强迫用户等待我们返回 HTTP 响应。Django 与 Celery 很好地配合，Celery 是一个流行的 Python 分布式任务队列，可以实现这一点。

Celery 是一个在代理中*排队* *任务*以供 Celery *工作者*处理的库。让我们更仔细地看看其中一些术语：

+   **Celery 任务**封装了我们想要异步执行的可调用对象。

+   **Celery 队列**是按照先进先出顺序存储在代理中的任务列表。

+   **Celery 代理**是提供快速高效的队列存储的服务器。流行的代理包括 RabbitMQ、Redis 和 AWS SQS。Celery 对不同代理有不同级别的支持。我们将在开发中使用 Redis 作为我们的代理。

+   **Celery 工作者**是单独的进程，它们检查任务队列以执行任务并执行它们。

在本节中，我们将做以下事情：

1.  安装 Celery

1.  配置 Celery 以与 Django 一起工作

1.  使用 Celery 队列发送确认电子邮件任务

1.  使用 Celery 工作者发送我们的电子邮件

让我们首先安装 Celery。

# 安装 celery

要安装 Celery，我们将使用这些新更改更新我们的`requirements.txt`文件：

```py
celery<4.2
celery[redis]
django-celery-results<2.0
```

我们将安装三个新包及其依赖项：

+   `celery`：安装主要的 Celery 包

+   `celery[redis]`：安装我们需要使用 Redis 作为代理的依赖项

+   `django-celery-results`：让我们将执行的任务结果存储在我们的 Django 数据库中；这只是存储和记录 Celery 结果的一种方式

接下来，让我们使用`pip`安装我们的新包：

```py
$ pip install -r requirements.txt
```

现在我们已经安装了 Celery，让我们配置 Mail Ape 来使用 Celery。

# 配置 Celery 设置

要配置 Celery，我们需要进行两组更改。首先，我们将更新 Django 配置以使用 Celery。其次，我们将创建一个 Celery 配置文件，供我们的工作者使用。

让我们首先更新`django/config/settings.py`：

```py
INSTALLED_APPS = [
    'user',
    'mailinglist',

    'crispy_forms',
    'markdownify',
    'django_celery_results',

    'django.contrib.admin',
    # other built in django apps unchanged.
]

CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'django-db'
```

让我们更仔细地看看这些新设置：

+   `django_celery_results`：这是一个我们安装为 Django 应用程序的 Celery 扩展，让我们将 Celery 任务的结果存储在 Django 数据库中。

+   `CELERY_BROKER_URL`：这是我们的 Celery 代理的 URL。在我们的情况下，我们将在开发中使用本地的 Redis 服务器。

+   `CELERY_RESULT_BACKEND`：这表示存储结果的位置。在我们的情况下，我们将使用 Django 数据库。

由于`django_celery_results`应用程序允许我们在数据库中保存结果，因此它包括新的 Django 模型。为了使这些模型存在于数据库中，我们需要迁移我们的数据库：

```py
$ cd django
$ python manage.py migrate django_celery_results
```

接下来，让我们为我们的 Celery 工作程序创建一个配置文件。工作程序将需要访问 Django 和我们的 Celery 代理。

让我们在`django/config/celery.py`中创建 Celery 工作程序配置：

```py
import os
from celery import Celery

# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

app = Celery('mailape')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
```

Celery 知道如何与 Django 项目直接配合。在这里，我们根据 Django 配置配置了 Celery 库的一个实例。让我们详细审查这些设置：

+   `setdefault('DJANGO_SETTINGS_MODULE', ...)`：这确保我们的 Celery 工作程序知道如果未为`DJANGO_SETTINGS_MODULE`环境变量设置它，应该使用哪个 Django 设置模块。

+   `Celery('mailape')`：这实例化了 Mail Ape 的 Celery 库。大多数 Django 应用程序只使用一个 Celery 实例，因此`mailape`字符串并不重要。

+   `app.config_from_object('django.conf:settings', namespace='CELERY')`：这告诉我们的 Celery 库从`django.conf.settings`对象配置自身。`namespace`参数告诉 Celery 其设置以`CELERY`为前缀。

+   `app.autodiscover_tasks()`：这使我们可以避免手动注册任务。当 Celery 与 Django 一起工作时，它将检查每个已安装的应用程序是否有一个`tasks`模块。该模块中的任何任务都将被自动发现。

通过创建一个任务来发送确认电子邮件来了解更多关于任务的信息。

# 创建一个任务来发送确认电子邮件

现在 Celery 已配置好，让我们创建一个任务，向订阅者发送确认电子邮件。

Celery 任务是`Celery.app.task.Task`的子类。但是，当我们创建 Celery 任务时，大多数情况下，我们使用 Celery 的装饰器将函数标记为任务。在 Django 项目中，使用`shared_task`装饰器通常是最简单的。

创建任务时，将其视为视图是有用的。Django 社区的最佳实践建议*视图应该简单*，这意味着视图应该简单。它们不应该负责复杂的任务，而应该将该工作委托给模型或服务模块（例如我们的`mailinglist.emails`模块）。

任务函数保持简单，并将所有逻辑放在模型或服务模块中。

让我们在`django/mailinglist/tasks.py`中创建一个任务来发送我们的确认电子邮件：

```py
from celery import shared_task

from mailinglist import emails

@shared_task
def send_confirmation_email_to_subscriber(subscriber_id):
    from mailinglist.models import Subscriber
    subscriber = Subscriber.objects.get(id=subscriber_id)
    emails.send_confirmation_email(subscriber)
```

关于我们的`send_confirmation_email_to_subscriber`函数有一些独特的事情：

+   `@shared_task`：这是一个 Celery 装饰器，将函数转换为`Task`。`shared_task`对所有 Celery 实例都可用（在大多数 Django 情况下，通常只有一个）。

+   `def send_confirmation_email_to_subscriber(subscriber_id):`：这是一个常规函数，它以订阅者 ID 作为参数。Celery 任务可以接收任何可 pickle 的对象（包括 Django 模型）。但是，如果您传递的是可能被视为机密的内容（例如电子邮件地址），您可能希望限制存储数据的系统数量（例如，不要在代理商处存储）。在这种情况下，我们将任务函数传递给`Subscriber`的 ID，而不是完整的`Subscriber`。然后，任务函数查询相关的`Subscriber`实例的数据库。

在这个函数中最后要注意的一点是，我们在函数内部导入了`Subscriber`模型，而不是在文件顶部导入。在我们的情况下，我们的`Subscriber`模型将调用此任务。如果我们在`tasks.py`的顶部导入`models`模块，并在`model.py`的顶部导入`tasks`模块，那么就会出现循环导入错误。为了防止这种情况，我们在函数内部导入`Subscriber`。

接下来，让我们从`Subscriber.send_confirmation_email()`中调用我们的任务。

# 向新订阅者发送电子邮件

现在我们有了任务，让我们更新我们的`Subscriber`，使用任务发送确认电子邮件，而不是直接使用`emails`模块。

让我们更新`django/mailinglist/models.py`：

```py
from django.db import models
from mailinglist import tasks

class Subscriber(models.Model):
    # skipping unchanged model 

     def send_confirmation_email(self):
        tasks.send_confirmation_email_to_subscriber.delay(self.id)
```

在我们更新的`send_confirmation_email()`方法中，我们将看看如何异步调用任务。

Celery 任务可以同步或异步调用。使用常规的`()`运算符，我们将同步调用任务（例如，`tasks.send_confirmation_email_to_subscriber(self.id)`）。同步执行的任务就像常规的函数调用一样执行。

Celery 任务还有`delay()`方法来异步执行任务。当告诉任务要异步执行时，它将在 Celery 的消息代理中排队一条消息。然后 Celery 的 worker 将（最终）从代理的队列中拉取消息并执行任务。任务的结果存储在存储后端（在我们的情况下是 Django 数据库）中。

异步调用任务会返回一个`result`对象，它提供了一个`get()`方法。调用`result.get()`会阻塞当前线程，直到任务完成。然后`result.get()`返回任务的结果。在我们的情况下，我们的任务不会返回任何东西，所以我们不会使用`result`函数。

`task.delay(1, a='b')`实际上是`task.apply_async((1,), kwargs={'a':'b'})`的快捷方式。大多数情况下，快捷方法是我们想要的。如果您需要更多对任务执行的控制，`apply_async()`在 Celery 文档中有记录（[`docs.celeryproject.org/en/latest/userguide/calling.html`](http://docs.celeryproject.org/en/latest/userguide/calling.html)）。

现在我们可以调用任务了，让我们启动一个 worker 来处理我们排队的任务。

# 启动 Celery worker

启动 Celery worker 不需要我们编写任何新代码。我们可以从命令行启动一个：

```py
$ cd django
$ celery worker -A config.celery -l info
```

让我们看看我们给`celery`的所有参数：

+   `worker`: 这表示我们想要启动一个新的 worker。

+   `-A config.celery`: 这是我们想要使用的应用程序或配置。在我们的情况下，我们想要的应用程序在`config.celery`中配置。

+   `-l info`: 这是要输出的日志级别。在这种情况下，我们使用`info`。默认情况下，级别是`WARNING`。

我们的 worker 现在能够处理 Django 中我们的代码排队的任务。如果我们发现我们排队了很多任务，我们可以启动更多的`celery worker`进程。

# 快速回顾一下这一部分

在本节中，您学会了如何使用 Celery 来异步处理任务。

我们学会了如何在我们的`settings.py`中使用`CELERY_BROKER_URL`和`CELERY_RESULT_BACKEND`设置来设置代理和后端。我们还为我们的 celery worker 创建了一个`celery.py`文件。然后，我们使用`@shared_task`装饰器将函数变成了 Celery 任务。有了任务可用，我们学会了如何使用`.delay()`快捷方法调用 Celery 任务。最后，我们启动了一个 Celery worker 来执行排队的任务。

现在我们知道了基础知识，让我们使用这种方法向我们的订阅者发送消息。

# 向订阅者发送消息

在本节中，我们将创建代表用户想要发送到其邮件列表的消息的`Message`模型实例。

要发送这些消息，我们需要做以下事情：

+   创建一个`SubscriberMessage`模型来跟踪哪些消息何时发送

+   为与新的`Message`模型实例相关联的每个确认的`Subscriber`模型实例创建一个`SubscriberMessage`模型实例

+   让`SubscriberMessage`模型实例向其关联的`Subscriber`模型实例的电子邮件发送邮件。

为了确保即使有很多相关的`Subscriber`模型实例的`MailingList`模型实例也不会拖慢我们的网站，我们将使用 Celery 来构建我们的`SubscriberMessage`模型实例列表*并*发送电子邮件。

让我们首先创建一个`SubscriberManager`来帮助我们获取确认的`Subscriber`模型实例的列表。

# 获取确认的订阅者

良好的 Django 项目使用自定义模型管理器来集中和记录与其模型相关的`QuerySet`对象。我们需要一个`QuerySet`对象来检索属于给定`MailingList`模型实例的所有已确认`Subscriber`模型实例。

让我们更新`django/mailinglist/models.py`，添加一个新的`SubscriberManager`类，它知道如何为`MailingList`模型实例获取已确认的`Subscriber`模型实例：

```py
class SubscriberManager(models.Manager):

    def confirmed_subscribers_for_mailing_list(self, mailing_list):
        qs = self.get_queryset()
        qs = qs.filter(confirmed=True)
        qs = qs.filter(mailing_list=mailing_list)
        return qs

class Subscriber(models.Model):
    # skipped fields 

    objects = SubscriberManager()

    class Meta:
        unique_together = ['email', 'mailing_list', ]

    # skipped methods
```

我们的新`SubscriberManager`对象取代了`Subscriber.objects`中的默认管理器。`SubscriberManager`类提供了`confirmed_subscribers_for_mailing_list()`方法以及默认管理器的所有方法。

接下来，让我们创建`SubscriberMessage`模型。

# 创建 SubscriberMessage 模型

现在，我们将创建一个`SubscriberMessage`模型和管理器。`SubscriberMessage`模型将让我们跟踪是否成功向`Subscriber`模型实例发送了电子邮件。自定义管理器将具有一个方法，用于创建`Message`模型实例所需的所有`SubscriberMessage`模型实例。

让我们从`django/mailinglist/models.py`中创建我们的`SubscriberMessage`开始：

```py
import uuid

from django.conf import settings
from django.db import models

from mailinglist import tasks

class SubscriberMessage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.ForeignKey(to=Message, on_delete=models.CASCADE)
    subscriber = models.ForeignKey(to=Subscriber, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    sent = models.DateTimeField(default=None, null=True)
    last_attempt = models.DateTimeField(default=None, null=True)

    objects = SubscriberMessageManager()

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        is_new = self._state.adding or force_insert
        super().save(force_insert=force_insert, force_update=force_update, using=using,
             update_fields=update_fields)
        if is_new:
            self.send()

    def send(self):
        tasks.send_subscriber_message.delay(self.id)
```

与我们其他大部分模型相比，我们的`SubscriberMessage`模型定制程度相当高：

+   `SubsriberMessage`字段将其连接到`Message`和`Subscriber`，让它跟踪创建时间、最后尝试发送电子邮件以及成功与否。

+   `SubscriberMessage.objects`是我们将在下一节中创建的自定义管理器。

+   `SubscriberMessage.save()`与`Subscriber.save()`类似。它检查`SubscriberMessage`是否是新的，然后调用`send()`方法。

+   `SubscriberMessage.send()`排队一个任务来发送消息。我们将在*向订阅者发送电子邮件*部分稍后创建该任务。

现在，让我们在`django/mailinglist/models.py`中创建一个`SubscriberMessageManager`：

```py
from django.db import models

class SubscriberMessageManager(models.Manager):

    def create_from_message(self, message):
        confirmed_subs = Subscriber.objects.\
            confirmed_subscribers_for_mailing_list(message.mailing_list)
        return [
            self.create(message=message, subscriber=subscriber)
            for subscriber in confirmed_subs
        ]
```

我们的新管理器提供了一个从`Message`创建`SubscriberMessages`的方法。`create_from_message()`方法返回使用`Manager.create()`方法创建的`SubscriberMessage`列表。

最后，为了使新模型可用，我们需要创建一个迁移并应用它：

```py
$ cd django
$ python manage.py makemigrations mailinglist
$ python manage.py migrate mailinglist
```

现在我们有了`SubscriberMessage`模型和表，让我们更新我们的项目，以便在创建新的`Message`时自动创建`SubscriberMessage`模型实例。

# 创建消息时创建 SubscriberMessages

Mail Ape 旨在在创建后立即发送消息。为了使`Message`模型实例成为订阅者收件箱中的电子邮件，我们需要构建一组`SubscriberMessage`模型实例。构建该组`SubscriberMessage`模型实例的最佳时间是在创建新的`Message`模型实例之后。

让我们在`django/mailinglist/models.py`中重写`Message.save()`：

```py
class Message(models.Model):
    # skipped fields

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        is_new = self._state.adding or force_insert
        super().save(force_insert=force_insert, force_update=force_update,
                     using=using, update_fields=update_fields)
        if is_new:
            tasks.build_subscriber_messages_for_message.delay(self.id)
```

我们的新`Message.save()`方法遵循了与之前类似的模式。`Message.save()`检查当前的`Message`是否是新的，然后是否将`build_subscriber_messages_for_message`任务排队等待执行。

我们将使用 Celery 异步构建一组`SubscriberMessage`模型实例，因为我们不知道有多少`Subscriber`模型实例与我们的`MailingList`模型实例相关联。如果有很多相关的`Subscriber`模型实例，那么可能会使我们的 Web 服务器无响应。使用 Celery，我们的 Web 服务器将在`Message`模型实例保存后立即返回响应。`SubscriberMessage`模型实例将由一个完全独立的进程创建。

让我们在`django/mailinglist/tasks.py`中创建`build_subscriber_messages_for_message`任务：

```py
from celery import shared_task

@shared_task
def build_subscriber_messages_for_message(message_id):
    from mailinglist.models import Message, SubscriberMessage
    message = Message.objects.get(id=message_id)
    SubscriberMessage.objects.create_from_message(message)
```

正如我们之前讨论的，我们的任务本身并不包含太多逻辑。`build_subscriber_messages_for_message`让`SubscriberMessage`管理器封装了创建`SubscriberMessage`模型实例的所有逻辑。

接下来，让我们编写发送包含用户创建的`Message`的电子邮件的代码。

# 向订阅者发送电子邮件

本节的最后一步将是根据`SubscriberMessage`发送电子邮件。早些时候，我们的`SubscriberMessage.save()`方法排队了一个任务，向`Subscriber`发送`Message`。现在，我们将创建该任务并更新`emails.py`代码以发送电子邮件。

让我们从更新`django/mailinglist/tasks.py`开始一个新的任务：

```py
from celery import shared_task

@shared_task
def send_subscriber_message(subscriber_message_id):
    from mailinglist.models import SubscriberMessage
    subscriber_message = SubscriberMessage.objects.get(
        id=subscriber_message_id)
    emails.send_subscriber_message(subscriber_message)
```

这个新任务遵循了我们之前创建的任务的相同模式：

+   我们使用`shared_task`装饰器将常规函数转换为 Celery 任务

+   我们在任务函数内导入我们的模型，以防止循环导入错误

+   我们让`emails`模块来实际发送邮件

接下来，让我们更新`django/mailinglist/emails.py`文件，根据`SubscriberMessage`发送电子邮件：

```py
from datetime import datetime

from django.conf import settings
from django.core.mail import send_mail
from django.template import engines 
from django.utils.datetime_safe import datetime

SUBSCRIBER_MESSAGE_TXT = 'mailinglist/email/subscriber_message.txt'

SUBSCRIBER_MESSAGE_HTML = 'mailinglist/email/subscriber_message.html'

def send_subscriber_message(subscriber_message):
    message = subscriber_message.message
    context = EmailTemplateContext(subscriber_message.subscriber, {
        'body': message.body,
    })

    dt_engine = engines['django'].engine
    text_body_template = dt_engine.get_template(SUBSCRIBER_MESSAGE_TXT)
    text_body = text_body_template.render(context=context)
    html_body_template = dt_engine.get_template(SUBSCRIBER_MESSAGE_HTML)
    html_body = html_body_template.render(context=context)

    utcnow = datetime.utcnow()
    subscriber_message.last_attempt = utcnow
    subscriber_message.save()

    success = send_mail(
        subject=message.subject,
        message=text_body,
        from_email=settings.MAILING_LIST_FROM_EMAIL,
        recipient_list=(subscriber_message.subscriber.email,),
        html_message=html_body)

    if success == 1:
        subscriber_message.sent = utcnow
        subscriber_message.save()
```

我们的新函数采取以下步骤：

1.  使用我们之前创建的`EmailTemplateContext`类构建模板的上下文

1.  使用 Django 模板引擎呈现电子邮件的文本和 HTML 版本

1.  记录当前发送尝试的时间

1.  使用 Django 的`send_mail()`函数发送电子邮件

1.  如果`send_mail()`返回发送了一封电子邮件，它记录了消息发送的时间

我们的`send_subscriber_message()`函数要求我们创建 HTML 和文本模板来渲染。

让我们在`django/mailinglist/templates/mailinglist/email_templates/subscriber_message.html`中创建我们的 HTML 电子邮件正文模板：

```py
{% extends "mailinglist/email_templates/email_base.html" %}
{% load markdownify %}

{% block body %}
  {{ body | markdownify }}
{% endblock %}
```

这个模板将`Message`的 markdown 正文呈现为 HTML。我们以前使用过`markdownify`标签库来将 markdown 呈现为 HTML。我们不需要 HTML 样板或包含退订链接页脚，因为`email_base.html`已经包含了。

接下来，我们必须在`mailinglist/templates/mailinglist/email_templates/subscriber_message.txt`中创建消息模板的文本版本：

```py
{{ body }}

---

You're receiving this message because you previously subscribed to {{ mailinglist }}.

If you'd like to unsubsribe go to {{ unsubscription_link }} and click unsubscribe.

Sent with Mail Ape .
```

这个模板看起来非常相似。在这种情况下，我们只是将正文输出为未呈现的 markdown。此外，我们没有一个用于文本电子邮件的基本模板，所以我们必须手动编写包含退订链接的页脚。

恭喜！您现在已经更新了 Mail Ape，可以向邮件列表订阅者发送电子邮件。

确保在更改代码时重新启动您的`celery worker`进程。`celery worker`不像 Django`runserver`那样包含自动重启。如果我们不重新启动`worker`，那么它就不会得到任何更新的代码更改。

接下来，让我们确保我们可以在不触发 Celery 或发送实际电子邮件的情况下运行我们的测试。

# 测试使用 Celery 任务的代码

在这一点上，我们的两个模型将在创建时自动排队 Celery 任务。这可能会给我们在测试代码时造成问题，因为我们可能不希望在运行测试时运行 Celery 代理。相反，我们应该使用 Python 的`mock`库来防止在运行测试时需要运行外部系统。

我们可以使用的一种方法是使用 Python 的`@patch()`装饰器来装饰使用`Subscriber`或`Message`模型的每个测试方法。然而，这个手动过程很可能出错。让我们来看看一些替代方案。

在本节中，我们将看一下使模拟 Celery 任务更容易的两种方法：

+   使用 mixin 来防止`send_confirmation_email_to_subscriber`任务在任何测试中被排队

+   使用工厂来防止`send_confirmation_email_to_subscriber`任务被排队

通过以两种不同的方式解决相同的问题，您将了解到哪种解决方案在哪种情况下更有效。您可能会发现在项目中同时拥有这两个选项是有帮助的。

我们可以使用完全相同的方法来修补对`send_mail`的引用，以防止在测试期间发送邮件。

让我们首先使用一个 mixin 来应用一个补丁。

# 使用 TestCase mixin 来修补任务

在这种方法中，我们将创建一个 mixin，`TestCase`作者在编写`TestCase`时可以选择使用。我们在我们的 Django 代码中使用了许多 mixin 来覆盖基于类的视图的行为。现在，我们将创建一个 mixin，它将覆盖`TestCase`的默认行为。我们将利用每个测试方法之前调用`setUp()`和之后调用`tearDown()`的特性来设置我们的修补程序和模拟。

让我们在`django/mailinglist/tests.py`中创建我们的 mixin：

```py
from unittest.mock import patch

class MockSendEmailToSubscriberTask:

    def setUp(self):
        self.send_confirmation_email_patch = patch(
            'mailinglist.tasks.send_confirmation_email_to_subscriber')
        self.send_confirmation_email_mock = self.send_confirmation_email_patch.start()
        super().setUp()

    def tearDown(self):
        self.send_confirmation_email_patch.stop()
        self.send_confirmation_email_mock = None
        super().tearDown()
```

我们的 mixin 的`setUp()`方法做了三件事：

+   创建一个修补程序并将其保存为对象的属性

+   启动修补程序并将生成的模拟对象保存为对象的属性，访问模拟是重要的，这样我们以后可以断言它被调用了

+   调用父类的`setUp()`方法，以便正确设置`TestCase`

我们的 mixin 的`tearDown`方法还做了以下三件事：

+   停止修补程序

+   删除对模拟的引用

+   调用父类的`tearDown`方法来完成任何其他需要发生的清理

让我们创建一个`TestCase`来测试`SubscriberCreation`，并看看我们的新`MockSendEmailToSubscriberTask`是如何工作的。我们将创建一个测试，使用其管理器的`create()`方法创建一个`Subscriber`模型实例。`create()`调用将进而调用新的`Subscriber`实例的`save()`。`Subscriber.save()`方法应该排队一个`send_confirmation_email`任务。

让我们将我们的测试添加到`django/mailinglist/tests.py`中：

```py
from mailinglist.models import Subscriber, MailingList

from django.contrib.auth import get_user_model
from django.test import TestCase

class SubscriberCreationTestCase(
    MockSendEmailToSubscriberTask,
    TestCase):

    def test_calling_create_queues_confirmation_email_task(self):
        user = get_user_model().objects.create_user(
            username='unit test runner'
        )
        mailing_list = MailingList.objects.create(
            name='unit test',
            owner=user,
        )
        Subscriber.objects.create(
            email='unittest@example.com',
            mailing_list=mailing_list)
        self.assertEqual(self.send_confirmation_email_mock.delay.call_count, 1)
```

我们的测试断言我们在 mixin 中创建的模拟已经被调用了一次。这让我们确信当我们创建一个新的`Subscriber`时，我们将排队正确的任务。

接下来，让我们看看如何使用 Factory Boy 工厂来解决这个问题。

# 使用工厂进行修补

我们在第八章中讨论了使用 Factory Boy 工厂，*测试 Answerly*。工厂使得创建复杂对象变得更容易。现在让我们看看如何同时使用工厂和 Python 的`patch()`来防止任务被排队。

让我们在`django/mailinglist/factories.py`中创建一个`SubscriberFactory`：

```py
from unittest.mock import patch

import factory

from mailinglist.models import Subscriber

class SubscriberFactory(factory.DjangoModelFactory):
    email = factory.Sequence(lambda n: 'foo.%d@example.com' % n)

    class Meta:
        model = Subscriber

    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        with patch('mailinglist.models.tasks.send_confirmation_email_to_subscriber'):
            return super()._create(model_class=model_class, *args, **kwargs)
```

我们的工厂覆盖了默认的`_create()`方法，以在调用默认的`_create()`方法之前应用任务修补程序。当默认的`_create()`方法执行时，它将调用`Subscriber.save()`，后者将尝试排队`send_confirmation_email`任务。但是，该任务将被替换为模拟。一旦模型被创建并且`_create()`方法返回，修补程序将被移除。

现在我们可以在测试中使用我们的`SubscriberFactory`。让我们在`django/mailinglist/tests.py`中编写一个测试，以验证`SubscriberManager.confirmed_subscribers_for_mailing_list()`是否正确工作：

```py
from django.contrib.auth import get_user_model
from django.test import TestCase

from mailinglist.factories import SubscriberFactory
from mailinglist.models import Subscriber, MailingList

class SubscriberManagerTestCase(TestCase):

    def testConfirmedSubscribersForMailingList(self):
        mailing_list = MailingList.objects.create(
            name='unit test',
            owner=get_user_model().objects.create_user(
                username='unit test')
        )
        confirmed_users = [
            SubscriberFactory(confirmed=True, mailing_list=mailing_list)
            for n in range(3)]
        unconfirmed_users = [
            SubscriberFactory(mailing_list=mailing_list)
            for n in range(3)]
        confirmed_users_qs = Subscriber.objects.confirmed_subscribers_for_mailing_list(
            mailing_list=mailing_list)
        self.assertEqual(len(confirmed_users), confirmed_users_qs.count())
        for user in confirmed_users_qs:
            self.assertIn(user, confirmed_users)
```

现在我们已经看到了两种方法，让我们来看一下这两种方法之间的一些权衡。

# 在修补策略之间进行选择

Factory Boy 工厂和`TestCase` mixin 都帮助我们解决了如何测试排队 Celery 任务的代码而不排队 Celery 任务的问题。让我们更仔细地看一些权衡。

使用 mixin 时的一些权衡如下：

+   修补程序在整个测试期间保持不变

+   我们可以访问生成的模拟

+   修补程序将被应用在不需要它的测试上

+   我们`TestCase`中的 mixin 由我们在代码中引用的模型所决定，这对于测试作者来说可能是一种令人困惑的间接层次

使用工厂时的一些权衡如下：

+   如果需要，我们仍然可以访问测试中的基础函数。

+   我们无法访问生成的模拟来断言（我们通常不需要它）。

+   我们不将`TestCase`的`parent class`与我们在测试方法中引用的模型连接起来。对于测试作者来说更简单。

选择使用哪种方法的最终决定取决于我们正在编写的测试。

# 总结

在本章中，我们赋予了 Mail Ape 向我们用户的`MailingList`的确认`Subscribers`发送电子邮件的能力。我们还学会了如何使用 Celery 来处理 Django 请求/响应周期之外的任务。这使我们能够处理可能需要很长时间或需要其他资源（例如 SMTP 服务器和更多内存）的任务，而不会减慢我们的 Django Web 服务器。

本章我们涵盖了各种与电子邮件和 Celery 相关的主题。我们看到了如何配置 Django 来使用 SMTP 服务器。我们使用了 Django 的`send_email()`函数来发送电子邮件。我们使用`@shared_task`装饰器创建了一个 Celery 任务。我们使用了`delay()`方法将一个 Celery 任务加入队列。最后，我们探讨了一些有用的方法来测试依赖外部资源的代码。

接下来，让我们为我们的 Mail Ape 构建一个 API，这样我们的用户就可以将其集成到他们自己的网站和应用程序中。
