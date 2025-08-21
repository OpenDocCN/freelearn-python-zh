# 第十章：启动 Mail Ape

在本章中，我们将开始构建 Mail Ape，一个邮件列表管理器，让用户可以开始邮件列表、注册邮件列表，然后给人发消息。订阅者必须确认他们对邮件列表的订阅，并且能够取消订阅。这将帮助我们确保 Mail Ape 不被用来向用户发送垃圾邮件。

在本章中，我们将构建 Mail Ape 的核心 Django 功能：

+   我们将构建描述 Mail Ape 的模型，包括`MailingList`和`Subscriber`

+   我们将使用 Django 的基于类的视图来创建网页

+   我们将使用 Django 内置的身份验证功能让用户登录

+   我们将确保只有`MailingList`模型实例的所有者才能给其订阅者发送电子邮件

+   我们将创建模板来生成 HTML 以显示订阅和给用户发送电子邮件的表单

+   我们将使用 Django 内置的开发服务器在本地运行 Mail Ape

该项目的代码可在[`github.com/tomaratyn/MailApe`](https://github.com/tomaratyn/MailApe)上找到。

Django 遵循**模型视图模板**（**MVT**）模式，以分离模型、控制和表示逻辑，并鼓励可重用性。模型代表我们将在数据库中存储的数据。视图负责处理请求并返回响应。视图不应该包含 HTML。模板负责响应的主体和定义 HTML。这种责任的分离已被证明使编写代码变得容易。

让我们开始创建 Mail Ape 项目。

# 创建 Mail Ape 项目

在本节中，我们将创建 MailApe 项目：

```py
$ mkdir mailape
$ cd mailape
```

本书中的所有路径都将相对于此目录。

# 列出我们的 Python 依赖项

接下来，让我们创建一个`requirements.txt`文件来跟踪我们的 Python 依赖项：

```py
django<2.1
psycopg2<2.8
django-markdownify==0.3.0
django-crispy-forms==1.7.0
```

现在我们知道我们的需求，我们可以按照以下方式安装它们：

```py
$ pip install -r requirements.txt
```

这将安装以下四个库：

+   `Django`：我们最喜欢的 Web 应用程序框架

+   `psycopg2`：Python PostgreSQL 库；我们将在生产和开发中都使用 PostgreSQL

+   `django-markdownify`：一个使在 Django 模板中呈现 markdown 变得容易的库

+   `django-crsipy-forms`：一个使在模板中创建 Django 表单变得容易的库

有了 Django 安装，我们可以使用`django-admin`实用程序来创建我们的项目。

# 创建我们的 Django 项目和应用程序

Django 项目由配置目录和一个或多个 Django 应用程序组成。已安装的应用程序封装了项目的实际功能。默认情况下，配置目录以项目命名。

Web 应用程序通常由远不止执行的 Django 代码组成。我们需要配置文件、系统依赖和文档。为了帮助未来的开发人员（包括我们未来的自己），我们将努力清晰地标记每个目录：

```py
$ django-admin startporject config
$ mv config django
$ tree django
django
├── config
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── manage.py
```

通过这种方法，我们的目录结构清楚地指明了我们的 Django 代码和配置的位置。

接下来，让我们创建将封装我们功能的应用程序：

```py
$ python manage.py startapp mailinglist
$ python manage.py startapp user
```

对于每个应用程序，我们应该创建一个 URLConf。URLConf 确保请求被路由到正确的视图。URLConf 是路径列表，提供路径的视图和路径的名称。URLConfs 的一个很棒的功能是它们可以相互包含。当创建 Django 项目时，它会得到一个根 URLConf（我们的在`django/config/urls.py`）。由于 URLConf 可能包含其他 URLConfs，名称提供了一种重要的方式来引用 URL 路径到视图，而不需要知道视图的完整 URL 路径。

# 创建我们应用的 URLConfs

让我们为`mailinglist`应用程序创建一个 URLConf，位于`django/mailinglist/urls.py`中：

```py
from django.urls import path

from mailinglist import views

app_name = 'mailinglist'

urlpatterns = [
]
```

`app_name`变量用于在名称冲突的情况下限定路径。在解析路径名时，我们可以使用`mailinglist:`前缀来确保它来自此应用程序。随着我们构建视图，我们将向`urlpatterns`列表添加`path`。

接下来，让我们通过创建`django/user/urls.py`为`user`应用程序创建另一个 URLConf：

```py
from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path

import user.views

app_name = 'user'
urlpatterns = [
]
```

太棒了！现在，让我们将它们包含在位于`django/config/urls.py`中的根 ULRConf 中：

```py
from django.contrib import admin
from django.urls import path, include

import mailinglist.urls
import user.urls

urlpatterns = [
    path('admin/', admin.site.urls),
    path('user/', include(user.urls, namespace='user')),
    path('mailinglist/', include(mailinglist.urls, namespace='mailinglist')),
]
```

根 URLConf 就像我们应用程序的 URLConfs 一样。它有一个`path()`对象的列表。根 URLConfs 中的`path()`对象通常没有视图，而是`include()`其他 URLConfs。让我们来看看这里的两个新函数：

+   `path()`: 这需要一个字符串和一个视图或`include()`的结果。Django 将在 URLConf 中迭代`path()`，直到找到与请求路径匹配的路径。然后 Django 将请求传递给该视图或 URLConf。如果是 URLConf，则会检查`path()`的列表。

+   `include()`: 这需要一个 URLConf 和一个命名空间名称。命名空间将 URLConfs 相互隔离，以便我们可以防止名称冲突，确保我们可以区分`appA:index`和`appB:index`。`include()`返回一个元组；`admin.site.urls`上的对象已经是一个正确格式的元组，所以我们不必使用`include()`。通常，我们总是使用`include()`。

如果 Django 找不到与请求路径匹配的`path()`对象，那么它将返回 404 响应。

这个 URLConf 的结果如下：

+   任何以`admin/`开头的请求将被路由到管理员应用的 URLConf

+   任何以`mailinglist/`开头的请求将被路由到`mailinglist`应用的 URLConf

+   任何以`user/`开头的请求将被路由到`user`应用的 URLConf

# 安装我们项目的应用程序

让我们更新`django/config/settings.py`以安装我们的应用程序。我们将更改`INSTALLED_APPS`设置，如下面的代码片段所示：

```py
INSTALLED_APPS = [
    'user',
    'mailinglist',

    'crispy_forms',
    'markdownify',

    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]
```

现在我们已经配置好了我们的项目和应用程序，让我们为我们的`mailinglist`应用创建模型。

# 创建邮件列表模型

在这一部分，我们将为我们的`mailinglist`应用创建模型。Django 提供了丰富而强大的 ORM，让我们能够在 Python 中定义我们的模型，而不必直接处理数据库。ORM 将我们的 Django 类、字段和对象转换为关系数据库概念：

+   模型类映射到关系数据库表

+   字段映射到关系数据库列

+   模型实例映射到关系数据库行

每个模型还带有一个默认的管理器，可在`objects`属性中使用。管理器提供了在模型上运行查询的起点。管理器最重要的方法之一是`create()`。我们可以使用`create()`在数据库中创建模型的实例。管理器也是获取模型的`QuerySet`的起点。

`QuerySet`代表模型的数据库查询。`QuerySet`是惰性的，只有在迭代或转换为`bool`时才执行。`QuerySet` API 提供了大部分 SQL 的功能，而不与特定的数据库绑定。两个特别有用的方法是`QuerySet.filter()`和`QuerySet.exclude()`。`QuerySet.filter()`让我们将`QuerySet`的结果过滤为只匹配提供的条件的结果。`QuerySet.exclude()`让我们排除不匹配条件的结果。

让我们从第一个模型`MailingList`开始。

# 创建邮件列表模型

我们的`MailingList`模型将代表我们的一个用户创建的邮件列表。这将是我们系统中的一个重要模型，因为许多其他模型将引用它。我们还可以预期`MailingList`的`id`将需要公开暴露，以便将订阅者关联回来。为了避免让用户枚举 Mail Ape 中的所有邮件列表，我们希望确保我们的`MailingList` ID 是非顺序的。

让我们将我们的`MailingList`模型添加到`django/mailinglist/models.py`中：

```py
import uuid

from django.conf import settings
from django.db import models
from django.urls import reverse

class MailingList(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=140)
    owner = models.ForeignKey(to=settings.AUTH_USER_MODEL,
                              on_delete=models.CASCADE)

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse(
            'mailinglist:manage_mailinglist',
            kwargs={'pk': self.id}
        )

    def user_can_use_mailing_list(self, user):
        return user == self.owner
```

让我们更仔细地看看我们的`MailingList`模型：

+   `class MailingList(models.Model):`：所有 Django 模型都必须继承自`Model`类。

+   `id = models.UUIDField`: 这是我们第一次为模型指定`id`字段。通常，我们让 Django 自动为我们提供一个。在这种情况下，我们想要非顺序的 ID，所以我们使用了一个提供**通用唯一标识符**（**UUID**）的字段。Django 将在我们生成迁移时创建适当的数据库字段（参考*创建数据库迁移*部分）。然而，我们必须在 Python 中生成 UUID。为了为每个新模型生成新的 UUID，我们使用了`default`参数和 Python 的`uuid4`函数。为了告诉 Django 我们的`id`字段是主键，我们使用了`primary_key`参数。我们进一步传递了`editable=False`以防止对`id`属性的更改。

+   `name = models.CharField`: 这将代表邮件列表的名称。`CharField`将被转换为`VARCHAR`列，所以我们必须为它提供一个`max_length`参数。

+   `owner = models.ForeignKey`: 这是对 Django 用户模型的外键。在我们的情况下，我们将使用默认的`django.contrib.auth.models.User`类。我们遵循 Django 避免硬编码这个模型的最佳实践。通过引用`settings.AUTH_USER_MODEL`，我们不会将我们的应用程序与项目过于紧密地耦合。这鼓励未来的重用。`on_delete=models.CASCADE`参数意味着如果用户被删除，他们的所有`MailingList`模型实例也将被删除。

+   `def __str__(self)`: 这定义了如何将邮件列表转换为`str`。当需要打印或显示`MailingList`时，Django 和 Python 都会使用这个方法。

+   `def get_absolute_url(self)`: 这是 Django 模型上的一个常见方法。`get_absolute_url()`返回代表模型的 URL 路径。在我们的情况下，我们返回这个邮件列表的管理页面。我们不会硬编码路径。相反，我们使用`reverse()`在运行时解析路径，提供 URL 的名称。我们将在*创建 URLConf*部分讨论命名 URL。

+   `def user_can_use_mailing_list(self, user)`: 这是我们为自己方便添加的一个方法。它检查用户是否可以使用（查看相关项目和/或发送消息）到这个邮件列表。Django 的*Fat models*哲学鼓励将这样的决策代码放在模型中，而不是在视图中。这为我们提供了一个决策的中心位置，确保**不要重复自己**（**DRY**）。

现在我们有了我们的`MailingList`模型。接下来，让我们创建一个模型来捕获邮件列表的订阅者。

# 创建`Subscriber`模型

在这一部分，我们将创建一个`Subscriber`模型。`Subscriber`模型只能属于一个`MailingList`，并且必须确认他们的订阅。由于我们需要引用订阅者以获取他们的确认和取消订阅页面，我们希望他们的`id`实例也是非顺序的。

让我们在`django/mailinglist/models.py`中创建`Subscriber`模型。

```py
class Subscriber(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField()
    confirmed = models.BooleanField(default=False)
    mailing_list = models.ForeignKey(to=MailingList, on_delete=models.CASCADE)

    class Meta:
        unique_together = ['email', 'mailing_list', ]
```

`Subscriber`模型与`MailingList`模型有一些相似之处。基类和`UUIDField`的功能相同。让我们看看一些不同之处：

+   `models.EmailField()`: 这是一个专门的`CharField`，但会进行额外的验证，以确保值是一个有效的电子邮件地址。

+   `models.BooleanField(default=False)`: 这让我们存储`True`/`False`值。我们需要使用这个来跟踪用户是否真的打算订阅邮件列表。

+   `models.ForeignKey(to=MailingList...)`: 这让我们在`Subscriber`和`MailingList`模型实例之间创建一个外键。

+   `unique_together`: 这是`Subscriber`的`Meta`内部类的一个属性。`Meta`内部类让我们可以在表上指定信息。例如，`unique_together`让我们在表上添加额外的唯一约束。在这种情况下，我们防止用户使用相同的电子邮件地址注册两次。

现在我们可以跟踪`Subscriber`模型实例了，让我们跟踪用户想要发送到他们的`MailingList`的消息。

# 创建`Message`模型

我们的用户将希望向他们的`MailingList`的`Subscriber`模型实例发送消息。为了知道要发送给这些订阅者什么，我们需要将消息存储为 Django 模型。

`Message`应该属于`MailingList`并具有非连续的`id`。我们需要保存这些消息的主题和正文。我们还希望跟踪发送开始和完成的时间。

让我们将`Message`模型添加到`django/mailinglist/models.py`中：

```py
class Message(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    mailing_list = models.ForeignKey(to=MailingList, on_delete=models.CASCADE)
    subject = models.CharField(max_length=140)
    body = models.TextField()
    started = models.DateTimeField(default=None, null=True)
    finished = models.DateTimeField(default=None, null=True)
```

再次，`Message`模型在其基类和字段上与我们之前的模型非常相似。我们在这个模型中看到了一些新字段。让我们更仔细地看看这些新字段：

+   `models.TextField()`: 用于存储任意长的字符数据。所有主要数据库都有`TEXT`列类型。这对于存储用户的`Message`的`body`属性非常有用。

+   `models.DateTimeField(default=None, null=True)`: 用于存储日期和时间值。在 Postgres 中，这将成为`TIMESTAMP`列。`null`参数告诉 Django 该列应该能够接受`NULL`值。默认情况下，所有字段都对它们有一个`NOT NULL`约束。

我们现在有了我们的模型。让我们使用数据库迁移在我们的数据库中创建它们。

# 使用数据库迁移

数据库迁移描述了如何将数据库转换为特定状态。在本节中，我们将做以下事情：

+   为我们的`mailinglist`应用程序模型创建数据库迁移

+   在 Postgres 数据库上运行迁移

当我们对模型进行更改时，我们可以让 Django 生成用于创建这些表、字段和约束的代码。Django 生成的迁移是使用 Django 开发人员也可以使用的 API 创建的。如果我们需要进行复杂的迁移，我们可以自己编写迁移。请记住，正确的迁移包括应用和撤消迁移的代码。如果出现问题，我们希望有一种方法来撤消我们的迁移。当 Django 生成迁移时，它总是为我们生成两个迁移。

让我们首先配置 Django 连接到我们的 PostgreSQL 数据库。

# 配置数据库

要配置 Django 连接到我们的 Postgres 数据库，我们需要更新`django/config/settings.py`中的`DATABASES`设置：

```py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'mailape',
        'USER': 'mailape',
        'PASSWORD': 'development',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

您不应该在`settings.py`文件中将密码硬编码到生产数据库中。如果您连接到共享或在线实例，请使用环境变量设置用户名、密码和主机，并使用`os.getenv()`访问它们，就像我们在之前的生产部署章节中所做的那样（第五章，“使用 Docker 部署”和第九章，*部署 Answerly*）。

Django 不能自行创建数据库和用户。我们必须自己做。您可以在本章的代码中找到执行此操作的脚本。

接下来，让我们为模型创建迁移。

# 创建数据库迁移

要创建我们的数据库迁移，我们将使用 Django 放在 Django 项目顶部的`manage.py`脚本（`django/manage.py`）：

```py
$ cd django
$ python manage.py makemigrations
Migrations for 'mailinglist':
  mailinglist/migrations/0001_initial.py
    - Create model MailingList
    - Create model Message
    - Create model Subscriber
    - Alter unique_together for subscriber (1 constraint(s))
```

太棒了！现在我们有了迁移，我们可以在我们的本地开发数据库上运行它们。

# 运行数据库迁移

我们使用`manage.py`将我们的数据库迁移应用到正在运行的数据库。在命令行上执行以下操作：

```py
$ cd django
$ python manage.py migrate
Operations to perform:
  Apply all migrations: admin, auth, contenttypes, mailinglist, sessions
Running migrations:
  Applying contenttypes.0001_initial... OK
  Applying auth.0001_initial... OK
  Applying admin.0001_initial... OK
  Applying admin.0002_logentry_remove_auto_add... OK
  Applying contenttypes.0002_remove_content_type_name... OK
  Applying auth.0002_alter_permission_name_max_length... OK
  Applying auth.0003_alter_user_email_max_length... OK
  Applying auth.0004_alter_user_username_opts... OK
  Applying auth.0005_alter_user_last_login_null... OK
  Applying auth.0006_require_contenttypes_0002... OK
  Applying auth.0007_alter_validators_add_error_messages... OK
  Applying auth.0008_alter_user_username_max_length... OK
  Applying auth.0009_alter_user_last_name_max_length... OK
  Applying mailinglist.0001_initial... OK
  Applying sessions.0001_initial... OK
```

当我们运行`manage.py migrate`而不提供应用程序时，它将在所有安装的 Django 应用程序上运行所有迁移。我们的数据库现在具有`mailinglist`应用程序模型和`auth`应用程序模型（包括`User`模型）的表。

现在我们有了我们的模型和数据库设置，让我们确保我们可以使用 Django 的表单 API 验证这些模型的用户输入。

# 邮件列表表单

开发人员必须解决的一个常见问题是如何验证用户输入。Django 通过其表单 API 提供输入验证。表单 API 可用于使用与模型 API 非常相似的 API 描述 HTML 表单。如果我们想创建描述 Django 模型的表单，那么 Django 表单的`ModelForm`为我们提供了一种快捷方式。我们只需要描述我们从默认表单表示中更改的内容。

当实例化 Django 表单时，可以提供以下三个参数中的任何一个：

+   `data`：最终用户请求的原始输入

+   `initial`：我们可以为表单设置的已知安全初始值

+   `instance`：表单描述的实例，仅在`ModelForm`中

如果表单提供了`data`，那么它被称为绑定表单。绑定表单可以通过调用`is_valid()`来验证它们的`data`。经过验证的表单的安全数据可以在`cleaned_data`字典下使用（以字段名称为键）。错误可以通过`errors`属性获得，它返回一个字典。绑定的`ModelForm`也可以使用`save()`方法创建或更新其模型实例。

即使没有提供任何参数，表单仍然能够以 HTML 形式打印自己，使我们的模板更简单。这种机制帮助我们实现了“愚蠢模板”的目标。

让我们通过创建`SubscriberForm`类来开始创建我们的表单。

# 创建订阅者表单

Mail Ape 必须执行的一个重要任务是接受新的`Subscriber`的邮件，用于`MailingList`。让我们创建一个表单来进行验证。

`SubscriberForm`必须能够验证输入是否为有效的电子邮件。我们还希望它保存我们的新`Subscriber`模型实例并将其与适当的`MailingList`模型实例关联起来。

让我们在`django/mailinglist/forms.py`中创建该表单：

```py
from django import forms

from mailinglist.models import MailingList, Subscriber

class SubscriberForm(forms.ModelForm):
    mailing_list = forms.ModelChoiceField(
        widget=forms.HiddenInput,
        queryset=MailingList.objects.all(),
        disabled=True,
    )

    class Meta:
        model = Subscriber
        fields = ['mailing_list', 'email', ]
```

让我们仔细看看我们的`SubscriberForm`：

+   `class SubscriberForm(forms.ModelForm):`：这表明我们的表单是从`ModelForm`派生的。`ModelForm`知道要检查我们的内部`Meta`类，以获取关于可以用作此表单基础的模型和字段的信息。

+   `mailing_list = forms.ModelChoiceField`：这告诉我们的表单使用我们自定义配置的`ModelChoiceField`，而不是表单 API 默认使用的。默认情况下，Django 将显示一个`ModelChoiceField`，它将呈现为下拉框。用户可以使用下拉框选择相关的模型。在我们的情况下，我们不希望用户能够做出选择。当我们显示一个渲染的`SubscriberForm`时，我们希望它配置为特定的邮件列表。为此，我们将`widget`参数更改为`HiddenInput`类，并将字段标记为`disabled`。我们的表单需要知道对于该表单有效的`MailingList`模型实例。我们提供一个匹配所有`MailingList`模型实例的`QuerySet`对象。

+   `model = Subscriber`：这告诉表单的`Meta`内部类，这个表单是基于`Subscriber`模型的。

+   `fields = ['mailing_list', 'email', ]`：这告诉表单只包括模型中的以下字段。

接下来，让我们创建一个表单，用于捕获我们的用户想要发送到他们的`MailingList`的`Message`。

# 创建消息表单

我们的用户将希望向他们的`MailingList`发送`Message`。我们将提供一个网页，用户可以在其中创建这些消息的表单。在我们创建页面之前，让我们先创建表单。

让我们将我们的`MessageForm`类添加到`django/mailinglist/forms.py`中：

```py
from django import forms

from mailinglist.models import MailingList, Message

class MessageForm(forms.ModelForm):
    mailing_list = forms.ModelChoiceField(
        widget=forms.HiddenInput,
        queryset=MailingList.objects.all(),
        disabled=True,
    )

    class Meta:
        model = Message
        fields = ['mailing_list', 'subject', 'body', ]
```

正如您在前面的代码中所注意到的，`MessageForm`的工作方式与`SubscriberFrom`相同。唯一的区别是我们在`Meta`内部类中列出了不同的模型和不同的字段。

接下来，让我们创建`MailingListForm`类，我们将用它来接受邮件列表的名称的输入。

# 创建邮件列表表单

现在，我们将创建一个`MailingListForm`，它将接受邮件列表的名称和所有者。我们将在`owner`字段上使用与之前相同的`HiddenInput`和`disabled`字段模式。我们希望确保用户无法更改邮件列表的所有者。

让我们将我们的表单添加到`django/mailinglist/forms.py`中：

```py
from django import forms
from django.contrib.auth import get_user_model

from mailinglist.models import MailingList

class MailingListForm(forms.ModelForm):
    owner = forms.ModelChoiceField(
        widget=forms.HiddenInput,
        queryset=get_user_model().objects.all(),
        disabled=True,
    )

    class Meta:
        model = MailingList
        fields = ['owner', 'name']
```

`MailingListForm`与我们之前的表单非常相似，但引入了一个新的函数`get_user_model()`。我们需要使用`get_user_model()`，因为我们不想将自己与特定的用户模型耦合在一起，但我们需要访问该模型的管理器以获取`QuerySet`。

现在我们有了我们的表单，我们可以为我们的`mailinglist` Django 应用程序创建视图。

# 创建邮件列表视图和模板

在前面的部分中，我们创建了可以用来收集和验证用户输入的表单。在本节中，我们将创建实际与用户通信的视图和模板。模板定义了文档的 HTML。

基本上，Django 视图是一个接受请求并返回响应的函数。虽然我们在本书中不会使用这些**基于函数的视图**（**FBVs**），但重要的是要记住，一个视图只需要满足这两个责任。如果处理视图还导致其他操作发生（例如，发送电子邮件），那么我们应该将该代码放在服务模块中，而不是直接放在视图中。

Web 开发人员面临的许多工作是重复的（例如，处理表单，显示特定模型，列出该模型的所有实例等）。Django 的“电池包含”哲学意味着它包含了工具，使这些重复的任务更容易。

Django 通过提供丰富的**基于类的视图**（**CBVs**）使常见的 Web 开发人员任务更容易。CBVs 使用**面向对象编程**（**OOP**）的原则来增加代码重用。Django 提供了丰富的 CBV 套件，使处理表单或为模型实例显示 HTML 页面变得容易。

HTML 视图返回的内容来自于渲染模板。Django 中的模板通常是用 Django 的模板语言编写的。Django 也可以支持其他模板语言（例如 Jinja）。通常，每个视图都与一个模板相关联。

让我们首先创建许多视图将需要的一些资源。

# 常见资源

在这一部分，我们将创建一些我们的视图和模板将需要的常见资源：

+   我们将创建一个基础模板，所有其他模板都可以扩展。在所有页面上使用相同的基础模板将给 Mail Ape 一个统一的外观和感觉。

+   我们将创建一个`MailingListOwnerMixin`类，它将让我们保护邮件列表消息免受未经授权的访问。

让我们从创建一个基础模板开始。

# 创建基础模板

让我们为 Mail Ape 创建一个基础模板。这个模板将被我们所有的页面使用，以给我们整个 Web 应用程序一个一致的外观。

**Django 模板语言**（**DTL**）让我们编写 HTML（或其他基于文本的格式），并让我们使用*标签*、*变量*和*过滤器*来执行代码以定制 HTML。让我们更仔细地看看这三个概念：

+   *标签*：它们被`{% %}`包围，可能（`{% block body%}{% endblock %}`）或可能不（`{% url "myurl" %}`）包含一个主体。

+   *variables*：它们被`{{ }}`包围，并且必须在模板的上下文中设置（例如，`{{ mailinglist }}`）。尽管 DTL 变量类似于 Python 变量，但也有区别。最关键的两个区别在于可执行文件和字典。首先，DTL 没有语法来传递参数给可执行文件（你永远不必使用`{{foo(1)}}`）。如果你引用一个变量并且它是可调用的（例如，一个函数），那么 Django 模板语言将调用它并返回结果（例如，`{{mailinglist.get_absolute_url}}`）。其次，DTL 不区分对象属性、列表中的项目和字典中的项目。所有这三个都使用点来访问：`{{mailinglist.name}}`，`{{mylist.1}}`和`{{mydict.mykey}}`。

+   *filters*：它们跟随一个变量并修改其值（例如，`{{ mailinglist.name | upper}}`将以大写形式返回邮件列表的名称）。

我们将在继续创建 Mail Ape 时查看这三个示例。

让我们创建一个公共模板目录—`django/templates`—并将我们的模板放在`django/templates/base.html`中：

```py
<!DOCTYPE html>
<html lang="en" >
<head >
  <meta charset="UTF-8" >
  <title >{% block title %}{% endblock %}</title >
  <link rel="stylesheet"
        href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.3/css/bootstrap.min.css"
  />
</head >
<body >
<div class="container" >
  <nav class="navbar navbar-light bg-light" >
    <a class="navbar-brand" href="#" >Mail Ape </a >
    <ul class="navbar-nav" >
      <li class="nav-item" >
        <a class="nav-link"
           href="{% url "mailinglist:mailinglist_list" %}" >
          Your Mailing Lists
        </a >
      </li >
      {% if request.user.is_authenticated %}
        <li class="nav-item" >
          <a class="nav-link"
             href="{% url "user:logout" %}" >
            Logout
          </a >
        </li >
      {% else %}
        <li class="nav-item" >
          <a class="nav-link"
             href="{% url "user:login" %}" >
            Your Mailing Lists
          </a >
        </li >
        <li class="nav-item" >
          <a class="nav-link"
             href="{% url "user:register" %}" >
            Your Mailing Lists
          </a >
        </li >
      {% endif %}
    </ul >
  </nav >
  {% block body %}
  {% endblock %}
</div >
</body >
</html >
```

在我们的基本模板中，我们将注意以下三个标签的示例：

+   `{% url ... %}`：这返回到视图的路径。这与我们之前看到的`reverse()`函数在 Django 模板中的工作方式相同。

+   `{% if ... %} ... {% else %} ... {% endif %}`：这与 Python 开发人员期望的工作方式相同。`{% else %}`子句是可选的。Django 模板语言还支持`{% elif ... %}`，如果我们需要在多个选择中进行选择。

+   `{% block ... %}`：这定义了一个块，一个扩展`base.html`的模板可以用自己的内容替换。我们有两个块，`body`和`title`。

我们现在有一个基本模板，我们的其他模板可以通过提供 body 和 title 块来使用。

既然我们有了模板，我们必须告诉 Django 在哪里找到它。让我们更新`django/config/settings.py`，让 Django 知道我们的新`django/templates`目录。

在`django/config/settings.py`中，找到以`Templates`开头的行。我们需要将我们的`templates`目录添加到`DIRS`键下的列表中：

```py
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            os.path.join(BASE_DIR, 'templates'),
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            # do not change OPTIONS, omitted for brevity
        },
    },
]
```

Django 让我们通过在运行时计算`BASE_DIR`来避免将路径硬编码到`django/templates`中。这样，我们可以在不同的环境中使用相同的设置。

我们刚刚看到的另一个重要设置是`APP_DIRS`。这个设置告诉 Django 在查找模板时检查每个安装的应用程序的`templates`目录。这意味着我们不必为每个安装的应用程序更新`DIRS`键，并且让我们将模板隔离在我们的应用程序下（增加可重用性）。最后，重要的是要记住应用程序按照它们在`INSTALLED_APPS`中出现的顺序进行搜索。如果有模板名称冲突（例如，两个应用程序提供名为`registration/login.html`的模板），那么将使用`INSTALLED_APPS`中列出的第一个。

接下来，让我们配置我们的项目在呈现 HTML 表单时使用 Bootstrap 4。

# 配置 Django Crispy Forms 以使用 Bootstrap 4

在我们的基本模板中，我们包含了 Bootstrap 4 的 css 模板。为了方便使用 Bootstrap 4 呈现表单并为其设置样式，我们将使用一个名为 Django Crispy Forms 的第三方 Django 应用程序。但是，我们必须配置 Django Crispy Forms 以告诉它使用 Bootstrap 4。

让我们在`django/config/settings.py`的底部添加一个新的设置：

```py
CRISPY_TEMPLATE_PACK = 'bootstrap4'
```

现在，Django Crispy Forms 配置为在呈现表单时使用 Bootstrap 4。我们将在本章后面的部分中查看它，在涵盖在模板中呈现表单的部分。

接下来，让我们创建一个 mixin，确保只有邮件列表的所有者才能影响它们。

# 创建一个 mixin 来检查用户是否可以使用邮件列表

Django 使用**基于类的视图**（**CBVs**）使代码重用更容易，简化重复的任务。在`mailinglist`应用程序中，我们将不得不做的重复任务之一是保护`MailingList`及其相关模型，以免被其他用户篡改。我们将创建一个 mixin 来提供保护。

mixin 是一个提供有限功能的类，旨在与其他类一起使用。我们之前见过`LoginRequired` mixin，它可以与视图类一起使用，以保护视图免受未经身份验证的访问。在本节中，我们将创建一个新的 mixin。

让我们在`django/mailinglist/mixins.py`中创建我们的`UserCanUseMailingList` mixin：

```py
from django.core.exceptions import PermissionDenied, FieldDoesNotExist

from mailinglist.models import MailingList

class UserCanUseMailingList:

    def get_object(self, queryset=None):
        obj = super().get_object(queryset)
        user = self.request.user
        if isinstance(obj, MailingList):
            if obj.user_can_use_mailing_list(user):
                return obj
            else:
                raise PermissionDenied()

        mailing_list_attr = getattr(obj, 'mailing_list')
        if isinstance(mailing_list_attr, MailingList):
            if mailing_list_attr.user_can_use_mailing_list(user):
                return obj
            else:
                raise PermissionDenied()
        raise FieldDoesNotExist('view does not know how to get mailing '
                                   'list.')
```

我们的类定义了一个方法，`get_object(self, queryset=None)`。这个方法与`SingleObjectMixin.get_object()`具有相同的签名，许多 Django 内置的 CBV（例如`DetailView`）使用它。我们的`get_object()`实现不做任何工作来检索对象。相反，我们的`get_object`只是检查父对象检索到的对象，以检查它是否是或者拥有`MailingList`，并确认已登录的用户可以使用邮件列表。

mixin 的一个令人惊讶的地方是它依赖于一个超类，但不继承自一个。在`get_object()`中，我们明确调用`super()`，但`UserCanUseMailingList`没有任何基类。mixin 类不希望单独使用。相反，它们将被类使用，这些类子类化它们*和*一个或多个其他类。

我们将在接下来的几节中看看这是如何工作的。

# 创建 MailingList 视图和模板

现在，让我们来看看将处理用户请求并返回从我们的模板创建的 UI 的响应的视图。

让我们首先创建一个列出所有我们的`MailingList`的视图。

# 创建 MailingListListView 视图

我们将创建一个视图，显示用户拥有的邮件列表。

让我们在`django/mailinglist/views.py`中创建我们的`MailingListListView`：

```py
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView

from mailinglist.models import  MailingList

class MailingListListView(LoginRequiredMixin, ListView):

    def get_queryset(self):
        return MailingList.objects.filter(owner=self.request.user)
```

我们的观点源自两个视图，`LoginRequiredMixin`和`ListView`。`LoginRequiredMixin`是一个 mixin，确保未经身份验证的用户发出的请求被重定向到登录视图，而不是被处理。为了帮助`ListView`知道*要*列出什么，我们将重写`get_queryset()`方法，并返回一个包含当前登录用户拥有的`MailingList`的`QuerySet`。为了显示结果，`ListView`将尝试在`appname/modelname_list.html`渲染模板。在我们的情况下，`ListView`将尝试渲染`mailinglist/mailinglist_list.html`。

让我们在`django/mailinglist/templates/mailinglist/mailinglist_list.html`中创建该模板：

```py
{% extends "base.html" %}

{% block title %}
  Your Mailing Lists
{% endblock %}

{% block body %}
  <div class="row user-mailing-lists" >
    <div class="col-sm-12" >
      <h1 >Your Mailing Lists</h1 >
      <div >
        <a class="btn btn-primary"
           href="{% url "mailinglist:create_mailinglist" %}" >New List</a >
      </div >
      <p > Your mailing lists:</p >
      <ul class="mailing-list-list">
        {% for mailinglist in mailinglist_list %}
          <li class="mailinglist-item">
            <a href="{% url "mailinglist:manage_mailinglist" pk=mailinglist.id %}" >
              {{ mailinglist.name }}
            </a >
          </li >
        {% endfor %}
      </ul >
    </div >
  </div >
{% endblock %}
```

我们的模板扩展了`base.html`。当一个模板扩展另一个模板时，它只能将 HTML 放入先前定义的`block`中。我们还将看到许多新的 Django 模板标签。让我们仔细看看它们：

+   `{% extends "base.html" %}`：这告诉 Django 模板语言我们正在扩展哪个模板。

+   `{% block title %}… {% endblock %}`：这告诉 Django 我们正在提供新的代码，它应该放在扩展模板的`title`块中。该块中的先前代码（如果有）将被替换。

+   `{% for mailinglist in mailinglist_list %} ... {% endfor %}`：这为列表中的每个项目提供了一个循环。

+   `{% url … %}`：`url`标签将为命名的`path`生成 URL 路径。

+   `{% url ... pk=...%}`：这与前面的点一样工作，但在某些情况下，`path`可能需要参数（例如要显示的`MailingList`的主键）。我们可以在`url`标签中指定这些额外的参数。

现在我们有一个可以一起使用的视图和模板。

任何视图的最后一步都是将应用的 URLConf 添加到其中。让我们更新`django/mailinglist/urls.py`：

```py
from django.urls import path

from mailinglist import views

app_name = 'mailinglist'

urlpatterns = [
    path('',
         views.MailingListListView.as_view(),
         name='mailinglist_list'),
]
```

考虑到我们之前如何配置了根 URLConf，任何发送到`/mailinglist/`的请求都将被路由到我们的`MailingListListView`。

接下来，让我们添加一个视图来创建新的`MailingList`。

# 创建 CreateMailingListView 和模板

我们将创建一个视图来创建邮件列表。当我们的视图接收到`GET`请求时，视图将向用户显示一个表单，用于输入邮件列表的名称。当我们的视图接收到`POST`请求时，视图将验证表单，要么重新显示带有错误的表单，要么创建邮件列表并将用户重定向到列表的管理页面。

现在让我们在`django/mailinglist/views.py`中创建视图：

```py
class CreateMailingListView(LoginRequiredMixin, CreateView):
    form_class = MailingListForm
    template_name = 'mailinglist/mailinglist_form.html'

    def get_initial(self):
        return {
            'owner': self.request.user.id,
        }
```

`CreateMailingListView`派生自两个类：

+   `LoginRequiredMixin`会重定向未与已登录用户关联的请求，使其无法被处理（我们将在本章后面的*创建用户应用*部分进行配置）

+   `CreateView`知道如何处理`form_class`中指定的表单，并使用`template_name`中列出的模板进行渲染

`CreateView`是在不需要提供几乎任何额外信息的情况下完成大部分工作的类。处理表单，验证它，并保存它总是相同的，而`CreateView`有代码来执行这些操作。如果我们需要更改某些行为，我们可以重写`CreateView`提供的钩子之一，就像我们在`get_initial()`中所做的那样。

当`CreateView`实例化我们的`MailingListForm`时，`CreateView`调用其`get_initial()`方法来获取表单的`initial`数据（如果有的话）。我们使用这个钩子来确保表单的所有者设置为已登录用户的`id`。请记住，`MailingListForm`的`owner`字段已被禁用，因此表单将忽略用户提供的任何数据。

接下来，让我们在`django/mailinglist/templates/mailinglist/mailinglist_form.html`中创建我们的`CreateView`的模板：

```py
{% extends "base.html" %}

{% load crispy_forms_tags %}

{% block title %}
  Create Mailing List
{% endblock %}

{% block body %}
  <h1 >Create Mailing List</h1 >
  <form method="post" class="col-sm-4" >
    {% csrf_token %}
    {{ form | crispy }}
    <button class="btn btn-primary" type="submit" >Submit</button >
  </form >
{% endblock %}
```

我们的模板扩展了`base.html`。当一个模板扩展另一个模板时，它只能在已被扩展模板定义的块中放置 HTML。我们还使用了许多新的 Django 模板标签。让我们仔细看看它们：

+   `{% load crispy_forms_tags %}`：这告诉 Django 加载一个新的模板标签库。在这种情况下，我们将加载我们安装的 Django Crispy Forms 应用的`crispy_from_tags`。这为我们提供了稍后在本节中将看到的`crispy`过滤器。

+   `{% csrf_token %}`：Django 处理的任何表单都必须具有有效的 CSRF 令牌，以防止 CSRF 攻击（参见第三章，*海报、头像和安全*）。`csrf_token`标签返回一个带有正确 CSRF 令牌的隐藏输入标签。请记住，通常情况下，Django 不会处理没有 CSRF 令牌的 POST 请求。

+   `{{ form | crispy }}`：`form`变量是我们的视图正在处理的表单实例的引用，并且通过我们的`CreateView`将其传递到这个模板的上下文中。`crispy`是由`crispy_form_tags`标签库提供的过滤器，将使用 HTML 标签和 Bootstrap 4 中使用的 CSS 类输出表单。

我们现在有一个视图和模板可以一起使用。视图能够使用模板创建用户界面以输入表单中的数据。然后视图能够处理表单的数据并从有效的表单数据创建`MailingList`模型，或者如果数据有问题，则重新显示表单。Django Crispy Forms 库使用 Bootstrap 4 CSS 框架的 HTML 和 CSS 渲染表单。

最后，让我们将我们的视图添加到`mailinglist`应用的 URLConf 中。在`django/mailinglist/urls.py`中，让我们向 URLConf 添加一个新的`path()`对象：

```py
    path('new',
         views.CreateMailingListView.as_view(),
         name='create_mailinglist')
```

考虑到我们之前如何配置了根 URLConf，任何发送到`/mailinglist/new`的请求都将被路由到我们的`CreatingMailingListView`。

接下来，让我们创建一个视图来删除`MailingList`。

# 创建 DeleteMailingListView 视图

用户在`MailingList`不再有用后会想要删除它们。让我们创建一个视图，在`GET`请求上提示用户进行确认，并在`POST`上删除`MailingList`。

我们将把我们的视图添加到`django/mailinglist/views.py`中：

```py
class DeleteMailingListView(LoginRequiredMixin, UserCanUseMailingList,
                            DeleteView):
    model = MailingList
    success_url = reverse_lazy('mailinglist:mailinglist_list')
```

让我们仔细看看`DeleteMailingListView`从中派生的类：

+   `LoginRequiredMixin`：这与前面的代码具有相同的功能，确保未经身份验证的用户的请求不被处理。用户只是被重定向到登录页面。

+   `UserCanUseMailingList`：这是我们在前面的代码中创建的 mixin。`DeleteView`使用`get_object()`方法来检索要删除的模型实例。通过将`UserCanUseMailingList`混合到`DeleteMailingListView`类中，我们保护了每个用户的`MailingList`不被未经授权的用户删除。

+   `DeleteView`：这是一个 Django 视图，它知道如何在`GET`请求上呈现确认模板，并在`POST`上删除相关的模型。

为了使 Django 的`DeleteView`正常工作，我们需要正确配置它。`DeleteView`知道从其`model`属性中删除哪个模型。当我们路由请求到它时，`DeleteView`要求我们提供一个`pk`参数。为了呈现确认模板，`DeleteView`将尝试使用`appname/modelname_confirm_delete.html`。在`DeleteMailingListView`的情况下，模板将是`mailinglist/mailinglist_confirm_delete.html`。如果成功删除模型，那么`DeleteView`将重定向到`success_url`值。我们避免了硬编码`success_url`，而是使用`reverse_lazy()`来引用名称的 URL。`reverse_lazy()`函数返回一个值，直到用它来创建一个`Response`对象时才会解析。

让我们创建`DeleteMailingListView`在`django/mailinglist/templates/mailinglist/mailinglist_confirm_delete.html`中需要的模板：

```py
{% extends "base.html" %}

{% block title %}
  Confirm delete {{ mailinglist.name }}
{% endblock %}

{% block body %}
  <h1 >Confirm Delete?</h1 >
  <form action="" method="post" >
    {% csrf_token %}
    <p >Are you sure you want to delete {{ mailinglist.name }}?</p >
    <input type="submit" value="Yes" class="btn btn-danger btn-sm ">
    <a class="btn btn-primary btn-lg" href="{% url "mailinglist:manage_mailinglist" pk=mailinglist.id %}">No</a>
  </form >
{% endblock %}
```

在这个模板中，我们不使用任何表单，因为没有任何输入需要验证。表单提交本身就是确认。

最后一步将是将我们的视图添加到`django/mailinglist/urls.py`中的`urlpatterns`列表中：

```py
 path('<uuid:pk>/delete',
     views.DeleteMailingListView.as_view(),
     name='delete_mailinglist'),
```

这个`path`看起来不同于我们之前见过的`path()`调用。在这个`path`中，我们包含了一个命名参数，它将被解析出路径并传递给视图。我们使用`<converter:name>`格式来指定`path`命名参数。转换器知道如何匹配路径的一部分（例如，`uuid`转换器知道如何匹配 UUID；`int`知道如何匹配数字；`str`将匹配除了`/`之外的任何非空字符串）。然后匹配的文本将作为关键字参数传递给视图，并提供名称。在我们的情况下，要将请求路由到`DeleteMailingListView`，它必须有这样的路径：`/mailinglist/bce93fec-f9c6-4ea7-b1aa-348d3bed4257/delete`。

现在我们可以列出、创建和删除`MailingList`，让我们创建一个视图来管理其`Subscriber`和`Message`。

# 创建 MailingListDetailView

让我们创建一个视图，列出与`MailingList`相关的所有`Subscriber`和`Message`。我们还需要一个地方来向用户显示`MailingList`的订阅页面链接。Django 可以很容易地创建一个表示模型实例的视图。

让我们在`django/mailinglist/views.py`中创建我们的`MailingListDetailView`：

```py
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import DetailView

from mailinglist.mixins import UserCanUseMailingList
from mailinglist.models import MailingList

class MailingListDetailView(LoginRequiredMixin, UserCanUseMailingList,
                            DetailView):
    model = MailingList
```

我们以与之前相同的方式使用`LoginRequiredMixin`和`UserCanUseMailingList`，并且目的也是相同的。这次，我们将它们与`DetailView`一起使用，这是最简单的视图之一。它只是为其配置的模型实例呈现模板。它通过从`path`接收`pk`参数来检索模型实例，就像`DeleteView`一样。此外，我们不必显式配置它将使用的模板，因为按照惯例，它使用`appname/modelname_detail.html`。在我们的情况下，它将是`mailinglist/mailinglist_detail.html`。

让我们在`django/mailinglist/templates/mailinglist/mailinglist_detail.html`中创建我们的模板：

```py
{% extends "base.html" %}

{% block title %}
  {{ mailinglist.name }} Management
{% endblock %}

{% block body %}
  <h1 >{{ mailinglist.name }} Management
    <a class="btn btn-danger"
       href="{% url "mailinglist:delete_mailinglist" pk=mailinglist.id %}" >
      Delete</a >
  </h1 >

  <div >
    <a href="{% url "mailinglist:create_subscriber" mailinglist_pk=mailinglist.id %}" >Subscription
      Link</a >

  </div >

  <h2 >Messages</h2 >
  <div > Send new
    <a class="btn btn-primary"
       href="{% url "mailinglist:create_message" mailinglist_pk=mailinglist.id %}">
      Send new Message</a >
  </div >
  <ul >
    {% for message in mailinglist.message_set.all %}
      <li >
        <a href="{% url "mailinglist:view_message" pk=message.id %}" >{{ message.subject }}</a >
      </li >
    {% endfor %}
  </ul >

  <h2 >Subscribers</h2 >
  <ul >
    {% for subscriber in mailinglist.subscriber_set.all %}
      <li >
        {{ subscriber.email }}
        {{ subscriber.confirmed|yesno:"confirmed,unconfirmed" }}
        <a href="{% url "mailinglist:unsubscribe" pk=subscriber.id %}" >
          Unsubscribe
        </a >
      </li >
    {% endfor %}
  </ul >
{% endblock %}
```

上述代码模板只介绍了一个新项目（`yesno`过滤器），但确实展示了 Django 模板语言的所有工具是如何结合在一起的。

`yesno`过滤器接受一个值，如果该值评估为`True`，则返回`yes`，如果评估为`False`，则返回`no`，如果为`None`，则返回`maybe`。在我们的情况下，我们传递了一个参数，告诉`yesno`如果为`True`则返回`confirmed`，如果为`False`则返回`unconfirmed`。

`MailingListDetailView`类和模板说明了 Django 如何简洁地完成常见的 Web 开发人员任务：显示数据库中行的页面。

接下来，让我们在`mailinglist`的 URLConf 中为我们的视图创建一个新的`path()`对象：

```py
    path('<uuid:pk>/manage',
         views.MailingListDetailView.as_view(),
         name='manage_mailinglist')
```

接下来，让我们为我们的`Subscriber`模型实例创建视图。

# 创建 Subscriber 视图和模板

在本节中，我们将创建视图和模板，让用户与我们的`Subscriber`模型进行交互。这些视图与`MailingList`和`Message`视图的主要区别之一是，它们不需要任何混合，因为它们将被公开。它们免受篡改的主要保护是`Subscriber`由 UUID 标识，具有大的密钥空间，这意味着篡改是不太可能的。

让我们从`SubscribeToMailingListView`开始。

# 创建 SubscribeToMailingListView 和模板

我们需要一个视图来收集`Subscriber`到`MailingList`。让我们在`django/mailinglist/views.py`中创建一个`SubscribeToMailingListView`类。

```py
class SubscribeToMailingListView(CreateView):
    form_class = SubscriberForm
    template_name = 'mailinglist/subscriber_form.html'

    def get_initial(self):
        return {
            'mailing_list': self.kwargs['mailinglist_id']
        }

    def get_success_url(self):
        return reverse('mailinglist:subscriber_thankyou', kwargs={
            'pk': self.object.mailing_list.id,
        })

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        mailing_list_id = self.kwargs['mailinglist_id']
        ctx['mailing_list'] = get_object_or_404(
            MailingList,
            id=mailing_list_id)
        return ctx
```

我们的`SubscribeToMailingListView`类似于`CreateMailingListView`，但覆盖了一些新方法：

+   `get_success_url()`: 这是由`CreateView`调用的，用于获取重定向用户到已创建模型的 URL。在`CreateMailingListView`中，我们不需要覆盖它，因为默认行为使用模型的`get_absolute_url`。我们使用`reverse()`函数解析路径到感谢页面。

+   `get_context_data()`: 这让我们向模板的上下文中添加新变量。在这种情况下，我们需要访问用户可能订阅的`MailingList`以显示`MailingList`的名称。我们使用 Django 的`get_object_or_404()`快捷函数通过其 ID 检索`MailingList`或引发 404 异常。我们将这个视图的`path`从我们请求的路径中解析出`mailinglist_id`（参见本节末尾的内容）。

接下来，让我们在`mailinglist/templates/mailinglist/subscriber_form.html`中创建我们的模板：

```py
{% extends "base.html" %}
{% load crispy_forms_tags %}
{% block title %}
Subscribe to {{ mailing_list }}
{% endblock %}

{% block body %}
<h1>Subscribe to {{ mailing_list }}</h1>
<form method="post" class="col-sm-6 ">
  {% csrf_token %}
  {{ form | crispy }}
  <button class="btn btn-primary" type="submit">Submit</button>
</form>
{% endblock %}
```

这个模板没有引入任何标签，但展示了另一个例子，说明我们如何使用 Django 的模板语言和 Django Crispy Forms API 快速构建漂亮的 HTML 表单。我们像以前一样扩展`base.html`，以使我们的页面具有一致的外观和感觉。`base.html`还提供了我们要放入内容的块。在任何块之外，我们使用`{% load %}`加载 Django Crispy Forms 标签库，以便我们可以在我们的表单上使用`crispy`过滤器来生成兼容 Bootstrap 4 的 HTML。

接下来，让我们确保 Django 知道如何将请求路由到我们的新视图，通过向`mailinglist`应用的 URLConf 的`urlpatterns`列表添加一个`path()`：

```py
    path('<uuid:mailinglist_id>/subscribe',
         views.SubscribeToMailingListView.as_view(),
         name='subscribe'),
```

在这个`path()`中，我们需要匹配我们作为`mailinglist_pk`传递给视图的`uuid`参数。这是我们的`get_context_data()`方法引用的关键字参数。

接下来，让我们创建一个感谢页面，感谢用户订阅邮件列表。

# 创建感谢订阅视图

用户订阅邮件列表后，我们希望向他们显示一个*感谢*页面。这个页面对于订阅相同邮件列表的所有用户来说是相同的，因为它将显示邮件列表的名称（而不是订阅者的电子邮件）。为了创建这个视图，我们将使用之前看到的`DetailView`，但这次没有额外的混合（这里没有需要保护的信息）。

让我们在`django/mailinglist/views.py`中创建我们的`ThankYouForSubscribingView`：

```py
from django.views.generic import DetailView

from mailinglist.models import  MailingList

class ThankYouForSubscribingView(DetailView):
    model = MailingList
    template_name = 'mailinglist/subscription_thankyou.html'
```

Django 在`DetailView`中为我们完成所有工作，只要我们提供`model`属性。`DetailView`知道如何查找模型，然后为该模型呈现模板。我们还提供了`template_name`属性，因为`mailinglist/mailinglist_detail.html`模板（`DetailView`默认使用的）已经被`MailingListDetailView`使用。

让我们在`django/mailinglist/templates/mailinglist/subscription_thankyou.html`中创建我们的模板：

```py
{% extends "base.html" %}

{% block title %}
  Thank you for subscribing to {{ mailinglist }}
{% endblock %}

{% block body %}
  <div class="col-sm-12" ><h1 >Thank you for subscribing
    to {{ mailinglist }}</h1 >
    <p >Check your email for a confirmation email.</p >
  </div >
{% endblock %}
```

我们的模板只是显示一个感谢和模板名称。

最后，让我们在`mailinglist`应用的 URLConf 的`urlpatterns`列表中添加一个`path()`到`ThankYouForSubscribingView`：

```py
    path('<uuid:pk>/thankyou',
         views.ThankYouForSubscribingView.as_view(),
         name='subscriber_thankyou'),
```

我们的`path`需要匹配 UUID，以便将请求路由到`ThankYouForSubscribingView`。UUID 将作为关键字参数`pk`传递到视图中。这个`pk`将被`DetailView`用来找到正确的`MailingList`。

接下来，我们需要让用户确认他们是否要在这个地址接收电子邮件。

# 创建订阅确认视图

为了防止垃圾邮件发送者滥用我们的服务，我们需要向我们的订阅者发送一封电子邮件，确认他们确实想要订阅我们用户的邮件列表之一。我们将涵盖发送这些电子邮件，但现在我们将创建确认页面。

这个确认页面的行为会有点奇怪。简单地访问页面将会将`Subscriber.confirmed`修改为`True`。这是邮件列表确认页面的标准行为（我们希望避免为我们的订阅者创建额外的工作），但根据 HTTP 规范来说有点奇怪，因为`GET`请求不应该修改资源。

让我们在`django/mailinglist/views.py`中创建我们的`ConfirmSubscriptionView`：

```py
from django.views.generic import DetailView

from mailinglist.models import  Subscriber

class ConfirmSubscriptionView(DetailView):
    model = Subscriber
    template_name = 'mailinglist/confirm_subscription.html'

    def get_object(self, queryset=None):
        subscriber = super().get_object(queryset=queryset)
        subscriber.confirmed = True
        subscriber.save()
        return subscriber
```

`ConfirmSubscriptionView`是另一个`DetailView`，因为它显示单个模型实例。在这种情况下，我们重写`get_object()`方法以在返回之前修改对象。由于`Subscriber`不需要成为我们系统的用户，我们不需要使用`LoginRequiredMixin`。我们的视图受到暴力枚举的保护，因为`Subscriber.id`的密钥空间很大，并且是非顺序分配的。

接下来，让我们在`django/mailinglist/templates/mailinglist/confirm_subscription.html`中创建我们的模板：

```py
{% extends "base.html" %}

{% block title %}
  Subscription to {{ subscriber.mailing_list }} confirmed.
{% endblock %}

{% block body %}
  <h1 >Subscription to {{ subscriber.mailing_list }} confirmed!</h1 >
{% endblock %}
```

我们的模板使用在`base.html`中定义的块，简单地通知用户他们已确认订阅。

最后，让我们在`mailinglist`应用的 URLConf 的`urlpatterns`列表中添加一个`path()`到`ConfirmSubscriptionView`：

```py
    path('subscribe/confirmation/<uuid:pk>',
         views.ConfirmSubscriptionView.as_view(),
         name='confirm_subscription')
```

我们的`confirm_subscription`路径定义了要匹配的路径，以便将请求路由到我们的视图。我们的匹配表达式包括 UUID 的要求，这将作为关键字参数`pk`传递给我们的`ConfirmSubscriptionView`。`ConfirmSubscriptionView`的父类（`DetailView`）将使用它来检索正确的`Subscriber`。

接下来，让我们允许`Subscribers`自行取消订阅。

# 创建 UnsubscribeView

作为道德邮件提供者的一部分，让我们的`Subscriber`取消订阅。接下来，我们将创建一个`UnsubscribeView`，在`Subscriber`确认他们确实想要取消订阅后，将删除`Subscriber`模型实例。

让我们将我们的视图添加到`django/mailinglist/views.py`中：

```py
from django.views.generic import DeleteView

from mailinglist.models import Subscriber

class UnsubscribeView(DeleteView):
    model = Subscriber
    template_name = 'mailinglist/unsubscribe.html'

    def get_success_url(self):
        mailing_list = self.object.mailing_list
        return reverse('mailinglist:subscribe', kwargs={
            'mailinglist_pk': mailing_list.id
        })
```

我们的`UnsubscribeView`让 Django 内置的`DeleteView`实现来呈现模板，并找到并删除正确的`Subscriber`。`DeleteView`要求它接收一个`pk`作为关键字参数，从路径中解析出`Subscriber`的`pk`（就像`DetailView`一样）。当删除成功时，我们将使用`get_success_url()`方法将用户重定向到订阅页面。在执行`get_success_url()`时，我们的`Subscriber`实例已经从数据库中删除，但相应对象的副本将在`self.object`下可用。我们将使用内存中的（但不在数据库中的）实例来获取相关邮件列表的`id`属性。

要呈现确认表单，我们需要在`django/mailinglist/templates/mailinglist/unsubscribe.html`中创建一个模板：

```py
{% extends "base.html" %}

{% block title %}
  Unsubscribe?
{% endblock %}

{% block body %}
  <div class="col">
    <form action="" method="post" >
      {% csrf_token %}
      <p >Are you sure you want to unsubscribe
        from {{ subscriber.mailing_list.name }}?</p >
      <input class="btn btn-danger" type="submit"
             value="Yes, I want to unsubscribe " >
    </form >
  </div >
{% endblock %}
```

这个模板呈现了一个`POST`表单，它将作为`subscriber`希望取消订阅的确认。

接下来，让我们向`mailinglist`应用的 URLConf 的`urlpatterns`列表中添加一个`path()`到`UnsubscribeView`：

```py
     path('unsubscribe/<uuid:pk>',
         views.UnsubscribeView.as_view(),
         name='unsubscribe'),
```

在处理从`DetailView`或`DeleteView`派生的视图时，要记住将路径匹配器命名为`pk`是至关重要的。

现在，让我们允许用户开始创建他们将发送给他们的`Subscriber`的`Message`。

# 创建消息视图

我们在`Message`模型中跟踪我们的用户想要发送给他们的`Subscriber`的电子邮件。为了确保我们有一个准确的日志记录用户发送给他们的`Subscribers`的内容，我们将限制`Message`上可用的操作。我们的用户只能创建和查看`Message`。支持编辑是没有意义的，因为已发送的电子邮件无法修改。我们也不会支持删除消息，这样我们和用户都有一个准确的日志记录请求发送的内容。

让我们从创建`CreateMessageView`开始！

# 创建 CreateMessageView

我们的`CreateMessageView`将遵循类似于我们为 Answerly 创建的 markdown 表单的模式。用户将获得一个表单，他们可以提交以保存或预览。如果提交是预览，那么表单将与`Message`的渲染 markdown 预览一起呈现。如果用户选择保存，那么他们将创建他们的新消息。

由于我们正在创建一个新的模型实例，我们将使用 Django 的`CreateView`。

让我们在`django/mailinglist/views.py`中创建我们的视图：

```py
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import CreateView

from mailinglist.models import Message

class CreateMessageView(LoginRequiredMixin, CreateView):
    SAVE_ACTION = 'save'
    PREVIEW_ACTION = 'preview'

    form_class = MessageForm
    template_name = 'mailinglist/message_form.html'

    def get_success_url(self):
        return reverse('mailinglist:manage_mailinglist',
                       kwargs={'pk': self.object.mailing_list.id})

    def get_initial(self):
        mailing_list = self.get_mailing_list()
        return {
            'mailing_list': mailing_list.id,
        }

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        mailing_list = self.get_mailing_list()
        ctx.update({
            'mailing_list': mailing_list,
            'SAVE_ACTION': self.SAVE_ACTION,
            'PREVIEW_ACTION': self.PREVIEW_ACTION,
        })
        return ctx

    def form_valid(self, form):
        action = self.request.POST.get('action')
        if action == self.PREVIEW_ACTION:
            context = self.get_context_data(
                form=form,
                message=form.instance)
            return self.render_to_response(context=context)
        elif action == self.SAVE_ACTION:
            return super().form_valid(form)

    def get_mailing_list(self):
        mailing_list = get_object_or_404(MailingList,
                                         id=self.kwargs['mailinglist_pk'])
        if not mailing_list.user_can_use_mailing_list(self.request.user):
            raise PermissionDenied()
        return mailing_list
```

我们的视图继承自`CreateView`和`LoginRequiredMixin`。我们使用`LoginRequiredMixin`来防止未经身份验证的用户向邮件列表发送消息。为了防止已登录但未经授权的用户发送消息，我们将创建一个中心的`get_mailing_list()`方法，该方法检查已登录用户是否可以使用此邮件列表。`get_mailing_list()`期望`mailinglist_pk`将作为关键字参数提供给视图。

让我们仔细看看`CreateMessageView`，看看这些是如何一起工作的：

+   `form_class = MessageForm`：这是我们希望`CreateView`渲染、验证和用于创建我们的`Message`模型的表单。

+   `template_name = 'mailinglist/message_form.html'`：这是我们接下来要创建的模板。

+   `def get_success_url()`: 在成功创建`Message`后，我们将重定向用户到`MailingList`的管理页面。

+   `def get_initial():`：我们的`MessageForm`将其`mailing_list`字段禁用，以防用户试图偷偷地为另一个用户的`MailingList`创建`Message`。相反，我们使用我们的`get_mailing_list()`方法来根据`mailinglist_pk`参数获取邮件列表。使用`get_mailing_list()`，我们检查已登录用户是否可以使用`MailingList`。

+   `def get_context_data()`: 这提供了额外的变量给模板的上下文。我们提供了`MailingList`以及保存和预览的常量。

+   `def form_valid()`: 这定义了表单有效时的行为。我们重写了`CreateView`的默认行为来检查`action` POST 参数。`action`将告诉我们是要渲染`Message`的预览还是让`CreateView`保存一个新的`Message`模型实例。如果我们正在预览消息，那么我们将通过我们的表单构建一个未保存的`Message`实例传递给模板的上下文。

接下来，让我们在`django/mailinglist/templates/mailinglist/message_form.html`中制作我们的模板：

```py
{% extends "base.html" %}
{% load crispy_forms_tags %}
{% load markdownify %}
{% block title %}
  Send a message to {{ mailing_list }}
{% endblock %}

{% block body %}
  <h1 >Send a message to {{ mailing_list.name }}</h1 >
  {% if message %}
    <div class="card" >
      <div class="card-header" >
        Message Preview
      </div >
      <div class="card-body" >
        <h5 class="card-title" >{{ message.subject }}</h5 >
        <div>{{ message.body|markdownify }}</div>
      </div >
    </div >
  {% endif %}
  <form method="post" class="col-sm-12 col-md-9" >
    {% csrf_token %}
    {{ form | crispy }}
    <button type="submit" name="action"
            value="{{ SAVE_ACTION }}"
            class="btn btn-primary" >Save
    </button >
    <button type="submit" name="action"
            value="{{ PREVIEW_ACTION }}"
            class="btn btn-primary" >Preview
    </button >
  </form >
{% endblock %}
```

这个模板加载了第三方的 Django Markdownify 标签库和 Django Crispy Forms 标签库。前者给我们提供了`markdownify`过滤器，后者给我们提供了`crispy`过滤器。`markdownify`过滤器将接收到的 markdown 文本转换为 HTML。我们之前在我们的 Answerly 项目的第二部分中使用了 Django Markdownify。

这个模板表单有两个提交按钮，一个用于保存表单，一个用于预览表单。只有在我们传入`message`来预览时，预览块才会被渲染。

现在我们有了视图和模板，让我们在`mailinglist`应用的 URLConf 中为`CreateMessageView`添加一个`path()`：

```py
     path('<uuid:mailinglist_ipk>/message/new',
         views.CreateMessageView.as_view(),
         name='create_message'),
```

现在我们可以创建消息了，让我们创建一个查看我们已经创建的消息的视图。

# 创建消息 DetailView

为了让用户查看他们发送给他们的`Subscriber`的`Message`，我们需要一个`MessageDetailView`。这个视图将简单地显示一个`Message`，但应该只允许已登录并且可以使用`Message`的`MailingList`的用户访问该视图。

让我们在`django/mailinglist/views.py`中创建我们的视图：

```py
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import DetailView

from mailinglist.mixins import UserCanUseMailingList
from mailinglist.models import Message

class MessageDetailView(LoginRequiredMixin, UserCanUseMailingList,
                        DetailView):
    model = Message
```

顾名思义，我们将使用 Django 的`DetailView`。为了提供我们需要的保护，我们将添加 Django 的`LoginRequiredMixin`和我们的`UserCanUseMailingList`混合。正如我们以前看到的那样，我们不需要指定模板的名称，因为`DetailView`将根据应用和模型的名称假定它。在我们的情况下，`DetailView`希望模板被称为`mailinglist/message_detail.html`。

让我们在`mailinglist/message_detail.html`中创建我们的模板：

```py
{% extends "base.html" %}
{% load markdownify %}

{% block title %}
  {{ message.subject }}
{% endblock %}

{% block body %}
  <h1 >{{ message.subject }}</h1 >
  <div>
    {{ message.body|markdownify }}
  </div>
{% endblock %}
```

我们的模板扩展了`base.html`并在`body`块中显示消息。在显示`Message.body`时，我们使用第三方 Django Markdownify 标签库的`markdownify`过滤器将任何 markdown 文本呈现为 HTML。

最后，我们需要向`mailinglist`应用的 URLConf 的`urlpatterns`列表中添加一个`path()`到`MessageDetailView`：

```py
    path('message/<uuid:pk>', 
         views.MessageDetailView.as_view(), 
         name='view_message')
```

我们现在已经完成了我们的`mailinglist`应用的模型、视图和模板。我们甚至创建了一个`UserCanUseMailingList`来让我们的视图轻松地阻止未经授权的用户访问`MailingList`或其相关视图。

接下来，我们将创建一个`user`应用来封装用户注册和身份验证。

# 创建用户应用

要在 Mail Ape 中创建一个`MailingList`，用户需要拥有一个帐户并已登录。在本节中，我们将编写我们的`user` Django 应用的代码，它将封装与用户有关的一切。请记住，Django 应用应该范围严密。我们不希望将这种行为放在我们的`mailinglist`应用中，因为这是两个不同的关注点。

我们的`user`应用将与 MyMDB（第一部分）和 Answerly（第二部分）中看到的`user`应用非常相似。由于这种相似性，我们将略过一些主题。要深入研究该主题，请参阅第二章，*将用户添加到 MyMDb*。

Django 通过其内置的`auth`应用（`django.contrib.auth`）使用户和身份验证管理变得更加容易。`auth`应用提供了默认的用户模型、用于创建新用户的`Form`，以及登录和注销视图。这意味着我们的`user`应用只需要填写一些空白，就可以在本地完全实现用户管理。

让我们首先在`django/user/urls.py`中为我们的`user`应用创建一个 URLConf：

```py
from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path

import user.views

app_name = 'user'

urlpatterns = [
    path('login', LoginView.as_view(), name='login'),
    path('logout', LogoutView.as_view(), name='logout'),
    path('register', user.views.RegisterView.as_view(), name='register'),
]
```

我们的 URLConf 由三个视图组成：

+   `LoginView.as_view()`: 这是`auth`应用的登录视图。`auth`应用提供了一个接受凭据的视图，但没有模板。我们需要创建一个名为`registration/login.html`的模板。默认情况下，它会在登录时将用户重定向到`settings.LOGIN_REDIRECT_URL`。我们还可以传递一个`next`的`GET`参数来取代该设置。

+   `LogoutView.as_view()`: 这是`auth`应用的注销视图。`LogoutView`是少数在`GET`请求上修改状态的视图之一，它会注销用户。该视图返回一个重定向响应。我们可以使用`settings.LOGOUT_REDIRECT_URL`来配置用户在注销时将被重定向到的位置。同样，我们可以使用`GET`参数`next`来自定义此行为。

+   `user.views.RegisterView.as_view()`: 这是我们将编写的用户注册视图。Django 为我们提供了`UserCreationForm`，但没有视图。

我们还需要添加一些设置，让 Django 正确使用我们的`user`视图。让我们在`django/config/settings.py`中更新一些新设置：

```py
LOGIN_URL = 'user:login'
LOGIN_REDIRECT_URL = 'mailinglist:mailinglist_list'
LOGOUT_REDIRECT_URL = 'user:login'
```

这三个设置告诉 Django 如何在不同的身份验证场景下重定向用户：

+   `LOGIN_URL`：当未经身份验证的用户尝试访问需要身份验证的页面时，`LoginRequiredMixin`使用此设置。

+   `LOGIN_REDIRECT_URL`：当用户登录时，我们应该将他们重定向到哪里？通常，我们将他们重定向到一个个人资料页面；在我们的情况下，是显示`MailingList`列表的页面。

+   `LOGOUT_REDIRECT_URL`：当用户注销时，我们应该将他们重定向到哪里？在我们的情况下，是登录页面。

我们现在还有两项任务：

+   创建登录模板

+   创建用户注册视图和模板

让我们从制作登录模板开始。

# 创建登录模板

让我们在`django/user/templates/registration/login.html`中制作我们的登录模板：

```py
{% extends "base.html" %}
{% load crispy_forms_tags %}

{% block title %} Login - {{ block.super }} {% endblock %}

{% block body %}
  <h1>Login</h1>
  <form method="post" class="col-sm-6">
    {% csrf_token %}
    {{ form|crispy }}
    <button type="submit" id="log_in" class="btn btn-primary">Log in</button>
  </form>
{% endblock %}
```

这个表单遵循了我们之前表单的所有做法。我们使用`csrf_token`来防止 CSRF 攻击。我们使用`crsipy`过滤器使用 Bootstrap 4 样式标签和类打印表单。

记住，我们不需要创建一个视图来处理我们的登录请求，因为我们正在使用`django.contrib.auth`中提供的视图。

接下来，让我们创建一个视图和模板来注册新用户。

# 创建用户注册视图

Django 没有为创建新用户提供视图，但它提供了一个用于捕获新用户注册的表单。我们可以将`UserCreationForm`与`CreateView`结合使用，快速创建一个`RegisterView`。

让我们在`django/user/views.py`中添加我们的视图：

```py
from django.contrib.auth.forms import UserCreationForm
from django.views.generic.edit import CreateView

class RegisterView(CreateView):
    template_name = 'user/register.html'
    form_class = UserCreationForm
```

这是一个非常简单的`CreateView`，就像我们在本章中已经看到的几次一样。

让我们在`django/user/templates/user/register.html`中创建我们的模板：

```py
{% extends "base.html" %}
{% load crispy_forms_tags %}
{% block body %}
  <div class="col-sm-12">
    <h1 >Register for Mail Ape</h1 >
    <form method="post" >
      {% csrf_token %}
      {{ form | crispy }}
      <button type="submit" class="btn btn-primary" >
        Register
      </button >
    </form >
  </div >
{% endblock %}
```

同样，该模板遵循了我们之前`CreateView`模板的相同模式。

现在，我们准备在本地运行 Mail Ape。

# 在本地运行 Mail Ape

Django 自带开发服务器。这个服务器不适合生产（甚至是暂存）部署，但适合本地开发。

让我们使用我们 Django 项目的`manage.py`脚本启动服务器：

```py
$ cd django
$ python manage.py runserver
Performing system checks...

System check identified no issues (0 silenced).
January 29, 2018 - 23:35:15
Django version 2.0.1, using settings 'config.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```

我们现在可以在`http://127.0.0.1:8000`上访问我们的服务器。

# 总结

在本章中，我们启动了 Mail Ape 项目。我们创建了 Django 项目并启动了两个 Django 应用程序。`mailinglist`应用程序包含了我们的邮件列表代码的模型、视图和模板。`user`应用程序包含了与用户相关的视图和模板。`user`应用程序要简单得多，因为它利用了 Django 的`django.contrib.auth`应用程序。

接下来，我们将构建一个 API，以便用户可以轻松地与 Mail Ape 集成。
