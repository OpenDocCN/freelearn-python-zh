# 在 Flask 中使用模型、SQLAlchemy 和超链接 API 进行工作

在本章中，我们将扩展上一章开始构建的 RESTful API 的功能。我们将使用 SQLAlchemy 作为我们的 ORM 来与 PostgreSQL 数据库交互，并且我们将利用 Flask 和 Flask-RESTful 中包含的先进功能，这将使我们能够轻松组织代码以处理复杂的 API，如模型和蓝图。

在本章中，我们将讨论以下主题：

+   设计一个与 PostgreSQL 10.5 数据库交互的 RESTful API

+   理解每个 HTTP 方法执行的任务

+   使用 `requirements.txt` 文件安装包以简化我们的常见任务

+   创建数据库

+   配置数据库

+   为模型编写代码，包括它们的 ... 

# 设计一个与 PostgreSQL 10.5 数据库交互的 RESTful API

到目前为止，我们的 RESTful API 已经在充当数据存储库的简单内存字典上执行了 CRUD 操作。该字典永远不会持久化，因此，每次我们重新启动 Flask 开发服务器时，数据都会丢失。

现在，我们想要使用 Flask RESTful 创建一个更复杂的 RESTful API，以便与一个数据库模型交互，该模型允许我们处理分组到通知类别中的通知。在我们的上一个 RESTful API 中，我们使用一个字符串属性来指定通知的通知类别。在这种情况下，我们希望能够轻松检索属于特定通知类别的所有通知，因此，我们将有一个通知与通知类别之间的关系。

我们必须能够对不同的相关资源和资源集合执行 CRUD 操作。以下表格列出了我们将创建以表示模型的资源和类名：

| 资源 | 代表模型的类名 |
| --- | --- |
| 通知类别 | `NotificationCategory` |
| 通知 | `Notification` |

通知类别（`NotificationCategory`）只需要以下数据：

+   一个整数标识符

+   一个字符串名称

我们需要一个通知（`Notification`）的以下数据：

+   一个整数标识符

+   一个指向通知类别（`NotificationCategory`）的外键

+   一个字符串消息

+   一个 **TTL**（即 **Time to Live**，表示 **生存时间**），即指示通知消息在 OLED 显示上显示的秒数

+   一个创建日期和时间。时间戳将在将新通知添加到集合时自动添加

+   一个整数计数器，表示通知消息在 OLED 显示上显示的次数

+   一个布尔值，表示通知消息是否至少在 OLED 显示上显示过一次

我们将利用许多与 Flask RESTful 和 SQLAlchemy 相关的包，这些包使得序列化和反序列化数据、执行验证以及将 SQLAlchemy 与 Flask 和 Flask RESTful 集成变得更加容易。这样，我们将减少样板代码。

# 理解每个 HTTP 方法执行的任务

以下表格显示了我们的新 API 必须支持的方法的 HTTP 动词、作用域和语义。每个方法由一个 HTTP 动词和一个作用域组成，并且所有方法对所有资源和集合都有明确的含义：

| HTTP 动词 | 作用域 | 语义 |
| --- | --- | --- |
| `GET` | 通知类别集合 | 获取集合中存储的所有通知类别，按名称升序排序。每个通知类别必须包含资源的完整 URL。此外，每个通知类别必须包含一个列表，其中包含属于该类别的所有通知的详细信息。通知不必包含 ... |

# 使用 requirements.txt 文件安装包以简化我们的常见任务

确保您已退出 Flask 的开发服务器。您只需在运行它的终端或命令提示符窗口中按 *Ctrl* + *C* 即可。

现在，我们将安装一些额外的包。请确保您已激活我们在上一章中创建并命名为 `Flask01` 的虚拟环境。激活虚拟环境后，就是运行大量命令的时候了，这些命令对 macOS、Linux 或 Windows 都是一样的。

现在，我们将编辑现有的 `requirements.txt` 文件，以指定我们的应用程序在任何支持平台上需要安装的额外包集。这样，在任意新的虚拟环境中重复安装指定包及其版本将变得极其容易。

使用您喜欢的编辑器编辑虚拟环境根目录下名为 `requirements.txt` 的现有文本文件。在最后一行之后添加以下行，以声明 API 新版本所需的额外包及其版本。示例代码文件包含在 `restful_python_2_02_01` 文件夹中，位于 `Flask01/requirements.txt` 文件中：

```py
Flask-SQLAlchemy==2.3.2 
Flask-Migrate==2.2.1 
marshmallow==2.16.0 
marshmallow-sqlalchemy==0.14.1 
flask-marshmallow==0.9.0 
psycopg2==2.7.5
```

在 `requirements.txt` 文件中添加的每一行都表示需要安装的包及其版本。以下表格总结了我们作为对先前包含的包的额外要求指定的包及其版本号：

| 包名 | 要安装的版本 |
| --- | --- |
| `Flask-SQLAlchemy` | 2.3.2 |
| `Flask-Migrate` | 2.2.1 |
| `marshmallow` | 2.16.0 |
| `marshmallow-sqlalchemy` | 0.14.1 |
| `flask-marshmallow` | 0.9.0 |
| `psycopg2` | 2.7.5 |

`Flask-SQLAlchemy` 为 Flask 应用程序添加了对 SQLAlchemy ORM 的支持。这个扩展简化了在 Flask 应用程序中执行常见的 SQLAlchemy 任务。SQLAlchemy 是 `Flask-SQLAlchemy` 的依赖项。

`Flask-Migrate` 使用 Alembic 包来处理 Flask 应用程序的 SQLAlchemy 数据库迁移。我们将使用 `Flask-Migrate` 来设置我们的 PostgreSQL 数据库。

如果您之前使用过 `Flask-Migrate` 的先前版本，请注意，Flask-Script 已不再是 `Flask-Migrate` 的依赖项。Flask-Script 是一个流行的包，它为 Flask 添加了编写外部脚本的支持，包括设置数据库的脚本。最新的 Flask 版本在虚拟环境中安装了 `flask` 脚本和基于 Click 包的命令行界面。因此，不再需要将 `Flask-Migrate` 与 Flask-Script 结合使用。

Marshmallow 是一个轻量级库，用于将复杂的数据类型转换为原生 Python 数据类型，反之亦然。Marshmallow 提供了模式，我们可以使用它们来验证输入数据，将输入数据反序列化为应用级别的对象，以及将应用级别的对象序列化为 Python 原始类型。

`marshmallow-sqlalchemy` 提供了与之前安装的 `marshmallow` 验证、序列化和反序列化轻量级库的 SQLAlchemy 集成。

Flask-Marshmallow 将之前安装的 `marshmallow` 库与 Flask 应用程序集成，使得生成 URL 和超链接字段变得简单易行。

Psycopg 2 (`psycopg2`) 是一个 Python-PostgreSQL 数据库适配器，SQLAlchemy 将使用它来与我们的最近创建的 PostgreSQL 数据库交互。同样，在运行此包的安装之前，确保 PostgreSQL 的 `bin` 文件夹包含在 `PATH` 环境变量中是非常重要的。

现在，我们必须在 macOS、Linux 或 Windows 上运行以下命令来安装先前表格中解释的附加包和版本，使用 `pip` 通过最近编辑的 `requirements` 文件。在运行命令之前，请确保您位于包含 `requirements.txt` 文件的文件夹中：

```py
    pip install -r requirements.txt
```

输出的最后几行将指示所有新安装的包及其依赖项已成功安装。如果您下载了示例的源代码，并且您没有使用 API 的先前版本，`pip` 还将安装 `requirements.txt` 文件中包含的其他包：

```py
Installing collected packages: SQLAlchemy, Flask-SQLAlchemy, Mako, python-editor, python-dateutil, alembic, Flask-Migrate, marshmallow, marshmallow-sqlalchemy, flask-marshmallow, psycopg2
      Running setup.py install for SQLAlchemy ... done
      Running setup.py install for Mako ... done
      Running setup.py install for python-editor ... done
Successfully installed Flask-Migrate-2.2.1 Flask-SQLAlchemy-2.3.2
Mako-1.0.7 SQLAlchemy-1.2.12 alembic-1.0.0 flask-marshmallow-0.9.0 marshmallow-2.16.0 marshmallow-sqlalchemy-0.14.1 psycopg2-2.7.5 
python-dateutil-2.7.3 python-editor-1.0.3

```

# 创建数据库

现在，我们将创建一个 PostgreSQL 10.5 数据库，我们将使用它作为我们的 API 的存储库。如果您还没有在您的计算机或开发服务器上运行 PostgreSQL 数据库服务器，您将需要下载并安装它。您可以从其网页([`www.postgresql.org`](http://www.postgresql.org))下载并安装这个数据库管理系统。如果您使用的是 macOS，`Postgres.app` 提供了一种非常简单的方法来安装和使用 PostgreSQL。您可以从 [`postgresapp.com`](http://postgresapp.com) 参考它。如果您使用的是 Windows，EnterpriseDB 和 BigSQL 提供了图形安装程序，这些安装程序简化了在现代 Windows 服务器或桌面版本上的配置过程（访问 [`www.postgresql.org/download/windows`](https://www.postgresql.org/download/windows)）。

# 配置数据库

如果你使用的是我们为之前示例创建的相同虚拟环境，或者你下载了代码示例，那么`service`文件夹已经存在。如果你创建了一个新的虚拟环境，请在虚拟环境根文件夹内创建一个名为`service`的文件夹。

在`service`文件夹内创建一个新的`config.py`文件。以下行展示了声明用于确定 Flask 和 SQLAlchemy 配置的变量的代码。`SQL_ALCHEMY_DATABASE_URI`变量生成用于 PostgreSQL 数据库的 SQLAlchemy URI。确保你在`DB_NAME`的值中指定所需的数据库名称，并根据你的 PostgreSQL 配置配置用户、密码、主机和端口。如果你遵循了之前的步骤，请使用这些步骤中指定的设置。示例代码文件包含在`restful_python_2_02_01`文件夹中，位于`Flask01/service/config.py`文件中：

```py
import os 

basedir = os.path.abspath(os.path.dirname(__file__)) 
SQLALCHEMY_ECHO = False 
SQLALCHEMY_TRACK_MODIFICATIONS = True 
# Replace your_user_name with the user name you configured for the database 
# Replace your_password with the password you specified for the database user 
SQLALCHEMY_DATABASE_URI = "postgresql://{DB_USER}:{DB_PASS}@{DB_ADDR}/{DB_NAME}".format(DB_USER="your_user_name", DB_PASS="your_password", DB_ADDR="127.0.0.1", DB_NAME="flask_notifications") 
SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')
```

我们将指定之前创建的模块（`config`）作为创建 Flask 应用的函数的参数。这样，我们有一个模块指定了与 SQLAlchemy 相关的所有不同配置变量的值，另一个模块创建 Flask 应用。我们将创建 Flask 应用工厂作为我们迈向新 API 的最终步骤。

# 创建具有其关系的模型

现在，我们将创建我们将用于在 PostgreSQL 数据库中表示和持久化通知类别、通知及其关系的模型。

打开`service/models.py`文件，并用以下代码替换其内容。代码中声明与其它模型相关字段的行被突出显示。如果你创建了一个新的虚拟环境，请在`service`文件夹内创建一个新的`models.py`文件。示例代码文件包含在`restful_python_2_02_01`文件夹中，位于`Flask01/service/models.py`文件中：

```py
from marshmallow import Schema, fields, pre_load from marshmallow import validate from flask_sqlalchemy import SQLAlchemy from flask_marshmallow import ...
```

# 创建用于验证、序列化和反序列化模型的模式

现在，我们将创建我们将用于验证、序列化和反序列化之前声明的`NotificationCategory`和`Notification`模型及其关系的 Flask-Marshmallow 模式。

打开`service`文件夹内的`models.py`文件，并在最后一行之后添加以下代码。代码中声明与其它模式相关字段的行被突出显示。示例代码文件包含在`restful_python_2_02_01`文件夹中，位于`Flask01/service/models.py`文件中：

```py
class NotificationCategorySchema(ma.Schema): 
    id = fields.Integer(dump_only=True) 
    # Minimum length = 3 characters 
    name = fields.String(required=True,  
        validate=validate.Length(3)) 
    url = ma.URLFor('service.notificationcategoryresource',  
        id='<id>',  
        _external=True) 
 notifications = fields.Nested('NotificationSchema',             
      many=True,         
      exclude=('notification_category',)) 

class NotificationSchema(ma.Schema): 
    id = fields.Integer(dump_only=True) 
    # Minimum length = 5 characters 
    message = fields.String(required=True,  
        validate=validate.Length(5)) 
    ttl = fields.Integer() 
    creation_date = fields.DateTime() 
 notification_category =
fields.Nested(NotificationCategorySchema,
         only=['id', 'url', 'name'],
         required=True) 
    displayed_times = fields.Integer() 
    displayed_once = fields.Boolean() 
    url = ma.URLFor('service.notificationresource',  
        id='<id>',  
        _external=True) 

    @pre_load 
    def process_notification_category(self, data): 
        notification_category = data.get('notification_category') 
        if notification_category: 
            if isinstance(notification_category, dict): 
                notification_category_name = notification_category.get('name') 
            else: 
                notification_category_name = notification_category 
            notification_category_dict = dict(name=notification_category_name) 
        else: 
            notification_category_dict = {} 
        data['notification_category'] = notification_category_dict 
        return data 
```

代码声明了以下两个模式，即`ma.Schema`类的两个子类：

+   `NotificationCategorySchema`

+   `NotificationSchema`

我们不使用 Flask-Marshmallow 允许我们根据模型中声明的字段自动确定每个属性适当类型的特性，因为我们想为每个字段使用特定的选项。

我们将表示字段的属性声明为`marshmallow.fields`模块中声明的适当类的实例。每当我们将`dump_only`参数指定为`True`时，这意味着我们希望该字段为只读。例如，我们无法在任何模式中为`id`字段提供值。该字段的值将由 PostgreSQL 数据库中的自增主键自动生成。

`NotificationCategorySchema`类将`name`属性声明为`fields.String`类的一个实例。`required`参数设置为`True`，以指定该字段不能为空字符串。`validate`参数设置为`validate.Length(3)`，以指定该字段必须至少有三个字符的长度。

类使用以下行声明了`url`字段：

```py
url = ma.URLFor('service.notificacion_categoryresource', 
    id='<id>', 
    _external=True)
```

`url`属性是`ma.URLFor`类的一个实例，并且这个字段将输出资源的完整 URL，即通知类别的 URL。第一个参数是 Flask 端点的名称：`'service.notificationcategoryresource'`。我们将在稍后创建`NotificationCategoryResource`类，`URLFor`类将使用它来生成 URL。`id`参数指定`'<id>'`，因为我们希望从要序列化的对象中提取`id`。小于（`<`）和大于（`>`）符号内的`id`字符串指定我们希望从必须序列化的对象中提取字段。`_external`属性设置为`True`，因为我们希望生成资源的完整 URL。这样，每次序列化`NotificationCategory`时，它都会在`url`键或属性中包含资源的完整 URL。

在这种情况下，我们正在使用不安全的 API 在 HTTP 后面。如果我们的 API 配置为 HTTPS，那么在创建`ma.URLFor`实例时，我们应该将`_scheme`参数设置为`'https'`。

类使用以下行声明了`notifications`字段：

```py
notifications = fields.Nested('NotificationSchema', 
    many=True, 
    exclude=('notification_category',)) 
```

`notifications`属性是`marshmallow.fields.Nested`类的一个实例，并且这个字段将嵌套一个`Schema`集合，因此，我们为`many`参数指定`True`。第一个参数指定嵌套`Schema`类的名称为一个字符串。我们在定义了`NotificationCategorySchema`类之后声明`NotificationSchema`类。因此，我们指定`Schema`类名为一个字符串，而不是使用我们尚未定义的类型。

事实上，我们将得到两个相互嵌套的对象；也就是说，我们将在通知类别和通知之间创建双向嵌套。我们使用一个字符串元组作为`exclude`参数，以指示我们希望`notification_category`字段从为每个通知序列化的字段中排除。这样，我们避免了无限递归，因为包含`notification_category`字段将序列化与该类别相关的所有通知。

当我们声明`Notification`模型时，我们使用了`orm.relationship`函数来提供对`NotificationCategory`模型的多对一关系。`backref`参数指定了一个调用`orm.backref`函数的调用，其中`'notifications'`作为第一个值，表示从相关的`NotificationCategory`对象返回到`Notification`对象的关系名称。通过之前解释的行，我们创建了使用我们为`db.backref`函数指定的相同名称的`notifications`字段。

`NotificationSchema`类将`notification`属性声明为`fields.String`类的一个实例。`required`参数设置为`True`，以指定该字段不能为空字符串。`validate`参数设置为`validate.Length(5)`，以指定该字段必须至少有五个字符长。该类使用与我们在`Message`模型中使用的类型相对应的类声明了`ttl`、`creation_date`、`displayed_times`和`displayed_once`字段。

类使用以下行声明了`notification_category`字段：

```py
notification_category = fields.Nested(CategorySchema,  
    only=['id', 'url', 'name'],  
    required=True) 
```

`notification_category`属性是`marshmallow.fields.Nested`类的一个实例，并且这个字段将嵌套一个`NotificationCategorySchema`。我们为`required`参数指定`True`，因为通知必须属于一个类别。第一个参数指定了嵌套`Schema`类的名称。我们已声明了`NotificationCategorySchema`类，因此我们将`NotificationCategorySchema`指定为第一个参数的值。我们使用带有字符串列表的`only`参数来指示在序列化嵌套的`NotificationCategorySchema`时要包含的字段名称。我们希望包含`id`、`url`和`name`字段。我们没有指定`notifications`字段，因为我们不希望通知类别序列化属于它的通知列表。

类使用以下行声明了`url`字段：

```py
url = ma.URLFor('service.notificationresource',  
    id='<id>',  
    _external=True)
```

`url`属性是`ma.URLFor`类的一个实例，并且这个字段将输出资源的完整 URL，即通知资源的 URL。第一个参数是 Flask 端点名称：`'service.notificationresource'`。我们稍后会创建`NotificationResource`类，`URLFor`类将使用它来生成 URL。`id`参数指定为`'<id>'`，因为我们希望从要序列化的对象中提取`id`。`_external`属性设置为`True`，因为我们希望为资源生成完整的 URL。这样，每次我们序列化一个`Notification`时，它都会在`url`键中包含资源的完整 URL。

`NotificationSchema` 类声明了一个 `process_notification_category` 方法，该方法使用 `@pre_load` 装饰器，具体来说，是 `marshmallow.pre_load`。这个装饰器注册了一个在反序列化对象之前调用的方法。这样，在 Marshmallow 反序列化通知之前，`process_category` 方法将被执行。

该方法接收 `data` 参数中的要反序列化的数据，并返回处理后的数据。当我们收到一个请求以 `POST` 新通知时，通知类别名称可以指定为名为 `'notification_category'` 的键。如果存在具有指定名称的类别，我们将使用现有的类别作为与新的通知相关联的类别。如果不存在具有指定名称的类别，我们将创建一个新的通知类别，然后我们将使用这个新类别作为与新的通知相关联的类别。这样，我们使用户创建与类别相关的新通知变得简单直接。

`data` 参数可能包含一个指定为 `'notification_category'` 键的字符串形式的通知类别名称。然而，在其他情况下，`'notification_category'` 键将包含具有字段名称和字段值的键值对，这些值对应于现有的通知类别。

`process_notification_category` 方法中的代码检查 `'notification_category'` 键的值，并返回一个包含适当数据的字典，以确保我们能够使用适当的键值对反序列化通知类别，无论传入数据之间的差异如何。最后，该方法返回处理后的字典。当我们在开始组合和发送对新 API 的 HTTP 请求时，我们将深入了解 `process_notification_category` 方法所做的工作。

# 将蓝图与资源路由相结合

现在，我们将创建组成我们 RESTful API 主要构建块的资源。首先，我们将创建一些将在不同资源中使用的实例。在 `services` 文件夹内创建一个新的 `views.py` 文件，并添加以下行。注意，代码导入了在上一章中创建的 `http_status.py` 模块中声明的 `HttpStatus` 枚举。示例代码文件包含在 `restful_python_2_02_01` 文件夹中，位于 `Flask01/service/views.py` 文件：

```py
from flask import Blueprint, request, jsonify, make_response from flask_restful import Api, Resource from http_status import HttpStatus from models import orm, NotificationCategory, NotificationCategorySchema, ...
```

# 理解和配置资源路由

下表显示了我们要为每个 HTTP 动词和范围组合执行的先前创建的类的操作方法：

| HTTP 动词 | 范围 | 类和方法 |
| --- | --- | --- |
| `GET` | 通知集合 | `NotificationListResource.get` |
| `GET` | 通知 | `NotificationResource.get` |
| `POST` | 通知集合 | `NotificationListResource.post` |
| `PATCH` | 通知 | `NotificationResource.patch` |
| `DELETE` | 通知 | `NotificationResource.delete` |
| `GET` | 通知类别集合 | `NotificationCategoryListResource.get` |
| `GET` | 通知类别 | `NotificationCategoryResource.get` |
| `POST` | 通知类别集合 | `NotificationCategoryListResource.post` |
| `PATCH` | 通知类别 | `NotificationCategoryResource.patch` |
| `DELETE` | 通知类别 | `NotificationCategoryResource.delete` |

如果请求导致调用一个不支持 HTTP 方法的资源，Flask-RESTful 将返回一个带有 HTTP `405 Method Not Allowed` 状态码的响应。

我们必须通过定义 URL 规则来进行必要的资源路由配置，以调用适当的方法，并通过传递所有必要的参数。以下行配置了服务的资源路由。在 `service` 文件夹中打开之前创建的 `views.py` 文件，并在最后一行之后添加以下代码。示例的代码文件包含在 `restful_python_2_02_01` 文件夹中，位于 `Flask01/service/views.py` 文件：

```py
service.add_resource(NotificationCategoryListResource,  
    '/notification_categories/') 
service.add_resource(NotificationCategoryResource,  
    '/notification_categories/<int:id>') 
service.add_resource(NotificationListResource,  
    '/notifications/') 
service.add_resource(NotificationResource,  
    '/notifications/<int:id>')
```

每次调用 `service.add_resource` 方法都会将一个 URL 路由到一个资源；具体来说，是到之前声明的 `flask_restful.Resource` 超类的一个先前声明的子类。每当有 API 请求，并且 URL 与 `service.add_resource` 方法中指定的 URL 之一匹配时，Flask 将调用与请求中指定的类匹配的 HTTP 动词的方法。

# 注册蓝图和运行迁移

在 `service` 文件夹中创建一个新的 `app.py` 文件。以下行显示了创建 Flask 应用程序的代码。示例的代码文件包含在 `restful_python_2_02_01` 文件夹中，位于 `Flask01/service/app.py` 文件：

```py
from flask import Flask 
from flask_sqlalchemy import SQLAlchemy 
from flask_migrate import Migrate 
from models import orm 
from views import service_blueprint 

def create_app(config_filename): 
    app = Flask(__name__) 
    app.config.from_object(config_filename) 
    orm.init_app(app) 
    app.register_blueprint(service_blueprint, url_prefix='/service') 
    migrate = Migrate(app, orm) 
    return app 

app = create_app('config') 
```

`service/app.py` 文件中的代码声明了一个 `create_app` 函数...

# 验证 PostgreSQL 数据库的内容

在我们运行前面的脚本之后，我们可以使用 PostgreSQL 命令行或任何允许我们轻松验证 PostgreSQL 10.5 数据库内容的其他应用程序来检查迁移生成的表。

运行以下命令以列出生成的表。如果您使用的数据库名称不是 `flask_notifications`，请确保您使用适当的数据库名称。示例的代码文件包含在 `restful_python_2_02_01` 文件夹中，位于 `Flask01/list_database_tables.sql` 文件：

```py
psql --username=your_user_name --dbname=flask_notifications --command="\dt"
```

以下行显示了所有生成的表名的输出：

```py

      **                    List of relations**
 **Schema |         Name          | Type  |     Owner** 
      **--------+-----------------------+-------+----------------** ** public | alembic_version       | table | your_user_name** ** public | notification          | table | your_user_name** ** public | notification_category | table | your_user_name** **(3 rows)**
```

SQLAlchemy 根据我们模型中包含的信息生成了以下两个表，具有唯一约束和外键：

+   `notification_category`：此表持久化 `NotificationCategory` 模型。

+   `notification`：此表持久化 `Notification` 模型。

以下命令将在我们向 RESTful API 发送 HTTP 请求并执行两个表上的 CRUD 操作后，允许你检查两个表的内容。这些命令假设你在运行命令的同一台计算机上运行 PostgreSQL 10.5。示例代码文件包含在`restful_python_2_02_01`文件夹中的`Flask01/check_tables_contents.sql`文件中：

```py
psql --username=your_user_name --dbname=flask_notifications --command="SELECT * FROM notification_category;"
psql --username=your_user_name --dbname=flask_notifications --command="SELECT * FROM notification;"

```

而不是使用 PostgreSQL 的命令行工具，你可以使用你喜欢的 GUI 工具来检查 PostgreSQL 数据库的内容。

Alembic 生成了一个名为`alembic_version`的额外表，该表在`version_num`列中保存数据库的版本号。这个表使得迁移命令能够检索数据库的当前版本，并根据我们的需求升级或降级。

# 创建和检索相关资源

现在，我们将使用`flask`脚本启动 Flask 的开发服务器和我们的 RESTful API。我们想启用调试模式，因此我们将`FLASK_ENV`环境变量的值设置为`development`。

在 Linux 或 macOS 的 bash shell 中的终端运行以下命令：

```py
    export FLASK_ENV=development
```

在 Windows 中，如果你正在使用命令提示符，请运行以下命令：

```py
    set FLASK_ENV=development
```

在 Windows 中，如果你正在使用 Windows PowerShell，请运行以下命令：

```py
    $env:FLASK_ENV = "development"
```

现在，运行启动 Flask 开发服务器和应用程序的`flask`脚本。

现在已经将`FLASK_ENV`环境变量配置为在开发模式下工作...

# 测试你的知识

让我们看看你是否能正确回答以下问题：

1.  以下哪个命令启动 Flask 开发服务器和 Flask 应用程序，并使其在`5000`端口上监听所有接口？

    1.  `flask run -h 0.0.0.0`

    1.  `flask run -p 0.0.0.0 -h 5000`

    1.  `flask run -p 0.0.0.0`

1.  `Flask-Migrate`是：

    1.  一个轻量级的库，用于将复杂的数据类型转换为原生 Python 数据类型，以及从原生 Python 数据类型转换回复杂的数据类型。

    1.  一个使用 Alembic 包来处理 Flask 应用程序的 SQLAlchemy 数据库迁移的库。

    1.  一个替代 SQLAlchemy 以在 PostgreSQL 上运行查询的库。

1.  Marshmallow 是：

    1.  一个轻量级的库，用于将复杂的数据类型转换为和从原生 Python 数据类型。

    1.  一个 ORM。

    1.  一个轻量级的 Web 框架，用于替代 Flask。

1.  SQLAlchemy 是：

    1.  一个轻量级的库，用于将复杂的数据类型转换为和从原生 Python 数据类型。

    1.  一个 ORM。

    1.  一个轻量级的 Web 框架，用于替代 Flask。

1.  `marshmallow.pre_load`装饰器：

    1.  在`Resource`类的任何实例创建后注册一个要调用的方法。

    1.  在序列化对象后注册一个要调用的方法。

    1.  在反序列化对象之前注册一个要调用的方法。

1.  `Schema`子类的任何实例的`dump`方法：

    1.  将 URL 路由到 Python 原语。

    1.  将作为参数传递的实例或实例集合持久化到数据库中。

    1.  接收作为参数传递的实例或实例集合，并将`Schema`子类中指定的字段过滤和输出格式应用于实例或实例集合。

1.  当我们将属性声明为`marshmallow.fields.Nested`类的实例时：

    1.  该字段将根据`many`参数的值嵌套单个`Schema`或`Schema`集合。

    1.  该字段将嵌套单个`Schema`。如果我们想嵌套`Schema`集合，我们必须使用`marshmallow.fields.NestedCollection`类的实例。

    1.  该字段将嵌套一个`Schema`集合。如果我们想嵌套单个`Schema`，我们必须使用`marshmallow.fields.NestedSingle`类的实例。

# 摘要

在本章中，我们扩展了上一章中创建的 RESTful API 的前一个版本的功能。我们使用 SQLAlchemy 作为我们的 ORM 来与 PostgreSQL 10.5 数据库一起工作。我们添加了许多包来简化许多常见任务，我们为模型及其关系编写了代码，并与模式一起工作以验证、序列化和反序列化这些模型。

我们将蓝图与资源路由相结合，从而能够从模型生成数据库。我们向 RESTful API 发送了许多 HTTP 请求，并分析了我们的代码中每个 HTTP 请求的处理方式以及模型在数据库表中的持久化情况。

现在我们已经使用 Flask、Flask-RESTful 和 SQLAlchemy 构建了一个复杂的 API，...
