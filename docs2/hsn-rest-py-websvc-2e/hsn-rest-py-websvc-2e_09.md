# 使用 Pyramid 1.10 开发 RESTful API

在本章中，我们将使用 Pyramid 1.10 来创建一个执行简单数据源 CRUD 操作的 RESTful Web API。我们将探讨以下主题：

+   设计一个与简单数据源交互的 RESTful API

+   理解每个 HTTP 方法执行的任务

+   使用 Pyramid 1.10 设置虚拟环境

+   基于模板创建新的 Pyramid 项目

+   创建模型

+   使用字典作为存储库

+   创建 Marshmallow 模式以验证、序列化和反序列化模型

+   与视图可调用和视图配置一起工作

+   理解和配置视图处理器

+   使用命令行工具向 API 发送 HTTP 请求

# 设计一个与简单数据源交互的 RESTful API

一位赢得数十场国际冲浪比赛的冲浪者成为了一名冲浪教练，并希望构建一个新工具来帮助冲浪者为奥运会训练。与冲浪教练合作的开发团队在与 Pyramid 网络框架合作方面拥有多年的经验，因此，他希望我们使用 Pyramid 构建一个简单的 RESTful API，以处理连接到冲浪板多个传感器的物联网板提供的数据。

每个物联网板将提供以下数据：

+   **状态**：每个冲浪者的湿式连体衣中嵌入的许多可穿戴无线传感器和其他包含在冲浪板中的传感器将提供数据，物联网板将对数据进行实时分析以指示...

# 使用 Pyramid 1.10 设置虚拟环境

在第一章《使用 Flask 1.0.2 开发 RESTful API 和微服务》中，我们了解到，在本书中，我们将使用 Python 3.4 中引入并改进的轻量级虚拟环境。现在，我们将遵循许多步骤来创建一个新的轻量级虚拟环境，以使用 Pyramid 1.10。如果您对现代 Python 中的轻量级虚拟环境没有经验，强烈建议您阅读第一章《使用 Flask 1.0.2 开发 RESTful API 和微服务》中名为*与轻量级虚拟环境一起工作*的部分。本章包含了我们将遵循的步骤的所有详细解释。

以下命令假设您已在 Linux、macOS 或 Windows 上安装了 Python 3.6.6。

首先，我们必须选择我们的轻量级虚拟环境的目标文件夹或目录。以下是我们将在示例中使用的 Linux 和 macOS 的路径：

```py
    ~/HillarPythonREST2/Pyramid01  
```

虚拟环境的目标文件夹将是我们家目录中的`HillarPythonREST2/Pyramid01`文件夹。例如，如果我们的 macOS 或 Linux 的家目录是`/Users/gaston`，虚拟环境将在`/Users/gaston/HillarPythonREST2/Pyramid01`中创建。您可以在每个命令中用您想要的路径替换指定的路径。

以下是我们将在示例中使用的 Windows 的路径：

```py
    %USERPROFILE%\HillarPythonREST2\Pyramid01
```

虚拟环境的目标文件夹将是我们的用户配置文件文件夹中的 `HillarPythonREST2\Pyramid01` 文件夹。例如，如果我们的用户配置文件文件夹是 `C:\Users\gaston`，则虚拟环境将在 `C:\Users\gaston\HillarPythonREST2\Pyramid01` 中创建。当然，您可以在每个命令中将指定的路径替换为您想要的路径。

在 Windows PowerShell 中，之前的路径将是以下：

```py
    $env:userprofile\HillarPythonREST2\Pyramid01
```

现在，我们必须使用 `-m` 选项，后跟 `venv` 模块名称和所需的路径，以便 Python 将此模块作为脚本运行并创建指定路径的虚拟环境。根据我们创建虚拟环境的平台，说明可能会有所不同。因此，请确保您遵循您操作系统的说明：

1.  在 Linux 或 macOS 中打开终端并执行以下命令以创建虚拟环境：

```py
python3 -m venv ~/HillarPythonREST2/Pyramid01
```

1.  在 Windows 中，请在命令提示符中执行以下命令以创建虚拟环境：

```py
python -m venv %USERPROFILE%\HillarPythonREST2\Pyramid01
```

1.  如果您想使用 Windows PowerShell，请执行以下命令以创建虚拟环境：

```py
python -m venv $env:userprofile\HillarPythonREST2\Pyramid01 
```

之前的命令不会产生任何输出。现在我们已经创建了虚拟环境，我们将运行特定于平台的脚本以激活它。激活虚拟环境后，我们将安装仅在此虚拟环境中可用的包。

1.  如果您的终端配置为在 macOS 或 Linux 中使用 `bash` shell，请运行以下命令以激活虚拟环境。该命令也适用于 `zsh` shell：

```py
source ~/HillarPythonREST2/Pyramid01/bin/activate
```

1.  如果您的终端配置为使用 `csh` 或 `tcsh` shell，请运行以下命令以激活虚拟环境：

```py
source ~/HillarPythonREST2/Pyramid01/bin/activate.csh  
```

1.  如果您的终端配置为使用 `fish` shell，请运行以下命令以激活虚拟环境：

```py
source ~/HillarPythonREST2/Pyramid01/bin/activate.fish  
```

1.  在 Windows 中，您可以在命令提示符中运行批处理文件，或者在 Windows PowerShell 中运行脚本以激活虚拟环境。如果您更喜欢命令提示符，请在 Windows 命令行中运行以下命令以激活虚拟环境：

```py
%USERPROFILE%\HillarPythonREST2\Pyramid01\Scripts\activate.bat
```

1.  如果您更喜欢 Windows PowerShell，启动它并运行以下命令以激活虚拟环境。但是请注意，您应该在 Windows PowerShell 中启用脚本执行才能运行脚本：

```py
cd $env:USERPROFILE
HillarPythonREST2\Pyramid01\Scripts\Activate.ps1
```

激活虚拟环境后，命令提示符将显示虚拟环境的根文件夹名称，用括号括起来作为默认提示的前缀，以提醒我们我们正在虚拟环境中工作。在这种情况下，我们将看到（`Pyramid01`）作为命令提示符的前缀，因为已激活的虚拟环境的根文件夹是 `Pyramid01`。

我们已经遵循了必要的步骤来创建和激活虚拟环境。现在，我们将创建一个`requirements.txt`文件来指定我们的应用程序在任意支持平台上需要安装的包集。这样，在任意新的虚拟环境中重复安装指定包及其版本将变得极其容易。

使用您喜欢的编辑器在最近创建的虚拟环境的根目录下创建一个名为`requirements.txt`的新文本文件。以下行显示了声明我们的 API 所需的包和版本的文件内容。示例代码文件包含在`restful_python_2_11_01`文件夹中，在`Pyramid01/requirements.txt`文件中：

```py
pyramid==1.10 
cookiecutter==1.6.0 
httpie==1.0.2 
```

`requirements.txt`文件中的每一行都指示需要安装的包和版本。在这种情况下，我们通过使用`==`运算符使用确切版本，因为我们想确保安装了指定的版本。以下表格总结了我们所指定的作为要求的包和版本号：

| 包名 | 要安装的版本 |
| --- | --- |
| `pyramid` | 1.10.1 |
| `cookiecutter` | 1.6.0 |
| `httpie` | 1.0.2 |

`cookiecutter`包安装了一个命令行工具，使得可以从项目模板中创建 Pyramid 项目。我们将使用这个工具创建一个基本的 Pyramid 1.10 项目，然后进行必要的更改来构建我们的 RESTful API，而不需要从头开始编写所有代码。请注意，我们将在稍后通过在 Pyramid 的`setup.py`文件中指定额外的必需包来安装额外的包。

进入虚拟环境的根目录：`Pyramid01`。在 macOS 或 Linux 中，输入以下命令：

```py
    cd ~/HillarPythonREST2/Pyramid01
```

在 Windows 命令提示符中，输入以下命令：

```py
    cd /d %USERPROFILE%\HillarPythonREST2\Pyramid01

```

在 Windows PowerShell 中，输入以下命令：

```py
    cd $env:USERPROFILE
    cd HillarPythonREST2\Pyramid01
```

现在，我们必须在 macOS、Linux 或 Windows 上运行以下命令，使用`pip`通过最近创建的`requirements.txt`文件安装上一表中解释的包和版本。在运行命令之前，请确保您位于包含`requirements.txt`文件的文件夹中（`Pyramid01`）：

```py
pip install -r requirements.txt 
```

输出的最后几行将指示`pyramid`、`cookiecutter`、`httpie`及其依赖项已成功安装：

```py
Installing collected packages: translationstring, plaster, PasteDeploy, plaster-pastedeploy, zope.deprecation, venusian, zope.interface, webob, hupper, pyramid, future, six, python-dateutil, arrow, MarkupSafe, jinja2, jinja2-time, click, chardet, binaryornot, poyo, urllib3, certifi, idna, requests, whichcraft, cookiecutter, Pygments, httpie
      Running setup.py install for future ... done
      Running setup.py install for arrow ... done
Successfully installed MarkupSafe-1.1.0 PasteDeploy-1.5.2 Pygments-2.2.0 arrow-0.12.1 binaryornot-0.4.4 certifi-2018.10.15 chardet-3.0.4 click-7.0 cookiecutter-1.6.0 future-0.17.1 httpie-1.0.2 hupper-1.4 idna-2.7 jinja2-2.10 jinja2-time-0.2.0 plaster-1.0 plaster-pastedeploy-0.6 poyo-0.4.2 pyramid-1.10.1 python-dateutil-2.7.5 requests-2.20.0 six-1.11.0 translationstring-1.3 urllib3-1.24.1 venusian-1.1.0 webob-1.8.3 whichcraft-0.5.2 zope.deprecation-4.3.0 zope.interface-4.6.0

```

# 基于模板创建新的 Pyramid 项目

现在，我们将使用应用程序模板（也称为**脚手架**）来生成一个 Pyramid 项目。请注意，您需要在开发计算机上安装 Git 才能使用下一个命令。您可以访问以下网页了解更多关于 Git 的信息：[`git-scm.com`](https://git-scm.com)。

运行以下命令以使用`cookiecutter`根据`pyramid-cookiecutter-starter`模板生成新的项目。我们使用`--checkout 1.10-branch`选项来使用一个特定的分支，确保模板与 Pyramid 1.10 兼容：

```py
cookiecutter gh:Pylons/pyramid-cookiecutter-starter --checkout 1.10-branch  
```

命令将要求您输入项目的名称。输入`metrics`并按*Enter*。您将看到一个... 

# 创建模型

现在，我们将创建一个简单的`SurfboardMetricModel`类，我们将使用它来表示指标。请记住，我们不会将模型持久化到任何数据库或文件中，因此在这种情况下，我们的类将只提供所需的属性，而不提供映射信息。

在`metrics/metrics`文件夹中创建一个新的`models`子文件夹。然后，在`metrics/metrics/models`子文件夹中创建一个新的`metrics.py`文件。以下行显示了声明我们将需要用于许多类的必要导入的代码。然后，在这个文件中创建一个`SurfboardMetricModel`类。示例代码文件包含在`restful_python_2_09_01`文件夹中，位于`Pyramid01/metrics/metrics/models/metrics.py`文件中：

```py
from enum import Enum 
from marshmallow import Schema, fields 
from marshmallow_enum import EnumField 

class SurfboardMetricModel: 
    def __init__(self, status, speed_in_mph, altitude_in_feet, water_temperature_in_f): 
        # We will automatically generate the new id 
        self.id = 0 
        self.status = status 
        self.speed_in_mph = speed_in_mph 
        self.altitude_in_feet = altitude_in_feet 
        self.water_temperature_in_f = water_temperature_in_f 
```

`SurfboardMetricModel`类仅声明了一个构造函数；即`__init__`方法。该方法接收许多参数，并使用它们来初始化具有相同名称的属性：`status`、`speed_in_mph`、`altitude_in_feet`和`water_temperature_in_f`。`id`属性被设置为`0`。我们将自动递增每个通过 API 调用生成的新的冲浪指标标识符。

# 使用字典作为存储库

现在，我们将创建一个`SurfboardMetricManager`类，我们将使用它来在内存字典中持久化`SurfboardMetricModel`实例。我们的 API 方法将调用`SurfboardMetricManager`类的相关方法来检索、插入和删除`SurfboardMetricModel`实例。

保持位于`metrics.py`文件中，该文件位于`metrics/metrics/models`子文件夹中。添加以下行以声明`SurfboardMetricManager`类。示例代码文件包含在`restful_python_2_09_01`文件夹中，位于`Pyramid01/metrics/metrics/models/metrics.py`文件中：

```py
class SurfboardMetricManager(): last_id = 0 def __init__(self): self.metrics = {} def insert_metric(self, metric): self.__class__.last_id += 1 metric.id = self.__class__.last_id ...
```

# 创建 Marshmallow 模式以验证、序列化和反序列化模型

现在，我们将创建一个简单的 Marshmallow 模式，我们将使用它来验证、序列化和反序列化之前声明的`SurfboardMetricModel`模型。

保持位于`metrics.py`文件中，该文件位于`metrics/metrics/models`子文件夹中。添加以下行以声明`SurferStatus`枚举和`SurfboardMetricSchema`类。示例代码文件包含在`restful_python_2_09_01`文件夹中，位于`Pyramid01/metrics/metrics/models/metrics.py`文件中：

```py
class SurferStatus(Enum): 
    IDLE = 0 
    PADDLING = 1 
    RIDING = 2 
    RIDE_FINISHED = 3 
    WIPED_OUT = 4 

class SurfboardMetricSchema(Schema): 
    id = fields.Integer(dump_only=True) 
    status = EnumField(SurferStatus, required=True) 
    speed_in_mph = fields.Integer(required=True) 
    altitude_in_feet = fields.Integer(required=True) 
    water_temperature_in_f = fields.Integer(required=True) 
```

首先，代码声明了我们将用于将描述映射到整数的`SurferStatus`枚举。我们希望 API 的用户能够指定状态为一个与`Enum`描述匹配的字符串。例如，如果用户想要创建一个新的指标，并将其状态设置为`SurferStatus.PADDLING`，他们应该在提供的 JSON 正文中使用`'PADDLING'`作为状态键的值。

然后，代码将`SurfboardMetricSchema`类声明为`marshmallow.Schema`类的子类。我们声明代表字段的属性为`marshmallow.fields`模块中声明的适当类的实例。每当我们将`dump_only`参数指定为`True`值时，这意味着我们希望字段为只读。例如，我们无法在模式中为`id`字段提供值。该字段的值将由`SurfboardMetricManager`类自动生成。

`SurfboardMetricSchema`类将`status`属性声明为`marshmallow_enum.EnumField`类的实例。`enum`参数设置为`SurferStatus`以指定只有此`Enum`的成员将被视为有效值。因此，在反序列化过程中，只有与`SurferStatus Enum`中的描述匹配的字符串将被接受为该字段的有效值。此外，每当此字段被序列化时，将使用`Enum`描述的字符串表示形式。

`speed_in_mph`、`altitude_in_feet`和`water_temperature_in_f`属性是`fields.Integer`类的实例，`required`参数设置为`True`。

# 与视图调用和视图配置一起工作

我们的 RESTful API 不会使用由应用程序模板生成的位于`metrics/metrics/views`子文件夹中的两个模块。因此，我们必须删除`metrics/metrics/views/default.py`和`metrics/metrics/views/notfound.py`文件。

Pyramid 使用视图调用作为 RESTful API 的主要构建块。每当有请求到达时，Pyramid 会找到并调用适当的视图调用以处理请求并返回适当的响应。

视图调用是可调用的 Python 对象，如函数、类或实现`__call__`方法的实例。任何视图调用都会接收到一个名为`request`的参数，该参数将提供代表...的`pyramid.request.Request`实例。

# 理解和配置视图处理器

以下表格显示了对于每个 HTTP 动词和范围的组合，我们想要执行的功能以及标识每个资源的路由名称：

| HTTP 动词 | 范围 | 路由名称 | 函数 |
| --- | --- | --- | --- |
| `GET` | 指标集合 | `'metrics'` | `metrics_collection` |
| `GET` | 指标 | `'metric'` | `metric` |
| `POST` | 指标集合 | `'metrics'` | `metrics_collection` |
| `DELETE` | 指标 | `'metrics'` | `metric` |

我们必须进行必要的资源路由配置来调用适当的函数，通过定义适当的路由传递所有必要的参数，并将适当的视图调用与路由匹配。

首先，我们将检查我们使用的应用程序模板是如何配置并返回一个将运行我们的 RESTful API 的 Pyramid WSGI 应用程序的。以下行显示了位于`metrics/metrics`文件夹中的`__init__.py`文件的代码：

```py
from pyramid.config import Configurator 

def main(global_config, **settings): 
    """ This function returns a Pyramid WSGI application. 
    """ 
    with Configurator(settings=settings) as config: 
        config.include('pyramid_jinja2') 
        config.include('.routes') 
        config.scan() 
    return config.make_wsgi_app() 
```

我们已经移除了对`jinja2`模板的使用，因此从之前的代码中删除了高亮行。示例代码文件包含在`restful_python_2_09_01`文件夹中，位于`Pyramid01/metrics/metrics/__init__.py`文件中。

代码定义了一个`main`函数，该函数创建一个名为`config`的`pyramid.config.Configurator`实例，并将接收到的`settings`作为参数。`main`函数使用`'.routes'`作为参数调用`config.include`方法，以包含来自`routes`模块的单个参数名为`config`的配置可调用项。这个可调用项将接收`config`参数中的`Configurator`实例，并将能够调用其方法来执行路由的适当配置。我们在分析完之前的代码后，将替换`routes`模块的现有代码。

然后，代码调用`config.scan`方法来扫描 Python 包和子包中具有特定装饰器对象的可调用项，这些装饰器对象执行配置，例如我们使用`@view.config`装饰器声明的函数。

最后，代码调用`config.make_wsgi_app`方法来提交任何挂起的配置语句，并返回代表提交的配置状态的 Pyramid WSGI 应用程序。这样，Pyramid 完成配置过程并启动服务器。

打开位于`metrics/metrics`文件夹中的现有`routes.py`文件，并用以下行替换现有代码。示例代码文件包含在`restful_python_2_09_01`文件夹中，位于`Pyramid01/metrics/metrics/routes.py`文件中：

```py
from metrics.views.metrics import metric, metrics_collection 

def includeme(config): 
    # Define the routes for metrics 
    config.add_route('metrics', '/metrics/') 
    config.add_route('metric', '/metrics/{id:\d+}/')         
    # Match the metrics views with the appropriate routes 
    config.add_view(metrics_collection,  
        route_name='metrics',  
        renderer='json') 
    config.add_view(metric,  
        route_name='metric',  
        renderer='json') 
```

代码定义了一个`includeme`函数，该函数接收之前解释过的`pyramid.config.Configurator`实例作为`config`参数。首先，代码两次调用`config.add_route`方法，将名为`'metrics'`的路由与`'/metrics/'`模式关联，将名为`'metric'`的路由与`'metrics/{id:\d+}/'`模式关联。请注意，`id`后面的分号（`;`）后面跟着一个正则表达式，确保`id`只由数字组成。

然后，代码两次调用`config.add_view`方法来指定视图可调用`metrics_collection`作为当路由名称等于`'metrics'`时必须调用的函数，以及视图可调用`metric`作为当路由名称等于`'metric'`时必须调用的函数。在这两种情况下，`config.add_view`方法指定我们希望使用`'json'`作为响应的渲染器。

# 使用命令行工具向 API 发送 HTTP 请求

`metrics/development.ini`文件是一个设置文件，它定义了开发环境中的 Pyramid 应用程序和服务器配置。与大多数`.ini`文件一样，配置设置按部分组织。例如，`[server:main]`部分指定了监听设置的值为`localhost:6543`，以便`waitress`服务器在端口`6543`上监听并绑定到 localhost 地址。

当我们基于模板创建新应用程序时，包含了此文件。打开`metrics/development.ini`文件，找到指定`pyramid.debug_routematch`设置`bool`值的以下行。示例的代码文件包含在`restful_python_2_09_01 ...`

# 测试你的知识

让我们看看你是否能正确回答以下问题：

1.  在 Pyramid 中，视图可调用是以下哪个？

    1.  实现了`__call__`方法的 Python 对象，如函数、类或实例

    1.  继承自`pyramid.views.Callable`超类的类

    1.  `pyramid.views.Callable`类的实例

1.  任何视图可调用接收到的`request`参数代表一个 HTTP 请求，它是以下哪个类的实例？

    1.  `pyramid.web.Request`

    1.  `pyramid.request.Request`

    1.  `pyramid.callable.Request`

1.  以下哪个属性允许我们在`pyramid.response.Response`实例中指定响应的状态码？

    1.  `status`

    1.  `http_status_code`

    1.  `status_code`

1.  在`pyramid.httpexceptions`模块中声明的以下哪个类代表响应的 HTTP `201 Created`状态码？

    1.  `HTTP_201_Created`

    1.  `HTTP_Created`

    1.  `HTTPCreated`

1.  以下哪个属性允许我们在`pyramid.response.Response`实例中指定 JSON 响应的响应体？

    1.  `json_body`

    1.  `body`

    1.  `body_as_json`

# 摘要

在本章中，我们使用 Pyramid 1.10 设计了一个 RESTful API 来与一个简单的数据源交互。我们定义了 API 的需求，并理解了每个 HTTP 方法执行的任务。我们使用 Pyramid 设置了一个虚拟环境，从一个现有的模板中构建了一个新的应用程序，并将额外的必需包添加到了 Pyramid 应用程序中。

我们创建了一个表示冲浪板指标的类，以及额外的类，以便能够生成一个简单的数据源，使我们能够专注于特定的 Pyramid 功能来构建 RESTful API。

然后，我们创建了一个 Marshmallow 模式来验证、序列化和反序列化指标模型。然后，我们开始使用视图可调用函数来处理特定的 HTTP ...
