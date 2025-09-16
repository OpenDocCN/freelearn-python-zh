# 15. Django 第三方库

概述

本章向您介绍 Django 第三方库。您将使用`dj-database-urls`通过 URL 配置您的数据库连接，并使用`django-crispy-forms`检查和调试您的应用程序，您将增强表单的外观，并通过使用`crispy`模板标签减少您需要编写的代码量。我们还将介绍`django-allauth`库，它允许您使用第三方提供者对用户进行身份验证。在最后的活动中，我们将使用`django-crispy-forms`增强 Bookr 的表单。

# 简介

由于 Django 自 2007 年以来一直存在，因此有一个丰富的第三方库生态系统，可以将其连接到应用程序以提供额外功能。到目前为止，我们已经学到了很多关于 Django 的知识，并使用了其许多功能，包括数据库模型、URL 路由、模板、表单等等。我们直接使用这些 Django 工具来构建 Web 应用程序，但现在我们将探讨如何利用他人的工作来快速为我们自己的应用程序添加更多高级功能。我们提到了用于存储文件的程序（在*第五章*，*服务静态文件*中，我们提到了一个程序，`django-storages`，可以将我们的静态文件存储在 CDN 上），但除了文件存储之外，我们还可以使用它们连接到第三方身份验证系统、集成支付网关、自定义设置构建方式、修改图像、更轻松地构建表单、调试我们的网站、使用不同类型的数据库等等。可能性很大，如果您想添加某个功能，可能已经存在一个应用程序可以做到这一点。

由于 Django 自 2007 年以来一直存在，因此有一个丰富的第三方库生态系统，可以将其连接到应用程序以提供额外功能。到目前为止，我们已经学到了很多关于 Django 的知识，并使用了其许多功能，包括数据库模型、URL 路由、模板、表单等等。我们直接使用这些 Django 工具来构建 Web 应用程序，但现在我们将探讨如何利用他人的工作来快速为我们自己的应用程序添加更多高级功能。我们提到了用于存储文件的程序（在*第五章*，*服务静态文件*中，我们提到了一个程序，`django-storages`，可以将我们的静态文件存储在 CDN 上），但除了文件存储之外，我们还可以使用它们连接到第三方身份验证系统、集成支付网关、自定义设置构建方式、修改图像、更轻松地构建表单、调试我们的网站、使用不同类型的数据库等等。可能性很大，如果您想添加某个功能，可能已经存在一个应用程序可以做到这一点。

对于这些库中的每一个，我们将涵盖安装和基本设置以及使用方法，主要针对 Bookr 应用。它们还有更多的配置选项，以便进一步自定义以适应您的应用程序。这些应用程序都可以使用`pip`安装。

我们还将简要介绍`django-allauth`，它允许 Django 应用程序对第三方提供者（如 Google、GitHub、Facebook 和 Twitter）进行用户身份验证。我们不会详细涵盖其安装和设置，但会提供一些示例来帮助您进行配置。

## 环境变量

当我们创建一个程序时，我们通常希望用户能够配置其行为的一些方面。例如，假设你有一个连接到数据库并将找到的所有记录保存到文件的程序。通常它可能只会向终端打印出 *成功* 消息，但你可能还希望以 *调试模式* 运行它，这样它也会打印出它正在执行的 SQL 语句。

配置这样的程序有许多方法。例如，你可以让它从配置文件中读取。但在某些情况下，用户可能希望快速运行 Django 服务器并使用特定的设置（比如，调试模式），然后再关闭该设置重新运行服务器。每次都更改配置文件可能不太方便。在这种情况下，我们可以从 *环境变量* 中读取。环境变量是在操作系统中设置的键/值对，然后程序可以读取它们。它们可以通过几种方式设置：

+   你的 shell（终端）在启动时可以读取配置脚本中的变量，然后每个程序都将能够访问这些变量。

+   你可以在终端内设置一个变量，然后它将对随后启动的任何程序可用。在 Linux 和 macOS 上，这通过 `export` 命令完成；Windows 使用 `set` 命令。以这种方式设置的任何变量都将覆盖配置脚本中的变量，但仅限于当前会话。当你关闭终端时，变量就会丢失。

+   你可以在终端运行命令的同时设置环境变量。这些变量将只对正在运行的程序持续存在，并且它们会覆盖已导出的环境变量和从配置脚本中读取的变量。

+   你可以在运行中的程序内设置环境变量，并且它们只会在程序内部（或程序启动的程序）中可用。以这种方式设置的环境变量将覆盖我们刚刚设置的所有其他方法。

这些可能听起来很复杂，但我们将通过一个简短的 Python 脚本来解释它们，并展示如何以最后三种方式（第一种方法取决于你使用的 shell）设置变量。脚本还将展示如何读取环境变量。

在 Python 中，可以使用 `os.environ` 变量访问环境变量。这是一个类似于字典的对象，可以用来通过名称访问环境变量。最安全的方法是使用 `get` 方法来访问值，以防它们未设置。它还提供了一个 `setdefault` 方法，允许只在未设置值的情况下设置值（即，它不会覆盖现有的键）。

这里是一个读取环境变量的示例 Python 脚本：

```py
import os
# This will set the value since it's not already set
os.environ.setdefault('UNSET_VAR', 'UNSET_VAR_VALUE')
# This value will not be set since it's already passed
# in from the command line
os.environ.setdefault('SET_VAR', 'SET_VAR_VALUE')
print('UNSET_VAR:' + os.environ.get('UNSET_VAR', ''))
print('SET_VAR:' + os.environ.get('SET_VAR', ''))
# All these values were provided from the shell in some way
print('HOME:' + os.environ.get('HOME', ''))
print('VAR1:' + os.environ.get('VAR1', ''))
print('VAR2:' + os.environ.get('VAR2', ''))
print('VAR3:' + os.environ.get('VAR3', ''))
print('VAR4:' + os.environ.get('VAR4', ''))
```

我们通过设置一些变量来设置我们的 shell。在 Linux 或 macOS 上，我们使用 `export`（注意这些命令没有输出）：

```py
$ export SET_VAR="Set Using Export"
$ export VAR1="Set Using Export"
$ export VAR2="Set Using Export"
```

在 Windows 中，我们会在命令行中使用 `set` 命令如下：

```py
set SET_VAR="Set Using Export"
set VAR1="Set Using Export"
set VAR2="Set Using Export"
```

在 Linux 和 macOS 上，我们也可以通过在命令之前设置环境变量来提供环境变量（实际命令仅为 `python3 env_example.py`）：

```py
$ VAR2="Set From Command Line" VAR3="Also Set From Command Line" python3 env_example.py 
```

注意

注意，上述命令在 Windows 上将无法工作。对于 Windows，环境变量必须在执行之前设置，并且不能同时传递。

该命令的输出如下：

```py
UNSET_VAR:UNSET_VAR_VALUE
SET_VAR:Set Using Export
HOME:/Users/ben
VAR1:Set Using Export
VAR2:Set From Command Line
VAR3:Also Set From Command Line
VAR4:
```

+   当脚本运行 `os.environ.setdefault('UNSET_VAR', 'UNSET_VAR_VALUE')` 时，由于 shell 没有为 `UNSET_VAR` 设置值，因此值是在脚本内部设置的。输出的值是脚本本身设置的值。

+   当执行 `os.environ.setdefault('SET_VAR', 'SET_VAR_VALUE')` 时，由于 shell 已经提供了一个值，因此该值没有被设置。这是通过 `export SET_VAR="Set Using Export"` 命令设置的。

+   `HOME` 的值没有被运行的任何命令设置——这是由 shell 提供的。它是用户的主目录。这只是一个示例，说明 shell 通常提供的环境变量。

+   `VAR1` 是通过 `export` 设置的，在执行脚本时没有被覆盖。

+   `VAR2` 是通过 `export` 设置的，但在执行脚本时被覆盖了。

+   `VAR3` 仅在执行脚本时设置。

+   `VAR4` 从未设置——我们使用 `get` 方法来访问它以避免 `KeyError`。

现在已经介绍了环境变量，我们可以回到讨论需要修改 `manage.py` 以支持 `django-configurations` 的更改。

## django-configurations

将 Django 应用程序部署到生产环境时的主要考虑之一是如何配置它。正如您在本章中看到的，`settings.py` 文件是定义所有 Django 配置的地方。甚至第三方应用程序的配置也在这个文件中。您已经在 *第十二章*，*构建 REST API* 中看到过这一点，当时您正在使用 Django REST 框架。

在 Django 中提供不同的配置并在它们之间切换有许多方法。如果您已经开始在一个已经存在并且已经具有在开发和生产环境中切换配置的特定方法的现有应用程序上工作，那么您可能应该继续使用该方法。

当我们将 Bookr 部署到产品 web 服务器时，在 *第十七章*，*Django 应用程序的部署（第一部分 – 服务器设置）* 中，我们需要切换到生产配置，那时我们将使用 `django-configurations`。

要安装 `django-configurations`，请使用以下 `pip3`：

```py
pip3 install django-configurations
```

注意

对于 Windows，您可以在前面的命令中使用 `pip` 而不是 `pip3`。

输出将如下所示：

```py
Collecting django-configurations
  Using cached https://files.pythonhosted.org/packages/96/ef/bddcce16f3cd36f03c9874d8ce1e5d35f3cedea27b7d8455265e79a77c3d/django_configurations-2.2-py2.py3-none-any.whl
Requirement already satisfied: six in /Users/ben/.virtualenvs/bookr/lib/python3.6/site-packages (from django-configurations) (1.14.0)
Installing collected packages: django-configurations
Successfully installed django-configurations-2.2
```

`django-configurations`更改`settings.py`文件，使得所有设置都从您定义的类中读取，该类将是`configurations.Configuration`的子类。设置不再是`settings.py`内部的全局变量，而是您定义的类上的属性。通过使用基于类的这种方法，我们可以利用面向对象范式，特别是继承。在类中定义的设置可以继承另一个类中的设置。例如，生产设置类可以继承开发设置类，并仅覆盖一些特定的设置——例如在生产中强制`DEBUG`为`False`。

我们可以通过仅显示文件中的前几个设置来展示需要对设置文件进行的更改。标准的 Django `settings.py`文件通常是这样开始的（已删除注释行）：

```py
import os
BASE_DIR =os.path.dirname\
          (os.path.dirname(os.path.abspath(__file__)))
SECRET_KEY =\
'y%ux@_^+#eahu3!^i2w71qtgidwpvs^o=w2*$=xy+2-y4r_!fw'
DEBUG = True
…
# The rest of the settings are not shown
```

要将设置转换为`django-configurations`，首先从`configurations`导入`Configuration`。然后定义一个`Configuration`子类。最后，将所有要设置的设置缩进到类下。在 PyCharm 中，这就像选择所有设置并按*Tab*键将它们全部缩进一样简单。

完成这些操作后，您的`settings.py`文件将看起来像这样：

```py
import os

from configurations import Configuration
class Dev(Configuration):
    BASE_DIR = os.path.dirname\
               (os.path.dirname(os.path.abspath(__file__)))
    SECRET_KEY = \
    'y%ux@_^+#eahu3!^i2w71qtgidwpvs^o=w2*$=xy+2-y4r_!fw'
    DEBUG = True
    …
    # All other settings indented in the same manner
```

要有不同的配置（不同的设置集），您只需扩展您的配置类并覆盖应该不同的设置。

例如，在生产环境中需要覆盖的一个变量是`DEBUG`：它应该是`False`（出于安全和性能原因）。可以定义一个扩展`Dev`并设置`DEBUG`的`Prod`类，如下所示：

```py
class Dev(Configuration):
    DEBUG = True
    …
    # Other settings truncated

class Prod(Dev):
    DEBUG = False
    # no other settings defined since we're only overriding DEBUG
```

当然，您也可以覆盖其他生产设置，而不仅仅是`DEBUG`。通常，出于安全考虑，您还可能需要重新定义`SECRET_KEY`和`ALLOWED_HOSTS`；并且为了配置 Django 使用您的生产数据库，您还需要设置`DATABASES`值。任何 Django 设置都可以按您选择的方式进行配置。

如果您现在尝试执行 runserver（或其他管理命令），您将得到一个错误，因为 Django 不知道如何找到这样的`settings.py`文件：

```py
django.core.exceptions.ImproperlyConfigured: django-configurations settings importer wasn't correctly installed. Please use one of the starter functions to install it as mentioned in the docs: https://django-configurations.readthedocs.io/
```

在`manage.py`文件开始工作之前，我们需要对其进行一些更改。但在我们进行更改之前，我们将简要讨论环境变量，以防您之前没有使用过它们。

## manage.py 更改

在`manage.py`中需要添加/更改两行以启用`django-configurations`。首先，我们需要定义一个默认环境变量，告诉 Django Configuration 它应该加载哪个`Configuration`类。

这行应该在`main()`函数中添加，以设置`DJANGO_CONFIGURATION`环境变量的默认值：

```py
os.environ.setdefault('DJANGO_CONFIGURATION', 'Dev')
```

这将默认设置为`Dev`——我们定义的类的名称。正如我们在示例脚本中看到的那样，如果这个值已经定义，它不会被覆盖。这将允许我们通过环境变量在配置之间切换。

第二个更改是将 `execute_from_command_line` 函数与 `django-configurations` 提供的函数交换。考虑以下行：

```py
from django.core.management import execute_from_command_line
```

这行代码的更改如下：

```py
from configurations.management import execute_from_command_line
```

从现在起，`manage.py` 将像以前一样工作，只是在启动时它会打印出它正在使用的 `Configuration` 类 (*图 15.1*)：

![图 15.1：django-configurations 正在使用配置 Dev](img/B15509_15_01.jpg)

图 15.1：django-configurations 正在使用配置 Dev

在第二行中，你可以看到 `django-configurations` 输出正在使用 `Dev` 类进行设置。

## 从环境变量配置

除了使用环境变量在 `Configuration` 类之间切换之外，`django-configurations` 还允许我们使用环境变量为单个设置提供值。它提供了 `Value` 类，这些类将自动从环境读取值。如果没有提供值，我们可以定义默认值。由于环境变量始终是字符串，因此不同的 `Value` 类用于将字符串转换为指定的类型。

让我们通过几个示例来实际看看这个。我们将允许 `DEBUG`、`ALLOWED_HOSTS`、`TIME_ZONE` 和 `SECRET_KEY` 使用以下环境变量进行设置：

```py
from configurations import Configuration, values
class Dev(Configuration):
    DEBUG = values.BooleanValue(True)
    ALLOWED_HOSTS = values.ListValue([])
    TIME_ZONE = values.Value('UTC')
    SECRET_KEY =\
    'y%ux@_^+#eahu3!^i2w71qtgidwpvs^o=w2*$=xy+2-y4r_!fw'
    …
    # Other settings truncated
class Prod(Dev):
    DEBUG = False
    SECRET_KEY = values.SecretValue()
    # no other settings are present
```

我们将逐个解释设置：

+   在 `Dev` 中，`DEBUG` 从环境变量中读取并转换为布尔值。值 `yes`、`y`、`true` 和 `1` 转换为 `True`；值 `no`、`n`、`false` 和 `0` 转换为 `False`。这允许我们在开发机器上关闭 `DEBUG`，这在某些情况下可能很有用（例如，测试自定义异常页面而不是 Django 的默认页面）。在 `Prod` 配置中，我们不希望 `DEBUG` 不小心变为 `True`，因此我们将其静态设置。

+   `ALLOWED_HOSTS` 在生产中是必需的。它是 Django 应该接受请求的主机列表。

+   `ListValue` 类将逗号分隔的字符串转换为 Python 列表。

+   例如，字符串 `www.example.com,example.com` 转换为 `["www.example.com", "example.com"]`

+   `TIME_ZONE` 只接受字符串值，因此它使用 `Value` 类设置。这个类只是读取环境变量，并不对其进行任何转换。

+   `SECRET_KEY` 在 `Dev` 配置中静态定义；它不能通过环境变量更改。在 `Prod` 配置中，它使用 `SecretValue` 设置。这类似于 `Value`，因为它只是一个字符串设置；然而，它不允许设置默认值。如果设置了默认值，则会引发异常。这是为了确保你永远不会将秘密值放入 `settings.py`，因为它可能会意外共享（例如，上传到 GitHub）。请注意，由于我们不在生产中使用 `Dev` 的 `SECRET_KEY`，所以我们不关心它是否泄露。

默认情况下，`django-configurations`期望每个环境变量都有`DJANGO_`前缀。例如，要设置`DEBUG`，使用`DJANGO_DEBUG`环境变量；要设置`ALLOWED_HOSTS`，使用`DJANGO_ALLOWED_HOSTS`，依此类推。

现在我们已经介绍了`django-configurations`以及需要对该项目进行的更改以支持它，让我们将其添加到 Bookr 并做出这些更改。在下一个练习中，你将在 Bookr 中安装和设置`django-configurations`。

## 练习 15.01：Django 配置设置

在这个练习中，你将使用`pip`安装`django-configurations`，然后更新`settings.py`以添加`Dev`和`Prod`配置。然后，你将对`manage.py`进行必要的更改以支持新的配置样式，并测试一切是否仍然正常工作：

1.  在终端中，确保你已经激活了`bookr`虚拟环境，然后运行以下命令使用`pip3`安装`django-configurations`：

    ```py
    pip3 install django-configurations
    ```

    注意

    对于 Windows，你可以在前面的命令中使用`pip`而不是`pip3`。

    安装过程将运行，你应该会有像*图 15.2*这样的输出：

    ![图 15.2：使用 pip 安装 django-configurations    ](img/B15509_15_02.jpg)

    图 15.2：使用 pip 安装 django-configurations

1.  在 PyCharm 中，打开`bookr`包内的`settings.py`。在现有的`os`导入下面，从`configurations`导入`Configuration`和`values`，如下所示：

    ```py
    from configurations import Configuration, values
    ```

1.  在导入之后但在你的第一个设置定义之前（设置`BASE_DIR`值的行），添加一个新的`Configuration`子类，称为`Dev`：

    ```py
    class Dev(Configuration):
    ```

1.  现在我们需要将所有现有的设置移动，使它们成为`Dev`类的属性而不是全局变量。在 PyCharm 中，这就像选择所有设置，然后按*Tab*键缩进它们一样简单。完成此操作后，你的设置应该如下所示：![图 15.3：新的 Dev 配置    ](img/B15509_15_03.jpg)

    图 15.3：新的 Dev 配置

1.  在缩进设置后，我们将更改一些设置，使其从环境变量中读取。首先，将`DEBUG`更改为以`BooleanValue`读取。它应该默认为`True`。考虑以下行：

    ```py
        DEBUG = True
    ```

    然后将其更改为：

    ```py
        DEBUG = values.BooleanValue(True)
    ```

    这将自动从`DJANGO_DEBUG`环境变量中读取`DEBUG`并将其转换为布尔值。如果环境变量未设置，则默认为`True`。

1.  还将`ALLOWED_HOSTS`转换为从环境变量中读取，使用`values.ListValue`类。它应该默认为`[]`（空列表）。考虑以下行：

    ```py
        ALLOWED_HOSTS = []
    ```

    然后将其更改为：

    ```py
        ALLOWED_HOSTS = values.ListValue([])
    ```

    `ALLOWED_HOSTS`将从`DJANGO_ALLOWED_HOSTS`环境变量中读取，并默认为空列表。

1.  到目前为止，你所做的一切都是在`Dev`类上添加/更改属性。现在，在同一文件的末尾，添加一个从`Dev`继承的`Prod`类。它应该定义两个属性，`DEBUG = True`和`SECRET_KEY = values.SecretValue()`。完成的类应该看起来像这样：

    ```py
    class Prod(Dev):
        DEBUG = False
        SECRET_KEY = values.SecretValue()
    ```

    保存`settings.py`。

1.  如果我们现在尝试运行任何管理命令，我们将收到一个错误，表明`django-configurations`设置不正确。我们需要对`manage.py`做一些更改才能使其再次工作。在`bookr`项目目录中打开`manage.py`。

    考虑以下行：

    ```py
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bookr.settings')
    ```

    在它下面，添加以下行：

    ```py
    os.environ.setdefault('DJANGO_CONFIGURATION', 'Dev')
    ```

    这将默认配置设置为`Dev`类。它可以由设置`DJANGO_CONFIGURATION`环境变量来覆盖（例如，设置为`Prod`）。

1.  在上一步的下一行，你已经有以下`import`语句：

    ```py
    from django.core.management import execute_from_command_line
    ```

    将其更改为：

    ```py
    from manage.py script use Django Configuration's execute_from_command_line function, instead of the Django built-in one.Save `manage.py`.
    ```

1.  启动 Django 开发服务器。如果它没有错误地开始，你可以确信你做的更改已经生效。为了确保这一点，检查页面是否在浏览器中加载。打开`http://127.0.0.1:8000/`并尝试浏览网站。一切应该看起来和感觉都像之前一样：![图 15.4：Bookr 网站应该看起来和感觉都像之前    ](img/B15509_15_04.jpg)

图 15.4：Bookr 网站应该看起来和感觉都像之前

在这个练习中，我们安装了`django-configurations`并重构了我们的`settings.py`文件，使用其`Configuration`类来定义我们的设置。我们添加了`Dev`和`Prod`配置，并使`DEBUG`、`ALLOWED_HOSTS`和`SECRET_KEY`可以通过环境变量设置。最后，我们更新了`manage.py`以使用 Django Configuration 的`execute_from_command_line`函数，这使得使用新的`settings.py`格式成为可能。

在下一节中，我们将介绍`dj-database-url`，这是一个使您能够使用 URL 配置 Django 数据库设置的包。

## dj-database-url

`dj-database-url`是另一个帮助配置 Django 应用程序的 app。具体来说，它允许您使用 URL 而不是配置值的字典来设置数据库（您的 Django 应用程序连接到）。正如您在现有的`settings.py`文件中可以看到的，`DATABASES`设置包含几个条目，当使用具有更多配置选项的不同数据库（例如用户名、密码等）时，它会变得更加详细。我们可以从 URL 设置这些值，该 URL 可以包含所有这些值。

根据您是使用本地 SQLite 数据库还是远程数据库服务器，URL 的格式会有所不同。要使用磁盘上的 SQLite（如 Bookr 目前所做的那样），URL 是这样的：

```py
sqlite:///<path>
```

注意这里有三个斜杠。这是因为 SQLite 没有主机名，所以这就像一个 URL 是这样的：

```py
<protocol>://<hostname>/<path>
```

即，URL 有一个空的主机名。因此，所有三个斜杠都在一起。

要为远程数据库服务器构建一个 URL，其格式通常是这样的：

```py
<protocol>://<username>:<password>@<hostname>:<port>/<database_name>
```

例如，要连接到主机`db.example.com`上的名为`bookr_django`的 PostgreSQL 数据库，端口为`5432`，用户名为`bookr`，密码为`b00ks`，URL 将是这样的：

```py
postgres://bookr:b00ks@db.example.com:5432/bookr_django
```

现在我们已经看到了 URL 的格式，让我们看看如何在 `settings.py` 文件中实际使用它们。首先，必须使用 `pip3` 安装 `dj-database-url`：

```py
pip3 install dj-database-url
```

注意

对于 Windows，你可以在前面的命令中使用 `pip` 而不是 `pip3`。

输出如下：

```py
Collecting dj-database-url
  Downloading https://files.pythonhosted.org/packages/d4/a6/4b8578c1848690d0c307c7c0596af2077536c9ef2a04d42b00fabaa7e49d/dj_database_url-0.5.0-py2.py3-none-any.whl
Installing collected packages: dj-database-url
Successfully installed dj-database-url-0.5.0
```

现在，可以将 `dj_database_url` 导入到 `settings.py` 中，并使用 `dj_database_url.parse` 方法将 URL 转换为 Django 可以使用的字典。我们可以使用它的返回值来设置 `DATABASES` 字典中的 `default`（或其它）项：

```py
import dj_database_url
DATABASES = {'default':dj_database_url.parse\
             ('postgres://bookr:b00ks@db.example.com:5432/\
               bookr_django')}
```

或者，对于我们的 SQLite 数据库，我们可以利用已经存在的 `BASE_DIR` 设置，并将其包含在 URL 中：

```py
import dj_database_url
DATABASES = {'default': dj_database_url.parse\
             ('sqlite:///{}/db.sqlite3'.format(BASE_DIR))}
```

解析后，`DATABASES` 字典与我们之前定义的类似。它包括一些不适用于 SQLite 数据库的冗余项（如 `USER`、`PASSWORD`、`HOST` 等），但 Django 会忽略它们：

```py
DATABASES = {'default': \
             {'NAME': '/Users/ben/bookr/bookr/db.sqlite3',\
              'USER': '',\
              'PASSWORD': '',\
              'HOST': '',\
              'PORT': '',\
              'CONN_MAX_AGE': 0,\
              'ENGINE': 'django.db.backends.sqlite3'}}
```

这种设置数据库连接信息的方法并不那么有用，因为我们仍然在 `settings.py` 中静态定义数据。唯一的区别是我们使用 URL 而不是字典。`dj-database-url` 还可以自动从环境变量中读取 URL。这将允许我们通过在环境中设置它们来覆盖这些值。

要从环境中读取数据，使用 dj_database_url.config 函数，如下所示：

```py
import dj_database_url
DATABASES = {'default': dj_database_url.config()}
```

URL 会自动从 `DATABASE_URL` 环境变量中读取。

我们可以通过向 `config` 函数提供 `default` 参数来改进这一点。这是在没有在环境变量中指定的情况下默认使用的 URL：

```py
import dj_database_url
DATABASES = {'default':dj_database_url.config\
             (default='sqlite:///{}/db.sqlite3'\
              .format(BASE_DIR))}
```

以这种方式，我们可以在生产环境中指定一个可以被环境变量覆盖的默认 URL。

我们还可以通过传递 `env` 参数来指定读取 URL 的环境变量——这是第一个位置参数。这样，你可以为不同的数据库设置读取多个 URL：

```py
import dj_database_url
DATABASES = {'default':dj_database_url.config\
             (default='sqlite:///{}/db.sqlite3'\
                      .format(BASE_DIR)),\
             'secondary':dj_database_url.config\
                         ('DATABASE_URL_SECONDARY'\
                          default=\
                          'sqlite:///{}/db-secondary.sqlite3'\
                          .format(BASE_DIR)),}
```

在这个例子中，`default` 项的 URL 是从 `DATABASE_URL` 环境变量中读取的，而 `secondary` 是从 `DATABASE_URL_SECONDARY` 中读取的。

`django-configurations` 还提供了一个与 `dj_database_url: DatabaseURLValue` 协同工作的配置类。它与 `dj_database_url.config` 略有不同，因为它生成包括 `default` 项在内的整个 `DATABASES` 字典。例如，考虑以下代码：

```py
import dj_database_url
DATABASES = {'default': dj_database_url.config()}
```

这段代码等同于以下代码：

```py
from configurations import values
DATABASES = values.DatabaseURLValue()
```

不要这样写 `DATABASES['default'] = values.DatabaseURLValue()`，因为你的字典将会嵌套两层。

如果你需要指定多个数据库，你需要直接回退到使用 `dj_database_url.config` 而不是使用 `DatabaseURLValue`。

与其他 `values` 类一样，`DatabaseURLValue` 将默认值作为其第一个参数。你可能还希望使用 `environment_prefix` 参数并将其设置为 `DJANGO`，以便其环境变量名称与其他设置保持一致。因此，使用 `DatabaseURLValue` 的完整示例将如下所示：

```py
DATABASES = values.DatabaseURLValue\
            ('sqlite:///{}/db.sqlite3'.format(BASE_DIR),\
              environment_prefix='DJANGO')
```

通过这样设置 `environment_prefix`，我们可以使用 `DJANGO_DATABASE_URL` 环境变量（而不是仅 `DATABASE_URL`）来设置数据库 URL。这意味着它与以 `DJANGO` 开头的其他环境变量设置保持一致，例如 `DJANGO_DEBUG` 或 `DJANGO_ALLOWED_HOSTS`。

注意，尽管我们没有在 `settings.py` 中导入 `dj-database-url`，但 `django-configurations` 会内部使用它，因此它仍然必须被安装。

在下一个练习中，我们将配置 Bookr 以使用 DatabaseURLValue 来设置其数据库配置。它将能够从环境变量中读取，并在我们指定的默认值回退。

## 练习 15.02：dj-database-url 和设置

在这个练习中，我们将使用 `pip3` 安装 `dj-database-url`。然后，我们将更新 Bookr 的 `settings.py` 文件以使用 URL 配置 `DATABASE` 设置，该 URL 从环境变量中读取：

1.  在终端中，确保你已经激活了 `bookr` 虚拟环境，然后运行以下命令使用 `pip3` 安装 `dj-database-url`：

    ```py
    pip3 install dj-database-url
    ```

    安装过程将运行，你应该有类似的输出：

    ![图 15.5：使用 pip 安装 dj-database-url](img/B15509_15_05.jpg)

    ](img/B15509_15_05.jpg)

    图 15.5：使用 pip 安装 dj-database-url

1.  在 PyCharm 中，打开 `bookr` 包目录中的 `settings.py` 文件。向下滚动以找到定义 `DATABASES` 属性的位置。将其替换为 `values.DatabaseURLValue` 类。第一个参数（默认值）应该是 SQLite 数据库的 URL：`'sqlite:///{}/db.sqlite3'.format(BASE_DIR)`。同时传递 `environ_prefix`，设置为 `DJANGO`。完成此步骤后，你应该像这样设置属性：

    ```py
    DATABASES = values.DatabaseURLValue\
                ('sqlite:///{}/db.sqlite3'.format(BASE_DIR),\
                  environ_prefix='DJANGO')
    ```

    保存 `settings.py`。

1.  启动 Django 开发服务器。与 *练习 15.01*，*Django 配置设置* 一样，如果启动正常，你可以确信你的更改是成功的。为了确保，请在浏览器中打开 `http://127.0.0.1:8000/` 并检查一切看起来和之前一样。你应该访问一个查询数据库的页面（例如 `Books List` 页面），并检查是否显示了一本书的列表：![图 15.6：Bookr 页面中的数据库查询仍然有效](img/B15509_15_06.jpg)

    ](img/B15509_15_06.jpg)

图 15.6：Bookr 页面中的数据库查询仍然有效

在这个练习中，我们更新了我们的 `settings.py` 文件，使其从环境变量中指定的 URL 确定其 `DATABASES` 设置。我们使用了 `values.DatabaseURLValue` 类来自动读取值，并提供了默认 URL。我们还设置了 `environ_prefix` 参数为 `DJANGO`，以便环境变量名称为 `DJANGO_DATABASE_URL`，这与其他设置保持一致。

在下一节中，我们将游览 Django 调试工具栏，这是一个通过浏览器帮助你调试 Django 应用程序的应用程序。

## Django 调试工具栏

Django 调试工具栏是一个应用，它直接在你的浏览器中显示有关网页的调试信息。它包括有关生成页面的 SQL 命令、请求和响应头信息、页面渲染所需时间等信息。这些信息在以下情况下可能很有用：

+   *页面加载时间过长 – 可能是因为运行了太多的数据库查询*。你可以查看是否运行了相同的查询多次，在这种情况下，你可以考虑使用缓存。否则，通过在数据库中添加索引，一些查询可能会加快速度。

+   *你想要确定为什么页面返回错误信息*。你的浏览器可能发送了你没有预料到的头信息，或者可能是 Django 的一些头信息不正确。

+   *你的页面运行缓慢，因为它在非数据库代码中花费了时间* – 你可以分析页面以查看哪些函数耗时最长。

+   *页面看起来不正确*。你可以看到 Django 渲染了哪些模板。可能有一个意外的第三方模板正在被渲染。你还可以检查所有正在使用的设置（包括我们未设置的内置 Django 设置）。这有助于确定错误的设置，导致页面行为不正确。

我们将解释如何使用 Django 调试工具栏来查看这些信息。在深入探讨如何设置 Django 调试工具栏以及如何使用它之前，让我们快速看一下它。工具栏显示在浏览器窗口的右侧，可以切换打开和关闭以显示信息：

![图 15.7：Django 调试工具栏已关闭](img/B15509_15_07.jpg)

图 15.7：Django 调试工具栏已关闭

前面的图显示了 Django 调试工具栏在关闭状态下的样子。注意窗口右上角的切换栏。点击工具栏可以打开它：

![图 15.8：Django 调试工具栏已打开](img/B15509_15_08.jpg)

图 15.8：Django 调试工具栏已打开

*图 15.8*显示了 Django 调试工具栏已打开。

使用`pip`安装 Django 调试工具栏：

```py
pip3 install django-debug-toolbar
```

注意

对于 Windows，你可以在前面的命令中使用`pip`而不是`pip3`。

然后有一些步骤来设置它，主要是通过修改`settings.py`：

1.  将`debug_toolbar`添加到`INSTALLED_APPS`设置列表中。

1.  将`debug_toolbar.middleware.DebugToolbarMiddleware`添加到`MIDDLEWARE`设置列表中。应尽可能早地进行此操作；对于 Bookr，它可以是此列表中的第一项。这是所有请求和响应都必须通过的中介。

1.  将`'127.0.0.1'`添加到`INTERNAL_IPS`设置列表中（此设置可能需要创建）。Django 调试工具栏将只显示在此列表中列出的 IP 地址。

1.  将 Django 调试工具栏的 URL 添加到基本`urls.py`文件中。我们只想在`DEBUG`模式下添加此映射：

    ```py
    path('__debug__/', include(debug_toolbar.urls))
    ```

在下一个练习中，我们将详细讲解这些步骤。

一旦安装并设置了 Django 调试工具栏，您访问的任何页面都会显示 DjDT 侧边栏（您可以使用 DjDT 菜单打开或关闭它）。当它打开时，您将能够看到另一组可以点击以获取更多信息的部分。

每个面板旁边都有一个复选框，这允许您启用或禁用该指标的收集。收集的每个指标都会略微减慢页面加载速度（尽管通常这是不明显的感觉）。如果您发现某个指标的收集速度较慢，您可以在这里将其关闭：

1.  我们将逐一介绍每个面板。第一个是`版本`，它显示了正在运行的 Django 版本。您可以点击它打开一个大的`版本`显示，它还将显示 Python 版本和 Django 调试工具栏（*图 15.9*）：![图 15.9: DjDT 版本面板（为了简洁，截图已裁剪）    ![图片](img/B15509_15_09.jpg)

    图 15.9: DjDT 版本面板（为了简洁，截图已裁剪）

1.  第二个面板是`时间`，它显示了处理请求所需的时间。它被细分为系统时间和用户时间（*图 15.10*）：![图 15.10: DjDT 时间面板    ![图片](img/B15509_15_10.jpg)

    图 15.10: DjDT 时间面板

    这些设置之间的区别超出了本书的范围，但基本上，系统时间是花费在内核中的时间（例如，进行网络或文件读写操作），而用户时间是位于操作系统内核之外（这包括你在 Django、Python 等中编写的代码）。

    还显示了在浏览器中花费的时间，例如获取请求所需的时间和渲染页面所需的时间。

1.  第三部分，`设置`，显示了应用程序使用的所有设置（*图 15.11*）：![图 15.11: DjDT 设置面板    ![图片](img/B15509_15_11.jpg)

    图 15.11: DjDT 设置面板

    这很有用，因为它显示了来自`settings.py`的设置和默认的 Django 设置。

1.  第四个面板是`标头`（*图 15.12*）。它显示了浏览器发出的请求的标头以及 Django 发送的响应标头：![图 15.12: DjDT 标头面板    ![图片](img/B15509_15_12.jpg)

    图 15.12: DjDT 标头面板

1.  第五部分，`请求`，显示了生成响应的视图以及调用时使用的参数和关键字参数（*图 15.13*）。您还可以看到在其 URL 映射中使用的 URL 名称：![图 15.13: DjDT 请求面板（为了简洁，一些面板未显示）    ![图片](img/B15509_15_13.jpg)

    图 15.13: DjDT 请求面板（为了简洁，一些面板未显示）

    它还显示了请求的 cookie、存储在会话中的信息（会话在*第八章*，*媒体服务和文件上传*中介绍）以及`request.GET`和`request.POST`数据。

1.  第六部分，`SQL`，显示了在构建响应时正在执行的 SQL 数据库查询（图 15.14）：![图 15.14: DjDT SQL 面板    ![图片](img/B15509_15_14.jpg)

    图 15.14: DjDT SQL 面板

    您可以看到每个查询执行所需的时间以及它们的执行顺序。它还会标记相似和重复的查询，以便您可能重构代码以删除它们。

    每个`SELECT`查询显示两个操作按钮，`Sel`，代表选择，和`Expl`，代表解释。这些按钮对于`INSERT`、`UDPATE`或`DELETE`查询不会显示。

    `Sel`按钮显示了执行的`SELECT`语句以及查询检索的所有数据（*图 15.15*）：

    ![图 15.15：DjDT SQL 选择面板

    ![img/B15509_15_15.jpg]

    图 15.15：DjDT SQL 选择面板

    `Expl`按钮显示了`SELECT`查询的`EXPLAIN`查询（*图 15.16*）：

    ![图 15.16：DjDT SQL 解释面板（部分面板因简洁性未显示）]

    ![img/B15509_15_16.jpg]

    图 15.16：DjDT SQL 解释面板（部分面板因简洁性未显示）

    `EXPLAIN`查询超出了本书的范围，但它们基本上显示了数据库如何尝试执行`SELECT`查询，例如，使用了哪些数据库索引。您可能会发现查询没有使用索引，因此您可以通过添加一个索引来获得更快的性能。

1.  第七个面板是“静态文件”，显示了在此请求中加载的静态文件（*图 15.17*）。它还显示了所有可用的静态文件以及它们如何被加载（即哪个静态文件查找器找到了它们）。静态文件面板的信息类似于您可以从`findstatic`管理命令中获得的信息：![图 15.17：DjDT 静态面板

    ![img/B15509_15_17.jpg]

    图 15.17：DjDT 静态面板

1.  第八个面板是“模板”，显示了已渲染的模板信息（*图 15.18*）：![图 15.18：DjDT 模板面板

    ![img/B15509_15_18.jpg]

    图 15.18：DjDT 模板面板

    它显示了模板加载的路径和继承链。

1.  第九个面板是“缓存”，显示了从 Django 缓存中检索的数据信息（*图 15.19*）：![图 15.19：DjDT 缓存面板（部分面板因简洁性未显示）

    ![img/B15509_15_19.jpg]

    图 15.19：DjDT 缓存面板（部分面板因简洁性未显示）

    由于我们在 Bookr 中没有使用缓存，因此本节为空白。如果我们使用缓存，我们将能够看到对缓存发出的请求数量，以及成功检索到项目请求的数量。我们还将看到添加到缓存中的项目数量。这可以给您一个关于您是否有效地使用缓存的线索。如果您添加了大量项目到缓存但未检索任何内容，那么您应该重新考虑您要缓存的哪些数据。相反，如果您有很多“缓存未命中”（未命中是指请求不在缓存中的数据），那么您应该比现在缓存更多的数据。

1.  第十个面板是“信号”，显示了有关 Django 信号的信息（*图 15.20*）：![图 15.20：DjDT 信号面板（部分面板因简洁性未显示）

    ![img/B15509_15_20.jpg]

    图 15.20：DjDT 信号面板（为简洁起见，一些面板未显示）

    尽管我们在这本书中没有涵盖信号，但它们类似于你可以挂钩的事件，当 Django 执行某些操作时可以执行函数；例如，如果创建了一个用户，可以发送欢迎邮件。本节显示了哪些信号被发送以及哪些函数接收了它们。

1.  第十一个面板“日志”显示了由你的 Django 应用程序生成的日志消息（如图 15.21 所示）：![图 15.21：DjDT 日志面板    ![img/B15509_15_21.jpg](img/B15509_15_21.jpg)

    图 15.21：DjDT 日志面板

    由于此请求没有生成日志消息，此面板为空。

    下一个选项“拦截重定向”不是一个包含数据的部分。相反，它允许你切换重定向拦截。如果你的视图返回重定向，它将不会跟随。相反，会显示一个类似于*图 15.22*的页面：

    ![图 15.22：DjDT 拦截的重定向    ![img/B15509_15_22.jpg](img/B15509_15_22.jpg)

    图 15.22：DjDT 拦截的重定向

    这允许你为生成重定向的视图打开 Django 调试工具栏——否则，你只能看到你被重定向到的视图的信息。

1.  最后一个面板是“分析”。默认情况下它是关闭的，因为分析可能会大大减慢你的响应速度。一旦打开，你必须刷新页面以生成分析信息（如图 15.23 所示）：![图 15.23：DjDT 分析面板    ![img/B15509_15_23.jpg](img/B15509_15_23.jpg)

图 15.23：DjDT 分析面板

这里显示的信息是关于你的响应中每个函数调用所花费时间的分解。页面左侧显示了所有执行的调用堆栈跟踪。右侧是包含时间数据的列。列包括：

+   **CumTime**：在函数及其调用的任何子函数中花费的总时间

+   `Count`)

+   **TotTime**：在此函数中花费的时间（不包括它调用的任何子函数）

+   `Count`)

+   **调用次数**：此函数的调用次数

这些信息可以帮助你确定在哪里加快你的应用程序。例如，优化一个被调用 1000 次的小函数可能比优化一个只被调用一次的大函数更容易。关于如何加快代码的更深入的建议超出了本书的范围。

## 练习 15.03：设置 Django 调试工具栏

在这个练习中，你将通过修改`INSTALLED_APPS`、`MIDDLEWARE`和`INTERNAL_IPS`设置来添加 Django 调试工具栏设置。然后你将添加`debug_toolbar.urls`映射到`bookr`包的`urls.py`。然后你将在浏览器中加载一个带有 Django 调试工具栏的页面并使用它：

1.  在终端中，确保您已激活了`bookr`虚拟环境，然后运行以下命令使用`pip3`安装 Django 调试工具栏：

    ```py
    pip3 install django-debug-toolbar
    INSTALLED_APPS = […\
                      'debug_toolbar']
    ```

    这将允许 Django 找到 Django 调试工具栏的静态文件。

1.  将`debug_toolbar.middleware.DebugToolbarMiddleware`添加到`MIDDLEWARE`设置中——它应该是列表中的第一项：

    ```py
    MIDDLEWARE = ['debug_toolbar.middleware.DebugToolbarMiddleware',\
                  …]
    ```

    这将使请求和响应通过`DebugToolbarMiddleware`路由，允许 Django 调试工具栏检查请求并将其 HTML 插入到响应中。

1.  需要添加的最后一个设置是将地址`127.0.0.1`添加到`INTERNAL_IPS`。您可能还没有定义`INTERNAL_IPS`设置，所以添加如下设置：

    ```py
    INTERNAL_IPS = ['127.0.0.1']
    ```

    这将使 Django 调试工具栏仅在开发者的计算机上显示。现在您可以保存`settings.py`。

1.  现在，我们需要添加 Django 调试工具栏的 URL。在`bookr`包目录中打开`urls.py`。我们已经有了一个检查`DEBUG`模式的`if`条件，然后添加了媒体 URL，如下所示：

    ```py
    if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,\
                              document_root=settings.MEDIA_ROOT)
    ```

    我们还将在这个`if`语句内添加`debug_toolbar.urls`的`include`，但是我们将将其添加到`urlpatterns`的开始处，而不是追加到末尾。在`if`语句内添加以下代码：

    ```py
        import debug_toolbar
        urlpatterns = [path\
                       ('__debug__/',\
                        include(debug_toolbar.urls)),] + urlpatterns
    ```

    保存`urls.py`。

1.  如果 Django 开发服务器尚未运行，请启动它并导航到`http://127.0.0.1:8000`。您应该看到 Django 调试工具栏已打开。如果没有打开，请点击右上角的`DjDT`切换按钮打开它：![图 15.25：DjDT 切换按钮显示在角落    ![图片](img/B15509_15_25.jpg)

    图 15.25：DjDT 切换按钮显示在角落

1.  尝试浏览一些面板并访问不同的页面，看看您可以找到哪些信息。还可以尝试开启`拦截重定向`，然后创建一个新的书评。提交表单后，您应该看到被拦截的页面而不是被重定向到新的评论页面（*图 15.26*）：![图 15.26：提交新评论后的重定向拦截页面    ![图片](img/B15509_15_26.jpg)

    图 15.26：提交新评论后的重定向拦截页面

    您可以点击`位置`链接跳转到它被重定向到的页面。

1.  您还可以尝试开启`性能分析`并查看哪些函数被频繁调用以及哪些函数占据了大部分渲染时间。

1.  一旦您完成对 Django 调试工具栏的实验，请关闭`拦截重定向`和`性能分析`。

在这个练习中，我们通过添加设置和 URL 映射来安装和设置 Django 调试工具栏。然后我们看到了它的实际应用，并检查了它可以给我们提供的有用信息，包括如何处理重定向和查看性能分析信息。

在下一节中，我们将查看`django-crispy-forms`应用程序，它将使我们减少编写表单所需的代码量。

# django-crispy-forms

在 Bookr 中，我们使用 Bootstrap CSS 框架。它提供了一些 CSS 类，可以应用于表单。由于 Django 独立于 Bootstrap，当我们使用 Django 表单时，它甚至不知道我们正在使用 Bootstrap，因此不知道应该将哪些类应用于表单小部件。

`django-crispy-forms`在 Django 表单和 Bootstrap 表单之间充当中间件。它可以接受 Django 表单并将其渲染为正确的 Bootstrap 元素和类。它不仅支持 Bootstrap，还支持其他框架，如`crispy-forms-foundation`）。

它的安装和设置相当简单。再次强调，它是通过`pip3`安装的：

```py
pip3 install django-crispy-forms 
```

注意

对于 Windows，你可以在前面的命令中使用`pip`而不是`pip3`。

然后，只需进行几项设置更改。首先，将`crispy_forms`添加到你的`INSTALLED_APPS`中。然后，你需要告诉`django-crispy-forms`你正在使用哪个框架，以便它加载正确的模板。这是通过`CRISPY_TEMPLATE_PACK`设置完成的。在我们的例子中，它应该设置为`bootstrap4`：

```py
CRISPY_TEMPLATE_PACK = 'bootstrap4'
```

`django-crispy-forms`有两种主要的工作模式，要么作为过滤器，要么作为模板标签。前者更容易插入到现有的模板中。后者提供了更多的配置选项，并将更多的 HTML 生成移动到`Form`类中。我们将依次查看这两种方法。

## Crispy 过滤器

使用`django-crispy-forms`渲染表单的第一种方法是使用`crispy`模板。首先，必须在模板中加载过滤器。库的名称是`crispy_forms_tags`：

```py
{% load crispy_forms_tags %}
```

然后，不要使用`as_p`方法（或其他方法）来渲染表单，而是使用`crispy`过滤器。考虑以下行：

```py
{{ form.as_p }}
```

并将其替换为：

```py
{{ form|crispy }}
```

这里是一个快速展示“创建回顾”表单的**前后对比**。除了表单的渲染之外，HTML 的其他部分都没有改变。*图 15.27*显示了标准的 Django 表单：

![图 15.27：带有默认样式的创建回顾表单![图 15.27：带有默认样式的创建回顾表单](img/B15509_15_27.jpg)

图 15.27：带有默认样式的创建回顾表单

*图 15.28*显示了在`django-crispy-forms`添加 Bootstrap 类之后的表单：

![图 15.28：通过 django-crispy-forms 添加 Bootstrap 类后的创建回顾表单![图 15.28：通过 django-crispy-forms 添加 Bootstrap 类后的创建回顾表单](img/B15509_15_28.jpg)

图 15.28：通过 django-crispy-forms 添加 Bootstrap 类后的创建回顾表单

当我们将`django-crispy-forms`集成到 Bookr 中时，我们不会使用这种方法，然而，由于它很容易插入到现有的模板中，所以了解这一点是很有价值的。

## Crispy 模板标签

使用`django-crispy-forms`渲染表单的另一种方法是使用`crispy`模板标签。要使用它，必须首先将`crispy_forms_tags`库加载到模板中（就像我们在上一节中所做的那样）。然后，表单可以这样渲染：

```py
{% crispy form %}
```

这与`crispy`过滤器有何不同？`crispy`模板标签也会为您渲染`<form>`元素和`{% csrf_token %}`模板标签。例如，你可以这样使用它：

```py
<form method="post">
  {% csrf_token %}
  {% crispy form %}
</form>
```

对于这个输出，结果如下：

```py
<form method="post" >
<input type="hidden" name="csrfmiddlewaretoken" value="…">
<form method="post">
<input type="hidden" name="csrfmiddlewaretoken" value="…">
    … form fields …
</form>
</form>
```

即，表单和 CSRF 令牌字段被重复了。为了自定义生成的 `<form>` 元素，`django-crispy-forms` 提供了一个 `FormHelper` 类，它可以被设置为 `Form` 实例的 `helper` 属性。它是 `FormHelper` 实例，`crispy` 模板标签使用它来确定 `<form>` 应该具有哪些属性。

让我们看看添加了辅助器的 `ExampleForm`。首先，导入所需的模块：

```py
from django import forms
from crispy_forms.helper import FormHelper
```

接下来，定义一个表单：

```py
class ExampleForm(forms.Form):
example_field = forms.CharField()
```

我们可以实例化一个 `FormHelper` 实例，然后将其设置为 `form.helper` 属性（例如，在一个视图中），但通常在表单的 `__init__` 方法中创建和分配它更有用。我们还没有创建一个带有 `__init__` 方法的表单，但它与其他 Python 类没有区别：

```py
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
```

接下来，我们设置辅助器和辅助器的 `form_method`（它随后将在表单 HTML 中渲染）：

```py
self.helper = FormHelper()
self.helper.form_method = 'post'
```

可以在辅助器上设置其他属性，例如 `form_action`、`form_id` 和 `form_class`。在 Bookr 中我们不需要使用这些。我们也不需要手动设置表单或其辅助器的 `enctype`，因为 `crispy` 表单标签会自动将其设置为 `multipart/form-data`，如果表单包含文件上传字段。

如果我们现在尝试渲染表单，我们将无法提交它，因为没有提交按钮（记住我们手动添加了提交按钮到我们的表单中，它们不是 Django 表单的一部分）。`django-crispy-forms` 还包括可以添加到表单中的布局辅助器。它们将在其他字段之后渲染。我们可以这样添加一个提交按钮——首先，导入 `Submit` 类：

```py
from crispy_forms.layout import Submit
```

注意

`django-crispy-forms` 并不完全支持使用 `<button>` 输入来提交表单，但就我们的目的而言，`<input type="submit">` 在功能上是相同的。

我们然后实例化它，并在一行中将其添加到辅助器的输入中：

```py
self.helper.add_input(Submit("submit", "Submit"))
```

`Submit` 构造函数的第一个参数是其 *name*，第二个参数是其 *label*。

`django-crispy-forms` 知道我们正在使用 Bootstrap，并将自动使用 `btn btn-primary` 类渲染按钮。

使用 crispy 模板标签和 `FormHelper` 的优点是，这意味着只有一处定义了属性和表单的行为。我们已经在 `Form` 类中定义了所有表单字段；这允许我们在同一位置定义表单的其他属性。我们可以轻松地将表单从 `GET` 提交更改为 `POST` 提交。然后，`FormHelper` 实例将自动知道在渲染时需要将其 CSRF 令牌添加到其 HTML 输出中。

我们将在下一个练习中将所有这些应用到实践中，在那里你将安装 `django-crispy-forms`，然后更新 `SearchForm` 以利用表单辅助器，然后使用 `crispy` 模板标签渲染它。

## 练习 15.04：使用 Django Crispy Forms 与 SearchForm

在这个练习中，您将安装 `django-crispy-forms`，然后将 `SearchForm` 转换为可以使用 `crispy` 模板标签的形式。这将通过添加一个 `__init__` 方法和在其中构建 `FormHelper` 实例来完成：

1.  在终端中，确保您已经激活了 `bookr` 虚拟环境，然后运行此命令使用 `pip3` 安装 `django-crispy-forms`：

    ```py
    pip3 install django-crispy-forms
    INSTALLED_APPS = […\
                      'reviews',\
                      'debug_toolbar',\
                      'crispy_forms'\]
    ```

    这将允许 Django 找到所需的模板。

1.  在 `settings.py` 中，为 `CRISPY_TEMPLATE_PACK` 添加一个新的设置 - 其值应该是 `bootstrap4`。这应该作为 `Dev` 类的一个属性添加：

    ```py
    CRISPY_TEMPLATE_PACK = 'bootstrap4'
    ```

    这让 `django-crispy-forms` 知道在渲染表单时应该使用为 Bootstrap 版本 4 设计的模板。您现在可以保存并关闭 `settings.py`。

1.  打开 `reviews` 应用的 `forms.py` 文件。首先，我们需要在文件顶部添加两个导入：从 `crispy_forms.helper` 导入 `FormHelper`，从 `crispy_forms.layout` 导入 `Submit`：

    ```py
    from crispy_forms.helper import FormHelper
    from crispy_forms.layout import Submit
    ```

1.  接下来，向 `SearchForm` 添加一个 `__init__` 方法。它应该接受 `*args` 和 `**kwargs` 作为参数，然后使用它们调用超类的 `__init__` 方法：

    ```py
    class SearchForm(forms.Form):
    …
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    ```

    这将简单地传递给超类构造函数提供的任何参数。

1.  仍然在 `__init__` 方法中，将 `self.helper` 设置为 `FormHelper` 的一个实例。然后设置 helper 的 `form_method` 为 `get`。最后，创建一个 `Submit` 的实例，将空字符串作为名称（第一个参数），将 `Search` 作为按钮标签（第二个参数）。使用 `add_input` 方法将此添加到 helper 中：

    ```py
    self.helper = FormHelper()
    self.helper.form_method = "get"
    self.helper.add_input(Submit("", "Search"))
    ```

    您现在可以保存并关闭 `forms.py`。

1.  在 `reviews` 应用程序的 `templates` 目录中，打开 `search-results.html` 文件。在文件开始处，在 `extends` 模板标签之后，使用 `load` 模板标签加载 `crispy_forms_tags`：

    ```py
    {% load crispy_forms_tags %}
    ```

1.  在模板中定位现有的 `<form>` 元素。它应该看起来像这样：

    ```py
    <form>
        {{ form.as_p }}
    <button type="submit" class="btn btn-primary">Search</button>
    </form>
    ```

    您可以删除输入的 `<form>` 元素，并用 `crispy` 模板标签替换它：

    ```py
    {% crispy form %}
    ```

    这将使用 `django-crispy-forms` 库渲染表单，包括 `<form>` 元素和提交按钮。在此更改后，模板的这一部分应该看起来像 *图 15.30*：

    ![图 15.30：将 <form> 替换为 crispy 表单渲染器后的 search-results.html](img/B15509_15_30.jpg)

    ![图 15.32：使用更新后的搜索表单进行搜索](img/B15509_15_30.jpg)

    图 15.30：将 <form> 替换为 crispy 表单渲染器后的 search-results.html

    您现在可以保存 `search-results.html`。

1.  如果尚未运行，请启动 Django 开发服务器并转到 `http://127.0.0.1:8000/book-search/`。您应该看到与 *图 15.31* 一样的图书搜索表单：![图 15.31：使用 django-crispy-forms 渲染的图书搜索表单](img/B15509_15_31.jpg)

    ![图 15.31：与之前相同的方式使用表单](img/B15509_15_31.jpg)

图 15.31：使用 django-crispy-forms 渲染的图书搜索表单

您应该能够以与之前相同的方式使用该表单（*图 15.32*）：

![图 15.32：使用更新后的搜索表单进行搜索](img/B15509_15_32.jpg)

![图 15.32：与之前相同的方式使用表单](img/B15509_15_32.jpg)

图 15.32：使用更新后的搜索表单进行搜索

尝试在您的网络浏览器中查看页面的源代码以查看渲染输出。您会看到`<form>`元素已使用`method="get"`属性渲染，正如我们在*步骤 5*中指定给`FormHelper`的。注意，`django-crispy-forms`没有插入 CSRF 令牌字段——它知道在通过`GET`提交的表单中不需要。

在这个练习中，我们使用`pip3`（Windows 上的`pip`）安装了`django-crispy-forms`，然后在`settings.py`中通过将其添加到`INSTALLED_APPS`并定义我们想要使用的`CRISPY_TEMPLATE_PACK`（在我们的情况下是`bootstrap4`）来配置它。然后，我们更新了`SearchForm`类，使用`FormHelper`实例来控制表单上的属性，并使用`Submit`类添加了一个提交按钮。最后，我们将`search-results.html`模板更改为使用`crispy`模板标签来渲染表单，这使得我们可以移除之前使用的`<form>`元素，并通过将所有与表单相关的代码移动到 Python 代码中（而不是部分在 HTML 中，部分在 Python 中）来简化表单生成。

## django-allauth

在浏览网站时，您可能已经看到了允许您使用其他网站的凭据登录的按钮。例如，使用您的 GitHub 登录：

![图 15.33：带有使用 Google 或 GitHub 登录选项的登录表单](img/B15509_15_33.jpg)

图 15.33：带有使用 Google 或 GitHub 登录选项的登录表单

在我们解释这个过程之前，让我们介绍我们将要使用的术语：

+   **请求网站**：用户试图登录的网站。

+   **身份验证提供者**：用户正在验证的第三方提供者（例如，Google、GitHub 等）。

+   **身份验证应用程序**：这是请求网站创建者在身份验证提供者处设置的东西。它决定了请求网站将拥有哪些权限。例如，请求应用程序可以获取您的 GitHub 用户名，但不会获得写入您仓库的权限。用户可以通过禁用对身份验证应用程序的访问来阻止请求网站访问身份验证提供者中的信息。

无论您选择哪种第三方登录选项，流程通常都是相同的。首先，您将被重定向到身份验证提供者网站，并要求允许身份验证应用程序访问您的账户（*图 15.34*）：

![图 15.34：身份验证提供者授权屏幕](img/B15509_15_34.jpg)

图 15.34：身份验证提供者授权屏幕

在您授权身份验证应用程序后，身份验证提供者将重定向回请求的网站。您将被重定向到的 URL 将包含一个秘密令牌，请求的网站可以使用该令牌在后台请求您的用户信息。这允许请求的网站通过直接与身份验证提供者通信来验证您的身份。在验证您的身份后，请求的网站可以将其重定向到您的页面。此流程在 *图 15.35* 中的序列图中表示：

![图 15.35：第三方身份验证流程![图片 B15509_15_35.jpg](img/B15509_15_35.jpg)

图 15.35：第三方身份验证流程

现在我们已经介绍了使用第三方服务进行身份验证，我们可以讨论 `django-allauth`。`django-allauth` 是一个应用程序，它可以将您的 Django 应用程序轻松地连接到第三方身份验证服务，包括 Google、GitHub、Facebook、Twitter 等。实际上，在撰写本文时，`django-allauth` 支持超过 75 个身份验证提供者。

当用户首次在您的网站上进行身份验证时，`django-allauth` 将为您创建一个标准的 Django `User` 实例。它还知道如何解析身份验证提供者在最终用户授权身份验证应用程序后加载的回调/重定向 URL。

`django-allauth` 为您的应用程序添加了三个模型：

+   `SocialApplication`：此模型存储用于识别您的身份验证应用程序的信息。您输入的信息将取决于提供者，提供者将为您提供 *客户端* ID、*密钥* 和（可选的）一个 *密钥*。请注意，这些是 `django-allauth` 用于这些值的名称，并且它们将根据提供者而有所不同。我们将在本节稍后提供一些这些值的示例。`SocialApplication` 是 `django-allauth` 模型中唯一一个您需要自己创建的，其他模型在用户进行身份验证时由 `django-allauth` 自动创建。

+   `SocialApplicationToken`：此模型包含用于将 Django 用户识别给身份验证提供者的值。它包含一个 *令牌* 和（可选的）一个 *令牌密钥*。它还包含一个引用创建它的 `SocialApplication` 和它应用的 `SocialAccount`。

+   `SocialAccount`：此模型将 Django 用户与提供者（例如，Google 或 GitHub）相关联，并存储提供者可能提供的额外信息。

由于存在如此多的身份验证提供者，我们不会涵盖如何设置它们的所有内容，但我们将提供简短的设置说明以及如何将提供者的身份验证令牌映射到 `SocialApplication` 中的正确字段。我们将为此章节中提到的两个身份验证提供者进行此操作：Google 和 GitHub。

### django-allauth 安装和设置

与本章中的其他应用程序一样，`django-allauth` 使用 `pip3` 进行安装：

```py
pip3 install django-allauth
```

注意

对于 Windows，您可以在前面的命令中使用 `pip` 而不是 `pip3`。

我们接下来需要做一些设置更改。`django-allauth` 需要运行 `django.contrib.sites` 应用，因此需要将其添加到 `INSTALLED_APPS`。然后需要添加一个新的设置来定义我们站点的 `SITE_ID`。我们可以在 `settings.py` 文件中将此设置为 `1`：

```py
INSTALLED_APPS = [# this entry added
                  'django.contrib.sites',\
                  'django.contrib.admin',\
                  'django.contrib.auth',\
                  # the rest of the values are truncated]
SITE_ID = 1
```

注意

有可能将单个 Django 项目托管在多个主机名下，并且使其在每个主机名上表现不同——但也可以在所有站点之间共享内容。我们不需要在我们的项目中其他任何地方使用 `SITE_ID`，但必须在此处设置。您可以在 [`docs.djangoproject.com/en/3.0/ref/contrib/sites/`](https://docs.djangoproject.com/en/3.0/ref/contrib/sites/) 中了解更多关于 `SITE_ID` 设置的信息。

我们还需要将 `allauth` 和 `allauth.socialaccount` 添加到 `INSTALLED_APPS`：

```py
INSTALLED_APPS = [# the rest of the values are truncated
                  'allauth',\
                  'allauth.socialaccount',]
```

然后，我们想要支持的每个提供者也必须添加到 `INSTALLED_APPS` 列表中；例如，考虑以下片段：

```py
INSTALLED_APPS = [# the rest of the values are truncated
                  'allauth.socialaccount.providers.github',\
                  'allauth.socialaccount.providers.google',]
```

在完成所有这些之后，我们需要运行 `migrate` 管理命令，以创建 `django-allauth` 模型：

```py
python3 manage.py migrate
```

一旦完成，您可以通过 Django 管理界面添加新的社交应用（*图 15.36*）：

![图 15.36：添加社交应用]

![img/B15509_15_36.jpg]

图 15.36：添加社交应用

要添加社交应用，选择一个 `Provider`（此列表将仅显示 `INSTALLED_APPS` 列表中的那些），输入一个名称（它可以与 `Provider` 相同），并输入提供者网站上的 `Client ID`（我们很快会详细介绍这一点）。您可能还需要 `Secret key` 和 `Key`。选择它应该应用的站点。（如果您只有一个 `Site` 实例，那么其名称无关紧要，只需选择它即可。站点名称可以在 Django 管理的 `Sites` 部分中更新。您也可以在那里添加更多站点。）

我们现在将查看我们三个示例提供者使用的令牌。

### GitHub 身份验证设置

您可以在您的 GitHub 个人资料下设置一个新的 GitHub 应用程序。在开发期间，您应用的回调 URL 应设置为 `http://127.0.0.1:8000/accounts/github/login/callback/`，并在您部署到生产时更新为真实的主机名。创建应用后，它将提供 `Client ID` 和 `Client Secret`。这些就是 `django-allauth` 中的 `Client id` 和 `Secret key`。

### Google 身份验证设置

创建 Google 应用程序是通过您的 Google 开发者控制台完成的。在开发期间，授权的重定向 URI 应设置为 `http://127.0.0.1:8000/accounts/google/login/callback/`，并在生产部署后更新。应用的 `Client ID` 也是在 `django-allauth` 中的 Client id，而应用的 `Client secret` 是 `Secret key`。

## 使用 django-allauth 启动身份验证

要通过第三方提供者启动身份验证，您首先需要在您的 URL 映射中添加 `django-allauth` 的 URL。在您的 `urlpatterns` 中的某个地方，有一个 `urls.py` 文件，包含 `allauth.urls`：

```py
urlpatterns = [path('allauth', include('allauth.urls')),]
```

然后，您可以使用类似`http://127.0.0.1:8000/allauth/github/login/?process=login`或`http://127.0.0.1:8000/allauth/google/login/?process=login`的 URL 启动登录，等等。`django-allauth`将为您处理所有重定向，并在用户返回网站时创建/验证 Django 用户。您可以在登录页面上添加带有“使用 GitHub 登录”或“使用 Google 登录”等文本的按钮，这些按钮链接到这些 URL。

### 其他 django-allauth 功能

除了使用第三方提供者进行身份验证之外，`django-allauth`还可以添加一些 Django 本身没有的实用功能。例如，您可以配置它要求用户提供一个电子邮件地址，并在用户登录之前通过点击他们收到的确认链接来验证他们的电子邮件地址，`django-allauth`还可以处理为用户生成一个通过电子邮件发送的密码重置 URL。您可以在[`django-allauth.readthedocs.io/en/stable/overview.html`](https://django-allauth.readthedocs.io/en/stable/overview.html)找到解释这些功能以及更多内容的`django-allauth`文档。

现在我们已经深入探讨了前四个第三方应用程序，并对`django-allauth`进行了简要概述，您可以为本章的活动承担任务。在这个活动中，您将重构我们正在使用的`ModelForm`实例，以使用`CrispyFormHelper`类。

## 活动十五.01：使用 FormHelper 更新表单

在这个活动中，我们将更新`ModelForm`实例（`PublisherForm`、`ReviewForm`和`BookMediaForm`）以使用`CrispyFormHelper`类。使用`FormHelper`，我们可以在`Form`类内部定义`Submit`按钮的文本。然后，我们可以将`<form>`渲染逻辑从`instance-form.html`模板中移除，并用`crispy`模板标签替换它。

这些步骤将帮助您完成活动：

1.  创建一个`InstanceForm`类，该类是`forms.ModelForm`的子类。这将是现有`ModelForm`类的基类。

1.  在`InstanceForm`的`__init__`方法中，为`self`设置一个`FormHelper`实例。

1.  向`FormHelper`添加一个`Submit`按钮。如果表单使用`instance`实例化，则按钮文本应为“保存”，否则应为“创建”。

1.  将 PublisherForm、ReviewForm 和 BookMediaForm 更新为从`InstanceForm`扩展。

1.  更新`instance-form.html`模板，以便使用`crispy`模板标签渲染`form`。其余的`<form>`可以删除。

1.  在`book_media`视图中，不再需要`is_file_upload`上下文项。

完成后，您应该会看到使用 Bootstrap 主题渲染的表单。*图 15.37*显示了“新出版商”页面：

![图 15.37：新出版商页面](img/B15509_15_37.jpg)

图 15.37：新出版商页面

*图 15.38*显示了“新评论”页面：

![图 15.38：新评论表单](img/B15509_15_38.jpg)

图 15.38：新评论表单

最后，图书媒体页面在*图 15.39*中显示：

![图 15.39：图书媒体页面![图片](img/B15509_15_39.jpg)

图 15.39：书籍媒体页面

您应该注意到表单仍然表现良好，并允许文件上传。`django-crispy-forms` 已自动将 `enctype="multipart/form-data"` 属性添加到 `<form>` 中。您可以通过查看页面源代码来验证这一点。

注意

本活动的解决方案可以在 [`packt.live/2Nh1NTJ`](http://packt.live/2Nh1NTJ) 找到。

# 摘要

在本章中，我们介绍了五个可以增强您网站功能的第三方 Django 应用程序。我们安装并设置了 `django-configurations`，这使得我们能够轻松地在不同的设置之间切换，并使用环境变量来更改它们。`dj-database-url` 也帮助处理设置，允许我们通过使用 URL 来进行数据库设置更改。我们看到了 Django 调试工具栏如何帮助我们了解我们的应用程序正在做什么，并帮助我们调试与之相关的问题。`django-crispy-forms` 不仅可以使用 Bootstrap CSS 渲染我们的表单，还允许我们通过将它们的行性行为定义为表单类本身的一部分来节省代码。我们简要地了解了 `django-allauth`，并看到了它如何集成到第三方身份验证提供商中。在本章的活动部分，我们将我们的 `ModelForm` 实例更新为使用 `django-crispy-forms` 的 `FormHelper`，并通过使用 `crispy` 模板标签从模板中移除一些逻辑。

在下一章中，我们将探讨如何将 React JavaScript 框架集成到 Django 应用程序中。
