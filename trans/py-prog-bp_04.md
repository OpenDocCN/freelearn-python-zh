# 汇率和货币转换工具

在上一章中，我们构建了一个非常酷的应用程序，用于在 Twitter 上计算投票，并学习了如何使用 Python 进行身份验证和消费 Twitter API。我们还对如何在 Python 中使用响应式扩展有了很好的介绍。在本章中，我们将创建一个终端工具，该工具将从`fixer.io`获取当天的汇率，并使用这些信息来在不同货币之间进行价值转换。

`Fixer.io`是由[`github.com/hakanensari`](https://github.com/hakanensari)创建的一个非常好的项目；它每天从欧洲央行获取外汇汇率数据。他创建的 API 使用起来简单，并且运行得很好。

我们的项目首先通过创建围绕 API 的框架来开始；当框架就位后，我们将创建一个终端应用程序，可以在其中执行货币转换。我们从`fixer.io`获取的所有数据都将存储在 MongoDB 数据库中，因此我们可以在不一直请求`fixer.io`的情况下执行转换。这将提高我们应用程序的性能。

在本章中，我们将涵盖以下内容：

+   如何使用`pipenv`来安装和管理项目的依赖项

+   使用 PyMongo 模块与 MongoDB 一起工作

+   使用 Requests 消费 REST API

说了这么多，让我们开始吧！

# 设置环境

像往常一样，我们将从设置环境开始；我们需要做的第一件事是设置一个虚拟环境，这将允许我们轻松安装项目依赖项，而不会干扰 Python 的全局安装。

在之前的章节中，我们使用`virtualenv`来创建我们的虚拟环境；然而，Kenneth Reitz（流行包*requests*的创建者）创建了`pipenv`。

`pipenv`对于 Python 来说就像 NPM 对于 Node.js 一样。但是，`pipenv`用于远不止包管理，它还为您创建和管理虚拟环境。在我看来，旧的开发工作流有很多优势，但对我来说，有两个方面很突出：第一个是您不再需要两种不同的工具（`pip`，`virtualenv`），第二个是在一个地方拥有所有这些强大功能变得更加简单。

我非常喜欢`pipenv`的另一点是使用`Pipfile`。有时，使用要求文件真的很困难。我们的生产环境和开发环境具有相同的依赖关系，您最终需要维护两个不同的文件；而且，每次需要删除一个依赖项时，您都需要手动编辑要求文件。

使用`pipenv`，您无需担心有多个要求文件。开发和生产依赖项都放在同一个文件中，`pipenv`还负责更新`Pipfile`。

安装`pipenv`非常简单，只需运行：

```py
pip install pipenv
```

安装后，您可以运行：

```py
pipenv --help
```

您应该看到以下输出：

![](img/b92b58f8-cc44-4f22-9c04-0e761751af85.png)

我们不会详细介绍所有不同的选项，因为这超出了本书的范围，但在创建环境时，您将掌握基础知识。

第一步是为我们的项目创建一个目录。让我们创建一个名为`currency_converter`的目录：

```py
mkdir currency_converter && cd currency_converter
```

现在您在`currency_converter`目录中，我们将使用`pipenv`来创建我们的虚拟环境。运行以下命令：

```py
pipenv --python python3.6
```

这将为当前目录中的项目创建一个虚拟环境，并使用 Python 3.6。`--python`选项还接受您安装 Python 的路径。在我的情况下，我总是下载 Python 源代码，构建它，并将其安装在不同的位置，因此这对我非常有用。

您还可以使用`--three`选项，它将使用系统上默认的 Python3 安装。运行命令后，您应该看到以下输出：

![](img/8784ad3b-7ce3-4d9e-bc95-e01455d19615.png)

如果你查看`Pipfile`的内容，你应该会看到类似以下的内容：

```py
[[source]]

url = "https://pypi.python.org/simple"
verify_ssl = true
name = "pypi"

[dev-packages]

[packages]

[requires]

python_version = "3.6"
```

这个文件开始定义从哪里获取包，而在这种情况下，它将从`pypi`下载包。然后，我们有一个地方用于项目的开发依赖项，在`packages`中是生产依赖项。最后，它说这个项目需要 Python 版本 3.6。

太棒了！现在你可以使用一些命令。例如，如果你想知道项目使用哪个虚拟环境，你可以运行`pipenv --venv`；你将看到以下输出：

![](img/f729d03d-b885-4866-a3f1-f69689ba4168.png)

如果你想为项目激活虚拟环境，你可以使用`shell`命令，如下所示：

![](img/17b5e4d5-1962-4054-afd5-50748321d710.png)

完美！有了虚拟环境，我们可以开始添加项目的依赖项。

我们要添加的第一个依赖是`requests`。

运行以下命令：

```py
pipenv install requests
```

我们将得到以下输出：

![](img/fef2a704-9e32-4cd2-b6e1-21d2a29d20a7.png)

正如你所看到的，`pipenv`安装了`requests`以及它的所有依赖项。

`pipenv`的作者是创建流行的 requests 库的同一个开发者。在安装输出中，你可以看到一个彩蛋，上面写着`PS: You have excellent taste!`。

我们需要添加到我们的项目中的另一个依赖是`pymongo`，这样我们就可以连接和操作 MongoDB 数据库中的数据。

运行以下命令：

```py
pipenv install pymongo
```

我们将得到以下输出：

![](img/cd80273e-1faa-4667-835c-54d65e0bfdd6.png)

让我们来看看`Pipfile`，看看它现在是什么样子：

```py
[[source]]

url = "https://pypi.python.org/simple"
verify_ssl = true
name = "pypi"

[dev-packages]

[packages]

requests = "*"
pymongo = "*"

[requires]

python_version = "3.6"
```

正如你所看到的，在`packages`文件夹下，我们现在有了两个依赖项。

与使用`pip`安装包相比，没有太多改变。唯一的例外是现在安装和移除依赖项将自动更新`Pipfile`。

另一个非常有用的命令是`graph`命令。运行以下命令：

```py
pipenv graph
```

我们将得到以下输出：

![](img/4b513feb-d291-4c36-abce-6d7272159f16.png)

正如你所看到的，`graph`命令在你想知道你安装的包的依赖关系时非常有帮助。在我们的项目中，我们可以看到`pymongo`没有任何额外的依赖项。然而，`requests`有四个依赖项：`certifi`、`chardet`、`idna`和`urllib3`。

现在你已经对`pipenv`有了很好的介绍，让我们来看看这个项目的结构会是什么样子：

```py
currency_converter
└── currency_converter
    ├── config
    ├── core   
```

`currency_converter`的顶层是应用程序的`root`目录。然后，我们再往下一级，有另一个`currency_converter`，那就是我们将要创建的`currency_converter`模块。

在`currency_converter`模块目录中，我们有一个核心，其中包含应用程序的核心功能，例如命令行参数解析器，处理数据的辅助函数等。

我们还配置了，与其他项目一样，哪个项目将包含读取 YAML 配置文件的函数；最后，我们有 HTTP，其中包含所有将执行 HTTP 请求到`fixer.io` REST API 的函数。

现在我们已经学会了如何使用`pipenv`以及它如何帮助我们提高生产力，我们可以安装项目的初始依赖项。我们也创建了项目的目录结构。拼图的唯一缺失部分就是安装 MongoDB。

我正在使用 Linux Debian 9，我可以很容易地使用 Debian 的软件包管理工具来安装它：

```py
sudo apt install mongodb
```

你会在大多数流行的 Linux 发行版的软件包存储库中找到 MongoDB，如果你使用 Windows 或 macOS，你可以在以下链接中看到说明：

对于 macOS：[`docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/`](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/)

对于 Windows：[`docs.mongodb.com/manual/tutorial/install-mongodb-on-windows/`](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-windows/)

安装完成后，您可以使用 MongoDB 客户端验证一切是否正常工作。打开终端，然后运行`mongo`命令。

然后你应该进入 MongoDB shell：

```py
MongoDB shell version: 3.2.11
connecting to: test
```

要退出 MongoDB shell，只需键入*CTRL *+ *D.*

太棒了！现在我们准备开始编码！

# 创建 API 包装器

在这一部分，我们将创建一组函数，这些函数将包装`fixer.io` API，并帮助我们在项目中以简单的方式使用它。

让我们继续在`currency_converter/currency_converter/core`目录中创建一个名为`request.py`的新文件。首先，我们将包括一些`import`语句：

```py
import requests
from http import HTTPStatus
import json
```

显然，我们需要`requests`，以便我们可以向`fixer.io`端点发出请求，并且我们还从 HTTP 模块导入`HTTPStatus`，以便我们可以返回正确的 HTTP 状态码；在我们的代码中也更加详细。在代码中，`HTTPStatus.OK`的返回要比只有`200`更加清晰和易读。

最后，我们导入`json`包，以便我们可以将从`fixer.io`获取的 JSON 内容解析为 Python 对象。

接下来，我们将添加我们的第一个函数。这个函数将返回特定货币的当前汇率：

```py
def fetch_exchange_rates_by_currency(currency):
    response = requests.get(f'https://api.fixer.io/latest?base=
                            {currency}')

    if response.status_code == HTTPStatus.OK:
        return json.loads(response.text)
    elif response.status_code == HTTPStatus.NOT_FOUND:
        raise ValueError(f'Could not find the exchange rates for: 
                         {currency}.')
    elif response.status_code == HTTPStatus.BAD_REQUEST:
        raise ValueError(f'Invalid base currency value: {currency}')
    else:
        raise Exception((f'Something went wrong and we were unable 
                         to fetch'
                         f' the exchange rates for: {currency}'))
```

这个函数以货币作为参数，并通过向`fixer.io` API 发送请求来获取使用该货币作为基础的最新汇率信息，这是作为参数给出的。

如果响应是`HTTPStatus.OK`（`200`），我们使用 JSON 模块的 load 函数来解析 JSON 响应；否则，我们根据发生的错误引发异常。

我们还可以在`currency_converter/currency_converter/core`目录中创建一个名为`__init__.py`的文件，并导入我们刚刚创建的函数：

```py
from .request import fetch_exchange_rates_by_currency
```

太好了！让我们在 Python REPL 中试一下：

```py
Python 3.6.3 (default, Nov 21 2017, 06:53:07)
[GCC 6.3.0 20170516] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from currency_converter.core import fetch_exchange_rates_by_currency
>>> from pprint import pprint as pp
>>> exchange_rates = fetch_exchange_rates_by_currency('BRL')
>>> pp(exchange_rates)
{'base': 'BRL',
 'date': '2017-12-06',
 'rates': {'AUD': 0.40754,
 'BGN': 0.51208,
 'CAD': 0.39177,
 'CHF': 0.30576,
 'CNY': 2.0467,
 'CZK': 6.7122,
 'DKK': 1.9486,
 'EUR': 0.26183,
 'GBP': 0.23129,
 'HKD': 2.4173,
 'HRK': 1.9758,
 'HUF': 82.332,
 'IDR': 4191.1,
 'ILS': 1.0871,
 'INR': 19.963,
 'JPY': 34.697,
 'KRW': 338.15,
 'MXN': 5.8134,
 'MYR': 1.261,
 'NOK': 2.5548,
 'NZD': 0.4488,
 'PHP': 15.681,
 'PLN': 1.1034,
 'RON': 1.2128,
 'RUB': 18.273,
 'SEK': 2.599,
 'SGD': 0.41696,
 'THB': 10.096,
 'TRY': 1.191,
 'USD': 0.3094,
 'ZAR': 4.1853}}
```

太棒了！它的工作方式正如我们所期望的那样。

接下来，我们将开始构建数据库辅助类。

# 添加数据库辅助类

现在我们已经实现了从`fixer.io`获取汇率信息的函数，我们需要添加一个类，该类将检索并保存我们获取的信息到我们的 MongoDB 中。

那么，让我们继续在`currency_converter/currency_converter/core`目录中创建一个名为`db.py`的文件；让我们添加一些`import`语句：

```py
  from pymongo import MongoClient
```

我们唯一需要`import`的是`MongoClient`。`MongoClient`将负责与我们的数据库实例建立连接。

现在，我们需要添加`DbClient`类。这个类的想法是作为`pymongo`包函数的包装器，并提供一组更简单的函数，抽象出一些在使用`pymongo`时重复的样板代码。

```py
class DbClient:

    def __init__(self, db_name, default_collection):
        self._db_name = db_name
        self._default_collection = default_collection
        self._db = None
```

一个名为`DbClient`的类，它的构造函数有两个参数，`db_name`和`default_collection`。请注意，在 MongoDB 中，我们不需要在使用之前创建数据库和集合。当我们第一次尝试插入数据时，数据库和集合将被自动创建。

如果您习惯于使用 MySQL 或 MSSQL 等 SQL 数据库，这可能看起来有些奇怪，在那里您必须连接到服务器实例，创建数据库，并在使用之前创建所有表。

在这个例子中，我们不关心安全性，因为 MongoDB 超出了本书的范围，我们只关注 Python。

然后，我们将向数据库添加两个方法，`connect`和`disconnect`：

```py
    def connect(self):
        self._client = MongoClient('mongodb://127.0.0.1:27017/')
        self._db = self._client.get_database(self._db_name)

    def disconnect(self):
        self._client.close()
```

`connect`方法将使用`MongoClient`连接到我们的本地主机上的数据库实例，使用端口`27017`，这是 MongoDB 安装后默认运行的端口。这两个值可能在您的环境中有所不同。`disconnect`方法只是调用客户端的 close 方法，并且，顾名思义，它关闭连接。

现在，我们将添加两个特殊函数，`__enter__`和`__exit__`：

```py
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        self.disconnect()

        if exec_type:
            raise exec_type(exec_value)

        return self
```

我们希望`DbClient`类在其自己的上下文中使用，并且这是通过使用上下文管理器和`with`语句来实现的。上下文管理器的基本实现是通过实现这两个函数`__enter__`和`__exit__`。当我们进入`DbClient`正在运行的上下文时，将调用`__enter__`。在这种情况下，我们将调用`connect`方法来连接到我们的 MongoDB 实例。

另一方面，`__exit__`方法在当前上下文终止时被调用。上下文可以由正常原因或抛出的异常终止。在我们的情况下，我们从数据库断开连接，如果`exec_type`不等于`None`，这意味着如果发生了异常，我们会引发该异常。这是必要的，否则在`DbClient`上下文中发生的异常将被抑制。

现在，我们将添加一个名为`_get_collection`的私有方法：

```py
    def _get_collection(self):
        if self._default_collection is None:
            raise AttributeError('collection argument is required')

        return self._db[self._default_collection]
```

这个方法将简单地检查我们是否定义了`default_collection`。如果没有，它将抛出一个异常；否则，我们返回集合。

我们只需要两个方法来完成这个类，一个是在数据库中查找项目，另一个是插入或更新数据：

```py
    def find_one(self, filter=None):
        collection = self._get_collection()
        return collection.find_one(filter)

    def update(self, filter, document, upsert=True):
        collection = self._get_collection()

        collection.find_one_and_update(
            filter,
            {'$set': document},
            upsert=upsert)
```

`find_one`方法有一个可选参数叫做 filter，它是一个带有条件的字典，将用于执行搜索。如果省略，它将只返回集合中的第一项。

在 update 方法中还有一些其他事情。它有三个参数：`filter`，`document`，以及可选参数`upsert`。

`filter`参数与`find_one`方法完全相同；它是一个用于搜索我们想要更新的集合项的条件。

`document`参数是一个包含我们想要在集合项中更新或插入的字段的字典。

最后，可选参数`upsert`，当设置为`True`时，意味着如果我们要更新的项目在数据库的集合中不存在，那么我们将执行插入操作并将项目添加到集合中。

该方法首先获取默认集合，然后使用集合的`find_on_and_update`方法，将`filter`传递给包含我们要更新的字段的字典，还有`upsert`选项。

我们还需要使用以下内容更新`currency_converter/currency_converter/core`目录中的`__init__.py`文件：

```py
from .db import DbClient
```

太好了！现在，我们可以开始创建命令行解析器了。

# 创建命令行解析器

我必须坦白一件事：我是一个命令行类型的人。是的，我知道有些人认为它已经过时了，但我喜欢在终端上工作。我绝对更有生产力，如果你使用 Linux 或 macOS，你可以结合工具来获得你想要的结果。这就是我们要为这个项目添加命令行解析器的原因。

我们需要实现一些东西才能开始创建命令行解析器。我们要添加的一个功能是设置默认货币的可能性，这将避免我们的应用用户总是需要指定基础货币来执行货币转换。

为了做到这一点，我们将创建一个动作，我们已经在第一章中看到了动作是如何工作的，*实现天气应用程序*，但是为了提醒我们，动作是可以绑定到命令行参数以执行某个任务的类。当命令行中使用参数时，这些动作会自动调用。

在进行自定义操作的开发之前，我们需要创建一个函数，从数据库中获取我们应用程序的配置。首先，我们将创建一个自定义异常，用于在无法从数据库中检索配置时引发错误。在`currency_converter/currency_converter/config`目录中创建一个名为`config_error.py`的文件，内容如下：

```py
    class ConfigError(Exception):
      pass
```

完美！这就是我们创建自定义异常所需要的全部内容。我们本可以使用内置异常，但那对我们的应用程序来说太具体了。为您的应用程序创建自定义异常总是一个很好的做法；当排除错误时，它将使您和您的同事的生活变得更加轻松。

在`currency_converter/currency_converter/config/`目录中创建一个名为`config.py`的文件，内容如下：

```py
from .config_error import ConfigError
from currency_converter.core import DbClient

def get_config():
    config = None

    with DbClient('exchange_rates', 'config') as db:
        config = db.find_one()

    if config is None:
        error_message = ('It was not possible to get your base 
                        currency, that '
                       'probably happened because it have not been '
                         'set yet.\n Please, use the option '
                         '--setbasecurrency')
        raise ConfigError(error_message)

    return config
```

在这里，我们首先从`import`语句开始。我们开始导入我们刚刚创建的`ConfigError`自定义异常，还导入`DbClient`类，以便我们可以访问数据库来检索应用程序的配置。

然后，我们定义了`get_config`函数。这个函数不会接受任何参数，函数首先定义了一个值为`None`的变量 config。然后，我们使用`DbClient`连接到`exchange_rate`数据库，并使用名为`config`的集合。在`DbClient`上下文中，我们使用`find_one`方法，没有任何参数，这意味着将返回该配置集合中的第一项。

如果`config`变量仍然是`None`，我们会引发一个异常，告诉用户数据库中还没有配置，需要再次运行应用程序并使用`--setbasecurrency`参数。我们将很快实现命令行参数。如果我们有配置的值，我们只需返回它。

我们还需要在`currency_converter/currency_converter/config`目录中创建一个`__init__.py`文件，内容如下：

```py
from .config import get_config
```

现在，让我们开始添加我们的第一个操作，它将设置默认货币。在`currency_converter/currency_converter/core`目录中添加一个名为`actions.py`的文件：

```py
  import sys
  from argparse import Action
  from datetime import datetime

  from .db import DbClient
  from .request import fetch_exchange_rates_by_currency
  from currency_converter.config import get_config
```

首先，我们导入`sys`，这样我们就可以在程序出现问题时终止执行。然后，我们从`argparse`模块中导入`Action`。在创建自定义操作时，我们需要从`Action`继承一个类。我们还导入`datetime`，因为我们将添加功能来检查我们将要使用的汇率是否过时。

然后，我们导入了一些我们创建的类和函数。我们首先导入`DbClient`，这样我们就可以从 MongoDB 中获取和存储数据，然后导入`fetch_exchange_rates_by_currency`以在必要时从`fixer.io`获取最新数据。最后，我们导入一个名为`get_config`的辅助函数，这样我们就可以从数据库的配置集合中获取默认货币。

让我们首先添加`SetBaseCurrency`类：

```py
class SetBaseCurrency(Action):
    def __init__(self, option_strings, dest, args=None, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
```

在这里，我们定义了`SetBaseCurrency`类，继承自`Action`，并添加了一个构造函数。它并没有做太多事情；它只是调用了基类的构造函数。

现在，我们需要实现一个特殊的方法叫做`__call__`。当解析绑定到操作的参数时，它将被调用：

```py
    def __call__(self, parser, namespace, value, option_string=None):
        self.dest = value

        try:
            with DbClient('exchange_rates', 'config') as db:
                db.update(
                    {'base_currency': {'$ne': None}},
                    {'base_currency': value})

            print(f'Base currency set to {value}')
        except Exception as e:
            print(e)
        finally:
            sys.exit(0)
```

这个方法有四个参数，解析器是我们即将创建的`ArgumentParser`的一个实例。`namespace`是参数解析器的结果的对象；我们在第一章中详细介绍了命名空间对象，*实现天气应用程序*。值是传递给基础参数的值，最后，`option_string`是操作绑定到的参数。

我们通过为参数设置值、目标变量和创建`DbClient`的实例来开始该方法。请注意，我们在这里使用`with`语句，因此我们在`DbClient`上下文中运行更新。

然后，我们调用`update`方法。在这里，我们向`update`方法传递了两个参数，第一个是`filter`。当我们有`{'base_currrency': {'$ne': None}}`时，这意味着我们将更新集合中基础货币不等于 None 的项目；否则，我们将插入一个新项目。这是`DbClient`类中`update`方法的默认行为，因为我们默认将`upsert`选项设置为`True`。

当我们完成更新时，我们向用户打印消息，说明默认货币已设置，并且当我们触发`finally`子句时，我们退出代码的执行。如果出现问题，由于某种原因，我们无法更新`config`集合，将显示错误并退出程序。

我们需要创建的另一个类是`UpdateForeignerExchangeRates`类：

```py
class UpdateForeignerExchangeRates(Action):
    def __init__(self, option_strings, dest, args=None, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
```

与之前的类一样，我们定义类并从`Action`继承。构造函数只调用基类中的构造函数：

```py
def __call__(self, parser, namespace, value, option_string=None):

        setattr(namespace, self.dest, True)

        try:
            config = get_config()
            base_currency = config['base_currency']
            print(('Fetching exchange rates from fixer.io'
                   f' [base currency: {base_currency}]'))
            response = 
            fetch_exchange_rates_by_currency(base_currency)
            response['date'] = datetime.utcnow()

            with DbClient('exchange_rates', 'rates') as db:
                db.update(
                    {'base': base_currency},
                    response)
        except Exception as e:
            print(e)
        finally:
            sys.exit(0)
```

我们还需要实现`__call__`方法，当使用此操作绑定到的参数时将调用该方法。我们不会再次讨论方法参数，因为它与前一个方法完全相同。

该方法开始时将目标属性的值设置为`True`。我们将用于运行此操作的参数不需要参数，并且默认为`False`，因此如果我们使用参数，我们将其设置为`True`。这只是一种表明我们已经使用了该参数的方式。

然后，我们从数据库中获取配置并获取`base_currency`。我们向用户显示一条消息，告诉他们我们正在从`fixer.io`获取数据，然后我们使用我们的`fetch_exchange_rates_by_currency`函数，将`base_currency`传递给它。当我们得到响应时，我们将日期更改为 UTC 时间，这样我们就可以更容易地计算给定货币的汇率是否需要更新。

请记住，`fixer.io`在中欧时间下午 4 点左右更新其数据。

然后，我们创建`DbClient`的另一个实例，并使用带有两个参数的`update`方法。第一个是`filter`，因此它将更改与条件匹配的集合中的任何项目，第二个参数是我们从`fixer.io` API 获取的响应。

在所有事情都完成之后，我们触发`finally`子句并终止程序的执行。如果出现问题，我们会在终端向用户显示一条消息，并终止程序的执行。

# 创建货币枚举

在开始命令行解析器之前，我们还需要创建一个枚举，其中包含我们的应用程序用户可以选择的可能货币。让我们继续在`currency_converter/currency_converter/core`目录中创建一个名为`currency.py`的文件，其中包含以下内容：

```py
from enum import Enum

class Currency(Enum):
    AUD = 'Australia Dollar'
    BGN = 'Bulgaria Lev'
    BRL = 'Brazil Real'
    CAD = 'Canada Dollar'
    CHF = 'Switzerland Franc'
    CNY = 'China Yuan/Renminbi'
    CZK = 'Czech Koruna'
    DKK = 'Denmark Krone'
    GBP = 'Great Britain Pound'
    HKD = 'Hong Kong Dollar'
    HRK = 'Croatia Kuna'
    HUF = 'Hungary Forint'
    IDR = 'Indonesia Rupiah'
    ILS = 'Israel New Shekel'
    INR = 'India Rupee'
    JPY = 'Japan Yen'
    KRW = 'South Korea Won'
    MXN = 'Mexico Peso'
    MYR = 'Malaysia Ringgit'
    NOK = 'Norway Kroner'
    NZD = 'New Zealand Dollar'
    PHP = 'Philippines Peso'
    PLN = 'Poland Zloty'
    RON = 'Romania New Lei'
    RUB = 'Russia Rouble'
    SEK = 'Sweden Krona'
    SGD = 'Singapore Dollar'
    THB = 'Thailand Baht'
    TRY = 'Turkish New Lira'
    USD = 'USA Dollar'
    ZAR = 'South Africa Rand'
    EUR = 'Euro'
```

这非常简单。我们已经在之前的章节中介绍了 Python 中的枚举，但在这里，我们定义了枚举，其中键是货币的缩写，值是名称。这与`fixer.io`中可用的货币相匹配。

打开`currency_converter/currency_converter/core`目录中的`__init__.py`文件，并添加以下导入语句：

```py
from .currency import Currency
```

# 创建命令行解析器

完美！现在，我们已经准备好创建命令行解析器。让我们继续在`currency_converter/currency_converter/core`目录中创建一个名为`cmdline_parser.py`的文件，然后像往常一样，让我们开始导入我们需要的一切：

```py
import sys
from argparse import ArgumentParser

from .actions import UpdateForeignerExchangeRates
from .actions import SetBaseCurrency
from .currency import Currency
```

从顶部开始，我们导入`sys`，这样如果出现问题，我们可以退出程序。我们还包括`ArgumentParser`，这样我们就可以创建解析器；我们还导入了我们刚刚创建的`UpdateforeignerExchangeRates`和`SetBaseCurrency`动作。在`Currency`枚举中的最后一件事是，我们将使用它来在解析器中的某些参数中设置有效的选择。

创建一个名为`parse_commandline_args`的函数：

```py
def parse_commandline_args():

    currency_options = [currency.name for currency in Currency]

    argparser = ArgumentParser(
        prog='currency_converter',
        description=('Tool that shows exchange rated and perform '
                     'currency convertion, using http://fixer.io 
                       data.'))
```

这里我们要做的第一件事是只获取`Currency`枚举键的名称；这将返回一个类似这样的列表：

![](img/c9385f16-58d4-4ec9-94db-6905c05a5be3.png)

在这里，我们最终创建了`ArgumentParser`的一个实例，并传递了两个参数：`prog`，这是程序的名称，我们可以称之为`currency_converter`，第二个是`description`（当在命令行中传递`help`参数时，将显示给用户的描述）。

这是我们要在`--setbasecurrency`中添加的第一个参数：

```py
argparser.add_argument('--setbasecurrency',
                           type=str,
                           dest='base_currency',
                           choices=currency_options,
                           action=SetBaseCurrency,
                           help='Sets the base currency to be 
                           used.')
```

我们定义的第一个参数是`--setbasecurrency`。它将把货币存储在数据库中，这样我们就不需要在命令行中一直指定基础货币。我们指定这个参数将被存储为一个字符串，并且用户输入的值将被存储在一个名为`base_currency`的属性中。

我们还将参数选择设置为我们在前面的代码中定义的`currency_options`。这将确保我们只能传递与`Currency`枚举匹配的货币。

`action`指定了当使用此参数时将执行哪个动作，我们将其设置为我们在`actions.py`文件中定义的`SetBaseCurrency`自定义动作。最后一个选项`help`是在显示应用程序帮助时显示的文本。

让我们添加`--update`参数：

```py
 argparser.add_argument('--update',
                           metavar='',
                           dest='update',
                           nargs=0,
                           action=UpdateForeignerExchangeRates,
                           help=('Update the foreigner exchange 
                                  rates '
                                 'using as a reference the base  
                                  currency'))
```

`--update`参数，顾名思义，将更新默认货币的汇率。它在`--setbasecurrency`参数之后使用。

在这里，我们使用名称`--update`定义参数，然后设置`metavar`参数。当生成帮助时，`metavar`关键字`--update`将被引用。默认情况下，它与参数的名称相同，但是大写。由于我们没有任何需要传递给此参数的值，我们将`metavar`设置为无。下一个参数是`nargs`，它告诉`argparser`这个参数不需要传递值。最后，我们设置`action`为我们之前创建的另一个自定义动作，即`UpdateForeignExchangeRates`动作。最后一个参数是`help`，它指定了参数的帮助文本。

下一个参数是`--basecurrency`参数：

```py
argparser.add_argument('--basecurrency',
                           type=str,
                           dest='from_currency',
                           choices=currency_options,
                           help=('The base currency. If specified it 
                                  will '
                                 'override the default currency set 
                                  by'
                                 'the --setbasecurrency option'))
```

这个参数的想法是，我们希望允许用户在请求货币转换时覆盖他们使用`--setbasecurrency`参数设置的默认货币。

在这里，我们使用名称`--basecurrency`定义参数。使用`string`类型，我们将把传递给参数的值存储在一个名为`from_currency`的属性中；我们还在这里将选择设置为`currency_option`，这样我们就可以确保只有在`Currency`枚举中存在的货币才被允许。最后，我们设置了帮助文本。

我们要添加的下一个参数称为`--value`。这个参数将接收我们的应用程序用户想要转换为另一种货币的值。

这是我们将如何编写它的方式：

```py
argparser.add_argument('--value',
                           type=float,
                           dest='value',
                           help='The value to be converted')
```

在这里，我们将参数的名称设置为`--value`。请注意，类型与我们之前定义的参数不同。现在，我们将接收一个浮点值，并且参数解析器将把传递给`--value`参数的值存储到名为 value 的属性中。最后一个参数是`help`文本。

最后，我们要添加的最后一个参数是指定值将被转换为哪种货币的参数，将被称为`--to`：

```py
   argparser.add_argument('--to',
                           type=str,
                           dest='dest_currency',
                           choices=currency_options,
                           help=('Specify the currency that the value 
                                  will '
                                 'be converted to.'))
```

这个参数与我们在前面的代码中定义的`--basecurrency`参数非常相似。在这里，我们将参数的名称设置为`--to`，它将是`string`类型。传递给此参数的值将存储在名为`dest_currency`的属性中。在这里，我们还将参数的选择设置为我们从`Currency`枚举中提取的有效货币列表；最后，我们设置帮助文本。

# 基本验证

请注意，我们定义的许多参数是必需的。然而，有一些参数是相互依赖的，例如参数`--value`和`--to`。您不能尝试转换价值而不指定要转换的货币，反之亦然。

这里的另一个问题是，由于许多参数是必需的，如果我们在不传递任何参数的情况下运行应用程序，它将接受并崩溃；在这里应该做的正确的事情是，如果用户没有使用任何参数，我们应该显示帮助菜单。也就是说，我们需要添加一个函数来执行这种类型的验证，所以让我们继续添加一个名为`validate_args`的函数。您可以在`import`语句之后的顶部添加此函数：

```py
def validate_args(args):

    fields = [arg for arg in vars(args).items() if arg]

    if not fields:
        return False

    if args.value and not args.dest_currency:
        return False
    elif args.dest_currency and not args.value:
        return False

    return True
```

因此，`args`将被传递给这个函数。`args`实际上是`time`和`namespace`的对象。这个对象将包含与我们在参数定义中指定的相同名称的属性。在我们的情况下，`namespace`将包含这些属性：`base_currency`、`update`、`from_currency`、`value`和`dest_currency`。

我们使用一个理解来获取所有未设置为`None`的字段。在这个理解中，我们使用内置函数`vars`，它将返回`args`的`__dict__`属性的值，这是`Namespace`对象的一个实例。然后，我们使用`.items()`函数，这样我们就可以遍历字典项，并逐一测试其值是否为`None`。

如果在命令行中传递了任何参数，那么这个理解的结果将是一个空列表，在这种情况下，我们返回`False`。

然后，我们测试需要成对使用的参数：`--value`（value）和`--to`（`dest_currency`）。如果我们有一个值，但`dest_currency`等于`None`，反之亦然，它将返回`False`。

现在，我们可以完成`parse_commandline_args`。让我们转到此函数的末尾，并添加以下代码：

```py
      args = argparser.parse_args()

      if not validate_args(args):
          argparser.print_help()
          sys.exit()

      return args
```

在这里，我们解析参数并将它们设置为变量`args`，请记住`args`将是`namespace`类型。然后，我们将`args`传递给我们刚刚创建的函数，即`validate_args`函数。如果`validate_args`返回`False`，它将打印帮助信息并终止程序的执行；否则，它将返回`args`。

接下来，我们将开发应用程序的入口点，它将把我们到目前为止开发的所有部分粘合在一起。

# 添加应用程序的入口点

这是本章我们一直在等待的部分；我们将创建应用程序的入口点，并将迄今为止编写的所有代码粘合在一起。

让我们在`currency_converter/currency_converter`目录中创建一个名为`__main__.py`的文件。我们之前在第一章中已经使用过`__main__`文件，*实现天气应用程序*。当我们在模块的`root`目录中放置一个名为`__main__.py`的文件时，这意味着该文件是模块的入口脚本。因此，如果我们运行以下命令：

```py
python -m currency_converter 
```

这与运行以下命令相同：

```py
python currency_converter/__main__.py
```

太好了！让我们开始向这个文件添加内容。首先，添加一些`import`语句：

```py
import sys

from .core.cmdline_parser import parse_commandline_args
from .config import get_config
from .core import DbClient
from .core import fetch_exchange_rates_by_currency
```

我们像往常一样导入`sys`包，以防需要调用 exit 来终止代码的执行，然后导入到目前为止我们开发的所有类和实用函数。我们首先导入`parse_commandline_args`函数进行命令行解析，然后导入`get_config`以便我们可以获取用户设置的默认货币，导入`DbClient`类以便我们可以访问数据库并获取汇率；最后，我们还导入`fetch_exchange_rates_by_currency`函数，当我们选择尚未在我们的数据库中的货币时将使用它。我们将从`fixer.io` API 中获取这个。

现在，我们可以创建`main`函数：

```py
def main():
    args = parse_commandline_args()
    value = args.value
    dest_currency = args.dest_currency
    from_currency = args.from_currency

    config = get_config()
    base_currency = (from_currency
                     if from_currency
                     else config['base_currency'])
```

`main`函数首先通过解析命令行参数来开始。如果用户输入的一切都正确，我们应该收到一个包含所有参数及其值的`namespace`对象。在这个阶段，我们只关心三个参数：`value`，`dest_currency`和`from_currency`。如果你还记得之前的话，`value`是用户想要转换为另一种货币的值，`dest_currency`是用户想要转换为的货币，`from_currency`只有在用户希望覆盖数据库中设置的默认货币时才会传递。

获取所有这些值后，我们调用`get_config`从数据库中获取`base_currency`，然后立即检查是否有`from_currency`可以使用该值；否则，我们使用数据库中的`base_currency`。这将确保如果用户指定了`from_currency`值，那么该值将覆盖数据库中存储的默认货币。

接下来，我们实现将实际从数据库或`fixer.io` API 获取汇率的代码，如下所示：

```py
    with DbClient('exchange_rates', 'rates') as db:
        exchange_rates = db.find_one({'base': base_currency})

        if exchange_rates is None:
            print(('Fetching exchange rates from fixer.io'
                   f' [base currency: {base_currency}]'))

            try:
                response = 
                fetch_exchange_rates_by_currency(base_currency)
            except Exception as e:
                sys.exit(f'Error: {e}')

            dest_rate = response['rates'][dest_currency]
            db.update({'base': base_currency}, response)
        else:
            dest_rate = exchange_rates['rates'][dest_currency]

        total = round(dest_rate * value, 2)
        print(f'{value} {base_currency} = {total} {dest_currency}')
```

我们使用`DbClient`类创建与数据库的连接，并指定我们将访问汇率集合。在上下文中，我们首先尝试找到基础货币的汇率。如果它不在数据库中，我们尝试从`fixer.io`获取它。

之后，我们提取我们要转换为的货币的汇率值，并将结果插入数据库，这样，下次运行程序并想要使用这种货币作为基础货币时，我们就不需要再次发送请求到`fixer.io`。

如果我们找到了基础货币的汇率，我们只需获取该值并将其分配给`dest_rate`变量。

我们要做的最后一件事是执行转换，并使用内置的 round 函数将小数点后的位数限制为两位，并在终端中打印值。

在文件末尾，在`main()`函数之后，添加以下代码：

```py
if __name__ == '__main__':
    main()
```

我们都完成了！

# 测试我们的应用程序

让我们测试一下我们的应用程序。首先，我们将显示帮助消息，看看我们有哪些选项可用：

![](img/b0f7d3ce-7807-4396-9065-ce200aabd67b.png)

很好！正如预期的那样。现在，我们可以使用`--setbasecurrency`参数来设置基础货币：

![](img/ad6fb0bd-bcf1-49eb-a325-f9fc65accb57.png)

在这里，我已将基础货币设置为 SEK（瑞典克朗），每次我需要进行货币转换时，我都不需要指定我的基础货币是 SEK。让我们将 100 SEK 转换为 USD（美元）：

![](img/38bfab21-ab5d-4b56-b505-d51ebb2957ee.png)

正如你所看到的，我们在数据库中没有该货币的汇率，所以应用程序的第一件事就是从`fixer.io`获取并将其保存到数据库中。

由于我是一名居住在瑞典的巴西开发人员，我想将 SEK 转换为 BRL（巴西雷亚尔），这样我就知道下次去巴西看父母时需要带多少瑞典克朗：

![](img/4666b921-54a0-4ed8-914d-96324ba2d6d6.png)

请注意，由于这是我们第二次运行应用程序，我们已经有了以 SEK 为基础货币的汇率，所以应用程序不会再次从`fixer.io`获取数据。

现在，我们要尝试的最后一件事是覆盖基础货币。目前，它被设置为 SEK。我们使用 MXN（墨西哥比索）并从 MXN 转换为 SEK：

![](img/30e9fa97-c5a3-4c03-be9f-cf743813f5de.png)

# 总结

在本章中，我们涵盖了许多有趣的主题。在设置应用程序环境时，您学会了如何使用超级新的、流行的工具`pipenv`，它已成为[python.org](https://www.python.org/)推荐的用于创建虚拟环境和管理项目依赖项的工具。

您还学会了面向对象编程的基本概念，如何为命令行工具创建自定义操作，Python 语言中关于上下文管理器的基础知识，如何在 Python 中创建枚举，以及如何使用`Requests`执行 HTTP 请求，这是 Python 生态系统中最受欢迎的包之一。

最后但并非最不重要的是，您学会了如何使用`pymongo`包在 MongoDB 数据库中插入、更新和搜索数据。

在下一章中，我们将转变方向，使用出色且非常流行的 Django web 框架开发一个完整、非常实用的网络应用程序！
