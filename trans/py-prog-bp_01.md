# 第一章：实现天气应用程序

本书中的第一个应用程序将是一个网络爬虫应用程序，它将从[`weather.com`](https://weather.com)爬取天气预报信息并在终端中呈现。我们将添加一些选项，可以将其作为应用程序的参数传递，例如：

+   温度单位（摄氏度或华氏度）

+   您可以获取天气预报的地区

+   用户可以在我们的应用程序中选择当前预报、五天预报、十天预报和周末的输出选项

+   补充输出的方式，例如风和湿度等额外信息

除了上述参数之外，此应用程序将被设计为可扩展的，这意味着我们可以为不同的网站创建解析器来获取天气预报，并且这些解析器将作为参数选项可用。

在本章中，您将学习如何：

+   在 Python 应用程序中使用面向对象编程概念

+   使用`BeautifulSoup`包从网站上爬取数据

+   接收命令行参数

+   利用`inspect`模块

+   动态加载 Python 模块

+   使用 Python 推导

+   使用`Selenium`请求网页并检查其 DOM 元素

在开始之前，重要的是要说，当开发网络爬虫应用程序时，您应该牢记这些类型的应用程序容易受到更改的影响。如果您从中获取数据的网站的开发人员更改了 CSS 类名或 HTML DOM 的结构，应用程序将停止工作。此外，如果我们获取数据的网站的 URL 更改，应用程序将无法发送请求。

# 设置环境

在我们开始编写第一个示例之前，我们需要设置一个环境来工作并安装项目可能具有的任何依赖项。幸运的是，Python 有一个非常好的工具系统来处理虚拟环境。

Python 中的虚拟环境是一个广泛的主题，超出了本书的范围。但是，如果您不熟悉虚拟环境，知道虚拟环境是一个与全局 Python 安装隔离的 Python 环境即可。这种隔离允许开发人员轻松地使用不同版本的 Python，在环境中安装软件包，并管理项目依赖项，而不会干扰 Python 的全局安装。

Python 的安装包含一个名为`venv`的模块，您可以使用它来创建虚拟环境；语法非常简单。我们将要创建的应用程序称为`weatherterm`（天气终端），因此我们可以创建一个同名的虚拟环境，以使其简单。

要创建一个新的虚拟环境，请打开终端并运行以下命令：

```py
$ python3 -m venv weatherterm
```

如果一切顺利，您应该在当前目录中看到一个名为`weatherterm`的目录。现在我们有了虚拟环境，我们只需要使用以下命令激活它：

```py
$ . weatherterm/bin/activate
```

我建议安装并使用`virtualenvwrapper`，这是`virtualenv`工具的扩展。这使得管理、创建和删除虚拟环境以及快速在它们之间切换变得非常简单。如果您希望进一步了解，请访问：[`virtualenvwrapper.readthedocs.io/en/latest/#`](https://virtualenvwrapper.readthedocs.io/en/latest/#)。

现在，我们需要创建一个目录，我们将在其中创建我们的应用程序。不要在创建虚拟环境的同一目录中创建此目录；相反，创建一个项目目录，并在其中创建应用程序目录。我建议您简单地使用与虚拟环境相同的名称命名它。

我正在设置环境并在安装了 Debian 9.2 的机器上运行所有示例，并且在撰写本文时，我正在运行最新的 Python 版本（3.6.2）。如果您是 Mac 用户，情况可能不会有太大差异；但是，如果您使用 Windows，步骤可能略有不同，但是很容易找到有关如何在其中设置虚拟环境的信息。现在，Windows 上的 Python 3 安装效果很好。

进入刚创建的项目目录并创建一个名为`requirements.txt`的文件，内容如下：

```py
beautifulsoup4==4.6.0
selenium==3.6.0
```

这些都是我们这个项目所需的所有依赖项：

+   `BeautifulSoup`**：**这是一个用于解析 HTML 和 XML 文件的包。我们将使用它来解析从天气网站获取的 HTML，并在终端上获取所需的天气数据。它非常简单易用，并且有在线上有很好的文档：[`beautiful-soup-4.readthedocs.io/en/latest/`](http://beautiful-soup-4.readthedocs.io/en/latest/)。

+   `selenium`**：**这是一个用于测试的知名工具集。有许多应用程序，但它主要用于自动测试 Web 应用程序。

要在我们的虚拟环境中安装所需的软件包，可以运行以下命令：

```py
pip install -r requirements.txt
```

始终使用 GIT 或 Mercurial 等版本控制工具是一个好主意。它非常有助于控制更改，检查历史记录，回滚更改等。如果您对这些工具不熟悉，互联网上有很多教程。您可以通过查看 GIT 的文档来开始：[`git-scm.com/book/en/v1/Getting-Started`](https://git-scm.com/book/en/v1/Getting-Started)。

我们需要安装的最后一个工具是 PhantomJS；您可以从以下网址下载：[`phantomjs.org/download.html`](http://phantomjs.org/download.html)

下载后，提取`weatherterm`目录中的内容，并将文件夹重命名为`phantomjs`。

在设置好我们的虚拟环境并安装了 PhantomJS 后，我们准备开始编码！

# 核心功能

首先，创建一个模块的目录。在项目的根目录内，创建一个名为`weatherterm`的子目录。`weatherterm`子目录是我们模块的所在地。模块目录需要两个子目录-`core`和`parsers`。项目的目录结构应该如下所示：

```py
weatherterm
├── phantomjs
└── weatherterm
    ├── core
    ├── parsers   
```

# 动态加载解析器

这个应用程序旨在灵活，并允许开发人员为不同的天气网站创建不同的解析器。我们将创建一个解析器加载器，它将动态发现`parsers`目录中的文件，加载它们，并使它们可供应用程序使用，而无需更改代码的其他部分。在实现新解析器时，我们的加载器将需要遵循以下规则：

+   创建一个实现获取当前天气预报以及五天、十天和周末天气预报方法的类文件

+   文件名必须以`parser`结尾，例如`weather_com_parser.py`

+   文件名不能以双下划线开头

说到这里，让我们继续创建解析器加载器。在`weatherterm/core`目录中创建一个名为`parser_loader.py`的文件，并添加以下内容：

```py
import os
import re
import inspect

def _get_parser_list(dirname):
    files = [f.replace('.py', '')
             for f in os.listdir(dirname)
             if not f.startswith('__')]

    return files

def _import_parsers(parserfiles):

    m = re.compile('.+parser$', re.I)

    _modules = __import__('weatherterm.parsers',
                          globals(),
                          locals(),
                          parserfiles,
                          0)

    _parsers = [(k, v) for k, v in inspect.getmembers(_modules)
                if inspect.ismodule(v) and m.match(k)]

    _classes = dict()

    for k, v in _parsers:
        _classes.update({k: v for k, v in inspect.getmembers(v)
                         if inspect.isclass(v) and m.match(k)})

    return _classes

def load(dirname):
    parserfiles = _get_parser_list(dirname)
    return _import_parsers(parserfiles)
```

首先，执行`_get_parser_list`函数并返回位于`weatherterm/parsers`中的所有文件的列表；它将根据先前描述的解析器规则过滤文件。返回文件列表后，就可以导入模块了。这是由`_import_parsers`函数完成的，它首先导入`weatherterm.parsers`模块，并利用标准库中的 inspect 包来查找模块中的解析器类。

`inspect.getmembers`函数返回一个元组列表，其中第一项是表示模块中的属性的键，第二项是值，可以是任何类型。在我们的情况下，我们对以`parser`结尾的键和类型为类的值感兴趣。

假设我们已经在`weatherterm/parsers`目录中放置了一个解析器，`inspect.getmembers(_modules)`返回的值将看起来像这样：

```py
[('WeatherComParser',
  <class 'weatherterm.parsers.weather_com_parser.WeatherComParser'>),
  ...]
```

`inspect.getmembers(_module)`返回了更多的项目，但它们已被省略，因为在这一点上展示它们并不相关。

最后，我们循环遍历模块中的项目，并提取解析器类，返回一个包含类名和稍后用于创建解析器实例的类对象的字典。

# 创建应用程序的模型

让我们开始创建将代表我们的应用程序从天气网站上爬取的所有信息的模型。我们要添加的第一项是一个枚举，用于表示我们应用程序的用户将提供的天气预报选项。在`weatherterm/core`目录中创建一个名为`forecast_type.py`的文件，内容如下：

```py
from enum import Enum, unique

@unique
class ForecastType(Enum):
    TODAY = 'today'
    FIVEDAYS = '5day'
    TENDAYS = '10day'
    WEEKEND = 'weekend'
```

枚举自 Python 3.4 版本以来一直存在于 Python 标准库中，可以使用创建类的语法来创建。只需创建一个从`enum.Enum`继承的类，其中包含一组设置为常量值的唯一属性。在这里，我们为应用程序提供的四种类型的预报设置了值，可以访问`ForecastType.TODAY`、`ForecastType.WEEKEND`等值。

请注意，我们正在分配与枚举的属性项不同的常量值，原因是以后这些值将用于构建请求天气网站的 URL。

应用程序需要另一个枚举来表示用户在命令行中可以选择的温度单位。这个枚举将包含摄氏度和华氏度项目。

首先，让我们包含一个基本枚举。在`weatherterm/core`目录中创建一个名为`base_enum.py`的文件，内容如下：

```py
from enum import Enum

class BaseEnum(Enum):
    def _generate_next_value_(name, start, count, last_value):
        return name
```

`BaseEnum`是一个非常简单的类，继承自`Enum`。我们在这里想要做的唯一一件事是覆盖`_generate_next_value_`方法，以便从`BaseEnum`继承的每个枚举和具有值设置为`auto()`的属性将自动获得与属性名称相同的值。

现在，我们可以为温度单位创建一个枚举。在`weatherterm/core`目录中创建一个名为`unit.py`的文件，内容如下：

```py
from enum import auto, unique

from .base_enum import BaseEnum

@unique
class Unit(BaseEnum):
    CELSIUS = auto()
    FAHRENHEIT = auto()
```

这个类继承自我们刚刚创建的`BaseEnum`，每个属性都设置为`auto()`，这意味着枚举中每个项目的值将自动设置。由于`Unit`类继承自`BaseEnum`，每次调用`auto()`时，`BaseEnum`上的`_generate_next_value_`方法将被调用，并返回属性本身的名称。

在我们尝试这个之前，让我们在`weatherterm/core`目录中创建一个名为`__init__.py`的文件，并导入我们刚刚创建的枚举，如下所示：

```py
from .unit import Unit
```

如果我们在 Python REPL 中加载这个类并检查值，将会发生以下情况：

```py
Python 3.6.2 (default, Sep 11 2017, 22:31:28) 
[GCC 6.3.0 20170516] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from weatherterm.core import Unit
>>> [value for key, value in Unit.__members__.items()]
[<Unit.CELSIUS: 'CELSIUS'>, <Unit.FAHRENHEIT: 'FAHRENHEIT'>]
```

我们还想要添加到我们应用程序的核心模块的另一项内容是一个类，用于表示解析器返回的天气预报数据。让我们继续在`weatherterm/core`目录中创建一个名为`forecast.py`的文件，内容如下：

```py
from datetime import date

from .forecast_type import ForecastType

class Forecast:
    def __init__(
            self,
            current_temp,
            humidity,
            wind,
            high_temp=None,
            low_temp=None,
            description='',
            forecast_date=None,
            forecast_type=ForecastType.TODAY):
        self._current_temp = current_temp
        self._high_temp = high_temp
        self._low_temp = low_temp
        self._humidity = humidity
        self._wind = wind
        self._description = description
        self._forecast_type = forecast_type

        if forecast_date is None:
            self.forecast_date = date.today()
        else:
            self._forecast_date = forecast_date

    @property
    def forecast_date(self):
        return self._forecast_date

    @forecast_date.setter
    def forecast_date(self, forecast_date):
        self._forecast_date = forecast_date.strftime("%a %b %d")

    @property
    def current_temp(self):
        return self._current_temp

    @property
    def humidity(self):
        return self._humidity

    @property
    def wind(self):
        return self._wind

    @property
    def description(self):
        return self._description

    def __str__(self):
        temperature = None
        offset = ' ' * 4

        if self._forecast_type == ForecastType.TODAY:
            temperature = (f'{offset}{self._current_temp}\xb0\n'
                           f'{offset}High {self._high_temp}\xb0 / '
                           f'Low {self._low_temp}\xb0 ')
        else:
            temperature = (f'{offset}High {self._high_temp}\xb0 / '
                           f'Low {self._low_temp}\xb0 ')

        return(f'>> {self.forecast_date}\n'
               f'{temperature}'
               f'({self._description})\n'
               f'{offset}Wind: '
               f'{self._wind} / Humidity: {self._humidity}\n')
```

在 Forecast 类中，我们将定义我们将要解析的所有数据的属性：

| `current_temp` | 表示当前温度。仅在获取今天的天气预报时才可用。 |
| --- | --- |
| `humidity` | 一天中的湿度百分比。 |
| `wind` | 有关今天当前风级的信息。 |
| `high_temp` | 一天中的最高温度。 |
| `low_temp` | 一天中的最低温度。 |
| `description` | 天气条件的描述，例如*部分多云*。 |
| `forecast_date` | 预测日期；如果未提供，将设置为当前日期。 |
| `forecast_type` | 枚举`ForecastType`中的任何值（`TODAY`，`FIVEDAYS`，`TENDAYS`或`WEEKEND`）。 |

我们还可以实现两个名为`forecast_date`的方法，使用`@property`和`@forecast_date.setter`装饰器。`@property`装饰器将方法转换为`Forecast`类的`_forecast_date`属性的 getter，而`@forecast_date.setter`将方法转换为 setter。之所以在这里定义 setter，是因为每次需要在`Forecast`的实例中设置日期时，我们都需要确保它将被相应地格式化。在 setter 中，我们调用`strftime`方法，传递格式代码`%a`（缩写的星期几名称），`%b`（缩写的月份名称）和`%d`（月份的第几天）。

格式代码`%a`和`%b`将使用在运行代码的机器上配置的区域设置。

最后，我们重写`__str__`方法，以便在使用`print`，`format`和`str`函数时以我们希望的方式格式化输出。

默认情况下，`weather.com`使用的温度单位是`华氏度`，我们希望我们的应用程序用户可以选择使用摄氏度。因此，让我们继续在`weatherterm/core`目录中创建一个名为`unit_converter.py`的文件，内容如下：

```py
from .unit import Unit

class UnitConverter:
    def __init__(self, parser_default_unit, dest_unit=None):
        self._parser_default_unit = parser_default_unit
        self.dest_unit = dest_unit

        self._convert_functions = {
            Unit.CELSIUS: self._to_celsius,
            Unit.FAHRENHEIT: self._to_fahrenheit,
        }

    @property
    def dest_unit(self):
        return self._dest_unit

    @dest_unit.setter
    def dest_unit(self, dest_unit):
        self._dest_unit = dest_unit

    def convert(self, temp):

        try:
            temperature = float(temp)
        except ValueError:
            return 0

        if (self.dest_unit == self._parser_default_unit or
                self.dest_unit is None):
            return self._format_results(temperature)

        func = self._convert_functions[self.dest_unit]
        result = func(temperature)

        return self._format_results(result)

    def _format_results(self, value):
        return int(value) if value.is_integer() else f'{value:.1f}'

    def _to_celsius(self, fahrenheit_temp):
        result = (fahrenheit_temp - 32) * 5/9
        return result

    def _to_fahrenheit(self, celsius_temp):
        result = (celsius_temp * 9/5) + 32
        return result
```

这个类将负责将摄氏度转换为华氏度，反之亦然。这个类的初始化器有两个参数；解析器使用的默认单位和目标单位。在初始化器中，我们将定义一个包含用于温度单位转换的函数的字典。

`convert`方法只接受一个参数，即温度。在这里，温度是一个字符串，因此我们需要尝试将其转换为浮点值；如果失败，它将立即返回零值。

您还可以验证目标单位是否与解析器的默认单位相同。在这种情况下，我们不需要继续执行任何转换；我们只需格式化值并返回它。

如果需要执行转换，我们可以查找`_convert_functions`字典，找到需要运行的`conversion`函数。如果找到我们正在寻找的函数，我们调用它并返回格式化的值。

下面的代码片段显示了`_format_results`方法，这是一个实用方法，将为我们格式化温度值：

```py
return int(value) if value.is_integer() else f'{value:.1f}'
```

`_format_results`方法检查数字是否为整数；如果`value.is_integer()`返回`True`，则表示数字是整数，例如 10.0。如果为`True`，我们将使用`int`函数将值转换为 10；否则，该值将作为具有精度为 1 的定点数返回。Python 中的默认精度为 6。最后，有两个实用方法执行温度转换，`_to_celsius`和`_to_fahrenheit`。

现在，我们只需要编辑`weatherterm/core`目录中的`__init__.py`文件，并包含以下导入语句：

```py
from .base_enum import BaseEnum
from .unit_converter import UnitConverter
from .forecast_type import ForecastType
from .forecast import Forecast
```

# 从天气网站获取数据

我们将添加一个名为`Request`的类，负责从天气网站获取数据。让我们在`weatherterm/core`目录中添加一个名为`request.py`的文件，内容如下：

```py
import os
from selenium import webdriver

class Request:
    def __init__(self, base_url):
        self._phantomjs_path = os.path.join(os.curdir,
                                          'phantomjs/bin/phantomjs')
        self._base_url = base_url
        self._driver = webdriver.PhantomJS(self._phantomjs_path)

    def fetch_data(self, forecast, area):
        url = self._base_url.format(forecast=forecast, area=area)
        self._driver.get(url)

        if self._driver.title == '404 Not Found':
            error_message = ('Could not find the area that you '
                             'searching for')
            raise Exception(error_message)

        return self._driver.page_source
```

这个类非常简单；初始化程序定义了基本 URL 并创建了一个 PhantomJS 驱动程序，使用 PhantomJS 安装的路径。`fetch_data`方法格式化 URL，添加预测选项和区域。之后，`webdriver`执行请求并返回页面源代码。如果返回的标记标题是`404 Not Found`，它将引发异常。不幸的是，`Selenium`没有提供获取 HTTP 状态代码的正确方法；这比比较字符串要好得多。

您可能会注意到，我在一些类属性前面加了下划线符号。我通常这样做是为了表明底层属性是私有的，不应该在类外部设置。在 Python 中，没有必要这样做，因为没有办法设置私有或公共属性；但是，我喜欢这样做，因为我可以清楚地表明我的意图。

现在，我们可以在`weatherterm/core`目录中的`__init__.py`文件中导入它：

```py
from .request import Request
```

现在我们有一个解析器加载器，可以加载我们放入`weatherterm/parsers`目录中的任何解析器，我们有一个表示预测模型的类，以及一个枚举`ForecastType`，因此我们可以指定要解析的预测类型。该枚举表示温度单位和实用函数，用于将温度从`华氏度`转换为`摄氏度`和从`摄氏度`转换为`华氏度`。因此，现在，我们应该准备好创建应用程序的入口点，以接收用户传递的所有参数，运行解析器，并在终端上呈现数据。

# 使用 ArgumentParser 获取用户输入

在我们第一次运行应用程序之前，我们需要添加应用程序的入口点。入口点是在执行应用程序时将首先运行的代码。

我们希望为我们的应用程序的用户提供尽可能好的用户体验，因此我们需要添加的第一个功能是能够接收和解析命令行参数，执行参数验证，根据需要设置参数，最后但并非最不重要的是，显示一个有组织且信息丰富的帮助系统，以便用户可以查看可以使用哪些参数以及如何使用应用程序。

听起来很繁琐，对吧？

幸运的是，Python 自带了很多功能，标准库中包含一个很棒的模块，可以让我们以非常简单的方式实现这一点；该模块称为`argparse`。

另一个很好的功能是让我们的应用程序易于分发给用户。一种方法是在`weatherterm`模块目录中创建一个`__main__.py`文件，然后可以像运行常规脚本一样运行模块。Python 将自动运行`__main__.py`文件，如下所示：

```py
$ python -m weatherterm
```

另一个选项是压缩整个应用程序目录并执行 Python，传递 ZIP 文件的名称。这是一种简单、快速、简单的分发 Python 程序的方法。

还有许多其他分发程序的方法，但这超出了本书的范围；我只是想给你一些使用`__main__.py`文件的例子。

有了这个说法，让我们在`weatherterm`目录中创建一个`__main__.py`文件，内容如下：

```py
import sys
from argparse import ArgumentParser

from weatherterm.core import parser_loader
from weatherterm.core import ForecastType
from weatherterm.core import Unit

def _validate_forecast_args(args):
    if args.forecast_option is None:
        err_msg = ('One of these arguments must be used: '
                   '-td/--today, -5d/--fivedays, -10d/--tendays, -
                    w/--weekend')
        print(f'{argparser.prog}: error: {err_msg}', 
        file=sys.stderr)
        sys.exit()

parsers = parser_loader.load('./weatherterm/parsers')

argparser = ArgumentParser(
    prog='weatherterm',
    description='Weather info from weather.com on your terminal')

required = argparser.add_argument_group('required arguments')

required.add_argument('-p', '--parser',
                      choices=parsers.keys(),
                      required=True,
                      dest='parser',
                      help=('Specify which parser is going to be  
                       used to '
                            'scrape weather information.'))

unit_values = [name.title() for name, value in Unit.__members__.items()]

argparser.add_argument('-u', '--unit',
                       choices=unit_values,
                       required=False,
                       dest='unit',
                       help=('Specify the unit that will be used to 
                       display '
                             'the temperatures.'))

required.add_argument('-a', '--areacode',
                      required=True,
                      dest='area_code',
                      help=('The code area to get the weather 
                       broadcast from. '
                            'It can be obtained at 
                              https://weather.com'))

argparser.add_argument('-v', '--version',
                       action='version',
                       version='%(prog)s 1.0')

argparser.add_argument('-td', '--today',
                       dest='forecast_option',
                       action='store_const',
                       const=ForecastType.TODAY,
                       help='Show the weather forecast for the 
                       current day')

args = argparser.parse_args()

_validate_forecast_args(args)

cls = parsers[args.parser]

parser = cls()
results = parser.run(args)

for result in results:
    print(results)
```

我们的应用程序将接受的天气预报选项（今天、五天、十天和周末预报）不是必需的；但是，至少必须在命令行中提供一个选项，因此我们创建了一个名为`_validate_forecast_args`的简单函数来执行此验证。此函数将显示帮助消息并退出应用程序。

首先，我们获取`weatherterm/parsers`目录中可用的所有解析器。解析器列表将用作解析器参数的有效值。

`ArgumentParser`对象负责定义参数、解析值和显示帮助，因此我们创建一个`ArgumentParser`的实例，并创建一个必需参数的参数组。这将使帮助输出看起来更加美观和有组织。

为了使参数和帮助输出更有组织，我们将在`ArgumentParser`对象中创建一个组。此组将包含我们的应用程序需要的所有必需参数。这样，我们的应用程序的用户可以轻松地看到哪些参数是必需的，哪些是不必需的。

我们通过以下语句实现了这一点：

```py
required = argparser.add_argument_group('required arguments')
```

在为必需参数创建参数组之后，我们获取枚举`Unit`的所有成员的列表，并使用`title()`函数使只有第一个字母是大写字母。

现在，我们可以开始添加我们的应用程序能够在命令行接收的参数。大多数参数定义使用相同的一组关键字参数，因此我不会覆盖所有参数。

我们将创建的第一个参数是`--parser`或`-p`：

```py
required.add_argument('-p', '--parser',
                      choices=parsers.keys(),
                      required=True,
                      dest='parser',
                      help=('Specify which parser is going to be 
                       used to '
                            'scrape weather information.'))
```

让我们分解创建解析器标志时使用的`add_argument`的每个参数：

+   前两个参数是标志。在这种情况下，用户可以使用`-p`或`--parser`在命令行中传递值给此参数，例如`--parser WeatherComParser`。

+   `choices`参数指定我们正在创建的参数的有效值列表。在这里，我们使用`parsers.keys()`，它将返回一个解析器名称的列表。这种实现的优势是，如果我们添加一个新的解析器，它将自动添加到此列表中，而且不需要对此文件进行任何更改。

+   `required`参数，顾名思义，指定参数是否为必需的。

+   `dest`参数指定要添加到解析器参数的结果对象中的属性的名称。`parser_args()`返回的对象将包含一个名为`parser`的属性，其值是我们在命令行中传递给此参数的值。

+   最后，`help`参数是参数的帮助文本，在使用`-h`或`--help`标志时显示。

转到`--today`参数：

```py
argparser.add_argument('-td', '--today',
                       dest='forecast_option',
                       action='store_const',
                       const=ForecastType.TODAY,
                       help='Show the weather forecast for the 
                       current day')
```

这里有两个我们以前没有见过的关键字参数，`action`和`const`。

行动可以绑定到我们创建的参数，并且它们可以执行许多操作。`argparse`模块包含一组很棒的操作，但如果您需要执行特定操作，可以创建自己的操作来满足您的需求。`argparse`模块中定义的大多数操作都是将值存储在解析结果对象属性中的操作。

在前面的代码片段中，我们使用了`store_const`操作，它将一个常量值存储到`parse_args()`返回的对象中的属性中。

我们还使用了关键字参数`const`，它指定在命令行中使用标志时的常量默认值。

记住我提到过可以创建自定义操作吗？参数 unit 是自定义操作的一个很好的用例。`choices`参数只是一个字符串列表，因此我们使用此推导式获取`Unit`枚举中每个项目的名称列表，如下所示：

```py
unit_values = [name.title() for name, value in Unit.__members__.items()]

required.add_argument('-u', '--unit',
                      choices=unit_values,
                      required=False,
                      dest='unit',
                      help=('Specify the unit that will be used to 
                       display '
                            'the temperatures.'))
```

`parse_args()`返回的对象将包含一个名为 unit 的属性，其值为字符串（`Celsius`或`Fahrenheit`），但这并不是我们想要的。我们可以通过创建自定义操作来更改此行为。

首先，在`weatherterm/core`目录中添加一个名为`set_unit_action.py`的新文件，内容如下：

```py
from argparse import Action

from weatherterm.core import Unit

class SetUnitAction(Action):

    def __call__(self, parser, namespace, values,    
     option_string=None):
        unit = Unit[values.upper()]
        setattr(namespace, self.dest, unit)
```

这个操作类非常简单；它只是继承自`argparse.Action`并覆盖`__call__`方法，当解析参数值时将调用该方法。这将设置为目标属性。

`parser`参数将是`ArgumentParser`的一个实例。命名空间是`argparser.Namespace`的一个实例，它只是一个简单的类，包含`ArgumentParser`对象中定义的所有属性。如果您使用调试器检查此参数，您将看到类似于这样的东西：

```py
Namespace(area_code=None, fields=None, forecast_option=None, parser=None, unit=None)
```

`values`参数是用户在命令行上传递的值；在我们的情况下，它可以是摄氏度或华氏度。最后，`option_string`参数是为参数定义的标志。对于单位参数，`option_string`的值将是`-u`。

幸运的是，Python 中的枚举允许我们使用项目访问它们的成员和属性：

```py
Unit[values.upper()]
```

在 Python REPL 中验证这一点，我们有：

```py
>>> from weatherterm.core import Unit
>>> Unit['CELSIUS']
<Unit.CELSIUS: 'CELSIUS'>
>>> Unit['FAHRENHEIT']
<Unit.FAHRENHEIT: 'FAHRENHEIT'>
```

在获取正确的枚举成员之后，我们设置了命名空间对象中`self.dest`指定的属性的值。这样更清晰，我们不需要处理魔术字符串。

有了自定义操作，我们需要在`weatherterm/core`目录中的`__init__.py`文件中添加导入语句：

```py
from .set_unit_action import SetUnitAction
```

只需在文件末尾包含上面的行。然后，我们需要将其导入到`__main__.py`文件中，就像这样：

```py
from weatherterm.core import SetUnitAction
```

然后，我们将在单位参数的定义中添加`action`关键字参数，并将其设置为`SetUnitAction`，就像这样：

```py
required.add_argument('-u', '--unit',
                      choices=unit_values,
                      required=False,
                      action=SetUnitAction,
                      dest='unit',
                      help=('Specify the unit that will be used to 
                       display '
                            'the temperatures.'))
```

所以，当我们的应用程序的用户使用摄氏度标志`-u`时，`parse_args()`函数返回的对象的属性单位的值将是：

`<Unit.CELSIUS: 'CELSIUS'>`

代码的其余部分非常简单；我们调用`parse_args`函数来解析参数并将结果设置在`args`变量中。然后，我们使用`args.parser`的值（所选解析器的名称）并访问解析器字典中的项。请记住，值是类类型，所以我们创建解析器的实例，最后调用 run 方法，这将启动网站抓取。

# 创建解析器

为了第一次运行我们的代码，我们需要创建一个解析器。我们可以快速创建一个解析器来运行我们的代码，并检查数值是否被正确解析。

让我们继续，在`weatherterm/parsers`目录中创建一个名为`weather_com_parser.py`的文件。为了简单起见，我们只会创建必要的方法，当这些方法被调用时，我们唯一要做的就是引发`NotImplementedError`：

```py
from weatherterm.core import ForecastType

class WeatherComParser:

    def __init__(self):
        self._forecast = {
            ForecastType.TODAY: self._today_forecast,
            ForecastType.FIVEDAYS: self._five_and_ten_days_forecast,
            ForecastType.TENDAYS: self._five_and_ten_days_forecast,
            ForecastType.WEEKEND: self._weekend_forecast,
            }

    def _today_forecast(self, args):
        raise NotImplementedError()

    def _five_and_ten_days_forecast(self, args):
        raise NotImplementedError()

    def _weekend_forecast(self, args):
        raise NotImplementedError()

    def run(self, args):
        self._forecast_type = args.forecast_option
        forecast_function = self._forecast[args.forecast_option]
        return forecast_function(args)
```

在初始化器中，我们创建了一个字典，其中键是`ForecasType`枚举的成员，值是绑定到任何这些选项的方法。我们的应用程序将能够呈现今天的、五天的、十天的和周末的预报，所以我们实现了所有四种方法。

`run`方法只做两件事；它使用我们在命令行中传递的`forecast_option`查找需要执行的函数，并执行该函数返回其值。

现在，如果你在命令行中运行命令，应用程序终于准备好第一次执行了：

```py
$ python -m weatherterm --help
```

应该看到应用程序的帮助选项：

```py
usage: weatherterm [-h] -p {WeatherComParser} [-u {Celsius,Fahrenheit}] -a AREA_CODE [-v] [-td] [-5d] [-10d] [-w]

Weather info from weather.com on your terminal

optional arguments:
 -h, --help show this help message and exit
 -u {Celsius,Fahrenheit}, --unit {Celsius,Fahrenheit}
 Specify the unit that will be used to display 
 the temperatures.
 -v, --version show program's version number and exit
 -td, --today Show the weather forecast for the current day

require arguments:
 -p {WeatherComParser}, --parser {WeatherComParser}
 Specify which parser is going to be used to scrape
 weather information.
 -a AREA_CODE, --areacode AREA_CODE
 The code area to get the weather broadcast from. It
 can be obtained at https://weather.com
```

正如你所看到的，`ArgumentParse`模块已经提供了开箱即用的帮助输出。你可以按照自己的需求自定义输出的方式，但我觉得默认布局非常好。

注意，`-p`参数已经给了你选择`WeatherComParser`的选项。因为解析器加载器已经为我们完成了所有工作，所以不需要在任何地方硬编码它。`-u`（`--unit`）标志也包含了枚举`Unit`的项。如果有一天你想扩展这个应用程序并添加新的单位，你唯一需要做的就是在这里添加新的枚举项，它将自动被捡起并包含为`-u`标志的选项。

现在，如果你再次运行应用程序并传递一些参数：

```py
$ python -m weatherterm -u Celsius -a SWXX2372:1:SW -p WeatherComParser -td
```

你会得到类似于这样的异常：

![](img/c2b594fc-7ad7-4b4f-877a-3476564ec7f6.png)

不用担心——这正是我们想要的！如果您跟踪堆栈跟踪，您会看到一切都按预期工作。当我们运行我们的代码时，我们在`__main__.py`文件中选择了所选解析器上的`run`方法，然后选择与预报选项相关联的方法，例如`_today_forecast`，最后将结果存储在`forecast_function`变量中。

当执行存储在`forecast_function`变量中的函数时，引发了`NotImplementedError`异常。到目前为止一切顺利；代码完美运行，现在我们可以开始为这些方法中的每一个添加实现。

# 获取今天的天气预报

核心功能已经就位，应用程序的入口点和参数解析器将为我们的应用程序的用户带来更好的体验。现在，终于到了我们一直在等待的时间，开始实现解析器的时间。我们将开始实现获取今天的天气预报的方法。

由于我在瑞典，我将使用区号`SWXX2372:1:SW`（瑞典斯德哥尔摩）；但是，您可以使用任何您想要的区号。要获取您选择的区号，请转到[`weather.com`](https://weather.com)并搜索您想要的区域。选择区域后，将显示当天的天气预报。请注意，URL 会更改，例如，搜索瑞典斯德哥尔摩时，URL 会更改为：

[`weather.com/weather/today/l/SWXX2372:1:SW`](https://weather.com/weather/today/l/SWXX2372:1:SW)

对于巴西圣保罗，将是：

[`weather.com/weather/today/l/BRXX0232:1:BR`](https://weather.com/weather/today/l/BRXX0232:1:BR)

请注意，URL 只有一个部分会更改，这就是我们要作为参数传递给我们的应用程序的区号。

# 添加辅助方法

首先，我们需要导入一些包：

```py
import re

from weatherterm.core import Forecast
from weatherterm.core import Request
from weatherterm.core import Unit
from weatherterm.core import UnitConverter
```

在初始化程序中，我们将添加以下代码：

```py
self._base_url = 'http://weather.com/weather/{forecast}/l/{area}'
self._request = Request(self._base_url)

self._temp_regex = re.compile('([0-9]+)\D{,2}([0-9]+)')
self._only_digits_regex = re.compile('[0-9]+')

self._unit_converter = UnitConverter(Unit.FAHRENHEIT)
```

在初始化程序中，我们定义了要使用的 URL 模板，以执行对天气网站的请求；然后，我们创建了一个`Request`对象。这是将代表我们执行请求的对象。

只有在解析今天的天气预报温度时才使用正则表达式。

我们还定义了一个`UnitConverter`对象，并将默认单位设置为`华氏度`。

现在，我们准备开始添加两个方法，这两个方法将负责实际搜索某个类中的 HTML 元素并返回其内容。第一个方法称为`_get_data`：

```py
def _get_data(self, container, search_items):
    scraped_data = {}

    for key, value in search_items.items():
        result = container.find(value, class_=key)

        data = None if result is None else result.get_text()

        if data is not None:
            scraped_data[key] = data

    return scraped_data
```

这种方法的想法是在匹配某些条件的容器中搜索项目。`container`只是 HTML 中的 DOM 元素，而`search_items`是一个字典，其中键是 CSS 类，值是 HTML 元素的类型。它可以是 DIV、SPAN 或您希望获取值的任何内容。

它开始循环遍历`search_items.items()`，并使用 find 方法在容器中查找元素。如果找到该项，我们使用`get_text`提取 DOM 元素的文本，并将其添加到一个字典中，当没有更多项目可搜索时将返回该字典。

我们将实现的第二个方法是`_parser`方法。这将使用我们刚刚实现的`_get_data`：

```py
def _parse(self, container, criteria):
    results = [self._get_data(item, criteria)
               for item in container.children]

    return [result for result in results if result]
```

在这里，我们还会得到一个`container`和`criteria`，就像`_get_data`方法一样。容器是一个 DOM 元素，标准是我们要查找的节点的字典。第一个推导式获取所有容器的子元素，并将它们传递给刚刚实现的`_get_data`方法。

结果将是一个包含所有已找到项目的字典列表，我们只会返回不为空的字典。

我们还需要实现另外两个辅助方法，以便获取今天的天气预报。让我们实现一个名为`_clear_str_number`的方法：

```py
def _clear_str_number(self, str_number):
    result = self._only_digits_regex.match(str_number)
    return '--' if result is None else result.group()
```

这种方法将使用正则表达式确保只返回数字。

还需要实现的最后一个方法是 `_get_additional_info` 方法：

```py
def _get_additional_info(self, content):
    data = tuple(item.td.span.get_text()
                 for item in content.table.tbody.children)
    return data[:2]
```

这个方法循环遍历表格行，获取每个单元格的文本。这个推导式将返回有关天气的大量信息，但我们只对前 `2` 个感兴趣，即风和湿度。

# 实施今天的天气预报

现在是时候开始添加 `_today_forecast` 方法的实现了，但首先，我们需要导入 `BeautifulSoup`。在文件顶部添加以下导入语句：

```py
from bs4 import BeautifulSoup
```

现在，我们可以开始添加 `_today_forecast` 方法：

```py
def _today_forecast(self, args):
    criteria = {
        'today_nowcard-temp': 'div',
        'today_nowcard-phrase': 'div',
        'today_nowcard-hilo': 'div',
        }

    content = self._request.fetch_data(args.forecast_option.value,
                                       args.area_code)

    bs = BeautifulSoup(content, 'html.parser')

    container = bs.find('section', class_='today_nowcard-container')

    weather_conditions = self._parse(container, criteria)

    if len(weather_conditions) < 1:
        raise Exception('Could not parse weather foreecast for 
        today.')

    weatherinfo = weather_conditions[0]

    temp_regex = re.compile(('H\s+(\d+|\-{,2}).+'
                             'L\s+(\d+|\-{,2})'))
    temp_info = temp_regex.search(weatherinfo['today_nowcard-hilo'])
    high_temp, low_temp = temp_info.groups()

    side = container.find('div', class_='today_nowcard-sidecar')
    humidity, wind = self._get_additional_info(side)

    curr_temp = self._clear_str_number(weatherinfo['today_nowcard- 
    temp'])

    self._unit_converter.dest_unit = args.unit

    td_forecast = Forecast(self._unit_converter.convert(curr_temp),
                           humidity,
                           wind,
                           high_temp=self._unit_converter.convert(
                               high_temp),
                           low_temp=self._unit_converter.convert(
                               low_temp),
                           description=weatherinfo['today_nowcard-
                            phrase'])

    return [td_forecast]
```

这是在命令行上使用`-td` 或`--today` 标志时将被调用的函数。让我们分解这段代码，以便我们可以轻松理解它的作用。理解这个方法很重要，因为这些方法解析了与此非常相似的其他天气预报选项（五天、十天和周末）的数据。

这个方法的签名非常简单；它只获取`args`，这是在`__main__` 方法中创建的`Argument` 对象。在这个方法中，我们首先创建一个包含我们想要在标记中找到的所有 DOM 元素的`criteria` 字典：

```py
criteria = {
    'today_nowcard-temp': 'div',
    'today_nowcard-phrase': 'div',
    'today_nowcard-hilo': 'div',
}
```

如前所述，`criteria` 字典的关键是 DOM 元素的 CSS 类的名称，值是 HTML 元素的类型：

+   `today_nowcard-temp` 类是包含当前温度的 DOM 元素的 CSS 类

+   `today_nowcard-phrase` 类是包含天气条件文本（多云，晴天等）的 DOM 元素的 CSS 类

+   `today_nowcard-hilo` 类是包含最高和最低温度的 DOM 元素的 CSS 类

接下来，我们将获取、创建和使用`BeautifulSoup` 来解析 DOM：

```py
content = self._request.fetch_data(args.forecast_option.value, 
                                   args.area_code)

bs = BeautifulSoup(content, 'html.parser')

container = bs.find('section', class_='today_nowcard-container')

weather_conditions = self._parse(container, criteria)

if len(weather_conditions) < 1:
    raise Exception('Could not parse weather forecast for today.')

weatherinfo = weather_conditions[0]
```

首先，我们利用我们在核心模块上创建的`Request` 类的`fetch_data` 方法，并传递两个参数；第一个是预报选项，第二个参数是我们在命令行上传递的地区代码。

获取数据后，我们创建一个`BeautifulSoup` 对象，传递`content`和一个`parser`。因为我们得到的是 HTML，所以我们使用`html.parser`。

现在是开始寻找我们感兴趣的 HTML 元素的时候了。记住，我们需要找到一个容器元素，`_parser` 函数将搜索子元素并尝试找到我们在字典条件中定义的项目。对于今天的天气预报，包含我们需要的所有数据的元素是一个带有 `today_nowcard-container` CSS 类的`section` 元素。

`BeautifulSoup` 包含了 `find` 方法，我们可以使用它来查找具有特定条件的 HTML DOM 中的元素。请注意，关键字参数称为`class_` 而不是`class`，因为`class` 在 Python 中是一个保留字。

现在我们有了容器元素，我们可以将其传递给`_parse` 方法，它将返回一个列表。我们检查结果列表是否至少包含一个元素，并在为空时引发异常。如果不为空，我们只需获取第一个元素并将其分配给`weatherinfo` 变量。`weatherinfo` 变量现在包含了我们正在寻找的所有项目的字典。

下一步是分割最高和最低温度：

```py
temp_regex = re.compile(('H\s+(\d+|\-{,2}).+'
                         'L\s+(\d+|\-{,2})'))
temp_info = temp_regex.search(weatherinfo['today_nowcard-hilo'])
high_temp, low_temp = temp_info.groups()
```

我们想解析从带有 `today_nowcard-hilo` CSS 类的 DOM 元素中提取的文本，文本应该看起来像 `H 50 L 60`，`H -- L 60` 等。提取我们想要的文本的一种简单方法是使用正则表达式：

`H\s+(\d+|\-{,2}).L\s+(\d+|\-{,2})`

我们可以将这个正则表达式分成两部分。首先，我们想要得到最高温度—`H\s+(\d+|\-{,2})`；这意味着它将匹配一个`H`后面跟着一些空格，然后它将分组一个匹配数字或最多两个破折号的值。之后，它将匹配任何字符。最后，第二部分基本上做了相同的事情；不过，它开始匹配一个`L`。

执行搜索方法后，调用`groups()`函数返回了正则表达式组，这种情况下将返回两个组，一个是最高温度，另一个是最低温度。

我们想要向用户提供的其他信息是关于风和湿度的信息。包含这些信息的容器元素具有一个名为`today_nowcard-sidecar`的 CSS 类：

```py
side = container.find('div', class_='today_nowcard-sidecar')
wind, humidity = self._get_additional_info(side)
```

我们只需找到容器并将其传递给`_get_additional_info`方法，该方法将循环遍历容器的子元素，提取文本，最后为我们返回结果。

最后，这个方法的最后一部分：

```py
curr_temp = self._clear_str_number(weatherinfo['today_nowcard-temp'])

self._unit_converter.dest_unit = args.unit

td_forecast = Forecast(self._unit_converter.convert(curr_temp),
                       humidity,
                       wind,
                       high_temp=self._unit_converter.convert(
                           high_temp),
                       low_temp=self._unit_converter.convert(
                           low_temp),
                       description=weatherinfo['today_nowcard- 
                        phrase'])

return [td_forecast]
```

由于当前温度包含一个我们此时不想要的特殊字符（度数符号），我们使用`_clr_str_number`方法将`weatherinfo`字典的`today_nowcard-temp`项传递给它。

现在我们有了所有需要的信息，我们构建`Forecast`对象并返回它。请注意，我们在这里返回一个数组；这是因为我们将要实现的所有其他选项（五天、十天和周末天气预报）都将返回一个列表，为了使其一致；也为了在终端上显示这些信息时更方便，我们也返回一个列表。

还要注意的一点是，我们正在使用`UnitConverter`的转换方法将所有温度转换为命令行中选择的单位。

再次运行命令时：

```py
$ python -m weatherterm -u Fahrenheit -a SWXX2372:1:SW -p WeatherComParser -td
```

你应该看到类似于这样的输出：

![](img/1f2ea039-104c-4786-a400-ae107a248609.png)

恭喜！你已经实现了你的第一个网络爬虫应用。接下来，让我们添加其他的预报选项。

# 获取五天和十天的天气预报

我们目前正在从([weather.com](https://weather.com/en-IN/))这个网站上爬取天气预报，它也提供了风和湿度的天气预报。

五天和十天，所以在这一部分，我们将实现解析这些预报选项的方法。

呈现五天和十天数据的页面的标记非常相似；它们具有相同的 DOM 结构和共享相同的 CSS 类，这使得我们可以实现只适用于这两个选项的方法。让我们继续并向`wheater_com_parser.py`文件添加一个新的方法，内容如下：

```py
def _parse_list_forecast(self, content, args):
    criteria = {
        'date-time': 'span',
        'day-detail': 'span',
        'description': 'td',
        'temp': 'td',
        'wind': 'td',
        'humidity': 'td',
    }

    bs = BeautifulSoup(content, 'html.parser')

    forecast_data = bs.find('table', class_='twc-table')
    container = forecast_data.tbody

    return self._parse(container, criteria)
```

正如我之前提到的，五天和十天的天气预报的 DOM 结构非常相似，因此我们创建了`_parse_list_forecast`方法，可以用于这两个选项。首先，我们定义了标准：

+   `date-time`是一个`span`元素，包含代表星期几的字符串

+   `day-detail`是一个`span`元素，包含一个日期的字符串，例如，`SEP 29`

+   `description`是一个`TD`元素，包含天气状况，例如，``Cloudy``

+   `temp`是一个`TD`元素，包含高低温度等温度信息

+   `wind`是一个`TD`元素，包含风力信息

+   `humidity`是一个`TD`元素，包含湿度信息

现在我们有了标准，我们创建一个`BeatufulSoup`对象，传递内容和`html.parser`。我们想要获取的所有数据都在一个名为`twc-table`的 CSS 类的表格中。我们找到表格并将`tbody`元素定义为容器。

最后，我们运行`_parse`方法，传递`container`和我们定义的`criteria`。这个函数的返回将看起来像这样：

```py
[{'date-time': 'Today',
  'day-detail': 'SEP 28',
  'description': 'Partly Cloudy',
  'humidity': '78%',
  'temp': '60°50°',
  'wind': 'ESE 10 mph '},
 {'date-time': 'Fri',
  'day-detail': 'SEP 29',
  'description': 'Partly Cloudy',
  'humidity': '79%',
  'temp': '57°48°',
  'wind': 'ESE 10 mph '},
 {'date-time': 'Sat',
  'day-detail': 'SEP 30',
  'description': 'Partly Cloudy',
  'humidity': '77%',
  'temp': '57°49°',
  'wind': 'SE 10 mph '},
 {'date-time': 'Sun',
  'day-detail': 'OCT 1',
  'description': 'Cloudy',
  'humidity': '74%',
  'temp': '55°51°',
  'wind': 'SE 14 mph '},
 {'date-time': 'Mon',
  'day-detail': 'OCT 2',
  'description': 'Rain',
  'humidity': '87%',
  'temp': '55°48°',
  'wind': 'SSE 18 mph '}]
```

我们需要创建的另一个方法是一个为我们准备数据的方法，例如，解析和转换温度值，并创建一个`Forecast`对象。添加一个名为`_prepare_data`的新方法，内容如下：

```py
def _prepare_data(self, results, args):
    forecast_result = []

    self._unit_converter.dest_unit = args.unit

    for item in results:
        match = self._temp_regex.search(item['temp'])
        if match is not None:
            high_temp, low_temp = match.groups()

        try:
            dateinfo = item['weather-cell']
            date_time, day_detail = dateinfo[:3], dateinfo[3:]
            item['date-time'] = date_time
            item['day-detail'] = day_detail
        except KeyError:
            pass

        day_forecast = Forecast(
            self._unit_converter.convert(item['temp']),
            item['humidity'],
            item['wind'],
            high_temp=self._unit_converter.convert(high_temp),
            low_temp=self._unit_converter.convert(low_temp),
            description=item['description'].strip(),
            forecast_date=f'{item["date-time"]} {item["day-
             detail"]}',
            forecast_type=self._forecast_type)
        forecast_result.append(day_forecast)

    return forecast_result
```

这个方法非常简单。首先，循环遍历结果，并应用我们创建的正则表达式来分割存储在`item['temp']`中的高温和低温。如果匹配成功，它将获取组并将值分配给`high_temp`和`low_temp`。

之后，我们创建一个`Forecast`对象，并将其附加到稍后将返回的列表中。

最后，我们添加一个在使用`-5d`或`-10d`标志时将被调用的方法。创建另一个名为`_five_and_ten_days_forecast`的方法，内容如下：

```py
def _five_and_ten_days_forecast(self, args):
    content = self._request.fetch_data(args.forecast_option.value, 
    args.area_code)
    results = self._parse_list_forecast(content, args)
    return self._prepare_data(results)
```

这个方法只获取页面的内容，传递`forecast_option`值和区域代码，因此可以构建 URL 来执行请求。当数据返回时，我们将其传递给`_parse_list_forecast`，它将返回一个`Forecast`对象的列表（每天一个）；最后，我们使用`_prepare_data`方法准备要返回的数据。

在运行命令之前，我们需要在我们实现的命令行工具中启用此选项；转到`__main__.py`文件，并在`-td`标志的定义之后，添加以下代码：

```py
argparser.add_argument('-5d', '--fivedays',
                       dest='forecast_option',
                       action='store_const',
                       const=ForecastType.FIVEDAYS,
                       help='Shows the weather forecast for the next         
                       5 days')
```

现在，再次运行应用程序，但这次使用`-5d`或`--fivedays`标志：

```py
$ python -m weatherterm -u Fahrenheit -a SWXX2372:1:SW -p WeatherComParser -5d
```

它将产生以下输出：

```py
>> [Today SEP 28]
 High 60° / Low 50° (Partly Cloudy)
 Wind: ESE 10 mph / Humidity: 78%

>> [Fri SEP 29]
 High 57° / Low 48° (Partly Cloudy)
 Wind: ESE 10 mph / Humidity: 79%

>> [Sat SEP 30]
 High 57° / Low 49° (Partly Cloudy)
 Wind: SE 10 mph / Humidity: 77%

>> [Sun OCT 1]
 High 55° / Low 51° (Cloudy)
 Wind: SE 14 mph / Humidity: 74%

>> [Mon OCT 2]
 High 55° / Low 48° (Rain)
 Wind: SSE 18 mph / Humidity: 87%
```

为了结束本节，让我们在`__main__.py`文件中添加一个选项，以便获取未来十天的天气预报，就在`-5d`标志定义的下面。添加以下代码：

```py
argparser.add_argument('-10d', '--tendays',
                       dest='forecast_option',
                       action='store_const',
                       const=ForecastType.TENDAYS,
                       help='Shows the weather forecast for the next  
                       10 days')
```

如果您运行与获取五天预报相同的命令，但将`-5d`标志替换为`-10d`，如下所示：

```py
$ python -m weatherterm -u Fahrenheit -a SWXX2372:1:SW -p WeatherComParser -10d
```

您应该看到十天的天气预报输出：

```py
>> [Today SEP 28]
 High 60° / Low 50° (Partly Cloudy)
 Wind: ESE 10 mph / Humidity: 78%

>> [Fri SEP 29]
 High 57° / Low 48° (Partly Cloudy)
 Wind: ESE 10 mph / Humidity: 79%

>> [Sat SEP 30]
 High 57° / Low 49° (Partly Cloudy)
 Wind: SE 10 mph / Humidity: 77%

>> [Sun OCT 1]
 High 55° / Low 51° (Cloudy)
 Wind: SE 14 mph / Humidity: 74%

>> [Mon OCT 2]
 High 55° / Low 48° (Rain)
 Wind: SSE 18 mph / Humidity: 87%

>> [Tue OCT 3]
 High 56° / Low 46° (AM Clouds/PM Sun)
 Wind: S 10 mph / Humidity: 84%

>> [Wed OCT 4]
 High 58° / Low 47° (Partly Cloudy)
 Wind: SE 9 mph / Humidity: 80%

>> [Thu OCT 5]
 High 57° / Low 46° (Showers)
 Wind: SSW 8 mph / Humidity: 81%

>> [Fri OCT 6]
 High 57° / Low 46° (Partly Cloudy)
 Wind: SW 8 mph / Humidity: 76%

>> [Sat OCT 7]
 High 56° / Low 44° (Mostly Sunny)
 Wind: W 7 mph / Humidity: 80%

>> [Sun OCT 8]
 High 56° / Low 44° (Partly Cloudy)
 Wind: NNE 7 mph / Humidity: 78%

>> [Mon OCT 9]
 High 56° / Low 43° (AM Showers)
 Wind: SSW 9 mph / Humidity: 79%

>> [Tue OCT 10]
 High 55° / Low 44° (AM Showers)
 Wind: W 8 mph / Humidity: 79%

>> [Wed OCT 11]
 High 55° / Low 42° (AM Showers)
 Wind: SE 7 mph / Humidity: 79%

>> [Thu OCT 12]
 High 53° / Low 43° (AM Showers)
 Wind: NNW 8 mph / Humidity: 87%
```

如您所见，我在瑞典写这本书时天气并不是很好。

# 获取周末天气预报

我们将在我们的应用程序中实现的最后一个天气预报选项是获取即将到来的周末天气预报的选项。这个实现与其他实现有些不同，因为周末天气返回的数据与今天、五天和十天的天气预报略有不同。

DOM 结构不同，一些 CSS 类名也不同。如果您还记得我们之前实现的方法，我们总是使用`_parser`方法，该方法为我们提供容器 DOM 和带有搜索条件的字典作为参数。该方法的返回值也是一个字典，其中键是我们正在搜索的 DOM 的类名，值是该 DOM 元素中的文本。

由于周末页面的 CSS 类名不同，我们需要实现一些代码来获取结果数组并重命名所有键，以便`_prepare_data`函数可以正确使用抓取的结果。

说到这一点，让我们继续在`weatherterm/core`目录中创建一个名为`mapper.py`的新文件，内容如下：

```py
class Mapper:

    def __init__(self):
        self._mapping = {}

    def _add(self, source, dest):
        self._mapping[source] = dest

    def remap_key(self, source, dest):
        self._add(source, dest)

    def remap(self, itemslist):
        return [self._exec(item) for item in itemslist]

    def _exec(self, src_dict):
        dest = dict()

        if not src_dict:
            raise AttributeError('The source dictionary cannot be  
            empty or None')

        for key, value in src_dict.items():
            try:
                new_key = self._mapping[key]
                dest[new_key] = value
            except KeyError:
                dest[key] = value
        return dest
```

`Mapper`类获取一个包含字典的列表，并重命名我们想要重命名的特定键。这里的重要方法是`remap_key`和`remap`。`remap_key`接收两个参数，`source`和`dest`。`source`是我们希望重命名的键，`dest`是该键的新名称。`remap_key`方法将其添加到一个名为`_mapping`的内部字典中，以便以后查找新的键名。

`remap`方法只是获取包含字典的列表，并对该列表中的每个项目调用`_exec`方法，该方法首先创建一个全新的字典，然后检查字典是否为空。在这种情况下，它会引发`AttributeError`。

如果字典有键，我们循环遍历其项，搜索当前项的键是否在映射字典中具有新名称。如果找到新的键名，将创建一个具有新键名的新项；否则，我们只保留旧名称。循环结束后，返回包含所有具有新名称键的字典的列表。

现在，我们只需要将其添加到`weatherterm/core`目录中的`__init__.py`文件中：

```py
from .mapper import Mapper
```

而且，在`weatherterm/parsers`目录中的`weather_com_parser.py`文件中，我们需要导入`Mapper`：

```py
from weatherterm.core import Mapper
```

有了映射器，我们可以继续在`weather_com_parser.py`文件中创建`_weekend_forecast`方法，如下所示：

```py
def _weekend_forecast(self, args):
    criteria = {
        'weather-cell': 'header',
        'temp': 'p',
        'weather-phrase': 'h3',
        'wind-conditions': 'p',
        'humidity': 'p',
    }

    mapper = Mapper()
    mapper.remap_key('wind-conditions', 'wind')
    mapper.remap_key('weather-phrase', 'description')

    content = self._request.fetch_data(args.forecast_option.value,
                                       args.area_code)

    bs = BeautifulSoup(content, 'html.parser')

    forecast_data = bs.find('article', class_='ls-mod')
    container = forecast_data.div.div

    partial_results = self._parse(container, criteria)
    results = mapper.remap(partial_results)

    return self._prepare_data(results, args)
```

该方法首先通过以与其他方法完全相同的方式定义标准来开始；但是，DOM 结构略有不同，一些 CSS 名称也不同：

+   `weather-cell`：包含预报日期：`FriSEP 29`

+   `temp`：包含温度（高和低）：`57°F48°F`

+   `weather-phrase`：包含天气条件：`多云`

+   `wind-conditions`：风信息

+   `humidity`：湿度百分比

正如你所看到的，为了使其与`_prepare_data`方法很好地配合，我们需要重命名结果集中字典中的一些键——`wind-conditions`应该是`wind`，`weather-phrase`应该是`description`。

幸运的是，我们引入了`Mapper`类来帮助我们：

```py
mapper = Mapper()
mapper.remap_key('wind-conditions', 'wind')
mapper.remap_key('weather-phrase', 'description')
```

我们创建一个`Mapper`对象并说，将`wind-conditions`重新映射为`wind`，将`weather-phrase`重新映射为`description`：

```py
content = self._request.fetch_data(args.forecast_option.value,
                                   args.area_code)

bs = BeautifulSoup(content, 'html.parser')

forecast_data = bs.find('article', class_='ls-mod')
container = forecast_data.div.div

partial_results = self._parse(container, criteria)
```

我们获取所有数据，使用`html.parser`创建一个`BeautifulSoup`对象，并找到包含我们感兴趣的子元素的容器元素。对于周末预报，我们有兴趣获取具有名为`ls-mod`的 CSS 类的`article`元素，并在`article`中向下移动到第一个子元素，这是一个 DIV，并获取其第一个子元素，这也是一个 DIV 元素。

HTML 应该看起来像这样：

```py
<article class='ls-mod'>
  <div>
    <div>
      <!-- this DIV will be our container element -->
    </div>
  </div>
</article>
```

这就是我们首先找到文章，将其分配给`forecast_data`，然后使用`forecast_data.div.div`，这样我们就可以得到我们想要的 DIV 元素。

在定义容器之后，我们将其与容器元素一起传递给`_parse`方法；当我们收到结果时，我们只需要运行`Mapper`实例的`remap`方法，它将在我们调用`_prepare_data`之前为我们规范化数据。

现在，在运行应用程序并获取周末天气预报之前的最后一个细节是，我们需要将`--w`和`--weekend`标志包含到`ArgumentParser`中。打开`weatherterm`目录中的`__main__.py`文件，并在`--tenday`标志的下方添加以下代码：

```py
argparser.add_argument('-w', '--weekend',
                       dest='forecast_option',
                       action='store_const',
                       const=ForecastType.WEEKEND,
                       help=('Shows the weather forecast for the 
                             next or '
                             'current weekend'))
```

太好了！现在，使用`-w`或`--weekend`标志运行应用程序：

```py
>> [Fri SEP 29]
 High 13.9° / Low 8.9° (Partly Cloudy)
 Wind: ESE 10 mph / Humidity: 79%

>> [Sat SEP 30]
 High 13.9° / Low 9.4° (Partly Cloudy)
 Wind: SE 10 mph / Humidity: 77%

>> [Sun OCT 1]
 High 12.8° / Low 10.6° (Cloudy)
 Wind: SE 14 mph / Humidity: 74%
```

请注意，这次我使用了`-u`标志来选择摄氏度。输出中的所有温度都以摄氏度表示，而不是华氏度。

# 总结

在本章中，您学习了 Python 中面向对象编程的基础知识；我们介绍了如何创建类，使用继承，并使用`@property`装饰器创建 getter 和 setter。

我们介绍了如何使用 inspect 模块来获取有关模块、类和函数的更多信息。最后但并非最不重要的是，我们利用了强大的`Beautifulsoup`包来解析 HTML 和`Selenium`来向天气网站发出请求。

我们还学习了如何使用 Python 标准库中的`argparse`模块实现命令行工具，这使我们能够提供更易于使用且具有非常有用的文档的工具。

接下来，我们将开发一个小包装器，围绕 Spotify Rest API，并使用它来创建一个远程控制终端。
