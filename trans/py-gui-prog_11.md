# 连接到云

似乎几乎每个应用程序迟早都需要与外部世界交流，你的`ABQ数据录入`应用程序也不例外。您收到了一些新的功能请求，这将需要与远程服务器和服务进行一些交互。首先，质量保证部门正在研究当地天气条件如何影响每个实验室的环境数据；他们要求以按需下载和存储当地天气数据的方式。第二个请求来自您的老板，她仍然需要每天上传CSV文件到中央公司服务器。她希望这个过程能够简化，并且可以通过鼠标点击来完成。

在本章中，您将学习以下主题：

+   连接到Web服务并使用`urllib`下载数据

+   使用`requests`库管理更复杂的HTTP交互

+   使用`ftplib`连接和上传到FTP服务

# 使用`urllib`进行HTTP连接

每次在浏览器中打开网站时，您都在使用**超文本传输协议，或HTTP**。 HTTP是在25年前创建的，作为Web浏览器下载HTML文档的一种方式，但已经发展成为最受欢迎的客户端-服务器通信协议之一，用于任何数量的目的。我们不仅可以使用它在互联网上传输从纯文本到流媒体视频的任何内容，而且应用程序还可以使用它来传输数据，启动远程过程或分发计算任务。

基本的HTTP事务包括客户端和服务器，其功能如下：

+   **客户端**：客户端创建请求。请求指定一个称为**方法**的操作。最常见的方法是`GET`，用于检索数据，以及`POST`，用于提交数据。请求有一个URL，指定了请求所在的主机、端口和路径，以及包含元数据的标头，如数据类型或授权令牌。最后，它有一个有效负载，其中可能包含键值对中的序列化数据。

+   **服务器**：服务器接收请求并返回响应。响应包含一个包含元数据的标头，例如响应的状态代码或内容类型。它还包含实际响应内容的有效负载，例如HTML、XML、JSON或二进制数据。

在Web浏览器中，这些操作是在后台进行的，但我们的应用程序将直接处理请求和响应对象，以便与远程HTTP服务器进行通信。

# 使用`urllib.request`进行基本下载

`urllib.request`模块是一个用于生成HTTP请求的Python模块。它包含一些用于生成HTTP请求的函数和类，其中最基本的是`urlopen()`函数。`urlopen()`函数可以创建`GET`或`POST`请求并将其发送到远程服务器。

让我们探索`urllib`的工作原理；打开Python shell并执行以下命令：

```py
>>> from urllib.request import urlopen
>>> response = urlopen('http://packtpub.com')
```

`urlopen()`函数至少需要一个URL字符串。默认情况下，它会向URL发出`GET`请求，并返回一个包装从服务器接收到的响应的对象。这个`response`对象公开了从服务器接收到的元数据或内容，我们可以在我们的应用程序中使用。

响应的大部分元数据都在标头中，我们可以使用`getheader()`来提取，如下所示：

```py
>>> response.getheader('Content-Type')
'text/html; charset=utf-8'
>>> response.getheader('Server')
'nginx/1.4.5'
```

响应具有状态，指示在请求过程中遇到的错误条件（如果有）；状态既有数字又有文本解释，称为`reason`。

我们可以从我们的`response`对象中提取如下：

```py
>>> response.status
200
>>> response.reason
'OK'
```

在上述代码中，`200`状态表示事务成功。客户端端错误，例如发送错误的URL或不正确的权限，由400系列的状态表示，而服务器端问题由500系列的状态表示。

可以使用类似于文件句柄的接口来检索`response`对象的有效负载，如下所示：

```py
>>> html = response.read()
>>> html[:15]
b'<!DOCTYPE html>'
```

就像文件句柄一样，响应只能使用`read()`方法读取一次；与文件句柄不同的是，它不能使用`seek()`“倒带”，因此如果需要多次访问响应数据，重要的是将响应数据保存在另一个变量中。`response.read()`的输出是一个字节对象，应将其转换或解码为适当的对象。

在这种情况下，我们有一个`utf-8`字符串如下：

```py
>>> html.decode('utf-8')[:15]
'<!DOCTYPE html>'
```

除了`GET`请求之外，`urlopen()`还可以生成`POST`请求。

为了做到这一点，我们包括一个`data`参数如下：

```py
>>> response = urlopen('http://duckduckgo.com', data=b'q=tkinter')
```

`data`值需要是一个URL编码的字节对象。URL编码的数据字符串由用`&`符号分隔的键值对组成，某些保留字符被编码为URL安全的替代字符（例如，空格字符是`%20`，或者有时只是`+`）。

这样的字符串可以手工创建，但使用`urllib.parse`模块提供的`urlencode`函数更容易。看一下以下代码：

```py
>>> from urllib.parse import urlencode
>>> data = {'q': 'tkinter, python', 'ko': '-2', 'kz': '-1'}
>>> urlencode(data)
'q=tkinter%2C+python&ko=-2&kz=-1'
>>> response = urlopen('http://duckduckgo.com', data=urlencode(data).encode())
```

`data`参数必须是字节，而不是字符串，因此在`urlopen`接受它之前必须对URL编码的字符串调用`encode()`。

让我们尝试下载我们应用程序所需的天气数据。我们将使用`http://weather.gov`提供美国境内的天气数据。我们将要下载的实际URL是[http://w1.weather.gov/xml/current_obs/STATION.xml](http://w1.weather.gov/xml/current_obs/STATION.xml)，其中`STATION`被本地天气站的呼号替换。在ABQ的情况下，我们将使用位于印第安纳州布卢明顿的KBMG。

QA团队希望您记录温度（摄氏度）、相对湿度、气压（毫巴）和天空状况（一个字符串，如阴天或晴天）。他们还需要天气站观测到天气的日期和时间。

# 创建下载函数

我们将创建几个访问网络资源的函数，这些函数不会与任何特定的类绑定，因此我们将它们放在自己的文件`network.py`中。让我们看看以下步骤：

1.  在`abq_data_entry`模块目录中创建`network.py`。

1.  现在，让我们打开`network.py`并开始我们的天气下载功能：

```py
from urllib.request import urlopen

def get_local_weather(station):
    url = (
        'http://w1.weather.gov/xml/current_obs/{}.xml'
        .format(station))
    response = urlopen(url)
```

我们的函数将以`station`字符串作为参数，以防以后需要更改，或者如果有人想在不同的设施使用这个应用程序。该函数首先通过构建天气数据的URL并使用`urlopen()`请求来开始。

1.  假设事情进行顺利，我们只需要解析出这个`response`数据，并将其放入`Application`类可以传递给数据库模型的形式中。为了确定我们将如何处理响应，让我们回到Python shell并检查其中的数据：

```py
>>> response = urlopen('http://w1.weather.gov/xml/current_obs/KBMG.xml')
>>> print(response.read().decode())
<?xml version="1.0" encoding="ISO-8859-1"?>
<?xml-stylesheet href="latest_ob.xsl" type="text/xsl"?>
<current_observation version="1.0"

         xsi:noNamespaceSchemaLocation="http://www.weather.gov/view/current_observation.xsd">
        <credit>NOAA's National Weather Service</credit>
        <credit_URL>http://weather.gov/</credit_URL>
....
```

1.  如URL所示，响应的有效负载是一个XML文档，其中大部分我们不需要。经过一些搜索，我们可以找到我们需要的字段如下：

```py
        <observation_time_rfc822>Wed, 14 Feb 2018 14:53:00 
        -0500</observation_time_rfc822>
        <weather>Fog/Mist</weather>
        <temp_c>11.7</temp_c>
        <relative_humidity>96</relative_humidity>
        <pressure_mb>1018.2</pressure_mb>
```

好的，我们需要的数据都在那里，所以我们只需要将它从XML字符串中提取出来，以便我们的应用程序可以使用。让我们花点时间了解一下解析XML数据。

# 解析XML天气数据

Python标准库包含一个`xml`包，其中包含用于解析或创建XML数据的几个子模块。`xml.etree.ElementTree`子模块是一个简单、轻量级的解析器，应该满足我们的需求。

让我们将`ElementTree`导入到我们的`network.py`文件中，如下所示：

```py
from xml.etree import ElementTree
```

现在，在函数的末尾，我们将解析我们的`response`对象中的XML数据，如下所示：

```py
    xmlroot = ElementTree.fromstring(response.read())
```

`fromstring()`方法接受一个XML字符串并返回一个`Element`对象。为了获得我们需要的数据，我们需要了解`Element`对象代表什么，以及如何使用它。

XML是数据的分层表示；一个元素代表这个层次结构中的一个节点。一个元素以一个标签开始，这是尖括号内的文本字符串。每个标签都有一个匹配的闭合标签，这只是在标签名称前加上一个斜杠的标签。在开放和关闭标签之间，一个元素可能有其他子元素，也可能有文本。一个元素也可以有属性，这些属性是放在开放标签的尖括号内的键值对，就在标签名称之后。

看一下以下XML的示例：

```py
<star_system starname="Sol">
  <planet>Mercury</planet>
  <planet>Venus</planet>
  <planet>Earth
    <moon>Luna</moon>
    </planet>
  <planet>Mars
    <moon>Phobos</moon>
    <moon>Deimos</moon>
    </planet>
  <dwarf_planet>Ceres</dwarf_planet>
</star_system>
```

这是太阳系的（不完整的）XML描述。根元素的标签是`<star_system>`，具有`starname`属性。在这个根元素下，我们有四个`<planet>`元素和一个`<dwarf_planet>`元素，每个元素都包含行星名称的文本节点。一些行星节点还有子`<moon>`节点，每个节点包含卫星名称的文本节点。

可以说，这些数据可以以不同的方式进行结构化；例如，行星名称可以在行星元素内部的子`<name>`节点中，或者作为`<planet>`标签的属性列出。虽然XML语法是明确定义的，但XML文档的实际结构取决于创建者，因此完全解析XML数据需要了解数据在文档中的布局方式。

如果您在之前在shell中下载的XML天气数据中查看，您会注意到它是一个相当浅的层次结构。在`<current_observations>`节点下，有许多子元素，它们的标签代表特定的数据字段，如温度、湿度、风寒等。

为了获得这些子元素，`Element`为我们提供了以下各种方法：

| **方法** | **返回** |
| `iter()` | 所有子节点的迭代器（递归） |
| `find(tag)` | 匹配给定标签的第一个元素 |
| `findall(tag)` | 匹配给定标签的元素列表 |
| `getchildren()` | 直接子节点的列表 |
| `iterfind(tag)` | 匹配给定标签的所有子节点的迭代器（递归） |

早些时候我们下载XML数据时，我们确定了包含我们想要从该文档中提取的数据的五个标签：`<observation_time_rfc822>`、`<weather>`、`<temp_c>`、`<relative_humidity>`和`<pressure_mb>`。我们希望我们的`get_local_weather()`函数返回一个包含每个键的Python `dict`。

让我们在`network.py`文件中添加以下行：

```py
    xmlroot = ElementTree.fromstring(response.read())
    weatherdata = {
        'observation_time_rfc822': None,
        'temp_c': None,
        'relative_humidity': None,
        'pressure_mb': None,
        'weather': None
    }
```

我们的第一行从响应中提取原始XML并将其解析为`Element`树，将根节点返回给`xmlroot`。然后，我们设置了包含我们想要从XML数据中提取的标签的`dict`。

现在，让我们通过执行以下代码来获取值：

```py
    for tag in weatherdata:
        element = xmlroot.find(tag)
        if element is not None:
            weatherdata[tag] = element.text
```

对于我们的每个标签名称，我们将使用`find()`方法来尝试在`xmlroot`中定位具有匹配标签的元素。这个特定的XML文档不使用重复的标签，所以任何标签的第一个实例应该是唯一的。如果匹配了标签，我们将得到一个`Element`对象；如果没有，我们将得到`None`，因此在尝试访问其`text`值之前，我们需要确保`element`不是`None`。

要完成函数，只需返回`weatherdata`。

您可以在Python shell中测试此函数；从命令行，导航到`ABQ_Data_Entry`目录并启动Python shell：

```py
>>> from abq_data_entry.network import get_local_weather
>>> get_local_weather('KBMG')
{'observation_time_rfc822': 'Wed, 14 Feb 2018 16:53:00 -0500',
 'temp_c': '11.7', 'relative_humidity': '96', 'pressure_mb': '1017.0',
 'weather': 'Drizzle Fog/Mist'}
```

您应该得到一个包含印第安纳州布卢明顿当前天气状况的`dict`。您可以在[http://w1.weather.gov/xml/current_obs/](http://w1.weather.gov/xml/current_obs/)找到美国其他城市的站点代码。

现在我们有了天气函数，我们只需要构建用于存储数据和触发操作的表格。

# 实现天气数据存储

为了存储我们的天气数据，我们将首先在ABQ数据库中创建一个表来保存单独的观测数据，然后构建一个`SQLModel`方法来存储数据。我们不需要担心编写代码来检索数据，因为我们实验室的质量保证团队有他们自己的报告工具，他们将使用它来访问数据。

# 创建SQL表

打开`create_db.sql`文件，并添加一个新的`CREATE TABLE`语句如下：

```py
CREATE TABLE local_weather (
        datetime TIMESTAMP(0) WITH TIME ZONE PRIMARY KEY,
        temperature NUMERIC(5,2),
        rel_hum NUMERIC(5, 2),
        pressure NUMERIC(7,2),
        conditions VARCHAR(32)
        );
```

我们在记录上使用`TIMESTAMP`数据类型作为主键；保存相同时间戳的观测两次是没有意义的，所以这是一个足够好的键。`TIMESTAMP`数据类型后面的`(0)`大小表示我们需要多少小数位来测量秒。由于这些测量大约每小时进行一次，而且我们每四个小时或更长时间（实验室检查完成时）只需要一次，所以在我们的时间戳中不需要秒的小数部分。

请注意，我们保存了时区；当时间戳可用时，始终将时区数据与时间戳一起存储！这可能看起来并不必要，特别是当您的应用程序将在永远不会改变时区的工作场所运行时，但是有许多边缘情况，比如夏令时变化，缺少时区可能会造成重大问题。

在数据库中运行这个`CREATE`查询来构建表，然后我们继续创建我们的`SQLModel`方法。

# 实现SQLModel.add_weather_data()方法

在`models.py`中，让我们添加一个名为`add_weather_data()`的新方法到`SQLModel`类中，它只接受一个数据`dict`作为参数。

让我们通过以下方式开始这个方法，编写一个`INSERT`查询：

```py
    def add_weather_data(self, data):
        query = (
            'INSERT INTO local_weather VALUES '
            '(%(observation_time_rfc822)s, %(temp_c)s, '
            '%(relative_humidity)s, %(pressure_mb)s, '
            '%(weather)s)'
        )
```

这是一个使用与`get_local_weather()`函数从XML数据中提取的`dict`键匹配的变量名的参数化`INSERT`查询。我们只需要将这个查询和数据`dict`传递给我们的`query()`方法。

然而，有一个问题；如果我们得到重复的时间戳，我们的查询将因为重复的主键而失败。我们可以先进行另一个查询来检查，但这有点多余，因为PostgreSQL在插入新行之前会检查重复的键。当它检测到这样的错误时，`psycopg2`会引发一个`IntegrityError`异常，所以我们只需要捕获这个异常，如果它被引发了，就什么都不做。

为了做到这一点，我们将在`try...except`块中包装我们的`query()`调用如下：

```py
        try:
            self.query(query, data)
        except pg.IntegrityError:
            # already have weather for this datetime
            pass
```

现在，我们的数据录入人员可以随意调用这个方法，但只有在有新的观测数据需要保存时才会保存记录。

# 更新`SettingsModel`类

在离开`models.py`之前，我们需要添加一个新的应用程序设置来存储首选的天气站。在`SettingsModel.variables`字典中添加一个新条目如下：

```py
    variables = {
        ...
        'weather_station': {'type': 'str', 'value': 'KBMG'},
        ...
```

我们不会为这个设置添加GUI，因为用户不需要更新它。这将由我们或其他实验室站点的系统管理员来确保在每台工作站上正确设置。

# 添加天气下载的GUI元素

`Application`对象现在需要将`network.py`中的天气下载方法与`SQLModel`中的数据库方法连接起来，并使用适当的回调方法，主菜单类可以调用。按照以下步骤进行：

1.  打开`application.py`并开始一个新的方法如下：

```py
    def update_weather_data(self):

      try:
           weather_data = n.get_local_weather(
               self.settings['weather_station'].get())
```

1.  请记住，在错误场景中，`urlopen()`可能会引发任意数量的异常，这取决于HTTP事务出了什么问题。应用程序除了通知用户并退出方法外，实际上没有什么可以处理这些异常的。因此，我们将捕获通用的`Exception`并在`messagebox`中显示文本如下：

```py
        except Exception as e:
            messagebox.showerror(
                title='Error',
                message='Problem retrieving weather data',
                detail=str(e)
            )
            self.status.set('Problem retrieving weather data')
```

1.  如果`get_local_weather()`成功，我们只需要将数据传递给我们的模型方法如下：

```py
        else:
            self.data_model.add_weather_data(weather_data)
            self.status.set(
                'Weather data recorded for {}'
                .format(weather_data['observation_time_rfc822']))
```

除了保存数据，我们还在状态栏中通知用户天气已更新，并显示更新的时间戳。

1.  回调方法完成后，让我们将其添加到我们的`callbacks`字典中：

```py
        self.callbacks = {
            ...
            'update_weather_data': self.update_weather_data,
            ...
```

1.  现在我们可以在主菜单中添加一个回调的命令项。在Windows上，这样的功能放在`Tools`菜单中，由于Gnome和macOS的指南似乎没有指示更合适的位置，我们将在`LinxMainMenu`和`MacOsMainMenu`类中实现一个`Tools`菜单来保存这个命令，以保持一致。在`mainmenu.py`中，从通用菜单类开始，添加一个新菜单如下：

```py
        #Tools menu
        tools_menu = tk.Menu(self, tearoff=False)
        tools_menu.add_command(
            label="Update Weather Data",
            command=self.callbacks['update_weather_data'])
        self.add_cascade(label='Tools', menu=tools_menu)
```

1.  将相同的菜单添加到macOS和Linux菜单类中，并将命令添加到Windows主菜单的`tools_menu`。更新菜单后，您可以运行应用程序并尝试从`Tools`菜单中运行新命令。如果一切顺利，您应该在状态栏中看到如下截图所示的指示：

![](assets/70ad36a4-d8f7-4fd2-ab4c-544dbb793be8.png)

1.  您还应该使用您的PostgreSQL客户端连接到数据库，并通过执行以下SQL命令来检查表中是否现在包含一些天气数据：

```py
SELECT * FROM local_weather;
```

该SQL语句应返回类似以下的输出：

| `datetime` | `temperature` | `rel[hum]` | `pressure` | `conditions` |
| `2018-02-14 22:53:00-06` | `15.00` | `87.00` | `1014.00` | `Overcast` |

# 使用requests进行HTTP

您被要求在您的程序中创建一个函数，将每日数据的CSV提取上传到ABQ的企业Web服务，该服务使用经过身份验证的REST API。虽然`urllib`足够简单，用于简单的一次性`GET`和`POST`请求，但涉及身份验证令牌、文件上传或REST服务的复杂交互令人沮丧和复杂，仅使用`urllib`就很困难。为了完成这项任务，我们将转向`requests`库。

**REST**代表**REpresentational State Transfer**，是围绕高级HTTP语义构建的Web服务的名称。除了`GET`和`POST`，REST API还使用额外的HTTP方法，如`DELETE`，`PUT`和`PATCH`，以及XML或JSON等数据格式，以提供完整范围的API交互。

Python社区强烈推荐第三方的`requests`库，用于涉及HTTP的任何严肃工作（即使`urllib`文档也推荐它）。正如您将看到的，`requests`消除了`urllib`中留下的许多粗糙边缘和过时假设，并为更现代的HTTP交易提供了方便的类和包装函数。`requests`的完整文档可以在[http://docs.python-requests.org](http://docs.python-requests.org)找到，但下一节将涵盖您有效使用它所需的大部分内容。

# 安装和使用requests

`requests`包是用纯Python编写的，因此使用`pip`安装它不需要编译或二进制下载。只需在终端中输入`pip install --user requests`，它就会被添加到您的系统中。

打开您的Python shell，让我们进行如下请求：

```py
>>> import requests
>>> response = requests.request('GET', 'http://www.alandmoore.com')
```

`requests.request`至少需要一个HTTP方法和一个URL。就像`urlopen()`一样，它构造适当的请求数据包，将其发送到URL，并返回表示服务器响应的对象。在这里，我们正在向这位作者的网站发出`GET`请求。

除了`request()`函数，`requests`还有与最常见的HTTP方法对应的快捷函数。

因此，可以进行相同的请求如下：

```py
response = requests.get('http://www.alandmoore.com')
```

`get()`方法只需要URL并执行`GET`请求。同样，`post()`，`put()`，`patch()`，`delete()`和`head()`函数使用相应的HTTP方法发送请求。所有请求函数都接受额外的可选参数。

例如，我们可以通过`POST`请求发送数据如下：

```py
>>> response = requests.post(
    'http://duckduckgo.com',
    data={'q': 'tkinter', 'ko': '-2', 'kz': '-1'})
```

请注意，与`urlopen()`不同的是，我们可以直接使用Python字典作为`data`参数；`requests`会将其转换为适当的字节对象。

与请求函数一起使用的一些常见参数如下：

| **参数** | **目的** |
| `params` | 类似于`data`，但添加到查询字符串而不是有效负载 |
| `json` | 要包含在有效负载中的JSON数据 |
| `headers` | 用于请求的头数据字典 |
| `files` | 一个`{fieldnames: file objects}`字典，作为多部分表单数据请求发送 |
| `auth` | 用于基本HTTP摘要身份验证的用户名和密码元组 |

# requests.session()函数

Web服务，特别是私人拥有的服务，通常是受密码保护的。有时，这是使用较旧的HTTP摘要身份验证系统完成的，我们可以使用请求函数的`auth`参数来处理这个问题。不过，如今更常见的是，身份验证涉及将凭据发布到REST端点以获取会话cookie或认证令牌，用于验证后续请求。

端点简单地是与API公开的数据或功能对应的URL。数据被发送到端点或从端点检索。

`requests`方法通过提供`Session`类使所有这些变得简单。`Session`对象允许您在多个请求之间持久保存设置、cookie和连接。

要创建一个`Session`对象，使用`requests.session()`工厂函数如下：

```py
s = requests.session()
```

现在，我们可以在我们的`Session`对象上调用请求方法，如`get()`、`post()`等，如下所示：

```py
# Assume this is a valid authentication service that returns an auth token
s.post('http://example.com/login', data={'u': 'test', 'p': 'test'})
# Now we would have an auth token
response = s.get('http://example.com/protected_content')
# Our token cookie would be listed here
print(s.cookies.items())
```

这样的令牌和cookie处理是在后台进行的，我们不需要采取任何明确的操作。Cookie存储在`CookieJar`对象中，存储为我们的`Session`对象的`cookies`属性。

我们还可以在`Session`对象上设置值，这些值将在请求之间持续存在，就像这个例子中一样：

```py
s.headers['User-Agent'] = 'Mozilla'
# will be sent with a user-agent string of "Mozilla"
s.get('http://example.com')
```

在这个例子中，我们将用户代理字符串设置为`Mozilla`，这将用于从这个`Session`对象发出的所有请求。我们还可以使用`params`属性设置默认的URL参数，或者使用`hooks`属性设置回调函数。

# 响应对象

从这些请求函数返回的响应对象与`urlopen()`返回的对象不同；它们包含相同的数据，但以稍微不同（通常更方便）的形式返回。

例如，响应头已经被转换成Python的`dict`，如下所示：

```py
>>> r = requests.get('http://www.alandmoore.com')
>>> r.headers
{'Date': 'Thu, 15 Feb 2018 21:13:42 GMT', 'Server': 'Apache',
 'Last-Modified': 'Sat, 17 Jun 2017 14:13:49 GMT',
 'ETag': '"20c003f-19f7-5945391d"', 'Content-Length': '6647',
 'Keep-Alive': 'timeout=15, max=200', 'Connection': 'Keep-Alive',
 'Content-Type': 'text/html'}
```

另一个区别是，`requests`不会自动在HTTP错误时引发异常。但是，可以调用`.raise_for_status()`响应方法来实现这一点。

例如，这个URL将返回一个HTTP `404`错误，如下面的代码所示：

```py
>>> r = requests.get('http://www.example.com/does-not-exist')
>>> r.status_code
404
>>> r.raise_for_status()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/lib/python3.6/site-packages/requests/models.py", line 935, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 404 Client Error: Not Found for url: http://www.example.com/does-not-exist
```

这使我们可以选择使用异常处理或更传统的流程控制逻辑来处理HTTP错误。

# 实现API上传

要开始实现我们的上传功能，我们需要弄清楚我们将要发送的请求的类型。我们已经从公司总部得到了一些关于如何与REST API交互的文档。

文档告诉我们以下内容：

+   首先，我们需要获取一个认证令牌。我们通过向`/auth`端点提交一个`POST`请求来实现这一点。`POST`请求的参数应包括`username`和`password`。

+   获得认证令牌后，我们需要提交我们的CSV文件。请求是一个发送到`/upload`端点的`PUT`请求。文件作为多部分表单数据上传，指定在`file`参数中。

我们已经知道足够的知识来使用`requests`实现我们的REST上传功能，但在这之前，让我们创建一个服务，我们可以用来测试我们的代码。

# 创建一个测试HTTP服务

开发与外部服务互操作的代码可能会很令人沮丧。在编写和调试代码时，我们需要向服务发送大量错误或测试数据；我们不希望在生产服务中这样做，而且“测试模式”并不总是可用的。自动化测试可以使用`Mock`对象来完全屏蔽网络请求，但在开发过程中，能够看到实际发送到Web服务的内容是很好的。

让我们实现一个非常简单的HTTP服务器，它将接受我们的请求并打印有关其接收到的信息。我们可以使用Python标准库的`http.server`模块来实现这一点。

模块文档显示了一个基本HTTP服务器的示例：

```py
from http.server import HTTPServer, BaseHTTPRequestHandler
def run(server_class=HTTPServer, handler_class=BaseHTTPRequestHandler):
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()
run()
```

服务器类`HTTPServer`定义了一个对象，该对象在配置的地址和端口上监听HTTP请求。处理程序类`BaseHTTPRequestHandler`定义了一个接收实际请求数据并返回响应数据的对象。我们将使用此代码作为起点，因此请将其保存在名为`sample_http_server.py`的文件中，保存在`ABQ_Data_Entry`目录之外。

如果您运行此代码，您将在本地计算机的端口`8000`上运行一个Web服务；但是，如果您对此服务进行任何请求，无论是使用`requests`、类似`curl`的工具，还是只是一个Web浏览器，您都会发现它只返回一个HTTP`501`（`不支持的方法`）错误。为了创建一个足够工作的服务器，就像我们的目标API用于测试目的一样，我们需要创建一个自己的处理程序类，该类可以响应必要的HTTP方法。

为此，我们将创建一个名为`TestHandler`的自定义处理程序类，如下所示：

```py
class TestHandler(BaseHTTPRequestHandler):
    pass

def run(server_class=HTTPServer, handler_class=TestHandler):
    ...
```

我们的公司API使用`POST`方法接收登录凭据，使用`PUT`方法接收文件，因此这两种方法都需要工作。要使HTTP方法在请求处理程序中起作用，我们需要实现一个`do_VERB`方法，其中`VERB`是我们的HTTP方法名称的大写形式。

因此，对于`PUT`和`POST`，添加以下代码：

```py
class TestHandler(BaseHTTPRequestHandler):
    def do_POST(self, *args, **kwargs):
        pass

    def do_PUT(self, *args, **kwargs):
        pass
```

仅仅这样还不能解决问题，因为这些方法需要导致我们的处理程序发送某种响应。对于我们的目的，我们不需要任何特定的响应；只要有一个状态为`200`（`OK`）的响应就可以了。

由于两种方法都需要这个，让我们添加一个第三种方法，我们可以从其他两种方法中调用如下：

```py
    def _send_200(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
```

这是大多数HTTP客户端所需的最小响应：状态为`200`，带有有效`Content-type`的标头。这不会向客户端发送任何实际数据，但会告诉客户端其请求已被接收并成功处理。

我们在我们的方法中还想做的另一件事是打印出发送的任何数据，以便我们可以确保我们的客户端发送了正确的数据。

我们将实现以下方法来实现这一点：

```py
    def _print_request_data(self):
        content_length = self.headers['Content-Length']
        print("Content-length: {}".format(content_length))
        data = self.rfile.read(int(content_length))
        print(data.decode('utf-8'))
```

处理程序对象的`headers`属性是一个包含请求标头的`dict`对象，其中包括发送的字节数（`content-length`）。除了打印该信息之外，我们还可以使用它来读取发送的数据。处理程序的`rfile`属性是一个类似文件的对象，其中包含数据；其`read()`方法需要一个长度参数来指定应该读取多少数据，因此我们使用我们提取的`content-length`值。返回的数据是一个`bytes`对象，因此我们将其解码为`utf-8`。

现在我们有了这两种方法，让我们更新`do_POST()`和`do_PUT()`来调用它们，如下所示：

```py
    def do_POST(self, *args, **kwargs):
        print('POST request received')
        self._print_request_data()
        self._send_200()

    def do_PUT(self, *args, **kwargs):
        print("PUT request received")
        self._print_request_data()
        self._send_200()
```

现在，每个方法都将打印出它接收到的`POST`或`PUT`的长度和数据，以及任何数据。在终端窗口中运行此脚本，以便您可以监视其输出。

现在，打开一个shell，让我们测试它，如下所示：

```py
>>> import requests
>>> requests.post('http://localhost:8000', data={1: 'test1', 2: 'test2'})
<Response[200]>
```

在Web服务器终端中，您应该看到以下输出：

```py
POST request received
Content-length: 15
1=test1&2=test2
127.0.0.1 - - [15/Feb/2018 16:22:41] "POST / HTTP/1.1" 200 -
```

我们可以实现其他功能，比如实际检查凭据并返回身份验证令牌，但目前此服务器已足够帮助我们编写和测试客户端代码。

# 创建我们的网络功能

现在我们的测试服务已经启动，让我们开始编写与REST API交互的网络功能：

1.  我们将首先在`network.py`中创建一个函数，该函数将接受CSV文件的路径、上传和身份验证URL以及用户名和密码：

```py
import requests

...

def upload_to_corporate_rest(
    filepath, upload_url, auth_url, username, password):
```

1.  由于我们将不得不处理身份验证令牌，我们应该做的第一件事是创建一个会话。我们将其称为`session`，如下所示：

```py
    session = requests.session()
```

1.  创建会话后，我们将用户名和密码发布到身份验证端点，如下所示：

```py
    response = session.post(
        auth_url,
        data={'username': username, 'password': password})
    response.raise_for_status()
```

如果成功，`session`对象将自动存储我们收到的令牌。如果出现问题，我们调用`raise_for_status()`，这样函数将中止，调用代码可以处理网络或数据问题引发的任何异常。

1.  假设我们没有引发异常，那么在这一点上我们必须经过身份验证，现在可以提交文件了。这将通过`put()`调用完成，如下所示：

```py
    files = {'file': open(filepath, 'rb')}
    response = session.put(
        upload_url,
        files=files
    )
```

发送文件，我们实际上必须打开它并将其作为文件句柄传递给`put()`；请注意，我们以二进制读取模式（`rb`）打开它。`requests`文档建议这样做，因为它确保正确的`content-length`值将被计算到头部中。

1.  发送请求后，我们关闭文件并再次检查失败状态，然后结束函数，如下所示：

```py
    files['file'].close()
    response.raise_for_status()
```

# 更新应用程序

在我们可以从`Application`中调用新函数之前，我们需要实现一种方法来创建每日数据的CSV提取。这将被多个函数使用，因此我们将它与调用上传代码的函数分开实现。按照以下步骤进行：

1.  首先，我们需要一个临时位置来存储我们生成的CSV文件。`tempfile`模块包括用于处理临时文件和目录的函数；我们将导入`mkdtemp()`，它将为我们提供一个特定于平台的临时目录的名称。

```py
from tempfile import mkdtemp
```

请注意，`mdktemp()`实际上并不创建目录；它只是在平台首选的`temp`文件位置中提供一个随机命名的目录的绝对路径。我们必须自己创建目录。

1.  现在，让我们开始我们的新`Application`方法，如下所示：

```py
    def _create_csv_extract(self):
        tmpfilepath = mkdtemp()
        csvmodel = m.CSVModel(
            filename=self.filename.get(), filepath=tmpfilepath)
```

创建临时目录名称后，我们创建了我们的`CSVModel`类的一个实例；即使我们不再将数据存储在CSV文件中，我们仍然可以使用该模型导出CSV文件。我们传递了`Application`对象的默认文件名，仍然设置为`abq_data_record-CURRENTDATE.csv`，以及临时目录的路径作为`filepath`。当然，我们的`CSVModel`目前并不接受`filepath`，但我们马上就会解决这个问题。

1.  创建CSV模型后，我们将从数据库中提取我们的记录，如下所示：

```py
        records = self.data_model.get_all_records()
        if not records:
            return None
```

请记住，我们的`SQLModel.get_all_records()`方法默认返回当天的所有记录的列表。如果我们碰巧没有当天的记录，最好立即停止并警告用户，而不是将空的CSV文件发送给公司，因此如果没有记录，我们从方法中返回`None`。我们的调用代码可以测试`None`返回值并显示适当的警告。

1.  现在，我们只需要遍历记录并将每个记录保存到CSV中，然后返回`CSVModel`对象的文件名，如下所示：

```py
        for record in records:
            csvmodel.save_record(record)

        return csvmodel.filename
```

1.  现在我们有了创建CSV提取文件的方法，我们可以编写回调方法，如下所示：

```py
    def upload_to_corporate_rest(self):

        csvfile = self._create_csv_extract()

        if csvfile is None:
            messagebox.showwarning(
                title='No records',
                message='There are no records to upload'
            )
            return
```

首先，我们创建了一个CSV提取文件并检查它是否为`None`。如果是，我们将显示错误消息并退出该方法。

1.  在上传之前，我们需要从用户那里获取用户名和密码。幸运的是，我们有一个完美的类来做到这一点：

```py
        d = v.LoginDialog(
            self,
            'Login to ABQ Corporate REST API')
        if d.result is not None:
            username, password = d.result
        else:
            return
```

我们的登录对话框在这里为我们服务。与数据库登录不同，我们不会在无限循环中运行它；如果密码错误，用户可以重新运行命令。请记住，如果用户点击取消，`result`将为`None`，因此在这种情况下我们将退出回调方法。

1.  现在，我们可以执行我们的网络函数，如下所示：

```py
        try:
            n.upload_to_corporate_rest(
                csvfile,
                self.settings['abq_upload_url'].get(),
                self.settings['abq_auth_url'].get(),
                username,
                password)
```

我们在`try`块中执行`upload_to_corporate_rest()`，因为它可能引发许多异常。我们从设置对象中传递上传和身份验证URL；我们还没有添加这些，所以在完成之前需要这样做。

1.  现在，让我们捕获一些异常，首先是`RequestException`。如果我们发送到API的数据出现问题，最有可能是用户名和密码错误，就会发生这种异常。我们将异常字符串附加到向用户显示的消息中，如下所示：

```py
        except n.requests.RequestException as e:
            messagebox.showerror('Error with your request', str(e))
```

1.  接下来我们将捕获`ConnectionError`；这个异常将是网络问题的结果，比如实验室的互联网连接断开，或者服务器没有响应：

```py
        except n.requests.ConnectionError as e:
            messagebox.showerror('Error connecting', str(e))
```

1.  任何其他异常都将显示为`General Exception`，如下所示：

```py
        except Exception as e:
            messagebox.showerror('General Exception', str(e))
```

1.  让我们用以下成功对话框结束这个方法：

```py
        else:
            messagebox.showinfo(
                'Success',
                '{} successfully uploaded to REST API.'
                .format(csvfile))
```

1.  让我们通过将此方法添加到`callbacks`中来完成对`Application`的更改：

```py
        self.callbacks = {
            ...
            'upload_to_corporate_rest':  
           self.upload_to_corporate_rest,
            ...
```

# 更新models.py文件

在我们测试新功能之前，`models.py`文件中有一些需要修复的地方。我们将按照以下步骤来解决这些问题：

1.  首先，我们的`CSVModel`类需要能够接受`filepath`：

```py
    def __init__(self, filename, filepath=None):
        if filepath:
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            self.filename = os.path.join(filepath, filename)
        else:
            self.filename = filename
```

如果指定了`filepath`，我们需要首先确保目录存在。由于在`Application`类中调用的`mkdtmp()`方法实际上并没有创建临时目录，我们将在这里创建它。完成后，我们将连接`filepath`和`filename`的值，并将其存储在`CSVModel`对象的`filename`属性中。

1.  我们在`models.py`中需要做的另一件事是添加我们的新设置。滚动到`SettingsModel`类，添加两个更多的`variables`条目如下：

```py
    variables = {
        ...
        'abq_auth_url': {
            'type': 'str',
            'value': 'http://localhost:8000/auth'},
        'abq_upload_url': {
            'type': 'str',
            'value': 'http://localhost:8000/upload'},
         ...
```

我们不会构建一个GUI来设置这些设置，它们需要在用户的配置文件中手动创建，尽管在测试时，我们可以使用默认值。

# 收尾工作

最后要做的事情是将命令添加到我们的主菜单中。

在每个菜单类中为`tools_menu`添加一个新条目：

```py
        tools_menu.add_command(
            label="Upload CSV to corporate REST",
            command=self.callbacks['upload_to_corporate_rest'])
```

现在，运行应用程序，让我们试试。为了使其工作，您至少需要有一个数据输入，并且需要启动`sample_http_server.py`脚本。

如果一切顺利，您应该会得到一个像这样的对话框：

![](assets/96de4996-9652-43f1-a4c6-a2d6bddc73de.png)

您的服务器还应该在终端上打印出类似这样的输出：

```py
POST request received
Content-length: 27
username=test&password=test
127.0.0.1 - - [16/Feb/2018 10:17:22] "POST /auth HTTP/1.1" 200 -
PUT request received
Content-length: 397
--362eadeb828747769e75d5b4b6d32f31
Content-Disposition: form-data; name="file"; filename="abq_data_record_2018-02-16.csv"

Date,Time,Technician,Lab,Plot,Seed sample,Humidity,Light,Temperature,Equipment Fault,Plants,Blossoms,Fruit,Min Height,Max Height,Median Height,Notes
2018-02-16,8:00,Q Murphy,A,1,AXM477,10.00,10.00,10.00,,1,2,3,1.00,3.00,2.00,"
"

--362eadeb828747769e75d5b4b6d32f31--

127.0.0.1 - - [16/Feb/2018 10:17:22] "PUT /upload HTTP/1.1" 200 -
```

注意`POST`和`PUT`请求，以及`PUT`有效负载中的CSV文件的原始文本。我们已成功满足了此功能的API要求。

# 使用ftplib的FTP

虽然HTTP和REST API是客户端-服务器交互的当前趋势，但企业依赖于旧的、经过时间考验的，有时是过时的技术来实现数据传输并不罕见。ABQ也不例外：除了REST上传，您还需要实现对依赖于FTP的ABQ公司的遗留系统的支持。

# FTP的基本概念

**文件传输协议**，或**FTP**，可以追溯到20世纪70年代初，比HTTP早了近20年。尽管如此，它仍然被许多组织广泛用于在互联网上交换大文件。由于FTP以明文形式传输数据和凭据，因此在许多领域被认为有些过时，尽管也有SSL加密的FTP变体可用。

与HTTP一样，FTP客户端发送包含纯文本命令的请求，类似于HTTP方法，FTP服务器返回包含头部和有效负载信息的响应数据包。

然而，这两种协议之间存在许多重大的区别：

+   FTP是**有状态连接**，这意味着客户端和服务器在会话期间保持恒定的连接。换句话说，FTP更像是一个实时电话，而HTTP则像是两个人在语音信箱中对话。

+   在发送任何其他命令或数据之前，FTP需要对会话进行身份验证，即使对于匿名用户也是如此。FTP服务器还实现了更复杂的权限集。

+   FTP有用于传输文本和二进制数据的不同模式（主要区别在于文本模式会自动纠正行尾和接收操作系统的编码）。

+   FTP服务器在其命令的实现上不够一致。

# 创建一个测试FTP服务

在实现FTP上传功能之前，有一个测试FTP服务是有帮助的，就像我们测试HTTP服务一样。当然，您可以下载许多免费的FTP服务器，如FileZilla、PureFTPD、ProFTPD或其他。

不要为了测试应用程序的一个功能而在系统上安装、配置和后来删除FTP服务，我们可以在Python中构建一个基本的服务器。第三方的`pyftpdlib`包为我们提供了一个简单的实现快速脏FTP服务器的方法，足以满足测试需求。

使用`pip`安装`pyftpdlib`：

```py
pip install --user pyftpdlib
```

就像我们简单的HTTP服务器一样，FTP服务由*服务器*对象和*处理程序*对象组成。它还需要一个*授权者*对象来处理身份验证和权限。

我们将从导入这些开始我们的`basic_ftp_server.py`文件：

```py
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
```

为了确保我们的身份验证代码正常工作，让我们用一个测试用户设置我们的`DummyAuthorizer`类：

```py
auth = DummyAuthorizer()
auth.add_user('test', 'test', '.', perm='elrw')
```

`perm`参数接受一个字符的字符串，每个字符代表服务器上的特定权限。在这种情况下，我们有`e`（连接）、`l`（列出）、`r`（读取）和`w`（写入新文件）。还有许多其他权限可用，默认情况下都是关闭的，直到授予，但这对我们的需求已经足够了。

现在，让我们设置处理程序：

```py
handler = FTPHandler
handler.authorizer = auth
```

请注意，我们没有实例化处理程序，只是给类取了别名。服务器类将管理处理程序类的创建。但是，我们可以将我们的`auth`对象分配为处理程序的`authorizer`类，以便任何创建的处理程序都将使用我们的授权者。

最后，让我们设置并运行服务器部分：

```py
address = ('127.0.0.1', 2100)
server = FTPServer(address, handler)

server.serve_forever()
```

这只是简单地用地址元组和处理程序类实例化一个`FTPServer`对象，然后调用对象的`server_forever()`方法。地址元组的形式是`（ip_address，port）`，所以`（'127.0.0.1'，2100）`的元组意味着我们将在计算机的回环地址上的端口`2100`上提供服务。FTP的默认端口通常是21，但在大多数操作系统上，启动监听在`1024`以下端口的服务需要root或系统管理员权限。为了简单起见，我们将使用一个更高的端口。

虽然可以使用`pyftpdlib`构建生产质量的FTP服务器，但我们在这里没有这样做。这个脚本对于测试是足够的，但如果您重视安全性，请不要在生产中使用它。

# 实现FTP上传功能

现在测试服务器已经启动，让我们构建我们的FTP上传功能和GUI的逻辑。虽然标准库中没有包含FTP服务器库，但它包含了`ftplib`模块形式的FTP客户端库。

首先在我们的`network.py`文件中导入`ftplib`：

```py
import ftplib as ftp
```

可以使用`ftplib.FTP`类创建一个FTP会话。因为这是一个有状态的会话，在完成后需要关闭；为了确保我们这样做，`FTP`可以用作上下文管理器。

让我们从连接到FTP服务器开始我们的函数：

```py
def upload_to_corporate_ftp(
        filepath, ftp_host,
        ftp_port, ftp_user, ftp_pass):

    with ftp.FTP() as ftp_cx:
        ftp_cx.connect(ftp_host, ftp_port)
        ftp_cx.login(ftp_user, ftp_pass)
```

`upload_to_corporate()`函数接受CSV文件路径和`FTP`主机、端口、用户和密码，就像我们的`upload_to_corporate_rest()`函数一样。我们首先创建我们的`FTP`对象，然后调用`FTP.connect()`和`FTP.login`。

接下来，`connect()`接受我们要交谈的主机和端口，并与服务器开始会话。在这一点上，我们还没有经过身份验证，但我们确实建立了连接。

然后，`login()`接受用户名和密码，并尝试验证我们的会话。如果我们的凭据检查通过，我们就登录到服务器上，并可以开始发送更多的命令；如果不通过，就会引发`error_perm`异常。但是，我们的会话仍然是活动的，直到我们关闭它，并且如果需要，我们可以发送额外的登录尝试。

要实际上传文件，我们使用`storbinary()`方法：

```py
        filename = path.basename(filepath)
        with open(filepath, 'rb') as fh:
            ftp_cx.storbinary('STOR {}'.format(filename), fh)
```

要发送文件，我们必须以二进制读取模式打开它，然后调用“storbinary”（是的，“stor”，而不是“store”—20世纪70年代的程序员对删除单词中的字母有一种偏好）。

“storbinary”的第一个参数是一个有效的FTP“STOR”命令，通常是“STOR filename”，其中“filename”是您希望在服务器上称为上传数据的名称。必须包含实际的命令字符串似乎有点违反直觉；据推测，这必须是指定的，以防服务器使用稍有不同的命令或语法。

第二个参数是文件对象本身。由于我们将其作为二进制数据发送，因此应该以二进制模式打开它。这可能看起来有点奇怪，因为我们发送的CSV文件本质上是一个纯文本文件，但将其作为二进制数据发送可以保证服务器在传输过程中不会以任何方式更改文件；这几乎总是在传输文件时所希望的，无论所交换数据的性质如何。

这就是我们的网络功能需要为FTP上传完成的所有工作。尽管我们的程序只需要“storbinary()”方法，但值得注意的是，如果您发现自己不得不使用FTP服务器，还有一些其他常见的“ftp”方法。

# 列出文件

在FTP服务器上列出文件有三种方法。“mlsd()”方法调用“MLSD”命令，通常是可用的最佳和最完整的输出。它可以接受一个可选的“path”参数，指定要列出的路径（否则它将列出当前目录），以及一个“facts”列表，例如“size”、“type”或“perm”，反映了您希望与文件名一起包括的数据。 “mlsd()”命令返回一个生成器对象，可以迭代或转换为另一种序列类型。

“MLSD”是一个较新的命令，不一定总是可用，因此还有另外两种可用的方法，“nlst()”和“dir()”，它们对应于较旧的“NLST”和“DIR”命令。这两种方法都接受任意数量的参数，这些参数将被原样附加到发送到服务器的命令字符串。

# 检索文件

从FTP服务器下载文件涉及“retrbinary()”或“retrlines()”方法中的一个，具体取决于我们是否希望使用二进制或文本模式（如前所述，您可能应该始终使用二进制）。与“storbinary”一样，每种方法都需要一个命令字符串作为其第一个参数，但在这种情况下，它应该是一个有效的“RETR”命令（通常“RETR filename”就足够了）。

第二个参数是一个回调函数，它将在每一行（对于“retrlines()”）或每个块（对于“retrbinary()”）上调用。此回调可用于存储已下载的数据。

例如，看一下以下代码：

```py
from ftplib import FTP
from os.path import join

filename = 'raytux.jpg'
path = '/pub/ibiblio/logos/penguins'
destination = open(filename, 'wb')
with FTP('ftp.nluug.nl', 'anonymous') as ftp:
    ftp.retrbinary(
        'RETR {}'.format(join(path, filename)),
        destination.write)
destination.close()
```

每个函数的返回值都是一个包含有关下载的一些统计信息的结果字符串，如下所示：

```py
'226-File successfully transferred\n226 0.000 seconds (measured here), 146.96 Mbytes per second'
```

# 删除或重命名文件

使用“ftplib”删除和重命名文件相对简单。 “delete()”方法只需要一个文件名，并尝试删除服务器上给定的文件。“rename()”方法只需要一个源和目标，并尝试将源重命名为目标名称。

自然地，任何一种方法的成功都取决于登录帐户被授予的权限。

# 将FTP上传添加到GUI

我们的FTP上传功能已经准备就绪，所以让我们将必要的部分添加到我们应用程序的其余部分，使其一起运行。

首先，我们将在“models.py”中的“SettingsModel”中添加FTP主机和端口：

```py
    variables = {
        ...
        'abq_ftp_host': {'type': 'str', 'value': 'localhost'},
        'abq_ftp_port': {'type': 'int', 'value': 2100}
        ...
```

请记住，我们的测试FTP使用端口“2100”，而不是通常的端口“21”，所以现在我们将“2100”作为默认值。

现在，我们将转到“application.py”并创建回调方法，该方法将创建CSV文件并将其传递给FTP上传功能。

在“Application”对象中创建一个新方法：

```py
    def upload_to_corporate_ftp(self):
        csvfile = self._create_csv_extract()
```

我们要做的第一件事是使用我们为“REST”上传创建的方法创建我们的CSV文件。

接下来，我们将要求用户输入FTP用户名和密码：

```py
        d = v.LoginDialog(
            self,
            'Login to ABQ Corporate FTP')
```

现在，我们将调用我们的网络功能：

```py
        if d.result is not None:
            username, password = d.result
            try:
                n.upload_to_corporate_ftp(
                    csvfile,
                    self.settings['abq_ftp_host'].get(),
                    self.settings['abq_ftp_port'].get(),
                    username,
                    password)
```

我们在`try`块中调用FTP上传函数，因为我们的FTP过程可能会引发多个异常。

与其逐个捕获它们，我们可以捕获`ftplib.all_errors`：

```py
            except n.ftp.all_errors as e:
                messagebox.showerror('Error connecting to ftp', str(e))
```

请注意，`ftplib.all_errors`是`ftplib`中定义的所有异常的基类，其中包括认证错误、权限错误和连接错误等。

结束这个方法时，我们将显示一个成功的消息：

```py
            else:
                messagebox.showinfo(
                    'Success',
                    '{} successfully uploaded to FTP'.format(csvfile))
```

写好回调方法后，我们需要将其添加到`callbacks`字典中：

```py
        self.callbacks = {
            ...
            'upload_to_corporate_ftp': self.upload_to_corporate_ftp
        }
```

我们需要做的最后一件事是将我们的回调添加到主菜单类中。

在`mainmenu.py`中，为每个类的`tools_menu`添加一个新的命令：

```py
        tools_menu.add_command(
            label="Upload CSV to corporate FTP",
            command=self.callbacks['upload_to_corporate_ftp'])
```

在终端中启动示例FTP服务器，然后运行你的应用程序并尝试FTP上传。记得输入`test`作为用户名和密码！

你应该会看到一个成功的对话框，类似这样：

![](assets/42dfde62-fea5-4555-ae37-9f3fe4037b68.png)

同样，在你运行示例FTP服务器的目录中应该有一个新的CSV文件。

FTP服务器应该已经打印出了一些类似这样的信息：

```py
127.0.0.1:32878-[] FTP session opened (connect)
127.0.0.1:32878-[test] USER 'test' logged in.
127.0.0.1:32878-[test] STOR /home/alanm/FTPserver/abq_data_record_2018-02-17.csv completed=1 bytes=235 seconds=0.001
127.0.0.1:32878-[test] FTP session closed (disconnect).
```

看起来我们的FTP上传效果很棒！

# 总结

在本章中，我们使用HTTP和FTP与云进行了交互。你学会了如何使用`urllib`下载数据并使用`ElementTree`解析XML。你还了解了`requests`库，并学会了与REST API进行交互的基础知识。最后，我们学会了如何使用Python的`ftplib`下载和上传文件到FTP。
