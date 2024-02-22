# 第四章：订阅 Web 服务

我们许多人都认为互联网建立在其上的技术是理所当然的。当我们访问我们喜爱的网站时，我们很少关心我们正在查看的网页是为我们的眼睛而制作的。然而，在底层是通信协议的互联网协议套件。机器也可以利用这些协议，通过 Web 服务进行机器之间的通信。

在本章中，我们将继续我们的连接设备通过**物联网**（**IoT**）的旅程。我们将探索 Web 服务和它们背后的各种技术。我们将以一些 Python 代码结束我们的章节，其中我们调用一个实时天气服务并提取信息。

本章将涵盖以下主题：

+   物联网的云服务

+   编写 Python 程序提取实时天气数据

# 先决条件

读者应该具有 Python 编程语言的工作知识，以完成本章，以及对基本面向对象编程的理解。这将为读者服务良好，因为我们将把我们的代码分成对象。

# 项目概述

在这个项目中，我们将探索各种可用的 Web 服务，并涉及它们的核心优势。然后，我们将编写调用 Yahoo! Weather Web 服务的代码。最后，我们将使用树莓派 Sense HAT 模拟器显示实时天气数据的“滚动”显示。

本章应该需要一个上午或下午来完成。

# 入门

要完成这个项目，需要以下内容：

+   树莓派 3 型号（2015 年或更新型号）

+   USB 电源适配器

+   计算机显示器（支持 HDMI）

+   USB 键盘

+   USB 鼠标

+   互联网接入

# 物联网的云服务

有许多云服务可供我们用于物联网开发。一些科技界最大的公司已经全力支持物联网，特别是具有人工智能的物联网。

以下是一些这些服务的详细信息。

# 亚马逊网络服务 IoT

亚马逊网络服务 IoT 是一个云平台，允许连接设备与其他设备或云应用程序安全地交互。这些是按需付费的服务，无需服务器，从而简化了部署和可扩展性。

**亚马逊网络服务**（**AWS**）的服务，AWS IoT 核心可以使用如下：

+   AWS Lambda

+   亚马逊 Kinesis

+   亚马逊 S3

+   亚马逊机器学习

+   亚马逊 DynamoDB

+   亚马逊 CloudWatch

+   AWS CloudTrail

+   亚马逊 Elasticsearch 服务

AWS IoT 核心应用程序允许收集、处理和分析由连接设备生成的数据，无需管理基础设施。定价是按发送和接收的消息计费。

以下是 AWS IoT 的使用示意图。在这种情况下，汽车的道路状况数据被发送到云端并存储在 S3 云存储服务中。AWS 服务将这些数据广播给其他汽车，警告它们可能存在危险的道路状况：

![](img/ec770430-be8c-48e1-b7d6-52bc9189240e.png)

# IBM Watson 平台

IBM Watson 是一个能够用自然语言回答问题的系统。最初设计用来参加电视游戏节目*Jeopardy!*，Watson 以 IBM 的第一任 CEO Thomas J. Watson 的名字命名。2011 年，Watson 挑战了*Jeopardy!*冠军 Brad Rutter 和 Ken Jennings 并获胜。

使用 IBM Watson 开发者云的应用程序可以通过 API 调用来创建。使用 Watson 处理物联网信息的潜力是巨大的。

直白地说，Watson 是 IBM 的一台超级计算机，可以通过 API 调用在网上访问。

Watson 与 IoT 的一个应用是 IBM Watson 助手汽车版，这是为汽车制造商提供的集成解决方案。通过这项技术，驾驶员和乘客可以与外界互动，例如预订餐厅和检查日历中的约会。车辆中的传感器可以集成，向 IBM Watson 助手提供车辆状态的信息，如轮胎压力。以下是一个图表，说明了 Watson 如何警告驾驶员轮胎压力过低，建议修理，并预约车库：

![](img/aae158bc-7e55-4d14-af47-1db370364d77.png)

IBM Watson 助手汽车版作为白标服务出售，以便制造商可以将其标记为适合自己的需求。IBM Watson 助手汽车版的成功将取决于它与亚马逊的 Alexa 和谷歌的 AI 助手等其他 AI 助手服务的竞争情况。与 Spotify 音乐和亚马逊购物等热门服务的整合也将在未来的成功中发挥作用。

# 谷歌云平台

虽然谷歌的 IoT 并不像 AWS IoT 那样广泛和有文档记录，但谷歌对 IoT 的兴趣很大。开发人员可以通过使用谷歌云服务来利用谷歌的处理、分析和机器智能技术。

以下是谷歌云服务提供的一些服务列表：

+   App Engine：应用程序托管服务

+   BigQuery：大规模数据库分析服务

+   Bigtable：可扩展的数据库服务

+   Cloud AutoML：允许开发人员访问谷歌神经架构搜索技术的机器学习服务

+   云机器学习引擎：用于 TensorFlow 模型的机器学习服务

+   谷歌视频智能：分析视频并创建元数据的服务

+   云视觉 API：通过机器学习返回图像数据的服务

以下是谷歌云视觉 API 的使用图表。一张狗站在一个颠倒的花盆旁边的图片通过 API 传递给服务。图像被扫描，并且使用机器学习在照片中识别物体。返回的 JSON 文件包含结果的百分比：

![](img/1699693d-94f5-49ec-91f7-180558df180d.png)

谷歌专注于使事情简单快捷，使开发人员可以访问谷歌自己的全球私人网络。谷歌云平台的定价低于 AWS IoT。

# 微软 Azure

微软 Azure（以前称为 Windows Azure）是微软的基于云的服务，允许开发人员使用微软庞大的数据中心构建、测试、部署和管理应用程序。它支持许多不同的编程语言，既是微软特有的，也来自外部第三方。

Azure Sphere 是微软 Azure 框架的一部分，于 2018 年 4 月推出，是 Azure 的 IoT 解决方案。以下是 Azure Sphere（或如图表所示的 Azure IoT）可能被使用的场景。在这种情况下，远程工厂中的机器人手臂通过手机应用程序进行监控和控制：

![](img/71cf93b4-2b0e-46e1-a77b-190e426a4551.png)

您可能已经注意到，前面的例子可以使用任何竞争对手的云服务来设置，这确实是重点。通过相互竞争，服务变得更好、更便宜，因此更易获得。

随着 IBM、亚马逊、谷歌和微软等大公司参与 IoT 数据处理，IoT 的未来是无限的。

# Weather Underground

虽然不像谷歌和 IBM 那样重量级，Weather Underground 提供了一个天气信息的网络服务，开发人员可以将他们的应用程序与之联系起来。通过使用开发人员账户，可以构建利用当前天气条件的 IoT 应用程序。

在撰写本章时，Weather Underground 网络为开发人员提供了 API 以访问天气信息。自那时起，Weather Underground API 网站发布了服务终止通知。要了解此服务的状态，请访问[`www.wunderground.com/weather/api/`](https://www.wunderground.com/weather/api/)。

# 从云中提取数据的基本 Python 程序

在第二章中，*使用树莓派编写 Python 程序*，我们介绍了一个名为`weather-api`的包，它允许我们访问 Yahoo! Weather Web 服务。在本节中，我们将在我们自己的类中包装`weather-api`包中的`Weather`对象。我们将重用我们的类名称`CurrentWeather`。在测试我们的`CurrentWeather`类之后，我们将在 Raspbian 中利用 Sense Hat 模拟器并构建一个天气信息滚动条。

# 访问 Web 服务

我们将首先修改我们的`CurrentWeather`类，以通过`weather-api`包对 Yahoo! Weather 进行 Web 服务调用：

1.  从应用程序菜单|编程|Thonny Python IDE 打开 Thonny。

1.  单击新图标创建新文件。

1.  输入以下内容：

```py
from weather import Weather, Unit

class CurrentWeather:
     temperature = ''
     weather_conditions = ''
     wind_speed = ''
     city = ''

     def __init__(self, city):
         self.city = city
         weather = Weather(unit = Unit.CELSIUS)
         lookup = weather.lookup_by_location(self.city)
         self.temperature = lookup.condition.temp
         self.weather_conditions = lookup.condition.text
         self.wind_speed = lookup.wind.speed

     def getTemperature(self):
         return self.temperature

     def getWeatherConditions(self):
         return self.weather_conditions

     def getWindSpeed(self):
         return self.wind_speed

     def getCity(self):
         return self.city

if __name__=="__main__":
        current_weather = CurrentWeather('Montreal')
        print("%s %sC %s wind speed %s km/h"
        %(current_weather.getCity(),
        current_weather.getTemperature(),
        current_weather.getWeatherConditions(),
        current_weather.getWindSpeed()))
```

1.  将文件保存为`CurrentWeather.py`。

1.  运行代码。

1.  您应该在 Thonny 的 shell 中看到来自 Web 服务的天气信息打印出来。当我运行程序时，我看到了以下内容：

```py
Toronto 12.0C Clear wind speed 0 km/h
```

1.  现在，让我们仔细看看代码，看看发生了什么。我们首先从我们需要的程序包中导入资源：

```py
from weather import Weather, Unit
```

1.  然后我们定义我们的类名`CurrentWeather`，并将类变量（`temperature`、`weather_conditions`、`wind_speed`和`city`）设置为初始值：

```py
class CurrentWeather:
     temperature = ''
     weather_conditions = ''
     wind_speed = ''
     city = ''
```

1.  在`init`方法中，我们根据传入方法的`city`设置我们的类变量。我们通过将一个名为`weather`的变量实例化为`Weather`对象，并将`unit`设置为`CELSIUS`来实现这一点。`lookup`变量是基于我们传入的`city`名称创建的。从那里，简单地设置我们的类变量（`temperature`、`weather_conditions`和`wind_speed`）从我们从`lookup`中提取的值。`weather-api`为我们完成了所有繁重的工作，因为我们能够使用点表示法访问值。我们无需解析 XML 或 JSON 数据：

```py
def __init__(self, city):
    self.city = city
    weather = Weather(unit = Unit.CELSIUS)
    lookup = weather.lookup_by_location(self.city)
    self.temperature = lookup.condition.temp
    self.weather_conditions = lookup.condition.text
     self.wind_speed = lookup.wind.speed
```

1.  在`init`方法中设置类变量后，我们使用方法调用来返回这些类变量：

```py
def getTemperature(self):
    return self.temperature

def getWeatherConditions(self):
    return self.weather_conditions

def getWindSpeed(self):
    return self.wind_speed

def getCity(self):
    return self.city
```

1.  由于我们在 Thonny 中作为程序运行`CurrentWeather.py`，我们可以使用`if __name__=="__main__"`方法并利用`CurrentWeather`类。请注意，`if __name__=="__main__"`方法的缩进与类名相同。如果不是这样，它将无法工作。

在 Python 的每个模块中，都有一个名为`__name__`的属性。如果您要检查已导入到程序中的模块的此属性，您将得到返回的模块名称。例如，如果我们在前面的代码中放置行`print(Weather.__name__)`，我们将得到名称`Weather`。在运行文件时检查`__name__`返回`__main__`值。

1.  在`if __name__=="__main__"`方法中，我们创建一个名为`current_weather`的`CurrentWeather`类型的对象，传入城市名`Montreal`。然后，我们使用适当的方法调用打印出`city`、`temperature`、`weather conditions`和`wind speed`的值：

```py
if __name__=="__main__":
    current_weather = CurrentWeather('Montreal')
    print("%s %sC %s wind speed %s km/h"
    %(current_weather.getCity(),
    current_weather.getTemperature(),
    current_weather.getWeatherConditions(),
    current_weather.getWindSpeed()))
```

# 使用 Sense HAT 模拟器

现在，让我们使用树莓派 Sense HAT 模拟器来显示天气数据。我们将利用我们刚刚创建的`CurrentWeather`类。要在 Sense HAT 模拟器中看到显示的天气信息，请执行以下操作：

1.  从应用程序菜单|编程|Thonny Python IDE 打开 Thonny

1.  单击新图标创建新文件

1.  输入以下内容：

```py
from sense_emu import SenseHat
from CurrentWeather import CurrentWeather

class DisplayWeather:
    current_weather = ''

    def __init__(self, current_weather):
        self.current_weather = current_weather

    def display(self):
        sense_hat_emulator = SenseHat()

        message = ("%s %sC %s wind speed %s km/h"
           %(self.current_weather.getCity(),
           self.current_weather.getTemperature(),
           self.current_weather.getWeatherConditions(),
           self.current_weather.getWindSpeed()))

        sense_hat_emulator.show_message(message)

if __name__ == "__main__":
    current_weather = CurrentWeather('Toronto')
    display_weather = DisplayWeather(current_weather)
    display_weather.display()
```

1.  将文件保存为`DisplayWeather.py`

1.  从应用程序菜单|编程|Sense HAT 模拟器加载 Sense HAT 模拟器

1.  将 Sense HAT 模拟器定位到可以看到显示的位置

1.  运行代码

你应该在 Sense HAT 模拟器显示器上看到`多伦多`的天气信息滚动条，类似于以下截图：

![](img/9cfaaa47-606f-426e-a44c-1c86b418d7dd.png)

那么，我们是如何做到这一点的呢？`init`和`message`方法是这个程序的核心。我们通过设置类变量`current_weather`来初始化`DisplayWeather`类。一旦`current_weather`被设置，我们就在`display`方法中从中提取值，以便构建我们称之为`message`的消息。然后我们也在`display`方法中创建一个`SenseHat`模拟器对象，并将其命名为`sense_hat_emulator`。我们通过`sense_hat_emulator.show_message(message)`这一行将我们的消息传递给`SenseHat`模拟器的`show_message`方法：

```py
def __init__(self, current_weather):
    self.current_weather = current_weather

def display(self):
    sense_hat_emulator = SenseHat()

    message = ("%s %sC %s wind speed %s km/h"
           %(self.current_weather.getCity(),
           self.current_weather.getTemperature(),
           self.current_weather.getWeatherConditions(),
           self.current_weather.getWindSpeed()))

    sense_hat_emulator.show_message(message)
```

# 总结

我们从讨论一些可用的各种网络服务开始了本章。我们讨论了一些在人工智能和物联网领域中最大的信息技术公司的工作。

亚马逊和谷歌都致力于成为物联网设备连接的平台。亚马逊通过其亚马逊网络服务提供了大量的文档和支持。谷歌也在建立一个强大的物联网平台。哪个平台会胜出还有待观察。

IBM 在人工智能领域的涉足集中在 Watson 上，他们的*Jeopardy!*游戏冠军。当然，赢得游戏秀并不是 Watson 的最终目标。然而，从这些追求中建立的知识和技术将会进入我们今天只能想象的领域。Watson 可能会被证明是物联网世界的所谓杀手应用程序。

也许没有什么比天气更多人谈论的了。在本章中，我们使用`weather-api`包利用内置在 Raspbian 操作系统中的树莓派 Sense HAT 模拟器构建了一个天气信息滚动条。

在第五章中，*使用 Python 控制舵机*，我们将探索使用舵机以提供模拟显示的其他通信方式。

# 问题

1.  IBM Watson 是什么？

1.  真的还是假的？亚马逊的物联网网络服务允许访问亚马逊的其他基于云的服务。

1.  真的还是假的？Watson 是*Jeopardy!*游戏秀的冠军吗？

1.  真的还是假的？谷歌有他们自己的全球私人网络。

1.  真的还是假的？当我们引入网络服务数据时，我们需要更改函数的名称，比如`getTemperature`。

1.  真的还是假的？在你的类中使用测试代码以隔离该类的功能是一个好主意。

1.  在我们的代码中，`DisplayWeather`类的目的是什么？

1.  在我们的代码中，我们使用`SenseHat`对象的哪种方法来在 Sense HAT 模拟器中显示天气信息？

# 进一步阅读

在扩展你对网络服务的知识时，通过谷歌搜索可用的各种网络服务是一个很好的起点。
