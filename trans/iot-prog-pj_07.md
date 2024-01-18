# 设置树莓派Web服务器

我们将通过学习如何使用CherryPy web服务器框架来开始创建IoT家庭安全仪表板的旅程。我们的章节将从介绍CherryPy开始。在我们创建一个修改版本的`CurrentWeather`类的HTML天气仪表板之前，我们将通过一些示例进行学习。

本章将涵盖以下主题：

+   介绍CherryPy——一个极简的Python Web框架

+   使用CherryPy创建一个简单的网页

# 完成本章所需的知识

读者应该具有Python的工作知识才能完成本章。还需要基本了解HTML，包括CSS，才能完成本章的项目。

# 项目概述

在本章中，我们将使用CherryPy和Bootstrap框架构建HTML天气仪表板。不需要对这些框架有深入的了解就可以完成项目。

这个项目应该需要几个小时来完成。

# 入门

要完成此项目，需要以下内容：

+   Raspberry Pi Model 3（2015年或更新型号）

+   一个USB电源适配器

+   一个计算机显示器

+   一个USB键盘

+   一个USB鼠标

# 介绍CherryPy——一个极简的Python Web框架

对于我们的项目，我们将使用CherryPy Python库（请注意，它是带有"y"的CherryPy，而不是带有"i"的CherryPi）。

# 什么是CherryPy？

根据他们的网站，CherryPy是一个Pythonic的面向对象的Web框架。CherryPy使开发人员能够构建Web应用程序，就像他们构建任何面向对象的Python程序一样。按照Python的风格，CherryPy程序的代码更少，开发时间比其他Web框架短。

# 谁在使用CherryPy？

一些使用CherryPy的公司包括以下内容：

+   Netflix：Netflix通过RESTful API调用在其基础设施中使用CherryPy。Netflix使用的其他Python库包括Bottle和SciPy。

+   Hulu：CherryPy被用于Hulu的一些项目。

+   Indigo Domotics：Indigo Domotics是一家使用CherryPy框架的家庭自动化公司。

# 安装CherryPy

我们将使用Python的`pip3`软件包管理系统来安装CherryPy。

软件包管理系统是一个帮助安装和配置应用程序的程序。它还可以进行升级和卸载。

要做到这一点，打开一个终端窗口，输入以下内容：

```py
sudo pip3 install cherrypy
```

按下*Enter*。您应该在终端中看到以下内容：

![](assets/2fe0a193-759c-49b0-9c82-57ce1f5766dc.png)

在Thonny中，转到工具|管理包。您应该看到CherryPy现在已安装，如下所示：

![](assets/7f7232a1-86e0-4f95-ab6b-c8c9c3cb45a2.png)

# 使用CherryPy创建一个简单的网页

让我们开始，让我们用CherryPy构建最基本的程序。我指的当然是无处不在的`Hello World`程序，我们将用它来说`Hello Raspberry Pi!`。在我们构建一个仪表板来显示天气数据之前，我们将通过一些示例进行学习，使用[第4章](626664bb-0130-46d1-b431-682994472fc1.xhtml)中的`CurrentWeather`类的修改版本，*订阅Web服务*。

# Hello Raspberry Pi!

要构建`Hello Raspberry Pi!`网页，执行以下操作：

1.  从应用程序菜单|编程|Thonny Python IDE中打开Thonny。

1.  单击新图标创建一个新文件。

1.  输入以下内容：

```py
import cherrypy

class HelloWorld():

     @cherrypy.expose
     def index(self):
         return "Hello Raspberry Pi!"

cherrypy.quickstart(HelloWorld())

```

1.  确保行`cherrypy.quickstart(HelloWorld())`与`import`和`class`语句一致。

1.  将文件保存为`HelloRaspberryPi.py`。

1.  点击绿色的`Run current script`按钮运行文件。

1.  您应该看到CherryPy web服务器正在终端中启动：

![](assets/ecbc3a89-f763-46b3-bbde-b32f6e46aa04.png)

1.  从终端输出中，您应该能够观察到CherryPy正在运行的IP地址和端口，`http://127.0.0.1:8080`。您可能会认出IP地址是环回地址。CherryPy使用端口`8080`。

1.  在树莓派上打开一个网络浏览器，并在地址栏中输入上一步的地址：

![](assets/8f293187-5738-4253-8d25-32b1172d5aac.png)

恭喜，您刚刚将您的谦卑的树莓派变成了一个网络服务器。

如果您和我一样，您可能没有想到一个网络服务器可以用如此少的代码创建。CherryPy基本上专注于一个任务，那就是接收HTTP请求并将其转换为Python方法。

它是如何工作的呢？我们的`HelloWorld`类中的装饰器`@cherrypy.expose`公开了恰好对应于网络服务器根目录的`index`方法。当我们使用回环地址（`127.0.0.1`）和CherryPy正在运行的端口（`8080`）加载我们的网页时，`index`方法将作为页面提供。在我们的代码中，我们只是返回字符串`Hello Raspberry Pi!`，然后它就会显示为我们的网页。

回环地址是用作机器软件回环接口的IP号码。这个号码通常是`127.0.0.1`。这个地址并没有物理连接到网络，通常用于测试安装在同一台机器上的网络服务器。

# 向myFriend问好

那么，如果我们在Python代码中公开另一个方法会发生什么呢？我们可以通过在方法之前使用装饰器轻松地检查到这一点。让我们编写一些代码来做到这一点：

1.  从应用程序菜单 | 编程 | Thonny Python IDE中打开Thonny。

1.  单击新建图标创建一个新文件。

1.  输入以下内容：

```py
import cherrypy

class HelloWorld():

     @cherrypy.expose
     def index(self):
         return "Hello Raspberry Pi!"

     @cherrypy.expose
     def sayHello(self, myFriend=" my friend"):
         return "Hello " + myFriend

cherrypy.quickstart(HelloWorld())
```

1.  将文件保存为`SayHello.py`。

1.  通过单击中断/重置按钮，然后单击运行当前脚本按钮来停止和启动CherryPy服务器。

1.  现在，输入以下内容到您的浏览器的地址栏中，然后按*Enter*：`http://127.0.0.1:8080/sayHello`

1.  您应该看到以下内容：

![](assets/41a50c34-5bce-48d0-817a-40f6cfed119b.png)

这次我们做了什么不同的事情？首先，我们不只是访问了服务器的根目录。我们在URL中添加了`/sayHello`。通常，当我们在网络服务器上这样做时，我们会被引导到一个子文件夹。在这种情况下，我们被带到了我们的`HelloWorld`类中的`sayHello()`方法。

如果我们仔细看`sayHello()`方法，我们会发现它接受一个名为`myFriend`的参数：

```py
@cherrypy.expose
def sayHello(self, myFriend=" my friend"):
         return "Hello " + myFriend
```

我们可以看到`myFriend`参数的默认值是`my Friend`。因此，当我们运行CherryPy并导航到`http://127.0.0.1:8080/sayHello`的URL时，将调用`sayHello`方法，并返回`"Hello " + my friend`字符串。

现在，将以下内容输入到地址栏中，然后按*Enter*：`http://127.0.0.1:8080/sayHello?myFriend=Raspberry%20Pi`

在这个URL中，我们将`myFriend`的值设置为`Raspberry%20Pi`（使用`%20`代替空格）。我们应该得到与我们的第一个示例相同的结果。

正如我们所看到的，将Python方法连接到HTML输出非常容易。

# 静态页面呢？

静态页面曾经是互联网上随处可见的。静态页面之间的简单链接构成了当时被认为是一个网站的内容。然而，自那时以来发生了很多变化，但是能够提供一个简单的HTML页面仍然是网络服务器框架的基本要求。

那么，我们如何在CherryPy中做到这一点呢？实际上很简单。我们只需在一个方法中打开一个静态HTML页面并返回它。让我们通过以下方式让CherryPy提供一个静态页面：

1.  从应用程序菜单 | 编程 | Thonny Python IDE中打开Thonny。

1.  单击新建图标创建一个新文件。

1.  输入以下内容：

```py
<html>
    <body>
        This is a static HTML page.
    </body>
</html>
```

1.  将文件保存为`static.html`。

1.  在Thonny中点击新建图标，在与`static.html`相同的目录中创建一个新文件。

1.  输入以下内容：

```py
import cherrypy

class StaticPage():

     @cherrypy.expose
     def index(self):
         return open('static.html')

cherrypy.quickstart(StaticPage())
```

1.  将文件保存为`StaticPage.py`。

1.  如果CherryPy仍在运行，请单击红色按钮停止它。

1.  运行文件`StaticPage.py`以启动CherryPy。

1.  您应该看到CherryPy正在启动，如终端中所示。

1.  要查看我们的新静态网页，请在树莓派上打开一个网络浏览器，并在地址栏中输入以下内容：`http://127.0.0.1:8080`

1.  您应该看到静态页面显示出来了：

![](assets/6418ac8e-b97b-49d3-9c72-7edc6d83348f.png)

那么我们在这里做了什么呢？我们修改了我们的`index`方法，使其返回一个打开的`static.html`文件，带有`return open('static.html')`这一行。这将在我们的浏览器中打开`static.html`作为我们的索引（或`http://127.0.0.1:8080/index`）。请注意，尝试在url中输入页面名称`static.html`（`http://127.0.0.1:8080/static.html`）将不起作用。CherryPy根据方法名提供内容。在这种情况下，方法名是index，这是默认值。

# HTML天气仪表板

现在是时候添加我们从之前章节学到的知识了。让我们重新访问[第4章](626664bb-0130-46d1-b431-682994472fc1.xhtml)中的`CurrentWeather`类，*订阅Web服务*。我们将把它重命名为`WeatherData`，因为这个名字更适合这个项目，并稍微修改一下。

1.  从应用程序菜单 | 编程 | Thonny Python IDE中打开Thonny

1.  单击新图标创建一个新文件

1.  输入以下内容：

```py
from weather import Weather, Unit
import time

class WeatherData:

    temperature = 0
    weather_conditions = ''
    wind_speed = 0
    city = ''

    def __init__(self, city):
        self.city = city
        weather = Weather(unit = Unit.CELSIUS)
        lookup = weather.lookup_by_location(self.city)
        self.temperature = lookup.condition.temp
        self.weather_conditions = lookup.condition.text
        self.wind_speed = lookup.wind.speed

    def getTemperature(self):
        return self.temperature + " C"

    def getWeatherConditions(self):
        return self.weather_conditions

    def getWindSpeed(self):
        return self.wind_speed + " kph"

    def getCity(self):
        return self.city

    def getTime(self):
        return time.ctime()

if __name__ == "__main__":

    current_weather = WeatherData('London')
    print(current_weather.getTemperature())
    print(current_weather.getWeatherConditions())
    print(current_weather.getTime())
```

1.  将文件保存为`WeatherData.py`

1.  运行代码

1.  您应该在以下shell中看到伦敦，英格兰的天气：

![](assets/88d68f15-1946-4c56-9a03-d4549b87e934.png)

让我们来看看代码。基本上`WeatherData.py`与[第4章](626664bb-0130-46d1-b431-682994472fc1.xhtml)中的`CurrentWeather.py`完全相同，但多了一个名为`getTime`的额外方法：

```py
def getTime(self):
    return time.ctime()
```

我们使用这种方法返回调用天气Web服务时的时间，以便在我们的网页中使用。

我们现在将使用CherryPy和[Bootstrap](https://getbootstrap.com)框架来创建我们的仪表板。要做到这一点，请执行以下操作：

1.  从应用程序菜单 | 编程 | Thonny Python IDE中打开Thonny

1.  单击新图标创建一个新文件

1.  输入以下内容（特别注意引号）：

```py
import cherrypy
from WeatherData import WeatherData

class WeatherDashboardHTML:

    def __init__(self, currentWeather):
        self.currentWeather = currentWeather

    @cherrypy.expose
    def index(self):
        return """
               <!DOCTYPE html>
                <html lang="en">

                <head>
                    <title>Weather Dashboard</title>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.1.0/css/bootstrap.min.css">
                    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.0/umd/popper.min.js"></script>
                    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.1.0/js/bootstrap.min.js"></script>
                    <style>
                        .element-box {
                            border-radius: 10px;
                            border: 2px solid #C8C8C8;
                            padding: 20px;
                        }

                        .card {
                            width: 600px;
                        }

                        .col {
                            margin: 10px;
                        }
                    </style>
                </head>

                <body>
                    <div class="container">
                        <br/>
                        <div class="card">
                            <div class="card-header">
                                <h3>Weather Conditions for """ + self.currentWeather.getCity() + """
                                </h3></div>
                             <div class="card-body">
                                <div class="row">
                                    <div class="col element-box">
                                        <h5>Temperature</h5>
                                        <p>""" + self.currentWeather.getTemperature() + """</p>
                                    </div>
                                    <div class="col element-box">
                                        <h5>Conditions</h5>
                                        <p>""" + self.currentWeather.getWeatherConditions() + """</p>
                                    </div>
                                    <div class="col element-box">
                                        <h5>Wind Speed</h5>
                                        <p>""" + self.currentWeather.getWindSpeed() + """</p>
                                    </div>
                                </div>
                            </div>
                            <div class="card-footer"><p>""" + self.currentWeather.getTime() + """</p></div>
                        </div>
                    </div>
                </body>

                </html>
               """

if __name__=="__main__":
    currentWeather = WeatherData('Paris')
    cherrypy.quickstart(WeatherDashboardHTML(currentWeather))
```

1.  将文件保存为`WeatherDashboardHTML.py`

这可能看起来是一大堆代码 - 而且确实是。不过，如果我们把它分解一下，实际上并不是那么复杂。基本上，我们使用CherryPy返回一个HTML字符串，这个字符串将通过`index`方法在我们的URL根目录中提供。

在我们可以这样做之前，我们通过传入一个`WeatherData`对象来实例化`WeatherDashboardHTML`类。我们给这个`WeatherData`对象起名为`currentWeather`，就像`init`（类构造函数）方法中所示的那样：

```py
def __init__(self, currentWeather):
         self.currentWeather = currentWeather
```

CherryPy通过打印一个HTML字符串来提供`index`方法，该字符串中包含来自我们`currentWeather`对象的参数。我们在我们的HTML代码中使用了Bootstrap组件库。我们通过合并标准的Bootstrap样板代码来添加它：

```py
<link rel="stylesheet"href="https://maxcdn.bootstrapcdn.com
        /bootstrap/4.1.0/css/bootstrap.min.css">

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.0/umd/popper.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.1.0/js/bootstrap.min.js"></script>
```

我们使用Bootstrap的`card`组件作为我们的内容容器。`card`允许我们创建一个标题、正文和页脚：

```py
<div class="card">
    <div class="card-header">
        .
        .
        .
```

`card`组件的标题部分显示了城市的名称。我们使用我们的`currentWeather`对象的`getCity`方法来获取城市的名称。

```py
<div class="card-header">
    <h3>Weather Conditions for """ + self.currentWeather.getCity() + """</h3>
</div>
```

在`card`组件的正文部分，我们创建了一个具有三列的行。每列包含一个标题（`<h5>`），以及从我们的`WeatherData`对象`currentWeather`中提取的数据。您可以看到标题`Temperature`，以及从`currentWeather`方法`getTemperature`中提取的温度值：

```py
<div class="card-body">
    <div class="row">
        <div class="col element-box">
            <h5>Temperature</h5>
            <p>""" + self.currentWeather.getTemperature() + """</p>
        .
        .
        .
```

对于页脚，我们只需返回`currentWeather`对象的实例化时间。我们将这个时间作为我们的程序从中检查天气信息的时间。

```py
<div class="card-footer">
    <p>""" + self.currentWeather.getTime() + """</p>
</div>
```

我们在顶部的样式部分允许我们自定义我们的仪表板的外观。我们创建了一个CSS类，称为`element-box`，以便在我们的天气参数周围创建一个银色（`#C8C8C8`）的圆角框。我们还限制了卡片（因此也是仪表板）的宽度为`600px`。最后，我们在列周围放置了`10px`的边距，以便圆角框不会彼此接触：

```py
<style>
    .element-box {
        border-radius: 10px;
        border: 2px solid #C8C8C8;
        padding: 20px;
    }

    .card {
        width: 600px;
    }

    .col {
        margin: 10px;
    }

</style>
```

我们在底部的`main`方法中将`WeatherData`类实例化为一个名为`currentWeather`的对象。在我们的示例中，我们使用来自`Paris`城市的数据。然后我们的代码将`currentWeather`对象传递给`cherrypy.quickstart()`方法，如下所示：

```py
if __name__=="__main__":
     currentWeather = WeatherData('Paris')
     cherrypy.quickstart(WeatherDashboardHTML(currentWeather))
```

在`WeatherDashboardHTML.py`文件上停止和启动CherryPy服务器。如果您的代码没有任何错误，您应该会看到类似以下的内容：

![](assets/ce7db347-f9c2-4e2b-a938-903edd69a563.png)

# 摘要

在本章中，我们使用CherryPy HTTP框架将我们的树莓派变成了一个Web服务器。凭借其简约的架构，CherryPy允许开发者在很短的时间内建立一个支持Web的Python程序。

我们在本章开始时在树莓派上安装了CherryPy。经过几个简单的示例后，我们通过修改和利用我们在[第4章](626664bb-0130-46d1-b431-682994472fc1.xhtml)中编写的Web服务代码，构建了一个HTML天气仪表盘。

在接下来的章节中，我们将利用本章学到的知识来构建一个物联网家庭安全仪表盘。

# 问题

1.  True或false？它是CherryPi，而不是CherryPy。

1.  True或false？Netflix使用CherryPy。

1.  我们如何告诉CherryPy我们想要公开一个方法？

1.  True或false？CherryPy需要很多样板代码。

1.  我们为什么将我们的`CurrentWeather`类重命名为`WeatherData`？

1.  True或false？CherryPy使用的默认端口是`8888`。

1.  我们为什么要向我们的`col` CSS类添加边距？

1.  我们从`WeatherData`类中使用哪个方法来获取当前天气状况的图片URL？

1.  我们使用哪个Bootstrap组件作为我们的内容容器？

1.  True或false？在我们的示例中，伦敦是晴天和炎热的。

# 更多阅读

在本章中，我们只是浅尝了一下CherryPy和Bootstrap框架。更多阅读材料可以在CherryPy网站上找到，网址为[www.cherrypy.org](http://www.cherrypy.org)，以及Bootstrap的网站，网址为[https://getbootstrap.com](https://getbootstrap.com)。建议开发者通过阅读来提高对这些强大框架的了解。
