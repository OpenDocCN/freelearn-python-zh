# 请求和Web框架

本章的主要内容是Python中的请求和Web框架。我们将讨论使得从Web检索数据（例如，获取天气更新）、将数据上传到远程服务器（例如，记录传感器数据）或控制本地网络上的设备成为可能的库和框架。我们还将讨论一些有助于学习本章核心主题的话题。

# `try`/`except`关键字

到目前为止，我们已经审查并测试了所有的例子，假设程序的执行不会遇到错误。相反，应用程序有时会由于外部因素（如无效的用户输入和糟糕的互联网连接）或程序员造成的程序逻辑错误而失败。在这种情况下，我们希望程序报告/记录错误的性质，并在退出程序之前继续执行或清理资源。`try`/`except`关键字提供了一种机制，可以捕获程序执行过程中发生的错误并采取补救措施。由于可能在代码的关键部分捕获和记录错误，`try`/`except`关键字在调试应用程序时特别有用。

通过比较两个例子来理解`try`/`except`关键字。让我们构建一个简单的猜数字游戏，用户被要求猜一个0到9之间的数字：

1.  使用Python的`random`模块生成一个随机数（在0到9之间）。如果用户猜测的数字正确，Python程序会宣布用户为赢家并退出游戏。

1.  如果用户输入是字母`x`，程序会退出游戏。

1.  用户输入使用`int()`函数转换为整数。进行了一个合理性检查，以确定用户输入是否是0到9之间的数字。

1.  整数与随机数进行比较。如果它们相同，程序会宣布用户为赢家并退出游戏。

让我们观察当我们故意向这个程序提供错误的输入时会发生什么（这里显示的代码片段可以在本章的下载中找到，文件名为`guessing_game.py`）：

```py
import random

if __name__ == "__main__":
    while True:
        # generate a random number between 0 and 9
        rand_num = random.randrange(0,10)

        # prompt the user for a number
        value = input("Enter a number between 0 and 9: ")

        if value == 'x':
            print("Thanks for playing! Bye!")
            break

        input_value = int(value)

        if input_value < 0 or input_value > 9:
            print("Input invalid. Enter a number between 0 and 9.")

        if input_value == rand_num:
            print("Your guess is correct! You win!")
            break
        else:
            print("Nope! The random value was %s" % rand_num)
```

让我们执行前面的代码片段，并向程序提供输入`hello`：

```py
    Enter a number between 0 and 9: hello
 Traceback (most recent call last):
 File "guessing_game.py", line 12, in <module>
 input_value = int(value)
 ValueError: invalid literal for int() with base 10: 'hello'
```

在前面的例子中，当程序试图将用户输入`hello`转换为整数时失败。程序执行以异常结束。异常突出了发生错误的行。在这种情况下，它发生在第10行：

```py
    File "guessing_game.py", line 12, in <module>
 input_value = int(value)
```

异常的性质也在异常中得到了突出。在这个例子中，最后一行表明抛出的异常是`ValueError`：

```py
    ValueError: invalid literal for int() with base 10: 'hello'
```

让我们讨论一个相同的例子（可以在本章的下载中找到，文件名为`try_and_except.py`），它使用了`try`/`except`关键字。在捕获异常并将其打印到屏幕后，可以继续玩游戏。我们有以下代码：

```py
import random

if __name__ == "__main__":
    while True:
        # generate a random number between 0 and 9
        rand_num = random.randrange(0,10)

        # prompt the user for a number
        value = input("Enter a number between 0 and 9: ")

        if value == 'x':
            print("Thanks for playing! Bye!")

        try:
            input_value = int(value)
        except ValueError as error:
            print("The value is invalid %s" % error)
            continue

        if input_value < 0 or input_value > 9:
            print("Input invalid. Enter a number between 0 and 9.")
            continue

        if input_value == rand_num:
            print("Your guess is correct! You win!")
            break
        else:
            print("Nope! The random value was %s" % rand_num)
```

让我们讨论相同的例子如何使用`try`/`except`关键字：

1.  从前面的例子中，我们知道当用户提供错误的输入时（例如，一个字母而不是0到9之间的数字），异常发生在第10行（用户输入转换为整数的地方），错误的性质被命名为`ValueError`。

1.  可以通过将其包装在`try...except`块中来避免程序执行的中断：

```py
      try: 
         input_value = int(value) 
      except ValueError as error:
         print("The value is invalid %s" % error)
```

1.  在接收到用户输入时，程序会在`try`块下尝试将用户输入转换为整数。

1.  如果发生了`ValueError`，`except`块会捕获`error`，并将以下消息与实际错误消息一起打印到屏幕上：

```py
       except ValueError as error:
           print("The value is invalid %s" % error)
```

1.  尝试执行代码示例并提供无效输入。您会注意到程序打印了错误消息（以及错误的性质），然后返回游戏循环的顶部并继续寻找有效的用户输入：

```py
       Enter a number between 0 and 9: 3
 Nope! The random value was 5
 Enter a number between 0 and 9: hello
 The value is invalid invalid literal for int() with
       base 10: 'hello'
 Enter a number between 0 and 10: 4
 Nope! The random value was 6
```

`try...except`块带来了相当大的处理成本。因此，将`try...except`块保持尽可能短是很重要的。因为我们知道错误发生在尝试将用户输入转换为整数的行上，所以我们将其包装在`try...except`块中以捕获错误。

因此，`try`/`except`关键字用于防止程序执行中的任何异常行为，因为出现错误。它使得能够记录错误并采取补救措施。与`try...except`块类似，还有`try...except...else`和`try...except...else`代码块。让我们通过几个例子快速回顾一下这些选项。

# try...except...else

`try...except...else`块在我们希望只有在没有引发异常时才执行特定代码块时特别有用。为了演示这个概念，让我们使用这个块来重写猜数字游戏示例：

```py
try:
    input_value = int(value)
except ValueError as error:
    print("The value is invalid %s" % error)
else:
    if input_value < 0 or input_value > 9:
        print("Input invalid. Enter a number between 0 and 9.")
    elif input_value == rand_num:
        print("Your guess is correct! You win!")
        break
    else:
        print("Nope! The random value was %s" % rand_num)
```

使用`try...except...else`块修改的猜数字游戏示例可与本章一起下载，文件名为`try_except_else.py`。在这个例子中，程序仅在接收到有效的用户输入时才将用户输入与随机数进行比较。否则，它会跳过`else`块并返回到循环顶部以接受下一个用户输入。因此，当`try`块中的代码没有引发异常时，`try...except...else`被用来执行特定的代码块。

# try...except...else...finally

正如其名称所示，`finally`块用于在离开`try`块时执行一块代码。即使在引发异常后，这段代码也会被执行。这在我们需要在进入下一个阶段之前清理资源和释放内存时非常有用。

让我们使用我们的猜数字游戏来演示`finally`块的功能。为了理解`finally`关键字的工作原理，让我们使用一个名为`count`的计数器变量，在`finally`块中递增，以及另一个名为`valid_count`的计数器变量，在`else`块中递增。我们有以下代码：

```py
count = 0
valid_count = 0
while True:
  # generate a random number between 0 and 9
  rand_num = random.randrange(0,10)

  # prompt the user for a number
  value = input("Enter a number between 0 and 9: ")

  if value == 'x':
      print("Thanks for playing! Bye!")

  try:
      input_value = int(value)
  except ValueError as error:
      print("The value is invalid %s" % error)
  else:
      if input_value < 0 or input_value > 9:
          print("Input invalid. Enter a number between 0 and 9.")
          continue

      valid_count += 1
      if input_value == rand_num:
          print("Your guess is correct! You win!")
          break
      else:
          print("Nope! The random value was %s" % rand_num)
  finally:
      count += 1

print("You won the game in %d attempts "\
      "and %d inputs were valid" % (count, valid_count))
```

上述代码片段来自`try_except_else_finally.py`代码示例（可与本章一起下载）。尝试执行代码示例并玩游戏。您将注意到赢得游戏所需的总尝试次数以及有效输入的数量：

```py
    Enter a number between 0 and 9: g
 The value is invalid invalid literal for int() with
    base 10: 'g'
 Enter a number between 0 and 9: 3
 Your guess is correct! You win!
 You won the game in 9 attempts and 8 inputs were valid
```

这演示了`try-except-else-finally`块的工作原理。当关键代码块（在`try`关键字下）成功执行时，`else`关键字下的任何代码都会被执行，而在退出`try...except`块时（在退出代码块时清理资源时）`finally`关键字下的代码块会被执行。

使用先前的代码示例玩游戏时提供无效的输入，以了解代码块流程。

# 连接到互联网 - 网络请求

现在我们已经讨论了`try`/`except`关键字，让我们利用它来构建一个连接到互联网的简单应用程序。我们将编写一个简单的应用程序，从互联网上获取当前时间。我们将使用Python的`requests`库（[http://requests.readthedocs.io/en/master/#](http://requests.readthedocs.io/en/master/#)）。

`requests`模块使得连接到网络和检索信息成为可能。为了做到这一点，我们需要使用`requests`模块中的`get()`方法来发出请求：

```py
import requests
response = requests.get('http://nist.time.gov/actualtime.cgi')
```

在上述代码片段中，我们将一个URL作为参数传递给`get()`方法。在这种情况下，它是返回当前时间的Unix格式的URL（[https://en.wikipedia.org/wiki/Unix_time](https://en.wikipedia.org/wiki/Unix_time)）。

让我们利用`try`/`except`关键字来请求获取当前时间：

```py
#!/usr/bin/python3

import requests

if __name__ == "__main__":
  # Source for link: http://stackoverflow.com/a/30635751/822170
  try:
    response = requests.get('http://nist.time.gov/actualtime.cgi')
    print(response.text)
  except requests.exceptions.ConnectionError as error:
    print("Something went wrong. Try again")
```

在前面的例子中（可以与本章一起下载，命名为`internet_access.py`），请求是在`try`块下进行的，响应（由`response.text`返回）被打印到屏幕上。

如果在执行请求以检索当前时间时出现错误，将引发`ConnectionError`（[http://requests.readthedocs.io/en/master/user/quickstart/#errors-and-exceptions](http://requests.readthedocs.io/en/master/user/quickstart/#errors-and-exceptions)）。这个错误可能是由于缺乏互联网连接或不正确的URL引起的。这个错误被`except`块捕获。尝试运行这个例子，它应该返回`time.gov`的当前时间：

```py
    <timestamp time="1474421525322329" delay="0"/>
```

# requests的应用-检索天气信息

让我们使用`requests`模块来检索旧金山市的天气信息。我们将使用**OpenWeatherMap** API ([openweathermap.org](http://openweathermap.org))来检索天气信息：

1.  为了使用API，注册一个API账户并获取一个API密钥（免费）：

![](Images/e491b6d7-eedd-4706-a6c2-7ffe0ae779fb.png)来自openweathermap.org的API密钥

1.  根据API文档（[openweathermap.org/current](http://openweathermap.org/current)），可以使用`http://api.openweathermap.org/data/2.5/weather?zip=SanFrancisco&appid=API_KEY&units=imperial`作为URL来检索一个城市的天气信息。

1.  用你的账户的密钥替换`API_KEY`，并在浏览器中使用它来检索当前的天气信息。你应该能够以以下格式检索到天气信息：

```py
 {"coord":{"lon":-122.42,"lat":37.77},"weather":[{"id":800, 
       "main":"Clear","description":"clear sky","icon":"01n"}],"base": 
       "stations","main":{"temp":71.82,"pressure":1011,"humidity":50, 
       "temp_min":68,"temp_max":75.99},"wind":
       {"speed":13.04,"deg":291},
       "clouds":{"all":0},"dt":1474505391,"sys":{"type":3,"id":9966, 
       "message":0.0143,"country":"US","sunrise":1474552682, 
       "sunset":1474596336},"id":5391959,"name":"San 
       Francisco","cod":200}
```

天气信息（如前所示）以JSON格式返回。**JavaScript对象表示法**（**JSON**）是一种广泛用于在网络上传递数据的数据格式。JSON格式的主要优点是它是一种可读的格式，许多流行的编程语言支持将数据封装在JSON格式中。如前面的片段所示，JSON格式使得以可读的名称/值对交换信息成为可能。

让我们回顾一下使用`requests`模块检索天气并解析JSON数据：

1.  用前面例子中的URL（`internet_access.py`）替换为本例中讨论的URL。这应该以JSON格式返回天气信息。

1.  requests模块提供了一个解析JSON数据的方法。响应可以按以下方式解析：

```py
       response = requests.get(URL) 
       json_data = response.json()
```

1.  `json()`函数解析来自OpenWeatherMap API的响应，并返回不同天气参数（`json_data`）及其值的字典。

1.  由于我们知道API文档中的响应格式，可以从解析后的响应中检索当前温度：

```py
       print(json_data['main']['temp'])
```

1.  把所有这些放在一起，我们有这个：

```py
       #!/usr/bin/python3

       import requests

       # generate your own API key
       APP_ID = '5d6f02fd4472611a20f4ce602010ee0c'
       ZIP = 94103
       URL = """http://api.openweathermap.org/data/2.5/weather?zip={}
       &appid={}&units=imperial""".format(ZIP, APP_ID)

       if __name__ == "__main__":
         # API Documentation: http://openweathermap.org/
         current#current_JSON
         try:
           # encode data payload and post it
           response = requests.get(URL)
           json_data = response.json()
           print("Temperature is %s degrees Fahrenheit" %
           json_data['main']['temp'])
         except requests.exceptions.ConnectionError as error:
           print("The error is %s" % error)
```

前面的例子可以与本章一起下载，命名为`weather_example.py`。该例子应该显示当前的温度如下：

```py
    Temperature is 68.79 degrees Fahrenheit
```

# requests的应用-将事件发布到互联网

在上一个例子中，我们从互联网上检索了信息。让我们考虑一个例子，在这个例子中，我们需要在互联网上发布传感器事件。这可能是你不在家时猫门打开，或者有人踩在你家门口的地垫上。因为我们在上一章中讨论了如何将传感器与树莓派Zero连接，所以让我们讨论一个场景，我们可以将这些事件发布到*Slack*——一个工作场所通讯工具，Twitter，或者云服务，比如**Phant** ([https://data.sparkfun.com/](https://data.sparkfun.com/))。

在这个例子中，我们将使用`requests`将这些事件发布到Slack。每当发生传感器事件，比如猫门打开时，让我们在Slack上给自己发送直接消息。我们需要一个URL来将这些传感器事件发布到Slack。让我们回顾一下生成URL以将传感器事件发布到Slack：

1.  生成URL的第一步是创建一个*incoming webhook*。Webhook是一种可以将消息作为有效负载发布到应用程序（如Slack）的请求类型。

1.  如果您是名为*TeamX*的Slack团队成员，请在浏览器中启动您团队的应用程序目录，即`teamx.slack.com/apps`：

![](Images/dea1e47a-e8f1-4848-b40e-1cdd2836fcbc.png)启动您团队的应用程序目录

1.  在应用程序目录中搜索`incoming webhooks`，并选择第一个选项，Incoming WebHooks（如下截图所示）：

![](Images/bb557455-62ba-4716-8699-695bbf6be867.png)选择incoming webhooks

1.  点击添加配置：

![](Images/2b0b1d70-c3f9-4f41-bbfb-ea0b0ed6c3a8.png)添加配置

1.  当事件发生时，让我们向自己发送私人消息。选择Privately to (you)作为选项，并通过单击添加Incoming WebHooks集成来创建一个webhook：

![](Images/e37bc0b4-1cce-4840-9313-f2bfe7d0b60e.png)选择Privately to you

1.  我们已经生成了一个URL，用于发送有关传感器事件的直接消息（URL部分隐藏）：

![](Images/21db2e44-ca8f-4a25-acd3-355752853efa.png)生成的URL

1.  现在，我们可以使用先前提到的URL在Slack上向自己发送直接消息。传感器事件可以作为JSON有效负载发布到Slack。让我们回顾一下如何将传感器事件发布到Slack。

1.  例如，让我们考虑在猫门打开时发布消息。第一步是为消息准备JSON有效负载。根据Slack API文档（[https://api.slack.com/custom-integrations)](https://api.slack.com/custom-integrations)），消息有效负载需要采用以下格式：

```py
       payload = {"text": "The cat door was just opened!"}
```

1.  为了发布此事件，我们将使用`requests`模块中的`post()`方法。在发布时，数据有效负载需要以JSON格式进行编码：

```py
       response = requests.post(URL, json.dumps(payload))
```

1.  将所有内容放在一起，我们有：

```py
       #!/usr/bin/python3

       import requests
       import json

       # generate your own URL
       URL = 'https://hooks.slack.com/services/'

       if __name__ == "__main__":
         payload = {"text": "The cat door was just opened!"}
         try:
           # encode data payload and post it
           response = requests.post(URL, json.dumps(payload))
           print(response.text)
         except requests.exceptions.ConnectionError as error:
           print("The error is %s" % error)
```

1.  在发布消息后，请求返回`ok`作为响应。这表明发布成功了。

1.  生成您自己的URL并执行上述示例（与本章一起提供的`slack_post.py`一起下载）。您将在Slack上收到直接消息：

![](Images/b593e295-094d-403e-a245-a79068060b26.png)在Slack上直接发送消息

现在，尝试将传感器接口到Raspberry Pi Zero（在前几章中讨论），并将传感器事件发布到Slack。

还可以将传感器事件发布到Twitter，并让您的Raspberry Pi Zero检查新邮件等。查看本书的网站以获取更多示例。

# Flask web框架

在我们的最后一节中，我们将讨论Python中的Web框架。我们将讨论Flask框架（[http://flask.pocoo.org/](http://flask.pocoo.org/)）。基于Python的框架使得可以使用Raspberry Pi Zero将传感器接口到网络。这使得可以在网络中的任何位置控制设备并从传感器中读取数据。让我们开始吧！

# 安装Flask

第一步是安装Flask框架。可以按以下方式完成：

```py
    sudo pip3 install flask
```

# 构建我们的第一个示例

Flask框架文档解释了构建第一个示例。根据文档修改示例如下：

```py
#!/usr/bin/python3

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run('0.0.0.0')
```

启动此示例（与本章一起提供的`flask_example.py`一起下载），它应该在Raspberry Pi Zero上启动一个对网络可见的服务器。在另一台计算机上，启动浏览器，并输入Raspberry Pi Zero的IP地址以及端口号`5000`作为后缀（如下快照所示）。它应该将您带到服务器的索引页面，显示消息Hello World!：

![](Images/c515611e-7de9-40cc-a867-0fbc6cd43c88.png)基于Flask框架的Raspberry Pi Zero上的Web服务器

您可以使用命令行终端上的`ifconfig`命令找到Raspberry Pi Zero的IP地址。

# 使用Flask框架控制设备

让我们尝试使用Flask框架在家中打开/关闭电器。在之前的章节中，我们使用*PowerSwitch Tail II*来控制树莓派Zero上的台灯。让我们尝试使用Flask框架来控制相同的东西。按照以下图示连接PowerSwitch Tail：

![](Images/587aacf5-ad6d-45f6-bc42-214248b72183.png)使用Flask框架控制台灯

根据Flask框架文档，可以将URL路由到特定函数。例如，可以使用`route()`将`/lamp/<control>`绑定到`control()`函数：

```py
@app.route("/lamp/<control>") 
def control(control): 
  if control == "on": 
    lights.on() 
  elif control == "off": 
    lights.off() 
  return "Table lamp is now %s" % control
```

在前面的代码片段中，`<control>`是一个可以作为参数传递给绑定函数的变量。这使我们能够打开/关闭灯。例如，`<IP地址>:5000/lamp/on`打开灯，反之亦然。把它们放在一起，我们有这样：

```py
#!/usr/bin/python3 

from flask import Flask 
from gpiozero import OutputDevice 

app = Flask(__name__) 
lights = OutputDevice(2) 

@app.route("/lamp/<control>") 
def control(control): 
  if control == "on": 
    lights.on() 
  elif control == "off": 
    lights.off() 
  return "Table lamp is now %s" % control 

if __name__ == "__main__": 
    app.run('0.0.0.0')
```

上述示例可与本章一起下载，文件名为`appliance_control.py`。启动基于Flask的Web服务器，并在另一台计算机上打开Web服务器。为了打开灯，输入`<树莓派Zero的IP地址>:5000/lamp/on`作为URL：

这应该打开灯：

![](Images/06292c52-263d-41b8-bf2c-9cb2503a77dd.png)

因此，我们建立了一个简单的框架，可以控制网络中的电器。可以在HTML页面中包含按钮，并将它们路由到特定的URL以执行特定的功能。Python中还有几个其他框架可以开发Web应用程序。我们只是向您介绍了Python可能的不同应用程序。我们建议您查看本书的网站，了解更多示例，例如使用Flask框架控制万圣节装饰和其他节日装饰。

# 摘要

在本章中，我们讨论了Python中的`try`/`except`关键字。我们还讨论了从互联网检索信息的应用程序，以及将传感器事件发布到互联网。我们还讨论了Python的Flask Web框架，并演示了在网络中控制电器。在下一章中，我们将讨论Python中的一些高级主题。
