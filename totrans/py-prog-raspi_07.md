# 请求和 Web 框架

本章的主要主题是 Python 中的请求和 Web 框架。我们将讨论使从网络检索数据（例如，获取天气更新）、上传数据到远程服务器（例如，记录传感器数据）或控制本地网络上的设备成为可能的库和框架。我们还将讨论有助于学习本章核心主题的话题。

# try/except 关键字

到目前为止，我们假设理想条件，即程序执行将不会遇到错误，来审查和测试了所有我们的示例。相反，应用程序有时会因外部因素（例如无效的用户输入和差的互联网连接）或程序员造成的程序逻辑错误而失败。在这种情况下，我们希望程序报告/记录错误的性质，并在退出程序之前继续执行或清理资源。`try`/`except`关键字提供了一种在程序执行过程中捕获错误并采取补救措施的方法。由于可以在代码的关键部分捕获和记录错误，因此`try`/`except`关键字在调试应用程序时特别有用。

通过比较两个示例来了解`try`/`except`关键字。让我们构建一个简单的猜数字游戏，用户被要求猜测一个介于 0 到 9 之间的数字：

1.  使用 Python 的`random`模块生成一个随机数（介于 0 到 9 之间）。如果用户猜对了生成的数字，Python 程序宣布用户为赢家并退出游戏。

1.  如果用户输入是字母`x`，程序将退出游戏。

1.  用户输入通过`int()`函数转换为整数。进行合理性检查以确定用户输入是否在 0 到 9 之间。

1.  整数与一个随机数进行比较。如果它们相同，用户被宣布为赢家，程序退出游戏。

让我们观察当我们故意向这个程序提供错误输入时会发生什么（这里显示的代码片段可以与本章一起作为`guessing_game.py`下载）：

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

让我们执行前面的代码片段，并将输入`hello`提供给程序：

```py
    Enter a number between 0 and 9: hello
 Traceback (most recent call last):
 File "guessing_game.py", line 12, in <module>
 input_value = int(value)
 ValueError: invalid literal for int() with base 10: 'hello'

```

在前面的例子中，程序在尝试将用户输入`hello`转换为整数时失败。程序执行以异常结束。异常突出了错误发生的位置。在这种情况下，它发生在第 10 行：

```py
    File "guessing_game.py", line 12, in <module>
 input_value = int(value)

```

错误的性质也在异常中得到了强调。在这个例子中，最后一行表明抛出的异常是`ValueError`：

```py
    ValueError: invalid literal for int() with base 10: 'hello'

```

让我们讨论相同的示例（可以与本章一起作为`try_and_except.py`下载），它使用了`try`/`except`关键字。在捕获此异常并将其打印到屏幕后，可以继续玩游戏。我们有以下代码：

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

让我们讨论一下使用`try`/`except`关键字时相同的示例是如何工作的：

1.  从前面的示例中，我们知道当用户提供错误的输入（例如，0 到 9 之间的字母而不是数字）时，异常会在第 10 行（用户输入转换为整数的地方）发生，错误的性质被命名为`ValueError`。

1.  可以通过将这段代码包裹在`try...except`块中来避免程序执行的中断：

```py
      try: 
         input_value = int(value) 
      except ValueError as error:
         print("The value is invalid %s" % error)

```

1.  在接收到用户输入后，程序尝试在`try`块中将用户输入转换为整数。

1.  如果发生了`ValueError`，`error`会被`except`块捕获，并且实际错误信息会与以下信息一起打印到屏幕上：

```py
       except ValueError as error:
           print("The value is invalid %s" % error)

```

1.  尝试执行代码示例，并尝试提供一个无效的输入。你会注意到程序会打印出错误信息（包括错误的性质），然后回到游戏循环的顶部并继续寻找有效的用户输入：

```py
       Enter a number between 0 and 9: 3
 Nope! The random value was 5
 Enter a number between 0 and 9: hello
 The value is invalid invalid literal for int() with
       base 10: 'hello'
 Enter a number between 0 and 10: 4
 Nope! The random value was 6

```

`try...except`块伴随着大量的处理能力成本。因此，保持`try...except`块尽可能短是很重要的。因为我们知道错误发生在我们尝试将用户输入转换为整数的那一行，所以我们将其包裹在`try...except`块中以捕获错误。

因此，`try`/`except`关键字用于防止程序执行过程中由于错误导致的任何异常行为。它允许记录错误并采取补救措施。类似于`try...except`块，也存在`try...except...else`和`try...except...else`代码块。让我们通过几个示例快速回顾这些选项。

# try...except...else

`try...except...else`块在当我们只想在没有引发异常的情况下执行特定代码块时特别有用。为了演示这个概念，让我们使用这个块重写猜数字游戏示例：

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

修改后的猜数字游戏示例，它使用了`try...except...else`块，可以与本章一起下载，文件名为`try_except_else.py`。在这个示例中，程序只有在接收到有效的用户输入时才会将用户输入与随机数进行比较。否则，它会跳过`else`块，并回到循环的顶部以接受下一个用户输入。因此，当我们在`try`块中没有因为代码而引发异常时，会使用`try...except...else`。

# try...except...else...finally

如其名所示，`finally`块用于在离开`try`块时执行一段代码。即使在抛出异常之后，这段代码也会被执行。这在我们需要在进入下一阶段之前清理资源并释放内存的情况下非常有用。

让我们通过我们的猜谜游戏来演示`finally`块的功能。为了理解`finally`关键字的工作原理，让我们使用一个名为`count`的计数变量，它在`finally`块中递增，以及另一个名为`valid_count`的计数变量，它在`else`块中递增。以下是我们的代码：

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

上述代码片段来自`try_except_else_finally.py`代码示例（与本章一起提供下载）。尝试执行代码示例并玩游戏。你会注意到赢得游戏所需的尝试总数以及有效输入的数量：

```py
    Enter a number between 0 and 9: g
 The value is invalid invalid literal for int() with
    base 10: 'g'
 Enter a number between 0 and 9: 3
 Your guess is correct! You win!
 You won the game in 9 attempts and 8 inputs were valid

```

这演示了`try-except-else-finally`块的工作原理。当关键的代码块（在`try`关键字下）成功执行时，任何在`else`关键字下的代码都会执行，而`finally`关键字下的代码块在退出`try...except`块时执行（在退出代码块时清理资源很有用）。

在玩游戏时，尝试使用之前的代码示例提供无效的输入，以了解代码块流程。

# 连接到互联网 - 网络请求

现在我们已经讨论了`try`/`except`关键字，让我们利用它来构建一个简单的应用程序，该程序可以连接到互联网。我们将编写一个简单的应用程序，从互联网获取当前时间。我们将使用 Python 的`requests`库（[`requests.readthedocs.io/en/master/#`](http://requests.readthedocs.io/en/master/#)）。

`requests`模块允许连接到网络并检索信息。为了做到这一点，我们需要使用`requests`模块中的`get()`方法来发送请求：

```py
import requests
response = requests.get('http://nist.time.gov/actualtime.cgi')

```

在前面的代码片段中，我们将一个 URL 作为参数传递给`get()`方法。在这种情况下，它是返回 Unix 格式当前时间的 URL（[`en.wikipedia.org/wiki/Unix_time`](https://en.wikipedia.org/wiki/Unix_time)）。

让我们使用`try`/`except`关键字来请求获取当前时间：

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

在前面的示例（与本章一起提供下载，文件名为`internet_access.py`）中，请求是在`try`块中发出的，并且响应（由`response.text`返回）被打印到屏幕上。

如果在检索当前时间时发生错误，将引发`ConnectionError`（[`requests.readthedocs.io/en/master/user/quickstart/#errors-and-exceptions`](http://requests.readthedocs.io/en/master/user/quickstart/#errors-and-exceptions)）。这个错误可能是由缺少互联网连接或错误的 URL 引起的。这个错误被`except`块捕获。尝试运行示例，它应该从`time.gov`返回当前时间：

```py
    <timestamp time="1474421525322329" delay="0"/>

```

# `requests`库的应用 - 获取天气信息

让我们使用`requests`模块来检索旧金山市的天气信息。我们将使用**OpenWeatherMap** API ([openweathermap.org](http://openweathermap.org))来检索天气信息：

1.  为了使用 API，注册一个 API 账户并获取一个 API 密钥（免费）：

![](img/image_07_001.png)

一个来自 openweathermap.org 的 API 密钥

1.  根据 API 文档([openweathermap.org/current](http://openweathermap.org/current))，可以使用`http://api.openweathermap.org/data/2.5/weather?zip=SanFrancisco&appid=API_KEY&units=imperial`作为 URL 来检索一个城市的天气信息。

1.  将`API_KEY`替换为您账户中的密钥，并在浏览器中用它来检索当前的天气信息。您应该能够以以下格式检索天气信息：

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

天气信息（如前所述）以 JSON 格式返回。**JavaScript 对象表示法**（**JSON**）是一种广泛用于在网络上交换数据的数据格式。JSON 格式的优点是它是一种可读的格式，许多流行的编程语言都支持以 JSON 格式封装数据。如前所述的代码片段所示，JSON 格式允许以可读的名称/值对的形式交换信息。

让我们回顾一下使用`requests`模块检索天气并解析 JSON 数据的过程：

1.  将前面例子（`internet_access.py`）中的 URL 替换为本文中讨论的 URL。这将返回 JSON 格式的天气信息。

1.  请求模块提供了一个解析 JSON 数据的方法。响应可以按照以下方式解析：

```py
       response = requests.get(URL) 
       json_data = response.json()

```

1.  `json()`函数解析来自 OpenWeatherMap API 的响应，并返回一个包含不同天气参数及其值的字典（`json_data`）。

1.  由于我们知道 API 文档中的响应格式，我们可以按照以下方式从解析的响应中检索当前温度：

```py
       print(json_data['main']['temp'])

```

1.  将所有内容整合，我们得到以下内容：

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

前面的例子可以作为本章的附件下载，名为`weather_example.py`。该示例应显示以下当前温度：

```py
    Temperature is 68.79 degrees Fahrenheit

```

# 请求的应用 - 向互联网发布事件

在前面的例子中，我们从互联网检索了信息。让我们考虑一个例子，其中我们不得不在互联网上的某个地方发布一个传感器事件。这可能是在您离家时猫门打开，或者有人在家门口踩到门垫。因为我们已经在上一章讨论了将传感器连接到树莓派 Zero，让我们讨论一个可以将这些事件发布到*Slack*（一个工作场所沟通工具）、Twitter 或云服务（如**Phant** [`data.sparkfun.com/`](https://data.sparkfun.com/)）的场景。

在此示例中，我们将使用`requests`将这些事件发布到 Slack。每当发生类似猫门开启的传感器事件时，我们都会给自己发送一条直接消息。我们需要一个 URL 来将这些传感器事件发布到 Slack。让我们回顾一下如何生成一个 URL 以发布传感器事件到 Slack：

1.  生成 URL 的第一步是创建一个*incoming webhook*。Webhook 是一种请求类型，可以将作为有效载荷的消息发布到像 Slack 这样的应用程序。

1.  如果你是名为*TeamX*的 Slack 团队的一员，请在浏览器中打开您的团队应用目录，即`teamx.slack.com/apps`：

![图片](img/image_07_002.png)

启动您的团队应用目录

1.  在您的应用目录中搜索`incoming webhooks`并选择第一个选项，即 Incoming WebHooks（如下面的截图所示）：

![图片](img/image_07_003.png)

选择“incoming webhooks”

1.  点击添加配置：

![图片](img/image_07_004.png)

添加配置

1.  当发生事件时，让我们给自己发送一条私密消息。选择“私下发送给你”作为选项，并通过点击添加 Incoming WebHooks 集成来创建 webhook：

![图片](img/image_07_005.png)

选择“私下发送给你”

1.  我们已经生成了一个用于发送关于传感器事件的直接消息的 URL（URL 部分被隐藏）：

![图片](img/image_07_006.png)

生成的 URL

1.  现在，我们可以使用之前提到的 URL 直接在 Slack 上给自己发送消息。传感器事件可以作为 JSON 有效载荷发布到 Slack。让我们回顾一下如何将传感器事件发布到 Slack。

1.  例如，让我们考虑在猫门打开时发布一条消息。第一步是为消息准备 JSON 有效载荷。根据 Slack API 文档（[`api.slack.com/custom-integrations`](https://api.slack.com/custom-integrations)），消息有效载荷需要以下格式：

```py
       payload = {"text": "The cat door was just opened!"}

```

1.  为了发布此事件，我们将使用`requests`模块中的`post()`方法。在发布时，数据有效载荷需要以 JSON 格式编码：

```py
       response = requests.post(URL, json.dumps(payload))

```

1.  将所有这些放在一起，我们得到这个：

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

1.  在发布消息时，请求返回`ok`作为响应。这表示发布成功。

1.  生成您自己的 URL 并执行前面的示例（作为本章的附件`slack_post.py`提供下载）。您将在 Slack 上收到一条直接消息：

![图片](img/image_07_007.png)

Slack 上的直接消息

现在，尝试将传感器连接到 Raspberry Pi Zero（在前面章节中讨论过）并将传感器事件发布到 Slack。

还可以将传感器事件发布到 Twitter，并让您的 Raspberry Pi Zero 检查新电子邮件等。请查看本书的网站以获取更多示例。

# Flask Web 框架

在我们的最后一节中，我们将讨论 Python 中的 Web 框架。我们将讨论 Flask 框架 ([`flask.pocoo.org/`](http://flask.pocoo.org/))。基于 Python 的框架允许使用 Raspberry Pi Zero 将传感器连接到网络。这使得可以在网络内的任何地方控制电器和读取传感器的数据。让我们开始吧！

# 安装 Flask

第一步是安装 Flask 框架。可以按照以下步骤进行：

```py
    sudo pip3 install flask

```

# 构建我们的第一个示例

Flask 框架文档解释了如何构建第一个示例。按照以下方式修改文档中的示例：

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

启动此示例（与本章一起提供下载，名为 `flask_example.py`），它应该在 Raspberry Pi Zero 上启动一个对网络可见的服务器。在另一台计算机上，启动浏览器并输入 Raspberry Pi Zero 的 IP 地址以及端口号 `5000` 作为后缀（如下面的快照所示）。它应该带您到显示消息 Hello World! 的服务器索引页面：

![图片](img/image_07_008.png)

基于 Flask 框架的 Raspberry Pi Zero 上的 Web 服务器

您可以使用命令行终端上的 `ifconfig` 命令找到您的 Raspberry Pi Zero 的 IP 地址。

# 使用 Flask 框架控制电器

让我们尝试使用 Flask 框架在家中的电器上打开/关闭。在之前的章节中，我们使用了 *PowerSwitch Tail II* 通过 Raspberry Pi Zero 控制台灯。让我们尝试使用 Flask 框架来控制它。按照以下图示连接 PowerSwitch Tail：

![图片](img/image_07_009.png)

使用 Flask 框架控制台灯

根据 Flask 框架文档，可以将 URL 路由到特定的函数。例如，可以使用 `route()` 将 `/lamp/<control>` 绑定到 `control()` 函数：

```py
@app.route("/lamp/<control>") 
def control(control): 
  if control == "on": 
    lights.on() 
  elif control == "off": 
    lights.off() 
  return "Table lamp is now %s" % control

```

在前面的代码片段中，`<control>` 是一个可以作为参数传递给绑定函数的变量。这使得我们能够控制灯的开关。例如，`<IP 地址>:5000/lamp/on` 会打开灯，反之亦然。将所有这些放在一起，我们得到如下：

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

前面的示例作为 `appliance_control.py` 与本章一起提供下载。启动基于 Flask 的 Web 服务器，并在另一台计算机上打开一个 Web 服务器。为了打开灯，输入 `<Raspberry Pi Zero 的 IP 地址>:5000/lamp/on` 作为 URL：

这应该会打开灯：

![图片](img/image_07_010.png)

因此，我们已经构建了一个简单的框架，该框架能够控制网络内的电器。可以在 HTML 页面中添加按钮并将它们路由到特定的 URL 以执行特定功能。Python 中还有其他几个框架可以用来开发 Web 应用程序。我们只是向您介绍了 Python 可能实现的不同应用。我们建议您查看本书的网站以获取更多示例，例如使用 Flask 框架控制万圣节装饰和其他节日装饰。

# 摘要

在本章中，我们讨论了 Python 中的`try`/`except`关键字。我们还讨论了开发从互联网检索信息的应用程序以及将传感器事件发布到互联网的应用程序。我们还讨论了 Python 的 Flask Web 框架，并演示了在网络内控制电器。在下一章中，我们将讨论 Python 的一些高级主题。
