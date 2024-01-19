# 使用 Web 服务调用控制机器人车

有一天，无人驾驶汽车将主导我们的街道和高速公路。尽管感应信息和控制算法将位于汽车本身，但我们将有能力（并且可能会成为立法要求）从其他地方控制汽车。控制无人驾驶汽车将需要将汽车的感应信息以速度、GPS 位置等形式发送到控制站。相反，控制站的信息将以交通和方向等形式发送到汽车。

在本章中，我们将探讨从 T.A.R.A.S 发送感应信息和接收 T.A.R.A.S 控制信息的两个方面。

本章将涵盖以下主题：

+   从云端读取机器人车的数据

+   使用 Python 程序通过云端控制机器人车

# 完成本章所需的知识

要完成本章，您应该有一个完整的 T.A.R.A.S 机器人车，详细描述在第十三章中，*介绍树莓派机器人车*。与本书中的其他章节一样，需要具备 Python 的工作知识，以及对面向对象编程的基本理解。

# 项目概述

本章的项目将涉及通过互联网与 T.A.R.A.S 进行通信。我们将深入研究在第十五章中创建的仪表板模拟表，然后在仪表板上创建控制 T.A.R.A.S 的开关。这些项目应该需要大约 2 小时才能完成。

# 技术要求

要完成此项目，需要以下内容：

+   一个树莓派 3 型号（2015 年或更新型号）

+   一个 USB 电源适配器

+   一台电脑显示器

+   一个 USB 键盘

+   一个 USB 鼠标

+   一个完整的 T.A.R.A.S 机器人车套件（参见第十三章，*介绍树莓派机器人车*）

# 从云端读取机器人车的数据

在第十五章中，*将机器人车的感应输入连接到网络*，我们能够使用网站[`thingsboard.io/`](https://thingsboard.io/)将距离感应数据发送到云端。最后，我们展示了一个显示距离数值的模拟仪表。在本节中，我们将深入研究模拟小部件并进行自定义。

# 改变距离表的外观

这是我们改变距离表外观的方法：

1.  登录您的 ThingsBoard 账户

1.  点击 DASHBOARDS

1.  点击 ROBOTEYES 标题

1.  单击屏幕右下角的橙色铅笔图标

1.  您会注意到距离模拟表已经改变（见下面的屏幕截图）

1.  首先，表盘右上角有三个新图标

1.  右下角的颜色也变成了浅灰色

1.  您可以通过将鼠标悬停在右下角来调整小部件的大小

1.  您也可以将小部件移动到仪表板上

1.  右上角的 X 允许您从仪表板中删除此小部件

1.  带有下划线箭头的图标允许您将小部件下载为`.json`文件。此文件可用于将小部件导入 ThingsBoard 上的另一个仪表板

1.  单击小部件上的铅笔图标会产生一个从右侧滑出的菜单：

![](img/947e50b2-1c6b-43f3-8c1b-029d601c9124.png)

1.  如前面的屏幕截图所示，菜单选项为 DATA、SETTINGS、ADVANCED 和 ACTION。默认为 DATA

1.  点击 SETTINGS 选项卡

1.  在标题下，将名称更改为`RobotEyes`：

![](img/67a0e9ed-a1df-4a5d-8e13-0909edd17784.png)

1.  点击显示标题复选框

1.  点击背景颜色下的白色圆圈：

![](img/3fcf61fa-9959-41d4-99b6-cdb3aae1e365.png)

1.  您将看到颜色选择对话框：

![](img/3045209d-d736-42a9-9e1e-00babfd775ff.png)

1.  将顶部更改为`rgb(50,87,126)`

1.  点击右上角的橙色复选框以接受更改

1.  您会注意到距离表有一些外观上的变化（请参见以下屏幕截图）：

![](img/e12eee79-c197-47b5-afc4-fb904b89f9d1.png)

# 更改距离表上的范围

看着距离模拟表，很明显，对于我们的应用程序来说，有负数并没有太多意义。让我们将范围更改为`0`到`100`：

1.  点击小部件上的铅笔图标

1.  点击“高级”选项卡

1.  将最小值更改为`0`，将最大值更改为`100`：

![](img/2292aa38-1f4e-4055-8dca-42262a44e8e5.png)

1.  点击右上角的橙色复选框以接受对小部件的更改

1.  关闭 ROBOTEYES 对话框

1.  点击右下角的橙色复选框以接受对仪表板的更改

1.  您会注意到距离模拟表现在显示范围为`0`到`100`：

![](img/c8e1a9b6-5b28-423b-9cf2-dd21c16c35c7.png)

# 在您的帐户之外查看仪表板

对于我们的最后一个技巧，我们将在我们的帐户之外显示我们的仪表板（我们在第十章中也这样做，*发布到 Web 服务*）。这也允许我们将我们的仪表板发送给朋友。那么，为什么我们要在帐户之外查看我们的仪表板呢？物联网的核心概念是我们可以从一个地方获取信息并在其他地方显示，也许是在世界的另一边的某个地方。通过使我们的仪表板在我们的帐户之外可访问，我们允许在任何地方设置仪表板，而无需共享我们的帐户信息。想象一下世界上某个地方有一块大屏幕，屏幕的一小部分显示我们的仪表板。从 T.A.R.A.S 显示距离信息可能对许多人来说并不是很感兴趣，但重要的是概念。

要分享我们的仪表板，请执行以下操作：

1.  在 ThingsBoard 应用程序中，点击“仪表板”选项

1.  点击 RobotEyes 仪表板下的中间图标：

![](img/2218b998-0b43-4bdd-af3e-69b5d68724f9.png)

1.  您将看到类似以下的对话框（URL 已部分模糊处理）：

![](img/0368bece-b8e4-42b2-9dc1-4f059ad502e8.png)

1.  点击 URL 旁边的图标将 URL 复制到剪贴板

1.  要测试 URL，请将其粘贴到计算机上的完全不同的浏览器中（或将其发送给朋友并让他们打开）

1.  您应该能够看到我们的距离模拟表的仪表板

# 使用 Python 程序通过云控制机器人车

能够在仪表板中看到传感器数据是非常令人印象深刻的。但是，如果我们想要从我们的仪表板实际控制某些东西怎么办？在本节中，我们将做到这一点。我们将首先构建一个简单的开关来控制 T.A.R.A.S 上的 LED。然后，我们将扩展此功能，并让 T.A.R.A.S 通过互联网上的按钮按下来跳舞。

让我们首先将仪表板的名称从`RobotEyes`更改为`RobotControl`：

1.  在 ThingsBoard 应用程序中，点击“仪表板”选项

1.  点击 RobotEyes 仪表板下的铅笔图标：

![](img/5d584ac7-e3ca-4193-adf0-9c86c0a719e3.png)

1.  点击橙色铅笔图标

1.  将瓷砖从`RobotEyes`更改为`RobotControl`：

![](img/0c99f107-af76-44a2-b4c6-0ec5cb7863b5.png)

1.  点击橙色复选框以接受更改

1.  退出侧边对话框

现在让我们从 ThingsBoard 仪表板上控制 T.A.R.A.S 上的 LED。

# 向我们的仪表板添加一个开关

为了控制 LED，我们需要创建一个开关：

1.  点击 RobotControl 仪表板

1.  点击橙色铅笔图标

1.  点击+图标

1.  点击“创建新小部件”图标

1.  选择“控制小部件”并点击“切换控制”：

![](img/a2a3f9d4-ab05-4e65-82a0-4581d1c2b6b6.png)

1.  在目标设备下，选择 RobotControl

1.  点击“设置”选项卡：

![](img/ad038ebb-a391-4dc2-8d92-901d1b7f5870.png)

1.  将标题更改为`Green Tail Light`，然后点击显示标题

1.  点击高级选项卡

1.  将 RPC 设置值方法更改为`toggleGreenTailLight`：

![](img/5b306916-8573-4478-922b-6930261347a0.png)

1.  点击橙色的勾号图标以接受对小部件的更改

1.  关闭侧边对话框

1.  点击橙色的勾号图标以接受对仪表板的更改

那么，我们刚刚做了什么？我们在我们的仪表板上添加了一个开关，它将发布一个名为`toggleGreenTailLight`的方法，该方法将返回一个值，要么是`true`，要么是`false`（默认返回值为`this is a switch`）。

既然我们有了开关，让我们在树莓派上编写一些代码来响应它。

# 控制 T.A.R.A.S 上的绿色 LED

要控制 T.A.R.A.S 上的绿色 LED，我们需要编写一些代码到 T.A.R.A.S 上的树莓派。我们需要我们仪表板的访问令牌（参见第十五章，*将机器人汽车的感应输入连接到网络*，关于如何获取）：

1.  从应用程序菜单中打开 Thonny | 编程 | Thonny Python IDE

1.  点击新建图标创建一个新文件

1.  输入以下内容：

```py
import paho.mqtt.client as mqtt
from gpiozero import LED
import json

THINGSBOARD_HOST = 'demo.thingsboard.io' ACCESS_TOKEN = '<<access token>>'
green_led=LED(21)

def on_connect(client, userdata, rc, *extra_params):
   print('Connected with result code ' + str(rc))
    client.subscribe('v1/devices/me/rpc/request/+')

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode("utf-8")) 

    if data['method'] == 'toggleGreenTailLight':
        if data['params']:
            green_led.on()
        else:
            green_led.off()

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set(ACCESS_TOKEN)
client.connect(THINGSBOARD_HOST, 1883, 60)

client.loop_forever()
```

1.  将文件保存为`control-green-led-mqtt.py`

1.  运行代码

1.  返回我们的 ThingsBoard 仪表板（如果您一直在 T.A.R.A.S 上的树莓派之外的计算机上使用，现在是一个好时机）

1.  点击开关以打开它

1.  您应该看到 T.A.R.A.S 上的绿色 LED 随开关的打开和关闭而打开和关闭

那么，我们刚刚做了什么？使用从 ThingsBoard 网站获取的样板代码，我们构建了一个**消息查询遥测传输**（**MQTT**）客户端，该客户端监听仪表板，并在接收到`toggleGreenTailLight`方法时做出响应。我们通过在`on_connect`方法中订阅`'v1/devices/me/rpc/request/+'`来实现这一点。我们在第十章中也使用了 MQTT，*发布到网络服务*。然而，由于这段代码几乎只是 MQTT 代码，让我们更仔细地研究一下。

MQTT 是一种基于`发布者`和`订阅者`方法的轻量级消息传递协议，非常适合在物联网中使用。理解发布者和订阅者的一个好方法是将它们与过去的报纸联系起来。发布者是制作报纸的实体；订阅者是购买和阅读报纸的人。发布者不知道，甚至不必知道，为了印刷报纸有多少订阅者（不考虑出版成本）。想象一下每天都会出版的巨大报纸，不知道有多少人会购买他们的报纸。因此，发布者可以有很多订阅者，反之亦然，订阅者可以订阅很多发布者，就像读者可以阅读很多不同的报纸一样。

我们首先导入我们代码所需的库：

```py
import paho.mqtt.client as mqtt
from gpiozero import LED
import json

THINGSBOARD_HOST = 'demo.thingsboard.io'
ACCESS_TOKEN = '<<access token>>'
green_led=LED(21)
```

这里需要注意的是`json`和`pho.mqtt.client`库，这些库是与 MQTT 服务器通信所需的。`THINGSBOARD_HOST`和`ACCESS_TOKEN`是连接到正确服务器和服务所需的标准变量。当然，还有`GPIO Zero LED`类，它将`green_led`变量设置为 GPIO 引脚`21`（这恰好是 T.A.R.A.S 上的绿色尾灯）。

`on_connect`方法打印出连接信息，然后订阅将我们连接到来自我们 ThingsBoard 仪表板的`rpc`方法的服务：

```py
def on_connect(client, userdata, rc, *extra_params):
    print('Connected with result code ' + str(rc))
    client.subscribe('v1/devices/me/rpc/request/+')
```

正是`on_message`方法使我们能够真正修改我们的代码以满足我们的目的：

```py
def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode("utf-8")) 

    if data['method'] == 'toggleGreenTailLight':
        if data['params']:
            green_led.on()
        else:
            green_led.off()
```

我们首先从我们的`msg`变量中收集`data`，然后使用`json.loads`方法将其转换为`json`文件。`method`声明`on_message(client, userdata, msg)`，再次是来自 ThingsBoard 网站的标准样板代码。我们真正关心的只是获取`msg`的值。

第一个`if`语句，`if data['method'] == 'toggleGreenTailLight'`，检查我们的`msg`是否包含我们在 ThingsBoard 仪表板上设置的`toggleGreenTailLight`方法。一旦我们知道`msg`包含这个方法，我们使用`if data['params']`提取`data`中的其他键值对，以检查是否有`True`值。换句话说，调用`on_message`方法返回的`json`文件看起来像`{'params': True, 'method': 'toggleGreenTailLight'}`。这基本上是一个包含两个键值对的 Python 字典。这可能看起来令人困惑，但最简单的想法是将其想象成一个`json`版本的方法（`toggleGreenTailLight`）和一个返回值（`True`）。

真正理解发生了什么的一种方法是在`on_message`方法中添加一个`print`语句来`print data`，就在`data = json.loads(msg.payload.decode("utf-8"))`之后。因此，该方法看起来像以下内容：

```py
def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode("utf-8")) 
    print(data)
    .
    .
    . 
```

当从`params`返回的值为`True`时，我们简单地使用标准的 GPIO Zero 代码打开 LED。当从`params`返回的值不是`True`（或`False`，因为只有两个可能的值）时，我们关闭 LED。

通过使用互联网看到 LED 开关是相当令人印象深刻的。然而，这还不够。让我们利用我们在之前章节中使用的一些代码，让 T.A.R.A.S 跳舞。这一次，我们将通过互联网让它跳舞。

# 使用互联网让 T.A.R.A.S 跳舞

要让 T.A.R.A.S 再次跳舞，我们需要确保第十四章中的代码*使用 Python 控制机器人车*与我们将要编写的代码在同一个目录中。

我们将从在我们的仪表板上创建一个跳舞开关开始：

1.  按照之前的步骤 1 到 9，在仪表板下添加一个开关来创建一个开关

1.  将标题更改为 Dance Switch 并点击显示标题

1.  点击高级选项卡

1.  将`RPC set value method`更改为`dance`

1.  点击橙色的勾号图标以接受对小部件的更改

1.  关闭侧边对话框

1.  点击橙色的勾号图标以接受对仪表板的更改

现在我们有了开关，让我们修改我们的代码：

1.  从应用程序菜单 | 编程 | Thonny Python IDE 打开 Thonny

1.  点击新图标创建一个新文件

1.  输入步骤 4 中的以下内容：

```py
import paho.mqtt.client as mqtt
import json
from RobotDance import RobotDance

THINGSBOARD_HOST = 'demo.thingsboard.io'
ACCESS_TOKEN = '<<access token>>'
robot_dance = RobotDance()

def on_connect(client, userdata, rc, *extra_params):
    print('Connected with result code ' + str(rc))
    client.subscribe('v1/devices/me/rpc/request/+')

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode("utf-8")) 

    if data['method'] == 'dance':
        if data['params']:
            robot_dance.lets_dance_incognito()

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set(ACCESS_TOKEN)
client.connect(THINGSBOARD_HOST, 1883, 60)

client.loop_forever()
```

1.  将文件保存为`internet-dance.py`

1.  运行代码

现在去仪表板上打开跳舞开关（不幸的是，它是一个开关而不是一个按钮）。T.A.R.A.S 应该开始跳舞，就像在第十四章中一样，*使用 Python 控制机器人车*。

那么，我们刚刚做了什么？嗯，我们拿了简单的代码，稍微修改了一下，通过面向对象编程的力量，我们能够让 T.A.R.A.S 跳舞，而无需更改甚至浏览我们旧的`RobotDance`代码（难道 OOP 不是自从你认为最好的东西以来最好的东西吗？）。

对于 MQTT 代码，我们所要做的就是在`RobotDance`类中添加`import`，去掉多余的 GPIO Zero 导入，去掉对 LED 的任何引用（因为这会引起冲突），然后修改我们的`on_message`方法以查找`dance`作为方法。

`RobotDance`类类型的`robot_dance`对象完成了所有工作。当我们在这个对象上调用`lets_dance_incognito`方法时，它会启动`RobotWheels`、`RobotBeep`、`TailLights`和`RobotCamera`类中用于移动的方法。最终结果是通过互联网上的开关让 T.A.R.A.S 跳舞的方法。

# 摘要

在本章中，我们进一步研究了我们用于距离传感信息的仪表盘模拟表。在更改范围并将其公开之前，我们对其进行了美学修改。然后，我们将注意力转向通过互联网控制 T.A.R.A.S。通过使用一个简单的程序，我们能够通过仪表盘开关打开 T.A.R.A.S 上的绿色 LED。我们利用这些知识修改了我们的代码，通过另一个仪表盘开关使 T.A.R.A.S 跳舞。

在第十七章 *构建 JavaScript 客户端*中，我们将继续编写一个 JavaScript 客户端，通过互联网控制 T.A.R.A.S。

# 问题

1.  无人驾驶汽车需要从中央站获取什么类型的信息？

1.  真/假？在 ThingsBoard 仪表盘中无法更改小部件的背景颜色。

1.  如何更改仪表盘模拟表的范围？

1.  真/假？从行`print(data)`返回的信息无法被人类阅读。

1.  我们从`RobotDance`类中调用哪个方法来使 T.A.R.A.S 跳舞？

1.  真/假？我们需要使用的处理`json`数据的库叫做`jason`。

1.  我们如何在仪表盘上创建一个开关？

1.  真/假？T.A.R.A.S 上的绿色 LED 连接到 GPIO 引脚 14。

1.  真/假？一个发布者只能有一个订阅者。

1.  使用`on_message`方法从`msg`返回多少个键值对？

# 进一步阅读

由于我们只是简单地涉及了 ThingsBoard，查看他们的文档是个好主意，网址是[`thingsboard.io/docs/guides/`](https://thingsboard.io/docs/guides/)。
