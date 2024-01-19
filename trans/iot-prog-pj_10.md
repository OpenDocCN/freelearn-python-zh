# 发布到Web服务

在物联网的核心是允许与物理设备交互的Web服务。在本章中，我们将探讨使用Web服务来显示来自树莓派的传感器数据的用途。我们还将研究Twilio，一个短信服务，以及我们如何使用此服务从树莓派发送短信给自己。

本章将涵盖以下主题：

+   将传感器数据发布到基于云的服务

+   为文本消息传输设置账户

# 项目概述

在本章中，我们将编写代码将我们的传感器数据显示到IoT仪表板上。此外，我们还将探索Twilio，一个短信服务。然后，我们将把这两个概念结合起来，以增强我们在[第9章](1a50006e-75d3-4dc1-96db-82114b437795.xhtml)中构建的家庭安全仪表板。

# 入门

要完成此项目，需要以下内容：

+   树莓派3型号（2015年或更新型号）

+   一个USB电源适配器

+   一个计算机显示器

+   一个USB键盘

+   一个USB鼠标

+   一个面包板

+   跳线

+   一个DHT-11温度传感器

+   一个PIR传感器

+   一个按钮（锁定）

+   一个按键开关（可选）

# 将传感器数据发布到基于云的服务

在本节中，我们将使用MQTT协议将传感器数据发布到在线仪表板。这将涉及在ThingsBoard网站上设置一个账户，并使用`demo`环境。

# 安装MQTT库

我们将使用MQTT协议与ThingsBoard仪表板进行通信。要在树莓派上设置库，请执行以下操作：

1.  从主工具栏打开终端设备

1.  输入`**sudo pip3 install pho-mqtt**`

1.  您应该看到库已安装

# 设置一个账户并创建一个设备

首先，转到ThingsBoard网站[www.thingsboard.io](http://www.thingsboard.io)：

1.  点击屏幕顶部的TRY IT NOW按钮。向下滚动并在Thing Board Community Edition部分下点击LIVE DEMO按钮：

![](assets/99834e79-3ac3-46de-89c3-3c7b4db4d192.png)

1.  您将看到一个注册窗口。输入适当的信息设置一个账户。一旦您的账户成功设置，您将看到一个对话框显示以下内容：

![](assets/59f38928-a10e-466c-a456-09d642cd38a3.png)

1.  点击登录进入应用程序。之后，您应该在屏幕左侧看到一个菜单：

![](assets/ec514cbb-75e8-431b-92fa-02591dd5704c.png)

1.  点击DEVICES。在屏幕右下角，找到一个带加号的圆形橙色图形，如下所示：

![](assets/9d87136a-700e-4000-b7d6-183d75e5d0cc.png)

1.  点击这个橙色圆圈添加一个新设备。在添加设备对话框中，输入`Room Conditions`作为名称*，并选择默认作为设备类型*。不要选择Is gateway。点击ADD：

![](assets/853a76a0-1f4c-456a-9f12-433668ea6a8f.png)

1.  您应该在您的设备下看到一个新的框，名称为Room Conditions：

![](assets/18e9ccb0-00e3-4284-b5ce-fcadb263d16e.png)

1.  点击此框，然后会从右侧滑出一个菜单。点击COPY ACCESS TOKEN按钮将此令牌复制到剪贴板上：

![](assets/115adb70-987e-4e89-9eb4-f79a2aadefb8.png)

我们在这里做的是设置ThingsBoard账户和ThingsBoard内的新设备。我们将使用此设备从树莓派检索传感信息，并制作这些值的仪表板。

# 读取传感器数据并发布到ThingsBoard

现在是时候创建我们的电路和代码了。使用GPIO引脚19安装DHT-11传感器（如果不确定如何将DHT-11传感器连接到树莓派，请参考[第9章](1a50006e-75d3-4dc1-96db-82114b437795.xhtml)，*构建家庭安全仪表板*）：

1.  打开Thonny并创建一个名为`dht11-mqtt.py`的新文件。在文件中输入以下内容并运行。确保粘贴从剪贴板中复制的访问令牌：

```py
from time import sleep
import Adafruit_DHT
import paho.mqtt.client as mqtt
import json

host = 'demo.thingsboard.io'
access_token = '<<access token>>'
dht_sensor = Adafruit_DHT.DHT11
pin = 19

sensor_data = {'temperature': 0, 'humidity': 0}

client = mqtt.Client()
client.username_pw_set(access_token)

while True:
 humidity, temperature = Adafruit_DHT
 .read_retry(dht_sensor, pin)

 print(u"Temperature: {:g}\u00b0C, Humidity
 {:g}%".format(temperature, humidity))

 sensor_data['temperature'] = temperature
 sensor_data['humidity'] = humidity
 client.connect(host, 1883, 20)
 client.publish('v1/devices/me/telemetry', 
 json.dumps(sensor_data), 1)
 client.disconnect()
 sleep(10)
```

1.  您应该在shell中看到类似以下截图的输出：

![](assets/b8f64a58-60a6-4d41-967e-0e21c3778318.png)

1.  每10秒应该有一个新行。正如您所看到的，房间里又热又潮。

让我们更仔细地看一下前面的代码：

1.  我们的`import`语句让我们可以访问代码所需的模块：

```py
from time import sleep
import Adafruit_DHT
import paho.mqtt.client as mqtt
import json
```

我们已经熟悉了`sleep`，`Adafruit_DHT`和`json`。`Paho MQTT`库让我们可以访问`client`对象，我们将使用它来将我们的传感器数据发布到仪表板。

1.  代码中的接下来两行用于设置`demo`服务器的URL和我们之前从设备检索到的访问令牌的变量。我们需要这两个值才能连接到MQTT服务器并发布我们的传感器数据：

```py
host = 'demo.thingsboard.io'
access_token = '<<access token>>'
```

1.  我们将`dht_sensor`变量定义为`Adafruit`库中的`DHT11`对象。我们使用传感器的引脚`19`：

```py
dht_sensor = Adafruit_DHT.DHT11
pin = 19
```

1.  然后我们定义一个`dictionary`对象来存储将发布到MQTT服务器的传感器数据：

```py
sensor_data = {'temperature': 0, 'humidity': 0}
```

1.  然后我们创建一个`mqtt Client`类型的`client`对象。用户名和密码使用代码中先前定义的`access_token`设置：

```py
client = mqtt.Client()
client.username_pw_set(access_token)
```

1.  连续的`while`循环包含读取传感器数据的代码，然后将其发布到MQTT服务器。通过从`read_retry`方法读取湿度和温度，并将相应的`sensor_data`字典值设置如下：

```py
while True:
    humidity, temperature = Adafruit_DHT
                                .read_retry(dht_sensor, pin)

    print(u"Temperature: {:g}\u00b0C, Humidity
               {:g}%".format(temperature, humidity))

    sensor_data['temperature'] = temperature
    sensor_data['humidity'] = humidity
```

1.  以下`client`代码是负责将我们的传感器数据发布到MQTT服务器的代码。我们使用`client`对象的`connect`方法连接，传入主机值、端口（默认端口）和`20`秒的保持活动时间。与许多MQTT示例不同，我们不创建循环并寻找回调，因为我们只对发布传感器值感兴趣，而不是订阅主题。在这种情况下，我们要发布的主题是`v1/devices/me/telemetry`，如ThingsBoard文档示例代码所示。然后我们断开与`client`的连接：

```py
client.connect(host, 1883, 20)
client.publish('v1/devices/me/telemetry', 
            json.dumps(sensor_data), 1)
client.disconnect()
sleep(10)
```

我们现在将在ThingsBoard中创建一个仪表板，以显示从我们的代码发送的传感器值。

# 在ThingsBoard中创建仪表板

以下是将湿度值添加到仪表板的步骤：

1.  返回ThingsBoard，单击“设备”，然后单击“ROOM CONDITIONS”。侧边菜单应该从右侧滑出：

![](assets/e4a0f8a8-5fb8-457e-89a9-d6db6d79d54a.png)

1.  单击“最新遥测”选项卡。

1.  您应该看到湿度和温度的值，以及上次更新这些值的时间。通过单击左侧的复选框选择湿度。现在，单击“在小部件上显示”：

![](assets/d9de168d-ad4f-4b08-b757-c5d7553a3fd2.png)

1.  选择当前捆绑到模拟表盘，并循环浏览表盘，直到找到湿度表盘小部件。单击“添加到仪表板”按钮：

![](assets/3d991f91-b3c7-4a34-b75c-a5aebe6bd90e.png)

1.  选择创建新仪表板，并输入`Room Conditions`作为名称：

![](assets/7f4d0b0c-8d41-4ba4-8d7e-eca154a676ba.png)

1.  不要选择“打开仪表板”复选框。单击“添加”按钮。

1.  重复上述步骤以添加温度值。选择温度小部件，并将小部件添加到“Room Conditions”仪表板。这次，在单击“添加”之前选择“打开仪表板”：

![](assets/fa7423ba-e6b0-4d5c-ac23-d2d2a348952b.png)

现在，您应该看到一个仪表板，其中显示了湿度和温度值，显示在模拟表盘上。

# 与朋友分享您的仪表板

如果您想要将此仪表板公开，以便其他人可以看到它，您需要执行以下操作：

1.  通过单击“DASHBOARDS”导航到仪表板屏幕：

![](assets/5ea867db-fe5f-48e6-aacd-5f3633c598de.png)

1.  单击“使仪表板公开”选项：

![](assets/bd1baa9b-615e-4340-8ed7-212823a62971.png)

1.  您将看到对话框显示“仪表板现在是公开的”，如下截图所示。您可以复制并粘贴URL，或通过社交媒体分享：

![](assets/10b1a040-1240-4aee-8848-b2c98cb8eb13.png)

# 设置用于文本消息传输的账户

在本节中，我们将连接到一个文本消息传输服务，并从树莓派向我们的手机发送一条短信。我们将利用这些信息以及我们迄今为止关于发布感知信息的所学知识，来创建一个增强版的安全仪表板，位于[第 9 章](1a50006e-75d3-4dc1-96db-82114b437795.xhtml)，“构建家庭安全仪表板”中。

# 设置 Twilio 账户

Twilio 是一个服务，它为软件开发人员提供通过其网络服务 API 来编程创建和接收文本和电话通话的能力。让我们从设置 Twilio 账户开始：

1.  在网页浏览器中，导航至 [www.twilio.com](http://www.twilio.com)

1.  点击页面右上角的红色注册按钮

1.  输入适当的个人信息和密码，然后选择短信、到达提醒和 Python 作为密码下面的字段：

![](assets/4dda7c52-477c-4c6e-b693-bcd26b0f1239.png)

1.  提供一个电话号码，以便通过短信接收授权码，如下所示：

![](assets/1ce5138e-b3f3-4c70-aaee-1e3368e4aa3d.png)

1.  输入您收到的授权码，如下所示：

![](assets/b4d3f813-f5e9-4206-808e-2cd1f811890a.png)

1.  下一步是为您将要使用的项目命名。我们将其命名为`Doorbell`。输入名称并点击“继续”：

![](assets/85292270-6310-40e4-ade5-646e203a0415.png)

1.  我们需要一个账户的电话号码才能与其进行交互。点击获取号码：

![](assets/b9cf840a-56b8-44a5-a41d-deb6258174a9.png)

1.  将向您呈现一个号码。如果这个号码适合您，请点击“选择此号码”：

![](assets/f0c888ec-7c2f-4a32-99a1-426eac78c2fb.png)

1.  您现在已经设置好并准备使用 Twilio：

![](assets/f8fcd1de-66d3-455f-8b87-d1f37e36f7bb.png)

Twilio 是一个付费服务。您将获得一个初始金额来使用。请在创建应用程序之前检查使用此服务的成本。

# 在我们的树莓派上安装 Twilio

要从 Python 访问 Twilio，我们需要安装`twilio`库。打开终端并输入以下内容：

```py
pip3 install twilio
```

您应该在终端中看到 Twilio 安装的进度。

# 通过 Twilio 发送短信

在发送短信之前，我们需要获取凭据。在您的 Twilio 账户中，点击“设置”|“常规”，然后滚动到“API 凭据”：

![](assets/287a1ff0-1d05-4954-a6ca-33d3d1cef52a.png)

我们将使用 LIVE 凭据和 TEST 凭据的值。打开 Thonny 并创建一个名为`twilio-test.py`的新文件。在文件中输入以下代码并运行。确保粘贴 LIVE 凭据（请注意，发送短信将收取您的账户费用）：

```py
from twilio.rest import Client

account_sid = '<<your account_sid>>'
auth_token = '<<your auth_token>>'
client = Client(account_sid, auth_token)

message = client.messages.create(
                              body='Twilio says hello!',
                              from_='<<your Twilio number>>',
                              to='<<your cell phone number>>'
                          )
print(message.sid)
```

您应该会在您的手机上收到一条消息“Twilio 问候！”的短信。

# 创建一个新的家庭安全仪表板

在[第 9 章](1a50006e-75d3-4dc1-96db-82114b437795.xhtml)，“构建家庭安全仪表板”中，我们使用 CherryPy 创建了一个家庭安全仪表板。物联网的强大之处在于能够构建一个连接到世界各地设备的应用程序。我们将把这个想法应用到我们的家庭安全仪表板上。如果尚未组装，请使用[第 9 章](1a50006e-75d3-4dc1-96db-82114b437795.xhtml)，“构建家庭安全仪表板”中的温度传感器来构建家庭安全仪表板：

1.  我们将通过将我们的感知数据封装在一个“类”容器中来开始我们的代码。打开 Thonny 并创建一个名为`SensoryData.py`的新文件：

```py
from gpiozero import MotionSensor
import Adafruit_DHT

class SensoryData:
    humidity=''
    temperature=''
    detected_motion=''

    dht_pin = 19
    dht_sensor = Adafruit_DHT.DHT11
    motion_sensor = MotionSensor(4)

    def __init__(self):
        self.humidity, self.temperature = Adafruit_DHT
                            .read_retry(self.dht_sensor, 
                            self.dht_pin)

        self.motion_detected = self.motion_sensor.motion_detected

    def getTemperature(self):
        return self.temperature

    def getHumidity(self):
        return self.humidity

    def getMotionDetected(self):
        return self.motion_detected

if __name__ == "__main__":

    while True:
        sensory_data = SensoryData()
        print(sensory_data.getTemperature())
        print(sensory_data.getHumidity())
        print(sensory_data.getMotionDetected())

```

1.  运行程序来测试我们的传感器。这里没有我们尚未涵盖的内容。基本上我们只是在测试我们的电路和传感器。您应该在 shell 中看到感知数据的打印。 

1.  现在，让我们创建我们的感知仪表板。打开 Thonny 并创建一个名为`SensoryDashboard.py`的新文件。代码如下：

```py
import paho.mqtt.client as mqtt
import json
from SensoryData import SensoryData
from time import sleep

class SensoryDashboard:

    host = 'demo.thingsboard.io'
    access_token = '<<your access_token>>'
    client = mqtt.Client()
    client.username_pw_set(access_token)
    sensory_data = ''

    def __init__(self, sensoryData):
        self.sensoryData = sensoryData

    def publishSensoryData(self):
        sensor_data = {'temperature': 0, 'humidity': 0,
                        'Motion Detected':False}

        sensor_data['temperature'] =  self.sensoryData
                                        .getTemperature()

        sensor_data['humidity'] = self.sensoryData.getHumidity()

        sensor_data['Motion Detected'] = self.sensoryData
                                        .getMotionDetected()

        self.client.connect(self.host, 1883, 20)
        self.client.publish('v1/devices/me/telemetry',         
                                json.dumps(sensor_data), 1)
        self.client.disconnect()

        return sensor_data['Motion Detected']

if __name__=="__main__":

    while True:
        sensoryData = SensoryData()
        sensory_dashboard = SensoryDashboard(sensoryData)

        print("Motion Detected: " +             
                str(sensory_dashboard.publishSensoryData()))

        sleep(10)
```

我们在这里所做的是将以前的代码中的`dht-mqtt.py`文件封装在一个`class`容器中。我们用一个`SensoryData`对象来实例化我们的对象，以便从传感器获取数据。`publishSensoryData()`方法将感官数据发送到我们的MQTT仪表板。注意它如何返回运动传感器的状态？我们在主循环中使用这个返回值来打印出运动传感器的值。然而，这个返回值在我们未来的代码中会更有用。

让我们将运动传感器添加到我们的ThingsBoard仪表板中：

1.  在浏览器中打开ThingsBoard

1.  点击设备菜单

1.  点击房间条件设备

1.  选择最新的遥测

1.  选择检测到的运动值

1.  点击小部件上的显示

1.  在卡片下面，找到由一个大橙色方块组成的小部件，如下所示：

![](assets/6cfc62b4-a7ca-4364-8662-c59f77450cb0.png)

1.  点击添加到仪表板

1.  选择现有的房间条件仪表板

1.  选中打开仪表板

1.  点击添加

您应该看到新的小部件已添加到房间条件仪表板。通过点击页面右下角的橙色铅笔图标，您可以移动和调整小部件的大小。编辑小部件，使其看起来像以下的屏幕截图：

![](assets/f83cee48-8874-4e06-92e9-86ec54584e03.png)

我们在这里所做的是重新创建[第9章](1a50006e-75d3-4dc1-96db-82114b437795.xhtml)中的家庭安全仪表板的第一个版本，*构建家庭安全仪表板*，并采用了更加分布式的架构。我们不再依赖于我们的树莓派通过CherryPy网页提供感官信息。我们能够将我们的树莓派的角色减少到感官信息的来源。正如您所能想象的，使用多个树莓派来使用相同的仪表板非常容易。

通过靠近PIR传感器来测试这个新的仪表板。看看能否使检测到运动的小部件变为`true`。

为了使我们的新家庭安全仪表板更加分布式，让我们添加在PIR运动传感器激活时发送文本消息的功能。打开Thonny并创建一个名为`SecurityDashboardDist.py`的新文件。以下是要插入文件的代码：

```py
from twilio.rest import Client
from SensoryData import SensoryData
from SensoryDashboard import SensoryDashboard
from gpiozero import Button
from time import time, sleep

class SecurityDashboardDist:
    account_sid = ''
    auth_token = ''
    time_sent = 0
    test_env = True 
    switch = Button(8)

    def __init__(self, test_env = True):
        self.test_env = self.setEnvironment(test_env)

    def setEnvironment(self, test_env):
        if test_env:
            self.account_sid = '<<your Twilio test account_sid>>'
            self.auth_token = '<<your Twilio test auth_token>>'
            return True
        else:
            self.account_sid = '<<your Twilio live account_sid>>'
            self.auth_token = '<<your Twilio live auth_token>>'
            return False

    def update_dashboard(self, sensoryDashboard):
        self.sensoryDashboard = sensoryDashboard

        motion_detected = self
                          .sensoryDashboard
                          .publishSensoryData()

        if motion_detected:
            return self.send_alert()
        else:
            return 'Alarm not triggered'

    def send_alert(self):
        if self.switch.is_pressed:
            return self.sendTextMessage()
        else:
            return "Alarm triggered but Not Armed"

    def sendTextMessage(self):
        message_interval = round(time() - self.time_sent)

        if message_interval > 600:
            twilio_client = Client(self.account_sid, 
                                   self.auth_token)

            if self.test_env:
                message = twilio_client.messages.create(
                            body='Intruder Alert',
                            from_= '+15005550006',
                            to='<<your cell number>>'
                          )
            else:
                message = twilio_client.messages.create(
                            body='Intruder Alert',
                            from_= '<<your Twilio number>>',
                            to='<<your cell number>>'
                          )

            self.time_sent=round(time())

            return 'Alarm triggered and text message sent - ' 
                    + message.sid
        else:
             return 'Alarm triggered and text 
                    message sent less than 10 minutes ago'   

if __name__=="__main__":  
    security_dashboard = SecurityDashboardDist()

    while True:
        sensory_data = SensoryData()
        sensory_dashboard = SensoryDashboard(sensory_data)
        print(security_dashboard.update_dashboard(
                sensory_dashboard))

        sleep(5)

```

利用[第9章](1a50006e-75d3-4dc1-96db-82114b437795.xhtml)中的家庭安全仪表板电路的第一个版本，*构建家庭安全仪表板*，这段代码使用钥匙开关来激活发送文本消息的呼叫，如果运动传感器检测到运动。当钥匙开关处于关闭位置时，每当运动传感器检测到运动时，您将收到一条消息，内容为`警报触发但未激活`。

如果还没有打开，请打开钥匙开关以激活电路。通过四处移动来激活运动传感器。您应该会收到一条通知，说明已发送了一条文本消息。消息的SID也应该显示出来。您可能已经注意到，您实际上并没有收到一条文本消息。这是因为代码默认为Twilio测试环境。在我们打开实时环境之前，让我们先看一下代码。

我们首先导入我们代码所需的库：

```py
from twilio.rest import Client
from SensoryData import SensoryData
from SensoryDashboard import SensoryDashboard
from gpiozero import Button
from time import time, sleep
```

这里没有太多我们以前没有见过的东西；然而，请注意`SensoryData`和`SensoryDashboard`的导入。由于我们已经封装了读取感官数据的代码，现在我们可以把它看作一个黑匣子。我们知道我们需要安全仪表板的感官数据，但我们不关心如何获取这些数据以及它将在哪里显示。`SensoryData`为我们提供了我们需要的感官数据，`SensoryDashboard`将其发送到某个仪表板。在我们的`SecurityDashboardDist.py`代码中，我们不必关心这些细节。

我们为我们的分布式安全仪表板创建了一个名为`SecurityDashboardDist`的类。重要的是要通过它们的名称来区分我们的类，并选择描述`class`是什么的名称。

```py
class SecurityDashboardDist:
```

在声明了一些整个类都可以访问的类变量之后，我们来到了我们的类初始化方法：

```py
    account_sid = ''
    auth_token = ''
    time_sent = 0
    test_env = True 
    switch = Button(8)

    def __init__(self, test_env = True):
        self.test_env = self.setEnvironment(test_env)
```

在`initialization`方法中，我们设置了类范围的`test_env`变量（用于`test`环境）。默认值为`True`，这意味着我们必须有意地覆盖默认值才能运行实时仪表板。我们使用`setEnvironment()`方法来设置`test_env`：

```py
def setEnvironment(self, test_env):
        if test_env:
            self.account_sid = '<<your Twilio test account_sid>>'
            self.auth_token = '<<your Twilio test auth_token>>'
            return True
        else:
            self.account_sid = '<<your Twilio live account_sid>>'
```

```py
            self.auth_token = '<<your Twilio live auth_token>>'
            return False
```

`setEnvironment()`方法根据`test_env`的值设置类范围的`account_id`和`auth_token`值，以便设置测试环境或实际环境。基本上，我们只是通过`setEnvironment()`方法传回`test_env`的状态，同时设置我们需要启用测试或实际短信环境的变量。

`update_dashboard()`方法通过传入的`SensoryDashboard`对象调用传感器和感官仪表板。这里是我们采取的面向对象方法的美妙之处，因为我们不需要关心传感器是如何读取的或仪表板是如何更新的。我们只需要传入一个`SensoryDashboard`对象就可以完成这个任务。

```py
def update_dashboard(self, sensoryDashboard):
        self.sensoryDashboard = sensoryDashboard

        motion_detected = self
                          .sensoryDashboard
                          .publishSensoryData()

        if motion_detected:
            return self.send_alert()
        else:
            return 'Alarm not triggered'
```

`update_dashboard`方法还负责确定是否发送短信，通过检查运动传感器的状态。您还记得我们在调用`SensoryDashboard`类的`publishSensoryData()`方法时返回了运动传感器的状态吗？这就是它真正方便的地方。我们可以使用这个返回值来确定是否应该发送警报。我们根本不需要在我们的类中检查运动传感器的状态，因为它可以很容易地从`SensoryDashboard`类中获得。

`send_alert()`方法检查开关的状态，以确定是否发送短信：

```py
def send_alert(self):
        if self.switch.is_pressed:
            return self.sendTextMessage()
        else:
            return "Alarm triggered but Not Armed"
```

也许你会想知道为什么我们在这里检查传感器（在这种情况下是开关）的状态，而不是从`SensoryDashboard`类中检查。答案是？我们正在通过封装传感数据仪表板来构建家庭安全仪表板。`SensorDashboard`类中不需要开关，因为它不涉及从GPIO到MQTT仪表板的传感数据的读取和传输。开关是安全系统的领域；在这种情况下是`SecurityDashboardDist`类。

`SecurityDasboardDist`类的核心是`sendTextMessage()`方法，如下所述：

```py
def sendTextMessage(self):
        message_interval = round(time() - self.time_sent)

        if message_interval > 600:
            twilio_client = Client(self.account_sid, 
                                   self.auth_token)

            if self.test_env:
                message = twilio_client.messages.create(
                            body='Intruder Alert',
                            from_= '+15005550006',
                            to='<<your cell number>>'
                          )
            else:
                message = twilio_client.messages.create(
                            body='Intruder Alert',
                            from_= '<<your Twilio number>>',
                            to='<<your cell number>>'
                          )

            self.time_sent=round(time())

            return 'Alarm triggered and text message sent - ' 
                    + message.sid
        else:
             return 'Alarm triggered and text 
                    message sent less than 10 minutes ago'   
```

我们使用`message_interval`方法变量来设置短信之间的时间间隔。我们不希望每次运动传感器检测到运动时都发送短信。在我们的情况下，短信之间的最短时间为`600`秒，或`10`分钟。

如果这是第一次，或者距离上次发送短信已经超过10分钟，那么代码将在测试环境或实际环境中发送短信。请注意`15005550006`电话号码在测试环境中的使用。实际环境需要您的Twilio号码，并且您自己的电话号码用于`to`字段。对于测试和实际环境，都会返回`触发警报并发送短信`的消息，然后是消息的SID。不同之处在于您实际上不会收到短信（尽管代码中有调用Twilio）。

如果距上次发送短信不到10分钟，则消息将显示`触发警报并发送短信不到10分钟`。

在我们的主函数中，我们创建了一个`SecurityDashboardDist`对象，并将其命名为`security_dashboard`。通过不传入任何内容，我们允许默认情况下设置测试环境的仪表板：

```py
if __name__=="__main__":  
    security_dashboard = SecurityDashboardDist()

    while True:
        sensory_data = SensoryData()
        sensory_dashboard = SensoryDashboard(sensory_data)
        print(security_dashboard.update_dashboard(
                sensory_dashboard))

        sleep(5)
```

随后的连续循环每5秒创建一个`SensoryData`和`SensoryDashboard`对象。`SensoryData`对象（`sensory_data`）用于实例化`SensoryDashboard`对象（`sensory_dashboard`），因为前者提供当前的感官数据，后者创建感官仪表板。

通过根据它们的名称命名我们的类，以及根据它们的功能命名我们的方法，代码变得相当自解释。

然后我们将这个`SensoryDashboard`对象(`sensory_dashboard`)传递给`SecurityDashboard`(`security_dashboard`)的`update_dashboard`方法。由于`update_dashboard`方法返回一个字符串，我们可以用它来打印到我们的shell，从而看到我们的仪表板每5秒打印一次状态。我们将`SecurityDashboardDist`对象的实例化放在循环之外，因为我们只需要设置环境一次。

现在我们了解了代码，是时候在实际的Twilio环境中运行它了。请注意，当我们切换到实际环境时，代码中唯一改变的部分是实际发送短信。要将我们的仪表板变成一个实时发送短信的机器，只需将主方法的第一行更改为以下内容：

```py
security_dashboard = SecurityDashboardDist(True)
```

# 摘要

完成本章后，我们应该非常熟悉将感应数据发布到物联网仪表板。我们还应该熟悉使用Twilio网络服务从树莓派发送短信。

我们将在[第11章](1668a45a-408f-4732-8643-623297983690.xhtml)中查看蓝牙库，*使用蓝牙创建门铃按钮*，然后将这些信息和我们在本章中获得的信息结合起来，制作一个物联网门铃。

# 问题

1.  我们用来从树莓派发送短信的服务的名称是什么？

1.  真或假？我们使用PIR传感器来读取温度和湿度值。

1.  如何在ThingsBoard中创建仪表板？

1.  真或假？我们通过使用感应仪表板来构建我们的增强安全仪表板。

1.  我们用来读取温度和湿度感应数据的库的名称是什么？

1.  真或假？我们需要预先安装用于发送短信的库与Raspbian一起。

1.  在我们的代码中命名类时，我们试图做什么？

1.  真或假？为了将我们的环境从测试切换到实际，我们是否需要重写增强家庭安全仪表板中的整个代码。

1.  真或假？我们Twilio账户的`account_sid`号码在实际环境和测试环境中是相同的。

1.  在我们的`SecurityDashboardDist.py`代码中，我们在哪里创建了`SecurityDashboardDist`对象？

# 进一步阅读

为了进一步了解Twilio和ThingsBoard背后的技术，请参考以下链接：

+   Twilio文档：[https://www.twilio.com/docs/quickstart](https://www.twilio.com/docs/quickstart)

+   ThingsBoard的文档：

[https://thingsboard.io/docs/](https://thingsboard.io/docs/)
