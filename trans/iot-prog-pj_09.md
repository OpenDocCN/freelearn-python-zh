# 构建家庭安全仪表板

在第七章中，*设置树莓派 Web 服务器*，我们介绍了 web 框架 CherryPy。使用 CherryPy，我们可以将树莓派变成一个 Web 服务器。在第八章中，*使用 Python 读取树莓派 GPIO 传感器数据*，我们学会了如何从 GPIO 读取传感器数据。

在本章中，我们将从前两章学到的经验中创建一个家庭安全仪表板。

本章将涵盖以下主题：

+   使用 CherryPy 创建我们的仪表板

+   在我们的仪表板上显示传感器数据

# 完成本章所需的知识

读者需要对 Python 编程语言有一定的了解才能完成本章。还需要基本了解 HTML，包括 CSS。

# 项目概述

在本章中，我们将构建两个不同的家庭安全仪表板。第一个将涉及使用温度和湿度传感器，下一个将涉及使用有源蜂鸣器。

这个项目应该需要几个小时才能完成。

# 入门

要完成此项目，需要以下内容：

+   树莓派 3 型（2015 年型号或更新型号）

+   一个 USB 电源适配器

+   一个计算机显示器

+   一个 USB 键盘

+   USB 鼠标

+   一个面包板

+   DHT11 温度传感器

+   一个锁存按钮、开关或键开关

+   一个 PIR 传感器

+   一个有源蜂鸣器

+   树莓派摄像头模块

# 使用 CherryPy 创建我们的仪表板

为了创建我们的家庭安全仪表板，我们将修改我们在第七章中编写的代码，*设置树莓派 Web 服务器*。这些修改包括添加来自 GPIO 的传感器数据——这是我们在第八章结束时变得非常擅长的事情，*使用 Python 读取树莓派 GPIO 传感器数据*。

两个输入，温度和湿度传感器以及树莓派摄像头，将需要额外的步骤，以便我们可以将它们整合到我们的仪表板中。

# 使用 DHT11 查找温度和湿度

DHT11 温度和湿度传感器是一种低成本的业余级传感器，能够提供基本的测量。DHT11 有两种不同的版本，四针模型和三针模型。

我们将在我们的项目中使用三针模型（请参阅以下图片）：

![](img/05c744f6-b954-47ce-9212-2af4eecdfbac.png)

我们将使用`Adafruit DHT`库来读取 DHT11 数据，该库在 Raspbian 上没有预安装（截至撰写时）。要安装它，我们将克隆库的 GitHub 项目并从源代码构建它。

打开终端窗口，输入以下命令使用`git`并下载源代码（撰写时，`git`已预装在 Raspbian 中）：

```py
git clone https://github.com/adafruit/Adafruit_Python_DHT.git
```

您应该看到代码下载的进度。现在，使用以下命令更改目录：

```py
cd Adafruit_Python_DHT
```

您将在`源代码`目录中。

使用以下命令构建项目：

```py
sudo python3 setup.py install
```

您应该在终端中看到显示的进度：

![](img/d072cdb7-4d9b-434e-a8ff-140ffe801ade.png)

如果您没有收到任何错误，`Adafruit DHT`库现在应该已安装在您的树莓派上。要验证这一点，打开 Thonny 并检查包：

![](img/f7b5c282-0a2d-4cbb-88a3-08fd18a698b0.png)

现在，让我们连接电路。将 DHT11 传感器连接到树莓派如下：

+   DHT11 的 GND 连接到树莓派的 GND

+   DHT11 的 VCC 连接到树莓派的 5V DC

+   DHT11 的信号连接到 GPIO 引脚 19

有关更多信息，请参阅以下 Fritzing 图表：

![](img/6caf4ad6-d61c-44e0-8d83-812dadc76f80.png)

一旦 DHT11 连接好，就是写一些代码的时候了：

1.  从应用程序菜单 | 编程 | Thonny Python IDE 打开 Thonny

1.  点击“新建”创建一个新文件

1.  在文件中输入以下内容：

```py
import Adafruit_DHT

dht_sensor = Adafruit_DHT.DHT11
pin = 19
humidity, temperature = Adafruit_DHT.read_retry(dht_sensor, pin)

print(humidity)
print(temperature)
```

1.  将文件保存为`dht-test.py`

1.  运行代码

1.  您应该看到类似以下的内容：

![](img/e261501e-b9b0-4a45-a941-c1d50a855082.png)

让我们看看代码。我们将从导入`Adafruit_DHT`库开始。然后我们创建一个新的`DHT11`对象，并将其命名为`dht_sensor`。`湿度`和`温度`是从`Adafruit_DHT`类的`read_retry`方法中设置的。

然后我们在 shell 中打印出`湿度`和`温度`的值。

# 使用 Pi 相机拍照

在第三章中，*使用 GPIO 连接到外部世界*，我们尝试了特殊的树莓派相机模块，并编写了代码来打开相机预览。是时候把相机投入使用了。

通过 CSI 相机端口将树莓派相机模块安装到树莓派上（如果尚未启用，请确保在树莓派配置屏幕中启用相机）。让我们写一些代码：

1.  从应用程序菜单中打开 Thonny | 编程 | Thonny Python IDE

1.  单击“新建”以创建新文件

1.  在文件中输入以下内容：

```py
from picamera import PiCamera
from time import sleep

pi_cam = PiCamera()

pi_cam.start_preview()
sleep(5)
pi_cam.capture('/home/pi/myimage.png')
pi_cam.stop
```

1.  将文件保存为`pi-camera-test.py`

1.  运行代码

该程序导入`PiCamera`并在创建一个名为`pi_cam`的新`PiCamera`对象之前休眠。`start_preview`方法向我们显示相机在全屏中看到的内容。

捕获方法创建一个名为`myimage.png`的新图像文件，并将其存储在默认目录`/home/pi`中。

我们有 5 秒的时间来调整相机的位置，然后拍照。

以下是使用树莓派相机拍摄的我的工作区的照片：

![](img/501926f5-b41a-4fb4-bb94-da115688f2e9.png)

# 使用 CherryPy 创建我们的仪表板

在第七章中，*设置树莓派 Web 服务器*，我们使用 Bootstrap 框架和`WeatherDashboardHTML.py`文件创建了一个天气仪表板。我们将重新访问该代码，并修改为我们的家庭安全仪表板。

要创建我们的家庭安全仪表板，请执行以下操作：

1.  从应用程序菜单中打开 Thonny | 编程 | Thonny Python IDE

1.  单击“新建”以创建新文件

1.  在文件中输入以下内容：

```py
import cherrypy
from SecurityData import SecurityData

class SecurityDashboard:

    def __init__(self, securityData):
        self.securityData = securityData

    @cherrypy.expose
    def index(self):
        return """
               <!DOCTYPE html>
                <html lang="en">

                <head>
                    <title>Home Security Dashboard</title>
                    <meta charset="utf-8">
                    <meta name="viewport"
                        content="width=device-width,
                        initial-scale=1">

                    <meta http-equiv="refresh" content="30">

                    <link rel="stylesheet"         
                        href="https://maxcdn.bootstrapcdn.com
                        /bootstrap/4.1.0/css/bootstrap.min.css">

                    <link rel="stylesheet" href="led.css">

                    <script src="https://ajax.googleapis.com
                        /ajax/libs/jquery/3.3.1/jquery.min.js">                
                    </script>

                    <script src="https://cdnjs.cloudflare.com
                        /ajax/libs/popper.js/1.14.0
                        /umd/popper.min.js">
                    </script>

                    <script src="https://maxcdn.bootstrapcdn.com
                        /bootstrap/4.1.0/js/bootstrap.min.js">
                    </script>

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
                                <h3>Home Security Dashboard</h3>
                             </div>
                             <div class="card-body">
                                <div class="row">
                                    <div class="col element-box">
                                        <h6>Armed</h6>
                                        <div class = """ +     
                                            self.securityData
                                            .getArmedStatus() + 
                                        """>
                                        </div>
                                    </div>
                                    <div class="col element-box">
                                        <h6>Temp / Humidity</h6>
                                        <p>""" + self.securityData
                                            .getRoomConditions() 
                                        + """</p>
                                    </div>
                                    <div class="col element-box">
                                        <h6>Last Check:</h6>
                                        <p>""" + self
                                            .securityData.getTime() 
                                         + """</p>
                                    </div>
                                </div>
                            </div>
                            <div class="card-footer" 
                                       align="center">

                                <img src=""" + self.securityData
                                    .getSecurityImage() + """/>
                                <p>""" + self.securityData
                                    .getDetectedMessage() + """</p>
                            </div>
                        </div>
                    </div>
                </body>

                </html>
               """

if __name__=="__main__":
    securityData = SecurityData()
    conf = {
        '/led.css':{
            'tools.staticfile.on': True,
            'tools.staticfile.filename': '/home/pi/styles/led.css'
            },
        '/intruder.png':{
            'tools.staticfile.on': True,
            'tools.staticfile.filename':                            
                '/home/pi/images/intruder.png'
            },
        '/all-clear.png':{
            'tools.staticfile.on': True,
            'tools.staticfile.filename': '/home/pi/images
                /all-clear.png'
            },
        '/not-armed.png':{
            'tools.staticfile.on': True,
            'tools.staticfile.filename': '/home/pi
                /images/not-armed.png'
            }
    }
    cherrypy.quickstart(SecurityDashboard(securityData),'/',conf)
```

1.  将文件保存为`security-dashboard.py`

尚未运行代码，因为我们还需要创建`SecurityData`类。

正如您所看到的，我们对`WeatherDashboardHTML.py`进行了一些更改，以创建`security-dashboard.py`。在运行代码之前，让我们指出一些更改。

最明显的变化是使用了`SecurityData`类。可以想象，这个类将用于获取我们仪表板的数据：

```py
from SecurityData import SecurityData
```

我们使用以下行来每 30 秒自动刷新我们的页面（我们没有自动刷新我们的天气仪表板，因为天气数据不经常变化）：

```py
<meta http-equiv="refresh" content="30">
```

对于我们的家庭安全仪表板，我们使用一些 CSS 魔术来表示闪烁的 LED。这是通过添加`led.css`文件来实现的：

```py
<link rel="stylesheet" href="led.css">
```

对于数据字段，我们将从我们的`SecurityData`对象中访问方法。我们将在接下来的部分详细介绍这些方法。对于我们的主要部分，我们将创建一个名为`conf`的字典：

```py
if __name__=="__main__":
    securityData = SecurityData()
    conf = {
        '/led.css':{
            'tools.staticfile.on': True,
            'tools.staticfile.filename': '/home/pi/styles/led.css'
            },
        '/intruder.png':{
            'tools.staticfile.on': True,
            'tools.staticfile.filename':                            
                '/home/pi/images/intruder.png'
            },
        '/all-clear.png':{
            'tools.staticfile.on': True,
            'tools.staticfile.filename': '/home/pi/images
                /all-clear.png'
            },
        '/not-armed.png':{
            'tools.staticfile.on': True,
            'tools.staticfile.filename': '/home/pi
                /images/not-armed.png'
            }
    }
    cherrypy.quickstart(SecurityDashboard(securityData),'/',conf)

```

我们使用`conf`字典将配置数据传递给`cherrypy quickstart`方法。此配置数据允许我们在 CherryPy 服务器中使用静态文件`led.css`，`intruder.png`，`all-clear.png`和`not-armed.png`。

先前提到了 CSS 文件`led.css`。其他三个文件是我们仪表板中使用的自描述图像。

为了在 CherryPy 中使用静态文件或目录，您必须创建并传递配置信息。配置信息必须包含绝对路径（而不是相对路径）。

配置信息说明 CSS 和图像文件分别位于名为`styles`和`images`的目录中。这些目录都位于`/home/pi`目录中。

以下是`images`目录中文件的屏幕截图（请确保将文件放在正确的目录中）：

![](img/54f70c24-38f3-45dd-9eee-7db79364230f.png)

# 在我们的仪表板上显示传感器数据

为了提供我们的仪表板数据，我们将创建一个名为`SecurityData.py`的新 Python 文件，我们将在其中存储`SecurityData`类。在这之前，让我们先建立我们的电路。

# 带有温度传感器的家庭安全仪表板

我们将使用 DHT11 温湿度传感器、PIR 传感器和一个 latching 按钮（或钥匙开关）来构建家庭安全仪表板的第一个版本。以下是我们家庭安全仪表板的 Fritzing 图表：

![](img/56fe1cf8-ea5b-4089-9019-42d61ca7b0b4.png)

电路连接如下：

+   DHT11 的 GND 连接到 GND

+   DHT11 的 VCC 连接到 5V 直流电源

+   DHT11 的信号连接到 GPIO 引脚 19

+   PIR 传感器的 GND 连接到 GND

+   PIR 传感器的 VCC 连接到 5V 直流电源

+   PIR 传感器的信号连接到 GPIO 引脚 4

+   拉 atching 按钮的一端连接到 GPIO 引脚 8

+   拉 atching 按钮的另一端接地

+   树莓派摄像头模块连接到 CSI 端口（未显示）

以下是我们电路的照片。需要注意的一点是我们为 DHT11 传感器使用了单独的面包板（更容易放在微型面包板上），以及钥匙开关代替 latching 按钮：

![](img/04677e64-df43-436f-a745-76726f0485fb.png)

现在是时候编写代码了：

1.  从应用程序菜单中打开 Thonny | 编程 | Thonny Python IDE

1.  点击“新建”创建一个新文件

1.  将以下内容输入文件：

```py
from gpiozero import MotionSensor
from gpiozero import Button
from datetime import datetime
from picamera import PiCamera
import Adafruit_DHT

class SecurityData:
    humidity=''
    temperature=''
    detected_message=''

    dht_pin = 19
    dht_sensor = Adafruit_DHT.DHT11
    switch = Button(8)
    motion_sensor = MotionSensor(4)
    pi_cam = PiCamera()

    def getRoomConditions(self):
        humidity, temperature = Adafruit_DHT
            .read_retry(self.dht_sensor, self.dht_pin)

        return str(temperature) + 'C / ' + str(humidity) + '%'

    def getDetectedMessage(self):
        return self.detected_message

    def getArmedStatus(self):
        if self.switch.is_pressed:
            return "on"
        else:
            return "off"

    def getSecurityImage(self):

        if not(self.switch.is_pressed):
            self.detected_message = ''
            return "/not-armed.png"

        elif self.motion_sensor.motion_detected:
            self.pi_cam.resolution = (500, 375)
            self.pi_cam.capture("/home/pi/images/intruder.png")
            self.detected_message = "Detected at: " + 
                self.getTime()
            return "/intruder.png"

        else:
            self.detected_message = ''
            return "/all-clear.png"

    def getTime(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":

    while True:
        security_data = SecurityData()
        print(security_data.getRoomConditions())
        print(security_data.getArmedStatus())
        print(security_data.getTime())
```

1.  将文件保存为`SecurityData.py`

1.  运行代码

您应该在 shell 中得到一个输出，指示房间中的`温度`和`湿度`水平，一个表示开关位置的`on`或`off`，以及当前时间。尝试打开和关闭开关，看看输出是否发生变化。

在运行仪表板代码（`security-dashboard.py`）之前，让我们先回顾一下`SecurityData`类。正如我们所看到的，代码的第一部分是我们已经熟悉的标准样板代码。`getRoomConditions`和`getDetectedMessage`方法要么是不言自明的，要么是我们已经讨论过的内容。

我们的`getArmedStatus`方法做了一个小技巧，以保持我们的代码简单而紧凑：

```py
def getArmedStatus(self):
    if self.switch.is_pressed:
        return "on"
    else:
        return "off"
```

我们可以看到`getArmedStatus`返回的是`on`或`off`，而不是大多数具有二进制返回的方法返回的`True`或`False`。我们这样做是为了我们仪表板代码的武装部分。

以下是`SecurityDashboard`类的`index`方法生成的 HTML 代码：

```py
<div class="col element-box">
    <h6>Armed</h6>
    <div class = """ + self.securityData.getArmedStatus() + """>
    </div>
</div>
```

正如我们所看到的，`getArmedStatus`方法在构建 div 标签时被调用，以替代 CSS 类名。单词`on`和`off`指的是我们`led.css`文件中的 CSS 类。当返回`on`时，我们得到一个闪烁的红色 LED 类型图形。当返回`off`时，我们得到一个黑点。

因此，拉 atching 开关（或钥匙开关）的位置决定了 div 标签是否具有 CSS 类名`on`或 CSS 类名`off`，通过`SecurityData`类的`getArmedStatus`方法。

我们的代码在`getSecurityImage`方法中变得非常有趣：

```py
def getSecurityImage(self):

        if not(self.switch.is_pressed):
            self.detected_message = ''
            return "/not-armed.png"

        elif self.motion_sensor.motion_detected:
            self.pi_cam.resolution = (500, 375)
            self.pi_cam.capture("/home/pi/images/intruder.png")
            self.detected_message = "Detected at: " + 
                self.getTime()
            return "/intruder.png"

        else:
            self.detected_message = ''
            return "/all-clear.png"
```

我们的第一个条件语句检查电路是否处于武装状态（开关处于“on”位置）。如果没有武装，那么我们只需要将检测到的消息设置为空，并返回对`not-armed.png`文件的引用（`/not-armed.png`在我们在`security-dashboard.py`文件中设置的配置信息中定义）。

如果我们看一下`SecurityDashboard`类（`security-dashboard.py`文件）中的代码，我们可以看到`getSecurityImage`方法在生成的 HTML 代码的底部附近被调用：

```py
<div class="card-footer" align="center">
    <img src=""" + self.securityData.getSecurityImage() + """/>
    <p>""" + self.securityData.getDetectedMessage() + """</p>
</div>
```

如果电路中的开关没有打开，我们将在仪表板页脚看到以下内容，后面没有描述（空的`detected_message`值）：

![](img/0a3c09bd-7a1f-4de6-b4c2-b55576a6eb85.png)

我们代码中的第二个条件语句是在开关处于`on`并且检测到运动时触发的。在这种情况下，我们设置我们树莓派摄像头的分辨率，然后拍照。

我们可能在类的实例化过程中设置了树莓派摄像头的分辨率，这可能更有意义。但是，将这行放在这里使得在完成代码之前调整分辨率更容易，因为这行存在于我们关注的方法中。

我们将文件命名为`intruder.png`，并将其存储在`security-dashboard.py`文件中的配置代码可以找到的位置。

我们还根据当前时间创建了一个`detected_message`值。这条消息将为我们从树莓派摄像头获取的图像提供时间戳。

最后的`else:`语句是我们返回`/all-clear.png`的地方。到达这一点时，我们知道开关是“开启”的，并且没有检测到任何运动。我们在仪表板页脚将看到以下图像：

![](img/681072ff-ecb3-46a7-8081-eaf54ec7bddd.png)

与“NOT ARMED”消息一样，在“ALL CLEAR”后面不会有描述。只有当开关处于“开启”状态且 PIR 传感器没有检测到任何运动（`motion_detected`为`false`）时，我们才会看到这个图形。

现在，让我们运行仪表板代码。如果您还没有这样做，请点击红色按钮停止`SecurityData`程序。点击`security-dashboard.py`文件的选项卡，然后点击运行。等待几秒钟，以便让 CherryPy 运行起来。

打开一个网络浏览器，然后导航到以下地址：

```py
http://127.0.0.1:8080
```

将开关置于“关闭”位置，您应该看到以下仪表板屏幕：

![](img/2747d648-0f9a-443a-b1c1-6b41b37f7b57.png)

正如我们所看到的，武装部分下的 LED 是黑色的，在页脚中会得到一个“NOT ARMED”消息。我们还可以看到`temperature`和`humidity`的显示，即使系统没有武装。

最后一个复选框显示了代码上次检查开关状态的时间。如果你等待 30 秒，你应该看到页面刷新并显示相同的信息。

现在，打开开关，站在一边，这样 PIR 传感器就不会检测到你。你应该看到一个类似于以下的屏幕：

![](img/59607798-8731-417c-a215-12ee661d22f3.png)

您会注意到武装部分的 LED 现在变成了闪烁的红色，`temperature`和`humidity`读数要么相同，要么略有不同，上次检查已更新到当前时间，并且页脚中出现了`ALL CLEAR`消息。

让我们看看是否能抓住入侵者。将树莓派摄像头对准门口，等待 PIR 传感器触发：

![](img/d3e9d7b6-cdec-4b3d-8562-6f73a6ccaf28.png)

看来我们已经抓到了入侵者！

# 具有快速响应的家庭安全仪表板

您可能已经注意到我们的页面刷新需要很长时间。当然，这是由于 30 秒的刷新时间，以及 DHT11 读取数值所需的长时间。

让我们改变我们的代码，使其更快，并给它一个蜂鸣器来吓跑入侵者。

用连接到 GPIO 引脚 17 的蜂鸣器替换 DHT11（对于这个简单的更改，我们不需要 Fritzing 图）。

我们将首先创建`SecurityDataQuick`数据类：

1.  从应用程序菜单中打开 Thonny | 编程 | Thonny Python IDE

1.  点击“新建”以创建一个新文件

1.  在文件中键入以下内容：

```py
from gpiozero import MotionSensor
from gpiozero import Button
from datetime import datetime
from picamera import PiCamera
from gpiozero import Buzzer
from time import sleep

class SecurityData:
    alarm_status=''
    detected_message=''

    switch = Button(8)
    motion_sensor = MotionSensor(4)
    pi_cam = PiCamera()
    buzzer = Buzzer(17)

    def sound_alarm(self):
        self.buzzer.beep(0.5,0.5, 5, True)
        sleep(1)

    def getAlarmStatus(self):

        if not(self.switch.is_pressed):
            self.alarm_status = 'not-armed'
            return "Not Armed"

        elif self.motion_sensor.motion_detected:
            self.alarm_status = 'motion-detected'
            self.sound_alarm()
            return "Motion Detected"

        else:
            self.alarm_status = 'all-clear'
            return "All Clear"

    def getDetectedMessage(self):
        return self.detected_message

    def getArmedStatus(self):
        if self.switch.is_pressed:
            return "on"
        else:
            return "off"

    def getSecurityImage(self):

        if self.alarm_status=='not-armed':
            self.detected_message = ''
            return "/not-armed.png"

        elif self.alarm_status=='motion-detected':
            self.pi_cam.resolution = (500, 375)
            self.pi_cam.capture("/home/pi/images/intruder.png")

            self.detected_message = "Detected at: " + 
                self.getTime()

            return "/intruder.png"

        else:
            self.detected_message = ''
            return "/all-clear.png"

    def getTime(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":

    while True:
        security_data = SecurityData()
        print(security_data.getArmedStatus())
        print(security_data.getTime())

```

1.  将文件保存为`SecurityDataQuick.py`

1.  运行代码

在我们的 shell 中，我们应该看到开关和当前时间的值。通过点击红色按钮停止程序。

正如我们所看到的，已经发生了一些变化。我们没有做的一个变化是更改类名。将其保持为`SecurityData`意味着以后对我们的仪表板代码的更改更少。

我们添加了`GPIO Zero`蜂鸣器的库，并删除了与 DHT11 传感器相关的任何代码。我们还创建了一个名为`sound_buzzer`的新方法，当检测到入侵者时我们将调用它。

添加了一个名为`alarm_status`的新变量，以及相应的`getAlarmStatus`方法。我们将类的核心逻辑移到了这个方法中（远离`getSecurityImage`），因为在这里我们检查开关和 PIR 传感器的状态。变量`alarm_status`在其他地方用于确定是否要拍照。如果检测到入侵者，我们还会在这个方法中发出警报。

通过添加新方法，我们更改了`getSecurityImage`。通过在`getSecurityImage`方法中使用`alarm_status`，我们无需检查传感器的状态。现在我们可以将`getSecurityImage`用于其预期用途——在检测到入侵者时拍照。

现在是时候更改仪表板代码了：

1.  从应用程序菜单|编程|Thonny Python IDE 打开 Thonny

1.  单击“新建”以创建新文件

1.  在文件中输入以下内容：

```py
import cherrypy
from SecurityDataQuick import SecurityData

class SecurityDashboard:

def __init__(self, securityData):
    self.securityData = securityData

@cherrypy.expose
def index(self):
    return """
        <!DOCTYPE html>
        <html lang="en">

        <head>
            <title>Home Security Dashboard</title>
            <meta charset="utf-8">

            <meta name="viewport" content="width=device-
        width, initial-scale=1">

            <meta http-equiv="refresh" content="2">

            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com
        /bootstrap/4.1.0/css/bootstrap.min.css">

            <link rel="stylesheet" href="led.css">

            <script src="https://ajax.googleapis.com
        /ajax/libs/jquery/3.3.1/jquery.min.js">
            </script>

            <script src="https://cdnjs.cloudflare.com
        /ajax/libs/popper.js/1.14.0
        /umd/popper.min.js">
            </script>

            <script src="https://maxcdn.bootstrapcdn.com
        /bootstrap/4.1.0/js/bootstrap.min.js">
            </script>

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
                <br />
                <div class="card">
                    <div class="card-header">
                        <h3>Home Security Dashboard</h3>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col element-box">
                                <h4>Armed</h4>

                                <div class=""" + self
        .securityData
        .getArmedStatus() 
        + """>
                                </div>
                            </div>

                            <div class="col element-box">
                                <h4>Status</h4>
                                <p>""" + self.securityData
                                    .getAlarmStatus()
                                    + """</p>
                            </div>

                            <div class="col element-box">
                                <h4>Last Check:</h4>

                                <p>""" + self.securityData
                                    .getTime() + """
                                </p>
                            </div>
                        </div>
                    </div>
                    <div class="card-footer" align="center">
                        <img src=""" + self.securityData
        .getSecurityImage() + """ />
                        <p>""" + self.securityData
                            .getDetectedMessage() + """</p>
                    </div>
                </div>
            </div>
        </body>

        </html>
    """

if __name__=="__main__":
    securityData = SecurityData()
    conf = {
        '/led.css':{
        'tools.staticfile.on': True,
        'tools.staticfile.filename': '/home/pi/styles/led.css'
        },
        '/intruder.png':{
        'tools.staticfile.on': True,
        'tools.staticfile.filename': '/home/pi
        /images/intruder.png'
        },
        '/all-clear.png':{
        'tools.staticfile.on': True,
        'tools.staticfile.filename': '/home/pi
        /images/all-clear.png'
        },
        '/not-armed.png':{
        'tools.staticfile.on': True,
        'tools.staticfile.filename': '/home/pi
        /images/not-armed.png'
        }
    }
    cherrypy.quickstart(SecurityDashboard(securityData),'/',conf)

```

1.  将文件保存为`SecurityDataQuick.py`

1.  运行代码

1.  返回到您的网络浏览器并刷新仪表板页面

我们的仪表板现在应该与以下截图匹配：

![](img/57ba904c-a2b4-411b-bff3-8947fa32c3a3.png)

![](img/3990c950-4f8d-47fd-a2e8-f7c7526f3164.png)

![](img/01356c4e-7034-4ecf-8ad8-52a1258fc0dc.png)

我们的仪表板应该每两秒刷新一次，而不是 30 秒，当处于武装模式时检测到运动时应该发出蜂鸣器声音。

让我们看看代码。我们仪表板的更改相当容易理解。但值得注意的是我们仪表板上中间框的更改：

```py
<div class="col element-box">
    <h4>Status</h4>
    <p>""" + self.securityData.getAlarmStatus() + """</p>
</div>
```

我们通过`getAlarmStatus`方法将房间的`温度`和`湿度`替换为开关和 PIR 传感器的状态。通过这种更改，我们可以使用`getAlarmStatus`方法作为我们的`初始化`方法，其中我们设置`SecurityData`类变量`alarm_status`的状态。

如果我们真的想要一丝不苟，我们可以更改我们的代码，以便使用开关和 PIR 传感器的值来初始化`SecurityData`类。目前，`SecurityData`更像是一种实用类，其中必须先调用某些方法。我们暂且放过它。

# 摘要

正如我们所看到的，使用树莓派构建安全应用程序非常容易。尽管我们正在查看我们的仪表板并在同一台树莓派上托管我们的传感器，但将树莓派设置为向网络中的其他计算机（甚至是互联网）提供仪表板并不太困难。在第十章中，*发布到 Web 服务*，我们将与

将传感器数据进一步处理并发布到互联网。

# 问题

1.  真或假？DHT11 传感器是一种昂贵且高精度的温湿度传感器。

1.  真或假？DHT11 传感器可以检测到来自太阳的紫外线。

1.  真或假？运行 DHT11 所需的代码已预装在 Raspbian 中。

1.  如何设置 Pi 摄像头模块的分辨率？

1.  如何设置 CherryPy 以便可以访问本地静态文件？

1.  如何设置网页的自动刷新？

1.  真或假？通过使用 CSS，我们可以模拟闪烁的 LED。

1.  “SecurityData”类的目的是什么？

1.  我们找到了谁或什么作为我们的入侵者？

1.  如果我们想要一丝不苟，我们将如何更改我们的`SecurityData`类？

# 进一步阅读

我们代码中使用的刷新方法很有效，但有点笨拙。我们的仪表板可以通过使用 AJAX 代码进行改进，其中字段被更新但页面不更新。请查阅 CherryPy 文档以获取更多信息。
