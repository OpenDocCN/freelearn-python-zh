# 使用伺服控制代码控制模拟设备

继续我们的旅程，将模拟仪表的优雅与数字数据的准确性相结合，我们将看看我们在前两章中学到的内容，并构建一个带有模拟仪表显示的物联网天气仪表盘。

在开始本章之前，请确保已经连接了[第5章](a180e8ce-8d3b-4158-b260-981ee3697af4.xhtml)中的电路，*使用Python控制伺服*。

这个仪表盘将根据室外温度和风速显示衣柜建议。我们还将在我们的仪表盘上使用LED指示是否需要带上雨伞。

本章将涵盖以下主题：

+   从云端访问天气数据

+   使用天气数据控制伺服

+   增强我们的项目

# 完成本章所需的知识

您应该具备Python编程语言的工作知识才能完成本章。还必须了解如何使用简单的面包板，以便连接组件。

在这个项目中可以使用乙烯基或手工切割机。了解如何使用切割机将是一个资产，这样你就可以完成这个项目。

# 项目概述

到本章结束时，我们应该有一个可以工作的物联网模拟天气仪表盘。我们将修改[第4章](626664bb-0130-46d1-b431-682994472fc1.xhtml)和[第5章](a180e8ce-8d3b-4158-b260-981ee3697af4.xhtml)中编写的代码，以向我们的仪表盘提供数据。将打印并剪切一个背景。这个背景将给我们的仪表盘一个卡通般的外观。

我们将使用[第5章](a180e8ce-8d3b-4158-b260-981ee3697af4.xhtml)中的电路，*使用Python控制伺服*。以下是来自该电路的接线图：

![](assets/80b31448-fb1a-4e05-8f81-ccecbeb974e0.png)

这个项目应该需要一个下午的时间来完成。

# 入门

要完成这个项目，需要以下设备：

+   树莓派3型号（2015年或更新型号）

+   一个USB电源适配器

+   一台电脑显示器

+   一个USB键盘

+   一个USB鼠标

+   一个小型伺服电机

+   一个LED（任何颜色）

+   一个面包板

+   面包板的跳线线

+   一个彩色打印机

+   一个乙烯基或手工切割机（可选）

# 从云端访问天气数据

在[第4章](626664bb-0130-46d1-b431-682994472fc1.xhtml)中，*订阅Web服务*，我们编写了一个Python程序，从Yahoo!天气获取天气数据。该程序中的`CurrentWeather`类返回了根据类实例化时的`city`值返回的温度、天气状况和风速。

我们将重新访问该代码，并将类名更改为`WeatherData`。我们还将添加一个方法，返回一个值从`0`-`100`，以指示天气。在确定这个数字时，我们将考虑温度和风速，0表示极端的冬季条件，`100`表示非常炎热的夏季极端条件。我们将使用这个数字来控制我们的伺服。我们还将检查是否下雨，并更新我们的LED以指示我们是否需要雨伞：

1.  从应用程序菜单 | 编程 | Thonny Python IDE 打开 Thonny

1.  单击新图标创建一个新文件

1.  在文件中输入以下内容：

```py
from weather import Weather, Unit

class WeatherData:

    temperature = 0
    weather_conditions = ''
    wind_speed = 0
    city = ''

    def __init__(self, city):
        self.city = city
        weather = Weather(unit = Unit.CELSIUS)
        lookup = weather.lookup_by_location(self.city)
        self.temperature = float(lookup.condition.temp)
        self.weather_conditions = lookup.condition.text
        self.wind_speed = float(lookup.wind.speed)

    def getServoValue(self):
        temp_factor = (self.temperature*100)/30
        wind_factor = (self.wind_speed*100)/20
        servo_value = temp_factor-(wind_factor/20)

        if(servo_value >= 100):
            return 100
        elif (servo_value <= 0):
            return 0
        else:
            return servo_value

    def getLEDValue(self): 
        if (self.weather_conditions=='Thunderstorm'):
            return 2;
        elif(self.weather_conditions=='Raining'):
            return 1
        else:
            return 0

if __name__=="__main__":

    weather = WeatherData('Paris')
    print(weather.getServoValue())
    print(weather.getLEDValue())
```

1.  将文件保存为`WeatherData.py`

我们的代码的核心在于`getServoValue()`和`getLEDValue()`方法：

```py
def getServoValue(self):
     temp_factor = (self.temperature*100)/30
     wind_factor = (self.wind_speed*100)/20
     servo_value = temp_factor-(wind_factor/20)

     if(servo_value >= 100):
         return 100
     elif (servo_value <= 0):
         return 0
     else:
         return servo_value
```

在`getServoValue`方法中，我们将`temp_factor`和`wind_factor`变量设置为基于最小值`0`和温度和风速的最大值`30`和`20`的百分比值。这些是任意的数字，因为我们将考虑`30`摄氏度为我们的极端高温，20公里/小时的风速为我们的极端风速。伺服值通过从温度减去风速的5%（除以`20`）来设置。当然，这也是任意的。随意调整所需的百分比。

为了进一步解释，考虑一下10摄氏度的温度和5公里/小时的风速。温度因子（temp_factor）将是10乘以100，然后除以30或33.33。风速因子（wind_factor）将是5乘以100，然后除以20或25。我们传递给舵机的值（servo_value）将是温度因子（33.33）减去风速因子（25）后除以`20`。`servo_value`的值为32.08，或者大约是最大舵机值的32%。

然后返回`servo_value`的值并将其用于控制我们的舵机。任何低于`0`和高于`100`的值都将超出我们的范围，并且无法与我们的舵机一起使用（因为我们将舵机在`0`和`100`之间移动）。我们在`getServoValue`方法中使用`if`语句来纠正这种情况。

`getLEDValue`方法只是检查天气条件并根据是否下雨返回代码。“雷暴”将返回值`2`，“雨”和“小雨”将返回值`1`，其他任何情况将返回值`0`。如果有雷暴，我们将使用这个值来在我们的仪表盘上闪烁LED，如果只是下雨，我们将保持其亮起，并在其他所有情况下关闭它：

```py
def getLEDValue(self):
     if (self.weather_conditions=='Thunderstorm'):
         return 2;
     elif(self.weather_conditions=='Rain'):
         return 1
     elif(self.weather_conditions=='Light Rain'):
         return 1
     else:
         return 0
```

在撰写本书时，“雷暴”、“雨”和“小雨”是在搜索世界各大城市天气时返回的值。请随时更新`if`语句以包括其他极端降水的描述。作为一个额外的增强，你可以考虑在`if`语句中使用正则表达式。

在Thonny中运行代码。你应该会得到巴黎天气条件下舵机和LED的值。我在运行代码时得到了以下结果：

```py
73.075
0
```

# 使用天气数据控制舵机

我们即将构建我们的物联网天气仪表盘。最后的步骤涉及根据从Yahoo! Weather网络服务返回的天气数据来控制我们舵机的位置，并在物理上建立一个背景板来支撑我们的舵机指针。

# 校正舵机范围

正如你们中的一些人可能已经注意到的那样，你的舵机并不能从最小到最大移动180度。这是由于GPIO Zero中设置的最小和最大脉冲宽度为1毫秒和2毫秒。为了解决这个差异，我们在实例化`Servo`对象时必须相应地调整`min_pulse_width`和`max_pulse_width`属性。

以下代码就是这样做的。变量`servoCorrection`对`min_pulse_width`和`max_pulse_width`值进行加减。以下代码在`5`秒后将舵机移动到最小位置，然后移动到最大位置：

1.  从应用程序菜单 | 编程 | Thonny Python IDE中打开Thonny。

1.  单击“新建”图标创建新文件。

1.  在文件中键入以下内容：

```py
from gpiozero import Servo
from time import sleep
servoPin=17

servoCorrection=0.5
maxPW=(2.0+servoCorrection)/1000
minPW=(1.0-servoCorrection)/1000

servo=Servo(servoPin, min_pulse_width=minPW, max_pulse_width=maxPW)

servo.min()
sleep(5)
servo.max()
sleep(5)
servo.min()
sleep(5)
servo.max()
sleep(5)
servo.min()
sleep(5)
servo.max()
sleep(5)

servo.close()
```

1.  将文件保存为`servo_correction.py`。

1.  运行代码，看看`servoCorrection`的值是否修复了你的舵机在`servo.min`到`servo.max`之间不能转动180度的问题。

1.  调整`servoCorrection`，直到你的舵机在`servo.min`和`servo.max`之间移动了180度。我们将在天气仪表盘的代码中使用`servoCorrection`的值。

# 根据天气数据改变舵机的位置

我们现在已经准备好根据天气条件控制我们舵机的位置。我们将修改我们在[第5章](a180e8ce-8d3b-4158-b260-981ee3697af4.xhtml)中创建的`WeatherDashboard`类，*用Python控制舵机*，执行以下步骤：

1.  从应用程序菜单 | 编程 | Thonny Python IDE中打开Thonny

1.  单击“新建”图标创建新文件

1.  在文件中键入以下内容：

```py
from gpiozero import Servo
from gpiozero import LED
from time import sleep
from WeatherData import WeatherData

class WeatherDashboard:

     servo_pin = 17
     led_pin = 14
     servoCorrection=0.5
     maxPW=(2.0+servoCorrection)/1000
     minPW=(1.0-servoCorrection)/1000

     def __init__(self, servo_position=0, led_status=0):
         self.servo = Servo(self.servo_pin, min_pulse_width=
                self.minPW, max_pulse_width=self.maxPW)
         self.led = LED(self.led_pin)

         self.move_servo(servo_position)
         self.set_led_status(led_status)

     def move_servo(self, servo_position=0): 
         self.servo.value = self.convert_percentage_to_integer(
                servo_position)

     def turnOffServo(self):
         sleep(5)
         self.servo.close()

     def set_led_status(self, led_status=0):
         if(led_status==0):
             self.led.off()
         elif (led_status==1):
             self.led.on()
         else:
             self.led.blink()

     def convert_percentage_to_integer(self, percentage_amount):
        #adjust for servos that turn counter clockwise by default
        adjusted_percentage_amount = 100 - percentage_amount
        return (adjusted_percentage_amount*0.02)-1

if __name__=="__main__":
     weather_data = WeatherData('Toronto')
     weather_dashboard = WeatherDashboard(
     weather_data.getServoValue(),
     weather_data.getLEDValue())
     weather_dashboard.turnOffServo()
```

1.  将文件保存为`WeatherDashboard.py`

1.  运行代码并观察舵机位置的变化

让我们来看看代码。

我们首先导入我们需要的资源：

```py
from time import sleep
from WeatherData import WeatherData
```

我们添加`time`到我们的项目中，因为我们将在关闭`Servo`对象之前使用它作为延迟。添加`WeatherData`以根据天气条件为我们的伺服和LED提供值。

`servoCorrection`，`maxPW`和`minPW`变量调整我们的伺服（如果需要），如前面的伺服校正代码所述：

```py
servoCorrection=0.5
maxPW=(2.0+servoCorrection)/1000
minPW=(1.0-servoCorrection)/1000
```

`turnOffServo`方法允许我们关闭与伺服的连接，停止可能发生的任何抖动运动：

```py
def turnOffServo(self):
    sleep(5)
    self.servo.close()
```

我们使用`sleep`函数延迟关闭伺服，以便在设置到位置之前不会关闭。

您可能还注意到了代码中`convert_percentage_to_integer`方法的更改[第5章](eff0f7cb-f99b-45d5-8781-42c841bd2fd9.xhtml)中的代码，*使用Python控制伺服*。为此项目测试的电机在右侧有一个最小位置。这与我们所需的相反，因此代码已更改为从100中减去`percentage_amount`，以扭转此行为并给出正确的伺服位置（有关此方法的更多信息，请参阅[第5章](a180e8ce-8d3b-4158-b260-981ee3697af4.xhtml)，*使用Python控制伺服*，如有需要，请使用本章中的`convert_percentage_to_integer`）：

```py
def convert_percentage_to_integer(self, percentage_amount):
        #adjust for servos that turn counter clockwise by default
        adjusted_percentage_amount = 100 - percentage_amount
        return (adjusted_percentage_amount*0.02)-1
```

在Thonny中运行代码。您应该看到伺服电机根据多伦多，加拿大的天气条件移动到一个位置。LED将根据多伦多的降雨情况闪烁、保持稳定或关闭。

现在，让我们通过为我们的伺服和LED建立一个物理背景来增强我们的项目。

# 增强我们的项目

现在我们的代码已经完成，现在是时候为我们的伺服添加一个物理背景了。通过这个背景，我们根据天气数据为我们的衣柜推荐穿什么。

# 打印主图形

以下是我们将在背景中使用的图形：

![](assets/6c3d81ed-cbcb-4f52-9d72-627d49d3fbfa.png)

使用彩色打印机，在可打印的乙烯基上打印图形（此图像文件可从我们的GitHub存储库中获取）。剪下伞下和主图形下的孔。

为了增加支撑，用刀或剪刀在硬纸板上切出背板：

![](assets/b712b994-07a0-423c-94ae-6e8940f87e8d.png)

将背景从可打印的乙烯基片上剥离并粘贴到背板上。使用孔将背景与背板对齐：

![](assets/caaa3b46-e584-45d2-a087-ee980b7fb01e.png)

# 添加指针和LED

将LED插入伞下的孔中：

![](assets/1e7514b8-e796-4bd3-929b-a49cdc0e52dc.png)

将伺服电机的轴心插入另一个孔。如有必要，使用双面泡沫胶带将伺服固定在背板上：

![](assets/1b142b34-2431-4bdb-89b0-8cbf65488fee.png)

使用跳线线将LED和伺服连接到面包板上（请参阅本章开头的接线图）。组件应该稍微倾斜。在我们用新的显示运行`WeatherDashboard`代码之前，我们必须将指针安装到最小位置：

1.  从应用程序菜单中打开Thonny | 编程 | Thonny Python IDE

1.  单击新图标创建一个新文件

1.  在文件中输入以下内容：

```py
from gpiozero import Servo
servoPin=17

servoCorrection=<<put in the correction you calculated>>
maxPW=(2.0+servoCorrection)/1000
minPW=(1.0-servoCorrection)/1000

servo=Servo(servoPin, min_pulse_width=minPW, max_pulse_width=maxPW)

servo.min()
```

1.  将文件保存为`servo_minimum.py`

1.  运行代码使伺服将自己定位到最小值

安装指针，使其指向左侧，如果伺服电机逆时针转到最小位置，使其指向右侧，如果伺服电机顺时针转到最小位置（一旦您开始实际使用伺服，这将更有意义）。

再次运行`WeatherDashboard`代码。伺服应该根据天气数据移动，指示衣柜选项。如果下雨，LED应该亮起。雷暴会闪烁LED。否则，LED将保持关闭状态。

在下图中，仪表盘建议多伦多，加拿大穿短袖衬衫。外部天气条件不需要雨伞：

![](assets/f64e1685-26b7-4020-9ea4-91021073ff0e.png)

恭喜！你刚刚建立了一个IoT天气仪表盘。

# 摘要

在这个项目中，我们利用了树莓派的力量来创建了一个IoT模拟天气仪表盘。在这种情况下，这涉及到使用作为模拟仪表的互联网控制的伺服。我们很容易想象如何修改我们的代码来显示除天气数据之外的其他数据。想象一下，一个模拟仪表显示远程工厂的油箱水平，其中水平数据通过互联网通信。

模拟仪表的直观性使其非常适合需要一瞥数据的应用程序。将模拟仪表与来自互联网的数据结合起来，创造了全新的数据显示世界。

在[第7章](4c4cf44d-ff8a-4cb4-9d8c-85530b0d873b.xhtml)中，*设置树莓派Web服务器*，我们将迈出模拟世界的一步，探索如何使用树莓派作为Web服务器并构建基于Web的仪表盘。

# 问题

1.  真还是假？伺服可以用作IoT设备。

1.  真还是假？更改`Servo`对象上的最小和最大脉冲宽度值会修改伺服的范围。

1.  为什么在调用`Servo`对象的`close()`方法之前我们要添加延迟？

1.  真还是假？我们在`WeatherData`类中不需要`getTemperature()`方法。

1.  真还是假？我们仪表盘上闪烁的LED表示晴天和多云的天气。

1.  我们在仪表盘上使用一对短裤来表示什么？

1.  在我们的代码中，你会在哪里使用正则表达式？

1.  为什么我们在代码中导入时间？

1.  真还是假？IoT启用的伺服只能用于指示天气数据。

# 进一步阅读

为了增强我们的代码，可以使用正则表达式。任何关于Python和正则表达式的文档都对发展强大的编码技能非常宝贵。
