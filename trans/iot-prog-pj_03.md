# 使用GPIO连接到外部世界

在本章中，我们将开始解锁树莓派背后真正的力量——GPIO，或通用输入输出。 GPIO允许您通过可以设置为输入或输出的引脚将树莓派连接到外部世界，并通过代码进行控制。

本章将涵盖以下主题：

+   树莓派的Python库

+   访问树莓派的GPIO

+   设置电路

+   你好LED

# 项目概述

在本章中，我们首先探索了Python的树莓派特定库。我们将使用树莓派相机模块和Pibrella HAT的几个示例来演示这些内容。在转到使用Fritzing程序设计物理电路之前，我们将尝试使用Sense Hat模拟器进行一些编码示例。使用面包板，我们将设置这个电路并将其连接到我们的树莓派。

我们将通过在[第2章](f310884c-55e6-48a4-8329-05ff04e48e19.xhtml)中创建的类中构建一个摩尔斯电码生成器，该生成器将以摩尔斯电码传输天气数据来结束本章，*使用树莓派编写Python程序*。完成本章应该需要一个下午的时间。

# 技术要求

完成此项目需要以下内容：

+   树莓派3型号（2015年或更新型号）

+   USB电源适配器

+   计算机显示器

+   USB键盘

+   USB鼠标

+   树莓派相机模块（可选）—[https://www.raspberrypi.org/products/camera-module-v2/](https://www.raspberrypi.org/products/camera-module-v2/)

+   Pribrella HAT（可选）—[www.pibrella.com](http://www.pibrella.com)

+   Sense HAT（可选，因为我们将在本章中使用模拟器）—[https://www.raspberrypi.org/products/sense-hat/a](https://www.raspberrypi.org/products/sense-hat/)

+   面包板

+   母对公跳线

+   LED

# 树莓派的Python库

我们将把注意力转向Raspbian预装的Python库或包。要从Thonny查看这些包，请单击工具|管理包。稍等片刻后，您应该会在对话框中看到许多列出的包：

![](assets/f8e2919c-6c5f-4f6b-b422-44c9bad7d1ed.png)

让我们来探索其中一些包。

# picamera

树莓派上的相机端口或CSI允许您将专门设计的树莓派相机模块连接到您的Pi。该相机可以拍摄照片和视频，并具有进行延时摄影和慢动作视频录制的功能。`picamera`包通过Python使我们可以访问相机。以下是连接到树莓派3 Model B的树莓派相机模块的图片：

![](assets/60d9b712-4c92-452c-9930-4b35240bbf9f.png)

将树莓派相机模块连接到您的Pi，打开Thonny，并输入以下代码：

```py
import picamera
import time

picam = picamera.PiCamera()
picam.start_preview()
time.sleep(10)
picam.stop_preview()
picam.close()
```

此代码导入了`picamera`和`time`包，然后创建了一个名为`picam`的`picamera`对象。从那里，我们开始预览，然后睡眠`10`秒，然后停止预览并关闭相机。运行程序后，您应该在屏幕上看到来自相机的`10`秒预览。

# 枕头

Pillow包用于Python图像处理。要测试这一点，请将图像下载到与项目文件相同的目录中。在Thonny中创建一个新文件，然后输入以下内容：

```py
from PIL import Image

img = Image.open('image.png')
print(img.format, img.size)
```

您应该在随后的命令行中看到图像的格式和大小（括号内）打印出来。

# sense-hat和sense-emu

Sense HAT是树莓派的一个复杂的附加板。Sense HAT是Astro Pi套件的主要组件，是一个让年轻学生为国际空间站编程树莓派的计划的一部分。

Astro Pi比赛于2015年1月正式向英国所有小学和中学年龄的孩子开放。在对国际空间站的任务中，英国宇航员蒂姆·皮克在航天站上部署了Astro Pi计算机。

获胜的Astro Pi比赛代码被加载到太空中的Astro Pi上。生成的数据被收集并发送回地球。

Sense HAT包含一组LED，可用作显示器。Sense HAT还具有以下传感器：

+   加速度计

+   温度传感器

+   磁力计

+   气压传感器

+   湿度传感器

+   陀螺仪

我们可以通过`sense-hat`包访问Sense HAT上的传感器和LED。对于那些没有Sense HAT的人，可以使用Raspbian中的Sense HAT模拟器。我们使用`sense-emu`包来访问Sense HAT模拟器上模拟的传感器和LED显示。

为了演示这一点，请执行以下步骤：

1.  在Thonny中创建一个新文件，并将其命名为`sense-hat-test.py`，或类似的名称。

1.  键入以下代码：

```py
from sense_emu import SenseHat

sense_emulator = SenseHat()
sense_emulator.show_message('Hello World')
```

1.  从应用程序菜单|编程|Sense HAT模拟器加载Sense HAT模拟器程序。

1.  调整屏幕，以便您可以看到Sense HAT模拟器的LED显示和Thonny的完整窗口（请参见下一张截图）：

![](assets/366ddf47-8fe3-40c4-ae0a-3eadd23b99ff.png)

1.  单击**运行当前脚本**按钮。

1.  你应该看到“Hello World！”消息一次一个字母地滚动在Sense HAT模拟器的LED显示器上（请参见上一张截图）。

# 访问树莓派的GPIO

通过GPIO，我们能够连接到外部世界。以下是树莓派GPIO引脚的图示：

![](assets/79dbd754-d3e8-462d-83c5-4eba89aed7ac.jpg)

以下是这些引脚的解释：

+   红色引脚代表GPIO输出的电源。GPIO提供3.3伏特和5伏特。

+   黑色引脚代表用于电气接地的引脚。正如您所看到的，GPIO上有8个接地引脚。

+   蓝色引脚用于树莓派的**硬件附加在顶部**（**HATs**）。它们允许树莓派和HAT的**电可擦可编程只读存储器**（**EEPROM**）之间的通信。

+   绿色引脚代表我们可以为其编程的输入和输出引脚。请注意，一些绿色GPIO引脚具有额外的功能。我们将不会涵盖这个项目的额外功能。

GPIO是树莓派的核心。我们可以通过GPIO将LED、按钮、蜂鸣器等连接到树莓派上。我们还可以通过为树莓派设计的HAT来访问GPIO。其中之一叫做“Pibrella”，这是我们接下来将使用的，用来通过Python代码探索连接到GPIO。

树莓派1型A和B型只有前26个引脚（如虚线所示）。从那时起的型号，包括树莓派1型A+和B+，树莓派2，树莓派Zero和Zero W，以及树莓派3型B和B+，都有40个GPIO引脚。

# Pibrella

Pibrella是一个相对便宜的树莓派HAT，可以轻松连接到GPIO。以下是Pibrella板上的组件：

+   1个红色LED

+   1个黄色LED

+   1个绿色LED

+   小音箱

+   按键

+   4个输入

+   4个输出

+   Micro USB电源连接器，用于向输出提供更多电源

Pibrella是为早期的树莓派型号设计的，因此只有26个引脚输入。但是，它可以通过前26个引脚连接到后来的型号。

要安装Pibrella Hat，将Pibrella上的引脚连接器与树莓派上的前26个引脚对齐，并向下按。在下图中，我们正在将Pibrella安装在树莓派3型B上：

![](assets/e0cdda19-675f-4dd9-8be8-92b39e77bb6b.png)

安装Pibrella时应该很合适：

![](assets/4db07f13-e208-4307-9997-1d85ee2adc90.png)

连接到Pibrella所需的库在Raspbian中没有预先安装（截至撰写本文的时间），因此我们必须自己安装它们。为此，我们将使用终端中的`pip3`命令：

1.  通过单击顶部工具栏上的终端（从左起的第四个图标）加载终端。在命令提示符下，键入以下内容：

```py
sudo pip3 install pibrella
```

1.  您应该看到终端加载软件包：

![](assets/98d08ff7-12f1-4e83-8cad-e317d2db3e8d.png)

1.  使用`Pibrella`库，无需知道GPIO引脚编号即可访问GPIO。该功能被包装在我们导入到代码中的`Pibrella`对象中。我们将进行一个简短的演示。

1.  在Thonny中创建一个名为`pibrella-test.py`的新文件，或者取一个类似的名字。键入以下代码：

```py
import pibrella
import time

pibrella.light.red.on()
time.sleep(5)
pibrella.light.red.off()
pibrella.buzzer.success()
```

1.  点击运行当前脚本按钮运行代码。如果您输入的一切都正确，您应该看到Pibrella板上的红灯在`5`秒钟内亮起，然后扬声器发出短暂的旋律。

恭喜，您现在已经跨越了物理计算的门槛。

# RPi.GPIO

用于访问GPIO的标准Python包称为`RPi.GPIO`。描述它的最佳方式是使用一些代码（这仅用于演示目的；我们将在接下来的部分中运行代码来访问GPIO）：

```py
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
GPIO.output(18, GPIO.HIGH)
time.sleep(5)
GPIO.output(18, GPIO.LOW)
```

正如您所看到的，这段代码似乎有点混乱。我们将逐步介绍它：

1.  首先，我们导入`RPi.GPIO`和`time`库：

```py
import RPi.GPIO as GPIO
import time
```

1.  然后，我们将模式设置为`BCM`：

```py
GPIO.setmode(GPIO.BCM)
```

1.  在BCM模式下，我们通过GPIO编号（显示在我们的树莓派GPIO图形中的编号）访问引脚。另一种方法是通过它们的物理位置（`GPIO.BOARD`）访问引脚。

1.  要将GPIO引脚`18`设置为输出，我们使用以下行：

```py
GPIO.setup(18, GPIO.OUT)
```

1.  然后我们将GPIO `18`设置为`HIGH`，持续`5`秒，然后将其设置为`LOW`：

```py
GPIO.output(18, GPIO.HIGH)
time.sleep(5)
GPIO.output(18, GPIO.LOW)
```

如果我们设置了电路并运行了代码，我们会看到LED在`5`秒钟内亮起，然后关闭，类似于Pibrella示例。

# GPIO零

`RPi.GPIO`的替代方案是GPIO Zero包。与`RPi.GPIO`一样，这个包已经预装在Raspbian中。名称中的零指的是零样板或设置代码（我们被迫每次输入的代码）。

为了完成打开和关闭LED灯5秒钟的相同任务，我们使用以下代码：

```py
from gipozero import LED
import time

led = LED(18)
led.on()
time.sleep(5)
led.off()
```

与我们的`RPi.GPIO`示例一样，这段代码仅用于演示目的，因为我们还没有设置电路。很明显，GPIO Zero代码比`RPi.GPIO`示例简单得多。这段代码非常容易理解。

在接下来的几节中，我们将在面包板上构建一个物理电路，其中包括LED，并使用我们的代码来打开和关闭它。

# 设置电路

Pibrella HAT为我们提供了一种简单的编程GPIO的方法，然而，树莓派项目的最终目标是创建一个定制的工作电路。我们现在将采取步骤设计我们的电路，然后使用面包板创建电路。

第一步是在计算机上设计我们的电路。

# Fritzing

Fritzing是一款免费的电路设计软件，适用于Windows、macOS和Linux。树莓派商店中有一个版本，我们将在树莓派上安装它：

1.  从应用菜单中，选择首选项|添加/删除软件。在搜索框中，键入`Fritzing`：

![](assets/732813b5-525b-4bad-aac3-de11bd72da0a.png)

1.  选择所有三个框，然后单击应用，然后单击确定。安装后，您应该能够从应用菜单|编程|Fritzing中加载Fritzing。

1.  点击面包板选项卡以访问面包板设计屏幕。一个全尺寸的面包板占据了屏幕的中间。我们将它缩小，因为我们的电路很小而简单。

1.  点击面包板。在检查器框中，您会看到一个名为属性的标题。

1.  点击大小下拉菜单，选择Mini。

1.  要将树莓派添加到我们的电路中，在搜索框中键入`Raspberry Pi`。将树莓派3拖到我们的面包板下方。

1.  从这里，我们可以将组件拖放到面包板上。

1.  将LED和330欧姆电阻器添加到我们的面包板上，如下图所示。我们使用电阻器来保护LED和树莓派免受可能造成损坏的过大电流：

![](assets/d26a7e02-67c4-48f9-9591-588675500457.png)

1.  当我们将鼠标悬停在树莓派组件的每个引脚上时，会弹出一个黄色提示，显示引脚的BCM名称。点击GPIO 18，将线拖到LED的正极（较长的引脚）。

1.  同样，将GND连接拖到电阻的左侧。

这是我们将为树莓派构建的电路。

# 构建我们的电路

要构建我们的物理电路，首先要将组件插入我们的面包板。参考之前的图表，我们可以看到一些孔是绿色的。这表示电路中有连续性。例如，我们通过同一垂直列将LED的负极连接到330欧姆电阻。因此，两个组件的引脚通过面包板连接在一起。

在我们开始在面包板上放置组件时，我们要考虑这一点：

![](assets/f4d79016-46de-4fd7-9baa-94beaf95042d.png)

1.  将LED插入我们的面包板，如上图所示。我们遵循我们的Fritzing图表，并将正极插入下方的孔中。

1.  按照我们的Fritzing图表，连接330欧姆电阻。使用母对公跳线，将树莓派连接到面包板上。

1.  参考我们的树莓派GPIO图表，在树莓派主板上找到GPIO 18和GND。

在连接跳线到GPIO时，最好将树莓派断电。

如下图所示，完整的电路类似于我们的Fritzing图表（只是我们的面包板和树莓派被转向）：

![](assets/df8f25c1-4ead-4c0a-a641-2146ccb891ab.png)

1.  将树莓派重新连接到显示器、电源、键盘和鼠标。

我们现在准备好编程我们的第一个真正的GPIO电路。

# Hello LED

我们将直接进入代码：

1.  在Thonny中创建一个新文件，并将其命名为`Hello LED.py`或类似的名称。

1.  输入以下代码并运行：

```py
from gpiozero import LED

led = LED(18)
led.blink(1,1,10)
```

# 使用gpiozero闪烁LED

如果我们正确连接了电路并输入了正确的代码，我们应该看到LED以1秒的间隔闪烁10秒。`gpiozero LED`对象中的`blink`函数允许我们设置`on_time`（LED保持打开的时间长度，以秒为单位）、`off_time`（LED关闭的时间长度，以秒为单位）、`n`或LED闪烁的次数，以及`background`（设置为`True`以允许LED闪烁时运行其他代码）。

带有默认参数的`blink`函数调用如下：

```py
blink(on_time=1, off_time=1, n=none, background=True)
```

在函数中不传递参数时，LED将以1秒的间隔不停地闪烁。请注意，我们不需要像使用`RPi.GPIO`包访问GPIO时那样导入`time`库。我们只需将一个数字传递给`blink`函数，表示我们希望LED打开或关闭的时间（以秒为单位）。

# 摩尔斯码天气数据

在[第2章](f310884c-55e6-48a4-8329-05ff04e48e19.xhtml)中，*使用树莓派编写Python程序*，我们编写了模拟调用提供天气信息的网络服务的代码。根据本章学到的知识，让我们重新审视该代码，并对其进行物理计算升级。我们将使用LED来闪烁表示我们的天气数据的摩尔斯码。

我们中的许多人认为，世界直到1990年代才开始通过万维网变得连接起来。我们很少意识到，19世纪引入电报和跨世界电报电缆时，我们已经有了这样一个世界。这个所谓的维多利亚时代互联网的语言是摩尔斯码，摩尔斯码操作员是它的门卫。

以下是闪烁摩尔斯码表示我们的天气数据的步骤：

1.  我们首先将创建一个`MorseCodeGenerator`类：

```py
from gpiozero import LED
from time import sleep

class MorseCodeGenerator:

    led = LED(18)
    dot_duration = 0.3
    dash_duration = dot_duration * 3
    word_spacing_duration = dot_duration * 7

    MORSE_CODE = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 
        'D': '-..', 'E': '.', 'F': '..-.',
        'G': '--.', 'H': '....', 'I': '..',
        'J': '.---', 'K': '-.-', 'L': '.-..',
        'M': '--', 'N': '-.', 'O': '---',
        'P': '.--.', 'Q': '--.-', 'R': '.-.',
       'S': '...', 'T': '-', 'U': '..-',
        'V': '...-', 'W': '.--', 'X': '-..-',
        'Y': '-.--', 'Z': '--..', '0': '-----',
        '1': '.----', '2': '..---', '3': '...--',
        '4': '....-', '5': '.....', '6': '-....',
        '7': '--...', '8': '---..', '9': '----.',
        ' ': ' '
        } 

    def transmit_message(self, message):
        for letter in message: 
            morse_code_letter = self.MORSE_CODE[letter.upper()]

            for dash_dot in morse_code_letter:

                if dash_dot == '.':
                    self.dot()

                elif dash_dot == '-':
                    self.dash()

                elif dash_dot == ' ':
                    self.word_spacing()

            self.letter_spacing()

    def dot(self):
        self.led.blink(self.dot_duration,self.dot_duration,1,False)

    def dash(self):
        self.led.blink(self.dash_duration,self.dot_duration,1,False)

    def letter_spacing(self):
        sleep(self.dot_duration)

    def word_spacing(self):
        sleep(self.word_spacing_duration-self.dot_duration)

if __name__ == "__main__":

    morse_code_generator = MorseCodeGenerator()
    morse_code_generator.transmit_message('SOS')    
```

1.  在我们的`MorseCodeGenerator`类中导入`gpiozero`和`time`库后，我们将GPIO 18定义为我们的LED，代码为`led=LED(18)`

1.  我们使用`dot_duration = 0.3`来设置`dot`持续的时间。

1.  然后我们根据`dot_duration`定义破折号的持续时间和单词之间的间距。

1.  为了加快或减慢我们的莫尔斯码转换，我们可以相应地调整`dot_duration`。

1.  我们使用一个名为`MORSE_CODE`的Python字典。我们使用这个字典将字母转换为莫尔斯码。

1.  我们的`transmit_message`函数逐个遍历消息中的每个字母，然后遍历莫尔斯码中的每个字符，这相当于使用`dash_dot`变量。

1.  我们的类的魔力在`dot`和`dash`方法中发生，它们使用了`gpiozero`库中的`blink`函数：

```py
def dot(self):
       self.led.blink(self.dot_duration, self.dot_duration,1,False)
```

在`dot`方法中，我们可以看到我们将LED打开的持续时间设置为`dot_duration`，然后我们将其关闭相同的时间。我们只闪烁一次，因为在`blink`方法调用中将其设置为数字`1`。我们还将背景参数设置为`False`。

这个最后的参数非常重要，因为如果我们将其保留为默认值`True`，那么LED在有机会闪烁之前，代码将继续运行。基本上，除非将背景参数设置为`False`，否则代码将无法工作。

在我们的测试消息中，我们放弃了通常的“Hello World”，而是使用了标准的“SOS”，这对于大多数莫尔斯码爱好者来说是熟悉的。我们可以通过单击“运行”按钮来测试我们的类，如果一切设置正确，我们将看到LED以莫尔斯码闪烁SOS。

现在，让我们重新审视一下[第2章](f310884c-55e6-48a4-8329-05ff04e48e19.xhtml)中的`CurrentWeather`类，即*使用树莓派编写Python程序*。我们将进行一些小的修改：

```py
from MorseCodeGenerator import MorseCodeGenerator

class CurrentWeather:

    weather_data={
        'Toronto':['13','partly sunny','8 NW'],
        'Montreal':['16','mostly sunny','22 W'],
        'Vancouver':['18','thunder showers','10 NE'],
        'New York':['17','mostly cloudy','5 SE'],
        'Los Angeles':['28','sunny','4 SW'],
        'London':['12','mostly cloudy','8 NW'],
        'Mumbai':['33','humid and foggy','2 S']
    }

    def __init__(self, city):
        self.city = city 

    def getTemperature(self):
        return self.weather_data[self.city][0]

    def getWeatherConditions(self):
        return self.weather_data[self.city][1]

    def getWindSpeed(self):
        return self.weather_data[self.city][2]

    def getCity(self):
        return self.city

if __name__ == "__main__":

    current_weather = CurrentWeather('Toronto')
    morse_code_generator = MorseCodeGenerator()
    morse_code_generator.transmit_message(current_weather.
    getWeatherConditions())

```

我们首先导入我们的`MorseCodeGenerator`类（确保两个文件在同一个目录中）。由于我们没有`/`的莫尔斯码等价物，我们从`weather_data`数据集中去掉了km/h。类的其余部分与[第2章](f310884c-55e6-48a4-8329-05ff04e48e19.xhtml)中的内容保持一致，即*使用树莓派编写Python程序*。在我们的测试部分，我们实例化了`CurrentWeather`类和`MorseCodeGenerator`类。使用`CurrentWeather`类，我们将多伦多的天气条件传递给`MorseCodeGenerator`类。

如果在输入代码时没有出现任何错误，我们应该能够看到LED以莫尔斯码闪烁“部分晴天”。

# 摘要

本章涵盖了很多内容。到最后，您应该对在树莓派上开发应用程序感到非常满意。

`picamera`，`Pillow`和`sense-hat`库使得使用树莓派与外部世界进行通信变得很容易。使用树莓派摄像头模块和`picamera`，我们为树莓派打开了全新的可能性。我们只是触及了`picamera`的一小部分功能。此外，我们只是浅尝了`Pillow`库的图像处理功能。Sense HAT模拟器使我们可以节省购买实际HAT的费用，并测试我们的代码。通过`sense-hat`和树莓派Sense HAT，我们真正扩展了我们在物理世界中的影响力。

廉价的Pibrella HAT提供了一个简单的方式来进入物理计算世界。通过安装`pibrella`库，我们让我们的Python代码可以访问一系列LED、扬声器和按钮，它们都被整齐地打包在一个树莓派HAT中。

然而，物理计算的真正终极目标是构建电子电路，以弥合我们的树莓派和外部世界之间的差距。我们开始使用树莓派商店提供的Fritzing电路构建器来构建电子电路。然后，我们在面包板上用LED和电阻器构建了我们的第一个电路。

我们通过使用树莓派和LED电路创建了一个莫尔斯码生成器来结束本章。在新旧结合的转折中，我们能够通过闪烁LED以莫尔斯码传输天气数据。

在[第4章]（626664bb-0130-46d1-b431-682994472fc1.xhtml）中，*订阅Web服务*，我们将把Web服务纳入我们的代码中，从而将互联网世界与现实世界连接起来，形成一个称为物联网的概念。

# 问题

1.  Python包的名称是什么，可以让您访问树莓派相机模块？

1.  真或假？由学生编写的树莓派已部署在国际空间站上。

1.  Sense HAT包含哪些传感器？

1.  真或假？我们不需要为开发购买树莓派Sense HAT，因为Raspbian中存在这个HAT的模拟器。

1.  GPIO上有多少个接地引脚？

1.  真或假？树莓派的GPIO引脚提供5V和3.3V。

1.  Pibrella是什么？

1.  真或假？只能在早期的树莓派计算机上使用Pibrella。

1.  BCM模式是什么意思？

1.  真或假？BOARD是BCM的替代品。

1.  `gpiozero`中的Zero指的是什么？

1.  真或假？使用Fritzing，我们可以为树莓派设计一个GPIO电路。

1.  `gpiozero` LED `blink`函数中的默认背景参数设置为什么？

1.  真或假？使用`gpiozero`库访问GPIO比使用`RPi.GPIO`库更容易。

1.  什么是维多利亚时代的互联网？

# 进一步阅读

本章涵盖了许多概念，假设所需的技能不超出普通开发人员和修补者的能力。为了进一步巩固对这些概念的理解，请谷歌以下内容：

+   如何安装树莓派相机模块

+   如何使用面包板？

+   Fritzing电路设计软件简介

+   Python字典

对于那些像我一样对过去的技术着迷的人，以下是一本关于维多利亚时代互联网的好书：*维多利亚时代的互联网*，作者汤姆·斯坦德奇。
