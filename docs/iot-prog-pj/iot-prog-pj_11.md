# 第十一章：使用蓝牙创建门铃按钮

在本章中，我们将把重点转向蓝牙。蓝牙是一种无线技术，用于在短距离内交换数据。它在 2.4 到 2.485 GHz 频段运行，通常的范围为 10 米。

在本章的项目中，我们将使用安卓上的蓝点应用程序，首先构建一个简单的蓝牙门铃，然后构建一个接受秘密滑动手势的更高级的门铃。

本章将涵盖以下主题：

+   介绍蓝点

+   RGB LED 是什么？

+   使用蓝牙和 Python 读取我们的按钮状态

# 项目概述

在本章中，我们将使用树莓派和安卓手机或平板电脑构建一个蓝牙门铃。我们将使用安卓手机或平板电脑上的一个名为蓝点的应用程序，该应用程序专为树莓派项目设计。

我们将从 RGB LED 开始，编写一个小程序来循环显示这三种颜色。然后，我们将使用 RGB LED 和有源蜂鸣器创建一个警报。我们将使用 Python 代码测试警报。

我们将编写 Python 代码来从蓝点读取按钮信息。然后，我们将结合警报和蓝点的代码来创建一个蓝牙门铃系统。

本章的项目应该需要一个上午或下午的时间来完成。

# 入门

完成此项目需要以下内容：

+   树莓派 3 型号（2015 年或更新型号）

+   USB 电源适配器

+   计算机显示器

+   USB 键盘

+   USB 鼠标

+   面包板

+   跳线线

+   330 欧姆电阻器（3 个）

+   RGB LED

+   有源蜂鸣器

+   安卓手机或平板电脑

# 介绍蓝点

蓝点是一个安卓应用程序，可在 Google Play 商店中获得。它可以作为树莓派的蓝牙遥控器。加载到您的安卓手机或平板电脑后，它基本上是一个大蓝点，您按下它就会向树莓派发送信号。以下是一个加载到平板电脑上的蓝点应用程序的图片：

![](img/cd63e306-d53d-4de9-b93e-d4dd48adb03e.png)

它可以作为一种蓝牙操纵杆，因为根据您如何与屏幕上的点交互，位置、滑块和旋转数据可以从应用程序发送到您的树莓派。我们将通过根据蓝点的按压方式创建自定义铃声，将一些功能添加到我们的门铃应用程序中。要在安卓手机或平板电脑上安装蓝点，请访问 Google Play 商店并搜索蓝点。

# 在树莓派上安装 bluedot 库

要在树莓派上安装`bluedot`库，请执行以下操作：

1.  打开终端应用程序

1.  在终端中输入以下内容：

```py
sudo pip3 install bluedot
```

1.  按*Enter*安装库

# 将蓝点与您的树莓派配对

为了使用蓝点应用程序，您必须将其与树莓派配对。要做到这一点，请按照以下步骤操作：

1.  从树莓派桌面客户端的右上角，点击蓝牙符号：

![](img/ea7c6ad0-ba64-4e00-a079-5f13c8a6e92f.png)

1.  如果蓝牙未打开，请点击蓝牙图标，然后选择打开蓝牙

1.  从蓝牙下拉菜单中选择“使可发现”

1.  在您的安卓手机或平板电脑上，转到蓝牙设置（这可能在手机或平板电脑上的特定操作系统上有不同的位置）

1.  您应该能够在“可用设备”列表中看到树莓派

1.  点击它以将您的设备与树莓派配对

1.  您应该在树莓派上收到一条消息，内容类似于“设备'Galaxy Tab E'请求配对。您接受请求吗？”

1.  点击“确定”接受

1.  可能会收到“连接失败”消息。我能够忽略这条消息，仍然可以让蓝点应用程序与我的树莓派配对，所以不要太担心

1.  将蓝点应用加载到您的安卓手机或平板电脑上

1.  您应该看到一个列表，其中树莓派是其中的一项

1.  点击树莓派项目以连接蓝点应用程序到树莓派

要测试我们的连接，请执行以下操作：

1.  通过以下方式打开 Thonny：应用程序菜单 | 编程 | Thonny Python IDE

1.  单击“新建”图标创建一个新文件

1.  在文件中键入以下内容：

```py
from bluedot import BlueDot
bd = BlueDot()
bd.wait_for_press()
print("Thank you for pressing the Blue Dot!")
```

1.  将文件保存为`bluest-test.py`并运行它

1.  您应该在 Thonny shell 中收到一条消息，上面写着`服务器已启动`，然后是树莓派的蓝牙地址

1.  然后您会收到一条消息，上面写着`等待连接`

1.  如果您的蓝点应用从树莓派断开连接，请通过在列表中选择树莓派项目来重新连接

1.  一旦蓝点应用连接到树莓派，您将收到消息`客户端已连接`，然后是您手机或平板电脑的蓝牙地址

1.  按下大蓝点

1.  Thonny shell 现在应该打印以下消息：`感谢您按下蓝点！`

# 接线我们的电路

我们将使用有源蜂鸣器和 RGB LED 创建一个门铃电路。由于我们之前没有讨论过 RGB LED，我们将快速看一下这个令人惊叹的小电子元件。然后，我们使用树莓派编写一个简单的测试程序，点亮 RGB LED 并发出有源蜂鸣器的声音。

# 什么是 RGB LED？

RGB LED 实际上只是一个单元内的三个 LED：一个红色，一个绿色，一个蓝色。通过在输入引脚的选择上以不同的功率电流来实现几乎可以达到任何颜色。以下是这样一个 LED 的图示：

![](img/bab49a3e-0a0e-40e3-88f5-1ac80a3ed730.png)

您可以看到有红色、绿色和蓝色引脚，还有一个负极引脚（-）。当 RGB LED 有一个负极引脚（-）时，它被称为共阴极。一些 RGB LED 有一个共阳极引脚（+），因此被称为共阳极。对于我们的电路，我们将使用一个共阴极的 RGB LED。共阴极和共阳极都有 RGB LED 的最长引脚，并且通过这个特征来识别。

# 测试我们的 RGB LED

我们现在将建立一个电路，用它我们可以测试我们的 RGB LED。以下是我们电路的接线图：

![](img/361e3f94-6df6-439c-8bd9-633b15fd5b39.png)

要按照图中所示的电路搭建，请执行以下操作：

1.  使用面包板，将 RGB LED 插入面包板，使得共阴极插入到左边第二个插槽中

1.  将 330 欧姆电阻器连接到面包板中央间隙上的红色、绿色和蓝色引脚

1.  从 GPIO 引脚 17 连接一根母对公跳线到面包板左侧的第一个插槽

1.  从 GPIO GND 连接一根母对公跳线到 RGB LED 的阴极引脚（从左边数第二个）

1.  从 GPIO 引脚 27 连接一根母对公跳线到面包板左侧的第三个插槽

1.  从 GPIO 引脚 22 连接一根母对公跳线到面包板左侧的第四个插槽

1.  从应用程序菜单 | 编程 | Thonny Python IDE 中打开 Thonny

1.  单击“新建”图标创建一个新文件

1.  在文件中键入以下内容：

```py
from gpiozero import RGBLED
from time import sleep

led = RGBLED(red=17, green=27, blue=22)

while True:
   led.color=(1,0,0)
    sleep(2)
    led.color=(0,1,0)
    sleep(2)
    led.color=(0,0,1)
    sleep(2)
    led.off()
    sleep(2)    
```

1.  将文件保存为`RGB-LED-test.py`并运行它

您应该看到 RGB LED 在红色亮起 2 秒钟。然后 RGB LED 应该在绿色亮起 2 秒钟，然后在蓝色亮起 2 秒钟。然后它将在 2 秒钟内关闭，然后再次开始序列。

在代码中，我们首先从 GPIO Zero 库导入`RGBLED`。然后，我们通过为 RGB LED 的红色、绿色和蓝色分配引脚号来设置一个名为`led`的变量。从那里，我们只需使用`led.color`属性打开每种颜色。很容易看出，将值`1, 0, 0`分配给`led.color`属性会打开红色 LED 并关闭绿色和蓝色 LED。`led.off`方法关闭 RGB LED。

尝试尝试不同的`led.color`值。您甚至可以输入小于`1`的值来改变颜色的强度（范围是`0`到`1`之间的任何值）。如果您仔细观察，您可能能够看到 RGB LED 内部不同的 LED 灯亮起。

# 完成我们的门铃电路

现在让我们向我们的电路中添加一个有源蜂鸣器，以完成我们门铃系统的构建。以下是我们门铃电路的图表：

![](img/f9a0218c-a5dd-465a-8d20-96247a7807d6.png)

要构建电路，请按照以下步骤进行：

1.  使用我们现有的电路，在面包板的另一端插入一个有源蜂鸣器

1.  将母对公跳线从 GPIO 引脚 26 连接到有源蜂鸣器的正引脚

1.  将母对公跳线从 GPIO GND 连接到有源蜂鸣器的负引脚

1.  从应用程序菜单中打开 Thonny |编程| Thonny Python IDE

1.  单击新图标创建新文件

1.  在文件中键入以下内容：

```py
from gpiozero import RGBLED
from gpiozero import Buzzer
from time import sleep

class DoorbellAlarm:

    led = RGBLED(red=17, green=22, blue=27)
    buzzer = Buzzer(26)
    num_of_times = 0

    def __init__(self, num_of_times):
        self.num_of_times = num_of_times

    def play_sequence(self):
        num = 0
        while num < self.num_of_times:
            self.buzzer.on()
            self.light_show()
            sleep(0.5)
            self.buzzer.off()
            sleep(0.5)
            num += 1

    def light_show(self):
        self.led.color=(1,0,0)
        sleep(0.1)
        self.led.color=(0,1,0)
        sleep(0.1)
        self.led.color=(0,0,1)
        sleep(0.1)
        self.led.off()

if __name__=="__main__":

    doorbell_alarm = DoorbellAlarm(5)
    doorbell_alarm.play_sequence()   
```

1.  将文件保存为`DoorbellAlarm.py`并运行它

1.  您应该听到蜂鸣器响了五次，并且 RGB LED 也应该按相同的次数进行灯光序列

让我们来看看代码：

1.  我们首先通过导入所需的库来开始：

```py
from gpiozero import RGBLED
from gpiozero import Buzzer
from time import sleep
```

1.  之后，我们使用`DoorbellAlarm`类名创建我们的类，然后设置初始值：

```py
led = RGBLED(red=17, green=22, blue=27)
buzzer = Buzzer(26)
num_of_times = 0
```

1.  类初始化使用`num_of_times`类变量设置警报序列将播放的次数：

```py
def __init__(self, num_of_times):
    self.num_of_times = num_of_times
```

1.  `light_show`方法只是按顺序闪烁 RGB LED 中的每种颜色，持续`0.1`秒：

```py
def light_show(self):
    self.led.color=(1,0,0)
    sleep(0.1)
    self.led.color=(0,1,0)
    sleep(0.1)
    self.led.color=(0,0,1)
    sleep(0.1)
    self.led.off()
```

1.  `play_sequence`方法打开和关闭蜂鸣器的次数设置在初始化`DoorbellAlarm`类时。每次蜂鸣器响起时，它还会运行 RGB LED `light_show`函数：

```py
def play_sequence(self):
    num = 0
    while num < self.num_of_times:
        self.buzzer.on()
        self.light_show()
        sleep(0.5)
        self.buzzer.off()
        sleep(0.5)
        num += 1
```

1.  我们通过用值`5`实例化`DoorbellAlarm`类并将其分配给`doorbell_alarm`变量来测试我们的代码。然后通过调用`play_sequence`方法来播放序列：

```py
if __name__=="__main__":

    doorbell_alarm = DoorbellAlarm(5)
    doorbell_alarm.play_sequence()   
```

# 使用蓝牙和 Python 读取我们的按钮状态

如前所述，我们能够以更多方式与 Blue Dot 应用进行交互，而不仅仅是简单的按钮按下。Blue Dot 应用可以解释用户在按钮上按下的位置，以及检测双击和滑动。在以下代码中，我们将使用 Python 从 Blue Dot 应用中读取。

# 使用 Python 读取按钮信息

做以下事情：

1.  从应用程序菜单中打开 Thonny |编程| Thonny Python IDE

1.  单击新图标创建新文件

1.  在文件中键入以下内容：

```py
from bluedot import BlueDot
from signal import pause

class BlueDotButton:

    def swiped(swipe):

        if swipe.up:
            print("Blue Dot Swiped Up")
        elif swipe.down:
            print("Blue Dot Swiped Down")
        elif swipe.left:
            print("Blue Dot Swiped Left")
        elif swipe.right:
            print("Blue Dot Swiped Right")

    def pressed(pos):
        if pos.top:
            print("Blue Dot Pressed from Top")
        elif pos.bottom:
            print("Blue Dot Pressed from Bottom")
        elif pos.left:
            print("Blue Dot Pressed from Left")
        elif pos.right:
            print("Blue Dot Pressed from Right")
        elif pos.middle:
            print("Blue Dot Pressed from Middle")

    def double_pressed():
        print("Blue Dot Double Pressed")

    blue_dot = BlueDot()
    blue_dot.when_swiped = swiped
    blue_dot.when_pressed = pressed
    blue_dot.when_double_pressed = double_pressed

 if __name__=="__main__":

    blue_dot_button = BlueDotButton()
    pause()       
```

1.  将文件保存为`BlueDotButton.py`并运行它

每次运行此程序时，您可能需要将 Blue Dot 应用连接到您的 Raspberry Pi（只需从 Blue Dot 应用中的列表中选择它）。尝试在中间，顶部，左侧等处按下 Blue Dot。您应该在 shell 中看到告诉您按下的位置的消息。现在尝试滑动和双击。shell 中的消息也应指示这些手势。

那么，我们在这里做了什么？让我们来看看代码：

1.  我们首先通过导入所需的库来开始：

```py
from bluedot import BlueDot
from signal import pause
```

我们显然需要`BlueDot`，我们还需要`pause`。我们使用`pause`来暂停程序，并等待来自 Blue Dot 应用的信号。由于我们正在使用`when_pressed`，`when_swiped`和`when_double_swiped`事件，我们需要暂停和等待（而不是其他方法，如`wait_for_press`）。我相信使用`when`而不是`wait`类型的事件使代码更清晰。

1.  在我们的程序的核心是实例化`BlueDot`对象及其相关的回调定义：

```py
blue_dot = BlueDot()
blue_dot.when_swiped = swiped
blue_dot.when_pressed = pressed
blue_dot.when_double_pressed = double_pressed
```

请注意，这些回调定义必须放在它们所引用的方法之后，否则将会出错。

1.  方法本身非常简单。以下是`swiped`方法：

```py
def swiped(swipe):

    if swipe.up:
        print("Blue Dot Swiped Up")
    elif swipe.down:
        print("Blue Dot Swiped Down")
    elif swipe.left:
        print("Blue Dot Swiped Left")
    elif swipe.right:
        print("Blue Dot Swiped Right")
```

1.  我们使用方法定义了一个名为`swipe`的变量。请注意，在方法签名中我们不必使用`self`，因为我们在方法中没有使用类变量。

# 创建蓝牙门铃

现在我们知道如何从 Blue Dot 读取按钮信息，我们可以构建一个蓝牙门铃按钮。我们将重写我们的`DoorbellAlarm`类，并使用来自 Blue Dot 的简单按钮按下来激活警报，如下所示：

1.  从应用程序菜单中打开 Thonny | 编程 | Thonny Python IDE

1.  单击新图标创建新文件

1.  在文件中键入以下内容：

```py
from gpiozero import RGBLED
from gpiozero import Buzzer
from time import sleep

class DoorbellAlarmAdvanced:

    led = RGBLED(red=17, green=22, blue=27)
    buzzer = Buzzer(26)
    num_of_times = 0
    delay = 0

    def __init__(self, num_of_times, delay):
        self.num_of_times = num_of_times
        self.delay = delay

    def play_sequence(self):
        num = 0
        while num < self.num_of_times:
            self.buzzer.on()
            self.light_show()
            sleep(self.delay)
            self.buzzer.off()
            sleep(self.delay)
            num += 1

    def light_show(self):
        self.led.color=(1,0,0)
        sleep(0.1)
        self.led.color=(0,1,0)
        sleep(0.1)
        self.led.color=(0,0,1)
        sleep(0.1)
        self.led.off()

if __name__=="__main__":

    doorbell_alarm = DoorbellAlarmAdvanced(5,1)
    doorbell_alarm.play_sequence()
```

1.  将文件保存为`DoorbellAlarmAdvanced.py`

我们的新类`DoorbellAlarmAdvanced`是`DoorbellAlarm`类的修改版本。我们所做的基本上是添加了一个我们称之为`delay`的新类属性。这个类属性将用于改变蜂鸣器响铃之间的延迟时间。正如您在代码中看到的，为了进行这一更改而修改的两个方法是`__init__`和`play_sequence`**。**

现在我们已经对我们的警报进行了更改，让我们创建一个简单的门铃程序如下：

1.  从应用程序菜单中打开 Thonny | 编程 | Thonny Python IDE

1.  单击新图标创建新文件

1.  在文件中键入以下内容：

```py
from bluedot import BlueDot
from signal import pause
from DoorbellAlarmAdvanced import DoorbellAlarmAdvanced

class SimpleDoorbell:

 def pressed():
 doorbell_alarm = DoorbellAlarmAdvanced(5, 1)
 doorbell_alarm.play_sequence()

 blue_dot = BlueDot()
 blue_dot.when_pressed = pressed

if __name__=="__main__":

 doorbell_alarm = SimpleDoorbell()
 pause()
```

1.  将文件保存为`SimpleDoorbell.py`并运行

1.  将蓝点应用程序连接到树莓派，如果尚未连接

1.  按下大蓝点

您应该听到五声持续一秒钟的响铃，每隔一秒钟响一次。您还会看到 RGB LED 经历了一个短暂的灯光秀。正如您所看到的，代码非常简单。我们导入我们的新`DoorbellAlarmAdvanced`类，然后在`pressed`方法中使用`doorbell_alarm`变量初始化类后调用`play_sequence`方法。

我们在创建`DoorbellAlarmAdvanced`类时所做的更改被用于我们的代码，以允许我们设置响铃之间的延迟时间。

# 创建一个秘密蓝牙门铃

在我们回答门铃之前知道谁在门口会不会很好？我们可以利用蓝点应用程序的滑动功能。要创建一个秘密的蓝牙门铃（秘密是我们与门铃互动的方式，而不是门铃的秘密位置），请执行以下操作：

1.  从应用程序菜单中打开 Thonny | 编程 | Thonny Python IDE

1.  单击新图标创建新文件

1.  在文件中键入以下内容：

```py
from bluedot import BlueDot
from signal import pause
from DoorbellAlarmAdvanced import DoorbellAlarmAdvanced

class SecretDoorbell:

    def swiped(swipe):

        if swipe.up:
            doorbell_alarm = DoorbellAlarmAdvanced(5, 0.5)
            doorbell_alarm.play_sequence()
        elif swipe.down:
            doorbell_alarm = DoorbellAlarmAdvanced(3, 2)
            doorbell_alarm.play_sequence()
        elif swipe.left:
            doorbell_alarm = DoorbellAlarmAdvanced(1, 5)
            doorbell_alarm.play_sequence()
        elif swipe.right:
            doorbell_alarm = DoorbellAlarmAdvanced(1, 0.5)
            doorbell_alarm.play_sequence()

    blue_dot = BlueDot()
    blue_dot.when_swiped = swiped    

if __name__=="__main__":

    doorbell = SecretDoorbell()
    pause()
```

1.  将文件保存为`SecretDoorbell.py`并运行

1.  将蓝点应用程序连接到树莓派，如果尚未连接

1.  向上滑动蓝点

您应该听到五声短促的响铃，同时看到 RGB LED 的灯光秀。尝试向下、向左和向右滑动。每次您应该得到不同的响铃序列。

那么，我们在这里做了什么？基本上，我们将回调附加到`when_swiped`事件，并通过`if`语句，创建了具有不同初始值的新`DoorbellAlarmAdvanced`对象。

通过这个项目，我们现在可以知道谁在门口，因为我们可以为不同的朋友分配各种滑动手势。

# 摘要

在本章中，我们使用树莓派和蓝点安卓应用程序创建了一个蓝牙门铃应用程序。我们首先学习了一些关于 RGB LED 的知识，然后将其与主动蜂鸣器一起用于警报电路。

通过蓝点应用程序，我们学会了如何将蓝牙按钮连接到树莓派。我们还学会了如何使用一些蓝点手势，并创建了一个具有不同响铃持续时间的门铃应用程序。

在第十二章中，*增强我们的物联网门铃*，我们将扩展我们的门铃功能，并在有人按下按钮时发送文本消息。

# 问题

1.  RGB LED 与普通 LED 有什么不同？

1.  正确还是错误？蓝点应用程序可以在 Google Play 商店中找到。

1.  什么是共阳极？

1.  正确还是错误？RGB LED 内的三种颜色是红色、绿色和黄色。

1.  如何将蓝点应用程序与树莓派配对？

1.  正确还是错误？蓝牙是一种用于极长距离的通信技术。

1.  `DoorbellAlarm`和`DoorbellAlarmAdvanced`之间有什么区别？

1.  正确还是错误？GPIO Zero 库包含一个名为`RGBLED`的类。

1.  正确还是错误？蓝点应用程序可以用于记录滑动手势。

1.  `SimpleDoorbell`和`SecretDoorbell`类之间有什么区别？

# 进一步阅读

要了解更多关于 Blue Dot Android 应用程序的信息，请访问文档页面[`bluedot.readthedocs.io`](https://bluedot.readthedocs.io)。
