# 使用 Python 控制机器人小车

在第十三章中，*介绍树莓派机器人小车*，我们建造了 T.A.R.A.S 机器人小车。在章节结束时，我们讨论了如何通过代码控制 T.A.R.A.S。在本章中，我们将开始编写代码来实现这一点。

我们将首先编写简单的 Python 代码，然后利用 GPIO Zero 库使车轮向前移动，移动携带摄像头的伺服电机，并点亮机器人小车后面的 LED 灯。

然后，我们将使用类组织我们的代码，然后进一步增强它，让 T.A.R.A.S 执行秘密安全任务。

本章将涵盖以下主题：

+   查看 Python 代码

+   修改机器人小车的 Python 代码

+   增强代码

# 完成本章所需的知识

如果您跳到本章而没有经历前几章的项目，让我概述一下您完成以下项目所需的技能。当然，我们必须知道如何在 Raspbian OS 中四处走动，以便找到我们的**集成开发环境**（**IDE**）。

在完成 T.A.R.A.S 的编程后，您可能会倾向于利用新技能与其他构建树莓派机器人的人竞争。Pi Wars ([`piwars.org/`](https://piwars.org/))就是这样一个地方。Pi Wars 是一个在英国剑桥举行的国际机器人竞赛。在一个周末内，最多有 76 支队伍参加基于挑战的机器人竞赛。尽管它被称为 Pi Wars，但您可以放心，您不会带着一箱破碎的零件回来，因为每场比赛都是非破坏性的挑战。查看[`piwars.org/`](https://piwars.org/)，或在 YouTube 上搜索 Pi Wars 视频以获取更多信息。

此外，需要对 Python 有一定的了解，因为本章的所有编码都将使用 Python 完成。由于我喜欢尽可能多地使用面向对象的方法，一些**面向对象编程**（**OOP**）的知识也将帮助您更好地从本章中受益。

# 项目概述

在本章中，我们将编程 T.A.R.A.S 在桌子周围跳舞并拍照。本章的项目应该需要几个小时才能完成。

# 入门

要完成这个项目，需要以下内容：

+   树莓派 3 型号（2015 年或更新型号）

+   USB 电源供应

+   计算机显示器

+   USB 键盘

+   USB 鼠标

+   已完成的 T.A.R.A.S 机器人小车套件（参见第十三章，*介绍树莓派机器人小车*）

# 查看 Python 代码

在某种程度上，我们的机器人小车项目就像是我们在前几章中所做的代码的概述。通过使用 Python 和令人惊叹的 GPIO Zero 库，我们能够从 GPIO 读取传感器数据，并通过向 GPIO 引脚写入数据来控制输出设备。在接下来的步骤中，我们将从非常简单的 Python 代码和 GPIO Zero 库开始。如果您已经完成了本书中的一些早期项目，那么这些代码对您来说将会非常熟悉。

# 控制机器人小车的驱动轮

让我们看看是否可以让 T.A.R.A.S 移动一点。我们将首先编写一些基本代码来让机器人小车前后移动：

1.  从应用程序菜单 | 编程 | Thonny Python IDE 中打开 Thonny

1.  单击新图标创建新文件

1.  将以下代码输入文件：

```py
from gpiozero import Robot
from time import sleep

robot = Robot(left=(5,6), right=(22,27))
robot.forward(0.2)
sleep(0.5)
robot.backward(0.2)
sleep(0.5)
robot.stop()
```

1.  将文件保存为`motor-test.py`

1.  运行代码

您应该看到机器人小车向前移动`0.5`秒，然后向后移动相同的时间。如果路上没有障碍物，机器人小车应该回到起始位置。代码相当简单明了；然而，我们现在将对其进行讨论。

我们首先导入我们需要的库：`Robot`和`sleep`。之后，我们实例化一个名为`robot`的`Robot`对象，并将其配置为左侧电机的`5`和`6`引脚，右侧电机的`22`和`27`引脚。之后，我们以`0.2`的速度将机器人向前移动。为了使机器人移动更快，我们增加这个值。稍作延迟后，我们使用`robot.backward(0.2)`命令将机器人返回到原始位置。

需要注意的一点是电机的旋转方式，它们会一直旋转，直到使用`robot.stop()`命令停止。

如果发现电机没有按预期移动，那是因为接线的问题。尝试尝试不同的接线和更改`Robot`对象的引脚号码（left=(5,6), right=(22,27)）。可能需要几次尝试才能搞定。

# 移动机器人车上的舵机

我们现在将测试舵机。为了做到这一点，我们将从右到左摆动机器人摄像头支架（机器人的头）：

1.  从应用程序菜单 | 编程 | Thonny Python IDE 打开 Thonny

1.  单击新图标创建一个新文件

1.  将以下代码输入文件：

```py
import Adafruit_PCA9685 from time import sleep pwm = Adafruit_PCA9685.PCA9685() servo_min = 150
servo_max = 600 while True:
    pwm.set_pwm(0, 0, servo_min)
    sleep(5)
    pwm.set_pwm(0, 0, servo_max)
    sleep(5) 
```

1.  将文件保存为`servo-test.py`

1.  运行代码

您应该看到机器人头部向右移动，等待`5`秒，然后向左移动。

在代码中，我们首先导入`Adafruit_PCA9685`库。在导入`sleep`函数后，我们创建一个名为`pwm`的`PCA9685`对象。当然，这是一个使用 Adafruit 代码构建的对象，用于支持 HAT。然后我们分别设置舵机可以移动的最小和最大值，分别为`servo_min`和`servo_max`。

如果您没有得到预期的结果，请尝试调整`servo_min`和`servo_max`的值。我们在第五章中稍微涉及了一些关于舵机的内容，*用 Python 控制舵机*。

# 拍照

您可能还记得在以前的章节中使用树莓派摄像头；特别是第九章，*构建家庭安全仪表板*，我们在那里使用它为我们的安全应用程序拍照。由于 T.A.R.A.S 将是我们可靠的安全代理，它有能力拍照是有意义的。让我们编写一些代码来测试一下我们的机器人车上的摄像头是否工作：

1.  从应用程序菜单 | 编程 | Thonny Python IDE 打开 Thonny

1.  单击新图标创建一个新文件

1.  输入以下代码：

```py
from picamera import PiCamera import time camera = PiCamera() camera.capture("/home/pi/image-" + time.ctime() + ".png")   
```

1.  将文件保存为`camera-test.py`

1.  运行代码

如果一切设置正确，您应该在`/home/pi`目录中看到一个图像文件，文件名为`image`，后跟今天的日期。

# 发出蜂鸣声

我们的安全代理受限于发出噪音以警示我们并吓跑潜在的入侵者。在这一部分，我们将测试安装在 T.A.R.A.S 上的有源蜂鸣器。

旧的英国警察口哨是过去警察官员必须自卫的最早和可信赖的装备之一。英国警察口哨以其独特的声音，允许警官之间进行交流。尽管警察口哨已不再使用，但它的遗产对社会产生了影响，以至于“吹哨人”这个术语至今仍用来指代揭露隐藏的不公正或腐败的人。

1.  从应用程序菜单 | 编程 | Thonny Python IDE 打开 Thonny

1.  单击新图标创建一个新文件

1.  将以下代码输入文件：

```py
from gpiozero import Buzzer
from time import sleep

buzzer = Buzzer(12)
buzzer.on()
sleep(5)
buzzer.off()
```

1.  将文件保存为`buzzer-test.py`

1.  运行代码

您应该听到蜂鸣器声音持续`5`秒，然后关闭。

# 让 LED 闪烁

在 T.A.R.A.S 的背面，我们安装了两个 LED（最好是一个红色和一个绿色）。我们以前使用简单的 GPIO Zero 库命令来闪烁 LED，所以这对我们来说不应该是一个挑战。让我们更进一步，创建可以用来封装 LED 闪烁模式的代码：

1.  从应用程序菜单中打开 Thonny | 编程 | Thonny Python IDE

1.  点击新图标创建一个新文件

1.  输入以下代码：

```py
from gpiozero import LEDBoard
from time import sleep

class TailLights:

    led_lights = LEDBoard(red=21, green=20)

    def __init__(self):
        self.led_lights.on()
        sleep(0.25)
        self.led_lights.off()
        sleep(0.25)

    def blink_red(self, num, duration):
        for x in range(num):
            self.led_lights.red.on()
            sleep(duration)
            self.led_lights.red.off()
            sleep(duration)

    def blink_green(self, num, duration):
        for x in range(num):
            self.led_lights.green.on()
            sleep(duration)
            self.led_lights.green.off()
            sleep(duration)

    def blink_alternating(self, num, duration):
        for x in range(num):
            self.led_lights.red.off()
            self.led_lights.green.on()
            sleep(duration)
            self.led_lights.red.on()
            self.led_lights.green.off()
            sleep(duration)
        self.led_lights.red.off()

    def blink_together(self, num, duration):
        for x in range(num):
            self.led_lights.on()
            sleep(duration)
            self.led_lights.off()
            sleep(duration)

    def alarm(self, num):
        for x in range(num):
            self.blink_alternating(2, 0.25)
            self.blink_together(2, 0.5)

if __name__=="__main__":

    tail_lights = TailLights()
    tail_lights.alarm(20) 
```

1.  将文件保存为`TailLights.py`

1.  运行代码

你应该看到 LED 显示器闪烁 20 秒。但值得注意的是我们的代码中使用了 GPIO Zero 库的`LEDBoard`类，如下所示：

```py
led_lights = LEDBoard(red=21, green=20)
```

在这段代码中，我们从`LEDBoard`类中实例化一个名为`led_lights`的对象，并使用`red`和`green`的值来配置它，分别指向`21`和`20`的 GPIO 引脚。通过使用`LEDBoard`，我们能够分别或作为一个单元来控制 LED。`blink_together`方法控制 LED 作为一个单元，如下所示：

```py
def blink_together(self, num, duration):
        for x in range(num):
            self.led_lights.on()
            sleep(duration)
            self.led_lights.off()
            sleep(duration)
```

我们的代码相当容易理解；然而，还有一些其他事情我们应该指出。当我们初始化`TailLights`对象时，我们让 LED 短暂闪烁以表示对象已被初始化。这样可以在以后进行故障排除；尽管，如果我们觉得代码是多余的，那么我们以后可以将其删除：

```py
def __init__(self):
        self.led_lights.on()
        sleep(0.25)
        self.led_lights.off()
        sleep(0.25)
```

保留初始化代码可能会很方便，尤其是当我们想要确保我们的 LED 没有断开连接时（毕竟，谁在尝试连接其他东西时没有断开过某些东西呢？）。要从 shell 中执行此操作，请输入以下代码：

```py
import TailLights
tail_lights = TailLights.TailLights()
```

你应该看到 LED 闪烁了半秒钟。

# 修改机器人车 Python 代码

现在我们已经测试了电机、舵机、摄像头和 LED，是时候将代码修改为类，以使其更加统一了。在本节中，我们将让 T.A.R.A.S 跳舞。

# 移动车轮

让我们从封装移动机器人车轮的代码开始：

1.  从应用程序菜单中打开 Thonny | 编程 | Thonny Python IDE

1.  点击新图标创建一个新文件

1.  将以下代码输入文件中：

```py
from gpiozero import Robot
from time import sleep

class RobotWheels:

    robot = Robot(left=(5, 6), right=(22, 27))

    def __init__(self):
        pass

    def move_forward(self):
        self.robot.forward(0.2)

    def move_backwards(self):
        self.robot.backward(0.2)

    def turn_right(self):
        self.robot.right(0.2)

    def turn_left(self):
        self.robot.left(0.2)

    def dance(self):
        self.move_forward()
        sleep(0.5)
        self.stop()
        self.move_backwards()
        sleep(0.5)
        self.stop()
        self.turn_right()
        sleep(0.5)
        self.stop()
        self.turn_left()
        sleep(0.5)
        self.stop()

    def stop(self):
        self.robot.stop()

if __name__=="__main__":

    robot_wheels = RobotWheels()
    robot_wheels.dance() 
```

1.  将文件保存为`RobotWheels.py`

1.  运行代码

你应该看到 T.A.R.A.S 在你面前跳了一小段舞。确保连接到 T.A.R.A.S 的电线松动，这样 T.A.R.A.S 就可以做自己的事情。谁说机器人不能跳舞呢？

这段代码相当容易理解。但值得注意的是我们如何从`dance`方法中调用`move_forward`、`move_backwards`、`turn_left`和`turn_right`函数。我们实际上可以参数化移动之间的时间，但这会使事情变得更加复杂。`0.5`秒的延迟（加上硬编码的速度`0.2`）似乎非常适合一个不会从桌子上掉下来的跳舞机器人。可以把 T.A.R.A.S 想象成在一个非常拥挤的舞池上，没有太多的移动空间。

但等等，还有更多。T.A.R.A.S 还可以移动头部、点亮灯光并发出一些声音。让我们开始添加这些动作。

# 移动头部

由于 T.A.R.A.S 上的摄像头连接到头部，因此将头部运动（摄像头支架舵机）与摄像头功能封装起来是有意义的：

1.  从应用程序菜单中打开 Thonny | 编程 | Thonny Python IDE

1.  点击新图标创建一个新文件

1.  将以下代码输入文件中：

```py
from time import sleep
from time import ctime
from picamera import PiCamera
import Adafruit_PCA9685

class RobotCamera:

    pan_min = 150
    pan_centre = 375
    pan_max = 600
    tilt_min = 150
    tilt_max = 200
    camera = PiCamera()
    pwm = Adafruit_PCA9685.PCA9685()

    def __init__(self):
        self.tilt_up()

    def pan_right(self):
        self.pwm.set_pwm(0, 0, self.pan_min)
        sleep(2)

    def pan_left(self):
        self.pwm.set_pwm(0, 0, self.pan_max)
        sleep(2)

    def pan_mid(self):
        self.pwm.set_pwm(0, 0, self.pan_centre)
        sleep(2)

    def tilt_down(self):
        self.pwm.set_pwm(1, 0, self.tilt_max)
        sleep(2)

    def tilt_up(self):
        self.pwm.set_pwm(1, 0, self.tilt_min)
        sleep(2)

    def take_picture(self):
        sleep(2)
        self.camera.capture("/home/pi/image-" + ctime() + ".png")

    def dance(self):
        self.pan_right()
        self.tilt_down()
        self.tilt_up()
        self.pan_left()
        self.pan_mid()

    def secret_dance(self):
        self.pan_right()
        self.tilt_down()
        self.tilt_up()
        self.pan_left()
        self.pan_mid()
        self.take_picture()

if __name__=="__main__":

    robot_camera = RobotCamera()
    robot_camera.dance()
```

1.  将文件保存为`RobotCamera.py`

1.  运行代码

你应该看到 T.A.R.A.S 把头向右转，然后向下，然后向上，然后全部向左，然后返回到中间并停止。

再次，我们尝试编写我们的代码，使其易于理解。当实例化`RobotCamera`对象时，`init`方法确保 T.A.R.A.S 在移动头部之前将头部抬起：

```py
def __init__(self):
    self.tilt_up()
```

通过调用`RobotCamera`类，我们将代码结构化为查看机器人车头部舵机和运动的一部分。尽管我们在示例中没有使用摄像头，但我们很快就会使用它。为舵机位置设置的最小和最大值是通过试验和错误确定的，如下所示：

```py
pan_min = 150
pan_centre = 375
pan_max = 600
tilt_min = 150
tilt_max = 200
```

尝试调整这些值以适应您的 T.A.R.A.S 机器人车的构建。

`dance`和`secret_dance`方法使用机器人车头执行一系列动作来模拟跳舞。它们基本上是相同的方法（除了`take_picture`在最后调用），`secret_dance`方法使用树莓派摄像头拍照，并以基于日期的名称存储在主目录中。

# 发出声音

现在 T.A.R.A.S 可以移动身体和头部了，是时候发出一些声音了：

1.  从应用程序菜单中打开 Thonny | 编程 | Thonny Python IDE

1.  单击新图标创建新文件

1.  将以下代码输入文件

```py
from gpiozero import Buzzer
from time import sleep

class RobotBeep:

    buzzer = Buzzer(12)
    notes = [[0.5,0.5],[0.5,1],[0.2,0.5],[0.5,0.5],[0.5,1],[0.2,0.5]]

    def __init__(self, play_init=False):

        if play_init:
            self.buzzer.on()
            sleep(0.1)
            self.buzzer.off()
            sleep(1)

    def play_song(self):

        for note in self.notes:
            self.buzzer.on()
            sleep(note[0])
            self.buzzer.off()
            sleep(note[1])

if __name__=="__main__":

    robot_beep = RobotBeep(True)
```

1.  将文件保存为`RobotBeep.py`

1.  运行代码

您应该听到 T.A.R.A.S 上的有源蜂鸣器发出短促的蜂鸣声。这似乎是为了做这个而写了很多代码，不是吗？啊，但是等到下一节，当我们充分利用`RobotBeep`类时。

`RobotBeep`的`init`函数允许我们打开和关闭类实例化时听到的初始蜂鸣声。这对于测试我们的蜂鸣器是否正常工作很有用，我们通过在创建`robot_beep`对象时向类传递`True`来进行测试：

```py
robot_beep = RobotBeep(True)
```

`notes`列表和`play_song`方法执行类的实际魔术。该列表实际上是一个列表的列表，因为每个值代表蜂鸣器播放或休息的时间：

```py
for note in self.notes:
    self.buzzer.on()
    sleep(note[0])
    self.buzzer.off()
    sleep(note[1])
```

循环遍历`notes`列表，查看`note`变量。我们使用第一个元素作为保持蜂鸣器开启的时间长度，第二个元素作为在再次打开蜂鸣器之前休息的时间量。换句话说，第一个元素确定音符的长度，第二个元素确定该音符与下一个音符之间的间隔。`notes`列表和`play_song`方法使 T.A.R.A.S 能够唱歌（尽管没有旋律）。

我们将在下一节中使用`play_song`方法。

# 增强代码

这是一个寒冷，黑暗和阴郁的十二月之夜。我们对我们的对手知之甚少，但我们知道他们喜欢跳舞。T.A.R.A.S 被指派到敌人领土深处的一个当地舞厅。在这个晚上，所有感兴趣的人都在那里。如果您选择接受的话，您的任务是编写一个程序，让 T.A.R.A.S 在舞厅拍摄秘密照片。但是，它不能看起来像 T.A.R.A.S 在拍照。T.A.R.A.S 必须跳舞！如果我们的对手发现 T.A.R.A.S 在拍照，那将是糟糕的。非常糟糕！想象一下帝国反击战中的 C3PO 糟糕。

# 将我们的代码连接起来

因此，我们有能力让 T.A.R.A.S 移动头部和身体，发出声音，发光和拍照。让我们把所有这些放在一起，以便完成任务：

1.  从应用程序菜单中打开 Thonny | 编程 | Thonny Python IDE

1.  单击新图标创建新文件

1.  将以下内容输入文件：

```py
from RobotWheels import RobotWheels
from RobotBeep import RobotBeep
from TailLights import TailLights
from RobotCamera import RobotCamera

class RobotDance:

    light_show = [2,1,4,5,3,1]

    def __init__(self):

        self.robot_wheels = RobotWheels()
        self.robot_beep = RobotBeep()
        self.tail_lights = TailLights()
        self.robot_camera = RobotCamera()

    def lets_dance_incognito(self):
        for tail_light_repetition in self.light_show:
            self.robot_wheels.dance()
            self.robot_beep.play_song()
            self.tail_lights.alarm(tail_light_repetition)
            self.robot_camera.secret_dance()

if __name__=="__main__":

    robot_dance = RobotDance()
    robot_dance.lets_dance_incognito()

```

1.  将文件保存为`RobotDance.py`

1.  运行代码

在秘密拍照之前，您应该看到 T.A.R.A.S 执行一系列动作。如果舞蹈结束后检查树莓派`home`文件夹，您应该会看到六张新照片。

我们代码中值得注意的是`light_show`列表的使用。我们以两种方式使用此列表。首先，将列表中存储的值传递给我们在`RobotDance`类中实例化的`TailLights`对象的`alarm`方法。我们在`lets_dance_incognito`方法中使用`tail_light_repetition`变量，如下所示：

```py
def lets_dance_incognito(self):
    for tail_light_repetition in self.light_show:
        self.robot_wheels.dance()
        self.robot_beep.play_song()
        self.tail_lights.alarm(tail_light_repetition)
        self.robot_camera.secret_dance()
```

如您在先前的代码中所看到的，`TailLights`类的变量`alarm`方法被命名为`tail_lights`。这将导致 LED 根据`tail_light_repetition`的值多次执行它们的序列。例如，当将值`2`传递给`alarm`方法（`light_show`列表中的第一个值）时，LED 序列将执行两次。

我们运行`lets_dance_incognito`方法六次。这基于`light_show`列表中的值的数量。这是我们使用`light_show`的第二种方式。为了增加或减少 T.A.R.A.S 执行舞蹈的次数，我们可以从`light_show`列表中添加或减去一些数字。

当我们在名为`robot_camera`的`RobotCamera`对象上调用`secret_dance`方法时，对于`light_show`列表中的每个值（在本例中为六），在舞蹈结束后，我们的家目录中应该有六张以日期命名的照片。

T.A.R.A.S 完成舞蹈后，请检查家目录中 T.A.R.A.S 在舞蹈期间拍摄的照片。任务完成！

# 总结

在本章结束时，您应该熟悉使用 Python 代码控制树莓派驱动的机器人。我们首先通过简单的代码使机器人车上的各种组件工作。在我们确信机器人车确实使用我们的 Python 命令移动后，我们将代码封装在类中，以便更容易使用。这导致了`RobotDance`类，其中包含对类的调用，这些类又封装了我们机器人的控制代码。这使我们能够使用`RobotDance`类作为黑匣子，将控制代码抽象化，并使我们能够专注于为 T.A.R.A.S 设计舞步的任务。

在第十五章中，*将机器人车的感应输入连接到网络*，我们将从 T.A.R.A.S（距离传感器值）中获取感官信息，并将其发布到网络上，然后将 T.A.R.A.S 从桌面上的电线中释放出来，让其自由行动。

# 问题

1.  真或假？`LEDBoard`对象允许我们同时控制许多 LED。

1.  真或假？`RobotCamera`对象上的笔记列表用于移动摄像机支架。

1.  真或假？我们虚构故事中的对手喜欢跳舞。

1.  `dance`和`secret_dance`方法之间有什么区别？

1.  机器人的`gpiozero`库的名称是什么？

1.  受老警察哨子启发，给出揭露犯罪行为的行为的术语是什么？

1.  真或假？封装控制代码是一个毫无意义和不必要的步骤。

1.  `TailLights`类的目的是什么？

1.  我们将使用哪个类和方法将机器人车转向右侧？

1.  `RobotCamera`类的目的是什么？

# 进一步阅读

学习 GPIO Zero 的最佳参考书之一是 GPIO Zero PDF 文档本身。搜索 GPIO Zero PDF，然后下载并阅读它。
