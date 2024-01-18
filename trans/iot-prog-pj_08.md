# 使用 Python 读取树莓派 GPIO 传感器数据

在第七章中，*设置树莓派 Web 服务器*，我们使用 GPIO Zero 库来控制舵机和 LED 灯。在本章中，我们将使用 GPIO Zero 来读取 GPIO 端口的输入。首先，我们将从一个简单的按钮开始，然后转向**被动红外**（**PIR**）运动传感器和蜂鸣器。

能够从 GPIO 读取传感器数据将使我们能够构建我们的物联网家庭安全仪表板。在本章结束时，我们应该对使用连接到 GPIO 的组件编程树莓派非常熟悉。

本章将涵盖以下主题：

+   读取按钮的状态

+   从红外运动传感器读取状态

+   使用红外传感器修改`Hello LED`

# 项目概述

在本章中，我们将创建两种不同类型的报警系统。我们将首先学习如何从按钮读取 GPIO 传感器数据。然后，我们将学习如何与 PIR 传感器和距离传感器交互。最后，我们将学习如何连接一个有源蜂鸣器。

本章应该需要大约 3 小时完成。

# 入门

要完成这个项目，需要以下材料：

+   树莓派 3 型（2015 年或更新型号）

+   一个 USB 电源供应

+   一台电脑显示器

+   一个 USB 键盘

+   一个 USB 鼠标

+   一个面包板

+   跳线

+   一个 PIR 传感器

+   一个距离传感器

+   一个有源蜂鸣器

+   一个 LED

+   一个按钮（瞬时）

+   一个按钮（锁定式）

+   一个键开关（可选）

# 读取按钮的状态

`Button`，来自`GPIO Zero`库，为我们提供了一种与连接到 GPIO 的典型按钮进行交互的简单方法。本节将涵盖以下内容：

+   使用 GPIO Zero 与按钮

+   使用 Sense HAT 模拟器和 GPIO Zero 按钮

+   使用长按按钮切换 LED

# 使用 GPIO Zero 与按钮

使用 GPIO 连接按钮相对容易。以下是显示连接过程的连接图：

![](img/d70e80e0-3c45-4b64-acf3-417770f347c0.png)

将按钮连接，使一端使用跳线连接到地。将另一端连接到树莓派上的 GPIO 4。

在 Thonny 中，创建一个新文件并将其命名为`button_press.py`。然后，输入以下内容并运行：

```py
from gpiozero import Button
from time import sleep

button = Button(4)
while True:
    if button.is_pressed:
     print("The Button on GPIO 4 has been pressed")
     sleep(1)
```

当你按下按钮时，你现在应该在壳中看到消息`“GPIO 4 上的按钮已被按下”`。代码将持续运行，直到你点击重置按钮。

让我们来看看代码。我们首先从`GPIO Zero`导入`Button`，并从`time`库导入`sleep`：

```py
from gpiozero import Button
from time import sleep
```

然后，我们创建一个新的`button`对象，并使用以下代码将其分配给 GPIO 引脚`4`：

```py
button = Button(4)
```

我们的连续循环检查按钮当前是否被按下，并在壳中打印出一条语句：

```py
while True:
    if button.is_pressed:
     print("The Button on GPIO 4 has been pressed")
     sleep(1)
```

# 使用 Sense HAT 模拟器和 GPIO Zero 按钮

我们每天都使用按钮，无论是在电梯中选择楼层还是启动汽车。现代技术使我们能够将按钮与其控制的物理设备分离。换句话说，按下按钮可以引发许多与按钮无关的事件。我们可以使用我们的按钮和 Sense HAT 模拟器来模拟这种分离。

我可以想象你们中的一些人在想分离按钮与其控制对象实际意味着什么。为了帮助你们形象化，想象一下一个控制灯的锁定式按钮。当按钮被按下时，电路闭合，电流通过按钮上的引线流动。通过使用控制器和计算机，比如树莓派，按钮所需做的就是改变它的状态。控制器或计算机接收该状态并执行与按钮本身完全分离的操作。

从 Raspbian 的编程菜单中加载 Sense HAT 模拟器。在 Thonny 中创建一个名为`sense-button.py`的新的 Python 文件。输入以下代码到文件中，然后在完成后单击运行图标：

![](img/38d82455-991a-4e29-8de5-31d33de635a7.png)

```py
from gpiozero import Button
from sense_emu import SenseHat
from time import sleep

button = Button(4)
sense = SenseHat()

def display_x_mark(rate=1):
    sense.clear()
    X = (255,0,0)
    O = (255,255,255)
    x_mark = [
              X,O,O,O,O,O,O,X,
              O,X,O,O,O,O,X,O,
              O,O,X,O,O,X,O,O,
              O,O,O,X,X,O,O,O,
              O,O,O,X,X,O,O,O,
              O,O,X,O,O,X,O,O,
              O,X,O,O,O,O,X,O,
              X,O,O,O,O,O,O,X
    ]
    sense.set_pixels(x_mark)

while True:
    if button.is_pressed:
        display_x_mark()
        sleep(1)
    else:
        sense.clear()
```

如果您的代码没有任何错误，当您按下按钮时，您应该会看到 Sense HAT 模拟器上的显示屏变成白色背景上的红色`X`：

让我们稍微解释一下上面的代码。我们首先导入我们代码所需的库：

```py
from gpiozero import Button
from sense_emu import SenseHat
from time import sleep
```

然后我们创建新的按钮和 Sense HAT 模拟器对象。我们的`button`再次连接到 GPIO 引脚`4`：

```py
button = Button(4)
sense = SenseHat()
```

`display_x_mark`方法通过使用`SenseHat`方法`set_pixels`在显示器上创建一个`X`：

```py
def display_x_mark(rate=1):
    sense.clear()
    X = (255,0,0)
    O = (255,255,255)
    x_mark = [
              X,O,O,O,O,O,O,X,
              O,X,O,O,O,O,X,O,
              O,O,X,O,O,X,O,O,
              O,O,O,X,X,O,O,O,
              O,O,O,X,X,O,O,O,
              O,O,X,O,O,X,O,O,
              O,X,O,O,O,O,X,O,
              X,O,O,O,O,O,O,X
    ]
    sense.set_pixels(x_mark)
```

`X`和`O`变量用于保存颜色代码，`(255,0,0)`表示红色，`(255,255,255)`表示白色。变量`x_mark`创建一个与 Sense HAT 模拟器屏幕分辨率匹配的 8 x 8 图案。`x_mark`被传递到`SenseHAT`对象的`set_pixels`方法中。

我们的连续循环检查按钮的`is_pressed`状态，并在状态返回`true`时调用`display_x_mark`方法。然后该方法会在白色背景上打印一个红色的`X`。

当按钮未处于按下状态时，使用`sense.clear()`清除显示：

```py
while True:
    if button.is_pressed:
        display_x_mark()
        sleep(1)
    else:
        sense.clear()
```

# 使用长按按钮切换 LED

使用`GPIO Zero`库，我们不仅可以检测按钮何时被按下，还可以检测按下多长时间。我们将使用`hold_time`属性和`when_held`方法来确定按钮是否被按下了一段时间。如果超过了这段时间，我们将打开和关闭 LED。

以下是我们程序的电路图。将按钮连接到 GPIO 引脚 4。使用 GPIO 引脚 17 来连接 LED，如图所示：

![](img/e895d707-f0b5-45aa-85f4-b08962d1569d.png)

在 Thonny 中创建一个名为`buttonheld-led.py`的新文件。输入以下内容并单击运行：

```py
from gpiozero import LED
from gpiozero import Button

led = LED(17)
button = Button(4)
button.hold_time=5

while True:
    button.when_held = lambda: led.toggle()
```

按下按钮保持`5`秒。您应该会看到 LED 切换开。现在再按下`5`秒。LED 应该会切换关闭。

我们已经在之前的示例中涵盖了代码的前四行。让我们看看按钮的保持时间是如何设置的：

```py
button.hold_time=5
```

这一行将按钮的保持时间设置为`5`秒。`when_held`方法在我们的连续循环中被调用：

```py
button.when_held = lambda: led.toggle()
```

使用 lambda，我们能够创建一个匿名函数，以便我们可以在`LED`对象`led`上调用`toggle()`。这会将 LED 打开和关闭。

# 从红外运动传感器中读取状态

使用运动传感器的报警系统是我们社会中无处不在的一部分。使用我们的树莓派，它们非常容易构建。我们将在本节中涵盖以下内容：

+   什么是 PIR 传感器？

+   使用`GPIO 蜂鸣器`类

+   构建一个基本的报警系统

# 什么是 PIR 传感器？

PIR 传感器是一种运动传感器，用于检测运动。 PIR 传感器的应用主要基于为安全系统检测运动。 PIR 代表被动红外线，PIR 传感器包含一个检测低级辐射的晶体。 PIR 传感器实际上是由两半构成的，因为两半之间的差异才能检测到运动。以下是一个廉价的 PIR 传感器的照片：

![](img/80f92423-d6d9-4649-b969-40074ac2036a.png)在上面的照片中，我们可以看到正（**+**）、负（**-**）和信号（**S**）引脚。这个特定的 PIR 传感器很适合面包板上。

以下是我们 PIR 电路的接线图。正极引脚连接到树莓派的 5V DC 输出。负极引脚连接到地（GND），信号引脚连接到 GPIO 引脚 4：

![](img/d056ccfc-b83f-47cb-b414-f5e204570fd2.png)

在 Thonny 中创建一个名为`motion-sensor.py`的新的 Python 文件。输入以下代码并运行：

```py
from gpiozero import MotionSensor
from time import sleep

motion_sensor = MotionSensor(4)

while True:
    if motion_sensor.motion_detected:
        print('Detected Motion!')
        sleep(2)
    else:
        print('No Motion Detected!')
        sleep(2)
```

当您靠近 PIR 传感器时，您应该看到一条消息，上面写着`检测到运动！`。尝试保持静止，看看是否可以在 shell 中显示消息`未检测到运动！`。

我们的代码开始时从`GPIO Zero`库中导入`MotionSensor`类：

```py
from gpiozero import MotionSensor
```

导入`sleep`类后，我们创建一个名为`motion_sensor`的新`MotionSensor`对象，附加了数字`4`，以便让我们的程序在 GPIO 引脚 4 上寻找信号：

```py
motion_sensor = MotionSensor(4)
```

在我们的循环中，我们使用以下代码检查`motion_sensor`是否有运动：

```py
if motion_sensor.motion_detected:
```

从这里开始，我们定义要打印到 shell 的消息。

# 使用 GPIO Zero 蜂鸣器类

通常，有两种类型的电子蜂鸣器：有源和无源。有源蜂鸣器具有内部振荡器，当直流（DC）施加到它时会发出声音。无源蜂鸣器需要交流（AC）才能发出声音。无源蜂鸣器基本上是小型电磁扬声器。区分它们的最简单方法是施加直流电源并听声音。对于我们的代码目的，我们将使用有源蜂鸣器，如下图所示：

![](img/9cabbcd7-9a5f-476a-a961-170e92323f84.png)

`GPIO Zero`库中有一个`buzzer`类。我们将使用这个类来生成有源蜂鸣器的刺耳警报声。按照以下图表配置电路。有源蜂鸣器的正极导线连接到 GPIO 引脚 17：

![](img/c1e46b62-c411-4348-beb9-266ae3e17aba.png)

在 Thonny 中创建一个新的 Python 文件，并将其命名为`buzzer-test1.py`。输入以下代码并运行：

```py
from gpiozero import Buzzer
from time import sleep

buzzer = Buzzer(17)

while True:
    buzzer.on()
    sleep(2)
    buzzer.off()
    sleep(2)
```

根据您选择的有源蜂鸣器，您应该听到持续两秒的刺耳声音，然后是 2 秒的静音。以下一行打开了蜂鸣器：

```py
buzzer.on()
```

同样，前面代码中的这行关闭了蜂鸣器：

```py
buzzer.off()
```

可以使用`buzzer`对象上的`toggle`方法简化代码。在 Thonny 中创建一个新的 Python 文件。将其命名为`buzzer-test2.py`。输入以下内容并运行：

```py
from gpiozero import Buzzer
from time import sleep

buzzer = Buzzer(17)

while True:
    buzzer.toggle()
    sleep(2)
```

您应该得到相同的结果。执行相同操作的第三种方法是使用`buzzer`对象的`beep`方法。在 Thonny 中创建一个新的 Python 文件。将其命名为`buzzer-test3.py`。输入以下内容并运行：

```py
from gpiozero import Buzzer

buzzer = Buzzer(17)

while True:
    buzzer.beep(2,2,10,False)
```

`buzzer`应该在`2`秒内打开，然后关闭`2`秒，重复进行`10`次。`beep`方法接受以下四个参数：

+   `on_time`：这是声音开启的秒数。默认值为`1`秒。

+   `off_time`：这是声音关闭的秒数。默认值为`1`秒。

+   `n`：这是进程运行的次数。默认值为`None`，表示永远。

+   `background`：这确定是否启动后台线程来运行进程。`True`值在后台线程中运行进程并立即返回。当设置为`False`时，直到进程完成才返回（请注意，当`n`为`None`时，该方法永远不会返回）。

# 构建一个基本的报警系统

现在让我们围绕蜂鸣器构建一个基本的报警系统。将 PIR 传感器连接到 GPIO 引脚 4，并将一个锁定按钮连接到 GPIO 引脚 8。以下是我们系统的电路图：

![](img/2227e962-05f3-42cd-8afd-f285c678cb38.png)

在 Thonny 中创建一个新文件，并将其命名为`basic-alarm-system.py`。输入以下内容，然后点击运行：

```py
from gpiozero import MotionSensor
from gpiozero import Buzzer
from gpiozero import Button
from time import sleep

buzzer = Buzzer(17)
motion_sensor = MotionSensor(4)
switch = Button(8)

while True:
    if switch.is_pressed:
        if motion_sensor.motion_detected:
            buzzer.beep(0.5,0.5, None, True)
            print('Intruder Alert')
            sleep(1)
        else:
            buzzer.off()
            print('Quiet')
            sleep(1)
    else:
        buzzer.off()
        sleep(1)
```

我们在这里所做的是使用我们的组件创建一个报警系统。我们使用一个锁定按钮来打开和关闭报警系统。我们可以很容易地用一个钥匙开关替换锁定按钮。以下图片显示了这个变化：

![](img/7850b80f-1614-4439-91b0-8dd9269c8ade.png)

这个电路可以很容易地转移到项目盒中，用作报警系统。

# 使用红外传感器修改 Hello LED

我们将通过修改我们最初的“Hello LED”代码来继续探索传感器。在这个项目中，我们将距离传感器与我们的 PIR 传感器相结合，并根据这些传感器的值闪烁 LED。这个电路不仅会告诉我们有人靠近，还会告诉我们他们有多近。

我们将在本节中涵盖以下内容：

+   配置距离传感器

+   将“Hello LED”提升到另一个水平

# 配置距离传感器

我们将从配置距离传感器和运行一些代码开始。以下是我们距离传感器电路的电路图：

![](img/b0566f96-1a30-491c-97be-90aff92585be.png)

需要进行以下连接：

+   来自运动传感器的 VCC 连接到树莓派的 5V 直流输出

+   树莓派的 GPIO 引脚 17 连接到距离传感器的 Trig

+   距离传感器上的回波连接到 330 欧姆电阻

+   距离传感器上的 GND 连接到树莓派上的 GND 和一个 470 欧姆电阻

+   来自距离传感器回波引脚的 330 欧姆电阻的另一端连接到 470 欧姆电阻（这两个电阻创建了一个电压分压电路）

+   来自树莓派的 GPIO 引脚 18 连接到电阻的交叉点

在这个电路中值得注意的是由两个电阻创建的电压分压器。我们使用这个分压器连接 GPIO 引脚 18。

在 Thonny 中创建一个新的 Python 文件，并将其命名为“distance-sensor-test.py”。输入以下代码并运行：

```py
from gpiozero import DistanceSensor
from time import sleep

distance_sensor = DistanceSensor(echo=18, trigger=17)
while True:
    print('Distance: ', distance_sensor.distance*100)
    sleep(1)
```

当您将手或其他物体放在距离传感器前时，Shell 中打印的值应该会发生变化，如下所示：

![](img/998c5ae2-fac9-47bf-80e5-508593e35f36.png)确保将距离传感器放在稳固的、不动的表面上，比如面包板。

# 将“Hello LED”提升到另一个水平

我们最初的“Hello LED！”系统是一个简单的电路，涉及制作一个 LED，连接到 GPIO 端口，闪烁开关。自从创建该电路以来，我们已经涵盖了更多内容。我们将利用我们所学到的知识创建一个新的“Hello LED”电路。通过这个电路，我们将创建一个报警系统，LED 的闪烁频率表示报警距离。

以下是我们新的“Hello LED”系统的电路图：

![](img/91600d51-e1a3-4ba4-8942-ac3b62b33e2e.png)

这可能看起来有点复杂，线路到处都是；然而，这是一个非常简单的电路。距离传感器部分与以前一样。对于其他组件，连接如下：

+   PIR 传感器的正极连接到面包板上的 5V 直流电源

+   PIR 传感器的负极连接到面包板上的 GND

+   PIR 传感器的信号引脚连接到 GPIO 引脚 4

+   LED 的正极通过 220 欧姆电阻连接到 GPIO 引脚 21

+   LED 的负极连接到面包板上的 GND

在 Thonny 中创建一个新的 Python 文件，并将其命名为“hello-led.py”。输入以下代码并运行：

```py
from gpiozero import DistanceSensor
from gpiozero import MotionSensor
from gpiozero import LED
from time import sleep

distance_sensor = DistanceSensor(echo=18, trigger=17)
motion_sensor = MotionSensor(4)
led = LED(21)

while True:  
    if(motion_sensor.motion_detected):
        blink_time=distance_sensor.distance
        led.blink(blink_time,blink_time,None,True)
    sleep(2)
```

LED 应该在检测到运动后立即开始闪烁。当您将手靠近距离传感器时，LED 的闪烁频率会加快。

# 总结

现在我们应该非常熟悉与传感器和树莓派的交互。这一章应该被视为使用我们的树莓派轻松创建感官电路的练习。

我们将在第九章中使用这些知识，*构建家庭安全仪表板*，在那里我们将创建一个物联网家庭安全仪表板。

# 问题

1.  主动蜂鸣器和被动蜂鸣器有什么区别？

1.  真或假？我们检查`button.is_pressed`参数来确认我们的按钮是否被按下。

1.  真或假？我们需要一个电压分压电路才能连接我们的 PIR 传感器。

1.  我们可以使用哪三种不同的方法让我们的主动蜂鸣器发出蜂鸣声？

1.  真或假？按键必须直接连接到电路才能发挥作用。

1.  我们使用哪个`DistanceSensor`参数来检查物体与距离传感器的距离？

1.  我们使用 Sense HAT 模拟器的哪个方法来将像素打印到屏幕上？

1.  我们如何设置我们的`MotionSensor`来从 GPIO 引脚 4 读取？

1.  真或假？基本的报警系统对于我们的树莓派来说太复杂了。

1.  真或假？Sense HAT 模拟器可以用来与连接到 GPIO 的外部传感器进行交互。

# 进一步阅读

请参阅 GPIO Zero 文档[`gpiozero.readthedocs.io/en/stable/`](https://gpiozero.readthedocs.io/en/stable/)，了解如何使用这个库的更多信息。
