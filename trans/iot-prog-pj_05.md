# 使用 Python 控制舵机

在数字技术兴起之前，模拟仪表和仪器是显示数据的唯一方式。一旦转向数字技术，模拟仪表就不再流行。在模拟时钟上学习报时的一代人可能会突然发现这项技能已经过时，因为数字显示时间已经成为常态。

在本章中，我们将通过根据数字值改变舵机的位置来弥合数字世界和模拟世界之间的差距。

本章将涵盖以下主题：

+   将舵机连接到树莓派

+   通过命令行控制舵机

+   编写一个 Python 程序来控制舵机

# 完成本章所需的知识

读者需要对 Python 编程语言有一定的了解才能完成本章。还必须了解使用简单的面包板连接组件。

# 项目概述

在这个项目中，我们将连接一个舵机和 LED，并使用`GPIO Zero`库来控制它。我们将首先在 Fritzing 中设计电路，然后进行组装。

我们将开始使用 Python shell 来控制舵机。

最后，我们将通过创建一个 Python 类来扩展这些知识，该类将根据传递给类的数字打开、关闭或闪烁 LED，并根据百分比量来转动舵机。

这个项目应该需要大约 2 个小时来完成。

# 入门

完成这个项目需要以下物品：

+   树莓派 3 型号（2015 年或更新型号）

+   USB 电源适配器

+   计算机显示器

+   USB 键盘

+   USB 鼠标

+   一个小型舵机

+   面包板

+   LED（任何颜色）

+   面包板的跳线

# 将舵机连接到树莓派

这个项目涉及将舵机连接到我们的树莓派。许多人将舵机与步进电机和直流电机混淆。让我们来看看这些类型的电机之间的区别。

# 步进电机

步进电机是无刷直流电动机，可以移动等步长的完整旋转。电机的位置是在没有使用反馈系统（开环系统）的情况下控制的。这使得步进电机相对廉价，并且在机器人、3D 打印机和数控机床等应用中很受欢迎。

以下是步进电机内部工作的粗略图示：

![](img/35ada15a-697a-46e7-9680-659725d49243.jpg)

通过按顺序打开和关闭线圈 A 和 B，可以旋转连接到电机轴的永磁体。使用精确的步骤，可以精确控制电机，因为步数可以轻松控制。

步进电机往往比其他类型的小型电机更重更笨重。

以下照片显示了 3D 打印机中使用的典型步进电机：

![](img/29d7c3f6-9842-4328-aa1e-6d1fac677db1.png)

# 直流电机

直流电机与步进电机类似，但不会将运动分成相等的步骤。它们是最早被广泛使用的电动机，并且在电动汽车、电梯和任何不需要精确控制电机位置的应用中使用。直流电机可以是刷式或无刷的。

刷式电机操作起来更简单，但在每分钟转数（RPM）和使用寿命上有限制。无刷电机更复杂，需要电子控制，例如一些无人机上使用的电子调速器（ESC）。无刷电机可以以更高的转速运行，并且比刷式电机有更长的使用寿命。

直流电机的响应时间比步进电机短得多，并且比可比较的步进电机更轻。

以下是典型的小型刷式直流电机的照片：

![](img/a1a836f9-1ba8-4389-a264-e9334868bcba.png)

# 舵机

舵机利用闭环反馈机制来提供对电机位置的极其精确的控制。它们被认为是步进电机的高性能替代品。范围可以根据舵机的不同而变化，有些舵机限制在 180 度运动，而其他舵机可以运动 360 度。

闭环控制系统与开环控制系统不同，它通过测量输出的实际条件并将其与期望的结果进行比较来维持输出。闭环控制系统通常被称为反馈控制系统，因为正是这种反馈被用来调整条件。

舵机的角度由传递到舵机控制引脚的脉冲决定。不同品牌的舵机具有不同的最大和最小值，以确定舵机指针的角度。

以下是一个图表，用于演示**脉冲宽度调制**（**PWM**）与 180 度舵机位置之间的关系：

![](img/2f396a7b-6ff8-4927-b78c-9b6e8e46d669.jpg)

以下是我们将在电路中使用的小型舵机的照片。我们可以直接将这个舵机连接到我们的树莓派（较大的舵机可能无法实现）：

![](img/f4c4ea4c-ac2d-443e-95c3-adeb4af4fd75.png)

以下是舵机颜色代码的图表：

![](img/d0ff3d63-d318-408e-aa5d-f8b7c7d0e08e.png)

# 将舵机连接到我们的树莓派

我们的电路将由一个简单的舵机和 LED 组成。

以下是电路的 Fritzing 图：

![](img/51200b9c-1606-418a-b96b-dfc27c026afe.png)

我们连接：

+   舵机的正电源到 5V 直流电源，地到 GND

+   从舵机到 GPIO 17 的控制信号

+   LED 的正极连接到 GPIO 14，电阻连接到 GND

确保使用小型舵机，因为较大的舵机可能需要比树莓派能够提供的更多电力。电路应该类似于以下内容：

![](img/02e23687-6fd6-40d8-9f3c-a9d313002c39.png)

# 通过命令行控制舵机

现在我们的舵机已连接到树莓派，让我们在命令行中编写一些代码来控制它。我们将使用树莓派 Python 库`GPIO Zero`来实现这一点。

加载 Thonny 并点击 Shell：

![](img/60473ee5-7d07-4f6d-a96d-697fe870bd5b.png)

在 Shell 中输入以下内容：

```py
from gpiozero import Servo
```

短暂延迟后，光标应该返回。我们在这里所做的是将`gpiozero`中的`servo`对象加载到内存中。我们将使用以下语句为引脚 GPIO `17`分配：

```py
servo = Servo(17)
```

现在，我们将舵机移动到最小（`min`）位置。在命令行中输入以下内容：

```py
servo.min()
```

你应该听到舵机在移动，指针将移动到最远的位置（如果它还没有在那里）。

使用以下命令将舵机移动到最大（`max`）位置：

```py
servo.max()
```

现在，使用以下命令将舵机移动到中间（`mid`）位置：

```py
servo.mid()
```

舵机应该移动到其中间位置。

当你把手放在舵机上时，你可能会感到轻微的抽搐运动。要暂时禁用对舵机的控制，请在命令行中输入以下内容并按 Enter 键：

```py
servo.detach()
```

抽搐运动应该停止，附在舵机上的指针指示器应该保持在当前位置。

正如我们所看到的，很容易将舵机移动到其最小、中间和最大值。但是如果我们想要更精确地控制舵机怎么办？在这种情况下，我们可以使用`servo`对象的 value 属性。可以使用介于`-1`（最小）和`1`（最大）之间的值来移动舵机。

在命令行中输入以下内容：

```py
servo.value=-1
```

`servo`应该移动到其最小位置。现在，输入以下内容：

```py
servo.value=1
```

`servo`现在应该移动到其最大位置。让我们使用 value 属性来指示天气条件。在命令行中输入以下内容：

```py
weather_conditions = {'cloudy':-1, 'partly cloudy':-0.5, 'partly sunny': 0.5, 'sunny':1}
```

在 Shell 中使用以下代码进行测试：

```py
weather_conditions['partly cloudy']
```

你应该在 Shell 中看到以下内容：

```py
-0.5
```

有了我们的`servo`对象和我们的`weather_conditions`字典，我们现在可以使用伺服电机来物理地指示天气条件。在 shell 中输入以下内容：

```py
servo.value = weather_conditions['cloudy']
```

伺服电机应该移动到最小位置，以指示天气条件为“多云”。现在，让我们尝试“晴朗”：

```py
servo.value = weather_conditions['sunny']
```

伺服应该移动到最大位置，以指示“晴朗”的天气条件。

对于“局部多云”和“局部晴朗”的条件，使用以下内容：

```py
servo.value = weather_conditions['partly cloudy']
```

```py
servo.value = weather_conditions['partly sunny']
```

# 编写一个 Python 程序来控制伺服

杰瑞·塞范菲尔德曾开玩笑说，我们需要知道天气的全部信息就是：我们是否需要带上外套？在本章和下一章的其余部分中，我们将建立一个模拟仪表针仪表板，以指示天气条件所需的服装。

我们还将添加一个 LED，用于指示需要雨伞，并闪烁以指示非常恶劣的风暴。

在我们可以在第六章中构建仪表板之前，我们需要代码来控制伺服和 LED。我们将首先创建一个类来实现这一点。

这个类将在我们的电路上设置伺服位置和 LED 状态：

1.  从应用程序菜单 | 编程 | Thonny Python IDE 打开 Thonny

1.  单击新图标创建一个新文件

1.  输入以下内容：

```py
from gpiozero import Servo
from gpiozero import LED

class WeatherDashboard:

    servo_pin = 17
    led_pin = 14

    def __init__(self, servo_position=0, led_status=0):      
        self.servo = Servo(self.servo_pin)
        self.led = LED(self.led_pin)      
        self.move_servo(servo_position)
        self.set_led_status(led_status)

    def move_servo(self, servo_position=0): 
        self.servo.value=self.convert_percentage_to_integer
        (servo_position)

    def set_led_status(self, led_status=0):       
        if(led_status==0):
            self.led.off()
        elif (led_status==1):
            self.led.on()
        else:
            self.led.blink()

    def convert_percentage_to_integer(self, percentage_amount):
        return (percentage_amount*0.02)-1

if __name__=="__main__":
    weather_dashboard = WeatherDashboard(50, 1)
```

1.  将文件保存为`WeatherDashboard.py`

1.  运行代码

1.  您应该看到伺服移动到中间位置，LED 应该打开

尝试其他值，看看是否可以将伺服移动到 75%并使 LED 闪烁。

让我们来看看代码。在定义类之后，我们使用以下内容为伺服和 LED 设置了 GPIO 引脚值：

```py
servo_pin = 17
led_pin = 14
```

正如您在我们建立的电路中看到的那样，我们将伺服和 LED 分别连接到 GPIO`17`和 GPIO`14`。GPIO Zero 允许我们轻松地分配 GPIO 值，而无需样板代码。

在我们的类初始化方法中，我们分别创建了名为`servo`和`led`的`Servo`和`LED`对象：

```py
self.servo = Servo(self.servo_pin)
self.led = LED(self.led_pin) 
```

从这里开始，我们调用我们类中移动伺服和设置 LED 的方法。让我们看看第一个方法：

```py
def move_servo(self, servo_position=0): 
        self.servo.value=self.convert_percentage_to_integer
        (servo_position)
```

在这个方法中，我们只需设置`servo`对象中的值属性。由于此属性仅接受从`-1`到`1`的值，而我们传递的值是从`0`到`100`，因此我们需要将我们的`servo_position`进行转换。我们使用以下方法来实现这一点：

```py
def convert_percentage_to_integer(self, percentage_amount):
    return (percentage_amount*0.02)-1
```

为了将百分比值转换为`-1`到`1`的比例值，我们将百分比值乘以`0.02`，然后减去`1`。通过使用百分比值为`50`来验证这个数学问题是很容易的。值为`50`代表了`0`到`100`比例中的中间值。将`50`乘以`0.02`得到了值为`1`。从这个值中减去`1`得到了`0`，这是`-1`到`1`比例中的中间值。

要设置 LED 的状态（关闭、打开或闪烁），我们从初始化方法中调用以下方法：

```py
def set_led_status(self, led_status=0):       
    if(led_status==0):
        self.led.off()
    elif (led_status==1):
        self.led.on()
    else:
        self.led.blink()
```

在`set_led_status`中，如果传入的值为`0`，我们将 LED 设置为“关闭”，如果值为`1`，我们将其设置为“打开”，如果是其他值，我们将其设置为“闪烁”。

我们用以下代码测试我们的类：

```py
if __name__=="__main__":
    weather_dashboard = WeatherDashboard(50, 1)
```

在第六章中，*使用伺服控制代码控制模拟设备*，我们将使用这个类来构建我们的模拟天气仪表板。

# 总结

正如我们所看到的，使用树莓派轻松地将数字世界和模拟世界之间的差距进行数据显示。其 GPIO 端口允许轻松连接各种输出设备，如电机和 LED。

在本章中，我们连接了一个伺服电机和 LED，并使用 Python 代码对它们进行了控制。我们将在第六章中扩展这一点，使用伺服控制代码来控制模拟设备，构建一个带有模拟仪表盘显示的物联网天气仪表板。

# 问题

1.  真还是假？步进电机是使用开环反馈系统控制的。

1.  如果您要建造一辆电动汽车，您会使用什么类型的电动机？

1.  真或假？舵机被认为是步进电机的高性能替代品。

1.  是什么控制了舵机的角度？

1.  真或假？直流电机的响应时间比步进电机短。

1.  我们使用哪个 Python 包来控制我们的舵机？

1.  真或假？我们能够在 Thonny 的 Python shell 中控制舵机。

1.  用什么命令将舵机移动到最大位置？

1.  真或假？我们只能将舵机移动到最小、最大和中间位置。

1.  我们如何将百分比值转换为代码中`servo`对象理解的相应值？

# 进一步阅读

`GPIO Zero`文档提供了对这个令人惊叹的树莓派 Python 库的完整概述。了解更多信息，请访问[`gpiozero.readthedocs.io/en/stable/`](https://gpiozero.readthedocs.io/en/stable/)。
