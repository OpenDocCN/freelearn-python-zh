# 第二十五章：用贾维斯识别人类

到目前为止，我们已经在上一章中了解到如何将多层条件组合在一起以获得所需的功能。我们刚刚完成了让贾维斯为您工作的第一步。现在，是时候让它变得更加强大了。

在本章中，我们将使其控制更多您家中的电子设备，这些设备可以在您没有告诉系统任何内容的情况下自主控制。所以，不要拖延，让我们直接进入并看看我们的收获。

# 打开灯，贾维斯

智能家居的基本功能之一是在您附近时为您打开灯光。这是任何系统可以为您做的最基本的事情之一。我们将从您进入房间时打开灯光开始，然后我们将使系统变得更加智能。

因此，我们需要做的第一件事是识别您是否在房间里。有多种方法可以做到这一点。生活的一个重要特征就是运动的存在。您可能会说植物不会移动，但它们会生长，不是吗？因此，检测运动可能是检测某人是否在场的关键步骤！

这一步对您来说并不那么困难，因为我们之前已经接口化了这个传感器。我们说的是老式的 PIR 传感器。因此，传感器将感知区域内的任何运动。如果有任何运动，那么贾维斯将打开灯光。我相信这是您现在可以自己做到的事情。您仍然可以参考这里的代码和电路图：

![](img/d18f3ec7-4d2a-452c-bf98-a522f1c87325.png)

现在上传以下代码：

```py
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
PIR = 24
LIGHT = 23
GPIO.setup(DOPPLER,GPIO.IN)
GPIO.setup(BUZZER,GPIO.OUT)
While True:
   if GPIO.input(PIR) == 1:
       GPIO.output(LIGHT,GPIO.HIGH)
   if GPIO.input(PIR) == 0:
       GPIO.output(LIGHT,GPIO.LOW)
```

在上述代码中，我们只是在检测到运动时立即打开灯光，但问题是它只会在有运动的时候打开灯光。这是什么意思？简单来说，只要有一些运动，灯就会保持开启，一旦运动停止，灯就会关闭。

对于想要减肥的人来说，这可能是一个很好的代码，但对于我们大多数人来说，这将是令人讨厌的。因此，让我们包含一个小循环，我们在上一章中使用过，并使其变得更好一些：

```py
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

PIR = 24
LIGHT = 23
TIME = 5

GPIO.setup(PIR,GPIO.IN)
GPIO.setup(BUZZER,GPIO.OUT)

While True:

   If GPIO.input(PIR) == 1:

       M = datetime.datetime.now().strftime('%M')
       M_final= M + TIME 
       for M < M_final:

         GPIO.output(LIGHT,GPIO.HIGH)
         M = datetime.datetime.now().strftime('%M')

         if GPIO.input(PIR) == 1:
            M_final = M_final + 1 if GPIO.input(PIR) = 0:

        GPIO.output(LIGHT, GPIO.LOW)} 
```

因此，在这个程序中，我们所做的就是添加了一个`for`循环，它会在设定的时间内打开灯光。这段时间有多长可以通过改变变量`TIME`的值来切换。

在那个循环中还有一个有趣的部分，如下所示：

```py
 if GPIO.input(PIR) == 1
            M_final = M_final + 1 
```

你可能会想为什么我们要这样做？每当灯光被打开时，它将保持开启 5 分钟。然后，它将关闭并等待运动发生。因此，基本上，这段代码的问题是，如果您在房间里，灯光打开后，它将在 5 分钟内查看是否有运动被检测到。有可能在 5 分钟后寻找运动时您正在运动。但大多数情况下，这不会发生。因此，我们使用 PIR 传感器来检测运动。每当检测到运动时，通过`M_final = M_final + 1`这一行来增加`M_final`的值，从而延长灯光打开的时间。

# 理解运动

到目前为止，您一定已经意识到 PIR 传感器并不是我们打开或关闭灯光的最理想传感器。主要是因为，尽管运动是存在的最佳指标之一，但有时您可能根本不会移动，例如休息、阅读书籍、观看电影等。

现在我们该怎么办？嗯，我们可以做一个小技巧。还记得在上一章中我们使用我们的接近传感器来感知一个人是否穿过了特定区域吗？我们将在这里植入类似的逻辑；但不只是简单地复制粘贴代码，我们将改进它，使其变得更好。

因此，我们将使用两个红外接近传感器，而不是使用一个。安装如下图所示：

![](img/ec73b93a-776b-4d21-b548-77a79d33417a.png)

现在很明显，每当有人从门边走进房间边时，**传感器 1**在检测到人体时会显示较低的读数。然后，当他朝房间一侧走去时，**传感器 2**将显示类似的读数。

如果首先触发**传感器 1**，然后触发**传感器 2**，那么我们可以安全地假设这个人是从门边走向房间边。同样，如果相反发生，那么可以理解这个人是从房间里走出去。

现在，这相当简单。但是我们如何在现实生活中实现它呢？首先，我们需要按以下方式连接电路：

![](img/f26bc00c-738b-49c4-b2d2-9ab5ea6176f3.png)

一旦完成，上传以下代码：

```py
import GPIO library
import RPi.GPIO as GPIO
import time

import Adafruit_ADS1x15 adc0 = Adafruit_ADS1x15.ADS1115()   GAIN = 1
LIGHT = 23 adc0.start_adc(0, gain=GAIN) adc1.start_adc(1, gain=GAIN)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

while True:

 F_value = adc0.get_last_result()  F1 = (1.0  / (F_value /  13.15)) -  0.35

   time.sleep(0.1)

 F_value = adc0.get_last_result()  F2 = (1.0  / (F_value /  13.15)) -  0.35

   F0_final = F1-F2

   if F0 > 10 :

        Time0 =  time.time()

 F_value = adc1.get_last_result()  F1 = (1.0  / (F_value /  13.15)) -  0.35

   time.sleep(0.1)

 F_value = adc1.get_last_result()  F2 = (1.0  / (F_value /  13.15)) -  0.35

   F1_final = F1-F2

   if F1 > 10: 

 Time1 =  time.time()

    if Time1 > Time0:

        GPIO.output(LIGHT, GPIO.HIGH)

    if Time1 < Time0:

        GPIO.output(LIGHT, GPIO.LOW)      }
```

现在，让我们看看我们在这里做了什么。和往常一样，大部分语法都非常简单明了。最重要的部分是逻辑。因此，让我们逐步了解我们在做什么。

```py
 F_value = adc0.get_last_result()  F1 = (1.0  / (F_value /  13.15)) -  0.35

   time.sleep(0.1)

 F_value = adc0.get_last_result()  F2 = (1.0  / (F_value /  13.15)) -  0.35
```

在上面的代码行中，我们正在获取红外接近传感器的值，并计算相应的距离，将该值存储在一个名为`F1`的变量中。一旦完成，我们将使用`time.sleep(0.1)`函数停止 0.1 秒。然后，我们再次从同一传感器读取并将值存储在名为`F2`的变量中。为什么我们要这样做？我们在之前的章节中已经理解了。

```py
 F0_final = F1-F2
```

一旦获得了`F1`和`F0`的值，我们将计算差值以找出是否有人通过。如果没有人通过，那么读数几乎相同，差异不会很大。但是，如果有人通过，那么读数将是相当大的，并且该值将存储在一个名为`F0_final`的变量中。

```py
 if F0 > 10 :

        Time0 =  time.time()
```

如果`F0`的值或第一次和第二次读数之间的距离差大于 10 厘米，则`if`条件将为真。一旦为真，它将将`Time0`变量的值设置为当前时间值。`time.time()`函数将记录下确切的时间。

```py
 F_value = adc1.get_last_result()  F1 = (1.0  / (F_value /  13.15)) -  0.35

   time.sleep(0.1)

 F_value = adc1.get_last_result()  F2 = (1.0  / (F_value /  13.15)) -  0.35 
```

```py
 F1_final = F1-F2   if F1 > 10: 

 Time1 =  time.time()
```

现在，我们将对**传感器 2**执行完全相同的步骤。这里没有什么新的要告诉的；一切都很简单明了。

```py
    if Time1 > Time0:

        GPIO.output(LIGHT, GPIO.HIGH)
```

一旦所有这些都完成了，我们比较`Time1 > Time0`。为什么我们要比较呢？因为`Time0`是**传感器 1**的记录时间。如果人在里面移动，那么**传感器 1**将首先被触发，然后**传感器 2**将被触发。因此，**传感器 2**的记录时间会更长，相对于**传感器 1**来说更早。如果发生这种情况，那么我们可以假设人正在进来。如果有人进来，我们只需要打开灯，这正是我们在这里要做的。

```py
    if Time1 < Time0:

        GPIO.output(LIGHT, GPIO.LOW)
```

同样，当一个人走出去时，首先触发的传感器将是**传感器 2**，然后将触发**传感器 1**。使得记录在`Time1`中的时间比`Time2`更早；因此，每当这个条件为真时，我们就会知道这个人正在离开房间，灯可以关闭。

继续安装在门附近，看看它的反应。我相信这将比我们之前通过 PIR 做的要好得多。玩得开心，并尝试找出它可能存在的任何缺陷。

# 完善运动

你能在以前的代码中找到任何缺陷吗？它们并不难找到；当房间里只有一个人时，代码运行得很好。但是如果安装在有多人出入的地方，可能会有挑战。这是因为每当有人走出去时，灯就会熄灭。

现在问题显而易见，是时候让代码变得更加智能了。为了做到这一点，硬件将保持完全相同；我们只需要让代码更加智能。让我们看看我们可以如何做到：

```py
import GPIO library
   import RPi.GPIO as GPIO
   import time
   import time
   import Adafruit_ADS1x15
   adc0 = Adafruit_ADS1x15.ADS1115()
GAIN = 1
 adc0.start_adc(0, gain=GAIN)
adc1.start_adc(1, gain=GAIN)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
PCount = 0
while True:
   F_value = adc0.get_last_result()
   F1 = (1.0 / (F_value / 13.15)) - 0.35
   time.sleep(0.1)
   F_value = adc0.get_last_result()
   F2 = (1.0 / (F_value / 13.15)) - 0.35
   F0_final = F1-F2
   if F0 > 10 :
        Time0 = time.time()
   F_value = adc1.get_last_result()
   F1 = (1.0 / (F_value / 13.15)) - 0.35
   time.sleep(0.1)
   F_value = adc1.get_last_result()
   F2 = (1.0 / (F_value / 13.15)) - 0.35
   F1_final = F1-F2
   if F1 > 10:
        Time1 = time.time()
    if Time1 > Time0:
        PCount = PCount + 1
    if Time1 < Time0:
        PCount = PCount - 1

if PCount > 0:

           GPIO.output(LIGHT, GPIO.HIGH)
       else if PCount = 0:
          GPIO.output(LIGHT, GPIO.LOW)        
```

我们所做的是非常基础的。我们声明了一个名为`PCount`的变量。这个变量被声明为计算房间或家里的人数。正如你在代码的前几行中所看到的，我们声明了`PCount`的值为`0`。我们假设一旦我们开始，房间内的人数将为`0`。

```py
    if Time1 > Time0:

        PCount = PCount + 1
```

每当条件`if Time1 > Time0:`满足时，`PCount`的值就会增加`1`。众所周知，只有当有人在房子里走动时，条件才会成立。

```py
    if Time1 < Time0:

        PCount = PCount - 1
```

同样，当一个人在外面走的时候，条件`if Time1 < Time0:`是真的；每当这种情况发生时，`PCount`的值就会减少`1`。

```py
    if PCount > 0:

       GPIO.output(LIGHT, GPIO.HIGH)
```

现在我们已经开始计算房间内的人数，我们现在应用条件，如果`PCount`的数量大于`0`，则会打开。因此，当房屋内的人数大于`0`时，灯将亮起。

```py
    else if PCount = 0:

       GPIO.output(LIGHT, GPIO.LOW)
```

以非常相似的方式，如果`PCount`的值或者房屋内的人数达到`0`，灯将被关闭。

因此，完美！

# 控制强度

我们现在已经控制了很多灯。现在是时候控制我们的风扇和其他空气循环系统了。每当我们谈论风扇或任何其他空气循环设备时，本质上我们在谈论电机。正如我们之前学到的，电机是简单的设备，可以使用电机驱动器非常容易地进行控制。但是你知道，当时我们控制的是直流电机。直流电机是非常简单的设备。但是当我们谈论我们的家用电器时，那么大多数这些设备将使用交流电或交流电流。我假设你一定知道那是什么，以及它与直流电的区别。

现在你知道我们家用的电机是交流电机，你也必须考虑到他们的控制机制将与直流电机大不相同。如果你这样想，你是对的。然而，电子产品的好处是，没有什么真的困难或复杂。基本原理基本上是一样的。所以，让我们看看如何在交流电源中控制电机的速度。

正如我们之前所见，我们可以简单地给直流电机一个 PWM 信号，电机将以 PWM 信号的平均电压速度运行。现在，你一定在想，这也可以应用于交流。事实是，是的，如果你想控制灯或类似设备，这是可以做到的，这些设备在波形失真的情况下没有任何主要特性变化。然而，当我们谈论其他组件时，我们遇到了一个大问题。交流波形看起来像这样：

![](img/e6c88285-4f38-493f-8a73-ab5d0a621507.png)

这基本上意味着电位定期变化。在大多数家庭中，这是每秒 50 次。现在，想象一下，如果我们有一个 PWM 控制的设备，它在特定间隔开关电路，只允许电源通过。然后，正弦波的不同部分将传递到最终输出。

![](img/d3c74564-5890-4819-8bd8-c37da64c8c2c.png)

正如你在前面的 PWM 中所看到的，幸运的是 PWM 信号与交流电源的相位匹配；然而，由于这个原因，只有相位的正端被传输到最终输出，而不是负端。这将给我们的负载造成严重问题，有很大的机会连接的设备将无法工作。

![](img/3d28446a-d7a4-4893-85e7-87dfa8a9a994.png)

我们还有另一个例子，其中 PWM 是随机的，它让波的随机部分通过。在这种情况下，我们可以清楚地看到随机地传输波的任何部分，正负端电压不同步，这将是一个巨大的问题。因此，我们不使用 PWM，而是使用一些非常有趣的东西。

最常用的方法称为**相位触发控制**。有时也称为相角控制或相位切割。它的本质是在相位的某些部分切割波，让其余的波通过。困惑吗？让我在这里给你展示：

![](img/047edc17-d0c2-4d67-9659-cf416b3bb5af.png)

现在，正如你所看到的，交流波的后半部分的相位被切割了，没有传递到最终输出。这使得最终输出只有总输入的 50%。这种技术的作用是，在减小总体输出电压的同时，保持电源的交流特性。同样，如下图所示，波在已经传递了 75%后被切割。这导致输出相对较低：

![](img/3b77b2b6-78b7-4683-b2c7-eea349d43a3c.png)

现在你可能会问，我们到底是如何做到这一点的？这是通过一个相对复杂的电路来完成的，它检测波的相位角，然后打开或控制一个双向高功率半导体晶闸管。这导致电源在某些相位通过或停止。我们将把这个电路的确切工作留到下一次，因为它相当复杂，与本书无关。

现在来到基本点，我们知道相位切割是什么，我们也知道晶闸管是让我们做到这一点的基本设备。但如何使用树莓派来实现这一点是个问题。

首先，我们需要一个交流调光模块。这个模块已经具备了相位检测和切割的所有组件。所以我们需要做的就是简单地使用 PWM 来控制它。

虽然我可能不需要演示如何连接电路或代码应该是什么，但为了理解起见，让我们使用这个模块将灯泡连接到我们的 Arduino，然后控制灯泡。现在，首先要记住的是负载应该是灯泡，而不是其他任何东西，比如 LED 灯。所以继续按照下图所示连接电路：

![](img/26e340ef-6b20-41d1-9951-8d8bd3b1850c.png)

完成后，上传以下代码：

```py
import RPi.GPIO as GPIO
import time                             
GPIO.setmode(GPIO.BCM)       
GPIO.setup(18,GPIO.OUT)         
I = 0
pwm= GPIO.PWM(18,50)

for I < 100:

    I = I+1
    pwm.start(I)
    time.sleep(0.1)

GPIO.cleanup()}
```

预期的是，连接的灯将首先微弱发光，然后逐渐增加强度，直到达到 100%。控制这样一个复杂的过程是如此简单。

# 智能温度控制

现在基础知识已经掌握，让我们继续使用这个系统构建有意义的东西。将空调设置到完美的温度是不是很困难？无论你做什么，最终都感觉不是最舒适的位置。这是由于身体在一天中温度的生理变化所致。

当你醒来时，你的体温相对较低。它比正常体温低多达 1°F。随着一天的进展，体温会上升，直到你上床睡觉。一旦你入睡，你的体温又开始下降，直到早上 4:00-6:00 达到最低点。这就是为什么当你上床睡觉时感觉温暖，但醒来时可能会感觉很冷的原因。现代空调有一个叫做睡眠模式的功能。它的作用是通过整个夜晚逐渐提高温度，这样你在任何时候都不会感到寒冷。但它的工作效果如何也是一个问题。

现在我们对机器人技术非常了解，我们将继续制作一个系统，来照顾一切。

在这部分，我们将空调和风扇连接在一起，这样它们可以一起工作，让你睡得更好。现在，在直接开始之前，我想让你看一下继电器上标明的额定值。正如你所看到的，继电器只能处理 250V 和 5 安培。现在，如果你查看空调的宣传册，你很容易就能明白我为什么要向你展示所有这些。空调的功耗将远远高于你的继电器所能承受的。因此，如果你尝试使用普通继电器来运行空调，那么你肯定会把继电器烧坏。你的电器可能的电流等级低于你的继电器。但是对于任何带有电机的设备，要记住该设备的初始功耗远高于额定功耗。因此，如果你的空调需要额定 10 安培，那么起动负载可能高达 15 安培。你可能会想，这不是问题，为什么我们不购买一个额定更高的继电器呢。好吧，正确！这正是我们将要做的。但是，电子设备的命名有时可能会很棘手。处理更高功率更高电压的电机开关设备通常被称为接触器，而不是继电器。从技术上讲，它们有相同的工作原理；然而，在这一点上的构造差异，这不是我们关心的问题。因此，我们将使用接触器来控制空调开关和调速器来控制风扇速度。既然这一点已经澄清，让我们继续并按照以下图表连接硬件：

![](img/44f748e3-7a93-46e6-afa7-765a6447d0ad.png)

```py
import RPi.GPIO as GPIO import time import Adafruit_DHT GPIO.setmode(GPIO.BCM) FAN = 18
AC = 17 pwm= GPIO.PWM(18,50)  GPIO.setup(FAN,GPIO.OUT) GPIO.setup(AC, GPIO.OUT)   while True: humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)

    if temperature =>20 && temperature <=30: Duty = 50 + ((temperature-25)*10)
  pwm.start(Duty)

    if temperature <22 :

         GPIO.output(AC, GPIO.LOW)

    if temperature >= 24

         GPIO.output(AC, GPIO.HIGH)}

```

这里使用的逻辑非常基本。让我们看看它在做什么：

```py
 humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)

    if temperature =>20 && temperature <=30: Duty = 50 + ((temperature-25)*10)
  pwm.start(Duty)
```

在这里，我们获取了`湿度`和`温度`的值。到目前为止一切都很好，但我们能否更进一步，使它变得更智能？以前的逻辑可能已经帮助你睡得更好，但我们能否让它对你来说更加完美？

我们身体中有多个指标可以让我们了解身体的状态。例如，如果你累了，你可能不会走得很快或者说得很大声。相反，你会做相反的事情！同样，有多个因素表明我们的睡眠周期是如何进行的。

其中一些因素是：体温、呼吸频率、快速动眼期睡眠和身体运动。测量准确的体温或呼吸频率和快速动眼期睡眠是一项挑战。但是当我们谈论身体运动时，我认为我们已经完善了。因此，基于身体运动，我们将感知我们的睡眠质量以及需要进行何种温度调节。

如果你注意到，每当有人睡觉并开始感到冷时，身体会呈胎儿姿势并且动作会少得多。这是自动发生的。然而，当一个人感到舒适时，会有一些不可避免的动作，比如翻身和手臂或腿部的运动。当一个人感到冷时，这是不会发生的。因此，通过这些动作，我们可以判断一个人是否感到冷。现在我们已经了解了身体的生理变化，让我们尝试围绕它构建一个程序，看看我们能实现什么。

为了做到这一点，首先，我们需要按照以下方式连接电路：

![](img/48c7ea27-075b-4caa-9601-797e8dc31680.png)

完成这些后，继续编写以下代码：

```py
import RPi.GPIO as GPIO import time import Adafruit_DHT GPIO.setmode(GPIO.BCM) FAN = 18
AC = 17
PIR = 22 PIN = 11
Sensor = 4 pwm= GPIO.PWM(18,50)  GPIO.setup(FAN,GPIO.OUT) GPIO.setup(AC, GPIO.OUT)   while True: humidity, temperature = Adafruit_DHT.read_retry(sensor, pin) H = datetime.datetime.now().strftime('%H') 
M = datetime.datetime.now().strftime('%M')

    if H <= 6 && H <= 22:

        if M <=58 : M = datetime.datetime.now().strftime('%M') humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
 if GPIO.input(PIR) == 0 :

                Movement = Movement + 1
                time.sleep(10)

           if temperature < 28: if Movement > 5 :

                    Duty = Duty + 10 pwm.start(Duty)
                    Movement = 0     

        if M = 59 : 

            if Movement = 0 :

                Duty = Duty -10
                pwm.start(Duty)

            Movement = 0

        if temperature <22 :

           GPIO.output(AC, GPIO.LOW)

       if temperature >= 24 && H <= 6 && H >= 22:

           GPIO.output(AC, GPIO.HIGH)

        if temperature > 27

            pwm.start(100)

    for H > 7 && H < 20 

        GPIO.output(AC, GPIO.LOW)

    if H = 20 

        GPIO.output(AC,GPIO.HIGH)
  }
```

让我们来看看引擎盖下面发生了什么：

```py
 if H <= 6 && H <= 22:

        if M <=58 : M = datetime.datetime.now().strftime('%M') humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
```

你会看到的第一件事是我们有一个条件：`if H,= 6 && H<= 22:`。只有在时间范围在上午 10 点到晚上 6 点之间时，这个条件才会成立。这是因为这是我们通常睡觉的时间。因此，在这个条件下的逻辑只有在睡觉的时候才会起作用。

第二个条件是`如果 M <= 58`，只有当时间在`0`和`58`分钟之间时才为真。因此，当时间为`M = 59`时，这个条件将不起作用。我们将看到为什么要有这个逻辑的原因。

此后，我们正在计算时间并将值存储在一个名为`M`的变量中。我们还在计算湿度和温度值，并将其存储在名为`temperature`和`humidity`的变量中：

```py
 if GPIO.input(PIR) == 0 :

                Movement = Movement + 1
                time.sleep(10) 
```

现在，在这一行中，我们正在实施一个条件，如果从 PIR 读取到的值很高，那么条件将为真。也就是说，会检测到一些运动。每当这种情况发生时，`Movement`变量将增加`1`。最后，我们使用`time.sleep(10)`函数等待`10`秒。这是因为 PIR 可能会在短暂的时间内保持高电平。在这种情况下，条件将一遍又一遍地为真，从而多次增加`Movement`的值。

我们增加`Movement`的值的目的是为了计算人移动的次数。因此，在一个时间内多次增加它将违背这个目标。

```py
 if temperature < 28: if Movement > 5 :

                    Duty = Duty + 10 pwm.start(Duty)
                    Movement = 0
```

现在我们有另一个条件，即`如果温度<28`。对于条件何时为真，不需要太多解释。因此，每当条件为真，如果计数的`Movement`次数超过`5`，那么`Duty`的值将增加`10`。因此，我们将 PWM 发送到空调调光器，从而增加风扇的速度。最后，我们将`Movement`的值重置为`0`。

因此，我们只是在计算移动次数。只有当温度低于 28°C 时才计算这一移动。如果移动次数超过`5`，那么我们将增加风扇速度 10%。

```py
        if M = 59 : 

            if Movement = 0 :

                Duty = Duty -10
                pwm.start(Duty)

            Movement = 0
```

在前一节中，逻辑只有在时间在`0`和`58`之间时才有效，也就是计数将发生的时间。当`M`的值为`59`时，那么条件`if Movement = 0`将被检查，如果为真，那么`Duty`的值将减少`10`。这将减慢风扇的速度 10%。此外，一旦执行了这个条件，`Movement`的值将被重置为`0`。因此，下一个小时可以开始一个新的循环。

基本上，这意味着计数将以小时为单位进行。如果`Movement`超过`5`，那么`Duty`的值将立即增加。但是，如果不是这种情况，程序将等待直到分钟接近`59`的值，每当发生这种情况时，它将检查是否有任何运动，如果有，风扇速度将降低。

```py
        if temperature <22 :

           GPIO.output(AC, GPIO.LOW)

        if temperature >= 24 && H <= 6 && H >= 22: 

           GPIO.output(AC, GPIO.HIGH)

        if temperature > 27

            pwm.start(100)
```

所有这些代码都非常直接。如果温度低于`22`，则空调将关闭。此外，如果温度等于或超过`24`，并且时间在晚上 10:00 到早上 6:00 之间，则空调将打开。最后，如果温度超过`27`，则风扇将以 100%的速度打开。

```py
    for H > 7 && H < 20 

        GPIO.output(AC, GPIO.LOW)

    if H = 20 

        GPIO.output(AC,GPIO.HIGH)
```

最后，我们通过使用条件`for H > 7 && H <20`来确保在这段时间内空调始终处于关闭状态。此外，如果`H = 20`，则应打开空调，以便在准备睡觉之前冷却房间。

# 添加更多

正如你现在可能已经了解的那样，我们可以根据自己的需求控制任何空调电器。我们已经理解了开关，并且已经完善了我们可以改变灯光强度和风扇速度的方式。但你有没有注意到一件事？随着我们的系统变得越来越复杂，所需的 GPIO 数量将会增加。总有一个时刻，你会想要连接更多的设备到你的树莓派上；然而，由于物理端口的不足，你将无法这样做。

这在电子学中是非常常见的情况。和往常一样，这个问题也有解决方案。这个解决方案被称为复用器。复用器的基本工作是在任何计算机系统中扩大端口的数量。现在你一定在想，它是如何做到的呢？

这个概念非常简单。让我们首先看一下复用器的图表：

![](img/aea10593-2bf8-4d57-9d45-65505431998a.png)

在上图中，您可以看到复用器有两端—一个是信号输出线，另一个是相对的。我们需要首先了解的是，复用器是一个双向设备，即它从复用器向连接的设备发送数据，反之亦然。

现在，首先是电源线，这很基本。它用于给复用器本身供电。然后，我们有**信号线**，它有两个端口，**Sig**和**EN**。**EN**代表使能，这意味着在**EN**不高的情况下，数据通信也不会发生。然后我们有一个叫做**Sig**的东西。这是连接到树莓派 GPIO 的用于数据通信的端口。接下来是选择线。正如您所看到的，我们有四个端口，分别是**S0**、**S1**、**S2**和**S3**。选择线的目的是选择需要选择的特定端口。以下是一个将澄清发生了什么的表：

| **S0** | **S1** | **S3** | **S4** | **选定输出** |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | C0 |
| 1 | 0 | 0 | 0 | C1 |
| 0 | 1 | 0 | 0 | C2 |
| 1 | 1 | 0 | 0 | C3 |
| 0 | 0 | 1 | 0 | C4 |
| 1 | 0 | 1 | 0 | C5 |
| 0 | 1 | 1 | 0 | C6 |
| 1 | 1 | 1 | 0 | C7 |
| 0 | 0 | 0 | 1 | C8 |
| 1 | 0 | 0 | 1 | C9 |
| 0 | 1 | 0 | 1 | C10 |
| 1 | 1 | 0 | 1 | C11 |
| 0 | 0 | 1 | 1 | C12 |
| 1 | 0 | 1 | 1 | C13 |
| 0 | 1 | 1 | 1 | C14 |
| 1 | 1 | 1 | 1 | C15 |

在上表中，您可以看到通过在选择线上使用各种逻辑组合，可以寻址各种线路。例如，假设我们在选择引脚上有以下序列—S0 = 1，S1 = 0，S2 = 1，S3 = 1。如果这是来自树莓派的选择引脚的输入，那么将选择引脚号 C13。这基本上意味着现在 C13 可以与复用器的引脚**Sig**进行数据通信。此外，我们必须记住，使能引脚必须高才能进行数据传输。

以类似的方式，我们可以继续处理复用器的所有 16 个引脚。因此，从逻辑上看，通过使用树莓派的六个引脚，我们可以继续利用 16 个 GPIO。既然我们已经了解了复用的基础知识，让我们继续尝试使用其中的一个。

![](img/d1d03c85-d1df-49df-8adb-5f7c95c895ed.png)

一旦硬件连接好了，让我们继续上传以下代码：

```py
import RPi.GPIO as GPIO import time  
GPIO.setmode(GPIO.BCM) GPIO.setwarnings(False) S0 = 21 S1 = 22 S2 = 23 S3 = 24 GPIO.setup(S0,GPIO.OUT) GPIO.setup(S1,GPIO.OUT) GPIO.setup(S2,GPIO.OUT) While True:  GPIO.output(S0,1) GPIO.output(S1,0) GPIO.output(S2,1) GPIO.output(S4,1) time.sleep(1) GPIO.output(S0,1) GPIO.output(S1,1) GPIO.output(S2,1) GPIO.output(S4,1) time.sleep(1) GPIO.output(S0,1) GPIO.output(S1,0) GPIO.output(S2,0) GPIO.output(S4,1) time.sleep(1) 'GPIO.output(S0,0) GPIO.output(S1,0) GPIO.output(S2,0) GPIO.output(S4,1) time.sleep(1) GPIO.output(S0,0) GPIO.output(S1,1) GPIO.output(S2,0) GPIO.output(S4,1) time.sleep(1) }
```

在这里，我们所做的实质上是，逐个触发选择线，以寻址 LED 连接的每个单个端口。每当发生这种情况时，相应的 LED 会发光。此外，它发光的原因是因为信号端`Sig`连接到树莓派的 3.3V。因此，向其连接的任何端口发送逻辑高电平。

这是复用器工作的基本方式之一。当我们使用多个设备和传感器时，这可能非常有用。

# 总结

在本章中，我们使 Jarvis 能够在不同条件下自动化您的家用电器，并将各种属性应用于系统。因此，请继续尝试许多其他情景，以增强您的家庭自动化系统。

在下一章中，我们将启用 Jarvis IoT，从而使用 Wi-Fi 和互联网从您的手机控制电器。
