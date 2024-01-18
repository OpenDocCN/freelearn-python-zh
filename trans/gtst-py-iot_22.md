# 障碍物避让的传感器接口

要制作一个能自行驾驶的机器人车，我们首先需要了解人类如何驾驶车辆。当我们开车时，我们不断分析空间和与其他物体的距离。然后，我们决定是否可以通过。这在我们的大脑-眼睛协调中不断发生。同样，机器人也需要做同样的事情。

在我们之前的章节中，你学到了我们可以使用传感器找到我们周围物体的接近程度。这些传感器可以告诉我们物体有多远，基于此，我们可以做出决定。我们之前使用超声波传感器主要是因为它非常便宜。然而，正如你记得的，附加超声波传感器并运行其代码稍微麻烦。现在是时候我们使用一个更简单的传感器并将其连接到汽车上了。

本章将涵盖以下主题：

+   红外近距离传感器

+   自主紧急制动

+   赋予它自动转向能力

+   使其完全自主

# 红外近距离传感器

以下照片描述了红外近距离传感器：

![](img/7608bfcf-f3d8-49e1-b982-1db4a12c2bdf.jpg)

它由两个主要部分组成-传感器和发射器。发射器发射红外波；这些红外（IR）波然后击中物体并返回到传感器，如下图所示。

![](img/8432224f-a4c0-45d9-ac6a-d0f80cd21fa1.png)

现在，正如你在前面的图表中所看到的，发射的红外波从与传感器不同距离的表面反弹回来，然后它们以一定角度接近传感器。现在，因为发射器和传感器之间的距离在任何时间点都是固定的，所以对应于反射的红外波的角度将与其反弹之前所走过的距离成比例。红外近距离传感器中有超精密传感器，能够感知红外波接近它的角度。通过这个角度，它给用户一个相应的距离值。这种找到距离的方法被称为三角测量，它在工业中被广泛使用。我们需要记住的另一件事是，正如我们在前面的章节中提到的，我们都被红外辐射所包围；任何绝对零度以上的物体都会发射相应的波。此外，我们周围的阳光也有大量的红外辐射。因此，这些传感器具有内置电路来补偿它；然而，它只能做到这么多。这就是为什么在处理直射阳光时，这个解决方案可能会有些麻烦。

现在，理论够了，让我们看看汽车实际上是如何工作的。我们在这个例子中使用的 IR 近距离传感器是夏普的模拟传感器，部件代码为 GP2D12。它的有效感应范围为 1000-800 毫米。范围还取决于所询问对象表面的反射性。物体越暗，范围越短。这个传感器有三个引脚。正如你可能已经猜到的，一个是 VCC，另一个是地，最后一个是信号。这是一个模拟传感器；因此，距离读数将基于电压给出。通常，大多数模拟传感器都会得到一个图表，其中会描述各种感应范围的各种电压。输出基本上取决于传感器的内部硬件和其结构，因此可能大不相同。下面是我们的传感器及其输出的图表：

![](img/9c9e2a5f-92fd-497a-9e13-31ab428dce10.png)

好吧，到目前为止一切都很好。正如我们所知，树莓派不接受模拟输入；因此，我们将继续使用我们之前使用过的 ADC。我们将使用之前使用过的相同 ADC。

# 自主紧急制动

有一种新技术，新车配备了这种技术。它被称为**自动紧急制动**；无论我们在驾驶时有多认真，我们都会分心，比如 Facebook 或 WhatsApp 的通知，这些会诱使我们从道路上的屏幕上看向手机。这可能是道路事故的主要原因；因此，汽车制造商正在使用自动制动技术。这通常依赖于远程和近程雷达，它检测车辆周围其他物体的接近，在即将发生碰撞的情况下，自动将车辆刹车，防止它们与其他车辆或行人相撞。这是一个非常酷的技术，但有趣的是，我们今天将亲手制作它。

为了实现这一点，我们将使用红外接近传感器来感知周围物体的接近。现在，继续，拿一张双面胶带，把红外距离传感器粘在车子的前面。一旦完成这一步，按照这里所示的连接电路。

![](img/ef0c2188-ae4d-490b-b00c-af05811945b0.png)

好了，我们已经准备好编写代码了。以下是代码，只需将其复制到你的树莓派上：

```py
import RPi.GPIO as GPIO import time GPIO.setmode(GPIO.BCM)  import Adafruit_ADS1x15 adc0 = Adafruit_ADS1x15.ADS1115()   GAIN = 1  adc0.start_adc(0, gain=GAIN)   Motor1a = 20 Motor1b = 21 Motor2b = 23
Motor2a = 24  GPIO.setup(Motor1a,GPIO.OUT) GPIO.setup(Motor1b,GPIO.OUT) GPIO.setup(Motor2a,GPIO.OUT) GPIO.setup(Motor2b,GPIO.OUT)  def forward(): GPIO.output(Motor1a,0) GPIO.output(Motor1b,1) GPIO.output(Motor2a,0) GPIO.output(Motor2b,1)    def stop():  GPIO.output(Motor1a,0) GPIO.output(Motor1b,0) GPIO.output(Motor2a,0) GPIO.output(Motor2b,0) while True:  F_value = adc0.get_last_result()  F = (1.0  / (F_value /  13.15)) -  0.35  forward()  min_dist = 20  if F < min_dist:  stop()  
```

现在，让我们看看这段代码实际上发生了什么。一切都非常基础；红外线接近传感器感知到其前方物体的接近，并以模拟信号的形式给出相应的距离值。然后这些信号被 ADC 获取，并转换为数字值。这些数字值最终通过 I2C 协议传输到树莓派上。

到目前为止，一切都很好。但你一定想知道这行代码是做什么的？

```py
 F = (1.0  / (F_value /  13.15)) -  0.35
```

这里我们并没有做太多事情，我们只是获取 ADC 给出的数字值，然后使用这个公式，将数字值转换为以厘米为单位的可理解的距离值。这个计算是由制造商提供的，我们不需要深究这个。大多数传感器都提供了这些计算。然而，如果你想了解我们为什么使用这个公式，我建议你查看传感器的数据表。数据表可以在以下链接上轻松找到：[`engineering.purdue.edu/ME588/SpecSheets/sharp_gp2d12.pdf`](https://engineering.purdue.edu/ME588/SpecSheets/sharp_gp2d12.pdf)。

接下来，代码的主要部分如下：

```py
min_dist = 20 If F < min_dist:
 stop()
```

这也很简单。我们输入了一个距离值，在这个程序中，我们将其设置为`20`。所以，每当`F`的值（红外接近传感器获取的距离）小于`20`时，就会调用`stop()`函数。`stop`函数只是让车子停下来，防止它与任何东西相撞。

让我们上传代码，看看它是否真的有效！确保你在室内测试这辆车；否则，如果没有障碍物，你将很难停下这辆车。玩得开心！

# 给车子自动转向的能力

希望你对这个小东西玩得开心。传感器的应用是如此简单，但它可以产生如此大的影响。既然你已经学会了基础知识，现在是时候向前迈进，给车子一些更多的能力了。

在之前的代码中，我们只是让机器人停在障碍物前面，为什么我们不让它绕过车子呢？这将非常简单又非常有趣。我们只需要调整`stop()`函数，使其能够转向。显然，我们还将把函数的名称从`stop()`改为`turn()`，只是为了清晰起见。要记住的一件事是，你不需要重写代码；我们只需要做一些微小的调整。所以，让我们看看代码，然后我会告诉你到底发生了什么变化以及为什么：

```py
import RPi.GPIO as GPIO import time GPIO.setmode(GPIO.BCM)  import Adafruit_ADS1x15 adc0 = Adafruit_ADS1x15.ADS1115()   GAIN = 1  adc0.start_adc(0, gain=GAIN)   Motor1a = 20 Motor1b = 21 Motor2a = 23 Motor2b = 24  GPIO.setup(Motor1a,GPIO.OUT) GPIO.setup(Motor1b,GPIO.OUT) GPIO.setup(Motor2a,GPIO.OUT) GPIO.setup(Motor2b,GPIO.OUT)  def forward(): GPIO.output(Motor1a,0) GPIO.output(Motor1b,1) GPIO.output(Motor2a,0) GPIO.output(Motor2b,1)   def turn():
 GPIO.output(Motor1a,0) GPIO.output(Motor1b,1) GPIO.output(Motor2a,1) GPIO.output(Motor2b,0) )  while True:
   forward() F_value = adc0.get_last_result()  F = (1.0  / (F_value /  13.15)) -  0.35
     min_dist = 20

 while F < min_dist: turn()  
```

你可能已经注意到，除了以下内容，其他都基本保持不变：

```py
def turn():
 GPIO.output(Motor1a,0) GPIO.output(Motor1b,1) GPIO.output(Motor2a,1) GPIO.output(Motor2b,0)
```

这部分代码定义了“转向()”函数，在这个函数中，车辆的对侧车轮会以相反的方向旋转；因此，使车辆绕着自己的轴转动：

```py
 min_dist = 20 while F < min_dist: turn()
```

现在这是程序的主要部分；在这部分中，我们正在定义汽车在遇到任何障碍物时会做什么。在我们之前的程序中，我们主要是告诉机器人一旦遇到障碍物就停下来；然而，现在我们正在将“停止”函数与“转向”函数链接起来，这两个函数我们之前在程序中已经定义过了。

我们只是放入了一个条件，如下所示：

```py
min_dist = 20 If F < min_dist:
 turn()
```

然后，它会转动一小段时间，因为微控制器会解析代码并执行它，然后跳出条件。为了做到这一点，我们的树莓派可能只需要几微秒。所以，我们甚至可能看不到发生了什么。因此，在我们的程序中，我们使用了一个“while”循环。这基本上保持循环运行，直到条件满足为止。我们的条件是“while F < min_dist:”，所以只要机器人在前面检测到物体，它就会继续执行其中的函数，而在我们的情况下，就是“转向()”函数。简而言之，直到它没有转到足够的程度来避开障碍物为止，车辆将继续转向，然后一旦循环执行完毕，它将再次跳回到主程序并继续直行。

简单吧？这就是编程的美妙之处！

# 使其完全自主

现在，你一定已经了解了使用简单的接近传感器进行自动驾驶的基础知识。现在是我们使其完全自主的时候了。要使其完全自主，我们必须了解并映射我们的环境，而不仅仅是在车辆遇到障碍物时转向。我们基本上需要将整个活动分为以下两个基本部分：

+   扫描环境

+   决定如何处理感知到的数据

现在，让我们先编写代码，然后看看我们需要做什么：

```py
import RPi.GPIO as GPIO import time GPIO.setmode(GPIO.BCM) import Adafruit_ADS1x15
adc0 = Adafruit_ADS1x15.ADS1115() GAIN = 1 adc0.start_adc(0, gain=GAIN) Motor1a = 20 Motor1b = 21 Motor2a = 23 Motor2b = 24 GPIO.setup(Motor1a,GPIO.OUT) GPIO.setup(Motor1b,GPIO.OUT) GPIO.setup(Motor2a,GPIO.OUT) GPIO.setup(Motor2b,GPIO.OUT)  def forward(): GPIO.output(Motor1a,0) GPIO.output(Motor1b,1) GPIO.output(Motor2a,0) GPIO.output(Motor2b,1)  def right(): GPIO.output(Motor1a,0) GPIO.output(Motor1b,1) GPIO.output(Motor2a,1) GPIO.output(Motor2b,0)  def left(): GPIO.output(Motor1a,1) GPIO.output(Motor1b,0) GPIO.output(Motor2a,0) GPIO.output(Motor2b,1)  def stop(): GPIO.output(Motor1a,0) GPIO.output(Motor1b,0) GPIO.output(Motor2a,0) GPIO.output(Motor2b,0)  while True:  forward()  F_value = adc0.get_last_result() F = (1.0  / (F_value /  13.15)) -  0.35  min_dist = 20 if F< min_dist: stop() right() time.sleep(1) F_value = adc0.get_last_result()  F = (1.0  / (F_value /  13.15)) -  0.35  R = F left() time.sleep(2)  F_value = adc0.get_last_result()   F = (1.0  / (F_value /  13.15)) -  0.3  L = F if L < R: right()
        time.sleep(2) else: forward()  
```

现在大部分程序就像我们之前的所有程序一样；在这个程序中，我们定义了以下函数：

+   “前进()”

+   “右()”

+   “左()”

+   “停止()”

关于定义函数，我没有太多需要告诉你的，所以让我们继续前进，看看我们还有什么。

主要的操作是在我们的无限循环“while True:”中进行的。让我们看看到底发生了什么：

```py
while True:

 forward() F_value = adc0.get_last_result() F = (1.0  / (F_value /  13.15)) -  0.35

 min_dist = 20 if F< min_dist: stop()
```

让我们看看这部分代码在做什么：

+   一旦我们的程序进入无限循环，首先执行的是“前进()”函数；也就是说，一旦无限循环执行，车辆就会开始向前行驶。

+   此后，“F_value = adc.get_last_result()”正在从 ADC 中获取读数并将其存储在一个名为“F_value”的变量中

+   “F = (1.0/(F-value/13.15))-0.35”正在计算可理解的度量距离值

+   “min_dist = 20”，我们只是定义了稍后将使用的最小距离

一旦这部分代码完成，那么“if”语句将检查是否“F < min_dist:”。如果是这样，那么“if”语句下的代码将开始执行。这部分代码的第一行将是“停止()”函数。所以每当车辆在前面遇到障碍物时，它将首先停下来。

现在，正如我所提到的，我们代码的第一部分是了解环境，所以让我们继续看看我们是如何做到的：

```py
right()
 time.sleep(1) F_value = adc0.get_last_result()  F = (1.0  / (F_value /  13.15)) -  0.35
 R = F left() time.sleep(2)  F_value = adc0.get_last_result()
  F = (1.0  / (F_value /  13.15)) -  0.35
 L = F 
```

车辆停下后，它将立即向右转。正如你所看到的，代码的下一行是“time.sleep(1)”，所以在另外的“1”秒钟内，车辆将继续向右转。我们随机选择了“1”秒的时间，你可以稍后调整它。

一旦它向右转，它将再次从接近传感器中获取读数，并使用这段代码“R=F”，我们将这个值存储在一个名为“R”的变量中。

在这样做之后，车辆将转向另一侧，也就是向左侧，使用`left()`函数，并且它将持续向左转动`2`秒，因为我们有`time.sleep(2)`。这将使车辆转向障碍物的左侧。一旦它向左转，它将再次接收接近传感器的值，并使用代码`L = F`将该值存储在变量`L`中。

所以，我们所做的实质上是扫描我们周围的区域。在中心，有一个障碍物。它将首先向右转，并获取右侧的距离值；然后，我们将向左转并获取左侧的距离值。因此，我们基本上知道了障碍物周围的环境。

现在我们来到了必须做出决定的部分，即我们必须向前走的方向。让我们看看我们将如何做到：

```py
 if L < R: right()
        time.sleep(2) else: forward()
```

使用`if`语句，我们通过这段代码`if L < R:`比较障碍物左右侧的接近传感器的值。如果`L`小于`R`，那么车辆将向右转动`2`秒。如果条件不成立，那么`else:`语句将生效，车辆将前进。

现在，如果我们从更大的角度看代码，以下事情正在发生：

+   车辆会一直前进，直到遇到障碍物

+   遇到障碍时，机器人会停下来

+   它将首先向右转，并测量其前方物体的距离

+   然后，它将向左转，并测量其前方物体的距离

+   之后，它将比较左右两侧的距离，并选择它需要前进的方向

+   如果它需要向右转，它将向右转，然后前进

+   如果它需要向左转，那么它已经处于左转方向，所以它只需要直走

让我们上传代码，看看事情是否按计划进行。请记住，尽管每个环境都不同，每辆车也不同，所以你可能需要调整代码以使其顺利运行。

现在我给你留下一个问题。如果在两种情况下传感器的读数都是无穷大或者它能给出的最大可能值，那么机器人会怎么做？

继续，进行一些头脑风暴，看看我们能做些什么来解决这个问题！

# 总结

在本章中，利用你迄今为止学到的所有基础知识，以及引入红外接近传感器，我们能够更进一步地发展我们的机器人车，以便检测障碍物并相应地改变方向。在下一章中，我们将学习如何制作我们自己的区域扫描仪——到时见！
