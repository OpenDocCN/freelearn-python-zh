# 手势识别

自人类诞生以来，人们就使用手势相互交流，甚至在没有任何正式语言之前。手势是交流的主要方式，这也可以从世界各地发现的古代雕塑中看出，手势一直是一种非常有效地传递大量数据的成功方式，有时甚至比语言本身更有效。

手势是自然的，它们可能是对某种情况的反射。即使在我们不知道的情况下，它也会在潜意识中发生。因此，它成为了与各种设备进行交流的理想方式。然而，问题仍然存在，如何？

我们可以肯定，如果我们谈论手势，那么我们肯定需要做大量的编程来识别视频中的手势；此外，这也需要大量的处理能力来实现。因此，这是不可能的。我们可以使用一系列接近传感器构建一些基本的手势识别系统。然而，识别的手势范围将非常有限，使用的端口也会多倍增加。

因此，我们需要找到一个易于使用且成本不会超过其提供的解决方案。

本章将涵盖以下主题：

+   电场感应

+   使用Flick HAT

+   基于手势识别的自动化

# 电场感应

近场传感是一个非常有趣的传感领域。准备好一些有趣的东西。如果你感到有点困倦，或者注意力不集中，那就喝点咖啡，因为这个系统的工作原理可能会有点新。

每当有电荷时，就会伴随着一个相关的电场。这些电荷在空间中传播并绕过物体。当这种情况发生时，与之相关的电场具有特定的特征。只要周围的环境是空的，这种特征就会保持不变。

对于我们使用的手势识别板，它周围的感应范围只有几厘米，所以超出这一点的任何东西都可以忽略不计。如果那个区域没有任何东西，那么我们可以安全地假设被感应到的电场模式不会改变。然而，每当一个物体，比如我们的手，靠近时，这些波就会被扭曲。这种扭曲直接与物体的位置和姿势有关。通过这种扭曲，我们可以感应到手指的位置，并通过持续的感应，看到正在执行的动作是什么。所讨论的板看起来像这样：

![](Images/73e2e142-b4a4-4f4c-aced-19fc2353a0b1.jpg)

板上的中央交叉区域是发射器，两侧是四个矩形结构。这些是感应元件。它们感应空间中的波纹模式。基于此，它们可以推导出物体的x、y和z坐标。这由一个名为MGC 3130的芯片提供动力。这个芯片进行所有计算，并将原始读数传递给用户，关于坐标。

# 使用Flick HAT

Flick HAT以盾牌的形式出现，你可以简单地将其插入树莓派并开始使用。然而，一旦你这样做了，你就不会剩下任何GPIO引脚。因此，为了避免这个问题，我们将使用公对母导线连接它。这将使我们可以访问其他GPIO引脚，然后我们可以玩得开心。

所以，继续按以下方式连接。以下是Flick板的引脚图：

![](Images/1685e1fc-657c-43f0-b058-708de0c1e97d.png)

然后，按照以下方式进行连接：

![](Images/80ba5ee5-864a-4132-96b9-eaca317eb73e.png)

连接完成后，只需上传这个代码，看看会发生什么：

```py
import signal
import flicklib
import time
def message(value):
   print value
@flicklib.move()
def move(x, y, z):
   global xyztxt
   xyztxt = '{:5.3f} {:5.3f} {:5.3f}'.format(x,y,z)
@flicklib.flick()
def flick(start,finish):
   global flicktxt
   flicktxt = 'FLICK-' + start[0].upper() + finish[0].upper()
   message(flicktxt)
def main():
   global xyztxt
   global flicktxt
   xyztxt = ''
   flicktxt = ''
   flickcount = 0
   while True:

  xyztxt = ''
  if len(flicktxt) > 0 and flickcount < 5:
      flickcount += 1
  else:
      flicktxt = ''
      flickcount = 0
main()
```

现在一旦你上传了代码，让我们继续了解这个代码实际在做什么。

我们正在使用一个名为`import flicklib`的库，这是由这块板的制造商提供的。这个库的函数将在本章中用于与挥动板通信和获取数据。

```py
def message(value):
    print value
```

在这里，我们定义了一个名为`message(value)`的函数，它将简单地打印传递给函数的任何值：

```py
@flicklib.move()
```

这有一个特殊的装饰器概念。根据定义，装饰器是一个接受另一个函数并扩展后者行为的函数，而不明确修改它。在上一行代码中，我们声明它是一个装饰器`@`。

这有一个特殊的作用：动态定义程序中的任何函数。这意味着用这种方法定义的函数可以根据用户的定义而有不同的工作方式。

函数`move()`将进一步由在其后定义的函数补充。这种函数称为嵌套函数。也就是函数内部的函数：

```py
def move(x, y, z):
    global xyztxt
    xyztxt = '{:5.3f} {:5.3f} {:5.3f}'.format(x,y,z)
```

在这里，我们定义了一个名为`move()`的函数，它的参数是`x`、`y`和`z`。在函数内部，我们定义了一个名为`xyztxt`的全局变量；现在，`xyztxt`的值将以五位数字的形式呈现，小数点后有三位。我们是如何知道的呢？正如你所看到的，我们使用了一个名为`format()`的函数。这个函数的作用是根据用户的要求格式化给定变量的值。我们在这里声明值为`{:5.3f}`。`:5`表示它将是五位数，`3f`表示小数点后将是三位数。因此，格式将是`xxx.xx`：

```py
def flick(start,finish):
    global flicktxt
    flicktxt = 'FLICK-' + start[0].upper() + finish[0].upper()
    message(flicktxt)
```

在这里，我们定义了一个名为`flick(start, finish)`的函数。它有两个参数：`start`和`finish`。使用行`flicktxt = 'FLICK-' + start[0].upper() + finish[0].upper()`，这是根据手势板识别的字符进行切片。如果检测到南-北挥动，则开始为南，结束为北。现在我们只使用单词的第一个字符：

```py
    global xyztxt
    global flicktxt
```

我们再次全局定义了名为`xyztxt`和`flicktxt`的变量。之前，我们所做的是在函数中定义它。因此，重要的是在主程序中定义它：

```py
if len(flicktxt) > 0 and flickcount < 5:
            flickcount += 1
else:
            flicktxt = ''
            flickcount = 0
```

当检测到手势时，`flicktxt`变量将获得与手势相对应的值。如果没有手势，那么`flicktxt`将保持为空。一个名为`flickcount`的变量将计算它被刷过多少次。如果值超出指定范围，那么`flicktxt`将使用行`flicktxt = ''`清除为空字符串，`flickcount`将被设为0。

这将产生一个文本输出，向用户提供手势挥动的方向。

# 基于手势识别的自动化

现在我们已经按照以下图表接口了连接：

![](Images/f11065c7-a56f-4673-a5e8-9604941953e7.png)

让我们继续上传以下代码：

```py
import signal
import flicklib
import time
import RPi.GPIO as GPIO
GIPO.setmode(GPIO.BCM)
GPIO.setup(light, GPIO.OUT)
GPIO.setup(fan,GPIO.OUT)
pwm = GPIO.PWM(fan,100)
def message(value):
   print value
@flicklib.move()
def move(x, y, z):
   global xyztxt
   xyztxt = '{:5.3f} {:5.3f} {:5.3f}'.format(x,y,z)
@flicklib.flick()
def flick(start,finish):
   global flicktxt
   flicktxt = 'FLICK-' + start[0].upper() + finish[0].upper()
   message(flicktxt)
def main():
   global xyztxt
   global flicktxt
   xyztxt = ''
   flicktxt = ''
   flickcount = 0
   dc_inc = 0
   dc_dec = 0

while True:
  pwm.start(0)
  xyztxt = ' '
  if len(flicktxt) > 0 and flickcount < 5:
    flickcount += 1
  else:
    flicktxt = ''

flickcount = 0
if flicktxt ==”FLICK-WE”:
  GPIO.output(light,GPIO.LOW)
if flicktxt ==”FLICK-EW”:
  GPIO.output(light,GPIO.HIGH)
if flicktxt ==”FLICK-SN”:
  if dc_inc < 100:
    dc_inc = dc_inc + 10
    pwm.changeDutyCycle(dc_inc)

else:
  Dc_inc = 10
  if flicktxt ==”FLICK-NS”:
    if dc_inc >0:
    dc_dec = dc_dec - 10
    pwm.changeDutyCycle(dc_dec)
main()
```

该程序是在我们之前完成的程序的基础上，我们始终有一些额外的功能，可以使用通过手势板接收到的数据来开启或关闭灯光。

与之前的程序一样，我们正在以手势板上的方向形式接收手势，并使用简单的条件来关闭灯光或打开它们。因此，让我们看看有哪些添加：

```py
 if flicktxt ==”FLICK-WE”: GPIO.output(light,GPIO.LOW)
```

第一个条件很简单。我们正在将`flicktxt`的值与给定变量进行比较，在我们的情况下是`FLICK-WE`，其中`WE`代表从**西**到**东**。因此，当我们从西向东挥动，或者换句话说，当我们从左向右挥动时，灯光将被关闭：

```py
 if flicktxt ==”FLICK-EW”: GPIO.output(light,GPIO.HIGH)
```

与之前一样，我们再次使用名为`FLICK-EW`的变量，它代表从东到西的挥动。它的作用是，每当我们从东向西挥动手，或者从右向左挥动手时，灯光将被打开：

```py
 if flicktxt ==”FLICK-SN”: if dc_inc <= 100:  dc_inc = dc_inc + 20
 pwm.changeDutyCycle(dc_inc)
```

现在我们已经加入了一个调光器和一个风扇来控制风扇的速度；因此，我们将不得不给它一个与我们想要驱动它的速度相对应的PWM。现在每当用户将手从南向北或从下到上甩动时。条件 `if dc_inc <100` 将检查 `dc_inc` 的值是否小于或等于 `100`。如果是，则它将增加 `20` 的值。使用函数 `ChangeDutyCycle()`，我们为调光器提供不同的占空比；因此改变了风扇的整体速度。每次你向上划动风扇的值，它将增加20%：

```py
 else: Dc_inc = 10 if flicktxt ==”FLICK-NS”: if dc_inc >0:  dc_dec = dc_dec - 10
 pwm.changeDutyCycle(dc_dec)
```

# 摘要

在本章中，我们能够理解手势识别是如何通过电场检测工作的概念。我们也了解到使用手势控制板和手势控制家庭是多么容易。我们将在下一章中涵盖机器学习部分。
