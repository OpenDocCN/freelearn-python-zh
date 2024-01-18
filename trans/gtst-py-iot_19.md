# 制作园丁机器人

好了，朋友们，你已经了解了一些输入和输出的基础知识；现在是时候制作一些我们可以交出一些日常责任的东西了。这个机器人可能看起来并不像一个机器人，但相信我，它会让你的生活更轻松。最重要的是，你花园中的大部分植物都会因为你的制作而祝福你。

我们将涵盖以下主题：

+   与电磁阀一起工作

+   制作机器人

+   使它更智能

+   使它真正智能

# 与电磁阀一起工作

我们要做的是一个自动系统，它会在植物需要时给它们浇水。所以从技术上讲，一旦它建立起来，你就不用担心给你的绿色生物浇水了。无论你是在家里、在办公室还是度假，它都会不管任何情况下继续工作。

现在，你一定在想它是如何给植物浇水的，所以让我告诉你，对于这个世界上的每个问题，都存在一个解决方案。在我们的情况下，这个解决方案被称为电磁阀。它的基本作用是切换液体的流动。市场上有各种各样的电磁阀；一些识别特征如下：

+   **尺寸**：它们有各种尺寸，如半英寸、四分之三英寸、1英寸等。这基本上将决定电磁阀的流量。

+   **介质**：无论是液体、气体、蒸汽等。

+   **正常状态**：

+   **通常打开**：这个阀门在关闭状态下会允许液体流动——当阀门没有供电时

+   **通常关闭**：这个阀门在关闭状态下会阻止液体流动——当阀门没有供电时

+   **方式数量**：一个简单的阀门会有一个进口和一个出口。所以，当它打开时，它会允许液体从进口流向出口。然而，还可以有其他类型的阀门，比如三通阀，可能有两个出口和一个进口。它会调节液体的流动方向。

阀门的一些具体细节也可能会有所不同，但目前我们只需要知道这些。关于电磁阀要注意的一点是，这些阀门可以打开或关闭。无法实现这些阀门之间的任何状态或通过这些阀门控制流动。为此，我们可以使用伺服阀或电动阀。但目前我们不需要。

在本章中，我们将使用一个半英寸的水/液体阀，它通常是关闭的。当你仔细看这个阀时，你会发现它在12伏特下运行，电流消耗接近1安培。这对树莓派来说是很大的电流。树莓派每个引脚可以提供的电流上限约为50毫安。所以如果我们把这个阀接到树莓派上，它肯定不会工作。

我们现在该怎么办？这个问题的答案是继电器。继电器的基本工作是重新布置电路。基本上，它是一个电子控制开关。继电器的基本工作是打开和关闭具有比控制单元提供的更高电流/电压消耗的设备。这是一个相当简单的设备，正如你在图中所看到的。有两个电路。一个是蓝色的，是低电压和低电流电路。这个电路正在给线圈供电。另一个电路是红色和黑色的。这个电路是高电压、高电流电路。

在初始阶段，正如你所看到的，高电压高电流电路不完整，烤箱不会工作：

![](Images/0045c276-12d5-466a-b712-5ae02a54e2e6.png)

现在，在这第二个图中，你可以看到蓝色电路连接到5V电源，线圈被激活。每当线圈被激活，它就形成一个电磁铁，吸引高功率电路的金属片，使电路完整，从而给烤箱供电：

![](Images/761c5221-8448-48a8-a7d4-3bd0e7711f4b.png)

这就是电磁阀的工作原理。线圈的消耗几乎只有几毫安，因此通过微控制器驱动线圈非常容易。这反过来使得最终电路之间产生接触。

市场上有各种类型的继电器；一些识别特征如下：

+   最大输出电压：它可以处理的最大电压

+   最大输出电流：它可以承受的连接到它的任何输出设备的最大电流

+   信号电压：它需要开关组件的电压

+   正常条件：

+   正常关闭：这将不允许任何电流流动，直到接收到信号为止

+   正常开启：它将允许电流流动，直到接收到信号为止

现在，回到我们的园艺机器人，连接到它的电磁阀将在1安培和12V上工作，因此任何可以提供等于或大于1安培和12V的继电器都可以工作。

通常，市场上可用的继电器是120V和12安培直流。要记住的一件重要事情是交流电压和直流电压和电流将有两个单独的等级。由于我们的电磁阀将在12V下工作，我们只考虑直流的上限。

# 制作机器人

现在，让我们开始制作机器人。首先，您需要从水龙头到电磁阀的水管连接，从电磁阀到洒水器的连接。您还需要进行以下连接：

![](Images/a2fe556c-551e-4c36-a0bd-eeafc2fcea3c.png)

现在让我们开始编程。在这个机器人中，我们将接口一个土壤湿度传感器。该传感器的工作是确定土壤中的水量。通过确定这一点，我们可以了解花园是否需要水。这个土壤湿度传感器是一个模拟传感器，因此我们将使用ADC将模拟读数转换为树莓派可理解的数字值。所以让我们开始吧：

```py
import time
import RPi.GPIO as GPIO
import Adafruit_ADS1x15
water_valve_pin = 23
moisture_percentage = 20
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(water_valve_pin, GPIO.OUT)
adc = Adafruit_ADS1x15.ADS1115()
channel = 0
GAIN = 1
while True:
 adc.start_adc(channel, gain=GAIN)
 moisture_value = adc.get_last_result()
 moisture_value= int(moisture_value/327)
 print moisture_value
 if moisture_value < moisture_percentage:
 GPIO.output(water_valve_pin, GPIO.HIGH)
 time.sleep(5)
 else:
 GPIO.output(water_valve_pin, GPIO.LOW)
```

在运行此代码之前，让我们先了解它实际上在做什么：

```py
moisture_percentage = 20
```

`moisture_percentage = 20`是一个阈值百分比；如果土壤中的湿度水平低于20%，那么您的花园就需要水。这是您的机器人将继续寻找的条件；一旦满足这个条件，就可以采取适当的行动。这个百分比也可以根据您花园的需要更改为`30`、`40`或其他任何值：

```py
moisture_value = int(moisture_value/327)
```

ADC是一个16位设备——有16个二进制数字可以表示一个值。因此，该值可以在`0`和`2^(15)`之间，换句话说，可以在`0`和`32768`之间。现在，很简单的数学，对于每个百分比的湿度，ADC将给出以下读数：`32768/100`，或`327.68`。因此，要找出土壤中的湿度百分比，我们需要将ADC给出的实际值除以`327.68`。

其余的代码非常简单，一旦您阅读它，您就不会很难理解。

# 使其更智能

祝贺您制作了您的第一个机器人！但您是否注意到了一个问题？我们制作的机器人一直在寻找湿度值，一旦注意到湿度值偏低，它就会突然泵水，并确保土壤的湿度始终高于20%。然而，这是不必要的。一般来说，我们每天浇水一两次。如果我们浇水更多，那对植物可能不利。

因此，让我们继续使它稍微更智能化，并且只在特定时间土壤湿度低时给植物浇水。这一次，我们不需要对硬件进行任何更改；我们只需要微调代码。

让我们继续上传以下代码，然后看看到底发生了什么：

```py
from time import sleep
from datetime import datetime
import RPi.GPIO as GPIO
import Adafruit_ADS1x15
water_valve_pin = 23
moisture_percentage = 20
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(water_valve_pin, GPIO.OUT)
adc = Adafruit_ADS1x15.ADS1115()
GAIN = 1
def check_moisture():
 adc.start_adc(0,gain= GAIN)
 moisture_value = adc.get_last_result()
 moisture_value = int(moisture_value/327)
 if moisture_value < moisture_level:
 GPIO.output(water_valve_pin, GPIO.HIGH)
 sleep(5)
 GPIO.output(water_valve_pin, GPIO.LOW)
 else:
 GPIO.output(water_valve_pin, GPIO.LOW)
while True:
 H = datetime.now().strftime('%H')
 M = datetime.now().strftime('%M')
 if H == ‘07’ and M <= ‘10’:
 check_moisture()
 if H == ‘17’ and M <= ‘01’:
 check_moisture()
```

这段代码可能对您来说有点陌生，但相信我，它就是这么简单。让我们一步一步地看看发生了什么：

```py
from datetime import datetime
```

这行代码是从日期时间库中导入日期时间实例。这是Python中默认的一个库。我们只需要调用它。它的作用是在我们的代码中轻松确定时间。

```py
def check_moisture():
```

有时我们必须一遍又一遍地做一些事情。这些代码集可以是几行重复的代码，也可以是多页的代码。因此，重写那些代码毫无意义。我们可以创建一个函数。在这个函数中，我们可以定义每次调用时会发生什么。在这行代码中，我们创建了一个名为`check_moisture()`的函数；现在，每当程序中调用这个函数时，将执行一系列活动。将要执行的一系列活动由用户定义。因此，每当我们写`def`时，就意味着我们正在定义一个函数；然后，我们写出需要定义的函数的名称。

完成后，然后我们在缩进中写的任何内容都将在调用函数时执行。请记住，每当我们调用或定义一个函数时，函数名称的末尾都有一个开放和关闭的`()`括号表示：

```py
 moisture_value = adc.get_last_result()
```

`adc.get_last_result()`是`adc`的一个函数。它的功能是简单地从之前定义的引脚（引脚号为`0`）获取结果，并将读数存储到变量`moisture_value`中。因此，在`moisture_value`之后将是ADC引脚号`0`的读数，或者换句话说，是湿度传感器的读数。

```py
H = datetime.now().strftime('%H')
```

代码`datetime`是`.now()`的一个实例和方法。这个函数的作用是更新时间。现在，`datetime.now()`已经更新了日期和时间的所有参数，包括小时、分钟、秒，甚至日期。我们可以选择是否要全部或者日期和时间的任何特定部分。目前，我们想要将小时的值放入变量`H`中，因此我们使用了`.strftime('%H')`方法。`strftime`代表时间的字符串格式。因此，它输出的任何值都是以字符串格式。`('%H')`表示它只会给我们小时的值。同样，我们也可以使用`('%M')`和`('%S)`来获取分钟的时间。我们还可以使用以下语法获取日期、月份和年份的值：

+   获取日期：`('%d')`

+   获取月份：`('%m')`

+   获取年份：`('%Y')`

```py
if H == ‘07’ and M <= ‘10’:
```

在前面的条件中，我们正在检查时间是否为7点；此外，我们还在检查时间是否小于或等于10分钟。因此，只有当时间为7小时并且在0到10分钟之间时，此代码段才会运行`if`语句中的语句。

特别要注意的一点是，我们在两个条件之间使用了`and`，因此只有在两个语句都绝对为真时才会运行其中的代码。我们还可以在其中使用一些其他语句，比如`or`，在这种情况下，如果其中一个语句为真，它将运行代码。

如果我们在这个`if`语句中用`or`替换`and`，那么它将在每个小时的0到10分钟内运行代码，并且将在上午7:00到7:59之间的整个时间内连续运行代码：

```py
check_moisture()
```

正如你可能记得的，之前我们定义了一个名为`check_moisture()`的函数。在定义该函数时，我们还定义了每次调用该函数时将发生的一系列活动。

现在是调用该函数的时候了。一旦程序到达代码的末尾，它将执行之前在函数中定义的一系列活动。

所以我们就是这样。现在，一旦你运行这段代码，它将等待程序中定义的时间。一旦达到特定的时间，它将检查湿度。如果湿度低于设定值，它将开始给植物浇水，直到湿度超过阈值为止。

# 真正智能化

了不起的工作！我们已经开始自己建造比我们更聪明的东西。但现在我们想要更进一步，让它比我们更聪明——这就是机器人存在的意义。不仅仅是做我们做的事情，而是以更好的方式做所有这些。

那么，我们能做些什么改进呢？在寒冷的冬天，我们不需要太多的水，但在夏天，我们需要比冬天喝的水多得多。植物也是一样的情况。

在冬天，它们需要的水量要少得多。此外，土壤中的水蒸发速度也较慢。因此，在这两种情况下，我们需要向花园供应不同数量的水。问题是，我们该如何做到呢？

首先，要知道外面是热还是冷，我们需要一个传感器。我们将使用一个名为DHT11的传感器。这是一个便宜但坚固的传感器，可以给我们提供温度和湿度的读数。最好的部分是，它的价格非常便宜，大约2美元。

它有四个引脚。但是，如果你认为它将适用于I2C协议，那么你就错了。它有自己的数据传输方法。拥有一个单一的协议来处理所有传感器是很好的，但通常你也会发现有各种传感器或设备使用不同或全新的协议。DHT11就是这样的传感器。在这种情况下，我们可以选择要么理解整个通信方法，要么简单地从制造商那里获取库并随时使用。目前我们将选择后者。

现在让我们看看DHT11的引脚是什么样子的：

![](Images/ca6928e6-75fe-41ad-b639-20471e708ddb.png)

你可以看到这里只有一个信号引脚，它将完成所有数字通信。有两个电源引脚，其中一个引脚没有使用。也就是说，这个引脚没有明显的用途。它可能只是用于焊接或将来使用。这个传感器使用5V电源，只需要几毫安，因此我们可以通过树莓派来为其供电。现在，对于数据通信，我们将把信号引脚连接到GPIO引脚号`4`。

在我们开始编写代码之前，让我们先安装DHT11和树莓派之间的通信库。我们之前已经在ADS1115的库中做过这个，但在这个库中有一些小技巧需要我们注意。所以让我们开始吧。

首先，我们需要确保你的树莓派操作系统是最新的。所以将树莓派连接到互联网，打开树莓派的命令提示符，输入以下命令：

```py
sudo apt-get update
```

这个命令将自动更新你的树莓派的raspbian操作系统。然后继续输入这个命令：

```py
sudo apt-get install build-essential python-dev python-openssl
```

在这个命令中，我们正在安装以下软件包：

+   `build-essential`

+   `python-dev`

+   `python-openssl`

你一定在想为什么我们要安装所有这些。好吧，长话短说，这些是我们即将安装的DHT11通信库的依赖项。如果这些软件包没有安装在树莓派上，我们将无法使用该库。

最后，我们必须安装库；这是一个通用库，其中还包括与DHT11传感器通信的功能。这应该足以满足我们的简单通信需求。以下是安装它的命令：

```py
sudo python setup.py install
```

好了，我们准备好了。我们的系统已经准备好与DHT11进行通信。让我们首先看看我们到目前为止所做的是否按我们想要的方式工作。为了做到这一点，按照以下方式连接DHT11；你可以将其他组件如电磁阀和土壤湿度传感器连接好。它们不应该干扰。现在在树莓派上上传以下代码：

```py
from time import sleep
from datetime import datetime
import RPi.GPIO as GPIO
import Adafruit_DHT
sensor = 11
pin = 4
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
while True:
 humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
 print("Temperature: " +temperature+ "C")
 print("Humidity: " +humidity+ "%")
 time.sleep(2)
```

一旦你上传了这段代码，你将在屏幕上看到传感器的读数。这段代码只是简单地为你提供传感器的原始读数。这段代码非常简单，你会理解其中的一切，除了一些代码行，其中包括：

```py
import Adafruit_DHT
```

在代码的这一行中，我们在代码中导入了`Adafruit_DHT`库。这是与DHT11传感器通信的相同库。

```py
sensor = 11 
```

DHT有不同的版本，如DHT11、DHT22等。我们需要告诉程序我们使用的是哪种传感器。因此，我们已经为变量传感器分配了一个值。稍后，你将看到我们将如何使用它：

```py
pin = 4  
```

在这一行中，我们将值4赋给一个名为`pin`的变量。这个变量将用于告诉程序我们已经连接了DHT11的树莓派引脚。

```py
humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
```

在这一行中，我们使用了`Adafruit`库的一个方法，名为`Adafruit_DHT.read_retry()`。现在，它的作用是读取DHT传感器，并将传感器的读数给变量`humidity`和`temperature`。需要注意的一点是，DHT11每2秒更新一次读数。因此，你将在每2秒后收到更新的读数。

一旦这段代码完成，我们就可以确信传感器正在按我们想要的方式工作。最后，是时候将所有传感器整合在一起，制作一个完全智能的机器人了。由于电磁阀、湿度传感器和温度传感器已经连接好，我们所需要做的就是将代码上传到树莓派上，然后看魔法发生。

```py
from time import sleep
from datetime import datetime
import RPi.GPIO as GPIO
import Adafruit_ADS1x15
import Adafruit_DHT
water_valve_pin = 23
sensor = 11
pin = 4
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(water_valve_pin, GPIO.OUT)
Channel =0
GAIN = 1
adc = Adafruit_ADS1x15.ADS1115()
def check_moisture(m):
 adc.start_adc(channel, gain=GAIN)
 moisture_value = adc.get_last_result()
 moisture_value = int(moisture_value/327)
 print moisture_value
 if moisture_value < m:
 GPIO.output(water_valve_pin, GPIO.HIGH)
 sleep(5)
 GPIO.output(water_valve_pin, GPIO.LOW)
 else:
 GPIO.output(water_valve_pin, GPIO.LOW)
while True:
 humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
 H = datetime.now().strftime(‘%H’)
 M = datetime.now().strftime(‘%M’)
 if H == ‘07’ and M <= ‘10’:
 if temperature < 15:
 check_moisture(20)
 elif temperature >= 15 and temperature < 28:
 check_moisture(30)
 elif temperature >= 28:
 check_moisture(40)
 if H == ‘17’ and M <= ‘10’:
 if temperature < 15:

 check_moisture(20)
 elif temperature >= 15 and temperature < 28:
 check_moisture(30)
 elif temperature >= 28:
 check_moisture(40)
```

代码很长，对吧？看起来是这样，但是一旦你逐行编写它，你肯定会明白，它可能比我们迄今为止编写的所有代码都长，但它一点也不复杂。你可能已经理解了大部分程序，但是让我解释一下我们在这里使用的一些新东西：

```py
def check_moisture(m):
  adc.start_adc(channel, gain = GAIN)

moisture_value = adc.get_last_result()
moisture_value = int(moisture_value / 327)
print moisture_value

if moisture_value < m:
  GPIO.output(water_valve_pin, GPIO.HIGH)
  sleep(5)
  GPIO.output(water_valve_pin, GPIO.LOW)
else :
  GPIO.output(water_valve_pin, GPIO.LOW)
```

在这一行中，我们定义了一个名为`check_moisture()`的函数。以前，如果你还记得，当我们制作`check_moisture`函数时，我们基本上是在检查湿度值是否大于或小于20％。如果我们需要检查30％、40％和50％的湿度怎么办？我们会为此制作一个单独的函数吗？

显然不是！我们所做的是向函数传递一个参数，参数基本上是放在函数括号内的变量。现在我们可以为这个变量分配值，例如`check_moisture(30)`-现在在执行该函数时`m`的值将为30。然后，如果再次调用`check_moisture(40)`，那么`m`的值将为40。

现在，你可以看到我们在整个函数中比较`m`的值。

```py
   if moisture_value < m:
```

if语句将检查调用函数时分配的“m”的值。这使我们的工作变得非常简单。

让我们看看程序的其余部分在做什么：

```py
            if temperature < 15:
                check_moisture(20)
```

每当达到所需的时间，它将继续检查温度。如果温度低于15度，它将调用函数`check_moisture`并将参数值设为20。因此，如果湿度低于20％，则会给花园浇水。

```py
 elif temperature >= 15 and temperature < 28:
                check_moisture(30)
```

`elif`或`else if`语句在`if`语句之后使用。通俗地说，这意味着如果前面的`if`语句不成立，它将检查这个`if`语句。因此，在前一行中，它将检查温度是否在15到28摄氏度之间。如果是，它将检查土壤的湿度。在这一行中，函数的参数是30。因此，它将检查湿度是否低于30。如果是，它将给花园供水。

```py
 elif temperature >= 28:
                check_moisture(40)
```

同样，在这行代码中，我们正在检查温度，如果温度等于或超过`28`摄氏度，那么它将把值`40`作为参数传递给函数`check_moisture`。因此，这次它将检查湿度是否达到或超过`28`。

正如您所看到的，现在系统将检查环境温度，并根据此调节植物所需的水量。最好的部分是它是一致的，并将提供植物所需的正确水量。

本章中提到的数值仅为假设值。我强烈建议根据您所在地区和您花园中种植的植物来调整数值，以便系统发挥最佳效果。

# 总结

在本章中，我们涵盖了一些主题，如电磁阀集成和土壤湿度传感器，以构建一个可以自动给您的后院花园浇水的机器人。接下来，我们将介绍电机的基础知识。
