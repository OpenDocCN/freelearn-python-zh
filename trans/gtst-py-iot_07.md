# 第七章：使用 Python 驱动硬件

在本章中，我们将涵盖以下主题：

+   控制 LED

+   响应按钮

+   控制关机按钮

+   GPIO 键盘输入

+   多路复用彩色 LED

+   使用视觉持久性编写消息

# 介绍

树莓派计算机的一个关键特性是它能够直接与其他硬件进行接口。树莓派上的通用输入/输出（GPIO）引脚可以控制各种低级电子设备，从发光二极管（LED）到开关、传感器、电机、伺服和额外的显示器。

本章将重点介绍如何连接树莓派与一些简单的电路，并掌握使用 Python 来控制和响应连接的组件。

树莓派硬件接口由板子一侧的 40 个引脚组成。

GPIO 引脚及其布局将根据您拥有的特定型号略有不同。

树莓派 3、树莓派 2 和树莓派 B+都具有相同的 40 针布局。

树莓派 1 代老款（非 plus 型号）有一个 26 针的引脚，与新款模型的 1-26 针相同。

树莓派 2、树莓派 B+和树莓派 Plus GPIO 引脚（引脚功能）

连接器的布局如上图所示；引脚编号从 GPIO 引脚的引脚 1 开始。

引脚 1 位于最靠近 SD 卡的一端，如下图所示：

树莓派 GPIO 引脚位置

在使用 GPIO 引脚时应当小心，因为它还包括电源引脚（3V3 和 5V），以及地线（GND）引脚。所有的 GPIO 引脚都可以用作标准 GPIO，但其中一些还具有特殊功能；这些被标记并用不同颜色突出显示。

工程师通常使用 3V3 标记来指定原理图中的值，以避免使用可能被忽略的小数位（使用 33V 而不是 3.3V 会对电路造成严重损坏）。同样的方法也可以应用于其他组件的值，比如电阻，例如，1.2K 欧姆可以写成 1K2 欧姆。

TX 和 RX 引脚用于串行通信，借助电压级转换器，信息可以通过串行电缆传输到另一台计算机或设备。

我们还有 SDA 和 SCL 引脚，它们能够支持一种名为 I²C 的双线总线通信协议（树莓派 3 和 Plus 板上有两个 I²C 通道：通道 1 ARM，用于通用用途，通道 0 VC，通常用于识别 HAT 模块上连接的硬件）。还有 SPI MOSI、SPI MISO、SPI SCLK、SPI CE0 和 SPI CE1 引脚，支持另一种名为 SPI 的高速数据总线协议。最后，我们有 PWM0/1 引脚，允许生成脉冲宽度调制信号，对于伺服和生成模拟信号非常有用。

然而，在本章中，我们将专注于使用标准的 GPIO 功能。GPIO 引脚布局如下图所示：

树莓派 GPIO 引脚（GPIO.BOARD 和 GPIO.BCM）

树莓派 Rev 2（2014 年 7 月之前）与树莓派 2 GPIO 布局相比有以下不同：

+   26 个 GPIO 引脚的引脚头（匹配前 26 个引脚）。

+   引脚头旁边的另一组八个孔（P5）。详细信息如下：

树莓派 Rev 2 P5 GPIO 引脚

+   原始的树莓派 Rev 1（2012 年 10 月之前）总共只有 26 个 GPIO 引脚（匹配当前树莓派的前 26 个引脚），除了以下细节：

![](img/ec7fdb0c-b61f-4108-b924-a6162d59bf0e.png)树莓派 Rev 1 GPIO 引脚头的差异

`RPi.GPIO`库可以使用两种系统之一引用树莓派上的引脚。中间显示的数字是引脚的物理位置，也是在**GPIO.BOARD**模式下`RPi.GPIO`库引用的数字。外部的数字（**GPIO.BCM**）是处理器物理端口的实际引用数字，指示哪些引脚被连接（这就是为什么它们没有特定的顺序）。当模式设置为**GPIO.BCM**时使用它们，并且它们允许控制 GPIO 引脚以及连接到其他 GPIO 线的任何外围设备。这包括 BCM GPIO 4 上的附加摄像头上的 LED 和板上的状态 LED。但是，这也可能包括用于读/写 SD 卡的 GPIO 线，如果干扰会导致严重错误。

如果您使用其他编程语言访问 GPIO 引脚，编号方案可能会有所不同，因此如果您了解 BCM GPIO 引用，将会很有帮助，它们指的是处理器的物理 GPIO 端口。

请务必查看附录*硬件和软件清单*，其中列出了本章中使用的所有物品以及您可以从哪里获得它们。

# 控制 LED

硬件上的`hello world`等同于 LED 闪烁，这是一个很好的测试，可以确保一切正常工作，并且你已经正确地连接了它。为了让它更有趣，我建议使用**红色、蓝色和绿色**（RGB）LED，但如果你只有单独的 LED 也可以。

# 准备工作

你将需要以下设备：

+   4 x 杜邦母对公补丁线

+   迷你面包板（170 个连接点）或更大的面包板

+   RGB LED（共阴）/3 个标准 LED（最好是红色、绿色和蓝色）

+   面包板线（实心线）

+   3 x 470 欧姆电阻

前面提到的每个组件成本都不会太高，并且可以在其他项目中重复使用。面包板是一个特别有用的物品，可以让你在不需要焊接的情况下尝试自己的电路：

![](img/f9a79a3b-7ca0-4c4e-b4e6-a4d6ff9c69fb.png)RGB LED、标准 LED 和 RGB 电路的图表

以下图表显示了面包板电路：

![](img/88293cf3-b0ab-45fc-999f-af4c445e2e1d.png)连接到 GPIO 引脚的 RGB LED/标准 LED 的接线图有几种不同类型的 RGB LED 可用，因此请检查您组件的数据表以确认引脚顺序和类型。有些是 RGB 的，所以确保你按照相应的方式连接引脚，或者在代码中调整`RGB_`引脚设置。你也可以获得共阳极变种，这将需要阳极连接到 3V3（GPIO 引脚 1）才能点亮（它们还需要将`RGB_ENABLE`和`RGB_DISABLE`设置为`0`和`1`）。

本书的面包板和组件图是使用一个名为**Fritzing**（[www.fritzing.org](http://www.fritzing.org)）的免费工具创建的；它非常适合规划您自己的树莓派项目。

# 如何做...

1.  创建`ledtest.py`脚本如下：

```py
#!/usr/bin/python3 
#ledtest.py 
import time 
import RPi.GPIO as GPIO 
# RGB LED module 
#HARDWARE SETUP 
# GPIO 
# 2[======XRG=B==]26[=======]40 
# 1[=============]25[=======]39 
# X=GND R=Red G=Green B=Blue  
#Setup Active States 
#Common Cathode RGB-LED (Cathode=Active Low) 
RGB_ENABLE = 1; RGB_DISABLE = 0 

#LED CONFIG - Set GPIO Ports 
RGB_RED = 16; RGB_GREEN = 18; RGB_BLUE = 22 
RGB = [RGB_RED,RGB_GREEN,RGB_BLUE] 

def led_setup(): 
  #Setup the wiring 
  GPIO.setmode(GPIO.BOARD) 
  #Setup Ports 
  for val in RGB: 
    GPIO.setup(val,GPIO.OUT) 

def main(): 
  led_setup() 
  for val in RGB: 
    GPIO.output(val,RGB_ENABLE) 
    print("LED ON") 
    time.sleep(5) 
    GPIO.output(val,RGB_DISABLE) 
    print("LED OFF") 

try: 
  main() 
finally: 
  GPIO.cleanup() 
  print("Closed Everything. END") 
#End
```

1.  `RPi.GPIO`库将需要`sudo`权限来访问 GPIO 引脚硬件，因此您需要使用以下命令运行脚本：

```py
sudo python3 ledtest.py  
```

运行脚本时，您应该看到 LED 的红色、绿色和蓝色部分（或者如果您使用单独的 LED，则分别点亮）。如果没有，请仔细检查您的接线或确认 LED 是否正常工作，方法是暂时将红色、绿色或蓝色线连接到 3V3 引脚（GPIO 引脚 1）。

大多数与硬件相关的脚本都需要`sudo`命令，因为用户通常不会直接在这么低的层次上控制硬件。例如，设置或清除作为 SD 卡控制器一部分的控制引脚可能会损坏正在写入的数据。因此，出于安全目的，需要超级用户权限，以防止程序意外（或恶意）使用硬件。

# 工作原理...

要使用 Python 访问 GPIO 引脚，我们导入`RPi.GPIO`库，该库允许通过模块函数直接控制引脚。我们还需要`time`模块来暂停程序一定数量的秒。

然后，我们为 LED 的接线和激活状态定义值（请参阅本食谱的*有更多...*部分中的*控制 GPIO 电流*段）。

在程序使用 GPIO 引脚之前，我们需要通过指定编号方法（`GPIO.BOARD`）和方向（`GPIO.OUT`或`GPIO.IN`）来设置它们（在这种情况下，我们将所有 RGB 引脚设置为输出）。如果引脚配置为输出，我们将能够设置引脚状态；同样，如果它配置为输入，我们将能够读取引脚状态。

接下来，我们使用`GPIO.ouput()`来控制引脚，指定 GPIO 引脚的编号和我们希望它处于的状态（`1` = 高/开启，`0` = 低/关闭）。我们打开每个 LED，等待五秒，然后关闭它。

最后，我们使用`GPIO.cleanup()`将 GPIO 引脚恢复到它们的原始默认状态，并释放对引脚的控制，以供其他程序使用。

# 有更多...

在树莓派上使用 GPIO 引脚必须小心，因为这些引脚直接连接到树莓派的主处理器，没有额外的保护。必须小心使用，因为任何错误的接线可能会损坏树莓派处理器，并导致其完全停止工作。

或者，您可以使用许多直接插入 GPIO 引脚排针的模块之一（减少接线错误的机会）：

例如，Pi-Stop 是一个简单的预制 LED 板，模拟了一组交通信号灯，旨在成为那些对控制硬件感兴趣但又想避免损坏树莓派的人的一个过渡阶段。掌握了基础知识后，它也是一个出色的指示器，有助于调试。

只需确保您在`ledtest.py`脚本中更新`LED CONFIG`引脚引用，以引用您使用的硬件的引脚布局和位置。

![](img/ad34ee37-0365-4a28-a71c-578b0daa28e3.png)

请参阅附录中的*硬件和软件清单*，了解树莓派硬件零售商的清单。

# 控制 GPIO 电流

每个 GPIO 引脚在烧毁之前只能处理一定电流（单个引脚最大 16mA，总共 30mA），同样，RGB LED 的电流应限制在 100mA 以下。通过在 LED 之前或之后添加电阻，我们将能够限制通过 LED 的电流并控制其亮度（更大的电流将使 LED 更亮）。

由于我们可能希望同时点亮多个 LED，因此我们通常会尽量将电流设置得尽可能低，同时仍然提供足够的功率点亮 LED。

我们可以使用欧姆定律来告诉我们需要多少电阻来提供特定的电流。该定律如下图所示：

![](img/f3fc16b6-3b93-46ac-9491-565b179ecd6a.png)欧姆定律：电路中电流、电阻和电压之间的关系

我们将以最小电流（3mA）和最大电流（16mA）为目标，同时仍然从每个 LED 产生相当明亮的光。为了获得 RGB LED 的平衡输出，我测试了不同的电阻，直到它们提供了接近白光（通过卡片查看）。每个 LED 选择了 470 欧姆的电阻（您的 LED 可能略有不同）：

![](img/64bf4bbc-c955-45b6-9567-4372306aa5d1.png)需要电阻器来限制通过 LED 的电流

电阻器上的电压等于 GPIO 电压（**Vgpio** = 3.3V）减去特定 LED 的电压降（**Vfwd**）；然后我们可以使用这个电阻来计算每个 LED 使用的电流，如下面的公式所示：

![](img/6a89db2a-ff7b-4579-b310-230c7c9c9ca4.png)我们可以计算每个 LED 的电流

# 响应按钮

许多使用树莓派的应用程序要求在不需要连接键盘和屏幕的情况下激活操作。 GPIO 引脚为树莓派提供了一种优秀的方式，使其可以通过您自己的按钮和开关进行控制，而无需鼠标/键盘和屏幕。

# 准备工作

您将需要以下设备：

+   2 x DuPont 母对公跳线

+   迷你面包板（170 个连接点）或更大的面包板

+   按钮开关（瞬时闭合）或导线连接以打开/关闭电路

+   面包板导线（实心线）

+   1K 欧姆电阻器

开关如下图所示：

![](img/325f711c-c617-4bcb-b66a-1b17ab3f2080.png)按钮开关和其他类型的开关以下示例中使用的开关是**单极，单刀**（**SPST**），瞬时闭合，按钮开关。**单极**（**SP**）意味着有一组使连接的触点。在这里使用的按钮开关的情况下，每侧的腿与中间的单极开关连接在一起。**双极**（**DP**）开关的作用就像单极开关，只是两侧在电上是分开的，允许您同时打开/关闭两个独立的组件。

**单刀**（**ST**）意味着开关将仅在一个位置进行连接；另一侧将保持开放。**双刀**（**DT**）意味着开关的两个位置将连接到不同的部分。

**瞬时闭合**意味着按下按钮时将关闭开关，并在释放时自动打开。**锁定**按钮开关将保持关闭状态，直到再次按下。

# 尝试使用树莓派的扬声器或耳机

![](img/c54871b1-71d9-4932-9f4b-d212a53578d3.png)按钮电路的布局

在此示例中，我们将使用声音，因此您还需要将扬声器或耳机连接到树莓派的音频插孔。

您需要使用以下命令安装名为`flite`的程序，这将让我们让树莓派说话：

```py
sudo apt-get install flite  
```

安装后，您可以使用以下命令进行测试：

```py
sudo flite -t "hello I can talk"  
```

如果太安静（或太吵），您可以使用以下命令调整音量（0-100％）：

```py
amixer set PCM 100%  
```

# 如何做...

创建`btntest.py`脚本如下：

```py
#!/usr/bin/python3 
#btntest.py 
import time 
import os 
import RPi.GPIO as GPIO 
#HARDWARE SETUP 
# GPIO 
# 2[==X==1=======]26[=======]40 
# 1[=============]25[=======]39 
#Button Config 
BTN = 12 

def gpio_setup(): 
  #Setup the wiring 
  GPIO.setmode(GPIO.BOARD) 
  #Setup Ports 
  GPIO.setup(BTN,GPIO.IN,pull_up_down=GPIO.PUD_UP) 

def main(): 
  gpio_setup() 
  count=0 
  btn_closed = True 
  while True: 
    btn_val = GPIO.input(BTN) 
    if btn_val and btn_closed: 
       print("OPEN") 
       btn_closed=False 
    elif btn_val==False and btn_closed==False: 
       count+=1 
       print("CLOSE %s" % count) 
       os.system("flite -t '%s'" % count) 
       btn_closed=True 
    time.sleep(0.1) 

try: 
  main() 
finally: 
  GPIO.cleanup() 
  print("Closed Everything. END") 
#End 
```

# 它是如何工作的...

与上一个示例一样，我们根据需要设置 GPIO 引脚，但这次是作为输入，并且还启用了内部上拉电阻器（有关更多信息，请参阅本示例的*更多内容...*部分中的*上拉和下拉电阻器电路*）使用以下代码：

```py
GPIO.setup(BTN,GPIO.IN,pull_up_down=GPIO.PUD_UP) 
```

在设置了 GPIO 引脚之后，我们创建一个循环，将不断检查`BTN`的状态，使用`GPIO.input()`。如果返回的值为`false`，则表示通过开关将引脚连接到 0V（地），我们将使用`flite`每次按下按钮时为我们大声计数。

由于我们在`try`/`finally`条件中调用了主函数，即使我们使用*Ctrl* + *Z*关闭程序，它仍将调用`GPIO.cleanup()`。

我们在循环中使用短延迟；这可以确保忽略开关上的接触产生的任何噪音。这是因为当我们按下按钮时，按下或释放时并不总是完美接触，如果我们再次按下它，可能会产生多个触发。这被称为**软件去抖动**；我们在这里忽略了信号中的弹跳。

# 更多内容...

树莓派 GPIO 引脚必须小心使用；用于输入的电压应该是

在特定范围内，并且从中抽取的任何电流应该最小化使用

保护电阻。

# 安全电压

我们必须确保只连接在 0（地）和 3V3 之间的输入。一些处理器使用 0V 到 5V 之间的电压，因此需要额外的组件才能安全地与它们接口。除非确定安全，否则永远不要连接使用 5V 的输入或组件，否则会损坏树莓派的 GPIO 端口。

# 上拉和下拉电阻电路

先前的代码设置了 GPIO 引脚使用内部上拉电阻。如果 GPIO 引脚上没有上拉电阻（或下拉电阻），电压可以在 3V3 和 0V 之间自由浮动，实际逻辑状态保持不确定（有时为 1，有时为 0）。

树莓派的内部上拉电阻为 50K 欧姆至 65K 欧姆，下拉电阻为 50K 欧姆至 65K 欧姆。外部上拉/下拉电阻通常用于 GPIO 电路（如下图所示），通常使用 10K 欧姆或更大的电阻出于类似的原因（当它们不活动时提供非常小的电流吸收）。

上拉电阻允许通过 GPIO 引脚流动少量电流，并且在开关未按下时提供高电压。当按下开关时，小电流被流向 0V 的大电流所取代，因此我们在 GPIO 引脚上得到低电压。开关在按下时为活动低电平和逻辑 0。它的工作原理如下图所示：

![](img/29c4c4eb-9bc8-4f96-bd07-7c8206af9a32.png)上拉电阻电路

下拉电阻的工作方式相同，只是开关为活动高电平（按下时 GPIO 引脚为逻辑 1）。它的工作原理如下图所示：

![](img/b8708774-f896-4447-acc1-a829863676d1.png)下拉电阻电路

# 保护电阻

除了开关外，电路还包括与开关串联的电阻，以保护 GPIO 引脚，如下图所示：

![](img/f47a0106-2712-472d-a88b-7efea679d893.png)GPIO 保护限流电阻

保护电阻的目的是保护 GPIO 引脚，如果它被意外设置为输出而不是输入。例如，假设我们的开关连接在 GPIO 和地之间。现在 GPIO 引脚被设置为输出并打开（驱动到 3V3），一旦我们按下开关，没有电阻的情况下，GPIO 引脚将直接连接到 0V。 GPIO 仍然会尝试将其驱动到 3V3；这将导致 GPIO 引脚烧毁（因为它将使用太多电流来驱动引脚到高状态）。如果我们在这里使用 1K 欧姆电阻，引脚可以使用可接受的电流驱动高（I = V/R = 3.3/1K = 3.3 毫安）。

# 受控关机按钮

树莓派应该始终正确关机，以避免 SD 卡损坏（在对卡进行写操作时断电）。如果您没有连接键盘或屏幕（可能正在运行自动化程序或通过网络远程控制），这可能会造成问题，因为您无法输入命令或查看您正在做什么。通过添加我们自己的按钮和 LED 指示灯，我们可以轻松地命令关机和重启，然后再次启动以指示系统处于活动状态。

# 准备工作

您将需要以下设备：

+   3 x DuPont 母对公跳线

+   迷你面包板（170 个连接点）或更大的面包板

+   按钮开关（瞬时闭合）

+   通用 LED

+   2 x 470 欧姆电阻

+   面包板导线（实心）

关机电路的整个布局将如下图所示：

![](img/32384f24-fc03-40a6-953b-a4e9a263e4ab.png)受控关机电路布局

# 如何操作...

1.  创建`shtdwn.py`脚本如下：

```py
#!/usr/bin/python3 
#shtdwn.py 
import time 
import RPi.GPIO as GPIO 
import os 

# Shutdown Script 
DEBUG=True #Simulate Only 
SNDON=True 
#HARDWARE SETUP 
# GPIO 
# 2[==X==L=======]26[=======]40 
# 1[===1=========]25[=======]39 

#BTN CONFIG - Set GPIO Ports 
GPIO_MODE=GPIO.BOARD 
SHTDWN_BTN = 7 #1 
LED = 12       #L 

def gpio_setup(): 
  #Setup the wiring 
  GPIO.setmode(GPIO_MODE) 
  #Setup Ports 
  GPIO.setup(SHTDWN_BTN,GPIO.IN,pull_up_down=GPIO.PUD_UP) 
  GPIO.setup(LED,GPIO.OUT) 

def doShutdown(): 
  if(DEBUG):print("Press detected") 
  time.sleep(3) 
  if GPIO.input(SHTDWN_BTN): 
    if(DEBUG):print("Ignore the shutdown (<3sec)") 
  else: 
    if(DEBUG):print ("Would shutdown the RPi Now") 
    GPIO.output(LED,0) 
    time.sleep(0.5) 
    GPIO.output(LED,1) 
    if(SNDON):os.system("flite -t 'Warning commencing power down 3 2 1'") 
    if(DEBUG==False):os.system("sudo shutdown -h now") 
    if(DEBUG):GPIO.cleanup() 
    if(DEBUG):exit() 

def main(): 
  gpio_setup() 
  GPIO.output(LED,1) 
  while True: 
    if(DEBUG):print("Waiting for >3sec button press") 
    if GPIO.input(SHTDWN_BTN)==False: 
       doShutdown() 
    time.sleep(1) 

try: 
  main() 
finally: 
  GPIO.cleanup() 
  print("Closed Everything. END") 
#End
```

1.  要使这个脚本自动运行（一旦我们测试过它），我们可以将脚本放在`~/bin`中（如果只想复制它，可以使用`cp`而不是`mv`），并使用以下代码将其添加到`crontab`中：

```py
mkdir ~/bin 
mv shtdwn.py ~/bin/shtdwn.py  
crontab -e 
```

1.  在文件末尾，我们添加以下代码：

```py
@reboot sudo python3 ~/bin/shtdwn.py 
```

# 它是如何工作的...

这次，当我们设置 GPIO 引脚时，我们将与关机按钮连接的引脚定义为输入，与 LED 连接的引脚定义为输出。我们打开 LED 以指示系统正在运行。

通过将`DEBUG`标志设置为`True`，我们可以测试脚本的功能，而不会导致实际关闭（通过读取终端消息）；我们只需要确保在实际使用脚本时将`DEBUG`设置为`False`。

我们进入一个`while`循环，并每秒检查引脚，以查看 GPIO 引脚是否设置为`LOW`（即检查开关是否被按下）；如果是，我们就进入`doShutdown()`函数。

程序将等待三秒，然后再次测试按钮是否仍然被按下。如果按钮不再被按下，我们将返回到之前的`while`循环。但是，如果在三秒后它仍然被按下，程序将闪烁 LED 并触发关闭（还会使用`flite`提供音频警告）。

当我们对脚本的运行状态感到满意时，我们可以禁用`DEBUG`标志（将其设置为`False`），并将脚本添加到`crontab`中。`crontab`是一个在后台运行的特殊程序，允许我们在系统启动时（`@reboot`）安排程序和操作的特定时间、日期或周期性。这使得脚本可以在每次树莓派上电时自动启动。当我们按住关机按钮超过三秒时，它会安全地关闭系统并进入低功耗状态（LED 在此之前会关闭，表明很快就可以拔掉电源）。要重新启动树莓派，我们简单地拔掉电源；这将重新启动系统，当树莓派加载完成时，LED 会亮起。

# 还有更多...

我们可以通过添加额外的功能并利用额外的 GPIO 连接（如果可用）来进一步扩展这个示例。

# 重置和重新启动树莓派

树莓派上有用于安装复位标头的孔（在树莓派 3/2 上标有**RUN**，在树莓派 1 型 A 和 B Rev 2 上标有**P6**）。复位引脚允许使用按钮而不是每次都拔掉微型 USB 连接器来重置设备的电源：

![](img/842a26a3-f0cc-4e68-993c-b5b9fda3ab38.png)树莓派复位标头-左边是树莓派 A/B 型（Rev2），右边是树莓派 3

要使用它，您需要将一根导线或引脚排焊接到树莓派上，并连接一个按钮（或每次在两个孔之间短暂触碰一根导线）。或者，我们可以扩展我们之前的电路，如下图所示：

![](img/7f05b9ba-5dc4-4c35-a484-fbb8c67ff403.png)受控关闭电路布局和复位按钮

我们可以将这个额外的按钮添加到我们的电路中，它可以连接到复位标头（这是树莓派 3 上最靠近中间的孔，其他型号上最靠近边缘的孔）。当暂时将此引脚拉低连接到地（例如旁边的孔或 GPIO 标头的第 6 引脚等其他地点），将重置树莓派并允许它在关闭后再次启动。

# 添加额外功能

由于现在脚本一直监视关机按钮，我们可以同时添加额外的按钮/开关/跳线来监视。这将允许我们通过改变输入来触发特定程序或设置特定状态。以下示例允许我们轻松地在自动 DHCP 网络（默认网络设置）和使用直接 IP 地址之间进行切换，就像第一章“使用树莓派 3 计算机入门”中的“直接连接到笔记本电脑或计算机”配方中使用的那样。

将以下组件添加到上一个电路中：

+   一个 470 欧姆电阻

+   两个带跳线连接器的引脚头（或者，可选地，一个开关）

+   面包板导线（实心线）

在添加了上述组件之后，我们的受控关机电路现在如下所示：

![](img/16390e01-579c-4baa-9947-a171dfde83ac.png)受控关机电路布局、复位按钮和跳线引脚

在上一个脚本中，我们添加了一个额外的输入来检测`LAN_SWA`引脚的状态（我们添加到电路中的跳线引脚），使用以下代码：

```py
LAN_SWA = 11    #2 
```

确保在`gpio_setup()`函数中设置为输入（带上拉电阻）使用以下代码：

```py
GPIO.setup(LAN_SWA,GPIO.IN,pull_up_down=GPIO.PUD_UP) 
```

添加一个新的功能来在 LAN 模式之间切换并读取新的 IP 地址。`doChangeLAN()`函数检查`LAN_SWA`引脚的状态是否自上次调用以来发生了变化，如果是，则将网络适配器设置为 DHCP，或者相应地设置直接 LAN 设置（如果可用，则使用`flite`来朗读新的 IP 设置）。最后，设置 LAN 为直接连接会导致 LED 在该模式激活时缓慢闪烁。使用以下代码来实现这一点：

```py
def doChangeLAN(direct): 
  if(DEBUG):print("Direct LAN: %s" % direct) 
  if GPIO.input(LAN_SWA) and direct==True: 
    if(DEBUG):print("LAN Switch OFF") 
    cmd="sudo dhclient eth0" 
    direct=False 
    GPIO.output(LED,1) 
  elif GPIO.input(LAN_SWA)==False and direct==False: 
    if(DEBUG):print("LAN Switch ON") 
    cmd="sudo ifconfig eth0 169.254.69.69" 
    direct=True 
  else: 
    return direct 
  if(DEBUG==False):os.system(cmd) 
  if(SNDON):os.system("hostname -I | flite") 
  return direct 
```

添加另一个函数`flashled()`，每次调用时切换 LED 的状态。该函数的代码如下：

```py
def flashled(ledon): 
  if ledon: 
    ledon=False 
  else: 
    ledon=True 
  GPIO.output(LED,ledon) 
  return ledon
```

最后，我们调整主循环，也调用`doChangeLAN()`，并使用结果决定是否使用`ledon`调用`flashled()`来跟踪 LED 的上一个状态。`main()`函数现在应该更新如下：

```py
def main(): 
  gpio_setup() 
  GPIO.output(LED,1) 
  directlan=False 
  ledon=True 
  while True: 
    if(DEBUG):print("Waiting for >3sec button press") 
    if GPIO.input(SHTDWN_BTN)==False: 
       doShutdown() 
    directlan= doChangeLAN(directlan) 
    if directlan: 
      flashled(ledon) 
    time.sleep(1) 
```

# GPIO 键盘输入

我们已经看到了如何监视 GPIO 上的输入来启动应用程序和控制树莓派；然而，有时我们需要控制第三方程序。使用`uInput`库，我们可以模拟键盘（甚至鼠标移动）来控制任何程序，使用我们自己的自定义硬件。

有关使用`uInput`的更多信息，请访问[`tjjr.fi/sw/python-uinput/`](http://tjjr.fi/sw/python-uinput/)。

# 准备工作

执行以下步骤安装`uInput`：

1.  首先，我们需要下载`uInput`。

您需要使用以下命令从 GitHub 下载`uInput` Python 库（约 50 KB）：

```py
wget https://github.com/tuomasjjrasanen/python-uinput/archive/master.zip
unzip master.zip

```

该库将解压缩到一个名为`python-uinput-master`的目录中。

1.  完成后，可以使用以下命令删除 ZIP 文件：

```py
rm master.zip  
```

1.  使用以下命令安装所需的软件包（如果已经安装了它们，`apt-get`命令将忽略它们）：

```py
sudo apt-get install python3-setuptools python3-dev
sudo apt-get install libudev-dev  
```

1.  使用以下命令编译和安装`uInput`：

```py
cd python-uinput-master
sudo python3 setup.py install  
```

1.  最后，使用以下命令加载新的`uinput`内核模块：

```py
sudo modprobe uinput  
```

为了确保在启动时加载，我们可以使用以下命令将`uinput`添加到`modules`文件中：

```py
sudo nano /etc/modules  
```

在文件中新建一行并保存（*Ctrl* + *X*, *Y*）。

1.  使用以下设备创建以下电路：

+   面包板（半尺寸或更大）

+   7 根 DuPont 母对公排线

+   六个按钮

+   6 个 470 欧姆电阻

+   面包板导线（实心线）

![](img/a4eac434-a135-49ff-bdb7-e4b98accf7ae.png)GPIO 键盘电路布局

键盘电路也可以通过将组件焊接到 Vero 原型板（也称为条板）中，制成永久电路，如下图所示：

![](img/121e0320-e6c0-4ade-b143-64e253df5e16.png)GPIO 键盘 Pi 硬件模块这个电路可以从[PiHardware.com](http://pihardware.com/)购买成套焊接套件。

1.  通过将适当的按钮与适当的引脚相匹配，将电路连接到树莓派 GPIO 引脚，如下表所示：

|  | **按钮** | **GPIO 引脚** |
| --- | --- | --- |
| GND |  | 6 |
| v | B_DOWN | 22 |
| < | B_LEFT | 18 |
| ^ | B_UP | 15 |
| > | B_RIGHT | 13 |
| 1 | B_1 | 11 |
| 2 | B_2 | 7 |

# 如何做...

创建一个名为`gpiokeys.py`的脚本，如下所示：

```py
#!/usr/bin/python3 
#gpiokeys.py 
import time 
import RPi.GPIO as GPIO 
import uinput 

#HARDWARE SETUP 
# GPIO 
# 2[==G=====<=V==]26[=======]40 
# 1[===2=1>^=====]25[=======]39 
B_DOWN  = 22    #V 
B_LEFT  = 18   #< 
B_UP    = 15   #^ 
B_RIGHT = 13   #> 
B_1  = 11   #1 
B_2  = 7   #2 

DEBUG=True 
BTN = [B_UP,B_DOWN,B_LEFT,B_RIGHT,B_1,B_2] 
MSG = ["UP","DOWN","LEFT","RIGHT","1","2"] 

#Setup the DPad module pins and pull-ups 
def dpad_setup(): 
  #Set up the wiring 
  GPIO.setmode(GPIO.BOARD) 
  # Setup BTN Ports as INPUTS 
  for val in BTN: 
    # set up GPIO input with pull-up control 
    #(pull_up_down can be: 
    #    PUD_OFF, PUD_UP or PUD_DOWN, default PUD_OFF) 
    GPIO.setup(val, GPIO.IN, pull_up_down=GPIO.PUD_UP) 

def main(): 
  #Setup uinput 
  events = (uinput.KEY_UP,uinput.KEY_DOWN,uinput.KEY_LEFT, 
           uinput.KEY_RIGHT,uinput.KEY_ENTER,uinput.KEY_ENTER) 
  device = uinput.Device(events) 
  time.sleep(2) # seconds 
  dpad_setup() 
  print("DPad Ready!") 

  btn_state=[False,False,False,False,False,False] 
  key_state=[False,False,False,False,False,False] 
  while True: 
    #Catch all the buttons pressed before pressing the related keys 
    for idx, val in enumerate(BTN): 
      if GPIO.input(val) == False: 
        btn_state[idx]=True 
      else: 
        btn_state[idx]=False 

    #Perform the button presses/releases (but only change state once) 
    for idx, val in enumerate(btn_state): 
      if val == True and key_state[idx] == False: 
        if DEBUG:print (str(val) + ":" + MSG[idx]) 
        device.emit(events[idx], 1) # Press. 
        key_state[idx]=True 
      elif val == False and key_state[idx] == True: 
        if DEBUG:print (str(val) + ":!" + MSG[idx]) 
        device.emit(events[idx], 0) # Release. 
        key_state[idx]=False 

    time.sleep(.1) 

try: 
  main() 
finally: 
  GPIO.cleanup() 
#End 
```

# 它是如何工作的...

首先，我们导入`uinput`并定义键盘按钮的接线。对于`BTN`中的每个按钮，我们将它们启用为输入，并启用内部上拉。

接下来，我们设置`uinput`，定义我们想要模拟的键，并将它们添加到`uinput.Device()`函数中。我们等待几秒钟，以便`uinput`初始化，设置初始按钮和键状态，并启动我们的`main`循环。

`main`循环分为两个部分：第一部分检查按钮并记录`btn_state`中的状态，第二部分将`btn_state`与当前的`key_state`数组进行比较。这样，我们可以检测到`btn_state`的变化，并调用`device.emit()`来切换键的状态。

为了让我们能够在后台运行此脚本，我们可以使用`&`运行它，如下所示

以下命令：

```py
sudo python3 gpiokeys.py &  
```

`&`字符允许命令在后台运行，因此我们可以继续使用命令行运行其他程序。您可以使用`fg`将其带回前台，或者如果有多个命令正在运行，则可以使用`%1`，`%2`等。使用`jobs`获取列表。

您甚至可以通过按下*Ctrl* + *Z*将进程/程序暂停以进入命令提示符，然后使用`bg`恢复它（这将使其在后台运行）。

# 更多信息...

我们可以使用`uinput`来为其他程序提供硬件控制，包括那些需要鼠标输入的程序。

# 生成其他按键组合

您可以在文件中创建几种不同的键映射以支持不同的程序。例如，`events_z80`键映射对于像**Fuse**这样的光谱模拟器非常有用（浏览[`raspi.tv/2012/how-to-install-fuse-zx-spectrum-emulator-on-raspberry-pi`](http://raspi.tv/2012/how-to-install-fuse-zx-spectrum-emulator-on-raspberry-pi)获取更多详细信息）。`events_omx`键映射适用于使用以下命令控制通过 OMXPlayer 播放的视频：

```py
omxplayer filename.mp4  
```

您可以使用`-k`参数获取`omxplayer`支持的键列表。

用新的键映射替换定义`events`列表的行，并通过以下代码将它们分配给事件来选择不同的键：

```py
events_dpad = (uinput.KEY_UP,uinput.KEY_DOWN,uinput.KEY_LEFT, 
              uinput.KEY_RIGHT,uinput.KEY_ENTER,uinput.KEY_ENTER) 
events_z80 = (uinput.KEY_Q,uinput.KEY_A,uinput.KEY_O, 
             uinput.KEY_P,uinput.KEY_M,uinput.KEY_ENTER) 
events_omx = (uinput.KEY_EQUAL,uinput.KEY_MINUS,uinput.KEY_LEFT, 
             uinput.KEY_RIGHT,uinput.KEY_P,uinput.KEY_Q) 
```

您可以在`input.h`文件中找到所有的`KEY`定义；您可以使用`less`命令查看它（按*Q*退出），如下所示：

```py
less /usr/include/linux/input.h  
```

# 模拟鼠标事件

`uinput`库可以模拟鼠标和操纵杆事件，以及键盘按键。要使用按钮模拟鼠标，我们可以调整脚本以使用鼠标事件（以及定义`mousemove`来设置移动的步长），使用以下代码：

```py
MSG = ["M_UP","M_DOWN","M_LEFT","M_RIGHT","1","Enter"] 
events_mouse=(uinput.REL_Y,uinput.REL_Y, uinput.REL_X, 
             uinput.REL_X,uinput.BTN_LEFT,uinput.BTN_RIGHT) 
mousemove=1 
```

我们还需要修改按钮处理以提供连续移动，因为我们不需要跟踪鼠标键的状态。为此，请使用以下代码：

```py
#Perform the button presses/releases 
#(but only change state once) 
for idx, val in enumerate(btn_state): 
  if MSG[idx] == "M_UP" or MSG[idx] == "M_LEFT": 
    state = -mousemove 
  else: 
    state = mousemove 
  if val == True: 
    device.emit(events[idx], state) # Press. 
  elif val == False: 
    device.emit(events[idx], 0) # Release. 
time.sleep(0.01) 
```

# 多路复用的彩色 LED

本章的下一个示例演示了一些看似简单的硬件如果通过软件控制可以产生一些令人印象深刻的结果。为此，我们将回到使用 RGB LED。我们将使用五个 RGB LED，这些 LED 被布线，以便我们只需要使用八个 GPIO 引脚来控制它们的红色、绿色和蓝色元素，使用一种称为**硬件多路复用**的方法（请参阅本食谱的*硬件多路复用*子部分中的*更多信息*部分）。

# 准备工作

您将需要以下图片中显示的 RGB LED 模块：

![](img/f240314f-3d20-4e18-839d-b4bea49a7025.png)PiHardware.com 的 RGB LED 模块

正如您在上面的照片中所看到的，来自[`pihardware.com/`](http://pihardware.com/)的 RGB LED 模块带有 GPIO 引脚和杜邦母对母电缆用于连接。虽然有两组从 1 到 5 标记的引脚，但只需要连接一侧。

或者，您可以使用五个共阳极 RGB LED、3 个 470 欧姆电阻和一个 Vero 原型板（或大型面包板）来重新创建自己的电路。电路将如下图所示：

![](img/2e014247-64dd-4885-a536-b51c3b1454b0.png)RGB LED 模块的电路图严格来说，我们应该在这个电路中使用 15 个电阻（每个 RGB LED 元件一个），这样可以避免 LED 共用同一个电阻的干扰，并且在一起开启时也会延长 LED 的寿命。然而，使用这种方法只有轻微的优势，特别是因为我们打算独立驱动每个 RGB LED，以实现多种颜色效果。

您需要将电路连接到树莓派 GPIO 引脚头，连接方式如下：

| **RGB LED** |  |  |  |  |  | 1 |  | 2 | 3 |  | 4 |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Rpi GPIO 引脚** | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 | 32 | 34 | 36 | 38 | 40 |
| **Rpi GPIO 引脚** | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 | 21 | 23 | 25 | 27 | 29 | 31 | 33 | 35 | 37 | 39 |
| **RGB LED** |  |  |  | 5 |  | R | G | B |  |  |  |  |  |  |  |  |  |  |  |  |

# 如何做到这一点...

创建`rgbled.py`脚本，并执行以下步骤：

1.  导入所有所需的模块，并使用以下代码定义要使用的值： 

```py
#!/usr/bin/python3 
#rgbled.py 
import time 
import RPi.GPIO as GPIO 

#Setup Active states 
#Common Cathode RGB-LEDs (Cathode=Active Low) 
LED_ENABLE = 0; LED_DISABLE = 1 
RGB_ENABLE = 1; RGB_DISABLE = 0 
#HARDWARE SETUP 
# GPIO 
# 2[=====1=23=4==]26[=======]40 
# 1[===5=RGB=====]25[=======]39 
#LED CONFIG - Set GPIO Ports 
LED1 = 12; LED2 = 16; LED3 = 18; LED4 = 22; LED5 = 7 
LED = [LED1,LED2,LED3,LED4,LED5] 
RGB_RED = 11; RGB_GREEN = 13; RGB_BLUE = 15 
RGB = [RGB_RED,RGB_GREEN,RGB_BLUE] 
#Mixed Colors 
RGB_CYAN = [RGB_GREEN,RGB_BLUE] 
RGB_MAGENTA = [RGB_RED,RGB_BLUE] 
RGB_YELLOW = [RGB_RED,RGB_GREEN] 
RGB_WHITE = [RGB_RED,RGB_GREEN,RGB_BLUE] 
RGB_LIST = [RGB_RED,RGB_GREEN,RGB_BLUE,RGB_CYAN, 
            RGB_MAGENTA,RGB_YELLOW,RGB_WHITE] 
```

1.  定义使用以下代码设置 GPIO 引脚的函数：

```py
def led_setup(): 
  '''Setup the RGB-LED module pins and state.''' 
  #Set up the wiring 
  GPIO.setmode(GPIO.BOARD) 
  # Setup Ports 
  for val in LED: 
    GPIO.setup(val, GPIO.OUT) 
  for val in RGB: 
    GPIO.setup(val, GPIO.OUT) 
  led_clear()
```

1.  使用以下代码定义我们的实用程序函数来帮助控制 LED：

```py
def led_gpiocontrol(pins,state): 
  '''This function will control the state of 
  a single or multiple pins in a list.''' 
  #determine if "pins" is a single integer or not 
  if isinstance(pins,int): 
    #Single integer - reference directly 
    GPIO.output(pins,state) 
  else: 
    #if not, then cycle through the "pins" list 
    for i in pins: 
      GPIO.output(i,state) 

def led_activate(led,color): 
  '''Enable the selected led(s) and set the required color(s) 
  Will accept single or multiple values''' 
  #Enable led 
  led_gpiocontrol(led,LED_ENABLE) 
  #Enable color 
  led_gpiocontrol(color,RGB_ENABLE) 

def led_deactivate(led,color): 
  '''Deactivate the selected led(s) and set the required 
  color(s) will accept single or multiple values''' 
  #Disable led 
  led_gpiocontrol(led,LED_DISABLE) 
  #Disable color 
  led_gpiocontrol(color,RGB_DISABLE) 

def led_time(led, color, timeon): 
  '''Switch on the led and color for the timeon period''' 
  led_activate(led,color) 
  time.sleep(timeon) 
  led_deactivate(led,color) 

def led_clear(): 
  '''Set the pins to default state.''' 
  for val in LED: 
    GPIO.output(val, LED_DISABLE) 
  for val in RGB: 
    GPIO.output(val, RGB_DISABLE) 

def led_cleanup(): 
  '''Reset pins to default state and release GPIO''' 
  led_clear() 
  GPIO.cleanup()
```

1.  创建一个测试函数来演示模块的功能：

```py
def main(): 
  '''Directly run test function. 
  This function will run if the file is executed directly''' 
  led_setup() 
  led_time(LED1,RGB_RED,5) 
  led_time(LED2,RGB_GREEN,5) 
  led_time(LED3,RGB_BLUE,5) 
  led_time(LED,RGB_MAGENTA,2) 
  led_time(LED,RGB_YELLOW,2) 
  led_time(LED,RGB_CYAN,2)  

if __name__=='__main__': 
  try: 
    main() 
  finally: 
    led_cleanup() 
#End 
```

# 它是如何工作的...

首先，我们通过定义所需的状态来定义硬件设置，以便根据使用的 RGB LED（共阳极）的类型来**启用**和**禁用**LED。如果您使用的是共阳极设备，只需颠倒**启用**和**禁用**状态。

接下来，我们定义 GPIO 映射到引脚，以匹配我们之前进行的接线。

我们还通过组合红色、绿色和/或蓝色来定义一些基本的颜色组合，如下图所示：

![](img/7e1997f5-945b-402a-8cfe-8be0de9ea29d.png)LED 颜色组合

我们定义了一系列有用的函数，首先是`led_setup()`，它将把 GPIO 编号设置为`GPIO.BOARD`，并定义所有要用作输出的引脚。我们还调用一个名为`led_clear()`的函数，它将把引脚设置为默认状态，所有引脚都被禁用。

这意味着 LED 引脚 1-5（每个 LED 的共阳极）被设置为`HIGH`，而 RGB 引脚（每种颜色的单独阳极）被设置为`LOW`。

我们创建一个名为`led_gpiocontrol()`的函数，它将允许我们设置一个或多个引脚的状态。`isinstance()`函数允许我们测试一个值，看它是否匹配特定类型（在本例中是单个整数）；然后我们可以设置单个引脚的状态，或者遍历引脚列表并设置每个引脚的状态。

接下来，我们定义两个函数，`led_activate()`和`led_deactivate()`，它们将启用和禁用指定的 LED 和颜色。最后，我们定义`led_time()`，它将允许我们指定 LED、颜色和开启时间。

我们还创建`led_cleanup()`来将引脚（和 LED）重置为默认值，并调用`GPIO.cleanup()`来释放正在使用的 GPIO 引脚。

这个脚本旨在成为一个库文件，因此我们将使用`if __name__=='__main__'`检查，只有在直接运行文件时才运行我们的测试代码：

通过检查`__name__`的值，我们可以确定文件是直接运行的（它将等于`__main__`），还是被另一个 Python 脚本导入的。

这使我们能够定义一个特殊的测试代码，只有在直接加载和运行文件时才执行。如果我们将此文件作为另一个脚本中的模块包含，那么此代码将不会被执行。

与以前一样，我们将使用`try`/`finally`来允许我们始终执行清理操作，即使我们提前退出。

为了测试脚本，我们将设置 LED 依次以各种颜色点亮。

# 还有更多...

我们可以通过一次打开 RGB LED 的一个或多个部分来创建几种不同的颜色。然而，通过一些巧妙的编程，我们可以创建整个颜色谱。此外，我们可以似乎同时在每个 LED 上显示不同的颜色。

# 硬件复用

LED 需要在阳极侧施加高电压，在阴极侧施加低电压才能点亮。电路中使用的 RGB LED 是共阳极的，因此我们必须在 RGB 引脚上施加高电压（3V3），在阴极引脚上施加低电压（0V）（分别连接到每个 LED 的 1 到 5 引脚）。

阴极和 RGB 引脚状态如下：

![](img/c3aaca65-07d4-4d44-b773-91f3cb7f242f.png)阴极和 RGB 引脚状态

因此，我们可以启用一个或多个 RGB 引脚，但仍然控制点亮哪个 LED。我们启用我们想要点亮的 LED 的引脚，并禁用我们不想点亮的引脚。这使我们可以使用比控制每个 RGB 线需要的引脚少得多的引脚。

# 显示随机图案

我们可以向我们的库中添加新的函数以产生不同的效果，例如生成随机颜色。以下函数使用`randint()`来获取 1 到颜色数量之间的值。我们忽略任何超出可用颜色数量的值，以便我们可以控制 LED 关闭的频率。执行以下步骤以添加所需的函数：

1.  使用以下代码将`random`模块中的`randint()`函数添加到`rgbled.py`脚本中：

```py
from random import randint
```

1.  现在使用以下代码添加`led_rgbrandom()`：

```py
def led_rgbrandom(led,period,colors): 
   ''' Light up the selected led, for period in seconds, 
   in one of the possible colors. The colors can be 
   1 to 3 for RGB, or 1-6 for RGB plus combinations, 
   1-7 includes white. Anything over 7 will be set as 
   OFF (larger the number more chance of OFF).'''  
  value = randint(1,colors) 
  if value < len(RGB_LIST): 
    led_time(led,RGB_LIST[value-1],period) 
```

1.  在`main()`函数中使用以下命令创建一系列

闪烁 LED：

```py
for i in range(20): 
  for j in LED: 
    #Select from all, plus OFF 
    led_rgbrandom(j,0.1,20) 
```

# 混合多种颜色

到目前为止，我们只在一个或多个 LED 上一次显示一种颜色。如果考虑电路的接线方式，您可能会想知道我们如何让一个 LED 同时显示一种颜色，而另一个显示不同的颜色。简单的答案是我们不需要-我们只是快速地做到这一点！

我们所需要做的就是一次显示一种颜色，但来回变换，变换得如此之快，以至于颜色看起来像两种颜色的混合（甚至是三种红/绿/蓝 LED 的组合）。幸运的是，树莓派等计算机可以很容易地做到这一点，甚至允许我们组合 RGB 元素以在所有五个 LED 上制作多种颜色。执行以下步骤来混合颜色：

1.  在`rgbled.py`脚本的顶部添加组合颜色定义，在混合颜色的定义之后，使用以下代码：

```py
#Combo Colors 
RGB_AQUA = [RGB_CYAN,RGB_GREEN] 
RGB_LBLUE = [RGB_CYAN,RGB_BLUE] 
RGB_PINK = [RGB_MAGENTA,RGB_RED] 
RGB_PURPLE = [RGB_MAGENTA,RGB_BLUE] 
RGB_ORANGE = [RGB_YELLOW,RGB_RED] 
RGB_LIME = [RGB_YELLOW,RGB_GREEN] 
RGB_COLORS = [RGB_LIME,RGB_YELLOW,RGB_ORANGE,RGB_RED, 
              RGB_PINK,RGB_MAGENTA,RGB_PURPLE,RGB_BLUE, 
              RGB_LBLUE,RGB_CYAN,RGB_AQUA,RGB_GREEN] 
```

上述代码将提供创建我们所需的颜色组合，`RGB_COLORS`提供了对颜色的平滑过渡。

1.  接下来，我们需要创建一个名为`led_combo()`的函数来处理单个或多个颜色。该函数的代码如下：

```py
def led_combo(pins,colors,period): 
  #determine if "colors" is a single integer or not 
  if isinstance(colors,int): 
    #Single integer - reference directly 
    led_time(pins,colors,period) 
  else: 
    #if not, then cycle through the "colors" list 
    for i in colors: 
      led_time(pins,i,period) 
```

1.  现在我们可以创建一个新的脚本`rgbledrainbow.py`，以利用我们`rgbled.py`模块中的新功能。`rgbledrainbow.py`脚本将如下所示：

```py
#!/usr/bin/python3 
#rgbledrainbow.py 
import time 
import rgbled as RGBLED 

def next_value(number,max): 
  number = number % max 
  return number 

def main(): 
  print ("Setup the RGB module") 
  RGBLED.led_setup() 

  # Multiple LEDs with different Colors 
  print ("Switch on Rainbow") 
  led_num = 0 
  col_num = 0 
  for l in range(5): 
    print ("Cycle LEDs") 
    for k in range(100): 
      #Set the starting point for the next set of colors 
      col_num = next_value(col_num+1,len(RGBLED.RGB_COLORS)) 
      for i in range(20):  #cycle time 
        for j in range(5): #led cycle 
          led_num = next_value(j,len(RGBLED.LED)) 
          led_color = next_value(col_num+led_num, 
                                 len(RGBLED.RGB_COLORS)) 
          RGBLED.led_combo(RGBLED.LED[led_num], 
                           RGBLED.RGB_COLORS[led_color],0.001) 

    print ("Cycle COLORs")         
    for k in range(100): 
      #Set the next color 
      col_num = next_value(col_num+1,len(RGBLED.RGB_COLORS)) 
      for i in range(20): #cycle time 
        for j in range(5): #led cycle 
          led_num = next_value(j,len(RGBLED.LED)) 
          RGBLED.led_combo(RGBLED.LED[led_num], 
                           RGBLED.RGB_COLORS[col_num],0.001) 
  print ("Finished") 

if __name__=='__main__': 
  try: 
    main() 
  finally: 
    RGBLED.led_cleanup() 
#End 
```

`main()`函数将首先循环遍历 LED，将`RGB_COLORS`数组中的每种颜色设置在所有 LED 上。然后，它将循环遍历颜色，在 LED 上创建彩虹效果：

![](img/2709e07f-71c2-43ff-b175-3bc84d4e4c45.png)在五个 RGB LED 上循环显示多种颜色

# 使用视觉持久性编写消息

**视觉持续性**（**POV**）显示可以产生一种几乎神奇的效果，通过快速来回移动一行 LED 或在圆圈中移动 LED 来在空中显示图像。这种效果的原理是因为您的眼睛无法调整得足够快，以分离出单独的闪光，因此您观察到一个合并的图像（显示的消息或图片）：

！[](Images/221d8d87-7773-4f9b-91eb-d4d610adade4.png)使用 RGB LED 的视觉持续性

# 准备工作

这个配方使用了前一个配方中使用的 RGB LED 套件；您还需要以下额外的物品：

+   面包板（半尺寸或更大）

+   2 x DuPont 母对公跳线

+   倾斜开关（适合滚珠类型）

+   1 x 470 欧姆电阻（R_Protect）

+   面包板线（实心线）

倾斜开关应添加到 RGB LED（如*准备工作*部分的*多路复用彩色 LED*配方中所述）。倾斜开关的接线如下：

！[](Images/60548e93-3f4c-411f-9f5a-76df3d0b07d4.png)倾斜开关连接到 GPIO 输入（GPIO 引脚 24）和 Gnd（GPIO 引脚 6）

为了重现 POV 图像，您需要能够快速移动 LED 并来回倾斜开关。请注意倾斜开关安装在侧面倾斜，因此当向左移动时开关将打开。建议将硬件安装在一根木头或类似设备上。您甚至可以使用便携式 USB 电池组和 Wi-Fi dongle 来通过远程连接为树莓派供电和控制（有关详细信息，请参见*第一章*中的*通过网络远程连接树莓派使用 SSH（和 X11 转发）*配方）：

！[](Images/dd5c1ecb-a851-40a1-b565-38d5f70b6666.png)持续视觉硬件设置

您还需要已完成的`rgbled.py`文件，我们将在*如何操作*...部分进一步扩展它。

# 如何操作...

1.  创建一个名为`tilt.py`的脚本来报告倾斜开关的状态：

```py
#!/usr/bin/python3 
#tilt.py 
import RPi.GPIO as GPIO 
#HARDWARE SETUP 
# GPIO 
# 2[===========T=]26[=======]40 
# 1[=============]25[=======]39 
#Tilt Config 
TILT_SW = 24 

def tilt_setup(): 
  #Setup the wiring 
  GPIO.setmode(GPIO.BOARD) 
  #Setup Ports 
  GPIO.setup(TILT_SW,GPIO.IN,pull_up_down=GPIO.PUD_UP) 

def tilt_moving(): 
  #Report the state of the Tilt Switch 
  return GPIO.input(TILT_SW) 

def main(): 
  import time 
  tilt_setup() 
  while True: 
    print("TILT %s"% (GPIO.input(TILT_SW))) 
    time.sleep(0.1) 

if __name__=='__main__': 
  try: 
    main() 
  finally: 
    GPIO.cleanup() 
    print("Closed Everything. END") 
#End 
```

1.  您可以通过直接运行以下命令来测试脚本：

```py
sudo python3 tilt.py
```

1.  将以下`rgbled_pov()`函数添加到我们之前创建的`rgbled.py`脚本中；这将允许我们显示图像的单行：

```py
def rgbled_pov(led_pattern,color,ontime): 
  '''Disable all the LEDs and re-enable the LED pattern in the required color''' 
  led_deactivate(LED,RGB) 
  for led_num,col_num in enumerate(led_pattern): 
    if col_num >= 1: 
      led_activate(LED[led_num],color) 
  time.sleep(ontime) 
```

1.  现在，我们将创建以下文件，名为`rgbledmessage.py`，以执行显示我们的消息所需的操作。首先，我们将导入所使用的模块：更新的`rgbled`模块，新的`tilt`模块和 Python `os`模块。最初，我们将`DEBUG`设置为`True`，这样 Python 终端在脚本运行时将显示额外的信息：

```py
#!/usr/bin/python3 
# rgbledmessage.py 
import rgbled as RGBLED 
import tilt as TILT 
import os 

DEBUG = True 
```

1.  添加一个`readMessageFile()`函数来读取`letters.txt`文件的内容，然后添加`processFileContent()`来为每个字母生成一个 LED 模式的**Python 字典**：

```py
def readMessageFile(filename): 
  assert os.path.exists(filename), 'Cannot find the message file: %s' % (filename) 
  try: 
    with open(filename, 'r') as theFile: 
    fileContent = theFile.readlines() 
  except IOError: 
    print("Unable to open %s" % (filename)) 
  if DEBUG:print ("File Content START:") 
  if DEBUG:print (fileContent) 
  if DEBUG:print ("File Content END") 
  dictionary = processFileContent(fileContent) 
  return dictionary  

def processFileContent(content): 
  letterIndex = [] #Will contain a list of letters stored in the file 
  letterList = []  #Will contain a list of letter formats 
  letterFormat = [] #Will contain the format of each letter 
  firstLetter = True 
  nextLetter = False 
  LETTERDIC={} 
  #Process each line that was in the file 
  for line in content: 
    # Ignore the # as comments 
    if '#' in line: 
      if DEBUG:print ("Comment: %s"%line) 
    #Check for " in the line = index name   
    elif '"' in line: 
      nextLetter = True 
      line = line.replace('"','') #Remove " characters 
      LETTER=line.rstrip() 
      if DEBUG:print ("Index: %s"%line) 
    #Remaining lines are formatting codes 
    else: 
      #Skip firstLetter until complete 
      if firstLetter: 
        firstLetter = False 
        nextLetter = False 
        lastLetter = LETTER 
      #Move to next letter if needed 
      if nextLetter: 
        nextLetter = False 
        LETTERDIC[lastLetter]=letterFormat[:] 
        letterFormat[:] = [] 
        lastLetter = LETTER 
      #Save the format data 
      values = line.rstrip().split(' ') 
      row = [] 
      for val in values: 
        row.append(int(val)) 
      letterFormat.append(row) 
  LETTERDIC[lastLetter]=letterFormat[:] 
  #Show letter patterns for debugging 
  if DEBUG:print ("LETTERDIC: %s" %LETTERDIC) 
  if DEBUG:print ("C: %s"%LETTERDIC['C']) 
  if DEBUG:print ("O: %s"%LETTERDIC['O']) 
  return LETTERDIC
```

1.  添加一个`createBuffer()`函数，它将把消息转换为每个字母的 LED 模式系列（假设该字母由`letters.txt`文件定义）：

```py
def createBuffer(message,dictionary): 
  buffer=[] 
  for letter in message: 
    try: 
      letterPattern=dictionary[letter] 
    except KeyError: 
      if DEBUG:print("Unknown letter %s: use _"%letter) 
      letterPattern=dictionary['_'] 
    buffer=addLetter(letterPattern,buffer) 
  if DEBUG:print("Buffer: %s"%buffer) 
  return buffer 

def addLetter(letter,buffer): 
  for row in letter: 
    buffer.append(row) 
  buffer.append([0,0,0,0,0]) 
  buffer.append([0,0,0,0,0]) 
  return buffer 
```

1.  接下来，我们定义一个`displayBuffer()`函数，使用`rgbled`模块中的`rgbled_pov()`函数来显示 LED 模式：

```py
def displayBuffer(buffer): 
  position=0 
  while(1): 
    if(TILT.tilt_moving()==False): 
      position=0 
    elif (position+1)<len(buffer): 
      position+=1 
      if DEBUG:print("Pos:%s ROW:%s"%(position,buffer[position])) 
    RGBLED.rgbled_pov(buffer[position],RGBLED.RGB_GREEN,0.001) 
    RGBLED.rgbled_pov(buffer[position],RGBLED.RGB_BLUE,0.001) 
```

1.  最后，我们创建一个`main()`函数来执行所需的每个步骤：

1.  设置硬件组件（RGB LED 和倾斜开关）。

1.  阅读`letters.txt`文件。

1.  定义 LED 字母模式的字典。

1.  生成一个缓冲区来表示所需的消息。

1.  使用`rgbled`模块显示缓冲区，并使用`tilt`模块进行控制：

```py
def main(): 
  RGBLED.led_setup() 
  TILT.tilt_setup() 
  dict=readMessageFile('letters.txt') 
  buffer=createBuffer('_COOKBOOK_',dict) 
  displayBuffer(buffer) 

if __name__=='__main__': 
  try: 
    main() 
  finally: 
    RGBLED.led_cleanup() 
    print("Closed Everything. END") 
#End 
```

1.  创建以下文件，名为`letters.txt`，以定义显示示例`'_COOKBOOK_'`消息所需的 LED 模式。请注意，此文件只需要为消息中的每个唯一字母或符号定义一个模式：

```py
#COOKBOOK 
"C" 
0 1 1 1 0 
1 0 0 0 1 
1 0 0 0 1 
"O" 
0 1 1 1 0 
1 0 0 0 1 
1 0 0 0 1 
0 1 1 1 0 
"K" 
1 1 1 1 1 
0 1 0 1 0 
1 0 0 0 1 
"B" 
1 1 1 1 1 
1 0 1 0 1 
0 1 0 1 0 
"_" 
0 0 0 0 0 
0 0 0 0 0 
0 0 0 0 0 
0 0 0 0 0 
0 0 0 0 0 
```

# 工作原理...

第一个函数“readMessageFile（）”将打开并读取给定文件的内容。然后使用“processFileContent（）”返回一个包含文件中定义的字母对应的 LED 图案的 Python 字典。处理文件时，会处理文件中的每一行，忽略包含“＃”字符的任何行，并检查“”字符以指示接下来的 LED 图案的名称。处理文件后，我们得到一个包含 LED 图案的 Python 字典，其中包含`'_'`、`'C'`、`'B'`、`'K'`和`'O'`字符。

```py
'_': [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] 
'C': [[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]] 
'B': [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]] 
'K': [[1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]] 
'O': [[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]] 
```

现在我们有一系列可供选择的字母，我们可以使用“createBuffer（）”函数创建 LED 图案序列。正如其名称所示，该函数将通过查找消息中的每个字母并逐行添加相关的图案来构建 LED 图案的缓冲区。如果在字典中找不到字母，则将使用空格代替。

最后，我们现在有一系列准备显示的 LED 图案。为了控制我们何时开始序列，我们将使用 TILT 模块并检查倾斜开关的状态：

当倾斜开关不移动时的位置（左）和移动时的位置（右）

倾斜开关由一个小滚珠封闭在一个空心绝缘圆柱体中组成；当球静止在圆柱体底部时，两个引脚之间的连接闭合。当球移动到圆柱体的另一端，远离引脚的接触时，倾斜开关打开：

倾斜开关电路，开关闭合和开关打开时

先前显示的倾斜开关电路将在开关闭合时将 GPIO 引脚 24 连接到地。然后，如果我们读取引脚，当它静止时将返回`False`。通过将 GPIO 引脚设置为输入并启用内部上拉电阻，当倾斜开关打开时，它将报告`True`。

如果倾斜开关是打开的（报告`True`），那么我们将假设单位正在移动，并开始显示 LED 序列，每次显示 LED 图案的一行时递增当前位置。为了使图案更加丰富多彩（只是因为我们可以！），我们会用另一种颜色重复每一行。一旦“TILT.tilt_moving（）”函数报告我们已经停止移动或者我们正在向相反方向移动，我们将重置当前位置，准备重新开始整个图案：

消息由 RGB LED 显示 - 在这里，我们一起使用绿色和蓝色

当 RGB LED 模块和倾斜开关来回移动时，我们应该看到消息在空中显示！

尝试尝试不同的颜色组合、速度和手臂挥动，看看你能产生什么效果。你甚至可以创建一个类似的设置，安装在车轮上，产生连续的 POV 效果。
