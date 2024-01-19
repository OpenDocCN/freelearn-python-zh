# 第十二章：通信接口

到目前为止，我们已经讨论了 Python 中的循环、条件语句和函数。我们还讨论了与树莓派接口的输出设备和简单的数字输入设备。

在本章中，我们将讨论以下通信接口：

+   UART - 串行端口

+   串行外围接口

+   I²C 接口

我们将使用不同的传感器/电子元件来演示在 Python 中编写这些接口的代码。我们留给您选择一个您喜欢的组件来探索这些通信接口。

# UART - 串行端口

**通用异步收发器**（**UART**），即串行端口，是一种通信接口，数据以位的形式从传感器串行传输到主机计算机。使用串行端口是最古老的通信协议之一。它用于数据记录，微控制器从传感器收集数据并通过串行端口传输数据。还有一些传感器以串行通信的形式响应传入的命令传输数据。

我们不会深入讨论串行端口通信的理论（网络上有大量理论可供参考，网址为[`en.wikipedia.org/wiki/Universal_asynchronous_receiver/transmitter`](https://en.wikipedia.org/wiki/Universal_asynchronous_receiver/transmitter)）。我们将讨论使用串行端口与树莓派接口不同的传感器。

# 树莓派 Zero 的 UART 端口

通常，UART 端口由接收器（*Rx*）和发送器（*Tx*）引脚组成，用于接收和发送数据。树莓派的 GPIO 引脚带有 UART 端口。 GPIO 引脚 14（*Tx*引脚）和 15（*Rx*引脚）用作树莓派的 UART 端口：

![](img/ba8f2fb4-fd2d-4800-89a9-acf61b5b8833.png)GPIO 引脚 14 和 15 是 UART 引脚（图片来源：https://www.rs-online.com/designspark/introducing-the-raspberry-pi-b-plus）

# 设置树莓派 Zero 串行端口

为了使用串行端口与传感器通信，串行端口登录/控制台需要被禁用。在**Raspbian**操作系统镜像中，默认情况下启用此功能，因为它可以方便调试。

串行端口登录可以通过`raspi-config`禁用：

1.  启动终端并运行此命令：

```py
       sudo raspi-config
```

1.  从`raspi-config`的主菜单中选择高级选项：

![](img/2c7415ed-c2c3-40b8-83c0-bcc3548d9e69.png)从 raspi-config 菜单中选择高级选项

1.  从下拉菜单中选择 A8 串行选项：

![](img/58a94cc8-560d-4c23-ad84-5bb5329bcaaf.png)从下拉菜单中选择 A8 串行

1.  禁用串行登录：

![](img/ba8f0b5b-4633-4cd2-9b9f-5a41ec3995d1.png)禁用串行登录

1.  完成配置并在最后重新启动：

![](img/631505a6-d950-47b8-8f9c-24bc22f50fff.png)保存配置并重新启动

# 示例 1 - 将二氧化碳传感器与树莓派连接

我们将使用 K30 二氧化碳传感器（其文档可在此处找到，[`co2meters.com/Documentation/Datasheets/DS30-01%20-%20K30.pdf`](http://co2meters.com/Documentation/Datasheets/DS30-01%20-%20K30.pdf)）。它的范围是 0-10,000 ppm，传感器通过串行端口以响应来自树莓派的特定命令提供二氧化碳浓度读数。

以下图显示了树莓派和 K30 二氧化碳传感器之间的连接：

![](img/b9774d1e-a0bd-4f58-9a42-3bba84a0f6e4.png)与树莓派连接的 K30 二氧化碳传感器

传感器的接收器（*Rx*）引脚连接到树莓派 Zero 的发送器（*Tx*-**GPIO 14（UART_TXD）**）引脚（前图中的黄色线）。传感器的发送器（*Tx*）引脚连接到树莓派 Zero 的接收器（*Rx*-**GPIO 15（UART_RXD）**）引脚（前图中的绿色线）。

为了给传感器供电，传感器的 G+引脚（前图中的红线）连接到树莓派 Zero 的**5V**引脚。传感器的 G0 引脚连接到树莓派 Zero 的**GND**引脚（前图中的黑线）。

通常，串行端口通信是通过指定波特率、帧中的位数、停止位和流控来初始化的。

# 用于串行端口通信的 Python 代码

我们将使用**pySerial**库（[`pyserial.readthedocs.io/en/latest/shortintro.html#opening-serial-ports`](https://pyserial.readthedocs.io/en/latest/shortintro.html#opening-serial-ports)）来接口二氧化碳传感器：

1.  根据传感器的文档，可以通过以波特率 9600、无奇偶校验、8 位和 1 个停止位初始化串行端口来读取传感器输出。 GPIO 串行端口为`ttyAMA0`。与传感器进行接口的第一步是初始化串行端口通信：

```py
       import serial 
       ser = serial.Serial("/dev/ttyAMA0")
```

1.  根据传感器文档（[`co2meters.com/Documentation/Other/SenseAirCommGuide.zip`](http://co2meters.com/Documentation/Other/SenseAirCommGuide.zip)），传感器对二氧化碳浓度的以下命令做出响应：

![](img/f94c1fbd-f4b0-4ad7-965b-058f4cd1d3a1.png)从传感器数据表中借用的读取二氧化碳浓度的命令

1.  命令可以如下传输到传感器：

```py
       ser.write(bytearray([0xFE, 0x44, 0x00, 0x08, 0x02, 0x9F, 0x25]))
```

1.  传感器以 7 个字节的响应做出响应，可以如下读取：

```py
       resp = ser.read(7)
```

1.  传感器的响应格式如下：

![](img/e4c08dba-07ab-4ea3-8865-db3c17c1b9dd.png)二氧化碳传感器响应

1.  根据数据表，传感器数据大小为 2 个字节。每个字节可用于存储 0 和 255 的值。两个字节可用于存储高达 65,535 的值（255 * 255）。二氧化碳浓度可以根据消息计算如下：

```py
       high = resp[3] 
       low = resp[4] 
       co2 = (high*256) + low
```

1.  把它全部放在一起：

```py
       import serial 
       import time 
       import array 
       ser = serial.Serial("/dev/ttyAMA0") 
       print("Serial Connected!") 
       ser.flushInput() 
       time.sleep(1) 

       while True: 
           ser.write(bytearray([0xFE, 0x44, 0x00, 0x08,
           0x02, 0x9F, 0x25])) 
           # wait for sensor to respond 
           time.sleep(.01) 
           resp = ser.read(7) 
           high = resp[3] 
           low = resp[4] 
           co2 = (high*256) + low 
           print() 
           print() 
           print("Co2 = " + str(co2)) 
           time.sleep(1)
```

1.  将代码保存到文件并尝试执行它。

# I2C 通信

**I²C**（Inter-Integrated Circuit）通信是一种串行通信类型，允许将多个传感器接口到计算机。 I²C 通信由时钟和数据线两根线组成。树莓派 Zero 的 I²C 通信的时钟和数据引脚分别为**GPIO 3**（**SCL**）和**GPIO 2**（**SDA**）。为了在同一总线上与多个传感器通信，通常通过 I²C 协议通信的传感器/执行器通常通过它们的 7 位地址进行寻址。可以有两个或更多树莓派板与同一 I²C 总线上的同一传感器进行通信。这使得可以在树莓派周围构建传感器网络。

I²C 通信线是开漏线路；因此，它们使用电阻上拉，如下图所示：

![](img/9a174b4f-84ae-4b04-8bae-8310ea6cdaa3.png)I²C 设置

让我们通过一个示例来回顾一下 I²C 通信。

# 示例 2 - PiGlow

**PiGlow**是树莓派的一个附加硬件，由 18 个 LED 与**SN3218**芯片接口。该芯片允许通过 I²C 接口控制 LED。芯片的 7 位地址为`0x54`。

为了接口附加硬件，**SCL**引脚连接到**GPIO 3**，**SDA**引脚连接到**GPIO 2**；地线引脚和电源引脚分别连接到附加硬件的对应引脚。

PiGlow 附带了一个抽象 I²C 通信的库：[`github.com/pimoroni/piglow`](https://github.com/pimoroni/piglow)。

尽管该库是对 I²C 接口的封装，但我们建议阅读代码以了解操作 LED 的内部机制：

![](img/681f2d8e-b110-4ae4-b62b-f326cdef6450.jpg)PiGlow 叠放在 Raspberry Pi 上

# 安装库

PiGlow 库可以通过从命令行终端运行以下命令来安装：

```py
    curl get.pimoroni.com/piglow | bash
```

# 示例

安装完成后，切换到示例文件夹（`/home/pi/Pimoroni/piglow`）并运行其中一个示例：

```py
    python3 bar.py
```

它应该运行*闪烁*灯效果，如下图所示：

![](img/2ac65de6-24b3-4945-ae7d-406e5946ed31.jpg)PiGlow 上的闪烁灯

同样，还有库可以使用 I²C 通信与实时时钟、LCD 显示器等进行通信。如果你有兴趣编写自己的接口，提供 I²C 通信与传感器/输出设备的细节，请查看本书附带网站上的一些示例。

# 示例 3 - 用于树莓派的 Sensorian 附加硬件

**Sensorian**是为树莓派设计的附加硬件。这个附加硬件配备了不同类型的传感器，包括光传感器、气压计、加速度计、LCD 显示器接口、闪存存储器、电容触摸传感器和实时时钟。

这个附加硬件上的传感器足以学习本章讨论的所有通信接口的使用方法：

![](img/23b57438-e4a5-4997-bc46-e485b31a001a.jpg)堆叠在树莓派 Zero 上的 Sensorian 硬件

在本节中，我们将讨论一个示例，我们将使用 I²C 接口通过树莓派 Zero 测量环境光水平。附加硬件板上的传感器是**APDS-9300**传感器（[www.avagotech.com/docs/AV02-1077EN](http://www.avagotech.com/docs/AV02-1077EN)）。

# 用于光传感器的 I2C 驱动程序

传感器硬件的驱动程序可从 GitHub 存储库中获取（[`github.com/sensorian/sensorian-firmware.git`](https://github.com/sensorian/sensorian-firmware.git)）。让我们从命令行终端克隆存储库：

```py
    git clone https://github.com/sensorian/sensorian-firmware.git 
```

让我们使用驱动程序（位于` ~/sensorian-firmware/Drivers_Python/APDS-9300`文件夹中）从传感器的两个 ADC 通道读取值：

```py
import time 
import APDS9300 as LuxSens 
import sys 

AmbientLight = LuxSens.APDS9300() 
while True: 
   time.sleep(1) 
   channel1 = AmbientLight.readChannel(1)                       
   channel2 = AmbientLight.readChannel(0) 
   Lux = AmbientLight.getLuxLevel(channel1,channel2) 
   print("Lux output: %d." % Lux)
```

有了两个通道的 ADC 值，驱动程序可以使用以下公式（从传感器数据表中检索）计算环境光值：

![](img/3af23ebe-b270-47a2-93a3-e379b7a6bea7.png)使用 ADC 值计算的环境光水平

这个计算是由属性`getLuxLevel`执行的。在正常照明条件下，环境光水平（以勒克斯为单位）约为`2`。当我们用手掌遮住光传感器时，测得的输出为`0`。这个传感器可以用来测量环境光，并相应地调整房间照明。

# 挑战

我们讨论了使用光传感器测量环境光水平。我们如何利用光输出（环境光水平）来控制房间照明？

# SPI 接口

还有一种名为**串行外围接口**（**SPI**）的串行通信接口。必须通过`raspi-config`启用此接口（这类似于在本章前面启用串行端口接口）。使用 SPI 接口类似于 I²C 接口和串行端口。

通常，SPI 接口由时钟线、数据输入、数据输出和**从机选择**（**SS**）线组成。与 I²C 通信不同（在那里我们可以连接多个主机），在同一总线上可以有一个主机（树莓派 Zero），但可以有多个从机。**SS**引脚用于选择树莓派 Zero 正在读取/写入数据的特定传感器，当同一总线上连接了多个传感器时。

# 示例 4 - 写入外部存储器芯片

让我们查看一个示例，我们将通过 SPI 接口向 Sensorian 附加硬件上的闪存存储器写入数据。SPI 接口和存储器芯片的驱动程序可从同一 GitHub 存储库中获取。

由于我们已经下载了驱动程序，让我们查看一下驱动程序中提供的示例：

```py
import sys 
import time   
import S25FL204K as Memory
```

让我们初始化并将消息`hello`写入存储器：

```py
Flash_memory = Memory.S25FL204K() 
Flash_memory.writeStatusRegister(0x00) 
message = "hello" 
flash_memory.writeArray(0x000000,list(message), message.len())
```

现在，让我们尝试读取刚刚写入外部存储器的数据：

```py
data = flash_memory.readArray(0x000000, message.len()) 
print("Data Read from memory: ") 
print(''.join(data))
```

本章提供了代码示例，可通过下载获得（`memory_test.py`）。

我们成功地演示了使用 SPI 读/写外部存储器芯片。

# 向读者提出挑战

在这里的图中，有一个 LED 灯带（[`www.adafruit.com/product/306`](https://www.adafruit.com/product/306)）与树莓派附加硬件的 SPI 接口相连，使用了 Adafruit Cobbler（[`www.adafruit.com/product/914`](https://www.adafruit.com/product/914)）。我们提供了一个线索，说明如何将 LED 灯带与树莓派 Zero 相连。我们希望看到您能否自己找到将 LED 灯带与树莓派 Zero 相连的解决方案。请参考本书网站获取答案。

LED 灯带与树莓派 Zero 的 Adafruit Cobbler 接口

# 总结

在本章中，我们讨论了树莓派 Zero 上可用的不同通信接口。这些接口包括 I²C、SPI 和 UART。我们将在我们的最终项目中使用这些接口。我们使用了二氧化碳传感器、LED 驱动器和传感器平台来讨论这些接口。在下一章中，我们将讨论面向对象编程及其独特的优势。我们将通过一个例子讨论面向对象编程的必要性。面向对象编程在您需要编写自己的驱动程序来控制机器人的组件或编写传感器的接口库的情况下尤其有帮助。
