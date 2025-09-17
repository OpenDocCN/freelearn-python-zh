# 第七章. 中期项目 – 可携带的 DIY 温度控制器

在完成第一个 Python-Arduino 项目之后，你学习了原型设计各种传感器、开发用户界面和绘制传感器数据的过程。你之前章节中学到的概念可以用来创建各种基于 Arduino 的硬件项目。一个好的应用程序概念的诞生总是始于现实世界的需求，如果执行得当，最终成为一个实用的项目。在本章中，我们将通过一个便携式传感器单元的例子来展示这个项目构建过程。正如你可以从章节标题中估计的那样，我们将构建一个简单且便携的 DIY 温度控制器，可以在没有台式计算机或笔记本电脑的情况下部署。

首先，我们将描述拟议的温度控制器，包括实现这些目标的具体目标和过程。一旦确定了实现这些目标的方法，你将介绍两个连续的编程阶段来开发可部署和可携带的单元。在第一阶段，我们将利用传统计算机成功开发程序，将 Arduino 与 Python 接口连接。在第二阶段，我们将用 Raspberry Pi 替换这台计算机，使其便携并可部署。

# 温度控制器 – 项目描述

在我们可以使用所学知识构建的多个项目中，一个帮助你监控周围环境的项⽬真正脱颖而出，成为一个重要的实际应用。从各种环境监控项目，如气象站、温度控制器和植物监控系统，我们将开发温度控制器，因为它专注于室内环境，可以成为你日常生活的部分。

温度控制器是任何远程家庭监控系统和家庭自动化系统中最重要组成部分之一。一个流行的商业温度控制器例子是 Nest 温度控制器([`www.nest.com`](https://www.nest.com))，它为现有家庭的供暖和冷却系统提供智能远程监控和调度功能。在我们考虑像 Nest 这样的全栈产品之前，我们首先需要构建一个具有基本功能的 DIY 温度控制器。之后，我们可以通过添加功能来改进 DIY 温度控制器的体验。让我们首先概述我们计划在本版温度控制器项目中实现的功能。

## 项目背景

温度、湿度和环境光线是我们希望通过温度控制器监控的三个主要物理特性。在用户体验方面，我们希望有一个优雅的用户界面来显示测量的传感器数据。如果任何传感器数据以折线图的形式绘制，用户体验将更加丰富。在温度控制器的例子中，传感器数据的可视化表示比仅仅显示纯数值更有意义。

该项目的重大目标之一是使恒温器便携和可部署，以便在日常生活中使用。为了满足这一要求，恒温器显示屏需要从常规显示器更改为更小、更便携的设备。为了确保其实际应用和意义，恒温器应展示实时操作。

重要的是要注意，恒温器将不会与任何执行器（如家用冷却和加热系统）接口。由于这些系统与恒温器项目的接口需要高级理解和与加热和冷却系统工作的经验，这将偏离章节原始目标，即教授你 Arduino 和 Python 编程。

## 项目目标和阶段

为了描述我们希望在恒温器中拥有的功能，让我们首先确定实现这些目标的目标和里程碑。项目的主要目标可以确定为以下：

+   确定项目所需的传感器和硬件组件

+   使用这些传感器和 Arduino 板设计并组装恒温器的电路

+   设计有效的用户体验并开发软件以适应用户体验

+   开发和实施代码以将设计的硬件与软件组件接口

恒温器项目的代码开发过程分为两个主要阶段。第一阶段的目标包括传感器接口、Arduino 脚本的开发以及开发你一直在使用的常规计算机上的 Python 代码。第一阶段编码里程碑可以进一步分配如下：

+   开发 Arduino 脚本以接口传感器和按钮，并通过串行端口将传感器数据输出到 Python 程序

+   开发 Python 代码，使用`pySerial`库从串行端口获取传感器数据，并使用在`Tkinter`中设计的 GUI 显示数据

+   使用`matplotlib`库创建一个图表来展示实时湿度读数

在第二阶段，我们将把 Arduino 硬件连接到单板计算机和微型显示屏，使其便携和可部署。实现第二阶段目标的里程碑如下：

+   安装和配置单板计算机 Raspberry Pi，以运行第一阶段中的 Python 代码

+   将微型屏幕与 Raspberry Pi 接口和配置

+   优化 GUI 和图表窗口以适应这个小屏幕的分辨率

在本节的下一小节中，你将了解到两个阶段的所需组件列表，随后是硬件电路设计和软件流程设计。这些阶段的编程练习将在本章的下一两个部分中解释。

## 所需组件列表

我们没有经过识别所需组件的过程，而是根据它们在之前练习中的使用情况、易用性和可用性，已经为这个项目选择了组件。您可以根据您构建项目时的可用性或对其他传感器的熟悉程度来替换这些组件。只需确保，如果这些新组件与我们使用的组件不兼容，您要处理好电路连接和代码的修改。

在原型设计的第一阶段，我们需要组件来开发恒温器单元的电子电路。正如我们之前提到的，我们将通过我们的单元测量温度、湿度和环境光线。我们已经学习了关于温度传感器 TMP102 和环境光线传感器 BH1750 的内容，这些内容在第四章 *深入 Python-Arduino 原型设计* 中有所介绍。我们将使用这些传感器来完成这个项目，同时使用湿度传感器 HIH-4030。项目将使用与之前章节相同的 Arduino Uno 板以及必要的电缆。我们还需要两个按钮来为单元提供手动输入。第一阶段所需组件的总结如下表所示：

| 组件（第一阶段） | 数量 | 网站 |
| --- | --- | --- |
| Arduino Uno | 1 | [`www.sparkfun.com/products/11021`](https://www.sparkfun.com/products/11021) |
| Arduino 用 USB 电缆 | 1 | [`www.sparkfun.com/products/512`](https://www.sparkfun.com/products/512) |
| 面包板 | 1 | [`www.sparkfun.com/products/9567`](https://www.sparkfun.com/products/9567) |
| TMP102 温度传感器 | 1 | [`www.sparkfun.com/products/11931`](https://www.sparkfun.com/products/11931) |
| HIH-4030 湿度传感器 | 1 | [`www.sparkfun.com/products/9569`](https://www.sparkfun.com/products/9569) |
| BH1750 环境光线传感器 | 1 | [`www.robotshop.com/en/dfrobot-light-sensor-bh1750.html`](http://www.robotshop.com/en/dfrobot-light-sensor-bh1750.html) |
| 推按钮开关 | 2 | [`www.sparkfun.com/products/97`](https://www.sparkfun.com/products/97) |
| 1 千欧姆电阻 | 2 |   |
| 10 千欧姆电阻 | 2 |   |
| 连接线 | 如需求数量 |   |

尽管表格提供了少数特定网站的链接，但您可以从您偏好的供应商那里获取这些组件。我们之前未使用过的两个主要组件 HIH-4030 湿度传感器和推按钮开关如下所述：

+   **HIH-4030 湿度传感器**：该传感器测量并提供相对湿度结果作为模拟输出。传感器的输出可以直接连接到 Arduino 的任何模拟引脚。以下图片展示了 SparkFun Electronics 销售的带有 HIH-4030 传感器的分线板。您可以从其数据表中了解更多关于 HIH-4030 传感器的信息，数据表可以从[`www.sparkfun.com/datasheets/Sensors/Weather/SEN-09569-HIH-4030-datasheet.pdf`](https://www.sparkfun.com/datasheets/Sensors/Weather/SEN-09569-HIH-4030-datasheet.pdf)获取：![所需组件列表](img/5938OS_07_06.jpg)

+   **按钮开关**：按钮开关是小型开关，可以在面包板上使用。按下时，开关输出状态变为**高电平**，否则为**低电平**。![所需组件列表](img/5938OS_07_07.jpg)

在第二阶段，我们将通过用 Raspberry Pi 替换电脑来使传感器单元变得便携。为此，您需要以下组件开始：

| 组件（第二阶段） | 数量 | 图片 |
| --- | --- | --- |
| Raspberry Pi | 1 | [`www.sparkfun.com/products/11546`](https://www.sparkfun.com/products/11546) |
| 带电源适配器的 Micro USB 线 | 1 | [`www.amazon.com/CanaKit-Raspberry-Supply-Adapter-Charger/dp/B00GF9T3I0/`](http://www.amazon.com/CanaKit-Raspberry-Supply-Adapter-Charger/dp/B00GF9T3I0/) |
| 8 GB SD 卡 | 1 | [`www.sparkfun.com/products/12998`](https://www.sparkfun.com/products/12998) |
| TFT 液晶显示屏 | 1 | [`www.amazon.com/gp/product/B00GASHVDU/`](http://www.amazon.com/gp/product/B00GASHVDU/) |
| USB 集线器 | 可选 |   |

本章后面将提供这些组件的进一步说明。

## 硬件设计

恒温器的整个硬件架构可以分为两个单元：物理世界接口单元和计算单元。物理世界接口单元，正如其名称所示，通过连接到 Arduino 板上的传感器监测物理世界现象，如温度、湿度和环境光线。物理世界接口单元在整章中交替提到恒温器传感器单元。计算单元负责通过 GUI 和图表显示传感器信息。

以下图表展示了第一阶段硬件组件，其中恒温器传感器单元通过 USB 端口连接到电脑。在恒温器传感器单元中，各种传感器组件通过 I2C、模拟和数字引脚连接到 Arduino 板：

![硬件设计](img/5938OS_07_08.jpg)

在第二个编程阶段，我们将把我们的恒温器变成一个可移动和可部署的单元，您将使用单板计算机 Raspberry Pi 作为计算设备。在这个阶段，我们将使用一个连接到 Raspberry Pi 的微型**薄膜晶体管液晶显示屏**（**TFT LCD**），它通过**通用输入/输出**（**GPIO**）引脚连接，并用作显示单元，以替代传统的显示器或笔记本电脑屏幕。以下图显示了新的恒温器计算单元，它真正减小了恒温器的整体尺寸，使其便携和移动。在这个阶段，Arduino 板的电路连接保持不变，我们将使用相同的硬件，无需进行任何重大修改。

![硬件设计](img/5938OS_07_09.jpg)

作为项目两个阶段的共同单元，以 Arduino 为中心的恒温器传感器单元与其他练习相比，需要更复杂的电路连接。在本节中，我们将将必要的传感器和按钮接口到 Arduino 板的相应引脚上，您需要使用面包板来建立这些连接。如果您熟悉 PCB 原型设计，您可以通过焊接这些组件来创建自己的 PCB 板，从而避免使用面包板。与面包板相比，PCB 板更坚固，且不太容易发生松动连接。请使用以下说明和 Fritzing 图表来完成电路连接：

1.  如下图中所示，将 TMP102 和 BH1750 的 SDA 和 SCL 引脚连接到 Arduino 板的模拟引脚 4 和 5，并创建一个 I2C 总线。为了进行这些连接，您可以使用多色编码的电线来简化调试过程。

1.  使用两个 10 千欧姆的上拉电阻连接 SDA 和 SCL 线。

1.  与这些 I2C 传感器相反，HIH-4030 湿度传感器是一个简单的模拟传感器，可以直接连接到模拟引脚。将 HIH-4030 连接到模拟引脚 A0。

1.  使用面包板的电源条，将 TMP102、BH1750 和 HIH-4030 的 VCC 和地连接到 Arduino 板的 +5V 和地，如图所示。我们建议您使用红色和黑色电线分别代表 +5V 和地线。

1.  按钮开关提供**高**或**低**状态的输出，并通过数字引脚进行接口。如图电路所示，使用两个 1 千欧姆电阻将按钮开关连接到数字引脚 2 和 3。

1.  按照以下图中的显示完成剩余的连接。在给 Arduino 板通电之前，请确保所有电线都已牢固连接：![硬件设计](img/5938OS_07_10.jpg)

### 注意

在进行任何连接之前，请确保您始终断开 Arduino 板的电源或 USB 端口。这将防止由于短路而损坏板子。

在前往下一节之前，完成恒温器传感器单元的所有连接。由于该单元在编程阶段都会使用，你不会对恒温器传感器单元进行任何进一步的更改。

## 软件流程用于用户体验设计

任何项目的关键组成部分之一是其可用性或可访问性。当你正在将你的项目原型转化为产品时，有必要拥有一个直观且资源丰富的用户界面，以便用户可以轻松地与你的产品交互。因此，在开始编码之前，有必要定义项目的用户体验和软件流程。软件流程包括流程图和程序的逻辑组件，这些组件是从项目需求中推导出来的。根据我们为恒温器项目定义的目标，软件流程可以在以下图中展示：

![软件流程用于用户体验设计](img/5938OS_07_11.jpg)

在实施过程中，项目的软件流程首先通过 Arduino 测量温度、湿度和环境光线，并逐行打印在串行端口上。Python 程序通过串行端口从 Arduino 获取传感器数据，然后在屏幕上显示数据。同时，Python 程序持续寻找新的数据行。

用户可以使用按钮与恒温器进行交互，这将允许用户更改温度数据的单位。一旦按钮被按下，标志位变为**HIGH**，温度单位从默认单位**华氏度**更改为**摄氏度**。如果再次按下按钮，将发生相反的过程，单位将恢复到默认值。同样，另一个用户交互点是第二个按钮，允许用户打开实时湿度值的图表。第二个按钮也使用类似的方法使用标志位捕获输入并打开新的图表窗口。如果连续按下相同的按钮，程序将关闭图表窗口。

# 第 1 阶段 – 恒温器的原型设计

在这个原型设计阶段，我们将为我们的恒温器开发 Arduino 和 Python 代码，这些代码将在第二阶段进行少量更改后使用。在开始编码练习之前，请确保你有恒温器传感器单元准备就绪，包括 Arduino 板和连接的传感器，如前所述。对于这个阶段，你将使用配备 Arduino IDE 和 Python 编程环境的常规计算机。原型设计阶段需要两个级别的编程，即恒温器传感器单元的 Arduino 草图和计算单元的 Python 代码。让我们开始为我们的恒温器编写代码。

## 恒温器的 Arduino 草图

这个 Arduino 程序的目标是连接传感器，从传感器获取测量值，并在串行端口上打印它们。正如我们之前讨论的，我们不会使用之前项目中使用的标准 Firmata 草图，而是在这个项目中开发一个自定义的 Arduino 草图。要开始，打开本章代码文件夹中的`Thermostat_Arduino.ino`草图，这是您为本书收到的源代码的一部分。

将 Arduino 板的 USB 端口连接到您的计算机。在 Arduino IDE 中选择适当的板和端口名称，并验证代码。一旦代码成功上传，打开**串行监视器**窗口。您应该能够看到类似于以下截图显示的文本：

![恒温器的 Arduino 草图](img/5938OS_07_12.jpg)

Arduino 代码结构和基本声明已在本书的各个部分中解释过。我们不会逐行解释整个代码，而是将重点放在我们之前描述的软件流程的主要组件上。

### 连接温度传感器

在 Arduino 草图中，使用`getTemperature()`函数从 TMP102 传感器获取温度数据。该函数在 TMP102 的 I2C 地址上实现了`Wire`库来读取传感器数据。然后将其转换为适当的温度值：

```py
 float getTemperature(){
  Wire.requestFrom(tmp102Address, 2);

  byte MSB = Wire.read();
  byte LSB = Wire.read();

  //it's a 12bit int, using two's compliment for negative
  int TemperatureSum = ((MSB << 8) | LSB) >> 4;

  float celsius = TemperatureSum*0.0625;
  return celsius;
}
```

`getTemperature()`函数返回摄氏度的温度值，然后将其发送到串行端口。

### 连接湿度传感器

虽然湿度传感器提供模拟输出，但由于它还取决于温度，因此直接获取相对湿度并不简单。`getHumidity()`函数从 HIH-4030 传感器提供的模拟输出计算相对湿度。计算相对湿度的公式来自数据表和传感器的参考示例。如果您使用的是不同的湿度传感器，请确保相应地更改公式，因为它们可能会显著改变结果：

```py
float getHumidity(float degreesCelsius){
//caculate relative humidity
float supplyVolt = 5.0;

// Get the sensor value:
int HIH4030_Value = analogRead(HIH4030_Pin);
// convert to voltage value
float voltage = HIH4030_Value/1023\. * supplyVolt;

// convert the voltage to a relative humidity
float sensorRH = 161.0 * voltage / supplyVolt - 25.8;
float trueRH = sensorRH / (1.0546 - 0.0026 * degreesCelsius);

   return trueRH;
}
```

由于我们正在计算相对湿度，因此返回的湿度值以百分比为单位发送到串行端口。

### 连接光传感器

要连接 BH1750 光传感器，我们将使用之前使用过的 BH1750 Arduino 库。使用此库，可以使用以下代码行直接获取环境光值：

```py
uint16_t lux = lightMeter.readLightLevel();
```

这行提供了以`lux`为单位的光亮度值。这些值也会发送到串行端口，以便 Python 程序可以进一步利用它。

### 使用 Arduino 中断

到目前为止，你使用 Arduino 程序通过`DigitalRead()`或`AnalogRead()`函数读取 I/O 引脚的物理状态。你是如何自动获取状态变化，而不是定期读取引脚并等待状态变化呢？Arduino 中断为 Arduino 板提供了非常方便的捕获信号的方法。中断是自动控制 Arduino 中各种事物的一种非常强大的方式。Arduino 使用`attachInterrupt()`方法支持中断。就物理引脚而言，Arduino Uno 提供了两个中断：中断 0（在数字引脚 2 上）和中断 1（在数字引脚 3 上）。各种 Arduino 板对中断引脚有不同的规格。如果你使用的是除 Uno 以外的任何板，请参考 Arduino 网站以了解你板的中断引脚。 

`attachInterrupt()`函数接受三个输入参数（`pin`、`ISR`和`mode`）。在这些输入参数中，`pin`指的是中断引脚的编号，`ISR`（代表中断服务例程）指的是当中断发生时被调用的函数，而`mode`定义了中断应该被触发时的条件。我们已经在我们描述的以下代码片段中使用了这个函数：

```py
  attachInterrupt(0, button1Press, RISING);
  attachInterrupt(1, button2Press, RISING);
```

`attachInterrupt()`函数支持的`mode`有`LOW`、`CHANGE`、`RISING`和`FALLING`。在我们的案例中，当模式为`RISING`时，即引脚从低电平变为高电平，中断被触发。对于声明在 0 和 1 的中断，我们调用`button1Press`和`button2Press`函数，分别改变`flagTemperature`和`flagPlot`。当`flagTemperature`设置为`HIGH`时，Arduino 发送摄氏温度，否则发送华氏温度。当`flagPlot`为`HIGH`时，Arduino 将在串行端口上打印标志，该标志将被 Python 程序稍后用于打开绘图窗口。你可以从[`arduino.cc/en/Reference/attachInterrupt`](http://arduino.cc/en/Reference/attachInterrupt)上的教程中了解更多关于 Arduino 中断的信息。

## 在 Python 中设计 GUI 和绘图

一旦你的恒温器传感器单元开始向串行端口发送传感器数据，就到了执行这个阶段的第二部分的时候了，即 GUI 和绘图用的 Python 代码。从本章的代码文件夹中，打开名为`Thermostat_Stage1.py`的 Python 文件。在文件中，找到包含`Serial()`函数的行，该函数声明了串行端口。将串行端口名称从`COM5`更改为适当的名称。你可以从 Arduino IDE 中找到这个信息。保存更改并退出编辑器。从同一文件夹中，在终端运行以下命令：

```py
$ python Thermostat_Stage1.py

```

这将执行 Python 代码，你将能够在屏幕上看到 GUI 窗口。

### 在你的 Python 程序中使用 pySerial 流式传输传感器数据

如软件流程所述，程序使用`pySerial`库从 Arduino 接收传感器数据。在 Python 代码中声明串行端口的代码如下：

```py
Import serial
port = serial.Serial('COM5',9600, timeout=1)
```

在使用`pySerial`库时，指定`timeout`参数非常重要，因为如果没有指定`timeout`，代码可能会出错。

### 使用 Tkinter 设计 GUI

本项目的 GUI 设计使用了之前我们使用的`Tkinter`库。作为一个 GUI 构建练习，程序中编写了三列标签（用于显示传感器类型、观测值和观测单位），如下代码片段所示：

```py
# Labels for sensor name
Tkinter.Label(top, text = "Temperature").grid(column = 1, row = 1)
Tkinter.Label(top, text = "Humidity").grid(column = 1, row = 2)
Tkinter.Label(top, text = "Light").grid(column = 1, row = 3)

# Labels for observation values
TempLabel = Tkinter.Label(top, text = " ")
TempLabel.grid(column = 2, row = 1)
HumdLabel = Tkinter.Label(top, text = " ")
HumdLabel.grid(column = 2, row = 2)
LighLabel = Tkinter.Label(top, text = " ")
LighLabel.grid(column = 2, row = 3)

# Labels for observation unit
TempUnitLabel = Tkinter.Label(top, text = " ")
TempUnitLabel.grid(column = 3, row = 1)
HumdUnitLabel = Tkinter.Label(top, text = "%")
HumdUnitLabel.grid(column = 3, row = 2)
LighUnitLabel = Tkinter.Label(top, text = "lx")
LighUnitLabel.grid(column = 3, row = 3)
```

在初始化代码并点击**开始**按钮之前，你将能够看到以下窗口。在此阶段，观测标签被填充，但没有任何值：

![使用 Tkinter 设计 GUI](img/5938OS_07_13.jpg)

点击**开始**按钮后，程序将激活恒温传感器单元并开始从串行端口读取传感器值。使用从串行端口获得的行，程序将用获取的值填充观测标签。以下代码片段更新了观测标签中的温度值，并更新了温度单位：

```py
TempLabel.config(text = cleanText(reading[1]))
TempUnitLabel.config(text = "C")
TempUnitLabel.update_idletasks()
```

在程序中，我们使用类似的方法来更新湿度和周围光线标签的值。正如你在以下截图中所看到的，GUI 现在有了温度、湿度和周围光线读数的值：

![使用 Tkinter 设计 GUI](img/5938OS_07_14.jpg)

**开始**和**退出**按钮被编程为当用户点击时调用`onStartButtonPress()`和`onExitButtonPress()`函数。当用户点击时，`onStartButtonPress()`函数执行创建用户界面所需的代码，而`onExitButtonPress()`函数关闭所有打开的窗口，断开恒温传感器单元的连接，并退出代码：

```py
StartButton = Tkinter.Button(top,
                             text="Start",
                             command=onStartButtonPress)
StartButton.grid(column=1, row=4)
ExitButton = Tkinter.Button(top,
                            text="Exit",
                            command=onExitButtonPress)
ExitButton.grid(column=2, row=4)
```

你可以通过**开始**和**退出**按钮来探索 Python 代码。要观察传感器读数的变化，尝试吹气或在一个恒温传感器单元上放置障碍物。如果程序表现不当，请检查终端以查找错误信息。

### 使用 matplotlib 绘制湿度百分比

我们将使用`matplotlib`库实时绘制相对湿度值。在本项目中，我们将绘制相对湿度值，因为数据范围固定在 0 到 100 百分比之间。使用类似的方法，你也可以绘制温度和周围光线传感器的值。在开发绘制温度和周围光线传感器数据的代码时，确保你使用适当的范围来覆盖同一图表中的传感器数据。现在，正如我们在`onStartButtonPress()`函数中指定的，当你按下绘图按钮时，将弹出一个类似于以下截图的窗口：

![使用 matplotlib 绘制百分比湿度](img/5938OS_07_15.jpg)

以下代码片段负责使用湿度传感器的值绘制折线图。这些值在 *y* 轴上限制在 0 到 100 之间，其中 *y* 轴表示相对湿度范围。每当程序接收到新的湿度值时，图表就会更新：

```py
pyplot.figure()
pyplot.title('Humidity')
ax1 = pyplot.axes()
l1, = pyplot.plot(pData)
pyplot.ylim([0,100])
```

### 使用按钮中断来控制参数

按钮中断是用户体验的关键部分，因为用户可以使用这些中断来控制温度单位和图表。使用按钮中断实现的 Python 功能如下。

#### 通过按按钮更改温度单位

Arduino 脚本包含处理按钮中断的逻辑，并使用它们来更改温度单位。当发生中断时，它不是打印华氏度温度，而是将摄氏度温度发送到串行端口。如以下截图所示，Python 代码只是打印获得的温度观测值的数值及其相关的单位：

![通过按按钮更改温度单位](img/5938OS_07_16.jpg)

如以下代码片段所示，如果 Python 代码接收到 `Temperature(C)` 字符串，它将打印摄氏度温度，如果接收到 `Temperature(F)` 字符串，它将打印华氏度温度：

```py
if (reading[0] == "Temperature(C)"):
    TempLabel.config(text=cleanText(reading[1]))
    TempUnitLabel.config(text="C")
    TempUnitLabel.update_idletasks()
if (reading[0] == "Temperature(F)"):
    TempLabel.config(text=cleanText(reading[1]))
    TempUnitLabel.config(text="F")
    TempUnitLabel.update_idletasks()
```

#### 通过按按钮在 GUI 和图表之间切换

如果 Python 代码从串行端口接收到标志值的 `1`（高电平），它将创建一个新的图表并将湿度值绘制为折线图。然而，如果它接收到 `0`（低电平）作为标志值的，它将关闭任何打开的图表。如以下代码片段所示，程序将始终尝试使用最新的湿度读数更新图表。如果程序找不到打开的图表来绘制此值，它将创建一个新的图表：

```py
if (reading[0] == "Flag"):
    print reading[1]
    if (int(reading[1]) == 1):
        try:
            l1.set_xdata(np.arange(len(pData)))
            l1.set_ydata(pData)  # update the data
            pyplot.ylim([0, 100])
            pyplot.draw()  # update the plot
        except:
            pyplot.figure()
            pyplot.title('Humidity')
            ax1 = pyplot.axes()
            l1, = pyplot.plot(pData)
            pyplot.ylim([0, 100])
    if (int(reading[1]) == 0):
        try:
            pyplot.close('all')
            l1 = None
        except:
```

到目前为止，你应该对恒温器传感器单元和计算单元所需的程序有一个完整的了解。由于涉及到的复杂性，你可能在执行这些程序时遇到一些已知的问题。如果你遇到任何麻烦，可以参考 *故障排除* 部分。

## 故障排除

这里有一些你可能遇到的错误及其修复方法：

+   I2C 传感器返回错误字符串：

    +   检查 SDA 和 SCL 引脚的连接。

    +   确保你在传感器的读数周期之间提供足够的延迟。检查数据表中的延迟和消息序列。

+   按钮按下时，图表窗口闪烁而不是保持显示：

    +   不要多次尝试按它。握住并快速释放。确保你的按钮连接正确。

    +   调整 Arduino 脚本中的延迟。

# 第二阶段 – 使用 Raspberry Pi 部署可用的恒温器

我们现在已经创建了一个恒温器，它作为一个 Arduino 原型存在，同时 Python 程序从您的电脑上运行。由于连接的电脑和如果您使用台式电脑时的显示器，这个原型仍然离可部署或便携状态相去甚远。一个现实世界的恒温器设备应该具有小巧的尺寸、便携的体积和微型显示屏来显示有限的信息。实现这一目标的流行且实用的方法是使用一个能够运行操作系统并提供基本 Python 编程接口的小型单板计算机。对于这个项目阶段，我们将使用一个单板计算机——树莓派——它配备了一个小型的 LCD 显示屏。

### 注意

注意，除非您想将项目扩展到日常可用的设备，否则这个项目阶段是可选的。如果您只是想学习 Python 编程，您可以跳过这一整个部分。

下面的图片是树莓派 Model B：

![阶段 2 - 使用树莓派构建可部署的恒温器](img/5938OS_07_17.jpg)

如果您之前没有使用过单板计算机，您可能会有很多未解答的问题，例如“树莓派究竟由什么组成？”、“在我们的项目中使用树莓派有什么好处？”以及“我们不能只用 Arduino 吗？”这些问题都是合理的，我们将在下一节中尝试回答它们。

## 什么是树莓派？

树莓派是一款小型（几乎与信用卡大小相当）的单板计算机，最初旨在帮助学生学习计算机科学的基础知识。如今，在树莓派基金会的指导下，树莓派运动已经变成了一种 DIY 现象，吸引了全球爱好者和开发者的关注。树莓派以低廉的成本（35 美元）提供的功能和特性，极大地提升了该设备的人气。

单板计算机这个术语用于指代那些在一个板上集成所有运行操作系统所需组件的设备，例如处理器、RAM、图形处理器、存储设备和基本的扩展适配器。这使得单板计算机成为便携式应用的合适候选者，因为它们可以成为我们试图创建的便携式硬件设备的一部分。尽管在树莓派推出之前市场上已经存在许多单板计算机，但硬件的开放源代码性质和经济价格是树莓派流行和快速采用的主要原因。以下图显示了树莓派 Model B 及其主要组件：

![什么是树莓派？](img/5938OS_07_18.jpg)

Raspberry Pi 的计算能力足以运行 Linux OS 的精简版。尽管人们尝试在 Raspberry Pi 上使用许多类型的操作系统，但我们将使用默认和推荐的操作系统，称为**Raspbian**。Raspbian 是基于 Debian 发行版的开源 Linux 操作系统，针对 Raspberry Pi 进行了优化。Raspberry Pi 使用 SD 卡作为存储设备，将用于存储您的操作系统和程序文件。在 Raspbian 中，您可以避免运行传统操作系统附带的不必要组件。这些包括网络浏览器、通信应用程序，在某些情况下甚至包括图形界面。

在其推出后，Raspberry Pi 经历了几次重大升级。早期版本称为**Model A**，不包括以太网端口，只有 256 MB 的内存。在我们的项目中，我们使用的是带有专用以太网端口、512 MB 内存和双 USB 端口的 Raspberry Pi Model B。最新的 Raspberry Pi 版本 Model B+也可以使用，因为它也配备了以太网端口。

## 安装操作系统和配置 Raspberry Pi

虽然 Raspberry Pi 是一台计算机，但在连接外围设备方面与传统台式计算机不同。Raspberry Pi 不支持传统的 VGA 或 DVI 显示端口，而是为电视提供 RCA 视频端口，为最新一代的显示器和电视提供 HDMI 端口。此外，Raspberry Pi 只有两个 USB 端口，需要用于连接各种外围设备，如鼠标、键盘、USB 无线适配器和 USB 闪存盘。让我们开始收集组件和电缆，以便开始使用 Raspberry Pi。

### 您需要什么来开始使用 Raspberry Pi？

开始使用 Raspberry Pi 所需的硬件组件如下：

+   **Raspberry Pi**：在这个项目阶段，您需要一个版本为 Model B 或最新的 Raspberry Pi。您可以从[`www.raspberrypi.org/buy`](http://www.raspberrypi.org/buy)购买 Raspberry Pi。

+   **电源线**：Raspberry Pi 使用 5V 直流电，至少需要 750 mA 的电流。电源通过位于板上的微型 USB 端口施加。在这个项目中，您需要一个微型 USB 电源。可选地，您可以使用基于微型 USB 的手机充电器为 Raspberry Pi 供电。

+   **显示器连接线**：如果您有一台 HDMI 显示器或电视，您可以使用 HDMI 线将其连接到您的 Raspberry Pi。如果您想使用基于 VGA 或 DVI 的显示器，您将需要一个 VGA 到 HDMI 或 DVI 到 HDMI 适配器转换器。您可以从 Amazon 或 Best Buy 购买这些适配器转换器。

+   **SD 卡**：您至少需要一个 8GB 的 SD 卡才能开始。最好使用质量为 4 级或更好的 SD 卡。您还可以在[`swag.raspberrypi.org/collections/frontpage/products/noobs-8gb-sd-card`](http://swag.raspberrypi.org/collections/frontpage/products/noobs-8gb-sd-card)购买预装操作系统的 SD 卡。

    ### 注意

    Raspberry Pi Model B+需要 microSD 卡而不是常规 SD 卡。

+   **鼠标和键盘**：您将需要一个标准的 USB 键盘和一个 USB 鼠标来与 Raspberry Pi 一起工作。

+   **USB 集线器（可选）**：由于 Model B 只有两个 USB 端口，如果您想连接 Wi-Fi 适配器或内存棒，您将不得不从 USB 端口移除现有设备以腾出空间。USB 集线器可以方便地将多个外围组件连接到您的 Raspberry Pi 上。我们建议您使用带外部电源的 USB 集线器，因为由于电源限制，Raspberry Pi 只能通过 USB 端口驱动有限数量的外围设备。

### 准备 SD 卡

要安装和配置如 Python 和所需库等软件组件，首先我们需要为 Raspberry Pi 提供一个操作系统。Raspberry Pi 官方支持基于 Linux 的开源操作系统，这些操作系统预先配置了针对定制 Raspberry Pi 硬件组件。这些操作系统的各种版本可在 Raspberry Pi 的网站上找到（[`www.raspberrypi.org/downloads`](http://www.raspberrypi.org/downloads)）。

Raspberry Pi 的网站为从新手到专家的各种用户提供了各种操作系统。对于初学者来说，很难识别合适的操作系统及其安装过程。如果您是第一次尝试使用 Raspberry Pi，我们建议您使用**新开箱即用软件**（**NOOBS**）包。从之前的链接下载`NOOBS`的最新版本。`NOOBS`包包括几个不同的操作系统，如 Raspbian、Pidora、Archlinux 和 RaspBMC。`NOOBS`简化了整个安装过程，并帮助您轻松安装和配置您首选的操作系统版本。需要注意的是，`NOOBS`只是一个安装包，一旦完成给定的安装步骤，您将只剩下 Raspbian 操作系统。

Raspberry Pi 使用 SD 卡来托管操作系统，您需要在将 SD 卡放入 Raspberry Pi 的 SD 卡槽之前，从您的电脑上准备 SD 卡。将您的 SD 卡插入电脑，并确保您备份了 SD 卡上任何重要的信息。在安装过程中，您将丢失 SD 卡上存储的所有数据。让我们先从准备您的 SD 卡开始。

按照以下步骤从 Windows 准备 SD 卡：

1.  你需要一个软件工具来格式化和准备 SD 卡以供 Windows 使用。你可以从[`www.sdcard.org/downloads/formatter_4/eula_windows/`](https://www.sdcard.org/downloads/formatter_4/eula_windows/)下载免费提供的格式化工具。

1.  在你的 Windows 计算机上下载并安装格式化工具。

1.  插入你的 SD 卡并启动格式化工具。

1.  在格式化工具中，打开**选项**菜单并将**格式大小调整**设置为**开启**。

1.  选择合适的 SD 卡并点击**格式化**。

1.  然后，等待格式化工具完成格式化 SD 卡。一旦完成，将下载的`NOOBS`ZIP 文件提取到 SD 卡上。确保你将 ZIP 文件夹的内容提取到 SD 卡的根目录。

按照以下指示从 Mac OS X 准备 SD 卡：

1.  你需要一个软件工具来格式化和准备 SD 卡以供 Mac OS X 使用。你可以从[`www.sdcard.org/downloads/formatter_4/eula_mac/`](https://www.sdcard.org/downloads/formatter_4/eula_mac/)下载免费提供的格式化工具。

1.  在你的机器上下载并安装格式化工具。

1.  插入你的 SD 卡并运行格式化工具。

1.  在格式化工具中，选择**覆盖格式**。

1.  选择合适的 SD 卡并点击**格式化**。

1.  然后，等待格式化工具完成格式化 SD 卡。一旦完成，将下载的`NOOBS`ZIP 文件提取到 SD 卡上。确保你将 ZIP 文件夹的内容提取到 SD 卡的根目录。

按照以下步骤从 Ubuntu Linux 准备 SD 卡：

1.  要在 Ubuntu 上格式化 SD 卡，你可以使用一个名为`gparted`的格式化工具。在终端中使用以下命令安装`gparted`：

    ```py
    $ sudo apt-get install gparted

    ```

1.  插入你的 SD 卡并运行`gparted`。

1.  在`gparted`窗口中，选择整个 SD 卡并使用**FAT32**格式化它。

1.  一旦格式化过程完成，将下载的`NOOBS`ZIP 文件提取到 SD 卡上。确保你将 ZIP 文件夹的内容提取到 SD 卡的根目录。

    ### 小贴士

    如果你在这几个步骤中遇到任何问题，你可以参考[`www.raspberrypi.org/documentation/installation/installing-images/`](http://www.raspberrypi.org/documentation/installation/installing-images/)中为 Raspberry Pi 准备 SD 卡的官方文档。

### Raspberry Pi 的设置过程

一旦你用`NOOBS`准备好了你的 SD 卡，将其插入 Raspberry Pi 的 SD 卡槽中。在连接电源适配器的微型 USB 线之前，先连接你的显示器、鼠标和键盘。一旦连接了电源适配器，Raspberry Pi 将自动开机，你将能在显示器上看到安装过程。如果你在连接电源适配器后无法在显示器上看到任何进度，请参考本章后面可用的故障排除部分。

一旦 Raspberry Pi 启动，它将重新分区 SD 卡并显示以下安装屏幕，以便您开始使用：

![Raspberry Pi 设置过程](img/5938OS_07_19.jpg)

### 注意

上述截图由 Simon Monk 从`raspberry_pi_F01_02_5a.jpg`中获取，并授权于 Attribution Creative Commons 许可([`learn.adafruit.com/assets/11384`](https://learn.adafruit.com/assets/11384))。

1.  作为首次用户，请选择**Raspbian [推荐]**作为推荐的操作系统，并点击**安装 OS**按钮。Raspbian 是一个基于 Debian 的操作系统，针对 Raspberry Pi 进行了优化，并支持我们在前几章中学到的有用 Linux 命令。整个过程大约需要 10 到 20 分钟才能完成。

1.  成功完成后，您将能够看到类似于以下截图的屏幕。该截图显示了`raspi-config`工具，它将允许您设置初始参数。我们将跳过此过程以完成安装。选择**<完成>**并按*Enter*：![Raspberry Pi 设置过程](img/5938OS_07_20.jpg)

1.  如果您想更改任何参数，可以再次回到这个屏幕，在终端中输入以下命令：

    ```py
    $ sudo raspi-config

    ```

1.  现在，Raspberry Pi 将重新启动，您将被提示默认登录屏幕。使用默认用户名`pi`和密码`raspberry`登录。

1.  您可以通过在终端中输入以下命令来启动 Raspberry Pi 的图形桌面：

    ```py
    $ startx

    ```

1.  要运行我们在第一阶段开发的 Python 代码，您需要在 Raspberry Pi 上设置所需的 Python 库。您需要使用以太网线将 Raspberry Pi 连接到互联网以安装包。使用以下命令在 Raspberry Pi 终端上安装所需的 Python 包：

    ```py
    $ sudo apt-get install python-setuptools, python-matplotlib, python-numpy

    ```

1.  使用 Setuptools 安装`pySerial`：

    ```py
    $ sudo easy_install pyserial

    ```

现在，您的 Raspberry Pi 已经准备好操作系统和必要的组件来支持 Python-Arduino 编程。

## 使用 Raspberry Pi 和便携式 TFT LCD 显示器

TFT LCD 是扩展 Raspberry Pi 功能并避免使用大型显示设备的好方法。这些 TFT LCD 显示器可以直接与 GPIO 引脚接口。TFT LCD 屏幕有各种形状和尺寸，但鉴于接口方便，我们建议您使用小于或等于 3.2 英寸的屏幕。大多数这些小型屏幕不需要额外的电源供应，可以直接使用 GPIO 引脚供电。在少数情况下，也提供触摸屏版本，以扩展 Raspberry Pi 的功能。

在本项目中，我们使用的是一款可直接通过 GPIO 与 Raspberry Pi 接口的 Tontec 2.4 英寸 TFT LCD 屏幕。虽然您可以使用任何可用的 TFT LCD 屏幕，但本书仅涵盖此特定屏幕的设置和配置过程。在大多数情况下，这些屏幕的制造商在其网站上提供了详细的配置教程。如果您使用的是不同类型的 TFT LCD 屏幕，Raspberry Pi 论坛和博客是寻找帮助的另一个好地方。以下图片显示了 Tontec 2.4 英寸 TFT LCD 屏幕的背面，以及 GPIO 引脚的位置。让我们开始使用这款屏幕与您的 Raspberry Pi 一起工作：

![使用便携式 TFT LCD 显示屏与 Raspberry Pi](img/5938OS_07_21.jpg)

### 使用 GPIO 连接 TFT LCD

在我们能够使用屏幕之前，我们必须将其连接到 Raspberry Pi。让我们从 Raspberry Pi 上断开微型 USB 电源适配器，并找到位于 Raspberry Pi RCA 视频端口附近的 GPIO 阳性引脚。取下您的 TFT 屏幕，并按照以下图片所示连接 GPIO 引脚。在少数情况下，屏幕上的标记可能会误导，因此我们建议您遵循制造商的指南进行连接：

![使用 GPIO 连接 TFT LCD](img/5938OS_07_22.jpg)

当您的屏幕连接到 Raspberry Pi 后，使用微型 USB 线缆为其供电。请勿断开 HDMI 线缆，因为您的屏幕尚未准备好。在我们进行任何配置步骤之前，让我们首先将 Raspberry Pi 连接到互联网。使用以太网线缆将 Raspberry Pi 的以太网端口连接到您的家庭或办公室网络。现在，让我们在 Raspbian OS 中配置 TFT LCD 屏幕，使其正常工作。

### 使用 Raspberry Pi OS 配置 TFT LCD

当您的 Raspberry Pi 启动后，使用您的用户名和密码登录。完成以下步骤以使用 Raspberry Pi 配置屏幕：

1.  使用以下命令在终端下载支持文件和手册：

    ```py
    $ wget https://s3.amazonaws.com/tontec/24usingmanual.zip

    ```

1.  解压文件。以下命令将文件提取到同一目录：

    ```py
    $ unzip 24usingmanual.zip

    ```

1.  导航到 `src` 目录：

    ```py
    $ cd cd mztx-ext-2.4/src/

    ```

1.  输入以下命令以编译源文件：

    ```py
    $ make

    ```

1.  打开引导配置文件：

    ```py
    $ sudo pico /boot/config.txt

    ```

1.  在 `config.txt` 文件中，找到并取消以下代码行的注释：

    ```py
    framebuffer_width=320
    framebuffer_height=240
    ```

1.  保存并退出文件。

1.  现在，每次 Raspberry Pi 重启时，我们都需要执行一个命令来启动 TFT LCD 屏幕。为此，使用以下命令打开 `rc.local` 文件：

    ```py
    $ sudo pico /etc/rc.local

    ```

1.  将以下代码行添加到启动屏幕的文件中：

    ```py
    sudo /home/pi/mztx-ext-2.4/src/mztx06a &
    ```

1.  保存并退出文件。然后，使用以下命令重启 Raspberry Pi：

    ```py
    $ sudo reboot

    ```

现在，你可以移除你的 HDMI 显示器，开始使用你的 TFT LCD 屏幕。你必须记住的一件事是屏幕分辨率非常小，并且它没有针对编码进行优化。我们更喜欢使用 HDMI 显示器来执行下一节所需的重大代码修改。在这个项目中使用 TFT LCD 屏幕是为了满足恒温器的移动性和便携性要求。

## 优化 TFT LCD 屏幕的 GUI

我们在上一节配置的 TFT LCD 屏幕的分辨率仅为 320 x 240 像素，但我们创建的第一编程阶段中的窗口相当大。因此，在我们将 Python 代码复制并运行在树莓派上之前，我们需要在代码中调整一些参数。

在你的常规计算机上，从书籍源代码中获取这一章文件夹的地方，打开`Thermostat_Stage2.py`文件。此文件包含实现最佳尺寸所需修改的详细信息，并进行了一些细微的美观改动。你将使用这个文件，而不是我们在前一阶段使用的文件，在你的树莓派上。代码中的这些调整将在以下代码行中解释。

第一次主要改动是在端口名称上。对于树莓派，你需要将你在第一阶段使用的 Arduino 端口的名称更改为`/dev/ttyACM0`，这是在大多数情况下分配给 Arduino 的地址：

```py
port = serial.Serial('/dev/ttyACM0',9600, timeout=1)
```

在这个程序文件中，`Tkinter`主窗口和`matplotlib`图的大小也调整以适应屏幕大小。如果你使用的是不同尺寸的屏幕，请相应地更改以下代码行：

```py
top.minsize(320,160)
pyplot.figure(figsize=(4,3))
```

现在，随着前面的更改，GUI 窗口应该能够适应树莓派的屏幕。由于树莓派的屏幕将被用作恒温器应用的专用屏幕，我们需要调整屏幕上的文本大小以适当地适应窗口。在标签的声明中添加`font=("Helvetica", 20)`文本以增加字体大小。以下代码行显示了在标签上执行的变化，以包含传感器名称：

```py
Tkinter.Label(top,
              text="Humidity",
              font=("Helvetica", 20)).grid(column=1, row=2)
```

同样，`font`选项也被添加到观察标签中：

```py
HumdUnitLabel = Tkinter.Label(top,
                              text="%",
                              font=("Helvetica", 20))
```

观察单位的标签也进行了类似的修改：

```py
HumdLabel.config(text=cleanText(reading[1]),
                 font=("Helvetica", 20))
```

`Thermostat_ Stage2.py`文件已包含前面的修改，并已准备好在您的 Raspberry Pi 上运行。在运行文件之前，我们首先需要将其复制到 Raspberry Pi 上。在这个阶段，USB 集线器将非常有用，可以复制文件。如果您没有 USB 集线器，您可以使用两个可用的 USB 端口同时连接 USB 闪存盘、鼠标和键盘。使用 USB 集线器，将包含 Python 文件的 USB 闪存盘连接到 USB 集线器的一端，并将它们复制到主目录。将 Arduino 板的 USB 端口连接到 USB 集线器的一端。从 Raspberry Pi 的开始菜单，通过导航到**附件** | **LXTerminal**来打开**LXTerminal**程序。从主目录运行 Python 代码，你将能够在 Raspberry Pi 的屏幕上看到优化的用户界面窗口。如果本章中提到的每个步骤都执行正确，当你点击**开始**按钮时，你将能够看到传感器观察结果被打印出来：

![优化 TFT 液晶屏的 GUI](img/5938OS_07_23.jpg)

在本章结束时，你可能想知道一个带有传感器、Arduino、Raspberry Pi 和 TFT 屏幕的移动单元可能是什么样子。以下图像显示了本章中给出的说明开发的样品恒温器。我们使用亚克力板将 Raspberry Pi 和 Arduino 板固定在一起，并创建了一个紧凑的形态：

![优化 TFT 液晶屏的 GUI](img/5938OS_07_24.jpg)

## 故障排除

在这个项目阶段，你可能会遇到一些已知的问题。以下部分描述了这些问题及其快速修复方法：

+   Raspberry Pi 无法启动：

    +   确保使用指定的工具正确格式化 SD 卡。如果 SD 卡没有正确准备，Raspberry Pi 将无法启动。

    +   检查 HDMI 线和显示器，看它们是否工作正常。

    +   确保电源适配器与 Raspberry Pi 兼容。

+   TFT 液晶屏未开启：

    +   确保屏幕已正确连接到 Raspberry Pi 的 GPIO 引脚。

    +   如果你使用的是其他 TFT 液晶屏，请确保从其数据表中确认你的屏幕不需要额外的电源。

    +   使用*优化 TFT 液晶屏的 GUI*部分中描述的步骤检查屏幕是否正确配置。

+   Raspberry Pi 上传感器数据的刷新率较慢：

    +   尝试减少 Arduino 发送的每个串行消息之间的延迟。

    +   终止任何在后台运行的其他应用程序。

# 摘要

通过这个项目，我们成功创建了一个便携式和可部署的恒温器，使用 Arduino 进行温度、湿度和环境光监测。在这个过程中，我们使用必要的组件组装了恒温器传感器单元，并开发了定制的 Arduino 程序来支持它们。我们还利用了 Python 编程方法，包括使用`Tkinter`库进行 GUI 开发和使用`matplotlib`库进行绘图。在章节的后面部分，我们使用了 Raspberry Pi 将一个简单的项目原型转化为实际应用。从现在起，你应该能够开发出类似的项目，这些项目需要你观察和可视化实时传感器信息。

在接下来的工作中，我们将扩展这个项目以适应即将到来的主题，例如 Arduino 网络、云通信和远程监控。在恒温器项目的下一个阶段，我们将集成这些高级功能，使其成为一个真正有价值的 DIY 项目，可以在日常生活中使用。在下一章中，我们将开始我们的旅程的下一阶段，从制作简单的 Python-Arduino 项目过渡到互联网连接和远程访问的物联网项目。
