# 第十章. 最终项目 – 一个远程家庭监控系统

现在是时候将我们在前几章中学到的每个主题结合起来，创建一个结合 Arduino 编程、Python GUI 开发、MQTT 消息协议和基于 Python 的云应用的项目。正如你可能已经从章节标题中推测出的那样，我们将使用这些组件开发一个远程家庭监控系统。

本章的第一部分涵盖了项目设计过程，包括目标、需求、架构和 UX。一旦我们完成设计过程，我们将进入项目的实际开发，这分为三个独立阶段。接下来，我们将涵盖在处理大型项目时通常会遇到的一些常见故障排除主题。在我们努力开发可用的 DIY 项目时，后面的部分涵盖了扩展项目的技巧和功能。由于与其他书籍中的项目相比，这是一个相当大的项目，我们不会在没有任何策略的情况下直接进入实际开发过程。让我们首先熟悉硬件项目的标准设计方法。

# 物联网项目的设计方法

开发一个将硬件设备与高级软件服务紧密耦合的复杂产品需要额外的规划层次。对于这个项目，我们将采用适当的产品开发方法，帮助你熟悉创建真实世界硬件项目的流程。然后，可以使用这种方法来规划你自己的项目并将它们提升到下一个层次。以下图表描述了一个典型的原型开发过程，它始终从定义你希望通过产品实现的主要目标开始：

![物联网项目的设计方法](img/5938OS_10_01.jpg)

一旦你定义了主要目标集合，你需要将它们分解成项目需求，这些需求包括实现这些目标时原型应执行的每个任务的详细信息。使用项目需求，你需要勾勒出系统的整体架构。下一步包括定义 UX 流程的过程，这将帮助你规划系统的用户交互点。在这个阶段，你将能够识别系统架构、硬件和软件组件中所需的所有更改，以便开始开发。

既然你已经定义了交互点，现在你需要将整个项目开发过程分成多个阶段，并在这些阶段之间分配任务。一旦你完成了这些阶段的发展，你将不得不根据你的架构将这些阶段相互连接，并在需要时调试组件。最后，你必须将你的项目作为一个整体系统进行测试，并解决小问题。在硬件项目中，在复杂开发过程完成后再次处理你的电线路是非常困难的，因为变化可能会对所有其他组件产生重复影响。这个过程将帮助你最小化任何硬件返工和随后的软件修改。

现在你已经了解了方法论，让我们开始实际开发我们的远程家庭监控系统。

# 项目概述

智能家居是物联网中最定义明确且最受欢迎的子领域之一。任何智能家居最重要的功能是其监控物理环境的能力。幸运的是，我们在前几章中涵盖的练习和项目包括可用于相同目的的组件和功能。在本章中，我们将定义一个将利用这些现有组件和编程练习的项目。在第七章的中期项目中，即“中期项目 – 便携式 DIY 恒温器”，我们创建了一个可部署的恒温器，能够测量温度、湿度和环境光线。如果我们想利用这个中期项目，我们可以在其基础上构建的最近的物联网项目是远程家庭监控系统。该项目将以 Arduino 作为物理环境和基于软件的服务之间的主要交互点。我们将有一个 Python 程序作为中间层，它将连接来自 Arduino 的传感器信息与面向用户的图形界面。让我们首先定义我们想要实现的目标以及满足这些目标的项目需求。

## 项目目标

Nest 恒温器提供了一个关于一个设计良好的远程监控系统应具备的特性的例子，该系统具有专业功能。实现这一级别的系统能力需要来自大型团队的大量开发工作。尽管很难在我们的项目中包含商业系统支持的每个功能，但我们仍将尝试实现原型项目可以整合的常见功能。

我们计划在这个项目中整合的顶级功能可以通过以下目标来描述。

+   观察物理环境并使其远程可访问

+   向用户提供基本级别的控制以与系统交互

+   展示基本的内置情境意识

## 项目需求

既然我们已经定义了主要目标，让我们将它们转换为详细的系统需求。项目完成后，系统应能够满足以下要求：

+   它必须能够观察物理现象，如温度、湿度、运动和周围光线。

+   它应提供对传感器信息和执行器（如蜂鸣器、按钮开关和 LED）的本地访问和控制。

+   监控应由使用开源硬件平台 Arduino 开发的单元进行。

+   监控单元应限于收集传感器信息并将其传达给控制单元。

+   控制单元不应包含台式计算机或笔记本电脑。相反，它应该使用像 Raspberry Pi 这样的平台来部署。

+   控制单元应通过利用收集到的传感器信息展示原始级别的态势感知能力。

+   控制单元应具有图形界面，以提供传感器的观察结果和系统的当前状态。

+   系统必须通过基于云的服务通过互联网访问。

+   提供远程访问的 Web 应用程序应具有通过 Web 浏览器显示传感器观察结果的能力。

+   系统还应提供对执行器的基本控制，通过使用 Web 应用程序完成远程访问体验。

+   由于监控单元可能受到计算资源的限制，系统应使用面向硬件的消息协议来传输信息。

尽管还有许多其他可能成为我们项目一部分的次要要求，但它们在这本书中被省略了。如果你对你的远程家庭监控系统有任何额外的计划，这是你必须在你开始设计架构之前定义这些要求的时候。对需求的任何未来更改都可能显著影响开发阶段，并使硬件和软件修改变得困难。在章节的最后部分，我们列出了一些你可能希望考虑实现于未来项目的附加功能。

## 设计系统架构

从项目目标继续，首先，你需要绘制出系统的概要架构。这个架构草图应包括使系统能够在传感器和远程用户之间传递信息的主要组件。以下图显示了我们的项目架构草图：

![设计系统架构](img/5938OS_10_02.jpg)

根据目标，用户应能够通过互联网访问系统；这意味着我们需要在架构中包含云组件。系统还需要使用资源受限的设备来监控物理环境，这可以使用 Arduino 实现。连接云服务和传感器系统的中间层可以使用 Raspberry Pi 构建。在上一个项目中，我们通过串行连接连接了 Arduino 和 Raspberry Pi，但我们希望摆脱串行连接，开始使用我们家的以太网网络来使系统可部署。因此，基于 Arduino 的单元通过以太网连接到网络，而 Raspberry Pi 则使用 Wi-Fi 连接到同一网络。

为了布局整体系统架构，让我们利用我们设计的草图，如图所示。正如您在下一张图中可以看到的，我们将整体系统转换成了三个主要架构单元：

+   监控站

+   控制中心

+   云服务

在这个图中，我们已经针对我们将要利用的每个主要组件及其相互关联进行了说明。在接下来的章节中，我们将简要定义这三个主要单元。这些单元的详细描述和实现步骤将在本章的单独部分提供。

![设计系统架构](img/5938OS_10_03.jpg)

### 监控站

我们需要一个资源受限且健壮的单元，该单元将定期与物理环境进行通信。这个监控单元可以使用 Arduino 构建，因为低级微控制器编程可以提供不间断的传感器数据流。在这个阶段使用 Arduino 也将帮助我们避免将基本低级传感器直接与运行在复杂操作系统上的计算机进行接口连接。传感器和执行器通过数字、模拟、PWM 和 I2C 接口连接到 Arduino。

### 控制中心

控制中心作为传感器信息和用户之间的主要交互点。它还负责将监控站中的传感器信息传递到云服务。控制中心可以使用您的普通计算机或单板计算机（如 Raspberry Pi）开发。我们将使用 Raspberry Pi，因为它可以轻松部署为硬件单元，并且它也足以托管 Python 程序。我们将用一个小型 TFT LCD 屏幕替换 Raspberry Pi 的计算机屏幕来显示 GUI。

### 云服务

云服务的主要目的是为控制中心提供一个基于互联网的接口，以便用户可以远程访问它。在我们托管一个网络应用程序来执行此操作之前，我们需要一个中间数据中继。这个传感器数据中继充当基于云的网络应用程序和控制中心之间的主机。在这个项目中，我们将使用 Xively 作为平台来收集这些传感器数据。网络应用程序可以托管在互联网服务器上；在我们的情况下，我们将使用我们熟悉的 Amazon AWS。

## 定义 UX 流程

现在，尽管我们知道整个系统的架构看起来是什么样子，但我们还没有定义用户将如何与之交互。为我们的系统设计用户交互的过程也将帮助我们弄清楚主要组件之间的数据流。

让我们从您家中本地运行的组件开始，即监测站和控制中心。如图所示，我们在控制中心有我们的第一个用户交互点。用户可以观察信息或对其采取行动，如果系统状态是警报。取消警报的用户操作会在控制中心和监测站引发多个操作。我们建议您仔细查看图表，以更好地理解系统的流程。

![定义 UX 流程](img/5938OS_10_04.jpg)

同样，第二个用户交互点位于网络应用程序中。网络应用程序显示我们在控制中心计算出的观察结果和系统状态，并提供一个界面来取消警报。在这种情况下，取消操作将通过 Xively 传输到控制中心，控制中心的适当操作将与之前的情况相同。然而，在网络应用程序中，用户每次都必须加载网络浏览器来请求数据，而这在控制中心是自动发生的。请查看以下图表以了解网络应用程序的用户体验流程：

![定义 UX 流程](img/5938OS_10_05.jpg)

## 所需组件列表

项目所需组件是根据以下三个主要标准推导出来的：

+   易于获取

+   与 Arduino 板兼容

+   由于在此书中之前的使用而熟悉组件

这是您开始项目所需的组件列表。如果您已经完成了之前的练习和项目，您应该已经拥有了大部分组件。如果您不想拆解项目，您可以从 SparkFun、Adafruit 或 Amazon 的网站上获取它们，这些网站的链接将在下一表中提供。

监测站点的硬件组件如下：

| 组件（第一阶段） | 数量 | 链接 |
| --- | --- | --- |
| Arduino Uno | 1 | [`www.sparkfun.com/products/11021`](https://www.sparkfun.com/products/11021) |
| Arduino 以太网盾 | 1 | [`www.sparkfun.com/products/9026`](https://www.sparkfun.com/products/9026) |
| 面包板 | 1 | [`www.sparkfun.com/products/9567`](https://www.sparkfun.com/products/9567) |
| TMP102 温度传感器 | 1 | [`www.sparkfun.com/products/11931`](https://www.sparkfun.com/products/11931) |
| HIH-4030 湿度传感器 | 1 | [`www.sparkfun.com/products/9569`](https://www.sparkfun.com/products/9569) |
| 小型光敏电阻 | 1 | [`www.sparkfun.com/products/9088`](https://www.sparkfun.com/products/9088) |
| PIR 运动传感器 | 1 | [`www.sparkfun.com/products/8630`](https://www.sparkfun.com/products/8630) |
| 超亮 RGB LED，共阳极 | 1 | [`www.adafruit.com/product/314`](http://www.adafruit.com/product/314) |
| 蜂鸣器 | 1 | [`www.adafruit.com/products/160`](http://www.adafruit.com/products/160) |
| 按钮开关 | 1 | [`www.sparkfun.com/products/97`](https://www.sparkfun.com/products/97) |
| Arduino 开发阶段用 USB 线 | 1 | [`www.sparkfun.com/products/512`](https://www.sparkfun.com/products/512) |
| Arduino 部署阶段电源 | 1 | [`www.amazon.com/Arduino-9V-1A-Power-Adapter/dp/B00CP1QLSC/`](http://www.amazon.com/Arduino-9V-1A-Power-Adapter/dp/B00CP1QLSC/) |
| 电阻 | 如需 | 220 欧姆、1 千欧姆和 10 千欧姆 |
| 连接线 | 如需 | 无 |

控制中心的硬件组件如下：

| 组件（第一阶段） | 数量 | 链接 |
| --- | --- | --- |
| 树莓派 | 1 | [`www.sparkfun.com/products/11546`](https://www.sparkfun.com/products/11546) |
| TFT LCD 屏幕 | 1 | [`www.amazon.com/gp/product/B00GASHVDU/`](http://www.amazon.com/gp/product/B00GASHVDU/) |
| SD 卡（8 GB） | 1 | [`www.sparkfun.com/products/12998`](https://www.sparkfun.com/products/12998) |
| Wi-Fi 拓展卡 | 1 | [`www.amazon.com/Edimax-EW-7811Un-150Mbps-Raspberry-Supports/dp/B003MTTJOY`](http://www.amazon.com/Edimax-EW-7811Un-150Mbps-Raspberry-Supports/dp/B003MTTJOY) |
| 树莓派电源 | 1 | [`www.amazon.com/CanaKit-Raspberry-Supply-Adapter-Charger/dp/B00GF9T3I0`](http://www.amazon.com/CanaKit-Raspberry-Supply-Adapter-Charger/dp/B00GF9T3I0) |
| 键盘、鼠标、USB 集线器和显示器 | 如需 | 开发和调试阶段所需 |

## 定义项目开发阶段

根据系统架构，我们拥有三个主要单元，它们协同创建远程家庭监控系统项目。整体硬件和软件开发流程也与这三个单元相一致，可以如下分配：

+   监控站开发阶段

+   控制中心开发阶段

+   网络应用程序开发阶段

监控站阶段的软件开发包括编写 Arduino 代码以监控传感器并执行执行器动作，同时将此信息发布到控制中心。开发阶段的中间层，即基于 Raspberry Pi 的控制中心，托管了 Mosquitto 代理。此阶段还包括包含 GUI、态势感知逻辑和与 Xively 云服务通信的子例程的 Python 程序。最后阶段，云服务，包括两个不同的组件，传感器数据中继和 Web 应用程序。我们将使用 Xively 平台作为我们的传感器数据中继，并将使用 Python 在 Amazon AWS 云实例上开发 Web 应用程序。现在，让我们进入实际的开发过程，我们的第一个目的地将是基于 Arduino 的监控站。

# 第 1 阶段 - 使用 Arduino 的监控站

正如我们所讨论的，监控系统的主要任务是接口传感器组件并将这些传感器生成的信息传达给观察者。你将使用 Arduino Uno 作为中央微控制器组件来集成这些传感器和执行器。我们还需要 Arduino Uno 和控制中心之间的通信手段，我们将利用 Arduino 以太网盾来实现这一目的。让我们讨论监控站的硬件架构及其组件。

## 设计监控站

我们已经在第八章“Arduino 网络介绍”和第九章“Arduino 与物联网”的各种练习中设计了基于 Arduino 和以太网盾的单元。因此，我们假设你已经熟悉将以太网盾与 Arduino 板进行接口。我们将使用 Arduino 板连接各种传感器和执行器，如下面的图所示。如图所示，传感器将向 Arduino 板提供数据，而执行器将从 Arduino 板获取数据。尽管我们自动收集这些传感器的环境数据，但按钮的数据将通过手动用户输入进行收集。

![设计监控站](img/5938OS_10_06.jpg)

查看以下 Fritzing 图以了解监控站中的详细连接。正如你在我们的硬件设计中看到的那样，温度传感器 TMP102 通过 I2C 接口连接，这意味着我们需要 SDA 和 SCL 线。我们将使用 Arduino 板的模拟引脚 5 和 6 来分别接口 SDA 和 SCL。湿度（HIH-4030）和环境光传感器也提供模拟输出，并连接到 Arduino 板的模拟引脚。同时，蜂鸣器、按钮开关和 PIR 运动传感器通过数字 I/O 引脚连接。超级流明 RGB LED 是正极 LED；这意味着它总是使用正极引脚供电，而 R、G 和 B 引脚通过 PWM 引脚控制。

确保你将所有组件正确连接到以下图中指定的引脚：

![设计监控站](img/5938OS_10_07.jpg)

### 注意

你可以从[`learn.adafruit.com/all-about-leds`](https://learn.adafruit.com/all-about-leds)教程中了解更多关于 RGB LED 与 Arduino 接口的信息。

如果你使用的是除 Arduino Uno 以外的 Arduino 板，你将不得不在 Arduino 代码中调整适当的引脚编号。此外，确保这个 Arduino 板与以太网盾兼容。

在电路连接方面，你可以使用前面图中所示的面包板，或者如果你感到舒适，你可以使用 PCB 原型板并焊接组件。在我们的设置中，我们首先在面包板上测试了组件，一旦测试通过，我们就焊接了组件，如图所示。如果你尝试焊接 PCB 板，请确保你有完成这项工作的必要组件。与面包板相比，PCB 原型板将提供更坚固的性能，但这也将使你在之后调试和更换组件变得困难。

![设计监控站](img/5938OS_10_08.jpg)

如果你已经准备好了电路连接，请使用 USB 线将 Arduino 连接到你的电脑。同时，使用以太网线将以太网盾连接到你的家庭路由器。

## 监控站的 Arduino 草图

在进入编码阶段之前，请确保你已经收集了项目的预构建 Arduino 代码。你可以在本章的代码文件夹中找到它，文件名为`Arduino_monitoring_station.ino`。该代码实现了支持监控站整体 UX 流程所需的基本逻辑，这是我们之前章节中讨论过的。在接下来的章节中，我们将逐一介绍程序的主要部分，以便你更好地理解这些代码片段。现在，在 Arduino IDE 中打开这个草图。你已经熟悉了为 Arduino 设置 IP 地址。在上一个章节中，你也学习了如何使用 Arduino MQTT 库`PubSubClient`，这意味着你的 Arduino IDE 上应该已经安装了`PubSubClient`库。在代码的开头，我们还声明了一些常量，例如 MQTT 服务器的 IP 地址和 Arduino 的 IP 地址，以及各种传感器和执行器的引脚号。

### 注意

你需要根据你的网络设置更改监控站和控制中心的 IP 地址。确保在上传 Arduino 代码之前执行这些修改。

在代码结构中，我们有两个强制性的 Arduino 函数，`setup()`和`loop()`。在`setup()`函数中，我们将设置 Arduino 引脚类型和 MQTT 订阅通道。在同一个函数中，我们还将设置一个用于`publishData()`函数的计时器，并附加一个按钮按下时的中断。

### 发布传感器信息

`publishData()`函数读取传感器输入，并将这些数据发布到位于控制中心的 Mosquitto 代理。正如你在下面的代码片段中可以看到的，我们正在逐个测量传感器值，并使用`client.publish()`方法将它们发布到代理：

```py
void publishData (){
    Wire.requestFrom(partAddress,2);
    byte MSB = Wire.read();
    byte LSB = Wire.read();

    int TemperatureData = ((MSB << 8) | LSB) >> 4; 

    float celsius = TemperatureData*0.0625;
    temperatureC = dtostrf(celsius, 5, 2, message_buff2);
    client.publish("MonitoringStation/temperature", temperatureC);

    float humidity = getHumidity(celsius);
    humidityC = dtostrf(humidity, 5, 2, message_buff2);
    client.publish("MonitoringStation/humidity", humidityC);

    int motion = digitalRead(MotionPin);
    motionC = dtostrf(motion, 5, 2, message_buff2);
    client.publish("MonitoringStation/motion", motionC);

    int light = analogRead(LightPin);
    lightC = dtostrf(light, 5, 2, message_buff2);
    client.publish("MonitoringStation/light", lightC);
}
```

如果你查看`setup()`函数，你会注意到我们使用了一个名为`SimpleTimer`的库来为这个函数设置一个`timer`方法。该方法定期执行`publishData()`函数，而不会中断和阻塞 Arduino 执行周期的实际流程。在下面的代码片段中，数字`300000`代表毫秒级的延迟时间，即 5 分钟：

```py
timer.setInterval(300000, publishData);
```

### 注意

你需要下载并导入`SimpleTimer`库才能成功编译和运行代码。你可以从[`github.com/infomaniac50/SimpleTimer`](https://github.com/infomaniac50/SimpleTimer)下载该库。

### 订阅执行器动作

你可以在`setup()`函数中看到，我们通过订阅`MonitoringStation/led`和`MonitoringStation/buzzer`通道来初始化代码。`client.subscribe()`方法将确保每当 Mosquitto 代理收到这些通道的任何更新时，基于 Arduino 的监控系统都会得到通知：

```py
if (client.connect("MonitoringStation")) {
    client.subscribe("MonitoringStation/led");
    client.subscribe("MonitoringStation/buzzer");
  }
```

### 编程一个中断来处理按钮的按下

我们已经处理了监控站的发布和订阅功能。现在，我们需要集成由用户输入控制的按钮开关。在 Arduino 编程例程中，我们运行一个周期性循环来检查引脚的状态。然而，如果按钮被按下，这可能没有用，因为它需要立即采取行动。按下按钮的动作是通过 Arduino 中断来处理的，如下面的代码行所示：

```py
attachInterrupt(0, buttonPress, RISING);
```

上一行代码将引脚 0（数字引脚 2）的中断与`buttonPress()`函数关联。这个函数会在中断状态改变时触发蜂鸣器。换句话说，当用户按下按钮时，无论蜂鸣器的当前状态如何，蜂鸣器都会立即关闭：

```py
void buttonPress(){
    digitalWrite(BUZZER, LOW);
    Serial.println("Set buzzer off");
}
```

## 测试

当前的 Arduino 代码用于与控制中心通信以发布和订阅数据，但我们还没有设置 Mosquitto 代理来处理这些请求。您仍然可以使用 USB 线将 Arduino 草图上传到您的监控站。这将不会导致监控站有任何有益的行动，您只能使用`Serial.prinln()`命令来打印各种传感器测量值。因此，我们将开发控制中心，这样我们就可以开始处理来自监控站的通信请求。

# 第二阶段 - 使用 Python 和树莓派的控制中心

为了将系统状态和其他传感器观察结果传达给用户，控制中心需要执行各种操作，包括从监控站获取原始传感器数据，计算系统状态，将此数据报告给云服务，并使用 GUI 显示观察结果。虽然控制中心包括两个主要硬件组件（树莓派和 TFT 液晶显示屏），但它还包括两个主要软件组件（Mosquitto 代理和 Python 代码）来处理控制中心逻辑。

### 小贴士

我们使用树莓派而不是普通计算机，因为我们希望控制中心成为一个可部署和便携的单位，可以安装在墙上。

您仍然可以使用自己的计算机来编辑和测试用于开发目的的 Python 代码，而不是直接使用树莓派。然而，一旦您准备部署，我们建议您切换回树莓派。

## 控制中心架构

树莓派是控制中心的主要计算单元，作为整个系统的“大脑”。由于树莓派被用作普通计算机的替代品，控制中心的架构可以互换地使用计算机代替树莓派。正如您在下图中可以看到的，控制中心通过 Wi-Fi 连接到家庭网络，这将使其对监控站可访问。控制中心包括 Mosquitto 代理；这是监控站和用于控制中心的 Python 程序之间的通信点。Python 程序利用 `Tkinter` 库进行 GUI，并使用 `paho_mqtt` 库与 Mosquitto 代理通信。通过利用这两个库，我们可以将监控站的传感器信息传递给用户。然而，我们需要一个单独的安排来建立控制中心和云服务之间的通信。在我们的整体系统架构中，控制中心被设计为与中间数据中继 Xively 通信。Python 代码使用 `xively-python` 库来实现这种通信。

![控制中心架构](img/5938OS_10_09.jpg)

在 第八章，*Arduino 网络入门*中，我们已向您提供了安装 Mosquitto 代理、`Python-mosquitto` 库和 `xively-python` 库的方法。我们还在 第七章，*中期项目 – 便携式 DIY 温度控制器*中学习了使用树莓派设置 TFT LCD 屏幕的过程。如果您尚未完成这些练习，请参阅那些教程。假设您已配置了 Mosquitto 代理和所需的 Python 库，您可以继续到下一节，该节包括实际的 Python 编程。

## 控制中心的 Python 代码

在您开始在 Python 代码中接口这些库之前，首先使用以下简单命令从命令行启动您的 Mosquitto 代理：

```py
$ mosquitto

```

确保每次启动或重启 Mosquitto 代理时都重新启动您的监控站。这个操作将确保您的监控站连接到 Mosquitto 代理，因为在我们 Arduino 代码中，建立连接的过程只会在设置过程的开始执行一次。

当前项目的 Python 代码位于本章代码文件夹中，文件名为 `controlCenter.py`。使用您的 Python IDE 打开此文件，在执行之前修改适当的参数值。这些参数包括 Mosquitto 代理的 IP 地址以及 Xively 虚拟设备的 feed ID 和 API 密钥。您应该已经从上一章中获得了 Xively 虚拟设备的 feed ID 和 API 密钥：

```py
cli.connect("10.0.0.18", 1883, 15)
FEED_ID = "<feed-id>"
API_KEY = "<api-key"
```

如果你正在使用 Mosquitto 代理的本地实例，你可以将 IP 地址替换为 `127.0.0.1`。否则，将 `10.0.0.18` 地址替换为托管 Mosquitto 代理的计算机的适当 IP 地址。现在让我们尝试理解代码。

### 注意

有时在 Mac OS X 上，由于一个未知的错误，你无法并行运行 `Tkinter` 窗口和 Python 线程。你应该能够在 Windows 和 Linux 环境中成功执行程序。这个程序已经在 Raspberry Pi 上进行了测试，这意味着在部署控制中心时，你不会遇到相同的错误。

### 使用 Tkinter 创建 GUI

在之前的练习中，我们总是使用单个 Python 线程来运行程序。这种做法不会帮助我们并行执行多个任务，例如从监控站获取传感器观察结果，并同时更新 GUI 以显示该信息。作为解决方案，我们在本次练习中引入了多线程。由于我们需要两个独立的循环，一个用于 `Tkinter`，另一个用于 `paho-mqtt`，我们将它们独立地在不同的线程中运行。主线程将运行与 Mosquitto 和云服务相关的函数，而第二个线程将处理 `Tkinter` GUI。在下面的代码片段中，你可以看到我们使用 `threading.thread` 参数初始化了 `controlCenterWindow()` 类。因此，当我们主程序中执行 `window = controlCenterWindow()` 时，它将为这个类创建另一个线程。基本上，这个类在填充标签和其他 GUI 组件的同时创建 GUI 窗口。当新的传感器观察结果到达时，标签需要更新，它们被声明为类变量，并且可以从类实例中访问。正如你在下面的代码片段中所看到的，我们已将温度、湿度、光和运动的标签声明为类变量：

```py
class controlCenterWindow(threading.Thread):
    def __init__(self):
        # Tkinter canvas
        threading.Thread.__init__(self)
        self.start()
    def callback(self):
        self.top.quit()
    def run(self):
        self.top = Tkinter.Tk()
        self.top.protocol("WM_DELETE_WINDOW", self.callback)
        self.top.title("Control Center")
        self.statusValue = Tkinter.StringVar()
        self.statusValue.set("Normal")
        self.tempValue = Tkinter.StringVar()
        self.tempValue.set('-')
        self.humdValue = Tkinter.StringVar()
        self.humdValue.set('-')
        self.lightValue = Tkinter.StringVar()
        self.lightValue.set('-')
        self.motionValue = Tkinter.StringVar()
        self.motionValue.set('No')

        # Begin code subsection 
        # Declares Tkinter components
        # Included in the code sample of the chapter
        # End code subsection

        self.top.mainloop()
```

之前的代码片段中没有包含我们声明 `Tkinter` 组件的部分，因为它与我们中期项目中编写的代码类似。如果你对 Tkinter 相关问题有疑问，请参阅第六章，*存储和绘制 Arduino 数据*，以及第七章，*中期项目 – 一款便携式 DIY 温度控制器*。

### 与 Mosquitto 代理通信

在控制中心级别，我们订阅了来自监测站发布的主题，即`MonitoringStation/temperature`、`MonitoringStation/humidity`等。如果你对 Arduino 代码进行了任何修改以更改 MQTT 主题，你需要在此部分反映这些更改。如果监测站发布的话题与控制中心代码中的话题不匹配，你将不会收到任何更新。正如你在 Python 代码中所看到的，我们将`on_message`和`on_publish`方法与非常重要的函数关联。每当有消息从订阅者那里到达时，客户端将调用与`on_message`方法关联的函数。然而，每当 Python 代码发布消息时，`onPublish()`函数将被调用：

```py
cli = mq.Client('ControlCenter')
cli.on_message = onMessage
cli.on_publish = onPublish

cli.connect("10.0.0.18", 1883, 15)

cli.subscribe("MonitoringStation/temperature", 0)
cli.subscribe("MonitoringStation/humidity", 0)
cli.subscribe("MonitoringStation/motion", 0)
cli.subscribe("MonitoringStation/light", 0)
cli.subscribe("MonitoringStation/buzzer", 0)
cli.subscribe("MonitoringStation/led", 0)
```

### 计算系统的状态和情况感知

控制中心被分配了计算整个系统状态的任务。控制中心使用当前温度和湿度的值来计算系统的状态，状态可以是`Alert`（警报）、`Caution`（注意）或`Normal`（正常）。为了计算状态，每当控制中心从监测站接收到温度或湿度的更新时，它都会执行`calculateStatus()`函数。根据当前的情况感知逻辑，如果测量的温度高于 45 摄氏度或低于 5 摄氏度，我们将系统的状态称为`Alert`。同样，你可以从以下代码片段中识别出温度和湿度值的范围，以确定`Caution`和`Normal`状态：

```py
def calculateStatus():
    if (tempG > 45):
        if (humdG > 80):
            status = "High Temperature, High Humidity"
        elif (humdG < 20):
            status = "High Temperature, Low Humidity"
        else:
            status = "High Temperature"
        setAlert(status)

    elif (tempG < 5):
        if (humdG > 80):
            status = "Low Temperature, High Humidity"
        elif (humdG < 20):
            status = "Low Temperature, Low Humidity"
        else:
            status = "Low Temperature"
        setAlert(status)
    else:
        if (humdG > 80):
            status = "High Humidity"
            setCaution(status)
        elif (humdG < 20):
            status = "Low Humidity"
            setCaution(status)
        else:
            status = "Normal"
            setNormal(status)
```

### 与 Xively 通信

控制中心在从订阅的话题接收到消息时也需要与 Xively 通信。我们已经熟悉了在 Xively 上设置虚拟设备和数据流的过程。打开你的 Xively 账户，创建一个名为`ControlCenter`的虚拟设备。记下该设备的 feed ID 和 API 密钥，并在当前代码中替换它们。一旦你有了这些值，在这个虚拟设备中创建`Temperature`、`Humidity`、`Light`、`Motion`、`Buzzer`和`Status`通道。

观察 Python 代码，你可以看到我们为每个主题声明了单独的数据流，并将它们与适当的 Xively 通道关联。以下代码片段显示了仅针对温度观测的数据流，但代码还包含了对所有其他传感器观测的类似配置：

```py
try:
  datastreamTemp = feed.datastreams.get("Temperature")
except HTTPError as e:
  print "HTTPError({0}): {1}".format(e.errno, e.strerror)
  datastreamTemp = feed.datastreams.create("Temperature", tags="C")
  print "Creating new channel 'Temperature'"
```

一旦控制中心从监测站接收到消息，它将使用最新的值更新数据流，并将这些更改推送到 Xively。同时，我们还将使用`onMessage()`函数更新`Tkinter` GUI 中的适当标签。我们将使用相同的代码片段为所有订阅的通道：

```py
if msg.topic == "MonitoringStation/temperature":
  tempG = float(msg.payload)
  window.tempValue.set(tempG)
  datastreamTemp.current_value = tempG
  try:
    datastreamTemp.update()
  except HTTPError as e:
    print "HTTPError({0}): {1}".format(e.errno, e.strerror)
```

控制中心还实现了跨系统设置系统状态的功能，一旦使用 `calculateStatus()` 函数计算后。有三个不同的函数通过一种类似于我们在前一个代码片段中描述的方法来执行此任务。这些函数包括 `setAlert()`、`setCaution()` 和 `setNormal()`，它们分别与 `Alert`、`Caution` 和 `Normal` 相关联。在更新系统状态时，这些函数还会通过将 LED 和蜂鸣器值发布到 Mosquitto 代理来执行蜂鸣器和 LED 动作：

```py
def setAlert(status):
    window.statusValue.set(status)
    datastreamStatus.current_value = "Alert"
    try:
        datastreamStatus.update()
    except HTTPError as e:
        print "HTTPError({0}): {1}".format(e.errno, e.strerror)
    cli.publish("MonitoringStation/led", 'red')
    cli.publish("MonitoringStation/buzzer", 'ON')
```

### 检查和更新蜂鸣器的状态

在控制中心，如果系统状态被确定为 `Alert`，我们将蜂鸣器的状态设置为 `ON`。如果你回顾 UX 流程，你会注意到我们还想包括一个让用户手动关闭蜂鸣器的功能。`checkBuzzerFromXively()` 函数跟踪来自 Xively 的蜂鸣器状态，如果用户使用 Web 应用程序手动关闭蜂鸣器，这个函数就会触发蜂鸣器。

为了从 GUI 和情况感知线程独立地继续此过程，我们需要为这个函数创建另一个线程。这个线程上的定时器将自动每 30 秒执行一次函数：

```py
def checkBuzzerFromXively():
  try:
    datastreamBuzzer = feed.datastreams.get("Buzzer")
    buzzerValue = datastreamBuzzer.current_value
    buzzerValue = str(buzzerValue)
    cli.publish("MonitoringStation/buzzer", buzzerValue)
  except HTTPError as e:
    print "HTTPError({0}): {1}".format(e.errno, e.strerror)
    print "Requested channel doesn't exist"
  threading.Timer(30, checkBuzzerFromXively).start()
```

通过在单独的线程上每 30 秒运行此函数，控制中心将检查 Xively 通道的状态，如果状态设置为 `OFF`，则停止蜂鸣器。我们将在下一节中解释用户如何更新蜂鸣器的 Xively 通道。

## 使用监控站测试控制中心

假设你的 Mosquitto 代理正在运行，使用更改后的参数执行 `controlCenter.py` 代码。然后，启动监控站。过一会儿，你将在终端上看到控制中心已经开始从监控站上初始化的发布者那里接收消息。控制中心发布者的更新间隔取决于监控站配置的发布间隔。

### 注意

Arduino 代码在开机后只执行一次连接到 Mosquitto 代理的过程。如果你在那之后启动 Mosquitto 代理，它将无法与代理通信。因此，你需要确保在启动监控站之前启动 Mosquitto 代理。

如果需要因任何原因重新启动 Mosquitto 代理，首先移除并重新启动监控站。

![使用监控站测试控制中心](img/5938OS_10_10.jpg)

在程序执行时，你将能够看到一个小的 GUI 窗口，如下面的截图所示。此窗口显示传感器的温度、湿度、环境光和运动值。除了这些值，GUI 还显示了系统的状态，在这个截图中是**正常**。你也可以观察到，每次控制中心从监控站获取更新时，系统的状态和传感器观察结果都会实时改变：

![使用监控站测试控制中心](img/5938OS_10_11.jpg)

如果这个设置在你的计算机上运行正确，让我们继续在树莓派上部署控制中心。

## 在树莓派上设置控制中心

安装 Raspbian 操作系统的过程在第七章《中期项目 – 一个便携式 DIY 恒温器》中解释。你可以使用中期项目中使用的相同模块，或者设置一个新的模块。一旦安装了 Raspbian 并配置了 TFT 屏幕，通过 USB 端口连接 Wi-Fi 适配器。在这个阶段，我们假设你的树莓派已经连接到显示器、键盘和鼠标以执行基本更改。虽然我们不推荐这样做，但如果你对 TFT 屏幕操作感到舒适，你也可以用它来进行以下操作：

1.  启动你的树莓派并登录。在命令提示符下，执行以下命令以进入图形桌面模式：

    ```py
    $ startx

    ```

1.  一旦你的图形桌面启动，你将能够看到**WiFi 配置**工具的图标。双击此图标并打开**WiFi 配置**工具。扫描无线网络并连接到有监控站的 Wi-Fi 网络。当被要求时，在名为**PSK**的表单窗口中输入你的网络密码，并连接到你的网络。

1.  现在，你的树莓派已经通过它连接到本地家庭网络和互联网。是时候更新现有包并安装所需的包了。要更新树莓派的现有系统，请在终端中执行以下命令：

    ```py
    $ sudo apt-get update
    $ sudo apt-get upgrade

    ```

1.  一旦你的系统更新到最新版本，就是时候在你的树莓派上安装 Mosquitto 代理了。Raspbian 操作系统默认仓库中有 Mosquitto，但没有我们需要的当前版本。要安装 Mosquitto 的最新版本，请在终端中执行以下命令：

    ```py
    $ curl -O http://repo.mosquitto.org/debian/mosquitto-repo.gpg.key
    $ sudo apt-key add mosquitto-repo.gpg.key
    $ rm mosquitto-repo.gpg.key
    $ cd /etc/apt/sources.list.d/
    $ sudo curl -O http://repo.mosquitto.org/debian/mosquitto-repo.list
    $ sudo apt-get update
    $ sudo apt-get install mosquitto, mosquitto-clients

    ```

1.  要安装其他 Python 依赖项，我们首先使用`apt-get`安装 Setuptools 包：

    ```py
    $ sudo apt-get install python-setuptools

    ```

1.  使用 Setuptools，我们现在可以安装所有必需的 Python 库，如`paho_mqtt`、`xively-python`和`web.py`：

    ```py
    $ sudo easy_install pip
    $ sudo pip install xively-python web.py paho_mqtt

    ```

现在我们已经安装了所有必要的软件工具，可以在树莓派上运行我们的控制中心，是时候配置树莓派，使其能够为远程家庭监控系统等关键系统提供不间断的运行：

1.  在当前配置的树莓派中，树莓派的屏幕会在一段时间后进入休眠状态，此时 Wi-Fi 连接也会被终止。为了避免这个问题并强制屏幕保持活跃，你需要执行以下更改。使用以下命令打开`lightdm.conf`文件：

    ```py
    $ sudo nano /etc/lightdm/lightdm.conf

    ```

1.  在文件中，导航到`SetDefaults`部分并编辑以下行：

    ```py
    xserver-command-X –s 0 dpms
    ```

1.  现在您的树莓派已经设置好了，是时候将程序文件从您的电脑复制到树莓派上了。您可以使用 SCP、PuTTY 或只是一个 USB 驱动器来将必要的文件传输到树莓派。

如果你按照指定的方式安装和配置了所有内容，你的程序应该会无错误运行。你可以使用以下命令在后台持续运行 Python 程序：

```py
$ nohup python controlCenter.py &

```

我们在树莓派上最后要设置的是 TFT LCD 屏幕。TFT LCD 屏幕的安装和配置过程在第七章《中期项目 – 一个便携式 DIY 恒温器》中有描述。请按照给定的顺序设置屏幕。现在，控制中心模块、树莓派和 TFT 屏幕可以部署在你家的任何地方。

# 第三阶段 – 使用 Xively、Python 和 Amazon 云服务的网络应用

整个系统的云服务模块使您可以通过互联网远程访问您的监控站。该单元通过作为控制中心扩展版本的 Web 应用与用户交互。使用这个 Web 应用，用户可以在远程控制关闭蜂鸣器的同时，观察来自监控站的传感器信息和由控制中心计算的系统状态。那么，云服务的架构是什么样的呢？

## 云服务架构

下面的图中显示了云服务模块及其相关组件的架构。在云服务架构中，我们使用 Xively 作为网络应用和控制中心之间的中间数据中继。控制中心将来自监控站的观测数据推送到 Xively 通道。Xively 存储并转发数据到托管在 Amazon AWS 上的网络应用。Amazon AWS 上的服务器实例用于使网络应用通过互联网可访问。服务器实例运行 Ubuntu 操作系统和用 Python 的`web.py`库开发的网络应用。

![云服务架构](img/5938OS_10_12.jpg)

在之前的阶段，我们已经介绍了设置 Xively 和通道以容纳传感器数据的过程。在控制中心代码中，我们也解释了如何将更新的观测数据推送到适当的 Xively 通道。因此，在这个阶段，我们实际上没有关于 Xively 平台的内容要介绍，我们可以继续到 Web 应用程序。

## 部署在 Amazon AWS 上的 Python Web 应用程序

在上一章中，我们设置了一个 Amazon AWS 云实例来托管 Web 应用程序。你也可以使用相同的实例来托管远程家庭监控系统中的 Web 应用程序。然而，请确保你在服务器上安装了`web.py`库。

1.  在你的电脑上，打开`Web_Application`文件夹，然后在你的编辑器中打开`RemoteMonitoringApplication.py`文件。

1.  在代码中，你可以看到我们只是扩展了我们在第九章中创建的 Web 应用程序程序，即*Arduino 和物联网*。我们使用基于`web.py`的模板以及`GET()`和`POST()`函数来启用 Web 应用程序。

1.  在应用程序中，我们从每个 Xively 通道获取信息，并通过一个单独的函数进行处理。例如，`fetchTempXively()`函数从 Xively 获取温度信息。每次执行`POST()`函数时，`fetchTempXively()`函数都会从 Xively 获取最新的温度读数。这也意味着 Web 应用程序不会自动填充和刷新最新信息，而是等待`POST()`执行适当的函数：

    ```py
    def fetchTempXively():
      try:
        datastreamTemp = feed.datastreams.get("Temperature")
      except HTTPError as e:
        print "HTTPError({0}): {1}".format(e.errno, e.strerror)
        print "Requested channel doesn't exist"
      return datastreamTemp.current_value
    ```

1.  Web 应用程序还提供了从用户界面控制蜂鸣器的功能。以下代码片段添加了**Buzzer Off**按钮和其他`Form`组件。在此按钮按下后提交表单，Web 应用程序将执行`setBuzzer()`函数：

    ```py
    inputData = web.input()
    if inputData.btn == "buzzerOff":
        setBuzzer("OFF")
    ```

1.  `setBuzzer()`函数访问 Xively 通道`Buzzer`，如果按下**Buzzer Off**按钮，则发送关闭值。当前的 Web 应用程序不包括**Buzzer On**按钮，但你可以通过重用我们为**Buzzer Off**按钮开发的代码轻松实现此功能。此函数为其他控制点提供了参考代码，你可以稍作修改后重用：

    ```py
    def setBuzzer(statusTemp):
      try:
        datastream = feed.datastreams.get("Buzzer")
      except HTTPError as e:
        print "HTTPError({0}): {1}".format(e.errno, e.strerror)
        datastream = feed.datastreams.create("Buzzer", 
                                             tags="buzzer")
        print "Creating new Channel 'Buzzer"
      datastream.current_value = statusTemp
      try:
        datastream.update()
      except HTTPError as e:
        print "HTTPError({0}): {1}".format(e.errno, e.strerror)
    ```

1.  在代码中，你还需要修改 Xively 的 feed ID 和 API 密钥，并将它们替换为你从虚拟设备获得的值。完成此修改后，运行以下命令。如果一切按计划进行，你将能够在你的网络浏览器中打开 Web 应用程序。

    ```py
    $ python RemoteMonitoringApplication.py

    ```

如果你正在电脑上运行 Python 代码，你可以打开`http://127.0.0.1:8080`来访问应用程序。如果你在云服务器上运行应用程序，你需要输入你的服务器 IP 地址或域名来访问 Web 应用程序，`http://<AWS-IP-address>:8080`。如果 Web 应用程序在云上运行，它可以通过互联网从任何地方访问，这是原始项目要求之一。通过这一最后一步，你已经成功完成了基于 Arduino 和 Python 的远程家庭监控系统开发。

## 测试 Web 应用程序

当你在浏览器中打开 Web 应用程序时，你将能够看到以下截图所示的类似输出。正如你所见，Web 应用程序显示了温度、湿度、光照和运动值。**刷新**按钮从 Xively 再次获取传感器数据并重新加载应用程序。**蜂鸣器关闭**按钮将 Xively 的`蜂鸣器`通道值设置为`关闭`，然后由控制中心接收到，随后在监测站关闭蜂鸣器：

![测试 Web 应用程序](img/5938OS_10_13.jpg)

# 测试和故障排除

由于涉及到的组件数量和与之相关的复杂编程，整个项目是一个复杂的系统，需要进行测试和调试。在你开始故障排除之前，确保你已经按照前面章节中描述的步骤正确操作。以下是在项目执行过程中可能出现的几个问题的解决方案：

+   故障排除单个传感器的性能：

    +   如果你的传感器测量值远远偏离预期值，你首先想要评估的是传感器引脚与 Arduino 板的连接。确保你已经正确连接了数字、模拟和 PWM 引脚。

    +   检查你的以太网盾板是否正确连接到 Arduino Uno。

    +   评估每个组件的 5V 电源和地线的连接。

+   避免 Xively 的更新限制

    +   Xively 对你在有限时间内可以执行的最大交易数量有限制。在运行控制中心代码时，如果你遇到超出限制的错误，请在你的访问限制解除之前等待 5 分钟。

    +   在控制中心级别增加连续 Xively 更新的延迟：

        ```py
        threading.Timer(120, checkBuzzerFromXively).start()
        ```

    +   减少监测站发布的消息频率：

        ```py
        timer.setInterval(600000, publishData);
        ```

    +   你还可以通过将数据格式化为 JSON 或 XML 来组合各种 Xively 通道。

+   与 Arduino 的最大电流消耗限制一起工作：

    +   Arduino 的+5V 电源引脚和数字引脚分别可以提供最大 200 mA 和 40 mA 的电流。当直接从 Arduino 板运行传感器时，确保你不超过这些限制。

    +   确保所有传感器的总电流需求小于 200 mA。否则，组件将无法获得足够的电力来运行，这将导致传感器信息错误。

    +   您可以为需要大量电流的组件提供外部电源，并通过 Arduino 本身控制此电源机制。您需要一个作为开关工作的晶体管，然后可以使用 Arduino 的数字引脚来控制它。[`learn.adafruit.com/adafruit-arduino-lesson-13-dc-motors/transistors`](https://learn.adafruit.com/adafruit-arduino-lesson-13-dc-motors/transistors)教程展示了类似用于直流电机的示例。

+   解决网络问题：

    +   在某些情况下，由于网络问题，您的监控站可能无法与控制中心通信。

    +   通过为 Arduino 和 Raspberry Pi 都使用手动 IP 地址，可以解决这个问题。在我们的项目中，我们为 Arduino 使用手动 IP 地址，但 Raspberry Pi 是通过 Wi-Fi 网络连接的。在大多数情况下，当您使用家庭 Wi-Fi 网络时，Wi-Fi 路由器被设置为在设备每次重新连接到路由器时提供动态 IP 地址。

    +   您可以通过将 Wi-Fi 路由器配置为为 Raspberry Pi 提供固定 IP 地址来解决这个问题。由于每个场景中的 Wi-Fi 路由器类型和型号都不同，您将需要使用其用户手册或在线帮助论坛来设置它。

+   处理蜂鸣器相关的问题：

    +   有时蜂鸣器的声音可能太大或太小，这取决于您使用的传感器。您可以使用 PWM 来配置蜂鸣器的强度。在我们的项目中，我们使用 Arduino 数字引脚 9 连接蜂鸣器。此引脚也支持 PWM。在您的 Arduino 代码中，修改行以反映 PWM 引脚的变化。将`digitalWrite(BUZZER, HIGH);`行替换为`analogWrite(BUZZER, 127);`。

    +   此程序将蜂鸣器的强度从原始水平减半。您还可以将 PWM 值从 0 更改为 255，并将蜂鸣器声音的强度从最低设置为最高。

+   控制中心 GUI 校准：

    +   根据您使用的 TFT 液晶屏幕的大小，您需要调整`Tkinter`的主窗口大小。

    +   首先，在您的 Raspberry Pi 上运行当前代码，如果您看到 GUI 窗口与屏幕不匹配，请在初始化主窗口后添加以下代码行：

        ```py
        top.minsize(320,200)
        ```

    +   此代码将修复 2.8 英寸 TFT 液晶屏幕尺寸的问题。在之前的代码片段中，`320`和`200`分别代表宽度和长度的像素大小。对于其他屏幕尺寸，相应地更改像素大小。

+   测试 LED：

    +   在当前的代码配置中，只有当系统切换到`Alert`或`Caution`时，LED 才会打开。这意味着除非这些情况发生，否则您无法测试 LED。要检查它们是否正常工作，请在控制中心执行以下命令：

        ```py
        $ mosquitto_pub –t "MonitoringStation/led" –m "red"

        ```

    +   这个命令将使 LED 变红。要关闭 LED，只需在之前的代码中将`red`替换为`off`。

    +   如果没有任何灯光亮起，你应该检查 LED 的连接线。此外，检查网络相关的问题，因为 Mosquitto 本身可能不工作。

    +   如果你看到除了红色以外的任何颜色，这意味着你没有正确连接 LED，你需要交换你的 LED 引脚配置。如果你使用的是非超级流光 RGB 的 LED，你应该检查数据表中的引脚布局并重新组织连接。

# 扩展你的远程家庭监控系统

要从 DIY 项目原型成功创建商业产品，你需要在基本功能之上添加额外的功能层。这些功能实际上在使用系统时为用户提供了便利。另一个可区分的特征是系统的可触摸性，这使得大规模生产和支持成为可能。尽管你可以实现很多功能，但我们建议以下主要改进来提升当前项目的水平。

## 利用多个监测站

在这个项目中，我们开发了一个作为原型的监测站，它具有一系列由远程家庭监控系统展示的功能。远程监控系统可以拥有多个监测站来覆盖各种地理区域，例如房屋内的不同房间，或不同的办公室隔间。基本上，大量的监测站可以覆盖更广泛的区域，并提供对你试图监控区域的效率监控。如果你想通过一系列监测站扩展当前项目，你将需要以下一些修改：

+   每个监测站可以有自己的控制中心，或者根据应用需求为所有监测站集中一个控制中心。

+   你将不得不更新控制中心的 Python 代码以适应这些变化。这些变化的例子包括修改 MQTT 的主题标题、在这些监测站之间协调、更新 Xively 数据模型以进行更新等。

+   免费 Xively 账户可能无法处理来自监测站的大量数据。在这种情况下，你可以优化更新速率和/或有效载荷大小，或者升级你的 Xively 账户以满足要求。你也可以求助于其他免费服务，如 ThingSpeak、Dweet.io 和 Carriots，但你将需要对现有的代码结构进行重大修改。

+   你还可以更新网络应用程序，为你提供监测站的选项菜单或一次性显示所有监测站。你还需要更改代码以生成修改后的数据模型。

## 扩展感官能力

在传感器方面，我们只接口温度、湿度、环境光和运动传感器。然而，执行功能仅限于蜂鸣器和 LED。您可以通过以下更改来提高项目的感官能力。

+   在实际场景中，一个远程家庭监控系统应该能够与家庭中的其他现有传感器接口，例如安全系统、监控摄像头、冰箱传感器、门传感器和车库传感器。

+   您还可以将此项目与其他家用电器如空调、加热器和安全报警器接口，这可以帮助您控制您已经监控的环境。作为试验，这些组件可以使用一组继电器和开关进行接口。

+   您可以使用更强大、更高效、更精确的传感器升级监控站上的当前传感器。然而，升级后的传感器监控站可能需要更强大的 Arduino 版本，具有更多的 I/O 引脚和计算能力。

+   您还可以在监控站使用除本项目使用的传感器以外的其他传感器。市面上有大量异构的、Arduino 支持的 DIY 传感器，您可以直接购买。这些传感器的例子包括酒精气体传感器（MQ-3）、液化石油气传感器（MQ-6）、一氧化碳传感器（MQ-7）、甲烷气体传感器（MQ-4）等等。这些传感器可以像我们之前连接的其他传感器一样简单地与 Arduino 接口。

+   为了适应这些变化，您将需要更改控制中心逻辑和算法。如果您正在接口第三方组件，您可能还需要重新审视系统架构并对其进行调整。

+   类似地，您还必须为额外的传感器频繁更新 Xively，这使得免费版本不足。为了解决这个问题，您可以支付 Xively 账户的商业版本或使用类似于以下代码片段显示的 JSON 文件格式进行有限数量的请求：

    ```py
    {
        "version": "1.0.0",
        "datastreams": [
            {
                "id": "example",
                "current_value": "333"
            },
            {
                "id": "key",
                "current_value": "value"
            },
            {
                "id": "datastream",
                "current_value": "1337"
            }
        ]
    }
    ```

## 改进 UX

当我们为这个项目设计用户体验时，我们的目标是展示 UX 设计在开发软件流程中的有用性。在当前的 UX 设计中，控制中心和网络应用对用户的控制和功能有限。以下是一些您需要实施以改进项目 UX 的更改：

+   为各种描述添加工具提示和适当的命名约定。实现适当的布局以区分不同的信息类别。

+   在控制中心 GUI 上添加蜂鸣器和 LED 控制的按钮。

+   在网络应用中，使用基于 JavaScript 和 Ajax 的界面来自动刷新传感器值的变化。

+   提供一个用户界面机制，以便用户可以在控制中心和网络应用程序中更改更新间隔。一旦做出这些更改，通过每个程序传播它们，以便监控站可以开始以新的间隔发布消息。

## 扩展基于云的功能

在当前的设置中，我们使用两个阶段来提供基于云的功能并启用远程监控。我们使用 Xively 作为数据中继，并使用 Amazon AWS 来托管网络应用程序。如果您正在开发商业级产品并希望简化架构的复杂性，您可以实施以下更改：

+   您可以使用如 ThingSpeak 等开源工具在您的云实例上开发自己的数据中继。然后，您的控制中心将直接与您的服务器通信，消除对第三方物联网服务的依赖。

+   如果 Xively 是您的平台，您还可以使用 Xively 提供的附加功能，例如智能手机上的图表。一旦您的手机与 Xively 配对，您可以直接访问此功能。

+   或者，您可以使用其他云服务，如 Microsoft Azure 和 Google App Engine，而不是 Amazon AWS。根据您对云计算的熟悉程度，您也可以设置自己的云服务器。尽管拥有自己的云将使您完全控制服务器，但与自托管服务器相比，第三方服务如 Amazon 可能更具成本效益且维护成本更低。

+   如果您计划开发基于当前架构的大型系统，您可以增加现有云实例的计算能力。您还可以实现分布式服务器系统，以适应大量远程监控系统，这些系统可以由更多的用户访问。

## 提高态势感知的智能程度

在这个项目中，我们使用了四种不同的传感器来监控物理环境——每个传感器都使用两种类型的执行器来获取用户输入以进行通知。尽管我们使用了大量的信息来源，但我们的态势感知算法仅限于识别超出范围的温度和湿度值。您可以通过实现一些扩展功能来使您的系统更加灵活和有用：

+   实现白天和夜晚场景的不同逻辑，这可以帮助您避免夜间不必要的误报。

+   使用运动传感器实现入侵检测算法，以便您不在家时使用。

+   利用环境光传感器值与运动传感器的组合来识别能源浪费。例如，当夜间运动显著减少时记录到的光线更多，这可能表明您在夜间忘记关闭灯光。

## 为硬件组件创建一个封装

就像基于软件的功能一样，如果你开发的是商业级产品，硬件组件也需要进行重大改造。如今，3D 打印机已经变得可行，设计和打印塑料 3D 组件变得非常容易。你也可以使用专业的 3D 打印服务，如 Shapeways ([`www.shapeways.com`](http://www.shapeways.com))、Sculpteo ([`www.sculpteo.com`](http://www.sculpteo.com)) 或 makexyz ([`www.makexyz.com`](http://www.makexyz.com)) 来打印你的外壳。你甚至可以使用激光切割机或其他模型制作手段来创建硬件外壳。以下是一些你可以实施的硬件改进：

+   组装在原型板上的传感器和执行器可以被组织在 PCB 板上，并永久固定以实现稳定和坚固的操作。

+   监控站的外壳可以使设备便携，并易于在任何环境中部署。在设计这个外壳时，你还应考虑运动传感器和环境光传感器的适当位置，以及一个按钮，以便用户可以访问它们。

+   构成控制中心硬件的 Raspberry Pi 和 TFT 液晶显示屏也可以封装在一个可安装的包装中。

+   将触摸屏功能添加到 TFT 液晶显示屏上可以实现对系统的额外控制，扩展用户体验的使用案例。

# 摘要

在本章中，我们开发了一个远程家庭监控系统的工作原型，并同时学习了硬件产品开发的过程。在项目中，我们使用了本书中大部分的硬件组件和软件工具。我们首先设计系统架构，以便能够协调这些工具的利用。随后，我们进入了实际的开发阶段，这包括设计硬件单元和开发运行这些单元的程序。最后，我们提供了一份改进清单，以便将这个原型转变为真正的商业产品。欢迎你使用这种方法来开发你未来的项目和产品，因为你现在已经有了与这个项目合作的经验。

在最后一章，我们将利用相同的项目开发方法来创建一个有趣的项目，该项目利用来自社交网络网站的消息来控制你的硬件。
