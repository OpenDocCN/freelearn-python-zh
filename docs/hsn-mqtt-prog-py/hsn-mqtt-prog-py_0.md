# 前言

MQTT 是首选的物联网发布-订阅轻量级消息传递协议。Python 绝对是最流行的编程语言之一。它是开源的，多平台的，您可以使用它开发任何类型的应用程序。如果您开发物联网、Web 应用程序、移动应用程序或这些解决方案的组合，您必须学习 MQTT 及其轻量级消息传递系统的工作原理。Python 和 MQTT 的结合使得开发能够与传感器、不同设备和其他应用程序进行通信的强大应用程序成为可能。当然，在使用该协议时，考虑安全性是非常重要的。

大多数情况下，当您使用现代 Python 3.6 编写的复杂物联网解决方案时，您将使用可能使用不同操作系统的不同物联网板。MQTT 有自己的特定词汇和不同的工作模式。学习 MQTT 是具有挑战性的，因为它包含太多需要真实示例才能易于理解的抽象概念。

本书将使您深入了解最新版本的 MQTT 协议：3.1.1。您将学习如何使用最新的 Mosquitto MQTT 服务器、命令行工具和 GUI 工具，以便了解 MQTT 的一切工作原理以及该协议为您的项目提供的可能性。您将学习安全最佳实践，并将其用于 Mosquitto MQTT 服务器。

然后，您将使用 Python 3.6 进行许多真实示例。您将通过与 Eclipse Paho MQTT 客户端库交换 MQTT 消息来控制车辆、处理命令、与执行器交互和监视冲浪比赛。您还将使用基于云的实时 MQTT 提供程序进行工作。

您将能够在各种现代物联网板上运行示例，例如 Raspberry Pi 3 Model B+、Qualcomm DragonBoard 410c、BeagleBone Black、MinnowBoard Turbot Quad-Core、LattePanda 2G 和 UP Core 4GB。但是，任何支持 Python 3.6 的其他板都可以运行这些示例。

# 本书适合对象

本书面向希望开发能够与其他应用程序和设备交互的 Python 开发人员，例如物联网板、传感器和执行器。

# 本书涵盖内容

第一章，*安装 MQTT 3.1.1 Mosquitto 服务器*，开始我们的旅程，使用首选的物联网发布-订阅轻量级消息传递协议在不同的物联网解决方案中，结合移动应用程序和 Web 应用程序。我们将学习 MQTT 及其轻量级消息传递系统的工作原理。我们将了解 MQTT 的谜题：客户端、服务器（以前称为代理）和连接。我们将学习在 Linux、macOS 和 Windows 上安装 MQTT 3.1.1 Mosquitto 服务器的程序。我们将学习在云上（Azure、AWS 和其他云提供商）运行 Mosquitto 服务器的特殊注意事项。

第二章，*使用命令行和 GUI 工具学习 MQTT 的工作原理*，教我们如何使用命令行和 GUI 工具详细了解 MQTT 的工作原理。我们将学习 MQTT 的基础知识，MQTT 的特定词汇和其工作模式。我们将使用不同的实用工具和图表来理解与 MQTT 相关的最重要的概念。我们将在编写 Python 代码与 MQTT 协议一起工作之前，了解一切必须知道的内容。我们将使用不同的服务质量级别，并分析和比较它们的开销。

第三章，*保护 MQTT 3.1.1 Mosquitto 服务器*，着重介绍如何保护 MQTT 3.1.1 Mosquitto 服务器。我们将进行所有必要的配置，以使用数字证书加密 MQTT 客户端和服务器之间发送的所有数据。我们将使用 TLS，并学习如何为每个 MQTT 客户端使用客户端证书。我们还将学习如何强制所需的 TLS 协议版本。

第四章，*使用 Python 和 MQTT 消息编写控制车辆的代码*，侧重于使用加密连接（TLS 1.2）通过 MQTT 消息控制车辆的 Python 3.x 代码。我们将编写能够在不同流行的 IoT 平台上运行的代码，例如树莓派 3 板。我们将了解如何利用我们对 MQTT 协议的了解来构建基于需求的解决方案。我们将学习如何使用最新版本的 Eclipse Paho MQTT Python 客户端库。

第五章，*测试和改进我们的 Python 车辆控制解决方案*，概述了如何使用 MQTT 消息和 Python 代码来处理我们的车辆控制解决方案。我们将学习如何使用 Python 代码处理接收到的 MQTT 消息中的命令。我们将编写 Python 代码来组成和发送带有命令的 MQTT 消息。我们将使用阻塞和线程化的网络循环，并理解它们之间的区别。最后，我们将利用遗嘱功能。

第六章，*使用基于云的实时 MQTT 提供程序和 Python 监控冲浪比赛*，介绍了如何编写 Python 代码，使用 PubNub 基于云的实时 MQTT 提供程序与 Mosquitto MQTT 服务器结合，监控冲浪比赛。我们将通过分析需求从头开始构建一个解决方案，并编写 Python 代码，该代码将在连接到冲浪板上的多个传感器的防水 IoT 板上运行。我们将定义主题和命令，并与基于云的 MQTT 服务器一起使用，结合了前几章中使用的 Mosquitto MQTT 服务器。

附录，*解决方案*，每章的*测试你的知识*部分的正确答案都包含在附录中。

# 为了充分利用本书

您需要对 Python 3.6.x 和 IoT 板有基本的了解。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便直接通过电子邮件接收文件。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packtpub.com](http://www.packtpub.com/support)。

1.  选择支持选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-MQTT-Programming-with-Python`](https://github.com/PacktPublishing/Hands-On-MQTT-Programming-with-Python)。如果代码有更新，将在现有的 GitHub 存储库中更新。

我们还有其他代码包，来自我们丰富的书籍和视频目录，可以在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上找到。去看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图片。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/HandsOnMQTTProgrammingwithPython_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/HandsOnMQTTProgrammingwithPython_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。例如："将下载的`WebStorm-10*.dmg`磁盘映像文件挂载为系统中的另一个磁盘。"

代码块设置如下：

```py
@staticmethod
    def on_subscribe(client, userdata, mid, granted_qos):
        print("I've subscribed with QoS: {}".format(
            granted_qos[0]))
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目会以粗体显示：

```py
 time.sleep(0.5) 
       client.disconnect() 
       client.loop_stop() 
```

任何命令行输入或输出都以以下方式书写：

```py
 sudo apt-add-repository ppa:mosquitto-dev/mosquitto-ppa
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会在文本中以这种方式出现。例如："从管理面板中选择系统信息。"

警告或重要提示会以这种方式出现。技巧和窍门会以这种方式出现。
