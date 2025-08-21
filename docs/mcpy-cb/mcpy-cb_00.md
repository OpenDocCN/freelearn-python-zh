# 前言

MicroPython 是 Python 3 编程语言的精简实现，能够在各种微控制器上运行。它为这些微控制器提供了 Python 编程语言的大部分功能，如函数、类、列表、字典、字符串、读写文件、列表推导和异常处理。

微控制器是通常包括 CPU、内存和输入/输出外围设备的微型计算机。尽管它们的资源相对于 PC 来说更有限，但它们可以制作成更小的尺寸，功耗更低，成本更低。这些优势使得它们可以在以前不可能的广泛应用中使用。

本书将涵盖 MicroPython 语言的许多不同特性，以及许多不同的微控制器板。最初的章节将提供简单易懂的配方，以使这些板与人们和他们的环境互动。主题涵盖从传感器读取温度、光线和运动数据到与按钮、滑动开关和触摸板互动。还将涵盖在这些板上产生音频播放和 LED 动画的主题。一旦打下了这个基础，我们将构建更多涉及的项目，如互动双人游戏、电子乐器和物联网天气机。您将能够将从这些配方中学到的技能直接应用于自己的嵌入式项目。

# 本书适合对象

这本书旨在帮助人们将 Python 语言的强大和易用性应用于微控制器的多功能性。预期读者具有 Python 的基础知识才能理解本书。

# 本书内容

第一章，*Getting Started with MicroPython*，介绍了 Adafruit Circuit Playground Express 微控制器，并教授了在此硬件上使用 MicroPython 的核心技能。

第二章，*Controlling LEDs*，涵盖了控制 NeoPixel LED、灯光颜色以及如何通过控制板上灯光变化的时间来创建动画灯光秀的方法。

第三章，*Creating Sound and Music*，讨论了如何在 Adafruit Circuit Playground Express 上制作声音和音乐的方法。将涵盖诸如使板在特定声音频率下发出蜂鸣声以及使用 WAV 文件格式和板载扬声器播放音乐文件等主题。

第四章，*Interacting with Buttons*，展示了与 Adafruit Circuit Playground Express 上的按钮和触摸板互动的方法。将讨论检测按钮何时被按下或未被按下的基础知识，以及高级主题，如微调电容触摸板的触摸阈值。

第五章，*Reading Sensor Data*，介绍了从各种不同类型的传感器（如温度、光线和运动传感器）读取传感器数据的方法。

第六章，*Button Bash Game*，指导我们创建一个名为*Button Bash*的双人游戏，您可以直接在 Circuit Playground Express 上使用按钮、NeoPixels 和内置扬声器进行游戏。

第七章，*Fruity Tunes*，解释了如何使用 Adafruit Circuit Playground Express 和一些香蕉创建一个乐器。触摸板将用于与香蕉互动，并在每次触摸不同的香蕉时播放不同的音乐声音。

第八章，“让我们动起来”，介绍了 Adafruit CRICKIT 硬件附加组件，它将帮助我们通过 Python 脚本控制电机和舵机；特别是它们的速度、旋转方向和角度将通过这些脚本进行控制。

第九章，“在 micro:bit 上编码”，涵盖了与 micro:bit 平台交互的方法。将讨论如何控制其 LED 网格显示并与板载按钮交互。

第十章，“控制 ESP8266”，介绍了 Adafruit Feather HUZZAH ESP8266 微控制器，并讨论了它与其他微控制器相比的特点和优势。将涵盖连接到 Wi-Fi 网络、使用 WebREPL 和通过 Wi-Fi 传输文件等主题。

第十一章，“与文件系统交互”，讨论了与操作系统（OS）相关的一些主题，如列出文件、删除文件、创建目录和计算磁盘使用量。

第十二章，“网络”，讨论了如何执行许多不同的网络操作，如 DNS 查找、实现 HTTP 客户端和 HTTP 服务器。

第十三章，“与 Adafruit FeatherWing OLED 交互”，介绍了 Adafruit FeatherWing OLED 硬件附加组件，它可以连接到 ESP8266，为互联网连接的微控制器添加显示，以显示文本图形并使用包含的三个硬件按钮与用户交互。

第十四章，“构建 IoT 天气机”，解释了如何创建一个 IoT 设备，该设备将在按下按钮时从 IoT 设备本身检索天气数据并向用户显示。

第十五章，“在 Adafruit HalloWing 上编码”，介绍了 Adafruit HalloWing 微控制器，它内置了一个 128x128 全彩薄膜晶体管（TFT）显示屏，可以在微控制器上显示丰富的图形图像。

# 为了充分利用本书

读者应具有 Python 编程语言的基本知识。读者最好具有基本的导入包和使用 REPL 的理解，以充分利用本书。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com/support)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便文件直接通过电子邮件发送给您。

您可以按照以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)上登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书名并按照屏幕上的说明操作。

下载示例代码文件后，请确保使用以下最新版本解压或提取文件夹：

+   Windows 的 WinRAR/7-Zip

+   Mac 的 Zipeg/iZip/UnRarX

+   Linux 的 7-Zip/PeaZip

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/MicroPython-Cookbook`](https://github.com/PacktPublishing/MicroPython-Cookbook)。我们还有来自丰富书籍和视频目录的其他代码包可供使用，网址为**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。去看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图像。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781838649951_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781838649951_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词，数据库表名，文件夹名，文件名，文件扩展名，路径名，虚拟 URL，用户输入和 Twitter 句柄。例如： "这个食谱需要在计算机上安装 Python 和`pip`。"

一个代码块设置如下：

```py
from adafruit_circuitplayground.express import cpx
import time

cpx.pixels[0] = (255, 0, 0) # set first NeoPixel to the color red
time.sleep(60)
```

当我们希望引起您对代码块的特定部分的注意时，相关的行或项目会以粗体显示：

```py
from adafruit_circuitplayground.express import cpx
import time

RAINBOW = [
 0xFF0000, # red 
 0xFFA500, # orange
```

任何命令行输入或输出都以如下形式书写：

```py
>>> 1+1
2
```

**粗体**：表示一个新术语，一个重要的词，或者您在屏幕上看到的词。例如，菜单或对话框中的单词会以这种形式出现在文本中。例如："单击工具栏上的串行按钮，以打开与设备的 REPL 会话。"

警告或重要说明会出现在这样的形式。提示和技巧会出现在这样的形式。

# 章节

在本书中，您会经常看到几个标题（*准备工作*，*如何做…*，*它是如何工作的…*，*还有更多…*，和*另请参阅*）。

为了清晰地说明如何完成一个食谱，使用以下章节：

# 准备工作

这一部分告诉您在食谱中可以期待什么，并描述如何设置食谱所需的任何软件或初步设置。

# 如何做…

这一部分包含了遵循食谱所需的步骤。

# 它是如何工作的…

这一部分通常包括对前一部分发生的事情的详细解释。

# 还有更多…

这一部分包括了有关食谱的额外信息，以使您对食谱更加了解。

# 另请参阅

这一部分提供了有用的链接，指向食谱的其他有用信息。
