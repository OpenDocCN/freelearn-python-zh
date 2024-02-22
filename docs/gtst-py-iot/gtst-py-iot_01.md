# 第一章：开始使用树莓派 3 电脑

在本章中，我们将涵盖以下主题：

+   连接外围设备到树莓派

+   使用 NOOBS 设置您的树莓派 SD 卡

+   通过 LAN 连接器将您的树莓派连接到互联网

+   在树莓派上使用内置的 Wi-Fi 和蓝牙

+   手动配置您的网络

+   直接连接到笔记本电脑或计算机

+   通过 USB 无线网络适配器将您的树莓派连接到互联网

+   通过代理服务器连接到互联网

+   使用 VNC 通过网络远程连接到树莓派

+   使用 SSH（和 X11 转发）通过网络远程连接到树莓派

+   通过 SMB 共享树莓派的主文件夹

+   保持树莓派最新

# 介绍

本章介绍了树莓派 3 和首次设置的过程。我们将树莓派连接到合适的显示器，电源和外围设备。我们将在 SD 卡上安装操作系统。这是系统启动所必需的。接下来，我们将确保我们可以通过本地网络成功连接到互联网。

最后，我们将利用网络提供远程连接和/或控制树莓派，以及确保系统保持最新状态的方法。

完成本章中的步骤后，您的树莓派将准备好供您进行编程使用。如果您已经设置并运行了您的树莓派，请确保您浏览以下部分，因为其中有许多有用的提示。

# 树莓派介绍

树莓派是由树莓派基金会创建的单板计算机，该基金会是一个旨在向英国儿童重新介绍低级计算机技能的慈善机构。其目标是重新点燃 20 世纪 80 年代的微型计算机革命，这产生了一整代熟练的程序员。

即使在 2012 年 2 月底发布计算机之前，树莓派已经在全球范围内获得了巨大的追随者，并且在撰写本书时已经销售了超过 1000 万台。以下图片显示了几种不同的树莓派型号：

![](img/83343a3b-cd26-4de4-b311-84d0521ea7eb.png)树莓派 3B 型号，A+型号和 Pi Zero

# 名字是怎么回事？

树莓派的名称是希望创建一个以水果命名的替代计算机（如苹果，黑莓和杏子），并向最初的概念致敬，即可以使用 Python 编程的简单计算机（缩写为 Pi）。

在这本书中，我们将拿起这台小电脑，了解如何设置它，然后逐章探索它的功能，使用 Python 编程语言。

# 为什么选择 Python？

经常有人问：“为什么选择 Python 作为树莓派上的编程语言？”事实上，Python 只是可以在树莓派上使用的许多编程语言之一。

有许多编程语言可供选择，从高级图形块编程，如 Scratch，到传统的 C，再到 BASIC，甚至原始的机器码汇编语言。一个优秀的程序员通常必须精通多种编程语言，以便能够充分发挥每种语言的优势和劣势，以最好地满足其所需应用的需求。了解不同语言（和编程技术）如何克服将*您想要的*转换为*您得到的*的挑战是有用的，因为这也是您在编程时所要做的。

Python 被选为学习编程的良好起点，因为它提供了丰富的编码工具，同时又允许编写简单的程序而无需烦恼。这使得初学者可以逐渐了解现代编程语言的概念和方法，而无需从一开始就了解所有内容。它非常模块化，有许多额外的库可以导入以快速扩展功能。随着时间的推移，您会发现这会鼓励您做同样的事情，并且您会想要创建自己的模块，可以将其插入到自己的程序中，从而迈出结构化编程的第一步。

Python 解决了格式和表现方面的问题。缩进会增加可读性，在 Python 中缩进非常重要。它们定义了代码块如何组合在一起。一般来说，Python 运行速度较慢；因为它是解释性的，所以在运行程序时创建模块需要时间。如果需要对时间关键事件做出响应，这可能会成为一个问题。然而，您可以预编译 Python 或使用其他语言编写的模块来克服这个问题。

它隐藏了细节；这既是优点也是缺点。对于初学者来说很好，但当您不得不猜测数据类型等方面时可能会有困难。然而，这反过来又迫使您考虑所有可能性，这可能是一件好事。

# Python 2 和 Python 3

对于初学者来说，一个巨大的困惑来源是树莓派上有两个版本的 Python（**版本 2.7**和**版本 3.6**），它们彼此不兼容，因此为 Python 2.7 编写的代码可能无法在 Python 3.6 上运行（反之亦然）。

**Python 软件基金会**不断努力改进并推动语言向前发展，这有时意味着他们必须牺牲向后兼容性来拥抱新的改进（并且重要的是，去除多余和过时的做法）。

支持 Python 2 和 Python 3

有许多工具可以帮助您从 Python 2 过渡到 Python 3，包括转换器，如`2to3`，它将解析并更新您的代码以使用 Python 3 的方法。这个过程并不完美，在某些情况下，您需要手动重写部分代码并进行全面测试。您可以编写同时支持两者的代码和库。`import __future__`语句允许您导入 Python 3 的友好方法，并在 Python 2.7 中运行它们。

# 您应该使用哪个版本的 Python？

基本上，选择使用哪个版本将取决于您的意图。例如，您可能需要 Python 2.7 库，这些库尚未适用于 Python 3.6。Python 3 自 2008 年就已经推出，因此这些库往往是较老或较大的库，尚未被翻译。在许多情况下，旧库有新的替代方案；然而，它们的支持程度可能有所不同。

在这本书中，我们使用的是 Python 3.6，它也兼容 Python 3.5 和 3.3。

# 树莓派家族 - 树莓派的简史

自发布以来，树莓派已经推出了各种版本，对原始的树莓派 B 型进行了小型和大型的更新和改进。虽然一开始可能会令人困惑，但树莓派有三种基本类型可用（以及一个特殊型号）。

主要的旗舰型号被称为**B 型**。它具有所有的连接和功能，以及最大的 RAM 和最新的处理器。多年来，已经推出了几个版本，最值得注意的是 B 型（拥有 256MB 和 512MB RAM），然后是 B+型（将 26 针 GPIO 增加到 40 针，改用 microSD 卡槽，并将 USB 端口从两个增加到四个）。这些原始型号都使用了 Broadcom BCM2835 **系统芯片**（**SOC**），包括单核 700MHz ARM11 和 VideoCore IV **图形处理单元**（**GPU**）。

2015 年发布的树莓派 2 型 B（也称为 2B）引入了新的 Broadcom BCM2836 SOC，提供了四核 32 位 ARM Cortex A7 1.2 GHz 处理器和 GPU，配备 1GB 的 RAM。改进的 SOC 增加了对 Ubuntu 和 Windows 10 IoT 的支持。最后，我们有了最新的树莓派 3 型 B，使用了另一个新的 Broadcom BCM2837 SOC，提供了四核 64 位 ARM Cortex-A53 和 GPU，以及板载 Wi-Fi 和蓝牙。

**A 型**一直被定位为精简版本。虽然具有与 B 型相同的 SOC，但连接有限，只有一个 USB 端口，没有有线网络（LAN）。A+型再次增加了更多的 GPIO 引脚和一个 microSD 卡槽。然而，RAM 后来升级为 512MB，再次只有一个 USB 端口/没有 LAN。Model A 上的 Broadcom BCM2835 SOC 到目前为止还没有更新（因此仍然是单核 ARM11）；但是，可能会推出 3A 型（很可能使用 BCM2837）。

**Pi Zero**是树莓派的超紧凑版本，适用于成本和空间有限的嵌入式应用。它具有与其他型号相同的 40 针 GPIO 和 microSD 卡槽，但缺少板载显示（CSI 和 DSI）连接。它仍然具有 HDMI（通过迷你 HDMI）和单个 micro USB **on-the-go**（**OTG**）连接。尽管在 Pi Zero 的第一个版本中没有，但最新型号还包括用于板载摄像头的 CSI 连接。

Pi Zero 在 2015 年被著名地发布，并随树莓派基金会的杂志*The MagPi*一起赠送，使该杂志成为第一本在封面上赠送计算机的杂志！这让我感到非常自豪，因为（正如你可能在本书开头的我的传记中读到的那样）我是该杂志的创始人之一。

特殊型号被称为**计算模块**。它采用 200 针 SODIMM 卡的形式。它适用于工业用途或商业产品中，所有外部接口将由主机/主板提供，模块将插入其中。示例产品包括 Slice Media Player（[`fiveninjas.com`](http://fiveninjas.com)）和 OTTO 相机。当前模块使用 BCM2835，尽管有一个更新的计算模块（CM3）。

树莓派维基百科页面提供了所有不同变体及其规格的完整列表：[`en.wikipedia.org/wiki/Raspberry_Pi#Specifications`](https://en.wikipedia.org/wiki/Raspberry_Pi#Specifications%20)

此外，树莓派产品页面提供了有关可用型号和配件规格的详细信息：[`www.raspberrypi.org/products/`](https://www.raspberrypi.org/products/)

# 选择哪种树莓派？

本书的所有部分都与当前所有版本的树莓派兼容，但建议首选 3B 型作为最佳型号。这提供了最佳性能（特别是对于 OpenCV 示例中使用的 GPU 示例，如第五章中的*检测图像中的边缘和轮廓*），大量连接和内置 Wi-Fi，非常方便。

Pi Zero 被推荐用于需要低功耗或减少重量/尺寸但不需要 Model 3B 全面处理能力的项目。然而，由于其超低成本，Pi Zero 非常适合在开发完成项目后部署。

# 连接到树莓派

有许多方法可以连接树莓派并使用各种接口查看和控制内容。对于典型的用途，大多数用户将需要电源、显示器（带音频）和输入方法，如键盘和鼠标。要访问互联网，请参阅*通过 LAN 连接器将树莓派连接到互联网*或*在树莓派上使用内置 Wi-Fi 和蓝牙*。

# 准备就绪

在使用树莓派之前，您需要一个安装了操作系统或者在其中安装了**新开箱系统**（**NOOBS**）的 SD 卡，如*使用 NOOBS 设置树莓派 SD 卡*中所讨论的那样。

以下部分将详细介绍您可以连接到树莓派的设备类型，以及重要的是如何在哪里插入它们。

正如您将在后面发现的那样，一旦您设置好了树莓派，您可能会决定通过网络连接远程连接并使用它，在这种情况下，您只需要电源和网络连接。请参考以下部分：*通过 VNC 远程连接树莓派*和*通过 SSH（和 X11 转发）远程连接树莓派*。

# 操作步骤如下...

树莓派的布局如下图所示：

![](img/56a82a7b-fb86-4235-aa5e-8692e6621894.jpg)树莓派连接布局（3 B 型，A+型和 Zero 型）

有关上图的更多信息如下：

+   **显示**：树莓派支持以下三种主要的显示连接；如果 HDMI 和复合视频都连接了，它将默认为仅 HDMI：

+   **HDMI**：为了获得最佳效果，请使用具有 HDMI 连接的电视或显示器，从而实现最佳分辨率显示（1080p）和数字音频输出。如果您的显示器具有 DVI 连接，您可以使用适配器通过 HDMI 连接。有几种类型的 DVI 连接；一些支持模拟（DVI-A），一些支持数字（DVI-D），一些都支持（DVI-I）。树莓派只能通过 HDMI 提供数字信号，因此建议使用 HDMI 到 DVI-D 适配器（在下图中带有勾号）。这缺少了四个额外的模拟引脚（在下图中带有叉号），因此可以适配到 DVI-D 和 DVI-I 类型插座中：

![](img/2bcb7807-a9ce-451c-860a-62ef60a37bdc.png)HDMI 到 DVI 连接（DVI-D 适配器）

如果您希望使用旧的显示器（带有 VGA 连接），则需要额外的 HDMI 到 VGA 转换器。树莓派还支持一个基本的 VGA 适配器（VGA Gert666 Adaptor），它直接驱动 GPIO 引脚。然而，这会使用 40 针引脚排头的所有引脚（旧的 26 针型号不支持 VGA 输出）：

![](img/42955bff-c5f3-4ab0-bf4d-b9242eb202d5.jpg)HDMI 到 VGA 适配器

+   +   **模拟**：另一种显示方法是使用模拟复合视频连接（通过音频插孔）；这也可以连接到 S-Video 或欧洲 SCART 适配器。然而，模拟视频输出的最大分辨率为 640 x 480 像素，因此不太适合一般使用：

![](img/99eac7fb-4cd9-42cd-b8dc-7160a4fc9241.jpg)3.5 毫米音频模拟连接

在使用 RCA 连接或 DVI 输入时，音频必须通过模拟音频连接单独提供。为了简化制造过程（避免穿孔元件），Pi Zero 没有模拟音频或模拟视频的 RCA 插孔（尽管可以通过一些修改添加）：

+   +   **直接显示 DSI**：由树莓派基金会生产的触摸显示器将直接连接到 DSI 插座。这可以与 HDMI 或模拟视频输出同时连接和使用，以创建双显示设置。

+   **立体声模拟音频（除 Pi Zero 外）**：这为耳机或扬声器提供了模拟音频输出。可以通过树莓派桌面上的配置工具或通过命令行使用`amixer`或`alsamixer`在模拟（立体插孔）和数字（HDMI）之间进行切换。

要了解有关终端中特定命令的更多信息，您可以在终端读取手册之前使用以下`man`命令（大多数命令应该都有手册）：

` man amixer`

有些命令还支持`--help`选项，以获得更简洁的帮助，如下所示：

` amixer --help`

+   **网络（不包括 A 型和 Pi Zero）**：网络连接将在本章后面的*通过 LAN 连接器将树莓派连接到互联网*配方中进行讨论。如果使用 A 型树莓派，可以添加 USB 网络适配器来添加有线或无线网络连接（参考*通过 USB Wi-Fi dongle 将树莓派连接到互联网*配方）。

+   **内置 Wi-Fi 和蓝牙（仅限 Model 3 B）**：Model 3 B 具有内置的 802.11n Wi-Fi 和蓝牙 4.1；参见*在树莓派上使用内置的 Wi-Fi 和蓝牙*配方。

+   **USB（1x Model A/Zero，2x Model 1 B，4x Model 2 B 和 3 B）**：使用键盘和鼠标：

+   树莓派应该可以与大多数 USB 键盘和鼠标兼容。您也可以使用使用 RF dongles 的无线鼠标和键盘。但是，对于使用蓝牙 dongles 的设备需要额外的配置。

+   如果您的电源供应不足或设备正在吸取过多电流，您可能会发现键盘按键似乎卡住了，并且在严重情况下，SD 卡可能会损坏。

早期 Model B 修订版 1 板的 USB 电源可能存在问题，这些板在 2012 年 10 月之前就已经上市。它们在 USB 输出上包含了额外的**Polyfuses**，如果超过 140 mA 的电流被吸取，就会跳闸。Polyfuses 可能需要数小时甚至数天才能完全恢复，因此即使电源改善了，也可能导致不可预测的行为。

您可以通过缺少后期型号上存在的四个安装孔来识别修订版 1 板。

+   +   Debian Linux（Raspbian 的基础）支持许多常见的 USB 设备，如闪存驱动器、硬盘驱动器（可能需要外部电源）、相机、打印机、蓝牙和 Wi-Fi 适配器。一些设备将被自动检测，而其他设备将需要安装驱动程序。

+   **Micro USB 电源**：树莓派需要一个能够舒适地提供至少 1,000 mA（特别是对于更耗电的 2 型和 3 型，建议提供 1,500 mA 或更多）的 5V 电源，带有一个 micro USB 连接。可以使用便携式电池组来为设备供电，比如适用于平板电脑的电池组。再次确保它们可以提供 5V 的电压，至少 1,000 mA。

在连接电源之前，您应该尽量将所有其他连接连接到树莓派上。但是，USB 设备、音频和网络可以在运行时连接和移除，而不会出现问题。

# 还有更多...

除了标准的主要连接之外，树莓派还具有许多其他连接。

# 次要硬件连接

以下每个连接都提供了树莓派的其他接口：

+   **20 x 2 GPIO 引脚排针（Model A+，B+，2 B，3 B 和 Pi Zero）**：这是树莓派的主要 40 针 GPIO 引脚排针，用于直接与硬件组件进行接口。本书中的章节也适用于具有 13 x 2 GPIO 引脚排针的较旧型号的树莓派。

+   **P5 8 x 2 GPIO 引脚排针（仅限 Model 1 B 修订版 2.0）**：我们在本书中不使用这个接口。

+   **复位连接**：这个连接出现在较新的型号上（没有插脚）。当引脚 1（复位）和引脚 2（GND）连接在一起时，会触发复位。我们在第七章的*A controlled shutdown button*概念中使用了这个接口，*Using Python to Drive Hardware*。

+   **GPU/LAN JTAG**：**联合测试行动组**（**JTAG**）是用于配置和测试处理器的编程和调试接口。这些接口出现在较新的型号上作为表面垫。使用这个接口需要专门的 JTAG 设备。我们在本书中不使用这个接口。

+   **直接相机 CSI**：这个连接支持树莓派相机模块。请注意，Pi Zero 的 CSI 连接器比其他型号要小，因此需要不同的排线连接器。

+   **直接显示 DSI**：此连接支持直接连接的显示器，例如 7 英寸 800 x 600 电容触摸屏。

# 使用 NOOBS 设置您的 Raspberry Pi SD 卡

在启动之前，树莓派需要将操作系统加载到 SD 卡上。设置 SD 卡的最简单方法是使用**NOOBS**；您可能会发现可以购买已经加载了 NOOBS 的 SD 卡。

NOOBS 提供了一个初始启动菜单，提供了安装几种可用操作系统到您的 SD 卡的选项。

# 准备工作

由于 NOOBS 创建了一个**RECOVERY**分区来保存原始安装映像，建议使用 8GB 或更大的 SD 卡。您还需要一个 SD 卡读卡器（经验表明，一些内置读卡器可能会导致问题，因此建议使用外部 USB 类型读卡器）。

如果您使用的是以前使用过的 SD 卡，可能需要重新格式化以删除任何先前的分区和数据。NOOBS 期望 SD 卡由单个 FAT32 分区组成。

如果使用 Windows 或 macOS X，您可以使用 SD 卡协会的格式化程序，如下面的屏幕截图所示（可在[`www.sdcard.org/downloads/formatter_4/`](https://www.sdcard.org/downloads/formatter_4/)找到）：

![](img/65551eb0-dc6e-41c3-b473-35b6251fb11e.png)使用 SD 格式化程序清除 SD 卡上的任何分区

从选项设置对话框中，设置 FORMAT SIZE ADJUSTMENT。这将删除以前创建的所有 SD 卡分区。

如果使用 Linux，您可以使用`gparted`清除任何先前的分区并将其重新格式化为 FAT32 分区。

完整的 NOOBS 软件包（通常略大于 1GB）包含了 Raspbian，最受欢迎的树莓派操作系统映像。还提供了一个精简版的 NOOBS，它没有预装的操作系统（尽管需要 Raspberry Pi 上的较小的初始下载 20MB 和网络连接来直接下载您打算使用的操作系统）。

NOOBS 可在[`www.raspberrypi.org/downloads`](http://www.raspberrypi.org/downloads)上获得，文档可在[`github.com/raspberrypi/noobs`](https://github.com/raspberrypi/noobs)上获得。

# 如何做...

通过执行以下步骤，我们将准备好 SD 卡来运行 NOOBS。然后，这将允许我们选择并安装我们想要使用的操作系统：

1.  准备好您的 SD 卡。

1.  在新格式化或新的 SD 卡上，复制`NOOBS_vX.zip`文件的内容。复制完成后，您应该得到类似于 SD 卡以下屏幕截图的东西：

![](img/3004e77d-458c-4bc9-9aaa-6fa1afe9f37c.png)从 SD 卡中提取的 NOOBS 文件这些文件可能会因不同版本的 NOOBS 而略有不同，并且在您的计算机上显示的图标可能会有所不同。

1.  您现在可以将卡插入树莓派，连接键盘和显示器，然后打开电源。有关所需的详细信息和操作方法，请参阅*连接到树莓派*配方。

默认情况下，NOOBS 将通过 HDMI 连接显示。如果您有其他类型的屏幕（或者什么也看不到），您需要通过按 1、2、3 或 4 手动选择输出类型，具体操作如下：

+   键 1 代表标准 HDMI 模式（默认模式）

+   键 2 代表安全 HDMI 模式（如果未检测到输出，则为备用 HDMI 设置）

+   键 3 代表复合 PAL（通过 RCA 模拟视频连接进行连接）

+   键 4 代表复合 NTSC（同样，适用于通过 RCA 连接器连接）

此显示设置也将用于安装的操作系统。

过了一会儿，您将看到列出可用发行版的 NOOBS 选择屏幕（离线版本仅包括 Raspbian）。 有许多其他可用的发行版，但只有选定的发行版可以直接通过 NOOBS 系统获得。 点击 Raspbian，因为这是本书中使用的操作系统。

按*Enter*或单击“安装操作系统”，并确认您希望覆盖卡上的所有数据

卡。 这将覆盖以前使用 NOOBS 安装的任何发行版，但不会删除 NOOBS 系统； 您可以在任何时候按下*Shift*键返回到它。

根据其速度，写入数据到卡上大约需要 20 到 40 分钟。 当完成并出现“图像应用成功”消息时，单击“确定”，树莓派将开始引导到`树莓派桌面`。

# 它是如何工作的...

以这种方式将映像文件写入 SD 卡的目的是确保 SD 卡格式化为预期的文件系统分区和文件，以正确引导操作系统。

当树莓派启动时，它会加载存储在 GPU 内存中的一些特殊代码（通常被树莓派基金会称为**二进制块**）。 二进制块提供了读取 SD 卡上的`BOOT`分区所需的指令（在 NOOBS 安装的情况下，将从`RECOVERY`分区加载 NOOBS）。 如果此时按下*Shift*，NOOBS 将加载恢复和安装菜单。 否则，NOOBS 将根据`SETTINGS`分区中存储的首选项开始加载操作系统。

在加载操作系统时，它将通过`BOOT`分区引导，使用`config.txt`中定义的设置和`cmdline.txt`中的选项最终加载到`root`分区上的桌面。 请参阅以下图表：

![](img/fb2c2666-39a2-43b9-985f-be7e7deb990d.png)NOOBS 在 SD 卡上创建了几个分区，以允许安装多个

操作系统，并提供恢复

NOOBS 允许用户在同一张卡上选择性地安装多个操作系统，并提供引导菜单以在它们之间进行选择（在超时期间设置默认值的选项）。

如果以后添加、删除或重新安装操作系统，请首先确保复制任何文件，包括您希望保留的系统设置，因为 NOOBS 可能会覆盖 SD 卡上的所有内容。

# 还有更多...

当您首次直接启动树莓派时，将加载桌面。 您可以使用 Raspberry Pi Configuration 菜单（在桌面上的首选项菜单下或通过`sudo raspi-config`命令）配置系统设置。 使用此菜单，您可以更改 SD 卡或设置一般首选项：

![](img/82110ea1-fcde-45ef-844f-f77baea5d0be.png)

# 更改默认用户密码

确保在登录后更改`pi`用户帐户的默认密码，因为默认密码是众所周知的。 如果您连接到公共网络，这一点尤为重要。 您可以使用`passwd`命令来执行此操作，如下面的屏幕截图所示：

![](img/4ea09f37-0f50-4cca-8e23-29a824830576.png)为 Pi 用户设置新密码

这样可以更加放心，因为如果您以后连接到另一个网络，只有您才能访问您的文件并控制您的树莓派。

# 确保安全关闭

为了避免任何数据损坏，您必须确保通过发出`shutdown`命令正确关闭树莓派，如下所示：

```py
sudo shutdown -h now  
```

或者，使用这个：

```py
sudo halt  
```

在从树莓派断电之前，必须等待此命令完成（在 SD 卡访问指示灯停止闪烁后等待至少 10 秒）。

您还可以使用`reboot`命令重新启动系统，如下所示：

```py
sudo reboot  
```

# 手动准备 SD 卡

使用 NOOBS 的替代方法是手动将操作系统映像写入 SD 卡。 尽管最初这是安装操作系统的唯一方法，但一些用户仍然更喜欢它。 它允许在将 SD 卡用于树莓派之前准备 SD 卡。 它还可以更容易地访问启动和配置文件，并且为用户留下更多的空间（与 NOOBS 不同，不包括`RECOVERY`分区）。

默认的 Raspbian 映像实际上由两个分区`BOOT`和`SYSTEM`组成，可以放入 2GB 的 SD 卡（建议使用 4GB 或更多）。

您需要一台运行 Windows/Mac OS X/Linux 的计算机（尽管可以使用另一台树莓派来写入您的卡；请准备等待很长时间）。

下载您希望使用的操作系统的最新版本。 本书假定您正在使用[`www.raspberrypi.org/downloads`](http://www.raspberrypi.org/downloads)上提供的最新版本的 Raspbian。

根据您计划用于写入 SD 卡的计算机类型执行以下步骤（您需要的`.img`文件有时会被压缩，因此在开始之前，您需要提取文件）。

以下步骤适用于 Windows：

1.  确保您已经下载了 Raspbian 映像，并将其提取到一个方便的文件夹以获取`.img`文件。

1.  获取[`www.sourceforge.net/projects/win32diskimager`](http://www.sourceforge.net/projects/win32diskimager)上提供的`Win32DiskImager.exe`文件。

1.  从下载位置运行`Win32DiskImager.exe`。

1.  单击文件夹图标，导航到`.img`文件的位置，然后单击保存。

1.  如果尚未这样做，请将 SD 卡插入卡读卡器并将其插入计算机。

1.  从小下拉框中选择与您的 SD 卡对应的设备驱动器号。 仔细检查这是否是正确的设备（因为在写入映像时，程序将覆盖设备上的任何内容）。

在选择源映像文件之前，可能不会列出驱动器号。

1.  最后，单击“写入”按钮，等待程序将映像写入 SD 卡，如下图所示：

![](img/049daa83-6669-47de-b5c0-01a76330e233.png)手动将操作系统映像写入 SD 卡，使用 Disk Imager

1.  完成后，您可以退出程序。 您的 SD 卡已准备就绪。

以下步骤适用于大多数常见的 Linux 发行版，如 Ubuntu 和 Debian：

1.  使用您喜欢的网络浏览器下载 Raspbian 映像并将其保存在合适的位置。

1.  从文件管理器中提取文件或在终端中找到文件夹并使用以下命令解压`.img`文件：

```py
unzip filename.zip  
```

1.  如果尚未这样做，请将 SD 卡插入卡读卡器并将其插入计算机。

1.  使用`df -h`命令并识别 SD 卡的**sdX**标识符。 每个分区将显示为 sdX1，sdX2 等，其中 X 将是`a`，`b`，`c`，`d`等，用于设备 ID。

1.  确保使用以下命令卸载 SD 卡上的所有分区

对于每个分区，使用`umount /dev/sdXn`命令，其中`sdXn`是要卸载的分区。

1.  使用以下命令将映像文件写入 SD 卡：

```py
sudo dd if=filename.img of=/dev/sdX bs=4M  
```

1.  写入 SD 卡的过程需要一些时间，在完成时返回终端提示符。

1.  使用以下命令卸载 SD 卡，然后从计算机中取出它：

```py
umount /dev/sdX1  
```

以下步骤适用于大多数 OS X 版本：

1.  使用您喜欢的网络浏览器下载 Raspbian 映像并将其保存在合适的位置。

1.  从文件管理器中提取文件或在终端中找到文件夹并解压`.img`文件，使用以下命令：

```py
unzip filename.zip  
```

1.  如果尚未这样做，请将 SD 卡插入卡读卡器并将其插入计算机。

1.  使用`diskutil list`命令并为 SD 卡标识**disk#**标识符。每个分区将显示为 disk#s1，disk#s2 等，其中#将是`1`，`2`，`3`，`4`等，用于设备 ID。

如果列出了 rdisk#，则使用它可以更快地写入（这使用原始路径并跳过数据缓冲）。

1.  确保使用`unmountdisk /dev/diskX`命令卸载 SD 卡，其中`diskX`是要卸载的设备。

1.  使用以下命令将映像文件写入 SD 卡：

```py
sudo dd if=filename.img of=/dev/diskX bs=1M  
```

1.  该过程将花费一些时间来写入 SD 卡，并在完成时返回到终端提示符。

1.  在从计算机中移除 SD 卡之前卸载 SD 卡，使用

以下命令：

```py
unmountdisk /dev/diskX  
```

请参阅以下图表：

![](img/30a0a201-910a-4aee-8735-3f354c3f9228.png)手动安装的 OS 映像的引导过程

# 扩展系统以适应您的 SD 卡

手动编写的映像将是固定大小的（通常是为了适应最小尺寸的 SD 卡）。要充分利用 SD 卡，您需要扩展系统分区以填满 SD 卡的其余部分。这可以通过 Raspberry Pi 配置工具实现。

选择“扩展文件系统”，如下截图所示：

![](img/82597568-2a12-462d-b662-d7333f7ae747.png)Raspberry Pi 配置工具

# 访问 RECOVERY/BOOT 分区

Windows 和 macOS X 不支持`ext4`格式，因此在读取 SD 卡时，只有**文件分配表**（**FAT**）分区可访问。此外，Windows 只支持 SD 卡上的第一个分区，因此如果您安装了 NOOBS，则只能看到`RECOVERY`分区。如果您手动写入了卡，您将能够访问`BOOT`分区。

`data`分区（如果您通过 NOOBS 安装了它）和`root`分区采用`ext4`格式，通常在非 Linux 系统上不可见。

如果您确实需要使用 Windows 从 SD 卡读取文件，一个免费软件**Linux Reader**（可在[www.diskinternals.com/linux-reader](https://www.diskinternals.com/linux-reader)获取）可以提供对 SD 卡上所有分区的只读访问。

从 Raspberry Pi 访问分区。要查看当前已挂载的分区，请使用`df`，如下截图所示：

![](img/0ba3c22d-241e-4ea5-a000-1eed0113eedd.png)df 命令的结果

要从 Raspbian 内部访问`BOOT`分区，请使用以下命令：

```py
cd /boot/  
```

要访问`RECOVERY`或`data`分区，我们必须通过执行以下操作来挂载它

以下步骤：

1.  通过列出所有分区（包括未挂载的分区）来确定分区的名称，系统将引用它。`sudo fdisk -l`命令列出分区，如下截图所示：

![](img/e3c01d3d-a6bc-4236-a11c-16928be5f30f.png)NOOBS 安装和数据分区

以下表格显示了分区的名称及其含义

| **分区名称** | **含义** |
| --- | --- |
| `mmcblk0p1` | (`VFAT`) `RECOVERY` |
| `mmcblk0p2` | (扩展分区) 包含 (`root`, `data`, `BOOT`) |
| `mmcblk0p5` | (`ext4`) `root` |
| `mmcblk0p6` | (`VFAT`) `BOOT` |
| `mmcblk0p7` | (`ext4`) `SETTINGS` |

如果您在同一张卡上安装了其他操作系统，则前面表中显示的分区标识符将不同。

1.  创建一个文件夹，并将其设置为分区的挂载点；对于`RECOVERY`分区，请使用以下命令：

```py
mkdir ~/recovery
sudo mount -t vfat /dev/mmcblk0p1 ~/recovery  
```

为了确保它们在每次系统启动时都被挂载，请执行以下步骤：

1.  在`/etc/rc.local`中添加`sudo`挂载命令，然后是`exit 0`。如果您有不同的用户名，您需要将`pi`更改为匹配的用户名：

```py
sudo nano /etc/rc.local
sudo mount -t vfat /dev/mmcblk0p1 /home/pi/recovery  
```

1.  按*Ctrl* + *X*，*Y*和*Enter*保存并退出。

添加到`/etc/rc.local`的命令将在登录到 Raspberry Pi 的任何用户上运行。如果您只希望该驱动器对当前用户进行挂载，则可以将命令添加到`.bash_profile`中。

如果必须在同一张卡上安装其他操作系统，则此处显示的分区标识符将不同。

# 使用工具备份 SD 卡以防故障

您可以使用**Win32 Disk Imager**通过将 SD 卡插入读卡器，启动程序并创建一个文件名来存储图像来制作 SD 卡的完整备份图像。只需点击读取按钮，将图像从 SD 卡读取并写入新的图像文件。

要备份系统，或者使用树莓派克隆到另一个 SD 卡，请使用 SD 卡复制器（可通过桌面菜单的附件| SD 卡复制器获得）。

将 SD 卡插入树莓派的 USB 端口的读卡器中，并选择新的存储设备，如下截图所示：

![](img/33074ea1-d5f9-4519-9f99-c56cdb9da373.png)SD 卡复制程序

在继续之前，SD 卡复制器将确认您是否希望格式化和覆盖目标设备，并且如果有足够的空间，将克隆您的系统。

`dd`命令也可以用来备份卡，如下所示：

+   对于 Linux，用您的设备 ID 替换`sdX`，使用以下命令：

```py
sudo dd if=/dev/sdX of=image.img.gz bs=1M  
```

+   对于 OS X，用以下命令替换`diskX`为您的设备 ID：

```py
sudo dd if=/dev/diskX of=image.img.gz bs=1M
```

+   您还可以使用`gzip`和 split 来压缩卡的内容并将其拆分成多个文件，如果需要的话，以便进行简单的存档，如下所示：

```py
sudo dd if=/dev/sdX bs=1M | gzip -c | split -d -b 2000m - image.img.gz

```

+   要恢复拆分的图像，请使用以下命令：

```py
sudo cat image.img.gz* | gzip -dc | dd of=/dev/sdX bs=1M  
```

# 通过以太网端口将树莓派连接到互联网的网络和连接，使用 CAT6 以太网电缆

将树莓派连接到互联网的最简单方法是使用 Model B 上的内置 LAN 连接。如果您使用的是 Model A 树莓派，则可以使用 USB 到 LAN 适配器（有关如何配置此适配器的详细信息，请参阅*还有更多...*部分的*通过 USB 无线网络适配器连接树莓派到互联网*配方）。

# 准备工作

您将需要访问适当的有线网络，该网络将连接到互联网，并且标准网络电缆（带有**RJ45**类型连接器，用于连接到树莓派）。

# 操作步骤

许多网络使用**动态主机配置协议**（**DHCP**）自动连接和配置，由路由器或交换机控制。如果是这种情况，只需将网络电缆插入路由器或网络交换机上的空闲网络端口（或者如果适用，墙壁网络插座）。

或者，如果没有 DHCP 服务器，您将不得不手动配置设置（有关详细信息，请参阅*还有更多...*部分）。

您可以通过以下步骤确认这一功能是否成功运行：

1.  确保树莓派两侧的两个 LED 灯亮起（左侧橙色 LED 指示连接，右侧绿色 LED 显示闪烁的活动）。这将表明与路由器有物理连接，并且设备已经供电并正常工作。

1.  使用`ping`命令测试与本地网络的连接。首先，找出网络上另一台计算机的 IP 地址（或者您的路由器的地址，通常为`192.168.0.1`或`192.168.1.254`）。现在，在树莓派终端上，使用`ping`命令（使用`-c 4`参数仅发送四条消息；否则，按*Ctrl* + *C*停止）ping IP 地址，如下所示：

```py
sudo ping 192.168.1.254 -c 4
```

1.  测试连接到互联网（如果您通常通过代理服务器连接到互联网，这将失败）如下：

```py
sudo ping www.raspberrypi.org -c 4

```

1.  最后，您可以通过发现来测试与树莓派的连接

在树莓派上使用`hostname -I`命令查找 IP 地址。然后，您可以在网络上的另一台计算机上使用`ping`命令来确保可以访问（使用树莓派的 IP 地址代替[www.raspberrypi.org](https://www.raspberrypi.org/)）。Windows 版本的`ping`命令将执行五次 ping 并自动停止，并且不需要`-c 4`选项。

如果上述测试失败，您需要检查您的连接，然后确认您的网络的正确配置。

# 还有更多...

如果您经常在网络上使用树莓派，您就不想每次连接时都要查找 IP 地址。

在一些网络上，您可以使用树莓派的主机名而不是其 IP 地址（默认为`raspberrypi`）。为了帮助实现这一点，您可能需要一些额外的软件，比如**Bonjour**，以确保网络上的主机名被正确注册。如果您使用的是 macOS X，那么 Bonjour 已经在运行了。

在 Windows 上，您可以安装 iTunes（如果您没有安装），它也包括了该服务，或者您可以单独安装（通过[`support.apple.com/kb/DL999`](https://support.apple.com/kb/DL999)提供的 Apple Bonjour Installer）。然后您可以使用主机名`raspberrypi`或`raspberrypi.local`来连接到树莓派。如果您需要更改主机名，那么您可以使用之前显示的树莓派配置工具来进行更改。

或者，您可能会发现手动设置 IP 地址为已知值并将其固定对您有所帮助。但是，请记住在连接到另一个网络时切换回使用 DHCP。

一些路由器还可以设置**静态 IP DHCP 地址**的选项，这样相同的地址就会始终分配给树莓派（如何设置取决于路由器本身）。

如果您打算使用后面描述的远程访问解决方案之一，那么了解树莓派的 IP 地址或使用主机名尤其有用，这样就避免了需要显示器。

# 使用树莓派上的内置 Wi-Fi 和蓝牙

许多家庭网络通过 Wi-Fi 提供无线网络；如果您有树莓派 3，那么您可以利用板载的 Broadcom Wi-Fi 来连接。树莓派 3 也支持蓝牙，因此您可以连接大多数标准蓝牙设备，并像在任何其他计算机上一样使用它们。

这种方法也适用于任何受支持的 USB Wi-Fi 和蓝牙设备；有关识别设备和安装固件（如果需要）的额外帮助，请参阅*通过 USB Wi-Fi dongle 将树莓派连接到互联网的网络和连接*配方。

# 准备好了

Raspbian 的最新版本包括有用的实用程序，通过图形界面快速轻松地配置您的 Wi-Fi 和蓝牙。

**注意**：如果您需要通过命令行配置 Wi-Fi，请参阅*通过 USB Wi-Fi dongle 将树莓派连接到互联网的网络和连接*配方以获取详细信息。![](img/c665e469-5c0e-4791-81c9-71f74ebd531a.png)Wi-Fi 和蓝牙配置应用程序

您可以使用内置的蓝牙连接无线键盘、鼠标，甚至无线扬声器。这对于一些需要额外电缆和线缆的项目非常有帮助，比如机器人项目，或者当树莓派安装在难以到达的位置时（作为服务器或安全摄像头）。

# 如何做...

以下是各种方法。

# 连接到您的 Wi-Fi 网络

要配置您的 Wi-Fi 连接，请单击网络符号以列出本地可用的 Wi-Fi 网络：

![](img/3bfd82d0-7b3a-4cca-9ab6-c883a400f68d.png)区域内可用接入点的 Wi-Fi 列表

选择所需的网络（例如，`Demo`），如果需要，输入您的密码（也称为`预共享密钥`）：

![](img/248e4d19-cf0c-4e24-8142-c96c598a74a6.png)为接入点提供密码

过一会儿，您应该会看到您已连接到网络，图标将变成 Wi-Fi 符号。如果遇到问题，请确保您有正确的密码/密钥：

![](img/51496409-6ba4-4dbd-ab18-6e596404e97b.png)成功连接到接入点

就是这样；就是这么简单！

您现在可以通过使用 Web 浏览器导航到网站或在终端中使用以下命令来测试连接并确保其正常工作：

```py
sudo ping www.raspberrypi.com
```

# 连接到蓝牙设备

首先，我们需要通过单击蓝牙图标并选择**使可发现**将蓝牙设备设置为可发现模式。您还需要使要连接的设备处于可发现和配对准备状态；这可能因设备而异（例如按配对按钮）：

![](img/4283c612-ff0d-4172-9a8d-315ed757ce2e.png)设置蓝牙为可发现状态

接下来，选择**添加设备...**并选择目标设备和**配对**：

![](img/a7c44c57-340a-47a4-84b8-e23818476ea5.png)选择并配对所需的设备

然后配对过程将开始；例如，BTKB-71DB 键盘将需要输入配对码`467572`到键盘上以完成配对。其他设备可能使用默认配对码，通常设置为 0000、1111、1234 或类似的：

按照说明使用所需的配对码将设备配对

一旦过程完成，设备将被列出，并且每次设备出现并引导时都会自动连接。

# 手动配置您的网络

如果您的网络不包括 DHCP 服务器或者已禁用（通常，这些都内置在大多数现代 ADSL/电缆调制解调器或路由器中），则可能需要手动配置您的网络设置。

# 准备工作

在开始之前，您需要确定网络的网络设置。

您需要从路由器的设置或连接到网络的另一台计算机中找到以下信息：

+   **IPv4 地址**：此地址需要选择与网络上其他计算机相似（通常，前三个数字应匹配，即，如果`netmask`为`255.255.255.0`，则应为`192.168.1.X`），但不应该已被其他计算机使用。但是，避免使用`x.x.x.255`作为最后一个地址，因为这是保留的广播地址。

+   **子网掩码**：此数字确定计算机将响应的地址范围（对于家庭网络，通常为`255.255.255.0`，允许最多 254 个地址）。这有时也被称为**netmask**。

+   **默认网关地址**：此地址通常是您的路由器的 IP 地址，通过它，计算机连接到互联网。

+   **DNS 服务器**：**域名服务**（**DNS**）服务器通过查找名称将名称转换为 IP 地址。通常，它们将已配置在您的路由器上，在这种情况下，您可以使用您的路由器的地址。或者，您的**互联网服务提供商**（**ISP**）可能会提供一些地址，或者您可以使用 Google 的公共 DNS 服务器的地址`8.8.8.8`和`8.8.4.4`。在某些系统中，这些也称为**名称服务器**。

对于 Windows，您可以通过连接到互联网并运行以下命令来获取此信息：

```py
ipconfig /all  
```

找到活动连接（通常称为`本地连接 1`或类似的，如果您使用有线连接，或者如果您使用 Wi-Fi，则称为无线网络连接），并找到所需的信息，如下所示：

![](img/a12947df-85ea-47f6-8a0d-ab3dde091efc.png)ipconfig/all 命令显示有关网络设置的有用信息

对于 Linux 和 macOS X，您可以使用以下命令获取所需的信息（请注意，这是`ifconfig`而不是`ipconfig`）：

```py
ifconfig  
```

DNS 服务器称为名称服务器，并且通常列在`resolv.conf`文件中。您可以使用以下方式使用`less`命令查看其内容（完成查看后按 Q 退出）：

```py
less /etc/resolv.conf  
```

# 如何做到...

要设置网络接口设置，请使用编辑`/etc/network/interfaces`

以下代码：

```py
sudo nano /etc/network/interfaces  
```

现在执行以下步骤：

1.  我们可以添加我们特定网络的详细信息，我们要为其分配的 IP`地址`号，网络的`netmask`地址和`gateway`地址，如下所示：

```py
iface eth0 inet static
 address 192.168.1.10
 netmask 255.255.255.0
 gateway 192.168.1.254

```

1.  按下*Ctrl* + *X*，*Y*和*Enter*保存并退出。

1.  要为 DNS 设置名称服务器，请使用以下代码编辑`/etc/resolv.conf`：

```py
sudo nano /etc/resolv.conf

```

1.  按照以下方式添加 DNS 服务器的地址：

```py
nameserver 8.8.8.8
nameserver 8.8.4.4  
```

1.  按下*Ctrl* + *X*，*Y*和*Enter*保存并退出。

# 还有更多...

您可以通过编辑`cmdline.txt`在`BOOT`分区中配置网络设置，并使用`ip`将设置添加到启动命令行。

`ip`选项采用以下形式：

```py
ip=client-ip:nfsserver-ip:gw-ip:netmask:hostname:device:autoconf  
```

+   `client-ip`选项是您要分配给树莓派的 IP 地址

+   `gw-ip`选项将手动设置网关服务器地址

+   `netmask`选项将直接设置网络的`netmask`

+   `hostname`选项将允许您更改默认的`raspberrypi`主机名

+   `device`选项允许您指定默认的网络设备（如果存在多个网络设备）

+   `autoconf`选项允许自动配置打开或关闭

# 直接连接到笔记本电脑或计算机

可以使用单个网络电缆直接连接树莓派 LAN 端口到笔记本电脑或计算机。这将在计算机之间创建一个本地网络链接，允许您进行连接到正常网络时可以做的所有事情，而无需使用集线器或路由器，包括连接到互联网，如果使用**Internet Connection Sharing** (**ICS**)，如下所示：

![](img/69bce998-225c-4d4e-ac1a-86b7b41ca09b.png)只需使用一个网络电缆，一个标准镜像的 SD 卡和电源即可使用树莓派。

ICS 允许树莓派通过另一台计算机连接到互联网。但是，需要对计算机进行一些额外的配置，以便它们在链接上进行通信，因为树莓派不会自动分配自己的 IP 地址。

我们将使用 ICS 来共享来自另一个网络链接的连接，例如笔记本电脑上的内置 Wi-Fi。或者，如果不需要互联网或计算机只有一个网络适配器，我们可以使用直接网络链接（请参阅*There's more...*部分下的*Direct network link*部分）。

尽管这个设置对大多数计算机都适用，但有些设置比其他设置更困难。有关更多信息，请参见[www.pihardware.com/guides/direct-network-connection](http://www.pihardware.com/guides/direct-network-connection)。

# 准备工作

您将需要带电源和标准网络电缆的树莓派。

树莓派 Model B LAN 芯片包括**Auto-MDIX**（**Automatic Medium-Dependent Interface Crossover**）。无需使用特殊的交叉电缆（一种特殊的网络电缆，布线使传输线连接到直接网络链接的接收线），芯片将根据需要自动决定和更改设置。

如果这是您第一次尝试，可能还需要键盘和显示器来进行额外的测试。

为了确保您可以将网络设置恢复到其原始值，您应该检查它是否具有固定的 IP 地址或网络是否自动配置。

要检查 Windows 10 上的网络设置，请执行以下步骤：

1.  从开始菜单打开设置，然后选择网络和 Internet，然后以太网，然后从相关设置列表中点击更改适配器选项。

要检查 Windows 7 和 Vista 上的网络设置，请执行以下步骤：

1.  从控制面板打开网络和共享中心，然后在左侧点击更改适配器设置。

1.  在 Windows XP 上检查网络设置，从控制面板打开网络连接。

1.  找到与您的有线网络适配器相关的项目（默认情况下，通常称为以太网或本地连接，如下图所示）：

![](img/8b2bf252-e5db-4372-8aa5-ca76a0055246.png)查找有线网络连接

1.  右键单击其图标，然后单击属性。将出现对话框，如此屏幕截图所示：

![](img/4945c960-282d-462e-b9a7-acaec76a4cd3.png)选择 TCP/IP 属性并检查设置

1.  选择名为 Internet Protocol（TCP/IP）或 Internet Protocol Version 4（TCP/IPv4）的项目（如果有两个版本（另一个是版本 6），然后单击属性按钮。

1.  您可以通过使用自动设置或特定 IP 地址来确认您的网络设置（如果是这样，请记下此地址和其余细节，因为您可能希望在以后的某个时间点恢复设置）。

要检查 Linux 上的网络设置，请执行以下步骤：

1.  打开网络设置对话框，然后选择配置接口。请参考以下屏幕截图：

![](img/73215059-cead-4ec9-ab5b-67042c8ae2ea.png)Linux 网络设置对话框

1.  如果有任何手动设置，请确保记下它们，以便以后可以恢复它们。

要检查 macOS X 上的网络设置，请执行以下步骤：

1.  打开系统偏好设置，然后单击网络。然后您可以确认 IP 地址是否自动分配（使用 DHCP）。

1.  确保如果有任何手动设置，您记下它们，以便以后可以恢复它们。请参考以下屏幕截图：

![](img/3f602406-f5b7-4a0e-a5c3-f1a6e094dd04.png)OS X 网络设置对话框

如果您只需要访问或控制树莓派而无需互联网连接，请参考*直接网络链接*部分中的*更多信息*部分。

# 如何做...

首先，我们需要在我们的网络设备上启用 ICS。在这种情况下，我们将通过以太网连接将可用于无线网络连接的互联网共享到树莓派。

对于 Windows，执行以下步骤：

1.  返回到网络适配器列表，右键单击链接

到互联网（在本例中，WiFi 或无线网络连接设备），然后单击属性：

![](img/26960d5e-bea7-4a84-add9-d377441017d6.png)查找有线网络连接

1.  在窗口顶部，选择第二个选项卡（在 Windows XP 中称为高级；在 Windows 7 和 Windows 10 中称为共享），如下屏幕截图所示：

![](img/d786b0e6-9517-4a54-b1f5-da1fc3533f2b.png)选择 TCP/IP 属性并记下分配的 IP 地址

1.  在 Internet Connection Sharing 部分，选中允许其他网络用户通过此计算机的 Internet 连接连接（如果有，请使用下拉框选择家庭网络连接：选项为以太网或本地连接）。单击确定并确认以前是否为本地连接设置了固定 IP 地址。

对于 macOS X，执行以下步骤启用 ICS：

1.  单击系统偏好设置，然后单击共享。

1.  单击 Internet 共享，然后选择要共享互联网的连接（在本例中，将是 Wi-Fi AirPort）。然后选择我们将连接树莓派的连接（在本例中，是以太网）。

对于 Linux 启用 ICS，执行以下步骤：

1.  从系统菜单中，单击首选项，然后单击网络连接。选择要共享的连接（在本例中为无线），然后单击编辑或配置。在 IPv4 设置选项卡中，将方法选项更改为共享到其他计算机。

网络适配器的 IP 地址将是用于树莓派的**网关 IP**地址，并将分配一个在相同范围内的 IP 地址（它们将匹配，除了最后一个数字）。例如，如果计算机的有线连接现在为`192.168.137.1`，则树莓派的网关 IP 将为`192.168.137.1`，其自己的 IP 地址可能设置为`192.168.137.10`。

幸运的是，由于操作系统的更新，Raspbian 现在将自动为自己分配一个合适的 IP 地址以加入网络，并适当设置网关。但是，除非我们将屏幕连接到树莓派或扫描我们的网络上的设备，否则我们不知道树莓派给自己分配了什么 IP 地址。

幸运的是（如在*网络和通过 LAN 连接将树莓派连接到互联网*食谱中的*还有更多...*部分中提到的），苹果的**Bonjour**软件将自动确保网络上的主机名正确注册。如前所述，如果您使用的是 Mac OS X，则已经运行了 Bonjour。在 Windows 上，您可以安装 iTunes，或者您可以单独安装它（可从[`support.apple.com/kb/DL999`](https://support.apple.com/kb/DL999)获取）。默认情况下，可以使用主机名**raspberrypi**。

现在我们准备测试新连接，如下所示：

1.  将网络电缆连接到树莓派和计算机的网络端口，然后启动树莓派，确保您已重新插入 SD 卡（如果之前已将其拔出）。要重新启动树莓派，如果您在那里编辑了文件，请使用`sudo reboot`来重新启动它。

1.  等待一两分钟，让树莓派完全启动。现在我们可以测试连接了。

1.  从连接的笔记本电脑或计算机上，通过 ping 树莓派的主机名来测试连接，如下命令所示（在 Linux 或 OS X 上，添加`-c 4`以限制为四条消息，或按 Ctrl + C 退出）：

```py
ping raspberrypi  
```

希望您会发现您已经建立了一个工作连接，并从连接的计算机接收到了回复

树莓派。

如果您连接了键盘和屏幕到树莓派，您可以执行

以下步骤：

1.  您可以通过以下方式在树莓派终端上对计算机进行返回 ping（例如，`192.168.137.1`）：

```py
sudo ping 192.168.137.1 -c 4  
```

1.  您可以通过使用`ping`连接到一个知名网站来测试与互联网的连接，假设您不是通过代理服务器访问互联网：

```py
sudo ping www.raspberrypi.org -c 4  
```

如果一切顺利，您将可以通过计算机完全访问互联网，从而可以浏览网页以及更新和安装新软件。

如果连接失败，请执行以下步骤：

1.  重复这个过程，确保前三组数字与树莓派和网络适配器的 IP 地址匹配。

1.  您还可以通过以下命令检查树莓派启动时是否设置了正确的 IP 地址：

```py
hostname -I  
```

1.  检查防火墙设置，确保防火墙不会阻止内部网络连接。

# 它是如何工作的...

当我们在主要计算机上启用 ICS 时，操作系统将自动为计算机分配一个新的 IP 地址。连接并启动后，树莓派将自动设置为兼容的 IP 地址，并将主要计算机的 IP 地址用作 Internet 网关。

通过使用 Apple Bonjour，我们能够使用`raspberrypi`主机名从连接的计算机连接到树莓派。

最后，我们检查计算机是否可以通过直接网络链路与树莓派进行通信，反之亦然，并且通过互联网进行通信。

# 还有更多...

如果您不需要在树莓派上使用互联网，或者您的计算机只有一个网络适配器，您仍然可以通过直接网络链路将计算机连接在一起。请参考以下图表：

![](img/a60939c0-0e9e-4de1-9b1e-d085ca1f4b6c.png)只使用网络电缆、标准镜像 SD 卡和电源连接和使用树莓派

# 直接网络链路

要使两台计算机之间的网络链接正常工作，它们需要使用相同的地址范围。可允许的地址范围由子网掩码确定（例如，`255.255.0.0`或`255.255.255.0`表示所有 IP 地址应该相同，除了最后两个，或者只是 IP 地址中的最后一个数字；否则，它们将被过滤）。

要在不启用 ICS 的情况下使用直接链接，请检查您正在使用的适配器的 IP 设置

要连接到的 IP 地址，并确定它是自动分配还是固定的

特定的 IP 地址。

直接连接到另一台计算机的大多数 PC 将在`169.254.X.X`范围内分配 IP 地址（子网掩码为`255.255.0.0`）。但是，我们必须确保网络适配器设置为自动获取 IP 地址。

为了让 Raspberry Pi 能够通过直接链接进行通信，它需要在相同的地址范围`169.254.X.X`内具有 IP 地址。如前所述，Raspberry Pi 将自动为自己分配一个合适的 IP 地址并连接到网络。

因此，假设我们有 Apple Bonjour（前面提到过），我们只需要知道分配给 Raspberry Pi 的主机名（`raspberrypi`）。

# 另请参阅

如果您没有键盘或屏幕连接到 Raspberry Pi，您可以使用此网络链接远程访问 Raspberry Pi，就像在普通网络上一样（只需使用您为连接设置的新 IP 地址）。参考*通过 VNC 远程连接到 Raspberry Pi*和*通过 SSH（和 X11 转发）远程连接到 Raspberry Pi*。

我的网站[`pihw.wordpress.com/guides/direct-network-connection`](https://pihw.wordpress.com/guides/direct-network-connection)上提供了大量额外信息，包括额外的故障排除提示和连接到 Raspberry Pi 的其他几种方式，而无需专用屏幕和键盘。

# 通过 USB Wi-Fi dongle 进行网络连接和连接 Raspberry Pi 到互联网

通过在 Raspberry Pi 的 USB 端口添加**USB Wi-Fi dongle**，即使没有内置 Wi-Fi 的型号也可以连接并使用 Wi-Fi 网络。

# 准备工作

您需要获取一个合适的 USB Wi-Fi dongle，并且在某些情况下，您可能需要一个有源的 USB 集线器（这将取决于您拥有的 Raspberry Pi 的硬件版本和您的电源供应的质量）。USB Wi-Fi dongles 的一般适用性将取决于内部使用的芯片组以及可用的 Linux 支持水平。您可能会发现一些 USB Wi-Fi dongles 可以在不安装额外驱动程序的情况下工作（在这种情况下，您可以跳转到为无线网络配置它）。

支持的 Wi-Fi 适配器列表可在[`elinux.org/RPi_USB_Wi-Fi_Adapters`](http://elinux.org/RPi_USB_Wi-Fi_Adapters)找到。

您需要确保您的 Wi-Fi 适配器也与您打算连接的网络兼容；例如，它支持相同类型的信号**802.11bgn**和加密**WEP**、**WPA**和**WPA2**（尽管大多数网络都是向后兼容的）。

您还需要了解您网络的以下详细信息：

+   服务集标识符（SSID）：这是您的 Wi-Fi 网络的名称，如果您使用以下命令，应该是可见的：

```py
sudo iwlist scan | grep SSID  
```

+   加密类型和密钥：此值将为 None、WEP、WPA 或 WPA2，密钥将是您通常在连接手机或笔记本电脑到无线网络时输入的代码（有时它会被打印在路由器上）。

您将需要一个工作的互联网连接（即有线以太网）来下载所需的驱动程序。否则，您可以找到所需的固件文件（它们将是`.deb`文件）并将它们复制到 Raspberry Pi（即通过 USB 闪存驱动器；如果您在桌面模式下运行，驱动器应该会自动挂载）。将文件复制到适当的位置并安装它，使用以下命令：

```py
sudo apt-get install firmware_file.deb  
```

# 如何做...

此任务有两个阶段：首先，我们识别并安装 Wi-Fi 适配器的固件，然后我们需要为无线网络配置它。

我们将尝试识别您的 Wi-Fi 适配器的芯片组（处理连接的部分）；这可能与设备的实际制造商不匹配。

可以使用此命令找到支持的固件的近似列表：

```py
sudo apt-cache search wireless firmware  
```

这将产生类似以下输出的结果（忽略任何没有`firmware`在软件包标题中的结果）：

```py
atmel-firmware - Firmware for Atmel at76c50x wireless networking chips.
firmware-atheros - Binary firmware for Atheros wireless cards
firmware-brcm80211 - Binary firmware for Broadcom 802.11 wireless cards
firmware-ipw2x00 - Binary firmware for Intel Pro Wireless 2100, 2200 and 2915
firmware-iwlwifi - Binary firmware for Intel PRO/Wireless 3945 and 802.11n cards
firmware-libertas - Binary firmware for Marvell Libertas 8xxx wireless cards
firmware-ralink - Binary firmware for Ralink wireless cards
firmware-realtek - Binary firmware for Realtek wired and wireless network adapters
libertas-firmware - Firmware for Marvell's libertas wireless chip series (dummy package)
zd1211-firmware - Firmware images for the zd1211rw wireless driver  
```

要找出无线适配器的芯片组，将 Wi-Fi 适配器插入树莓派，然后从终端运行以下命令：

```py
dmesg | grep 'Product:|Manufacturer:'
```

这个命令将两个命令合并成一个。首先，`dmesg`显示内核的消息缓冲区（这是自动开机以来发生的系统事件的内部记录，比如检测到的 USB 设备）。您可以尝试单独使用该命令来观察完整的输出。

`|`（管道）将输出发送到`grep`命令；`grep 'Product:|Manufacturer'`检查它，并且只返回包含`Product`或`Manufacturer`的行（因此我们应该得到列为`Product`和`Manufacturer`的任何项目的摘要）。如果您找不到任何内容，或者想要查看所有 USB 设备，请尝试`grep 'usb'`命令。

这应该返回类似于以下输出——在这种情况下，我有一个`ZyXEL`设备，它有一个`ZyDAS`芯片组（快速的谷歌搜索显示`zd1211-firmware`适用于`ZyDAS`设备）：

```py
[    1.893367] usb usb1: Product: DWC OTG Controller
[    1.900217] usb usb1: Manufacturer: Linux 3.6.11+ dwc_otg_hcd
[    3.348259] usb 1-1.2: Product: ZyXEL G-202
[    3.355062] usb 1-1.2: Manufacturer: ZyDAS  
```

一旦您确定了您的设备和正确的固件，您可以像安装其他通过`apt-get`可用的软件包一样安装它（其中`zd1211-firmware`可以替换为您需要的固件）。如下所示：

```py
sudo apt-get install zd1211-firmware  
```

拔出并重新插入 USB Wi-Fi dongle，以便它被检测到并加载驱动程序。我们现在可以使用`ifconfig`测试新适配器是否正确安装。输出如下所示：

```py
wlan0     IEEE 802.11bg  ESSID:off/any
 Mode:Managed  Access Point: Not-Associated   Tx-Power=20 dBm
 Retry  long limit:7   RTS thr:off   Fragment thr:off
 Power Management:off  
```

该命令将显示系统上存在的网络适配器。对于 Wi-Fi，通常是`wlan0`或`wlan1`，如果您安装了多个，则会有更多。如果没有，请仔细检查所选的固件，或者尝试替代方案，或者在网站上查看故障排除提示。

一旦我们安装了 Wi-Fi 适配器的固件，我们就需要为我们希望连接的网络进行配置。我们可以使用 GUI，就像前面的示例中所示，或者我们可以通过终端手动配置，就像以下步骤中所示：

1.  我们需要将无线适配器添加到网络接口列表中，该列表设置在`/etc/network/interfaces`中，如下所示：

```py
sudo nano -c /etc/network/interfaces   
```

使用以前的`wlan#`值替换`wlan0`，如果需要，添加以下命令：

```py
allow-hotplug wlan0
iface wlan0 inet manual
wpa-conf /etc/wpa_supplicant/wpa_supplicant.conf  
```

更改后，按*Ctrl* + *X*，*Y*和*Enter*保存并退出。

1.  我们现在将我们网络的 Wi-Fi 网络设置存储在`wpa_supplicant.conf`文件中（如果您的网络不使用`wpa`加密，不用担心；这只是文件的默认名称）：

```py
sudo nano -c /etc/wpa_supplicant/wpa_supplicant.conf  
```

它应该包括以下内容：

```py
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev 
update_config=1 
country=GB 
```

网络设置可以写入此文件，如下所示（即，如果 SSID 设置为`theSSID`）：

+   +   如果不使用加密，请使用此代码：

```py
network={ 
  ssid="theSSID" 
  key_mgmt=NONE 
} 
```

+   +   使用`WEP`加密（即，如果`WEP`密钥设置为`theWEPkey`），使用以下代码：

```py
network={ 
  ssid="theSSID" 
  key_mgmt=NONE 
  wep_key0="theWEPkey" 
} 
```

+   +   对于`WPA`或`WPA2`加密（即，如果`WPA`密钥设置为`theWPAkey`），使用以下代码：

```py
network={ 
  ssid="theSSID" 
  key_mgmt=WPA-PSK 
  psk="theWPAkey"     
} 
```

1.  您可以使用以下命令启用适配器（如果需要，请再次替换`wlan0`）：

```py
sudo ifup wlan0

```

使用以下命令列出无线网络连接：

```py
iwconfig

```

您应该看到您的无线网络连接，并列出您的 SSID，如下所示：

```py
wlan0     IEEE 802.11bg  ESSID:"theSSID"
 Mode:Managed  Frequency:2.442 GHz  Access Point: 
       00:24:BB:FF:FF:FF
 Bit Rate=48 Mb/s   Tx-Power=20 dBm
 Retry  long limit:7   RTS thr:off   Fragment thr:off
 Power Management:off
 Link Quality=32/100  Signal level=32/100
 Rx invalid nwid:0  Rx invalid crypt:0  Rx invalid frag:0
 Tx excessive retries:0  Invalid misc:15   Missed beacon:0  
```

如果不是，请调整您的设置，并使用`sudo ifdown wlan0`关闭网络接口，然后使用`sudo ifup wlan0`打开它。这将确认您已成功连接到您的 Wi-Fi 网络。

1.  最后，我们需要检查我们是否可以访问互联网。在这里，我们假设网络已自动配置为 DHCP，并且不使用代理服务器。如果不是，请参考*通过代理服务器连接到互联网*的示例。

拔掉有线网络电缆（如果仍然连接），然后查看是否可以 ping 通树莓派网站，如下所示：

```py
**sudo ping** www.raspberrypi.org  
```

如果您想快速知道树莓派当前使用的 IP 地址，可以使用`hostname -I`，或者要找出哪个适配器连接到哪个 IP 地址，可以使用`ifconfig`。

# 还有更多...

Raspberry Pi 的 A 型版本没有内置的网络端口，因此为了获得网络连接，必须添加 USB 网络适配器（可以是 Wi-Fi dongle，如前一节所述，也可以是 LAN-to-USB 适配器，如下一节所述）。

# 使用 USB 有线网络适配器

就像 USB Wi-Fi 一样，适配器的支持取决于所使用的芯片组和可用的驱动程序。除非设备配备了 Linux 驱动程序，否则您可能需要在互联网上搜索以获取适用的 Debian Linux 驱动程序。

如果找到合适的`.deb`文件，可以使用以下命令进行安装：

```py
sudo apt-get install firmware_file.deb  
```

还可以使用`ifconfig`进行检查，因为一些设备将自动受支持，显示为`eth1`（或者在 A 型上为`eth0`），并且可以立即使用。

# 通过代理服务器连接到互联网

一些网络，例如工作场所或学校内的网络，通常要求您通过代理服务器连接到互联网。

# 准备工作

您需要代理服务器的地址，包括用户名和密码（如果需要）。

您应该确认树莓派已连接到网络，并且可以访问代理服务器。

使用`ping`命令进行检查，如下所示：

```py
ping proxy.address.com -c 4  
```

如果失败（没有响应），您需要确保继续之前网络设置正确。

# 如何操作...

1.  使用`nano`创建一个新文件，如下所示（如果文件中已经有一些内容，可以在末尾添加代码）：

```py
sudo nano -c ~/.bash_profile
```

1.  要允许通过代理服务器进行基本的网页浏览，例如**Midori**，您可以使用以下脚本：

```py
function proxyenable { 
# Define proxy settings 
PROXY_ADDR="proxy.address.com:port" 
# Login name (leave blank if not required): 
LOGIN_USER="login_name" 
# Login Password (leave blank to prompt): 
LOGIN_PWD= 
#If login specified - check for password 
if [[ -z $LOGIN_USER ]]; then 
  #No login for proxy 
  PROXY_FULL=$PROXY_ADDR 
else 
  #Login needed for proxy Prompt for password -s option hides input 
  if [[ -z $LOGIN_PWD ]]; then 
    read -s -p "Provide proxy password (then Enter):" LOGIN_PWD 
    echo 
  fi 
  PROXY_FULL=$LOGIN_USER:$LOGIN_PWD@$PROXY_ADDR 
fi 
#Web Proxy Enable: http_proxy or HTTP_PROXY environment variables 
export http_proxy="http://$PROXY_FULL/" 
export HTTP_PROXY=$http_proxy 
export https_proxy="https://$PROXY_FULL/" 
export HTTPS_PROXY=$https_proxy 
export ftp_proxy="ftp://$PROXY_FULL/" 
export FTP_PROXY=$ftp_proxy 
#Set proxy for apt-get 
sudo cat <<EOF | sudo tee /etc/apt/apt.conf.d/80proxy > /dev/null 
Acquire::http::proxy "http://$PROXY_FULL/"; 
Acquire::ftp::proxy "ftp://$PROXY_FULL/"; 
Acquire::https::proxy "https://$PROXY_FULL/"; 
EOF 
#Remove info no longer needed from environment 
unset LOGIN_USER LOGIN_PWD PROXY_ADDR PROXY_FULL 
echo Proxy Enabled 
} 

function proxydisable { 
#Disable proxy values, apt-get and git settings 
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY 
unset ftp_proxy FTP_PROXY 
sudo rm /etc/apt/apt.conf.d/80proxy 
echo Proxy Disabled 
} 
```

1.  完成后，按*Ctrl* + *X*，*Y*和*Enter*保存并退出。

脚本被添加到用户自己的`.bash_profile`文件中，在特定用户登录时运行。这将确保代理设置被分别保存给每个用户。如果您希望所有用户使用相同的设置，可以将代码添加到`/etc/rc.local`中（此文件必须在末尾有`exit 0`）。

# 工作原理...

许多使用互联网的程序在连接之前会检查`http_proxy`或`HTTP_PROXY`环境变量。如果存在，它们将使用代理设置进行连接。一些程序也可能使用`HTTPS`和`FTP`协议，因此我们也可以在这里为它们设置代理设置。

如果代理服务器需要用户名，则会提示输入密码。通常不建议在脚本中存储密码，除非您确信没有其他人能够访问您的设备（无论是物理上还是通过互联网）。

最后一部分允许使用`sudo`命令执行的任何程序在扮演超级用户时使用代理环境变量（大多数程序首先尝试使用普通权限访问网络，即使作为超级用户运行，所以并不总是需要）。

# 还有更多...

我们还需要允许某些程序使用代理设置，这些程序在访问网络时使用超级用户权限（这取决于程序；大多数不需要这样做）。我们需要通过以下步骤将命令添加到存储在`/etc/sudoers.d/`中的文件中：

1.  使用以下命令打开一个新的`sudoer`文件：

```py
sudo visudo -f /etc/sudoers.d/proxy  
```

1.  在文件中输入以下文本（一行）：

```py
Defaults env_keep += "http_proxy HTTP_PROXY https_proxy HTTPS_PROXY ftp_proxy FTP_PROXY"  
```

1.  完成后，按*Ctrl* + *X*，*Y*和*Enter*保存并退出；不要更改`proxy.tmp`文件名（这对于`visudo`是正常的；完成后它会将其更改为 proxy）。

1.  如果提示“现在怎么办？”，则命令中存在错误。按*X*退出而不保存并重新输入命令。

1.  重新启动后（使用`sudo reboot`），您将能够使用以下命令分别启用和禁用代理：

```py
proxyenable
proxydisable  
```

在这里使用`visudo`很重要，因为它确保了为`sudoers`目录正确创建文件的权限（只能由`root`用户读取）。

# 通过 VNC 远程连接到树莓派网络

通常，最好远程连接和控制树莓派跨网络，例如，使用笔记本电脑或台式电脑作为屏幕和键盘，或者当树莓派连接到其他地方时，也许甚至连接到一些需要靠近的硬件。

VNC 只是远程连接到树莓派的一种方式。它将创建一个新的桌面会话，可以远程控制和访问。这里的 VNC 会话与树莓派显示上可能活动的会话是分开的。

# 准备工作

确保您的树莓派已经启动并连接到互联网。我们将使用互联网连接来使用`apt-get`安装程序。这是一个允许我们直接从官方存储库中查找和安装应用程序的程序。

# 如何做...

1.  首先，我们需要使用以下命令在树莓派上安装 TightVNC 服务器。建议先运行`update`命令以获取要安装的软件包的最新版本，如下所示：

```py
sudo apt-get update
sudo apt-get install tightvncserver  
```

1.  接受提示进行安装，并等待直到完成。要启动会话，请使用以下命令：

```py
vncserver :1  
```

1.  第一次运行时，它将要求您输入一个密码（不超过八个字符）以访问桌面（您在从计算机连接时将使用此密码）。

以下消息应该确认已启动新的桌面会话：

```py
New 'X' desktop is raspberrypi:1  
```

如果您还不知道树莓派的 IP 地址，请使用`hostname -I`并记下它。

接下来，我们需要运行 VNC 客户端。**VNC Viewer**是一个合适的程序，可以在[`www.realvnc.com/`](http://www.realvnc.com/)上找到，并且应该可以在 Windows、Linux 和 OS X 上运行。

运行 VNC Viewer 时，将提示您输入服务器地址和加密类型。使用您的树莓派的 IP 地址与`:1`。也就是说，对于 IP 地址`192.168.1.69`，使用`192.168.1.69:1`地址。

您可以将加密类型保留为关闭或自动。

根据您的网络，您可以使用主机名；默认值为`raspberrypi`，即`raspberrypi:1`。

可能会有一个关于以前未连接到计算机或没有加密的警告。如果您正在使用公共网络或在互联网上进行连接（以阻止其他人能够拦截您的数据），您应该启用加密。

# 还有更多...

您可以添加选项到命令行以指定显示的分辨率和颜色深度。分辨率和颜色深度越高（可以调整为每像素使用 8 位到 32 位以提供低或高颜色细节），通过网络链接传输的数据就越多。如果发现刷新速率有点慢，请尝试按照以下方式减少这些数字：

```py
vncserver :1 -geometry 1280x780 -depth 24  
```

要允许 VNC 服务器在打开时自动启动，您可以将`vncserver`命令添加到`.bash_profile`（这在每次树莓派启动时执行）。

使用`nano`编辑器如下（`-c`选项允许显示行号）：

```py
sudo nano -c ~/.bash_profile  
```

将以下行添加到文件的末尾：

```py
vncserver :1  
```

下次启动时，您应该能够使用 VNC 从远程连接到树莓派

另一台计算机。

# 通过 SSH（和 X11 转发）远程连接到树莓派

**安全外壳**（**SSH**）通常是进行远程连接的首选方法，因为它只允许终端连接，并且通常需要更少的资源。

SSH 的一个额外功能是能够将**X11**数据传输到运行在您的计算机上的**X Windows**服务器。这允许您启动通常在树莓派桌面上运行的程序，并且它们将出现在本地计算机上的自己的窗口中，如下所示：

![](img/13e18668-dd51-4086-8cc7-2c9e57cb722c.png)在本地显示上的 X11 转发

X11 转发可用于在 Windows 计算机上显示在树莓派上运行的应用程序。

# 准备工作

如果您正在运行最新版本的 Raspbian，则 SSH 和 X11 转发将默认启用（否则，请仔细检查*它是如何工作...*部分中解释的设置）。

# 如何做...

Linux 和 OS X 都内置支持 X11 转发，但如果你使用 Windows，你需要在计算机上安装和运行 X Windows 服务器。

从**Xming**网站（[`sourceforge.net/projects/xming/`](http://sourceforge.net/projects/xming/)）下载并运行`xming`。

安装`xming`，按照安装步骤进行安装，包括安装**PuTTY**（如果您还没有）。您也可以从[`www.putty.org/`](http://www.putty.org/)单独下载 PuTTY。

接下来，我们需要确保我们连接时使用的 SSH 程序启用了 X11。

对于 Windows，我们将使用 PuTTY 连接到树莓派。

在 PuTTY 配置对话框中，导航到连接 | SSH | X11，并选中启用 X11 转发的复选框。如果将 X 显示位置选项留空，它将假定默认的`Server 0:0`如下（您可以通过将鼠标移到运行时系统托盘中的 Xming 图标上来确认服务器号）：

![](img/97414251-ec25-4472-8a03-afe01c95f283.jpg)在 PuTTY 配置中启用 X11 转发

在会话设置中输入树莓派的 IP 地址（您还可以在这里使用树莓派的主机名；默认主机名是`raspberrypi`）。

使用适当的名称保存设置，`RaspberryPi`，然后单击打开以连接到您的树莓派。

您可能会看到一个警告消息弹出，指出您以前没有连接到计算机（这样可以在继续之前检查是否一切正常）：

![](img/3029d1b0-93e5-48fc-840e-fc238508e837.jpg)使用 PuTTY 打开到树莓派的 SSH 连接

对于 OS X 或 Linux，单击终端以打开到树莓派的连接。

要使用默认的`pi`用户名连接，IP 地址为`192.168.1.69`，使用以下命令；`-X`选项启用 X11 转发：

```py
ssh -X pi@192.168.1.69  
```

一切顺利的话，您应该会收到一个输入密码的提示（请记住`pi`用户的默认值是`raspberry`）。

确保 Xming 正在运行，方法是从计算机的开始菜单启动 Xming 程序。然后，在终端窗口中，输入通常在树莓派桌面内运行的程序，如`leafpad`或`scratch`。等一会儿，程序应该会出现在您的计算机桌面上（如果出现错误，您可能忘记启动 Xming，所以请运行它并重试）。

# 它是如何工作的...

X Windows 和 X11 是提供树莓派（以及许多其他基于 Linux 的计算机）显示和控制图形窗口作为桌面一部分的方法。

要使 X11 转发在网络连接上工作，我们需要在树莓派上同时启用 SSH 和 X11 转发。执行以下步骤：

1.  要打开（或关闭）SSH，您可以访问树莓派配置

在桌面的首选项菜单下的 SSH 中，单击接口选项卡，如下截图所示（大多数发行版通常默认启用 SSH，以帮助允许远程连接而无需显示器进行配置）：

![](img/72707656-f19f-49e9-9c76-0c1cf7b134a2.png)raspi-config 工具中的高级设置菜单

1.  确保在树莓派上启用了 X11 转发（大多数发行版现在默认已启用此功能）。

1.  使用以下命令使用`nano`：

```py
sudo nano /etc/ssh/sshd_config  
```

1.  在`/etc/ssh/sshd_config`文件中查找控制 X11 转发的行，并确保它说`yes`（之前没有`#`符号），如下所示：

```py
X11Forwarding yes  
```

1.  如果需要，按*Ctrl* + *X*，*Y*和*Enter*保存并重新启动（如果需要更改）如下：

```py
sudo reboot  
```

# 还有更多...

SSH 和 X11 转发是远程控制树莓派的便捷方式；我们将在以下部分探讨如何有效使用它的一些额外提示。

# 通过 X11 转发运行多个程序

如果您想运行**X 程序**，但仍然可以在同一终端控制台上运行其他内容，可以使用`&`将命令在后台运行，如下所示：

```py
leafpad &  
```

只需记住，您运行的程序越多，一切就会变得越慢。您可以通过输入`fg`切换到后台程序，并使用`bg`检查后台任务。

# 通过 X11 转发作为桌面运行

您甚至可以通过 X11 运行完整的桌面会话，尽管它并不特别用户友好，而且 VNC 会产生更好的结果。要实现这一点，您必须使用`lxsession`而不是`startx`（以您通常从终端启动桌面的方式）。

另一种选择是使用`lxpanel`，它提供了程序菜单栏，您可以从菜单中启动和运行程序，就像在桌面上一样。

# 通过 X11 转发运行 Pygame 和 Tkinter

在运行**Pygame**或**Tkinter**脚本时，您可能会遇到以下错误（或类似错误）：

```py
_tkinter.TclError: couldn't connect to display "localhost:10.0"  
```

在这种情况下，使用以下命令来修复错误：

```py
sudo cp ~/.Xauthority ~root/ 
```

# 使用 SMB 共享树莓派的主文件夹

当您将树莓派连接到网络时，您可以通过设置文件共享来访问主文件夹；这样可以更轻松地传输文件，并提供了一种快速简便的备份数据的方法。**服务器消息块**（**SMB**）是一种与 Windows 文件共享、OS X 和 Linux 兼容的协议。

# 准备工作

确保您的树莓派已接通电源并连接到互联网。

您还需要另一台在同一本地网络上的计算机来测试新的共享。

# 如何做...

首先，我们需要安装`samba`，这是一款处理与 Windows 共享方法兼容的文件夹共享的软件：

1.  确保您使用以下命令来获取可用软件包的最新列表：

```py
sudo apt-get update
sudo apt-get install samba  
```

安装将需要大约 20MB 的空间，并需要几分钟的时间。

1.  安装完成后，我们可以按照以下方式复制配置文件，以便在需要时恢复默认设置：

```py
sudo cp /etc/samba/smb.conf /etc/samba/smb.conf.backup
sudo nano /etc/samba/smb.conf  
```

向下滚动并找到名为`Authentication`的部分；将`# security = user`行更改为`security = user`。

如文件中所述，此设置确保您必须输入用户名和密码才能访问树莓派的文件（这对于共享网络非常重要）。

找到名为`Share Definitions`和`[homes]`的部分，并将`read only = yes`行更改为`read only = no`。

这将允许我们查看并向共享的主文件夹写入文件。完成后，按*Ctrl* + *X*，*Y*和*Enter*保存并退出。

如果您已将默认用户从`pi`更改为其他内容，请在以下说明中进行替换。

1.  现在，我们可以添加`pi`（默认用户）来使用`samba`：

```py
sudo pdbedit -a -u pi
```

1.  现在，输入密码（您可以使用与登录相同的密码或选择不同的密码，但避免使用默认的树莓密码，这对某人来说将非常容易猜到）。重新启动`samba`以使用新的配置文件，如下所示：

```py
sudo /etc/init.d/samba restart
[ ok ] Stopping Samba daemons: nmbd smbd.
[ ok ] Starting Samba daemons: nmbd smbd.  
```

1.  要进行测试，您需要知道树莓派的`hostname`（默认`hostname`为`raspberrypi`）或其 IP 地址。您可以使用以下命令找到这两者。

```py
hostname
```

1.  对于 IP 地址，添加`-I`：

```py
hostname -I  
```

在网络上的另一台计算机上，在资源管理器路径中输入`\raspberrypipi`地址。

根据您的网络，计算机应该能够在网络上找到树莓派，并提示输入用户名和密码。如果它无法使用`hostname`找到共享，您可以直接使用 IP 地址，其中`192.168.1.69`应更改为匹配 IP 地址`\192.168.1.69pi`。

# 保持树莓派最新

树莓派使用的 Linux 镜像经常更新，以包括对系统的增强、修复和改进，以及对新硬件的支持或对最新板的更改。您安装的许多软件包也可以进行更新。

如果您打算在另一个树莓派板上使用相同的系统镜像（特别是较新的板），这一点尤为重要，因为旧镜像将缺乏对任何布线更改或替代 RAM 芯片的支持。新固件应该可以在较旧的树莓派板上工作，但是旧固件可能与最新的硬件不兼容。

幸运的是，每次有新版本发布时，您无需重新刷写 SD 卡，因为可以进行更新。

# 准备就绪

您需要连接到互联网才能更新系统。始终建议首先备份您的镜像（至少复制您的重要文件）。

您可以使用`uname -a`命令检查当前固件的版本，如下所示：

```py
Linux raspberrypi 4.4.9-v7+ #884 SMP Fri May 6 17:28:59 BST 2016 armv7l GNU/Linux  
```

可以使用`/opt/vc/bin/vcgencmd version`命令来检查 GPU 固件，如下所示：

```py
 May  6 2016 13:53:23
Copyright (c) 2012 Broadcom
version 0cc642d53eab041e67c8c373d989fef5847448f8 (clean) (release)
```

如果您在较新的板上使用较旧的固件（2012 年 11 月之前），这一点很重要，因为最初的 B 型板只有 254MB RAM。升级可以使固件利用额外的内存（如果可用）。

`free -h`命令将详细说明主处理器可用的 RAM（总 RAM 在 GPU 和 ARM 核心之间分配），并将给出以下输出：

```py
                 total       used       free     shared    buffers     cached
    Mem:          925M       224M       701M       7.1M        14M       123M
    -/+ buffers/cache:        86M       839M
    Swap:          99M         0B        99M

```

然后可以在重新启动后重新检查前面的输出，以确认它们已经更新（尽管它们可能已经是最新的）。

# 如何做...

1.  在运行任何升级或安装任何软件包之前，值得确保您拥有存储库中最新的软件包列表。`update`命令获取可用软件和版本的最新列表：

```py
sudo apt-get update  
```

1.  如果您只想获取当前软件包的升级，`upgrade`将使它们全部保持最新状态：

```py
sudo apt-get upgrade
```

1.  为确保您运行的是最新版本的 Raspbian，可以运行`dist-upgrade`（请注意：这可能需要一小时或更长时间，具体取决于需要升级的数量）。这将执行`upgrade`将执行的所有更新，但还将删除多余的软件包并进行清理：

```py
sudo apt-get dist-upgrade  
```

这两种方法都将升级软件，包括在启动和启动时使用的固件（`bootcode.bin`和`start.elf`）。

1.  要更新固件，可以使用以下命令：

```py
sudo rpi-update  
```

# 还有更多...

您经常会发现您想要对设置进行干净的安装，但是这意味着您将不得不从头开始安装所有内容。为了避免这种情况，我开发了 Pi-Kitchen 项目（[`github.com/PiHw/Pi-Kitchen`](https://github.com/PiHw/Pi-Kitchen)），基于*Kevin Hill*的基础工作。这旨在提供一个灵活的平台，用于创建可以自动部署到 SD 卡的定制设置：

![](img/9f995257-ae35-4309-b96c-8545c63a6cdb.png)Pi Kitchen 允许在启动之前配置树莓派

Pi-Kitchen 允许配置各种口味，可以从 NOOBS 菜单中选择。每种口味都包括一系列食谱，每个食谱都提供最终操作系统的特定功能或特性。食谱可以从为 Wi-Fi 设备设置自定义驱动程序，到在您的网络上映射共享驱动器，再到提供一个功能齐全的网络服务器，所有这些都组合在一起，以满足您的要求。

该项目处于测试阶段，作为概念验证开发，但是一旦您配置好一切，将完全工作的设置直接部署到 SD 卡上将非常有用。最终，该项目可以与 Kevin Hill 的 NOOBS 的高级版本**PINN Is Not NOOBS**（**PINN**）结合使用，旨在为高级用户提供额外功能，例如允许操作系统和配置存储在您的网络上或外部 USB 存储器上。
