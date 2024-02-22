# 第十五章：PyQt 树莓派

树莓派是过去十年中最成功和令人兴奋的计算机之一。这款由英国非营利组织于 2012 年推出的微型**高级 RISC 机器**（**ARM**）计算机，旨在教育孩子们计算机科学知识，已成为业余爱好者、改装者、开发人员和各类 IT 专业人士的普遍工具。由于 Python 和 PyQt 在其默认操作系统上得到了很好的支持，树莓派也是 PyQt 开发人员的绝佳工具。

在本章中，我们将在以下部分中查看在树莓派上使用 PyQt5 开发：

+   在树莓派上运行 PyQt5

+   使用 PyQt 控制**通用输入/输出**（**GPIO**）设备

+   使用 GPIO 设备控制 PyQt

# 技术要求

为了跟随本章的示例，您需要以下物品：

+   一台树莓派——最好是 3 型 B+或更新的型号

+   树莓派的电源供应、键盘、鼠标、显示器和网络连接

+   安装了 Raspbian 10 或更高版本的微型 SD 卡；您可以参考官方文档[`www.raspberrypi.org/documentation/installation/`](https://www.raspberrypi.org/documentation/installation/)上的说明来安装 Raspbian

在撰写本文时，Raspbian 10 尚未发布，尽管可以将 Raspbian 9 升级到测试版本。如果 Raspbian 10 不可用，您可以参考本书的附录 B，*将 Raspbian 9 升级到 Raspbian 10*，了解升级的说明。

为了编写基于 GPIO 的项目，您还需要一些电子元件来进行接口。这些零件通常可以在电子入门套件中找到，也可以从当地的电子供应商那里购买。

第一个项目将需要以下物品：

+   一个面包板

+   三个相同的电阻（阻值在 220 到 1000 欧姆之间）

+   一个三色 LED

+   四根母对公跳线

第二个项目将需要以下物品：

+   一个面包板

+   一个 DHT11 或 DHT22 温湿度传感器

+   一个按钮开关

+   一个电阻（值不重要）

+   三根母对公跳线

+   Adafruit DHT 传感器库，可使用以下命令从 PyPI 获取：

```py
$ sudo pip3 install Adafruit_DHT
```

您可以参考 GitHub 存储库[`github.com/adafruit/Adafruit_Python_DHT`](https://github.com/adafruit/Adafruit_Python_DHT)获取更多信息。

您可能还想从[`github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter15`](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter15)下载示例代码。

查看以下视频以查看代码运行情况：[`bit.ly/2M5xDSx`](http://bit.ly/2M5xDSx)

# 在树莓派上运行 PyQt5

树莓派能够运行许多不同的操作系统，因此安装 Python 和 PyQt 完全取决于您选择的操作系统。在本书中，我们将专注于树莓派的官方（也是最常用的）操作系统**Raspbian**。

Raspbian 基于 Debian GNU/Linux 的稳定版本，目前是 Debian 9（Stretch）。不幸的是，本书中的代码所需的 Python 和 PyQt5 版本对于这个 Debian 版本来说太旧了。如果在阅读本书时，Raspbian 10 尚未发布，请参考附录 B，*将 Raspbian 9 升级到 Raspbian 10*，了解如何将 Raspbian 9 升级到 Raspbian 10 的说明。

Raspbian 10 预装了 Python 3.7，但我们需要自己安装 PyQt5。请注意，您不能使用`pip`在树莓派上安装 PyQt5，因为所需的 Qt 二进制文件在 PyPI 上不适用于 ARM 平台（树莓派所基于的平台）。但是，PyQt5 的一个版本可以从 Raspbian 软件存储库中获取。这将*不*是 PyQt5 的最新版本，而是在 Debian 开发过程中选择的最稳定和兼容发布的版本。对于 Debian/Raspbian 10，这个版本是 PyQt 5.11。

要安装它，首先确保您的设备连接到互联网。然后，打开命令行终端并输入以下命令：

```py
$ sudo apt install python3-pyqt5
```

**高级打包工具**（**APT**）实用程序将下载并安装 PyQt5 及所有必要的依赖项。请注意，此命令仅为 Python 3 安装 PyQt5 的主要模块。某些模块，如`QtSQL`、`QtMultimedia`、`QtChart`和`QtWebEngineWidgets`，是单独打包的，需要使用额外的命令进行安装：

```py
$ sudo apt install python3-pyqt5.qtsql python3-pyqt5.qtmultimedia python3-pyqt5.qtchart python3-pyqt5.qtwebengine
```

有许多为 PyQt5 打包的可选库。要获取完整列表，可以使用`apt search`命令，如下所示：

```py
$ apt search pyqt5
```

APT 是在 Raspbian、Debian 和许多其他 Linux 发行版上安装、删除和更新软件的主要方式。虽然类似于`pip`，APT 用于整个操作系统。

# 在树莓派上编辑 Python

尽管您可以在自己的计算机上编辑 Python 并将其复制到树莓派上执行，但您可能会发现直接在设备上编辑代码更加方便。如果您喜欢的代码编辑器或**集成开发环境**（**IDE**）在 Linux 或 ARM 上不可用，不要担心；Raspbian 提供了几种替代方案：

+   **Thonny** Python IDE 预装了默认的 Raspbian 镜像，并且非常适合本章的示例

+   **IDLE**，Python 的默认编程环境也是预装的

+   **Geany**，一个适用于许多语言的通用编程文本编辑器，也是预装的

+   传统的代码编辑器，如**Vim**和**Emacs**，以及 Python IDE，如**Spyder**、**Ninja IDE**和**Eric**，可以使用添加/删除软件工具（在程序菜单的首选项下找到）或使用`apt`命令从软件包存储库安装

无论您选择哪种应用程序或方法，请确保将文件备份到另一台设备，因为树莓派的 SD 卡存储并不是最稳健的。

# 在树莓派上运行 PyQt5 应用程序

一旦 Python 和 PyQt5 安装在您的树莓派上，您应该能够运行本书中到目前为止我们编写的任何应用程序。基本上，树莓派是一台运行 GNU/Linux 的计算机，本书中的所有代码都与之兼容。考虑到这一点，您*可以*简单地将其用作运行 PyQt 应用程序的小型、节能计算机。

然而，树莓派有一些独特的特性，最显著的是其 GPIO 引脚。这些引脚使树莓派能够以一种非常简单和易于访问的方式与外部数字电路进行通信。Raspbian 预装了软件库，允许我们使用 Python 控制这些引脚。

为了充分利用这一特性提供给我们的独特平台，我们将在本章的其余部分中专注于使用 PyQt5 与树莓派的 GPIO 功能结合，创建 GUI 应用程序，以与现实世界的电路进行交互，这只有像树莓派这样的设备才能做到。

# 使用 PyQt 控制 GPIO 设备

对于我们的第一个项目，我们将学习如何可以从 PyQt 应用程序控制外部电路。您将连接一个多色 LED，并使用`QColorDialog`来控制其颜色。收集第一个项目中列出的组件，并让我们开始吧。

# 连接 LED 电路

让我们通过在面包板上连接电路的组件来开始这个项目。关闭树莓派并断开电源，然后将其放在面包板附近。

在连接电路到 GPIO 引脚之前，关闭树莓派并断开电源总是一个好主意。这将减少在连接错误的情况下破坏树莓派的风险，或者如果您意外触摸到组件引脚。

这个电路中的主要组件是三色 LED。尽管它们略有不同，但这个元件的最常见引脚布局如下：

![](img/1d1018be-3dc8-4a65-ad28-831c18015ade.png)

基本上，三色 LED 是将红色 LED、绿色 LED 和蓝色 LED 组合成一个包。它提供单独的输入引脚，以便分别向每种颜色发送电流，并提供一个共同的地引脚。通过向每个引脚输入不同的电压，我们可以混合红色、绿色和蓝色光，从而创建各种各样的颜色，就像我们在应用程序中混合这三种元素来创建 RGB 颜色一样。

将 LED 添加到面包板上，使得每个引脚都在面包板的不同行上。然后，连接其余的组件如下：

![](img/37d2d5a0-2e88-4956-8fbb-04b724336b0b.png)

如前图所示，我们正在进行以下连接：

+   LED 上的地针直接连接到树莓派左侧第三个外部引脚。

+   LED 上的红色引脚连接到一个电阻，然后连接到右侧的下一个引脚（即引脚 8）

+   LED 上的绿色引脚连接到另一个电阻，然后连接到右侧的下一个空闲引脚（即引脚 10）

+   LED 上的蓝色引脚连接到最后一个电阻，然后连接到 Pi 上右侧的下一个空闲引脚（引脚 12）

重要的是要仔细检查您的电路，并确保您已将电线连接到树莓派上的正确引脚。树莓派上并非所有的 GPIO 引脚都相同；其中一些是可编程的，而其他一些具有硬编码目的。您可以通过在终端中运行`pinout`命令来查看 Pi 上的引脚列表；您应该看到以下输出：

![](img/5f9f8eee-267d-4935-ab36-6c1c18bd542d.png)

前面的屏幕截图显示了引脚的布局，就好像您正面对着树莓派，USB 端口朝下。请注意，其中有几个引脚标有**GND**；这些始终是地引脚，因此您可以将电路的地连接到其中任何一个引脚。其他引脚标有**5V**或**3V3**；这些始终是 5 伏或 3.3 伏。其余带有 GPIO 标签的引脚是可编程引脚。您的电线应连接到引脚**8**（**GPIO14**）、**10**（**GPIO15**）和**12**（**GPIO18**）。

仔细检查您的电路连接，然后启动树莓派。是时候开始编码了！

# 编写驱动程序库

现在我们的电路已连接好，我们需要编写一些代码来控制它。为此，我们将在树莓派上使用`GPIO`库。从第四章中创建一个 PyQt 应用程序模板的副本，*使用 QMainWindow 构建应用程序*，并将其命名为`three_color_led_gui.py`。

我们将从导入`GPIO`库开始：

```py
from RPi import GPIO
```

我们首先要做的是创建一个 Python 类，作为我们电路的 API。我们将称之为`ThreeColorLed`，然后开始如下：

```py
class ThreeColorLed():
    """Represents a three color LED circuit"""

    def __init__(self, red, green, blue, pinmode=GPIO.BOARD, freq=50):
        GPIO.setmode(pinmode)
```

我们的`__init__()`方法接受五个参数：前三个参数是红色、绿色和蓝色 LED 连接的引脚号；第四个参数是用于解释引脚号的引脚模式；第五个参数是频率，我们稍后会讨论。首先，让我们谈谈引脚模式。

如果你查看`pinout`命令的输出，你会注意到在树莓派上用整数描述引脚有两种方法。第一种是根据板子上的位置，从 1 到 40。第二种是根据它的 GPIO 编号（即在引脚描述中跟在 GPIO 后面的数字）。`GPIO`库允许你使用任一种数字来指定引脚，但你必须通过向`GPIO.setmode()`函数传递两个常量中的一个来告诉它你要使用哪种方法。`GPIO.BOARD`指定你使用位置编号（如 1 到 40），而`GPIO.BCM`表示你要使用 GPIO 名称。正如你所看到的，我们默认在这里使用`BOARD`。

每当你编写一个以 GPIO 引脚号作为参数的类时，一定要允许用户指定引脚模式。这些数字本身没有引脚模式的上下文是没有意义的。

接下来，我们的`__init__()`方法需要设置输出引脚：

```py
        self.pins = {
            "red": red,
            "green": green,
            "blue": blue
            }
        for pin in self.pins.values():
            GPIO.setup(pin, GPIO.OUT)
```

GPIO 引脚可以设置为`IN`或`OUT`模式，取决于你是想从引脚状态读取还是向其写入。在这个项目中，我们将从软件发送信息到电路，所以我们需要将所有引脚设置为`OUT`模式。在将引脚号存储在`dict`对象中后，我们已经通过使用`GPIO.setup()`函数迭代它们并将它们设置为适当的模式。

设置好后，我们可以使用`GPIO.output()`函数告诉单个引脚是高电平还是低电平，如下所示：

```py
        # Turn all on and all off
        for pin in self.pins.values():
            GPIO.output(pin, GPIO.HIGH)
            GPIO.output(pin, GPIO.LOW)
```

这段代码简单地打开每个引脚，然后立即关闭（可能比你看到的更快）。我们可以使用这种方法来设置 LED 为几种简单的颜色；例如，我们可以通过将红色引脚设置为`HIGH`，其他引脚设置为`LOW`来使其变为红色，或者通过将蓝色和绿色引脚设置为`HIGH`，红色引脚设置为`LOW`来使其变为青色。当然，我们希望产生更多种颜色，但我们不能简单地通过完全打开或关闭引脚来做到这一点。我们需要一种方法来在每个引脚的电压之间平稳地变化，从最小值（0 伏）到最大值（5 伏）。

不幸的是，树莓派无法做到这一点。输出是数字的，而不是模拟的，因此它们只能完全开启或完全关闭。然而，我们可以通过使用一种称为**脉宽调制**（**PWM**）的技术来*模拟*变化的电压。

# PWM

在你家里找一个有相对灵敏灯泡的开关（LED 灯泡效果最好）。然后，尝试每秒钟打开和关闭一次。现在越来越快地按开关，直到房间里的灯几乎看起来是恒定的。你会注意到房间里的光似乎比你一直开着灯时要暗，即使灯泡只是完全开启或完全关闭。

PWM 的工作方式相同，只是在树莓派上，我们可以如此快速（当然是无声地）地打开和关闭电压，以至于在打开和关闭之间的切换看起来是无缝的。此外，通过在每个周期中调整引脚打开时间和关闭时间的比例，我们可以模拟在零电压和最大电压之间的变化电压。这个比例被称为**占空比**。

关于脉宽调制的概念和用法的更多信息可以在[`en.wikipedia.org/wiki/Pulse-width_modulation`](https://en.wikipedia.org/wiki/Pulse-width_modulation)找到。

要在我们的引脚上使用 PWM，我们首先要通过在每个引脚上创建一个`GPIO.PWM`对象来设置它们：

```py
        self.pwms = dict([
             (name, GPIO.PWM(pin, freq))
             for name, pin in self.pins.items()
            ])
```

在这种情况下，我们使用列表推导来生成另一个包含每个引脚名称和`PWM`对象的`dict`。通过传入引脚号和频率值来创建`PWM`对象。这个频率将是引脚切换开和关的速率。

一旦我们创建了我们的`PWM`对象，我们需要启动它们：

```py
        for pwm in self.pwms.values():
            pwm.start(0)
```

`PWM.start()`方法开始引脚的闪烁。传递给`start()`的参数表示占空比的百分比；这里，`0`表示引脚将在 0%的时间内打开（基本上是关闭）。值为 100 将使引脚始终完全打开，而介于两者之间的值表示引脚在每个周期内接收的打开时间的量。

# 设置颜色

现在我们的引脚已经配置为 PWM，我们需要创建一个方法，通过传入红色、绿色和蓝色值，使 LED 显示特定的颜色。大多数软件 RGB 颜色实现（包括`QColor`）将这些值指定为 8 位整数（0 到 255）。然而，我们的 PWM 值表示占空比，它表示为百分比（0 到 100）。

因此，由于我们需要多次将 0 到 255 范围内的数字转换为 0 到 100 范围内的数字，让我们从一个静态方法开始，该方法将执行这样的转换：

```py
    @staticmethod
    def convert(val):
        val = abs(val)
        val = val//2.55
        val %= 101
        return val
```

该方法确保我们将获得有效的占空比，而不管输入如何，都使用简单的算术运算：

+   首先，我们使用数字的绝对值来防止传递任何负值。

+   其次，我们将值除以 2.55，以找到它代表的 255 的百分比。

+   最后，我们对数字取 101 的模，这样百分比高于 100 的数字将循环并保持在范围内。

现在，让我们编写我们的`set_color()`方法，如下所示：

```py
    def set_color(self, red, green, blue):
        """Set color using RGB color values of 0-255"""
        self.pwms['red'].ChangeDutyCycle(self.convert(red))
        self.pwms['green'].ChangeDutyCycle(self.convert(green))
        self.pwms['blue'].ChangeDutyCycle(self.convert(blue))
```

`PWM.ChangeDutyCycle()`方法接受 0 到 100 的值，并相应地调整引脚的占空比。在这个方法中，我们只是将我们的输入 RGB 值转换为适当的比例，并将它们传递给相应的 PWM 对象。

# 清理

我们需要添加到我们的类中的最后一个方法是清理方法。树莓派上的 GPIO 引脚可以被视为一个状态机，其中每个引脚都有高状态或低状态（即打开或关闭）。当我们在程序中设置这些引脚时，这些引脚的状态将在程序退出后保持设置。

请注意，如果我们连接了不同的电路到我们的 Pi，这可能会导致问题；在连接电路时，如果在错误的时刻将引脚设置为`HIGH`，可能会烧坏一些组件。因此，我们希望在退出程序时将所有东西关闭。

这可以使用`GPIO.cleanup()`函数完成：

```py
    def cleanup(self):
        GPIO.cleanup()
```

通过将这个方法添加到我们的 LED 驱动程序类中，我们可以在每次使用后轻松清理 Pi 的状态。

# 创建 PyQt GUI

现在我们已经处理了 GPIO 方面，让我们创建我们的 PyQt GUI。在`MainWindow.__init__()`中，添加以下代码：

```py
        self.tcl = ThreeColorLed(8, 10, 12)
```

在这里，我们使用连接到面包板的引脚号创建了一个`ThreeColorLed`实例。请记住，默认情况下，该类使用`BOARD`号码，因此这里的正确值是`8`、`10`和`12`。如果要使用`BCM`号码，请确保在构造函数参数中指定这一点。

现在让我们添加一个颜色选择对话框：

```py
        ccd = qtw.QColorDialog()
        ccd.setOptions(
            qtw.QColorDialog.NoButtons
            | qtw.QColorDialog.DontUseNativeDialog)
        ccd.currentColorChanged.connect(self.set_color)
        self.setCentralWidget(ccd)
```

通常，我们通过调用`QColorDialog.getColor()`来调用颜色对话框，但在这种情况下，我们希望将对话框用作小部件。因此，我们直接实例化一个对话框，并设置`NoButtons`和`DontUseNativeDialog`选项。通过去掉按钮并使用对话框的 Qt 版本，我们可以防止用户取消或提交对话框。这允许我们将其视为常规小部件并将其分配为主窗口的中央小部件。

我们已经将`currentColorChanged`信号（每当用户选择颜色时发出）连接到一个名为`set_color()`的`MainWindow`方法。我们将在接下来添加这个方法，如下所示：

```py
    def set_color(self, color):
        self.tcl.set_color(color.red(), color.green(), color.blue())
```

`currentColorChanged`信号包括表示所选颜色的`QColor`对象，因此我们可以简单地使用`QColor`属性访问器将其分解为红色、绿色和蓝色值，然后将该信息传递给我们的`ThreeColorLed`对象的`set_color()`方法。

现在脚本已经完成。您应该能够运行它并点亮 LED-试试看！

请注意，您选择的颜色可能不会完全匹配 LED 的颜色输出，因为不同颜色 LED 的相对亮度不同。但它们应该是相当接近的。

# 使用 GPIO 设备控制 PyQt

使用 GPIO 引脚从 Python 控制电路非常简单。只需调用`GPIO.output()`函数，并使用适当的引脚编号和高或低值。然而，现在我们要看相反的情况，即从 GPIO 输入控制或更新 PyQt GUI。

为了演示这一点，我们将构建一个温度和湿度读数。就像以前一样，我们将从连接电路开始。

# 连接传感器电路

DHT 11 和 DHT 22 传感器都是温度和湿度传感器，可以很容易地与树莓派一起使用。两者都打包为四针元件，但实际上只使用了三根引脚。一些元件套件甚至将 DHT 11/22 安装在一个小 PCB 上，只有三根活动引脚用于输出。

无论哪种情况，如果您正在查看 DHT 的正面（即，格栅一侧），则从左到右的引脚如下：

+   输入电压——5 或 3 伏特

+   传感器输出

+   死引脚（在 4 针配置中）

+   地线

DHT 11 或 DHT 22 对于这个项目都同样适用。11 更小更便宜，但比 22 慢且不太准确。否则，它们在功能上是一样的。

将传感器插入面包板中，使每个引脚都在自己的行中。然后，使用跳线线将其连接到树莓派，如下面的屏幕截图所示：

![](img/a29672ae-3bab-464d-891e-54cde52127fe.png)

传感器的电压输入引脚可以连接到任何一个 5V 引脚，地线可以连接到任何一个 GND 引脚。此外，数据引脚可以连接到树莓派上的任何 GPIO 引脚，但在这种情况下，我们将使用引脚 7（再次，按照`BOARD`编号）。

仔细检查您的连接，确保一切正确，然后打开树莓派的电源，我们将开始编码。

# 创建传感器接口

要开始我们的传感器接口软件，首先创建另一个 Qt 应用程序模板的副本，并将其命名为`temp_humid_display.py`。

我们将首先导入必要的库，如下所示：

```py
import Adafruit_DHT
from RPi import GPIO
```

`Adafruit_DHT`将封装与 DHT 单元通信所需的所有复杂部分，因此我们只需要使用高级功能来控制和读取设备的数据。

在导入下面，让我们设置一个全局常量：

```py
SENSOR_MODEL = 11
GPIO.setmode(GPIO.BCM)
```

我们正在设置一个全局常量，指示我们正在使用哪个型号的 DHT；如果您有 DHT 22，则将此值设置为 22。我们还设置了树莓派的引脚模式。但这次，我们将使用`BCM`模式来指定我们的引脚编号。Adafruit 库只接受`BCM`编号，因此在我们所有的类中保持一致是有意义的。

现在，让我们开始为 DHT 创建传感器接口类：

```py
class SensorInterface(qtc.QObject):

    temperature = qtc.pyqtSignal(float)
    humidity = qtc.pyqtSignal(float)
    read_time = qtc.pyqtSignal(qtc.QTime)
```

这一次，我们将基于`QObject`类来创建我们的类，以便在从传感器读取值时发出信号，并在其自己的线程中运行对象。DHT 单元有点慢，当我们请求读数时可能需要一秒或更长时间来响应。因此，我们希望在单独的执行线程中运行其接口。正如您可能记得的来自第十章 *使用 QTimer 和 QThread 进行多线程处理*，当我们可以使用信号和插槽与对象交互时，这很容易实现。

现在，让我们添加`__init__()`方法，如下所示：

```py
    def __init__(self, pin, sensor_model, fahrenheit=False):
        super().__init__()
        self.pin = pin
        self.model = sensor_model
        self.fahrenheit = fahrenheit
```

构造函数将接受三个参数：连接到数据线的引脚，型号（11 或 22），以及一个布尔值，指示我们是否要使用华氏或摄氏温标。我们暂时将所有这些参数保存到实例变量中。

现在我们想要创建一个方法来告诉传感器进行读数：

```py
    @qtc.pyqtSlot()
    def take_reading(self):
        h, t = Adafruit_DHT.read_retry(self.model, self.pin)
        if self.fahrenheit:
            t = ((9/5) * t) + 32
        self.temperature.emit(t)
        self.humidity.emit(h)
        self.read_time.emit(qtc.QTime.currentTime())
```

正如您所看到的，`Adafruit_DHT`库消除了读取传感器的所有复杂性。我们只需使用传感器的型号和引脚号调用`read_entry()`，它就会返回一个包含湿度和温度值的元组。温度以摄氏度返回，因此对于美国用户，如果对象配置为这样做，我们将进行计算将其转换为华氏度。然后，我们发出三个信号——分别是温度、湿度和当前时间。

请注意，我们使用`pyqtSlot`装饰器包装了这个函数。再次回想一下第十章中的内容，*使用 QTimer 和 QThread 进行多线程处理*，这将消除将这个类移动到自己的线程中的一些复杂性。

这解决了我们的传感器驱动程序类，现在让我们构建 GUI。

# 显示读数

在本书的这一部分，创建一个 PyQt GUI 来显示一些数字应该是轻而易举的。为了增加趣味性并创建时尚的外观，我们将使用一个我们还没有讨论过的小部件——`QLCDNumber`。

首先，在`MainWindow.__init__()`中创建一个基本小部件，如下所示：

```py
        widget = qtw.QWidget()
        widget.setLayout(qtw.QFormLayout())
        self.setCentralWidget(widget)
```

现在，让我们应用一些我们在第六章中学到的样式技巧，*Qt 应用程序样式*：

```py
        p = widget.palette()
        p.setColor(qtg.QPalette.WindowText, qtg.QColor('cyan'))
        p.setColor(qtg.QPalette.Window, qtg.QColor('navy'))
        p.setColor(qtg.QPalette.Button, qtg.QColor('#335'))
        p.setColor(qtg.QPalette.ButtonText, qtg.QColor('cyan'))
        self.setPalette(p)
```

在这里，我们为这个小部件及其子级创建了一个自定义的`QPalette`对象，给它一个类似于蓝色背光 LCD 屏幕的颜色方案。

接下来，让我们创建用于显示我们的读数的小部件：

```py
        tempview = qtw.QLCDNumber()
        humview = qtw.QLCDNumber()
        tempview.setSegmentStyle(qtw.QLCDNumber.Flat)
        humview.setSegmentStyle(qtw.QLCDNumber.Flat)
        widget.layout().addRow('Temperature', tempview)
        widget.layout().addRow('Humidity', humview)
```

`QLCDNumber`小部件是用于显示数字的小部件。它类似于一个八段数码管显示，例如您可能在仪表板或数字时钟上找到的。它的`segmentStyle`属性在几种不同的视觉样式之间切换；在这种情况下，我们使用`Flat`，它用前景颜色填充了段。

现在布局已经配置好了，让我们创建一个传感器对象：

```py
        self.sensor = SensorInterface(4, SENSOR_MODEL, True)
        self.sensor_thread = qtc.QThread()
        self.sensor.moveToThread(self.sensor_thread)
        self.sensor_thread.start()
```

在这里，我们创建了一个连接到 GPIO4 引脚（即 7 号引脚）的传感器，传入我们之前定义的`SENSOR_MODEL`常量，并将华氏度设置为`True`（如果您喜欢摄氏度，可以随时将其设置为`False`）。之后，我们创建了一个`QThread`对象，并将`SensorInterface`对象移动到其中。

接下来，让我们连接我们的信号和插槽，如下所示：

```py
        self.sensor.temperature.connect(tempview.display)
        self.sensor.humidity.connect(humview.display)
        self.sensor.read_time.connect(self.show_time)
```

`QLCDNumber.display()`插槽可以连接到发出数字的任何信号，因此我们直接连接我们的温度和湿度信号。然而，发送到`read_time`信号的`QTime`对象将需要一些解析，因此我们将其连接到一个名为`show_time()`的`MainWindow`方法。

该方法看起来像以下代码块：

```py
    def show_time(self, qtime):
        self.statusBar().showMessage(
            f'Read at {qtime.toString("HH:mm:ss")}')
```

这个方法将利用`MainWindow`对象方便的`statusBar()`方法，在状态区域显示最后一次温度读数的时间。

因此，这解决了我们的 GUI 输出显示；现在我们需要一种方法来触发传感器定期进行读数。我们可以采取的一种方法是创建一个定时器来定期执行它：

```py
        self.timer = qtc.QTimer(interval=(60000))
        self.timer.timeout.connect(self.sensor.take_reading)
        self.timer.start()
```

在这种情况下，这个定时器将每分钟调用`sensor.take_reading()`，确保我们的读数定期更新。

我们还可以在界面中添加`QPushButton`，以便用户可以随时获取新的读数：

```py
        readbutton = qtw.QPushButton('Read Now')
        widget.layout().addRow(readbutton)
        readbutton.clicked.connect(self.sensor.take_reading)
```

这相当简单，因为我们只需要将按钮的`clicked`信号连接到传感器的`take_reading`插槽。但是硬件控制呢？我们如何实现外部触发温度读数？我们将在下一节中探讨这个问题。

# 添加硬件按钮

从传感器读取值可能是有用的，但更有用的是能够响应电路中发生的事件并作出相应的行动。为了演示这个过程，我们将在电路中添加一个硬件按钮，并监视它的状态，以便我们可以在按下按钮时进行温度和湿度读数。

# 扩展电路

首先，关闭树莓派的电源，让我们向电路中添加一些组件，如下图所示：

![](img/b493a2fa-96bd-4435-9ba8-d2ae3ee2fe9c.png)

在这里，我们基本上添加了一个按钮和一个电阻。按钮需要连接到树莓派上的引脚 8 的一侧，而电阻连接到地面的另一侧。为了保持布线整洁，我们还利用了面包板侧面的公共地和公共电压导轨，尽管这是可选的（如果您愿意，您可以直接将东西连接到树莓派上的适当 GND 和 5V 引脚）。

在入门套件中经常找到的按钮有四个连接器，每侧两个开关。确保您的连接在按钮被按下之前不连接。如果您发现即使没有按下按钮，它们也总是连接在一起，那么您可能需要将按钮在电路中旋转 90 度。

在这个电路中，按钮在被按下时将简单地将我们的 GPIO 引脚连接到地面，这将允许我们检测按钮按下。当我们编写软件时，我们将更详细地了解它是如何工作的。

# 实现按钮驱动程序

在脚本的顶部开始一个新的类，作为我们按钮的驱动程序：

```py
class HWButton(qtc.QObject):

    button_press = qtc.pyqtSignal()
```

再次，我们使用`QObject`，以便我们可以发出 Qt 信号，当我们检测到按钮被按下时，我们将这样做。

现在，让我们编写构造函数，如下所示：

```py
    def __init__(self, pin):
        super().__init__()
        self.pin = pin
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
```

在调用`super().__init__()`之后，我们的`__init__()`方法的第一件事是通过将`GPIO.IN`常量传递给`setup()`函数来将我们的按钮的 GPIO 引脚配置为输入引脚。

我们在这里传递的`pull_up_down`值非常重要。由于我们连接电路的方式，当按钮被按下时，引脚将连接到地面。但是当按钮没有被按下时会发生什么？嗯，在这种情况下，它处于**浮动**状态，其中输入将是不可预测的。为了在按钮没有被按下时保持引脚处于可预测的状态，`pull_up_down`参数将导致在没有其他连接时将其拉到`HIGH`或`LOW`。在我们的情况下，我们希望它被拉到`HIGH`，因为我们的按钮将把它拉到`LOW`；传递`GPIO.PUD_UP`常量将实现这一点。

这也可以以相反的方式工作；例如，我们可以将按钮的另一侧连接到 5V，然后在`setup()`函数中将`pull_up_down`设置为`GPIO.PUD_DOWN`。

现在，我们需要弄清楚如何检测按钮何时被按下，以便我们可以发出信号。

这项任务的一个简单方法是**轮询**。轮询简单地意味着我们将定期检查按钮，并在上次检查时发生变化时发出信号。

为此，我们首先需要创建一个实例变量来保存按钮的上一个已知状态：

```py
       self.pressed = GPIO.input(self.pin) == GPIO.LOW
```

我们可以通过调用`GPIO.input()`函数并传递引脚号来检查按钮的当前状态。此函数将返回`HIGH`或`LOW`，指示引脚是否为 5V 或地面。如果引脚为`LOW`，那么意味着按钮被按下。我们将将结果保存到`self.pressed`。

接下来，我们将编写一个方法来检查按钮状态的变化：

```py
    def check(self):
        pressed = GPIO.input(self.pin) == GPIO.LOW
        if pressed != self.pressed:
            if pressed:
                self.button_press.emit()
            self.pressed = pressed
```

这个检查方法将采取以下步骤：

1.  首先，它将比较`input()`的输出与`LOW`常量，以查看按钮是否被按下

1.  然后，我们比较按钮的当前状态与保存的状态，以查看按钮的状态是否发生了变化

1.  如果有，我们需要检查状态的变化是按下还是释放

1.  如果是按下（`pressed`为`True`），那么我们发出信号

1.  无论哪种情况，我们都会使用新状态更新`self.pressed`

现在，剩下的就是定期调用这个方法来轮询变化；在`__init__()`中，我们可以使用定时器来做到这一点，如下所示：

```py
        self.timer = qtc.QTimer(interval=50, timeout=self.check)
        self.timer.start()
```

在这里，我们创建了一个定时器，每 50 毫秒超时一次，当这样做时调用`self.check()`。这应该足够频繁，以至于可以捕捉到人类可以执行的最快的按钮按下。

轮询效果很好，但使用`GPIO`库的`add_event_detect()`函数有一种更干净的方法来做到这一点：

```py
        # Comment out timer code
        #self.timer = qtc.QTimer(interval=50, timeout=self.check)
        #self.timer.start()
        GPIO.add_event_detect(
            self.pin,
            GPIO.RISING,
            callback=self.on_event_detect)
```

`add_event_detect()`函数将在另一个线程中开始监视引脚，以侦听`RISING`事件或`FALLING`事件，并在检测到此类事件时调用配置的`callback`方法。

在这种情况下，我们只需调用以下实例方法：

```py
    def on_event_detect(self, *args):
        self.button_press.emit()
```

我们可以直接将我们的`emit()`方法作为回调传递，但是`add_event_detect()`将使用引脚号调用回调函数作为参数，而`emit()`将不接受。

使用`add_event_detect()`的缺点是它引入了另一个线程，使用 Python 的`threading`库，这可能会导致与 PyQt 事件循环的微妙问题。轮询是一个完全可行的替代方案，可以避免这种复杂性。

这两种方法都适用于我们的简单脚本，所以让我们回到`MainWindow.__init__()`来为我们的按钮添加支持：

```py
        self.hwbutton = HWButton(8)
        self.hwbutton.button_press.connect(self.sensor.take_reading)
```

我们所需要做的就是创建一个`HWButton`类的实例，使用正确的引脚号，并将其`button_press`信号连接到传感器的`take_reading()`插槽。

现在，如果您在树莓派上启动所有内容，当您按下按钮时，您应该能够看到更新。

# 总结

树莓派是一项令人兴奋的技术，不仅因为其小巧、低成本和低资源使用率，而且因为它使得将编程世界与真实电路的连接变得简单和易于访问，这是以前没有的。在本章中，您学会了如何配置树莓派来运行 PyQt 应用程序。您还学会了如何使用 PyQt 和 Python 控制电路，以及电路如何控制软件中的操作。

在下一章中，我们将使用`QtWebEngineWidgets`将全球网络引入我们的 PyQt 应用程序，这是一个完整的基于 Chromium 的浏览器，内置在 Qt Widget 中。我们将构建一个功能齐全的浏览器，并了解网络引擎库的各个方面。

# 问题

尝试回答以下问题，以测试您从本章中获得的知识：

1.  您刚刚购买了一个预装了 Raspbian 的树莓派来运行您的 PyQt5 应用程序。当您尝试运行您的应用程序时，您遇到了一个错误，试图导入`QtNetworkAuth`，这是您的应用程序所依赖的。问题可能是什么？

1.  您已经为传统扫描仪设备编写了一个 PyQt 前端。您的代码通过一个名为`scanutil.exe`的专有驱动程序实用程序与扫描仪通信。它目前正在运行在 Windows 10 PC 上，但您的雇主希望通过将其移动到树莓派来节省成本。这是一个好主意吗？

1.  您已经获得了一个新的传感器，并希望尝试将其与树莓派一起使用。它有三个连接，标有 Vcc、GND 和 Data。您将如何将其连接到树莓派？您还需要更多信息吗？

1.  您正在点亮连接到最左边的第四个 GPIO 引脚的 LED。这段代码有什么问题？

```py
   GPIO.setmode(GPIO.BCM)
   GPIO.setup(8, GPIO.OUT)
   GPIO.output(8, 1)
```

1.  您正在调暗连接到 GPIO 引脚 12 的 LED。以下代码有效吗？

```py
   GPIO.setmode(GPIO.BOARD)
   GPIO.setup(12, GPIO.OUT)
   GPIO.output(12, 0.5)
```

1.  您有一个运动传感器，当检测到运动时，数据引脚会变为`HIGH`。它连接到引脚`8`。以下是您的驱动代码：

```py
   class MotionSensor(qtc.QObject):

       detection = qtc.pyqtSignal()

       def __init__(self):
           super().__init__()
           GPIO.setmode(GPIO.BOARD)
           GPIO.setup(8, GPIO.IN)
           self.state = GPIO.input(8)

       def check(self):
           state = GPIO.input(8)
           if state and state != self.state:
               detection.emit()
           self.state = state
```

您的主窗口类创建了一个`MotionSensor`对象，并将其`detection`信号连接到回调方法。然而，没有检测到任何东西。缺少了什么？

1.  以创造性的方式将本章中的两个电路结合起来；例如，您可以创建一个根据湿度和温度变化颜色的灯。

# 进一步阅读

有关更多信息，请参阅以下内容：

+   有关树莓派的`GPIO`库的更多文档可以在[`sourceforge.net/p/raspberry-gpio-python/wiki/Home/`](https://sourceforge.net/p/raspberry-gpio-python/wiki/Home/)找到

+   Packt 提供了许多详细介绍树莓派的书籍；您可以在[`www.packtpub.com/books/content/raspberry-pi`](https://www.packtpub.com/books/content/raspberry-pi)找到更多信息。
