# 第十二章：增强我们的物联网门铃

在第十章中，我们探索了网络服务。然后在第十一章中引入了蓝牙，并使用 Android 应用蓝点和我们的树莓派构建了蓝牙门铃。

在本章中，我们将通过添加在有人敲门时发送短信的功能来增强我们的蓝牙门铃。我们将运用所学知识，并使用我们在第十章中设置的 Twilio 账户，添加短信功能。

本章将涵盖以下主题：

+   有人敲门时发送短信

+   创建一个带有短信功能的秘密门铃应用

# 项目概述

在本章的两个项目中，我们将使用第十一章中的电路，同时还将使用 Android 设备上的蓝点应用，如第十一章中所述。以下是本章中我们将创建的应用的图表：

![](img/48881639-1b37-4934-880b-12c34b044a47.png)

我们将创建这个应用的两个版本。我们的应用的第一个版本将是一个简单的蓝牙门铃，按下蓝点会触发蜂鸣器和 RGB LED 灯光秀。警报触发后，将使用 Twilio 云服务发送一条短信。

应用程序的修改版本将使用蓝点应用上的滑动手势来指示特定的访客。四位潜在的访客将各自拥有自己独特的蓝点滑动手势。在自定义蜂鸣器响铃和 RGB LED 灯光秀之后，将发送一条文本消息通知收件人门口有谁。Twilio 云也将用于此功能。

这两个项目应该需要一个上午或一个下午的时间来完成。

# 入门

完成此项目需要以下步骤：

+   树莓派 3 型（2015 年或更新型号）

+   USB 电源适配器

+   计算机显示器

+   USB 键盘

+   USB 鼠标

+   面包板

+   跳线线

+   330 欧姆电阻（3 个）

+   RGB LED

+   有源蜂鸣器

+   Android 设备（手机/平板）

# 有人敲门时发送短信

在第十章中，我们使用了一种叫做 Twilio 的技术来创建文本消息。在那个例子中，我们使用 Twilio 在检测到入侵者时发送文本消息。在第十一章中，我们使用了 Android 手机或平板上的蓝点应用创建了一个蓝牙门铃。门铃响起蜂鸣器，并在 RGB LED 上进行了一些灯光秀。

对于这个项目，我们将结合 Twilio 和蓝牙门铃，当有人按下蓝点门铃时，将发送一条短信（参考第十章和第十一章，熟悉这些技术）。

# 创建一个带有短信功能的简单门铃应用

要创建我们的简单门铃应用，请执行以下操作：

1.  从应用程序菜单 | 编程 | Thonny Python IDE 中打开 Thonny

1.  点击“新建”图标创建一个新文件

1.  输入以下内容：

```py
from twilio.rest import Client
from gpiozero import RGBLED
from gpiozero import Buzzer
from bluedot import BlueDot
from signal import pause
from time import sleep

class Doorbell:
    account_sid = ''
    auth_token = ''
    from_phonenumber=''
    test_env = True
    led = RGBLED(red=17, green=22, blue=27)
    buzzer = Buzzer(26)
    num_of_rings = 0
    ring_delay = 0
    msg = ''

    def __init__(self, 
                 num_of_rings = 1, 
                 ring_delay = 1, 
                 message = 'ring', 
                 test_env = True):
        self.num_of_rings = num_of_rings
        self.ring_delay = ring_delay
        self.message = message
        self.test_env = self.setEnvironment(test_env)

    def setEnvironment(self, test_env):
        if test_env:
            self.account_sid = '<<test account_sid>>'
            self.auth_token = '<<test auth_token>>'
            return True
        else:
            self.account_sid = '<<live account_sid>>'
            self.auth_token = '<<live auth_token>>'
            return False

    def doorbell_sequence(self):
        num = 0
        while num < self.num_of_rings:
            self.buzzer.on()
            self.light_show()
            sleep(self.ring_delay)
            self.buzzer.off()
            sleep(self.ring_delay)
            num += 1
        return self.sendTextMessage()

    def sendTextMessage(self):
        twilio_client = Client(self.account_sid, self.auth_token)
        if self.test_env:
            message = twilio_client.messages.create(
                        body=self.message,
                        from_= '+15005550006',
                        to='<<your phone number>>'
            )
        else:
            message = twilio_client.messages.create(
                        body=self.message,
                        from_= '<<your twilio number>>',
                        to='<<your phone number>>'
            ) 
        return 'Doorbell text message sent - ' + message.sid

    def light_show(self):
        self.led.color=(1,0,0)
        sleep(0.5)
        self.led.color=(0,1,0)
        sleep(0.5)
        self.led.color=(0,0,1)
        sleep(0.5)
        self.led.off()

def pressed():
    doorbell = Doorbell(2, 0.5, 'There is someone at the door')
    print(doorbell.doorbell_sequence())

blue_dot = BlueDot()
blue_dot.when_pressed = pressed

if __name__=="__main__":
    pause()

```

1.  将文件保存为`Doorbell.py`并运行

1.  在您的 Android 设备上打开蓝点应用

1.  连接到树莓派

1.  按下大蓝点

你应该听到铃声并看到灯光序列循环两次，两次之间有短暂的延迟。你应该在 shell 中得到类似以下的输出：

```py
Server started B8:27:EB:12:77:4F
Waiting for connection
Client connected F4:0E:22:EB:31:CA
Doorbell text message sent - SM5cf1125acad44016840a6b76f99b3624
```

前三行表示 Blue Dot 应用程序已通过我们的 Python 程序连接到我们的 Raspberry Pi。最后一行表示已发送了一条短信。由于我们使用的是测试环境，实际上没有发送短信，但是调用了 Twilio 服务。

让我们来看看代码。我们首先定义了我们的类，并给它命名为`Doorbell`。这是我们类的一个很好的名字，因为我们已经编写了我们的代码，使得一切与门铃有关的东西都包含在`Doorbell.py`文件中。这个文件包含了`Doorbell`类，用于提醒用户，以及 Blue Dot 代码，用于触发门铃。Blue Dot 代码实际上位于`Doorbell`类定义之外，因为我们认为它是 Blue Dot 应用的一部分，而不是门铃本身。我们当然可以设计我们的代码，使得`Doorbell`类包含触发警报的代码；然而，将警报与警报触发器分开使得在将来更容易重用`Doorbell`类作为警报机制。

选择类名可能有些棘手。然而，选择正确的类名非常重要，因为使用适合其预期用途的类名将更容易构建应用程序。类名通常是名词，类中的方法是动词。通常，最好让一个类代表一件事或一个想法。例如，我们将我们的类命名为`Doorbell`，因为我们已经设计它来封装门铃的功能：提醒用户有人在门口。考虑到这个想法，`Doorbell`类包含点亮 LED、发出蜂鸣器声音和发送短信的代码是有意义的，因为这三个动作都属于提醒用户的想法。

在我们定义了我们的类之后，我们创建了以下用于我们类的类变量：

```py
class Doorbell:
    account_sid = ''
    auth_token = ''
    from_phonenumber=''
    test_env = True
    led = RGBLED(red=17, green=22, blue=27)
    buzzer = Buzzer(26)
    num_of_rings = 0
    ring_delay = 0
    msg = ''
```

`init`和`setEnvironment`方法设置了我们在类中使用的变量。`test_env`变量确定我们在代码中使用 Twilio 测试环境还是实时环境。测试环境是默认使用的：

```py
def __init__(self, 
             num_of_rings = 1, 
             ring_delay = 1, 
             message = 'ring', 
             test_env = True):
     self.num_of_rings = num_of_rings
     self.ring_delay = ring_delay
     self.message = message
     self.test_env = self.setEnvironment(test_env)

 def setEnvironment(self, test_env):
     if test_env:
         self.account_sid = '<<test account sid>>'
         self.auth_token = '<<test auth token>>'
         return True
     else:
         self.account_sid = '<<live account sid>>'
         self.auth_token = '<<auth_token>>'
         return False
```

`doorbell_sequence`、`sendTextMessage`和`light_show`方法与本书先前介绍的方法类似。通过这三种方法，我们通知用户有人在门口。这里需要注意的是从`sendTextMessage`方法发送的返回值：`return 'Doorbell text message sent - ' + message.sid`。通过在代码中加入这一行，我们能够使用`sendTextMessage`方法在我们的 shell 中提供一个打印确认，即已发送了一条短信。

如前所述，我们的代码中的 Blue Dot 部分位于类定义之外：

```py
def pressed():
    doorbell = Doorbell(2, 0.5, 'There is someone at the door')
    print(doorbell.doorbell_sequence())

blue_dot = BlueDot()
blue_dot.when_pressed = pressed
```

前面的代码是我们以前见过的。我们定义了`pressed`方法，在这里我们实例化了一个新的`doorbell`对象，然后调用了`doorbell`的`doorbell_sequence`方法。`blue_dot`变量是一个`BlueDot`对象，我们只关心`when_pressed`事件。

这里需要注意的是包含`doorbell = Doorbell(2, 0.5, 'There is someone at the door')`语句的那一行。在这一行中，我们实例化了一个`Doorbell`对象，我们称之为`doorbell`，`num_of_rings`等于`2`；`ring_delay`（或持续时间）等于`0.5`；消息等于`门口有人`。我们没有传递`test_env`环境值。因此，默认设置为`True`，用于设置我们的`doorbell`对象使用 Twilio 测试环境，不发送短信。要更改为发送短信，将语句更改为：

```py
doorbell = Doorbell(2, 0.5, 'There is someone at the door', False)
```

确保您相应地设置了 Twilio 帐户参数。您应该收到一条短信，告诉您有人在门口。以下是我在 iPhone 上收到的消息：

![](img/1d69e13e-94fe-4952-8dc1-998cca3258f9.png)

# 创建一个带有短信功能的秘密门铃应用程序

现在我们有能力在安卓设备上的大蓝色按钮被按下时发送文本消息，让我们把它变得更加复杂一些。我们将修改我们在第十一章中创建的`SecretDoorbell`类，*使用蓝牙创建门铃按钮*，并赋予它发送文本消息告诉我们谁在门口的能力。就像之前一样，我们将把所有的代码放在一个文件中以保持紧凑：

1.  从应用程序菜单中打开 Thonny | 编程 | Thonny Python IDE

1.  点击新建图标创建一个新文件

1.  输入以下内容：

```py
from twilio.rest import Client
from gpiozero import RGBLED
from gpiozero import Buzzer
from bluedot import BlueDot
from signal import pause
from time import sleep

class Doorbell:
    account_sid = ''
    auth_token = ''
    from_phonenumber=''
    test_env = True
    led = RGBLED(red=17, green=22, blue=27)
    buzzer = Buzzer(26)
    num_of_rings = 0
    ring_delay = 0
    msg = ''

    def __init__(self, 
                 num_of_rings = 1, 
                 ring_delay = 1, 
                 message = 'ring', 
                 test_env = True):
        self.num_of_rings = num_of_rings
        self.ring_delay = ring_delay
        self.message = message
        self.test_env = self.setEnvironment(test_env)

    def setEnvironment(self, test_env):
        if test_env:
            self.account_sid = '<<test account_sid>>'
            self.auth_token = '<<test auth_token>>'
            return True
        else:
            self.account_sid = '<<live account_sid>>'
            self.auth_token = '<<live auth_token>>'
            return False

    def doorbell_sequence(self):
        num = 0
        while num < self.num_of_rings:
            self.buzzer.on()
            self.light_show()
            sleep(self.ring_delay)
            self.buzzer.off()
            sleep(self.ring_delay)
            num += 1
        return self.sendTextMessage()

    def sendTextMessage(self):
        twilio_client = Client(self.account_sid, self.auth_token)
        if self.test_env:
            message = twilio_client.messages.create(
                        body=self.message,
                        from_= '+15005550006',
                        to='<<your phone number>>'
            )
        else:
            message = twilio_client.messages.create(
                        body=self.message,
                        from_= '<<your twilio number>>',
                        to='<<your phone number>>'
            ) 
        return 'Doorbell text message sent - ' + message.sid

    def light_show(self):
        self.led.color=(1,0,0)
        sleep(0.5)
        self.led.color=(0,1,0)
        sleep(0.5)
        self.led.color=(0,0,1)
        sleep(0.5)
        self.led.off()

class SecretDoorbell(Doorbell):
    names=[['Bob', 4, 0.5], 
           ['Josephine', 1, 3], 
           ['Ares', 6, 0.2], 
           ['Constance', 2, 1]]
    message = ' is at the door!'

    def __init__(self, person_num, test_env = True):
        Doorbell.__init__(self,
                          self.names[person_num][1],
                          self.names[person_num][2],
                          self.names[person_num][0] + self.message,
                          test_env)

def swiped(swipe):
    if swipe.up:
        doorbell = SecretDoorbell(0)
        print(doorbell.doorbell_sequence())
    elif swipe.down:
        doorbell = SecretDoorbell(1)
        print(doorbell.doorbell_sequence())
    elif swipe.left:
        doorbell = SecretDoorbell(2)
        print(doorbell.doorbell_sequence())
    elif swipe.right:
        doorbell = SecretDoorbell(3)
        print(doorbell.doorbell_sequence())

blue_dot = BlueDot()
blue_dot.when_swiped = swiped

if __name__=="__main__":
    pause()
```

1.  将文件保存为`SecretDoorbell.py`并运行它

1.  在您的安卓设备上打开蓝点应用

1.  连接到树莓派

1.  从顶部位置向下滑动蓝点

1.  您应该听到蜂鸣器响一次，大约持续三秒钟，并且看到 RGB LED 进行一次灯光表演。在 shell 底部将显示类似以下内容：

```py
Server started B8:27:EB:12:77:4F
Waiting for connection
Client connected F4:0E:22:EB:31:CA
Doorbell text message sent - SM62680586b32a42bdacaff4200e0fed78
```

1.  和之前的项目一样，我们将会收到一条文本消息已发送的消息，但实际上我们不会收到文本消息，因为我们处于 Twilio 测试环境中

在让我们的应用程序根据他们的滑动给我们发送一条告诉我们门口有谁的短信之前，让我们看一下代码。

我们的`SecretDoorbell.py`文件与我们的`Doorbell.py`文件完全相同，除了以下代码：

```py
class SecretDoorbell(Doorbell):
    names=[['Bob', 4, 0.5], 
           ['Josephine', 1, 3], 
           ['Ares', 6, 0.2], 
           ['Constance', 2, 1]]
    message = ' is at the door!'

    def __init__(self, person_num, test_env = True):
        Doorbell.__init__(self,
                          self.names[person_num][1],
                          self.names[person_num][2],
                          self.names[person_num][0] + self.message,
                          test_env)

def swiped(swipe):
    if swipe.up:
        doorbell = SecretDoorbell(0)
        print(doorbell.doorbell_sequence())
    elif swipe.down:
        doorbell = SecretDoorbell(1)
        print(doorbell.doorbell_sequence())
    elif swipe.left:
        doorbell = SecretDoorbell(2)
        print(doorbell.doorbell_sequence())
    elif swipe.right:
        doorbell = SecretDoorbell(3)
        print(doorbell.doorbell_sequence())

blue_dot = BlueDot()
blue_dot.when_swiped = swiped
```

`SecretDoorbell`类被创建为`Doorbell`的子类，从而继承了`Doorbell`的方法。我们创建的`names`数组存储了数组中的名称和与名称相关的铃声属性。例如，第一个元素的名称是`Bob`，`num_of_rings`值为`4`，`ring_delay`（持续时间）值为`0.5`。当这条记录在 Twilio 实时环境中使用时，您应该听到蜂鸣器响四次，并看到 RGB LED 灯光表演循环，之间有短暂的延迟。`SecretDoorbell`的`init`方法收集`person_num`（或者基本上是`names`数组中的位置信息），并用它来实例化`Doorbell`父类。`test_env`值默认为`True`，这意味着我们只能通过明确覆盖这个值来打开 Twilio 实时环境。这样可以防止我们在准备好部署应用程序之前意外使用完 Twilio 账户余额。

我们文件中的蓝点代码位于`SecretDoorbell`类定义之外。和之前的项目一样，这样做可以让我们将门铃功能与门铃触发器（我们安卓设备上的蓝点应用）分开。

在我们的蓝点代码中，我们实例化一个名为`blue_dot`的`BlueDot`对象，然后将`when_swiped`事件赋给`swiped`。在`swiped`中，我们实例化一个`SecretDoorbell`对象，为`swipe.up`手势赋值`0`，为`swipe.down`赋值`1`，为`swipe.left`赋值`2`，为`swipe.right`赋值`3`。这些值对应于`SecretDoorbell`类的`names`数组中的位置。我们在为任何手势实例化`SecretDoorbell`对象时不传递`test_env`的值，因此不会发送文本消息。就像之前的项目一样，我们在 shell 中打印`doorbell_sequence`方法运行成功的结果。

要发送文本消息，我们只需要用`False`值覆盖默认的`test_env`值。我们在`swiped`方法中为我们的滑动手势实例化`SecretDoorbell`对象时这样做。我们的代码设计成这样的方式，我们可以为一个或多个手势发送文本消息。修改`swiped`中的以下`elif`语句：

```py
elif swipe.down:
    doorbell = SecretDoorbell(1, False)
    print(doorbell.doorbell_sequence())
```

我们在这里所做的是通过覆盖`test_env`变量，为`swipe.down`手势打开了 Twilio 实时环境。我们为`SecretDoorbell`对象实例化时使用的`1`值对应于`SecretDoorbell`中`names`数组中的第二个元素。

因此，当你运行应用程序并在蓝点上从上向下滑动时，你应该收到来自 Twilio 的一条短信，内容是 Josephine 在门口，如下所示：

![](img/fa0067b8-c184-4ae2-9be1-281df3a40d2f.png)

# 摘要

在本章中，我们学习了如何将短信功能添加到我们的门铃应用程序中。这使得门铃适应了物联网时代。很容易看出物联网蓝牙门铃的概念可以被扩展——想象一下当有人按门铃时打开门廊灯。

我们还可以看到蓝点应用程序也可以以其他方式被利用。我们可以使用蓝点应用程序编程特定的滑动序列，也许是为了解锁门。想象一下不必随身携带钥匙！

这是我们介绍我们的机器人车之前的最后一章。在接下来的章节中，我们将把我们迄今为止学到的概念应用到我们通过互联网控制的机器人上。

# 问题

1.  蓝点应用程序如何连接到我们的树莓派？

1.  正确还是错误？通过 Twilio 测试环境运行消息会创建一条发送到你手机的短信。

1.  我们用来发送短信的服务的名称是什么？

1.  正确还是错误？我们将我们的`SecretDoorbell`类创建为`Doorbell`类的子类。

1.  我们在第二个应用程序中使用的四个蓝点手势是什么？

1.  正确还是错误？以描述其功能的方式命名一个类会使编码变得更容易。

1.  `Doorbell`和`SecretDoorbell`之间有什么区别？

1.  正确还是错误？Josephine 的铃声模式包括一个长的蜂鸣声。

1.  正确还是错误？为了从我们的应用程序接收短信，你需要使用安卓手机。

1.  康斯坦斯应该如何滑动蓝点，这样我们就知道是她在门口？

# 进一步阅读

我们稍微涉及了 Twilio 服务。然而，还有更多需要学习的地方——访问[`www.twilio.com/docs/tutorials`](https://www.twilio.com/docs/tutorials)获取更多信息。
