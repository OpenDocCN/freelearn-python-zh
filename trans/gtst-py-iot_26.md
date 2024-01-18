# 使贾维斯成为物联网设备

曾经我们曾经想象用手指控制世界。现在，这种想象已经成为现实。随着智能手机的出现，我们已经在做一些在十年前只能想象的事情。随着手机变得智能，行业和企业也尽力跟上这种颠覆性的变化。然而，仍然有一部分落后了。那是哪一部分？你的家！

想想你可以用智能手机控制家里的什么？并不多！有一些设备可以打开或关闭一堆设备，比如你的空调。然而，这个清单是详尽的。因此，凭借在前几章中获得的所有知识和我们手中强大的硬件，为什么我们不成为引领潮流和颠覆者，创造一些仍然只存在于我们想象中的东西呢。

本章将涵盖以下主题：

+   **物联网**（**IoT**）的基础知识

+   **消息队列遥测传输**（**MQTT**）协议

+   设置 MQTT 代理

+   制作基于物联网的入侵检测器

+   控制家庭

# 物联网的基础知识

在本章中，我们将使用智能手机控制家里的设备，但在这之前，我们应该了解这项技术的基础知识。本章的第一个主题是物联网——现代世界中被滥用的行话。这是每个人都想了解但却没有人知道的东西。物联网可以与一种技术相关联，你的冰箱会告诉你哪些物品供应不足，并会自动为你订购。可怜的东西！这项技术还需要一些时间来进入我们的家。但物联网不仅仅意味着这个。物联网是一个非常广泛的术语，几乎可以应用于所有的地方进行优化。那么，物联网是什么呢？

让我们来解释一下这个缩写，**物联网**，有时也被称为网络物理系统。那么，什么是**物**？在这里，任何有能力在没有人类干预的情况下收集或接收数据的电子物体都可以被称为物。因此，这个物可以是你的手机、心脏起搏器、健康监测设备等等。唯一的*条件*是它必须连接到互联网并具有收集和/或接收数据的能力。第二个术语是**互联网**；互联网指的是互联网，废话！现在，所有这些物联网设备都会向云端或中央计算机发送和接收数据。它之所以这样做，是因为任何物联网设备，无论大小，都被认为是资源受限的环境。也就是说，资源，比如计算能力，要少得多。这是因为物联网设备必须简单和便宜。想象一下，你必须在所有的路灯上安装物联网传感器来监控交通。如果设备的成本是 500 美元，那么安装这种设备是不切实际的。然而，如果它可以做到 5-10 美元，那么没有人会在意。这就是物联网设备的问题；它们非常便宜。现在，这个故事的另一面是，它们没有很多计算能力。因此，为了平衡这个方程，它们不是在自己的处理器上计算原始数据，而是将这些数据简单地发送到云计算设备或者服务器，这些数据在那里被计算，得出有意义的结果。所以，这样就解决了我们所有的问题。嗯，不是！这些设备的第二个问题是它们也可以是电池操作的一次性设备。例如，在森林的各个地方安装了温度传感器；在这种情况下，没有人会每周去更换电池。因此，这些设备是这样制作的，它们消耗很少甚至几乎没有电力，从而使编程变得非常棘手。

现在我们已经了解了物联网的概念，在本章中，我们将使我们的家居具备物联网功能。这意味着，我们将能够从家中的传感器接收和收集数据，在我们的移动设备上查看数据，并且如果需要，我们也可以使用智能手机控制设备。不过有一点，我们不会在云端进行计算，而是简单地将所有数据上传到云端，只需访问该数据或将我们的数据发送到云端，然后可以访问。我们将在另一本书中讨论云计算方面，因为这可能是一个全新的维度，超出了本书的范围。

# MQTT 协议

MQTT 是 ISO 认证的协议，被广泛使用。这个协议的有趣之处在于，它是由 Andy Stanford 和 Arlen Nipper 于 1999 年为监控沙漠中的油管开发的。您可以想象，在沙漠中，他们开发的协议必须是节能和带宽高效的。

这个协议的工作方式非常有趣。它具有发布-订阅架构。这意味着它有一个中央服务器，我们也称之为代理。任何设备都可以向该代理注册并发布任何有意义的数据。现在，被发布的数据应该有一个主题，例如，空气温度。

这些主题特别重要。为什么，您可能会问？对于代理，可以连接一个或多个设备。连接时，它们还需要订阅一个主题。假设它们订阅了主题*Air-*Temperature。现在，每当有新数据到来时，它都会发布到订阅设备。

需要知道的一件重要事情是，与 HTTP 中的请求不同，无需请求来获取代理的数据。相反，每当接收到数据时，它将被推送到订阅该主题的设备。很明显，TCP 协议也将一直处于工作状态，并且与代理相关的端口将始终连接以实现无缝的数据传输。但是，如果数据中断，代理将缓冲所有数据，并在连接恢复时将其发送给订阅者。

![](img/ce8f2b8a-d40d-4856-b691-6970e0a04005.png)

如您所见，运动传感器和温度传感器通过特定主题即**Temperature**和**Motion**向 MQTT 服务器提供数据。订阅这些主题的人将从此设备获取读数。因此，实际传感器和移动设备之间不需要直接通信。

整个架构的好处是，可以连接无限数量的设备，并且不需要任何可扩展性问题。此外，该协议相对简单，即使处理大量数据也很容易。因此，这成为物联网的首选协议，因为它为数据生产者和数据接收者之间提供了一种简单、可扩展和无缝的连接。

# 设置 MQTT 代理

在这个主题中，让我们看看我们需要做什么来设置这个服务器。打开命令行，输入以下命令：

```py
sudo apt-get update
sudo apt-get upgrade
```

一旦更新和升级过程完成，继续安装以下软件包：

```py
sudo apt-get install mosquitto -y
```

这将在您的树莓派上安装 Mosquitto 代理。该代理将负责所有数据传输：

```py
sudo apt-get install mosquitto-clients -y
```

现在，这行将安装客户端软件包。您可以想象，树莓派本身将是代理的客户端。因此，它将处理必要的事情。

我们现在已经安装了软件包；是的，确切地说，就是这么简单。现在，我们需要做的就是配置 Mosquitto 代理。要做到这一点，您需要输入以下命令：

```py
sudo nano etc/mosquitto/mosquitto.conf
```

现在，这个命令将打开保存 Mosquitto 文件配置的文件。要进行配置，您需要到达此文件的末尾，您将看到以下内容：

```py
include_dir/etc/mosquitto/conf.d
```

现在，您可以通过在这些行之前添加`#`来注释掉前面的代码行。完成后，继续添加以下行：

```py
allow_anonymous false

password_file /etc/mosquitto/pwfile

listener 1883
```

让我们看看我们在这里做了什么。`allow_anonymous false`这一行告诉经纪人不是每个人都可以访问数据。接下来的一行，`password_file /etc/mosquitto/pwfile`告诉经纪人密码文件的位置，位于`/etc/mosquitto/pwfile`。最后，我们将使用`listener 1883`命令定义这个经纪人的端口，即`1883`。

最后，我们已经完成了在树莓派中设置 MQTT 客户端。现在我们准备继续并将其用于物联网启用的家庭。

# 制作基于物联网的入侵检测器

现在树莓派已经设置好，我们准备将其启用物联网，让我们看看我们将如何连接系统到互联网并使其正常工作。首先，我们需要将树莓派连接到我们想使用物联网技术控制的设备。所以继续使用以下图表进行连接：

![](img/7fa73dbb-b3fd-4494-9b69-ac9943103add.png)

一旦您设置好所有的组件，让我们继续上传以下代码：

```py
import time  import paho.mqtt.client as mqtt import RPi.gpio as gpio
pir = 23
gpio.setmode(gpio.BCM)
gpio.setup(pir, gpio.IN)
client = mqtt.Client() broker="broker.hivemq.com" port = 1883
pub_topic = "IntruderDetector_Home" def SendData():
  client.publish(pub_topic,"WARNING : SOMEONE DETECTED AT YOUR PLACE")   def on_connect(client, userdata, flag,rc):
  print("connection returned" + str(rc))   SendData() while True:
 client.connect(broker,port) client.on_connect = on_connect   if gpio.output(pir) == gpio.HIGH :
    SendData() client.loop_forever() 
```

与迄今为止我们看到的其他代码块不同，这段代码对你来说可能会很新。所以我将解释除一些明显的部分之外的每个部分。所以，让我们看看我们在这里有什么：

```py
import paho.mqtt.client as mqtt
```

在这部分，我们将`pho.mqtt.client`库导入为`mqtt`。所以每当需要访问这个库时，我们只需要使用`mqtt`这一行，而不是整个库的名称。

```py
client = mqtt.Client()
```

我们使用`mqtt`库的`client`方法定义了一个客户端。这可以通过`client`变量来调用。

```py
broker="broker.hivemq.com"
```

所以我们正在程序中定义经纪人。对于这个程序，我们使用的经纪人是`broker.hivemq.com`，它为我们提供了经纪人服务。

```py
port = 1883
```

现在，我们将再次定义协议将工作的端口，即在我们的情况下是`1883`。

```py
pub_topic = "IntuderDetector_Home"
```

在这里，我们定义了名为`pub_topic`的变量的值，即`IntruderDetector_Home`。这将是在代码运行时可以订阅的最终主题。

```py
def SendData():
 client.publish(pub.topic, "WARNING : SOMEONE DETECTED AT YOUR PLACE")
```

在这里，我们定义了一个名为`SendData()`的函数，将数据`Warning : SOMEONE DETECTED AT YOUR PLACE`发布到我们之前声明的主题的经纪人。

```py
def on_message(client, userdata, message):
  print('message is : ')
 print(str(message.payload)) 
```

在这一行中，我们定义了一个名为`on_message()`的函数，它将打印一个值`message is :`，后面跟着数据是什么。这将使用`print(str(message.payload))`这一行来完成。它的作用是打印传递给函数参数的任何内容。

```py
 def on_connect(client, userdata, flag,rc):

     print("connection returned" + str(rc)) 
  SendData()
```

在这一行中，我们定义了`on_connect()`函数，它将打印`connection returned`一行，后面跟着`rc`的值。`rc`代表返回码。所以，每当消息被传递时，都会生成一个代码，即使没有，也会返回特定的代码通知错误。所以，可以将其视为确认。完成后，我们之前定义的`SendData()`函数将用于将数据发送到经纪人。

```py
client.connect(broker,port)
```

`connect()`是 MQTT 库的一个函数，它将客户端连接到经纪人。这很简单。我们只需要传递我们想要连接的经纪人的参数和要使用的端口。在我们的情况下，`broker = broker.hivemq.com`和`port = 1883`。所以当我们调用这个函数时，树莓派就连接到我们的经纪人了。

```py
client.on_connect = on_connect 
```

这是程序的核心。`client.on_connect`函数所做的是，每当树莓派连接到经纪人时，它就开始执行我们定义的`on_connect`函数。这将连续不断地将数据发送到经纪人，每隔 5 秒一次，就像我们在函数中定义的方式一样。这个过程也被称为回调，它使其成为事件驱动。也就是说，如果它没有连接，它就不会尝试将数据发送到经纪人。

```py
  if gpio.output(pir) == HIGH :
        sendData()
```

当 PIR 传感器变高或者检测到运动时，将调用`sendData()`函数，消息将被发送到代理，警告有人在你的地方被探测到。

```py
client.loop_forever()
```

这是我最喜欢的功能，特别是因为它有可爱的名字。正如你所期望的，`client.loop_forver()`函数将继续寻找任何事件，每当检测到事件时，它将触发数据发送到代理。现在我们将看到这些数据的部分。为此，我们需要从 App Store（如果你使用 iOS）或 Playstore（如果你使用 android）下载*MyMQTT*应用程序。

![](img/3fb1bc6b-f76b-420f-8552-dd40ca79ea66.jpeg)

一旦你启动应用程序，你将看到上面的屏幕。你需要填写代理 URL 的名称，在我们的例子中是`broker.hivemq.com`。然后，填写端口，在我们的例子中是`1883`。

完成后，你将看到一个类似以下的屏幕：

![](img/d328cb27-6910-4078-8afc-5119e25e41df.jpeg)

只需添加你需要的订阅名称，即`IntruderDetector_Home`。完成后，你将看到魔法发生！

在下一节中，我们将基于物联网来控制事物；到时见。

# 控制家庭

最后，使用以下图表进行连接并上传以下代码：

![](img/1fe8d775-6898-4fa5-b7bf-e20e87071fcc.png)

```py
import time
import paho.mqtt.client as paho
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(14,GPIO.OUT)
broker="broker.hivemq.com"
sub_topic = light/control
client = paho.Client()
def on_message(client, userdata, message):
    print('message is : ')
    print(str(message.payload))
    data = str(message.payload)
    if data == "on":
        GPIO.output(3,GPIO.HIGH)
    elif data == "off":
        GPIO.output(3,GPIO.LOW)

def on_connect(client,userdata, flag, rc):
    print("connection returned" + str(rc))
    client.subscribe(sub_topic)
client.connect(broker,port)
client.on_connect = on_connect
client.on_message=on_message
client.loop_forever()
```

现在，在这段代码中，我没有太多需要告诉你的；它非常直接了当。我们发送数据就像上次一样。然而，这次我们使用了一个新的函数。所以，让我们看看这段代码到底是什么：

```py
def on_message(client, userdata, message):
       print('message is : ')
 print(str(message.payload)) data = str(message.payload) if data == "on": GPIO.output(3,GPIO.HIGH) elif data == "off": GPIO.output(3,GPIO.LOW)
```

在这里，我们定义了`on_message()`函数在做什么。函数有三个参数，消息将在这些参数上工作。这包括`client`，我们之前已经声明过；`userdata`，我们现在没有使用；最后是`message`，我们将通过智能手机通过互联网发送。

一旦你查看程序内部，这个函数将使用`print('message is : ')`和`print(str(message.payload))`来打印消息。完成后，`data`的值将被设置为订阅者发送的消息。

这些数据将由我们的条件来评估。如果数据保持`on`，那么 GPIO 端口号`3`将被设置为`HIGH`，如果字符串是`off`，那么 GPIO 端口号`3`将被设置为`LOW`—简单来说，打开或关闭你的设备。

```py
def on_connect(client,userdata, flag, rc):
    print("connection returned" + str(rc))
    client.subscribe(sub_topic)
```

我们之前也定义了`on_connect()`函数。然而，这次有些不同。我们不仅打印连接返回的值`rc`，还使用了另一个名为`client.subscribe(sub_topic)`的函数，它将让我们在程序中之前定义的特定主题上连接到代理。

```py
client.on_message=on_message
```

由于整个算法是基于事件驱动系统，这个`client.on_message`函数将一直等待接收消息。一旦接收到，它将执行`on_message`函数。这将决定是否打开或关闭设备。

要使用它，只需继续发送基于主题的数据，它将被你的树莓派接收。

![](img/88174504-1729-4472-8917-f6a4f2ac0ab3.jpeg)

一旦接收到，决策函数`on_message()`将决定 MyMQTT 应用程序接收到了什么数据。如果接收到的数据是`on`，那么灯将被打开。如果接收到的数据是`off`，那么灯将被关闭。就是这么简单。

# 总结

在本章中，我们已经了解了物联网的基础知识以及 MQTT 服务器的工作原理。我们还制作了一个入侵者检测系统，无论你身在何处，只要有人进入你的家，它都会提醒你。最后，我们还创建了一个系统，可以通过简单的手机命令打开家中的设备。在下一章中，我们将让贾维斯能够让你根据你的声音与系统交互。
