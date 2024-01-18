# 构建JavaScript客户端

让我们面对现实吧。如果没有互联网，我们真的不会有物联网。JavaScript，连同HTML和CSS，是互联网的核心技术之一。物联网的核心是设备之间通信的协议MQTT。

在这一章中，我们将把注意力从Python转移到使用JavaScript构建JavaScript客户端以订阅MQTT服务器上的主题。

本章将涵盖以下主题：

+   介绍JavaScript云库

+   使用JavaScript连接到云服务

# 项目概述

我们将从创建一个简单的JavaScript客户端开始这一章，该客户端连接到MQTT Broker（服务器）。我们将向MQTT Broker发送一条测试消息，然后让该消息返回到我们创建JavaScript客户端的同一页。然后我们将从Raspberry Pi发布一条消息到我们的MQTT Broker。

完成本章应该需要几个小时。

# 入门

要完成这个项目，需要以下内容：

+   Raspberry Pi 3型号（2015年或更新型号）

+   USB电源适配器

+   计算机显示器

+   USB键盘

+   USB鼠标

+   用于编写和执行JavaScript客户端程序的单独计算机

# 介绍JavaScript云库

让我们首先介绍一下JavaScript云库的背景。JavaScript自互联网诞生以来就存在（1995年，举例而言）。它已经成为一种可以将HTML网页转变为完全功能的桌面等效应用程序的语言。就我个人而言，我发现JavaScript是最有用的编程语言之一（当然，除了Python）。

JavaScript于1995年发布，旨在与当时最流行的网络浏览器Netscape Navigator一起使用。它最初被称为livescript，但由于在Netscape Navigator浏览器中使用和支持Java，名称被更改为JavaScript。尽管语法相似，但Java和JavaScript实际上与彼此无关——这是一个令人困惑的事实，直到今天仍然存在。

# 谷歌云

通过`google-api-javascript-client`，我们可以访问谷歌云服务。具体来说，我们可以访问谷歌计算引擎，这是谷歌云平台的一个组件。通过谷歌计算引擎，我们可以通过按需虚拟机访问运行Gmail、YouTube、谷歌搜索引擎和其他谷歌服务的基础设施。如果这听起来像是能让你的朋友印象深刻的技术术语，你可能需要更深入地了解这个JavaScript库。您可以在这里了解更多关于`google-api-javascript-client`的信息：[https://cloud.google.com/compute/docs/tutorials/javascript-guide](https://cloud.google.com/compute/docs/tutorials/javascript-guide)。

# AWS SDK for JavaScript

AWS SDK for JavaScript in Node.js提供了AWS服务的JavaScript对象。这些服务包括Amazon S3、Amazon EC2、Amazon SWF和DynamoDB。此库使用Node.js运行时环境。您可以在这里了解更多关于这个库的信息：[https://aws.amazon.com/sdk-for-node-js/](https://aws.amazon.com/sdk-for-node-js/)。

Node.js于2009年5月发布。最初的作者是Ryan Dhal，目前由Joyent公司开发。Node.js允许在浏览器之外执行JavaScript代码，从而使其成为一种JavaScript无处不在的技术。这使JavaScript可以在服务器端和客户端用于Web应用程序。

# Eclipse Paho JavaScript客户端

Eclipse Paho JavaScript客户端库是一个面向JavaScript客户端的MQTT基于浏览器的库。Paho本身是用JavaScript编写的，可以轻松地插入到Web应用程序项目中。Eclipse Paho JavaScript客户端库使用Web套接字连接到MQTT Broker。我们将在本章的项目中使用这个库。

# 使用JavaScript连接到云服务

对于我们的项目，我们将构建一个JavaScript客户端并将其连接到MQTT Broker。我们将**发布**和**订阅**名为**test**的**topic**。然后，我们将在树莓派上编写一个小的简单程序来发布到名为test的主题。这段代码将演示使用MQTT发送和接收消息是多么容易。

请查看以下图表，了解我们将通过此项目实现的内容：

![](assets/84267fcf-dc03-4879-a939-a007bd125ecb.png)

# 设置CloudMQTT帐户

第一步是设置MQTT Broker。我们可以通过在本地安装Mosquitto平台（[www.mosquitto.org](http://www.mosquitto.org)）来完成此操作。相反，我们将使用网站[www.cloudmqtt.com](http://www.cloudmqtt.com)设置基于云的MQTT Broker。

要设置帐户：

1.  在浏览器中，导航到[www.cloudmqtt.com.](http://www.cloudmqtt.com)

1.  在右上角点击登录。

1.  在创建帐户框中，输入您的电子邮件地址：

![](assets/f04e69e9-3f08-4ba7-a01f-2c681a01a8e3.png)

1.  您将收到一封发送到该电子邮件地址的电子邮件，要求您确认。您可以通过单击电子邮件中的确认电子邮件按钮来完成确认过程。

1.  然后您将进入一个页面，需要输入密码。选择密码，确认密码，然后按提交：

![](assets/4078ed85-99ff-471d-9ca8-90f36549b436.png)

1.  然后您将进入实例页面。这是我们将创建MQTT Broker实例以发送和发布MQTT消息的地方。

# 设置MQTT Broker实例

现在我们已经设置了CloudMQTT帐户，是时候创建一个用于我们应用程序的实例了：

1.  从实例页面，单击标有创建新实例的大绿色按钮。

1.  您将看到以下页面：

![](assets/01a45002-ed39-4088-981a-c57dfa9a50a1.png)

1.  在名称框中，输入`T.A.R.A.S`（我们将将MQTT Broker实例命名为此，因为我们将考虑此Broker是T.A.R.A.S机器人汽车的一部分）。

1.  在计划下拉菜单中，选择Cute Cat（这是用于开发目的的免费选项）。

1.  点击绿色的选择区域按钮。

1.  根据您所在的世界位置，选择一个靠近您地理位置的区域。由于我位于加拿大，我将选择US-East-1（北弗吉尼亚）：

![](assets/d41391c9-48ce-4cfb-8894-5732fe6f80e0.png)

1.  点击绿色的确认按钮。

1.  您将看到确认新实例页面。在点击绿色的确认实例按钮之前，请查看此信息：

![](assets/62dd7a70-3784-465c-a9ca-9143c8705e4c.png)

1.  您应该看到T.A.R.A.S实例在列表中的实例列表中：

![](assets/0e5c8309-381e-4d4a-8fbb-515b4ef2a5f9.png)

# 编写JavaScript客户端代码

这是我在我的帐户上设置的T.A.R.A.S实例的屏幕截图。请注意列表中的值。这些值来自我的实例，您的值将不同。我们将在编写JavaScript客户端时使用这些值：

![](assets/5519d33b-365e-40d2-8ef0-02657aef5ef6.png)

要编写我们的JavaScript客户端代码，我们应该使用T.A.R.A.S上的树莓派以外的计算机。您可以使用任何您喜欢的操作系统和HTML编辑器。我使用macOS和Visual Studio Code编写了我的JavaScript客户端代码。您还需要Paho JavaScript库：

1.  转到Eclipse Paho下载站点[https://projects.eclipse.org/projects/technology.paho/downloads](https://projects.eclipse.org/projects/technology.paho/downloads)。

1.  点击JavaScript客户端链接。它将以`JavaScript客户端`的名称标记，后跟版本号。在撰写本文时，版本号为1.03。

1.  JavaScript客户端库将以`paho.javascript-1.0.3`的ZIP文件形式下载。解压文件。

1.  我们需要在计算机上创建一个用作项目文件夹的文件夹。在计算机上创建一个新文件夹，并将其命名为`MQTT HTML Client`。

1.  在`MQTT HTML Client`文件夹内创建一个名为`scripts`的子文件夹。

1.  将解压后的`paho.javascript-1.0.3`文件夹拖放到`MQTT HTML Client`文件夹中。

1.  `MQTT HTML Client`文件夹内的目录结构应如下所示：

![](assets/5e404183-5f7a-4f40-84e3-e31297a50130.png)

现在，是时候编写代码了。我们将尽可能简化我们的代码，以便更好地理解MQTT如何与JavaScript配合使用。我们的客户端代码将包括两个文件，一个HTML页面和一个`.js`（JavaScript）文件。让我们从创建HTML页面开始：

1.  使用您喜欢的HTML编辑器，创建一个名为`index.html`的文件并保存到项目根目录。

1.  您的`project`文件夹应该如下所示：

![](assets/6a9ccea0-16b5-4bd3-ac42-b289a09df395.png)

1.  在`index.html`文件中输入以下内容：

```py
<!DOCTYPE html>
<html>

<head>
 <title>MQTT Message Client</title>
 <script src="paho.javascript-1.0.3/paho-mqtt.js" type="text/javascript"></script>
 <script src="scripts/index.js" type='text/javascript'></script>
</head>

<body>

 <h2>MQTT Message Client</h2>
 <button onclick="sendTestData()">
 <h4>Send test message</h4>
 </button>

 <button onclick="subscribeTestData()">
 <h4>Subscribe to test</h4>
 </button>

 <div>
 <input type="text" id="messageTxt" value="Waiting for MQTT message" size=34 />
 </div>

</body>

</html>
```

1.  保存对`index.html`的更改。

1.  我们在这里做的是创建一个简单的HTML页面，并导入了两个JavaScript库，Paho JavaScript库和一个名为`index.js`的文件，我们还没有创建：

```py
<script src="paho.javascript-1.0.3/paho-mqtt.js" type="text/javascript"></script>
<script src="scripts/index.js" type='text/javascript'></script>
```

1.  然后，我们需要创建两个按钮；在顶部按钮上，我们将`onclick`方法设置为`sendTestData`。在底部按钮上，我们将`onclick`方法设置为`subscribeTestData`。这些方法将在我们编写的JavaScript文件中创建。为简单起见，我们不给这些按钮分配ID名称，因为我们不会在我们的JavaScript代码中引用它们：

```py
<button onclick="sendTestData()">
        <h4>Send test Message</h4>
</button>
<button onclick="subscribeTestData()">
        <h4>Subscribe to test</h4>
</button>
```

1.  我们在`index.html`页面中将创建的最后一个元素是一个文本框。我们为文本框分配了一个`id`为`messageTxt`和一个值为`Waiting for MQTT message`：

```py
<div>
    <input type="text" id="messageTxt" value="Waiting for MQTT message" size=34 />
</div>
```

1.  如果我们将`index.html`加载到浏览器中，它将如下所示：

![](assets/d3b89900-edd2-4fb5-9e45-d39a24510d8f.png)

# 运行代码

在运行客户端代码之前，我们需要创建一个JavaScript文件，该文件将提供我们需要的功能：

1.  使用HTML编辑器，在我们的项目目录中的`scripts`文件夹中创建一个名为`index.js`的文件并保存。

1.  将以下代码添加到`index.js`并保存。用您的实例中的值替换`Server`、`User`、`Password`和`Websockets Port`（分别显示为`"m10.cloudmqtt.com"`、`38215`、`"vectydkb"`和`"ZpiPufitxnnT"`）：

```py
function sendTestData() {
 client = new Paho.MQTT.Client
 ("m10.cloudmqtt.com", 38215, "web_" + 
 parseInt(Math.random() * 100, 10));

 // set callback handlers
 client.onConnectionLost = onConnectionLost;

 var options = {
 useSSL: true,
 userName: "vectydkb",
 password: "ZpiPufitxnnT",
 onSuccess: sendTestDataMessage,
 onFailure: doFail
 }

 // connect the client
 client.connect(options);
}

// called when the client connects
function sendTestDataMessage() {
 message = new Paho.MQTT.Message("Hello from JavaScript 
 client");
 message.destinationName = "test";
 client.send(message);
}

function doFail() {
 alert("Error!");
}

// called when the client loses its connection
function onConnectionLost(responseObject) {
 if (responseObject.errorCode !== 0) {
 alert("onConnectionLost:" + responseObject.errorMessage);
 }
}

// called when a message arrives
function onMessageArrived(message) {
 document.getElementById('messageTxt').value = message.payloadString; 
}

function onsubsribeTestDataSuccess() {
 client.subscribe("test");
 alert("Subscribed to test");
}

function subscribeTestData() {
 client = new Paho.MQTT.Client
 ("m10.cloudmqtt.com", 38215, "web_" + 
 parseInt(Math.random() * 100, 10));

 // set callback handlers
 client.onConnectionLost = onConnectionLost;
 client.onMessageArrived = onMessageArrived;

 var options = {
 useSSL: true,
 userName: "vectydkb",
 password: "ZpiPufitxnnT",
 onSuccess: onsubsribeTestDataSuccess,
 onFailure: doFail
 }

 // connect the client
 client.connect(options);
}
```

1.  通过刷新加载了`index.html`的浏览器中运行代码。

1.  点击`Subscribe to test`按钮。您应该会收到一个弹出对话框，显示`Subscribed to test`消息。

1.  关闭弹出对话框。

1.  点击发送测试消息按钮。

1.  您应该在文本框中看到消息`Hello from JavaScript client`。

这是我们刚刚执行的某种魔术吗？在某种程度上是。我们刚刚成功订阅了MQTT Broker上的一个主题，然后发布到相同的主题，然后在同一个JavaScript客户端中接收到了一条消息。要从MQTT Broker中观察到这一点，请执行以下操作：

1.  登录到您的CloudMQTT帐户

1.  点击T.A.R.A.S实例

1.  点击WEBSOCKET UI菜单选项

1.  您应该会看到以下对话框，显示您已连接：

![](assets/a3d25e03-1fdd-4109-9f78-44479e68140a.png)

1.  在浏览器的另一个标签或窗口中，导航回JavaScript客户端`index.html`

1.  再次点击发送测试消息按钮

1.  返回CloudMQTT页面

1.  在接收到的消息列表下，您应该看到一条消息：

![](assets/50628610-fba2-47f4-a0db-e16acfd31ad2.png)

1.  点击发送测试消息按钮几次，您应该会在接收到的消息下看到相同消息的列表。

# 理解JavaScript代码

在为树莓派编写代码之前，让我们先看一下`index.js`中的JavaScript代码。

我们首先来看订阅代码。我们用来从我们的MQTT Broker订阅主题的两种方法是`subscribeTestData`和`onsubsribeTestDataSuccess`。`subscribeTestData`创建了一个名为`client`的Paho MQTT客户端对象。它使用`client`对象通过实例化对象与我们的MQTT Broker连接，并使用`Server`和`Websockets Port`值（为简单起见，我在代码中留下了我的帐户中的值）：

```py
function subscribeTestData() {
    client = new Paho.MQTT.Client
        ("m10.cloudmqtt.com", 38215, "web_" +     
                        parseInt(Math.random() * 100, 10));

    // set callback handlers
    client.onConnectionLost = onConnectionLost;
    client.onMessageArrived = onMessageArrived;

    var options = {
        useSSL: true,
        userName: "vectydkb",
        password: "ZpiPufitxnnT",
        onSuccess: onsubsribeTestDataSuccess,
        onFailure: doFail
    }

    // connect the client
    client.connect(options);
}
```

然后，我们使用“client.onConnectionLost”和“client.onMessageArrived”设置回调处理程序。回调处理程序将我们JavaScript代码中的函数与我们的“client”对象的事件相关联。在这种情况下，当与MQTT代理的连接丢失或从MQTT代理接收到消息时。 “options”变量将SSL的使用设置为“true”，设置“User”和“Password”设置，然后将成功连接的条件设置为“onsubsribeTestDataSuccess”方法，将连接尝试不成功的条件设置为“doFail”方法。然后，我们通过传递我们的“options”变量通过“client.connect”方法连接到我们的MQTT代理。

当成功连接到MQTT代理时，将调用“onsubsribeTestDataSuccess”方法。它设置“client”对象以订阅“test”主题。然后，它创建一个带有消息“Subscribed to test”的警报：

```py
function onsubsribeTestDataSuccess() {
    client.subscribe("test");
    alert("Subscribed to test");
}
```

如果与客户端的连接不成功，则调用“doFail”方法。它只是创建一个带有消息“错误！”的弹出警报：

```py
function doFail() {
    alert("Error!");
}
```

现在我们了解了订阅“test”主题的代码，让我们看一下发布到“test”主题的代码。

“sendTestData”函数与“subscribeTestData”函数非常相似：

```py
function sendTestData() {
    client = new Paho.MQTT.Client
        ("m10.cloudmqtt.com", 38215, "web_" + parseInt(Math.random() * 100, 10));

    // set callback handlers
    client.onConnectionLost = onConnectionLost;

    var options = {
        useSSL: true,
        userName: "vectydkb",
        password: "ZpiPufitxnnT",
        onSuccess: sendTestDataMessage,
        onFailure: doFail
    }

    // connect the client
    client.connect(options);
}
```

创建了一个名为“client”的Paho MQTT客户端对象，其参数与“subscribeTestData”函数中使用的参数相同。设置的唯一回调处理程序是“onConnectionLost”。我们没有设置“onMessageArrived”，因为我们正在发送消息而不是接收消息。将“options”变量设置为与“subscribeTestData”函数中使用的相同值，唯一的例外是将“onSuccess”分配给“sendTestDataMessage”函数。

“sendTestDataMessage”函数创建一个新的Paho MQTT消息对象，其值为“Hello from JavaScript client”，并将其命名为“message”。 “destinationName”是我们为其创建消息的主题，设置为“test”值。然后，我们使用“client.send”发送消息：

```py
function sendTestDataMessage() {
    message = new Paho.MQTT.Message("Hello from JavaScript client");
    message.destinationName = "test";
    client.send(message);
}
```

“onConnectionLost”函数用于订阅和发布，并简单地创建一个带有来自JavaScript响应对象的错误消息的警报弹出窗口：

```py
// called when the client loses its connection
function onConnectionLost(responseObject) {
    if (responseObject.errorCode !== 0) {
        alert("onConnectionLost:" + responseObject.errorMessage);
    }
}
```

既然我们的JavaScript客户端已经订阅并发布到我们的MQTT代理，让我们让树莓派也参与其中。

# 从我们的树莓派发布MQTT消息

让我们返回到我们的树莓派（如果您一直在使用另一台计算机），并编写一些代码与我们的MQTT代理进行通信：

1.  从应用程序菜单 | 编程 | Thonny Python IDE打开Thonny。

1.  单击“新建”图标创建一个新文件。

1.  在文件中输入以下内容：

```py
import paho.mqtt.client as mqtt
from time import sleep

mqttc = mqtt.Client()
mqttc.username_pw_set("vectydkb", "ZpiPufitxnnT")
mqttc.connect('m10.cloudmqtt.com', 18215)

while True:
    try:
        mqttc.publish("test", "Hello from Raspberry Pi")
    except:
        print("Could not send message!")
    sleep(10)

```

1.  将文件保存为“CloudMQTT-example.py”并运行它。

1.  返回到CloudMQTT页面。您应该看到来自树莓派的消息：

！[](assets/3e8afc3a-0e68-4a3f-a61c-120c53b71bc9.png)

1.  导航到我们的JavaScript客户端“index.html”。您应该在文本框中看到消息“Hello from the Raspberry Pi”（如果您没有看到消息，请刷新页面并再次单击“Subscribe to test”）：

！[](assets/3ce2d3c4-6320-486b-9e96-6c57db5fcb98.png)

树莓派Python代码故意保持简单，以便可以理解这些概念。我们通过导入所需的库来启动代码。然后，我们创建一个名为“mqttc”的MQTT客户端对象。使用“username_pw_set”方法设置用户名和密码。然后，我们使用“connect”方法连接到MQTT代理，通过传递“Server”和“Port”值（我们为Python客户端使用“Port”而不是“Websockets Port”）。在一个连续的循环内，我们通过传递主题“test”和消息“Hello from Raspberry Pi”来通过“publish”方法发布到MQTT代理。

# 摘要

在本章中，我们在使用JavaScript创建MQTT客户端之前探索了JavaScript库。我们设置了一个基于云的MQTT代理，并能够使用我们的JavaScript客户端和树莓派上的Python程序发布和订阅消息。

在[第18章](74462726-806d-4214-8818-17f4627477c3.xhtml)中，*将所有内容放在一起*，我们将扩展本章学到的知识，并构建一个可以通过互联网控制T.A.R.A.S的JavaScript客户端。

# 问题

1.  我们可以使用哪个程序（平台）在本地安装MQTT Broker？

1.  JavaScript和Java是相同的技术，是真是假？

1.  我们可以使用JavaScript来创建一个MQTT客户端吗？

1.  我们可以使用`google-api-javascript-client`库来访问哪些谷歌服务？

1.  MQTT是物联网中使用的协议，是真是假？

1.  JavaScript Node.js技术允许您做什么？

1.  Python可以用于开发MQTT客户端，是真是假？

1.  我们可以通过使用脚本标签将外部JavaScript库的功能添加到我们的网页中，是真是假？

1.  我们如何在JavaScript代码中为我们的MQTT客户端设置用户名和密码？

1.  我们可以在Cloud MQTT应用程序中查看我们发布的消息吗？

# 进一步阅读

有关使用基于云的MQTT Broker的更多信息，请参阅[https://www.cloudmqtt.com/docs.html](https://www.cloudmqtt.com/docs.html)。
