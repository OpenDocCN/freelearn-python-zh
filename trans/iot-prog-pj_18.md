# 第十八章：将所有内容放在一起

对于我们的最后一步，我们将让 T.A.R.A.S 响应使用 JavaScript 客户端发送的 MQTT 控制信号。我们将通过修改到目前为止编写的代码来实现这一点。如果您从头开始阅读本书，感谢您的毅力。这是一个漫长的旅程。我们终于做到了。在本章结束时，我们将完成构建物联网设备的终极目标，即一个可以通过互联网控制的机器人车。

系好安全带（双关语）-是时候将 T.A.R.A.S 提升到下一个级别了。

在本章中，我们将涵盖以下主题：

+   构建一个 JavaScript 客户端以连接到我们的树莓派

+   JavaScript 客户端以访问我们的机器人车的感知数据

+   增强我们的 JavaScript 客户端以控制我们的机器人车

# 项目概述

在本章中，我们将 T.A.R.A.S 连接到 MQTT 代理。通过 MQTT 消息，我们将控制 T.A.R.A.S 的移动，并从 T.A.R.A.S 的距离传感器中读取信息。以下是我们将要构建的图表：

![](img/81f1811c-9a9e-42f9-b355-e13691cbb16d.png)

我们将首先编写 HTML JavaScript 客户端（在图表中显示为**HTML 客户端**），并使用它发送和接收 MQTT 消息。然后，我们将把注意力转向编写 T.A.R.A.S 上的代码，以从相同的 MQTT 代理接收和发送消息。我们将使用这些消息来使用浏览器控制 T.A.R.A.S。最后，我们还将使用浏览器从 T.A.R.A.S 实时传输视频。

完成此项目应该需要半天的时间。

# 入门

要完成此项目，需要以下内容：

+   一个树莓派 3 型（2015 年或更新型号）

+   一个 USB 电源适配器

+   一个计算机显示器

+   一个 USB 键盘

+   一个 USB 鼠标

+   一个 T.A.R.A.S 机器人车

# 构建一个 JavaScript 客户端以连接到我们的树莓派

以下是我们将构建的 HTML JavaScript 客户端的屏幕截图，用于通过网络控制 T.A.R.A.S。HTML JavaScript 客户端可能不会赢得任何设计奖，但它将作为一个优秀的学习平台，用于通过互联网发送机器人控制信息。

![](img/03b0f1b6-f68b-4d53-9ada-0d5ad736a6c1.png)

大紫色按钮用于向 T.A.R.A.S 发送“前进”和“后退”命令。较小的绿色按钮向 T.A.R.A.S 发送“左转”和“右转”控制信息。底部的小银色按钮允许我们使用 T.A.R.A.S 的摄像头拍照，触发 T.A.R.A.S 的警报，并让 T.A.R.A.S 跳舞。`跟踪距离`按钮将 HTML JavaScript 客户端连接到 T.A.R.A.S 传来的距离信息。

在我们为树莓派构建 Python MQTT 客户端之前，我们将使用 CloudMQTT 仪表板跟踪控制信息。

# 编写 HTML 代码

我们将首先为我们的 HTML JavaScript 客户端编写 HTML 代码。您可以使用树莓派以外的计算机：

1.  在您的计算机上创建一个名为`HTML JavaScript Client`的`project`文件夹

1.  从第十七章中复制 Paho JavaScript 库，*构建 JavaScript 客户端*，到`project`文件夹中

1.  使用您喜欢的 HTML 编辑器，创建一个名为`index.html`的文件，并将其保存在*步骤 1*中创建的文件夹中

1.  将以下内容输入到`index.html`中，然后再次保存：

```py
<html>
    <head>
        <title>T.A.R.A.S Robot Car Control</title>
        <script src="paho.javascript-1.0.3/paho-mqtt.js" 
                        type="text/javascript"></script>        
        <script src="scripts/index.js"        
                        type='text/javascript'></script>            

        <link rel="stylesheet" href="styles/styles.css">        
    </head>
    <body>
        <h2>T.A.R.A.S Robot Car Control</h2>
        <div>
            <button onclick="moveForward()" 
                            class="big_button">    
                <h4>Forward</h4>
            </button>
        </div>
        <div>
            <button onclick="turnLeft()" 
                            class="small_button">
                <h4>Turn Left</h4>
            </button>
            <button onclick="turnRight()" 
                            class="small_button">
                <h4>Turn Right</h4>
            </button>
        </div>
        <div>
            <button onclick="moveBackward()" 
                                class="big_button">        
                <h4>Backwards</h4>
            </button>
        </div>
        <div>
            <button onclick="takePicture()" 
                            class="distance_button">        
                <h4>Take Picture</h4>
            </button>
            <button onclick="TARASAlarm()" 
                            class="distance_button">        
                <h4>T.A.R.A.S Alarm</h4>
            </button>
            <button onclick="makeTARASDance()" 
                            class="distance_button">        
                <h4>T.A.R.A.S Dance</h4>
            </button>
            <button onclick="subscribeDistanceData()" 
                            class="distance_button">
                <h4>Track Distance</h4>
            </button>
            <input type="text" id="messageTxt" value="0" 
                            size=34 class="distance" />        
        </div>
    </body>
</html>
```

在我们可以在浏览器中查看`index.html`之前，我们必须为样式创建一个`.css`文件。我们还将为我们的 JavaScript 文件创建一个文件夹：

1.  在您的`project`文件夹中，创建一个新文件夹，并将其命名为`styles`

1.  在`project`文件夹中创建另一个文件夹，并将其命名为`scripts`

1.  您的`project`目录应该与以下内容相同：

![](img/285b49bd-af91-4b65-a64a-2e12ece753d9.png)

1.  在`styles`文件夹中，使用 HTML 编辑器创建一个名为`styles.css`的文件

1.  将以下内容输入到`styles.css`文件中，然后保存：

```py
.big_button {
    background-color: rgb(86, 76, 175);
    border: none;
    color: white;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    width: 400px;
}
.small_button {
    background-color: rgb(140, 175, 76);
    border: none;
    color: white;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    width: 195px;
}
.distance_button {
    background-color: rgb(192, 192, 192);
    border: none;
    color: white;
    padding: 1px 1px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 10px;
    margin: 2px 2px;
    cursor: pointer;
    width: 60px;
}
.distance {
    background-color: rgb(255, 255, 255);
    border: none;
    color: rgb(192,192,192);
    padding: 1px 1px;
    text-align: top;
    text-decoration: none;
    display: inline-block;
    font-size: 20px;
    margin: 2px 2px;
    cursor: pointer;
    width: 300px;
}
```

1.  打开浏览器，导航到`project`文件夹中的`index.html`文件

1.  您应该看到 T.A.R.A.S 机器人车控制仪表板

在添加 JavaScript 代码之前，让我们看一下我们刚刚写的内容。我们将从导入我们需要的资源开始。我们需要 Paho MQTT 库、一个`index.js`文件（我们还没有写），以及我们的`styles.css`文件。

```py
<script  src="paho.javascript-1.0.3/paho-mqtt.js"  type="text/javascript"></script> <script  src="scripts/index.js"  type='text/javascript'></script> <link  rel="stylesheet"  href="styles/styles.css"> 
```

然后，我们将创建一系列按钮，将这些按钮与我们即将编写的`index.js` JavaScript 文件中的函数绑定：

```py
<div>
 <button  onclick="moveForward()"  class="big_button"> <h4>Forward</h4> </button> </div>
```

由于我们的按钮几乎相似，我们只讨论第一个按钮。第一个按钮通过`onclick`属性绑定到我们 JavaScript 文件中的`moveForward`函数。按钮的样式通过将`class`分配给`big_button`来设置。我们使用第一个按钮来向前移动 T.A.R.A.S。

# 编写与我们的 MQTT 代理通信的 JavaScript 代码

现在我们有了 HTML 和 CSS 文件，让我们创建一个 JavaScript 文件，让 MQTT 的魔力发生：

1.  在`scripts`文件夹中，使用 HTML 编辑器创建一个名为`index.js`的文件。

1.  在`index.js`文件中输入以下内容并保存：

```py
function moveForward() {
    client = new Paho.MQTT.Client("m10.cloudmqtt.com", 38215, "web_" + parseInt(Math.random() * 100, 10));

    // set callback handlers
    client.onConnectionLost = onConnectionLost;
    var options = {
        useSSL: true,
        userName: "vectydkb",
        password: "ZpiPufitxnnT",
        onSuccess: sendMoveForwardMessage,
        onFailure: doFail
    }

    // connect the client
    client.connect(options);
}

// called when the client connects
function sendMoveForwardMessage() {
    message = new Paho.MQTT.Message("Forward");
    message.destinationName = "RobotControl";
    client.send(message);
}

function moveBackward() {
    client = new Paho.MQTT.Client("m10.cloudmqtt.com", 38215, "web_" + parseInt(Math.random() * 100, 10));

    // set callback handlers
    client.onConnectionLost = onConnectionLost;
    var options = {
        useSSL: true,
        userName: "vectydkb",
        password: "ZpiPufitxnnT",
        onSuccess: sendMoveBackwardMessage,
        onFailure: doFail
    }

    // connect the client
    client.connect(options);
}

// called when the client connects
function sendMoveBackwardMessage() {
    message = new Paho.MQTT.Message("Backward");
    message.destinationName = "RobotControl";
    client.send(message);
}

function turnLeft() {
    client = new Paho.MQTT.Client("m10.cloudmqtt.com", 38215, "web_" + parseInt(Math.random() * 100, 10));

    // set callback handlers
    client.onConnectionLost = onConnectionLost;
    var options = {
        useSSL: true,
        userName: "vectydkb",
        password: "ZpiPufitxnnT",
        onSuccess: sendTurnLeftMessage,
        onFailure: doFail
    }

    // connect the client
    client.connect(options);
}

// called when the client connects
function sendTurnLeftMessage() {
    message = new Paho.MQTT.Message("Left");
    message.destinationName = "RobotControl";
    client.send(message);
}

function turnRight() {
    client = new Paho.MQTT.Client("m10.cloudmqtt.com", 38215, "web_" + parseInt(Math.random() * 100, 10));

    // set callback handlers
    client.onConnectionLost = onConnectionLost;
    var options = {
        useSSL: true,
        userName: "vectydkb",
        password: "ZpiPufitxnnT",
        onSuccess: sendTurnRightMessage,
        onFailure: doFail
    }

    // connect the client
    client.connect(options);
}

// called when the client connects
function sendTurnRightMessage() {
    message = new Paho.MQTT.Message("Right");
    message.destinationName = "RobotControl";
    client.send(message);
}

function takePicture() {
    client = new Paho.MQTT.Client("m10.cloudmqtt.com", 38215, "web_" + parseInt(Math.random() * 100, 10));

    // set callback handlers
    client.onConnectionLost = onConnectionLost;
    var options = {
        useSSL: true,
        userName: "vectydkb",
        password: "ZpiPufitxnnT",
        onSuccess: sendTakePictureMessage,
        onFailure: doFail
    }

    // connect the client
    client.connect(options);
}

// called when the client connects
function sendTakePictureMessage() {
    message = new Paho.MQTT.Message("Picture");
    message.destinationName = "RobotControl";
    client.send(message);
}

function TARASAlarm() {
    client = new Paho.MQTT.Client("m10.cloudmqtt.com", 38215, "web_" + parseInt(Math.random() * 100, 10));

    // set callback handlers
    client.onConnectionLost = onConnectionLost;
    var options = {
        useSSL: true,
        userName: "vectydkb",
        password: "ZpiPufitxnnT",
        onSuccess: sendTARASAlarmMessage,
        onFailure: doFail
    }

    // connect the client
    client.connect(options);
}

// called when the client connects
function sendTARASAlarmMessage() {
    message = new Paho.MQTT.Message("Alarm");
    message.destinationName = "RobotControl";
    client.send(message);
}

function makeTARASDance() {
    client = new Paho.MQTT.Client("m10.cloudmqtt.com", 38215, "web_" + parseInt(Math.random() * 100, 10));

    // set callback handlers
    client.onConnectionLost = onConnectionLost;
    var options = {
        useSSL: true,
        userName: "vectydkb",
        password: "ZpiPufitxnnT",
        onSuccess: makeTARASDanceMessage,
        onFailure: doFail
    }

    // connect the client
    client.connect(options);
}

// called when the client connects
function makeTARASDanceMessage() {
    message = new Paho.MQTT.Message("Dance");
    message.destinationName = "RobotControl";
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

function onsubsribeDistanceDataSuccess() {
    client.subscribe("distance");
    alert("Subscribed to distance data");
}

function subscribeDistanceData() {
    client = new Paho.MQTT.Client("m10.cloudmqtt.com", 38215, "web_" + parseInt(Math.random() * 100, 10));

    // set callback handlers
    client.onConnectionLost = onConnectionLost;
    client.onMessageArrived = onMessageArrived;
    var options = {
        useSSL: true,
        userName: "vectydkb",
        password: "ZpiPufitxnnT",
        onSuccess: onsubsribeDistanceDataSuccess,
        onFailure: doFail
    }

    // connect the client
    client.connect(options);
}
```

1.  我已经在代码中留下了我的 CloudMQTT 实例的值。就像我们在第十七章中所做的那样，*构建 JavaScript 客户端*，用您实例的值（`服务器`、`Websockets 端口`、`用户名`、`密码`）替换这些值。

1.  在浏览器中导航回到`index.html`并刷新页面。

1.  现在我们已经有了我们的 HTML JavaScript 客户端。我们所做的实质上是修改了第十七章中的`index.js`代码，*构建 JavaScript 客户端*，以便我们可以向我们的 MQTT 代理发送控制消息，最终控制我们的机器人车：

```py
function moveForward() {
    client = new Paho.MQTT.Client("m10.cloudmqtt.com", 38215, "web_" + parseInt(Math.random() * 100, 10));

    // set callback handlers
    client.onConnectionLost = onConnectionLost;
    var options = {
        useSSL: true,
        userName: "vectydkb",
        password: "ZpiPufitxnnT",
        onSuccess: sendMoveForwardMessage,
        onFailure: doFail
    }

    // connect the client
    client.connect(options);
}

// called when the client connects
function sendMoveForwardMessage() {
    message = new Paho.MQTT.Message("Forward");
    message.destinationName = "RobotControl";
    client.send(message);
}
```

我们已经更改了上一个示例中的代码。`moveForward`函数创建了一个名为`client`的 Paho MQTT 客户端，其中包含从我们的 CloudMQTT 实例获取的`服务器`和`Websockets 端口`连接信息。设置了一个回调处理程序来处理连接丢失时的情况，该处理程序设置为`onConnectionLost`函数。使用从我们的 CloudMQTT 实例获取的`userName`和`password`信息创建了`options`变量。我们将成功连接到 MQTT 代理设置为`sendMoveForwardMessage`函数。然后通过传入`options`变量连接到我们的客户端。

`sendMoveForwardMessage`函数创建了一个名为`Forward`的新 Paho MQTT 消息。然后将此消息分配给`RobotControl`主题，并使用我们的 Paho MQTT 客户端对象`client`发送。

发送后退、右转、左转、拍照、触发警报和跳舞的消息的函数以类似的方式编写为`moveForward`函数。

现在我们已经为控制 T.A.R.A.S 在网络上构建了 HTML JavaScript 客户端，让我们使用 CloudMQTT 实例上的`WEBSOCKETS UI`页面进行测试：

1.  导航回到您的 CloudMQTT 帐户。

1.  选择您获取服务器、用户、密码和 Web 套接字端口连接信息的实例（在第十七章中，*构建 JavaScript 客户端*，我们创建了名为`T.A.R.A.S`的实例）。

1.  点击左侧的 WEBSOCKETS UI 菜单选项。您应该在右侧收到一个成功连接的通知。

1.  导航回到`index.html`并点击“前进”按钮。

1.  现在，导航回到您的 CloudMQTT 实例。您应该在“接收到的消息”列表中看到一条新消息：

![](img/d27aef97-918d-4ec1-b8fa-ed685f711ea6.png)

恭喜！您刚刚连接了一个 HTML JavaScript 客户端到一个 MQTT 代理并发送了一条消息。现在我们将在另一台设备上使用完全不同的编程语言开发另一个客户端，然后使用该客户端订阅来自我们的 HTML JavaScript 客户端的消息。

# 创建一个 JavaScript 客户端来访问我们机器人车的感知数据

我们创建的`index.js`文件包含订阅我们的 HTML JavaScript 客户端到`distance`主题的函数：

```py
function subscribeDistanceData() {
    client = new Paho.MQTT.Client("m10.cloudmqtt.com", 38215, "web_" + parseInt(Math.random() * 100, 10));

    // set callback handlers
    client.onConnectionLost = onConnectionLost;
    client.onMessageArrived = onMessageArrived;
    var options = {
        useSSL: true,
        userName: "vectydkb",
        password: "ZpiPufitxnnT",
        onSuccess: onsubsribeDistanceDataSuccess,
        onFailure: doFail
    }

    // connect the client
    client.connect(options);
}

function onsubsribeDistanceDataSuccess() {
    client.subscribe("distance");
    alert("Subscribed to distance data");
}
```

类似于我们在第十七章中编写的代码，*构建 JavaScript 客户端*，`subscribeDistanceData`函数创建了一个 Paho MQTT 客户端，其中包含来自 CloudMQTT 实例的连接信息。成功连接后，将调用`onsubscribeDistanceDataSuccess`函数，该函数将`client`订阅到`distance`主题。

还创建了一个警报，告诉我们 HTML JavaScript 客户端现在已订阅了`distance`主题。

# 编写 T.A.R.A.S 的代码

现在我们将把注意力转回到我们的树莓派机器人车上，并编写 Python 代码来与我们的 MQTT 代理通信，最终与我们的 HTML JavaScript 客户端通信。以下代码应直接从 T.A.R.A.S 运行。如果您想要无线运行 T.A.R.A.S，请使用 USB 电源适配器为树莓派供电，并在运行以下程序后断开 HDMI 电缆：

1.  从应用程序菜单 | 编程 | Thonny Python IDE 中打开 Thonny。

1.  单击新图标创建一个新文件。

1.  将以下代码输入文件中：

```py
import paho.mqtt.client as mqtt
from time import sleep
from RobotDance import RobotDance
from RobotWheels import RobotWheels
from RobotBeep import RobotBeep
from RobotCamera import RobotCamera
from gpiozero import DistanceSensor

distance_sensor = DistanceSensor(echo=18, trigger=17)

def on_message(client, userdata, message):
    command = message.payload.decode("utf-8")

    if command == "Forward":
        move_forward()
    elif command == "Backward":
        move_backward()
    elif command == "Left":
        turn_left()
    elif command == "Right":
        turn_right()
    elif command == "Picture":
        take_picture()
    elif command == "Alarm":
        sound_alarm()
    elif command == "Dance":
        robot_dance()

def move_forward():
    robotWheels = RobotWheels()
    robotWheels.move_forward()
    sleep(1)
    print("Moved forward")
    robotWheels.stop()
    watchMode()

def move_backward():
    robotWheels = RobotWheels()
    robotWheels.move_backwards()
    sleep(1)
    print("Moved backwards")
    robotWheels.stop()
    watchMode()

def turn_left():
    robotWheels = RobotWheels()
    robotWheels.turn_left()
    sleep(1)
    print("Turned left")
    robotWheels.stop()
    watchMode()

def turn_right():
    robotWheels = RobotWheels()
    robotWheels.turn_right()
    print("Turned right")
    robotWheels.stop()
    watchMode()

def take_picture():
    robotCamera = RobotCamera()
    robotCamera.take_picture()
    watchMode()

def sound_alarm():
    robotBeep = RobotBeep()
    robotBeep.play_song()

def robot_dance():
    robotDance = RobotDance()
    robotDance.lets_dance_incognito()
    print("Finished dancing now back to work")
    watchMode()

def watchMode():
    print("Watching.....")
    mqttc = mqtt.Client()
    mqttc.username_pw_set("vectydkb", "ZpiPufitxnnT")
    mqttc.connect('m10.cloudmqtt.com', 18215)
    mqttc.on_message = on_message
    mqttc.subscribe("RobotControl")

    while True:
        distance = distance_sensor.distance*100
        mqttc.loop()
        mqttc.publish("distance", distance)
        sleep(2)

watchMode()
```

1.  将文件保存为`MQTT-RobotControl.py`。

1.  从 Thonny 运行代码。

1.  转到 HTML JavaScript 客户端，然后单击前进按钮：

![](img/a3a9aa2f-9422-4ebf-9872-52f768cdc3d3.png)

1.  T.A.R.A.S 应该向前移动一秒，然后停止。

1.  底部的小灰色按钮允许您执行与 T.A.R.A.S 的各种任务：

![](img/4c7e63b9-3ddf-430a-80ea-19c0822004f3.png)

1.  通过单击这些按钮来探索每个按钮的功能。`Take Picture`按钮将拍照并将其存储在文件系统中，`T.A.R.A.S Alarm`将在 T.A.R.A.S 上触发警报，`T.A.R.A.S Dance`将使 T.A.R.A.S 跳舞。

1.  要订阅来自 T.A.R.A.S 距离传感器的`distance`数据，请单击 Track Distance 按钮：

![](img/2cc958d7-a0e0-4e21-bd37-36ea31d9c59e.png)

1.  单击 Track Distance 按钮后，您应该会看到一个弹出窗口，告诉您 HTML JavaScript 客户端现在已订阅了`distance`数据：

![](img/415dfa7a-89b4-48ad-8513-e091c2aa3d0e.png)

1.  单击关闭以关闭弹出窗口。现在您应该看到 T.A.R.A.S 的距离数据信息显示在 Track Distance 按钮旁边。

1.  与迄今为止我们编写的所有代码一样，我们的目标是使其尽可能简单和易于理解。我们代码的核心是`watch_mode`方法：

```py
def watchMode():
    print("Watching.....")
    mqttc = mqtt.Client()
    mqttc.username_pw_set("vectydkb", "ZpiPufitxnnT")
    mqttc.connect('m10.cloudmqtt.com', 18215)
    mqttc.on_message = on_message
    mqttc.subscribe("RobotControl")

    while True:
        distance = distance_sensor.distance*100
        mqttc.loop()
        mqttc.publish("distance", distance)
        sleep(2)
```

`watch_mode`方法是我们代码的默认方法。它在代码运行后立即调用，并在另一个方法完成时调用。在`watch_mode`中，我们需要创建一个名为`mqttc`的 MQTT 客户端对象，然后使用它连接到我们的 CloudMQTT 实例。从那里，我们将`on_message`回调设置为`on_message`方法。然后我们订阅`RobotControl`主题。随后的 while 循环调用我们的 MQTT 客户端`mqttc`的`loop`方法。由于我们已经设置了`on_message`回调，因此每当从`RobotControl`主题接收到消息时，程序都会退出 while 循环，并执行我们代码的`on_message`方法。

在`watch_mode`中，每 2 秒将距离传感器信息发布到`distance`主题。由于我们的 HTML JavaScript 客户端已设置为订阅`distance`主题上的消息，因此我们的 HTML JavaScript 客户端将每两秒在页面上更新`distance`信息。

# 从 T.A.R.A.S 直播视频。

从网络上控制 T.A.R.A.S 是一件了不起的事情，但如果我们看不到我们在做什么，那就没什么用了。如果你在树莓派上安装 RPi-Cam-Web-Interface，就可以很简单地从树莓派上直播视频。现在让我们来做这个：

1.  如果您的树莓派上没有安装`git`，请在终端中使用`sudo apt-get install git`进行安装。

1.  使用终端，通过运行`git clone https://github.com/silvanmelchior/RPi_Cam_Web_Interface.git`命令获取安装文件。

1.  使用`cd RPi_Cam_Web_Interface`命令更改目录。

1.  使用`./install.sh`命令运行安装程序。

1.  您应该看到配置选项屏幕：

![](img/721fe07d-1a6c-410b-9453-b884580c6170.png)

1.  通过在键盘上按*Tab*，接受所有默认设置，直到 OK 选项被突出显示。然后按*Enter*。

1.  在看到“现在启动摄像头系统”对话框时选择“是”：

![](img/61bcc260-0ba5-4de2-90f1-5a8dd2685c95.png)

1.  现在，我们已经准备好从我们的树莓派（T.A.R.A.S）实时传输视频。在另一台计算机上，打开浏览器，输入地址`http://<<您的树莓派 IP 地址>>/html`（在您的树莓派上使用`ifconfig`来查找您的 IP 地址；在我的情况下，视频流的 URL 是`http://192.168.0.31/html`）。

1.  现在，您应该看到视频流播放器加载到您的浏览器中，并从您的树莓派实时播放视频。以下是我办公室 T.A.R.A.S 的直播截图，显示我的无人机：

![](img/2c0a7d4a-e9e2-493f-b74b-0585dc0461dd.png)

RPi-Cam-Web-Interface 实用程序是一个令人惊叹的工具。花些时间尝试一下可用的各种选项和功能。

# 增强我们的 JavaScript 客户端以控制我们的机器人小车

正如我们已经提到的，我们的 HTML JavaScript 客户端是最具吸引力的界面。我设计它尽可能简单直接，以便解释各种概念。但是，如果我们想把它提升到另一个水平呢？以下是一些可能用于增强我们的 HTML JavaScript 客户端的 JavaScript 库的列表。

# Nipple.js

Nipple.js ([`www.bypeople.com/touch-screen-joystick/`](https://www.bypeople.com/touch-screen-joystick/))是一个 JavaScript 触摸屏操纵杆库，可用于控制机器人。Nipple.js 基本上是一种屏幕上的指向杆控制，类似于一些笔记本电脑上的控制。

![](img/279a34fc-8690-419e-a37f-3216132c51a3.png)

如果您要为触摸屏平板电脑或笔记本电脑创建 JavaScript 客户端，Nipple.js 可能是一个很好的构建技术。将 Nipple.js 等技术纳入我们的设计中，需要相当多的编码工作，以便将移动转换为 T.A.R.A.S 能理解的消息。简单的前进消息可能不够。消息可能是`Forward-1-Left-2.3`之类的，必须对其进行解析并提取信息，以确定转动电机的时间和移动哪些电机。

# HTML5 Gamepad API

您想连接物理操纵杆来控制我们的机器人小车吗？您可以使用 HTML5 Gamepad API ([`www.w3.org/TR/gamepad/`](https://www.w3.org/TR/gamepad/))。使用 HTML5 Gamepad API，您可以在构建的 Web 应用程序中使用标准游戏操纵杆。通过 HTML5 Gamepad API 控制您的机器人小车可能就像玩您最喜欢的视频游戏一样简单。

# Johnny-Five

Johnny-Five ([`johnny-five.io`](http://johnny-five.io))是一个 JavaScript 机器人和物联网平台。这是一个完全不同于我们为机器人小车开发的平台。现在我们已经从头开始构建了我们的机器人小车，并且已经手工编写了控制代码，我们可能有兴趣尝试一些新东西。Johnny-Five 可能是您决定成为专家的下一个技术。

# 摘要

我们做到了！我们已经完成了树莓派物联网之旅。在本章中，我们将所学知识整合在一起，并创建了自己的 HTML JavaScript 客户端，用于通过网页控制 T.A.R.A.S。我们使用类来控制 T.A.R.A.S，使得创建控制代码相对容易，因为我们只需要在类上调用方法，而不是从头开始创建控制代码。

我们简要介绍了如何轻松地从树莓派实时传输视频。尽管我们做所有这些是为了通过网络控制机器人小车，但不难想象我们可以利用所学知识来构建任意数量的不同物联网项目。

我们生活在一个非常激动人心的时代。我们中的任何一个人都可以仅凭我们的智慧和一些相对便宜的电子元件来构建下一个杀手级应用程序。如果可能的话，我希望我能激励您使用令人惊叹的树莓派计算机来构建您的下一个伟大项目。

对于那些质疑我们如何将这视为物联网项目的人，当我们只使用我们的本地网络时，请研究一下如何在路由器上打开端口以连接外部世界。然而，这不是一项应该轻率对待的任务，因为在这样做时必须解决安全问题。请注意，您的互联网服务提供商可能没有为您提供静态 IP 地址，因此您构建的任何用于从外部访问您的网络的东西都会在 IP 地址更改时中断（我曾经构建过一个定期检查我的 IP 地址的 PHP 页面，存储最新地址，并有外部客户端会访问该 PHP 获取地址，而不是将其硬编码）。

# 问题

1.  在我们的项目中，我们向哪个主题发布控制类型的消息？

1.  真或假？MQTT Broker 和 MQTT Server 是用来描述同一件事情的词语。

1.  真或假？T.A.R.A.S 在相同的 MQTT 主题上发布和订阅。

1.  我们的 HTML JavaScript 客户端中的大前进和后退按钮是什么颜色？

1.  真或假？使用 HTML JavaScript 客户端，我们能够远程使用 T.A.R.A.S 上的摄像头拍照。

1.  我们使用什么 MQTT 主题名称来订阅来自 T.A.R.A.S 的距离数据？

1.  真或假？我们的 HTML JavaScript 客户端采用了屡获殊荣的 UI 设计。

1.  真或假？使用我们的 CloudMQTT 账户，我们能够查看我们实例中发布的消息。

1.  我们使用什么技术来从 T.A.R.A.S 进行视频直播？

1.  真或假？Johnny-Five 是可口可乐公司推出的一种新果汁饮料。

# 进一步阅读

当我们在 T.A.R.A.S 上设置实时流时，我们简要地介绍了 RPi-Cam-Web-Interface 网页界面。这个网页界面非常惊人，对它的更深入了解只会增强我们对树莓派的所有可能性的理解。请访问[`elinux.org/RPi-Cam-Web-Interface`](https://elinux.org/RPi-Cam-Web-Interface)获取更多信息。
