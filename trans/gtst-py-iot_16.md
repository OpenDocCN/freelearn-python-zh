# 第十六章：使用 Python 可以开发的一些很棒的东西

在本章中，我们将讨论 Python 中的一些高级主题。我们还将讨论一些独特的主题（如图像处理），让您开始使用 Python 进行应用程序开发。

# 使用 Raspberry Pi Zero 进行图像处理

Raspberry Pi Zero 是一款价格便宜的硬件，配备了 1 GHz 处理器。虽然它不足以运行某些高级图像处理操作，但可以帮助您在 25 美元的预算内学习基础知识（Raspberry Pi Zero 和摄像头的成本）。

我们建议您在 Raspberry Pi Zero 上使用 16 GB（或更高）的卡来安装本节讨论的图像处理工具集。

例如，您可以使用 Raspberry Pi Zero 来跟踪后院的鸟。在本章中，我们将讨论在 Raspberry Pi Zero 上开始图像处理的不同方法。

为了在本节中使用摄像头测试一些示例，需要 Raspberry Pi Zero v1.3 或更高版本。检查您的 Raspberry Pi Zero 的背面以验证板的版本：

识别您的 Raspberry Pi Zero 的版本

# OpenCV

**OpenCV**是一个开源工具箱，包括为图像处理开发的不同软件工具。OpenCV 是一个跨平台的工具箱，支持不同的操作系统。由于 OpenCV 在开源许可下可用，全世界的研究人员通过开发工具和技术为其增长做出了贡献。这使得开发应用程序相对容易。OpenCV 的一些应用包括人脸识别和车牌识别。

由于其有限的处理能力，安装框架可能需要几个小时。在我们这里大约花了 10 个小时。

我们按照[`www.pyimagesearch.com/2015/10/26/how-to-install-opencv-3-on-raspbian-jessie/`](http://www.pyimagesearch.com/2015/10/26/how-to-install-opencv-3-on-raspbian-jessie/)上的指示在 Raspberry Pi Zero 上安装 OpenCV。我们特别按照了使用 Python 3.x 绑定安装 OpenCV 的指示，并验证了安装过程。我们大约花了 10 个小时来完成在 Raspberry Pi Zero 上安装 OpenCV。出于不重复造轮子的考虑，我们不会重复这些指示。

# 安装的验证

让我们确保 OpenCV 安装及其 Python 绑定工作正常。启动命令行终端，并确保您已经通过执行`workon cv`命令启动了`cv`虚拟环境（您可以验证您是否在`cv`虚拟环境中）：

验证您是否在 cv 虚拟环境中

现在，让我们确保我们的安装工作正常。从命令行启动 Python 解释器，并尝试导入`cv2`模块：

```py
    >>> import cv2
 >>> cv2.__version__
 '3.0.0'
```

这证明了 OpenCV 已经安装在 Raspberry Pi Zero 上。让我们编写一个涉及 OpenCV 的*hello world*示例。在这个示例中，我们将打开一张图像（这可以是您的 Raspberry Pi Zero 桌面上的任何彩色图像），并在将其转换为灰度后显示它。我们将使用以下文档来编写我们的第一个示例：[`docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html`](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html)。

根据文档，我们需要使用`imread()`函数来读取图像文件的内容。我们还需要指定要读取图像的格式。在这种情况下，我们将以灰度格式读取图像。这由作为函数的第二个参数传递的`cv2.IMREAD_GRAYSCALE`来指定：

```py
import cv2 

img = cv2.imread('/home/pi/screenshot.jpg',cv2.IMREAD_GRAYSCALE)
```

现在图像以灰度格式加载并保存到`img`变量中，我们需要在新窗口中显示它。这是通过`imshow()`函数实现的。根据文档，我们可以通过将窗口名称指定为第一个参数，将图像指定为第二个参数来显示图像：

```py
cv2.imshow('image',img)
```

在这种情况下，我们将打开一个名为`image`的窗口，并显示我们在上一步加载的`img`的内容。我们将显示图像，直到收到按键。这是通过使用`cv2.waitKey()`函数实现的。根据文档，`waitkey()`函数监听键盘事件：

```py
cv2.waitKey(0)
```

`0`参数表示我们将无限期等待按键。根据文档，当以毫秒为单位的持续时间作为参数传递时，`waitkey()`函数会监听指定持续时间的按键。当按下任何键时，窗口会被`destroyAllWindows()`函数关闭：

```py
cv2.destroyAllWindows()
```

将所有部件组装在一起，我们有：

```py
import cv2

img = cv2.imread('/home/pi/screenshot.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

上述代码示例可在本章的`opencv_test.py`中下载。安装 OpenCV 库后，尝试加载图像，如本示例所示。它应该以灰度加载图像，如下图所示：

树莓派桌面以灰度加载

这个窗口会在按下任意键时关闭。

# 向读者提出挑战

在上面的示例中，窗口在按下任意键时关闭。查看文档，确定是否可能在按下鼠标按钮时关闭所有窗口。

# 将相机安装到树莓派 Zero

测试我们下一个示例需要相机连接器和相机。购买相机和适配器的一个来源如下：

| **名称** | **来源** |
| --- | --- |
| 树莓派 Zero 相机适配器 | [`thepihut.com/products/raspberry-pi-zero-camera-adapter`](https://thepihut.com/products/raspberry-pi-zero-camera-adapter) |
| 树莓派相机 | [`thepihut.com/products/raspberry-pi-camera-module`](https://thepihut.com/products/raspberry-pi-camera-module) |

执行以下步骤将相机安装到树莓派 Zero 上：

1.  第一步是将相机连接到树莓派 Zero。相机适配器可以安装如下图所示。抬起连接器标签，滑动相机适配器并轻轻按下连接器：

![](img/ce975b8d-7043-48b9-888c-5cc2f83c2bbc.jpg)

1.  我们需要在树莓派 Zero 上启用相机接口。在桌面上，转到首选项并启动树莓派配置。在树莓派配置的接口选项卡下，启用相机，并保存配置：

![](img/f4cb1bc3-eb01-4a84-93d0-ab51a26c6525.png)启用相机接口

1.  通过从命令行终端运行以下命令来拍照测试相机：

```py
       raspistill -o /home/pi/Desktop/test.jpg
```

1.  它应该拍照并保存到树莓派桌面上。验证相机是否正常工作。如果无法使相机工作，我们建议查看树莓派基金会发布的故障排除指南：[`www.raspberrypi.org/documentation/raspbian/applications/camera.md`](https://www.raspberrypi.org/documentation/raspbian/applications/camera.md)。

相机电缆有点笨重，拍照时可能会有些困难。我们建议使用相机支架。我们发现这个很有用（如下图所示）[`a.co/hQolR7O`](http://a.co/hQolR7O)：

使用树莓派相机的支架

让我们试试相机，并与 OpenCV 库一起使用：

1.  我们将使用相机拍照，并使用 OpenCV 框架显示它。为了在 Python 中访问相机，我们需要`picamera`包。可以按照以下方式安装：

```py
       pip3 install picamera
```

1.  让我们确保包能够按预期使用一个简单的程序。`picamera`包的文档可在[`picamera.readthedocs.io/en/release-1.12/api_camera.html`](https://picamera.readthedocs.io/en/release-1.12/api_camera.html)找到。

1.  第一步是初始化`PiCamera`类。接下来是翻转图像，使其在垂直轴上翻转。这仅在相机倒置安装时才需要。在其他安装中可能不需要：

```py
       with PiCamera() as camera: 
       camera.vflip = True
```

1.  在拍照之前，我们可以使用`start_preview()`方法预览即将捕获的图片：

```py
       camera.start_preview()
```

1.  在我们拍照之前，让我们预览`10`秒钟。我们可以使用`capture()`方法拍照：

```py
       sleep(10) 
       camera.capture("/home/pi/Desktop/desktop_shot.jpg") 
       camera.stop_preview()
```

1.  `capture()`方法需要文件位置作为参数（如前面的代码片段所示）。完成后，我们可以使用`stop_preview()`关闭相机预览。

1.  总结一下，我们有：

```py
       from picamera import PiCamera 
       from time import sleep

       if __name__ == "__main__": 
         with PiCamera() as camera: 
           camera.vflip = True 
           camera.start_preview() 
           sleep(10) 
           camera.capture("/home/pi/Desktop/desktop_shot.jpg") 
           camera.stop_preview()
```

上述代码示例可与本章一起下载，文件名为`picamera_test.py`。使用相机拍摄的快照如下图所示：

![](img/355f9dc6-2b49-4a4a-82df-4dc5cb381f71.png)使用树莓派摄像头模块捕获的图像

1.  让我们将此示例与上一个示例结合起来——将此图像转换为灰度并显示，直到按下键。确保您仍然在`cv`虚拟环境工作空间中。

1.  让我们将捕获的图像转换为灰度，如下所示：

```py
       img = cv2.imread("/home/pi/Desktop/desktop_shot.jpg",
       cv2.IMREAD_GRAYSCALE)
```

以下是捕获后转换的图像：

![](img/48620f3f-ce1a-4271-8e97-d07f43502fbf.png)图像在捕获时转换为灰度

1.  现在我们可以按如下方式显示灰度图像：

```py
       cv2.imshow("image", img) 
       cv2.waitKey(0) 
       cv2.destroyAllWindows()
```

修改后的示例可作为`picamera_opencvtest.py`进行下载。

到目前为止，我们已经展示了在 Python 中开发图像处理应用程序。我们还建议查看 OpenCV Python 绑定文档中提供的示例（在本节介绍部分提供了链接）。

# 语音识别

在本节中，我们将讨论在 Python 中开发语音识别示例涉及语音识别。我们将利用`requests`模块（在上一章中讨论）来使用`wit.ai`（[`wit.ai/`](https://wit.ai/)）转录音频。

有几种语音识别工具，包括 Google 的语音 API、IBM Watson、Microsoft Bing 的语音识别 API。我们以`wit.ai`为例进行演示。

语音识别在我们希望使树莓派零对语音命令做出响应的应用中非常有用。

让我们回顾使用`wit.ai`在 Python 中构建语音识别应用程序（其文档可在[`github.com/wit-ai/pywit`](https://github.com/wit-ai/pywit)找到）。为了进行语音识别和识别语音命令，我们需要一个麦克风。但是，我们将演示使用一个现成的音频样本。我们将使用一篇研究出版物提供的音频样本（可在[`ecs.utdallas.edu/loizou/speech/noizeus/clean.zip`](http://ecs.utdallas.edu/loizou/speech/noizeus/clean.zip)找到）。

`wit.ai` API 许可证规定，该工具可免费使用，但上传到其服务器的音频用于调整其语音转录工具。

我们现在将尝试转录`sp02.wav`音频样本，执行以下步骤：

1.  第一步是注册`wit.ai`帐户。请注意以下截图中显示的 API：

![](img/f4e59c40-82a5-4fcf-8ccc-d7512b47c0fe.png)

1.  第一步是安装 requests 库。可以按以下方式安装：

```py
       pip3 install requests 
```

1.  根据`wit.ai`的文档，我们需要向我们的请求添加自定义标头，其中包括 API 密钥（用您的帐户中的令牌替换`$TOKEN`）。我们还需要在标头中指定文件格式。在这种情况下，它是一个`.wav`文件，采样频率为 8000 Hz：

```py
       import requests 

       if __name__ == "__main__": 
         url = 'https://api.wit.ai/speech?v=20161002' 
         headers = {"Authorization": "Bearer $TOKEN", 
                    "Content-Type": "audio/wav"}
```

1.  为了转录音频样本，我们需要将音频样本附加到请求体中：

```py
       files = open('sp02.wav', 'rb') 
       response = requests.post(url, headers=headers, data=files) 
       print(response.status_code) 
       print(response.text)
```

1.  将所有这些放在一起，我们得到了这个：

```py
       #!/usr/bin/python3 

       import requests 

       if __name__ == "__main__": 
         url = 'https://api.wit.ai/speech?v=20161002' 
         headers = {"Authorization": "Bearer $TOKEN", 
                    "Content-Type": "audio/wav"} 
         files = open('sp02.wav', 'rb') 
         response = requests.post(url, headers=headers, data=files) 
         print(response.status_code) 
         print(response.text)
```

前面的代码示例可与本章一起下载，文件名为`wit_ai.py`。尝试执行前面的代码示例，它应该会转录音频样本：`sp02.wav`。我们有以下代码：

```py
200
{
  "msg_id" : "fae9cc3a-f7ed-4831-87ba-6a08e95f515b",
  "_text" : "he knew the the great young actress",
  "outcomes" : [ {
    "_text" : "he knew the the great young actress",
    "confidence" : 0.678,
    "intent" : "DataQuery",
    "entities" : {
      "value" : [ {
        "confidence" : 0.7145905790744499,
        "type" : "value",
        "value" : "he",
        "suggested" : true
      }, {
        "confidence" : 0.5699616515542044,
        "type" : "value",
        "value" : "the",
        "suggested" : true
      }, {
        "confidence" : 0.5981701138805214,
        "type" : "value",
        "value" : "great",
        "suggested" : true
      }, {
        "confidence" : 0.8999612482250062,
        "type" : "value",
        "value" : "actress",
        "suggested" : true
      } ]
    }
  } ],
  "WARNING" : "DEPRECATED"
}
```

音频样本包含以下录音：*他知道那位年轻女演员的技巧*。根据`wit.ai` API，转录为*他知道了那位年轻女演员*。词错误率为 22%（[`en.wikipedia.org/wiki/Word_error_rate`](https://en.wikipedia.org/wiki/Word_error_rate)）。

# 自动化路由任务

在这一部分，我们将讨论如何在 Python 中自动化路由任务。我们举了两个例子，它们展示了树莓派 Zero 作为个人助手的能力。第一个例子涉及改善通勤，而第二个例子则是帮助提高词汇量。让我们开始吧。

# 改善日常通勤

许多城市和公共交通系统已经开始向公众分享数据，以增加透明度并提高运营效率。交通系统已经开始通过 API 向公众分享公告和交通信息。这使任何人都能开发提供给通勤者信息的移动应用。有时，这有助于缓解公共交通系统内的拥堵。

这个例子是受到一位朋友的启发，他追踪旧金山共享单车站点的自行车可用性。在旧金山湾区，有一个自行车共享计划，让通勤者可以从交通中心租一辆自行车到他们的工作地点。在像旧金山这样拥挤的城市，特定站点的自行车可用性会根据一天的时间而波动。

这位朋友想要根据最近的共享单车站点的自行车可用性来安排他的一天。如果站点上的自行车非常少，这位朋友更喜欢早点出发租一辆自行车。他正在寻找一个简单的技巧，可以在自行车数量低于某个阈值时向他的手机推送通知。旧金山的共享单车计划在[`feeds.bayareabikeshare.com/stations/stations.json`](http://feeds.bayareabikeshare.com/stations/stations.json)上提供了这些数据。

让我们回顾一下构建一个简单的例子，可以使其向移动设备发送推送通知。为了发送移动推送通知，我们将使用**If This Then That**（**IFTTT**）——这是一个使您的项目连接到第三方服务的服务。

在这个例子中，我们将解析以 JSON 格式可用的数据，检查特定站点的可用自行车数量，如果低于指定的阈值，就会触发手机设备上的通知。

让我们开始吧：

1.  第一步是从共享单车服务中检索自行车的可用性。这些数据以 JSON 格式在[`feeds.bayareabikeshare.com/stations/stations.json`](http://feeds.bayareabikeshare.com/stations/stations.json)上提供。数据包括整个网络的自行车可用性。

1.  每个站点的自行车可用性都有一些参数，比如站点 ID、站点名称、地址、可用自行车数量等。

1.  在这个例子中，我们将检索旧金山`Townsend at 7th`站点的自行车可用性。站点 ID 是`65`（在浏览器中打开前面提到的链接以找到`id`）。让我们编写一些 Python 代码来检索自行车可用性数据并解析这些信息：

```py
       import requests 

       BIKE_URL = http://feeds.bayareabikeshare.com/stations 
       /stations.json 

       # fetch the bike share information 
       response = requests.get(BIKE_URL) 
       parsed_data = response.json()
```

第一步是使用`GET`请求（通过`requests`模块）获取数据。`requests`模块提供了内置的 JSON 解码器。可以通过调用`json()`函数来解析 JSON 数据。

1.  现在，我们可以遍历站点的字典，并通过以下步骤找到`Townsend at 7th`站点的自行车可用性：

1.  在检索到的数据中，每个站点的数据都附带一个 ID。问题站点的 ID 是`65`（在浏览器中打开之前提供的数据源 URL 以了解数据格式；数据的片段如下截图所示）：

![](img/5dc6a42b-f6ff-49b5-b0e5-01f33cebc4ce.png)使用浏览器获取的自行车共享数据源的片段

1.  我们需要遍历数值并确定站点`id`是否与`Townsend at 7th`的匹配：

```py
              station_list = parsed_data['stationBeanList'] 
              for station in station_list: 
                if station['id'] == 65 and 
                   station['availableBikes'] < 2: 
                  print("The available bikes is %d" % station
                  ['availableBikes'])
```

如果站点上的自行车少于`2`辆，我们会向我们的移动设备推送移动通知。

1.  为了接收移动通知，您需要安装*IF by IFTTT*应用程序（适用于苹果和安卓设备）。

1.  我们还需要在 IFTTT 上设置一个配方来触发移动通知。在[`ifttt.com/`](https://ifttt.com/)注册一个账户。

IFTTT 是一个服务，可以创建连接设备到不同应用程序并自动化任务的配方。例如，可以将树莓派 Zero 跟踪的事件记录到您的 Google Drive 上的电子表格中。

IFTTT 上的所有配方都遵循一个通用模板——*如果这样，那么那样*，也就是说，如果发生了特定事件，那么就会触发特定的动作。例如，我们需要创建一个 applet，以便在收到 web 请求时触发移动通知。

1.  您可以使用您的帐户下拉菜单开始创建一个 applet，如下截图所示：

![](img/b3366418-76f2-48ad-809c-1b1f1a7430b3.png)开始在 IFTTT 上创建一个配方

1.  它应该带您到一个配方设置页面（如下所示）。点击这个并设置一个传入的 web 请求：

![](img/23037a0e-b3a4-47e9-8049-ddc92161b253.png)点击这个

1.  选择 Maker Webhooks 频道作为传入触发器：

![](img/67372bef-0815-42f2-8df2-81aa088e6aab.png)选择 Maker Webhooks 频道

1.  选择接收 web 请求。来自树莓派的 web 请求将作为触发器发送移动通知：

![](img/73429b8a-fcb0-4be5-9e76-bd54cde37d86.png)选择接收 web 请求

1.  创建一个名为`mobile_notify`的触发器：

![](img/adc6d023-51ad-4120-8e02-3fcaa1fc4645.png)创建一个名为 mobile_notify 的新触发器

1.  现在是时候为传入触发器创建一个动作了。点击那个。

![](img/07d3564b-95ee-496e-927f-3c66400bc4e5.png)点击这个

1.  选择通知：

![](img/31cf0258-7cb2-4ab3-ab1d-6207f545b4dd.png)选择通知

1.  现在，让我们格式化我们想要在设备上收到的通知：

![](img/624dbdf8-0882-422f-9795-2207e23aa71f.png)为您的设备设置通知

1.  在移动通知中，我们需要接收自行车共享站点上可用自行车的数量。点击+ Ingredient 按钮，选择`Value1`。

![](img/67d4c4f5-3a6b-4c6b-95ea-c1671f8bb5a0.png)

格式化消息以满足您的需求。例如，当树莓派触发通知时，希望以以下格式收到消息：`该回家了！Townsend & 7th 只有 2 辆自行车可用！`

![](img/21ebe4f5-60ce-42ca-b57a-a7ac6ce9b362.png)

1.  一旦您对消息格式满意，选择创建动作，您的配方就应该准备好了！

![](img/77fb0b9e-af9f-4ed6-ac95-fd5881e6545e.png)创建一个配方

1.  为了在我们的移动设备上触发通知，我们需要一个 URL 来进行`POST`请求和一个触发键。这在您的 IFTTT 帐户的 Services | Maker Webhooks | Settings 下可用。

触发器可以在这里找到：

![](img/b4bd2cc0-41f1-45c2-a1b5-2bccbd8fb0d1.png)

在新的浏览器窗口中打开前面截图中列出的 URL。它提供了`POST`请求的 URL 以及如何进行 web 请求的解释（如下截图所示）：

![](img/eb376740-2b7f-4b22-8f8e-76a6d19b7657.png)使用之前提到的 URL 进行 POST 请求（为了隐私而隐藏密钥）

1.  在发出请求时（如 IFTTT 文档中所述），如果我们在请求的 JSON 主体中包括自行车的数量（使用`Value1`），它可以显示在移动通知上。

1.  让我们重新查看 Python 示例，当自行车数量低于一定阈值时进行网络请求。将`IFTTT` URL 和您的 IFTTT 访问密钥（从您的 IFTTT 帐户中检索）保存到您的代码中，如下所示：

```py
       IFTTT_URL = "https://maker.ifttt.com/trigger/mobile_notify/ 
       with/key/$KEY"
```

1.  当自行车数量低于一定阈值时，我们需要使用 JSON 主体中编码的自行车信息进行`POST`请求：

```py
       for station in station_list: 
         if station['id'] == 65 and 
            station['availableBikes'] < 3: 
           print("The available bikes is %d" % 
           station['availableBikes']) 
           payload = {"value1": station['availableBikes']} 
           response = requests.post(IFTTT_URL, json=payload) 
           if response.status_code == 200: 
             print("Notification successfully triggered")
```

1.  在上述代码片段中，如果自行车少于三辆，将使用`requests`模块进行`POST`请求。可用自行车的数量使用键`value1`进行编码：

```py
       payload = {"value1": station['availableBikes']}
```

1.  将所有这些放在一起，我们有这个：

```py
       #!/usr/bin/python3 

       import requests 
       import datetime 

       BIKE_URL = "http://feeds.bayareabikeshare.com/stations/
       stations.json" 
       # find your key from ifttt 
       IFTTT_URL = "https://maker.ifttt.com/trigger/mobile_notify/
       with/key/$KEY" 

       if __name__ == "__main__": 
         # fetch the bike share information 
         response = requests.get(BIKE_URL) 
         parsed_data = response.json() 
         station_list = parsed_data['stationBeanList'] 
         for station in station_list: 
           if station['id'] == 65 and 
              station['availableBikes'] < 10: 
             print("The available bikes is %d" % station
             ['availableBikes']) 
  payload = {"value1": station['availableBikes']} 
             response = requests.post(IFTTT_URL, json=payload) 
             if response.status_code == 200: 
               print("Notification successfully triggered")
```

上述代码示例可与本章一起下载，名称为`bike_share.py`。在设置 IFTTT 上的配方后尝试执行它。如果需要，调整可用自行车数量的阈值。您应该会收到移动设备上的通知：

![](img/0e3e05fa-062c-4929-887e-afabf3fd16d8.png)在您的移动设备上通知

# 读者的挑战

在此示例中，自行车信息被获取和解析，如果必要，将触发通知。您将如何修改此代码示例以确保它在一天中的特定时间执行？（提示：使用`datetime`模块）。

您将如何构建一个作为视觉辅助的桌面显示？

# 项目挑战

尝试找出您所在地区的交通系统是否向其用户提供此类数据。您将如何利用数据帮助通勤者节省时间？例如，您将如何使用此类数据向您的朋友/同事提供交通系统建议？

完成书后，我们将发布一个类似的示例，使用旧金山湾区快速交通（BART）的数据。

# 提高你的词汇量

使用 Python 可以提高您的词汇量！想象一下设置一个大型显示屏，它显眼地安装在某个地方，并且每天更新。我们将使用`wordnik` API（在[`www.wordnik.com/signup`](https://www.wordnik.com/signup)注册 API 密钥）。

1.  第一步是为 python3 安装`wordnik` API 客户端：

```py
       git clone https://github.com/wordnik/wordnik-python3.git
 cd wordnik-python3/
 sudo python3 setup.py install
```

wordnik API 有使用限制。有关更多详细信息，请参阅 API 文档。

1.  让我们回顾一下使用`wordnik` Python 客户端编写我们的第一个示例。为了获取当天的单词，我们需要初始化`WordsApi`类。根据 API 文档，可以这样做：

```py
       # sign up for an API key 
       API_KEY = 'API_KEY' 
       apiUrl = 'http://api.wordnik.com/v4' 
       client = swagger.ApiClient(API_KEY, apiUrl) 
       wordsApi = WordsApi.WordsApi(client)
```

1.  现在`WordsApi`类已初始化，让我们继续获取当天的单词：

```py
       example = wordsApi.getWordOfTheDay()
```

1.  这将返回一个`WordOfTheDay`对象。根据`wordnik` Python 客户端文档，该对象包括不同的参数，包括单词、其同义词、来源、用法等。当天的单词及其同义词可以打印如下：

```py
       print("The word of the day is %s" % example.word) 
       print("The definition is %s" %example.definitions[0].text)
```

1.  将所有这些放在一起，我们有这个：

```py
       #!/usr/bin/python3 

       from wordnik import * 

       # sign up for an API key 
       API_KEY = 'API_KEY' 
       apiUrl = 'http://api.wordnik.com/v4' 

       if __name__ == "__main__": 
         client = swagger.ApiClient(API_KEY, apiUrl) 
         wordsApi = WordsApi.WordsApi(client) 
         example = wordsApi.getWordOfTheDay() 
         print("The word of the day is %s" % example.word) 
         print("The definition is %s" %example.definitions[0].text)
```

上述代码片段可与本章一起下载，名称为`wordOfTheDay.py`。注册 API 密钥，您应该能够检索当天的单词：

```py
       The word of the day is transpare
 The definition is To be, or cause to be, transparent; to appear,
       or cause to appear, or be seen, through something.
```

# 读者的挑战

您将如何将此应用程序守护程序化，以便每天更新当天的单词？（提示：cronjob 或`datetime`）。

# 项目挑战

可以使用`wordnik` API 构建一个单词游戏。想想一个既有趣又有助于提高词汇量的单词游戏。您将如何构建一个提示玩家并接受答案输入的东西？

尝试在显示器上显示当天的单词。您将如何实现这一点？

# 日志记录

日志（[`docs.python.org/3/library/logging.html`](https://docs.python.org/3/library/logging.html)）有助于解决问题。它通过跟踪应用程序记录的事件序列来确定问题的根本原因。让我们通过一个简单的应用程序来回顾日志。为了回顾日志，让我们通过发出一个`POST`请求来回顾它：

1.  日志的第一步是设置日志文件位置和日志级别：

```py
       logging.basicConfig(format='%(asctime)s : %(levelname)s :
       %(message)s', filename='log_file.log', level=logging.INFO)
```

在初始化`logging`类时，我们需要指定日志信息、错误等的格式到文件中。在这种情况下，格式如下：

```py
       format='%(asctime)s : %(levelname)s : %(message)s'
```

日志消息的格式如下：

```py
       2016-10-25 20:28:07,940 : INFO : Starting new HTTPS
       connection (1):
       maker.ifttt.com
```

日志消息保存在名为`log_file.log`的文件中。

日志级别确定我们应用程序所需的日志级别。不同的日志级别包括`DEBUG`、`INFO`、`WARN`和`ERROR`。

在这个例子中，我们将日志级别设置为`INFO`。因此，属于`INFO`、`WARNING`或`ERROR`级别的任何日志消息都将保存到文件中。

如果日志级别设置为`ERROR`，则只有这些日志消息会保存到文件中。

1.  让我们根据`POST`请求的结果记录一条消息：

```py
       response = requests.post(IFTTT_URL, json=payload) 
       if response.status_code == 200: 
         logging.info("Notification successfully triggered") 
       else: 
         logging.error("POST request failed")
```

1.  将所有这些放在一起，我们有：

```py
       #!/usr/bin/python3 

       import requests 
       import logging 

       # find your key from ifttt 
       IFTTT_URL = "https://maker.ifttt.com/trigger/rf_trigger/
       with/key/$key" 

       if __name__ == "__main__": 
         # fetch the bike share information 
         logging.basicConfig(format='%(asctime)s : %(levelname)s
         : %(message)s', filename='log_file.log', level=logging.INFO) 
         payload = {"value1": "Sample_1", "value2": "Sample_2"} 
         response = requests.post(IFTTT_URL, json=payload) 
         if response.status_code == 200: 
           logging.info("Notification successfully triggered") 
         else: 
           logging.error("POST request failed")
```

前面的代码示例（`logging_example.py`）可与本章一起下载。这是 Python 中日志概念的一个非常简单的介绍。

# Python 中的线程

在本节中，我们将讨论 Python 中的线程概念。线程使得能够同时运行多个进程成为可能。例如，我们可以在监听传感器的同时运行电机。让我们通过一个例子来演示这一点。

我们将模拟一个情况，我们希望处理相同类型传感器的事件。在这个例子中，我们只是打印一些内容到屏幕上。我们需要定义一个函数来监听每个传感器的事件：

```py
def sensor_processing(string): 
  for num in range(5): 
    time.sleep(5) 
    print("%s: Iteration: %d" %(string, num))
```

我们可以利用前面的函数同时使用 Python 中的`threading`模块监听三个不同传感器的事件：

```py
thread_1 = threading.Thread(target=sensor_processing, args=("Sensor 1",)) 
thread_1.start() 

thread_2 = threading.Thread(target=sensor_processing, args=("Sensor 2",)) 
thread_2.start() 

thread_3 = threading.Thread(target=sensor_processing, args=("Sensor 3",)) 
thread_3.start()
```

将所有这些放在一起，我们有：

```py
import threading 
import time 

def sensor_processing(string): 
  for num in range(5): 
    time.sleep(5) 
    print("%s: Iteration: %d" %(string, num)) 

if __name__ == '__main__': 
  thread_1 = threading.Thread(target=sensor_processing, args=("Sensor 1",)) 
  thread_1.start() 

  thread_2 = threading.Thread(target=sensor_processing, args=("Sensor 2",)) 
  thread_2.start() 

  thread_3 = threading.Thread(target=sensor_processing, args=("Sensor 3",)) 
  thread_3.start()
```

前面的代码示例（可作为`threading_example.py`下载）启动三个线程，同时监听来自三个传感器的事件。输出看起来像这样：

```py
Thread 1: Iteration: 0 
Thread 2: Iteration: 0 
Thread 3: Iteration: 0 
Thread 2: Iteration: 1 
Thread 1: Iteration: 1 
Thread 3: Iteration: 1 
Thread 2: Iteration: 2 
Thread 1: Iteration: 2 
Thread 3: Iteration: 2 
Thread 1: Iteration: 3 
Thread 2: Iteration: 3 
Thread 3: Iteration: 3 
Thread 1: Iteration: 4 
Thread 2: Iteration: 4 
Thread 3: Iteration: 4
```

# Python 的 PEP8 样式指南

**PEP8**是 Python 的样式指南，它帮助程序员编写可读的代码。遵循某些约定以使我们的代码可读是很重要的。一些编码约定的例子包括以下内容：

+   内联注释应以`# `开头，后面跟着一个空格。

+   变量应该遵循以下约定：`first_var`。

+   避免每行末尾的空格。例如，`if name == "test":`后面不应该有空格。

你可以在[`www.python.org/dev/peps/pep-0008/#block-comments`](https://www.python.org/dev/peps/pep-0008/#block-comments)阅读完整的 PEP8 标准。

# 验证 PEP8 指南

有工具可以验证您的代码是否符合 PEP8 标准。编写代码示例后，请确保您的代码符合 PEP8 标准。可以使用`pep8`包来实现。

```py
    pip3 install pep8
```

让我们检查我们的代码示例是否符合 PEP8 规范。可以按照以下步骤进行：

```py
    pep8 opencv_test.py
```

检查指出了以下错误：

```py
    opencv_test.py:5:50: E231 missing whitespace after ','
 opencv_test.py:6:19: E231 missing whitespace after ','
```

根据输出结果，以下行缺少逗号后的空格，分别是第`5`行和第`6`行：

![](img/1b943d8f-20da-4864-b184-b775bba8fd7b.png)逗号后缺少尾随空格

让我们修复这个问题，并且我们的代码应该遵循 PEP8 规范。重新检查文件，错误将会消失。为了使你的代码可读，总是在将代码提交到公共存储库之前运行 PEP8 检查。

# 总结

在这一章中，我们讨论了 Python 中的高级主题。我们讨论了包括语音识别、构建通勤信息工具以及改善词汇量的 Python 客户端在内的主题。Python 中有许多在数据科学、人工智能等领域广泛使用的高级工具。我们希望本章讨论的主题是学习这些工具的第一步。
