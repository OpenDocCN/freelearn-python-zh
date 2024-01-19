# 第二十七章：给 Jarvis 发声

曾经想过是否可以使用机器人来完成我们的工作吗？是的！在一些高科技小说或漫威电影甚至漫画书中肯定是可能的。所以，系好安全带，准备好迎接这个令人惊叹的章节，在这里，您将实际实现我刚才提到的内容。

本章将涵盖以下主题：

+   基本安装

+   自动交付答录机

+   制作一个交互式门答录机器人

+   让 Jarvis 理解我们的声音

# 基本安装

有各种方法和方法可以控制我们的智能家居 Jarvis，其中一些我们之前已经探讨过，比如通过控制它。因此，首先，我们需要准备我们的系统以能够进行语音合成；为此，让我们执行以下过程。

首先，转到终端并输入以下命令：

```py
sudo apt-get install alsa-utils
```

这将安装依赖项`alsa-utils`。`alsa-utils`包包含各种实用程序，用于控制您的声卡驱动程序。

完成后，您需要编辑文件。为此，我们需要打开文件。使用以下命令：

```py
sudo nano /etc/modules
```

完成后，将打开一个文件；在该文件的底部，您需要添加以下行：

```py
snd_bcm2835
```

您不需要深究我们为什么这样做。它只是用来设置事情。我可以给你解释；但是，在这个激动人心的时刻，我不想让你感到无聊。

此外，如果你幸运的话，有时你可能会发现该行已经存在。如果是这种情况，就让它在那里，不要动它。

现在，要播放我们需要 Jarvis 说的声音，我们需要一个音频播放器。不，不是你家里的那种。我们说的是能够播放的软件。

要安装播放器，我们需要运行以下命令：

```py
sudo apt-get install mplayer
```

好了，我们已经完成了音频播放器；让我们看看接下来要做什么。现在，我们需要再次编辑媒体播放器的文件。我们将使用相同的步骤打开文件并编辑它：

```py
sudo nano /etc/mplayer/mplayer.conf
```

这将打开文件。与之前一样，只需添加以下行：

```py
nolirc=yes
```

最后，我们需要给它一些声音，所以运行以下命令：

```py
sudo apt-get install festvox-rablpc16k
```

这将为 Jarvis 安装一个 16 kHz 的英国男声。我们喜欢英国口音，不是吗？

完美。一旦我们完成了之前提到的所有步骤，我们就可以开始了。要测试声音，只需将 USB 扬声器连接到树莓派并运行以下代码：

```py
import os
from time import sleep
os.system('echo "hello! i am raspberry pi robot"|festival --tts ')
sleep(2)
os.system('echo "how are you?"| festival --tts ')
sleep(2)
os.system('echo "I am having fun."| festival --tts ')
sleep(2)
```

好了，让我们看看我们实际做了什么：

```py
import os
```

您可能已经发现，我们正在导入名为`os`的库。该库提供了一种使用操作系统相关功能的方法：

```py
os.system('echo "Hello from the other side"|festival --tts ')
```

在这里，我们使用了一个名为`system()`的方法；它的作用是执行一个 shell 命令。也许你会想知道这是什么。shell 命令是用户用来访问系统功能并与之交互的命令。所以现在我们想要将文本转换为语音，我们将向这个函数提供两个参数。首先，文本是什么？在我们的例子中，它是`Hello from the other side`；我们这里的第二个参数是`festival --tts`。现在`festival`是一个库，`tts`代表文本到语音转换。因此，当我们将其传递给参数时，系统将知道要将传递给参数的文本从文本转换为语音。

就是这样！是的，就是这样。这就是我们让您的树莓派说话所需做的一切。

# 自动交付答录机

如今，我们都在网上订购东西。然而，无论亚马逊的流程有多么自动化，在谈论 2018 年时，我们仍然有人类将包裹送到我们的门口。有时，你希望他们知道一些关于放置包裹的地方。现在我们变得越来越自动化，过去你可能会在大门外留个便条的日子已经一去不复返了。是时候用我们的技术做些有趣的事情了。要做到这一点，我们几乎不需要做任何严肃的事情。我们只需要按照以下图示连接组件即可：

![](img/b7b909b5-f145-40fc-bdb1-3d49aa5ffab9.png)

PIR 传感器必须放置在大门周围有运动时产生逻辑高电平的位置。

完成后，继续上传以下代码：

```py
import RPi.GPIO as GPIO
import time
Import os
GPIO.setmode(GPIO.BCM)
PIR = 13
GPIO.setup(PIR,GPIO.IN)
while True:

  if GPIO.input(PIR) == 1 :
     os.system('echo "Hello, welcome to my house"|festival --tts ')
     time.sleep(0.2)
     os.system('echo "If you are a delivery agent then please leave the package here"|festival --tts ')
     time.sleep(0.2)
     os.system('echo "If you are a guest then I'm sorry I have to leave I will be back after 7pm"|festival --tts ')
     time.sleep(0.2)
     os.system('echo "also Kindly don't step over the grass, its freshly grown and needs some time"|festival --tts ')
     time.sleep(1)
     os.system('echo "Thank you !"|festival --tts ')
```

现在我们所做的非常简单。一旦 PIR 传感器产生逻辑高电平，就会发出特定的指令。无需解释。如果需要澄清，可以参考之前的代码。

# 制作一个互动门 - 回答机器人

在上一章中，我们使用了 PIR 传感器来感知任何人类活动，然而传感器的问题是，无论谁来了或离开了，它都会传递相同的消息。这基本上意味着，即使你在漫长的一天后回家，它最终也会问同样的问题。相当愚蠢，是吧？

因此，在本章中，我们将使用之前的存储库，将视觉和语音整合在一起，形成一个令人惊叹的二人组。在这个过程中，摄像头将识别大门上的人，并且会识别是否是人类和陌生人，如果是的话，它会传递你打算传达的消息。另一方面，如果是你，它会简单地让你通过并问候。但是，如果检测到人脸但无法识别，则会向站在摄像头前的人提供一系列指令。

要实现这一切，你只需要在门口安装一个摄像头和 PIR。PIR 基本上是用来激活摄像头的。换句话说，只有在检测到运动时摄像头才会被激活。这个设置非常简单，不需要使用任何 GPIO。只需固定摄像头和 PIR，然后上传以下代码即可。

```py
import RPi.GPIO as GPIO
import time
Import os
import cv2
import numpy as np
import cv2

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainningData.yml")
id = 0

while True:

  GPIO.setmode(GPIO.BCM)
PIR = 13
GPIO.setup(PIR, GPIO.IN)

if GPIO.input(PIR) == 1:

  ret, img = cam.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
id, conf = rec.predict(gray[y: y + h, x: x + w])

if id == 1:
  id = "BEN"
os.system('echo "Hello, welcome to the house BEN"|festival --tts ')
time, sleep(0.2)

else :

  os.system('echo "If you are a delivery agent then please leave the package here"|festival --tts ')
time, sleep(0.2)

os.system('echo "If you are a guest then I'
    m sorry I have to leave I will be back after 7 pm "|festival --tts ')
    time, sleep(0.2)

    os.system('echo "also Kindly don'
      t step over the grass, its freshly grown and needs some time "|festival --tts ')
      time.sleep(1)

      os.system('echo "Thank you !"|festival --tts ') cv2.imshow("face", img) if cv2.waitKey(1) == ord('q'):
      break cam.release()

      cv2.destroyAllWindows()
```

```py
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```

在上述代码中，我们使用`CascadeClassifier`方法创建级联分类器，以便摄像头可以检测到人脸。

```py
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
```

在上述代码中，我们使用`cv2`的`VideoCapture(0)`方法从摄像头读取帧。此外，正在创建人脸识别器以识别特定的人脸。

```py
 ret, img = cam.read()
```

现在使用`cam.read()`从摄像头读取数据，就像在之前的代码中所做的那样。

```py
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray,1.3,5)
```

图像被转换为灰色。然后，`faceDetect.detectMultiScale()`将使用灰色转换的图像。

```py
 for (x,y,w,h) in faces:
     cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
     id, conf = rec.predict(gray[y:y+h, x:x+w])
     if id==1:
         id = "BEN" 
         os.system('echo "Hello, welcome to my house BEN"|festival --tts ')
         time, sleep(0.2)
```

当检测到人脸时，包含人脸的图像部分将被转换为灰色并传递给预测函数。该方法将告诉我们人脸是否被识别，如果识别出人脸，还会返回 ID。假设这个人是`BEN`，那么 Jarvis 会说`你好，欢迎来到我的家 BEN`。现在`BEN`可以告诉 Jarvis 打开灯，然后当唤醒词 Jarvis 被激活时，Jarvis 会做出回应。如果识别不出这个人，那么可能是个快递员。然后，执行以下命令：

```py
os.system('echo "If you are a delivery agent then please leave the package here"|festival --tts ')
time, sleep(0.2)

os.system('echo "If you are a guest then I'm sorry I have to leave I will be back after 7pm"|festival --tts ')
 time, sleep(0.2)

os.system('echo "also Kindly don't step over the grass, its freshly grown and needs some time"|festival --tts ')
time.sleep(1)

os.system('echo "Thank you !"|festival --tts ')
```

# 让 Jarvis 理解我们的声音

声音是沟通的本质。它帮助我们在很短的时间内传输大量数据。它肯定比打字更快更容易。因此，越来越多的公司正在努力制作能够理解人类语音和语言并根据其工作的系统。这绝对不容易，因为语言中存在着巨大的变化；然而，我们已经走了相当长的路。因此，不用花费太多时间，让我们的系统准备好识别我们的声音。

因此，在这里，我们将使用来自 Google Voice 的 API。您可能知道，Google 非常擅长理解您说的话。非常字面意思。因此，使用他们的 API 是有道理的。现在，它的工作方式非常简单。我们捕获声音，然后将其转换为文本。然后，我们比较文本是否与配置文件中定义的内容相似。如果匹配任何内容，则将执行与其关联的 bash 命令。

首先，我们需要检查麦克风是否连接。为此，请运行以下命令：

```py
lsusb
```

此命令将显示连接到 USB 的设备列表。如果您在列表上看到自己的设备，那么很好，您走上了正确的道路。否则，请尝试通过连接找到它，或者尝试其他硬件。

我们还需要将录音音量设置为高。要做到这一点，请继续输入以下命令：

```py
alsamixer
```

现在一旦 GUI 弹出到屏幕上，使用箭头键切换音量。

最好由您自己听取录制的声音，而不是直接将其传输到树莓派。为此，我们首先需要录制我们的声音，因此需要运行以下命令：

```py
arecord -l
```

这将检查摄像头是否在列表中。然后，输入以下命令进行录制：

```py
arecord -D plughw:1,0 First.wav
```

声音将以`First.wav`的名称记录。

现在我们也想听一下我们刚刚录制的声音。这样做的简单方法是输入以下命令：

```py
aplay test.wav
```

检查声音是否正确。如果不正确，您可以自由调整系统。

一旦我们完成了检查声音和麦克风，就该安装真正的工作软件了。有简单的方法可以做到这一点。以下是您需要运行的命令列表：

```py
wget –- no-check-certificate “http://goo.gl/KrwrBa” -O PiAUISuite.tar.gz

tar -xvzf PiAUISuite.tar.gz

cd PiAUISuite/Install/

sudo ./InstallAUISuite.sh
```

现在当您运行此程序时，将开始发生非常有趣的事情。它将开始向您提出各种问题。其中一些将是直截了当的。您可以用正确的思维以是或否的形式回答。其他可能非常技术性。由于这些问题可能随时间而变化，似乎没有必要明确提及您需要填写的答案，但作为一个一般的经验法则——除非您真的想说不，否则给出肯定的答案。

好了，我们已经安装了软件。现在在继续进行该软件之前，让我们继续编写以下程序：

```py
import RPi.GPIO as GPIO
import time
import os
GPIO.setmode(GPIO.BCM)
LIGHT = 2
GPIO.setup(LIGHT,GPIO.OUT)
GPIO.output(LIGHT, GPIO.HIGH)
os.system('echo "LIGHTS TURNED ON "|festival --tts')
```

每当此程序运行时，连接到 PIN 号为`2`的灯将被打开。此外，它将朗读`灯已打开`。将此文件保存为`lighton.py`：

```py
import RPi.GPIO as GPIO
import time
import os
GPIO.setmode(GPIO.BCM)
LIGHT = 23
GPIO.setup(LIGHT,GPIO.OUT)
GPIO.output(LIGHT, GPIO.LOW)
os.system('echo "LIGHTS TURNED OFF "|festival --tts')
```

同样，在此程序中，灯将被关闭，并且它将朗读`灯已关闭`。将其保存为`lightoff.py`：

```py
import RPi.GPIO as GPIO
import time
Import os
GPIO.setmode(GPIO.BCM)
FAN = 22
GPIO.setup(FAN,GPIO.OUT)
GPIO.output(LIGHT, GPIO.HIGH)
os.system('echo "FAN TURNED ON "|festival --tts')
```

现在我们也为风扇做同样的事情。在这个中，风扇将被打开；将其保存为`fanon.py`：

```py
import RPi.GPIO as GPIO
import time
Import os
GPIO.setmode(GPIO.BCM)
FAN = 22
GPIO.setup(FAN,GPIO.OUT)
GPIO.output(LIGHT, GPIO.LOW)os.system('echo "FAN TURNED OFF "|festival --tts')
```

我不需要为此解释相同的事情，对吧？正如您所猜到的，将其保存为`fanoff.py`。

好了！当所有这些都完成后，然后输入以下命令来检查软件是否正确安装：

```py
voicecommand -c 
```

树莓派响应唤醒词`pi`；让我们将其更改为`jarvis`。可以在打开配置文件后使用以下命令进行所有这些更改：

```py
voicecommand -e. 
```

在该文件中，输入您自己的命令。在这里，让我们添加以下代码：

```py
LIGHT_ON

LIGHT_OFF

FAN_ON

FAN_OFF
```

现在对于每个命令，定义动作。动作将是运行包含打开或关闭灯光和风扇的代码的 Python 文件。代码基本且简单易懂。将以下内容添加到文件中：

```py
LIGHT ON = sudo python lighton.py

LIGHT OFF = sudo python lightoff.py

FAN ON = sudo python fanon.py

FAN OFF = sudo python fanoff.py
```

现在，让我们看看我们做了什么。每当你说“贾维斯，开灯”，它会将你的语速转换成文本，将其与相应的程序进行比较，并执行程序中的操作。因此，在这个程序中，每当我们说“开灯”，灯就会亮起，其他命令也是类似。记得让它听到你说的话。你必须说“贾维斯”这个词，这样它才会听从命令并准备倾听。

# 总结

在这一章中，我们了解了如何与贾维斯互动，并根据我们的需求让它工作。如果这一章是关于口头交流，那么下一章将是关于手势识别，利用先进的电容技术，你将能够通过挥手来控制你的自动化系统。
