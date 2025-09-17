# 给 Jarvis 赋予声音

你是否想过使用机器人来完成我们的工作是否可能？嗯，是的！当然在一些高科技小说、漫威电影甚至漫画书中是可能的。所以，系好你的安全带，准备好进入这个令人惊叹的章节，你将实际实施我刚才提到的事情。

本章将涵盖以下主题：

+   基本安装

+   自动应答机

+   制作一个交互式门应答机器人

+   让 Jarvis 理解我们的声音

# 基本安装

有许多方法和途径可以控制我们的智能家居 Jarvis，其中一些我们在之前已经探索过，例如通过控制它。因此，为了开始，我们需要准备我们的系统以便能够进行语音合成；为了做到这一点，让我们执行以下过程。

首先，转到终端并输入以下命令：

```py
sudo apt-get install alsa-utils
```

这将安装依赖项`alsa-utils`。`alsa-utils`包包含各种有用的工具，可用于控制你的声音驱动程序。

一旦完成，你需要编辑文件。为了做到这一点，我们需要打开文件。使用以下命令：

```py
sudo nano /etc/modules
```

一旦完成，一个文件将打开；在该文件的底部，你需要添加以下行：

```py
snd_bcm2835
```

你不需要太深入地了解我们为什么要做这件事。它只是用来设置一些基础。我可以给你一个解释；然而，我不希望在这么激动人心的时刻让你感到无聊。

此外，如果你很幸运的话，有时你可能会发现这一行已经存在。如果是这样，那就让它在那里，不要去动它。

现在，为了播放 Jarvis 需要说的声音，我们需要一个音频播放器。不，不是你家里有的那个。我们说的是能够播放它的软件。

要安装播放器，我们需要运行以下命令：

```py
sudo apt-get install mplayer
```

好的，我们已经完成了音频播放器的设置；让我们看看接下来是什么。现在，再次，我们需要编辑媒体播放器的文件。我们将使用相同的步骤来打开和编辑文件：

```py
sudo nano /etc/mplayer/mplayer.conf
```

这将打开文件。和之前一样，只需添加以下行：

```py
nolirc=yes
```

最后，我们需要给它一些声音，所以运行以下命令：

```py
sudo apt-get install festvox-rablpc16k
```

这将为 Jarvis 安装一个 16 kHz、英国男性声音。我们喜欢英国口音，不是吗？

完美。一旦我们完成了之前提到的所有步骤，我们就可以开始了。为了测试声音，只需将 USB 扬声器连接到树莓派，并运行以下代码：

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

好吧，让我们看看我们实际上已经做了什么：

```py
import os
```

如你所想，我们正在导入名为`os`的库。这个库提供了一种使用操作系统依赖功能的方法：

```py
os.system('echo "Hello from the other side"|festival --tts ')
```

在这里，我们使用了一个名为 `system()` 的方法；这个方法的作用是执行一个 shell 命令。你可能想知道这是什么。shell 命令是用户用来访问系统功能并与系统交互的命令。所以现在，我们想要将我们的文本转换为语音，我们将向这个函数提供两个参数。首先，文本是什么？在我们的例子中，它是 `Hello from the other side`；这里的第二个参数是 `festival --tts`。现在 `festival` 是一个库，而 `tts` 代表文本到语音转换。所以当我们传递给参数时，系统将知道传递给参数的文本需要从文本转换为语音。

就这样！是的，就是这样。这就是我们让树莓派说话所需要做的全部。

# 自动投递应答机

这些天，我们都在网上订购东西。然而，无论亚马逊的过程多么自动化，当我们谈到 2018 年时，我们仍然有人将包裹送到我们的家门口。有时，你希望他们知道一些关于在哪里留下包裹的事情。现在我们变得越来越自动化，那些可能在你门口留下便条的日子已经过去了。是时候用我们的技术做一些真正有趣的事情了。为了做到这一点，我们几乎不需要做什么严肃的事情。我们只需要按照以下图示连接组件：

![图片](img/8c510d45-0f99-45ac-a675-5a2769835770.png)

PIR 传感器必须放置在门周围有移动时给出逻辑高电平的位置。

一旦完成，请上传以下代码：

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

现在我们所做的是非常简单的。一旦 PIR 传感器给出逻辑高电平，就会发出一定的指令。这里不需要解释。如果你需要任何澄清，可以参考之前的代码。

# 制作一个交互式门应答机器人

在上一章中，我们使用 PIR 传感器来感应任何人类活动，然而，这个传感器的问题在于，无论谁来或离开，它都会传达相同的信息。这基本上意味着，即使你在漫长的一天后回到家，它也会提出同样的问题。不是很愚蠢吗？

因此，在本章中，我们将使用之前的仓库，将视觉和语音集成在一起，以创造一个惊人的组合。在这里，摄像头将识别门上是谁，并识别是否是人或陌生人，如果是的话，它将传达你想要传达的信息。另一方面，如果你是那个人，它将简单地用一个简单的问候让你通过。然而，如果检测到人脸但没有识别出来，它将向站在摄像头前的人给出一系列指令。

要实现这一切，你只需要在门上设置一个带有 PIR（被动红外探测器）的摄像头。PIR 基本上是用来激活摄像头的。换句话说，只有在检测到没有运动时，摄像头才会被激活。这个设置非常简单，不需要使用任何 GPIO。只需固定摄像头和 PIR，然后上传以下代码：

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
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```

在前面的代码中，我们正在使用`CascadeClassifier`方法创建一个级联分类器，以便摄像头可以检测到人脸。

```py
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
```

在前面的代码中，我们正在使用`cv2`的`VideoCapture(0)`方法从摄像头读取帧。同时，正在创建人脸识别器以识别特定的人脸。

```py
 ret, img = cam.read()
```

现在，使用`cam.read()`从摄像头读取数据，就像在之前的代码中所做的那样。

```py
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray,1.3,5)
```

图像被转换为灰色。然后，`faceDetect.detectMultiScale()`将使用灰度转换后的图像。

```py
 for (x,y,w,h) in faces:
     cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
     id, conf = rec.predict(gray[y:y+h, x:x+w])
     if id==1:
         id = "BEN" 
         os.system('echo "Hello, welcome to my house BEN"|festival --tts ')
         time, sleep(0.2)
```

当检测到人脸时，包含人脸的图像部分将被转换为灰色并传递给预测函数。这种方法将告诉人脸是否为人所知，如果人脸被识别，它还会返回 ID。假设这个人是`BEN`，那么 Jarvis 会说“你好，欢迎来到我的家，BEN”。现在`BEN`可以告诉 Jarvis 打开灯光，当唤醒词 Jarvis 被激活时，Jarvis 会做出响应。如果人脸没有被识别，那么可能是一个快递员。然后，以下命令将被执行：

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

声音是沟通的精髓。它帮助我们非常快速地在短时间内传输大量数据。它当然比打字快且容易。因此，越来越多的公司正在努力开发能够理解人类声音和语言并据此工作的系统。这当然不容易，因为语言中存在巨大的变化；然而，我们已经取得了相当大的进步。所以，让我们在不浪费太多时间的情况下，让我们的系统准备好识别我们的声音。

因此，这里我们将使用 Google Voice 的 API。正如你可能知道的，谷歌在理解你说的话方面真的很擅长。比如，非常字面地。所以使用他们的 API 是有意义的。现在，它的工作方式非常简单。我们捕捉声音，并将其转换为文本。然后，我们比较文本是否与我们配置文件中定义的内容相似。如果匹配，则执行与之关联的 bash 命令。

首先，我们需要检查麦克风是否已连接。为此，运行以下命令：

```py
lsusb
```

此命令将显示连接到 USB 的设备列表。如果你在列表中看到了你的设备，那么恭喜你，你走对了路。否则，尝试通过连接找到它，或者也许尝试使用其他硬件。

我们还需要将录音音量设置为高。为此，请在串行上输入以下命令：

```py
alsamixer
```

现在，一旦 GUI 出现在屏幕上，就可以使用箭头键切换音量。

最好是亲自听一下录制的声音，而不是直接将其传给树莓派。为此，我们首先需要录制我们的声音，所以我们需要运行以下命令：

```py
arecord -l
```

这将检查摄像头是否在列表中。然后，输入以下命令进行录制：

```py
arecord -D plughw:1,0 First.wav
```

声音将以以下名称记录，`First.wav`。

现在我们还想听听我们刚才录制的声音。最简单的方法是输入以下命令：

```py
aplay test.wav
```

检查声音是否正确。如果不正确，那么你可以自由地对系统进行任何调整。

一旦我们检查完声音和麦克风，就是时候安装实际的工作软件了。有一些简单的方法可以做到这一点。以下是你需要运行的命令列表：

```py
wget –- no-check-certificate “http://goo.gl/KrwrBa” -O PiAUISuite.tar.gz

tar -xvzf PiAUISuite.tar.gz

cd PiAUISuite/Install/

sudo ./InstallAUISuite.sh
```

现在当你运行这个程序时，一些非常有趣的事情将会开始发生。它将开始问你各种问题。其中一些将是直接的。你可以用你的理智以是或否的形式回答它。其他问题可能非常技术性。由于这些问题可能会随时间而变化，似乎没有必要明确指出需要填写的答案，但作为一个一般性的规则——除非你真的想说不，否则就给它一个肯定回答。

完美，我们已经安装了软件。现在在你进一步使用该软件之前，让我们先写下以下程序：

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

无论何时运行此程序，连接在 PIN 编号`2`上的灯将会点亮。同时，它将读出`LIGHTS TURNED ON`。将此文件保存为`lighton.py`：

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

同样，在这个程序中，灯将会关闭，并且它会读出`LIGHTS TURNED OFF`。将其保存为`lightoff.py`：

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

现在我们也对风扇做了同样的事情。在这个程序中，风扇将会打开；将其保存为`fanon.py`：

```py
import RPi.GPIO as GPIO
import time
Import os
GPIO.setmode(GPIO.BCM)
FAN = 22
GPIO.setup(FAN,GPIO.OUT)
GPIO.output(LIGHT, GPIO.LOW)os.system('echo "FAN TURNED OFF "|festival --tts')
```

我不需要解释同样的事情，对吧？正如你所猜想的，保存为`fanoff.py`。

好的！当这一切都完成之后，然后输入以下命令来检查软件是否正确安装：

```py
voicecommand -c 
```

树莓派对唤醒词`pi`做出响应；让我们将其更改为`jarvis`。所有这些更改都可以在通过以下命令打开配置文件后进行：

```py
voicecommand -e. 
```

在那个文件中，输入你自己的命令。这里，让我们添加以下代码：

```py
LIGHT_ON

LIGHT_OFF

FAN_ON

FAN_OFF
```

现在对于每个命令，定义动作。动作将是运行包含切换灯和风扇开关代码的 Python 文件。代码是基本且易于理解的。将以下内容添加到文件中：

```py
LIGHT ON = sudo python lighton.py

LIGHT OFF = sudo python lightoff.py

FAN ON = sudo python fanon.py

FAN OFF = sudo python fanoff.py
```

现在，让我们看看我们完成了什么。每当你说 `<q>Jarvis，开灯</q>`，它就会将你的速度转换为文本，将其与它必须运行的相应程序进行比较，并执行程序中的任何操作。因此，在这个程序中，每当我说 `<q>开灯</q>`，灯光就会打开，其余的命令也是如此。记得让它倾听你说的话。你将不得不说出单词 `<q>Jarvis</q>`，这将让它对命令保持警觉并准备好倾听。

# 摘要

在本章中，我们了解了如何交互并让 Jarvis 根据我们的需求工作。如果本章是关于口头交流，那么下一章就是关于手势识别，在那里，通过使用先进的电容技术，你只需挥动手臂就能控制你的自动化系统。
