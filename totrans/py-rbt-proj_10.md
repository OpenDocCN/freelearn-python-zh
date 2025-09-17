# 制作一个守卫机器人

我相信你一定看过电影《我，机器人》或《超能陆战队》。看完电影后，很多人会对制作一个能够保护你的机器人产生兴趣。然而，最先进的安全系统几乎不能被归类为机器人。在本章中，我们将进一步探索视觉处理领域，制作一个守卫机器人。它的目的是守卫你的大门，如果陌生人来到门口，它会开始触发警报。然而，有趣的是，如果熟悉的人回家，机器人不会触发任何警报。更重要的是，它会清除道路，从门口区域退出，让你进入。一旦你进入，它将自动回到其位置，继续守卫并再次开始工作。

那会多么酷啊？所以让我们开始行动，把这个机器人变成现实。

# 人脸检测

现在，在我们继续进行人脸检测之前，我们需要告诉机器人什么是人脸以及它看起来是什么样子。树莓派不知道如何从南瓜中精确地分类人脸。因此，首先，我们将使用一个数据集来告诉机器人我们的脸看起来是什么样子；然后，我们将开始识别人脸。所以，让我们继续看看如何做到这一点。

首先，你需要安装一个名为 Haar-cascade 的依赖项。这是一个用于快速检测对象的级联依赖算法。为此，请在你的终端上运行以下语法：

```py
git clone https://github.com/opencv/opencv/tree/master/data/haarcascades
```

这将把`haarcascades`文件保存到你的树莓派上，你将准备好使用它。完成之后，查看以下代码，但请在逐行查看以下解释之后，只在你的树莓派上写入：

```py
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:

        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        for (x,y,w,h) in faces:
           cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

        cv2.imshow('img',img)

        k = cv2.waitKey(1) & 0xff
        if k == ord(‘q’):
                break

cap.release()
cv2.destroyAllWindows()
```

现在，这可能会看起来像是来自我们世界之外的东西，几乎所有东西都是新的，所以让我们理解我们在这里做了什么：

```py
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```

一旦我们安装了 Haar-cascade，我们基本上是在将已经训练好的数据输入到我们的树莓派中。在这一行中，我们正在打开一个分类器，它从名为`haarcascade_frontalface_default.xml`的文件中读取数据。这个文件将告诉树莓派捕获到的图像是否是正面人脸。这个文件有一个训练好的数据集，使树莓派能够做到这一点。现在，我们正在使用 OpenCV 的一个函数`CascadeClassifier()`，它使用文件中提到的学习数据，然后对图像进行分类：

```py
cap = cv2.VideoCapture(0)
```

这将捕获来自端口号为`0`的摄像头的视频。所以，每当需要捕获数据时，可以使用变量`cap`而不是编写整个程序。

```py
        ret, img = cap.read()
```

我们已经在上一章中理解了这一行。它只是从摄像头捕获图像并将其保存在名为`img`的变量中，然后`ret`将返回 true 表示捕获成功或返回 false 表示有错误。

```py
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

我们之前也使用过这一行。它的作用是，简单地使用`cv2.cvtColor()`函数转换捕获的图像。传递给它的参数如下：`img`，它基本上会告诉需要转换哪张图片。然后，`cv2.COLOR_BGR2GRAY`会告诉从哪种图像类型转换成什么类型。

```py
        faces = face_cascade.detectMultiScale(gray)
```

`face_cascade.detectMultiScale()`函数是`face_cascade`的一个函数。它检测各种大小的物体，并在其周围创建一个相似大小的矩形。返回变量`faces`的值将是检测到的物体的`x`和`y`坐标，以及物体的宽度和高度。因此，我们需要定义检测到的物体的尺寸和位置。

```py
        for (x,y,w,h) in faces:
           cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
```

在上一行代码中，我们已经取出了矩形的坐标值和高度值。然而，我们还没有在真正的图片中绘制一个矩形。这个`for`循环将要执行的操作是，使用`cv2.rectangle()`函数向图片中添加一个矩形。`img`表示需要处理的图片。`(x,y)`定义了物体的起始坐标。值`(x+w, y+h)`定义了矩形的终点。值`(255,0,0)`定义了颜色，而参数`2`定义了线的粗细。

![图片](img/ec518dd5-ca51-4f9f-8be4-289922921837.png)

```py
        cv2.imshow('img',img)
```

在这一行代码中，我们简单地使用`imshow()`函数给出最终输出，这个输出将包含我们刚刚绘制的矩形覆盖的图像。这将表明我们已经成功识别了图像。`'img'`参数将告诉窗口的名称，第二个`img`将告诉函数需要显示哪张图片。

```py
        k = cv2.waitKey(1) & 0xff
        if k == ord(‘q’):
                break
```

这一行代码只是在等待用户按下`q`键。当用户按下`q`键时，`if`语句就会变为真，从而打破无限循环。

```py
cap.release()
cv2.destroyAllWindows()
```

最后，我们使用`cap.release()`释放摄像头，然后使用`cv2.destroyAllWindows()`关闭所有窗口。

现在，运行代码并查看它是否能够检测到你的脸。祝你好运！

# 了解人脸

好的，我们已经通过几行代码检测到了人脸，但我不会认为这是一个很大的胜利，因为我们是在用其他开发者制作的剑在战斗。导入的学习集是一个通用的面部学习集。然而，在本章中，我们将继续创建我们自己的学习集来识别特定的人脸。这真的非常酷，我相信你会喜欢做这件事的。

那么，让我们开始吧。就像你之前做的那样，先通过解释来理解，然后再编写代码，这样你就能很好地理解它。

首先，我们使用程序来捕捉需要检测的物体的图像。在我们的案例中，这个物体将是一个人和他的脸。那么，让我们看看我们需要做什么：

```py
import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

sampleNum = 0

id = raw_input('enter user id')

while True:
        ret,img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
                sampleNum = sampleNum + 1
                cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",  gray[y:y+h, x:x+w])

                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
                cv2.waitKey(100)
        cv2.imshow("Face", img)
        cv2.waitKey(1)
        if sampleNum>20:

                break
cam.release()
cv2.destroyAllWindows()
```

这里是解释：

```py
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
```

如你之前所见，我们使用前面的两行代码来导入学习数据集并启动摄像头。

```py
id = raw_input('enter user id')
```

由于我们将训练系统学习特定的面孔，程序知道它检测的是谁（无论是通过名字还是 ID）非常重要。这将帮助我们明确我们检测到的是谁。因此，为了继续通过人脸检测一个人，我们需要提供他的 ID，这在以下代码中完成：

```py
        ret,img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

这与我们在代码的上一部分所做的是完全相同的。如果需要解释，请参考它。

```py
        faces = faceDetect.detectMultiScale(gray,1.3,5)
```

现在，这一行代码可能听起来也像是重复的；然而，它有所增加。在这个函数中，已经传递了两个参数而不是一个。第一个是`grey`，第二个是可以检测到的对象的最低和最高尺寸。这很重要，以确保检测到的对象足够大，以便学习过程发生。

```py
        for (x,y,w,h) in faces:
                sampleNum = sampleNum + 1
                cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",  gray[y:y+h, x:x+w])
```

在这里，我们使用相同的`for`循环来执行以下条件。因此，当检测到人脸时，循环才会为真。每次检测到人脸时，`sampleNum`变量会通过计算检测到的人脸数量增加`1`。此外，为了将图像捕获到我们的系统中，我们需要以下代码行：

```py
cv2.inwrite('dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
```

它所做的是，简单地将图像保存到名为`dataSet/User`的文件夹中。能够自己创建这样一个文件夹非常重要。如果你不这样做，当它找不到应该保存的文件夹时，就会出问题。"`+str(id)`"将通过人的 ID 保存名称，并通过使用`+str(sampleNum)`来增加样本计数。此外，我们已经提到图像将以`.jpg`格式保存，最后`gray[y:y+h, x:x+w]`是选择包含人脸的图像部分。

从这一点开始，程序的其他部分是自我解释的，我怀疑你可以自己理解。用非常简单的英语来说，这将在一个文件夹中保存图像，并且会一直这样做，直到达到 20 张图像。

现在我们已经捕获了图像，是时候让系统学习这些图像并理解如何识别它们了。为此，我们需要安装一个名为`pillow`的库。安装它很容易。你只需要在终端中写下以下行：

```py
sudo -H pip install pillow
```

这个`pillow`库将帮助我们读取数据集。我们稍后会了解更多。一旦安装了它，让我们继续看看我们是如何进行学习部分的。所以，请继续理解以下代码，然后我们就可以开始了：

```py
import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = 'dataSet'

def getImageID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

    faces=[]
    IDs=[]

    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')

       faceNp = np.array(faceImg, 'unit8')

        ID = int(os.path.split(imagePath)[-1].split('.')[1])

        faces.append(faceNp)
        print ID
        IDs.append(ID)

    return IDs, faces

Ids, faces = getImageID(path)
recognizer.train(faces, np.array(Ids))
recognizer.save('recognizer/trainningData.yml')

cv2.destroyAllWindows()
```

看到这段代码后，你可能会想到“外星人”这个词，但当你看过这个解释后，它肯定不会是外星的了。所以让我们看看：

```py
recognizer = cv2.face.LBPHFaceRecognizer_create()
```

它使用`cv2.face.LBPHFaceRecognizer_create()`函数创建一个识别器。

```py
path = 'dataSet'
```

这一行说明了捕获的数据在 Raspberry Pi 中的存储位置。我们已经用这个名字创建了一个文件夹，它包含我们之前存储的图像。

```py
def getImageID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
```

在这里，我们正在定义一个名为`getImageID(path)`的函数。

这个连接函数会将路径与`f`连接起来。现在，`f`是一个变量，它包含文件名，当它通过`os.listdir(path)`遍历定义路径内的文件列表时。

```py
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
```

这里的`for`循环将对每个我们拥有的图像都为真，并运行其内的代码。`Image.open(imagePath).convert('L')`所做的是，它只是将图像转换为单色格式。使用这一行，我们将拥有的每个图像都转换为单色。

```py
faceNp = np.array(faceImg, 'unit8')
```

OpenCV 与`numpy`数组一起工作；因此，我们需要将图像转换为所需的格式。为此，我们使用一个名为`faceNp`的变量来调用`np.array()`函数。这个函数将图像转换为名为`faceImg`的`numpy`数组，具有 8 位整数值，因为我们传递了参数`unit8`。

```py
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
```

在这一行，我们使用一个变量来调用`int()`函数，该函数将分割正在捕获的图像的路径名。我们为什么要这样做？这是为了从实际文件名中提取 ID 号码。因此，我们使用以下函数来完成这项工作：

```py
        faces.append(faceNp)
        print ID
        IDs.append(ID)
```

在这里，使用`faces.append(faceNp)`，我们将数据添加到名为`faces`的数组中，添加的数据是`faceNp`。然后，我们打印该图像的`ID`。

完成后，`IDs.append(ID)`将`ID`添加到数组`IDs`中。整个过程是作为我们将要使用的训练函数，它只接受数组形式的值。因此，我们必须将整个数据转换为数组形式并传递给训练器。

所以到目前为止的整个解释都是定义一个名为`getImageId(Path)`的函数。

```py
    return IDs, faces
```

现在这行将返回面部的`IDs`值，这些值将被进一步用于训练数据集。

```py
Ids, faces = getImageID(path)
recognizer.train(faces, np.array(Ids))
recognizer.save('recognizer/trainningData.yml')
```

在这里的第一行，`getImageID(path)`函数将接受任何图像的路径并返回图像的`Ids`。然后，`faces`将包含图像的数组数据。

最后，在`recognizer.train(faces, np.array(Ids))`中，我们使用`recognizer`的一个名为`train`的函数来根据它们的图像训练系统。传递给这里的参数是`faces`包含图像数组。此外，`np.array(Ids)`是一个数组，它是通过名为`getImageID()`的函数返回的。

一旦使用以下程序训练了系统，我们将训练数据保存到文件中。这是通过`recognizer.save()`函数完成的。传递给它的参数是保存文件的名称和扩展名。

这可能有点复杂，有时也可能令人困惑。然而，一旦你做了，它就会变得简单。现在，是你继续前进并让系统学习你的面部及其数据的时候了。

# 识别面部

现在我们已经学会了如何让我们的系统学习，现在是时候使用这些学习数据来识别面部了。所以，不多说，让我们了解这将如何完成：

```py
import numpy as np
import cv2

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()

rec.read("recognizer/trainningData.yml")
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:

   ret, img = cam.read()
   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   faces = faceDetect.detectMultiScale(gray,1.3,5)

   for (x,y,w,h) in faces:
       cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
       id, conf = rec.predict(gray[y:y+h, x:x+w])

       if id==1:
           id = "BEN"
       cv2.putText(img, str(id), (x,y+h),font,2, (255,0,0),1,)

    cv2.imshow("face", img)

    if cv2.waitKey(1)==ord('q'):
       break

cam.release()
cv2.destroyAllWindows()
```

在这段代码中，没有太多你可能遇到的新东西。它与我们在本章开始时使用的第一个代码非常相似。本质上，它也在做相同的工作。唯一的区别是它通过 ID 识别人。所以，让我们看看有什么不同，以及它的表现如何。

现在，大部分代码都是重复的。所以，我只会涉及到新的部分。下面是：

```py
font = cv2.FONT_HERSHEY_SIMPLEX
```

就像上次一样，我们在一个已识别的图像上画了一个矩形。然而，这次还需要在某个地方用文本进行叠加。所以，我们在这里选择这个程序中需要使用的字体：

```py
id, conf = rec.predict(gray[y:y+h, x:x+w])
```

这一行，预测是通过识别器的`predict()`函数进行的。它预测图像并返回检测到的图像的`id`。

```py
       if id==1:
           id = "BEN"
```

现在，最后，如果`id`等于`1`，那么`id`的值将更改为`BEN`。

```py
     cv2.putText(img, str(id), (x,y+h),font,2, (255,0,0),1,)
```

`putText()` 函数将在检测到的物体上放置文本。每个参数的定义如下：

+   `img`: 这是需要放置文本的图像。

+   `str(id)`: 这是需要打印的字符串，在我们的例子中，它将打印人的 ID。

+   `(x, y+h)`: 这是文本将被打印的位置。

+   `font`: 这是打印文本的字体。在我们的例子中，它将是之前定义的字体值。

+   `2`: 这是字体大小，也就是说字体的大小。这可以类似于放大。

+   `(255,0,0)`: 这是字体的颜色。

+   `1`: 这是字体厚度。

使用这个程序，我们可以找出学习集是否按照我们的要求工作。一旦你写好了代码，试着用它来识别人，看看它是否准确。如果准确性不满意，那么你可能需要为学习选择超过 20 个样本。进行几次试验，我确信你很快就能达到完美。

# 制作守卫机器人

现在我们已经了解了学习是如何工作的，以及如何使用学习数据来识别人，现在是时候将其付诸实践了。正如章节的名称所暗示的，我们将使用这项技术制作一个守卫机器人。现在，让我们看看下面的程序。在你开始编程之前，拿出你之前有的机器人车辆，并像我们在第六章中做的那样建立连接，*蓝牙控制机器人车*。连接电机、电机驱动器和树莓派。一旦你完成了这些，然后编写以下代码。这段代码利用了本章前面程序的所有学习内容，我们能够根据视觉处理区分入侵者和房屋居民。所以，让我们开始吧：

![图片](img/a9a88f57-272e-415d-a2b4-222f672bcaf4.png)

```py
import numpy as np
import cv2
Import RPi.GPIO as GPIO

Motor1F = 20
Motor1R = 21
Motor2F = 2
Motor2R = 3
Buzzer = 24

GPIO.setmode(GPIO.BCM)  
GPIO.setwarnings(False)
GPIO.setup(Motor1a,GPIO.OUT)
GPIO.setup(Motor1b,GPIO.OUT)
GPIO.setup(Motor2a,GPIO.OUT)
GPIO.setup(Motor2b,GPIO.OUT)
GPIO.setup(Buzzer, GPIO.OUT)

def forward():

        GPIO.output(Motor1F,1)
        GPIO.output(Motor1R,0)
        GPIO.output(Motor2F,1)
        GPIO.output(Motor2R,0)

def backward():

        GPIO.output(Motor1F,0)
        GPIO.output(Motor1R,1)
        GPIO.output(Motor2F,0)
        GPIO.output(Motor2R,1)

def stop():

        GPIO.output(Motor1F,0)
        GPIO.output(Motor1R,0)
        GPIO.output(Motor2F,0)
        GPIO.output(Motor2R,0)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainningData.yml")

id = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:

 ret, img = cam.read()
 gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 faces = faceDetect.detectMultiScale(gray,1.3,5)

 for (x,y,w,h) in faces:
     cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
     id, conf = rec.predict(gray[y:y+h, x:x+w])

     if id==1:
         id = "BEN"

         forward()
         time.sleep(1)
         stop()
         time.sleep(5)
         backward()
         time.sleep(1)

     else :

         GPIO.output(Buzzer, 1)
         time.sleep(5)

     cv2.putText(img, str(id), (x,y+h),font,2, (255,0,0),1,cv2.LINE_AA)
     cv2.imshow("face", img)

 id = 0 
 if cv2.waitKey(1)==ord('q'):
 break

cam.release()
cv2.destroyAllWindows()
```

和往常一样，我们只会关注程序中的特殊变化，大部分内容将来自上一章。因此，除非必要，我们不会重复解释。

```py
def forward():

        GPIO.output(Motor1a,0)
        GPIO.output(Motor1b,1)
        GPIO.output(Motor2a,0)
        GPIO.output(Motor2b,1)

def backward():

        GPIO.output(Motor1a,1)
        GPIO.output(Motor1b,0)
        GPIO.output(Motor2a,1)
        GPIO.output(Motor2b,0)

def stop():

        GPIO.output(Motor1a,0)
        GPIO.output(Motor1b,0)
        GPIO.output(Motor2a,0)
        GPIO.output(Motor2b,0)
```

就像回顾一样，我们定义了两个函数，分别是`backwards`、`reverse`和`stop`。这些函数将帮助我们将车辆移动到我们想要的方向。

```py
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```

这行代码正在导入名为`harrcascade_frontalface_default.xml`的先前学习的数据集。这将帮助我们识别出现在摄像头前的任何面部。

```py
 for (x,y,w,h) in faces:
     cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
     id, conf = rec.predict(gray[y:y+h, x:x+w])

     if id==1:
         id = "BEN"

         forward()
         time.sleep(1)
         stop()
         time.sleep(5)
         backward()
         time.sleep(1)

     else :

         GPIO.output(Buzzer, 1)
         time.sleep(5)
```

在这段代码中，我们正在识别面部并根据它做出决策。正如我们之前所做的那样，如果检测到面部，程序会给出相应的 ID。然而，如果面部没有被先前学习的数据集检测到，那么就不会给出任何 ID，因为这个数据集可以检测到任何面部。因此，根据程序，如果`id == 1`，那么机器人车辆会向前移动，偏离路径，然后它会停止`5`秒并回到原来的位置。如果生成的 ID 不是`1`，那么蜂鸣器会开启`5`秒，提醒用户。

通过使用这个系统，任何被识别的人都可以被允许进入 premises；然而，如果人员没有被识别，那么就会触发警报。

# 摘要

在这一章中，我们学习了如何使用预学习的数据集检测对象。我们还学习了如何为特定对象创建我们自己的学习数据集。最后，我们利用所有这些学习来制作一个守卫机器人，它将利用视觉处理的力量来守护我们的家园。
