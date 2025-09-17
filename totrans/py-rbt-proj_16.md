# 机器学习

从其原始时期到如今，机器人和计算机都在被编程执行一系列活动。这些活动可能非常庞大。因此，为了开发复杂的程序，需要大量的软件工程师日夜工作以实现特定的功能。当问题定义得很好时，这是可行的。但是，当问题也非常复杂时怎么办呢？

学习是我们人类之所以成为我们的原因。我们的经验塑造了我们以更好地、更有效地适应情况。每次我们做某事，我们就知道得更多。这使得我们在一段时间内更好地完成这项任务。俗话说，熟能生巧，而通过反复做事情来学习使我们变得更好。

然而，让我们退一步来定义什么是学习？我想引用谷歌的定义，*它是通过学习、经验或被教导获得的知识*。因此，学习基本上是一种从我们的周围环境中获取信息以理解过程及其本质的方式。

现在，你可能正在想，等等，我们不是在制作守卫机器人的前几章中使我们的系统学习了很多视觉数据吗？你的想法完全正确。然而，学习可以通过不同的方式进行。可能适用于一种类型的问题的方法可能对其他类型的问题毫无用处。因此，存在各种类型的机器学习算法及其原理。在本章中，我们将重点关注一个名为**k-最近邻**的算法。它被称为**懒惰算法**。我个人非常喜欢这个算法用于分类。为什么？因为从技术上讲，没有训练阶段。怎么呢？

k-最近邻实际上是一个智能算法。它不是对提供的数据进行回归计算并进行大量的数学计算，而是简单地从提供的数据集中获取结构化数据。每当有新的数据用于预测时，它就会根据用户提供的数据的分类，在数据库中简单地搜索与用户提供的数据最接近的*k*个匹配项。因此，在本章中，我们将学习这个算法是如何工作的，以及我们如何使用它来使我们的家庭智能化。

在本章中，我们将涵盖以下主题：

+   制作数据集

+   使用数据集进行预测

+   使你的家庭学会学习

+   家庭学习和自动化

# 制作数据集

就像在第十章制作守卫机器人中一样，我们使用了多张图片来训练模型，以确定图像中的物体是人还是其他东西。以非常相似的方式，我们不得不制作一个虚拟数据集，这样机器学习算法就可以根据这些数据预测应该做什么。

要创建一个数据集，我们需要了解正在考虑哪些数据。在本章中，我们将基于时间和温度创建一个机器学习算法，以预测风扇应该开启还是关闭。因此，至少有两件事应该由我们提供给系统，一个是`Temperature`，另一个是`Time`，以便进行预测。但有一点要记住，我们正在谈论一个监督学习算法，因此为了训练模型，我们还需要提供`Temperature`和`Time`的输出到风扇的状态。在这里，风扇的状态将是开启或关闭。因此，我们可以用`0`或`1`来表示它。现在让我们自己创建一个数据集。

现在，要创建一个数据集，你只需打开 Microsoft Excel 并按照以下方式开始编写数据集：

![图片](img/f96de387-2d29-4a30-a656-6aabaaf3b3ab.png)

拥有一个包含超过 20 组数据的数据集总是更好的。此外，数据必须具有独特的特征，而不是随机数据。例如，在前面的例子中，你可以看到当温度为`28`时，在`12.44`时风扇是开启的；然而，在相同的时间，当时间是`12.13`且温度为`21`时，风扇是关闭的。

一旦你创建了一个数据集，你必须以`dataset`的名称将其保存为 CSV 格式。可能会有一些用户不会使用 Microsoft Excel，在这种情况下，你可以在文本编辑器中以相同的格式写入数据，最后以 CSV 格式保存。

一旦你有了`dataset.csv`文件，你必须将它们复制到将要保存即将到来的代码的地方。一旦你完成，我们就可以继续下一步。

记住，数据的质量越好，学习过程就越好。所以你可能需要花一些时间，仔细制作你的数据集，以确保它有意义。

# 使用数据集进行预测

不多说了，让我们看看下面的代码：

```py
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
data = pd.read_csv('dataset.csv')

x = np.array(data[['Time', 'Temp']])
y = np.array(data[['State']]).ravel()

knn.fit(x,y)

time = raw_input("Enter time")
temp = raw_input("Enter temp")

data =. []

data.append(float(time))
data.append(float(temp))

a = knn.predict([data])

print(a[0])}
```

那么，让我们看看我们在做什么：

```py
import numpy as np
```

我们将`numpy`导入到我们的程序中；这有助于我们处理列表和矩阵：

```py
import pandas as pd
```

这里，我们导入了一个名为`pandas`的库；这有助于我们读取以逗号分隔的值文件，换句话说，CSV 文件。我们将使用 CSV 文件来存储我们的数据，并在学习过程中访问它：

```py
from sklearn.neighbors import KNeighborsClassifier
```

这里，我们从`sklearn`库中导入`KneighborsClassifier`。`sklearn`本身是一个庞大的库；因此，我们只导入其中的一部分，因为我们不会在这个程序中使用它：

```py
knn = KNeighborsClassifier(n_neighbors=5)
```

这里，我们给变量`knn`赋值，其值将是`KNeighborsClassifer(n_neighbors =5)`；这意味着我们正在使用带有参数`n_neighbors=5`的`KneighborsClassifer()`函数。这个参数告诉`KneighborsClassifer`函数，在算法中我们将有五个邻居。进一步来说，使用这个声明，整个函数可以通过`knn`来调用：

```py
data = pd.read_csv('dataset.csv')
```

在这里，我们正在为名为`data`的变量赋值，传递的值是`pd.read_csv('dataset.csv')`；这意味着每当调用`data`时，就会调用来自`pandas`库的`pd.read_csv()`函数。该函数的目的是从 CSV 文件中读取数据。在这里，传递的参数是`dataset.csv`；因此，它指示函数将读取哪些数据。在我们的例子中，它将从一个名为`dataset.csv`的文件中读取：

```py
x = np.array(data[['Time', 'Temp']])
```

在下一行中，我们正在将值传递给变量`x`，传递的值是`np.array(data[['Time', 'Temp']])`。现在，使用`numpy`库的`np.array`函数创建一个数组。这个数组将按`Time`和`Temp`的名称存储数据：

```py
y = np.array(data[['State']]).ravel()
```

就像之前一样，我们使用`numpy`库的`.ravel()`函数将`State`存储在一个数组中。这样做是为了使数学函数可以在两个数组——`x`和`y`之间进行操作：

```py
knn.fit(x,y)
```

在这短短的一行中，我们使用`knn`库中的`fit()`函数，它的作用是使用`x`作为主要数据，`y`作为输出结果数据来拟合模型：

```py
time = raw_input("Enter time")
temp = raw_input("Enter temp")
```

在这一行中，我们请求用户的数据。在第一行，我们将打印`Enter time`，然后等待用户输入时间。用户输入时间后，它将被存储在名为`time`的变量中。一旦完成，然后就会进入下一行；代码将打印`Enter temp`。一旦提示用户，它将等待收集数据。一旦用户收集到数据，它将把数据存储在名为`temp`的变量中：

```py
data =. []
```

在这里，我们创建了一个名为`data`的空列表；这个列表将用于计算输出结果的状态。因为所有的机器学习算法都是在列表数据类型上工作的。因此，输入必须以列表的形式给出，以便进行决策：

```py
data.append(float(time))
data.append(float(temp))
```

在这里，我们将数据添加到我们刚刚创建的名为`data`的列表中。首先，添加`time`，然后是`temp`：

```py
a = knn.predict([data])
```

一旦完成，将使用名为`predict`的`knn`算法中的函数来根据名为`data`的列表预测输出。预测算法的输出被检索到一个名为`a`的变量中：

```py
print(a[0])
```

最后，一旦完成预测，我们就会读取`a`的值，并记住所有数据输入/输出都是以列表的形式发生的。因此，预测算法给出的数据输出也将以列表格式。因此，我们正在打印列表的第一个元素。

这个输出将预测根据用户提供的数据集，风扇将处于哪种状态。所以，请继续提供一个温度和时间，让系统为您预测结果。看看它是否工作正常。如果不正常，那么尝试向 CSV 文件中添加更多的数据集，或者看看数据集中的值是否有意义。我相信你最终会得到一个出色的预测系统。

# 让你的家学会

一旦完成这一宪法，就可以按照下面的图示进行接线：

![](img/855c4bae-d685-44b1-bb0c-5926da8035a2.png)

一旦设置好，就轮到我们将在我们的树莓派上编写以下代码：

```py
import Adafruit_DHT
import datetime
import RPi.GPIO as GPIO
import time
import numpy as np
import pandas as pd
import Adafruit_DHT
from sklearn.neighbors import KNeighborsClassifier

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

fan = 22
light = 23
sw1 = 13
sw2 = 14

GPIO.setup(led1,GPIO.OUT)
GPIO.setup(led2,GPIO.OUT)
GPIO.setup(sw1,GPIO.IN)
GPIO.setup(sw2,GPIO.IN)

sensor = 11
pin = 2

f = open("dataset.csv","a+")
count = 0
while count < 50:

 data = ""

 H = datetime.datetime.now().strftime('%H')
 M = datetime.datetime.now().strftime('%M')

 data = str(H)+"."+str(M)
 humidity,temperature = Adafruit_DHT.read_retry(sensor,pin)
 data = data + "," + str(temperature)

prev_state = state

 if (GPIO.input(sw1) == 0) and (GPIO.input(sw2) == 0):
     state = 0
     GPIO.output(light,GPIO.LOW)
     GPIO.output(fan,GPIO.LOW)

 elif (GPIO.input(sw1) == 0) and (GPIO.input(sw2) == 1):
     state = 1
     GPIO.output(light,GPIO.HIGH)
     GPIO.output(fan,GPIO.LOW)

 elif (GPIO.input(sw1) == 1) and (GPIO.input(sw2) == 0):
    state = 2
     GPIO.output(light,GPIO.LOW)
     GPIO.output(fan,GPIO.HIGH)

 elif (GPIO.input(sw1) == 1) and (GPIO.input(sw2) == 1):
    state = 3
     GPIO.output(light,GPIO.HIGH)
     GPIO.output(fan,GPIO.HIGH)

 data = ","+str(state)

if prev_state =! state:

     f.write(data)
     count = count+1

f.close()
```

现在，让我们看看我们在这里做了什么：

```py
f = open("dataset.csv","a+")
```

在这一行代码中，我们将`open("dataset.csv", "a+")`的值赋给了变量`f`。之后，`open()`函数将打开传递给其参数的文件，在我们的例子中是`dataset.csv`；参数`a+`表示在 CSV 文件的末尾追加值。因此，这一行将打开`dataset.csv`文件，并添加我们稍后传递的值：

```py
 data = ""
```

我们通过名为`data`的名称声明了一个空字符串：

```py
 data = str(H)+"."+str(M)
```

我们正在将小时和分钟的值添加到字符串中，中间用点分隔以示区别。因此，数据将看起来像`HH.MM`：

```py
 humidity,temperature = Adafruit_DHT.read_retry(sensor,pin)
```

我们使用这一行来读取 DHT 11 传感器的湿度和温度读数，并将这些值传递给变量`humidity`和`temperature`：

```py
data = data + "," + str(temperature)
```

一旦读取数据，我们将温度也添加到变量`data`中。因此，现在数据将看起来像这样`HH.MM`和`TT.TT`：

```py
 if (GPIO.input(sw1) == 0) and (GPIO.input(sw2) == 0):
 state = 0
 elif (GPIO.input(sw1) == 0) and (GPIO.input(sw2) == 1):
 state = 1
 elif (GPIO.input(sw1) == 1) and (GPIO.input(sw2) == 0):
 state = 2
 elif (GPIO.input(sw1) == 1) and (GPIO.input(sw2) == 1):
 state = 3
```

在这里，我们定义了不同类型的与开关组合相对应的状态。相应的表格如下：

| **开关 1** | **开关 2** | **状态** |
| --- | --- | --- |
| `0` | `0` | `0` |
| `0` | `1` | `1` |
| `1` | `0` | `2` |
| `1` | `1` | `3` |

因此，通过状态值，我们可以理解哪个开关会被打开，哪个会被关闭：

```py
 data = ","+str(state)
```

最后，将状态值也添加到名为`data`的变量中。现在，数据将看起来像`HH.MM`，`TT.TT`和`S`：

```py
f.write(data)
```

现在，使用`write()`函数，我们将数据的值写入我们之前通过值`f`定义的文件。

因此，每当单个开关打开或关闭时，数据都会被收集，并且值会记录在该文件中的时间戳。然后，可以使用这些数据来预测在任何给定时间家里的状态，而不需要任何干预：

```py
if prev_state =! state:

     f.write(data)
     count = count+1
```

在这里，我们正在将状态与`prev_state`进行比较，正如你在我们的程序中看到的那样。上一个状态是在我们程序开始时计算的。所以，如果系统状态有任何变化，那么`prev_state`和`state`的值就会不同。这将导致`if`语句为真。当这种情况发生时，数据将通过`write()`函数写入我们的文件。传递的参数是需要写入的值。最后，计数器的值增加`1`。

一旦这个程序运行几个小时或者可能是几天，它就会收集一些关于你的灯光和风扇开关模式的有用数据。然后，这些数据可以被检索到之前的程序中，它将能够根据时间和温度做出自己的决定。

# 家庭学习和自动化

既然在前一节我们已经了解了学习是如何工作的，现在是时候使用这个概念来制作一个能够自动理解我们如何运作和做决定的机器人了。基于我们的决定，系统将判断应该做什么。但这次，我们不是通过用户提供一组数据，而是让这个程序自己生成数据。一旦数据看起来足够它自己运作，那么，不多做解释，让我们直接进入正题：

```py
import Adafruit_DHT
import datetime
import RPi.GPIO as GPIO
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

light = 22
fan = 23
sw1 = 13
sw2 = 14

GPIO.setup(light,GPIO.OUT)
GPIO.setup(fan,GPIO.OUT)
GPIO.setup(sw1,GPIO.IN)
GPIO.setup(sw2,GPIO.IN)

sensor = 11
pin = 2

f = open("dataset.csv","a+")
count = 0

while count < 200:

        data = ""

        H = datetime.datetime.now().strftime('%H')
        M = datetime.datetime.now().strftime('%M')

        data = str(H)+"."+str(M)
        humidity,temperature = Adafruit_DHT.read_retry(sensor,pin)
        data = data + "," + str(temperature)

prev_state = state

 if (GPIO.input(sw1) == 0) and (GPIO.input(sw2) == 0):
     state = 0
     GPIO.output(light,GPIO.LOW)
     GPIO.output(fan,GPIO.LOW)

 elif (GPIO.input(sw1) == 0) and (GPIO.input(sw2) == 1):
     state = 1
     GPIO.output(light,GPIO.HIGH)
     GPIO.output(fan,GPIO.LOW)

 elif (GPIO.input(sw1) == 1) and (GPIO.input(sw2) == 0):
    state = 2
     GPIO.output(light,GPIO.LOW)
     GPIO.output(fan,GPIO.HIGH)

 elif (GPIO.input(sw1) == 1) and (GPIO.input(sw2) == 1):
    state = 3
     GPIO.output(light,GPIO.HIGH)
     GPIO.output(fan,GPIO.HIGH)

 data = ","+str(state)

 if prev_state =! state:

     f.write(data)
     count = count+1

Test_set = []
knn = KNeighborsClassifier(n_neighbors=5)
data = pd.read_csv('dataset.csv')

X = np.array(data[['Time', 'Temp']])
y = np.array(data[['State']]).ravel()

knn.fit(X,y)

While Count > 200:

    time = ""

    H = datetime.datetime.now().strftime('%H')
    M = datetime.datetime.now().strftime('%M')

    time = float(str(H)+"."+str(M))

    humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)

 temp = int(temperature)
 test_set.append(time)
 test_set.append(temp)

 a = knn.predict([test_set]])
 Out = a[0]

 If out == 0:
 GPIO.output(light,GPIO.LOW)
 GPIO.output(fan,GPIO.LOW)

 If out == 1:
 GPIO.output(light,GPIO.LOW)
 GPIO.output(fan,GPIO.HIGH)

 If out == 2:
 GPIO.output(light,GPIO.HIGH)
 GPIO.output(fan,GPIO.LOW)

 If out == 3:
 GPIO.output(light,GPIO.HIGH)
 GPIO.output(fan,GPIO.HIGH)

```

现在我们来看看我们在这里做了什么。在这个程序中，`while count < 200:`条件下的程序的第一部分与我们之前所做的代码完全相同。所以它只是在按照用户的要求做事，同时，它正在从用户那里获取值以了解他们的工作行为：

```py
while count > 200:
```

此后，当计数器超过`200`时，将开始执行代码的第二部分，这部分代码位于前面的循环中：

```py
    time = ""
```

在这一行中，我们创建了一个名为`time`的空字符串，我们将在这里存储时间的值：

```py
    H = datetime.datetime.now().strftime('%H')
    M = datetime.datetime.now().strftime('%M')
```

我们将时间的值存储在名为`H`和`M`的变量中：

```py
    time = float(str(H)+"."+str(M))
```

我们现在将时间的值存储在字符串`time`中。这将包括小时和分钟：

```py
 temp = int(temperature)
```

为了方便计算和减少系统计算负载，我们正在减小温度变量的尺寸。我们通过删除小数点来实现这一点。为了做到这一点，`TT.TT`；我们只是消除了小数点，将其转换为整数。这是通过名为`int()`的函数完成的。温度的整数值将存储在名为`temp`的变量中：

```py
 test_set.append(time)
 test_set.append(temp)
```

在这里，我们将时间和温度的值添加到名为`test_set`的列表中，如果你查看程序，你会看到程序中间声明了一个空集合。所以，现在这个`test_set`包含了`time`和`temp`的值，这些值可以被预测算法进一步用于预测状态：

```py
 a = knn.predict([test_set]])
```

使用`knn`函数中的简单函数`predict()`，我们可以预测状态值。我们所需做的只是将数据或`test_set`列表传递给预测函数。该函数的输出将是一个列表，将被存储在一个名为`a`的变量中：

```py
 Out = a[0]
```

`Out`的值将被设置为列表`a`的第一个元素：

```py
 If out == 0:
 GPIO.output(light,GPIO.LOW)
 GPIO.output(fan,GPIO.LOW)

 If out == 1:
 GPIO.output(light,GPIO.LOW)
 GPIO.output(fan,GPIO.HIGH)

 If out == 2:
 GPIO.output(light,GPIO.HIGH)
 GPIO.output(fan,GPIO.LOW)

 If out == 3:
 GPIO.output(light,GPIO.HIGH)
 GPIO.output(fan,GPIO.HIGH)
```

使用前面的代码块，我们能够根据算法预测的状态来选择性地打开灯和风扇。因此，使用这种方法，程序能够在没有你的干预下自动预测并打开或关闭灯和风扇。

# 摘要

在本章中，我们了解了即使不学习也能如何进行机器学习。我们了解了如何提供数据集，以及我们可以如何使用现有系统创建一个新的数据集。最后，我们了解了系统如何无缝地收集数据、从数据中学习，并最终提供输入。想要构建一个轮式自平衡机器人？那么，下一章见！
