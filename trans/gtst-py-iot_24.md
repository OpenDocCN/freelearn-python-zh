# 第二十四章：基本开关

到目前为止一定是一段史诗般的旅程！回想一下你开始阅读这本书的时候，你是否曾想象过事情会变得如此简单？值得注意的是，一切都是从非常简单的开始，然后慢慢地，随着对更复杂系统的需求，技术的复杂性也增加了。回到个人计算并不是真正的事情的时候。它只在商业中使用，像 IBM 这样的公司只为商业客户提供服务。那时，想要个人计算机的人只有一个选择。他们需要从头开始建造，老实说，很多人过去都这样做。至少从我的角度来看，这真的并不难。但是，与那个时代相比，想想它们现在变成了什么样子。曾经想过在家里建造一台计算机吗？我说的是设计一切，而不仅仅是组装 CPU。这并不容易。

我在这里想告诉你的是，曾经有一段时间，计算机是稀有的；它们并不常见，功能也非常有限。然而，随着时间的推移和像史蒂夫·乔布斯、比尔·盖茨、休利特和帕卡德这样的人的智慧，计算机变得更加用户友好，更容易获得，并成为一种令人向往的商品。想象一下同样的情况发生在机器人身上。它们很昂贵；对于大多数人来说，它们并没有太多用处，而且在公共场所也很少见。但是，正如你所学到的，为我们个人使用构建机器人并不是很难，再加上一些调整和你这样有创造力的头脑，事情可以朝着完全不同的方向发展。你可能会因为你的愿景而受到嘲笑。但请记住，每个发明家在某个时候都被称为疯子。所以下次有人称你为疯子时，你可以非常确定你正在进步！

嗯，我非常确定，如果你是一个机器人爱好者，那么你一定看过电影《钢铁侠》。如果你还没有看过，那就停下来阅读这本书，去打开 Netflix 看看那部电影。

有一次我看了那部电影，我想要制作两件东西：一件是钢铁侠的战衣，另一件是他的个人助手贾维斯，他照顾他的一切需求。虽然战衣似乎是我可能需要一段时间来研究的东西，但到那时，你可以继续为自己建立个人助手。

想象一下你的家自己做事情。那会多酷啊？它知道你喜欢什么，你什么时候醒来，你什么时候回家，基于此，它会自动为你做事情。最重要的是，它不会是你从货架上购买的东西，而是你亲手制作的。

在你做任何这些之前，我必须告诉你，你将处理高电压和相当大的电流。电力不是闹着玩的，你必须随时小心并佩戴所有安全设备。如果你不确定，那么最好找一个电工来帮助你。在触摸或打开任何电气板之前，确保你穿着不导电的鞋子；还要检查螺丝刀、钳子、鼻钳、剪刀和其他工具是否绝缘良好且处于良好状态。戴手套是个好主意，增加安全性。如果你未满 18 岁，那么你必须有一个成年人随时帮助你。

既然说到这里，让我们开始看看我们有什么。

# 让贾维斯叫醒你

现在，这个非常有趣，正如大家所知，我们的人体是按照一定的方式编程的。因此，我们对不同的刺激作出非常熟悉的反应。比如当天黑了，我们的大脑会产生触发睡眠的激素。一旦阳光照到我们的眼睛，我们就会醒来。好吧，至少应该是这样！最近，我们的生活方式发生了巨大变化，开始违背这种周期。这就是为什么我们看到越来越多的失眠病例。被闹钟吵醒绝对不是自然的。因此，即使它的铃声是您最喜欢的歌曲，您早上听到闹钟也不会开心。我们的睡眠周期应该与阳光同步，但现在几乎没有人会通过这种方式醒来。因此，在本章中，让我们首先制作一个智能闹钟，模拟我们醒来的自然方式。

# 使用继电器和 PIR 传感器

由于我们正在处理高电压和更高电流，我们将使用继电器。为此，请按以下方式连接电线：

![](img/7f4fd337-c5a2-4a66-93c5-8e0155e8e213.png)

连接完成后，上传以下代码，让我们看看会发生什么：

```py
import RPi.GPIO as GPIO import time LIGHT = 23 GPIO.setmode(GPIO.BCM) GPIO.setwarnings(False) GPIO.setup(LIGHT,GPIO.OUT) import datetime H = datetime.datetime.now().strftime('%H') M = datetime.datetime.now().strftime('%M') 
 while True: if H = '06'and M < 20 : GPIO.output(LIGHT,GPIO.HIGH) else: GPIO.output(LIGHT,GPIO.LOW)
```

好的，这是一个非常简单的代码，不需要太多解释。我们以前也做过一个非常类似的代码。你还记得吗？那是在最初的几章，当我们正在制作一个浇水机器人时，我们必须在特定时间给植物浇水。现在它所做的就是检查时间，以及时间是否为`06`小时，分钟是否小于`20`。也就是说，灯会在 07:00 到 07:19 之间打开。之后，它会关闭。

# 制作令人讨厌的闹钟

但是有一个问题。问题是灯会打开，无论您是否起床，灯都会在 20 分钟内自动关闭。这有点问题，因为您并不是每次都会在 20 分钟内醒来。那么，在这种情况下，我们应该怎么办呢？我们需要做的第一件事是检测您是否醒来了。这非常简单，这里不需要太多解释。如果您早上醒来，非常肯定您会离开床。一旦您离开床，我们就可以检测到运动，告诉我们的自动系统您是否真的醒来了。

现在，我们可以在这里做一些非常简单的事情。我们可以检测您的动作，并根据检测结果决定您是否真的醒来了。这似乎不是什么大任务。我们只需要添加一个运动检测传感器。为此，我们可以使用 PIR 传感器，它可以告诉我们是否检测到了运动。所以，让我们继续，在我们的系统顶部添加另一层传感器，看看会发生什么。

首先，按以下方式连接电路。在安装 PIR 传感器时，请确保它面向床，并检测其周围的任何运动。一旦 PIR 设置好，将传感器连接如下图所示，并看看会发生什么：

![](img/e0c96c13-a539-42bc-bf6c-ae78bdad8e4c.png)

完成后，继续编写以下代码：

```py
import RPi.GPIO as GPIO import time LIGHT = 23 PIR = 24 Irritation_flag = 3  GPIO.setmode(GPIO.BCM) GPIO.setwarnings(False) GPIO.setup(LIGHT,GPIO.OUT) GPIO.setup(PIR, GPIO.IN) import datetime H = datetime.datetime.now().strftime('%H') M = datetime.datetime.now().strftime('%M')       while True:

        if H = '07' and M <= '15' and Iriitation_Flag > 0 and GPIO.input(PIR) == 0:

  GPIO.output(LIGHT,GPIO.HIGH)

  if H = '07'and GPIO.input(PIR)==1:

 GPIO.output(LIGHT,GPIO.LOW)
            time.sleep(10) Irritation_Flag = Irritation_Flag - 1  for H = '07'and M > '15' and Irritation_Flag > 0 and GPIO.input(PIR) = 0: GPIO.output(LIGHT,GPIO.HIGH)
            time.sleep(5) GPIO.output(LIGHT,GPIO.LOW)
            time.sleep(5)  if H != '07':

            Irritation_flag = 3
            GPIOP.output(LIGHT, GPIO.LOW)  
```

好的，让我们看看我们做了什么。代码非常简单，但我们在其中有一个小变化，那就是“烦躁标志”：

```py
Irritation_flag = 3
```

现在，这个变量的作用有点像贪睡按钮。我们知道，当我们醒来时，有时，或者事实上，大多数时候，我们会再次回去睡觉，直到很久以后才意识到我们迟到了。为了防止这种情况，我们有这个“烦躁标志”，它的基本作用是检测您停止闹钟的次数。我们稍后会看到它的使用方法：

```py
        if H = '07' and M <= '15' and Irritation_Flag > 0 and GPIO.input(PIR) == 0:

  GPIO.output(LIGHT,GPIO.HIGH)
```

在这一行中，我们只是比较小时和分钟的时间值。如果小时是`07`，分钟少于或等于`15`，那么灯将关闭。还有一个条件是`Irritation_Flag > 0`，因为我们在开始时已经声明了`Irritation_flag = 3`；因此，最初这个条件总是为真。最后一个条件是`GPIO.input(PIR) == 0`；这意味着只有当 PIR 没有检测到任何运动时，条件才会满足。简单地说，如果 PIR 没有检测到任何运动，那么闹钟将在每天 07:00 和 07:15 之间响起：

```py
  if H = '07'and GPIO.input(PIR)==1:

 GPIO.output(LIGHT,GPIO.LOW)
            time.sleep(10) Irritation_Flag = Irritation_Flag - 1
```

在程序的这一部分，只有当小时或`H`等于`7`并且 PIR 检测到一些运动时，条件才会为真。因此，每当时间在 07:00 和 07:59 之间，以及每当检测到运动时，条件就会为真。一旦为真，程序将首先使用`GPIO.output*LIGHT,GPIO.LOW`关闭灯。一旦关闭，它会使用`time.sleep(10)`等待`10`秒。时间到后，它将执行以下操作：`Irritation_Flag - Irritation_Flag - 1`。现在它所做的是每次检测到运动时将`Irritation_Flag`的值减少`1`。因此，第一次发生运动时，`Irritation_Flag`的值将为`2`；之后将为`1`，最后将为`0`。

如果你看一下代码的前一部分，你会发现只有当`Irritation_Flag`的值大于`0`时，灯才会打开。因此，如果你想关闭灯，你至少要移动三次。为什么是三次？因为代码`Irritation_Flag = Irritation - 1`将被执行三次，以使值减少到`0`，这显然会使条件`GPIO.input(PIR) > 0`为假：

```py
  for H = '07'and M > '15' and Irritation_Flag > 0 and GPIO.input(PIR) = 0: GPIO.output(LIGHT,GPIO.HIGH)
            time.sleep(5) GPIO.output(LIGHT,GPIO.LOW)
            time.sleep(5) 
```

现在，假设即使经过了所有这些，你仍然没有醒来。那么应该发生什么？我们在这里为您准备了一些特别的东西。现在，我们不是使用`if`条件，而是使用`for`循环。它将检查时间是否为`07`小时，分钟是否大于`15`，`Irritation_Flag > 0`，显然没有检测到运动。只要所有这些条件都为真，灯就会在之后打开`5`秒，使用`time.sleep(5)`保持打开。然后灯会再次打开。这将一直持续下去，直到条件为真，或者换句话说，直到时间在 07:15 和 07:59 之间。`Irritation)_Flag > 0`，也就是说，连续三次未检测到运动。在此期间，for 循环将继续打开和关闭灯。由于频繁的灯光闪烁，你醒来的机会非常高。这可能非常有效，但肯定不是最方便的。然而，无论多么不方便，它仍然比传统的闹钟要好：

```py
 if H != '07':

            Irritation_flag = 3
```

我们已经准备好了整个基于灯光的闹钟，可以在每天早上叫醒我们。但是，有一个问题。一旦关闭，`Irritation_Flag`的值将为`0`。一旦变为`0`，无论时间如何，灯都不会启动。因此，为了确保闹钟每天都在同一时间运行，我们需要将标志的值设置为大于`0`的任何数字。

现在，在前一行中，如果`H != '07'`，那么`Irritation_flag`将为`3`。也就是说，每当时间不是`07`小时时，`Irritation_Flag`的值将为`3`。

这很简单，不是吗？但我相信它会很好地确保你按时醒来。

# 让它变得更加恼人

您能完全依赖前面的系统吗？如果您真的能控制自己早上不想起床的情绪，那么，是的，您可以。但对于那些喜欢躺在床上并在按掉贪睡按钮后再次入睡的人来说，我相信您一定能找到一种方法来关闭灯光而不是真正醒来。因此，就像代码中一样，当检测到运动三次时，灯光会关闭。但运动可以是任何东西。您可以在床上挥手，系统会将其检测为运动，这将违背整个目的。那么现在我们该怎么办呢？

我们有一个解决方案！我们可以使用一种方法，确保您必须起床。为此，我们将使用我们之前在项目中使用过的红外近距传感器，并根据传感器的距离读数，我们可以检测您是否已经穿过了特定区域。这可能非常有趣，因为您可以将该传感器安装在床的另一侧，或者可能安装在浴室的门口，直到您穿过特定线路为止。系统不会关闭闹钟。所以让我们看看我们将如何做。首先，按照以下图表连接硬件：

![](img/4b798678-b477-4322-86ad-e480bc2a2583.png)

完成图表后，继续上传以下代码：

```py
import RPi.GPIO as GPIO import time import Adafruit_ADS1x15 adc0 = Adafruit_ADS1x15.ADS1115() GAIN = 1  adc0.start_adc(0, gain=GAIN)  LIGHT = 23 PIR = 24 Irritation_flag = 1 IR = 2 GPIO.setmode(GPIO.BCM) GPIO.setwarnings(False) GPIO.setup(LIGHT,GPIO.OUT) GPIO.setup(PIR, GPIO.IN)
GPIO.setup(IR. GPIO.IN) import datetime H = datetime.datetime.now().strftime('%H') M = datetime.datetime.now().strftime('%M')       while True:

  if H = '07' and M <= '15' and Iriitation_Flag > 0 and GPIO.input(PIR) == 0:

  GPIO.output(LIGHT,GPIO.HIGH)

  if H = '07'and GPIO.input(PIR)==1: M_snooze = datetime.datetime.now().strftime('%M')
   M_snooze = M_snooze + 5
 for M <= M_snoozeGPIO.output(LIGHT,GPIO.LOW) F_value = adc0.get_last_result()  F1 = (1.0  / (F_value /  13.15)) -  0.35

     time.sleep(0.1)

     F_value = adc0.get_last_result()  F2 = (1.0  / (F_value /  13.15)) -  0.35

     F_final = F1-F2 M = datetime.datetime.now().strftime('%M') if F_final > 25

         Irritation_flag = 0     for H = '07'and M > '15' and Irritation_Flag > 0 and GPIO.input(PIR) = 0: GPIO.output(LIGHT,GPIO.HIGH)
 time.sleep(5) GPIO.output(LIGHT,GPIO.LOW)
 time.sleep(5)  if H != '07':

 Irritation_flag = 1 
```

震惊了吗？这段代码似乎相当复杂，内部嵌套了条件，再加上更多的条件。欢迎来到机器人领域！这些条件构成了大部分机器人的编程。机器人必须不断观察周围发生的事情，并根据情况做出决策。这也是人类的工作方式，不是吗？

说了这么多，让我们看看我们实际上在这里做了什么。大部分代码基本上与上一个相同。主要区别在于编程部分的中间某处：

```py
  if H = '07' and M <= '15' and Iriitation_Flag > 0 and GPIO.input(PIR) == 0:

  GPIO.output(LIGHT,GPIO.HIGH)
```

我们会在时间介于 07:00 和 07:15 之间时打开灯光：

```py
  if H = '07'and GPIO.input(PIR)==1: M_snooze = datetime.datetime.now().strftime('%M')
   M_snooze = M_snooze + 5
```

在`07`点的时候，每当 PIR 传感器被触发，或者换句话说，PIR 传感器检测到任何运动，那么它将在`if`条件内执行一系列活动，包括通过函数`datetime.datetime.now().strftime('%M')`记录时间，然后将其存储在名为`M_snooze`的变量中。

在下一行，我们取出存储在`M_snooze`中的分钟值，并再加上`5`分钟。因此，`M_snooze`的值现在增加了`5`：

```py
 for M <= M_snooze 
```

现在，在我们之前使用的相同`if`条件中，我们放置了一个`for`循环，看起来像这样：`for M <= M_snooze`。但这是什么意思？在这里，我们所做的事情非常简单。`for`循环内的程序将继续运行，并且会一直保持在循环中，直到我们所述的条件为真。现在，这里的条件规定了只要`M`小于或等于`M_snooze`的时间，条件就会保持为真。正如您之前学到的，`M`是当前的分钟值，而`M_snooze`是循环开始时的`M`的值，增加了`5`。因此，循环将在开始时的`5`分钟内保持为真：

```py
 GPIO.output(LIGHT,GPIO.LOW) F_value = adc0.get_last_result()  F1 = (1.0  / (F_value /  13.15)) -  0.35

     time.sleep(0.1)

     F_value = adc0.get_last_result()  F2 = (1.0  / (F_value /  13.15)) -  0.35

     F_final = F1-F2
```

现在，这是程序中最有趣的部分。直到`for M <= M_snooze`为真，前面的代码行将运行。让我们看看它在做什么。在`F-value = adc0.get_last_result()`这一行中，它获取红外距离传感器的值并将其存储在`F_value`中。然后，在`F1 = (1.0/(F_value/13.15))-0.35`这一行中，我们简单地计算了以厘米为单位的距离。我们已经学习了这是如何发生的，所以这里不需要做太多解释。距离的值存储在一个名为`F1`的变量中。然后，使用`time.sleep(0.1)`函数，我们暂停程序`0.1`秒。然后，我们再次重复相同的任务；也就是说，我们再次获取距离的值。但是这次，计算出的距离值存储在另一个名为`F2`的变量中。最后，在所有这些都完成之后，我们计算`F_final`，即`F_final = F1 - F2`。所以我们只是计算了第一次和第二次读数之间的距离差。但是，你可能会问我们为什么要这样做。这有什么好处呢？

嗯，你还记得，我们把红外距离传感器放在浴室门口。现在，如果没有人经过，数值将保持相当恒定。但是每当有人经过时，距离就会发生变化。因此，如果从第一次到最后一次读数的总距离发生变化，那么我们可以说有人通过了红外传感器。

这很酷，但为什么我们不像以前那样保留一个阈值呢？答案很简单。因为如果你需要改变传感器的位置，那么你又需要根据位置重新校准传感器。所以这是一个简单但健壮的解决方案，可以在任何地方使用：

```py
 if F_final > 10

        Irritation_flag = 1
```

现在我们已经得到了读数，可以告诉我们是否有人经过。但是除非我们把它放在某个地方，否则这些数据是没有用的。

所以，在条件`if F_final > 10`中，每当距离变化超过`10`厘米时，条件就会成立，`Irritation_flag`将被设置为`1`。

如果你回到前面的行，你就会发现只有在时间在 07:00 和 07:15 之间，且`Irritation_flag`必须为`0`时，灯才会亮起。由于这个条件，我们通过将`Irritation_flag = 1`使条件的一部分变为假；因此，开灯的程序将不起作用。

现在，让我们回顾一下我们到目前为止所做的事情：

+   当时间是 07:00-07:15 时，灯将被打开

+   如果检测到有人移动，灯将被关闭

+   另一个条件将再持续五分钟，等待红外距离传感器检测到人体运动

+   如果一个人在五分钟内通过，那么警报将被停用，否则警报将再次开始打开灯

挺酷的，是吧？ 话虽如此，让我们从之前的程序中再添加另一个功能：

```py
  for H = '07'and M > '15' and Irritation_Flag = 0 and GPIO.input(PIR) = 0: GPIO.output(LIGHT,GPIO.HIGH)
    time.sleep(5) GPIO.output(LIGHT,GPIO.LOW)
    time.sleep(5)
```

你知道这是做什么的。如果在第一个`15`分钟内你不活动，也就是从 07:00 到 07:15，那么它将开始每五秒闪烁灯，迫使你醒来：

```py
 if H != '07':

            Irritation_flag = 0 
```

最后，我们使用条件`if H != '07':`。所以，每当`H`的值不是`07`时，条件就会成立，这将把`Irritation_flag`重置为`0`。到现在为止，你知道将`Irritation_flag`设置为`0`的作用。

# 总结

所以，最后，我们做出了我们的第一个迷你贾维斯，它可以在早上叫醒你，甚至在你没有按时醒来时还会惹你生气。希望你通过学习两个运动传感器及其在自动化电器中的应用来真正享受了这一章节。所以，继续在家里尝试一下，根据自己的需求修改代码，制作一些真正酷炫的东西。接下来，我们将让我们的贾维斯做一些更酷炫的事情，并且我们将介绍一些更令人兴奋的有关人体检测的东西。
