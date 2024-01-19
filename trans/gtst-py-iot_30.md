# 制作机械臂

最后，我们终于到达了大多数人自本书开始以来就想要到达的地方。制作一个机械臂！在本章中，我们将学习机械臂工作背后的概念。毫无疑问，我们也将制作一个用于个人使用的机械臂，它可以为我们做无限的事情。

# 机械臂的基础

如果你看到一个人体，那么使我们能够与大多数其他物种不同的最显著的部分之一就是手臂。这是我们用来做大部分工作的身体部分。

人类的手臂是一个由关节和肌肉组成的非常复杂的机制，它们协同工作以赋予它我们所知道的灵巧性。以我们的肩关节为例。如果你注意观察，你会注意到它有能力向上、向下、向右、向左移动，甚至可以在自己的轴线上旋转，而这一切只是因为它只有一个单一的关节，我们称之为球关节。

当我们谈论机器人上的机械臂时，我们无疑是在谈论一种复杂的安排，它由执行器和身体（也称为底盘）组成，以在三维空间中获得所需的运动。

现在，让我们了解一些机械臂的基本部件。第一部分是执行器。我们可以使用电机来控制机械臂；然而，正如我们之前学过的，使用之前使用过的电机不是它的理想解决方案，因为它既不能保持位置，也没有反馈机制。因此，我们只剩下一个选择，那就是使用舵机。正如我们所知，它们有大量的扭矩，并且有能力知道它在哪里，并且可以保持其位置，只要我们想要。

机器人的第二部分是底盘，也就是将所有电机固定在一起并为机器人提供结构支持的部分。这必须以这样的方式制作，以便为任何给定的关节提供所有理想轴线的运动。这很重要，因为单个舵机只能在一个单一轴线上提供运动。然而，有多个地方可以使用复杂的安排使机器人在多个轴线上移动。此外，底盘应该是坚固的，这非常重要。正如我们所知，地球上的所有材料都具有一定程度的柔韧性。材料的构造也取决于材料的不服从性。这对于重复性非常重要。

现在，什么是重复性？你可能在工业或任何制造单位中看到，机器人被安装并一遍又一遍地执行相同的任务。这是可能的，因为机器人被编程执行一组特定的功能在特定的情况下。现在，假设机器人的底盘不是坚固的。在这种情况下，即使舵机是100%精确并且一遍又一遍地到达完全相同的位置，机器人仍然可能与其实际目标位置不同。这是因为底盘可能有一定的柔韧性，这就是为什么最终位置可能会有所不同。因此，正确的底盘是必不可少的。当我们谈论大型机器人时，这变得更加重要，因为即使最轻微的变形也可能导致机械臂最终位置的非常大的变化。

我们在谈论机器人手臂时经常使用的一个术语是末端执行器。这基本上是机器人手臂的末端，它将为我们做所有最终的工作。在真实的人类手臂的情况下，末端执行器可以被认为是手。这位于手臂的顶部，手臂的所有运动基本上是为了在三维空间中表达手的位置。此外，正是手拿起物体或进行必要的物理动作。因此，术语末端执行器。

现在，由于机械臂在三维空间中移动，定义运动发生的轴成为一个真正的大问题。因此，我们通常使用正在执行的运动类型来定义运动，这给我们一个关于运动是什么以及在哪个轴上的现实想法。为了分析运动，我们使用**偏航、俯仰和翻滚**（**YPR**）的概念。

![](Images/3627647f-ae6b-419e-a164-3f5eaa509cf8.png)

前面的图表将清除关于YPR的大部分疑惑。这个概念通常用于飞机；然而，它也是机械手的一个重要部分。因此，正如您可以从前面的图表中看到的，当飞机的机头上下移动时，它将被视为俯仰运动。同样，如果飞机改变航向，那么**偏航**也可以相应地改变——**偏航**只是飞机在*y*轴上的运动。最后，我们有一个叫做**翻滚**的东西。它用于理解旋转的角度。正如您所看到的，所有这三个实体是彼此独立的，追逐其中任何一个都不会对其他产生影响。这个概念也很有用，因为无论飞机的方向如何，YPR仍然保持不变，非常容易理解。因此，我们直接从飞机上将这个概念引入到我们的机器人中。

最后，我们怎么能忘记处理单元呢？它是命令所有执行器并进行协调和决策的单元。在我们的情况下，这个处理单元是树莓派，它将命令所有执行器。所有这些前述的组件构成了一个机械臂。

# 自由度

并非每个机械臂都相同。它们具有不同的负载评级，即末端执行器可以承受的最大负载，速度和范围，即末端执行器可以达到的距离。然而，机械臂非常重要的一部分是它所拥有的电机数量。因此，对于每个轴，您至少需要一个电机来使机器人在该轴上移动。例如，人类手臂在肩关节具有三维自由度。因此，为了模仿该关节，您将需要每个轴的电机，也就是说，至少需要三个电机才能使手臂在所有三个轴上独立移动。同样，当我们谈论手肘关节时，它只能在两个维度上移动。也就是手臂的张合和最终手臂的旋转，手肘不在第三维度上移动。因此，为了复制它的运动，我们至少需要两个电机，这样我们就可以在*w*轴上移动机器人。

根据我们目前所了解的，我们可以安全地假设电机数量越多，机器人的灵巧性也越高。这在大多数情况下是成立的；然而，您可能使用多个电机使机器人在单个轴上旋转。在这种情况下，通过计算执行器数量来确定机器人的灵巧性的基本概念将不起作用。那么我们如何确定机器人的灵巧性呢？

我们有一个叫做**自由度**（**DOF**）的概念。如果按照标准定义，我可以非常肯定地说你会对它的实际含义感到困惑。如果你不相信，那就自己去Google上找找看。用非常简单和平实的英语来说，自由度是指关节可以在任何给定的轴上独立移动。所以，例如，如果我们谈论肩关节，那么我们在所有三个轴上都有运动。因此，自由度就是三。现在，让我们考虑一下我们手臂的肘关节。因为它只能在俯仰和滚动中移动，所以我们最终得到两个自由度。如果我们把肩关节和肘关节连接起来，那么自由度就会增加，整个系统将被称为具有六个自由度。请记住，这个定义是非常简化的。如果你选择深入挖掘，你会遇到多种复杂性。

现在，你会遇到的大多数机械臂都有接近六个自由度。虽然你可能会说这比人类的手臂少，但实际上，它完成了大部分工作，显然自由度较少意味着更少的电机数量，从而降低成本，显然编程复杂性也更低。因此，我们尽量使用尽可能少的自由度。

![](Images/a2bb6d91-ba4e-46da-ab66-3d9495652574.png)

现在，在前面的图表中，你可以看到一个典型的机械臂，它有六个自由度。编号为**1**的基本执行器提供了滚动和改变俯仰的自由度。编号为**2**的肘部执行器只为机器人增加了一个俯仰的自由度。此外，第**3**关节有能力在**俯仰和滚动**中移动。最后，我们在这里有末端执行器作为夹具；夹具本身有一个自由度。因此，总体上，我们可以说这个机器人是一个六自由度的机器人。

# 动力源

在我们所有的项目中，我们一直在使用一个单位，但我想在这一章中强调一下。这个单位是功率单位。我们谈论它的原因是因为在这一章中我们将控制多个舵机。当我们谈论多个舵机时，自然我们将谈论大量的功耗。在机械臂中，我们有六个舵机电机。现在，根据电机的品牌和型号，功耗会有所不同。但是为了保险起见，假设每个舵机的功耗约为1安培是个好主意。你可能使用的大多数电源可能无法提供这么多的突发电流。那么我们该怎么办呢？

我们可以采取更高功率输出的简单方法。但是，相反，我们可以采取非常规的途径。我们可以有一个电池，在需要时可以提供这么多的功率。但问题是，任何电池都能满足我们的目的吗？显然，答案是否定的。

存在多种类型的电池。这些电池可以根据以下参数进行区分：

+   电压

+   容量

+   功率重量比

+   最大充电和放电速率

+   化学成分

这些将在接下来的小节中详细介绍。

# 电压

电压是电池可以产生的总体电位差。每个电池都有它提供的特定电压。要记住的一件事是，这个电压会根据电池的充电状态略有变化。也就是说，当一个12V的电池充满电时，它可能会输出12.6V。然而，当它完全放电时，可能会降到11.4V。因此，电池电压的意思是电池将提供的名义电压。

# 容量

现在，第二个参数是容量。通常，当你购买电池时，你会看到它的容量以毫安时（mAh）或安时（Ah）为单位。这是一个非常简单的术语。让我用一个例子来解释这个术语给你。假设你有一个容量为5Ah的电池。现在，如果我连续绘制5安培1小时，那么电池将完全放电。相反，如果我连续绘制10安培，那么电池将在半小时内放电。通过这个，我们还可以使用以下简单的公式推导出电池的总功率：*电池的总功率=电池的标称电压x电池的总容量*

因此，如果你有一个12V的电池，其容量为10Ah，那么总容量将是120瓦特。

# 功率重量比

重量在机器人技术中扮演着非常关键的角色，如果我们增加机器人的重量，那么移动它所需的力量可能会呈指数级增长。因此，功率重量比的概念就出现了。我们总是更喜欢一个极轻的电池，它在重量方面提供了大量的功率。功率重量比的方程可以定义如下：*每公斤瓦时的功率重量比=瓦特的最大功率/电池的总重量*

现在，假设一个电池提供了500瓦的功率，重量为5公斤，那么功率重量比将是100瓦时/公斤。功率重量比越高，电池就越好。

# 最大充电和放电速率

这可能是电池中最关键的部分之一。通常，电池能够让机器人运行1小时。然而，机器人的功耗并不是恒定的。假设在90%的时间里，我们的机械臂消耗2安培的功率，所以电池容量为2Ah。然而，在操作过程中的某些时刻，机器人需要所有电机以最大功率工作。机器人的峰值功耗约为6安培。现在，问题是，2Ah的电池能否为机器人提供6安培的功率？

这是一个非常实际的挑战。你可能会说，最好选择一个比2Ah电池大得多的电池。但是，正如你所知，这将显著增加重量。那么解决方案是什么呢？

还有一个叫做峰值放电电流的东西。这由*C*评级表示。因此，如果我们的电池是1C评级，那么2Ah的电池一次只能提供最多2Ah的电源。然而，如果电池是10C评级，那么它应该能够提供高达20安培的突发电源。如今，你可以找到可以提供高达100C甚至更高的突发电源的电池。我们之所以有这个是因为机器人的峰值功耗可能比它们的恒定功耗高得多。如果在任何时候，电池无法提供足够的电力，那么机器人将表现异常，甚至可能关闭。

这个故事的第二部分是充电评级。这是你可以提供给电池的最大充电电流。它也由相同的C评级表示。因此，如果C评级为0.5，那么你可以为2Ah的电池提供最大1安培的充电。 

换句话说，你可以给电池充电的最快速度是2小时。

# 化学成分

市场上有不同类型的电池，它们通常根据其化学成分进行广泛分类。所有这些电池都有各自的优缺点。因此，我们不能说哪一个比另一个更好。这总是在各种因素之间进行权衡。以下是市场上可以找到的电池列表，以及它们的优缺点：

| **电池** | **峰值功率输出** | **功率重量比** | **价格** |
| --- | --- | --- | --- |
| 湿电池 | 低 | 极低 | 最便宜 |
| 镍氢电池 | 中等 | 低 | 便宜 |
| 锂离子 | 高 | 好 | 高 |
| 锂聚合物 | 极高 | 极好 | 极高 |

从这个表中可以看出，峰值功率输出是我们非常想要的，良好的功率重量比也是如此；因此，在锂聚合物电池上花费一定的金额是有道理的。

这些电池，至少具有20C的额定值，功率重量比约为普通湿电池的五倍。然而，它们的价格可能是普通湿电池的10倍。

现在我们知道了为这些更高电流要求选择哪些电池。一块11.1V和2200毫安时的锂聚合物电池不会花费你超过20美元，并且将为你提供你可能永远不需要的巨大功率。所以，我们已经解决了电源供应问题。现在是时候继续使机械手运行了。

# 寻找极限

机械臂套件在eBay或亚马逊上相对容易获得。这并不难组装，需要几个小时来准备。一些机械臂套件可能不会随舵机一起发货，如果是这样，你可能需要单独订购。我建议选择与舵机捆绑在一起的套件，因为如果你选择单独订购舵机，可能会出现兼容性问题。

正如你所知，这些舵机将使用PWM工作，控制它们也不难。所以，让我们直接开始并看看我们能做些什么。一旦你组装好了机械臂套件，将舵机的线连接如下：

![](Images/3a2a8a1f-c77f-4279-b01b-093d65b5da18.png)

现在，首先，我们需要知道我们机器人上连接的每个舵机的最大物理极限是什么。有各种各样的技术可以做到这一点。最基本的方法是进行物理测量。这种方法可能很好，但你将无法充分利用舵机电机的全部潜力，因为在测量时会有一定程度的误差。因此，你放入舵机的值将略小于你认为它可以达到的值。第二种方法是手动输入数据并找出确切的角度。所以，让我们继续用第二种方法做事情，并上传以下代码：

```py
import RPi.GPIO as GPIO import time GPIO.setmode(GPIO.BCM) GPIO.setup(14,GPIO.OUT) GPIO.setup(16,GPIO.OUT) GPIO.setup(18,GPIO.OUT) GPIO.setup(20,GPIO.OUT) GPIO.setup(21,GPIO.OUT) GPIO.setup(22,GPIO.OUT)
  GPIO.setwarnings(False) pwm1 = GPIO.PWM(14, 50) pwm2 = GPIO.PWM(16, 50) pwm3 = GPIO.PWM(18, 50) pwm4 = GPIO.PWM(20, 50) pwm5 = GPIO.PWM(21, 50) pwm6 = GPIO.PWM(22, 50)
  pwm1.start(0) pwm2.start(0) pwm3.start(0) pwm4.start(0) pwm5.start(0) pwm6.start(0)  def cvt_angle(angle):     dc = float(angle/90) + 0.5
    return dc  while 1:

 j = input('select servo')  if j == 1: i = input('select value to rotate')
  pwm1.ChangeDutyCycle(cvt_angle(i))
 time.sleep(2)
  pwm1.ChangeDutyCycle(cvt_angle(90)) elif j ==2:  i = input('select value to rotate')   pwm2.ChangeDutyCycle(cvt_angle(i))
 time.sleep(2) pwm2.ChangeDutyCycle(cvt_angle(90))   elif j ==3:   i = input('select value to rotate') pwm3.ChangeDutyCycle(cvt_angle(i))
 time.sleep(2) pwm3.ChangeDutyCycle(cvt_angle(90))  elif j ==4:  i = input('select value to rotate')
  pwm4.ChangeDutyCycle(cvt_angle(i))
 time.sleep(2) pwm4.ChangeDutyCycle(cvt_angle(90))  elif j ==5:  i = input('select value to rotate')
  pwm5.ChangeDutyCycle(cvt_angle(i))
 time.sleep(2) pwm5.ChangeDutyCycle(cvt_angle(90))  elif j ==6:  i = input('select value to rotate')   pwm6.ChangeDutyCycle(cvt_angle(i))
 time.sleep(2) pwm6.ChangeDutyCycle(cvt_angle(90)) }
```

现在，让我们看看这段代码在做什么。这段代码看起来可能相当复杂，但它所做的事情非常简单。

```py
 j = input('select servo from 1-6')
```

使用前面的代码行，我们正在为用户打印`从1-6选择舵机`的语句。当用户输入舵机的值时，这个值被存储在一个名为`j`的变量中：

```py
 if j == 1: i = input('select value to rotate')
  pwm1.ChangeDutyCycle(cvt_angle(i))
 time.sleep(2) pwm1.ChangeDutyCycle(cvt_angle(90))  
```

这里的`if`条件检查`j`的值。如果在这一行中，`j=1`，那么它将运行与舵机编号`1`对应的代码。在这段代码中，第一行将打印`选择要旋转的值`。完成后，程序将等待用户输入。一旦用户输入任何值，它将被存储在一个名为`I`的变量中。然后，使用`cvt_angle(i)`函数，用户输入的值将被转换为相应的占空比值。这个占空比值将被获取到`pwm1.ChangeDutyCycle()`参数中，从而给予机器人你想要的特定关节角度。由于`time.sleep(2)`函数，舵机将等待到下一行。之后，我们使用`pwm1.ChangeDutyCycle(cvt_angle(90))`这一行，这将把它带回到90度。

你可能会问，为什么我们要这样做？这是一个非常重要的原因。假设您已经给它一个超出其物理极限的命令。如果是这种情况，那么舵机将继续尝试朝那个方向移动，不管发生什么。然而，由于物理限制，它将无法继续前进。一旦发生这种情况，然后在几秒钟内，您将看到蓝烟从舵机中冒出，表明它的损坏。问题在于，制造这种类型的错误非常容易，损失是非常明显的。因此，为了防止这种情况，我们迅速将其带回到中心位置，这样它就不会有任何烧毁的可能性。

现在，根据前面的代码，通过机器人对舵机1-6执行相同的操作。现在你知道发生了什么，是时候拿起笔和纸开始给舵机赋予角度值了。请记住，这段代码的最终目标是找出最大限制。因此，让我们从90度开始做起。在每一侧给它一个值，直到你能够接受的值。在纸上列出清单，因为我们将需要它用于下一段代码。

# 使机器人安全

在本章的前一部分中，通过我们的多次尝试，我们已经能够找到每个舵机的最大位置。现在是时候使用这些值了。在本章中，我们将为舵机编程其绝对最大值。在这个程序中，我们将确保舵机永远不需要超出两侧的定义参数。如果用户给出超出它的值，那么它将选择忽略用户输入，而不是造成自身损坏。

那么，让我们看看如何完成它。在程序的某些部分，数字值已经用粗体标出。这些是您需要用本章前面记录的值替换的值。例如，对于舵机1，记录下的值是`23`和`170`，作为每一侧的最大值。因此，代码的更改将从`if a[0] < 160 and a[0] > 30`变为`ifa[0] < 170 and a[0] > 23`。同样，对于每个舵机，必须遵循相同的程序：

```py
import RPi.GPIO as GPIO import time GPIO.setmode(GPIO.BCM) GPIO.setup(14,GPIO.OUT) GPIO.setup(16,GPIO.OUT) GPIO.setup(18,GPIO.OUT) GPIO.setup(20,GPIO.OUT) GPIO.setup(21,GPIO.OUT) GPIO.setup(22,GPIO.OUT)
  GPIO.setwarnings(False) pwm1 = GPIO.PWM(14, 50) pwm2 = GPIO.PWM(16, 50) pwm3 = GPIO.PWM(18, 50) pwm4 = GPIO.PWM(20, 50) pwm5 = GPIO.PWM(21, 50) pwm6 = GPIO.PWM(22, 50)
  pwm1.start(cvt_angle(90)) pwm2.start(cvt_angle(90)) pwm3.start(cvt_angle(90)) pwm4.start(cvt_angle(90)) pwm5.start(cvt_angle(90)) pwm6.start(cvt_angle(90))

def cvt_angle(angle):
    dc = float(angle/90) + 0.5
    return dc

while True:

    a = raw_input("enter a list of 6 values")

    if a[0] < 160 and  a[0] > 30:
        pwm1.ChangeDutyCycle(cvt_angle(a[0]))

    if a[1] < 160 and  a[1] > 30:  pwm2.ChangeDutyCycle(cvt)angle(a[1]))

    if a[0] < 160 and  a[0] > 30: pwm3.ChangeDutyCycle(cvt_angle(a[2]))    if a[0] < 160 and  a[0] > 30: pwm4.ChangeDutyCycle(cvt_angle(a[3]))    if a[0] < 160 and  a[0] > 30: pwm5.ChangeDutyCycle(cvt_angle(a[4]))    if a[0] < 160 and  a[0] > 30: pwm6.ChangeDutyCycle(cvt_angle(a[5]))}
```

现在，在这段代码中，我们做了一些非常基础的事情。您可以放心地说，我们所做的一切就是将`ChangeDutyCycle()`函数放在一个`if`语句中。这个`if`语句将决定舵机是移动还是保持在原位。对一些人来说，将这个程序放在一个特殊的部分似乎很天真。但是，请相信我，不是这样的。这个语句现在将作为以后每个程序的一部分。为了检查通过这个`if`语句传递给舵机的最终值，必须检查为舵机移动编写的所有代码；因此，对代码的基本可视化是非常必要的。

现在解释完毕，是时候给出不同的命令并查看它们是否在安全工作限制内工作了。

# 编写多个帧

在上一章中，我们已经学习了如何确保机器人在安全限制下工作的基础知识。在本章中，我们将看看如何使机器人能够在点击按钮的同时执行不同的活动，而不是逐个输入值。

为了做到这一点，我们需要了解一些高级运动概念。每当您观看任何视频或玩任何视频游戏时，您一定会遇到“每秒帧数”（FPS）这个术语。如果您还没有听说过这个术语，那么让我为您解释一下。现在制作的每个视频实际上都是由静止图像制成的。这些静止图像是由摄像机捕捉的，每秒点击25-30次。当这些图像以与它们被捕捉的速率相同的速率在屏幕上播放时，它形成了一个平滑的视频。

同样，在机器人中，我们也有帧的概念。然而，这些帧不是图像，而是机器人必须遵循的多个步骤。在一个简单的机器人程序中，可能只有两个帧，即初始帧和最终帧。这两个帧将对应于初始位置或最终位置。

然而，在现实世界中，这并不总是可能的，因为当机器人直接从初始位置到最终位置时，它会沿着特定的路径运动，并具有特定的曲率。然而，在这条路径上可能会有障碍物，或者这条路径可能不是所需的，因为需要遵循的路径可能是另一条。因此，我们需要帧。这些帧不仅定义了机器人从初始位置到最终位置的运动，而且将这两个位置之间的过渡分解为多个步骤，使机器人遵循所需的路径。

这可以被称为帧编程，在本章中我们将介绍。要记住的一件事是，帧数越多，机器人的运行就越平稳。你还记得我们看到的闭路电视录像吗？我们可以说它不够平滑，而且有很多抖动。这是由于闭路电视摄像头的低帧率造成的。它们不是以30FPS工作，而是以15FPS工作。这是为了减少视频的存储空间。然而，如果你看到最新的视频，有一些游戏和视频的帧率比正常的要高得多。我们最新的摄像头有60FPS的工作，使视频更加平滑和愉快。机器人也是如此。帧数越多，运动就越平滑和可控。但是，请确保不要过度使用。

现在，要从一个位置移动到另一个位置，我们将不得不在一开始就放入每个舵机的角度值。一旦获取，它将自动开始逐个执行这些值。为此，请继续编写以下代码：

```py
import RPi.GPIO as GPIO import time GPIO.setmode(GPIO.BCM) GPIO.setup(14,GPIO.OUT) GPIO.setup(16,GPIO.OUT) GPIO.setup(18,GPIO.OUT) GPIO.setup(20,GPIO.OUT) GPIO.setup(21,GPIO.OUT) GPIO.setup(22,GPIO.OUT)
  GPIO.setwarnings(False) pwm1 = GPIO.PWM(14, 50) pwm2 = GPIO.PWM(16, 50) pwm3 = GPIO.PWM(18, 50) pwm4 = GPIO.PWM(20, 50) pwm5 = GPIO.PWM(21, 50) pwm6 = GPIO.PWM(22, 50)
  pwm1.start(0) pwm2.start(0) pwm3.start(0) pwm4.start(0) pwm5.start(0) pwm6.start(0)

def cvt_angle(angle):
    dc = float(angle/90) + 0.5
    return dc

prev0 = 90
prev1 = 90
prev2 = 90
prev3 = 90
prev4 = 90
prev5 = 90 

while True:

    a = raw_input("enter a list of 6 values for motor 1")
    b = raw_input("enter a list of 6 values for motor 2")
    c = raw_input("enter a list of 6 values for motor 3")
    d = raw_input("enter a list of 6 values for motor 4")
    e = raw_input("enter a list of 6 values for motor 5")
    f = raw_input("enter a list of 6 values for motor 6")

    for i in range(6):

        if a[i] > 10 and a[i]< 180 :  
            pwm1.ChangeDutyCycle(cvt_angle(a[i]))

        if b[i] > 10 and b[i] < 180:
  pwm2.ChangeDutyCycle(cvt_angle(b[i]))

        if c[i] > 10 and c[i] < 180:
 pwm3.ChangeDutyCycle(cvt_angle(c[i]))

        if d[i] > 10 and d[i] < 180:
 pwm4.ChangeDutyCycle(cvt_angle(d[i]))

        if e[i] > 10 and e[i] < 180:
 pwm5.ChangeDutyCycle(cvt_angle(e[i]))

        if f[i] > 10 and f[i] < 180:
 pwm6.ChangeDutyCycle(cvt_angle(f[i])) 
```

在这个程序中，你可以看到我们复制了以前的程序并进行了一些非常小的改动。所以，让我们看看这些改动是什么：

```py
    a = raw_input("enter a list of 6 values for motor 1")
    b = raw_input("enter a list of 6 values for motor 2")
    c = raw_input("enter a list of 6 values for motor 3")
    d = raw_input("enter a list of 6 values for motor 4")
    e = raw_input("enter a list of 6 values for motor 5")
    f = raw_input("enter a list of 6 values for motor 6")
```

在这里，我们正在为每个舵机获取输入值并将其存储在不同的列表中。对于舵机1，将使用列表`a`；类似地，对于舵机2，将使用`b`，依此类推直到`f`。在代码的前面几行中，机器人将提示用户填写`电机1`的六个帧值。然后，它将要求`电机2`的六个值，以此类推直到`电机6`。

```py
    for i in range(6):
```

给舵机提供PWM的整个程序都集中在这个for循环中。这个循环将检查`i`的值并每次递增。`i`的值将从`1`开始，循环将运行并递增`i`的值直到达到`6`。

```py
        if a[i] > 10 and a[i]< 180 :  
            pwm1.ChangeDutyCycle(cvt_angle(a[i]))
```

在程序的这一行中，列表中包含的值是基于`1`的值进行排序的。因此，第一次它将读取`a[1]`的值，这将对应于列表`a[]`的第一个值。这个值应该在安全工作范围内，因此使用`if`循环。如果在安全工作范围内，那么`if`条件中的程序将执行，否则不会执行。在`if`循环内，我们有一个简单的语句：`pwm1.ChangeDutyCycle(cvt_angle(a[I]))`。这将简单地取`a[1]`的值并将其转换为相应的PWM值，并将其提取到`ChangeDutyCycle()`函数中，这将改变舵机1的PWM。

对于其余的舵机也制作了类似的程序，从舵机1到舵机6。因此，所有这些都将逐一读取其对应列表中的值，并根据用户编程的方式改变舵机的角度。此外，随着循环的执行，`i`的值将增加，从而使程序读取列表中提取的不同值。列表中舵机的每个值将对应一个不同的帧，从而使机器人通过它。

所以继续玩一些有趣的东西，让你的机器人做一些很棒的动作。只要小心对待它！

# 速度控制

能够如此轻松地制作一个机械臂真是太神奇了，只需一点点代码，我们现在就能够按照自己的意愿来控制它。然而，你可能已经注意到了一个问题，那就是，机器人按照我们的意愿移动，但速度不是我们想要的。这是在使用基于数字PWM的舵机时非常常见的问题。

这些舵机没有内置的速度控制。它们的控制系统被编程为尽可能快地移动舵机以达到目标位置。因此，要控制速度，我们必须对程序本身进行调整，并给它一个平稳的线性进展。

速度控制可以通过几种不同的技术来实现。所以，不多说了，让我们去看看代码。在你编写代码之前，先读一遍，然后看一下下面的解释。之后，你会更清楚我们在做什么。这将使编写代码更快、更容易。所以，让我们来看看：

```py
import RPi.GPIO as GPIO import time GPIO.setmode(GPIO.BCM) GPIO.setup(14,GPIO.OUT) GPIO.setup(16,GPIO.OUT) GPIO.setup(18,GPIO.OUT) GPIO.setup(20,GPIO.OUT) GPIO.setup(21,GPIO.OUT) GPIO.setup(22,GPIO.OUT)
  GPIO.setwarnings(False) pwm1 = GPIO.PWM(14, 50) pwm2 = GPIO.PWM(16, 50) pwm3 = GPIO.PWM(18, 50) pwm4 = GPIO.PWM(20, 50) pwm5 = GPIO.PWM(21, 50) pwm6 = GPIO.PWM(22, 50)
  pwm1.start(0) pwm2.start(0) pwm3.start(0) pwm4.start(0) pwm5.start(0) pwm6.start(0)

def cvt_angle(angle):
    dc = float(angle/90) + 0.5
    return dc

prev0 = 90
prev1 = 90
prev2 = 90
prev3 = 90
prev4 = 90
prev5 = 90 

pwm1.ChangeDutyCycle(cvt_angle(prev0)) pwm2.ChangeDutyCycle(cvt_angle(prev1)) pwm3.ChangeDutyCycle(cvt_angle(prev2)) pwm4.ChangeDutyCycle(cvt_angle(prev3)) pwm5.ChangeDutyCycle(cvt_angle(prev4)) pwm6.ChangeDutyCycle(cvt_angle(prev5)) 

while True:

 a = raw_input("enter a list of 6 values for motor 1")
 b = raw_input("enter a list of 6 values for motor 2")
 c = raw_input("enter a list of 6 values for motor 3")
 d = raw_input("enter a list of 6 values for motor 4")
 e = raw_input("enter a list of 6 values for motor 5")
 f = raw_input("enter a list of 6 values for motor 6")

    speed = raw_input("enter one of the following speed 0.1, 0.2, 0.5, 1")

 for i in range(6):

   while prev0 =! a[i] and prev1 =! b[i] and prev2 =! c[i] and prev3 =! d[i] and prev4 =! e[i] and prev 5 =! f[i]

     if a[i] > 10 and a[i]< 180 : 

        if prev0 > a[i]
            prev0 = prev0 - speed

         if prev0 < a[i]
             prev0 = prev0 + speed

         if prev0 = a[i]
             prev0 = prev0 

         pwm1.ChangeDutyCycle(cvt_angle(prev0))

    if b[i] > 10 and b[i] < 180:

        if prev2 > b[i]
            prev2 = prev2 - speed

         if prev2 < b[i]
             prev2 = prev2 + speed

         if prev2 = b[i]
            prev2 = prev2

  pwm2.ChangeDutyCycle(cvt_angle(b[i]))

    if c[i] > 10 and c[i] < 180: if prev3 > c[i]
             prev3 = prev3 - speed

        if prev3 < c[i]
            prev3 = prev3 + speed

        if prev3 = c[i]
             prev3 = prev3

 pwm3.ChangeDutyCycle(cvt_angle(c[i]))

    if d[i] > 10 and d[i] < 180: if prev4 > d[i]
             prev4 = prev4 - speed

        if prev4 < d[i]
            prev4 = prev4 + speed

        if prev4 = d[i]
             prev4 = prev4

 pwm4.ChangeDutyCycle(cvt_angle(d[i]))

     if e[i] > 10 and e[i] < 180: if prev5 > e[i]
             prev5 = prev5 - speed

        if prev0 < e[i]
            prev5 = prev5 + speed

        if prev5 = e[i]
             prev5 = prev5

 pwm5.ChangeDutyCycle(cvt_angle(e[i]))

     if f[i] > 10 and f[i] < 180: if prev6 > f[i]
            prev6 = prev6 - speed

         if prev6 < f[i]
            prev6 = prev6 + speed

        if prev6 = f[i]
            prev6 = prev6

 pwm6.ChangeDutyCycle(cvt_angle(f[i]))

 flag = 0 
```

在这个程序中，有很多东西。我们应该逐一了解它们。所以，让我们看看我们在做什么：

```py
prev0 = 90
prev1 = 90
prev2 = 90
prev3 = 90
prev4 = 90
prev5 = 90 
```

在这里，我们定义了六个新变量，名称为`prev0`到`prev5`，它们都被赋予了值`90`。这里的术语`prev`代表之前的值，因此它将指示先前的值。

```py
        while prev0 =! a[i] and prev1 =! b[i] and prev2 =! c[i] and prev3 =! d[i]   and prev4 =! e[i] and prev 5 =! f[i]
```

在代码行`for i in range 6`之后，我们有前面的代码行，基本上是检查`a[i]`的值与`prev0`的值。类似地，它正在检查`b[i]`的值与`prev1`的值，依此类推。直到所有这些条件都成立，`while`循环将为真，并在其中循环程序，直到条件不再为假。也就是说，所有的`prev`值恰好等于列表相应值的值。

再次，这对你可能有点奇怪，但相信我，它会非常有用，我们一会儿会看到：

```py
     if a[i] > 10 and a[i]< 180 : 

         if prev0 > a[i]
             prev0 = prev0 - speed

         if prev0 < a[i]
             prev0 = prev0 + speed

         if prev0 = a[i]
             prev0 = prev0 

         pwm1.ChangeDutyCycle(cvt_angle(prev0))
```

现在，真正的问题来了。这是将控制舵机速度的主程序。在这个程序中，第一行很简单；它将检查给定的值是否有效，也就是在安全极限之间。一旦完成，它将检查`a[Ii]`的值是否小于或大于先前的值。如果大于`a[i]`的值，那么它将采用先前的值，并用用户指定的速度递减。如果小于`a[i]`的值，那么它将用指定的速度递增先前的值。

因此，如果你看一下，代码只是在`while`循环运行时每次递增或递减先前的值。现在，`while`循环将一直运行，直到`prev`的值等于相应列表值。也就是说，循环将一直递增值，直到达到指定位置。

因此，速度值越低，每次递增的值就越低，从而整体减慢速度。

这个过程对所有其他舵机也是一样的。听起来可能很复杂，但实际上并不是！编程很容易，每次你把它分解成小块并逐一理解时，它都会继续保持简单！

# 总结

在本章中，我们已经了解了机械臂的基础知识、其电源和其编程。通过一个非常简单的程序，我们能够找出舵机的极限，然后应用这些极限以确保舵机不会损坏自己。我们对框架有了一个基本的概念，并根据框架进行了一些编程。最后，我们还继续控制了舵机的速度，使用了我们自己的基本级别的程序。
