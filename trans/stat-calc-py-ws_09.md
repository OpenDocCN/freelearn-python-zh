# 9\. 使用 Python 进行中级统计

概述

在本章中，我们将进一步学习一些中级统计概念。我们将了解大数定律告诉我们随着样本的增大，样本均值的价值是什么。

通过本章的学习，你将能够应用中心极限定理来描述样本均值的分布，创建置信区间来描述平均值的可能取值并带有一定的置信度，使用假设检验来评估基于样本提供的证据的结论，并使用回归方程来分析数据。

# 介绍

在之前的章节中，我们已经使用描述性统计和可视化技术描述和探索了数据。我们还研究了概率、随机性以及使用随机变量的模拟来解决问题。分布的概念也被讨论过，这在本章后面将扮演更重要的角色。

在应用统计思想时，有一些重要的问题需要回答，涉及方法论。这些问题的例子可能包括“我应该让我的样本有多大？”或者“我们对结果有多有信心？”。在本章中，我们将看看我们如何应用统计学中最重要的两个定理，首先从它们的实际影响开始，然后再转向使用这些重要思想衍生出的更有用的技术来解决常见问题。

在本章中，我们将解释大数定律是什么，并澄清样本大小如何影响样本均值。我们将讨论中心极限定理，以及它在置信区间和假设检验中的应用。使用 Python，我们将构建函数来计算置信区间，描述样本统计和民意调查中的误差范围。我们将在 Python 中进行假设检验，评估收集样本的证据与一组相互矛盾的假设。最后，利用 Python 的线性回归能力，我们将创建一个线性模型来预测新的数据值。

# 大数定律

有很多人声称有很多方案和系统可以让你在赌场大赢家。但是这些人忽视了赌场为什么能够赚钱的原因；赔率总是对赌场有利，确保赌场最终总是赢得胜利（从长远来看）。赌场所依赖的是一种叫做大数定律的东西。

在我们弄清楚赌场为什么总是在长期内让自己成为赢家之前，我们需要定义几个术语。第一个是**样本平均值**，或者**样本均值**。当人们想到平均值时，他们通常会想到样本均值。你可以通过将结果相加然后除以结果的数量来计算样本均值。比如说我们抛硬币 10 次，有 7 次是正面。我们可以计算样本均值，或者每次抛硬币得到正面的平均次数，如下所示：

![图 9.1：样本均值公式](img/B15968_09_01.jpg)

图 9.1：样本均值公式

样本均值通常用*x̄*表示，读作*x bar*。

我们需要理解的第二个术语是**期望值**。期望值是基于概率我们可以期望的理论值。对于离散的例子，比如我们的抛硬币实验，我们通过将每个结果乘以其发生的概率来计算它。对于我们的抛硬币例子，我们将硬币的每一面的正反面数，正面为 1，反面为 0，乘以每一面发生的概率，这种情况下每一面的概率都是 0.5。数学上写出来就是：

![图 9.2：期望值公式](img/B15968_09_02.jpg)

图 9.2：期望值公式

我们可以期望每次抛硬币得到 0.5 个正面，这是有道理的，因为在任何一次抛硬币中我们有 50%的机会得到正面。

另一个术语是**样本**，它是一组结果。在这种情况下，抛硬币结果的集合就是我们的样本。样本的一个重要特征是其大小，或者你拥有的结果数量。我们有 10 次抛硬币，所以我们的样本大小是 10。最后一个术语是**独立性**的概念，即一个结果绝对不会影响另一个结果。我们的抛硬币是独立的；在第一次抛硬币得到正面不会以任何方式影响第 10 次抛硬币的结果。

注意我们的样本平均值和期望值不同。虽然在 10 次抛硬币的样本中得到 7 次正面似乎不太可能，但这并不是不可能的结果。然而，我们知道大约一半的样本应该是正面。如果我们继续抛 10 次硬币会发生什么？甚至再抛 100 次或 1,000 次呢？这个问题的答案由大数定律提供。**大数定律**规定，随着样本大小的增长，样本均值将收敛到我们的期望值。换句话说，随着我们抛硬币的次数越来越多，样本平均值应该越来越接近 0.5。

## Python 和随机数

在本章中，我们将多次使用 random 库，但它实际上并不是真正的随机数——它是我们所谓的伪随机数。**伪随机**数是通常从算法生成的数。我们使用一个称为**种子**的数字初始化算法。很多时候，种子是基于程序执行的时间或日期。然而，Python（以及大多数其他语言）允许你将种子设置为任何你想要的数字。如果你用相同的种子初始化你的算法，那么每次都会生成相同的伪随机数。当你使用随机数并希望每次产生相同的结果时，这是很有用的。

## 练习 9.01：大数定律的实践

让我们在 Python 中扩展我们的抛硬币实验。首先，让我们创建一个抛硬币模拟器。打开你的 Jupyter 笔记本并输入以下代码：

1.  我们首先需要导入`random` Python 包并设置`seed`方法：

```py
# coin_flip_scenario.py
# import the random module
import random
random.seed(54321)
```

1.  让我们为样本大小定义一个变量，并在这种情况下将其设置为`10`：

```py
# set the sample size or coin flips you what to run
sample_size = 10
```

1.  我们创建一个空列表，以便收集我们抛硬币实验的结果：

```py
# create a for loop and collect the results in a list
# 1 = heads and 0 = tails
result_list = []
for i in range(sample_size):
    result = random.randint(0, 1)
    result_list.append(result)
```

1.  定义两个变量来编译结果（正面次数和每次抛硬币的平均正面次数）：

```py
# compile results
num_of_heads = sum(result_list)
avg_of_heads = float(num_of_heads) / sample_size
```

1.  最后，我们将结果打印到控制台：

```py
# print the results
print(f'Results: {num_of_heads} heads out of {sample_size} \
flips.')
print(f'Average number of heads per flip is {avg_of_heads}.')
```

1.  运行你的笔记本应该得到以下类似的结果：

```py
Results: 4 heads out of 10 flips. Average number of 
heads per flip is 0.4.
```

1.  由于我们在这个模拟中生成随机数，你得到的结果可能会有所不同。在 10 次抛硬币中得到 4 次正面（每次抛硬币平均 0.4 次正面）似乎是可能的，但与我们的期望值 0.5 不同。但是请注意，当我们将样本大小从 10 增加到 100 时会发生什么：

```py
# set the sample size or coin flips you what to run
sample_size = 100
```

1.  重新运行整个程序（确保包括带有`random.seed(54321)`的行），这次结果将如下所示：

```py
Results: 51 heads out of 100 flips. Average number     of heads per flip is 0.51.
```

注意，样本平均值（`0.51`）现在与样本大小为 100 相比，更接近期望值（`0.50`）而不是 10。这是大数定律的一个典型例子。

注意

要访问本节的源代码，请参阅 https://packt.live/2VCT9An。

你也可以在 https://packt.live/2NOMGhk 上在线运行此示例。

## 练习 9.02：随时间变化的抛硬币平均值

让我们回到我们的抛硬币模拟器代码，并将其构建出来，以便在我们抛硬币时保持一个运行的样本平均值。我们将抛硬币 20,000 次，并使用折线图来显示样本均值随时间的变化，并与我们的期望值进行比较。

1.  导入`random`和`matplotlib` Python 包并设置随机种子：

```py
# coin_clip_scenario_2.py
# import the module
import random
import matplotlib.pyplot as plt
random.seed(54321)
```

1.  定义样本大小或抛硬币次数：

```py
# set the sample size or coin flips you what to run
sample_size = 20000
```

1.  初始化我们将用来收集模拟结果的变量：

```py
# initialize the variables required for our loop
# 1 = heads and 0 = tails
num_of_heads = 0
heads_list = []
trials_list = []
freq_list = []
```

1.  运行模拟并收集结果：

```py
# create a for loop and collect the results in a list
for i in range(1,sample_size+1):
    result = random.randint(0, 1)
    if result == 1:
        num_of_heads += 1
    avg_of_heads = float(num_of_heads) / i
    heads_list.append(num_of_heads)
    trials_list.append(i)
    freq_list.append(avg_of_heads)
```

1.  将结果打印到控制台：

```py
# print the results
print(f'Results: {num_of_heads} heads out of {sample_size} flips.')
print(f'Average number of heads is {avg_of_heads}')
```

1.  创建一条线图，显示随时间变化的样本均值，并使用虚线标记我们的期望值：

```py
#create a simple line graph to show our results over time
plt.plot(trials_list, freq_list)
plt.ylabel('Sample Average')
plt.xlabel('Sample Size')
plt.hlines(0.50,0,sample_size,linestyles='dashed')
plt.show()
```

1.  运行我们的笔记本将产生以下结果：

```py
Results: 10008 heads out of 20000 flips. Average number of 
heads is 0.5004
```

该代码将生成以下图表，显示随着样本量的增加，每次抛硬币的平均正面朝上数量的变化（用实线表示）。请注意，大约在 2,000 次抛硬币后，样本均值与期望值相匹配（约为每次抛硬币 0.5 个正面朝上）：

![图 9.3：样本量每次抛硬币的平均正面朝上数量](img/B15968_09_03.jpg)

图 9.3：样本量每次抛硬币的平均正面朝上数量

注意

要访问此特定部分的源代码，请参阅 https://packt.live/2BZcR2h。

您还可以在 https://packt.live/31AIxpc 上在线运行此示例。

## 大数定律在现实世界中的实际应用

通过概率的角度分析的最佳赌场游戏之一是轮盘赌。玩这个游戏相对简单。游戏的中心是一个巨大的轮盘，上面有 1 到 36 的空格和标签，0 和 00（双零）。奇数是红色的，偶数是黑色的，两个零空格都是绿色的。轮盘旋转，球被放入轮盘空格中，与轮盘旋转的方向相反。最终，球落入轮盘上的 38 个空格之一。球落在哪里的结果是人们下注的对象。他们可以下很多不同类型的赌注，从落在一个特定的数字上到球会落在哪个颜色的空格上。赌场根据您下注的类型进行支付。当大多数人第一次看到轮盘赌时，他们中的许多人会问这样的问题：“两个绿色空格是怎么回事？”我们将在接下来的几页中清楚地看到为什么绿色空格对赌场非常重要，但首先让我们谈谈我们可以从玩轮盘赌中期望什么。

为了使游戏更简单，我们将每次都下注球落在红色号码上。赢得这样的赌注的支付比例是 1:1，所以如果我们下注 5 美元并赢了，我们可以保留我们的 5 美元并赢得 5 美元。如果我们输了赌注，我们什么也不赢，失去了 5 美元的赌注。如果我们下注红色，以下是可能发生的概率：

+   在红色上下注，如果球落在红色上，我们赢得：![1](img/B15968_09_InlineEquation1.png)

+   在红色上下注，如果球落在黑色上，我们失去：![2](img/B15968_09_InlineEquation1a.png)

+   在红色上下注，如果球落在绿色上，我们失去：![3](img/B15968_09_InlineEquation2.png)

让我们看看以我们可以赢得或失去的金额来看结果，当下注 1 美元时：

+   在红色上下注，如果球落在红色上，我们赢得 1 美元

+   在红色上下注，如果球落在黑色上，我们失去 1 美元

+   在红色上下注，如果球落在绿色上，我们失去 1 美元

这是一个离散分布的例子。要计算离散分布的期望值，您需要将结果的值乘以它发生的概率。如果您看一下前面的两个列表，我们有轮盘赌游戏中每个结果的概率和值，所以现在我们可以计算我们可以赢得或输掉的预期金额：

*(落在红色上的概率*落在红色上时的赢或输) + (落在黑色上的概率*落在黑色上时的赢或输) + (落在绿色上的概率*落在绿色上时的赢或输)*

现在，如果我们根据我们计算的概率计算我们可以赢得的预期金额，我们将得到*(0.474*1)+(0.474*-1)+(0.053*-1) ≈ -0.05*值：

前面的计算告诉我们，我们预计每下注 1 美元在红色上会损失约 5 美分。如果我们增加下注，我们预计会损失更多的钱。

## 练习 9.03：计算轮盘赌游戏的平均赢利如果我们不断下注红色

让我们重新调整我们的模拟代码，模拟玩轮盘赌游戏并跟踪每场游戏我们赢得或输掉的平均金额。然后，我们将像我们为抛硬币的情况那样绘制结果的图表：

1.  导入`random`和`matplotlib`包：

```py
# roulette simulation.py
# import the module
import random
import matplotlib.pyplot as plt
random.seed(54321)
```

1.  创建一个样本大小的变量，并将其设置为`10`。创建一个名为`bet`的变量，并将其设置为 1 美元：

```py
# set the number of games of roulette you want to play
sample_size = 10
#set the amount of money you want to bet
bet = 1
```

1.  初始化我们将用于收集模拟结果的变量：

```py
# initialize the variables required for our loop
# 1 to 36 represent numbers on roulette wheel, 37 represents 0, 38 represents 00
net_money = 0
wins = 0
money_track = []
trials_track = []
```

1.  运行模拟并收集结果：

```py
# create a for loop and collect the results in a list
for i in range(1,sample_size+1):
    result = random.randint(1,38)
    if result % 2 == 1 and result != 37:
        net_money += bet
        wins += 1
    else:
        net_money -= bet
    money_track.append(net_money/i)
    trials_track.append(i)
```

1.  打印模拟结果和平均期望值：

```py
# print the results
print(f'Results: You won {wins} games out of\
{sample_size} and won an average of\
{net_money/sample_size} dollars per game')
print(f'Results: You are expected to win\
{((18/38)*bet+(20/38)*(-bet))} per game')
```

1.  绘制每场游戏中净变化的期望值和净变化的样本平均值的图表：

```py
#create a simple line graph to show our results over time
plt.plot(trials_track, money_track)
plt.ylabel('Net Money')
plt.xlabel('Number of games')
plt.hlines(((18/38)*bet+(20/38)*(-bet)), 0,            sample_size, linestyles='dashed')
plt.show()
```

1.  运行你的笔记本，你会得到以下结果：

```py
Results: You won 4 games out of 10 and won an average of -0.2 dollars per game
Results: You are expected to win -0.05263157894736842 per game
```

上述代码将生成以下图表：

![图 9.4：进行了 10 场游戏的轮盘每场游戏的平均净收益](img/B15968_09_04.jpg)

图 9.4：进行了 10 场游戏的轮盘每场游戏的平均净收益

在上图中，实线代表我们玩的 10 场游戏中每场游戏的平均赢钱数。虚线代表我们每场游戏可以期望赢得或输掉的金额。我们应该每场游戏输掉大约 5 美分，但在这种特定情况下，我们总共输掉了 20 美分，远少于每场游戏输掉 5 美分。如果从代码中删除`random.seed(54321)`并重新运行模拟，结果将会有所不同。随意尝试并改变每次下注的金额，看看会发生什么。

注意

要访问此特定部分的源代码，请参阅 https://packt.live/3dTdlEb。

你也可以在 https://packt.live/2ZtkOEV 上在线运行此示例。

但这并不反映赌场的情况。没有赌场一天只开放 10 场轮盘赌游戏。那么，如果我们将游戏次数从 10 次改为 100,000 次，我们的图会发生什么变化？将样本大小变量设置为 100,000 并重新运行代码，得到的图看起来像这样：

![图 9.5：进行了 100,000 场游戏的轮盘每场游戏的平均净收益](img/B15968_09_05.jpg)

图 9.5：进行了 100,000 场游戏的轮盘每场游戏的平均净收益

请注意，蓝线迅速收敛到每场游戏的平均净收益-0.05 美元。具体来说，这次模拟产生了-0.054 美元的净收益，与我们的预期值相差不远。实际上，长期来看，赌场会赚钱，赌徒会输钱。现在，回到绿色空间的问题。如果我们从游戏中移除它们，将会有 18 个红色和 18 个黑色的空间。让我们在这些条件下重新计算我们的期望值：

*(落在红色上的概率*落在红色上时的赢得或输掉的钱)+*

*(落在黑色上的概率*落在黑色上时的赢得或输掉的钱)*

![图 9.6：计算期望值的公式](img/B15968_09_06.jpg)

图 9.6：计算期望值的公式

这意味着没有绿色空间，赌场和赌徒在长期内都不会赢钱或输钱；双方都会带着与他们开始时一样的金额离开。

# 中心极限定理

通过快速回顾前一节，大数定律告诉我们，随着样本的增大，样本均值越接近于总体均值。虽然这告诉我们样本均值的值应该是什么，但它并不告诉我们分布的任何信息。为此，我们需要中心极限定理。**中心极限定理**（**CLT**）指出，如果样本量足够大，样本均值的分布近似正态分布，均值为总体均值，标准差为总体标准差除以*n*的平方根。这很重要，因为我们不仅知道我们的总体均值可以取得的典型值，而且我们也知道分布的形状和方差。

## 正态分布和中心极限定理

在*第八章*，*基础概率概念及其应用*中，我们看了一种连续分布，称为*正态分布*，也称为钟形曲线或高斯曲线（这三个名称的意思是一样的）。虽然有许多正态分布的实例，但这并不是它特殊的主要原因。正态分布之所以特殊是因为许多统计数据的分布都遵循正态分布，包括样本均值。

了解样本均值的分布在我们日常解决的许多典型统计问题中非常重要。我们获取均值和方差信息并将其放在一起，以了解我们的样本均值将如何从样本到样本变化。这告诉我们样本均值是否是我们期望出现的东西，还是我们不期望出现并且需要更仔细研究的东西。我们可以从据称相同的两个不同群体中取两个不同的样本，并证明它们实际上彼此显着不同。

## 从均匀分布中随机抽样

我们可以通过在 Python 中构建一些模拟来说明和验证中心极限定理，这正是我们将在接下来的练习中要做的。我们将要运行的第一个模拟是从均匀分布中随机抽取样本。**均匀分布**是每个结果被选中的可能性都是相等的分布。如果我们绘制均匀分布，它看起来像是一条横穿页面的直线。均匀分布的一些例子是掷骰子、抛硬币或典型的随机数生成器。

## 练习 9.04：显示均匀分布的样本均值

让我们从一个生成 0 到 100 之间的随机数的随机数生成器中抽取一个随机样本并计算样本平均值：

1.  导入我们将使用的以下 Python 包并设置`seed`：

```py
# sample_from_uniform_dist.py
# import the module
import random
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.stats as stats
random.seed(54312)
```

1.  创建每个样本的大小和要抽取的总样本数的变量。由于中心极限定理规定我们需要足够大的样本，我们选择了样本大小为 30。接下来，我们需要大量的样本均值来绘制图表，并将该值设置为 10,000：

```py
# select the sample size you want to take
sample_size = 30
# select the number of sample mean you want to simulate
calc_means = 10000
```

1.  初始化我们将用于收集样本均值的列表，并运行我们的模拟指定次数，收集每个样本的样本均值：

```py
mean_list = []
# run our loop and collect a sample
for j in range(calc_means):
    # initialize the variables to track our results
    sample_list = []
    for i in range(sample_size):
        sample_list.append(random.randint(0, 100))
    sample_mean = sum(sample_list) / sample_size
    mean_list.append(sample_mean)
```

1.  创建我们收集的样本均值的直方图。在直方图的顶部，我们将覆盖中心极限定理所说的样本均值分布应该是什么样子的：

```py
"""
create a histogram of our sample and compare it 
to what the CLT says it should be 
"""
n, bins, patches = plt.hist(mean_list, \
                            math.floor(math.sqrt(calc_means)),\
                            density=True, facecolor='g', alpha=0.75)
plt.grid(True)
mu = 50
sigma = math.sqrt(((100 ** 2) / 12)) / (math.sqrt(sample_size))
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()
```

在我们的笔记本中运行此代码将给我们以下结果：

![图 9.7：从样本大小为 30 的均匀分布的 10,000 个样本中的样本平均值的分布](img/B15968_09_07.jpg)

图 9.7：从样本大小为 30 的均匀分布的 10,000 个样本中的样本平均值的分布

中心极限定理给出的期望分布几乎完全覆盖了我们模拟结果的直方图。随意尝试并更改样本大小和用于生成图表的样本均值的数量。

注意

要访问此特定部分的源代码，请参阅 https://packt.live/31JG77I。

您还可以在 https://packt.live/3ggAq5m 上在线运行此示例。

## 从指数分布中随机抽样

我们知道中心极限定理适用于从均匀分布中取得的样本均值，但是对于看起来一点也不像均匀分布的东西呢？中心极限定理不限制我们抽取的样本的分布，那么对于看起来一点也不像正态分布的东西，它会起作用吗？让我们看看指数分布。**指数分布**是一种分布，它在从左到右迅速下降，然后趋于平稳，但并没有完全接触到零。以下图表是典型的指数分布：

![图 9.8：指数分布示例](img/B15968_09_08.jpg)

图 9.8：指数分布示例

现实世界中有很多指数分布的例子。例如，热液体冷却的速度，放射性衰变，以及机械零件的故障建模。

## 练习 9.05：从指数分布中抽取样本

在这个练习中，我们将随机抽样指数分布。以下是我们可以用来模拟从指数分布中抽样的代码：

1.  导入我们需要的 Python 包。为了看到取较小样本的影响，我们将样本大小设置为`5`（参考以下代码），但保持样本数为`10000`：

```py
# sample_from_exp_dist.py
# import the module
import random
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.stats as stats
# select the sample size you want to take
sample_size = 5
# select the number of sample mean you want to simulate
calc_means = 10000
```

1.  初始化我们将用来收集模拟结果的变量。运行模拟，但这次从指数分布中取样，而不是均匀分布：

```py
mean_list = []
# run our loop and collect a sample
for j in range(calc_means):
    # initialize the variables to track our results
    sample_list = []
    for i in range(sample_size):
        draw = np.random.exponential(1)
        sample_list.append(draw)
    sample_mean = sum(sample_list) / sample_size
    mean_list.append(sample_mean)
```

1.  创建我们收集的样本均值的直方图，并叠加中心极限定理对其的预期：

```py
""" create a histogram of our sample and compare it to what the CLT says it should be """
n, bins, patches = plt.hist(mean_list, \
                   math.floor(math.sqrt(calc_means)), \
                   density=True, facecolor='g', \
                   alpha=0.75)
plt.grid(True)
mu = 1
sigma = 1 / (math.sqrt(sample_size))
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()
```

1.  在我们的 Jupyter 笔记本中输入的代码将给我们以下图形：![图 9.9：来自指数分布的 5 个样本的 10,000 个样本平均值分布](img/B15968_09_09.jpg)

图 9.9：来自指数分布的 5 个样本的 10,000 个样本平均值分布

与之前的练习*Exercise 9.04*中的*显示均匀分布的样本均值*一样，橙色线告诉我们中心极限定理对我们的预期。虽然我们的绿色直方图与我们的预期相似，但显然向右倾斜，根本不是钟形曲线。但请记住，中心极限定理要求我们取足够大的样本。显然，5 不够大，所以让我们将样本大小从 5 增加到 50 并重新运行代码。这样做应该会产生以下结果：

![图 9.10：来自指数分布的 50 个样本的 10,000 个样本平均值分布](img/B15968_09_10.jpg)

图 9.10：来自指数分布的 50 个样本的 10,000 个样本平均值分布

这看起来更接近我们的预期。显然，50 个样本的大小足够大，而 5 个样本的大小不够。但现在你可能会有一个问题：“什么样本大小足够大，我们如何知道？”。答案实际上取决于基础分布；基础分布偏斜越大，您必须取足够大的样本来确保中心极限定理的样本足够大。

注意

要访问此特定部分的源代码，请参阅 https://packt.live/2D2phXE。

您也可以在 https://packt.live/2NRcvNP 上在线运行此示例。

在本章后面，我们将介绍如何计算所需的样本大小，但我们只考虑中心极限定理适用于样本大小为 30 或更大的样本。

# 置信区间

正如我们在之前的模拟中看到的，我们的样本均值可能会因样本而异。在模拟中，我们有奢侈的取 10,000 个样本的条件，但在现实世界中我们做不到；这将是非常昂贵和耗时的。通常，我们只有足够的资源来收集一个样本。那么我们如何能对我们样本的结果有信心呢？在报告我们的样本均值时，有没有办法考虑到这种变异性？

好消息是中心极限定理给了我们关于样本均值的方差的一个概念。我们可以应用中心极限定理，并通过使用置信区间来考虑抽样变异性。更一般地，**置信区间**是一个统计量（样本均值的一个例子）的一系列值，基于一个具有一定置信度的分布，用于估计包含真实均值的值的可能性有多大。我们不总是只计算样本均值的置信区间；这个想法适用于从样本中计算出的任何统计量（唯一的区别是如何计算它）。置信区间可以用来计算我们需要抽取多大的样本以及误差范围是多少。

## 计算样本均值的置信区间

我们将计算的第一种置信区间是**z 置信区间**，它将根据标准正态模型（有时称为 z 分布）为我们的样本均值提供一个区间（或范围）的值。

为了计算样本均值的 z 置信区间，我们需要知道四件事：

+   样本均值

+   样本大小

+   人口方差

+   关键值或某个置信水平

我们计算样本均值和大小是从我们收集的样本中计算出来的。人口方差不是我们从样本中计算出来的；人口方差是一个给定给我们的值。通常，这是一些先前研究和研究中给出的方差的接受值。谜题的最后一块是关键值，或置信水平；这就是正态分布和中心极限定理发挥作用的地方。为了了解关键值是什么，让我们看一下标准正态分布（它是一个总是具有均值为 0 和方差为 1 的正态分布）及其曲线下的面积：

![图 9.11：标准正态模型的示例](img/B15968_09_11.jpg)

图 9.11：标准正态模型的示例

我们知道在我们的正态分布中，我们的均值在中心（在这种情况下是 0）。曲线下面积从-1 到 1 占总面积的 68%。另一种说法是，由这个分布描述的值中有 68%在-1 和 1 之间。大约 95%的值在-2 和 2 之间。将这应用于样本均值的分布，我们可以找到 95%的样本均值将取的范围。参考*图 9.7*：

![图 9.12：来自 30 个样本的均匀分布的 10,000 个样本的样本平均值分布](img/B15968_09_12.jpg)

图 9.12：来自 30 个样本的均匀分布的 10,000 个样本的样本平均值分布

如果我们看一下，我们的钟形曲线的中心是 50，这是从 0 到 100 的均匀分布的预期值。从 0 到 100 的均匀分布的预期标准差约为 5.27 (![4](img/B15968_09_InlineEquation3.png))。因此，应用与之前相同的逻辑，大约 68%的值在 45 和 55 之间，大约 95%的值在 40 和 60 之间。这些范围就是我们的置信区间。

计算 z 置信区间的更正式方程如下：

![图 9.13：计算 z 置信区间的公式](img/B15968_09_13.jpg)

图 9.13：计算 z 置信区间的公式

在这个方程中：

+   *x̄*是样本均值。

+   *n*是样本大小。

+   *σ*是人口标准差。

+   *Z*是我们置信水平的关键值。

我们最终的置信区间将是两个数字：一个上限，我们将两个项相加，一个下限，我们将两个项相减。幸运的是，这是我们可以在 Python 中编写一个函数的事情，如下所示：

```py
def z_confidence_interval(data, st_dev, con_lvl):
    import statistics as st
    import scipy.stats as sp
    import math
    sample_mean = st.mean(data)
    n = len(data)
    crit_value = sp.norm.ppf(((1 - con_lvl) / 2) + \
                             con_lvl)
    lower_limit = sample_mean - (crit_value * \
                                 (st_dev/math.sqrt(n)))
    higher_limit = sample_mean + (crit_value * \
                                  (st_dev / math.sqrt(n)))
    print (f'Your {con_lvl} z confidence interval         is ({lower_limit}, {higher_limit})')
    return (lower_limit,higher_limit)
```

此函数将以下内容作为输入：我们收集的数据，以及总体标准偏差（由我们给出），以及置信水平。它将在控制台上打印置信水平并将其作为元组返回。

## 练习 9.06：找到民意调查数据的置信区间

您正在进行一场政治竞选，并决定进行 30 个焦点群体，每个群体约有 10 人。您获得了结果，并希望向您的候选人报告典型 10 人群体中会投票给他们的人数。由于每个焦点群体都有一些变化，您决定最准确的方法是给出 95%的 z-置信区间。您假设根据过去的经验，标准偏差为 2.89。让我们使用 Python 对此进行建模：

1.  导入`random` Python 包并将种子设置为`39809`。这将确保我们每次运行程序时都获得相同的结果：

```py
import random
random.seed(39809)
```

1.  初始化我们的样本列表并从焦点群体中收集我们的样本。然后，我们只需将信息输入到我们的函数中：

```py
sample_list = []
for i in range(30):
    sample_list.append(random.randint(0, 10))
z_confidence_interval(sample_list,2.89,0.95)
```

1.  如果您做得正确，那么在运行笔记本时应打印出以下内容：

```py
Your 0.95 z confidence interval is (3.965845784931483, 6.034154215068517)
```

这告诉我们，在典型的焦点群体中，每个群体中有 4 到 6 人会投票给我们的候选人。这向您表明，竞选活动应继续努力说服更多人投票给您的候选人。

注意

要访问此特定部分的源代码，请参阅 https://packt.live/2Zp6XiU。

您还可以在 https://packt.live/3eUBL1B 上在线运行此示例。

## 小样本置信区间

z-置信区间适用于样本足够大的情况（记住我们的经验法则是样本大小为 30 或更大）；但是如果您的样本不够大怎么办？那么您可以使用**t-置信区间**，它基本上与 z-置信区间相同，但有两个例外：

+   t-置信区间不假设您知道总体标准偏差，因此我们使用样本标准偏差。

+   它使用 t-分布来计算临界值，而不是 z（标准正态）分布。两者之间的区别在于 t-分布在平均值周围的集中程度较低，以解释不知道总体标准偏差的情况。

为了计算 t-置信区间，我们需要知道两件事；第一是自由度，它是样本大小减 1（*n-1*）计算得出。第二是置信水平。t-置信区间的公式如下：

![图 9.14：计算 t-置信区间的公式](img/B15968_09_14.jpg)

图 9.14：计算 t-置信区间的公式

在这个方程中：

+   *x̄*是样本均值。

+   *t*n-1 是具有*n-1*自由度的临界值。

+   *s*是样本标准偏差。

+   *n*是样本大小。

就像 z-区间一样，我们的最终答案将是下限和上限。我们将编写一个 Python 函数来为我们完成所有计算工作：

```py
def t_confidence_interval(data, con_lvl):
    import statistics as st
    import scipy.stats as sp
    import math
    sample_mean = st.mean(data)
    sample_st_dev = st.stdev(data)
    n = len(data)
    crit_value = sp.t.ppf(((1 - con_lvl) / 2) + \
                          con_lvl, n-1)
    lower_limit = sample_mean - (crit_value * \
                  (sample_st_dev/math.sqrt(n)))
    higher_limit = sample_mean + (crit_value * \
                   (sample_st_dev/math.sqrt(n)))
    print(f'Your {con_lvl} t confidence interval is \
({lower_limit},{higher_limit})')
    return (lower_limit,higher_limit)
```

让我们使用与 z-置信区间相同的样本列表。`t_confidence_interval`函数的使用方式与我们的 z-置信区间函数相同；我们将输入要计算 t-置信区间的数据列表，并指定我们的置信水平。无需包括总体标准偏差；t-置信区间使用样本标准偏差，并将自动为我们计算。`t_confidence_interval`函数的正确使用方式如下：

```py
t_confidence_interval(sample_list,0.95)
```

如果您做得正确，当您运行前面的代码时，笔记本中应输出以下内容：

```py
Your 0.95 t confidence interval is (3.827357936126168,6.172642063873832)
```

注意，t-置信区间比我们的 z-置信区间更宽。这是因为我们在使用样本标准偏差估计总体标准偏差时存在更多的不确定性，而不是使用已知值。

t-置信区间的好处在于它不仅可以用于小样本或者你不知道总体标准差的情况；它可以在任何需要使用 z-置信区间的情况下使用。事实上，随着样本量的增大，t-分布越接近于 z（标准正态）分布。所以，如果你对所给的总体标准差的值不确定，或者在查看以前的研究时发现，你总是可以保险起见使用 t-置信区间。

## 样本比例的置信区间

让我们回到政治竞选的例子。在不同的焦点小组给出了不明确的结果之后，一项新的民意调查显示你的候选人正在赢得竞选，其中 350 人的样本中有 54%表示他们将投票给你的候选人，而你的对手得到了另外 46%。你想计算这个比例的置信区间，以便考虑抽样变异性。

我们知道如何计算样本均值的置信区间，但是如何计算样本比例的置信区间呢？样本的百分比与样本的均值不同。幸运的是，我们有一个计算样本比例置信区间的公式：

图 9.15：计算置信区间的公式

](image/B15968_09_15.jpg)

图 9.15：计算置信区间的公式

在这个方程中：

+   *p̂*是样本比例。在这个例子中，是投票给你的人的 54%。

+   *n*是样本量。在这个例子中，是 350 人。

+   *Z*是我们从标准正态分布中得到的临界值。我们计算方法与 z-置信区间相同。

在应用之前，有一些条件需要满足：

+   我们样本中的观察结果是独立的-所以在我们的例子中，一个人的答案不会影响另一个人的答案。

+   我们需要至少有 10 个成功和 10 个失败-所以我们需要至少有 10 个人投票给我们，还有 10 个人会投票给你的对手。

同样，我们可以在 Python 中创建一个函数来进行计算：

```py
def prop_confidenct_interval(p_hat, n, con_lvl):
    import math
    import scipy.stats as sp
    crit_value = sp.norm.ppf(((1 - con_lvl) / 2) + \
                             con_lvl)
    lower_limit = p_hat - (crit_value * (math.sqrt(\
                 (p_hat * (1-p_hat)) / n)))
    higher_limit = p_hat + (crit_value * (math.sqrt(\
                  (p_hat * (1 - p_hat)) / n)))
    print(f'Your {con_lvl} proportional confidence \
interval is ({lower_limit},{higher_limit})')
    return (lower_limit,higher_limit)
```

与我们创建的其他函数不同，我们不需要输入我们数据值的列表。相反，我们可以直接输入我们的统计数据并设置置信水平。为了创建我们的民意调查的置信区间，我们输入信息如下：

```py
prop_confidenct_interval(0.54,350, 0.95)
```

并且以下结果将被打印在控制台中：

```py
Your 0.95 proportional confidence interval is (0.4877856513683282,0.5922143486316719)
```

这告诉我们，我们可以有 95%的把握，我们的候选人得到的选票比例的真实值在 48.8%和 59.2%之间。因此，民意调查的结果是不确定的，这表明我们仍然需要更多的工作来说服人们投票给我们的候选人。请注意，这通常是民意调查得到的误差范围。**误差范围**是我们的点估计器（在这个例子中是*p̂*）与任一边界之间的距离（因为置信区间是对称的；无论我们使用上限还是下限都没有关系）。对于这次民意调查，我们的误差范围将是*0.592 - 0.54 = 0.052*。

因此，前面民意调查的误差范围约为 5.2%。这是在你接受任何民意调查结果时需要记住的事情，无论是政治还是其他方面。

# 假设检验

在前一节中，我们进行了模拟，样本均值在不同样本中发生了变化，尽管是从同一总体中抽样的。但是我们如何知道我们计算的样本均值是否与预设值或者不同样本显著不同？我们如何知道差异是变异性的作用，还是测量值不同？答案在于进行假设检验。

**假设检验**是一种旨在确定统计量是否与我们的预期显著不同的统计检验。假设检验的例子包括检查样本均值是否与预先建立的标准显著不同，或者比较两个不同样本，看它们是否在统计上不同或相同。

## 假设检验的部分

任何假设检验都有三个主要部分：假设、检验统计量和 p 值。*假设*是你进行检验的对象，以确定它们是否应该被拒绝或接受。任何测试都有两个假设：一个**零假设**（通常用符号*H*0 表示）和一个**备择假设**（通常用符号*H*A 表示）。零假设是我们一直假定或已知为真的东西；换句话说，它是我们预先建立的标准。备择假设是我们将要与零假设进行比较的备选项；在实际情况下，它是我们想要证明为真的东西。

以下是一些假设的例子：

+   你是一家制造公司的领导，你有一个通常每小时使用 15 升燃料的流程。你的公司正在测试对这个流程的改变，以尝试使用更少的燃料。他们对 24 小时进行了抽样，发现新流程每小时使用 13.7 升燃料。公司需要知道这种减少是否显著，或者是否可以归因于流程中的差异。你的零假设将是流程通常使用的：*H*O*: μ = 15*。我们希望证明新流程使用更少的燃料，所以我们的备择假设是：*H*A*: μ < 15*。

+   Richard 是你们城市的一名商业面包师。他在考虑是否要投资于他工厂的制面包设备。通常情况下，他的工厂在一个班次可以制作大约 15,000 个面包。Richard 派了一个班次去尝试新设备，连续 5 个班次平均每班次可以制作 17,500 个面包。你告诉 Richard 要测试一下这是否显著不同；零假设将基于他通常的产量（*H*O*: μ=15000*），备择假设将是他想要尝试证明的（*H*A*: μ = 15000*）。

+   Linda 是她公司质量控制部门的分析师。公司生产的一个零件需要 15 英寸长。由于公司无法测量每个零件，Linda 抽样了 100 个零件，发现这个样本的平均长度是 14.89 英寸。她告诉你，他们期望每个零件都是 15 英寸（*H*O*: μ = 15*），他们想要尝试弄清楚样本是否证明平均零件通常不是 15 英寸（*H*A*:μ ≠ 15*）。

前述情况中的每一种都描述了你将遇到的三种典型假设检验之一：上尾检验、下尾检验和双尾检验。了解你正在进行的测试类型是必要的，这样你才能正确地写出你的假设并计算你的 p 值。

**检验统计量**是一个描述我们观察到的样本与我们假设或已知的平均值相比的数字。这是我们进行不同测试时最容易变化的部分；它基于我们正在测试的特定统计量和所使用的检验。这是统计检验中最数学化的部分，通常用公式表示。**p 值**是我们假设检验的最后部分；它通常被定义为假设零假设为真时观察到类似我们收集的样本的概率。我们将这个值与某个显著性水平进行比较（0.05 是最常用的显著性水平）；如果我们的 p 值小于显著性水平，那么我们拒绝零假设，并且有证据表明备择假设为真。同样，如果我们的 p 值大于显著性水平，我们未能拒绝零假设，并且没有证据表明备择假设为真。

## Z-检验

就像我们的 z-置信区间一样，基于标准正态模型的假设检验称为**z-检验**。就像 z-置信区间一样，z-检验假设我们知道总体标准偏差，并且我们有足够大的样本（同样，经验法则是样本大小至少为 30）。z-检验的基本设置如下：

+   *H*O*:μ = μ*O（不用担心；*μ*O 通常是我们认为的平均值，只是一个数字）

+   *H*A*: μ < μ*O 或 *H*A*: μ > μ*O 或 *H*A*: μ ≠ μ*O（*μ*O 将始终与我们的零假设相匹配）

+   检验统计量：![5](img/B15968_09_InlineEquation4.png)

其中：

*x̄*是样本平均值。

*σ*是已知的总体标准偏差。

*n*是样本大小。

+   P 值：![6](img/B15968_09_InlineEquation5.png)

一旦你掌握了这些数学，这些计算就不难，我们可以使用 Python 使计算变得非常简单。

## 练习 9.07：Z-检验实例

让我们从具有已知总体均值的分布中随机抽取一个样本，并看看我们的 z-检验是否能选择正确的假设：

1.  让我们从导入我们将需要的所有库开始这个练习，以便能够运行我们的代码并设置`seed`值：

```py
import scipy.stats as st
import numpy as np
import pandas as pd
import math as mt
import statistics as stat
import statsmodels.stats.weightstats as mod
import statsmodels.stats.proportion as prop
np.random.seed(12345)
```

1.  我们将编写一个函数来进行 z-检验。输入将是一个样本（以列表的形式），总体标准偏差（记住，指定这一点是 z-检验的要求之一），我们假设的值，测试的显著性水平，以及测试类型（上尾、下尾或双尾检验）。我们将从给定的列表中计算样本均值和样本大小。然后，我们将输入计算我们的检验统计量。然后，根据我们决定要进行的假设检验，我们相应地计算 p 值。最后，我们将我们的 p 值与显著性水平进行比较，如果它小于我们的显著性水平，我们拒绝零假设。否则，我们未能拒绝零假设：

```py
def z_test(sample, pop_st_dev, hypoth_value, \
           sig_level, test_type):
    sample_mean = stat.mean(sample)
    sample_size = len(sample)
    test_statistic = (sample_mean - hypoth_value) / \
                     (pop_st_dev / (mt.sqrt(sample_size)))
    if test_type == 'lower':
        p_value = st.norm.cdf(test_statistic)
    if test_type == 'upper':
        p_value = 1 - st.norm.cdf(test_statistic)
    if test_type == 'two':
        p_value = 2 * (1 - st.norm.cdf(abs(            test_statistic)))
    print(f'P Value = {p_value}')
    if p_value < sig_level:
        print(f'Results are significant. Reject the Null')
    else:
        print(f'Results are insignificant. '\
               'Do Not Reject the Null')
```

1.  我们从均值为`15`，标准偏差为`1`的正态分布中抽取一个随机样本大小为`50`。我们将把样本均值打印到控制台，以便我们知道它是多少（每次运行此代码时它都会有所不同，因为我们每次都会抽取一个随机样本）。我们使用我们的 z-检验函数进行一个下尾检验，因为我们想要看到我们的均值是否显著小于`16`。我们指定包含我们数据的列表（`data1`），总体标准偏差（我们知道这是`1`），假设的值（我们想要看到它是否显著小于`16`），显著性水平（大多数情况下这将是`0.05`），最后是测试类型（因为我们想要看到均值是否小于`16`，这是一个下尾检验）：

```py
# 1 - Lower Tailed Test
# Randomly Sample from Normal Distribution mu=     and st_dev = 3
data1 = np.random.normal(15, 1, 50)
# Test to see if Mean is significantly less then 16
print(f'Sample mean: {stat.mean(data1)}')
z_test(data1,1,16,0.05,'lower')
# most of the time, the null should be rejected
```

当我们运行此代码时，我们应该得到类似以下的结果：

```py
Sample mean: 14.94804802516884
P Value = 5.094688086201483e-14
Results are significant.  Reject the Null
(-7.43842374885694, 5.094688086201483e-14)
```

由于我们的检验统计量的 p 值小于 0.05（从科学计数法写出来，是 0.0000000000000509），我们知道 15.06 的样本均值显著小于 16，基于我们的样本量为 50。由于我们从平均值为 15 的总体中抽取了样本，测试结果符合我们的预期。同样，由于我们一开始是随机抽样的，您的结果可能会有所不同，但是对于大多数样本来说，这个测试应该会拒绝零假设。在返回的元组中，第一个值是检验统计量，第二个是我们的 p 值。

1.  接下来，让我们测试一下我们的均值是否显著大于`14`。按照单侧检验的相同模式，我们的代码将如下所示：

```py
#test to see if the mean is significantly more than 14
print(f'Sample mean: {stat.mean(data1)}')
z_test(data1,1,14,0.05,'upper')
#most of the time the null should reject
```

当我们运行代码时，以下输出将显示在控制台中：

```py
Sample mean: 14.94804802516884
P Value = 1.0159539876042345e-11
Results are significant.  Reject the Null
(6.703711874874011, 1.0159539876042345e-11)
```

1.  对于我们最后的 z 检验，我们将执行一个双侧检验，并查看我们的样本均值是否与`15`显著不同。在这个测试中，我们实际上并不关心它是高于还是低于`15`；我们只是想看看它是否不同：

```py
#test to see if the mean is significantly different than 15
print(f'Sample mean: {stat.mean(data1)}')
z_test(data1,1,15,0.05,'two')
#most of the type we should not reject the null
```

当我们运行此代码时，结果如下：

```py
Sample mean: 14.94804802516884
P Value = 0.7133535345453159
Results are insignificant.  Do Not Reject the Null
(-0.3673559369914646, 0.7133535345453159)
```

这个结果是有道理的，因为我们对平均值为 15 的总体进行了抽样。

注意

要访问此特定部分的源代码，请参阅 https://packt.live/2C24ItD。

您还可以在 https://packt.live/2NNyntn 上在线运行此示例。

## 比例 Z-检验

z 检验最常见的用途不是测试样本均值的显著性，而是测试百分比的显著性。这需要的假设与比例 z 置信区间的要求相同：随机样本、独立性，以及至少 10 次成功和 10 次失败。我们将按照以下方式计算此测试的检验统计量：

![图 9.16：计算检验统计量的公式](img/B15968_09_16.jpg)

图 9.16：计算检验统计量的公式

我们将以与样本均值的 z 检验相同的方式计算 p 值。我们不需要为此测试创建一个函数；在`statsmodels.stats.proportion` Python 包中已经存在一个名为`proportions_ztest`的函数。此函数的语法如下：

```py
proportions_ztest(x,n,Po, alternative=['smaller',\
                                       'larger','two-sided'])
```

这里：

`x`是我们样本中的成功次数。

`n`是我们样本的大小。

`Po`是我们想要进行检验的假设值。

备用项指定了一个单侧、双侧或双侧检验。

此函数的输出是一个元组；第一个元素是检验统计量，第二个元素是 p 值。让我们回到我们的民意调查例子：你的竞选活动进行了一项民意调查，并对 350 人进行了抽样。在 350 人中，有 193 人表示他们会投票给你。我们想要看看我们收集的这个样本是否证明了大多数人会投票给你。

我们将把我们的 z 检验结果分配给一个名为`results`的变量。我们调用函数，其中`193`是成功/将为我们投票的人数，样本大小为`350`。由于我们想要测试我们的样本是否证明我们获得了大多数选票，我们想要执行一个单侧检验，其中假设值为`0.50`：

```py
#z-test for proportion
results = prop.proportions_ztest(193,350,.50, \
                                 alternative='larger')
print(results)
```

当代码运行时，以下内容将打印到控制台：

```py
(1.93454148164361, 0.026523293494118718)
```

我们的 p 值约为 0.027，这是一个显著的结果，显著水平为 0.05。这告诉我们，我们的样本证明了我们获得了大多数选票。

## T-检验

虽然 z 检验对比例进行假设检验很有用，但在测试样本均值时并不是很实用，因为我们通常不知道总体的标准差。还有其他情况下，我们的样本量非常小。对于这种情况，我们可以使用 t 检验，它类似于我们的 t 置信区间。就像 t 置信区间一样，您不需要知道总体标准差；您可以使用样本来估计它。

t 检验的公式如下：

![图 9.17：计算 t 检验的公式](img/B15968_09_17.jpg)

图 9.17：计算 t-检验的公式

在这个方程中：

+   *x̄*是样本均值。

+   *μ*O 是我们正在测试的假设值。

+   *s*是样本标准差。

+   *n*是样本大小。

我们将使用 t-分布而不是标准正态分布来计算 p 值。但是，我们不会过多关注这个特定测试的机制，因为它与我们已经涵盖的其他假设检验类似。我们将创建一个函数来进行我们的 t-检验，类似于我们的 z-检验：

```py
def t_test(sample, hypoth_value, sig_level, test_type):
    sample_mean = stat.mean(sample)
    sample_st_dev = stat.stdev(sample)
    sample_size = len(sample)
    test_statistic = (sample_mean - hypoth_value) / \
                     (sample_st_dev/(mt.sqrt(sample_size)))
    if test_type == 'lower':
        p_value = st.t.cdf(test_statistic,df=sample_size-1)
    if test_type == 'upper':
        p_value = 1 - st.t.cdf(test_statistic,df=sample_size-1)
    if test_type == 'two':
        p_value = 2 * (1 - st.t.cdf(abs(test_statistic), \
                                    df=sample_size-1))
    print(f'P Value = {p_value}')
    if p_value < sig_level:
        print(f'Results are significant.  Reject the Null')
    else:
        print(f'Results are insignificant. '\
               'Do Not Reject the Null')
```

在上述代码中：

+   `sample`是样本测量的列表。

+   `hypoth_value`是您正在测试的值。

+   `sig_level`是显著性水平。

+   `test_type`是测试类型——较低、较高或两者。

## 练习 9.08：T-检验

我们将检查两个不同的样本：一个大样本和一个小样本。这两个样本都将从均值为 50、标准差为 10 的正态分布中随机选择。两个样本之间唯一的区别是大样本的大小为 100，而较小的样本的大小为 10：

1.  首先，让我们导入我们将使用的库，设置种子，然后随机生成我们的大样本：

```py
import scipy.stats as st
import numpy as np
import pandas as pd
import math as mt
import statistics as stat
import statsmodels.stats.weightstats as mod
import statsmodels.stats.proportion as prop
np.random.seed(1)
data1 = np.random.normal(50, 10, 100)
```

1.  为我们的 t-检验创建函数：

```py
def t_test(sample, hypoth_value, sig_level, test_type):
    sample_mean = stat.mean(sample)
    sample_st_dev = stat.stdev(sample)
    sample_size = len(sample)
    test_statistic = (sample_mean - hypoth_value) / \
                     (sample_st_dev/(mt.sqrt(sample_size)))
    if test_type == 'lower':
        p_value = st.t.cdf(test_statistic,df=sample_size-1)
    if test_type == 'upper':
        p_value = 1 - st.t.cdf(test_statistic,df=sample_size-1)
    if test_type == 'two':
        p_value = 2 * (1 - st.t.cdf(abs(test_statistic), \
                                    df=sample_size-1))
    print(f'P Value = {p_value}')
    if p_value < sig_level:
        print(f'Results are significant.  Reject the Null')
    else:
        print(f'Results are insignificant. '\
               'Do Not Reject the Null')
```

1.  我们将运行三个不同的测试：一个是看样本均值是否与`50`显著不同，一个是看样本均值是否显著低于`51`，还有一个是看样本均值是否显著高于`48`：

```py
print('large sample')
print(f'Sample mean: {stat.mean(data1)}')
t_test(data1,50,0.05,'two')
t_test(data1,51,0.05,'lower')
t_test(data1,48,0.05,'upper')
```

运行此代码将产生以下结果：

```py
large sample
Sample mean: 50.60582852075699
P Value = 0.4974609984410545
Results are insignificant.  Do Not Reject the Null
P Value = 0.32933701868279674
Results are insignificant.  Do Not Reject the Null
P Value = 0.002109341573010237
Results are significant.  Reject the Null
```

第一个测试是不显著的，我们没有证据表明均值与`50`显著不同。第二个测试也是不显著的；样本也不能证明均值显著大于`51`。最后一个测试是显著的；样本证明均值显著高于`48`。

1.  现在，我们将运行相同的三个测试，只是这次我们将使用大小为`5`的样本（我们将使用大样本的前`5`个元素）：

```py
# select the first 5 elements of the data set
data2 = data1[:5]
print(data2)
#two-tailed test = Is the sample mean significantly 
#different from 50?
print('small sample')
print(f'Sample mean: {stat.mean(data2)}')
t_test(data2,50,0.05,'two')
#lower tailed = Is the sample mean significantly 
#lower than 51?
t_test(data2,51,0.05,'lower')
#upper tailed = is the sample mean significantly 
#more than 48?
t_test(data2,48,0.05,'upper')
```

运行上述代码会产生以下结果：

```py
[66.24345364 43.88243586 44.71828248 39.27031378 58.65407629]
small sample
Sample mean: 50.553712409836436
P Value = 0.918572770568147
Results are insignificant.  Do Not Reject the Null
P Value = 0.4671568669546634
Results are insignificant.  Do Not Reject the Null
P Value = 0.32103491333328793
Results are insignificant.  Do Not Reject the Null
```

前两个测试的结果没有改变，而第三个测试尽管样本均值几乎相同，但结果确实改变了。差异的原因是由于样本量较小；由于样本量较小，不确定性较小，测试更保守，不太可能拒绝零假设。这可以在我们的检验统计量方程中显示：

![图 9.18：计算 t-检验的检验统计量的公式](img/B15968_09_18.jpg)

图 9.18：计算 t-检验的检验统计量的公式

注意分母![7](img/B15968_09_InlineEquation6.png)；如果*n*较小，则![8](img/B15968_09_InlineEquation6a.png)的值将较大（对于一个恒定的*s*）。这会导致测试统计量的分母值较大，从而导致整体测试统计量较小。

注意

要访问此特定部分的源代码，请参阅 https://packt.live/38mMShg。

您还可以在 https://packt.live/3gkBdlK 上在线运行此示例。

## 2-样本 t-检验或 A/B 测试

我们将要看的最后一个测试是 2-样本 t-检验。这是一个假设检验，比较两个不同样本的均值，并可以告诉您一个均值是否显著更高、显著更低或与另一个均值显著不同。其中之一的应用是一种称为 A/B 测试的东西。A/B 测试是您向两个不同的群体展示网站或应用的两个不同版本，并收集某种性能度量。性能度量的例子可能是花费的金额、点击广告的人数，或者人们在您的手机游戏内进行微交易的金额。收集数据后，您测试两个样本均值，并查看两个不同版本之间的差异是否显著。

对于双样本检验，零假设和备择假设的工作方式与单样本检验有些不同。你不是将样本均值与一个值进行比较，而是将其与另一个均值进行比较。我们通常通过将差异与零进行比较来展示这一点。使用一些代数，你可以弄清备择假设应该如何设置：

+   上侧（均值 1 大于均值 2）：![13](img/B15968_09_InlineEquation7.png)

+   下侧（均值 1 小于均值 2）：![14](img/B15968_09_InlineEquation8.png)

+   双尾（均值 1 与均值 2 不同）：![15](img/B15968_09_InlineEquation9.png)

对于双样本 t 检验，零假设将始终设置为 0 (![9](img/B15968_09_InlineEquation10.png))。换句话说，零假设是说两个均值之间没有差异，而另一个是说有差异。双样本 t 检验的检验统计量如下：

![10](img/B15968_09_InlineEquation11.png)，自由度为 ![16](img/B15968_09_InlineEquation12.png)

对于这个好消息是，我们不必手工计算，也不必费力创建自己的函数来执行这个操作。`scipy.stats` 包中有一个专门用于这个检验的函数。该函数如下：

```py
scipy.stats.ttest_ind(x1,x2,equal_var=False)
```

在这里：

+   `x1`是第一个样本中的数据列表。

+   `x2`是第二个样本中的数据列表。

+   我们将 `equal_var` 设置为 `False`，因为我们不知道两个样本的方差是否相同。

该函数返回两个值：有符号的检验统计量和 p 值。一些人可能已经注意到，没有选项来指定你正在执行哪种检验。这是因为该函数始终假定你正在进行双尾检验。那么你如何使用它来获得你的单尾检验的结果呢？由于 t 分布是对称的，单尾检验的 p 值将是双尾检验的 p 值的一半。第二件要看的事情是检验统计量的符号。对于下侧检验，只有在检验统计量为负数时才能拒绝零假设。同样，对于上侧检验，只有在检验统计量为正数时才能拒绝零假设。因此，对于单尾检验：

+   **下侧**：如果 ![a](img/B15968_09_InlineEquation13.png) 小于你的显著性水平并且你的检验统计量为负数，则拒绝零假设。

+   **上侧**：如果 ![b](img/B15968_09_InlineEquation13a.png) 小于你的显著性水平并且你的检验统计量为正数，则拒绝零假设。

## 练习 9.09：A/B 测试示例

我们有两个样本，一个来自均值为 50 的正态分布，另一个来自均值为 100 的分布。两个样本的大小都为 100。在这个练习中，我们将确定一个样本的样本均值是否显著不同、较低或较高于另一个样本：

1.  首先，让我们导入我们将使用的库：

```py
import scipy.stats as st
import numpy as np
```

1.  让我们绘制我们的随机样本并打印样本均值，这样我们就知道它们是什么。记得设置种子：

```py
# Randomly Sample from Normal Distributions 
np.random.seed(16172)
sample1 = np.random.normal(50, 10, 100)
sample2 = np.random.normal(100,10,100)
print(f'Sample mean 1: {stat.mean(sample1)}')
print(f'Sample mean 2: {stat.mean(sample2)}')
```

结果如下：

```py
Sample mean 1: 50.54824784997514
Sample mean 2: 97.95949096047315
```

1.  我们将使用 `scipy` 包中的函数执行双样本 t 检验并打印结果：

```py
two_tail_results = st.ttest_ind(sample1, sample2, \
                                equal_var=False)
print(two_tail_results)
```

结果如下：

```py
Ttest_indResult(statistic=-33.72952277672986,     pvalue=6.3445365508664585e-84)
```

由于默认情况下，该函数进行双尾检验，我们知道样本 1 的均值与样本 2 的均值显著不同。如果我们想进行下侧检验（其中样本 1 的均值显著低于样本 2），我们将使用相同的代码。唯一的区别是我们会将 p 值除以 2，并检查我们的检验统计量是否为负数。由于我们的 p 值除以 2 小于 0.05，并且我们的检验统计量为负数，我们知道样本 1 的均值显著低于样本 2 的均值。

1.  如果我们想测试样本 2 的均值是否显著高于样本 1 的均值，我们只需在函数中交换样本 1 和样本 2 的位置：

```py
upper_tail = st.ttest_ind(sample2, sample1, equal_var=False)
print(upper_tail)
```

结果如下：

```py
Ttest_indResult(statistic=33.72952277672986, 
pvalue=6.3445365508664585e-84)
```

就像下尾检验一样，我们会将 p 值除以 2。但是，我们会检查检验统计量是否为正。由于 p 值除以 2 小于 0.05 且检验统计量为正，我们知道样本 2 的均值明显大于样本 1 的均值。

注意

要访问此特定部分的源代码，请参阅 https://packt.live/3iuHmOr。

您还可以在 https://packt.live/3ghpdl4 上在线运行此示例。

## 线性回归简介

我们已经描述并测试了样本统计数据，但是如果我们想要使用数据的特征来描述另一个特征怎么办？例如，移动应用的价格如何影响下载量？为了做到这一点，我们将使用线性回归对数据进行建模。**线性回归**是指我们使用一个或多个自变量的线性方程来描述一个因变量。通常，我们的回归方程是斜率-截距形式，如下所示：

![图 9.19：线性回归公式](img/B15968_09_19.jpg)

图 9.19：线性回归公式

在这里：

+   β1 是我们方程的斜率，通常称为系数。

+   βO 是方程的截距。

我们如何得出系数和截距的值？它始于**残差**——即预测 y 值与实际 y 值之间的差异。查看残差的另一种方式是，这是我们方程预测偏离的量。虽然我们在这里不会详细介绍，但我们使用微积分来找出最小化所有残差总和的*β*1、*β*O 的值。我们不一定受限于一个系数；我们可以有多个（两个或更多）系数，如下所示：

![图 9.20：具有多个系数的线性回归公式](img/B15968_09_20.jpg)

图 9.20：具有多个系数的线性回归公式

幸运的是，我们可以使用 Python 来为我们做所有的计算，特别是`sklearn`包中的线性模型函数。

## 练习 9.10：线性回归

我们的任务是尝试使用葡萄酒的其他特征来预测红葡萄酒的 pH 水平。可以从 GitHub 存储库 https://packt.live/3imVXv5 下载数据集。

注意

这是 UCI 机器学习库（http://archive.ics.uci.edu/ml）提供的葡萄酒质量数据集。尔湾，加利福尼亚：加利福尼亚大学，信息与计算机科学学院。*P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reiss*。通过从理化性质中进行数据挖掘来建模葡萄酒偏好。在决策支持系统中，Elsevier，47(4)：547-553，2009。 

1.  导入我们需要的包并读取数据：

```py
# import packages and read in data
import pandas as pd
import statistics as st
import scipy.stats as sp
import math
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
data = pd.read_csv("winequality-red.csv")
```

1.  将数据子集化为我们需要的两列（我们将尝试使用柠檬酸的量来预测`pH`水平）。将`pH`水平设置为我们的因变量，柠檬酸作为自变量：

```py
data1 = data[['pH','citric acid']]
plt.scatter(x=data1['citric acid'], y=data1['pH'])
y = data1['pH']
x = data1[['citric acid']]
```

1.  拟合线性模型并将数据作为散点图和我们的线性回归模型进行绘制：

```py
model = lm.LinearRegression()
model.fit(x,y)
plt.scatter(x, y,color='g')
plt.plot(x, model.predict(x),color='k')
plt.show()
```

输出如下：

![图 9.21：线性方程似乎很好地适配了我们的数据](img/B15968_09_21.jpg)

图 9.21：线性方程似乎很好地适配了我们的数据

如果你看图片，你会注意到这条线很好地适配了数据。让我们添加另一个自变量；在这种情况下，残留糖的数量，并看看它是否改善了预测。

1.  这一次，我们将柠檬酸和残留糖设置为自变量并拟合模型：

```py
#can we predict the pH of the wine using 
#citric acid and residual sugar?
data2 = data[['pH','citric acid','residual sugar']]
y = data2['pH']
x = data2[['citric acid', 'residual sugar']]
model = lm.LinearRegression()
model.fit(x,y)
y_pred = model.predict(x)
```

1.  创建一个三维散点图并在`3d`空间中绘制线以检查它是否很好地适配我们的数据：

```py
threedee = plt.figure().gca(projection='3d')
threedee.scatter(data2['citric acid'],     data2['residual sugar'],data2['pH'])
threedee.set_xlabel('citric acid')
threedee.set_ylabel('residual sugar')
threedee.set_zlabel('pH')
xline = np.linspace(0, 1, 100)
yline = np.linspace(0, 16, 100)
zline = xline*(-0.429) + yline*(-0.000877)+3.430
threedee.plot3D(xline, yline, zline, 'red')
plt.show()
```

输出如下：

![图 9.22：线性方程似乎不太适配我们的数据](img/B15968_09_22.jpg)

图 9.22：线性方程似乎不太适配我们的数据

如果您看图片，我们的线性模型似乎不如我们拟合的第一个模型那样适合数据。 基于此，残留糖可能不会出现在我们的最终模型中。

注意

要访问此特定部分的源代码，请参阅 https://packt.live/2Anl3ZA。

您也可以在 https://packt.live/3eOmPlv 上在线运行此示例。

## 活动 9.01：标准化测试表现

您被要求描述 2015 年 PISA 测试的结果，并调查互联网基础设施的普及对测试成绩可能产生的影响。

要下载数据集，请转到 GitHub 存储库 https://packt.live/3gi2hCg，下载`pisa_test_scores.csv`文件，并将该文件保存到您的工作目录中。

注意

这个 PISA 测试成绩数据集是基于世界银行提供的数据（https://datacatalog.worldbank.org/dataset/education-statistics）。 世界银行 Edstats。

保存文件后，执行以下操作：

1.  使用置信区间描述学生在阅读、科学和数学方面的典型得分。

1.  使用假设检验，评估互联网基础设施的普及是否会导致更高的测试成绩。

1.  构建一个线性模型，使用阅读和写作成绩来预测数学成绩。

注意

此活动的解决方案可在第 688 页找到。

# 摘要

在本章中，我们研究了大数定律，以及样本均值统计的稳定性如何受样本大小的影响。 通过中心极限定理，我们研究了置信区间和假设检验的理论基础。 置信区间用于描述样本统计数据，如样本均值、样本比例和误差限。 假设检验是通过收集样本的证据来评估两个相反的假设。

下一章将开始您的微积分学习，您将研究瞬时变化率和找到曲线斜率等主题。 在研究完这些内容后，我们将研究积分，即找到曲线下的面积。 最后，我们将使用导数来找到复杂方程和图形的最优值。

NKJ24

VBM37
