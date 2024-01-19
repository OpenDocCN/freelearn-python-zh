# 第十六章：学习游戏人工智能-构建一个玩家机器人

- 游戏开发人员的目标是创建具有挑战性和乐趣的游戏。尽管许多程序员尝试过，但许多游戏失败的主要原因是，人类玩家喜欢在游戏中受到人工玩家的挑战。创造这样的人工玩家的结果通常被称为**非玩家角色**（**NPC**）或人工玩家。虽然创建这样的玩家很有趣（只对程序员来说），但除非我们为这些人工玩家注入一些智能，否则它不会为游戏增添任何价值。创建这样的 NPC 并使它们以某种程度的意识和智能（与人类智能相当）与人类玩家互动的过程称为**人工智能**（**AI**）。

在本章中，我们将创建一个*智能系统*，该系统将能够与人类玩家竞争。该系统将足够智能，能够进行类似于人类玩家的移动。系统将能够自行检查碰撞，检查不同的可能移动，并进行最有利的移动。哪种移动是有利的将高度依赖于目标。人工玩家的目标将由程序员明确定义，并且基于该目标，计算机玩家将能够做出智能的移动。例如，在蛇 AI 游戏中，计算机玩家的目标是进行一次移动，使它们更接近蛇食物，而在**第一人称射击**（**FPS**）游戏中，人工玩家的目标是接近人类玩家并开始向人类玩家开火。

通过本章结束时，您将学会如何通过定义机器状态来创建一个人工系统，以定义人工玩家在任何情况下会做什么。同样，我们将以蛇 AI 为例，以说明如何向计算机玩家添加智能。我们将为游戏角色创建不同的实体：玩家、计算机和青蛙（蛇食物），并探索面向对象和模块化编程的强大功能。在本章中，您将主要找到我们已经涵盖的内容，并学会如何有效地使用它以制作有生产力的游戏。

本章将涵盖以下主题：

+   理解人工智能

+   开始蛇 AI

+   添加计算机玩家

+   为计算机玩家添加智能

+   构建游戏和青蛙实体

+   构建表面渲染器和处理程序

+   可能的修改

# 技术要求

为了有效地完成本章，必须获得以下要求清单：

+   Pygame 编辑器（IDLE）-建议使用 3.5+版本

+   PyCharm IDE（参见第一章，*了解 Python-设置 Python 和编辑器*，安装程序）

+   资产（蛇和青蛙`.png`文件）-可在 GitHub 链接获取：[`github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter16`](https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter16)

查看以下视频以查看代码的运行情况：

[`bit.ly/2n79HSP`](http://bit.ly/2n79HSP)

# 理解人工智能

随着众多算法和模型的出现，今天的游戏开发者利用它们来创建人工角色，然后让它们与人类玩家竞争。在现实世界的游戏中，被动地玩游戏并与自己竞争已经不再有趣，因此，程序员故意设置了几种难度和状态，使游戏更具挑战性和乐趣。程序员使用的几种方法中，最好且最流行的之一是让计算机与人类竞争。听起来有趣且复杂吗？问题是如何可能创建这样的算法，使其能够与聪明的人类竞争。答案很简单。作为程序员，我们将定义几种聪明的移动，使计算机能够以与人类类似的方式应对这些情况。

在玩游戏时，人类足够聪明，可以保护他们的游戏角色免受障碍物和失败。因此，在本章中，我们的主要目标是为 NPC 提供这样的技能。我们将使用之前制作的贪吃蛇游戏（第十一章，*使用 Pygame 制作贪吃蛇游戏 UI*），稍微完善一下，并为其添加一个具有一定意识的计算机玩家，它将知道食物（蛇吃的东西）在哪里，以及障碍物在哪里。确切地说，我们将为我们的计算机角色定义不同的移动，使其拥有自己的生活。

首先，回顾一下第四章，*数据结构和函数*。在那一章中，我们创建了一个简单的井字游戏，并在其中嵌入了一个简单的*智能*算法。在那个井字游戏中，我们能够让人类玩家与计算机竞争。我们首先定义了模型，处理了用户事件，然后最终添加了不同的移动，以便计算机自行游戏。我们还测试了游戏，计算机能够在某些情况下击败玩家。因此，我们在第四章，*数据结构和函数*中已经学习了基本的 AI 概念。然而，在本章中，我们将更深入地探索 AI 的世界，并揭示关于*智能算法*的其他有趣内容，这些内容可以添加到我们之前制作的贪吃蛇游戏中。

要了解 AI 算法的工作原理，我们必须对状态机图表有相当多的了解。状态机图表（通常源自*计算理论*）定义了 NPC 在不同情况下必须做什么。我们将在下一个主题中学习状态机图表或动画图表。

# 实现状态。

每个游戏的状态数量都不同，这取决于游戏的复杂程度。例如，在像 FPS 这样的游戏中，NPC 或敌人必须有不同的状态：随机寻找人类玩家，在玩家位置随机生成一定数量的敌人，向人类玩家射击等等。每个状态之间的关系由状态机图表定义。这个图表（不一定是图片）代表了从一个状态到另一个状态的转变。例如，敌人在什么时候应该向人类玩家开火？在什么距离应该生成随机数量的敌人？

以下图表代表不同的状态，以及这些状态何时必须从一个状态改变到另一个状态：

![](img/8dc4199a-500c-48da-ab7b-e1879473d42a.png)

观察前面的图表，你可能会觉得它并不陌生。我们之前在为井字游戏添加智能计算机玩家时做过类似的事情。在图中，我们从随机的敌人移动开始，因为我们不希望每个敌人都在同一个地方渲染。同样，在敌人被渲染后，它们被允许接近人类玩家。敌人的移动没有限制。因此，可以实现敌人位置和人类玩家位置之间的简单条件检查，以执行敌人的矢量移动（第十章，*用海龟升级蛇游戏*）。同样，在每次位置改变后，敌人的位置与人类玩家的位置进行检查，如果它们彼此靠近，那么敌人可以开始朝向人类玩家开火。

在每个状态之间，都有一些步骤的检查，以确保计算机玩家足够智能，可以与人类玩家竞争。我们可以观察到以下伪代码，它代表了前述的机器状态：

```py
#pseudocode for random movement
state.player_movement():
    if state.hits_boundary:
        state.change_movement()
```

在前面的伪代码中，每个状态定义了必须执行的代码，以执行诸如`player_movement`、`hits_boundary`和`change_movements`之类的检查操作。此外，在接近人类玩家的情况下，伪代码看起来像下面这样：

```py
#pseudocode for check if human player and computer are near
if state.player == "explore":
    if human(x, y) == computer(x, y):
        state.fire_player()
    else:
        state.player_movement()
```

前面的伪代码并不是实际代码，但它为我们提供了关于我们可以期望 AI 为我们做什么的蓝图。在下一个主题中，我们将看到如何利用伪代码和状态机的知识，为我们的蛇游戏创建不同的实体。

# 开始蛇 AI

如在 FPS 的情况下讨论的那样，蛇 AI 的情况下可以使用类似的机器状态。在蛇 AI 游戏中，我们的计算机玩家需要考虑的两个重要状态如下：

+   计算机玩家有哪些有效的移动？

+   从一个状态转换到另一个状态的关键阶段是什么？

关于前面的几点，第一点指出，每当计算机玩家接近边界线或墙壁时，必须改变计算机玩家的移动（确保它保持在边界线内），以便计算机玩家可以与人类玩家竞争。其次，我们必须为计算机蛇玩家定义一个目标。在 FPS 的情况下，如前所述，计算机敌人的主要目标是找到人类玩家并执行*射击*操作，但是在蛇 AI 中，计算机玩家必须接近游戏中的食物。蛇 AI 中真正的竞争在于人类和计算机玩家谁能更快地吃到食物。

现在我们知道了必须为 NPC（计算机玩家）定义的动作，我们可以为游戏定义实体。与我们在第十一章中所做的类似，*使用 Pygame 制作 Outdo Turtle - 蛇游戏 UI*，我们的蛇 AI 有三个主要实体，它们列举如下：

+   **类**`Player`：它代表人类玩家，所有动作都与人类相关——事件处理、渲染和移动。

+   **类**`Computer`：它代表计算机玩家（一种 AI 形式）。它执行诸如更新位置和更新目标之类的动作。

+   **类**`Frog`：它代表游戏中的食物。人类和计算机之间的竞争目标是尽快接近青蛙。

除了这三个主要的游戏实体之外，还有两个剩余的游戏实体来定义外围任务，它们如下：

+   **类**`Collision`：它代表将具有方法以检查任何实体（玩家或计算机）是否与边界发生碰撞。

+   **类**`App`：它代表将渲染显示屏并检查任何实体是否吃掉青蛙的类。

现在，借助这些实体蓝图，我们可以开始编码。我们将首先添加一个`Player`类，以及可以渲染玩家并处理其移动的方法。打开你的 PyCharm 编辑器，在其中创建一个新的项目文件夹，然后在其中添加一个新的 Python 文件，并将以下代码添加到其中：

```py
from pygame.locals import *
from random import randint
import pygame
import time
from operator import *
```

在前面的代码中，每个模块对你来说都很熟悉，除了`operator`。在编写程序时（特别是在检查游戏实体与边界墙之间的碰撞时），使用数学函数来执行操作比直接使用数学运算符要非常有帮助。例如，如果要检查`if value >= 2`，我们可以通过使用`operator`模块内定义的函数来执行相同的操作。在这种情况下，我们可以调用`ge`方法，它表示*大于等于*：`if ge(value, 2)`。类似于`ge`方法，我们可以调用诸如以下的不同方法：

+   `gt(a, b)`: 检查 a > b—如果 a > b 则返回`True`；否则返回`False`

+   `lt(a, b)`**:** 检查 a < b—如果 a < b 则返回`True`；否则返回`False`

+   `le(a, b)`: 检查 a <= b—如果 a <= b 则返回`True`；否则返回`False`

+   `eq(a, b)`: 检查 a == b—如果 a == b 则返回`True`；否则返回`False`

现在你已经导入了必要的模块，让我们开始有趣的事情，创建`Player`类：

```py
class Player:
    x = [0] #x-position
    y = [0] #y-position
    size = 44 #step size must be same for Player, Computer, Food
  direction = 0 #to track which direction snake is moving
  length = 3 #initial length of snake    MaxMoveAllow = 2
  updateMove = 0    def __init__(self, length):
        self.length = length
        for i in range(0, 1800):
            self.x.append(-100)
            self.y.append(-100)

        # at first rendering no collision
  self.x[0] = 1 * 44
  self.x[0] = 2 * 44
```

在前面的代码中，我们开始定义类属性：(`x`，`y`)代表蛇的初始位置，`size`代表蛇块的步长，`direction`（值范围从 0 到 4）代表蛇移动的当前方向，`length`是蛇的原始长度。名为`direction`的属性的值将在 0 到 3 之间变化，其中 0 表示蛇向*右*移动，1 表示蛇向*左*移动，类似地，2 和 3 分别表示*上*和*下*方向。

接下来的两个类属性是`MaxMoveAllow`和`update`。这两个属性将在名为`updateMove`的函数中使用（在下面的代码中显示），它们确保玩家不被允许使蛇移动超过两次。玩家可能会一次输入多于两个箭头键，但如果所有效果或箭头键同时反映，蛇将移动不协调。为了避免这种情况，我们定义了`maxMoveAllowed`变量，以确保最多同时处理两次箭头键按下。

同样地，我们在类内部定义了构造函数，用于执行类属性的初始化。在渲染蛇玩家在随机位置之后（通过`for`循环完成），我们编写了一条语句，确保在游戏开始时没有碰撞（高亮部分）。代码暗示了蛇的每个方块之间的位置必须相隔三个单位。如果将`self.x[0] = 2*44`的值更改为`self.x[0] = 1 *44`，那么蛇头和其之间将发生碰撞。因此，为了确保在游戏开始时（玩家开始玩之前）没有碰撞，我们必须在方块之间提供特定的位置间隔。

现在，让我们使用`MaxMoveAllow`和`updateMove`属性来创建`update`函数：

```py
def update(self):

    self.updateMove = self.updateMove + 1
  if gt(self.updateMove, self.MaxAllowedMove):

        # update previous to new position
  for i in range(self.length - 1, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]

        # updating the position of snake by size of block (44)
  if self.direction == 0:
            self.x[0] = self.x[0] + self.size
        if self.direction == 1:
            self.x[0] = self.x[0] - self.size
        if self.direction == 2:
            self.y[0] = self.y[0] - self.size
        if self.direction == 3:
            self.y[0] = self.y[0] + self.size

        self.updateMove = 0
```

前面的代码对你来说并不陌生。你以前多次见过这样的逻辑（在第六章，“面向对象编程”中，以及第十一章，“用 Pygame 制作贪吃蛇游戏 UI”中，处理蛇的位置时）。简而言之，前面的代码行改变了人类玩家的当前位置，根据按下的箭头键。你可以在代码中看到，我们还没有处理任何箭头键（我们将在`App`类中处理），但我们已经创建了一个名为`direction`的属性，它可以跟踪哪个键被按下。如果`direction`等于`0`，这意味着右箭头键被按下，因此我们增加*x*位置与块大小。

同样，如果`direction`是`1`，我们通过减去块大小`44`来改变*x*位置值，这意味着蛇将朝负*x*轴移动。（这不是新信息；可以在第九章，“数据模型实现”中找到详细讨论。）

现在，为了确保每个`direction`属性与值 0 到 3 相关联，我们将为每个创建函数，如下所示：

```py
def moveRight(self):
    self.direction = 0   def moveLeft(self):
    self.direction = 1   def moveUp(self):
    self.direction = 2   def moveDown(self):
    self.direction = 3   def draw(self, surface, image):
 for item in range(0, self.length):
 surface.blit(image, (self.x[item], self.y[item]))
```

观察前面的代码，你可能已经注意到`direction`属性的重要性。每个移动都有一个相关联的值，可以在处理用户事件时使用`pygame`模块（我们将在本章后面讨论）。但是，现在只需看一下`draw`函数，它接受蛇（人类玩家）的`surface`和`image`作为参数，并相应地进行 blits。你可能会有这样的问题：为什么不使用传统方法（自第八章，“Turtle Class – Drawing on the Screen”以来一直在使用的方法）来处理用户事件，而是使用`direction`属性？这个问题是合理的，显然你也可以以这种方式做，但在 Snake AI 的情况下，实施这样的代码存在重大缺点。由于 Snake AI 有两个主要玩家或游戏实体（人类和计算机），它们每个都必须有独立的移动。因此，对每个实体使用传统方法处理事件将会很繁琐和冗长。更好的选择是使用一个属性来跟踪哪个键被按下，并为每个玩家独特地处理它，这正是我们将要做的，使用`direction`属性。

现在我们已经完成了主要的人类玩家，我们将转向计算机玩家。我们将开始为`Computers`类编写代码，它将在下一个主题中处理计算机的移动。

# 添加计算机玩家

最后，我们来到了本章的主要部分——重点部分——将计算机蛇角色添加到游戏中变得更容易。与外观一样，计算机的移动处理技术必须类似于人类玩家。我们可以重用`Player`类中编写的代码。唯一不同的是`Player`类的*目标*。对于人类玩家，目标未定义，因为移动的目标由玩家的思想实现。例如，人类玩家可以通过控制蛇的移动方向来有效地玩游戏。如果蛇食物在左边，那么人类玩家不会按右箭头键，使蛇朝相反方向移动。但是，计算机不够聪明，无法自行考虑赢得游戏的最佳方式。因此，我们必须明确指定计算机玩家的目标。为个别玩家/系统指定目标的技术将导致智能系统，并且其应用范围广泛——从游戏到机器人。

目前，让我们复制写在`Player`类内部的代码，并将其添加到名为`Computer`的新类中。以下代码表示了`Computer`类的创建，以及它的构造函数：

```py
class Computer:
    x = [0]
    y = [0]
    size = 44 #size of each block of snake
  direction = 0
  length = 3    MaxAllowedMove = 2
  updateMove = 0    def __init__(self, length):
        self.length = length
        for item in range(0, 1800):
            self.x.append(-100)
            self.y.append(-100)

      # making sure no collision with player
  self.x[0] = 1 * 44
  self.y[0] = 4 * **44** 
```

与`Player`类类似，它有四个属性，其中`direction`的初始值为`0`，这意味着在计算机实际开始玩之前，蛇将自动向右（正*x*轴）方向移动。此外，构造函数中初始化的所有内容都与`Player`类相似，除了代码的突出部分。代码的最后一行是`y[0]`，它从`4*44`开始。回想一下在人类玩家的情况下，代码的相同部分是`2*44`，表示列位置。编写这段代码，我们暗示游戏开始时人类玩家蛇和计算机玩家蛇之间不应该发生碰撞。但是，`x[0]`的值是相同的，因为我们希望每条蛇都从同一行开始，但不在同一列。通过这样做，我们避免了它们的碰撞，并且每个玩家的蛇将被正确渲染。

同样，我们必须添加`update`方法，它将根据`direction`属性反映计算机蛇的*x*、*y*位置的变化。以下代码表示了`update`方法，它将确保计算机蛇只能同时使用两个箭头键移动的组合：

```py
def update(self):

    self.updateMove = self.updateMove + 1
  if gt(self.updateMove, self.MaxAllowedMove):

        # Previous position changes one by one
  for i in range(self.length - 1, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]

        # head position change
  if self.direction == 0:
            self.x[0] = self.x[0] + self.size
        if self.direction == 1:
            self.x[0] = self.x[0] - self.size
        if self.direction == 2:
            self.y[0] = self.y[0] - self.size
        if self.direction == 3:
            self.y[0] = self.y[0] + self.size

        self.updateMove = 0
```

前面的代码与`Player`类类似，所以我不会费心解释它。您可以参考`Player`类的`update`函数，了解这个方法是如何工作的。与`Player`类类似，我们必须添加四个方法，这些方法将相应地改变`direction`变量的值：

```py
def moveRight(self):
    self.direction = 0   def moveLeft(self):
    self.direction = 1   def moveUp(self):
    self.direction = 2   def moveDown(self):
    self.direction = 3
```

编写的代码将能够更新计算机玩家的*direction*，但这还不足以做出聪明的移动。比如，如果蛇食在右侧，到目前为止编写的代码将无法跟踪食物的位置，因此计算机蛇可能会去相反的地方。因此，我们必须明确指定计算机玩家将朝着靠近蛇食的位置移动。我们将在下一个主题中介绍这一点。

# 为计算机玩家添加智能

到目前为止，已经定义了两个游戏实体，它们都处理玩家的移动。与`Player`类不同，另一个游戏实体（计算机玩家）不会自行决定下一步的移动。因此，我们必须明确要求计算机玩家做出一步将蛇靠近食物的移动。通过这样做，计算机玩家和人类玩家之间将会有巨大的竞争。这看起来实现起来相当复杂；然而，这个想法仍然保持不变，正如之前讨论的那样，以及机器状态图。

通过机器状态图，AI 玩家必须考虑两件事：

+   检查蛇食的位置，并采取行动以靠近它。

+   检查蛇的当前位置，并确保它不会撞到边界墙。

第一步将实现如下：

```py
def target(self, food_x, food_y):
    if gt(self.x[0] , food_x):

        self.moveLeft()

    if lt(self.x[0] , food_x):
        self.moveRight()

    if self.x[0] == food_x:
        if lt(self.y[0] , food_y):
            self.moveDown()

        if gt(self.y[0] , food_y):
            self.moveUp()

def draw(self, surface, image):
     for item in range(0, self.length):
         surface.blit(image, (self.x[item], self.y[item]))
```

在上一行代码中，我们调用了不同的先前创建的方法，如`moveLeft()`，`moveRight()`等。这些方法将导致蛇根据`direction`属性值移动。`target()`方法接受两个参数：`food_x`和`food_y`，它们组合地指代蛇食物的位置。操作符`gt`和`lt`用于执行与蛇的*x*-head 和*y*-head 位置的比较操作。例如，如果蛇食物在负*x*-轴上，那么将对蛇的*x*-位置和食物的*x*-位置进行比较（`gt(self.x[0], food_x)`）。显然，`food_x`在负*x*-轴上，这意味着蛇的*x*-位置更大，因此调用`moveLeft()`。正如方法的签名所暗示的，我们将转向，并将计算机玩家蛇朝着负*x*-轴移动。对食物的每个(*x*, *y*)位置进行类似的比较，每次调用不同的方法，以便我们可以引导计算机玩家朝着蛇食物移动。

现在我们已经添加了简单的计算机玩家，它能够通过多个障碍物，让我们在下一个主题中添加`Frog`和`Collision`类。`Frog`类负责在屏幕上随机位置渲染青蛙（蛇的食物），`Collision`将检查蛇之间是否发生碰撞，或者蛇与边界墙之间是否发生碰撞。

# 构建游戏和青蛙实体

如前所述，我们将在本主题中向我们的代码中添加另外两个类。这些类在我们的 Snake AI 中有不同的用途。`Game`实体将通过检查传递给它们的成员方法的参数来检查是否发生任何碰撞。对于`Game`实体，我们将定义一个简单但强大的方法，名为`checkCollision()`，它将根据碰撞返回`True`或`False`的布尔值。

以下代码表示`Game`类及其成员方法：

```py
class Game:
    def checkCollision(self, x1, y1, x2, y2, blockSize):
        if ge(x1 , x2) and le(x1 , x2 + blockSize):
            if ge(y1 , y2) and le(y1, y2 + blockSize):
                return True
 return False 
```

对`checkCollision()`方法的调用将在主类中进行（稍后将定义）。但是，你会注意到传递的参数（*x*和*y*值）将是蛇的当前位置，从中调用此方法。假设你创建了`Game`类的一个实例，并传递了人类玩家的（`x1`，`y1`，`x2`和`y2`）位置值。这样做，你就是在为人类玩家调用`checkCollision`方法。条件语句将检查蛇的位置值是否与边界墙相同。如果是，它将返回`True`；否则，它将返回`False`。

接下来重要的游戏实体是`Frog`。这个类在随机位置渲染`Frog`的图像，每次被任何玩家（人类或计算机）吃掉后都会重新渲染。以下代码表示了`Frog`类的声明：

```py
class Frog:
    x = 0
  y = 0
  size = 44    def __init__(self, x, y):
        self.x = x * self.size
        self.y = y * self.size

    def draw(self, surface, image):
        surface.blit(image, (self.x, self.y))
```

在上述代码中，我们定义了*x*-位置、*y*-位置和`draw`方法，以便渲染青蛙图像。通过创建`Frog`类来调用这个方法。

在下一个主题中，我们将通过创建和实现最后一个实体：主`App`实体来完成我们的程序。这将是我们游戏的中央指挥官。

# 构建表面渲染器和处理程序

首先，让我们回顾一下我们到目前为止所做的事情。我们开始编写代码，定义了两个主要的游戏实体：`Player`和`Computer`。这两个实体在行为和渲染方法方面都非常相似，只是在`Computer`类中引入了额外的`target()`方法，以确保计算机玩家足够聪明，能够与人类玩家竞争。同样，我们声明了另外两个实体：`Game`和`Frog`。这两个类为贪吃蛇 AI 提供了后端功能，比如添加碰撞逻辑，以及检查蛇食物应该渲染的位置。我们在这些不同的实体中创建了多个方法，但我们从未创建过实例/对象。这些实例可以从主要的单一类中创建，我们现在要实现这个类。我将称这个类为`App`类。

看一下以下代码片段，以便为`App`类编写代码：

```py
class App:
    Width = 800 #window dimension
  Height = 600
  player = 0 #to track either human or computer 
 Frog = 0 #food    def __init__(self):
        self._running = True
  self.surface = None
  self._image_surf = None
  self._Frog_surf = None
  self.game = Game()
        self.player = Player(5) #instance of Player with length 5 (5            
        blocks) 
        self.Frog = Frog(8, 5) #instance of Frog with x and y position
        self.computer = Computer(5) #instance of Computer player with 
        length 5 
```

前面的代码定义了一些属性，比如游戏控制台的`Height`和`Width`。同样，它有一个构造函数，用于初始化不同的类属性，以及创建`Player`、`Frog`和`Computer`实例。

接下来，要从计算机加载图像并将其添加到 Python 项目中（参考第十一章，*使用 Pygame 创建 Outdo Turtle-贪吃蛇游戏 UI*，了解更多关于`load`方法的信息）。游戏的资源，比如蛇身和食物，可以在这个 GitHub 链接上找到：[`github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter16`](https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter16)。但是，你也可以自己创建并进行实验。我之前在第十一章，*使用 Pygame 创建 Outdo Turtle-贪吃蛇游戏 UI*中教过你如何使用 GIMP 和简单的绘图应用程序创建透明精灵。试着回顾一下这些概念，并自己尝试一下。现在，我要将两个图像加载到 Python 项目中。

最好使用.png 文件作为精灵，并且不要在文件名中包含数字值。例如，名为`snake12.png`的蛇身文件名是无效的。文件名应该不包含数字值。同样，确保将这些`.png`文件添加到 Python 项目文件夹中。回顾一下第十一章，*使用 Pygame 创建 Outdo Turtle-贪吃蛇游戏 UI*，查看如何将图像加载到 Python 项目中。

以下代码将加载两个图像文件到 Python 项目中：

```py
def loader(self):
 pygame.init()
 self.surface = pygame.display.set_mode((self.Width, self.Height), 
 pygame.HWSURFACE)

 self._running = True
  self._image_surf = pygame.image.load("snake.png").convert()
 self._Frog_surf = pygame.image.load("frog-main.png").convert()
```

在前面的代码行中，我们使用`pygame.display`模块创建了一个`surface`对象。然后，我们将两个图像——`snake.png`和`frog-main.png`——加载到 Python 项目中。`convert()`方法将改变渲染对象的像素格式，使其在任何表面上都能完美工作。

同样，如果一个游戏有事件，并且与用户交互，那么必须实现`on_event`方法：

```py
def on_event(self, event):
 if event.type == QUIT:
 self._running = False
def on_cleanup(self):
    pygame.quit() 
```

最后，让我们定义`main`函数：

```py
def main(self):
    self.computer.target(self.Frog.x, self.Frog.y)
    self.player.update()
    self.computer.update()
```

在前面的函数中，我们调用了`target`方法，以确保计算机玩家能够使用其中定义的功能。如前所述，`target()`方法接受食物的*x*、*y*坐标，计算机会决定靠近食物。同样，调用了`Player`和`Computer`类的`update`方法。

现在让我们定义`renderer()`方法。这个方法将把蛇和食物绘制到游戏表面上。这是使用`pygame`和`draw`模块完成的：

```py
def renderer(self):
    self.surface.fill((0, 0, 0))
    self.player.draw(self.surface, self._image_surf)
    self.Frog.draw(self.surface, self._Frog_surf)
    self.computer.draw(self.surface, self._image_surf)
    pygame.display.flip()
```

如果你觉得你不理解`renderer()`方法的工作原理，去第十一章，*使用 Pygame 创建 Outdo Turtle-贪吃蛇游戏 UI*。简而言之，这个方法将不同的对象（`image_surf`和`Frog_surf`）绘制到游戏屏幕上。

最后，让我们创建一个`handler`方法。这个方法将处理用户事件。根据用户按下的箭头键，将调用不同的方法，比如`moveUp()`、`moveDown()`、`moveLeft()`和`moveRight()`。这四个方法都在`Player`和`Computer`实体中创建。以下代码定义了`handler`方法：

```py
def handler(self):
    if self.loader() == False:
        self._running = False   while (self._running):
        keys = pygame.key.get_pressed()

        if (keys[K_RIGHT]):
            self.player.moveRight()

        if (keys[K_LEFT]):
            self.player.moveLeft()

        if (keys[K_UP]):
            self.player.moveUp()

        if (keys[K_DOWN]):
            self.player.moveDown()     self.main()
        self.renderer()

        time.sleep(50.0 / 1000.0);
```

前面的`handler`方法已经被创建了很多次（我们看到了高级和简单的方法），这个是最简单的一个。我们使用了`pygame`模块来监听传入的按键事件，并根据需要处理它们，通过调用不同的方法。例如，当用户按下向下箭头键时，就会调用`moveDown()`方法。最后的`sleep`方法将嵌入计时器，以便在两次连续的按键事件之间有所区别。

最后，让我们调用这个`handler`方法：

```py
if __name__ == "__main__":
    main = App()
    main.handler()
```

让我们运行游戏并观察输出：

![](img/d6ac8fdd-419c-4f70-911f-1a8a03bde72f.png)

正如预期的那样，这个游戏还需要添加一些东西，包括：当人类玩家和电脑玩家吃到食物时会发生什么，以及蛇与自身碰撞时会发生什么？如果你一直正确地跟随本书，这对你来说应该是小菜一碟。我们已经多次添加了相同的逻辑（在第七章，*列表推导和属性*；第十章，*用海龟升级蛇游戏*；和第十一章，*用 Pygame 超越海龟-蛇游戏 UI*）。但除了这个逻辑，还要关注两条相似的蛇：一条必须根据人类玩家的行动移动，另一条则独立移动。计算机蛇知道与边界墙的碰撞和食物的位置。一旦你运行游戏，计算机玩家将立即做出反应，并试图做出聪明的移动，早于人类玩家。这就是在现实游戏行业中应用人工智能。虽然你可能认为蛇 AI 示例更简单，但在现实世界中，AI 也是关于机器独立行动，无论算法有多复杂。

但是，游戏中必须进行一些调整，这将在下一个主题“可能的修改”中进行讨论。

# 游戏测试和可能的修改

首先，我建议你回头观察我们定义`Game`类的部分。我们在其中定义了`checkCollision()`方法。这个方法可以用于多种目的：首先，检查玩家是否与蛇食物发生碰撞；其次，检查玩家是否与边界墙发生碰撞。这个时候你一定会有一个“恍然大悟”的时刻。第七章，*列表推导和属性*，到第十一章，*用 Pygame 超越海龟-蛇游戏 UI*，都是关于使用这种技术来实现碰撞原理的，即*如果食物对象的（x，y）位置与任何玩家的（x，y）坐标相同，则称为发生碰撞*。

让我们添加代码来检查任何玩家是否与食物发生了碰撞：

```py
# Does human player snake eats Frog for i in range(0, self.player.length):
    if self.game.checkCollision(self.Frog.x, self.Frog.y, 
    self.player.x[i], self.player.y[i], 44):
        #after each player eats frog; next frog should be spawn in next  
        position
        self.Frog.x = randint(2, 9) * 44
  self.Frog.y = randint(2, 9) * 44
  self.player.length = self.player.length + 1   # Does computer player eats Frog for i in range(0, self.player.length):
    if self.game.checkCollision(self.Frog.x, self.Frog.y, 
        self.computer.x[i], self.computer.y[i], 44):
        self.Frog.x = randint(2, 9) * 44
  self.Frog.y = randint(2, 9) * 44
```

同样，让我们使用相同的函数来检查人类玩家的蛇是否撞到了边界墙。你可能认为在计算机玩家的情况下也需要检查这一点，但这是没有意义的，因为在`Computer`类中定义的`target`方法不会让这种情况发生。换句话说，计算机玩家永远不会撞到边界墙，因此检查是否发生碰撞是没有意义的。但是，在人类玩家的情况下，我们将使用以下代码进行检查：

```py
# To check if the human player snake collides with its own body for i in range(2, self.player.length):
    if self.game.checkCollision(self.player.x[0], self.player.y[0], 
   self.player.x[i], self.player.y[i], 40):
        print("You lose!")
        exit(0)
 pass
```

我们将在这里结束这个话题，但是您可以通过添加一个游戏结束屏幕使这个游戏更具吸引力，我们已经学会了如何使用`pygame`在第十一章中创建。您可以创建一个表面并在其中渲染一个带有标签的字体，以创建这样一个游戏结束屏幕，而不是最后的`pass`语句。

但是，在结束本章之前，让我们来看看我们游戏的最终输出：

![](img/2c8da6fd-a322-4771-9da9-f4c1773a3e55.png)

在游戏中您可能注意到的另一件事是，计算机玩家的蛇*长度*是恒定的，即使它吃了食物。我故意这样做，以免我的游戏屏幕被污染太多。但是，如果您想增加计算机玩家的蛇长度（每次蛇吃食物时），您可以在计算机玩家蛇吃青蛙后添加一个语句：

```py
self.computer.length = self.computer.length + 1
```

最后，我们来到了本章的结束。我们学到了不同的东西，也复习了旧知识。与人工智能相关的概念是广泛的；我们只是尝试触及表面。您可以通过访问以下网址找到使用 Python 在游戏中的其他 AI 含义：[`www.pygame.org/tags/ai`](https://www.pygame.org/tags/ai)。

# 总结

在本章中，我们探讨了在游戏中实现 AI 的基本方法。然而，AI 的工作方式在很大程度上取决于奖励智能系统的每一步。我们使用了机器状态图来定义计算机玩家的可能状态，并用它来执行每个实体的不同动作。在这一章中，我们采用了不同的编程范式；事实上，这是对我们迄今为止学到的一切的回顾，另外还使用了智能算法来处理 NPC。

对于每个定义的实体，我们都创建了一个类，并采用了基于属性和方法的封装和模型的面向对象范式。此外，我们定义了不同的类，如`Frog`和`Game`，以实现碰撞的逻辑。为了实现单一逻辑，我们为每个游戏实体（`Player`和`Computer`）创建了单独的类。您可以将其理解为多重继承。本书的主要目的是让读者了解如何使用 Python 创建游戏机器人。此外，某种程度上，目的是在单一章节中复习我们在整本书中学到的所有编程范式。

正如古谚所说：*已知乃一滴，未知则是海洋*。我希望您仍然渴望更多地了解 Python。我建议您加强基本的编程技能并经常进行实验，这将确实帮助您实现成为游戏开发人员的梦想工作。游戏行业是巨大的，掌握 Python 知识将会产生巨大影响。Python 是一种美丽的语言，因此您将受到更深入学习的激励，而这本书将是您迈向成为 Python 专家的第一步。
