# 第十章。继续编码！

在上一章中，我们使用 pygame 在图形环境中构建了一个完整的两人游戏。在这本书的最后一章中，我们将回顾你在这一旅程开始时学到的所有内容，然后探索一些你可以尝试的新编码技能的其他想法。其中许多想法将是游戏，但也有一些想法将涉及 Python 可以使用的其他方式。

# 我们学到了什么以及你的下一步

在这本书的开头，你开始学习关于你的电脑。你学习了如何安装 Python 以及使用不同的免费工具，如文本编辑器、Python 外壳和终端/命令行来运行你的游戏。此外，你还学习了如何导航到你的桌面目录并保存你的工作，以便你可以完成每个项目。接下来的步骤包括以下内容：

+   在你的电脑上导航到其他文件夹和目录

+   学习更多终端/命令提示符命令

我们随后开始了我们的编码之旅，通过创建函数和变量，并使用不同的数据类型。我们创建了一些用于数学运算的函数，然后将这些函数组合起来创建了一个计算器。你学习了如何通过使用`input()`命令给出提示来从某人那里获取信息。

我们使用了`if`和`else`这样的逻辑来教会计算机如何根据用户决定的行为做出决策。我们还使用了循环来帮助我们完成游戏中的不同任务。接下来的步骤将包括以下内容：

+   查找并尝试理解嵌套的`if`语句

+   使用循环处理大量文本或数据集

你学习了在 Python 中使用和存储数据的不同方式，例如字典和列表。了解 Python 中数据是如何存储的很有帮助，Python 的一个最快特性是其存储和检索数据非常快速的能力。

在第一章，“欢迎！让我们开始吧”，和第九章，“迷你网球”中，我们构建了几个项目来展示你如何使用所学的技能。理解如何使用 Python 的强大功能来解决问题非常重要。了解每个工具意味着你可以更好地想象如何使用你的编码技能来解决问题。在本章的剩余部分，让我们看看我们可以解决的一些问题，这将扩展我们的 Python 技能。

# 类和对象——非常重要的下一步！

立即，你需要开始学习关于类和对象的知识。这些是简化可能重复代码的绝佳方式。例如，在 pygame 中有一个名为`Sprites`的类。`pygame.Sprites`模块中的类使得管理不同的游戏对象变得更加容易。

### 注意

要了解更多关于精灵（Sprites）的信息，最好查阅文档：

[`www.pygame.org/docs/tut/SpriteIntro.html`](http://www.pygame.org/docs/tut/SpriteIntro.html)。

要了解更多关于类和对象的信息，一个好的主意是在互联网上搜索诸如面向对象编程（这是 Python 使用的编程类型）以及更具体地说，类和对象等内容。如果你觉得类和对象很困惑，不要担心。这是一个需要一些时间来适应的概念。

### 注意

这里有一些资源可以帮助你了解类和对象：

[`www.tutorialspoint.com/python/python_classes_objects.htm`](http://www.tutorialspoint.com/python/python_classes_objects.htm)

[`www.learnpython.org/en/Classes_and_Objects`](http://www.learnpython.org/en/Classes_and_Objects)

# 游戏中的更多乐趣

由于本书的重点是制作游戏项目，我们将探讨一些更复杂的事情，这些事情在你更深入地了解 pygame 之后可以进行。你可以通过以下方式开始使 Tiny Tennis 变得更加复杂：

+   添加音乐文件

+   添加图形

# 在游戏中添加音乐

pygame 允许你在游戏中添加音乐。有一个音乐模块，允许你将几种格式的音乐添加到游戏文件中。有一些限制，包括文件类型。例如，使用普遍支持的`.ogg`文件类型比使用如`.mp3`这样的文件类型更好，因为后者并不是所有操作系统都原生支持的。

### 注意

如需更多信息，你可以访问 pygame 网站[`www.pygame.org/docs/ref/music.html`](https://www.pygame.org/docs/ref/music.html)，了解如何添加你喜欢的声音。

# 在游戏中添加图形

虽然你已经学会了如何制作一些基本形状，但如果我们的世界只有矩形、圆形和方形以及基本颜色，那将会非常无聊。通过实验使用`pygame.image()`模块等模块，你可以学习如何处理 pygame 之外创建的图像。如果你有一个有艺术天赋的兄弟姐妹或朋友，或者你自己就是艺术家，你可以在电脑上创建或扫描艺术品，然后将其添加到你的游戏中。

### 注意

你可以在[`www.pygame.org/docs/ref/image.html`](http://www.pygame.org/docs/ref/image.html)了解`pygame.image()`模块。

# 重做或设计游戏

如果你想要一个全新的挑战，你可以尝试自己重做一个经典游戏。有很多经典游戏，如 PacMan、Asteroids 或 Zelda 传奇。一个不错的挑战是尝试使用你的技能重做这些游戏的一个版本。这个练习将要求你做一些重要的事情：

+   提前规划你的程序

+   确定你的程序中是否需要类

+   确定如何在程序中使用对象

+   管理程序中的循环

+   管理程序中的`if`/`else`决策

+   在你的程序中管理用户信息，如姓名和分数

一旦你制作了几款基于经典游戏的电子游戏，你可能会有一些自己游戏的灵感。如果你确实有想法，请在你的电脑上的一个文件中记下它们。当你考虑一个新的游戏时，你需要做与重新创建一个经典游戏相同的事情，除了你需要做出关于游戏目的、游戏胜利条件和控制器的其他决定。

# 其他游戏

许多程序员已经用 Python 制作了小型游戏来练习他们的编程技能。首先，你可以查看一些人们已经在 pygame 网站上制作并发布的其他游戏。导航到[`pygame.org/tags/pygame`](http://pygame.org/tags/pygame)，查看一些人们的贡献。

## PB-Ball

PB-Ball 是一款使用 pygame 开发的篮球游戏，它增加了类和对象。当你导航到项目页面时，你会看到一些指向代码的不同链接。以下链接将帮助你找到游戏并查看代码。当你查看代码时，你会注意到有用于图像和声音的文件夹。因此，为了创建一个具有更复杂背景的游戏，你需要学习许多新技能。以下是一些游戏的截图和一些链接，以便你可以查看代码并学习：

### 注意

这是 PB-Ball 游戏的链接：

[`pygame.org/project-PB-Ball-2963-.html`](http://pygame.org/project-PB-Ball-2963-.html)

这里有一个链接到主代码，包括两个类和源代码：

[`bitbucket.org/tjohnson2/pb-ball/src/88e324263a63eb97d6a2427f7ea719df85010dfe/main.py?fileviewer=file-view-default`](https://bitbucket.org/tjohnson2/pb-ball/src/88e324263a63eb97d6a2427f7ea719df85010dfe/main.py?fileviewer=file-view-default)

这里有一些包含游戏所需图像和声音的文件：

[`bitbucket.org/tjohnson2/pb-ball/src`](https://bitbucket.org/tjohnson2/pb-ball/src)

## 蛇

许多人玩过的游戏之一是蛇游戏，玩家开始时是一条短蛇，随着游戏的进行，蛇会变长。保持生存的唯一规则是蛇不能碰到自己的尾巴。互联网上有许多这种游戏的样本。你可以查看一些代码样本，并检查你是否能够重新创建这个游戏。

### 注意

从以下链接了解更多关于蛇游戏的详细信息：

[`programarcadegames.com/python_examples/f.php?file=snake.py`](http://programarcadegames.com/python_examples/f.php?file=snake.py)

[`github.com/YesIndeed/SnakesForPython`](https://github.com/YesIndeed/SnakesForPython)

[`github.com/xtur/SnakesAllAround`](https://github.com/xtur/SnakesAllAround)（这是一个多人游戏！）

除了前面提到的游戏，还有一些程序员非常努力地让 Python 游戏指令对新程序员可用！一些这样的书籍在互联网上免费提供。请参考：

*《使用 Python 进行快速游戏开发》*（Richard Jones 著）[`richard.cgpublisher.com/product/pub.84/prod.11`](http://richard.cgpublisher.com/product/pub.84/prod.11)。

# Python 的其他用途

Python 除了制作游戏之外还有很多用途。学习 Python 可以打开通往数据科学、网络应用开发或软件测试等职业的大门。如果你真的想在计算机编程方面建立职业生涯，那么查看 Python 可以做的不同事情是一个很好的主意。

### 注意

好奇 Python 在现实世界中的应用吗？了解 Python 在许多不同领域的应用！访问[`www.python.org/about/success/`](https://www.python.org/about/success/)获取更多详情。

## SciPy

**SciPy**库有一系列开源（免费）的程序，可用于数学、科学和数据分析。这里将介绍其中的两个程序。尽管一些程序在功能上相当先进，但它们也可以用来做简单的事情。如果你想在数学或科学相关的工作中使用 Python，那么了解这个程序套件是很有价值的。

### 注意

了解所有程序，请访问[`www.scipy.org/`](http://www.scipy.org/)。

## iPython

**iPython**是一个程序，类似于我们用于项目的 Python 外壳，包括**IDLE**或终端。然而，iPython 有一个服务器，使用*笔记本*来跟踪你的代码以及你与代码一起做的其他笔记。该项目正在进行一些积极的改进。

### 注意

了解 iPython 笔记本，请访问[`ipython.org/`](http://ipython.org/)。

Packt Publishing 提供了一本名为*《学习 IPython 进行交互式计算和数据可视化》*（Cyrille Rossant 著，2015 年）的入门书籍，帮助你学习如何使用 iPython：

[`www.packtpub.com/big-data-and-business-intelligence/learning-ipython-interactive-computing-and-data-visualization`](https://www.packtpub.com/big-data-and-business-intelligence/learning-ipython-interactive-computing-and-data-visualization)。

## MatPlotLib

**MatPlotLib**是一个高级工具，可以使用 Python 编写代码来创建简单或复杂的图表、图形、直方图，甚至动画。它是一个开源项目，因此它也是免费的。有许多使用这个工具的方法，这对于任何 2D 可视化特别有用。有关其下载和安装的所有说明都在其网站上。有许多依赖项，但如果你对数学或 2D 图形表示（或两者）感兴趣，那么你应该查看网站和代码示例。

## 树莓派

流行的树莓派是一款专为计算和机器人实验设计的小型计算机板。其操作系统与 Windows 和 Mac 不同，预装了 Python 和 pygame，因此它是开始游戏开发的一种非常方便的方式，因为你不需要做我们在第一章中做过的所有工作。

要使用树莓派，你需要电源、一个带有 HDMI 输入的显示器、一根 HDMI 线、一个键盘和鼠标，如果你打算使用互联网，还需要一个 Wi-Fi 转换器或以太网线。此外，你还需要一张 SD 卡来安装最新的树莓派操作系统。有了这些物品，你可以将树莓派作为你的主要计算机使用并进行实验，知道如果你崩溃了，你只需免费制作另一个操作系统的副本即可！

许多人已经使用树莓派制作游戏，甚至小型便携式游戏系统！除了制作游戏，人们还使用树莓派制作机器人项目和媒体中心项目。关于树莓派的一个非常酷的特点是，你可以学习更多关于构建计算机的知识，并尝试为不同的用途制作计算机。你可以使用 Python 和树莓派编写控制开关、门铃甚至家用电器的代码！你可以访问树莓派的官方网站来了解更多关于其硬件和基于 Linux 的操作系统。

### 注意

访问[`www.raspberrypi.org/`](https://www.raspberrypi.org/)并阅读 Samarth Shah 的《学习树莓派》（*Packt Publishing*，2015 年）和 Tim Cox 的《Python 程序员树莓派食谱》（*Packt Publishing*，2014 年 4 月）以获取更多关于树莓派的信息。

![树莓派](img/B04681_10_04.jpg)

# 编码挑战

除了你可以用 Python 代码做的所有酷炫的事情之外，你可以通过寻找编码挑战并独自或与朋友一起完成它们来练习 Python 编码。这些挑战从简到繁，从易到难，是保持项目之间技能锐利的好方法。编码挑战通常针对每个特定的编码技能，如下所示：

+   打印

+   循环迭代

+   创建变量、字符串和整数

+   数据管理

+   函数

+   `if`/`elif`/`else`

+   嵌套 `if`/`elif`/`else`

+   嵌套逻辑

+   递归

如果你对这些术语完全不熟悉，查查它们，了解更多关于它们的信息，并尝试一些编码挑战来加强你的技能。以下是一些提供 Python 编码挑战的网站：

+   [`codingbat.com/python`](http://codingbat.com/python)

+   [`www.pythonchallenge.com/`](http://www.pythonchallenge.com/)

+   [`usingpython.com/python-programming-challenges/`](http://usingpython.com/python-programming-challenges/)

+   [`wiki.python.org/moin/ProblemSets`](https://wiki.python.org/moin/ProblemSets)

+   [`www.hackerrank.com/login`](https://www.hackerrank.com/login)

你可以在这些链接中找到数百个练习问题！

# 摘要

希望这本书为你提供了对 Python 基本概念的坚实基础。你绝对不是专家，因为 Python 是一种功能强大的语言，它可以做很多一本书无法展示的事情。然而，如果你完成了每个游戏，你将有一个坚实的 Python 基础，可以在此基础上继续前进。

继续使用 Python 的一种方法是继续在挑战和游戏中工作，同时深入研究代码架构、类和对象，以及使用对象、自定义图像、声音和其他效果进行更高级的游戏编程。Python 不用于传统的游戏系统，但游戏设计概念在任何面向对象的语言中都适用。一旦你在 Python 中感到舒适，你可以更容易地转向更常见的游戏设计语言，如 C++。

使用 Python 的另一种方法是更多地了解数据应用以及如何使用 Python 处理不同类型的数据和数学。这是深入了解探索 Python 并创建一份可以展示给中学甚至大学的工作集的极好方式。互联网上有关于各种主题的大量数据集，包括人口和天气等。

最后，你可能决定你想了解使用 Python 构建的 Web 应用程序。如果你选择这样做，你可以查看 GitHub 或 Bitbucket 这样的地方，程序员们在那里保存他们的代码，有时甚至允许免费获取。阅读其他程序员的代码是学习新且有趣的使用代码方式的一种极好方法。此外，寻找并帮助构建免费程序，也称为开源，是帮助社区在编程方面变得更好的极好方式。你还可以提出很好的问题并得到回答。

在你努力写出更好的游戏和更好的代码的征途中，祝你好运！继续学习！
