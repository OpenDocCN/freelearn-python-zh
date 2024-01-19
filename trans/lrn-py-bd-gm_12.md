# 学习角色动画、碰撞和移动

*动画是一门艺术*。这引发了关于如何通过为每个角色添加纹理或皮肤，或者通过保持无可挑剔的图形用户界面来创建模拟人物或物体的物理行为的虚拟世界的问题。在创建动画时，我们不需要了解控制器或物理设备的工作原理，但动画是物理设备和游戏角色之间的媒介。动画通过在图像视图中以适当的阴影和动作引导玩家，因此它是一门艺术。作为程序员，我们负责游戏角色在特定方向移动的位置和原因，而动画师负责它们的外观和动作。

在 Python 的`pygame`模块中，我们可以使用精灵来创建动画和碰撞-这是大型图形场景的一部分的二维图像。也许我们可以自己制作一个，或者从互联网上下载一个。在使用 pygame 加载这样的精灵之后，我们将学习构建游戏的两个基本模块：处理用户事件和构建动画逻辑。动画逻辑是一个简单而强大的逻辑，它使精灵或图像在用户事件控制下朝特定方向移动。

通过本章，您将熟悉游戏控制器的概念以及使用它为游戏角色创建动画的方法。除此之外，您还将了解有关碰撞原理以及使用 pygame 掩模方法处理碰撞的方法。不仅如此，您还将学习处理游戏角色的移动方式，如跳跃、轻拍和滚动，同时制作类似 flappy bird 的游戏。

在本章中，我们将涵盖以下主题：

+   游戏动画概述

+   滚动背景和角色动画

+   随机对象生成

+   检测碰撞

+   得分和结束屏幕

+   游戏测试

# 技术要求

您需要以下要求清单才能完成本章：

+   Pygame 编辑器（IDLE）版本 3.5 或更高。

+   Pycharm IDE（参考第一章，*了解 Python-设置 Python 和编辑器*，进行安装程序）。

+   Flappy Bird 游戏的代码资产和精灵可在本书的 GitHub 存储库中找到：[`github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter12`](https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter12)

观看以下视频，查看代码的运行情况：

[`bit.ly/2oKQQxC`](http://bit.ly/2oKQQxC)

# 了解游戏动画

就像你在电脑游戏中看到的一切一样，动画模仿现实世界，或者试图创造一个让玩家感觉自己正在与之交互的世界。用二维精灵绘制游戏相当简单，就像我们在上一章中为贪吃蛇游戏制作角色时所看到的那样。即使是二维角色，我们也可以通过适当的阴影和动作创建三维运动。使用`pygame`模块可以更容易地为单个对象创建动画；我们在上一章中看到了一点实际操作，当时我们为贪吃蛇游戏创建了一个简单的动画。在本节中，我们将使用`pygame`模块为多个对象创建动画。我们将制作一个简单的程序，用于创建下雪的动画。首先，我们将使用一些形状填充雪花（在此程序中，我们使用的是圆形几何形状，但您可以选择任何形状），然后创建一些动画逻辑，使雪花在环境中移动。

在编写代码之前，确保你进行了一些头脑风暴。由于在上一章中我们编写了一些高级逻辑，所以这一部分对你来说可能更容易，但是确保你也学习了我们在这里做的事情，因为对接下来的部分非常有用，我们将开始制作 Flappy Bird 游戏的克隆版本。

正如我们所知，雪花动画需要一个位置（*x*，*y*）来渲染雪花。这个位置可以任意选择，因此你可以使用随机模块来选择这样的位置。以下代码展示了如何使用`pygame`模块在随机位置绘制任何形状。由于使用了`for`循环进行迭代，我们将使用它来创建一个迭代的范围，最多进行 50 次调用（`eachSnow`的值从 0 到 49）。回想一下前一章，你学习了如何使用 pygame 的`draw`模块将任何形状绘制到屏幕上。考虑到这一点，让我们看看以下代码：

```py
#creates snow 
for eachSnow in range(50):
     x_pos = random.randrange(0, 500)
     y_pos = random.randrange(0, 500)
     pygame.draw.circle(displayScreen, (255,255,255) , [x_pos, y_pos], 2) #size:2
```

想象一下，我们使用了前面的代码来制作动画，这将绘制圆形雪花。运行后，你会发现输出中有些奇怪的地方。你可能已经猜到了，但让我为你解释一下。前面的代码制作了一个圆圈——在某个随机位置——并且先前制作的圆圈在新圆圈创建时立即消失。我们希望我们的代码生成多个雪花，并确保先前制作的圆圈位于右侧位置而不是消失。你发现前面的代码有点 bug 吗？既然你知道了错误的原因，花点时间考虑如何解决这个错误。你可能会想到一个普遍的想法，那就是使用数据结构来解决这个问题。我倾向于使用列表。让我们对前面的代码进行一些修改：

```py
for eachSnow in range(50):
     x_pos = random.randrange(0, 500)
     y_pos = random.randrange(0, 500)
     snowArray.append([x_pos, y_pos])
```

现在，在`snowArray`列表中，我们已经添加了随机创建的雪的位置，即*x*和*y*。对于雪的多个`x_pos`和`y_pos`值，将形成一个嵌套列表。例如，一个列表可能看起来像`[[20,40],[40,30],[30,33]]`，表示随机制作的三个圆形雪花。

对于使用前面的`for`循环创建的每一片雪花，你必须使用另一个循环进行渲染。获取`snow_list`变量的长度可能会有所帮助，因为这将给我们一个关于应该绘制多少雪花的想法。对于由`snow_list`指示的位置数量，我们可以使用`pygame.draw`模块绘制任何形状，如下所示：

```py
for eachSnow in range(len(snowArray)):
 # Draw the snow flake
     pygame.draw.circle(displayScreen, (255,255,255) , snowArray[i], 2)
```

你能看到使用`pygame`模块绘制图形有多容易吗？即使这对你来说并不陌生，这个概念很快就会派上用场。接下来，我们将看看如何让雪花向下飘落。按照以下步骤创建圆形雪花的向下运动：

1.  首先，你必须让雪向下移动一个单位像素。你只需要对`snowArray`元素的`y_pos`坐标进行更改，如下所示：

```py
      color_WHITE = (255, 255, 255)
      for eachSnow in range(len(snowArray)):

       # Draw the snow flake
       pygame.draw.circle(displayScreen, color_WHITE, snow_Array[i], 2)

       # moving snow one step or pixel below
       snowArray[i][1] += 1
```

1.  其次，你必须确保，无论何时雪花消失在视野之外，都会不断地创建。在*步骤 1*中，我们已经为圆形雪花创建了向下运动。在某个时候，它将与较低的水平边界相撞。如果它碰到了这个边界，你必须将它重置，以便从顶部重新渲染。通过添加以下代码，圆形雪花将在屏幕顶部使用随机库进行渲染：

```py
      if snowArray[i][1] > 500:
      # Reset it just above the top
      y_pos = random.randrange(-50, -10)
      snowArray[i][1] = y_pos
      # Give it a new x position
      x_pos = random.randrange(0, 500)
      snowArray[i][0] = y_pos
```

这个动画的完整代码如下（带有注释的代码是不言自明的）：

1.  首先，我们编写的前面的代码需要重新定义和重构，以使代码看起来更好。让我们从初始化开始：

```py
      import pygame as p
      import random as r

      # Initialize the pygame
      p.init()

      color_code_black = [0, 0, 0]
      color_code_white = [255, 255, 255]

      # Set the height and width of the screen
      DISPLAY = [500, 500]

      WINDOW = p.display.set_mode(DISPLAY)

      # Create an empty list to store position of snow
      snowArray = []
```

1.  现在，在初始化的下面添加你的`for`循环：

```py
      # Loop 50 times and add a snow flake in a random x,y position
      for eachSnow in range(50):
          x_pos = r.randrange(0, 500)
          y_pos = r.randrange(0, 500)
          snowArray.append([x_pos, y_pos])

          objectClock = game.time.Clock()
```

1.  类似地，我们将通过创建主循环来结束逻辑，该循环将一直循环，直到用户显式点击关闭按钮：

```py
      # Loop until the user clicks the close button.
      finish = False
      while not finish:

           for anyEvent in p.event.get(): # User did something
               if anyEvent.type == p.QUIT: # If user clicked close
                   finish = True # Flag that we are done so we 
                            exit this loop

       # Set the screen background
               WINDOW.fill(BLACK)

       # Process each snow flake in the list
               for eachSnow in range(len(snowArray)):

       # Draw the snow flake
                   p.draw.circle(WINDOW, color_code_white, snowArray[i], 2)

       # One step down for snow [falling of snow]
                   snowArray[i][1] += 1
```

1.  最后，检查雪花是否在边界内：

```py
# checking if snow is out of boundary or not
 if snowArray[i][1] > 500:
 # reset if it from top
 y_pos = r.randrange(-40, -10)
 snowArray[i][1] = y_pos
 # New random x_position
 x_pos = r.randrange(0, 500)
 snowArray[i][0] = x_pos
```

1.  最后，更新屏幕上已经绘制的内容：

```py
      # Update screen with what you've drawn.
          game.display.update()
          objectClock.tick(20)

      #if you remove following line of code, IDLE will hang at exit
      game.quit()
```

上述代码由许多代码片段组成：初始化游戏变量，然后创建游戏模型。在*步骤 3*中，我们创建了一些简单的逻辑来控制游戏的动画。我们在*步骤 3*中构建了两个代码模型，使我们的游戏对用户进行交互（处理用户事件），并创建一个游戏对象（圆形降雪），它使用`for`循环进行渲染。尽管我们将在接下来的章节中创建更复杂的动画，但这是一个很好的动画程序开始。您可以清楚地看到，在幕后，创建动画需要使用循环、条件和游戏对象。我们使用 Python 编程范式，如 if-else 语句、循环、算术和向量操作来创建游戏对象动画。

除了动画几何形状，您甚至可以动画精灵或图像。为此，您必须制作自己的精灵或从互联网上下载一些。在接下来的部分中，我们将使用`pygame`模块来动画精灵。

# 动画精灵

动画精灵与动画几何形状没有什么不同，但它们被认为是复杂的，因为您必须编写额外的代码来使用动画逻辑`blit`这样的图像。然而，这种动画逻辑对于您加载的每个图像都不会相同；它因游戏而异。因此，您必须事先分析适合您的精灵的动画类型，以便您可以相应地编写代码。在本节中，我们不打算创建任何自定义图像；相反，我们将下载一些（感谢互联网！）。我们将在这些精灵中嵌入动画逻辑，以便我们的程序将促进适当的阴影和移动。

为了让您了解动画静态图像或精灵有多容易，我们将创建一个简单的程序，该程序将加载大约 15 个角色图像（向左和向右移动）。每当用户按键盘上的左键或右键时，我们将`blit`（渲染）它们。执行以下步骤来学习如何创建一个动画精灵程序：

1.  首先，您应该从为`pygame`程序创建一个基本模板开始。您必须导入一些重要的模块，为动画控制台创建一个表面，并声明*空闲*友好的`quit()`函数。

```py
 import pygame
      pygame.init()

      win = pygame.display.set_mode((500,480)) pygame.quit()
```

1.  其次，您必须加载*images*目录中列出的所有精灵和图像。该目录包含几个精灵。您必须下载它并保存在存储 Python 文件的目录中（可以在 GitHub 上找到 sprites/images 文件，网址为[`github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter12`](https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter12)）：

```py
 #walk_Right contains images in which character is turning towards 
         Right direction 
      walkRight = [pygame.image.load('Right1.png'), 
 pygame.image.load('Right2.png'), pygame.image.load('Right3.png'), 
 pygame.image.load('Right4.png'), pygame.image.load('Right5.png'), 
       pygame.image.load('Right6.png'), pygame.image.load('Right7.png'), 
 pygame.image.load('Right8.png'), pygame.image.load('Right9.png')]        #walk_left contains images in which character is turning towards 
         left direction
      walkLeft = [pygame.image.load('Left1.png'), 
 pygame.image.load('Left2.png'), pygame.image.load('Left3.png'), 
 pygame.image.load('Left4.png'), pygame.image.load('Left5.png'), 
 pygame.image.load('Left6.png'), pygame.image.load('Left7.png'), 
 pygame.image.load('Left8.png'), pygame.image.load('Left9.png')]

      #Background and stand still images
      background = pygame.image.load('bg.jpg')
      char = pygame.image.load('standing.png')
```

1.  接下来，我们需要声明一些基本变量，例如角色的初始位置和速度，即游戏精灵每单位按键击移动的距离。在下面的代码中，我已经将速度声明为五个单位，这意味着游戏角色将从当前位置移动固定的 5 个像素：

```py
 x = 50
      y = 400
      width = 40
      height = 60
      vel = 5

      clock = pygame.time.Clock()
```

1.  您必须声明一些额外的变量，以便根据用户在键盘上按下什么来跟踪精灵的移动。如果按下左箭头键，则`left`变量将为`True`，而如果按下右箭头键，则`right`变量将为`False`。`walkCount`变量将跟踪按下键的次数：

```py
 left = False
      right = False
      walkCount = 0
```

在这里，我们已经完成了任何 pygame 程序的基本布局——导入适当的模块，声明变量以跟踪移动，加载精灵等等。程序的另外两个部分是最重要的，所以请确保您理解它们。我们将开始创建一个主循环，像往常一样。这个主循环将处理用户事件，也就是说，当用户按下左或右键时要做什么。其次，您必须创建一些动画逻辑，这将根据用户事件确定在什么时间点`blit`什么图像。

我们将从处理用户事件开始。按照以下步骤进行：

1.  首先，您必须声明一个主循环，它必须是一个无限循环。我们将使用`tick`方法为游戏提供**FPS**。正如您可能记得的那样，这个方法应该在每帧调用一次。它将计算自上一次调用以来经过了多少毫秒：

```py
 finish = False 

 while not finish: clock.tick(27)
```

1.  其次，开始处理关键的用户事件。在简单的精灵动画中，您可以从处理两种基本移动开始：左和右。在接下来的部分中，我们将通过处理跳跃/轻击动作来制作游戏。这段代码应该写在一个 while 循环内：

```py
      while not finish:
           clock.tick(27)
           for anyEvent in pygame.event.get():
              if anyEvent.type == pygame.QUIT:
                  finish = True

           keys = pygame.key.get_pressed()

          #checking key pressed and if character is at x(boundary) or not?
           if keys[pygame.K_LEFT] and x > vel: 
              x -= vel #going left by 5pixels
              left = True
              right = False

          #checking RIGHT key press and is character coincides with 
             RIGHT boundary.
          # value (500 - vel - width) is maximum width of screen, 
             thus x should be less
           elif keys[pygame.K_RIGHT] and x < 500 - vel - width:  
              x += vel #going right by 5pixels
              left = False
              right = True

           else: 
              #not pressing any keys
              left = False
              right = False
              walkCount = 0

          Animation_Logic()
```

观察上述代码的最后一行——对`Animation_Logic()`函数的调用已经完成。然而，这个方法还没有被声明。这个方法是由精灵或图像制作的任何游戏的核心模块。在动画逻辑内编写的代码将执行两个不同的任务：

+   从加载精灵时定义的图像列表中 blit 或渲染图像。在我们的情况下，这些是`walkRight`、`walkLeft`、`bg`和`char`。

+   根据逻辑重新绘制游戏窗口，这将检查从图像池中选择哪个图像。请注意，`walkLeft`包含九个不同的图像。这个逻辑将从这些图像中进行选择。

现在我们已经处理了用户事件，让我们学习如何为之前加载的精灵制作动画逻辑。

# 动画逻辑

精灵是包含角色并具有透明背景的静态图像。这些精灵的额外 alpha 信息是必不可少的，因为在 2D 游戏中，我们希望用户只看到角色而不是他们的背景。想象一下一个游戏，其中一个角色与单调的背景 blit。这会给玩家留下对游戏的坏印象。例如，以下精灵是马里奥角色。假设您正在制作一个马里奥游戏，并且从以下精灵中裁剪一个角色，却忘记去除其蓝色背景。角色连同其蓝色背景将在游戏中呈现，使游戏变得糟糕。因此，我们必须手动使用在线工具或离线工具（如 GIMP）去除（如果有的话）角色背景。精灵表的一个示例如下：

![](img/3e5970c4-966c-4bd4-92fa-1e9e2452d573.png)

现在，让我们继续我们的精灵动画。到目前为止，我们已经使用`pygame`声明了处理事件的模板；现在，让我们编写我们的动画逻辑。正如我们之前所断言的那样，*动画逻辑是简单的逻辑，将在图像之间进行选择并相应地进行 blit。*现在让我们制定这个逻辑：

```py
def Animation_Logic():
    global walkCount

    win.blit(background, (0,0))  

    #check_1
    if walkCount + 1 >= 27:
        walkCount = 0

    if left:  
        win.blit(walkLeft[walkCount//3], (x,y))
        walkCount += 1                          
    elif right:
        win.blit(walkRight[walkCount//3], (x,y))
        walkCount += 1
    else:
        win.blit(char, (x, y))
        walkCount = 0

    pygame.display.update()
```

你将看到的第一件事是`global`变量。`walkCount`变量最初在主循环中声明，并计算用户按下任何键的次数。然而，如果你删除`global walkCount`语句，你将无法在`Animation_Logic`函数内改变`walkCount`的值。如果你只想在函数内访问或打印`walkCount`的值，你不需要将其定义为全局变量。但是，如果你想在函数内操作它的值，你必须将其声明为全局变量。`blit`命令将采用两个参数：一个是需要渲染的精灵，另一个是精灵必须渲染到屏幕上的位置。在前面的代码中，写在`#check_1`之后的代码是为了在角色到达极限位置时对其进行限定。这是一个检查，我们必须渲染一个*char*图像，这是一个角色静止的图像。

渲染精灵始于我们检查左移动是否激活。如果为`True`，则在(*x*, *y*)位置`blit`图像。(*x*, *y*)的值由事件处理程序操作。每当用户按下左箭头键时，*x*的值将从其先前的值减少五个单位，并且图像将被渲染到该位置。由于这个动画只允许角色在水平方向上移动，要么在正的*X*轴上，要么在负的*X*轴上，y 坐标没有变化。同样，对于右移动，我们将从`walkRight`的图像池中渲染图像到指定的(*x*, *y*)位置。在代码的 else 部分，我们`blit`一个 char 图像，这是一个角色静止的图像，没有移动。因此，`walkCount`等于零。在我们`blit`完所有东西之后，我们必须更新它以反映这些变化。我们通过调用`display.update`方法来做到这一点。

让我们运行动画并观察输出：

![](img/d090b846-6b24-4c12-9ce4-197a81fc49fa.png)

在控制台中，如果你按下左箭头键，角色将开始向左移动，如果你按下右箭头键，角色将向右移动。由于 y 坐标没有变化，并且我们没有在主循环中处理任何事件来促进垂直移动，角色只能在水平方向移动。我强烈建议你尝试这些精灵，并尝试通过改变 y 坐标来处理垂直移动。虽然我已经为你提供了一个包含图像列表的资源列表，但如果你想在游戏中使用其他精灵，你可以去以下网站下载任何你想要的精灵：[`www.spriters-resource.com/`](https://www.spriters-resource.com/)。这个网站对于任何 pygame 开发者来说都是一个天堂，所以一定要去访问并下载任何你想要的游戏精灵，这样你就可以尝试这个（用马里奥来尝试可能会更好）。

从下一节开始，我们将开始制作 Flappy Bird 游戏的克隆。我们将学习滚动背景和角色动画、随机对象生成、碰撞和得分等技术。

# 滚动背景和角色动画

现在你已经了解足够关于 pygame 精灵和动画，你有能力制作一个包含复杂精灵动画和多个对象的游戏。在这一部分，我们将通过制作一个 Flappy Bird 游戏来学习滚动背景和角色动画。这个游戏包含多个对象，鸟是游戏的主角，游戏中的障碍物是一对管道。如果你以前没有玩过这个游戏，可以访问它的官方网站试一试：[`flappybird.io/`](https://flappybird.io/)。

说到游戏，制作起来并不难，但通过照顾游戏编程的多个方面，对于初学者来说可能是一项艰巨的任务。话虽如此，我们不打算自己制作任何精灵——它们在互联网上是免费提供的。这使得我们的任务变得更加容易。由于游戏角色的设计是开源的，我们可以直接专注于游戏的编码部分。但是，如果你想从头开始设计你的游戏角色，可以使用任何简单的绘图应用程序开始制作它们。对于这个 Flappy Bird 游戏，我将使用免费提供的精灵。

我已经在 GitHub 链接中添加了资源。如果你打开图像文件夹，然后打开背景图像文件，你会看到它包含特定高度和宽度的背景图像。但是在 Flappy Bird 游戏中，你可以观察到背景图像是连续渲染的。因此，使用 pygame，我们可以制作一个滚动背景，这样我们就可以连续`blit`背景图像。因此，我们可以使用一张图像并连续`blit`它，而不是使用成千上万份相同的背景图像副本。

让我们从制作一个角色动画和一个滚动背景开始。以下步骤向我们展示了如何使用面向对象编程为每个游戏角色制作一个类：

1.  首先，你必须开始声明诸如 math、os（用于加载具有指定文件名的图像）、random、collections 和 pygame 等模块。你还必须声明一些变量，表示每秒帧数设置、动画速度和游戏控制台的高度和宽度：

```py
 import math
 import os
 from random import randint
 from collections import deque

 import pygame
 from pygame.locals import *

      Frame_Rate = 60 #FPS
      ANIMATION_SPEED = 0.18 # pixels per millisecond
      WINDOW_WIDTH = 284 * 2 # Background image sprite size: 284x512 px;                                                                                                  
                              #our screen is twice so to rendered twice: *2
      WINDOW_HEIGHT = 512 
```

1.  现在，让我们将图像文件夹中的所有图像加载到 Python 项目中。我还将创建两个方法，用于在帧和毫秒之间进行转换。

1.  让我们看看`loading_Images`函数是如何通过以下代码工作的：

```py

 def loading_Images():
       """Function to load images"""
  def loading_Image(image_name):

 """Return the sprites of pygame by create unique filename so that 
           we can reference them"""
 new_filename = os.path.join('.', 'images', image_name)
              image = pygame.image.load(new_filename) #loading with pygame 
                                                       module 
              image.convert()
              return image

          return {'game_background': loading_Image('background.png'),
  'endPipe': loading_Image('endPipe.png'),
  'bodyPipe': loading_Image('bodyPipe.png'),
  # GIF format file/images are not supported by Pygame
  'WingUp': loading_Image('bird-wingup.png'),
  'WingDown': loading_Image('bird-wingdown.png')}
```

在前面的程序中，我们定义了`loading_Image`函数，它从特定目录加载/提取所有图像，并将它们作为包含名称作为键和图像作为值的字典返回。让我们通过以下参数分析这样一个字典中的键和值将如何存储：

+   `background.png`：Flappy Bird 游戏的背景图像。

+   `img:bird-wingup.png`：这张 Flappy Bird 的图像有一只翅膀向上指，当在游戏中点击屏幕时渲染。

+   `img:bird-wingdown.png`：这部分图像在 Flappy Bird 自由下落时使用，也就是当用户没有点击屏幕时。这张图像有 Flappy Bird 的翅膀向下指。

+   `img:bodyPipe.png`：这包含了可以用来创建单个管道的离散身体部位。例如，在 Flappy Bird 游戏中，应该从顶部和底部渲染两个离散的管道片段，它们之间留有一个间隙。

+   `img:endPipe.png`：这部分图像是管道对的底部。有两种类型的这样的图像：小管道对的小管道底部和大管道对的大管道底部图像。

同样，我们有一个嵌套的`loading_Image`函数，用于为每个加载的精灵创建一个文件名。它从`/images/文件夹`加载图像。在连续加载每个图像之后，它们会使用`convert()`方法进行调用，以加快 blitting（渲染）过程。传递给`loading_Image`函数的参数是图像的文件名。`image_name`是给定的文件名（连同其扩展名；`.png`是首选）通过`os.path.join`方法加载它，以及`convert()`方法以加快 blitting（渲染）过程。

加载图像后，我们需要创建两个函数，用于在指定的帧速率下执行帧率的转换（请参阅第十章，*使用海龟升级贪吃蛇游戏*，了解更多关于帧速率的信息）。这些函数集主要执行从帧到毫秒的转换以及相反的转换。帧到毫秒的转换很重要，因为我们必须使用毫秒来移动`Bird`角色，也就是鸟要上升的毫秒数，一个完整的上升需要`Bird.CLIMB_DURATION`毫秒。如果你想让鸟在游戏开始时做一个（小）上升，可以使用这个。让我们创建这样两组函数（代码的详细描述也可以在 GitHub 上找到：[`github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter12`](https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter12)）：

```py
def frames_to_msec(frames, fps=FPS):
    """Convert frames to milliseconds at the specified framerate.   Arguments: frames: How many frames to convert to milliseconds. fps: The framerate to use for conversion.  Default: FPS. """  return 1000.0 * frames / fps

def msec_to_frames(milliseconds, fps=FPS):
    """Convert milliseconds to frames at the specified framerate.   Arguments: milliseconds: How many milliseconds to convert to frames. fps: The framerate to use for conversion.  Default: FPS. """  return fps * milliseconds / 1000.0
```

现在，为鸟角色声明一个类。回想一下第六章，*面向对象编程*，我们学到每个实体都应该由一个单独的类来表示。在 Flappy Bird 游戏中，代表`PipePair`（障碍物）的实体或模型与另一个实体（比如鸟）是不同的。因此，我们必须创建一个新的类来表示另一个实体。这个类将代表由玩家控制的鸟。由于鸟是我们游戏的“英雄”，鸟角色的任何移动只允许由玩游戏的用户来控制。玩家可以通过点击屏幕使鸟上升（快速上升），否则它会下沉（缓慢下降）。鸟必须通过管道对之间的空间，每通过一个管道就会得到一个积分。同样，如果鸟撞到管道，游戏就结束了。

现在，我们可以开始编写我们的主角了。你还记得如何做吗？这是任何优秀游戏程序员的最重要特征之一——他们会进行大量头脑风暴，然后写出小而优化的代码。因此，让我们先进行头脑风暴，预测我们想要如何构建鸟角色，以便之后可以无缺陷地编写代码。以下是一些必须作为 Bird 类成员定义的基本属性和常量：

+   **类的属性**：`x`是鸟的 X 坐标，`y`是鸟的 Y 坐标，`msec_to_climb`表示鸟要上升的毫秒数，一个完整的上升需要`Bird.CLIMB_DURATION`毫秒。

+   **常量**：

+   `WIDTH`：鸟图像的宽度（以像素为单位）。

+   `HEIGHT`：鸟图像的高度（以像素为单位）。

+   `SINK_SPEED`：鸟在不上升时每毫秒下降的像素速度。

+   `CLIMB_SPEED`：鸟在上升时每毫秒上升的像素速度，平均而言。更多信息请参阅`Bird.update`文档字符串。

+   `CLIMB_DURATION`：鸟执行完整上升所需的毫秒数。

现在我们已经有了关于游戏中鸟角色的足够信息，我们可以开始为其编写代码了。下面的代码行表示 Bird 类，其中成员被定义为类属性和常量：

```py
class Bird(pygame.sprite.Sprite):     WIDTH = HEIGHT = 50
  SINK_SPEED = 0.18
  CLIMB_SPEED = 0.3   CLIMB_DURATION = 333.3    def __init__(self, x, y, msec_to_climb, images):
        """Initialize a new Bird instance."""    super(Bird, self).__init__() 
        self.x, self.y = x, y
        self.msec_to_climb = msec_to_climb
        self._img_wingup, self._img_wingdown = images
        self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
        self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)
```

让我们来谈谈鸟类内部定义的构造函数或初始化器。它包含许多参数，可能会让你感到不知所措，但它们实际上很容易理解。在构造函数中，我们通常定义类的属性，比如代表鸟位置的 x 和 y 坐标，以及其他参数。现在让我们来看看这些：

+   `x`：鸟的初始 X 坐标。

+   `y`：鸟的初始 Y 坐标。

+   `msec_to_climb`: 剩余的毫秒数要爬升，完整的爬升需要 `Bird.CLIMB_DURATION` 毫秒。如果你想让小鸟在游戏开始时做一个（小）爬升，可以使用这个。

+   `images`: 包含此小鸟使用的图像的元组。它必须按照以下顺序包含以下图像：

+   小鸟上飞时的翅膀

+   小鸟下落时的翅膀

最后，应声明三个重要属性。这些属性是`image`、`mask`和`rect`。想象属性是小鸟在游戏中的基本动作。它可以上下飞行，这在图像属性中定义。然而，小鸟类的另外两个属性相当不同。`rect`属性将获取小鸟的位置、高度和宽度作为`Pygame.Rect`（矩形的形式）。记住，`pygame`可以使用`rect`属性跟踪每个游戏角色，类似于一个无形的矩形将被绘制在精灵周围。mask 属性获取一个位掩码，可用于与障碍物进行碰撞检测：

```py
@property def image(self):
    "Gets a surface containing this bird image"   if pygame.time.get_ticks() % 500 >= 250:
        return self._img_wingup
    else:
        return self._img_wingdown

@property def mask(self):
    """Get a bitmask for use in collision detection.   The bitmask excludes all pixels in self.image with a transparency greater than 127."""  if pygame.time.get_ticks() % 500 >= 250:
        return self._mask_wingup
    else:
        return self._mask_wingdown

@property def rect(self):
    """Get the bird's position, width, and height, as a pygame.Rect."""
  return Rect(self.x, self.y, Bird.WIDTH, Bird.HEIGHT)
```

由于我们已经熟悉了`rect`和`mask`属性的概念，我就不再重复了，所以让我们详细了解一下图像属性。图像属性获取指向小鸟当前图像的表面。这将决定根据`pygame.time.get_ticks()`返回一个图像，其中小鸟的可见翅膀指向上方或指向下方。这将使 Flappy Bird 动画化，即使 pygame 不支持*动画 GIF*。

现在是时候结束`Bird`类了，但在此之前，你必须声明一个方法，用于更新小鸟的位置。确保你阅读了我在三引号中添加的描述，作为注释：

```py
def update(self, delta_frames=1):
    """Update the bird's position.
 One complete climb lasts CLIMB_DURATION milliseconds, during which the bird ascends with an average speed of CLIMB_SPEED px/ms. This Bird's msec_to_climb attribute will automatically be decreased accordingly if it was > 0 when this method was called.   Arguments: delta_frames: The number of frames elapsed since this method was last called. """  if self.msec_to_climb > 0:
        frac_climb_done = 1 - self.msec_to_climb/Bird.CLIMB_DURATION
        #logic for climb movement
        self.y -= (Bird.CLIMB_SPEED * frames_to_msec(delta_frames) *
                   (1 - math.cos(frac_climb_done * math.pi)))
        self.msec_to_climb -= frames_to_msec(delta_frames)
    else:
        self.y += Bird.SINK_SPEED * frames_to_msec(delta_frames)
```

数学`cosine(angle)`函数用于使小鸟平稳爬升。余弦是一个偶函数，这意味着小鸟会做一个平稳的爬升和下降运动：当小鸟在屏幕中间时，可以执行一个高跳，但当小鸟靠近顶部/底部边界时，只能做一个轻微的跳跃（这是 Flappy Bird 运动的基本原理）。

让我们运行游戏，看看小鸟是如何渲染的。然而，我们还没有创建任何逻辑来让玩家玩游戏（我们很快会做到）。现在，让我们运行游戏，观察界面的样子：

![](img/be35d72a-846f-4d44-b17f-72a7b29304f2.png)

根据上述代码，你必须能够创建一个完整的`Bird`类，其中包含用于遮罩、更新和获取位置（即高度和宽度）的属性，使用`rect`。我们 Flappy Bird 游戏中的小鸟角色仅与运动相关——垂直上下移动。我们游戏中的下一个角色是管道（小鸟的障碍物），处理起来相当复杂。我们必须随机连续地`blit`管道对。让我们看看如何做到这一点。

# 理解随机对象生成

我们已经在前面的部分中介绍了`Bird`角色的动画。它包括一系列处理小鸟垂直运动的属性和特性。由于`Bird`类仅限于为小鸟角色执行动作，我们无法向其添加任何其他角色属性。例如，如果你想在游戏中为障碍物（管道）添加属性，不能将其添加到`Bird`类中。你必须创建另一个类来定义下一个对象。这个概念被称为封装（我们在第六章中学习过，*面向对象编程*），其中代码和数据被包装在一个单元内，以便其他实体无法伤害它。

让我们创建一个新的类来生成游戏的障碍物。你必须首先定义一个类，以及一些常量。我已经在代码中添加了注释，以便你能理解这个类的主要用途：

```py
class PipePair(pygame.sprite.Sprite):
    """class that provides obstacles in the way of the bird in the form of pipe-pair.""" 

 WIDTH = 80
  HEIGHT_PIECE = 32
  ADD_INTERVAL = 3000
```

在我们实际编写这个`PipePair`类之前，让我给你一些关于这个类的简洁信息，以便你能理解以下每个概念。我们将使用不同的属性和常量，如下所示：

+   `PipePair`类：一个管道对（两根管道的组合）被插入以形成两根管道，它们之间只提供了一个小间隙，这样小鸟才能穿过它们。每当小鸟触碰或与任何管道对碰撞时，游戏就会结束。

+   **属性**：`x`是`pipePair`的*X*位置。这个值是一个浮点数，以使移动更加平滑。`pipePair`没有*Y*位置，因为它在*y*方向上不会改变；它始终保持为 0。

+   `image`：这是`pygame`模块提供的表面，用于`blit` `pipePair`。

+   `mask`：有一个位掩码，排除了所有`self.image`中透明度大于 127 的像素。这可以用于碰撞检测。

+   `top_pieces`：顶部管道与末端部分的组合，这是管道顶部部分的基础（这是一个由管道顶部部分组成的一对）。

+   `bottom_pieces`：下管道（向上指向的隧道）与末端部分的组合，这是底部管道的基础。

+   **常量**：

+   `WIDTH`：管道片段的宽度，以像素为单位。因为管道只有一片宽，这也是`PipePair`图像的宽度。

+   `PIECE_HEIGHT`：管道片段的高度，以像素为单位。

+   `ADD_INTERVAL`：添加新管道之间的间隔，以毫秒为单位。

正如我们已经知道的，对于任何类，我们需要做的第一件事就是初始化一个类或构造函数。这个方法将初始化新的随机管道对。以下截图显示了管道对应该如何渲染。管道有两部分，即顶部和底部，它们之间插入了一个小空间：

![](img/66b04e30-29e3-4eaf-b559-771accc219a8.png)

让我们为`PipePair`类创建一个初始化器，它将`blit`管道的底部和顶部部分，并对其进行蒙版处理。让我们了解一下需要在这个构造函数中初始化的参数：

+   `end_image_pipe`：代表管道底部（末端部分）的图像

+   `body_image_pipe`：代表管道垂直部分（管道的一部分）的图像

管道对只有一个 x 属性，y 属性为 0。因此，`x`属性的值被赋为`WIN_WIDTH`，即`float(WIN_WIDTH - 1)`。

以下步骤代表了需要添加到构造函数中以在游戏界面中创建一个随机管道对的代码：

1.  让我们为`PipePair`初始化一个新的随机管道对：

```py
 def __init__(self, end_image_pipe, body_image_pipe):
          """Initialises a new random PipePair.  """  self.x = float(WINDOW_WIDTH - 1)
          self.score_counted = False
  self.image = pygame.Surface((PipePair.WIDTH, WINDOW_HEIGHT), 
                       SRCALPHA)
          self.image.convert() # speeds up blitting
  self.image.fill((0, 0, 0, 0))

        #Logic 1: **create pipe-pieces**--- Explanation is provided after
                     the code
 total_pipe_body_pieces = int((WINDOW_HEIGHT - # fill window from 
                                                           top to bottom
  3 * Bird.HEIGHT - # make room for bird to fit through
  3 * PipePair.HEIGHT_PIECE) / # 2 end pieces + 1 body piece
  PipePair.HEIGHT_PIECE # to get number of pipe pieces
  )
 self.bottom_pipe_pieces = randint(1, total_pipe_body_pieces)
 self.top_pipe_pieces = total_pipe_body_pieces - 
 self.bottom_pieces
```

1.  接下来，我们需要定义两种类型的管道对——底部管道和顶部管道。添加管道对的代码会将管道图像 blit，并且只关心管道对的*y*位置。管道对不需要水平坐标（它们应该垂直渲染）：

```py
       # bottom pipe
  for i in range(1, self.bottom_pipe_pieces + 1):
              piece_pos = (0, WIN_HEIGHT - i*PipePair.PIECE_HEIGHT)
              self.image.blit(body_image_pipe, piece_pos)
          end_y_bottom_pipe = WIN_HEIGHT - self.bottom_height_px
          bottom_end_piece_pos = (0, end_y_bottom_pipe - 
                                 PipePair.PIECE_HEIGHT)
          self.image.blit(end_image_pipe, bottom_end_piece_pos)

          # top pipe
  for i in range(self.top_pipe_pieces):
              self.image.blit(body_image_pipe, (0, i * 
                   PipePair.PIECE_HEIGHT))
          end_y_top_pipe = self.top_height_px
          self.image.blit(end_image_pipe, (0, end_y_top_pipe))

          # external end pieces are further added to make compensation
  self.top_pipe_pieces += 1
  self.bottom_pipe_pieces += 1    # for collision detection
  self.mask = pygame.mask.from_surface(self.image)
```

尽管代码旁边提供的注释有助于理解代码，但我们需要以更简洁的方式了解逻辑。`total_pipe_body_piece`变量存储了一帧中可以添加的管道数量的高度。例如，它推断了可以插入当前实例的底部管道和顶部管道的数量。我们将其强制转换为整数，因为管道对始终是整数。`bottom_pipe_piece`类属性表示底部管道的高度。它可以在 1 到`total_pipe_piece`支持的最大宽度范围内。类似地，顶部管道的高度取决于总管道件数。例如，如果画布的总高度为 10，底部管道的高度为 1，那么通过在两个管道对之间留下一个间隙（假设为 3），剩下的高度应该是顶部管道的高度（即其高度为 10 - (3+1) = 6），这意味着除了管道对之间的间隙外，不应提供其他间隙。

前面的代码中的所有内容都是不言自明的。尽管代码很简单，但我希望你专注于代码的最后一行，我们用它来检测碰撞。检测的过程很重要，因为在 Flappy Bird 游戏中，我们必须检查小鸟是否与管道对发生碰撞。通常通过使用`pygame.mask`模块添加蒙版来实现。

现在，是时候向`PipePair`类添加一些属性了。我们将添加四个属性：`visible`、`rect`、`height_topPipe_px`和`height_bottomPipe_px`。`rect`属性的工作方式类似于`Bird`类的`rect`调用，它返回包含`PipePair`的矩形。类的`visible`属性检查管道对在屏幕上是否可见。另外两个属性返回以像素为单位的顶部和底部管道的高度。以下是`PipePair`类的前四个属性的代码：

```py
@property def height_topPipe_px(self):
 """returns the height of the top pipe, measurement is done in pixels"""
  return (self.top_pipe_pieces * PipePair.HEIGHT_PIECE)

@property def height_bottomPipe_px(self):
 """returns the height of the bottom pipe, measurement is done in pixels"""
  return (self.bottom_pipe_pieces * PipePair.HEIGHT_PIECE)

@property def visible(self):
    """Get whether this PipePair on screen, visible to the player."""
  return -PipePair.WIDTH < self.x < WINDOW_WIDTH

@property def rect(self):
    """Get the Rect which contains this PipePair."""
  return Rect(self.x, 0, PipePair.WIDTH, PipePair.HEIGHT_PIECE)
```

现在，在封装之前，我们需要向`PipePair`类添加另外两个方法。第一个方法`collides_with`将检查小鸟是否与管道对中的管道发生碰撞：

```py
def collides_with(self, bird):
    """check whether bird collides with any pipe in the pipe-pair. The 
       collide-mask deploy a method which returns a list of sprites--in 
       this case images of bird--which collides or intersect with 
       another sprites (pipe-pair)   Arguments: bird: The Bird which should be tested for collision with this PipePair. """  return pygame.sprite.collide_mask(self, bird)
```

第二个方法`update`将更新管道对的位置：

```py
def update(self, delta_frames=1):
    """Update the PipePair's position.   Arguments: delta_frames: The number of frames elapsed since this method was last called. """  self.x -= ANIMATION_SPEED * frames_to_msec(delta_frames)
```

现在我们知道每个方法的工作原理，让我们看看代码的运行情况。在运行游戏之前，你不会了解游戏中的任何缺陷。花时间运行游戏并观察输出：

![](img/3876e226-50d8-48e1-9258-9fc566b22a87.png)

好的，游戏足够吸引人了。点击事件完美地工作，背景图像与鸟的图像一起呈现，并且上升和下沉动作的物理效果也很好。然而，你可能已经观察到一个奇怪的事情（如果没有，请看前面的截图），即在与管道对碰撞后，我们的小鸟能够继续向前移动。这是我们游戏中的一个大缺陷，我们不希望出现这种情况。相反，我们希望在发生这种情况时关闭游戏。因此，为了克服这样的错误，我们必须使用碰撞的概念（一种处理多个游戏对象相互碰撞的技术）。

现在我们已经完成了两个游戏角色类，即`Bird`和`PipePair`，让我们继续制作游戏的物理部分：初始化显示和处理碰撞。

# 检测碰撞

*处理碰撞*的过程是通过找出两个独立对象触碰时必须执行的操作来完成的。在前面的部分中，我们为每个对象添加了一个掩码，以检查两个对象是否发生碰撞。`pygame`模块使得检查碰撞过程非常容易；我们可以简单地使用`sprite.collide_mask`来检查两个对象是否接触。然而，这个方法所需的参数是掩码对象。在前一节中，我们添加了`collides_with`方法来检查鸟是否与管道对中的一个碰撞。现在，让我们使用该方法来检查碰撞。

除了检测碰撞，我们还将为游戏制作一个物理布局/模板。我在这一部分没有强调基本的 pygame 布局，因为自从我们开始做这个以来，这对你来说应该是不言自明的。以下步骤描述了制作一个检测游戏角色碰撞（`Bird`与`pipePairs`）的模型的布局：

1.  首先定义主函数，之后将被外部调用：

```py
 def main():
          """Only function that will be externally called, this 
            is main function  Instead of importing externally, if we call this function from 
            if **name** == __main__(), this main module will be executed.  """   pygame.init()

          display_surface = pygame.display.set_mode((WIN_WIDTH, 
              WIN_HEIGHT)) #display for screen

          objectClock = pygame.time.Clock()   images = loading_Images()
```

1.  让我们创建一些逻辑，使鸟出现在屏幕的中心。如果你玩过 Flappy Bird 游戏，你会知道鸟被放在画布的中心，它可以向上或向下移动：

```py
       #at any moment of game, bird can only change its y position, 
         so x is constant
          #lets put bird at center           Objectbird = Bird(50, int(WIN_HEIGHT/2 - Bird.HEIGHT/2), 2,
  (images['WingUp'], images['WingDown']))

          pipes = deque() 
      #deque is similar to list which is preferred otherwise 
         if we need faster operations like 
      #append and pop

          frame_clock = 0 # this counter is only incremented 
            if the game isn't paused
```

1.  现在，我们必须将管道对图像添加到`pipes`变量中，因为一个管道是由`pipe-body`和`pipe-end`连接而成的。这个连接是在`PipePair`类内部完成的，因此在创建实例后，我们可以将管道对附加到管道列表中：

```py
  done = paused = False
 while not done:
              clock.tick(FPS)

              # Handle this 'manually'.  
                If we used pygame.time.set_timer(),
 # pipe addition would be messed up when paused.  if not (paused or frame_clock % 
                msec_to_frames(PipePair.ADD_INTERVAL)):
                  pipe_pair = PipePair(images['endPipe'], 
                    images['bodyPipe'])
                  pipes.append(pipe_pair)
```

1.  现在，处理用户的操作。由于 Flappy Bird 游戏是一个点击游戏，我们将处理鼠标事件（参考我们在第十一章中涵盖的*鼠标控制*部分，*使用 Pygame 制作超越乌龟-贪吃蛇游戏 UI*）：

```py
      *#handling events
          **#Since Flappy Bird is Tapped game**
 **#we will handle mouse events***
 *for anyEvent in pygame.event.get():
              #EXIT GAME IF QUIT IS PRESSED*
 *if anyEvent.type == QUIT or (anyEvent.type == KEYUP and 
                anyEvent.key == K_ESCAPE):*
 *done = True
 break elif anyEvent.type == KEYUP and anyEvent.key in 
              (K_PAUSE, K_p):* *paused = not paused*
 *elif anyEvent.type == MOUSEBUTTONUP or 
                (anyEvent.type == KEYUP and anyEvent.key in 
                (K_UP, K_RETURN, K_SPACE)):* *bird.msec_to_climb = 
                Bird.CLIMB_DURATION*

           if paused: 
              continue #not doing anything [halt position]  
```

1.  最后，这就是你一直在等待的：如何利用 Python 的`pygame`模块构建碰撞接口。在完成这些步骤的其余部分后，我们将详细讨论以下代码的突出部分：

```py
 # check for collisions  pipe_collision = any(eachPipe.collides_with(bird) 
                for eachPipe in pipes)
 if pipe_collision or 0 >= bird.y or 
                bird.y >= WIN_HEIGHT - Bird.HEIGHT:
 done = True
 #blit background for position_x_coord in (0, WIN_WIDTH / 2):
 display_surface.blit(images['game_background'], 
                    (position_x_coord, 0))

              #pipes that are out of visible, remove them
 while pipes and not pipes[0].visible:
 pipes.popleft()

 for p in pipes:
 p.update()
 display_surface.blit(p.image, p.rect)

 bird.update()
 display_surface.blit(bird.image, bird.rect) 
```

1.  最后，以一些多余的步骤结束程序，比如使用更新函数渲染游戏，给用户一个多余的消息等等：

```py
              pygame.display.flip()
              frame_clock += 1
          print('Game Over!')
          pygame.quit()
      #----------uptill here add it to main function----------

      if __name__ == '__main__':
        #indicates two things:
        #In case other program import this file, then value of 
           __name__ will be flappybird
        #if we run this program by double clicking filename 
           (flappybird.py), main will be called

          main()     #calling main function
```

在前面的代码中，突出显示的部分很重要，所以确保你理解它们。在这里，`any()`函数通过检查鸟是否与管道对碰撞来返回一个布尔值。根据这个检查，如果是`True`，我们就退出游戏。我们还将检查鸟是否触碰到了水平最低或水平最高的边界，如果是的话也会退出游戏。

让我们运行游戏并观察输出：

![](img/4296ca36-a86e-479b-9b3b-7178b05417fe.png)

游戏已经足够可玩了，所以让我们为游戏添加一个告诉玩家他们得分如何的功能。

# 得分和结束屏幕

给 Flappy Bird 游戏添加分数非常简单。玩家的分数将是玩家通过的管道或障碍物的数量。如果玩家通过了 20 个管道，他们的分数将是 20。让我们给游戏添加一个得分屏幕：

```py
score = 0
scoreFont = pygame.font.SysFont(None, 30, bold=True) #Score default font: WHITE

while not done:
    #after check for collision
    # procedure for displaying and updating scores of player
     for eachPipe in pipes:
         if eachPipe.x + PipePair.WIDTH < bird.x and not 
           eachPipe.score_counted: 
            #when bird crosses each pipe
             score += 1
             eachPipe.score_counted = True

     Surface_Score = scoreFont.render(str(score), 
        True, (255, 255, 255)) #surface
     x_score_dim = WIN_WIDTH/2 - score_surface.get_width()/2 
     #to render score, no y-position
     display_surface.blit(Surface_Score, (x_score_dim, 
        PipePair.HEIGHT_PIECE)) #rendering

     pygame.display.flip() #update
     frame_clock += 1
print('Game over! Score: %i' % score)
pygame.quit() 
```

现在，游戏看起来更吸引人了：

![](img/70199436-5df8-4e03-acf8-bef436d2120a.png)

在下一节中，我们将看看如何测试一切，并尝试应用一些修改。

# 游戏测试

虽然 Flappy Bird 可以修改的地方较少，但你总是可以通过修改一些游戏角色属性来测试游戏，以改变游戏的难度。在前一节中，我们运行了我们的游戏，并看到管道对之间有很大的空间。这将使游戏对许多用户来说非常容易，所以我们需要通过缩小两个管道对之间的空间来增加难度。例如，在`Bird`类中，我们声明了四个属性。将它们更改为不同的值以观察效果：

```py
WIDTH = HEIGHT = 30 #change it to make space between pipe pairs 
                     smaller/bigger SINK_SPEED = 0.18 #speed at which bird falls CLIMB_SPEED = 0.3 #when user taps on screen, it is climb speed
                  #make it smaller to make game harder CLIMB_DURATION = 333.3
```

您还可以改变游戏属性的值，使您的游戏看起来独一无二。Flappy Bird 中使用的一些不同游戏属性包括*每秒帧数*和*动画速度*。您可以改变这些值来实现必要的变化。虽然您可以改变动画速度的值，但对于 Flappy Bird 游戏来说，每秒帧数为 60 是足够的。

与手动调试和搜索可能的修改不同，您可以简单地在调试模式下运行程序以更快地测试它。假设您已经在 Pycharm 的 IDE 中编写了 Flappy Bird 游戏（我推荐这样做），您可以通过按下*Shift* + *F9*或简单地点击运行选项卡并从那里以调试模式运行程序。运行后，尝试玩游戏，并尝试使其适应用户可能遇到的任何情况。任何错误都将出现在程序的终端中，您可以从中跳转到具有多个错误的程序位置。

# 总结

在本章中，我们更深入地探讨了精灵动画和碰撞的概念。我们看了如何为几何形状制作简单动画，创建复杂的精灵动画，并了解了在某些情况下哪种方法最有效。我们将 pygame 的事件处理方法与动画逻辑相结合，根据当前的游戏状态渲染图像。基本上，动画逻辑维护一个队列，用户事件将被存储在其中。一次获取一个动作将图像渲染到一个位置。

使用 pygame 制作的游戏原型有三个核心模块：加载精灵（原始精灵或从互联网下载的精灵）、处理用户事件和动画逻辑，控制游戏角色的移动。有时，您可能不是拥有独立的精灵图像，而是精灵表—包含角色图像的表。您可以使用在线工具或甚至 pygame 的`rect`方法来裁剪它们。在获得游戏的适当图像或精灵后，我们处理了用户事件，并创建了动画逻辑来使游戏精灵移动。我们还研究了 pygame 的遮罩属性，可以用来检测对象之间的碰撞。

完成本章后，您现在了解了游戏控制器和动画，已经了解了碰撞原理（包括 pygame 的遮罩属性），已经了解了精灵动画（创建角色的奔跑动画），并已经了解了添加交互式记分屏幕以使游戏更加用户友好。

您在本章中获得的知识可以应用的领域范围广泛，对大多数 Python pygame 开发人员来说是*纯金*。处理精灵对于几乎所有基于 pygame 的游戏都很重要。尽管角色动画、碰撞和移动是简单但强大的概念，但它们是使 Python 游戏具有吸引力和互动性的三个主要方面。现在，尝试创建一个简单的**角色扮演游戏**（**RPG**）游戏，比如 Junction Jam（如果您还没有听说过，可以搜索一下），并尝试在其中嵌入碰撞和精灵移动的概念。

在下一章中，我们将通过创建游戏网格和形状来学习 pygame 的基本图形编程。我们将通过编写俄罗斯方块游戏来学习多维列表处理和有效空间确定。
