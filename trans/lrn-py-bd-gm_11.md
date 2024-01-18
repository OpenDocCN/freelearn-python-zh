# 使用Pygame超越Turtle - 使用Pygame制作贪吃蛇游戏UI

Python游戏开发在某种程度上与`pygame`模块相关。到目前为止，我们已经学习了关于Python的各种主题和技术，因为在我们进入`pygame`模块之前，我们必须了解它们。所有这些概念将被用作构建Pygame游戏时的技术。我们现在可以开始使用面向对象的原则，矢量化移动进行事件处理，旋转技术来旋转游戏中使用的图像或精灵，甚至使用我们在turtle模块中学到的东西。在turtle模块中，我们学习了如何创建对象（参见[第6章](7f11f831-b5e7-4605-a9bd-25bfb5e3098e.xhtml)，*面向对象编程*），这些对象可以用于在我们可能使用Pygame构建的游戏的基本阶段调试不同的功能。因此，我们迄今为止学到的东西将与Pygame模块的其他功能一起使用，这些功能可以帮助我们制作更吸引人的游戏。

在本章中，我们将涵盖多个内容，从学习Pygame的基础知识——安装、构建模块和不同功能开始。之后，我们将学习Pygame的不同对象。它们是可以用于多种功能的模块，例如将形状绘制到屏幕上，处理鼠标和键盘事件，将图像加载到Pygame项目中等等。在本章的最后，我们将尝试通过添加多个功能使我们的贪吃蛇游戏在视觉上更具吸引力，例如自定义的贪吃蛇图像、苹果作为食物以及游戏的菜单屏幕。最后，我们将把我们的贪吃蛇游戏转换为可执行文件，以便您可以将游戏与朋友和家人分享，并从他们那里获得反馈。本章将涵盖以下主题：

+   Pygame基础知识

+   Pygame对象

+   初始化显示和处理事件

+   对象渲染——制作贪吃蛇游戏

+   游戏菜单

+   转换为可执行文件

+   游戏测试和可能的修改

# 技术要求

您需要以下要求才能完成本章：

+   Python—3.5或更高版本

+   PyCharm IDE——参考[第1章](0ef9574b-5690-454e-971f-85748021018d.xhtml)，*了解Python-设置Python和编辑器*，了解下载过程

本章的文件可以在[https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter11](https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter11)找到。

查看以下视频，以查看代码的运行情况：

[http://bit.ly/2o2GngQ](http://bit.ly/2o2GngQ)

# 理解pygame

使用`pygame`模块编写游戏需要在您的计算机上安装pygame。您可以通过访问官方Pygame库的网站（[www.pygame.org](http://www.pygame.org)）手动下载，或者使用终端并使用`pip install pygame`命令进行安装。

Pygame模块可以免费从上述网站下载，因此我们可以按照与下载其他Python模块相似的过程进行下载。但是，我们可以通过使用视觉上更具吸引力和有效的替代IDE **PyCharm** 来消除手动下载pygame的麻烦，我们在[第1章](0ef9574b-5690-454e-971f-85748021018d.xhtml)，*了解Python-设置Python和编辑器*中下载了PyCharm。在该章节中，我们熟悉了在PyCharm中下载和安装第三方包的技术。

一旦您将pygame包下载到PyCharm中，请给它一些时间来加载。现在，我们可以通过编写以下代码来测试它。以下两行代码检查`pygame`模块是否已下载，如果已下载，它将打印其版本：

```py
import pygame
print(pygame.version.ver) #this command will check pygame version installed
print(pygame.version.vernum) #alternate command
```

如果pygame成功安装到您的计算机上，您将观察到以下输出。版本可能有所不同，但在撰写本书时，它是1.9.6版（2019年最新版本）。本书的内容适用于任何版本的`pygame`，因为它具有向后兼容性。请确保您的pygame版本新于1.9+：

```py
pygame 1.9.6
Hello from the pygame community. https://www.pygame.org/contribute.html
1.9.6
```

Pygame对许多Python游戏开发者来说是一个乌托邦；它包含大量的模块，从制作界面到处理用户事件。pygame中定义的所有这些模块都可以根据我们的需求独立使用。最重要的是，您也可以使用pygame制作游戏，这可能是平台特定的，也可能不是。调用pygame的模块类似于调用类的方法。您可以始终使用pygame命名空间访问这些类，然后使用您想要使用的类。例如，`pygame.key`将读取键盘上按下的键。因此，`key`类负责处理键盘操作。类似地，`pygame.mouse`模块用于管理鼠标事件。pygame的这些以及许多其他模块都可以相互独立地调用，这使得我们的代码更易于管理和阅读。您可以从pygame模块的官方文档页面搜索可用模块的列表，但几乎80%的游戏只需要四到六个模块。如果您想了解更多信息，最好是探索其官方文档页面。在其中，我们在每个游戏中主要使用两个类，即显示模块，以便访问和操作游戏显示；以及鼠标和键盘或操纵杆模块，以处理游戏的输入事件。我不会说其他模块不重要，但这些模块是游戏的基石。以下表格摘自Python pygame官方文档；它给了我们关于`pygame`模块及其用法的简洁概念：

| **模块名称** | **描述** |
| `pygame.draw` | 绘制形状、线条和点。 |
| `pygame.event` | 处理外部事件。 |
| `pygame.font` | 处理系统字体。 |
| `pygame.image` | 将图像加载到项目中。 |
| `pygame.joystick` | 处理操纵杆移动/事件。 |
| `pygame.key` | 从键盘读取按键。 |
| `pygame.mixer` | 混音、加载和播放声音。 |
| `pygame.mouse` | 读取鼠标事件。 |
| `pygame.movie` | 播放/运行电影文件。 |
| `pygame.music` | 播放流式音频文件。 |
| `pygame` | 捆绑为高级pygame函数/方法。 |
| `pygame.rect` | 处理矩形区域并可以创建一个框结构。 |

此外还有一些其他模块，比如surface、time和transform。我们将在本章和接下来的章节中探讨它们。所有前述的模块都是平台无关的，这意味着它们可以被调用，无论机器使用的操作系统是什么。但是会有一些特定于操作系统的错误，以及由于硬件不兼容或不正确的设备驱动程序而导致的错误。如果任何模块与任何机器不兼容，Python解析器将其返回为“None”，这意味着我们可以事先检查以确保游戏正常工作。以下代码将检查是否存在任何指定的模块（`pygame.module_name`），如果没有，它将在打印语句中返回一个自定义消息，本例中是“没有这样的模块！尝试其他”：

```py
if pygame.overlay is None:
    print("No such module! Try other one")
    print("https://www.pygame.org/contribute.html")
    exit()
```

要完全掌握`pygame`的概念，我们必须养成观察其他pygame开发者编写的代码的习惯。通过这样做，您将学习使用`pygame`构建游戏的模式。如果像我一样，只有在陷入僵局时才查看文档，那么我们可以编写一个简单的程序来帮助我们理解`pygame`的概念以及我们可以调用其不同模块的方式。我们将编写一个简单的代码来说明这一点：

```py
import pygame as p #abbreviating pygame with p

p.init()
screen = p.display.set_mode((400, 350)) #size of screen
finish = False   while not finish:
    for each_event in p.event.get():
        if each_event.type == p.QUIT:
            finish = True
  p.draw.rect(screen, (0, 128, 0), p.Rect(35, 35, 65, 65))
    p.display.flip()
```

在讨论上述代码之前，让我们运行它并观察输出。您将得到一个几何形状—一个绿色的矩形框，它将呈现在特定高度和宽度的屏幕内。现在，是时候快速地记下`pygame`模块的构建块了。为了简化事情，我已经在以下几点中列出了它们：

+   `import pygame`: 这是我们从本书开始就熟悉的导入语句。这次，我们将pygame框架导入到我们的Python文件中。

+   `pygame.init()`: 这个方法将初始化pygame内嵌的一系列模块/类。这意味着我们可以使用pygame的命名空间调用其他模块。

+   `pygame.display.set_mode((width, height))`: 作为元组(width, height)传递的大小是期望的屏幕大小。这个大小代表我们的游戏控制台。返回的对象将是一个窗口屏幕，或者表面，我们将在其中执行不同的图形计算。

+   `pygame.event.get()`: 这个语句将处理事件队列。正如我们在前几章中讨论的那样，队列将存储用户的不同事件。如果不显式调用此语句，游戏将受到压倒性的Windows消息的阻碍，最终将变得无响应。

+   `pygame.draw.rect()`: 我们将能够使用绘图模块在屏幕上绘制。不同的形状可以使用此模块绘制。关于这一点，我们将在下一节—*Pygame对象*中进行更多讨论。`rect()`方法以屏幕对象、颜色和位置作为参数，绘制一个矩形。第一个参数代表屏幕对象，它是显示类的返回对象；第二个是颜色代码，以RGB(red, green, blue)代码的形式作为元组传递；第三个是矩形的尺寸。为了操纵和存储矩形区域，pygame使用`Rect`对象。`Rect()`可以通过组合四个不同的值—高度、宽度、左侧和顶部来创建。

+   `pygame.QUIT`: 每当您明确关闭pygame屏幕时，就会调用此事件，这是通过按游戏控制台最右上角的`close(X)`按钮来完成的。

+   `pygame.display.flip()`: 这与`update()`函数相同，可以使屏幕上的任何新更新可见。在制作或blitting形状或字符时，必须在游戏结束时调用此方法，以确保所有对象都被正确渲染。这将交换pygame缓冲区，因为pygame是一个双缓冲框架。

上述代码在执行时呈现绿色矩形形状。正如我们之前提到的，`rect()`方法负责创建矩形区域，颜色代码(0, 128, 0)代表绿色。

不要被这些术语所压倒；您将在接下来的章节中详细了解它们。在阅读本章时，请确保养成一个习惯，即在代码之间建立逻辑连接：从一个位置映射游戏到另一个位置，也就是显示屏，渲染字符，处理事件。

如果您遇到无法关闭pygame终端的情况，那肯定是因为您没有正确处理事件队列。在这种情况下，您可以通过按下*Ctrl* + *C*来停止终端中的Python。

在跳转到下一节之前，我想讨论一下命令的简单但深奥的工作—pygame初始化—这是通过`pygame.init()`语句完成的。这只是一条简单的命令，但它执行的任务比我们想象的要多。顾名思义，这是pygame的初始化。因此，它必须初始化`pygame`包的每个子模块，即`display`、`rect`、`key`等。不仅如此，它还将加载所有必要的驱动程序和硬件组件的查询，以便进行通信。

如果您想更快地加载任何子模块，可以显式初始化特定的子模块，并避免所有不必要的子模块。例如，`pygame.music.init()`将只初始化pygame维护的子模块中的音乐子模块。对于本书中将要涵盖的大多数游戏，`pygame`模块需要超过三个子模块。因此，我们可以使用通用的`pygame.init()`方法来执行初始化。在进行了上述调用之后，我们将能够使用`pygame`模块的所有指定子模块。

初始化过程之后，开始创建显示屏是一个良好的实践。显示屏的尺寸取决于游戏的需求。有时，您可能需要为游戏提供全屏分辨率，以使其完全互动和吸引人。可以通过pygame表面对象来操作屏幕大小。在显示类上调用`set_mode`方法将返回表示整个窗口屏幕的对象。如果需要，还可以为显示屏设置标题；标题将添加到顶部导航栏中，与关闭按钮一起。以下代码表示了向游戏屏幕添加标题或游戏名称的方法：

```py
pygame.display.set_caption("My First Game")
```

现在，让我们谈谈传递给`set_mode`方法的参数。第一个——也是最重要的——参数是屏幕表面的尺寸。尺寸应该以元组的形式传递，即宽度和高度，这是强制性的。其他参数是可选的（在之前的程序中，我们甚至都没有使用它们）；它们被称为标志。我们需要它们是因为与宽度和高度相关的信息有时不足以进行适当的显示。

我们可能希望有**全屏**或**可调整大小**的显示，在这种情况下，标志可能更适合于显示创建。说到标志，它是一个可以根据情况打开和关闭的功能，有时候使用它可能会节省时间，相对而言。让我们来看一下下表中的一些标志，尽管我们不会很快使用它们，但在这里介绍它们可以避免在即将到来的部分中不必要的介绍：

| **标志** | **目的** |
| `FULLSCREEN` | 创建覆盖整个屏幕的显示。建议用于调试的窗口化屏幕。 |
| `DOUBLEBUF` | 用于创建*双缓冲*显示。强烈建议用于`HWSURFACE`或`OPENGL`，它模拟了3D显示。 |
| `HWSURFACE` | 用于创建硬件加速的显示，即使用视频卡内存而不是主内存（必须与`FULLSCREEN`标志结合使用）。 |
| `RESIZABLE` | 创建可调整大小的显示。 |
| `NOFRAME` | 无边框或边框的显示，也没有标题栏。 |
| `OPENGL` | 创建可渲染的OpenGL显示。 |

您可以使用按位或运算符将多个标志组合在一起，这有助于在屏幕表面方面获得更好的体验。为了创建一个双缓冲的OpenGL渲染显示，您可以将可选的标志参数设置为`DOUBLEBUF|OPENGL;`这里，(`|`)是按位`OR`运算符。即使pygame无法渲染我们要求的完美显示，这可能是由于缺乏适当的显卡，pygame将为我们在选择与我们的硬件兼容的显示方面做出决定。

游戏开发中最重要的一个方面是处理用户事件，通常是在游戏循环内完成的。在主游戏循环内，通常有另一个循环来处理用户事件——事件循环。事件是一系列消息，通知pygame在代码外部可以期待什么。事件可能是用户按键事件，也可能是通过第三方库传输的任何信息，例如互联网。

作为一组创建的事件被存储在队列中，并保留在那里，直到我们明确地处理它们。虽然在pygame的事件模块中有不同的函数提供了捕获事件的方法，`get()`是最可靠的，也很容易使用。在获取了各种操作后，我们可以使用pygame事件处理程序来处理它们，使用`pump`或`get`等函数。请记住，如果您只处理特定的操作，事件队列可能会混入其他您不感兴趣的表面事件。因此，必须明确地使用事件属性来处理事件，类似于我们在前面的示例中使用`QUIT`事件属性所做的。您还可以通过`eventType.__dict__`属性完全访问事件对象的属性。我们将在即将到来的*事件处理*部分中彻底学习它们。

在学习如何使用pygame升级我们之前制作的*snake*游戏之前，我们必须学习pygame的一些重要概念——*Pygame对象*、*绘制到屏幕*和*处理用户事件*。我们将逐一详细学习这些概念。我们将从*Pygame对象*开始，学习表面对象、创建表面和矩形对象。我们还将学习如何使用pygame绘制形状。

# Pygame对象

由内部使用类制作的`pygame`模块通过允许我们创建对象并使用它们的属性，使代码可读性和可重用性。正如我们之前提到的，`pygame`模块中定义了几个类，可以独立调用以执行独立的任务。例如，`draw`类可用于绘制不同的形状，如矩形、多边形、圆形等；`event`类可以调用`get`或`pump`等函数来处理用户事件。可以通过创建对象来进行这些调用，首先为每个操作创建对象。在本节中，您将探索这些概念，这将帮助您学习如何访问表面对象、矩形对象和绘制到屏幕。

创建自定义尺寸的空白表面最基本的方法是从pygame命名空间调用`Surface`构造函数。在创建`Surface`类的对象时，必须传递包含宽度和高度信息的元组。以下代码行创建了一个200x200像素的空白表面：

```py
screen_surface = pygame.Surface((200,200))
```

我们可以指定一些可选参数，最终会影响屏幕的视觉效果。您可以将标志参数设置为以下一个或多个参数之一：

+   `HWSURFACE`：创建硬件表面。在游戏的上下文中这并不是很重要，因为它是由pygame内部完成的。

+   `SRCALPHA`：它使用*alpha信息*来转换背景，这是指使屏幕背景透明的过程。它创建一个带有alpha转换的表面。alpha信息将使表面的一部分变为透明。如果您将其用作可选标志，您必须指定一个以上的强制参数，包括深度，并将其值分配为32，这是alpha信息的标准值。

此外，如果您想创建一个包含图像作为背景的表面，可以从`pygame`模块中调用`image`类。image类包含`load`方法，可以使用需要呈现的背景图像文件名作为参数进行调用。传递的文件名应该是完整的名称，带有其原始扩展名：

```py
background_surface = pygame.image.load(image_file_name.extension).convert()
```

从`image`类调用的load函数会从您的计算机中读取图像文件，然后返回包含图像的表面。在这里，屏幕尺寸将由图像大小确定。`Surface`对象的`convert()`成员函数将把指定的图像转换为显示屏支持的格式。

现在，让我们学习如何在单个表面内创建多个表面，通常称为子表面。

# 子表面

顾名思义，子表面是单个主表面内的嵌套表面列表。主表面可以被称为父表面。父表面可以使用`Surface`构造函数、`set_mode`或图像创建。当你在子表面上绘制时，它也会绘制在父表面上，因为子表面也是父表面的一部分。创建子表面很容易；你只需要从`Surface`对象调用`subsurface`方法，并且传递的参数应该指示要覆盖的`parent`类的位置。通常传递的坐标应该在父屏幕内创建一个小矩形。下面的代码显示了如何创建一个子表面：

```py
screen = Pygame.load("image.png")
screen.subsurface((0,0),(20,20))
screen.subsurface((20,0),(20,20))
```

你可以将这些子表面存储到数据结构中，比如字典，这样你就可以轻松地引用它们。你可以观察到传递给子表面方法的位置——它们与其他位置不同。点（0，0）总是表示子表面从父屏幕的左上角开始。

子表面有几种可用的方法，你可以从官方文档中了解到所有这些方法。其中最有用的方法之一是`get_parent()`，它返回子表面的父表面。如果没有使用`get_parent`方法调用任何子表面，它将返回`None`。

现在，我们将学习关于表面对象的下一个方法，这是在使用pygame制作任何游戏时经常使用的`blit`，它代表**位块传输**。

# `blit`你的对象

虽然术语*blitting*可能没有在牛津词典中定义，但在使用pygame制作游戏时具有更大的意义。`blit`通常被称为位边界块传输，或块信息传输，是一种将图像从一个表面复制到另一个表面的方法，通常是通过裁剪或移动。假设你有`Surfaceb`（你的屏幕），你想在屏幕上绘制一个形状，比如一个矩形。所以，你需要做的是绘制一个矩形，然后将缓冲区的矩形块传输到屏幕缓冲区。这个过程叫做*blitting*。当我们使用pygame制作游戏时，你会发现它被用来绘制背景、字体、角色，以及你能想象到的一切。

为了`blit`表面，你可以从结果表面对象（通常是显示对象）调用`blit`方法。你必须传递你的源表面，比如角色、动画和图像，以及要`blit`的坐标作为参数。与理论上听起来的相比，调用`blit`方法相当简单。下面的代码显示了如何在指定位置（0,0）`blit`背景图像，即屏幕的左上角：

```py
screen.blit(image_file_name.png, (0,0))
```

假设你有一组需要根据不同帧率渲染的图像。我们也可以使用`blit`方法来做到这一点。我们可以改变帧数的值，并在结果屏幕的不同区域`blit`图像，以制作图像的动画。这通常是在静态图像的情况下完成的。例如，我们将在下一章中使用Pygame创建flappy bird游戏的克隆。

在那个游戏中，我们需要在不同的位置（通常称为精灵）上`blit`管道和小鸟（flappy游戏的角色）的静态图像。这些精灵只是可以直接从互联网使用的图像，或者根据我们的需要自己制作的图像。以下代码展示了一种根据不同帧率`blit`图像的简单方法：

```py
screen.blit(list_of_images, (400, 300), (frame_number*10, 0, 100, 100))
```

在Flappy Bird游戏中，一个图像列表包含了鸟在飞行和下落两种姿势的图像。根据用户事件，我们将使用`blit`方法渲染它们。

在跳转到下一节之前，让我们了解一下可能微不足道但必须了解的*帧率*主题。这个术语经常被用作衡量游戏性能的基准。视频游戏中的帧率意味着你在屏幕上观察到的图像刷新或获取的次数。帧率是以**每秒帧数**或**FPS**（不要与**第一人称射击**混淆）来衡量的。

决定游戏帧率的因素有很多，但当代游戏玩家希望的是没有任何滞后或游戏运行缓慢。因此，更高的帧率总是更好。低帧率可能会在不合适的时候产生不幸的情况。一个例子可能是在用户能够跳跃或从一定高度跌落的游戏中；低帧率会导致系统滞后，并经常使屏幕*冻结*，使用户无法与游戏进行交互。许多现代游戏，例如第一人称射击游戏，如绝地求生和堡垒之夜，都是以达到大约60帧每秒的帧率为目标开发的。但在Pygame开发的简单游戏中，15到30帧每秒之间被认为是可以接受的。一些批评者认为30帧每秒以下会产生断断续续的动画和不真实的运动，但正如我们所知，pygame允许我们创建大多数迷你游戏。因此，15到30帧每秒之间对我们来说是足够的。

让我们进入下一节，我们将学习如何使用`pygame`绘制不同的形状。

# 使用pygame绘制模块进行绘制

最常用的模块之一是`draw`，它声明了许多方法，可以用来在游戏屏幕上绘制形状。使用此模块的目的是绘制线条、圆形和多边形，事实上，任何几何形状。你可能会想知道使用它的重要性——它有广泛的用途。我们可能需要创建形状以执行裁剪，或者将精灵或图像渲染到屏幕上。有时，您可能希望将这些形状用作游戏中的角色；像俄罗斯方块这样的游戏就是一个完美的例子。即使在创建游戏时您可能不会发现它非常有用，而是会使用精灵，但在测试游戏动画时可能会有所帮助。您不必去任何地方了解这些形状在游戏开发中的重要性；您可以观察到我们迄今为止创建的游戏。直到现在，在贪吃蛇游戏中，我们一直在使用简单的矩形形状来表示蛇的身体和头部。虽然这可能并不十分吸引人，在游戏的初期阶段，我们总是可以使用这样的形状来制作游戏。

使用pygame创建这样的形状比使用任何其他模块都要容易。我们可以调用绘制模块，以及函数名称。函数名称将是您想要绘制的形状的名称。例如，对于一个圆，我们将使用`pygame.draw.circle()`，对于一个矩形，我们将使用：`pygame.draw.rect()`。`pygame.draw`中函数的前两个参数是要绘制的表面，后面是要用来绘制的颜色。绘制函数的第一个参数是`Surface`对象，表示要在其上绘制的屏幕。下一个参数表示要在其上绘制形状的屏幕位置。

这三个参数对于每个几何形状都是强制性的，但最后一个取决于形状。该方法的最后一个参数表示在绘制这些形状时使用的数学量，例如圆的半径或直径。通常，传递的第三个参数应该表示坐标位置，以*x*和*y*坐标的形式，其中点（0, 0）表示屏幕左上角的位置。下表列出了在绘制模块中可用的方法数量，这些方法可用于绘制任何几何形状：

| **函数** | **描述** |
| `rect` | 绘制矩形 |
| `polygon` | 绘制正多边形（具有三个或更多封闭边的几何形状） |
| `line` | 绘制线条 |
| `lines` | 绘制多条线 |
| `circle` | 绘制圆 |
| `ellipse` | 绘制椭圆 |

举个例子，让我们使用`circle`方法并观察`pygame`绘图模块的运行情况。我们需要知道半径的值才能画一个圆。半径是从圆的中心到圆的边缘的距离，也就是圆的弧长。调用圆函数时应传递的参数是屏幕，代表表面对象；圆的颜色；圆应该被绘制的位置；最后是圆的半径。由于我们使用随机模块生成圆的半径的随机值，而不是给定特定值，以下代码创建了多个圆，具有随机宽度和随机位置，并且使用随机颜色。如果为每个参数输入特定值，将会绘制一个形状：

```py
import pygame as game
from pygame.locals import *
from random import *
import sys

game.init()
display_screen = game.display.set_mode((650, 470), 0, 32)
while True:
    for eachEvent in game.event.get():
        if eachEvent.type == QUIT:
            sys.exit()
    circle_generate_color = (randint(0,255), randint(0,255), 
                            randint(0,255))
 circle_position_arbitary = (randint(0,649), randint(0,469))
 circle_radius_arbitary = randint(1,230)
    game.draw.circle(display_screen, circle_generate_color, 
    circle_position_arbitary, circle_radius_arbitary)
    game.display.update()
```

从本章开始编写的代码在PyCharm Community IDE中，该IDE是在[第1章](0ef9574b-5690-454e-971f-85748021018d.xhtml)中下载的，*了解Python-设置Python和编辑器*。确保`pygame`安装在解释器的主目录上，以便在任何新创建的Python文件上都可以通用地使用`pygame`。

在使用PyCharm IDE时可以注意到的一个重要特性是，它可以为我们提供有关安装`pygame`模块的所有模块的信息。要确定`draw`模块中存在哪些函数，选择代码中的`circle`或`draw`关键字，然后在键盘上按*Ctrl* + *B*，这将将您重定向到`draw`模块的声明文件。

在谈论代码时，很容易理解。主要的三行代码被突出显示，以便您可以直接观察它们的重要性。大多数情况下，第三行调用`circle`方法，声明在`draw`模块中，它接受参数，屏幕对象，颜色，位置和半径以绘制一个圆。前面程序的输出将不断打印具有随机半径和随机颜色的圆，直到用户手动关闭屏幕，这是由于事件处理程序完成的，由`pygame.event.get`方法完成。

同样，您可以绘制许多形状和大小的多边形，范围可以从三边形到9999边形。就像我们使用`pygame.draw.circle`函数创建圆形一样，我们可以使用`pygame.draw.polygon`来绘制任何类型的多边形。对多边形函数的调用以点列表的形式作为参数，并将使用这些点绘制多边形形状。我们可以使用类似的方式使用特定的称谓绘制不同的几何形状。

在接下来的部分中，我们将学习使用`pygame`模块初始化显示屏和处理键盘和鼠标事件的不同方法。

# 初始化显示屏和处理事件

游戏开发人员主要将专注于如何使玩家感到参与其中，使游戏更具互动性。在这种情况下，必须将两个方面紧密联系在一起，即视觉上吸引人的显示和处理玩家的事件。我们不希望玩家被糟糕的显示屏和游戏运动中的滞后所压倒。在本节中，我们将讨论开发人员在制作游戏时必须考虑的两个主要方面：通过适应可用的可选参数来初始化显示的不同方式，以及处理用户操作事件，例如按下键盘键或鼠标按钮时。您想要创建的显示类型取决于您计划开发的游戏类型。

在使用`pygame`模块制作游戏时，您必须记住的一件事是，向游戏添加更多操作将影响游戏的流畅性，这意味着如果您向游戏中添加多个功能，游戏的互动性就会越来越差。因此，我们将主要专注于使用`pygame`模块制作迷你游戏。市场上还有更先进的Python模块可用于制作高功能游戏，我们将在接下来的章节中探讨它们。目前，我们将看到如何初始化显示，这是通过选择较低的分辨率来完成的，因为我们不希望游戏以任何方式滞后。

从现在开始制作的任何游戏都将具有固定和低分辨率，但您可以通过让用户选择自定义显示来进行实验。以下代码是创建pygame窗口的简单方法，我们之前编写的代码中也见过：

```py
displayScreen = pygame.display.set_mode((640, 480), 0, 32) #standard size
```

`set_mode()`的第一个参数将是屏幕的尺寸。元组中的值（640, 480）表示屏幕的高度和宽度。这个尺寸值将创建一个小窗口，与大多数桌面屏幕兼容。然而，我们可能会遇到一个情况，即游戏必须具有`FULLSCREEN`，而不是小屏幕。在这种情况下，我们可以使用一个可选参数，给出`FULLSCREEN`的值。显示全屏的代码看起来像这样：

```py
displayScreen = pygame.display.set_mode((640, 480), FULLSCREEN, 32)
```

然而，我们可能会观察到使用全屏模式与自定义显示之间的性能差异。在全屏模式下打开游戏将运行得更快，因为它不会与其他后台桌面屏幕进行交互，而另一个屏幕，具有自定义显示，可能会与您的机器上运行的其他显示屏合并。除此之外，在小屏幕上调试游戏比全屏游戏更容易，因为您应该考虑在全屏模式下关闭游戏的替代方法，因为关闭按钮将不可见。要检查PC支持的不同显示分辨率，您可以调用`list_modes()`方法，它将返回包含分辨率列表的元组，看起来像这样：

```py
>>> import pygame as p
>>> p.init()
>>> print(p.display.list_modes())
[(1366, 768), (1360, 768), (1280, 768), (1280, 720), (1280, 600), (1024, 768), (800, 600), (640, 480), (640, 400), (512, 384), (400, 300), (320, 240), (320, 200)]
```

有时，您可能会感到屏幕上显示的图像质量略有下降。这主要是由于显卡功能较少，无法提供您请求的图像颜色。这由`pygame`进行补偿，它将图像转换为适合您设备的图像。

在某些游戏中，您可能希望用户决定选择显示屏的大小。权衡的问题在于玩家选择高质量视觉还是使游戏运行顺畅。我们的主要目标将是处理事件，可以在可调整大小的屏幕和全屏之间切换。以下代码说明了在窗口化屏幕和全屏之间切换的方法。当用户在键盘上按下*F*时，它将在屏幕之间切换。

当你运行程序时，窗口屏幕和全屏之间的切换过程并不是即时的。这是因为`pygame`需要一些时间来检查显卡的特性，如果显卡不够强大，它会自行处理图像的质量：

```py
import pygame as p #abbreviating pygame module as p
from pygame.locals import *
import sys
p.init()
displayScreen = p.display.set_mode((640, 480), 0, 32)

displayFullscreen = False while True:
    for Each_event in p.event.get():
        if Each_event.type == QUIT:
            sys.exit()
        if Each_event.type == KEYDOWN:
            if Each_event.key == K_f:
                    displayFullscreen = not displayFullscreen
                    if displayFullscreen:
                        displayScreen = p.display.set_mode((640, 480), 
                                        FULLSCREEN, 32)
                    else:
                        displayScreen = p.display.set_mode((640, 480), 0, 32)

    p.display.update()
```

让我们逐行学习显示切换的过程：

1.  你必须从`pygame`模块开始导入。第二个导入语句将导入Pygame使用的常量。然而，它的内容会自动放置在`pygame`模块的命名空间中，我们可以使用`pygame.locals`来仅包含`pygame`常量。常量的例子包括：KEYDOWN，键盘`k_constants`等。

1.  你将在游戏开始时设置默认的显示模式。这个显示将是默认显示，每当你第一次运行程序时，当前定制的显示将被渲染。我们默认传递了一个(640, 480)的显示屏。

1.  要切换显示屏，你必须创建一个布尔变量`Fullscreen`，它将是`True`或`False`，基于这一点，我们将设置屏幕的模式。

1.  在主循环中，你必须处理键盘按键动作的事件。每当用户在键盘上按下*F*键时，我们将改变布尔变量的值，如果`FULLSCREEN`变量的值为`True`，我们必须将显示切换到全屏模式。额外的标志`FULLSCREEN`作为第二个参数添加到`add_mode()`函数中，深度为32。

1.  在else部分，如果全屏的值为`False`，你必须以窗口版本显示屏幕。相同的键*F*用于在窗口和全屏之间切换屏幕。

现在我们已经学会了如何使用不同的可用标志修改窗口可视化效果，让我们进入下一部分，我们将讨论接受用户输入和控制游戏，这通常被称为*处理用户事件*。

# 处理用户事件

在传统的PC游戏中，我们通常看到玩家只使用键盘来玩游戏。即使在今天，大多数游戏仍然完全依赖于键盘操作。随着游戏行业的发展，我们可以从多种输入设备接受用户输入，如鼠标和操纵杆。通常，鼠标用于处理动作，它可以给游戏画面提供全景视图。如果你玩过反恐精英或任何第一人称射击游戏，鼠标允许玩家在多个角度旋转视角，而键盘操作则处理玩家的移动，如向左移动、向右移动、跳跃等。键盘通常用于触发射击和躲避等动作，因为它的操作就像一个开关。开关只有两种可能性：打开或关闭；键盘按键也只有按下或未按下，这概括了处理键盘操作的技术。在典型的19世纪游戏中，我们曾经通过检查键盘的操作来生成游戏敌人。当用户不断按下键盘按键时，我们会生成更多的敌人。

鼠标和键盘这两种输入设备的组合非常适合这些游戏，因为鼠标能够处理方向运动，并且以平滑的方式进行操作。例如，当你玩第一人称射击游戏时，你可以使用键盘和鼠标来旋转玩家。当有敌人在你身后时，通常会使用鼠标快速旋转到那个位置，而不是使用键盘来旋转。

为了检测和监听所有的键盘按键，你必须使用`pygame.key`模块。这个模块能够检测任何键是否被按下，甚至支持方向运动。这个模块还能够处理任何键盘动作。基本上，有两种处理pygame中按键的方法：

+   通过处理按键按下事件，当键盘上的键被按下时触发。

+   通过处理键盘上释放键时触发的KEYUP事件。

虽然这些事件处理程序是检查按键的一个很好的方法，但处理键盘输入以进行移动并不适合它们。我们需要事先知道键盘键是否被按下，以便绘制下一帧。因此，直接使用`pygame.key`模块将使我们能够有效地处理键盘键。键盘的键（a-z，0-9和F1-F12）具有由pygame预定义的键常量。这些键常量可以被称为键码，用于唯一标识它们。键码总是以`K_`开头。对于每个可能的键，键码看起来像（`K_a`到`K_z`），（`K_0`到`K_9`），并包含其他常量，如`K_SPACE`，`K_LEFT`和`K_RETURN`。由于硬件不兼容性，pygame无法处理一些键盘键。这个异常在网上由几位开发者讨论过。你可能需要参考他们以更详细地了解这一点。

处理任何键盘动作的最基本方法是使用`pygame.key get_pressed`函数。这个方法非常强大，因为它为所有键盘常量分配布尔值，要么是`True`，要么是`False`。我们可以通过使用`if`条件来检查：键盘常量的值是`True`还是`False`？如果是`True`，显然是有键被按下了。`get_pressed`方法调用返回一个键常量的字典，字典的键是键盘的键常量，字典的值是布尔值，`dictionary_name[K_a] = True`。假设你正在制作一个程序，它将使用*up*作为跳跃按钮。你需要编写以下代码：

```py
import pygame as p
any_key_pressed = p.key.get_pressed()
if any_key_pressed[K_UP]:
    #UP key has been pressed
    jump()
```

让我们更详细地了解`pygame.key`模块。以下每个函数都将以不同的方式处理键盘键：

+   `pygame.key.get_pressed()`: 正如我们在前面的代码中看到的，这个方法返回一个包含键盘每个键的布尔值的字典。你必须检查键的值来确定它是否被按下。换句话说，如果键盘键的任何值被设置为`True`，则该索引的键被认为是被按下的。

+   `pygame.key.name()`: 正如其名称所示，这个方法调用将返回按下的键的名称。例如，如果我得到一个值为115的`KEY_UP`事件，你可以使用`key.name`来打印出这个键的名称，这种情况下是一个字符串，*s*。

+   `pygame.key.get_mods()`: 这将确定哪个修改键被按下。修改键是与*Shift*、*Alt*和*Ctrl*组合的普通键。为了检查是否有任何修改键被按下，你必须首先调用`get_mods`方法，然后跟着`K_MOD`。方法调用和常量之间用按位与运算符分隔，例如，`event.key == pygame.K_RIGHT`和`pygame.key.get_mods() & pygame`。`KMOD_LSHIFT`方法可用于检查左*Shift*键。

+   `pygame.key.set_mods()`: 你也可以临时设置修改键以观察修改键被按下的效果。要设置多个修改键，通常使用按位或运算符（|）将它们组合起来。例如，`pygame.key.set_mods(KMOD_SHIFT | KMOD_LSHIFT)`将设置SHIFT和LEFT *Shift*修改键。

+   `pygame.key.get_focused()`: 要从键盘获取每个按下的键，显示必须专注于键盘操作。这个方法调用将通过检查显示是否正在从系统接收键盘输入来返回一个布尔值。在游戏中可能有一个自定义屏幕的情况下，游戏屏幕没有焦点，因为你可能在使用其他应用程序；这将返回`False`，这意味着显示不活跃或没有专注于监听键盘操作。但在全屏显示模式下，你将完全专注于单个屏幕，在这种情况下，这个方法将始终返回`True`。

还有一些pygame按键功能，比如`get_repeat`和`set_repeat`，它们在你想要在键盘上连续按住任意键时发生重复动作的情况下非常有用。例如，打开记事本并连续按下*s*键。你会看到字符`s`会被打印多次。这个功能可以使用`pygame.key set_repeat`函数嵌入。这个函数将接受两个参数：延迟和间隔，单位为毫秒。

第一个延迟值是按键重复之前的初始延迟，而下一个间隔值是重复按键之间的延迟。您可以使用`调用set_repeat`方法并不带参数来禁用这些按键重复功能。默认情况下，当pygame被初始化时，按键重复功能是被禁用的。因此，您不需要手动禁用它。请访问以下网站以获取pygame官方文档，以了解更多关于pygame按键功能的信息：[https://www.pygame.org/docs/ref/key.html](https://www.pygame.org/docs/ref/key.html)。

您可以通过分配上、下、左或右键来使用键盘为游戏屏幕的精灵/图像/对象设置移动。直到现在，我们一直在使用不同的模块，如Python turtle和curses来做到这一点。然而，我们无法处理静态精灵或图像的移动。我们只处理了上、下、左、右和几何对象的按键事件，但现在pygame允许我们使用更复杂的图形并相应地处理它们。

我们可以分配任何键盘键来执行方向移动，但按照传统方法，我们可以适当地将光标键或箭头键分配为它们在键盘上的位置完美，这样玩家可以轻松游戏。但在一些复杂的多人游戏中，比如第一人称射击游戏，*A*、*W*、*S*和*D*键被分配用于方向移动。现在，你可能想知道为了使任何箭头键以这样的方式行为，可以用于方向移动，你需要做什么。只需回想一下向量的力量：这是一个数学概念，无论你使用什么语言或模块，都对游戏开发有用。移动任何几何形状和图像的技术是相同的；我们需要创建一个指向我们可能想要前进的方向的向量。表示游戏角色的位置非常简单：你可以用2D坐标(*x*, *y*)表示它，用3D坐标(*x*, *y*, *z*)表示它。然而，方向向量是必须添加到当前向量位置的单位量，以便转到下一帧。例如，通过按下键盘上的下键，我们必须向下移动，*x*位置不变，但*y*坐标增加一个单位。下表解释了四个方向的方向移动：

| **位置** | **方向向量** |
| 上 | (0, -1) |
| 下 | (0, 1) |
| 左 | (-1, 0) |
| 右 | (1, 0) |

我们可能还希望玩家允许对角线移动，如下图所示：

![](Images/8a263cf7-95c9-4277-ad7c-2dfd24d02cd3.png)

上面的插图代表了上和右键盘键的矢量运动。假设在游戏开始时，玩家位于位置(0, 0)，这意味着他们位于中心。现在，当用户按上（箭头键）键盘键时，将(0, 0)与上方向矢量(0, -1)相加，得到的矢量将是玩家的新位置。对角线移动（两个键的组合，这种情况下是上和右）将在玩家当前矢量位置上增加(0.707, -0.707)。我们可以使用这种矢量运动技术来为任何游戏对象提供方向运动，无论是精灵/静态图像还是几何形状。以下代码代表了使用pygame事件处理技术的矢量运动：

```py
import pygame as p
import sys
while True:
    for anyEvent in p.event.get():
        if anyEvent.type == QUIT:
            sys.exit()
        any_keys_pressed = p.key.get_pressed()
        movement_keys = Vector2(0, 0) #Vector2 imported from gameobjects
        #movement keys are diectional (arrow) keys
        if any_keys_pressed[K_LEFT]:
            movement_keys.x = –1
  elif any_keys_pressed[K_RIGHT]:
            movement_keys.x = +1
  if any_keys_pressed[K_UP]:
            movement_keys.y = -1
  elif any_keys_pressed[K_DOWN]:
            movement_keys.y = +1
  movement_keys.normalize() #creates list comprehension 
                                   [refer chapter 7]
```

尽管了解如何使物体在八个方向移动（四个基本方向和四个对角线移动）是值得的，但使用所有八个方向不会使游戏更加流畅。在假设中，使物体朝八个方向移动有点不自然。然而，现在的游戏允许玩家以360度的方式观察视图。因此，为了制作具有这种功能的游戏，我们可以使用键进行旋转运动，而不是使用八个键动作。为了计算旋转后的矢量，我们必须使用数学模块计算角度的正弦和余弦。角度的正弦负责*x*分量的运动，而余弦负责*y*分量的运动。这两个函数都使用弧度角；如果旋转角度是度数，你必须使用(`degree*pi/180`)将其转换为弧度：

```py
resultant_x = sin(angle_of_rotational_sprite*pi/180.0) 
#sin(theta) represents base rotation about x-axix
resultant_y = cos(angle_of_rotational_sprite*pi/180.0)
#cos(theta) represents height rotation about y-axis
new_heading_movement = Vector2(resultant_x, resultant_y)
new_heading_movement *= movement_direction
```

现在，让我们学习实现鼠标控制，并观察它如何在游戏开发中使用。

# 鼠标控制

拥有鼠标控制，以及键盘控制，如果你想使游戏更加互动，这是很方便的。有时，处理八个方向键是不够的，在这种情况下，你还必须处理鼠标事件。例如，在像flappy bird这样的游戏中，用户基本上必须能够使用鼠标玩，尽管在移动游戏中使用屏幕点击，在PC上，你必须能够提供鼠标操作。在显示屏中绘制鼠标光标非常简单；你只需要从`MOUSEMOTION`事件中获取鼠标的坐标。类似于键盘`get_pressed`函数，你可以调用`pygame.mouse.get_pos()`函数来获取鼠标的位置。鼠标移动在游戏中非常有用——如果你想使游戏角色旋转，或者制作一个屏幕点击游戏，甚至如果你想上下查看游戏屏幕。

为了理解处理鼠标事件的方法，让我们看一个简单的例子：

```py
import pygame as game #now instead of using pygame, you can use game

game.init()
windowScreen = game.display.set_mode((300, 300))
done = False   # Draw Rect as place where mouse pointer can be clicked RectangularPlace = game.draw.rect(windowScreen, (255, 0, 0),(150, 150, 150, 150))
game.display.update()
# Main Loop while not done:
    # Mouse position and button clicking.
  position = game.mouse.get_pos()
    leftPressed, rightPressed, centerPressed = game.mouse.get_pressed() #checking if left mouse button is collided with rect place or not if RectangularPlace.collidepoint(position) and leftPressed:
        print("You have clicked on a rectangle")
    # Quit pygame.
  for anyEvent in game.event.get():
        if anyEvent.type == game.QUIT:
            done = True
```

我已经突出了代码的一些重要部分。重点主要放在帮助我们理解鼠标事件实现的那些部分上。让我们逐行看代码：

1.  首先，你必须定义一个对象——一个将有鼠标事件监听器设置以捕获它的区域。在这种情况下，你必须使用`pygame.draw.rect`方法调用将区域声明为矩形。

1.  在主循环内，你必须使用`pygame.mouse.get_pos()`函数获取鼠标的位置，这将表示当前光标坐标。

1.  然后，你必须从`pygame.mouse`模块调用`get_pressed()`方法。将返回一个布尔值列表。对于左、右或中间，布尔值`True`表示在特定实例中，特定鼠标按钮被按下，而其余两个没有。在这里，我们捕获了三个鼠标按钮的布尔值。

1.  现在，要检查用户是否按在矩形内，你必须调用`collidepoint`方法并向其传递一个位置值。位置表示当前光标位置。如果鼠标在当前位置点击，`pressed1`将为`True`。

1.  当这两个语句都为`True`时，您可以相应地执行任何操作。请记住，即使您在窗口屏幕中点击了，这个程序也不会打印消息，因为它不属于矩形的一部分。

与`pygame.key`模块类似，让我们详细了解`pygame.mouse`模块。该模块包含八个函数：

+   `pygame.mouse.get_rel()`: 它将以元组形式返回相对鼠标移动，包括*x*和*y*的相对移动。

+   `pygame.mouse.get_pressed()`: 它将返回三个布尔值，代表鼠标按钮，如果任何一个为`True`，则相应的按钮被视为按下。

+   `pygame.mouse.set_cursor()`: 它将设置标准光标图像。这很少需要，因为通过在鼠标坐标上绘制图像可以获得更好的效果。

+   `pygame.mouse.get_cursor()`: 它执行两个不同的任务：首先，它设置光标的标准图像，其次，它获取关于系统光标的确定性数据。

+   `pygame.mouse.set_visible()`: 它改变标准鼠标光标的可见性。如果为`False`，光标将不可见。

+   `pygame.mouse.get_pos()`: 它返回一个元组，包含鼠标在画布中点击位置的*x*和*y*值。

+   `pygame.mouse.set_pos()`: 它将设置鼠标位置。它接受一个元组作为参数，其中包含画布中*x*和*y*的坐标。

+   `pygame.mouse.get_focused()`: 这个布尔函数的结果基于窗口屏幕是否接收鼠标输入的条件。它类似于`key.get_focused`函数。当pygame在当前窗口屏幕中运行时，窗口将接收鼠标输入，但只有当pygame窗口被选中并在显示器的最前面运行时才会接收。如果另一个程序在后台运行并被选中，那么pygame窗口将无法接收鼠标输入，这个方法调用的输出将是`False`。

您可能玩过一些飞机或坦克游戏，鼠标用作瞄准设备，键盘用于移动和射击动作。这些游戏非常互动。因此，您应该尝试制作一个可以尽可能结合这两种事件的游戏。这两种类型的事件非常有用，对于任何游戏开发都很重要。我建议您花时间尝试这些事件。如果可能的话，尝试只使用几何对象制作自己的游戏。现在，我们将学习如何使用pygame和我们自己的精灵制作游戏。

这个游戏将是前一章中由turtle模块制作的贪吃蛇游戏的修改版本。所有的概念都是一样的，但是我们将制作外观吸引人的角色，并且我们将使用pygame处理事件。

# 对象渲染

计算机以颜色网格的形式存储图像。通常，RGB（红色、绿色和蓝色）足以提供像素的信息。但除了RGB值之外，在处理pygame游戏开发时，图像的另一个组成部分也很有用，那就是alpha信息（通常称为属性组件）。alpha信息代表图像的透明度。这些额外的信息非常有用；在pygame的情况下，通常我们会激活alpha属性，然后将一张图像绘制或放置在另一张图像的顶部。通过这样做，我们可以看到部分背景。通常，我们会使用GIMP等第三方软件来使图像的背景透明。

除了知道如何使图像的背景透明之外，我们还必须知道如何将它们导入到我们的项目中，以便我们可以使用它们。将任何静态图像或精灵导入Python项目非常容易，pygame使其变得更加容易。我们有一个图像模块，它提供了一个load方法来导入图像。在调用load方法时，您必须传递一个带有完整文件名的图像，包括扩展名。以下代码表示了一种将图像导入Python项目的方法：

```py
gameBackground = pygame.image.load(image_filename_for_background).convert()
Image_Cursor = pygame.image.load(image_filename_mouseCursor).convert_alpha()
```

您想要导入游戏项目的图像应该与游戏项目所在的目录相同。例如，如果Python文件保存在snake目录中，则Python文件加载的图像也应保存在snake目录中。

在图像模块中，load函数将从硬盘加载文件并返回一个包含要加载的图像的新生成的表面。对`pygame.image.load`的第一次调用将读取图像文件，然后立即调用`convert`方法，将图像转换为与我们的显示器相同的格式。由于图像和显示屏的转换处于相同的深度级别，因此绘制到屏幕上相对较快。

第二个语句是加载鼠标光标。有时，您可能希望将自定义鼠标光标加载到游戏中，第二行代码就是这样做的方法。在加载`mouse_cursor`的情况下，使用`convert_alpha`而不是convert函数。这是因为鼠标光标的图像包含有关透明度的特殊信息，称为*alpha信息*，并使图像的一部分变得不可见。通过禁用alpha信息，我们的鼠标光标将被矩形或正方形形状包围，从而使光标看起来不太吸引人。基本上，alpha信息用于表示将具有透明背景的图像。

现在我们已经学会了如何将图像导入Python项目，让我们学习如何旋转这些图像。这是一种非常有用的技术，因为在构建游戏时，我们可能需要按一定角度旋转图像，以使游戏更具吸引力。例如，假设我们正在制作一个贪吃蛇游戏，我们正在使用一张图像作为蛇头。现在，当用户在键盘上按下“上”键时，蛇头应该旋转，并且必须平稳地向上移动。这是通过`pygame.transform`模块完成的。`Rotate`方法可以从transform模块中调用以便进行旋转。旋转方法接受从`image.load()`函数加载的图像表面，并指定旋转的角度。通常，转换操作会调整像素的大小或移动部分像素，以使表面与显示屏兼容：

```py
pygame.transform.rotate(img, 270) #rotation of image by 270 degree
```

在我们开始开发自己的视觉吸引人的贪吃蛇游戏之前，您必须了解Pygame `time`模块。点击此链接了解更多信息：[https://www.pygame.org/docs/ref/time.html#pygame.time.Clock](https://www.pygame.org/docs/ref/time.html#pygame.time.Clock)。`Pygame.time`模块用于监控时间。时间时钟还提供了几个函数来帮助控制游戏的帧速率。帧速率是连续图像出现在显示屏上的速率或频率。每当调用时间模块的`Clock()`构造函数时，它将创建一个对象，该对象可用于跟踪时间。Pygame开发人员在Pygame时间模块内部定义了各种函数。但是，我们只会使用`tick`方法，它将更新时钟。

`Pygame.time.Clock.tick()`应该在每帧调用一次。在函数的两次连续调用之间，`tick()`方法跟踪每次调用之间的时间（以毫秒为单位）。通过每帧调用`Clock.tick(60)`，程序被限制在60 FPS的范围内运行，并且即使处理能力更高，也不能超过它。因此，它可以用来限制游戏的运行速度。这在由Pygame开发的游戏中很重要，因为我们希望游戏能够平稳运行，而不是通过CPU资源来补偿。每秒帧数（帧速率）的值可以在由Pygame开发的游戏中的游戏中任何地方从15到40。

现在，我们已经有足够的信息来使用Pygame制作我们自己的游戏，其中将有精灵和游戏角色的平滑移动。我们将在下一节中开始初始化显示。我们将使用Pygame模块更新我们的贪吃蛇游戏。

# 初始化显示

初始化显示非常基础；您可以始终通过导入必要的模块并在`set_mode()`方法中提供显示的特定尺寸来创建窗口化屏幕。除此之外，我们将声明一个主循环。请参考以下代码以观察主循环的声明：

```py
import pygame as game
from sys import exit
game.init()

DisplayScreen = game.display.set_mode((850,650))
game.display.set_caption('The Snake Game') #game title

game.display.update()

gameOver = False

while not gameOver:
    for anyEvent in game.event.get():
        print(event)
        exit()

game.quit()
quit()
```

初始化后，您可以运行程序检查一切是否正常。如果出现“没有pygame模块”的错误，请确保您按照上述步骤在PyCharm IDE上安装Pygame。现在，我们将学习如何使用颜色。

# 使用颜色

计算机颜色的基本原理是*颜色相加*，这是一种将三种基本颜色相加以创建新颜色的技术。三种基本颜色是红色、绿色和蓝色，通常称为RGB值。每当Pygame需要将任何颜色添加到游戏中时，您必须将其传递给三个整数的元组，每个整数分别对应红色、绿色或蓝色。

将整数值传递给元组的顺序很重要，对整数进行微小的更改会导致不同的颜色。颜色的每个组件的值必须在0到255之间，其中255表示颜色具有绝对强度，而0表示该颜色根本没有强度。例如，(255, 0, 0)表示红色。以下表格指示了不同颜色的颜色代码：

| 颜色名称 十六进制码#RRGGBB 十进制码(R,G,B) |
| --- |
| 黑色 #000000 (0,0,0) |
| 白色 #FFFFFF (255,255,255) |
| 红色 #FF0000 (255,0,0) |
| 酸橙色 #00FF00 (0,255,0) |
| 蓝色 #0000FF (0,0,255) |
| 黄色 #FFFF00 (255,255,0) |
| 青色/水绿色 #00FFFF (0,255,255) |
| 洋红/紫红 #FF00FF (255,0,255) |

现在，让我们为我们的贪吃蛇游戏项目添加一些颜色：

```py
white = (255,255,255)
color_black = (0,0,0)
green = (0,255,0)
color_red = (255,0,0)

while not gameOver:
    #1 EVENT GET
    DisplayScreen.fill(white) #BACKGROUND WHITE
    game.display.update()
```

现在，在下一节中，我们将学习如何使用`pygame`模块创建游戏对象。

# 制作游戏对象

为了开始创建游戏对象，我们不会直接使用贪吃蛇精灵或图像。相反，我们将从使用一个小矩形框开始，然后我们将用贪吃蛇图像替换它。这在大多数游戏中都需要做，因为我们必须在游戏开发的开始测试多个事物，比如帧速率、碰撞、旋转等。在处理所有这些之后，很容易将图像添加到pygame项目中。因此，在本节中，我们将制作类似矩形框的游戏对象。我们将制作贪吃蛇的头部和身体，它将是一个小矩形框。我们最初将为贪吃蛇的头部制作一个盒子，另一个为食物，然后为其添加颜色：

```py
while not gameOver:
    DisplayScreen.fill(white) #background of game 
    game.draw.rect(DisplayScreen, color_black, [450,300,10,10]) #1\. snake
    #two ways of defining rect objects
    DisplayScreen.fill(color_red, rect=[200,200,50,50]) #2\. food
```

现在我们将为`game`对象添加移动。在之前的章节中，我们已经谈论了很多这些内容，比如在处理方向移动时使用向量：

```py
change_x = 300
change_y = 300
while not gameOver:
    for anyEvent in game.event.get():
        if anyEvent.type == game.QUIT:
            gameOver = True
        if anyEvent.type == game.KEYDOWN:
            if anyEvent.key == game.K_LEFT:
                change_x -= 10
            if anyEvent.key == game.K_RIGHT:
                change_x += 10

    DisplayScreen.fill(white)
    game.draw.rect(DisplayScreen, black, [change_x,change_y,10,10])
    game.display.update()
```

在先前的代码中，`change_x`和`change_y`表示蛇的初始位置。每当开始玩我们的游戏时，蛇的默认位置将是(`change_x`, `change_y`)。通过按下左键或右键，我们改变它的位置。

当你此刻运行游戏时，你可能会观察到你的游戏只会移动一步，当你按下并立即释放键盘键时，游戏会立即停止。这种异常行为可以通过处理多个运动来纠正。在这种情况下，我们将创建`lead_x_change`，这将根据主`change_x`变量的变化。请记住，我们没有处理上下键事件；因此，不需要`lead_y_change`。

```py
lead_x_change = 0

while not gameOver:
    for anyEvent in game.event.get():
        if anyEent.type == game.QUIT:
            gameOver = True
        if anyEvent.type == game.KEYDOWN:
            if anyEvent.key == game.K_LEFT:
                lead_x_change = -10
            if anyEvent.key == game.K_RIGHT:
                lead_x_change = 10

    change_x += lead_x_change
    DisplayScreen.fill(white)
    game.draw.rect(DisplayScreen, black, [change_x,change_y,10,10])
    game.display.update()
```

在新的代码行中，我们添加了额外的信息`lead_x_change`，它将被称为*x*坐标的变化，每当用户按下左右键盘键时，蛇就会自动移动。代码的突出部分(`change_x += lead_x_change`)负责使蛇持续移动，即使用户不按任何键（蛇游戏的规则）。

现在，当你按下一个键时，你可能会在游戏中看到另一种不寻常的行为。在我的情况下，我运行了我的游戏，当我开始按下左键时，蛇开始快速地连续地从左到右移动。这是由于帧速率的宽松性；我们现在必须明确指示游戏的帧速率，以限制游戏的运行速度。我们将在下一节中介绍这个问题。

# 使用帧速率概念

这个话题对我们来说并不陌生；我已经尽我最大的努力尽早介绍这个话题。在讨论时钟模块时，我们也学习了帧速率的概念。在本节中，我们将看到帧速率的概念在实际中的应用。到目前为止，我们已经制作了一个可以运行的游戏，但它在移动上没有任何限制。它在一个方向或另一个方向上持续移动，速度很快，我们当然不希望这样。我们真正想要的是使蛇持续移动，但在一定的帧速率内。我们将使用`pygame.time.Clock`来创建一个对象，它将跟踪我们游戏的时间。我们将使用`tick`函数来更新时钟。tick方法应该每帧调用一次。通过每帧调用`Clock.tick(15)`，游戏将永远不会以超过15 FPS的速度运行。

```py
clock = game.time.Clock()
while not gameOver:
    #event handling
    #code from preceding topic
    clock.tick(30) #FPS
```

重要的是要理解FPS并不等同于游戏中精灵的速度。开发者制作游戏的方式是可以在高端和低端设备上玩。你会发现在低配置的机器上游戏有点迟缓和抖动，但两种设备上的精灵或角色都会以平均速度移动。我们并不否认使用基于时间的运动游戏的机器，帧速率慢会导致视觉体验不佳，但它不会减慢动作的速度。

因此，为了制作一个视觉上吸引人的游戏，甚至在普及设备上也兼容，通常最好将帧速率设置在20到40 FPS之间。

在接下来的部分，我们将处理剩余的方向运动。处理这些运动并没有什么不同；它们可以通过矢量运动来处理。

# 处理方向运动

我们已经处理了*x*轴变化的运动。现在，让我们添加一些代码来处理*y*轴的运动。为了使蛇持续移动，我们必须使`lead_y_change`，它代表连续添加到当前位置的方向量，即使用户不按任何键盘键：

```py
lead_y_change = 0
while not gameOver:
        if anyEvent.type == game.KEYDOWN:
            if anyEvent.key == game.K_LEFT:
                lead_x_change = -10
                lead_y_change = 0
            elif anyEvent.key == game.K_RIGHT:
                lead_x_change = 10
                lead_y_change = 0
            elif anyEvent.key == game.K_UP:
                lead_y_change = -10
                lead_x_change = 0
            elif anyEvent.key == game.K_DOWN:
                lead_y_change = 10
                lead_x_change = 0  

    change_x += lead_x_change
    change_y += lead_y_change
```

现在我们已经处理了蛇的每种可能的运动，让我们为蛇游戏定义边界。`change_x`和`change_y`的值表示头部的当前位置。如果头部撞到边界，游戏将终止。

```py
while not gameOver:
    if change_x >= 800 or change_x < 0 or change_y >= 600 or change_y < 0:
            gameOver = True
```

现在，我们将学习另一个编程概念，这将使我们的代码看起来更清晰。到目前为止，我们已经为许多组件使用了数值，比如高度、宽度、FPS等。但是如果你必须更改其中一个这些值会发生什么？在搜索代码和再次调试时会有很多开销。现在，我们可以创建常量变量，而不是直接使用这些数值，我们将这些值存储在其中，并在需要时检索它们。这个过程叫做*去除硬编码*。让我们为每个这些数值创建一个合适的名称的变量。代码应该看起来像这样：

```py
#variable initialization step
import pygame as game

game.init()

color_white = (255,255,255)
color_black = (0,0,0)
color_red = (255,0,0)

#display size
display_width = 800 
display_height = 600

DisplayScreen = game.display.set_mode((display_width,display_height))
game.display.set_caption('') #game title

gameOver = False

change_x = display_width/2
change_y = display_height/2

lead_x_change = 0
lead_y_change = 0

objectClock = game.time.Clock()

pixel_size = 10 #box size 
FPS = 30 #frame rate
```

在变量初始化步骤中去除硬编码后，我们将转向主游戏循环。以下代码表示主游戏循环（在初始化步骤之后添加）：

```py
#main loop
while not gameOver:
    for anyEvent in game.event.get():
        if anyEvent.type == game.QUIT:
            gameOver = True
        if anyEvent.type == game.KEYDOWN:
            if anyEvent.key == game.K_LEFT:
                lead_x_change = -pixel_size
                lead_y_change = 0
            elif anyEvent.key == game.K_RIGHT:
                lead_x_change = pixel_size
                lead_y_change = 0
            elif anyEvent.key == game.K_UP:
                lead_y_change = -pixel_size
                lead_x_change = 0
            elif anyEvent.key == game.K_DOWN:
                lead_y_change = pixel_size
                lead_x_change = 0

       #step 3: adding logic which will check if snake hit boundary or not
```

现在我们已经添加了处理用户事件的方法到主循环中，让我们重构代表逻辑的代码，比如当蛇撞到游戏边界时会发生什么，或者当蛇改变速度时会发生什么。在处理用户事件后，应该在主循环中添加以下代码：

```py
 if change_x >= display_width or change_x < 0 or change_y >= display_height 
                or change_y < 0:
        gameOver = True

    change_x += lead_x_change
    change_y += lead_y_change
    DisplayScreen.fill(color_white)
    game.draw.rect(DisplayScreen, color_black, 
      [change_x,change_y,pixel_size,pixel_size])
    game.display.update()

    objectClock.tick(FPS)
```

前面的所有代码已经简要描述过了，我们在前面的三个代码块中实际上是将变量重构为一些有意义的名称，以消除硬编码；例如，为显示宽度添加一个变量名，为颜色代码添加一个变量名，等等。

在接下来的部分，我们将在屏幕上添加一个食物字符，并创建一些逻辑来检查蛇是否吃了苹果。

# 添加食物到游戏中

在屏幕上添加一个字符非常简单。首先，为字符创建一个位置，最后，在该位置上`blit`字符。在蛇游戏中，食物必须在任意位置渲染。因此，我们将使用随机模块创建随机位置。我创建了一个新的函数`gameLoop()`，它将使用前面部分的代码。我使用`apple`作为食物。稍后，我将为它添加一个苹果图像。以下代码定义了游戏的主循环：

```py
def MainLoopForGame():
    global arrow_key #to track which arrow key user pressed

    gameOver = False
    gameFinish = False
    #initial change_x and change_y represent center of screen
    #initial position for snake
    change_x = display_width/2
    change_y = display_height/2

    lead_x_change = 0
    lead_y_change = 0
```

在为游戏显示和角色定义一些初始值之后，让我们添加一些逻辑来为蛇游戏添加苹果（食物）（这应该在`MainLoopForGame`函数内）。

```py
 XpositionApple = round(random.randrange(0, display_width-pixel_size))
 YpositionApple = round(random.randrange(0, display_height-pixel_size))
```

这两行代码将为*x*和*y*创建随机位置。确保导入随机模块。

接下来，我们需要在`MainLoopForGame`函数内定义主游戏循环。添加到主循环内的代码将处理多个事情，比如处理用户事件，绘制游戏角色等。让我们从以下代码中获取用户事件开始：

```py
 while not gameOver:

        while gameFinish == True:
            DisplayScreen.fill(color_white)
            game.display.update()

            #game is object of pygame
            for anyEvent in game.event.get():
                if anyEvent.type == pygame.KEYDOWN:
                    if anyEvent.key == pygame.K_q:
                        gameOver = True
                        gameFinish = False
                    if anyEvent.key == pygame.K_c:
                        MainLoopForGame()
```

前面的代码将很容易理解，因为我们在本章的前面已经做过这个。我们首先用白色填充游戏的背景屏幕，然后使用`pygame`模块的事件类获取事件。我们检查用户是否输入了`q`键，如果是，我们就退出游戏。同样，既然我们从用户那里得到了一个事件，让我们处理使蛇游戏移动的事件，比如左右箭头键。在获取用户事件后，应该添加以下代码：

```py
 #event to make movement for snake based on arrow keys
        for anyEvent in game.event.get():
            if anyEvent.type == game.QUIT:
                gameOver = True
            if anyEvent.type == game.KEYDOWN:
                if anyEvent.key == game.K_LEFT:
                    arrow_key = 'left'
                    lead_x_change = -pixel_size
                    lead_y_change = 0
                elif anyEvent.key == game.K_RIGHT:
                    arrow_key = 'right'
                    lead_x_change = pixel_size
                    lead_y_change = 0
                elif anyEvent.key == game.K_UP:
                    arrow_key = 'up'
                    lead_y_change = -pixel_size
                    lead_x_change = 0
                elif anyEvent.key == game.K_DOWN:
                    arrow_key = 'down'
                    lead_y_change = pixel_size
                    lead_x_change = 0
```

先前的代码已经编写好了，所以确保你按照程序的顺序进行。参考提供的代码资产[https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter11](https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter11)。让我们把剩下的代码添加到主循环中，处理渲染蛇食物的逻辑。在处理用户事件之后，应该添加以下代码：

```py
         if change_x >= display_width or change_x < 0 or change_y >= 
                        display_height or change_y < 0:
            gameFinish = True

        change_x += lead_x_change
        change_y += lead_y_change
        DisplayScreen.fill(color_white)
        Width_Apple = 30
        game.draw.rect(DisplayScreen, color_red, [XpositionApple, 
            YpositionApple, Width_Apple, Width_Apple])
        game.draw.rect(DisplayScreen, color_black, 
            [change_x,change_y,pixel_size, pixel_size])
        game.display.update()

        objectClock.tick(FPS)

    game.quit()
    quit()

MainLoopForGame()
```

在代码的突出部分，我们将绘制一个红色的矩形，并将其渲染在由`pixel_size= 10`的高度和宽度的随机模块定义的位置。

现在我们已经为蛇添加了食物，让我们制作一个函数，使蛇的身体增长。到目前为止，我们只处理了蛇的头部；现在是时候制作一个函数，通过单位块来增加蛇的身体。请记住，只有在蛇吃了食物之后才会调用这个函数：

```py
def drawSnake(pixel_size, snakeArray):
    for eachSegment in snakeArray:
        game.draw.rect(DisplayScreen, color_green  [eachSegment[0],eachSegment[1],pixel_size, pixel_size])

```

在主游戏循环中，我们必须声明多个东西。首先，我们将声明`snakeArray`，它将包含蛇的身体。游戏开始时，蛇的长度为1。每当蛇吃食物时，我们将增加它：

```py
def MainLoopForGame():
 snakeArray = []
 snakeLength = 1

    while not gameOver:
        head_of_Snake = []
 #at the beginning, snake will have only head
 head_of_Snake.append(change_x)
 head_of_Snake.append(change_y)

        snakeArray.append(head_of_Snake)

        if len(snakeArray) > snakeLength:
            del snakeArray[0] #deleting overflow of elements

        for eachPart in snakeArray[:-1]:
            if eachPart == head_of_Snake:
                gameFinish = True #when snake collides with own body

        drawSnake(pixel_size, snakeArray)  
        game.display.update()
```

变量的名称告诉你一切你需要知道的。我们以前做过很多次，也就是为蛇的头部制作列表，并检查它是否与蛇的身体发生碰撞。蛇方法调用`pixel_size`，这是蛇的尺寸，以及包含与蛇身体相关的位置列表的蛇列表。蛇将根据这些列表进行`blit`，通过在`snake`函数内定义的绘制语句。

接下来，我们需要定义逻辑来使蛇吃食物。这个逻辑已经被反复使用，在pygame的情况下也不例外。每当蛇的头部位置与食物位置相同时，我们将增加蛇的长度，并在一个新的随机位置生成食物。确保在更新显示后，在主游戏循环中添加以下代码：

```py
#condition where snake rect is at the top of apple rect  
if change_x > XpositionApple and change_x < XpositionApple + Width_Apple or change_x + pixel_size > XpositionApple and change_x + pixel_size < XpositionApple + Width_Apple:

      if change_y > YpositionApple and change_y < YpositionApple + 
        Width_Apple:
                #generate apple to new position
                XpositionApple = round(random.randrange(0, 
                                 display_width-pixel_size))
                YpositionApple = round(random.randrange(0, 
                                 display_height-pixel_size))
                snakeLength += 1

      elif change_y + pixel_size > YpositionApple and change_y + pixel_size 
            < YpositionApple + Width_Apple:

                XpositionApple = round(random.randrange(0, display_width-
                                 pixel_size))
                YpositionApple = round(random.randrange(0, display_height-
                                 pixel_size))
                snakeLength += 1
```

由于我们能够添加一些逻辑来检查蛇是否吃了食物，并做出相应的反应，现在是时候为角色添加精灵或图像了。正如我们之前提到的，我们将添加我们自己的蛇头，而不是使用沉闷的矩形形状。让我们开始创建一个。

# 添加蛇的精灵

最后，我们可以开始使我们的游戏更具吸引力——我们将制作蛇的头。我们不需要额外的知识来为游戏角色创建图像。你也可以从互联网上下载图像并使用它们。然而，在这里，我将向你展示如何为自己创建一个，并如何在我们的蛇游戏中使用它。

按照以下步骤，逐行进行：

1.  打开任何*绘图*应用程序，或者在搜索栏中搜索绘图，然后打开应用程序。

1.  按下*Ctrl* + *W*来调整和扭曲你选择的图片，或者直接使用上方菜单栏的调整按钮。这将打开一个新的调整窗口。可以按百分比和像素进行调整。使用百分比调整并保持20x20的纵横比，即水平：20，垂直：20。

1.  之后，你会得到一个绘制屏幕。选择你想要制作的蛇头的颜色。在制作游戏时，我们创建了一个绿色的蛇身体；因此，我也会选择绿色作为蛇头的颜色。我会使用画笔画出类似以下图片的东西。如果你愿意，你可以花时间创作一个更好的。完成后，保存文件：

![](Images/38c3a904-cd84-46db-9eb7-458cab37f736.png)

1.  现在，你必须使图像的背景透明。你也可以使用一些在线工具，但我将使用之前提到过的GIMP软件。你必须从官方网站上下载它。它是开源的，可以免费使用。去网站上下载GIMP：[https://www.gimp.org/downloads/](https://www.gimp.org/downloads/)。

1.  用GIMP软件打开你之前制作的蛇头。从最上面的菜单中选择图层选项卡，选择透明度，然后点击添加Alpha通道。这将添加一个通道，可以用来使我们图像的背景透明。

1.  从菜单屏幕中点击颜色选项卡。将会出现一个下拉菜单。点击颜色到Alpha，使背景透明。将该文件导出到与您的Python文件存储在同一目录中。

现在我们有了蛇头的精灵，让我们在Python文件中使用`blit`命令来渲染它。如你所知，在使用任何图像之前，你必须导入它。由于我已经将蛇头图像保存在与Python文件相同的目录中，我可以使用`pygame.image.load`命令：

```py
image = game.image.load('snakehead.png')
```

在`drawSnake`方法的主体内，你必须blit图像；就像这样：

```py
DisplayScreen.blit(image, (snakeArray[-1][0], snakeArray[-1][1]))
```

现在，当你运行游戏时，你会观察到一个奇怪的事情。当我们按下任何一个箭头键时，头部不会相应地旋转。它将保持在默认位置。因此，为了使精灵根据方向的移动而旋转，我们必须使用`transform.rotate`函数。观察蛇的方法，因为它有一种方法可以在没有旋转的情况下`blit`图像。现在，我们将添加几行代码，使精灵旋转：

```py
def drawSnake(pixel_size, snakeArray):

 if arrow_key == "right":
 head_of_Snake = game.transform.rotate(image, 270) #making rotation of 270 

 if arrow_key== "left":
 head_of_Snake = game.transform.rotate(image, 90)

 if arrow_key== "up":
 head_of_Snake = image #default

 if arrow_key== "down":
 head_of_Snake = game.transform.rotate(image, 180)

 DisplayScreen.blit(head_of_Snake, (snakeArray[-1][0], snakeArray[-1][1]))
 for eachSegment in snakeArray[:-1]:
 game.draw.rect(DisplayScreen, color_green,[eachSegment[0],eachSegment[1], 
 pixel_size, pixel_size])
```

现在，不再使用苹果的矩形框，让我从互联网上下载一个苹果的样本，以PNG的形式（透明背景），也`blit`它：

```py
appleimg = game.image.load('apple.png') 
#add apple.png file in same directory of python file
while not gameOver:
    #code must be added before checking if user eats apple or not
    DisplayScreen.blit(appleimg, (XpositionApple, YpositionApple))
```

让我们运行游戏并观察输出。虽然蛇头看起来更大了，但我们可以随时调整它的大小：

![](Images/ade1f80d-3bf7-4b01-84e0-f80900e83a5f.png)

在下一节中，我们将学习如何为我们的游戏添加一个菜单。菜单是每次打开游戏时看到的屏幕，通常是一个欢迎屏幕。

# 为游戏添加一个菜单

为任何游戏添加一个介绍屏幕需要我们具备使用`pygame`模块处理字体的知识。pygame提供了一个功能，使我们可以使用不同类型的字体，包括改变它们的大小的功能。`pygame.font`模块用于向游戏添加字体。字体用于向游戏屏幕添加文本。由于介绍或欢迎屏幕需要玩家显示一个包含字体的屏幕，我们必须使用这个模块。调用`SysFont`方法向屏幕添加字体。`SysFont`方法接受两个参数：第一个是字体的名称，第二个是字体的大小。以下一行代码初始化了相同字体的三种不同大小：

```py
font_small = game.font.SysFont("comicsansms", 25)
font_medium = game.font.SysFont("comicsansms", 50)
font_large = game.font.SysFont("comicsansms", 80)
```

我们将首先使用`text_object`函数创建一个表面，用于小号、中号和大号字体。文本对象函数将使用文本创建一个矩形表面。传递给此方法的文本将添加到框形对象中，并从中返回，如下所示：

```py
def objects_text(sample_text, sample_color, sample_size):
 if sample_size == "small":
 surface_for_text = font_small.render(sample_text, True, sample_color)
 elif sample_size == "medium":
 surface_for_text= font_medium.render(sample_text, True, sample_color)
 elif sample_size == "large":
 surface_for_text = font_large.render(sample_text, True, sample_color)

 return surface_for_text, surface_for_text.get_rect()
```

让我们在Python文件中创建一个新的函数，使用上述字体向屏幕添加一条消息：

```py
def display_ScreenMessage(message, font_color, yDisplace=0, font_size="small"):
 textSurface, textRectShape = objects_text(message, font_color, font_size)
 textRectShape.center = (display_width/ 2), (display_height/ 2) + yDisplace
 DisplaySurface.blit(textSurface, textRectShape)
```

向`screen`方法传递的消息将创建一个矩形表面，以`blit`传递给它的文本作为`msg`。默认字体大小是小号，文本居中对齐在矩形表面的中心。现在，让我们为我们的游戏创建一个游戏介绍方法：

```py
def intro_for_game(): #function for adding game intro
 intro_screen = True   while intro_screen:

 for eachEvent in game.event.get():
 if eachEvent.type == game.QUIT:
 game.quit()
 quit()

 if eachEvent.type == game.KEYDOWN:
 if eachEvent.key == game.K_c:
 intro_screen = False
 if eachEvent.key == game.K_q:
 game.quit()
 quit()

 DisplayScreen.fill(color_white)
 display_ScreenMessage("Welcome to Snake",
 color_green,
  -99,
  "large")

 display_ScreenMessage("Made by Python Programmers",
 color_black,
  50)

 display_ScreenMessage("Press C to play or Q to quit.",
  color_red,
  180)

 game.display.update()
 objectClock.tick(12)
```

这个游戏的`intro`方法在游戏`loop`方法调用之前被调用。例如，看看下面的代码：

```py
intro_for_game()
MainLoopForGame()
```

最后，欢迎菜单的输出应该是这样的：

![](Images/6fb16dc8-2ad6-4d90-bfc6-9ef51d3d3be5.png)

最后，我们的游戏已经准备好分发了。你可能会看到我们的游戏是一个扩展名为`.py`的Python文件，它不能在没有安装Python的机器上执行。因此，在下一节中，我们将学习如何将Python文件转换为可执行文件，以便我们可以在Windows机器上全球分发我们的游戏。

# 转换为可执行文件

如果您已经制作了自己的pygame游戏，显然您希望与朋友和家人分享。在互联网世界中，共享文件非常容易，但当另一端的用户没有预安装Python时，问题就会出现。不是每个人都能为了测试您的游戏而安装Python。更好的想法是制作可在许多这些机器上执行的可执行文件。我们将在本节中学习如何转换为`.exe`，其他版本（Linux和Mac）将在接下来的章节中介绍。

如果使用Python提供的模块，将Python文件转换为可执行文件会更容易。其中有几个模块——`py2exe`和`cx_Freeze`。我们将在本节中使用第一个。

# 使用py2exe

要将Python文件转换为可执行文件，我们可以使用另一个名为`py2exe`的Python模块。`py2exe`模块不是pygame中预安装的——它不是标准库——但可以通过使用以下命令进行下载：

```py
pip install py2exe 
OR
py -3.7 -m pip install py2exe
```

下载`py2exe`模块后，转到包含您的Python文件的文件夹。在该位置打开命令提示符或终端并运行代码。它将把您的Python文件打包成一个`.exe`文件，或者成为可执行文件。以下命令将搜索并复制脚本使用的所有文件到一个名为`dist`的文件夹中。在`dist`中将会有一个`snake.exe`文件；这个文件将是Python代码的输出模拟，可以在没有安装Python的机器上执行。例如，您的朋友可能没有在他们的机器上安装Python，但他们仍然可以运行这个文件。为了将游戏分发到任何其他Windows机器，您只需发送`dist`文件夹或`snake.exe`文件的内容。只需运行以下命令：

```py
python snake.py py2exe #conversion command
```

这将创建一个名为*snake*的游戏，并带有`.exe`的扩展名。您可以在Windows平台上分发这些文件并从中获得响应。恭喜！你终于做到了。现在，让我们学习使用pygame进行游戏测试。

# 游戏测试和可能的修改

有时，您的机器可能会出现内存不足的情况。如果内存不足，并且您尝试将更多图像加载到游戏中，即使使用了pygame的最大努力，此过程也将被中止。`pygame.image.load`必须伴随一些内存才能正常执行任务。在内存不足的情况下，您可以预测到肯定会触发某种异常。即使有足够的内存，如果尝试加载不在硬盘驱动器中的图像，或者说，在编写文件名时出现了拼写错误，您可能会收到异常。因此，最好事先处理它们，这样我们就不必事后再去调试它们。

其次，让我们检查当我们向`set_mode`方法提供不寻常的屏幕尺寸时会发生什么。回想一下，`set_mode`是我们用来创建`Surface`对象的方法。例如，假设我们忘记向`set_mode`添加两个值，而只添加了一个。在这种情况下，我们也会触发错误：

```py
screen = pygame.display.set_mode((640))
TypeError: 2 argument expected
```

假设，与其忘记为高度和宽度添加适当的尺寸，如果我们将高度值添加为0会发生什么？在PyCharm IDE的情况下，这个问题不会创建任何异常。相反，程序将无限运行，导致您的机器崩溃。然而，这些程序通常会抛出一个`pygame.error: cannot set 0 sized display`的异常。现在您知道了`pygame`可能出错的地方，可以捕获这些异常并相应地处理它们：

```py
try:
    display = pygame.display.set_mode((640,0))
except pygame.error:
    print("Not possible to create display")
    exit()
```

因此，最好明智地选择您的显示屏，以消除任何不必要的异常。但更有可能的是，如果您尝试加载不在硬盘上的图像，您可能会遇到`pygame`错误的异常。因此，处理异常是一个很好的做法，以确保游戏的精灵或图像被正确加载。

# 总结

在本章中，我们研究了`pygame`模块，并发现了在游戏开发中使用它的原因。我们从下一章开始涵盖的大多数游戏都将在某种程度上基于`pygame`模块。因此，在继续之前，请确保自己使用pygame制作一个简单的游戏。

我们开始学习如何使用pygame对象制作游戏。我们学到了各种东西，包括处理涉及鼠标和键盘等输入设备的用户按键事件；我们制作了精灵动画；我们学习了颜色属性；并且使用向量运动处理了不同的对角线和方向性移动。我们使用简单的绘图应用程序创建了自己的精灵，并使用GIMP应用程序添加了alpha属性。我们尝试通过整合交互式游戏屏幕，也就是菜单屏幕，使游戏更具互动性。最后，我们学会了如何使用`py2exe`模块将Python文件转换为可执行文件。

本章的主要目标是让您熟悉精灵的使用，以便您可以制作2D游戏。您还学会了如何处理用户事件和不同的移动，包括对角线移动。您还学会了如何使用外部软件创建自定义精灵和图像，以及在游戏中使用它们的方法。不仅如此，您还熟悉了颜色和`rect`对象的概念，并学会了如何使用它们使游戏更具用户互动性，通过部署菜单和得分屏幕。

在下一章中，我们将运用本章学到的概念制作自己的flappy bird克隆游戏。除了本章学到的内容，我们还将学习游戏动画、角色动画、碰撞原理、随机对象生成、添加分数等许多概念。
