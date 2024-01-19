# 使用 Pygame 编写俄罗斯方块游戏

*打破常规思维*，这是一个老话，对于游戏开发者来说可能听起来陈词滥调，但仍然非常适用。大多数改变游戏行业的游戏都包含一些独特的元素，并代表了普通观众的口味。但这种全球性的假设通过丢弃可能在大多数游戏开发者中普遍存在的方法而被高估。毕竟，数学范式、对象渲染工具和软件保持不变。因此，在本章中，我们将探索一些每个游戏程序员都必须了解的高级数学变换和范式。

在本章中，我们将学习如何创建本世纪最受欢迎和下载量最大的游戏之一，这是 90 年代孩子们非常熟悉的游戏——*俄罗斯方块*。我们将学习如何通过从多维列表中格式化的形状来从头开始创建它。我们将学习如何绘制基本图形和游戏网格，这将帮助我们定位游戏对象。我们还将学习如何实现几何形状和图形的旋转变换。尽管这个概念听起来可能很简单，但这些概念的应用范围从不同的 2D 到 3D 的**角色扮演游戏**（**RPGs**）。

通过本章结束时，您将熟悉不同的概念，如创建网格（虚拟和物理）结构，以根据位置和颜色代码定位游戏对象。然后，您将学习如何使用列表推导来处理多维列表。此外，读者还将了解不同的移位变换和碰撞检查原则。在上一章中，我们使用 pygame 使用掩码实现了碰撞检查。然而，在本章中，我们将以程序员的方式来做这件事——这可能有点复杂，但包含了丰富的知识。

在本章中，我们将涵盖以下主题：

+   了解俄罗斯方块的基本要素

+   创建网格和随机形状

+   设置窗口和游戏循环

+   转换形状格式

+   修改游戏循环

+   清除行

+   游戏测试

# 技术要求

您需要以下要求才能完成本章：

+   Pygame 编辑器（IDLE）—建议使用 3.5+版本。

+   PyCharm IDE-参考第一章，*了解 Python-设置 Python 和编辑器*，了解安装过程。

+   俄罗斯方块游戏的代码资产可以在 GitHub 上找到，网址为[`github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter13`](https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter13)

查看以下视频以查看代码的运行情况：

[`bit.ly/2oDbq2J`](http://bit.ly/2oDbq2J)

# 了解俄罗斯方块的基本要素

将 pygame 精灵和图像合并到我们的 Python 游戏中是一个简单的过程。它需要一个内置的 Python 模块—*os—*，它将从您的计算机加载文件。在上一章中，我们在构建 Flappy Bird 游戏时学习了如何对精灵进行旋转、平移和碰撞，并逐个处理它们。这些变换不仅仅适用于图像，还适用于不同的几何图形和形状。当我们谈论使用这样的变换操作时，俄罗斯方块是每个人心中的游戏——玩家被允许通过周期运动改变几何形状的形状和大小。这种周期性运动将在顺时针和逆时针方向上创建逼真的几何形状的旋转变换。对于不熟悉俄罗斯方块的人，请查看[`www.freetetris.org/game.php`](https://www.freetetris.org/game.php)并观察游戏的网格和环境。

通过观察游戏环境，您会注意到三个主要的事情：

+   **几何形状，如 L、T、S、I 和正方形**：这些几何形状将以字母字符的形式呈现，并且为了区分它们，每个形状将有不同的颜色。

+   **网格**：这将是几何形状可以移动的地方。这将是游戏画布，几何形状将从顶部落到底部。玩家无法控制这个网格，但他们可以控制形状。

+   **旋转形状**：当形状/块向下掉落时，玩家可以使用键盘上的箭头键来改变形状的结构（请记住，只允许旋转变换）。

以下图表显示了我们将在游戏中使用的形状：

![](img/ea74cf4a-be7c-4741-a349-b3cf92a2a87e.png)

如果你玩过上述链接中的游戏，你会看到前面的形状在游戏的网格（画布）内移动。相应的字母代表它们所类似的每个几何形状。玩家只能使用箭头键来旋转这些形状。例如，当形状**I**掉落到网格时，玩家可以在垂直**I**和水平**I**之间切换。但对于正方形形状，我们不必定义任何旋转，因为正方形（由于其相等的边）在旋转后看起来完全相同。

现在你已经熟悉了我们俄罗斯方块游戏的游戏角色（几何形状），让我们进一步进行头脑风暴，以提取关于游戏的一些关键信息。让我们谈谈俄罗斯方块的基本要素。由于俄罗斯方块需要创建不同的几何形状，毫无疑问我们将需要`pygame`模块。`pygame`模块可以用来创建网格、边界和游戏角色。你还记得`pygame`的`draw`模块（来自第十一章，*使用 Pygame 制作 Outdo Turtle - 贪吃蛇游戏 UI*）吗？显然，如果不使用`pygame`的`draw`模块，你无法制作出好的游戏。同样，为了处理用户操作事件，如键盘操作，我们需要 pygame。

函数的蓝图代表了可以通过 Python 的`pygame`模块构建的俄罗斯方块的顶层视图：

+   `build_Grid()`: 这个函数将在游戏画布中绘制网格。网格是我们可以用不同颜色渲染几何形状的地方。

+   `create_Grid()`: 这个函数将在网格中创建不同的水平线，以便我们可以跟踪每个形状进行旋转变换。

+   `rotating_shapes`：这种技术将在相同的原点内旋转几何形状。这意味着旋转不会改变对象的尺寸（长度和高度）。

现在我们已经完成了头脑风暴的过程，让我们深入了解俄罗斯方块的基本概念。俄罗斯方块的环境简单而强大。我们必须在其中绘制网格，以便我们可以跟踪不同形状的每个（*x*，*y*）位置。同样，为了跟踪每个几何形状，我们需要创建一个字典，它将以*键*的形式存储对象的**位置**，以*值*的形式存储对象的**颜色**。

让我们从为我们的游戏编写模板代码开始：

```py
import pygame
import random

#declare GLOBALS
width = 800
height = 700

#since each shape needs equal width and height as of square 
game_width = 300 #each block will have 30 width
game_height = 600 #each block will have 30 height
shape_size = 30

#check top left position for rendering shapes afterwards

top_left_x, top_left_y = (width - game_width) // 2, height - game_height
```

现在我们已经完成了为我们的游戏声明全局变量的工作，这些变量主要负责屏幕的宽度和高度，我们可以开始为游戏对象定义形状格式。在下一节中，我们将定义一个嵌套列表，我们可以用它来定义游戏对象的多个结构（主要用于几何形状）。

# 创建形状格式

接下来的信息有点棘手。我们将声明俄罗斯方块的形状格式（所有必要的几何形状）。让我们看一个简单的例子，如下所示：

```py
#Example for creating shapes I
I = [['..**0**..',
      '..**0**..',
      '..**0**..',
      '..**0**..',
      '.....'],
     ['.....',
      '**0000**.',
      '.....',
      '.....',
      '.....']] #each 0 indicates block for shapes
```

观察前面代码中的形状格式。它是一个嵌套列表，我们需要它是因为`I`支持一次旋转，这将把垂直的`I`变成水平的`I`。观察前面列表的第一个元素；它包含一个句点（`.`），以及一个标识符（`0`），表示空和块的放置。在点或句点的位置，我们不会有任何东西，所以它将保持空白。但在`0`的位置，我们将存储块。为了做到这一点，从前面的代码中删除句点，并观察只有元素`0`。你会在零索引中看到垂直`I`，在第一个索引中看到水平`I`。对于正方形形状，我们不需要额外的*旋转*，所以我们最终将在列表内部声明正方形形状的一个元素。它将是这样的：

```py
#for square shapes square = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]
```

现在我们知道如何为几何形状创建格式了，让我们为不同的形状创建代码的起始部分：

```py
#following is for shape I
""" first element of list represents original structure,
    Second element represents rotational shape of objects """ I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]
#for square shape
O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

#for shape J
J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]
```

同样，让我们像之前一样为另外几个几何形状定义形状格式：

```py
#for shape L
L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]
#for shape T
T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]
```

现在我们已经成功地为我们的游戏定义了角色，让我们创建一个数据结构来保存这些对象，以及它们的颜色。让我们编写以下代码来实现这一点：

```py
game_objects = [I, O, J, L, T] #you can create as many as you want
objects_color = [(255, 255, 0), (255, 0, 0), (0, 0 , 255), (255, 255, 0), (128, 165, 0)] 
```

由于我们已经完成了基本的起始文件，也就是说，我们已经理解并创建了我们的游戏对象，在下一节中，我们将开始为我们的游戏创建一个网格，并将游戏对象渲染到屏幕上。

# 创建网格和随机形状

现在我们已经定义了形状的格式，是时候给它们实际的特征了。我们为形状提供特征的方式是定义尺寸和颜色。之前，我们将方块的尺寸定义为 30，这并不是任意的；形状的尺寸必须在高度和宽度上相等。在本章中我们要绘制的每个几何形状都将至少类似于正方形。感到困惑吗？看看我们定义形状格式的代码，包括句点（`.`）和字符（`0`）。如果你仔细观察列表的每个元素，你会看到正方形的格式，行和列中排列着相等数量的点。

正如我们在*了解俄罗斯方块的基本要素*部分中提到的，网格是我们游戏角色将驻留的地方或环境。玩家控制或动作只能在网格区域内激活。让我们谈谈网格在我们的游戏中如何使用。网格是屏幕以垂直和水平线的形式划分，每行和每列都由此组成。让我们自己制作一个并观察结果：

```py
#observe that this is not defined inside any class
def build_Grid(occupied = {}):
    shapes_grid = [[(0, 0, 0) for _ *in range(10)] for* _ in range(20)]
    for row in range(len(shapes_grid)):
        for column in range(len(shapes_grid[row])):
            if (column, row) in occupied:
 piece = occupied[(column, row)]
 shapes_grid[row][column] = piece
    return shapes_grid
```

前面的代码很复杂，但它是 pygame 大多数游戏的基本构建块。前面的代码将返回一个网格，显然是我们俄罗斯方块游戏的环境，但它也可以用于多种用途，比如稍加修改就可以用于制作井字游戏或吃豆人等。`build_Grid()`函数的参数是一个参数——*occupied* 字典。这个字典将从调用这个函数的地方传递给这个函数。主要是这个函数将在主函数内部调用，这将启动创建游戏网格的过程。

传递给`build_Grid`的 occupied 字典将包含一个键和一个值（因为它是一个字典）。键将表示每个块或形状所在的位置。值将包含每个形状的颜色代码，由键表示。例如，在你的打印字典中，你会看到类似`{位置:颜色代码}`的东西。

操作的下一行应该是一个让你大吃一惊的时刻。如果没有，你就错过了什么！这可以在第七章中找到，*列表推导和属性*。借助一行代码，我们定义了行和列的排列（多维列表）。它将为我们提供一系列值，可以用来创建一系列线的网格。当然，线将在主函数中稍后借助`pygame`的`draw`模块来绘制。我们将创建一个包含 10 行和一个包含 20 列的列表。现在，让我们谈谈代码的最后几行（高亮部分）。这些代码将循环遍历每个占用的位置，并通过修改它将其添加到网格中。

在为我们的游戏定义环境之后，我们需要做的下一件大事是定义游戏的形状。记住，每个形状都会有这样的属性：

+   **行和列位置**：网格特定位置将被指定为一定行和列的形状或几何图形。

+   **形状名称**：形状的标识符，表示要渲染哪些形状。我们将为每个形状添加字母字符，例如，形状 S 的字符 S。

+   颜色：每个形状的颜色。

+   **旋转**：每个形状的旋转角度。

现在我们已经了解了每个形状的可用属性，让我们为形状定义类，并将每个属性附加到它上面。按照以下代码创建`Shape`类：

```py
class Shape:
    no_of_rows = 20 #for y dimension
    no_of_columns = 10 #for x dimension

    #constructor
    def __init__(self, column, row, shape):
        self.x = column
        self.y = row
        self.shape = shape
        #class attributes
        self.color = objects_color[game_objects.index(shape)] 
#get color based on character indicated by shape name or shape variable
        self.rotation = 0 
```

`objects_color`和`game_objects`变量之前已经定义，它们是两个包含一个列表中的字母字符的不同列表。另一个列表中包含它们的颜色代码。

此刻，如果你运行你的游戏，你除了一个空的黑屏之外什么也看不到，这是因为我们的网格背景是用黑色代码渲染的。我们知道，如果我们想要画任何东西，可以借助 Python 的`pygame`模块来实现。此外，我们是从网格的顶部到底部绘制形状，所以我们必须随机生成形状。因为我们有五种形状，即 I、O、J、L 和 T，我们需要随机地渲染它们，一一地。让我们编写一个函数来实现以下代码片段。记住，我们在开始时已经导入了一个随机模块：

```py
def generate_shapes():
     global game_objects, objects_color
     return Shape(4, 0, random.choice(game_objects)) #creating instance
```

前面的后端逻辑对于任何涉及几何形状和图形的游戏都是至关重要的。这种知识的范围比你想象的要广泛得多。许多 RPG 游戏，包括 Minecraft，都让玩家与不同的几何形状进行交互。因此，创建网格是至关重要的，这样我们就可以引用每个图形的位置和颜色。现在我们已经创建了一些通用逻辑，可以创建不同形状和颜色的图形，我们需要一个工具，可以将这些形状渲染到网格中，通常是通过 OpenGL 或 pygame 来完成（PyOpenGL 将在接下来的第十四章中介绍，*了解 PyOpenGL*）。然而，在 Python 的情况下，更优秀的工具将是 pygame。因此，我们将使用`pygame`模块来制作俄罗斯方块游戏的形状和字符。

在下一节中，我们将创建一些逻辑，为网格结构设置游戏窗口。我们还将尝试运行游戏并观察其环境。

# 设置窗口和游戏循环

在设置游戏对象之后，我们游戏中的下一个重要步骤是渲染网格。不要被误导以为我们已经创建了网格，因为我们定义了`build_Grid()`方法之后。虽然这是一个有效的观点，但我们建立的网格到目前为止都是虚拟的。如果你简单地调用`build_Grid`方法，你将看不到任何东西，只会看到一个黑屏，这是网格的背景。在这里，我们将为这个网格提供一个结构。使用每个位置，由行和列指定，我们将使用`pygame`模块创建一条直线。

让我们创建一个简单的函数来为我们的游戏绘制一个窗口（主窗口），网格将驻留在其中：

```py
def create_Grid(screen_surface, grid_scene):
     screen_surface.fill(0, 0, 0) #black background
     for i in range(len(grid_scene)):
     for j in range(len(grid_scene[i])):

 #draw main rectangle which represents window
     pygame.draw.rect(screen_surface, grid_scene[i][j], (top_left_x + 
       j* 30, top_left_y + i * 30, 30, 30), 0)
 #above code will draw a rectangle at the middle of surface screen 

    build_Grid(screen_surface, 20 , 10) #creating grid positions       
    pygame.draw.rect(screen_surface, (255, 0, 0), (top_left_x, top_left_y, 
      game_width, game_height), 5)
    pygame.display.update() 
```

上述代码行将创建网格的物理结构，它将有不同的行和列。在循环遍历整个网格场景或网格的位置之后，我们将进入网格范围，以便使用先前突出显示的代码部分绘制一个矩形和网格边框。

同样，让我们通过为其定义边界来为这个网格提供物理结构。每一行和每一列都将通过在其中创建线条来区分。由于我们可以使用 pygame `draw`模块绘制线条，我们将使用它来编写以下函数：

```py
"""function that will create borders in each row and column positions """

def show_grid(screen_Surface, grid):
    """ --- following two variables will show from where to 
     draw lines---- """
    side_x = top_left_x
    side_y = top_left_y 
    for eachRow in range(grid):
        pygame.draw.line(screen_Surface, (128,128,128), (side_x, side_y+ 
        eachRow*30), (side_x + game_width, side_y + eachRow * 30))  
         # drawing horizontal lines (30) 
        for eachCol in range(grid[eachRow]):
            pygame.draw.line(screen_Surface, (128,128,128), (side_x + 
            eachCol * 30, side_y), (side_x + eachCol * 30, side_y +
               game_height))  
            # drawing vertical group of lines
```

上述函数有一个主循环，它循环进入由`build_Grid`方法确定的几行。在进入网格结构的每一行之后，它将使用`pygame` `draw`模块以颜色代码(128, 128, 128)绘制线条，从(`side_x`, `side_y`)开始，然后指向下一个坐标(`side_x + game_width, side_y + eachRow *30`)。起始点(`side_x`, `side_y`)是网格的最左侧角，而下一个坐标值(`side_x + game_width, side_y + eachRow *30`)表示网格的最右侧角的坐标。因此，我们将从网格的最左侧角绘制一条线到最右侧角。

在显式调用了前一个函数之后，你会看到以下输出：

![](img/308f5211-105f-4eb1-b438-b076b37c299b.png)

在设置了上述的网格或环境之后，我们将进入有趣的部分，也就是创建主函数。主函数将包含不同的内容，主要是用于调用和设置网格，并处理用户事件或操作，比如用户按下退出键或键盘上的箭头键时会发生什么。让我们用以下代码来定义它：

```py
def main():
 occupied = {} #this refers to the shapes occupied into the screen
 grid = build_Grid(occupied)

 done = False
 current_shape = generate_shapes() #random shapes chosen from lists. 
 next_shape = generate_shapes() 
 clock = pygame.time.Clock()
 time_of_fall = 0 #for automatic fall of shapes

 while not done:
 for eachEvent in pygame.event.get():
 if eachEvent.type == pygame.QUIT:
 done = True
 exit()    
```

既然我们已经开始定义主函数，它是我们游戏的指挥官，让我们定义它必须做的事情，如下所示：

+   调用多个函数，比如`build_Grid()`和`create_Grid()`，它们将设置游戏的环境

+   定义一个方法，执行代表字符的形状的旋转

+   定义一些逻辑，将下落时间限制添加到游戏中，也就是物体下落的速度

+   改变一个形状，在一个形状落到地面后

+   创建一些逻辑来检查形状的占用位置

上述过程是主函数的功能，我们应该解决它们。我们将在本节中解决前两个问题，但剩下的两个问题将在接下来的部分中解决。因此，主函数的第一个操作是调用一些关键函数，用于创建游戏的网格。如果你看上述的代码行，你会看到我们已经调用了`build_Grid`方法，它负责创建网格结构的行和列的虚拟位置。现在，剩下的任务只是调用`create_Grid()`方法，它将使用`pygame` `draw`模块为这个虚拟网格提供适当的物理结构。我们已经定义了这两个函数。

在下一节中，我们将学习一个重要的数学变换范式，即旋转，并将在我们的俄罗斯方块游戏中添加旋转游戏对象的功能。

# 理解旋转

在我们继续编写代码并修改主函数之前，让我们先了解一下数学知识。如果游戏与数学范式无关，那么游戏就什么都不是。运动、形状、角色和控制都由数学表达式处理。在本节中，我们将介绍数学的另一个重要概念：变换。尽管变换在数学中是一个模糊的概念，但我们将尽力学习这个概念。具体来说，有不同类型的变换：旋转、平移、反射和放大。在大多数游戏中，我们只需要两种类型的变换：旋转和放大。在本章中，我们将使用俄罗斯方块实现旋转变换，然后在第十六章中实现放大变换（构建愤怒的小鸟游戏时，*学习游戏人工智能-构建一个玩游戏的机器人*）。

术语*旋转*是一个数学概念，它表示*当一个对象被旋转时，意味着它以特定角度顺时针或逆时针旋转*。考虑以下例子：

![](img/7370dec7-a5e6-4beb-a586-3e05a340cc64.png)

在前面的例子中，我们有一个矩形形状，代表了俄罗斯方块游戏中的字母`I`字符。现在，想象一下玩家按下键盘上的*上*箭头键。在这种情况下，`I`的矩形形状必须以 90 度的角度旋转，并放置为水平的`I`字符，如前面的图表所示。因此，这些旋转是为了改变图形的形状，而不是尺寸。水平`I`和垂直`I`具有相同的尺寸（高度和宽度）。现在您已经了解了一些关于旋转的知识，您可以回到我们为每个字符（I、O、J、L 和 T）定义形状格式的代码，并观察多维列表。在`I`的情况下，您可以观察到它有两个元素。列表的第一个元素是游戏对象`I`的原始形状，列表的第二个元素是在旋转约 90 度后的扭曲形状。观察一下`O`字符，它是一个正方形。即使旋转任意角度，正方形仍然保持不变。因此，在正方形形状的情况下，列表中只有一个元素。

尽管我们已经了解了关于旋转的这些琐事，以及它们如何与每个形状格式相关联，但问题仍然存在：何时可以渲染每个形状，何时应执行旋转操作？答案很简单。当玩家按下键盘上的任何箭头键时，我们将执行旋转。但是哪里的代码暗示用户正在按键盘键？显然，这是在事件处理过程中完成的！在主函数中，我们开始捕获事件，并处理`QUIT`键的操作。现在，让我们使用以下代码对任何箭头键执行旋转：

代码应该添加在事件处理步骤中，在处理`QUIT`键之后。确保为代码提供适当的缩进。代码将在[`github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter13`](https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter13)上提供。

```py
        if anyEvent.type == pygame.KEYDOWN:
                if anyEvent.key == pygame.K_LEFT:
                    current_shape.x -= 1  #go left with shape

                elif anyEvent.key == pygame.K_RIGHT:
                    current_shape.x += 1 #go right with shape

                elif anyEvent.key == pygame.K_UP:
                    # rotate shape with angle of rotation 
                     (rotation variable)
                    current_shape.rotation = current_shape.rotation + 1 % 
                     len(current_shape.game_objects)

                if anyEvent.key == pygame.K_DOWN:
                    # moving current shape down into the grid
                    current_shape.y += 1
```

如果您想了解更多关于对象旋转如何在幕后工作的知识，请确保查看以下网址：[`mathsdoctor.co.uk`](https://mathsdoctor.co.uk)。

为了设置窗口画布或游戏屏幕，我们可以简单地调用`pygame set_mode`方法，并相应地渲染网格的窗口。方法调用的以下行应该在主函数中添加，在您设置了用户处理事件之后：

```py
    create_Grid(screen_surface) #screen surface will be initialized with 
                                 pygame below
```

现在我们已经为屏幕创建了一个网格，让我们设置主屏幕并调用主函数：

```py
screen_surface = pygame.display.set_mode((width, height))
main() #calling only
```

我们已经涵盖了几乎所有重要的事情，包括渲染显示，旋转对象，创建网格，渲染网格边界；但还有一个问题：我们如何将形状渲染到网格中？显然，我们的计算机还不够聪明，无法理解我们之前创建的多维列表来定义形状格式。还是困惑？检查我们为每个字符创建的多维列表，比如 I，O，J，L 和 T——我们的计算机无法理解这样的列表。因此，我们必须将这些列表值或属性转换为我们的计算机将进一步处理的维度值。我们的计算机将理解的维度值是指位置值。由于我们已经建立了网格，我们可以使用网格结构的行和列为计算机提供位置值。因此，让我们创建一个函数来实现它。

# 转换形状格式

我们的计算机无法理解数据结构的模糊内容，比如存储在多维列表中的内容。例如，看一下以下代码：

```py
#for square shapes square = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]
```

在以前的方形模式中，我们将一系列句点（`.`）与`0`配对。计算机不会认识 0 代表什么，句点代表什么。我们只知道句点在一个空位上，这意味着它的位置可以被忽略，而`0`所在的位置是块的位置。因此，我们需要编写一个程序，告诉计算机从网格中提取只有`0`所在的位置的程序。我们将通过定义以下函数来实现它：

```py
def define_shape_position(shape_piece):
    positions = []
    list_of_shapes = shape_piece.game_objects[shape_piece.rotation % 
                     len(shape_piece.shape)]

    for i, line in enumerate(list_of_shapes):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape_piece.x + j, shape_piece.y + i))

    for p, block_pos in enumerate(positions):
        positions[p] = (block_pos[0] - 2, block_pos[1] - 4)

    return positions
```

让我们详细看一下以前的代码：

1.  首先，这个函数返回对象的块的位置。因此，我们首先创建一个块字典。

1.  其次，我们存储了几个形状的列表，由多维字符列表`game_objects`（I，O，J，L 和 T）定义，并进行了旋转。

1.  现在，重要的部分：这个函数必须返回什么位置？这些位置是放置在网格中的`0`的位置。

1.  再次观察多维列表。你会看到一堆点（`.`）和`0`作为元素。我们只想要`0`所在的位置，而不是句点或点所在的位置。

1.  在我们使用`if column == \'0\'`命令检查每一列是否有`0`之后，我们只将这样的位置存储到 positions 字典中，并从函数中返回。

当进行旋转和移动等操作时，用户可能会触发一些无效的移动，比如将对象旋转到网格外部。因此，我们必须检查这些无效的移动并阻止它们发生。我们将创建`check_Moves()`函数来实现这一点。这个函数的参数将是形状和网格位置；形状是必要的，以检查特定旋转是否允许在由网格参数指示的位置内进行。如果网格指定的当前位置已经被占据，那么我们将摆脱这样的移动。有不同的实现方式，但最快最简单的方式是检查网格背景的颜色。如果网格中特定位置的颜色不是黑色，那么这意味着该位置已经被占据。因此，你可以从这个逻辑中得出一个详细的参考，解释为什么我们将网格的背景颜色设为黑色。通过这样做，我们可以检查对象是否已经在网格中。如果任何新对象下降到网格中，我们不应该通过已经存在于网格中的对象。

现在，让我们创建一个函数来检查位置是否被占用：

```py
def check_Moves(shape, grid):
    """ checking if the background color of particular position is 
        black or not, if it is, that means position is not occupied """

    valid_pos = [[(j, i) for j in range(10) if grid[i][j] == (0,0,0)] 
                for i in range(20)] 
    """ valid_pos contains color code in i variable and 
        position in j variable--we have to filter to get only 
        j variable """

    valid_pos = [j for p in valid_pos for j in p]

           """ list comprehension --same as writing
                    for p in valid_pos:
                        for j in p:
                            p
                            """
    """ Now get only the position from such shapes using 
        define_shape_position function """
    shape_pos = define_shape_position(shape)

    """check if pos is valid or not """
    for eachPos in shape_pos:
        if eachPos not in valid_pos:
            if eachPos[1] > -1: #eachPos[1] represents y value of shapes 
              and if it hits boundary
                return False #not valid move

    return True
```

到目前为止，我们一直在为我们的游戏构建后端逻辑，这涉及到渲染网格、操作网格、改变网格位置、实现决定两个对象碰撞时发生什么的逻辑等。尽管我们已经做了这么多，但当你运行游戏时，你仍然只会看到网格的形成，什么都没有。这是因为我们的主循环是游戏的指挥官——它将顺序地命令其他函数，但在主循环内，除了处理用户事件的代码之外，我们什么都没有。因此，在下一节中，我们将修改游戏的主循环并观察输出。

# 修改游戏循环

正如我们之前提到的，我们的主游戏循环负责执行许多任务，包括处理用户事件、处理网格、检查可能的移动等。我们一直在制作将检查这些动作、移动和环境的函数，但我们还没有调用它们一次，这将在本节中完成。如果你从高层次的角度观察主游戏循环，它将包含四个主要的架构构建块：

+   创建网格和处理游戏对象的移动。例如，掉落到网格中的对象的速度应该是多少？

+   处理用户事件。我们已经在检查事件并相应地旋转对象时做过这个，但前面的代码没有考虑`check_Moves()`函数，它将检查移动是否有效。因此，我们将相应地修改前面的代码。

+   为游戏对象添加颜色（唯一颜色）。例如，`S` 的颜色应该与 `I` 不同。

+   添加逻辑，检查对象撞击网格底部时会发生什么。

我们将逐步实现上述每个步骤。让我们从为对象添加速度开始。速度指的是网格结构中对象的自由下落速度。以下代码应该添加到主函数中：

```py
 global grid

 occupied = {} # (x pos, y pos) : (128, 0, 128)
 grid = build_Grid(occupied)
 change_shape = False
 done = False
 current_shape = generate_shapes()
 next_shape = generate_shapes()
 clock = pygame.time.Clock()
 timeforFall = 0

 while not done:
 speedforFall = 0.25

 grid = build_Grid(occupied)
 timeforFall += clock.get_rawtime()
 clock.tick()

 # code for making shape fall freely down the grid
 if timeforFall/1000 >= speedforFall:
 timeForFall = 0
 current_shape.y += 1 #moving downward
 #moving freely downward for invalid moves
 if not (check_Moves(current_shape, grid)) and current_shape.y > 0:
 current_shape.y -= 1
 change_shape = True
```

假设玩家尝试进行无效的移动。即使在这种情况下，游戏对象（形状）也必须自由向下掉落。这样的操作是在前面代码的最后三行中完成的。除此之外，代码是不言自明的；我们已经为对象定义了下落到网格中的速度，并使用了时钟模块来实现时间约束。

实现下一个逻辑，这相对容易一些。我们已经讨论了在俄罗斯方块中处理用户事件，考虑了旋转对象和进行简单的左右移动等细节。然而，在这些代码中，我们没有检查用户尝试的移动是否有效。我们必须首先检查这一点，以确保用户不能进行任何无效的移动。为了实现这一点，我们将调用之前创建的`check_Moves()`方法。以下代码将处理用户事件：

```py
if anyEvent.type == pygame.KEYDOWN:
                if anyEvent.key == pygame.K_LEFT:
                    current_shape.x -= 1
                    if not check_Moves(current_shape, grid):
                        current_shape.x += 1  # not valid move thus 
                           free falling shape

                elif anyEvent.key == pygame.K_RIGHT:
                    current_shape.x += 1
                   if not check_Moves(current_shape, grid):
                        current_shape.x -= **1**
      """ ROTATING OBJECTS """
                elif anyEvent.key == pygame.K_UP:

                    current_shape.rotation = current_shape.rotation + 1 % 
 len(current_shape.shape)
                    if not check_Moves(current_shape, grid):
                        current_shape.rotation = current_shape.rotation - 1 
 % len(current_shape.shape)

"""Moving faster while user presses down action key """
                if anyEvent.key == pygame.K_DOWN:

                    current_shape.y += 1
                    if not check_Moves(current_shape, grid):
                        current_shape.y -= 1
```

首先，关注被突出显示的代码。代码的第一个突出显示的部分是指移动是否有效进入网格，这是由`check_Moves()`函数检查的。我们允许当前形状向右角移动，即朝着正 *x* 轴。同样，关于上键，它负责检查对象是否允许旋转（只有上键会旋转对象；*左* 和 *右* 键会将对象从左到右移动，反之亦然）。在旋转的情况下，我们通过像素变换来旋转它，这是通过选择多维列表中指示的位置之一来完成的。例如，在形状 I 的情况下，列表中有两个元素：一个原始形状和另一个旋转形状。因此，为了使用另一个旋转形状，我们将检查移动是否有效，如果有效，我们将呈现新的形状。

应该添加到主函数中的第三段代码将处理为绘制网格中的形状添加颜色的技术。以下代码将为游戏范围内的每个对象添加颜色：

```py
     position_of_shape = define_shape_position(current_shape) 
     """ define_shape_function was created to return position of blocks of 
         an object """

        # adding color to each objects in to the grid. 
        for pos in range(len(position_of_shape)):
            x, y = position_of_shape[pos]

            """ when shapes is outside the main grid, we don't care """
            if y > -1: # But if we are inside the screen or grid, 
               we add color
                grid[y][x] = current_shape.color #adding color to the grid
```

最后，必须添加到主函数中的最后一段逻辑将处理当对象触地时的情况。让我们添加以下代码到主函数中以实现它：

```py
    if change_shape:
            for eachPos in position_of_shape:
                pos = (eachPos[0], eachPos[1])
                occupied[pos] = current_shape.color
            current_shape = next_shape
            next_shape = generate_shapes()
            change_shape = False
```

在上述代码中，我们通过检查布尔变量`change_shape`的内容来检查对象是否自由下落。然后，我们检查形状的当前位置并创建（*x*，*y*），它将表示占用的位置。然后将这样的位置添加到名为 occupied*的字典中。您必须记住，该字典的值是相同对象的颜色代码。在将当前对象分配给网格范围后，我们将使用`generate_shapes()`方法生成一个新形状。

最后，让我们通过调用`create_Grid()`函数来结束我们的主函数，参数是在以下代码中由 pygame 的`set_mode()`方法初始化的网格和表面对象（我们之前初始化了 pygame 的`surface`对象）：

```py
create_Grid(screen_surface, grid)
```

让我们运行游戏并观察输出：

![](img/c2d46b20-3945-47a1-b433-1dd79e686249.png)

现在，您可以清楚地看到我们能够制作一个俄罗斯方块游戏，用户可以根据需要转换对象并进行游戏。但等等！我们的游戏缺少一个重要的逻辑。我们如何激励玩家玩这个游戏？如果游戏只是关于旋转对象和用对象填充网格，那它就不会是历史悠久的游戏（这个游戏改变了 90 年代的游戏产业）。是的！游戏中必须添加一些逻辑，当调用这个逻辑时，我们将观察到每当行位置被*占用*时，我们必须清除这些行并将行向下移动一步，这将使我们比以前少了几行。我们将在下一节中实现这一点。

# 清除行

正如我们之前提到的，在本节中，我们将检查所有行的每个位置是否完全被占用。如果它们被占用，我们将从网格中删除这些行，并且这将导致每一行在网格中向下移动一步。这个逻辑很容易实现。我们将检查整行是否被占用，并相应地删除这些行。您还记得`check_Moves()`函数的情况吗？如果此函数检查每行的背景颜色，如果每行都没有黑色背景颜色，这意味着这样的行是被占用的。但即使我们有一个空位置，这意味着这个位置的背景颜色将是黑色，并且将被视为未被占用。因此，在清除行的情况下，我们可以使用类似的技术：如果在任何行中，任何位置的背景颜色是黑色，这意味着该位置未被占用，这样的行不能被清除。

让我们创建一个函数来实现清除行的逻辑：

```py
def delete_Row(grid, occupied):
    # check if the row is occupied or not
    black_background_color = (0, 0, 0)
    number_of_rows_deleted = 0
    for i in range(len(grid)-1,-1,-1):
        eachRow = grid[i]
        if black_background_color not in eachRow:
            number_of_rows_deleted += 1

            index_of_deleted_rows = i
            for j in range(len(eachRow)):
 try:
 del occupied[(j, i)]
                except:
                    continue
```

让我们消化前面的代码。这是一个相当复杂的逻辑，所以确保你学会了所有的东西；这些概念不仅适用于游戏创建，而且在技术面试中也经常被问到。问题在于如何通过创建逻辑来移动数据结构的值，而不是使用 Python 内置函数。我想以这种方式教给你，而不是使用任何内置方法，因为知道这个可能对编程的任何技术领域都有帮助。现在，让我们观察代码。它以创建一个`number_of_rows_deleted`变量开始，该变量表示已从网格中删除的行数。关于已删除行数的信息很重要，因为在删除这些行数后，我们需要将位于已删除行上方的行数向下移动相同的数量。例如，看看下面的图表：

![](img/712a0c70-671e-482c-a693-9565bc90a21a.png)

同样，现在我们知道了使用`if black_background_color not in eachRow`表达式要删除什么，我们可以确定网格的每一行是否有空位。如果有空位，这意味着行没有被占据，如果有，那么黑色背景颜色，即(0, 0, 0)，不会出现在任何行中。如果我们没有找到黑色背景颜色，那么我们可以确定行被占据，我们可以通过进一步检查条件来删除它们。在代码的突出部分中，你可以看到我们只取第 j 个元素，这只是一列。这是因为在删除行时，`I`的值保持不变，但第 j 列的值不同。因此，我们在单行内循环整个列，并使用`del`命令删除被占据的位置。

从上一行代码中，我们能够删除整行，如果有任何行被占据，但我们没有解决删除后应该发生什么，这是棘手的部分。在我们删除每一行后，不仅会删除方块，整个包含行的网格也会被删除。因此，在删除的方块位置，我们不会有空行；相反，包含网格的整行将被删除。因此，为了确保我们不减少实际网格的数量，我们需要从顶部添加另一行来补偿。让我们编写一些代码来实现这一点：

```py
#code should be added within delete_Row function outside for loop
if number_of_rows_deleted > 0:       #if there is at least one rows deleted 

        for position in sorted(list(occupied), position=lambda x: 
          x[1])[::-1]:
            x, y = position
            if y < index_of_deleted_rows:
                """ shifting operation """
                newPos = (x, y + number_of_rows_deleted)
                occupied[newPos] = occupied.pop(position)

return number_of_rows_deleted
```

好了！让我们消化一下。这是相当复杂但非常强大的信息。前面的代码将实现将行块从顶部向下移入网格。首先，只有在我们删除了任何行时才需要移位；如果是，我们就进入逻辑来执行移位。首先，让我们只观察涉及 lambda 函数的代码，即`list(occupied), position=lambda x: x[1]`。该代码将创建一个包含网格所有位置的列表，然后使用 lambda 函数仅获取位置的*y*部分。请记住，获取方块的*x*位置是多余的——对于每一行，*x*的值保持不变，但*y*的值不同。因此，我们将获取*y*位置的值，然后使用`sorted(x)`函数对其进行排序。排序函数将根据*y*坐标的值对位置进行排序。

首先，排序将根据*y*的较小值到*y*的较大值进行。例如，看看下面的图表：

![](img/f4cf74ea-2a52-4699-bc7a-b649e98cbffb.png)

调用 sorted 方法，然后反转列表（参见第四章，*数据结构和函数*，了解更多关于如何反转列表的信息）很重要，因为有时网格的底部部分可能没有被占据，只有上层会被占据。在这种情况下，我们不希望移位操作对未被占据的底部行造成任何伤害。

同样，在追踪每一行的位置后，我们将检查是否有任何删除行上方的行，使用`if y < index_of_deleted_rows`表达式。同样，在这种情况下，*x*的值是无关紧要的，因为它在单行内是相同的；在我们检查是否有任何删除行上方的行之后，我们执行移位操作。移位操作非常简单；我们将尝试为位于删除行正上方的每一行分配新位置。我们可以通过增加删除行的数量来创建新位置的值。例如，如果有两行被删除，我们需要将*y*的值增加两个，以便删除行上方的方块和随后的方块将向下移动两行。在我们将行向下移动到网格后，我们必须从先前的位置弹出方块。

既然我们已经定义了一个函数，如果整行被占据，它将清除整行，让我们从主函数中调用它来观察其效果：

```py
def main():
    ...
    while not done:
        ... 
        if change_shape:
            ...
            change_shape = False
            delete_Row(grid, occupied)
```

最后，在这个漫长而乏味的编码日子里，我们取得了非常有成效的结果。当您运行声明了主函数的模块时，您将看到以下输出：

![](img/529f38a4-88da-4c63-9baa-01e261a7d2de.png)

游戏看起来很吸引人，我已经在代码中测试了一切。代码看起来非常全面和详尽，没有漏洞。同样，您可以玩它并与朋友分享，并发现可以对这个游戏进行的可能修改。这是一个高级游戏，当用 Python 从头开始编码时，它充分提高了自己的水准。在构建这个游戏的过程中，我们学到了很多东西。我们学会了如何定义形状格式（我们以前做过更复杂的事情，比如精灵的转换和处理精灵的碰撞），但这一章在不同方面都具有挑战性。例如，我们必须注意诸如无效移动、可能的碰撞、移位等事项。我们实现了一些逻辑，通过比较两种不同的颜色对象：网格或表面的**背景颜色**与**游戏对象颜色**，来确定对象是否放置在某个位置。

我们还没有完成；我们将在下一节尝试实现更多逻辑。我们将看看我们的游戏可以进行哪些其他修改。我们将尝试构建一些逻辑，随着游戏的进行，将增加游戏的难度级别。

# 游戏测试

我们的游戏可以进行多种修改，但最重要的修改将是添加欢迎屏幕、增加难度级别和得分屏幕。让我们从欢迎屏幕开始，因为它很容易实现。我们可以使用`pygame`模块创建一个窗口，并使用文本表面向用户提供消息。以下代码显示了如何为我们的俄罗斯方块游戏创建一个主屏幕：

```py

def Welcome_Screen(surface):  
    done = False
    while not done:
        surface.fill((128,0,128))
        font = pygame.font.SysFont("comicsans", size, bold=True)
        label = font.render('Press ANY Key To Play Tetris!!', 1, (255, 255, 
                255))

        surface.blit(label, (top_left_x + game_width /2 - 
         (label.get_width()/2), top_left_y + game_height/2 - 
          label.get_height()/2))

        pygame.display.update()
        for eachEvent in pygame.event.get():
            if eachEvent.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                main(surface) #calling main when user enters Enter key 

    pygame.display.quit()
```

运行游戏后，您将看到以下输出，其中将呈现欢迎屏幕。按下任意键后，您将被重定向到俄罗斯方块游戏：

![](img/ff97e2d4-de62-4a0c-8075-dda1d8e9657b.png)

同样，让我们添加一些逻辑，以增加游戏的难度。有两种实现这种逻辑的方法。首先，您可以创建一个计时器，如果玩家玩的时间超过关联计时器的范围，我们可以减慢下落速度，使形状比以前下落得更快（增加速度）：

```py
timeforLevel = 0

while not done:
    speedforFall = 0.27 - timeforLevel 
    ...
    if timeforLevel / 10000 > 0.5:
        timeforLevel = 0
        if timeforLevel > 0.15:
            timeforLevel += 0.05
    ...

"""  ---------------------------------------------------
        speedforFall = 0.24 will make object to fall faster comparative 
                       to speedforFall = 0.30 

    ----------------------------------------------------- """ 
```

同样，我们可以实现另一段逻辑来增加游戏的难度。这种方法比之前的更好。在这种方法中，我们将使用*分数*来增加游戏的难度。以下代码表示了如何实现玩家的得分以增加游戏级别的蓝图：

```py
def increaseSpeed(score):
    game_level = int(score*speedForFall)
    speedforFall = 0.28 - (game_level)
    return speedforFall
```

在前面的代码中，我们实现了分数和物体速度之间的关系。假设玩家的分数更高。这意味着用户一直在玩较低难度的级别，因此，这样一个高分值将与更高的下落速度值相乘，导致`speedforFall`的增加，然后从物体的速度中减去，这将创建一个更快的下落动作。相反，玩在更高级别的玩家将有一个较低的分数，这将与物体速度的较低值相乘，导致一个较低的数字，然后从`speedforFall`变量中减去。这将导致玩更难级别的玩家速度变化较小。但假设玩家是专业的，并且在更难的级别中得分更高。在这种情况下，物体的下落速度相应增加。

我们最终完成了一个完全功能的俄罗斯方块游戏。在本章中，我们学习了使用 Python 进行游戏编程的几个高级概念。在创建过程中，我们复习了一些我们之前学到的关于 Python 的基本概念，比如操作多维列表，列表推导，面向对象的范式和数学变换。除了复习这些概念，我们还发现了一些新颖的概念，比如实现旋转，实现移位操作，从头开始创建形状格式，创建网格（虚拟和物理）结构，并在网格中放置物体。

# 总结

在本章中，我们探索了实现多维列表处理的*Pythonic*方式。我们创建了一个多维列表来存储不同几何形状的格式，并使用数学变换对其进行操作。

我们使用了俄罗斯方块的简单示例来演示游戏中几种数据结构的使用，以及它们的操作。我们实现了一个字典，将键存储为位置，值存储为这些物体的颜色代码。构建这样一个字典对于俄罗斯方块等游戏来说是救命的。在制作检查碰撞和移位操作的逻辑时，我们使用字典来观察任何物体的背景颜色是否与任何位置的背景相同。尽管俄罗斯方块只是一个案例研究，但在这个游戏中使用的技术也被用于许多现实世界的游戏，包括 Minecraft，几乎每个 RPG 游戏。

数学变换涉及的操作对我们非常重要。在本章中，我们使用了旋转原理来改变物体的结构而不改变其尺寸。从本章中您将掌握的知识是巨大的。诸如操作多维列表之类的概念可以扩展到数据应用程序，并被称为 2D Numpy 数组，用于创建不同的类比，比如街道类比，多旅行者问题等。尽管字典被认为是数据结构之王，但处理多维列表并不逊色，因为它与列表推导的简单性相结合。除了实现这些复杂的数据结构，我们还学会了如何实现数学变换，即游戏物体的旋转运动。这个特性在任何 3D 游戏中都非常有用，因为它将为用户提供对场景的 360 度视图。同样，我们还学会了如何创建网格结构。

网格结构用于跟踪物体的位置。在像 WorldCraft 这样的复杂游戏中，跟踪游戏的物体和资源是任何游戏开发者的强制性任务，在这种情况下，网格非常有效。可以将不可见的网格实现为字典，或者作为任何复杂的集合。

本章的主要目标是让您熟悉 2D 游戏图形，即绘制基本图形和游戏网格。同样，您还了解了另一种检测游戏对象之间碰撞的方法（在 Flappy Bird 游戏中，我们使用了 pygame 掩模技术来检测碰撞）。在本章中，我们实现了一种通用和传统的碰撞检测方法：通过检查背景颜色属性和对象颜色属性。同样，我们学会了如何通过旋转来创建不同结构的对象。这种技术可以用来在游戏中生成多个敌人。我们没有为每个角色设计多个不同的对象（这可能耗时且昂贵），而是使用变换来改变对象的结构。

下一章是关于 Python OpenGL，通常称为 PyOpenGL。我们将看到如何使用 OpenGL 创建不同的几何结构，并观察如何将 PyOpenGL 和 pygame 一起使用。我们将主要关注不同的数学范式。我们将看到顶点和边等属性如何用于创建不同的复杂数学形状。此外，我们将看到如何使用 PyOpenGL 实现游戏中的放大和缩小功能。
