# 第十章：使用 Turtle 升级蛇游戏

大多数电脑游戏玩家认为游戏因其外观而令人兴奋和吸引人。在某种程度上，这是真的。计算机游戏必须在视觉上具有吸引力，以便玩家感觉自己在其中参与。大多数游戏开发人员和游戏设计师花费大量时间开发游戏图形和动画，以提供更好的体验给玩家。

本章将教您如何使用 Python 的`turtle`模块从头开始构建游戏的基本布局。正如我们所知，`turtle`模块允许我们制作具有二维（2D）运动的游戏；因此，本章我们将只制作 2D 游戏，如 flappy bird、pong 和 snake。本章将涵盖的概念非常重要，以便将运动与游戏角色的用户操作绑定起来。

通过本章结束时，您将学会通过创建 2D 动画和游戏来实现数据模型。因此，您将学会如何处理游戏逻辑的不同组件，例如定义碰撞、边界、投影和屏幕点击事件。通过学习游戏编程的这些方面，您将能够学会如何使用`turtle`模块定义和设计游戏组件。

本章将涵盖以下主题：

+   计算机像素概述

+   使用 Turtle 模块进行简单动画

+   使用 Turtle 升级蛇游戏

+   乒乓球游戏

+   flappy bird 游戏

+   游戏测试和可能的修改

# 技术要求

您需要以下资源：

+   Python 3.5 或更新版本

+   Python IDLE（Python 内置的 IDE）

+   文本编辑器

+   网络浏览器

本章的文件可以在这里找到：[`github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter10`](https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter10)

查看以下视频以查看代码运行情况：

[`bit.ly/2oJLeTY`](http://bit.ly/2oJLeTY)

# 探索计算机像素

当您仔细观察计算机屏幕时，您可能会发现形成行和列的小点。从一定距离上看，这些点的矩阵代表图像，这是我们在屏幕上看到的。这些点称为像素。由于计算机游戏应该在视觉上令人愉悦，我们必须使用这些像素来创建和自定义游戏屏幕，甚至使用它们来使玩家在游戏中移动，这将显示在屏幕上。每当玩家在键盘上按下任何键时，移动的变化必须反映在屏幕的像素上。例如，当玩家按下**右**键时，特定字符必须在屏幕上向右移动若干个像素单位，以表示运动。我们在上一章中讨论了矢量运动，它能够覆盖一些类的方法以实现运动。我们将使用矢量的技术来使游戏角色进行像素移动。让我们观察以下大纲，我们将使用矢量和 turtle 模块来制作任何游戏：

1.  制作一个`Vector`类，其中将具有`__add__()`、`__mul__()`和`__div__()`等方法，这些方法将对我们的向量点执行算术运算。

1.  使用`Vector`类在游戏屏幕上实例化玩家，并设置其瞄准目标或移动。

1.  使用`turtle`模块制作游戏边界。

1.  使用`turtle`模块绘制游戏角色。

1.  应该使用`Vector`类的旋转、前进和移动等操作，以使游戏角色移动。

1.  使用主循环处理用户事件。

我们将通过制作简单的**Mario**像素艺术来学习像素表示。以下代码显示了多维列表中像素的表示，这是一个列表的列表。我们使用多维列表将每个像素存储在单独的行中：

```py
>>> grid = [[1,0,1,0,1,0],[0,1,0,1,0,1],[1,0,1,0,1,0]]
```

前面的网格由三行组成，代表像素位置。类似于列表元素提取方法，`>>> grid[1][4]`语句从网格的第二个列表（即[0,1,0,1,0,1]）中返回'0'的位置值。 （请参考第四章，*数据结构和函数*，以了解更多关于列表操作的信息。）因此，我们可以访问网格内的任何单元格。

以下代码应该写在 Python 脚本中。通过创建一个`mario.py`文件，我们将用它来创建马里奥像素艺术：

1.  首先导入 turtle——`import turtle`——这是我们将要使用的唯一模块。

1.  使用`>>> Pen = turtle.Turtle()`命令实例化`turtle`模块。

1.  使用速度和颜色属性为画笔指定两个属性：

```py
      Pen.speed(0)
          Pen.color("#0000000")   #or Pen.color(0, 0, 0)
```

1.  我们必须创建一个名为`box`的`new`函数，该函数将使用画笔方法绘制正方形形状来绘制一个盒子。这个盒子大小代表像素艺术的尺寸：

```py
       def box(Dimension): #box method creates rectangular box
               Pen.begin_fill()
           # 0 deg.
               Pen.forward(Dimension)
               Pen.left(90)
           # 90 deg.
               Pen.forward(Dimension)
               Pen.left(90)
           # 180 deg.
               Pen.forward(Dimension)
               Pen.left(90)
           # 270 deg.
               Pen.forward(Dimension)
               Pen.end_fill()
               Pen.setheading(0)
```

1.  我们必须将画笔定位到屏幕左上角的位置开始绘画。这些命令应该在`box()`函数之外定义：

```py
      Pen.penup()
      Pen.forward(-100)
      Pen.setheading(90)
      Pen.forward(100)
      Pen.setheading(0)
```

1.  定义盒子大小，代表我们要绘制的像素艺术的尺寸：

```py
      boxSize = 10
```

1.  在第二阶段，您必须以多维列表的形式声明像素，这些像素代表每个像素的位置。以下的`grid_of_pixels`变量代表了代表像素位置的线网格。下面的代码行必须添加到`box`函数定义之外。（请参考[`github.com/PacktPublishing/Learning-Python-by-building-games`](https://github.com/PacktPublishing/Learning-Python-by-building-games)来定位游戏文件，即`mario.py`。）：

请记住，单个形式的像素组合代表一条直线。

```py
      grid_of_pixels = [[1,1,1,1,2,2,2,2,2,2,2,2,1,1,1,1]]
      grid_of_pixels.append([1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,1])
      grid_of_pixels.append([1,1,1,0,0,0,3,3,3,3,3,0,3,1,1,1])
      grid_of_pixels.append([1,1,0,3,0,3,3,3,3,3,3,0,3,3,3,1])
      grid_of_pixels.append([1,1,0,3,0,0,3,3,3,3,3,3,0,3,3,3])
      grid_of_pixels.append([1,1,0,0,3,3,3,3,3,3,3,0,0,0,0,1])
      grid_of_pixels.append([1,1,1,1,3,3,3,3,3,3,3,3,3,3,1,1])
      grid_of_pixels.append([1,1,1,0,0,2,0,0,0,0,2,0,1,1,1,1])
      grid_of_pixels.append([1,1,0,0,0,2,0,0,0,0,2,0,0,0,1,1])
      grid_of_pixels.append([0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,0])
      grid_of_pixels.append([3,3,3,0,2,3,2,2,2,2,3,2,0,3,3,3])
      grid_of_pixels.append([3,3,3,3,2,2,2,2,2,2,2,2,3,3,3,3])
      grid_of_pixels.append([3,3,3,2,2,2,2,1,1,2,2,2,2,3,3,3])
      grid_of_pixels.append([1,1,1,2,2,2,1,1,1,1,2,2,2,1,1,1])
      grid_of_pixels.append([1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1])
      grid_of_pixels.append([0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0])
```

1.  使用颜色定义像素艺术的调色板。我们将使用颜色代码来定义艺术品的颜色，如下面的代码所示。十六进制颜色代码（HEX）代表红色、绿色和蓝色的颜色组合（#RRGGBB）。请参考[`htmlcolorcodes.com/`](https://htmlcolorcodes.com/)以分析不同颜色的不同代码：

```py
      palette = ["#4B610B" , "#FAFAFA" , "#DF0101" , "#FE9A2E"]
```

1.  接下来，我们应该开始使用我们在*步骤 7*和*步骤 8*中定义的像素网格和调色板来绘制像素艺术。我们必须使用我们之前制作的`box()`函数来制作像素艺术。像素艺术由行和列组成；因此，我们必须声明两个循环来绘制艺术品。以下代码调用了`turtle`模块的不同函数，如`forward()`、`penup()`和`pendown()`。我们在上一章中学习了它们；它们将利用画笔根据像素网格的列表来绘制。

```py
       for i in range (0,len(grid_of_pixels)):
               for j in range (0,len(grid_of_pixels[i])):
                   Pen.color(palette[grid_of_pixels[i][j]])
                   box(boxSize)
                   Pen.penup()
                   Pen.forward(boxSize)
                   Pen.pendown()    
               Pen.setheading(270)
               Pen.penup()
               Pen.forward(boxSize)
               Pen.setheading(180)
               Pen.forward(boxSize*len(grid_of_pixels[i]))
               Pen.setheading(0)
               Pen.pendown()
```

让我们消化前面的代码片段。它包含一个`for`循环，从 0 的初始值循环到代表画布中位置的像素网格的长度。每个像素代表一个位置，我们必须使用画笔进行绘制；因此，我们逐个循环每个像素。在二维`for`循环内，我们从调色板中获取颜色并调用`box`方法，该方法创建一个矩形框，我们的马里奥艺术应该在其中呈现。我们使用`turtle`画笔在这个框内绘制，使用`forward()`函数。我们在像素的行中执行相同的操作，如第 i 个循环所示。

一旦我们完成了前面的代码组合，也就是我们执行了`box`方法、初始化和两个主要的`for`循环，我们就可以运行代码并观察以下马里奥像素艺术。运行我们的代码后，`turtle`模块的画笔将开始绘制，最终会给我们以下艺术品：

![](img/ffcd57a4-69b5-4504-aedf-e409af2f5370.png)

由于我们熟悉像素和矢量运动的概念，现在是使用 2D 图形制作游戏的时候了。我们将使用`turtle`模块以及数据模型来创建游戏角色并使它们移动。我们将通过在下一节中制作一个简单的动画来开始这个冒险。

# 使用 Turtle 模块理解简单动画

到目前为止，我们可能已经熟悉了`turtle`模块的不同方法。这意味着我们不会在创建游戏角色时遇到任何问题。同样，游戏角色的运动是使用矢量运动来实现的。矢量加法和减法等操作通过对象的旋转提供直线运动（有关更多信息，请参阅第九章，*数据模型实现*）。以下代码片段中定义的`move`操作将为游戏角色提供随机移动。`move`方法将以另一个矢量作为催化剂，并执行数学运算以更新当前位置，同时考虑游戏角色的方向：

```py
>>> v = (1,2) #vector coordinates
>>> v.move(3,4) # vector addition is done (1,2) + (3,4)
>>> v
(4,6)
```

`rotate`方法将按逆时针方向旋转矢量特定角度（原地）。以下示例表示`rotate`方法的调用：

```py
>>> v = vector(1, 2)
>>> v.rotate(90)
>>> v == vector(-2, 1)
True
```

我们必须在`Vector`类中定义前面两种方法。按照以下步骤实现`Vector`类：

1.  您必须从使用 class 关键字定义`Vector`类开始。我们将定义 slots 作为类属性，其中包含三个属性。slots 表示一个包含三个关键信息的属性：*x*、*y*和 hash。*x*和*y*的值是游戏角色的当前位置，而 hash 用于定位数据记录。例如，如果使用*x*和*y*坐标实例化`Vector`类，则将激活 hash 属性。否则，它将保持未激活状态。

1.  矢量元素的坐标，即(5,6)，由*x*和*y*表示，其中*x=5*，*y=6*，hash 变量表示插槽是否为空。hash 变量用于定位数据记录并检查`Vector`类是否已实例化。如果插槽属性已经包含*x*和*y*，则 hash 属性将阻止对插槽的进一步赋值。我们还将定义`PRECISION`属性（用户定义），它将把*x*和*y*的坐标四舍五入到一定的级别。为了使事情清楚，代码中添加了几个示例，并且您可以在三行注释中观察到这一点：

```py
      #following class will create vector 
      #representing current position of game character
      class vector(collections.Sequence):
          """Two-dimensional vector.

          Vectors can be modified in-place.

          >>> v = vector(0, 1)
          >>> v.move(1)
          >>> v
          vector(1, 2)
          >>> v.rotate(90)
          >>> v
          vector(-2.0, 1.0)

          """

          PRECISION = 6 #value 6 represents level of rounding
          #for example: 4.53434343 => 4.534343
          __slots__ = ('_x', '_y', '_hash')
```

1.  接下来，我们需要定义类的第一个成员。我们知道类的第一个成员是`__init__()`方法。我们将定义它以初始化类属性，即*x*和*y*。我们已经将*x*和*y*的值四舍五入到`PRECISION`属性指示的一定精度级别。`round()`是 Python 的内置函数。以下代码行包含一个构造函数，我们在其中使用`round`方法初始化矢量坐标（*x*，*y*）：

```py
      def __init__(self, x, y):
              """Initialize vector with coordinates: x, y.

              >>> v = vector(1, 2)
              >>> v.x
              1
              >>> v.y
              2

              """
              self._hash = None
              self._x = round(x, self.PRECISION)
              self._y = round(y, self.PRECISION)
```

1.  您可能已经注意到，您已将*x*和*y*属性作为私有属性，因为它们以单下划线(`_x`, `_y`)开头。因此，无法直接初始化这些类型的属性，这导致了**数据封装**，这是我们在面向对象范例主题中讨论过的。现在，为了获取和设置这些属性的值，您必须使用`getter`和`setter`方法。这两种方法将成为`Vector`类的属性。以下代码表示如何为我们的`Vector`类实现`getter`和`setter`：

```py
      @property
          def x(self):
              """X-axis component of vector.

              >>> v = vector(1, 2)
              >>> v.x
              1
              >>> v.x = 3
              >>> v.x
              3

              """
              return self._x

          @x.setter
          def x(self, value):
              if self._hash is not None:
                  raise ValueError('cannot set x after hashing')
              self._x = round(value, self.PRECISION)

          @property
          def y(self):
              """Y-axis component of vector.

              >>> v = vector(1, 2)
              >>> v.y
              2
              >>> v.y = 5
              >>> v.y
              5

              """
              return self._y

          @y.setter
          def y(self, value):
              if self._hash is not None:
                  raise ValueError('cannot set y after hashing')
              self._y = round(value, self.PRECISION)
```

1.  除了`getter`和`setter`方法之外，您可能已经注意到了`_hash`，它表示插槽是否已分配。为了检查插槽是否已经被分配，我们必须实现一个数据模型，即`__hash__()`。

简单回顾一下：数据模型或魔术函数允许我们更改由其祖先之一提供的方法的实现。

现在，我们将在我们的`Vector`类上定义`hash`方法，并以不同的方式实现它：

```py
      def __hash__(self):
              """v.__hash__() -> hash(v)

              >>> v = vector(1, 2)
              >>> h = hash(v)
              >>> v.x = 2
              Traceback (most recent call last):
                  ...
              ValueError: cannot set x after hashing

              """
              if self._hash is None:
                  pair = (self.x, self.y)
                  self._hash = hash(pair)
              return self._hash
```

1.  最后，您必须在`Vector`类中实现两个主要方法：`move()`和`rotate()`。我们将从`move`方法开始。`move`方法将移动向量到其他位置（原地）。这里，其他是传递给`move`方法的参数。例如，`(1, 2).move(2, 3)`将得到(3, 5)。记住：移动是通过任何向量算术运算来完成的，即加法、乘法、除法等。我们将使用`__add__()`魔术函数（参考第九章，*数据模型实现*）来为向量创建移动。在此之前，我们必须创建一个返回向量副本的`copy`方法。`copy()`方法很重要，因为我们不希望操作损害我们的原始向量；相反，我们将在原始向量的副本上执行算术运算：

```py
      def copy(self):
              """Return copy of vector.

              >>> v = vector(1, 2)
              >>> w = v.copy()
              >>> v is w
              False

              """
              type_self = type(self)
              return type_self(self.x, self.y)
```

1.  在实现`add`函数之前，您必须实现`iadd`魔术函数。我们使用`__iadd__`方法来实现扩展的`add`运算符赋值。我们可以在`Vector`类中实现`__iadd__()`魔术函数，如下所示。我们在上一章中看到了它的实现（第九章，*数据模型实现*）：

```py
      def __iadd__(self, other):
              """v.__iadd__(w) -> v += w

              >>> v = vector(1, 2)
              >>> w = vector(3, 4)
              >>> v += w
              >>> v
              vector(4, 6)
              >>> v += 1
              >>> v
              vector(5, 7)

              """
              if self._hash is not None:
                  raise ValueError('cannot add vector after hashing')
              elif isinstance(other, vector):
                  self.x += other.x
                  self.y += other.y
              else:
                  self.x += other
                  self.y += other
              return self
```

1.  现在，您需要创建一个新的方法`__add__`，它将在原始向量的副本上调用前面的`__iadd__()`方法。最后一条语句`__radd__ = __add__`具有重要的意义。让我们观察一下`radd`和`add`之间的下面的图示关系。它的工作原理是这样的：Python 尝试评估表达式*Vector(1,4) + Vector(4,5)*。首先，它调用`int.__add__((1,4), (4,5))`，这会引发异常。之后，它将尝试调用`Vector.__radd__((1,4), (4,5))`：

![](img/3bc12aac-5ec9-44c4-b189-2339422283c9.png)

很容易看出，`__radd__`的实现类似于`add`：（参考`__add__()`方法中注释中定义的示例代码）：

```py
       def __add__(self, other):
              """v.__add__(w) -> v + w

              >>> v = vector(1, 2)
              >>> w = vector(3, 4)
              >>> v + w
              vector(4, 6)
              >>> v + 1
              vector(2, 3)
              >>> 2.0 + v
              vector(3.0, 4.0)

              """
              copy = self.copy()
              return copy.__iadd__(other)

          __radd__ = __add__
```

1.  最后，我们准备为我们的动画制作第一个移动序列。我们将从在我们的类中定义`move`方法开始。`move()`方法将接受一个向量作为参数，并将其添加到表示游戏角色当前位置的当前向量中。`move`方法将实现直线加法。以下代码表示了`move`方法的定义：

```py
      def move(self, other):
              """Move vector by other (in-place).

              >>> v = vector(1, 2)
              >>> w = vector(3, 4)
              >>> v.move(w)
              >>> v
              vector(4, 6)
              >>> v.move(3)
              >>> v
              vector(7, 9)

              """
              self.__iadd__(other)
```

1.  接下来，我们需要创建`rotate()`方法。这个方法相当棘手，因为它会逆时针旋转向量一个指定的角度（原地）。这个方法将使用三角函数操作，比如角度的正弦和余弦；因此，我们首先要导入一个数学模块：`import math`。

1.  以下代码描述了定义旋转方法的方式；在其中，我们添加了注释以使这个操作对您清晰明了。首先，我们用`angle*π/ 180.0`命令/公式将角度转换为弧度。之后，我们获取了向量类的*x*和*y*坐标，并执行了`x = x*cosθ - y*sinθ`和`y = y*cosθ + x*sinθ`操作：

```py
      import math
      def rotate(self, angle):
              """Rotate vector counter-clockwise by angle (in-place).

              >>> v = vector(1, 2)
              >>> v.rotate(90)
              >>> v == vector(-2, 1)
              True

              """
              if self._hash is not None:
                  raise ValueError('cannot rotate vector after hashing')
              radians = angle * math.pi / 180.0
              cosine = math.cos(radians)
              sine = math.sin(radians)
              x = self.x
              y = self.y
              self.x = x * cosine - y * sine
              self.y = y * cosine + x * sine
```

数学公式*x = x*cosθ - y*sin**θ*在向量运动中非常重要。这个公式用于为游戏角色提供旋转运动。*x*cosθ*代表基础*x*轴运动，而*y*sinθ*代表垂直*y*轴运动。因此，这个公式实现了在二维平面上以角度θ旋转一个点。

最后，我们完成了两个方法：`move()`和`rotate()`。这两种方法完全独特，但它们都代表向量运动。`move()`方法实现了`__iadd_()`魔术函数，而`rotate()`方法具有自己的自定义三角函数实现。这两种方法的组合可以形成游戏角色在画布或游戏屏幕上的完整运动。为了构建任何类型的 2D 游戏，我们必须实现类似的运动。现在，我们将制作一个蚂蚁游戏的简单动画，以开始我们的游戏冒险之旅。

以下步骤描述了制作 2D 游戏动画的过程：

1.  首先，您必须导入必要的模块。由于我们必须为先前制作的`move()`方法提供随机向量坐标，我们可以预测我们将需要一个随机模块。

1.  之后，我们需要另一个模块——`turtle`模块，它将允许我们调用`ontimer`和`setup`等方法。我们还需要向量类的方法，即`move()`和`rotate()`。

1.  如果该类维护在任何其他模块或文件中，我们必须导入它。创建两个文件：`base.py`用于向量运动和`animation.py`用于动画。然后，导入以下语句：

```py
      from random import *
      from turtle import *
      from base import vector
```

1.  前两个语句将从 random 和 turtle 模块中导入所有内容。第三个语句将从基本文件或模块中导入向量类。

1.  接下来，我们需要为游戏角色定义初始位置以及其目标。它应该被初始化为向量类的一个实例：

```py
      ant = vector(0, 0) #ant is character
      aim = vector(2, 0) #aim is next position
```

1.  现在，您需要定义 wrap 方法。该方法以*x*和*y*位置作为参数，称为`value`，并返回它。在即将推出的游戏中，如 flappy bird 和 Pong，我们将扩展此功能，并使其将值环绕在某些边界点周围：

```py
      def wrap(value):
          return value 
```

1.  游戏的主控单元是`draw()`函数，它调用一个方法来使游戏角色移动。它还为游戏绘制屏幕。我们将从`Vector`类中调用`move`和`rotate`方法。从 turtle 模块中，我们将调用`goto`、`dot`和`ontimer`方法。`goto`方法将在游戏屏幕上的指定位置移动海龟画笔，`dot`方法在调用时创建指定长度的小点，`ontimer(function, t)`方法将安装一个定时器，在`t`毫秒后调用该函数：

```py
      def draw():
          "Move ant and draw screen."
          ant.move(aim)
          ant.x = wrap(ant.x)
          ant.y = wrap(ant.y)

          aim.move(random() - 0.5)
          aim.rotate(random() * 10 - 5)

          clear()
          goto(ant.x, ant.y)
          dot(10)

          if running:
              ontimer(draw, 100)
```

1.  在上述代码中，`running`变量尚未声明。我们现在将在`draw()`方法的定义之外进行声明。我们还将使用以下代码设置游戏屏幕：

```py
      setup(420, 420, 370, 0)
      hideturtle()
      tracer(False)
      up()
      running = True
      draw()
      done()
```

最后，我们完成了一个简单的 2D 动画。它由一个长度为 10 像素的简单点组成，但更重要的是，它具有附加的运动，这是在`Vector`类中实现魔术函数的结果。下一节将教我们如何使用本节中实现的魔术函数来制作更健壮的游戏，即蛇游戏。我们将使用 turtle 模块和魔术函数制作蛇游戏。

# 使用 Turtle 升级蛇游戏

事实证明，在本书的前几章中我们一直在构建贪吃蛇游戏：在第五章中，使用 curses 模块学习贪吃蛇游戏；在第六章中，面向对象编程；以及在第七章中，通过属性和列表推导式进行改进。我们从 curses 模块开始(第五章，*学习使用 curses 构建贪吃蛇游戏*)，并使用面向对象的范例进行修改。curses 模块能够提供基于字符的终端游戏屏幕，这最终使游戏角色看起来很糟糕。尽管我们学会了如何使用 OOP 和 curses 构建逻辑，以及制作贪吃蛇游戏，但应该注意到游戏主要关注视觉：玩家如何看到角色并与之交互。因此，我们的主要关注点是使游戏具有视觉吸引力。在本节中，我们将尝试使用 turtle 模块和向量化移动来升级贪吃蛇游戏。由于在贪吃蛇游戏中只有一种可能的移动方式，即通过按**左、右、上**或**下**键进行直线移动，我们不必在基本文件的向量类中定义任何新内容。我们之前创建的`move()`方法足以为贪吃蛇游戏提供移动。

让我们开始使用 turtle 模块和`Vector`类编写贪吃蛇游戏，按照以下步骤进行：

1.  像往常一样，首先导入必要的模块，如下面的代码所示。您不必先导入所有内容；我们也可以在编写其他内容时一起导入，但一次导入所有内容是一个好习惯，这样我们之后就不会忘记任何东西：

```py
      from turtle import *
      from random import randrange
      from base import vector
```

1.  现在，让我们进行一些头脑风暴。我们暂时不能使用精灵或图像。在开始使用 Pygame 之后，我们将在即将到来的章节中学习这些内容。现在，我们必须制作一个代表 2D 蛇角色的形状。您必须打开`base.py`文件，在那里我们创建了`Vector`类并定义了`Square`方法。请注意，`Square`方法是在`Vector`类之外声明的。以下代码是使用 turtle 方法创建正方形形状的简单实现：

```py
      def square(x, y, size, name):
          """Draw square at `(x, y)` with side length `size` and fill color 
           `name`.

          The square is oriented so the bottom left corner is at (x, y).

          """
          import turtle
          turtle.up()
          turtle.goto(x, y)
          turtle.down()
          turtle.color(name)
          turtle.begin_fill()

          for count in range(4):
              turtle.forward(size)
              turtle.left(90)

          turtle.end_fill()
```

1.  接下来，在贪吃蛇游戏模块中导入这个新方法。现在，我们可以在贪吃蛇游戏的 Python 文件中调用 square 方法：

```py
      from base import square
```

1.  导入所有内容后，我们将声明变量，如 food、snake 和 aim。food 表示向量坐标，是`Vector`类的一个实例，例如 vector(0,0)。snake 表示蛇角色的初始向量位置，即(vector(10,0))，而蛇的身体必须是向量表示的列表，即(vector(10,0)、vector(10,1)和 vector(10,2))表示长度为 3 的蛇。`aim`向量表示必须根据用户的键盘操作添加或减去到当前蛇向量的单位：

```py
      food = vector(0, 0)
      snake = [vector(10, 0)]
      aim = vector(0, -10)
```

1.  在`snake-Python`文件（主文件）中导入所有内容并声明其属性后，我们将开始定义贪吃蛇游戏的边界，如下所示：

```py
      def inside(head):
          "Return True if head inside boundaries."
          return -200 < head.x < 190 and -200 < head.y < 190
```

1.  您还应该定义贪吃蛇游戏的另一个重要方法，即`move()`，因为这将负责在游戏屏幕上移动贪吃蛇角色，如下所示：

```py
      def move():
          "Move snake forward one segment."
          head = snake[-1].copy()
          head.move(aim)

          if not inside(head) or head in snake:
              square(head.x, head.y, 9, 'red')
              update()
              return

          snake.append(head)

          if head == food:
              print('Snake:', len(snake))
              food.x = randrange(-15, 15) * 10
              food.y = randrange(-15, 15) * 10
          else:
              snake.pop(0)

          clear()

          for body in snake:
              square(body.x, body.y, 9, 'black')

          square(food.x, food.y, 9, 'green')
          update()
          ontimer(move, 100)
```

1.  让我们逐行理解代码：

+   在`move`方法的开始，我们获取了`snakehead`并执行了一个复制操作，这个操作是在`Vector`类中定义的，我们让蛇自动向前移动了一个段落，因为我们希望蛇在用户开始玩游戏时自动移动。

+   之后，`if not inside(head) or head in snake`语句用于检查是否有任何碰撞。如果有，我们将通过将`红色`渲染到蛇上来返回。

+   在语句的下一行`head == food`中，我们检查蛇是否能够吃到食物。一旦玩家吃到食物，我们将在另一个随机位置生成食物，并在 Python 控制台中打印分数。

+   在`for body in snake: ..`语句中，我们循环遍历了蛇的整个身体，并将其渲染为`黑色`。

+   在`Vector`类内部定义的`square`方法被调用以为游戏创建食物。

+   在代码的最后一条语句中，调用了`ontimer()`方法，该方法接受`move()`函数，并将安装一个定时器，每 100 毫秒调用一次`move`方法。

1.  在定义了`move()`方法之后，您必须设置游戏屏幕并处理乌龟屏幕。与`setup`方法一起传递的参数是`宽度`、`高度`、`setx`和`sety`位置：

```py
      setup(420, 420, 370, 0)
      hideturtle()
      tracer(False)
```

1.  我们游戏的最后部分是处理用户事件。我们必须让用户玩游戏；因此，每当用户从键盘输入时，我们必须调用适当的函数。由于 Snake 是一个简单的游戏，只有几个移动，我们将在下一节中介绍它。一旦用户按下任意键，我们必须通过改变蛇的方向来处理它。因此，我们必须为处理用户操作制作一个快速的方法。以下的`change()`方法将根据用户事件改变蛇的方向。在这里，我们使用了 turtle 模块提供的`listen`接口，它将监听任何传入的用户事件或键盘输入。`onkey()`接受一个函数，该函数将根据用户事件调用 change 方法。例如，当按下`Up`键时，我们将通过增加当前`y`值 10 个单位来改变*y*坐标：

```py
      def change(x, y):
          "Change snake direction."
          aim.x = x
          aim.y = y

      listen()
      onkey(lambda: change(10, 0), 'Right')
      onkey(lambda: change(-10, 0), 'Left')
      onkey(lambda: change(0, 10), 'Up')
      onkey(lambda: change(0, -10), 'Down')
      move()
      done()
```

现在是时候运行我们的游戏了，但在此之前，请记住将包含`vector`和`square`类的文件（以及包含 Snake 游戏的文件）放在同一个目录中。游戏的输出看起来像这样：

![](img/f5bbd336-8dfb-4c54-aeb5-520decddd003.png)

除了乌龟图形，我们还可以在 Python 终端中打印分数：

![](img/984005d2-933c-48b4-9b5d-73f9e654662a.png)

现在我们已经通过使用 Python 模块和面向对象编程范式提供的多种方法来完成了 Snake 游戏，我们可以在即将到来的游戏中一次又一次地重复使用这些东西。在`base.py`文件中定义的`Vector`类可以在许多 2D 游戏中反复使用。因此，代码的重复使用是面向对象编程提供的主要优点之一。我们将在接下来的几节中只使用`Vector`类制作几个游戏，例如乒乓球和飞翔的小鸟。在下一节中，我们将从头开始构建乒乓球游戏。

# 探索乒乓球游戏

现在我们已经通过使用 Python 模块和面向对象编程范式提供的多种方法来完成了 Snake 游戏（尽管它很陈词滥调，但它非常适合掌握 2D 游戏编程的知识），现在是时候制作另一个有趣的游戏了。我们将在本节中介绍的游戏是乒乓球游戏。如果您以前玩过，您可能会发现更容易理解我们将在本节中介绍的概念。对于那些以前没有玩过的人，不用担心！我们将在本节中涵盖一切，这将帮助您制作自己的乒乓球游戏并玩它，甚至与朋友分享。以下的图表是乒乓球游戏的图形表示：

![](img/4e32203e-d7f5-4984-8af8-cbba3d9b6b76.png)

前面的图表描述了乒乓游戏的游戏场地，其中两个玩家是两个矩形。他们可以上下移动，但不能左右移动。中间的**点**是球，必须由任一玩家击中。在这个游戏中，我们必须为游戏角色的两种运动类型解决问题：

+   对于球来说，它可以在任何位置移动，但如果任一方的玩家未接到球，他们将输掉比赛，而对方玩家将获胜。

+   对于玩家，他们只能向上或向下移动：应该处理两个玩家的四个键盘动作。

除了运动之外，为游戏指定边界甚至更加棘手。水平线可以上下移动，是球必须击中并在另一个方向上反射的位置，但如果球击中左侧或右侧的垂直边界，游戏应该停止，错过球的玩家将输掉比赛。现在，让我们进行头脑风暴，以便在实际开始编码之前了解必要的要点：

+   创建一个随机函数，它可以返回一个随机值，但在屏幕高度和宽度确定的范围内。从这个函数返回的值可能对使球在游戏中进行随机移动很有用。

+   创建一个方法，在屏幕上绘制两个矩形，实际上是我们游戏的玩家。

+   应该声明第三个函数，它将绘制游戏并将乒乓球移动到屏幕上。我们可以使用在先前制作的`Vector`类中定义的`move()`方法，该方法将移动向量（就地）。

现在我们已经完成了后勤工作，可以开始编码了。按照以下步骤制作自己的乒乓游戏：

1.  首先导入必要的模块，即 random、turtle 和我们自定义的名为`base`的模块，其中包含一堆用于向量运动的方法：

```py
      from random import choice, random
      from turtle import *
      from base import vector
```

1.  以下代码表示`value()`方法的定义，以及三个变量的赋值。`value()`方法将在(-5, -3)和(3, 5)之间随机生成值。这三个赋值语句根据它们的名称是可以理解的：

+   第一个语句表示球的初始位置。

+   第二个语句是球的进一步目标。

+   第三个语句是`state`变量，用于跟踪两个玩家的状态：

```py
      def value():
          "Randomly generate value between (-5, -3) or (3, 5)."
          return (3 + random() * 2) * choice([1, -1])
      ball = vector(0, 0)
      aim = vector(value(), value())
      state = {1: 0, 2: 0}
```

1.  下一个函数很有趣；这将在游戏屏幕上呈现矩形形状。我们可以使用 turtle 模块及其方法来呈现任何形状，如下所示：

```py
      def rectangle(x, y, width, height):
          "Draw rectangle at (x, y) with given width and height."
          up()
          goto(x, y)
          down()
          begin_fill()
          for count in range(2):
              forward(width)
              left(90)
              forward(height)
              left(90)
          end_fill()
```

1.  制作绘制矩形的函数后，我们需要制作一个新的方法，该方法可以调用在前面步骤中定义的方法。除此之外，新方法还应该将乒乓球无缝地移动到游戏屏幕上：

```py
      def draw():
          "Draw game and move pong ball."
          clear()
          rectangle(-200, state[1], 10, 50)
          rectangle(190, state[2], 10, 50)

          ball.move(aim)
          x = ball.x
          y = ball.y

          up()
          goto(x, y)
          dot(10)
          update()
```

1.  现在，是时候解决游戏的主要难题了：当球击中水平和垂直边界，或者当球击中玩家的矩形球拍时会发生什么？我们可以使用`setup`方法创建具有自定义高度和宽度的游戏屏幕。以下代码应该添加到`draw()`函数中：

```py
      #when ball hits upper or lower boundary  
      #Total height is 420 (-200 down and 200 up)
          if y < -200 or y > 200: 
              aim.y = -aim.y
      #when ball is near left boundary
          if x < -185:
              low = state[1]
              high = state[1] + 50

              #when player1 hits ball
              if low <= y <= high:
                  aim.x = -aim.x
              else:
                  return
      #when ball is near right boundary
          if x > 185:
              low = state[2]
              high = state[2] + 50

              #when player2 hits ball
              if low <= y <= high:
                  aim.x = -aim.x
              else:   
                  return

          ontimer(draw, 50)
```

1.  现在我们已经解决了游戏角色的移动问题，我们需要制作游戏屏幕并找到处理用户事件的方法。以下代码将设置游戏屏幕，该屏幕从 turtle 模块中调用：

```py
      setup(420, 420, 370, 0)
      hideturtle()
      tracer(False)
```

1.  制作游戏屏幕后，我们必须通过制作自定义函数来监听和处理用户的键盘事件。我们将制作`move()`函数，该函数将通过在调用此函数时传递的一定数量的单位来移动玩家的位置。这个移动函数将处理矩形球拍的上下移动：

```py
      def move(player, change):
          "Move player position by change."
          state[player] += change
```

1.  最后，我们将使用 turtle 方法提供的`listen`接口来处理传入的键盘事件。由于有四种可能的移动，即每个玩家的上下移动，我们将保留四个键盘键[*W*、*S*、*I*和*K*]，这些键将由 turtle 内部附加监听器，如下面的代码所示：

```py
      listen()
      onkey(lambda: move(1, 20), 'w')
      onkey(lambda: move(1, -20), 's')
      onkey(lambda: move(2, 20), 'i')
      onkey(lambda: move(2, -20), 'k')
      draw()
      done()
```

前面的步骤非常简单易懂，但让我们更加流畅地掌握*步骤 4*和*步骤 5*中定义的概念。在*步骤 4*中，`clear()`方法之后的前两行代码将创建指定高度和宽度的矩形几何形状。`state[1]`代表第一个玩家，而`state[2]`代表第二个玩家。`ball.move(aim)`语句是对矢量类内声明的`move`方法的调用。

这个方法调用将执行指定矢量之间的加法，结果是直线运动。`dot(10)`语句将创建一个宽度为 10 个单位的球。

同样，在*步骤 5*中，我们使用了`>>> setup(420, 420, 370, 0)`语句来创建一个宽度为 420px，高度为 420px 的屏幕。当球击中上下边界时，必须改变方向一定量，而该量恰好是当前*y*的负值（*-y*改变方向）。然而，当球击中左边界或右边界时，游戏必须终止。在检查上下边界之后，我们对*x*坐标进行比较，并检查低和高状态。如果球在这些值下面，它必定与球拍碰撞，否则我们返回`from`函数。确保将此代码添加到先前定义的`draw()`函数中。

当您运行 Pong 游戏文件时，您会看到两个屏幕；一个屏幕将有一个乌龟图形屏幕，其中包含两个玩家准备玩您自己的 Pong 游戏。输出将类似于我们在头脑风暴 Pong 游戏时之前看到的图表。现在您已经了解了处理键盘操作的方式，以及使用 turtle 的`ontimer`函数调用自定义函数，让我们做一些新的事情，这将有一个控制器。它将监听屏幕点击操作并对其做出响应。我们在诸如 Flappy Bird 这样的游戏中需要这个功能，用户在屏幕上点击并改变鸟的位置。

# 理解 Flappy Bird 游戏

每当我们谈论有屏幕点击操作或屏幕点击操作的游戏时，Flappy Bird 就会浮现在脑海中。如果您以前没有玩过，确保您在[`flappybird.io/`](https://flappybird.io/)上查看它，以便熟悉它。尽管您在该网站看到的界面与我们将在本节中制作的 Flappy Bird 游戏不同，但不用担心——在学习 Python 的 GUI 模块*Pygame*之后，我们将模拟其界面。但现在，我们将使用 Python turtle 模块和矢量运动制作一个简单的 2D Flappy Bird 游戏。我们一直在使用`onkey`方法来处理键盘操作，在前面的部分中，我们使用`onkey`方法来嵌入特定键盘键的监听器。

然而，也有一些可以使用鼠标操作玩的游戏——通过点击游戏屏幕。在本节中，我们将按照以下步骤创建 Flappy，这是一款受到 Flappy Bird 启发的游戏：

1.  首先，您应该为游戏玩法定义一个边界。您可以创建一个函数，该函数以矢量点作为参数，并检查它是否在边界内，然后相应地返回`True`或`False`。

1.  您必须制作一个渲染函数，用于将游戏角色绘制到屏幕上。正如我们所知，turtle 无法处理 GUI 中的许多图像或精灵；因此，您的游戏角色将类似于几何形状。您可以通过制作任何形状来代表您的鸟角色。如果可能的话，尽量使它小一些。

1.  制作了一个渲染函数之后，您需要创建一个能够更新对象位置的函数。这个函数应该能够处理`tap`动作。

我们可以在整个 Flappy Bird 游戏的编码过程中使用预定义的`Vector`蓝图。之前的路线图清楚地暗示了我们可以通过定义三个函数来制作一个简单的 Flappy Bird 游戏。让我们逐个定义这些函数：

1.  首先，您需要设置屏幕。这个屏幕代表了输出游戏控制台，在这里您将玩我们的 Flappy Bird 游戏。您可以使用海龟模块通过`setup()`来创建一个游戏屏幕。让我们创建一个宽度为 420 像素，高度为 420 像素的屏幕：

```py
      from turtle import *
      setup(420, 420, 370, 0)
```

1.  您应该定义一个函数，用来检查用户是否在边界内点击或触摸。这个函数应该是一个布尔值，如果点击点在边界内，应该返回`True`；否则，应该返回`False`：

```py
      def inside(point):
          "Return True if point on screen."
          return -200 < point.x < 200 and -200 < point.y < 200
```

1.  我已经建议您如果以前没有玩过 Flappy Bird 游戏，可以去试试。在玩游戏时，您会发现游戏的目标是保护*小鸟*角色免受障碍物的影响。在现实世界游戏中，我们有垂直管道形式的障碍物。由于我们在使用海龟模块编码时没有足够的资源来使用这样的精灵或界面，我们将无法在本节中使用。正如我已经告诉过您的，我们将在学习 Pygame 时自己制作很酷的界面，但现在，我们将高度关注游戏逻辑，而不是 GUI。因此，我们将给游戏角色一些随机形状；小圆形状的小鸟角色和大圆形状的障碍物。小鸟将从向量类实例化，表示其初始位置。球（障碍物）必须作为列表制作，因为我们希望障碍物在小鸟的路径上：

```py
      bird = vector(0, 0)
      balls = []
```

1.  现在您已经熟悉了游戏角色，可以通过创建一些函数来渲染它们。在函数中，我们已经传递了`alive`作为一个变量，它将是一个布尔值，这将检查玩家是否死亡。如果小鸟还活着，我们使用`goto()`跳转到该位置，并用绿色渲染一个点。如果小鸟死了，我们用红色渲染这个点。以下代码中的 for 循环将渲染一些障碍物：

```py
      def draw(alive):
          "Draw screen objects."
          clear()

          goto(bird.x, bird.y)

          if alive:
              dot(10, 'green')
          else:
              dot(10, 'red')

          for ball in balls:
              goto(ball.x, ball.y)
              dot(20, 'black')

          update()
```

1.  正如我们在之前的蓝图中讨论的，接下来是游戏的主控制器。这个函数必须执行多个任务，但所有这些任务都与更新对象的位置有关。对于那些以前没有玩过 Flappy Bird 的用户来说，他们可能很难理解下面的代码；这就是为什么我鼓励您去玩原版 Flappy Bird 游戏。如果您检查游戏中小鸟的移动，您会发现它只能在*y*轴上移动，即上下移动。同样对于障碍物，它们必须从右向左移动，就像现实世界游戏中的垂直管道一样。以下的`move()`函数包括了小鸟的初始运动。最初，我们希望它下降 5 个单位，并相应地减少。对于障碍物的部分，我们希望它向左移动 3 个单位：

```py
      from random import *
      from base import vector #for vectored motion 
      def move():
          "Update object positions."
          bird.y -= 5

          for ball in balls:
              ball.x -= 3
```

1.  您必须在`move`函数内明确地创建多个障碍物。由于障碍物应该随机生成，我们可以使用随机模块来创建它：

```py
       if randrange(10) == 0:
          y = randrange(-199, 199)
          ball = vector(199, y)
          balls.append(ball)    #append each obstacles to list
```

1.  接下来，我们需要检查玩家是否能够阻止小鸟触碰障碍物。检查的方法很简单。如果球或障碍物超出了左边界，我们可以将它从球的列表中移除。最初，我们制作了`inside`函数来检查任何点是否在边界内；现在，我们可以用它来检查障碍物是否在边界内。它应该看起来像这样：

```py
      while len(balls) > 0 and not inside(balls[0]):
          balls.pop(0)
```

1.  请注意，我们已经为障碍物添加了一个条件；现在是时候添加一个条件来检查小鸟是否还活着。如果小鸟掉下来并触及下边界，程序应该终止：

```py
      if not inside(bird):
          draw(False)
          return
```

1.  现在，我们将添加另一个条件——检查障碍物是否与小鸟发生了碰撞。有几种方法可以做到这一点，但现在，我们将通过检查球和障碍物的位置来实现这一点。首先，您必须检查障碍物和小鸟的大小：障碍物或球的大小为 20 像素，小鸟的大小为 10 像素（在第 4 点定义）；因此，我们可以假设它们在彼此之间的距离为 0 时发生了碰撞。因此，`>>> if abs(ball - bird) < 15`表达式将检查它们之间的距离是否小于 15（考虑到球和小鸟的宽度）：

```py
      for ball in balls:
          if abs(ball - bird) < 15:         
              draw(False)
              return
      draw(True)
      ontimer(move, 50) #calls move function at every 50ms
```

1.  现在我们已经完成了更新对象的位置，我们需要处理用户事件——这是当玩家轻击游戏屏幕时应该实现的内容。当用户轻击屏幕时，我们希望小鸟上升一定数量的像素。传递给轻击函数（*x，y*）的参数是游戏屏幕上点击点的坐标：

```py
      def tap(x, y):
          "Move bird up in response to screen tap."
          up = vector(0, 30)
          bird.move(up)
```

1.  最后，是时候使用 turtle 模块添加一个监听器了。我们将使用`onscreenclick()`函数，它将以用户定义的任何函数作为参数（在我们的情况下是`tap()`函数），并将以画布上点击点的坐标（*x，y*）调用该函数。我们已经使用 tap 函数来调用这个监听器：

```py
      hideturtle()
      up()
      tracer(False)
      onscreenclick(tap)
      move()
      done()
```

这似乎是很多工作，对吧？的确是。在本节中，我们已经涵盖了很多内容：定义边界的方法，渲染游戏对象，更新对象位置以及处理轻击事件或鼠标事件。我觉得我们已经学到了很多关于使用 turtle 模块构建 2D 游戏的逻辑。尽管使用 turtle 模块制作的游戏并不是很吸引人，但我们通过构建这些游戏学到的逻辑将在接下来的章节中反复使用。在这类游戏中，我们并不太关心界面，而是会在 Python shell 中运行我们的游戏并观察它的外观。上述程序的结果将是这样的：

![](img/6d05307b-b062-4ca1-9cbb-bf7c212d3805.png)

**错误消息**：没有名为'base'的模块。这是因为您还没有将您的`Base`模块（包含我们在*使用 Turtle 模块进行简单动画*部分中制作的`Vector`类的 Python 文件）和 Python 游戏文件添加到同一个目录中。确保您创建一个新目录并将这两个文件存储在一起，或者从以下 GitHub 链接获取代码：[`github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter10`](https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter10)。

对于由 Turtle 制作的游戏，修改的空间很小。但我强烈建议您自行测试游戏，并发现可能的修改。如果您发现了任何修改，尝试实现它们。在下一节中，我们将介绍如何正确测试游戏并应用修改，使这些游戏比以前更加稳固。

# 游戏测试和可能的修改

许多人错误地认为，要成为一名熟练的游戏测试人员，您应该是一名游戏玩家。这在某种程度上可能是正确的，但大多数情况下，游戏测试人员并不关心游戏的前端设计。他们主要关注处理游戏服务器和客户端计算机之间的*数据*通信的后端部分。我将带您了解我们的 Pong 游戏的游戏测试和修改过程，同时涵盖以下几点：

1.  **增强游戏角色**：以下代码代表游戏角色的新模型。我们仅使用乌龟模块来实现它。*挡板*是代表乒乓球游戏玩家的矩形框。有两个，即挡板 A 和挡板 B：

```py
      import turtle
      # Paddle A
      paddle_a = turtle.Turtle()
      paddle_a.speed(0)
      paddle_a.shape('square')
      paddle_a.color('white')
      paddle_a.penup()
      paddle_a.goto(-350, 0)
      paddle_a.shapesize(5, 1)

      # Paddle B
      paddle_b = turtle.Turtle()
      paddle_b.speed(0)
      paddle_b.shape('square')
      paddle_b.color('white')
      paddle_b.penup()
      paddle_b.goto(350, 0)
      paddle_b.shapesize(5, 1)
```

1.  **在游戏中添加主角**（一个球）：与创建 A 和 B 挡板类似，我们将使用乌龟模块以及`speed()`、`shape()`和`color()`等命令来创建一个球角色并为其添加功能：

```py
      # Ball
      ball = turtle.Turtle()
      ball.speed(0)
      ball.shape('circle')
      ball.color('white')
      ball.penup()
      ball.dx = 0.15
      ball.dy = 0.15
```

1.  **为游戏添加得分界面**：我们将使用乌龟画笔为每个玩家得分绘制一个界面。以下代码包括了从乌龟模块调用的方法，即`write()`方法，用于写入文本。它将*arg*的字符串表示放在指定位置：

```py
      # Pen
      pen = turtle.Turtle()
      pen.speed(0)
      pen.color('white')
      pen.penup()
      pen.goto(0, 260)
      pen.write("Player A: 0  Player B: 0", align='center', 
        font=('Courier', 24, 'bold'))
      pen.hideturtle()

      # Score
      score_a = 0
      score_b = 0
```

1.  **键盘绑定与适当的动作**：在以下代码中，我们已经将键盘与适当的函数绑定。每当按下键盘键时，将使用`onkeypress`调用指定的函数；这就是**事件处理**。对于`paddle_a_up`和`paddle_b_up`等方法感到困惑吗？一定要复习*乒乓球游戏*部分：

```py
      def paddle_a_up():
          y = paddle_a.ycor()
          y += 20
          paddle_a.sety(y)

      def paddle_b_up():
          y = paddle_b.ycor()
          y += 20
          paddle_b.sety(y)

      def paddle_a_down():
          y = paddle_a.ycor()
          y += -20
          paddle_a.sety(y)

      def paddle_b_down():
          y = paddle_b.ycor()
          y += -20
          paddle_b.sety(y)

      # Keyboard binding
      wn.listen()
      wn.onkeypress(paddle_a_up, 'w')
      wn.onkeypress(paddle_a_down, 's')
      wn.onkeypress(paddle_b_up, 'Up')
      wn.onkeypress(paddle_b_down, 'Down')
```

1.  **乌龟屏幕和主游戏循环**：以下几个方法调用代表了乌龟屏幕的设置：游戏的屏幕大小和标题。`bgcolor()`方法将以指定颜色渲染乌龟画布的背景。这里，屏幕的背景将是黑色：

```py
      wn = turtle.Screen()
      wn.title('Pong')
      wn.bgcolor('black')
      wn.setup(width=800, height=600)
      wn.tracer(0)
```

主游戏循环看起来有点棘手，但如果你仔细看，你会发现我们已经了解了这个概念。主循环从设置球的运动开始。`dx`和`dy`的值是其运动的恒定单位。对于**#边界检查**部分，我们首先检查球是否击中了上下墙壁。如果是，我们就改变它的方向，让球重新进入游戏。对于**#2：对于右边界**，我们检查球是否击中了右侧的垂直边界，如果是，我们就将得分写给另一个玩家，然后结束游戏。左边界也是一样的：

```py
while True:
    wn.update()

    # Moving Ball
    ball.setx(ball.xcor() + ball.dx)
    ball.sety(ball.ycor() + ball.dy)

    # Border checking
    #1: For upper and lower boundary
    if ball.ycor() > 290 or ball.ycor() < -290:
        ball.dy *= -1

    #2: for RIGHT boundary
    if ball.xcor() > 390:
        ball.goto(0, 0)
        ball.dx *= -1
        score_a += 1
        pen.clear()
        pen.write("Player A: {}  Player B: {}".format(score_a, score_b), 
          align='center', font=('Courier', 24, 'bold'))

    #3: For LEFT boundary
    if ball.xcor() < -390:
        ball.goto(0, 0)
        ball.dx *= -1
        score_b += 1
        pen.clear()
        pen.write("Player A: {}  Player B: {}".format(score_a, score_b), 
          align='center', font=('Courier', 24, 'bold'))
```

现在，我们必须处理球击中玩家的挡板的情况。以下两个条件代表了挡板和球之间的碰撞：前一个是针对挡板 B 的，后一个是针对挡板 A 的。由于挡板 B 位于屏幕的右侧，我们检查球的坐标是否与挡板的坐标加上其宽度相同。如果是，我们使用`ball.dx *= -1`命令来改变球的方向。`setx`方法将把球的第一个坐标改为**340**，而将*y*坐标保持不变。这里的逻辑与我们制作贪吃蛇游戏时使用的逻辑类似，当蛇头与食物碰撞时：

```py
# Paddle and ball collisions
    if (ball.xcor() > 340 and ball.xcor() < 350) and (ball.ycor() 
        < paddle_b.ycor() + 60 and ball.ycor() > paddle_b.ycor() -60):

        ball.setx(340)
        ball.dx *= -1

    if (ball.xcor() < -340 and ball.xcor() > -350) and (ball.ycor() 
        < paddle_a.ycor() + 60 and ball.ycor() > paddle_a.ycor() -60):

        ball.setx(-340)
        ball.dx *= -1
```

实施如此严格的修改的好处不仅在于增强游戏角色，还在于控制不一致的帧速率——即连续图像（帧）在显示屏上出现的速率。我们将在即将到来的关于*Pygame*的章节中详细了解这一点，在那里我们将使用自己的精灵来定制基于乌龟的贪吃蛇游戏。在总结本章之前，让我们运行定制的乒乓球游戏并观察结果，如下所示：

![](img/ef58bc53-4a0b-4aaa-91bc-f82cd598211b.png)

# 总结

在本章中，我们探索了 2D 乌龟图形的世界，以及矢量运动。

我尽量使这一章尽可能全面，特别是在处理矢量运动时。我们创建了两个单独的文件；一个是`Vector`类，另一个是游戏文件本身。`Vector`类提供了一种表示*x*和*y*位置的 2D 坐标的方法。我们执行了多个操作，比如*move*和*rotation*，使用数据模型——覆盖了我们自定义的`Vector`类的实际行为。我们简要地观察了通过创建马里奥像素艺术来处理计算机像素的方法。我们制作了一个像素网格（列表的列表）来表示像素的位置，并最终使用 turtle 方法来渲染像素艺术。之后，我们通过定义一个独立的`Vector`类来制作了一个简单的动画，该类表示游戏角色的位置。我们在整个游戏过程中都使用了 turtle 模块和我们自定义的`Vector`类。虽然我觉得你已经准备好开始你的 2D 游戏程序员生涯了，但正如我们所说，“熟能生巧”，在你感到舒适之前，你需要大量尝试。

这一章对于我们所有想成为游戏程序员的人来说都是一个突破。我们学习了使用 Python 和 turtle 模块构建游戏的基础知识，学会了如何处理鼠标和键盘等不同的用户事件。最后，我们还学会了如何使用 turtle 模块创建不同的游戏角色。当你继续阅读本书时，你会发现 turtle 的这些概念是非常重要的，所以确保在继续之前复习它们。

在下一章中，我们将学习 Pygame 模块——这是使用 Python 构建交互式游戏最重要的平台。从下一章开始，我们将深入探讨一些话题，比如你可以加载图像或精灵，制作自己的游戏动画。你还会发现，与 C 或 C++相比，使用 Python 构建游戏是多么容易。
