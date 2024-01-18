# 了解PyOpenGL

几何形状和图形在游戏开发中起着至关重要的作用。当涉及到先进的图形技术的开发时，我们往往忽视它们的重要性。然而，许多流行的游戏仍然使用这些形状和图形来渲染游戏角色。数学概念，如变换、向量运动以及放大和缩小的能力，在操纵游戏对象时具有重要作用。Python有几个模块来支持这种操纵。在本章中，我们将学习一个强大的Python功能——PyOpenGL模块。

在探索PyOpenGL时，我们将学习如何使用基本图形（即顶点和边）创建复杂的几何形状。我们将从安装Python PyOpenGL并开始用它绘图开始。我们将使用它制作几个对象，如三角形和立方体。我们不会使用pygame来创建这些形状；相反，我们将使用纯数学概念来定义顶点和边的直角坐标点。我们还将探索不同的PyOpenGL方法，如裁剪和透视。我们将涵盖每一个方法，以了解PyOpenGL如何用于创建吸引人的游戏角色。

在本章结束时，您将熟悉创建基本图形的传统和数学方法。这种创建形状的方式为程序员和设计师提供了操纵他们的游戏对象和角色的能力。您还将学习如何在游戏中实现放大和缩小的功能，以及如何通过绘制几何基本图形来使用颜色属性。

本章将涵盖以下主题：

+   理解PyOpenGL

+   使用PyOpenGL制作对象

+   理解PyOpenGL方法

+   理解颜色属性

# 技术要求

您需要以下要求清单才能完成本章：

+   建议使用Pygame编辑器（IDLE）版本3.5+。

+   您将需要Pycharm IDE（参考[第1章](0ef9574b-5690-454e-971f-85748021018d.xhtml)，*了解Python-设置Python和编辑器*，了解安装过程）。

+   本章的代码资产可以在本书的GitHub存储库中找到：[https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter14](https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter14)

查看以下视频以查看代码的运行情况：

[http://bit.ly/2oJMfLM](http://bit.ly/2oJMfLM)

# 理解PyOpenGL

在过去，包含经过3D加速硬件处理的三维场景的图形程序是每个游戏程序员都想要的东西。尽管按照今天的标准来看这是正常的，但硬件与多年前并不相同。大部分游戏的图形必须使用低处理设备中的软件进行渲染。因此，除了创建这样的场景，渲染也需要相当长的时间，最终会使游戏变慢。游戏界面的出现，也被称为图形卡，为游戏行业带来了革命；程序员现在只需要关注界面、动画和自主游戏逻辑，而不必关心处理能力。因此，90年代后创建的游戏具有更丰富的游戏性和一丝人工智能（多人游戏）。

众所周知，图形卡可以处理渲染和优化场景等三维功能。但是，要使用这些功能，我们需要一个在我们的项目和这些接口之间进行通信的编程接口。我们在本章中要使用的**应用程序编程接口**（**API**）是OpenGL。OpenGL是一个跨平台（程序可以在任何机器上运行）的API，通常用于渲染2D和3D图形。该API类似于用于促进与图形处理单元的交互的库，并且通过使用硬件加速渲染来加速图形渲染方法。它作为图形驱动程序的一部分预装在大多数机器上，尽管您可以使用*GL视图* *实用程序*来检查其版本。在我们开始编写程序以便使用PyOpenGL绘制几何形状和图形之前，我们需要在我们的机器上安装它。

# 安装PyOpenGL

即使OpenGL已经存在于您的系统上，您仍然需要单独安装PyOpenGL模块，以便所需的OpenGL驱动程序和Python框架可以相互通信。Pycharm IDE提供了一个服务，可以定位Python解释器并安装PyOpenGL，从而消除了手动安装的开销。按照以下步骤在Pycharm IDE中安装PyOpenGL：

1.  单击顶部导航栏中的“文件”，然后单击“设置”。然后，将鼠标悬停在左侧导航窗口上，并选择项目：解释器选项。

1.  选择当前项目的Python解释器，即Python 3.8+（后跟您的项目名称），并从解释器下拉菜单旁边的菜单屏幕上按下添加（+）按钮。

1.  在搜索栏中搜索PyOpenGL，然后按“安装包”按钮。

或者，如果您想要外部安装PyOpenGL，您可以将其下载为Python蛋文件。

*Python蛋*是一个逻辑结构，包含了Python项目特定版本的发布，包括其代码、资源和元数据。有多种格式可用于物理编码Python蛋，也可以开发其他格式。但是，Python蛋的一个关键原则是它们应该是可发现和可导入的。也就是说，Python应用程序应该能够轻松高效地找出系统上存在哪些蛋，并确保所需蛋的内容是可导入的。

这些类型的文件被捆绑在一起，以创建可以通过简单的安装过程从**Python企业应用套件**（**PEAK**）下载的Python模块。要下载Python蛋文件，您必须下载Python `easy_install`模块。转到[http://peak.telecommunity.com/DevCenter/EasyInstall](http://peak.telecommunity.com/DevCenter/EasyInstall)，然后下载并运行`ez_setup.py`文件。成功安装easy install后，在命令行/终端中运行以下命令以安装PyOpenGL：

```py
easy_install PyOpenGL
```

Easy install不仅用于安装PyOpenGL，还可以借助它下载或升级大量的Python模块。例如，`easy_install` SQLObject用于安装SQL PyPi包。

通常情况下，当我们需要使用包时，我们需要将它们导入到我们的项目中。在这种情况下，您可以创建一个演示项目（`demo.py`）来开始测试OpenGL项目。这样我们就可以使用诸如代码可维护性和调试之类的功能，我们将使用Pycharm IDE制作PyOpenGL项目，而不是使用Python的内置IDE。打开任何新项目，并按照以下步骤检查PyOpenGL是否正在运行：

1.  使用以下命令导入PyOpenGL的每个类：

```py
      from OpenGL.GL import *
```

1.  现在，使用以下命令导入所需的OpenGL函数：

```py
      from OpenGL.GLU import *
```

1.  接下来，您应该将`pygame`导入到您的项目中：

```py
      from pygame.locals import *
```

1.  使用`pygame`命令为您的项目初始化显示：

```py
      import pygame
      from pygame.locals import *
      window_screen = pygame.display.set_mode((640, 480), 
        HWSURFACE|OPENGL|DOUBLEBUF)
```

1.  运行您的项目并分析结果。如果出现新屏幕，您可以继续制作项目。但是，如果提示说PyOpenGL未安装，请确保按照前面的安装过程进行操作。

前面的四行很容易理解。让我们逐一讨论它们。第一步非常简单-它告诉解释器导入PyOpenGL以及其多个类，这些类可用于不同的功能。以这种方式导入可以减少逐个导入PyOpenGL的每个类的工作量。第一个导入是强制性的，因为这一行导入以`gl`关键字开头的不同OpenGL函数。例如，我们可以使用诸如`glVertex3fv()`之类的命令，用于绘制不同的3D形状（我们稍后会介绍这个）。

导入语句的下一行，即`from OpenGL.GLU import *`，是为了我们可以使用以`glu`开头的命令，例如`gluPerspective()`。这些类型的命令对于更改显示屏的视图以及渲染的对象非常有用。例如，我们可以使用这样的`glu`命令进行裁剪和剪裁等转换。

类似于PyOpenGL GL库，GLU是一个Python库，用于探索相关数据集内部或之间的关系。它们主要用于在影响渲染对象的形状和尺寸的同时对显示屏进行更改。要了解有关GLU内部的更多信息，请查看其官方文档页面：[http://pyopengl.sourceforge.net/pydoc/OpenGL.GLU.html](http://pyopengl.sourceforge.net/pydoc/OpenGL.GLU.html)。

下一行只是将`pygame`导入到我们的项目中。使用OpenGL创建的表面是3D的，它需要`pygame`模块来渲染它。在使用`gl`或`glu`模块的任何命令之前，我们需要调用`pygame`模块使用`set_mode()`函数创建一个显示（感受`pygame`模块的强大）。由`pygame`模块创建的显示将是3D而不是2D，同时使用OpenGL库的`set_mode`函数。之后，我们告诉Python解释器创建一个OpenGL表面并将其作为`window_screen`对象返回。传递给`set_mode`函数的元组（高度，宽度）表示表面大小。

在最后一步，我希望您关注以下可选参数：

+   `HWSURFACE`：它在硬件中创建表面。主要用于创建加速的3D显示屏，但仅在全屏模式下使用。

+   `OPENGL`：它向pygame建议创建一个OpenGL渲染表面。

+   `DOUBLEBUF`：它代表双缓冲，pygame建议对`HWSURFACE`和`OPENGL`使用。它减少了屏幕上颜色闪烁的现象。

还有一些其他可选参数，如下：

+   `FULLSCREEN`：这将使屏幕显示渲染为全屏视图。

+   `RESIZABLE`：这允许我们调整窗口屏幕的大小。

+   `NOFRAME`：这将使窗口屏幕无边框，无控件等。有关pygame可选参数的更多信息，请访问[https://www.pygame.org/docs/ref/display.html#pygame.display.set_mode](https://www.pygame.org/docs/ref/display.html#pygame.display.set_mode)。

现在我们已经在我们的机器上安装了PyOpenGL并为屏幕对象设置了一个窗口，我们可以开始绘制对象和基本图形。

# 使用PyOpenGL制作对象

OpenGL主要用于绘制不同的几何形状或基元，所有这些都可以用于创建3D画布的场景。我们可以制作多边形（多边形）形状，如三角形、四边形或六边形。应该向基元提供多个信息，如顶点和边，以便PyOpenGL可以相应地渲染它们。由于与顶点和边相关的信息对于每个形状都是不同的，因此我们有不同的函数来创建不同的基元。这与pygame的2D函数（`pygame.draw`）不同，后者用于使用相同的单个函数创建多个形状。例如，三角形有三个顶点和三条边，而四边形有四个顶点。

如果您具有数学背景，对顶点和边的了解对您来说将是小菜一碟。但对于那些不了解的人来说，任何几何形状的顶点都是两条或两条以上线相交的角或点。例如，三角形有三个顶点。在下图中，**A**、**B**和**C**是三角形ABC的顶点。同样，边是连接一个顶点到另一个顶点的线段。在下面的三角形中，AB、BC和AC是三角形ABC的边：

![](Images/26d029c3-1800-45d8-aedd-b1a92e547f8a.png)

要使用PyOpenGL绘制这样的几何形状，我们需要首先调用一些基本的OpenGL基元，这些基元列在下面：

1.  首先，使用要绘制的任何基元调用`glBegin()`函数。例如，应调用`glBegin(GL_TRIANGLES)`来通知解释器我们将要绘制的三角形形状。

1.  关于顶点（A、B、C）的下一个重要信息对于绘制形状至关重要。我们使用`glVertex()`函数发送有关顶点的信息。

1.  除了有关顶点和边的信息之外，您还可以使用`glColor()`函数提供有关形状颜色的其他信息。

1.  在提供足够的基本信息之后，您可以调用`glEnd()`方法通知OpenGL已经提供了足够的信息。然后，它可以开始绘制指定的形状，如`glBegin`方法提供的常量所示。

以下代码是使用PyOpenGL绘制三角形形状的伪代码（参考前面的插图以了解PyOpenGL函数的操作）：

```py
#Draw a geometry for the scene
def Draw():
 #translation (moving) about 6 unit into the screen and 1.5 unit to left
     glTranslatef(-1.5,0.0,-6.0)
     glBegin(GL_TRIANGLES) #GL_TRIANGLE is constant for TRIANGLES 
     glVertex3f( 0.0, 1.0, 0.0) #first vertex 
     glVertex3f(-1.0, -1.0, 0.0) #second vertex 
     glVertex3f( 1.0, -1.0, 0.0) #third vertex 
     glEnd() 
```

下图显示了三角形的法线。法线是一个数学术语，表示单位向量（具有1的大小和方向，请参考[第10章](b6bfaeca-a5ea-4d39-a757-653f2e2be083.xhtml)，*使用海龟升级蛇游戏*，了解更多关于向量的信息）。这个信息（法线）很重要，因为它告诉PyOpenGL每个顶点的位置。例如，`glVertex3f(0, 1, 0)`会在*y*轴上放置一个顶点。因此，(*x*, *y*, *z*)表示*x*轴、*y*轴和*z*轴上的大小，如下所示：

![](Images/cfe2348d-185f-47f9-b2b2-73ed7cdd6841.png)

现在我们知道如何创建基本的三角形基元，让我们看一下以下表格，了解可以使用PyOpenGL绘制的其他不同类型的基元：

| **常量关键字** | **形状** |
| `GL_POINTS` | 将点或点绘制到屏幕上 |
| `GL_LINES` | 绘制线条（单独的线条） |
| `GL_TRIANGLES` | 绘制三角形 |
| `GL_QUADS` | 绘制四边形（四边形） |
| `GL_POLYGON` | 绘制多边形（任何边或顶点） |

现在我们能够使用基元常量绘制任何基元，前提是我们有关于它们顶点的信息。让我们创建以下四边形：

![](Images/3ae798eb-a9ce-4cdf-804d-e163c4c08409.png)

以下是绘制前述立方体基元的伪代码：

```py
glBegin(GL_QUADS)
glColor(0.0, 1.0, 0.0) # vertex at y-axis
glVertex(1.0, 1.0, 0.0) # Top left
glVertex(1.0, 1.0, 0.0) # Top right
glVertex(1.0, 1.0, 0.0) # Bottom right
glVertex(1.0, 1.0, 0.0) # Bottom left
glEnd()
```

在上一行代码中，我们首先定义了`GL_QUADS`常量，以通知PyOpenGL我们正在绘制的基本图元的名称。然后，我们使用`glColor`方法添加了颜色属性。同样，我们使用`glVertex`方法定义了立方体的四个主要顶点。作为`glVertex`方法的参数传递的坐标代表了平面上的*x*、*y*和*z*轴。

现在我们能够使用PyOpenGL绘制不同的几何形状，让我们了解PyOpenGL的不同渲染函数/基本图元，以便我们可以制作其他复杂的结构。

# 理解PyOpenGL方法

众所周知，计算机屏幕具有二维视图（高度和宽度）。为了显示由OpenGL创建的三维场景，场景必须经过几次矩阵变换，通常称为投影。这允许将3D场景呈现为2D视图。在各种变换方法中，常用于投影的有两种（裁剪和归一化）。这些矩阵变换应用于3D坐标系，并缩减为2D坐标系。`GL_PROJECTION`矩阵经常用于执行与投影相关的变换。投影变换的数学推导是另一回事，我们永远不会使用它们，但理解它的工作原理对于任何游戏程序员来说都是重要的。让我们来看看`GL_PROJECTION`的工作原理：

+   **裁剪**：这将把场景的顶点坐标转换为场景的裁剪坐标。裁剪是一个调整场景长度的过程，以便从`视口`（窗口显示）中裁剪掉一些部分。

+   **归一化**：这个过程被称为**标准化设备坐标**（**NDC**），它通过将裁剪坐标除以裁剪坐标的`w`分量来将裁剪坐标转换为设备坐标。例如，裁剪坐标x[c]、y[c]和z[c]通过与w[c]进行比较。不在-w[c]到+w[c]范围内的顶点被丢弃。这里的下标*c*表示裁剪坐标系。

因此，更容易推断矩阵变换的过程，包括`GL_PROJECTION`，包括两个步骤：裁剪，紧接着是归一化到设备坐标。以下图示了裁剪的过程：

![](Images/4af7b0fe-7069-405c-9ba1-131b483d844d.png)

我们可以清楚地观察到裁剪（有时称为剔除）的过程只在裁剪坐标中执行，这些坐标由2D视口的大小定义。要找出哪些裁剪坐标已被丢弃，我们需要看一个例子。假设*x*、*y*和*z*是裁剪坐标，它们的值与*w*（*x*、*y*）的坐标进行比较，决定任何顶点（或形状的一部分）是否保留在屏幕上或被丢弃。如果任何坐标位于-w[c]的值以下和+w[c]的值以上，那个顶点就被丢弃。在上图中，顶点A位于+w[c]之上，而顶点B和C位于-w[c]之下，因此两个顶点都被丢弃。此外，顶点D和E位于(-w[c]，+w[c])的值范围内，因此它们保留在视图中。w[c]的值由视口的宽度确定。因此，OpenGL的投影矩阵（`GL_PROJECTION`）接受3D坐标并执行投影，将其转换为可以呈现在2D计算机显示屏上的2D坐标。尽管可能会丢失一些信息，但它被认为是将3D场景渲染到2D屏幕上的最有效方法之一。

然而，我们还没有完成——在投影完成后，我们必须将3D场景转换为2D，这需要使用另一个OpenGL矩阵变换，称为`GL_MODELVIEW`。然而，这种转换的步骤是相当不同的。首先进行矩阵变换，将坐标系乘以*视距*。

为了将它们转换为2D组件，为每个*z*分量提供了。要理解模型视图矩阵，我们必须理解构成其组成部分的两个矩阵：模型矩阵和视图矩阵。模型矩阵在模型世界中执行多个转换，如旋转、缩放和平移，而视图矩阵调整相对于摄像机位置的场景。视图矩阵负责处理对象在玩家观看场景时的外观，类似于第一人称角色的屏幕/视点。

现在我们了解了OpenGL的变换矩阵，让我们制作一个简单的程序（`resize.py`），可以相应地调整显示屏的大小：

1.  首先导入OpenGL。

```py
      from OpenGL.GL import *
      from OpenGL.GLU import *
```

1.  制作一个简单的函数`change_View()`，以显示屏的大小为参数，如下所示：

```py
      def change_View():
          pass
```

1.  从*步骤3*到*步骤6*中的代码应该添加到`change_View()`函数中。添加一个对`ViewPort`的函数调用，它以初始值和显示大小为参数，如下所示：

```py
      glViewport(0, 0 , WIDTH, HEIGHT)
```

1.  现在，是时候添加投影矩阵了。要添加`GL_PROJECTION`，我们必须调用`glMatrixMode()`方法，检查被调用的矩阵的模式，如下所示：

```py
      glMatrixMode(GL_PROJECTION) #first step to apply projection matrix
```

1.  在应用投影矩阵后，应调用两个重要的方法，即`glLoadIdentity()`和`gluPerspective()`，它们为投影矩阵设置了“基准”：

```py
      aspect_ratio = float(width/height)
      glLoadIdentity()
      gluPerspective(40., aspect_ratio, 1., 800.)
```

1.  设置投影矩阵后，下一步是设置模型视图矩阵。可以通过调用`glMatrixMode()`方法激活模型视图矩阵模式：

```py
      glMatrixMode(GL_MODELVIEW)
      glLoadIdentity()
```

前面的六个步骤向我们展示了如何调整显示屏，将3D场景显示在2D显示屏中。*步骤1*和*步骤2*专注于导入OpenGL。在*步骤3*中，我们调用了`glViewport()`方法，并传递了一个参数，范围从(`0`, `0`)到(`width`, `height`)，这告诉OpenGL我们要使用整个屏幕来显示场景。下一步调用了`glMatrixMode()`方法，告诉OpenGL每次函数调用都将应用投影矩阵。

*步骤5*调用了两个新方法，正如`glLoadIdentity()`的签名所述，用于使投影矩阵成为单位矩阵，这意味着投影矩阵的所有坐标都应该更改为`1`。最后，我们调用另一个方法`gluPerspective()`，它设置了分类/标准投影矩阵。您可能已经注意到`gluPerspective()`方法以`glu`开头而不是`gl`，因此，此函数是从GLU库中调用的。`gluPerspective`方法传递了四个浮点参数，即相机视点的视场角，宽高比和两个裁剪平面点（近和远）。因此，裁剪是通过`gluPerspective`函数完成的。要了解裁剪是如何完成的，请参考我们在本主题开头讨论的星形几何形状的示例。

现在，是时候将我们学到的知识付诸实践，制作一个与PyOpenGL结构交互的程序。我们还将定义另一个属性，使对象更具吸引力。这被称为*颜色属性*。我们将定义一个立方体，以及关于顶点和边的数学信息。

# 理解颜色属性

在现实世界的场景中，与物体相关联的颜色有很多，但是计算机设备并不足够智能或者能力强大到可以区分和捕捉所有这些颜色。因此，几乎不可能在数字形式中容纳每一种可能的颜色。因此，科学家们为我们提供了一种表示不同颜色的方法：*RGB*模式。这是三种主要颜色组件的组合：红色、绿色和蓝色。通过组合这些组件，我们可以创建几乎所有可能的颜色。每个组件的值范围从0到255；对每个组件的代码的更改会导致新的颜色。

OpenGL中使用的颜色属性与现实世界的颜色反射属性非常相似。我们观察到的物体的颜色实际上并不是它的颜色；相反，它是物体反射的颜色。物体可能具有某种波长的属性，物体可以吸收某种颜色并反射出另一种颜色。例如，树木吸收阳光除了绿色。我们感知并假设它是绿色的，但实际上物体没有颜色。这种光反射的概念在OpenGL中得到了很好的应用——通常我们定义一个可能具有明确颜色代码的光源。此外，我们还将定义物体的颜色代码，然后将其与光源相乘。结果的颜色代码或光是从物体反射出来的结果，被认为是物体的颜色。

在OpenGL中，颜色以包含四个组件的元组形式给出，其中三个是红色、绿色和蓝色。第四个组件代表alpha信息，表示物体的透明级别。在OpenGL中，与RGB组件的值为0到255不同，我们提供的值范围是0到1。例如，黄色是红色和绿色的组合，因此它的alpha信息是(1, 1, 0)。请参考[https://community.khronos.org/t/color-tables/22518](https://community.khronos.org/t/color-tables/22518)了解更多关于OpenGL颜色代码的信息。

以下函数/特性在OpenGL的颜色属性中可用：

+   `glClearColor()`: 这个函数设置一个清晰的颜色，这意味着它填充在尚未绘制的区域上的颜色。颜色代码的值可以作为一个元组给出，范围从0到1。例如，`glClearColor(1.0, 1.0, 1.0, 0.0)`表示用白色填充。

+   `glShadeModel()`: 这个函数启用了OpenGL的光照特性。通常传递给`glShadeModel`的参数是`GL_FLAT`，用于给形状的面或边缘上色，比如立方体和金字塔。如果你想给曲面对象上色而不是给面体对象上色，你可以使用`GL_SMOOTH`。

+   `glEnable()`: 这实际上不是与颜色属性相关的方法，但是用于启用它们。例如，`glEnable(GL_COLOR_MATERIAL)`将启用*材料*，这允许我们与表面和光源进行交互。此外，通过调整设置，材料的属性主要用于使任何物体更轻和更锐利。

现在我们熟悉了颜色属性的概念和创建颜色属性的方法，让我们编写一个简单的程序，使用PyOpenGL的颜色属性来绘制一个立方体。

# 头脑风暴网格

在我们开始编码之前，头脑风暴一下并获取必要的信息总是一个好习惯，这样我们才能创建一个程序。因为我们将创建一个渲染立方体的程序——一个有八个顶点、12条边和六个面的表面——我们需要明确定义这样的信息。我们可以将这些属性定义为嵌套元组——单个元组内的元组。

以一个顶点作为参考，我们可以同时获取其他顶点的位置。假设一个立方体有一个顶点在（`1`，`-1`，`-1`）。现在，假设立方体的所有边都是1个单位长，我们可以得到顶点的坐标。以下代码显示了立方体的顶点列表：

```py
cube_Vertices = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1),
    )
```

同样，有12条边（边是从一个顶点到另一个顶点画出的线）。由于有八个顶点（0到7），让我们编写一些代码，使用八个顶点定义12条边。在以下代码中，作为元组传递的标识符表示从一个顶点到另一个顶点画出的边或面。例如，元组（`0`，`1`）表示从顶点0到顶点1画出的边：

```py
cube_Edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7),
    )
```

最后，必须提供的最后一部分信息是关于表面的。一个立方体有六个面，每个面包含四个顶点和四条边。我们可以这样提供这些信息：

```py
cube_Surfaces = (
 (0,1,2,3),
 (3,2,7,6),
 (6,7,5,4),
 (4,5,1,0),
 (1,5,7,2),
 (4,0,3,6) 
 )
```

注意提供顶点、边和表面的顺序很重要。例如，在`cube_Surfaces`数据结构中，如果你交换了元组的第二个项目和第一个项目，立方体的形状将会恶化。这是因为每个信息都与顶点信息相关联，也就是说，表面（`0`，`1`，`2`，`3`）包含了第一个、第二个、第三个和第四个顶点。

现在我们已经完成了头脑风暴，并收集了关于我们要绘制的形状的一些有用信息，是时候开始使用PyOpenGL及其库来渲染立方体了，这个库通常被称为*GLU库*。

# 理解GLU库

现在我们已经收集了关于我们形状的边、面和顶点的信息，我们可以开始编写模型了。我们已经学习了如何使用`glBegin()`和`glVertex3fv()`等方法使用OpenGL绘制形状。让我们使用它们，并创建一个可以绘制立方体结构的函数：

1.  首先导入OpenGL和GLU库。在导入库之后，将我们在头脑风暴中定义的有关顶点、边和表面的信息添加到同一个文件中：

```py
      from OpenGL.GL import *
      from OpenGL.GLU import *
```

1.  接下来，定义函数并获取表面和顶点。这个过程非常简单；我们将从绘制立方体的表面开始。我们应该使用`GL_QUADS`属性来绘制四面体表面（困惑吗？请参考本章的*使用OpenGL制作对象*部分获取更多信息）：

```py
      def renderCube():
          glBegin(GL_QUADS)
          for eachSurface in cube_Surfaces:
              for eachVertex in eachSurface:
                  glColor3fv((1, 1, 0)) #yellow color code
                  glVertex3fv(cube_Surfaces[eachVertex])
          glEnd()
```

1.  最后，在`renderCube()`方法中，编写一些可以绘制线段的代码。使用`GL_LINES`参数来绘制线段：

```py
     glBegin(GL_LINES)
       for eachEdge in cube_Edges:
           for eachVertex in eachEdge:
               glVertex3fv(cube_Vertices[eachVertex])
       glEnd()
```

这个三行的过程足以创建复杂的几何形状。现在，你可以对这些立方体执行多个操作。例如，你可以使用鼠标触控板旋转物体。正如我们所知，处理这样的用户事件需要一个`pygame`模块。因此，让我们定义一个函数，来处理事件，并使用PyOpenGL的一些特性。从`import pygame`语句开始你的代码，并添加以下代码：

```py
def ActionHandler():
    pygame.init()
    screen = (800, 500)
    pygame.display.set_mode(screen, DOUBLEBUF|OPENGL) #OPENGL is essential

    #1: ADD A CLIPPING TRANSFORMATION
    gluPerspective(85.0, (screen[0]/screen[1]), 0.1, 50) 

    # 80.0 -> field view of camera 
    #screen[0]/screen[1] -> aspect ration (width/height)
    #0.1 -> near clipping plane
    #50 -> far clipping plane
    glRotatef(18, 2, 0, 0) #start point
```

前面的代码片段非常容易理解，因为我们从本章的开始就一直在做这个。在这里，我们使用了`pygame`模块，它使用OpenGL场景或接口设置游戏屏幕。我们添加了一个变换矩阵，它使用`gluPerspective()`函数执行裁剪。最后，我们在实际旋转之前添加了立方体的初始位置（在开始时可能在哪里）。

现在我们已经介绍了OpenGL的基本知识，让我们使用pygame的事件处理方法来操纵立方体的结构，就像这样：

```py
while True:

        for anyEvent in pygame.event.get():
            if anyEvent.type == pygame.QUIT:
                pygame.quit()
                quit()

            if anyEvent.type == pygame.MOUSEBUTTONDOWN:
                print(anyEvent)
                print(anyEvent.button) #printing mouse event

                #mouse button 4 and 5 are at the left side of the mouse
                #mouse button 4 is used as forward and backward navigation
                if anyEvent.button == 4: 
 glTranslatef(0.0,0.0,1.0) #produces translation 
                      of (x, y, z)
 elif anyEvent.button == 5:
 glTranslatef(0.0,0.0,-1.0)
```

在处理基于鼠标按钮导航的事件之后，让我们使用PyOpenGL提供的一些方法来渲染立方体。我们将使用`glRotatef()`等方法来执行矩阵变换。在处理事件的地方之后，写入以下代码：

```py

        glRotatef(1, 3, 1, 1) 
#The glRotatef is used to perform matrix transformation which performs a rotation 
#of counterclockwise with an angle of degree about origin through the point #provided as (x, y, z). 
        #-----------------------------------------------------------------
        #indicates the buffer that needs to be cleared
        #GL_COLOR_BUFFER_BIT: enabled for color drawing
        #GL_DEPTH_BUFFER_BIT: depth buffer which needs to be cleared

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        #render cube
        renderCube()
        pygame.display.flip()
        pygame.time.wait(12)

#call main function only externally
ActionHandler()
```

上述代码的突出部分表示调整大小的变换，最终导致了使用ZOOM-UP和ZOOM-DOWN功能。现在，您可以运行程序，观察立方体在pygame屏幕中心以黄色渲染。尝试使用外部鼠标和导航按钮（按钮4和5）进行放大和缩小。您还可以观察项目中如何使用裁剪：每当我们使一个立方体变得如此之大以至于超出裁剪平面时，立方体的一些部分将从视口中移除。

通过这种方式，我们可以结合两个强大的Python游戏模块，即*pygame*和*PyOpenGL*，制作3D场景和界面。我们只是简单地介绍了创建一些形状和如何变换它们的方法。现在，轮到您去发现更多关于PyOpenGL的知识，并尝试制作一个更加用户友好和吸引人的游戏，提供丰富的纹理和内容。

# 总结

在本章中，我们涵盖了许多有趣的主题，主要涉及表面和几何形状。虽然在本章中我们使用了术语*矩阵*，但我们并没有使用数学方法进行矩阵计算，因为Python内置了执行此类操作的一切。尽管如此，我们应该记住这句古老的格言，*游戏程序员不需要拥有数学博士学位*，因为只要我们想制作游戏，基本的数学水平就足够了。在这里，我们只学习了平移、缩放和旋转，如果我们想制作一个3D场景，这已经足够了。我们没有陷入使用数学方法进行平移或缩放的概念中——相反，我们学习了使用编程方法。

我们首先学习了如何使用pygame的`setting`方法设置OpenGL显示屏。由于OpenGL是一个广阔而深奥的研究领域，无法在单一章节中涵盖所有内容。因此，我们只涵盖了如何加载/存储三维模型以及如何通过应用裁剪、旋转和调整大小变换将它们应用到OpenGL渲染表面上。我们还研究了颜色属性，并将它们与PyOpenGL和pygame一起使用。本章的主要目标是让您更容易理解如何使用OpenGL创建3D形状，同时提供关键的几何信息，如顶点、边和表面。现在您将能够使用OpenGL创建3D形状、图形和可视化。您现在也知道如何将OpenGL的颜色属性与其他着色模式区分开来。

在下一章中，我们将学习另一个重要的模块，名为*Pymunk*。这是一个非常强大的物理库，为游戏角色增加了物理能力。我们将学习在需要讨论真实世界环境时使用的不同术语，如速度和加速度，这些术语用于处理碰撞和游戏角色的移动。在学习这些概念的同时，我们还将制作一个愤怒的小鸟游戏，并将其部署到各种平台上。
