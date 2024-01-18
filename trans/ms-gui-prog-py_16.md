# 使用QtOpenGL创建3D图形

从游戏到数据可视化到工程模拟，3D图形和动画是许多重要软件应用的核心。几十年来，事实上的**应用程序编程接口**（**API**）标准一直是OpenGL。

用于跨平台3D图形的API一直是OpenGL。尽管存在许多Python和C的API实现，Qt提供了一个直接集成到其小部件中的API，使我们能够在GUI中嵌入交互式的OpenGL图形和动画。

在本章中，我们将在以下主题中探讨这些功能：

+   OpenGL的基础知识

+   使用`QOpenGLWidget`嵌入OpenGL绘图

+   动画和控制OpenGL绘图

# 技术要求

对于本章，你需要一个基本的Python 3和PyQt5设置，就像我们在整本书中一直在使用的那样，并且你可能想从[https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter13](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter13)下载示例代码。你还需要确保你的图形硬件和驱动程序支持OpenGL 2.0或更高版本，尽管如果你使用的是过去十年内制造的传统台式机或笔记本电脑，这几乎肯定是真的。

查看以下视频，看看代码是如何运行的：[http://bit.ly/2M5xApP](http://bit.ly/2M5xApP)

# OpenGL的基础知识

OpenGL不仅仅是一个库；它是一个与图形硬件交互的API的**规范**。这个规范的实现是由你的图形硬件、该硬件的驱动程序和你选择使用的OpenGL软件库共享的。因此，你的基于OpenGL的代码的确切行为可能会因其中任何一个因素而略有不同，就像同样的HTML代码在不同的网络浏览器中可能会稍有不同地呈现一样。

OpenGL也是一个**有版本的**规范，这意味着OpenGL的可用功能和推荐用法会随着你所针对的规范版本的不同而改变。随着新功能的引入和旧功能的废弃，最佳实践和建议也在不断发展，因此为OpenGL 2.x系统编写的代码可能看起来完全不像为OpenGL 4.x编写的代码。

OpenGL规范由Khronos Group管理，这是一个维护多个与图形相关的标准的行业联盟。撰写本文时的最新规范是4.6，发布于2019年2月，可以在[https://www.khronos.org/registry/OpenGL/index_gl.php](https://www.khronos.org/registry/OpenGL/index_gl.php)找到。然而，并不总是跟随最新规范是一个好主意。计算机运行给定版本的OpenGL代码的能力受到硬件、驱动程序和平台考虑的限制，因此，如果你希望你的代码能够被尽可能广泛的用户运行，最好是针对一个更旧和更成熟的版本。许多常见的嵌入式图形芯片只支持OpenGL 3.x或更低版本，一些低端设备，如树莓派（我们将在[第15章](77583d1b-8a70-4118-8210-b0a5f09c9603.xhtml)，*树莓派上的PyQt*中看到）只支持2.x。

在本章中，我们将限制我们的代码在OpenGL 2.1，因为它得到了PyQt的良好支持，大多数现代计算机应该能够运行它。然而，由于我们将坚持基础知识，我们所学到的一切同样适用于4.x版本。

# 渲染管线和绘图基础知识

将代码和数据转化为屏幕上的像素需要一个多阶段的过程；在OpenGL中，这个过程被称为**渲染管线。** 这个管线中的一些阶段是可编程的，而其他的是固定功能的，意味着它们的行为是由OpenGL实现预先确定的，不能被改变。

让我们从头到尾走一遍这个管道的主要阶段：

1.  **顶点规范**：在第一个阶段，绘图的**顶点**由您的应用程序确定。**顶点**本质上是3D空间中的一个点，可以用来绘制形状。顶点还可以包含关于点的元数据，比如它的颜色。

1.  **顶点处理**：这个可用户定义的阶段以各种方式处理每个顶点，计算每个顶点的最终位置；例如，在这一步中，您可能会旋转或移动顶点规范中定义的基本形状。

1.  **顶点后处理**：这个固定功能阶段对顶点进行一些额外的处理，比如裁剪超出视图空间的部分。

1.  **基元组装**：在这个阶段，顶点被组合成基元。一个基元是一个2D形状，比如三角形或矩形，从中可以构建更复杂的3D形状。

1.  **光栅化**：这个阶段将基本图元转换为一系列单独的像素点，称为片段，通过在顶点之间进行插值。

1.  **片段着色**：这个用户定义阶段的主要工作是确定每个片段的深度和颜色值。

1.  **逐样本操作**：这个最后阶段对每个片段执行一系列测试，以确定其最终的可见性和颜色。

作为使用OpenGL的程序员，我们主要关注这个操作的三个阶段 - 顶点规范、顶点处理和片段着色。对于顶点规范，我们将简单地在Python代码中定义一些点来描述OpenGL绘制的形状；对于其他两个阶段，我们需要学习如何创建OpenGL程序和着色器。

# 程序和着色器

尽管名字上是着色器，但它与阴影或着色无关；它只是在GPU上运行的代码单元的名称。在前一节中，我们谈到了渲染管线的一些阶段是可用户定义的；事实上，其中一些*必须*被定义，因为大多数OpenGL实现不为某些阶段提供默认行为。为了定义这些阶段，我们需要编写一个着色器。

至少，我们需要定义两个着色器：

+   **顶点着色器**：这个着色器是顶点处理阶段的第一步。它的主要工作是确定每个顶点的空间坐标。

+   **片段着色器**：这是管线倒数第二个阶段，它唯一的必要工作是确定单个片段的颜色。

当我们有一组着色器组成完整的渲染管线时，这被称为一个程序。

着色器不能用Python编写。它们必须用一种叫做**GL着色语言**（**GLSL**）的语言编写，这是OpenGL规范的一部分的类似C的语言。没有GLSL的知识，就不可能创建严肃的OpenGL绘图，但幸运的是，写一组足够简单的着色器对于基本示例来说是相当简单的。

# 一个简单的顶点着色器

我们将组成一个简单的GLSL顶点着色器，我们可以用于我们的演示；创建一个名为`vertex_shader.glsl`的文件，并复制以下代码：

```py
#version 120
```

我们从一个注释开始，指明我们正在使用的GLSL版本。这很重要，因为每个OpenGL版本只兼容特定版本的GLSL，GLSL编译器将使用这个注释来检查我们是否不匹配这些版本。

可以在[https://www.khronos.org/opengl/wiki/Core_Language_(GLSL)](https://www.khronos.org/opengl/wiki/Core_Language_(GLSL))找到GLSL和OpenGL版本之间的兼容性图表。

接下来，我们需要进行一些**变量声明**：

```py
attribute highp vec4 vertex;
uniform highp mat4 matrix;
attribute lowp vec4 color_attr;
varying lowp vec4 color;
```

在类似C的语言中，变量声明用于创建变量，定义关于它的各种属性，并在内存中分配空间。我们的每个声明有四个标记；让我们按顺序来看一下这些：

+   第一个标记是`attribute`，`uniform`或`varying`中的一个。这表明变量将分别用于每个顶点（`attribute`），每个基本图元（`uniform`）或每个片段（`varying`）。因此，我们的第一个变量将对每个顶点都不同，但我们的第二个变量将对同一基本图元中的每个顶点都相同。

+   第二个标记指示变量包含的基本数据类型。在这种情况下，它可以是`highp`（高精度数字），`mediump`（中等精度数字）或`lowp`（低精度数字）。我们可以在这里使用`float`或`double`，但这些别名有助于使我们的代码跨平台。

+   第三个术语定义了这些变量中的每一个是指向**向量**还是矩阵。你可以将向量看作是Python的`list`对象，将矩阵看作是一个每个项目都是相同长度的`list`对象的`list`对象。末尾的数字表示大小，所以`vec4`是一个包含四个值的列表，`mat4`是一个4x4值的矩阵。

+   最后一个标记是变量名。这些名称将在整个程序中使用，因此我们可以在管道中更深的着色器中使用它们来访问来自先前着色器的数据。

这些变量可以用来将数据插入程序或将数据传递给程序中的其他着色器。我们将在本章后面看到如何做到这一点，但现在要明白，在我们的着色器中，`vertex`，`matrix`和`color_attr`代表着将从我们的PyQt应用程序接收到的数据。

在变量声明之后，我们将创建一个名为`main()`的函数：

```py
void main(void)
{
  gl_Position = matrix * vertex;
  color = color_attr;
}
```

`vertex`着色器的主要目的是使用`vertex`的坐标设置一个名为`gl_Position`的变量。在这种情况下，我们将其设置为传入着色器的`vertex`值乘以`matrix`值。正如你将在后面看到的，这种安排将允许我们在空间中操作我们的绘图。

在创建3D图形时，矩阵和向量是关键的数学概念。虽然在本章中我们将大部分时间都从这些数学细节中抽象出来，但如果你想深入学习OpenGL编程，了解这些概念是个好主意。

我们着色器中的最后一行代码可能看起来有点无意义，但它允许我们在顶点规范阶段为每个顶点指定一个颜色，并将该颜色传递给管道中的其他着色器。着色器中的变量要么是输入变量，要么是输出变量，这意味着它们期望从管道的前一个阶段接收数据，或者将数据传递给下一个阶段。在顶点着色器中，使用`attribute`或`uniform`限定符声明变量会将变量隐式标记为输入变量，而使用`varying`限定符声明变量会将其隐式标记为输出变量。因此，我们将`attribute`类型的`color_attr`变量的值复制到`varying`类型的`color`变量中，以便将该值传递给管道中更深的着色器；具体来说，我们想将其传递给`fragment`着色器。

# 一个简单的片段着色器

我们需要创建的第二个着色器是`fragment`着色器。请记住，这个着色器的主要工作是确定每个基本图元上每个点（或*片段*）的颜色。

创建一个名为`fragment_shader.glsl`的新文件，并添加以下代码：

```py
#version 120

varying lowp vec4 color;

void main(void)
{
  gl_FragColor = color;
}
```

就像我们的`vertex`着色器一样，我们从一个指定我们要针对的GLSL版本的注释开始。然后，我们将声明一个名为`color`的变量。

因为这是`fragment`着色器，将变量指定为`varying`会使其成为输入变量。使用`color`这个名称，它是我们着色器的输出变量，意味着我们将从该着色器接收它分配的颜色值。

然后在`main()`中，我们将该颜色分配给内置的`gl_FragColor`变量。这个着色器的有效作用是告诉OpenGL使用`vertex`着色器传入的颜色值来确定单个片段的颜色。

这是我们可以得到的最简单的`fragment`着色器。更复杂的`fragment`着色器，例如在游戏或模拟中找到的着色器，可能实现纹理、光照效果或其他颜色操作；但对于我们的目的，这个着色器应该足够了。

现在我们有了所需的着色器，我们可以创建一个PyQt应用程序来使用它们。

# 使用QOpenGLWidget嵌入OpenGL绘图

为了了解OpenGL如何与PyQt一起工作，我们将使用我们的着色器制作一个简单的OpenGL图像，通过PyQt界面我们将能够控制它。从[第4章](61ff4931-02af-474a-996c-5da827e0684f.xhtml)中创建一个Qt应用程序模板的副本，*使用QMainWindow构建应用程序*，并将其命名为`wedge_animation.py`。将其放在与您的`shader`文件相同的目录中。

然后，首先在`MainWindow.__init__()`中添加此代码：

```py
        self.resize(800, 600)
        main = qtw.QWidget()
        self.setCentralWidget(main)
        main.setLayout(qtw.QVBoxLayout())
        oglw = GlWidget()
        main.layout().addWidget(oglw)
```

此代码创建我们的中央小部件并向其添加一个`GlWidget`对象。`GlWidget`类是我们将创建的用于显示我们的OpenGL绘图的类。要创建它，我们需要对可以显示OpenGL内容的小部件进行子类化。

# OpenGLWidget的第一步

有两个Qt类可用于显示OpenGL内容：`QtWidgets.QOpenGLWidget`和`QtGui.QOpenGLWindow`。在实践中，它们的行为几乎完全相同，但`OpenGLWindow`提供了稍微更好的性能，如果您不想使用任何其他Qt小部件（即，如果您的应用程序只是全屏OpenGL内容），可能是更好的选择。在我们的情况下，我们将把我们的OpenGL绘图与其他小部件组合在一起，因此我们将使用`QOpenGLWidget`作为我们的类的基础：

```py
class GlWidget(qtw.QOpenGLWidget):
    """A widget to display our OpenGL drawing"""
```

要在我们的小部件上创建OpenGL内容，我们需要重写两个`QOpenGLWidget`方法：

+   `initializeGL()`，它只运行一次来设置我们的OpenGL绘图

+   `paintGL()`在我们的小部件需要绘制自己时（例如，响应`update()`调用）调用

我们将从`initializeGL()`开始：

```py
    def initializeGL(self):
        super().initializeGL()
        gl_context = self.context()
        version = qtg.QOpenGLVersionProfile()
        version.setVersion(2, 1)
        self.gl = gl_context.versionFunctions(version)
```

我们需要做的第一件事是访问我们的OpenGL API。API由一组函数、变量和常量组成；在诸如PyQt之类的面向对象平台中，我们将创建一个包含这些函数作为方法以及变量和常量作为属性的特殊OpenGL函数对象。

为此，我们首先从`QOpenGLWidget`方法中检索一个OpenGL**上下文**。上下文表示我们当前绘制的OpenGL表面的接口。从上下文中，我们可以检索包含我们的API的对象。

因为我们需要访问特定版本的API（2.1），我们首先需要创建一个`QOpenGLVersionProfile`对象，并将其`version`属性设置为`(2, 1)`。这可以传递给上下文的`versionFunctions()`方法，该方法将返回一个`QOpenGLFunctions_2_1`对象。这是包含我们的OpenGL 2.1 API的对象。

Qt还为其他版本的OpenGL定义了OpenGL函数对象，但请注意，根据您的平台、硬件以及您获取Qt的方式，可能会或可能不会支持特定版本。

我们将`functions`对象保存为`self.gl`；我们所有的API调用都将在这个对象上进行。

既然我们可以访问API，让我们开始配置OpenGL：

```py
        self.gl.glEnable(self.gl.GL_DEPTH_TEST)
        self.gl.glDepthFunc(self.gl.GL_LESS)
        self.gl.glEnable(self.gl.GL_CULL_FACE)
```

与Qt类似，OpenGL使用定义的常量来表示各种设置和状态。配置OpenGL主要是将这些常量传递给各种API函数，以切换各种设置。

在这种情况下，我们执行三个设置：

+   将`GL_DEPTH_TEST`传递给`glEnable()`会激活**深度测试**，这意味着OpenGL将尝试弄清楚其绘制的点中哪些在前景中，哪些在背景中。

+   `glDepthFunc()`设置将确定是否绘制深度测试像素的函数。在这种情况下，`GL_LESS`常量表示将绘制深度最低的像素（即最接近我们的像素）。通常，这是您想要的设置，也是默认设置。

+   将`GL_CULL_FACE`传递给`glEnable()`会激活**面剔除**。这意味着OpenGL不会绘制观看者实际看不到的物体的侧面。这也是有意义的，因为它节省了本来会被浪费的资源。

这三个优化应该有助于减少我们的动画使用的资源；在大多数情况下，您会想要使用它们。还有许多其他可以启用和配置的选项；有关完整列表，请参见[https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glEnable.xml](https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glEnable.xml)。请注意，有些选项只适用于使用OpenGL的旧固定功能方法。

如果你看到使用`glBegin()`和`glEnd()`的OpenGL代码，那么它使用的是非常古老的OpenGL 1.x固定功能绘图API。这种方法更容易，但更有限，所以不应该用于现代OpenGL编程。

# 创建一个程序

在实现OpenGL绘图的下一步是创建我们的程序。您可能还记得，OpenGL程序是由一组着色器组成的，形成一个完整的管道。

在Qt中，创建程序的过程如下：

1.  创建一个`QOpenGLShaderProgram`对象

1.  将您的着色器代码添加到程序中

1.  将代码链接成完整的程序

以下代码将实现这一点：

```py
        self.program = qtg.QOpenGLShaderProgram()
        self.program.addShaderFromSourceFile(
            qtg.QOpenGLShader.Vertex, 'vertex_shader.glsl')
        self.program.addShaderFromSourceFile(
            qtg.QOpenGLShader.Fragment, 'fragment_shader.glsl')
        self.program.link()
```

着色器可以从文件中添加，就像我们在这里使用`addShaderFromSourceFile()`做的那样，也可以从字符串中添加，使用`addShaderFromSourceCode()`。我们在这里使用相对文件路径，但最好的方法是使用Qt资源文件（参见[第6章](c3eb2567-0e73-4c37-9a9e-a0e2311e106c.xhtml)中的*使用Qt资源文件*部分，*Qt应用程序的样式*）。当文件被添加时，Qt会编译着色器代码，并将任何编译错误输出到终端。

在生产代码中，您会想要检查`addShaderFromSourceFile()`的布尔输出，以查看您的着色器是否成功编译，然后再继续。

请注意，`addShaderFromSourceFile()`的第一个参数指定了我们要添加的着色器的类型。这很重要，因为顶点着色器和片段着色器有非常不同的要求和功能。

一旦所有着色器都加载完毕，我们调用`link()`将所有编译的代码链接成一个准备执行的程序。

# 访问我们的变量

我们的着色器程序包含了一些我们需要能够访问并放入值的变量，因此我们需要检索这些变量的句柄。`QOpenGLProgram`对象有两种方法，`attributeLocation()`和`uniformLocation()`，分别用于检索属性和统一变量的句柄（对于`varying`类型没有这样的函数）。

让我们为我们的`vertex`着色器变量获取一些句柄：

```py
        self.vertex_location = self.program.attributeLocation('vertex')
        self.matrix_location = self.program.uniformLocation('matrix')
        self.color_location = self.program.attributeLocation('color_attr')
```

这些方法返回的值实际上只是整数；在内部，OpenGL只是使用顺序整数来跟踪和引用对象。然而，这对我们来说并不重要。我们可以将其视为对象句柄，并将它们传递到OpenGL调用中，以访问这些变量，很快您就会看到。

# 配置投影矩阵

在OpenGL中，**投影矩阵**定义了我们的3D模型如何投影到2D屏幕上。这由一个4x4的数字矩阵表示，可以用来计算顶点位置。在我们进行任何绘图之前，我们需要定义这个矩阵。

在Qt中，我们可以使用`QMatrix4x4`对象来表示它：

```py
        self.view_matrix = qtg.QMatrix4x4()
```

`QMatrix4x4`对象非常简单，它是一个按四行四列排列的数字表。然而，它有几种方法，允许我们以这样的方式操纵这些数字，使它们代表3D变换，比如我们的投影。

OpenGL可以使用两种投影方式——**正交**，意味着所有深度的点都被渲染为相同的，或者**透视**，意味着视野随着我们远离观察者而扩展。对于逼真的3D绘图，您将希望使用透视投影。这种投影由**视锥体**表示。

视锥体是两个平行平面之间的一个常规几何固体的一部分，它是用来描述视野的有用形状。要理解这一点，把你的手放在头两侧。现在，把它们向前移动，保持它们刚好在你的视野之外。注意，为了做到这一点，你必须向外移动（向左和向右）。再试一次，把你的手放在头上和头下。再一次，你必须垂直向外移动，以使它们远离你的视野。

您刚刚用手做的形状就像一个金字塔，从您的眼睛延伸出来，其顶点被切成与底部平行的形状，换句话说，是一个视锥体。

要创建表示透视视锥体的矩阵，我们可以使用`matrix`对象的`perspective()`方法：

```py
        self.view_matrix.perspective(
            45,  # Angle
            self.width() / self.height(),  # Aspect Ratio
            0.1,  # Near clipping plane
            100.0  # Far clipping plane
        )
```

`perspective()`方法需要四个参数：

+   从近平面到远平面扩展的角度，以度为单位

+   近平面和远平面的纵横比（相同）

+   近平面向屏幕的深度

+   远平面向屏幕的深度

不用深入复杂的数学，这个矩阵有效地表示了我们相对于绘图的视野。当我们开始绘图时，我们将看到，我们移动对象所需做的就是操纵矩阵。

例如，我们可能应该从我们将要绘制的地方稍微后退一点，这样它就不会发生在视野的最前面。这种移动可以通过`translate()`方法来实现：

```py
        self.view_matrix.translate(0, 0, -5)
```

`translate`需要三个参数——x量、y量和z量。在这里，我们指定了一个z平移量为`-5`，这将使对象深入屏幕。

现在这一切可能看起来有点混乱，但是，一旦我们开始绘制形状，事情就会变得更清晰。

# 绘制我们的第一个形状

现在我们的OpenGL环境已经初始化，我们可以继续进行`paintGL()`方法。这个方法将包含绘制我们的3D对象的所有代码，并且在小部件需要更新时将被调用。

绘画时，我们要做的第一件事是清空画布：

```py
    def paintGL(self):
        self.gl.glClearColor(0.1, 0, 0.2, 1)
        self.gl.glClear(
            self.gl.GL_COLOR_BUFFER_BIT | self.gl.GL_DEPTH_BUFFER_BIT)
        self.program.bind()
```

`glClearColor()`用于用指定的颜色填充绘图的背景。在OpenGL中，颜色使用三个或四个值来指定。在三个值的情况下，它们代表红色、绿色和蓝色。第四个值，当使用时，代表颜色的**alpha**或不透明度。与Qt不同，其中RGB值是从`0`到`255`的整数，OpenGL颜色值是从`0`到`1`的浮点数。我们前面的值描述了深紫蓝色；可以随意尝试其他值。

您应该在每次重绘时使用`glClearColor`重新绘制背景；如果不这样做，之前的绘画操作仍然可见。如果您进行动画或调整绘图大小，这将是一个问题。

`glClear()`函数用于清除GPU上的各种内存缓冲区，我们希望在重绘之间重置它们。在这种情况下，我们指定了一些常量，导致OpenGL清除颜色缓冲区和深度缓冲区。这有助于最大化性能。

最后，我们`bind()`程序对象。由于OpenGL应用程序可以有多个程序，我们调用`bind()`告诉OpenGL我们即将发出的命令适用于这个特定的程序。

现在我们可以绘制我们的形状了。

OpenGL中的形状是用顶点描述的。您可能还记得，顶点本质上是3D空间中的一个点，由*X*、*Y*和*Z*坐标描述，并定义了一个基本图元的一个角或端点。

让我们创建一个顶点列表来描述一个楔形的前面是三角形：

```py
        front_vertices = [
            qtg.QVector3D(0.0, 1.0, 0.0),  # Peak
            qtg.QVector3D(-1.0, 0.0, 0.0),  # Bottom left
            qtg.QVector3D(1.0, 0.0, 0.0)  # Bottom right
            ]
```

我们的顶点数据不必分组成任何类型的不同对象，但是为了方便和可读性，我们使用`QVector3D`对象来保存三角形中每个顶点的坐标。

这里使用的数字代表网格上的点，其中`(0, 0, 0)`是我们OpenGL视口的中心在最前面的点。x轴从屏幕左侧的`-1`到右侧的`1`，y轴从屏幕顶部的`1`到底部的`-1`。z轴有点不同；如果想象视野（我们之前描述的视锥体）作为一个形状从显示器背面扩展出来，负z值会推进到视野的更深处。正z值会移出屏幕朝着（最终在后面）观察者。因此，通常我们将使用负值或零值的z来保持在可见范围内。

默认情况下，OpenGL将以黑色绘制，但是有一些颜色会更有趣。因此，我们将定义一个包含一些颜色的`tuple`对象：

```py
        face_colors = (
            qtg.QColor('red'),
            qtg.QColor('orange'),
            qtg.QColor('yellow'),
        )
```

我们在这里定义了三种颜色，每个三角形顶点一个。这些是`QColor`对象，但是请记住OpenGL需要颜色作为值在`0`和`1`之间的向量。

为了解决这个问题，我们将创建一个小方法将`QColor`转换为OpenGL友好的向量：

```py
    def qcolor_to_glvec(self, qcolor):
        return qtg.QVector3D(
            qcolor.red() / 255,
            qcolor.green() / 255,
            qcolor.blue() / 255
        )
```

这段代码相当不言自明，它将创建另一个带有转换后的RGB值的`QVector3D`对象。

回到`paintGL()`，我们可以使用列表推导将我们的颜色转换为可用的东西：

```py
        gl_colors = [
            self.qcolor_to_glvec(color)
            for color in face_colors
        ]
```

此时，我们已经定义了一些顶点和颜色数据，但是我们还没有发送任何数据到OpenGL；这些只是我们Python脚本中的数据值。要将这些传递给OpenGL，我们需要在`initializeGL()`中获取的那些变量句柄。

我们将传递给我们的着色器的第一个变量是`matrix`变量。我们将使用我们在`initializeGL()`中定义的`view_matrix`对象：

```py
        self.program.setUniformValue(
            self.matrix_location, self.view_matrix)
```

`setUniformValue()`可以用来设置`uniform`变量的值；我们可以简单地传递`uniformLocation()`获取的`GLSL`变量的句柄和我们创建的`matrix`对象来定义我们的投影和视野。

您还可以使用`setAttributeValue()`来设置`attribute`变量的值。例如，如果我们希望所有顶点都是红色，我们可以添加这个：

```py
        self.program.setAttributeValue(
            self.color_location, gl_colors[0])
```

但我们不要这样做；如果每个顶点都有自己的颜色会看起来更好。

为此，我们需要创建一些**属性数组。**属性数组是将传递到属性类型变量中的数据数组。请记住，在GLSL中标记为属性的变量将为每个顶点应用一个不同的值。因此，实际上我们告诉OpenGL，*这里有一些数据数组，其中每个项目都应用于一个顶点*。

代码看起来像这样：

```py
        self.program.enableAttributeArray(self.vertex_location)
        self.program.setAttributeArray(
            self.vertex_location, front_vertices)
        self.program.enableAttributeArray(self.color_location)
        self.program.setAttributeArray(self.color_location, gl_colors)
```

第一步是通过使用要设置数组的变量的句柄调用`enableAttributeArray()`来启用`GLSL`变量上的数组。然后，我们使用`setAttributeArray()`传递数据。这实际上意味着我们的`vertex`着色器将在`front_vertices`数组中的每个项目上运行。每次该着色器运行时，它还将从`gl_colors`列表中获取下一个项目，并将其应用于`color_attr`变量。

如果您像这样使用多个属性数组，您需要确保数组中有足够的项目来覆盖所有顶点。如果我们只定义了两种颜色，第三个顶点将为`color_attr`提取垃圾数据，导致未定义的输出。

现在我们已经排队了我们第一个基元的所有数据，让我们使用以下代码进行绘制：

```py
        self.gl.glDrawArrays(self.gl.GL_TRIANGLES, 0, 3)
```

`glDrawArrays()`将发送我们定义的所有数组到管道中。`GL_TRIANGLES`参数告诉OpenGL它将绘制三角形基元，接下来的两个参数告诉它从数组项`0`开始绘制三个项。

如果此时运行程序，您应该会看到我们绘制了一个红色和黄色的三角形。不错！现在，让我们让它成为3D。

# 创建一个3D对象

为了制作一个3D对象，我们需要绘制楔形对象的背面和侧面。我们将首先通过列表推导来计算楔形的背面坐标：

```py
        back_vertices = [
            qtg.QVector3D(x.toVector2D(), -0.5)
            for x in front_vertices]
```

为了创建背面，我们只需要复制每个正面坐标并将z轴向后移一点。因此，我们使用`QVector3D`对象的`toVector2D()`方法来产生一个只有x和y轴的新向量，然后将其传递给一个新的`QVector3D`对象的构造函数，同时指定新的z坐标作为第二个参数。

现在，我们将把这组顶点传递给OpenGL并进行绘制如下：

```py
        self.program.setAttributeArray(
            self.vertex_location, reversed(back_vertices))
        self.gl.glDrawArrays(self.gl.GL_TRIANGLES, 0, 3)
```

通过将这些写入`vertex_location`，我们已经覆盖了已经绘制的正面顶点，并用背面顶点替换了它们。然后，我们对`glDrawArrays()`进行相同的调用，新的顶点集将被绘制，以及相应的颜色。

您将注意到我们在绘制之前会颠倒顶点的顺序。当OpenGL显示一个基元时，它只显示该基元的一面，因为假定该基元是某个3D对象的一部分，其内部不需要被绘制。OpenGL根据基元的点是顺时针还是逆时针绘制来确定应该绘制哪一面的基元。默认情况下，绘制逆时针的基元的近面，因此我们将颠倒背面顶点的顺序，以便绘制顺时针并显示其远面（这将是楔形的外部）。

让我们通过绘制其侧面来完成我们的形状。与前面和后面不同，我们的侧面是矩形，因此每个侧面都需要四个顶点来描述它们。

我们将从我们的另外两个列表中计算出这些顶点：

```py
        sides = [(0, 1), (1, 2), (2, 0)]
        side_vertices = list()
        for index1, index2 in sides:
            side_vertices += [
                front_vertices[index1],
                back_vertices[index1],
                back_vertices[index2],
                front_vertices[index2]
            ]
```

`sides`列表包含了`front_vertices`和`back_vertices`列表的索引，它们定义了每个三角形的侧面。我们遍历这个列表，对于每一个，定义一个包含四个顶点描述楔形一个侧面的列表。

请注意，这四个顶点是按逆时针顺序绘制的，就像正面一样（您可能需要在纸上草图来看清楚）。

我们还将定义一个新的颜色列表，因为现在我们需要更多的颜色：

```py
        side_colors = [
            qtg.QColor('blue'),
            qtg.QColor('purple'),
            qtg.QColor('cyan'),
            qtg.QColor('magenta'),
        ]
        gl_colors = [
            self.qcolor_to_glvec(color)
            for color in side_colors
        ] * 3
```

我们的侧面顶点列表包含了总共12个顶点（每个侧面4个），所以我们需要一个包含12个颜色的列表来匹配它。我们可以通过只指定4种颜色，然后将Python的`list`对象乘以3来产生一个重复的列表，总共有12个项目。

现在，我们将把这些数组传递给OpenGL并进行绘制：

```py
        self.program.setAttributeArray(self.color_location, gl_colors)
        self.program.setAttributeArray(self.vertex_location, side_vertices)
        self.gl.glDrawArrays(self.gl.GL_QUADS, 0, len(side_vertices))
```

这一次，我们使用`GL_QUADS`作为第一个参数，而不是`GL_TRIANGLES`，以指示我们正在绘制四边形。

OpenGL可以绘制多种不同的基元类型，包括线、点和多边形。大多数情况下，您应该使用三角形，因为这是大多数图形硬件上最快的基元。

现在我们所有的点都绘制完毕，我们来清理一下：

```py
        self.program.disableAttributeArray(self.vertex_location)
        self.program.disableAttributeArray(self.color_location)
        self.program.release()
```

在我们简单的演示中，这些调用并不是严格必要的，但是在一个更复杂的程序中，它们可能会为您节省一些麻烦。OpenGL作为一个状态机运行，其中操作的结果取决于系统的当前状态。当我们绑定或启用特定对象时，OpenGL就会指向该对象，并且某些操作（例如设置数组数据）将自动指向它。当我们完成绘图操作时，我们不希望将OpenGL指向我们的对象，因此在完成后释放和禁用对象是一个良好的做法。

如果现在运行应用程序，您应该会看到您惊人的3D形状：

![](assets/67c57d27-65dc-43bf-ada2-e4e1549f0e6d.png)

哎呀，不太3D，是吧？实际上，我们*已经*绘制了一个3D形状，但你看不到，因为我们直接在它上面看。在下一节中，我们将创建一些代码来使这个形状动起来，并充分欣赏它的所有维度。

# OpenGL绘图的动画和控制

为了感受我们绘图的3D特性，我们将在GUI中构建一些控件，允许我们围绕绘图进行旋转和缩放。

我们将从在`MainWindow.__init__()`中添加一些按钮开始，这些按钮可以用作控件：

```py
        btn_layout = qtw.QHBoxLayout()
        main.layout().addLayout(btn_layout)
        for direction in ('none', 'left', 'right', 'up', 'down'):
            button = qtw.QPushButton(
                direction,
                autoExclusive=True,
                checkable=True,
                clicked=getattr(oglw, f'spin_{direction}'))
            btn_layout.addWidget(button)
        zoom_layout = qtw.QHBoxLayout()
        main.layout().addLayout(zoom_layout)
        zoom_in = qtw.QPushButton('zoom in', clicked=oglw.zoom_in)
        zoom_layout.addWidget(zoom_in)
        zoom_out = qtw.QPushButton('zoom out', clicked=oglw.zoom_out)
        zoom_layout.addWidget(zoom_out)
```

我们在这里创建了两组按钮；第一组将是一组单选样式的按钮（因此一次只能有一个被按下），它们将选择对象的旋转方向——无（不旋转）、左、右、上或下。每个按钮在激活时都会调用`GlWidget`对象上的相应方法。

第二组包括一个放大和一个缩小按钮，分别在`GlWidget`上调用`zoom_in()`或`zoom_out()`方法。通过将这些按钮添加到我们的GUI，让我们跳到`GlWidget`并实现回调方法。

# 在OpenGL中进行动画

动画我们的楔形纯粹是通过操纵`view`矩阵并重新绘制我们的图像。我们将在`GlWidget.initializeGL()`中通过创建一个实例变量来保存旋转值：

```py
        self.rotation = [0, 0, 0, 0]
```

此列表中的第一个值表示旋转角度；其余的值是`view`矩阵将围绕的点的*X*、*Y*和*Z*坐标。

在`paintGL()`的末尾，我们可以将这些值传递给`matrix`对象的`rotate()`方法：

```py
        self.view_matrix.rotate(*self.rotation)
```

现在，这将不起作用，因为我们的旋转值都是`0`。要进行旋转，我们将不得不改变`self.rotation`并触发图像的重绘。

因此，我们的旋转回调看起来像这样：

```py
    def spin_none(self):
        self.rotation = [0, 0, 0, 0]

    def spin_left(self):
        self.rotation = [-1, 0, 1, 0]

    def spin_right(self):
        self.rotation = [1, 0, 1, 0]

    def spin_up(self):
        self.rotation = [1, 1, 0, 0]

    def spin_down(self):
        self.rotation = [-1, 1, 0, 0]
```

每个方法只是改变了我们旋转向量的值。角度向前（`1`）或向后（`1`）移动一个度数，围绕一个适当的点产生所需的旋转。

现在，我们只需要通过触发重复的重绘来启动动画。在`paintGL()`的末尾，添加这一行：

```py
        self.update()
```

`update()`在`event`循环中安排了一次重绘，这意味着这个方法会一遍又一遍地被调用。每次，我们的`view`矩阵都会按照`self.rotation`中设置的角度进行旋转。

# 放大和缩小

我们还想要实现缩放。每次点击放大或缩小按钮时，我们希望图像可以稍微靠近或远离一点。

这些回调看起来像这样：

```py
    def zoom_in(self):
        self.view_matrix.scale(1.1, 1.1, 1.1)

    def zoom_out(self):
        self.view_matrix.scale(.9, .9, .9)
```

`QMatrix4x4`的`scale()`方法会使矩阵将每个顶点点乘以给定的数量。因此，我们可以使我们的对象缩小或放大，产生它更近或更远的错觉。

我们可以在这里使用`translate()`，但是在旋转时使用平移可能会导致一些混乱的结果，我们很快就会失去对我们对象的视野。

现在，当您运行应用程序时，您应该能够旋转您的楔形并以其所有的3D光辉看到它：

![](assets/58b84077-1876-4e42-81a2-61c883b69a2d.png)

这个演示只是OpenGL可以做的开始。虽然本章可能没有使您成为OpenGL专家，但希望您能更加自如地深入挖掘本章末尾的资源。

# 总结

在本章中，您已经了解了如何使用OpenGL创建3D动画，以及如何将它们集成到您的PyQt应用程序中。我们探讨了OpenGL的基本原理，如渲染管道、着色器和GLSL。我们学会了如何使用Qt小部件作为OpenGL上下文来绘制和动画一个简单的3D对象。

在下一章中，我们将学习使用`QtCharts`模块交互地可视化数据。我们将创建基本的图表和图形，并学习如何使用模型-视图架构构建图表。

# 问题

尝试这些问题来测试您从本章中学到的知识：

1.  OpenGL渲染管线的哪些步骤是可由用户定义的？为了渲染任何东西，必须定义哪些步骤？您可能需要参考文档[https://www.khronos.org/opengl/wiki/Rendering_Pipeline_Overview](https://www.khronos.org/opengl/wiki/Rendering_Pipeline_Overview)。

1.  您正在为一个OpenGL 2.1程序编写着色器。以下内容看起来正确吗？

```py
   #version 2.1

   attribute highp vec4 vertex;

   void main (void)
   {
   gl_Position = vertex;
   }
```

1.  以下是“顶点”还是“片段”着色器？你如何判断？

```py
   attribute highp vec4 value1;
   varying highp vec3 x[4];
   void main(void)
   {
     x[0] = vec3(sin(value1[0] * .4));
     x[1] = vec3(cos(value1[1]));
     gl_Position = value1;
     x[2] = vec3(10 * x[0])
   }
```

1.  给定以下“顶点”着色器，您需要编写什么代码来为这两个变量分配简单的值？

```py
   attribute highp vec4 coordinates;
   uniform highp mat4 matrix1;

   void main(void){
     gl_Position = matrix1 * coordinates;
   }
```

1.  您启用面剔除以节省一些处理能力，但发现绘图中的几个可见基元现在没有渲染。问题可能是什么？

1.  以下代码对我们的OpenGL图像有什么影响？

```py
   matrix = qtg.QMatrix4x4()
   matrix.perspective(60, 4/3, 2, 10)
   matrix.translate(1, -1, -4)
   matrix.rotate(45, 1, 0, 0)
```

1.  尝试使用演示，看看是否可以添加以下功能：

+   +   更有趣的形状（金字塔、立方体等）

+   移动对象的更多控制

+   阴影和光照效果

+   对象中的动画形状变化

# 进一步阅读

欲了解更多信息，请参考以下内容：

+   现代OpenGL编程的完整教程可以在[https://paroj.github.io/gltut](https://paroj.github.io/gltut)找到。

+   Packt Publications的*Learn OpenGL*，网址为[https://www.packtpub.com/game-development/learn-opengl](https://www.packtpub.com/game-development/learn-opengl)，是学习OpenGL基础知识的良好资源

+   中央康涅狄格州立大学提供了一份关于3D图形矩阵数学的免费教程，网址为[https://chortle.ccsu.edu/VectorLessons/vectorIndex.html](https://chortle.ccsu.edu/VectorLessons/vectorIndex.html)。
