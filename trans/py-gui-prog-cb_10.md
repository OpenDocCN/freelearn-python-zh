# 第10章。使用PyOpenGL和PyGLet创建令人惊叹的3D GUI

在本章中，我们将创建令人惊叹的Python GUI，显示真正的可以旋转的三维图像，这样我们可以从各个角度观察它们。

+   PyOpenGL转换了我们的GUI

+   我们的3D GUI！

+   使用位图使我们的GUI更漂亮

+   PyGLet比PyOpenGL更容易地转换了我们的GUI

+   我们的GUI有惊人的颜色

+   使用tkinter创建幻灯片放映

# 介绍

在本章中，我们将通过赋予它真正的三维能力来转换我们的GUI。我们将使用两个Python第三方包。PyOpenGL是OpenGL标准的Python绑定，它是一个内置于所有主要操作系统中的图形库。这使得生成的小部件具有本地的外观和感觉。

Pyglet是另一个Python绑定到OpenGL库，但它也可以创建GUI应用程序，这使得使用Pyglet编码比使用PyOpenGL更容易。

# PyOpenGL转换了我们的GUI

在这个教程中，我们将成功创建一个导入PyOpenGL模块并实际工作的Python GUI！

为了做到这一点，我们需要克服一些最初的挑战。

这个教程将展示一个已经被证明有效的方法。如果你自己尝试并卡住了，记住托马斯·爱迪生的著名话语。

### 注意

发明家托马斯·爱迪生在回答一位记者关于爱迪生的失败的问题时说：

*"我并没有失败。我只是找到了一万种行不通的方法。"*

首先，我们必须安装PyOpenGL扩展模块。

成功安装与我们的操作系统架构匹配的PyOpenGL模块后，我们将创建一些示例代码。

## 准备工作

我们将安装PyOpenGL包。在本书中，我们使用的是Windows 7 64位操作系统和Python 3.4。接下来的下载截图是针对这个配置的。

我们还将使用wxPython。如果你没有安装wxPython，你可以阅读前一章关于如何安装wxPython以及如何使用这个GUI框架的一些教程。

### 注意

我们正在使用wxPython Phoenix版本，这是最新版本，旨在将原始的Classic wxPython版本替换掉。

## 如何做...

为了使用PyOpenGL，我们必须首先安装它。以下URL是官方的Python包安装程序网站：

[https://pypi.python.org/pypi/PyOpenGL/3.0.2#downloads](https://pypi.python.org/pypi/PyOpenGL/3.0.2#downloads)

![如何做...](graphics/B04829_10_01.jpg)

这似乎是正确的安装，但事实证明，它在Windows 7 64位操作系统和Python 3.4.3 64位上不起作用。

在前一章的教程中提到了一个更好的查找Python安装包的地方。你可能已经很熟悉了。我们下载与我们的操作系统和Python版本匹配的包。它使用新的`.whl`格式，所以我们首先要安装Python轮式包。

### 注意

如何安装Python轮式包的方法在之前的教程中有描述。

使用`pip`命令通过`PyOpenGL-3.1.1a1-cp34-none-win_amd64.whl`文件安装PyOpenGL既成功又安装了我们需要的所有64位模块。

用下载的轮式安装程序的完整路径替换`<your full path>`。

```py
pip install <your full path> PyOpenGL-3.1.1a1-cp34-none-win_amd64.whl
```

当我们尝试导入一些PyOpenGL模块时，它可以工作，就像在这个代码示例中所看到的那样：

```py
# Ch10_import_OpenGL.py
import wx                  
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLUT import *
```

所有这些代码都在做的是导入几个OpenGL Python模块。它除此之外什么也不做，但是当我们运行我们的Python模块时，我们不会收到任何错误。

这证明我们已成功将OpenGL绑定到Python中。

现在我们的开发环境已经成功设置，我们可以使用wxPython来尝试它。

### 注意

许多在线示例都限制在使用Python 2.x，以及使用Classic版本的wxPython。我们使用的是Python 3和Phoenix。

使用基于wxPython演示示例的代码创建了一个工作的3D立方体。相比之下，运行圆锥体示例没有成功，但这个示例让我们在正确的轨道上开始了。

这是URL：

[http://wiki.wxpython.org/GLCanvas%20update](http://wiki.wxpython.org/GLCanvas%20update)

以下是对代码的一些修改：

```py
import wx
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLUT import *

class MyCanvasBase(glcanvas.GLCanvas):
    def __init__(self, parent):
        glcanvas.GLCanvas.__init__(self, parent, -1)

# This context was missing from the code
        self.context = glcanvas.GLContext(self)  # <- added

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
# We have to pass in a context ------
#         self.SetCurrent()                      # commented out
        self.SetCurrent(self.context)            # <- changed
```

我们现在可以创建以下GUI：

![如何做...](graphics/B04829_10_02.jpg)

在wxPython的经典版本中，`SetCurrent()`不需要上下文。这是我们在网上搜索时可能会找到的一些代码。

```py
    def OnPaint(self, event):

        dc = wx.PaintDC(self)
        self.SetCurrent()
        if not self.init:
            self.InitGL()
            self.init = True
        self.OnDraw()
```

在使用wxPython Phoenix时，前面的代码不起作用。我们可以在网上查找Phoenix的正确语法。

![如何做...](graphics/B04829_10_03.jpg)

## 它是如何工作的...

在这个配方中，我们首次使用了PyOpenGL Python绑定的OpenGL。虽然OpenGL可以创建真正惊人的真3D图像，但我们在这个过程中遇到了一些挑战，然后找到了解决这些挑战的方法，使其工作起来。

### 注意

我们正在用Python编写3D图像！

# 我们的GUI是3D的！

在这个配方中，我们将使用wxPython创建自己的GUI。我们正在重用一些来自wxPython演示示例的代码，我们已经将其减少到显示3D OpenGL所需的最少代码。

### 注意

OpenGL是一个非常庞大的库。我们不会详细解释这个库。如果你想进一步学习OpenGL，有很多书籍和在线文档可供参考。它有自己的着色语言。

## 准备工作

阅读前面的配方可能是准备这个配方的好方法。

## 如何做...

由于整个Python代码有点长，我们只会展示一小部分代码。

整个代码都可以在线获得，这个Python模块被称为：

```py
# Ch10_wxPython_OpenGL_GUI
import wx                  
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLUT import *

#---------------------------------------------------
class CanvasBase(glcanvas.GLCanvas):
    def __init__(self, parent):
        glcanvas.GLCanvas.__init__(self, parent, -1)
        self.context = glcanvas.GLContext(self)
        self.init = False

        # Cube 3D start rotation
        self.last_X = self.x = 30
        self.last_Y = self.y = 30

        self.Bind(wx.EVT_SIZE, self.sizeCallback)
        self.Bind(wx.EVT_PAINT, self.paintCallback)
        self.Bind(wx.EVT_LEFT_DOWN, self.mouseDownCallback)
        self.Bind(wx.EVT_LEFT_UP, self.mouseUpCallback)
        self.Bind(wx.EVT_MOTION, self.mouseMotionCallback)

    def sizeCallback(self, event):
        wx.CallAfter(self.setViewport)
        event.Skip()

    def setViewport(self):
        self.size = self.GetClientSize()
        self.SetCurrent(self.context)
        glViewport(0, 0, self.size.width, self.size.height)

    def paintCallback(self, event):
        wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.init:
            self.initGL()
            self.init = True
        self.onDraw()

    def mouseDownCallback(self, event):
        self.CaptureMouse()
        self.x, self.y = self.last_X, self.last_Y = event.GetPosition()

    def mouseUpCallback(self, evt):
        self.ReleaseMouse()

    def mouseMotionCallback(self, evt):
        if evt.Dragging() and evt.LeftIsDown():
            self.last_X, self.last_Y = self.x, self.y
            self.x, self.y = evt.GetPosition()
            self.Refresh(False)

#-----------------------------------------------------
class CubeCanvas(CanvasBase):
    def initGL(self):
        # set viewing projection
        glMatrixMode(GL_PROJECTION)
        glFrustum(-0.5, 0.5, -0.5, 0.5, 1.0, 3.0)

        # position viewer
        glMatrixMode(GL_MODELVIEW)
        glTranslatef(0.0, 0.0, -2.0)

        # position object
        glRotatef(self.y, 1.0, 0.0, 0.0)
        glRotatef(self.x, 0.0, 1.0, 0.0)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

    def onDraw(self):
        # clear color and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # draw six faces of a cube
        glBegin(GL_QUADS)
        glNormal3f( 0.0, 0.0, 1.0)
        glVertex3f( 0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5,-0.5, 0.5)
        glVertex3f( 0.5,-0.5, 0.5)

        glNormal3f( 0.0, 0.0,-1.0)
        glVertex3f(-0.5,-0.5,-0.5)

#===========================================================
app = wx.App()
frame = wx.Frame(None, title="Python GUI using wxPython", size=(300,230))
GUI(frame)
frame.Show()        
app.MainLoop()      
```

![如何做...](graphics/B04829_10_04.jpg)

前面的屏幕截图显示了我们的wxPython GUI。当我们点击按钮小部件时，会出现以下第二个窗口。

![如何做...](graphics/B04829_10_05.jpg)

### 注意

我们现在可以使用鼠标将立方体转动起来，看到它的所有六个面。

![如何做...](graphics/B04829_10_06.jpg)

我们还可以最大化这个窗口，坐标会缩放，我们可以在这个更大的窗口中旋转这个立方体！

![如何做...](graphics/B04829_10_07.jpg)

这个立方体也可以是一艘星际迷航太空飞船！

如果这是我们想要开发的内容，我们只需要成为这项技术的高级程序员。

### 注意

许多视频游戏正在使用OpenGL开发。

## 它是如何工作的...

我们首先创建了一个常规的wxPython GUI，并在上面放置了一个按钮小部件。单击此按钮会调用导入的OpenGL 3D库。使用的代码是wxPython演示示例的一部分，我们稍微修改了它以使其与Phoenix一起工作。

### 注意

这个配方将我们自己的GUI与这个库粘合在一起。

OpenGL是一个如此庞大和令人印象深刻的库。这个配方让我们体验了如何在Python中创建一个工作示例。

### 注意

通常，一个工作示例就足以让我们开始我们的旅程。

# 使用位图使我们的GUI漂亮

这个配方受到了一个wxPython IDE构建框架的启发，该框架在某个时候曾经起作用。

它不能在Python 3和wxPython Phoenix中工作，但这段代码非常酷。

我们将重用这个项目提供的大量代码中的一个位图图像。

在时间耗尽之前，你可以在GitHub上fork Google代码。

![使用位图使我们的GUI漂亮](graphics/B04829_10_08.jpg)

## 准备工作

在这个配方中，我们将继续使用wxPython，因此阅读至少前一章的部分可能对准备这个配方有用。

## 如何做...

在反向工程gui2py代码并对此代码进行其他更改后，我们可能会实现以下窗口小部件，它显示了一个漂亮的平铺背景。

![如何做...](graphics/B04829_10_09.jpg)

当然，我们在重构之前的网站代码时丢失了很多小部件，但它确实给了我们一个很酷的背景，点击“退出”按钮仍然有效。

下一步是弄清楚如何将代码的有趣部分集成到我们自己的GUI中。

我们通过将以下代码添加到上一个教程的GUI中来实现这一点。

```py
#----------------------------------------------------------
class GUI(wx.Panel):              # Subclass wxPython Panel
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        imageFile = 'Tile.bmp'
        self.bmp = wx.Bitmap(imageFile)
        # react to a resize event and redraw image
        parent.Bind(wx.EVT_SIZE, self.canvasCallback)

    def canvasCallback(self, event=None):
        # create the device context
        dc = wx.ClientDC(self)
        brushBMP = wx.Brush(self.bmp)
        dc.SetBrush(brushBMP)
        width, height = self.GetClientSize()
        dc.DrawRectangle(0, 0, width, height)
```

### 注意

我们必须绑定到父级，而不是self，否则我们的位图将不会显示出来。

现在运行我们改进的代码会将位图平铺为GUI的背景。

![如何做...](graphics/B04829_10_10.jpg)

点击按钮仍然会调用我们的OpenGL 3D绘图，所以我们没有失去任何功能。

![如何做...](graphics/B04829_10_11.jpg)

## 它是如何工作的...

在这个教程中，我们通过使用位图作为背景来增强了我们的GUI。我们平铺了位图图像，当我们调整GUI窗口的大小时，位图会自动调整以填充我们正在使用设备上绘制的画布的整个区域。

### 注意

上述wxPython代码可以加载不同的图像文件格式。

# PyGLet比PyOpenGL更容易地转换我们的GUI

在这个教程中，我们将使用PyGLet GUI开发框架来创建我们的GUI。

PyGLet比PyOpenGL更容易使用，因为它自带了自己的GUI事件循环，所以我们不需要使用tkinter或wxPython来创建我们的GUI。

## 如何做...

为了使用Pyglet，我们首先必须安装这个第三方Python插件。

使用`pip`命令，我们可以轻松安装这个库，成功安装在我们的`site-packages` Python文件夹中看起来像这样：

![如何做...](graphics/B04829_10_12.jpg)

在线文档位于当前版本的这个网站：

[https://pyglet.readthedocs.org/en/pyglet-1.2-maintenance/](https://pyglet.readthedocs.org/en/pyglet-1.2-maintenance/)

![如何做...](graphics/B04829_10_13.jpg)

使用Pyglet库的第一次体验可能是这样的：

```py
import pyglet

window = pyglet.window.Window()
label = pyglet.text.Label('PyGLet GUI', 
                          font_size=42,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center')

@window.event
def on_draw():
    window.clear()
    label.draw()

pyglet.app.run()
```

上述代码来自官方网站pyglet.org，并导致以下完全功能的GUI：

![如何做...](graphics/B04829_10_14.jpg)

## 它是如何工作的...

在这个教程中，我们使用了另一个包装了OpenGL库的第三方Python模块。

这个库自带了自己的事件循环处理能力，这使我们不必依赖另一个库来创建一个运行中的Python GUI。

我们已经探索了官方网站，它向我们展示了如何安装和使用这个奇妙的GUI库。

# 我们GUI中惊人的颜色

在这个教程中，我们将扩展我们使用Pyglet编写的GUI，将其转变为真正的3D。

我们还将为其添加一些花哨的颜色。这个教程受到了*OpenGL SuperBible*图书系列中一些示例代码的启发。它创建了一个非常丰富多彩的立方体，我们可以使用键盘上、下、左、右按钮在三维空间中旋转它。

我们稍微改进了示例代码，使图像在按住一个键时转动，而不是必须按下并释放键。

## 准备工作

上一个教程解释了如何安装PyGLet，并为您介绍了这个库。如果您还没有这样做，浏览一下那一章可能是个好主意。

### 注意

在在线文档中，PyGLet通常以全小写拼写。虽然这可能是一种Pythonic的方式，但我们会将类的第一个字母大写，并且我们使用小写来开始每个变量、方法和函数名。

除非必要澄清代码，否则本书不使用下划线。

## 如何做...

以下代码创建了下面显示的3D彩色立方体。这次，我们将使用键盘箭头键来旋转图像，而不是鼠标。

```py
import pyglet
from pyglet.gl import *
from pyglet.window import key
from OpenGL.GLUT import *

WINDOW    = 400
INCREMENT = 5

class Window(pyglet.window.Window):

    # Cube 3D start rotation
    xRotation = yRotation = 30    

    def __init__(self, width, height, title=''):
        super(Window, self).__init__(width, height, title)
        glClearColor(0, 0, 0, 1)
        glEnable(GL_DEPTH_TEST)    

    def on_draw(self):
        # Clear the current GL Window
        self.clear()

        # Push Matrix onto stack
        glPushMatrix()

        glRotatef(self.xRotation, 1, 0, 0)
        glRotatef(self.yRotation, 0, 1, 0)

        # Draw the six sides of the cube
        glBegin(GL_QUADS)

        # White
        glColor3ub(255, 255, 255)
        glVertex3f(50,50,50)

        # Yellow
        glColor3ub(255, 255, 0)
        glVertex3f(50,-50,50)

        # Red
        glColor3ub(255, 0, 0)
        glVertex3f(-50,-50,50)
        glVertex3f(-50,50,50)

        # Blue
        glColor3f(0, 0, 1)
        glVertex3f(-50,50,-50)

        # <… more color defines for cube faces>

        glEnd()

        # Pop Matrix off stack
        glPopMatrix()

    def on_resize(self, width, height):
        # set the Viewport
        glViewport(0, 0, width, height)

        # using Projection mode
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        aspectRatio = width / height
        gluPerspective(35, aspectRatio, 1, 1000)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, -400)

    def on_text_motion(self, motion): 
        if motion == key.UP:
            self.xRotation -= INCREMENT
        elif motion == key.DOWN:
            self.xRotation += INCREMENT
        elif motion == key.LEFT:
            self.yRotation -= INCREMENT
        elif motion == key.RIGHT:
            self.yRotation += INCREMENT

if __name__ == '__main__':
    Window(WINDOW, WINDOW, 'Pyglet Colored Cube')
    pyglet.app.run()
```

![如何做...](graphics/B04829_10_15.jpg)

使用键盘箭头键，我们可以旋转3D立方体。

![如何做...](graphics/B04829_10_16.jpg)

## 它是如何工作的...

在这个教程中，我们使用pyglet创建了一个丰富多彩的立方体，我们可以使用键盘箭头键在三维空间中旋转它。

我们已经为我们的立方体的六个面定义了几种颜色，并且我们已经使用pyglet创建了我们的主窗口框架。

这段代码与本章中以前的一个食谱类似，在那个食谱中，我们使用wxPython库创建了一个立方体。原因是在底层，wxPython和pyglet都使用OpenGL库。

# 使用tkinter创建幻灯片

在这个食谱中，我们将使用纯Python创建一个漂亮的幻灯片GUI。

我们将看到核心Python内置功能的限制，然后我们将探索另一个可用的第三方模块Pillow，它扩展了tkinter在图像处理方面的内置功能。

虽然一开始Pillow这个名字听起来有点奇怪，但它实际上有很多历史背景。

### 注意

在本书中，我们只使用Python 3.4及以上版本。

我们不会回到Python 2。

Guido已经表达了他有意打破向后兼容性的决定，并决定Python 3是Python编程的未来。

对于GUI和图像，Python 2的旧线有一个非常强大的模块，名为PIL，代表Python图像库。这个库具有非常多的功能，在Python 3非常成功创建几年后，这些功能仍未被翻译成Python 3。

许多开发人员仍然选择使用Python 2而不是未来版本，因为Python 2仍然有更多的可用库。

这有点令人伤感。

幸运的是，另一个图像处理库已经被创建出来，可以与Python 3一起使用，它的名字是PIL加一些东西。

### 注意

Pillow与Python 2的PIL库不兼容。

## 准备就绪

在这个食谱的第一部分中，我们将使用纯Python。为了改进代码，我们将使用pip功能安装另一个Python模块。因此，虽然您很可能熟悉pip，但了解如何使用它可能会有用。

## 如何做...

首先，我们将使用纯Python创建一个工作的GUI，在窗口框架内对幻灯片进行洗牌。

这是工作代码，接下来是运行此代码的一些截图的结果：

```py
from tkinter import Tk, PhotoImage, Label
from itertools import cycle
from os import listdir

class SlideShow(Tk):
    # inherit GUI framework extending tkinter
    def __init__(self, msShowTimeBetweenSlides=1500):
        # initialize tkinter super class
        Tk.__init__(self)

        # time each slide will be shown
        self.showTime = msShowTimeBetweenSlides

        # look for images in current working directory 
        listOfSlides = [slide for slide in listdir() if slide.endswith('gif')]

        # cycle slides to show on the tkinter Label 
        self.iterableCycle = cycle((PhotoImage(file=slide), slide) for slide in listOfSlides)

        # create tkinter Label widget which can display images
        self.slidesLabel = Label(self)

        # create the Frame widget
        self.slidesLabel.pack()

    def slidesCallback(self):
        # get next slide from iterable cycle
        currentInstance, nameOfSlide = next(self.iterableCycle)

        # assign next slide to Label widget
        self.slidesLabel.config(image=currentInstance)

        # update Window title with current slide
        self.title(nameOfSlide)

        # recursively repeat the Show
        self.after(self.showTime, self.slidesCallback)

#=================================
# Start GUI
#=================================
win = SlideShow()
win.after(0, win.slidesCallback())
win.mainloop()
```

![如何做...](graphics/B04829_10_21.jpg)

这是幻灯片展示中的另一个时刻。

![如何做...](graphics/B04829_10_22.jpg)

虽然幻灯片的滑动确实令人印象深刻，但纯Python tkinter GUI的内置功能不支持非常流行的`.jpg`格式，因此我们必须使用另一个Python库。

为了使用Pillow，我们首先必须使用`pip`命令安装它。

成功的安装看起来像这样：

![如何做...](graphics/B04829_10_23.jpg)

Pillow支持`.jpg`格式，并且为了使用它，我们必须稍微改变我们的语法。

使用Pillow是一个高级主题，在本书的这一版本中不会涉及。

## 它是如何工作的...

Python是一个非常棒的工具，在这个食谱中，我们已经探索了几种使用和扩展它的方法。

### 注意

当手指指向月亮时，它并不是月亮本身，只是一个指针。
