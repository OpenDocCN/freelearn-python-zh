# 第八章。数据输入与数据输出

在本章中，我们将涵盖：

+   在硬盘上创建新文件

+   将数据写入新创建的文件

+   将数据写入多个文件

+   向现有文件添加数据

+   将 Tkinter 绘图形状保存到磁盘

+   从磁盘检索 Python 数据

+   简单的鼠标输入

+   存储和检索鼠标绘制的形状

+   鼠标线条编辑器

+   所有可能的鼠标操作

# 简介

现在我们来讨论在硬盘等存储介质上存储和检索图形数据的细节。除了位图图像外，我们还需要能够创建、存储和检索越来越复杂的矢量图形。我们还希望有将位图图像的部分转换为矢量图像的技术。

到目前为止，我们所有的程序都将其数据包含在源代码中。这限制了我们可以方便地在几分钟内键入的数据列表和数组复杂性。我们不希望有这种限制。我们希望能够处理和操作可能达到数百兆字节大小的原始数据块。手动键入这样的文件是难以想象的低效。有更好的做事方式。这就是命名文件、数据流和硬盘的作用。

# 在硬盘上创建新文件

我们编写并执行了最简单的程序，该程序将在磁盘上创建一个数据文件。

到目前为止，我们不需要在硬盘或 USB 闪存盘上存储任何数据。现在，我们将通过一系列简单的练习来存储和检索存储介质上的文件数据。然后，我们使用这些方法以实际的方式保存和编辑 Tkinter 线条。Tkinter 线条可以是一组单独的线段和形状的集合。如果我们正在开发复杂且丰富的绘图，我们能够存储和检索工作进度是至关重要的。

## 如何做到...

按照常规方式编写、保存并执行显示的程序。当你运行程序时，你将观察到的唯一成功执行迹象是在你点击*Enter*后短暂的暂停。执行将没有任何消息而终止。然而，现在在目标目录`constr`中存在一个名为`brand_new_file.dat`的新文件。我们应该打开`constr`并验证这确实如此。

```py
# file_make_1 .py
# >>>>>>>>>>>>>>
filename = "constr/brand_new_file.dat"
FILE = open(filename,"w")

```

## 它是如何工作的...

这个看起来简约的程序实现了以下目标：

+   它验证了 Python 的文件 I/O 函数存在且正在工作。不需要导入任何模块

+   它展示了 Python 访问存储设备上的数据文件的方式并没有什么异常

+   它证明了操作系统遵循 Python 的文件创建指令

## 如何读取新创建的文件

一旦创建了一个文件，就可以读取它。因此，一个读取磁盘上现有文件的程序会是这样的：

```py
# file_read_1 .py
# >>>>>>>>>>>
filename = "constr/brand_new_file.dat"
FILE = open(filename,"r")

```

如你所见，唯一的区别是`r`而不是`w`。

注意，Python 以多种格式读取和写入文件。在`rb`和`wb`中的`b`表示以字节或二进制格式读取和写入。这些是每个字节中的`1s`和`0s`。在我们的例子中，没有`b`的`r`和`w`告诉 Python 解释器必须将字节解释为 ASCII 字符。我们唯一需要记住的是保持格式分离。

# 将数据写入新创建的文件

我们现在创建一个文件，然后向其中写入一小部分数据。这些非常简单的菜谱的价值在于，当我们尝试一些复杂且不符合预期的任务时，简单的一步测试程序允许我们将问题分解成简单的任务，我们可以逐步增加复杂性，验证每个新更改的有效性。这是许多最佳程序员使用的经过验证和信任的哲学。

```py
# file_write_1.py
#>>>>>>>>>>>>>
# Let's create a file and write it to disk.
filename = "/constr/test_write_1.dat"
filly = open(filename,"w") # Create a file object, in write # mode
for i in range(0,2):
filly.write("everything inside quotes is a string, even 3.1457")
filly.writelines("\n")
filly.write("How will stored data be delimited so we can read \ chunks of it into elements of list, tuple or dictionart?")
filly.writelines("\n")
#filly.close()

```

## 它是如何工作的...

在这一点上需要注意的重要事情是，换行符`\n`是 Python 区分变量的自然方式。空格字符也将用作数字或字符值的分隔符或定界符。

# 将数据写入多个文件

我们在这里看到，像我们预期的那样，使用 Python 打开和写入一系列单独的文件非常简单直接。一旦我们看到了正确的语法示例，它就会正常工作。

```py
# file_write_2.py
#>>>>>>>>>>>>>
# Let's create a file and write it to disk.
filename_1 = "/constr/test_write_1.dat"
filly = open(filename_1,"w") # Create a file object, in # write mode
filly.write("This is number one and the fun has just begun")
filename_2 = "/constr/test_write_2.dat"
filly = open(filename_2,"w") # Create a file object, in # write mode
filly.write("This is number two and he has lost his shoe")
filename_3 = "/constr/test_write_3.dat"
filly = open(filename_3,"w") # Create a file object, in # write mode
filly.write("This is number three and a bump is on his knee")
#filly.close()

```

## 它是如何工作的...

这个示例的价值在于它提供了正确的调试语法示例。因此，它可以以最小的麻烦进行重用和修改。

# 向现有文件添加数据

我们测试了三种将数据写入现有文件的方法，以发现一些基本的数据存储规则。

```py
# file_append_1.py
#>>>>>>>>>>>>>>>>>
# Open an existing file and add (append) data to it.
filename_1 = "/constr/test_write_1.dat"
filly = open(filename_1,"a") # Open a file in append mode
filly.write("\n")
filly.write("This is number four and he has reached the door")
for i in range(0,5):
filename_2 = "/constr/test_write_2.dat"
filly = open(filename_2,"a") # Create a file in append mode
filly.write("This is number five and the cat is still alive")
filename_3 = "/constr/test_write_2.dat"
filly = open(filename_3,"w") # Open an existing file in # write mode
# The command below WILL fail "w" is really "overwrite"
filly.write("This is number six and they cannot find the fix")

```

## 它是如何工作的...

构成第一种方法的是两件事：首先，我们以追加模式打开文件（"a"），这意味着我们将数据添加到文件中已有的内容。不会销毁或覆盖任何内容。其次，我们通过以下行将新数据与旧数据分开：

`filly.write("\n")`

第二种方法可行，但这是一个非常不好的实践，因为没有方法来分离不同的条目。

第三种方法会清除文件中之前存储的内容。

## 所以记住 write 和 append 之间的区别

如果我们清楚地记住上述三种方法，我们将能够成功存储和检索我们的数据，而不会出现错误和挫败感。

# 将 Tkinter 绘制的形状保存到磁盘

当我们使用 Tkinter 创建一个复杂的形状时，我们通常希望保留这个形状以供以后使用。实际上，我们希望建立一个整个形状库。如果其他人做类似的工作，我们可能希望共享和交换形状。这种社区努力是大多数强大且成功的开源程序成功的关键。

## 准备工作

如果我们回到名为“绘制复杂形状——螺旋藤”的例子，在*第二章，绘制基本形状*中，我们会看到形状是由两个坐标列表`vine_x`和`vine_y`定义的。我们首先将这些形状保存到磁盘文件中，然后看看成功检索和绘制它们需要什么。

在你的硬盘上创建一个名为`/constr/vector_shapes`的文件夹，以便接收你的存储数据。

## 如何做...

按照常规方式执行显示的程序。

```py
# save_curly_vine_1.py
#>>>>>>>>>>>>>>>>>
vine_x = [23, 20, 11, 9, 29, 52, 56, 39, 24, 32, 53, 69, 63, \ 47, 35, 35, 51,\
82, 116, 130, 95, 67, 95, 114, 95, 78, 95, 103, 95, 85, 95, 94.5]
vine_y = [36, 44, 39, 22, 16, 32, 56, 72, 91, 117,125, 138, 150, \ 151, 140, 123, 107,\
92, 70, 41, 5, 41, 66, 41, 24, 41, 53, 41, 33, 41, 41, 39]
vine_1 = open('/constr/vector_shapes/curley_vine_1.txt', 'w')
vine_1.write(str(vine_x ))
vine_1.write("\n")
vine_1.write(str(vine_y ))

```

## 它是如何工作的...

首先要注意的是，存储的数据没有“类型”，它仅仅是文本字符。因此，任何要追加到打开文件的数據都必须使用字符串转换函数`str(some_integer_or_float_object)`转换为字符串格式。

第二点要注意的是，将整个列表作为一个列表对象存储，例如`str(vine_x)`，是做事的最佳方式，因为以这种方式存储后，可以直接作为一个整行读入到类似的列表对象中，参见下一道菜谱了解如何做到这一点。在典型的 Python 风格中，简单而明显的方法似乎总是最好的。

## 存储命令

当我们检索混合整数和浮点数据的列表时面临的问题是，它被存储为一个长的字符字符串。那么我们如何让 Python 将包含方括号、逗号、空格和新行字符的长字符串列表转换为正常的 Python 数值列表呢？我们希望我们的绘图不受损坏。有一个很棒的功能`eval()`可以轻松完成这项工作。

另有一种名为 pickle 的方法也能做到同样的事情。

# 从磁盘存储中检索 Python 数据

我们从存储的文件`curley_vine_1.txt`中检索两个列表`vine_x`和`vine_y`。我们希望它们与存储之前的形式完全相同。

## 准备工作

这个菜谱的准备是通过运行之前的程序`save_curly_vine_1.py`来完成的。如果这个程序运行成功，那么在`/constr/vector_shapes`目录下将会有一个名为`curly_vine_1.txt`的文件。如果你打开这个文本文件，你会看到两行，第一行是我们原始的`vine_x`的字符串表示，同样地，这个文件的第二行将代表`vine_y`。

```py
# retrieve_curly_vine_1.py
#>>>>>>>>>>>>>>>>>>>>>
#vine_x = []
vine_1 = open('/constr/vector_shapes/curley_vine_1.txt', 'r')
vine_x = eval(vine_1.readline())
vine_y = eval(vine_1.readline())
# Tests to confirm that everything worked.
print "vine_x = ",vine_x
print vine_x[31]
print "vine_y = ",vine_y
print vine_y[6]

```

## 它是如何工作的...

这之所以如此简单而优雅，是因为`eval()`函数的存在。文档中提到：“*expression*参数被解析并作为 Python 表达式评估”以及“返回值是评估表达式的结果”。这意味着括号内的文本被当作普通的 Python 表达式处理并执行。在我们的特定例子中，花括号内的字符串被解释为一个数字列表，而不是字符，这正是我们想要的。

# 简单鼠标输入

我们现在开发代码，通过捕捉电子图纸上鼠标点击而不是使用铅笔、橡皮和由死树制成的纸张来帮助绘制复杂的形状。我们将这个复杂任务分解为下一个三个菜谱中涵盖的简单步骤。

```py
# mouseclick_1.py
#>>>>>>>>>>>>>>>
from Tkinter import *
root = Tk()
frame = Frame(root, width=100, height=100)
def callback(event):
print "clicked at", event.x, event.y
frame.bind("<Button-1>", callback)
frame.grid()
root.mainloop()
root.destroy()

```

## 它是如何工作的...

点击鼠标按钮被称为事件。如果我们想让我们的程序在程序内部执行某些操作，那么我们需要编写一个`callback`函数，每当事件发生时都会调用这个函数。`callback`的旧术语是“中断服务例程”。

这行`frame.bind("<Button-1>", callback)`实际上表示：

“在事件（即鼠标左键点击`(<Button-1>)`）和被调用的函数`callback`之间建立一个连接（绑定）”。你可以给这个函数取任何你喜欢的名字，但“callback”这个词可以使代码更容易理解。

需要注意的最后一个点是，变量`event.x`和`event.y`是保留用于记录鼠标的 x-y 坐标。在这个特定的`callback`函数中，我们打印出鼠标点击时的位置，在一个称为“frame”的框架中。

## 还有更多...

在接下来的两个菜谱中，我们基于使用鼠标触发的`callback`函数，目的是制作一个形状追踪工具。

# 存储和检索鼠标绘制的形状

我们制作一个程序，通过使用鼠标和三个按钮，我们可以将形状存储到磁盘上，清除画布，然后召回并在屏幕上显示形状。

## 准备工作

确保你已经创建了一个名为`constr`的文件夹，因为这是我们的程序期望能够保存绘制的形状的地方。它也是当被命令检索并显示时将从中检索的地方。

```py
# mouse_shape_recorder_1.py
#>>>>>>>>>>>>>>>>>>>>>
from Tkinter import *
root = Tk()
root.title("Mouse Drawn Shape Saver")
cw = 600 # canvas width
ch = 400 # canvas height
chart_1 = Canvas(root, width=cw, height=ch, background="#ffffff")
chart_1.grid(row=1, column=1)
pt = [0]
x0 = [0]
y0 = [0]
count_point = 0
x_end = 10
y_end = 10
#============================================
# Create a new circle where the click happens and draw a new line
# segment to the last point (where the mouse was left clicked).
def callback_1(event): # Left button pressed.
global count_point, x_end, y_end
global x0, y0
global x0_n, y0_n, pt
x_start = x_end
y_start = y_end
x_end = event.x
y_end = event.y
chart_1.create_line(x_start, y_start , x_end,y_end , fill = \ "#0088ff")
chart_1.create_oval(x_end-5,y_end-5, x_end+5, y_end+5, outline = \ "#0088ff")
count_point += 1
pt = pt + [count_point]
x0 = x0 + [x_end] # extend list of all points
y0 = y0 + [y_end]
chart_1.bind("<Button-1>", callback_1) # <button-1> left mouse button
#==============================================
# 1\. Button control to store segmented line
def callback_6():
global x0, y0
xy_points = open('/constr/shape_xy_1.txt', 'w')
xy_points.write(str(x0))
xy_points.write('\n')
xy_points.write(str(y0))
xy_points.close()
Button(root, text="Store", command=callback_6).grid(row=0, column=2)
#=============================================
# 2\. Button control to retrieve line from file.
def callback_7():
global x0, y0 # Stored list of mouse-click positions.
xy_points = open('/constr/shape_xy_1.txt', 'r')
x0 = eval(xy_points.readline())
y0 = eval(xy_points.readline())
xy_points.close()
print "x0 = ",x0
print "y0 = ",y0
for i in range(1, count_point): # Re-plot the stored and # retreived line
chart_1.create_line(x0[i], y0[i] , x0[i+1], y0[i+1] , \ fill = "#0088ff")
chart_1.create_oval(x_end - 5,y_end - 5, x_end + 5, \ y_end + 5 , outline = "#0088ff")
Button(root, text="retrieve", command=callback_7).grid(row=1, \ column=2)
#=============================================
# 3\. Button control to clear canvas
def callback_8():
chart_1.delete(ALL)
Button(root, text="CLEAR", command=callback_8).grid(row=2, column=2)
root.mainloop()

```

## 它是如何工作的...

除了用于将左鼠标点击的位置添加到列表`x0`和`y0`（x 和 y 坐标）的`callback`函数外，我们还有另外三个`callback`函数。这三个额外的`callback`函数是用来触发执行以下功能的：

+   将列表`x0`和`y0`保存到名为`shape_xy_1.txt`的磁盘文件中。

+   清除画布上的所有绘制线条和圆圈

+   检索`shape_xy_1.txt`的内容，并将其重新绘制到画布上

## 还有更多...

绘制是一个不完美的过程，艺术家和制图员会使用橡皮和铅笔。当我们用连接到计算机的鼠标绘制时，我们也需要对所绘制的任何线条进行调整和修正。我们需要编辑能力。

绘制是一个不完美的过程。我们希望能够调整一些点的位置，以改善绘制。我们将在下一个菜谱中这样做。

# 鼠标线条编辑器

在绘制完成后，我们编辑（更改）使用鼠标绘制的形状。

## 准备工作

为了限制代码的复杂性和长度，我们排除了上一个菜谱中提供的存储和回忆绘制形状的功能。因此，在这个菜谱中不会使用任何存储文件夹。

```py
# mouse_shape_editor_1.py
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from Tkinter import *
import math
root = Tk()
root.title("Left drag to draw, right to re-position.")
cw = 600 # canvas width
ch = 650 # canvas height
chart_1 = Canvas(root, width=cw, height=ch, background="#ffffff")
chart_1.grid(row=1, column=1)
linedrag = {'x_start':0, 'y_start':0, 'x_end':0, 'y_end':0}
map_distance = 0
dist_meter = 0
x_initial = 0
y_initial = 0
#==============================================
# Adjust the distance between points if desired
way_points = 50 # Distance between editable way-points
#==============================================
magic_circle_flag = 0 # 0-> normal dragging, 1 -> double-click: # Pull point.
point_num = 0
x0 = []
y0 = []
#================================================
def separation(x_now, y_now, x_dot, y_dot): # DISTANCE MEASUREMENT
# Distance to points - used to find out if the mouse # clicked inside a circle
sum_squares = (x_now - x_dot)**2 + (y_now -y_dot)**2
distance= int(math.sqrt(sum_squares)) # Get Pythagorean # distance
return( distance)
#================================================
# CALLBACK EVENT PROCESSING FUNCTIONS
def callback_1(event): # LEFT DOWN
global x_initial, y_initial
x_initial = event.x
y_initial = event.y
def callback_2(event): # LEFT DRAG
global x_initial, y_initial
global map_distance, dist_meter
global x0, y0
linedrag['x_start'] = linedrag['x_end'] # update positions
linedrag['y_start'] = linedrag['y_end']
linedrag['x_end'] = event.x
linedrag['y_end'] = event.y
increment = separation(linedrag['x_start'],linedrag['y_start'], \ linedrag['x_end'], linedrag['y_end'] )
map_distance += increment # Total distance - # potentiasl use as a map odometer.
dist_meter += increment # Distance from last circle
if dist_meter>way_points: # Action at way-points
x0.append(linedrag['x_end']) # append to line
y0.append(linedrag['y_end'])
xb = linedrag['x_end'] - 5 ; yb = linedrag['y_end'] - 5 # Centre circle on line
x1 = linedrag['x_end'] + 5 ; y1 = linedrag['y_end'] + 5
chart_1.create_oval(xb,yb, x1,y1, outline = "green")
dist_meter = 0 # re-zero the odometer.
linexy = [ x_initial, y_initial, linedrag['x_end'] , \ linedrag['y_end'] ]
chart_1.create_line(linexy, fill='green')
x_initial = linedrag['x_end'] # start of next segment
y_initial = linedrag['y_end']
def callback_5(event): # RIGHT CLICK
global point_num, magic_circle_flag, x0, y0
# Measure distances to each point in turn, determine if any are # inside magic circle.
# That is, identify which point has been clicked on.
for i in range(0, len(x0)):
d = separation(event.x, event.y, x0[i], y0[i])
if d <= 5:
point_num = i # this is the index that controls editing
magic_circle_flag = 1
chart_1.create_oval(x0[i] - 10,y0[i] - 10, x0[i] + 10, \ y0[i] + 10 , width = 4, outline = "#ff8800")
x0[i] = event.x
y0[i] = event.y
def callback_6(event): # RIGHT RELEASE
global point_num, magic_circle_flag, x0, y0
if magic_circle_flag == 1: # The point is going to be # repositioned.
x0[point_num] =event.x
y0[point_num] =event.y
chart_1.delete(ALL)
chart_1.update() # Refreshes the drawing on the # canvas.
q=[]
for i in range(0,len(x0)):
q.append(x0[i])
q.append(y0[i])
chart_1.create_oval(x0[i] - 5,y0[i] - 5, x0[i] + 5, \ y0[i] + 5 , outline = "#00ff00")
chart_1.create_line(q , fill = "#ff00ff") # Now show the # new positions
magic_circle_flag = 0
#==============================
chart_1.bind("<Button-1>", callback_1) # <Button-1> ->LEFT mouse # button
chart_1.bind("<B1-Motion>", callback_2)
chart_1.bind("<Button-3>", callback_5) # <Button-3> ->RIGHT mouse # button
chart_1.bind("<ButtonRelease-3>", callback_6)
root.mainloop()

```

## 它是如何工作的...

之前的程序现在包括：

+   处理左右鼠标点击和拖动的 `callback` 函数。

距离测量函数 `separation(x_now, y_now, x_dot, y_dot)`。当右键点击时，测量到每个线段的距离。如果这些距离中的任何一个在现有接点内部，则绘制一个橙色圆圈，并将控制权传递给 `callback_6`，该函数更新新点的坐标并刷新修改后的绘图。是否移动点的决定由 `magic_circle_flag` 的值决定。这个标志的状态由 `separation()` 计算的距离确定。当右键按下时，如果距离测量发现它在接点内部，则将其设置为 `1`，在移动一个点之后将其设置为 `0`。

## 还有更多...

现在我们有了通过鼠标操作来控制和调整线条和曲线绘制的方法，其他可能性也被打开了。

### 为什么不添加更多功能？

好好扩展这个程序的功能，包括：

+   删除点的功能

+   处理未连接段的能力

+   选择或点击创建点的功能

+   拖动仙女灯（等长段）

随着我们工作的扩展，这个列表会变得越来越长。最终，我们将创建一个有用的矢量图形编辑器，并且会有压力去匹配现有专有和开源编辑器的功能。为什么重造轮子？如果这是一个实际的选择，那么与现有成熟的矢量编辑器产生的矢量图像一起工作的努力可能会更有成效。

### 使用其他工具获取和重新处理图像

在下一章中，我们将探讨如何使用开源矢量图形编辑器 Inkscape 中的矢量图像。Inkscape 能够导出多种格式的图像，包括一种标准化的网络格式，称为**缩放矢量图形**或简称**SVG**。

### 如何利用鼠标

本章大量使用了鼠标作为用户交互工具，在 Tkinter 画布上绘制形状。为了充分利用鼠标的技能，下一道菜谱将是对鼠标交互完整工具包的考察。

### 我们可以测量蜿蜒线上的距离

在代码中，有一个名为 `map_distance` 的变量尚未使用。它可以用来追踪地图上蜿蜒路径的距离。想法是，如果我们想在类似谷歌地图这样的地图上测量未标记路径和道路的距离，我们就能将这个方法适应到这项任务中。

# 所有可能的鼠标动作

现在我们编写一个程序来测试 Python 能够响应的每个可能的鼠标事件。

## 如何操作...

以正常方式执行显示的程序。

```py
# all_mouse_actions_1.py
#>>>>>>>>>>>>>>>>>>
from Tkinter import *
root = Tk()
root.title("Mouse follower")
# The Canvas here is bound to the mouse events
cw = 200 # canvas width
ch = 100 # canvas height
chart_1 = Canvas(root, width=cw, height=ch, background="#ffffff")
chart_1.grid(row=1, column=1)
#========= Left Mouse Button Events ===============
# callback events
def callback_1(event):
print "left mouse clicked"
def callback_2(event):
print "left dragged"
def callback_3(event):
print "left doubleclick"
def callback_4(event):
print "left released"
#======== Center Mouse Button Events ======================
def callback_5(event):
print "center mouse clicked"
def callback_6(event):
print "center dragged"
def callback_7(event):
print "center doubleclick"
def callback_8(event):
print "center released"
#======== Right Mouse Button Events ======================
def callback_9(event):
print "right mouse clicked"
def callback_10(event):
print "right dragged"
def callback_11(event):
print "right doubleclick"
def callback_12(event):
print "right released"
# <button-1> is the left mouse button
chart_1.bind("<Button-1>", callback_1)
chart_1.bind("<B1-Motion>", callback_2)
chart_1.bind("<Double-1>", callback_3)
chart_1.bind("<ButtonRelease-1>", callback_4)
# <button-2> is the center mouse button
chart_1.bind("<Button-2>", callback_5)
chart_1.bind("<B2-Motion>", callback_6)
chart_1.bind("<Double-2>", callback_7)
chart_1.bind("<ButtonRelease-2>", callback_8)
# <button-3> is the right mouse button
chart_1.bind("<Button-3>", callback_9)
chart_1.bind("<B3-Motion>", callback_10)
chart_1.bind("<Double-3>", callback_11)
chart_1.bind("<ButtonRelease-3>", callback_12)
root.mainloop()
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

```

## 它是如何工作的...

上述代码相当直观。创建了一个小画布，它能够响应所有鼠标操作。确认响应是否正确工作是通过在系统控制台上输入确认消息来实现的。我们可以通过在 `callback` 函数中插入适当的 Python 命令来调整 `callback` 函数，以执行我们选择的任何任务。

## 还有更多...

鼠标事件和 Tkinter 小部件通常协同工作。大多数 Tkinter 图形用户界面小部件都是设计用来通过鼠标事件进行控制的，例如左键或右键点击，或者按住按钮进行拖动。Tkinter 提供了丰富的选择，这些小部件将在 *第十章，GUI 构建第一部分* 和 *第十一章，GUI 构建第二部分* 中进行探讨。
