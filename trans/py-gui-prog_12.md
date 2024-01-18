# 使用 Canvas 小部件可视化数据

在数据库中记录了数月的实验数据后，现在是开始可视化和解释数据的过程。你的同事分析师们询问程序本身是否可以创建图形数据可视化，而不是将数据导出到电子表格中创建图表和图形。为了实现这一功能，你需要了解 Tkinter 的`Canvas`小部件。

在本章中，你将学习以下主题：

+   使用 Canvas 小部件进行绘图和动画

+   使用 Canvas 构建简单的折线图

+   使用 Matplotlib 集成更高级的图表和图表

# 使用 Tkinter 的 Canvas 进行绘图和动画

`Canvas`小部件无疑是 Tkinter 中最强大的小部件。它可以用于构建从自定义小部件和视图到完整用户界面的任何内容。顾名思义，`Canvas`是一个可以绘制图形和图像的空白区域。

可以像创建其他小部件一样创建`Canvas`对象：

```py
root = tk.Tk()
canvas = tk.Canvas(root, width=1024, height=768)
canvas.pack()
```

`Canvas`接受通常的小部件配置参数，以及用于设置其大小的`width`和`height`。创建后，我们可以使用其许多`create_()`方法开始向`canvas`添加项目。

例如，我们可以使用以下代码添加一个矩形：

```py
canvas.create_rectangle(100, 100, 200, 200, fill='orange')
```

前四个参数是左上角和右下角的坐标，以像素为单位，从画布的左上角开始。每个`create_()`方法都是以定义形状的坐标开始的。`fill`选项指定了对象内部的颜色。

坐标也可以指定为元组对，如下所示：

```py
canvas.create_rectangle((600, 100), (700, 200), fill='#FF8800')
```

尽管这是更多的字符，但它显着提高了可读性。还要注意，就像 Tkinter 中的其他颜色一样，我们可以使用名称或十六进制代码。

我们还可以创建椭圆，如下所示：

```py
canvas.create_oval((350, 250), (450, 350), fill='blue')
```

椭圆和矩形一样，需要其**边界框**的左上角和右下角的坐标。边界框是包含项目的最小矩形，因此在这个椭圆的情况下，你可以想象一个圆在一个角坐标为`(350, 250)`和`(450, 350)`的正方形内。

我们可以使用`create_line()`创建线，如下所示：

```py
canvas.create_line((100, 400), (400, 500),
    (700, 400), (100, 400), width=5, fill='red')
```

行可以由任意数量的点组成，Tkinter 将连接这些点。我们已经指定了线的宽度以及颜色（使用`fill`参数）。额外的参数可以控制角和端点的形状，线两端箭头的存在和样式，线条是否虚线，以及线条是直线还是曲线。

类似地，我们可以创建多边形，如下所示：

```py
canvas.create_polygon((400, 150), (350,  300), (450, 300),
    fill='blue', smooth=True)
```

这与创建线条类似，只是 Tkinter 将最后一个点连接回第一个点，并填充内部。将`smooth`设置为`True`会使用贝塞尔曲线使角变圆。

除了简单的形状之外，我们还可以按照以下方式在`canvas`对象上放置文本或图像：

```py
canvas.create_text((400, 600), text='Smile!',
    fill='cyan', font='TkDefaultFont 64')
smiley = tk.PhotoImage(file='smile.gif')
image_item = canvas.create_image((400, 300), image=smiley)
```

任何`create_()`方法的返回值都是一个字符串，它在`Canvas`对象的上下文中唯一标识该项。我们可以使用该标识字符串在创建后对该项进行操作。

例如，我们可以这样绑定事件：

```py
canvas.tag_bind(image_item, '<Button-1>', lambda e: canvas.delete(image_item))
```

在这里，我们使用`tag_bind`方法将鼠标左键单击我们的图像对象绑定到画布的`delete()`方法，该方法（给定一个项目标识符）会删除该项目。

# 为 Canvas 对象添加动画

Tkinter 的`Canvas`小部件没有内置的动画框架，但我们仍然可以通过将其`move()`方法与对事件队列的理解相结合来创建简单的动画。

为了演示这一点，我们将创建一个虫子赛跑模拟器，其中两只虫子（用彩色圆圈表示）将杂乱地向屏幕的另一侧的终点线赛跑。就像真正的虫子一样，它们不会意识到自己在比赛，会随机移动，赢家是哪只虫子碰巧先到达终点线。

首先，打开一个新的 Python 文件，并从以下基本样板开始：

```py
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
```

```py
        super().__init__()

App().mainloop()
```

# 创建我们的对象

让我们创建用于游戏的对象：

1.  在`App.__init__()`中，我们将简单地创建我们的`canvas`对象，并使用`pack()`添加它：

```py
self.canvas = tk.Canvas(self, background='black')
self.canvas.pack(fill='both', expand=1)
```

1.  接下来，我们将创建一个`setup()`方法如下：

```py
   def setup(self):
       self.canvas.left = 0
       self.canvas.top = 0
       self.canvas.right = self.canvas.winfo_width()
       self.canvas.bottom = self.canvas.winfo_height()
       self.canvas.center_x = self.canvas.right // 2
       self.canvas.center_y = self.canvas.bottom // 2

       self.finish_line = self.canvas.create_rectangle(
           (self.canvas.right - 50, 0),
           (self.canvas.right, self.canvas.bottom),
           fill='yellow', stipple='gray50')
```

在上述代码片段中，`setup()`首先通过计算`canvas`对象上的一些相对位置，并将它们保存为实例属性，这将简化在`canvas`对象上放置对象。终点线是窗口右边的一个矩形，使用`stipple`参数指定一个位图，该位图将覆盖实色以赋予其一些纹理；在这种情况下，`gray50`是一个内置的位图，交替黑色和透明像素。

1.  在`__init__()`的末尾添加一个对`setup()`的调用如下：

```py
self.after(200, self.setup)
```

因为`setup()`依赖于`canvas`对象的`width`和`height`值，我们需要确保在操作系统的窗口管理器绘制和调整窗口大小之前不调用它。最简单的方法是将调用延迟几百毫秒。

1.  接下来，我们需要创建我们的玩家。让我们创建一个类来表示他们如下：

```py
class Racer:

    def __init__(self, canvas, color):
        self.canvas = canvas
        self.name = "{} player".format(color.title())
        size = 50
        self.id = canvas.create_oval(
            (canvas.left, canvas.center_y),
            (canvas.left + size, canvas.center_y + size),
            fill=color)
```

`Racer`类将使用对`canvas`的引用和一个`color`字符串创建，并从中派生其颜色和名称。我们将最初在屏幕的中间左侧绘制赛车，并使其大小为`50`像素。最后，我们将其项目 ID 字符串的引用保存在`self.id`中。

1.  现在，在`App.setup()`中，我们将通过执行以下代码创建两个赛车：

```py
               self.racers = [
                   Racer(self.canvas, 'red'),
                   Racer(self.canvas, 'green')]
```

1.  到目前为止，我们游戏中的所有对象都已设置好。运行程序，你应该能看到右侧的黄色点线终点线和左侧的绿色圆圈（红色圆圈将被隐藏在绿色下面）。

# 动画赛车

为了使我们的赛车动画化，我们将使用`Canvas.move()`方法。`move()`接受一个项目 ID，一定数量的`x`像素和一定数量的`y`像素，并将项目移动该数量。通过使用`random.randint()`和一些简单的逻辑，我们可以生成一系列移动，将每个赛车发送到一条蜿蜒的路径朝着终点线。

一个简单的实现可能如下所示：

```py
def move_racer(self):
    x = randint(0, 100)
    y = randint(-50, 50)
    t = randint(500, 2000)
    self.canvas.after(t, self.canvas.move, self.id, x, y)
    if self.canvas.bbox(self.id)[0] < self.canvas.right:
        self.canvas.after(t, self.move_racer)
```

然而，这并不是我们真正想要的；问题在于`move()`是瞬间发生的，导致错误跳跃到屏幕的另一侧；我们希望我们的移动在一段时间内平稳进行。

为了实现这一点，我们将采取以下方法：

1.  计算一系列线性移动，每个移动都有一个随机的增量`x`，增量`y`和`时间`，可以到达终点线

1.  将每个移动分解为由时间分成的一定间隔的步骤

1.  将每个移动的每一步添加到队列中

1.  在我们的常规间隔中，从队列中提取下一步并传递给`move()`

让我们首先定义我们的帧间隔并创建我们的动画队列：

```py
from queue import Queue
...
class Racer:
    FRAME_RES = 50

    def __init__(...):
        ...
        self.animation_queue = Queue()
```

`FRAME_RES`（帧分辨率的缩写）定义了每个`Canvas.move()`调用之间的毫秒数。`50`毫秒给我们 20 帧每秒，应该足够平滑移动。

现在创建一个方法来绘制到终点线的路径：

```py
    def plot_course(self):
        start_x = self.canvas.left
        start_y = self.canvas.center_y
        total_dx, total_dy = (0, 0)

        while start_x + total_dx < self.canvas.right:
            dx = randint(0, 100)
            dy = randint(-50, 50)
            target_y = start_y + total_dy + dy
            if not (self.canvas.top < target_y < self.canvas.bottom):
                dy = -dy
            time = randint(500, 2000)
            self.queue_move(dx, dy, time)
            total_dx += dx
            total_dy += dy
```

这个方法通过生成随机的`x`和`y`移动，从`canvas`的左中心绘制一条到右侧的路径，直到总`x`大于`canvas`对象的宽度。`x`的变化总是正的，使我们的错误向着终点线移动，但`y`的变化可以是正的也可以是负的。为了保持我们的错误在屏幕上，我们通过否定任何会使玩家超出画布顶部或底部边界的`y`变化来限制总的`y`移动。

除了`dx`和`dy`，我们还生成了移动所需的随机`time`数量，介于半秒和两秒之间，并将生成的值发送到`queue_move()`方法。

`queue_move()`命令将需要将大移动分解为描述在一个`FRAME_RES`间隔中应该发生多少移动的单个帧。为此，我们需要一个**partition 函数**：一个数学函数，将整数`n`分解为大致相等的整数`k`。例如，如果我们想将-10 分成四部分，我们的函数应返回一个类似于[-3, -3, -2, -2]的列表。

将`partition()`创建为`Racer`的静态方法：

```py
    @staticmethod
    def partition(n, k):
        """Return a list of k integers that sum to n"""
        if n == 0:
            return [0] * k
```

我们从简单的情况开始：当`n`为`0`时，返回一个由`k`个零组成的列表。

代码的其余部分如下所示：

```py
        base_step = int(n / k)
        parts = [base_step] * k
        for i in range(n % k):
                parts[i] += n / abs(n)
        return parts
```

首先，我们创建一个长度为`k`的列表，由`base_step`组成，即`n`除以`k`的整数部分。我们在这里使用`int()`的转换而不是地板除法，因为它在负数时表现更合适。接下来，我们需要尽可能均匀地在列表中分配余数。为了实现这一点，我们在部分列表的前`n % k`项中添加`1`或`-1`（取决于余数的符号）。

使用我们的例子`n = -10`和`k = 4`，按照这里的数学：

+   -10 / 4 = -2.5，截断为-2。

+   所以我们有一个列表：[-2, -2, -2, -2]。

+   -10 % 4 = 2，所以我们在列表的前两个项目中添加-1（即-10 / 10）。

+   我们得到了一个答案：[-3, -3, -2, -2]。完美！

现在我们可以编写`queue_move()`：

```py
    def queue_move(self, dx, dy, time):
        num_steps = time // self.FRAME_RES
        steps = zip(
            self.partition(dx, num_steps),
            self.partition(dy, num_steps))

        for step in steps:
            self.animation_queue.put(step)
```

我们首先通过使用地板除法将时间除以`FRAME_RES`来确定此移动中的步数。我们通过将`dx`和`dy`分别传递给我们的`partition()`方法来创建`x`移动列表和`y`移动列表。这两个列表与`zip`结合形成一个`(dx, dy)`对的单个列表，然后添加到动画队列中。

为了使动画真正发生，我们将编写一个`animate()`方法：

```py
    def animate(self):
        if not self.animation_queue.empty():
            nextmove = self.animation_queue.get()
            self.canvas.move(self.id, *nextmove)
        self.canvas.after(self.FRAME_RES, self.animate)
```

`animate()`方法检查队列是否有移动。如果有，将调用`canvas.move()`，并传递赛车的 ID 和需要进行的移动。最后，`animate()`方法被安排在`FRAME_RES`毫秒后再次运行。

动画赛车的最后一步是在`__init__()`的末尾调用`self.plot_course()`和`self.animate()`。如果现在运行游戏，你的两个点应该从左到右在屏幕上漫游。但目前还没有人获胜！

# 检测和处理获胜条件

为了检测获胜条件，我们将定期检查赛车是否与终点线项目重叠。当其中一个重叠时，我们将宣布它为获胜者，并提供再玩一次的选项。

物品之间的碰撞检测在 Tkinter 的 Canvas 小部件中有些尴尬。我们必须将一组边界框坐标传递给`find_overlapping()`，它会返回与边界框重叠的项目标识的元组。

让我们为我们的`Racer`类创建一个`overlapping()`方法：

```py
    def overlapping(self):
        bbox = self.canvas.bbox(self.id)
        overlappers = self.canvas.find_overlapping(*bbox)
        return [x for x in overlappers if x!=self.id]
```

这个方法使用画布的`bbox()`方法检索`Racer`项目的边界框。然后使用`find_overlapping()`获取与此边界框重叠的项目的元组。接下来，我们将过滤此元组，以删除`Racer`项目的 ID，有效地返回与`Racer`类重叠的项目列表。

回到我们的`App()`方法，我们将创建一个`check_for_winner()`方法：

```py
    def check_for_winner(self):
        for racer in self.racers:
            if self.finish_line in racer.overlapping():
                self.declare_winner(racer)
                return
        self.after(Racer.FRAME_RES, self.check_for_winner)
```

这个方法迭代我们的赛车列表，并检查赛车的`overlapping()`方法返回的列表中是否有`finish_line` ID。如果有，`racer`就到达了终点线，并将被宣布为获胜者。

如果没有宣布获胜者，我们将在`Racer.FRAME_RES`毫秒后再次安排检查运行。

我们在`declare_winner()`方法中处理获胜条件：

```py
    def declare_winner(self, racer):
        wintext = self.canvas.create_text(
            (self.canvas.center_x, self.canvas.center_y),
            text='{} wins!\nClick to play again.'.format(racer.name),
            fill='white',
            font='TkDefaultFont 32',
            activefill='violet')
        self.canvas.tag_bind(wintext, '<Button-1>', self.reset)
```

在这个方法中，我们刚刚创建了一个`text`项目，在`canvas`的中心声明`racer.name`为获胜者。`activefill`参数使颜色在鼠标悬停在其上时变为紫色，向用户指示此文本是可点击的。

当点击该文本时，它调用`reset()`方法：

```py
    def reset(self, *args):
        for item in self.canvas.find_all():
            self.canvas.delete(item)
        self.setup()
```

`reset()`方法需要清除画布，因此它使用`find_all()`方法检索所有项目标识符的列表，然后对每个项目调用`delete()`。最后，我们调用`setup()`来重置游戏。

如您在下面的截图中所见，游戏现在已经完成：

![](img/5199652b-2987-472d-b74e-4d00c79ddc46.png)

虽然不是很简单，但 Tkinter 中的动画可以通过一些仔细的规划和一点数学来提供流畅和令人满意的结果。

不过，够玩游戏了；让我们回到实验室，看看如何使用 Tkinter 的`Canvas`小部件来可视化数据。

# 在画布上创建简单的图表

我们想要生成的第一个图形是一个简单的折线图，显示我们植物随时间的生长情况。每个实验室的气候条件各不相同，我们想要看到这些条件如何影响所有植物的生长，因此图表将显示每个实验室的一条线，显示实验期间实验室中所有地块的中位高度测量的平均值。

我们将首先创建一个模型方法来返回原始数据，然后创建一个基于`Canvas`的折线图视图，最后创建一个应用程序回调来获取数据并将其发送到图表视图。

# 创建模型方法

假设我们有一个 SQL 查询，通过从`plot_checks`表中的最旧日期中减去其日期来确定地块检查的天数，然后在给定实验室和给定日期上拉取`lab_id`和所有植物的`median_height`的平均值。

我们将在一个名为`get_growth_by_lab()`的新`SQLModel`方法中运行此查询：

```py
    def get_growth_by_lab(self):
        query = (
            'SELECT date - (SELECT min(date) FROM plot_checks) AS day, '
            'lab_id, avg(median_height) AS avg_height FROM plot_checks '
            'GROUP BY date, lab_id ORDER BY day, lab_id;')
        return self.query(query)
```

我们将得到一个数据表，看起来像这样：

| **Day** | **Lab ID** | **Average height** |
| 0 | A | 7.4198750000000000 |
| 0 | B | 7.3320000000000000 |
| 0 | C | 7.5377500000000000 |
| 0 | D | 8.4633750000000000 |
| 0 | E | 7.8530000000000000 |
| 1 | A | 6.7266250000000000 |
| 1 | B | 6.8503750000000000 |  |

我们将使用这些数据来构建我们的图表。

# 创建图形视图

转到`views.py`，在那里我们将创建`LineChartView`类：

```py
class LineChartView(tk.Canvas):

    margin = 20

    def __init__(self, parent, chart_width, chart_height,
                 x_axis, y_axis, x_max, y_max):
        self.max_x = max_x
        self.max_y = max_y
        self.chart_width = chart_width
        self.chart_height = chart_height
```

`LineChartView`是`Canvas`的子类，因此我们将能够直接在其上绘制项目。我们将接受父小部件、图表部分的高度和宽度、`x`和`y`轴的标签作为参数，并显示`x`和`y`的最大值。我们将保存图表的尺寸和最大值以供以后使用，并将边距宽度设置为 20 像素的类属性。

让我们开始设置这个`Canvas`：

```py
        view_width = chart_width + 2 * self.margin
        view_height = chart_height + 2 * self.margin
        super().__init__(
            parent, width=view_width,
            height=view_height, background='lightgrey')
```

通过将边距添加到两侧来计算视图的`width`和`height`值，然后使用它们调用超类`__init__()`，同时将背景设置为`lightgrey`。我们还将保存图表的`width`和`height`作为实例属性。

接下来，让我们绘制轴：

```py
        self.origin = (self.margin, view_height - self.margin)
        self.create_line(
            self.origin, (self.margin, self.margin), width=2)
        self.create_line(
            self.origin,
            (view_width - self.margin,
             view_height - self.margin))
```

我们的图表原点将距离左下角`self.margin`像素，并且我们将绘制`x`和`y`轴，作为简单的黑色线条从原点向左和向上延伸到图表的边缘。

接下来，我们将标记轴：

```py
        self.create_text(
            (view_width // 2, view_height - self.margin),
            text=x_axis, anchor='n')
        # angle requires tkinter 8.6 -- macOS users take note!
        self.create_text(
            (self.margin, view_height // 2),
            text=y_axis, angle=90, anchor='s')
```

在这里，我们创建了设置为`x`和`y`轴标签的`text`项目。这里使用了一些新的参数：`anchor`设置文本边界框的哪一侧与提供的坐标相连，`angle`将文本对象旋转给定的角度。请注意，`angle`是 Tkinter 8.6 的一个特性，因此对于 macOS 用户可能会有问题。另外，请注意，我们将旋转的文本的`anchor`设置为 south；即使它被旋转，基本方向仍然指的是未旋转的边，因此 south 始终是文本的底部，就像正常打印的那样。

最后，我们需要创建一个包含实际图表的第二个`Canvas`：

```py
        self.chart = tk.Canvas(
            self, width=chart_width, height=chart_height,
            background='white')
        self.create_window(
            self.origin, window=self.chart, anchor='sw')
```

虽然我们可以使用`pack()`或`grid()`等几何管理器在`canvas`上放置小部件，但`create_window()`方法将小部件作为`Canvas`项目放置在`Canvas`上，使用坐标。我们将图表的左下角锚定到我们图表的原点。

随着这些部分的就位，我们现在将创建一个在图表上绘制数据的方法：

```py
    def plot_line(self, data, color):
        x_scale = self.chart_width / self.max_x
        y_scale = self.chart_height / self.max_y

        coords = [(round(x * x_scale),
            self.chart_height - round(y * y_scale))
            for x, y in data]

        self.chart.create_line(*coords, width=2, fill=color)
```

在`plot_line()`中，我们首先必须将原始数据转换为可以绘制的坐标。我们需要缩放我们的`数据`点，使它们的范围从图表对象的高度和宽度为`0`。我们的方法通过将图表尺寸除以`x`和`y`的最大值来计算`x`和`y`的比例（即每个单位`x`或`y`有多少像素）。然后我们可以通过使用列表推导将每个数据点乘以比例值来转换我们的数据。

此外，数据通常是以左下角为原点绘制的，但坐标是从左上角开始测量的，因此我们需要翻转`y`坐标；这也是我们的列表推导中所做的，通过从图表高度中减去新的`y`值来完成。现在可以将这些坐标传递给`create_line()`，并与合理的`宽度`和调用者传入的`颜色`参数一起传递。

我们需要的最后一件事是一个**图例**，告诉用户图表上的每种颜色代表什么。没有图例，这个图表将毫无意义。

让我们创建一个`draw_legend()`方法：

```py
    def draw_legend(self, mapping):
        y = self.margin
        x = round(self.margin * 1.5) + self.chart_width
        for label, color in mapping.items():
              self.create_text((x, y), text=label, fill=color, 
              anchor='w')
              y += 20
```

我们的方法接受一个将标签映射到颜色的字典，这将由应用程序提供。对于每一个，我们只需绘制一个包含`标签`文本和相关`填充`颜色的文本项。由于我们知道我们的标签会很短（只有一个字符），我们可以只把它放在边缘。

# 更新应用程序

在`Application`类中，创建一个新方法来显示我们的图表：

```py
    def show_growth_chart(self):
        data = self.data_model.get_growth_by_lab()
        max_x = max([x['day'] for x in data])
        max_y = max([x['avg_height'] for x in data])
```

首要任务是从我们的`get_growth_by_lab()`方法中获取数据，并计算`x`和`y`轴的最大值。我们通过使用列表推导将值提取到列表中，并在其上调用内置的`max()`函数来完成这一点。

接下来，我们将构建一个小部件来容纳我们的`LineChartView`对象：

```py
        popup = tk.Toplevel()
        chart = v.LineChartView(popup, 600, 300, 'day',
                                'centimeters', max_x, max_y)
        chart.pack(fill='both', expand=1)
```

在这种情况下，我们使用`Toplevel`小部件，它在我们的主应用程序窗口之外创建一个新窗口。然后我们创建了`LineChartView`，它是`600`乘`300`像素，带有*x*轴和*y*轴标签，并将其添加到`Toplevel`中使用`pack()`。

接下来，我们将为每个实验室分配颜色并绘制`图例`：

```py
        legend = {'A': 'green', 'B': 'blue', 'C': 'cyan',
                  'D': 'yellow', 'E': 'purple'}
        chart.draw_legend(legend)
```

最后要做的是绘制实际的线：

```py
        for lab, color in legend.items():
            dataxy = [(x['day'], x['avg_height'])
                for x in data
                if x['lab_id'] == lab]
            chart.plot_line(dataxy, color)
```

请记住，我们的数据包含所有实验室的值，因此我们正在`图例`中迭代实验室，并使用列表推导来提取该实验室的数据。然后我们的`plot_line()`方法完成其余工作。

完成此方法后，将其添加到`callbacks`字典中，并为每个平台的工具菜单添加一个菜单项。

当您调用您的函数时，您应该看到类似这样的东西：

![](img/db64ecf3-2169-4ea1-ac5c-954a960ba237.png)没有一些示例数据，图表看起来不会很好。除非您只是喜欢进行数据输入，否则在`sql`目录中有一个加载示例数据的脚本。

# 使用 Matplotlib 和 Tkinter 创建高级图表

我们的折线图很漂亮，但要使其完全功能，仍需要相当多的工作：它缺少比例、网格线和其他功能，这些功能将使它成为一个完全有用的图表。

我们可以花很多时间使它更完整，但在我们的 Tkinter 应用程序中获得更令人满意的图表和图形的更快方法是**Matplotlib**。

Matplotlib 是一个第三方库，用于生成各种类型的专业质量、交互式图表。这是一个庞大的库，有许多附加组件，我们不会涵盖其实际用法的大部分内容，但我们应该看一下如何将 Matplotlib 集成到 Tkinter 应用程序中。为此，我们将创建一个气泡图，显示每个地块的产量与`湿度`和`温度`的关系。

您应该能够使用`pip install --user matplotlib`命令使用`pip`安装`matplotlib`。有关安装的完整说明，请参阅[`matplotlib.org/users/installing.html.`](https://matplotlib.org/users/installing.html)

# 数据模型方法

在我们制作图表之前，我们需要一个`SQLModel`方法来提取数据：

```py
    def get_yield_by_plot(self):
        query = (
            'SELECT lab_id, plot, seed_sample, MAX(fruit) AS yield, '
            'AVG(humidity) AS avg_humidity, '
            'AVG(temperature) AS avg_temperature '
            'FROM plot_checks WHERE NOT equipment_fault '
            'GROUP BY lab_id, plot, seed_sample')
        return self.query(query)
```

此图表的目的是找到每个种子样本的`温度`和`湿度`的最佳点。因此，我们需要每个`plot`的一行，其中包括最大的`fruit`测量值，`plot`列处的平均湿度和温度，以及`seed_sample`。由于我们不想要任何错误的数据，我们将过滤掉具有`Equipment` `Fault`的行。

# 创建气泡图表视图

要将 MatplotLib 集成到 Tkinter 应用程序中，我们需要进行几次导入。

第一个是`matplotlib`本身：

```py
import matplotlib
matplotlib.use('TkAgg')
```

在“导入”部分运行代码可能看起来很奇怪，甚至您的编辑器可能会对此进行投诉。但在我们从`matplotlib`导入任何其他内容之前，我们需要告诉它应该使用哪个后端。在这种情况下，我们想要使用`TkAgg`后端，这是专为集成到 Tkinter 中而设计的。

现在我们可以从`matplotlib`中再引入一些内容：

```py
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2TkAgg)
```

`Figure`类表示`matplotlib`图表可以绘制的基本绘图区域。`FigureCanvasTkAgg`类是`Figure`和 Tkinter`Canvas`之间的接口，`NavigationToolbar2TkAgg`允许我们在图表上放置一个预制的`Figure`工具栏。

为了看看这些如何配合，让我们在`views.py`中启动我们的`YieldChartView`类：

```py
class YieldChartView(tk.Frame):
    def __init__(self, parent, x_axis, y_axis, title):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
```

在调用`super().__init__()`创建`Frame`对象之后，我们创建一个`Figure`对象来保存我们的图表。`Figure`对象不是以像素为单位的大小，而是以**英寸**和**每英寸点数**（**dpi**）设置为单位（在这种情况下，得到的是一个 600x400 像素的`Figure`）。接下来，我们创建一个`FigureCanvasTkAgg`对象，将我们的`Figure`对象与 Tkinter`Canvas`连接起来。`FigureCanvasTkAgg`对象本身不是`Canvas`对象或子类，但它有一个`Canvas`对象，我们可以将其放置在我们的应用程序中。

接下来，我们将工具栏和`pack()`添加到我们的`FigureCanvasTkAgg`对象中：

```py
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
```

我们的工具栏被传递给了我们的`FigureCanvasTkAgg`对象和根窗口（在这种情况下是`self`），将其附加到我们的图表和它的画布上。要将`FigureCanvasTkAgg`对象放在我们的`Frame`对象上，我们需要调用`get_tk_widget()`来检索其 Tkinter`Canvas`小部件，然后我们可以使用`pack()`和`grid()`按需要对其进行打包或网格化。

下一步是设置轴：

```py
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.axes.set_xlabel(x_axis)
        self.axes.set_ylabel(y_axis)
        self.axes.set_title(title)
```

在 Matplotlib 中，`axes`对象表示可以在其上绘制数据的单个`x`和`y`轴集，使用`add_subplot()`方法创建。传递给`add_subplot()`的三个整数建立了这是一个子图中一行中的第一个`axes`集。我们的图表可能包含多个以表格形式排列的子图，但我们只需要一个。创建后，我们设置`axes`对象上的标签。

要创建气泡图表，我们将使用 Matplotlib 的**散点图**功能，但使用每个点的大小来指示水果产量。我们还将对点进行颜色编码以指示种子样本。

让我们实现一个绘制散点图的方法：

```py
    def draw_scatter(self, data, color, label):
        x, y, s = zip(*data)
        s = [(x ** 2)//2 for x in s]
        scatter = self.axes.scatter(
            x, y, s, c=color, label=label, alpha=0.5)
```

传入的数据应该包含每条记录的三列，并且我们将这些分解为包含`x`、`y`和`size`值的三个单独的列表。接下来，我们将放大大小值之间的差异，使它们更加明显，方法是将每个值平方然后除以一半。这并不是绝对必要的，但在差异相对较小时，它有助于使图表更易读。

最后，我们通过调用`scatter()`将数据绘制到`axes`对象上，同时传递`color`和`label`值给点，并使用`alpha`参数使它们半透明。

`zip(*data)`是一个 Python 习语，用于将 n 长度元组的列表分解为值的 n 个列表，本质上是`zip(x, y, s)`的反向操作。

为了为我们的`axes`对象绘制图例，我们需要两样东西：我们的`scatter`对象的列表和它们的标签列表。为了获得这些，我们将不得不在`__init__()`中创建一些空列表，并在每次调用`draw_scatter()`时进行追加。

在`__init__()`中，添加一些空列表：

```py
        self.scatters = []
        self.scatter_labels = []
```

现在，在`draw_scatter()`的末尾，追加列表并更新`legend()`方法：

```py
        self.scatters.append(scatter)
        self.scatter_labels.append(label)
        self.axes.legend(self.scatters, self.scatter_labels)
```

我们可以反复调用`legend()`，它会简单地销毁并重新绘制图例。

# 应用程序方法

回到`Application`，让我们创建一个显示产量数据的方法。

首先创建一个`Toplevel`方法并添加我们的图表视图：

```py
        popup = tk.Toplevel()
        chart = v.YieldChartView(popup,
            'Average plot humidity', 'Average Plot temperature',
            'Yield as a product of humidity and temperature')
        chart.pack(fill='both', expand=True)
```

现在让我们为我们的散点图设置数据：

```py
        data = self.data_model.get_yield_by_plot()
        seed_colors = {'AXM477': 'red', 'AXM478': 'yellow',
            'AXM479': 'green', 'AXM480': 'blue'}
```

我们从数据模型中检索了产量`data`，并创建了一个将保存我们想要为每个种子样本使用的颜色的字典。

现在我们只需要遍历种子样本并绘制散点图：

```py
        for seed, color in seed_colors.items():
            seed_data = [
                (x['avg_humidity'], x['avg_temperature'], x['yield'])
                for x in data if x['seed_sample'] == seed]
            chart.draw_dots(seed_data, color, seed)
```

再次，我们使用列表推导式格式化和过滤我们的数据，为`x`提供平均湿度，为`y`提供平均温度，为`s`提供产量。

将该方法添加到`callbacks`字典中，并在生长图选项下方创建一个菜单项。

您的气泡图应该看起来像这样：

![](img/335e2505-6f35-4708-aaa0-5a5e1948112e.png)

请利用导航工具栏玩一下这个图表，注意你可以缩放和平移，调整图表的大小，并保存图像。这些都是 Matplotlib 自动提供的强大工具。

# 总结

在本章中，您了解了 Tkinter 的图形能力。您学会了如何在 Tkinter 的`Canvas`小部件上绘制和动画图形，以及如何利用这些能力来可视化数据。您还学会了如何将 Matplotlib 图形集成到您的应用程序中，并通过将 SQL 查询连接到我们的图表视图，在我们的应用程序中实现了两个图表。
