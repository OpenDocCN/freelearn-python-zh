# 窗口布局

在本章中，我们将介绍以下食谱：

+   使用框架对小部件进行分组

+   使用Pack几何管理器

+   使用Grid几何管理器

+   使用Place几何管理器

+   使用FrameLabel小部件对输入进行分组

+   动态布置小部件

+   创建水平和垂直滚动条

# 介绍

小部件确定用户可以在GUI应用程序中执行的操作；但是，我们应该注意它们的放置和我们与该安排建立的关系。有效的布局帮助用户识别每个图形元素的含义和优先级，以便他们可以快速理解如何与我们的程序交互。

布局还确定了用户期望在整个应用程序中一致找到的视觉外观，例如始终将确认按钮放在屏幕右下角。尽管这些信息对我们作为开发人员来说可能是显而易见的，但如果我们不按照自然顺序引导他们通过应用程序，最终用户可能会感到不知所措。

本章将深入探讨Tkinter提供的不同机制，用于布置和分组小部件以及控制其他属性，例如它们的大小或间距。

# 使用框架对小部件进行分组

框架表示窗口的矩形区域，通常用于复杂布局以包含其他小部件。由于它们有自己的填充、边框和背景，您可以注意到小部件组在逻辑上是相关的。

框架的另一个常见模式是封装应用程序功能的一部分，以便您可以创建一个抽象，隐藏子部件的实现细节。

我们将看到一个示例，涵盖了从`Frame`类继承并公开包含小部件上的某些信息的组件的两种情况。

# 准备就绪

我们将构建一个应用程序，其中包含两个列表，第一个列表中有一系列项目，第二个列表最初为空。两个列表都是可滚动的，并且您可以使用两个中央按钮在它们之间移动项目：

![](images/4c30ba49-f25a-48ca-85ed-5533ffb88ce7.png)

# 如何做…

我们将定义一个`Frame`子类来表示可滚动列表，然后创建该类的两个实例。两个按钮也将直接添加到主窗口：

```py
import tkinter as tk

class ListFrame(tk.Frame):
    def __init__(self, master, items=[]):
        super().__init__(master)
        self.list = tk.Listbox(self)
        self.scroll = tk.Scrollbar(self, orient=tk.VERTICAL,
                                   command=self.list.yview)
        self.list.config(yscrollcommand=self.scroll.set)
        self.list.insert(0, *items)
        self.list.pack(side=tk.LEFT)
        self.scroll.pack(side=tk.LEFT, fill=tk.Y)

    def pop_selection(self):
        index = self.list.curselection()
        if index:
            value = self.list.get(index)
            self.list.delete(index)
            return value

    def insert_item(self, item):
        self.list.insert(tk.END, item)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        months = ["January", "February", "March", "April",
                  "May", "June", "July", "August", "September",
                  "October", "November", "December"]
        self.frame_a = ListFrame(self, months)
        self.frame_b = ListFrame(self)
        self.btn_right = tk.Button(self, text=">",
                                   command=self.move_right)
        self.btn_left = tk.Button(self, text="<",
                                  command=self.move_left)

        self.frame_a.pack(side=tk.LEFT, padx=10, pady=10)
        self.frame_b.pack(side=tk.RIGHT, padx=10, pady=10)
        self.btn_right.pack(expand=True, ipadx=5)
        self.btn_left.pack(expand=True, ipadx=5)

    def move_right(self):
        self.move(self.frame_a, self.frame_b)

    def move_left(self):
        self.move(self.frame_b, self.frame_a)

    def move(self, frame_from, frame_to):
        value = frame_from.pop_selection()
        if value:
            frame_to.insert_item(value)

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

# 工作原理…

我们的`ListFrame`类只有两种方法与内部列表进行交互：`pop_selection()`和`insert_item()`。第一个返回并删除当前选择的项目，如果没有选择项目，则返回None，而第二个在列表末尾插入新项目。

这些方法用于父类中将项目从一个列表转移到另一个列表：

```py
def move(self, frame_from, frame_to):
    value = frame_from.pop_selection()
    if value:
        frame_to.insert_item(value)
```

我们还利用父框架容器正确地打包它们，以适当的填充：

```py
# ...
self.frame_a.pack(side=tk.LEFT, padx=10, pady=10) self.frame_b.pack(side=tk.RIGHT, padx=10, pady=10)
```

由于这些框架，我们对几何管理器的调用在全局布局中更加隔离和有组织。

# 还有更多...

这种方法的另一个好处是，它允许我们在每个容器小部件中使用不同的几何管理器，例如在框架内使用`grid()`来布置小部件，在主窗口中使用`pack()`来布置框架。

但是，请记住，在Tkinter中不允许在同一个容器中混合使用这些几何管理器，否则会使您的应用程序崩溃。

# 另请参阅

+   *使用Pack几何管理器*食谱

# 使用Pack几何管理器

在之前的食谱中，我们已经看到创建小部件并不会自动在屏幕上显示它。我们调用了每个小部件上的`pack()`方法来实现这一点，这意味着我们使用了Pack几何管理器。

这是Tkinter中三种可用的几何管理器之一，非常适合简单的布局，例如当您想要将所有小部件放在彼此上方或并排时。

# 准备就绪

假设我们想在应用程序中实现以下布局：

![](images/fce828e9-1f75-4590-a50e-bb8adbc1d8eb.png)

它由三行组成，最后一行有三个小部件并排放置。在这种情况下，Pack布局管理器可以轻松地按预期添加小部件，而无需额外的框架。

# 操作步骤

我们将使用五个具有不同文本和背景颜色的`Label`小部件来帮助我们识别每个矩形区域：

```py
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        label_a = tk.Label(self, text="Label A", bg="yellow")
        label_b = tk.Label(self, text="Label B", bg="orange")
        label_c = tk.Label(self, text="Label C", bg="red")
        label_d = tk.Label(self, text="Label D", bg="green")
        label_e = tk.Label(self, text="Label E", bg="blue")

        opts = { 'ipadx': 10, 'ipady': 10, 'fill': tk.BOTH }
        label_a.pack(side=tk.TOP, **opts)
        label_b.pack(side=tk.TOP, **opts)
        label_c.pack(side=tk.LEFT, **opts)
        label_d.pack(side=tk.LEFT, **opts)
        label_e.pack(side=tk.LEFT, **opts)

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

我们还向`opts`字典中添加了一些选项，以便清楚地确定每个区域的大小：

![](images/1a0e72a4-090a-4af8-af83-c8574e730056.png)

# 工作原理

为了更好地理解Pack布局管理器，我们将逐步解释它如何将小部件添加到父容器中。在这里，我们特别关注`side`选项的值，它指示小部件相对于下一个将被打包的小部件的位置。

首先，我们将两个标签打包到屏幕顶部。虽然`tk.TOP`常量是`side`选项的默认值，但我们明确设置它以清楚地区分它与我们使用`tk.LEFT`值的调用。

![](images/77b7063f-2c0b-4a2b-ab75-c1704861201d.jpg)

然后，我们使用`side`选项设置为`tk.LEFT`来打包下面的三个标签，这会使它们并排放置：

![](images/0f54aec2-5957-4da5-9e32-d63acc06903f.jpg)

指定`label_e`上的side实际上并不重要，只要它是我们添加到容器中的最后一个小部件即可。

请记住，这就是在使用Pack布局管理器时顺序如此重要的原因。为了防止复杂布局中出现意外结果，通常将小部件与框架分组，这样当您将所有小部件打包到一个框架中时，就不会干扰其他小部件的排列。

在这些情况下，我们强烈建议您使用网格布局管理器，因为它允许您直接调用几何管理器设置每个小部件的位置，并且避免了额外框架的需要。

# 还有更多...

除了`tk.TOP`和`tk.LEFT`，您还可以将`tk.BOTTOM`和`tk.RIGHT`常量传递给`side`选项。它们执行相反的堆叠，正如它们的名称所暗示的那样；但是，这可能是反直觉的，因为我们遵循的自然顺序是从上到下，从左到右。

例如，如果我们在最后三个小部件中用`tk.RIGHT`替换`tk.LEFT`的值，它们从左到右的顺序将是`label_e`，`label_d`和`label_c`。

# 参见

+   *使用网格布局管理器*食谱

+   *使用Place布局管理器*食谱

# 使用网格布局管理器

网格布局管理器被认为是三种布局管理器中最通用的。它直接重新组合了通常用于用户界面设计的*网格*概念，即一个二维表格，分为行和列，其中每个单元格代表小部件的可用空间。

# 准备工作

我们将演示如何使用网格布局管理器来实现以下布局：

![](images/2f7823b8-c9f3-4408-8b62-eccfb7ab446d.png)

这可以表示为一个3 x 3的表格，其中第二列和第三列的小部件跨越两行，底部行的小部件跨越三列。

# 操作步骤

与前面的食谱一样，我们将使用五个具有不同背景的标签来说明单元格的分布：

```py
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        label_a = tk.Label(self, text="Label A", bg="yellow")
        label_b = tk.Label(self, text="Label B", bg="orange")
        label_c = tk.Label(self, text="Label C", bg="red")
        label_d = tk.Label(self, text="Label D", bg="green")
        label_e = tk.Label(self, text="Label E", bg="blue")

        opts = { 'ipadx': 10, 'ipady': 10 , 'sticky': 'nswe' }
        label_a.grid(row=0, column=0, **opts)
        label_b.grid(row=1, column=0, **opts)
        label_c.grid(row=0, column=1, rowspan=2, **opts)
        label_d.grid(row=0, column=2, rowspan=2, **opts)
        label_e.grid(row=2, column=0, columnspan=3, **opts)

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

我们还传递了一个选项字典，以添加一些内部填充并将小部件扩展到单元格中的所有可用空间。

# 工作原理

`label_a`和`label_b`的放置几乎是不言自明的：它们分别占据第一列的第一行和第二行，记住网格位置是从零开始计数的：

![](images/956d78d6-3cda-41e4-949e-45489d30fdda.png)

为了扩展`label_c`和`label_d`跨越多个单元格，我们将把`rowspan`选项设置为`2`，这样它们将跨越两个单元格，从`row`和`column`选项指示的位置开始。最后，我们将使用`columnspan`选项将`label_e`放置到`3`。

需要强调的是，与Pack几何管理器相比，可以更改对每个小部件的`grid()`调用的顺序，而不修改最终布局。

# 还有更多...

`sticky`选项表示小部件应粘附的边界，用基本方向表示：北、南、西和东。这些值由Tkinter常量`tk.N`、`tk.S`、`tk.W`和`tk.E`表示，以及组合版本`tk.NW`、`tk.NE`、`tk.SW`和`tk.SE`。

例如，`sticky=tk.N`将小部件对齐到单元格的顶部边界（北），而`sticky=tk.SE`将小部件放置在单元格的右下角（东南）。

由于这些常量代表它们对应的小写字母，我们用`"nswe"`字符串简写了`tk.N + tk.S + tk.W + tk.E`表达式。这意味着小部件应该在水平和垂直方向上都扩展，类似于Pack几何管理器的`fill=tk.BOTH`选项。

如果`sticky`选项没有传递值，则小部件将在单元格内居中。

# 另请参阅

+   *使用Pack几何管理器*配方

+   *使用Place几何管理器*配方

# 使用Place几何管理器

Place几何管理器允许您以绝对或相对于另一个小部件的位置和大小。

在三种几何管理器中，它是最不常用的一种。另一方面，它可以适应一些复杂的情况，例如您想自由定位一个小部件或重叠一个先前放置的小部件。

# 准备工作

为了演示如何使用Place几何管理器，我们将通过混合绝对位置和相对位置和大小来复制以下布局：

![](images/78190289-c61d-422c-8422-84a24b8a0d78.png)

# 如何做...

我们将显示的标签具有不同的背景，并按从左到右和从上到下的顺序定义：

```py
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        label_a = tk.Label(self, text="Label A", bg="yellow")
        label_b = tk.Label(self, text="Label B", bg="orange")
        label_c = tk.Label(self, text="Label C", bg="red")
        label_d = tk.Label(self, text="Label D", bg="green")
        label_e = tk.Label(self, text="Label E", bg="blue")

        label_a.place(relwidth=0.25, relheight=0.25)
        label_b.place(x=100, anchor=tk.N,
                      width=100, height=50)
        label_c.place(relx=0.5, rely=0.5, anchor=tk.CENTER,
                      relwidth=0.5, relheight=0.5)
        label_d.place(in_=label_c, anchor=tk.N + tk.W,
                      x=2, y=2, relx=0.5, rely=0.5,
                      relwidth=0.5, relheight=0.5)
        label_e.place(x=200, y=200, anchor=tk.S + tk.E,
                      relwidth=0.25, relheight=0.25)

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

如果运行前面的程序，您可以看到`label_c`和`label_d`在屏幕中心的重叠，这是我们使用其他几何管理器没有实现的。

# 它是如何工作的...

第一个标签的`relwidth`和`relheight`选项设置为`0.25`，这意味着它的宽度和高度是其父容器的25%。默认情况下，小部件放置在`x=0`和`y=0`位置，并对齐到西北，即屏幕的左上角。

第二个标签放置在绝对位置`x=100`，并使用`anchor`选项设置为`tk.N`（北）常量与顶部边界对齐。在这里，我们还使用`width`和`height`指定了绝对大小。

第三个标签使用相对定位在窗口中心，并将`anchor`设置为`tk.CENTER`。请记住，`relx`和`relwidth`的值为`0.5`表示父容器宽度的一半，`rely`和`relheight`的值为`0.5`表示父容器高度的一半。

第四个标签通过将其作为`in_`参数放置在`label_c`上（请注意，Tkinter在其后缀中添加了下划线，因为`in`是一个保留关键字）。使用`in_`时，您可能会注意到对齐不是几何上精确的。在我们的示例中，我们必须在每个方向上添加2个像素的偏移量，以完全重叠`label_c`的右下角。

最后，第五个标签使用绝对定位和相对大小。正如您可能已经注意到的那样，这些尺寸可以很容易地切换，因为我们假设父容器为200 x 200像素；但是，如果调整主窗口的大小，只有相对权重才能按预期工作。您可以通过调整窗口大小来测试此行为。

# 还有更多...

Place几何管理器的另一个重要优势是它可以与Pack或Grid一起使用。

例如，假设您希望在右键单击小部件时动态显示标题。您可以使用Label小部件表示此标题，并将其放置在单击小部件的相对位置：

```py
def show_caption(self, event):
    caption = tk.Label(self, ...)
    caption.place(in_=event.widget, x=event.x, y=event.y)
    # ...
```

作为一般建议，我们建议您在Tkinter应用程序中尽可能多地使用其他几何管理器，并且仅在需要自定义定位的专门情况下使用此几何管理器。

# 另请参阅

+   使用Pack几何管理器的食谱

+   使用网格几何管理器的食谱

# 使用LabelFrame小部件对输入进行分组

`LabelFrame`类可用于对多个输入小部件进行分组，指示它们表示的逻辑实体的标签。它通常用于表单，与`Frame`小部件非常相似。

# 准备就绪

我们将构建一个带有一对`LabelFrame`实例的表单，每个实例都有其相应的子输入小部件：

![](images/e9759bcf-5dd1-41ce-9de4-bdb6de6a32e9.png)

# 如何做…

由于此示例的目的是显示最终布局，我们将添加一些小部件，而不将它们的引用保留为属性：

```py
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        group_1 = tk.LabelFrame(self, padx=15, pady=10,
                               text="Personal Information")
        group_1.pack(padx=10, pady=5)

        tk.Label(group_1, text="First name").grid(row=0)
        tk.Label(group_1, text="Last name").grid(row=1)
        tk.Entry(group_1).grid(row=0, column=1, sticky=tk.W)
        tk.Entry(group_1).grid(row=1, column=1, sticky=tk.W)

        group_2 = tk.LabelFrame(self, padx=15, pady=10,
                               text="Address")
        group_2.pack(padx=10, pady=5)

        tk.Label(group_2, text="Street").grid(row=0)
        tk.Label(group_2, text="City").grid(row=1)
        tk.Label(group_2, text="ZIP Code").grid(row=2)
        tk.Entry(group_2).grid(row=0, column=1, sticky=tk.W)
        tk.Entry(group_2).grid(row=1, column=1, sticky=tk.W)
        tk.Entry(group_2, width=8).grid(row=2, column=1,
                                        sticky=tk.W)

        self.btn_submit = tk.Button(self, text="Submit")
        self.btn_submit.pack(padx=10, pady=10, side=tk.RIGHT)

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

# 工作原理…

`LabelFrame`小部件采用`labelwidget`选项来设置用作标签的小部件。如果不存在，它将显示作为`text`选项传递的字符串。例如，可以用以下语句替换`tk.LabelFrame(master, text="Info")`的实例：

```py
label = tk.Label(master, text="Info", ...)
frame = tk.LabelFrame(master, labelwidget=label)
# ...
frame.pack()
```

这将允许您进行任何类型的自定义，例如添加图像。请注意，我们没有为标签使用任何几何管理器，因为当您放置框架时，它会被管理。

# 动态布局小部件

网格几何管理器在简单和高级布局中都很容易使用，也是与小部件列表结合使用的强大机制。

我们将看看如何通过列表推导和`zip`和`enumerate`内置函数，可以减少行数并仅用几行调用几何管理器方法。

# 准备就绪

我们将构建一个应用程序，其中包含四个`Entry`小部件，每个小部件都有相应的标签，指示输入的含义。我们还将添加一个按钮来打印所有条目的值：

![](images/a4586bd3-a7cf-4f0d-9ce6-a538e7114f37.png)

我们将使用小部件列表而不是创建和分配每个小部件到单独的属性。由于我们将在这些列表上进行迭代时跟踪索引，因此我们可以轻松地使用适当的`column`选项调用`grid()`方法。

# 如何做…

我们将使用`zip`函数聚合标签和输入列表。按钮将单独创建和显示，因为它与其余小部件没有共享任何选项：

```py
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        fields = ["First name", "Last name", "Phone", "Email"]
        labels = [tk.Label(self, text=f) for f in fields]
        entries = [tk.Entry(self) for _ in fields]
        self.widgets = list(zip(labels, entries))
        self.submit = tk.Button(self, text="Print info",
                                command=self.print_info)

        for i, (label, entry) in enumerate(self.widgets):
            label.grid(row=i, column=0, padx=10, sticky=tk.W)
            entry.grid(row=i, column=1, padx=10, pady=5)
        self.submit.grid(row=len(fields), column=1, sticky=tk.E,
                         padx=10, pady=10)

    def print_info(self):
        for label, entry in self.widgets:
            print("{} = {}".format(label.cget("text"), "=", entry.get()))

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

您可以在每个输入上输入不同的文本，并单击“打印信息”按钮以验证每个元组包含相应的标签和输入。

# 工作原理…

每个列表推导式都会迭代字段列表的字符串。标签使用每个项目作为显示的文本，输入只需要父容器的引用——下划线是一个常见的习惯用法，表示变量值被忽略。

从Python 3开始，`zip`返回一个迭代器而不是列表，因此我们使用列表函数消耗聚合。结果，`widgets`属性包含一个可以安全多次迭代的元组列表：

```py
fields = ["First name", "Last name", "Phone", "Email"]
labels = [tk.Label(self, text=f) for f in fields]
entries = [tk.Entry(self) for _ in fields]
self.widgets = list(zip(labels, entries))
```

现在，我们必须在每个小部件元组上调用几何管理器。使用`enumerate`函数，我们可以跟踪每次迭代的索引并将其作为*行*号传递：

```py
for i, (label, entry) in enumerate(self.widgets):
    label.grid(row=i, column=0, padx=10, sticky=tk.W)
    entry.grid(row=i, column=1, padx=10, pady=5)
```

请注意，我们使用了`for i, (label, entry) in ...`语法，因为我们必须解压使用`enumerate`生成的元组，然后解压`widgets`属性的每个元组。

在`print_info()`回调中，我们迭代小部件以打印每个标签文本及其相应的输入值。要检索标签的`text`，我们使用了`cget()`方法，它允许您通过名称获取小部件选项的值。

# 创建水平和垂直滚动条

在Tkinter中，几何管理器会占用所有必要的空间，以适应其父容器中的所有小部件。但是，如果容器具有固定大小或超出屏幕大小，将会有一部分区域对用户不可见。

在Tkinter中，滚动条小部件不会自动添加，因此您必须像其他类型的小部件一样创建和布置它们。另一个考虑因素是，只有少数小部件类具有配置选项，使其能够连接到滚动条。

为了解决这个问题，您将学习如何利用**Canvas**小部件的灵活性使任何容器可滚动。

# 准备就绪

为了演示`Canvas`和`Scrollbar`类的组合，创建一个可调整大小和可滚动的框架，我们将构建一个通过加载图像动态更改大小的应用程序。

当单击“加载图像”按钮时，它会将自身移除，并将一个大于可滚动区域的图像加载到`Canvas`中-例如，我们使用了一个预定义的图像，但您可以修改此程序以使用文件对话框选择任何其他GIF图像：

![](images/5ed14b60-2769-43a6-9204-75d6a42f8198.png)

这将启用水平和垂直滚动条，如果主窗口被调整大小，它们会自动调整自己：

![](images/1c73a705-07ce-497d-a353-1cc76c01b56e.png)

# 操作步骤…

当我们将在单独的章节中深入了解Canvas小部件的功能时，本应用程序将介绍其标准滚动界面和`create_window()`方法。请注意，此脚本需要将文件`python.gif`放置在相同的目录中：

```py
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.scroll_x = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.scroll_y = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(self, width=300, height=100,
                                xscrollcommand=self.scroll_x.set,
                                yscrollcommand=self.scroll_y.set)
        self.scroll_x.config(command=self.canvas.xview)
        self.scroll_y.config(command=self.canvas.yview)

        self.frame = tk.Frame(self.canvas)
        self.btn = tk.Button(self.frame, text="Load image",
                             command=self.load_image)
        self.btn.pack()

        self.canvas.create_window((0, 0), window=self.frame,  
                                          anchor=tk.NW)

        self.canvas.grid(row=0, column=0, sticky="nswe")
        self.scroll_x.grid(row=1, column=0, sticky="we")
        self.scroll_y.grid(row=0, column=1, sticky="ns")

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.bind("<Configure>", self.resize)
        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())

    def resize(self, event):
        region = self.canvas.bbox(tk.ALL)
        self.canvas.configure(scrollregion=region)

    def load_image(self):
        self.btn.destroy()
        self.image = tk.PhotoImage(file="python.gif")
        tk.Label(self.frame, image=self.image).pack()

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

# 工作原理…

我们应用程序的第一行创建了滚动条，并使用`xscrollcommand`和`yscrollcommand`选项将它们连接到`Canvas`对象，这些选项分别使用`scroll_x`和`scroll_y`的`set()`方法的引用-这是负责移动滚动条滑块的方法。

还需要在定义`Canvas`后配置每个滚动条的`command`选项：

```py
self.scroll_x = tk.Scrollbar(self, orient=tk.HORIZONTAL)
self.scroll_y = tk.Scrollbar(self, orient=tk.VERTICAL)
self.canvas = tk.Canvas(self, width=300, height=100,
                        xscrollcommand=self.scroll_x.set,
                        yscrollcommand=self.scroll_y.set)
self.scroll_x.config(command=self.canvas.xview)
self.scroll_y.config(command=self.canvas.yview)
```

也可以先创建`Canvas`，然后在实例化滚动条时配置其选项。

下一步是使用`create_window()`方法将框架添加到我们可滚动的`Canvas`中。它接受的第一个参数是使用`window`选项传递的小部件的位置。由于`Canvas`小部件的*x*和*y*轴从左上角开始，我们将框架放置在`(0, 0)`位置，并使用`anchor=tk.NW`将其对齐到该角落（西北）：

```py
self.frame = tk.Frame(self.canvas)
# ...
self.canvas.create_window((0, 0), window=self.frame, anchor=tk.NW)
```

然后，我们将使用`rowconfigure()`和`columnconfigure()`方法使第一行和列可调整大小。`weight`选项指示相对权重以分配额外的空间，但在我们的情况下，没有更多的行或列需要调整大小。

绑定到`<Configure>`事件将帮助我们在主窗口调整大小时正确重新配置`canvas`。处理这种类型的事件遵循我们在上一章中看到的相同原则，以处理鼠标和键盘事件：

```py
self.rowconfigure(0, weight=1)
self.columnconfigure(0, weight=1)
self.bind("<Configure>", self.resize)
```

最后，我们将使用`winfo_width()`和`winfo_height()`方法设置主窗口的最小大小，这些方法可以检索当前的宽度和高度。

为了获得容器的真实大小，我们必须通过调用`update_idletasks()`强制几何管理器首先绘制所有子小部件。这个方法在所有小部件类中都可用，并强制Tkinter处理所有待处理的空闲事件，如重绘和几何重新计算：

```py
self.update_idletasks()
self.minsize(self.winfo_width(), self.winfo_height())
```

`resize`方法处理窗口调整大小事件，并更新`scrollregion`选项，该选项定义了可以滚动的`canvas`区域。为了轻松地重新计算它，您可以使用`bbox()`方法和`ALL`常量。这将返回整个Canvas小部件的边界框：

```py
def resize(self, event):
    region = self.canvas.bbox(tk.ALL)
    self.canvas.configure(scrollregion=region)
```

当我们启动应用程序时，Tkinter将自动触发多个`<Configure>`事件，因此无需在`__init__`方法的末尾调用`self.resize()`。

# 还有更多...

只有少数小部件类支持标准滚动选项：`Listbox`、`Text`和`Canvas`允许`xscrollcommand`和`yscrollcommand`，而输入小部件只允许`xscrollcommand`。我们已经看到如何将此模式应用于`canvas`，因为它可以用作通用解决方案，但您可以遵循类似的结构使这些小部件中的任何一个可滚动和可调整大小。

还有一点要指出的是，我们没有调用任何几何管理器来绘制框架，因为`create_window()`方法会为我们完成这项工作。为了更好地组织我们的应用程序类，我们可以将属于框架及其内部小部件的所有功能移动到专用的`Frame`子类中。

# 另请参阅

+   处理鼠标和键盘事件的方法

+   使用框架对小部件进行分组的方法
