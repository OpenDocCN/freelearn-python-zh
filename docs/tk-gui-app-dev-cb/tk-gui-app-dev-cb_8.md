# 第八章：主题小部件

在本章中，我们将涵盖以下内容：

+   替换基本的小部件类

+   使用 Combobox 创建可编辑的下拉菜单

+   使用 Treeview 小部件

+   在 Treeview 中填充嵌套项目

+   使用 Notebook 显示可切换的窗格

+   应用 Ttk 样式

+   创建日期选择器小部件

# 介绍

Tk 主题小部件是 Tk 小部件的一个单独集合，具有本机外观和感觉，并且它们的样式可以使用特定的 API 进行高度定制。

这些类在`tkinter.ttk`模块中定义。除了定义新的小部件，如 Treeview 和 Notebook，这个模块还重新定义了经典 Tk 小部件的实现，如 Button、Label 和 Frame。

在本章中，我们将不仅涵盖如何将应用程序 Tk 小部件更改为主题小部件，还将涵盖如何对其进行样式设置和使用新的小部件类。

主题 Tk 小部件集是在 Tk 8.5 中引入的，这不应该是一个问题，因为 Python 3.6 安装程序可以让您包含 Tcl/Tk 解释器的 8.6 版本。

但是，您可以通过在命令行中运行`python -m tkinter`来验证任何平台，这将启动以下程序，输出 Tcl/Tk 版本：

![](img/5e1aac09-b958-416e-aaaa-3a752a2ff65f.png)

# 替换基本的小部件类

作为使用主题 Tkinter 类的第一种方法，我们将看看如何从这个不同的模块中使用相同的小部件（按钮、标签、输入框等），在我们的应用程序中保持相同的行为。

尽管这不会充分发挥其样式能力，但我们可以轻松欣赏到带来主题小部件本机外观和感觉的视觉变化。

# 准备工作

在下面的屏幕截图中，您可以注意到带有主题小部件的 GUI 和使用标准 Tkinter 小部件的相同窗口之间的差异：

![](img/64a181c4-fef7-4ef4-ab23-4d4a70afe4ed.png)

我们将构建第一个窗口中显示的应用程序，但我们还将学习如何轻松地在两种样式之间切换。

请注意，这高度依赖于平台。在这种情况下，主题变化对应于 Windows 10 上主题小部件的外观。

# 操作步骤

要开始使用主题小部件，您只需要导入`tkinter.ttk`模块，并像往常一样在您的 Tkinter 应用程序中使用那里定义的小部件：

```py
import tkinter as tk
import tkinter.ttk as ttk

class App(tk.Tk):
    greetings = ("Hello", "Ciao", "Hola")

    def __init__(self):
        super().__init__()
        self.title("Tk themed widgets")

        var = tk.StringVar()
        var.set(self.greetings[0])
        label_frame = ttk.LabelFrame(self, text="Choose a greeting")
        for greeting in self.greetings:
            radio = ttk.Radiobutton(label_frame, text=greeting,
                                    variable=var, value=greeting)
            radio.pack()

        frame = ttk.Frame(self)
        label = ttk.Label(frame, text="Enter your name")
        entry = ttk.Entry(frame)

        command = lambda: print("{}, {}!".format(var.get(), 
                                         entry.get()))
        button = ttk.Button(frame, text="Greet", command=command)

        label.grid(row=0, column=0, padx=5, pady=5)
        entry.grid(row=0, column=1, padx=5, pady=5)
        button.grid(row=1, column=0, columnspan=2, pady=5)

        label_frame.pack(side=tk.LEFT, padx=10, pady=10)
        frame.pack(side=tk.LEFT, padx=10, pady=10)

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

如果您想要使用常规的 Tkinter 小部件运行相同的程序，请将所有`ttk.`出现替换为`tk.`。

# 它是如何工作的...

开始使用主题小部件的常见方法是使用`import ... as`语法导入`tkinter.ttk`模块。因此，我们可以轻松地用`tk`名称标识标准小部件，用`ttk`名称标识主题小部件：

```py
import tkinter as tk
import tkinter.ttk as ttk
```

正如您可能已经注意到的，在前面的代码中，将`tkinter`模块中的小部件替换为`tkinter.ttk`中的等效小部件就像更改别名一样简单：

```py
import tkinter as tk
import tkinter.ttk as ttk

# ...
entry_1 = tk.Entry(root)
entry_2 = ttk.Entry(root)
```

在我们的示例中，我们为`ttk.Frame`、`ttk.Label`、`ttk.Entry`、`ttk.LabelFrame`和`ttk.Radiobutton`小部件这样做。这些类接受的基本选项几乎与它们的标准 Tkinter 等效类相同；事实上，它们实际上是它们的子类。

然而，这个翻译很简单，因为我们没有移植任何样式选项，比如`foreground`或`background`。在主题小部件中，这些关键字通过`ttk.Style`类分别使用，我们将在另一个食谱中介绍。

# 另请参阅

+   *应用 Ttk 样式*食谱

# 使用 Combobox 创建可编辑的下拉菜单

下拉列表是一种简洁的方式，通过垂直显示数值列表来选择数值，只有在需要时才显示。这也是让用户输入列表中不存在的另一个选项的常见方式。

这个功能结合在`ttk.Combobox`类中，它采用您平台下拉菜单的本机外观和感觉。

# 准备工作

我们的下一个应用程序将包括一个简单的下拉输入框，带有一对按钮来确认选择或清除其内容。

如果选择了预定义的值之一或单击了提交按钮，则当前 Combobox 值将以以下方式打印在标准输出中：

![](img/90f6db52-c463-4a4c-bde6-1aeaf09dff48.png)

# 如何做到...

我们的应用程序在初始化期间创建了一个`ttk.Combobox`实例，传递了一个预定义的数值序列，可以在下拉列表中进行选择：

```py
import tkinter as tk
import tkinter.ttk as ttk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ttk Combobox")
        colors = ("Purple", "Yellow", "Red", "Blue")

        self.label = ttk.Label(self, text="Please select a color")
        self.combo = ttk.Combobox(self, values=colors)
        btn_submit = ttk.Button(self, text="Submit",
                                command=self.display_color)
        btn_clear = ttk.Button(self, text="Clear",
                                command=self.clear_color)

        self.combo.bind("<<ComboboxSelected>>", self.display_color)

        self.label.pack(pady=10)
        self.combo.pack(side=tk.LEFT, padx=10, pady=5)
        btn_submit.pack(side=tk.TOP, padx=10, pady=5)
        btn_clear.pack(padx=10, pady=5)

    def display_color(self, *args):
        color = self.combo.get()
        print("Your selection is", color)

    def clear_color(self):
        self.combo.set("")

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

# 它是如何工作的...

与往常一样，通过将`Tk`实例作为其构造函数的第一个参数传递给我们的应用程序，我们将`ttk.Combobox`小部件添加到应用程序中。`values`选项指定了在单击下拉箭头时显示的可选择选项列表。

我们绑定了`"<<ComboboxSelected>>"`虚拟事件，当从值列表中选择一个选项时生成该事件：

```py
        self.label = ttk.Label(self, text="Please select a color")
        self.combo = ttk.Combobox(self, values=colors)
        btn_submit = ttk.Button(self, text="Submit",
                                command=self.display_color)
        btn_clear = ttk.Button(self, text="Clear",
                                command=self.clear_color)

        self.combo.bind("<<ComboboxSelected>>", self.display_color)
```

当单击`提交`按钮时，也会调用相同的方法，因此它会接收用户输入的值。

我们定义`display_color()`使用`*`语法接受可选参数列表。这是因为当通过事件绑定调用它时，会传递一个事件给它，但当从按钮回调中调用它时，不会接收任何参数。

在这个方法中，我们通过其`get()`方法检索当前 Combobox 值，并将其打印在标准输出中：

```py
    def display_color(self, *args):
        color = self.combo.get()
        print("Your selection is", color)
```

最后，`clear_color()`通过调用其`set()`方法并传递空字符串来清除 Combobox 的内容：

```py
    def clear_color(self):
        self.combo.set("")
```

通过这些方法，我们已经探讨了如何与 Combobox 实例的当前选择进行交互。

# 还有更多...

`ttk.Combobox`类扩展了`ttk.Entry`，后者又扩展了`tkinter`模块中的`Entry`类。

这意味着如果需要，我们也可以使用我们已经介绍的`Entry`类的方法：

```py
    combobox.insert(0, "Add this at the beginning: ")
```

前面的代码比`combobox.set("Add this at the beginning: " + combobox.get())`更简单。

# 使用 Treeview 小部件

在这个示例中，我们将介绍`ttk.Treeview`类，这是一个多功能的小部件，可以让我们以表格和分层结构显示信息。

添加到`ttk.Treeview`类的每个项目都分成一个或多个列，其中第一列可能包含文本和图标，并用于指示项目是否可以展开并显示更多嵌套项目。其余的列包含我们想要为每一行显示的值。

`ttk.Treeview`类的第一行由标题组成，通过其名称标识每一列，并可以选择性地隐藏。

# 准备好了

使用`ttk.Treeview`，我们将对存储在 CSV 文件中的联系人列表的信息进行制表，类似于我们在第五章中所做的*面向对象编程和 MVC*：

![](img/5d623144-43ad-44ef-b852-aa63a1428d0b.png)

# 如何做到...

我们将创建一个`ttk.Treeview`小部件，其中包含三列，分别用于每个联系人的字段：一个用于姓，另一个用于名，最后一个用于电子邮件地址。

联系人是使用`csv`模块从 CSV 文件中加载的，然后我们为`"<<TreeviewSelect>>"`虚拟元素添加了绑定，当选择一个或多个项目时生成该元素：

```py
import csv
import tkinter as tk
import tkinter.ttk as ttk

class App(tk.Tk):
    def __init__(self, path):
        super().__init__()
        self.title("Ttk Treeview")

        columns = ("#1", "#2", "#3")
        self.tree = ttk.Treeview(self, show="headings", columns=columns)
        self.tree.heading("#1", text="Last name")
        self.tree.heading("#2", text="First name")
        self.tree.heading("#3", text="Email")
        ysb = ttk.Scrollbar(self, orient=tk.VERTICAL, 
                            command=self.tree.yview)
        self.tree.configure(yscroll=ysb.set)

        with open("contacts.csv", newline="") as f:
            for contact in csv.reader(f):
                self.tree.insert("", tk.END, values=contact)
        self.tree.bind("<<TreeviewSelect>>", self.print_selection)

        self.tree.grid(row=0, column=0)
        ysb.grid(row=0, column=1, sticky=tk.N + tk.S)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def print_selection(self, event):
        for selection in self.tree.selection():
            item = self.tree.item(selection)
            last_name, first_name, email = item["values"][0:3]
            text = "Selection: {}, {} <{}>"
            print(text.format(last_name, first_name, email))

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

如果您运行此程序，每次选择一个联系人，其详细信息都将以标准输出的方式打印出来，以说明如何检索所选行的数据。

# 它是如何工作的...

要创建一个具有多列的`ttk.Treeview`，我们需要使用`columns`选项指定每个列的标识符。然后，我们可以通过调用`heading()`方法来配置标题文本。

我们使用标识符`#1`、`#2`和`#3`，因为第一列始终使用`#0`标识符生成，其中包含可展开的图标和文本。

我们还将`"headings"`值传递给`show`选项，以指示我们要隐藏`#0`列，因为不会有嵌套项目。

`show`选项的有效值如下：

+   `"tree"`：显示列`#0`

+   `"headings"`：显示标题行

+   `"tree headings"`：显示列`#0`和标题行—这是默认值

+   `""`：不显示列`#0`或标题行

之后，我们将垂直滚动条附加到我们的`ttk.Treeview`小部件：

```py
        columns = ("#1", "#2", "#3")
        self.tree = ttk.Treeview(self, show="headings", columns=columns)
        self.tree.heading("#1", text="Last name")
        self.tree.heading("#2", text="First name")
        self.tree.heading("#3", text="Email")
        ysb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=ysb.set)
```

要将联系人加载到表中，我们使用`csv`模块的`reader()`函数处理文件，并在每次迭代中读取的行添加到`ttk.Treeview`中。

这是通过调用`insert()`方法来完成的，该方法接收父节点和放置项目的位置。

由于所有联系人都显示为顶级项目，因此我们将空字符串作为第一个参数传递，并将`END`常量传递以指示每个新项目插入到最后位置。

您可以选择为`insert()`方法提供一些关键字参数。在这里，我们指定了`values`选项，该选项接受在 Treeview 的每一列中显示的值序列：

```py
        with open("contacts.csv", newline="") as f:
            for contact in csv.reader(f):
                self.tree.insert("", tk.END, values=contact)
        self.tree.bind("<<TreeviewSelect>>", self.print_selection)
```

`<<TreeviewSelect>>`事件是用户从表中选择一个或多个项目时生成的虚拟事件。在`print_selection()`处理程序中，我们通过调用`selection()`方法检索当前选择，对于每个结果，我们将执行以下步骤：

1.  使用`item()`方法，我们可以获取所选项目的选项和值的字典

1.  我们从`item`字典中检索前三个值，这些值对应于联系人的姓氏、名字和电子邮件

1.  值被格式化并打印到标准输出：

```py
    def print_selection(self, event):
        for selection in self.tree.selection():
            item = self.tree.item(selection)
            last_name, first_name, email = item["values"][0:3]
            text = "Selection: {}, {} <{}>"
            print(text.format(last_name, first_name, email))
```

# 还有更多...

到目前为止，我们已经涵盖了`ttk.Treeview`类的一些基本方面，因为我们将其用作常规表。但是，还可以通过更高级的功能扩展我们现有的应用程序。

# 在 Treeview 项目中使用标签

`ttk.Treeview`项目可用标签，因此可以为`contacts`表的特定行绑定事件序列。

假设我们希望在双击时打开一个新窗口以给联系人写电子邮件；但是，这仅适用于填写了电子邮件字段的记录。

我们可以通过在插入项目时有条件地向其添加标签，然后在小部件实例上使用`"<Double-Button-1>"`序列调用`tag_bind()`来轻松实现这一点——这里我们只是通过其名称引用`send_email_to_contact()`处理程序函数的实现：

```py
    columns = ("Last name", "First name", "Email")
    tree = ttk.Treeview(self, show="headings", columns=columns)

    for contact in csv.reader(f):
        email = contact[2]
 tags = ("dbl-click",) if email else ()
 self.tree.insert("", tk.END, values=contact, tags=tags)

 tree.tag_bind("dbl-click", "<Double-Button-1>", send_email_to_contact)
```

与将事件绑定到`Canvas`项目时发生的情况类似，始终记住在调用`tag_bind()`之前将带有标签的项目添加到`ttk.Treeview`中，因为绑定仅添加到现有的匹配项目。

# 另请参阅

+   *在 Treeview 中填充嵌套项目*食谱

# 在 Treeview 中填充嵌套项目

虽然`ttk.Treeview`可以用作常规表，但它也可能包含分层结构。这显示为树，其中的项目可以展开以查看层次结构的更多节点。

这对于显示递归调用的结果和多层嵌套项目非常有用。在此食谱中，我们将研究适合这种结构的常见场景。

# 准备就绪

为了说明如何在`ttk.Treeview`小部件中递归添加项目，我们将创建一个基本的文件系统浏览器。可展开的节点将表示文件夹，一旦打开，它们将显示它们包含的文件和文件夹：

![](img/9bd56ff2-7170-4c16-821d-af9d1f31e0ca.png)

# 如何做...

树将最初由`populate_node()`方法填充，该方法列出当前目录中的条目。如果条目是目录，则还会添加一个空子项以显示它作为可展开节点。

打开表示目录的节点时，它会通过再次调用`populate_node()`来延迟加载目录的内容。这次，不是将项目添加为顶级节点，而是将它们嵌套在打开的节点内部：

```py
import os
import tkinter as tk
import tkinter.ttk as ttk

class App(tk.Tk):
    def __init__(self, path):
        super().__init__()
        self.title("Ttk Treeview")

        abspath = os.path.abspath(path)
        self.nodes = {}
        self.tree = ttk.Treeview(self)
        self.tree.heading("#0", text=abspath, anchor=tk.W)
        ysb = ttk.Scrollbar(self, orient=tk.VERTICAL,
                            command=self.tree.yview)
        xsb = ttk.Scrollbar(self, orient=tk.HORIZONTAL,
                            command=self.tree.xview)
        self.tree.configure(yscroll=ysb.set, xscroll=xsb.set)

        self.tree.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E +     tk.W)
        ysb.grid(row=0, column=1, sticky=tk.N + tk.S)
        xsb.grid(row=1, column=0, sticky=tk.E + tk.W)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.tree.bind("<<TreeviewOpen>>", self.open_node)
        self.populate_node("", abspath)

    def populate_node(self, parent, abspath):
        for entry in os.listdir(abspath):
            entry_path = os.path.join(abspath, entry)
            node = self.tree.insert(parent, tk.END, text=entry, open=False)
            if os.path.isdir(entry_path):
                self.nodes[node] = entry_path
                self.tree.insert(node, tk.END)

    def open_node(self, event):
        item = self.tree.focus()
        abspath = self.nodes.pop(item, False)
        if abspath:
            children = self.tree.get_children(item)
            self.tree.delete(children)
            self.populate_node(item, abspath)

if __name__ == "__main__":
    app = App(path=".")
    app.mainloop()
```

当运行上述示例时，它将显示脚本所在目录的文件系统层次结构，但您可以通过`App`构造函数的`path`参数明确设置所需的目录。

# 工作原理

在这个例子中，我们将使用`os`模块，它是 Python 标准库的一部分，提供了执行操作系统调用的便携方式。

`os`模块的第一个用途是将树的初始路径转换为绝对路径，以及初始化`nodes`字典，它将存储可展开项和它们表示的目录路径之间的对应关系：

```py
import os
import tkinter as tk
import tkinter.ttk as ttk

class App(tk.Tk):
    def __init__(self, path):
        # ...
 abspath = os.path.abspath(path)
 self.nodes = {}
```

例如，`os.path.abspath(".")`将返回你从脚本运行的路径的绝对版本。我们更喜欢这种方法而不是使用相对路径，因为这样可以避免在应用程序中处理路径时出现混淆。

现在，我们使用垂直和水平滚动条初始化`ttk.Treeview`实例。图标标题的`text`将是我们之前计算的绝对路径：

```py
        self.tree = ttk.Treeview(self)
        self.tree.heading("#0", text=abspath, anchor=tk.W)
        ysb = ttk.Scrollbar(self, orient=tk.VERTICAL,
                            command=self.tree.yview)
        xsb = ttk.Scrollbar(self, orient=tk.HORIZONTAL,
                            command=self.tree.xview)
        self.tree.configure(yscroll=ysb.set, xscroll=xsb.set)
```

然后，我们使用 Grid 布局管理器放置小部件，并使`ttk.Treeview`实例在水平和垂直方向上自动调整大小。

之后，我们绑定了`"<<TreeviewOpen>>"`虚拟事件，当展开项时生成，调用`open_node()`处理程序并调用`populate_node()`加载指定目录的条目：

```py
        self.tree.bind("<<TreeviewOpen>>", self.open_node)
        self.populate_node("", abspath)
```

请注意，第一次调用此方法时，父目录为空字符串，这意味着它们没有任何父项，并显示为顶级项。

在`populate_node()`方法中，我们通过调用`os.listdir()`列出目录条目的名称。对于每个条目名称，我们执行以下操作：

+   计算条目的绝对路径。在类 UNIX 系统上，这是通过用斜杠连接字符串来实现的，但 Windows 使用反斜杠。由于`os.path.join()`方法，我们可以安全地连接路径，而不必担心平台相关的细节。

+   我们将`entry`字符串插入到指定的`parent`节点的最后一个子项中。我们总是将节点初始设置为关闭，因为我们希望仅在需要时延迟加载嵌套项。

+   如果条目的绝对路径是一个目录，我们在`nodes`属性中添加节点和路径之间的对应关系，并插入一个空的子项，允许该项展开：

```py
    def populate_node(self, parent, abspath):
        for entry in os.listdir(abspath):
            entry_path = os.path.join(abspath, entry)
            node = self.tree.insert(parent, tk.END, text=entry, open=False)
            if os.path.isdir(entry_path):
                self.nodes[node] = entry_path
                self.tree.insert(node, tk.END)
```

当单击可展开项时，`open_node()`处理程序通过调用`ttk.Treeview`实例的`focus()`方法检索所选项。

此项标识符用于获取先前添加到`nodes`属性的绝对路径。为了避免在字典中节点不存在时引发`KeyError`，我们使用了它的`pop()`方法，它将第二个参数作为默认值返回——在我们的例子中是`False`。

如果节点存在，我们清除可展开节点的“虚假”项。调用`self.tree.get_children(item)`返回`item`的子项的标识符，然后通过调用`self.tree.delete(children)`来删除它们。

一旦清除了该项，我们通过使用`item`作为父项调用`populate_node()`来添加“真实”的子项：

```py
    def open_node(self, event):
        item = self.tree.focus()
        abspath = self.nodes.pop(item, False)
        if abspath:
            children = self.tree.get_children(item)
            self.tree.delete(children)
            self.populate_node(item, abspath)
```

# 显示带有 Notebook 的选项卡窗格

`ttk.Notebook`类是`ttk`模块中引入的另一种新的小部件类型。它允许您在同一窗口区域中添加许多应用程序视图，让您通过单击与每个视图关联的选项卡来选择应该显示的视图。

选项卡面板是重用 GUI 相同部分的好方法，如果多个区域的内容不需要同时显示。

# 准备工作

以下应用程序显示了一些按类别分隔的待办事项列表，列表显示为只读数据，以简化示例：

![](img/de95909e-d3ee-4c23-9283-94af0b2928d9.png)

# 操作步骤

我们使用固定大小实例化`ttk.Notebook`，然后循环遍历具有一些预定义数据的字典，这些数据将用于创建选项卡并向每个区域添加一些标签：

```py
import tkinter as tk
import tkinter.ttk as ttk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ttk Notebook")

        todos = {
            "Home": ["Do the laundry", "Go grocery shopping"],
            "Work": ["Install Python", "Learn Tkinter", "Reply emails"],
            "Vacations": ["Relax!"]
        }

        self.notebook = ttk.Notebook(self, width=250, height=100)
        self.label = ttk.Label(self)
        for key, value in todos.items():
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=key, underline=0,
                              sticky=tk.NE + tk.SW)
            for text in value:
                ttk.Label(frame, text=text).pack(anchor=tk.W)

        self.notebook.pack()
        self.label.pack(anchor=tk.W)
        self.notebook.enable_traversal()
        self.notebook.bind("<<NotebookTabChanged>>", self.select_tab)

    def select_tab(self, event):
        tab_id = self.notebook.select()
        tab_name = self.notebook.tab(tab_id, "text")
        text = "Your current selection is: {}".format(tab_name)
        self.label.config(text=text)

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

每次单击选项卡时，窗口底部的标签都会更新其内容，显示当前选项卡的名称。

# 它是如何工作的...

我们的`ttk.Notebook`小部件具有特定的宽度和高度，以及外部填充。

`todos`字典中的每个键都用作选项卡的名称，并且值列表被添加为标签到`ttk.Frame`，它代表窗口区域：

```py
 self.notebook = ttk.Notebook(self, width=250, height=100, padding=10)
        for key, value in todos.items():
            frame = ttk.Frame(self.notebook)
 self.notebook.add(frame, text=key,
                              underline=0, sticky=tk.NE+tk.SW)
            for text in value:
                ttk.Label(frame, text=text).pack(anchor=tk.W)
```

之后，我们在`ttk.Notebook`小部件上调用`enable_traversal()`。这允许用户使用*Ctrl* + *Shift* + *Tab*和*Ctrl* + *Tab*在选项卡面板之间来回切换选项卡。

它还可以通过按下*Alt*和下划线字符来切换到特定的选项卡，即*Alt* + *H*代表`Home`选项卡，*Alt* + *W*代表`Work`选项卡，*Alt* + *V*代表`Vacation`选项卡。

当选项卡选择更改时，生成`"<<NotebookTabChanged>>"`虚拟事件，并将其绑定到`select_tab()`方法。请注意，当 Tkinter 添加一个选项卡到`ttk.Notebook`时，此事件会自动触发：

```py
        self.notebook.pack()
        self.label.pack(anchor=tk.W)
 self.notebook.enable_traversal()
 self.notebook.bind("<<NotebookTabChanged>>", self.select_tab)
```

当我们打包项目时，不需要放置`ttk.Notebook`子窗口，因为`ttk.Notebook`调用几何管理器内部完成了这一点：

```py
    def select_tab(self, event):
        tab_id = self.notebook.select()
        tab_name = self.notebook.tab(tab_id, "text")
        self.label.config(text=f"Your current selection is: {tab_name}")
```

# 还有更多...

如果您想要检索`ttk.Notebook`当前显示的子窗口，您不需要使用任何额外的数据结构来将选项卡索引与小部件窗口进行映射。

Tkinter 的`nametowidget()`方法可从所有小部件类中使用，因此您可以轻松获取与小部件名称对应的小部件对象：

```py
    def select_tab(self, event):
        tab_id = self.notebook.select()
        frame = self.nametowidget(tab_id)
        # Do something with the frame
```

# 应用 Ttk 样式

正如我们在本章的第一个配方中提到的，主题小部件具有特定的 API 来自定义它们的外观。我们不能直接设置选项，例如前景色或内部填充，因为这些值是通过`ttk.Style`类设置的。

在这个配方中，我们将介绍如何修改第一个配方中的小部件以添加一些样式选项。

# 如何做...

为了添加一些默认设置，我们只需要一个`ttk.Style`对象，它提供以下方法：

+   `configure(style, opts)`: 更改小部件`style`的外观`opts`。在这里，我们设置诸如前景色、填充和浮雕等选项。

+   `map(style, query)`: 更改小部件`style`的动态外观。参数`query`是一个关键字参数，其中每个键都是样式选项，值是`(state, value)`形式的元组列表，表示选项的值由其当前状态确定。

例如，我们已经标记了以下两种情况的示例：

```py
import tkinter as tk
import tkinter.ttk as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tk themed widgets")

        style = ttk.Style(self)
 style.configure("TLabel", padding=10)
 style.map("TButton",
 foreground=[("pressed", "grey"), ("active", "white")],
 background=[("pressed", "white"), ("active", "grey")]
 ) # ...
```

现在，每个`ttk.Label`显示为`10`的填充，`ttk.Button`具有动态样式：当状态为`pressed`时，灰色前景和白色背景，当状态为`active`时，白色前景和灰色背景。

# 它是如何工作的...

为我们的应用程序构建`ttk.Style`非常简单——我们只需要使用我们的父小部件作为它的第一个参数创建一个实例。

然后，我们可以为我们的主题小部件设置默认样式选项，使用大写的`T`加上小部件名称：`TButton`代表`ttk.Button`，`TLabel`代表`ttk.Label`，依此类推。然而，也有一些例外，因此建议您在 Python 解释器上调用小部件实例的`winfo_class()`方法来检查类名。

我们还可以添加前缀来标识我们不想默认使用的样式，但明确地将其设置为某些特定的小部件：

```py
        style.configure("My.TLabel", padding=10)
        # ...
        label = ttk.Label(master, text="Some text", style="My.TLabel")
```

# 创建日期选择器小部件

如果我们想让用户在我们的应用程序中输入日期，我们可以添加一个文本输入，强制他们编写一个带有有效日期格式的字符串。另一种解决方案是添加几个数字输入，用于日期、月份和年份，但这也需要一些验证。

与其他 GUI 框架不同，Tkinter 不包括一个专门用于此目的的类；然而，我们可以选择应用我们对主题小部件的知识来构建一个日历小部件。

# 准备就绪

在这个配方中，我们将逐步解释使用 Ttk 小部件和功能制作日期选择器小部件的实现：

![](img/2af6aa0e-0480-4c96-b406-d51e96c7d0f1.png)

这是 [`svn.python.org/projects/sandbox/trunk/ttk-gsoc/samples/ttkcalendar.py`](http://svn.python.org/projects/sandbox/trunk/ttk-gsoc/samples/ttkcalendar.py) 的重构版本，不需要任何外部包。

# 操作步骤...

除了 `tkinter` 模块，我们还需要标准库中的 `calendar` 和 `datetime` 模块。它们将帮助我们对小部件中保存的数据进行建模和交互。

小部件标题显示了一对箭头，根据 Ttk 样式选项来前后移动当前月份。小部件的主体由一个 `ttk.Treeview` 表格组成，其中包含一个 `Canvas` 实例来突出显示所选日期单元格：

```py
import calendar
import datetime
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkfont
from itertools import zip_longest

class TtkCalendar(ttk.Frame):
    def __init__(self, master=None, **kw):
        now = datetime.datetime.now()
        fwday = kw.pop('firstweekday', calendar.MONDAY)
        year = kw.pop('year', now.year)
        month = kw.pop('month', now.month)
        sel_bg = kw.pop('selectbackground', '#ecffc4')
        sel_fg = kw.pop('selectforeground', '#05640e')

        super().__init__(master, **kw)

        self.selected = None
        self.date = datetime.date(year, month, 1)
        self.cal = calendar.TextCalendar(fwday)
        self.font = tkfont.Font(self)
        self.header = self.create_header()
        self.table = self.create_table()
        self.canvas = self.create_canvas(sel_bg, sel_fg)
        self.build_calendar()

    def create_header(self):
        left_arrow = {'children': [('Button.leftarrow', None)]}
        right_arrow = {'children': [('Button.rightarrow', None)]}
        style = ttk.Style(self)
        style.layout('L.TButton', [('Button.focus', left_arrow)])
        style.layout('R.TButton', [('Button.focus', right_arrow)])

        hframe = ttk.Frame(self)
        btn_left = ttk.Button(hframe, style='L.TButton',
                              command=lambda: self.move_month(-1))
        btn_right = ttk.Button(hframe, style='R.TButton',
                               command=lambda: self.move_month(1))
        label = ttk.Label(hframe, width=15, anchor='center')

        hframe.pack(pady=5, anchor=tk.CENTER)
        btn_left.grid(row=0, column=0)
        label.grid(row=0, column=1, padx=12)
        btn_right.grid(row=0, column=2)
        return label

    def move_month(self, offset):
        self.canvas.place_forget()
        month = self.date.month - 1 + offset
        year = self.date.year + month // 12
        month = month % 12 + 1
        self.date = datetime.date(year, month, 1)
        self.build_calendar()

    def create_table(self):
        cols = self.cal.formatweekheader(3).split()
        table = ttk.Treeview(self, show='', selectmode='none',
                             height=7, columns=cols)
        table.bind('<Map>', self.minsize)
        table.pack(expand=1, fill=tk.BOTH)
        table.tag_configure('header', background='grey90')
        table.insert('', tk.END, values=cols, tag='header')
        for _ in range(6):
            table.insert('', tk.END)

        width = max(map(self.font.measure, cols))
        for col in cols:
            table.column(col, width=width, minwidth=width, anchor=tk.E)
        return table

    def minsize(self, e):
        width, height = self.master.geometry().split('x')
        height = height[:height.index('+')]
        self.master.minsize(width, height)

    def create_canvas(self, bg, fg):
        canvas = tk.Canvas(self.table, background=bg,
                           borderwidth=0, highlightthickness=0)
        canvas.text = canvas.create_text(0, 0, fill=fg, anchor=tk.W)
        handler = lambda _: canvas.place_forget()
        canvas.bind('<ButtonPress-1>', handler)
        self.table.bind('<Configure>', handler)
        self.table.bind('<ButtonPress-1>', self.pressed)
        return canvas

    def build_calendar(self):
        year, month = self.date.year, self.date.month
        month_name = self.cal.formatmonthname(year, month, 0)
        month_weeks = self.cal.monthdayscalendar(year, month)

        self.header.config(text=month_name.title())
        items = self.table.get_children()[1:]
        for week, item in zip_longest(month_weeks, items):
            week = week if week else [] 
            fmt_week = ['%02d' % day if day else '' for day in week]
            self.table.item(item, values=fmt_week)

    def pressed(self, event):
        x, y, widget = event.x, event.y, event.widget
        item = widget.identify_row(y)
        column = widget.identify_column(x)
        items = self.table.get_children()[1:]

        if not column or not item in items:
            # clicked te header or outside the columns
            return

        index = int(column[1]) - 1
        values = widget.item(item)['values']
        text = values[index] if len(values) else None
        bbox = widget.bbox(item, column)
        if bbox and text:
            self.selected = '%02d' % text
            self.show_selection(bbox)

    def show_selection(self, bbox):
        canvas, text = self.canvas, self.selected
        x, y, width, height = bbox
        textw = self.font.measure(text)
        canvas.configure(width=width, height=height)
        canvas.coords(canvas.text, width - textw, height / 2 - 1)
        canvas.itemconfigure(canvas.text, text=text)
        canvas.place(x=x, y=y)

    @property
    def selection(self):
        if self.selected:
            year, month = self.date.year, self.date.month
            return datetime.date(year, month, int(self.selected))

def main():
    root = tk.Tk()
    root.title('Tkinter Calendar')
    ttkcal = TtkCalendar(firstweekday=calendar.SUNDAY)
    ttkcal.pack(expand=True, fill=tk.BOTH)
    root.mainloop()

if __name__ == '__main__':
    main()
```

# 工作原理...

我们的 `TtkCalendar` 类可以通过传递一些选项作为关键字参数来进行自定义。它们在初始化时被检索出来，并在没有提供的情况下使用一些默认值；例如，如果当前日期用于日历的初始年份和月份：

```py
    def __init__(self, master=None, **kw):
        now = datetime.datetime.now()
        fwday = kw.pop('firstweekday', calendar.MONDAY)
        year = kw.pop('year', now.year)
        month = kw.pop('month', now.month)
        sel_bg = kw.pop('selectbackground', '#ecffc4')
        sel_fg = kw.pop('selectforeground', '#05640e')

        super().__init__(master, **kw)
```

然后，我们定义一些属性来存储日期信息：

+   `selected`：保存所选日期的值

+   `date`：表示在日历上显示的当前月份的日期

+   `calendar`：具有周和月份名称信息的公历日历

小部件的可视部分在 `create_header()` 和 `create_table()` 方法中内部实例化，稍后我们将对其进行介绍。

我们还使用了一个 `tkfont.Font` 实例来帮助我们测量字体大小。

一旦这些属性被初始化，通过调用 `build_calendar()` 方法来安排日历的可视部分：

```py
        self.selected = None
        self.date = datetime.date(year, month, 1)
        self.cal = calendar.TextCalendar(fwday)
        self.font = tkfont.Font(self)
        self.header = self.create_header()
        self.table = self.create_table()
        self.canvas = self.create_canvas(sel_bg, sel_fg)
        self.build_calendar()
```

`create_header()` 方法使用 `ttk.Style` 来显示箭头以前后移动月份。它返回显示当前月份名称的标签：

```py
    def create_header(self):
        left_arrow = {'children': [('Button.leftarrow', None)]}
        right_arrow = {'children': [('Button.rightarrow', None)]}
        style = ttk.Style(self)
        style.layout('L.TButton', [('Button.focus', left_arrow)])
        style.layout('R.TButton', [('Button.focus', right_arrow)])

        hframe = ttk.Frame(self)
        lbtn = ttk.Button(hframe, style='L.TButton',
                          command=lambda: self.move_month(-1))
        rbtn = ttk.Button(hframe, style='R.TButton',
                          command=lambda: self.move_month(1))
        label = ttk.Label(hframe, width=15, anchor='center')

        # ...
        return label
```

`move_month()` 回调隐藏了用画布字段突出显示的当前选择，并将指定的 `offset` 添加到当前月份以设置 `date` 属性为上一个或下一个月份。然后，日历再次重绘，显示新月份的日期：

```py
    def move_month(self, offset):
        self.canvas.place_forget()
        month = self.date.month - 1 + offset
        year = self.date.year + month // 12
        month = month % 12 + 1
        self.date = datetime.date(year, month, 1)
        self.build_calendar()
```

日历主体是在 `create_table()` 中使用 `ttk.Treeview` 小部件创建的，它在一行中显示当前月份的每周：

```py
    def create_table(self):
        cols = self.cal.formatweekheader(3).split()
        table = ttk.Treeview(self, show='', selectmode='none',
                             height=7, columns=cols)
        table.bind('<Map>', self.minsize)
        table.pack(expand=1, fill=tk.BOTH)
        table.tag_configure('header', background='grey90')
        table.insert('', tk.END, values=cols, tag='header')
        for _ in range(6):
            table.insert('', tk.END)

        width = max(map(self.font.measure, cols))
        for col in cols:
            table.column(col, width=width, minwidth=width, anchor=tk.E)
        return table
```

在 `create_canvas()` 方法中实例化了突出显示选择的画布。由于它根据所选项的尺寸调整其大小，因此如果窗口被调整大小，它也会隐藏自己：

```py
    def create_canvas(self, bg, fg):
        canvas = tk.Canvas(self.table, background=bg,
                           borderwidth=0, highlightthickness=0)
        canvas.text = canvas.create_text(0, 0, fill=fg, anchor=tk.W)
        handler = lambda _: canvas.place_forget()
        canvas.bind('<ButtonPress-1>', handler)
        self.table.bind('<Configure>', handler)
        self.table.bind('<ButtonPress-1>', self.pressed)
        return canvas
```

通过迭代 `ttk.Treeview` 表格的周和项目位置来构建日历。使用 `itertools` 模块中的 `zip_longest()` 函数，我们遍历包含大多数项目的集合，并将缺少的日期留空字符串：

![](img/92fb9a31-6eb8-431c-8244-13316aa52fde.png)

这种行为对每个月的第一周和最后一周很重要，因为这通常是我们找到这些空白位置的地方：

```py
    def build_calendar(self):
        year, month = self.date.year, self.date.month
        month_name = self.cal.formatmonthname(year, month, 0)
        month_weeks = self.cal.monthdayscalendar(year, month)

        self.header.config(text=month_name.title())
        items = self.table.get_children()[1:]
        for week, item in zip_longest(month_weeks, items):
            week = week if week else [] 
            fmt_week = ['%02d' % day if day else '' for day in week]
            self.table.item(item, values=fmt_week)
```

当您单击表项时，`pressed()` 事件处理程序如果该项存在则设置选择，并重新显示画布以突出显示选择：

```py
    def pressed(self, event):
        x, y, widget = event.x, event.y, event.widget
        item = widget.identify_row(y)
        column = widget.identify_column(x)
        items = self.table.get_children()[1:]

        if not column or not item in items:
            # clicked te header or outside the columns
            return

        index = int(column[1]) - 1
        values = widget.item(item)['values']
        text = values[index] if len(values) else None
        bbox = widget.bbox(item, column)
        if bbox and text:
            self.selected = '%02d' % text
            self.show_selection(bbox)
```

`show_selection()` 方法将画布放置在包含选择的边界框上，测量文本大小以使其适合其上方：

```py
    def show_selection(self, bbox):
        canvas, text = self.canvas, self.selected
        x, y, width, height = bbox
        textw = self.font.measure(text)
        canvas.configure(width=width, height=height)
        canvas.coords(canvas.text, width - textw, height / 2 - 1)
        canvas.itemconfigure(canvas.text, text=text)
        canvas.place(x=x, y=y)
```

最后，`selection` 属性使得可以将所选日期作为 `datetime.date` 对象获取。在我们的示例中没有直接使用它，但它是与 `TtkCalendar` 类一起使用的 API 的一部分：

```py
    @property
    def selection(self):
        if self.selected:
            year, month = self.date.year, self.date.month
            return datetime.date(year, month, int(self.selected))
```

# 另请参阅

+   *使用 Treeview 小部件* 配方

+   *应用 Ttk 样式* 配方
