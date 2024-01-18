# 自定义小部件

在本章中，我们将涵盖以下示例：

+   使用颜色

+   设置小部件字体

+   使用选项数据库

+   更改光标图标

+   介绍文本小部件

+   向文本小部件添加标签

# 介绍

默认情况下，Tkinter 小部件将显示本机外观和感觉。虽然这种标准外观可能足够快速原型设计，但我们可能希望自定义一些小部件属性，如字体、颜色和背景。

这种自定义不仅影响小部件本身，还影响其内部项目。我们将深入研究文本小部件，它与画布小部件一样是最多功能的 Tkinter 类之一。文本小部件表示具有格式化内容的多行文本区域，具有几种方法，使得可以格式化字符或行并添加特定事件绑定。

# 使用颜色

在以前的示例中，我们使用颜色名称（如白色、蓝色或黄色）来设置小部件的颜色。这些值作为字符串传递给`foreground`和`background`选项，这些选项修改了小部件的文本和背景颜色。

颜色名称内部映射到**RGB**值（一种通过红、绿和蓝强度的组合来表示颜色的加法模型），这种转换基于一个因平台而异的表。因此，如果要在不同平台上一致显示相同的颜色，可以将 RGB 值传递给小部件选项。

# 准备就绪

以下应用程序显示了如何动态更改显示固定文本的标签的`foreground`和`background`选项： 

![](img/2988a6bc-c48d-49b5-843b-b37f4a7f858a.png)

颜色以 RGB 格式指定，并由用户使用本机颜色选择对话框选择。以下屏幕截图显示了 Windows 10 上的此对话框的外观：

![](img/7be30749-7137-4489-9b0a-ed89b07827d5.png)

# 如何做...

像往常一样，我们将使用标准按钮触发小部件配置——每个选项一个按钮。与以前的示例的主要区别是，可以直接使用`tkinter.colorchooser`模块的`askcolor`对话框直接选择值：

```py
from functools import partial

import tkinter as tk
from tkinter.colorchooser import askcolor

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Colors demo")
        text = "The quick brown fox jumps over the lazy dog"
        self.label = tk.Label(self, text=text)
        self.fg_btn = tk.Button(self, text="Set foreground color",
                                command=partial(self.set_color, "fg")) 
        self.bg_btn = tk.Button(self, text="Set background color",
                                command=partial(self.set_color, "bg"))

        self.label.pack(padx=20, pady=20)
        self.fg_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.bg_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def set_color(self, option):
        color = askcolor()[1]
        print("Chosen color:", color)
        self.label.config(**{option: color})

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

如果要查看所选颜色的 RGB 值，在对话框确认时会在控制台上打印出来，如果关闭而没有选择颜色，则不会显示任何值。

# 它是如何工作的...

正如您可能已经注意到的，两个按钮都使用了部分函数作为回调。这是`functools`模块中的一个实用程序，它创建一个新的可调用对象，其行为类似于原始函数，但带有一些固定的参数。例如，考虑以下语句：

```py
tk.Button(self, command=partial(self.set_color, "fg"), ...)
```

前面的语句执行与以下语句相同的操作：

```py
tk.Button(self, command=lambda: self.set_color("fg"), ...)
```

我们这样做是为了同时重用我们的`set_color()`方法和引入`functools`模块。这些技术在更复杂的场景中非常有用，特别是当您想要组合多个函数并且非常清楚地知道一些参数已经预定义时。

要记住的一个小细节是，我们用`fg`和`bg`分别缩写了`foreground`和`background`。在这个语句中，这些字符串使用`**`进行解包，用于配置小部件：

```py
def set_color(self, option):
    color = askcolor()[1]
    print("Chosen color:", color)
    self.label.config(**{option: color}) # same as (fg=color)
                      or (bg=color)
```

`askcolor`返回一个包含两个项目的元组，表示所选颜色——第一个是表示 RGB 值的整数元组，第二个是十六进制代码作为字符串。由于第一个表示不能直接传递给小部件选项，我们使用了十六进制格式。

# 还有更多...

如果要将颜色名称转换为 RGB 格式，可以在先前创建的小部件上使用`winfo_rgb()`方法。由于它返回一个整数元组，表示 16 位 RGB 值的整数从 0 到 65535，您可以通过向右移动 8 位将其转换为更常见的*#RRGGBB*十六进制表示：

```py
rgb = widget.winfo_rgb("lightblue")
red, green, blue = [x>>8 for x in rgb]
print("#{:02x}{:02x}{:02x}".format(red, green, blue))
```

在前面的代码中，我们使用`{:02x}`将每个整数格式化为两个十六进制数字。

# 设置小部件字体

在 Tkinter 中，可以自定义用于向用户显示文本的小部件的字体，例如按钮、标签和输入框。默认情况下，字体是特定于系统的，但可以使用`font`选项进行更改。

# 准备工作

以下应用程序允许用户动态更改具有静态文本的标签的字体系列和大小。尝试不同的值以查看字体配置的结果：

![](img/cd7bce85-0d22-479b-9990-315236013a54.png)

# 如何做...

我们将有两个小部件来修改字体配置：一个下拉选项，其中包含字体系列名称，以及一个输入字体大小的微调框：

```py
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fonts demo")
        text = "The quick brown fox jumps over the lazy dog"
        self.label = tk.Label(self, text=text)

        self.family = tk.StringVar()
        self.family.trace("w", self.set_font)
        families = ("Times", "Courier", "Helvetica")
        self.option = tk.OptionMenu(self, self.family, *families)

        self.size = tk.StringVar()
        self.size.trace("w", self.set_font)
        self.spinbox = tk.Spinbox(self, from_=8, to=18,
                                  textvariable=self.size)

        self.family.set(families[0])
        self.size.set("10")
        self.label.pack(padx=20, pady=20)
        self.option.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.spinbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def set_font(self, *args):
        family = self.family.get()
        size = self.size.get()
        self.label.config(font=(family, size))

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

请注意，我们已为与每个输入连接的 Tkinter 变量设置了一些默认值。

# 它是如何工作的...

`FAMILIES`元组包含`Tk`保证在所有平台上支持的三种字体系列：`Times`（Times New Roman）、`Courier`和`Helvetica`。它们可以通过与`self.family`变量连接的`OptionMenu`小部件进行切换。

类似的方法用于使用`Spinbox`设置字体大小。这两个变量触发了更改`font`标签的方法：

```py
def set_font(self, *args):
    family = self.family.get()
    size = self.size.get()
    self.label.config(font=(family, size))
```

传递给`font`选项的元组还可以定义以下一个或多个字体样式：粗体、罗马体、斜体、下划线和删除线：

```py
widget1.config(font=("Times", "20", "bold"))
widget2.config(font=("Helvetica", "16", "italic underline"))
```

您可以使用`tkinter.font`模块的`families()`方法检索可用字体系列的完整列表。由于您需要首先实例化`root`窗口，因此可以使用以下脚本：

```py
import tkinter as tk
from tkinter import font

root = tk.Tk()
print(font.families())
```

如果您使用的字体系列未包含在可用系列列表中，Tkinter 不会抛出任何错误，而是会尝试匹配类似的字体。

# 还有更多...

`tkinter.font`模块包括一个`Font`类，可以在多个小部件上重复使用。修改`font`实例的主要优势是它会影响与`font`选项共享它的所有小部件。

使用`Font`类的工作方式与使用字体描述符非常相似。例如，此代码段创建一个 18 像素的`Courier`粗体字体：

```py
from tkinter import font
courier_18 = font.Font(family="Courier", size=18, weight=font.BOLD)
```

要检索或更改选项值，您可以像往常一样使用`cget`和`configure`方法：

```py
family = courier_18.cget("family")
courier_18.configure(underline=1)
```

# 另请参阅

+   *使用选项数据库*配方

# 使用选项数据库

Tkinter 定义了一个称为*选项数据库*的概念，这是一种用于自定义应用程序外观的机制，而无需为每个小部件指定它。它允许您将一些小部件选项与单个小部件配置分离开来，根据小部件层次结构提供标准化的默认值。

# 准备工作

在此配方中，我们将构建一个具有不同样式的多个小部件的应用程序，这些样式将在选项数据库中定义：

![](img/ed5977dc-0974-4b65-8e65-d56373bfdbf7.png)

# 如何做...

在我们的示例中，我们将通过`option_add()`方法向数据库添加一些选项，该方法可以从所有小部件类访问：

```py
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Options demo")
        self.option_add("*font", "helvetica 10")
        self.option_add("*header.font", "helvetica 18 bold")
        self.option_add("*subtitle.font", "helvetica 14 italic")
        self.option_add("*Button.foreground", "blue")
        self.option_add("*Button.background", "white")
        self.option_add("*Button.activeBackground", "gray")
        self.option_add("*Button.activeForeground", "black")

        self.create_label(name="header", text="This is the header")
        self.create_label(name="subtitle", text="This is the subtitle")
        self.create_label(text="This is a paragraph")
        self.create_label(text="This is another paragraph")
        self.create_button(text="See more")

    def create_label(self, **options):
        tk.Label(self, **options).pack(padx=20, pady=5, anchor=tk.W)

    def create_button(self, **options):
        tk.Button(self, **options).pack(padx=5, pady=5, anchor=tk.E)

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

因此，Tkinter 将使用选项数据库中定义的默认值，而不是与其他选项一起配置字体、前景和背景。

# 它是如何工作的...

让我们从解释对`option_add`的每个调用开始。第一次调用添加了一个选项，将`font`属性设置为所有小部件——通配符代表任何应用程序名称：

```py
self.option_add("*font", "helvetica 10")
```

下一个调用将匹配限制为具有`header`名称的元素——规则越具体，优先级越高。稍后在使用`name="header"`实例化标签时指定此名称：

```py
self.option_add("*header.font", "helvetica 18 bold")
```

对于`self.option_add("*subtitle.font", "helvetica 14 italic")`，也是一样的，所以每个选项都匹配到不同命名的小部件实例。

下一个选项使用`Button`类名而不是实例名。这样，您可以引用给定类的所有小部件以提供一些公共默认值：

```py
self.option_add("*Button.foreground", "blue")
self.option_add("*Button.background", "white")
self.option_add("*Button.activeBackground", "gray")
self.option_add("*Button.activeForeground", "black")
```

正如我们之前提到的，选项数据库使用小部件层次结构来确定适用于每个实例的选项，因此，如果我们有嵌套的容器，它们也可以用于限制优先级选项。

这些配置选项不适用于现有小部件，只适用于修改选项数据库后创建的小部件。因此，我们始终建议在应用程序开头调用`option_add()`。

这些是一些示例，每个示例比前一个更具体：

+   `*Frame*background`：匹配框架内所有小部件的背景

+   `*Frame.background`：匹配所有框架的背景

+   `*Frame.myButton.background`：匹配名为`myButton`的小部件的背景

+   `*myFrame.myButton.background`：匹配容器名为`myFrame`内名为`myButton`的小部件的背景

# 还有更多...

不仅可以通过编程方式添加选项，还可以使用以下格式在单独的文本文件中定义它们：

```py
*font: helvetica 10
*header.font: helvetica 18 bold
*subtitle.font: helvetica 14 italic
*Button.foreground: blue
*Button.background: white
*Button.activeBackground: gray
*Button.activeForeground: black
```

这个文件应该使用`option_readfile()`方法加载到应用程序中，并替换所有对`option_add()`的调用。在我们的示例中，假设文件名为`my_options_file`，并且它放在与我们的脚本相同的目录中：

```py
def __init__(self):
        super().__init__()
        self.title("Options demo")
        self.option_readfile("my_options_file")
        # ...
```

如果文件不存在或其格式无效，Tkinter 将引发`TclError`。

# 另请参阅

+   使用颜色

+   设置小部件字体

# 更改光标图标

Tkinter 允许您在悬停在小部件上时自定义光标图标。这种行为有时是默认启用的，比如显示 I 型光标的 Entry 小部件。

# 准备工作

以下应用程序显示了如何在执行长时间操作时显示繁忙光标，以及在帮助菜单中通常使用的带有问号的光标：

![](img/c660453b-6895-4f3e-a6ed-87a59c90693d.png)

# 如何做...

鼠标指针图标可以使用`cursor`选项更改。在我们的示例中，我们使用`watch`值来显示本机繁忙光标，`question_arrow`来显示带有问号的常规箭头：

```py
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cursors demo")
        self.resizable(0, 0)
        self.label = tk.Label(self, text="Click the button to start")
        self.btn_launch = tk.Button(self, text="Start!",
                                    command=self.perform_action)
        self.btn_help = tk.Button(self, text="Help",
                                  cursor="question_arrow")

        btn_opts = {"side": tk.LEFT, "expand":True, "fill": tk.X,
                    "ipadx": 30, "padx": 20, "pady": 5}
        self.label.pack(pady=10)
        self.btn_launch.pack(**btn_opts)
        self.btn_help.pack(**btn_opts)

    def perform_action(self):
        self.config(cursor="watch")
        self.btn_launch.config(state=tk.DISABLED)
        self.btn_help.config(state=tk.DISABLED)
        self.label.config(text="Working...")
        self.after(3000, self.end_action)

    def end_action(self):
        self.config(cursor="arrow")
        self.btn_launch.config(state=tk.NORMAL)
        self.btn_help.config(state=tk.NORMAL)
        self.label.config(text="Done!")

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

您可以在官方 Tcl/Tk 文档的[`www.tcl.tk/man/tcl/TkCmd/cursors.htm`](https://www.tcl.tk/man/tcl/TkCmd/cursors.htm)中查看有效`cursor`值和特定于系统的完整列表。

# 它是如何工作的...

如果一个小部件没有指定`cursor`选项，它将采用父容器中定义的值。因此，我们可以通过在`root`窗口级别设置它来轻松地将其应用于所有小部件。这是通过在`perform_action()`方法中调用`set_watch_cursor()`来完成的：

```py
def perform_action(self):
    self.config(cursor="watch")
    # ...
```

这里的例外是`Help`按钮，它明确将光标设置为`question_arrow`。此选项也可以在实例化小部件时直接设置：

```py
self.btn_help = tk.Button(self, text="Help",
                          cursor="question_arrow")
```

# 还有更多...

请注意，如果在调用预定方法之前单击`Start!`按钮并将鼠标放在`Help`按钮上，光标将显示为`help`而不是`watch`。这是因为如果小部件的`cursor`选项已设置，它将优先于父容器中定义的`cursor`。

为了避免这种情况，我们可以保存当前的`cursor`值并将其更改为`watch`，然后稍后恢复它。执行此操作的函数可以通过迭代`winfo_children()`列表在子小部件中递归调用：

```py
def perform_action(self):
    self.set_watch_cursor(self)
    # ...

def end_action(self):
 self.restore_cursor(self)
    # ...

def set_watch_cursor(self, widget):
    widget._old_cursor = widget.cget("cursor")
    widget.config(cursor="watch")
    for w in widget.winfo_children():
        self.set_watch_cursor(w)

def restore_cursor(self, widget):
    widget.config(cursor=widget._old_cursor)
    for w in widget.winfo_children():
        self.restore_cursor(w)
```

在前面的代码中，我们为每个小部件添加了`_old_cursor`属性，因此如果您遵循类似的方法，请记住在`set_watch_cursor()`之前不能调用`restore_cursor()`。

# 介绍 Text 小部件

Text 小部件提供了与其他小部件类相比更高级的功能。它显示可编辑文本的多行，可以按行和列进行索引。此外，您可以使用标签引用文本范围，这些标签可以定义自定义外观和行为。

# 准备工作

以下应用程序展示了 `Text` 小部件的基本用法，您可以动态插入和删除文本，并检索所选内容：

![](img/a753a28a-5bb3-434a-8efd-712c2e758a2e.png)

# 如何做...

除了 `Text` 小部件，我们的应用程序还包含三个按钮，这些按钮调用方法来清除整个文本内容，在当前光标位置插入`"Hello, world"`字符串，并打印用鼠标或键盘进行的当前选择：

```py
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Text demo")
        self.resizable(0, 0)
        self.text = tk.Text(self, width=50, height=10)
        self.btn_clear = tk.Button(self, text="Clear text",
                                   command=self.clear_text)
        self.btn_insert = tk.Button(self, text="Insert text",
                                    command=self.insert_text)
        self.btn_print = tk.Button(self, text="Print selection",
                                   command=self.print_selection)
        self.text.pack()
        self.btn_clear.pack(side=tk.LEFT, expand=True, pady=10)
        self.btn_insert.pack(side=tk.LEFT, expand=True, pady=10)
        self.btn_print.pack(side=tk.LEFT, expand=True, pady=10)

    def clear_text(self):
        self.text.delete("1.0", tk.END)

    def insert_text(self):
        self.text.insert(tk.INSERT, "Hello, world")

    def print_selection(self):
        selection = self.text.tag_ranges(tk.SEL)
        if selection:
            content = self.text.get(*selection)
            print(content)

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

# 它是如何工作的...

我们的 `Text` 小部件最初是空的，宽度为 50 个字符，高度为 10 行。除了允许用户输入任何类型的文本，我们还将深入研究每个按钮使用的方法，以更好地了解如何与这个小部件交互。

`delete(start, end)` 方法从 `start` 索引到 `end` 索引删除内容。如果省略第二个参数，它只删除 `start` 位置的字符。

在我们的示例中，我们通过从 `1.0` 索引（第一行的第 0 列）调用此方法到 `tk.END` 索引（指向最后一个字符）来删除所有文本：

```py
def clear_text(self):
    self.text.delete("1.0", tk.END)
```

`insert(index, text)` 方法在`index`位置插入给定的文本。在这里，我们使用`INSERT`索引调用它，该索引对应于插入光标的位置：

```py
def insert_text(self):
    self.text.insert(tk.INSERT, "Hello, world")
```

`tag_ranges(tag)` 方法返回一个元组，其中包含给定 `tag` 的所有范围的第一个和最后一个索引。我们使用特殊的 `tk.SEL` 标签来引用当前选择。如果没有选择，这个调用会返回一个空元组。这与 `get(start, end)` 方法结合使用，该方法返回给定范围内的文本：

```py
def print_selection(self):
    selection = self.text.tag_ranges(tk.SEL)
    if selection:
        content = self.text.get(*selection)
        print(content)
```

由于 `SEL` 标签只对应一个范围，我们可以安全地解包它来调用 `get` 方法。

# 向 Text 小部件添加标记

在本示例中，您将学习如何配置 `Text` 小部件中标记的字符范围的行为。

所有的概念都与适用于常规小部件的概念相同，比如事件序列或配置选项，这些概念在之前的示例中已经涵盖过了。主要的区别是，我们需要使用文本索引来识别标记的内容，而不是使用对象引用。

# 准备工作

为了说明如何使用文本标记，我们将创建一个模拟插入超链接的 `Text` 小部件。点击时，此链接将使用默认浏览器打开所选的 URL。

例如，如果用户输入以下内容，`python.org` 文本可以被标记为超链接：

![](img/8a673cd4-d0d1-438b-b1d9-44ae42a5aa4e.png)

# 如何做...

对于此应用程序，我们将定义一个名为`"link"`的标记，它表示可点击的超链接。此标记将被添加到当前选择中，鼠标点击将触发打开浏览器中的链接的事件：

```py
import tkinter as tk
import webbrowser

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Text tags demo")
        self.text = tk.Text(self, width=50, height=10)
        self.btn_link = tk.Button(self, text="Add hyperlink",
                                  command=self.add_hyperlink)

        self.text.tag_config("link", foreground="blue", underline=1)
        self.text.tag_bind("link", "<Button-1>", self.open_link)
        self.text.tag_bind("link", "<Enter>",
                           lambda _: self.text.config(cursor="hand2"))
        self.text.tag_bind("link", "<Leave>",
                           lambda e: self.text.config(cursor=""))

        self.text.pack()
        self.btn_link.pack(expand=True)

    def add_hyperlink(self):
        selection = self.text.tag_ranges(tk.SEL)
        if selection:
            self.text.tag_add("link", *selection)

    def open_link(self, event):
        position = "@{},{} + 1c".format(event.x, event.y)
        index = self.text.index(position)
        prevrange = self.text.tag_prevrange("link", index)
        url = self.text.get(*prevrange)
        webbrowser.open(url)

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

# 它是如何工作的...

首先，我们将通过配置颜色和下划线样式来初始化标记。我们添加事件绑定来使用浏览器打开点击的文本，并在鼠标悬停在标记文本上时改变光标外观：

```py
def __init__(self):
    # ...
    self.text.tag_config("link", foreground="blue", underline=1)
    self.text.tag_bind("link", "<Button-1>", self.open_link)
    self.text.tag_bind("link", "<Enter>",
                       lambda e: self.text.config(cursor="hand2"))
    self.text.tag_bind("link", "<Leave>",
                       lambda e: self.text.config(cursor=""))
```

在 `open_link` 方法中，我们使用 `Text` 类的 `index` 方法将点击的位置转换为相应的行和列：

```py
position = "@{},{} + 1c".format(event.x, event.y)
index = self.text.index(position)
prevrange = self.text.tag_prevrange("link", index)
```

请注意，与点击的索引对应的位置是`"@x,y"`，但我们将其移动到下一个字符。我们这样做是因为 `tag_prevrange` 返回给定索引的前一个范围，因此如果我们点击第一个字符，它将不返回当前范围。

最后，我们将从范围中检索文本，并使用 `webbrowser` 模块的 `open` 函数在默认浏览器中打开它：

```py
url = self.text.get(*prevrange)
webbrowser.open(url)
```

# 还有更多...

由于 `webbrowser.open` 函数不检查 URL 是否有效，可以通过包含基本的超链接验证来改进此应用程序。例如，您可以使用 `urlparse` 函数来验证 URL 是否具有网络位置：

```py
from urllib.parse import urlparse def validate_hyperlink(self, url):
    return urlparse(url).netloc
```

尽管这个解决方案并不打算处理一些特殊情况，但它可能作为丢弃大多数无效 URL 的第一步。

一般来说，您可以使用标签来创建复杂的基于文本的程序，比如带有语法高亮的 IDE。事实上，IDLE——默认的 Python 实现中捆绑的——就是基于 Tkinter 的。

# 另请参阅

+   *更改光标图标*食谱

+   *介绍文本小部件*食谱
