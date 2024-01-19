# 使用Menu和Tkinter对话框创建菜单

随着应用程序的增长，组织对其功能的访问变得越来越重要。传统上，应用程序通过**菜单系统**来解决这个问题，通常位于应用程序窗口的顶部或（在某些平台上）全局桌面菜单中。虽然这些菜单是特定于应用程序的，但已经制定了一些组织惯例，我们应该遵循以使我们的软件更加用户友好。

在本章中，我们将涵盖以下主题：

+   分析一些报告的问题并决定解决方案

+   探索一些Tkinter的对话框类，并使用它们来实现常见菜单功能

+   学习如何使用Tkinter的Menu小部件，并使用它为我们的应用程序创建菜单

+   为我们的应用程序创建一些选项并将它们保存到磁盘

# 解决我们应用程序中的问题

您的老板给您带来了需要在您的应用程序中解决的第一组问题。首先，在无法在第二天之前输入当天最后的报告的情况下，文件名中的硬编码日期字符串是一个问题。数据输入人员需要一种手动选择要追加的文件的方法。

此外，数据输入人员对表单中的自动填充功能有不同的看法。有些人觉得这非常有帮助，但其他人真的希望看到它被禁用。您需要一种允许用户打开和关闭此功能的方法。

最后，一些用户很难注意到底部状态栏的文本，并希望应用程序在由于错误而无法保存数据时更加显眼。

# 决定如何解决这些问题

很明显，您需要实现一种选择文件和切换表单自动填充功能的方法。首先，您考虑只向主应用程序添加这两个控件，并进行快速的模拟：

![](assets/b7aaf697-23dc-40a6-8c42-0e8c13ba42d7.png)

您很快就会意识到这不是一个很好的设计，当然也不是一个能够适应增长的设计。您的用户不想盲目地在框中输入文件路径和文件名，也不想让很多额外的字段混乱UI。

幸运的是，Tkinter提供了一些工具，可以帮助我们解决这些问题：

+   **文件对话框**：Tkinter的`filedialog`库将帮助简化文件选择

+   **错误对话框**：Tkinter的`messagebox`库将让我们更加显眼地显示错误消息

+   **主菜单**：Tkinter的`Menu`类可以帮助我们组织常见功能，以便轻松访问

# 实现简单的Tkinter对话框

状态栏适用于不应中断用户工作流程的偶发信息，但对于阻止工作按预期继续的错误，用户应该以更有力的方式受到警告。一个中断程序直到通过鼠标点击确认的**错误对话框**是相当有力的，似乎是解决用户看不到错误的问题的好方法。为了实现这些，您需要了解Tkinter的`messagebox`库。

# Tkinter messagebox

在Tkinter中显示简单对话框的最佳方法是使用`tkinter.messagebox`库，其中包含几个方便的函数，允许您快速创建常见的对话框类型。每个函数显示一个预设的图标和一组按钮，带有您指定的消息和详细文本，并根据用户点击的按钮返回一个值。

以下表格显示了一些`messagebox`函数及其图标和返回值：

| **函数** | **图标** | **按钮** / **返回值** |
| --- | --- | --- |
| `askokcancel` | 问题 | 确定 (`True`), 取消 (`False`) |
| `askretrycancel` | 警告 | 重试 (`True`), 取消 (`False`) |
| `askyesno` | 问题 | 是 (`True`), 否 (`False`) |
| `askyesnocancel` | 问题 | 是 (`True`), 否 (`False`), 取消 (`None`) |
| `showerror` | 错误 | 确定 (`ok`) |
| `showinfo` | 信息 | 确定（`ok`） |
| `showwarning` | 警告 | 确定（`ok`） |

我们可以将以下三个文本参数传递给任何`messagebox`函数：

+   `title`：此参数设置窗口的标题，在您的桌面环境中显示在标题栏和/或任务栏中。

+   `message`：此参数设置对话框的主要消息。通常使用标题字体，应保持相当简短。

+   `detail`：此参数设置对话框的正文文本，通常显示在标准窗口字体中。

这是对`showinfo()`的基本调用：

```py
messagebox.showinfo(
    title='This is the title',
    message="This is the message",
    detail='This is the detail')
```

在Windows 10中，它会导致一个对话框（在其他平台上可能看起来有点不同），如下面的屏幕截图所示：

![](assets/3ae47462-cc7e-4428-bee7-9a8e23f15ea4.png)

Tkinter的`messagebox`对话框是**模态**的，这意味着程序执行会暂停，而UI的其余部分在对话框打开时无响应。没有办法改变这一点，所以只能在程序暂停执行时使用它们。

让我们创建一个小例子来展示`messagebox`函数的使用：

```py
import tkinter as tk
from tkinter import messagebox
```

要使用`messagebox`，我们需要从Tkinter导入它；你不能简单地使用`tk.messagebox`，因为它是一个子模块，必须显式导入。

让我们创建一个是-否消息框，如下所示：

```py
see_more = messagebox.askyesno(title='See more?',
    message='Would you like to see another box?',
    detail='Click NO to quit')
if not see_more:
    exit()
```

这将创建一个带有是和否按钮的对话框；如果点击是，函数返回`True`。如果点击否，函数返回`False`，应用程序退出。

如果我们的用户想要看到更多的框，让我们显示一个信息框：

```py
messagebox.showinfo(title='You got it',
    message="Ok, here's another dialog.",
    detail='Hope you like it!')
```

注意`message`和`detail`在您的平台上显示方式的不同。在某些平台上，没有区别；在其他平台上，`message`是大而粗体的，这对于短文本是合适的。对于跨平台软件，最好使用`detail`进行扩展输出。

# 显示错误对话框

现在您了解了如何使用`messagebox`，错误对话框应该很容易实现。`Application.on_save()`方法已经在状态栏中显示错误；我们只需要通过以下步骤使此错误显示在错误消息框中：

1.  首先，我们需要在`application.py`中导入它，如下所示：

```py
from tkinter import messagebox
```

1.  现在，在`on_save()`方法中检查错误后，我们将设置错误对话框的消息。我们将通过使用`"\n *"`将错误字段制作成项目符号列表。不幸的是，`messagebox`不支持任何标记，因此需要使用常规字符手动构建类似项目符号列表的结构，如下所示：

```py
        message = "Cannot save record"
        detail = "The following fields have errors: \n  * {}".format(
            '\n  * '.join(errors.keys()))
```

1.  现在，我们可以在`status()`调用之后调用`showerror()`，如下所示：

```py
        messagebox.showerror(title='Error', message=message, detail=detail)
```

1.  现在，打开程序并点击保存；您将看到一个对话框，提示应用程序中的错误，如下面的屏幕截图所示：

![](assets/b7f981fb-dc26-4abf-9ff9-b839d434646d.png)

这个错误应该对任何人来说都很难错过！

`messagebox`对话框的一个缺点是它们不会滚动；长错误消息将创建一个可能填满（或超出）屏幕的对话框。如果这是一个潜在的问题，您将需要创建一个包含可滚动小部件的自定义对话框。

# 设计我们的菜单

大多数应用程序将功能组织成一个分层的**菜单系统**，通常显示在应用程序或屏幕的顶部（取决于操作系统）。虽然这个菜单的组织在操作系统之间有所不同，但某些项目在各个平台上都是相当常见的。

在这些常见项目中，我们的应用程序将需要以下内容：

+   包含文件操作（如打开/保存/导出）的文件菜单，通常还有退出应用程序的选项。我们的用户将需要此菜单来选择文件并退出程序。

+   一个选项、首选项或设置菜单，用户可以在其中配置应用程序。我们将需要此菜单来进行切换设置；暂时我们将其称为选项。

+   帮助菜单，其中包含指向帮助文档的链接，或者至少包含一个关于应用程序的基本信息的消息。我们将为关于对话框实现这个菜单。

苹果、微软和Gnome项目分别发布了macOS、Windows和Gnome桌面（在Linux和BSD上使用）的指南；每套指南都涉及特定平台的菜单布局。

在我们实现菜单之前，我们需要了解Tkinter中菜单的工作原理。

# 在Tkinter中创建菜单

`tkinter.Menu`小部件用于在Tkinter应用程序中实现菜单；它是一个相当简单的小部件，作为任意数量的菜单项的容器。

菜单项可以是以下五种类型之一：

+   `command`：这些项目是带有标签的按钮，当单击时运行回调。

+   `checkbutton`：这些项目就像我们表单中的`Checkbutton`一样，可以用来切换`BooleanVar`。

+   `radiobutton`：这些项目类似于`Checkbutton`，但可以用来在几个互斥选项之间切换任何类型的Tkinter变量。

+   `separator`：这些项目用于将菜单分成几个部分。

+   `cascade`：这些项目允许您向菜单添加子菜单。子菜单只是另一个`tkinter.Menu`对象。

让我们编写以下小程序来演示Tkinter菜单的使用：

```py
import tkinter as tk

root = tk.Tk()
main_text = tk.StringVar(value='Hi')
label = tk.Label(root, textvariable=main_text)
label.pack()

root.mainloop()
```

该应用程序设置了一个标签，其文本由字符串变量`main_text`控制。如果您运行此应用程序，您将看到一个简单的窗口，上面写着Hi。让我们开始添加菜单组件。

在`root.mainloop()`的正上方，添加以下代码：

```py
main_menu = tk.Menu(root)
root.config(menu=main_menu)
```

这将创建一个主菜单，然后将其设置为我们应用程序的主菜单。

目前，该菜单是空的，所以让我们通过添加以下代码来添加一个项目：

```py
main_menu.add('command', label='Quit', command=root.quit)
```

我们已经添加了一个退出应用程序的命令。`add`方法允许我们指定一个项目类型和任意数量的属性来创建一个新的菜单项。对于命令，我们至少需要有一个`label`参数来指定菜单中显示的文本，以及一个指向Python回调的`command`参数。

一些平台，如macOS，不允许在顶级菜单中使用命令。

让我们尝试创建一个子菜单，如下所示：

```py
text_menu = tk.Menu(main_menu, tearoff=False)
```

创建子菜单就像创建菜单一样，只是我们将`parent`菜单指定为小部件的`parent`。注意`tearoff`参数；在Tkinter中，默认情况下子菜单是可撕下的，这意味着它们可以被拆下并作为独立窗口移动。您不必禁用此选项，但这是一个相当古老的UI功能，在现代平台上很少使用。用户可能会觉得困惑，最好在创建子菜单时禁用它。

添加一些命令到菜单中，如下所示：

```py
text_menu.add_command(label='Set to "Hi"',
              command=lambda: main_text.set('Hi'))
text_menu.add_command(label='Set to "There"',
              command=lambda: main_text.set('There'))
```

我们在这里使用`lambda`函数是为了方便，但您可以传递任何Python可调用的函数。这里使用的`add_command`方法只是`add('command')`的快捷方式。添加其他项目的方法也是类似的（级联，分隔符等）。

让我们使用`add_cascade`方法将我们的菜单添加回其`parent`小部件，如下所示：

```py
main_menu.add_cascade(label="Text", menu=text_menu)
```

在将子菜单添加到其`parent`菜单时，我们只需提供菜单的标签和菜单本身。

我们也可以将`Checkbutton`和`Radiobutton`小部件添加到菜单中。为了演示这一点，让我们创建另一个子菜单来改变标签的外观。

首先，我们需要以下设置代码：

```py
font_bold = tk.BooleanVar()
font_size = tk.IntVar()

def set_font(*args):
    font_spec = 'TkDefaultFont {size} {bold}'.format(
        size=font_size.get(),
        bold='bold' if font_bold.get() else '')
    label.config(font=font_spec)

font_bold.trace('w', set_font)
font_size.trace('w', set_font)
```

在这里，我们只是创建变量来存储粗体选项和字体大小的状态，然后创建一个回调方法，当调用时实际上从这些变量设置标签的字体。然后，我们在两个变量上设置了一个跟踪，以便在它们的值发生变化时调用回调。

现在，我们只需要通过添加以下代码来创建菜单选项来改变变量：

```py
# appearance menu
appearance_menu = tk.Menu(main_menu, tearoff=False)
main_menu.add_cascade(label="Appearance", menu=appearance_menu)

# bold text button
appearance_menu.add_checkbutton(label="Bold", variable=font_bold)
```

像普通的`Checkbutton`小部件一样，`add_checkbutton`方法接受`BooleanVar`，它被传递给`variable`参数，该参数将绑定到其选中状态。与普通的`Checkbutton`小部件不同，使用`label`参数而不是`text`参数来分配标签文本。

为了演示单选按钮，让我们向我们的子菜单添加一个子菜单，如下所示：

```py
size_menu = tk.Menu(appearance_menu, tearoff=False)
appearance_menu.add_cascade(label='Font size', menu=size_menu)
for size in range(8, 24, 2):
    size_menu.add_radiobutton(label="{} px".format(size),
        value=size, variable=font_size)
```

就像我们在主菜单中添加了一个子菜单一样，我们也可以在子菜单中添加子菜单。理论上，你可以无限嵌套子菜单，但大多数UI指南不鼓励超过两个级别。为了创建我们的大小菜单项，我们只需迭代一个在8和24之间生成的偶数列表；对于每一个，我们都添加一个值等于该大小的`radiobutton`项。就像普通的`Radiobutton`小部件一样，`variable`参数中给定的变量在按钮被选中时将被更新为`value`参数中给定的值。

启动应用程序并尝试一下，如下面的屏幕截图所示：

![](assets/18dd3562-6645-4d90-84e7-3fd42afc8a8f.png)

现在你了解了`Menu`小部件，让我们在我们的应用程序中添加一个。

# 实现我们的应用程序菜单

作为GUI的一个重要组件，我们的菜单显然是一个视图，并且应该在`views.py`文件中实现。但是，它还需要设置影响其他视图的选项（例如我们现在正在实现的表单选项）并运行影响应用程序的函数（如退出）。我们需要以这样一种方式实现它，即我们将控制器函数保留在`Application`类中，但仍将UI代码保留在`views.py`中。让我们看看以下步骤：

1.  让我们首先打开`views.py`并创建一个继承了`tkinter.Menu`的`MainMenu`类：

```py
class MainMenu(tk.Menu):
"""The Application's main menu"""
```

我们重写的`__init__()`方法将使用两个字典，`settings`字典和`callbacks`字典，如下所示：

```py
    def __init__(self, parent, settings, callbacks, **kwargs):
        super().__init__(parent, **kwargs)
```

我们将使用这些字典与控制器进行通信：`settings`将包含可以绑定到我们菜单控件的Tkinter变量，`callbacks`将是我们可以绑定到菜单命令的控制器方法。当然，我们需要确保在我们的`Application`对象中使用预期的变量和可调用对象来填充这些字典。

1.  现在，让我们开始创建我们的子菜单，首先是文件菜单如下：

```py
        file_menu = tk.Menu(self, tearoff=False)
        file_menu.add_command(
            label="Select file…",
            command=callbacks['file->open'])
```

我们文件菜单中的第一个命令是“选择文件...”。注意标签中的省略号：这向用户表明该选项将打开另一个需要进一步输入的窗口。我们将`command`设置为从我们的`callbacks`字典中使用`file->open`键的引用。这个函数还不存在；我们将很快实现它。让我们添加我们的下一个文件菜单命令，`file->quit`：

```py
        file_menu.add_separator()
        file_menu.add_command(label="Quit",
                command=callbacks['file->quit'])
```

再次，我们将这个命令指向了一个尚未定义的函数，它在我们的`callbacks`字典中。我们还添加了一个分隔符；由于退出程序与选择目标文件是一种根本不同的操作，将它们分开是有意义的，你会在大多数应用程序菜单中看到这一点。

1.  这完成了文件菜单，所以我们需要将它添加到主`menu`对象中，如下所示：

```py
        self.add_cascade(label='File', menu=file_menu)
```

1.  我们需要创建的下一个子菜单是我们的“选项”菜单。由于我们只有两个菜单选项，我们将直接将它们添加到子菜单中作为`Checkbutton`。选项菜单如下所示：

```py
    options_menu = tk.Menu(self, tearoff=False)
    options_menu.add_checkbutton(label='Autofill Date',
        variable=settings['autofill date'])
    options_menu.add_checkbutton(label='Autofill Sheet data',
        variable=settings['autofill sheet data'])
    self.add_cascade(label='Options', menu=options_menu)
```

绑定到这些`Checkbutton`小部件的变量在`settings`字典中，因此我们的`Application`类将用两个`BooleanVar`变量填充`settings`：`autofill date`和`autofill sheet data`。

1.  最后，我们将创建一个“帮助”菜单，其中包含一个显示“关于”对话框的选项：

```py
        help_menu = tk.Menu(self, tearoff=False)
        help_menu.add_command(label='About…', command=self.show_about)
        self.add_cascade(label='Help', menu=help_menu)
```

我们的“关于”命令指向一个名为`show_about`的内部`MainMenu`方法，我们将在下面实现。关于对话框将是纯UI代码，没有实际的应用程序功能，因此我们可以完全在视图中实现它。

# 显示关于对话框

我们已经看到如何使用`messagebox`来创建错误对话框。现在，我们可以应用这些知识来创建我们的`About`框，具体步骤如下：

1.  在`__init__()`之后开始一个新的方法定义：

```py
    def show_about(self):
        """Show the about dialog"""
```

1.  `About`对话框可以显示您认为相关的任何信息，包括您的联系信息、支持信息、版本信息，甚至整个`README`文件。在我们的情况下，我们会保持它相当简短。让我们指定`message`标题文本和`detail`正文文本：

```py
        about_message = 'ABQ Data Entry'
        about_detail = ('by Alan D Moore\n'
            'For assistance please contact the author.')
```

我们只是在标题中使用应用程序名称，然后在详细信息中简要介绍我们的姓名以及联系支持的方式。请随意在您的`About`框中放入任何文本。

在Python代码中，有几种处理长的多行字符串的方法；这里使用的方法是在括号之间放置多个字符串，它们之间只有空格。Python会自动连接只有空格分隔的字符串，因此对Python来说，这看起来像是一组括号内的单个长字符串。与其他方法相比，例如三引号，这允许您保持清晰的缩进并明确控制换行。

1.  最后，我们需要显示我们的`About`框如下：

```py
        messagebox.showinfo(title='About', message=about_message,  
            detail=about_detail)
```

在上述代码中，`showinfo()`函数显然是最合适的，因为我们实际上是在显示信息。这完成了我们的`show_about()`方法和我们的`MainMenu`类。接下来，我们需要对`Application`进行必要的修改以使其正常工作。

# 在控制器中添加菜单功能

现在我们的菜单类已经定义，我们的`Application`对象需要创建一个实例并将其添加到主窗口中。在我们这样做之前，我们需要定义一些`MainMenu`类需要的东西。

从上一节中记住以下事项：

+   我们需要一个包含我们两个设置选项的Tkinter变量的`settings`字典

+   我们需要一个指向`file->select`和`file->quit`回调的`callbacks`字典

+   我们需要实际实现文件选择和退出的函数

让我们定义一些`MainMenu`类需要的东西。

打开`application.py`，让我们在创建`self.recordform`之前开始添加代码：

```py
    self.settings = {
        'autofill date': tk.BooleanVar(),
        'autofill sheet data': tk.BooleanVar()
    }
```

这将是我们的全局设置字典，用于存储两个配置选项的布尔变量。接下来，我们将创建`callbacks`字典：

```py
    self.callbacks = {
        'file->select': self.on_file_select,
        'file->quit': self.quit
    }
```

在这里，我们将我们的两个回调指向`Application`类的方法，这些方法将实现功能。对我们来说，幸运的是，Tkinter已经实现了`self.quit`，它确实做了您期望它做的事情，因此我们只需要自己实现`on_file_select`。我们将通过创建我们的`menu`对象并将其添加到应用程序来完成这里：

```py
    menu = v.MainMenu(self, self.settings, self.callbacks)
    self.config(menu=menu)
```

# 处理文件选择

当用户需要输入文件或目录路径时，首选的方法是显示一个包含迷你文件浏览器的对话框，通常称为文件对话框。与大多数工具包一样，Tkinter为我们提供了用于打开文件、保存文件和选择目录的对话框。这些都是`filedialog`模块的一部分。

就像`messagebox`一样，`filedialog`是一个Tkinter子模块，需要显式导入才能使用。与`messagebox`一样，它包含一组方便的函数，用于创建适合不同场景的文件对话框。

以下表格列出了函数、它们的返回值和它们的UI特性：

| **功能** | **返回值** | **特点** |
| --- | --- | --- |
| `askdirectory` | 目录路径字符串 | 仅显示目录，不显示文件 |
| `askopenfile` | 文件句柄对象 | 仅允许选择现有文件 |
| `askopenfilename` | 文件路径字符串 | 仅允许选择现有文件 |
| `askopenfilenames` | 字符串列表的文件路径 | 类似于`askopenfilename`，但允许多个选择 |
| `askopenfiles` | 文件句柄对象列表 | 类似于`askopenfile`，但允许多个选择 |
| `asksaveasfile` | 文件句柄对象 | 允许创建新文件，在现有文件上进行确认提示 |
| `asksaveasfilename` | 文件路径字符串 | 允许创建新文件，在现有文件上进行确认提示 |

正如您所看到的，每个文件选择对话框都有两个版本：一个返回路径作为字符串，另一个返回打开的文件对象。

每个函数都可以使用以下常见参数：

+   `title`：此参数指定对话框窗口标题。

+   `parent`：此参数指定（可选的）`parent`小部件。文件对话框将出现在此小部件上方。

+   `initialdir`：此参数是文件浏览器应该开始的目录。

+   `filetypes`：此参数是一个元组列表，每个元组都有一个标签和匹配模式，用于创建过滤下拉类型的文件，通常在文件名输入框下方看到。这用于将可见文件过滤为仅由应用程序支持的文件。

`asksaveasfile`和`asksaveasfilename`方法还接受以下两个附加选项：

+   `initialfile`：此选项是要选择的默认文件路径

+   `defaultextension`：此选项是一个文件扩展名字符串，如果用户没有这样做，它将自动附加到文件名

最后，返回文件对象的方法接受一个指定文件打开模式的`mode`参数；这些是Python的`open`内置函数使用的相同的一到两个字符字符串。

我们的应用程序需要使用哪个对话框？让我们考虑一下我们的需求：

+   我们需要一个对话框，允许我们选择一个现有文件

+   我们还需要能够创建一个新文件

+   由于打开文件是模型的责任，我们只想获得一个文件名传递给模型

这些要求清楚地指向了`asksaveasfilename`函数。让我们看看以下步骤：

1.  在`Application`对象上启动一个新方法：

```py
    def on_file_select(self):
    """Handle the file->select action from the menu"""

    filename = filedialog.asksaveasfilename(
        title='Select the target file for saving records',
        defaultextension='.csv',
        filetypes=[('Comma-Separated Values', '*.csv *.CSV')])
```

该方法首先要求用户选择一个具有`.csv`扩展名的文件；使用`filetypes`参数，现有文件的选择将被限制为以`.csv`或CSV结尾的文件。对话框退出时，函数将将所选文件的路径作为字符串返回给`filename`。不知何故，我们必须将此路径传递给我们的模型。

1.  目前，文件名是在`Application`对象的`on_save`方法中生成并传递到模型中。我们需要将`filename`移动到`Application`对象的属性中，以便我们可以从我们的`on_file_select()`方法中覆盖它。

1.  回到`__init__()`方法，在`settings`和`callbacks`定义之前添加以下代码行：

```py
        self.filename = tk.StringVar()
```

1.  `self.filename`属性将跟踪当前选择的保存文件。以前，我们在`on_save()`方法中设置了我们的硬编码文件名；没有理由每次调用`on_save()`时都这样做，特别是因为我们只在用户没有选择文件的情况下使用它。相反，将这些行从`on_save()`移到`self.filename`定义的上方：

```py
    datestring = datetime.today().strftime("%Y-%m-%d")
    default_filename = "abq_data_record_{}.csv".
    format(datestring)
    self.filename = tk.StringVar(value=default_filename)
```

1.  定义了默认文件名后，我们可以将其作为`StringVar`的默认值提供。每当用户选择文件名时，`on_file_select()`将更新该值。这是通过`on_file_select()`末尾的以下行完成的：

```py
    if filename:
        self.filename.set(filename)
```

1.  `if`语句的原因是，我们只想在用户实际选择了文件时才设置一个值。请记住，如果用户取消操作，文件对话框将返回`None`；在这种情况下，用户希望当前设置的文件名仍然是目标。

1.  最后，当设置了这个值时，我们需要让我们的`on_save()`方法使用它，而不是硬编码的默认值。

1.  在`on_save()`方法中，找到定义`filename`的行，并将其更改为以下行：

```py
    filename = self.filename.get()
```

1.  这完成了代码更改，使文件名选择起作用。此时，您应该能够运行应用程序并测试文件选择功能。保存几条记录并注意它们确实保存到您选择的文件中。

# 使我们的设置生效

虽然文件保存起作用，但设置却没有。`settings`菜单项应该按预期工作，保持选中或取消选中，但它们尚未改变数据输入表单的行为。让我们让它起作用。

请记住，`DataRecordForm`类的`reset()`方法中实现了两个自动填充功能。为了使用我们的新设置，我们需要通过以下步骤让我们的表单访问`settings`字典：

1.  打开`views.py`并更新`DataRecordForm.__init__()`方法如下：

```py
    def __init__(self, parent, fields, settings, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.settings = settings
```

1.  我们添加了一个额外的位置参数`settings`，然后将其设置为`self.settings`，以便类中的所有方法都可以访问它。现在，看一下`reset()`方法；目前，日期自动填充代码如下：

```py
        current_date = datetime.today().strftime('%Y-%m-%d')
        self.inputs['Date'].set(current_date)
        self.inputs['Time'].input.focus()
```

1.  我们只需要确保这仅在`settings['autofill date']`为`True`时发生：

```py
 if self.settings['autofill date'].get():
        current_date = datetime.today().strftime('%Y-%m-%d')
        self.inputs['Date'].set(current_date)
        self.inputs['Time'].input.focus()
```

表格数据的自动填充已经在条件语句下，如下所示：

```py
    if plot not in ('', plot_values[-1]):
        self.inputs['Lab'].set(lab)
        self.inputs['Time'].set(time)
       ...
```

1.  为了使设置生效，我们只需要在`if`语句中添加另一个条件：

```py
    if (self.settings['autofill sheet data'].get() and
        plot not in ('', plot_values[-1])):
        ...
```

最后一部分的难题是确保我们在创建`DataRecordForm`时将我们的`settings`字典发送到`DataRecordForm`。

1.  回到`Application`代码，更新我们对`DataRecordForm()`的调用，包括`self.settings`如下：

```py
        self.recordform = v.DataRecordForm(self, 
            m.CSVModel.fields, self.settings)
```

1.  现在，如果运行程序，您应该会发现设置得到了尊重；尝试勾选和取消勾选它们，然后保存记录后查看发生了什么。

# 持久化设置

我们的设置有效，但有一个主要的烦恼：它们在会话之间不持久。关闭应用程序并重新启动，您会发现设置恢复为默认值。这不是一个主要问题，但这是一个我们不应该留给用户的粗糙边缘。

Python为我们提供了各种将数据持久保存在文件中的方法。我们已经体验过CSV，它是为表格数据设计的；还有其他设计用于不同功能的格式。

以下表格仅显示了Python标准库中可用的存储数据选项中的一些选项：

| **库** | **数据类型** | **适用** | **优点** | **缺点** |
| --- | --- | --- | --- | --- |
| `pickle` | 二进制 | 任何类型的对象 | 快速、简单、文件小 | 不安全，文件不易读，必须读取整个文件 |
| `configparser` | 文本 | `key->value`对 | 快速、简单、易读的文件 | 无法处理序列或复杂对象，层次有限 |
| `json` | 文本 | 简单值和序列 | 广泛使用，易读的文件 | 无法序列化复杂对象而不经修改 |
| `xml` | 文本 | 任何类型的Python对象 | 强大、灵活、大部分易读的文件 | 不安全，使用复杂，文件语法冗长 |
| `sqlite` | 二进制 | 关系数据 | 快速而强大的文件 | 需要SQL知识，对象必须转换为表 |

如果这还不够，第三方库中甚至还有更多选项可用。几乎任何一个都适合存储一些布尔值，那么我们该如何选择呢？

+   SQL和XML功能强大，但对于我们这里的简单需求来说太复杂了。

+   我们希望坚持使用文本格式，以防需要调试损坏的设置文件，因此`pickle`不适用。

+   `configparser`现在可以工作了，但它无法处理列表、元组和字典，这在将来可能会有限制。

+   这留下了`json`，这是一个不错的选择。虽然它不能处理每种类型的Python对象，但它可以处理字符串、数字和布尔值，以及列表和字典。这应该可以很好地满足我们的配置需求。

当我们说一个库是“不安全”时，这意味着什么？一些数据格式设计有强大的功能，比如可扩展性、链接或别名，解析库必须实现这些功能。不幸的是，这些功能可能被用于恶意目的。例如，十亿次笑XML漏洞结合了三个XML功能，制作了一个文件，当解析时，会扩展到一个巨大的大小（通常导致程序或者在某些情况下，系统崩溃）。

# 为设置持久性构建模型

与任何数据持久化一样，我们需要先实现一个模型。与我们的`CSVModel`类一样，设置模型需要保存和加载数据，以及定义设置数据的布局。

在`models.py`文件中，让我们按照以下方式开始一个新的类：

```py
class SettingsModel:
    """A model for saving settings"""
```

就像我们的`CSVModel`类一样，我们需要定义我们模型的模式：

```py
    variables = {
        'autofill date': {'type': 'bool', 'value': True},
        'autofill sheet data': {'type': 'bool', 'value': True}
     }
```

`variables`字典将存储每个项目的模式和值。每个设置都有一个列出数据类型和默认值的字典（如果需要，我们可以在这里列出其他属性，比如最小值、最大值或可能的值）。`variables`字典将是我们保存到磁盘并从磁盘加载以持久化程序设置的数据结构。

模型还需要一个位置来保存配置文件，因此我们的构造函数将以文件名和路径作为参数。现在，我们只提供并使用合理的默认值，但在将来我们可能会想要更改这些值。

然而，我们不能只提供一个单一的文件路径；我们在同一台计算机上有不同的用户，他们会想要保存不同的设置。我们需要确保设置保存在各个用户的主目录中，而不是一个单一的公共位置。

因此，我们的`__init__()`方法如下：

```py
    def __init__(self, filename='abq_settings.json', path='~'):
        # determine the file path
        self.filepath = os.path.join(
            os.path.expanduser(path), filename)
```

作为Linux或macOS终端的用户会知道，`~`符号是Unix的快捷方式，指向用户的主目录。Python的`os.path.expanduser()`函数将这个字符转换为绝对路径（即使在Windows上也是如此），这样文件将被保存在运行程序的用户的主目录中。`os.path.join()`将文件名附加到扩展路径上，给我们一个完整的路径到用户特定的配置文件。

一旦模型被创建，我们就希望从磁盘加载用户保存的选项。从磁盘加载数据是一个非常基本的模型操作，我们应该能够在类外部控制，所以我们将这个方法设为公共方法。

我们将称这个方法为`load()`，并在这里调用它：

```py
        self.load()
```

`load()`将期望找到一个包含与`variables`字典相同格式的字典的JSON文件。它将需要从文件中加载数据，并用文件副本替换自己的`variables`副本。

一个简单的实现如下：

```py
    def load(self):
        """Load the settings from the file"""

        with open(self.filepath, 'r') as fh:
            self.variables = json.loads(fh.read())
```

`json.loads()`函数读取JSON字符串并将其转换为Python对象，我们直接保存到我们的`variables`字典中。当然，这种方法也存在一些问题。首先，如果设置文件不存在会发生什么？在这种情况下，`open`会抛出一个异常，程序会崩溃。不好！

因此，在我们尝试打开文件之前，让我们测试一下它是否存在，如下所示：

```py
        # if the file doesn't exist, return
        if not os.path.exists(self.filepath):
            return
```

如果文件不存在，该方法将简单地返回并不执行任何操作。文件不存在是完全合理的，特别是如果用户从未运行过程序或编辑过任何设置。在这种情况下，该方法将保持`self.variables`不变，用户将最终使用默认值。

第二个问题是我们的设置文件可能存在，但不包含任何数据或无效数据（比如`variables`字典中不存在的键），导致程序崩溃。为了防止这种情况，我们将JSON数据拉到一个本地变量中；然后通过询问`raw_values`只获取那些存在于`variables`中的键来更新`variables`，如果它们不存在，则提供一个默认值。

新的、更安全的代码如下：

```py
        # open the file and read in the raw values
        with open(self.filepath, 'r') as fh:
            raw_values = json.loads(fh.read())

        # don't implicitly trust the raw values, 
        # but only get known keys
        for key in self.variables:
            if key in raw_values and 'value' in raw_values[key]:
                raw_value = raw_values[key]['value']
                self.variables[key]['value'] = raw_value
```

由于`variables`已经使用默认值创建，如果`raw_values`没有给定键，或者该键中的字典不包含`values`项，我们只需要忽略`raw_values`。

现在`load()`已经编写好了，让我们编写一个`save()`方法将我们的值写入文件：

```py
    def save(self, settings=None):
        json_string = json.dumps(self.variables)
        with open(self.filepath, 'w') as fh:
            fh.write(json_string)
```

`json.dumps()`函数是`loads()`的反函数：它接受一个Python对象并返回一个JSON字符串。保存我们的`settings`数据就像将`variables`字典转换为字符串并将其写入指定的文本文件一样简单。

我们的模型需要的最后一个方法是让外部代码设置值的方法；他们可以直接操作`variables`，但为了保护我们的数据完整性，我们将通过方法调用来实现。遵循Tkinter的惯例，我们将称这个方法为`set()`。

`set()`方法的基本实现如下：

```py
    def set(self, key, value):
        self.variables[key]['value'] = value
```

这个简单的方法只是接受一个键和值，并将它们写入`variables`字典。不过，这又带来了一些潜在的问题；如果提供的值对于数据类型来说不是有效的怎么办？如果键不在我们的`variables`字典中怎么办？这可能会导致难以调试的情况，因此我们的`set()`方法应该防范这种情况。

将代码更改如下：

```py
    if (
        key in self.variables and
        type(value).__name__ == self.variables[key]['type']
    ):
        self.variables[key]['value'] = value
```

通过使用与实际Python类型名称相对应的`type`字符串，我们可以使用`type(value).__name__`将其与值的类型名称进行匹配（我们本可以在我们的`variables`字典中使用实际的类型对象，但这些对象无法序列化为JSON）。现在，尝试写入未知键或不正确的变量类型将会失败。

然而，我们不应该让它悄悄失败；我们应该立即引发`ValueError`来提醒我们存在问题，如下所示：

```py
    else:
        raise ValueError("Bad key or wrong variable type")
```

为什么要引发异常？如果测试失败，这只能意味着调用代码中存在错误。通过异常，我们将立即知道调用代码是否向我们的模型发送了错误的请求。如果没有异常，请求将悄悄失败，留下难以发现的错误。

故意引发异常的想法对于初学者来说通常似乎很奇怪；毕竟，我们正在尽量避免异常，对吧？对于主要是使用现有模块的小脚本来说，这是正确的；然而，当编写自己的模块时，异常是模块与使用它的代码交流问题的正确方式。试图处理或更糟糕的是消除外部调用代码的不良行为，最好会破坏模块化；在最坏的情况下，它会产生难以追踪的微妙错误。

# 在我们的应用程序中使用设置模型

我们的应用程序在启动时需要加载设置，然后在更改设置时自动保存。目前，应用程序的`settings`字典是手动创建的，但是我们的模型应该真正告诉它创建什么样的变量。让我们按照以下步骤在我们的应用程序中使用`settings`模型：

1.  用以下代码替换定义`Application.settings`的代码：

```py
        self.settings_model = m.SettingsModel()
        self.load_settings()
```

首先，我们创建一个`settings`模型并将其保存到我们的`Application`对象中。然后，我们将运行一个`load_settings()`方法。这个方法将负责根据`settings_model`设置`Application.settings`字典。

1.  现在，让我们创建`Application.load_settings()`：

```py
    def load_settings(self):
        """Load settings into our self.settings dict."""
```

1.  我们的模型存储了每个变量的类型和值，但我们的应用程序需要Tkinter变量。我们需要一种方法将模型对数据的表示转换为`Application`可以使用的结构。一个字典提供了一个方便的方法来做到这一点，如下所示：

```py
      vartypes = {
          'bool': tk.BooleanVar,
          'str': tk.StringVar,
          'int': tk.IntVar,
         'float': tk.DoubleVar
      }
```

注意，每个名称都与Python内置函数的类型名称匹配。我们可以在这里添加更多条目，但这应该涵盖我们未来的大部分需求。现在，我们可以将这个字典与模型的`variables`字典结合起来构建`settings`字典：

```py
        self.settings = {}
        for key, data in self.settings_model.variables.items():
            vartype = vartypes.get(data['type'], tk.StringVar)
            self.settings[key] = vartype(value=data['value'])
```

1.  在这里使用Tkinter变量的主要原因是，我们可以追踪用户通过UI对值所做的任何更改并立即做出响应。具体来说，我们希望在用户进行更改时立即保存我们的设置，如下所示：

```py
        for var in self.settings.values():
            var.trace('w', self.save_settings)
```

1.  当然，这意味着我们需要编写一个名为`Application.save_settings()`的方法，每当值发生更改时都会运行。`Application.load_settings()`已经完成，所以让我们接着做这个：

```py
    def save_settings(self, *args):
        """Save the current settings to a preferences file"""
```

1.  `save_settings()`方法只需要从`Application.settings`中获取数据并保存到模型中：

```py
        for key, variable in self.settings.items():
            self.settings_model.set(key, variable.get())
        self.settings_model.save()
```

这很简单，只需要循环遍历`self.settings`，并调用我们模型的`set()`方法逐个获取值。然后，我们调用模型的`save()`方法。

1.  现在，你应该能够运行程序并观察到设置被保存了，即使你关闭并重新打开应用程序。你还会在你的主目录中找到一个名为`abq_settings.json`的文件。

# 总结

在这一章中，我们简单的表单迈出了成为一个完全成熟的应用程序的重要一步。我们实现了一个主菜单，选项设置在执行之间是持久的，并且有一个“关于”对话框。我们增加了选择保存记录的文件的能力，并通过错误对话框改善了表单错误的可见性。在这个过程中，你学到了关于Tkinter菜单、文件对话框和消息框，以及标准库中持久化数据的各种选项。

在下一章中，我们将被要求让程序读取和写入。我们将学习关于Tkinter的树部件，如何在主视图之间切换，以及如何使我们的`CSVModel`和`DataRecordForm`类能够读取和更新现有数据。
