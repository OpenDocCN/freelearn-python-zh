# 图形用户界面

在本章中，我们将涵盖以下配方：

+   警报-在图形系统上显示警报对话框

+   对话框-如何使用对话框询问简单问题

+   ProgressBar 对话框-如何提供图形进度对话框

+   列表-如何实现可滚动的元素列表以供选择

+   菜单-如何在 GUI 应用程序中创建菜单以允许多个操作

# 介绍

Python 带有一个编程语言很少提供的功能：内置的**图形用户界面**（**GUI**）库。

Python 附带了一个可通过标准库提供的`tkinter`模块控制的`Tk`小部件工具包的工作版本。

`Tk`工具包实际上是通过一种称为`Tcl`的简单语言使用的。所有`Tk`小部件都可以通过`Tcl`命令进行控制。

大多数这些命令都非常简单，采用以下形式：

```py
classname widgetid options
```

例如，以下内容会导致一个按钮（标识为`mybutton`）上有红色的“点击这里”文本：

```py
button .mybutton -fg red  -text "click here"
```

由于这些命令通常相对简单，Python 附带了一个内置的`Tcl`解释器，并使用它来驱动`Tk`小部件。

如今，几乎每个人，甚至更加专注的计算机用户，都习惯于依赖 GUI 来完成他们的许多任务，特别是对于需要基本交互的简单应用程序，例如选择选项，确认输入或显示一些进度。因此，使用 GUI 可能非常方便。

对于图形应用程序，用户通常无需查看应用程序的帮助页面，阅读文档并浏览应用程序提供的选项以了解其特定的语法。 GUI 已经提供了几十年的一致交互语言，如果正确使用，是保持软件入门门槛低的好方法。

由于 Python 提供了创建强大的控制台应用程序和良好的 GUI 所需的一切，因此下次您需要创建新工具时，如果您选择图形应用程序，也许停下来考虑一下您的用户会发现什么更方便，前往`tkinter`可能是一个不错的选择。

虽然`tkinter`与强大的工具包（如 Qt 或 GTK）相比可能有限，但它确实是一个完全独立于平台的解决方案，对于大多数应用程序来说已经足够好了。

# 警报

最简单的 GUI 类型是警报。只需在图形框中打印一些内容以通知用户结果或事件：

![](img/ca5ecdd4-5d78-49d5-a9d8-50c84ff5b921.png)

# 如何做...

`tkinter`中的警报由`messagebox`对象管理，我们可以通过要求`messagebox`为我们显示一个来创建一个：

```py
from tkinter import messagebox

def alert(title, message, kind='info', hidemain=True):
    if kind not in ('error', 'warning', 'info'):
        raise ValueError('Unsupported alert kind.')

    show_method = getattr(messagebox, 'show{}'.format(kind))
    show_method(title, message)
```

一旦我们有了`alert`助手，我们可以初始化`Tk`解释器并显示我们想要的多个警报：

```py
from tkinter import Tk

Tk().withdraw()
alert('Hello', 'Hello World')
alert('Hello Again', 'Hello World 2', kind='warning')
```

如果一切按预期工作，我们应该看到一个弹出对话框，一旦解除，新的对话框应该出现“再见”。

# 工作原理...

`alert`函数本身只是`tkinter.messagebox`提供的一个薄包装。

我们可以显示三种类型的消息框：`error`，`warning`和`info`。如果请求了不支持的对话框类型，我们会拒绝它：

```py
if kind not in ('error', 'warning', 'info'):
    raise ValueError('Unsupported alert kind.')
```

每种对话框都是通过依赖`messagebox`的不同方法来显示的。信息框使用`messagebox.showinfo`显示，而错误使用`messagebox.showerror`显示，依此类推。

因此，我们获取`messagebox`的相关方法：

```py
show_method = getattr(messagebox, 'show{}'.format(kind))
```

然后，我们调用它来显示我们的框：

```py
show_method(title, message)
```

`alert`函数非常简单，但还有一件事情我们需要记住。

`tkinter`库通过与`Tk`的解释器和环境交互来工作，必须创建和启动它。

如果我们自己不开始，`tkinter`需要在需要发送一些命令时立即为我们启动一个。但是，这会导致始终创建一个空的主窗口。

因此，如果您像这样使用`alert`，您将收到警报，但您也会在屏幕角落看到空窗口。

为了避免这种情况，我们需要自己初始化`Tk`环境并禁用主窗口，因为我们对它没有任何用处：

```py
from tkinter import Tk
Tk().withdraw()
```

然后我们可以显示任意数量的警报，而不会出现在屏幕周围泄漏空的不需要的窗口的风险。

# 对话框

对话框是用户界面可以提供的最简单和最常见的交互。询问一个简单的输入，比如数字、文本或是是/否，可以满足简单应用程序与用户交互的许多需求。

`tkinter`提供了大多数情况下的对话框，但如果你不知道这个库，可能很难找到它们。作为一个指针，`tkinter`提供的所有对话框都有非常相似的签名，因此很容易创建一个`dialog`函数来显示它们：

![](img/f0d43442-8643-4db8-a0bc-86a03b14bcdf.png)

对话框将如下所示：

![](img/28197129-73c7-41d8-ab0b-85af9edb584b.png)

打开文件的窗口如下截图所示：

![](img/04fb7778-7b6f-42c7-8c77-b323c93b81e0.png)

# 如何做...

我们可以创建一个`dialog`函数来隐藏对话框类型之间的细微差异，并根据请求的类型调用适当的对话框：

```py
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog

def dialog(ask, title, message=None, **kwargs):
    for widget in (messagebox, simpledialog, filedialog):
        show = getattr(widget, 'ask{}'.format(ask), None)
        if show:
            break
    else:
        raise ValueError('Unsupported type of dialog: {}'.format(ask))

    options = dict(kwargs, title=title)
    for arg, replacement in dialog._argsmap.get(widget, {}).items():
        options[replacement] = locals()[arg]
    return show(**options)
dialog._argsmap = {
    messagebox: {'message': 'message'},
    simpledialog: {'message': 'prompt'}
}
```

然后我们可以测试我们的`dialog`方法来显示所有可能的对话框类型，并显示用户的选择：

```py
>>> from tkinter import Tk

>>> Tk().withdraw()
>>> for ask in ('okcancel', 'retrycancel', 'yesno', 'yesnocancel',
...             'string', 'integer', 'float', 'directory', 'openfilename'):
...     choice = dialog(ask, 'This is title', 'What?')
...     print('{}: {}'.format(ask, choice))
okcancel: True
retrycancel: False
yesno: True
yesnocancel: None
string: Hello World
integer: 5
float: 1.3
directory: /Users/amol/Documents
openfilename: /Users/amol/Documents/FileZilla_3.27.1_macosx-x86.app.tar.bz2
```

# 它是如何工作的...

`tkinter`提供的对话框类型分为`messagebox`、`simpledialog`和`filedialog`模块（你可能也考虑`colorchooser`，但它很少需要）。

因此，根据用户想要的对话框类型，我们需要选择正确的模块并调用所需的函数来显示它：

```py
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog

def dialog(ask, title, message=None, **kwargs):
    for widget in (messagebox, simpledialog, filedialog):
        show = getattr(widget, 'ask{}'.format(ask), None)
        if show:
            break
    else:
        raise ValueError('Unsupported type of dialog: {}'.format(ask))
```

如果没有模块公开函数来显示请求的对话框类型（所有函数都以`ask*`命名），循环将在没有打破的情况下结束，因此将进入`else`子句，引发异常以通知调用者请求的类型不可用。

如果循环以`break`退出，`widget`变量将指向能够显示请求的对话框的模块，而`show`变量将导致实际能够显示它的函数。

一旦我们有了正确的函数，我们需要考虑各种对话框函数之间的细微差异。

主要的问题与`messagebox`对话框有一个`message`参数有关，而`simpledialog`对话框有一个提示参数来显示用户的消息。`filedialog`根本不需要任何消息。

这是通过创建一个基本的选项字典和自定义提供的选项以及`title`选项来完成的，因为在所有类型的对话框中始终可用：

```py
options = dict(kwargs, title=title)
```

然后，通过查找`dialog._argsmap`字典中从`dialog`参数的名称到预期参数的映射，将`message`选项替换为正确的名称（或跳过）。

例如，在`simpledialog`的情况下，使用`{'message': 'prompt'}`映射。`message`变量在函数局部变量中查找（`locals()[arg]`），然后将其分配给选项字典，`prompt`名称由`replacement`指定。然后，最终调用分配给`show`的函数来显示对话框：

```py
for arg, replacement in dialog._argsmap.get(widget, {}).items():
    options[replacement] = locals()[arg]
return show(**options)

dialog._argsmap = {
    messagebox: {'message': 'message'}, 
    simpledialog: {'message': 'prompt'}
}
```

# 进度条对话框

在进行长时间运行的操作时，向用户显示进度的最常见方式是通过进度条。

在线程中运行操作时，我们可以更新进度条以显示操作正在向前推进，并向用户提示可能需要完成工作的时间：

![](img/e0ec92d1-34a2-46f5-92e1-6616f7d96c3f.png)

# 如何做...

`simpledialog.SimpleDialog`小部件用于创建带有一些文本和按钮的简单对话框。我们将利用它来显示进度条而不是按钮：

```py
import tkinter
from tkinter import simpledialog
from tkinter import ttk

from queue import Queue

class ProgressDialog(simpledialog.SimpleDialog):
    def __init__(self, master, text='', title=None, class_=None):
        super().__init__(master=master, text=text, title=title, 
                         class_=class_)
        self.default = None
        self.cancel = None

        self._bar = ttk.Progressbar(self.root, orient="horizontal", 
                                    length=200, mode="determinate")
        self._bar.pack(expand=True, fill=tkinter.X, side=tkinter.BOTTOM)
        self.root.attributes("-topmost", True)

        self._queue = Queue()
        self.root.after(200, self._update)

    def set_progress(self, value):
        self._queue.put(value)

    def _update(self):
        while self._queue.qsize():
            try:
                self._bar['value'] = self._queue.get(0)
            except Queue.Empty:
                pass
        self.root.after(200, self._update)
```

然后可以创建`ProgressDialog`，并使用后台线程让操作进展（比如下载），然后在我们的操作向前推进时更新进度条：

```py
if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()

    # Prepare the progress dialog
    p = ProgressDialog(master=root, text='Downloading Something...',
                    title='Download')

    # Simulate a download running for 5 seconds in background
    import threading
    def _do_progress():
        import time
        for i in range(1, 11):
            time.sleep(0.5)
            p.set_progress(i*10)
        p.done(0)
    t = threading.Thread(target=_do_progress)
    t.start()

    # Display the dialog and wait for the download to finish.
    p.go()
    print('Download Completed!')
```

# 它是如何工作的...

我们的对话框本身主要基于`simpledialog.SimpleDialog`小部件。我们创建它，然后设置`self.default = None`以防止用户能够通过按`<Return>`键关闭对话框，并且我们还设置`self.default = None`以防止用户通过按窗口上的按钮关闭对话框。我们希望对话框保持打开状态，直到完成为止：

```py
class ProgressDialog(simpledialog.SimpleDialog):
    def __init__(self, master, text='', title=None, class_=None):
        super().__init__(master=master, text=text, title=title, class_=class_)
        self.default = None
        self.cancel = None
```

然后我们实际上需要进度条本身，它将显示在文本消息下方，并且我们还将对话框移到前面，因为我们希望用户意识到正在发生某事：

```py
self._bar = ttk.Progressbar(self.root, orient="horizontal", 
                            length=200, mode="determinate")
self._bar.pack(expand=True, fill=tkinter.X, side=tkinter.BOTTOM)
self.root.attributes("-topmost", True)
```

在最后一部分，我们需要安排`self._update`，它将继续循环，直到对话框停止更新进度条，如果`self._queue`中有新的进度值可用。进度值可以通过`self._queue`提供，我们将在通过`set_progress`方法提供新的进度值时插入新的进度值：

```py
self._queue = Queue()
self.root.after(200, self._update)
```

我们需要通过`Queue`进行，因为具有进度条更新的对话框会阻塞整个程序。

当`Tkinter mainloop`函数运行时（由`simpledialog.SimpleDialog.go()`调用），没有其他东西可以继续进行。

因此，UI 和下载必须在两个不同的线程中进行，并且由于我们无法从不同的线程更新 UI，因此必须从生成它们的线程将进度值发送到将其消耗以更新进度条的 UI 线程。

执行操作并生成进度更新的线程可以通过`set_progress`方法将这些进度更新发送到 UI 线程：

```py
def set_progress(self, value):
    self._queue.put(value)
```

另一方面，UI 线程将不断调用`self._update`方法（每 200 毫秒一次），以检查`self._queue`中是否有更新请求，然后应用它：

```py
def _update(self):
    while self._queue.qsize():
        try:
            self._bar['value'] = self._queue.get(0)
        except Queue.Empty:
            pass
    self.root.after(200, self._update)
```

在更新结束时，该方法将重新安排自己：

```py
self.root.after(200, self._update)
```

这样，我们将永远继续每 200 毫秒检查进度条是否有更新，直到`self.root mainloop`退出。

为了使用`ProgressDialog`，我们模拟了一个需要 5 秒钟的下载。这是通过创建对话框本身完成的：

```py
if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()

    # Prepare the progress dialog
    p = ProgressDialog(master=root, text='Downloading Something...',
                    title='Download')
```

然后我们启动了一个后台线程，持续 5 秒，每隔半秒更新一次进度：

```py
# Simulate a download running for 5 seconds in background
import threading

def _do_progress():
    import time
    for i in range(1, 11):
        time.sleep(0.5)
        p.set_progress(i*10)
    p.done(0)

t = threading.Thread(target=_do_progress)
t.start()
```

更新发生是因为线程调用`p.set_progress`，它将在队列中设置一个新的进度值，向 UI 线程发出新的进度值设置信号。

一旦下载完成，进度对话框将通过`p.done(0)`退出。

一旦我们的下载线程就位，我们就可以显示进度对话框并等待其退出：

```py
# Display the dialog and wait for the download to finish.
p.go()
print('Download Completed!')
```

# 列表

当用户有两个以上的选择时，最好的列出它们的方式是通过列表。`tkinter`模块提供了一个`ListBox`，允许我们在可滚动的小部件中显示一组条目供用户选择。

我们可以使用它来实现一个对话框，用户可以从中选择许多选项并抓取所选项：

![](img/3d954334-8b5f-4610-b619-ac1f212250db.png)

# 如何做...

`simpledialog.Dialog`类可用于实现简单的确定/取消对话框，并允许我们提供具有自定义内容的对话框主体。

我们可以使用它向对话框添加消息和列表，并让用户进行选择：

```py
import tkinter
from tkinter import simpledialog

class ChoiceDialog(simpledialog.Dialog):
    def __init__(self, parent, title, text, items):
        self.selection = None
        self._items = items
        self._text = text
        super().__init__(parent, title=title)

    def body(self, parent):
        self._message = tkinter.Message(parent, text=self._text, aspect=400)
        self._message.pack(expand=1, fill=tkinter.BOTH)
        self._list = tkinter.Listbox(parent)
        self._list.pack(expand=1, fill=tkinter.BOTH, side=tkinter.TOP)
        for item in self._items:
            self._list.insert(tkinter.END, item)
        return self._list

    def validate(self):
        if not self._list.curselection():
            return 0
        return 1

    def apply(self):
        self.selection = self._items[self._list.curselection()[0]]
```

一旦有了`ChoiceDialog`，我们可以显示它并提供一个项目列表，让用户选择一个或取消对话框：

```py
if __name__ == '__main__':
    tk = tkinter.Tk()
    tk.withdraw()

    dialog = ChoiceDialog(tk, 'Pick one',
                        text='Please, pick a choice?',
                        items=['first', 'second', 'third'])
    print('Selected "{}"'.format(dialog.selection))
```

`ChoiceDialog.selection`属性将始终包含所选项目，如果对话框被取消，则为`None`。

# 它是如何工作的...

`simpledialog.Dialog`默认创建一个带有`确定`和`取消`按钮的对话框，并且只提供一个标题。

在我们的情况下，除了创建对话框本身之外，我们还希望保留对话框的消息和可供选择的项目，以便我们可以向用户显示它们。此外，默认情况下，我们希望设置尚未选择任何项目。最后，我们可以调用`simpledialog.Dialog.__init__`，一旦调用它，主线程将阻塞，直到对话框被解除：

```py
import tkinter
from tkinter import simpledialog

class ChoiceDialog(simpledialog.Dialog):
    def __init__(self, parent, title, text, items):
        self.selection = None
        self._items = items
        self._text = text
        super().__init__(parent, title=title)
```

我们可以通过重写`simpledialog.Dialog.body`方法来添加任何其他内容。这个方法可以将更多的小部件添加为对话框主体的子级，并且可以返回应该具有焦点的特定小部件：

```py
def body(self, parent):
    self._message = tkinter.Message(parent, text=self._text, aspect=400)
    self._message.pack(expand=1, fill=tkinter.BOTH)
    self._list = tkinter.Listbox(parent)
    self._list.pack(expand=1, fill=tkinter.BOTH, side=tkinter.TOP)
    for item in self._items:
        self._list.insert(tkinter.END, item)
    return self._list
```

`body`方法是在`simpledialog.Dialog.__init__`中创建的，因此在阻塞主线程之前调用它。

对话框的内容放置好后，对话框将阻塞等待用户点击按钮。

如果点击`cancel`按钮，则对话框将自动关闭，`ChoiceDialog.selection`将保持为`None`。

如果点击`Ok`，则调用`ChoiceDialog.validate`方法来检查选择是否有效。我们的`validate`实现将检查用户在点击`Ok`之前是否实际选择了条目，并且只有在有选定项目时才允许用户关闭对话框：

```py
def validate(self):
    if not self._list.curselection():
        return 0
    return 1
```

如果验证通过，将调用`ChoiceDialog.apply`方法来确认选择，然后我们只需在`self.selection`中设置所选项目的名称，这样一旦对话框不再可见，调用者就可以访问它了：

```py
def apply(self):
    self.selection = self._items[self._list.curselection()[0]]
```

这使得可以显示对话框并在其关闭后从`selection`属性中读取所选值成为可能：

```py
dialog = ChoiceDialog(tk, 'Pick one',
                    text='Please, pick a choice?',
                    items=['first', 'second', 'third'])
print('Selected "{}"'.format(dialog.selection))
```

# 菜单

当应用程序允许执行多个操作时，菜单通常是允许访问这些操作的最常见方式：

![](img/ca3d5d42-6626-40f4-925f-c9ef2ff3a557.png)

# 如何做...

`tkinter.Menu`类允许我们创建菜单、子菜单、操作和分隔符。因此，它提供了我们在基于 GUI 的应用程序中创建基本菜单所需的一切：

```py
import tkinter

def set_menu(window, choices):
    menubar = tkinter.Menu(root)
    window.config(menu=menubar)

    def _set_choices(menu, choices):
        for label, command in choices.items():
            if isinstance(command, dict):
                # Submenu
                submenu = tkinter.Menu(menu)
                menu.add_cascade(label=label, menu=submenu)
                _set_choices(submenu, command)
            elif label == '-' and command == '-':
                # Separator
                menu.add_separator()
            else:
                # Simple choice
                menu.add_command(label=label, command=command)

    _set_choices(menubar, choices)
```

`set_menu`函数允许我们轻松地从嵌套的操作和子菜单的字典中创建整个菜单层次结构：

```py
import sys
root = tkinter.Tk()

from collections import OrderedDict
set_menu(root, {
    'File': OrderedDict([
        ('Open', lambda: print('Open!')),
        ('Save', lambda: print('Save')),
        ('-', '-'),
        ('Quit', lambda: sys.exit(0))
    ])
})
root.mainloop()
```

如果您使用的是 Python 3.6+，还可以避免使用`OrderedDict`，而是使用普通字典，因为字典已经是有序的。

# 它是如何工作的...

提供一个窗口，`set_menu`函数创建一个`Menu`对象并将其设置为窗口菜单：

```py
def set_menu(window, choices):
    menubar = tkinter.Menu(root)
    window.config(menu=menubar)
```

然后，它使用通过`choices`参数提供的选择填充菜单。这个参数预期是一个字典，其中键是菜单条目的名称，值是在选择时应调用的可调用对象，或者如果选择应导致子菜单，则是另一个字典。最后，当标签和选择都设置为`-`时，它支持分隔符。

菜单通过递归函数遍历选项树来填充，该函数调用`Menu.add_command`、`Menu.add_cascade`和`Menu.add_separator`，具体取决于遇到的条目：

```py
def _set_choices(menu, choices):
    for label, command in choices.items():
        if isinstance(command, dict):
            # Submenu
            submenu = tkinter.Menu(menu)
            menu.add_cascade(label=label, menu=submenu)
            _set_choices(submenu, command)
        elif label == '-' and command == '-':
            # Separator
            menu.add_separator()
        else:
            # Simple choice
            menu.add_command(label=label, command=command)

_set_choices(menubar, choices)
```
