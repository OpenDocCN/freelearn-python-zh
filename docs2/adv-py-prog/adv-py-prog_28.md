# *第二十五章*：命令模式

在本章中，我们将介绍**命令模式**。使用这种设计模式，我们可以将操作（如“复制粘贴”）封装为一个对象。命令模式也非常适合组合多个命令。它对于实现宏、多级撤销和事务非常有用。在我们的讨论过程中，我们将了解将操作视为对象的想法，并使用这种命令思维来处理应用程序事务。

我们将讨论以下内容：

+   理解命令模式

+   现实世界示例

+   用例

+   实现方式

# 技术要求

本章的代码文件可以通过以下链接访问：[`github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter25`](https://github.com/PacktPublishing/Advanced-Python-Programming-Second-Edition/tree/main/Chapter25)。

# 理解命令模式

现今的大多数应用程序都有**撤销**操作。这很难想象，但在许多年前，任何软件中都没有撤销功能。撤销功能是在 1974 年引入的([j.mp/wiundo](http://j.mp/wiundo))，但 Fortran 和 Lisp 这两种至今仍被广泛使用的编程语言分别在 1957 年和 1958 年创建([j.mp/proghist](http://j.mp/proghist))！用户没有简单的方法来修复错误。我不希望在那几年里成为一个应用程序用户。

历史就到这里吧！我们想知道如何在我们的应用程序中实现撤销功能，既然你已经阅读了本章的标题，你就已经知道推荐使用哪种设计模式来实现撤销：命令模式。

命令设计模式帮助我们封装操作（撤销、重做、复制、粘贴等）为一个对象。这简单意味着我们创建一个包含所有逻辑和实现操作所需方法的类。这样做的好处如下([j.mp/cmdpattern](http://j.mp/cmdpattern))：

+   我们不必直接执行命令。它可以在任何时候执行。

+   调用命令的对象与知道如何执行它的对象解耦。调用者不需要了解命令的任何实现细节。

+   如果有道理，可以将多个命令组合起来，以便调用者可以按顺序执行它们。这在实现多级撤销命令时非常有用。

如你所想，这个模式在现实世界中有着广泛的应用场景，我们将在下一节中看到。

# 现实世界示例

当我们去餐馆吃晚餐时，我们会向服务员下单。他们用来写下订单的订单本（通常是纸张）是一个命令的例子。写下订单后，服务员将其放入厨师执行的队列中。每个检查都是独立的，可以用来执行许多不同的命令，例如，为每个将要烹饪的项目执行一个命令。

如你所预期，我们也有几个软件中的示例。以下是我能想到的两个：

+   PyQt 是 QT 工具包的 Python 绑定。PyQt 包含一个`QAction`类，它将操作建模为命令。每个动作都支持额外的可选信息，例如描述、工具提示或快捷键([j.mp/qaction](http://j.mp/qaction))。

+   Git Cola ([j.mp/git-cola](http://j.mp/git-cola))，一个用 Python 编写的 Git 图形用户界面，使用命令模式来修改模型、修改提交、应用不同的选举、检出等([j.mp/git-cola-code](http://j.mp/git-cola-code))。

现在，我们将更一般地讨论在下一节中命令模式何时会证明是有用的。

# 用例

许多开发者将撤销示例视为命令模式的唯一用例。实际上，撤销是命令模式的杀手级功能。然而，命令模式实际上可以做更多([j.mp/commddp](http://j.mp/commddp))：

+   **GUI 按钮和菜单项**：前面提到的 PyQt 示例使用命令模式来实现按钮和菜单项上的操作。

+   **其他操作**：除了撤销之外，命令还可以用于实现任何操作。一些例子是剪切、复制、粘贴、重做和首字母大写文本。

+   **事务行为和日志记录**：事务行为和日志记录对于保持更改的持久日志非常重要。操作系统用于从系统崩溃中恢复，关系数据库用于实现事务，文件系统用于实现快照，安装程序（向导）用于撤销已取消的安装。

+   **宏**：在这里，我们指的是可以记录并在任何时间点按需执行的行动序列。流行的编辑器如 Emacs 和 Vim 支持宏。

为了展示我们迄今为止讨论的内容，我们将实现一个文件实用程序管理应用程序。

# 实现

在本节中，我们将使用命令模式来实现最基本的文件实用程序：

+   创建一个文件，并可选择向其中写入文本（一个字符串）

+   读取文件内容

+   重命名文件

+   删除文件

我们不会从头实现这些实用程序，因为 Python 已经在`os`模块中提供了良好的实现。我们想要在它们之上添加一个额外的抽象层，以便将它们视为命令。通过这样做，我们可以获得命令提供的所有优势。

从显示的操作中可以看出，重命名文件和创建文件支持撤销。删除文件和读取文件内容不支持撤销。实际上，可以在删除文件操作上实现撤销。一种技术是使用一个特殊的垃圾箱目录来存储所有已删除的文件，以便在用户请求时恢复它们。这是所有现代桌面环境默认使用的功能，留作练习。

每个命令有两个部分：

+   `__init__()` 方法，并包含命令执行有用操作所需的所有信息（文件的路径、将要写入文件的内容等等）。

+   `execute()` 方法。当我们想要实际运行命令时，我们会调用 `execute()` 方法。这不一定是在初始化后立即进行的。

让我们从重命名实用工具开始，它是通过 `RenameFile` 类实现的。`__init__()` 方法接受源 (`src`) 和目标 (`dest`) 文件路径作为参数（字符串）。如果没有使用路径分隔符，则使用当前目录来创建文件。使用路径分隔符的一个例子是将 `/tmp/file1` 字符串作为 `src`，将 `/home/user/file2` 字符串作为 `dest`。另一个例子，我们不会使用路径，是将 `file1` 作为 `src`，将 `file2` 作为 `dest`：

```py
class RenameFile:
     def __init__(self, src, dest):  
         self.src = src
         self.dest = dest
```

我们向类中添加了 `execute()` 方法。此方法使用 `os.rename()` 实际执行重命名。`verbose` 变量对应于全局 `print()`，对于示例来说足够好，通常可以使用更成熟和强大的工具，例如，logging 模块 ([j.mp/py3log](http://j.mp/py3log))：

```py
    def execute(self):  
        if verbose:  
            print(f"[renaming '{self.src}' to \
              '{self.dest}']")  
        os.rename(self.src, self.dest)
```

我们的重命名实用工具 (`RenameFile`) 通过其 `undo()` 方法支持撤销操作。在这种情况下，我们再次使用 `os.rename()` 来恢复文件的原始名称：

```py
    def undo(self):  
        if verbose:  
            print(f"[renaming '{self.dest}' back to \
              '{self.src}']")  
        os.rename(self.dest, self.src)
```

在这个例子中，删除文件是在一个函数中实现的，而不是在类中。这是为了表明，对于您想要添加的每个命令，不一定必须创建一个新的类（更多内容将在后面介绍）。`delete_file()` 函数接受一个文件路径作为字符串，并使用 `os.remove()` 来删除它：

```py
def delete_file(path):
    if verbose:
        print(f"deleting file {path}")
    os.remove(path)
```

回到使用类。`CreateFile` 类用于创建文件。该类的 `__init__()` 方法接受熟悉的 `path` 参数和一个 `txt` 参数，用于写入文件的内容（一个字符串）。如果没有传递 `txt`，则将默认的 `hello world` 文本写入文件。通常，相同的默认行为是创建一个空文件，但为了这个示例的需要，我决定在其中写入一个默认字符串。

`CreateFile` 类的定义如下开始：

```py
class CreateFile:

    def __init__(self, path, txt='hello world\n'):  
        self.path = path 
        self.txt = txt
```

然后我们添加一个 `execute()` 方法，在其中我们使用 `with` 语句和 Python 的内置 `open()` 函数来打开文件（`mode='w'` 表示写入模式），并使用 `write()` 函数将 `txt` 字符串写入其中，如下所示：

```py
    def execute(self):  
        if verbose:  
            print(f"[creating file '{self.path}']")  
        with open(self.path, mode='w', encoding='utf-8') \
           as out_file:  
            out_file.write(self.txt)
```

创建文件操作的撤销动作是删除该文件。因此，我们添加到类中的 `undo()` 方法简单地使用 `delete_file()` 函数来实现，如下所示：

```py
    def undo(self):  
        delete_file(self.path)
```

最后一个实用工具使我们能够读取文件的内容。`ReadFile` 类的 `execute()` 方法再次使用 `open()`，这次是以读取模式，然后使用 `print()` 打印文件的内容。

`ReadFile` 类被定义为如下：

```py
class ReadFile:

     def __init__(self, path):  
         self.path = path

     def execute(self):  
         if verbose:  
             print(f"[reading file '{self.path}']")  
         with open(self.path, mode='r', encoding='utf-8') \
           as in_file:  
             print(in_file.read(), end='')
```

`main()`函数使用了我们所定义的实用工具。`orig_name`和`new_name`参数是创建并重命名文件的原始名称和新名称。使用命令列表来添加（并配置）我们希望在以后某个时间点执行的命令。请注意，除非我们明确为每个命令调用`execute()`，否则命令不会执行：

```py
def main():
     orig_name, new_name = 'file1', 'file2'
     commands = (
         CreateFile(orig_name),
         ReadFile(orig_name),  
         RenameFile(orig_name, new_name)
     )

     [c.execute() for c in commands]
```

下一步是询问用户他们是否想要撤销已执行的命令。用户可以选择是否撤销命令。如果他们选择撤销，将对命令列表中的所有命令执行`undo()`操作。然而，由于并非所有命令都支持撤销，因此使用异常处理来捕获（并忽略）当`undo()`方法缺失时产生的`AttributeError`异常。代码看起来如下所示：

```py
answer = input('reverse the executed commands? [y/n] ')  

if answer not in 'yY':
   print(f"the result is {new_name}")  
   exit()  

for c in reversed(commands):  
   try:  
   c.undo()  
except AttributeError as e:  
    print("Error", str(e))
```

对于这种情况使用异常处理是一种可接受的做法，但如果你不喜欢，你可以通过添加`supports_undo()`或`can_be_undone()`来显式检查命令是否支持撤销操作。再次强调，这并不是强制性的。

让我们看看使用`python command.py`命令行执行的两个示例执行。

在第一种情况下，没有命令的撤销：

```py
[creating file 'file1']
[reading file 'file1']
hello world
[renaming 'file1' to 'file2']
reverse the executed commands? [y/n] y
[renaming 'file2' back to 'file1']
Error 'ReadFile' object has no attribute 'undo'
deleting file file1
```

在第二种情况下，我们有命令的撤销：

```py
[creating file 'file1']
[reading file 'file1']
hello world
[renaming 'file1' to 'file2']
reverse the executed commands? [y/n] n
the result is file2
```

但等等！让我们看看我们的命令实现示例中可以改进的地方。需要考虑的事项包括以下内容：

+   如果我们尝试重命名一个不存在的文件会发生什么？

+   对于存在但因为我们没有适当的文件系统权限而无法重命名的文件怎么办？

我们可以通过进行某种错误处理来尝试改进实用工具。检查`os`模块中函数的返回状态可能很有用。我们可以在尝试删除操作之前使用`os.path.exists()`函数检查文件是否存在。

此外，文件创建实用工具使用文件系统决定的默认文件权限创建文件。例如，在 POSIX 系统中，权限是`-rw-rw-r--`。你可能希望允许用户通过将适当的参数传递给`CreateFile`来提供他们自己的权限。你该如何做到这一点？提示：一种方法是通过使用`os.fdopen()`。

现在，这里有一些事情要你思考。我之前提到过，命令不一定是类。这就是删除实用工具的实现方式；只有一个`delete_file()`函数。这种方法的优缺点是什么？提示：是否可以将删除命令放入命令列表中，就像对其他命令所做的那样？我们知道在 Python 中函数是一等公民，所以我们可以做如下操作（参见`first-class.py`文件）：

```py
import os  
verbose = True  
class CreateFile:
    def undo(self):  
        try:
            delete_file(self.path)
        except:
            print('delete action not successful...')
            print('... file was probably already deleted.')

def main():
    orig_name = 'file1'  
    df=delete_file  
    commands = [CreateFile(orig_name),] 
    commands.append(df)  

    for c in commands:  
        try:  
            c.execute()  
        except AttributeError as e:  
            df(orig_name)  

    for c in reversed(commands):  
        try:  
            c.undo()  
        except AttributeError as e:  
            pass

if __name__ == "__main__":  
    main()
```

运行`first-class.py`会给出以下输出：

```py
[creating file 'file1']
deleting file file1...
deleting file file1...
delete action not successful...
... file was probably already deleted.
```

我们可以看到，这个实现示例的变体按预期工作，因为第二次删除操作抛出一个错误，表示操作未成功。

话虽如此，这个程序还有一些潜在的改进可以考虑。首先，我们现有的代码并不统一；我们过度依赖异常处理，这不是程序的正常流程。虽然我们实现的所有其他命令都有一个`execute()`方法，但在这个情况下，却没有`execute()`。

此外，当前删除文件工具没有撤销支持。如果我们最终决定为它添加撤销支持，会发生什么？通常，我们会为表示命令的类添加一个`undo()`方法。然而，在这种情况下，没有类。我们可以创建另一个函数来处理撤销，但创建一个类是更好的方法。

# 摘要

在本章中，我们介绍了命令模式。使用这个设计模式，我们可以将操作，如复制和粘贴，封装为一个对象。使用这个模式，我们可以在任何时候执行命令，而无需在创建时执行，而执行命令的客户端代码不需要了解其实现的任何细节。此外，我们可以分组命令并按特定顺序执行它们。

为了演示命令，我们在 Python 的`os`模块之上实现了一些基本的文件工具。我们的工具支持撤销，并且有一个统一的接口，这使得分组命令变得容易。

下一章将介绍观察者模式。

# 问题

这些问题的答案可以在本书末尾的*评估*部分找到：

1.  命令模式的优点是什么？

1.  从应用程序客户端的角度来看，命令模式具体有什么好处？

1.  命令模式是如何在文件管理的 Python 示例中实现的？
