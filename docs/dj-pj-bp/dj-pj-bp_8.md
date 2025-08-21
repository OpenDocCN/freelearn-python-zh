# 附录 A.开发环境设置详细信息和调试技巧

本附录将更详细地介绍我们在整本书中一直在使用的 Django 开发环境设置。我们将深入了解设置的细节，并解释我们采取的每个步骤。我还将向您展示一种调试 Django 应用程序的技术。在本附录中，我们将假设我们正在设置的项目是第一章中的 Blueblog 项目。

我们将首先创建一个根目录，然后`cd`到该目录，以便所有命令都在其中运行：

```py
> mkdir blueblog
> cd blueblog

```

这没有技术原因。我只是喜欢将与项目相关的所有文件放在一个目录中，因为当您必须添加与项目相关的其他文件（如设计和其他文档）时，这样做会更容易组织。

接下来，我们将创建一个虚拟环境用于该项目。虚拟环境是一个功能，允许您创建 Python 的轻量级安装，以便每个项目都可以拥有自己使用的所有库的安装。当您同时在多个项目上工作，并且每个项目需要某个库的单独版本时，这将非常有用。例如，在工作中，我曾经不得不同时处理两个项目。一个需要 Django 1.4；另一个需要 Django 1.9。如果我没有使用虚拟环境，将很难同时保留 Django 的两个版本。

虚拟环境还可以让您保持 Python 环境的清洁，这在最终准备将应用程序部署到生产服务器时非常重要。当将应用程序部署到服务器时，您需要能够准确地复制与开发机器中相同的 Python 环境。如果您不为每个项目使用单独的虚拟环境，您将需要准确确定项目使用的 Python 库，然后仅在生产服务器上安装这些库。有了虚拟环境，您不再需要花时间弄清楚安装的 Python 库中哪些与您的项目相关。您只需创建虚拟环境中安装的所有库的列表，并在生产服务器上安装它们，确信您不会错过任何内容或安装任何多余的内容。

如果您想了解更多关于虚拟环境的信息，可以阅读官方文档[`docs.python.org/3/library/venv.html`](https://docs.python.org/3/library/venv.html)。

要创建虚拟环境，我们使用`pyvenv`命令：

```py
> pyvenv blueblogEnv 

```

这将在`blueblogEnv`文件夹内创建一个新的环境。创建环境后，我们激活它：

```py
> 
source blueblogEnv/bin/activate

```

激活环境可以确保我们运行的任何 Python 命令或我们安装的任何库都将使用激活的环境。接下来，在我们的新环境中安装 Django 并启动我们的项目：

```py
> pip install django
> django-admin.py startproject blueblog src

```

这将创建一个名为`src`的目录，其中包含我们的 Django 项目。您可以将目录命名为任何您喜欢的名称；这只是我喜欢的约定。

这就是我们开发环境的设置。

# 使用 pdb 调试 Django 视图

在 Django 应用程序中，您经常会遇到一些不太清楚的问题。当我遇到棘手的错误，特别是在 Django 视图中时，我会使用 Python 调试器来逐步执行我的视图代码并调试问题。为此，您需要在认为问题存在的地方的视图中放入这行代码：

```py
import pdb; pdb.set_trace()
```

然后，下次加载与该视图相关的页面时，你会发现你的浏览器似乎没有加载任何内容。这是因为你的 Django 应用现在已经暂停了。如果你在运行`runserver`命令的控制台中查看，你应该会看到一个`pdb`的提示。在提示符中，你可以输入当前 Python 范围内（通常是你正在调试的视图的范围）可用的任何变量的名称，它会打印出该变量的当前值。你还可以运行一系列其他调试命令。要查看可用功能的完整列表，请查看 Python 调试器的文档[`docs.python.org/3/library/pdb.html`](https://docs.python.org/3/library/pdb.html)。

一个很好的 Stack Overflow 问题，列出了一些其他有用的答案和调试技巧，链接是[`stackoverflow.com/questions/1118183/how-to-debug-in-django-the-good-way`](http://stackoverflow.com/questions/1118183/how-to-debug-in-django-the-good-way)。

# 在 Windows 上开发

如果你在阅读本书时要使用 Windows 操作系统，请注意有一些事情需要做出不同的处理。首先，本书中提供的所有指令都是基于 Linux/Mac OS X 环境的，有些指令可能无法直接使用。最重要的变化是 Windows 如何处理文件路径。在 Linux/OS X 环境中，路径是用正斜杠写的。书中提到的所有路径都是类似格式的，例如，`PROJECT_DIR/main/settings.py`。在 Windows 上引用这些路径时，你需要将正斜杠改为反斜杠。这个路径将变成`PROJECT_DIR\main\settings.py`。

其次，虽然 Python 通常包含在 Linux/OS X 中，或者很容易安装，但你需要按照`https://www.python.org/downloads/windows/`上的说明在 Windows 上安装 Python。安装了 Python 之后，你可以按照[`docs.djangoproject.com/en/stable/howto/windows/`](https://docs.djangoproject.com/en/stable/howto/windows/)上的说明安装 Django。

有一些其他的东西需要在 Windows 上进行修改。我在书中提到了这些，但可能会漏掉一些。如果是这样，通过谷歌搜索通常会找到答案。如果找不到，你可以在 Twitter 上找到我，我的用户名是`@theonejb`，我会尽力帮助你。
