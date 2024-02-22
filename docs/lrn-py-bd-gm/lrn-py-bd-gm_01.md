# 第一章：了解 Python-设置 Python 和编辑器

Python 在数据和分析行业中臭名昭著，但在游戏行业中仍然是一个隐藏的工具。在使用其他游戏引擎（如 Unity 和 Godot）制作游戏时，我们倾向于将设计逻辑与核心编程原则相结合。但在 Python 的情况下，主要是问题分析和编程范式的融合。程序流程或结构是一个与其编程范式交织在一起的序列。编程范式正如其名称所示，它便于程序员以最经济和高效的方式编写问题的解决方案。例如，用两行代码而不是十行代码编写程序是使用编程范式的结果。程序流程分析或结构分析的目的是揭示有关需要调用各种设计模式的程序的信息。

在本章中，我们将学习以下主题：

+   使用 Python 介绍编程

+   安装 Python

+   Python 的构建模块

+   安装 PyCharm IDE

+   编写*Hello World*之外的编程代码

# 技术要求

以下是您在本书中需要的最低硬件要求的列表：

+   具有至少 4GB RAM 的工作 PC

+   外部鼠标适配器（如果您使用笔记本电脑）

+   至少需要 5GB 的硬盘空间来下载外部 IDE 和 Python 软件包

您需要以下软件才能充分利用本书（我们将在本章中下载所有这些软件）：

+   各种开源 Python 软件包，如 pygame，pymunk 和 pyopenGL

+   Pycharm IDE（社区版），您可以在[`www.jetbrains.com/pycharm/`](https://www.jetbrains.com/pycharm/)找到

+   各种开源软件包，如`pygame`和`pycharm`

+   本章的代码可以在本书的 GitHub 存储库中找到：[`github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter01`](https://github.com/PacktPublishing/Learning-Python-by-building-games/tree/master/Chapter01)

查看以下视频以查看代码的运行情况：

[`bit.ly/2o2pVgA`](http://bit.ly/2o2pVgA)

# 使用 Python 介绍编程

编程的古老格言如下：

“编码基本上是用于开发应用程序、网站和软件的计算机语言。没有它，我们将无法使用我们所依赖的主要技术，如 Facebook、智能手机、我们选择查看最喜欢的博客的浏览器，甚至博客本身。这一切都运行在代码上。”

我们对此深表认同。计算机编程既可以是一项有益的工作，也可能是一项乏味的活动。有时，我们可能会遇到无法找到程序中捕获的异常（程序的意外行为）的调整的情况，后来我们发现错误是因为错误的模块或不良的实践。编写程序类似于*写作文章*；首先，我们必须了解文章的模式；然后，我们分析主题并写下来；最后，我们检查语法。

与写作文章的过程类似，编写代码时，我们必须分析编程语言的模式或语法，然后分析问题，然后编写程序。最后，我们检查它的语法，通常是通过 alpha 和 beta 测试来完成的。

本书将尝试让您成为一个可以分析问题、建立高贵逻辑并提出解决问题的想法的人。我们不会让这段旅程变得单调；相反，我们将通过在每一章节中构建游戏来学习 Python 语法。到本书结束时，您将会像一个程序员一样思考——也许不是专业的程序员，但至少您将已经掌握了使用 Python 制作自己程序的技能。

在本书中，您将学习到两个关键的内容：

+   首先，您将学习 Python 的词汇和语法。我不是指学习 Python 的理论或历史。首先，我们必须学习 Python 的语法；然后，我们将看看如何使用该语法创建语句和表达式。这一步包括收集数据和信息，并将其存储在适当的数据结构中。

+   然后，您将学习与调用适当方法的想法相对应的程序。这个过程包括使用在第一步收集的数据来获得预期的输出。这第二步不是特定于任何编程语言。这将教我们各种编程原型，而不仅仅是 Python。

在学习了 Python 之后，学习任何其他编程语言都会更容易。您将观察到的唯一区别是其他编程语言的语法复杂性和程序调试工具。在本书中，我们将尽可能多地学习各种编程范式，以便我们可以开始编程生涯。

您对 Python 还有疑问吗？

让我们看一下一些使用 Python 制作的产品：

+   没有提到谷歌的清单是不完整的。他们在他们的网络搜索系统和页面排名算法中使用它。

+   迪士尼使用 Python 进行创意过程。

+   BitTorrent 和 DropBox 都是用 Python 编写的。

+   Mozilla Firefox 用它来探索内容，并且是 Python 软件包的主要贡献者。

+   NASA 用它进行科学研究。

清单还在继续！

让我们简单地看一下代码程序是如何工作的。

# 解释代码程序

为了简单地解释代码程序是如何工作的，让我们以制作煎蛋卷为例。您首先从食谱书中学习基础知识。首先，您收集一些器具，并确保它们干净干燥。之后，您打蛋，加盐和胡椒，直到混合均匀。然后，您在不粘锅中加入黄油，加入蛋液，煮熟，甚至可以倾斜锅来检查煎蛋卷的每个部分是否煮熟。

在编程方面，首先，我们谈论收集我们的工具，比如器具和鸡蛋，这涉及收集将被我们程序中的指令操纵的数据。之后，我们谈论煮鸡蛋，这就是你的方法。我们通常在方法中操纵数据，以便以对用户有意义的形式得到输出。这里，输出是一个煎蛋卷。

给程序提供指令是程序员的工作。但让我们区分客户和程序员。如果您使用的产品是让您向计算机发出指令来为您执行任务，那么您是客户，但如果您设计了为您为所有人创建的产品完成任务的指令，这表明您是程序员。只是“为一个人”还是“为所有人”决定了用户是客户还是程序员。

我们将在 Windows 命令提示符或 Linux 终端中使用的一些指令是用来打开我们机器的目录。有两种执行此操作的方法。您可以使用图形用户界面，也可以使用终端或命令提示符。如果您在相应的字段中键入`dir`命令，那么您现在正在告诉计算机在该位置显示目录。在任何编程语言中都可以做同样的事情。在 Python 中，我们有模块来为我们做这个。我们必须在使用之前导入该模块。Python 提供了许多模块和库来执行此类操作。在诸如 C 之类的过程式编程语言中，它允许与内存进行低级交互，这使得编码更加困难，但使用 Python 可以更容易地使用标准库，这使得代码更短更易读。《如何像计算机科学家一样思考学习 Python》的作者大卫·比兹利曾经被问到，“为什么选择 Python？”他简单地回答说，“Python 只是更有趣和更高效”。

# 与 Python 交谈

Python 已经存在了很多年（将近 29 年），尽管它经历了许多升级，但它仍然是最容易让初学者学习的语言。这主要是因为它可以与英语词汇相关联。类似于我们用英语单词和词汇做陈述，我们可以用 Python 语法编写陈述和操作，命令可以解释、执行并给我们提供结果。我们可以用条件和流控制来反映某物的位置，比如*去那里*作为一个命令。学习 Python 的语法非常容易；真正的任务是利用 Python 提供的所有资源来构建全新的逻辑，以解决复杂的问题。仅仅学习基本的语法和写几个程序是不够的；你必须练习足够多，以便能够提出解决现实问题的革命性想法。

我们在英语词典中有很多词汇。与英语词典不同，Python 只包含少量单词，我们通常称之为保留字。总共有 33 个。它们是指示 Python 解释器执行特定操作的指令。修改它们是不可能的——它们只能用于执行特定任务。此外，当我们调用打印语句并在其中写一些文本时，预期它会打印出该消息。如果你想制作一个从用户那里获取输入的程序，调用打印语句是无用的；必须调用输入语句才能实现这一点。以下表格显示了我们的 33 个保留字：

| `False` | `class` | `finally` | `is` | `return` |
| --- | --- | --- | --- | --- |
| `None` | `continue` | `for` | `lambda` | `try` |
| `True` | `def` | `from` | `nonlocal` | `while` |
| `and` | `del` | `global` | `not` | `with` |
| `as` | `elif` | `if` | `or` | `yield` |
| `assert` | `else` | `import` | `pass` |  |
| `break` | `except` | `in` | `raise` |  |

这些单词都可以在我们的英语词典中找到。此外，如果我们在词典中搜索单词`return`，它只会给我们返回原始位置的动词含义。Python 中也使用相同的语义；当你在函数中使用 return 语句时，你是在从函数中取出一些东西。在接下来的章节中，我们将看到所有这些关键字的用法。

现在我们已经开始学习如何通过检查其关键字来使用 Python 进行对话，我们将安装 Python。做好准备，打开你的机器，开始一些有趣的事情。

# 安装 Python

在本节中，我们将看看如何在 Windows 和 macOS 上安装 Python。

# 对于 Windows 平台

Python 不会预装在 Windows 上。我们必须从官方网站手动下载并安装它。让我们看看如何做到这一点：

1.  首先，打开你喜欢的浏览器，打开以下网址：[`www.Python.org/`](https://www.python.org/)。

1.  你将被引导到下图所示的页面。一旦你被重定向到 Python 的官方网站，你会看到三个部分：下载、文档和工作。点击页面底部的下载部分：

![](img/418d7334-70bb-4486-a8ec-a9fb501b7e51.png)

1.  你会看到一个文件列表，如下截图所示。选择适合你平台的文件。在本节中，我们将看一下 Windows 的安装，所以我们会点击 Windows 可执行文件链接。如下截图所示：

![](img/a9ed1edb-d919-4d0c-85cc-b1777249bac6.png)

1.  点击后，你将得到一个需要下载的文件。打开下载的文件后，你将得到安装程序，如下所示：

![](img/32dc46f2-a12d-4779-97d3-0836ea61f2d2.png)

1.  在安装程序中，确保您选中“将 Python 添加到 PATH”框。这将在我们的环境变量中放置 Python 库文件，以便我们可以执行我们的 Python 程序。之后，您将收到有关其成功安装的消息：

![](img/cd48caf9-57f4-4522-8a64-2781194055fa.png)

1.  按下 Windows 键+*R*打开运行，然后在运行选项卡中键入`cmd`打开 Windows 命令提示符。然后，在命令 shell 中键入`Python`：

![](img/1c6076a2-5cd9-4d90-858d-cb9f9d069092.png)

如果您得到前面截图中显示的 Python 版本，那么 Python 已成功安装在您的计算机上。恭喜！现在，您可以通过使用 Python 编写您的第一个程序来动手实践。

如果出现错误提示**Python is not recognized as an internal or external command**，则必须显式将 Python 添加到路径环境变量中。按照以下步骤执行：

1.  打开控制面板，导航到“系统和安全”，然后转到“系统”以查看有关您系统的基本信息。

1.  打开高级系统设置，然后选择“环境变量...”。

1.  在“变量”部分，搜索“路径”。选择“路径”变量，然后按“编辑...”选项卡。

1.  在“编辑环境变量”选项卡中单击“新建”。

1.  添加此路径，使其指向您的 Python 安装目录，即 C:\Users\admin\AppData\Local\Programs\Python\Python37\。

1.  单击“确定”按钮以保存这些更改：

![](img/7b357caf-b0c1-43a3-8edd-8975e0497710.png)

现在，我们已成功在 Windows 上安装了 Python。如果您使用的是 Mac，下一节将帮助您也访问 Python。

# 对于 Mac 平台

Python 在 Mac OS X 上预先安装。要检查您安装的 Python 版本，您应该打开命令行并输入`Python --version`。如果您得到 3.5 或更新的版本号，您就不需要进行安装过程，但如果您有 2.7 版本，您应该按照以下说明下载最新可用版本：

1.  打开浏览器，输入[`www.Python.org/downloads/`](https://www.python.org/downloads/)。您将被发送到以下页面：

![](img/663aa19c-0e4d-47ca-87f9-a0688af61e29.jpg)

1.  单击 macOS 64 位/32 位安装程序。您将获得一个`.pkg`文件。下载它。然后，导航到已安装的目录并单击该安装程序。您将看到以下选项卡。按“继续”以启动安装程序：

![](img/60d48855-ca1a-4a01-aca4-2655797228b4.jpg)

每当您下载 Python 时，一堆软件包将安装在您的计算机上。我们不能直接使用这些软件包，所以我们应该为每个独立的任务单独调用它们。要编写程序，我们需要一个环境，我们可以在其中调用 Python，以便它可以为我们完成任务。在下一节中，我们将探索 Python 提供的用户友好环境，我们可以在其中编写自己的程序并运行它们以查看它们的输出。

现在，您已在 Mac OS X 上安装了 Python 3.7 版本，您可以打开终端并使用`python --version`命令检查您的 Python 版本。您将看到 Python 2.7.10。原因是 Mac OS X 预先安装了 Python 2.7+版本。要使用更新的 Python 版本，您必须使用`python3`命令。在终端中键入以下命令并观察结果：

```py
python3 --version
```

现在，为了确保 Python 使用您刚刚安装的较新版本的解释器，您可以使用一种别名技术，将当前工作的 Python 版本替换为 Python3。要执行别名，您必须按照以下步骤执行：

1.  打开终端并输入`nano ~/.bash_profile`命令以使用 nano 编辑器打开 bash 文件。

1.  接下来，转到文件末尾（在导入路径之后）并键入`alias python=python3`命令。要保存 nano 文件，请按*Ctrl* + *X*，然后按*Y*保存。

现在，再次打开您的终端，并输入我们之前使用的相同命令来检查我们拥有的 Python 版本。它将更新到较新版本的 Python。从现在开始，为了从 Mac 运行任何 Python 文件，您可以使用这个 Python 命令，后面跟着文件的签名或文件名。

# 介绍 Python Shell 和 IDLE

Python Shell 类似于 Windows 的命令提示符和 Linux 和 Mac OS X 的终端，您可以在其中编写将在文件系统中执行的命令。这些命令的结果会立即在 shell 中打印出来。您还可以使用任何终端中的 Python 命令（> python）直接访问此 shell。结果将包含由于代码执行不正确而导致的异常和错误，如下所示：

```py
>>> imput("Enter something")
Traceback (most recent call last):
  File "<pyshell#5>", line 1, in <module>
    imput()
NameError: name 'imput' is not defined

>>> I love Python
SyntaxError: invalid syntax
```

正如您所看到的，我们遇到了一个错误，Python IDE 明确告诉我们我们遇到的错误名称，在这种情况下是`NameError`（一种语法错误）。`SyntaxError`是由于代码的不正确模式而发生的。在前面的代码示例中，当您编写`I love Python`语法时，这对 Python 解释器来说什么都不意味着。如果要纠正这个问题，您应该编写正确的命令或正确定义某些内容。写`imput`而不是 input 也是语法错误。

逻辑错误或语义错误即使您的程序语法正确也会发生。然而，这并不能解决您的问题领域。它们很难追踪，因此很危险。程序完全正确，但并没有解决它本来要解决的任何问题。

当您在计算机上下载 Python 软件包时，一个名为 IDLE（Python 内置 IDE）的**Python 集成开发环境**（**IDE**）会自动下载到您的计算机上。您可以在搜索栏中输入`IDLE`来进入这个环境。IDLE 是一个免费的开源程序，提供了两个界面，您可以在其中编写代码。我们可以在 IDLE 中编写脚本和终端命令。

现在我们已经熟悉了在 Python Shell 中不应该做什么，让我们谈谈 Python Shell 的细节——这是一个您可以编写 Python 代码的环境。

# Python Shell 的细节

正如我们之前提到的，在这一部分，我们将参观 Python 的细节。这包括 Python 的内置 shell，Python 的文本编辑器（通常称为 Python 脚本）和 Python 文档页面。

按照以下步骤了解 Python Shell 的细节：

1.  当您打开 Python Shell 时，您会看到以下窗口。您在 shell 中看到的第一件事是 Python 的当前版本号：

![](img/a0bf29a2-3565-4a42-b1ed-133c27e51b65.png)

1.  在 Python shell 中，有三个角括号相邻放置在一起，就像这样：`>>>`。您可以从那里开始编写您的代码：

![](img/90030578-dd0c-4d5c-b99c-0ba50a871dd7.png)

1.  按下*F1*打开 Python 文档，或转到帮助选项卡，单击 Python Docs F1（在 Windows 机器上）。要在线访问文档，请转到[`docs.python.org/3/`](https://docs.python.org/3/)：

![](img/152dcd04-ce27-4a80-bf41-db37e2330bdd.png)

希望您现在对 Python Shell 已经很熟悉了。我们将在 shell 中编写大量代码，因此请确保通过自定义或长时间玩耍来熟悉它。完成后，您可以继续下一节，在那里您将学习在编写第一个 Python 程序之前需要了解的内容。

# Python 的基本组成部分

在编写 Python 程序时，我们使用一些常规模式。Python 作为一种高级语言，不关心低级例程，但具有与它们交互的能力。Python 由六个构建块组成。用 Python 制作的每个程序都围绕着它们展开。这些构建块是输入、输出、顺序执行、条件、递归和重用。现在让我们来详细了解一下它们：

+   **输入**：输入无处不在。如果你用 Python 制作一个应用程序，它主要处理用户输入的格式，以便收集有意义的结果。Python 中有一个内置的`input()`方法，因此我们可以从用户那里获取数据输入。

+   **输出**：在我们操纵用户输入的数据之后，是时候向用户呈现它了。在这一层，我们利用设计工具和演示工具来格式化有意义的输出并发送给用户。

+   **顺序执行**：这保留了语句的执行顺序。在 Python 中，我们通常使用缩进，即表示作用域的空格。任何在零级缩进的命令都会首先执行。

+   **条件**：这些为程序提供流程控制。基于比较，我们制定逻辑，使代码流动，并执行或跳过它。

+   递归：这是需要做的任何事情，直到满足某些条件。我们通常称它们为**循环**。

+   **重用**：编写一次代码，使用无数次。重用是一种范式，我们编写一组代码，给它一个引用，并在需要时使用它。函数和对象提供了可重用性。

在 Python Shell 中编写程序对大多数程序员来说可能很容易调试，但从长远来看可能会产生额外的开销。如果你想保存代码以备将来参考，或者你想编写多行语句，你可能会被 Python 解释器的功能不足所压倒。为了解决这个问题，我们必须创建一个脚本文件。它们被称为脚本，因为它们允许你在单个文件中编写多行语句，然后立即运行。当我们有多个数据存储和文件要处理时，这将非常方便。你可以通过扩展名来区分 Python 文件和其他文件，也就是`.py`。你还应该将 Python 脚本文件保存为`.py`扩展名。

要从终端或 Windows 命令提示符中运行你的脚本文件，你必须告诉 Python 解释器通过文件名运行该文件，就像这样：

```py
$ Python Python_file_name.py
```

在上述命令中，`$`是操作系统提示符。首先，你必须使用`Python`命令调用 Python 解释器，并告诉它执行其旁边的文件名。

如果你想在终端中查看`Python`文件的内容，请使用以下命令：

```py
$ cat Python_file_name.py
$ nano Python_file_name.py
```

要退出 Python 终端，在终端中写入`exit()`命令。

现在我们已经学会了如何打开和退出 Python 环境的界面，我们必须了解它的构建模块。许多初学者错误地认为程序只有两个构建模块：输入和输出。在下一节中，我们将看到如何通过使用编程的六个构建模块来驳斥这一假设。

编程最困难的部分是学习编程范式，比如面向对象编程、DRY 原则或线性时间复杂度模型。如果你掌握了这些原型，学习任何新的编程语言都将变得轻而易举。话虽如此，使用 Python 学习所有这些范式要比 Java 或 C#容易得多，因为在 Python 中，代码会更短，语法也更符合英语习惯：

![](img/940ac0b9-a662-43c9-9a87-88d7accbce8b.png)

在我们编写第一个程序之前，我们将安装另一个 IDLE，以备后面的章节中我们将编写复杂的游戏。在这些类型的游戏中，IDLE 提供的功能是不够的，因此我们将看到如何在下一节中安装 PyCharm——一个高级的 IDLE。

# 安装 PyCharm IDE

在本章的前面，我们发现了 IDLE。我们已经看到了一个环境，我们可以在其中编写代码并立即获得输出。但是，您可以想象一下，如果我们有很多代码要一次执行，可能是 1000 行代码，一次执行一行。我们必须通过编写脚本来解决这个问题，这是 Python 代码的集合。这将一次执行，而不是在 IDLE 的 shell 中逐行执行。

如果您想编写脚本，请按照以下步骤操作：

1.  从您的 PC 中打开搜索选项卡，然后键入`IDLE`。

1.  点击文件选项卡。

1.  按下 New File。

1.  将生成一个新文件。您可以在单个文件中编写多个表达式、语句和命令。以下屏幕截图的左侧显示了 Python 脚本，您可以在其中编写多行语句，而以下屏幕截图的右侧显示了 Python Shell，您将在其中执行脚本并获得即时结果：

![](img/83321431-77d1-42cd-8cef-856e1ffb2058.png)

在编写脚本完成后，您必须在运行之前保存它。要保存文件，请转到文件并单击保存。通过在末尾放置`.py`扩展名为您的脚本提供适当的文件名，例如`test.py`。按下*F5*执行您的脚本文件。

在本书中，我们将构建许多游戏，其中我们将处理图像、物理、渲染和安装 Python 包。这个 IDE，也就是 IDLE，无法提供智能 IDE 功能，比如代码完成、集成和插件以及包的分支。因此，我们必须升级到最好的 Python 文本丰富的 IDE，也就是 PyCharm IDE。让我们开始吧：

1.  访问[`www.jetbrains.com/pycharm/`](https://www.jetbrains.com/pycharm/)下载 PyCharm 环境。安装 PyCharm 与安装任何其他程序一样简单。从网站下载安装程序后，点击该安装程序。您应该会看到以下窗口：

![](img/c9a4caf9-823a-4248-81e6-f5ab098bec4c.png)

1.  点击<Next>按钮并将其安装在适当的驱动器上。安装完成后，在搜索栏中搜索`PyCharm`并打开它。您应该会看到以下窗口：![](img/ac6b5f5e-7caa-41c4-9661-e5a39606e5ca.png)

1.  现在，点击+创建新项目并给您的项目命名。要创建新的 Python 文件，请在项目名称上单击左键，单击 New，然后单击 Python File 选项卡：![](img/c7753959-cb44-41e4-b346-8f4a7b602dbb.png)

现在，我们拥有了掌握本书所需的一切——我的意思是工具，但显然，我们必须学习 Python 的每种可能的范式来掌握 Python 的概念。现在您已经全副武装了这些工具，让我们编写我们的第一个有效的 Python 程序，*No Hello World*。

# 编程代码没有 Hello World

在编程世界中有一个传统，即将*Hello World*打印为我们的第一个程序。让我们打破常规，使我们的第一个程序成为从用户那里获取输入并将其打印到控制台的程序。按照以下步骤执行您的第一个程序：

1.  打开您的 IDLE 并输入以下命令：

```py
 >>> print(input("Enter your Name: "))
```

1.  按下*Enter*执行命令。您将收到一条消息，上面写着输入您的姓名：。输入您的姓名并按*Enter*。您将看到输出打印您刚刚传递的姓名。

我们在这里使用了两个命令，也称为函数。我们将在接下来的章节中学习它们。现在让我们来看看这两个函数：

+   `input()`是 Python 的内置函数，将从用户那里获取输入。空格也包括在字符中。

+   `print()`是 Python 的内置函数，将打印括号内传递的任何内容。

现在我们已经开始使用 Python 的内置 IDLE 编写我们的第一个程序，轮到您测试 IDLE 的工作原理了。由于我们将使用 IDLE 构建大量游戏，请确保您熟悉其界面。本章学习的核心编程模块，如 Python 关键字和输入-打印函数，非常重要，因为它们帮助我们构建可以从用户那里获取输入并显示的程序。

# 总结

在本章中，我们对 Python 的基础知识进行了概述，并学习了它与英语的词汇有多么相似。我们在计算机上安装了 Python 软件包，并查看了 Python 的预安装集成开发环境（IDE）IDLE。我们看到了如何在 Python IDE 上编写脚本以及如何执行它们。然后，我们在计算机上安装了功能丰富的 Python 文本编辑器 PyCharm IDE。我们编写了我们的第一个 Python 程序，它能够从用户那里获取输入并在屏幕上显示。

本章中您所学到的技能对于构建程序的流程至关重要。例如，我们的程序能够处理输入/输出数据。任何用 Python 制作的游戏都必须对用户或玩家进行交互，这是通过输入和输出界面来实现的。在本章中，我们学习了如何从用户那里获取输入并显示它。随着我们继续阅读本书，我们将探索各种构建程序的方式，包括处理来自鼠标、键盘和屏幕点击的用户事件。

下一章将至关重要，因为我们将学习 Python 的基本要素，如值、类型、变量、运算符和模块。我们还将开始构建一个井字棋游戏。