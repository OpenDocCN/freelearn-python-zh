# 第二章。语句和语法

在本章中，我们将查看以下配方：

+   编写 Python 脚本和模块文件

+   编写长行代码

+   包括描述和文档

+   在文档字符串中更好的 RST 标记

+   设计复杂的 if...elif 链

+   设计一个终止的 while 语句

+   避免 break 语句可能出现的问题

+   利用异常匹配规则

+   避免 except:子句可能出现的问题

+   使用 raise from 语句链接异常

+   使用 with 语句管理上下文

# 介绍

Python 语法设计得非常简单。有一些规则；我们将查看语言中一些有趣的语句，以了解这些规则。仅仅看规则而没有具体的例子可能会令人困惑。

我们将首先介绍创建脚本文件的基础知识。然后我们将继续查看一些常用语句。Python 语言中只有大约二十种不同类型的命令语句。我们已经在第一章中看过两种语句，*Numbers, Strings, and Tuples*：赋值语句和表达式语句。

当我们写这样的东西时：

```py
 **>>> print("hello world")** 

 **hello world** 

```

我们实际上执行的是一个只包含函数`print()`评估的语句。这种语句-在其中我们评估一个函数或对象的方法-是常见的。

我们已经看到的另一种语句是赋值语句。Python 在这个主题上有很多变化。大多数时候，我们将一个值赋给一个变量。然而，有时我们可能会同时给两个变量赋值，就像这样：

```py
 **quotient, remainder = divmod(355, 113)** 

```

这些配方将查看一些更复杂的语句，包括`if`，`while`，`for`，`try`，`with`和`raise`。在探索不同的配方时，我们还将涉及其他一些。

![](img/614271.jpg)

# 编写 Python 脚本和模块文件-语法基础

为了做任何真正有用的事情，我们需要编写 Python 脚本文件。我们可以在交互`>>>`提示符下尝试语言。然而，对于真正的工作，我们需要创建文件。编写软件的整个目的是为我们的数据创建可重复的处理。

我们如何避免语法错误，并确保我们的代码与常用的代码匹配？我们需要查看一些*style*的常见方面-我们如何使用空白来澄清我们的编程。

我们还将研究一些更多的技术考虑因素。例如，我们需要确保以 UTF-8 编码保存我们的文件。虽然 Python 仍然支持 ASCII 编码，但对于现代编程来说，这是一个不好的选择。我们还需要确保使用空格而不是制表符。如果我们尽可能使用 Unix 换行符，我们也会发现事情稍微简单一些。

大多数文本编辑工具都可以正确处理 Unix（换行符）和 Windows 或 DOS（回车换行符）的行尾。任何不能处理这两种行尾的工具都应该避免使用。

## 准备好了

要编辑 Python 脚本，我们需要一个好的编程文本编辑器。Python 自带一个方便的编辑器，IDLE。它工作得相当不错。它让我们可以在文件和交互`>>>`提示之间来回跳转，但它不是一个很好的编程编辑器。

有数十种优秀的编程编辑器。几乎不可能只建议一个。所以我们将建议几个。

ActiveState 有非常复杂的 Komodo IDE。Komodo Edit 版本是免费的，并且与完整的 Komodo IDE 做了一些相同的事情。它可以在所有常见的操作系统上运行；这是一个很好的第一选择，因为无论我们在哪里编写代码，它都是一致的。

请参阅[`komodoide.com/komodo-edit/`](http://komodoide.com/komodo-edit/)。

Notepad++适用于 Windows 开发人员。请参阅[`notepad-plus-plus.org`](https://notepad-plus-plus.org)。

BBEdit 非常适合 Mac OS X 开发人员。请参阅[`www.barebones.com/products/bbedit/`](http://www.barebones.com/products/bbedit/)。

对于 Linux 开发人员，有几个内置的编辑器，包括 VIM、gedit 或 Kate。这些都很好。由于 Linux 倾向于偏向开发人员，可用的编辑器都适合编写 Python。

重要的是，我们在工作时通常会打开两个窗口：

+   我们正在处理的脚本或文件。

+   Python 的`>>>`提示（可能来自 shell，也可能来自 IDLE），我们可以尝试一些东西，看看什么有效，什么无效。我们可能会在 Notepad++中创建脚本，但使用 IDLE 来尝试数据结构和算法。

实际上我们这里有两个配方。首先，我们需要为我们的编辑器设置一些默认值。然后，一旦编辑器正确设置，我们就可以为我们的脚本文件创建一个通用模板。

## 如何做...

首先，我们将看一下我们首选编辑器中需要做的一般设置。我们将使用 Komodo 示例，但基本原则适用于所有编辑器。一旦我们设置了编辑首选项，我们就可以创建我们的脚本文件。

1.  打开首选编辑器。查看首选项页面。

1.  查找首选文件编码的设置。使用 Komodo Edit 首选项，它在**国际化**选项卡上。将其设置为**UTF-8**。

1.  查找缩进设置。如果有一种方法可以使用空格而不是制表符，请检查此选项。使用 Komodo Edit，我们实际上是反过来做的——我们取消**优先使用空格而不是制表符**。

### 注意

规则是：我们想要*空格*；我们不想要*制表符*。

还要将每个缩进的空格设置为四个。这对于 Python 代码来说很典型。它允许我们有几个缩进级别，但仍然保持代码相当窄。

一旦我们确定我们的文件将以 UTF-8 编码保存，并且我们也确定我们使用空格而不是制表符，我们可以创建一个示例脚本文件：

1.  大多数 Python 脚本文件的第一行应该是这样的：

```py
            #!/usr/bin/env python3 

```

这将在你正在编写的文件和 Python 之间建立关联。

对于 Windows，文件名到程序的关联是通过 Windows 控制面板中的一个设置来完成的。在**默认程序**控制面板中，有一个**设置关联**面板。此控制面板显示`.py`文件绑定到 Python 程序。这通常由安装程序设置，我们很少需要更改它或手动设置它。

### 注意

Windows 开发人员可以无论如何包含序言行。这将使 Mac OS X 和 Linux 的人们从 GitHub 下载项目时感到高兴。

1.  在序言之后，应该有一个三引号的文本块。这是我们要创建的文件的文档字符串（称为**docstring**）。这在技术上不是强制性的，但对于解释文件包含的内容至关重要。

```py
        ''' 
        A summary of this script. 
        ''' 

```

因为 Python 的三引号字符串可以无限长，所以可以随意写入必要的内容。这应该是描述脚本或库模块的主要方式。这甚至可以包括它是如何工作的示例。

1.  现在来到脚本的有趣部分：真正执行操作的部分。我们可以编写所有需要完成工作的语句。现在，我们将使用这个作为占位符：

```py
        print('hello world') 

```

有了这个，我们的脚本就有了作用。在其他示例中，我们将看到许多其他用于执行操作的语句。通常会创建函数和类定义，并编写语句来使用函数和类执行操作。

在我们的脚本的顶层，所有语句必须从左边缘开始，并且必须在一行上完成。有一些复杂的语句，其中将嵌套在其中的语句块。这些内部语句块必须缩进。通常情况下，因为我们将缩进设置为四个空格，我们可以按*Tab*键进行缩进。

我们的文件应该是这样的：

```py
    #!/usr/bin/env python3 
    ''' 
    My First Script: Calculate an important value. 
    ''' 

    print(355/113) 

```

## 它是如何工作的...

与其他语言不同，Python 中几乎没有*样板*。只有一行*开销*，甚至`#!/usr/bin/env python3`行通常是可选的。

为什么要将编码设置为 UTF-8？整个语言都是设计为仅使用最初的 128 个 ASCII 字符。

我们经常发现 ASCII 有限制。将编辑器设置为使用 UTF-8 编码更容易。有了这个设置，我们可以简单地使用任何有意义的字符。如果我们将程序保存在 UTF-8 编码中，我们可以将字符如`µ`用作 Python 变量。

如果我们将文件保存为 UTF-8，这是合法的 Python：

```py
    π=355/113 
    print(π) 

```

### 注意

在 Python 中在选择空格和制表符之间保持一致是很重要的。它们都是几乎看不见的，混合它们很容易导致混乱。建议使用空格。

当我们设置编辑器使用四个空格缩进后，我们可以使用键盘上标有 Tab 的按钮插入四个空格。我们的代码将对齐，缩进将显示语句如何嵌套在彼此内。

初始的`#!`行是一个注释：从`#`到行尾的所有内容都会被忽略。像**bash**和**ksh**这样的操作系统 shell 程序会查看文件的第一行，以确定文件包含的内容。文件的前几个字节有时被称为*魔术*，因为 shell 程序正在窥视它们。Shell 程序会寻找`#!`这个两个字符的序列，以确定负责这些数据的程序。我们更喜欢使用`/usr/bin/env`来启动 Python 程序。我们可以利用这一点来通过`env`程序进行 Python 特定的环境设置。

## 还有更多...

*Python 标准库*文档部分源自模块文件中存在的文档字符串。在模块中编写复杂的文档字符串是常见做法。有一些工具，如 Pydoc 和 Sphinx，可以将模块文档字符串重新格式化为优雅的文档。我们将在单独的部分中学习这一点。

此外，单元测试用例可以包含在文档字符串中。像**doctest**这样的工具可以从文档字符串中提取示例并执行代码，以查看文档中的答案是否与运行代码找到的答案匹配。本书的大部分内容都是通过 doctest 验证的。

三引号文档字符串优于`#`注释。`#`和行尾之间的文本会被忽略，并被视为注释。由于这仅限于单行，因此使用得很少。文档字符串的大小可以是无限的；它们被广泛使用。

在 Python 3.5 中，我们有时会在脚本文件中看到这样的东西：

```py
    color = 355/113 # type: float 

```

`# type: float`注释可以被类型推断系统用来确定程序实际执行时可能出现的各种数据类型。有关更多信息，请参阅**Python Enhancement Proposal 484**：[`www.python.org/dev/peps/pep-0484/`](https://www.python.org/dev/peps/pep-0484/)。

有时文件中还包含另一个开销。VIM 编辑器允许我们在文件中保留编辑首选项。这被称为**modeline**。我们经常需要通过在我们的`~/.vimrc`文件中包含`set modeline`设置来启用 modelines。

一旦我们启用了 modelines，我们可以在文件末尾包含一个特殊的`# vim`注释来配置 VIM。

这是一个对 Python 有用的典型 modeline：

```py
    # vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4 

```

这将把 Unicode `u+0009` TAB 字符转换为八个空格，当我们按下*Tab*键时，我们将移动四个空格。这个设置被保存在文件中；我们不需要进行任何 VIM 设置来将这些设置应用到我们的 Python 脚本文件中。

## 另请参阅

+   我们将在*包括描述和文档*和*在文档字符串中编写更好的 RST 标记*这两个部分中学习如何编写有用的文档字符串

+   有关建议的样式的更多信息，请参阅[`www.python.org/dev/peps/pep-0008/`](https://www.python.org/dev/peps/pep-0008/)

# 编写长行代码

有很多时候，我们需要编写非常长的代码行，以至于它们非常难以阅读。许多人喜欢将代码行的长度限制在 80 个字符或更少。这是一个众所周知的图形设计原则，即较窄的行更容易阅读；意见不一，但 65 个字符经常被认为是理想的长度。参见[`webtypography.net/2.1.2`](http://webtypography.net/2.1.2)。

虽然较短的行更容易阅读，但我们的代码可能不遵循这个原则。长语句是一个常见的问题。我们如何将长的 Python 语句分解为更易处理的部分？

## 准备就绪

通常，我们会有一个语句，它非常长且难以处理。比如说我们有这样的东西：

```py
 **>>> import math** 

 **>>> example_value = (63/25) * (17+15*math.sqrt(5)) / (7+15*math.sqrt(5))** 

 **>>> mantissa_fraction, exponent = math.frexp(example_value)** 

 **>>> mantissa_whole = int(mantissa_fraction*2**53)** 

 **>>> message_text = 'the internal representation is {mantissa:d}/2**53*2**{exponent:d}'.format(mantissa=mantissa_whole, exponent=exponent)** 

 **>>> print(message_text)** 

 **the internal representation is 7074237752514592/2**53*2**2** 

```

这段代码包括一个长公式和一个长格式字符串，我们要将值注入其中。这在书中排版时看起来很糟糕。在尝试编辑此脚本时，屏幕上看起来很糟糕。

我们不能简单地将 Python 语句分成块。语法规则明确指出语句必须在单个*逻辑*行上完成。

术语逻辑行是如何进行的一个提示。Python 区分逻辑行和物理行；我们将利用这些语法规则来分解长语句。

## 如何做...

Python 给了我们几种包装长语句使其更易读的方法。

+   我们可以在行尾使用`\`继续到下一行。

+   我们可以利用 Python 的规则，即语句可以跨越多个逻辑行，因为`()`、`[]`和`{}`字符必须平衡。除了使用`()`和`\`，我们还可以利用 Python 自动连接相邻字符串文字的方式，使其成为一个更长的文字；`("a" "b")`与`ab`相同。

+   在某些情况下，我们可以通过将中间结果分配给单独的变量来分解语句。

我们将在本教程的不同部分分别讨论每一个。

### 使用反斜杠将长语句分解为逻辑行

这个技巧的背景是：

```py
 **>>> import math** 

 **>>> example_value = (63/25) * (17+15*math.sqrt(5)) / (7+15*math.sqrt(5))** 

 **>>> mantissa_fraction, exponent = math.frexp(example_value)** 

 **>>> mantissa_whole = int(mantissa_fraction*2**53)** 

```

Python 允许我们使用`\`并换行。

1.  将整个语句写在一行上，即使它很混乱：

```py
 **>>> message_text = 'the internal representation is {mantissa:d}/2**53*2**{exponent:d}'.format(mantissa=mantissa_whole, exponent=exponent)** 

```

1.  如果有*逻辑*断点，在那里插入`\`。有时，没有真正好的断点：

```py
 **>>> message_text = 'the internal representation is \** 

 **... {mantissa:d}/2**53*2**{exponent:d}'.\** 

 **... format(mantissa=mantissa_whole, exponent=exponent)** 

 **>>> message_text** 

 **'the internal representation is 7074237752514592/2**53*2**2'** 

```

为了使其工作，`\`必须是行上的最后一个字符。我们甚至不能在`\`后有一个空格。这很难看出来；因此，我们不鼓励这样做。

尽管这有点难以理解，但`\`总是可以使用的。把它看作是使代码行更易读的最后手段。

### 使用()字符将长语句分解为合理的部分

1.  将整个语句写在一行上，即使它很混乱：

```py
 **>>> import math** 

 **>>> example_value1 = (63/25) * (17+15*math.sqrt(5)) / (7+15*math.sqrt(5))** 

```

1.  添加额外的`()`字符不改变值，但允许将表达式分解为多行：

```py
 **>>> example_value2 = (63/25) * ( (17+15*math.sqrt(5)) / (7+15*math.sqrt(5)) )** 

 **>>> example_value2 == example_value1** 

 **True** 

```

1.  在`()`字符内部断开行：

```py
 **>>> example_value3 = (63/25) * (** 

 **...      (17+15*math.sqrt(5))** 

 **...    / ( 7+15*math.sqrt(5))** 

 **... )** 

 **>>> example_value3 == example_value1** 

 **True** 

```

匹配`()`字符的技术非常强大，适用于各种情况。这是被广泛使用和强烈推荐的。

我们几乎总是可以找到一种方法向语句添加额外的`()`字符。在我们无法添加`()`字符或添加`()`字符无法改善情况的罕见情况下，我们可以退而使用`\`将语句分解为几个部分。

### 使用字符串文字连接

我们可以将`()`字符与另一条规则相结合，该规则结合字符串文字。这对于长而复杂的格式字符串特别有效：

1.  用`()`字符包装一个长字符串值。

1.  将字符串分解为子字符串：

```py
 **>>> message_text = (** 

 **... 'the internal representation '** 

 **... 'is {mantissa:d}/2**53*2**{exponent:d}'** 

 **... ).format(** 

 **... mantissa=mantissa_whole, exponent=exponent)** 

 **>>> message_text** 

 **'the internal representation is 7074237752514592/2**53*2**2'** 

```

我们总是可以将长字符串分解为相邻的片段。通常，当片段被`()`字符包围时，这是最有效的。然后我们可以使用尽可能多的物理行断开。这仅限于那些我们有特别长的字符串值的情况。

### 将中间结果分配给单独的变量

这个技巧的背景是：

```py
 **>>> import math** 

 **>>> example_value = (63/25) * (17+15*math.sqrt(5)) / (7+15*math.sqrt(5))** 

```

我们可以将这分解为三个中间值。

1.  识别整体表达式中的子表达式。将这些分配给变量：

```py
 **>>> a = (63/25)** 

 **>>> b = (17+15*math.sqrt(5))** 

 **>>> c = (7+15*math.sqrt(5))** 

```

这通常相当简单。可能需要一点小心来进行代数运算，以找到合理的子表达式。

1.  用创建的变量替换子表达式：

```py
 **>>> example_value = a * b / c** 

```

这是对原始复杂子表达式的一个必要的文本替换，用一个变量来代替。

我们没有给这些变量起描述性的名称。在某些情况下，子表达式具有一些语义，我们可以用有意义的名称来捕捉。在这种情况下，我们没有理解表达式足够深刻，无法提供深层有意义的名称。相反，我们选择了简短的、任意的标识符。

## 它是如何工作的...

Python 语言手册对逻辑行和物理行进行了区分。逻辑行包含一个完整的语句。它可以通过称为**行连接**的技术跨越多个物理行。手册称这些技术为**显式行连接**和**隐式行连接**。

显式行连接的使用有时是有帮助的。因为很容易忽视，所以通常不受鼓励。这是最后的手段。

隐式行连接的使用可以在许多情况下使用。它通常在语义上与表达式的结构相吻合，因此是受鼓励的。我们可能需要`()`字符作为必需的语法。例如，我们已经将`()`字符作为`print()`函数的语法的一部分。我们可能这样做来分解一个长语句：

```py
 **>>> print(** 

 **...    'several values including',** 

 **...    'mantissa =', mantissa,** 

 **...    'exponent =', exponent** 

 **... )** 

```

## 还有更多...

表达式广泛用于许多 Python 语句。任何表达式都可以添加`()`字符。这给了我们很大的灵活性。

然而，有一些地方可能有一个不涉及特定表达式的长语句。其中最显著的例子是`import`语句 - 它可能变得很长，但不使用可以加括号的任何表达式。

然而，语言设计者允许我们使用`()`字符，以便将一长串名称分解为多个逻辑行：

```py
 **>>> from math import (sin, cos, tan,** 

 **...    sqrt, log, frexp)** 

```

在这种情况下，`()`字符绝对不是表达式的一部分。`()`字符只是额外的语法，包括使语句与其他语句一致。

## 另请参阅

+   隐式行连接也适用于匹配的`[]`字符和`{}`字符。这些适用于我们将在第四章中查看的集合数据结构，*内置数据结构 - 列表、集合、字典*。

# 包括描述和文档

当我们有一个有用的脚本时，我们经常需要为自己和其他人留下关于它的说明，它是如何解决某个特定问题的，以及何时应该使用它的笔记。

因为清晰很重要，有一些格式化的方法可以帮助使文档非常清晰。这个方法还包含了一个建议的大纲，以便文档会相当完整。

## 准备工作

如果我们使用*编写 Python 脚本和模块文件 - 语法基础*的方法来构建一个脚本文件，我们将在我们的脚本文件中放置一个小的文档字符串。我们将扩展这个文档字符串。

还有其他应该使用文档字符串的地方。我们将在第三章和第六章中查看这些额外的位置，*函数定义*和*类和对象的基础知识*。

我们有两种一般类型的模块，我们将编写摘要文档字符串：

+   **库模块**：这些文件将主要包含函数定义以及类定义。在这种情况下，文档字符串摘要可以侧重于模块是什么，而不是它做什么。文档字符串可以提供使用模块中定义的函数和类的示例。在第三章，*函数定义*，和第六章，*类和对象的基础*，我们将更仔细地研究这个函数包或类包的概念。

+   **脚本**：这些通常是我们期望能够完成一些实际工作的文件。在这种情况下，我们希望关注的是做而不是存在。文档字符串应该描述它的功能以及如何使用它。选项、环境变量和配置文件是这个文档字符串的重要部分。

我们有时会创建包含两者的文件。这需要一些仔细的编辑来在做和存在之间取得适当的平衡。在大多数情况下，我们将简单地提供两种文档。

## 如何做...

编写文档的第一步对于库模块和脚本是相同的：

1.  写一个简要概述脚本或模块是什么或做什么。摘要不要深入介绍它的工作原理。就像报纸文章中的导语一样，它介绍了模块的谁、什么、何时、何地、如何和为什么。详细信息将在文档字符串的正文中提供。

工具如 sphinx 和 pydoc 显示信息的方式暗示了特定的样式提示。在这些工具的输出中，上下文是非常清晰的，因此在摘要句中通常可以省略主语。句子通常以动词开头。

例如，像这样的摘要：*这个脚本下载并解码了当前的特殊海洋警告（SMW）*有一个多余的*这个脚本*。我们可以去掉它，然后以动词短语*下载并解码...*开始。

我们可能会这样开始我们的模块文档字符串：

```py
    ''' 
    Downloads and decodes the current Special Marine Warning (SMW) 
    for the area 'AKQ'. 
    ''' 

```

我们将根据模块的一般重点分开其他步骤。

### 为脚本编写文档字符串

当我们记录脚本时，我们需要关注将使用脚本的人的需求。

1.  像之前展示的那样开始，创建一个摘要句。

1.  勾勒出文档字符串的其余部分的大纲。我们将使用**ReStructuredText**（**RST**）标记。在一行上写出主题，然后在主题下面放一行`=`，使它们成为一个适当的章节标题。记得在每个主题之间留下一个空行。

主题可能包括：

+   **概要**：如何运行这个脚本的摘要。如果脚本使用`argparse`模块来处理命令行参数，那么`argparse`生成的帮助文本就是理想的摘要文本。

+   **描述**：这个脚本的更完整的解释。

+   **选项**：如果使用了`argparse`，这是放置每个参数详细信息的地方。通常我们会重复`argparse`的帮助参数。

+   **环境**：如果使用了`os.environ`，这是描述环境变量及其含义的地方。

+   **文件**：由脚本创建或读取的文件名称是非常重要的信息。

+   **示例**：始终有一些使用脚本的示例会很有帮助。

+   **另请参阅**：任何相关的脚本或背景信息。

其他可能有趣的主题包括**退出状态**，**作者**，**错误**，**报告错误**，**历史**或**版权**。在某些情况下，例如关于报告错误的建议，实际上并不属于模块的文档字符串，而是属于项目的 GitHub 或 SourceForge 页面的其他位置。

1.  在每个主题下填写细节。准确性很重要。由于我们将这些文档嵌入到与代码相同的文件中，因此很容易在模块的其他地方检查内容是否正确和完整。

1.  对于代码示例，我们可以使用一些很酷的 RST 标记。回想一下，所有元素都是由空行分隔的。在一个段落中，只使用`::`。在下一个段落中，将代码示例缩进四个空格。

这是一个脚本的 docstring 示例：

```py
    ''' 
    Downloads and decodes the current Special Marine Warning (SMW) 
    for the area 'AKQ' 

    SYNOPSIS 
    ======== 

    :: 

        python3 akq_weather.py 

    DESCRIPTION 
    =========== 

    Downloads the Special Marine Warnings 

    Files 
    ===== 

    Writes a file, ``AKW.html``. 

    EXAMPLES 
    ======== 

    Here's an example:: 

        slott$ python3 akq_weather.py 
        <h3>There are no products active at this time.</h3> 
    ''' 

```

在概要部分，我们使用`::`作为单独的段落。在示例部分，我们在段落末尾使用`::`。这两个版本都是对 RST 处理工具的提示，表明接下来的缩进部分应该被排版为代码。

### 为库模块编写 docstrings

当我们记录库模块时，我们需要关注的是那些将导入模块以在其代码中使用的程序员的需求。

1.  为 docstring 的其余部分草拟一个大纲。我们将使用 RST 标记。在一行上写出主题。在每个主题下面加一行`=`，使主题成为一个适当的标题。记得在每个段落之间留下一个空行。

1.  如前所示开始，创建一个摘要句子。

+   **描述**：模块包含的内容以及模块的用途摘要。

+   **模块内容**：此模块中定义的类和函数。

+   **示例**：使用模块的示例。

1.  为每个主题填写详细信息。模块内容可能是一个很长的类或函数定义列表。这应该是一个摘要。在每个类或函数内部，我们将有一个单独的 docstring，其中包含该项的详细信息。

1.  有关代码示例，请参阅前面的示例。使用`::`作为段落或段落结束。将代码示例缩进四个空格。

## 工作原理...

几十年来，*man page*的大纲已经发展成为 Linux 命令的有用摘要。这种撰写文档的一般方法被证明是有用和有韧性的。我们可以利用这一大量的经验，并结构化我们的文档以遵循 man page 模型。

这两种描述软件的方法都是基于许多单独页面文档的摘要。目标是利用众所周知的主题集。这使得我们的模块文档与常见做法相一致。

我们希望准备模块 docstrings，这些 docstrings 可以被 Sphinx Python 文档生成器使用（参见[`www.sphinx-doc.org/en/stable/`](http://www.sphinx-doc.org/en/stable/)）。这是用于生成 Python 文档文件的工具。Sphinx 中的`autodoc`扩展将读取我们的模块、类和函数上的 docstring 头，以生成最终的文档，看起来像 Python 生态系统中的其他模块。

## 还有更多...

RST 有一个简单的语法规则，即段落之间用空行分隔。

这条规则使得编写的文档可以被各种 RST 处理工具检查，并重新格式化得非常漂亮。

当我们想要包含一段代码块时，我们将有一些特殊的段落：

+   用空行将代码与文本分开。

+   代码缩进四个空格。

+   提供一个`::`前缀。我们可以将其作为自己单独的段落，或者作为引导段落末尾的特殊双冒号：

```py
        Here's an example:: 

            more_code()  

```

+   `::`用于引导段落。

在软件开发中有创新和艺术的地方。文档并不是推动创新的地方。聪明的算法和复杂的数据结构可能是新颖和聪明的。

### 注意

对于只想使用软件的用户来说，独特的语气或古怪的表达并不有趣。在调试时，幽默的风格也不会有帮助。文档应该是平常和常规的。

编写良好的软件文档可能是具有挑战性的。在太少的信息和仅仅重复代码的文档之间存在着巨大的鸿沟。在某个地方，有一个很好的平衡。重要的是要专注于那些对软件或其工作原理了解不多的人的需求。为这些*半知识*用户提供他们需要描述软件做什么以及如何使用它的信息。

在许多情况下，我们需要解决用例的两个部分：

+   软件的预期用途

+   如何自定义或扩展软件

这可能是两个不同的受众。可能有用户与开发人员不同。每个人都有不同的观点，文档的不同部分需要尊重这两种观点。

## 另请参阅

+   我们将在*在 docstrings 中编写更好的 RST 标记*中查看其他技术。

+   如果我们使用了*编写 python 脚本和模块文件-语法基础*的方法，我们将在我们的脚本文件中放置一个文档字符串。当我们在第三章中构建函数时，*函数定义*，以及在第六章中构建类时，*类和对象的基础*，我们将看到其他可以放置文档字符串的地方。

+   有关 Sphinx 的更多信息，请参阅[`www.sphinx-doc.org/en/stable/`](http://www.sphinx-doc.org/en/stable/)。

+   有关 man 页面大纲的更多背景信息，请参阅[`en.wikipedia.org/wiki/Man_page`](https://en.wikipedia.org/wiki/Man_page)。

# 在 docstrings 中编写更好的 RST 标记

当我们有一个有用的脚本时，通常需要留下关于它的功能、工作原理以及何时使用的注释。许多用于生成文档的工具，包括 Docutils，都使用 RST 标记。我们可以使用哪些 RST 功能来使文档更易读？

## 准备工作

在*包括描述和文档*的方法中，我们看到了将基本的文档放入模块中。这是编写我们的文档的起点。有许多 RST 格式规则。我们将看一些对于创建可读文档很重要的规则。

## 如何做...

1.  一定要写下关键点的大纲。这可能会导致创建 RST 部分标题来组织材料。部分标题是一个两行的段落，标题后面跟着一个下划线，使用`=`，`-`，`^`，`~`或其他 Docutils 字符来划线。

标题将看起来像这样。

```py
        Topic 
        ===== 

```

标题文本在一行上，下划线字符在下一行上。这必须被空行包围。下划线字符可以比标题字符多，但不能少。

RST 工具将推断我们使用下划线字符的模式。只要下划线字符一致使用，匹配下划线字符到期望标题的算法将检测到这种模式。这取决于一致性和对部分和子部分的清晰理解。

刚开始时，可以帮助制作一个明确的提醒便条，如下所示：

| **字符** | **级别** |
| --- | --- |
| = | 1 |
| - | 2 |
| ^ | 3 |
| ~ | 4 |

1.  填写各种段落。用空行分隔段落（包括部分标题）。额外的空行不会有害。省略空行将导致 RST 解析器看到一个单一的长段落，这可能不是我们想要的。

我们可以使用内联标记来强调、加重强调、代码、超链接和内联数学等，还有其他一些东西。如果我们打算使用 Sphinx，那么我们将有一个更大的文本角色集合可以使用。我们将很快看到这些技术。

1.  如果编程编辑器有拼写检查器，请使用。这可能会令人沮丧，因为我们经常会有包含拼写检查失败的缩写的代码示例。

## 工作原理...

docutils 转换程序将检查文档，寻找部分和正文元素。一个部分由一个标题标识。下划线用于将部分组织成正确嵌套的层次结构。推断这一点的算法相对简单，并具有以下规则：

+   如果之前已经看到了下划线字符，则已知级别

+   如果之前没有看到下划线字符，则必须缩进到前一个大纲级别的下一级

+   如果没有上一级，这就是第一级

一个正确嵌套的文档可能具有以下下划线字符序列：

```py
    ==== 
    ----- 
    ^^^^^^ 
    ^^^^^^ 
    ----- 
    ^^^^^^ 
    ~~~~~~~~ 
    ^^^^^^ 

```

我们可以看到，第一个大纲字符`=`将是一级。接下来的`-`是未知的，但出现在一级之后，所以必须是二级。第三个标题有`^`，之前未知，必须是三级。下一个`^`仍然是三级。接下来的两个`-`和`^`分别是二级和三级。

当我们遇到新字符`~`时，它位于三级之下，因此必须是四级标题。

### 注意

从这个概述中，我们可以看到不一致会导致混乱。

如果我们在文档的中途改变主意，这个算法就无法检测到。如果出于莫名其妙的原因，我们决定跳过一个级别并尝试在二级部分内有一个四级标题，那是不可能的。

RST 解析器可以识别几种不同类型的正文元素。我们展示了一些。更完整的列表包括：

+   **文本段落**：这些可能使用内联标记来强调或突出不同种类的内容。

+   **文字块**：这些是用`::`引入并缩进空格的。它们也可以用`.. parsed-literal::`指令引入。一个 doctest 块缩进四个空格，并包括 Python 的`>>>`提示符。

+   **列表、表格和块引用**：我们稍后会看到这些。这些可以包含其他正文元素。

+   **脚注**：这些是可以放在页面底部或章节末尾的特殊段落。这些也可以包含其他正文元素。

+   **超链接目标、替换定义和 RST 注释**：这些是专门的文本项目。

## 还有更多...

为了完整起见，我们在这里指出，RST 段落之间用空行分隔。RST 比这个核心规则要复杂得多。

在*包括描述和文档*配方中，我们看了几种不同类型的正文元素：

+   **文本段落**：这是由空行包围的文本块。在其中，我们可以使用内联标记来强调单词，或者使用字体来显示我们正在引用代码元素。我们将在*使用内联标记*配方中查看内联标记。

+   **列表**：这些是以看起来像数字或项目符号开头的段落。对于项目符号，使用简单的`-`或`*`。也可以使用其他字符，但这些是常见的。我们可能有这样的段落。

有项目符号会有帮助，因为：

+   它们可以帮助澄清

+   它们可以帮助组织

+   **编号列表**：有各种被识别的模式。我们可能会使用这样的东西。

四种常见的编号段落：

+   数字后面跟着像`.`或`)`这样的标点符号。

+   一个字母后面跟着像`.`或`)`这样的标点符号。

+   一个罗马数字后面跟着标点符号。

+   一个特殊情况是使用与前面项目相同的标点符号的`#`。这继续了前面段落的编号。

+   **文字块**：代码示例必须以文字形式呈现。这个文本必须缩进。我们还需要用`::`前缀代码。`::`字符必须是一个单独的段落，或者是代码示例的引导结束。

+   **指令**：指令是一个段落，通常看起来像`.. directive::`。它可能有一些内容，缩进以便包含在指令内。它可能看起来像这样：

```py
        ..  important:: 

            Do not flip the bozo bit. 

```

`.. important::`段落是指令。这之后是一个缩进在指令内的短段落文字。在这种情况下，它创建了一个包含*important*警告的单独段落。

### 使用指令

Docutils 有许多内置指令。Sphinx 添加了许多具有各种功能的指令。

最常用的指令之一是警告指令：*注意*，*小心*，*危险*，*错误*，*提示*，*重要*，*注意*，*提示*，*警告*和通用*警告*。这些是复合主体元素，因为它们可以有多个段落和其中嵌套的指令。

我们可能有这样的东西来提供适当的强调：

```py
    ..  note:: Note Title 

        We need to indent the content of an admonition. 
        This will set the text off from other material. 

```

另一个常见的指令是`parsed-literal`指令。

```py
    ..  parsed-literal:: 

        any text 
            *almost* any format 
        the text is preserved 
            but **inline** markup can be used. 

```

这对于提供代码示例非常方便，其中代码的某些部分被突出显示。这样的文字就是一个简单的主体元素，里面只能有文本。它不能有列表或其他嵌套结构。

### 使用内联标记

在段落中，我们可以使用几种内联标记技术：

+   我们可以用`*`将单词或短语括起来以进行`*强调*`。

+   我们可以用`**`将单词或短语括起来以进行`**强调**`。

+   我们用单个反引号（```py ). Links are followed by a `_` . We might use ``section title`_` to refer to a specific section within a document. We don't generally need to put anything around URL's. The Docutils tools recognize these. Sometimes we want a word or phrase to be shown and the URL concealed. We can use this: ``the Sphinx documentation <http://www.sphinx-doc.org/en/stable/>`_` .
*   We can surround code-related words with double back-tick (````）括起引用，使其看起来像```pycode```。 

还有一种更一般的技术叫做文本角色。角色看起来比简单地用`*`字符包装一个单词或短语要复杂一些。我们使用`:word:`作为角色名称，后面跟着适用的单词或短语在单个```py back-ticks. A text role looks like this `:strong:`this`` .

There are a number of standard role names including `:emphasis:` , `:literal:` , `:code:` , `:math:` , `:pep-reference:` , `:rfc-reference:` , `:strong:` , `:subscript:` , `:superscript:` , and `:title-reference:` . Some of these are also available with simpler markup like `*emphasis*` or `**strong**` . The rest are only available as explicit roles.

Also, we can define new roles with a simple directive. If we want to do very sophisticated processing, we can provide docutils with class definitions for handling roles, allowing us to tweak the way our document is processed. Sphinx adds a large number of roles to support detailed cross references among functions, methods, exceptions, classes, and modules.

## See also

*   For more information on RST syntax, see [`docutils.sourceforge.net`](http://docutils.sourceforge.net) . This includes a description of the docutils tools.
*   For information on **Sphinx Python Documentation Generator** , see [`www.sphinx-doc.org/en/stable/`](http://www.sphinx-doc.org/en/stable/) .
*   The Sphinx tool adds many additional directives and text roles to the basic definitions.

# Designing complex if...elif chains

In most cases, our scripts will involve a number of choices. Sometimes the choices are simple, and we can judge the quality of the design with a glance at the code. In other cases, the choices are more complex, and it's not easy to determine whether or not our if statements are designed properly to handle all of the conditions.

In the simplest case, we have one condition, *C* , and its inverse, *C* . These are the two conditions for an `if...else` statement. One condition, ¬ *C* , is stated in the `if` clause, the other is implied in the `else` .

We'll use *p* ∨ *q* to mean Python's **OR** operator in this explanation. We can call these two conditions *complete* because:

*C* ∨ *C =* ¬ ***T***

We call this complete because no other conditions can exist. There's no third choice. This is the **Law of the Excluded Middle** . It's also the operating principle behind the `else` clause. The `if` statement body is executed or the `else` statement is executed. There's no third choice.

In practical programming, we often have complex choices. We may have a set of conditions, *C* = { *C[1] , C[2] , C[3] , ..., C[n]* }.

We don't want to simply assume that:

*C[1]* ∨ *C[2]* ∨ *C[3]* ∨ *...* ∨ *C[n] = **T***

We can use ![Designing complex if...elif chains](img/Image00003.jpg)  to have a meaning similar to `any(C)` , or perhaps `any([C_1, C_2, C_3, ..., C_n])` . We need to prove that ![Designing complex if...elif chains](img/Image00004.jpg)  ; we can't assume this is `true` .

Here's what might go wrong—we might miss some condition, *C[n+1]* , that got lost in the tangle of logic. Missing this will mean that our program will fail to work properly for this case.

How can we be sure we haven't missed something?

## Getting ready

Let's look at a concrete example of an `if...elif` chain. In the casino game of *Craps* , there are a number of rules that apply to a roll of two dice. These rules apply on the first roll of the game, called the *come out* roll:

*   2, 3, or 12, is *Craps* , which is a loss for all bets placed on the pass line
*   7 or 11 is a winner for all bets placed on the pass line
*   The remaining numbers establish a *point*

Many players place their bets on the pass line. There's also a *don't pass* line, which is less commonly used. We'll use this set of three conditions as an example for looking at this recipe because it has a potentially vague clause in it.

## How to do it...

When we write an `if` statement, even when it appears trivial, we need to be sure that all conditions are covered.

1.  Enumerate the alternatives we know. In our example, we have three rules: (2, 3, 12), (7, 11), and the vague remaining numbers.
2.  Determine the universe of all possible conditions. For this example, there are 10 conditions: the numbers from 2 to 12.
3.  Compare the known alternatives with the universe. There are three possible outcomes of this comparison between the set of conditions, *C* , and the universe of all possible conditions, *U* :

The known alternatives have more conditions than the universe; *C* ⊃ *U* . This is a huge design problem. This requires rethinking the design from the foundations.

There's a gap between the known conditions and the universe of possible conditions; U \ C ≠ ∅. In some cases, it's clear that we haven't covered all of the possible conditions. In other cases, it takes some careful reasoning. We'll need to replace any vague or poorly defined terms with something much more precise.

In this example, we have a vague term, which we can replace with something more specific. The term **remaining numbers** appears to be the list of values (4, 5, 6, 8, 9, 10). Supplying this list removes any possible gaps and doubts.

The known alternatives match the universe of possible alternatives; *U* ≡ *C* . There are two common cases:

*   We have something as simple as *C* ∨ ¬ *C* . We can use a single `if` and `else` clause—we do not need to use this recipe because we can easily deduce ¬ *C* .
*   We might have something more complex. Since we know the entire universe, we can show that ![How to do it...](img/Image00004.jpg)  . We need to use this recipe to write a chain of `if` and `elif` statements, one clause per condition.

The distinction is not always crisp. In our example, we don't have a detailed specification for one of the conditions, but the condition is *mostly* clear. If we think the missing condition is obvious, we can use an `else` clause instead of writing it out explicitly. If we think the missing condition might be misunderstood, we should treat it as vague and use this recipe.

1.  Write the `if...elif...elif` chain that covers all of the known conditions. For our example, it will look like this:

    ```中

骰子= die_1 + die_2

如果骰子在（2,3,12）中：

game.craps()

否则如果骰子在（7,11）中：

游戏.获胜者（）

否则如果骰子在（4,5,6,8,9,10）中：

游戏.得分（骰子）

```py

2.  Add an `else` clause that raises an exception, like this:

```

否则：

引发异常（'设计问题：未考虑所有条件'）

```py

This extra `else` crash condition gives us a way to positively identify when a logic problem is found. We can be sure that any error we make will lead to a conspicuous problem.

## How it works...

Our goal is to be sure that our program always works. While testing helps, we can still have wrong assumptions in both design and test cases.

While rigorous logic is essential, we can still make errors. Further, someone else could try to tweak our code and introduce an error. More embarrassingly, we could make a change in our own code that leads to breakage.

The `else` crash option forces us to be explicit for each and every condition. Nothing is assumed. As we noted previously, any error in our logic will be uncovered when an exception gets raised.

The `else` crash option doesn't have a significant performance impact. A simple `else` clause is slightly faster than an `elif` clause with a condition. If we think that our application performance depends in any way on the cost of a single expression, we've got more serious design problems to solve. The cost of evaluating a single expression is rarely the costliest part of an algorithm.

Crashing with an exception is a sensible behavior in the presence of a design problem. It doesn't make much sense to follow the design pattern of writing a warning message to a log. If we have this kind of logic gap, the program is fatally broken and it's important to find and fix this as soon as it's known.

## There's more...

In many cases, we can derive an `if...elif...elif` chain from an examination of the desired post-condition at some point in the program's processing. For example, we may need a statement that establishes something simple like *m* the larger of *a* or *b* .

(For the sake of working through the logic, we'll avoid `m = max(a, b)` .)

We can formalize the final condition like this:

*(m = a* ∨ *m = b)* ∧ *m > a * ∧ *m > b*

We can work backwards from this final condition, by writing the goal as an assert statement:

```

# 做一些事情

断言（m = a 或 m = b）和 m> a 和 m> b

```py

Once we have the goal stated, we can identify statements that lead to that goal. Clearly assignment statements like `m = a` and `m = b` will be appropriate, but only under certain conditions.

Each of these statements is part of the solution, and we can derive a precondition that shows when the statement should be used. The preconditions for each assignment statement are the `if` and `elif` expressions. We need to use `m = a` when `a >= b` ; we need to use `m=b` when `b >= a` . Rearranging logic into code gives us this:

```

如果 a> = b：

m = a

如果 b> = a：

m = b

否则：引发异常（'设计问题'）

断言（m = a 或 m = b）和 m> a 和 m> b

```py

Note that our universe of conditions,   *U* = { *a ≥ b, b ≥ a* }, is complete; there's no other possible relationship. Also notice that in the edge case of *a = b* , we don't actually care which assignment statement we use. Python will process the decisions in order, and will execute `m = a` . The fact that this choice is consistent shouldn't have any impact on our design of `if...elif...elif` chains. We should always write the conditions without regard to order of evaluation of the clauses.

## See also

*   This is similar to the syntactic problem of a **dangling else** . See [`en.wikipedia.org/wiki/Dangling_else`](https://en.wikipedia.org/wiki/Dangling_else) .
*   Python's indentation removes the dangling else syntax problem. It doesn't remove the semantic issue of trying to be sure that all conditions are properly accounted for in a complex `if...elif...elif` chain.
*   Also, see [`en.wikipedia.org/wiki/Predicate_transformer_semantics`](https://en.wikipedia.org/wiki/Predicate_transformer_semantics) .

# Designing a while statement which terminates properly

Much of the time, the Python `for` statement provides all of the iteration controls we need. In many cases, we can use built-in functions like `map()` , `filter()` , and `reduce()` to process collections of data.

There are a few situations, however, where we need to use a `while` statement. Some of those situations involve data structures where we can't create a proper iterator to step through the items. Other items involve interactions with human users, where we don't have the data until we get input from the person.

## Getting ready

Let's say that we're going to be prompting a user for their password. We'll use the `getpass` module so that there's no echo.

Further, to be sure they've entered it properly, we'll want to prompt them twice and compare the results. This is a situation where a simple `for` statement isn't going to work out well. It can be pressed into service, but the resulting code looks strange: `for` statements have an explicit upper bound; prompting a user for input doesn't really have an upper bound.

## How to do it...

We'll look at a six-step process that outlines the core of designing this kind of iterative algorithm. This is the kind of thing we need to do when a simple `for` statement doesn't solve our problem.

1.  Define done. In our case, we'll have two copies of the password, `password_text` and `confirming_password_text` . The condition which must be `true` after the loop is that `password_text == confirming_password_text` . Ideally, reading from people (or files) is a bounded activity. Eventually, people will enter the matching pair of values. Until they enter the matching pair, we'll iterate indefinitely.

There are other boundary conditions. For example, end of file. Or we allow the person to go back to a previous prompt. Generally, we handle these other conditions with exceptions in Python.

Of course, we can always add these additional conditions to our definition of done. We may need to have a complex terminating condition like end of file OR `password_text == confirming_password_text` .

In this example, we'll opt for exception handling and assume that a `try:` block will be used. It greatly simplifies the design to have only a single clause in the terminating condition.

We can rough out the loop like this:

```

# 初始化一些东西

而#未终止：

# 做一些事情

断言密码文本==确认密码文本

```py

We've written our definition of done as a final `assert` statement. We've included comments for the rest of the iteration that we'll fill in in subsequent steps.

2.  Define a condition that's `true` while the loop is iterating. This is called an **invariant** because it's always `true` at the start and end of loop processing. It's often created by generalizing the post-condition or introducing another variable.

When reading from people (or files) we have an implied state change that is an important part of the invariant. We can call this the *get the next input* change in state. We often have to articulate clearly that our loop will be acquiring some next value from an input stream.

We have to be sure that our loop properly gets the next item in spite of any complex logic in the body of the `while` statement. It's a common bug to have a condition where a next input is not actually fetched. This leads to programs which *hang* —there's no state change in one logic path through the `if` statements in the body of the `while` statement. The invariant wasn't reset properly, or it wasn't articulated properly when designing the loop.

In our case, the invariant will use a conceptual `new-input()` condition. This condition is `true` when we've read a new value using the `getpass()` function. Here's our expanded loop design:

```

# 初始化一些东西

# 断言不变的新输入（密码文本）

# 和新输入（确认密码文本）

而#未终止：

# 做一些事情

# 断言不变的新输入（密码文本）

# 和新输入（确认密码文本）

断言密码文本==确认密码文本

```py

3.  Define the condition for leaving the loop. We need to be sure that this condition depends on the invariant being `true` . We also need to be sure that, when this termination condition is finally `false,` the target state will become `true` .

In most cases, the loop condition is the logical negation of the target state. Here's the expanded design:

```

# 初始化一些东西

# 断言不变的新输入（密码文本）

# 和新输入（确认密码文本）

而密码文本！=确认密码文本：

# 做一些事情

# 断言不变的新输入（密码文本）

# 和新输入（确认密码文本）

断言密码文本==确认密码文本

```py

4.  Define the initialization that will make sure that both the invariant will be `true` and that we can actually test the terminating condition. In this case, we need to get values for the two variables. The loop now looks like this:

```

password_text= getpass()

确认密码文本= getpass（“确认：”）

# 断言新输入（密码文本）

# 和新输入（确认密码文本）

而密码文本！=确认密码文本：

# 做一些事情

# 断言新输入（密码文本）

# 和新输入（确认密码文本）

断言密码文本==确认密码文本

```py

5.  Write the body of the loop which will reset the invariant to `true` . We need to write the fewest statements that will do this. For this example loop, the fewest statements are pretty obvious—they match the initialization. Our updated loop looks like this:

```

password_text= getpass()

确认密码文本= getpass（“确认：”）

# 断言新输入（密码文本）

# 和新输入（确认密码文本）

而密码文本！=确认密码文本：

password_text= getpass()

确认密码文本= getpass（“确认：”）

# 断言新输入（密码文本）

# 和新输入（确认密码文本）

断言密码文本==确认密码文本

```py

6.  Identify a clock—a monotonically decreasing function that shows that each iteration of the loop really does make progress toward the terminating condition.

    When gathering input from people, we're forced to make an assumption that—eventually—they'll enter a matching pair. Each trip through the loop brings us one step closer to that matching pair. To be properly formal, we can assume that there will be *n* inputs before they match; we have to show that each trip through the loop decreases the number which remain.

    In complex situations, we may need to treat the user's input as a list of values. For our example, we'd think of the user input as a sequence of pairs: *[(p[1] , q[1] ),(p[2] , q[2] ),(p[3] , q[3] ),...,(p[n] , q[n] )]* . With a finite list, we can more easily reason about whether or not our loop really is making progress towards completion.

Because we built the loop based on the target `final` condition, we can be absolutely sure that it does what we want it to do. If our logic is sound, the loop will terminate, and will terminate with the expected results. This is the goal of all programming—to have the machine reach a desired state given some initial state.

Removing some comments, we have this as our final loop:

```

password_text= getpass()

确认密码文本= getpass（“确认：”）

而密码文本！=确认密码文本：

password_text= getpass()

确认密码文本= getpass（“确认：”）

断言密码文本==确认密码文本

```py

We left the final post-condition in place as an `assert` statement. For complex loops it's both a built-in test, as well as a comment that explains how the loop works.

This design process often produces a loop that looks similar to one we might develop based on intuition. There's nothing wrong with having a step by step justification for an intuitive design. Once we've done this a few times, we can be much more confident in using a loop knowing that we can justify the design.

In this case, the loop body and the initialization happen to be the same code. If this is a problem, we can define a tiny two-line function to avoid repeating the code. We'll look at this in Chapter 3 , *Function Definitions* .

## How it works...

We start out by articulating the goal for the loop. Everything else that we do will assure that the code written leads to that goal condition. Indeed, this is the motivation behind all software design—we're always trying to write the fewest statements that lead to a given goal state. We're often working *backwards* from goal to initialization. Each step in the chain of reasoning is essentially stating the weakest precondition for some statement, `S` , that leads to our desired outcome condition.

Given a post-condition, we're trying to solve for a statement and a precondition. We're always building this pattern:

```

断言前置条件

S

断言后置条件

```py

The post-condition is our definition of done. We need to hypothesize a statement, `S` , that leads to done, and a precondition for that statement. There are always an infinite number of alternative statements; we focus on the weakest precondition—the one that has the fewest assumptions.

At some point—usually when writing the initialization statements—we find that the pre-condition is merely `true` : any initial state will do as the precondition for a statement. That's how we know that our program can start from any initial state and complete as expected. This is ideal.

When designing a `while` statement, we have a nested context inside the statement's body. The body should always be in a process of resetting the invariant condition to be `true` again. In our example, this means reading more input from the user. In other examples, we might be processing another character in a string, or another number from a set of numbers.

We need to prove that when the invariant is `true` and the loop condition is `false` then our final goal is achieved. This proof is easier when we start from the final goal and create the invariant and the loop condition based on that final goal.

What's important is patiently doing each step so that our reasoning is solid. We need to be able to prove that the loop will work. Then we can run unit tests with confidence.

## See also

*   We look at some other aspects of advanced loop design in the *Avoiding a potential problem with break statements* recipe.
*   We also looked at this concept in the *Designing complex if...elif chains* recipe.
*   A classic article on this topic is by David Gries, *A note on a standard strategy for developing loop invariants and loops* . See [`www.sciencedirect.com/science/article/pii/0167642383900151`](http://www.sciencedirect.com/science/article/pii/0167642383900151) .
*   Algorithm design is a big subject. A good introduction is by Skiena, *Algorithm Design Manual* . See [`www3.cs.stonybrook.edu/~algorith/`](http://www3.cs.stonybrook.edu/~algorith/) .

# Avoiding a potential problem with break statements

The common way to understand a `for` statement is that it creates a *for all* condition. At the end of the statement, we can assert that, for all items in a collection, some processing has been done.

This isn't the only meaning for a `for` statement. When we introduce the `break` statement inside the body of a `for` , we change the semantics to *there exists* . When the `break` statement leaves the `for` (or `while` ) statement, we can assert only that there exists at least one item that caused the statement to end.

There's a side issue here. What if the loop ends without executing the `break` ? We are forced to assert that there does not exist even one item that triggered the `break` . **DeMorgan's Law** tells us that a not exists condition can be restated as a *for all* condition: ¬∃ [*x*] *B* ( *x* ) ≡ ∀ [*x*] ¬ *B* ( *x* ). In this formula, *B(x)* is the condition on the `if` statement that includes the `break` . If we never found *B(x)* , then for all items ¬ *B(x)* was `true` . This shows some of the symmetry between a typical *for all* loop and a *there exists* loop which includes a `break` .

The condition that's `true` upon leaving a `for` or `while` statement can be ambiguous. Did it end normally? Did it execute the `break` ? We can't *easily* tell, so we'll provide a recipe that gives us some design guidance.

This can become an even bigger problem when we have multiple `break` statements, each with its own condition. How can we minimize the problems created by having complex `break` conditions?

## Getting ready

Let's find the first occurrence of a `:` or `=` in a string. This is a good example of a *there exists* modification to a `for` statement. We don't want to process all characters, we want to know where there exists the left-most `:` or `=` .

```

**>>> sample_1 = "some_name = the_value"

>>>对于范围内的位置

...如果 sample_1[position]在'：='中：

...中断

>>>打印（'名称=', sample_1[:position]，

... 'value=', sample_1[position+1:]）

名称= some_name  value=  the_value**

```py

What about this edge case?

```

**>>> sample_2 = "name_only"

>>>对于范围内的位置

...如果 sample_2[position]在'：='中：

...中断

>>>打印（'名称=', sample_2[:position]，

... 'value=', sample_2[position+1:]）

名称= name_onl value=**

```py

That's awkwardly wrong. What happened?

## How to do it...

As we noted in the *Designing a while statement which terminates properly* recipe, every statement establishes a post-condition. When designing a loop, we need to articulate that condition. In this case, we didn't properly articulate the post-condition.

Ideally, the post-condition would be something simple like `text[position] in '=:'` . However, if there's no `=` or `:` in the given text, the simple post-condition doesn't make logical sense. When no character exists which matches the criteria, we can't make any assertion about the position of a character that's not there.

1.  Write the obvious post-condition. We sometimes call this the *happy-path* condition because it's the one that's `true` when nothing unusual has happened.

```

文本[位置]在'：='中

```py

2.  Add post-conditions for the edge cases. In this example, we have two additional conditions:

    *   There's no `=` or `:` .
    *   There are no characters at all. The `len()` is zero, and the loop never actually does anything. In this case, the position variable will never be created.

```

（len（text）== 0

或不是（'='在文本中或'：'在文本中）

或文本[位置]在'：='中）

```py

3.  If a `while` statement is being used, consider redesigning it to have completion conditions. This can eliminate the need for a `break` statement.
4.  If a `for` statement is being used, be sure a proper initialization is done, and add the various terminating conditions to the statements after the loop. It can look redundant to have `x = 0` followed by `for x = ...` . It's necessary in the case of a loop which doesn't execute the `break` statement, though.

```

**>>>位置= -1#如果长度为零

>>>对于范围内的位置

... 如果 sample_2[position]在'：='中：

... 休息

...

>>> 如果位置== -1：

... 打印（“名称=”，无，“值=”，无）

... 否则不是（text[position] == ':'或 text[position] == '='）：

... 打印（“名称=”，sample_2，“值=”，无）

... 其他：

... 打印（'name ='，sample_2[:position]，

... 'value ='，sample_2[position+1:]）

名称=仅名称值=无**

```py

In the statements after the `for` , we've enumerated all of the terminating conditions explicitly. The final output, `name= name_only value= None` , confirms that we've correctly processed the sample text.

## How it works...

This approach forces us to work out the post-condition carefully so that we can be absolutely sure that we know all the reasons for the loop terminating.

In more complex loops—with multiple `break` statements—the post-condition can be difficult to work out fully. The post-condition for a loop must include all of the reasons for leaving the loop—the *normal* reasons plus all of the `break` conditions.

In many cases, we can refactor the loop to push the processing into the body of the loop. Rather than simply assert that `position` is the index of the `=` or `:` character, we include the next processing steps of assigning the `name` and `value` values. We might have something like this:

```

如果 len（sample_2）> 0：

名称，值= sample_2，无

其他：

名称，值=无，无

对于 position in range（len（sample_2））：

如果 sample_2[position]在'：='中：

名称，值= sample_2[:position]，sample2[position:]

打印（'name ='，name，'value ='，value）

```py

This version pushes some of the processing forward, based on the complete set of post-conditions evaluated previously. This kind of refactoring is common.

The idea is to forego any assumptions or intuition. With a little bit of discipline, we can be sure of the post-conditions from any statement.

Indeed, the more we think about post-conditions, the more precise our software can be. It's imperative to be explicit about the goal for our software and work backwards from the goal by choosing the simplest statements that will make the goal become `true` .

## There's more...

We can also use an `else` clause on a `for` statement to determine if the loop finished normally or a `break` statement was executed. We can use something like this:

```

对于 position in range（len（sample_2））：

如果 sample_2[position]在'：='中：

名称，值= sample_2[:position]，sample_2[position+1:]

休息

其他：

如果 len（sample_2）> 0：

名称，值= sample_2，无

其他：

名称，值=无，无

```py

The `else` condition is sometimes confusing, and we don't recommend it. It's not clear that it is substantially better than any of the alternatives. It's too easy to forget the reason why the `else` is executed because it's used so rarely.

## See also

*   A classic article on this topic is by David Gries, *A note on a standard strategy for developing loop invariants and loops* . See [`www.sciencedirect.com/science/article/pii/0167642383900151`](http://www.sciencedirect.com/science/article/pii/0167642383900151) .

# Leveraging the exception matching rules

The `try` statement lets us capture an exception. When an exception is raised, we have a number of choices for handling it:

*   **Ignore it** : If we do nothing, the program stops. We can do this in two ways—don't use a `try` statement in the first place, or don't have a matching `except` clause in the `try` statement.
*   **Log it** : We can write a message and let it propagate; generally this will stop the program.
*   **Recover from it** : We can write an `except` clause to do some recovery action to undo the effects of something that was only partially completed in the `try` clause. We can take this a step further and wrap the `try` statement in a `while` statement and keep retrying until it succeeds.
*   **Silence it** : If we do nothing (that is, `pass` ) then processing is resumed after the `try` statement. This silences the exception.
*   **Rewrite it** : We can raise a different exception. The original exception becomes a context for the newly-raised exception.
*   **Chain it** : We chain a different exception to the original exception. We'll look at this in the *Chaining exceptions with the raise from statement* recipe.

What about nested contexts? In this case, an exception could be ignored by an inner `try` but handled by an outer context. The basic set of options for each `try` context are the same. The overall behavior of the software depends on the nested definitions.

Our design of a `try` statement depends on the way that Python exceptions form a class hierarchy. For details, see *Section 5.4* , *Python Standard Library* . For example, `ZeroDivisionError` is also an `ArithmeticError` and an `Exception` . For another example, a `FileNotFoundError` is also an `OSError` as well as an `Exception` .

This hierarchy can lead to confusion if we're trying to handle detailed exceptions as well as generic exceptions.

## Getting ready

Let's say we're going to make simple use of the `shutil` to copy a file from one place to another. Most of the exceptions that might be raised indicate a problem too serious to work around. However, in the rare event of a `FileExistsError` , we'd like to attempt a recovery action.

Here's a rough outline of what we'd like to do:

```

从路径导入路径

导入 shutil

进口

source_path = Path（os.path.expanduser（

'〜/Documents/Writing/Python Cookbook/source'））

target_path = Path（os.path.expanduser（

'〜/Dropbox/B05442/demo/''）

对于 source_file_path in source_path.glob（'* / * .rst'）：

source_file_detail = source_file_path.relative_to（source_path）

target_file_path = target_path / source_file_detail

shutil.copy（str（source_file_path），str（target_file_path

```py

We have two paths, `source_path` and `target_path` . We've located all of the directories under the `source_path` that have `*.rst` files.

The expression `source_file_path.relative_to(source_path)` gives us the tail end of the file name, the portion after the base directory. We use this to build a new path under the `target` directory.

While we can use `pathlib.Path` objects for a lot of ordinary path processing, in Python 3.5 modules like `shutil` expect string filenames instead of `Path` objects; we need to explicitly convert the `Path` objects. We can only hope that Python 3.6 changes this.

The problems arise with handling exceptions raised by the `shutil.copy()` function. We need a `try` statement so that we can recover from certain kinds of errors. We'll see this kind of error if we try to run this:

```

FileNotFoundError：[Errno 2]

没有这样的文件或目录：

'/Users/slott/Dropbox/B05442/demo/ch_01_numbers_strings_and_tuples/index.rst'

```py

How do we create a `try` statement that handles the exceptions in the proper order?

## How to do it...

1.  Write the code we want to use indented in the `try` block:

```

尝试：

shutil.copy（str（source_file_path），str（target_file_path））

```py

2.  Include the most specific exception classes first. In this case, we have separate responses for the specific `FileNotFoundError` and the more general `OSError` .

```

尝试：

shutil.copy（str（source_file_path），str（target_file_path））

除了 FileNotFoundError：

os.makedir（target_file_path.parent）

shutil.copy（str（source_file_path），str（target_file_path））

```py

3.  Include any more general exceptions later:

```

尝试：

shutil.copy（str（source_file_path），str（target_file_path））

除了 FileNotFoundError：

os.makedirs（str（target_file_path.parent））

shutil.copy（str（source_file_path），str（target_file_path））

除了 OSError as ex：

打印（ex）

```py

    We've matched exceptions with the most specific first and the more generic after that.

    We handled the `FileNotFoundError` by creating the missing directories. Then we did the `copy()` again, knowing it would now work properly.

    We silenced any other exceptions of the class `OSError` . For example, if there's a permission problem, that error will simply be logged. Our objective is to try and copy all of the files. Any files that cause problems will be logged, but the copying process will continue.

## How it works...

Python's matching rules for exceptions are intended to be simple:

*   Process the `except` clauses in order
*   Match the actual exception against the exception class (or tuple of exception classes). A match means that the actual exception object (or any of the base classes of the exception object) is of the given class in the `except` clause.

These rules show why we put the most specific exception classes first and the more general exception classes last. A generic exception class like the `Exception` will match almost every kind of exception. We don't want this first, because no other clauses will be checked. We must always put generic exceptions last.

There's an even more generic class, the `BaseException` class. There's no good reason to ever handle exceptions of this class. If we do, we will be catching `SystemExit` and `KeyboardInterrupt` exceptions, which interferes with the ability to kill a misbehaving application. We only use the `BaseException` class as a superclass when defining new exception classes that exist outside the normal exception hierarchy.

## There's more...

Our example includes a nested context in which a second exception can be raised. Consider this `except` clause:

```

除了 FileNotFoundError：

os.makedirs（str（target_file_path.parent））

shutil.copy（str（source_file_path），str（target_file_path））

```py

If the `os.makedirs()` or `shutil.copy()` functions raise another exception, it won't be handled by this `try` statement. Any exceptions raised here will crash the program as a whole. We have two ways to handle this, both of which involve nested `try` statements.

We can rewrite this to include a nested `try` during recovery:

```

尝试：

shutil.copy（str（source_file_path），str（target_file_path））

除了 FileNotFoundError：

尝试：

os.makedirs（str（target_file_path.parent））

shutil.copy（str（source_file_path），str（target_file_path））

除了 OSError as ex：

打印（ex）

除了 OSError as ex：

打印（ex）

```py

In this example, we've repeated the `OSError` processing in two places. In our nested context, we'll log the exception and let it propagate, which will likely stop the program. In the outer context, we'll do the same thing.

We say *likely stop the program* because this code could be used inside a `try` statement, which might handle these exceptions. If there's no other `try` context, then these unhandled exceptions will stop the program.

We can also rewrite our overall statement to have nested `try` statements that separate the two exception handling strategies into more local and more global considerations. It would look like this:

```

尝试：

尝试：

shutil.copy（str（source_file_path），str（target_file_path））

除了 FileNotFoundError：

os.makedirs（str（target_file_path.parent））

shutil.copy（str（source_file_path），str（target_file_path））

除了 OSError as ex：

打印（ex）

```py

The copy with `makedirs` processing in the inner `try` statement handles only the `FileNotFoundError` exception. Any other exception will propagate out to the outer `try` statement. In this example, we've nested the exception handling so that the generic processing wraps the specific processing.

## See also

*   In the *Avoiding a potential problem with an except: clause* recipe we look at some additional considerations when designing exceptions
*   In the *Chaining exceptions with the raise from statement* recipe we look at how we can chain exceptions so that a single class of exception wraps different detailed exceptions

# Avoiding a potential problem with an except: clause

There are some common mistakes in exception handling. These can cause programs to become unresponsive.

One of the mistakes we can make is to use the `except:` clause. There are a few other mistakes which we can make if we're not cautious about the exceptions we try to handle.

This recipe will show some common exception handling errors that we can avoid.

## Getting ready

In the *Avoiding a potential problem with an except: clause* recipe we looked at some considerations when designing exception handling. In that recipe, we discouraged the use of `BaseException` because we can interfere with stopping a misbehaving Python program.

We'll extend the idea of *what not to do* in this recipe.

## How to do it...

Use `except Exception:` as the most general kind of exception managing.

Handling too many exceptions can interfere with our ability to stop a misbehaving Python program. When we hit *Ctrl* + *C* , or send a `SIGINT` signal via `kill -2` , we generally want the program to stop. We rarely want the program to write a message and keep running, or stop responding altogether.

There are a few other classes of exceptions which we should be wary of attempting to handle:

*   SystemError
*   RuntimeError
*   MemoryError

Generally, these exceptions mean that things are going badly somewhere in Python's internals. Rather than silence these exceptions, or attempt some recovery, we should allow the program to fail, find the root cause, and fix it.

## How it works...

There are two techniques we should avoid:

*   Don't capture the `BaseException` class
*   Don't use `except:` with no exception class. This matches all exceptions; this will include exceptions we should avoid trying to handle.

Using `except BaseException` or except without a specific class can cause a program to become unresponsive at exactly the time we need to stop it.

Further, if we capture any of these exceptions, we can interfere with the way these internal exceptions are handled:

*   `SystemExit`
*   `KeyboardInterrupt`
*   `GeneratorExit`

If we silence, wrap, or rewrite any of these, we may have created a problem where none existed. We may have exacerbated a simple problem into a larger and more mysterious problem.

### Note

It's a noble aspiration to write a program which never crashes. Interfering with some of Python's internal exceptions doesn't create a more reliable program. Instead, it creates a program where a clear failure is masked and made into an obscure mystery.

## See also

*   In the *Leveraging the exception matching rules* recipe we look at some considerations when designing exceptions
*   In the *Chaining exceptions with the raise from statement* recipe we look at how we can chain exceptions so that a single class of exception wraps different detailed exceptions.

# Chaining exceptions with the raise from statement

In some cases, we may want to merge some seemingly unrelated exceptions into a single generic exception. It's common for a complex module to define a single generic `Error` exception which applies to many situations that can arise within the module.

Most of the time, the generic exception is all that's required. If the module's `Error` is raised, something didn't work.

Less frequently, we want the details for debugging or monitoring purposes. We might want to write them to a log, or include the details in an e-mail. In this case, we need to provide supporting details that amplify or extend the generic exception. We can do this by chaining from the generic exception to the root cause exception.

## Getting ready

Assume we're writing some complex string processing. We'd like to treat a number of different kinds of detailed exceptions as a single generic error so that users of our software are insulated from the implementation details. We can attach details to the generic error.

## How to do it...

1.  To create a new exception, we can do this:

```

类错误（异常）：

通过

```py

That's sufficient to define a new class of exception.

2.  When handling exceptions, we can chain them using the `raise from` statement like this:

```

尝试：

某事

除了（IndexError，NameError）作为异常：

打印（“预期”，异常）

引发错误（“出了些问题”）来自异常

除了异常作为异常：

打印（“意外”，异常）

提高

```py

    In the first `except` clause, we matched two kinds of exception classes. No matter which kind we get, we'll raise a new exception from the module's generic `Error` exception class. The new exception will be chained to the root cause exception.

    In the second `except` clause, we matched the generic `Exception` class. We wrote a log message and re-raised the exception. Here, we're not chaining, but simply continuing exception handling in another context.

## How it works...

The Python exception classes all have a place to record the cause of the exception. We can set this `__cause__` attribute using the `raise Exception from Exception` statement.

Here's how it looks when this exception is raised:

```

**>>> 类错误（异常）：

...     pass

>>> 尝试：

... 'hello world'[99]

... 除了（IndexError，NameError）作为异常：

... 引发错误（“索引问题”）来自异常

...

最近一次的跟踪（最近的调用）：

文件“<doctest default[0]>”，第 2 行，在<module>

'hello world'[99]

IndexError：字符串索引超出范围**

```py

The exception that we just saw was the direct cause of the following exception:

```

**最近一次的跟踪（最近的调用）：

文件“/Library/Frameworks/Python.framework/Versions/3.4/lib/python3.4/doctest.py”，第 1318 行，在 __run

compileflags，1），test.globs）

文件“<doctest default[0]>”，第 4 行，在<module>

引发错误（“索引问题”）来自异常

错误：索引问题**

```py

This shows a chained exception. The first exception in the `Traceback` message is an `IndexError` exception. This is the direct cause. The second exception in the `Traceback` is our generic `Error` exception. This is a generic summary exception, which was chained to the original cause.

An application will see the `Error` exception in a `try:` statement. We might have something like this:

```

尝试：

some_function（）

除了错误作为异常：

打印（异常）

打印（exception .__cause__）

```py

Here we've shown a function named `some_function()` that can raise the generic `Error` exception. If this function does raise the exception, the `except` clause will match the generic `Error` exception. We can print the exception's message, `exception` , as well as the root cause exception, `exception.__cause__` . In many applications, the `exception.__cause__` value may get written to a debugging log rather than be displayed to users.

## There's more...

If an exception is raised inside an exception handler, this also creates a kind of chained exception relationship. This is the *context* relationship rather than the *cause* relationship.

The context message looks similar. The message is slightly different. It says `During handling of the above exception, another exception occurred:` . The first `Traceback` will show the original exception. The second message is the exception raised without using an explicit from connection.

Generally, the context is something unplanned that indicates an error in the `except` processing block. For example, we might have this:

```

尝试：

某事

除了 ValueError 作为异常：

打印（“一些消息”，exceotuib）

```py

This will raise a `NameError` exception with a context of a `ValueError` exception. The `NameError` exception stems from misspelling the exception variable as `exceotuib` .

## See also

*   In the *Leveraging the exception matching rules* recipe we look at some considerations when designing exceptions
*   In the *Avoiding a potential problem with an except: clause* recipe we look at some additional considerations when designing exceptions

# Managing a context using the with statement

There are many instances where our scripts will be entangled with external resources. The most common examples are disk files and network connections to external hosts. A common bug is retaining these entanglements forever, tying up these resources uselessly. These are sometimes called memory **leaks** because the available memory is reduced each time a new file is opened without closing a previously used file.

We'd like to isolate each entanglement so that we can be sure that the resource is acquired and released properly. The idea is to create a context in which our script uses an external resource. At the end of the context, our program is no longer bound to the resource and we want to be guaranteed that the resource is released.

## Getting ready

Let's say we want to write lines of data to a file in CSV format. When we're done, we want to be sure that the file is closed and the various OS resources—including buffers and file handles—are released. We can do this in a context manager, which guarantees that the file will be properly closed.

Since we'll be working with CSV files, we can use the `csv` module to handle the details of the formatting:

```

**>>> 导入 csv**

```py

We'll also use the `pathlib` module to locate the files we'll be working with:

```

**>>> import pathlib**

```py

For the purposes of having something to write, we'll use this silly data source:

```

**>>> some_source = [[2,3,5]，[7,11,13]，[17,19,23]]**

```py

This will give us a context in which to learn about the `with` statement.

## How to do it...

1.  Create the context by opening the file, or creating the network connection with `urllib.request.urlopen()` . Other common contexts include archives like `zip` files and `tar` files:

```

target_path = pathlib.Path（'code/test.csv'）

与 target_path.open（'w'，newline =''）作为 target_file：

```py

2.  Include all the processing, indented within the `with` statement:

```

target_path = pathlib.Path（'code/test.csv'）

与 target_path.open（'w'，newline =''）作为 target_file：

写入器= csv.writer（target_file）

writer.writerow（['column'，'data'，'headings']）

对于数据中的一些源：

writer.writerow（data）

```py

3.  When we use a file as a context manager, the file is automatically closed at the end of the indented context block. Even if an exception is raised, the file is still closed properly. Outdent the processing that is done after the context is finished and the resources are released:

```

target_path = pathlib.Path（'code/test.csv'）

with target_path.open('w', newline='') as target_file:

写入器=csv.writer(target_file)

写入器.writerow(['列', '标题'])

对于一些来源的数据：

写入器.writerow(data)

打印'完成写入'，目标路径

```py

The statements outside the `with` context will be executed after the context is closed. The named resource—the file opened by `target_path.open()` —will be properly closed.

Even if an exception is raised inside the `with` statement, the file is still properly closed. The context manager is notified of the exception. It can close the file and allow the exception to propagate.

## How it works...

A context manager is notified of two kinds of exits from the block of code:

*   Normal exit with no exception
*   An exception was raised

The context manager will—under all conditions—disentangle our program from external resources. Files can be closed. Network connections can be dropped. Database transactions can be committed or rolled back. Locks can be released.

We can experiment with this by including a manual exception inside the `with` statement. This can show that the file was properly closed.

```

尝试：

目标路径=pathlib.Path('code/test.csv')

with target_path.open('w', newline='') as target_file:

写入器=csv.writer(target_file)

写入器.writerow(['列', '标题'])

对于一些来源的数据：

写入器.writerow(data)

引发异常("只是测试")

除了异常 as exc:

打印目标文件是否关闭

打印异常

打印'完成写入'，目标路径

```py

In this example, we've wrapped the real work in a `try` statement. This allows us to raise an exception after writing the first to the CSV file. When the exception is raised, we can print the exception. At this point, the file will also be closed. The output is simply this:

```

真

只是测试

完成写入代码/测试.csv

```

这向我们表明文件已经正确关闭。它还向我们显示了与异常相关的消息，以确认它是我们手动引发的异常。输出的`test.csv`文件将只包含`some_source`变量的第一行数据。

## 还有更多...

Python 为我们提供了许多上下文管理器。我们注意到，打开的文件是一个上下文，`urllib.request.urlopen()`创建的打开网络连接也是一个上下文。

对于所有文件操作和所有网络连接，我们应该使用`with`语句作为上下文管理器。很难找到这个规则的例外。

事实证明，`decimal`模块使用上下文管理器来允许对十进制算术执行的方式进行本地化更改。我们可以使用`decimal.localcontext()`函数作为上下文管理器，以更改由`with`语句隔离的计算的舍入规则或精度。

我们也可以定义自己的上下文管理器。`contextlib`模块包含函数和装饰器，可以帮助我们在不明确提供上下文管理器的资源周围创建上下文管理器。

在处理锁时，`with`上下文是获取和释放锁的理想方式。请参阅[`docs.python.org/3/library/threading.html#with-locks`](https://docs.python.org/3/library/threading.html#with-locks)了解由`threading`模块创建的锁对象与上下文管理器之间的关系。

## 另请参阅

+   请参阅[`www.python.org/dev/peps/pep-0343/`](https://www.python.org/dev/peps/pep-0343/)了解 with 语句的起源
