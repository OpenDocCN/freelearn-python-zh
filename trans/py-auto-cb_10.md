# 调试技术

在本章中，我们将介绍以下配方：

+   学习 Python 解释器基础知识

+   通过日志调试

+   使用断点调试

+   提高你的调试技能

# 介绍

编写代码并不容易。实际上，它非常困难。即使是世界上最好的程序员也无法预见代码的任何可能的替代方案和流程。

这意味着执行我们的代码将总是产生惊喜和意外的行为。有些会非常明显，而其他的则会非常微妙，但是识别和消除代码中的这些缺陷的能力对于构建稳固的软件至关重要。

这些软件中的缺陷被称为**bug**，因此消除它们被称为**调试**。

仅通过阅读来检查代码并不好。总会有意外，复杂的代码很难跟踪。这就是为什么通过停止执行并查看当前状态的能力是重要的。

每个人，每个人都会在代码中引入 bug，通常稍后会对此感到惊讶。有些人将调试描述为*在一部犯罪电影中扮演侦探，而你也是凶手*。

任何调试过程大致遵循以下路径：

1.  你意识到有一个问题

1.  你了解正确的行为应该是什么

1.  你发现了当前代码产生 bug 的原因

1.  你改变代码以产生正确的结果

在这 95%的时间里，除了步骤 3 之外的所有事情都是微不足道的，这是调试过程的主要部分。

意识到 bug 的原因，本质上使用了科学方法：

1.  测量和观察代码的行为

1.  对为什么会这样产生假设

1.  验证或证明是否正确，也许通过实验

1.  使用得到的信息来迭代这个过程

调试是一种能力，因此随着时间的推移会得到改善。实践在培养对哪些路径看起来有希望识别错误的直觉方面起着重要作用，但也有一些一般的想法可能会帮助你：

+   **分而治之：**隔离代码的小部分，以便理解代码。尽可能简化问题。

这有一个称为**狼围栏算法**的格式，由爱德华·高斯描述：

"阿拉斯加有一只狼；你怎么找到它？首先在州的中间建造一道围栏，等待狼嚎叫，确定它在围栏的哪一边。然后只在那一边重复这个过程，直到你能看到狼为止。"

+   **从错误处向后移动：**如果在特定点有明显的错误，那么 bug 可能位于周围。从错误处逐渐向后移动，沿着轨迹直到找到错误的源头。

+   **只要你证明了你的假设，你可以假设任何东西：**代码非常复杂，无法一次性记住所有内容。您需要验证小的假设，这些假设结合起来将为检测和修复问题提供坚实的基础。进行小实验，这将允许您从头脑中删除实际工作的代码部分，并专注于未经测试的代码部分。

或用福尔摩斯的话说：

"一旦你排除了不可能的，无论多么不可能，剩下的，必定是真相。"

但记住要证明它。避免未经测试的假设。

所有这些听起来有点可怕，但实际上大多数 bug 都是相当明显的。也许是拼写错误，或者一段代码还没有准备好接受特定的值。尽量保持简单。简单的代码更容易分析和调试。

在本章中，我们将看到一些调试工具和技术，并将它们特别应用于 Python 脚本。这些脚本将有一些 bug，我们将作为配方的一部分来修复它们。

# 学习 Python 解释器基础知识

在这个配方中，我们将介绍一些 Python 内置的功能，以检查代码，调查发生了什么事情，并检测当事情不正常时。

我们还可以验证事情是否按预期进行。记住，能够排除代码的一部分作为错误源是非常重要的。

在调试时，我们通常需要分析来自外部模块或服务的未知元素和对象。鉴于 Python 的动态特性，代码在执行的任何时刻都是高度可发现的。

这个方法中的所有内容都是 Python 解释器的默认内容。

# 如何做到...

1.  导入`pprint`：

```py
>>> from pprint import pprint
```

1.  创建一个名为`dictionary`的新字典：

```py
>>> dictionary = {'example': 1}
```

1.  将`globals`显示到此环境中：

```py
>>> globals()
{...'pprint': <function pprint at 0x100995048>, 
...'dictionary': {'example': 1}}
```

1.  以可读格式使用`pprint`打印`globals`字典：

```py
>>> pprint(globals())
{'__annotations__': {},
 ...
 'dictionary': {'example': 1},
 'pprint': <function pprint at 0x100995048>}
```

1.  显示`dictionary`的所有属性：

```py
>>> dir(dictionary)
['__class__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values']
```

1.  展示`dictionary`对象的帮助：

```py
>>> help(dictionary)

Help on dict object:

class dict(object)
 | dict() -> new empty dictionary
 | dict(mapping) -> new dictionary initialized from a mapping object's
 | (key, value) pairs
...
```

# 它是如何工作的...

在第 1 步导入`pprint`（漂亮打印）之后，我们创建一个新的字典作为第 2 步中的示例。

第 3 步显示了全局命名空间包含已定义的字典和模块等内容。`globals()`显示所有导入的模块和其他全局变量。

本地命名空间有一个等效的`locals()`。

`pprint`有助于以第 4 步中更可读的格式显示`globals`，增加更多空间并将元素分隔成行。

第 5 步显示了如何使用`dir()`获取 Python 对象的所有属性。请注意，这包括所有双下划线值，如`__len__`。

使用内置的`help()`函数将显示对象的相关信息。

# 还有更多...

`dir()`特别适用于检查未知对象、模块或类。如果需要过滤默认属性并澄清输出，可以通过以下方式过滤输出：

```py
>>> [att for att in dir(dictionary) if not att.startswith('__')]
['clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values']
```

同样，如果要搜索特定方法（例如以`set`开头的方法），也可以以相同的方式进行过滤。

`help()`将显示函数或类的`docstring`。`docstring`是在定义之后定义的字符串，用于记录函数或类的信息：

```py
>>> def something():
...     '''
...     This is help for something
...     '''
...     pass
...
>>> help(something)
Help on function something in module __main__:

something()
    This is help for something
```

请注意，在下一个示例中，*这是某物的帮助*字符串是在函数定义之后定义的。

`docstring`通常用三引号括起来，以允许编写多行字符串。Python 将三引号内的所有内容视为一个大字符串，即使有换行符也是如此。您可以使用`'`或`"`字符，只要使用三个即可。您可以在[`www.python.org/dev/peps/pep-0257/`](https://www.python.org/dev/peps/pep-0257/)找到有关`docstrings`的更多信息。

内置函数的文档可以在[`docs.python.org/3/library/functions.html#built-in-functions`](https://docs.python.org/3/library/functions.html#built-in-functions)找到，`pprint`的完整文档可以在[`docs.python.org/3/library/pprint.html#`](https://docs.python.org/3/library/pprint.html#)找到。

# 另请参阅

+   *提高调试技能*的方法

+   *通过日志进行调试*的方法

# 通过日志进行调试

毕竟，调试就是检测程序内部发生了什么以及可能发生的意外或不正确的影响。一个简单但非常有效的方法是在代码的战略部分输出变量和其他信息，以跟踪程序的流程。

这种方法的最简单形式称为**打印调试**，或者在调试时在某些点插入打印语句以打印变量或点的值。

但是，稍微深入了解这种技术，并将其与第二章中介绍的日志技术相结合，*轻松实现自动化任务*使我们能够创建程序执行的半永久跟踪，这在检测运行中的程序中的问题时非常有用。

# 准备工作

从 GitHub 下载 `debug_logging.py` 文件：[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter10/debug_logging.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter10/debug_logging.py)。

它包含了冒泡排序算法的实现（[`www.studytonight.com/data-structures/bubble-sort`](https://www.studytonight.com/data-structures/bubble-sort)），这是对元素列表进行排序的最简单方式。它在列表上进行多次迭代，每次迭代都会检查并交换两个相邻的值，使得较大的值在较小的值之后。这样就使得较大的值像气泡一样在列表中上升。

冒泡排序是一种简单但天真的排序实现方式，有更好的替代方案。除非你有极好的理由，否则依赖列表中的标准 `.sort` 方法。

运行时，它检查以下列表以验证其正确性：

```py
assert [1, 2, 3, 4, 7, 10] == bubble_sort([3, 7, 10, 2, 4, 1])
```

我们在这个实现中有一个 bug，所以我们可以将其作为修复的一部分来修复！

# 如何做...

1.  运行 `debug_logging.py` 脚本并检查是否失败：

```py
$ python debug_logging.py
INFO:Sorting the list: [3, 7, 10, 2, 4, 1]
INFO:Sorted list:      [2, 3, 4, 7, 10, 1]
Traceback (most recent call last):
  File "debug_logging.py", line 17, in <module>
    assert [1, 2, 3, 4, 7, 10] == bubble_sort([3, 7, 10, 2, 4, 1])
AssertionError
```

1.  启用调试日志，更改`debug_logging.py`脚本的第二行：

```py
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
```

将前一行改为以下一行：

```py
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
```

注意不同的 `level`。

1.  再次运行脚本，增加更多信息：

```py
$ python debug_logging.py
INFO:Sorting the list: [3, 7, 10, 2, 4, 1]
DEBUG:alist: [3, 7, 10, 2, 4, 1]
DEBUG:alist: [3, 7, 10, 2, 4, 1]
DEBUG:alist: [3, 7, 2, 10, 4, 1]
DEBUG:alist: [3, 7, 2, 4, 10, 1]
DEBUG:alist: [3, 7, 2, 4, 10, 1]
DEBUG:alist: [3, 2, 7, 4, 10, 1]
DEBUG:alist: [3, 2, 4, 7, 10, 1]
DEBUG:alist: [2, 3, 4, 7, 10, 1]
DEBUG:alist: [2, 3, 4, 7, 10, 1]
DEBUG:alist: [2, 3, 4, 7, 10, 1]
INFO:Sorted list : [2, 3, 4, 7, 10, 1]
Traceback (most recent call last):
  File "debug_logging.py", line 17, in <module>
    assert [1, 2, 3, 4, 7, 10] == bubble_sort([3, 7, 10, 2, 4, 1])
AssertionError
```

1.  分析输出后，我们意识到列表的最后一个元素没有排序。我们分析代码并发现第 7 行有一个 off-by-one 错误。你看到了吗？让我们通过更改以下一行来修复它：

```py
for passnum in reversed(range(len(alist) - 1)):
```

将前一行改为以下一行：

```py
for passnum in reversed(range(len(alist))):
```

（注意移除了 `-1` 操作。）

1.  再次运行它，你会发现它按预期工作。调试日志不会显示在这里：

```py
$ python debug_logging.py
INFO:Sorting the list: [3, 7, 10, 2, 4, 1]
...
INFO:Sorted list     : [1, 2, 3, 4, 7, 10]
```

# 它是如何工作的...

第 1 步介绍了脚本，并显示代码有错误，因为它没有正确地对列表进行排序。

脚本已经有一些日志来显示开始和结束结果，以及一些调试日志来显示每个中间步骤。在第 2 步中，我们激活了显示 `DEBUG` 日志的显示，因为在第 1 步中只显示了 `INFO`。

注意，默认情况下日志会显示在标准错误输出中。这在终端中是默认显示的。如果你需要将日志重定向到其他地方，比如文件中，可以查看如何配置不同的处理程序。查看 Python 中的日志配置以获取更多详细信息：[`docs.python.org/3/howto/logging.html`](https://docs.python.org/3/howto/logging.html)。

第 3 步再次运行脚本，这次显示额外信息，显示列表中的最后一个元素没有排序。

这个 bug 是一个 off-by-one 错误，这是一种非常常见的错误，因为它应该迭代整个列表的大小。这在第 4 步中得到修复。

检查代码以了解为什么会出现错误。整个列表应该被比较，但我们错误地减少了一个大小。

第 5 步显示修复后的脚本运行正确。

# 还有更多...

在这个示例中，我们已经有策略地放置了调试日志，但在实际的调试练习中可能不是这样。你可能需要添加更多或更改位置作为 bug 调查的一部分。

这种技术的最大优势是我们能够看到程序的流程，能够检查代码执行的一个时刻到另一个时刻，并理解流程。但缺点是我们可能会得到一大堆不提供关于问题的具体信息的文本。你需要在提供太多和太少信息之间找到平衡。

出于同样的原因，除非必要，尽量限制非常长的变量。

记得在修复 bug 后降低日志级别。很可能你发现一些不相关的日志需要被删除。

这种技术的快速而粗糙的版本是添加打印语句而不是调试日志。虽然有些人对此持反对意见，但实际上这是一种用于调试目的的有价值的技术。但记得在完成后清理它们。

所有的内省元素都可用，因此可以创建显示例如`dir(object)`对象的所有属性的日志：

```py
logging.debug(f'object {dir(object)}')
```

任何可以显示为字符串的内容都可以在日志中呈现，包括任何文本操作。

# 另请参阅

+   *学习 Python 解释器基础*食谱

+   *提高调试技能*食谱

# 使用断点进行调试

Python 有一个现成的调试器叫做`pdb`。鉴于 Python 代码是解释性的，这意味着可以通过设置断点来在任何时候停止代码的执行，这将跳转到一个命令行，可以在其中使用任何代码来分析情况并执行任意数量的指令。

让我们看看如何做。

# 准备工作

下载`debug_algorithm.py`脚本，可从 GitHub 获取：[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter10/debug_algorithm.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter10/debug_algorithm.py)。

在下一节中，我们将详细分析代码的执行。代码检查数字是否符合某些属性：

```py
def valid(candidate):
    if candidate <= 1:
        return False

    lower = candidate - 1
    while lower > 1:
        if candidate / lower == candidate // lower:
            return False
        lower -= 1

    return True

assert not valid(1)
assert valid(3)
assert not valid(15)
assert not valid(18)
assert not valid(50)
assert valid(53)
```

可能你已经认识到代码在做什么，但请跟着我一起交互分析它。

# 如何做...

1.  运行代码以查看所有断言是否有效：

```py
$ python debug_algorithm.py
```

1.  在`while`循环之后添加`breakpoint()`，就在第 7 行之前，结果如下：

```py
    while lower > 1:
        breakpoint()
        if candidate / lower == candidate // lower:
```

1.  再次执行代码，看到它在断点处停止，进入交互式`Pdb`模式：

```py
$ python debug_algorithm.py
> .../debug_algorithm.py(8)valid()
-> if candidate / lower == candidate // lower:
(Pdb)
```

1.  检查候选值和两个操作的值。这一行是在检查`candidate`除以`lower`是否为整数（浮点数和整数除法是相同的）：

```py
(Pdb) candidate
3
(Pdb) candidate / lower
1.5
(Pdb) candidate // lower
1
```

1.  使用`n`继续到下一条指令。看到它结束了 while 循环并返回`True`：

```py
(Pdb) n
> ...debug_algorithm.py(10)valid()
-> lower -= 1
(Pdb) n
> ...debug_algorithm.py(6)valid()
-> while lower > 1:
(Pdb) n
> ...debug_algorithm.py(12)valid()
-> return True
(Pdb) n
--Return--
> ...debug_algorithm.py(12)valid()->True
-> return True
```

1.  继续执行，直到找到另一个断点，使用`c`。请注意，这是对`valid()`的下一个调用，输入为 15：

```py
(Pdb) c
> ...debug_algorithm.py(8)valid()
-> if candidate / lower == candidate // lower:
(Pdb) candidate
15
(Pdb) lower
14
```

1.  继续运行和检查数字，直到`valid`函数的操作有意义。你能找出代码在做什么吗？（如果你不能，不要担心，查看下一节。）完成后，使用`q`退出。这将停止执行：

```py
(Pdb) q
...
bdb.BdbQuit
```

# 工作原理...

代码正在检查一个数字是否是质数，这点你可能已经知道。它试图将数字除以比它小的所有整数。如果在任何时候可以被整除，它将返回`False`结果，因为它不是质数。

实际上，这是一个检查质数的非常低效的方法，因为处理大数字将需要很长时间。不过，对于我们的教学目的来说，它足够快。如果你有兴趣找质数，可以查看 SymPy 等数学包（[`docs.sympy.org/latest/modules/ntheory.html?highlight=prime#sympy.ntheory.primetest.isprime`](https://docs.sympy.org/latest/modules/ntheory.html?highlight=prime#sympy.ntheory.primetest.isprime)）。

在步骤 1 中检查了一般的执行，在步骤 2 中，在代码中引入了一个`breakpoint`。

当在步骤 3 中执行代码时，它会在`breakpoint`位置停止，进入交互模式。

在交互模式下，我们可以检查任何变量的值，以及执行任何类型的操作。如步骤 4 所示，有时，通过重现其部分，可以更好地分析一行代码。

可以检查代码并在命令行中执行常规操作。可以通过调用`n(ext)`来执行下一行代码，就像步骤 5 中多次执行一样，以查看代码的流程。

步骤 6 显示了如何使用`c(ontinue)`命令恢复执行，以便在下一个断点处停止。所有这些操作都可以迭代以查看流程和值，并了解代码在任何时候正在做什么。

可以使用`q(uit)`停止执行，如步骤 7 所示。

# 还有更多...

要查看所有可用的操作，可以在任何时候调用`h(elp)`。

您可以使用`l(ist)`命令在任何时候检查周围的代码。例如，在步骤 4 中：

```py
(Pdb) l
  3   return False
  4
  5   lower = candidate - 1
  6   while lower > 1:
  7     breakpoint()
  8 ->  if candidate / lower == candidate // lower:
  9       return False
 10     lower -= 1
 11
 12   return True
```

另外两个主要的调试器命令是`s(tep)`，它将执行下一步，包括进入新的调用，以及`r(eturn)`，它将从当前函数返回。

您可以使用`pdb`命令`b(reak)`设置（和禁用）更多断点。您需要为断点指定文件和行，但实际上更直接，更不容易出错的方法是改变代码并再次运行它。

您可以覆盖变量以及读取它们。或者创建新变量。或进行额外的调用。或者您能想象到的其他任何事情。Python 解释器的全部功能都在您的服务中！用它来检查某些东西是如何工作的，或者验证某些事情是否发生。

避免使用调试器保留的名称创建变量，例如将列表称为`l`。这将使事情变得混乱，并在尝试调试时干扰，有时以非明显的方式。

`breakpoint()`函数是 Python 3.7 中的新功能，但如果您使用该版本，强烈推荐使用它。在以前的版本中，您需要用以下内容替换它：

```py
import pdb; pdb.set_trace()
```

它们的工作方式完全相同。请注意同一行中的两个语句，这在 Python 中通常是不推荐的，但这是保持断点在单行中的一个很好的方法。

记得在调试完成后删除任何`breakpoints`！特别是在提交到 Git 等版本控制系统时。

您可以在官方 PEP 中阅读有关新的`breakpoint`调用的更多信息，该 PEP 描述了其用法：[`www.python.org/dev/peps/pep-0553/`](https://www.python.org/dev/peps/pep-0553/)。

完整的`pdb`文档可以在这里找到：[`docs.python.org/3.7/library/pdb.html#module-pdb`](https://docs.python.org/3.7/library/pdb.html#module-pdb)。它包括所有的调试命令。

# 另请参阅

+   *学习 Python 解释器基础*食谱

+   *改进您的调试技能*食谱

# 改进您的调试技能

在这个食谱中，我们将分析一个小脚本，它复制了对外部服务的调用，分析并修复了一些错误。我们将展示不同的技术来改进调试。

脚本将一些个人姓名 ping 到互联网服务器（`httpbin.org`，一个测试站点）以获取它们，模拟从外部服务器检索它们。然后将它们分成名和姓，并准备按姓氏排序。最后，它将对它们进行排序。

脚本包含了几个我们将检测和修复的错误。

# 准备工作

对于这个食谱，我们将使用`requests`和`parse`模块，并将它们包含在我们的虚拟环境中：

```py
$ echo "requests==2.18.3" >> requirements.txt
$ echo "parse==1.8.2" >> requirements.txt
$ pip install -r requirements.txt
```

`debug_skills.py`脚本可以从 GitHub 获取：[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter10/debug_skills.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter10/debug_skills.py)。请注意，它包含我们将在本食谱中修复的错误。

# 如何做...

1.  运行脚本，将生成错误：

```py
$ python debug_skills.py
Traceback (most recent call last):
 File "debug_skills.py", line 26, in <module>
 raise Exception(f'Error accessing server: {result}')
Exception: Error accessing server: <Response [405]>
```

1.  分析状态码。我们得到了 405，这意味着我们发送的方法不被允许。我们检查代码并意识到，在第 24 行的调用中，我们使用了`GET`，而正确的方法是`POST`（如 URL 中所述）。用以下内容替换代码：

```py
# ERROR Step 2\. Using .get when it should be .post
# (old) result = requests.get('http://httpbin.org/post', json=data)
result = requests.post('http://httpbin.org/post', json=data)
```

我们将旧的错误代码用`(old)`进行了注释，以便更清楚地进行更改。

1.  再次运行代码，将产生不同的错误：

```py
$ python debug_skills.py
Traceback (most recent call last):
  File "debug_skills_solved.py", line 34, in <module>
    first_name, last_name = full_name.split()
ValueError: too many values to unpack (expected 2)
```

1.  在第 33 行插入一个断点，一个在错误之前。再次运行它并进入调试模式：

```py
$ python debug_skills_solved.py
..debug_skills.py(35)<module>()
-> first_name, last_name = full_name.split()
(Pdb) n
> ...debug_skills.py(36)<module>()
-> ready_name = f'{last_name}, {first_name}'
(Pdb) c
> ...debug_skills.py(34)<module>()
-> breakpoint()
```

运行`n`不会产生错误，这意味着它不是第一个值。在`c`上运行几次后，我们意识到这不是正确的方法，因为我们不知道哪个输入是产生错误的。

1.  相反，我们用`try...except`块包装该行，并在那一点产生一个`breakpoint`：

```py
    try:
        first_name, last_name = full_name.split()
    except:
        breakpoint()
```

1.  我们再次运行代码。这次代码在数据产生错误的时候停止：

```py
$ python debug_skills.py
> ...debug_skills.py(38)<module>()
-> ready_name = f'{last_name}, {first_name}'
(Pdb) full_name
'John Paul Smith'
```

1.  现在原因很明显，第 35 行只允许我们分割两个单词，但如果添加中间名就会引发错误。经过一些测试，我们确定了这行来修复它：

```py
    # ERROR Step 6 split only two words. Some names has middle names
    # (old) first_name, last_name = full_name.split()
    first_name, last_name = full_name.rsplit(maxsplit=1)
```

1.  我们再次运行脚本。确保移除`breakpoint`和`try..except`块。这次，它生成了一个名字列表！并且它们按姓氏字母顺序排序。然而，一些名字看起来不正确：

```py
$ python debug_skills_solved.py
['Berg, Keagan', 'Cordova, Mai', 'Craig, Michael', 'Garc\\u00eda, Roc\\u00edo', 'Mccabe, Fathima', "O'Carroll, S\\u00e9amus", 'Pate, Poppy-Mae', 'Rennie, Vivienne', 'Smith, John Paul', 'Smyth, John', 'Sullivan, Roman']
```

谁叫`O'Carroll, S\\u00e9amus`？

1.  为了分析这个特殊情况，但跳过其余部分，我们必须创建一个`if`条件，只在第 33 行为那个名字中断。注意`in`，以避免必须完全正确：

```py
    full_name = parse.search('"custname": "{name}"', raw_result)['name']
    if "O'Carroll" in full_name:
        breakpoint()
```

1.  再次运行脚本。`breakpoint`在正确的时刻停止了：

```py
$ python debug_skills.py
> debug_skills.py(38)<module>()
-> first_name, last_name = full_name.rsplit(maxsplit=1)
(Pdb) full_name
"S\\u00e9amus O'Carroll"
```

1.  向上移动代码，检查不同的变量：

```py
(Pdb) full_name
"S\\u00e9amus O'Carroll"
(Pdb) raw_result
'{"custname": "S\\u00e9amus O\'Carroll"}'
(Pdb) result.json()
{'args': {}, 'data': '{"custname": "S\\u00e9amus O\'Carroll"}', 'files': {}, 'form': {}, 'headers': {'Accept': '*/*', 'Accept-Encoding': 'gzip, deflate', 'Connection': 'close', 'Content-Length': '37', 'Content-Type': 'application/json', 'Host': 'httpbin.org', 'User-Agent': 'python-requests/2.18.3'}, 'json': {'custname': "Séamus O'Carroll"}, 'origin': '89.100.17.159', 'url': 'http://httpbin.org/post'}
```

1.  在`result.json()`字典中，实际上有一个不同的字段，似乎正确地呈现了名字，这个字段叫做`'json'`。让我们仔细看一下；我们可以看到它是一个字典：

```py
(Pdb) result.json()['json']
{'custname': "Séamus O'Carroll"}
(Pdb) type(result.json()['json'])
<class 'dict'>
```

1.  改变代码，不要解析`'data'`中的原始值，直接使用结果中的`'json'`字段。这简化了代码，非常棒！

```py
    # ERROR Step 11\. Obtain the value from a raw value. Use
    # the decoded JSON instead
    # raw_result = result.json()['data']
    # Extract the name from the result
    # full_name = parse.search('"custname": "{name}"', raw_result)['name']
    raw_result = result.json()['json']
    full_name = raw_result['custname']
```

1.  再次运行代码。记得移除`breakpoint`：

```py
$ python debug_skills.py
['Berg, Keagan', 'Cordova, Mai', 'Craig, Michael', 'García, Rocío', 'Mccabe, Fathima', "O'Carroll, Séamus", 'Pate, Poppy-Mae', 'Rennie, Vivienne', 'Smith, John Paul', 'Smyth, John', 'Sullivan, Roman']
```

这一次，一切都正确了！您已成功调试了程序！

# 它是如何工作的...

食谱的结构分为三个不同的问题。让我们分块分析它：

+   **第一个错误-对外部服务的错误调用**：

在步骤 1 中显示第一个错误后，我们仔细阅读了产生的错误，说服务器返回了 405 状态码。这对应于不允许的方法，表明我们的调用方法不正确。

检查以下行：

```py
result = requests.get('http://httpbin.org/post', json=data)
```

它告诉我们，我们正在使用`GET`调用一个为`POST`定义的 URL，所以我们在步骤 2 中进行了更改。

请注意，在这个错误中并没有额外的调试步骤，而是仔细阅读错误消息和代码。记住要注意错误消息和日志。通常，这已经足够发现问题了。

我们在步骤 3 中运行代码，找到下一个问题。

+   **第二个错误-中间名处理错误**：

在步骤 3 中，我们得到了一个值过多的错误。我们在步骤 4 中创建一个`breakpoint`来分析这一点的数据，但发现并非所有数据都会产生这个错误。在步骤 4 中进行的分析表明，当错误没有产生时停止执行可能会非常令人困惑，必须继续直到产生错误。我们知道错误是在这一点产生的，但只对某种类型的数据产生错误。

由于我们知道错误是在某个时候产生的，我们在步骤 5 中用`try..except`块捕获它。当异常产生时，我们触发`breakpoint`。

这使得步骤 6 执行脚本时停止，当`full_name`是`'John Paul Smith'`时。这会产生一个错误，因为`split`期望返回两个元素，而不是三个。

这在步骤 7 中得到了修复，允许除了最后一个单词之外的所有内容都成为名字的一部分，将任何中间名归为第一个元素。这符合我们这个程序的目的，按姓氏排序。

实际上，名字处理起来相当复杂。如果您想对关于名字的错误假设感到惊讶，请查看这篇文章：[`www.kalzumeus.com/2010/06/17/falsehoods-programmers-believe-about-names/`](https://www.kalzumeus.com/2010/06/17/falsehoods-programmers-believe-about-names/)。

以下行使用`rsplit`：

```py
first_name, last_name = full_name.rsplit(maxsplit=1)
```

它通过单词从右边开始分割文本，最多分割一次，确保只返回两个元素。

当代码更改时，第 8 步再次运行代码以发现下一个错误。

+   **第三个错误——使用外部服务返回的错误值**：

在第 8 步运行代码会显示列表，并且不会产生任何错误。 但是，检查结果，我们可以看到一些名称被错误处理了。

我们选择第 9 步中的一个示例，并创建一个条件断点。 只有在数据满足`if`条件时才激活`breakpoint`。

在这种情况下，`if`条件在任何时候停止`"O'Carroll"`字符串出现，而不必使用等号语句使其更严格。 对于这段代码要实用主义，因为在修复错误后，您将需要将其删除。

代码在第 10 步再次运行。 从那里，一旦验证数据符合预期，我们就开始*向后*寻找问题的根源。 第 11 步分析先前的值和到目前为止的代码，试图找出导致不正确值的原因。

然后我们发现我们在从服务器的`result`返回值中使用了错误的字段。 `json`字段的值更适合这个任务，而且它已经为我们解析了。 第 12 步检查该值并查看应该如何使用它。

在第 13 步，我们更改代码进行调整。 请注意，不再需要`parse`模块，而且使用`json`字段的代码实际上更清晰。

这个结果实际上比看起来更常见，特别是在处理外部接口时。 我们可能会以一种有效的方式使用它，但也许这并不是最好的。 花点时间阅读文档，并密切关注改进并学习如何更好地使用工具。

一旦这个问题解决了，代码在第 14 步再次运行。 最后，代码按姓氏按字母顺序排列。 请注意，包含奇怪字符的其他名称也已修复。

# 还有更多...

修复后的脚本可以从 GitHub 获取：[`github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter10/debug_skills_fixed.py`](https://github.com/PacktPublishing/Python-Automation-Cookbook/blob/master/Chapter10/debug_skills_fixed.py)。 您可以下载并查看其中的差异。

还有其他创建条件断点的方法。 实际上，调试器支持创建断点，但仅当满足某些条件时才停止。 可以在 Python `pdb`文档中查看如何创建它：[`docs.python.org/3/library/pdb.html#pdbcommand-break`](https://docs.python.org/3/library/pdb.html#pdbcommand-break)。

在第一个错误中显示的捕获异常的断点类型演示了在代码中制作条件是多么简单。 只是要小心在之后删除它们！

还有其他可用的调试器具有更多功能。 例如：

+   `ipdb` ([`github.com/gotcha/ipdb`](https://github.com/gotcha/ipdb))：添加制表符补全和语法高亮显示

+   `pudb` ([`documen.tician.de/pudb/`](https://documen.tician.de/pudb/))：显示旧式的半图形文本界面，以自动显示环境变量的方式显示 90 年代早期工具的风格

+   `web-pdb` ([`pypi.org/project/web-pdb/`](https://pypi.org/project/web-pdb/))：打开一个 Web 服务器以访问带有调试器的图形界面

查看它们的文档以了解如何安装和运行它们。

还有更多可用的调试器，通过互联网搜索将为您提供更多选项，包括 Python IDE。 无论如何，要注意添加依赖项。 能够使用默认调试器总是很好的。

Python 3.7 中的新断点命令允许我们使用`PYTHONBREAKPOINT`环境变量轻松地在调试器之间切换。 例如：

```py
$ PYTHONBREAKPOINT=ipdb.set_trace python my_script.py
```

这将在代码中的任何断点上启动`ipdb`。您可以在`breakpoint()`文档中了解更多信息：[`www.python.org/dev/peps/pep-0553/#environment-variable`](https://www.python.org/dev/peps/pep-0553/#environment-variable)。

对此的一个重要影响是通过设置`PYTHONBREAKPOINT=0`来禁用所有断点，这是一个很好的工具，可以确保生产中的代码永远不会因为错误留下的`breakpoint()`而中断。

Python `pdb`文档可以在这里找到：[`docs.python.org/3/library/pdb.html`](https://docs.python.org/3/library/pdb.html) `parse`模块的完整文档可以在[`github.com/r1chardj0n3s/parse`](https://github.com/r1chardj0n3s/parse)找到，`requests`的完整文档可以在[`docs.python-requests.org/en/master/`](http://docs.python-requests.org/en/master)找到。

# 另请参阅

+   *学习 Python 解释器基础*配方

+   *使用断点进行调试*配方
