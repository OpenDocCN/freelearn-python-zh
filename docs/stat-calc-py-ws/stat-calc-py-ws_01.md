# 第一章：Python 基础

概述

本章回顾了将在未来讨论中使用的基本 Python 数据结构和工具。这些概念将使我们能够刷新我们对 Python 最基本和重要特性的记忆，同时为以后章节中的高级主题做好准备。

通过本章结束时，您将能够使用控制流方法设计 Python 程序并初始化常见的 Python 数据结构，以及操纵它们的内容。您将巩固对 Python 算法设计中函数和递归的理解。您还将能够为 Python 程序进行调试、测试和版本控制。最后，在本章末尾的活动中，您将创建一个数独求解器。

# 介绍

Python 近年来在数学领域的受欢迎程度和使用率前所未有。然而，在深入讨论数学的高级主题之前，我们需要巩固对语言基础知识的理解。

本章将对 Python 的一般概念进行复习；所涵盖的主题将使您在本书的后续讨论中处于最佳位置。具体来说，我们将复习一般编程中的基本概念，如条件和循环，以及 Python 特定的数据结构，如列表和字典。我们还将讨论函数和算法设计过程，这是包括与数学相关的程序在内的任何中型或大型 Python 项目中的重要部分。所有这些都将通过实践练习和活动来完成。

通过本章结束时，您将能够在本书后续章节中处理更复杂、更有趣的问题。

# 控制流方法

控制流是一个通用术语，表示可以重定向程序执行的任何编程语法。一般来说，控制流方法是使程序在执行和计算时具有动态性的原因：根据程序的当前状态或输入，程序的执行和输出将动态改变。

## if 语句

任何编程语言中最常见的控制流形式是条件语句，或者`if`语句。`if`语句用于检查程序当前状态的特定条件，并根据结果（条件是真还是假）执行不同的指令集。

在 Python 中，`if`语句的语法如下：

```py
if [condition to check]:
    [instruction set to execute if condition is true]
```

鉴于 Python 的可读性，你可能已经猜到条件语句的工作原理：当给定程序的执行达到条件语句并检查`if`语句中的条件时，如果条件为真，则将执行缩进的指令集*在*`if`语句内部；否则，程序将简单地跳过这些指令并继续执行。

在`if`语句内部，我们可以检查复合条件，这是多个单独条件的组合。例如，使用`and`关键字，当满足其两个条件时，将执行以下`if`块：

```py
if [condition 1] and [condition 2]:
    [instruction set]
```

与此相反，我们可以在复合条件中使用`or`关键字，如果关键字左侧或右侧的条件为真，则显示正（真）。还可以使用多个`and`/`or`关键字扩展复合条件，以实现嵌套在多个级别上的条件语句。

当条件不满足时，我们可能希望程序执行不同的一组指令。为了实现这种逻辑，我们可以使用`elif`和`else`语句，这些语句应该紧随在`if`语句之后。如果`if`语句中的条件不满足，我们的程序将继续并评估`elif`语句中的后续条件；如果没有一个条件被满足，`else`块中的任何代码都将被执行。Python 中的`if...elif...else`块的形式如下：

```py
if [condition 1]:
    [instruction set 1]
elif [condition 2]:
    [instruction set 2]
...
elif [condition n]:
    [instruction set n]
else:
    [instruction set n + 1]
```

当程序需要检查一组可能性时，这种控制流方法非常有价值。根据给定时刻的真实可能性，程序应该执行相应的指令。

## 练习 1.01：条件除法

在数学中，对变量及其内容的分析是非常常见的，其中最常见的分析之一是整数的可整除性。在这个练习中，我们将使用`if`语句来考虑给定数字是否可以被 5、6 或 7 整除。

为了实现这一点，请按照以下步骤进行：

1.  创建一个新的 Jupyter 笔记本，并声明一个名为`x`的变量，其值为任何整数，如下面的代码所示：

```py
x = 130
```

1.  在声明之后，编写一个`if`语句，检查`x`是否可以被 5 整除。相应的代码块应该打印出一个指示条件是否满足的语句：

```py
if x % 5 == 0:
    print('x is divisible by 5')
```

在这里，`%`是 Python 中的取模运算符；`var % n`表达式返回当我们用数字`n`除以变量`var`时的余数。

1.  在同一个代码单元格中，编写两个`elif`语句，分别检查`x`是否可以被 6 和 7 整除。适当的`print`语句应该放在相应条件下面：

```py
elif x % 6 == 0:
    print('x is divisible by 6')
elif x % 7 == 0:
    print('x is divisible by 7')
```

1.  编写最终的`else`语句，以打印一条消息，说明`x`既不能被 5 整除，也不能被 6 或 7 整除（在同一个代码单元格中）：

```py
else:
    print('x is not divisible by 5, 6, or 7')
```

1.  每次为`x`分配一个不同的值来测试我们的条件逻辑。以下输出是`x`被赋值为`104832`的一个示例：

```py
x is divisible by 6
```

1.  现在，我们不想打印关于`x`的可整除性的消息，而是想将该消息写入文本文件。具体来说，我们想创建一个名为`output.txt`的文件，其中包含我们先前打印出的相同消息。

为了做到这一点，我们可以使用`with`关键字和`open()`函数与文本文件进行交互。请注意，`open()`函数接受两个参数：要写入的文件名，在我们的例子中是`output.txt`，以及`w`（表示写入），指定我们想要写入文件，而不是从文件中读取内容：

```py
if x % 5 == 0:
    with open('output.txt', 'w') as f:
        f.write('x is divisible by 5')
elif x % 6 == 0:
    with open('output.txt', 'w') as f:
        f.write('x is divisible by 6')
elif x % 7 == 0:
    with open('output.txt', 'w') as f:
        f.write('x is divisible by 7')
else:
    with open('output.txt', 'w') as f:
        f.write('x is not divisible by 5, 6, or 7')
```

1.  检查输出文本文件中的消息是否正确。如果`x`变量仍然保持值`104832`，则您的文本文件应该包含以下内容：

```py
x is divisible by 6
```

在这个练习中，我们应用了条件语句来编写一个程序，使用`%`运算符来确定给定数字是否可以被 6、3 和 2 整除。我们还学习了如何在 Python 中向文本文件写入内容。在下一节中，我们将开始讨论 Python 中的循环。

注意

`elif`块中的代码行按顺序执行，并在任何一个条件为真时中断序列。这意味着当 x 被赋值为 30 时，一旦满足`x%5==0`，就不会检查`x%6==0`。

要访问此特定部分的源代码，请参阅[`packt.live/3dNflxO.`](https://packt.live/3dNflxO )

您也可以在[`packt.live/2AsqO8w`](https://packt.live/2AsqO8w)上在线运行此示例。

## 循环

另一个广泛使用的控制流方法是使用循环。这些用于在指定范围内重复执行相同的指令，或者在满足条件时重复执行相同的指令。Python 中有两种类型的循环：`while`循环和`for`循环。让我们详细了解每种循环。

### while 循环

`while`循环，就像`if`语句一样，检查指定的条件，以确定给定程序的执行是否应该继续循环。例如，考虑以下代码：

```py
>>> x = 0
>>> while x < 3:
...     print(x)
...     x += 1
0
1
2
```

在前面的代码中，`x`被初始化为值`0`后，使用`while`循环来连续打印变量的值，并在每次迭代中递增相同的变量。可以想象，当这个程序执行时，将打印出`0`、`1`和`2`，当`x`达到`3`时，`while`循环中指定的条件不再满足，因此循环结束。

请注意，`x += 1`命令对应于`x = x + 1`，它在循环的每次迭代中增加`x`的值。如果我们删除这个命令，那么我们将得到一个无限循环，每次打印`0`。

### for 循环

另一方面，`for`循环通常用于迭代特定序列的值。使用 Python 中的`range`函数，以下代码产生了与我们之前相同的输出：

```py
>>> for x in range(3):
...     print(x)
0
1
2
```

`in`关键字是 Python 中任何`for`循环的关键：当使用它时，其前面的变量将被分配在我们想要顺序循环的迭代器中的值。在前面的例子中，`x`变量被分配了`range(3)`迭代器中的值，依次是`0`、`1`和`2`，在`for`循环的每次迭代中。

在 Python 的`for`循环中，除了`range()`之外，还可以使用其他类型的迭代器。以下表格简要总结了一些最常见的用于`for`循环的迭代器。如果您对此表中包含的数据结构不熟悉，不要担心；我们将在本章后面介绍这些概念：

![图 1.1：数据集及其示例列表](img/B15968_01_01.jpg)

图 1.1：数据集及其示例列表

还可以在彼此内部嵌套多个循环。当给定程序的执行位于循环内部时，我们可以使用`break`关键字退出当前循环并继续执行。

## 练习 1.02：猜数字游戏

在这个练习中，我们将把我们对循环的知识付诸实践，并编写一个简单的猜数字游戏。程序开始时随机选择一个介于 0 和 100 之间的目标整数。然后，程序将接受用户输入作为猜测这个数字的猜测。作为回应，程序将打印出一条消息，如果猜测大于实际目标，则打印`Lower`，如果相反，则打印`Higher`。当用户猜对时，程序应该终止。

执行以下步骤完成这个练习：

1.  在新的 Jupyter 笔记本的第一个单元格中，导入 Python 中的`random`模块，并使用其`randint`函数生成随机数：

```py
import random
true_value = random.randint(0, 100)
```

每次调用`randint()`函数时，它都会生成两个传递给它的数字之间的随机整数；在我们的情况下，将生成介于 0 和 100 之间的整数。

虽然它们在本练习的其余部分中并不需要，但如果您对随机模块提供的其他功能感兴趣，可以查看其官方文档[`docs.python.org/3/library/random.html`](https://docs.python.org/3/library/random.html)。

注意

程序的其余部分也应该放在当前代码单元格中。

1.  使用 Python 中的`input()`函数接受用户的输入，并将返回的值赋给一个变量（在以下代码中为`guess`）。这个值将被解释为用户对目标的猜测：

```py
guess = input('Enter your guess: ')
```

1.  使用`int()`函数将用户输入转换为整数，并将其与真实目标进行比较。针对比较的所有可能情况打印出适当的消息：

```py
guess = int(guess)
if guess == true_value:
    print('Congratulations! You guessed correctly.')
elif guess > true_value:
    print('Lower.')  # user guessed too high
else:
    print('Higher.')  # user guessed too low
```

注意

下面代码片段中的`#`符号表示代码注释。注释被添加到代码中，以帮助解释特定的逻辑部分。

1.  使用我们当前的代码，如果`int()`函数的输入无法转换为整数（例如，输入为字符串字符），它将抛出错误并使整个程序崩溃。因此，我们需要在`try...except`块中实现我们的代码，以处理用户输入非数字值的情况：

```py
try:
    if guess == true_value:
        print('Congratulations! You guessed correctly.')
    elif guess > true_value:
        print('Lower.')  # user guessed too high
    else:
        print('Higher.')  # user guessed too low
# when the input is invalid
except ValueError:
    print('Please enter a valid number.')
```

1.  目前，用户只能在程序终止之前猜一次。为了实现允许用户重复猜测直到找到目标的功能，我们将迄今为止开发的逻辑包装在一个`while`循环中，只有当用户猜对时（通过适当放置`while True`循环和`break`关键字来实现）才会中断。

完整的程序应该类似于以下代码：

```py
import random
true_value = random.randint(0, 100)
while True:
    guess = input('Enter your guess: ')
    try:
        guess = int(guess)
        if guess == true_value:
            print('Congratulations! You guessed correctly.')
            break
        elif guess > true_value:
            print('Lower.')  # user guessed too high
        else:
            print('Higher.')  # user guessed too low
    # when the input is invalid
    except ValueError:
        print('Please enter a valid number.')
```

1.  尝试通过执行代码单元格重新运行程序，并尝试不同的输入选项，以确保程序可以很好地处理其指令，并处理无效输入的情况。例如，当目标数字被随机选择为 13 时，程序可能产生的输出如下：

```py
Enter your guess: 50
Lower.
Enter your guess: 25
Lower.
Enter your guess: 13
Congratulations! You guessed correctly.
```

在这个练习中，我们已经练习了在猜数字游戏中使用`while`循环，以巩固我们对编程中循环使用的理解。此外，您已经了解了在 Python 中读取用户输入和`random`模块的方法。

注意

要访问此特定部分的源代码，请参阅[`packt.live/2BYK6CR.`](https://packt.live/2BYK6CR )

您也可以在[`packt.live/2CVFbTu`](https://packt.live/2CVFbTu)上在线运行此示例。

接下来，我们将开始考虑常见的 Python 数据结构。

# 数据结构

数据结构是代表您可能想要在程序中创建、存储和操作的不同形式的信息的变量类型。与控制流方法一起，数据结构是任何编程语言的另一个基本构建块。在本节中，我们将介绍 Python 中一些最常见的数据结构，从字符串开始。

## 字符串

字符串是字符序列，通常用于表示文本信息（例如，消息）。Python 字符串由单引号或双引号中的任何给定文本数据表示。例如，在以下代码片段中，`a`和`b`变量保存相同的信息：

```py
a = 'Hello, world!'
b = "Hello, world!"
```

由于在 Python 中字符串大致被视为序列，因此可以将常见的与序列相关的操作应用于此数据结构。特别是，我们可以将两个或多个字符串连接在一起以创建一个长字符串，我们可以使用`for`循环遍历字符串，并且可以使用索引和切片访问单个字符和子字符串。这些操作的效果在以下代码中进行了演示：

```py
>>> a = 'Hello, '
>>> b = 'world!'
>>> print(a + b)
Hello, world!
>>> for char in a:
...     print(char)
H
e
l
l
o
,
 # a blank character printed here, the last character in string a
>>> print(a[2])
l
>>> print(a[1: 4]) 
ell
```

在 Python 3.6 中添加的最重要的功能之一是 f-strings，这是 Python 中格式化字符串的语法。由于我们使用的是 Python 3.7，因此可以使用此功能。字符串格式化用于在我们想要将给定变量的值插入预定义字符串时使用。在 f-strings 之前，还有两种其他格式化选项，您可能熟悉：%-格式化和`str.format()`。不详细介绍这两种方法，这两种方法都有一些不良特性，因此开发了 f-strings 来解决这些问题。

f-strings 的语法是用大括号`{`和`}`定义的。例如，我们可以使用 f-string 将变量的打印值组合如下：

```py
>>> a = 42
>>> print(f'The value of a is {a}.')
The value of a is 42.
```

当将变量放入 f-string 大括号中时，它的`__str__()`表示将在最终的打印输出中使用。这意味着在使用 Python 对象时，您可以通过覆盖和自定义 dunder 方法`__str__()`来获得 f-strings 的更多灵活性。

在 f-strings 中，可以使用冒号来指定字符串的常见数字格式选项，例如指定小数点后的位数或日期时间格式，如下所示：

```py
>>> from math import pi
>>> print(f'Pi, rounded to three decimal places, is {pi:.3f}.')
Pi, rounded to three decimal places, is 3.142.
>>> from datetime import datetime
>>> print(f'Current time is {datetime.now():%H:%M}.')
Current time is 21:39.
```

f-strings 的另一个好处是它们比其他两种字符串格式化方法更快渲染和处理。接下来，让我们讨论 Python 列表。

## 列表

列表可以说是 Python 中最常用的数据结构。它是 Python 版本的 Java 或 C/C++中的数组。列表是可以按顺序访问或迭代的元素序列。与 Java 数组不同，Python 列表中的元素不必是相同的数据结构，如下所示：

```py
>>> a = [1, 'a', (2, 3)]  # a list containing a number, a string, and a tuple
```

注意

我们将在下一节更多地讨论元组。

正如我们之前讨论过的，列表中的元素可以在`for`循环中以与字符串中字符类似的方式进行迭代。列表也可以像字符串一样进行索引和切片：

```py
>>> a = [1, 'a', (2, 3), 2]
>>> a[2]
(2, 3)
>>> a[1: 3]
['a', (2, 3)]
```

有两种方法可以向 Python 列表添加新元素：`append()`将一个新的单个元素插入到列表的末尾，而列表连接简单地将两个或多个字符串连接在一起，如下所示：

```py
>>> a = [1, 'a', (2, 3)]
>>> a.append(3)
>>> a
[1, 'a', (2, 3), 3]
>>> b = [2, 5, 'b']
>>> a + b
[1, 'a', (2, 3), 3, 2, 5, 'b']
```

要从列表中删除一个元素，可以使用`pop()`方法，该方法接受要删除的元素的索引。

使 Python 列表独特的操作之一是列表推导：一种 Python 语法，可以使用放置在方括号内的`for`循环来高效地初始化列表。列表推导通常用于当我们想要对现有列表应用操作以创建新列表时。例如，假设我们有一个包含一些整数的列表变量`a`：

```py
>>> a = [1, 4, 2, 9, 10, 3]
```

现在，我们想要创建一个新的列表`b`，其元素是`a`中元素的两倍，按顺序。我们可以潜在地将`b`初始化为空列表，并迭代地遍历`a`并将适当的值附加到`b`。然而，使用列表推导，我们可以用更优雅的语法实现相同的结果：

```py
>>> b = [2 * element for element in a]
>>> b
[2, 8, 4, 18, 20, 6]
```

此外，我们甚至可以在列表推导中结合条件语句来实现在创建 Python 列表的过程中的复杂逻辑。例如，要创建一个包含`a`中奇数元素两倍的列表，我们可以这样做：

```py
>>> c = [2 * element for element in a if element % 2 == 1]
>>> c
[2, 18, 6]
```

另一个经常与列表进行对比的 Python 数据结构是元组，我们将在下一节中讨论。然而，在继续之前，让我们通过一个新概念的练习来了解多维列表/数组。

多维数组，也称为表或矩阵（有时称为张量），是数学和机器学习领域中常见的对象。考虑到 Python 列表中的元素可以是任何 Python 对象，我们可以使用列表中的列表来模拟跨越多个维度的数组。具体来说，想象一下，在一个总体的 Python 列表中，我们有三个子列表，每个子列表中有三个元素。这个对象可以被看作是一个 2D 的 3 x 3 表。一般来说，我们可以使用嵌套在其他列表中的 Python 列表来模拟*n*维数组。

## 练习 1.03：多维列表

在这个练习中，我们将熟悉多维列表的概念以及通过它们进行迭代的过程。我们的目标是编写逻辑命令，动态显示 2D 列表的内容。

执行以下步骤完成此练习：

1.  创建一个新的 Jupyter 笔记本，并在一个代码单元格中声明一个名为`a`的变量，如下所示：

```py
a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

这个变量表示一个 3 x 3 的 2D 表，列表中的各个子列表表示行。

1.  在一个新的代码单元格中，通过循环遍历列表`a`中的元素来迭代行（暂时不要运行单元格）：

```py
for row in a:
```

1.  在这个`for`循环的每次迭代中，`a`中的一个子列表被分配给一个名为`row`的变量。然后，我们可以通过索引访问 2D 表中的单个单元格。以下`for`循环将打印出每个子列表中的第一个元素，或者换句话说，表中每行的第一个单元格中的数字（`1`、`4`和`7`）：

```py
for row in a:
    print(row[0])
```

1.  在一个新的代码单元格中，通过嵌套的`for`循环打印出表`a`中所有单元格的值，内部循环将遍历`a`中的子列表：

```py
for row in a:
    for element in row:
        print(element)
```

这应该打印出从 1 到 9 的数字，每个数字在单独的行中。

1.  最后，在一个新的单元格中，我们需要以格式良好的消息打印出这个表的对角线元素。为此，我们可以使用一个索引变量——在我们的例子中是`i`——从`0`循环到`2`来访问表的对角线元素：

```py
for i in range(3):
    print(a[i][i])
```

您的输出应该是 1、5 和 9，每个在单独的行中。

注意

这是因为表/矩阵中对角线元素的行索引和列索引相等。

1.  在一个新的单元格中，使用 f-strings 更改前面的`print`语句以格式化我们的打印输出：

```py
for i in range(3):
    print(f'The {i + 1}-th diagonal element is: {a[i][i]}')
```

这应该产生以下输出：

```py
The 1-th diagonal element is: 1
The 2-th diagonal element is: 5
The 3-th diagonal element is: 9
```

在这个练习中，我们结合了关于循环、索引和 f-string 格式化的知识，创建了一个动态迭代 2D 列表的程序。

注意

要访问此特定部分的源代码，请参阅[`packt.live/3dRP8OA.`](https://packt.live/3dRP8OA )

您也可以在[`packt.live/3gpg4al`](https://packt.live/3gpg4al)上线上运行此示例。

接下来，我们将继续讨论其他 Python 数据结构。

## 元组

用括号而不是方括号声明的 Python 元组仍然是不同元素的序列，类似于列表（尽管在赋值语句中可以省略括号）。这两种数据结构之间的主要区别在于元组是 Python 中的不可变对象——这意味着它们在初始化后无法以任何方式进行变异或更改，如下所示：

```py
>>> a = (1, 2)
>>> a[0] = 3  # trying to change the first element
Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
>>> a.append(2)  # trying to add another element
Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
AttributeError: 'tuple' object has no attribute 'append'
```

鉴于元组和列表之间的这一关键差异，我们可以相应地利用这些数据结构：当我们希望一个元素序列由于任何原因（例如，确保逻辑完整性函数）是不可变的时，可以使用元组；如果允许序列在初始化后进行更改，可以将其声明为列表。

接下来，我们将讨论数学计算中常见的数据结构：集合。

## 集合

如果您已经熟悉数学概念，Python 集合的定义本质上是相同的：Python 集合是无序元素的集合。可以使用大括号初始化集合，并可以使用`add()`方法向集合添加新元素，如下所示：

```py
>>> a = {1, 2, 3}
>>> a.add(4)
>>> a
{1, 2, 3, 4}
```

由于集合是 Python 元素的集合，或者换句话说，是一个迭代器，因此其元素仍然可以使用`for`循环进行迭代。但是，鉴于其定义，不能保证这些元素将以与它们初始化或添加到集合中相同的顺序进行迭代。

此外，当将已存在于集合中的元素添加到该集合时，该语句将不起作用：

```py
>>> a
{1, 2, 3, 4}
>>> a.add(3)
>>> a
{1, 2, 3, 4}
```

对两个给定集合进行并集或交集操作是最常见的集合操作，并可以分别通过 Python 中的`union()`和`intersection()`方法来实现：

```py
>>> a = {1, 2, 3, 4}
>>> b = {2, 5, 6}
>>> a.union(b)
{1, 2, 3, 4, 5, 6}
>>> a.intersection(b)
{2}
```

最后，要从集合中删除给定的元素，我们可以使用`discard()`方法或`remove()`方法。两者都会从集合中删除传递给它们的项目。但是，如果项目不存在于集合中，前者将不会改变集合，而后者将引发错误。与元组和列表一样，您可以选择在程序中使用这两种方法之一来实现特定逻辑，具体取决于您的目标。

接下来，我们将讨论本节中要讨论的最后一个 Python 数据结构，即字典。

## 字典

Python 字典相当于 Java 中的哈希映射，我们可以指定键值对关系，并对键进行查找以获得其对应的值。我们可以通过在花括号内用逗号分隔的形式列出键值对来声明 Python 字典。

例如，一个包含学生姓名映射到他们在课堂上的最终成绩的样本字典可能如下所示：

```py
>>> score_dict = {'Alice': 90, 'Bob': 85, 'Carol': 86}
>>> score_dict
{'Alice': 90, 'Bob': 85, 'Carol': 86}
```

在这种情况下，学生的姓名（'Alice'，'Bob'和'Carol'）是字典的键，而他们的成绩是键映射到的值。一个键不能用来映射到多个不同的值。可以通过将键传递给方括号内的字典来访问给定键的值：

```py
>>> score_dict['Alice']
90
>>> score_dict['Carol']
86
>>> score_dict['Chris']
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'Chris'
```

请注意，在前面片段的最后一个语句中，`'Chris'`不是字典中的键，因此当我们尝试访问它的值时，Python 解释器会返回`KeyError`。

可以使用相同的语法更改现有键的值或向现有字典添加新的键值对：

```py
>>> score_dict['Alice'] = 89
>>> score_dict
{'Alice': 89, 'Bob': 85, 'Carol': 86}
>>> score_dict['Chris'] = 85
>>> score_dict
{'Alice': 89, 'Bob': 85, 'Carol': 86, 'Chris': 85}
```

类似于列表推导，可以使用字典推导来声明 Python 字典。例如，以下语句初始化了一个将整数从`-1`到`1`（包括边界）映射到它们的平方的字典：

```py
>>> square_dict = {i: i ** 2 for i in range(-1, 2)}
>>> square_dict
{-1: 1, 0: 0, 1: 1}
```

正如我们所看到的，这个字典包含了每个`x`在`-1`和`1`之间的`x` - `x ** 2`键值对，这是通过在字典初始化中放置`for`循环来完成的。

要从字典中删除键值对，我们需要使用`del`关键字。假设我们想删除`'Alice'`键及其对应的值。我们可以这样做：

```py
>>> del score_dict['Alice']
```

尝试访问已删除的键将导致 Python 解释器引发错误：

```py
>>> score_dict['Alice']
KeyError: 'Alice'
```

要牢记的 Python 字典最重要的一点是，只有不可变对象可以作为字典键。到目前为止，我们已经看到字符串和数字作为字典键。列表是可变的，初始化后可以改变，不能用作字典键；而元组可以。

## 练习 1.04：购物车计算

在这个练习中，我们将使用字典数据结构构建一个购物应用程序的骨架版本。这将使我们能够复习和进一步了解数据结构以及可以应用于它的操作。

执行以下步骤完成此练习：

1.  在第一个代码单元中创建一个新的 Jupyter 笔记本，并声明一个字典，表示可以购买的任何商品及其相应的价格。在这里，我们将添加三种不同类型的笔记本电脑及其美元价格：

```py
prices = {'MacBook 13': 1300, 'MacBook 15': 2100, \
          'ASUS ROG': 1600}
```

注意

这里显示的代码片段使用反斜杠（`\`）将逻辑分割成多行。当代码执行时，Python 将忽略反斜杠，并将下一行的代码视为当前行的直接延续。

1.  在下一个单元格中，初始化一个表示我们购物车的字典。字典在开始时应该是空的，但它应该将购物车中的商品映射到要购买的副本数量：

```py
cart = {}
```

1.  在一个新的单元格中，编写一个`while True`循环，表示购物过程的每个步骤，并询问用户是否想继续购物。使用条件语句来处理输入的不同情况（您可以留下用户想要继续购物直到下一步的情况）：

```py
while True:
    _continue = input('Would you like to continue '\
                      'shopping? [y/n]: ')
    if _continue == 'y':
        ...
    elif _continue == 'n':
        break
    else:
        print('Please only enter "y" or "n".')
```

1.  在第一个条件情况下，接受另一个用户输入，询问应该将哪个商品添加到购物车。使用条件语句来增加`cart`字典中商品的数量或处理无效情况：

```py
    if _continue == 'y':
        print(f'Available products and prices: {prices}')
        new_item = input('Which product would you like to '\
                         'add to your cart? ')
        if new_item in prices:
            if new_item in cart:
                cart[new_item] += 1
            else:
                cart[new_item] = 1
        else:
            print('Please only choose from the available products.')
```

1.  在下一个单元格中，循环遍历`cart`字典，并计算用户需要支付的总金额（通过查找购物车中每件商品的数量和价格）：

```py
# Calculation of total bill.
running_sum = 0
for item in cart:
    running_sum += cart[item] * prices[item]  # quantity times price
```

1.  最后，在一个新的单元格中，通过`for`循环打印出购物车中的商品及其各自的数量，并在最后打印出总账单。使用 f-string 格式化打印输出：

```py
print(f'Your final cart is:')
for item in cart:
    print(f'- {cart[item]} {item}(s)')
print(f'Your final bill is: {running_sum}')
```

1.  运行程序并尝试使用不同的购物车来确保我们的程序是正确的。例如，如果您要将两台 MacBook 13 和一台华硕 ROG 添加到我的购物车中并停止，相应的输出将如下所示：![图 1.2：购物车应用程序的输出](img/B15968_01_02.jpg)

图 1.2：购物车应用程序的输出

这就结束了我们的购物车练习，通过这个练习，我们熟悉了使用字典查找信息。我们还回顾了使用条件和循环来实现控制流方法在 Python 程序中的使用。

注意

要访问本节的源代码，请参阅[`packt.live/2C1Ra1C`](https://packt.live/2C1Ra1C)

您也可以在[`packt.live/31F7QXg`](https://packt.live/31F7QXg)上线运行此示例。

在下一节中，我们将讨论任何复杂程序的两个重要组成部分：函数和算法。

# 函数和算法

虽然函数在 Python 编程中表示特定的对象，我们可以用它来对程序进行排序和分解，但术语*算法*通常指的是一系列逻辑的一般组织，用于处理给定的输入数据。在数据科学和科学计算中，算法是无处不在的，通常以处理数据并可能进行预测的机器学习模型的形式出现。

在本节中，我们将讨论 Python 函数的概念和语法，然后解决一些示例算法设计问题。

## 函数

在其最抽象的定义中，函数只是一个可以接受输入并根据给定的一组指令产生输出的对象。Python 函数的形式如下：

```py
def func_name(param1, param2, ...):
     […]
    return […]
```

`def`关键字表示 Python 函数的开始。函数的名称可以是任何东西，尽管规则是避免名称开头的特殊字符，并使用蛇形命名法。括号内是函数接受的参数，它们用逗号分隔，并可以在函数的缩进代码中使用。

例如，以下函数接受一个字符串（尽管这个要求未指定），并打印出问候消息：

```py
>>> def greet(name):
...     print(f'Hello, {name}!')
```

然后，我们可以在任何想要的字符串上调用这个函数，并实现函数内部指令所期望的效果。如果我们以某种方式错误地指定了函数所需的参数（例如，以下代码片段中的最后一条语句），解释器将返回一个错误：

```py
>>> greet('Quan')
Hello, Quan!
>>> greet('Alice')
Hello, Alice!
>>> greet()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: greet() missing 1 required positional argument: 'name'
```

重要的是要注意，任何局部变量（在函数内部声明的变量）都不能在函数范围之外使用。换句话说，一旦函数完成执行，它的变量将无法被其他代码访问。

大多数情况下，我们希望我们的函数在结束时返回某种值，这是由`return`关键字实现的。一旦执行`return`语句，程序的执行将退出给定的函数，并返回调用函数的父级范围。这使我们能够设计许多动态逻辑序列。

例如，想象一个函数，它接受一个 Python 整数列表，并返回第一个可以被 2 整除的元素（如果列表中没有偶数元素，则返回`False`）：

```py
def get_first_even(my_list):
    [...]
    return  # should be the first even element
```

现在，编写这个函数的自然方式是循环遍历列表中的元素，并检查它们是否可以被`2`整除：

```py
def get_first_even(my_list):
    for item in my_list:
        if item % 2 == 0:
            [...]
    return  # should be the first even element
```

然而，如果条件满足（即我们正在迭代的当前元素可以被`2`整除），那么该元素应该是函数的返回值，因为它是列表中第一个可以被`2`整除的元素。这意味着我们实际上可以在`if`块内返回它（最后在函数末尾返回`False`）：

```py
def get_first_even(my_list):
    for item in my_list:
        if item % 2 == 0:
            return item
    return False
```

这种方法与另一种版本形成对比，另一种版本只在循环结束时返回满足我们条件的元素，这将更耗时（执行方面），并需要额外的检查，以确定输入列表中是否有偶数元素。我们将在下一个练习中深入研究这种逻辑的变体。

## 练习 1.05：查找最大值

在任何入门编程课程中，查找数组或列表的最大/最小值是一个常见的练习。在这个练习中，我们将考虑这个问题的一个提升版本，即我们需要编写一个函数，返回列表中最大元素的索引和实际值（如果需要进行平局处理，我们返回最后一个最大元素）。

执行以下步骤完成这个练习：

1.  创建一个新的 Jupyter 笔记本，并在一个代码单元格中声明我们目标函数的一般结构：

```py
def get_max(my_list):
    ...
    return ...
```

1.  创建一个变量来跟踪当前最大元素的索引，称为`running_max_index`，应初始化为`0`：

```py
def get_max(my_list):
    running_max_index = 0
    ...
    return ...
```

1.  使用`for`循环和`enumerate`操作循环遍历参数列表中的值及其对应的索引：

```py
def get_max(my_list):
    running_max_index = 0
    # Iterate over index-value pairs.
    for index, item in enumerate(my_list):
         [...]
    return ...
```

1.  在每一步迭代中，检查当前元素是否大于或等于与运行索引变量对应的元素。如果是这样，将当前元素的索引分配给运行的最大索引：

```py
def get_max(my_list):
    running_max_index = 0
    # Iterate over index-value pairs.
    for index, item in enumerate(my_list):
        if item >= my_list[running_max_index]:
            running_max_index = index
    return [...]
```

1.  最后，将运行的最大索引及其对应的值作为一个元组返回：

```py
def get_max(my_list):
    running_max_index = 0
    # Iterate over index-value pairs.
    for index, item in enumerate(my_list):
        if item >= my_list[running_max_index]:
            running_max_index = index
    return running_max_index, my_list[running_max_index]
```

1.  在一个新的单元格中，调用这个函数来测试不同情况下的各种列表。一个例子如下：

```py
>>> get_max([1, 3, 2])
(1, 3)
>>>  get_max([1, 3, 56, 29, 100, 99, 3, 100, 10, 23])
(7, 100)
```

这个练习帮助我们复习了 Python 函数的一般语法，也提供了一个循环的复习。此外，我们考虑的逻辑变体通常在科学计算项目中找到（例如，在迭代器中找到最小值或满足某些给定条件的元素）。

注意

要访问本节的源代码，请参阅[`packt.live/2Zu6KuH.`](https://packt.live/2Zu6KuH )

您也可以在[`packt.live/2BUNjDk.`](https://packt.live/2BUNjDk )上线运行此示例

接下来，让我们讨论一种非常特定的函数设计风格，称为*递归*。

## 递归

在编程中，术语**递归**表示使用函数解决问题的风格，通过使函数递归调用自身。其思想是每次调用函数时，其逻辑将向问题的解决方案迈出一小步，通过这样做多次，最终解决原始问题。如果我们以某种方式有办法将我们的问题转化为一个可以以相同方式解决的小问题，我们可以重复分解问题以达到基本情况，并确保解决原始的更大问题。

考虑计算*n*个整数的总和的问题。如果我们已经有了前*n-1*个整数的总和，那么我们只需将最后一个数字加到这个总和中，就可以计算出*n*个数字的总和。但是如何计算前*n-1*个数字的总和呢？通过递归，我们再次假设如果我们有前*n-2*个数字的总和，那么我们将最后一个数字加进去。这个过程重复进行，直到我们达到列表中的第一个数字，整个过程完成。

让我们在以下示例中考虑这个函数：

```py
>>> def find_sum(my_list):
...     if len(my_list) == 1:
...             return my_list[0]
...     return find_sum(my_list[: -1]) + my_list[-1]
```

我们可以看到，在一般情况下，该函数计算并返回了将输入列表的最后一个元素`my_list[-1]`与不包括这个最后一个元素的子列表`my_list[: -1]`的总和的结果，而这又是由`find_sum()`函数本身计算的。同样，我们可以理解，如果`find_sum()`函数可以以某种方式解决在较小情况下对列表求和的问题，我们可以将结果推广到任何给定的非空列表。

处理基本情况因此是任何递归算法的一个组成部分。在这里，我们的基本情况是当输入列表是一个单值列表（通过我们的`if`语句检查），在这种情况下，我们应该简单地返回列表中的那个元素。

我们可以看到这个函数正确地计算了任何非空整数列表的总和，如下所示：

```py
>>> find_sum([1, 2, 3])
6
>>> find_sum([1])
1
```

这是一个相当基本的例子，因为可以通过保持运行总和并使用`for`循环来迭代输入列表中的所有元素来轻松地找到列表的总和。实际上，大多数情况下，递归不如迭代高效，因为在程序中重复调用函数会产生重大开销。

然而，正如我们将在接下来的练习中看到的那样，通过将我们对问题的方法抽象为递归算法，我们可以显著简化问题的解决方法。

## 练习 1.06：汉诺塔

汉诺塔是一个众所周知的数学问题，也是递归的一个经典应用。问题陈述如下。

有三个盘堆，可以在其中放置盘子，有*n*个盘子，所有盘子都是不同大小的。一开始，盘子按升序堆叠（最大的在底部）在一个单独的堆栈中。在游戏的每一步中，我们可以取一个堆栈的顶部盘子，并将其放在另一个堆栈的顶部（可以是一个空堆栈），条件是不能将盘子放在比它更小的盘子的顶部。

我们被要求计算将整个*n*个盘子从一个堆栈移动到另一个堆栈所需的最小移动次数。如果我们以线性方式思考这个问题，它可能会变得非常复杂，但是当我们使用递归算法时，它变得更简单。

具体来说，为了移动*n*个盘子，我们需要将顶部的*n - 1*个盘子移动到另一个堆栈，将底部最大的盘子移动到最后一个堆栈，最后将另一个堆栈中的*n - 1*个盘子移动到与最大盘子相同的堆栈中。现在，想象我们可以计算移动*(n - 1)*个盘子所需的最小步骤，表示为*S(n - 1)*，然后移动*n*个盘子，我们需要*2 S(n - 1) + 1*步。

这就是问题的递归分析解决方案。现在，让我们编写一个函数来实际计算任何给定*n*的数量。

执行以下步骤以完成此练习：

1.  在一个新的 Jupyter 笔记本中，定义一个函数，该函数接受一个名为`n`的整数，并返回我们之前得到的数量：

```py
def solve(n):
    return 2 * solve(n - 1) + 1
```

1.  在函数中创建一个条件来处理基本情况，即`n = 1`（注意，只需一步即可移动单个盘子）：

```py
def solve(n):
    if n == 1:
        return 1
    return 2 * solve(n - 1) + 1
```

1.  在另一个单元格中，调用该函数以验证函数返回问题的正确分析解决方案，即*2*n *- 1*：

```py
>>> print(solve(3) == 2 ** 3 - 1)
True
>>> print(solve(6) == 2 ** 6 - 1)
True
```

在这里，我们使用`==`运算符来比较两个值：从我们的`solve()`函数返回的值和解决方案的分析表达式。如果它们相等，我们应该看到布尔值`True`被打印出来，这是我们这里的两个比较的情况。

在这个练习中的代码虽然很短，但它已经说明了递归可以提供优雅的解决方案来解决许多问题，并且希望巩固了我们对递归算法程序的理解（包括一般步骤和基本案例）。

注意

要访问此特定部分的源代码，请参考[`packt.live/2NMrGrk.`](https://packt.live/2NMrGrk )

您也可以在[`packt.live/2AnAP6R`](https://packt.live/2AnAP6R)上在线运行此示例。

有了这个，我们将继续讨论算法设计的一般过程。

## 算法设计

设计算法实际上是我们一直在做的事情，特别是在本节中，这一节主要讨论函数和算法：讨论一个函数对象应该接受什么，它应该如何处理输入，以及在执行结束时应该返回什么输出。在本节中，我们将简要讨论一般算法设计过程中的一些实践，然后考虑一个稍微复杂的问题，称为*N-Queens 问题*作为练习。

在编写 Python 函数时，一些程序员可能选择实现子函数（在其他函数中的函数）。遵循软件开发中的封装思想，当子函数只被另一个函数内的指令调用时，应该实现子函数。如果是这种情况，第一个函数可以被视为第二个函数的辅助函数，因此应该*放在*第二个函数内。这种封装形式使我们能够更有条理地组织我们的程序/代码，并确保如果一段代码不需要使用给定函数内的逻辑，则不应该访问它。

下一个讨论点涉及递归搜索算法，我们将在下一个练习中进行讨论。具体来说，当算法递归地尝试找到给定问题的有效解决方案时，它可能会达到一个没有有效解决方案的状态（例如，当我们试图在仅包含奇数的列表中找到一个偶数元素时）。这导致需要一种方式来指示我们已经达到了一个无效状态。

在我们找到第一个偶数的例子中，我们选择返回`False`来指示一个无效状态，即我们的输入列表只包含奇数。返回`False`或`0`这样的标志实际上是一个常见的做法，我们在本章的后续示例中也会遵循这种做法。

考虑到这一点，让我们开始本节的练习。

## 练习 1.07：N-Queens 问题

数学和计算机科学中的另一个经典算法设计问题是 N 皇后问题，它要求我们在* n * x * n *棋盘上放置* n *个皇后棋子，以便没有皇后棋子可以攻击另一个。如果两个皇后棋子共享相同的行、列或对角线，那么一个皇后可以攻击另一个棋子，因此问题实质上是找到皇后棋子的位置组合，使得任意两个皇后在不同的行、列和对角线上。

对于这个练习，我们将设计一个*回溯*算法，为任何正整数*n*搜索这个问题的有效解决方案。算法如下：

1.  考虑到问题的要求，我们认为为了放置*n*个棋子，棋盘的每一行都需要包含恰好一个棋子。

1.  对于每一行，我们迭代地遍历该行的所有单元格，并检查是否可以在给定单元格中放置一个新的皇后棋子：

a. 如果存在这样的单元格，我们在该单元格中放置一个棋子，然后转到下一行。

b. 如果新的皇后棋子无法放置在当前行的任何单元格中，我们知道已经达到了一个无效状态，因此返回`False`。

1.  我们重复这个过程，直到找到一个有效的解决方案。

以下图描述了这个算法在*n=4*时的工作方式：

![图 1.3：N-Queens 问题的递归](img/B15968_01_03.jpg)

图 1.3：N-Queens 问题的递归

现在，让我们实际实现算法：

1.  创建一个新的 Jupyter 笔记本。在第一个单元格中，声明一个名为`N`的变量，表示棋盘的大小，以及我们需要在棋盘上放置的皇后数量：

```py
N = 8
```

1.  国际象棋棋盘将被表示为一个 2D 的*n* x *n*列表，其中 0 表示一个空单元格，1 表示一个带有皇后棋子的单元格。现在，在一个新的代码单元中，实现一个函数，该函数接受这种形式的列表并以良好的格式打印出来：

```py
# Print out the board in a nice format.
def display_solution(board):
    for i in range(N):
        for j in range(N):
            print(board[i][j], end=' ')
        print()
```

请注意，我们`print`语句中的`end=' '`参数指定，不是用换行符结束打印输出，而是用空格字符。这样我们就可以使用不同的`print`语句打印出同一行中的单元格。

1.  在下一个单元格中，编写一个函数，该函数接受一个棋盘、一个行号和一个列号。该函数应该检查是否可以在给定的行和列号位置的棋盘上放置一个新的皇后棋子。

请注意，由于我们正在逐行放置棋子，每次检查新棋子是否可以放在给定位置时，我们只需要检查位置上方的行：

```py
# Check if a queen can be placed in the position.
def check_next(board, row, col):
    # Check the current column.
    for i in range(row):
        if board[i][col] == 1:
            return False
    # Check the upper-left diagonal.
    for i, j in zip(range(row, -1, -1), \
                    range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    # Check the upper-right diagonal.
    for i, j in zip(range(row, -1, -1), \
                    range(col, N)):
        if board[i][j] == 1:
            return False
    return True
```

1.  在同一个代码单元中，实现一个函数，该函数接受一个棋盘和一个行号。该函数应该遍历给定行中的所有单元格，并检查是否可以在特定单元格放置一个新的皇后棋子（使用前面步骤中编写的`check_next()`函数）。

对于这样的单元格，在该单元格中放置一个皇后（将单元格值更改为`1`），并递归调用函数本身以获取下一个行号。如果最终解决方案有效，则返回`True`；否则，从单元格中移除皇后棋子（将其更改回`0`）。

如果在考虑了给定行的所有单元格后没有找到有效解决方案，则返回`False`表示`无效`状态。函数还应该在开始时有一个条件检查，检查行号是否大于棋盘大小`N`，在这种情况下，我们只需返回`True`表示已经找到有效的最终解决方案：

```py
def recur_generate_solution(board, row_id):
    # Return if we have reached the last row.
    if row_id >= N:
        return True
    # Iteratively try out cells in the current row.
    for i in range(N):
        if check_next(board, row_id, i):
            board[row_id][i] = 1 
            # Return if a valid solution is found.
            final_board = recur_generate_solution(\
                          board, row_id + 1)
            if final_board:
                return True
            board[row_id][i] = 0  
    # When the current board has no valid solutions.
    return False
```

1.  在同一个代码单元中，编写一个最终求解器函数，该函数包装了两个函数`check_next()`和`recur_generate_solution()`（换句话说，这两个函数应该是我们正在编写的函数的子函数）。该函数应该初始化一个空的 2D *n* x *n*列表（表示国际象棋棋盘），并调用`recur_generate_solution()`函数，行号为 0。

函数还应该在最后打印出解决方案：

```py
# Generate a valid solution.
def generate_solution():
    # Check if a queen can be placed in the position.
    def check_next(board, row, col):
        [...]
    # Recursively generate a solution.
    def recur_generate_solution(board, row_id):
        [...]
    # Start out with en empty board.
    my_board = [[0 for _ in range(N)] for __ in range(N)]
    final_solution = recur_generate_solution(my_board, 0)
    # Display the final solution.
    if final_solution is False:
        print('A solution cannot be found.')
    else:
        print('A solution was found.')
        display_solution(my_board)
```

1.  在另一个代码单元中，运行前面步骤中的总体函数以生成并打印出解决方案：

```py
>>> generate_solution()
A solution was found.
1 0 0 0 0 0 0 0 
0 0 0 0 1 0 0 0 
0 0 0 0 0 0 0 1 
0 0 0 0 0 1 0 0 
0 0 1 0 0 0 0 0 
0 0 0 0 0 0 1 0 
0 1 0 0 0 0 0 0 
0 0 0 1 0 0 0 0 
```

在整个练习过程中，我们实现了一个回溯算法，该算法旨在通过迭代向潜在解决方案迈出一步（在安全单元格中放置一个皇后棋子），如果算法以某种方式达到无效状态，它将通过撤消先前的移动（在我们的情况下，通过移除我们放置的最后一个棋子）并寻找新的移动来进行*回溯*。正如您可能已经注意到的那样，回溯与递归密切相关，这就是为什么我们选择使用递归函数来实现我们的算法，从而巩固我们对一般概念的理解。

注意

要访问此特定部分的源代码，请参阅[`packt.live/2Bn7nyt.`](https://packt.live/2Bn7nyt )

您还可以在[`packt.live/2ZrKRMQ.`](https://packt.live/2ZrKRMQ )上在线运行此示例

在本章的下一个和最后一节中，我们将考虑 Python 编程中经常被忽视的一些行政任务，即调试、测试和版本控制。

# 测试、调试和版本控制

在编程中，需要注意的是，编写代码的实际任务并不是整个过程的唯一元素。还有其他行政程序在流程中扮演着重要角色，但通常被忽视了。在本节中，我们将逐个讨论每个任务，并考虑在 Python 中实现它们的过程，从测试开始。

## 测试

为了确保我们编写的软件按照我们的意图工作并产生正确的结果，有必要对其进行特定的测试。在软件开发中，我们可以对程序应用多种类型的测试：集成测试、回归测试、系统测试等等。其中最常见的是单元测试，这是我们在本节讨论的主题。

单元测试表示关注软件的个别小单元，而不是整个程序。单元测试通常是测试流水线的第一步——一旦我们相当有信心认为程序的各个组件工作正常，我们就可以继续测试这些组件如何一起工作，以及它们是否产生我们想要的结果（通过集成或系统测试）。

在 Python 中，可以使用`unittest`模块轻松实现单元测试。采用面向对象的方法，`unittest`允许我们将程序的测试设计为 Python 类，使过程更加模块化。这样的类需要从`unittest`的`TestCase`类继承，并且单独的测试需要在不同的函数中实现，如下所示：

```py
import unittest
class SampleTest(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(2 ** 3 - 1, 7)
        self.assertEqual('Hello, world!', 'Hello, ' + 'world!')

    def test_true(self):
        self.assertTrue(2 ** 3 < 3 ** 2)
        for x in range(10):
            self.assertTrue(- x ** 2 <= 0)
```

在`SampleTest`类中，我们放置了两个测试用例，希望使用`assertEqual()`方法在`test_equal()`函数中检查两个给定的数量是否相等。在这里，我们测试 23-1 是否确实等于 7，以及 Python 中的字符串连接是否正确。

类似地，`test_true()`函数中使用的`assertTrue()`方法测试给定参数是否被评估为`True`。在这里，我们测试 23 是否小于 32，以及 0 到 10 之间整数的完全平方的负值是否为非正数。

要运行我们实现的测试，可以使用以下语句：

```py
>>> unittest.main()
test_equal (__main__.SampleTest) ... ok
test_true (__main__.SampleTest) ... ok
----------------------------------------------------------------------
Ran 2 tests in 0.001s
OK
```

生成的输出告诉我们，我们的两个测试都返回了积极的结果。需要记住的一个重要的副作用是，如果在 Jupyter 笔记本中运行单元测试，最后的语句需要如下所示：

```py
unittest.main(argv=[''], verbosity=2, exit=False)
```

由于单元测试需要作为 Python 类中的函数实现，`unittest`模块还提供了两个方便的方法`setUp()`和`tearDown()`，它们分别在每个测试之前和之后自动运行。我们将在下一个练习中看到这方面的一个例子。现在，我们将继续讨论调试。

## 调试

*调试*一词的字面意思是从给定的计算机程序中消除一个或多个错误，从而使其正确工作。在大多数情况下，调试过程是在测试失败后进行的，确定程序中存在错误。然后，为了调试程序，我们需要确定导致测试失败的错误的源头，并尝试修复与该错误相关的代码。

程序可能采用多种形式的调试。这些包括以下内容：

+   **打印调试**：可以说是最常见和基本的调试方法之一，打印调试涉及识别可能导致错误的变量，在程序中的各个位置放置这些变量的`print`语句，以便跟踪这些变量值的变化。一旦发现变量值的变化是不希望的，我们就会查看程序中`print`语句的具体位置，从而（粗略地）确定错误的位置。

+   日志记录：如果我们决定将变量的值输出到日志文件而不是标准输出，这就被称为日志记录。通常会使用日志记录来跟踪我们正在调试或监视的程序执行中发生的特定事件。

+   **跟踪**: 要调试一个程序，在这种情况下，我们将跟踪程序执行时的低级函数调用和执行堆栈。通过从低级别的角度观察变量和函数的使用顺序，我们也可以确定错误的来源。在 Python 中，可以使用`sys`模块的`sys.settrace()`方法来实现跟踪。

在 Python 中，使用打印调试非常容易，因为我们只需要使用`print`语句。对于更复杂的功能，我们可以使用调试器，这是专门设计用于调试目的的模块/库。Python 中最主要的调试器是内置的`pdb`模块，以前是通过`pdb.set_trace()`方法运行的。

从 Python 3.7 开始，我们可以选择更简单的语法，通过调用内置的`breakpoint()`函数。在每个调用`breakpoint()`函数的地方，程序的执行将暂停，允许我们检查程序的行为和当前特性，包括其变量的值。

具体来说，一旦程序执行到`breakpoint()`函数，将会出现一个输入提示，我们可以在其中输入`pdb`命令。模块的文档中包含了许多可以利用的命令。一些值得注意的命令如下：

+   `h`: 用于*帮助*，打印出您可以使用的完整命令列表。

+   `u`/`d`: 分别用于*上*和*下*，将运行帧计数向一个方向移动一级。

+   `s`: 用于*步骤*，执行程序当前所在的指令，并在执行中的第一个可能的位置暂停。这个命令在观察代码对程序状态的即时影响方面非常有用。

+   `n`: 用于*下一个*，执行程序当前所在的指令，并且只在当前函数中的下一个指令处暂停，当执行返回时也会暂停。这个命令与`s`有些类似，不过它以更高的速率跳过指令。

+   `r`: 用于*返回*，直到当前函数返回为止。

+   `c`: 用于*继续*，直到达到下一个`breakpoint()`语句为止。

+   `ll`: 用于*longlist*，打印出当前指令的源代码。

+   `p [expression]`: 用于*打印*，评估并打印给定表达式的值

总的来说，一旦程序的执行被`breakpoint()`语句暂停，我们可以利用前面不同命令的组合来检查程序的状态并识别潜在的错误。我们将在下面的练习中看一个例子。

## 练习 1.08: 并发测试

在这个练习中，我们将考虑并发或并行相关程序中一个众所周知的错误，称为*竞争条件*。这将作为一个很好的用例来尝试我们的测试和调试工具。由于在 Jupyter 笔记本中集成`pdb`和其他调试工具仍处于不成熟阶段，所以我们将在这个练习中使用`.py`脚本。

执行以下步骤来完成这个练习：

1.  我们程序的设置（在以下步骤中实现）如下。我们有一个类，实现了一个计数器对象，可以被多个线程并行操作。这个计数器对象的实例的值（存储在其初始化为`0`的`value`属性中）在每次调用其`update()`方法时递增。计数器还有一个目标，即其值应该递增到。当调用其`run()`方法时，将会生成多个线程。每个线程将调用`update()`方法，因此将其`value`属性递增到与原始目标相等的次数。理论上，计数器的最终值应该与目标相同，但由于竞争条件，我们将看到这并不是这样。我们的目标是应用`pdb`来跟踪程序内部变量的变化，以分析这种竞争条件。

1.  创建一个新的`.py`脚本，并输入以下代码：

```py
import threading
import sys; sys.setswitchinterval(10 ** -10)
class Counter:
    def __init__(self, target):
        self.value = 0
        self.target = target        
    def update(self):
        current_value = self.value
        # breakpoint()
        self.value = current_value + 1

    def run(self):
        threads = [threading.Thread(target=self.update) \
                                    for _ in range(self.target)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
```

这段代码实现了我们之前讨论过的`Counter`类。请注意，有一行代码设置了系统的切换间隔；我们稍后会讨论这个。

1.  希望`counter`对象的值应该增加到其真正的目标值，我们将用三个不同的目标值测试其性能。在同一个`.py`脚本中，输入以下代码来实现我们的单元测试：

```py
import unittest
class TestCounter(unittest.TestCase):
    def setUp(self):
        self.small_params = 5
        self.med_params = 5000
        self.large_params = 10000

    def test_small(self):
        small_counter = Counter(self.small_params)
        small_counter.run()
        self.assertEqual(small_counter.value, \
                         self.small_params)

    def test_med(self):
        med_counter = Counter(self.med_params)
        med_counter.run()
        self.assertEqual(med_counter.value, \
                         self.med_params)

    def test_large(self):
        large_counter = Counter(self.large_params)
        large_counter.run()
        self.assertEqual(large_counter.value, \
                         self.large_params)
    if __name__ == '__main__':
        unittest.main()
```

在这里，我们可以看到在每个测试函数中，我们初始化一个新的`counter`对象，运行它，最后将其值与真实目标进行比较。测试用例的目标在`setUp()`方法中声明，正如我们之前提到的，在测试执行之前运行：

```py
Run this Python script:test_large (__main__.TestCounter) ... FAIL
test_med (__main__.TestCounter) ... FAIL
test_small (__main__.TestCounter) ... ok
====================================================================
FAIL: test_large (__main__.TestCounter)
--------------------------------------------------------------------
Traceback (most recent call last):
    File "<ipython-input-57-4ed47b9310ba>", line 22, in test_large
    self.assertEqual(large_counter.value, self.large_params)
AssertionError: 9996 != 10000
====================================================================
FAIL: test_med (__main__.TestCounter)
--------------------------------------------------------------------
Traceback (most recent call last):
    File "<ipython-input-57-4ed47b9310ba>", line 17, in test_med
    self.assertEqual(med_counter.value, self.med_params)
AssertionError: 4999 != 5000
--------------------------------------------------------------------
Ran 3 tests in 0.890s
FAILED (failures=2)
```

正如你所看到的，程序在两个测试中失败了：`test_med`（计数器的最终值只有 4,999，而不是 5,000）和`test_large`（值为 9,996，而不是 10,000）。你可能会得到不同的输出。

1.  多次重新运行这段代码，看到结果可能会有所不同。

1.  现在我们知道程序中有一个 bug，我们将尝试调试它。在`update()`方法的两条指令之间放置一个`breakpoint()`语句，重新实现我们的`Counter`类，如下面的代码所示，并重新运行代码：

```py
class Counter:
    ...
    def update(self):
        current_value = self.value
        breakpoint()
        self.value = current_value + 1
    ...
```

1.  在我们的 Python 脚本的主范围内，注释掉对单元测试的调用。相反，声明一个新的`counter`对象，并使用终端运行脚本：

```py
sample_counter = Counter(10)
sample_counter.run()
```

在这里，你会看到终端中出现一个`pdb`提示（你可能需要先按*Enter*让调试器继续）：

![图 1.4：pdb 界面](img/B15968_01_04.jpg)

图 1.4：pdb 界面

1.  输入`ll`并按*Enter*键，查看我们在程序中暂停的位置：

```py
(Pdb) ll
  9         def update(self):
 10             current_value = self.value
 11             breakpoint()
 12  ->         self.value = current_value + 1
```

这里，输出表明我们当前在`update()`方法内增加计数器值的两条指令之间暂停。

1.  再次按*Enter*返回到`pdb`提示符，并运行`p self.value`命令：

```py
(Pdb) p self.value
0
```

我们可以看到计数器的当前值是`0`。

1.  返回到提示符并输入`n`命令。然后再次使用`p self.value`命令检查计数器的值：

```py
(Pdb) n
--Return--
> <ipython-input-61-066f5069e308>(12)update()->None
-> self.value = current_value + 1
(Pdb) p self.value
1
```

1.  我们可以看到值已经增加了 1。重复这个在`n`和`p self.value`之间交替的过程，观察在程序进行过程中`self.value`中存储的值没有更新。换句话说，值通常保持在 1。这就是我们在计数器的大值中看到的 bug 表现方式，就像我们在单元测试中看到的那样。

1.  使用*Ctrl* + *C*退出调试器。

注意

要访问这一特定部分的源代码，请参阅[`packt.live/2YPCZFJ`](https://packt.live/2YPCZFJ)。

这一部分目前没有在线交互示例，需要在本地运行。

对于那些感兴趣的人，我们程序的错误源于多个线程可以在大致相同的时间增加计数器的值，覆盖彼此所做的更改。随着线程数量的增加（例如我们在测试用例中有的 5,000 或 10,000），这种事件发生的概率变得更高。正如我们之前提到的，这种现象称为竞争条件，是并发和并行程序中最常见的错误之一。

除了演示一些`pdb`命令之外，这个练习还说明了设计测试以覆盖不同情况是必要的事实。虽然程序通过了我们的目标为 5 的小测试，但在目标值较大时失败了。在现实生活中，我们应该对程序进行测试，模拟各种可能性，确保程序即使在边缘情况下也能正常工作。

有了这些，让我们继续进行本章的最后一个主题，版本控制。

## 版本控制

在本节中，我们将简要讨论版本控制的一般理论，然后讨论使用 Git 和 GitHub 实现版本控制的过程，这两者可以说是行业中最流行的版本控制系统。版本控制对于编程项目来说就像定期备份数据对于常规文件一样重要。实质上，版本控制系统允许我们将项目中的进度与本地文件分开保存，以便以后可以回到它，即使本地文件丢失或损坏。

使用当前版本控制系统（如 Git 和 GitHub）提供的功能，我们还可以做更多事情。例如，这些系统的分支和合并功能为用户提供了一种创建共同项目的多个版本的方法，以便可以探索不同的方向；实现最受欢迎方向的分支最终将与主分支合并。此外，Git 和 GitHub 允许平台上的用户之间的工作无缝进行，这在团队项目中非常受欢迎。

为了了解我们可以利用 Git 和 GitHub 的可用功能，让我们进行以下练习。

## 练习 1.09：使用 Git 和 GitHub 进行版本控制

这个练习将引导我们完成开始使用 Git 和 GitHub 所需的所有步骤。如果您还没有使用版本控制的经验，这个练习对您将是有益的。

执行以下步骤完成此练习：

1.  首先，如果您还没有，请注册 GitHub 帐户，方法是访问[`www.github.com/`](https://www.github.com/)并注册。这将允许您在他们的云存储上托管您想要进行版本控制的文件。

1.  前往[`git-scm.com/downloads`](https://git-scm.com/downloads)并下载适用于您系统的 Git 客户端软件并安装。这个 Git 客户端将负责与 GitHub 服务器通信。如果您可以在终端中运行`git`命令，那么您就知道您的 Git 客户端已成功安装：

```py
$ git
usage: git [--version] [--help] [-C <path>] [-c <name>=<value>]
           [--exec-path[=<path>]] [--html-path] [--man-path] [--info-path]
           [-p | --paginate | -P | --no-pager] [--no-replace-objects] [--bare]
           [--git-dir=<path>] [--work-tree=<path>] [--namespace=<name>]
           <command> [<args>]
```

否则，您的系统可能需要重新启动才能完全生效。

1.  现在，让我们开始将版本控制应用于一个示例项目的过程。首先，创建一个虚拟文件夹，并生成一个 Jupyter 笔记本和一个名为`input.txt`的文本文件，其中包含以下内容：

```py
1,1,1
1,1,1
```

1.  在 Jupyter 笔记本的第一个单元格中，编写一个名为`add_elements()`的函数，该函数接受两个数字列表并按元素相加。该函数应返回一个由元素和总和组成的列表；您可以假设两个参数列表的长度相同：

```py
def add_elements(a, b):
    result = []
    for item_a, item_b in zip(a, b):
        result.append(item_a + item_b)
    return result
```

1.  在下一个代码单元格中，使用`with`语句读取`input.txt`文件，并使用`readlines()`函数和列表索引提取文件的最后两行：

```py
with open('input.txt', 'r') as f:
    lines = f.readlines()
last_line1, last_line2 = lines[-2], lines[-1]
```

请注意，在`open()`函数中，第二个参数`'r'`指定我们正在读取文件，而不是写入文件。

1.  在一个新的代码单元格中，使用`str.split()`函数和`','`参数将这两个文本输入字符串转换为数字列表，然后使用`map()`和`int()`函数逐个元素地应用转换：

```py
list1 = list(map(int, last_line1[: -1].split(',')))
list2 = list(map(int, last_line2[: -1].split(',')))
```

1.  在一个新的代码单元格中，对`list1`和`list2`调用`add_elements()`。将返回的列表写入相同的输入文件中的新行，格式为**逗号分隔值**（**CSV**）：

```py
new_list = add_elements(list1, list2)
with open('input.txt', 'a') as f:
    for i, item in enumerate(new_list):
        f.write(str(item))

        if i < len(new_list) - 1:
            f.write(',')
        else:
            f.write('\n')
```

这里的`'a'`参数指定我们正在写入文件以追加一个新行，而不是完全读取和覆盖文件。

1.  运行代码单元格，并验证文本文件是否已更新为以下内容：

```py
1,1,1
1,1,1
2,2,2
```

1.  到目前为止，我们的示例项目的当前设置是：我们有一个文件夹中的文本文件和 Python 脚本；当运行时，脚本可以更改文本文件的内容。这种设置在现实生活中是相当常见的：您可以有一个包含一些信息的数据文件，您希望跟踪，并且可以读取该数据并以某种方式更新它的 Python 程序（也许是通过预先指定的计算或添加外部收集的新数据）。

现在，让我们在这个示例项目中实现版本控制。

1.  转到您的在线 GitHub 帐户，单击窗口右上角的加号图标（**+**），然后选择`New repository`选项，如下所示：![图 1.5：创建一个新的存储库](img/B15968_01_05.jpg)

图 1.5：创建一个新的存储库

在表单中输入一个新存储库的示例名称，并完成创建过程。将这个新存储库的 URL 复制到剪贴板上，因为我们以后会用到它。

正如名称所示，这将创建一个新的在线存储库，用于托管我们想要进行版本控制的代码。

1.  在您的本地计算机上，打开终端并导航到文件夹。运行以下命令以初始化本地 Git 存储库，这将与我们的文件夹关联：

```py
$ git init
```

1.  仍然在终端中，运行以下命令将我们项目中的所有内容添加到 Git 并提交它们：

```py
git add .
git commit -m [any message with double quotes]
```

您可以用文件的名称替换`git add .`中的`.`。当您只想注册一个或两个文件时，这个选项是有帮助的，而不是您在文件夹中的每个文件。

1.  现在，我们需要链接我们的本地存储库和我们创建的在线存储库。为此，请运行以下命令：

```py
git remote add origin [URL to GitHub repository]
```

请注意，“origin”只是 URL 的一个传统昵称。

1.  最后，通过运行以下命令将本地注册的文件上传到在线存储库：

```py
git push origin master
```

1.  转到在线存储库的网站，验证我们创建的本地文件是否确实已上传到 GitHub。

1.  在您的本地计算机上，运行 Jupyter 笔记本中包含的脚本并更改文本文件。

1.  现在，我们想要将这个更改提交到 GitHub 存储库。在您的终端上，再次运行以下命令：

```py
git add .
git commit
git push origin master
```

1.  转到 GitHub 网站验证我们第二次所做的更改是否也已在 GitHub 上进行了更改。

通过这个练习，我们已经走过了一个示例版本控制流水线，并看到了 Git 和 GitHub 在这方面的一些用法示例。我们还复习了使用`with`语句在 Python 中读写文件的过程。

注意

要访问本节的源代码，请参阅[`packt.live/2VDS0IS`](https://packt.live/2VDS0IS)

您还可以在[`packt.live/3ijJ1pM`](https://packt.live/3ijJ1pM)上在线运行此示例。

这也结束了本书第一章的最后一个主题。在下一节中，我们提供了一个活动，这个活动将作为一个实践项目，概括了本章中我们讨论的重要主题和内容。

## 活动 1.01：构建数独求解器

让我们通过一个更复杂的问题来测试我们迄今为止学到的知识：编写一个可以解决数独谜题的程序。该程序应能够读取 CSV 文本文件作为输入（其中包含初始谜题），并输出该谜题的完整解决方案。

这个活动作为一个热身，包括科学计算和数据科学项目中常见的多个程序，例如从外部文件中读取数据并通过算法操纵这些信息。

1.  使用本章的 GitHub 存储库中的`sudoku_input_2.txt`文件作为程序的输入文件，将其复制到下一步中将要创建的 Jupyter 笔记本的相同位置（或者创建一个格式相同的自己的输入文件，其中空单元格用零表示）。

1.  在新的 Jupyter 笔记本的第一个代码单元中，创建一个`Solver`类，该类接受输入文件的路径。它应将从输入文件中读取的信息存储在一个 9x9 的 2D 列表中（包含九个子列表，每个子列表包含谜题中各行的九个值）。

1.  添加一个辅助方法，以以下方式打印出谜题的格式：

```py
-----------------------
0 0 3 | 0 2 0 | 6 0 0 | 
9 0 0 | 3 0 5 | 0 0 1 | 
0 0 1 | 8 0 6 | 4 0 0 | 
-----------------------
0 0 8 | 1 0 2 | 9 0 0 | 
7 0 0 | 0 0 0 | 0 0 8 | 
0 0 6 | 7 0 8 | 2 0 0 | 
-----------------------
0 0 2 | 6 0 9 | 5 0 0 | 
8 0 0 | 2 0 3 | 0 0 9 | 
0 0 5 | 0 1 0 | 3 0 0 | 
-----------------------
```

1.  在类中创建一个`get_presence(cells)`方法，该方法接受任何 9x9 的 2D 列表，表示未解决/半解决的谜题，并返回一个关于给定数字（1 到 9 之间）是否出现在给定行、列或象限中的指示器。

例如，在前面的示例中，该方法的返回值应能够告诉您第一行中是否存在 2、3 和 6，而第二列中是否没有数字。

1.  在类中创建一个`get_possible_values(cells)`方法，该方法还接受表示不完整解决方案的任何 2D 列表，并返回一个字典，其键是当前空单元格的位置，相应的值是这些单元格可以取的可能值的列表/集合。

这些可能值的列表应通过考虑一个数字是否出现在给定空单元格的同一行、列或象限中来生成。

1.  在类中创建一个`simple_update(cells)`方法，该方法接受任何 2D 不完整解决方案列表，并在该列表上调用`get_possible_values()`方法。根据返回的值，如果有一个只包含一个可能解的空单元格，就用该值更新该单元格。

如果发生了这样的更新，该方法应再次调用自身以继续更新单元格。这是因为更新后，剩余空单元格的可能值列表可能会发生变化。该方法最终应返回更新后的 2D 列表。

1.  在类中创建一个`recur_solve(cells)`方法，该方法接受任何 2D 不完整解决方案列表并执行回溯。首先，该方法应调用`simple_update()`，并返回谜题是否完全解决（即 2D 列表中是否有空单元格）。

接下来，考虑剩余空单元格的可能值。如果还有空单元格，并且没有可能的值，返回一个负结果，表示我们已经达到了一个无效的解决方案。

另一方面，如果所有单元格至少有两个可能的值，找到可能值最少的单元格。依次循环这些可能的值，将它们填入空单元格，并在其中调用`recur_solve()`以使用算法的递归性质更新单元格。在每次迭代中，返回最终解是否有效。如果通过任何可能的值都找不到有效的最终解决方案，则返回一个负结果。

1.  将前面的方法封装在一个`solve()`方法中，该方法应打印出初始的谜题，将其传递给`recur_solve()`方法，并打印出该方法返回的解决方案。

例如，在前面的谜题中，当调用`solve()`时，`Solver`实例将打印出以下输出。

初始谜题：

```py
-----------------------
0 0 3 | 0 2 0 | 6 0 0 | 
9 0 0 | 3 0 5 | 0 0 1 | 
0 0 1 | 8 0 6 | 4 0 0 | 
-----------------------
0 0 8 | 1 0 2 | 9 0 0 | 
7 0 0 | 0 0 0 | 0 0 8 | 
0 0 6 | 7 0 8 | 2 0 0 | 
-----------------------
0 0 2 | 6 0 9 | 5 0 0 | 
8 0 0 | 2 0 3 | 0 0 9 | 
0 0 5 | 0 1 0 | 3 0 0 | 
-----------------------
```

解决的谜题：

```py
-----------------------
4 8 3 | 9 2 1 | 6 5 7 | 
9 6 7 | 3 4 5 | 8 2 1 | 
2 5 1 | 8 7 6 | 4 9 3 | 
-----------------------
5 4 8 | 1 3 2 | 9 7 6 | 
7 2 9 | 5 6 4 | 1 3 8 | 
1 3 6 | 7 9 8 | 2 4 5 | 
-----------------------
3 7 2 | 6 8 9 | 5 1 4 | 
8 1 4 | 2 5 3 | 7 6 9 | 
6 9 5 | 4 1 7 | 3 8 2 | 
-----------------------
```

扩展

1. 前往*Project Euler*网站，[`projecteuler.net/problem=96`](https://projecteuler.net/problem=96)，测试你的算法是否能解决包含的谜题。

2. 编写一个程序，生成数独谜题，并包括单元测试，检查我们的求解器生成的解是否正确。

注意

此活动的解决方案可在第 648 页找到。

# 摘要

本章介绍了 Python 编程的最基本构建模块：控制流、数据结构、算法设计以及各种日常任务（调试、测试和版本控制）。我们在本章中获得的知识将为我们在未来章节中的讨论做好准备，在那里我们将学习 Python 中其他更复杂和专业的工具。特别是在下一章中，我们将讨论 Python 在统计学、科学计算和数据科学领域提供的主要工具和库。

PGM59

MAF28
