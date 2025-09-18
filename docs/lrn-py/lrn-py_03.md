# 第三章。迭代和决策

|   | *"Insanity: doing the same thing over and over again and expecting different results."* |   |
| --- | --- | --- |
|   | --*阿尔伯特·爱因斯坦* |

在上一章中，我们了解了 Python 的内置数据类型。现在你已经熟悉了各种形式和形状的数据，是时候开始了解程序如何使用它了。

根据维基百科：

> *在计算机科学中，控制流（或称为控制流程）指的是指定 imperative 程序中各个语句、指令或函数调用的执行或评估的顺序*。

为了控制程序的流程，我们有两个主要的工具：**条件编程**（也称为**分支**）和**循环**。我们可以用许多不同的组合和变化来使用它们，但在这个章节中，我更愿意先给你讲解基础知识，然后我会和你一起编写几个小脚本。在第一个脚本中，我们将看到如何创建一个基本的素数生成器，而在第二个脚本中，我们将看到如何根据优惠券为客户应用折扣。这样你应该能更好地理解条件编程和循环是如何被使用的。

# 条件编程

条件编程，或者说分支，是我们每天都在做、每时每刻都在做的事情。它涉及到条件的评估：*如果灯是绿色的，那么我可以过马路*，*如果下雨了，那么我会带伞*，*如果我上班迟到了，那么我会给经理打电话*。

主要工具是`if`语句，它有不同的形式和颜色，但基本上它所做的就是评估一个表达式，并根据结果选择执行代码的哪个部分。像往常一样，让我们看一个例子：

`conditional.1.py`

```py
late = True
if late:
    print('I need to call my manager!')
```

这可能是最简单的例子：当输入到`if`语句时，`late`作为一个条件表达式，在布尔上下文中被评估（就像我们调用`bool(late)`一样）。如果评估的结果是`True`，那么我们就立即进入`if`语句之后的代码块。注意，`print`指令是缩进的：这意味着它属于由`if`子句定义的作用域。执行此代码的结果是：

```py
$ python conditional.1.py
I need to call my manager!

```

由于`late`是`True`，所以执行了`print`语句。让我们扩展这个例子：

`conditional.2.py`

```py
late = False
if late:
    print('I need to call my manager!')  #1
else:
    print('no need to call my manager...')  #2
```

这次我设置了`late = False`，所以当我执行代码时，结果就不同了：

```py
$ python conditional.2.py
no need to call my manager...

```

根据评估`late`表达式的结果，我们可以进入块`#1`或块`#2`，*但不能同时进入两个块*。当`late`评估为`True`时，执行块`#1`，而当`late`评估为`False`时，执行块`#2`。尝试将`False/True`值分配给`late`名称，并观察此代码的输出如何相应地改变。

先前的例子还介绍了`else`子句，当我们在`if`子句中想要提供一个当表达式评估为`False`时的替代指令集时，它变得非常有用。`else`子句是可选的，正如通过比较前两个例子所显示的那样。

## 特殊的`else`: `elif`

有时候，只要满足一个条件，你只需要做某件事情（简单的`if`子句）。其他时候，你可能需要提供一个替代方案，以防条件为`False`（`if`/`else`子句），但有时你可能需要从多个路径中选择，所以，既然联系经理（或不联系他们）是一种二进制类型的例子（要么联系要么不联系），那么让我们改变例子类型并继续扩展。这次我们决定税率。如果我的收入低于 10k，我不会缴纳任何税费。如果它在 10k 到 30k 之间，我将缴纳 20%的税费。如果它在 30k 到 100k 之间，我将缴纳 35%的税费，超过 100k，我将（乐意地）缴纳 45%的税费。让我们把这些都写进漂亮的 Python 代码中：

`taxes.py`

```py
income = 15000
if income < 10000:
    tax_coefficient = 0.0  #1
elif income < 30000:
    tax_coefficient = 0.2  #2
elif income < 100000:
    tax_coefficient = 0.35  #3
else:
    tax_coefficient = 0.45  #4

print('I will pay:', income * tax_coefficient, 'in taxes')
```

执行前面的代码会产生：

```py
$ python taxes.py
I will pay: 3000.0 in taxes

```

让我们逐行分析这个例子：我们首先设置收入值。在这个例子中，我的收入是 15k。我们进入`if`子句。注意，这次我们还引入了`elif`子句，它是`else-if`的缩写，它与裸`else`子句不同，因为它也有自己的条件。所以，`if`表达式`income < 10000`评估为`False`，因此块`#1`没有被执行。控制权传递到下一个条件评估器：`elif income < 30000`。这个评估为`True`，因此块`#2`被执行，因此 Python 随后在执行完整个`if`/`elif`/`elif`/`else`子句（从现在起我们可以简单地称之为`if`子句）之后继续执行。`if`子句之后只有一个指令，即`print`调用，它告诉我们今年我将缴纳 3k 的税费（*15k * 20%*）。注意，顺序是强制性的：`if`首先，然后（可选地）需要多少个`elif`，最后（可选地）一个`else`子句。

有趣，对吧？无论每个代码块中有多少行代码，只要其中一个条件评估为`True`，相关的代码块就会被执行，然后执行继续到整个子句之后。如果没有一个条件评估为`True`（例如，`income = 200000`），那么`else`子句的主体就会被执行（块`#4`）。这个例子扩展了我们对于`else`子句行为的理解。它的代码块在先前的`if`/`elif`/.../`elif`表达式都没有评估为`True`时执行。

尝试修改`income`的值，直到你可以随意执行所有代码块（每次执行一个，当然）。然后尝试**边界值**。这非常重要，无论你将条件表达为**等式**或**不等式**（`==`、`!=`、`<`、`>`、`<=`、`>=`），这些数字都代表边界。彻底测试边界是至关重要的。我应该允许你在 18 岁或 17 岁时开车吗？我是用`age < 18`还是`age <= 18`来检查你的年龄？你无法想象我不得不多少次修复由使用错误的运算符引起的微妙错误，所以请继续实验前面的代码。将一些`<`改为`<=`，并将收入设置为边界值之一（10k、30k、100k）以及任何介于这些值之间的值。看看结果如何变化，在继续之前，对它有一个良好的理解。

在我们转到下一个主题之前，让我们看看另一个例子，这个例子展示了如何嵌套`if`语句。假设你的程序遇到一个错误。如果警报系统是控制台，我们打印错误。如果警报系统是电子邮件，我们根据错误的严重性发送它。如果警报系统不是控制台或电子邮件，我们不知道该怎么办，因此我们什么也不做。让我们把这个写成代码：

`errorsalert.py`

```py
alert_system = 'console'  # other value can be 'email'
error_severity = 'critical'  # other values: 'medium' or 'low'
error_message = 'OMG! Something terrible happened!'

if alert_system == 'console':
    print(error_message)  #1
elif alert_system == 'email':
    if error_severity == 'critical':
        send_email('admin@example.com', error_message)  #2
    elif error_severity == 'medium':
        send_email('support.1@example.com', error_message)  #3
    else:
        send_email('support.2@example.com', error_message)  #4
```

前面的例子非常有趣，其荒谬之处在于它向我们展示了两个嵌套的`if`语句（**外层**和**内层**）。它还展示了外层的`if`语句没有`else`部分，而内层有。注意缩进是如何使我们能够在另一个语句内部嵌套一个语句的。

如果`alert_system == 'console'`，则执行`body #1`，其他什么也不发生。另一方面，如果`alert_system == 'email'`，那么我们将进入另一个`if`语句，我们称之为内层。在内层`if`语句中，根据`error_severity`，我们将向管理员、一级支持或二级支持发送电子邮件（`blocks #2`、`#3`和`#4`）。在这个例子中，`send_email`函数没有定义，因此尝试运行它会给你一个错误。在本书的源代码中，你可以从网站上下载，我包括了一个技巧来将那个调用重定向到一个普通的`print`函数，这样你就可以在控制台上进行实验，而不必实际发送电子邮件。尝试更改值，看看它如何工作。

## 三元运算符

在我们继续下一个主题之前，我想向你展示最后一件事，就是**三元运算符**，或者用通俗的话说，是`if`/`else`语句的简短版本。当要根据某个条件为某个名称赋值时，有时使用三元运算符而不是正确的`if`语句更容易、更易读。在下面的例子中，两个代码块做了完全相同的事情：

`ternary.py`

```py
order_total = 247  # GBP

# classic if/else form
if order_total > 100:
    discount = 25  # GBP
else:
    discount = 0  # GBP
print(order_total, discount)

# ternary operator
discount = 25 if order_total > 100 else 0
print(order_total, discount)
```

对于这种简单的情况，我发现能够在一行中表达这种逻辑而不是四行非常好。记住，作为一个程序员，你花在阅读代码上的时间比写作代码的时间要多得多，所以 Python 的简洁性是无价的。

你清楚三元运算符的工作原理吗？基本上是`name = something if condition else something-else`。所以，如果`condition`评估为`True`，则`name`被分配`something`，如果`condition`评估为`False`，则分配`something-else`。

现在你已经了解了控制代码路径的所有内容，让我们继续下一个主题：循环。

# 循环

如果你在其他编程语言中有循环的经验，你会发现 Python 的循环方式略有不同。首先，什么是循环？**循环**意味着能够根据我们给出的循环参数多次重复执行代码块。有不同的循环结构，它们有不同的用途，Python 将它们精简为只有两个，你可以使用它们来实现所需的一切。这些是**for**和**while**语句。

虽然确实可以使用其中任何一个来完成所有需要的功能，但它们有不同的用途，因此通常在不同的上下文中使用。我们将通过本章彻底探讨这种差异。

## `for`循环

当需要遍历序列，如列表、元组或对象集合时，会使用`for`循环。让我们从一个类似于 C++风格的简单示例开始，然后逐步了解如何在 Python 中实现相同的结果（你会喜欢 Python 的语法的）。

`simple.for.py`

```py
for number in [0, 1, 2, 3, 4]:
    print(number)
```

这段简单的代码片段在执行时，会打印出从 0 到 4 的所有数字。`for`循环接收列表`[0, 1, 2, 3, 4]`，并在每次迭代中，`number`变量被赋予序列中的下一个值（按顺序迭代），然后执行循环体（打印行）。`number`变量在每次迭代中都会改变，根据下一个来自序列的值。当序列耗尽时，`for`循环结束，代码的正常执行继续进行循环之后的代码。

### 遍历范围

有时候我们需要遍历一系列数字，如果必须通过硬编码列表来实现，那将会相当不愉快。在这种情况下，`range`函数就派上用场了。让我们看看之前代码片段的等效代码：

`simple.for.py`

```py
for number in range(5):
    print(number)
```

在 Python 程序中，当涉及到创建序列时，`range`函数被广泛使用：你可以通过传递一个值来调用它，这个值作为`stop`（从 0 开始计数），或者你可以传递两个值（`start`和`stop`），甚至三个值（`start`、`stop`和`step`）。查看以下示例：

```py
>>> list(range(10))  # one value: from 0 to value (excluded)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> list(range(3, 8))  # two values: from start to stop (excluded)
[3, 4, 5, 6, 7]
>>> list(range(-10, 10, 4))  # three values: step is added
[-10, -6, -2, 2, 6]

```

目前，忽略我们需要在`list`中包装`range(...)`的事实。`range`对象有点特殊，但在这个例子中，我们只是想了解它将返回给我们哪些值。你看到的情况与切片相同：`start`包含在内，`stop`排除在外，你可以选择添加一个`step`参数，默认值为 1。

尝试修改我们`simple.for.py`代码中`range()`调用的参数，看看它打印了什么，熟悉一下。

### 迭代序列

现在我们有了迭代序列的所有工具，所以让我们在此基础上构建：

`simple.for.2.py`

```py
surnames = ['Rivest', 'Shamir', 'Adleman']
for position in range(len(surnames)):
    print(position, surnames[position])
```

之前的代码给游戏增加了一点点复杂性。执行结果将显示：

```py
$ python simple.for.2.py
0 Rivest
1 Shamir
2 Adleman

```

让我们使用**内外反转**技术来分解它，好吗？我们从我们试图理解的内部最深处开始，然后向外扩展。所以，`len(surnames)`是`surnames`列表的长度：`3`。因此，`range(len(surnames))`实际上被转换成了`range(3)`。这给我们的是范围[0, 3)，这基本上是一个序列`(0, 1, 2)`。这意味着`for`循环将运行三次迭代。在第一次迭代中，`position`将取值`0`，在第二次迭代中，它将取值`1`，最后在第三次和最后一次迭代中取值`2`。`(0, 1, 2)`如果不是`surnames`列表的可能索引位置，那会是什么？在位置`0`我们找到`'Rivest'`，在位置`1`，`'Shamir'`，在位置`2`，`'Adleman'`。如果你对这三位男士共同创造了什么感到好奇，将`print(position, surnames[position])`改为`print(surnames[position][0], end='')`，并在循环外添加一个`print()`，然后再次运行代码。

现在，这种循环风格实际上更接近 Java 或 C++等语言。在 Python 中，很少看到这样的代码。你可以直接迭代任何序列或集合，因此没有必要在每次迭代中获取位置列表并从序列中检索元素。这是昂贵的，不必要的昂贵。让我们将示例改为更 Python 化的形式：

`simple.for.3.py`

```py
surnames = ['Rivest', 'Shamir', 'Adleman']
for surname in surnames:
    print(surname)
```

现在是时候了！这几乎就像是英语。`for`循环可以迭代`surnames`列表，并在每次交互中按顺序返回每个元素。运行此代码将逐个打印出三个姓氏。它更容易阅读，对吧？

如果你想要打印位置怎么办？或者，如果你实际上需要它用于任何原因，你应该回到`range(len(...))`的形式吗？不。你可以使用内置的`enumerate`函数，如下所示：

`simple.for.4.py`

```py
surnames = ['Rivest', 'Shamir', 'Adleman']
for position, surname in enumerate(surnames):
    print(position, surname)
```

这段代码也非常有趣。注意，`enumerate`在每次迭代中返回一个 2 元组`(position, surname)`，但即便如此，它仍然比`range(len(...))`示例更易读（并且更高效）。你可以用`start`参数调用`enumerate`，例如`enumerate(iterable, start)`，它将从`start`开始，而不是从`0`开始。这只是另一个小细节，展示了在 Python 的设计中投入了多少思考，以便让生活变得更简单。

使用`for`循环可以迭代列表、元组，以及 Python 中称为可迭代对象的一切。这是一个非常重要的概念，所以让我们多谈谈它。

## 迭代器和可迭代对象

根据 Python 文档，一个可迭代对象是：

> *“一个能够一次返回其成员的对象。可迭代的例子包括所有序列类型（如 `list`、`str` 和 `tuple`）以及一些非序列类型，如 `dict`、`file` 对象，以及任何具有 `__iter__()` 或 `__getitem__()` 方法的类定义的对象。可迭代的可以在 `for` 循环和许多其他需要序列的地方使用（`zip()`、`map()` 等）。当一个可迭代对象作为参数传递给内置函数 `iter()` 时，它返回该对象的一个迭代器。这个迭代器适用于对值集进行一次遍历。在使用可迭代对象时，通常不需要调用 `iter()` 或自己处理迭代器对象。`for` 语句会为你自动完成这些操作，创建一个临时未命名的变量来在循环期间持有迭代器。”*

简单来说，当你编写 `for k in sequence: ... body ...` 时发生的情况是，`for` 循环会向 `sequence` 请求下一个元素，它得到一些东西，并将其称为 `k`，然后执行其主体。然后，再次，`for` 循环再次向 `sequence` 请求下一个元素，再次将其称为 `k`，并再次执行主体，以此类推，直到序列耗尽。空序列将导致主体执行零次。

一些数据结构在迭代时按顺序产生它们的元素，如列表、元组和字符串，而另一些则不会，如集合和字典。

Python 给我们提供了使用称为 **迭代器** 的对象来迭代可迭代对象的能力。根据官方文档，迭代器是：

> *“一个表示数据流的对象。对迭代器的 `__next__()` 方法（或将其传递给内置函数 `next()`）的重复调用将返回数据流中的连续项。当没有更多数据可用时，将引发 `StopIteration` 异常。此时，迭代器对象已耗尽，对其 `__next__()` 方法的任何进一步调用都将再次引发 `StopIteration`。迭代器必须有一个 `__iter__()` 方法，该方法返回迭代器对象本身，因此每个迭代器也是可迭代的，可以在接受其他可迭代对象的大多数地方使用。一个值得注意的例外是尝试多次迭代遍历的代码。容器对象（如 `list`）每次你将其传递给 `iter()` 函数或用于 `for` 循环时，都会产生一个新的迭代器。尝试使用迭代器这样做只会返回在先前迭代遍历中使用的同一个耗尽的迭代器对象，使其看起来像一个空容器。”*

如果你对前面的法律术语不完全理解，不要担心，你会在适当的时候理解。我把它们放在这里，作为未来方便的参考。

在实践中，整个可迭代/迭代器机制在代码背后是有些隐藏的。除非你出于某种原因需要自己编写可迭代或迭代器，否则你不必太担心这个问题。但是，理解 Python 如何处理这个关键的控制流方面非常重要，因为它将塑造你编写代码的方式。

## 遍历多个序列

让我们看看另一个例子，说明如何迭代两个长度相同的序列，以便成对地处理它们的各自元素。比如说，我们有一个包含人的列表和一个表示第一个列表中人的年龄的数字列表。我们想要打印出所有人的姓名/年龄对。让我们从一个例子开始，然后逐步改进它。

`multiple.sequences.py`

```py
people = ['Jonas', 'Julio', 'Mike', 'Mez']
ages = [25, 30, 31, 39]
for position in range(len(people)):
    person = people[position]
    age = ages[position]
    print(person, age)
```

到现在为止，你应该已经能够理解这段代码了。我们需要遍历位置列表（0，1，2，3），因为我们想要从两个不同的列表中检索元素。执行它我们得到以下结果：

```py
$ python multiple.sequences.py
Jonas 25
Julio 30
Mike 31
Mez 39

```

这段代码既低效又不符合 Python 风格。低效是因为给定位置检索元素可能是一个昂贵的操作，而且我们每次迭代都是从零开始做的。邮递员每次送信时不会回到路的起点，对吧？他是从一栋房子走到另一栋。让我们尝试使用 enumerate 来让它变得更好：

`multiple.sequences.enumerate.py`

```py
people = ['Jonas', 'Julio', 'Mike', 'Mez']
ages = [25, 30, 31, 39]
for position, person in enumerate(people):
    age = ages[position]
    print(person, age)
```

更好，但仍然不完美。而且仍然有点丑。我们正确地迭代了`people`，但我们仍然使用位置索引来获取`age`，这是我们想要丢弃的。嗯，不用担心，Python 给你提供了`zip`函数，记得吗？让我们来用它！

`multiple.sequences.zip.py`

```py
people = ['Jonas', 'Julio', 'Mike', 'Mez']
ages = [25, 30, 31, 39]
for person, age in zip(people, ages):
    print(person, age)
```

啊！好多了！再次比较前面的代码和第一个例子，欣赏一下 Python 的优雅。我想展示这个例子的原因有两个。一方面，我想让你了解 Python 的代码相比其他语言可以多么简短，在其他语言中，语法不允许你像在 Python 中那样轻松地对序列或集合进行迭代。另一方面，更重要的是，注意当`for`循环请求`zip(sequenceA, sequenceB)`的下一个元素时，它返回的是一个`tuple`，而不仅仅是一个单一的对象。它返回一个`tuple`，其中的元素数量与我们提供给`zip`函数的序列数量相同。让我们从两个方面对前面的例子进行扩展：使用显式和隐式赋值：

`multiple.sequences.explicit.py`

```py
people = ['Jonas', 'Julio', 'Mike', 'Mez']
ages = [25, 30, 31, 39]
nationalities = ['Belgium', 'Spain', 'England', 'Bangladesh']
for person, age, nationality in zip(people, ages, nationalities):
    print(person, age, nationality)
```

在前面的代码中，我们添加了国籍列表。现在我们向`zip`函数提供了三个序列，for 循环在每次迭代时都会返回一个*3-tuple*。注意，元组中元素的位置与`zip`调用中序列的位置相匹配。执行代码将产生以下结果：

```py
$ python multiple.sequences.explicit.py
Jonas 25 Belgium
Julio 30 Spain
Mike 31 England
Mez 39 Bangladesh

```

有时候，可能由于前面的简单示例中不明显的原因，你可能想在`for`循环体内部展开元组。如果你有这个愿望，这样做是完全可能的。

`multiple.sequences.implicit.py`

```py
people = ['Jonas', 'Julio', 'Mike', 'Mez']
ages = [25, 30, 31, 39]
nationalities = ['Belgium', 'Spain', 'England', 'Bangladesh']
for data in zip(people, ages, nationalities):
    person, age, nationality = data
    print(person, age, nationality)
```

它基本上是在做`for`循环自动为你做的事情，但在某些情况下，你可能想自己来做。在这里，来自`zip(...)`的 3 元组`data`在`for`循环体内部展开成三个变量：`person`、`age`和`nationality`。

## `while`循环

在前面的页面中，我们看到了`for`循环的实际应用。当你需要遍历一个序列或集合时，它非常有用。当你需要能够区分使用哪种循环结构时，需要记住的关键点是，当你需要遍历有限数量的元素时，`for`循环非常出色。它可以是非常大的数量，但仍然，在某一点上会结束。

然而，还有其他情况，你可能只需要循环直到满足某个条件，或者无限循环直到应用程序停止。在这些情况下，我们实际上没有可以迭代的，因此`for`循环可能不是一个好的选择。但不要担心，对于这些情况，Python 为我们提供了`while`循环。

`while`循环与`for`循环类似，因为它们都循环，并且在每次迭代中执行一组指令。它们之间的不同之处在于，`while`循环不是遍历一个序列（它可以，但你需要手动编写逻辑，这不会有什么意义，你只会想使用`for`循环），而是只要满足某个条件就循环。当条件不再满足时，循环结束。

如同往常，让我们看一个例子，这个例子将为我们澄清一切。我们想要打印一个正数的二进制表示。为了做到这一点，我们反复将数字除以 2，收集余数，然后生成余数列表的逆序。让我给你一个使用数字 6 的小例子，它在二进制中表示为 110。

```py
6 / 2 = 3 (remainder: 0)
3 / 2 = 1 (remainder: 1)
1 / 2 = 0 (remainder: 1)
List of remainders: 0, 1, 1.
Inverse is 1, 1, 0, which is also the binary representation of 6: 110
```

让我们编写一些代码来计算数字 39 的二进制表示：100111[2]。

`binary.py`

```py
n = 39
remainders = []
while n > 0:
    remainder = n % 2  # remainder of division by 2
    remainders.append(remainder)  # we keep track of remainders
    n //= 2  # we divide n by 2

# reassign the list to its reversed copy and print it
remainders = remainders[::-1]
print(remainders)
```

在前面的代码中，我强调了两个要点：`n > 0`，这是保持循环的条件，以及`remainders[::-1]`，这是一个获取列表反转的好方法（缺少`start`和`end`参数，`step = -1`会产生相同的列表，从`end`到`start`，以相反的顺序）。我们可以通过使用`divmod`函数使代码更短（并且更 Pythonic），该函数用一个数和一个除数调用，并返回一个包含整数除法结果及其余数的元组。例如，`divmod(13, 5)`将返回`(2, 3)`，确实*5 * 2 + 3 = 13*。

`binary.2.py`

```py
n = 39
remainders = []
while n > 0:
    n, remainder = divmod(n, 2)
    remainders.append(remainder)

# reassign the list to its reversed copy and print it
remainders = remainders[::-1]
print(remainders)
```

在前面的代码中，我们将`n`重新赋值为除以 2 的结果，以及余数，都在一行中完成。

注意，`while`循环中的条件是一个继续循环的条件。如果它评估为`True`，则执行主体，然后进行另一次评估，依此类推，直到条件评估为`False`。当这种情况发生时，循环会立即退出，而不会执行其主体。

### 注意

如果条件永远不会评估为`False`，循环就变成了所谓的**无限循环**。无限循环在例如从网络设备轮询时使用：你询问套接字是否有数据，如果有，你处理它，然后你睡一小会儿，然后你再次询问套接字，一次又一次，永不停止。

能够根据条件循环或无限循环，是为什么仅仅`for`循环不足以满足需求，因此 Python 提供了`while`循环。

### 小贴士

顺便说一句，如果你需要数字的二进制表示，可以查看`bin`函数。

只为了好玩，让我们使用 while 逻辑修改一个例子（`multiple.sequences.py`）。

`multiple.sequences.while.py`

```py
people = ['Jonas', 'Julio', 'Mike', 'Mez']
ages = [25, 30, 31, 39]
position = 0
while position < len(people):
    person = people[position]
    age = ages[position]
    print(person, age)
    position += 1

```

在前面的代码中，我已经突出显示了变量`position`的*初始化*、*条件*和*更新*，这使得通过手动处理迭代变量来模拟等效的`for`循环代码成为可能。你可以用`for`循环完成的事情，也可以用`while`循环完成，尽管你可以看到为了达到相同的结果，你必须经历一些样板代码。相反的情况也是真实的，但是使用`for`循环模拟一个永不结束的`while`循环需要一些真正的技巧，所以你为什么要这样做呢？使用适合的工具，99.9%的情况下你会没事的。

因此，总结一下，当你需要遍历一个（或多个）可迭代对象时，使用`for`循环；当你需要根据条件是否满足来循环时，使用`while`循环。如果你记住这两个目的之间的区别，你就永远不会选择错误的循环结构。

让我们看看如何改变循环的正常流程。

## `break`和`continue`语句

根据当前任务，有时你可能需要改变循环的正常流程。你可以跳过单个迭代（多次你想跳过的次数），或者你可以完全跳出循环。跳过迭代的常见用例，例如当你遍历一个项目列表，并且只有当某些条件得到验证时，你才需要处理每个项目。另一方面，如果你正在遍历一个项目集合，并且你已经找到了满足你需求的一个项目，你可能会决定不继续整个循环，因此跳出它。可能存在无数种可能的场景，所以最好看看几个例子。

假设你想要对购物车列表中今天过期的所有产品应用 20%的折扣。你实现这一点的办法是使用 **continue** 语句，它告诉循环结构（`for` 或 `while`）立即停止执行体并转到下一次迭代，如果有的话。这个例子将带我们深入到兔子洞中，所以请准备好跳跃。

`discount.py`

```py
from datetime import date, timedelta

today = date.today()
tomorrow = today + timedelta(days=1)  # today + 1 day is tomorrow
products = [
    {'sku': '1', 'expiration_date': today, 'price': 100.0},
    {'sku': '2', 'expiration_date': tomorrow, 'price': 50},
    {'sku': '3', 'expiration_date': today, 'price': 20},
]
for product in products:
    if product['expiration_date'] != today:
        continue
    product['price'] *= 0.8  # equivalent to applying 20% discount
    print(
        'Price for sku', product['sku'],
        'is now', product['price'])
```

你可以看到，我们首先导入 `date` 和 `timedelta` 对象，然后设置我们的产品。那些 sku 为 `1` 和 `3` 的产品有 `today` 的过期日期，这意味着我们想要对它们应用 20%的折扣。我们遍历每个 `product` 并检查过期日期。如果它不是（不等号操作符，`!=`）`today`，我们不想执行其余的代码块，所以使用 `continue`。

注意，`continue` 语句在代码块中的位置并不重要（你甚至可以使用它多次）。当你到达它时，执行停止并回到下一次迭代。如果我们运行 `discount.py` 模块，这是输出结果：

```py
$ python discount.py
Price for sku 1 is now 80.0
Price for sku 3 is now 16.0

```

这表明 sku 编号 2 的代码块的最后两行没有被执行。

现在让我们看看如何从循环中退出的一个例子。假设我们想要判断列表中的至少一个元素在传递给 `bool` 函数时评估为 `True`。鉴于我们需要知道是否至少有一个，当我们找到它时，我们不需要继续扫描列表。在 Python 代码中，这相当于使用 **break** 语句。让我们把这个写下来：

`any.py`

```py
items = [0, None, 0.0, True, 0, 7]  # True and 7 evaluate to True
found = False  # this is called "flag"
for item in items:
    print('scanning item', item)
    if item:
        found = True  # we update the flag
        break

if found:  # we inspect the flag
    print('At least one item evaluates to True')
else:
    print('All items evaluate to False')
```

上述代码是编程中非常常见的模式，你将经常看到它。当你这样检查项目时，基本上你做的是设置一个 `flag` 变量，然后开始检查。如果你找到一个符合你标准（在这个例子中，评估为 `True`）的元素，那么你更新标志并停止迭代。迭代后，你检查标志并相应地采取行动。执行结果如下：

```py
$ python any.py
scanning item 0
scanning item None
scanning item 0.0
scanning item True
At least one item evaluates to True

```

你可以看到，在找到 `True` 后执行停止了吗？

`break` 语句的行为与 `continue` 语句完全相同，即它立即停止执行循环体，但同时也阻止任何其他迭代运行，有效地退出循环。

`continue` 和 `break` 语句可以与 `for` 和 `while` 循环结构一起使用，数量没有限制。

### 小贴士

顺便说一句，检测一个序列中是否至少有一个元素评估为 `True` 并不需要编写代码。只需查看内置的 `any` 函数即可。

## 一个特殊的 `else` 子句

我只在 Python 语言中看到的一个特性是能够在`while`和`for`循环之后有`else`子句。这很少被使用，但确实很好。简而言之，你可以在`for`或`while`循环之后有一个`else`子句。如果循环正常结束，因为迭代器耗尽（`for`循环）或者因为条件最终不满足（`while`循环），那么（如果存在）`else`子句将被执行。如果执行被`break`语句中断，则`else`子句不会执行。让我们举一个`for`循环的例子，它遍历一组项目，寻找符合某些条件的项目。如果我们找不到至少一个满足条件的项目，我们想要抛出一个**异常**。这意味着我们想要阻止程序的常规执行，并发出信号，表示出现了我们无法处理的错误或异常。异常将是第七章的主题，*测试、分析和处理异常*，所以如果你现在不完全理解它们，不要担心。只需记住，它们将改变代码的常规流程。现在让我给你展示两个做同样事情的例子，其中一个使用了特殊的`for` ... `else`语法。假设我们想在人群中找到一个能开车的人。

`for.no.else.py`

```py
class DriverException(Exception):
    pass

people = [('James', 17), ('Kirk', 9), ('Lars', 13), ('Robert', 8)]
driver = None
for person, age in people:
    if age >= 18:
        driver = (person, age)
 break

if driver is None:
    raise DriverException('Driver not found.')
```

再次注意`flag`模式。我们将`driver`设置为`None`，然后如果我们找到它，我们更新`driver`标志，然后在循环结束时检查它是否被找到。我有一种感觉，那些孩子会开一辆非常金属的车，但无论如何，注意如果找不到司机，将抛出一个`DriverException`，向程序发出信号，表示无法继续执行（我们缺少司机）。

同样的功能可以用以下代码以更优雅的方式重写：

`for.else.py`

```py
class DriverException(Exception):
    pass

people = [('James', 17), ('Kirk', 9), ('Lars', 13), ('Robert', 8)]
for person, age in people:
    if age >= 18:
        driver = (person, age)
        break
else:
    raise DriverException('Driver not found.')
```

注意，我们不再被迫使用`flag`模式。异常作为`for`循环逻辑的一部分被抛出，这很有意义，因为`for`循环正在检查某个条件。我们只需要设置一个`driver`对象，以防我们找到它，因为代码的其余部分将使用这些信息。注意代码变得更短、更优雅，因为逻辑现在被正确地组合在一起。

# 将这些放在一起

现在你已经看到了关于条件和循环的所有内容，是时候让事情变得有趣一些，看看我在本章开头提到的两个例子。我们将混合使用，这样你可以看到如何将这些概念结合起来。让我们先编写一些代码来生成一个到某个限制的质数列表。请记住，我将编写一个非常低效和原始的算法来检测质数。对你来说，重要的是要专注于代码中属于本章主题的部分。

## 示例 1 – 一个素数生成器

根据维基百科：

> *"素数（或素数）是一个大于 1 的自然数，它除了 1 和它本身外没有其他正除数。一个大于 1 但不是素数的自然数称为合数。"*

根据这个定义，如果我们考虑前 10 个自然数，我们可以看到 2、3、5 和 7 是素数，而 1、4、6、8、9、10 则不是。为了让计算机告诉你一个数 *N* 是否是素数，你可以将这个数除以 [2, *N*) 范围内的所有自然数。如果这些除法中的任何一个产生了余数为零，那么这个数就不是素数。说够了，让我们开始工作。我会写两个版本，第二个版本将利用 `for` ... `else` 语法。

`primes.py`

```py
primes = []  # this will contain the primes in the end
upto = 100  # the limit, inclusive
for n in range(2, upto + 1):
    is_prime = True  # flag, new at each iteration of outer for
    for divisor in range(2, n):
        if n % divisor == 0:
            is_prime = False
 break
    if is_prime:  # check on flag
        primes.append(n)
print(primes)
```

在前面的代码中有很多需要注意的地方。首先，我们设置了一个空列表 `primes`，它将包含最后的素数。限制是 100，你可以看到我们在外层循环中调用 `range()` 的方式是包含的。如果我们写成 `range(2, upto)`，那么就是 [2, upto)，对吗？因此 `range(2, upto + 1)` 给我们 *[2, upto + 1) == [2, upto]*。

因此，两个 `for` 循环。在外层循环中，我们遍历候选素数，即从 2 到 `upto` 的所有自然数。在每个外层循环的迭代中，我们设置一个标志（每个迭代都设置为 `True`），然后开始将当前的 `n` 除以从 2 到 *n* – 1 的所有数字。如果我们找到了 `n` 的一个合适的除数，这意味着 `n` 是合数，因此我们将标志设置为 `False` 并跳出循环。注意，当我们跳出内层循环时，外层循环会正常继续。我们找到 `n` 的一个合适的除数后跳出循环的原因是，我们不需要任何更多的信息就能判断 `n` 不是素数。

当我们检查 `is_prime` 标志时，如果它仍然是 `True`，这意味着我们在 [2, *n*) 范围内没有找到任何是 `n` 的合适除数的数字，因此 `n` 是素数。我们将 `n` 添加到 `primes` 列表中，然后！另一个迭代，直到 *n* 等于 100。

运行此代码会产生：

```py
$ python primes.py
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

```

在我们继续之前，有一个问题：在外层循环的所有迭代中，有一个与其他所有迭代不同。你能告诉我哪一个，为什么吗？想一下，回到代码中，试着自己找出答案，然后再继续阅读。

你弄懂了吗？如果没有，别难过，这是完全正常的。我让你做这个小练习是因为这就是程序员一直做的事情。通过仅仅看代码就能理解代码的功能的技能是你在时间中逐渐培养出来的。这非常重要，所以尽量在可能的情况下练习它。我现在告诉你答案：表现与其他迭代不同的迭代是第一个。原因是第一个迭代中 `n` 是 2。因此最内层的 `for` 循环甚至不会运行，因为它是一个迭代 `range(2, 2)` 的 `for` 循环，而这如果不是 [2, 2) 又是什么呢？自己试试，写一个简单的 `for` 循环，用那个可迭代对象，在主体中放一个 `print`，看看是否会发生什么（不会...）。

现在，从算法的角度来看，这段代码效率不高，所以至少让我们让它更美观一些：

`primes.else.py`

```py
primes = []
upto = 100
for n in range(2, upto + 1):
    for divisor in range(2, n):
        if n % divisor == 0:
            break
    else:
 primes.append(n)
print(primes)
```

更好了，对吧？`is_prime` 标志已经完全消失，当我们知道内部 `for` 循环没有遇到任何 `break` 语句时，我们会将 `n` 添加到 `primes` 列表中。看看代码看起来更整洁，读起来更好吗？

## 示例 2 – 应用折扣

在这个例子中，我想向你展示我非常喜欢的一种技术。在许多编程语言中，除了 `if`/`elif`/`else` 构造之外，无论它们以何种形式或语法出现，你都可以找到一个称为 `switch`/`case` 的另一个语句，在 Python 中这是缺失的。它相当于一系列的 `if`/`elif`/.../`elif`/`else` 子句，其语法类似于这样（警告！JavaScript 代码！）：

`switch.js`

```py
switch (day_number) {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
        day = "Weekday";
        break;
    case 6:
        day = "Saturday";
        break;
    case 0:
        day = "Sunday";
        break;
    default:
        day = "";
        alert(day_number + ' is not a valid day number.')
}
```

在前面的代码中，我们根据一个名为 `day_number` 的变量进行 `switch` 操作。这意味着我们得到它的值，然后我们决定它适合哪个情况（如果有的话）。从 1 到 5 是一个级联，这意味着无论数字是什么，[1, 5] 都会下降到设置 `day` 为 `"Weekday"` 的逻辑部分。然后我们有针对 0 和 6 的单个情况，以及一个 `default` 情况来防止错误，这会通知系统 `day_number` 不是一个有效的天数，也就是说，不在 [0, 6] 范围内。Python 完全可以使用 `if`/`elif`/`else` 语句实现这样的逻辑：

`switch.py`

```py
if 1 <= day_number <= 5:
    day = 'Weekday'
elif day_number == 6:
    day = 'Saturday'
elif day_number == 0:
    day = 'Sunday'
else:
    day = ''
    raise ValueError(
        str(day_number) + ' is not a valid day number.')
```

在前面的代码中，我们使用 `if`/`elif`/`else` 语句在 Python 中重现了 JavaScript 片段的相同逻辑。我在最后仅仅作为一个例子抛出了 `ValueError` 异常，如果 `day_number` 不在 [0, 6] 范围内。这是翻译 `switch`/`case` 逻辑的一种可能方式，但还有一种方式，有时被称为分派，我将在下一个示例的最后版本中向你展示。

### 小贴士

顺便问一下，你注意到前面片段的第一行了吗？你注意到 Python 可以进行双重（实际上，甚至是多重的）比较吗？这真是太棒了！

让我们从编写一些代码开始新的示例，这些代码根据客户的优惠券价值给客户分配折扣。我会尽量简化这里的逻辑，记住我们真正关心的是条件和循环。

`coupons.py`

```py
customers = [
    dict(id=1, total=200, coupon_code='F20'),  # F20: fixed, £20
    dict(id=2, total=150, coupon_code='P30'),  # P30: percent, 30%
    dict(id=3, total=100, coupon_code='P50'),  # P50: percent, 50%
    dict(id=4, total=110, coupon_code='F15'),  # F15: fixed, £15
]
for customer in customers:
    code = customer['coupon_code']
    if code == 'F20':
        customer['discount'] = 20.0
    elif code == 'F15':
        customer['discount'] = 15.0
    elif code == 'P30':
        customer['discount'] = customer['total'] * 0.3
    elif code == 'P50':
        customer['discount'] = customer['total'] * 0.5
    else:
        customer['discount'] = 0.0

for customer in customers:
    print(customer['id'], customer['total'], customer['discount'])
```

我们首先设置一些客户。他们有一个订单总额、一个优惠券代码和一个 ID。我编造了四种不同的优惠券类型，两种是固定的，两种是百分比基础的。您可以看到，在`if`/`elif`/`else`级联中，我相应地应用了折扣，并将其设置为`customer`字典中的`'discount'`键。

最后，我只是打印出部分数据来查看我的代码是否正常工作。

```py
$ python coupons.py
1 200 20.0
2 150 45.0
3 100 50.0
4 110 15.0

```

这段代码很容易理解，但所有这些条款都使得逻辑显得有些杂乱。一开始看不太清楚发生了什么，我不喜欢这样。在这种情况下，您可以利用字典的优势，如下所示：

`coupons.dict.py`

```py
customers = [
    dict(id=1, total=200, coupon_code='F20'),  # F20: fixed, £20
    dict(id=2, total=150, coupon_code='P30'),  # P30: percent, 30%
    dict(id=3, total=100, coupon_code='P50'),  # P50: percent, 50%
    dict(id=4, total=110, coupon_code='F15'),  # F15: fixed, £15
]
discounts = {
    'F20': (0.0, 20.0),  # each value is (percent, fixed)
    'P30': (0.3, 0.0),
    'P50': (0.5, 0.0),
    'F15': (0.0, 15.0),
}
for customer in customers:
    code = customer['coupon_code']
    percent, fixed = discounts.get(code, (0.0, 0.0))
    customer['discount'] = percent * customer['total'] + fixed

for customer in customers:
    print(customer['id'], customer['total'], customer['discount'])
```

运行前面的代码会产生与之前代码片段完全相同的结果。我们节省了两行代码，但更重要的是，我们在可读性上取得了很大的进步，因为`for`循环的主体现在只有三行长，非常容易理解。这里的思路是使用一个字典作为**调度器**。换句话说，我们尝试根据一个代码（我们的`coupon_code`）从字典中获取一些内容，通过使用`dict.get(key, default)`，我们确保当`code`不在字典中且需要默认值时也能处理。

注意，我必须应用一些非常简单的线性代数来正确计算折扣。每个折扣在字典中都有一个百分比和固定部分，由一个二元组表示。通过应用`percent * total + fixed`，我们得到正确的折扣。当`percent`为`0`时，公式只给出固定金额，当固定为`0`时，它给出`percent * total`。简单但有效。

这种技术很重要，因为它还用于其他上下文中，比如函数，在那里它实际上比我们在前面的代码片段中看到的功能要强大得多。如果您对它的工作原理还不完全清楚，我建议您花点时间实验一下。改变值并添加打印语句，以查看程序运行时的状态。

# 快速浏览`itertools`模块

在关于可迭代对象、迭代器、条件逻辑和循环的章节中，如果不花点时间谈谈`itertools`模块，那就不是一个完整的章节。如果您喜欢迭代，这将是一个天堂。

根据 Python 官方文档，`itertools`模块是：

> *"一个模块，它实现了一系列由 APL、Haskell 和 SML 中的构造灵感的迭代器构建块。每个都已被重新塑造成适合 Python 的形式。该模块标准化了一组快速、内存高效的工具，这些工具本身或组合使用都很有用。它们共同形成了一个“迭代器代数”，使得在纯 Python 中简洁且高效地构建专用工具成为可能。"*

在这里，我绝对没有足够的空间向您展示这个模块中您可以找到的所有好东西，所以我鼓励您自己去看看，我保证您会喜欢的。

简而言之，它为你提供了三种广泛的迭代器类别。我将给出一个例子，从每个类别中取一个迭代器，只是为了让你稍微有点兴趣。

## 无限迭代器

无限迭代器允许你以不同的方式使用`for`循环，就像它是`while`循环一样。

`infinite.py`

```py
from itertools import count
for n in count(5, 3):
    if n > 20:
        break
    print(n, end=', ')  # instead of newline, comma and space
```

运行代码得到以下结果：

```py
$ python infinite.py
5, 8, 11, 14, 17, 20,

```

`count`工厂类创建一个不断计数并继续的迭代器。它从 5 开始，并持续加 3。如果我们不希望陷入无限循环，我们需要手动中断它。

## 输入序列最短时终止的迭代器

这个类别非常有趣。它允许你基于多个迭代器创建一个迭代器，根据某些逻辑组合它们的值。关键点在于，在这些迭代器中，如果其中任何一个比其他的长，结果迭代器不会断裂，它会在最短的迭代器耗尽时简单地停止。这非常理论化，我知道，所以让我给你举一个使用`compress`的例子。这个迭代器根据选择器中相应项的`True`或`False`返回数据：

`compress('ABC', (1, 0, 1))`会返回`'A'`和`'C'`，因为它们对应于`1's`。让我们看看一个简单的例子：

`compress.py`

```py
from itertools import compress
data = range(10)
even_selector = [1, 0] * 10
odd_selector = [0, 1] * 10

even_numbers = list(compress(data, even_selector))
odd_numbers = list(compress(data, odd_selector))

print(odd_selector)
print(list(data))
print(even_numbers)
print(odd_numbers)
```

注意，`odd_selector`和`even_selector`长度为 20 个元素，而`data`只有 10 个元素长。`compress`会在`data`产生最后一个元素时停止。运行此代码会产生以下结果：

```py
$ python compress.py
[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[0, 2, 4, 6, 8]
[1, 3, 5, 7, 9]

```

这是一个从可迭代对象中选择元素非常快且很棒的方法。代码非常简单，只需注意，我们不是使用`for`循环遍历由`compress`调用返回的每个值，而是使用了`list()`，它做的是同样的事情，但不同的是，它不是执行一系列指令，而是将所有值放入列表中并返回它。

## 组合生成器

最后但同样重要的是，组合生成器。如果你对这类东西感兴趣，这真的很有趣。让我们看看排列的一个简单例子。

根据 Wolfram Mathworld：

> *"排列，也称为“排列数”或“顺序”，是将有序列表 S 的元素重新排列，使其与 S 本身形成一一对应关系。"*

例如，ABC 的排列有 6 个：ABC、ACB、BAC、BCA、CAB 和 CBA。

如果一个集合有*N*个元素，那么它们的排列数是*N!*（*N*的阶乘）。对于字符串 ABC，排列是*3! = 3 * 2 * 1 = 6*。让我们用 Python 来做：

`permutations.py`

```py
from itertools import permutations
print(list(permutations('ABC')))
```

这段非常短的代码片段产生了以下结果：

```py
$ python permutations.py
[('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'), ('B', 'C', 'A'), ('C', 'A', 'B'), ('C', 'B', 'A')]

```

当你玩排列时，一定要非常小心。它们的数量以与你要排列的元素数量的阶乘成比例的速度增长，而这个数字可以变得非常大，非常快。

# 摘要

在本章中，我们进一步扩展了我们的编码词汇。我们看到了如何通过评估条件来驱动代码的执行，我们也看到了如何循环遍历对象序列和集合。这给了我们控制代码运行时发生什么的能力，这意味着我们正在获得如何塑造它以实现我们想要的功能，并对其动态变化的数据做出反应的想法。

我们也看到了如何在几个简单的例子中将所有东西结合起来，最后我们还简要地浏览了`itertools`模块，它充满了有趣的迭代器，可以进一步丰富我们在 Python 中的能力。

现在是时候转换方向，迈出另一步，来谈谈函数了。下一章全部都是关于它们的，因为它们极其重要。确保你对到目前为止所做的一切感到舒适：我想给你提供一些有趣的例子，所以我需要稍微加快一点速度。准备好了吗？翻到下一页。
