# 5

# 理解和生成器

> “不是每天的增量，而是每天的减量。砍掉不必要的东西。”
> 
> ——李小龙

上述引言的第二部分，“砍掉不必要的东西”，对我们来说正是使计算机程序优雅的原因。我们不断努力寻找更好的做事方式，以便我们不浪费时间或内存。

有一些合理的理由不将我们的代码推到最大极限。例如，有时我们不得不牺牲可读性或可维护性以换取微小的改进。当我们可以用可读、干净的代码在 1.05 秒内提供服务时，用难以阅读、复杂的代码在 1 秒内提供服务是没有意义的。

另一方面，有时尝试从函数中节省一毫秒是完全合理的，特别是当函数打算被调用数千次时。在数千次调用中节省的一毫秒可以累积成秒，这可能会对你的应用程序有意义。

考虑到这些因素，本章的重点不是给你工具来让你的代码无论什么情况下都能达到性能和优化的绝对极限，而是让你能够编写高效、优雅的代码，易于阅读，运行速度快，并且不会以明显的方式浪费资源。

在本章中，我们将涵盖以下内容：

+   `map()`、`zip()`和`filter()`函数

+   理解

+   生成器

+   性能

我们将进行多次测量和比较，并谨慎地得出一些结论。请务必记住，在不同的机器、不同的设置或操作系统上，结果可能会有所不同。

看看这段代码：

```py
# squares.py
def square1(n):
    return n**2  # squaring through the power operator
def square2(n):
    return n * n  # squaring through multiplication 
```

这两个函数都返回`n`的平方，但哪个更快？从我们运行的一个简单基准测试来看，第二个似乎稍微快一点。如果你这么想，这是有道理的：计算一个数的幂涉及乘法。因此，无论你使用什么算法来执行幂运算，都不太可能打败`square2`中的简单乘法。

我们关心这个结果吗？在大多数情况下，不关心。如果你正在编写一个电子商务网站，你很可能永远不需要将一个数字提高到平方，即使你这样做，也很可能是偶尔的操作。你不需要担心在调用几次函数时节省几分之一微秒。

那么，何时优化变得重要呢？一个常见的例子是当你必须处理大量数据时。如果你要对一百万个`customer`对象应用相同的函数，那么你希望你的函数调优到最佳状态。对一个被调用一百万次的函数节省十分之一秒可以节省 100,000 秒，这大约是 27.7 小时。因此，让我们关注集合，看看 Python 为你提供了哪些工具来高效、优雅地处理它们。

本章中我们将看到的大多数概念都是基于迭代器和可迭代对象，我们在 *第三章* ，*条件语句和迭代* 中遇到了它们。我们将看到如何在 *第六章* ，*面向对象编程、装饰器和迭代器* 中编码自定义迭代器和可迭代对象。

本章我们将探索的一些对象是迭代器，它们通过一次只操作集合中的一个元素来节省内存，而不是创建一个修改后的副本。因此，如果我们只想显示操作的结果，就需要做一些额外的工作。我们通常会求助于将迭代器包裹在 `list()` 构造函数中。这是因为将迭代器传递给 `list()` 会耗尽它，并将所有生成的项放入一个新创建的列表中，我们可以轻松地打印出来以显示其内容。让我们看看在 `range` 对象上使用该技术的例子：

```py
# list.iterable.txt
>>> range(7)
**range****(****0****,** **7****)**
>>> list(range(7))  # put all elements in a list to view them
[0, 1, 2, 3, 4, 5, 6] 
```

我们已经突出显示了在 Python 控制台中输入 `range(7)` 的结果。注意，它并没有显示 `range` 的内容，因为 `range` 从来不会实际将整个数字序列加载到内存中。第二个突出显示的行显示了将 `range` 包裹在 `list()` 中是如何使我们能够看到它生成的数字的。

让我们开始查看 Python 为高效操作数据集合提供的各种工具。

# `map`、`zip` 和 `filter` 函数

我们将首先回顾 `map()`、`filter()` 和 `zip()`，这些是处理集合时可以使用的内置函数的主要函数，然后我们将学习如何使用两个重要的结构：**列表推导式**和**生成器**来实现相同的结果。

## `map`

根据 Python 的官方文档（[`docs.python.org/3/library/functions.html#map`](https://docs.python.org/3/library/functions.html#map)），以下内容是正确的：

> `map(function, iterable, *iterables)`
> 
> 返回一个迭代器，它将函数应用于可迭代对象的每个元素，并产生结果。如果传递了额外的可迭代参数，函数必须接受那么多参数，并且将并行应用于所有可迭代对象中的元素。在有多个可迭代对象的情况下，迭代器在最短的迭代器耗尽时停止。

我们将在本章后面解释生成器的概念。现在，让我们将其转换为代码——我们将使用一个接受可变数量位置参数的 `lambda` 函数，并返回一个元组：

```py
# map.example.txt
>>> map(lambda *a: a, range(3))  # 1 iterable
<map object at 0x7f0db97adae0>  # Not useful! Let us use list
>>> list(map(lambda *a: a, range(3)))  # 1 iterable
[(0,), (1,), (2,)]
>>> list(map(lambda *a: a, range(3), "abc"))  # 2 iterables
[(0, 'a'), (1, 'b'), (2, 'c')]
>>> list(map(lambda *a: a, range(3), "abc", range(4, 7)))  # 3
[(0, 'a', 4), (1, 'b', 5), (2, 'c', 6)]
>>> # map stops at the shortest iterator
>>> list(map(lambda *a: a, (), "abc"))  # empty tuple is shortest
[]
>>> list(map(lambda *a: a, (1, 2), "abc"))  # (1, 2) shortest
[(1, 'a'), (2, 'b')]
>>> list(map(lambda *a: a, (1, 2, 3, 4), "abc"))  # "abc" shortest
[(1, 'a'), (2, 'b'), (3, 'c')] 
```

在前面的代码中，你可以看到为什么我们必须将调用包裹在 `list()` 中。没有它，我们会得到一个 `map` 对象的字符串表示形式。Python 对象的默认字符串表示会给出它们的类型和内存位置，在这个上下文中，这对我们来说并不特别有用。

您还可以注意到每个可迭代元素的函数应用方式；最初，每个可迭代元素的第一个元素被应用，然后是每个可迭代元素的第二个元素，依此类推。请注意，`map()` 在我们调用的最短可迭代对象耗尽时停止。这是一个非常有用的行为；它不会强迫我们将所有可迭代对象调整到相同的长度，也不会在它们长度不同时中断。

作为更有趣的例子，假设我们有一个包含学生字典的集合，每个字典中都有一个嵌套的学生学分字典。我们希望根据学生学分的总和对学生进行排序。然而，现有的数据并不允许直接应用排序函数。

为了解决这个问题，我们将应用 **装饰-排序-取消装饰** 惯用（也称为 **Schwartzian 转换**）。这是一种在较旧的 Python 版本中相当流行的技术，当时排序不支持使用 *键函数*。如今，它不再经常需要，但它偶尔仍然很有用。

要 **装饰** 一个对象意味着对其进行转换，无论是向其添加额外数据还是将其放入另一个对象中。相反，要 **取消装饰** 一个对象意味着将装饰过的对象恢复到其原始形式。

这种技术与 Python 装饰器无关，我们将在本书的后面部分探讨。

在以下示例中，我们可以看到 `map()` 是如何应用这个惯用的：

```py
# decorate.sort.undecorate.py
from pprint import pprint
students = [
    dict(id=0, credits=dict(math=9, physics=6, history=7)),
    dict(id=1, credits=dict(math=6, physics=7, latin=10)),
    dict(id=2, credits=dict(history=8, physics=9, chemistry=10)),
    dict(id=3, credits=dict(math=5, physics=5, geography=7)),
]
def decorate(student):
    # create a 2-tuple (sum of credits, student) from student dict
    return (**sum****(student["credits"].values())**, student)
def undecorate(decorated_student):
    # discard sum of credits, return original student dict
    return decorated_student[1]
print(students[0])
print(decorate(students[0])
students = sorted(**map****(decorate, students)**, reverse=True)
students = list(**map****(undecorate, students)**)
pprint(students) 
```

让我们先来了解每个学生对象是什么。实际上，让我们打印第一个：

```py
{'id': 0, 'credits': {'math': 9, 'physics': 6, 'history': 7}} 
```

您可以看到这是一个包含两个键的字典：`id` 和 `credits`。`credits` 的值也是一个字典，其中包含三个科目/成绩键值对。如您从 *第二章* ，*内置数据类型* 中回忆的那样，调用 `dict.values()` 返回一个可迭代对象，其中只有字典的值。因此，第一个学生的 `sum(student["credits"].values())` 等同于 `sum((9, 6, 7))`。

让我们打印调用 `decorate` 时第一个学生的结果：

```py
(22, {'id': 0, 'credits': {'math': 9, 'physics': 6, 'history': 7}}) 
```

如果我们这样装饰所有学生，我们只需对元组列表进行排序，就可以根据他们的总学分数对他们进行排序。为了将装饰应用到 `students` 中的每个项上，我们调用 `map(decorate, students)`。我们排序结果，然后以类似的方式取消装饰。

在运行整个代码后打印 `students` 会得到以下结果：

```py
[{'credits': {'chemistry': 10, 'history': 8, 'physics': 9}, 'id': 2},
 {'credits': {'latin': 10, 'math': 6, 'physics': 7}, 'id': 1},
 {'credits': {'history': 7, 'math': 9, 'physics': 6}, 'id': 0},
 {'credits': {'geography': 7, 'math': 5, 'physics': 5}, 'id': 3}] 
```

如您所见，学生对象确实是根据他们学分的总和进行排序的。

关于 *装饰-排序-取消装饰* 惯用，官方 Python 文档的 *排序 HOW TO* 部分有一个很好的介绍：

[`docs.python.org/3.12/howto/sorting.html#decorate-sort-undecorate`](https://docs.python.org/3.12/howto/sorting.html#decorate-sort-undecorate)

在排序部分需要注意的一点是，当两个或多个学生的总分相同时会发生什么。排序算法将接着通过比较 `student` 对象来对元组进行排序。这没有任何意义，并且在更复杂的情况下可能会导致不可预测的结果，甚至错误。如果你想要避免这个问题，一个简单的解决方案是创建一个三重元组而不是双重元组，将学分总和放在第一个位置，`student` 对象在原始 `students` 列表中的位置放在第二个位置，`student` 对象本身放在第三个位置。这样，如果学分总和相同，元组将根据位置进行排序，位置总是不同的，因此足以解决任何一对元组之间的排序问题。

## zip

我们已经在前面的章节中介绍了 `zip()`，所以让我们正确地定义它，之后我们想向你展示如何将它与 `map()` 结合使用。

根据 Python 文档（[`docs.python.org/3/library/functions.html#zip`](https://docs.python.org/3/library/functions.html#zip)），以下适用：

> zip(*iterables, strict=False)
> 
> ... 返回一个元组的迭代器，其中第 i 个元组包含来自每个参数可迭代对象的第 i 个元素。
> 
> 另一种思考 `zip()` 的方式是它将行转换为列，将列转换为行。这与矩阵转置类似。

让我们看看一个例子：

```py
# zip.grades.txt
>>> grades = [18, 23, 30, 27]
>>> avgs = [22, 21, 29, 24]
>>> list(zip(avgs, grades))
[(22, 18), (21, 23), (29, 30), (24, 27)]
>>> list(map(lambda *a: a, avgs, grades))  # equivalent to zip
[(22, 18), (21, 23), (29, 30), (24, 27)] 
```

在这里，我们将每个学生的平均分和最后一次考试的成绩进行组合。注意，使用 `map()`（示例中的最后两条指令）来重现 `zip()` 是多么简单。再次，我们必须使用 `list()` 来可视化结果。

与 `map()` 类似，`zip()` 通常会在到达最短可迭代对象的末尾时停止。然而，这可能会掩盖输入数据的问题，导致错误。例如，假设我们需要将学生名单和成绩列表合并到一个字典中，将每个学生的名字映射到他们的成绩。数据输入错误可能会导致成绩列表比学生名单短。以下是一个例子：

```py
# zip.strict.txt
>>> students = ["Sophie", "Alex", "Charlie", "Alice"]
>>> grades = ["A", "C", "B"]
>>> dict(zip(students, grades))
{'Sophie': 'A', 'Alex': 'C', 'Charlie': 'B'} 
```

注意，字典中没有 `"Alice"` 的条目。`zip()` 的默认行为掩盖了数据错误。因此，在 Python 3.10 中添加了仅关键字参数 `strict`。如果 `zip()` 接收到 `strict=True` 作为参数，当可迭代对象长度不相同时，它会引发异常：

```py
>>> dict(zip(students, grades, strict=True))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: zip() argument 2 is shorter than argument 1 
```

`itertools` 模块还提供了一个 `zip_longest()` 函数。它的行为类似于 `zip()`，但只有在最长的可迭代对象耗尽时才会停止。较短的迭代器将被指定为参数的值填充，默认为 `None` 。

## filter

根据 Python 文档（[`docs.python.org/3/library/functions.html#filter`](https://docs.python.org/3/library/functions.html#filter)），以下适用：

> filter(function, iterable)
> 
> 从可迭代对象中构建一个迭代器，该迭代器对于函数为真。可迭代对象可以是序列、支持迭代的容器或迭代器。如果函数是 None，则假定是恒等函数，即移除可迭代对象中所有为假的元素。

让我们看看一个快速示例：

```py
# filter.txt
>>> test = [2, 5, 8, 0, 0, 1, 0]
>>> list(filter(None, test))
[2, 5, 8, 1]
>>> list(filter(lambda x: x, test))  # equivalent to previous one
[2, 5, 8, 1]
>>> list(filter(lambda x: x > 4, test))  # keep only items > 4
[5, 8] 
```

注意到第二个 `filter()` 调用与第一个调用等效。如果我们传递一个接受一个参数并返回该参数本身的函数，只有那些使函数返回 `True` 的参数才会使函数返回 `True`。这种行为与传递 `None` 相同。模仿一些内置的 Python 行为通常是一个很好的练习。当你成功时，你可以说你完全理解了 Python 在特定情况下的行为。

有了 `map()`、`zip()`、`filter()`（以及 Python 标准库中的几个其他函数），我们可以非常有效地操作序列。但这些都是实现方式之一。让我们看看 Python 最强大的功能之一：*理解*。

# 理解

理解是一种对一组对象中的每个元素执行某些操作或选择满足某些条件的元素子集的简洁表示。它们借鉴了函数式编程语言 Haskell ([`www.haskell.org/`](https://www.haskell.org/))，并与迭代器和生成器一起，为 Python 增添了函数式风格。

Python 提供了多种类型的理解：列表、字典和集合。我们将专注于列表理解；一旦你理解了它们，其他类型就会很容易掌握。

让我们从简单的例子开始。我们想要计算一个包含前 10 个自然数的平方的列表。我们可以使用 `for` 循环并在每次迭代中将一个平方数追加到列表中：

```py
# squares.for.txt
>>> squares = []
>>> for n in range(10):
...     squares.append(n**2)
...
>>> squares
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81] 
```

这不是很优雅，因为我们必须首先初始化列表。使用 `map()`，我们可以在一行代码中实现相同的功能：

```py
# squares.map.txt
>>> squares = list(map(lambda n: n**2, range(10)))
>>> squares
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81] 
```

现在，让我们看看如何使用列表理解达到相同的结果：

```py
# squares.comprehension.txt
>>> [n**2 for n in range(10)]
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81] 
```

这更容易阅读，我们不再需要使用 lambda。我们已经在方括号内放置了一个 `for` 循环。现在让我们过滤掉奇数平方。我们首先会展示如何使用 `map()` 和 `filter()` 来实现，然后再使用列表理解：

```py
# even.squares.py
# using map and filter
sq1 = list(
    map(lambda n: n**2, filter(lambda n: not n % 2, range(10)))
)
# equivalent, but using list comprehensions
sq2 = [n**2 for n in range(10) if not n % 2]
print(sq1, sq1 == sq2)  # prints: [0, 4, 16, 36, 64] True 
```

我们认为，现在可读性的差异已经很明显。列表理解读起来更顺畅。它几乎就像英语：如果 n 是偶数，请给出 0 到 9 之间所有平方数（`n**2`）。

根据 Python 文档 ([`docs.python.org/3.12/tutorial/datastructures.html#list-comprehensions`](https://docs.python.org/3.12/tutorial/datastructures.html#list-comprehensions))，以下是真的：

> 列表理解由包含一个表达式、后跟一个 `for` 子句、然后是零个或多个 `for` 或 `if` 子句的括号组成。结果将是一个新列表，该列表是在评估随后的 `for` 和 `if` 子句的上下文中的表达式后生成的。

## 嵌套列表推导

让我们看看嵌套循环的一个例子。这相当常见，因为许多算法都涉及使用两个占位符迭代一个序列。第一个占位符从左到右遍历整个序列。第二个占位符也这样做，但它从第一个占位符开始，而不是从 0 开始。这个概念是测试所有对而不重复。让我们看看经典的`for`循环等效：

```py
# pairs.for.loop.py
items = "ABCD"
pairs = []
for a in range(len(items)):
    for b in range(a, len(items)):
        pairs.append((items[a], items[b])) 
```

如果您在最后打印`pairs`，您将得到以下内容：

```py
$ python pairs.for.loop.py
[('A', 'A'), ('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'B'), ('B', 'C'), ('B', 'D'), ('C', 'C'), ('C', 'D'), ('D', 'D')] 
```

所有具有相同字母的元组都是那些`b`位于与`a`相同位置的元组。现在，让我们看看我们如何将这个转换成列表推导：

```py
# pairs.list.comprehension.py
items = "ABCD"
pairs = [
    (items[a], items[b])    
    for a in range(len(items))
    for b in range(a, len(items))
] 
```

注意，因为`for`循环中的`b`依赖于`a`，所以在推导中它必须位于`a`的`for`循环之后。如果您交换它们的位置，您将得到一个命名错误。

实现相同结果的另一种方法是使用`itertools`模块中的`combinations_with_replacement()`函数（我们在*第三章*，*条件与迭代*中简要介绍了它）。您可以在官方 Python 文档中了解更多信息。

## 推导的过滤

我们也可以将过滤应用于推导。让我们首先使用`filter()`，找到所有短边小于 10 的毕达哥拉斯三元组。**毕达哥拉斯三元组**是一组满足方程*a² + b² = c²*的整数数的三元组*(a, b, c)*。

我们显然不希望测试组合两次，因此，我们将使用与上一个示例中看到类似的技巧：

```py
# pythagorean.triple.py
from math import sqrt
# this will generate all possible pairs
mx = 10
triples = [
    (a, b, sqrt(a**2 + b**2))
    for a in range(1, mx)
    for b in range(a, mx)
]
# this will filter out all non-Pythagorean triples
triples = list(
    filter(lambda triple: triple[2].is_integer(), triples)
)
print(triples)  # prints: [(3, 4, 5.0), (6, 8, 10.0)] 
```

在前面的代码中，我们生成了一组*三元组*，`triples`。每个元组包含两个整数数（两腿），以及勾股三角形的斜边，其两腿是元组中的前两个数。例如，当`a`是 3 且`b`是 4 时，元组将是`(3, 4, 5.0)`，当`a`是 5 且`b`是 7 时，元组将是`(5, 7, 8.602325267042627)`。

在生成所有`triples`之后，我们需要过滤掉那些斜边不是整数数的所有情况。为了实现这一点，我们根据`float_number.is_integer()`为`True`进行过滤。这意味着在我们刚刚向您展示的两个示例元组中，斜边为`5.0`的那个将被保留，而斜边为`8.602325267042627`的那个将被丢弃。

这很好，但我们不喜欢三元组中有两个整数数和一个浮点数——它们都应该都是整数。我们可以使用`map()`来解决这个问题：

```py
# pythagorean.triple.int.py
from math import sqrt
mx = 10
triples = [
    (a, b, sqrt(a**2 + b**2))
    for a in range(1, mx)
    for b in range(a, mx)
]
triples = filter(lambda triple: triple[2].is_integer(), triples)
# this will make the third number in the tuples integer
triples = list(
    map(lambda triple: **triple[:****2****] + (****int****(triple[****2****]),)**, triples)
)
print(triples)  # prints: [(3, 4, 5), (6, 8, 10)] 
```

注意我们添加的步骤。我们切片`triples`中的每个元素，只取前两个元素。然后，我们将切片与包含我们不喜欢的那浮点数的整数版本的单一元组连接起来。这段代码变得越来越复杂。我们可以用更简单的列表推导来实现相同的结果：

```py
# pythagorean.triple.comprehension.py
from math import sqrt
# this step is the same as before
mx = 10
triples = [
    (a, b, sqrt(a**2 + b**2))
    for a in range(1, mx)
    for b in range(a, mx)
]
# here we combine filter and map in one CLEAN list comprehension
triples = [
    (a, b, int(c)) for a, b, c in triples if c.is_integer()
]
print(triples)  # prints: [(3, 4, 5), (6, 8, 10)] 
```

这样更简洁、更易读、更短。尽管如此，仍有改进的空间。我们仍然在构建一个包含许多最终会被丢弃的三元组的列表中浪费内存。我们可以通过将两个推导式合并为一个来解决这个问题：

```py
# pythagorean.triple.walrus.py
from math import sqrt
# this step is the same as before
mx = 10
# We can combine generating and filtering in one comprehension
triples = [
    (a, b, int(c))
    for a in range(1, mx)
    for b in range(a, mx)
    if (**c := sqrt(a******2** **+ b******2****)**).is_integer()
]
print(triples)  # prints: [(3, 4, 5), (6, 8, 10)] 
```

现在这是优雅的。通过在同一个列表推导式中生成三元组和过滤它们，我们避免了在内存中保留任何未通过测试的三元组。注意，我们使用了一个`赋值表达式`来避免需要两次计算`sqrt(a**2 + b**2)`的值。

## 字典推导式

字典推导式与列表推导式的工作方式完全相同，但用于构建字典。在语法上只有细微的差别。以下示例足以解释你需要知道的一切：

```py
# dictionary.comprehensions.py
from string import ascii_lowercase
lettermap = {c: k for k, c in enumerate(ascii_lowercase, 1)} 
```

如果你打印`lettermap`，你会看到以下内容：

```py
$ python dictionary.comprehensions.py
{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,
'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15,
'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22,
'w': 23, 'x': 24, 'y': 25, 'z': 26} 
```

在前面的代码中，我们正在枚举所有小写 ASCII 字母的序列（使用`enumerate`函数）。然后我们构建一个字典，将结果字母/数字对作为键和值。注意语法与熟悉的字典语法相似。

还有另一种做同样事情的方法：

```py
lettermap = dict((c, k) for k, c in enumerate(ascii_lowercase, 1)) 
```

在这种情况下，我们向`dict`构造函数提供了一个生成器表达式（我们将在本章后面更多地讨论这些内容）。

字典不允许重复键，如下例所示：

```py
# dictionary.comprehensions.duplicates.py
word = "Hello"
swaps = {c: c.swapcase() for c in word}
print(swaps)  # prints: {'H': 'h', 'e': 'E', 'l': 'L', 'o': 'O'} 
```

我们创建了一个字典，以字符串`"Hello"`中的字母作为键，以相同字母但大小写互换作为值。注意，只有一个`"l": "L"`对。构造函数不会抱怨；它只是将重复的值重新分配给最后一个值。让我们用一个将每个键分配到字符串中位置的另一个例子来使这一点更清晰：

```py
# dictionary.comprehensions.positions.py
word = "Hello"
positions = {c: k for k, c in enumerate(word)}
print(positions)  # prints: {'H': 0, 'e': 1, 'l': 3, 'o': 4} 
```

注意与字母`l`关联的值：`3`。`l: 2`对不存在；它已被`l: 3`覆盖。

## 集合推导式

集合推导式与列表和字典推导式类似。让我们看一个快速示例：

```py
# set.comprehensions.py
word = "Hello"
letters1 = {c for c in word}
letters2 = set(c for c in word)
print(letters1)  # prints: {'H', 'o', 'e', 'l'}
print(letters1 == letters2)  # prints: True 
```

注意，对于集合推导式，就像字典一样，不允许重复，因此结果集合只有四个字母。此外，注意分配给`letters1`和`letters2`的表达式产生等效的集合。

创建`letters1`使用的语法与字典推导式类似。你只能通过以下事实来发现差异：字典需要键和值，通过冒号分隔，而集合不需要。对于`letters2`，我们向`set()`构造函数提供了一个生成器表达式。

# 生成器

**生成器**基于我们之前提到的**迭代**概念，并允许结合优雅与效率的编码模式。

生成器有两种类型：

+   **生成器函数**：这些与常规函数类似，但它们不是通过`return`语句返回结果，而是使用`yield`，这允许它们在每次调用之间挂起和恢复其状态。

+   **生成器表达式**：这些与我们在本章中看到的列表推导式类似，但它们返回的对象会逐个产生结果。

## 生成器函数

生成器函数在所有方面都表现得像常规函数，只有一个区别：它们不是一次性收集结果并返回，而是自动转换为迭代器，一次产生一个结果。

假设我们要求你从 1 数到 1,000,000。你开始数，在某个时刻，我们要求你停下来。过了一段时间，我们要求你继续。只要你记得你最后到达的数字，你就能从你离开的地方继续。例如，如果我们在你数到 31,415 后停下来，你就可以从 31,416 继续数下去。关键是，你不需要记住你之前说的所有数字，也不需要将它们写下来。生成器的行为与此非常相似。

仔细看看下面的代码：

```py
# first.n.squares.py
def get_squares(n): # classic function approach
    return [x**2 for x in range(n)]
print(get_squares(10))
def get_squares_gen(n):  # generator approach
    for x in range(n):
        yield x**2  # we yield, we do not return
print(list(get_squares_gen(10))) 
```

两个`print`语句的结果将是相同的：`[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]`。但这两个函数之间有一个重要的区别。`get_squares()`是一个经典函数，它将*[0, n)*区间内所有数字的平方收集到一个列表中，并返回它。另一方面，`get_squares_gen()`是一个生成器，其行为不同。每次解释器到达`yield`行时，它的执行就会暂停。那些`print`语句返回相同结果的原因仅仅是因为我们将`get_squares_gen()`传递给了`list()`构造函数，它通过请求下一个元素直到抛出`StopIteration`异常来完全耗尽生成器。让我们详细看看这一点：

```py
# first.n.squares.manual.py
def get_squares_gen(n):
    for x in range(n):
        yield x**2
squares = get_squares_gen(4)  # this creates a generator object
print(squares)  # <generator object get_squares_gen at 0x10dd...>
print(next(squares))  # prints: 0
print(next(squares))  # prints: 1
print(next(squares))  # prints: 4
print(next(squares))  # prints: 9
# the following raises StopIteration, the generator is exhausted,
# any further call to next will keep raising StopIteration
print(next(squares)) 
```

每次我们对生成器对象调用`next()`时，我们要么启动它（第一次`next()`），要么从最后一个暂停点恢复（任何其他`next()`）。我们第一次调用`next()`时，得到 0，这是 0 的平方，然后是 1，然后是 4，然后是 9，由于`for`循环在那之后停止（`n`是 4），生成器自然结束。在这一点上，一个经典函数会简单地返回`None`，但为了遵守迭代协议，生成器会抛出一个`StopIteration`异常。

这解释了`for`循环的工作原理。当你调用`for k in range(n)`时，幕后发生的事情是`for`循环从`range(n)`中获取一个迭代器，并开始调用它的`next`方法，直到抛出`StopIteration`异常，这告诉`for`循环迭代已达到其结束。

将这种行为内置到 Python 的每个迭代方面，使得生成器更加强大，因为一旦我们编写了它们，我们就能将它们插入到我们想要的任何迭代机制中。

到目前为止，你可能正在问自己，为什么你想使用生成器而不是常规函数。答案是节省时间和（尤其是）内存。

我们将在稍后讨论性能，但现在，让我们集中在一个方面：有时，生成器允许你做一些用简单列表不可能做到的事情。例如，假设你想分析一个序列的所有排列。如果序列的长度为*N*，那么它的排列数是*N!*。这意味着如果序列有 10 个元素，排列数是 3,628,800。但 20 个元素的序列将有 2,432,902,008,176,640,000 个排列。它们以阶乘的方式增长。

现在想象一下，你有一个经典函数，试图计算所有排列，将它们放入列表中，然后返回给你。对于 10 个元素，可能只需要几秒钟，但对于 20 个元素，根本无法完成（它可能需要数千年，并需要数十亿千兆字节的内存）。

另一方面，生成器函数能够开始计算，并给你返回第一个排列，然后是第二个，以此类推。当然，你可能没有时间处理它们全部——因为太多了——但至少你将能够处理其中的一些。有时，你必须迭代的数量数据如此巨大，以至于你不能将它们全部保存在列表中。在这种情况下，生成器是无价的：它们使得那些在其他情况下不可能的事情成为可能。

因此，为了节省内存（和时间），尽可能使用生成器函数。

还值得注意的是，你可以在生成器函数中使用`return`语句。它将引发`StopIteration`异常，从而有效地结束迭代。如果`return`语句使函数返回某些内容，它将破坏迭代协议。Python 的这种一致性防止了这种情况，并使我们编码时更加方便。让我们看一个简单的例子：

```py
# gen.yield.return.py
def geometric_progression(a, q):
    k = 0
    while True:
        result = a * q**k
        if result <= 100000:
            yield result
        else:
            return
        k += 1
for n in geometric_progression(2, 5):
    print(n) 
```

上述代码生成了几何级数的所有项，*a, aq, aq2, aq3, ...*。当级数产生一个大于 100,000 的项时，生成器停止（使用`return`语句）。运行代码会产生以下结果：

```py
$ python gen.yield.return.py
2
10
50
250
1250
6250
31250 
```

下一个项将是`156250`，这太大了。

## 超越`next`

生成器对象有方法可以让我们控制它们的行为：`send()`、`throw()`和`close()`。`send()`方法允许我们向生成器对象发送一个值，而`throw()`和`close()`分别允许我们在生成器内部引发异常并关闭它。它们的使用相当高级，我们在这里不会详细讨论，但我们要简单谈谈`send()`，以下是一个简单的例子：

```py
# gen.send.preparation.py
def counter(start=0):
    n = start
    while True:
        yield n
        n += 1
c = counter()
print(next(c))  # prints: 0
print(next(c))  # prints: 1
print(next(c))  # prints: 2 
```

前面的迭代器创建了一个会无限运行的生成器对象。你可以不断调用它，它永远不会停止。但如果你想在某个时刻停止它呢？一个解决方案是使用全局变量来控制`while`循环：

```py
# gen.send.preparation.stop.py
stop = False
def counter(start=0):
    n = start
    while not stop:
        yield n
        n += 1
c = counter()
print(next(c))  # prints: 0
print(next(c))  # prints: 1
stop = True
print(next(c))  # raises StopIteration 
```

我们最初将 `stop` 设置为 `False`，直到我们将其更改为 `True`，生成器将像之前一样继续运行。但是，当我们把 `stop` 改为 `True` 时，`while` 循环将退出，接下来的 `next` 调用将引发一个 `StopIteration` 异常。这个技巧是可行的，但不是一个令人满意的解决方案。函数依赖于外部变量，这可能导致问题。例如，如果另一个无关的函数更改了全局变量，生成器可能会意外停止。函数理想上应该是自包含的，不应依赖于全局状态。

生成器的 `send()` 方法接受一个单一参数，该参数作为 `yield` 表达式的值传递给生成器函数。我们可以使用这个方法将一个标志值传递给生成器，以指示它应该停止：

```py
# gen.send.py
def counter(start=0):
    n = start
    while True:
        result = yield n  # A
        print(type(result), result)  # B
        if result == "Q":
            break
        n += 1
c = counter()
print(next(c))  # C
print(c.send("Wow!"))  # D
print(next(c))  # E
print(c.send("Q"))  # F 
```

执行此代码将产生以下输出：

```py
$ python gen.send.py
0
<class 'str'> Wow!
1
<class 'NoneType'> None
2
<class 'str'> Q
Traceback (most recent call last):
  File "gen.send.py", line 16, in <module>
    print(c.send("Q")) # F
          ^^^^^^^^^^^
StopIteration 
```

我们认为逐行分析这段代码是值得的，就像我们正在执行它一样，以了解正在发生什么。

我们通过调用 `next()`（`#C`）开始生成器的执行。在生成器内部，`n` 被设置为与 `start` 相同的值。进入 `while` 循环，执行停止（`#A`），并将 `n`（`0`）返回给调用者。`0` 在控制台上打印。

然后我们调用 `send()`（`#D`），执行继续，`result` 被设置为 `"Wow!"`（仍然是 `#A`），其类型和值再次在控制台上打印（`#B`）。`result` 不是 `"Q"`，所以 `n` 增加 1，执行回到循环的顶部。`while` 条件为 `True`，因此开始另一个循环迭代。执行再次在 `#A` 处停止，并将 `n`（`1`）返回给调用者。`1` 在控制台上打印。

在这一点上，我们调用 `next()`（`#E`），执行继续（`#A`），因为我们没有明确地向生成器发送任何内容，所以 `yield n` 表达式（`#A`）返回 `None`（行为与调用不返回任何内容的函数相同）。因此，`result` 被设置为 `None`，其类型和值再次在控制台上打印（`#B`）。执行继续，`result` 不是 `"Q"`，所以 `n` 增加 1，然后再次开始另一个循环。执行再次停止（`#A`），并将 `n`（`2`）返回给调用者。`2` 在控制台上打印。

现在我们再次调用 `send`（`#F`），这次传递参数 `"Q"`。生成器继续执行，`result` 被设置为 `"Q"`（`#A`），其类型和值再次在控制台上打印（`#B`）。当我们再次到达 `if` 语句时，`result == "Q"` 评估为 `True`，`while` 循环通过 `break` 语句停止。生成器自然终止，这意味着引发了一个 `StopIteration` 异常。您可以在控制台打印的最后几行中看到异常的跟踪信息。

这一点在最初并不容易理解，所以如果您对此感到困惑，请不要气馁。您可以继续阅读，稍后再回到这个例子。

使用`send()`允许有趣的模式，并且值得注意的是，`send()`也可以用来启动生成器的执行（只要你用`None`调用它）。

## `yield from`表达式

另一个有趣的构造是`yield from`表达式。这个表达式允许你从子迭代器产生值。它的使用允许相当高级的模式，所以让我们快速看看它的一个例子：

```py
# gen.yield.for.py
def print_squares(start, end):
    for n in range(start, end):
        yield n**2
for n in print_squares(2, 5):
    print(n) 
```

上面的代码在控制台（单独的行）上打印了数字`4`、`9`和`16`。到现在为止，我们希望你自己能够理解它，但让我们快速回顾一下发生了什么。函数外部的`for`循环从`print_squares(2, 5)`获取一个迭代器，并对其调用`next()`直到迭代结束。每次调用生成器时，执行会在`yield n**2`处暂停（稍后恢复），这返回了当前`n`的平方。让我们看看我们如何使用`yield from`表达式来实现相同的结果：

```py
# gen.yield.from.py
def print_squares(start, end):
    yield from (n**2 for n in range(start, end))
for n in print_squares(2, 5):
    print(n) 
```

这段代码产生了相同的结果，但正如你所见，`yield from`实际上是在运行一个子迭代器`(n**2 ...)`。`yield from`表达式返回子迭代器产生的每个值。它更短，读起来更好。

## 生成器表达式

除了生成器函数之外，生成器还可以使用**生成器表达式**来创建。创建生成器表达式的语法与列表推导式相同，只是我们使用圆括号而不是方括号。

生成器表达式将生成与等效列表推导式相同的值序列。然而，生成器不会立即在内存中创建包含整个序列的列表对象，而是逐个产生值。重要的是要记住，你只能迭代生成器一次。之后，它将耗尽。

让我们看看一个例子：

```py
# generator.expressions.txt
>>> cubes = [k**3 for k in range(10)]  # regular list
>>> cubes
[0, 1, 8, 27, 64, 125, 216, 343, 512, 729]
>>> type(cubes)
<class 'list'>
>>> cubes_gen = (k**3 for k in range(10))  # create as generator
>>> cubes_gen
<generator object <genexpr> at 0x7f08b2004860>
>>> type(cubes_gen)
<class 'generator'>
>>> list(cubes_gen)  # this will exhaust the generator
[0, 1, 8, 27, 64, 125, 216, 343, 512, 729]
>>> list(cubes_gen)  # nothing more to give
[] 
```

如从尝试打印时的输出所看到的，`cubes_gen`是一个生成器对象。要查看它产生的值，我们可以使用一个`for`循环或手动调用`next()`，或者简单地将其传递给`list()`构造函数，这正是我们所做的。

注意，一旦生成器耗尽，就无法再次从中恢复相同的元素。如果我们想从头开始再次使用它，我们需要重新创建它。

在接下来的几个例子中，让我们看看如何使用生成器表达式来重现`map()`和`filter()`。首先，让我们看看`map()`：

```py
# gen.map.py
def adder(*n):
    return sum(n)
s1 = sum(map(adder, range(100), range(1, 101)))
s2 = sum(adder(*n) for n in zip(range(100), range(1, 101))) 
```

在前面的例子中，`s1`和`s2`都等于`adder(0, 1)`、`adder(1, 2)`、`adder(2, 3)`等之和，这相当于`sum(1, 3, 5, ...)`。我们发现生成器表达式语法更容易阅读。

现在来看`filter()`：

```py
# gen.filter.py
cubes = [x**3 for x in range(10)]
odd_cubes1 = filter(lambda cube: cube % 2, cubes)
odd_cubes2 = (cube for cube in cubes if cube % 2) 
```

在这个例子中，`odd_cubes1`和`odd_cubes2`是等效的：它们生成一个奇数立方序列。再次，我们更喜欢生成器语法。当事情变得稍微复杂一些时，这一点应该很明显：

```py
# gen.map.filter.py
N = 20
cubes1 = map(
    lambda n: (n, n**3),
    filter(lambda n: n % 3 == 0 or n % 5 == 0, range(N)),
)
cubes2 = ((n, n**3) for n in range(N) if n % 3 == 0 or n % 5 == 0) 
```

上述代码创建了两个迭代器`cubes1`和`cubes2`。它们都将产生相同的元组序列*(n, n3)*，其中`n`是 3 或 5 的倍数。如果你打印从任一迭代器获得的值列表，你会得到以下结果：`[(0, 0), (3, 27), (5, 125), (6, 216), (9, 729), (10, 1000), (12, 1728), (15, 3375), (18, 5832)]`。

注意到生成器表达式更容易阅读。对于简单的例子，这可能是有争议的，但一旦开始执行更复杂的操作，生成器语法的优越性就显而易见了。它更短，更简单，也更优雅。

现在，让我们问你：以下两行代码之间的区别是什么？

```py
# sum.example.py
s1 = sum([n**2 for n in range(10**6)])
s2 = sum((n**2 for n in range(10**6)))
s3 = sum(n**2 for n in range(10**6)) 
```

严格来说，它们都产生了相同的总和。获取`s2`和`s3`的表达式是等效的，因为`s2`中的括号是多余的。两者都是传递给`sum()`函数的生成器表达式。

获取`s1`的表达式是不同的。在这里，我们正在将列表推导式的结果传递给`sum()`。这既浪费了时间又浪费了内存，因为我们首先创建了一个包含一百万个元素的列表（必须存储在内存中）。然后我们将列表传递给`sum`，它遍历这个列表，之后我们丢弃这个列表。使用生成器表达式会更好，因为我们不需要等待列表构建完成，也不需要将一百万个值的整个序列存储在内存中。

因此，*在编写表达式时要小心额外的括号*。这样的细节很容易忽略，但它们可以产生重大差异。例如，看看下面的代码：

```py
# sum.example.2.py
s = sum([n**2 for n in range(10**10)])  # this is killed
# s = sum(n**2 for n in range(10**10))  # this succeeds
print(s)  # prints: 333333333283333333335000000000 
```

如果我们运行这段代码，我们会得到：

```py
$ python sum.example.2.py
Killed 
```

另一方面，如果我们取消注释第一行，并取消注释第二行，这是结果：

```py
$ python sum.example.2.py
333333333283333333335000000000 
```

这两行代码之间的区别在于，在第一行中，Python 解释器必须构建一个包含前十亿个数的平方的列表，以传递给`sum`函数。这个列表非常大，所以我们耗尽了内存，操作系统杀死了进程。

当我们移除方括号时，我们不再有一个列表。`sum`函数接收一个生成器，它产生 0、1、4、9 等，并计算总和，而不需要将所有值都保留在内存中。

# 一些性能考虑

通常有几种方法可以达到相同的结果。我们可以使用`map()`、`zip()`和`filter()`的任何组合，或者选择使用推导式或生成器。我们甚至可以决定使用`for`循环。在决定这些方法之间的选择时，可读性通常是一个因素。列表推导式或生成器表达式通常比复杂的`map()`和`filter()`组合更容易阅读。对于更复杂的操作，生成器函数或`for`循环通常更好。

然而，除了可读性方面的考虑，我们在决定使用哪种方法时还必须考虑性能。在比较不同实现性能时需要考虑两个因素：`空间`和`时间`。

空间指的是你的数据结构将要使用的内存量。最好的选择是问问自己你是否真的需要一个列表（或元组），或者是否可以使用生成器。

如果对后者的回答是肯定的，就选择生成器，因为它会节省大量空间。对于函数也是如此：如果你实际上不需要它们返回列表或元组，那么你也可以将它们转换为生成器函数。

有时候，你将不得不使用列表（或元组）；例如，有些算法使用多个指针扫描序列，还有一些需要多次遍历序列。生成器（函数或表达式）在耗尽之前只能迭代一次，所以在这种情况下，它可能不是最佳选择。

时间比空间复杂一些，因为它依赖于更多的变量，并且对于所有情况，我们并不总是能够绝对肯定地说“X 比 Y 快”。然而，基于今天在 Python 上运行的测试，我们可以这样说，平均而言，`map()`的性能与推导式和生成器表达式相似，而`for`循环则始终较慢。

要完全理解这些陈述背后的推理，我们需要了解 Python 是如何工作的，这超出了本书的范围，因为它相当技术性和详细。我们只能说，在解释器中，`map()`和推导式以 C 语言的速度运行，而 Python 的`for`循环在 Python 虚拟机中以 Python 字节码的形式运行，这通常要慢得多。

Python 有几种不同的实现方式。最初的一个，也是目前最常见的一个，是 CPython（[`github.com/python/cpython`](https://github.com/python/cpython)），它是用 C 语言编写的。C 语言是今天仍在使用的最强大和最受欢迎的编程语言之一。

在本节的剩余部分，我们将进行一些简单的实验来验证这些性能主张。我们将编写一小段代码，收集整数对`(a, b)`的`divmod(a, b)`的结果。我们将使用`time`模块中的`time()`函数来计算我们执行的操作的经过时间：

```py
# performance.py
from time import time
mx = 5000
t = time()  # start time for the for loop
floop = []
for a in range(1, mx):
    for b in range(a, mx):
        floop.append(divmod(a, b))
print("for loop: {:.4f} s".format(time() - t))  # elapsed time
t = time()  # start time for the list comprehension
compr = [divmod(a, b) for a in range(1, mx) for b in range(a, mx)]
print("list comprehension: {:.4f} s".format(time() - t))
t = time()  # start time for the generator expression
gener = list(
    divmod(a, b) for a in range(1, mx) for b in range(a, mx)
)
print("generator expression: {:.4f} s".format(time() - t)) 
```

如你所见，我们正在创建三个列表：`floop`、`compr`和`gener`。运行代码会产生以下结果：

```py
$ python performance.py
for loop: 2.3832 s
list comprehension: 1.6882 s
generator expression: 1.6525 s 
```

列表推导式的执行时间大约占`for`循环时间的 71%。生成器表达式的执行速度略快，大约为 69%。列表推导式和生成器表达式之间的时间差异几乎不显著，如果你多次重新运行示例，你可能会看到列表推导式比生成器表达式用时更少。

值得注意的是，在`for`循环的主体中，我们正在向列表中追加数据。这意味着在幕后，Python 解释器偶尔需要调整列表的大小，为追加更多项分配空间。我们猜测，创建一个零列表，并简单地填充结果，可能会加快`for`循环的速度，但我们错了。自己试试看；你只需要`mx * (mx - 1) // 2`个元素预先分配。

我们在这里用于计时执行的方法相当天真。在*第十一章*，*调试和性能分析*中，我们将探讨更好的代码性能分析和计时方法。

让我们看看一个类似的例子，比较`for`循环和`map()`调用：

```py
# performance.map.py
from time import time
mx = 2 * 10**7
t = time()
absloop = []
for n in range(mx):
    absloop.append(abs(n))
print("for loop: {:.4f} s".format(time() - t))
t = time()
abslist = [abs(n) for n in range(mx)]
print("list comprehension: {:.4f} s".format(time() - t))
t = time()
absmap = list(map(abs, range(mx)))
print("map: {:.4f} s".format(time() - t)) 
```

这段代码在概念上与前面的例子相似。唯一不同的是，我们正在应用`abs()`函数而不是`divmod()`，我们只有一个循环而不是两个嵌套循环。执行结果如下：

```py
$ python performance.map.py
for loop: 1.9009 s
list comprehension: 1.0973 s
map: 0.5862 s 
```

这次，`map`是最快的：它所需的时间是列表推导式的约 53%，是`for`循环所需时间的约 31%。

这些实验的结果给我们提供了一个关于`for`循环、列表推导式、生成器表达式和`map()`函数相对速度的大致指示。然而，不要过分依赖这些结果，因为我们在这里进行的实验相当简单，准确测量和比较执行时间是很困难的。测量很容易受到多个因素的影响，例如在同一台计算机上运行的其它进程。性能结果也严重依赖于硬件、操作系统和 Python 版本。

很明显，`for`循环比列表推导式或`map()`函数慢，所以讨论为什么我们仍然经常选择它们而不是替代方案是值得的。

# 不要过度使用列表推导式和生成器

我们已经看到了列表推导式和生成器表达式有多么强大。然而，我们发现，你在一个单一的列表推导式或生成器表达式中尝试做的事情越多，就越难阅读、理解和维护或更改。

如果你再次考虑 Python 的禅意，有几行代码我们认为在处理优化代码时值得记住：

```py
>>> import this
...
Explicit is better than implicit.
Simple is better than complex.
...
Readability counts.
...
If the implementation is hard to explain, it's a bad idea.
... 
```

列表推导式和生成器表达式比显式表达式更隐晦，可能相当难以阅读和理解，也可能难以解释。有时，你必须使用从内到外的技术将其分解，才能理解正在发生的事情。

为了给你一个例子，让我们再详细谈谈毕达哥拉斯三元组。只是为了提醒你，毕达哥拉斯三元组是一个正整数元组 *(a, b, c)*，其中 *a*² + *b*² = *c*²。我们在*过滤列表推导式*部分看到了如何计算它们，但我们以一种非常低效的方式做了这件事。我们扫描了低于某个阈值的所有数字对，计算斜边，并过滤掉那些不是有效的毕达哥拉斯三元组的数字对。

获取毕达哥拉斯三元组的更好方法是直接生成它们。你可以使用许多不同的公式来做这件事；在这里，我们将使用 **欧几里得公式**。这个公式表明，任何三元组 *(a, b, c)*，其中 *a = m* ² *- n* ²*，*b = 2mn*，*c = m* ² *+ n* ²，*m* 和 *n* 是满足 *m > n* 的正整数，都是一个毕达哥拉斯三元组。例如，当 *m = 2* 和 *n = 1* 时，我们找到最小的三元组：*(3, 4, 5)*。

但是有一个问题：考虑三元组 *(6, 8, 10)*，它类似于 *(3, 4, 5)*，只是所有的数字都乘以了 *2*。这个三元组是毕达哥拉斯三元组，因为 *6* ² *+ 8* ² *= 10* ²，但我们可以通过将它的每个元素乘以 *2* 从 *(3, 4, 5)* 推导出来。同样适用于 *(9, 12, 15)*，*(12, 16, 20)*，以及一般地，我们可以写成 *(3k, 4k, 5k)* 的所有三元组，其中 *k* 是大于 *1* 的正整数。

不能通过将另一个三元组的元素乘以某个因子 *k* 得到的三元组被称为 **原始的**。另一种说法是：如果一个三元组的三个元素是 **互质的**，那么这个三元组是原始的。两个数互质是指它们在它们的除数中没有共享任何质因数，也就是说，当它们的 **最大公约数** ( **GCD** ) 是 *1* 时。例如，3 和 5 是互质的，而 3 和 6 不是，因为它们都可以被 3 整除。

欧几里得公式告诉我们，如果 *m* 和 *n* 互质，并且 *m - n* 是奇数，它们生成的三元组是 *原始的*。在下面的例子中，我们将编写一个生成器表达式来计算所有斜边 *c* 小于或等于某个整数 *N* 的原始毕达哥拉斯三元组。这意味着我们想要所有满足 *m* ² *+ n* ² *≤ N* 的三元组。当 *n* 为 *1* 时，公式看起来是这样的：*m* ² *≤ N - 1*，这意味着我们可以用 *m ≤ N* ^(1/2) 的上界来近似计算。

总结一下：*m* 必须大于 *n*，它们也必须是互质的，并且它们的差 *m - n* 必须是奇数。此外，为了避免无用的计算，我们将 *m* 的上界设置为 *floor(sqrt(N)) + 1*。

实数 *x* 的 `floor` 函数给出最大的整数 *n*，使得 *n < x*，例如，*floor(3.8) = 3*，*floor(13.1) = 13*。取 *floor(sqrt(N)) + 1* 的意思是取 *N* 的平方根的整数部分，并加上一个最小的边距，以确保我们不会错过任何数字。

让我们一步步将这些内容放入代码中。我们首先编写一个简单的 `gcd()` 函数，它使用 **欧几里得算法**：

```py
# functions.py
def gcd(a, b):
    """Calculate the Greatest Common Divisor of (a, b)."""
    while b != 0:
        a, b = b, a % b
    return a 
```

欧几里得算法的解释可以在网上找到，所以我们不会在这里花费时间讨论它，因为我们需要专注于生成器表达式。下一步是使用我们之前收集的知识来生成一个原始毕达哥拉斯三元组的列表：

```py
# pythagorean.triple.generation.py
from functions import gcd
N = 50
triples = sorted(  # 1
    (
        (a, b, c) for a, b, c in (  # 2
            ((m**2 - n**2), (2 * m * n), (m**2 + n**2))  # 3
            for m in range(1, int(N**.5) + 1)  # 4
            for n in range(1, m)  # 5
            if (m - n) % 2 and gcd(m, n) == 1  # 6
        )
        if c <= N  # 7
    ),
    key=sum  # 8
) 
```

这段代码不易阅读，让我们逐行分析。在`#3`行，我们开始一个生成器表达式，创建三元组。你可以从`#4`和`#5`看到，我们在`*[1, M]*`上循环`m`，其中`M`是`*sqrt(N)*`的整数部分，加上`*1*`。另一方面，`n`在`*[1, m)*`范围内循环，以遵守`*m > n*`规则。值得注意的是我们如何计算`*sqrt(N)*`，即`N**.5`，这是我们想展示的另一种方法。

在`#6`处，你可以看到用于使三元组成为原始的筛选条件：`(m - n) % 2`在`(m - n)`为奇数时评估为`True`，而`gcd(m, n) == 1`意味着`m`和`n`是互质的。有了这些条件，我们知道三元组将是原始的。这解决了最内层的生成器表达式。最外层的生成器表达式从`#2`开始，到`#7`结束。我们取三元组`(a, b, c)`，其中`c <= N`，来自`(...innermost generator...)`。

最后，在`#1`处，我们应用排序来按顺序展示列表。在`#8`处，在外层生成器表达式关闭后，你可以看到我们指定排序键为`a + b + c`的总和。这仅仅是我们个人的偏好；背后没有数学上的原因。

这段代码当然不容易理解或解释。这样的代码也难以调试或修改。它不应该出现在专业环境中。

让我们看看我们能否将此代码重写为更易读的形式：

```py
# pythagorean.triple.generation.for.py
from functions import gcd
def gen_triples(N):
    for m in range(1, int(N**.5) + 1):  # 1
        for n in range(1, m):  # 2
            if (m - n) % 2 and gcd(m, n) == 1:  # 3
                c = m**2 + n**2  # 4
                if c <= N:  # 5
                    a = m**2 - n**2  # 6
                    b = 2 * m * n  # 7
                    yield (a, b, c)  # 8
triples = sorted(gen_triples(50), key=sum)  # 9 
```

这段代码更容易阅读。让我们逐行分析。你会发现它也更容易理解。

我们从`#1`和`#2`开始循环，范围与上一个示例相同。在`#3`行，我们筛选原始三元组。在`#4`行，我们稍微偏离了之前的行为：我们计算`c`，在`#5`行，我们筛选`c`小于或等于`N`。我们只计算`a`和`b`，如果`c`满足该条件，则产生结果元组。我们本可以在更早的时候计算`a`和`b`的值，但通过推迟到我们知道所有有效三元组的条件都满足，我们避免了浪费时间和 CPU 周期。在最后一行，我们使用与生成器表达式示例中相同的键进行排序。

我们希望你会同意这个示例更容易理解。如果我们需要修改代码，这将更容易，并且与生成器表达式相比，工作起来更不容易出错。

如果你打印出这两个示例的结果，你会得到以下内容：

```py
[(3, 4, 5), (5, 12, 13), (15, 8, 17), (7, 24, 25), (21, 20, 29), (35, 12, 37), (9, 40, 41)] 
```

在性能和可读性之间往往存在权衡，而且并不总是容易找到平衡点。我们的建议是尽可能使用列表推导和生成器表达式。但如果代码开始变得难以修改或难以阅读或解释，你可能想要将其重构为更易读的形式。

# 名称本地化

现在我们已经熟悉了所有类型的推导式和生成器表达式，让我们来谈谈它们内部的名称本地化。Python 3 在所有四种推导式形式中本地化循环变量：列表、字典、集合和生成器表达式。这种行为与 `for` 循环的行为不同。让我们看看一些简单的例子来展示所有情况：

```py
# scopes.py
A = 100
ex1 = [A for A in range(5)]
print(A)  # prints: 100
ex2 = list(A for A in range(5))
print(A)  # prints: 100
ex3 = {A: 2 * A for A in range(5)}
print(A)  # prints: 100
ex4 = {A for A in range(5)}
print(A)  # prints: 100
s = 0
for A in range(5):
    s += A
print(A)  # prints: 4 
```

在前面的代码中，我们声明了一个全局名称，`A = 100.` 然后，我们有列表、字典和集合推导式，以及一个生成器表达式。尽管它们都使用了名称 `A`，但它们都没有改变全局名称 `A`。另一方面，最后的 `for` 循环确实修改了全局的 `A`。最后的 `print` 语句打印了 4。

让我们看看如果全局的 `A` 不存在会发生什么：

```py
# scopes.noglobal.py
ex1 = [A for A in range(5)]
print(A)  # breaks: NameError: name 'A' is not defined 
```

前面的代码在处理任何其他类型的理解或生成器表达式时都会以相同的方式工作。运行第一行后，`A` 在全局命名空间中未定义。再次，`for` 循环的行为不同：

```py
# scopes.for.py
s = 0
for A in range(5):
    s += A
print(A) # prints: 4
print(globals()) 
```

前面的代码表明，在 `for` 循环之后，如果循环变量在它之前未定义，我们可以在全局命名空间中找到它。我们可以通过检查 `globals()` 内置函数返回的字典来验证这一点：

```py
$ python scopes.for.py
4
{'__name__': '__main__', '__doc__': None, ..., 's': 10, 'A': 4} 
```

除了各种内置的全局名称（我们在此没有重复），我们还看到了 `'A': 4` 。

# 内置函数的生成器行为

生成器行为在内置类型和函数中相当常见。这是 Python 2 和 Python 3 之间的一个主要区别。在 Python 2 中，`map()`、`zip()` 和 `filter()` 等函数返回列表而不是可迭代对象。这种变化背后的想法是，如果你需要创建一个包含这些结果的列表，你总是可以用 `list()` 类来包装调用。另一方面，如果你只需要迭代并且希望尽可能减少内存影响，你可以安全地使用这些函数。另一个值得注意的例子是 `range()` 函数。在 Python 2 中，它返回一个列表，还有一个名为 `xrange()` 的函数，其行为与 Python 3 中 `range()` 函数的行为相似。

函数和方法返回可迭代对象的想法相当普遍。你可以在 `open()` 函数中找到它，该函数用于操作文件对象（我们将在 *第八章* ，*文件和数据持久性* 中看到它），也可以在 `enumerate()`、字典的 `keys()`、`values()` 和 `items()` 方法以及几个其他地方找到。

这一切都有道理：Python 旨在通过尽可能避免浪费空间来减少内存占用，尤其是在那些在大多数情况下广泛使用的函数和方法中。

在本章的开头，我们说过，优化必须处理大量对象代码的性能比从每天调用两次的函数中节省几毫秒更有意义。这正是 Python 本身在这里所做的事情。

# 最后一个例子

在我们完成这一章之前，我们将向你展示一个简单的问题，Fabrizio 曾经用它来测试应聘他曾经工作过的公司 Python 开发者职位的候选人。

问题如下：编写一个函数，返回数列 *0 1 1 2 3 5 8 13 21 ...* 的项，直到某个限制 *N* 。

如果你还没有认出它，那就是斐波那契数列，它被定义为 *F(0) = 0, F(1) = 1* ，并且对于任何 *n > 1* ，*F(n) = F(n-1) + F(n-2)* 。这个数列非常适合测试关于递归、记忆化技术以及其他技术细节的知识，但在这个情况下，这是一个检查候选人是否了解生成器的好机会。

让我们从最基础版本开始，然后对其进行改进：

```py
# fibonacci.first.py
def fibonacci(N):
    """Return all fibonacci numbers up to N."""
    result = [0]
    next_n = 1
    while next_n <= N:
        result.append(next_n)
        next_n = sum(result[-2:])
    return result
print(fibonacci(0))  # [0]
print(fibonacci(1))  # [0, 1, 1]
print(fibonacci(50))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34] 
```

从顶部开始：我们将 `result` 列表设置为起始值 `[0]`。然后我们从下一个元素（`next_n`）开始迭代，它是 1。当下一个元素不大于 `N` 时，我们继续将其追加到列表中，并计算序列中的下一个值。我们通过从 `result` 列表中的最后两个元素中取一个切片并将其传递给 `sum` 函数来计算下一个元素。

如果你难以理解代码，添加一些 `print()` 语句可能会有所帮助，这样你就可以看到在执行过程中值是如何变化的。

当循环条件评估为 `False` 时，我们退出循环并返回 `result`。你可以看到每个 `print` 语句旁边的注释中的结果。

到这一点，Fabrizio 会问候选人以下问题：*如果我只想迭代这些数字怎么办？* 一个好的候选人会相应地更改代码如下：

```py
# fibonacci.second.py
def fibonacci(N):
    """Return all fibonacci numbers up to N."""
    yield 0
    if N == 0:
        return
    a = 0
    b = 1
    while b <= N:
        yield b
        a, b = b, a + b
print(list(fibonacci(0)))  # [0]
print(list(fibonacci(1)))  # [0, 1, 1]
print(list(fibonacci(50)))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34] 
```

这实际上是他被给出的解决方案之一。现在，`fibonacci()` 函数是一个 *生成器函数* 。首先，我们产生 0，然后，如果 `N` 是 0，我们 `return`（这将引发 `StopIteration` 异常）。如果不是这种情况，我们开始循环，在每次迭代中产生 `b`，然后更新 `a` 和 `b`。这个解决方案依赖于我们只需要最后两个元素（`a` 和 `b`）来产生下一个元素。

这段代码更好，内存占用更少，我们只需要用 `list()` 包装调用，就像平常一样，就可以得到斐波那契数的列表。不过，我们可以让它更加优雅：

```py
# fibonacci.elegant.py
def fibonacci(N):
    """Return all fibonacci numbers up to N."""
    a, b = 0, 1
    while a <= N:
        yield a
        a, b = b, a + b 
```

函数的主体现在只有四行，如果你把文档字符串也算上，则是五行。注意，在这种情况下，使用元组赋值（`a, b = 0, 1` 和 `a, b = b, a + b`）如何有助于使代码更短、更易读。

# 摘要

在这一章中，我们更深入地探讨了迭代和生成的概念。我们详细研究了 `map()`、`zip()` 和 `filter()` 函数，并学习了如何将它们用作常规 `for` 循环方法的替代方案。

然后，我们介绍了构建列表、字典和集合的推导式概念。我们探讨了它们的语法以及如何将它们用作经典`for`循环方法以及`map()`、`zip()`和`filter()`函数的替代方案。

最后，我们讨论了生成器的两种形式：生成器函数和表达式。我们学习了如何通过使用生成技术来节省时间和空间。我们还看到了原本用列表无法执行的操作，可以用生成器来完成。

我们讨论了性能，并看到在速度方面，`for`循环排在最后，但它们提供了最佳的可读性和灵活性，便于更改。另一方面，`map()`和`filter()`等函数以及推导式可以更快。

使用这些技术编写的代码的复杂度呈指数增长，因此为了提高可读性和易于维护，我们有时仍然需要使用经典的`for`循环方法。另一个区别在于名称本地化，`for`循环的行为与其他所有类型的推导式不同。

下一章将全部关于对象和类。在结构上与这一章相似，即我们不会探索很多不同的主题——只是其中的一些——但我们将尝试更深入地探讨它们。

# 加入我们的 Discord 社区

加入我们的社区 Discord 空间，与作者和其他读者进行讨论：

[加入我们的 Discord 社区](https://discord.com/invite/uaKmaz7FEC)

![二维码](img/QR_Code119001106417026468.png)
