# 第五章：节省时间和内存

“不是每天增加，而是每天减少。砍掉不必要的部分。”- 李小龙

我喜欢李小龙的这句话。他是一个很聪明的人！特别是第二部分，“*砍掉不必要的部分*”，对我来说是使计算机程序优雅的原因。毕竟，如果有更好的方法来做事情，这样我们就不会浪费时间或内存，为什么不呢？

有时，不将我们的代码推向最大限度是有合理的原因的：例如，有时为了实现微不足道的改进，我们必须牺牲可读性或可维护性。当我们可以用可读性强、清晰的代码在 1.05 秒内提供网页，而不是用难以理解、复杂的代码在 1 秒内提供网页时，这是没有意义的。

另一方面，有时候从一个函数中削减一毫秒是完全合理的，特别是当这个函数被调用数千次时。你在那里节省的每一毫秒意味着每一千次调用节省一秒，这对你的应用可能是有意义的。

鉴于这些考虑，本章的重点不是为你提供将代码推向性能和优化的绝对极限的工具，“不管怎样”，而是使你能够编写高效、优雅的代码，读起来流畅，运行快速，并且不会明显浪费资源。

在本章中，我们将涵盖以下内容：

+   map、zip 和 filter 函数

+   推导式

+   生成器

我将进行几项测量和比较，并谨慎得出一些结论。请记住，在一个不同的盒子上，使用不同的设置或不同的操作系统，结果可能会有所不同。看看这段代码：

```py
# squares.py
def square1(n):
    return n ** 2  # squaring through the power operator

def square2(n):
    return n * n  # squaring through multiplication
```

这两个函数都返回`n`的平方，但哪个更快？从我对它们进行的简单基准测试来看，第二个似乎稍微更快。如果你仔细想想，这是有道理的：计算一个数字的幂涉及乘法，因此，无论你使用什么算法来执行幂运算，它都不太可能击败`square2`中的简单乘法。

我们在乎这个结果吗？在大多数情况下，不在乎。如果你正在编写一个电子商务网站，很可能你甚至不需要将一个数字提高到二次方，如果你需要，这可能是一个零星的操作。你不需要担心在你调用几次的函数上节省一小部分微秒。

那么，优化什么时候变得重要呢？一个非常常见的情况是当你必须处理大量的数据集时。如果你在一百万个“客户”对象上应用相同的函数，那么你希望你的函数调整到最佳状态。在一个被调用一百万次的函数上节省 1/10 秒，可以节省你 100,000 秒，大约 27.7 小时。这不一样，对吧？所以，让我们专注于集合，让我们看看 Python 给你提供了哪些工具来高效优雅地处理它们。

我们将在本章中看到的许多概念都是基于迭代器和可迭代对象的概念。简单地说，当要求一个对象返回其下一个元素时，以及在耗尽时引发`StopIteration`异常的能力。我们将看到如何在第六章中编写自定义迭代器和可迭代对象，*面向对象编程、装饰器和迭代器*。

由于我们将在本章中探讨的对象的性质，我经常被迫将代码包装在`list`构造函数中。这是因为将迭代器/生成器传递给`list(...)`会耗尽它，并将所有生成的项目放入一个新创建的列表中，我可以轻松地打印出来显示它的内容。这种技术会影响可读性，所以让我介绍一个`list`的别名：

```py
# alias.py
>>> range(7)
range(0, 7)
>>> list(range(7))  # put all elements in a list to view them
[0, 1, 2, 3, 4, 5, 6]
>>> _ = list  # create an "alias" to list
>>> _(range(7))  # same as list(range(7))
[0, 1, 2, 3, 4, 5, 6]
```

我已经突出显示的三个部分中，第一个是我们需要执行的调用，以便显示`range(7)`生成的内容，第二个是我创建别名到`list`的时刻（我选择了希望不引人注目的下划线），第三个是等效的调用，当我使用别名而不是`list`时。

希望这样做可以提高可读性，请记住，我将假设这个别名已经在本章的所有代码中定义了。

# map、zip 和 filter 函数

我们将从回顾`map`、`filter`和`zip`开始，这些是处理集合时可以使用的主要内置函数，然后我们将学习如何使用两个非常重要的构造来实现相同的结果：**推导**和**生成器**。系好安全带！

# 地图

根据官方 Python 文档：

map(function, iterable, ...)返回一个迭代器，它将函数应用于可迭代对象的每个项目，产生结果。如果传递了额外的可迭代参数，函数必须接受相同数量的参数，并且会并行应用于所有可迭代对象的项目。对于多个可迭代对象，当最短的可迭代对象耗尽时，迭代器会停止。

我们将在本章后面解释 yielding 的概念。现在，让我们将其翻译成代码——我们将使用一个接受可变数量的位置参数的`lambda`函数，并将它们返回为一个元组：

```py
# map.example.py
>>> map(lambda *a: a, range(3))  # 1 iterable
<map object at 0x10acf8f98>  # Not useful! Let's use alias
>>> _(map(lambda *a: a, range(3)))  # 1 iterable
[(0,), (1,), (2,)]
>>> _(map(lambda *a: a, range(3), 'abc'))  # 2 iterables
[(0, 'a'), (1, 'b'), (2, 'c')]
>>> _(map(lambda *a: a, range(3), 'abc', range(4, 7)))  # 3
[(0, 'a', 4), (1, 'b', 5), (2, 'c', 6)]
>>> # map stops at the shortest iterator
>>> _(map(lambda *a: a, (), 'abc'))  # empty tuple is shortest
[]
>>> _(map(lambda *a: a, (1, 2), 'abc'))  # (1, 2) shortest
[(1, 'a'), (2, 'b')]
>>> _(map(lambda *a: a, (1, 2, 3, 4), 'abc'))  # 'abc' shortest
[(1, 'a'), (2, 'b'), (3, 'c')]
```

在前面的代码中，你可以看到为什么我们必须用`list(...)`（或者在这种情况下使用它的别名`_`）来包装调用。没有它，我会得到一个`map`对象的字符串表示，这在这种情况下并不真正有用，是吗？

你还可以注意到每个可迭代对象的元素是如何应用于函数的；首先是每个可迭代对象的第一个元素，然后是每个可迭代对象的第二个元素，依此类推。还要注意，`map`在我们调用它的可迭代对象中最短的一个耗尽时停止。这实际上是一种非常好的行为；它不强迫我们将所有可迭代对象平齐到一个公共长度，并且如果它们的长度不相同时也不会中断。

当你必须将相同的函数应用于一个或多个对象集合时，`map`非常有用。作为一个更有趣的例子，让我们看看**装饰-排序-解除装饰**惯用法（也称为**Schwartzian transform**）。这是一种在 Python 排序没有提供*key-functions*时非常流行的技术，因此今天使用较少，但偶尔还是会派上用场的一个很酷的技巧。

让我们在下一个例子中看一个变体：我们想按照学生所累积的学分总和降序排序，以便将最好的学生放在位置 0。我们编写一个函数来生成一个装饰对象，然后进行排序，然后进行 undecorate。每个学生在三个（可能不同的）科目中都有学分。在这种情况下，装饰对象意味着以一种允许我们按照我们想要的方式对原始对象进行排序的方式来转换它，无论是向其添加额外数据，还是将其放入另一个对象中。这种技术与 Python 装饰器无关，我们将在本书后面探讨。

在排序之后，我们将装饰的对象恢复为它们的原始对象。这被称为 undecorate：

```py
# decorate.sort.undecorate.py
students = [
    dict(id=0, credits=dict(math=9, physics=6, history=7)),
    dict(id=1, credits=dict(math=6, physics=7, latin=10)),
    dict(id=2, credits=dict(history=8, physics=9, chemistry=10)),
    dict(id=3, credits=dict(math=5, physics=5, geography=7)),
]

def decorate(student):
    # create a 2-tuple (sum of credits, student) from student dict
    return (sum(student['credits'].values()), student)

def undecorate(decorated_student):
    # discard sum of credits, return original student dict
    return decorated_student[1]

students = sorted(map(decorate, students), reverse=True)
students = _(map(undecorate, students))
```

让我们首先了解每个学生对象是什么。实际上，让我们打印第一个：

```py
{'credits': {'history': 7, 'math': 9, 'physics': 6}, 'id': 0}
```

你可以看到它是一个具有两个键的字典：`id`和`credits`。`credits`的值也是一个字典，在其中有三个科目/成绩键/值对。正如你在数据结构世界中所记得的，调用`dict.values()`会返回一个类似于`iterable`的对象，只有值。因此，第一个学生的`sum(student['credits'].values())`等同于`sum((9, 6, 7))`。

让我们打印调用 decorate 与第一个学生的结果：

```py
>>> decorate(students[0])
(22, {'credits': {'history': 7, 'math': 9, 'physics': 6}, 'id': 0})
```

如果我们对所有学生都这样装饰，我们可以通过仅对元组列表进行排序来按学分总额对它们进行排序。为了将装饰应用到 students 中的每个项目，我们调用`map(decorate, students)`。然后我们对结果进行排序，然后以类似的方式进行解除装饰。如果你已经正确地阅读了之前的章节，理解这段代码不应该太难。

运行整个代码后打印学生：

```py
$ python decorate.sort.undecorate.py
[{'credits': {'chemistry': 10, 'history': 8, 'physics': 9}, 'id': 2},
 {'credits': {'latin': 10, 'math': 6, 'physics': 7}, 'id': 1},
 {'credits': {'history': 7, 'math': 9, 'physics': 6}, 'id': 0},
 {'credits': {'geography': 7, 'math': 5, 'physics': 5}, 'id': 3}]
```

你可以看到，根据学生对象的顺序，它们确实已经按照他们的学分总和进行了排序。

有关*装饰-排序-解除装饰*习惯用法的更多信息，请参阅官方 Python 文档的排序指南部分（[`docs.python.org/3.7/howto/sorting.html#the-old-way-using-decorate-sort-undecorate`](https://docs.python.org/3.7/howto/sorting.html#the-old-way-using-decorate-sort-undecorate)）。

关于排序部分要注意的一件事是：如果两个或更多的学生总分相同怎么办？排序算法将继续通过比较`student`对象来对元组进行排序。这没有任何意义，在更复杂的情况下，可能会导致不可预测的结果，甚至错误。如果你想确保避免这个问题，一个简单的解决方案是创建一个三元组而不是两元组，将学分总和放在第一个位置，`students`列表中`student`对象的位置放在第二个位置，`student`对象本身放在第三个位置。这样，如果学分总和相同，元组将根据位置进行排序，位置总是不同的，因此足以解决任何一对元组之间的排序问题。

# zip

我们已经在之前的章节中介绍了`zip`，所以让我们正确定义它，然后我想向你展示如何将它与`map`结合起来使用。

根据 Python 文档：

zip(*iterables)返回一个元组的迭代器，其中第 i 个元组包含来自每个参数序列或可迭代对象的第 i 个元素。当最短的输入可迭代对象耗尽时，迭代器停止。使用单个可迭代对象参数时，它返回一个 1 元组的迭代器。没有参数时，它返回一个空的迭代器。

让我们看一个例子：

```py
# zip.grades.py
>>> grades = [18, 23, 30, 27]
>>> avgs = [22, 21, 29, 24]
>>> _(zip(avgs, grades))
[(22, 18), (21, 23), (29, 30), (24, 27)]
>>> _(map(lambda *a: a, avgs, grades))  # equivalent to zip
[(22, 18), (21, 23), (29, 30), (24, 27)]
```

在上面的代码中，我们将每个学生的平均值和最后一次考试的成绩进行了`zip`。注意使用`map`来复制`zip`是多么容易（示例的最后两条指令）。同样，在可视化结果时，我们必须使用我们的`_`别名。

`map`和`zip`的结合使用的一个简单例子可能是计算序列中每个元素的最大值，即每个序列的第一个元素的最大值，然后是第二个元素的最大值，依此类推：

```py
# maxims.py
>>> a = [5, 9, 2, 4, 7]
>>> b = [3, 7, 1, 9, 2]
>>> c = [6, 8, 0, 5, 3]
>>> maxs = map(lambda n: max(*n), zip(a, b, c))
>>> _(maxs)
[6, 9, 2, 9, 7]
```

注意计算三个序列的最大值是多么容易。当然，严格来说并不一定需要`zip`，我们可以使用`map`。有时候在展示一个简单的例子时，很难理解为什么使用某种技术可能是好的或坏的。我们忘记了我们并不总是能控制源代码，我们可能必须使用第三方库，而我们无法按照自己的意愿进行更改。因此，有不同的方法来处理数据真的很有帮助。

# 筛选

根据 Python 文档：

filter(function, iterable)从可迭代对象中构建一个迭代器，其中包含函数返回 True 的那些元素。可迭代对象可以是序列、支持迭代的容器，或者是迭代器。如果函数为 None，则假定为恒等函数，即删除可迭代对象中所有为假的元素。

让我们看一个非常快速的例子：

```py
# filter.py
>>> test = [2, 5, 8, 0, 0, 1, 0]
>>> _(filter(None, test))
[2, 5, 8, 1]
>>> _(filter(lambda x: x, test))  # equivalent to previous one
[2, 5, 8, 1]
>>> _(filter(lambda x: x > 4, test))  # keep only items > 4
[5, 8]
```

在上面的代码中，注意第二次调用`filter`等同于第一次调用。如果我们传递一个接受一个参数并返回参数本身的函数，只有那些为`True`的参数才会使函数返回`True`，因此这种行为与传递`None`完全相同。模仿一些内置的 Python 行为通常是一个很好的练习。当你成功时，你可以说你完全理解了 Python 在特定情况下的行为。

有了`map`，`zip`和`filter`（以及 Python 标准库中的其他几个函数），我们可以非常有效地处理序列。但这些函数并不是唯一的方法。所以让我们看看 Python 最好的特性之一：推导。

# 推导

推导是一种简洁的表示法，既对一组元素执行某些操作，又/或选择满足某些条件的子集。它们借鉴自函数式编程语言 Haskell（[`www.haskell.org/`](https://www.haskell.org/)），并且与迭代器和生成器一起为 Python 增添了函数式风味。

Python 为您提供不同类型的推导：`list`，`dict`和`set`。我们现在将集中在第一个上，然后解释另外两个将会很容易。

让我们从一个非常简单的例子开始。我想计算一个包含前 10 个自然数的平方的列表。你会怎么做？有几种等效的方法：

```py
# squares.map.py
# If you code like this you are not a Python dev! ;)
>>> squares = []
>>> for n in range(10):
...     squares.append(n ** 2)
...
>>> squares
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# This is better, one line, nice and readable
>>> squares = map(lambda n: n**2, range(10))
>>> _(squares)
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

前面的例子对你来说应该不是什么新鲜事。让我们看看如何使用`list`推导来实现相同的结果：

```py
# squares.comprehension.py
>>> [n ** 2 for n in range(10)]
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

就是这么简单。是不是很优雅？基本上我们在方括号内放了一个`for`循环。现在让我们过滤掉奇数平方。我将首先向你展示如何使用`map`和`filter`，然后再次使用`list`推导：

```py
# even.squares.py
# using map and filter
sq1 = list(
    map(lambda n: n ** 2, filter(lambda n: not n % 2, range(10)))
)
# equivalent, but using list comprehensions
sq2 = [n ** 2 for n in range(10) if not n % 2]

print(sq1, sq1 == sq2)  # prints: [0, 4, 16, 36, 64] True
```

我认为现在可读性的差异是明显的。列表推导读起来好多了。它几乎是英语：如果 n 是偶数，给我所有 0 到 9 之间的 n 的平方（n ** 2）。

根据 Python 文档：

列表推导由包含表达式的括号组成，后面跟着一个 for 子句，然后是零个或多个 for 或 if 子句。结果将是一个新列表，由在 for 和 if 子句的上下文中评估表达式得出。

# 嵌套推导

让我们看一个嵌套循环的例子。在处理算法时，经常需要使用两个占位符对序列进行迭代是很常见的。第一个占位符从左到右遍历整个序列。第二个也是如此，但它从第一个开始，而不是从 0 开始。这个概念是为了测试所有对而不重复。让我们看看经典的`for`循环等价：

```py
# pairs.for.loop.py
items = 'ABCD'
pairs = []

for a in range(len(items)):
    for b in range(a, len(items)):
        pairs.append((items[a], items[b]))
```

如果你在最后打印出对，你会得到：

```py
$ python pairs.for.loop.py
[('A', 'A'), ('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'B'), ('B', 'C'), ('B', 'D'), ('C', 'C'), ('C', 'D'), ('D', 'D')]
```

所有具有相同字母的元组都是`b`与`a`处于相同位置的元组。现在，让我们看看如何将其转换为`list`推导：

```py
# pairs.list.comprehension.py
items = 'ABCD'
pairs = [(items[a], items[b])
    for a in range(len(items)) for b in range(a, len(items))]
```

这个版本只有两行长，但实现了相同的结果。请注意，在这种特殊情况下，因为`for`循环在`b`上有一个对`a`的依赖，所以它必须在推导中跟在`a`上的`for`循环之后。如果你交换它们，你会得到一个名称错误。

# 过滤推导

我们可以对推导应用过滤。让我们首先用`filter`来做。让我们找出所有勾股数的短边小于 10 的三元组。显然，我们不想测试两次组合，因此我们将使用与我们在上一个例子中看到的类似的技巧：

```py
# pythagorean.triple.py
from math import sqrt
# this will generate all possible pairs
mx = 10
triples = [(a, b, sqrt(a**2 + b**2))
    for a in range(1, mx) for b in range(a, mx)]
# this will filter out all non pythagorean triples
triples = list(
    filter(lambda triple: triple[2].is_integer(), triples))

print(triples)  # prints: [(3, 4, 5.0), (6, 8, 10.0)]
```

勾股数是满足整数方程 a² + b² = c²的整数三元组（a，b，c）。

在前面的代码中，我们生成了一个*三元组*列表`triples`。每个元组包含两个整数（腿）和勾股定理三角形的斜边，其腿是元组中的前两个数字。例如，当`a`为`3`，`b`为`4`时，元组将是`(3, 4, 5.0)`，当`a`为`5`，`b`为`7`时，元组将是`(5, 7, 8.602325267042627)`。

在完成所有`triples`之后，我们需要过滤掉所有没有整数斜边的三元组。为了做到这一点，我们基于`float_number.is_integer()`为`True`进行过滤。这意味着在我之前向您展示的两个示例元组中，具有`5.0`斜边的元组将被保留，而具有`8.602325267042627`斜边的元组将被丢弃。

这很好，但我不喜欢三元组有两个整数和一个浮点数。它们应该都是整数，所以让我们使用`map`来修复这个问题：

```py
# pythagorean.triple.int.py
from math import sqrt
mx = 10
triples = [(a, b, sqrt(a**2 + b**2))
    for a in range(1, mx) for b in range(a, mx)]
triples = filter(lambda triple: triple[2].is_integer(), triples)
# this will make the third number in the tuples integer
triples = list(
    map(lambda triple: triple[:2] + (int(triple[2]), ), triples))

print(triples)  # prints: [(3, 4, 5), (6, 8, 10)]
```

注意我们添加的步骤。我们取`triples`中的每个元素，并对其进行切片，仅取其中的前两个元素。然后，我们将切片与一个一元组连接起来，在其中放入我们不喜欢的那个浮点数的整数版本。看起来像是很多工作，对吧？确实是。让我们看看如何使用`list`推导来完成所有这些工作：

```py
# pythagorean.triple.comprehension.py
from math import sqrt
# this step is the same as before
mx = 10
triples = [(a, b, sqrt(a**2 + b**2))
    for a in range(1, mx) for b in range(a, mx)]
# here we combine filter and map in one CLEAN list comprehension
triples = [(a, b, int(c)) for a, b, c in triples if c.is_integer()]
print(triples)  # prints: [(3, 4, 5), (6, 8, 10)]
```

我知道。这样会好得多，不是吗？它干净、可读、更短。换句话说，它是优雅的。

我在这里走得很快，就像在第四章的*摘要*中预期的那样，*函数，代码的构建块*。您在玩这个代码吗？如果没有，我建议您这样做。非常重要的是，您要玩耍，打破事物，改变事物，看看会发生什么。确保您清楚地了解发生了什么。您想成为一个忍者，对吧？

# dict 推导

字典和`set`推导的工作方式与列表推导完全相同，只是语法上有一点不同。以下示例足以解释您需要了解的所有内容：

```py
# dictionary.comprehensions.py
from string import ascii_lowercase
lettermap = dict((c, k) for k, c in enumerate(ascii_lowercase, 1))
```

如果打印`lettermap`，您将看到以下内容（我省略了中间结果，您会明白的）：

```py
$ python dictionary.comprehensions.py
{'a': 1,
 'b': 2,
 ...
 'y': 25,
 'z': 26}
```

在前面的代码中发生的是，我们正在用推导（技术上是生成器表达式，我们稍后会看到）向`dict`构造函数提供数据。我们告诉`dict`构造函数从推导中的每个元组中制作*键*/*值*对。我们使用`enumerate`列举所有小写 ASCII 字母的序列，从`1`开始。小菜一碟。还有另一种做同样事情的方法，更接近其他字典语法：

```py
lettermap = {c: k for k, c in enumerate(ascii_lowercase, 1)} 
```

它确实做了完全相同的事情，只是语法略有不同，更突出了*键：值*部分。

字典不允许键中有重复，如下例所示：

```py
# dictionary.comprehensions.duplicates.py
word = 'Hello'
swaps = {c: c.swapcase() for c in word}
print(swaps)  # prints: {'H': 'h', 'e': 'E', 'l': 'L', 'o': 'O'}
```

我们创建一个字典，其中键是`'Hello'`字符串中的字母，值是相同的字母，但大小写不同。请注意只有一个`'l': 'L'`对。构造函数不会抱怨，它只是将重复的键重新分配给最新的值。让我们通过另一个例子来更清楚地说明这一点；让我们为每个键分配其在字符串中的位置：

```py
# dictionary.comprehensions.positions.py
word = 'Hello'
positions = {c: k for k, c in enumerate(word)}
print(positions)  # prints: {'H': 0, 'e': 1, 'l': 3, 'o': 4}
```

请注意与字母`'l'`关联的值：`3`。`'l': 2`对不在那里；它已被`'l': 3`覆盖。

# set 推导

`set`推导非常类似于列表和字典推导。Python 允许使用`set()`构造函数，或显式的`{}`语法。让我们看一个快速的例子：

```py
# set.comprehensions.py
word = 'Hello'
letters1 = set(c for c in word)
letters2 = {c for c in word}
print(letters1)  # prints: {'H', 'o', 'e', 'l'}
print(letters1 == letters2)  # prints: True
```

请注意，对于`set`推导和字典推导，不允许重复，因此生成的集合只有四个字母。还要注意，分配给`letters1`和`letters2`的表达式产生了等效的集合。

用于创建`letters2`的语法与用于创建字典推导的语法非常相似。您只能通过字典需要使用冒号分隔的键和值来区分它们，而集合则不需要。

# 生成器

**生成器**是 Python 赋予我们的非常强大的工具。它们基于*迭代*的概念，正如我们之前所说的，它们允许结合优雅和高效的编码模式。

生成器有两种类型：

+   **生成器函数**：这些与常规函数非常相似，但是它们不是通过返回语句返回结果，而是使用 yield，这使它们能够在每次调用之间暂停和恢复它们的状态。

+   **生成器表达式**：这些与我们在本章中看到的`list`推导非常相似，但是它们不是返回一个列表，而是返回一个逐个产生结果的对象。

# 生成器函数

生成器函数在所有方面都像常规函数一样，只有一个区别。它们不是一次性收集结果并返回它们，而是在每次调用`next`时自动转换为产生结果的迭代器。

这一切都是非常理论的，所以让我们清楚地说明为什么这样的机制是如此强大，然后让我们看一个例子。

假设我让你大声数数从 1 数到 1,000,000。你开始了，然后在某个时候我让你停下来。过了一段时间，我让你继续。在这一点上，你需要记住能够正确恢复的最小信息是什么？嗯，你需要记住你最后一个叫的数字。如果我在 31,415 后停止了你，你就会继续 31,416，依此类推。

重点是，你不需要记住 31,415 之前说的所有数字，也不需要它们被写在某个地方。嗯，你可能不知道，但你已经像一个生成器一样行为了！

仔细看一下以下代码：

```py
# first.n.squares.py
def get_squares(n): # classic function approach
    return [x ** 2 for x in range(n)]
print(get_squares(10))

def get_squares_gen(n):  # generator approach
    for x in range(n):
        yield x ** 2  # we yield, we don't return
print(list(get_squares_gen(10)))
```

两个`print`语句的结果将是相同的：`[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]`。但是这两个函数之间有很大的区别。`get_squares`是一个经典函数，它收集[0，*n*)范围内所有数字的平方，并将其返回为列表。另一方面，`get_squares_gen`是一个生成器，行为非常不同。每当解释器到达`yield`行时，它的执行就会被暂停。这些`print`语句返回相同结果的唯一原因是因为我们将`get_squares_gen`传递给`list`构造函数，它通过请求下一个元素直到引发`StopIteration`来完全耗尽生成器。让我们详细看一下：

```py
# first.n.squares.manual.py
def get_squares_gen(n):
    for x in range(n):
        yield x ** 2

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

在前面的代码中，每次我们在生成器对象上调用`next`时，要么启动它（第一个`next`），要么使它从上次暂停的地方恢复（任何其他`next`）。

第一次在它上面调用`next`时，我们得到`0`，这是`0`的平方，然后是`1`，然后是`4`，然后是`9`，由于`for`循环在那之后停止了（`n`是`4`），然后生成器自然结束了。经典函数在那一点上只会返回`None`，但为了符合迭代协议，生成器将会引发`StopIteration`异常。

这解释了`for`循环的工作原理。当你调用`for k in range(n)`时，在幕后发生的是`for`循环从`range(n)`中获取一个迭代器，并开始在其上调用`next`，直到引发`StopIteration`，这告诉`for`循环迭代已经结束。

Python 的每个迭代方面内置了这种行为，这使得生成器更加强大，因为一旦我们编写它们，我们就能够将它们插入到我们想要的任何迭代机制中。

此时，你可能会问自己为什么要使用生成器而不是普通函数。好吧，本章的标题应该暗示了答案。稍后我会谈论性能，所以现在让我们集中在另一个方面：有时生成器允许你做一些用简单列表无法做到的事情。例如，假设你想分析一个序列的所有排列。如果序列的长度为*N*，那么它的排列数就是*N!*。这意味着如果序列长度为 10 个元素，排列数就是 3,628,800。但是 20 个元素的序列将有 2,432,902,008,176,640,000 个排列。它们呈阶乘增长。

现在想象一下，你有一个经典函数，它试图计算所有的排列，把它们放在一个列表中，并返回给你。对于 10 个元素，可能需要几十秒，但对于 20 个元素，根本不可能完成。

另一方面，一个生成器函数将能够开始计算并返回第一个排列，然后是第二个，依此类推。当然你没有时间解析它们所有，因为太多了，但至少你能够处理其中的一些。

还记得我们在谈论`for`循环中的`break`语句吗？当我们找到一个能整除*候选素数*的数时，我们就打破了循环，没有必要继续下去了。

有时情况完全相同，只是你需要迭代的数据量太大，无法将其全部保存在列表中。在这种情况下，生成器是非常宝贵的：它们使得原本不可能的事情成为可能。

因此，为了节省内存（和时间），尽可能使用生成器函数。

值得注意的是，你可以在生成器函数中使用 return 语句。它将产生一个`StopIteration`异常被引发，有效地结束迭代。这是非常重要的。如果`return`语句实际上使函数返回了什么东西，它将打破迭代协议。Python 的一致性防止了这种情况，并且在编码时为我们提供了极大的便利。让我们看一个快速的例子：

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

前面的代码产生了等比级数的所有项，*a*，*aq*，*aq²*，*aq³*，.... 当级数产生一个大于`100000`的项时，生成器就会停止（使用`return`语句）。 运行代码会产生以下结果：

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

下一个项本来会是`156250`，这太大了。

说到`StopIteration`，从 Python 3.5 开始，生成器中异常处理的方式已经发生了变化。在这一点上理解这种变化的影响可能要求你付出太多，所以只需知道你可以在 PEP 479 中阅读有关它的所有内容即可（[`legacy.python.org/dev/peps/pep-0479/`](https://legacy.python.org/dev/peps/pep-0479/)）。

# 超越 next

在本章的开头，我告诉过你生成器对象是基于迭代协议的。我们将在第六章中看到一个完整的例子，说明如何编写自定义的迭代器/可迭代对象。现在，我只是希望你理解`next()`是如何工作的。

当你调用`next(generator)`时，你调用了`generator.__next__()`方法。记住，**方法**只是属于对象的函数，而 Python 中的对象可以有特殊的方法。`__next__()`只是其中之一，它的目的是返回迭代的下一个元素，或者在迭代结束时引发`StopIteration`，并且没有更多的元素可以返回。

如果你还记得，在 Python 中，对象的特殊方法也被称为**魔术方法**，或者**dunder**（来自“双下划线”）**方法**。

当我们编写一个生成器函数时，Python 会自动将其转换为一个与迭代器非常相似的对象，当我们调用`next(generator)`时，该调用会转换为`generator.__next__()`。让我们重新讨论一下关于生成平方数的先前示例：

```py
# first.n.squares.manual.method.py
def get_squares_gen(n):
    for x in range(n):
        yield x ** 2

squares = get_squares_gen(3)
print(squares.__next__())  # prints: 0
print(squares.__next__())  # prints: 1
print(squares.__next__())  # prints: 4
# the following raises StopIteration, the generator is exhausted,
# any further call to next will keep raising StopIteration
```

结果与前面的示例完全相同，只是这次我们直接调用`squares.__next__()`，而不是使用`next(squares)`代理调用。

生成器对象还有另外三种方法，允许我们控制它们的行为：`send`，`throw`和`close`。`send`允许我们向生成器对象发送一个值，而`throw`和`close`分别允许我们在生成器内部引发异常并关闭它。它们的使用非常高级，我不会在这里详细介绍它们，但我想在`send`上花几句话，举个简单的例子：

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

前面的迭代器创建了一个将永远运行的生成器对象。您可以不断调用它，它永远不会停止。或者，您可以将其放入`for`循环中，例如，`for n in counter(): ...`，它也将永远运行。但是，如果您想在某个时刻停止它怎么办？一种解决方案是使用变量来控制`while`循环。例如：

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

这样就可以了。我们从`stop = False`开始，直到我们将其更改为`True`，生成器将像以前一样继续运行。然而，一旦我们将`stop`更改为`True`，`while`循环将退出，并且下一次调用将引发`StopIteration`异常。这个技巧有效，但我不喜欢它。我们依赖于一个外部变量，这可能会导致问题：如果另一个函数改变了`stop`会怎么样？此外，代码是分散的。简而言之，这还不够好。

我们可以通过使用`generator.send()`来改进它。当我们调用`generator.send()`时，我们向`send`提供的值将被传递给生成器，执行将恢复，我们可以通过`yield`表达式获取它。用文字解释这一切都很复杂，所以让我们看一个例子：

```py
# gen.send.py
def counter(start=0):
    n = start
    while True:
        result = yield n             # A
        print(type(result), result)  # B
        if result == 'Q':
            break
        n += 1

c = counter()
print(next(c))         # C
print(c.send('Wow!'))  # D
print(next(c))         # E
print(c.send('Q'))     # F
```

执行上述代码会产生以下结果：

```py
$ python gen.send.py
0
<class 'str'> Wow!
1
<class 'NoneType'> None
2
<class 'str'> Q
Traceback (most recent call last):
 File "gen.send.py", line 14, in <module>
 print(c.send('Q')) # F
StopIteration
```

我认为逐行阅读这段代码是值得的，就好像我们在执行它一样，看看我们是否能理解发生了什么。

我们通过调用`next`(`#C`)开始生成器执行。在生成器中，`n`被设置为与`start`相同的值。进入`while`循环，执行停止（`#A`），`n`（`0`）被返回给调用者。`0`被打印在控制台上。

然后我们调用`send`(`#D`)，执行恢复，`result`被设置为`'Wow!'`（仍然是`#A`），然后它的类型和值被打印在控制台上（`#B`）。`result`不是`'Q'`，因此`n`增加了`1`，执行返回到`while`条件，这时，`True`被评估为`True`（这不难猜到，对吧？）。另一个循环开始，执行再次停止（`#A`），`n`（`1`）被返回给调用者。`1`被打印在控制台上。

在这一点上，我们调用`next`(`#E`)，执行再次恢复（`#A`），因为我们没有明确向生成器发送任何内容，Python 的行为与不使用`return`语句的函数完全相同；`yield n`表达式（`#A`）返回`None`。因此，`result`被设置为`None`，其类型和值再次被打印在控制台上（`#B`）。执行继续，`result`不是`'Q'`，所以`n`增加了`1`，我们再次开始另一个循环。执行再次停止（`#A`），`n`（`2`）被返回给调用者。`2`被打印在控制台上。

现在到了大结局：我们再次调用`send`（`#F`），但这次我们传入了`'Q'`，因此当执行恢复时，`result`被设置为`'Q'`（`#A`）。它的类型和值被打印在控制台上（`#B`），最后`if`子句评估为`True`，`while`循环被`break`语句停止。生成器自然终止，这意味着会引发`StopIteration`异常。您可以在控制台上看到它的回溯打印在最后几行上。

这一开始并不容易理解，所以如果对您来说不清楚，不要气馁。您可以继续阅读，然后过一段时间再回到这个例子。

使用`send`允许有趣的模式，值得注意的是`send`也可以用于启动生成器的执行（只要您用`None`调用它）。

# `yield from`表达式

另一个有趣的构造是`yield from`表达式。这个表达式允许您从子迭代器中产生值。它的使用允许相当高级的模式，所以让我们快速看一个非常快速的例子：

```py
# gen.yield.for.py def print_squares(start, end):
    for n in range(start, end):
        yield n ** 2

for n in print_squares(2, 5):
    print(n)
```

前面的代码在控制台上打印出数字`4`，`9`，`16`（分别在不同的行上）。到现在为止，我希望您能够自己理解它，但让我们快速回顾一下发生了什么。函数外部的`for`循环从`print_squares(2, 5)`获取一个迭代器，并在其上调用`next`，直到迭代结束。每次调用生成器时，执行都会被暂停（稍后恢复）在`yield n ** 2`上，它返回当前`n`的平方。让我们看看如何利用`yield from`表达式改变这段代码：

```py
# gen.yield.from.py
def print_squares(start, end):
    yield from (n ** 2 for n in range(start, end))

for n in print_squares(2, 5):
    print(n)
```

这段代码产生了相同的结果，但是您可以看到`yield from`实际上正在运行一个子迭代器`(n ** 2 ...)`。`yield from`表达式将子迭代器产生的每个值返回给调用者。它更短，阅读起来更好。

# 生成器表达式

现在让我们谈谈其他一次生成值的技术。

语法与`list`推导完全相同，只是，不是用方括号包装推导，而是用圆括号包装。这就是所谓的**生成器表达式**。

通常，生成器表达式的行为类似于等效的`list`推导，但有一件非常重要的事情要记住：生成器只允许一次迭代，然后它们将被耗尽。让我们看一个例子：

```py
# generator.expressions.py
>>> cubes = [k**3 for k in range(10)]  # regular list
>>> cubes
[0, 1, 8, 27, 64, 125, 216, 343, 512, 729]
>>> type(cubes)
<class 'list'>
>>> cubes_gen = (k**3 for k in range(10))  # create as generator
>>> cubes_gen
<generator object <genexpr> at 0x103fb5a98>
>>> type(cubes_gen)
<class 'generator'>
>>> _(cubes_gen)  # this will exhaust the generator
[0, 1, 8, 27, 64, 125, 216, 343, 512, 729]
>>> _(cubes_gen)  # nothing more to give
[]
```

看看生成器表达式被创建并分配名称`cubes_gen`的行。您可以看到它是一个生成器对象。为了看到它的元素，我们可以使用`for`循环，手动调用`next`，或者简单地将其传递给`list`构造函数，这就是我所做的（记住我使用`_`作为别名）。

请注意，一旦生成器被耗尽，就没有办法再次从中恢复相同的元素。如果我们想要再次从头开始使用它，我们需要重新创建它。

在接下来的几个例子中，让我们看看如何使用生成器表达式复制`map`和`filter`： 

```py
# gen.map.py
def adder(*n):
    return sum(n)
s1 = sum(map(lambda *n: adder(*n), range(100), range(1, 101)))
s2 = sum(adder(*n) for n in zip(range(100), range(1, 101)))
```

在前面的例子中，`s1`和`s2`完全相同：它们是`adder(0, 1), adder(1, 2), adder(2, 3)`的和，依此类推，这对应于`sum(1, 3, 5, ...)`。尽管语法不同，但我发现生成器表达式更易读：

```py
# gen.filter.py
cubes = [x**3 for x in range(10)]

odd_cubes1 = filter(lambda cube: cube % 2, cubes)
odd_cubes2 = (cube for cube in cubes if cube % 2)
```

在前面的例子中，`odd_cubes1`和`odd_cubes2`是相同的：它们生成奇数立方体的序列。当事情变得有点复杂时，我再次更喜欢生成器语法。这应该在事情变得有点复杂时显而易见：

```py
# gen.map.filter.py
N = 20
cubes1 = map(
    lambda n: (n, n**3),
    filter(lambda n: n % 3 == 0 or n % 5 == 0, range(N))
)
cubes2 = (
    (n, n**3) for n in range(N) if n % 3 == 0 or n % 5 == 0)
```

前面的代码创建了两个生成器，`cubes1`和`cubes2`。它们完全相同，当`n`是`3`或`5`的倍数时返回两个元组（*n，n³*）。

如果打印列表（`cubes1`），您会得到：`[(0, 0), (3, 27), (5, 125), (6, 216), (9, 729), (10, 1000), (12, 1728), (15, 3375), (18, 5832)]`。

看看生成器表达式读起来好多了？当事情非常简单时，这可能是值得商榷的，但是一旦你开始嵌套函数一点，就像我们在这个例子中所做的那样，生成器语法的优越性就显而易见了。它更短，更简单，更优雅。

现在，让我问你一个问题——以下代码的区别是什么：

```py
# sum.example.py
s1 = sum([n**2 for n in range(10**6)])
s2 = sum((n**2 for n in range(10**6)))
s3 = sum(n**2 for n in range(10**6))
```

严格来说，它们都产生相同的总和。获取`s2`和`s3`的表达式完全相同，因为`s2`中的括号是多余的。它们都是`sum`函数中的生成器表达式。然而，获取`s1`的表达式是不同的。在`sum`中，我们找到了一个`list`理解。这意味着为了计算`s1`，`sum`函数必须在列表上调用一百万次`next`。

你看到我们在浪费时间和内存吗？在`sum`可以开始在列表上调用`next`之前，列表需要被创建，这是一种浪费时间和空间。对于`sum`来说，在一个简单的生成器表达式上调用`next`要好得多。没有必要将`range(10**6)`中的所有数字存储在列表中。

因此，*在编写表达式时要注意额外的括号*：有时很容易忽略这些细节，这使得我们的代码非常不同。如果你不相信我，看看下面的代码：

```py
# sum.example.2.py
s = sum([n**2 for n in range(10**8)])  # this is killed
# s = sum(n**2 for n in range(10**8))    # this succeeds
print(s)  # prints: 333333328333333350000000
```

尝试运行前面的例子。如果我在我的旧 Linux 框上运行第一行，内存为 8GB，这就是我得到的：

```py
$ python sum.example.2.py
Killed  
```

另一方面，如果我注释掉第一行，并取消注释第二行，这就是结果：

```py
$ python sum.example.2.py
333333328333333350000000  
```

甜蜜的生成器表达式。两行之间的区别在于，在第一行中，必须先制作一个前一亿个数字的平方的列表，然后才能将它们相加。那个列表很大，我们的内存用完了（至少，我的内存用完了，如果你的内存没有用完，试试更大的数字），因此 Python 为我们终止了进程。悲伤的脸。

但是当我们去掉方括号时，我们不再有一个列表。`sum`函数接收`0`，`1`，`4`，`9`，直到最后一个，然后将它们相加。没有问题，开心脸。

# 一些性能考虑

因此，我们已经看到了实现相同结果的许多不同方法。我们可以使用`map`，`zip`和`filter`的任何组合，或者选择使用理解，或者可能选择使用生成器，无论是函数还是表达式。我们甚至可以决定使用`for`循环；当要应用于每个运行参数的逻辑不简单时，它们可能是最佳选择。

除了可读性问题之外，让我们谈谈性能。在性能方面，通常有两个因素起着重要作用：**空间**和**时间**。

空间意味着数据结构要占用的内存大小。选择的最佳方法是问自己是否真的需要一个列表（或元组），或者一个简单的生成器函数是否同样有效。如果答案是肯定的，那就选择生成器，它会节省很多空间。对于函数也是一样；如果你实际上不需要它们返回一个列表或元组，那么你也可以将它们转换为生成器函数。

有时，你将不得不使用列表（或元组），例如有一些算法使用多个指针扫描序列，或者可能多次运行序列。生成器函数（或表达式）只能迭代一次，然后就用完了，所以在这些情况下，它不是正确的选择。

时间比空间更难，因为它取决于更多的变量，因此不可能绝对肯定地说*X 比 Y 更快*对于所有情况。然而，基于今天在 Python 上运行的测试，我们可以说，平均而言，`map`表现出与`list`理解和生成器表达式类似的性能，而`for`循环一直较慢。

为了充分理解这些陈述背后的推理，我们需要了解 Python 的工作原理，这有点超出了本书的范围，因为它在技术细节上太复杂。让我们只说`map`和`list`理解在解释器内以 C 语言速度运行，而 Python `for`循环作为 Python 虚拟机内的 Python 字节码运行，通常要慢得多。

Python 有几种不同的实现。最初的，也是最常见的一个是 CPython ([`github.com/python/cpython`](https://github.com/python/cpython))，它是用 C 语言编写的。C 语言是今天仍然使用的最强大和流行的编程语言之一。

我们来做一个小练习，试着找出我所说的是否准确？我将编写一小段代码，收集`divmod(a, b)`的结果，对于一定的整数对`(a, b)`。我将使用`time`模块中的`time`函数来计算我将执行的操作的经过时间：

```py
# performances.py
from time import time
mx = 5000

t = time()  # start time for the for loop
floop = []
for a in range(1, mx):
    for b in range(a, mx):
        floop.append(divmod(a, b))
print('for loop: {:.4f} s'.format(time() - t))  # elapsed time

t = time()  # start time for the list comprehension
compr = [
    divmod(a, b) for a in range(1, mx) for b in range(a, mx)]
print('list comprehension: {:.4f} s'.format(time() - t))

t = time()  # start time for the generator expression
gener = list(
    divmod(a, b) for a in range(1, mx) for b in range(a, mx))
print('generator expression: {:.4f} s'.format(time() - t))
```

你可以看到，我们创建了三个列表：`floop`、`compr`和`gener`。运行代码会产生以下结果：

```py
$ python performances.py
for loop: 4.4814 s
list comprehension: 3.0210 s
generator expression: 3.4334 s
```

`list`理解运行时间约为`for`循环时间的 67%。这令人印象深刻。生成器表达式接近这个时间，约为`for`循环时间的 77%。生成器表达式较慢的原因是我们需要将其提供给`list()`构造函数，这与纯粹的`list`理解相比有更多的开销。如果我不必保留这些计算的结果，生成器可能是更合适的选择。

有趣的是，在`for`循环的主体中，我们正在向列表中添加数据。这意味着 Python 在幕后做着工作，不时地调整大小，为要添加的项目分配空间。我猜想创建一个零列表，并简单地用结果填充它，可能会加快`for`循环的速度，但我错了。你自己检查一下，你只需要预分配`mx * (mx - 1) // 2`个元素。

让我们看一个类似的例子，比较一下`for`循环和`map`调用：

```py
# performances.map.py
from time import time
mx = 2 * 10 ** 7

t = time()
absloop = []
for n in range(mx):
    absloop.append(abs(n))
print('for loop: {:.4f} s'.format(time() - t))

t = time()
abslist = [abs(n) for n in range(mx)]
print('list comprehension: {:.4f} s'.format(time() - t))

t = time()
absmap = list(map(abs, range(mx)))
print('map: {:.4f} s'.format(time() - t))
```

这段代码在概念上与前面的例子非常相似。唯一改变的是我们应用了`abs`函数而不是`divmod`，并且我们只有一个循环而不是两个嵌套的循环。执行后得到以下结果：

```py
$ python performances.map.py
for loop: 3.8948 s
list comprehension: 1.8594 s
map: 1.1548 s
```

而`map`赢得了比赛：约为`list`理解时间的 62%，`for`循环时间的 30%。这些结果可能会有所不同，因为各种因素，如操作系统和 Python 版本。但总的来说，我认为这些结果足够好，可以让我们对编写性能代码有一个概念。

尽管有一些个案的小差异，很明显`for`循环选项是最慢的，所以让我们看看为什么我们仍然想要使用它。

# 不要过度使用理解和生成器

我们已经看到了`list`理解和生成器表达式有多么强大。它们确实如此，不要误会我的意思，但当我处理它们时的感觉是，它们的复杂性呈指数增长。你尝试在一个单一的理解或生成器表达式中做的越多，它就越难以阅读、理解，因此也就越难以维护或更改。

如果你再次查看 Python 之禅，有几行我认为值得在处理优化代码时牢记：

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

理解和生成器表达式比较隐式而不是显式，可能相当难以阅读和理解，也很难解释。有时你必须使用由内而外的技术来分解它们，以理解发生了什么。

举个例子，让我们再谈谈毕达哥拉斯三元组。只是提醒一下，毕达哥拉斯三元组是一组正整数元组(*a*, *b*, *c*)，使得*a² + b² = c²*。

我们在*过滤理解*部分看到了如何计算它们，但我们以一种非常低效的方式进行了，因为我们正在扫描所有低于某个阈值的数字对，计算斜边，并过滤掉那些没有产生三元组的数字对。

获得勾股数三元组的更好方法是直接生成它们。有许多不同的公式可以用来做到这一点，我们将使用**欧几里得公式**。

这个公式表明，任何三元组(*a*，*b*，*c*)，其中*a = m² - n²*，*b = 2mn*，*c = m² + n²*，*m*和*n*是正整数，满足*m > n*，都是勾股数三元组。例如，当*m = 2*，*n = 1*时，我们找到了最小的三元组：(*3*，*4*，*5*)。

然而，有一个问题：考虑一下三元组(*6*，*8*，*10*)，它就像(*3*，*4*，*5*)一样，只是所有数字都乘以*2*。这个三元组肯定是勾股数三元组，因为*6² + 8² = 10²*，但我们可以通过简单地将其每个元素乘以*2*来从(*3*，*4*，*5*)派生出它。对于所有可以写成(*3k*，*4k*，*5k*)的三元组，其中*k*是大于*1*的正整数，情况也是如此。

不能通过将另一个三元组的元素乘以某个因子*k*获得的三元组称为**原始**。另一种陈述这一点的方式是：如果三元组的三个元素是**互质**的，那么这个三元组就是原始的。当两个数在它们的除数中没有共享任何质因数时，它们就是互质的，也就是说，它们的**最大公约数**（**GCD**）是*1*。例如，3 和 5 是互质的，而 3 和 6 不是，因为它们都可以被 3 整除。

因此，欧几里得公式告诉我们，如果*m*和*n*是互质的，并且*m - n*是奇数，那么它们生成的三元组是原始的。在下面的例子中，我们将编写一个生成器表达式，计算所有原始的勾股数三元组，其斜边(*c*)小于或等于某个整数*N*。这意味着我们希望所有满足*m² + n² ≤ N*的三元组。当*n*为*1*时，公式如下：*m² ≤ N - 1*，这意味着我们可以用*m ≤ N^(1/2)*的上限来近似计算。

因此，总结一下：*m*必须大于*n*，它们也必须互质，它们的差异*m - n*必须是奇数。此外，为了避免无用的计算，我们将*m*的上限设定为*floor(sqrt(N)) + 1*。

实数*x*的`floor`函数给出最大整数*n*，使得*n < x*，例如，*floor(3.8) = 3*，*floor(13.1) = 13*。取*floor(sqrt(N)) + 1*意味着取*N*的平方根的整数部分，并添加一个最小的边距，以确保我们不会错过任何数字。

让我们一步一步地将所有这些放入代码中。让我们首先编写一个使用**欧几里得算法**的简单`gcd`函数：

```py
# functions.py
def gcd(a, b):
    """Calculate the Greatest Common Divisor of (a, b). """
    while b != 0:
        a, b = b, a % b
    return a
```

欧几里得算法的解释可以在网上找到，所以我不会在这里花时间谈论它；我们需要专注于生成器表达式。下一步是利用之前收集的知识来生成一个原始勾股数三元组的列表：

```py
# pythagorean.triple.generation.py
from functions import gcd
N = 50

triples = sorted(                                    # 1
    ((a, b, c) for a, b, c in (                      # 2
        ((m**2 - n**2), (2 * m * n), (m**2 + n**2))  # 3
        for m in range(1, int(N**.5) + 1)            # 4
        for n in range(1, m)                         # 5
        if (m - n) % 2 and gcd(m, n) == 1            # 6
    ) if c <= N), key=lambda *triple: sum(*triple)   # 7
)
```

这就是了。它不容易阅读，所以让我们逐行进行解释。在`#3`处，我们开始一个生成器表达式，用于创建三元组。从`#4`和`#5`可以看出，我们在*[1，M]*中循环*m*，其中*M*是*sqrt(N)*的整数部分，再加上*1*。另一方面，*n*在*[1，m)*中循环，以遵守*m > n*的规则。值得注意的是我如何计算*sqrt(N)*，即`N**.5`，这只是另一种我想向你展示的方法。

在`＃6`，您可以看到使三元组原始的过滤条件：当`(m - n)`为奇数时，`(m - n)％2`的值为`True`，而`gcd(m, n) == 1`表示`m`和`n`是互质的。有了这些条件，我们知道三元组将是原始的。这照顾了最内层的生成器表达式。最外层的生成器表达式从`＃2`开始，结束于`＃7`。我们取(*a*, *b*, *c*)在(...最内层生成器...)中，使得`c <= N`。

最后，在`＃1`我们应用排序，以按顺序呈现列表。在最外层生成器表达式关闭后的`＃7`处，您可以看到我们指定排序键为和的总和*a + b + c*。这只是我的个人偏好，没有数学原因。

那么，你觉得呢？阅读起来简单吗？我不这么认为。相信我，这仍然是一个简单的例子；在我的职业生涯中，我见过更糟糕的情况。这种代码难以理解、调试和修改。它不应该出现在专业环境中。

所以，让我们看看是否可以将这段代码重写成更易读的形式：

```py
# pythagorean.triple.generation.for.py
from functions import gcd

def gen_triples(N):
    for m in range(1, int(N**.5) + 1):                  # 1
        for n in range(1, m):                           # 2
            if (m - n) % 2 and gcd(m, n) == 1:          # 3
                c = m**2 + n**2                         # 4
                if c <= N:                              # 5
                    a = m**2 - n**2                     # 6
                    b = 2 * m * n                       # 7
                    yield (a, b, c)                     # 8

triples = sorted(
    gen_triples(50), key=lambda *triple: sum(*triple))  # 9
```

这好多了。让我们逐行看一下。你会看到它有多容易理解。

我们从`＃1`和`＃2`开始循环，方式与之前的示例中的循环方式完全相同。在第`＃3`行，我们对原始三元组进行了过滤。在第`＃4`行，我们有了一点偏离之前的做法：我们计算了`c`，在第`＃5`行，我们对`c`小于或等于`N`进行了过滤。只有当`c`满足这个条件时，我们才计算`a`和`b`，并产生结果的元组。尽可能延迟所有计算总是很好的，这样我们就不会浪费时间和 CPU。在最后一行，我们使用了与生成器表达式示例中相同的键进行排序。

希望你同意，这个例子更容易理解。我向你保证，如果有一天你不得不修改这段代码，你会发现修改这个代码很容易，而修改另一个版本将需要更长的时间（而且容易出错）。

如果打印两个示例的结果（它们是相同的），你会得到这个：

```py
[(3, 4, 5), (5, 12, 13), (15, 8, 17), (7, 24, 25), (21, 20, 29), (35, 12, 37), (9, 40, 41)]  
```

故事的寓意是，尽量使用理解和生成器表达式，但如果代码开始变得复杂，难以修改或阅读，你可能需要将其重构为更易读的形式。你的同事会感谢你。

# 名称本地化

既然我们熟悉了所有类型的理解和生成器表达式，让我们谈谈它们内部的名称本地化。Python 3.*在所有四种理解形式中都将循环变量本地化：`list`、`dict`、`set`和生成器表达式。这种行为与`for`循环的行为不同。让我们看一个简单的例子来展示所有情况：

```py
# scopes.py
A = 100
ex1 = [A for A in range(5)]
print(A)  # prints: 100

ex2 = list(A for A in range(5))
print(A)  # prints: 100

ex3 = dict((A, 2 * A) for A in range(5))
print(A)  # prints: 100

ex4 = set(A for A in range(5))
print(A)  # prints: 100

s = 0
for A in range(5):
    s += A
print(A)  # prints: 4
```

在前面的代码中，我们声明了一个全局名称`A = 100`，然后我们使用了四种理解方式：`list`、生成器表达式、字典和`set`。它们都没有改变全局名称`A`。相反，您可以在最后看到`for`循环修改了它。最后的打印语句打印出`4`。

让我们看看如果没有`A`会发生什么：

```py
# scopes.noglobal.py
ex1 = [A for A in range(5)]
print(A)  # breaks: NameError: name 'A' is not defined
```

前面的代码可以使用任何四种理解方式来完成相同的工作。运行第一行后，`A`在全局命名空间中未定义。再次，`for`循环的行为不同：

```py
# scopes.for.py
s = 0
for A in range(5):
    s += A
print(A) # prints: 4
print(globals())
```

前面的代码表明，在`for`循环之后，如果循环变量在之前没有定义，我们可以在全局框架中找到它。为了确保这一点，让我们调用`globals()`内置函数来一探究竟：

```py
$ python scopes.for.py
4
{'__name__': '__main__', '__doc__': None, ..., 's': 10, 'A': 4}
```

除了我省略的大量样板之外，我们可以发现`'A': 4`。

# 内置生成行为

在内置类型中，生成行为现在非常普遍。这是 Python 2 和 Python 3 之间的一个重大区别。许多函数，如`map`、`zip`和`filter`，都已经改变，以便它们返回像可迭代对象一样的对象。这种改变背后的想法是，如果你需要制作这些结果的列表，你可以总是将调用包装在`list()`类中，然后你就完成了。另一方面，如果你只需要迭代，并希望尽可能减少对内存的影响，你可以安全地使用这些函数。

另一个显著的例子是`range`函数。在 Python 2 中，它返回一个列表，还有另一个叫做`xrange`的函数，它返回一个你可以迭代的对象，它会动态生成数字。在 Python 3 中，这个函数已经消失了，`range`现在的行为就像它。

但是，这个概念，总的来说，现在是相当普遍的。你可以在`open()`函数中找到它，这个函数用于操作文件对象（我们将在第七章中看到它，*文件和数据持久性*），但也可以在`enumerate`、字典`keys`、`values`和`items`方法以及其他一些地方找到它。

这一切都是有道理的：Python 的目标是尽可能减少内存占用，尽量避免浪费空间，特别是在大多数情况下广泛使用的那些函数和方法中。

你还记得本章开头吗？我说过，优化那些必须处理大量对象的代码的性能比每天调用两次的函数节省几毫秒更有意义。

# 最后一个例子

在我们结束本章之前，我会向你展示一个我曾经在一家我曾经工作过的公司提交给 Python 开发人员角色的一个简单问题。

问题是：给定序列`0 1 1 2 3 5 8 13 21 ...`，编写一个函数，它将返回这个序列的项直到某个限制`N`。

如果你没有意识到，那就是斐波那契数列，它被定义为*F(0) = 0*，*F(1) = 1*，对于任何*n > 1*，*F(n) = F(n-1) + F(n-2)*。这个序列非常适合测试关于递归、记忆化技术和其他技术细节的知识，但在这种情况下，这是一个检查候选人是否了解生成器的好机会。

让我们从一个基本版本的函数开始，然后对其进行改进：

```py
# fibonacci.first.py
def fibonacci(N):
    """Return all fibonacci numbers up to N. """
    result = [0]
    next_n = 1
    while next_n <= N:
        result.append(next_n)
        next_n = sum(result[-2:])
    return result

print(fibonacci(0))   # [0]
print(fibonacci(1))   # [0, 1, 1]
print(fibonacci(50))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

从头开始：我们将`result`列表设置为起始值`[0]`。然后我们从下一个元素（`next_n`）开始迭代，即`1`。只要下一个元素不大于`N`，我们就不断将它附加到列表中并计算下一个。我们通过取`result`列表中最后两个元素的切片并将其传递给`sum`函数来计算下一个元素。如果这对你来说不清楚，可以在这里和那里添加一些`print`语句，但到现在我希望这不会成为一个问题。

当`while`循环的条件评估为`False`时，我们退出循环并返回`result`。你可以在每个`print`语句旁边的注释中看到这些`print`语句的结果。

在这一点上，我会问候选人以下问题：*如果我只想迭代这些数字怎么办？* 一个好的候选人会改变代码，你会在这里找到（一个优秀的候选人会从这里开始！）：

```py
# fibonacci.second.py
def fibonacci(N):
    """Return all fibonacci numbers up to N. """
    yield 0
    if N == 0:
        return
    a = 0
    b = 1
    while b <= N:
        yield b
        a, b = b, a + b

print(list(fibonacci(0)))   # [0]
print(list(fibonacci(1)))   # [0, 1, 1]
print(list(fibonacci(50)))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

这实际上是我得到的解决方案之一。我不知道为什么我保存了它，但我很高兴我这样做了，这样我就可以向你展示它。现在，`fibonacci`函数是一个*生成器函数*。首先我们产生`0`，然后如果`N`是`0`，我们返回（这将导致引发`StopIteration`异常）。如果不是这种情况，我们开始迭代，每个循环周期产生`b`，然后更新`a`和`b`。为了能够产生序列的下一个元素，我们只需要过去的两个：`a`和`b`。

这段代码好多了，内存占用更少，我们只需要用`list()`将调用包装起来，就像往常一样，就可以得到一个斐波那契数列。但是优雅呢？我不能就这样把它留下吧？让我们试试下面的方法：

```py
# fibonacci.elegant.py
def fibonacci(N):
    """Return all fibonacci numbers up to N. """
    a, b = 0, 1
    while a <= N:
        yield a
        a, b = b, a + b
```

好多了。这个函数的整个主体只有四行，如果算上文档字符串的话就是五行。请注意，在这种情况下，使用元组赋值（`a, b = 0, 1`和`a, b = b, a + b`）有助于使代码更短、更易读。

# 摘要

在本章中，我们更深入地探讨了迭代和生成的概念。我们详细研究了`map`、`zip`和`filter`函数，并学会了如何将它们作为常规`for`循环方法的替代方法。

然后我们讨论了列表、字典和集合的理解概念。我们探讨了它们的语法以及如何将它们作为传统的`for`循环方法和`map`、`zip`和`filter`函数的替代方法来使用。

最后，我们讨论了生成的概念，有两种形式：生成器函数和表达式。我们学会了如何通过使用生成技术来节省时间和空间，并看到它们如何使得通常情况下无法实现的事情成为可能。

我们谈到了性能，并看到`for`循环在速度上是最慢的，但它们提供了最佳的可读性和灵活性。另一方面，诸如`map`和`filter`以及`list`推导这样的函数可能会快得多。

使用这些技术编写的代码复杂度呈指数级增长，因此，为了更有利于可读性和易维护性，我们仍然需要有时使用传统的`for`循环方法。另一个区别在于名称本地化，其中`for`循环的行为与所有其他类型的推导不同。

下一章将全面讨论对象和类。它在结构上与本章类似，我们不会探讨许多不同的主题，只是其中的一些，但我们会尝试更深入地探讨它们。

在继续下一章之前，请确保您理解了本章的概念。我们正在一砖一瓦地建造一堵墙，如果基础不牢固，我们将走不远。
