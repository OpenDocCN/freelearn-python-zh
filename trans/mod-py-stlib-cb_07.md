# 第七章：算法

在本章中，我们将涵盖以下配方：

+   搜索、排序、过滤-在排序的容器中进行高性能搜索

+   获取任何可迭代对象的第 n 个元素-抓取任何可迭代对象的第 n 个元素，包括生成器

+   分组相似项目-将可迭代对象分成相似项目的组

+   合并-将来自多个可迭代对象的数据合并成单个可迭代对象

+   展平列表的列表-将列表的列表转换为平面列表

+   生成排列和-计算一组元素的所有可能排列

+   累积和减少-将二进制函数应用于可迭代对象

+   记忆-通过缓存函数加速计算

+   从运算符到函数-如何保留对 Python 运算符的可调用引用

+   部分-通过预应用一些函数来减少函数的参数数量

+   通用函数-能够根据提供的参数类型改变行为的函数

+   适当的装饰-适当地装饰函数以避免丢失其签名和文档字符串

+   上下文管理器-在进入和退出代码块时自动运行代码

+   应用可变上下文管理器-如何应用可变数量的上下文管理器

# 介绍

在编写软件时，有很多事情你会发现自己一遍又一遍地做，与你正在编写的应用程序类型无关。

除了您可能需要在不同应用程序中重用的整个功能（例如登录、日志记录和授权）之外，还有一堆可以在任何类型的软件中重用的小构建块。

本章将尝试收集一堆可以用作可重用片段的配方，以实现您可能需要独立于软件目的执行的非常常见的操作。

# 搜索、排序、过滤

在编程中查找元素是一个非常常见的需求。在容器中查找项目基本上是您的代码可能会执行的最频繁的操作，因此它非常重要，它既快速又可靠。

排序经常与搜索相关联，因为当你知道你的集合是排序的时，往往可以使用更智能的查找解决方案，并且排序意味着不断搜索和移动项目，直到它们按排序顺序排列。所以它们经常一起出现。

Python 具有内置函数，可以对任何类型的容器进行排序并在其中查找项目，甚至可以利用排序序列的函数。

# 如何做...

对于这个配方，需要执行以下步骤：

1.  取以下一组元素：

```py
>>> values = [ 5, 3, 1, 7 ]
```

1.  通过`in`运算符可以在序列中查找元素：

```py
>>> 5 in values
True
```

1.  排序可以通过`sorted`函数完成：

```py
>>> sorted_value = sorted(values)
>>> sorted_values
[ 1, 3, 5, 7 ]
```

1.  一旦我们有了一个排序的容器，我们实际上可以使用`bisect`模块更快地找到包含的条目：

```py
def bisect_search(container, value):
    index = bisect.bisect_left(container, value)
    return index < len(container) and container[index] == value
```

1.  `bisect_search`可以用来知道一个条目是否在列表中，就像`in`运算符一样：

```py
>>> bisect_search(sorted_values, 5)
True
```

1.  但是，优点是对于许多排序的条目来说可能会更快：

```py
>>> import timeit
>>> values = list(range(1000))
>>> 900 in values
True
>>> bisect_search(values, 900)
True
>>> timeit.timeit(lambda: 900 in values)
timeit.timeit(lambda: bisect_search(values, 900))
13.61617108999053
>>> timeit.timeit(lambda: bisect_search(values, 900))
0.872136551013682
```

因此，在我们的示例中，`bisect_search`函数比普通查找快 17 倍。

# 它是如何工作的...

`bisect`模块使用二分搜索来查找已排序容器中元素的插入点。

如果元素存在于数组中，它的插入位置正是元素所在的位置（因为它应该正好在它所在的位置）：

```py
>>> values = [ 1, 3, 5, 7 ]
>>> bisect.bisect_left(values, 5)
2
```

如果元素缺失，它将返回下一个立即更大的元素的位置：

```py
>>> bisect.bisect_left(values, 4)
2
```

这意味着我们将获得一个位置，即使对于不存在于我们的容器中的元素。这就是为什么我们将返回的位置处的元素与我们正在寻找的元素进行比较。如果两者不同，这意味着返回了最近的元素，因此元素本身没有找到。

出于同样的原因，如果未找到元素并且它大于容器中包含的最大值，则返回容器本身的长度（因为元素应该放在最后），因此我们还需要确保`index < len(container)`来检查不在容器中的元素。

# 还有更多...

到目前为止，我们只对条目本身进行了排序和查找，但在许多情况下，您将拥有复杂的对象，您有兴趣对对象的特定属性进行排序和搜索。

例如，您可能有一个人员列表，您想按其姓名排序：

```py
class Person:
    def __init__(self, name, surname):
        self.name = name
        self.surname = surname
    def __repr__(self):
        return '<Person: %s %s>' % (self.name, self.surname)

people = [Person('Derek', 'Zoolander'),
          Person('Alex', 'Zanardi'),
          Person('Vito', 'Corleone')
          Person('Mario', 'Rossi')]
```

通过依赖`sorted`函数的`key`参数，可以对这些人按姓名进行排序，该参数指定应返回应对条目进行排序的值的可调用对象：

```py
>>> sorted_people = sorted(people, key=lambda v: v.name)
[<Person: Alex Zanardi>, <Person: Derek Zoolander>, 
 <Person: Mario Rossi>, <Person: Vito Corleone>]
```

通过`key`函数进行排序比通过比较函数进行排序要快得多。因为`key`函数只需要对每个项目调用一次（然后结果被保留），而`comparison`函数需要在每次需要比较两个项目时一遍又一遍地调用。因此，如果计算我们应该排序的值很昂贵，`key`函数方法可以实现显着的性能改进。

现在的问题是，`bisect`不允许我们提供一个键，因此为了能够在 people 列表上使用`bisect`，我们首先需要构建一个`keys`列表，然后我们可以应用`bisect`：

```py
>>> keys = [p.name for p in people]
>>> bisect_search(keys, 'Alex')
True
```

这需要通过列表进行一次额外的传递来构建`keys`列表，因此只有在您必须查找多个条目（或多次查找相同的条目）时才方便，否则在列表上进行线性搜索将更快。

请注意，即使要使用`in`运算符，您也必须构建`keys`列表。因此，如果要搜索一个属性而不构建一个特定的列表，您将不得不依赖于`filter`或列表推导。

# 获取任何可迭代对象的第 n 个元素

随机访问容器是我们经常做的事情，而且没有太多问题。对于大多数容器类型来说，这甚至是一个非常便宜的操作。另一方面，当使用通用可迭代对象和生成器时，情况并不像我们期望的那样简单，通常最终会导致我们将它们转换为列表或丑陋的`for`循环。

Python 标准库实际上有办法使这变得非常简单。

# 如何做...

`itertools`模块是一个宝库，当处理可迭代对象时具有非常有价值的功能，并且只需很少的努力就可以获得任何可迭代对象的第 n 个项目：

```py
import itertools

def iter_nth(iterable, nth):
    return next(itertools.islice(iterable, nth, nth+1))
```

给定一个随机的可迭代对象，我们可以使用它来获取我们想要的元素：

```py
>>> values = (x for x in range(10))
>>> iter_nth(values, 4)
4
```

# 它是如何工作的...

`itertools.islice`函数能够获取任何可迭代对象的切片。在我们的特定情况下，我们需要的是从我们要查找的元素到下一个元素的切片。

一旦我们有了包含我们要查找的元素的切片，我们就需要从切片本身中提取该项。

由于`islice`作用于可迭代对象，它本身返回一个可迭代对象。这意味着我们可以使用`next`来消耗它，由于我们要查找的项实际上是切片的第一个项，因此使用`next`将正确返回我们要查找的项。

如果元素超出范围（例如，我们在仅有三个元素的情况下寻找第四个元素），则会引发`StopIteration`错误，我们可以像在普通列表中一样捕获它，就像对`IndexError`一样。

# 分组相似的项目

有时，您可能会面对一个具有多个重复条目的条目列表，并且您可能希望根据某种属性对相似的条目进行分组。

例如，这里是一个名字列表：

```py
names = [('Alex', 'Zanardi'),
         ('Julius', 'Caesar'),
         ('Anakin', 'Skywalker'),
         ('Joseph', 'Joestar')]
```

我们可能希望构建一个所有名字以相同字符开头的人的组，这样我们就可以按字母顺序保留我们的电话簿，而不是让名字随机散落在这里和那里。

# 如何做...

`itertools`模块再次是一个非常强大的工具，它为我们提供了处理可迭代对象所需的基础：

```py
import itertools

def group_by_key(iterable, key):
    iterable = sorted(iterable, key=key)
    return {k: list(g) for k,g in itertools.groupby(iterable, key)}
```

给定我们的姓名列表，我们可以应用一个键函数，该函数获取名称的第一个字符，以便所有条目都将按其分组：

```py
>>> group_by_key(names, lambda v: v[0][0])
{'A': [('Alex', 'Zanardi'), ('Anakin', 'Skywalker')], 
 'J': [('Julius', 'Caesar'), ('Joseph', 'Joestar')]}
```

# 它是如何工作的...

这里的函数核心由`itertools.groupby`提供。

此函数将迭代器向前移动，抓取项目，并将其添加到当前组中。当面对具有不同键的项目时，将创建一个新组。

因此，实际上，它只会将共享相同键的附近条目分组：

```py
>>> sample = [1, 2, 1, 1]
>>> [(k, list(g)) for k,g in itertools.groupby(sample)]
[(1, [1]), (2, [2]), (1, [1, 1])]
```

正如您所看到的，这里有三个组，而不是预期的两个，因为数字`1`的第一组立即被数字`2`中断，因此我们最终得到了两个不同的`1`组。

我们在对它们进行分组之前对元素进行排序，原因是排序可以确保相等的元素都靠在一起：

```py
>>> sorted(sample)
[1, 1, 1, 2]
```

在那一点上，分组函数将创建正确数量的组，因为每个等效元素都有一个单独的块：

```py
>>> sorted_sample = sorted(sample)
>>> [(k, list(g)) for k,g in itertools.groupby(sorted_sample)]
[(1, [1, 1, 1]), (2, [2])]
```

我们在现实生活中经常使用复杂的对象，因此`group_by_key`函数还接受`key`函数。这将说明应该根据哪个键对元素进行分组。

由于排序在排序时接受一个键函数，因此我们知道在分组之前所有元素都将根据该键进行排序，因此我们将返回正确数量的组。

最后，由于`groupby`返回一个迭代器或迭代器（顶级可迭代对象中的每个组也是一个迭代器），我们将每个组转换为列表，并构建一个字典，以便可以通过`key`轻松访问这些组。

# 压缩

Zipping 意味着附加两个不同的可迭代对象，以创建一个包含两者值的新对象。

当您有多个值轨道应该同时进行时，这是非常方便的。想象一下，您有名字和姓氏，您只想得到一个人的列表：

```py
names = [ 'Sam', 'Axel', 'Aerith' ]
surnames = [ 'Fisher', 'Foley', 'Gainsborough' ]
```

# 如何做到这一点...

我们想要将名称和姓氏一起压缩：

```py
>>> people = zip(names, surnames)
>>> list(people)
[('Sam', 'Fisher'), ('Axel', 'Foley'), ('Aerith', 'Gainsborough')]
```

# 它是如何工作的...

Zip 将创建一个新的可迭代对象，其中新创建的可迭代对象中的每个项目都是通过从所提供的可迭代对象中选择一个项目而生成的集合。

因此，`result[0] = (i[0], j[0])`，`result[1] = (i[1], j[1])`，依此类推。如果`i`和`j`的长度不同，它将在两者之一耗尽时立即停止。

如果要继续直到耗尽所提供的可迭代对象中最长的一个，而不是在最短的一个上停止，可以依靠`itertools.zip_longest`。已经耗尽的可迭代对象的值将填充默认值。

# 展平列表的列表

当您有多个嵌套列表时，通常需要遍历所有列表中包含的项目，而不太关心它们实际存储的深度。

假设您有这个列表：

```py
values = [['a', 'b', 'c'],
          [1, 2, 3],
          ['X', 'Y', 'Z']]
```

如果您只想抓取其中的所有项目，那么您真的不想遍历列表中的列表，然后再遍历其中每一个项目。我们只想要叶子项目，我们根本不在乎它们在列表中的列表中。

# 如何做到这一点...

我们想要做的就是将所有列表连接成一个可迭代对象，该对象将产生项目本身，因为我们正在谈论迭代器，`itertools`模块具有正确的函数，可以让我们像单个迭代器一样链接所有列表：

```py
>>> import itertools
>>> chained = itertools.chain.from_iterable(values)
```

生成的`chained`迭代器将在消耗时逐个产生底层项目：

```py
>>> list(chained)
['a', 'b', 'c', 1, 2, 3, 'X', 'Y', 'Z']
```

# 它是如何工作的...

`itertools.chain`函数在您需要依次消耗多个可迭代对象时非常方便。

默认情况下，它接受这些可迭代对象作为参数，因此我们将不得不执行：

```py
itertools.chain(values[0], values[1], values[2])
```

但是，为了方便起见，`itertools.chain.from_iterable`将链接提供的参数中包含的条目，而不必逐个显式传递它们。

# 还有更多...

如果您知道原始列表包含多少项，并且它们的大小相同，那么很容易应用反向操作。

我们已经知道可以使用`zip`从多个来源合并条目，所以我们实际上想要做的是将原始列表的元素一起压缩，这样我们就可以从`chained`返回到原始的列表列表：

```py
>>> list(zip(chained, chained, chained))
[('a', 'b', 'c'), (1, 2, 3), ('X', 'Y', 'Z')]
```

在这种情况下，我们有三个项目列表，所以我们必须提供`chained`三次。

这是因为`zip`将顺序地从每个提供的参数中消耗一个条目。 因此，由于我们提供了相同的参数三次，实际上我们正在消耗前三个条目，然后是接下来的三个，然后是最后的三个。

如果`chained`是一个列表而不是一个迭代器，我们将不得不从列表中创建一个迭代器：

```py
>>> chained = list(chained) 
>>> chained ['a', 'b', 'c', 1, 2, 3, 'X', 'Y', 'Z'] 
>>> ichained = iter(chained) 
>>> list(zip(ichained, ichained, ichained)) [('a', 'b', 'c'), (1, 2, 3), ('X', 'Y', 'Z')]
```

如果我们没有使用`ichained`而是使用原始的`chained`，结果将与我们想要的相去甚远：

```py
>>> chained = list(chained)
>>> chained
['a', 'b', 'c', 1, 2, 3, 'X', 'Y', 'Z']
>>> list(zip(chained, chained, chained))
[('a', 'a', 'a'), ('b', 'b', 'b'), ('c', 'c', 'c'), 
 (1, 1, 1), (2, 2, 2), (3, 3, 3), 
 ('X', 'X', 'X'), ('Y', 'Y', 'Y'), ('Z', 'Z', 'Z')]
```

# 生成排列和组合

给定一组元素，如果您曾经感到有必要对这些元素的每个可能的排列执行某些操作，您可能会想知道生成所有这些排列的最佳方法是什么。

Python 在`itertools`模块中有各种函数，可帮助进行排列和组合，这些之间的区别并不总是容易理解，但一旦您调查它们的功能，它们就会变得清晰。

# 如何做...

笛卡尔积通常是在谈论组合和排列时人们所考虑的。

1.  给定一组元素`A`，`B`和`C`，我们想要提取所有可能的两个元素的组合，`AA`，`AB`，`AC`等等：

```py
>>> import itertools
>>> c = itertools.product(('A', 'B', 'C'), repeat=2)
>>> list(c)
[('A', 'A'), ('A', 'B'), ('A', 'C'),
 ('B', 'A'), ('B', 'B'), ('B', 'C'), 
 ('C', 'A'), ('C', 'B'), ('C', 'C')]
```

1.  如果您想要省略重复的条目（`AA`，`BB`，`CC`），您可以只使用排列：

```py
>>> c = itertools.permutations(('A', 'B', 'C'), 2)
>>> list(c)
[('A', 'B'), ('A', 'C'), 
 ('B', 'A'), ('B', 'C'), 
 ('C', 'A'), ('C', 'B')]
```

1.  您甚至可能希望确保相同的夫妇不会发生两次（例如`AB`与`BA`），在这种情况下，`itertools.combinations`可能是您要寻找的。

```py
>>> c = itertools.combinations(('A', 'B', 'C'), 2)
>>> list(c)
[('A', 'B'), ('A', 'C'), ('B', 'C')]
```

因此，大多数需要组合值的需求都可以通过`itertools`模块提供的函数轻松解决。

# 累积和减少

列表推导和`map`是非常方便的工具，当您需要将函数应用于可迭代对象的所有元素并返回结果值时。 但这些工具大多用于应用一元函数并保留转换值的集合（例如将所有数字加`1`），但是如果您想要应用应该一次接收多个元素的函数，它们就不太合适。

减少和累积函数实际上是为了从可迭代对象中接收多个值并返回单个值（在减少的情况下）或多个值（在累积的情况下）。

# 如何做...

这个食谱的步骤如下：

1.  减少的最简单的例子是对可迭代对象中的所有项目求和：

```py
>>> values = [ 1, 2, 3, 4, 5 ]
```

1.  这是可以通过`sum`轻松完成的事情，但是为了这个例子，我们将使用`reduce`：

```py
>>> import functools, operator
>>> functools.reduce(operator.add, values)
15
```

1.  如果您不是要获得单个最终结果，而是要保留中间步骤的结果，您可以使用`accumulate`：

```py
>>> import itertools
>>> list(itertools.accumulate(values, operator.add))
[1, 3, 6, 10, 15]
```

# 还有更多...

`accumulate`和`reduce`不仅限于数学用途。 虽然这些是最明显的例子，但它们是非常灵活的函数，它们的目的取决于它们将应用的函数。

例如，如果您有多行文本，您也可以使用`reduce`来计算所有文本的总和：

```py
>>> lines = ['this is the first line',
...          'then there is one more',
...          'and finally the last one.']
>>> functools.reduce(lambda x, y: x + len(y), [0] + lines)
69
```

或者，如果您有多个需要折叠的字典：

```py
>>> dicts = [dict(name='Alessandro'), dict(surname='Molina'),
...          dict(country='Italy')]
>>> functools.reduce(lambda d1, d2: {**d1, **d2}, dicts)
{'name': 'Alessandro', 'surname': 'Molina', 'country': 'Italy'}
```

这甚至是访问深度嵌套的字典的一种非常方便的方法：

```py
>>> import operator
>>> nesty = {'a': {'b': {'c': {'d': {'e': {'f': 'OK'}}}}}}
>>> functools.reduce(operator.getitem, 'abcdef', nesty)
'OK'
```

# 记忆化

一遍又一遍地运行函数，避免调用该函数的成本可以大大加快生成的代码。

想象一下`for`循环或递归函数，也许必须调用该函数数十次。 如果它能够保留对函数的先前调用的已知结果，而不是调用它，那么它可以大大加快代码。

最常见的例子是斐波那契数列。 该序列是通过添加前两个数字来计算的，然后将第二个数字添加到结果中，依此类推。

这意味着在序列`1`，`1`，`2`，`3`，`5`中，计算`5`需要我们计算`3 + 2`，这又需要我们计算`2 + 1`，这又需要我们计算`1 + 1`。

以递归方式进行斐波那契数列是最明显的方法，因为它导致`5 = fib(n3) + fib(n2)`，其中`3 = fib(n2) + fib(n1)`，所以你可以很容易地看到我们必须计算`fib(n2)`两次。记忆`fib(n2)`的结果将允许我们只执行这样的计算一次，然后在下一次调用时重用结果。

# 如何做...

这是这个食谱的步骤：

1.  Python 提供了内置的 LRU 缓存，我们可以用它来进行记忆化：

```py
import functools

@functools.lru_cache(maxsize=None)
def fibonacci(n):
    '''inefficient recursive version of Fibonacci number'''
    if n > 1:
        return fibonacci(n-1) + fibonacci(n-2)
    return n
```

1.  然后我们可以使用该函数来计算整个序列：

```py
fibonacci_seq = [fibonacci(n) for n in range(100)]
```

1.  结果将是一个包含所有斐波那契数的列表，直到第 100 个：

```py
>>> print(fibonacci_seq)
[0, 1, 1, 2, 3, 5, 8, 13, 21 ...
```

性能上的差异是巨大的。如果我们使用`timeit`模块来计时我们的函数，我们可以很容易地看到记忆化对性能有多大帮助。

1.  当使用`fibonacci`函数的记忆化版本时，计算在不到一毫秒内结束：

```py
>>> import timeit
>>> timeit.timeit(lambda: [fibonacci(n) for n in range(40)], number=1)
0.000033469987101
```

1.  然后，如果我们移除`@functools.lru_cache()`，实现记忆化的时间会发生根本性的变化：

```py
>>> timeit.timeit(lambda: [fibonacci(n) for n in range(40)], number=1)
89.14927123498637
```

所以很容易看出记忆化如何将性能从 89 秒提高到几分之一秒。

# 它是如何工作的...

每当调用函数时，`functools.lru_cache`都会保存返回的值以及提供的参数。

下一次调用函数时，参数将在保存的参数中搜索，如果找到，将提供先前返回的值，而不是调用函数。

实际上，这改变了调用我们的函数的成本，只是在字典中查找的成本。

所以第一次调用`fibonacci(5)`时，它被计算，然后下一次调用时它将什么都不做，之前存储的`5`的值将被返回。由于`fibonacci(6)`必须调用`fibonacci(5)`才能计算，很容易看出我们为任何`fibonacci(n)`提供了主要的性能优势，其中`n>5`。

同样，由于我们想要整个序列，所以节省不仅仅是单个调用，而是在第一个需要记忆值的列表推导式之后的每次调用。

`lru_cache`函数诞生于**最近最少使用**（**LRU**）缓存，因此默认情况下，它只保留最近的`128`个，但通过传递`maxsize=None`，我们可以将其用作标准缓存，并丢弃其中的 LRU 部分。所有调用将永远被缓存，没有限制。

纯粹针对斐波那契情况，你会注意到将`maxsize`设置为大于`3`的任何值都不会改变，因为每个斐波那契数只需要前两个调用就能计算。

# 函数到运算符

假设你想创建一个简单的计算器。第一步是解析用户将要写的公式以便执行它。基本公式由一个运算符和两个操作数组成，所以你实际上有一个函数和它的参数。

但是，考虑到`+`，`-`等等，我们的解析器如何返回相关的函数呢？通常，为了对两个数字求和，我们只需写`n1 + n2`，但我们不能传递`+`本身来调用任何`n1`和`n2`。

这是因为`+`是一个运算符而不是一个函数，但在 CPython 中它仍然只是一个函数被执行。

# 如何做...

我们可以使用`operator`模块来获取一个可调用的对象，表示我们可以存储或传递的任何 Python 运算符：

```py
import operator

operators = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv
}

def calculate(expression):
    parts = expression.split()

    try:
        result = int(parts[0])
    except:
        raise ValueError('First argument of expression must be numberic')

    operator = None
    for part in parts[1:]:
        try:
            num = int(part)
            if operator is None:
                raise ValueError('No operator proviede for the numbers')
        except ValueError:
            if operator:
                raise ValueError('operator already provided')
            operator = operators[part]
        else:
            result = operator(result, num)
            operator = None

    return result
```

我们的`calculate`函数充当一个非常基本的计算器（没有运算符优先级，实数，负数等）：

```py
>>> print(calculate('5 + 3'))
8
>>> print(calculate('1 + 2 + 3'))
6
>>> print(calculate('3 * 2 + 4'))
10
```

# 它是如何工作的...

因此，我们能够在`operators`字典中存储四个数学运算符的函数，并根据表达式中遇到的文本查找它们。

在`calculate`中，表达式被空格分隔，因此`5 + 3`变成了`['5'，'+'，'3']`。一旦我们有了表达式的三个元素（两个操作数和运算符），我们只需遍历部分，当我们遇到`+`时，在`operators`字典中查找以获取应该调用的关联函数，即`operator.add`。

`operator`模块包含了最常见的 Python 运算符的函数，从比较（`operator.gt`）到基于点的属性访问（`operator.attrgetter`）。

大多数提供的函数都是为了与`map`、`sorted`、`filter`等配对使用。

# 部分

我们已经知道可以使用`map`将一元函数应用于多个元素，并使用`reduce`将二元函数应用于多个元素。

有一整套函数接受 Python 中的可调用函数，并将其应用于一组项目。

主要问题是，我们想要应用的可调用函数可能具有稍有不同的签名，虽然我们可以通过将可调用函数包装到另一个适应签名的可调用函数中来解决问题，但如果你只想将函数应用到一组项目中，这并不是很方便。

例如，如果你想将列表中的所有数字乘以 3，没有一个函数可以将给定的参数乘以 3。

# 如何做...

我们可以很容易地将`operator.mul`调整为一元函数，然后将其传递给`map`以将其应用于整个列表：

```py
>>> import functools, operator
>>>
>>> values = range(10)
>>> mul3 = functools.partial(operator.mul, 3)
>>> list(map(mul3, values))
[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
```

正如你所看到的，`operator.mul`被调用时带有`3`和项目作为其参数，因此返回`item*3`。

# 它是如何工作的...

我们通过`functools.partial`创建了一个新的`mul3`可调用函数。这个可调用函数只是调用`operator.mul`，将`3`作为第一个参数传递，然后将提供给可调用函数的任何参数作为第二、第三等参数传递给`operator.mul`。

因此，最终执行`mul3(5)`意味着`operator.mul(3, 5)`。

这是因为`functools.partial`通过提供的函数硬编码提供的参数创建一个新函数。

当然，也可以传递关键字参数，这样我们就可以设置任何参数，而不是硬编码第一个参数。

然后，将生成的函数应用于所有数字通过`map`，这将导致创建一个新列表，其中包含所有从 0 到 10 的数字乘以 3。

# 通用函数

通用函数是标准库中我最喜欢的功能之一。Python 是一种非常动态的语言，通过鸭子类型，你经常能够编写适用于许多不同条件的代码（无论你收到的是列表还是元组），但在某些情况下，你确实需要根据接收到的输入有两个完全不同的代码库。

例如，我们可能希望有一个函数，以人类可读的格式打印所提供的字典内容，但我们也希望它在元组列表上正常工作，并报告不支持的类型的错误。

# 如何做...

`functools.singledispatch`装饰器允许我们基于参数类型实现通用分派：

```py
from functools import singledispatch

@singledispatch
def human_readable(d):
    raise ValueError('Unsupported argument type %s' % type(d))

@human_readable.register(dict)
def human_readable_dict(d):
    for key, value in d.items():
        print('{}: {}'.format(key, value))

@human_readable.register(list)
@human_readable.register(tuple)
def human_readable_list(d):
    for key, value in d:
        print('{}: {}'.format(key, value))
```

调用这三个函数将正确地将请求分派到正确的函数：

```py
>>> human_readable({'name': 'Tifa', 'surname': 'Lockhart'})
name: Tifa
surname: Lockhart

>>> human_readable([('name', 'Nobuo'), ('surname', 'Uematsu')])
name: Nobuo
surname: Uematsu

>>> human_readable(5)
Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "<stdin>", line 2, in human_readable
ValueError: Unsupported argument type <class 'int'>
```

# 它是如何工作的...

使用`@singledispatch`装饰的函数实际上被一个对参数类型的检查所取代。

每次调用`human_readable.register`都会记录到一个注册表中，指定每种参数类型应该使用哪个可调用函数：

```py
>>> human_readable.registry
mappingproxy({
    <class 'list'>: <function human_readable_list at 0x10464da60>, 
    <class 'object'>: <function human_readable at 0x10464d6a8>, 
    <class 'dict'>: <function human_readable_dict at 0x10464d950>, 
    <class 'tuple'>: <function human_readable_list at 0x10464da60>
})
```

每当调用装饰的函数时，它将在注册表中查找参数的类型，并将调用转发到关联的函数以执行。

使用`@singledispatch`装饰的函数应该始终是通用实现，即在参数没有明确支持时应该使用的实现。

在我们的示例中，这只是抛出一个错误，但通常情况下，它将尝试提供在大多数情况下有效的实现。

然后，可以使用 `@function.register` 注册特定的实现，以覆盖主要函数无法覆盖的情况，或者实际实现行为，如果主要函数只是抛出错误。

# 适当的装饰

对于第一次面对装饰器的任何人来说，装饰器通常并不直接，但一旦你习惯了它们，它们就成为扩展函数行为或实现轻量级面向方面的编程的非常方便的工具。

但即使装饰器变得自然并成为日常开发的一部分，它们也有细微之处，直到您第一次面对它们时才会变得不明显。

当您应用 `decorator` 时，可能并不立即明显，但通过使用它们，您正在改变 `decorated` 函数的签名，直到函数本身的名称和文档都丢失：

```py
def decorator(f):
    def _f(*args, **kwargs):
        return f(*args, **kwargs)
    return _f

@decorator
def sumtwo(a, b):
    """Sums a and b"""
    return a + back
```

`sumtwo` 函数被 `decorator` 装饰，但现在，如果我们尝试访问函数文档或名称，它们将不再可访问：

```py
>>> print(sumtwo.__name__)
'_f'
>>> print(sumtwo.__doc__)
None
```

即使我们为 `sumtwo` 提供了文档字符串，并且我们确切知道它的名称是 `sumtwo`，我们仍需要确保我们的装饰被正确应用并保留原始函数的属性。

# 如何做...

对于这个配方，需要执行以下步骤：

1.  Python 标准库提供了一个 `functools.wraps` 装饰器，可以应用于装饰器，以使它们保留装饰函数的属性：

```py
from functools import wraps

def decorator(f):
    @wraps(f)
    def _f(*args, **kwargs):
        return f(*args, **kwargs)
    return _f
```

1.  在这里，我们将装饰器应用于一个函数：

```py
@decorator
def sumthree(a, b):
    """Sums a and b"""
    return a + back
```

1.  正如您所看到的，它将正确保留函数的名称和文档字符串：

```py
>>> print(sumthree.__name__)
'sumthree'
>>> print(sumthree.__doc__)
'Sums a and b'
```

如果装饰的函数有自定义属性，这些属性也将被复制到新函数中。

# 还有更多...

`functools.wraps` 是一个非常方便的工具，尽最大努力确保装饰函数看起来与原始函数完全一样。

但是，虽然函数的属性可以很容易地被复制，但函数本身的签名并不容易复制。

因此，检查我们装饰的函数参数不会返回原始参数：

```py
>>> import inspect
>>> inspect.getfullargspec(sumthree)
FullArgSpec(args=[], varargs='args', varkw='kwargs', defaults=None, 
            kwonlyargs=[], kwonlydefaults=None, annotations={})
```

因此，报告的参数只是 `*args` 和 `**kwargs` 而不是 `a` 和 `b`。要访问真正的参数，我们必须通过 `__wrapped__` 属性深入到底层函数中：

```py
>>> inspect.getfullargspec(sumthree.__wrapped__)
FullArgSpec(args=['a', 'b'], varargs=None, varkw=None, defaults=None, 
            kwonlyargs=[], kwonlydefaults=None, annotations={})
```

幸运的是，标准库为我们提供了一个 `inspect.signature` 函数来做到这一点：

```py
>>> inspect.signature(sumthree)
(a, b)
```

因此，最好在想要检查函数的参数时依赖于 `inspect.signature`，以便支持装饰和未装饰的函数。

应用装饰也可能与其他装饰器冲突。最常见的例子是 `classmethod`：

```py
class MyClass(object):
    @decorator
    @classmethod
    def dosum(cls, a, b):
        return a+b
```

尝试装饰 `classmethod` 通常不起作用：

```py
>>> MyClass.dosum(3, 3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    return f(*args, **kwargs)
TypeError: 'classmethod' object is not callable
```

您需要确保 `@classmethod` 始终是最后应用的装饰器，以确保它将按预期工作：

```py
class MyClass(object):
    @classmethod
    @decorator
    def dosum(cls, a, b):
        return a+b
```

在那时，`classmethod` 将按预期工作：

```py
>>> MyClass.dosum(3, 3)
6
```

Python 环境中有许多与装饰器相关的怪癖，因此有一些库试图为日常使用正确实现装饰。如果您不想考虑如何处理它们，您可能想尝试 `wrapt` 库，它将为您处理大多数装饰怪癖。

# 上下文管理器

装饰器可用于确保在进入和退出函数时执行某些操作，但在某些情况下，您可能希望确保在代码块的开头和结尾始终执行某些操作，而无需将其移动到自己的函数中或重写应该每次执行的部分。

上下文管理器存在是为了解决这个需求，将您必须一遍又一遍地重写的代码因 `try:except:finally:` 子句而被分解出来。

上下文管理器最常见的用法可能是关闭上下文管理器，它确保文件在开发人员完成使用它们后关闭，但标准库使编写新的上下文管理器变得很容易。

# 如何做...

对于这个配方，需要执行以下步骤：

1.  `contextlib`提供了与上下文管理器相关的功能，`contextlib.contextmanager`可以使编写上下文管理器变得非常容易：

```py
@contextlib.contextmanager
def logentrance():
    print('Enter')
    yield
    print('Exit')
```

1.  然后创建的上下文管理器可以像任何其他上下文管理器一样使用：

```py
>>> with logentrance():
>>>    print('This is inside')
Enter
This is inside
Exit
```

1.  在包装块内引发的异常将传播到上下文管理器，因此可以使用标准的`try:except:finally:`子句来处理它们并进行适当的清理：

```py
@contextlib.contextmanager
def logentrance():
    print('Enter')
    try:
        yield
    except:
        print('Exception')
        raise
    finally:
        print('Exit')
```

1.  更改后的上下文管理器将能够记录异常，而不会干扰异常的传播。

```py
>>> with logentrance():
        raise Exception('This is an error')
Enter
Exception
Exit
Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
        raise Exception('This is an error')
Exception: This is an error
```

# 应用可变上下文管理器

在使用上下文管理器时，必须依赖`with`语句来应用它们。虽然可以通过用逗号分隔它们来在一个语句中应用多个上下文管理器，但是要应用可变数量的上下文管理器并不那么容易：

```py
@contextlib.contextmanager
def first():
    print('First')
    yield

@contextlib.contextmanager
def second():
    print('Second')
    yield
```

在编写代码时必须知道要应用的上下文管理器：

```py
>>> with first(), second():
>>>     print('Inside')
First
Second
Inside
```

但是如果有时我们只想应用`first`上下文管理器，有时又想同时应用两个呢？

# 如何做...

`contextlib.ExitStack`有各种用途，其中之一是允许我们对一个块应用可变数量的上下文管理器。

例如，我们可能只想在循环中打印偶数时同时应用两个上下文管理器：

```py
from contextlib import ExitStack

for n in range(5):
    with ExitStack() as stack:
        stack.enter_context(first())
        if n % 2 == 0:
            stack.enter_context(second())
        print('NUMBER: {}'.format(n))
```

结果将是`second`只被添加到上下文中，因此仅对偶数调用：

```py
First
Second
NUMBER: 0
First
NUMBER: 1
First
Second
NUMBER: 2
First
NUMBER: 3
First
Second
NUMBER: 4
```

正如你所看到的，对于`1`和`3`，只有`First`被打印出来。

当通过`ExitStack`上下文管理器声明的上下文退出时，`ExitStack`中注册的所有上下文管理器也将被退出。
