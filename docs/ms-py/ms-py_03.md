# 第三章：容器和集合-正确存储数据

Python 捆绑了几个非常有用的集合，其中一些是基本的 Python 集合数据类型。其余的是这些类型的高级组合。在本章中，我们将解释其中一些集合的使用方法，以及它们各自的优缺点。

在我们正式讨论数据结构和相关性能之前，需要对时间复杂度（特别是大 O 符号）有基本的了解。不用担心！这个概念非常简单，但没有它，我们无法轻松地解释操作的性能特征。

一旦大 O 符号清晰，我们将讨论基本数据结构：

+   `list`

+   `dict`

+   `set`

+   `tuple`

在基本数据结构的基础上，我们将继续介绍更高级的集合，例如以下内容：

+   类似字典的类型：

+   `ChainMap`

+   `Counter`

+   `Defaultdict`

+   `OrderedDict`

+   列表类型：

+   `Deque`

+   `Heapq`

+   元组类型：

+   `NamedTuple`

+   其他类型：

+   `Enum`

# 时间复杂度-大 O 符号

在开始本章之前，您需要了解一个简单的符号。本章大量使用大 O 符号来指示操作的时间复杂度。如果您已经熟悉这个符号，可以跳过这一段。虽然这个符号听起来很复杂，但实际概念非常简单。

当我们说一个函数需要`O(1)`的时间时，这意味着通常只需要`1`步来执行。同样，一个具有`O(n)`的函数将需要`n`步来执行，其中`n`通常是对象的大小。这种时间复杂度只是对执行代码时可以预期的基本指示，因为这通常是最重要的。

该系统的目的是指示操作的大致性能；这与代码速度无关，但仍然相关。执行单个步骤的代码`1000`次更快，但需要执行`O(2**n)`步骤的代码仍然比另一个版本慢，因为对于 n 等于`10`或更高的值，它只需要`O(n)`步骤。这是因为`n=10`时`2**n`为`2**10=1024`，也就是说，执行相同代码需要 1,024 步。这使得选择正确的算法非常重要。即使`C`代码通常比 Python 快，如果使用错误的算法，也毫无帮助。

例如，假设您有一个包含`1000`个项目的列表，并且您遍历它们。这将花费`O(n)`的时间，因为有`n=1000`个项目。检查项目是否存在于列表中需要`O(n)`的时间，因此需要 1,000 步。这样做 100 次将花费`100*O(n) = 100 * 1000 = 100,000`步。当您将其与`dict`进行比较时，检查项目是否存在只需要`O(1)`的时间，差异是巨大的。使用`dict`，将是`100*O(1) = 100 * 1 = 100`步。因此，对于包含 1000 个项目的对象，使用`dict`而不是`list`将大约快 1,000 倍：

```py
n = 1000
a = list(range(n))
b = dict.fromkeys(range(n))
for i in range(100):
    i in a  # takes n=1000 steps
    i in b  # takes 1 step
```

为了说明`O(1)`，`O(n)`和`O(n**2)`函数：

```py
def o_one(items):
    return 1  # 1 operation so O(1)

def o_n(items):
    total = 0
    # Walks through all items once so O(n)
    for item in items:
        total += item
    return total

def o_n_squared(items):
    total = 0
    # Walks through all items n*n times so O(n**2)
    for a in items:
        for b in items:
            total += a * b
    return total

n = 10
items = range(n)
o_one(items)  # 1 operation
o_n(items)  # n = 10 operations
o_n_squared(items)  # n*n = 10*10 = 100 operations
```

应该注意，本章中的大 O 是关于平均情况，而不是最坏情况。在某些情况下，它们可能更糟，但这些情况很少，可以忽略不计。

# 核心集合

在本章稍后讨论更高级的组合集合之前，您需要了解核心 Python 集合的工作原理。这不仅仅是关于使用，还涉及到时间复杂度，这会对应用程序随着增长而产生强烈影响。如果您熟悉这些对象的时间复杂度，并且熟记 Python 3 的元组打包和解包的可能性，那么可以直接跳到*高级集合*部分。

## list - 一个可变的项目列表

`list`很可能是您在 Python 中最常用的容器结构。它的使用简单，对于大多数情况，性能很好。

虽然你可能已经熟悉了列表的使用，但你可能不知道`list`对象的时间复杂度。幸运的是，`list`的许多时间复杂度非常低；`append`，`get`，`set`和`len`都需要`O(1)`的时间-这是最好的可能性。但是，你可能不知道`remove`和`insert`的时间复杂度是`O(n)`。因此，要从 1000 个项目中删除一个项目，Python 将不得不遍历 1000 个项目。在内部，`remove`和`insert`操作执行类似于这样的操作：

```py
>>> def remove(items, value):
...     new_items = []
...     found = False
...     for item in items:
...         # Skip the first item which is equal to value
...         if not found and item == value:
...             found = True
...             continue
...         new_items.append(item)
...
...     if not found:
...         raise ValueError('list.remove(x): x not in list')
...
...     return new_items

>>> def insert(items, index, value):
...     new_items = []
...     for i, item in enumerate(items):
...         if i == index:
...             new_items.append(value)
...         new_items.append(item)
...     return new_items

>>> items = list(range(10))
>>> items
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

>>> items = remove(items, 5)
>>> items
[0, 1, 2, 3, 4, 6, 7, 8, 9]

>>> items = insert(items, 2, 5)
>>> items
[0, 1, 5, 2, 3, 4, 6, 7, 8, 9]

```

要从列表中删除或插入单个项目，Python 需要复制整个列表，这在列表较大时特别耗费资源。当执行一次时，当然不是那么糟糕。但是当执行大量删除时，`filter`或`list`推导是一个更快的解决方案，因为如果结构良好，它只需要复制列表一次。例如，假设我们希望从列表中删除一组特定的数字。我们有很多选项。第一个是使用`remove`，然后是列表推导，然后是`filter`语句。第四章, *功能编程-可读性与简洁性*，将更详细地解释`list`推导和`filter`语句。但首先，让我们看看这个例子：

```py
>>> primes = set((1, 2, 3, 5, 7))

# Classic solution
>>> items = list(range(10))
>>> for prime in primes:
...     items.remove(prime)
>>> items
[0, 4, 6, 8, 9]

# List comprehension
>>> items = list(range(10))
>>> [item for item in items if item not in primes]
[0, 4, 6, 8, 9]

# Filter
>>> items = list(range(10))
>>> list(filter(lambda item: item not in primes, items))
[0, 4, 6, 8, 9]

```

后两种对于大量项目的列表要快得多。这是因为操作要快得多。比较使用`n=len(items)`和`m=len(primes)`，第一个需要`O(m*n)=5*10=50`次操作，而后两个需要`O(n*1)=10*1=10`次操作。

### 注意

第一种方法实际上比这更好一些，因为`n`在循环过程中减少。所以，实际上是`10+9+8+7+6=40`，但这是一个可以忽略的效果。在`n=1000`的情况下，这将是`1000+999+998+997+996=4990`和`5*1000=5000`之间的差异，在大多数情况下是可以忽略的。

当然，`min`，`max`和`in`都需要`O(n)`，但这对于一个不是为这些类型的查找进行优化的结构来说是可以预料的。

它们可以这样实现：

```py
>>> def in_(items, value):
...     for item in items:
...         if item == value:
...             return True
...     return False

>>> def min_(items):
...     current_min = items[0]
...     for item in items[1:]:
...         if current_min > item:
...             current_min = item
...     return current_min

>>> def max_(items):
...     current_max = items[0]
...     for item in items[1:]:
...         if current_max < item:
...             current_max = item
...     return current_max

>>> items = range(5)
>>> in_(items, 3)
True
>>> min_(items)
0
>>> max_(items)
4

```

通过这些例子，很明显`in`运算符如果你幸运的话可以工作`O(1)`，但我们将其视为`O(n)`，因为它可能不存在，如果不存在，那么所有的值都需要被检查。

## dict-无序但快速的项目映射

`dict`必须至少是你在 Python 中使用的前三种容器结构之一。它快速，易于使用，非常有效。平均时间复杂度正如你所期望的那样-`O(1)`对于`get`，`set`和`del`-但也有一些例外。`dict`的工作方式是通过使用`hash`函数（调用对象的`__hash__`函数）将键转换为哈希并将其存储在哈希表中。然而，哈希表有两个问题。第一个和最明显的问题是，项目将按哈希排序，这在大多数情况下是随机的。哈希表的第二个问题是它们可能会发生哈希冲突，哈希冲突的结果是在最坏的情况下，所有先前的操作可能需要`O(n)`。哈希冲突并不太可能发生，但它们可能发生，如果一个大的`dict`表现不佳，那就是需要查看的地方。

让我们看看这在实践中是如何工作的。为了举例说明，我将使用我能想到的最简单的哈希算法，即数字的最高位。所以，对于`12345`，它将返回`1`，对于`56789`，它将返回`5`：

```py
>>> def most_significant(value):
...     while value >= 10:
...         value //= 10
...     return value

>>> most_significant(12345)
1
>>> most_significant(99)
9
>>> most_significant(0)
0

```

现在我们将使用这种哈希方法使用一个列表的列表来模拟一个`dict`。我们知道我们的哈希方法只能返回`0`到`9`之间的数字，所以我们在列表中只需要 10 个桶。现在我们将添加一些值，并展示 spam in eggs 可能如何工作：

```py
>>> def add(collection, key, value):
...     index = most_significant(key)
...     collection[index].append((key, value))

>>> def contains(collection, key):
...     index = most_significant(key)
...     for k, v in collection[index]:
...         if k == key:
...             return True
...     return False

# Create the collection of 10 lists
>>> collection = [[], [], [], [], [], [], [], [], [], []]

# Add some items, using key/value pairs
>>> add(collection, 123, 'a')
>>> add(collection, 456, 'b')
>>> add(collection, 789, 'c')
>>> add(collection, 101, 'c')

# Look at the collection
>>> collection
[[], [(123, 'a'), (101, 'c')], [], [],
 **[(456, 'b')], [], [], [(789, 'c')], [], []]

# Check if the contains works correctly
>>> contains(collection, 123)
True
>>> contains(collection, 1)
False

```

这段代码显然与`dict`的实现不同，但在内部实际上非常相似。因为我们可以通过简单的索引获取值为`123`的项`1`，所以在一般情况下，我们只有`O(1)`的查找成本。然而，由于`123`和`101`两个键都在`1`桶中，运行时实际上可能增加到`O(n)`，在最坏的情况下，所有键都具有相同的散列。这就是我们所说的散列冲突。

### 提示

要调试散列冲突，可以使用`hash()`函数与计数集合配对，这在*counter – keeping track of the most occurring elements*部分有讨论。

除了散列冲突性能问题，还有另一种可能让你感到惊讶的行为。当从字典中删除项时，它实际上不会立即调整内存中的字典大小。结果是复制和迭代整个字典都需要`O(m)`时间（其中 m 是字典的最大大小）；当前项数 n 不会被使用。因此，如果向`dict`中添加 1000 个项并删除 999 个项，迭代和复制仍将需要 1000 步。解决此问题的唯一方法是重新创建字典，这是`copy`和`insert`操作都会在内部执行的操作。请注意，`insert`操作期间的重新创建不是保证的，而是取决于内部可用的空闲插槽数量。

## set - 没有值的字典

`set`是一种使用散列方法获取唯一值集合的结构。在内部，它与`dict`非常相似，具有相同的散列冲突问题，但`set`有一些方便的功能需要展示：

```py
# All output in the table below is generated using this function
>>> def print_set(expression, set_):
...     'Print set as a string sorted by letters'
...     print(expression, ''.join(sorted(set_)))

>>> spam = set('spam')
>>> print_set('spam:', spam)
spam: amps

>>> eggs = set('eggs')
>>> print_set('eggs:', spam)
eggs: amps

```

前几个基本上都是预期的。在操作符处，它变得有趣起来。

| 表达式 | 输出 | 解释 |
| --- | --- | --- |
| `spam` | `amps` | 所有唯一的项。`set` 不允许重复。 |
| `eggs` | `egs` |
| `spam & eggs` | `s` | 两者中的每一项。 |
| `spam &#124; eggs` | `aegmps` | 两者中的任一项或两者都有的。 |
| `spam ^ eggs` | `aegmp` | 两者中的任一项，但不是两者都有的。 |
| `spam - eggs` | `amp` | 第一个中的每一项，但不是后者中的。 |
| `eggs - spam` | `eg` |
| `spam > eggs` | `False` | 如果后者中的每一项都在前者中，则为真。 |
| `eggs > spam` | `False` |
| `spam > sp` | `True` |
| `spam < sp` | `False` | 如果第一个中的每一项都包含在后者中，则为真。 |

`set`操作的一个有用示例是计算两个对象之间的差异。例如，假设我们有两个列表：

+   `current_users`: 组中的当前用户

+   `new_users`: 组中的新用户列表

在权限系统中，这是一个非常常见的场景——从组中批量添加和/或删除用户。在许多权限数据库中，不容易一次设置整个列表，因此你需要一个要插入的列表和一个要删除的列表。这就是`set`真正方便的地方：

```py
The set function takes a sequence as argument so the double ( is
required.
>>> current_users = set((
...     'a',
...     'b',
...     'd',
... ))

>>> new_users = set((
...     'b',
...     'c',
...     'd',
...     'e',
... ))

>>> to_insert = new_users - current_users
>>> sorted(to_insert)
['c', 'e']
>>> to_delete = current_users - new_users
>>> sorted(to_delete)
['a']
>>> unchanged = new_users & current_users
>>> sorted(unchanged)
['b', 'd']

```

现在我们有了所有被添加、删除和未更改的用户列表。请注意，`sorted`仅用于一致的输出，因为`set`与`dict`类似，没有预定义的排序顺序。

## 元组 - 不可变列表

`tuple`是一个你经常使用而甚至都没有注意到的对象。当你最初看到它时，它似乎是一个无用的数据结构。它就像一个你无法修改的列表，那么为什么不只使用`list`呢？有一些情况下，`tuple`提供了一些`list`没有的非常有用的功能。

首先，它们是可散列的。这意味着你可以将`tuple`用作`dict`中的键，这是`list`无法做到的：

```py
>>> spam = 1, 2, 3
>>> eggs = 4, 5, 6

>>> data = dict()
>>> data[spam] = 'spam'
>>> data[eggs] = 'eggs'

>>> import pprint  # Using pprint for consistent and sorted output
>>> pprint.pprint(data)
{(1, 2, 3): 'spam', (4, 5, 6): 'eggs'}

```

然而，它实际上可以比简单的数字更复杂。只要`tuple`的所有元素都是可散列的，它就可以工作。这意味着你可以使用嵌套的元组、字符串、数字和任何其他`hash()`函数返回一致结果的东西：

```py
>>> spam = 1, 'abc', (2, 3, (4, 5)), 'def'
>>> eggs = 4, (spam, 5), 6

>>> data = dict()
>>> data[spam] = 'spam'
>>> data[eggs] = 'eggs'
>>> import pprint  # Using pprint for consistent and sorted output
>>> pprint.pprint(data)
{(1, 'abc', (2, 3, (4, 5)), 'def'): 'spam',
 **(4, ((1, 'abc', (2, 3, (4, 5)), 'def'), 5), 6): 'eggs'}

```

你可以使它们变得如你所需的那样复杂。只要所有部分都是可散列的，它就会按预期运行。

也许更有用的是元组也支持元组打包和解包：

```py
# Assign using tuples on both sides
>>> a, b, c = 1, 2, 3
>>> a
1

# Assign a tuple to a single variable
>>> spam = a, (b, c)
>>> spam
(1, (2, 3))

# Unpack a tuple to two variables
>>> a, b = spam
>>> a
1
>>> b
(2, 3)

```

除了常规的打包和解包外，从 Python 3 开始，我们实际上可以使用可变数量的项目打包和解包对象：

```py
# Unpack with variable length objects which actually assigns as a
list, not a tuple
>>> spam, *eggs = 1, 2, 3, 4
>>> spam
1
>>> eggs
[2, 3, 4]

# Which can be unpacked as well of course
>>> a, b, c = eggs
>>> c
4

# This works for ranges as well
>>> spam, *eggs = range(10)
>>> spam
0
>>> eggs
[1, 2, 3, 4, 5, 6, 7, 8, 9]

# Which works both ways
>>> a
2
>>> a, b, *c = a, *eggs
>>> a, b
(2, 1)
>>> c
[2, 3, 4, 5, 6, 7, 8, 9]

```

这种方法在许多情况下都可以应用，甚至用于函数参数：

```py
>>> def eggs(*args):
...     print('args:', args)

>>> eggs(1, 2, 3)
args: (1, 2, 3)

```

同样，从函数返回多个参数也很有用：

```py
>>> def spam_eggs():
...     return 'spam', 'eggs'

>>> spam, eggs = spam_eggs()
>>> print('spam: %s, eggs: %s' % (spam, eggs))
spam: spam, eggs: eggs

```

# 高级集合

以下集合大多只是基本集合的扩展，其中一些非常简单，另一些则稍微复杂一些。不过，对于所有这些集合，了解底层结构的特性是很重要的。如果不了解它们，将很难理解这些集合的特性。

出于性能原因，有一些集合是用本机 C 代码实现的，但所有这些集合也可以很容易地在纯 Python 中实现。

## ChainMap - 字典列表

在 Python 3.3 中引入的`ChainMap`允许您将多个映射（例如字典）合并为一个。这在合并多个上下文时特别有用。例如，在查找当前作用域中的变量时，默认情况下，Python 会在`locals()`，`globals()`，最后是`builtins`中搜索。

通常，您会这样做：

```py
import builtins

builtin_vars = vars(builtins)
if key in locals():
    value = locals()[key]
elif key in globals():
    value = globals()[key]
elif key in builtin_vars:
    value = builtin_vars[key]
else:
    raise NameError('name %r is not defined' % key)
```

这样做是有效的，但至少可以说很丑陋。当然，我们可以让它更漂亮：

```py
import builtins

mappings = globals(), locals(), vars(builtins)
for mapping in mappings:
    if key in mapping:
        value = mapping[key]
        break
else:
    raise NameError('name %r is not defined' % key)
```

好多了！而且，这实际上可以被认为是一个不错的解决方案。但自从 Python 3.3 以来，它变得更容易了。现在我们可以简单地使用以下代码：

```py
import builtins
import collections

mappings = collections.ChainMap(globals(), locals(), vars(builtins))
value = mappings[key]
```

`ChainMap`集合对于命令行应用程序非常有用。最重要的配置是通过命令行参数进行的，然后是目录本地配置文件，然后是全局配置文件，最后是默认配置：

```py
import argparse
import collections

defaults = {
    'spam': 'default spam value',
    'eggs': 'default eggs value',
}

parser = argparse.ArgumentParser()
parser.add_argument('--spam')
parser.add_argument('--eggs')

args = vars(parser.parse_args())
# We need to check for empty/default values so we can't simply use vars(args)
filtered_args = {k: v for k, v in args.items() if v}

combined = collections.ChainMap(filtered_args, defaults)

print(combined ['spam'])
```

请注意，仍然可以访问特定的映射：

```py
print(combined.maps[1]['spam'])

for map_ in combined.maps:
    print(map_.get('spam'))
```

## counter - 跟踪最常出现的元素

`counter`是一个用于跟踪元素出现次数的类。它的基本用法如您所期望的那样：

```py
>>> import collections

>>> counter = collections.Counter('eggs')
>>> for k in 'eggs':
...     print('Count for %s: %d' % (k, counter[k]))
Count for e: 1
Count for g: 2
Count for g: 2
Count for s: 1

```

但是，`counter`不仅仅可以返回计数。它还有一些非常有用且快速（它使用`heapq`）的方法来获取最常见的元素。即使向计数器添加了一百万个元素，它仍然在一秒内执行：

```py
>>> import math
>>> import collections

>>> counter = collections.Counter()
>>> for i in range(0, 100000):
...    counter[math.sqrt(i) // 25] += 1

>>> for key, count in counter.most_common(5):
...     print('%s: %d' % (key, count))
11.0: 14375
10.0: 13125
9.0: 11875
8.0: 10625
12.0: 10000

```

但等等，还有更多！除了获取最频繁的元素之外，还可以像我们之前看到的`set`操作一样添加、减去、交集和"联合"计数器。那么添加两个计数器和对它们进行联合有什么区别呢？正如您所期望的那样，它们是相似的，但有一点不同。让我们看看它的工作原理：

```py
>>> import collections

>>> def print_counter(expression, counter):
...     sorted_characters = sorted(counter.elements())
...     print(expression, ''.join(sorted_characters))

>>> eggs = collections.Counter('eggs')
>>> spam = collections.Counter('spam')
>>> print_counter('eggs:', eggs)
eggs: eggs
>>> print_counter('spam:', spam)
spam: amps
>>> print_counter('eggs & spam:', eggs & spam)
eggs & spam: s
>>> print_counter('spam & eggs:', spam & eggs)
spam & eggs: s
>>> print_counter('eggs - spam:', eggs - spam)
eggs - spam: egg
>>> print_counter('spam - eggs:', spam - eggs)
spam - eggs: amp
>>> print_counter('eggs + spam:', eggs + spam)
eggs + spam: aeggmpss
>>> print_counter('spam + eggs:', spam + eggs)
spam + eggs: aeggmpss
>>> print_counter('eggs | spam:', eggs | spam)
eggs | spam: aeggmps
>>> print_counter('spam | eggs:', spam | eggs)
spam | eggs: aeggmps

```

前两个是显而易见的。`eggs`字符串只是一个包含两个"`g`"，一个"`s`"和一个"`e`"的字符序列，spam 几乎相同，但字母不同。

`spam & eggs`的结果（以及反向）也是非常可预测的。spam 和 eggs 之间唯一共享的字母是`s`，因此这就是结果。在计数方面，它只是对来自两者的共享元素执行`min(element_a, element_b)`，并得到最低值。

从 eggs 中减去字母`s`，`p`，`a`和`m`，剩下`e`和`g`。同样，从 spam 中删除`e`，`g`和`s`，剩下`p`，`a`和`m`。

现在，添加就像您所期望的那样 - 只是对两个计数器的每个元素进行逐个相加。

那么联合（OR）有什么不同呢？它获取每个计数器中元素的`max(element_a, element_b)`，而不是将它们相加；与添加的情况一样。

最后，正如前面的代码所示，elements 方法返回一个由计数重复的所有元素扩展列表。

### 注意

`Counter`对象将在执行数学运算期间自动删除零或更少的元素。

## deque - 双端队列

`deque`（双端队列）对象是最古老的集合之一。它是在 Python 2.4 中引入的，所以到目前为止已经有 10 多年的历史了。一般来说，这个对象对于大多数目的来说现在都太低级了，因为许多操作本来会使用它，现在有很好的支持库可用，但这并不使它变得不那么有用。

在内部，`deque`被创建为一个双向链表，这意味着每个项目都指向下一个和上一个项目。由于`deque`是双端的，列表本身指向第一个和最后一个元素。这使得从列表的开头/结尾添加和删除项目都是非常轻松的`O(1)`操作，因为只需要改变指向列表开头/结尾的指针，并且需要添加指针到第一个/最后一个项目，具体取决于是在开头还是结尾添加项目。

对于简单的堆栈/队列目的，使用双端队列似乎是浪费的，但性能足够好，我们不必担心产生的开销。`deque`类是完全在 C 中实现的（使用 CPython）。

它作为队列的使用非常简单：

```py
>>> import collections

>>> queue = collections.deque()
>>> queue.append(1)
>>> queue.append(2)
>>> queue
deque([1, 2])
>>> queue.popleft()
1
>>> queue.popleft()
2
>>> queue.popleft()
Traceback (most recent call last):
 **...
IndexError: pop from an empty deque

```

正如预期的那样，由于只有两个项目，我们尝试获取三个项目，所以会出现`IndexError`。

作为堆栈的使用几乎相同，但我们必须使用`pop`而不是`popleft`（或者使用`appendleft`而不是`append`）：

```py
>>> import collections

>>> queue = collections.deque()
>>> queue.append(1)
>>> queue.append(2)
>>> queue
deque([1, 2])
>>> queue.pop()
2
>>> queue.pop()
1
>>> queue.pop()
Traceback (most recent call last):
 **...
IndexError: pop from an empty deque

```

另一个非常有用的功能是`deque`可以使用`maxlen`参数作为循环队列。通过使用这个参数，它可以用来保留最后的`n`个状态消息或类似的东西：

```py
>>> import collections

>>> circular = collections.deque(maxlen=2)
>>> for i in range(5):
...     circular.append(i)
...     circular
deque([0], maxlen=2)
deque([0, 1], maxlen=2)
deque([1, 2], maxlen=2)
deque([2, 3], maxlen=2)
deque([3, 4], maxlen=2)
>>> circular
deque([3, 4], maxlen=2)

```

每当您需要单线程应用程序中的队列或堆栈类时，`deque`是一个非常方便的选择。如果您需要将对象同步到多线程操作，则`queue.Queue`类更适合。在内部，它包装了`deque`，但它是一个线程安全的替代方案。在同一类别中，还有一个用于异步操作的`asyncio.Queue`和一个用于多进程操作的`multiprocessing.Queue`。`asyncio`和多进程的示例分别可以在第七章和第十三章中找到。

## defaultdict - 具有默认值的字典

`defaultdict`绝对是我在 collections 包中最喜欢的对象。我仍然记得在它被添加到核心之前写过自己的版本。虽然它是一个相当简单的对象，但它对各种设计模式非常有用。您只需从一开始声明默认值，而不必每次都检查键的存在并添加值，这使得它非常有用。

例如，假设我们正在从连接的节点列表构建一个非常基本的图结构。

这是我们的连接节点列表（单向）：

```py
nodes = [
    ('a', 'b'),
    ('a', 'c'),
    ('b', 'a'),
    ('b', 'd'),
    ('c', 'a'),
    ('d', 'a'),
    ('d', 'b'),
    ('d', 'c'),
]
```

现在让我们将这个图放入一个普通的字典中：

```py
>>> graph = dict()
>>> for from_, to in nodes:
...     if from_ not in graph:
...         graph[from_] = []
...     graph[from_].append(to)

>>> import pprint
>>> pprint.pprint(graph)
{'a': ['b', 'c'],
 **'b': ['a', 'd'],
 **'c': ['a'],
 **'d': ['a', 'b', 'c']}

```

当然，也有一些变化，例如使用`setdefault`。但它们比必要的复杂。

真正的 Python 版本使用`defaultdict`代替：

```py
>>> import collections

>>> graph = collections.defaultdict(list)
>>> for from_, to in nodes:
...     graph[from_].append(to)

>>> import pprint
>>> pprint.pprint(graph)
defaultdict(<class 'list'>,
 **{'a': ['b', 'c'],
 **'b': ['a', 'd'],
 **'c': ['a'],
 **'d': ['a', 'b', 'c']})

```

这是一段美妙的代码吗？`defaultdict`实际上可以被看作是`counter`对象的前身。它没有`counter`那么花哨，也没有所有`counter`的功能，但在许多情况下它可以胜任：

```py
>>> counter = collections.defaultdict(int)
>>> counter['spam'] += 5
>>> counter
defaultdict(<class 'int'>, {'spam': 5})

```

`defaultdict`的默认值需要是一个可调用对象。在前面的例子中，这些是`int`和`list`，但您可以轻松地定义自己的函数来用作默认值。下面的例子就是这样做的，尽管我不建议在生产中使用，因为它缺乏一些可读性。然而，我相信这是 Python 强大之处的一个美好例子。

这是我们如何在一行 Python 中创建一个`tree`：

```py
import collections
def tree(): return collections.defaultdict(tree)
```

太棒了，不是吗？这是我们实际上如何使用它的方式：

```py
>>> import json
>>> import collections

>>> def tree():
...     return collections.defaultdict(tree)

>>> colours = tree()
>>> colours['other']['black'] = 0x000000
>>> colours['other']['white'] = 0xFFFFFF
>>> colours['primary']['red'] = 0xFF0000
>>> colours['primary']['green'] = 0x00FF00
>>> colours['primary']['blue'] = 0x0000FF
>>> colours['secondary']['yellow'] = 0xFFFF00
>>> colours['secondary']['aqua'] = 0x00FFFF
>>> colours['secondary']['fuchsia'] = 0xFF00FF

>>> print(json.dumps(colours, sort_keys=True, indent=4))
{
 **"other": {
 **"black": 0,
 **"white": 16777215
 **},
 **"primary": {
 **"blue": 255,
 **"green": 65280,
 **"red": 16711680
 **},
 **"secondary": {
 **"aqua": 65535,
 **"fuchsia": 16711935,
 **"yellow": 16776960
 **}
}

```

这个好处是你可以让它变得更深。由于`defaultdict`的基础，它会递归生成自己。

## namedtuple - 带有字段名称的元组

`namedtuple`对象确实就像名字暗示的那样 - 一个带有名称的元组。它有一些有用的用例，尽管我必须承认我在实际中并没有找到太多用例，除了一些 Python 模块，比如 inspect 和`urllib.parse`。2D 或 3D 空间中的点是它明显有用的一个很好的例子：

```py
>>> import collections

>>> Point = collections.namedtuple('Point', ['x', 'y', 'z'])
>>> point_a = Point(1, 2, 3)
>>> point_a
Point(x=1, y=2, z=3)

>>> point_b = Point(x=4, z=5, y=6)
>>> point_b
Point(x=4, y=6, z=5)

```

关于`namedtuple`，并没有太多可以说的；它做你期望的事情，最大的优势是属性可以通过名称和索引执行，这使得元组解包非常容易：

```py
>>> x, y, z = point_a
>>> print('X: %d, Y: %d, Z: %d' % (x, y, z))
X: 1, Y: 2, Z: 3
>>> print('X: %d, Y: %d, Z: %d' % point_b)
X: 4, Y: 6, Z: 5
>>> print('X: %d' % point_a.x)

```

## enum - 一组常量

`enum`包与`namedtuple`非常相似，但目标和接口完全不同。基本的`enum`对象使得在模块中拥有常量变得非常容易，同时避免了魔术数字。这是一个基本的例子：

```py
>>> import enum

>>> class Color(enum.Enum):
...     red = 1
...     green = 2
...     blue = 3

>>> Color.red
<Color.red: 1>
>>> Color['red']
<Color.red: 1>
>>> Color(1)
<Color.red: 1>
>>> Color.red.name
'red'
>>> Color.red.value
1
>>> isinstance(Color.red, Color)
True
>>> Color.red is Color['red']
True
>>> Color.red is Color(1)
True

```

`enum`包的一些方便功能是，对象是可迭代的，可以通过值的数字和文本表示进行访问，并且，通过适当的继承，甚至可以与其他类进行比较。

以下代码展示了基本 API 的使用：

```py
>>> for color in Color:
...     color
<Color.red: 1>
<Color.green: 2>
<Color.blue: 3>

>>> colors = dict()
>>> colors[Color.green] = 0x00FF00
>>> colors
{<Color.green: 2>: 65280}

```

还有更多。`enum`包中较少为人知的可能性之一是，你可以通过特定类型的继承使值比较起作用，这对于任何类型都有效，不仅仅是整数，还包括（你自己的）自定义类型。

这是常规的`enum`：

```py
>>> import enum

>>> class Spam(enum.Enum):
...     EGGS = 'eggs'

>>> Spam.EGGS == 'eggs'
False

```

以下是带有`str`继承的`enum`：

```py
>>> import enum

>>> class Spam(str, enum.Enum):
...     EGGS = 'eggs'

>>> Spam.EGGS == 'eggs'
True

```

## OrderedDict - 插入顺序很重要的字典

`OrderdDict`是一个跟踪插入顺序的`dict`。而普通的`dict`会按照哈希的顺序返回键，`OrderedDict`会按照插入的顺序返回键。所以，它不是按键或值排序的，但这也很容易实现：

```py
>>> import collections

>>> spam = collections.OrderedDict()
>>> spam['b'] = 2
>>> spam['c'] = 3
>>> spam['a'] = 1
>>> spam
OrderedDict([('b', 2), ('c', 3), ('a', 1)])

>>> for key, value in spam.items():
...     key, value
('b', 2)
('c', 3)
('a', 1)

>>> eggs = collections.OrderedDict(sorted(spam.items()))
>>> eggs
OrderedDict([('a', 1), ('b', 2), ('c', 3)])

```

虽然你可能猜到了它是如何工作的，但内部可能会让你有点惊讶。我知道我原本期望的实现方式是不同的。

在内部，`OrderedDict`使用普通的`dict`来存储键/值，并且除此之外，它还使用一个双向链表来跟踪下一个/上一个项目。为了跟踪反向关系（从双向链表返回到键），还有一个额外的`dict`存储在内部。

简而言之，`OrderedDict`可以是一个非常方便的工具，用来保持你的`dict`排序，但它是有代价的。这个系统的结构使得`set`和`get`非常快速（O(1)），但是与普通的`dict`相比，这个对象仍然更加沉重（内存使用量增加一倍或更多）。当然，在许多情况下，内部对象的内存使用量将超过`dict`本身的内存使用量，但这是需要记住的一点。

## heapq - 有序列表

`heapq`模块是一个非常好的小模块，它可以非常容易地在 Python 中创建一个优先队列。这种结构总是可以在最小的（或最大的，取决于实现）项目上进行最小的努力。API 非常简单，它的使用最好的例子之一可以在`OrderedDict`对象中看到。你可能不想直接使用`heapq`，但了解内部工作原理对于分析诸如`OrderedDict`之类的类是很重要的。

### 提示

如果你正在寻找一个结构来保持你的列表始终排序，可以尝试使用`bisect`模块。

基本用法非常简单：

```py
>>> import heapq

>>> heap = [1, 3, 5, 7, 2, 4, 3]
>>> heapq.heapify(heap)
>>> heap
[1, 2, 3, 7, 3, 4, 5]

>>> while heap:
...     heapq.heappop(heap), heap
(1, [2, 3, 3, 7, 5, 4])
(2, [3, 3, 4, 7, 5])
(3, [3, 5, 4, 7])
(3, [4, 5, 7])
(4, [5, 7])
(5, [7])
(7, [])

```

这里有一件重要的事情需要注意 - 你可能已经从前面的例子中理解了 - `heapq`模块并不创建一个特殊的对象。它只是一堆方法，用于将常规列表视为`heap`。这并不使它变得不那么有用，但这是需要考虑的一点。你可能也会想为什么`heap`没有排序。实际上，它是排序的，但不是你期望的方式。如果你将`heap`视为一棵树，它就会变得更加明显：

```py
   1
 2   3
7 3 4 5
```

最小的数字总是在顶部，最大的数字总是在树的底部。因此，找到最小的数字非常容易，但找到最大的数字就不那么容易了。要获得堆的排序版本，我们只需要不断地删除树的顶部，直到所有项目都消失。

## bisect - 排序列表

我们在前一段中看到了`heapq`模块，它使得从列表中始终获取最小的数字变得非常简单，因此也很容易对对象列表进行排序。`heapq`模块将项目附加到形成类似树的结构，而`bisect`模块以使它们保持排序的方式插入项目。一个很大的区别是，使用`heapq`模块添加/删除项目非常轻便，而使用`bisect`模块查找项目非常轻便。如果您的主要目的是搜索，那么`bisect`应该是您的选择。

与`heapq`一样，`bisect`并不真正创建一个特殊的数据结构。它只是在一个标准的`list`上操作，并期望该`list`始终保持排序。重要的是要理解这一点的性能影响；仅仅使用`bisect`算法向列表添加项目可能会非常慢，因为在列表上插入需要`O(n)`的时间。实际上，使用 bisect 创建一个排序列表需要`O(n*n)`的时间，这相当慢，特别是因为使用`heapq`或 sorted 创建相同的排序列表只需要`O(n * log(n))`的时间。

### 注意

`log(n)`是指以 2 为底的对数函数。要计算这个值，可以使用`math.log2()`函数。这意味着每当数字的大小加倍时，值就会增加 1。对于`n=2`，`log(n)`的值为`1`，因此对于`n=4`和`n=8`，对数值分别为`2`和`3`。

这意味着 32 位数字，即`2**32 = 4294967296`，具有`32`的对数。

如果您有一个排序的结构，并且只需要添加一个单个项目，那么可以使用`bisect`算法进行插入。否则，通常更快的方法是简单地附加项目，然后调用`.sort()`。

为了说明，我们有这些行：

```py
>>> import bisect

Using the regular sort:
>>> sorted_list = []
>>> sorted_list.append(5)  # O(1)
>>> sorted_list.append(3)  # O(1)
>>> sorted_list.append(1)  # O(1)
>>> sorted_list.append(2)  # O(1)
>>> sorted_list.sort()  # O(n * log(n)) = O(4 * log(4)) = O(8)
>>> sorted_list
[1, 2, 3, 5]

Using bisect:
>>> sorted_list = []
>>> bisect.insort(sorted_list, 5)  # O(n) = O(1)
>>> bisect.insort(sorted_list, 3)  # O(n) = O(2)
>>> bisect.insort(sorted_list, 1)  # O(n) = O(3)
>>> bisect.insort(sorted_list, 2)  # O(n) = O(4)
>>> sorted_list
[1, 2, 3, 5]

```

对于少量项目，差异是可以忽略的，但它很快就会增长到一个差异很大的程度。对于`n=4`，差异只是`4 * 1 + 8 = 12`和`1 + 2 + 3 + 4 = 10`之间，使得 bisect 解决方案更快。但是，如果我们要插入 1,000 个项目，那么结果将是`1000 + 1000 * log(1000) = 10966`与`1 + 2 + … 1000 = 1000 * (1000 + 1) / 2 = 500500`。因此，在插入许多项目时要非常小心。

不过，在列表中进行搜索非常快；因为它是排序的，我们可以使用一个非常简单的二分搜索算法。例如，如果我们想要检查列表中是否存在一些数字呢？

```py
>>> import bisect

>>> sorted_list = [1, 2, 3, 5]
>>> def contains(sorted_list, value):
...     i = bisect.bisect_left(sorted_list, value)
...     return i < len(sorted_list) and sorted_list[i] == value

>>> contains(sorted_list, 2)
True
>>> contains(sorted_list, 4)
False
>>> contains(sorted_list, 6)
False

```

如您所见，`bisect_left`函数找到了数字应该在的位置。这实际上也是`insort`函数所做的；它通过搜索数字的位置来将数字插入到正确的位置。

那么这与`sorted_list`中的常规值有什么不同呢？最大的区别在于`bisect`在内部执行二分搜索，这意味着它从中间开始，根据值是大还是小而向左或向右跳转。为了说明，我们将在从`0`到`14`的数字列表中搜索`4`：

```py
sorted_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
Step 1: 4 > 7                       ^
Step 2: 4 > 3           ^
Step 3: 4 > 5                 ^
Step 4: 4 > 5              ^
```

如您所见，经过仅四步（实际上是三步；第四步只是为了说明），我们已经找到了我们搜索的数字。根据数字（例如`7`），可能会更快，但是找到一个数字永远不会超过`O(log(n))`步。

使用常规列表，搜索将简单地遍历所有项目，直到找到所需的项目。如果你幸运的话，它可能是你遇到的第一个数字，但如果你不幸的话，它可能是最后一个项目。对于 1,000 个项目来说，这将是 1000 步和`log(1000) = 10`步之间的差异。

# 总结

Python 内置了一些非常有用的集合。由于越来越多的集合定期添加，最好的做法就是简单地跟踪集合手册。你是否曾经想过任何结构是如何工作的，或者为什么会这样？只需在这里查看源代码：

[`hg.python.org/cpython/file/default/Lib/collections/__init__.py`](https://hg.python.org/cpython/file/default/Lib/collections/__init__.py)

完成本章后，你应该了解核心集合和集合模块中最重要的集合，更重要的是这些集合在几种情景下的性能特征。在应用程序中选择正确的数据结构是你的代码将经历的最重要的性能因素，这对于任何程序员来说都是必不可少的知识。

接下来，我们将继续讨论函数式编程，其中包括`lambda`函数、`list`推导、`dict`推导、`set`推导以及一系列相关主题。这包括一些涉及的数学背景信息，可能会很有趣，但可以安全地跳过。
