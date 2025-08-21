# 第二章：语法最佳实践-类级别以下

编写高效的语法能力随着时间自然而然地产生。如果你回顾一下你的第一个程序，你可能会同意这一点。正确的语法会让你眼前一亮，而错误的语法会让人感到不安。

除了实现的算法和程序的架构设计，对程序的编写方式进行精心设计对其未来的发展至关重要。许多程序因其晦涩的语法、不清晰的 API 或非常规的标准而被抛弃并从头开始重写。

但是 Python 在过去几年里发生了很大的变化。因此，如果你被邻居（当地 Ruby 开发者用户组的一个嫉妒的家伙）绑架了一段时间并远离了新闻，你可能会对它的新功能感到惊讶。从最早的版本到当前版本（此时为 3.5），已经进行了许多增强，使语言更加清晰、干净和易于编写。Python 的基础并没有发生根本性的变化，但现在玩耍它们的工具更加人性化。

本章介绍了现代语法的最重要元素以及有关它们使用的提示：

+   列表推导式

+   迭代器和生成器

+   描述符和属性

+   装饰器

+   `with`和`contextlib`

有关代码性能提升或内存使用的速度改进的提示在第十一章 *优化-一般原则和分析技术*和第十二章 *优化-一些强大的技术*中有所涉及。

# Python 的内置类型

Python 提供了一组很棒的数据类型。这对于数字类型和集合类型都是如此。关于数字类型，它们的语法没有什么特别之处。当然，对于每种类型的字面量定义和一些（也许）不太为人所知的运算符细节有一些差异，但对于开发人员来说，选择余地并不多。当涉及到集合和字符串时情况就不同了。尽管"应该只有一种方法来做某事"的信条，但 Python 开发人员确实有很多选择。对于初学者来说，一些看起来直观简单的代码模式经常被有经验的程序员认为是非*Pythonic*的，因为它们要么效率低下，要么太啰嗦。

解决常见问题的*Pythonic*模式（许多程序员称之为习语）通常看起来只是美学。这是完全错误的。大多数习语是由 Python 内部实现以及内置结构和模块的工作方式驱动的。了解更多这样的细节对于对语言的深入理解至关重要。此外，社区本身也不乏关于 Python 工作原理的神话和刻板印象。只有通过自己深入挖掘，你才能判断关于 Python 的流行说法中哪些是真的。

## 字符串和字节

对于只习惯于在 Python 2 中编程的程序员来说，字符串可能会带来一些困惑。在 Python 3 中，只有一种数据类型能够存储文本信息。它是`str`或者简单地说，字符串。它是一个存储 Unicode 码点的不可变序列。这是与 Python 2 的主要区别，其中`str`表示字节字符串-现在由`bytes`对象处理（但处理方式并不完全相同）。

Python 中的字符串是序列。这一事实应该足以将它们包括在涵盖其他容器类型的部分中，但它们与其他容器类型在一个重要的细节上有所不同。字符串对它们可以存储的数据类型有非常具体的限制，那就是 Unicode 文本。

`字节`及其可变替代品（`bytearray`）与`str`的不同之处在于只允许字节作为序列值——范围在`0 <= x < 256`的整数。这可能会在开始时造成困惑，因为打印时它们可能看起来与字符串非常相似：

```py
>>> print(bytes([102, 111, 111]))
b'foo'

```

当将`bytes`和`bytearray`转换为另一种序列类型（如`list`或`tuple`）时，它们的真实性质就显露出来了：

```py
>>> list(b'foo bar')
[102, 111, 111, 32, 98, 97, 114]
>>> tuple(b'foo bar')
(102, 111, 111, 32, 98, 97, 114)

```

Python 3 的许多争议都是关于打破字符串字面量的向后兼容性以及如何处理 Unicode。从 Python 3.0 开始，每个未加前缀的字符串字面量都是 Unicode。因此，用单引号（`'`）、双引号（`"`）或三引号（单引号或双引号）括起来的字面量代表`str`数据类型：

```py
>>> type("some string")
<class 'str'>

```

在 Python 2 中，Unicode 字面量需要`u`前缀（如`u"some string"`）。这个前缀仍然允许用于向后兼容（从 Python 3.3 开始），但在 Python 3 中没有任何语法意义。

在之前的一些示例中已经介绍了字节字面量，但为了保持一致，让我们明确介绍其语法。字节字面量也可以用单引号、双引号或三引号括起来，但必须以`b`或`B`前缀开头：

```py
>>> type(b"some bytes")
<class 'bytes'>

```

请注意，Python 语法中没有`bytearray`字面量。

最后但同样重要的是，Unicode 字符串包含与字节表示独立的“抽象”文本。这使它们无法在磁盘上保存或在网络上传输而不进行编码为二进制数据。有两种方法可以将字符串对象编码为字节序列：

+   使用`str.encode(encoding, errors)`方法，使用注册的编解码器对字符串进行编码。编解码器使用`encoding`参数指定，默认为`'utf-8'`。第二个 errors 参数指定错误处理方案。它可以是`'strict'`（默认值）、`'ignore'`、`'replace'`、`'xmlcharrefreplace'`或任何其他已注册的处理程序（参考内置`codecs`模块文档）。

+   使用`bytes(source, encoding, errors)`构造函数创建一个新的字节序列。当源是`str`类型时，`encoding`参数是必需的，且没有默认值。`encoding`和`errors`参数的使用与`str.encode()`方法相同。

由`bytes`表示的二进制数据可以以类似的方式转换为字符串：

+   使用`bytes.decode(encoding, errors)`方法，使用为编码注册的编解码器对字节进行解码。此方法的参数与`str.encode()`的参数具有相同的含义和默认值。

+   使用`str(source, encoding, error)`构造函数创建一个新的字符串实例。与`bytes()`构造函数类似，`str()`调用中的`encoding`参数没有默认值，如果字节序列用作源，则必须提供。

### 提示

**命名——字节与字节字符串**

由于 Python 3 中的更改，一些人倾向于将`bytes`实例称为字节字符串。这主要是由于历史原因——Python 3 中的`bytes`是与 Python 2 中的`str`类型最接近的序列类型（但不完全相同）。但是，`bytes`实例是字节序列，也不需要表示文本数据。因此，为了避免混淆，最好总是将它们称为`bytes`或字节序列，尽管它们与字符串相似。在 Python 3 中，字符串的概念是保留给文本数据的，现在始终是`str`。

### 实现细节

Python 字符串是不可变的。这也适用于字节序列。这是一个重要的事实，因为它既有优点又有缺点。它还影响了在 Python 中高效处理字符串的方式。由于不可变性，字符串可以用作字典键或`set`集合元素，因为一旦初始化，它们将永远不会改变其值。另一方面，每当需要修改字符串（即使只有微小的修改）时，都需要创建一个全新的实例。幸运的是，`bytearray`作为`bytes`的可变版本不会引入这样的问题。字节数组可以通过项目赋值进行就地修改（无需创建新对象），并且可以像列表一样动态调整大小 - 使用附加、弹出、插入等。

### 字符串连接

知道 Python 字符串是不可变的这个事实会在需要连接多个字符串实例时带来一些问题。如前所述，连接任何不可变序列都会导致创建一个新的序列对象。考虑到通过多个字符串的重复连接构建新字符串，如下所示：

```py
s = ""
for substring in substrings:
    s += substring
```

这将导致总字符串长度的二次运行时成本。换句话说，这是非常低效的。为了处理这种情况，有`str.join()`方法可用。它接受字符串的可迭代对象作为参数并返回一个连接的字符串。因为它是方法，实际的习惯用法使用空字符串文字作为方法的来源：

```py
s = "".join(substrings)
```

提供此方法的字符串将用作连接的子字符串之间的分隔符；请考虑以下示例：

```py
>>> ','.join(['some', 'comma', 'separated', 'values'])
'some,comma,separated,values'

```

值得记住的是，仅仅因为它更快（特别是对于大型列表），并不意味着在每个需要连接两个字符串的情况下都应该使用`join()`方法。尽管它是一个广泛认可的习惯用法，但它并不会提高代码的可读性 - 可读性很重要！还有一些情况下，`join()`可能不如普通的加法连接表现得好。以下是一些例子：

+   如果子字符串的数量很少，并且它们尚未被某个可迭代对象包含 - 在某些情况下，创建新序列的开销可能会掩盖使用`join()`的收益。

+   在连接短文字时，由于 CPython 中的常量折叠，一些复杂的文字（不仅仅是字符串），例如`'a' + 'b' + 'c'`到`'abc'`可以在编译时转换为更短的形式。当然，这仅对相对较短的常量（文字）启用。

最终，如果字符串连接的数量事先已知，最佳的可读性是通过适当的字符串格式化来确保的，可以使用`str.format()`方法或`%`运算符。在性能不是关键或优化字符串连接的收益非常小的代码部分，建议使用字符串格式化作为最佳选择。

### 提示

**常量折叠和窥孔优化器**

CPython 在编译源代码上使用窥孔优化器以提高性能。该优化器直接在 Python 的字节码上实现了许多常见的优化。如前所述，常量折叠就是其中之一。生成的常量受硬编码值的长度限制。在 Python 3.5 中，它仍然不变，等于 20。无论如何，这个特定的细节更像是一个好奇心，而不是日常编程中可以依赖的东西。有关窥孔优化器执行的其他有趣优化的信息可以在 Python 源代码的`Python/peephole.c`文件中找到。

## 集合

Python 提供了一系列内置的数据集合，如果选择得当，可以有效地解决许多问题。您可能已经知道的类型是那些具有专用文字的类型：

+   列表

+   元组

+   字典

+   集合

当然，Python 不仅限于这四种选择，并通过其标准库扩展了可能的选择列表。在许多情况下，解决问题的解决方案可能就像为数据结构做出良好选择一样简单。本书的这一部分旨在通过深入了解可能的选择来简化这样的决定。

### 列表和元组

Python 中最基本的两种集合类型是列表和元组，它们都表示对象的序列。它们之间的基本区别对于任何花费了几个小时以上的 Python 用户来说应该是显而易见的—列表是动态的，因此可以改变其大小，而元组是不可变的（它们在创建后无法修改）。

尽管元组具有许多各种优化，使得小对象的分配/释放变得快速，但它们是存储元素位置本身信息的推荐数据类型。例如，元组可能是存储一对(x, y)坐标的良好选择。无论如何，关于元组的细节都相当无聊。在本章的范围内，它们唯一重要的事情是`tuple`是**不可变的**，因此**可散列的**。这意味着什么将在*字典*部分中介绍。比元组更有趣的是它的动态对应物`list`，它是如何工作的，以及如何有效地处理它。

#### 实现细节

许多程序员很容易将 Python 的`list`类型与其他语言的标准库中经常找到的链表概念混淆，比如 C、C++或 Java。实际上，在 CPython 中，列表根本不是列表。在 CPython 中，列表是作为可变长度数组实现的。尽管这些实现细节通常在这些项目中没有记录，但这对于 Jython 和 IronPython 等其他实现也是正确的。造成这种混淆的原因很明显。这种数据类型被命名为**list**，并且还具有可以从任何链表实现中预期的接口。

这为什么重要，意味着什么？列表是最流行的数据结构之一，它们的使用方式极大地影响了每个应用程序的性能。此外，CPython 是最流行和使用最广泛的实现，因此了解其内部实现细节至关重要。

具体来说，Python 中的列表是对其他对象的连续数组的引用。指向此数组的指针和长度存储在列表头结构中。这意味着每次添加或删除项目时，都需要调整引用数组的大小（重新分配）。幸运的是，在 Python 中，这些数组是以指数过分分配创建的，因此不是每个操作都需要调整大小。这就是为什么在复杂度方面，附加和弹出元素的摊销成本可以很低。不幸的是，在 Python 中，一些在普通链表中被认为是“便宜”的其他操作具有相对较高的计算复杂度：

+   使用`list.insert`方法在任意位置插入项目—复杂度为 O(n)

+   使用`list.delete`或使用`del`删除项目—复杂度为 O(n)

在这里，*n*是列表的长度。至少使用索引检索或设置元素是一个与列表大小无关的操作。以下是大多数列表操作的平均时间复杂度的完整表格：

| 操作 | 复杂度 |
| --- | --- |
| - --- | --- |
| - 复制 | O(n) |
| - 添加 | O(1) |
| - 插入 | O(n) |
| - 获取项目 | O(1) |
| - 删除项目 | O(n) |
| - 迭代 | O(n) |
| - 获取长度为*k*的切片 | O(k) |
| - 删除切片 | O(n) |
| - 设置长度为*k*的切片 | O(k+n) |
| - 扩展 | O(k) |
| - 乘以*k* | O(nk) |
| - 测试存在性（`element in list`） | O(n) |
| - `min()`/`max()` | O(n) |
| - 获取长度 | O(1) |

在需要真正的链表（或者简单地说，具有`appends`和`pop`的数据结构，复杂度为 O(1)）的情况下，Python 在`collections`内置模块中提供了`deque`。这是栈和队列的一般化，应该在需要双向链表的任何地方都能正常工作。

#### 列表推导

正如您可能知道的，编写这样的代码是痛苦的：

```py
>>> evens = []
>>> for i in range(10):
...     if i % 2 == 0:
...         evens.append(i)
...** 
>>> evens
[0, 2, 4, 6, 8]

```

这对于 C 语言可能有效，但实际上对于 Python 来说会使事情变慢，因为：

+   它使解释器在每次循环中工作，以确定序列的哪一部分必须被更改

+   它使您保持一个计数器来跟踪哪个元素必须被处理

+   它需要在每次迭代时执行额外的函数查找，因为`append()`是列表的方法

列表推导是这种模式的正确答案。它使用了自动化前一种语法的部分的奇怪特性：

```py
>>> [i for i in range(10) if i % 2 == 0]
[0, 2, 4, 6, 8]

```

除了这种写法更有效外，它更短，涉及的元素更少。在更大的程序中，这意味着更少的错误和更容易阅读和理解的代码。

### 提示

**列表推导和内部数组调整**

有一种迷思在一些 Python 程序员中流传，即列表推导可以是内部数组代表列表对象必须在每次添加时调整大小的一种变通方法。有人说数组将在恰到好处的大小时只分配一次。不幸的是，这是不正确的。

在计算推导时，解释器无法知道最终容器的大小，也无法为其预分配数组的最终大小。因此，内部数组的重新分配与`for`循环中的模式相同。然而，在许多情况下，使用推导创建列表既更清晰又更快，而不是使用普通循环。

#### 其他习惯用法

Python 习惯用法的另一个典型例子是使用`enumerate`。这个内置函数提供了一种方便的方法，在循环中使用序列时获得索引。考虑以下代码片段作为例子：

```py
>>> i = 0
>>> for element in ['one', 'two', 'three']:
...     print(i, element)
...     i += 1
...
0 one
1 two
2 three

```

这可以被以下更短的代码替换：

```py
>>> for i, element in enumerate(['one', 'two', 'three']):
...     print(i, element)
...
0 one
1 two
2 three

```

当需要将多个列表（或任何可迭代对象）的元素以一对一的方式聚合时，可以使用内置的`zip()`函数。这是对两个相同大小的可迭代对象进行统一迭代的非常常见的模式：

```py
>>> for item in zip([1, 2, 3], [4, 5, 6]):
...     print(item)
...** 
(1, 4)
(2, 5)
(3, 6)

```

请注意，`zip()`的结果可以通过另一个`zip()`调用进行反转：

```py
>>> for item in zip(*zip([1, 2, 3], [4, 5, 6])):
...     print(item)
...** 
(1, 2, 3)
(4, 5, 6)

```

另一个流行的语法元素是序列解包。它不仅限于列表和元组，而且适用于任何序列类型（甚至字符串和字节序列）。它允许您将一系列元素解包到另一组变量中，只要在赋值运算符的左侧有与序列中元素数量相同的变量：

```py
>>> first, second, third = "foo", "bar", 100
>>> first
'foo'
>>> second
'bar'
>>> third
100

```

解包还允许您使用星号表达式捕获单个变量中的多个元素，只要它可以被明确解释。解包也可以在嵌套序列上执行。当在由序列构建的一些复杂数据结构上进行迭代时，这可能会很方便。以下是一些更复杂解包的示例：

```py
>>> # starred expression to capture rest of the sequence
>>> first, second, *rest = 0, 1, 2, 3
>>> first
0
>>> second
1
>>> rest
[2, 3]

>>> # starred expression to capture middle of the sequence
>>> first, *inner, last = 0, 1, 2, 3
>>> first
0
>>> inner
[1, 2]
>>> last
3

>>> # nested unpacking
>>> (a, b), (c, d) = (1, 2), (3, 4)
>>> a, b, c, d
(1, 2, 3, 4)

```

### 字典

字典是 Python 中最通用的数据结构之一。`dict`允许将一组唯一的键映射到值，如下所示：

```py
{
    1: ' one',
    2: ' two',
    3: ' three',
}
```

字典文字是一件非常基本的事情，你应该已经知道它们。无论如何，Python 允许程序员使用类似于前面提到的列表推导的推导来创建一个新的字典。这是一个非常简单的例子：

```py
squares = {number: number**2 for number in range(100)}
```

重要的是，使用列表推导的相同好处也适用于字典推导。因此，在许多情况下，它们更有效、更短、更清晰。对于更复杂的代码，当需要许多`if`语句或函数调用来创建字典时，简单的`for`循环可能是更好的选择，特别是如果它提高了可读性。

对于 Python 3 中的 Python 程序员，有一个关于迭代字典元素的重要说明。字典方法：`keys()`、`values()`和`items()`不再具有列表作为它们的返回值类型。此外，它们的对应方法`iterkeys()`、`itervalues()`和`iteritems()`在 Python 3 中已经消失，而不是返回迭代器。现在，`keys()`、`values()`和`items()`返回的是视图对象：

+   `keys()`: 这返回`dict_keys`对象，提供了字典的所有键的视图

+   `values()`: 这返回`dict_values`对象，提供了字典的所有值的视图

+   `items()`: 这返回`dict_items`对象，提供了字典的所有`(key, value)`两个元组的视图

视图对象以动态方式查看字典内容，因此每次字典发生更改，视图都会反映这些更改，如下例所示：

```py
>>> words = {'foo': 'bar', 'fizz': 'bazz'}
>>> items = words.items()
>>> words['spam'] = 'eggs'
>>> items
dict_items([('spam', 'eggs'), ('fizz', 'bazz'), ('foo', 'bar')])

```

视图对象将旧方法的返回列表的行为与它们的“iter”对应方法返回的迭代器相结合。视图不需要在内存中冗余存储所有值（像列表一样），但仍然允许获取它们的长度（使用`len`）和测试成员资格（使用`in`子句）。视图当然是可迭代的。

最后一个重要的事情是`keys()`和`values()`方法返回的两个视图确保了相同的键和值顺序。在 Python 2 中，如果要确保检索到的键和值的顺序相同，你不能在这两个调用之间修改字典内容。`dict_keys`和`dict_values`现在是动态的，因此即使在`keys()`和`values()`调用之间更改字典的内容，迭代的顺序也在这两个视图之间保持一致。

#### 实现细节

CPython 使用伪随机探测的哈希表作为字典的底层数据结构。这似乎是一个非常深入的实现细节，但在不久的将来很不可能改变，因此对于程序员来说也是一个非常有趣的事实。

由于这个实现细节，只有**可哈希**的对象才能用作字典键。如果一个对象是可哈希的，那么它在其生命周期内的哈希值永远不会改变，并且可以与不同的对象进行比较。每个 Python 的内置类型都是不可变的，因此也是可哈希的。可变类型，如列表、字典和集合，是不可哈希的，因此不能用作字典键。定义类型是否可哈希的协议由两个方法组成：

+   `__hash__`: 这提供了内部`dict`实现所需的哈希值（作为整数）。对于用户定义类的实例对象，它是从它们的`id()`派生的。

+   `__eq__`: 这比较具有相同值的两个对象。默认情况下，所有用户定义类的实例对象都不相等，除了它们自己。

相等的两个对象必须具有相同的哈希值。反之则不需要成立。这意味着哈希碰撞是可能的——具有相同哈希的两个对象可能不相等。这是允许的，每个 Python 实现都必须能够解决哈希碰撞。CPython 使用**开放寻址**来解决这种碰撞（[`en.wikipedia.org/wiki/Open_addressing`](https://en.wikipedia.org/wiki/Open_addressing)）。然而，碰撞的概率极大地影响性能，如果碰撞概率很高，字典将无法从其内部优化中受益。

虽然三个基本操作：添加、获取和删除项目的平均时间复杂度为 O(1)，但它们的摊销最坏情况复杂度要高得多——O(n)，其中*n*是当前字典大小。此外，如果将用户定义的类对象用作字典键，并且它们的哈希不当（存在高风险的碰撞），那么这将对字典的性能产生巨大的负面影响。CPython 字典的完整时间复杂度表如下：

| 操作 | 平均复杂度 | 分摊最坏情况复杂度 |
| --- | --- | --- |
| 获取项 | O(1) | O(n) |
| 集合项 | O(1) | O(n) |
| 删除项 | O(1) | O(n) |
| 复制 | O(n) | O(n) |
| 迭代 | O(n) | O(n) |

还有一点很重要，那就是复制和迭代字典的最坏情况复杂度中的*n*是字典曾经达到的最大大小，而不是当前的项数。换句话说，迭代曾经很大但在时间上大大缩小的字典可能需要花费出乎意料的长时间。因此，在某些情况下，如果需要经常迭代，可能最好创建一个新的字典对象，而不是仅仅从以前的字典中删除元素。

#### 弱点和替代方案

使用字典的一个常见陷阱是它们不保留添加新键的顺序。在某些情况下，当字典键使用连续的键，其哈希值也是连续的值（例如使用整数）时，由于字典的内部实现，结果顺序可能是相同的：

```py
>>> {number: None for number in range(5)}.keys()
dict_keys([0, 1, 2, 3, 4])

```

然而，使用其他哈希方式不同的数据类型表明顺序不会被保留。以下是 CPython 的一个例子：

```py
>>> {str(number): None for number in range(5)}.keys()
dict_keys(['1', '2', '4', '0', '3'])
>>> {str(number): None for number in reversed(range(5))}.keys()
dict_keys(['2', '3', '1', '4', '0'])

```

如前面的代码所示，结果顺序既取决于对象的哈希，也取决于添加元素的顺序。这是不可靠的，因为它可能会随着不同的 Python 实现而变化。

然而，在某些情况下，开发人员可能需要保留添加顺序的字典。幸运的是，Python 标准库在`collections`模块中提供了一个有序字典`OrderedDict`。它可以选择接受一个可迭代对象作为初始化参数：

```py
>>> from collections import OrderedDict
>>> OrderedDict((str(number), None) for number in range(5)).keys()
odict_keys(['0', '1', '2', '3', '4'])

```

它还具有一些额外的功能，比如使用`popitem()`方法从两端弹出项，或者使用`move_to_end()`方法将指定的元素移动到其中一个端点。有关该集合的完整参考，请参阅 Python 文档（参见[`docs.python.org/3/library/collections.html`](https://docs.python.org/3/library/collections.html)）。

另一个重要的注意事项是，在非常古老的代码库中，`dict`可能被用作保证元素唯一性的原始集合实现。虽然这会给出正确的结果，但除非针对的是低于 2.3 的 Python 版本，否则应该避免这样做。以这种方式使用字典在资源方面是浪费的。Python 有一个内置的`set`类型来实现这个目的。实际上，它在 CPython 中有一个非常相似的内部实现，但也提供了一些额外的功能以及特定的与集合相关的优化。

### 集合

集合是一种非常健壮的数据结构，主要在元素的顺序不如它们的唯一性和测试效率重要的情况下非常有用。它们与类似的数学概念非常相似。集合以两种形式作为内置类型提供：

+   `set()`: 这是一个可变的、无序的、有限的唯一不可变（可哈希）对象的集合

+   `frozenset()`: 这是一个不可变的、可哈希的、无序的唯一不可变（可哈希）对象的集合

`frozenset()`的不可变性使其可以用作字典键，也可以用作其他`set()`和`frozenset()`元素。普通的可变`set()`不能在另一个集合或 frozenset 内容中使用，否则会引发`TypeError`：

```py
>>> set([set([1,2,3]), set([2,3,4])])
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'set'

```

以下的集合初始化是完全正确的：

```py
>>> set([frozenset([1,2,3]), frozenset([2,3,4])])
{frozenset({1, 2, 3}), frozenset({2, 3, 4})}
>>> frozenset([frozenset([1,2,3]), frozenset([2,3,4])])
frozenset({frozenset({1, 2, 3}), frozenset({2, 3, 4})})

```

可变集合可以通过三种方式创建：

+   使用接受可选可迭代对象作为初始化参数的`set()`调用，比如`set([0, 1, 2])`

+   使用集合推导，例如`{element for element in range(3)}`

+   使用集合字面量，例如`{1, 2, 3}`

请注意，对于集合，使用文字和理解需要额外小心，因为它们在形式上与字典文字和理解非常相似。此外，空集对象没有文字 - 空花括号`{}`保留用于空字典文字。

#### 实现细节

在 CPython 中，集合与字典非常相似。事实上，它们是使用虚拟值实现的字典，其中只有键是实际的集合元素。此外，集合利用映射中缺少值的优化。

由于这一点，集合允许非常快速的添加、删除和检查元素是否存在，平均时间复杂度为 O(1)。然而，由于 CPython 中集合的实现依赖于类似的哈希表结构，这些操作的最坏情况复杂度为 O(n)，其中*n*是集合的当前大小。

其他实现细节也适用。要包含在集合中的项目必须是可散列的，如果用户定义类的实例在集合中的哈希值很差，这将对性能产生负面影响。

### 基本集合之外 - collections 模块

每种数据结构都有其缺点。没有单一的集合可以适用于每个问题，而且四种基本类型（元组、列表、集合和字典）仍然不是一种广泛的选择。这些是最基本和重要的集合，具有专用的文字语法。幸运的是，Python 在其标准库中提供了更多选项，通过`collections`内置模块。其中一个已经提到了（`deque`）。以下是此模块提供的最重要的集合：

+   `namedtuple()`：这是一个用于创建元组子类的工厂函数，其索引可以作为命名属性访问

+   `deque`：这是一个双端队列，类似于堆栈和队列的列表泛化，可以在两端快速添加和弹出

+   `ChainMap`：这是一个类似字典的类，用于创建多个映射的单个视图

+   `Counter`：这是一个用于计算可散列对象的字典子类

+   `OrderedDict`：这是一个保留条目添加顺序的字典子类

+   `defaultdict`：这是一个字典子类，可以使用提供的默认值提供缺失的值

### 注意

有关来自 collections 模块的选定集合的更多详细信息以及在何处值得使用它们的建议，请参见第十二章，“优化 - 一些强大的技术”。

# 高级语法

客观地说，很难判断语言语法的哪个元素是先进的。对于本章关于高级语法元素的目的，我们将考虑那些与任何特定的内置数据类型没有直接关系，并且在开始时相对难以理解的元素。可能难以理解的最常见的 Python 特性是：

+   迭代器

+   生成器

+   装饰器

+   上下文管理器

## 迭代器

**迭代器**只不过是实现迭代器协议的容器对象。它基于两种方法：

+   `__next__`：这返回容器的下一个项目

+   `__iter__`：这返回迭代器本身

可以使用`iter`内置函数从序列创建迭代器。考虑以下示例：

```py
>>> i = iter('abc')
>>> next(i)
'a'
>>> next(i)
'b'
>>> next(i)
'c'
>>> next(i)
Traceback (most recent call last):
 **File "<input>", line 1, in <module>
StopIteration

```

当序列耗尽时，会引发`StopIteration`异常。它使迭代器与循环兼容，因为它们捕获此异常以停止循环。要创建自定义迭代器，可以编写一个具有`__next__`方法的类，只要它提供返回迭代器实例的特殊方法`__iter__`：

```py
class CountDown:def __init__(self, step):
        self.step = step
    def __next__(self):
        """Return the next element."""
        if self.step <= 0:
            raise StopIteration
        self.step -= 1
        return self.step
    def __iter__(self):
        """Return the iterator itself."""
        return self
```

以下是这种迭代器的示例用法：

```py
>>> for element in CountDown(4):
...     print(element)
...** 
3
2
1
0

```

迭代器本身是一个低级特性和概念，程序可以没有它们。但是它们为一个更有趣的特性 - 生成器提供了基础。

## yield 语句

生成器提供了一种优雅的方式来编写返回元素序列的简单高效的代码。基于`yield`语句，它们允许您暂停函数并返回中间结果。函数保存其执行上下文，如果必要的话可以稍后恢复。

例如，斐波那契数列可以用迭代器编写（这是关于迭代器的 PEP 中提供的示例）：

```py
def fibonacci():
    a, b = 0, 1
    while True:
        yield b
        a, b = b, a + b
```

您可以像使用`next()`函数或`for`循环一样从生成器中检索新值：

```py
>>> fib = fibonacci()
>>> next(fib)
1
>>> next(fib)
1
>>> next(fib)
2
>>> [next(fib) for i in range(10)]
[3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

```

这个函数返回一个`generator`对象，一个特殊的迭代器，它知道如何保存执行上下文。它可以被无限调用，每次产生套件的下一个元素。语法简洁，算法的无限性不再影响代码的可读性。它不必提供一种使函数可停止的方法。事实上，它看起来类似于伪代码中设计系列的方式。

在社区中，生成器并不经常使用，因为开发人员不习惯以这种方式思考。开发人员多年来一直习惯于使用直接函数。每当处理返回序列的函数或在循环中工作时，都应该考虑使用生成器。逐个返回元素可以提高整体性能，当它们被传递给另一个函数进行进一步处理时。

在这种情况下，用于计算一个元素的资源大部分时间不那么重要，而用于整个过程的资源更为重要。因此，它们可以保持较低，使程序更加高效。例如，斐波那契数列是无限的，但生成它的生成器不需要无限的内存来一次提供值。一个常见的用例是使用生成器流式传输数据缓冲区。它们可以被第三方代码暂停、恢复和停止，而不需要在开始处理之前加载所有数据。

例如，标准库中的`tokenize`模块可以从文本流中生成标记，并为每个处理的行返回一个`iterator`，可以传递给某些处理：

```py
>>> import tokenize
>>> reader = open('hello.py').readline
>>> tokens = tokenize.generate_tokens(reader)
>>> next(tokens)
TokenInfo(type=57 (COMMENT), string='# -*- coding: utf-8 -*-', start=(1, 0), end=(1, 23), line='# -*- coding: utf-8 -*-\n')
>>> next(tokens)
TokenInfo(type=58 (NL), string='\n', start=(1, 23), end=(1, 24), line='# -*- coding: utf-8 -*-\n')
>>> next(tokens)
TokenInfo(type=1 (NAME), string='def', start=(2, 0), end=(2, 3), line='def hello_world():\n')

```

在这里，我们可以看到`open`迭代文件的行，`generate_tokens`在管道中迭代它们，执行额外的工作。生成器还可以帮助打破复杂性，并提高基于几个套件的一些数据转换算法的效率。将每个套件视为`iterator`，然后将它们组合成一个高级函数是避免一个庞大、丑陋和难以阅读的函数的好方法。此外，这可以为整个处理链提供实时反馈。

在下面的例子中，每个函数定义了对序列的转换。然后它们被链接并应用。每个函数调用处理一个元素并返回其结果：

```py
def power(values):
    for value in values:
        print('powering %s' % value)
        yield value

def adder(values):
    for value in values:
        print('adding to %s' % value)
        if value % 2 == 0:
            yield value + 3
        else:
            yield value + 2
```

以下是使用这些生成器的可能结果：

```py
>>> elements = [1, 4, 7, 9, 12, 19]
>>> results = adder(power(elements))
>>> next(results)
powering 1
adding to 1
3
>>> next(results)
powering 4
adding to 4
7
>>> next(results)
powering 7
adding to 7
9

```

### 提示

**保持代码简单，而不是数据**

最好有很多简单的可迭代函数，可以处理值序列，而不是一次计算整个集合的复杂函数。

关于`generators`，Python 中另一个重要的功能是能够使用`next`函数与代码进行交互。`yield`变成了一个表达式，可以通过一个称为`send`的新方法传递一个值：

```py
def psychologist():
    print('Please tell me your problems')
    while True:
        answer = (yield)
        if answer is not None:
            if answer.endswith('?'):
                print("Don't ask yourself too much questions")
            elif 'good' in answer:
                print("Ahh that's good, go on")
            elif 'bad' in answer:
                print("Don't be so negative")
```

以下是使用我们的`psychologist()`函数的示例会话：

```py
>>> free = psychologist()
>>> next(free)
Please tell me your problems
>>> free.send('I feel bad')
Don't be so negative
>>> free.send("Why I shouldn't ?")
Don't ask yourself too much questions
>>> free.send("ok then i should find what is good for me")
Ahh that's good, go on

```

`send`的作用类似于`next`，但使`yield`返回函数定义内传递的值。因此，函数可以根据客户端代码改变其行为。为了完成这种行为，还添加了另外两个函数——`throw`和`close`。它们将错误引发到生成器中：

+   `throw`：这允许客户端代码发送任何类型的异常来引发。

+   `close`：这样做的方式相同，但会引发特定的异常`GeneratorExit`。在这种情况下，生成器函数必须再次引发`GeneratorExit`或`StopIteration`。

### 注意

生成器是 Python 中其他概念的基础——协程和异步并发，这些概念在第十三章中有所涵盖，*并发*。

## 装饰器

Python 中添加装饰器是为了使函数和方法包装（接收一个函数并返回一个增强的函数）更易于阅读和理解。最初的用例是能够在其定义的头部将方法定义为类方法或静态方法。没有装饰器语法，这将需要一个相当稀疏和重复的定义：

```py
class WithoutDecorators:
    def some_static_method():
        print("this is static method")
    some_static_method = staticmethod(some_static_method)

    def some_class_method(cls):
        print("this is class method")
    some_class_method = classmethod(some_class_method)
```

如果装饰器语法用于相同的目的，代码会更短，更容易理解：

```py
class WithDecorators:
    @staticmethod
    def some_static_method():
        print("this is static method")

    @classmethod
    def some_class_method(cls):
        print("this is class method")
```

### 一般语法和可能的实现

装饰器通常是一个命名对象（不允许`lambda`表达式），在调用时接受一个参数（它将是装饰的函数），并返回另一个可调用对象。这里使用“可调用”而不是“函数”是有预谋的。虽然装饰器经常在方法和函数的范围内讨论，但它们并不局限于它们。事实上，任何可调用的东西（任何实现`__call__`方法的对象都被认为是可调用的）都可以用作装饰器，而且它们返回的对象通常不是简单的函数，而是更复杂的类的实例，实现了自己的`__call__`方法。

装饰器语法只是一种语法糖。考虑以下装饰器的用法：

```py
@some_decorator
def decorated_function():
    pass
```

这总是可以被显式的装饰器调用和函数重新分配替代：

```py
def decorated_function():
    pass
decorated_function = some_decorator(decorated_function)
```

然而，后者不太可读，而且如果在单个函数上使用多个装饰器，很难理解。

### 提示

**装饰器甚至不需要返回一个可调用对象！**

事实上，任何函数都可以用作装饰器，因为 Python 不强制装饰器的返回类型。因此，使用一些函数作为装饰器，它接受一个参数但不返回可调用的，比如`str`，在语法上是完全有效的。如果用户尝试以这种方式调用装饰过的对象，最终会失败。无论如何，装饰器语法的这一部分为一些有趣的实验创造了一个领域。

#### 作为一个函数

有许多编写自定义装饰器的方法，但最简单的方法是编写一个返回包装原始函数调用的子函数的函数。

通用模式如下：

```py
def mydecorator(function):
    def wrapped(*args, **kwargs):     
        # do some stuff before the original
        # function gets called
        result = function(*args, **kwargs)
        # do some stuff after function call and
        # return the result
        return result
    # return wrapper as a decorated function
    return wrapped
```

#### 作为类

虽然装饰器几乎总是可以使用函数来实现，但在某些情况下，使用用户定义的类是更好的选择。当装饰器需要复杂的参数化或依赖于特定状态时，这通常是正确的。

作为类的非参数化装饰器的通用模式如下：

```py
class DecoratorAsClass:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        # do some stuff before the original
        # function gets called
        result = self.function(*args, **kwargs)
        # do some stuff after function call and
        # return the result
        return result
```

#### 参数化装饰器

在实际代码中，通常需要使用可以带参数的装饰器。当函数用作装饰器时，解决方案很简单——必须使用第二层包装。这是装饰器的一个简单示例，它重复执行装饰函数指定的次数，每次调用时：

```py
def repeat(number=3):
    """Cause decorated function to be repeated a number of times.

    Last value of original function call is returned as a result
    :param number: number of repetitions, 3 if not specified
    """
    def actual_decorator(function):
        def wrapper(*args, **kwargs):
            result = None
            for _ in range(number):
                result = function(*args, **kwargs)
            return result
        return wrapper
    return actual_decorator
```

这种方式定义的装饰器可以接受参数：

```py
>>> @repeat(2)
... def foo():
...     print("foo")
...** 
>>> foo()
foo
foo

```

请注意，即使带有默认值的参数化装饰器，其名称后面的括号也是必需的。使用具有默认参数的前述装饰器的正确方法如下：

```py
>>> @repeat()
... def bar():
...     print("bar")
...** 
>>> bar()
bar
bar
bar

```

如果省略这些括号，当调用装饰函数时将导致以下错误：

```py
>>> @repeat
... def bar():
...     pass
...** 
>>> bar()
Traceback (most recent call last):
 **File "<input>", line 1, in <module>
TypeError: actual_decorator() missing 1 required positional
argument: 'function'

```

#### 保留内省的装饰器

使用装饰器的常见陷阱是在使用装饰器时不保留函数元数据（主要是文档字符串和原始名称）。所有先前的示例都有这个问题。它们通过组合创建了一个新函数，并返回了一个新对象，而没有尊重原始函数的身份。这使得以这种方式装饰的函数的调试更加困难，并且也会破坏大多数可能使用的自动文档工具，因为原始文档字符串和函数签名不再可访问。

但让我们详细看一下。假设我们有一些虚拟装饰器，除了装饰和一些其他函数被装饰以外，什么都不做：

```py
def dummy_decorator(function):
    def wrapped(*args, **kwargs):
        """Internal wrapped function documentation."""
        return function(*args, **kwargs)
    return wrapped

@dummy_decorator
def function_with_important_docstring():
    """This is important docstring we do not want to lose."""
```

如果我们在 Python 交互会话中检查`function_with_important_docstring()`，我们会注意到它已经失去了原始名称和文档字符串：

```py
>>> function_with_important_docstring.__name__
'wrapped'
>>> function_with_important_docstring.__doc__
'Internal wrapped function documentation.'

```

解决这个问题的一个合适的方法是使用`functools`模块提供的内置`wraps()`装饰器：

```py
from functools import wraps

def preserving_decorator(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        """Internal wrapped function documentation."""
        return function(*args, **kwargs)
    return wrapped

@preserving_decorator
def function_with_important_docstring():
    """This is important docstring we do not want to lose."""
```

通过这种方式定义的装饰器，重要的函数元数据得到了保留：

```py
>>> function_with_important_docstring.__name__
'function_with_important_docstring.'
>>> function_with_important_docstring.__doc__
'This is important docstring we do not want to lose.'

```

### 用法和有用的示例

由于装饰器在模块首次读取时由解释器加载，它们的使用应该限于可以通用应用的包装器。如果装饰器与方法的类或增强的函数签名相关联，应将其重构为常规可调用对象以避免复杂性。无论如何，当装饰器处理 API 时，一个良好的做法是将它们分组在一个易于维护的模块中。

装饰器的常见模式有：

+   参数检查

+   缓存

+   代理

+   上下文提供者

#### 参数检查

检查函数接收或返回的参数在特定上下文中执行时可能是有用的。例如，如果一个函数要通过 XML-RPC 调用，Python 将无法像静态类型语言那样直接提供其完整签名。当 XML-RPC 客户端请求函数签名时，需要此功能来提供内省能力。

### 提示

**XML-RPC 协议**

XML-RPC 协议是一种轻量级的**远程过程调用**协议，它使用 XML 通过 HTTP 来编码调用。它经常用于简单的客户端-服务器交换而不是 SOAP。与提供列出所有可调用函数的页面的 SOAP 不同，XML-RPC 没有可用函数的目录。提出了一种允许发现服务器 API 的协议扩展，并且 Python 的`xmlrpc`模块实现了它（参考[`docs.python.org/3/library/xmlrpc.server.html`](https://docs.python.org/3/library/xmlrpc.server.html)）。

自定义装饰器可以提供这种类型的签名。它还可以确保输入和输出符合定义的签名参数：

```py
rpc_info = {}

def xmlrpc(in_=(), out=(type(None),)):
    def _xmlrpc(function):
        # registering the signature
        func_name = function.__name__
        rpc_info[func_name] = (in_, out)
        def _check_types(elements, types):
            """Subfunction that checks the types."""
            if len(elements) != len(types):
                raise TypeError('argument count is wrong')
            typed = enumerate(zip(elements, types))
            for index, couple in typed:
                arg, of_the_right_type = couple
                if isinstance(arg, of_the_right_type):
                    continue
                raise TypeError(
                    'arg #%d should be %s' % (index, of_the_right_type))

        # wrapped function
        def __xmlrpc(*args):  # no keywords allowed
            # checking what goes in
            checkable_args = args[1:]  # removing self
            _check_types(checkable_args, in_)
            # running the function
            res = function(*args)
            # checking what goes out
            if not type(res) in (tuple, list):
                checkable_res = (res,)
            else:
                checkable_res = res
            _check_types(checkable_res, out)

            # the function and the type
            # checking succeeded
            return res
        return __xmlrpc
    return _xmlrpc
```

装饰器将函数注册到全局字典中，并保留其参数和返回值的类型列表。请注意，示例被大大简化以演示参数检查装饰器。

使用示例如下：

```py
class RPCView:
    @xmlrpc((int, int))  # two int -> None
    def meth1(self, int1, int2):
        print('received %d and %d' % (int1, int2))

    @xmlrpc((str,), (int,))  # string -> int
    def meth2(self, phrase):
        print('received %s' % phrase)
        return 12
```

当它被读取时，这个类定义会填充`rpc_infos`字典，并且可以在特定环境中使用，其中检查参数类型：

```py
>>> rpc_info
{'meth2': ((<class 'str'>,), (<class 'int'>,)), 'meth1': ((<class 'int'>, <class 'int'>), (<class 'NoneType'>,))}
>>> my = RPCView()
>>> my.meth1(1, 2)
received 1 and 2
>>> my.meth2(2)
Traceback (most recent call last):
 **File "<input>", line 1, in <module>
 **File "<input>", line 26, in __xmlrpc
 **File "<input>", line 20, in _check_types
TypeError: arg #0 should be <class 'str'>

```

#### 缓存

缓存装饰器与参数检查非常相似，但侧重于那些内部状态不影响输出的函数。每组参数都可以与唯一的结果相关联。这种编程风格是**函数式编程**的特征（参考[`en.wikipedia.org/wiki/Functional_programming`](http://en.wikipedia.org/wiki/Functional_programming)），并且可以在输入值集合是有限的情况下使用。

因此，缓存装饰器可以将输出与计算所需的参数一起保留，并在后续调用时直接返回。这种行为称为**记忆化**（参考[`en.wikipedia.org/wiki/Memoizing`](http://en.wikipedia.org/wiki/Memoizing)），作为装饰器实现起来非常简单：

```py
import time
import hashlib
import pickle

cache = {}

def is_obsolete(entry, duration):
    return time.time() - entry['time']> duration

def compute_key(function, args, kw):
    key = pickle.dumps((function.__name__, args, kw))
    return hashlib.sha1(key).hexdigest()

def memoize(duration=10):
    def _memoize(function):
        def __memoize(*args, **kw):
            key = compute_key(function, args, kw)

            # do we have it already ?
            if (key in cache and
                not is_obsolete(cache[key], duration)):
                print('we got a winner')
                return cache[key]['value']

            # computing
            result = function(*args, **kw)
            # storing the result
            cache[key] = {
                'value': result,
                'time': time.time()
            }
            return result
        return __memoize
    return _memoize
```

使用有序参数值构建`SHA`哈希键，并将结果存储在全局字典中。哈希是使用 pickle 制作的，这是一个冻结传递的所有对象状态的快捷方式，确保所有参数都是良好的候选者。例如，如果线程或套接字被用作参数，将会发生`PicklingError`。（参见[`docs.python.org/3/library/pickle.html`](https://docs.python.org/3/library/pickle.html)。）`duration`参数用于在上次函数调用后经过太长时间后使缓存值无效。

以下是一个使用示例：

```py
>>> @memoize()
... def very_very_very_complex_stuff(a, b):
...     # if your computer gets too hot on this calculation
...     # consider stopping it
...     return a + b
...
>>> very_very_very_complex_stuff(2, 2)
4
>>> very_very_very_complex_stuff(2, 2)
we got a winner
4
>>> @memoize(1) # invalidates the cache after 1 second
... def very_very_very_complex_stuff(a, b):
...     return a + b
...
>>> very_very_very_complex_stuff(2, 2)
4
>>> very_very_very_complex_stuff(2, 2)
we got a winner
4
>>> cache
{'c2727f43c6e39b3694649ee0883234cf': {'value': 4, 'time':
1199734132.7102251)}
>>> time.sleep(2)
>>> very_very_very_complex_stuff(2, 2)
4

```

缓存昂贵的函数可以显著提高程序的整体性能，但必须小心使用。缓存的值也可以与函数本身绑定，以管理其范围和生命周期，而不是集中的字典。但无论如何，一个更有效的装饰器会使用基于高级缓存算法的专用缓存库。

### 注意

第十二章，*优化-一些强大的技术*，提供了关于缓存的详细信息和技术。

#### 代理

代理装饰器用于标记和注册具有全局机制的函数。例如，一个保护代码访问的安全层，取决于当前用户，可以使用一个带有可调用的关联权限的集中检查器来实现。

```py
class User(object):
    def __init__(self, roles):
        self.roles = roles

class Unauthorized(Exception):
    pass

def protect(role):
    def _protect(function):
        def __protect(*args, **kw):
            user = globals().get('user')
            if user is None or role not in user.roles:
                raise Unauthorized("I won't tell you")
            return function(*args, **kw)
        return __protect
    return _protect
```

这个模型经常被用在 Python 的 web 框架中来定义可发布类的安全性。例如，Django 提供了装饰器来保护函数的访问。

这是一个例子，其中当前用户保存在全局变量中。装饰器在访问方法时检查他或她的角色：

```py
>>> tarek = User(('admin', 'user'))
>>> bill = User(('user',))
>>> class MySecrets(object):
...     @protect('admin')
...     def waffle_recipe(self):
...         print('use tons of butter!')
...
>>> these_are = MySecrets()
>>> user = tarek
>>> these_are.waffle_recipe()
use tons of butter!
>>> user = bill
>>> these_are.waffle_recipe()
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
File "<stdin>", line 7, in wrap
__main__.Unauthorized: I won't tell you

```

#### 上下文提供程序

上下文装饰器确保函数可以在正确的上下文中运行，或在函数之前和之后运行一些代码。换句话说，它设置并取消特定的执行环境。例如，当一个数据项必须在多个线程之间共享时，必须使用锁来确保它受到多重访问的保护。这个锁可以编码在装饰器中，如下所示：

```py
from threading import RLock
lock = RLock()

def synchronized(function):
    def _synchronized(*args, **kw):
        lock.acquire()
        try:
            return function(*args, **kw)
        finally:
            lock.release()
    return _synchronized

@synchronized
def thread_safe():  # make sure it locks the resource
    pass
```

上下文装饰器更多地被上下文管理器（`with`语句）的使用所取代，这也在本章后面描述。

## 上下文管理器-`with`语句

`try...finally`语句对于确保一些清理代码即使发生错误也会运行是有用的。有许多这样的用例，比如：

+   关闭文件

+   释放锁

+   制作临时代码补丁

+   在特殊环境中运行受保护的代码

`with`语句通过提供一种简单的方式来包装一段代码来因素化这些用例。这允许您在块执行之前和之后调用一些代码，即使这个块引发异常。例如，通常是这样处理文件的：

```py
>>> hosts = open('/etc/hosts')
>>> try:
...     for line in hosts:
...         if line.startswith('#'):
...             continue
...         print(line.strip())
... finally:
...     hosts.close()
...
127.0.0.1       localhost
255.255.255.255 broadcasthost
::1             localhost

```

### 注意

这个例子是特定于 Linux 的，因为它读取位于`etc`中的主机文件，但任何文本文件都可以以同样的方式在这里使用。

通过使用`with`语句，可以重写成这样：

```py
>>> with open('/etc/hosts') as hosts:
...     for line in hosts:
...         if line.startswith('#'):
...             continue
...         print(line.strip )
...
127.0.0.1       localhost
255.255.255.255 broadcasthost
::1             localhost

```

在上面的例子中，`open`作为上下文管理器确保在执行`for`循环后文件将被关闭，即使发生异常。

与此语句兼容的其他项目是`threading`模块中的类：

+   `threading.Lock`

+   `threading.RLock`

+   `threading.Condition`

+   `threading.Semaphore`

+   `threading.BoundedSemaphore`

### 一般语法和可能的实现

最简单形式的`with`语句的一般语法是：

```py
with context_manager:
    # block of code
    ...
```

此外，如果上下文管理器提供一个上下文变量，可以使用`as`子句在本地存储它：

```py
with context_manager as context:
    # block of code
    ...
```

请注意，可以同时使用多个上下文管理器，如下所示：

```py
with A() as a, B() as b:
    ...
```

这相当于将它们嵌套，如下所示：

```py
with A() as a:
    with B() as b:
        ...
```

#### 作为一个类

任何实现**上下文管理器协议**的对象都可以用作上下文管理器。这个协议包括两个特殊方法：

+   `__enter__(self)`：更多信息可以在[`docs.python.org/3.3/reference/datamodel.html#object.__enter__`](https://docs.python.org/3.3/reference/datamodel.html#object.__enter__)找到

+   `__exit__(self, exc_type, exc_value, traceback)`：更多信息可以在[`docs.python.org/3.3/reference/datamodel.html#object.__exit__`](https://docs.python.org/3.3/reference/datamodel.html#object.__exit__)找到

简而言之，`with`语句的执行如下：

1.  `__enter__`方法被调用。任何返回值都绑定到指定为子句的目标。

1.  执行内部代码块。

1.  `__exit__`方法被调用。

`__exit__`接收三个参数，当代码块内发生错误时会填充这些参数。如果没有发生错误，所有三个参数都设置为`None`。当发生错误时，`__exit__`不应重新引发它，因为这是调用者的责任。它可以通过返回`True`来防止异常被引发。这是为了实现一些特定的用例，比如我们将在下一节中看到的`contextmanager`装饰器。但对于大多数用例，这个方法的正确行为是做一些清理，就像`finally`子句所做的那样；无论在块中发生了什么，它都不返回任何东西。

以下是一个实现了这个协议的一些上下文管理器的示例，以更好地说明它是如何工作的：

```py
class ContextIllustration:
    def __enter__(self):
        print('entering context')

    def __exit__(self, exc_type, exc_value, traceback):
        print('leaving context')

        if exc_type is None:
            print('with no error')
        else:
            print('with an error (%s)' % exc_value)
```

当没有引发异常时，输出如下：

```py
>>> with ContextIllustration():
...     print("inside")
...** 
entering context
inside
leaving context
with no error

```

当引发异常时，输出如下：

```py
>>> with ContextIllustration():
...     raise RuntimeError("raised within 'with'")
...** 
entering context
leaving context
with an error (raised within 'with')
Traceback (most recent call last):
 **File "<input>", line 2, in <module>
RuntimeError: raised within 'with'

```

#### 作为函数-上下文管理器模块

使用类似乎是实现 Python 语言中提供的任何协议的最灵活的方式，但对于许多用例来说可能是太多的样板文件。标准库中添加了一个`contextlib`模块，提供了一些与上下文管理器一起使用的帮助器。它最有用的部分是`contextmanager`装饰器。它允许您在单个函数中提供`__enter__`和`__exit__`部分，中间用`yield`语句分隔（请注意，这会使函数成为生成器）。使用这个装饰器编写的前面的示例将如下所示：

```py
from contextlib import contextmanager

@contextmanager
def context_illustration():
    print('entering context')

    try:
        yield
    except Exception as e:
        print('leaving context')
        print('with an error (%s)' % e)
        # exception needs to be reraised
        raise
    else:
        print('leaving context')
        print('with no error')
```

如果发生任何异常，函数需要重新引发它以便传递它。请注意，`context_illustration`如果需要的话可以有一些参数，只要它们在调用中提供。这个小助手与基于类的迭代器 API 一样简化了正常的基于类的上下文 API。

这个模块提供的另外三个帮助器是：

+   `closing(element)`：这会返回一个上下文管理器，在退出时调用元素的 close 方法。这对于处理流的类非常有用。

+   `supress(*exceptions)`：如果在 with 语句的主体中发生指定的任何异常，则抑制它们。

+   `redirect_stdout(new_target)`和`redirect_stderr(new_target)`：这将代码块内的`sys.stdout`或`sys.stderr`输出重定向到另一个文件或类文件对象。

# 其他你可能还不知道的语法元素

Python 语法中有一些不太流行且很少使用的元素。这是因为它们要么提供的收益很少，要么它们的使用方法很难记住。因此，许多 Python 程序员（即使有多年的经验）根本不知道它们的存在。这些特性的最显著的例子如下：

+   `for … else`子句

+   函数注释

## `for … else …`语句

在`for`循环之后使用`else`子句允许您仅在循环以“自然”方式结束而不是用`break`语句终止时执行代码块：

```py
>>> for number in range(1):
...     break
... else:
...     print("no break")
...
>>>
>>> for number in range(1):
...     pass
... else:
...     print("break")
...
break

```

在某些情况下，这很方便，因为它有助于消除可能需要的一些“标记”变量，如果用户想要存储信息，以确定是否发生了`break`。这使得代码更清晰，但可能会让不熟悉这种语法的程序员感到困惑。有人说`else`子句的这种含义是违反直觉的，但这里有一个简单的提示，可以帮助您记住它的工作原理-记住`for`循环后的`else`子句只是表示“没有 break”。

## 函数注释

函数注释是 Python 3 最独特的功能之一。官方文档指出*注释是关于用户定义函数使用的类型的完全可选的元数据信息*，但实际上，它们并不局限于类型提示，Python 及其标准库也没有利用这样的注释。这就是为什么这个功能是独特的-它没有任何语法意义。注释可以简单地为函数定义，并且可以在运行时检索，但仅此而已。如何处理它们留给开发人员。

### 一般语法

Python 文档中略微修改的示例最好地展示了如何定义和检索函数注释：

```py
>>> def f(ham: str, eggs: str = 'eggs') -> str:
...     pass
...** 
>>> print(f.__annotations__)
{'return': <class 'str'>, 'eggs': <class 'str'>, 'ham': <class 'str'>}

```

如所示，参数注释由表达式定义，该表达式评估为注释值，并在冒号之前。返回注释由冒号后的`def`语句结束和参数列表后面的`->`之间的表达式定义。

一旦定义，注释将作为函数对象的`__annotations__`属性以字典的形式可用，并且可以在应用运行时检索。

任何表达式都可以用作注释，并且它位于默认参数旁边，这允许创建一些令人困惑的函数定义，如下所示：

```py
>>> def square(number: 0<=3 and 1=0) -> (\
...     +9000): return number**2
>>> square(10)
100

```

然而，这种注释的用法除了混淆之外没有其他目的，即使没有它们，编写难以阅读和维护的代码也相对容易。

### 可能的用途

尽管注释具有巨大潜力，但它们并不被广泛使用。一篇解释 Python 3 新增功能的文章（参见[`docs.python.org/3/whatsnew/3.0.html`](https://docs.python.org/3/whatsnew/3.0.html)）表示，这一功能的目的是“通过元类、装饰器或框架鼓励实验”。另一方面，正式提出函数注释的**PEP 3107**列出了以下一系列可能的用例：

+   提供类型信息

+   类型检查

+   让 IDE 显示函数期望和返回的类型

+   函数重载/通用函数

+   外语桥梁

+   适应

+   谓词逻辑函数

+   数据库查询映射

+   RPC 参数编组

+   其他信息

+   参数和返回值的文档

尽管函数注释与 Python 3 一样古老，但仍然很难找到任何流行且积极维护的软件包，除了类型检查之外还使用它们。因此，函数注释仍然主要用于实验和玩耍-这是它们被包含在 Python 3 的初始版本中的初衷。

# 总结

本章涵盖了与 Python 类和面向对象编程无直接关系的各种最佳语法实践。本章的第一部分专门讨论了围绕 Python 序列和集合的语法特性，还讨论了字符串和字节相关序列。本章的其余部分涵盖了两组独立的语法元素-相对于初学者来说相对难以理解的元素（如迭代器、生成器和装饰器）和相对较少知名的元素（`for…else`子句和函数注释）。
