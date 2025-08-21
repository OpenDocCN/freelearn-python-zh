# 第六章：Python 数据结构

到目前为止，我们已经在示例中看到了许多内置的 Python 数据结构。你可能也在入门书籍或教程中涵盖了许多这些内容。在本章中，我们将讨论这些数据结构的面向对象特性，以及它们应该在何时使用而不是使用常规类，以及何时不应该使用。特别是，我们将讨论：

+   元组和命名元组

+   字典

+   列表和集合

+   如何以及为什么扩展内置对象

+   三种类型的队列

# 空对象

让我们从最基本的 Python 内置对象开始，这是我们已经看到很多次的对象，我们在创建的每个类中都扩展了它：`object`。从技术上讲，我们可以实例化一个`object`而不编写子类。

```py
>>> o = object()
>>> o.x = 5
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
AttributeError: 'object' object has no attribute 'x'

```

不幸的是，正如你所看到的，不可能在直接实例化的`object`上设置任何属性。这不是因为 Python 开发人员想要强迫我们编写自己的类，或者有什么邪恶的目的。他们这样做是为了节省内存；大量的内存。当 Python 允许对象具有任意属性时，它需要一定量的系统内存来跟踪每个对象具有的属性，用于存储属性名称和其值。即使没有存储属性，也会为*潜在*的新属性分配内存。在典型的 Python 程序中有数十、数百或数千个对象（每个类都扩展了 object）；这小量的内存很快就会变成大量的内存。因此，Python 默认禁用`object`和其他几个内置对象上的任意属性。

### 注意

我们可以使用**slots**在我们自己的类上限制任意属性。Slots 超出了本书的范围，但现在你有了一个搜索词，如果你想要更多信息。在正常使用中，使用 slots 并没有太多好处，但如果你正在编写一个将在整个系统中复制成千上万次的对象，它们可以帮助节省内存，就像对`object`一样。

然而，创建一个空对象类非常简单；我们在最早的示例中看到了它：

```py
class MyObject:
    pass
```

而且，正如我们已经看到的，可以在这样的类上设置属性：

```py
>>> m = MyObject()
>>> m.x = "hello"
>>> m.x
'hello'

```

如果我们想要将属性分组在一起，我们可以将它们存储在一个空对象中。但是，通常最好使用其他专门用于存储数据的内置对象。本书始终强调，只有在想要指定*数据和行为*时才应该使用类和对象。创建一个空类的主要原因是为了快速地阻止某些东西，知道我们稍后会回来添加行为。将行为适应类要容易得多，而将数据结构替换为对象并更改所有引用则要困难得多。因此，重要的是从一开始就决定数据只是数据，还是伪装成对象。一旦做出了这个设计决定，其余的设计自然而然地就会落实。

# 元组和命名元组

元组是可以按顺序存储特定数量的其他对象的对象。它们是不可变的，因此我们无法在运行时添加、删除或替换对象。这可能看起来像是一个巨大的限制，但事实是，如果你需要修改一个元组，你正在使用错误的数据类型（通常列表更合适）。元组不可变的主要好处是我们可以将它们用作字典中的键，以及其他需要哈希值的对象的位置。

元组用于存储数据；无法在元组中存储行为。如果我们需要行为来操作元组，我们必须将元组传递给执行该操作的函数（或另一个对象的方法）。

元组通常应该存储一些在某种程度上不同的值。例如，我们不会在一个元组中放入三个股票符号，但我们可能会创建一个包含股票符号、当前价格、最高价和最低价的元组。元组的主要目的是将不同的数据片段聚合到一个容器中。因此，元组可能是最简单的工具，用来替换“没有数据的对象”习语。

我们可以通过用逗号分隔值来创建一个元组。通常，元组用括号括起来，以使它们易于阅读并与表达式的其他部分分开，但这并不总是强制性的。以下两个赋值是相同的（它们记录了一家相当有利可图的公司的股票、当前价格、最高价和最低价）：

```py
>>> stock = "FB", 75.00, 75.03, 74.90
>>> stock2 = ("FB", 75.00, 75.03, 74.90)

```

如果我们将元组分组到其他对象中，比如函数调用、列表推导或生成器中，括号是必需的。否则，解释器将无法知道它是一个元组还是下一个函数参数。例如，以下函数接受一个元组和一个日期，并返回一个包含日期和股票最高价和最低价之间的中间值的元组：

```py
import datetime
def middle(stock, date):
    **symbol, current, high, low = stock
    return (((high + low) / 2), date)

mid_value, date = middle(("FB", 75.00, 75.03, 74.90),
        **datetime.date(2014, 10, 31))

```

元组是直接在函数调用中通过用逗号分隔值并将整个元组括在括号中创建的。然后，这个元组后面跟着一个逗号，以将它与第二个参数分开。

这个例子也说明了元组的解包。函数内的第一行将`stock`参数解包成四个不同的变量。元组的长度必须与变量的数量完全相同，否则会引发异常。我们还可以在最后一行看到元组解包的例子，其中函数内返回的元组被解包成两个值，`mid_value`和`date`。当然，这是一个奇怪的做法，因为我们首先向函数提供了日期，但这让我们有机会看到解包的工作原理。

在 Python 中，解包是一个非常有用的功能。我们可以将变量组合在一起，使得存储和传递它们变得更简单，但是当我们需要访问它们所有时，我们可以将它们解包成单独的变量。当然，有时我们只需要访问元组中的一个变量。我们可以使用与其他序列类型（例如列表和字符串）相同的语法来访问单个值：

```py
>>> stock = "FB", 75.00, 75.03, 74.90
>>> high = stock[2]
>>> high
75.03

```

我们甚至可以使用切片表示法来提取元组的较大部分：

```py
>>> stock[1:3]
(75.00, 75.03)

```

这些例子展示了元组的灵活性，但也展示了它们的一个主要缺点：可读性。阅读这段代码的人怎么知道特定元组的第二个位置是什么？他们可以猜测，从我们分配给它的变量名，它是某种“高”，但如果我们在计算中只是访问了元组的值而没有分配它，就没有这样的指示。他们必须在代码中搜索元组声明的位置，然后才能发现它的作用。

直接访问元组成员在某些情况下是可以的，但不要养成这样的习惯。这种所谓的“魔术数字”（似乎毫无意义地出现在代码中的数字）是许多编码错误的根源，并导致了数小时的沮丧调试。尽量只在你知道所有的值一次性都会有用，并且在访问时通常会被解包时使用元组。如果必须直接访问成员或使用切片，并且该值的目的不是立即明显的，至少要包含一个解释它来自哪里的注释。

## 命名元组

那么，当我们想要将值组合在一起，但知道我们经常需要单独访问它们时，我们该怎么办？嗯，我们可以使用空对象，如前一节中讨论的（但除非我们预期稍后添加行为，否则很少有用），或者我们可以使用字典（如果我们不知道将存储多少个或哪些特定数据，这是最有用的），我们将在下一节中介绍。

然而，如果我们不需要向对象添加行为，并且事先知道需要存储哪些属性，我们可以使用命名元组。命名元组是带有态度的元组。它们是将只读数据组合在一起的绝佳方式。

构造命名元组比普通元组需要更多的工作。首先，我们必须导入`namedtuple`，因为它不是默认的命名空间中。然后，我们通过给它一个名称并概述其属性来描述命名元组。这将返回一个类似的对象，我们可以根据需要实例化多次：

```py
from collections import namedtuple
Stock = namedtuple("Stock", "symbol current high low")
stock = Stock("FB", 75.00, high=75.03, low=74.90)
```

`namedtuple`构造函数接受两个参数。第一个是命名元组的标识符。第二个是命名元组可以具有的以空格分隔的属性字符串。应该列出第一个属性，然后是一个空格（或者如果你喜欢，逗号），然后是第二个属性，然后是另一个空格，依此类推。结果是一个可以像普通类一样调用的对象，以实例化其他对象。构造函数必须具有可以作为参数或关键字参数传递的恰好正确数量的参数。与普通对象一样，我们可以创建任意数量的此“类”的实例，并为每个实例提供不同的值。

然后，生成的`namedtuple`可以像普通元组一样打包、解包和以其他方式处理，但我们也可以像访问对象一样访问它的单个属性：

```py
>>> stock.high
75.03
>>> symbol, current, high, low = stock
>>> current
75.00

```

### 提示

请记住，创建命名元组是一个两步过程。首先，使用`collections.namedtuple`创建一个类，然后构造该类的实例。

命名元组非常适合许多“仅数据”表示，但并非适用于所有情况。与元组和字符串一样，命名元组是不可变的，因此一旦设置了属性，就无法修改属性。例如，自从我们开始讨论以来，我的公司股票的当前价值已经下跌，但我们无法设置新值：

```py
>>> stock.current = 74.98
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
AttributeError: can't set attribute

```

如果我们需要能够更改存储的数据，可能需要使用字典。

# 字典

字典是非常有用的容器，允许我们直接将对象映射到其他对象。具有属性的空对象是一种字典；属性的名称映射到属性值。这实际上比听起来更接近事实；在内部，对象通常将属性表示为字典，其中值是对象上的属性或方法（如果你不相信我，请查看`__dict__`属性）。甚至模块上的属性也是在字典中存储的。

字典在查找特定键对象映射到该值时非常高效。当您想要根据其他对象找到一个对象时，应该始终使用它们。被存储的对象称为**值**；用作索引的对象称为**键**。我们已经在一些先前的示例中看到了字典语法。

字典可以使用`dict()`构造函数或使用`{}`语法快捷方式创建。实际上，几乎总是使用后一种格式。我们可以通过使用冒号分隔键和值，并使用逗号分隔键值对来预填充字典。

例如，在股票应用程序中，我们最常常希望按股票符号查找价格。我们可以创建一个使用股票符号作为键，当前价格、最高价格和最低价格的元组作为值的字典，如下所示：

```py
stocks = {"GOOG": (613.30, 625.86, 610.50),
          "MSFT": (30.25, 30.70, 30.19)}
```

正如我们在之前的例子中看到的，我们可以通过在方括号内请求一个键来查找字典中的值。如果键不在字典中，它会引发一个异常：

```py
>>> stocks["GOOG"]
(613.3, 625.86, 610.5)
>>> stocks["RIM"]
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
KeyError: 'RIM'

```

当然，我们可以捕获`KeyError`并处理它。但我们还有其他选择。记住，字典是对象，即使它们的主要目的是保存其他对象。因此，它们有几种与之相关的行为。其中最有用的方法之一是`get`方法；它接受一个键作为第一个参数，以及一个可选的默认值（如果键不存在）：

```py
>>> print(stocks.get("RIM"))
None
>>> stocks.get("RIM", "NOT FOUND")
'NOT FOUND'

```

为了更多的控制，我们可以使用`setdefault`方法。如果键在字典中，这个方法的行为就像`get`一样；它返回该键的值。否则，如果键不在字典中，它不仅会返回我们在方法调用中提供的默认值（就像`get`一样），它还会将键设置为相同的值。另一种思考方式是，`setdefault`只有在该值以前没有被设置时才在字典中设置一个值。然后它返回字典中的值，无论是已经存在的值，还是新提供的默认值。

```py
>>> stocks.setdefault("GOOG", "INVALID")
(613.3, 625.86, 610.5)
>>> stocks.setdefault("BBRY", (10.50, 10.62, 10.39))
(10.50, 10.62, 10.39)
>>> stocks["BBRY"]
(10.50, 10.62, 10.39)

```

`GOOG`股票已经在字典中，所以当我们尝试将其`setdefault`为一个无效值时，它只是返回了已经在字典中的值。`BBRY`不在字典中，所以`setdefault`返回了默认值，并为我们在字典中设置了新值。然后我们检查新的股票是否确实在字典中。

另外三个非常有用的字典方法是`keys()`，`values()`和`items()`。前两个返回字典中所有键和所有值的迭代器。如果我们想要处理所有键或值，我们可以像列表一样使用它们，或者在`for`循环中使用它们。`items()`方法可能是最有用的；它返回一个元组的迭代器，其中包含字典中每个项目的`(key, value)`对。这与在`for`循环中使用元组解包很好地配合，以循环遍历相关的键和值。这个例子就是这样做的，以打印出字典中每个股票及其当前值：

```py
>>> for stock, values in stocks.items():
...     print("{} last value is {}".format(stock, values[0]))
...
GOOG last value is 613.3
BBRY last value is 10.50
MSFT last value is 30.25

```

每个键/值元组都被解包成两个名为`stock`和`values`的变量（我们可以使用任何我们想要的变量名，但这两个似乎都合适），然后以格式化的字符串打印出来。

请注意，股票并没有按照插入的顺序显示出来。由于用于使键查找如此快速的高效算法（称为哈希），字典本身是无序的。

因此，一旦字典被实例化，就有许多种方法可以从中检索数据；我们可以使用方括号作为索引语法，`get`方法，`setdefault`方法，或者遍历`items`方法，等等。

最后，你可能已经知道，我们可以使用与检索值相同的索引语法来在字典中设置一个值：

```py
>>> stocks["GOOG"] = (597.63, 610.00, 596.28)
>>> stocks['GOOG']
(597.63, 610.0, 596.28)

```

谷歌的价格今天较低，所以我更新了字典中元组的值。我们可以使用这种索引语法为任何键设置一个值，而不管该键是否在字典中。如果它在字典中，旧值将被新值替换；否则，将创建一个新的键/值对。

到目前为止，我们一直在使用字符串作为字典的键，但我们并不局限于字符串键。通常在存储数据以便将其聚集在一起时，使用字符串作为键是很常见的（而不是使用具有命名属性的对象）。但我们也可以使用元组、数字，甚至是我们自己定义的对象作为字典的键。我们甚至可以在单个字典中使用不同类型的键：

```py
random_keys = {}
random_keys["astring"] = "somestring"
random_keys[5] = "aninteger"
random_keys[25.2] = "floats work too"
random_keys[("abc", 123)] = "so do tuples"

class AnObject:
    def __init__(self, avalue):
        self.avalue = avalue

my_object = AnObject(14)
random_keys[my_object] = "We can even store objects"
my_object.avalue = 12
try:
    random_keys[[1,2,3]] = "we can't store lists though"
except:
    print("unable to store list\n")

for key, value in random_keys.items():
    print("{} has value {}".format(key, value))
```

这段代码展示了我们可以提供给字典的几种不同类型的键。它还展示了一种不能使用的对象类型。我们已经广泛使用了列表，并且在下一节中将看到更多关于它们的细节。因为列表可以随时更改（例如通过添加或删除项目），它们无法哈希到一个特定的值。

具有**可哈希性**的对象基本上具有一个定义好的算法，将对象转换为唯一的整数值，以便快速查找。这个哈希值实际上是用来在字典中查找值的。例如，字符串根据字符串中的字符映射到整数，而元组则组合了元组内部项目的哈希值。任何两个被视为相等的对象（比如具有相同字符的字符串或具有相同值的元组）应该具有相同的哈希值，并且对象的哈希值永远不应该改变。然而，列表的内容可以改变，这会改变它们的哈希值（只有当列表的内容相同时，两个列表才应该相等）。因此，它们不能用作字典的键。出于同样的原因，字典也不能用作其他字典的键。

相比之下，对于可以用作字典值的对象类型没有限制。例如，我们可以使用字符串键映射到列表值，或者我们可以在另一个字典中将嵌套字典作为值。

## 字典的用例

字典非常灵活，有很多用途。字典可以有两种主要用法。第一种是所有键表示类似对象的不同实例的字典；例如，我们的股票字典。这是一个索引系统。我们使用股票符号作为值的索引。这些值甚至可以是复杂的自定义对象，而不是我们简单的元组。

第二种设计是每个键表示单个结构的某个方面的字典；在这种情况下，我们可能会为每个对象使用一个单独的字典，并且它们都具有相似（尽管通常不完全相同）的键集。这种情况通常也可以用命名元组解决。当我们确切地知道数据必须存储的属性，并且知道所有数据必须一次性提供（在构造项目时）时，应该使用这些。但是，如果我们需要随时间创建或更改字典键，或者我们不知道键可能是什么，那么字典更合适。

## 使用 defaultdict

我们已经看到如何使用`setdefault`来设置默认值，如果键不存在，但是如果我们需要每次查找值时都设置默认值，这可能会有点单调。例如，如果我们正在编写代码来计算给定句子中字母出现的次数，我们可以这样做：

```py
def letter_frequency(sentence):
    frequencies = {}
    for letter in sentence:
        **frequency = frequencies.setdefault(letter, 0)
        frequencies[letter] = frequency + 1
    return frequencies
```

每次访问字典时，我们需要检查它是否已经有一个值，如果没有，将其设置为零。当每次请求一个空键时需要做这样的事情时，我们可以使用字典的另一个版本，称为`defaultdict`：

```py
from collections import defaultdict
def letter_frequency(sentence):
    **frequencies = defaultdict(int)
    for letter in sentence:
        frequencies[letter] += 1
    return frequencies
```

这段代码看起来似乎不可能工作。`defaultdict`在其构造函数中接受一个函数。每当访问一个不在字典中的键时，它调用该函数，不带任何参数，以创建一个默认值。

在这种情况下，它调用的函数是`int`，这是整数对象的构造函数。通常，整数是通过在代码中键入整数来创建的，如果我们使用`int`构造函数创建一个整数，我们将传递要创建的项目（例如，将数字字符串转换为整数）。但是，如果我们在没有任何参数的情况下调用`int`，它会方便地返回数字零。在这段代码中，如果字母不存在于`defaultdict`中，当我们访问它时将返回数字零。然后我们将这个数字加一，以表示我们找到了该字母的一个实例，下次再找到一个实例时，将返回该数字，然后我们可以再次递增该值。

`defaultdict`对于创建容器字典非常有用。如果我们想要创建一个过去 30 天股票价格的字典，我们可以使用股票符号作为键，并将价格存储在`list`中；第一次访问股票价格时，我们希望它创建一个空列表。只需将`list`传递给`defaultdict`，它将在每次访问空键时被调用。如果我们想要将一个集合或者一个空字典与一个键关联起来，我们也可以做类似的事情。

当然，我们也可以编写自己的函数并将它们传递给`defaultdict`。假设我们想创建一个`defaultdict`，其中每个新元素都包含一个元组，该元组包含了在该时间插入字典中的项目数和一个空列表来保存其他东西。没有人知道为什么我们要创建这样一个对象，但让我们来看一下：

```py
from collections import defaultdict
num_items = 0
def tuple_counter():
    global num_items
    num_items += 1
    return (num_items, [])

d = defaultdict(tuple_counter)

```

当我们运行这段代码时，我们可以在一个语句中访问空键并插入列表：

```py
>>> d = defaultdict(tuple_counter)
>>> d['a'][1].append("hello")
>>> d['b'][1].append('world')
>>> d
defaultdict(<function tuple_counter at 0x82f2c6c>,
{'a': (1, ['hello']), 'b': (2, ['world'])})

```

当我们在最后打印`dict`时，我们看到计数器确实在工作。

### 注意

这个例子虽然简洁地演示了如何为`defaultdict`创建自己的函数，但实际上并不是很好的代码；使用全局变量意味着如果我们创建了四个不同的`defaultdict`段，每个段都使用了`tuple_counter`，它将计算所有字典中的条目数，而不是为每个字典单独计数。最好创建一个类，并将该类的方法传递给`defaultdict`。

### 计数器

您可能会认为`defaultdict(int)`比这更简单，但“我想要计算可迭代对象中特定实例的数量”这种用例是足够常见，以至于 Python 开发人员为此创建了一个特定的类。在一个单行中很容易计算以前的代码中字符串中的字符数量：

```py
from collections import Counter
def letter_frequency(sentence):
    return Counter(sentence)
```

`Counter`对象的行为类似于一个强化的字典，其中键是被计数的项目，值是这些项目的数量。其中最有用的函数之一是`most_common()`方法。它返回一个按计数排序的（键，计数）元组列表。您还可以选择将整数参数传递给`most_common()`，以请求仅返回最常见的元素。例如，您可以编写一个简单的投票应用程序如下：

```py
from collections import Counter

responses = [
    "vanilla",
    "chocolate",
    "vanilla",
    "vanilla",
    "caramel",
    "strawberry",
    "vanilla"
]

print(
    "The children voted for {} ice cream".format(
        Counter(responses).most_common(1)[0][0]
    )
)
```

据推测，您可以从数据库中获取响应，或者使用复杂的视觉算法来计算举手的孩子。在这里，我们将其硬编码，以便我们可以测试`most_common`方法。它返回一个只有一个元素的列表（因为我们在参数中请求了一个元素）。这个元素在位置零存储了最受欢迎的选择的名称，因此在调用结束时有两个`[0][0]`。我觉得它们看起来像是一个惊讶的脸，你觉得呢？你的计算机可能对它能够如此轻松地计数数据感到惊讶。它的祖先，霍勒里斯的 1890 年美国人口普查用的整理机，一定会非常嫉妒！

# 列表

列表是 Python 数据结构中最不面向对象的。虽然列表本身是对象，但在 Python 中有很多语法可以尽可能地减少它们的使用痛苦。与许多其他面向对象的语言不同，Python 中的列表是直接可用的。我们不需要导入它们，也很少需要调用它们的方法。我们可以在不明确请求迭代器对象的情况下循环遍历列表，并且可以使用自定义语法构造列表（与字典一样）。此外，列表推导和生成器表达式将它们转变为计算功能的多功能工具。

我们不会过多介绍语法；你在网络上的入门教程和本书中的先前示例中已经见过它。你不能长时间编写 Python 代码而不学会如何使用列表！相反，我们将介绍何时应该使用列表以及它们作为对象的性质。如果你不知道如何创建或附加到列表，如何从列表中检索项目，或者什么是“切片表示法”，我建议你立即查看官方 Python 教程。它可以在[`docs.python.org/3/tutorial/`](http://docs.python.org/3/tutorial/)上找到。

在 Python 中，当我们想要存储“相同”类型的对象的多个实例时，通常应该使用列表；字符串列表或数字列表；最常见的是我们自己定义的对象列表。当我们想要按某种顺序存储项目时，应该始终使用列表。通常，这是它们被插入的顺序，但它们也可以按某些标准排序。

正如我们在上一章的案例研究中看到的，当我们需要修改内容时，列表也非常有用：在列表的任意位置插入或删除，或者更新列表中的值。

与字典一样，Python 列表使用非常高效和良好调整的内部数据结构，因此我们可以关注我们存储的内容，而不是我们如何存储它。许多面向对象的语言为队列、栈、链表和基于数组的列表提供了不同的数据结构。如果需要优化对大量数据的访问，Python 确实提供了这些类的特殊实例。然而，通常情况下，列表数据结构可以同时满足所有这些目的，并且编码人员可以完全控制他们如何访问它。

不要使用列表来收集单个项目的不同属性。例如，我们不希望一个特定形状的属性列表。元组、命名元组、字典和对象都更适合这个目的。在某些语言中，它们可能创建一个列表，其中每个交替项是不同的类型；例如，他们可能为我们的字母频率列表写`['a', 1, 'b', 3]`。他们必须使用一个奇怪的循环，一次访问两个元素，或者使用模运算符来确定正在访问的位置。

在 Python 中不要这样做。我们可以使用字典将相关项目分组在一起，就像我们在上一节中所做的那样（如果排序顺序不重要），或者使用元组列表。下面是一个相当复杂的示例，演示了我们如何使用列表来进行频率示例。它比字典示例复杂得多，并且说明了选择正确（或错误）的数据结构对我们代码的可读性产生的影响。

```py
import string
CHARACTERS  = list(string.ascii_letters) + [" "]

def letter_frequency(sentence):
    **frequencies = [(c, 0) for c in CHARACTERS]
    for letter in sentence:
        index = CHARACTERS.index(letter)
        **frequencies[index] = (letter,frequencies[index][1]+1)
    return frequencies
```

这段代码以可能的字符列表开始。`string.ascii_letters`属性提供了一个按顺序排列的所有字母（大写和小写）的字符串。我们将其转换为列表，然后使用列表连接（加号运算符将两个列表合并为一个）添加一个额外的字符，即空格。这些是我们频率列表中可用的字符（如果我们尝试添加不在列表中的字母，代码将会出错，但可以使用异常处理程序来解决这个问题）。

函数内的第一行使用列表推导将`CHARACTERS`列表转换为元组列表。列表推导是 Python 中一个重要的非面向对象的工具；我们将在下一章详细介绍它们。

然后我们循环遍历句子中的每个字符。我们首先查找`CHARACTERS`列表中字符的索引，我们知道它在我们的频率列表中具有相同的索引，因为我们刚刚从第一个列表创建了第二个列表。然后我们通过创建一个新元组来更新频率列表中的索引，丢弃原始元组。除了垃圾收集和内存浪费的担忧外，这是相当难以阅读的！

像字典一样，列表也是对象，并且有几种可以在它们上调用的方法。以下是一些常见的方法：

+   `append(element)`方法将一个元素添加到列表的末尾

+   `insert(index, element)`方法在特定位置插入一个项目

+   `count(element)`方法告诉我们一个元素在列表中出现了多少次

+   `index()`方法告诉我们列表中项目的索引，如果找不到它会引发异常

+   `find()`方法也是做同样的事情，但是找不到项目时返回`-1`而不是引发异常

+   `reverse()`方法确实做了它所说的事情——将列表倒转过来

+   `sort()`方法具有一些相当复杂的面向对象的行为，我们现在来介绍一下

## 排序列表

没有任何参数时，`sort`通常会做预期的事情。如果是字符串列表，它会按字母顺序排列。这个操作是区分大小写的，所以所有大写字母会排在小写字母之前，即`Z`排在`a`之前。如果是数字列表，它们将按数字顺序排序。如果提供了一个包含不可排序项目的混合列表，排序将引发`TypeError`异常。

如果我们想把自己定义的对象放入列表并使这些对象可排序，我们需要做更多的工作。类上应该定义特殊方法`__lt__`，它代表“小于”，以使该类的实例可比较。列表上的`sort`方法将访问每个对象上的这个方法来确定它在列表中的位置。如果我们的类在某种程度上小于传递的参数，则该方法应返回`True`，否则返回`False`。下面是一个相当愚蠢的类，它可以根据字符串或数字进行排序：

```py
class WeirdSortee:
    def __init__(self, string, number, sort_num):
        self.string = string
        self.number = number
        self.sort_num = sort_num

    **def __lt__(self, object):
        **if self.sort_num:
            **return self.number < object.number
        **return self.string < object.string

    def __repr__(self):
        return"{}:{}".format(self.string, self.number)
```

`__repr__`方法使我们在打印列表时很容易看到这两个值。`__lt__`方法的实现将对象与相同类的另一个实例（或具有`string`、`number`和`sort_num`属性的任何鸭子类型对象；如果这些属性缺失，它将失败）进行比较。以下输出展示了这个类在排序时的工作原理：

```py
>>> a = WeirdSortee('a', 4, True)
>>> b = WeirdSortee('b', 3, True)
>>> c = WeirdSortee('c', 2, True)
>>> d = WeirdSortee('d', 1, True)
>>> l = [a,b,c,d]
>>> l
[a:4, b:3, c:2, d:1]
>>> l.sort()
>>> l
[d:1, c:2, b:3, a:4]
>>> for i in l:
...     i.sort_num = False
...
>>> l.sort()
>>> l
[a:4, b:3, c:2, d:1]

```

第一次调用`sort`时，它按数字排序，因为所有被比较的对象上的`sort_num`都是`True`。第二次，它按字母排序。我们只需要实现`__lt__`方法来启用排序。然而，从技术上讲，如果实现了它，类通常还应该实现类似的`__gt__`、`__eq__`、`__ne__`、`__ge__`和`__le__`方法，以便所有的`<`、`>`、`==`、`!=`、`>=`和`<=`操作符也能正常工作。通过实现`__lt__`和`__eq__`，然后应用`@total_ordering`类装饰器来提供其余的方法，你可以免费获得这些方法：

```py
from functools import total_ordering

@total_ordering
class WeirdSortee:
    def __init__(self, string, number, sort_num):
        self.string = string
        self.number = number
        self.sort_num = sort_num

    def __lt__(self, object):
        if self.sort_num:
            return self.number < object.number
        return self.string < object.string

    def __repr__(self):
        return"{}:{}".format(self.string, self.number)

    def __eq__(self, object):
        return all((
            self.string == object.string,
            self.number == object.number,
            self.sort_num == object.number
        ))
```

如果我们想要能够在我们的对象上使用运算符，这是很有用的。然而，如果我们只想自定义我们的排序顺序，即使这样也是过度的。对于这样的用例，`sort`方法可以接受一个可选的`key`参数。这个参数是一个函数，可以将列表中的每个对象转换为某种可比较的对象。例如，我们可以使用`str.lower`作为键参数，在字符串列表上执行不区分大小写的排序：

```py
>>> l = ["hello", "HELP", "Helo"]
>>> l.sort()
>>> l
['HELP', 'Helo', 'hello']
>>> l.sort(key=str.lower)
>>> l
['hello', 'Helo', 'HELP']

```

记住，即使`lower`是字符串对象上的一个方法，它也是一个可以接受单个参数`self`的函数。换句话说，`str.lower(item)`等同于`item.lower()`。当我们将这个函数作为键传递时，它会对小写值进行比较，而不是进行默认的区分大小写比较。

有一些排序键操作是如此常见，以至于 Python 团队已经提供了它们，这样你就不必自己编写了。例如，通常常见的是按列表中的第一个项目之外的其他内容对元组列表进行排序。`operator.itemgetter`方法可以用作键来实现这一点：

```py
>>> from operator import itemgetter
>>> l = [('h', 4), ('n', 6), ('o', 5), ('p', 1), ('t', 3), ('y', 2)]
>>> l.sort(key=itemgetter(1))
>>> l
[('p', 1), ('y', 2), ('t', 3), ('h', 4), ('o', 5), ('n', 6)]

```

`itemgetter`函数是最常用的一个（如果对象是字典，它也可以工作），但有时你会发现`attrgetter`和`methodcaller`也很有用，它们返回对象的属性和对象的方法调用的结果，用于相同的目的。有关更多信息，请参阅`operator`模块文档。

# 集合

列表是非常多才多艺的工具，适用于大多数容器对象应用。但是当我们想要确保列表中的对象是唯一的时，它们就不太有用了。例如，歌曲库可能包含同一位艺术家的许多歌曲。如果我们想要整理库并创建所有艺术家的列表，我们必须检查列表，看看我们是否已经添加了艺术家，然后再添加他们。

这就是集合的用武之地。集合来自数学，它们代表一个无序的（通常是）唯一数字的组。我们可以将一个数字添加到集合五次，但它只会出现一次。

在 Python 中，集合可以容纳任何可散列的对象，不仅仅是数字。可散列的对象与字典中可以用作键的对象相同；所以再次，列表和字典都不行。像数学集合一样，它们只能存储每个对象的一个副本。因此，如果我们试图创建一个歌手名单，我们可以创建一个字符串名称的集合，并简单地将它们添加到集合中。这个例子从一个（歌曲，艺术家）元组列表开始，并创建了一个艺术家的集合：

```py
song_library = [("Phantom Of The Opera", "Sarah Brightman"),
        ("Knocking On Heaven's Door", "Guns N' Roses"),
        ("Captain Nemo", "Sarah Brightman"),
        ("Patterns In The Ivy", "Opeth"),
        ("November Rain", "Guns N' Roses"),
        ("Beautiful", "Sarah Brightman"),
        ("Mal's Song", "Vixy and Tony")]

artists = set()
for song, artist in song_library:
    **artists.add(artist)

print(artists)
```

与列表和字典一样，没有内置的空集语法；我们使用`set()`构造函数创建一个集合。然而，我们可以使用花括号（从字典语法中借用）来创建一个集合，只要集合包含值。如果我们使用冒号来分隔值对，那就是一个字典，比如`{'key': 'value', 'key2': 'value2'}`。如果我们只用逗号分隔值，那就是一个集合，比如`{'value', 'value2'}`。可以使用`add`方法将项目单独添加到集合中。如果运行此脚本，我们会看到集合按照广告中的方式工作：

```py
{'Sarah Brightman', "Guns N' Roses", 'Vixy and Tony', 'Opeth'}

```

如果你注意输出，你会注意到项目的打印顺序并不是它们添加到集合中的顺序。集合和字典一样，是无序的。它们都使用基于哈希的数据结构来提高效率。因为它们是无序的，集合不能通过索引查找项目。集合的主要目的是将世界分为两组：“在集合中的事物”和“不在集合中的事物”。检查一个项目是否在集合中或循环遍历集合中的项目很容易，但如果我们想要对它们进行排序或排序，我们就必须将集合转换为列表。这个输出显示了这三种活动：

```py
>>> "Opeth" in artists
True
>>> for artist in artists:
...     print("{} plays good music".format(artist))
...
Sarah Brightman plays good music
Guns N' Roses plays good music
Vixy and Tony play good music
Opeth plays good music
>>> alphabetical = list(artists)
>>> alphabetical.sort()
>>> alphabetical
["Guns N' Roses", 'Opeth', 'Sarah Brightman', 'Vixy and Tony']

```

集合的主要*特征*是唯一性，但这并不是它的主要*目的*。当两个或更多个集合组合使用时，集合最有用。集合类型上的大多数方法都作用于其他集合，允许我们有效地组合或比较两个或更多个集合中的项目。这些方法有奇怪的名称，因为它们使用数学中使用的相同术语。我们将从三种返回相同结果的方法开始，不管哪个是调用集合，哪个是被调用集合。

`union`方法是最常见和最容易理解的。它将第二个集合作为参数，并返回一个新集合，其中包含两个集合中*任何一个*的所有元素；如果一个元素在两个原始集合中，它当然只会在新集合中出现一次。联合就像一个逻辑的`or`操作，实际上，`|`运算符可以用于两个集合执行联合操作，如果你不喜欢调用方法。

相反，交集方法接受第二个集合并返回一个新集合，其中只包含*两个*集合中的元素。这就像一个逻辑的`and`操作，并且也可以使用`&`运算符来引用。

最后，`symmetric_difference` 方法告诉我们剩下什么；它是一个集合，其中包含一个集合或另一个集合中的对象，但不包含两者都有的对象。以下示例通过比较我的歌曲库中的一些艺术家和我妹妹的歌曲库中的艺术家来说明这些方法：

```py
my_artists = {"Sarah Brightman", "Guns N' Roses",
        "Opeth", "Vixy and Tony"}

auburns_artists = {"Nickelback", "Guns N' Roses",
        "Savage Garden"}

print("All: {}".format(my_artists.union(auburns_artists)))
print("Both: {}".format(auburns_artists.intersection(my_artists)))
print("Either but not both: {}".format(
    my_artists.symmetric_difference(auburns_artists)))
```

如果我们运行这段代码，我们会发现这三种方法确实做了打印语句所暗示的事情：

```py
All: {'Sarah Brightman', "Guns N' Roses", 'Vixy and Tony',
'Savage Garden', 'Opeth', 'Nickelback'}
Both: {"Guns N' Roses"}
Either but not both: {'Savage Garden', 'Opeth', 'Nickelback',
'Sarah Brightman', 'Vixy and Tony'}

```

这些方法无论哪个集合调用另一个集合，都会返回相同的结果。我们可以说 `my_artists.union(auburns_artists)` 或 `auburns_artists.union(my_artists)`，结果都是一样的。还有一些方法，根据调用者和参数的不同会返回不同的结果。

这些方法包括 `issubset` 和 `issuperset`，它们是彼此的反义。两者都返回一个 `bool` 值。`issubset` 方法返回 `True`，如果调用集合中的所有项也在作为参数传递的集合中。`issuperset` 方法返回 `True`，如果参数中的所有项也在调用集合中。因此 `s.issubset(t)` 和 `t.issuperset(s)` 是相同的。如果 `t` 包含了 `s` 中的所有元素，它们都会返回 `True`。

最后，`difference` 方法返回调用集合中的所有元素，但不在作为参数传递的集合中；这类似于`symmetric_difference` 的一半。`difference` 方法也可以用 `-` 运算符表示。以下代码说明了这些方法的运行方式：

```py
my_artists = {"Sarah Brightman", "Guns N' Roses",
        "Opeth", "Vixy and Tony"}

bands = {"Guns N' Roses", "Opeth"}

print("my_artists is to bands:")
print("issuperset: {}".format(my_artists.issuperset(bands)))
print("issubset: {}".format(my_artists.issubset(bands)))
print("difference: {}".format(my_artists.difference(bands)))
print("*"*20)
print("bands is to my_artists:")
print("issuperset: {}".format(bands.issuperset(my_artists)))
print("issubset: {}".format(bands.issubset(my_artists)))
print("difference: {}".format(bands.difference(my_artists)))
```

这段代码简单地打印出了在一个集合上调用另一个集合时每个方法的响应。运行代码会得到以下输出：

```py
my_artists is to bands:
issuperset: True
issubset: False
difference: {'Sarah Brightman', 'Vixy and Tony'}
********************
bands is to my_artists:
issuperset: False
issubset: True
difference: set()

```

在第二种情况下，`difference` 方法返回一个空集，因为 `bands` 中没有不在 `my_artists` 中的项目。

`union`、`intersection` 和 `difference` 方法都可以接受多个集合作为参数；它们会返回我们所期望的，即在调用所有参数时创建的集合。

因此，集合上的方法清楚地表明集合是用来操作其他集合的，并且它们不仅仅是容器。如果我们有来自两个不同来源的数据，并且需要快速地以某种方式将它们合并，以确定数据重叠或不同之处，我们可以使用集合操作来高效地比较它们。或者，如果我们有可能包含已经处理过的数据的重复数据，我们可以使用集合来比较这两者，并仅处理新数据。

最后，了解到在使用 `in` 关键字检查成员资格时，集合比列表要高效得多。如果在集合或列表上使用语法 `value in container`，如果 `container` 中的一个元素等于 `value`，则返回 `True`，否则返回 `False`。但是，在列表中，它会查看容器中的每个对象，直到找到该值，而在集合中，它只是对该值进行哈希处理并检查成员资格。这意味着集合将以相同的时间找到值，无论容器有多大，但列表在搜索值时会花费越来越长的时间，因为列表包含的值越来越多。

# 扩展内置对象

我们在第三章中简要讨论了内置数据类型如何使用继承进行扩展。现在，我们将更详细地讨论何时需要这样做。

当我们有一个内置容器对象需要添加功能时，我们有两个选择。我们可以创建一个新对象，将该容器作为属性（组合），或者我们可以对内置对象进行子类化，并添加或调整方法以实现我们想要的功能（继承）。

如果我们只想使用容器来存储一些对象，使用组合通常是最好的选择，使用容器的特性。这样，很容易将数据结构传递到其他方法中，它们将知道如何与它交互。但是，如果我们想要改变容器的实际工作方式，我们需要使用继承。例如，如果我们想要确保`list`中的每个项目都是一个具有确切五个字符的字符串，我们需要扩展`list`并覆盖`append()`方法以引发无效输入的异常。我们还至少需要覆盖`__setitem__(self, index, value)`，这是列表上的一个特殊方法，每当我们使用`x[index] = "value"`语法时都会调用它，以及`extend()`方法。

是的，列表是对象。我们一直在访问列表或字典键，循环容器以及类似任务的特殊非面向对象的语法实际上是“语法糖”，它映射到对象导向范式下面。我们可能会问 Python 设计者为什么这样做。难道面向对象编程*总是*更好吗？这个问题很容易回答。在下面的假设例子中，哪个更容易阅读，作为程序员？哪个需要输入更少？

```py
c = a + b
c = a.add(b)

l[0] = 5
l.setitem(0, 5)
d[key] = value
d.setitem(key, value)

for x in alist:
    #do something with x
it = alist.iterator()
while it.has_next():
 **x = it.next()
    **#do something with x

```

突出显示的部分展示了面向对象的代码可能是什么样子（实际上，这些方法实际上存在于相关对象的特殊双下划线方法中）。Python 程序员一致认为，非面向对象的语法更容易阅读和编写。然而，所有前述的 Python 语法都映射到面向对象的方法下面。这些方法有特殊的名称（在前后都有双下划线），提醒我们有更好的语法。但是，它给了我们覆盖这些行为的手段。例如，我们可以创建一个特殊的整数，当我们将两个整数相加时总是返回`0`：

```py
class SillyInt(int):
    **def __add__(self, num):
        return 0
```

这是一个极端奇怪的事情，毫无疑问，但它完美地诠释了这些面向对象的原则：

```py
>>> a = SillyInt(1)
>>> b = SillyInt(2)
>>> a + b
0

```

`__add__`方法的绝妙之处在于我们可以将其添加到我们编写的任何类中，如果我们在该类的实例上使用`+`运算符，它将被调用。这就是字符串、元组和列表连接的工作原理，例如。

这适用于所有特殊方法。如果我们想要为自定义对象使用`x in myobj`语法，我们可以实现`__contains__`。如果我们想要使用`myobj[i] = value`语法，我们提供一个`__setitem__`方法，如果我们想要使用`something = myobj[i]`，我们实现`__getitem__`。

`list`类上有 33 个这样的特殊方法。我们可以使用`dir`函数查看所有这些方法：

```py
>>> dir(list)

['__add__', '__class__', '__contains__', '__delattr__','__delitem__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'append', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort'

```

此外，如果我们想要了解这些方法的工作方式的其他信息，我们可以使用`help`函数：

```py
>>> help(list.__add__)
Help on wrapper_descriptor:

__add__(self, value, /)
 **Return self+value.

```

列表上的加号运算符连接两个列表。我们没有空间来讨论本书中所有可用的特殊函数，但是现在您可以使用`dir`和`help`来探索所有这些功能。官方在线 Python 参考([`docs.python.org/3/`](https://docs.python.org/3/))也有很多有用的信息。特别关注`collections`模块中讨论的抽象基类。

因此，回到之前关于何时使用组合与继承的观点：如果我们需要以某种方式更改类上的任何方法，包括特殊方法，我们绝对需要使用继承。如果我们使用组合，我们可以编写执行验证或更改的方法，并要求调用者使用这些方法，但没有任何阻止他们直接访问属性。他们可以向我们的列表中插入一个不具有五个字符的项目，这可能会使列表中的其他方法感到困惑。

通常，需要扩展内置数据类型是使用错误类型的数据类型的迹象。这并不总是这样，但是如果我们想要扩展内置的话，我们应该仔细考虑是否不同的数据结构更合适。

例如，考虑创建一个记住插入键的顺序的字典需要做些什么。做到这一点的一种方法是保持一个有序的键列表，该列表存储在 `dict` 的特殊派生子类中。然后我们可以覆盖方法 `keys`、`values`、`__iter__` 和 `items` 以按顺序返回所有内容。当然，我们还必须覆盖 `__setitem__` 和 `setdefault` 以保持我们的列表最新。在 `dir(dict)` 的输出中可能还有一些其他方法需要覆盖以保持列表和字典一致（`clear` 和 `__delitem__` 记录了何时删除项目），但是在这个例子中我们不用担心它们。

因此，我们将扩展 `dict` 并添加一个有序键列表。这很简单，但我们在哪里创建实际的列表呢？我们可以将它包含在 `__init__` 方法中，这样就可以正常工作，但我们不能保证任何子类都会调用该初始化程序。还记得我们在第二章中讨论过的 `__new__` 方法吗？我说它通常只在非常特殊的情况下有用。这就是其中之一。我们知道 `__new__` 将被调用一次，并且我们可以在新实例上创建一个列表，该列表将始终对我们的类可用。考虑到这一点，这就是我们整个排序字典：

```py
from collections import KeysView, ItemsView, ValuesView
class DictSorted(dict):
    def __new__(*args, **kwargs):
        new_dict = dict.__new__(*args, **kwargs)
        new_dict.ordered_keys = []
        return new_dict

    def __setitem__(self, key, value):
        '''self[key] = value syntax'''
        if key not in self.ordered_keys:
            self.ordered_keys.append(key)
        super().__setitem__(key, value)

    def setdefault(self, key, value):
        if key not in self.ordered_keys:
            self.ordered_keys.append(key)
        return super().setdefault(key, value)

    def keys(self):
        return KeysView(self)

    def values(self):
        return ValuesView(self)

    def items(self):
        return ItemsView(self)

    def __iter__(self):
        '''for x in self syntax'''
        return self.ordered_keys.__iter__()
```

`__new__` 方法创建一个新的字典，然后在该对象上放置一个空列表。我们不覆盖 `__init__`，因为默认实现有效（实际上，只有在我们初始化一个空的 `DictSorted` 对象时才是真的。如果我们想要支持 `dict` 构造函数的其他变体，它接受字典或元组列表，我们需要修复 `__init__` 以更新我们的 `ordered_keys` 列表）。设置项目的两种方法非常相似；它们都更新键列表，但只有在项目之前没有添加时才更新。我们不希望列表中有重复项，但我们不能在这里使用集合；它是无序的！

`keys`、`items` 和 `values` 方法都返回字典的视图。collections 库为字典提供了三个只读的 `View` 对象；它们使用 `__iter__` 方法循环遍历键，然后使用 `__getitem__`（我们不需要覆盖）来检索值。因此，我们只需要定义我们自定义的 `__iter__` 方法来使这三个视图工作。你可能会认为超类会使用多态性正确地创建这些视图，但如果我们不覆盖这三个方法，它们就不会返回正确排序的视图。

最后，`__iter__` 方法是真正特殊的；它确保如果我们循环遍历字典的键（使用 `for`...`in` 语法），它将按正确的顺序返回值。它通过返回 `ordered_keys` 列表的 `__iter__` 来实现这一点，该列表返回的是与我们在列表上使用 `for`...`in` 时使用的相同的迭代器对象。由于 `ordered_keys` 是所有可用键的列表（由于我们覆盖其他方法的方式），这也是字典的正确迭代器对象。

让我们看看这些方法中的一些是如何运作的，与普通字典相比：

```py
>>> ds = DictSorted()
>>> d = {}
>>> ds['a'] = 1
>>> ds['b'] = 2
>>> ds.setdefault('c', 3)
3
>>> d['a'] = 1
>>> d['b'] = 2
>>> d.setdefault('c', 3)
3
>>> for k,v in ds.items():
...     print(k,v)
...
a 1
b 2
c 3
>>> for k,v in d.items():
...     print(k,v)
...
a 1
c 3
b 2

```

啊，我们的字典是有序的，而普通字典不是。万岁！

### 注意

如果您想在生产中使用这个类，您将不得不覆盖其他几个特殊方法，以确保在所有情况下键都是最新的。但是，您不需要这样做；这个类提供的功能在 Python 中已经可用，使用 `collections` 模块中的 `OrderedDict` 对象。尝试从 `collections` 导入该类，并使用 `help(OrderedDict)` 了解更多信息。

# 队列

队列是奇特的数据结构，因为像集合一样，它们的功能可以完全使用列表来处理。然而，虽然列表是非常多才多艺的通用工具，但有时它们并不是最有效的容器数据结构。如果您的程序使用的是小型数据集（在今天的处理器上最多有数百甚至数千个元素），那么列表可能会涵盖所有您的用例。但是，如果您需要将数据扩展到百万级别，您可能需要一个更有效的容器来满足您特定的用例。因此，Python 提供了三种类型的队列数据结构，具体取决于您要查找的访问类型。所有三种都使用相同的 API，但在行为和数据结构上有所不同。

然而，在我们开始使用队列之前，考虑一下可靠的列表数据结构。Python 列表是许多用例中最有利的数据结构：

+   它们支持对列表中的任何元素进行高效的随机访问

+   它们有严格的元素排序

+   它们支持高效的附加操作

然而，如果您在列表的末尾之外的任何位置插入元素，它们往往会很慢（特别是如果是列表的开头）。正如我们在集合部分讨论的那样，它们对于检查元素是否存在于列表中，以及通过扩展搜索也很慢。存储数据按排序顺序或重新排序数据也可能效率低下。

让我们来看看 Python `queue`模块提供的三种类型的容器。

## FIFO 队列

FIFO 代表**先进先出**，代表了“队列”这个词最常见的定义。想象一下在银行或收银台排队的人群。第一个进入队列的人先得到服务，第二个人得到第二个服务，如果有新的人需要服务，他们加入队列的末尾等待轮到他们。

Python `Queue`类就像这样。它通常被用作一种通信媒介，当一个或多个对象产生数据，而一个或多个其他对象以某种方式消耗数据时，可能以不同的速率。想象一下一个消息应用程序，它从网络接收消息，但只能一次向用户显示一条消息。其他消息可以按接收顺序缓存在队列中。FIFO 队列在这种并发应用程序中被广泛使用。（我们将在第十二章中更多地讨论并发，*测试面向对象的程序*。）

当您不需要访问数据结构内部的任何数据，只需要访问下一个要消耗的对象时，`Queue`类是一个不错的选择。使用列表会更低效，因为在列表的底层，插入数据（或从列表中删除数据）可能需要移动列表中的每个其他元素。

队列有一个非常简单的 API。一个“队列”可以有“无限”（直到计算机耗尽内存）的容量，但更常见的是限制到某个最大大小。主要方法是`put()`和`get()`，它们将一个元素添加到队列的末尾，并按顺序从前面检索它们。这两种方法都接受可选参数来控制如果操作无法成功完成会发生什么，因为队列要么为空（无法获取）要么已满（无法放置）。默认行为是阻塞或空闲等待，直到`Queue`对象有数据或空间可用来完成操作。您可以通过传递`block=False`参数来代替引发异常。或者您可以通过传递`timeout`参数在引发异常之前等待一定的时间。

该类还有方法来检查`Queue`是否`full()`或`empty()`，还有一些额外的方法来处理并发访问，我们这里不讨论。这是一个演示这些原则的交互式会话：

```py
>>> from queue import Queue
>>> lineup = Queue(maxsize=3)
>>> lineup.get(block=False)
Traceback (most recent call last):
 **File "<ipython-input-5-a1c8d8492c59>", line 1, in <module>
 **lineup.get(block=False)
 **File "/usr/lib64/python3.3/queue.py", line 164, in get
 **raise Empty
queue.Empty
>>> lineup.put("one")
>>> lineup.put("two")
>>> lineup.put("three")
>>> lineup.put("four", timeout=1)
Traceback (most recent call last):
 **File "<ipython-input-9-4b9db399883d>", line 1, in <module>
 **lineup.put("four", timeout=1)
 **File "/usr/lib64/python3.3/queue.py", line 144, in put
raise Full
queue.Full
>>> lineup.full()
True
>>> lineup.get()
'one'
>>> lineup.get()
'two'
>>> lineup.get()
'three'
>>> lineup.empty()
True

```

在底层，Python 使用`collections.deque`数据结构实现队列。双端队列是一种先进先出的数据结构，可以有效地访问集合的两端。它提供了一个比`Queue`更灵活的接口。如果你想要更多地尝试它，我建议你参考 Python 文档。

## LIFO 队列

**LIFO**（**后进先出**）队列更常被称为**栈**。想象一叠文件，你只能访问最顶部的文件。你可以在栈的顶部放另一张纸，使其成为新的顶部纸，或者你可以拿走最顶部的纸，以显示其下面的纸。

传统上，栈的操作被命名为 push 和 pop，但 Python 的`queue`模块使用与 FIFO 队列完全相同的 API：`put()`和`get()`。然而，在 LIFO 队列中，这些方法操作的是栈的“顶部”，而不是队列的前后。这是多态的一个很好的例子。如果你查看 Python 标准库中`Queue`的源代码，你会发现实际上有一个超类和子类，用于实现 FIFO 和 LIFO 队列之间的一些关键不同的操作（在栈的顶部而不是`deque`实例的前后进行操作）。

以下是 LIFO 队列的一个示例：

```py
>>> from queue import LifoQueue
>>> stack = LifoQueue(maxsize=3)
>>> stack.put("one")
>>> stack.put("two")
>>> stack.put("three")
>>> stack.put("four", block=False)
Traceback (most recent call last):
 **File "<ipython-input-21-5473b359e5a8>", line 1, in <module>
 **stack.put("four", block=False)
 **File "/usr/lib64/python3.3/queue.py", line 133, in put
 **raise Full
queue.Full

>>> stack.get()
'three'
>>> stack.get()
'two'
>>> stack.get()
'one'
>>> stack.empty()
True
>>> stack.get(timeout=1)
Traceback (most recent call last):
 **File "<ipython-input-26-28e084a84a10>", line 1, in <module>
 **stack.get(timeout=1)
 **File "/usr/lib64/python3.3/queue.py", line 175, in get
 **raise Empty
queue.Empty

```

你可能会想为什么不能只是在标准列表上使用`append()`和`pop()`方法。坦率地说，那可能是我会做的事情。我很少有机会在生产代码中使用`LifoQueue`类。与列表的末尾一起工作是一个高效的操作；实际上，`LifoQueue`在内部使用了标准列表！

有几个原因你可能想要使用`LifoQueue`而不是列表。最重要的原因是`LifoQueue`支持多个线程的干净并发访问。如果你需要在并发环境中使用类似栈的行为，你应该把列表留在家里。其次，`LifoQueue`实施了栈接口。你不能无意中在`LifoQueue`中插入一个值到错误的位置（尽管作为一个练习，你可以想出如何完全有意识地这样做）。

## 优先队列

优先队列实施了一种与以前队列实现非常不同的排序方式。再次强调，它们遵循完全相同的`get()`和`put()`API，但是不是依赖于项目到达的顺序来确定它们应该何时被返回，而是返回最“重要”的项目。按照约定，最重要或最高优先级的项目是使用小于运算符排序最低的项目。

一个常见的约定是在优先队列中存储元组，其中元组中的第一个元素是该元素的优先级，第二个元素是数据。另一个常见的范例是实现`__lt__`方法，就像我们在本章前面讨论的那样。在队列中可以有多个具有相同优先级的元素，尽管不能保证哪一个会被首先返回。

例如，搜索引擎可能使用优先队列来确保在爬行不太可能被搜索的网站之前刷新最受欢迎的网页的内容。产品推荐工具可能使用它来显示关于排名最高的产品的信息，同时加载排名较低的数据。

请注意，优先队列总是返回当前队列中最重要的元素。`get()`方法将阻塞（默认情况下）如果队列为空，但如果队列中已经有东西，它不会阻塞并等待更高优先级的元素被添加。队列对尚未添加的元素一无所知（甚至对先前提取的元素也一无所知），只根据队列当前的内容做出决定。

这个交互式会话展示了优先队列的工作原理，使用元组作为权重来确定处理项目的顺序：

```py
>>> heap.put((3, "three"))
>>> heap.put((4, "four"))
>>> heap.put((1, "one") )
>>> heap.put((2, "two"))
>>> heap.put((5, "five"), block=False)
Traceback (most recent call last):
 **File "<ipython-input-23-d4209db364ed>", line 1, in <module>
 **heap.put((5, "five"), block=False)
 **File "/usr/lib64/python3.3/queue.py", line 133, in put
 **raise Full
Full
>>> while not heap.empty():
 **print(heap.get())
(1, 'one')
(2, 'two')
(3, 'three')
(4, 'four')

```

几乎所有的优先队列都是使用`heap`数据结构实现的。Python 的实现利用`heapq`模块来有效地在普通列表中存储一个堆。我建议您查阅算法和数据结构的教科书，以获取有关堆的更多信息，更不用说我们在这里没有涵盖的许多其他迷人的结构了。无论数据结构如何，您都可以使用面向对象的原则来封装相关的算法（行为），就像`queue`模块在标准库中为我们所做的那样。

# 案例研究

为了把一切联系在一起，我们将编写一个简单的链接收集器，它将访问一个网站，并收集该站点上每个页面上的每个链接。不过，在我们开始之前，我们需要一些测试数据来使用。简单地编写一些 HTML 文件，这些文件包含彼此之间的链接，以及到互联网上其他站点的链接，就像这样：

```py
<html>
    <body>
        <a href="contact.html">Contact us</a>
        <a href="blog.html">Blog</a>
        <a href="esme.html">My Dog</a>
        <a href="/hobbies.html">Some hobbies</a>
        <a href="/contact.html">Contact AGAIN</a>
        <a href="http://www.archlinux.org/">Favorite OS</a>
    </body>
</html>
```

将其中一个文件命名为`index.html`，这样当页面被提供时它会首先显示出来。确保其他文件存在，并且保持复杂，以便它们之间有很多链接。本章的示例包括一个名为`case_study_serve`的目录（存在的最无聊的个人网站之一！）如果您不想自己设置它们。

现在，通过进入包含所有这些文件的目录来启动一个简单的 Web 服务器，并运行以下命令：

```py
python3 -m http.server

```

这将启动一个运行在 8000 端口的服务器；您可以通过在浏览器中访问`http://localhost:8000/`来查看您创建的页面。

### 注意

我怀疑没有人能够轻松地让一个网站运行起来！永远不要说，“你不能用 Python 轻松地做到这一点。”

目标是向我们的收集器传递站点的基本 URL（在本例中为：`http://localhost:8000/`），并让它创建一个包含站点上每个唯一链接的列表。我们需要考虑三种类型的 URL（指向外部站点的链接，以`http://`开头，绝对内部链接，以`/`字符开头，以及其他情况的相对链接）。我们还需要意识到页面可能会以循环方式相互链接；我们需要确保我们不会多次处理相同的页面，否则它可能永远不会结束。在所有这些唯一性发生时，听起来我们需要一些集合。

在我们开始之前，让我们从基础知识开始。我们需要什么代码来连接到一个页面并解析该页面上的所有链接？

```py
from urllib.request import urlopen
from urllib.parse import urlparse
import re
import sys
LINK_REGEX = re.compile(
        "<a [^>]*href='\"['\"][^>]*>")

class LinkCollector:
    def __init__(self, url):
        self.url = "" + urlparse(url).netloc

    def collect_links(self, path="/"):
        full_url = self.url + path
        page = str(urlopen(full_url).read())
        links = LINK_REGEX.findall(page)
        print(links)

if __name__ == "__main__":
    LinkCollector(sys.argv[1]).collect_links()
```

考虑到它的功能，这是一小段代码。它连接到命令行传递的服务器，下载页面，并提取该页面上的所有链接。`__init__`方法使用`urlparse`函数从 URL 中提取主机名；因此，即使我们传入`http://localhost:8000/some/page.html`，它仍将在主机的顶层`http://localhost:8000/`上运行。这是有道理的，因为我们想收集站点上的所有链接，尽管它假设每个页面都通过某些链接序列连接到索引。

`collect_links`方法连接到服务器并下载指定页面，并使用正则表达式在页面中找到所有链接。正则表达式是一种非常强大的字符串处理工具。不幸的是，它们有一个陡峭的学习曲线；如果您以前没有使用过它们，我强烈建议您学习任何一本完整的书籍或网站上的相关主题。如果您认为它们不值得了解，那么尝试在没有它们的情况下编写前面的代码，您会改变主意的。

示例还在`collect_links`方法的中间停止，以打印链接的值。这是测试程序的常见方法：停下来输出值，以确保它是我们期望的值。这是我们示例的输出：

```py
['contact.html', 'blog.html', 'esme.html', '/hobbies.html',
'/contact.html', 'http://www.archlinux.org/']
```

现在我们已经收集了第一页中的所有链接。我们可以用它做什么？我们不能只是将链接弹出到一个集合中以删除重复项，因为链接可能是相对的或绝对的。例如，`contact.html`和`/contact.html`指向同一个页面。因此，我们应该做的第一件事是将所有链接规范化为它们的完整 URL，包括主机名和相对路径。我们可以通过向我们的对象添加一个`normalize_url`方法来实现这一点：

```py
    def normalize_url(self, path, link):
        if link.startswith("http://"):
            return link
        elif link.startswith("/"):
            return self.url + link
        else:
            return self.url + path.rpartition(
                '/')[0] + '/' + link
```

这种方法将每个 URL 转换为包括协议和主机名的完整地址。现在两个联系页面具有相同的值，我们可以将它们存储在一个集合中。我们将不得不修改`__init__`来创建这个集合，以及`collect_links`来将所有链接放入其中。

然后，我们将不得不访问所有非外部链接并收集它们。但等一下；如果我们这样做，我们如何防止在遇到同一个页面两次时重新访问链接？看起来我们实际上需要两个集合：一个收集链接的集合，一个访问链接的集合。这表明我们明智地选择了一个集合来表示我们的数据；我们知道在操作多个集合时，集合是最有用的。让我们设置这些：

```py
class LinkCollector:
    def __init__(self, url):
        self.url = "http://+" + urlparse(url).netloc
        **self.collected_links = set()
        **self.visited_links = set()

    def collect_links(self, path="/"):
        full_url = self.url + path
        **self.visited_links.add(full_url)
        page = str(urlopen(full_url).read())
        links = LINK_REGEX.findall(page)
        **links = {self.normalize_url(path, link
            **) for link in links}
        **self.collected_links = links.union(
                **self.collected_links)
        **unvisited_links = links.difference(
                **self.visited_links)
        **print(links, self.visited_links,
                **self.collected_links, unvisited_links)

```

创建规范化链接列表的行使用了`set`推导，与列表推导没有什么不同，只是结果是一组值。我们将在下一章中详细介绍这些。再次，该方法停下来打印当前值，以便我们可以验证我们没有混淆我们的集合，并且`difference`确实是我们想要调用的方法来收集`unvisited_links`。然后我们可以添加几行代码，循环遍历所有未访问的链接，并将它们添加到收集中：

```py
        for link in unvisited_links:
            if link.startswith(self.url):
                self.collect_links(urlparse(link).path)
```

`if`语句确保我们只从一个网站收集链接；我们不想去收集互联网上所有页面的所有链接（除非我们是 Google 或互联网档案馆！）。如果我们修改程序底部的主要代码以输出收集到的链接，我们可以看到它似乎已经收集了它们所有：

```py
if __name__ == "__main__":
    collector = LinkCollector(sys.argv[1])
    collector.collect_links()
    for link in collector.collected_links:
        print(link)
```

它显示了我们收集到的所有链接，只显示了一次，即使我的示例中的许多页面多次链接到彼此：

```py
$ python3 link_collector.py http://localhost:8000
http://localhost:8000/
http://en.wikipedia.org/wiki/Cavalier_King_Charles_Spaniel
http://beluminousyoga.com
http://archlinux.me/dusty/
http://localhost:8000/blog.html
http://ccphillips.net/
http://localhost:8000/contact.html
http://localhost:8000/taichi.html
http://www.archlinux.org/
http://localhost:8000/esme.html
http://localhost:8000/hobbies.html

```

即使它收集了指向外部页面的链接，它也没有去收集我们链接到的任何外部页面的链接。如果我们想收集站点中的所有链接，这是一个很棒的小程序。但它并没有给我提供构建站点地图所需的所有信息；它告诉我我有哪些页面，但它没有告诉我哪些页面链接到其他页面。如果我们想要做到这一点，我们将不得不进行一些修改。

我们应该做的第一件事是查看我们的数据结构。收集链接的集合不再起作用；我们想知道哪些链接是从哪些页面链接过来的。因此，我们可以做的第一件事是将该集合转换为我们访问的每个页面的集合字典。字典键将表示当前集合中的确切数据。值将是该页面上的所有链接的集合。以下是更改：

```py
from urllib.request import urlopen
from urllib.parse import urlparse
import re
import sys
LINK_REGEX = re.compile(
        "<a [^>]*href='\"['\"][^>]*>")

class LinkCollector:
    def __init__(self, url):
        self.url = "http://%s" % urlparse(url).netloc
        **self.collected_links = {}
        self.visited_links = set()

    def collect_links(self, path="/"):
        full_url = self.url + path
        self.visited_links.add(full_url)
        page = str(urlopen(full_url).read())
        links = LINK_REGEX.findall(page)
        links = {self.normalize_url(path, link
            ) for link in links}
        **self.collected_links[full_url] = links
        **for link in links:
            **self.collected_links.setdefault(link, set())
        unvisited_links = links.difference(
                self.visited_links)
        for link in unvisited_links:
            if link.startswith(self.url):
                self.collect_links(urlparse(link).path)

    def normalize_url(self, path, link):
        if link.startswith("http://"):
            return link
        elif link.startswith("/"):
            return self.url + link
        else:
            return self.url + path.rpartition('/'
                    )[0] + '/' + link
if __name__ == "__main__":
    collector = LinkCollector(sys.argv[1])
    collector.collect_links()
    **for link, item in collector.collected_links.items():
        **print("{}: {}".format(link, item))

```

这是一个令人惊讶的小改变；原来创建两个集合的行已被三行代码替换，用于更新字典。其中第一行简单地告诉字典该页面的收集链接是什么。第二行使用`setdefault`为字典中尚未添加到字典中的任何项目创建一个空集。结果是一个包含所有链接的字典，将其键映射到所有内部链接的链接集，外部链接为空集。

最后，我们可以使用队列来存储尚未处理的链接，而不是递归调用`collect_links`。这种实现不支持它，但这将是创建一个多线程版本的良好第一步，该版本可以并行进行多个请求以节省时间。

```py
from urllib.request import urlopen
from urllib.parse import urlparse
import re
import sys
from queue import Queue
LINK_REGEX = re.compile("<a [^>]*href='\"['\"][^>]*>")

class LinkCollector:
    def __init__(self, url):
        self.url = "http://%s" % urlparse(url).netloc
        self.collected_links = {}
        self.visited_links = set()

    def collect_links(self):
        queue = Queue()
        queue.put(self.url)
        while not queue.empty():
            url = queue.get().rstrip('/')
            self.visited_links.add(url)
            page = str(urlopen(url).read())
            links = LINK_REGEX.findall(page)
            links = {
                self.normalize_url(urlparse(url).path, link)
                for link in links
            }
            self.collected_links[url] = links
            for link in links:
                self.collected_links.setdefault(link, set())
            unvisited_links = links.difference(self.visited_links)
            for link in unvisited_links:
                if link.startswith(self.url):
                    queue.put(link)

    def normalize_url(self, path, link):
        if link.startswith("http://"):
            return link.rstrip('/')
        elif link.startswith("/"):
            return self.url + link.rstrip('/')
        else:
            return self.url + path.rpartition('/')[0] + '/' + link.rstrip('/')

if __name__ == "__main__":
    collector = LinkCollector(sys.argv[1])
    collector.collect_links()
    for link, item in collector.collected_links.items():
        print("%s: %s" % (link, item))
```

在这个版本的代码中，我不得不手动去除`normalize_url`方法中的任何尾部斜杠，以消除重复项。

因为最终结果是一个未排序的字典，所以对链接进行处理的顺序没有限制。因此，在这里我们可以使用`LifoQueue`而不是`Queue`。由于在这种情况下没有明显的优先级可附加到链接上，使用优先级队列可能没有太多意义。

# 练习

选择正确数据结构的最佳方法是多次选择错误。拿出你最近写过的一些代码，或者写一些使用列表的新代码。尝试使用一些不同的数据结构来重写它。哪些更合理？哪些不合理？哪些代码最优雅？

尝试使用几种不同的数据结构。你可以查看你之前章节练习中做过的例子。有没有对象和方法，你本来可以使用`namedtuple`或`dict`？尝试一下，看看结果如何。有没有本来可以使用集合的字典，因为你实际上并没有访问值？有没有检查重复项的列表？集合是否足够？或者可能需要几个集合？哪种队列实现更有效？将 API 限制在堆栈顶部是否有用，而不是允许随机访问列表？

如果你想要一些具体的例子来操作，可以尝试将链接收集器改编为同时保存每个链接使用的标题。也许你可以生成一个 HTML 站点地图，列出站点上的所有页面，并包含一个链接到其他页面的链接列表，使用相同的链接标题命名。

最近是否编写了任何容器对象，可以通过继承内置对象并重写一些“特殊”双下划线方法来改进？你可能需要进行一些研究（使用`dir`和`help`，或 Python 库参考）来找出哪些方法需要重写。你确定继承是应用的正确工具吗？基于组合的解决方案可能更有效吗？在决定之前尝试两种方法（如果可能的话）。尝试找到不同的情况，其中每种方法都比另一种更好。

如果在开始本章之前，你已经熟悉各种 Python 数据结构及其用途，你可能会感到无聊。但如果是这种情况，很可能你使用数据结构太多了！看看你以前的一些代码，并重写它以使用更多自制对象。仔细考虑各种替代方案，并尝试它们所有；哪一个使系统更易读和易维护？

始终对你的代码和设计决策进行批判性评估。养成审查旧代码的习惯，并注意如果你对“良好设计”的理解自你编写代码以来有所改变。软件设计有很大的审美成分，就像带有油画的艺术家一样，我们都必须找到最适合自己的风格。

# 总结

我们已经介绍了几种内置数据结构，并试图了解如何为特定应用程序选择其中一种。有时，我们能做的最好的事情就是创建一类新的对象，但通常情况下，内置的数据结构提供了我们需要的东西。当它不提供时，我们总是可以使用继承或组合来使它们适应我们的用例。我们甚至可以重写特殊方法来完全改变内置语法的行为。

在下一章中，我们将讨论如何整合 Python 的面向对象和非面向对象的方面。在此过程中，我们将发现它比乍一看更面向对象化！
