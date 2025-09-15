# Python 数据结构

在我们之前的例子中，我们已经看到了许多内置的 Python 数据结构在行动。你可能也在入门书籍或教程中了解了它们中的许多。在本章中，我们将讨论这些数据结构的面向对象特性，它们应该在什么情况下代替常规类使用，以及它们不应该在什么情况下使用。特别是，我们将涵盖以下主题：

+   元组和命名元组

+   数据类

+   字典

+   列表和集合

+   如何和为什么扩展内置对象

+   三种类型的队列

# 空对象

让我们从最基本的 Python 内置对象开始，这是我们已经多次见过的，我们在创建的每个类中都扩展了它：`object`。技术上，我们可以不写子类就实例化一个`object`，如下所示：

```py
    >>> o = object()
    >>> o.x = 5
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    AttributeError: 'object' object has no attribute 'x'  
```

不幸的是，正如你所看到的，直接实例化的`object`上无法设置任何属性。这并不是因为 Python 开发者想要强迫我们编写自己的类，或者任何如此邪恶的事情。他们这样做是为了节省内存；大量的内存。当 Python 允许一个对象具有任意属性时，它需要一定量的系统内存来跟踪每个对象具有哪些属性，包括存储属性名称及其值。即使没有存储属性，也会为*潜在的*新属性分配内存。考虑到典型的 Python 程序中可能有数十、数百或数千个对象（每个类都扩展了对象），这样一小块内存很快就会变成大量的内存。因此，Python 默认禁用了`object`和几个其他内置对象的任意属性。

我们可以使用**槽**来限制我们自己的类上的任意属性。槽超出了本书的范围，但如果你在寻找更多信息，你现在有一个搜索词。在正常使用中，使用槽并没有多少好处，但如果你正在编写将在整个系统中重复数千次的对象，它们可以帮助节省内存，就像它们对`object`所做的那样。

然而，创建我们自己的空对象类是非常简单的；我们在最早期的例子中看到了它：

```py
class MyObject: 
    pass 
```

正如我们之前看到的，可以在这样的类上设置属性，如下所示：

```py
>>> m = MyObject()
>>> m.x = "hello"
>>> m.x
'hello'  
```

如果我们想要将属性组合在一起，我们可以将它们存储在一个空对象中，如下所示。但通常我们更倾向于使用其他专为存储数据设计的内置函数。本书一直强调，只有当你想指定**数据**和**行为**时，才应该使用类和对象。编写一个空类的主要原因是为了快速排除某些内容，知道我们稍后会回来添加行为。将行为适应类比用对象替换数据结构并更改所有对其的引用要容易得多。因此，从一开始就决定数据仅仅是数据，还是伪装成对象的，这一点很重要。一旦做出这个设计决策，其余的设计就会自然而然地就位。

# 元组和命名元组

元组是可以按顺序存储其他特定数量对象的对象。它们是**不可变**的，这意味着我们无法在运行时添加、删除或替换对象。这看起来可能是一个巨大的限制，但事实是，如果你需要修改元组，你使用的数据类型是错误的（通常，列表会更合适）。元组不可变性的主要好处是我们可以将它们用作字典的键，以及在需要对象哈希值的其他位置。

元组用于存储数据；无法将行为与元组关联。如果我们需要行为来操作元组，我们必须将元组传递给一个函数（或另一个对象上的方法）以执行该操作。

元组通常应存储彼此不同的值。例如，我们不会在元组中放入三个股票代码，但我们可以创建一个包含股票代码及其当日当前价、最高价和最低价的元组。元组的主要目的是将不同的数据片段聚合到一个容器中。因此，元组可以是最容易替换“无数据对象”习语的工具。

我们可以通过用逗号分隔值来创建一个元组。通常，元组被括号包围以使其易于阅读，并与其他表达式的其他部分区分开来，但这并非总是必需的。以下两个赋值是相同的（它们记录了一家相当盈利公司的股票、当前价格、最高价和最低价）：

```py
>>> stock = "FB", 177.46, 178.67, 175.79
>>> stock2 = ("FB", 177.46, 178.67, 175.79)
```

如果我们在某个其他对象内部组合一个元组，例如函数调用、列表推导或生成器，则需要括号。否则，解释器将无法知道它是一个元组还是下一个函数参数。例如，以下函数接受一个元组和日期，并返回一个包含日期和股票最高价与最低价之间中间值的元组：

```py
import datetime

def middle(stock, date):
 symbol, current, high, low = stock
    return (((high + low) / 2), date)

mid_value, date = middle(
    ("FB", 177.46, 178.67, 175.79), datetime.date(2018, 8, 27)
)
```

元组直接在函数调用内部通过用逗号分隔值并包围整个元组在括号中创建。然后，元组后面跟着一个逗号，以将其与第二个参数分开。

这个例子也说明了*元组解包*。函数内部的第 一行将`stock`参数解包成四个不同的变量。元组必须与变量的数量完全相同，否则会引发异常。我们还可以在最后一条语句中看到元组解包的例子，其中函数返回的元组被解包成两个值，`mid_value`和`date`。当然，这做起来很奇怪，因为我们最初已经向函数提供了日期，但这给了我们一个机会看到解包是如何工作的。

解包是 Python 中的一个非常有用的特性。我们可以将变量分组在一起，以便更容易地存储和传递它们，但当我们需要访问所有变量时，我们可以将它们解包成单独的变量。当然，有时我们只需要访问元组中的一个变量。我们可以使用与其他序列类型（例如列表和字符串）相同的语法来访问单个值：

```py
>>> stock = "FB", 75.00, 75.03, 74.90
>>> high = stock[2]
>>> high
75.03  
```

我们甚至可以使用切片符号来提取元组的更大部分，如下所示：

```py
>>> stock[1:3]
(75.00, 75.03)  
```

这些示例虽然说明了元组有多灵活，但也展示了它们的一个主要缺点：可读性。阅读这段代码的人如何知道特定元组的第二个位置是什么？他们可以猜测，从我们分配给它的变量的名字来看，它可能是某种“高”值，但如果我们在没有分配的情况下直接在计算中访问元组值，就没有这样的提示。他们必须翻遍代码，找到元组声明的地方，才能发现它的作用。

在某些情况下，直接访问元组成员是可以的，但不要养成这种习惯。这种所谓的*魔法数字*（似乎从空中出现，在代码中没有明显的意义）是许多编码错误的来源，并导致数小时的沮丧调试。尽量只在知道所有值都将同时有用并且通常在访问时会被解包的情况下使用元组。如果你必须直接访问成员或使用切片，并且该值的用途不是立即显而易见，至少包括一个注释说明它从何而来。

# 命名元组

那么，当我们想要将值分组在一起，但又知道我们经常需要单独访问它们时，我们该怎么办呢？实际上有几个选择。我们可以使用一个空对象，正如之前所讨论的（但这很少有用，除非我们预计以后会添加行为），或者我们可以使用一个字典（如果我们不知道确切的数据量或哪些具体数据将被存储，这非常有用），我们将在后面的章节中介绍。另外两种选择是命名元组，我们将在本节中讨论，以及数据类，在下一节中介绍。

如果我们不需要向对象添加行为，并且事先知道需要存储哪些属性，我们可以使用命名元组。命名元组是带有态度的元组。它们是分组只读数据的一个好方法。

构建命名元组比正常元组要复杂一些。首先，我们必须导入 `namedtuple`，因为它默认不在命名空间中。然后，我们通过给它一个名称并概述其属性来描述命名元组。这返回一个类对象，我们可以用所需值实例化它，并且可以多次实例化，如下所示：

```py
from collections import namedtuple 
Stock = namedtuple("Stock", ["symbol", "current", "high", "low"])
stock = Stock("FB", 177.46, high=178.67, low=175.79) 
```

`namedtuple` 构造函数接受两个参数。第一个是一个用于命名元组的标识符。第二个是命名元组所需的字符串属性列表。结果是可以通过像正常类一样调用来实例化其他对象的实例。构造函数必须具有恰好正确的参数数量，这些参数可以作为参数或关键字参数传递。与正常对象一样，我们可以创建任意数量的此类实例，每个实例具有不同的值。

请注意，不要将保留关键字（例如 class）用作命名元组的属性。

结果的 `namedtuple` 可以像正常元组一样打包、解包、索引、切片，以及其他处理，但我们也可以像访问对象上的单个属性一样访问它：

```py
>>> stock.high
175.79
>>> symbol, current, high, low = stock
>>> current
177.46  
```

记住，创建命名元组是一个两步过程。首先，使用 `collections.namedtuple` 创建一个类，然后构建该类的实例。

命名元组非常适合许多仅用于数据的表示，但它们并不适用于所有情况。像元组和字符串一样，命名元组是不可变的，因此一旦设置属性后，我们无法修改它。例如，自从我们开始这次讨论以来，我们公司股票的当前价值已经下降，但我们无法设置新值，如下所示：

```py
>>> stock.current = 74.98
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
AttributeError: can't set attribute  
```

如果我们需要能够更改存储的数据，数据类可能就是我们需要的东西。

# 数据类

数据类基本上是常规对象，但带有一些额外的特性。

提供了用于预定义属性的简洁语法。有几种创建方法，我们将在本节中逐一探讨。

最简单的方法是使用与命名元组类似的构造，如下所示：

```py
from dataclasses import make_dataclass
Stock = make_dataclass("Stock", "symbol", "current", "high", "low")
stock = Stock("FB", 177.46, high=178.67, low=175.79)
```

一旦实例化，股票对象就可以像任何常规类一样使用。您可以访问和更新属性，甚至可以给对象分配其他任意属性，如下所示：

```py
>>> stock
Stock(symbol='FB', current=177.46, high=178.67, low=175.79)
>>> stock.current
177.46
>>> stock.current=178.25
>>> stock
Stock(symbol='FB', current=178.25, high=178.67, low=175.79)
>>> stock.unexpected_attribute = 'allowed'
>>> stock.unexpected_attribute
'allowed'
```

乍一看，数据类似乎并没有比具有适当构造函数的正常对象带来多少好处：

```py
class StockRegClass:
    def __init__(self, name, current, high, low):
        self.name = name
        self.current = current
        self.high = high
        self.low = low

stock_reg_class = Stock("FB", 177.46, high=178.67, low=175.79)
```

明显的好处是，使用 `make_dataclass`，您可以在一行中定义类，而不是六行。如果您再仔细一点，您会发现数据类还提供了一个比常规版本更有用的字符串表示。它还免费提供等价比较。以下示例比较了常规类与这些数据类功能：

```py
>>> stock_reg_class
<__main__.Stock object at 0x7f506bf4ec50>
>>> stock_reg_class2 = StockRegClass("FB", 177.46, 178.67, 175.79)
>>> stock_reg_class2 == stock_reg_class
False
>>> stock2 = Stock("FB", 177.46, 178.67, 175.79)
>>> stock2 == stock
True 
```

正如我们很快就会看到的，数据类还有许多其他有用的功能。但首先，让我们看看定义数据类的另一种（更常见）方法。参考以下代码块：

```py
from dataclasses import dataclass

@dataclass
class StockDecorated:
    name: str
    current: float
    high: float
    low: float
```

如果您之前没有见过类型提示，这种语法可能看起来真的很奇怪。这些所谓的变量注解是在 Python 3.6 中引入到语言中的。我将类型提示归类为**本书范围之外的内容**，所以如果您想了解更多关于它们的信息，请自行进行网络搜索。现在，只需知道前面的确实是合法的 Python 语法，并且它确实有效。您不必相信我的话；只需运行代码并观察是否存在语法错误！

如果您不想使用类型提示，或者您的属性接受一个复杂类型或类型集的值，请指定类型为`Any`。您可以使用`from typing import Any`将`Any`类型引入您的命名空间。

`dataclass`函数作为类装饰器应用。我们在上一章讨论属性时遇到了装饰器。我承诺在未来的章节中会详细介绍它们。我将在第十章履行这个承诺。现在，只需知道这种语法是生成数据类所必需的。

承认，这种语法与带有`__init__`的常规类相比并没有少多少冗余，但它给了我们访问几个额外的数据类功能。例如，您可以指定数据类的默认值。也许市场目前关闭，您不知道当天的值：

```py
@dataclass
class StockDefaults:
    name: str
    current: float = 0.0
    high: float = 0.0
    low: float = 0.0
```

您可以使用股票名称来构建这个类；其余的值将采用默认值。但您仍然可以指定值，如下所示：

```py
>>> StockDefaults('FB')
StockDefaults(name='FB', current=0.0, high=0.0, low=0.0)
>>> StockDefaults('FB', 177.46, 178.67, 175.79)
StockDefaults(name='FB', current=177.46, high=178.67, low=175.79) 
```

我们之前看到，数据类自动支持相等比较。如果所有属性都相等，则数据类也相等。默认情况下，数据类不支持其他比较，如小于或大于，并且不能排序。但是，如果您愿意，可以轻松添加比较，如下所示：

```py
@dataclass(order=True)
class StockOrdered:
    name: str
    current: float = 0.0
    high: float = 0.0
    low: float = 0.0

stock_ordered1 = StockDecorated("FB", 177.46, high=178.67, low=175.79)
stock_ordered2 = StockOrdered("FB")
stock_ordered3 = StockDecorated("FB", 178.42, high=179.28, low=176.39)
```

在这个例子中，我们所做的唯一改变是在数据类构造函数中添加了`order=True`关键字。但这给了我们排序和比较以下值的机会：

```py
>>> stock_ordered1 < stock_ordered2
False
>>> stock_ordered1 > stock_ordered2
True
>>> from pprint import pprint
>>> pprint(sorted([stock_ordered1, stock_ordered2, stock_ordered3]))
[StockOrdered(name='FB', current=0.0, high=0.0, low=0.0),
 StockOrdered(name='FB', current=177.46, high=178.67, low=175.79),
 StockOrdered(name='FB', current=178.42, high=179.28, low=176.39)] 
```

当数据类接收到`order=True`参数时，它将默认根据每个属性定义的顺序比较值。因此，在这种情况下，它首先比较两个类上的名称。如果它们相同，它将比较当前价格。如果这些也相同，它将比较最高价和最低价。您可以通过在类的`__post_init__`方法内提供一个`sort_index`属性来自定义排序顺序，但我将让您自行上网搜索以获取此和其他高级用法（如不可变性）的完整细节，因为这一部分已经相当长，我们还有许多其他数据结构要研究。

# 字典

字典是非常有用的容器，允许我们直接将对象映射到其他对象。一个具有属性的空对象就像是一种字典；属性的名称映射到属性值。这实际上比听起来更接近真相；内部，对象通常将属性表示为字典，其中值是对象上的属性或方法（如果你不相信我，请查看`__dict__`属性）。甚至模块上的属性也是内部存储在字典中的。

字典在根据特定的键对象查找值方面非常高效。当你想根据其他对象找到某个对象时，应该始终使用字典。被存储的对象称为**值**；用作索引的对象称为**键**。我们在之前的某些示例中已经看到了字典的语法。

字典可以通过使用`dict()`构造函数或使用`{}`语法快捷方式来创建。实际上，后者格式几乎总是被使用。我们可以通过使用冒号分隔键和值，以及使用逗号分隔键值对来预先填充字典。

例如，在一个股票应用程序中，我们通常会想通过股票符号来查找价格。我们可以创建一个使用股票符号作为键的字典，以及包含当前价、最高价和最低价的元组（当然，你也可以使用命名元组或数据类作为值）。如下所示：

```py
stocks = {
    "GOOG": (1235.20, 1242.54, 1231.06),
    "MSFT": (110.41, 110.45, 109.84),
}
```

如前所述的示例，我们可以在字典中通过请求方括号内的键来查找值。如果键不在字典中，它将引发异常，如下所示：

```py
>>> stocks["GOOG"]
(1235.20, 1242.54, 1231.06)
>>> stocks["RIM"]
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
KeyError: 'RIM'  
```

我们当然可以捕获`KeyError`并处理它。但我们还有其他选择。记住，字典是对象，即使它们的目的是持有其他对象。因此，它们具有与之相关的几种行为。其中最有用的方法之一是`get`方法；它接受一个键作为第一个参数，如果键不存在，则接受一个可选的默认值：

```py
>>> print(stocks.get("RIM"))
None
>>> stocks.get("RIM", "NOT FOUND")
'NOT FOUND'  
```

为了获得更多的控制，我们可以使用`setdefault`方法。如果键在字典中，此方法的行为就像`get`一样；它返回该键的值。否则，如果键不在字典中，它不仅会返回我们在方法调用中提供的默认值（就像`get`做的那样）；它还会将该键设置为该值。另一种思考方式是，`setdefault`仅在值之前未设置的情况下才在字典中设置值。然后，它返回字典中的值；要么是已经存在的值，要么是新提供的默认值，如下所示：

```py
>>> stocks.setdefault("GOOG", "INVALID")
(613.3, 625.86, 610.5)
>>> stocks.setdefault("BBRY", (10.87, 10.76, 10.90))
(10.50, 10.62, 10.39)
>>> stocks["BBRY"]
(10.50, 10.62, 10.39)  
```

`GOOG` 股票已经在字典中，所以当我们尝试将其 `setdefault` 到一个无效值时，它只是返回字典中已有的值。`BBRY` 不在字典中，所以 `setdefault` 返回了默认值，并为我们设置了字典中的新值。然后我们检查新股票确实在字典中。

有三个非常实用的字典方法：`keys()`、`values()` 和 `items()`。前两个返回字典中所有键和所有值的迭代器。如果我们想处理所有的键或值，我们可以像使用列表一样使用它们或在 `for` 循环中使用它们。`items()` 方法可能是最有用的；它返回一个迭代器，遍历字典中每个项的 `(key, value)` 对。这非常适合在 `for` 循环中进行元组解包，以遍历相关的键和值。这个例子就是这样做的，以打印出字典中每个股票及其当前值：

```py
>>> for stock, values in stocks.items():
... print(f"{stock} last value is {values[0]}")
...
GOOG last value is 1235.2
MSFT last value is 110.41
BBRY last value is 10.5
```

每个键/值元组被解包成两个变量，分别命名为 `stock` 和 `values`（我们可以使用任何我们想要的变量名，但这两个似乎都很合适），然后以格式化的字符串形式打印出来。

注意，股票显示的顺序与它们被插入的顺序相同。在 Python 3.6 之前，这不是真的，直到 Python 3.7 才成为语言定义的正式部分。在此之前，底层字典实现使用了一个不同的底层数据结构，它不是有序的。在字典中需要排序的情况相当罕见，但如果确实需要，并且需要支持 Python 3.5 或更早版本，请确保使用 `OrderedDict` 类，它可以从 `collections` 模块中获取。

因此，一旦实例化了一个字典，就有许多方法可以检索数据：我们可以使用方括号作为索引语法，使用 `get` 方法，使用 `setdefault` 方法，或者遍历 `items` 方法，等等。

最后，正如你可能已经知道的，我们可以使用与检索值相同的索引语法在字典中设置一个值：

```py
>>> stocks["GOOG"] = (1245.21, 1252.64, 1245.18)
>>> stocks['GOOG']
(1245.21, 1252.64, 1245.18)
```

由于今天谷歌的价格更高，所以我已更新了字典中的元组值。我们可以使用这种索引语法为任何键设置值，无论该键是否在字典中。如果它在字典中，旧值将被新值替换；否则，将创建一个新的键/值对。

我们到目前为止一直在使用字符串作为字典键，但我们并不局限于字符串键。使用字符串作为键很常见，特别是当我们将数据存储在字典中以收集它们时（而不是使用具有命名属性的对象或数据类）。但我们也可以使用元组、数字，甚至是我们自己定义的对象作为字典键。我们甚至可以在单个字典中使用不同类型的键，如下所示：

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

这段代码展示了我们可以提供给字典的几种不同类型的键。它还展示了一种不能使用的对象类型。我们已经广泛使用了列表，我们将在下一节看到更多关于它们的细节。因为列表可以在任何时候改变（例如，通过添加或删除项），它们不能**哈希**到特定的值。

**可哈希**的对象基本上有一个将对象转换为唯一整数值的算法，以便在字典中进行快速查找。这个哈希值实际上是用来在字典中查找值的。例如，字符串根据字符串中字符的字节值映射到整数，而元组则结合元组内项的哈希值。任何被认为相等（如具有相同字符的字符串或具有相同值的元组）的两个对象都应该有相同的哈希值，并且对象的哈希值永远不应该改变。然而，列表的内容可以改变，这会改变其哈希值（两个列表只有在内容相同的情况下才应该相等）。正因为如此，它们不能用作字典键。出于同样的原因，字典也不能用作其他字典的键。

相比之下，用作字典值的对象类型没有限制。例如，我们可以使用一个字符串键映射到一个列表值，或者我们可以在另一个字典中有一个嵌套字典作为值。

# 字典使用案例

字典非常灵活，有众多用途。字典可以有两种主要的使用方式。第一种是所有键都代表类似对象的不同实例的字典；例如，我们的股票字典。这是一个索引系统。我们使用股票符号作为值的索引。值甚至可以是复杂的、自定义的对象，具有进行买卖决策或设置止损的方法，而不仅仅是我们的简单元组。

第二种设计是每个键代表单个结构的某个方面的字典；在这种情况下，我们可能会为每个对象使用一个单独的字典，并且它们都会有相似（尽管通常不是完全相同）的键集。这种后一种情况通常也可以用命名元组或数据类来解决。这可能会让人困惑；我们如何决定使用哪种？

当我们知道数据必须存储的确切属性时，我们通常会使用数据类，特别是如果我们还想将类定义作为最终用户的文档时。

数据类是 Python 标准库中的一个较新的添加（自 Python 3.7 以来）。我预计它们将取代许多命名元组的使用场景。如果打算从函数返回它们，命名元组也可能很有用。这允许调用函数在需要时使用元组解包。数据类是不可迭代的，因此不能遍历或解包它们的值。

另一方面，如果描述对象的键事先未知，或者不同的对象在键上有所差异，那么字典会是一个更好的选择。如果我们事先不知道所有键是什么，那么使用字典可能更好。

从技术上讲，大多数 Python 对象都是在底层使用字典实现的。你可以通过将一个对象加载到交互式解释器中并查看 `obj.__dict__` 魔术属性来看到这一点。当你使用 `obj.attr_name` 在对象上访问属性时，它本质上是在底层将查找转换为 `obj['attr_name']`。这比那更复杂，但你能抓住要点。甚至数据类也有一个 `__dict__` 属性，这仅仅表明字典有多么灵活。请注意，并非所有对象都存储在字典中。有一些特殊类型，如列表、字典和日期时间，是以不同的方式实现的，主要是为了效率。如果 `dict` 实例的 `__dict__` 属性是一个 `dict` 实例，那当然会很奇怪，不是吗？

# 使用 `defaultdict`

我们已经看到了如何使用 `setdefault` 来设置一个默认值，如果键不存在的话，但如果每次查找值时都需要设置默认值，这可能会变得有点单调。例如，如果我们正在编写代码来计算一个句子中某个字母出现的次数，我们可以这样做：

```py
def letter_frequency(sentence): 
    frequencies = {} 
    for letter in sentence: 
        frequency = frequencies.setdefault(letter, 0) 
        frequencies[letter] = frequency + 1 
    return frequencies 
```

每次我们访问字典时，都需要检查它是否已经有了一个值，如果没有，就将其设置为零。当每次请求一个空键时都需要这样做时，我们可以使用字典的不同版本，称为 `defaultdict`：

```py
from collections import defaultdict 
def letter_frequency(sentence): 
 frequencies = defaultdict(int) 
    for letter in sentence: 
        frequencies[letter] += 1 
    return frequencies 
```

这段代码看起来根本不可能工作。`defaultdict` 在其构造函数中接受一个函数。每当访问一个不在字典中的键时，它会调用该函数，不带任何参数，以创建一个默认值。

在这种情况下，它调用的函数是 `int`，这是整数对象的构造函数。通常，我们通过在代码中输入一个整数来创建整数，如果我们使用 `int` 构造函数创建一个，我们传递给它我们想要创建的项目（例如，将数字字符串转换为整数）。但是，如果我们不带任何参数调用 `int`，它将方便地返回数字零。在这段代码中，如果字母不存在于 `defaultdict` 中，当我们访问它时将返回数字零。然后，我们给这个数字加一，以表示我们找到了该字母的一个实例，下次我们再找到它时，这个数字将被返回，我们再次增加值。

`defaultdict`对于创建容器字典很有用。如果我们想创建过去 30 天的收盘价字典，我们可以使用股票代码作为键，并将价格存储在`list`中；当我们第一次访问股票价格时，我们希望它创建一个空列表。只需将`list`传递给`defaultdict`，每次访问空键时它都会被调用。如果我们想将一个空集合或空字典与一个键关联起来，我们可以做类似的事情。

当然，我们也可以编写自己的函数并将它们传递给`defaultdict`。假设我们想创建一个`defaultdict`，其中每个新元素都包含一个元组，表示在该时刻插入字典中的项目数量，以及一个空列表来存储其他东西。我们不太可能想创建这样的对象，但让我们看看：

```py
from collections import defaultdict

num_items = 0 

def tuple_counter(): 
    global num_items 
    num_items += 1 
    return (num_items, []) 

d = defaultdict(tuple_counter) 
```

当我们运行此代码时，我们可以在一个单独的语句中访问空键并将它们插入到列表中：

```py
>>> d = defaultdict(tuple_counter)
>>> d['a'][1].append("hello")
>>> d['b'][1].append('world')
>>> d
defaultdict(<function tuple_counter at 0x82f2c6c>,
{'a': (1, ['hello']), 'b': (2, ['world'])})  
```

当我们打印`dict`在最后，我们看到计数器确实在起作用。

这个例子虽然简洁地展示了如何为`defaultdict`创建自己的函数，但实际上代码并不好；使用全局变量意味着如果我们创建了四个不同的`defaultdict`段，每个都使用了`tuple_counter`，它将计算所有字典中的条目数，而不是每个字典都有自己的计数。最好创建一个类并将该类的方法传递给`defaultdict`。

# 计数器

你可能会认为`defaultdict(int)`已经很简单了，但“我想在可迭代对象中计数特定实例”的使用场景足够常见，以至于 Python 开发者为它创建了一个特定的类。之前计算字符串中字符数量的代码可以很容易地在一行中计算：

```py
from collections import Counter 
def letter_frequency(sentence): 
 return Counter(sentence) 
```

`Counter`对象的行为类似于一个增强的字典，其中键是被计数的项，值是这些项的数量。最有用的函数之一是`most_common()`方法。它返回一个按计数排序的（键，计数）元组列表。你可以可选地传递一个整数参数到`most_common()`，以请求只获取最常见的元素。例如，你可以编写一个简单的投票应用程序如下：

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

想必你会从数据库中获取响应，或者通过使用计算机视觉算法来计数举手的孩子。在这里，我们将其硬编码，以便我们可以测试`most_common`方法。它返回一个只有一个元素的列表（因为我们请求了一个参数中的元素）。这个元素存储了位置为零的最高选择的名字，因此调用末尾的`[0][0]`。我认为它们看起来像惊讶的表情，不是吗？你的电脑可能对它能够如此容易地计数数据感到惊讶。它的祖先，霍勒里斯的制表机，是为 1890 年美国人口普查开发的，一定非常嫉妒！

# 列表

列表是 Python 数据结构中最非面向对象的。虽然列表本身是对象，但 Python 中有大量的语法来使使用它们尽可能无痛。与许多其他面向对象的语言不同，Python 中的列表是直接可用的。我们不需要导入它们，也很少需要调用它们的方法。我们可以遍历列表而不需要显式请求迭代器对象，并且我们可以使用自定义语法构造列表（就像字典一样）。此外，列表推导和生成器表达式使它们成为计算功能的瑞士军刀。

我们不会过多地介绍语法；你已经在网络上的入门教程和本书之前的例子中看到了它。你不可能长时间编写 Python 代码而不学习如何使用列表！相反，我们将讨论何时应该使用列表，以及它们作为对象的本性。如果你不知道如何创建或向列表中追加，如何从列表中检索项目，或者什么是*切片表示法*，我建议你立即查阅官方 Python 教程。它可以在网上找到，网址是[`docs.python.org/3/tutorial/`](http://docs.python.org/3/tutorial/)。

在 Python 中，当我们想要存储同一类型的多个实例时，通常应该使用列表；字符串列表或数字列表；通常是自定义对象的列表。当我们想要按某种顺序存储项目时，应该始终使用列表。通常，这是它们被插入的顺序，但它们也可以根据其他标准进行排序。

正如我们在上一章的案例研究中看到的，当我们需要修改内容时，列表也非常有用：向列表中插入或从列表中删除任意位置的内容，或者更新列表中的值。

与字典一样，Python 列表使用一个极其高效和调优良好的内部数据结构，这样我们就可以关注我们存储的内容，而不是如何存储它。许多面向对象的语言为队列、栈、链表和基于数组的列表提供了不同的数据结构。Python 确实提供了这些类的一些特殊实例，如果需要优化对大量数据的访问。然而，通常情况下，列表数据结构可以同时满足所有这些目的，并且程序员可以完全控制它们如何访问它。

不要使用列表来收集单个项目的不同属性。我们不想，例如，有一个特定形状的属性列表。元组、命名元组、字典和对象都更适合这个目的。在某些语言中，他们可能会创建一个列表，其中每个交替的项目是不同类型；例如，他们可能会为我们的字母频率列表写`['a', 1, 'b', 3]`。他们必须使用一个奇怪的循环同时访问列表中的两个元素，或者使用模运算符来确定正在访问哪个位置。

在 Python 中不要这样做。我们可以使用字典，就像我们在上一节中所做的那样，或者使用元组列表来将相关项分组在一起。以下是一个相当复杂的反例，演示了我们可以如何使用列表执行频率示例。它比字典示例复杂得多，并说明了选择正确（或错误）的数据结构对我们的代码可读性的影响。这如下所示：

```py
import string 
CHARACTERS  = list(string.ascii_letters) + [" "] 

def letter_frequency(sentence): 
    frequencies = [(c, 0) for c in CHARACTERS] 
    for letter in sentence: 
        index = CHARACTERS.index(letter) 
        frequencies[index] = (letter,frequencies[index][1]+1) 
    return frequencies 
```

这段代码从一个可能的字符列表开始。`string.ascii_letters` 属性提供了一个包含所有字母（大小写）的字符串，并按顺序排列。我们将这个字符串转换为列表，然后使用列表连接（`+` 运算符将两个列表合并为一个）添加一个额外的字符，一个空格。这些就是我们的频率列表中可用的字符（如果我们尝试添加不在列表中的字母，代码会出错，但异常处理程序可以解决这个问题）。

函数内部的第一行使用列表推导将 `CHARACTERS` 列表转换为元组列表。列表推导是 Python 中一个重要的非面向对象工具；我们将在下一章详细讲解它们。

然后，我们遍历句子中的每个字符。我们首先在 `CHARACTERS` 列表中查找字符的索引，我们知道这个索引在我们的频率列表中也是相同的，因为我们刚刚从第一个列表创建了第二个列表。然后我们通过创建一个新的元组来更新频率列表中的那个索引，丢弃原始的元组。除了垃圾回收和内存浪费的担忧之外，这相当难以阅读！

和字典一样，列表也是对象，并且它们有多个可以在其上调用的方法。以下是一些常见的方法：

+   `append(element)` 方法将一个元素添加到列表的末尾

+   `insert(index, element)` 方法在指定位置插入一个项目

+   `count(element)` 方法告诉我们一个元素在列表中出现的次数

+   `index()` 方法告诉我们列表中项目的索引，如果找不到它则抛出异常

+   `find()` 方法做同样的事情，但如果没有找到项目则返回 `-1` 而不是抛出异常

+   `reverse()` 方法确实如其名所示——将列表反转

+   `sort()` 方法有一些相当复杂的面向对象行为，我们现在将讲解

# 排序列表

不带任何参数，`sort` 通常会按预期工作。如果是一个字符串列表，它将按字母顺序排列。这个操作是区分大小写的，所以所有大写字母都会在所有小写字母之前排序；也就是说，`Z` 在 `a` 之前。如果是一个数字列表，它们将按数值顺序排序。如果提供了一个包含不可排序项的列表，排序将引发 `TypeError` 异常。

如果我们想要将我们定义的对象放入列表中并使这些对象可排序，我们必须做更多的工作。特殊的 `__lt__` 方法，代表“小于”，应该在类中定义，以便该类的实例可以进行比较。列表上的 `sort` 方法将访问每个对象上的此方法以确定它在列表中的位置。此方法应该在我们类以某种方式小于传递的参数时返回 `True`，否则返回 `False`。以下是一个相当愚蠢的类，可以根据字符串或数字进行排序：

```py
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
        return f"{self.string}:{self.number}"

```

`__repr__` 方法使得在打印列表时很容易看到两个值。`__lt__` 方法的实现是将对象与同一类的另一个实例（或任何具有 `string`、`number` 和 `sort_num` 属性的 duck-typed 对象）进行比较（如果这些属性缺失，它将失败）。以下输出展示了当涉及到排序时这个类是如何工作的：

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

第一次调用 `sort` 时，它按数字排序，因为所有被比较的对象上 `sort_num` 都是 `True`。第二次，它按字母排序。我们只需要实现 `__lt__` 方法来启用排序。然而，从技术上讲，如果实现了它，类通常也应该实现类似的 `__gt__`、`__eq__`、`__ne__`、`__ge__` 和 `__le__` 方法，以便所有 `<`、`>`、`==`、`!=`、`>=` 和 `<=` 运算符也能正常工作。通过实现 `__lt__` 和 `__eq__`，然后应用 `@total_ordering` 类装饰器来提供其余部分，你可以免费获得这些功能：

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
        return f"{self.string}:{self.number}"

 def __eq__(self, object): 
        return all(( 
            self.string == object.string, 
            self.number == object.number, 
            self.sort_num == object.number 
        )) 
```

这很有用，如果我们想能够在我们的对象上使用运算符。然而，如果我们只想自定义我们的排序顺序，即使是这样做也是过度的。对于这样的用例，`sort` 方法可以接受一个可选的 `key` 参数。这个参数是一个函数，可以将列表中的每个对象转换成可以比较的对象。例如，我们可以使用 `str.lower` 作为键参数，在字符串列表上执行不区分大小写的排序，如下所示：

```py
>>> l = ["hello", "HELP", "Helo"]
>>> l.sort()
>>> l
['HELP', 'Helo', 'hello']
>>> l.sort(key=str.lower)
>>> l
['hello', 'Helo', 'HELP']  
```

记住，尽管 `lower` 是字符串对象上的一个方法，但它也是一个可以接受单个参数 `self` 的函数。换句话说，`str.lower(item)` 等同于 `item.lower()`。当我们把这个函数作为键传递时，它会在小写值上执行比较，而不是执行默认的大小写敏感比较。

有一些排序键操作非常常见，Python 团队已经提供了它们，这样你就不必自己编写它们。例如，按列表中的第一个元素以外的其他元素对元组列表进行排序是很常见的。可以使用 `operator.itemgetter` 方法作为键来完成这个操作：

```py
>>> from operator import itemgetter
>>> l = [('h', 4), ('n', 6), ('o', 5), ('p', 1), ('t', 3), ('y', 2)]
>>> l.sort(key=itemgetter(1))
>>> l
[('p', 1), ('y', 2), ('t', 3), ('h', 4), ('o', 5), ('n', 6)]  
```

`itemgetter` 函数是最常用的一个（如果对象是字典也可以使用），但有时你也会用到 `attrgetter` 和 `methodcaller`，它们返回对象的属性和对象方法调用的结果，用于相同的目的。有关更多信息，请参阅 `operator` 模块文档。

# 集合

列表是极其多才多艺的工具，适用于许多容器对象应用。但是，当我们想要确保列表中的对象是唯一的时候，它们就不再有用。例如，一个音乐库可能包含许多同一艺术家的歌曲。如果我们想要在库中排序并创建所有艺术家的列表，我们必须在再次添加之前检查列表，看看我们是否已经添加了该艺术家。

这就是集合发挥作用的地方。集合来自数学，它们代表一个无序的（通常是）唯一数字的组。我们可以将一个数字添加到集合中五次，但它只会出现在集合中一次。

在 Python 中，集合可以包含任何可哈希的对象，而不仅仅是数字。可哈希的对象是那些可以用作字典键的对象；因此，列表和字典又排除了。像数学集合一样，它们只能存储每个对象的单个副本。所以如果我们试图创建一个歌曲艺术家的列表，我们可以创建一个字符串名称的集合，并将它们简单地添加到集合中。这个例子从一个包含（歌曲，艺术家）元组的列表开始，创建了一个艺术家集合：

```py
song_library = [
    ("Phantom Of The Opera", "Sarah Brightman"),
    ("Knocking On Heaven's Door", "Guns N' Roses"),
    ("Captain Nemo", "Sarah Brightman"),
    ("Patterns In The Ivy", "Opeth"),
    ("November Rain", "Guns N' Roses"),
    ("Beautiful", "Sarah Brightman"),
    ("Mal's Song", "Vixy and Tony"),
]

artists = set()
for song, artist in song_library:
    artists.add(artist)

print(artists)
```

与列表和字典不同，没有为空集合提供内置语法；我们使用 `set()` 构造函数创建集合。然而，只要集合包含值，我们就可以使用大括号（从字典语法中借用）来创建一个集合。如果我们使用冒号分隔值对，它就是一个字典，例如 `{'key': 'value', 'key2': 'value2'}`。如果我们只是用逗号分隔值，它就是一个集合，例如 `{'value', 'value2'}`。

可以使用集合的 `add` 方法单独向集合中添加项目。如果我们运行这个脚本，我们会看到集合按预期工作：

```py
{'Sarah Brightman', "Guns N' Roses", 'Vixy and Tony', 'Opeth'}  
```

如果你注意到了输出，你会注意到项目不是按照它们被添加到集合中的顺序打印的。由于基于哈希的数据结构以提高效率，集合本身是无序的。由于这种无序性，集合不能通过索引查找项目。集合的主要目的是将世界分为两组：*集合中的事物*和*不在集合中的事物*。检查一个项目是否在集合中或遍历集合中的项目很容易，但如果我们想要对它们进行排序或排序，我们必须将集合转换为列表。这个输出显示了所有这三个活动：

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

虽然集合的主要特征是唯一性，但这并不是它的主要用途。当两个或多个集合组合使用时，集合最有用。集合类型上的大多数方法都作用于其他集合，使我们能够有效地组合或比较两个或多个集合中的项目。这些方法有奇怪的名字，因为它们使用了数学中的术语。我们将从三个返回相同结果的方法开始，无论调用集合和被调用集合是哪一个。

`union`方法是最常见的，也最容易理解。它接受一个作为参数的第二个集合，并返回一个新集合，该集合包含两个集合中的所有元素；如果一个元素在原始的两个集合中，它当然只会在新集合中出现一次。Union 就像一个逻辑`or`操作。实际上，如果你不喜欢调用方法，可以使用`|`运算符对两个集合执行并集操作。

相反，交集方法接受一个第二个集合，并返回一个新集合，该集合仅包含两个集合中都有的元素。它就像一个逻辑`and`操作，也可以使用`&`运算符来引用。

最后，`symmetric_difference`方法告诉我们剩下的是什么；它是那些在一个集合或另一个集合中但不在两个集合中的对象的集合。以下示例通过比较两个不同的人喜欢的某些艺术家来说明这些方法：

```py
first_artists = {
    "Sarah Brightman",
    "Guns N' Roses",
    "Opeth",
    "Vixy and Tony",
}

second_artists = {"Nickelback", "Guns N' Roses", "Savage Garden"}

print("All: {}".format(first_artists.union(second_artists)))
print("Both: {}".format(second_artists.intersection(first_artists)))
print(
    "Either but not both: {}".format(
        first_artists.symmetric_difference(second_artists)
    )
)

```

如果我们运行这段代码，我们会看到这三个方法都做了打印语句所暗示的事情：

```py
All: {'Sarah Brightman', "Guns N' Roses", 'Vixy and Tony',
'Savage Garden', 'Opeth', 'Nickelback'}
Both: {"Guns N' Roses"}
Either but not both: {'Savage Garden', 'Opeth', 'Nickelback',
'Sarah Brightman', 'Vixy and Tony'}  
```

这些方法无论哪个集合调用另一个集合都返回相同的结果。我们可以说`first_artists.union(second_artists)`或`second_artists.union(first_artists)`并得到相同的结果。还有一些方法根据调用者和参数返回不同的结果。

这些方法包括`issubset`和`issuperset`，它们是彼此的逆。两者都返回一个布尔值。`issubset`方法在调用集合中的所有项目也在作为参数传递的集合中时返回`True`。`issuperset`方法在参数中的所有项目也在调用集合中时返回`True`。因此，`s.issubset(t)`和`t.issuperset(s)`是相同的。如果`t`包含`s`中的所有元素，它们都会返回`True`。

最后，`difference`方法返回调用集合中但在作为参数传递的集合中不存在的所有元素；这就像半个`symmetric_difference`。`difference`方法也可以用`-`运算符表示。以下代码展示了这些方法的作用：

```py
first_artists = {"Sarah Brightman", "Guns N' Roses", 
        "Opeth", "Vixy and Tony"} 

bands = {"Guns N' Roses", "Opeth"} 

print("first_artists is to bands:") 
print("issuperset: {}".format(first_artists.issuperset(bands))) 
print("issubset: {}".format(first_artists.issubset(bands))) 
print("difference: {}".format(first_artists.difference(bands))) 
print("*"*20) 
print("bands is to first_artists:") 
print("issuperset: {}".format(bands.issuperset(first_artists))) 
print("issubset: {}".format(bands.issubset(first_artists))) 
print("difference: {}".format(bands.difference(first_artists))) 
```

这段代码只是简单地打印出从一个集合调用另一个集合时每个方法的响应。运行它给出以下输出：

```py
first_artists is to bands:
issuperset: True
issubset: False
difference: {'Sarah Brightman', 'Vixy and Tony'}
********************
bands is to first_artists:
issuperset: False
issubset: True
difference: set()  
```

在第二种情况下，`difference`方法返回一个空集，因为`bands`中没有不在`first_artists`中的项目。

`union`、`intersection`和`difference`方法都可以接受多个集合作为参数；正如我们可能预期的，它们将返回当操作被调用在所有参数上时创建的集合。

因此，集合上的方法清楚地表明，集合旨在操作其他集合，并且它们不仅仅是容器。如果我们从两个不同的来源接收数据，并需要以某种方式快速合并它们，以确定数据重叠或不同，我们可以使用集合操作来有效地比较它们。或者，如果我们接收到的数据可能包含已经处理过的数据的重复项，我们可以使用集合来比较这两个数据集，并只处理新数据。

最后，了解集合在检查成员资格时比列表更有效率是有价值的。如果你在集合或列表上使用`value in container`语法，如果`container`中的任何一个元素等于`value`，它将返回`True`，否则返回`False`。然而，在列表中，它将检查容器中的每个对象，直到找到该值，而在集合中，它只是对值进行哈希并检查成员资格。这意味着无论容器有多大，集合都会在相同的时间内找到值，但列表随着包含更多值而搜索值所需的时间会越来越长。

# 扩展内置函数

我们在第三章中简要讨论了*当对象相似时*，如何使用继承扩展内置数据类型。现在，我们将更详细地讨论我们何时想要这样做。

当我们有一个想要添加功能的内置容器对象时，我们有两个选择。我们可以创建一个新的对象，它将该容器作为属性持有（组合），或者我们可以创建内置对象的子类，并添加或修改方法以实现我们想要的功能（继承）。

如果我们只想使用容器来存储一些对象并利用该容器的特性，那么组合通常是最好的选择。这样，很容易将这种数据结构传递给其他方法，它们将知道如何与之交互。但是，如果我们想要改变容器实际工作的方式，我们就需要使用继承。例如，如果我们想要确保`list`中的每个元素都是一个恰好有五个字符的字符串，我们需要扩展`list`并重写`append()`方法来为无效输入抛出异常。我们可能还需要最小化地重写`__setitem__(self, index, value)`，这是一个列表上的特殊方法，每次我们使用`x[index] = "value"`语法时都会被调用，以及`extend()`方法。

是的，列表是对象。我们之前看到的用于访问列表或字典键、遍历容器和类似任务的特殊非面向对象语法，实际上是映射到面向对象范式的 `syntactic sugar`。我们可能会问 Python 设计者为什么这样做。面向对象编程难道不是“总是”更好的吗？这个问题很容易回答。在以下假设的例子中，作为程序员，哪个更容易阅读？哪个需要更少的输入？：

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
    x = it.next() 
    #do something with x 
```

突出的部分显示了面向对象代码可能的样子（实际上，这些方法作为特殊双下划线方法存在于相关对象上）。Python 程序员一致认为，非面向对象的语法在阅读和编写时都更容易。然而，所有前面的 Python 语法在底层都映射到面向对象的方法。这些方法有特殊的名字（前后都有双下划线），以提醒我们还有更好的语法。然而，它给了我们覆盖这些行为的方法。例如，我们可以创建一个特殊的整数，当我们将其与另一个整数相加时，它总是返回 `0`，如下所示：

```py
class SillyInt(int): 
    def __add__(self, num): 
        return 0 
```

虽然这样做非常奇怪，但它完美地说明了这些面向对象原则的实际应用：

```py
>>> a = SillyInt(1)
>>> b = SillyInt(2)
>>> a + b
0  
```

`__add__` 方法的神奇之处在于我们可以将其添加到我们编写的任何类中，如果我们使用该类的实例上的 `+` 运算符，它将被调用。这就是字符串、元组和列表连接工作的方式。

这一点对所有特殊方法都适用。如果我们想为自定义对象使用 `x``in``myobj` 语法，我们可以实现 `__contains__`。如果我们想使用 `myobj[i]``=``value` 语法，我们提供 `__setitem__` 方法，而如果我们想使用 `something``=``myobj[i]`，我们实现 `__getitem__`。

列表类中有 33 个这样的特殊方法。我们可以使用 `dir` 函数查看所有这些方法，如下所示：

```py
>>> dir(list)

['__add__', '__class__', '__contains__', '__delattr__','__delitem__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'append', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort'  
```

此外，如果我们想了解这些方法中的任何一种的工作方式，我们可以使用 `help` 函数：

```py
>>> help(list.__add__)
Help on wrapper_descriptor:

__add__(self, value, /)
 Return self+value.  
```

列表上的 `+` 运算符将两个列表连接起来。我们没有足够的空间来讨论本书中所有可用的特殊函数，但你现在可以使用 `dir` 和 `help` 探索所有这些功能。官方在线 Python 参考 ([`docs.python.org/3/`](https://docs.python.org/3/)) 也有大量有用的信息。特别关注 `collections` 模块中讨论的抽象基类。

因此，回到我们之前讨论的关于何时使用组合而不是继承的问题：如果我们需要以某种方式更改类上的任何方法，包括特殊方法，我们绝对需要使用继承。如果我们使用组合，我们可以编写执行验证或更改的方法，并要求调用者使用这些方法，但没有任何阻止他们直接访问属性的方法。他们可以在我们的列表中插入一个没有五个字符的项目，这可能会使列表中的其他方法产生混淆。

通常，需要扩展内置数据类型的需求表明我们可能使用了错误类型的数据。这并不总是这种情况，但如果我们正在寻找扩展内置类型，我们应该仔细考虑是否不同的数据结构会更合适。

# 案例研究

为了将所有这些内容串联起来，我们将编写一个简单的链接收集器，它将访问一个网站并收集该网站上每个页面的每个链接。在我们开始之前，我们需要一些测试数据来工作。只需编写一些包含相互链接以及指向互联网上其他网站的链接的 HTML 文件即可，类似于以下内容：

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

将其中一个文件命名为`index.html`，以便在页面被提供时首先显示。确保其他文件存在，并使事情复杂化，以便它们之间有大量的链接。如果不想自己设置，本章的示例包括一个名为`case_study_serve`的目录（现有最糟糕的个人网站之一！）。

现在，通过进入包含所有这些文件的目录并运行以下命令来启动一个简单的 Web 服务器：

```py
$python3 -m http.server  
```

这将在端口 8000 上启动一个服务器；您可以通过在 Web 浏览器中访问`http://localhost:8000/`来查看您创建的页面。

目标是传递给我们的收集器网站的基准 URL（在这个例子中：`http://localhost:8000/`），并让它创建一个包含网站上每个唯一链接的列表。我们需要考虑三种类型的 URL（指向外部网站的链接，以`http://`开头，绝对内部链接，以`/`字符开头，以及相对链接，用于其他所有内容）。我们还需要意识到页面可能通过循环相互链接；我们需要确保我们不会多次处理同一页面，否则它可能永远不会结束。考虑到所有这些独特性，听起来我们可能需要一些集合。

在我们深入探讨之前，让我们从基础知识开始。以下是连接到页面并解析该页面中所有链接的代码：

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

这段代码很短，考虑到它所做的事情。它连接到命令行参数传递的服务器，下载页面，并提取该页面上所有的链接。`__init__` 方法使用 `urlparse` 函数从 URL 中提取主机名；因此，即使我们传递 `http://localhost:8000/some/page.html`，它仍然在 `http://localhost:8000/` 主机的高级别上操作。这很有意义，因为我们想要收集网站上所有的链接，尽管它假设每个页面都通过一系列链接与索引相连。

`collect_links` 方法连接到并从服务器下载指定的页面，并使用正则表达式在该页面上找到所有链接。正则表达式是一个非常强大的字符串处理工具。不幸的是，它们的学习曲线很陡峭；如果你以前没有使用过它们，我强烈建议你研究关于这个主题的许多完整书籍或网站。如果你认为它们不值得了解，试着在不使用它们的情况下编写前面的代码，你会改变主意的。

示例还在 `collect_links` 方法的中间停止以打印链接的值。这是我们编写程序时测试程序的一种常见方式：停止并输出值以确保它是我们期望的值。以下是它为我们示例输出的内容：

```py
['contact.html', 'blog.html', 'esme.html', '/hobbies.html', 
'/contact.html', 'http://www.archlinux.org/'] 
```

因此，现在我们收集了第一页上所有的链接。我们能用它做什么呢？我们不能简单地将链接放入一个集合中以去除重复项，因为链接可能是相对的或绝对的。例如，`contact.html` 和 `/contact.html` 都指向同一个页面。所以我们应该做的第一件事是将所有链接规范化为它们的完整 URL，包括主机名和相对路径。我们可以通过给我们的对象添加一个 `normalize_url` 方法来实现这一点：

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

此方法将每个 URL 转换为包含协议和主机名的完整地址。现在，两个联系页面具有相同的值，我们可以将它们存储在集合中。我们将不得不修改 `__init__` 以创建集合，并将 `collect_links` 修改为将所有链接放入其中。

然后，我们还需要访问所有非外部链接并将它们也收集起来。但是等等；如果我们这样做，当我们遇到相同的页面两次时，我们如何避免重复访问链接呢？看起来我们实际上需要两个集合：一个是收集到的链接集合，另一个是已访问的链接集合。这表明我们选择集合来表示数据是明智的；我们知道集合在处理多个集合时最有用。让我们按照以下方式设置它们：

```py
class LinkCollector: 
    def __init__(self, url): 
        self.url = "http://+" + urlparse(url).netloc 
        self.collected_links = set() 
        self.visited_links = set() 

    def collect_links(self, path="/"): 
        full_url = self.url + path 
        self.visited_links.add(full_url) 
        page = str(urlopen(full_url).read()) 
        links = LINK_REGEX.findall(page) 
        links = {self.normalize_url(path, link 
            ) for link in links} 
        self.collected_links = links.union( 
                self.collected_links) 
        unvisited_links = links.difference( 
                self.visited_links) 
        print(links, self.visited_links, 
                self.collected_links, unvisited_links) 
```

创建包含链接的标准化列表的行使用了一个`set`推导式（我们将在下一章详细讲解这些内容）。再次强调，该方法会停止打印当前值，以便我们可以验证我们没有混淆集合，并且确实调用了`difference`方法来收集`unvisited_links`。然后我们可以添加几行代码，遍历所有未访问的链接并将它们添加到集合中，如下所示：

```py
        for link in unvisited_links: 
            if link.startswith(self.url): 
                self.collect_links(urlparse(link).path) 
```

`if`语句确保我们只从单个网站收集链接；我们不希望离开并从互联网上的所有页面收集所有链接（除非我们是谷歌或互联网档案馆！）。如果我们修改程序底部的主体代码以输出收集到的链接，我们可以看到它似乎已经收集了它们，如下面的代码块所示：

```py
if __name__ == "__main__": 
    collector = LinkCollector(sys.argv[1]) 
    collector.collect_links() 
    for link in collector.collected_links: 
        print(link) 
```

它显示了所有收集到的链接，并且只显示一次，尽管在我的示例中许多页面相互链接多次，如下所示：

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

尽管它收集了指向外部页面的链接，但它并没有从我们链接到的任何外部页面收集链接。如果我们想收集网站上所有的链接，这是一个很棒的程序。但它并没有给我提供我可能需要构建网站地图的所有信息；它告诉我我有哪些页面，但它没有告诉我哪些页面链接到其他页面。如果我们想做到这一点，我们不得不做一些修改。

我们首先应该查看我们的数据结构。收集到的链接集合不再起作用；我们想知道哪些链接是从哪些页面链接过来的。我们可以将这个集合转换成每个访问的页面的集合字典。字典的键将代表当前集合中确切相同的数据。值将是该页面上所有链接的集合。以下是更改内容：

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
        self.collected_links = {} 
        self.visited_links = set() 

    def collect_links(self, path="/"): 
        full_url = self.url + path 
        self.visited_links.add(full_url) 
        page = str(urlopen(full_url).read()) 
        links = LINK_REGEX.findall(page) 
        links = {self.normalize_url(path, link 
            ) for link in links} 
        self.collected_links[full_url] = links 
        for link in links: 
            self.collected_links.setdefault(link, set()) 
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
    for link, item in collector.collected_links.items(): 
        print("{}: {}".format(link, item)) 
```

改变很少；原本创建两个集合并集的行已被替换为三条更新字典的行。第一条简单地告诉字典该页面的收集链接是什么。第二条使用`setdefault`为字典中尚未添加到字典中的任何项创建一个空集合。结果是包含所有链接作为键的字典，映射到所有内部链接的链接集合，以及外部链接的空集合。

最后，我们不再递归调用`collect_links`，而是可以使用队列来存储尚未处理的链接。这种实现不会支持并发，但这将是创建一个多线程版本的良好第一步，该版本可以并行发送多个请求以节省时间：

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

我不得不手动在`normalize_url`方法中删除任何尾随的正斜杠，以消除此代码版本中的重复项。

因为最终结果是未排序的字典，所以没有限制链接应该以什么顺序处理。因此，我们也可以同样容易地使用`LifoQueue`而不是`Queue`。在这种情况下，优先队列可能没有太多意义，因为没有明显的优先级可以附加到链接上。

# 练习

学习如何选择正确的数据结构的最佳方式是先错误地做几次（故意或意外地！）！取一些你最近写的代码，或者写一些使用列表的新代码。尝试使用不同的数据结构重写它。哪一些更有意义？哪一些没有？哪一些的代码最优雅？

用几对不同数据结构对的方法试一试。你可以查看之前章节练习中做的例子。有没有对象有方法，你可以使用数据类、`namedtuple`或`dict`来代替？尝试两种方法，看看。有没有字典可以变成集合，因为你实际上并没有访问值？你有没有检查重复的列表？一个集合是否足够？或者可能是几个集合？队列实现中的一个是否更高效？将 API 限制在栈顶而不是允许随机访问列表是否有用？

如果你想要一些具体的例子来操作，尝试调整链接收集器，使其也保存每个链接使用的标题。也许你可以生成一个 HTML 网站地图，列出网站上的所有页面，并包含一个指向其他页面的链接列表，这些链接使用相同的链接标题。

你最近是否编写过任何可以通过继承内置类型并重写一些特殊双下划线方法来改进的容器对象？你可能需要做一些研究（使用`dir`和`help`，或者 Python 库参考）来找出哪些方法需要重写。你确定继承是正确的工具吗？基于组合的解决方案可能更有效？在你决定之前，尝试两种方法（如果可能的话）。尝试找到不同的情况下，每种方法比另一种方法更好的情况。

如果你在这章开始之前就已经熟悉了各种 Python 数据结构和它们的用途，你可能感到无聊。但如果是这样的话，你很可能过度使用了数据结构！看看你的一些旧代码，并尝试将其重写为使用更多自定义类。仔细考虑替代方案，并尝试所有方案；哪一个能让你构建出最易读和可维护的系统？

总是批判性地评估你的代码和设计决策。养成回顾旧代码的习惯，并注意自从你编写它以来你对“良好设计”的理解是否发生了变化。软件设计有很大的美学成分，就像在画布上用油画的艺术家一样，我们都需要找到最适合我们的风格。

# 摘要

我们已经介绍了几个内置的数据结构，并尝试理解如何为特定的应用选择一个。有时，我们能做的最好的事情就是创建一个新的对象类，但通常，内置的其中一个就能提供我们所需的一切。当它不能满足需求时，我们总能使用继承或组合来适应我们的使用场景。我们甚至可以覆盖特殊方法来完全改变内置语法的行为。

在下一章中，我们将讨论如何整合 Python 的面向对象和非面向对象方面。在这个过程中，我们会发现 Python 的面向对象特性比第一眼看上去要丰富得多！
