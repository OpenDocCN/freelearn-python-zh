# 第七章：使用生成器

生成器是 Python 作为一种特殊语言的另一个特性。在本章中，我们将探讨它们的基本原理，它们为什么被引入到语言中以及它们解决的问题。我们还将介绍如何通过使用生成器来惯用地解决问题，以及如何使我们的生成器（或任何可迭代对象）符合 Python 的风格。

我们将了解为什么迭代（以迭代器模式的形式）在语言中得到了自动支持。从那里，我们将再次探索生成器如何成为 Python 的一个基本特性，以支持其他功能，如协程和异步编程。

本章的目标如下：

+   创建提高程序性能的生成器

+   研究迭代器（特别是迭代器模式）如何深度嵌入 Python

+   解决涉及迭代的问题

+   了解生成器作为协程和异步编程的基础是如何工作的

+   探索协程的语法支持——`yield from`、`await`和`async def`

# 技术要求

本章中的示例将适用于任何平台上的 Python 3.6 的任何版本。

本章中使用的代码可以在[`github.com/PacktPublishing/Clean-Code-in-Python`](https://github.com/PacktPublishing/Clean-Code-in-Python)找到

说明可在`README`文件中找到。

# 创建生成器

生成器在很久以前就被引入 Python 中（PEP-255），其目的是在 Python 中引入迭代的同时提高程序的性能（通过使用更少的内存）。

生成器的想法是创建一个可迭代的对象，当被迭代时，它将逐个产生它包含的元素。生成器的主要用途是节省内存——而不是在内存中拥有一个非常大的元素列表，一次性保存所有元素，我们有一个知道如何逐个产生每个特定元素的对象，只要它们被需要。

这个特性使得惰性计算或内存中的重量级对象成为可能，类似于其他函数式编程语言（例如 Haskell）提供的方式。甚至可以处理无限序列，因为生成器的惰性特性允许这样的选项。

# 首先看一下生成器

让我们从一个例子开始。现在手头的问题是，我们想处理一个大量的记录列表，并对它们进行一些指标和度量。给定一个包含有关购买信息的大型数据集，我们希望处理它以获得最低销售额、最高销售额和销售额的平均价格。

为了简化这个例子，我们将假设一个只有两个字段的 CSV，格式如下：

```py
<purchase_date>, <price>
...
```

我们将创建一个接收所有购买的对象，并且这将为我们提供必要的指标。我们可以通过简单地使用`min()`和`max()`内置函数来获得其中一些值，但这将需要多次迭代所有的购买，因此我们使用我们的自定义对象，它将在单次迭代中获取这些值。

将为我们获取数字的代码看起来相当简单。它只是一个具有一种方法的对象，该方法将一次性处理所有价格，并且在每一步中，将更新我们感兴趣的每个特定指标的值。首先，我们将在以下清单中显示第一个实现，然后在本章的后面（一旦我们更多地了解迭代），我们将重新访问这个实现，并获得一个更好（更紧凑）的版本。现在，我们暂时采用以下方式：

```py
class PurchasesStats:

    def __init__(self, purchases):
        self.purchases = iter(purchases)
        self.min_price: float = None
        self.max_price: float = None
        self._total_purchases_price: float = 0.0
        self._total_purchases = 0
        self._initialize()

    def _initialize(self):
        try:
            first_value = next(self.purchases)
        except StopIteration:
            raise ValueError("no values provided")

        self.min_price = self.max_price = first_value
        self._update_avg(first_value)

    def process(self):
        for purchase_value in self.purchases:
            self._update_min(purchase_value)
            self._update_max(purchase_value)
            self._update_avg(purchase_value)
        return self

    def _update_min(self, new_value: float):
        if new_value < self.min_price:
            self.min_price = new_value

    def _update_max(self, new_value: float):
        if new_value > self.max_price:
            self.max_price = new_value

    @property
    def avg_price(self):
        return self._total_purchases_price / self._total_purchases

    def _update_avg(self, new_value: float):
        self._total_purchases_price += new_value
        self._total_purchases += 1

    def __str__(self):
        return (
            f"{self.__class__.__name__}({self.min_price}, "
            f"{self.max_price}, {self.avg_price})"
        )
```

这个对象将接收`purchases`的所有总数并处理所需的值。现在，我们需要一个函数，将这些数字加载到这个对象可以处理的东西中。以下是第一个版本：

```py
def _load_purchases(filename):
    purchases = []
    with open(filename) as f:
        for line in f:
            *_, price_raw = line.partition(",")
            purchases.append(float(price_raw))

    return purchases
```

这段代码可以工作；它将文件中的所有数字加载到一个列表中，当传递给我们的自定义对象时，将产生我们想要的数字。但它有一个性能问题。如果你用一个相当大的数据集运行它，它将需要一段时间才能完成，如果数据集足够大以至于无法放入主内存中，甚至可能会失败。

如果我们看一下消耗这些数据的代码，它是逐个处理`purchases`的，所以我们可能会想知道为什么我们的生产者一次性将所有内容都放入内存。它创建了一个列表，将文件的所有内容都放入其中，但我们知道我们可以做得更好。

解决方案是创建一个生成器。我们不再将文件的整个内容加载到列表中，而是逐个产生结果。现在的代码看起来是这样的：

```py
def load_purchases(filename):
    with open(filename) as f:
        for line in f:
            *_, price_raw = line.partition(",")
            yield float(price_raw)
```

如果你这次测量这个过程，你会注意到内存的使用显著减少了。我们还可以看到代码看起来更简单——不需要定义列表（因此也不需要向其添加元素），`return`语句也消失了。

在这种情况下，`load_purchases`函数是一个生成器函数，或者简单地说是一个生成器。

在 Python 中，任何函数中存在`yield`关键字都会使其成为一个生成器，因此，当调用它时，除了创建一个生成器实例之外，什么都不会发生：

```py
>>> load_purchases("file")
<generator object load_purchases at 0x...>
```

生成器对象是可迭代的（我们稍后会更详细地讨论可迭代对象），这意味着它可以与`for`循环一起工作。请注意，我们在消费者代码上没有改变任何东西——我们的统计处理器保持不变，在新实现后`for`循环也没有修改。

使用可迭代对象使我们能够创建这些强大的抽象，这些抽象对`for`循环是多态的。只要我们保持可迭代接口，我们就可以透明地迭代该对象。

# 生成器表达式

生成器节省了大量内存，而且由于它们是迭代器，它们是其他需要更多内存空间的可迭代对象或容器的方便替代品，比如列表、元组或集合。

就像这些数据结构一样，它们也可以通过推导来定义，只是它被称为生成器表达式（关于它们是否应该被称为生成器推导有一个持续的争论。在本书中，我们将只用它们的规范名称来提及它们，但请随意使用你更喜欢的名称）。

同样，我们可以定义一个列表推导。如果我们用括号替换方括号，我们就得到了一个生成器，它是表达式的结果。生成器表达式也可以直接传递给那些与可迭代对象一起工作的函数，比如`sum()`和`max()`：

```py
>>> [x**2 for x in range(10)]
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

>>> (x**2 for x in range(10))
<generator object <genexpr> at 0x...>

>>> sum(x**2 for x in range(10))
285
```

总是传递一个生成器表达式，而不是列表推导，给那些期望可迭代对象的函数，比如`min()`、`max()`和`sum()`。这样更有效率和符合 Python 的风格。

# 迭代习语

在本节中，我们将首先探讨一些在 Python 中处理迭代时非常有用的习语。这些代码示例将帮助我们更好地了解我们可以用生成器做什么类型的事情（特别是在我们已经看过生成器表达式之后），以及如何解决与它们相关的典型问题。

一旦我们看过一些习语，我们将继续更深入地探讨 Python 中的迭代，分析使迭代成为可能的方法，以及可迭代对象的工作原理。

# 迭代的习语

我们已经熟悉了内置的`enumerate()`函数，它给定一个可迭代对象，将返回另一个可迭代对象，其中元素是一个元组，其第一个元素是第二个元素的枚举（对应于原始可迭代对象中的元素）：

```py
>>> list(enumerate("abcdef"))
[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'f')]
```

我们希望创建一个类似的对象，但以更低级的方式；一个可以简单地创建一个无限序列的对象。我们想要一个对象，可以从一个起始数字开始产生一个数字序列，没有任何限制。

一个简单的对象就可以解决问题。每次调用这个对象，我们都会得到序列的下一个数字*无穷*：

```py
class NumberSequence:

    def __init__(self, start=0):
        self.current = start

    def next(self):
        current = self.current
        self.current += 1
        return current
```

基于这个接口，我们必须通过显式调用它的`next()`方法来使用这个对象：

```py
>>> seq = NumberSequence()
>>> seq.next()
0
>>> seq.next()
1

>>> seq2 = NumberSequence(10)
>>> seq2.next()
10
>>> seq2.next()
11
```

但是，使用这段代码，我们无法像我们想要的那样重建`enumerate()`函数，因为它的接口不支持在常规的 Python `for`循环中进行迭代，这也意味着我们无法将其作为参数传递给期望迭代的函数。请注意以下代码的失败：

```py
>>> list(zip(NumberSequence(), "abcdef"))
Traceback (most recent call last):
 File "...", line 1, in <module>
TypeError: zip argument #1 must support iteration
```

问题在于`NumberSequence`不支持迭代。为了解决这个问题，我们必须通过实现魔术方法`__iter__()`使对象成为可迭代的。我们还改变了之前的`next()`方法，使用了魔术方法`__next__`，这使得对象成为了迭代器：

```py
class SequenceOfNumbers:

    def __init__(self, start=0):
        self.current = start

    def __next__(self):
        current = self.current
        self.current += 1
        return current

    def __iter__(self):
        return self
```

这有一个优点——不仅可以迭代元素，而且我们甚至不再需要`.next()`方法，因为有了`__next__()`，我们可以使用`next()`内置函数：

```py
>>> list(zip(SequenceOfNumbers(), "abcdef"))
[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'f')]
>>> seq = SequenceOfNumbers(100)
>>> next(seq)
100
>>> next(seq)
101
```

# next()函数

`next()`内置函数将使可迭代对象前进到它的下一个元素并返回它：

```py
>>> word = iter("hello")
>>> next(word)
'h'
>>> next(word)
'e'  # ...
```

如果迭代器没有更多的元素产生，就会引发`StopIteration`异常：

```py
>>> ...
>>> next(word)
'o'
>>> next(word)
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
StopIteration
>>>
```

这个异常表示迭代已经结束，没有更多的元素可以消耗了。

如果我们希望处理这种情况，除了捕获`StopIteration`异常，我们可以在第二个参数中为这个函数提供一个默认值。如果提供了这个值，它将成为`StopIteration`抛出时的返回值：

```py
>>> next(word, "default value")
'default value'
```

# 使用生成器

通过简单地使用生成器，可以显著简化上述代码。生成器对象是迭代器。这样，我们可以定义一个函数，根据需要`yield`值，而不是创建一个类：

```py
def sequence(start=0):
    while True:
        yield start
        start += 1
```

记住，根据我们的第一个定义，函数体中的`yield`关键字使其成为一个生成器。因为它是一个生成器，所以像这样创建一个无限循环是完全可以的，因为当调用这个生成器函数时，它将运行到下一个`yield`语句被执行之前的所有代码。它将产生它的值并在那里暂停：

```py
>>> seq = sequence(10)
>>> next(seq)
10
>>> next(seq)
11

>>> list(zip(sequence(), "abcdef"))
[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'f')]
```

# Itertools

使用可迭代对象的好处在于，代码与 Python 本身更好地融合在一起，因为迭代是语言的一个关键组成部分。除此之外，我们还可以充分利用`itertools`模块（ITER-01）。实际上，我们刚刚创建的`sequence()`生成器与`itertools.count()`非常相似。但是，我们还可以做更多的事情。

迭代器、生成器和 itertools 最好的一点是它们是可组合的对象，可以链接在一起。

例如，回到我们的第一个例子，处理`purchases`以获得一些指标，如果我们想做同样的事情，但只针对某个阈值以上的值怎么办？解决这个问题的天真方法是在迭代时放置条件：

```py
# ...
    def process(self):
        for purchase in self.purchases:
            if purchase > 1000.0:
                ...
```

这不仅不符合 Python 的风格，而且也很死板（死板是一个表明糟糕代码的特征）。它不能很好地处理变化。如果数字现在改变了怎么办？我们通过参数传递吗？如果我们需要多个怎么办？如果条件不同（比如小于），我们传递一个 lambda 吗？

这些问题不应该由这个对象来回答，它的唯一责任是计算一组以数字表示的购买流的明确定义的指标。当然，答案是否定的。将这样的改变是一个巨大的错误（再次强调，清晰的代码是灵活的，我们不希望通过将这个对象与外部因素耦合来使其变得死板）。这些要求必须在其他地方解决。

最好让这个对象独立于它的客户端。这个类的责任越少，对更多客户端来说就越有用，从而增加它被重用的机会。

我们不会改变这段代码，而是保持原样，并假设新数据根据该类的每个客户的要求进行了过滤。

例如，如果我们只想处理前 10 个购买金额超过 1,000 的购买，我们将执行以下操作：

```py
>>> from itertools import islice
>>> purchases = islice(filter(lambda p: p > 1000.0, purchases), 10)
>>> stats = PurchasesStats(purchases).process()  # ...
```

这种过滤方式不会对内存造成惩罚，因为它们都是生成器，评估总是延迟的。这使我们有能力像一次性过滤整个集合然后将其传递给对象一样思考，但实际上并没有将所有内容都适应到内存中。

# 通过迭代器简化代码

现在，我们将简要讨论一些可以通过迭代器和偶尔的`itertools`模块帮助改进的情况。在讨论每种情况及其提出的优化后，我们将用一个推论来结束每个观点。

# 重复迭代

现在我们已经更多地了解了迭代器，并介绍了`itertools`模块，我们可以向您展示本章的第一个示例（用于计算有关某些购买的统计信息）如何被大大简化：

```py
def process_purchases(purchases):
    min_, max_, avg = itertools.tee(purchases, 3)
    return min(min_), max(max_), median(avg)
```

在这个例子中，`itertools.tee`将原始可迭代对象分成三个新的可迭代对象。我们将使用每个对象进行不同类型的迭代，而无需重复三个不同的循环。

读者可以简单地验证，如果我们将可迭代对象作为`purchases`参数传递，这个对象只被遍历一次（感谢`itertools.tee`函数[参见参考资料]），这是我们的主要要求。还可以验证这个版本如何等价于我们的原始实现。在这种情况下，不需要手动引发`ValueError`，因为将空序列传递给`min()`函数将产生相同的效果。

如果您正在考虑在同一个对象上多次运行循环，请停下来思考一下`itertools.tee`是否有所帮助。

# 嵌套循环

在某些情况下，我们需要在多个维度上进行迭代，寻找一个值，嵌套循环是第一个想法。当找到值时，我们需要停止迭代，但`break`关键字并不完全起作用，因为我们需要从两个（或更多）`for`循环中逃离，而不仅仅是一个。

这个问题的解决方案是什么？一个信号逃脱的标志？不是。引发异常？不，这与标志相同，但更糟，因为我们知道异常不应该用于控制流逻辑。将代码移到一个更小的函数并返回它？接近，但不完全。

答案是，尽可能将迭代简化为单个`for`循环。

这是我们想要避免的代码类型：

```py
def search_nested_bad(array, desired_value):
    coords = None
    for i, row in enumerate(array):
        for j, cell in enumerate(row):
            if cell == desired_value:
                coords = (i, j)
                break

        if coords is not None:
            break

    if coords is None:
        raise ValueError(f"{desired_value} not found")

    logger.info("value %r found at [%i, %i]", desired_value, *coords)
    return coords
```

以下是一个简化版本，它不依赖于标志来表示终止，并且具有更简单、更紧凑的迭代结构：

```py
def _iterate_array2d(array2d):
    for i, row in enumerate(array2d):
        for j, cell in enumerate(row):
            yield (i, j), cell

def search_nested(array, desired_value):
    try:
        coord = next(
            coord
            for (coord, cell) in _iterate_array2d(array)
            if cell == desired_value
        )
    except StopIteration:
        raise ValueError("{desired_value} not found")

    logger.info("value %r found at [%i, %i]", desired_value, *coord)
    return coord
```

值得一提的是，创建的辅助生成器如何作为所需迭代的抽象。在这种情况下，我们只需要在两个维度上进行迭代，但如果我们需要更多，不同的对象可以处理这一点，而客户端无需知道。这就是迭代器设计模式的本质，在 Python 中是透明的，因为它自动支持迭代器对象，这是下一节讨论的主题。

尽量简化迭代，使用尽可能多的抽象，尽可能将循环展平。

# Python 中的迭代器模式

在这里，我们将从生成器中稍微偏离，更深入地了解 Python 中的迭代。生成器是可迭代对象的特殊情况，但 Python 中的迭代超越了生成器，能够创建良好的可迭代对象将使我们有机会创建更高效、更紧凑和更可读的代码。

在前面的代码清单中，我们一直在看一些可迭代对象的示例，这些对象也是迭代器，因为它们实现了`__iter__()`和`__next__()`魔术方法。虽然这在一般情况下是可以的，但并不严格要求它们总是必须实现这两个方法，这里我们将展示可迭代对象（实现`__iter__`）和迭代器（实现`__next__`）之间的细微差别。

我们还探讨了与迭代相关的其他主题，如序列和容器对象。

# 迭代的接口

可迭代对象是支持迭代的对象，从非常高的层次来看，这意味着我们可以在其上运行`for .. in ...`循环，并且不会出现任何问题。然而，可迭代并不意味着与迭代器相同。

一般来说，可迭代只是我们可以迭代的东西，并且它使用迭代器来实现。这意味着在`__iter__`魔术方法中，我们希望返回一个迭代器，即一个实现了`__next__()`方法的对象。

迭代器是一个只知道如何产生一系列值的对象，每次被已探索的内置`next()`函数调用时，它都会一次产生一个值。当迭代器没有被调用时，它只是被冻结，静静地坐着，直到再次为下一个值调用它。在这个意义上，生成器是迭代器。

| **Python 概念** | **魔术方法** | **注意事项** |
| --- | --- | --- |
| 可迭代对象 | `__iter__` | 它们与迭代器一起工作，构建迭代逻辑。这些对象可以在`for ... in ...:`循环中迭代 |
| 迭代器 | `__next__` | 定义逐个产生值的逻辑。`StopIteration`异常表示迭代结束。可以通过内置的`next()`函数逐个获取值。 |

在下面的代码中，我们将看到一个迭代器对象的示例，它不是可迭代的——它只支持一次调用其值。在这里，名称`sequence`只是指一系列连续的数字，并不是 Python 中的序列概念，我们稍后会探讨：

```py
class SequenceIterator:
    def __init__(self, start=0, step=1):
        self.current = start
        self.step = step

    def __next__(self):
        value = self.current
        self.current += self.step
        return value
```

请注意，我们可以逐个获取序列的值，但我们无法迭代这个对象（这是幸运的，否则将导致无限循环）：

```py
>>> si = SequenceIterator(1, 2)
>>> next(si)
1
>>> next(si)
3
>>> next(si)
5
>>> for _ in SequenceIterator(): pass
... 
Traceback (most recent call last):
 ...
TypeError: 'SequenceIterator' object is not iterable
```

错误消息很清楚，因为对象没有实现`__iter__()`。

仅仅为了说明的目的，我们可以将迭代分离到另一个对象中（同样，只需使对象分别实现`__iter__`和`__next__`即可，但这样做可以帮助澄清我们在这个解释中试图阐明的不同点）。

# 序列对象作为可迭代对象

正如我们刚刚看到的，如果一个对象实现了`__iter__()`魔术方法，这意味着它可以在`for`循环中使用。虽然这是一个很好的特性，但我们可以实现的迭代形式并不仅限于此。当我们编写`for`循环时，Python 会尝试查看我们使用的对象是否实现了`__iter__`，如果实现了，它将使用它来构建迭代，但如果没有，还有备用选项。

如果对象恰好是一个序列（意味着它实现了`__getitem__()`和`__len__()`魔术方法），它也可以被迭代。如果是这种情况，解释器将按顺序提供值，直到引发`IndexError`异常，这与前面提到的`StopIteration`类似，也表示迭代的结束。

为了说明这种行为，我们运行以下实验，展示了一个实现`map()`在一系列数字上的序列对象：

```py
# generators_iteration_2.py

class MappedRange:
    """Apply a transformation to a range of numbers."""

    def __init__(self, transformation, start, end):
        self._transformation = transformation
        self._wrapped = range(start, end)

    def __getitem__(self, index):
        value = self._wrapped.__getitem__(index)
        result = self._transformation(value)
        logger.info("Index %d: %s", index, result)
        return result

    def __len__(self):
        return len(self._wrapped)
```

请记住，这个示例只是为了说明这样一个对象可以用常规的`for`循环进行迭代。在`__getitem__`方法中放置了一个日志行，以探索在迭代对象时传递了哪些值，正如我们从以下测试中所看到的：

```py
>>> mr = MappedRange(abs, -10, 5)
>>> mr[0]
Index 0: 10
10
>>> mr[-1]
Index -1: 4
4
>>> list(mr)
Index 0: 10
Index 1: 9
Index 2: 8
Index 3: 7
Index 4: 6
Index 5: 5
Index 6: 4
Index 7: 3
Index 8: 2
Index 9: 1
Index 10: 0
Index 11: 1
Index 12: 2
Index 13: 3
Index 14: 4
[10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4]
```

需要注意的是，重要的是要强调，虽然了解这一点很有用，但它也是对象不实现`__iter__`时的后备机制，因此大多数时候我们会希望通过考虑创建适当的序列来使用这些方法，而不仅仅是我们想要迭代的对象。

在设计用于迭代的对象时，更倾向于一个适当的可迭代对象（带有`__iter__`），而不是一个偶然也可以被迭代的序列。

# 协程

正如我们已经知道的，生成器对象是可迭代的。它们实现了`__iter__()`和`__next__()`。这是由 Python 自动提供的，因此当我们创建一个生成器对象函数时，我们会得到一个可以通过`next()`函数进行迭代或推进的对象。

除了这个基本功能，它们还有更多的方法，以便它们可以作为协程（PEP-342）工作。在这里，我们将探讨生成器如何演变成协程，以支持异步编程的基础，然后在下一节中更详细地探讨 Python 的新特性和涵盖异步编程的语法。用于支持协程的(PEP-342)中添加的基本方法如下：

+   `.close()`

+   `.throw(ex_type[, ex_value[, ex_traceback]])`

+   `.send(value)`

# 生成器接口的方法

在本节中，我们将探讨上述每个方法的作用，工作原理以及预期的使用方式。通过理解如何使用这些方法，我们将能够使用简单的协程。

稍后，我们将探讨协程的更高级用法，以及如何委托给子生成器（协程）以重构代码，以及如何编排不同的协程。

# close()

调用这个方法时，生成器将接收`GeneratorExit`异常。如果没有处理，那么生成器将在不产生更多值的情况下结束，并且它的迭代将停止。

这个异常可以用来处理完成状态。一般来说，如果我们的协程进行某种资源管理，我们希望捕获这个异常并使用该控制块来释放协程持有的所有资源。一般来说，这类似于使用上下文管理器或将代码放在异常控制的`finally`块中，但专门处理这个异常使得它更加明确。

在下面的例子中，我们有一个协程，它利用一个持有数据库连接的数据库处理程序对象，并在其上运行查询，通过固定长度的页面流式传输数据（而不是一次性读取所有可用的数据）：

```py
def stream_db_records(db_handler):
    try:
        while True:
            yield db_handler.read_n_records(10)
    except GeneratorExit:
        db_handler.close()
```

在每次调用生成器时，它将返回从数据库处理程序获取的`10`行，但当我们决定明确完成迭代并调用`close()`时，我们还希望关闭与数据库的连接：

```py
>>> streamer = stream_db_records(DBHandler("testdb"))
>>> next(streamer)
[(0, 'row 0'), (1, 'row 1'), (2, 'row 2'), (3, 'row 3'), ...]
>>> next(streamer)
[(0, 'row 0'), (1, 'row 1'), (2, 'row 2'), (3, 'row 3'), ...]
>>> streamer.close()
INFO:...:closing connection to database 'testdb'
```

使用`close()`方法关闭生成器以在需要时执行收尾任务。

# throw(ex_type[, ex_value[, ex_traceback]])

这个方法将在生成器当前暂停的地方`throw`异常。如果生成器处理了发送的异常，那么特定的`except`子句中的代码将被调用，否则，异常将传播到调用者。

在这里，我们稍微修改了之前的例子，以展示当我们使用这个方法处理协程处理的异常和未处理的异常时的区别：

```py
class CustomException(Exception):
    pass

def stream_data(db_handler):
    while True:
        try:
            yield db_handler.read_n_records(10)
        except CustomException as e:
            logger.info("controlled error %r, continuing", e)
        except Exception as e:
            logger.info("unhandled error %r, stopping", e)
            db_handler.close()
            break
```

现在，接收`CustomException`已经成为控制流的一部分，如果出现这种情况，生成器将记录一条信息性消息（当然，我们可以根据每种情况的业务逻辑进行调整），然后继续执行下一个`yield`语句，这是协程从数据库读取并返回数据的地方。

这个特定的例子处理了所有异常，但如果最后一个块（`except Exception:`）不在那里，结果将是生成器在生成器暂停的地方被引发（再次是`yield`*），然后从那里传播到调用者：

```py
>>> streamer = stream_data(DBHandler("testdb"))
>>> next(streamer)
[(0, 'row 0'), (1, 'row 1'), (2, 'row 2'), (3, 'row 3'), (4, 'row 4'), ...]
>>> next(streamer)
[(0, 'row 0'), (1, 'row 1'), (2, 'row 2'), (3, 'row 3'), (4, 'row 4'), ...]
>>> streamer.throw(CustomException)
WARNING:controlled error CustomException(), continuing
[(0, 'row 0'), (1, 'row 1'), (2, 'row 2'), (3, 'row 3'), (4, 'row 4'), ...]
>>> streamer.throw(RuntimeError)
ERROR:unhandled error RuntimeError(), stopping
INFO:closing connection to database 'testdb'
Traceback (most recent call last):
 ...
StopIteration
```

当我们收到来自领域的异常时，生成器继续。然而，当它收到另一个意外的异常时，捕获了默认块，我们关闭了与数据库的连接并完成了迭代，这导致生成器停止。正如我们从引发的`StopIteration`中看到的，这个生成器不能进一步迭代。

# send(value)

在前面的例子中，我们创建了一个简单的生成器，从数据库中读取行，当我们希望完成它的迭代时，这个生成器释放了与数据库相关的资源。这是使用生成器提供的方法之一（close）的一个很好的例子，但我们还可以做更多的事情。

这样的生成器很明显是从数据库中读取了固定数量的行。

我们希望参数化那个数字（`10`），以便我们可以在不同的调用中更改它。不幸的是，`next()`函数不为我们提供这样的选项。但幸运的是，我们有`send()`：

```py
def stream_db_records(db_handler):
    retrieved_data = None
    previous_page_size = 10
    try:
        while True:
            page_size = yield retrieved_data
            if page_size is None:
                page_size = previous_page_size

            previous_page_size = page_size

            retrieved_data = db_handler.read_n_records(page_size)
    except GeneratorExit:
        db_handler.close()
```

我们现在的想法是，我们现在已经使协程能够通过`send()`方法从调用者那里接收值。这个方法实际上是区分生成器和协程的方法，因为当它被使用时，意味着`yield`关键字将出现在语句的右侧，并且它的返回值将被分配给其他东西。

在协程中，我们通常发现`yield`关键字以以下形式使用：

```py
receive = yield produced
```

在这种情况下，`yield`将做两件事。它将`produced`发送回调用者，调用者将在下一轮迭代（例如在调用`next()`之后）中接收到它，并在那里暂停。稍后，调用者将想要通过使用`send()`方法向协程发送一个值。这个值将成为`yield`语句的结果，在这种情况下赋给名为`receive`的变量。

只有当协程在`yield`语句处暂停等待某些东西产生时，向协程发送值才有效。为了实现这一点，协程必须被推进到这种状态。唯一的方法是通过调用`next()`来做到这一点。这意味着在向协程发送任何东西之前，必须通过`next()`方法至少推进一次。如果不这样做，将导致异常：

```py
>>> c = coro()
>>> c.send(1)
Traceback (most recent call last):
 ...
TypeError: can't send non-None value to a just-started generator
```

在向协程发送任何值之前，请记住通过调用`next()`来推进协程。

回到我们的例子。我们正在改变元素被生成或流式传输的方式，使其能够接收它期望从数据库中读取的记录的长度。

第一次调用`next()`时，生成器将前进到包含`yield`的行；它将向调用者提供一个值（如变量中设置的`None`），并在那里暂停。从这里，我们有两个选择。如果我们选择通过调用`next()`来推进生成器，将使用`10`的默认值，并且它将像往常一样继续进行。这是因为`next()`在技术上与`send(None)`相同，但这是在我们之前设置的值的`if`语句中处理的。

另一方面，如果我们决定通过`send(<value>)`提供一个显式值，这个值将成为`yield`语句的结果，这将被赋给包含要使用的页面长度的变量，而这个变量将被用来从数据库中读取。

后续的调用将具有这种逻辑，但重要的是现在我们可以在迭代中间动态改变要读取的数据的长度。

现在我们了解了之前的代码是如何工作的，大多数 Pythonistas 都希望有一个简化版本（毕竟，Python 也是关于简洁和干净紧凑的代码）：

```py
def stream_db_records(db_handler):
    retrieved_data = None
    page_size = 10
    try:
        while True:
            page_size = (yield retrieved_data) or page_size
            retrieved_data = db_handler.read_n_records(page_size)
    except GeneratorExit:
        db_handler.close()
```

这个版本不仅更紧凑，而且更好地说明了这个想法。`yield`周围的括号使它更清晰，表明它是一个语句（把它想象成一个函数调用），我们正在使用它的结果与先前的值进行比较。

这符合我们的预期，但我们总是要记住在向其发送任何数据之前先推进协程。如果我们忘记调用第一个`next()`，我们将得到一个`TypeError`。这个调用可以被忽略，因为它不会返回我们将使用的任何东西。

如果我们能够直接使用协程，在创建后不必记住每次使用它时都调用`next()`第一次，那将是很好的。一些作者（PYCOOK）设计了一个有趣的装饰器来实现这一点。这个装饰器的想法是推进协程，所以下面的定义可以自动工作：

```py
@prepare_coroutine
def stream_db_records(db_handler):
    retrieved_data = None
    page_size = 10
    try:
        while True:
            page_size = (yield retrieved_data) or page_size
            retrieved_data = db_handler.read_n_records(page_size)
    except GeneratorExit:
        db_handler.close()

>>> streamer = stream_db_records(DBHandler("testdb"))
>>> len(streamer.send(5))
5
```

让我们举个例子，我们创建了`prepare_coroutine()`装饰器。

# 更高级的协程

到目前为止，我们对协程有了更好的理解，并且能够创建简单的协程来处理小任务。我们可以说这些协程实际上只是更高级的生成器（这是正确的，协程只是花哨的生成器），但是，如果我们真的想要开始支持更复杂的场景，通常我们必须采用一种处理许多协程并发的设计，并且需要更多的功能。

处理许多协程时，我们发现了新的问题。随着应用程序的控制流变得更加复杂，我们希望能够在堆栈上传递值和异常，能够捕获我们可能在任何级别调用的子协程的值，并最终安排多个协程朝着共同的目标运行。

为了简化事情，生成器必须再次扩展。这就是 PEP-380 所解决的问题——通过改变生成器的语义，使其能够返回值，并引入新的`yield from`构造。

# 在协程中返回值

正如本章开头介绍的那样，迭代是一种机制，它在可迭代对象上多次调用`next()`，直到引发`StopIteration`异常。

到目前为止，我们一直在探索生成器的迭代性质——我们一次产生一个值，并且通常只关心`for`循环的每一步产生的每个值。这是一种非常逻辑的生成器思维方式，但是协程有一个不同的想法；尽管它们在技术上是生成器，但它们并不是以迭代为目的而构思的，而是以挂起代码的执行直到稍后恢复为目标。

这是一个有趣的挑战；当我们设计一个协程时，我们通常更关心挂起状态而不是迭代（迭代协程将是一个奇怪的情况）。挑战在于很容易混合它们两者。这是因为技术实现细节；Python 中对协程的支持是建立在生成器之上的。

如果我们想要使用协程来处理一些信息并挂起其执行，那么把它们看作轻量级线程（或者在其他平台上称为绿色线程）是有意义的。在这种情况下，如果它们能够返回值，就像调用任何其他常规函数一样，那将是有意义的。

但让我们记住生成器不是常规函数，因此在生成器中，构造`value = generator()`除了创建一个`generator`对象之外什么也不会做。那么使生成器返回一个值的语义是什么？它将必须在迭代完成后。

当生成器返回一个值时，迭代立即停止（不能再迭代）。为了保持语义，`StopIteration`异常仍然被引发，并且要返回的值存储在`exception`对象中。捕获它是调用者的责任。

在下面的例子中，我们创建了一个简单的`generator`，产生两个值，然后返回第三个值。请注意，我们必须捕获异常以获取这个`value`，以及它如何精确地存储在异常的属性`value`下：

```py
>>> def generator():
...     yield 1
...     yield 2
...     return 3
... 
>>> value = generator()
>>> next(value)
1
>>> next(value)
2
>>> try:
...     next(value)
... except StopIteration as e:
...     print(">>>>>> returned value ", e.value)
... 
>>>>>> returned value  3
```

# 委托到更小的协程 - yield from 语法

以前的特性很有趣，因为它为协程（生成器）打开了许多新的可能性，现在它们可以返回值。但是，这个特性本身如果没有适当的语法支持，就不会那么有用，因为以这种方式捕获返回值有点麻烦。

这是`yield from`语法的主要特性之一。除了其他事情（我们将详细审查），它可以收集子生成器返回的值。记住我们说过在生成器中返回数据很好，但不幸的是，编写语句`value = generator()`是行不通的吗？好吧，将其编写为`value = yield from generator()`就可以了。

# `yield from`的最简单用法

在其最基本的形式中，新的`yield from`语法可以用于将嵌套的`for`循环中的生成器链接成一个单一的生成器，最终将得到一个连续流中所有值的单个字符串。

典型的例子是创建一个类似于`standard`库中的`itertools.chain()`的函数。这是一个非常好的函数，因为它允许您传递任意数量的`iterables`，并将它们一起返回一个流。

天真的实现可能看起来像这样：

```py
def chain(*iterables):
    for it in iterables:
        for value in it:
            yield value
```

它接收可变数量的`iterables`，遍历所有这些 iterables，由于每个值都是`iterable`，它支持`for... in..`结构，因此我们有另一个`for`循环来获取每个特定 iterable 中的每个值，这是由调用函数产生的。这在多种情况下可能会有所帮助，例如将生成器链接在一起或尝试迭代通常不可能一次比较的东西（例如列表与元组等）。

然而，`yield from`语法允许我们更进一步，避免嵌套循环，因为它能够直接从子生成器产生值。在这种情况下，我们可以简化代码如下：

```py
def chain(*iterables):
    for it in iterables:
        yield from it
```

请注意，对于这两种实现，生成器的行为完全相同：

```py
>>> list(chain("hello", ["world"], ("tuple", " of ", "values.")))
['h', 'e', 'l', 'l', 'o', 'world', 'tuple', ' of ', 'values.']
```

这意味着我们可以在任何其他可迭代对象上使用`yield from`，它将起到作用，就好像顶层生成器（使用`yield from`的那个）自己生成这些值一样。

这适用于任何可迭代对象，甚至生成器表达式也不例外。现在我们熟悉了它的语法，让我们看看如何编写一个简单的生成器函数，它将产生一个数字的所有幂（例如，如果提供`all_powers(2, 3)`，它将产生`2⁰, 2¹,... 2³`）：

```py
def all_powers(n, pow):
    yield from (n ** i for i in range(pow + 1))
```

虽然这样简化了语法，节省了一个`for`语句的行数并不是一个很大的优势，这并不能证明向语言中添加这样的更改是合理的。

实际上，这实际上只是一个副作用，而`yield from`结构的真正存在意义是我们将在接下来的两个部分中探讨的。

# 捕获子生成器返回的值

在下面的例子中，我们有一个生成器调用另外两个嵌套的生成器，按顺序产生值。每一个嵌套的生成器都返回一个值，我们将看到顶层生成器如何能够有效地捕获返回值，因为它通过`yield from`调用内部生成器：

```py
def sequence(name, start, end):
    logger.info("%s started at %i", name, start)
    yield from range(start, end)
    logger.info("%s finished at %i", name, end)
    return end

def main():
    step1 = yield from sequence("first", 0, 5)
    step2 = yield from sequence("second", step1, 10)
    return step1 + step2
```

这是在主函数中迭代时代码的可能执行方式：

```py
>>> g = main()
>>> next(g)
INFO:generators_yieldfrom_2:first started at 0
0
>>> next(g)
1
>>> next(g)
2
>>> next(g)
3
>>> next(g)
4
>>> next(g)
INFO:generators_yieldfrom_2:first finished at 5
INFO:generators_yieldfrom_2:second started at 5
5
>>> next(g)
6
>>> next(g)
7
>>> next(g)
8
>>> next(g)
9
>>> next(g)
INFO:generators_yieldfrom_2:second finished at 10
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
StopIteration: 15
```

main 的第一行委托给内部生成器，并产生值，直接从中提取。这并不是什么新鲜事，因为我们已经见过了。请注意，`sequence()`生成器函数返回结束值，在第一行赋给名为`step1`的变量，这个值在下一个生成器实例的开始正确使用。

最后，这个其他生成器也返回第二个结束值（`10`），而主生成器则返回它们的和（`5+10=15`），这是我们在迭代停止后看到的值。

我们可以使用`yield from`在协程完成处理后捕获最后一个值。

# 向子生成器发送和接收数据

现在，我们将看到`yield from`语法的另一个很好的特性，这可能是它赋予它完整力量的原因。正如我们在探索生成器作为协程时已经介绍的，我们知道我们可以向它们发送值和抛出异常，在这种情况下，协程要么接收值进行内部处理，要么必须相应地处理异常。

如果现在我们有一个委托给其他协程的协程（就像在前面的例子中），我们也希望保留这个逻辑。手动这样做将会相当复杂（如果我们没有通过`yield from`自动处理的话，可以看一下 PEP-380 中描述的代码）。

为了说明这一点，让我们保持相同的顶层生成器（main）与之前的例子相同，但让我们修改内部生成器，使它们能够接收值并处理异常。这段代码可能不是惯用的，只是为了展示这个机制是如何工作的。

```py
def sequence(name, start, end):
    value = start
    logger.info("%s started at %i", name, value)
    while value < end:
        try:
            received = yield value
            logger.info("%s received %r", name, received)
            value += 1
        except CustomException as e:
            logger.info("%s is handling %s", name, e)
            received = yield "OK"
    return end
```

现在，我们将调用`main`协程，不仅通过迭代它，还通过向它传递值和抛出异常，以查看它们在`sequence`内部是如何处理的：

```py
>>> g = main()
>>> next(g)
INFO: first started at 0
0
>>> next(g)
INFO: first received None
1
>>> g.send("value for 1")
INFO: first received 'value for 1'
2
>>> g.throw(CustomException("controlled error"))
INFO: first is handling controlled error
'OK'
... # advance more times
INFO:second started at 5
5
>>> g.throw(CustomException("exception at second generator"))
INFO: second is handling exception at second generator
'OK'
```

这个例子向我们展示了很多不同的东西。请注意，我们从未向`sequence`发送值，而只是向`main`发送，即使如此，接收这些值的代码是嵌套生成器。即使我们从未明确向`sequence`发送任何东西，它也会接收数据，因为它是通过`yield from`传递的。

`main`协程在内部调用另外两个协程，产生它们的值，并且在任何一个特定时间点被挂起。当它停在第一个时，我们可以看到日志告诉我们，正是这个协程实例接收了我们发送的值。当我们向它抛出异常时也是一样的。当第一个协程完成时，它返回了在名为`step1`的变量中分配的值，并作为第二个协程的输入，第二个协程也会做同样的事情（它将相应地处理`send()`和`throw()`调用）。

对于每个协程产生的值也是如此。当我们处于任何给定步骤时，调用`send()`返回的值对应于子协程（`main`当前挂起的那个）产生的值。当我们抛出一个正在处理的异常时，`sequence`协程产生值`OK`，这个值传播到被调用的（`main`），然后最终到达 main 的调用者。

# 异步编程

通过我们迄今为止看到的构造，我们能够在 Python 中创建异步程序。这意味着我们可以创建具有许多协程的程序，安排它们按特定顺序工作，并在每个协程上调用`yield from`后挂起时在它们之间切换。

我们可以从中获得的主要优势是以非阻塞的方式并行化 I/O 操作的可能性。我们需要的是一个低级生成器（通常由第三方库实现），它知道如何在协程被挂起时处理实际的 I/O。协程的目的是实现挂起，以便我们的程序可以在此期间处理另一个任务。应用程序重新获取控制的方式是通过`yield from`语句，它将挂起并向调用者产生一个值（就像我们之前看到的例子中使用这个语法来改变程序的控制流）。

这大致是多年来 Python 中异步编程的工作方式，直到决定需要更好的语法支持。

协程和生成器在技术上是相同的，这导致了一些混淆。从语法上（和技术上）来看，它们是相同的，但从语义上来看，它们是不同的。当我们想要实现高效的迭代时，我们创建生成器。我们通常创建协程的目的是运行非阻塞 I/O 操作。

尽管这种差异是明显的，Python 的动态特性仍然允许开发人员混合这些不同类型的对象，在程序的非常后期出现运行时错误。记住，在最简单和最基本的`yield from`语法中，我们使用这个结构来迭代（我们创建了一种在字符串、列表等上应用的`chain`函数）。这些对象都不是协程，但它仍然有效。然后，我们看到我们可以有多个协程，使用`yield from`发送值（或异常），并获得一些结果。这显然是两种非常不同的用例，但是，如果我们写出类似以下语句的内容：

```py
result = yield from iterable_or_awaitable()
```

不清楚`iterable_or_awaitable`返回什么。它可以是一个简单的可迭代对象，比如字符串，这可能仍然是语法上正确的。或者，它可能是一个实际的协程。这个错误的代价将在以后付出。

因此，Python 中的输入系统必须得到扩展。在 Python 3.5 之前，协程只是应用了`@coroutine`装饰器的生成器，并且它们需要使用`yield from`语法进行调用。现在，有一种特定类型的对象，即协程。

这个改变也带来了语法的改变。引入了`await`和`async def`语法。前者旨在替代`yield from`，它只能与`awaitable`对象一起使用（方便地，协程恰好是这种对象）。尝试使用不符合`awaitable`接口的东西来调用`await`将引发异常。`async def`是定义协程的新方法，取代了前面提到的装饰器，实际上创建了一个对象，当调用时，将返回一个协程的实例。

不去深入讨论 Python 中异步编程的所有细节和可能性，我们可以说，尽管有新的语法和新的类型，但这并没有从本质上做任何不同于我们在本章中介绍的概念。

在 Python 中异步编程的思想是有一个事件循环（通常是`asyncio`，因为它是`标准`库中包含的一个，但还有许多其他可以正常工作的），它管理一系列协程。这些协程属于事件循环，事件循环将根据其调度机制来调用它们。当这些协程中的每一个运行时，它将调用我们的代码（根据我们在编写的协程中定义的逻辑），当我们想要将控制返回给事件循环时，我们调用`await <coroutine>`，这将异步处理一个任务。事件循环将恢复，另一个协程将代替正在运行的操作。

实际上，还有更多的细节和边缘情况超出了本书的范围。然而，值得一提的是，这些概念与本章介绍的思想相关，并且这个领域是生成器证明是语言的核心概念的另一个地方，因为有许多东西是在它们的基础上构建的。

# 总结

生成器在 Python 中随处可见。自它们在 Python 中诞生以来，它们被证明是一个很好的补充，使程序更加高效，迭代更加简单。

随着时间的推移，以及需要向 Python 添加更复杂的任务，生成器再次帮助支持协程。

而在 Python 中，协程是生成器，我们仍然不必忘记它们在语义上是不同的。生成器是为了迭代而创建的，而协程的目标是异步编程（在任何给定时间暂停和恢复程序的执行部分）。这种区别变得如此重要，以至于它使 Python 的语法（和类型系统）发生了演变。

迭代和异步编程构成了 Python 编程的最后一根主要支柱。现在，是时候看看所有这些概念如何结合在一起，并将我们在过去几章中探讨的所有这些概念付诸实践了。

接下来的章节将描述 Python 项目的其他基本方面，如测试、设计模式和架构。

# 参考资料

以下是您可以参考的信息列表：

+   *PEP-234*：迭代器（[`www.python.org/dev/peps/pep-0234/`](https://www.python.org/dev/peps/pep-0234/)）

+   *PEP-255*：简单生成器（[`www.python.org/dev/peps/pep-0255/`](https://www.python.org/dev/peps/pep-0255/)）

+   *ITER-01*：Python 的 itertools 模块（[`docs.python.org/3/library/itertools.html`](https://docs.python.org/3/library/itertools.html)）

+   *GoF*：由 Erich Gamma, Richard Helm, Ralph Johnson, John Vlissides 撰写的书籍*Design Patterns: Elements of Reusable Object-Oriented Software*

+   *PEP-342*：通过增强生成器实现协程（[`www.python.org/dev/peps/pep-0342/`](https://www.python.org/dev/peps/pep-0342/)）

+   *PYCOOK*：由 Brian Jones, David Beazley 撰写的书籍*Python Cookbook: Recipes for Mastering Python 3, Third Edition*

+   *PY99*：虚拟线程（生成器、协程和续延）（[`mail.python.org/pipermail/python-dev/1999-July/000467.html`](https://mail.python.org/pipermail/python-dev/1999-July/000467.html)）

+   *CORO-01*：协程（[`wiki.c2.com/?CoRoutine`](http://wiki.c2.com/?CoRoutine)）

+   *CORO-02*：生成器不是协程（[`wiki.c2.com/?GeneratorsAreNotCoroutines`](http://wiki.c2.com/?GeneratorsAreNotCoroutines)）

+   *TEE*：`itertools.tee` 函数（[`docs.python.org/3/library/itertools.html#itertools.tee`](https://docs.python.org/3/library/itertools.html#itertools.tee)）
