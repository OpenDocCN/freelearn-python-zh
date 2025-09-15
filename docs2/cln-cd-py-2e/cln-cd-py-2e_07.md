# 7

# 生成器、迭代器和异步编程

生成器是 Python 区别于更传统语言的特征之一。在本章中，我们将探讨其原理，为什么它们被引入到语言中，以及它们解决的问题。我们还将介绍如何通过使用生成器以惯用的方式解决问题，以及如何使我们的生成器（或任何可迭代对象）具有 Python 风格。

我们将理解为什么迭代（以迭代器模式的形式）在语言中得到自动支持。从那里，我们将再次踏上旅程，探索生成器如何成为 Python 的一个基本特性，以支持其他功能，如协程和异步编程。

本章的目标如下：

+   创建能够提高我们程序性能的生成器

+   研究迭代器（特别是迭代器模式）在 Python 中如何深入嵌入

+   为了以惯用的方式解决涉及迭代的问题

+   理解生成器作为协程和异步编程基础的工作原理

+   探索对协程的语法支持——`yield from`、`await`和`async def`

精通生成器将大大提高你编写惯用 Python 代码的能力，因此它们对于本书的重要性不言而喻。在本章中，我们不仅研究如何使用生成器，还探索其内部机制，以便深入理解它们是如何工作的。

# 技术要求

本章中的示例将适用于任何平台上的 Python 3.9 的任何版本。

本章中使用的代码可以在[`github.com/PacktPublishing/Clean-Code-in-Python-Second-Edition`](https://github.com/PacktPublishing/Clean-Code-in-Python-Second-Edition)找到。说明文档在`README`文件中。

# 创建生成器

生成器在 Python 中引入已久（PEP-255），其想法是在 Python 中引入迭代的同时，通过使用更少的内存来提高程序的性能。

生成器的想法是创建一个可迭代的对象，在迭代过程中，将逐个产生它包含的元素。生成器的主要用途是节省内存——而不是在内存中保留一个非常大的元素列表，一次性持有所有元素，我们有一个知道如何逐个产生每个特定元素的对象，正如它被需要时。

此功能使内存中重载对象的延迟计算成为可能，类似于其他函数式编程语言（例如 Haskell）提供的方式。甚至可以处理无限序列，因为生成器的延迟特性使得这种选项成为可能。

## 初识生成器

让我们从例子开始。现在的问题是我们想要处理大量记录并获取一些关于它们的指标和指标。给定一个包含购买信息的庞大数据集，我们想要处理它以获取最低销售额、最高销售额和平均销售价格。

为了简化这个例子，我们将假设一个只有两个字段的 CSV 文件，其格式如下：

```py
<purchase_date>, <price>
... 
```

我们将创建一个接收所有购买的实例，这将给我们必要的指标。我们可以通过简单地使用内置函数 `min()` 和 `max()` 来获得一些这些值，但这将需要多次迭代所有购买，所以相反，我们使用我们的自定义对象，它将在单次迭代中获取这些值。

获取我们所需数字的代码看起来相当简单。它只是一个具有一个方法的对象，该方法一次处理所有价格，并在每个步骤中更新我们感兴趣的每个特定指标的值。首先，我们将展示以下列表中的第一个实现，稍后在本章中（一旦我们了解了更多关于迭代的内容），我们将重新审视这个实现，并得到一个更好（更紧凑）的版本。现在，我们暂时采用以下内容：

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

这个对象将接收所有关于“购买”的总数，并处理所需值。现在，我们需要一个函数将这些数字加载到这个对象可以处理的东西中。以下是第一个版本：

```py
def _load_purchases(filename):
    purchases = []
    with open(filename) as f:
        for line in f:
            *_, price_raw = line.partition(",")
            purchases.append(float(price_raw))
    return purchases 
```

这段代码是有效的；它将文件中的所有数字加载到一个列表中，当传递给我们的自定义对象时，将产生我们想要的数字。尽管如此，它有一个性能问题。如果你用相当大的数据集运行它，它将需要一段时间才能完成，如果数据集足够大以至于无法放入主内存，它甚至可能会失败。

如果我们看一下消费这些数据的代码，它一次处理一个购买，所以我们可能会想知道为什么我们的生产者一次将所有内容放入内存。它正在创建一个列表，将文件的所有内容都放入其中，但我们知道我们可以做得更好。

解决方案是创建一个生成器。而不是将整个文件内容加载到一个列表中，我们将一次产生一个结果。现在的代码将看起来像这样：

```py
def load_purchases(filename):
    with open(filename) as f:
        for line in f:
            *_, price_raw = line.partition(",")
            yield float(price_raw) 
```

如果你这次测量过程，你会注意到内存使用量显著下降。我们还可以看到代码看起来更简单——不需要定义列表（因此，不需要向其中添加内容），`return` 语句也消失了。

在这种情况下，`load_purchases` 函数是一个生成器函数，或者简单地说，是一个生成器。

在 Python 中，任何函数中关键字 `yield` 的存在都使其成为一个生成器，因此，在调用它时，除了创建生成器实例之外，不会发生任何事情：

```py
>>> load_purchases("file")
<generator object load_purchases at 0x...> 
```

生成器对象是一个可迭代对象（我们稍后会更详细地回顾可迭代对象），这意味着它可以与`for`循环一起工作。注意我们并没有在消费者代码上做任何改变——我们的统计处理器保持不变，`for`循环在新的实现后也没有被修改。

使用可迭代对象允许我们创建这些类型的强大抽象，它们在`for`循环方面是多态的。只要我们保持迭代器接口，我们就可以透明地遍历该对象。

我们在本章中探讨的是另一种与 Python 本身很好地融合的惯用代码案例。在之前的章节中，我们看到了如何实现我们自己的上下文管理器来将我们的对象连接到 with 语句中，或者如何创建自定义容器对象来利用`in`运算符，或者布尔值用于`if`语句，等等。现在轮到`for`运算符了，为此，我们将创建迭代器。

在深入探讨生成器的细节和细微差别之前，我们可以快速看一下生成器与我们已经看到的概念之间的关系：理解。以理解形式存在的生成器被称为生成器表达式，我们将在下一节简要讨论。

## 生成器表达式

生成器可以节省大量内存，并且由于它们是迭代器，它们是其他需要更多内存的迭代器或容器的方便替代品，例如列表、元组或集合。

与这些数据结构类似，它们也可以通过理解来定义，只是它们被称为生成器表达式（关于它们是否应该被称为生成器理解表达式，目前存在争议。在这本书中，我们将只按其标准名称来称呼它们，但你可以自由选择你喜欢的名称）。

同样，我们也可以定义列表理解。如果我们用圆括号替换方括号，我们就会得到一个由表达式生成的生成器。生成器表达式也可以直接传递给处理可迭代对象的函数，例如`sum()`和`max()`：

```py
>>> [x**2 for x in range(10)]
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
>>> (x**2 for x in range(10))
<generator object <genexpr> at 0x...>
>>> sum(x**2 for x in range(10))
285 
```

总是传递生成器表达式，而不是列表理解，给期望可迭代对象的函数，如`min()`、`max()`和`sum()`。这更高效，也更符合 Python 风格。

之前的推荐意味着尽量避免将列表传递给已经可以与生成器一起工作的函数。下面代码中的例子是你想要避免的，而应该优先考虑之前列表中的方法：

```py
>>> sum([x**2 for x in range(10)])  # here the list can be avoided 
```

当然，你还可以将生成器表达式赋值给变量，并在其他地方使用它（就像理解一样）。请注意，在这种情况下有一个重要的区别，因为我们在这里讨论的是生成器。列表可以被重复使用和迭代多次，但生成器在迭代过后就会耗尽。因此，请确保表达式的结果只被消费一次，否则你会得到意外的结果。

记住，生成器在迭代后就会耗尽，因为它们不会在内存中保存所有数据。

一种常见的方法是在代码中创建新的生成器表达式。这样，第一个在迭代后会耗尽，但随后会创建一个新的。以这种方式链式生成器表达式是有用的，并且有助于节省内存，同时使代码更具表现力，因为它在不同的步骤中解决不同的迭代。这种用法的一个场景是当你需要对可迭代对象应用多个过滤器时；你可以通过使用多个作为链式过滤器的生成器表达式来实现这一点。

现在我们工具箱中有了新的工具（迭代器），让我们看看我们如何使用它来编写更符合习惯的代码。

# 符合习惯的迭代

在本节中，我们将首先探索一些在处理 Python 中的迭代时非常有用的习语。这些代码配方将帮助我们更好地了解我们可以使用生成器（尤其是在我们已经看到生成器表达式之后）做什么，以及如何解决与它们相关的典型问题。

一旦我们看到了一些习语，我们将进一步探索 Python 中的迭代，分析使迭代成为可能的方法，以及可迭代对象是如何工作的。

## 迭代习语

我们已经熟悉了内置的`enumerate()`函数，给定一个可迭代对象，它将返回另一个对象，其元素是一个元组，第一个元素是第二个元素的索引（对应于原始可迭代对象中的元素）：

```py
>>> list(enumerate("abcdef"))
[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'f')] 
```

我们希望创建一个类似的对象，但以更低级的模式；一个可以简单地创建无限序列的对象。我们希望有一个可以产生从起始数字开始的序列的对象，没有任何限制。

如下简单的一个对象就可以做到这一点。每次我们调用这个对象时，我们都会得到序列中的下一个数字，无限循环：

```py
class NumberSequence:
    def __init__(self, start=0):
        self.current = start
    def next(self):
        current = self.current
        self.current += 1
        return current 
```

根据这个接口，我们必须通过显式调用其`next()`方法来使用这个对象：

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

但使用这段代码，我们无法像期望的那样重构`enumerate()`函数，因为它的接口不支持在常规 Python `for`循环中迭代，这也意味着我们无法将其作为参数传递给期望迭代对象的函数。注意以下代码是如何失败的：

```py
>>> list(zip(NumberSequence(), "abcdef"))
Traceback (most recent call last):
  File "...", line 1, in <module>
TypeError: zip argument #1 must support iteration 
```

问题在于`NumberSequence`不支持迭代。为了解决这个问题，我们必须通过实现魔法方法`__iter__()`使对象成为一个可迭代对象。我们还改变了之前的`next()`方法，通过使用`__next__`魔法方法，使对象成为一个迭代器：

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

这有一个优点——我们不仅可以迭代元素，而且我们甚至不再需要`.next()`方法，因为`__next__()`允许我们使用内置的`next()`函数：

```py
>>> list(zip(SequenceOfNumbers(), "abcdef"))
[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'f')]
>>> seq = SequenceOfNumbers(100)
>>> next(seq)
100
>>> next(seq)
101 
```

这利用了迭代协议。类似于我们在前几章中探索的上下文管理器协议，该协议由`__enter__`和`__exit__`方法组成，这个协议依赖于`__iter__`和`__next__`方法。

在 Python 中拥有这些协议有优势：所有了解 Python 的人都会熟悉这个接口，因此存在一种“标准合同”。这意味着，我们不需要定义自己的方法并与团队（或任何潜在的代码阅读者）达成一致（就像我们在第一个例子中的自定义`next()`方法那样）；Python 已经提供了一个接口和协议。我们只需要正确实现它。

### `next()`函数

`next()`内置函数将迭代器推进到其下一个元素并返回它：

```py
>>> word = iter("hello")
>>> next(word)
'h'
>>> next(word)
'e'  # ... 
```

如果迭代器没有更多元素可以产生，将引发`StopIteration`异常：

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

这个异常表示迭代已经结束，没有更多的元素可以消费。

如果我们希望处理这种情况，除了捕获`StopIteration`异常之外，我们还可以在函数的第二个参数中提供一个默认值。如果提供了这个值，它将作为替代`StopIteration`抛出的返回值：

```py
>>> next(word, "default value")
'default value' 
```

大多数情况下，建议使用默认值，以避免在程序运行时出现异常。如果我们绝对确信我们正在处理的迭代器不能为空，仍然最好对此进行隐式（且有意）的说明，而不是依赖于内置函数的副作用（即，正确断言情况）。

`next()`函数可以与生成器表达式结合使用，在我们要查找满足某些条件的可迭代对象的第一元素的情况下非常有用。我们将在本章中看到这个习惯用法的例子，但主要思想是使用这个函数而不是创建一个列表推导式然后取其第一个元素。

### 使用生成器

通过简单地使用生成器，可以显著简化之前的代码。生成器对象是迭代器。这样，我们不需要创建一个类，而可以定义一个函数，按需产生值：

```py
def sequence(start=0):
    while True:
        yield start
        start += 1 
```

记住，从我们的第一个定义来看，函数体内的`yield`关键字使其成为一个生成器。因为它是生成器，所以创建一个无限循环是完全可行的，因为当这个生成器函数被调用时，它将运行所有代码直到下一个`yield`语句。它将产生其值并暂停在那里：

```py
>>> seq = sequence(10)
>>> next(seq)
10
>>> next(seq)
11
>>> list(zip(sequence(), "abcdef"))
[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'f')] 
```

这种差异可以被视为我们在上一章中探讨的不同创建装饰器方式的一个类比（使用函数对象）。在这里，我们也可以使用生成器函数或可迭代对象，就像上一节中那样。只要可能，建议构造生成器，因为它在语法上更简单，因此更容易理解。

### itertools

与可迭代对象一起工作的好处是，代码与 Python 本身更好地融合，因为迭代是语言的关键组成部分。除此之外，我们可以充分利用`itertools`模块（ITER-01）。实际上，我们刚刚创建的`sequence()`生成器与`itertools.count()`相当相似。然而，我们还能做更多。

迭代器、生成器和 itertools 最令人愉悦的一点是，它们是可组合的对象，可以串联在一起。

例如，回到我们最初处理`purchases`以获取一些度量的例子，如果我们想做同样的事情，但只针对超过某个阈值的值，该怎么办？解决这个问题的天真方法是在迭代时放置条件：

```py
# ...
    def process(self):
        for purchase in self.purchases:
            if purchase > 1000.0:
                ... 
```

这不仅不符合 Python 风格，而且也很僵化（僵化是表示糟糕代码的特征）。它处理变化的能力很差。如果现在数字变了怎么办？我们通过参数传递吗？如果我们需要不止一个呢？如果条件不同（比如小于）怎么办？我们传递一个`lambda`吗？

这些问题不应该由这个对象来回答，它的唯一责任是在表示为数字的购买流上计算一系列定义良好的度量。当然，答案是否定的。做出这样的改变将是一个巨大的错误（再次强调，干净的代码是灵活的，我们不希望通过将这个对象与外部因素耦合使其变得僵化）。这些需求将不得不在其他地方解决。

最好保持这个对象与其客户独立。这个类承担的责任越少，它对更多客户就越有用，从而增加其被重用的机会。

我们不会改变这段代码，而是保持原样，并假设新数据是根据每个类客户的要求进行过滤的。

例如，如果我们只想处理前`10`笔超过`1000`的购买，我们会这样做：

```py
>>> from itertools import islice
>>> purchases = islice(filter(lambda p: p > 1000.0, purchases), 10)
>>> stats = PurchasesStats(purchases).process()  # ... 
```

以这种方式进行过滤不会产生内存惩罚，因为它们都是生成器，所以评估总是延迟的。这让我们能够像一次性过滤整个集合然后传递给对象一样思考，但实际上并不需要在内存中放入所有内容。

请记住章节开头提到的权衡，即在内存和 CPU 使用之间的权衡。虽然代码可能使用更少的内存，但它可能需要更多的 CPU 时间，但大多数时候，这是可以接受的，当我们需要在内存中处理大量对象的同时保持代码的可维护性。

### 通过迭代器简化代码

现在，我们将简要讨论一些可以用迭代器帮助改进的情况，以及偶尔使用 `itertools` 模块的情况。在讨论每个案例及其提出的优化后，我们将用推论结束每个要点。

#### 重复迭代

现在我们已经了解了更多关于迭代器的信息，并介绍了 `itertools` 模块，我们可以向您展示本章的第一个例子（计算一些购买数据的统计信息）可以如何显著简化：

```py
def process_purchases(purchases):
    min_, max_, avg = itertools.tee(purchases, 3)
    return min(min_), max(max_), median(avg) 
```

在这个例子中，`itertools.tee` 将原始可迭代对象拆分为三个新的对象。我们将使用这些对象来完成不同类型的迭代，而无需对 `purchases` 重复三次不同的循环。

读者可以简单地验证，如果我们传递一个可迭代对象作为 `purchases` 参数，这个对象只会被遍历一次（多亏了 `itertools.tee` 函数 [TEE]），这是我们主要的要求。也可以验证这个版本如何与我们的原始实现等效。在这种情况下，没有必要手动引发 `ValueError`，因为将空序列传递给 `min()` 函数会这样做。

如果你正在考虑对同一个对象运行多次循环，请停下来思考 `itertools.tee` 是否能有所帮助。

`itertools` 模块包含许多有用的函数和方便的抽象，当处理 Python 中的迭代时非常有用。它还包含关于如何以习惯用法解决典型迭代问题的良好食谱。作为一般建议，如果你在考虑如何解决涉及迭代的具体问题，就去看看这个模块。即使答案不是字面上的，它也会是一个很好的灵感来源。

#### 嵌套循环

在某些情况下，我们需要遍历多个维度，寻找一个值，嵌套循环是第一个想法。当找到值时，我们需要停止迭代，但 `break` 关键字并不完全起作用，因为我们必须从两个（或更多）`for` 循环中退出，而不仅仅是其中一个。

这个问题的解决方案会是什么？一个表示退出的标志？不。引发异常？不，这将与标志相同，但更糟，因为我们知道异常不应该用于控制流逻辑。将代码移动到更小的函数并返回它？接近，但还不够。

答案是，尽可能地将迭代扁平化为单个 `for` 循环。

这是我们希望避免的代码类型：

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

下面是这个简化的版本，它不依赖于标志来指示终止，并且具有更简单、更紧凑的迭代结构：

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
    except StopIteration as e:
        raise ValueError(f"{desired_value} not found") from e
    logger.info("value %r found at [%i, %i]", desired_value, *coord)
    return coord 
```

值得注意的是，我们创建的辅助生成器是如何作为所需迭代抽象的。在这种情况下，我们只需要迭代两个维度，但如果我们需要更多，一个不同的对象可以处理这些，而客户端无需了解这些。这就是迭代器设计模式的核心，在 Python 中，它是透明的，因为它自动支持迭代器对象，这是下一节将要讨论的主题。

尽可能地使用尽可能多的抽象来简化迭代，在可能的地方简化循环。

希望这个例子能给你带来灵感，让你明白我们可以使用生成器做的不只是节省内存。我们可以利用迭代作为抽象。也就是说，我们不仅可以通过定义类或函数来创建抽象，还可以利用 Python 的语法。就像我们看到了如何通过上下文管理器抽象掉一些逻辑（这样我们就不需要知道`with`语句下发生的事情的细节），我们也可以用迭代器做到同样的事情（这样我们就可以忘记`for`循环的底层逻辑）。

因此，我们将从下一节开始探索 Python 中迭代器模式的工作原理。

### Python 中的迭代器模式

在这里，我们将从生成器稍微偏离一下，以更深入地理解 Python 中的迭代。生成器是可迭代对象的一个特例，但 Python 中的迭代不仅仅局限于生成器，能够创建良好的可迭代对象将给我们机会编写更高效、紧凑和易于阅读的代码。

在之前的代码示例中，我们已经看到了既是`可迭代对象`又是迭代器的例子，因为它们实现了`__iter__()`和`__next__()`魔法方法。虽然这在一般情况下是可以的，但它们并不严格需要总是实现这两个方法，在这里我们将展示`可迭代对象`（实现了`__iter__`）和迭代器（实现了`__next__`）之间的微妙差异。

我们还探讨了与迭代相关的一些其他主题，例如序列和容器对象。

#### 迭代接口

一个`可迭代对象`是一个支持迭代的对象，在非常高的层面上，这意味着我们可以运行一个`for` .. `in` ... 循环来遍历它，而不会出现任何问题。然而，`可迭代对象`并不等同于迭代器。

通常来说，一个`可迭代对象`就是我们能够迭代的任何东西，它通过迭代器来实现这一点。这意味着在`__iter__`魔法方法中，我们希望返回一个迭代器，即实现了`__next__()`方法的对象。

迭代器是一个对象，它只知道如何在被已经探索过的内置`next()`函数调用时逐个产生一系列值，当迭代器没有被调用时，它只是简单地冻结，无所事事地坐着，直到再次被调用以产生下一个值。从这个意义上说，生成器是迭代器。

| Python 概念 | 魔法方法 | 考虑事项 |
| --- | --- | --- |
| 可迭代对象 | `__iter__` | 它们使用迭代器来构建迭代逻辑。这些对象可以在`for` ... `in` ...循环中进行迭代。 |
| 迭代器 | `__next__` | 定义逐个产生值的逻辑。`StopIteration`异常表示迭代结束。值可以通过内置的`next()`函数逐个获取。 |

表 7.1：可迭代对象和迭代器

在下面的代码中，我们将看到一个迭代器对象的例子，它不是可迭代的——它只支持逐个调用其值。在这里，名称`sequence`仅仅指一系列连续的数字，并不指 Python 中的序列概念，我们将在稍后探讨：

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

注意，我们可以逐个获取序列的值，但我们不能遍历这个对象（这是幸运的，因为否则会导致无限循环）：

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

错误信息很明确，因为这个对象没有实现`__iter__()`。

仅为了解释目的，我们可以将迭代分离到另一个对象中（再次，只要对象实现了`__iter__`和`__next__`，就足够了，但这样分开做将有助于阐明我们在这个解释中试图说明的独特点）。

#### 序列对象作为可迭代对象

正如我们刚才看到的，如果一个对象实现了`__iter__()`魔法方法，这意味着它可以在`for`循环中使用。虽然这是一个很好的特性，但它不是我们能够实现的唯一迭代形式。当我们编写`for`循环时，Python 会尝试查看我们使用的对象是否实现了`__iter__`，如果实现了，它将使用这个来构建迭代，如果没有实现，还有回退选项。

如果对象恰好是一个序列（意味着它实现了`__getitem__()`和`__len__()`魔法方法），它也可以进行迭代。如果是这样，解释器将按顺序提供值，直到抛出`IndexError`异常，这与前面提到的`StopIteration`类似，也标志着迭代的结束。

为了仅说明这种行为，我们将运行以下实验，展示一个实现了对数字范围应用`map()`的序列对象：

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

请记住，这个例子只是为了说明像这样的对象可以用普通的`for`循环进行迭代。在`__getitem__`方法中放置了一个日志行，以探索在对象被迭代时传递了哪些值，正如我们从下面的测试中可以看到：

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

作为一句警告，重要的是要强调，虽然了解这一点很有用，但它也是当对象没有实现`__iter__`时的回退机制，所以大多数时候我们都会通过考虑创建合适的序列，而不是仅仅迭代对象来使用这些方法。

在考虑设计用于迭代的对象时，优先考虑一个合适的迭代器对象（具有`__iter__`），而不是一个偶然也可以迭代的序列。

迭代器是 Python 的重要组成部分，不仅因为它们为我们软件工程师提供的功能，还因为它们在 Python 内部起着基本的作用。

在第二章“Pythonic Code”中的“*异步代码简介*”中，我们看到了如何阅读异步代码。现在我们已经探索了 Python 中的迭代器，我们可以看到这两个概念是如何相关的。特别是，下一节将探讨协程，我们将看到迭代器是如何成为其核心的。

# 协程

协程的想法是有一个函数，其执行可以在某个特定时间点暂停，稍后可以恢复。通过这种功能，程序可能能够暂停代码的一部分，以便处理其他事情，然后返回到这个原始点继续执行。

如我们所知，生成器对象是可迭代的。它们实现了`__iter__()`和`__next__()`。这是 Python 自动提供的，以便当我们创建生成器对象函数时，我们得到一个可以迭代或通过`next()`函数推进的对象。

除了这个基本功能之外，它们还有更多方法，以便它们可以作为协程（PEP-342）工作。在这里，我们将在下一节更详细地探讨生成器如何演变成协程以支持异步编程的基础之前，探索生成器是如何演变成协程的。在下一节中，我们将探讨 Python 的新特性和用于异步编程的语法。

PEP-342 中添加的基本方法以支持协程如下：

+   `.close()`

+   `.throw(ex_type[, ex_value[, ex_traceback]])`

+   `.send(value)`

Python 利用生成器来创建协程。因为生成器可以自然地暂停，所以它们是一个方便的起点。但是，生成器并不足以满足最初的想法，因此添加了这些方法。这是因为通常，仅仅能够暂停代码的一部分是不够的；你还想与之通信（传递数据，并通知上下文的变化）。

通过更详细地探索每个方法，我们将能够更多地了解 Python 协程的内部机制。在此之后，我将再次概述异步编程的工作原理，但与第二章“Pythonic Code”中介绍的不同，这一次它将与我们刚刚学到的内部概念相关。

## 生成器接口的方法

在本节中，我们将探讨上述每种方法的作用、工作原理以及预期如何使用。通过了解如何使用这些方法，我们将能够利用简单的协程。

之后，我们将探讨协程的更高级用法，以及如何委派给子生成器（协程）以重构代码，以及如何编排不同的协程。

### close()

当调用此方法时，生成器将接收到`GeneratorExit`异常。如果没有处理，那么生成器将完成而不会产生更多值，并且迭代将停止。

此异常可用于处理完成状态。一般来说，如果我们的协程执行某种资源管理，我们希望捕获此异常并使用该控制块释放协程持有的所有资源。这类似于使用上下文管理器或将代码放在异常控制的`finally`块中，但专门处理此异常使其更加明确。

在下面的例子中，我们有一个协程，它使用一个数据库处理对象，该对象保持对数据库的连接，并对其运行查询，通过固定长度的页面流式传输数据（而不是一次性读取所有可用的数据）：

```py
def stream_db_records(db_handler):
    try:
        while True:
            yield db_handler.read_n_records(10)
    except GeneratorExit:
        db_handler.close() 
```

在每次调用生成器时，它将返回从数据库处理程序获得的`10`行，但当我们决定显式地完成迭代并调用`close()`时，我们也希望关闭到数据库的连接：

```py
>>> streamer = stream_db_records(DBHandler("testdb"))
>>> next(streamer)
[(0, 'row 0'), (1, 'row 1'), (2, 'row 2'), (3, 'row 3'), ...]
>>> next(streamer)
[(0, 'row 0'), (1, 'row 1'), (2, 'row 2'), (3, 'row 3'), ...]
>>> streamer.close()
INFO:...:closing connection to database 'testdb' 
```

当需要执行收尾任务时，请使用生成器的`close()`方法。

此方法旨在用于资源清理，因此你通常会在无法自动执行此操作时（例如，如果你没有使用上下文管理器）手动释放资源。接下来，我们将看到如何将异常传递给生成器。

### throw(ex_type[, ex_value[, ex_traceback]])

此方法将在生成器当前挂起的那一行抛出异常。如果生成器处理了发送的异常，那么将调用特定于该`except`子句的代码；否则，异常将传播到调用者。

在这里，我们稍微修改了之前的例子，以展示当我们使用此方法处理协程处理的异常和未处理的异常时的差异：

```py
class CustomException(Exception):
    """A type of exception that is under control."""
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

现在，接收`CustomException`已成为控制流的一部分，在这种情况下，生成器将记录一条信息性消息（当然，我们可以根据每个案例的业务逻辑进行适配），然后继续到下一个`yield`语句，这是协程从数据库读取并返回数据的行。

在这个特定的例子中，它处理了所有异常，但如果最后的块（除了`Exception`：）不存在，结果将是生成器在生成器暂停的那一行（再次，`yield`）被引发，并且异常将从那里传播到调用者：

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

当我们收到来自域的异常时，生成器继续运行。然而，当它收到一个未预期的异常时，默认的块捕获了我们关闭数据库连接并完成迭代的地方，这导致生成器停止。正如我们从抛出的`StopIteration`中可以看到的，这个生成器不能进一步迭代。

### send(value)

在上一个例子中，我们创建了一个简单的生成器，它从数据库中读取行，当我们希望结束其迭代时，这个生成器释放了与数据库关联的资源。这是使用生成器提供的方法（`close()`）的一个很好的例子，但我们还可以做更多。

对生成器的观察是，它从数据库中读取固定数量的行。

我们希望将那个数字（`10`）参数化，这样我们就可以在不同的调用中更改它。不幸的是，`next()`函数没有为我们提供这样的选项。但幸运的是，我们有`send()`：

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

现在的想法是，我们已经使协程能够通过`send()`方法从调用者那里接收值。这个方法是真正区分生成器和协程的方法，因为当它被使用时，意味着`yield`关键字将出现在语句的右侧，并且它的返回值将被分配给其他某个东西。

在协程中，我们通常发现`yield`关键字以以下形式使用：

```py
receive = yield produced 
```

在这个例子中，`yield`将执行两个操作。它将`produced`发送回调用者，调用者将在下一次迭代中（例如，在调用`next()`之后）获取它，并且在那里暂停。在稍后的某个时刻，调用者将通过使用`send()`方法将一个值发送回协程。这个值将成为`yield`语句的结果，在本例中分配给名为`receive`的变量。

仅当协程在`yield`语句处暂停，等待产生某些内容时，向协程发送值才有效。为了实现这一点，必须将协程推进到该状态。做到这一点的唯一方法是在其上调用`next()`。这意味着在向协程发送任何内容之前，至少要通过`next()`方法推进一次。未能这样做将导致异常：

```py
>>> def coro():
...     y = yield
...
>>> c = coro()
>>> c.send(1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: can't send non-None value to a just-started generator
>>> 
```

总是记得在向协程发送任何值之前，通过调用`next()`来推进协程。

回到我们的例子。我们正在改变元素的产生或流式传输方式，使其能够接收它从数据库中期望读取的记录长度。

第一次调用`next()`时，生成器将推进到包含`yield`的行；它将提供一个值给调用者（`None`，如变量中设置的），并且它将在那里暂停）。从那里，我们有两种选择。如果我们选择通过调用`next()`推进生成器，默认值`10`将被使用，并且它将像往常一样继续。这是因为调用`next()`在技术上等同于`send(None)`，但这在处理我们之前设置的值的`if`语句中已经讨论过了。

如果，另一方面，我们决定通过`send(<value>)`提供一个显式的值，这个值将成为`yield`语句的结果，它将被分配给包含要使用页面长度的变量，这个变量反过来将被用来从数据库中读取。

连续调用将具有这种逻辑，但重要的是现在我们可以在迭代过程中动态地改变要读取的数据长度，在任何时候都可以。

现在我们已经理解了前面的代码是如何工作的，大多数 Python 开发者都会期待一个简化版本（毕竟，Python 也关于简洁和干净、紧凑的代码）：

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

这个版本不仅更紧凑，而且更好地说明了这个想法。`yield`周围的括号使它更清楚地表明它是一个语句（把它想象成一个函数调用），并且我们正在使用它的结果来与之前的值进行比较。

这就像我们预期的那样工作，但我们总是必须记得在向它发送任何数据之前推进协程。如果我们忘记调用第一个`next()`，我们会得到一个`TypeError`。这个调用可以忽略，因为不会返回我们将要使用的内容。

如果我们能够在创建协程后立即使用它，而不必每次使用时都记得调用`next()`，那就太好了。一些作者（PYCOOK）设计了一个有趣的装饰器来实现这一点。这个装饰器的想法是推进协程，所以下面的定义可以自动工作：

```py
@prepare_coroutine
def auto_stream_db_records(db_handler):
    retrieved_data = None
    page_size = 10
    try:
        while True:
            page_size = (yield retrieved_data) or page_size
            retrieved_data = db_handler.read_n_records(page_size)
    except GeneratorExit:
        db_handler.close() 
```

```py
>>> streamer = auto_stream_db_records(DBHandler("testdb"))
>>> len(streamer.send(5))
5 
```

请记住，这些是 Python 中协程工作的基础。通过遵循这些示例，你会对 Python 在处理协程时实际发生的事情有一个概念。然而，在现代 Python 中，你通常不会自己编写这类协程，因为已经有了新的语法（我们之前提到过，但我们将重新审视，看看它们如何与我们刚刚看到的概念相关）。

在深入研究新的语法功能之前，我们需要探索协程在功能上所做的最后跳跃，以便填补缺失的空白。之后，我们将能够理解在异步编程中使用的每个关键字和语句背后的含义。

## 更高级的协程

到目前为止，我们对协程有了更好的理解，我们可以创建简单的协程来处理小任务。我们可以认为这些协程实际上只是更高级的生成器（这将是正确的，协程只是花哨的生成器），但如果我们实际上想要开始支持更复杂的场景，我们通常必须选择一个可以同时处理许多协程的设计，这需要更多的功能。

当处理许多协程时，我们会遇到新的问题。随着我们的应用程序的控制流变得更加复杂，我们希望在上传和下传堆栈（以及异常），能够从任何级别的子协程中捕获值，并最终安排多个协程共同实现一个目标。

为了使事情更简单，生成器不得不再次扩展。这就是 PEP-380 通过改变生成器的语义，使它们能够返回值，并引入新的`yield from`构造来解决的问题。

### 协程中的返回值

如本章开头所述，迭代是一种机制，它多次在可迭代对象上调用`next()`，直到抛出`StopIteration`异常。

到目前为止，我们一直在探索生成器的迭代特性——我们一次产生一个值，通常我们只关心在`for`循环的每一步产生的每个值。这是关于生成器的一种非常逻辑的思考方式，但协程有不同的想法；尽管它们在技术上也是生成器，但它们并不是以迭代的概念来构思的，而是以在稍后恢复执行时挂起代码执行为目标。

这是一个有趣的挑战；当我们设计协程时，我们通常更关心挂起状态而不是迭代（迭代协程将是一个奇怪的情况）。挑战在于很容易将它们两者混合。这是因为一个技术实现细节；Python 中对协程的支持建立在生成器的基础上。

如果我们想使用协程来处理一些信息并挂起其执行，那么将它们视为轻量级线程（在其他平台上被称为绿色线程）是有意义的。在这种情况下，如果它们能够返回值，就像调用任何其他常规函数一样，那就更有意义了。

但让我们记住，生成器不是常规函数，所以在生成器中，构造`value = generator()`除了创建一个生成器对象之外，不会做任何事情。生成器返回值的语义应该是什么？它必须在迭代完成后才能进行。

当生成器返回一个值时，其迭代立即停止（它不能再迭代）。为了保持语义，`StopIteration`异常仍然会被抛出，而要返回的值被存储在`exception`对象中。这是调用者的责任去捕获它。

在下面的例子中，我们创建了一个简单的生成器，它产生两个值，然后返回第三个。注意我们如何必须捕获异常才能获取这个值，以及它是如何精确地存储在异常的`value`属性下的：

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
...     print(f">>>>>> returned value: {e.value}")
... 
>>>>>> returned value: 3 
```

正如我们稍后将会看到的，这个机制被用来使协程返回值。在 PEP-380 之前，这并没有什么意义，任何在生成器内部使用`return`语句的尝试都被视为语法错误。但现在，我们的想法是，当迭代结束时，我们想要返回一个最终值，而提供它的方式是将它存储在迭代结束时的异常中（`StopIteration`）。这可能不是最干净的方法，但它完全向后兼容，因为它没有改变生成器的接口。

### 委派到更小的协程 - 'yield from' 语法

前面的特性在意义上很有趣，因为它为协程（生成器）打开了大量新的可能性，现在它们可以返回值。但这个特性本身，如果没有适当的语法支持，将不会那么有用，因为以这种方式捕获返回值有点繁琐。

这是`yield from`语法的最主要特性之一。在其他方面（我们将在详细回顾），它可以收集子生成器返回的值。记住我们说过，在生成器中返回数据是很好的，但不幸的是，将语句写成`value = generator()`是不行的？好吧，将它们写成`value = yield from generator()`就可以。

#### `yield from`的最简单用法

在其最基本的形式中，新的`yield from`语法可以用来将嵌套的`for`循环中的生成器链式连接成一个单一的循环，最终得到一个连续流中所有值的单个字符串。

一个典型的例子是创建一个类似于`itertools.chain()`的函数，这个函数来自`standard`库。这是一个非常好的函数，因为它允许你传递任意数量的`iterables`，并将它们全部作为一个流返回。

天真的实现可能看起来像这样：

```py
def chain(*iterables):
    for it in iterables:
        for value in it:
            yield value 
```

它接收一个可变数量的`iterables`，遍历它们，由于每个值都是可迭代的，它支持`for... in..`构造，因此我们有一个额外的`for`循环来获取每个特定可迭代对象中的每个值，这些值是由调用函数产生的。

这可能在多种情况下很有用，比如将生成器链式连接起来，或者尝试迭代那些通常不可能一次性比较的事物（比如列表和元组等）。

然而，`yield from`语法允许我们更进一步，避免嵌套循环，因为它能够直接从子生成器产生值。在这种情况下，我们可以将代码简化如下：

```py
def chain(*iterables):
    for it in iterables:
        yield from it 
```

注意，对于两种实现，生成器的行为完全相同：

```py
>>> list(chain("hello", ["world"], ("tuple", " of ", "values.")))
['h', 'e', 'l', 'l', 'o', 'world', 'tuple', ' of ', 'values.'] 
```

这意味着我们可以将 `yield from` 应用于任何其他可迭代对象，并且它将像顶级生成器（使用 `yield from` 的那个）自己生成这些值一样工作。

这适用于任何可迭代对象，甚至生成器表达式也不例外。现在我们熟悉了它的语法，让我们看看我们如何编写一个简单的生成器函数，该函数将产生一个数的所有幂（例如，如果提供 `all_powers(2, 3)`，它将必须产生 `2⁰`，`2¹`，`... 2³`）：

```py
def all_powers(n, pow):
    yield from (n ** i for i in range(pow + 1)) 
```

虽然这简化了语法，节省了一行 `for` 语句，但这并不是一个很大的优势，而且这不足以证明将这种改变添加到语言中的合理性。

事实上，这实际上只是一个副作用，`yield from` 构造的真正目的是我们将在接下来的两个部分中探讨的。

#### 捕获子生成器返回的值

在下面的例子中，我们有一个生成器调用了另外两个嵌套生成器，按顺序产生值。这些嵌套生成器中的每一个都返回一个值，我们将看到顶级生成器是如何有效地捕获返回值的，因为它通过 `yield from` 调用内部生成器：

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

这是 `main` 代码在迭代过程中的一个可能的执行情况：

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

`main` 的第一行将任务委托给内部生成器，并产生值，直接从中提取。这并不新鲜，因为我们已经看到了。注意，`sequence()` 生成器函数返回的结束值被分配在第一行给名为 `step1` 的变量，以及这个值是如何在下一个生成器实例的开始处正确使用的。

最后，这个其他的生成器也返回第二个结束值（`10`），而主生成器则返回它们的和（`5+10=15`），这就是迭代停止后我们看到的值。

我们可以使用 `yield from` 来捕获一个协程在完成其处理后的最后一个值。

通过这个例子和上一节中展示的例子，你可以了解 `yield from` 构造在 Python 中的工作原理。`yield from` 构造将获取生成器，并将其迭代传递到下游，但一旦完成，它将捕获其 `StopIteration` 异常，获取其值，并将该值返回给调用函数。`StopIteration` 异常的值属性成为该语句的结果。

这是一个强大的构造，因为它与下一节的主题（如何从子生成器发送和接收上下文信息）结合，这意味着协程可以采取类似于线程的形状。

#### 向子生成器发送和接收数据

现在，我们将看到`yield from`语法的另一个优点，这可能是它全部力量的来源。正如我们在探索作为协程的生成器时已经介绍过的，我们知道我们可以向它们发送值和抛出异常，在这种情况下，协程将接收值以进行内部处理，或者它必须相应地处理异常。

如果我们现在有一个将任务委托给其他协程的协程（例如在之前的例子中），我们也希望保留这种逻辑。手动这样做将会非常复杂（如果`yield from`自动处理了这个问题，你可以查看 PEP-380 中描述的代码）。

为了说明这一点，让我们保持与之前例子（调用其他内部生成器）相同的顶层生成器（main）不变，但让我们修改内部生成器，使它们能够接收值和处理异常。

代码可能不是典型的用法，只是为了展示这个机制是如何工作的：

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

现在，我们将通过迭代它，并给它提供值以及抛出异常来调用`main`协程，以便观察它们在序列中的处理方式：

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

这个例子告诉我们很多不同的事情。注意我们从未向`sequence`发送值，而只向`main`发送，即便如此，接收这些值的代码是嵌套的生成器。尽管我们从未明确地向`sequence`发送任何东西，但它仍然通过`yield from`传递数据时接收数据。

`main`协程内部调用两个其他协程，产生它们的值，并且它将在这些协程中的某个特定时间点挂起。当它停止在第一个协程上时，我们可以看到日志告诉我们是那个协程实例接收了我们发送的值。当我们向它抛出异常时，情况也是如此。当第一个协程完成时，它返回分配在名为`step1`的变量中的值，并将其作为输入传递给第二个协程，该协程将做同样的事情（它将处理`send()`和`throw()`调用，相应地）。

同样，每个协程产生的值也会发生这种情况。当我们处于任何给定步骤时，调用`send()`的返回值对应于子协程（`main`当前挂起的那个）产生的值。当我们抛出一个被处理的异常时，`sequence`协程产生值`OK`，这个值被传播到被调用的协程（`main`），然后最终到达`main`的调用者。

如预期的那样，这些方法与`yield from`一起为我们提供了很多新的功能（这可以类似于线程）。这为异步编程打开了大门，我们将在下一节中探讨。

# 异步编程

到目前为止我们所看到的构造，我们可以在 Python 中创建异步程序。这意味着我们可以创建具有许多协程的程序，按特定顺序调度它们工作，并在每个协程在调用`yield from`后挂起时在它们之间切换。

我们可以从这里获得的主要优势是能够以非阻塞方式并行化 I/O 操作。我们需要的是一个低级生成器（通常由第三方库实现），它知道如何在协程挂起时处理实际的 I/O。想法是让协程实现挂起，以便我们的程序可以在同时处理另一个任务。应用程序通过`yield from`语句恢复控制，这将挂起并产生一个值给调用者（正如我们在之前使用此语法改变程序控制流时的示例中看到的那样）。

这大致是 Python 中异步编程工作了几年的方式，直到决定需要更好的语法支持。

协程和生成器在技术上相同的事实造成了一些混淆。在语法（和技术）上，它们是相同的，但在语义上，它们是不同的。我们创建生成器是为了实现高效的迭代。我们通常创建协程的目的是运行非阻塞的 I/O 操作。

虽然这种区别很清楚，但 Python 的动态性质仍然允许开发者混合这些不同类型的对象，最终在程序非常晚的阶段出现运行时错误。记住，在最简单和最基本形式的`yield from`语法中，我们是在可迭代对象上使用这种构造（我们创建了一种应用于字符串、列表等的`chain`函数）。这些对象都不是协程，但仍然可以工作。然后，我们看到我们可以有多个协程，使用`yield from`发送值（或异常），并得到一些结果。这些显然是两种非常不同的用例；然而，如果我们编写类似以下语句的内容：

```py
result = yield from iterable_or_awaitable() 
```

`iterable_or_awaitable`返回的内容并不明确。它可能是一个简单的可迭代对象，例如一个`string`，并且可能仍然是语法正确的。或者，它可能是一个实际的协程。这个错误的代价将在很久以后，在运行时付出。

因此，Python 中的类型系统必须得到扩展。在 Python 3.5 之前，协程只是应用了`@coroutine`装饰器的生成器，并且它们需要使用`yield from`语法来调用。现在，Python 解释器识别出一种特定的对象，即协程。

这一变化也预示了语法的改变。引入了`await`和`async def`语法。前者旨在替代`yield from`，并且仅与`awaitable`对象（协程恰好是）一起使用。尝试用不遵守`awaitable`接口的东西调用`await`将引发异常（这是一个很好的例子，说明了接口如何有助于实现更稳固的设计，防止运行时错误）。

`async def`是定义协程的新方法，取代了上述装饰器，并且实际上创建了一个对象，当调用它时，将返回一个协程的实例。与调用生成器函数的方式相同，解释器将返回一个生成器对象，当你调用用`async def`定义的对象时，它将给你一个具有`__await__`方法的协程对象，因此可以在`await`表达式中使用。

不深入探讨 Python 异步编程的所有细节和可能性，我们可以这样说，尽管有新的语法和新类型，但这并没有做任何本质上与我们本章所讨论的概念不同的东西。

Python 中异步编程的核心理念是存在一个`事件循环`（通常是`asyncio`，因为它包含在`标准`库中，但还有许多其他同样可以工作的循环），它管理一系列的协程。这些协程属于事件循环，它将根据其调度机制调用它们。当这些协程中的任何一个运行时，它将调用我们的代码（根据我们在编写的协程中定义的逻辑），当我们想要将控制权交还给事件循环时，我们调用`await <协程>`，这将异步处理一个任务。事件循环将继续运行，并启动另一个协程，同时之前的操作仍在进行中。

这种机制代表了 Python 中异步编程工作的基本原理。你可以认为为协程添加的新语法（`async def` / `await`）只是为你编写代码的一个 API，以便由事件循环调用。默认情况下，该事件循环通常是`asyncio`，因为它包含在`标准`库中，但任何符合 API 的事件循环系统都可以工作。这意味着你可以使用像`uvloop`([`github.com/MagicStack/uvloop`](https://github.com/MagicStack/uvloop))和`trio`([`github.com/python-trio/trio`](https://github.com/python-trio/trio))这样的库，代码将按相同的方式工作。你甚至可以注册自己的事件循环，它也应该按相同的方式工作（前提是符合 API 规范）。

实际上，还有更多特定的特性和边缘情况超出了本书的范围。然而，值得指出的是，这些概念与本章中介绍的思想相关，而且这个领域是另一个展示生成器作为语言核心概念的地方，因为许多东西都是建立在它们之上的。

## 魔法异步方法

我在前几章中已经提出（并希望说服你）只要有可能，我们就可以利用 Python 中的魔法方法，使我们所创建的抽象与语言的语法自然融合，从而实现更好、更紧凑、可能更干净的代码。

但如果在这些方法中的任何一个我们需要调用协程怎么办？如果我们必须在函数中调用`await`，这意味着该函数本身必须是一个协程（使用`async def`定义），否则将会出现语法错误。

然而，这与当前的语法和魔法方法如何工作呢？它不起作用。我们需要新的语法和新的魔法方法才能与异步编程一起工作。好消息是，它们与之前的类似。

这里是对新魔法方法和它们如何与新语法相关的一个总结。

| 概念 | 魔法方法 | 语法用法 |
| --- | --- | --- |
| 上下文管理器 | `__aenter__` `__aexit__` | `async with async_cm() as x:`... |
| 迭代 | `__aiter__` `__anext__` | `async for e in aiter:`... |

表 7.2：异步语法及其魔法方法

这种新语法在 PEP-492 中有所提及（[`www.python.org/dev/peps/pep-0492/`](https://www.python.org/dev/peps/pep-0492/))。

### 异步上下文管理器

简单来说，如果我们想使用上下文管理器但需要在其上调用协程，我们不能使用正常的`__enter__`和`__exit__`方法，因为它们被定义为常规函数，所以我们需要使用新的`__aenter__`和`__aexit__`协程方法。而且，我们不仅需要使用`with`来调用它，还需要使用`async` with。

在`contextlib`模块中甚至还有一个`@asynccontextmanager`装饰器，可以创建与之前所示相同的异步上下文管理器。

异步上下文管理器的`async` with 语法以类似的方式工作：当上下文进入时，`__aenter__`协程会自动调用，当它退出时，`__aexit__`将被触发。甚至可以在同一个`async` with 语句中组合多个异步上下文管理器，但不能与常规的混合使用。尝试使用常规上下文管理器与`async` with 语法将失败并抛出`AttributeError`。

如果将我们的例子从*第二章*，*Pythonic Code*，改编为异步编程，它将看起来像以下代码：

```py
@contextlib.asynccontextmanager
async def db_management():
    try:
        await stop_database()
        yield
    finally:
        await start_database() 
```

此外，如果我们想使用多个上下文管理器，我们可以这样做，例如：

```py
@contextlib.asynccontextmanager
async def metrics_logger():
    yield await create_metrics_logger()

async def run_db_backup():
    async with db_management(), metrics_logger():
        print("Performing DB backup...") 
```

如你所预期，`contextlib`模块提供了一个抽象基类`AbstractAsyncContextManager`，它要求实现`__aenter__`和`__aexit__`方法。

### 其他魔法方法

那些其他魔法方法会发生什么？它们都会得到它们的异步对应物吗？不，但我想指出的是：这不应该需要。

记住，编写干净代码的部分是确保你在代码中正确分配责任，并将事物放在适当的位置。举个例子，如果你在考虑在`__getattr__`方法中调用协程，那么你的设计可能存在问题，因为可能有一个更好的地方来放置那个协程。

我们等待的协程用于使代码的某些部分并发运行，因此它们通常与外部资源的管理相关，而我们在其他魔法方法（`(__getitem__`, `__getattr__`, 等）中放入的逻辑应该是面向对象的代码，或者可以仅根据该对象的内部表示来解决的代码。

同样地（并且遵循良好的设计实践），将`__init__`设计为协程并不是一个好的选择，因为我们通常希望创建轻量级对象，这样我们可以在没有副作用的情况下安全地初始化它们。更好的是，我们已经讨论了使用依赖注入的好处，因此这更是我们不希望有一个异步初始化方法的原因：我们的对象应该与已经初始化的依赖项一起工作。

上一表的第二个案例，异步迭代，对于本章的目的来说更有兴趣，所以我们将在下一节中探讨它。

异步迭代的语法（`async for`）适用于任何异步迭代器，无论是我们自己创建的（我们将在下一节中看到如何做到这一点），还是异步生成器（我们将在下一节中看到）。

## 异步迭代

就像我们在本章开头看到的迭代器对象（即支持使用 Python 内置的`for`循环进行迭代的对象）一样，我们也可以这样做，但以异步的方式进行。

想象一下，我们想要创建一个迭代器来抽象我们从外部源（如数据库）读取数据的方式，但提取数据本身的操作是一个协程，所以我们不能像以前那样在已熟悉的`__next__`操作中调用它。这就是为什么我们需要使用`__anext__`协程的原因。

以下示例以简单的方式说明了如何实现这一点。不考虑外部依赖或任何其他意外复杂性，我们将专注于使此类操作成为可能的方法，以便研究它们：

```py
import asyncio
import random

async def coroutine():
    await asyncio.sleep(0.1)
    return random.randint(1, 10000)

class RecordStreamer:
    def __init__(self, max_rows=100) -> None:
        self._current_row = 0
        self._max_rows = max_rows

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._current_row < self._max_rows:
            row = (self._current_row, await coroutine())
            self._current_row += 1
            return row
        raise StopAsyncIteration 
```

第一种方法，`__aiter__`，用于表示该对象是一个异步迭代器。正如在同步版本中，大多数情况下返回 self 就足够了，因此它不需要是一个协程。

但另一方面，`__anext__`正是我们代码中异步逻辑所在的部分，因此它一开始就需要是一个协程。在这种情况下，我们正在等待另一个协程以返回要返回的部分数据。

它还需要一个单独的异常来表示迭代的结束，在这种情况下，称为`StopAsyncIteration`。

这个异常以类似的方式工作，只是它是为`async for`类型的循环设计的。当遇到这种情况时，解释器将结束循环。

这种对象可以用以下形式使用：

```py
async for row in RecordStreamer(10):
    ... 
```

你可以清楚地看到这与我们在本章开头探索的同步版本是如何类似的。不过，有一个重要的区别是，正如我们所预期的，`next()`函数不会在这个对象上工作（毕竟它没有实现`__next__`），因此要向前推进一个异步生成器，就需要不同的语法。

通过以下方式可以向前推进异步迭代器一个位置：

```py
await async_iterator.__anext__() 
```

但更有趣的结构，比如我们之前看到的，使用`next()`函数在生成器表达式中工作以搜索满足某些条件的第一值，将不会得到支持，因为它们无法处理异步迭代器。

受到前面语法的启发，我们可以使用异步迭代创建一个生成器表达式，然后从中获取第一个值。更好的是，我们可以创建我们自己的这个函数版本来与异步生成器一起工作，它可能看起来像这样：

```py
NOT_SET = object()

async def anext(async_generator_expression, default=NOT_SET):
    try:
        return await async_generator_expression.__anext__()
    except StopAsyncIteration:
        if default is NOT_SET:
            raise
        return default 
```

从 Python 3.8 开始，`asyncio`模块有一个很好的功能，允许我们从 REPL 直接与协程交互。这样，我们可以交互式地测试前面的代码将如何工作：

```py
$ python -m asyncio
>>> streamer = RecordStreamer(10)
>>> await anext(streamer)
(0, 5017)
>>> await anext(streamer)
(1, 5257)
>>> await anext(streamer)
(2, 3507)
...
>>> await anext(streamer)
(9, 5440)
>>> await anext(streamer)
Traceback (most recent call last):
    ...
    raise StopAsyncIteration
StopAsyncIteration
>>> 
```

你会注意到它在接口和行为上都与原始的`next()`函数相似。

现在我们知道了如何在异步编程中使用迭代，但我们可以做得更好。大多数时候我们只需要一个生成器，而不是整个迭代器对象。生成器的优势在于它们的语法使得它们更容易编写和理解，所以在下节中，我将提到如何为异步程序创建生成器。

## 异步生成器

在 Python 3.6 之前，上一节中探索的功能是 Python 中实现异步迭代的唯一方法。由于我们在前几节中探讨了协程和生成器的复杂性，尝试在协程内部使用`yield`语句并没有完全定义，因此不允许（例如，`yield`会尝试挂起协程，还是为调用者生成一个值？）。

异步生成器是在 PEP-525 中引入的 ([`www.python.org/dev/peps/pep-0525/`](https://www.python.org/dev/peps/pep-0525/))。

在这个 PEP 中解决了在协程中使用 `yield` 关键字的问题，现在允许使用，但具有不同的和明确的意义。与我们所看到的第一个协程示例不同，协程中的 `yield`（使用 `async def` 正确定义）并不意味着暂停或暂停该协程的执行，而是为调用者生成一个值。这是一个异步生成器：与我们在章节开头看到的生成器相同，但可以以异步方式使用（意味着它们可能在定义内部等待其他协程）。

异步生成器相对于迭代器的主要优势与常规生成器相同的优势；它们允许我们以更紧凑的方式实现相同的事情。

正如承诺的那样，使用异步生成器编写的上一个示例看起来更紧凑：

```py
async def record_streamer(max_rows):
    current_row = 0
    while current_row < max_rows:
        row = (current_row, await coroutine())
        current_row += 1
        yield row 
```

它感觉更接近常规生成器，因为结构是相同的，只是多了 `async def` / `await` 构造。此外，你将不得不记住更少的细节（关于需要实现的方法和必须触发的正确异常），因此我建议，在可能的情况下，你应尽可能优先考虑异步生成器而不是迭代器。

这标志着我们通过 Python 的迭代和异步编程之旅的结束。特别是，我们刚刚探讨的最后一个主题是它的巅峰，因为它与我们在这章中学到的所有概念都有关。

# 摘要

生成器在 Python 中无处不在。自从它们在 Python 中很久以前引入以来，它们证明是一个伟大的补充，使程序更高效，迭代更简单。

随着时间的推移，需要添加到 Python 中的更复杂任务越来越多，生成器再次帮助支持协程。

而且，尽管在 Python 中协程是生成器，我们仍然不能忘记它们在语义上是不同的。生成器是以迭代的概念创建的，而协程的目的是异步编程（在任何给定时间暂停和恢复我们程序的一部分执行）。这种区别变得如此重要，以至于它使 Python 的语法（以及类型系统）发生了演变。

迭代和异步编程构成了 Python 编程的主要支柱的最后一部分。现在，是时候看看所有这些内容是如何结合在一起的，并将我们在过去几章中探索的所有这些概念付诸实践。这意味着，到目前为止，你已经完全理解了 Python 的功能。

现在是时候利用这个优势了，所以接下来几章，我们将看到如何将这些概念付诸实践，与更通用的软件工程思想相关，如测试、设计模式和架构。

我们将在下一章开始探索单元测试和重构这一新部分。

# 参考资料

这里有一份您可以参考的信息列表：

+   *PEP-234*: *迭代器* ([`www.python.org/dev/peps/pep-0234/`](https://www.python.org/dev/peps/pep-0234/))

+   *PEP-255*: *简单生成器* ([`www.python.org/dev/peps/pep-0255/`](https://www.python.org/dev/peps/pep-0255/))

+   *ITER-01*: *Python 的 itertools 模块* ([`docs.python.org/3/library/itertools.html`](https://docs.python.org/3/library/itertools.html))

+   *GoF*: 由*Erich Gamma*、*Richard Helm*、*Ralph Johnson*和*John Vlissides*合著的名为*设计模式：可重用面向对象软件元素*的书籍

+   *PEP-342*: *通过增强生成器实现协程* ([`www.python.org/dev/peps/pep-0342/`](https://www.python.org/dev/peps/pep-0342/))

+   *PYCOOK*: 由*Brian Jones*和*David Beazley*合著的名为*Python Cookbook: Recipes for Mastering Python 3, Third Edition*的书籍

+   *PY99*: *模拟线程（生成器、协程和延续）* ([`mail.python.org/pipermail/python-dev/1999-July/000467.html`](https://mail.python.org/pipermail/python-dev/1999-July/000467.html))

+   *CORO-01*: *协程* ([`wiki.c2.com/?CoRoutine`](http://wiki.c2.com/?CoRoutine))

+   *CORO-02*: *生成器不是协程* ([`wiki.c2.com/?GeneratorsAreNotCoroutines`](http://wiki.c2.com/?GeneratorsAreNotCoroutines))

+   *PEP-492*: *使用 async 和 await 语法的协程* ([`www.python.org/dev/peps/pep-0492/`](https://www.python.org/dev/peps/pep-0492/))

+   *PEP-525*: *异步生成器* ([`www.python.org/dev/peps/pep-0525/`](https://www.python.org/dev/peps/pep-0525/))

+   *TEE*: *itertools.tee 函数* ([`docs.python.org/3/library/itertools.html#itertools.tee`](https://docs.python.org/3/library/itertools.html#itertools.tee))
