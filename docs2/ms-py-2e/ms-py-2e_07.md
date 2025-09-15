# 7

# 生成器和协程 – 一次一步，无限可能

生成器函数是一种通过逐个生成返回值来表现得像迭代器的函数。当传统方法构建并返回一个具有固定长度的`list`或`tuple`时，生成器只有在被调用者请求时才会`yield`单个值。副作用是，这些生成器可以无限大，因为你可以永远地持续生成。

除了生成器之外，还有对生成器语法的变体，它创建协程。协程是允许在不要求多个线程或进程的情况下进行多任务的函数。与生成器只能根据初始参数向调用者`yield`值不同，协程在运行时允许与调用函数进行双向通信。Python 中协程的现代实现是通过`asyncio`模块，这在第十三章“*asyncio – 无线程的多线程*”中有详细说明，但其基础源于本章讨论的协程。如果协程或`asyncio`适用于你的情况，它们可以提供巨大的性能提升。

在本章中，我们将涵盖以下主题：

+   生成器的优缺点

+   生成器的特性和怪癖

+   使用常规函数创建生成器

+   类似于`list`、`dict`和`set`的生成器推导式

+   使用类创建生成器

+   Python 附带生成器

+   协程的基本实现及其一些怪癖

# 生成器

生成器是一个非常有用的工具，但它们附带一些需要记住的规则。

首先，让我们来探索生成器的优势：

+   生成器通常比生成列表的函数更容易编写。你不需要声明`list`、使用`list.append(value)`和`return`，你只需要`yield value`。

+   内存使用。项目可以一次处理一个，因此通常不需要在内存中保留整个列表。

+   结果可能依赖于外部因素。而不是有一个静态的列表，你将在请求时生成值。例如，考虑处理队列/栈。

+   生成器是惰性的。这意味着如果你只使用生成器的前五个结果，其余的结果甚至不会被计算。此外，在获取项目之间，生成器是完全冻结的。

最重要的缺点是：

+   结果只能使用一次。在处理生成器的结果后，它不能再被使用。

+   大小未知。在你完成处理之前，你无法获取生成器的大小信息。它甚至可能是无限的。这使得`list(some_infinite_generator)`成为一个危险的操作。它可能会迅速崩溃你的 Python 解释器，甚至整个系统。

+   无法进行切片操作，所以`some_generator[10:20]`将不会工作。你可以使用`itertools.islice`来解决这个问题，就像你将在本章后面看到的那样，但这实际上会丢弃未使用的索引。

+   与切片类似，索引生成器也是不可能的。这意味着以下操作将不起作用：`some_generator[5]`。

现在你已经知道了可以期待什么，让我们创建一些生成器。

## 创建生成器

最简单的生成器是一个包含`yield`语句而不是`return`语句的函数。与包含`return`的常规函数的关键区别在于，你的函数中可以有多个`yield`语句。

下面是一个包含几个固定`yield`语句的生成器示例以及它在几个操作中的表现：

```py
>>> def generator():
...     yield 1
...     yield 'a'
...     yield []
...     return 'end'

>>> result = generator()

>>> result
<generator object generator at ...>

>>> len(result)
Traceback (most recent call last):
    ...
TypeError: object of type 'generator' has no len()

>>> result[:10]
Traceback (most recent call last):
    ...
TypeError: 'generator' object is not subscriptable

>>> list(result)
[1, 'a', []]

>>> list(result)
[] 
```

在这个例子中，生成器的一些缺点立即显现出来。当查看其`repr()`、获取`len()`（长度）或切片时，`result`并不提供多少有意义的 信息。而且尝试再次使用`list()`来获取值是不起作用的，因为生成器已经耗尽。

此外，你可能已经注意到函数的`return`值似乎完全消失了。实际上并非如此；`return`的值仍然被使用，但作为生成器抛出的`StopIteration`异常的值，以指示生成器已耗尽：

```py
>>> def generator_with_return():
...     yield 'some_value'
...     return 'The end of our generator'

>>> result = generator_with_return()

>>> next(result)
'some_value'
>>> next(result)
Traceback (most recent call last):
    ...
StopIteration: The end of our generator 
```

以下示例演示了生成器的惰性执行：

```py
>>> def lazy():
...     print('before the yield')
...     yield 'yielding'
...     print('after the yield')

>>> generator = lazy()

>>> next(generator)
before the yield
'yielding'

>>> next(generator)
Traceback (most recent call last):
    ...
StopIteration 
```

正如你在本例中所见，`yield`之后的代码没有执行。这是由`StopIteration`异常引起的；如果我们正确地捕获这个异常，代码将会执行：

```py
>>> def lazy():
...     print('before the yield')
...     yield 'yielding'
...     print('after the yield')

>>> generator = lazy()

>>> next(generator)
before the yield
'yielding'

>>> try:
...     next(generator)
... except StopIteration:
...     pass
after the yield

>>> for item in lazy():
...     print(item)
before the yield
yielding
after the yield 
```

要正确处理生成器，你总是需要自己捕获`StopIteration`，或者使用循环或其他隐式处理`StopIteration`的结构。

## 创建无限生成器

创建一个无限生成器（如第五章中讨论的`itertools.count`迭代器，*函数式编程 – 可读性与简洁性*）同样简单。如果我们不在函数中像上一个函数那样有固定的`yield <value>`行，而是在无限循环中`yield`，我们就可以轻松地创建一个无限生成器。

与`itertools.count()`生成器相反，我们将添加一个`stop`参数以简化测试：

```py
>>> def count(start=0, step=1, stop=None):
...     n = start
...     while stop is not None and n < stop:
...         yield n
...         n += step

>>> list(count(10, 2.5, 20))
[10, 12.5, 15.0, 17.5] 
```

由于生成器的潜在无限性，需要谨慎。如果没有`stop`变量，简单地执行`list(count())`会导致无限循环，这会很快导致内存不足的情况。

那么，这是怎么工作的呢？本质上它只是一个普通的循环，但与返回项目列表的常规方法相比，`yield`语句一次返回一个项目，这意味着你只需要计算所需的项目，而不需要将所有结果都保存在内存中。

## 包装可迭代对象的生成器

当从零开始生成值时，生成器已经非常有用，但真正的力量在于包装其他可迭代对象。为了说明这一点，我们将创建一个生成器，它会自动平方给定输入的所有数字：

```py
>>> def square(iterable):
...     for i in iterable:
...         yield i ** 2

>>> list(square(range(5)))
[0, 1, 4, 9, 16] 
```

自然，您不能阻止您在循环之外添加额外的`yield`语句：

```py
>>> def padded_square(iterable):
...     yield 'begin'
...     for i in iterable:
...         yield i ** 2
...     yield 'end'

>>> list(padded_square(range(5)))
['begin', 0, 1, 4, 9, 16, 'end'] 
```

由于这些生成器是可迭代的，您可以通过多次包装它们来将它们链接在一起。将`square()`和`odd()`生成器链接在一起的基本示例是：

```py
>>> import itertools

>>> def odd(iterable):
...     for i in iterable:
...         if i % 2:
...             yield i

>>> def square(iterable):
...     for i in iterable:
...         yield i ** 2

>>> list(square(odd(range(10))))
[1, 9, 25, 49, 81] 
```

如果我们分析代码的执行方式，我们需要从内部到外部开始：

1.  `range(10)`语句为我们生成 10 个数字。

1.  `odd()`生成器过滤输入值，所以从`[0, 1, 2 … ]`值中只返回`[1, 3, 5, 7, 9]`。

1.  `square()`函数对给定的输入进行平方，这是由`odd()`生成的奇数列表。

链接的真正力量在于，生成器只有在请求值时才会执行操作。如果我们用`next()`而不是`list()`请求单个值，这意味着只有`square()`中的第一个循环迭代会被运行。然而，对于`odd()`和`range()`，它必须处理两个值，因为`odd()`会丢弃`range()`给出的第一个值，并且不会`yield`任何内容。

## 生成器推导式

在前面的章节中，您已经看到了`list`、`dict`和`set`的推导式，它们可以生成集合。使用生成器推导式，我们可以创建类似的集合，但使它们变得懒加载，这样它们只会在需要时才被评估。基本前提与`list`推导式相同，但使用圆括号/括号而不是方括号：

```py
>>> squares = (x ** 2 for x in range(4))

>>> squares
<generator object <genexpr> at 0x...>

>>> list(squares)
[0, 1, 4, 9] 
```

当您需要包装不同生成器的结果时，这非常有用，因为它只计算您请求的值：

```py
>>> import itertools

>>> result = itertools.count()
>>> odd = (x for x in result if x % 2)
>>> sliced_odd = itertools.islice(odd, 5)
>>> list(sliced_odd)
[1, 3, 5, 7, 9]

>>> result = itertools.count()
>>> sliced_result = itertools.islice(result, 5)
>>> odd = (x for x in sliced_result if x % 2)
>>> list(odd)
[1, 3] 
```

您可能已经从结果中推断出，这对于无限大小的生成器，如`itertools.count()`，可能是危险的。操作顺序非常重要，因为`itertools.islice()`函数在该点切片结果，而不是原始生成器。这意味着如果我们用永远不会对给定集合求值为`True`的函数替换`odd()`，它将永远运行，因为它永远不会`yield`任何结果。

## 基于类的生成器和迭代器

除了通过常规函数和生成器推导式创建生成器之外，我们还可以使用类来创建生成器。这对于需要记住状态或可以使用继承的更复杂的生成器来说是有益的。

首先，让我们看看创建一个基本的生成器`class`的示例，该类模仿了`itertools.count()`的行为，并添加了`stop`参数：

```py
>>> class CountGenerator:
...     def __init__(self, start=0, step=1, stop=None):
...         self.start = start
...         self.step = step
...         self.stop = stop
...
...     def __iter__(self):
...         i = self.start
...         while self.stop is None or i < self.stop:
...             yield i
...             i += self.step

>>> list(CountGenerator(start=2.5, step=0.5, stop=5))
[2.5, 3.0, 3.5, 4.0, 4.5] 
```

现在，让我们将生成器类转换为具有更多功能的迭代器：

```py
>>> class CountIterator:
...     def __init__(self, start=0, step=1, stop=None):
...         self.i = start
...         self.start = start
...         self.step = step
...         self.stop = stop
...
...     def __iter__(self):
...         return self
...
...     def __next__(self):
...         if self.stop is not None and self.i >= self.stop:
...             raise StopIteration
...
...         # We need to return the value before we increment to
...         # maintain identical behavior
...         value = self.i
...         self.i += self.step
...         return value

>>> list(CountIterator(start=2.5, step=0.5, stop=5))
[2.5, 3.0, 3.5, 4.0, 4.5] 
```

生成器和迭代器之间最重要的区别是，我们现在有一个完整的类，它充当迭代器，这意味着我们也可以将其扩展到常规生成器的功能之外。

正常生成器的一些限制是它们没有长度，我们无法对它们进行切片。使用迭代器，我们可以在需要的情况下显式定义这些场景的行为：

```py
>>> import itertools

>>> class AdvancedCountIterator:
...     def __init__(self, start=0, step=1, stop=None):
...         self.i = start
...         self.start = start
...         self.step = step
...         self.stop = stop
...
...     def __iter__(self):
...         return self
...
...     def __next__(self):
...         if self.stop is not None and self.i >= self.stop:
...             raise StopIteration
...
...         value = self.i
...         self.i += self.step
...         return value
...
...     def __len__(self):
...         return int((self.stop - self.start) // self.step)
...
...     def __contains__(self, key):
...         # To check 'if 123 in count'.
...         # Note that this does not look at 'step'!
...         return self.start < key < self.stop
...
...     def __repr__(self):
...         return (
...             f'{self.__class__.__name__}(start={self.start}, '
...             f'step={self.step}, stop={self.stop})')
...
...     def __getitem__(self, slice_):
...         return itertools.islice(self, slice_.start,
...                                 slice_.stop, slice_.step) 
```

现在我们有了支持 `len()`、`in` 和 `repr()` 等功能的先进计数迭代器，我们可以测试它是否按预期工作：

```py
>>> count = AdvancedCountIterator(start=2.5, step=0.5, stop=5)

# Pretty representation using '__repr__'
>>> count
AdvancedCountIterator(start=2.5, step=0.5, stop=5)

# Check if item exists using '__contains__'
>>> 3 in count
True
>>> 3.1 in count
True
>>> 1 in count
False

# Getting the length using '__len__'
>>> len(count)
5
# Slicing using '__getitem__' with a slice as a parameter
>>> count[:3]
<itertools.islice object at 0x...>

>>> list(count[:3])
[2.5, 3.0, 3.5]

>>> list(count[:3])
[4.0, 4.5] 
```

除了解决一些限制之外，在最后一个示例中，您还可以看到生成器的一个非常有用的功能。我们可以逐个耗尽项目，并随时停止/开始。由于我们仍然可以完全访问该对象，我们可以更改 `count.i` 来重新启动迭代器。

# 生成器示例

现在您已经知道了如何创建生成器，让我们看看一些有用的生成器和它们的使用示例。

在您开始为项目编写生成器之前，请务必查看 Python 的 `itertools` 模块。它包含大量有用的生成器，涵盖了广泛的使用案例。以下部分展示了几个自定义生成器和标准库中最有用的生成器。

这些生成器适用于所有可迭代对象，而不仅仅是生成器。因此，您也可以将它们应用于 `list`、`tuple`、`string` 或其他类型的可迭代对象。

## 将可迭代对象拆分成块/组

当在数据库中执行大量查询或在多个进程中运行任务时，通常更高效的做法是将操作分块。单个巨大的操作可能会导致内存不足的问题；由于启动/拆除序列，许多微小的操作可能会很慢。

为了提高效率，一个很好的方法是按块拆分输入。Python 文档([`docs.python.org/3/library/itertools.html?highlight=chunk#itertools-recipes`](https://docs.python.org/3/library/itertools.html?highlight=chunk#itertools-recipes))已经提供了一个使用 `itertools.zip_longest()` 来实现此操作的示例：

```py
>>> import itertools

>>> def grouper(iterable, n, fillvalue=None):
...     '''Collect data into fixed-length chunks or blocks'''
...     args = [iter(iterable)] * n 
...     return itertools.zip_longest(*args, fillvalue=fillvalue)

>>> list(grouper('ABCDEFG', 3, 'x'))
[('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')] 
```

这段代码是一个很好的例子，说明了如何轻松地将数据分块，但它必须将整个块保留在内存中。为了解决这个问题，我们可以创建一个版本，为块生成子生成器：

```py
>>> def chunker(iterable, chunk_size):
...     # Make sure 'iterable' is an iterator
...     iterable = iter(iterable)
...
...     def chunk(value):
...         # Make sure not to skip the given value
...         yield value
...         # We already yielded a value so reduce the chunk_size
...         for _ in range(chunk_size - 1):
...             try:
...                 yield next(iterable)
...             except StopIteration:
...                 break
...
...     while True:
...         try:
...             # Check if we're at the end by using 'next()'
...             yield chunk(next(iterable))
...         except StopIteration:
...             break

>>> for chunk in chunker('ABCDEFG', 3):
...     for value in chunk:
...         print(value, end=', ')
...     print()
A, B, C,
D, E, F,
G, 
```

由于我们需要捕获 `StopIteration` 异常，这个例子在我看来并不好看。部分代码可以通过使用 `itertools.islice()`（将在下一部分介绍）来改进，但这仍然会留下我们无法知道何时达到末尾的问题。

如果您感兴趣，可以使用 `itertools.islice()` 和 `itertools.chains()` 在本书的 GitHub 上找到的实现：[`github.com/mastering-python/code_2`](https://github.com/mastering-python/code_2)。

## itertools.islice – 可迭代对象的切片

生成器的一个限制是它们不能被切片。您可以通过在切片之前将生成器转换为 `list` 来解决这个问题，但对于无限生成器来说这是不可能的，如果您只需要几个值，这可能会很低效。

为了解决这个问题，`itertools`库有一个`islice()`函数，它可以切片任何可迭代对象。该函数是切片操作符的生成器版本，类似于切片，支持`start`、`stop`和`step`参数。以下说明了常规切片和`itertools.islice()`的比较：

```py
>>> import itertools

>>> some_list = list(range(1000))
>>> some_list[:5]
[0, 1, 2, 3, 4]
>>> list(itertools.islice(some_list, 5))
[0, 1, 2, 3, 4]

>>> some_list[10:20:2]
[10, 12, 14, 16, 18]
>>> list(itertools.islice(some_list, 10, 20, 2))
[10, 12, 14, 16, 18] 
```

需要注意的是，尽管输出是相同的，但这些方法在内部并不等价。常规切片仅适用于可切片的对象；实际上，这意味着对象必须实现`__getitem__(self, slice)`方法。

此外，我们期望切片对象是一个快速且高效的操作。对于`list`和`tuple`来说，这当然是对的，但对于给定的生成器来说可能并非如此。

如果对于大小为`n=1000`的列表，我们取任何`k=10`个元素的切片，我们可以期望其时间复杂度仅为`O(k)`；也就是说，10 步。我们做`some_list[:10]`或`some_list[900:920:2]`都无关紧要。

对于`itertools.islice()`来说，情况并非如此，因为它所做的唯一假设是输入是可迭代的。这意味着获取前 10 个元素很容易；只需遍历元素，返回前 10 个，然后停止。因此，`itertools.islice(some_list, 10)`也需要 10 步。然而，获取第 900 到第 920 个元素意味着需要遍历并丢弃前 900 个元素，然后只返回接下来的 20 个元素中的 10 个。因此，这是 920 步。

为了说明这一点，这里有一个对`itertools.islice()`的略微简化的实现，它期望始终有一个`stop`可用：

```py
>>> def islice(iterable, start, stop=None, step=1):
...     # 'islice' has signatures: 'islice(iterable, stop)' and:
...     # 'islice(iterable, start, stop[, step])'
...     # 'fill' stop with 'start' if needed
...     if stop is None and step == 1 and start is not None:
...         start, stop = 0, start
...
...     # Create an iterator and discard the first 'start' items
...     iterator = iter(iterable)
...     for _ in range(start):
...         next(iterator)
...
...     # Enumerate the iterator making 'i' start at 'start'
...     for i, item in enumerate(iterator, start):
...         # Stop when we've reached 'stop' items
...         if i >= stop:
...             return
...         # Use modulo 'step' to discard non-matching items
...         if i % step:
...             continue
...         yield item

>>> list(islice(range(1000), 10))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

>>> list(islice(range(1000), 900, 920, 2))
[900, 902, 904, 906, 908, 910, 912, 914, 916, 918]

>>> list(islice(range(1000), 900, 910))
[900, 901, 902, 903, 904, 905, 906, 907, 908, 909] 
```

如您所见，`start`和`step`部分都丢弃了不需要的元素。这并不意味着您不应该使用`itertools.islice()`，但要注意其内部机制。同样，如您所预期的那样，这个生成器不支持索引为负值，并期望所有值都是正数。

## itertools.chain – 连接多个可迭代对象

`itertools.chain()`生成器是 Python 库中最简单但最有用的生成器之一。它简单地按顺序返回每个传递的可迭代对象的每个元素，并且可以用仅仅三行代码实现：

```py
>>> def chain(*iterables):
...     for iterable in iterables:
...         yield from iterable

>>> a = 1, 2, 3
>>> b = [4, 5, 6]
>>> c = 'abc'
>>> list(chain(a, b, c))
[1, 2, 3, 4, 5, 6, 'a', 'b', 'c']

>>> a + b + c
Traceback (most recent call last):
    ...
TypeError: can only concatenate tuple (not "list") to tuple 
```

如您可能注意到的，这也引入了一个尚未讨论的功能：`yield from`表达式。`yield from`确实如您从其名称中可以预期的那样，从给定的可迭代对象中产生所有元素。因此，`itertools.chain()`也可以用稍微冗长的形式替换：

```py
>>> def chain(*iterables):
...     for iterable in iterables:
...         for i in iterable:
...             yield i 
```

有趣的是，这种方法比添加集合更强大，因为它不关心类型，只要它们是可迭代的——这就是鸭子类型的最纯粹形式。

## itertools.tee – 使用输出多次

如前所述，生成器最大的缺点之一是结果只能使用一次。幸运的是，Python 有一个函数允许你将输出复制到多个生成器。如果你习惯于在 Linux/Unix 命令行 shell 中工作，`tee`这个名字可能对你来说很熟悉。`tee`程序允许你将输出写入屏幕和文件，这样你就可以在保持实时查看的同时存储输出。

Python 版本中的`itertools.tee()`做类似的事情，但它返回几个迭代器，允许你分别处理结果。

默认情况下，`tee`会将你的生成器拆分为一个包含两个不同生成器的元组，这就是为什么在这里使用元组解包工作得很好。通过传递`n`参数，你可以告诉`itertools.tee()`创建超过两个生成器。以下是一个示例：

```py
>>> import itertools

>>> def spam_and_eggs():
...     yield 'spam'
...     yield 'eggs'

>>> a, b = itertools.tee(spam_and_eggs())
>>> next(a)
'spam'
>>> next(a)
'eggs'
>>> next(b)
'spam'
>>> next(b)
'eggs'
>>> next(b)
Traceback (most recent call last):
    ...
StopIteration 
```

在看到这段代码后，你可能会对`tee`的内存使用情况感到好奇。它是否需要为你存储整个列表？幸运的是，不需要。`tee`函数在处理这个问题上相当聪明。假设你有一个包含 1,000 个元素的生成器，你同时从`a`中读取前 100 个元素和从`b`中读取前 75 个元素。然后`tee`将只保留差异（`100 - 75 = 25`个元素）在内存中，并在迭代结果时丢弃其余部分。

是否`tee`是你情况下的最佳解决方案取决于情况。如果实例`a`在读取实例`b`之前（几乎）从开始到结束都被读取，那么使用`tee`可能不是一个好主意。简单地将生成器转换为`list`会更快，因为它涉及的操作要少得多。

## contextlib.contextmanager – 创建上下文管理器

你已经在*第五章*，*函数式编程 – 可读性与简洁性之间的权衡*和*第六章*，*装饰器 – 通过装饰实现代码重用*中看到了上下文管理器，但还有许多其他有用的东西可以用上下文管理器来完成。虽然`contextlib.contextmanager()`生成器并不是像本章前面看到的例子那样用于生成结果的生成器，但它确实使用了`yield`，所以它是一个非标准生成器使用的良好示例。

一些有用的示例是将输出记录到文件并测量函数执行时间：

```py
>>> import time
>>> import datetime
>>> import contextlib

# Context manager that shows how long a context was active
>>> @contextlib.contextmanager
... def timer(name):
...     start_time = datetime.datetime.now()
...     yield
...     stop_time = datetime.datetime.now()
...     print('%s took %s' % (name, stop_time - start_time))

>>> with timer('basic timer'):
...     time.sleep(0.1)
basic timer took 0:00:00.1...

# Write standard print output to a file temporarily
>>> @contextlib.contextmanager
... def write_to_log(name):
...     with open(f'{name}.txt', 'w') as fh:
...         with contextlib.redirect_stdout(fh):
...             with timer(name):
...                 yield

# Using as a decorator also works in addition to with-statements
>>> @write_to_log('some_name')
... def some_function():
...     print('This will be written to 'some_name.txt'')

>>> some_function() 
```

所有这些都工作得很好，但代码可以更漂亮。有三个级别的上下文管理器往往难以阅读，这通常可以通过装饰器来解决，如*第六章*中所述。然而，在这种情况下，我们需要一个上下文管理器的输出作为下一个上下文管理器的输入，这将使装饰器设置更加复杂。

这就是`ExitStack`上下文管理器发挥作用的地方。它允许轻松组合多个上下文管理器，而不会增加缩进级别：

```py
>>> import contextlib

>>> @contextlib.contextmanager
... def write_to_log(name):
...     with contextlib.ExitStack() as stack:
...         fh = stack.enter_context(open(f'{name}.txt', 'w'))
...         stack.enter_context(contextlib.redirect_stdout(fh))
...         stack.enter_context(timer(name))
...         yield

>>> @write_to_log('some_name')
... def some_function():
...     print('This will be written to 'some_name.txt'')

>>> some_function() 
```

看起来简单一些，不是吗？虽然这个例子在没有 `ExitStack` 上下文管理器的情况下仍然相当易于阅读，但当需要执行特定的清理操作时，`ExitStack` 的便利性很快就会变得明显。除了之前看到的自动处理之外，还可以将上下文转移到新的 `ExitStack` 中以手动处理关闭：

```py
>>> import contextlib

>>> with contextlib.ExitStack() as stack:
...     fh = stack.enter_context(open('file.txt', 'w'))
...     # Move the context(s) to a new ExitStack
...     new_stack = stack.pop_all()

>>> bytes_written = fh.write('fh is still open')

# After closing we can't write anymore
>>> new_stack.close()
>>> fh.write('cant write anymore')
Traceback (most recent call last):
    ...
ValueError: I/O operation on closed file. 
```

大多数 `contextlib` 函数在 Python 手册中都有详细的文档。特别是 `ExitStack`，它使用许多示例进行了文档说明，可以在 [`docs.python.org/3/library/contextlib.html#contextlib.ExitStack`](https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack) 找到。我建议关注 `contextlib` 文档，因为它随着每个 Python 版本的更新而不断改进。

现在我们已经涵盖了常规生成器，是时候继续介绍协程了。

# 协程

协程是通过多个入口点提供非抢占式多任务处理的子程序。基本前提是协程允许两个函数在单个线程中运行时相互通信。通常，这种类型的通信仅限于多任务或多线程解决方案，但协程提供了一种相对简单的方法来实现这一点，而几乎不需要额外的性能开销。

由于生成器默认是惰性的，你可能能够猜到协程是如何工作的。直到结果被消费，生成器会处于休眠状态；但在消费结果时，生成器变得活跃。常规生成器和协程之间的区别在于，使用协程时，通信是双向的；协程不仅可以接收值，还可以将值 `yield` 给调用函数。

如果你熟悉 `asyncio`，你可能会注意到 `asyncio` 和协程之间有很强的相似性。这是因为 `asyncio` 是基于协程的概念构建的，并且已经从一点语法糖发展成为一个完整的生态系统。出于实用目的，我建议使用 `asyncio` 而不是这里解释的协程语法；然而，出于教育目的，了解它们是如何工作的非常有用。`asyncio` 模块正在非常活跃地开发中，并且拥有一个不那么笨拙的语法。

## 一个基本示例

在前面的章节中，你看到了常规生成器如何 `yield` 值。但生成器可以做更多；它们实际上可以通过 `yield` 接收值。基本用法相当简单：

```py
>>> def generator():
...     value = yield 'value from generator'
...     print('Generator received:', value)
...     yield f'Previous value: {value!r}'

>>> g = generator()
>>> print('Result from generator:', next(g))
Result from generator: value from generator

>>> print(g.send('value from caller'))
Generator received: value from caller
Previous value: 'value from caller' 
```

就这些了。函数会在 `send` 方法被调用之前保持冻结状态，此时它将处理到下一个 `yield` 语句。从这个限制中你可以看到一个限制是协程不能自己醒来。值交换只能在调用代码运行 `next(generator)` 或 `generator.send()` 时发生。

## 预激

由于生成器是惰性的，你不能直接向一个全新的生成器发送一个值。在值可以发送到生成器之前，必须使用`next()`获取结果，或者发出`send(None)`以便代码实际上能够执行。这是可以理解的，但有时会有些繁琐。让我们创建一个简单的装饰器来省略这个需求：

```py
>>> import functools

>>> def coroutine(function):
...     # Copy the 'function' description with 'functools.wraps'
...     @functools.wraps(function)
...     def _coroutine(*args, **kwargs):
...         active_coroutine = function(*args, **kwargs)
...         # Prime the coroutine and make sure we get no values
...         assert not next(active_coroutine)
...         return active_coroutine
...
...     return _coroutine

>>> @coroutine
... def our_coroutine():
...     while True:
...         print('Waiting for yield...')
...         value = yield
...         print('our coroutine received:', value)

>>> generator = our_coroutine()
Waiting for yield...

>>> generator.send('a')
our coroutine received: a
Waiting for yield... 
```

如你所注意到的，尽管生成器仍然是惰性的，但它现在会自动执行所有代码，直到再次遇到`yield`语句。到那时，它将保持休眠状态，直到发送新的值。

注意，从现在开始，本章将一直使用`coroutine`装饰器。为了简洁，以下示例中将省略协程函数的定义。

## 关闭和抛出异常

与常规生成器不同，一旦输入序列耗尽，生成器就会立即退出，而协程通常使用无限`while`循环，这意味着它们不会以常规方式被销毁。这就是为什么协程也支持`close`和`throw`方法，这些方法可以退出函数。这里重要的是关闭的可能性，而不是关闭本身。本质上，这与上下文包装器使用`__enter__`和`__exit__`方法非常相似，但在这个情况下是协程。

以下示例展示了使用上一段中的`coroutine`装饰器的协程，包括正常和异常退出情况：

```py
>>> from coroutine_decorator import coroutine

>>> @coroutine
... def simple_coroutine():
...     print('Setting up the coroutine')
...     try:
...         while True:
...             item = yield
...             print('Got item:', item)
...     except GeneratorExit:
...         print('Normal exit')
...     except Exception as e:
...         print('Exception exit:', e)
...         raise
...     finally:
...         print('Any exit') 
```

这个`simple_coroutine()`函数可以向我们展示协程的一些内部流程以及它们是如何被中断的。特别是`try`/`finally`行为可能会让你感到惊讶：

```py
>>> active_coroutine = simple_coroutine()
Setting up the coroutine
>>> active_coroutine.send('from caller')
Got item: from caller
>>> active_coroutine.close()
Normal exit
Any exit

>>> active_coroutine = simple_coroutine()
Setting up the coroutine
>>> active_coroutine.throw(RuntimeError, 'caller sent an error')
Traceback (most recent call last):
    ...
RuntimeError: caller sent an error

>>> active_coroutine = simple_coroutine()
Setting up the coroutine
>>> try:
...     active_coroutine.throw(RuntimeError, 'caller sent an error')
... except RuntimeError as exception:
...     print('Exception:', exception)
Exception exit: caller sent an error
Any exit
Exception: caller sent an error 
```

大部分输出都是你预期的，但就像生成器中的`StopIteration`一样，你必须捕获异常以确保正确处理清理。

## 混合生成器和协程

尽管生成器和协程由于`yield`语句而看起来非常相似，但它们实际上是两种不同的生物。让我们创建一个双向管道来处理给定的输入，并在过程中传递给多个协程：

```py
# The decorator from the Priming section in this chapter
>>> from coroutine_decorator import coroutine

>>> lines = 'some old text', 'really really old', 'old old old'

>>> @coroutine
... def replace(search, replace):
...     while True:
...         item = yield
...         print(item.replace(search, replace))

>>> old_replace = replace('old', 'new')
>>> for line in lines:
...     old_replace.send(line)
some new text
really really new
new new new 
```

给定这个示例，你可能想知道为什么我们现在打印值而不是产生它。我们可以`yield`这个值，但记住，生成器在产生值之前会冻结。让我们看看如果我们简单地`yield`值而不是调用`print`会发生什么。默认情况下，你可能会倾向于这样做：

```py
>>> @coroutine
... def replace(search, replace):
...     while True:
...         item = yield
...         yield item.replace(search, replace)

>>> old_replace = replace('old', 'new')
>>> for line in lines:
...     old_replace.send(line)
'some new text'
'new new new' 
```

现在已经消失了一半的值；我们的“`really really new`"行已经消失了。注意第二个`yield`没有存储结果，并且`yield`实际上使这个函数成为一个生成器而不是协程。我们需要从那个`yield`存储结果：

```py
>>> @coroutine
... def replace(search, replace):
...     item = yield
...     while True:
...         item = yield item.replace(search, replace)

>>> old_replace = replace('old', 'new')
>>> for line in lines:
...     old_replace.send(line)
'some new text'
'really really new'
'new new new' 
```

但这还远远不够优化。我们实际上正在使用协程来模仿生成器的行为。它确实有效，但有点多余，并且没有带来真正的益处。

让我们这次创建一个真正的管道，其中协程将数据发送到下一个协程或协程。这展示了协程的真正力量，即能够将多个协程链接在一起：

```py
>>> @coroutine
... def replace(target, search, replace):
...     while True:
...         target.send((yield).replace(search, replace))

# Print will print the items using the provided formatstring
>>> @coroutine
... def print_(formatstring):
...     count = 0
...     while True:
...         count += 1
...         print(count, formatstring.format((yield)))
# tee multiplexes the items to multiple targets
>>> @coroutine
... def tee(*targets):
...     while True:
...         item = yield
...         for target in targets:
...             target.send(item) 
```

现在我们有了我们的协程函数，让我们看看我们如何将这些函数链接在一起：

```py
# Because we wrap the results we need to work backwards from the
# inner layer to the outer layer.

# First, create a printer for the items:
>>> printer = print_('print: {}')

# Create replacers that send the output to the printer
>>> old_replace = replace(printer, 'old', 'new')
>>> current_replace = replace(printer, 'old', 'current')

# Send the input to both replacers
>>> branch = tee(old_replace, current_replace)

# Send the data to the tee routine for processing
>>> for line in lines:
...     branch.send(line)
1 print: some new text
2 print: some current text
3 print: really really new
4 print: really really current
5 print: new new new
6 print: current current current 
```

这使代码更加简单和易于阅读，并展示了如何将单个输入源同时发送到多个目的地。乍一看，这个例子并不那么令人兴奋，但令人兴奋的部分是，尽管我们使用 `tee()` 分割了输入并通过两个独立的 `replace()` 实例进行处理，但我们最终仍然到达了具有相同状态的同一个 `print_()` 函数。这意味着你可以根据你的方便来路由和修改你的数据，同时几乎不需要任何努力就能到达同一个终点。

目前来说，最重要的收获是，在大多数情况下，混合生成器和协程不是一个好主意，因为如果使用不当，可能会产生非常奇怪的副作用。尽管两者都使用 `yield` 语句，但它们是显著不同的实体，具有不同的行为。下一节将演示混合协程和生成器可以有用的一小部分情况。

## 使用状态

现在你已经知道了如何编写基本的协程以及需要注意哪些陷阱，那么写一个需要记住状态的函数怎么样？也就是说，一个总是给你所有发送值的平均值的函数。这是少数几种仍然相对安全和有用的结合协程和生成器语法的情况之一：

```py
>>> import itertools

>>> @coroutine
... def average():
...     total = yield
...     for count in itertools.count(start=1):
...         total += yield total / count

>>> averager = average()
>>> averager.send(20)
20.0
>>> averager.send(10)
15.0 
```

尽管如此，它仍然需要一些额外的逻辑才能正常工作。我们需要使用 `yield` 来初始化我们的协程，但在那个时刻我们不发送任何数据，因为第一个 `yield` 是初始化器，在我们得到值之前执行。一旦一切准备就绪，我们就可以轻松地在求和的同时产生平均值。这并不那么糟糕，但纯协程版本稍微简单一些，因为我们只有一个执行路径，因为我们不必担心初始化。为了说明这一点，这里是有纯协程版本：

```py
>>> import itertools

>>> @coroutine
... def print_(formatstring):
...     while True:
...         print(formatstring.format((yield)))

>>> @coroutine
... def average(target):
...     total = 0
...     for count in itertools.count(start=1):
...         total += yield
...         target.send(total / count)

>>> printer = print_('{:.1f}')
>>> averager = average(printer)
>>> averager.send(20)
20.0
>>> averager.send(10)
15.0 
```

虽然这个例子比包含生成器的版本多几行，但它更容易理解。让我们分析它以确保工作原理清晰：

1.  我们将 `total` 设置为 `0` 以开始计数。

1.  我们通过使用 `itertools.count()` 来跟踪测量次数，我们将其配置为从 1 开始计数。

1.  我们使用 `yield` 来获取下一个值。

1.  我们将平均值发送给给定的协程，而不是返回值，以使代码更易于理解。

另一个很好的例子是 `itertools.groupby`，它也相当简单，可以使用协程来重新创建。为了比较，我将继续展示生成器协程和纯协程版本：

```py
>>> @coroutine
... def groupby():
...     # Fetch the first key and value and initialize the state
...     # variables
...     key, value = yield
...     old_key, values = key, []
...     while True:
...         # Store the previous value so we can store it in the
...         # list
...         old_value = value
...         if key == old_key:
...             key, value = yield
...         else:
...             key, value = yield old_key, values
...             old_key, values = key, []
...         values.append(old_value)

>>> grouper = groupby()
>>> grouper.send('a1')
>>> grouper.send('a2')
>>> grouper.send('a3')
>>> grouper.send('b1')
('a', ['1', '2', '3'])
>>> grouper.send('b2')
>>> grouper.send('a1')
('b', ['1', '2'])
>>> grouper.send('a2')
>>> grouper.send((None, None))
('a', ['1', '2']) 
```

如您所见，这个函数使用了一些技巧。首先，我们存储了之前的`key`和`value`，这样我们就可以检测到组（`key`）何时发生变化。其次，我们显然不能在组发生变化之前识别一个组，因此只有在组发生变化之后才会返回结果。这意味着只有当在最后一个组之后发送了不同的组时，才会发送最后一个组，这就是为什么有`(None, None)`。

示例使用字符串的元组解包，将`'a1'`拆分为组`'a'`和值`'1'`。或者，您也可以使用`grouper.send(('a', 1))`。

现在是纯协程版本：

```py
>>> @coroutine
... def print_(formatstring):
...     while True:
...         print(formatstring.format(*(yield)))

>>> @coroutine
... def groupby(target):
...     old_key = None
...     while True:
...         key, value = yield
...         if old_key != key:
...             # A different key means a new group so send the
...             # previous group and restart the cycle.
...             if old_key and values:
...                 target.send((old_key, values))
...             values = []
...             old_key = key
...         values.append(value)

>>> grouper = groupby(print_('group: {}, values: {}'))
>>> grouper.send('a1')
>>> grouper.send('a2')
>>> grouper.send('a3')
>>> grouper.send('b1')
group: a, values: ['1', '2', '3']
>>> grouper.send('b2')
>>> grouper.send('a1')
group: b, values: ['1', '2']
>>> grouper.send('a2')
>>> grouper.send((None, None))
group: a, values: ['1', '2'] 
```

虽然这些函数相当相似，但协程版本的控制路径更简单，只需要在一个地方`yield`。这是因为我们不必考虑初始化和可能丢失值的问题。

# 练习

生成器有各种各样的用途，您可能可以直接在自己的代码中使用它们。尽管如此，以下练习可能有助于您更好地理解其特性和局限性：

+   创建一个类似于`itertools.islice()`的生成器，允许使用负步长，以便您可以执行`some_list[20:10:-1]`。

+   创建一个类，它包装一个生成器，使其可以通过内部使用`itertools.islice()`来切片。

+   编写一个生成斐波那契数的生成器。

+   编写一个使用欧几里得筛法生成素数的生成器。

这些练习的示例答案可以在 GitHub 上找到：[`github.com/mastering-python/exercises`](https://github.com/mastering-python/exercises)。鼓励您提交自己的解决方案，并从他人的替代方案中学习。

# 摘要

本章向您展示了如何创建生成器以及它们所具有的优缺点。此外，现在应该很清楚如何克服它们的局限性及其影响。

通常，我总是推荐使用生成器而不是传统的集合生成函数。它们更容易编写，消耗的内存更少，如果需要，可以通过将`some_generator()`替换为`list(some_generator())`或一个为您处理该问题的装饰器来减轻其缺点。

虽然关于协程的段落提供了一些关于它们是什么以及如何使用的见解，但这只是对协程的温和介绍。纯协程和协程生成器组合仍然有些笨拙，这就是为什么创建了`asyncio`库。第十三章，“*asyncio – 无线程的多线程*”，详细介绍了`asyncio`，并介绍了`async`和`await`语句，这使得协程的使用比`yield`更加直观。

在上一章中，你看到了我们如何使用类装饰器来修改类。在下一章中，我们将介绍如何使用元类来创建类。使用元类，你可以在创建类本身的过程中修改类。请注意，我所说的不是类的实例，而是实际的类对象。使用这种技术，你可以创建自动注册的插件系统，向类添加额外的属性，等等。

# 加入我们的 Discord 社区

加入我们的 Discord 空间，与作者和其他读者进行讨论：[`discord.gg/QMzJenHuJf`](https://discord.gg/QMzJenHuJf)

![](img/QR_Code156081100001293319171.png)
