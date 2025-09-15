# 6

# 装饰器 – 通过装饰实现代码重用

在本章中，你将学习关于 Python 装饰器的内容。前几章已经展示了几个装饰器的用法，但现在你将了解更多关于它们的信息。装饰器本质上是对函数/类的包装，可以在执行之前修改输入、输出，甚至函数/类本身。这种包装同样可以通过拥有一个调用内部函数的单独函数，或者通过继承通常称为 **mixins** 的小功能类来实现。与许多 Python 构造一样，装饰器不是达到目标的唯一方法，但在许多情况下确实很方便。

虽然你可以不用太多了解装饰器就能过得很好，但它们提供了大量的“重用能力”，因此在框架库（如 Web 框架）中得到了广泛使用。Python 实际上附带了一些有用的装饰器，最著名的是 `@property`、`@classmethod` 和 `@staticmethod` 装饰器。

然而，有一些特定的注意事项需要注意：包装一个函数会创建一个新的函数，并使得访问内部函数及其属性变得更加困难。一个例子是 Python 的 `help(function)` 功能；默认情况下，你、你的编辑器和你的文档生成器可能会丢失函数属性，如帮助文本和函数所在的模块。

本章将涵盖函数和类装饰器的用法，以及装饰类内函数时需要了解的复杂细节。

以下是一些涵盖的主题：

+   装饰函数

+   装饰类函数

+   装饰类

+   Python 标准库中的有用装饰器

# 装饰函数

装饰器是包装其他函数和/或类的函数或类。在其最基本的形式中，你可以将常规函数调用视为 `add(1, 2)`，当应用装饰器时，它将转换为 `decorator(add(1, 2))`。这还有更多内容，但我们会稍后讨论。让我们来实现这个 `decorator()` 函数：

```py
>>> def decorator(function):
...     return function

>>> def add(a, b):
...     return a + b

>>> add = decorator(add) 
```

为了使语法更容易使用，Python 为这种情况提供了一种特殊的语法。因此，你可以在函数下方添加一行，如前面的示例，而不是使用 `@` 操作符作为快捷方式来装饰一个函数：

```py
>>> @decorator
... def add(a, b):
...     return a + b 
```

这个例子展示了最简单且最无用的装饰器：简单地返回输入函数而不做其他任何事情。

从这个例子中，你可能会想知道装饰器的用途以及它们有什么特别之处。装饰器的一些可能性包括：

+   注册函数/类

+   修改函数/类输入

+   修改函数/类输出

+   记录函数调用/类实例化

所有这些内容将在本章的后续部分进行介绍，但现在我们先从简单开始。

我们的第一个装饰器将展示我们如何修改函数调用的输入和输出。此外，它还添加了一些 `logging` 调用，以便我们可以看到发生了什么：

```py
>>> import functools

>>> def decorator(function):
...    # This decorator makes sure we mimic the wrapped function
...    @functools.wraps(function)
...    def _decorator(a, b):
...        # Pass the modified arguments to the function
...        result = function(a, b + 5)
...
...        # Log the function call
...        name = function.__name__
...        print(f'{name}(a={a}, b={b}): {result}')
...
...        # Return a modified result
...        return result + 4
...
...    return _decorator

>>> @decorator
... def func(a, b):
...     return a + b

>>> func(1, 2)
func(a=1, b=2): 8
12 
```

这应该能展示出装饰器的强大之处。我们可以修改、添加和/或删除参数。我们可以修改返回值，或者如果我们想的话，甚至可以调用一个完全不同的函数。而且，如果需要，我们可以轻松地记录所有行为，这在调试时非常有用。我们可以返回与 `return function(...)` 完全不同的内容。

更多关于如何使用装饰器进行日志记录的示例，请参阅第十二章，“调试 – 解决错误”。

## 通用函数装饰器

我们之前编写的装饰器明确使用了 `a` 和 `b` 参数，因此它仅适用于具有非常类似接受 `a` 和 `b` 参数签名的函数。如果我们想使生成器更通用，我们可以将 `a, b` 替换为 `*args` 和 `**kwargs` 来分别获取位置参数和关键字参数。然而，这引入了一个新问题。我们需要确保只使用常规参数或关键字参数，否则检查将变得越来越困难：

```py
>>> import functools

>>> def decorator(function):
...    @functools.wraps(function)
...    def _decorator(*args, **kwargs):
...        a, b = args
...        return function(a, b + 5)
...
...    return _decorator

>>> @decorator
... def func(a, b):
...     return a + b

>>> func(1, 2)
8

>>> func(a=1, b=2)
Traceback (most recent call last):
...
ValueError: not enough values to unpack (expected 2, got 0) 
```

如所示，在这种情况下，关键字参数被破坏。为了解决这个问题，我们有几种不同的方法。我们可以将参数更改为仅位置参数或仅关键字参数：

此代码使用仅位置参数（`/` 作为最后一个函数参数），这自 Python 3.8 起已被支持。对于旧版本，你可以使用 `*args` 而不是显式参数来模拟此行为。

```py
>>> def add(a, b, /):
...     return a + b

>>> add(a=1, b=2)
Traceback (most recent call last):
...
TypeError: add() got some positional-only arguments passed ...

>>> def add(*, a, b):
...     return a + b

>>> add(1, 2)
Traceback (most recent call last):
...
TypeError: add() takes 0 positional arguments but 2 were given 
```

或者，我们可以让 Python 自动处理这个问题，通过获取签名并将其绑定到给定的参数：

```py
>>> import inspect
>>> import functools

>>> def decorator(function):
...    # Use the inspect module to get function signature. More
...    # about this in the logging chapter
...    signature = inspect.signature(function)
... 
...    @functools.wraps(function)
...    def _decorator(*args, **kwargs):
...        # Bind the arguments to the given *args and **kwargs.
...        # If you want to make arguments optional, use
...        # signature.bind_partial instead.
...        bound = signature.bind(*args, **kwargs)
...
...        # Apply the defaults so b is always filled
...        bound.apply_defaults()
...
...        # Extract the filled arguments. If the number of
...        # arguments is still expected to be fixed, you can use
...        # tuple unpacking: 'a, b = bound.arguments.values()'
...        a = bound.arguments['a']
...        b = bound.arguments['b']
...        return function(a, b + 5)
...
...    return _decorator

>>> @decorator
... def func(a, b=3):
...     return a + b

>>> func(1, 2)
8

>>> func(a=1, b=2)
8

>>> func(a=1)
9 
```

通过使用这种方法，函数变得更加灵活。我们可以轻松地向 `add` 函数添加参数，并仍然确保装饰器函数正常工作。

## `functools.wraps` 的重要性

每次编写装饰器时，务必确保添加 `functools.wraps` 以包装内部函数。如果不进行包装，你将失去原始函数的所有属性，这可能导致混淆和意外的行为。看看以下没有 `functools.wraps` 的代码：

```py
>>> def decorator(function):
...    def _decorator(*args, **kwargs):
...        return function(*args, **kwargs)
...
...    return _decorator

>>> @decorator
... def add(a, b):
...     '''Add a and b'''
...     return a + b

>>> help(add)
Help on function _decorator in module ...:
<BLANKLINE>
_decorator(*args, **kwargs)
<BLANKLINE>

>>> add.__name__
'_decorator' 
```

现在，我们的 `add` 方法已经没有文档说明，名称也消失了。它已经被重命名为 `_decorator`。由于我们确实在调用 `_decorator`，这是可以理解的，但这对依赖于这些信息的代码来说非常不方便。现在我们将尝试相同的代码，但略有不同；我们将使用 `functools.wraps`：

```py
>>> import functools

>>> def decorator(function):
...     @functools.wraps(function)
...     def _decorator(*args, **kwargs):
...         return function(*args, **kwargs)
...
...     return _decorator

>>> @decorator
... def add(a, b):
...     '''Add a and b'''
...     return a + b

>>> help(add)
Help on function add in module ...:
<BLANKLINE>
add(a, b)
    Add a and b
<BLANKLINE>

>>> add.__name__
'add' 
```

在没有进行任何其他更改的情况下，我们现在有了文档和预期的函数名称。`functools.wraps` 的工作原理并不神奇；它复制并更新了几个属性。具体来说，以下属性被复制：

+   `__doc__`

+   `__name__`

+   `__module__`

+   `__annotations__`

+   `__qualname__`

此外，`__dict__` 使用 `_decorator.__dict__.update(add.__dict__)` 进行更新，并添加了一个名为 `__wrapped__` 的新属性，它包含原始函数（在这种情况下是 `add`）。实际的 `wraps` 函数可在你的 Python 发行版的 `functools.py` 文件中找到。

## 连接或嵌套装饰器

由于我们正在包装函数，我们无法阻止添加多个包装器。然而，需要注意的是顺序，因为装饰器是从内部开始初始化的，但却是从外部开始调用的。此外，拆卸也是从内部开始的：

```py
>>> import functools

>>> def track(function=None, label=None):
...     # Trick to add an optional argument to our decorator
...     if label and not function:
...         return functools.partial(track, label=label)
...
...     print(f'initializing {label}')
...
...     @functools.wraps(function)
...     def _track(*args, **kwargs):
...         print(f'calling {label}')
...         function(*args, **kwargs)
...         print(f'called {label}')
...
...     return _track

>>> @track(label='outer')
... @track(label='inner')
... def func():
...     print('func')
initializing inner
initializing outer

>>> func()
calling outer
calling inner
func
called inner
called outer 
```

如您在输出中看到的，在运行函数之前，装饰器是从外部到内部调用的，而在处理结果时是从内部到外部调用的。

## 使用装饰器注册函数

我们已经看到了如何跟踪调用，修改参数，以及更改返回值。现在，我们将看到如何使用装饰器注册一个对注册插件、回调等有用的函数。

这种情况在用户界面中非常有用。让我们假设我们有一个 GUI，它有一个可以被点击的按钮。通过创建一个可以注册回调的系统，我们可以使按钮触发一个“点击”信号，并将函数连接到该事件。

要创建一个类似的事件管理器，我们现在将创建一个类来跟踪所有已注册的函数并允许触发事件：

```py
>>> import collections

>>> class EventRegistry:
...     def __init__(self):
...         self.registry = collections.defaultdict(list)
... 
...     def on(self, *events):
...         def _on(function):
...             for event in events:
...                 self.registry[event].append(function)
...             return function
... 
...         return _on
... 
...     def fire(self, event, *args, **kwargs):
...         for function in self.registry[event]:
...             function(*args, **kwargs)

>>> events = EventRegistry()

>>> @events.on('success', 'error')
... def teardown(value):
...     print(f'Tearing down got: {value}')

>>> @events.on('success')
... def success(value):
...     print(f'Successfully executed: {value}')

>>> events.fire('non-existing', 'nothing to see here')
>>> events.fire('error', 'Oops, some error here')
Tearing down got: Oops, some error here
>>> events.fire('success', 'Everything is fine')
Tearing down got: Everything is fine
Successfully executed: Everything is fine 
```

首先，我们创建`EventRegistry`类来处理所有事件并存储所有回调。之后，我们将一些函数注册到注册表中。最后，我们触发一些事件以查看它是否按预期工作。

虽然这个示例相当基础，但这种模式可以应用于许多场景：处理网络服务器的事件，让插件注册事件，让插件在应用程序中注册，等等。

## 使用装饰器的记忆化

记忆化是一个简单的技巧，用于记住结果，以便在特定场景中使代码运行得更快。这里的技巧是存储输入和预期输出的映射，这样你只需要计算一次值。这种技术最常见的一个例子是简单的（递归）斐波那契函数。

斐波那契序列从 0 或 1 开始（这取决于你如何看待它），每个连续的数字都是前两个数字之和。为了说明从初始的`0`和`1`加法开始的模式：

```py
`1 = 0 + 1`
`2 = 1 + 1`
`3 = 1 + 2`
`5 = 2 + 3`
`8 = 3 + 5` 
```

我现在将展示如何构建一个非常基本的记忆化函数装饰器，以及如何使用它：

```py
>>> import functools

>>> def memoize(function):
...     # Store the cache as attribute of the function so we can
...     # apply the decorator to multiple functions without
...     # sharing the cache.
...     function.cache = dict()
...
...     @functools.wraps(function)
...     def _memoize(*args):
...         # If the cache is not available, call the function
...         # Note that all args need to be hashable
...         if args not in function.cache:
...             function.cache[args] = function(*args)
...         return function.cache[args]
...
...     return _memoize 
```

`memoize`装饰器必须无参数使用，并且缓存也可以被检查：

```py
>>> @memoize
... def fibonacci(n):
...     if n < 2:
...         return n
...     else:
...         return fibonacci(n - 1) + fibonacci(n - 2)

>>> for i in range(1, 7):
...     print(f'fibonacci {i}: {fibonacci(i)}')
fibonacci 1: 1
fibonacci 2: 1
fibonacci 3: 2
fibonacci 4: 3
fibonacci 5: 5
fibonacci 6: 8

>>> fibonacci.__wrapped__.cache
{(1,): 1, (0,): 0, (2,): 1, (3,): 2, (4,): 3, (5,): 5, (6,): 8} 
```

当给出参数时，它将崩溃，因为装饰器并没有被构建来支持这些参数：

```py
# It breaks keyword arguments:
>>> fibonacci(n=2)
Traceback (most recent call last):
...
TypeError: _memoize() got an unexpected keyword argument 'n' 
```

此外，参数需要是可哈希的才能与这个实现一起工作：

```py
# Unhashable types don't work as dict keys:
>>> fibonacci([123])
Traceback (most recent call last):
...
TypeError: unhashable type: 'list' 
```

当使用小的`n`值时，示例将很容易工作而不需要记忆化，但对于较大的数字，它将运行非常长的时间。对于`n=2`，函数将递归地执行`fibonacci(n - 1)`和`fibonacci(n - 2)`，导致指数级的时间复杂度。对于`n=30`，斐波那契函数已经被调用 2,692,537 次；在`n=50`时，它可能会停滞或甚至崩溃你的系统。

没有缓存的情况下，调用栈会变成一个树，它很快就会迅速增长。为了说明，让我们假设我们想要计算 `fibonacci(4)`。

首先，`fibonacci(4)` 调用了 `fibonacci(3)` 和 `fibonacci(2)`。这里没有什么特别之处。

现在，`fibonacci(3)` 调用了 `fibonacci(2)` 和 `fibonacci(1)`。你会注意到现在我们第二次得到了 `fibonacci(2)`。`fibonacci(4)` 也执行了它。

每次调用时的这种分割正是问题所在。每个函数调用开始两个新的函数调用，这意味着每次调用都会翻倍。而且它们会一次又一次地翻倍，直到我们达到计算的末尾。

由于缓存版本缓存了结果并且只需要计算每个数字一次，它甚至不费吹灰之力，对于 `n=30` 只需要执行 `31` 次。

这个装饰器还展示了如何将上下文附加到函数本身。在这种情况下，缓存属性成为内部（包装的 `fibonacci`）函数的一个属性，这样就不会与任何其他装饰的函数发生冲突。

然而，请注意，由于 Python 在 3.2 版本中引入了 `lru_cache`（**最近最少使用缓存**），自己实现缓存函数现在通常不再那么有用。`lru_cache` 与前面的 `memoize` 装饰器函数类似，但更高级。它维护一个固定的缓存大小（默认为 `128`），以节省内存，并存储统计信息，这样你就可以检查是否应该增加缓存大小。

如果你只关心统计信息并且不需要缓存，你也可以将 `maxsize` 设置为 `0`。或者，如果你想放弃 LRU 算法并保存所有内容，可以将 `maxsize` 传递为 `None`。具有固定大小的 `lru_cache` 将只保留最近访问的项目，并在填满后丢弃最旧的项。

在大多数情况下，我建议使用 `lru_cache` 而不是你自己的装饰器，但如果你需要存储所有项目或者需要在存储之前处理键，你总是可以自己实现。至少，了解如何编写这样的装饰器是有用的。

为了演示 `lru_cache` 内部是如何工作的，我们将计算 `fibonacci(100)`，如果没有缓存，这将使我们的计算机忙碌到宇宙的尽头。此外，为了确保我们实际上可以看到 `fibonacci` 函数被调用的次数，我们将添加一个额外的装饰器来跟踪计数，如下所示：

```py
>>> import functools

# Create a simple call counting decorator
>>> def counter(function):
...     function.calls = 0
...     @functools.wraps(function)
...     def _counter(*args, **kwargs):
...         function.calls += 1
...         return function(*args, **kwargs)
...
...     return _counter

# Create a LRU cache with size 3 
>>> @functools.lru_cache(maxsize=3)
... @counter
... def fibonacci(n):
...     if n < 2:
...         return n
...     else:
...         return fibonacci(n - 1) + fibonacci(n - 2)

>>> fibonacci(100)
354224848179261915075

# The LRU cache offers some useful statistics
>>> fibonacci.cache_info()
CacheInfo(hits=98, misses=101, maxsize=3, currsize=3)

# The result from our counter function which is now wrapped both by
# our counter and the cache
>>> fibonacci.__wrapped__.__wrapped__.calls
101 
```

你可能会想知道为什么我们只需要 `3` 的缓存大小进行 `101` 次调用。那是因为我们递归地只需要 `n - 1` 和 `n - 2`，所以在这种情况下我们不需要更大的缓存。如果你的缓存没有按预期工作，缓存大小可能是罪魁祸首。

此外，这个例子展示了为单个函数使用两个装饰器的情况。你可以把它们看作是洋葱的层。当调用 `fibonacci` 时，执行顺序如下：

1.  `functools.lru_cache`

1.  `counter`

1.  `fibonacci`

返回值的工作顺序是相反的，当然；`fibonacci` 将其值返回给 `counter`，然后 `counter` 将值传递给 `lru_cache`。

## 带有（可选）参数的装饰器

之前的例子大多使用了不带任何参数的简单装饰器。正如您已经通过 `lru_cache` 看到的，装饰器也可以接受参数，因为它们只是普通的函数，但这给装饰器增加了一个额外的层次。这意味着我们需要检查装饰器的参数，以确定它们是被装饰的方法还是普通参数。唯一的注意事项是可选参数不应该可调用。如果参数必须可调用，您需要将其作为关键字参数传递。

下面的代码展示了具有可选（关键字）参数的装饰器：

```py
>>> import functools

>>> def add(function=None, add_n=0):
...     # function is not callable so it's probably 'add_n'
...     if not callable(function):
...         # Test to make sure we don't pass 'None' as 'add_n'
...         if function is not None:
...             add_n = function
...         return functools.partial(add, add_n=add_n)
...     
...     @functools.wraps(function)
...     def _add(n):
...         return function(n) + add_n
...
...     return _add

>>> @add
... def add_zero(n):
...     return n

>>> @add(1)
... def add_one(n):
...     return n

>>> @add(add_n=2)
... def add_two(n):
...     return n

>>> add_zero(5)
5

>>> add_one(5)
6

>>> add_two(5)
7 
```

此装饰器使用 `callable()` 测试来查看参数是否是可调用的，例如函数。这种方法在许多情况下都有效，但如果您的 `add()` 装饰器的参数是可调用的，这将导致错误，因为它将被调用而不是函数。

每当您有选择时，我建议您要么使用带参数的装饰器，要么使用不带参数的装饰器。拥有可选参数会使函数的流程不那么明显，当出现问题时，调试稍微困难一些。

## 使用类创建装饰器

与我们创建常规函数装饰器的方式类似，也可以使用类来创建装饰器。正如类总是那样，这使得存储数据、继承和重用比函数更方便。毕竟，函数只是一个可调用对象，而类也可以实现可调用接口。以下装饰器的工作方式与之前使用的 `debug` 装饰器类似，但使用的是类而不是普通函数：

```py
>>> import functools

>>> class Debug(object):
...
...     def __init__(self, function):
...         self.function = function
...         # functools.wraps for classes
...         functools.update_wrapper(self, function)
...
...     def __call__(self, *args, **kwargs):
...         output = self.function(*args, **kwargs)
...         name = self.function.__name__
...         print(f'{name}({args!r}, {kwargs!r}): {output!r}')
...         return output

>>> @Debug
... def add(a, b=0):
...     return a + b
...

>>> output = add(3)
add((3,), {}): 3

>>> output = add(a=4, b=2)
add((), {'a': 4, 'b': 2}): 6 
```

函数和类之间唯一的显著区别是，在 `__init__` 方法中，`functools.wraps` 现在由 `functools.update_wrapper` 替换。

由于类方法除了常规参数外还有一个 `self` 参数，您可能会想知道装饰器是否会在那种情况下工作。下一节将介绍类内部装饰器的使用。

# 装饰类函数

装饰类函数与常规函数非常相似，但您需要意识到所需的首个参数 `self`——类实例。您可能已经使用了一些类函数装饰器。例如，`classmethod`、`staticmethod` 和 `property` 装饰器在许多不同的项目中都有使用。为了解释这一切是如何工作的，我们将构建自己的 `classmethod`、`staticmethod` 和 `property` 装饰器版本。首先，让我们看看一个简单的类函数装饰器，以展示它与常规装饰器的区别：

```py
>>> import functools

>>> def plus_one(function):
...     @functools.wraps(function)
...     def _plus_one(self, n, *args):
...         return function(self, n + 1, *args)
...
...     return _plus_one

>>> class Adder(object):
...     @plus_one
...     def add(self, a, b=0):
...         return a + b

>>> adder = Adder()
>>> adder.add(0)
1
>>> adder.add(3, 4)
8 
```

正如常规函数的情况一样，类函数装饰器现在将 `self` 作为实例传递。没有什么意外的！

## 跳过实例 – 类方法和静态方法

`classmethod` 和 `staticmethod` 之间的区别相当简单。`classmethod` 传递一个类对象而不是类实例（`self`），而 `staticmethod` 完全跳过类和实例。这实际上使得 `staticmethod` 在类外部与普通函数非常相似。

在以下示例中，我们将使用 `pprint.pprint(... width=60)` 来考虑书籍的宽度。此外，`locals()` 是 Python 的内置函数，显示所有局部变量。同样，`globals()` 函数也是可用的。

在我们重新创建 `classmethod` 和 `staticmethod` 之前，我们需要看一下这些方法的预期行为：

```py
>>> import pprint

>>> class Spam(object):
...     def some_instancemethod(self, *args, **kwargs):
...         pprint.pprint(locals(), width=60)
...
...     @classmethod
...     def some_classmethod(cls, *args, **kwargs):
...         pprint.pprint(locals(), width=60)
...
...     @staticmethod
...     def some_staticmethod(*args, **kwargs):
...         pprint.pprint(locals(), width=60)

# Create an instance so we can compare the difference between
# executions with and without instances easily
>>> spam = Spam() 
```

以下示例将使用上面的示例来说明普通（类实例）方法、`classmethod` 和 `staticmethod` 之间的区别。请注意 `spam`（小写）的实例和 `Spam`（大写）的类之间的区别：

```py
# With an instance (note the lowercase spam)
>>> spam.some_instancemethod(1, 2, a=3, b=4)
{'args': (1, 2),
 'kwargs': {'a': 3, 'b': 4},
 'self': <__main__.Spam object at ...>}

# Without an instance (note the capitalized Spam)
>>> Spam.some_instancemethod()
Traceback (most recent call last):
    ...
TypeError: some_instancemethod() missing ... argument: 'self'

# But what if we add parameters? Be very careful with these!
# Our first argument is now used as an argument, this can give
# very strange and unexpected errors
>>> Spam.some_instancemethod(1, 2, a=3, b=4)
{'args': (2,), 'kwargs': {'a': 3, 'b': 4}, 'self': 1} 
```

特别是，最后一个示例相当棘手。因为我们向函数传递了一些参数，这些参数自动作为 `self` 参数传递。同样，最后一个示例展示了如何使用这个参数处理来使用给定的实例调用方法。`Spam.some_instancemethod(spam)` 与 `spam.some_instancemethod()` 相同。

现在让我们看看 `classmethod`：

```py
# Classmethods are expectedly identical
>>> spam.some_classmethod(1, 2, a=3, b=4)
{'args': (1, 2),
 'cls': <class '__main__.Spam'>,
 'kwargs': {'a': 3, 'b': 4}}

>>> Spam.some_classmethod()
{'args': (), 'cls': <class '__main__.Spam'>, 'kwargs': {}}

>>> Spam.some_classmethod(1, 2, a=3, b=4)
{'args': (1, 2),
 'cls': <class '__main__.Spam'>,
 'kwargs': {'a': 3, 'b': 4}} 
```

主要区别在于，我们现在有 `cls` 而不是 `self`，它包含类（`Spam`）而不是实例（`spam`）。

`self` 和 `cls` 是约定俗成的命名，并且没有任何强制要求。你可以轻松地将它们命名为 `s` 和 `c` 或者其他完全不同的名称。

接下来是 `staticmethod`。`staticmethod` 在类外部表现得与普通函数相同。

```py
# Staticmethods are also identical
>>> spam.some_staticmethod(1, 2, a=3, b=4)
{'args': (1, 2), 'kwargs': {'a': 3, 'b': 4}}

>>> Spam.some_staticmethod()
{'args': (), 'kwargs': {}}

>>> Spam.some_staticmethod(1, 2, a=3, b=4)
{'args': (1, 2), 'kwargs': {'a': 3, 'b': 4}} 
```

在我们继续使用装饰器之前，你需要了解 Python 描述符是如何工作的。描述符可以用来修改对象属性的绑定行为。这意味着如果描述符被用作属性值，你可以在这些操作被调用时修改设置的值、获取的值和删除的值。以下是一个基本示例：

```py
>>> class Spam:
...     def __init__(self, spam=1):
...         self.spam = spam
...
...     def __get__(self, instance, cls):
...         return self.spam + instance.eggs
...
...     def __set__(self, instance, value):
...         instance.eggs = value - self.spam

>>> class Sandwich:
...     spam = Spam(5)
...
...     def __init__(self, eggs):
...         self.eggs = eggs

>>> sandwich = Sandwich(1)
>>> sandwich.eggs
1
>>> sandwich.spam
6

>>> sandwich.eggs = 10
>>> sandwich.spam
15 
```

如你所见，每次我们从 `sandwich.spam` 设置或获取值时，实际上是在调用 `Spam` 的 `__get__` 或 `__set__`，它不仅有权访问自己的变量，还可以访问调用类。这是一个非常有用的特性，对于自动转换和类型检查非常有用，我们将在下一节中看到的 `property` 装饰器只是这种技术的更方便的实现。

现在你已经了解了描述符的工作原理，我们可以继续创建 `classmethod` 和 `staticmethod` 装饰器。对于这两个装饰器，我们只需要修改 `__get__` 而不是 `__call__`，这样我们就可以控制传递的类型实例（或根本不传递）：

```py
>>> import functools

>>> class ClassMethod(object):
...     def __init__(self, method):
...         self.method = method
... 
...     def __get__(self, instance, cls):
...         @functools.wraps(self.method)
...         def method(*args, **kwargs):
...             return self.method(cls, *args, **kwargs)
...
...         return method

>>> class StaticMethod(object):
...     def __init__(self, method):
...         self.method = method
... 
...     def __get__(self, instance, cls):
...         return self.method

>>> class Sandwich:
...     spam = 'class'
...
...     def __init__(self, spam):
...         self.spam = spam
...
...     @ClassMethod
...     def some_classmethod(cls, arg):
...         return cls.spam, arg
...
...     @StaticMethod
...     def some_staticmethod(arg):
...         return Sandwich.spam, arg

>>> sandwich = Sandwich('instance')
>>> sandwich.spam
'instance'
>>> sandwich.some_classmethod('argument')
('class', 'argument')
>>> sandwich.some_staticmethod('argument')
('class', 'argument') 
```

`ClassMethod` 装饰器仍然具有一个子函数来实际生成一个可工作的装饰器。查看该函数，你很可能会猜到它是如何工作的。它不是将 `instance` 作为 `self.method` 的第一个参数传递，而是传递 `cls`。

`StaticMethod` 更简单，因为它完全忽略了 `instance` 和 `cls`。它可以返回未经修改的原始方法。因为它返回的是未经修改的原始方法，所以我们不需要 `functools.wraps` 调用。

## 属性 - 智能描述符使用

`property` 装饰器可能是 Python 中使用最广泛的装饰器。它允许你向现有实例属性添加获取器和设置器，这样你就可以添加验证器并在将值设置到实例属性之前修改它们。

`property` 装饰器既可以作为赋值使用，也可以作为装饰器使用。以下示例展示了两种语法，以便你知道 `property` 装饰器可以期待什么。

Python 3.8 添加了 `functools.cached_property`，该函数与 `property` 功能相同，但每个实例只执行一次。

```py
>>> import functools

>>> class Sandwich(object):
...     def get_eggs(self):
...         print('getting eggs')
...         return self._eggs
...
...     def set_eggs(self, eggs):
...         print('setting eggs to %s' % eggs)
...         self._eggs = eggs
...
...     def delete_eggs(self):
...         print('deleting eggs')
...         del self._eggs
...
...     eggs = property(get_eggs, set_eggs, delete_eggs)
...
...     @property
...     def spam(self):
...         print('getting spam')
...         return self._spam
...
...     @spam.setter
...     def spam(self, spam):
...         print('setting spam to %s' % spam)
...         self._spam = spam
...
...     @spam.deleter
...     def spam(self):
...         print('deleting spam')
...         del self._spam
...
...     @functools.cached_property
...     def bacon(self):
...         print('getting bacon')
...         return 'bacon!'

>>> sandwich = Sandwich()

>>> sandwich.eggs = 123
setting eggs to 123

>>> sandwich.eggs
getting eggs
123
>>> del sandwich.eggs
deleting eggs
>>> sandwich.bacon
getting bacon
'bacon!' 
>>> sandwich.bacon
'bacon!' 
```

类似于我们实现 `classmethod` 和 `staticmethod` 装饰器的方式，我们再次需要 Python 描述符。这次，我们需要描述符的全部功能，不仅仅是 `__get__`，还包括 `__set__` 和 `__delete__`。然而，为了简洁起见，我们将跳过处理文档和一些错误处理：

```py
>>> class Property(object):
...     def __init__(self, fget=None, fset=None, fdel=None):
...         self.fget = fget
...         self.fset = fset
...         self.fdel = fdel
... 
...     def __get__(self, instance, cls):
...         if instance is None:
...             # Redirect class (not instance) properties to self
...             return self
...         elif self.fget:
...             return self.fget(instance)
... 
...     def __set__(self, instance, value):
...         self.fset(instance, value)
... 
...     def __delete__(self, instance):
...         self.fdel(instance)
... 
...     def getter(self, fget):
...         return Property(fget, self.fset, self.fdel)
... 
...     def setter(self, fset):
...         return Property(self.fget, fset, self.fdel)
... 
...     def deleter(self, fdel):
...         return Property(self.fget, self.fset, fdel) 
```

这看起来并不那么复杂，对吧？描述符构成了大部分代码，相当直接。只有 `getter`/`setter`/`deleter` 函数可能看起来有点奇怪，但它们实际上也很直接。

为了确保 `property` 仍然按预期工作，类在复制其他方法的同时返回一个新的 `Property` 实例。这里要使这个功能正常工作的小提示是在 `__get__` 方法中的 `return self`。

```py
>>> class Sandwich:
...     @Property
...     def eggs(self):
...         return self._eggs
...
...     @eggs.setter
...     def eggs(self, value):
...         self._eggs = value
...
...     @eggs.deleter
...     def eggs(self):
...         del self._eggs

>>> sandwich = Sandwich()
>>> sandwich.eggs = 5
>>> sandwich.eggs
5 
```

如预期的那样，我们的 `Property` 装饰器按预期工作。但请注意，这是一个比内置的 `property` 装饰器更有限的版本；我们的版本没有对边缘情况进行检查。

自然地，作为 Python，有更多方法可以达到属性的效果。在先前的例子中，你看到了裸描述符实现，在我们的先前的例子中，你看到了 `property` 装饰器。现在我们将通过实现 `__getattr__` 或 `__getattribute__` 来查看一个通用的解决方案。以下是一个简单的演示：

```py
>>> class Sandwich(object):
...     def __init__(self):
...         self.registry = {}
...
...     def __getattr__(self, key):
...         print('Getting %r' % key)
...         return self.registry.get(key, 'Undefined')
...
...     def __setattr__(self, key, value):
...         if key == 'registry':
...             object.__setattr__(self, key, value)
...         else:
...             print('Setting %r to %r' % (key, value))
...             self.registry[key] = value
...
...     def __delattr__(self, key):
...         print('Deleting %r' % key)
...         del self.registry[key]

>>> sandwich = Sandwich()

>>> sandwich.a
Getting 'a'
'Undefined'

>>> sandwich.a = 1
Setting 'a' to 1

>>> sandwich.a
Getting 'a'
1

>>> del sandwich.a
Deleting 'a' 
```

`__getattr__` 方法查找现有属性，例如，它检查键是否存在于 `instance.__dict__` 中，并且仅在它不存在时调用。这就是为什么我们从未看到对注册属性的 `__getattr__` 的调用。`__getattribute__` 方法在所有情况下都会被调用，这使得它使用起来稍微有些危险。使用 `__getattribute__` 方法时，你需要对 `registry` 进行特定的排除，因为它在尝试访问 `self.registry` 时会无限递归执行。

很少有必要查看描述符，但它们被几个内部 Python 进程使用，例如在继承类时使用的`super()`方法。

现在你已经知道了如何为常规函数和类方法创建装饰器，让我们继续通过装饰整个类来继续。

# 装饰类

Python 2.6 引入了类装饰器语法。与函数装饰器语法一样，这也不是一项新技术。即使没有语法，也可以通过执行`DecoratedClass = decorator(RegularClass)`简单地装饰一个类。在之前的章节中，你应该已经熟悉了编写装饰器。类装饰器与常规装饰器没有区别，只是它们接受一个类而不是一个函数。与函数一样，这发生在声明时间，而不是在实例化/调用时间。

由于有相当多的方法可以修改类的工作方式，例如标准继承、混入和元类（更多内容请参阅*第八章*，*元类 – 使类（而非实例）更智能*），类装饰器从未是严格必需的。这并不减少它们的有用性，但它确实解释了为什么你不太可能看到太多类装饰的示例。

## 单例 – 只有一个实例的类

单例是始终只允许存在一个实例的类。因此，你总是得到同一个实例，而不是为你的调用获取一个特定的实例。这对于像数据库连接池这样的东西非常有用，你不想总是打开连接，但想重用原始的连接：

```py
>>> import functools

>>> def singleton(cls):
...     instances = dict()
...     @functools.wraps(cls)
...     def _singleton(*args, **kwargs):
...         if cls not in instances:
...             instances[cls] = cls(*args, **kwargs)
...         return instances[cls]
...     return _singleton

>>> @singleton
... class SomeSingleton(object):
...     def __init__(self):
...         print('Executing init')

>>> a = SomeSingleton()
Executing init
>>> b = SomeSingleton()

>>> a is b

True
>>> a.x = 123
>>> b.x
123 
```

正如你在`a is b`比较中看到的，两个对象具有相同的身份，因此我们可以得出结论，它们确实是同一个对象。正如常规装饰器的情况一样，由于`functools.wraps`功能，如果需要，我们仍然可以通过`Spam.__wrapped__`访问原始类。

`is`运算符通过身份比较对象，这在 CPython 中实现为内存地址。如果`a is b`返回`True`，我们可以得出结论，`a`和`b`是同一个实例。

## 完全排序 – 使类可排序

在某个时候，你可能需要排序数据结构。虽然使用`sorted`函数的键参数可以轻松实现这一点，但如果需要经常这样做，有一个更方便的方法——通过实现`__gt__`、`__ge__`、`__lt__`、`__le__`和`__eq__`函数。这听起来有点冗长，不是吗？如果你想获得最佳性能，这仍然是一个好主意，但如果你可以接受一点性能损失和一些稍微复杂一点的堆栈跟踪，那么`total_ordering`可能是一个不错的选择。

`total_ordering` 类装饰器可以根据具有 `__eq__` 函数和其中一个比较函数（`__lt__`、`__le__`、`__gt__` 或 `__ge__`）的类实现所有所需的排序函数。这意味着你可以大大缩短你的函数定义。让我们比较一下常规函数定义和使用 `total_ordering` 装饰器的函数定义：

```py
>>> import functools

>>> class Value(object):
...     def __init__(self, value):
...         self.value = value
...                                                               
...     def __repr__(self):
...         return f'<{self.__class__.__name__} {self.value}>'

>>> class Spam(Value):
...     def __gt__(self, other):
...         return self.value > other.value
...                                                                
...     def __ge__(self, other):
...         return self.value >= other.value
...                                                                
...     def __lt__(self, other):
...         return self.value < other.value
...                                                                
...     def __le__(self, other):
...         return self.value <= other.value
...                                                                
...     def __eq__(self, other):
...         return self.value == other.value

>>> @functools.total_ordering
... class Egg(Value):
...     def __lt__(self, other):
...         return self.value < other.value
...                                                                  
...     def __eq__(self, other):
...         return self.value == other.value 
```

如你所见，没有 `functools.total_ordering`，创建一个完全可排序的类需要相当多的工作。现在我们将测试它们是否确实以类似的方式排序：

```py
>>> numbers = [4, 2, 3, 4]
>>> spams = [Spam(n) for n in numbers]
>>> eggs = [Egg(n) for n in numbers]

>>> spams
[<Spam 4>, <Spam 2>, <Spam 3>, <Spam 4>]

>>> eggs
[<Egg 4>, <Egg 2>, <Egg 3>, <Egg 4>]

>>> sorted(spams)
[<Spam 2>, <Spam 3>, <Spam 4>, <Spam 4>]

>>> sorted(eggs)
[<Egg 2>, <Egg 3>, <Egg 4>, <Egg 4>]

# Sorting using key is of course still possible and in this case
# perhaps just as easy:
>>> values = [Value(n) for n in numbers]
>>> values
[<Value 4>, <Value 2>, <Value 3>, <Value 4>]

>>> sorted(values, key=lambda v: v.value)
[<Value 2>, <Value 3>, <Value 4>, <Value 4>] 
```

现在，你可能想知道，“为什么没有类装饰器可以用来通过指定的键属性使类可排序？”好吧，这确实可能是 `functools` 库的一个好主意，但现在还没有。所以，让我们看看我们如何在仍然使用 `functools.total_ordering` 的同时实现类似的功能：

```py
>>> def sort_by_attribute(attr, keyfunc=getattr):
...     def _sort_by_attribute(cls):
...         def __lt__(self, other):
...             return getattr(self, attr) < getattr(other, attr)
...                                           
...         def __eq__(self, other):
...             return getattr(self, attr) <= getattr(other, attr)
...                                           
...         cls.__lt__ = __lt__               
...         cls.__eq__ = __eq__               
...                                 
...         return functools.total_ordering(cls)
...
...     return _sort_by_attribute

>>> class Value(object):
...     def __init__(self, value):
...         self.value = value
...         
...     def __repr__(self):
...         return f'<{self.__class__.__name__} {self.value}>'

>>> @sort_by_attribute('value')
... class Spam(Value):
...     pass

>>> numbers = [4, 2, 3, 4]
>>> spams = [Spam(n) for n in numbers]
>>> sorted(spams)
[<Spam 2>, <Spam 3>, <Spam 4>, <Spam 4>] 
```

当然，这大大简化了创建可排序类的过程。如果你更愿意使用自己的键函数而不是 `getattr`，那就更容易了。只需将 `getattr(self, attr)` 调用替换为 `key_function(self)`，同样对 `other` 也这样做，并将装饰器的参数更改为你的函数。你甚至可以使用它作为基本函数，并通过简单地传递一个包装的 `getattr` 函数来实现 `sort_by_attribute`。

现在你已经知道了如何创建所有类型的装饰器，让我们看看 Python 内置的一些有用的装饰器示例。

# 有用的装饰器

除了本章中提到的那些之外，Python 还内置了一些其他有用的装饰器。有些装饰器目前不在标准库中（也许将来会加入）。

## 单分发 – Python 中的多态

如果你之前使用过 C++ 或 Java，你可能已经习惯了有专门的泛型多态可用——根据参数类型调用不同的函数。Python 作为一种动态类型语言，大多数人不会期望存在单分发模式。然而，Python 不仅是一种动态类型语言，而且是一种强类型语言，这意味着我们可以依赖我们接收到的类型。

动态类型语言不需要严格的类型定义。而像 C 这样的语言需要以下内容来声明一个整数：

```py
`int some_integer = 123;` 
```

Python 只是接受我们的值具有类型：

```py
`some_integer = 123` 
```

虽然我们可以使用类型提示来做同样的事情：

```py
`some_integer: int = 123` 
```

然而，与 JavaScript 和 PHP 等语言相比，Python 做的隐式类型转换非常少。在 Python 中，以下将返回错误，而 JavaScript 会无任何问题执行它：

```py
`'spam' + 5` 
```

在 Python 中，结果是 `TypeError`。在 JavaScript 中，结果是 `'spam5'`。

单分发的基本思想是，根据你传递的类型，调用正确的函数。由于在 Python 中 `str + int` 会导致错误，这可以在将参数传递给函数之前自动转换参数变得非常方便。这可以用来将函数的实际工作与类型转换分离。

自从 Python 3.4 以来，有一个装饰器可以轻松地在 Python 中实现单分派模式。如果你需要根据输入变量的`type()`执行不同的函数，这个装饰器很有用。以下是一个基本示例：

```py
>>> import functools

>>> @functools.singledispatch
... def show_type(argument):
...     print(f'argument: {argument}')

>>> @show_type.register(int)
... def show_int(argument):
...     print(f'int argument: {argument}')

>>> @show_type.register
... def show_float(argument: float):
...     print(f'float argument: {argument}')

>>> show_type('abc')
argument: abc

>>> show_type(123)
int argument: 123

>>> show_type(1.23)
float argument: 1.23 
```

`singledispatch`装饰器会自动调用作为第一个参数传递的类型对应的正确函数。正如你在示例中看到的，这在使用类型注解和显式传递类型到`register`函数时都有效。

让我们看看我们如何自己实现这个方法的简化版本：

```py
>>> import functools

>>> registry = dict()

>>> def register(function):
...     # Fetch the first type from the type annotation but be
...     # careful not to overwrite the 'type' function
...     type_ = next(iter(function.__annotations__.values()))
...     registry[type_] = function
...
...     @functools.wraps(function)
...     def _register(argument):
...         # Fetch the function using the type of argument, and
...         # fall back to the main function
...         new_function = registry.get(type(argument), function)
...         return new_function(argument)
...
...     return _register

>>> @register
... def show_type(argument: any):
...     print(f'argument: {argument}')

>>> @register
... def show_int(argument: int):
...     print(f'int argument: {argument}')

>>> show_type('abc')
argument: abc

>>> show_type(123)
int argument: 123 
```

自然地，这种方法有点基础，它使用单个全局注册表，这限制了它的应用。但这个确切的模式可以用来注册插件或回调函数。

在命名函数时，确保不要覆盖原始的`singledispatch`函数。如果你将`show_int`命名为`show_type`，它将覆盖初始的`show_type`函数。这将使得无法访问原始的`show_type`函数，并导致之后的`register`操作也失败。

现在，一个稍微有用一点的例子——区分文件名和文件句柄：

```py
>>> import json
>>> import functools

>>> @functools.singledispatch
... def write_as_json(file, data):
...     json.dump(data, file)

>>> @write_as_json.register(str)
... @write_as_json.register(bytes)
... def write_as_json_filename(file, data):
...     with open(file, 'w') as fh:
...         write_as_json(fh, data)

>>> data = dict(a=1, b=2, c=3)
>>> write_as_json('test1.json', data)
>>> write_as_json(b'test2.json', 'w')
>>> with open('test3.json', 'w') as fh:
...     write_as_json(fh, data) 
```

因此，我们现在有一个单一的`write_as_json`函数；它根据类型调用正确的代码。如果它是一个`str`或`bytes`对象，它将自动打开文件并调用`write_as_json`的常规版本，该版本接受文件对象。

当然，编写一个执行此操作的装饰器并不难，但仍然很方便在基础库中有`singledispatch`装饰器。它无疑比手动使用`isinstance()`的`if`/`elif`/`elif`/`else`语句检查给定的参数类型要方便得多。

要查看哪个函数将被调用，你可以使用`write_as_json.dispatch`函数和一个特定的类型。当传递一个`str`时，你会得到`write_as_json_filename`函数。需要注意的是，分派函数的名称完全是任意的。它们当然可以作为常规函数访问，但你喜欢怎么命名就怎么命名。

要检查已注册的类型，你可以通过`write_as_json.registry`访问注册表，它是一个字典：

```py
>>> write_as_json.registry.keys()
dict_keys([<class 'bytes'>, <class 'object'>, <class 'str'>]) 
```

## contextmanager — 使 with 语句变得简单

使用`contextmanager`类，我们可以使创建上下文包装器变得非常简单。上下文包装器在每次使用`with`语句时都会使用。一个例子是`open`函数，它也作为一个上下文包装器工作，允许你使用以下代码：

```py
with open(filename) as fh:
    pass 
```

让我们假设现在`open`函数不能用作上下文管理器，我们需要构建自己的函数来完成这个任务。创建上下文管理器的标准方法是通过创建一个实现`__enter__`和`__exit__`方法的类：

```py
>>> class Open:
...     def __init__(self, filename, mode):
...         self.filename = filename
...         self.mode = mode
...
...     def __enter__(self):
...         self.handle = open(self.filename, self.mode)
...         return self.handle
...
...     def __exit__(self, exc_type, exc_val, exc_tb):
...         self.handle.close()

>>> with Open('test.txt', 'w') as fh:
...     print('Our test is complete!', file=fh) 
```

虽然这样工作得很好，但有点冗长。使用`contextlib.contextmanager`，我们可以在几行代码中实现相同的行为：

```py
>>> import contextlib

>>> @contextlib.contextmanager
... def open_context_manager(filename, mode='r'):
...     fh = open(filename, mode)
...     yield fh
...     fh.close()

>>> with open_context_manager('test.txt', 'w') as fh:
...     print('Our test is complete!', file=fh) 
```

简单吗？然而，我应该提到，对于这个特定的案例——对象的关闭——`contextlib`中有一个专门的函数，而且使用起来甚至更简单。

对于文件对象、数据库连接和连接，始终有一个`close()`调用来清理资源是很重要的。在文件的情况下，它告诉操作系统将数据写入磁盘（而不是临时缓冲区），而在网络连接和数据库连接的情况下，它释放两端的网络连接和相关资源。对于数据库连接，它还会通知服务器该连接不再需要，因此这部分也可以优雅地处理。

没有这些调用，你可能会迅速遇到“打开的文件太多”或“连接太多”的错误。

让我们用一个最基本的情况来演示`closing()`何时会有用：

```py
>>> import contextlib

>>> with contextlib.closing(open('test.txt', 'a')) as fh:
...     print('Yet another test', file=fh) 
```

对于文件对象，你通常也可以使用`with open(...)`，因为它本身就是一个上下文管理器，但如果代码的其他部分处理了打开操作，你就不总是有这种便利，在这些情况下，你需要自己关闭它。此外，一些对象，如`urllib`发出的请求，不支持以这种方式自动关闭，并从中受益于这个函数。

但等等；还有更多！除了可以在`with`语句中使用外，从 Python 3.2 开始，`contextmanager`的结果实际上也可以用作装饰器。在较老的 Python 版本中，`contextmanager`只是一个小的包装器，但自从 Python 3.2 以来，它基于`ContextDecorator`类，这使得它成为一个装饰器。

`open_context_manager`上下文管理器并不真正适合作为装饰器，因为它有一个`yield <value>`而不是空的`yield`（更多内容请参阅*第七章*，*生成器和协程 – 一次一步的无限*），但我们可以考虑其他函数：

```py
>>> @contextlib.contextmanager
... def debug(name):
...     print(f'Debugging {name}:')
...     yield
...     print(f'Finished debugging {name}')

>>> @debug('spam')
... def spam():
...     print('This is the inside of our spam function')

>>> spam()
Debugging spam:
This is the inside of our spam function
Finished debugging spam 
```

对于这个，有很多很好的用例，但至少，它是一个方便地将函数包装在上下文中的方法，而不需要所有的（嵌套）`with`语句。

## 验证、类型检查和转换

虽然在 Python 中检查类型通常不是最好的方法，但在某些情况下，如果你知道你需要一个特定的类型（或可以转换为该类型的某个东西），它可能是有用的。为了方便起见，Python 3.5 引入了类型提示系统，这样你就可以做以下操作：

```py
>>> def sandwich(bacon: float, eggs: int):
...     pass 
```

在某些情况下，将提示转换为要求可能是有用的。我们不再使用`isinstance()`，而是简单地通过类型转换来强制执行类型，这更接近鸭子类型。

鸭子类型的本质是：如果它看起来像鸭子，走起路来像鸭子，叫起来也像鸭子，那么它可能就是一只鸭子。本质上，这意味着我们不在乎值是`duck`还是其他什么，只要它支持我们需要的`quack()`方法。

为了强制执行类型提示，我们可以创建一个装饰器：

```py
>>> import inspect
>>> import functools

>>> def enforce_type_hints(function):
...     # Construct the signature from the function which contains
...     # the type annotations
...     signature = inspect.signature(function)
... 
...     @functools.wraps(function)
...     def _enforce_type_hints(*args, **kwargs):
...         # Bind the arguments and apply the default values
...         bound = signature.bind(*args, **kwargs)
...         bound.apply_defaults()
... 
...         for key, value in bound.arguments.items():
...             param = signature.parameters[key]
...             # The annotation should be a callable
...             # type/function so we can cast as validation
...             if param.annotation:
...                 bound.arguments[key] = param.annotation(value)
... 
...         return function(*bound.args, **bound.kwargs)
... 
...     return _enforce_type_hints

>>> @enforce_type_hints
... def sandwich(bacon: float, eggs: int):
...     print(f'bacon: {bacon!r}, eggs: {eggs!r}')

>>> sandwich(1, 2)
bacon: 1.0, eggs: 2
>>> sandwich(3, 'abc')
Traceback (most recent call last):
...
ValueError: invalid literal for int() with base 10: 'abc' 
```

这是一个相当简单但非常通用的类型强制器，应该可以与大多数类型注解一起工作。

## 无用的警告 - 如何安全地忽略它们

当用 Python 编写代码时，警告在编写代码时通常非常有用。然而，在执行时，每次运行你的脚本/应用程序时都收到相同的消息是没有用的。所以，让我们创建一些代码，允许轻松隐藏预期的警告，但不是所有的警告，这样我们就可以轻松捕捉到新的警告：

```py
>>> import warnings
>>> import functools

>>> def ignore_warning(warning, count=None):
...     def _ignore_warning(function):
...         @functools.wraps(function)
...         def __ignore_warning(*args, **kwargs):
...             # Execute the code while catching all warnings
...             with warnings.catch_warnings(record=True) as ws:
...                 # Catch all warnings of the given type
...                 warnings.simplefilter('always', warning)
...                 # Execute the function
...                 result = function(*args, **kwargs)
... 
...             # Re-warn all warnings beyond the expected count
...             if count is not None:
...                 for w in ws[count:]:
...                     warnings.warn(w.message)
... 
...             return result
...
...         return __ignore_warning
...
...     return _ignore_warning

>>> @ignore_warning(DeprecationWarning, count=1)
... def spam():
...     warnings.warn('deprecation 1', DeprecationWarning)
...     warnings.warn('deprecation 2', DeprecationWarning)

# Note, we use catch_warnings here because doctests normally
# capture the warnings quietly
>>> with warnings.catch_warnings(record=True) as ws:
...     spam()
...
...     for i, w in enumerate(ws):
...         print(w.message)
deprecation 2 
```

使用这种方法，我们可以捕捉到第一个（预期的）警告，同时仍然看到第二个（意外的）警告。

现在你已经看到了一些有用的装饰器示例，是时候继续进行一些练习，看看你能自己写多少了。

# 练习

装饰器有巨大的用途范围，所以在阅读完这一章后，你可能自己就能想到一些，但你很容易就可以详细阐述我们之前编写的某些装饰器：

+   将`track`函数扩展以监控执行时间。

+   将`track`函数扩展以包含最小/最大/平均执行时间和调用次数。

+   修改记忆化函数以使其能够处理不可哈希的类型。

+   修改记忆化函数，使其每个函数都有自己的缓存而不是全局缓存。

+   创建一个版本的`functools.cached_property`，可以根据需要重新计算。

+   创建一个单次调用的装饰器，它考虑所有或可配置数量的参数，而不是只考虑第一个参数。

+   增强`type_check`装饰器，包括额外的检查，例如要求一个数字大于或小于给定的值。

这些练习的示例答案可以在 GitHub 上找到：`github.com/mastering-python/exercises`。我们鼓励你提交自己的解决方案，并从他人的替代方案中学习。

# 概述

本章向您展示了装饰器可以用于使我们的代码更简单，并为非常简单的函数添加一些相当复杂的行为。诚实地讲，大多数装饰器比直接添加功能的功能性函数要复杂，但将相同的模式应用于许多函数和类所带来的额外优势通常是非常值得的。

装饰器有如此多的用途，可以让你的函数和类更智能、更方便使用：

+   调试

+   验证

+   参数便利性（预填充或转换参数）

+   输出便利性（将输出转换为特定类型）

本章最重要的收获应该是永远不要忘记在包装函数时使用`functools.wraps`。由于（意外的）行为修改，调试装饰函数可能相当困难，但丢失属性也会使这个问题变得更糟。

下一章将向您展示如何以及何时使用`generators`和`coroutines`。这一章已经简要介绍了`with`语句的用法，但`generators`和`coroutines`在这方面可以做得更多。我们仍然会经常使用装饰器，无论是在这本书中还是在一般使用 Python 时，所以请确保您对它们的工作方式有很好的理解。

# 加入我们的 Discord 社区

加入我们的 Discord 空间，与作者和其他读者进行讨论：[`discord.gg/QMzJenHuJf`](https://discord.gg/QMzJenHuJf)

![](img/QR_Code156081100001293319171.png)
