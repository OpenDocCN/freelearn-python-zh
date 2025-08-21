# 第五章：装饰器 - 通过装饰实现代码重用

在本章中，您将学习 Python 装饰器。装饰器本质上只是可以用于修改输入、输出甚至在执行之前修改函数/类本身的函数/类包装器。这种包装可以通过有一个调用内部函数的单独函数或通过混合来轻松实现。与许多 Python 构造一样，装饰器并不是达到目标的唯一方法，但在许多情况下确实很方便。

虽然您可以完全不了解装饰器，但它们给您带来了很多“重用能力”，因此在框架库中被广泛使用，例如 Web 框架。Python 实际上附带了一些有用的装饰器，最著名的是`property`装饰器。

但是，有一些需要注意的特殊情况：包装函数会创建一个新函数，并且使得难以访问内部函数及其属性。Python 的一个例子是`help(function)`功能；默认情况下，您将丢失函数属性，例如帮助文本和函数存在的模块。

本章将涵盖函数和类装饰器的用法，以及在类内装饰函数时需要了解的复杂细节。

以下是涵盖的主题：

+   装饰函数

+   装饰类函数

+   装饰类

+   使用类作为装饰器

+   Python 标准库中有用的装饰器

# 装饰函数

本质上，装饰器只不过是一个函数或类包装器。如果我们有一个名为`spam`的函数和一个名为`eggs`的装饰器，那么以下内容将使用`eggs`装饰`spam`：

```py
spam = eggs(spam)
```

为了使语法更易于使用，Python 对此情况有一个特殊的语法。因此，您可以使用`@`运算符简单地装饰一个函数，而不是在函数下面添加一行如上面的行：

```py
@eggs
def spam():
    pass
```

装饰器只是接收函数并返回一个通常不同的函数。最简单的装饰器是：

```py
def eggs(function):
    return function
```

看看前面的例子，我们意识到这将`spam`作为`function`的参数，并再次返回该函数，实际上什么也没有改变。但大多数装饰器会嵌套函数。以下装饰器将打印发送到`spam`的所有参数，并将它们不加修改地传递给`spam`：

```py
>>> import functools

>>> def eggs(function):
...    @functools.wraps(function)
...    def _eggs(*args, **kwargs):
...        print('%r got args: %r and kwargs: %r' % (
...            function.__name__, args, kwargs))
...        return function(*args, **kwargs)
...
...    return _eggs

>>> @eggs
... def spam(a, b, c):
...     return a * b + c

>>> spam(1, 2, 3)
'spam' got args: (1, 2, 3) and kwargs: {}
5

```

这应该表明装饰器可以有多么强大。通过修改`*args`和`**kwargs`，您可以完全添加、修改和删除参数。此外，返回语句也可以被修改。您可以返回完全不同的东西，而不是`return function(...)`。

## 为什么 functools.wraps 很重要

每当编写装饰器时，一定要确保添加`functools.wraps`来包装内部函数。如果不包装它，您将丢失原始函数的所有属性，这可能会导致混淆。看看下面的代码，没有`functools.wraps`：

```py
>>> def eggs(function):
...    def _eggs(*args, **kwargs):
...        return function(*args, **kwargs)
...    return _eggs

>>> @eggs
... def spam(a, b, c):
...     '''The spam function Returns a * b + c'''
...     return a * b + c

>>> help(spam)
Help on function _eggs in module ...:
<BLANKLINE>
_eggs(*args, **kwargs)
<BLANKLINE>

>>> spam.__name__
'_eggs'

```

现在，我们的`spam`方法再也没有文档了，名称也消失了。它已被重命名为`_eggs`。由于我们确实调用了`_eggs`，这是可以理解的，但对于依赖这些信息的代码来说非常不方便。现在我们将尝试使用`functools.wraps`进行相同的代码，只有一个细微的区别：

```py
>>> import functools

>>> def eggs(function):
...     @functools.wraps(function)
...     def _eggs(*args, **kwargs):
...         return function(*args, **kwargs)
...     return _eggs

>>> @eggs
... def spam(a, b, c):
...     '''The spam function Returns a * b + c'''
...     return a * b + c

>>> help(spam)
Help on function spam in module ...:
<BLANKLINE>
spam(a, b, c)
 **The spam function Returns a * b + c
<BLANKLINE>

>>> spam.__name__
'spam'

```

没有任何进一步的更改，我们现在有了文档和预期的函数名称。然而，`functools.wraps`的工作并不神奇；它只是复制和更新了几个属性。具体来说，复制了以下属性：

+   `__doc__`

+   `__name__`

+   `__module__`

+   `__annotations__`

+   `__qualname__`

此外，使用`_eggs.__dict__.update(spam.__dict__)`更新`__dict__`，并添加一个名为`__wrapped__`的新属性，其中包含原始（在本例中为`spam`）函数。实际的`wraps`函数可以在 Python 分发的`functools.py`文件中找到。

## 装饰器有什么用？

装饰器的用例很多，但其中一些最有用的用例是调试。关于这一点的更多详细示例将在第十一章中进行介绍，*调试 - 解决错误*，但我可以给你一个小窥探，看看如何使用装饰器来跟踪代码的运行情况。

假设你有一堆可能被调用或可能不被调用的函数，并且你并不完全确定每个函数的输入和输出是什么。在这种情况下，你当然可以修改函数，并在开始和结束时添加一些打印语句来打印输出。然而，这很快就会变得乏味，这是一个简单的装饰器可以让你轻松做同样的事情的情况之一。

对于这个例子，我们使用了一个非常简单的函数，但我们都知道在现实生活中，我们并不总是那么幸运：

```py
>>> def spam(eggs):
...     return 'spam' * (eggs % 5)
...
>>> output = spam(3)

```

让我们拿我们简单的`spam`函数，并添加一些输出，这样我们就可以看到内部发生了什么：

```py
>>> def spam(eggs):
...     output = 'spam' * (eggs % 5)
...     print('spam(%r): %r' % (eggs, output))
...     return output
...
>>> output = spam(3)
spam(3): 'spamspamspam'

```

虽然这样做是有效的，但是有一个小装饰器来解决这个问题会不会更好呢？

```py
>>> def debug(function):
...     @functools.wraps(function)
...     def _debug(*args, **kwargs):
...         output = function(*args, **kwargs)
...         print('%s(%r, %r): %r' % (function.__name__, args, kwargs, output))
...         return output
...     return _debug
...
>>>
>>> @debug
... def spam(eggs):
...     return 'spam' * (eggs % 5)
...
>>> output = spam(3)
spam((3,), {}): 'spamspamspam'

```

现在我们有一个装饰器，可以轻松地重用于打印输入、输出和函数名称的任何函数。这种类型的装饰器在日志应用程序中也非常有用，我们将在第十章中看到，*测试和日志 - 为错误做准备*。值得注意的是，即使无法修改包含原始代码的模块，也可以使用此示例。我们可以在本地包装函数，甚至在需要时对模块进行 monkey-patch：

```py
import some_module

# Regular call
some_module.some_function()

# Wrap the function
debug_some_function = debug(some_module.some_function)

# Call the debug version
debug_some_function()

# Monkey patch the original module
some_module.some_function = debug_some_function

# Now this calls the debug version of the function
some_module.some_function()
```

当然，在生产代码中使用 monkey-patching 并不是一个好主意，但在调试时可能非常有用。

## 使用装饰器进行记忆化

记忆化是使某些代码运行速度更快的一个简单技巧。这里的基本技巧是存储输入和期望输出的映射，这样你只需要计算一次值。这种技术最常见的示例之一是演示天真（递归）的斐波那契函数：

```py
>>> import functools

>>> def memoize(function):
...     function.cache = dict()
...
...     @functools.wraps(function)
...     def _memoize(*args):
...         if args not in function.cache:
...             function.cache[args] = function(*args)
...         return function.cache[args]
...     return _memoize

>>> @memoize
... def fibonacci(n):
...     if n < 2:
...         return n
...     else:
...         return fibonacci(n - 1) + fibonacci(n - 2)

>>> for i in range(1, 7):
...     print('fibonacci %d: %d' % (i, fibonacci(i)))
fibonacci 1: 1
fibonacci 2: 1
fibonacci 3: 2
fibonacci 4: 3
fibonacci 5: 5
fibonacci 6: 8

>>> fibonacci.__wrapped__.cache
{(5,): 5, (0,): 0, (6,): 8, (1,): 1, (2,): 1, (3,): 2, (4,): 3}

```

虽然这个例子在没有任何记忆化的情况下也可以正常工作，但对于更大的数字，它会使系统崩溃。对于`n=2`，函数将递归执行`fibonacci(n - 1)`和`fibonacci(n - 2)`，有效地给出指数时间复杂度。此外，对于`n=30`，斐波那契函数被调用了 2,692,537 次，尽管这仍然是可以接受的。在`n=40`时，计算将需要很长时间。

然而，记忆化版本甚至不费吹灰之力，只需要执行`31`次，`n=30`。

这个装饰器还展示了如何将上下文附加到函数本身。在这种情况下，cache 属性成为内部（包装的`fibonacci`）函数的属性，因此不同对象的额外`memoize`装饰器不会与任何其他装饰的函数发生冲突。

然而，需要注意的是，自己实现记忆化函数通常不再那么有用，因为 Python 在 Python 3.2 中引入了`lru_cache`（最近最少使用缓存）。`lru_cache`类似于前面的 memoize 函数，但更加先进。它只保持一个固定的（默认为 128）缓存大小以节省内存，并使用一些统计数据来检查是否应增加缓存大小。

为了演示`lru_cache`的内部工作原理，我们将计算`fibonacci(100)`，这将使我们的计算机忙到宇宙的尽头，而没有任何缓存。此外，为了确保我们实际上可以看到`fibonacci`函数被调用的次数，我们将添加一个额外的装饰器来跟踪计数，如下所示：

```py
>>> import functools

# Create a simple call counting decorator
>>> def counter(function):
...     function.calls = 0
...     @functools.wraps(function)
...     def _counter(*args, **kwargs):
...         function.calls += 1
...         return function(*args, **kwargs)
...     return _counter

# Create a LRU cache with size 3** 
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

你可能会想知道为什么在缓存大小为`3`的情况下我们只需要调用 101 次。这是因为我们递归地只需要`n - 1`和`n - 2`，所以在这种情况下我们不需要更大的缓存。对于其他情况，它仍然是有用的。

此外，此示例显示了对单个函数使用两个装饰器的用法。您可以将这些视为洋葱的层。第一个是外层，它朝向内部工作。在调用`fibonacci`时，将首先调用`lru_cache`，因为它是列表中的第一个装饰器。假设尚未有缓存可用，将调用`counter`装饰器。在计数器内部，将调用实际的`fibonacci`函数。

返回值当然是按相反的顺序工作的；`fibonacci`将其值返回给`counter`，后者将该值传递给`lru_cache`。

## 带（可选）参数的装饰器

以前的示例大多使用了没有任何参数的简单装饰器。正如我们已经在`lru_cache`中看到的那样，装饰器也可以接受参数，因为它们只是常规函数，但这会给装饰器增加一个额外的层。这意味着添加参数可以像下面这样简单：

```py
>>> import functools

>>> def add(extra_n=1):
...     'Add extra_n to the input of the decorated function'
...
...     # The inner function, notice that this is the actual
...     # decorator
...     def _add(function):
...         # The actual function that will be called
...         @functools.wraps(function)
...         def __add(n):
...             return function(n + extra_n)
...
...         return __add
...
...     return _add

>>> @add(extra_n=2)
... def eggs(n):
...     return 'eggs' * n

>>> eggs(2)
'eggseggseggseggs'

```

然而，可选参数是另一回事，因为它们使额外的函数层变得可选。有参数时，您需要三层，但没有参数时，您只需要两层。由于装饰器本质上是返回函数的常规函数，区别在于返回子函数或子子函数，取决于参数。这只留下一个问题——检测参数是函数还是常规参数。举例说明，使用参数的实际调用如下所示：

```py
add(extra_n=2)(eggs)(2)
```

没有参数的调用将如下所示：

```py
add(eggs)(2)
```

要检测装饰器是使用函数还是常规参数作为参数调用的，我们有几种选择，但在我看来都不是完全理想的：

+   使用关键字参数作为装饰器参数，以便常规参数始终是函数

+   检测第一个且唯一的参数是否可调用

在我看来，第一种使用关键字参数的方法是两种选项中更好的，因为它有点更明确，留下的混淆空间较少。如果您的参数也是可调用的话，第二种选项可能会有问题。

使用第一种方法，普通（非关键字）参数必须是装饰函数，其他两个检查仍然适用。我们仍然可以检查函数是否确实可调用，以及是否只有一个可用参数。以下是使用先前示例的修改版本的示例：

```py
>>> import functools

>>> def add(*args, **kwargs):
...     'Add n to the input of the decorated function'
...
...     # The default kwargs, we don't store this in kwargs
...     # because we want to make sure that args and kwargs
...     # can't both be filled
...     default_kwargs = dict(n=1)
...
...     # The inner function, notice that this is actually a
...     # decorator itself
...     def _add(function):
...         # The actual function that will be called
...         @functools.wraps(function)
...         def __add(n):
...             default_kwargs.update(kwargs)
...             return function(n + default_kwargs['n'])
...
...         return __add
...
...     if len(args) == 1 and callable(args[0]) and not kwargs:
...         # Decorator call without arguments, just call it
...         # ourselves
...         return _add(args[0])
...     elif not args and kwargs:
...         # Decorator call with arguments, this time it will
...         # automatically be executed with function as the
...         # first argument
...         default_kwargs.update(kwargs)
...         return _add
...     else:
...         raise RuntimeError('This decorator only supports '
...                            'keyword arguments')

>>> @add
... def spam(n):
...     return 'spam' * n

>>> @add(n=3)
... def eggs(n):
...     return 'eggs' * n

>>> spam(3)
'spamspamspamspam'

>>> eggs(2)
'eggseggseggseggseggs'

>>> @add(3)
... def bacon(n):
...     return 'bacon' * n
Traceback (most recent call last):   ...
RuntimeError: This decorator only supports keyword arguments

```

每当您有选择时，我建议您要么有带参数的装饰器，要么没有，而不是使用可选参数。但是，如果您有一个真正充分的理由使参数可选，那么您有一种相对安全的方法来实现这一点。

## 使用类创建装饰器

与创建常规函数装饰器的方式类似，也可以使用类来创建装饰器。毕竟，函数只是一个可调用对象，类也可以实现可调用接口。以下装饰器与我们之前使用的`debug`装饰器类似，但使用的是类而不是常规函数：

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
...         print('%s(%r, %r): %r' % (
...             self.function.__name__, args, kwargs, output))
...         return output

>>> @Debug
... def spam(eggs):
...     return 'spam' * (eggs % 5)
...
>>> output = spam(3)
spam((3,), {}): 'spamspamspam'

```

函数和类之间唯一显著的区别是，在`__init__`方法中，`functools.wraps`现在被`functools.update_wrapper`替换了。

# 装饰类函数

装饰类函数与常规函数非常相似，但您需要注意所需的第一个参数`self`——类实例。您很可能已经使用了一些类函数装饰器。例如，`classmethod`、`staticmethod`和`property`装饰器在许多不同的项目中都被使用。为了解释所有这些是如何工作的，我们将构建自己版本的`classmethod`、`staticmethod`和`property`装饰器。首先，让我们看一个简单的用于类函数的装饰器，以展示与常规装饰器的区别：

```py
>>> import functools

>>> def plus_one(function):
...     @functools.wraps(function)
...     def _plus_one(self, n):
...         return function(self, n + 1)
...     return _plus_one

>>> class Spam(object):
...     @plus_one
...     def get_eggs(self, n=2):
...         return n * 'eggs'

>>> spam = Spam()
>>> spam.get_eggs(3)
'eggseggseggseggs'
```

与常规函数一样，类函数装饰器现在会将`self`作为实例传递。没有什么意外的！

## 跳过实例 - 类方法和静态方法

`classmethod`和`staticmethod`之间的区别非常简单。`classmethod`传递的是类对象而不是类实例（`self`），而`staticmethod`完全跳过了类和实例。这使得`staticmethod`在类外部非常类似于常规函数。

在重新创建`classmethod`和`staticmethod`之前，我们需要了解这些方法的预期行为：

```py
>>> import pprint

>>> class Spam(object):
...
...     def some_instancemethod(self, *args, **kwargs):
...         print('self: %r' % self)
...         print('args: %s' % pprint.pformat(args))
...         print('kwargs: %s' % pprint.pformat(kwargs))
...
...     @classmethod
...     def some_classmethod(cls, *args, **kwargs):
...         print('cls: %r' % cls)
...         print('args: %s' % pprint.pformat(args))
...         print('kwargs: %s' % pprint.pformat(kwargs))
...
...     @staticmethod
...     def some_staticmethod(*args, **kwargs):
...         print('args: %s' % pprint.pformat(args))
...         print('kwargs: %s' % pprint.pformat(kwargs))

# Create an instance so we can compare the difference between
# executions with and without instances easily
>>> spam = Spam()

# With an instance (note the lowercase spam)
>>> spam.some_instancemethod(1, 2, a=3, b=4)
self: <...Spam object at 0x...>
args: (1, 2)
kwargs: {'a': 3, 'b': 4}

# Without an instance (note the capitalized Spam)
>>> Spam.some_instancemethod()
Traceback (most recent call last):

 **...
TypeError: some_instancemethod() missing 1 required positional argument: 'self'

# But what if we add parameters? Be very careful with these!
# Our first argument is now used as an argument, this can give
# very strange and unexpected errors
>>> Spam.some_instancemethod(1, 2, a=3, b=4)
self: 1
args: (2,)
kwargs: {'a': 3, 'b': 4}

# Classmethods are expectedly identical
>>> spam.some_classmethod(1, 2, a=3, b=4)
cls: <class '...Spam'>
args: (1, 2)
kwargs: {'a': 3, 'b': 4}

>>> Spam.some_classmethod()
cls: <class '...Spam'>
args: ()
kwargs: {}

>>> Spam.some_classmethod(1, 2, a=3, b=4)
cls: <class '...Spam'>
args: (1, 2)
kwargs: {'a': 3, 'b': 4}

# Staticmethods are also identical
>>> spam.some_staticmethod(1, 2, a=3, b=4)
args: (1, 2)
kwargs: {'a': 3, 'b': 4}

>>> Spam.some_staticmethod()
args: ()
kwargs: {}

>>> Spam.some_staticmethod(1, 2, a=3, b=4)
args: (1, 2)
kwargs: {'a': 3, 'b': 4}

```

请注意，如果在没有实例的情况下调用`some_instancemethod`，会导致缺少`self`的错误。正如预期的那样（因为在这种情况下我们没有实例化类），对于带有参数的版本，它似乎可以工作，但实际上是有问题的。这是因为现在假定第一个参数是`self`。在这种情况下显然是不正确的，因为您传递的是一个整数，但如果您传递了其他类实例，这可能是非常奇怪的错误的根源。`classmethod`和`staticmethod`都可以正确处理这种情况。

在继续使用装饰器之前，您需要了解 Python 描述符的工作原理。描述符可用于修改对象属性的绑定行为。这意味着如果将描述符用作属性的值，您可以修改在对属性进行这些操作时设置、获取和删除的值。以下是这种行为的基本示例：

```py
>>> class MoreSpam(object):
...
...     def __init__(self, more=1):
...         self.more = more
...
...     def __get__(self, instance, cls):
...         return self.more + instance.spam
...
...     def __set__(self, instance, value):
...         instance.spam = value - self.more

>>> class Spam(object):
...
...     more_spam = MoreSpam(5)
...
...     def __init__(self, spam):
...         self.spam = spam

>>> spam = Spam(1)
>>> spam.spam
1
>>> spam.more_spam
6

>>> spam.more_spam = 10
>>> spam.spam
5

```

正如你所看到的，无论我们从`more_spam`中设置或获取值，实际上都会调用`MoreSpam`上的`__get__`或`__set__`。对于自动转换和类型检查非常有用，我们将在下一段中看到的`property`装饰器只是这种技术的更方便的实现。

现在我们知道了描述符是如何工作的，我们可以继续创建`classmethod`和`staticmethod`装饰器。对于这两个装饰器，我们只需要修改`__get__`而不是`__call__`，以便我们可以控制传递哪种类型的实例（或根本不传递）：

```py
import functools

class ClassMethod(object):

    def __init__(self, method):
        self.method = method

    def __get__(self, instance, cls):
        @functools.wraps(self.method)
        def method(*args, **kwargs):
            return self.method(cls, *args, **kwargs)
        return method

class StaticMethod(object):

    def __init__(self, method):
        self.method = method

    def __get__(self, instance, cls):
        return self.method
```

`ClassMethod`装饰器仍然具有一个子函数来实际生成一个可工作的装饰器。看看这个函数，你很可能猜到它的功能。它不是将`instance`作为`self.method`的第一个参数传递，而是传递`cls`。

`StaticMethod`更简单，因为它完全忽略了`instance`和`cls`。它可以直接返回原始方法而不进行任何修改。因为它返回原始方法而不进行任何修改，我们也不需要`functools.wraps`调用。

## 属性 - 智能描述符用法

`property`装饰器可能是 Python 中最常用的装饰器。它允许您向现有实例属性添加 getter/setter，以便您可以在将它们设置为实例属性之前添加验证器和修改值。`property`装饰器可以用作赋值和装饰器。下面的示例展示了这两种语法，以便我们知道从`property`装饰器中可以期望到什么：

```py
>>> class Spam(object):
...
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

>>> spam = Spam()
>>> spam.eggs = 123
setting eggs to 123
>>> spam.eggs
getting eggs
123
>>> del spam.eggs
deleting eggs

```

### 注意

请注意，`property`装饰器仅在类继承`object`时才起作用。

与我们实现`classmethod`和`staticmethod`装饰器的方式类似，我们再次需要 Python 描述符。这一次，我们需要描述符的全部功能，而不仅仅是`__get__`，还有`__set__`和`__delete__`：

```py
class Property(object):
    def __init__(self, fget=None, fset=None, fdel=None,
                 doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        # If no specific documentation is available, copy it
        # from the getter
        if fget and not doc:
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, instance, cls):
        if instance is None:
            # Redirect class (not instance) properties to
            # self
            return self
        elif self.fget:
            return self.fget(instance)
        else:
            raise AttributeError('unreadable attribute')

    def __set__(self, instance, value):
        if self.fset:
            self.fset(instance, value)
        else:
            raise AttributeError("can't set attribute")

    def __delete__(self, instance):
        if self.fdel:
            self.fdel(instance)
        else:
            raise AttributeError("can't delete attribute")

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel)
```

正如你所看到的，大部分`Property`实现只是描述符方法的实现。`getter`、`setter`和`deleter`函数只是使装饰器的使用变得可能的快捷方式，这就是为什么如果没有`instance`可用，我们必须`return self`。

当然，还有更多实现这种效果的方法。在前面的段落中，我们看到了裸描述符的实现，在我们之前的例子中，我们看到了属性装饰器。一个更通用的类的解决方案是实现 `__getattr__` 或 `__getattribute__`。这里有一个简单的演示：

```py
>>> class Spam(object):
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

>>> spam = Spam()

>>> spam.a
Getting 'a'
'Undefined'

>>> spam.a = 1
Setting 'a' to 1

>>> spam.a
Getting 'a'
1

>>> del spam.a
Deleting 'a'

```

`__getattr__` 方法首先在 `instance.__dict__` 中查找键，并且只有在键不存在时才会被调用。这就是为什么我们从来没有看到 `registry` 属性的 `__getattr__`。`__getattribute__` 方法在所有情况下都会被调用，这使得它使用起来更加危险。使用 `__getattribute__` 方法时，你需要对 `registry` 进行特定的排除，因为如果你尝试访问 `self.registry`，它将会被递归执行。

很少需要查看描述符，但它们被几个内部 Python 进程使用，比如在继承类时使用 `super()` 方法。

# 装饰类

Python 2.6 引入了类装饰器语法。与函数装饰器语法一样，这实际上也不是一种新技术。即使没有语法，一个类也可以通过简单地执行 `DecoratedClass = decorator(RegularClass)` 来装饰。在前面的段落中，你应该已经熟悉了编写装饰器。类装饰器与常规装饰器没有什么不同，只是它们接受一个类而不是一个函数。与函数一样，这发生在声明时而不是在实例化/调用时。

由于有很多修改类工作方式的替代方法，比如标准继承、混入和元类（关于这一点在第八章中有更多介绍，*元类 - 使类（而不是实例）更智能*），类装饰器从来都不是绝对必需的。这并不减少它们的用处，但它确实解释了为什么你很可能不会在野外看到太多类装饰的例子。

## 单例 - 只有一个实例的类

单例是始终只允许存在一个实例的类。因此，你总是得到相同的实例，而不是为你的调用获取一个特定的实例。这对于诸如数据库连接池之类的事物非常有用，你不想一直打开连接，而是想重用原始连接。

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
... class Spam(object):
...     def __init__(self):
...         print('Executing init')

>>> a = Spam()
Executing init
>>> b = Spam()

>>> a is b
True

>>> a.x = 123
>>> b.x
123

```

正如你在 `a is b` 比较中看到的，两个对象具有相同的标识，所以我们可以得出它们确实是同一个对象。与常规装饰器一样，由于 `functools.wraps` 功能，如果需要，我们仍然可以通过 `Spam.__wrapped__` 访问原始类。

### 注意

`is` 运算符通过标识比较对象，这在 CPython 中是以内存地址实现的。如果 `a is b` 返回 `True`，我们可以得出结论，`a` 和 `b` 都是同一个实例。

## 完全排序 - 简单的可排序类

在某个时候，你可能需要对数据结构进行排序。虽然可以使用 `sorted` 函数的 key 参数轻松实现这一点，但如果你经常需要这样做，还有一种更方便的方法 - 通过实现 `__gt__`、`__ge__`、`__lt__`、`__le__` 和 `__eq__` 函数。这似乎有点冗长，不是吗？如果你想要最佳性能，这仍然是一个好主意，但如果你可以承受一点性能损失和一些稍微复杂的堆栈跟踪，那么 `total_ordering` 可能是一个不错的选择。`total_ordering` 类装饰器可以基于具有 `__eq__` 函数和比较函数之一（`__lt__`、`__le__`、`__gt__` 或 `__ge__`）的类实现所有必需的排序函数。这意味着你可以严重缩短你的函数定义。让我们比较常规的函数和使用 `total_ordering` 装饰器的函数：

```py
>>> import functools

>>> class Value(object):
...     def __init__(self, value):
...         self.value = value
...
...     def __repr__(self):
...         return '<%s[%d]>' % (self.__class__, self.value)

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

>>> numbers = [4, 2, 3, 4]
>>> spams = [Spam(n) for n in numbers]
>>> eggs = [Egg(n) for n in numbers]

>>> spams
[<<class 'H05.Spam'>[4]>, <<class 'H05.Spam'>[2]>,
<<class 'H05.Spam'>[3]>, <<class 'H05.Spam'>[4]>]

>>> eggs
[<<class 'H05.Egg'>[4]>, <<class 'H05.Egg'>[2]>,
<<class 'H05.Egg'>[3]>, <<class 'H05.Egg'>[4]>]

>>> sorted(spams)
[<<class 'H05.Spam'>[2]>, <<class 'H05.Spam'>[3]>,
<<class 'H05.Spam'>[4]>, <<class 'H05.Spam'>[4]>]

>>> sorted(eggs)
[<<class 'H05.Egg'>[2]>, <<class 'H05.Egg'>[3]>,
<<class 'H05.Egg'>[4]>, <<class 'H05.Egg'>[4]>]

# Sorting using key is of course still possible and in this case
# perhaps just as easy:
>>> values = [Value(n) for n in numbers]
>>> values
[<<class 'H05.Value'>[4]>, <<class 'H05.Value'>[2]>,
<<class 'H05.Value'>[3]>, <<class 'H05.Value'>[4]>]

>>> sorted(values, key=lambda v: v.value)
[<<class 'H05.Value'>[2]>, <<class 'H05.Value'>[3]>,
<<class 'H05.Value'>[4]>, <<class 'H05.Value'>[4]>]

```

现在，你可能会想，“为什么没有一个类装饰器来使用指定的键属性使类可排序？”嗯，这确实可能是`functools`库的一个好主意，但它还没有。所以让我们看看我们如何实现类似的东西：

```py
>>> def sort_by_attribute(attr, keyfunc=getattr):
...     def _sort_by_attribute(cls):
...         def __gt__(self, other):
...             return getattr(self, attr) > getattr(other, attr)
...
...         def __ge__(self, other):
...             return getattr(self, attr) >= getattr(other, attr)
...
...         def __lt__(self, other):
...             return getattr(self, attr) < getattr(other, attr)
...
...         def __le__(self, other):
...             return getattr(self, attr) <= getattr(other, attr)
...
...         def __eq__(self, other):
...             return getattr(self, attr) <= getattr(other, attr)
...
...         cls.__gt__ = __gt__
...         cls.__ge__ = __ge__
...         cls.__lt__ = __lt__
...         cls.__le__ = __le__
...         cls.__eq__ = __eq__
...
...         return cls
...     return _sort_by_attribute

>>> class Value(object):
...     def __init__(self, value):
...         self.value = value
...
...     def __repr__(self):
...         return '<%s[%d]>' % (self.__class__, self.value)

>>> @sort_by_attribute('value')
... class Spam(Value):
...     pass

>>> numbers = [4, 2, 3, 4]
>>> spams = [Spam(n) for n in numbers]
>>> sorted(spams)
[<<class '...Spam'>[2]>, <<class '...Spam'>[3]>,
<<class '...Spam'>[4]>, <<class '...Spam'>[4]>]

```

当然，这极大地简化了可排序类的制作。如果你宁愿使用自己的键函数而不是`getattr`，那就更容易了。只需用`key_function(self)`替换`getattr(self, attr)`调用，对`other`也这样做，并将装饰器的参数更改为你的函数。你甚至可以将其用作基本函数，并通过简单地传递一个包装的`getattr`函数来实现`sort_by_attribute`。

# 有用的装饰器

除了本章已经提到的之外，Python 还捆绑了一些其他有用的装饰器。还有一些不在标准库中的（还没有？）。

## Python 中的单分派——多态

如果你以前使用过 C++或 Java，你可能已经习惯了可用的特定多态性——根据参数类型调用不同的函数。Python 作为一种动态类型语言，大多数人不会期望存在单分派模式的可能性。然而，Python 不仅是一种动态类型的语言，而且是一种强类型的语言，这意味着我们可以依赖我们收到的类型。

### 注意

动态类型的语言不需要严格的类型定义。另一方面，像 C 这样的语言需要以下内容来声明一个整数：

```py
int some_integer = 123;
```

Python 只是接受你的值有一个类型：

```py
some_integer = 123
```

然而，与 JavaScript 和 PHP 等语言相反，Python 几乎不进行隐式类型转换。在 Python 中，以下内容将返回错误，而在 JavaScript 中将无任何问题地执行：

```py
'spam' + 5
```

在 Python 中，结果是`TypeError`。在 Javascript 中，是`'spam5'`。

单分派的想法是，根据你传递的类型，调用正确的函数。由于在 Python 中`str + int`会导致错误，这可以非常方便地在将参数传递给函数之前自动转换你的参数。这对于将函数的实际工作与类型转换分离开来非常有用。

自 Python 3.4 以来，有一个装饰器可以轻松实现 Python 中的单分派模式。对于那些需要处理与正常执行不同的特定类型的情况之一。这是一个基本的例子：

```py
>>> import functools

>>> @functools.singledispatch
... def printer(value):
...     print('other: %r' % value)

>>> @printer.register(str)
... def str_printer(value):
...     print(value)

>>> @printer.register(int)
... def int_printer(value):
...     printer('int: %d' % value)

>>> @printer.register(dict)
... def dict_printer(value):
...     printer('dict:')
...     for k, v in sorted(value.items()):
...         printer('    key: %r, value: %r' % (k, v))

>>> printer('spam')
spam

>>> printer([1, 2, 3])
other: [1, 2, 3]

>>> printer(123)
int: 123

>>> printer({'a': 1, 'b': 2})
dict:
 **key: 'a', value: 1
 **key: 'b', value: 2

```

看到了吗，根据类型，其他函数被调用了吗？这种模式对于减少接受多种类型参数的单个函数的复杂性非常有用。

### 注意

在命名函数时，请确保不要覆盖原始的`singledispatch`函数。如果我们将`str_printer`命名为`printer`，它将覆盖最初的`printer`函数。这将使得无法访问原始的`printer`函数，并且在此之后的所有`register`操作也将失败。

现在，一个稍微更有用的例子——区分文件名和文件处理程序：

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

所以现在我们有了一个单一的`write_as_json`函数；它根据类型调用正确的代码。如果是`str`或`bytes`对象，它将自动打开文件并调用`write_as_json`的常规版本，该版本接受文件对象。

当然，编写一个能够做到这一点的装饰器并不难，但在基本库中拥有它仍然非常方便。这肯定比在函数中进行几次`isinstance`调用要好。要查看将调用哪个函数，可以使用特定类型的`write_as_json.dispatch`函数。传递一个`str`时，将得到`write_as_json_filename`函数。应该注意，分派函数的名称是完全任意的。它们当然可以像常规函数一样访问，但你可以随意命名它们。

要检查已注册的类型，可以通过`write_as_json.registry`访问字典注册表：

```py
>>> write_as_json.registry.keys()
dict_keys([<class 'bytes'>, <class 'object'>, <class 'str'>])

```

## 上下文管理器，简化了 with 语句

使用`contextmanager`类，我们可以很容易地创建上下文包装器。上下文包装器在使用`with`语句时使用。一个例子是`open`函数，它也可以作为上下文包装器工作，允许你使用以下代码：

```py
with open(filename) as fh:
    pass
```

让我们暂时假设`open`函数不能作为上下文管理器使用，我们需要构建自己的函数来实现这一点。创建上下文管理器的标准方法是创建一个实现`__enter__`和`__exit__`方法的类，但这有点冗长。我们可以让它更短更简单：

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

简单，对吧？然而，我应该提到，对于这种特定情况——对象的关闭——在`contextlib`中有一个专门的函数，它甚至更容易使用。让我们来演示一下：

```py
>>> import contextlib

>>> with contextlib.closing(open('test.txt', 'a')) as fh:
...     print('Yet another test', file=fh)

```

对于`file`对象，这当然是不需要的，因为它已经作为上下文管理器起作用。然而，一些对象，比如`urllib`发出的请求，不支持以这种方式自动关闭，并且从这个函数中受益。

但等等；还有更多！除了可以在`with`语句中使用之外，`contextmanager`的结果实际上也可以作为装饰器使用，自 Python 3.2 起。在较早的 Python 版本中，它只是一个小包装器，但自 Python 3.2 起，它基于`ContextDecorator`类，这使它成为一个装饰器。之前的装饰器并不适合这个任务，因为它产生了一个结果（关于这一点，可以在第六章中了解更多，*生成器和协程-无限，一步一步*），但我们可以考虑其他函数：

```py
>>> @contextlib.contextmanager
... def debug(name):
...     print('Debugging %r:' % name)
...     yield
...     print('End of debugging %r' % name)

>>> @debug('spam')
... def spam():
...     print('This is the inside of our spam function')

>>> spam()
Debugging 'spam':
This is the inside of our spam function
End of debugging 'spam'

```

这有很多很好的用例，但至少，它是一个方便的方式来在上下文中包装一个函数，而不需要所有（嵌套的）`with`语句。

## 验证、类型检查和转换

虽然在 Python 中检查类型通常不是最佳选择，但有时如果你知道你需要一个特定的类型（或者可以转换为该类型的东西），它可能是有用的。为了方便起见，Python 3.5 引入了类型提示系统，这样你就可以这样做：

```py
def spam(eggs: int):
    pass
```

由于 Python 3.5 还不太常见，这里有一个装饰器，它实现了更高级的类型检查。为了允许这种类型的检查，必须使用一些魔法，特别是使用`inspect`模块。就我个人而言，我不太喜欢检查代码来执行这样的技巧，因为它们很容易被破坏。这段代码实际上在函数和这个装饰器之间使用一个常规装饰器（不复制`argspec`）时会出错，但它仍然是一个很好的例子：

```py
>>> import inspect
>>> import functools

>>> def to_int(name, minimum=None, maximum=None):
...     def _to_int(function):
...         # Use the method signature to map *args to named
...         # arguments
...         signature = inspect.signature(function)
...
...         # Unfortunately functools.wraps doesn't copy the
...         # signature (yet) so we do it manually.
...         # For more info: http://bugs.python.org/issue23764
...         @functools.wraps(function, ['__signature__'])
...         @functools.wraps(function)
...         def __to_int(*args, **kwargs):
...             # Bind all arguments to the names so we get a single
...             # mapping of all arguments
...             bound = signature.bind(*args, **kwargs)
...
...             # Make sure the value is (convertible to) an integer
...             default = signature.parameters[name].default
...             value = int(bound.arguments.get(name, default))
...
...             # Make sure it's within the allowed range
...             if minimum is not None:
...                 assert value >= minimum, (
...                     '%s should be at least %r, got: %r' %
...                     (name, minimum, value))
...
...             if maximum is not None:
...                 assert value <= maximum, (
...                     '%s should be at most %r, got: %r' %
...                     (name, maximum, value))
...
...             return function(*args, **kwargs)
...         return __to_int
...     return _to_int

>>> @to_int('a', minimum=10)
... @to_int('b', maximum=10)
... @to_int('c')
... def spam(a, b, c=10):
...     print('a', a)
...     print('b', b)
...     print('c', c)

>>> spam(10, b=0)
a 10
b 0
c 10

>>> spam(a=20, b=10)
a 20
b 10
c 10

>>> spam(1, 2, 3)
Traceback (most recent call last):
 **...
AssertionError: a should be at least 10, got: 1

>>> spam()
Traceback (most recent call last):
 **...
TypeError: 'a' parameter lacking default value

>>> spam('spam', {})
Traceback (most recent call last):
 **...
ValueError: invalid literal for int() with base 10: 'spam'

```

由于`inspect`的魔法，我仍然不确定是否推荐像这样使用装饰器。相反，我会选择一个更简单的版本，它完全不使用`inspect`，只是从`kwargs`中解析参数：

```py
>>> import functools

>>> def to_int(name, minimum=None, maximum=None):
...     def _to_int(function):
...         @functools.wraps(function)
...         def __to_int(**kwargs):
...             value = int(kwargs.get(name))
...
...             # Make sure it's within the allowed range
...             if minimum is not None:
...                 assert value >= minimum, (
...                     '%s should be at least %r, got: %r' %
...                     (name, minimum, value))
...
...             if maximum is not None:
...                 assert value <= maximum, (
...                     '%s should be at most %r, got: %r' %
...                     (name, maximum, value))
...
...             return function(**kwargs)
...         return __to_int
...     return _to_int

>>> @to_int('a', minimum=10)
... @to_int('b', maximum=10)
... def spam(a, b):
...     print('a', a)
...     print('b', b)

>>> spam(a=20, b=10)
a 20
b 10

>>> spam(a=1, b=10)
Traceback (most recent call last):
 **...
AssertionError: a should be at least 10, got: 1

```

然而，正如所示，支持`args`和`kwargs`并不是不可能的，只要记住默认情况下不会复制`__signature__`。没有`__signature__`，inspect 模块就不知道哪些参数是允许的，哪些是不允许的。

### 注意

缺少的`__signature__`问题目前正在讨论中，可能会在未来的 Python 版本中得到解决：

[`bugs.python.org/issue23764`](http://bugs.python.org/issue23764)。

## 无用的警告-如何忽略它们

通常在编写 Python 代码时，警告在你第一次编写代码时非常有用。但在执行代码时，每次运行脚本/应用程序时得到相同的消息是没有用的。因此，让我们创建一些代码，可以轻松隐藏预期的警告，但不是所有的警告，这样我们就可以轻松地捕获新的警告：

```py
import warnings
import functools

def ignore_warning(warning, count=None):
    def _ignore_warning(function):
        @functools.wraps(function)
        def __ignore_warning(*args, **kwargs):
            # Execute the code while recording all warnings
            with warnings.catch_warnings(record=True) as ws:
                # Catch all warnings of this type
                warnings.simplefilter('always', warning)
                # Execute the function
                result = function(*args, **kwargs)

            # Now that all code was executed and the warnings
            # collected, re-send all warnings that are beyond our
            # expected number of warnings
            if count is not None:
                for w in ws[count:]:
                    warnings.showwarning(
                        message=w.message,
                        category=w.category,
                        filename=w.filename,
                        lineno=w.lineno,
                        file=w.file,
                        line=w.line,
                    )

            return result
        return __ignore_warning
    return _ignore_warning

@ignore_warning(DeprecationWarning, count=1)
def spam():
    warnings.warn('deprecation 1', DeprecationWarning)
    warnings.warn('deprecation 2', DeprecationWarning)
```

使用这种方法，我们可以捕获第一个（预期的）警告，并且仍然可以看到第二个（不期望的）警告。

# 总结

本章向我们展示了装饰器可以用于简化代码并向非常简单的函数添加一些相当复杂的行为的一些地方。事实上，大多数装饰器比直接添加功能的常规函数更复杂，但将相同的模式应用于许多函数和类的附加优势通常是非常值得的。

装饰器有很多用途，可以使您的函数和类更智能、更方便使用：

+   调试

+   验证

+   参数方便（预填充或转换参数）

+   输出方便（将输出转换为特定类型）

本章最重要的收获应该是在包装函数时永远不要忘记`functools.wraps`。由于（意外的）行为修改，调试装饰函数可能会非常困难，但丢失属性也会使这个问题变得更糟。

下一章将向我们展示如何以及何时使用`生成器`和`协程`。本章已经向我们展示了`with`语句的使用，但`生成器`和`协程`在这方面更进一步。尽管如此，我们仍然经常使用装饰器，所以确保你对它们的工作原理有很好的理解。
