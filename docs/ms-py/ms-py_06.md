# 第六章：生成器和协程-无限，一步一步

生成器是一种通过函数生成值的特定类型的迭代器。传统方法构建并返回项目的`list`，而生成器只会在调用者请求时单独`yield`每个值。这种方法有几个好处：

+   生成器完全暂停执行，直到下一个值被产生，这使得它们完全是惰性的。如果从生成器中获取五个项目，只会生成五个项目，因此不需要其他计算。

+   生成器不需要保存值。而传统函数需要创建一个`list`并存储所有结果，直到它们被返回，生成器只需要存储一个单一的值。

+   生成器可以具有无限的大小。没有必要在某一点停止。

然而，这些好处是有代价的。这些好处的直接结果是一些缺点：

+   在处理完成之前，您永远不知道还有多少值；甚至可能是无限的。这在某些情况下使用是危险的；执行`list(some_infinite_generator)`将耗尽内存。

+   您无法切片生成器。

+   您无法在产生指定的项目之前获取所有值。

+   您无法重新启动生成器。所有值只产生一次。

除了生成器之外，还有一种变体的生成器语法，可以创建协程。协程是允许进行多任务处理而不需要多个线程或进程的函数。生成器只能向调用者产生值，而协程实际上可以在运行时从调用者那里接收值。虽然这种技术有一些限制，但如果符合您的目的，它可以以非常低的成本实现出色的性能。

简而言之，本章涵盖的主题有：

+   生成器的特点和用途

+   生成器推导

+   生成器函数

+   生成器类

+   捆绑生成器

+   协程

# 生成器是什么？

生成器，最简单的形式是一个函数，它一次返回一个元素，而不是返回一组项目。这样做的最重要的优点是它需要非常少的内存，而且不需要预先定义的大小。创建一个无限的生成器（比如在第四章中讨论的`itertools.count`迭代器，*功能编程-可读性与简洁性*）实际上是相当容易的，但当然也是有代价的。没有对象的大小可用，使得某些模式难以实现。

编写生成器（作为函数）的基本技巧是使用`yield`语句。让我们以`itertools.count`生成器为例，并用一个`stop`变量扩展它：

```py
>>> def count(start=0, step=1, stop=10):
...     n = start
...     while n <= stop:
...         yield n
...         n += step

>>> for x in count(10, 2.5, 20):
...     print(x)
10
12.5
15.0
17.5
20.0

```

由于生成器可能是无限的，因此需要谨慎。如果没有`stop`变量，简单地执行`list(count())`将很快导致内存不足的情况。

那么这是如何工作的呢？这只是一个普通的`for`循环，但这与返回项目列表的常规方法之间的重要区别在于`yield`语句一次返回一个项目。这里需要注意的一点是，`return`语句会导致`StopIteration`，并且将某些东西传递给`return`将成为`StopIteration`的参数。应该注意的是，这种行为在 Python 3.3 中发生了变化；在 Python 3.2 和更早的版本中，除了`None`之外，根本不可能返回任何东西。这里有一个例子：

```py
>>> def generator():
...     yield 'this is a generator'
...     return 'returning from a generator'

>>> g = generator()
>>> next(g)
'this is a generator'
>>> next(g)
Traceback (most recent call last):
 **...
StopIteration: returning from a generator

```

当然，与以往一样，有多种使用 Python 创建生成器的方法。除了函数之外，还有生成器推导和类可以做同样的事情。生成器推导与列表推导几乎完全相同，但使用括号而不是方括号，例如：

```py
>>> generator = (x ** 2 for x in range(4))

>>> for x in generator:
...    print(x)
0
1
4
9

```

为了完整起见，`count`函数的类版本如下：

```py
>>> class Count(object):
...     def __init__(self, start=0, step=1, stop=10):
...         self.n = start
...         self.step = step
...         self.stop = stop
...
...     def __iter__(self):
...         return self
...
...     def __next__(self):
...         n = self.n
...         if n > self.stop:
...             raise StopIteration()
...
...         self.n += self.step
...         return n

>>> for x in Count(10, 2.5, 20):
...     print(x)
10
12.5
15.0
17.5
20.0

```

类和基于函数的方法之间最大的区别是你需要显式地引发`StopIteration`而不是简单地返回它。除此之外，它们非常相似，尽管基于类的版本显然增加了一些冗余。

## 生成器的优缺点

你已经看到了一些生成器的例子，并了解了你可以用它们做什么的基础知识。然而，重要的是要记住它们的优缺点。

以下是最重要的优点：

+   内存使用。项目可以一次处理一个，因此通常不需要将整个列表保存在内存中。

+   结果可能取决于外部因素，而不是具有静态列表。例如，考虑处理队列/堆栈。

+   生成器是懒惰的。这意味着如果你只使用生成器的前五个结果，剩下的甚至不会被计算。

+   一般来说，编写生成函数比编写列表生成函数更简单。

最重要的缺点：

+   结果只能使用一次。处理生成器的结果后，不能再次使用。

+   在处理完成之前，大小是未知的，这可能对某些算法有害。

+   生成器是不可索引的，这意味着`some_generator[5]`是行不通的。

考虑到所有的优缺点，我的一般建议是尽可能使用生成器，只有在实际需要时才返回`list`或`tuple`。将生成器转换为`list`就像`list(some_generator)`一样简单，所以这不应该阻止你，因为生成函数往往比生成`list`的等效函数更简单。

内存使用的优势是可以理解的；一个项目需要的内存比许多项目少。然而，懒惰部分需要一些额外的解释，因为它有一个小问题：

```py
>>> def generator():
...     print('Before 1')
...     yield 1
...     print('After 1')
...     print('Before 2')
...     yield 2
...     print('After 2')
...     print('Before 3')
...     yield 3
...     print('After 3')

>>> g = generator()
>>> print('Got %d' % next(g))
Before 1
Got 1

>>> print('Got %d' % next(g))
After 1
Before 2
Got 2

```

正如你所看到的，生成器在`yield`语句后有效地冻结，所以即使`After 2`在`3`被产生之前也不会打印。

这有重要的优势，但这绝对是你需要考虑的事情。你不能在`yield`后立即清理，因为它直到下一个`yield`才会执行。

## 管道-生成器的有效使用

生成器的理论可能性是无限的（无意冒犯），但它们的实际用途可能很难找到。如果你熟悉 Unix/Linux shell，你可能以前使用过管道，比如`ps aux | grep python'`，例如列出所有 Python 进程。当然，有很多方法可以做到这一点，但让我们在 Python 中模拟类似的东西，以便看到一个实际的例子。为了创建一个简单和一致的输出，我们将创建一个名为`lines.txt`的文件，其中包含以下行：

```py
spam
eggs
spam spam
eggs eggs
spam spam spam
eggs eggs eggs
```

现在，让我们来看下面的 Linux/Unix/Mac shell 命令，以读取带有一些修改的文件：

```py
# cat lines.txt | grep spam | sed 's/spam/bacon/g'
bacon
bacon bacon
bacon bacon bacon

```

这使用`cat`读取文件，使用`grep`输出包含`spam`的所有行，并使用`sed`命令将`spam`替换为`bacon`。现在让我们看看如何可以利用 Python 生成器来重新创建这个过程：

```py
>>> def cat(filename):
...     for line in open(filename):
...         yield line.rstrip()
...
>>> def grep(sequence, search):
...     for line in sequence:
...         if search in line:
...             yield line
...
>>> def replace(sequence, search, replace):
...     for line in sequence:
...         yield line.replace(search, replace)
...
>>> lines = cat('lines.txt')
>>> spam_lines = grep(lines, 'spam')
>>> bacon_lines = replace(spam_lines, 'spam', 'bacon')

>>> for line in bacon_lines:
...     print(line)
...
bacon
bacon bacon
bacon bacon bacon

# Or the one-line version, fits within 78 characters:
>>> for line in replace(grep(cat('lines.txt'), 'spam'),
...                     'spam', 'bacon'):
...     print(line)
...
bacon
bacon bacon
bacon bacon bacon

```

这就是生成器的最大优势。你可以用很少的性能影响多次包装一个列表或序列。在请求值之前，涉及的任何函数都不会执行任何操作。

## tee-多次使用输出

如前所述，生成器最大的缺点之一是结果只能使用一次。幸运的是，Python 有一个函数允许你将输出复制到多个生成器。如果你习惯在命令行 shell 中工作，`tee`这个名字可能对你来说很熟悉。`tee`程序允许你将输出同时写到屏幕和文件，这样你就可以在保持实时查看的同时存储输出。

Python 版本的`itertools.tee`也做了类似的事情，只是它返回了几个迭代器，允许你分别处理结果。

默认情况下，`tee`会将您的生成器分成一个包含两个不同生成器的元组，这就是为什么元组解包在这里能很好地工作。通过传递`n`参数，这可以很容易地改变以支持超过 2 个生成器。这是一个例子：

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
 **...
StopIteration

```

看到这段代码后，您可能会对`tee`的内存使用情况感到好奇。它是否需要为您存储整个列表？幸运的是，不需要。`tee`函数在处理这个问题时非常聪明。假设您有一个包含 1,000 个项的生成器，并且同时从`a`中读取前 100 个项和从`b`中读取前 75 个项。那么`tee`将只在内存中保留差异（`100-75=25`个项），并在您迭代结果时丢弃其余的部分。

当然，`tee`是否是您的最佳解决方案取决于情况。如果实例`a`在实例`b`之前从头到（几乎）末尾被读取，那么使用`tee`就不是一个好主意。将生成器简单地转换为`list`会更快，因为它涉及的操作要少得多。

## 从生成器生成

正如我们之前所看到的，我们可以使用生成器来过滤、修改、添加和删除项。然而，在许多情况下，您会注意到在编写生成器时，您将从子生成器和/或序列中返回。一个例子是使用`itertools`库创建`powerset`时：

```py
>>> import itertools

>>> def powerset(sequence):
...     for size in range(len(sequence) + 1):
...         for item in itertools.combinations(sequence, size):
...             yield item

>>> for result in powerset('abc'):
...     print(result)
()
('a',)
('b',)
('c',)
('a', 'b')
('a', 'c')
('b', 'c')
('a', 'b', 'c')

```

这种模式是如此常见，以至于`yield`语法实际上得到了增强，使得这更加容易。Python 3.3 引入了`yield from`语法，使这种常见模式变得更加简单：

```py
>>> import itertools

>>> def powerset(sequence):
...     for size in range(len(sequence) + 1):
...         yield from itertools.combinations(sequence, size)

>>> for result in powerset('abc'):
...     print(result)
()
('a',)
('b',)
('c',)
('a', 'b')
('a', 'c')
('b', 'c')
('a', 'b', 'c')

```

这就是你只用三行代码创建一个幂集的方法。

也许，这种情况下更有用的例子是递归地扁平化一个序列。

```py
>>> def flatten(sequence):
...     for item in sequence:
...         try:
...             yield from flatten(item)
...         except TypeError:
...             yield item
...
>>> list(flatten([1, [2, [3, [4, 5], 6], 7], 8]))
[1, 2, 3, 4, 5, 6, 7, 8]

```

请注意，此代码使用`TypeError`来检测非可迭代对象。结果是，如果序列（可能是一个生成器）返回`TypeError`，它将默默地隐藏它。

还要注意，这是一个非常基本的扁平化函数，没有任何类型检查。例如，包含`str`的可迭代对象将被递归地扁平化，直到达到最大递归深度，因为`str`中的每个项也会返回一个`str`。

## 上下文管理器

与本书中描述的大多数技术一样，Python 也捆绑了一些有用的生成器。其中一些（例如`itertools`和`contextlib.contextmanager`）已经在第四章和第五章中讨论过，但我们可以使用一些额外的例子来演示它们可以多么简单和强大。

Python 上下文管理器似乎与生成器没有直接关联，但这是它们内部使用的一个很大的部分：

```py
>>> import datetime
>>> import contextlib

# Context manager that shows how long a context was active
>>> @contextlib.contextmanager
... def timer(name):
...     start_time = datetime.datetime.now()
...     yield
...     stop_time = datetime.datetime.now()
...     print('%s took %s' % (name, stop_time - start_time))

# The write to log function writes all stdout (regular print data) to
# a file. The contextlib.redirect_stdout context wrapper
# temporarily redirects standard output to a given file handle, in
# this case the file we just opened for writing.
>>> @contextlib.contextmanager
... def write_to_log(name):
...     with open('%s.txt' % name, 'w') as fh:
...         with contextlib.redirect_stdout(fh):
...             with timer(name):
...                 yield

# Use the context manager as a decorator
>>> @write_to_log('some function')
... def some_function():
...     print('This function takes a bit of time to execute')
...     ...
...     print('Do more...')

>>> some_function()

```

虽然所有这些都可以正常工作，但是三层上下文管理器往往会变得有点难以阅读。通常，装饰器可以解决这个问题。然而，在这种情况下，我们需要一个上下文管理器的输出作为下一个上下文管理器的输入。

这就是`ExitStack`的用武之地。它允许轻松地组合多个上下文管理器：

```py
>>> import contextlib

>>> @contextlib.contextmanager
... def write_to_log(name):
...     with contextlib.ExitStack() as stack:
...         fh = stack.enter_context(open('stdout.txt', 'w'))
...         stack.enter_context(contextlib.redirect_stdout(fh))
...         stack.enter_context(timer(name))
...
...         yield

>>> @write_to_log('some function')
... def some_function():
...     print('This function takes a bit of time to execute')
...     ...
...     print('Do more...')

>>> some_function()

```

看起来至少简单了一点，不是吗？虽然在这种情况下必要性有限，但当您需要进行特定的拆卸时，`ExitStack`的便利性很快就会显现出来。除了之前看到的自动处理外，还可以将上下文传递给一个新的`ExitStack`并手动处理关闭：

```py
>>> import contextlib

>>> with contextlib.ExitStack() as stack:
...     spam_fh = stack.enter_context(open('spam.txt', 'w'))
...     eggs_fh = stack.enter_context(open('eggs.txt', 'w'))
...     spam_bytes_written = spam_fh.write('writing to spam')
...     eggs_bytes_written = eggs_fh.write('writing to eggs')
...     # Move the contexts to a new ExitStack and store the
...     # close method
...     close_handlers = stack.pop_all().close

>>> spam_bytes_written = spam_fh.write('still writing to spam')
>>> eggs_bytes_written = eggs_fh.write('still writing to eggs')

# After closing we can't write anymore
>>> close_handlers()
>>> spam_bytes_written = spam_fh.write('cant write anymore')
Traceback (most recent call last):
 **...
ValueError: I/O operation on closed file.

```

大多数`contextlib`函数在 Python 手册中都有详尽的文档。特别是`ExitStack`，可以在[`docs.python.org/3/library/contextlib.html#contextlib.ExitStack`](https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack)上找到许多示例。我建议密切关注`contextlib`文档，因为它在每个 Python 版本中都有很大的改进。

# 协程

协程是通过多个入口点提供非抢占式多任务处理的子例程。基本前提是，协程允许两个函数在运行时相互通信。通常，这种类型的通信仅保留给多任务处理解决方案，但协程以几乎没有额外性能成本的相对简单方式提供了这种实现。

由于生成器默认是惰性的，协程的工作方式是非常明显的。直到结果被消耗，生成器都会休眠；但在消耗结果时，生成器会变得活跃。普通生成器和协程之间的区别在于，协程不仅仅将值返回给调用函数，还可以接收值。

## 一个基本的例子

在前面的段落中，我们看到了普通生成器如何产出值。但生成器能做的不仅仅是这些。它们也可以接收值。基本用法非常简单：

```py
>>> def generator():
...     value = yield 'spam'
...     print('Generator received: %s' % value)
...     yield 'Previous value: %r' % value

>>> g = generator()
>>> print('Result from generator: %s' % next(g))
Result from generator: spam
>>> print(g.send('eggs'))
Generator received: eggs
Previous value: 'eggs'

```

就是这样。在调用`send`方法之前，函数会被冻结，此时它将处理到下一个`yield`语句。

## 启动

由于生成器是惰性的，你不能直接向全新的生成器发送一个值。在值被发送到生成器之前，要么必须使用`next()`获取结果，要么必须发出`send(None)`，以便实际到达代码。这种需求是可以理解的，但有时有点乏味。让我们创建一个简单的装饰器来省略这个需求：

```py
>>> import functools

>>> def coroutine(function):
...     @functools.wraps(function)
...     def _coroutine(*args, **kwargs):
...         active_coroutine = function(*args, **kwargs)
...         next(active_coroutine)
...         return active_coroutine
...     return _coroutine

>>> @coroutine
... def spam():
...     while True:
...         print('Waiting for yield...')
...         value = yield
...         print('spam received: %s' % value)

>>> generator = spam()
Waiting for yield...

>>> generator.send('a')
spam received: a
Waiting for yield...

>>> generator.send('b')
spam received: b
Waiting for yield...

```

你可能已经注意到，即使生成器仍然是惰性的，它现在会自动执行所有代码，直到再次到达`yield`语句。在那时，它将保持休眠状态，直到发送新值。

### 注意

请注意，从现在开始，`coroutine`装饰器将在本章中使用。为简洁起见，我们将在以下示例中省略它。

## 关闭和抛出异常

与普通生成器不同，一旦输入序列耗尽，协程通常采用无限的`while`循环，这意味着它们不会以正常方式被关闭。这就是为什么协程也支持`close`和`throw`方法，它们将退出函数。这里重要的不是关闭，而是添加拆卸方法的可能性。从本质上讲，这与上下文包装器如何使用`__enter__`和`__exit__`方法的方式非常相似，但在这种情况下是协程：

```py
@coroutine
def simple_coroutine():
    print('Setting up the coroutine')
    try:
        while True:
            item = yield
            print('Got item: %r' % item)
    except GeneratorExit:
        print('Normal exit')
    except Exception as e:
        print('Exception exit: %r' % e)
        raise
    finally:
        print('Any exit')

print('Creating simple coroutine')
active_coroutine = simple_coroutine()
print()

print('Sending spam')
active_coroutine.send('spam')
print()

print('Close the coroutine')
active_coroutine.close()
print()

print('Creating simple coroutine')
active_coroutine = simple_coroutine()
print()

print('Sending eggs')
active_coroutine.send('eggs')
print()

print('Throwing runtime error')
active_coroutine.throw(RuntimeError, 'Oops...')
print()
```

这将生成以下输出，应该是预期的——没有奇怪的行为，只是退出协程的两种方法：

```py
# python3 H06.py
Creating simple coroutine
Setting up the coroutine

Sending spam
Got item: 'spam'

Close the coroutine
Normal exit
Any exit

Creating simple coroutine
Setting up the coroutine

Sending eggs
Got item: 'eggs'

Throwing runtime error
Exception exit: RuntimeError('Oops...',)
Any exit
Traceback (most recent call last):
...
 **File ... in <module>
 **active_coroutine.throw(RuntimeError, 'Oops...')
 **File ... in simple_coroutine
 **item = yield
RuntimeError: Oops...

```

## 双向管道

在前面的段落中，我们看到了管道；它们按顺序处理输出并且是单向的。然而，有些情况下这还不够——有时你需要一个不仅将值发送到下一个管道，而且还能从子管道接收信息的管道。我们可以通过这种方式在执行之间保持生成器的状态，而不是始终只有一个单一的列表被处理。因此，让我们首先将之前的管道转换为协程。首先，再次使用`lines.txt`文件：

```py
spam
eggs
spam spam
eggs eggs
spam spam spam
eggs eggs eggs
```

现在，协程管道。这些函数与以前的相同，但使用协程而不是生成器：

```py
>>> @coroutine
... def replace(search, replace):
...     while True:
...         item = yield
...         print(item.replace(search, replace))

>>> spam_replace = replace('spam', 'bacon')
>>> for line in open('lines.txt'):
...     spam_replace.send(line.rstrip())
bacon
eggs
bacon bacon
eggs eggs
bacon bacon bacon
eggs eggs eggs

```

鉴于这个例子，你可能会想知道为什么我们现在打印值而不是产出它。嗯！我们可以，但要记住生成器会冻结，直到产出一个值。让我们看看如果我们只是`yield`值而不是调用`print`会发生什么。默认情况下，你可能会想这样做：

```py
>>> @coroutine
... def replace(search, replace):
...     while True:
...         item = yield
...         yield item.replace(search, replace)

>>> spam_replace = replace('spam', 'bacon')
>>> spam_replace.send('spam')
'bacon'
>>> spam_replace.send('spam spam')
>>> spam_replace.send('spam spam spam')
'bacon bacon bacon'

```

现在一半的值已经消失了，所以问题是，“它们去哪了？”注意第二个`yield`没有存储结果。这就是值消失的地方。我们需要将它们也存储起来：

```py
>>> @coroutine
... def replace(search, replace):
...     item = yield
...     while True:
...         item = yield item.replace(search, replace)

>>> spam_replace = replace('spam', 'bacon')
>>> spam_replace.send('spam')
'bacon'
>>> spam_replace.send('spam spam')
'bacon bacon'
>>> spam_replace.send('spam spam spam')
'bacon bacon bacon'

```

但即使这样还远非最佳。我们现在基本上是在使用协程来模仿生成器的行为。虽然它能工作，但有点傻而且不是很清晰。这次让我们真正建立一个管道，让协程将数据发送到下一个协程（或多个协程），并通过将结果发送到多个协程来展示协程的力量：

```py
# Grep sends all matching items to the target
>>> @coroutine
... def grep(target, pattern):
...     while True:
...         item = yield
...         if pattern in item:
...             target.send(item)

# Replace does a search and replace on the items and sends it to
# the target once it's done
>>> @coroutine
... def replace(target, search, replace):
...     while True:
...         target.send((yield).replace(search, replace))

# Print will print the items using the provided formatstring
>>> @coroutine
... def print_(formatstring):
...     while True:
...         print(formatstring % (yield))

# Tee multiplexes the items to multiple targets
>>> @coroutine
... def tee(*targets):
...     while True:
...         item = yield
...         for target in targets:
...             target.send(item)

# Because we wrap the results we need to work backwards from the
# inner layer to the outer layer.

# First, create a printer for the items:
>>> printer = print_('%s')

# Create replacers that send the output to the printer
>>> replacer_spam = replace(printer, 'spam', 'bacon')
>>> replacer_eggs = replace(printer, 'spam spam', 'sausage')

# Create a tee to send the input to both the spam and the eggs
# replacers
>>> branch = tee(replacer_spam, replacer_eggs)

# Send all items containing spam to the tee command
>>> grepper = grep(branch, 'spam')

# Send the data to the grepper for all the processing
>>> for line in open('lines.txt'):
...     grepper.send(line.rstrip())
bacon
spam
bacon bacon
sausage
bacon bacon bacon
sausage spam

```

这使得代码更简单、更易读，但更重要的是，它展示了如何将单一源拆分为多个目的地。虽然这看起来可能不那么令人兴奋，但它肯定是。如果你仔细观察，你会发现`tee`方法将输入分成两个不同的输出，但这两个输出都写回到同一个`print_`实例。这意味着你可以将数据沿着任何方便的方式路由，而无需任何努力就可以将其最终发送到同一个终点。

尽管如此，这个例子仍然不是那么有用，因为这些函数仍然没有充分利用协程的全部功能。最重要的特性，即一致的状态，在这种情况下并没有真正被使用。

从这些行中学到的最重要的一课是，在大多数情况下混合使用生成器和协程并不是一个好主意，因为如果使用不正确，它可能会产生非常奇怪的副作用。尽管两者都使用`yield`语句，但它们是具有不同行为的显著不同的实体。下一段将展示混合协程和生成器可以有用的为数不多的情况之一。

## 使用状态

既然我们知道如何编写基本的协程以及需要注意的陷阱，那么如何编写一个需要记住状态的函数呢？也就是说，一个始终给出所有发送值的平均值的函数。这是为数不多的情况之一，仍然相对安全和有用地结合协程和生成器语法：

```py
>>> @coroutine
... def average():
...     count = 1
...     total = yield
...     while True:
...         total += yield total / count
...         count += 1

>>> averager = average()
>>> averager.send(20)
20.0
>>> averager.send(10)
15.0
>>> averager.send(15)
15.0
>>> averager.send(-25)
5.0

```

尽管这仍然需要一些额外的逻辑才能正常工作。为了确保我们不会除以零，我们将`count`初始化为`1`。之后，我们使用`yield`获取我们的第一个项目，但在那时我们不发送任何数据，因为第一个`yield`是启动器，并且在我们获得值之前执行。一旦设置好了，我们就可以轻松地在求和的同时产生平均值。并不是太糟糕，但纯协程版本稍微更容易理解，因为我们不必担心启动：

```py
>>> @coroutine
... def print_(formatstring):
...     while True:
...         print(formatstring % (yield))

>>> @coroutine
... def average(target):
...     count = 0
...     total = 0
...     while True:
...         count += 1
...         total += yield
...         target.send(total / count)

>>> printer = print_('%.1f')
>>> averager = average(printer)
>>> averager.send(20)
20.0
>>> averager.send(10)
15.0
>>> averager.send(15)
15.0
>>> averager.send(-25)
5.0

```

就像应该的那样，只需保持计数和总值，然后简单地为每个新值发送新的平均值。

另一个很好的例子是`itertools.groupby`，也很容易用协程实现。为了比较，我们将再次展示生成器协程和纯协程版本：

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
>>> grouper.send(('a', 1))
>>> grouper.send(('a', 2))
>>> grouper.send(('a', 3))
>>> grouper.send(('b', 1))
('a', [1, 2, 3])
>>> grouper.send(('b', 2))
>>> grouper.send(('a', 1))
('b', [1, 2])
>>> grouper.send(('a', 2))
>>> grouper.send((None, None))
('a', [1, 2])

```

正如你所看到的，这个函数使用了一些技巧。我们存储了前一个`key`和`value`，以便我们可以检测到组（`key`）何时发生变化。这就是第二个问题；显然我们只有在组发生变化后才能识别出一个组，因此只有在组发生变化后才会返回结果。这意味着最后一组只有在它之后发送了不同的组之后才会发送，因此是`(None, None)`。现在，这是纯协程版本：

```py
>>> @coroutine
... def print_(formatstring):
...     while True:
...         print(formatstring % (yield))

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

>>> grouper = groupby(print_('group: %s, values: %s'))
>>> grouper.send(('a', 1))
>>> grouper.send(('a', 2))
>>> grouper.send(('a', 3))
>>> grouper.send(('b', 1))
group: a, values: [1, 2, 3]
>>> grouper.send(('b', 2))
>>> grouper.send(('a', 1))
group: b, values: [1, 2]
>>> grouper.send(('a', 2))
>>> grouper.send((None, None))
group: a, values: [1, 2]

```

虽然这些函数非常相似，但纯协程版本再次要简单得多。这是因为我们不必考虑启动和可能丢失的值。

# 总结

本章向我们展示了如何创建生成器以及它们的优势和劣势。此外，现在应该清楚如何解决它们的限制以及这样做的影响。

虽然关于协程的段落应该已经提供了一些关于它们是什么以及如何使用它们的见解，但并非一切都已经展示出来。我们看到了纯协程和同时是生成器的协程的构造，但它们仍然是同步的。协程允许将结果发送给许多其他协程，因此可以有效地同时执行许多函数，但如果某个操作被阻塞，它们仍然可以完全冻结 Python。这就是我们下一章将会帮助解决的问题。

Python 3.5 引入了一些有用的功能，比如`async`和`await`语句。这使得协程可以完全异步和非阻塞，而本章节使用的是自 Python 2.5 以来可用的基本协程功能。

下一章将扩展新功能，包括`asyncio`模块。这个模块使得使用协程进行异步 I/O 到诸如 TCP、UDP、文件和进程等端点变得几乎简单。
