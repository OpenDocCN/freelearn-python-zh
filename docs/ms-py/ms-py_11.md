# 第十一章：调试-解决错误

上一章向您展示了如何向代码添加日志记录和测试，但无论您有多少测试，您总会有 bug。最大的问题始终是用户输入，因为不可能测试所有可能的输入，这意味着在某个时候，我们将需要调试代码。

有许多调试技术，而且很可能您已经使用了其中一些。在本章中，我们将专注于打印/跟踪调试和交互式调试。

使用打印语句、堆栈跟踪和日志记录进行调试是最通用的方法之一，很可能是您使用过的第一种调试方法。即使`print 'Hello world'`也可以被视为这种类型，因为输出将向您显示代码正在正确执行。显然没有必要解释如何以及在何处放置打印语句来调试代码，但使用装饰器和其他 Python 模块有一些很好的技巧，使得这种类型的调试更加有用，比如`faulthandler`。

交互式调试是一种更复杂的调试方法。它允许您在程序运行时调试程序。使用这种方法，甚至可以在应用程序运行时更改变量并在任何所需的地方暂停应用程序。缺点是它需要一些关于调试器命令的知识才能真正有用。

总之，我们将涵盖以下主题：

+   使用`print`、`trace`、`logging`和`faulthandler`进行调试

+   使用`pdb`进行交互式调试

# 非交互式调试

最基本的调试形式是在代码中添加简单的打印语句，以查看仍在工作和不在工作的内容。这在各种情况下都很有用，并且可能有助于解决大部分问题。在本章后面，我们将展示一些交互式调试方法，但这些方法并不总是适用。在多线程环境中，交互式调试往往变得困难甚至不可能，而在封闭的远程服务器上，您可能也需要不同的解决方案。这两种方法都有其优点，但我个人 90%的时间都选择非交互式调试，因为简单的打印/日志语句通常足以分析问题的原因。

这是一个基本示例（我已经知道做类似的事情）使用生成器可以如下所示：

```py
>>> def spam_generator():
...     print('a')
...     yield 'spam'
...     print('b')
...     yield 'spam!'
...     print('c')
...     yield 'SPAM!'
...     print('d')

>>> generator = spam_generator()

>>> next(generator)
a
'spam'

>>> next(generator)
b
'spam!'

```

这清楚地显示了代码的执行情况，因此也清楚地显示了代码未执行的情况。如果没有这个例子，您可能会期望在`spam_generator()`调用之后立即出现第一个打印，因为它是一个生成器。然而，执行完全停滞，直到我们`yield`一个项目。假设在第一个`yield`之前有一些设置代码，它将不会在实际调用`next`之前运行。

虽然这是使用打印语句调试函数的最简单方法之一，但绝对不是最佳方法。我们可以从制作一个自动打印函数开始，该函数会自动递增字母：

```py
>>> import string

>>> def print_character():
...     i = 0
...     while True:
...         print('Letter: %r' % string.ascii_letters[i])
...         i = (i + 1) % len(string.ascii_letters)
...         yield
>>> # Always initialize
>>> print_character = print_character()

>>> next(print_character)
Letter: 'a'
>>> next(print_character)
Letter: 'b'
>>> next(print_character)
Letter: 'c'

```

虽然打印语句生成器比裸打印语句稍好一些，但帮助并不是很大。在运行代码时，看到实际执行了哪些行将更有用。我们可以使用`inspect.currentframe`手动执行此操作，但没有必要进行黑客攻击。Python 为您提供了一些专用工具。

## 使用跟踪检查脚本

简单的打印语句在许多情况下都很有用，因为您几乎可以在几乎每个应用程序中轻松地加入打印语句。无论是远程还是本地，使用线程还是使用多进程，都没有关系。它几乎可以在任何地方工作，使其成为最通用的解决方案，除了日志记录之外。然而，通用解决方案通常不是最佳解决方案。对于最常见的情况，有更好的解决方案可用。其中之一是`trace`模块。它为您提供了一种跟踪每次执行、函数之间关系以及其他一些内容的方法。

为了演示，我们将使用我们之前的代码，但不包括打印语句：

```py
def eggs_generator():
    yield 'eggs'
    yield 'EGGS!'

def spam_generator():
    yield 'spam'
    yield 'spam!'
    yield 'SPAM!'

generator = spam_generator()
print(next(generator))
print(next(generator))

generator = eggs_generator()
print(next(generator))
```

我们将使用 trace 模块执行它：

```py
# python3 -m trace --trace --timing tracing.py
 **--- modulename: tracing, funcname: <module>
0.00 tracing.py(1): def eggs_generator():
0.00 tracing.py(6): def spam_generator():
0.00 tracing.py(11): generator = spam_generator()
0.00 tracing.py(12): print(next(generator))
 **--- modulename: tracing, funcname: spam_generator
0.00 tracing.py(7):     yield 'spam'
spam
0.00 tracing.py(13): print(next(generator))
 **--- modulename: tracing, funcname: spam_generator
0.00 tracing.py(8):     yield 'spam!'
spam!
0.00 tracing.py(15): generator = eggs_generator()
 **--- modulename: tracing, funcname: spam_generator
0.00 tracing.py(16): print(next(generator))
 **--- modulename: tracing, funcname: eggs_generator
0.00 tracing.py(2):     yield 'eggs'
eggs
 **--- modulename: trace, funcname: _unsettrace
0.00 trace.py(77):         sys.settrace(None)

```

相当不错，不是吗？它准确地显示了正在执行的每一行代码以及函数名称，更重要的是，显示了每一行代码是由哪个语句（或多个语句）引起的。此外，它还显示了它相对于程序开始时间的执行时间。这是由于`--timing`标志。

你可能期望，这个输出有点太啰嗦了，不能普遍适用。尽管你可以选择使用命令行参数来忽略特定的模块和目录，但在许多情况下仍然太啰嗦了。所以让我们来尝试下一个解决方案——上下文管理器。前面的输出已经揭示了一些`trace`的内部情况。最后一行显示了一个`sys.settrace`调用，这正是我们需要的手动跟踪：

```py
import- sys
import trace as trace_module
import contextlib

@contextlib.contextmanager
def trace(count=False, trace=True, timing=True):
    tracer = trace_module.Trace(
        count=count, trace=trace, timing=timing)
    sys.settrace(tracer.globaltrace)
    yield tracer
    sys.settrace(None)

    result = tracer.results()
    result.write_results(show_missing=False, summary=True)

def eggs_generator():
    yield 'eggs'
    yield 'EGGS!'

def spam_generator():
    yield 'spam'
    yield 'spam!'
    yield 'SPAM!'

with trace():
    generator = spam_generator()
    print(next(generator))
    print(next(generator))

generator = eggs_generator()
print(next(generator))
```

当作为常规 Python 文件执行时，返回：

```py
# python3 tracing.py
 **--- modulename: tracing, funcname: spam_generator
0.00 tracing.py(24):     yield 'spam'
spam
 **--- modulename: tracing, funcname: spam_generator
0.00 tracing.py(25):     yield 'spam!'
spam!
 **--- modulename: contextlib, funcname: __exit__
0.00 contextlib.py(64):         if type is None:
0.00 contextlib.py(65):             try:
0.00 contextlib.py(66):                 next(self.gen)
 **--- modulename: tracing, funcname: trace
0.00 tracing.py(12):     sys.settrace(None)

```

这段代码立即揭示了跟踪代码的内部操作：它使用`sys.settrace`告诉 Python 解释器在执行每个语句时将其发送到哪里。鉴于此，将函数编写为装饰器显然是微不足道的，但如果你需要的话，我会把它留给你作为一个练习。

从中还可以得到的另一个收获是，你可以通过包装`tracer.globaltrace`轻松地向你的跟踪函数添加额外的过滤器。该函数接受以下参数（来自标准 Python 文档）：

| 参数 | 描述 |
| --- | --- |
| `Call` | 调用函数（或进入某些其他代码块）。调用全局跟踪函数；`arg`是`None`。返回值指定了本地跟踪函数。 |
| `Line` | 解释器即将执行新的一行代码或重新执行循环的条件。调用本地跟踪函数；`arg`是`None`。返回值指定了新的本地跟踪函数。有关其工作原理的详细解释，请参阅`Objects/lnotab_notes.txt`。 |
| `return` | 一个函数（或其他代码块）即将返回。调用本地跟踪函数；`arg`是将要返回的值，或者如果事件是由引发异常引起的，则为`None`。跟踪函数的返回值被忽略。 |
| `exception` | 这意味着发生了异常。调用本地跟踪函数；`arg`是一个元组（`exception`，`value`，`traceback`）。返回值指定了新的本地跟踪函数。 |
| `c_call` | 即将调用一个 C 函数。这可能是一个扩展函数或内置函数。`arg`是 C 函数对象。 |
| `c_return` | 一个 C 函数已经返回，`arg`是 C 函数对象。 |
| `c_exception` | 一个 C 函数引发了异常，`arg`是 C 函数对象。 |

正如你所期望的那样，通过一个简单的过滤函数，你可以轻松地确保只返回特定的函数，而不是通常会得到的长列表。你真的不应该低估使用几个导入来跟踪代码生成的数据量。前面的上下文管理器代码产生了 300 多行输出。

## 使用日志进行调试

在第十章中，*测试和日志 - 为错误做准备*，我们看到了如何创建自定义记录器，为它们设置级别，并为特定级别添加处理程序。我们将使用`logging.DEBUG`级别进行日志记录，这本身并不特别，但通过一些装饰器，我们可以添加一些非常有用的仅用于调试的代码。

每当我调试时，我总是发现了解函数的输入和输出非常有用。使用装饰器的基本版本足够简单；只需打印`args`和`kwargs`，就完成了。以下示例稍微深入一些。通过使用`inspect`模块，我们还可以检索默认参数，从而可以在所有情况下显示所有参数及其参数名和值，即使未指定参数也可以。

```py
import pprint
import inspect
import logging
import functools

logging.basicConfig(level=logging.DEBUG)

def debug(function):
    @functools.wraps(function)
    def _debug(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
        finally:
            # Extract the signature from the function
            signature = inspect.signature(function)
            # Fill the arguments
            arguments = signature.bind(*args, **kwargs)
            # NOTE: This only works for Python 3.5 and up!
            arguments.apply_defaults()

            logging.debug('%s(%s): %s' % (
                function.__qualname__,
                ', '.join('%s=%r' % (k, v) for k, v in
                          arguments.arguments.items()),
                pprint.pformat(result),
            ))

    return _debug

@debug
def spam(a, b=123):
    return 'some spam'

spam(1)
spam(1, 456)
spam(b=1, a=456)
```

返回以下输出：

```py
# python3 logged.py
DEBUG:root:spam(a=1, b=123): 'some spam'
DEBUG:root:spam(a=1, b=456): 'some spam'
DEBUG:root:spam(a=456, b=1): 'some spam'

```

当然非常好，因为我们清楚地知道函数何时被调用，使用了哪些参数，以及返回了什么。但是，这可能只有在您积极调试代码时才会执行。您还可以通过添加特定于调试的记录器使代码中的常规`logging.debug`语句更加有用，该记录器显示更多信息。只需用前面示例的日志配置替换此示例：

```py
import logging

log_format = (
    '[%(relativeCreated)d %(levelname)s] '
    '%(pathname)s:%(lineno)d:%(funcName)s: %(message)s'
)
logging.basicConfig(level=logging.DEBUG, format=log_format)
```

那么你的结果会是这样的：

```py
# time python3 logged.py
[0 DEBUG] logged.py:31:_debug: spam(a=1, b=123): 'some spam'
[0 DEBUG] logged.py:31:_debug: spam(a=1, b=456): 'some spam'
[0 DEBUG] logged.py:31:_debug: spam(a=456, b=1): 'some spam'
python3 logged.py  0.04s user 0.01s system 96% cpu 0.048 total

```

它显示相对于应用程序启动的时间（毫秒）和日志级别。然后是一个标识块，显示产生日志的文件名、行号和函数名。当然，最后还有一条消息。

## 显示无异常的调用堆栈

在查看代码的运行方式和原因时，通常有必要查看整个堆栈跟踪。当然，简单地引发异常是一个选择。但是，那将终止当前的代码执行，这通常不是我们要寻找的。这就是`traceback`模块派上用场的地方。只需几行简单的代码，我们就可以得到完整的（或有限的，如果您愿意的话）堆栈列表：

```py
import traceback

class Spam(object):

    def run(self):
        print('Before stack print')
        traceback.print_stack()
        print('After stack print')

class Eggs(Spam):
    pass

if __name__ == '__main__':
    eggs = Eggs()
    eggs.run()
```

这导致以下结果：

```py
# python3 traceback_test.py
Before stack print
 **File "traceback_test.py", line 18, in <module>
 **eggs.run()
 **File "traceback_test.py", line 8, in run
 **traceback.print_stack()
After stack print

```

如您所见，回溯只是简单地打印而没有任何异常。`traceback`模块实际上有很多其他方法，用于基于异常等打印回溯，但您可能不经常需要它们。最有用的可能是`limit`参数；此参数允许您将堆栈跟踪限制为有用的部分。例如，如果您使用装饰器或辅助函数添加了此代码，则可能不需要在堆栈跟踪中包含它们。这就是`limit`参数的作用所在：

```py
import traceback

class Spam(object):

    def run(self):
        print('Before stack print')
        traceback.print_stack(limit=-1)
        print('After stack print')

class Eggs(Spam):
    pass

if __name__ == '__main__':
    eggs = Eggs()
    eggs.run()
```

这导致以下结果：

```py
# python3 traceback_test.py
Before stack print
 **File "traceback_test.py", line 18, in <module>
 **eggs.run()
After stack print

```

如您所见，`print_stack`函数本身现在已从堆栈跟踪中隐藏，这使得一切都变得更加清晰。

### 注意

在 Python 3.5 中添加了负限制支持。在那之前，只支持正限制。

## 调试 asyncio

`asyncio`模块有一些特殊规定，使得调试变得更容易一些。鉴于`asyncio`内部函数的异步特性，这是一个非常受欢迎的功能。在调试多线程/多进程函数或类时可能会很困难——因为并发类可以轻松并行更改环境变量——而使用`asyncio`，情况可能会更加困难。

### 注意

在大多数 Linux/Unix/Mac shell 会话中，可以使用它作为前缀设置环境变量：

```py
SOME_ENVIRONMENT_VARIABLE=value python3 script.py

```

此外，可以使用`export`为当前 shell 会话进行配置：

```py
export SOME_ENVIRONMENT_VARIABLE=value

```

可以使用以下行来获取当前值：

```py
echo $SOME_ENVIRONMENT_VARIABLE

```

在 Windows 上，可以使用`set`命令为本地 shell 会话配置环境变量：

```py
set SOME_ENVIRONMENT_VARIABLE=value

```

可以使用以下行来获取当前值：

```py
set SOME_ENVIRONMENT_VARIABLE

```

使用`PYTHONASYNCIODEBUG`环境设置启用调试模式时，`asyncio`模块将检查每个定义的协程是否实际运行：

```py
import asyncio

@asyncio.coroutine
def printer():
    print('This is a coroutine')

printer()
```

这导致打印器协程出现错误，这里从未产生过：

```py
# PYTHONASYNCIODEBUG=1 python3 asyncio_test.py
<CoroWrapper printer() running, defined at asyncio_test.py:4, created at asyncio_test.py:8> was never yielded from
Coroutine object created at (most recent call last):
 **File "asyncio_test.py", line 8, in <module>
 **printer()

```

另外，`event`循环默认会有一些日志消息：

```py
import asyncio
import logging

logging.basicConfig(level=logging.DEBUG)
loop = asyncio.get_event_loop()
```

这导致以下调试消息：

```py
# PYTHONASYNCIODEBUG=1 python3 asyncio_test.py
DEBUG:asyncio:Using selector: KqueueSelector
DEBUG:asyncio:Close <_UnixSelectorEventLoop running=False closed=False debug=True>

```

你可能会想为什么我们使用`PYTHONASYNCIODEBUG`标志而不是`loop.set_debug(True)`。原因是有些情况下这种方法不起作用，因为调试启用得太晚。例如，当尝试在前面的`printer()`中使用`loop.set_debug(True)`时，你会发现单独使用`loop.set_debug(True)`时不会出现任何错误。

启用调试后，以下内容将发生变化：

+   未被 yield 的协程（如前面的行所示）将引发异常。

+   从“错误”的线程调用协程会引发异常。

+   选择器的执行时间将被记录。

+   慢回调（超过 100 毫秒）将被记录。可以通过`loop.slow_callback_duration`修改此超时时间。

+   当资源未正确关闭时，将引发警告。

+   在执行之前被销毁的任务将被记录。

## 使用 faulthandler 处理崩溃

`faulthandler`模块在调试真正低级的崩溃时很有帮助，也就是说，只有在使用对内存的低级访问时才可能发生的崩溃，比如 C 扩展。

例如，这里有一小段代码，会导致你的 Python 解释器崩溃：

```py
import ctypes

# Get memory address 0, your kernel shouldn't allow this:
ctypes.string_at(0)
```

它会产生类似以下的结果：

```py
# python faulthandler_test.py
zsh: segmentation fault  python faulthandler_test.py

```

当然，这是一个相当丑陋的响应，而且没有处理错误的可能性。以防你想知道，使用`try/except`结构在这些情况下也无济于事。以下代码将以完全相同的方式崩溃：

```py
import ctypes

try:
    # Get memory address 0, your kernel shouldn't allow this:
    ctypes.string_at(0)
except Exception as e:
    print('Got exception:', e)
```

这就是`faulthandler`模块的作用。它仍然会导致解释器崩溃，但至少你会看到一个正确的错误消息，所以如果你（或任何子库）与原始内存有任何交互，这是一个很好的默认选择：

```py
import ctypes
import faulthandler

faulthandler.enable()

# Get memory address 0, your kernel shouldn't allow this:
ctypes.string_at(0)
```

它会产生类似以下的结果：

```py
# python faulthandler_test.py
Fatal Python error: Segmentation fault

Current thread 0x00007fff79171300 (most recent call first):
 **File "ctypes/__init__.py", line 491 in string_at
 **File "faulthandler_test.py", line 7 in <module>
zsh: segmentation fault  python faulthandler_test.py

```

显然，以这种方式退出 Python 应用程序是不可取的，因为代码不会以正常的清理退出。资源不会被干净地关闭，退出处理程序也不会被调用。如果你以某种方式需要捕获这种行为，最好的办法是将 Python 可执行文件包装在一个单独的脚本中。

# 交互式调试

现在我们已经讨论了一些基本的调试方法，这些方法总是有效的，我们将看一些更高级的调试技术。之前的调试方法通过修改代码和/或预见使变量和堆栈可见。这一次，我们将看一种稍微更智能的方法，即在需要时以交互方式执行相同的操作。

## 按需控制台

在测试一些 Python 代码时，你可能已经使用过交互式控制台几次，因为它是测试 Python 代码的一个简单而有效的工具。你可能不知道的是，从你的代码中启动自己的 shell 实际上是很简单的。因此，每当你想从代码的特定点进入常规 shell 时，这是很容易实现的：

```py
import code

def spam():
    eggs = 123
    print('The begin of spam')
    code.interact(banner='', local=locals())
    print('The end of spam')
    print('The value of eggs: %s' % eggs)

if __name__ == '__main__':
    spam()
```

在执行时，我们将在交互式控制台中间停下来：

```py
# python3 test_code.py
The begin of spam
>>> eggs
123
>>> eggs = 456
>>>
The end of spam
The value of eggs: 123

```

要退出这个控制台，我们可以在 Linux/Mac 系统上使用*^d*（*Ctrl* + *d*），在 Windows 系统上使用*^z*（*Ctrl* + *Z*）。

这里需要注意的一件重要的事情是，这两者之间的范围是不共享的。尽管我们传递了`locals()`以便共享本地变量以方便使用，但这种关系并不是双向的。结果是，即使我们在交互会话中将`eggs`设置为`456`，它也不会传递到外部函数。如果你愿意，你可以通过直接操作（例如设置属性）来修改外部范围的变量，但所有在本地声明的变量都将保持本地。

## 使用 pdb 进行调试

在实际调试代码时，常规的交互式控制台并不适用。通过一些努力，你可以让它工作，但它并不方便调试，因为你只能看到当前的范围，不能轻松地在堆栈中跳转。使用`pdb`（Python 调试器）可以轻松实现这一点。让我们看一个使用`pdb`的简单例子：

```py
import pdb

def spam():
    eggs = 123
    print('The begin of spam')
    pdb.set_trace()
    print('The end of spam')
    print('The value of eggs: %s' % eggs)

if __name__ == '__main__':
    spam()
```

这个例子与前一段中的例子几乎完全相同，只是这一次我们最终进入了`pdb`控制台，而不是常规的交互式控制台。所以让我们试试交互式调试器：

```py
# python3 test_pdb.py
The begin of spam
> test_pdb.py(8)spam()
-> print('The end of spam')
(Pdb) eggs
123
(Pdb) eggs = 456
(Pdb) continue
The end of spam
The value of eggs: 456

```

正如你所看到的，我们现在实际上修改了`eggs`的值。在这种情况下，我们使用了完整的`continue`命令，但所有`pdb`命令也有简写版本。因此，使用`c`而不是`continue`会得到相同的结果。只需输入`eggs`（或任何其他变量）将显示内容，并且设置变量将简单地设置它，就像我们从交互式会话中期望的那样。

要开始使用`pdb`，首先显示了最有用的（完整）命令列表及其简写：

| Command | Explanation |
| --- | --- |
| `h(elp)` | 显示命令列表（本列表）。 |
| `h(elp) command` | 显示给定命令的帮助信息。 |
| `w(here)` | 当前堆栈跟踪，箭头指向当前帧。 |
| `d(own)` | 移动到堆栈中的下一个帧。 |
| `u(p)` | 移动到堆栈中的较旧帧。 |
| `s(tep)` | 执行当前行并尽快停止。 |
| `n(ext)` | 执行当前行并停在当前函数内的下一行。 |
| `r(eturn)` | 继续执行，直到函数返回。 |
| `c(ont(inue))` | 继续执行直到下一个断点。 |
| `l(ist) [first[, last]]` | 列出（默认情况下，11 行）当前行周围的源代码行。 |
| `ll &#124; longlist` | 列出当前函数或帧的所有源代码。 |
| `source expression` | 列出给定对象的源代码。这类似于 longlist。 |
| `a(rgs)` | 打印当前函数的参数。 |
| `pp expression` | 漂亮地打印给定的表达式。 |
| `b(reak)` | 显示断点列表。 |
| `b(reak) [filename:]lineno` | 在给定的行号和（可选）文件处设置断点。 |
| `b(reak) function[, condition]` | 在给定的函数处设置断点。条件是一个必须评估为`True`的表达式，断点才能起作用。 |
| `cl(ear) [filename:]lineno` | 清除这一行的断点（或断点）。 |
| `cl(ear) breakpoint [breakpoint ...]` | 清除这些编号的断点（或断点）。 |
| `Command` | 列出所有定义的命令。 |
| `command breakpoint` | 指定在遇到给定断点时执行的命令列表。使用`end`命令结束列表。 |
| `Alias` | 列出所有别名。 |

| `alias name command` | 创建一个别名。命令可以是任何有效的 Python 表达式，所以你可以这样做来打印对象的所有属性：

```py
alias pd pp %1.__dict__** 

```

|

| `unalias name` | 删除别名。 |
| --- | --- |
| `! statement` | 在堆栈的当前位置执行语句。通常情况下不需要`!`符号，但如果与调试器命令发生冲突，这可能会有用。例如，尝试`b = 123`。 |
| `Interact` | 打开一个类似于前一段的交互式会话。请注意，设置在该局部范围内的变量不会被传递。 |

### 断点

这是一个相当长的列表，但你可能会经常使用其中的大部分。为了突出显示前表中显示的选项之一，让我们演示断点的设置和使用：

```py
import pdb

def spam():
    print('The begin of spam')
    print('The end of spam')

if __name__ == '__main__':
    pdb.set_trace()
    spam()
```

到目前为止，没有发生什么新的事情，但现在让我们打开交互式调试会话，如下所示：

```py
# python3 test_pdb.py
> test_pdb.py(11)<module>()
-> while True:
(Pdb) source spam  # View the source of spam
 **4     def spam():
 **5         print('The begin of spam')
 **6         print('The end of spam')

(Pdb) b 5  # Add a breakpoint to line 5
Breakpoint 1 at test_pdb.py:5

(Pdb) w  # Where shows the current line
> test_pdb.py(11)<module>()
-> while True:

(Pdb) c  # Continue (until the next breakpoint or exception)
> test_pdb.py(5)spam()
-> print('The begin of spam')

(Pdb) w  # Where again
 **test_pdb.py(12)<module>()
-> spam()
> test_pdb.py(5)spam()
-> print('The begin of spam')

(Pdb) ll  # List the lines of the current function
 **4     def spam():
 **5 B->     print('The begin of spam')
 **6         print('The end of spam')

(Pdb) b  # Show the breakpoints
Num Type         Disp Enb   Where
1   breakpoint   keep yes   at test_pdb.py:5
 **breakpoint already hit 1 time

(Pdb) cl 1  # Clear breakpoint 1
Deleted breakpoint 1 at test_pdb.py:5

```

输出很多，但实际上并不像看起来那么复杂：

1.  首先，我们使用`source spam`命令查看`spam`函数的源代码。

1.  在那之后，我们知道了第一个`print`语句的行号，我们用它在第 5 行放置了一个断点(`b 5`)。

1.  为了检查我们是否仍然在正确的位置，我们使用了`w`命令。

1.  由于断点已设置，我们使用`c`继续到下一个断点。

1.  在第 5 行的断点停下后，我们再次使用`w`来确认。

1.  使用`ll`列出当前函数的代码。

1.  使用`b`列出断点。

1.  再次使用`cl 1`移除断点，断点号来自于前一个命令。

一开始似乎有点复杂，但你会发现，一旦你尝试了几次，它实际上是一种非常方便的调试方式。

为了使它更好用，这次我们将只在`eggs = 3`时执行断点。代码基本上是一样的，尽管在这种情况下我们需要一个变量：

```py
import pdb

def spam(eggs):
    print('eggs:', eggs)

if __name__ == '__main__':
    pdb.set_trace()
    for i in range(5):
        spam(i)
```

现在，让我们执行代码，并确保它只在特定时间中断：

```py
# python3 test_breakpoint.py
> test_breakpoint.py(10)<module>()
-> for i in range(5):
(Pdb) source spam
 **4     def spam(eggs):
 **5         print('eggs:', eggs)
(Pdb) b 5, eggs == 3  # Add a breakpoint to line 5 whenever eggs=3
Breakpoint 1 at test_breakpoint.py:5
(Pdb) c  # Continue
eggs: 0
eggs: 1
eggs: 2
> test_breakpoint.py(5)spam()
-> print('eggs:', eggs)
(Pdb) a  # Show function arguments
eggs = 3
(Pdb) c  # Continue
eggs: 3
eggs: 4

```

总结我们所做的：

1.  首先，使用`source` spam，我们查找了行号。

1.  之后，我们使用`eggs == 3`条件放置了一个断点。

1.  然后我们使用`c`继续执行。如你所见，值`0`、`1`和`2`都正常打印出来了。

1.  断点在值`3`处被触发。为了验证这一点，我们使用`a`来查看函数参数。

1.  然后我们继续执行剩下的代码。

### 捕获异常

所有这些都是手动调用`pdb.set_trace()`函数，但一般情况下，你只是运行你的应用程序，并不真的期望出现问题。这就是异常捕获非常有用的地方。除了自己导入`pdb`，你也可以将脚本作为模块通过`pdb`运行。让我们来看看这段代码，一旦它遇到零除法就会中断：

```py
print('This still works')
1/0
print('We shouldnt reach this code')
```

如果我们使用`pdb`参数运行它，每当它崩溃时我们就会进入 Python 调试器：

```py
# python3 -m pdb test_zero.py
> test_zero.py(1)<module>()
-> print('This still works')
(Pdb) w  # Where
 **bdb.py(431)run()
-> exec(cmd, globals, locals)
 **<string>(1)<module>()
> test_zero.py(1)<module>()
-> print('This still works')
(Pdb) s  # Step into the next statement
This still works
> test_zero.py(2)<module>()
-> 1/0
(Pdb) c  # Continue
Traceback (most recent call last):
 **File "pdb.py", line 1661, in main
 **pdb._runscript(mainpyfile)
 **File "pdb.py", line 1542, in _runscript
 **self.run(statement)
 **File "bdb.py", line 431, in run
 **exec(cmd, globals, locals)
 **File "<string>", line 1, in <module>
 **File "test_zero.py", line 2, in <module>
 **1/0
ZeroDivisionError: division by zero
Uncaught exception. Entering post mortem debugging
Running 'cont' or 'step' will restart the program
> test_zero.py(2)<module>()
-> 1/0

```

### 提示

`pdb`中一个有用的小技巧是使用*Enter*按钮，默认情况下，它会再次执行先前执行的命令。当逐步执行程序时，这非常有用。

### 命令

`commands`命令有点复杂，但非常有用。它允许你在遇到特定断点时执行命令。为了说明这一点，让我们再从一个简单的例子开始：

```py
import pdb

def spam(eggs):
    print('eggs:', eggs)

if __name__ == '__main__':
    pdb.set_trace()
    for i in range(5):
        spam(i)
```

代码足够简单，所以现在我们将添加断点和命令，如下所示：

```py
# python3 test_breakpoint.py
> test_breakpoint.py(10)<module>()
-> for i in range(3):
(Pdb) b spam  # Add a breakpoint to function spam
Breakpoint 1 at test_breakpoint.py:4
(Pdb) commands 1  # Add a command to breakpoint 1
(com) print('The value of eggs: %s' % eggs)
(com) end  # End the entering of the commands
(Pdb) c  # Continue
The value of eggs: 0
> test_breakpoint.py(5)spam()
-> print('eggs:', eggs)
(Pdb) c  # Continue
eggs: 0
The value of eggs: 1
> test_breakpoint.py(5)spam()
-> print('eggs:', eggs)
(Pdb) cl 1  # Clear breakpoint 1
Deleted breakpoint 1 at test_breakpoint.py:4
(Pdb) c  # Continue
eggs: 1
eggs: 2

```

正如你所看到的，我们可以很容易地向断点添加命令。在移除断点后，这些命令显然不会再被执行。

## 使用 ipdb 进行调试

通用的 Python 控制台虽然有用，但有时会有点粗糙。IPython 控制台提供了许多额外功能，使其成为一个更好用的控制台。其中一个功能是更方便的调试器。

首先确保你已经安装了`ipdb`：

```py
pip install ipdb

```

接下来，让我们再次尝试使用我们之前的脚本进行调试。唯一的小改变是，我们现在导入的是`ipdb`而不是`pdb`：

```py
import ipdb

def spam(eggs):
    print('eggs:', eggs)

if __name__ == '__main__':
    ipdb.set_trace()
    for i in range(3):
        spam(i)
```

然后我们执行它：

```py
# python3 test_ipdb.py
> test_ipdb.py(10)<module>()
 **9     ipdb.set_trace()
---> 10     for i in range(3):
 **11         spam(i)

ipdb> b spam  # Set a breakpoint
Breakpoint 1 at test_ipdb.py:4
ipdb> c  # Continue (until exception or breakpoint)
> test_ipdb.py(5)spam()
1     4 def spam(eggs):
----> 5     print('eggs:', eggs)
 **6

ipdb> a  # Show the arguments
eggs = 0
ipdb> c  # Continue
eggs: 0
> test_ipdb.py(5)spam()
1     4 def spam(eggs):
----> 5     print('eggs:', eggs)
 **6

ipdb>   # Repeat the previous command, so continue again
eggs: 1
> test_ipdb.py(5)spam()
1     4 def spam(eggs):
----> 5     print('eggs:', eggs)
 **6

ipdb> cl 1  # Remove breakpoint 1
Deleted breakpoint 1 at test_ipdb.py:4
ipdb> c  # Continue
eggs: 2

```

命令都是一样的，但在我看来输出更易读一些。实际版本还包括语法高亮，使输出更容易跟踪。

简而言之，在大多数情况下，你可以简单地用`ipdb`替换`pdb`来获得一个更直观的调试器。但我也会给你推荐`ipdb`上下文管理器：

```py
import ipdb

with ipdb.launch_ipdb_on_exception():
    main()
```

这就像看起来的那样方便。它只是将`ipdb`连接到你的异常中，这样你可以在需要时轻松调试。将其与应用程序的调试标志结合使用，可以轻松地在需要时允许调试。

## 其他调试器

`pdb`和`ipdb`只是众多可用于 Python 的调试器中的两个。目前一些值得注意的调试器如下：

+   `pudb`：这提供了一个全屏命令行调试器

+   `pdbpp`：这是对常规`pdb`的一个钩子

+   `rpdb2`：这是一个允许连接到运行中（远程）应用程序的远程调试器

+   `Werkzeug`：这是一个基于 Web 的调试器，允许在运行时调试 Web 应用程序

当然还有许多其他调试器，并没有一个绝对最好的。就像所有工具一样，它们都有各自的优势和缺陷，而最适合你当前目的的工具只有你自己才能决定。很可能你当前使用的 Python IDE 已经集成了调试器。

## 调试服务

除了在遇到问题时进行调试之外，有时您只需要跟踪错误以供以后调试。特别是在与远程服务器一起工作时，这些可以非常宝贵，可以检测 Python 进程何时以及如何发生故障。此外，这些服务还提供错误分组，使它们比简单的异常类型脚本更有用，后者可能会快速填满您的收件箱。

一个很好的开源解决方案，用于跟踪错误的是`sentry`。如果您需要一个提供性能跟踪的完整解决方案，那么 Opbeat 和 Newrelic 都是非常好的解决方案；它们提供免费和付费版本。请注意，所有这些解决方案还支持跟踪其他语言，例如 JavaScript。

# 总结

本章介绍了一些不同的调试技术和陷阱。当然，关于调试还有很多可以说的，但我希望您现在已经获得了一个很好的调试 Python 代码的视角。交互式调试技术对于单线程应用程序和可用交互式会话的位置非常有用。但由于情况并非总是如此，我们还讨论了一些非交互式选项。

以下是本章讨论的所有要点概述：

+   使用非交互式调试：

+   打印

+   记录

+   跟踪

+   回溯

+   `asyncio`

+   故障处理程序

+   使用`pdb`和`ipdb`进行交互式调试

在下一章中，我们将看到如何监视和改善 CPU 和内存性能，以及查找和修复内存泄漏。
