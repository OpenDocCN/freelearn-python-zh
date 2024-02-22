# 第十四章：开发工具

在本章中，我们将介绍以下内容：

+   调试-如何利用 Python 内置调试器

+   测试-使用 Python 标准库测试框架编写测试套件

+   模拟-在测试中修补对象以模拟虚假行为

+   在生产中报告错误-通过电子邮件报告崩溃

+   基准测试-如何使用标准库对函数进行基准测试

+   检查-检查对象提供的类型、属性和方法

+   代码评估-在 Python 代码中运行 Python 代码

+   跟踪-如何跟踪执行了哪些代码行

+   性能分析-如何跟踪代码中的瓶颈

# 介绍

在编写软件时，您需要工具来更轻松地实现目标，以及帮助您管理代码库的复杂性，代码库可能包含数百万行代码，并且可能涉及您不熟悉的其他人的代码。

即使是对于小型项目，如果涉及第三方库、框架和工具，实际上是将其他人的代码引入到自己的代码中，您将需要一套工具来理解在依赖于此代码时发生了什么，并且保持自己的代码受控并且没有错误。

在这里，诸如测试、调试、性能分析和跟踪等技术可以派上用场，以验证代码库，了解发生了什么，发现瓶颈，并查看执行了什么以及何时执行。

Python 标准库提供了许多您在日常开发中需要实现大多数最佳实践和软件开发技术的工具。

# 调试

在开发过程中，您可能会遇到代码的意外行为或崩溃，并且希望深入了解，查看变量的状态，并检查发生了什么，以了解如何处理意外情况，以便软件能够正常运行。

这通常是调试的一部分，通常需要专用工具、调试器，以使您的生活更轻松（是否曾经发现自己在代码中到处添加`print`语句，只是为了查看某个变量的值？）。

Python 标准库提供了一个非常强大的调试器，虽然存在其他第三方解决方案，但内部的`pdb`调试器非常强大，并且能够在几乎所有情况下帮助您。

# 如何做...

如果您想在特定点停止代码执行，并在交互式地向前移动，同时检查变量如何变化以及执行的流程，您只需设置一个跟踪点，然后您将进入一个交互式会话，在那里您的代码正在运行：

```py
def divide(x, y):
    print('Going to divide {} / {}'.format(x, y))

    # Stop execution here and enter the debugger
    import pdb; pdb.set_trace()

    return x / y
```

现在，如果我们调用`divide`函数，我们将进入一个交互式调试器，让我们看到`x`和`y`的值，并继续执行：

```py
>>> print(divide(3, 2))
Going to divide 3 / 2
> ../sources/devtools/devtools_01.py(4)divide()
-> return x / y
(Pdb) x
3
(Pdb) y
2
(Pdb) continue
1.5
```

# 它是如何工作的...

`pdb`模块公开了一个`set_trace`函数，当调用时，会停止执行并进入交互式调试器。

从这里开始，您的提示将更改（为`Pdb`），您可以向调试器发送命令，或者只需写出变量名称即可打印变量值。

`pdb`调试器有许多命令；最有用的命令如下：

+   `next`：逐行执行代码

+   `continue`：继续执行代码，直到达到下一个断点

+   `list`：打印当前正在执行的代码

要查看完整的命令列表，您可以使用`help`命令，它将列出所有可用的命令。您还可以使用`help`命令获取有关特定命令的帮助。

# 还有更多...

自 Python 3.7 版本以来，不再需要进行奇怪的`import pdb`；`pdb.set_trace()`操作。您只需编写`breakpoint()`，就会进入`pdb`。

更好的是，如果您的系统配置了更高级的调试器，您将依赖于这些调试器，因为`breakpoint()`使用当前配置的调试器，而不仅仅依赖于`pdb`。

# 测试

为了确保您的代码正确，并且不会在将来的更改中出现问题，编写测试通常是您可以做的最好的事情之一。

在 Python 中，有一些框架可以实现自动验证代码可靠性的测试套件，实现不同的模式，比如**行为驱动开发**（**BDD**），甚至可以自动为您找到边界情况。

但是，只需依赖标准库本身就可以编写简单的自动测试，因此只有在需要特定插件或模式时才需要第三方测试框架。

标准库有`unittest`模块，它允许我们为我们的软件编写测试，运行它们，并报告测试套件的状态。

# 如何做...

对于这个配方，需要执行以下步骤：

1.  假设我们有一个`divide`函数，我们想为它编写测试：

```py
def divide(x, y):
    return x / y
```

1.  我们需要创建一个名为`test_divide.py`的文件（包含测试的文件必须命名为`test_*.py`，否则测试将无法运行）。在`test_divide.py`文件中，我们可以放置所有的测试：

```py
from divide import divide
import unittest

class TestDivision(unittest.TestCase):
    def setUp(self):
        self.num = 6

    def test_int_division(self):
        res = divide(self.num, 3)
        self.assertEqual(res, 2)

    def test_float_division(self):
        res = divide(self.num, 4)
        self.assertEqual(res, 1.5)

    def test_divide_zero(self):
        with self.assertRaises(ZeroDivisionError) as err:
            res = divide(self.num, 0)
        self.assertEqual(str(err.exception), 'division by zero')
```

1.  然后，假设`test_divide.py`模块在同一个目录中，我们可以用`python -m unittest`来运行我们的测试：

```py
$ python -m unittest
...
------------------------------------------------------------------
Ran 3 tests in 0.000s

OK
```

1.  如果我们还想看到哪些测试正在运行，我们也可以提供`-v`选项：

```py
$ python -m unittest -v
test_divide_zero (test_devtools_02.TestDivision) ... ok
test_float_division (test_devtools_02.TestDivision) ... ok
test_int_division (test_devtools_02.TestDivision) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.000s

OK
```

# 它是如何工作的...

`unittest`模块提供了两个主要功能：

+   `unittest.TestCase`类提供了编写测试和固定的基础

+   `unittest.TestLoader`类提供了从多个来源找到并运行多个测试的基础，一次运行；然后可以将结果提供给运行器来运行它们所有并报告它们的进度。

通过创建一个`unittest.TestCase`类，我们可以在相同的固定集下收集多个测试，这些固定集由类作为`setUp`和`setUpClass`方法提供。`setUpClass`方法对整个类执行一次，而`setUp`方法对每个测试执行一次。测试是所有名称以`test*`开头的类方法。

一旦测试完成，`tearDown`和`tearDownClass`方法可以用来清理状态。

因此，我们的`TestDivision`类将为其中声明的每个测试提供一个`self.num`属性：

```py
class TestDivision(unittest.TestCase):
    def setUp(self):
        self.num = 6
```

然后将有三个测试，其中两个（`test_int_division`和`test_float_division`）断言除法的结果是预期的（通过`self.assertEqual`）：

```py
def test_int_division(self):
    res = divide(self.num, 3)
    self.assertEqual(res, 2)

def test_float_division(self):
    res = divide(self.num, 4)
    self.assertEqual(res, 1.5)
```

然后，第三个测试（`test_divide_zero`）检查我们的`divide`函数在提供`0`作为除数时是否实际引发了预期的异常：

```py
def test_divide_zero(self):
    with self.assertRaises(ZeroDivisionError) as err:
        res = divide(self.num, 0)
    self.assertEqual(str(err.exception), 'division by zero')
```

然后检查异常消息是否也是预期的。

然后将这些测试保存在一个名为`test_divide.py`的文件中，以便`TestLoader`能够找到它们。

当执行`python -m unittest`时，实际发生的是调用了`TestLoader.discover`。这将查找本地目录中命名为`test*`的所有模块和包，并运行这些模块中声明的所有测试。

# 还有更多...

标准库`unittest`模块几乎提供了您为库或应用程序编写测试所需的一切。

但是，如果您发现需要更多功能，比如重试不稳定的测试、以更多格式报告和支持驱动浏览器，您可能想尝试像`pytest`这样的测试框架。这些通常提供了一个插件基础架构，允许您通过附加功能扩展它们的行为。

# Mocking

在测试代码时，您可能会面临替换现有函数或类的行为并跟踪函数是否被调用以及是否使用了正确的参数的需求。

例如，假设你有一个如下的函数：

```py
def print_division(x, y):
    print(x / y)
```

为了测试它，我们不想去屏幕上检查输出，但我们仍然想知道打印的值是否是预期的。

因此，一个可能的方法是用不打印任何东西的东西来替换`print`，但允许我们跟踪提供的参数（这是将要打印的值）。

这正是 mocking 的意思：用一个什么都不做但允许我们检查调用的对象或函数替换代码库中的对象或函数。

# 它是如何工作的...

您需要执行以下步骤来完成此操作：

1.  `unittest`包提供了一个`mock`模块，允许我们创建`Mock`对象和`patch`现有对象，因此我们可以依赖它来替换`print`的行为：

```py
from unittest import mock

with mock.patch('builtins.print') as mprint:
    print_division(4, 2)

mprint.assert_called_with(2)
```

1.  一旦我们知道模拟的`print`实际上是用`2`调用的，这是我们预期的值，我们甚至可以进一步打印它接收到的所有参数：

```py
mock_args, mock_kwargs = mprint.call_args
>>> print(mock_args)
(2, )
```

在这种情况下，这并不是很有帮助，因为只有一个参数，但在只想检查部分参数而不是整个调用的情况下，能够访问其中一些参数可能会很方便。

# 工作原理...

`mock.patch`在上下文中用`Mock`实例替换指定的对象或类。

`Mock`在被调用时不会执行任何操作，但会跟踪它们的参数，并允许您检查它们是否按预期被调用。

因此，通过`mock.patch`，我们用`Mock`替换`print`，并将`Mock`的引用保留为`mprint`：

```py
with mock.patch('builtins.print') as mprint:
    print_division(4, 2)
```

这使我们能够检查`print`是否通过`Mock`以预期的参数被调用：

```py
mprint.assert_called_with(2)
```

# 还有更多...

`Mock`对象实际上并不受限于什么都不做。

通过为`mock.patch`提供`side_effect`参数，您可以在调用时引发异常。这对于模拟代码中的故障非常有帮助。

或者，您甚至可以通过为`mock.patch`提供`new`来将它们的行为替换为完全不同的对象，这对于在实现的位置注入伪造对象非常有用。

因此，通常情况下，`unittest.mock`可以用来替换现有类和对象的行为，从模拟对象到伪造对象，再到不同的实现，都可以。

但是在使用它们时要注意，因为如果调用者保存了对原始对象的引用，`mock.patch`可能无法为其替换函数，因为它仍然受到 Python 是基于引用的语言这一事实的限制，如果您有一个对象的引用，第三方代码就无法轻松地劫持该引用。

因此，请务必在使用要打补丁的对象之前应用`mock.patch`，以减少对原始对象的引用风险。

# 在生产中报告错误

生产软件中最重要的一个方面是在发生错误时得到通知。由于我们不是软件本身的用户，所以只有在软件通知我们时（或者当为时已晚并且用户在抱怨时）才能知道出了什么问题。

基于 Python 标准库，我们可以轻松构建一个解决方案，以便在发生崩溃时通过电子邮件通知开发人员。

# 如何做...

`logging`模块有一种通过电子邮件报告异常的方法，因此我们可以设置一个记录器，并捕获异常以通过电子邮件记录它们：

```py
import logging
import logging.handlers
import functools

crashlogger = logging.getLogger('__crashes__')

def configure_crashreport(mailhost, fromaddr, toaddrs, subject, 
                        credentials, tls=False):
    if configure_crashreport._configured:
        return

    crashlogger.addHandler(
        logging.handlers.SMTPHandler(
            mailhost=mailhost,
            fromaddr=fromaddr,
            toaddrs=toaddrs,
            subject=subject,
            credentials=credentials,
            secure=tuple() if tls else None
        )
    )
    configure_crashreport._configured = True
configure_crashreport._configured = False

def crashreport(f):
    @functools.wraps(f)
    def _crashreport(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            crashlogger.exception(
                '{} crashed\n'.format(f.__name__)
            )
            raise
    return _crashreport
```

一旦这两个函数就位，我们可以配置`logging`，然后装饰我们的主代码入口点，以便代码库中的所有异常都通过电子邮件报告：

```py
@crashreport
def main():
    3 / 0

configure_crashreport(
    'your-smtp-host.com',
    'no-reply@your-smtp-host.com',
    'crashes_receiver@another-smtp-host.com',
    'Automatic Crash Report from TestApp',
    ('smtpserver_username', 'smtpserver_password'),
    tls=True
)

main()
```

# 工作原理...

`logging`模块能够向附加到记录器的任何处理程序发送消息，并且具有通过`.exception`显式记录崩溃的功能。

因此，我们解决方案的根本是用装饰器包装代码库的主函数，以捕获所有异常并调用记录器：

```py
def crashreport(f):
    @functools.wraps(f)
    def _crashreport(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            crashlogger.exception(
                '{} crashed\n'.format(f.__name__)
            )
            raise
    return _crashreport
```

`crashlogger.exception`方法将构建一个包含我们自定义文本的消息（报告装饰函数的名称）以及崩溃的回溯，并将其发送到关联的处理程序。

通过`configure_crashreport`方法，我们为`crashlogger`提供了自定义处理程序。然后处理程序通过电子邮件发送消息：

```py
def configure_crashreport(mailhost, fromaddr, toaddrs, subject, 
                        credentials, tls=False):
    if configure_crashreport._configured:
        return

    crashlogger.addHandler(
        logging.handlers.SMTPHandler(
            mailhost=mailhost,
            fromaddr=fromaddr,
            toaddrs=toaddrs,
            subject=subject,
            credentials=credentials,
            secure=tuple() if tls else None
        )
    )
    configure_crashreport._configured = True
configure_crashreport._configured = False
```

额外的`_configured`标志用作保护，以防止处理程序被添加两次。

然后我们只需调用`configure_crashreport`来提供电子邮件服务的凭据：

```py
configure_crashreport(
    'your-smtp-host.com',
    'no-reply@your-smtp-host.com',
    'crashes_receiver@another-smtp-host.com',
    'Automatic Crash Report from TestApp',
    ('smtpserver_username', 'smtpserver_password'),
    tls=True
)
```

并且函数中的所有异常都将在`crashlogger`中记录，并通过关联的处理程序发送电子邮件。

# 基准测试

在编写软件时，通常需要确保某些性能约束得到保证。标准库中有大部分我们编写的函数的时间和资源消耗的工具。

假设我们有两个函数，我们想知道哪一个更快：

```py
def function1():
    l = []
    for i in range(100):
        l.append(i)
    return l

def function2():
    return [i for i in range(100)]
```

# 如何做...

`timeit`模块提供了一堆实用程序来计时函数或整个脚本：

```py
>>> import timeit

>>> print(
...     timeit.timeit(function1)
... )
10.132873182068579

>>> print(
...     timeit.timeit(function2)
... )
5.13165780401323
```

从报告的时间中，我们知道`function2`比`function1`快两倍。

# 还有更多...

通常，这样的函数会在几毫秒内运行，但报告的时间是以秒为单位的。

这是因为，默认情况下，`timeit.timeit`将运行被基准测试的代码 100 万次，以提供一个结果，其中执行速度的任何临时变化都不会对最终结果产生太大影响。

# 检查

作为一种强大的动态语言，Python 允许我们根据它正在处理的对象的状态来改变其运行时行为。

检查对象的状态是每种动态语言的基础，标准库`inspect`模块具有大部分这种情况所需的功能。

# 如何做...

对于这个示例，需要执行以下步骤：

1.  基于`inspect`模块，我们可以快速创建一个辅助函数，它将告诉我们大多数对象的主要属性和类型：

```py
import inspect

def inspect_object(o):
    if inspect.isfunction(o) or inspect.ismethod(o):
        print('FUNCTION, arguments:', inspect.signature(o))
    elif inspect.isclass(o):
        print('CLASS, methods:', 
              inspect.getmembers(o, inspect.isfunction))
    else:
        print('OBJECT ({}): {}'.format(
            o.__class__, 
            [(n, v) for n, v in inspect.getmembers(o) 
                if not n.startswith('__')]
        ))
```

1.  然后，如果我们将其应用于任何对象，我们将获得有关其类型、属性、方法的详细信息，如果它是一个函数，还有关其参数。我们甚至可以创建一个自定义类型：

```py
class MyClass:
    def __init__(self):
        self.value = 5

    def sum_to_value(self, other):
        return self.value + other
```

1.  我们检查它的方法：

```py
>>> inspect_object(MyClass.sum_to_value)
FUNCTION, arguments: (self, other)
```

该类型的一个实例：

```py
>>> o = MyClass()
>>> inspect_object(o)
OBJECT (<class '__main__.MyClass'>): [
    ('sum_to_value', <bound method MyClass.sum_to_value of ...>), 
    ('value', 5)
]
```

或者类本身：

```py
>>> inspect_object(MyClass)
CLASS, methods: [
    ('__init__', <function MyClass.__init__ at 0x107bd0400>), 
    ('sum_to_value', <function MyClass.sum_to_value at 0x107bd0488>)
]
```

# 它是如何工作的...

`inspect_object`依赖于`inspect.isfunction`、`inspect.ismethod`和`inspect.isclass`来决定提供的参数的类型。

一旦清楚提供的对象适合其中一种类型，它就会为该类型的对象提供更合理的信息。

对于函数和方法，它查看函数的签名：

```py
if inspect.isfunction(o) or inspect.ismethod(o):
    print('FUNCTION, arguments:', inspect.signature(o))
```

`inspect.signature`函数返回一个包含给定方法接受的所有参数详细信息的`Signature`对象。

当打印时，这些参数会显示在屏幕上，这正是我们所期望的：

```py
FUNCTION, arguments: (self, other)
```

对于类，我们主要关注类公开的方法。因此，我们将使用`inspect.getmembers`来获取类的所有属性，然后使用`inspect.isfunction`来仅过滤函数：

```py
elif inspect.isclass(o):
    print('CLASS, methods:', inspect.getmembers(o, inspect.isfunction))
```

`inspect.getmembers`的第二个参数可以是任何谓词，用于过滤成员。

对于对象，我们想要显示对象的属性和方法。

对象通常有数十种方法，这些方法在 Python 中默认提供，以支持标准操作符和行为。这些就是所谓的魔术方法，我们通常不关心。因此，我们只需要列出公共方法和属性：

```py
else:
    print('OBJECT ({}): {}'.format(
        o.__class__, 
        [(n, v) for n, v in inspect.getmembers(o) 
            if not n.startswith('__')]
    ))
```

正如我们所知，`inspect.getmembers`接受一个谓词来过滤要返回的成员。但是谓词只能作用于成员本身；它无法知道它的名称。因此，我们必须使用列表推导来过滤`inspect.getmembers`的结果，删除任何名称以`dunder（__）`开头的属性。

结果是提供的对象的公共属性和方法：

```py
OBJECT (<class '__main__.MyClass'>): [
    ('sum_to_value', <bound method MyClass.sum_to_value of ...>), 
    ('value', 5)
]
```

我们还打印了对象本身的`__class__`，以提供关于我们正在查看的对象类型的提示。

# 还有更多...

`inspect`模块有数十个函数，可以用来深入了解 Python 对象。

在调查第三方代码或实现必须处理未知形状和类型的对象的高度动态代码时，它可以是一个非常强大的工具。

# 代码评估

Python 是一种解释性语言，解释器的功能也暴露在标准库中。

这意味着我们可以评估来自文件或文本源的表达式和语句，并让它们作为 Python 代码在 Python 代码本身中运行。

还可以以相当安全的方式评估代码，允许我们从表达式中创建对象，但阻止执行任何函数。

# 如何做...

本教程的步骤如下：

1.  `eval`、`exec` 和 `ast` 函数和模块提供了执行字符串代码所需的大部分机制：

```py
import ast

def run_python(code, mode='evalsafe'):
    if mode == 'evalsafe':
        return ast.literal_eval(code)
    elif mode == 'eval':
        return eval(compile(code, '', mode='eval'))
    elif mode == 'exec':
        return exec(compile(code, '', mode='exec'))
    else:
        raise ValueError('Unsupported execution model 
                         {}'.format(mode))
```

1.  `evalsafe` 模式中的 `run_python` 函数允许我们以安全的方式运行基本的 Python 表达式。这意味着我们可以根据它们的文字表示创建 Python 对象：

```py
>>> print(run_python('[1, 2, 3]'))
[1, 2, 3]
```

1.  我们不能运行函数或执行更高级的命令，比如索引：

```py
>>> print(run_python('[1, 2, 3][0]'))
[ ... ]
malformed node or string: <_ast.Subscript object at 0x10ee57ba8>
```

1.  如果我们想要运行这些，我们需要以不安全的方式 `eval`：

```py
>>> print(run_python('[1, 2, 3][0]', 'eval'))
1
```

1.  这是不鼓励的，因为它允许在当前解释器会话中执行恶意代码。但即使它允许更广泛的执行，它仍然不允许更复杂的语句，比如函数的定义：

```py
>>> print(run_python('''
... def x(): 
...     print("printing hello")
... x()
... ''', 'eval'))
[ ... ]
invalid syntax (, line 2)
```

1.  为了允许完整的 Python 支持，我们需要使用 `exec` 模式，这将允许执行所有 Python 代码，但不再返回表达式的结果（因为提供的代码可能根本不是表达式）：

```py
>>> print(run_python('''
... def x(): 
...     print("printing hello")
... x()
... ''', 'exec'))
printing hello
None
```

# 跟踪代码

`trace` 模块提供了一个强大且易于使用的工具，可以跟踪运行过程中执行了哪些代码行。

跟踪可以用于确保测试覆盖率，并查看我们的软件或第三方函数的行为。

# 如何做...

您需要执行以下步骤来完成此教程：

1.  我们可以实现一个函数，跟踪提供的函数的执行并返回执行的模块以及每个模块的行：

```py
import trace
import collections

def report_tracing(func, *args, **kwargs):
    outputs = collections.defaultdict(list)

    tracing = trace.Trace(trace=False)
    tracing.runfunc(func, *args, **kwargs)

    traced = collections.defaultdict(set)
    for filename, line in tracing.results().counts:
        traced[filename].add(line)

    for filename, tracedlines in traced.items():
        with open(filename) as f:
            for idx, fileline in enumerate(f, start=1):
                outputs[filename].append(
                  (idx, idx in tracedlines, fileline))
                )  
    return outputs
```

1.  然后，一旦我们有了跟踪，我们需要实际打印它，以便人类能够阅读。为此，我们将阅读每个被跟踪模块的源代码，并使用 `+` 标记打印它，该标记将指示哪些行被执行或未执行：

```py
def print_traced_execution(tracings):
    for filename, tracing in tracings.items():
        print(filename)
        for idx, executed, content in tracing:
            print('{:04d}{}  {}'.format(idx, 
                                        '+' if executed else ' ', 
                                        content),
                end='')
        print()
```

1.  给定任何函数，我们都可以看到在各种条件下执行的代码行：

```py
def function(should_print=False):
    a = 1
    b = 2
    if should_print:
        print('Usually does not execute!')
    return a + b
```

1.  首先，我们可以使用 `should_print=False` 打印函数的跟踪：

```py
>>> print_traced_execution(
...     report_tracing(function)
... )
devtools_08.py
0001   def function(should_print=False):
0002+      a = 1
0003+      b = 2
0004+      if should_print:
0005           print('Usually does not execute!')
0006+      return a + b
```

1.  然后我们可以检查 `should_print=True` 时会发生什么：

```py
>>> print_traced_execution(
...     report_tracing(function, True)
... )
Usually does not execute!
devtools_08.py
0001   def function(should_print=False):
0002+      a = 1
0003+      b = 2
0004+      if should_print:
0005+          print('Usually does not execute!')
0006+      return a + b
```

您可以看到行 `0005` 现在标记为 `+`，因为它被执行了。

# 工作原理...

`report_tracing` 函数实际上负责跟踪另一个函数的执行。

首先，由于执行是按模块进行的，它创建了 `defaultdict`，用于存储跟踪。键将是模块，值将是包含该模块每行信息的列表：

```py
def report_tracing(func, *args, **kwargs):
    outputs = collections.defaultdict(list)
```

然后，它创建了实际的跟踪机制。`trace=False` 选项特别重要，以避免在屏幕上打印跟踪。现在，我们希望将其保存在一边，而不是打印出来。

```py
tracing = trace.Trace(trace=False)
```

一旦跟踪器可用，我们就可以使用它来运行提供的函数并提供任何给定的参数：

```py
tracing.runfunc(func, *args, **kwargs)
```

跟踪的结果保存在跟踪器本身中，因此我们可以使用 `tracing.results()` 访问它。我们感兴趣的是代码行是否至少执行了一次，因此我们将寻找计数，并将每个执行的代码行添加到给定模块的执行代码行集合中：

```py
traced = collections.defaultdict(set)
for filename, line in tracing.results().counts:
    traced[filename].add(line)
```

`traced` 字典包含了给定模块实际执行的所有代码行。顺便说一句，它不包含任何关于未执行的代码行的详细信息。

到目前为止，我们只有行号，没有关于执行的代码行的其他细节。当然，我们也希望有代码行本身，并且希望有所有代码行，而不仅仅是执行的代码行，这样我们就可以打印出没有间隙的源代码。

这就是为什么 `report_tracing` 打开每个执行模块的源代码并读取其内容。对于每一行，它检查它是否在该模块的执行集合中，并存储一对元组，其中包含行号、一个布尔值，指示它是否被执行，以及行内容本身：

```py
for filename, tracedlines in traced.items():
    with open(filename) as f:
        for idx, fileline in enumerate(f, start=1):
            outputs[filename].append((idx, idx in tracedlines, fileline))
```

最后，结果字典包含了所有被执行的模块，以及它们的源代码，注释了关于行号和是否执行的详细信息：

```py
return outputs
```

`print_traced_execution`则更容易：它的唯一目的是获取我们收集的数据并将其打印到屏幕上，以便人类可以看到源代码和执行的内容。

该函数会迭代每个被跟踪的模块并打印`filename`模块：

```py
def print_traced_execution(tracings):
    for filename, tracing in tracings.items():
        print(filename)
```

然后，对于每个模块，它会迭代跟踪详细信息并打印行号（作为四位数，以便对任何行号最多到 9999 进行正确缩进），如果执行了该行，则打印一个`+`号，以及行内容本身：

```py
for idx, executed, content in tracing:
    print('{:04d}{}  {}'.format(idx, 
                                '+' if executed else ' ', 
                                content),
        end='')
print()
```

# 还有更多...

使用跟踪，您可以轻松地检查您编写的代码是否被测试执行。您只需将跟踪限制在您编写并感兴趣的模块上即可。

有一些第三方模块专门用于测试覆盖率报告；最广泛使用的可能是`coverage`模块，它支持最常见的测试框架，如`pytest`和`nose`。

# 性能分析

当您需要加快代码速度或了解瓶颈所在时，性能分析是最有效的技术之一。

Python 标准库提供了一个内置的分析器，用于跟踪每个函数的执行和时间，并允许您找出更昂贵或运行次数过多的函数，消耗了大部分执行时间。

# 如何做...

对于这个示例，需要执行以下步骤：

1.  我们可以选择任何要进行性能分析的函数（甚至可以是程序的主入口点）：

```py
import time

def slowfunc(goslow=False):
    l = []
    for i in range(100):
        l.append(i)
        if goslow:
            time.sleep(0.01)
    return l
```

1.  我们可以使用`cProfile`模块对其进行性能分析。

```py
from cProfile import Profile

profiler = Profile()
profiler.runcall(slowfunc, True)
profiler.print_stats()
```

1.  这将打印函数的时间以及分析函数调用的最慢函数：

```py
202 function calls in 1.183 seconds

Ordered by: standard name

ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    1    0.002    0.002    1.183    1.183 devtools_09.py:3(slowfunc)
  100    1.181    0.012    1.181    0.012 {built-in method time.sleep}
  100    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
```

# 它是如何工作的...

`cProfile.Profile`对象能够使用少量负载运行任何函数并收集执行统计信息。

`runcall`函数是实际运行函数并提供传递的参数的函数（在本例中，`True`作为第一个函数参数提供，这意味着`goslow=True`）：

```py
profiler = Profile()
profiler.runcall(slowfunc, True)
```

一旦收集到了性能分析数据，我们可以将其打印到屏幕上，以提供关于执行的详细信息：

```py
profiler.print_stats()
```

打印输出包括在调用期间执行的函数列表，每个函数所花费的总时间，每个调用中每个函数所花费的时间，以及调用的总次数：

```py
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    1    0.002    0.002    1.183    1.183 devtools_09.py:3(slowfunc)
  100    1.181    0.012    1.181    0.012 {built-in method time.sleep}
  ...
```

我们可以看到，`slowfunc`的主要瓶颈是`time.sleep`调用：它占用了总共`1.183`时间中的`1.181`。

我们可以尝试使用`goslow=False`调用`slowfunc`，并查看时间的变化：

```py
profiler.runcall(slowfunc, False)
profiler.print_stats()
```

而且，在这种情况下，我们看到整个函数运行时间为`0.000`而不是`1.183`，并且不再提到`time.sleep`：

```py
102 function calls in 0.000 seconds

Ordered by: standard name

ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    1    0.000    0.000    0.000    0.000 devtools_09.py:3(slowfunc)
  100    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
```
