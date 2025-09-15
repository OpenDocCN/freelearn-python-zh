

# 第七章：异常和上下文管理器

> 最精心的计划，老鼠和人类都会出错。
> 
> – 罗伯特·彭斯

罗伯特·彭斯这句著名的话应该铭刻在每个程序员的脑海中。即使我们的代码是正确的，错误仍然会发生。如果我们没有正确处理它们，它们可能会使我们的精心策划的计划走向歧途。

未处理的错误可能导致软件崩溃或行为异常。根据所涉及软件的性质，这可能会产生严重后果。因此，学习如何检测和处理错误非常重要。我们鼓励你养成总是思考可能发生的错误以及当它们发生时你的代码应该如何响应的习惯。

本章全部关于错误和应对意外情况。我们将学习关于**异常**的内容，这是 Python 表示错误或其他异常事件发生的方式。我们还将讨论**上下文管理器**，它提供了一种封装和重用错误处理代码的机制。

在本章中，我们将涵盖以下内容：

+   异常

+   上下文管理器

# 异常

尽管我们还没有涉及这个主题，但我们预计到现在你至少对异常有一个模糊的概念。在前几章中，我们看到当迭代器耗尽时，对其调用`next()`会引发`StopIteration`异常。当我们尝试访问一个不在有效范围内的列表位置时，我们得到了`IndexError`。当我们尝试访问一个对象上不存在属性时，我们遇到了`AttributeError`，当我们尝试访问字典中不存在的键时，我们遇到了`KeyError`。在本章中，我们将更深入地讨论异常。

即使一个操作或一段代码是正确的，也常常存在可能出现错误的情况。例如，如果我们正在将用户输入从`str`转换为`int`，用户可能会不小心输入了一个字母代替数字，这使得我们无法将那个值转换为数字。在除法操作中，我们可能事先不知道是否可能尝试除以 0。在打开文件时，它可能不存在或已损坏。

当在执行过程中检测到错误时，它被称为**异常**。异常并不一定是致命的；实际上，`StopIteration`异常已经深深集成到 Python 的生成器和迭代器机制中。然而，通常情况下，如果你不采取必要的预防措施，异常将导致你的应用程序崩溃。有时，这是期望的行为，但在其他情况下，我们希望防止和控制这些问题。例如，如果用户尝试打开一个损坏的文件，我们可以提醒他们问题，并给他们机会修复它。让我们看看几个异常的例子：

```py
# exceptions/first.example.txt 
>>> gen = (n for n in range(2)) 
>>> next(gen) 
0 
>>> next(gen) 
1 
>>> next(gen) 
Traceback (most recent call last): 
  File "<stdin>", line 1, in <module> 
StopIteration 
>>> print(undefined_name) 
Traceback (most recent call last): 
  File "<stdin>", line 1, in <module> 
NameError: name 'undefined_name' is not defined 
>>> mylist = [1, 2, 3] 
>>> mylist[5] 
Traceback (most recent call last): 
  File "<stdin>", line 1, in <module> 
IndexError: list index out of range 
>>> mydict = {"a": "A", "b": "B"} 
>>> mydict["c"] 
Traceback (most recent call last): 
  File "<stdin>", line 1, in <module> 
KeyError: 'c' 
>>> 1 / 0 
Traceback (most recent call last): 
  File "<stdin>", line 1, in <module> 
ZeroDivisionError: division by zero 
```

如你所见，Python 的 shell 非常宽容。我们可以看到`Traceback`，这样我们就有关于错误的信息，但 shell 本身仍然正常运行。这是一种特殊的行为；一个常规程序或脚本如果没有处理异常，通常会立即退出。让我们看一个快速示例：

```py
# exceptions/unhandled.py
1 + "one"
print("This line will never be reached") 
```

如果我们运行这段代码，我们会得到以下输出：

```py
$ python exceptions/unhandled.py
Traceback (most recent call last):
  File "exceptions/unhandled.py", line 2, in <module>
    1 + "one"
    ~~^~~~~~~
TypeError: unsupported operand type(s) for +: 'int' and 'str' 
```

因为我们没有做任何处理异常的事情，所以一旦发生异常，Python 就会立即退出（在打印出错误信息之后）。

## 引发异常

我们之前看到的异常是由 Python 解释器在检测到错误时引发的。然而，你也可以在发生你自己的代码认为的错误的情况时引发异常。要引发异常，请使用`raise`语句。以下是一个示例：

```py
# exceptions/raising.txt
>>> raise NotImplementedError("I'm afraid I can't do that")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NotImplementedError: I'm afraid I can't do that 
```

你可以引发任何类型的异常没有限制。这允许你选择最能描述已发生错误条件的异常类型。你还可以定义自己的异常类型（我们将在下一刻看到如何做到这一点）。请注意，我们传递给`Exception`类的参数被打印出来作为错误消息的一部分。

Python 有太多内置异常无法在此列出，但它们都在[`docs.python.org/3.12/library/exceptions.html#bltin-exceptions`](https://docs.python.org/3.12/library/exceptions.html#bltin-exceptions)中进行了文档说明。

## 定义自己的异常

正如我们在上一节中提到的，你可以定义自己的自定义异常。实际上，对于库来说，定义自己的异常是很常见的。

你需要做的只是定义一个继承自任何其他异常类的类。所有异常都源自`BaseException`；然而，这个类并不打算被直接子类化。你的自定义异常应该继承自`Exception`。实际上，几乎所有内置异常也都继承自`Exception`。不继承自`Exception`的异常是打算由 Python 解释器内部使用的。

## Tracebacks

当 Python 遇到未处理的异常时打印的**traceback**可能一开始看起来令人畏惧，但它对于理解导致异常的原因非常有用。在这个例子中，我们使用一个数学公式来解二次方程；如果你不熟悉它，没关系，因为你不需要理解它。让我们看看 traceback，看看它能告诉我们什么：

```py
# exceptions/trace.back.py
def squareroot(number):
    if number < 0:
        raise ValueError("No negative numbers please") 
    return number**.5

def quadratic(a, b, c):
    d = b**2 - 4 * a * c
    return (
        (-b - squareroot(d)) / (2 * a),
        (-b + squareroot(d)) / (2 * a)
    )
quadratic(1, 0, 1)  # x**2 + 1 == 0 
```

在这里，我们定义了一个名为`quadratic()`的函数，它使用著名的二次公式来找到二次方程的解。我们不是使用`math`模块中的`sqrt()`函数，而是编写了自己的版本（`squareroot()`），如果数字是负数，它会引发异常。当我们调用`quadratic(1, 0, 1)`来解方程*x*² + 1 = 0 时，我们会得到一个`ValueError`，因为`d`是负数。当我们运行这个程序时，我们得到以下结果：

```py
$ python exceptions/trace.back.py
Traceback (most recent call last):
  File "exceptions/trace.back.py", line 16, in <module>
    quadratic(1, 0, 1)  # x**2 + 1 == 0
    ^^^^^^^^^^^^^^^^^^
  File "exceptions/trace.back.py", line 11, in quadratic
    (-b - squareroot(d)) / (2 * a),
          ^^^^^^^^^^^^^
  File "exceptions/trace.back.py", line 4, in squareroot
    raise ValueError("No negative numbers please")
ValueError: No negative numbers please 
```

从下到上阅读堆栈跟踪通常很有用。在最后一行，我们有错误消息，告诉我们出了什么问题：`ValueError: No negative numbers please`。前面的行告诉我们异常是在哪里引发的（`squareroot()` 函数中的 `exceptions/trace.back.py` 文件的第 4 行）。

我们还可以看到导致异常发生的函数调用序列：在模块的最顶层，函数 `quadratic()` 在第 16 行被调用，它又调用了在第 11 行的 `squareroot()` 函数。正如你所看到的，堆栈跟踪就像一张地图，显示了代码中异常发生的位置。沿着这条路径检查每个函数中的代码，当你想要了解异常发生的原因时通常很有帮助。

Python 3.10、3.11 和 3.12 中对错误消息进行了几次改进。例如，在 Python 3.11 中添加了 `^^^^` 字符，在堆栈跟踪中下划线了导致异常的每个语句或表达式的确切部分。

## 异常处理

要在 Python 中处理异常，你使用 `try` 语句。当你进入 `try` 子句时，Python 会监视一个或多个不同类型的异常（根据你的指示），如果它们被引发，它允许你做出反应。

`try` 语句由 `try` 子句组成，它打开语句，后面跟着一个或多个 `except` 子句，这些子句定义了在捕获异常时要执行的操作。`except` 子句后面可以有一个可选的 `else` 子句，当 `try` 子句在没有引发任何异常的情况下退出时执行。在 `except` 和 `else` 子句之后，我们可以有一个可选的 `finally` 子句，其代码无论在其他子句中发生什么都会执行。`finally` 子句通常用于清理资源。你也可以省略 `except` 和 `else` 子句，只保留一个 `try` 子句后跟一个 `finally` 子句。如果我们希望异常在其他地方传播和处理，但我们必须执行一些无论是否发生异常都必须执行的清理代码，这很有帮助。

子句的顺序很重要。它必须是 `try`，`except`，`else`，然后是 `finally`。同时，记住 `try` 后必须跟至少一个 `except` 子句或一个 `finally` 子句。让我们看一个例子：

```py
# exceptions/try.syntax.py 
def try_syntax(numerator, denominator): 
    try: 
        print(f"In the try block: {numerator}/{denominator}") 
        result = numerator / denominator 
    except ZeroDivisionError as zde: 
        print(zde) 
    else: 
        print("The result is:", result) 
        return result 
    finally: 
        print("Exiting")
print(try_syntax(12, 4)) 
print(try_syntax(11, 0)) 
```

这个例子定义了一个简单的 `try_syntax()` 函数。我们执行两个数的除法。我们准备捕获一个 `ZeroDivisionError` 异常，如果用 `denominator = 0` 调用函数，这个异常就会发生。最初，代码进入 `try` 块。如果 `denominator` 不是 `0`，则计算 `result`，然后离开 `try` 块后，执行继续在 `else` 块中。我们打印 `result` 并返回它。看看输出，你会发现就在返回 `result` 之前，这是函数的退出点，Python 执行了 `finally` 子句。

当 `denominator` 为 `0` 时，情况会改变。我们尝试计算 `numerator / denominator` 会引发一个 `ZeroDivisionError`。因此，我们进入 `except` 块并打印 `zde`。

`else` 块没有执行，因为在 `try` 块中引发了异常。在（隐式地）返回 `None` 之前，我们仍然会执行 `finally` 块。看看输出，看看它对你是否有意义：

```py
$ python exceptions/try.syntax.py 
In the try block: 12/4 
The result is: 3.0 
Exiting 
3.0 
In the try block: 11/0 
division by zero 
Exiting 
None 
```

当你执行一个 `try` 块时，你可能想要捕获多个异常。例如，当调用 `divmod()` 函数时，如果第二个参数是 `0`，你会得到一个 `ZeroDivisionError`，如果任一参数不是数字，你会得到一个 `TypeError`。如果你想以相同的方式处理这两个异常，你可以这样组织你的代码：

```py
# exceptions/multiple.py
values = (1, 2)
try:
    q, r = divmod(*values)
except (ZeroDivisionError, TypeError) as e:
    print(type(e), e) 
```

这段代码将捕获 `ZeroDivisionError` 和 `TypeError`。尝试将 `values = (1, 2)` 改为 `values = (1, 0)` 或 `values = ('one', 2)`，你将看到输出发生变化。

如果你需要以不同的方式处理不同的异常类型，你可以使用多个 `except` 子句，如下所示：

```py
# exceptions/multiple.py 
try:
    q, r = divmod(*values)
except ZeroDivisionError:
    print("You tried to divide by zero!")
except TypeError as e:
    print(e) 
```

请记住，异常是在第一个匹配该异常类或其基类的块中处理的。因此，当你像我们这里这样做多个 `except` 子句时，确保将特定的异常放在顶部，通用的异常放在底部。在面向对象编程术语中，派生类应该放在其基类之前。此外，请记住，当引发异常时，只有一个 `except` 处理器被执行。

Python 还允许你使用一个不指定任何异常类型的 `except` 子句（这相当于写 `except BaseException`）。然而，你应该避免这样做，因为这意味着你也会捕获到那些打算由解释器内部使用的异常。这些包括所谓的 *退出系统异常*。这些是 `SystemExit`，当解释器通过调用 `exit()` 函数退出时引发，以及 `KeyboardInterrupt`，当用户通过按下 *Ctrl* + *C*（或在某些系统上是 *Delete*）来终止应用程序时引发。

你也可以在 `except` 子句内部引发异常。例如，你可能想用一个自定义异常替换内置异常（或第三方库中的异常）。当编写库时，这是一个相当常见的技巧，因为它有助于保护用户免受库的实现细节的影响。让我们看一个例子：

```py
# exceptions/replace.txt
>>> class NotFoundError(Exception):
...     pass
...
>>> vowels = {"a": 1, "e": 5, "i": 9, "o": 15, "u": 21}
>>> try:
...     pos = vowels["y"]
... except KeyError as e:
...     raise NotFoundError(*e.args)
...
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
KeyError: 'y'
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
NotFoundError: y 
```

默认情况下，Python 假设发生在 `except` 子句中的异常是一个意外错误，并且会友好地打印出两个异常的跟踪信息。我们可以通过使用 `raise from` 语句来告诉解释器我们故意引发新的异常：

```py
# exceptions/replace.py
>>> try:
...     pos = vowels["y"]
... except KeyError as e:
...     raise NotFoundError(*e.args) from e
...
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
KeyError: 'y'
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
NotFoundError: y 
```

错误信息已更改，但我们仍然得到两个跟踪信息，这对于调试非常有用。如果你真的想完全抑制原始异常，可以使用 `from None` 而不是 `from e`（自己试试）。

你也可以仅使用`raise`，而不指定新的异常，来重新引发原始异常。如果你只想记录异常发生的事实，而不抑制或替换异常，这有时是有用的。

自 Python 3.11 以来，也可以向异常添加注释。这允许你添加额外的信息，作为跟踪信息的一部分显示，而不抑制或替换原始异常。为了了解这是如何工作的，我们将修改本章前面提到的二次公式示例，并向异常添加注释：

```py
# exceptions/note.py
def squareroot(number):
    if number < 0:
        raise ValueError("No negative numbers please")
    return number**0.5
def quadratic(a, b, c):
    d = b**2 - 4 * a * c
    try:
        return (
            (-b - squareroot(d)) / (2 * a),
            (-b + squareroot(d)) / (2 * a),
        )
    except ValueError as e:
        **e.add_note(f"Cannot solve {a}x******2** **+ {b}x + {c} ==** **0****")**
        **raise**
quadratic(1, 0, 1) 
```

我们已经突出显示了添加注释和重新引发异常的行。运行此代码时的输出如下：

```py
$ python exceptions/note.py
Traceback (most recent call last):
  File "exceptions/note.py", line 20, in <module>
    quadratic(1, 0, 1)
  File "exceptions/note.py", line 12, in quadratic
    (-b - squareroot(d)) / (2 * a),
          ^^^^^^^^^^^^^
  File "exceptions/note.py", line 4, in squareroot
    raise ValueError("No negative numbers please")
ValueError: No negative numbers please
Cannot solve 1x**2 + 0x + 1 == 0 
```

注释已打印在原始错误消息下方。你可以通过多次调用`add_note()`来添加所需数量的注释。所有注释都必须是字符串。

使用异常进行编程可能很棘手。你可能会无意中隐藏那些本应提醒你其存在的错误。通过牢记以下简单指南来确保安全：

+   尽量使`try`子句尽可能短。它应该只包含可能引发你想要处理的异常（s）的代码。

+   尽可能使`except`子句尽可能具体。可能有人会想只写`except Exception`，但如果你这样做，你几乎肯定会捕获到你实际上并不想捕获的异常。

+   使用测试来确保你的代码能够正确处理预期的和意外的错误。我们将在第十章*测试*中更详细地讨论编写测试。

如果你遵循这些建议，你将最大限度地减少出错的可能性。

## 异常组

当处理大量数据集时，如果发生错误，立即停止并引发异常可能不方便。通常更好的做法是处理所有数据，并在最后报告所有发生的错误。这使用户能够一次性处理所有错误，而不是需要多次重新运行过程，逐个修复错误。

实现这一目标的一种方法是通过构建一个错误列表并返回它。然而，这种方法有一个缺点，那就是你不能使用`try` / `except`语句来处理错误。一些库通过创建一个容器异常类并将收集到的错误包装在这个类的实例中来解决这个问题。这允许你在`except`子句中处理容器异常，并检查它以访问嵌套的异常。

自 Python 3.11 以来，有一个新的内置异常类`ExceptionGroup`，它被专门设计为这样的容器异常。将此功能内置到语言中的优点是，跟踪信息也会显示每个嵌套异常的跟踪信息。

例如，假设我们需要验证一个年龄列表，以确保所有值都是正整数。我们可以编写如下内容：

```py
# exceptions/groups/util.py
def validate_age(age):
    if not isinstance(age, int):
        raise TypeError(f"Not an integer: {age}")
    if age < 0:
        raise ValueError(f"Negative age: {age}")
def validate_ages(ages):
    errors = []
    for age in ages:
        try:
            validate_age(age)
        except Exception as e:
            errors.append(e)
    if errors:
        raise ExceptionGroup("Validation errors", errors) 
```

`validate_ages()` 函数对 `ages` 的每个元素调用 `validate_age()`。它捕获发生的任何异常并将它们追加到 `errors` 列表中。如果循环完成后错误列表不为空，我们抛出 `ExceptionGroup`，传入错误消息 `"Validation errors"` 和发生的错误列表。

如果我们从 Python 控制台调用这个函数，并传入包含一些无效年龄的列表，我们会得到以下跟踪输出：

```py
# exceptions/groups/exc.group.txt
>>> from util import validate_ages
>>> validate_ages([24, -5, "ninety", 30, None])
  + Exception Group Traceback (most recent call last):
  |   File "<stdin>", line 1, in <module>
  |   File "exceptions/groups/util.py", line 20, in validate_ages
  |     raise ExceptionGroup("Validation errors", errors)
  | ExceptionGroup: Validation errors (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "exceptions/groups/util.py", line 15, in validate_ages
    |     validate_age(age)
    |   File "exceptions/groups/util.py", line 8, in validate_age
    |     raise ValueError(f"Negative age: {age}")
    | ValueError: Negative age: -5
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "exceptions/groups/util.py", line 15, in validate_ages
    |     validate_age(age)
    |   File "exceptions/groups/util.py", line 6, in validate_age
    |     raise TypeError(f"Not an integer: {age}")
    | TypeError: Not an integer: ninety
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "exceptions/groups/util.py", line 15, in validate_ages
    |     validate_age(age)
    |   File "exceptions/groups/util.py", line 6, in validate_age
    |     raise TypeError(f"Not an integer: {age}")
    | TypeError: Not an integer: None
    +------------------------------------ 
```

注意，我们得到了 `ExceptionGroup` 的跟踪输出，包括我们抛出时指定的错误消息（`"Validation errors"`）以及指示该组包含三个子异常。在此之下缩进，我们得到每个嵌套子异常的跟踪输出。为了提高可读性，子异常跟踪输出被编号并用虚线分隔。

我们可以像处理其他类型的异常一样处理 `ExceptionGroup` 异常：

```py
# exceptions/groups/handle.group.txt
>>> from util import validate_ages
>>> try:
...     validate_ages([24, -5, "ninety", 30, None])
... except ExceptionGroup as e:
...     print(e)
...     print(e.exceptions)
...
Validation errors (3 sub-exceptions)
(ValueError('Negative age: -5'),
 TypeError('Not an integer: ninety'),
 TypeError('Not an integer: None')) 
```

注意，我们可以通过（只读的）`exceptions` 属性访问嵌套的子异常列表。

PEP 654（[`peps.python.org/pep-0654/`](https://peps.python.org/pep-0654/)），它将 `ExceptionGroup` 引入语言，还引入了 `try` / `except` 语句的新变体，允许我们在 `ExceptionGroup` 内部处理特定类型的嵌套子异常。这种新语法使用关键字 `except*` 而不是 `except`。在我们的验证示例中，这允许我们对无效类型和无效值进行单独处理，而无需手动迭代和过滤异常：

```py
# exceptions/groups/handle.nested.txt
>>> from util import validate_ages
>>> try:
...     validate_ages([24, -5, "ninety", 30, None])
... except* TypeError as e:
...     print("Invalid types")
...     print(type(e), e)
...     print(e.exceptions)
... except* ValueError as e:
...     print("Invalid values")
...     print(type(e), e)
...     print(e.exceptions)
...
Invalid types
<class 'ExceptionGroup'> Validation errors (2 sub-exceptions)
(TypeError('Not an integer: ninety'),
 TypeError('Not an integer: None'))
Invalid values
<class 'ExceptionGroup'> Validation errors (1 sub-exception)
(ValueError('Negative age: -5'),) 
```

`validate_ages()` 的调用抛出一个包含三个异常的异常组：两个 `TypeError` 实例和一个 `ValueError`。解释器将每个 `except*` 子句与嵌套异常匹配。第一个子句匹配，因此解释器创建一个新的 `ExceptionGroup`，包含原始组中的所有 `TypeError` 实例，并将其分配给此子句体内的 `e`。我们打印字符串 `"Invalid types"`，然后是 `e` 的类型和值以及 `e.exceptions`。然后剩余的异常将与下一个 `except*` 子句匹配。

这次，所有的 `ValueError` 实例都匹配，因此 `e` 被分配给一个新的包含这些异常的 `ExceptionGroup`。我们打印字符串 `"Invalid values"`，然后是 `type(e)`，`e` 和 `e.exceptions`。此时，组中不再有未处理的异常，因此执行恢复正常。

重要的是要注意，这种行为与正常的 `try` / `except` 语句不同。在正常的 `try` / `except` 语句中，只有一个 `except` 子句被执行：第一个匹配抛出异常的子句。在 `try` / `except*` 语句中，每个匹配的 `except*` 子句都会被执行，直到组中不再有未处理的异常。如果在所有 `except*` 子句处理完毕后仍有未处理的异常，它们将在最后作为新的 `ExceptionGroup` 重新抛出：

```py
# exceptions/groups/handle.nested.txt
>>> try:
...     validate_ages([24, -5, "ninety", 30, None])
... except* ValueError as e:
...     print("Invalid values")
...
Invalid values
  + Exception Group Traceback (most recent call last):
  |   File "<stdin>", line 2, in <module>
  |   File "exceptions/groups/util.py", line 20, in validate_ages
  |     raise ExceptionGroup("Validation errors", errors)
  | ExceptionGroup: Validation errors (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "exceptions/groups/util.py", line 15, in validate_ages
    |     validate_age(age)
    |   File "exceptions/groups/util.py", line 6, in validate_age
    |     raise TypeError(f"Not an integer: {age}")
    | TypeError: Not an integer: ninety
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "exceptions/groups/util.py", line 15, in validate_ages
    |     validate_age(age)
    |   File "exceptions/groups/util.py", line 6, in validate_age
    |     raise TypeError(f"Not an integer: {age}")
    | TypeError: Not an integer: None
    +------------------------------------ 
```

另一个需要注意的重要点是，如果在不是`ExceptionGroup`实例的`try` / `except*`语句中抛出异常，其类型将与`except*`子句进行匹配。如果找到匹配项，异常将在传递到`except*`主体之前被包装在一个`ExceptionGroup`中：

```py
# exceptions/groups/handle.nested.txt
>>> try:
...     raise RuntimeError("Ungrouped")
... except* RuntimeError as e:
...     print(type(e), e)
...     print(e.exceptions)
...
<class 'ExceptionGroup'>  (1 sub-exception)
(RuntimeError('Ungrouped'),) 
```

这意味着我们总是可以安全地假设在`except*`子句中处理的异常是一个`ExceptionGroup`实例。

## 不仅用于错误

在我们继续讨论上下文管理器之前，我们想向你展示异常的不同用法。在这个例子中，我们将演示异常不仅可以用于错误：

```py
# exceptions/for.loop.py 
n = 100 
found = False 
for a in range(n): 
    if found:
        break 
    for b in range(n): 
        if found:
            break 
        for c in range(n): 
            if 42 * a + 17 * b + c == 5096: 
                found = True 
                print(a, b, c)  # 79 99 95
                break 
```

在上面的代码中，我们使用三个嵌套循环来找到一个满足特定方程的三个整数（`a`、`b`和`c`）的组合。在每个外部循环的开始，我们检查一个标志（`found`）的值，当我们找到一个方程的解时，该标志被设置为`True`。这使我们能够在找到解时尽可能快地跳出所有三个循环。我们认为检查标志的逻辑相当不优雅，因为它掩盖了其余的代码，所以我们想出了一个替代方法：

```py
# exceptions/for.loop.py 
class ExitLoopException(Exception): 
    pass 

try: 
    n = 100 
    for a in range(n): 
        for b in range(n): 
            for c in range(n): 
                if 42 * a + 17 * b + c == 5096: 
                    raise ExitLoopException(a, b, c) 
except ExitLoopException as ele: 
    print(ele.args)  # (79, 99, 95) 
```

希望你能欣赏这种方式的优雅。现在，跳出逻辑完全由一个简单的异常来处理，其名称甚至暗示了其目的。一旦找到结果，我们就使用满足我们条件的值抛出`ExitLoopException`，然后立即将控制权交给处理它的`except`子句。注意，我们可以使用异常的`args`属性来获取传递给构造函数的值。

现在我们应该已经很好地理解了异常是什么，以及它们是如何被用来管理错误、流程和异常情况的。我们准备好继续下一个主题：**上下文管理器**。

# 上下文管理器

当与外部资源一起工作时，我们通常在完成工作后需要执行一些清理步骤。例如，在将数据写入文件后，我们需要关闭文件。未能正确清理可能会导致各种错误。因此，我们必须确保即使在发生异常的情况下，我们的清理代码也会被执行。我们可以使用`try` / `finally`语句，但这并不总是方便，并且可能会导致大量重复，因为我们经常在处理特定类型的资源时必须执行类似的清理步骤。**上下文管理器**通过创建一个执行上下文来解决这个问题，在这个上下文中我们可以使用资源，并在离开该上下文时自动执行任何必要的清理，即使抛出了异常。

上下文管理器的另一个用途是暂时更改我们程序的全球状态。我们可能想要暂时修改的全球状态的一个例子是十进制计算的精度。例如，在数据科学应用中，我们有时需要以特定的精度执行特定的计算，但我们希望保留其余计算中的默认精度。我们可以通过以下方式实现这一点：

```py
# context/decimal.prec.py
from decimal import Context, Decimal, getcontext, setcontext
one = Decimal("1")
three = Decimal("3")
orig_ctx = getcontext()
ctx = Context(prec=5)
**setcontext(ctx)**
print(f"{ctx}\n")
print("Custom decimal context:", one / three)
**setcontext(orig_ctx)**
print("Original context restored:", one / three) 
```

注意，我们存储了当前上下文，设置了一个新的上下文（具有修改后的精度），执行了我们的计算，最后恢复了原始上下文。

你可能还记得，`Decimal` 类允许我们使用十进制数进行任意精度的计算。如果不记得，现在你可以回顾一下 *第二章*，*内置数据类型*，的相关部分。

运行此代码会产生以下输出：

```py
$ python context/decimal.prec.py
Context(prec=5, rounding=ROUND_HALF_EVEN, Emin=-999999,
        Emax=999999, capitals=1, clamp=0, flags=[],
        traps=[InvalidOperation, DivisionByZero, Overflow])
Custom decimal context: 0.33333
Original context restored: 0.3333333333333333333333333333 
```

在上面的例子中，我们打印了 `context` 对象以显示它包含的内容。其余的代码看起来没有问题，但如果在恢复原始上下文之前发生异常，所有后续计算的结果都将是不正确的。我们可以通过使用 `try` / `finally` 语句来修复这个问题：

```py
# context/decimal.prec.try.py
from decimal import Context, Decimal, getcontext, setcontext
one = Decimal("1")
three = Decimal("3")
orig_ctx = getcontext()
ctx = Context(prec=5)
setcontext(ctx)
try:
    print("Custom decimal context:", one / three)
finally:
    setcontext(orig_ctx)
print("Original context restored:", one / three) 
```

这样更安全。即使 `try` 块中发生异常，我们也会始终恢复原始上下文。但是，每次需要使用修改后的精度工作时，都必须保存上下文并在 `try` / `finally` 语句中恢复它，这并不方便。这样做也会违反 **DRY** 原则。我们可以通过使用 `decimal` 模块中的 `localcontext` 上下文管理器来避免这种情况。这个上下文管理器会为我们设置和恢复上下文：

```py
# context/decimal.prec.ctx.py
from decimal import Context, Decimal, localcontext
one = Decimal("1")
three = Decimal("3")
**with** localcontext(Context(prec=5)) as ctx:
    print("Custom decimal context:", one / three)
print("Original context restored:", one / three) 
```

`with` 语句用于进入由 `localcontext` 上下文管理器定义的运行时上下文。当退出由 `with` 语句分隔的代码块时，上下文管理器定义的任何清理操作（在这种情况下，恢复十进制上下文）会自动执行。

还有可能在一个 `with` 语句中组合多个上下文管理器。这在需要同时处理多个资源的情况下非常有用：

```py
# context/multiple.py
from decimal import Context, Decimal, localcontext
one = Decimal("1")
three = Decimal("3")
with (
    localcontext(Context(prec=5)),
    open("output.txt", "w") as out_file
):
    out_file.write(f"{one} / {three} = {one / three}\n") 
```

在这里，我们进入一个局部上下文，并在一个 `with` 语句中打开一个文件（它充当上下文管理器）。我们执行计算并将结果写入文件。当我们退出 `with` 块时，文件会自动关闭，并且默认的十进制上下文会恢复。现在不必太担心与文件操作相关的细节；我们将在 *第八章*，*文件和数据持久性* 中详细讨论。

在 Python 3.10 之前，像我们这里这样在多个上下文管理器周围使用括号会导致 `SyntaxError`。在 Python 的旧版本中，我们必须将两个上下文管理器放入一行代码中，或者将换行符放在 `localcontext()` 或 `open()` 调用的括号内。

除了十进制上下文和文件之外，Python 标准库中的许多其他对象也可以用作上下文管理器。以下是一些示例：

+   实现低级网络接口的套接字对象可以用作上下文管理器来自动关闭网络连接。

+   在并发编程中用于同步的锁类使用上下文管理器协议来自动释放锁。

在本章的其余部分，我们将向您展示如何实现您自己的上下文管理器。

## 基于类的上下文管理器

上下文管理器通过两个魔术方法工作：`__enter__()`在进入`with`语句的主体之前被调用，而`__exit__()`在退出`with`语句主体时被调用。这意味着您可以通过编写一个实现这些方法的类来创建自己的上下文管理器：

```py
# context/manager.class.py
class MyContextManager:
    def __init__(self):
        print("MyContextManager init", id(self))
    def __enter__(self):
        print("Entering 'with' context")
        **return****self**
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"{exc_type=} {exc_val=} {exc_tb=}")
        print("Exiting 'with' context")
        **return****True** 
```

在这里，我们定义了一个名为`MyContextManager`的上下文管理器类。关于这个类有几个有趣的地方需要注意。请注意，`__enter__()`方法返回`self`。这是很常见的，但并非必须；您可以从`__enter__()`返回任何您想要的内容，甚至`None`。`__enter__()`方法的返回值将被分配给`with`语句中`as`子句中命名的变量。此外，请注意`__exit__()`函数的`exc_type`、`exc_val`和`exc_tb`参数。如果在`with`语句的主体内部抛出异常，解释器将通过这些参数将异常的*类型*、*值*和*跟踪信息*作为参数传递。如果没有抛出异常，所有三个参数都将为`None`。

此外，请注意`__exit__()`方法返回`True`。这将导致在`with`语句主体内部抛出的任何异常被抑制（就像我们在`try` / `except`语句中处理它一样）。如果我们返回`False`而不是`True`，那么这样的异常在`__exit__()`方法执行后将继续传播。抑制异常的能力意味着上下文管理器可以用作异常处理程序。这种做法的好处是我们可以在需要的地方重用我们的异常处理逻辑。

让我们看看我们的上下文管理器是如何工作的：

```py
# context/manager.class.py
**ctx_mgr = MyContextManager()**
print("About to enter 'with' context")
**with** **ctx_mgr** **as** **mgr:**
    print("Inside 'with' context")
    print(id(mgr))
    raise Exception("Exception inside 'with' context")
    print("This line will never be reached")
print("After 'with' context") 
```

在这里，我们在`with`语句之前单独声明了我们的上下文管理器。我们这样做是为了让您更容易看到正在发生的事情。然而，将这些步骤合并，如`with MyContextManager() as mgr`，更为常见。运行此代码会产生以下输出：

```py
$ python context/manager.class.py
MyContextManager init 140340228792272
About to enter 'with' context
Entering 'with' context
Inside 'with' context
140340228792272
exc_type=<class 'Exception'> exc_val=Exception("Exception inside
'with' context") exc_tb=<traceback object at 0x7fa3817c5340>
Exiting 'with' context
After 'with' context 
```

仔细研究这个输出，以确保您理解正在发生的事情。我们打印了一些 ID，以帮助验证分配给`mgr`的对象确实是来自`__enter__()`返回的同一个对象。尝试更改`__enter__()`和`__exit__()`方法的返回值，看看会有什么影响。

## 基于生成器的上下文管理器

如果你正在实现一个表示需要获取和释放的资源类的类，将其实现为上下文管理器是有意义的。然而，有时我们想要实现上下文管理器行为，但没有一个类适合附加这种行为。例如，我们可能只想使用上下文管理器来重用一些错误处理逻辑。在这种情况下，不得不编写一个额外的类来纯粹实现所需的上下文管理器行为，这会相当繁琐。

来自 `contextlib` 模块的 `contextmanager` 装饰器对于这种情况非常有用。它接受一个 *生成器函数* 并将其转换为上下文管理器（如果你不记得生成器函数是如何工作的，你应该回顾一下 *第五章* ，*列表推导式和生成器*）。装饰器将生成器包装在一个上下文管理器对象中。该对象的 `__enter__()` 方法启动生成器并返回生成器产生的任何内容。如果在 `with` 语句的主体内部发生异常，`__exit__()` 方法将异常传递给生成器（使用生成器的 `throw` 方法）。否则，`__exit__()` 简单地调用生成器的 `next` 方法。请注意，生成器只能产生一次；如果生成器第二次产生，将引发 `RuntimeError`。让我们将之前的示例转换为基于生成器的上下文管理器：

```py
# context/generator.py
from contextlib import contextmanager
@contextmanager
def my_context_manager():
    print("Entering 'with' context")
    val = object()
    print(id(val))
    try:
        **yield** **val**
    except Exception as e:
        print(f"{type(e)=} {e=} {e.__traceback__=}")
    finally:
        print("Exiting 'with' context")
print("About to enter 'with' context")
with my_context_manager() as val:
    print("Inside 'with' context")
    print(id(val))
    raise Exception("Exception inside 'with' context")
    print("This line will never be reached")
print("After 'with' context") 
```

运行此代码的输出与之前的示例类似：

```py
$ python context/generator.py
About to enter 'with' context
Entering 'with' context
139768531985040
Inside 'with' context
139768531985040
type(e)=<class 'Exception'> e=Exception("Exception inside 'with'
context") e.__traceback__=<traceback object at 0x7f1e65a42800>
Exiting 'with' context
After 'with' context 
```

大多数基于生成器的上下文管理器生成器在这个示例中具有类似的结构 `my_context_manager()`。它们有一些设置代码，然后是在 `try` 语句内部的 `yield`。在这里，我们产生了一个任意对象，以便你可以看到通过 `with` 语句的 `as` 子句提供了相同的对象。通常，也会有一个不带值的裸 `yield`（在这种情况下，产生 `None`）。这相当于从上下文管理器类的方法 `__enter__()` 中返回 `None`。在这种情况下，`with` 语句的 `as` 子句通常会被省略。

基于生成器的上下文管理器的另一个有用特性是它们也可以用作函数装饰器。这意味着如果函数的全部主体需要位于 `with` 语句的上下文中，你可以节省一个缩进级别，只需装饰该函数即可。

除了 `contextmanager` 装饰器之外，`contextlib` 模块还包含许多有用的上下文管理器。文档还提供了使用和实现上下文管理器的几个有帮助的示例。确保你阅读了它：[`docs.python.org/3/library/contextlib.html`](https://docs.python.org/3/library/contextlib.html)。

我们在本节中给出的示例并没有做任何有用的事情。它们被创建纯粹是为了向您展示上下文管理器是如何工作的。仔细研究这些示例，直到您确信您完全理解了它们。然后开始编写您自己的上下文管理器（无论是作为类还是生成器）。尝试将本章前面看到的用于从嵌套循环中退出的 `try` / `except` 语句转换为上下文管理器。我们在 *第六章* 中编写的 `measure` 装饰器也是一个很好的候选，可以转换为上下文管理器。

# 摘要

在本章中，我们探讨了异常和上下文管理器。

我们看到异常是 Python 用来表示发生错误的方式。我们向您展示了如何捕获异常，以便在错误不可避免地发生时，您的程序不会失败。

我们还向您展示了您如何在自己的代码检测到错误时引发异常，并且您可以定义自己的异常类型。我们看到了异常组和扩展 `except` 子句的新语法。我们通过看到异常不仅用于表示错误，还可以用作流程控制机制来结束对异常的探索。

我们在本章的结尾简要概述了上下文管理器。我们展示了如何使用 `with` 语句进入由上下文管理器定义的上下文，当退出上下文时，上下文管理器会执行清理操作。我们还向您展示了如何创建自己的上下文管理器，无论是作为类的一部分还是通过使用生成器函数。

我们将在下一章中看到更多上下文管理器的实际应用，该章重点介绍文件和数据持久性。

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

`discord.com/invite/uaKmaz7FEC`

![img](img/QR_Code119001106417026468.png)
