# 第四章：预料之外的情况

程序非常脆弱。如果代码总是返回有效的结果那就太理想了，但有时候无法计算出有效的结果。例如，不能除以零，或者访问五项列表中的第八项。

在过去，唯一的解决方法是严格检查每个函数的输入，以确保它们是有意义的。通常，函数有特殊的返回值来指示错误条件；例如，它们可以返回一个负数来表示无法计算出正值。不同的数字可能表示不同的错误。调用这个函数的任何代码都必须明确检查错误条件并相应地采取行动。很多代码并没有这样做，程序就会崩溃。然而，在面向对象的世界中，情况并非如此。

在本章中，我们将学习**异常**，这些特殊的错误对象只有在有意义处理它们时才需要处理。特别是，我们将涵盖：

+   如何引发异常

+   当异常发生时如何恢复

+   如何以不同的方式处理不同类型的异常

+   当异常发生时进行清理

+   创建新类型的异常

+   使用异常语法进行流程控制

# 引发异常

原则上，异常只是一个对象。有许多不同的异常类可用，我们也可以轻松地定义更多自己的异常类。它们所有的共同之处就是它们都继承自一个内置类叫做`BaseException`。当这些异常对象在程序的控制流中被处理时，它们就变得特殊起来。当异常发生时，原本应该发生的一切都不会发生，除非在异常发生时本来就应该发生。有道理吗？别担心，你会明白的！

引发异常的最简单方法是做一些愚蠢的事情！你很可能已经这样做过，并看到了异常输出。例如，每当 Python 遇到你的程序中无法理解的一行时，它就会以`SyntaxError`退出，这是一种异常。这是一个常见的例子：

```py
>>> print "hello world"
 **File "<stdin>", line 1
 **print "hello world"
 **^
SyntaxError: invalid syntax

```

这个`print`语句在 Python 2 和之前的版本中是一个有效的命令，但在 Python 3 中，因为`print`现在是一个函数，我们必须用括号括起参数。所以，如果我们将前面的命令输入到 Python 3 解释器中，就会得到`SyntaxError`。

除了`SyntaxError`，还有一些其他常见的异常，我们可以处理，如下例所示：

```py
>>> x = 5 / 0
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
ZeroDivisionError: int division or modulo by zero

>>> lst = [1,2,3]
>>> print(lst[3])
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
IndexError: list index out of range

>>> lst + 2
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
TypeError: can only concatenate list (not "int") to list

>>> lst.add
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
AttributeError: 'list' object has no attribute 'add'

>>> d = {'a': 'hello'}
>>> d['b']
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
KeyError: 'b'

>>> print(this_is_not_a_var)
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
NameError: name 'this_is_not_a_var' is not defined

```

有时这些异常是我们程序中出了问题的指示器（在这种情况下，我们会去到指示的行号并修复它），但它们也会出现在合法的情况下。`ZeroDivisionError`并不总是意味着我们收到了无效的输入。它也可能意味着我们收到了不同的输入。用户可能误输入了零，或者故意输入了零，或者它可能代表一个合法的值，比如一个空的银行账户或一个新生儿的年龄。

你可能已经注意到所有前面的内置异常都以`Error`结尾。在 Python 中，`error`和`exception`这两个词几乎可以互换使用。错误有时被认为比异常更严重，但它们的处理方式完全相同。事实上，前面例子中的所有错误类都以`Exception`（它扩展自`BaseException`）作为它们的超类。

## 引发异常

我们将在一分钟内处理异常，但首先，让我们发现如果我们正在编写一个需要通知用户或调用函数输入某种无效的程序，我们应该做什么。如果我们能使用 Python 使用的相同机制，那不是很好吗？好吧，我们可以！这是一个简单的类，只有当它们是偶数的整数时才向列表添加项目：

```py
class EvenOnly(list):
    def append(self, integer):
        if not isinstance(integer, int):
            **raise TypeError("Only integers can be added")
        if integer % 2:
            **raise ValueError("Only even numbers can be added")
        super().append(integer)
```

这个类扩展了内置的`list`，就像我们在第二章中讨论的那样，*Python 中的对象*，并覆盖了`append`方法以检查两个条件，以确保项目是偶数。我们首先检查输入是否是`int`类型的实例，然后使用模运算符确保它可以被 2 整除。如果两个条件中的任何一个不满足，`raise`关键字会引发异常。`raise`关键字后面紧跟着作为异常引发的对象。在前面的例子中，两个对象是从内置类`TypeError`和`ValueError`新构造的。引发的对象也可以是我们自己创建的新异常类的实例（我们很快就会看到），在其他地方定义的异常，甚至是先前引发和处理的异常对象。如果我们在 Python 解释器中测试这个类，我们可以看到在异常发生时输出了有用的错误信息，就像以前一样：

```py
>>> e = EvenOnly()
>>> e.append("a string")
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
 **File "even_integers.py", line 7, in add
 **raise TypeError("Only integers can be added")
TypeError: Only integers can be added

>>> e.append(3)
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
 **File "even_integers.py", line 9, in add
 **raise ValueError("Only even numbers can be added")
ValueError: Only even numbers can be added
>>> e.append(2)

```

### 注意

虽然这个类对于演示异常很有效，但它并不擅长其工作。仍然可以使用索引表示法或切片表示法将其他值放入列表中。可以通过覆盖其他适当的方法来避免所有这些问题，其中一些是双下划线方法。

## 异常的影响

当引发异常时，似乎会立即停止程序执行。在引发异常后应该运行的任何行都不会被执行，除非处理异常，否则程序将以错误消息退出。看看这个简单的函数：

```py
def no_return():
    print("I am about to raise an exception")
    **raise Exception("This is always raised")
    print("This line will never execute")
    return "I won't be returned"
```

如果我们执行这个函数，我们会看到第一个`print`调用被执行，然后引发异常。第二个`print`语句从未执行，`return`语句也从未执行：

```py
>>> no_return()
I am about to raise an exception
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
 **File "exception_quits.py", line 3, in no_return
 **raise Exception("This is always raised")
Exception: This is always raised

```

此外，如果我们有一个调用另一个引发异常的函数的函数，在第二个函数被调用的地方之后，第一个函数中的任何内容都不会被执行。引发异常会停止所有执行，直到函数调用堆栈，直到它被处理或迫使解释器退出。为了演示，让我们添加一个调用先前函数的第二个函数：

```py
def call_exceptor():
    print("call_exceptor starts here...")
    **no_return()
    print("an exception was raised...")
    print("...so these lines don't run")
```

当我们调用这个函数时，我们看到第一个`print`语句被执行，以及`no_return`函数中的第一行。但一旦引发异常，就不会执行其他任何内容：

```py
>>> call_exceptor()
call_exceptor starts here...
I am about to raise an exception
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
 **File "method_calls_excepting.py", line 9, in call_exceptor
 **no_return()
 **File "method_calls_excepting.py", line 3, in no_return
 **raise Exception("This is always raised")
Exception: This is always raised

```

很快我们会看到，当解释器实际上没有采取捷径并立即退出时，我们可以在任一方法内部对异常做出反应并处理。事实上，异常可以在初始引发后的任何级别上处理。

从下到上查看异常的输出（称为回溯），注意两个方法是如何列出的。在`no_return`内部，异常最初被引发。然后，在其上方，我们看到在`call_exceptor`内部，那个讨厌的`no_return`函数被调用，异常上升到调用方法。从那里，它再上升一级到主解释器，不知道该怎么处理它，于是放弃并打印了回溯。

## 处理异常

现在让我们看看异常的另一面。如果我们遇到异常情况，我们的代码应该如何反应或从中恢复？我们通过在`try`...`except`子句中包装可能引发异常的任何代码（无论是异常代码本身，还是调用任何可能在其中引发异常的函数或方法）来处理异常。最基本的语法看起来像这样：

```py
try:
    no_return()
except:
    print("I caught an exception")
print("executed after the exception")
```

如果我们使用我们现有的`no_return`函数运行这个简单的脚本，我们知道它总是会引发异常，我们会得到这个输出：

```py
I am about to raise an exception
I caught an exception
executed after the exception
```

`no_return`函数愉快地通知我们它即将引发异常，但我们欺骗了它并捕获了异常。一旦捕获，我们就能够清理自己（在这种情况下，通过输出我们正在处理的情况），然后继续前进，而不受那个冒犯性的函数的干扰。`no_return`函数中剩余的代码仍未执行，但调用函数的代码能够恢复并继续。

请注意`try`和`except`周围的缩进。`try`子句包装可能引发异常的任何代码。然后，`except`子句与`try`行处于相同的缩进级别。在`except`子句之后缩进处理异常的任何代码。然后，正常的代码在原始缩进级别上恢复。

前面代码的问题是它会捕获任何类型的异常。如果我们编写的代码可能引发`TypeError`和`ZeroDivisionError`，我们可能想捕获`ZeroDivisionError`，但让`TypeError`传播到控制台。你能猜到语法是什么吗？

这是一个相当愚蠢的函数：

```py
def funny_division(divider):
    try:
        return 100 / divider
    **except ZeroDivisionError:
        return "Zero is not a good idea!"

print(funny_division(0))
print(funny_division(50.0))
print(funny_division("hello"))
```

该功能通过`print`语句进行测试，显示其行为符合预期：

```py
Zero is not a good idea!
2.0
Traceback (most recent call last):
 **File "catch_specific_exception.py", line 9, in <module>
 **print(funny_division("hello"))
 **File "catch_specific_exception.py", line 3, in funny_division
 **return 100 / anumber
TypeError: unsupported operand type(s) for /: 'int' and 'str'.

```

输出的第一行显示，如果我们输入`0`，我们会得到适当的模拟。如果我们使用有效的数字调用（请注意它不是整数，但仍然是有效的除数），它会正确运行。但是，如果我们输入一个字符串（你一直在想如何获得`TypeError`，对吧？），它会引发异常。如果我们使用一个没有指定`ZeroDivisionError`的空`except`子句，当我们发送一个字符串时，它会指责我们除以零，这根本不是正确的行为。

我们甚至可以捕获两个或更多不同的异常，并用相同的代码处理它们。以下是一个引发三种不同类型异常的示例。它使用相同的异常处理程序处理`TypeError`和`ZeroDivisionError`，但如果您提供数字`13`，它也可能引发`ValueError`：

```py
def funny_division2(anumber):
    try:
        if anumber == 13:
            raise ValueError("13 is an unlucky number")
        return 100 / anumber
    **except (ZeroDivisionError, TypeError):
        return "Enter a number other than zero"

for val in (0, "hello", 50.0, 13):

    print("Testing {}:".format(val), end=" ")
    print(funny_division2(val))
```

底部的`for`循环循环遍历了几个测试输入并打印了结果。如果你对`print`语句中的`end`参数感到困惑，它只是将默认的尾随换行符转换为空格，以便与下一行的输出连接在一起。以下是程序的运行：

```py
Testing 0: Enter a number other than zero
Testing hello: Enter a number other than zero
Testing 50.0: 2.0
Testing 13: Traceback (most recent call last):
 **File "catch_multiple_exceptions.py", line 11, in <module>
 **print(funny_division2(val))
 **File "catch_multiple_exceptions.py", line 4, in funny_division2
 **raise ValueError("13 is an unlucky number")
ValueError: 13 is an unlucky number

```

数字`0`和字符串都被`except`子句捕获，并打印出合适的错误消息。数字`13`的异常没有被捕获，因为它是一个`ValueError`，而这种类型的异常没有被处理。这一切都很好，但是如果我们想捕获不同的异常并对它们采取不同的处理怎么办？或者也许我们想对异常进行处理，然后允许它继续向上层函数传播，就好像它从未被捕获过一样？对于这些情况，我们不需要任何新的语法。可以堆叠`except`子句，只有第一个匹配的子句会被执行。对于第二个问题，`raise`关键字，不带参数，将重新引发最后一个异常，如果我们已经在异常处理程序中。观察以下代码：

```py
def funny_division3(anumber):
    try:
        if anumber == 13:
            raise ValueError("13 is an unlucky number")
        return 100 / anumber
    **except ZeroDivisionError:
        return "Enter a number other than zero"
    **except TypeError:
        return "Enter a numerical value"
    **except ValueError:
        print("No, No, not 13!")
        **raise

```

最后一行重新引发了`ValueError`，因此在输出`No, No, not 13!`后，它将再次引发异常；我们仍然会在控制台上获得原始的堆栈跟踪。

如果我们像前面的例子中那样堆叠异常子句，只有第一个匹配的子句会被执行，即使有多个子句符合条件。多个子句如何匹配呢？请记住，异常是对象，因此可以被子类化。正如我们将在下一节中看到的，大多数异常都扩展了`Exception`类（它本身是从`BaseException`派生的）。如果我们在捕获`TypeError`之前捕获`Exception`，那么只有`Exception`处理程序会被执行，因为`TypeError`是通过继承成为`Exception`的。

这在某些情况下非常有用，例如我们想要专门处理一些异常，然后将所有剩余的异常作为更一般的情况处理。在捕获所有特定异常后，我们只需捕获`Exception`并在那里处理一般情况。

有时，当我们捕获异常时，我们需要引用`Exception`对象本身。这最常发生在我们使用自定义参数定义自己的异常时，但也可能与标准异常相关。大多数异常类在其构造函数中接受一组参数，我们可能希望在异常处理程序中访问这些属性。如果我们定义自己的异常类，甚至可以在捕获时调用自定义方法。捕获异常作为变量的语法使用`as`关键字：

```py
try:
    raise ValueError("This is an argument")
except ValueError as e:
    print("The exception arguments were", e.args)
```

如果我们运行这个简单的片段，它会打印出我们在初始化`ValueError`时传递的字符串参数。

我们已经看到了处理异常的语法的几种变体，但我们仍然不知道如何执行代码，无论是否发生异常。我们也无法指定只有在没有发生异常时才执行的代码。另外两个关键字`finally`和`else`可以提供缺失的部分。两者都不需要额外的参数。以下示例随机选择一个要抛出的异常并引发它。然后运行一些不那么复杂的异常处理代码，演示了新引入的语法：

```py
import random
some_exceptions = [ValueError, TypeError, IndexError, None]

try:
    choice = random.choice(some_exceptions)
    print("raising {}".format(choice))
    if choice:
        raise choice("An error")
except ValueError:
    print("Caught a ValueError")
except TypeError:
    print("Caught a TypeError")
except Exception as e:
    print("Caught some other error: %s" %
        ( e.__class__.__name__))
else:
    print("This code called if there is no exception")
finally:
    print("This cleanup code is always called")
```

如果我们运行这个例子——它几乎涵盖了每种可能的异常处理场景——几次，每次都会得到不同的输出，这取决于`random`选择的异常。以下是一些示例运行：

```py
$ python finally_and_else.py
raising None
This code called if there is no exception
This cleanup code is always called

$ python finally_and_else.py
raising <class 'TypeError'>
Caught a TypeError
This cleanup code is always called

$ python finally_and_else.py
raising <class 'IndexError'>
Caught some other error: IndexError
This cleanup code is always called

$ python finally_and_else.py
raising <class 'ValueError'>
Caught a ValueError
This cleanup code is always called

```

请注意`finally`子句中的`print`语句无论发生什么都会被执行。当我们需要在代码运行结束后执行某些任务时（即使发生异常），这是非常有用的。一些常见的例子包括：

+   清理一个打开的数据库连接

+   关闭一个打开的文件

+   通过网络发送关闭握手

当我们在`try`子句内部执行`return`语句时，`finally`子句也非常重要。在返回值之前，`finally`处理程序仍将被执行。

还要注意当没有引发异常时的输出：`else`和`finally`子句都会被执行。`else`子句可能看起来多余，因为只有在没有引发异常时才应执行的代码可以直接放在整个`try`...`except`块之后。不同之处在于，如果捕获并处理了异常，`else`块仍将被执行。我们稍后在讨论使用异常作为流程控制时会更多地了解这一点。

在`try`块之后可以省略任何`except`、`else`和`finally`子句（尽管单独使用`else`是无效的）。如果包含多个子句，则`except`子句必须首先出现，然后是`else`子句，最后是`finally`子句。`except`子句的顺序通常从最具体到最一般。

## 异常层次结构

我们已经看到了几个最常见的内置异常，你可能会在你的常规 Python 开发过程中遇到其余的异常。正如我们之前注意到的，大多数异常都是`Exception`类的子类。但并非所有异常都是如此。`Exception`本身实际上是从一个叫做`BaseException`的类继承而来的。事实上，所有异常都必须扩展`BaseException`类或其子类之一。

有两个关键异常，`SystemExit`和`KeyboardInterrupt`，它们直接从`BaseException`而不是`Exception`派生。`SystemExit`异常在程序自然退出时引发，通常是因为我们在代码中的某个地方调用了`sys.exit`函数（例如，当用户选择退出菜单项，单击窗口上的“关闭”按钮，或输入命令关闭服务器时）。该异常旨在允许我们在程序最终退出之前清理代码，因此我们通常不需要显式处理它（因为清理代码发生在`finally`子句内）。

如果我们处理它，通常会重新引发异常，因为捕获它会阻止程序退出。当然，也有一些情况，我们可能希望阻止程序退出，例如，如果存在未保存的更改，我们希望提示用户是否真的要退出。通常，如果我们处理`SystemExit`，那是因为我们想对其进行特殊处理，或者直接预期它。我们特别不希望它在捕获所有正常异常的通用子句中被意外捕获。这就是为什么它直接从`BaseException`派生出来的原因。

`KeyboardInterrupt`异常在命令行程序中很常见。当用户使用依赖于操作系统的组合键（通常是*Ctrl* + *C*）明确中断程序执行时，它会被抛出。这是用户有意中断运行中程序的标准方式，与`SystemExit`一样，它几乎总是应该通过终止程序来响应。此外，与`SystemExit`一样，它应该在`finally`块内处理任何清理任务。

这是一个完全说明异常层次结构的类图：

![异常层次结构](img/8781OS_04_01.jpg)

当我们使用`except：`子句而没有指定任何类型的异常时，它将捕获`BaseException`的所有子类；也就是说，它将捕获所有异常，包括这两个特殊的异常。由于我们几乎总是希望这些得到特殊处理，因此不明智地使用没有参数的`except：`语句。如果要捕获除`SystemExit`和`KeyboardInterrupt`之外的所有异常，请明确捕获`Exception`。

此外，如果您确实想捕获所有异常，我建议使用语法`except BaseException：`而不是原始的`except：`。这有助于明确告诉您代码的未来读者，您有意处理特殊情况异常。

## 定义我们自己的异常

通常，当我们想要引发异常时，我们会发现没有内置的异常适合。幸运的是，我们可以轻松地定义自己的新异常。类的名称通常设计为传达出了什么问题，我们可以在初始化程序中提供任意参数以包含额外信息。

我们所要做的就是从`Exception`类继承。我们甚至不必向类添加任何内容！当然，我们可以直接扩展`BaseException`，但那样它将不会被通用的`except Exception`子句捕获。

这是我们可能在银行应用程序中使用的一个简单异常：

```py
class InvalidWithdrawal(Exception):
    pass

raise InvalidWithdrawal("You don't have $50 in your account")
```

最后一行说明了如何引发新定义的异常。我们能够将任意数量的参数传递到异常中。通常使用字符串消息，但可以存储任何在以后的异常处理程序中可能有用的对象。`Exception.__init__`方法设计为接受任何参数并将它们存储为名为`args`的属性中的元组。这使得异常更容易定义，而无需覆盖`__init__`。

当然，如果我们确实想要自定义初始化程序，我们可以自由地这样做。这是一个初始化程序接受当前余额和用户想要提取的金额的异常。此外，它添加了一个方法来计算请求超支了多少：

```py
class InvalidWithdrawal(Exception):
    **def __init__(self, balance, amount):
        **super().__init__("account doesn't have ${}".format(
            **amount))
        self.amount = amount
        self.balance = balance

    def overage(self):
        return self.amount - self.balance

raise InvalidWithdrawal(25, 50)

```

结尾处的`raise`语句说明了如何构造这个异常。正如你所看到的，我们可以对异常做任何其他对象可以做的事情。我们可以捕获异常并将其作为一个工作对象传递，尽管通常将工作对象的引用作为异常的属性包含在异常中，然后传递它。

如果出现`InvalidWithdrawal`异常，我们会这样处理：

```py
try:
    raise InvalidWithdrawal(25, 50)
except InvalidWithdrawal as e:
    print("I'm sorry, but your withdrawal is "
            "more than your balance by "
            "${}".format(e.overage()))
```

在这里，我们看到了`as`关键字的有效使用。按照惯例，大多数 Python 程序员将异常变量命名为`e`，尽管通常情况下，你可以自由地将其命名为`ex`、`exception`或者`aunt_sally`。

定义自己的异常有很多原因。通常情况下，向异常中添加信息或以某种方式记录异常是很有用的。但是，自定义异常的实用性真正体现在创建面向其他程序员访问的框架、库或 API 时。在这种情况下，要小心确保你的代码引发的异常对客户程序员是有意义的。它们应该易于处理，并清楚地描述发生了什么。客户程序员应该很容易看到如何修复错误（如果它反映了他们代码中的错误）或处理异常（如果这是他们需要知道的情况）。

异常并不是异常的。新手程序员倾向于认为异常只对异常情况有用。然而，异常情况的定义可能模糊不清，需要根据具体情况解释。考虑以下两个函数：

```py
def divide_with_exception(number, divisor):
    try:
        print("{} / {} = {}".format(
            number, divisor, number / divisor * 1.0))
    except ZeroDivisionError:
        print("You can't divide by zero")

def divide_with_if(number, divisor):
    if divisor == 0:
        print("You can't divide by zero")
    else:
        print("{} / {} = {}".format(
            number, divisor, number / divisor * 1.0))
```

这两个函数的行为是一样的。如果`divisor`是零，会打印一个错误消息；否则，会显示一个打印除法结果的消息。我们可以通过使用`if`语句来避免`ZeroDivisionError`被抛出。同样地，我们可以通过明确检查参数是否在列表范围内来避免`IndexError`，并且通过检查键是否在字典中来避免`KeyError`。

但我们不应该这样做。首先，我们可能会编写一个`if`语句，检查索引是否低于列表的参数，但忘记检查负值。

### 注意

记住，Python 列表支持负索引；`-1`指的是列表中的最后一个元素。

最终，我们会发现这一点，并且必须找到我们检查代码的所有地方。但如果我们简单地捕获了`IndexError`并处理它，我们的代码就能正常工作。

Python 程序员倾向于遵循“宁可请求原谅，而不是事先获得许可”的模式，也就是说，他们执行代码，然后处理任何出现的问题。相反，事先检查再执行的方法通常是不被赞同的。这样做的原因有几个，但主要原因是不应该需要消耗 CPU 周期来寻找在正常代码路径中不会出现的异常情况。因此，明智的做法是将异常用于异常情况，即使这些情况只是稍微异常。进一步地，我们实际上可以看到异常语法也对流程控制是有效的。与`if`语句一样，异常可以用于决策、分支和消息传递。

想象一个销售小部件和小工具的公司的库存应用程序。当客户购买时，商品可以是有货的，这种情况下商品将从库存中移除并返回剩余商品数量，或者可能是缺货。现在，缺货在库存应用程序中是一个完全正常的事情。这绝对不是一个特殊情况。但如果缺货了，我们应该返回什么呢？一个说缺货的字符串？一个负数？在这两种情况下，调用方法都必须检查返回值是正整数还是其他值，以确定是否缺货。这看起来有点混乱。相反，我们可以引发`OutOfStockException`并使用`try`语句来控制程序流程。有道理吗？此外，我们还要确保不会将同一商品卖给两个不同的客户，或者出售尚未备货的商品。促进这一点的一种方法是锁定每种商品，以确保只有一个人可以同时更新它。用户必须锁定商品，操作商品（购买、补充库存、计算剩余商品数量...），然后解锁商品。以下是一个带有描述部分方法应该做什么的文档字符串的不完整的`Inventory`示例：

```py
class Inventory:
    def lock(self, item_type):
        '''Select the type of item that is going to
        be manipulated. This method will lock the
        item so nobody else can manipulate the
        inventory until it's returned. This prevents
        selling the same item to two different
        customers.'''
        pass

    def unlock(self, item_type):
        '''Release the given type so that other
        customers can access it.'''
        pass

    def purchase(self, item_type):
        '''If the item is not locked, raise an
        exception. If the item_type  does not exist,
        raise an exception. If the item is currently
        out of stock, raise an exception. If the item
        is available, subtract one item and return
        the number of items left.'''
        pass
```

我们可以将这个对象原型交给开发人员，让他们实现方法，以确保它们按照预期工作，而我们则可以在需要进行购买的代码上进行工作。我们将使用 Python 强大的异常处理来考虑不同的分支，具体取决于购买方式：

```py
item_type = 'widget'
inv = Inventory()
inv.lock(item_type)
try:
    num_left = inv.purchase(item_type)
except InvalidItemType:
    print("Sorry, we don't sell {}".format(item_type))
except OutOfStock:
    print("Sorry, that item is out of stock.")
else:
    print("Purchase complete. There are "
            "{} {}s left".format(num_left, item_type))
finally:
    inv.unlock(item_type)
```

注意所有可能的异常处理子句是如何被用来确保在正确的时间发生正确的操作。即使`OutOfStock`并不是一个非常特殊的情况，我们仍然可以使用异常来适当处理它。这段代码也可以用`if`...`elif`...`else`结构来编写，但这样不容易阅读或维护。

我们还可以使用异常来在不同的方法之间传递消息。例如，如果我们想要告知客户商品预计何时会再次备货，我们可以确保我们的`OutOfStock`对象在构造时需要一个`back_in_stock`参数。然后，当我们处理异常时，我们可以检查该值并向客户提供额外信息。附加到对象的信息可以很容易地在程序的两个不同部分之间传递。异常甚至可以提供一个方法，指示库存对象重新订购或预订商品。

使用异常来控制流程可以设计出一些方便的程序。从这次讨论中重要的是要明白异常并不是我们应该尽量避免的坏事。发生异常并不意味着你应该阻止这种特殊情况的发生。相反，这只是一种在两个可能不直接调用彼此的代码部分之间传递信息的强大方式。

# 案例研究

我们一直在以相当低的细节水平来看待异常的使用和处理——语法和定义。这个案例研究将帮助我们将它与之前的章节联系起来，这样我们就可以看到异常在对象、继承和模块的更大背景下是如何使用的。

今天，我们将设计一个简单的中央认证和授权系统。整个系统将放置在一个模块中，其他代码将能够查询该模块对象以进行认证和授权。我们应该承认，从一开始，我们并不是安全专家，我们设计的系统可能存在安全漏洞。我们的目的是研究异常，而不是保护系统。然而，对于其他代码可以与之交互的基本登录和权限系统来说，这是足够的。稍后，如果其他代码需要更安全，我们可以让安全或密码专家审查或重写我们的模块，最好不改变 API。

认证是确保用户确实是他们所说的人的过程。我们将遵循当今常见的网络系统的做法，使用用户名和私人密码组合。其他的认证方法包括语音识别、指纹或视网膜扫描仪以及身份证。

另一方面，授权是确定给定（经过认证的）用户是否被允许执行特定操作的全部内容。我们将创建一个基本的权限列表系统，用于存储允许执行每个操作的特定人员的列表。

此外，我们将添加一些管理功能，以允许新用户被添加到系统中。为简洁起见，我们将略去添加密码或在添加后更改权限的编辑，但这些（非常必要的）功能当然可以在将来添加。

这是一个简单的分析；现在让我们继续设计。显然，我们需要一个`User`类来存储用户名和加密密码。这个类还将允许用户通过检查提供的密码是否有效来登录。我们可能不需要一个`Permission`类，因为可以将它们作为字符串映射到使用字典的用户列表中。我们应该有一个中央的`Authenticator`类，用于处理用户管理和登录或注销。谜题的最后一块是一个`Authorizor`类，用于处理权限和检查用户是否可以执行某项活动。我们将在`auth`模块中提供这些类的单个实例，以便其他模块可以使用这个中央机制来满足其所有的认证和授权需求。当然，如果他们想要为非中央授权活动实例化私有实例，他们是可以自由这样做的。

随着我们的进行，我们还将定义几个异常。我们将从一个特殊的`AuthException`基类开始，它接受`username`和可选的`user`对象作为参数；我们大部分自定义的异常都将继承自这个基类。

让我们首先构建`User`类；这似乎足够简单。一个新用户可以用用户名和密码初始化。密码将被加密存储，以减少被盗的可能性。我们还需要一个`check_password`方法来测试提供的密码是否正确。以下是完整的类：

```py
import hashlib

class User:
    def __init__(self, username, password):
        '''Create a new user object. The password
        will be encrypted before storing.'''
        self.username = username
        self.password = self._encrypt_pw(password)
        self.is_logged_in = False

    def _encrypt_pw(self, password):
        '''Encrypt the password with the username and return
        the sha digest.'''
        hash_string = (self.username + password)
        hash_string = hash_string.encode("utf8")
        return hashlib.sha256(hash_string).hexdigest()

    def check_password(self, password):
        '''Return True if the password is valid for this
        user, false otherwise.'''
        encrypted = self._encrypt_pw(password)
        return encrypted == self.password
```

由于加密密码的代码在`__init__`和`check_password`中都需要，我们将其提取到自己的方法中。这样，如果有人意识到它不安全并需要改进，它只需要在一个地方进行更改。这个类可以很容易地扩展到包括强制或可选的个人详细信息，比如姓名、联系信息和出生日期。

在我们编写代码添加用户之前（这将发生在尚未定义的`Authenticator`类中），我们应该检查一些用例。如果一切顺利，我们可以添加一个带有用户名和密码的用户；`User`对象被创建并插入到字典中。但是，有哪些情况可能不顺利呢？显然，我们不希望添加一个已经存在于字典中的用户名的用户。如果这样做，我们将覆盖现有用户的数据，新用户可能会访问该用户的权限。因此，我们需要一个`UsernameAlreadyExists`异常。另外，出于安全考虑，如果密码太短，我们也应该引发异常。这两个异常都将扩展`AuthException`，这是我们之前提到的。因此，在编写`Authenticator`类之前，让我们定义这三个异常类：

```py
class AuthException(Exception):
    def __init__(self, username, user=None):
        super().__init__(username, user)
        self.username = username
        self.user = user

class UsernameAlreadyExists(AuthException):
    pass

class PasswordTooShort(AuthException):
    pass
```

`AuthException`需要一个用户名，并具有一个可选的用户参数。第二个参数应该是与该用户名关联的`User`类的实例。我们正在定义的这两个具体异常只需要通知调用类发生异常情况，因此我们不需要为它们添加任何额外的方法。

现在让我们开始`Authenticator`类。它可以简单地将用户名映射到用户对象，因此我们将从初始化函数中的字典开始。添加用户的方法需要在创建新的`User`实例并将其添加到字典之前检查两个条件（密码长度和先前存在的用户）：

```py
class Authenticator:
    def __init__(self):
        '''Construct an authenticator to manage
        users logging in and out.'''
        self.users = {}

    def add_user(self, username, password):
        if username in self.users:
            **raise UsernameAlreadyExists(username)
        if len(password) < 6:
            **raise PasswordTooShort(username)
        **self.users[username] = User(username, password)

```

当然，如果需要，我们可以扩展密码验证以引发其他方式太容易破解的密码的异常。现在让我们准备`login`方法。如果我们刚才没有考虑异常，我们可能只希望该方法根据登录是否成功返回`True`或`False`。但我们正在考虑异常，这可能是一个不太异常的情况使用它们的好地方。我们可以引发不同的异常，例如，如果用户名不存在或密码不匹配。这将允许尝试登录用户的任何人使用`try`/`except`/`else`子句优雅地处理情况。因此，首先我们添加这些新异常：

```py
class InvalidUsername(AuthException):
    pass

class InvalidPassword(AuthException):
    pass
```

然后我们可以为我们的`Authenticator`类定义一个简单的`login`方法，如果有必要，引发这些异常。如果不需要，它会标记`user`已登录并返回：

```py
    def login(self, username, password):
        try:
            user = self.users[username]
        **except KeyError:
            **raise InvalidUsername(username)

        if not user.check_password(password):
            **raise InvalidPassword(username, user)

        user.is_logged_in = True
        return True
```

注意`KeyError`是如何处理的。这可以使用`if username not in self.users:`来处理，但我们选择直接处理异常。我们最终吞掉了这个第一个异常，并引发了一个更适合用户界面 API 的全新异常。

我们还可以添加一个方法来检查特定用户名是否已登录。在这里决定是否使用异常更加棘手。如果用户名不存在，我们应该引发异常吗？如果用户没有登录，我们应该引发异常吗？

要回答这些问题，我们需要考虑方法如何被访问。最常见的情况是，这个方法将用于回答“我应该允许他们访问*<something>*吗？”这个问题的答案要么是“是的，用户名有效并且他们已登录”，要么是“不，用户名无效或者他们未登录”。因此，布尔返回值就足够了。在这里没有必要使用异常，只是为了使用异常。

```py
    def is_logged_in(self, username):
        if username in self.users:
            return self.users[username].is_logged_in
        return False
```

最后，我们可以向我们的模块添加一个默认的认证器实例，以便客户端代码可以轻松地使用`auth.authenticator`进行访问：

```py
authenticator = Authenticator()
```

这行代码放在模块级别，不在任何类定义之外，因此可以将 auth.authenticator 作为`auth.authenticator`进行访问。现在我们可以开始`Authorizor`类，它将权限映射到用户。`Authorizor`类不应允许用户访问权限，如果他们没有登录，因此他们将需要引用特定的认证器。我们还需要在初始化时设置权限字典：

```py
class Authorizor:
    def __init__(self, authenticator):
        self.authenticator = authenticator
        self.permissions = {}
```

现在我们可以编写方法来添加新的权限，并设置哪些用户与每个权限相关联：

```py
    def add_permission(self, perm_name):
        '''Create a new permission that users
        can be added to'''
        try:
            perm_set = self.permissions[perm_name]
        except KeyError:
            self.permissions[perm_name] = set()
        else:
            raise PermissionError("Permission Exists")

    def permit_user(self, perm_name, username):
        '''Grant the given permission to the user'''
        try:
            perm_set = self.permissions[perm_name]
        except KeyError:
            raise PermissionError("Permission does not exist")
        else:
            if username not in self.authenticator.users:
                raise InvalidUsername(username)
            perm_set.add(username)
```

第一个方法允许我们创建新的权限，除非它已经存在，否则会引发异常。第二个方法允许我们将用户名添加到权限中，除非权限或用户名尚不存在。

我们使用`set`而不是`list`来存储用户名，这样即使您多次授予用户权限，集合的性质意味着用户只会出现一次。我们将在后面的章节中进一步讨论集合。

这两种方法都会引发`PermissionError`。这个新错误不需要用户名，所以我们将它直接扩展为`Exception`，而不是我们自定义的`AuthException`：

```py
class PermissionError(Exception):
    pass
```

最后，我们可以添加一个方法来检查用户是否具有特定的`permission`。为了获得访问权限，他们必须同时登录到认证器并且在被授予该特权的人员集合中。如果这些条件中的任何一个不满足，就会引发异常：

```py
    def check_permission(self, perm_name, username):
        if not self.authenticator.is_logged_in(username):
            raise NotLoggedInError(username)
        try:
            perm_set = self.permissions[perm_name]
        except KeyError:
            raise PermissionError("Permission does not exist")
        else:
            if username not in perm_set:
                raise NotPermittedError(username)
            else:
                return True
```

这里有两个新的异常；它们都使用用户名，所以我们将它们定义为`AuthException`的子类：

```py
class NotLoggedInError(AuthException):
    pass

class NotPermittedError(AuthException):
    pass
```

最后，我们可以添加一个默认的`authorizor`来配合我们的默认验证器：

```py
authorizor = Authorizor(authenticator)
```

这完成了一个基本的身份验证/授权系统。我们可以在 Python 提示符下测试系统，检查用户`joe`是否被允许在油漆部门执行任务：

```py
>>> import auth
>>> auth.authenticator.add_user("joe", "joepassword")
>>> auth.authorizor.add_permission("paint")
>>> auth.authorizor.check_permission("paint", "joe")
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
 **File "auth.py", line 109, in check_permission
 **raise NotLoggedInError(username)
auth.NotLoggedInError: joe
>>> auth.authenticator.is_logged_in("joe")
False
>>> auth.authenticator.login("joe", "joepassword")
True
>>> auth.authorizor.check_permission("paint", "joe")
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
 **File "auth.py", line 116, in check_permission
 **raise NotPermittedError(username)
auth.NotPermittedError: joe
>>> auth.authorizor.check_permission("mix", "joe")
Traceback (most recent call last):
 **File "auth.py", line 111, in check_permission
 **perm_set = self.permissions[perm_name]
KeyError: 'mix'

During handling of the above exception, another exception occurred:
Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
 **File "auth.py", line 113, in check_permission
 **raise PermissionError("Permission does not exist")
auth.PermissionError: Permission does not exist
>>> auth.authorizor.permit_user("mix", "joe")
Traceback (most recent call last):
 **File "auth.py", line 99, in permit_user
 **perm_set = self.permissions[perm_name]
KeyError: 'mix'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
 **File "<stdin>", line 1, in <module>
 **File "auth.py", line 101, in permit_user
 **raise PermissionError("Permission does not exist")
auth.PermissionError: Permission does not exist
>>> auth.authorizor.permit_user("paint", "joe")
>>> auth.authorizor.check_permission("paint", "joe")
True

```

虽然冗长，但前面的输出显示了我们所有的代码和大部分异常的运行情况，但要真正理解我们定义的 API，我们应该编写一些实际使用它的异常处理代码。这是一个基本的菜单界面，允许某些用户更改或测试程序：

```py
import auth

# Set up a test user and permission
auth.authenticator.add_user("joe", "joepassword")
auth.authorizor.add_permission("test program")
auth.authorizor.add_permission("change program")
auth.authorizor.permit_user("test program", "joe")

class Editor:
    def __init__(self):
        self.username = None
        self.menu_map = {
                "login": self.login,
                "test": self.test,
                "change": self.change,
                "quit": self.quit
           }

    def login(self):
        logged_in = False
        while not logged_in:
            username = input("username: ")
            password = input("password: ")
            try:
                logged_in = auth.authenticator.login(
                        username, password)
            except auth.InvalidUsername:
                print("Sorry, that username does not exist")
            except auth.InvalidPassword:
                print("Sorry, incorrect password")
            else:
                self.username = username
    def is_permitted(self, permission):
        try:
            auth.authorizor.check_permission(
                permission, self.username)
        except auth.NotLoggedInError as e:
            print("{} is not logged in".format(e.username))
            return False
        except auth.NotPermittedError as e:
            print("{} cannot {}".format(
                e.username, permission))
            return False
        else:
            return True

    def test(self):
        if self.is_permitted("test program"):
            print("Testing program now...")

    def change(self):
        if self.is_permitted("change program"):
            print("Changing program now...")

    def quit(self):
        raise SystemExit()

    def menu(self):
        try:
            answer = ""
            while True:
                print("""
Please enter a command:
\tlogin\tLogin
\ttest\tTest the program
\tchange\tChange the program
\tquit\tQuit
""")
                answer = input("enter a command: ").lower()
                try:
                    func = self.menu_map[answer]
                except KeyError:
                    print("{} is not a valid option".format(
                        answer))
                else:
                    func()
        finally:
            print("Thank you for testing the auth module")

Editor().menu()
```

这个相当长的例子在概念上非常简单。`is_permitted`方法可能是最有趣的；这是一个大部分是内部方法，由`test`和`change`调用，以确保用户在继续之前被允许访问。当然，这两种方法都是存根，但我们这里不是在编写编辑器；我们是通过测试身份验证和授权框架来说明异常和异常处理程序的使用！

# 练习

如果您以前从未处理过异常，首先要做的是查看您编写的任何旧 Python 代码，并注意是否有应该处理异常的地方。您会如何处理它们？您是否需要处理它们？有时，让异常传播到控制台是向用户传达信息的最佳方式，特别是如果用户也是脚本的编码者。有时，您可以从错误中恢复并允许程序继续。有时，您只能将错误重新格式化为用户可以理解的内容，并向他们显示。

一些常见的查找地方包括文件 I/O（您的代码是否可能尝试读取不存在的文件？）、数学表达式（您要除以的值是否可能为零？）、列表索引（列表是否为空？）和字典（键是否存在？）。问问自己是否应该忽略问题，通过先检查值来处理它，还是通过异常来处理它。特别注意可能使用`finally`和`else`的地方，以确保在所有条件下执行正确的代码。

现在写一些新代码。想想一个需要身份验证和授权的程序，并尝试编写一些使用我们在案例研究中构建的`auth`模块的代码。如果模块不够灵活，可以随意修改模块。尝试以明智的方式处理所有异常。如果您在想出需要身份验证的内容方面遇到困难，可以尝试在第二章中的记事本示例中添加授权，或者在`auth`模块本身中添加授权-如果任何人都可以开始添加权限，那么这个模块就不是一个非常有用的模块！也许在允许添加或更改权限之前，需要管理员用户名和密码。

最后，尝试考虑代码中可能引发异常的地方。可以是您编写或正在处理的代码；或者您可以编写一个新项目作为练习。您可能最容易设计一个小型框架或 API，供其他人使用；异常是您的代码和其他人之间的绝妙沟通工具。请记住，设计和记录任何自行引发的异常作为 API 的一部分，否则他们将不知道如何处理这些异常！

# 总结

在本章中，我们深入讨论了引发、处理、定义和操作异常的细节。异常是一种强大的方式，可以在不要求调用函数显式检查返回值的情况下传达异常情况或错误条件。有许多内置异常，引发它们非常容易。处理不同异常事件的语法有几种不同的形式。

在下一章中，我们将讨论到目前为止所学的一切，以及如何在 Python 应用程序中最好地应用面向对象编程原则和结构。
