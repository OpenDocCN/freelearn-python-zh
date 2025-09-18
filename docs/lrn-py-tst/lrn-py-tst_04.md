# 第四章。使用 unittest.mock 解耦单元

在过去几章中，有好几次在面临将测试相互隔离的问题时，我告诉你们只需将问题记在心里，并说我们会在本章中处理它。终于，是时候真正解决这个问题了。

不依赖于其他函数、方法或数据的行为的函数和方法很少见；常见的情况是它们会调用其他函数或方法多次，并至少实例化一个类。这些调用和实例化中的每一个都会破坏单元的隔离；或者，如果你更喜欢这样想，它将更多的代码纳入了隔离部分。

无论你怎么看它——是作为隔离破坏者还是作为扩展隔离部分——这都是你想要有能力防止的事情。模拟对象通过取代外部函数或对象让你能够做到这一点。

使用`unittest.mock`包，你可以轻松执行以下操作：

+   用我们第三章中用到的`time.time`一样，替换你自己的代码或外部包中的函数和对象。

+   控制替换对象的行为。你可以控制它们提供的返回值，是否抛出异常，甚至是否调用其他函数，或创建其他对象的实例。

+   检查替换对象是否按预期使用：函数或方法是否被正确次数地调用，调用是否按正确的顺序发生，以及传递的参数是否正确。

# 一般的模拟对象

好吧，在我们深入探讨`unittest.mock`的细节之前，让我们花几分钟时间来谈谈模拟对象的整体情况。

从广义上讲，模拟对象是你可以在测试代码中使用作为替代品的对象，以防止测试重叠，并确保被测试的代码不会渗透到错误的测试中。因此，我们来自第三章的假`time.time`，*使用 doctest 进行单元测试*，就是一个模拟对象。然而，就像编程中的大多数事情一样，当它被正式化为一个设计良好的库，你可以在需要时调用它时，这个想法会更好。大多数编程语言都有许多这样的库可用。

随着时间的推移，模拟对象库的作者已经为模拟对象开发了两种主要的设计模式：在一种模式中，你可以创建一个模拟对象，并对其执行所有预期的操作。对象记录这些操作，然后你将对象放入回放模式，并将其传递给你的代码。如果你的代码未能复制预期的操作，模拟对象会报告失败。

在第二种模式中，你可以创建一个模拟对象，进行必要的最小配置以允许它模仿它所替代的真实对象，并将其传递给代码。它会记录代码如何使用它，然后你可以在事后执行断言来检查代码是否按预期使用对象。

在使用它编写的测试方面，第二种模式在功能上略胜一筹，但总体上，两种模式都工作得很好。

# 根据 unittest.mock 的模拟对象

Python 有几个模拟对象库；然而，截至 Python 3.3，其中一个已经成为标准库的成员。自然地，我们将关注这个库。这个库当然是`unittest.mock`。

`unittest.mock`库是第二种类型，一种记录实际使用情况然后断言的库。该库包含几种不同的模拟对象，它们共同让你可以模拟 Python 中几乎任何存在的东西。此外，该库还包含几个有用的辅助工具，简化了与模拟对象相关的各种任务，例如临时用模拟对象替换真实对象。

## 标准模拟对象

`unittest.mock`的基本元素是`unittest.mock.Mock`类。即使没有进行任何配置，`Mock`实例也能很好地模仿其他对象、方法或函数。

### 注意

对于 Python，有许多模拟对象库；严格来说，“模拟对象”这个短语可能意味着由这些库中的任何一个创建的对象。从现在起，在这本书中，你可以假设“模拟对象”是`unittest.mock.Mock`或其子类的实例。

模拟对象能够通过一个巧妙且有些递归的技巧来完成这种模仿。当你访问模拟对象的未知属性时，而不是抛出`AttributeError`异常，模拟对象会创建一个子模拟对象并返回它。由于模拟对象擅长模仿其他对象，返回模拟对象而不是实际值在常见情况下是有效的。

同样，模拟对象是可调用的；当你将模拟对象作为函数或方法调用时，它会记录调用参数，然后默认返回一个子模拟对象。

一个子模拟对象是一个独立的模拟对象，但它知道它与它所来的模拟对象——它的父对象相连。你对子对象所做的任何操作也会记录在父对象的记忆中。当需要检查模拟对象是否被正确使用时，你可以使用父对象来检查其所有后代。

示例：在交互式外壳中玩转模拟对象（亲自试试！）

```py
$ python3.4
Python 3.4.0 (default, Apr  2 2014, 08:10:08)
[GCC 4.8.2] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from unittest.mock import Mock, call
>>> mock = Mock()
>>> mock.x
<Mock name='mock.x' id='140145643647832'>
>>> mock.x
<Mock name='mock.x' id='140145643647832'>
>>> mock.x('Foo', 3, 14)
<Mock name='mock.x()' id='140145643690640'>
>>> mock.x('Foo', 3, 14)
<Mock name='mock.x()' id='140145643690640'>
>>> mock.x('Foo', 99, 12)
<Mock name='mock.x()' id='140145643690640'>
>>> mock.y(mock.x('Foo', 1, 1))
<Mock name='mock.y()' id='140145643534320'>
>>> mock.method_calls
[call.x('Foo', 3, 14),
 call.x('Foo', 3, 14),
 call.x('Foo', 99, 12),
 call.x('Foo', 1, 1),
 call.y(<Mock name='mock.x()' id='140145643690640'>)]
>>> mock.assert_has_calls([call.x('Foo', 1, 1)])
>>> mock.assert_has_calls([call.x('Foo', 1, 1), call.x('Foo', 99, 12)])
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "/usr/lib64/python3.4/unittest/mock.py", line 792, in assert_has_calls
 ) from cause
AssertionError: Calls not found.
Expected: [call.x('Foo', 1, 1), call.x('Foo', 99, 12)]
Actual: [call.x('Foo', 3, 14),
 call.x('Foo', 3, 14),
 call.x('Foo', 99, 12),
 call.x('Foo', 1, 1),
 call.y(<Mock name='mock.x()' id='140145643690640'>)]
>>> mock.assert_has_calls([call.x('Foo', 1, 1),...                        call.x('Foo', 99, 12)], any_order = True)
>>> mock.assert_has_calls([call.y(mock.x.return_value)])
>>>
```

在这个交互式会话中展示了几个重要的事情。

首先，注意每次我们访问`mock.x`时都返回了相同的模拟对象。这始终成立：如果你访问模拟对象的相同属性，你将得到相同的模拟对象作为结果。

下一个需要注意的事情可能看起来更令人惊讶。无论何时调用模拟对象，你都会得到相同的模拟对象作为返回值。返回的模拟对象不是为每次调用而创建的，也不是为每个参数组合而唯一的。我们很快就会看到如何覆盖返回值，但默认情况下，每次调用模拟对象时，你都会得到相同的模拟对象。你可以使用 `return_value` 属性名来访问这个模拟对象，正如你可能从示例的最后一句中注意到的。

`unittest.mock` 包包含一个 `call` 对象，它有助于更容易地检查是否已经进行了正确的调用。`call` 对象是可调用的，并以类似于模拟对象的方式记录其参数，这使得它很容易与模拟对象的调用历史进行比较。然而，`call` 对象真正闪耀的时候是你必须检查对派生模拟对象的调用。正如你可以在前面的例子中看到，`call('Foo', 1, 1)` 将匹配对父模拟对象的调用，但如果调用使用了这些参数，`call.x('Foo', 1, 1)`，它将匹配对名为 `x` 的子模拟对象的调用。你可以构建一个长长的查找和调用链。例如：

```py
>>> mock.z.hello(23).stuff.howdy('a', 'b', 'c')
<Mock name='mock.z.hello().stuff.howdy()' id='140145643535328'>
>>> mock.assert_has_calls([
...     call.z.hello().stuff.howdy('a', 'b', 'c')
... ])
>>>
```

注意，原始调用包括了 `hello(23)`，但调用规范只是简单地将其写成 `hello()`。每个调用规范只关心最终被调用对象的参数。中间调用的参数不被考虑。这没关系，因为它们总是产生相同的返回值，除非你覆盖了这种行为，在这种情况下，它们可能根本不会产生模拟对象。

### 注意

你可能之前没有遇到过断言。断言只有一个任务，而且只有一个任务：如果某事不是预期的，它会引发异常。特别是 `assert_has_calls` 方法，如果模拟对象的历史记录不包括指定的调用，则会引发异常。在我们的例子中，调用历史记录是一致的，所以断言方法没有做任何明显的操作。

尽管如此，你可以检查中间调用是否使用了正确的参数，因为模拟对象在记录对 `mock.z.hello(23)` 的调用之前立即记录了对 `mock.z.hello().stuff.howdy('a', 'b', 'c')` 的调用：

```py
>>> mock.mock_calls.index(call.z.hello(23))
6
>>> mock.mock_calls.index(call.z.hello().stuff.howdy('a', 'b', 'c'))
7
```

这也指出了所有模拟对象都携带的 `mock_calls` 属性。如果各种断言函数对你来说还不够用，你总是可以编写自己的函数来检查 `mock_calls` 列表，并验证事情是否如预期那样。我们很快就会讨论模拟对象断言方法。

### 非模拟属性

如果你希望模拟对象在查找属性时返回的不仅仅是子模拟对象，怎么办？这很简单；只需将值分配给该属性：

```py
>>> mock.q = 5
>>> mock.q
5
```

另有一个常见的情况是模拟对象的默认行为是错误的：如果访问特定属性应该引发一个`AttributeError`怎么办？幸运的是，这也很简单：

```py
>>> del mock.w
>>> mock.w
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "/usr/lib64/python3.4/unittest/mock.py", line 563, in __getattr__
 raise AttributeError(name)
AttributeError: w
```

### 非模拟返回值和引发异常

有时，实际上相当频繁，您会希望模拟对象扮演函数或方法的角色，返回特定的值或一系列特定的值，而不是返回另一个模拟对象。

要使模拟对象始终返回相同的值，只需更改`return_value`属性：

```py
>>> mock.o.return_value = 'Hi'
>>> mock.o()
'Hi'
>>> mock.o('Howdy')
'Hi'
```

如果您希望模拟对象在每次调用时返回不同的值，您需要将一个返回值序列分配给`side_effect`属性，如下所示：

```py
>>> mock.p.side_effect = [1, 2, 3]
>>> mock.p()
1
>>> mock.p()
2
>>> mock.p()
3
>>> mock.p()
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "/usr/lib64/python3.4/unittest/mock.py", line 885, in __call__
 return _mock_self._mock_call(*args, **kwargs)
 File "/usr/lib64/python3.4/unittest/mock.py", line 944, in _mock_call
 result = next(effect)
StopIteration
```

如果您不希望模拟对象引发`StopIteration`异常，您需要确保为测试中的所有调用提供足够的返回值。如果您不知道它将被调用多少次，一个无限迭代器，如`itertools.count`可能就是您需要的。这很容易做到：

```py
>>> mock.p.side_effect = itertools.count()
```

如果您希望模拟在返回值而不是引发异常，只需将异常对象分配给`side_effect`，或者将其放入分配给`side_effect`的迭代器中：

```py
>>> mock.e.side_effect = [1, ValueError('x')]
>>> mock.e()
1
>>> mock.e()
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "/usr/lib64/python3.4/unittest/mock.py", line 885, in __call__
 return _mock_self._mock_call(*args, **kwargs)
 File "/usr/lib64/python3.4/unittest/mock.py", line 946, in _mock_call
 raise result
ValueError: x
```

`side_effect`属性还有另一个用途，我们将在后面讨论。

### 模拟类或函数的细节

有时，模拟对象的一般行为并不能足够接近被替换对象的模拟。这尤其适用于当它们在不当使用时需要引发异常的情况，因为模拟对象通常乐于接受任何使用方式。

`unittest.mock`包使用一种称为**speccing**的技术来解决此问题。如果你将一个对象传递给`unittest.mock.create_autospec`，返回的值将是一个模拟对象，但它会尽力假装它就是传递给`create_autospec`的那个对象。这意味着它将：

+   如果您尝试访问原始对象没有的属性，将引发一个`AttributeError`，除非您首先明确地为该属性分配一个值。

+   如果您尝试在原始对象不可调用时调用模拟对象，将引发一个`TypeError`。

+   如果传递了错误的参数数量或传递了在原始对象可调用时不合理的关键字参数，将引发一个`TypeError`。

+   欺骗`isinstance`认为模拟对象是原始对象类型的

由`create_autospec`创建的模拟对象与其所有子对象也共享这一特性，这通常是您想要的。如果您确实只想让特定的模拟对象被 specced，而其子对象不是，您可以使用`spec`关键字将模板对象传递给`Mock`构造函数。

这里是一个使用`create_autospec`的简短演示：

```py
>>> from unittest.mock import create_autospec
>>> x = Exception('Bad', 'Wolf')
>>> y = create_autospec(x)
>>> isinstance(y, Exception)
True
>>> y
<NonCallableMagicMock spec='Exception' id='140440961099088'>
```

### 模拟函数或方法副作用

有时，为了使模拟对象能够成功地取代一个函数或方法，模拟对象实际上必须调用其他函数，设置变量值，或者一般地执行函数可以做的任何事情。

这种需求不如你想象的那么常见，而且对于测试目的来说也有些危险，因为当你的模拟对象可以执行任意代码时，它们可能会停止成为强制测试隔离的简化工具，反而成为问题的一个复杂部分。

话虽如此，仍然有需要模拟函数执行比简单地返回值更复杂操作的情况，我们可以使用模拟对象的 `side_effect` 属性来实现这一点。我们之前已经见过 `side_effect`，当时我们给它分配了一个返回值序列。

如果你将一个可调用对象分配给 `side_effect`，当模拟对象被调用并传递相同的参数时，这个可调用对象将被调用。如果 `side_effect` 函数引发异常，模拟对象也会这样做；否则，模拟对象会返回 `side_effect` 的返回值。

换句话说，如果你将一个函数分配给模拟对象的 `side_effect` 属性，这个模拟对象实际上就变成了那个函数，唯一的区别是模拟对象仍然记录了它的使用细节。

`side_effect` 函数中的代码应该是最小的，并且不应该尝试实际执行模拟对象所替代的代码的工作。它应该做的只是执行任何预期的外部可见操作，然后返回预期的 `result.Mock` 对象断言方法。

正如我们在 *标准模拟对象* 部分所看到的，你总是可以编写代码来检查模拟对象的 `mock_calls` 属性，以查看事物是否按预期运行。然而，已经为你编写了一些特别常见的检查，这些检查作为模拟对象的断言方法提供。对于断言来说，这些断言方法在通过时返回 `None`，在失败时引发 `AssertionError`。

`assert_called_with` 方法接受任意集合的参数和关键字参数，除非这些参数在上次调用模拟对象时被传递，否则会引发 `AssertionError`。

`assert_called_once_with` 方法的表现类似于 `assert_called_with`，除了它还会检查模拟对象是否只被调用了一次。如果这不是真的，则会引发 `AssertionError`。

`assert_any_call` 方法接受任意参数和关键字参数，如果模拟对象从未使用这些参数被调用，则会引发 `AssertionError`。

我们已经看到了`assert_has_calls`方法。此方法接受一个调用对象列表，检查它们是否以相同的顺序出现在历史记录中，如果不出现，则引发异常。请注意，“以相同的顺序”并不一定意味着“相邻”。只要列出的调用都按正确的顺序出现，列表之间可以有其他调用。如果你将`any_order`参数设置为 true 值，则此行为会改变。在这种情况下，`assert_has_calls`不会关心调用的顺序，只检查它们是否都出现在历史记录中。

`assert_not_called`方法如果模拟对象曾被调用，将引发异常。

## 模拟具有特殊行为的容器和对象

`Mock`类没有处理的是 Python 特殊语法构造背后的所谓魔法方法：`__getitem__`、`__add__`等等。如果你需要你的模拟对象记录并响应魔法方法——换句话说，如果你想它们假装是字典或列表等容器对象，或者响应数学运算符，或者作为上下文管理器或任何其他将语法糖转换为方法调用的东西——你将使用`unittest.mock.MagicMock`来创建你的模拟对象。

由于它们（和模拟对象）的工作细节，有一些魔法方法甚至不被`MagicMock`支持：`__getattr__`、`__setattr__`、`__init__`、`__new__`、`__prepare__`、`__instancecheck__`、`__subclasscheck__`和`__del__`。

这里有一个简单的例子，我们使用`MagicMock`创建一个支持`in`运算符的模拟对象：

```py
>>> from unittest.mock import MagicMock
>>> mock = MagicMock()
>>> 7 in mock
False
>>> mock.mock_calls
[call.__contains__(7)]
>>> mock.__contains__.return_value = True
>>> 8 in mock
True
>>> mock.mock_calls
[call.__contains__(7), call.__contains__(8)]
```

其他魔法方法的工作方式也类似。例如，加法：

```py
>>> mock + 5
<MagicMock name='mock.__add__()' id='140017311217816'>
>>> mock.mock_calls
[call.__contains__(7), call.__contains__(8), call.__add__(5)]
```

注意，加法操作的返回值是一个模拟对象，它是原始模拟对象的子对象，但`in`运算符返回了一个布尔值。Python 确保某些魔法方法返回特定类型的价值，如果不符合该要求，将引发异常。在这些情况下，`MagicMock`的方法实现返回一个最佳猜测的正确类型值，而不是子模拟对象。

在使用原地数学运算符（如`+=`（`__iadd__`）和`|=`（`__ior__`））时，你需要小心的一点是，`MagicMock`处理它们的方式有些奇怪。它所做的仍然是有用的，但它可能会让你感到意外：

```py
>>> mock += 10
>>> mock.mock_calls
[]
```

那是什么？它删除了我们的通话记录吗？幸运的是，没有。它所做的只是将通过加法操作创建的子模拟对象分配给名为 mock 的变量。这完全符合原地数学运算符应有的工作方式。不幸的是，它仍然使我们失去了访问通话记录的能力，因为我们不再有一个指向父模拟对象的变量的引用。

### 小贴士

如果你打算检查就地数学运算符，请确保将父模拟对象放在一个不会被重新分配的变量中。此外，你应该确保你的模拟就地运算符返回操作的结果，即使这意味着`return self.return_value`，否则 Python 将把`None`分配给左边的变量。

在地运算符还有另一个你应该记住的详细工作方式：

```py
>>> mock = MagicMock()
>>> x = mock
>>> x += 5
>>> x
<MagicMock name='mock.__iadd__()' id='139845830142216'>
>>> x += 10
>>> x
<MagicMock name='mock.__iadd__().__iadd__()' id='139845830154168'>
>>> mock.mock_calls
[call.__iadd__(5), call.__iadd__().__iadd__(10)]
```

因为操作的结果被分配给了原始变量，一系列的就地数学运算构建了一个子模拟对象链。如果你这么想，那是对的，但人们一开始很少期望是这样。

## 属性和描述符的模拟对象

基本的`Mock`对象在模拟某些事物方面并不擅长：**描述符**。

描述符是允许你干扰正常变量访问机制的对象。最常用的描述符是由 Python 的内置函数`property`创建的，它简单地允许你编写函数来控制获取、设置和删除变量。

要模拟属性（或其他描述符），创建一个`unittest.mock.PropertyMock`实例并将其分配给属性名称。唯一的复杂性是你不能将描述符分配给对象实例；你必须将其分配给对象的类型，因为描述符是在类型中查找的，而不是首先检查实例。

幸运的是，用模拟对象做这件事并不难：

```py
>>> from unittest.mock import PropertyMock
>>> mock = Mock()
>>> prop = PropertyMock()
>>> type(mock).p = prop
>>> mock.p
<MagicMock name='mock()' id='139845830215328'>
>>> mock.mock_calls
[]
>>> prop.mock_calls
[call()]
>>> mock.p = 6
>>> prop.mock_calls
[call(), call(6)]
```

在这里需要注意的事情是，该属性不是名为 mock 的对象的子属性。正因为如此，我们必须保留它自己的变量，否则我们就无法访问其历史记录。

`PropertyMock`对象将变量查找记录为不带参数的调用，将变量赋值记录为带有新值的参数的调用。

### 小贴士

如果你确实需要在模拟对象的历史记录中记录变量访问，可以使用`PropertyMock`对象。通常你不需要这样做，但这个选项是存在的。

即使你是通过将属性分配给类型的属性来设置属性的，你也不必担心你的`PropertyMock`对象会溢出到其他测试中。你创建的每个`Mock`都有自己的类型对象，尽管它们都声称属于同一个类：

```py
>>> type(Mock()) is type(Mock())
False
```

多亏了这个特性，你对模拟对象类型对象所做的任何更改都是针对该特定模拟对象的。

## 模拟文件对象

你可能会偶尔需要用模拟对象替换文件对象。`unittest.mock`库通过提供`mock_open`来帮助你，这是一个伪造打开函数的工厂。这些函数具有与真实打开函数相同的接口，但它们返回一个配置为假装是打开文件对象的模拟对象。

这听起来比实际情况要复杂。请亲自看看：

```py
>>> from unittest.mock import mock_open
>>> open = mock_open(read_data = 'moose')
>>> with open('/fake/file/path.txt', 'r') as f:
...   print(f.read())
...
moose
```

如果您将字符串值传递给 `read_data` 参数，最终创建的模拟文件对象将在其读取方法被调用时使用该值作为数据源。截至 Python 3.4.0，`read_data` 只支持字符串对象，不支持字节。

如果您没有传递 `read_data`，`read` 方法调用将返回一个空字符串。

之前代码的问题在于它使真实打开函数不可访问，并留下一个模拟对象，其他测试可能会遇到它。请继续阅读以了解如何解决这些问题。

## 用模拟对象替换真实代码

`unittest.mock` 库提供了一个非常棒的临时用模拟对象替换对象的工具，并在我们的测试完成后撤销更改。这个工具就是 `unittest.mock.patch`。

那个 `patch` 可以以多种不同的方式使用：它作为一个上下文管理器、一个函数装饰器和一个类装饰器；此外，它还可以创建一个用于替换的模拟对象，或者使用您指定的替换对象。还有一些其他可选参数可以进一步调整 `patch` 的行为。

基本用法很简单：

```py
>>> from unittest.mock import patch, mock_open
>>> with patch('builtins.open', mock_open(read_data = 'moose')) as mock:
...    with open('/fake/file.txt', 'r') as f:
...       print(f.read())
...
moose
>>> open
<built-in function open>
```

如您所见，`patch` 将由 `mock_open` 创建的模拟打开函数覆盖在真实打开函数之上；然后，当我们离开上下文时，它会自动为我们替换原始函数。

`patch` 的第一个参数是唯一必需的参数。它是一个描述要替换的对象的绝对路径的字符串。路径可以包含任意数量的包和子包名称，但必须包括模块名称和模块中被替换的对象的名称。如果路径不正确，`patch` 将根据路径的具体错误抛出 `ImportError`、`TypeError` 或 `AttributeError`。

如果您不想担心创建一个模拟对象作为替换，您可以直接省略该参数：

```py
>>> import io
>>> with patch('io.BytesIO'):
...    x = io.BytesIO(b'ascii data')
...    io.BytesIO.mock_calls
[call(b'ascii data')]
```

如果您没有告诉 `patch` 使用什么作为替换对象，`patch` 函数将为您创建一个新的 `MagicMock`。这通常工作得很好，但您可以通过传递新的参数（也是本节第一个示例中的第二个参数）来指定替换对象应该是特定的对象；或者您可以通过传递 `new_callable` 参数来让 `patch` 使用该参数的值来创建替换对象。

我们也可以通过传递 `autospec=True` 来强制 `patch` 使用 `create_autospec` 创建替换对象：

```py
>>> with patch('io.BytesIO', autospec = True):
...    io.BytesIO.melvin
Traceback (most recent call last):
 File "<stdin>", line 2, in <module>
 File "/usr/lib64/python3.4/unittest/mock.py", line 557, in __getattr__
 raise AttributeError("Mock object has no attribute %r" % name)
AttributeError: Mock object has no attribute 'melvin'
```

通常，`patch` 函数会拒绝替换不存在的对象；但是，如果您传递 `create=True`，它将愉快地在您喜欢的任何地方放置一个模拟对象。当然，这与 `autospec=True` 不兼容。

`patch` 函数涵盖了最常见的用例。还有一些相关的函数处理较少见但仍很有用的用例。

`patch.object`函数与`patch`做的是同样的事情，只不过它接受一个对象和一个属性名称作为其前两个参数，而不是路径字符串。有时这比找出对象的路径更方便。许多对象甚至没有有效的路径（例如，仅存在于函数局部作用域中的对象），尽管修补它们的需求比您想象的要少。

`patch.dict`函数临时将一个或多个对象放入字典中的特定键下。第一个参数是目标字典；第二个参数是从中获取键值对以放入目标字典的字典。如果您传递`clear=True`，则在插入新值之前将清空目标字典。请注意，`patch.dict`不会为您创建替换值。如果您想使用它们，您需要自己创建模拟对象。

# 模拟对象在行动

那是很多理论与不切实际的例子交织在一起。让我们回顾一下我们已经学到的内容，并将其应用到前几章的测试中，以便更真实地了解这些工具如何帮助我们。

## 更好的 PID 测试

PID 测试主要受到必须进行大量额外工作来修补和取消修补`time.time`的影响，并且在打破对构造函数的依赖方面有一些困难。

### 修补 time.time

使用`patch`，我们可以消除处理`time.time`的许多重复性；这意味着我们不太可能在某个地方犯错误，并节省了我们花费时间在某种程度上既无聊又令人烦恼的事情上。所有的测试都可以从类似的变化中受益：

```py
>>> from unittest.mock import Mock, patch
>>> with patch('time.time', Mock(side_effect = [1.0, 2.0, 3.0, 4.0, 5.0])):
...    import pid
...    controller = pid.PID(P = 0.5, I = 0.5, D = 0.5, setpoint = 0,
...                         initial = 12)
...    assert controller.gains == (0.5, 0.5, 0.5)
...    assert controller.setpoint == [0.0]
...    assert controller.previous_time == 1.0
...    assert controller.previous_error == -12.0
...    assert controller.integrated_error == 0.0
```

除了使用`patch`来处理`time.time`之外，这个测试已经发生了变化。我们现在可以使用`assert`来检查事物是否正确，而不是让 doctest 直接比较值。这两种方法几乎没有区别，只不过我们可以将`assert`语句放在`patch`管理的上下文中。

### 与构造函数解耦

使用模拟对象，我们最终可以将 PID 方法的测试与构造函数分开，这样构造函数中的错误就不会影响结果：

```py
>>> with patch('time.time', Mock(side_effect = [2.0, 3.0, 4.0, 5.0])):
...    pid = imp.reload(pid)
...    mock = Mock()
...    mock.gains = (0.5, 0.5, 0.5)
...    mock.setpoint = [0.0]
...    mock.previous_time = 1.0
...    mock.previous_error = -12.0
...    mock.integrated_error = 0.0
...    assert pid.PID.calculate_response(mock, 6) == -3.0
...    assert pid.PID.calculate_response(mock, 3) == -4.5
...    assert pid.PID.calculate_response(mock, -1.5) == -0.75
...    assert pid.PID.calculate_response(mock, -2.25) == -1.125
```

我们在这里所做的是设置一个具有适当属性的模拟对象，并将其作为 self 参数传递给`calculate_response`。我们可以这样做，因为我们根本就没有创建 PID 实例。相反，我们在类内部查找方法的函数并直接调用它，这使得我们可以传递任何我们想要的作为 self 参数，而不是让 Python 的自动机制来处理它。

从不调用构造函数意味着我们对它可能包含的任何错误免疫，并保证了对象状态正是我们在`calculate_response`测试中期望的。

# 摘要

在本章中，我们了解了一组专门模仿其他类、对象、方法和函数的对象家族。我们看到了如何配置这些对象以处理它们默认行为不足的边缘情况，并且我们学习了如何检查这些模拟对象所保留的活动日志，以便我们可以决定这些对象是否被正确使用。

在下一章中，我们将探讨 Python 的 `unittest` 包，这是一个比 `doctest` 更为结构化的测试框架，它在与人沟通方面不如 `doctest` 有用，但更能处理大规模测试的复杂性。
