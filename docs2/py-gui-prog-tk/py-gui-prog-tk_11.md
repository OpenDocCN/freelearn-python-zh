

# 使用 unittest 创建自动测试

随着你的应用程序规模和复杂性的快速扩展，你对做出更改感到紧张。万一你弄坏了什么？你怎么知道？当然，你可以手动运行程序的所有功能，使用各种输入并监视错误，但随着你添加更多功能，这种方法会变得更加困难且耗时。你真正需要的是一个快速且可靠的方法来确保每次代码更改时程序都能正常工作。

幸运的是，有一种方法：自动测试。在本章中，你将学习以下关于自动测试的内容：

+   在*自动测试基础*中，你将了解使用`unittest`在 Python 中进行自动测试的基本原理。

+   在*测试 Tkinter 代码*中，我们将讨论测试 Tkinter 应用程序的具体策略。

+   在*为我们的应用程序编写测试*中，我们将应用这些知识到 ABQ 数据录入应用程序。

# 自动测试基础

到目前为止，测试我们的应用程序一直是一个启动它，运行一些基本程序，并验证它是否按预期执行的过程。这种方法在非常小的脚本上可以接受，但随着我们的应用程序增长，验证应用程序行为的过程变得越来越耗时且容易出错。

使用自动测试，我们可以在几秒钟内一致地验证我们的应用程序逻辑。有几种自动测试的形式，但最常见的是**单元测试**和**集成测试**。单元测试与独立的代码片段一起工作，使我们能够快速验证特定部分的行为。集成测试验证多个代码单元之间的交互。我们将编写这两种类型的测试来验证我们应用程序的行为。

## 简单单元测试

最基本的单元测试只是一个短程序，它在不同的条件下运行一段代码单元，并将其输出与预期结果进行比较。

考虑以下计算类：

```py
# unittest_demo/mycalc.py
import random
class MyCalc:
  def __init__(self, a, b):
    self.a = a
    self.b = b
  def add(self):
    return self.a + self.b
  def mod_divide(self):
    if self.b == 0:
      raise ValueError("Cannot divide by zero")
    return (int(self.a / self.b), self.a % self.b)
  def rand_between(self):
    return (
      (random.random() * abs(self.a - self.b))
      + min(self.a, self.b)
    ) 
```

这个类初始化时带有两个数字，它可以在其后执行各种数学运算。

假设我们想编写一些代码来测试这个类是否按预期工作。一种简单的方法可能如下所示：

```py
# unittest_demo/test_mycalc_no_unittest.py
from mycalc import MyCalc
mc1 = MyCalc(1, 100)
mc2 = MyCalc(10, 4)
try:
  assert mc1.add() == 101, "Test of add() failed."
  assert mc2.mod_divide() == (2, 2), "Test of mod_divide() failed."
except AssertionError as e:
  print("Test failed: ", e)
else:
  print("Tests succeeded!") 
```

这段测试代码创建了一个`MyCalc`对象，然后使用`assert`语句检查`add()`和`mod_divide()`的输出是否与预期值相符。Python 中的`assert`关键字是一个特殊的语句，如果其后的表达式评估为`False`，则会引发`AssertionError`异常。逗号后面的消息字符串是传递给`AssertionError`异常初始化器的错误字符串。

换句话说，`assert expression, "message"`语句等同于：

```py
if not expression:
  raise AssertionError("message") 
```

目前，如果你运行`MyCalc`的测试脚本，所有测试都通过。让我们尝试更改`add()`方法使其失败：

```py
def add(self):
  return self.a - self.b 
```

现在，运行测试给出以下错误：

```py
Test failed: Test of add() failed. 
```

这样的测试有什么价值？对于这样一个简单的函数，这似乎是多余的。但是，假设有人决定按照以下方式重构我们的`mod_divide()`方法：

```py
def mod_divide(self):
  #...
  return (self.a // self.b, self.a % self.b) 
```

这种方法稍微复杂一些，你可能对涉及的运算符熟悉或不熟悉。然而，由于这种方法通过了我们的测试，我们有证据表明这个算法是正确的，即使我们没有完全理解代码。如果重构存在问题，我们的测试可以帮助我们快速识别问题。

测试纯数学函数相对简单；不幸的是，测试实际应用程序代码给我们带来了挑战，需要更复杂的方法。

考虑这些问题：

+   代码单元通常依赖于在测试之前必须设置并在测试之后清理的预存在状态。

+   代码可能存在副作用，会改变代码单元外部的对象。

+   代码可能与慢速、不可靠或不可预测的资源交互。

+   实际应用程序包含许多需要测试的函数和类，理想情况下我们希望一次性被提醒所有问题。按照目前编写的测试脚本，它会在第一个失败的断言处停止，所以我们只会一次被提醒一个问题。

为了解决这些问题以及其他问题，程序员依赖于**测试框架**，以使编写和执行自动化测试尽可能简单、高效和可靠。

## unittest 模块

`unittest`模块是 Python 标准库的自动化测试框架。它为我们提供了一些强大的工具，使测试我们的代码变得相对容易，并且基于许多测试框架中发现的某些标准单元测试概念。这些概念包括：

+   **测试**：测试是一个单独的方法，它要么完成，要么引发异常。测试通常关注一个代码单元，如函数、方法或过程。测试可以成功，表示测试成功；失败，表示代码未通过测试；或者错误，表示测试本身遇到了问题。

+   **测试用例**：测试用例是一组应该一起运行的测试，包含类似的设置和清理要求，通常对应于一个类或模块。测试用例可以有**固定装置**，这些是在每个测试之前需要设置并在每个测试之后清理的项目，以提供一个干净、可预测的环境，使测试可以运行。

+   **测试套件**：测试套件是一组测试用例，用于覆盖应用程序或模块的所有代码。

+   **模拟对象**：模拟对象是代表另一个对象的实体。通常，它们用于替换外部资源，如文件、数据库或库模块。在测试期间，模拟对象会覆盖这些资源，以提供一个快速且可预测的替代品，且没有副作用。

为了深入探索这些概念，让我们使用`unittest`测试我们的`MyCalc`类。

### 编写测试用例

让我们为`MyCalc`类创建一个测试用例。创建一个名为`test_mycalc.py`的新文件，并输入以下代码：

```py
# unittest_demo/test_mycalc.py
import mycalc
import unittest
class TestMyCalc(unittest.TestCase):
  def test_add(self):
    mc = mycalc.MyCalc(1, 10)
    assert mc.add() == 11
if __name__ == '__main__':
  unittest.main() 
```

你的测试模块和测试方法的名称都应该以`test_`为前缀。这样做允许`unittest`运行器自动找到测试模块，并区分测试用例类中的测试方法和其他方法。

如你所猜，`TestCase`类代表一个测试用例。为了对`MyCalc`进行测试用例，我们已从`TestCase`派生并添加了一个`test_`方法，该方法将测试我们类的一些方面。在`test_add()`方法内部，我们创建了一个`MyCalc`对象，然后对`add()`的输出进行了断言。

在文件末尾，我们添加了对`unittest.main()`的调用，这将导致文件中的所有测试用例被执行。

如果你在命令行中运行你的测试文件，你应该得到以下输出：

```py
.
---------------------------------------------------------------------
Ran 1 test in 0.000s
OK 
```

第一行上的单个点代表我们的一个测试（`test_add()`）。对于每个测试方法，`unittest.main()`将输出以下之一：

+   一个点，表示测试通过

+   `F`，表示它失败了

+   `E`，表示测试引发了错误

最后，我们得到一个总结，包括运行了多少个测试以及花费了多长时间。`OK`表示所有测试都成功通过。

要查看测试失败时会发生什么，让我们修改我们的测试，使其故意失败：

```py
def test_add(self):
  mc = mycalc.MyCalc(1, 10)
  assert mc.add() == **12** 
```

现在你运行测试模块时，你应该看到如下输出：

```py
F
=====================================================================
FAIL: test_add (__main__.TestMyCalc)
---------------------------------------------------------------------Traceback (most recent call last):
File "test_mycalc.py", line 8, in test_add
assert mc.add() == 12
AssertionError
---------------------------------------------------------------------Ran 1 test in 0.000s
FAILED (failures=1) 
```

注意顶部单个的`F`，代表我们的失败测试。在所有测试运行完毕后，我们得到任何失败测试的完整跟踪记录，这样我们就可以轻松地定位失败的代码并更正它。

尽管如此，这个跟踪输出并不理想；我们可以看到`mc.add()`没有返回`12`，但我们不知道它*实际上*返回了什么。我们可以在我们的`assert`调用中添加一个注释字符串，但`unittest`提供了一个更好的方法：`TestCase`断言方法。

### TestCase 断言方法

`TestCase`对象具有许多断言方法，提供了一种更干净、更健壮的方式来运行我们代码输出的各种测试。

例如，这里有`TestCase.assertEqual()`方法来测试相等性，我们可以如下使用：

```py
 def test_add(self):
    mc = mycalc.MyCalc(1, 10)
    **self.assertEqual(mc.add(), 12)** 
```

当我们用此代码运行测试用例时，你可以看到跟踪记录得到了改进：

```py
Traceback (most recent call last):
File "test_mycalc.py", line 11, in test_add
self.assertEqual(mc.add(), 12)
AssertionError: 11 != 12 
```

现在，我们可以看到`mc.add()`返回的值，这对调试非常有帮助。`TestCase`包含 20 多种断言方法，可以简化对各种条件（如类继承、抛出异常和序列成员资格）的测试。

一些更常用的方法列在以下表格中：

| 方法 | 测试 |
| --- | --- |
| `assertEqual(a`, `b)` | `a` `==` `b` |
| `assertTrue(a)` | `a` `is` `True` |
| `assertFalse(a)` | `a` `is` `False` |
| `assertIn(item`, `sequence)` | `item` `in` `sequence` |
| `assertRaises(exception`, `callable`, `*args)` | `callable` raises `exception` when called with `args` |
| `assertGreater(a`, `b)` | `a` is greater than `b` |
| `assertLess(a`, `b)` | `a` is less than `b` |

可用的断言方法的完整列表可以在 `unittest` 文档的 [`docs.python.org/3/library/unittest.html#unittest.TestCase`](https://docs.python.org/3/library/unittest.html#unittest.TestCase) 找到。

让我们使用一个断言方法来测试当 `b` 为 `0` 时，`mod_divide()` 会引发一个 `ValueError` 异常：

```py
 def test_mod_divide(self):
    mc = mycalc.MyCalc(1, 0)
    self.assertRaises(ValueError, mc.mod_divide) 
```

当函数在调用时引发给定的异常时，`assertRaises()` 会 *通过*。如果我们需要将任何参数传递给测试的函数，它们可以作为额外的参数指定给 `assertRaises()`。

`assertRaises()` 也可以像这样用作上下文管理器：

```py
 def test_mod_divide(self):
    mc = mycalc.MyCalc(1, 0)
    with self.assertRaises(ValueError):
      mc.mod_divide() 
```

这段代码实现了完全相同的功能，但更加清晰和灵活，因为它允许我们将多行代码放入代码块中。

您也可以轻松地将自己的自定义断言方法添加到测试用例中；这只是一个创建在某种条件下引发 `AssertionError` 异常的方法的问题。

### 固定设施

应该很明显，我们测试用例中的每个测试都需要访问一个 `MyCalc` 对象。如果我们不需要在每个测试方法中手动做这件事，那就很好了。为了帮助我们避免这项繁琐的任务，`TestCase` 对象提供了一个 `setUp()` 方法。该方法在运行每个测试用例之前运行，通过覆盖它，我们可以处理每个测试所需的任何设置。

例如，我们可以用它来创建 `MyCalc` 对象，如下所示：

```py
 def setUp(self):
    self.mycalc1_0 = mycalc.MyCalc(1, 0)
    self.mycalc36_12 = mycalc.MyCalc(36, 12) 
```

现在，每个测试用例都可以使用这些对象来运行其测试，而不是创建它们自己的。理解到 `setUp()` 方法将在 *每个* 测试之前重新运行，因此这些对象将在测试方法之间始终重置。如果我们有需要在每次测试后清理的项目，我们也可以覆盖 `tearDown()` 方法，该方法在每次测试之后运行（在这种情况下，这不是必要的）。

现在我们有了 `setUp()` 方法，我们的 `test_add()` 方法可以更加简单：

```py
 def test_add(self):
    self.assertEqual(self.mycalc1_0.add(), 1)
    self.assertEqual(self.mycalc36_12.add(), 48) 
```

除了实例方法 `setUp()` 和 `tearDown()` 之外，`TestCase` 还具有用于对象本身设置和清理的类方法；这些是 `setUpClass()` 和 `tearDownClass()`。这两个方法可以用于在测试用例创建和销毁时运行的较慢操作，而不是需要在每次测试之间刷新；例如，您可能可以使用它们来创建测试所需的复杂对象，而这些对象不会被任何测试所更改。

### 使用 Mock 和 patch

`MyCalc.rand_between()` 方法生成一个介于 `a` 和 `b` 之间的随机数。因为我们无法预测其输出，所以我们不能提供一个固定值来测试它。我们如何测试这个方法？

一个简单的方法可能看起来像这样：

```py
def test_rand_between(self):
  rv = self.mycalc1_0.rand_between()
  self.assertLessEqual(rv, 1)
  self.assertGreaterEqual(rv, 0) 
```

如果我们的代码是正确的，这个测试就会通过，但代码错误时它不一定失败；事实上，如果代码错误，它可能会不可预测地通过或失败，因为`rand_between()`的返回值是随机的。例如，如果`MyCalc(1,10).rand_between()`错误地返回了 2 到 11 之间的值，那么如果返回 2 到 10 之间的值，测试就会通过，只有当返回 11 时才会失败。因此，即使代码是错误的，测试套件在每次运行时只有 10%的几率会失败。

为了测试的目的，我们可以安全地假设标准库函数，如`random()`，是正确工作的；因此，我们的单元测试实际上应该测试我们自己的方法是否正确处理了`random()`提供的数值。如果我们能够临时替换`random()`为一个返回可预测固定值的函数，那么测试我们后续计算的准确性就会变得简单。

`unittest.mock`模块为我们提供了`Mock`类来实现这个目的。`Mock`对象可以用来可预测地模拟另一个类、方法或库的行为。我们可以为我们的`Mock`对象提供返回值、副作用、属性、方法和其他特征，以模拟另一个类、对象、函数或模块的行为，然后在运行测试之前将其放置在适当的位置。

为了看到这个效果，让我们使用`Mock`创建一个假的`random()`函数，如下所示：

```py
from unittest.mock import Mock
  #... inside TestMyCalc
  def test_rand_between(self):
    fakerandom = Mock(return_value=.5) 
```

`Mock`对象的`return_value`参数允许我们在每次调用函数时硬编码一个返回值。在这里，我们的模拟对象`fakerandom`将表现得像一个总是返回`0.5`的函数。

现在，我们可以将`fakerandom`替换为`random()`，如下所示：

```py
 #...
    orig_random = mycalc.random.random
    mycalc.random.random = fakerandom
    rv = self.mycalc1_0.rand_between()
    self.assertEqual(rv, 0.5)
    mycalc.random.random = orig_random 
```

我们首先保存对`mycalc.random.random`的引用，然后再进行替换。请注意，我们特别替换了`mycalc.py`中使用的`random`版本，这样就不会影响到其他地方的`random()`调用。在修补库时尽可能具体是一种最佳实践，以避免不可预见的副作用。

在`fakerandom`模块就绪后，我们可以调用`rand_between()`并测试输出。因为`fakerandom()`总是返回`0.5`，所以当`a`为`1`且`b`为`0`时，答案应该是`(0.5 × 1 + 0) = 0.5`。任何其他值都表明我们的算法中存在错误。在测试代码的末尾，我们将`random`恢复到原始的标准库函数，这样其他测试（或它们调用的类或函数）就不会意外地使用模拟。

每次需要存储或恢复原始库时都是一种不必要的麻烦，所以`unittest.mock`提供了一个更干净的方法，使用`patch()`。`patch()`函数可以用作上下文管理器或装饰器，两种方法都可以使将`Mock`对象修补到我们的代码中变得更加干净。

使用`patch()`作为上下文管理器来交换`fakerandom()`看起来是这样的：

```py
# test_mycalc.py
from unittest.mock import patch
#... inside TestMyCalc
  def test_rand_between(self):
    with patch('mycalc.random.random') as fakerandom:
      fakerandom.return_value = 0.5
      rv = self.mycalc1_0.rand_between()
      self.assertEqual(rv, 0.5) 
```

`patch()`命令接受一个导入路径字符串，并为我们提供一个新创建的`Mock`对象，该对象已在该路径上替换了原始对象。在上下文管理器块内部，我们可以在`Mock`对象上设置方法和属性，然后运行我们的实际测试。当块结束时，修补的函数将恢复到其原始版本。

将`patch()`用作装饰器类似：

```py
 @patch('mycalc.random.random')
  def test_rand_between2(self, fakerandom):
    fakerandom.return_value = 0.5
    rv = self.mycalc1_0.rand_between()
    self.assertEqual(rv, 0.5) 
```

在这种情况下，由`patch()`创建的`Mock`对象作为参数传递给我们的测试方法，并在装饰函数的持续时间内在其位置上保持修补。如果我们计划在测试方法中多次使用模拟，这种方法效果很好。

### 运行多个单元测试

虽然我们可以在文件末尾包含对`unittest.main()`的调用以运行我们的单元测试，但这种方法扩展性不好。随着我们的应用程序增长，我们将编写许多测试文件，我们希望可以分组或一次性运行。

幸运的是，`unittest`可以通过一条命令发现并运行项目中的所有测试：

```py
$ python -m unittest 
```

只要你遵循了推荐的命名方案，即以`test_`前缀命名你的测试模块，在项目根目录下运行此命令就应该会运行所有测试脚本。

# 测试 Tkinter 代码

测试 Tkinter 代码给我们带来了一些特定的挑战。首先，Tkinter 异步处理许多回调和方法，这意味着我们不能指望某些代码的结果立即显现。此外，测试 GUI 行为通常依赖于外部因素，如窗口管理或视觉提示，这些因素是我们测试无法检测的。

在本节中，我们将学习一些工具和策略来解决这些问题，并帮助你为 Tkinter 代码编写测试。

## 管理异步代码

每当你与 Tkinter UI 交互时——无论是点击按钮、在字段中输入还是打开窗口等——响应不会立即执行。

相反，这些操作被放置在一个类似于待办事项列表的**事件队列**中，以便在代码执行继续的同时进行处理。虽然这些操作对用户来说似乎是瞬间的，但测试代码不能指望在运行下一行代码之前完成请求的操作。

为了解决这个问题，Tkinter 部件有一些方法允许我们管理事件队列：

+   `wait_visibility()`: 此方法导致代码等待直到部件完全绘制到屏幕上，然后执行下一行代码。

+   `update_idletasks()`: 此方法强制 Tkinter 处理当前在部件上挂起的任何空闲任务。空闲任务是指低优先级任务，如绘图和渲染。

+   `update()`: 此方法强制 Tkinter 处理部件上挂起的全部事件，包括调用回调、重绘和几何管理。它包括`update_idletasks()`所做的一切以及更多。

事件队列将在第十四章“使用线程和队列的异步编程”中更详细地讨论。

## 模拟用户操作

在自动化 GUI 测试时，我们可能想知道当用户点击某个控件或输入某个按键时会发生什么。当这些动作在 GUI 中发生时，Tkinter 会为该控件生成一个 `Event` 对象并将其传递给事件队列。我们可以在代码中做同样的事情，使用控件的 `event_generate()` 方法。

### 指定事件序列

正如我们在 *第六章* 中学到的，*为应用扩展做准备*，我们可以通过将事件**序列字符串**传递给 `event_generate()` 方法中的格式 `<EventModifier-EventType-EventDetail>` 来在控件上注册一个事件。让我们更详细地看看序列字符串。

事件序列字符串的核心部分是**事件类型**。它指定了我们发送的事件类型，例如按键、鼠标点击、窗口事件等。

Tkinter 大约有 30 种事件类型，但通常你只需要处理以下几种：

| 事件类型 | 表示的动作 |
| --- | --- |
| `ButtonPress` 或 `Button` | 鼠标按钮点击 |
| `ButtonRelease` | 松开鼠标按钮 |
| `KeyPress` 或 `Key` | 按下键盘上的键 |
| `KeyRelease` | 松开键盘上的键 |
| `FocusIn` | 将焦点给予一个控件，例如按钮或输入控件 |
| `FocusOut` | 离开一个聚焦的控件 |
| `Enter` | 鼠标光标进入一个控件 |
| `Leave` | 鼠标光标从一个控件上移开 |
| `Configure` | 控件配置的改变，例如，调用 `config()`，或者用户调整窗口大小等 |

**事件修饰符**是可选的单词，可以改变事件类型；例如，`Control`、`Alt` 和 `Shift` 可以用来表示这些修饰键之一被按下；`Double` 或 `Triple` 可以与 `Button` 结合使用，以表示描述的按钮的双击或三击。如果需要，可以串联多个修饰符。

**事件细节**，仅对键盘或鼠标事件有效，描述了哪个键或按钮被按下。例如，`<Button-1>` 指的是左鼠标按钮，而 `<Button-3>` 指的是右。对于字母和数字键，可以使用实际的字母或数字，例如 `<Control-KeyPress-a>`；然而，大多数符号都由一个单词（如 `minus`、`colon`、`semicolon` 等）描述，以避免语法冲突。

对于按钮点击和按键，事件类型在技术上不是必需的；例如，你可以使用 `<Control-a>` 而不是 `<Control-KeyPress-a>`。然而，出于清晰起见，保留它可能是个好主意。例如，`<1>` 是一个有效的事件，但它指的是按下左鼠标按钮还是数字键 1？你可能惊讶地发现它是鼠标按钮。

下表展示了有效事件序列的一些示例：

| 序列 | 含义 |
| --- | --- |
| `<Double-Button-3>` | 双击右鼠标按钮 |
| `<Alt-KeyPress-exclam>` | 按住 Alt 并输入感叹号 |
| `<Control-Alt-Key-m>` | 按住 Control 和 Alt 并按下 M 键 |
| `<KeyRelease-minus>` | 释放按下的减号键 |

除了序列之外，我们还可以向`event_generate()`传递其他参数，以描述事件的各个方面。其中许多是多余的，但在某些情况下，我们需要为事件提供额外信息，以便它具有任何意义；例如，鼠标按钮事件需要包括一个`x`和`y`参数，以指定点击的坐标。

单括号包围的序列表示内置事件类型。双括号用于自定义事件，例如我们在主菜单和其他地方使用的事件。

## 管理焦点和抓取

**焦点**指的是当前接收键盘输入的控件或窗口。控件也可以**抓取焦点**，防止鼠标在其边界之外移动或按键。

Tkinter 为我们提供了这些用于管理焦点和抓取的控件方法，其中一些对于运行测试很有用：

| 方法 | 描述 |
| --- | --- |
| `focus_set()` | 当其窗口下次获得焦点时聚焦控件 |
| `focus_force()` | 立即聚焦控件及其所在的窗口 |
| `grab_set()` | 控件抓取应用程序的所有事件 |
| `grab_set_global()` | 控件抓取屏幕上的所有事件 |
| `grab_release()` | 控件放弃其抓取 |

在测试环境中，我们可以使用这些方法来确保我们生成的键盘和鼠标事件将发送到正确的控件或窗口。

大多数时候`focus_set()`方法就足够了，但根据您应用程序的行为和操作系统窗口环境，您可能需要更极端的强制措施，如`focus_force()`或`grab_set()`。

## 获取控件信息

Tkinter 控件有一组`winfo_`方法，使我们能够访问有关控件的信息。虽然可用的功能还有很多不足，但这些方法包括一些我们可以在测试中使用的信息，以提供有关给定控件状态的反馈。

以下是一些我们将发现很有用的`winfo_`方法：

| 方法 | 描述 |
| --- | --- |
| `winfo_height()`、`winfo_width()` | 获取控件的高度和宽度 |
| `winfo_children()` | 获取子控件的列表 |
| `winfo_geometry()` | 获取控件的大小和位置 |
| `winfo_ismapped()` | 确定控件是否已映射（即，它已被添加到布局中，使用几何管理器） |
| `winfo_viewable()` | 确定控件是否可见（即，它及其所有父控件都已映射） |
| `winfo_x()`、`winfo_y()` | 获取控件左上角的*x*或*y*坐标 |

# 为我们的应用程序编写测试

让我们将我们对`unittest`和 Tkinter 的知识运用起来，为我们的应用程序编写一些自动化测试。要开始，我们需要创建一个测试模块。在`abq_data_entry`包内创建一个名为`test`的目录，并在其中创建传统的空`__init__.py`文件。我们将在该目录内创建所有的测试模块。

## 测试数据模型

除了需要读取和写入文件之外，我们的`CSVModel`类相当独立。我们需要模拟这一功能，以便测试不会干扰文件系统。由于文件操作是在测试中需要模拟的更常见的事情之一，因此`mock`模块提供了`mock_open()`，这是一个现成的`Mock`子类，用于替换 Python 的`open()`方法。当调用时，`mock_open`对象返回一个模拟文件句柄对象，并支持`read()`、`write()`和`readlines()`方法。

在`test`目录中创建一个名为`test_models.py`的新文件。这将是我们的数据模型类测试模块。以一些模块导入开始：

```py
# test_models.py
from .. import models
from unittest import TestCase
from unittest import mock
from pathlib import Path 
```

除了`models`模块，我们还需要`TestCase`和`mock`，当然，以及`Path`类，因为我们的`CSVModel`内部使用`Path`对象。

现在，我们将开始对`CSVModel`类进行测试用例，如下所示：

```py
class TestCSVModel(TestCase):
  def setUp(self):
    self.file1_open = mock.mock_open(
      read_data=(
        "Date,Time,Technician,Lab,Plot,Seed Sample,"
        "Humidity,Light,Temperature,Equipment Fault,"
        "Plants,Blossoms,Fruit,Min Height,Max Height,"
        "Med Height,Notes\r\n"
        "2021-06-01,8:00,J Simms,A,2,AX478,24.47,1.01,21.44,"
        "False,14,27,1,2.35,9.2,5.09,\r\n"
        "2021-06-01,8:00,J Simms,A,3,AX479,24.15,1,20.82,"
        "False,18,49,6,2.47,14.2,11.83,\r\n"
      )
    )
    self.file2_open = mock.mock_open(read_data='')
    self.model1 = models.CSVModel('file1')
    self.model2 = models.CSVModel('file2') 
```

在此案例的`setUp()`方法中，我们创建了两个模拟数据文件。第一个包含 CSV 标题和两行 CSV 数据，而第二个是空的。`mock_open`对象的`read_data`参数允许我们指定当代码尝试从它读取数据时将返回的字符串。

我们还创建了两个`CSVModel`对象，一个文件名为`file1`，另一个文件名为`file2`。值得一提的是，我们的模型和`mock_open`对象之间没有实际的联系；给定的文件名是任意的，因为我们实际上不会打开文件，而选择使用哪个`mock_open`对象将在我们的测试方法中使用`patch()`来决定。

### 在`get_all_records()`中测试文件读取

要了解我们如何使用这些，让我们从对`get_all_records()`方法的测试开始，如下所示：

```py
# test_models.py, inside TestCSVModel
  @mock.patch('abq_data_entry.models.Path.exists')
  def test_get_all_records(self, mock_path_exists):
    mock_path_exists.return_value = True 
```

由于我们的文件名实际上不存在，我们使用`patch()`的装饰器版本将`Path.exists()`替换为始终返回`True`的模拟函数。如果我们想测试文件不存在的情况，我们可以稍后更改此对象的`return_value`属性。

要针对我们的一个`mock_open`对象运行`get_all_records()`方法，我们将使用`patch()`的上下文管理器形式，如下所示：

```py
 with mock.patch(
      'abq_data_entry.models.open',
      self.file1_open
    ):
      records = self.model1.get_all_records() 
```

在此上下文管理器块内的代码中，对`open()`的任何调用都将由我们的`mock_open`对象替换，返回的文件句柄将包含我们指定的`read_data`字符串。

现在我们可以开始对返回的记录进行断言：

```py
# test_models.py, inside TestCSVModel.test_get_all_records()
    self.assertEqual(len(records), 2)
    self.assertIsInstance(records, list)
    self.assertIsInstance(records[0], dict) 
```

在这里，我们检查`records`包含两行（因为我们的读取数据包含两个 CSV 记录），它是一个`list`对象，并且它的第一个成员是一个`dict`对象（或`dict`的子类）。

接下来，让我们确保所有字段都已通过，并且布尔转换已成功：

```py
 fields = (
      'Date', 'Time', 'Technician', 'Lab', 'Plot',
      'Seed Sample', 'Humidity', 'Light',
      'Temperature', 'Equipment Fault', 'Plants',
      'Blossoms', 'Fruit', 'Min Height', 'Max Height',
      'Med Height', 'Notes')
    for field in fields:
      self.assertIn(field, records[0].keys())
    self.assertFalse(records[0]['Equipment Fault']) 
```

通过遍历所有字段名称的元组，我们可以检查记录输出中是否包含所有我们的字段。不要害怕在测试中使用这种方式来快速检查大量内容。

`Mock`对象不仅可以代替另一个类或函数；它还具有自己的断言方法，可以告诉我们它是否被调用、调用了多少次以及调用的参数。

例如，我们可以检查我们的`mock_open`对象以确保它以预期的参数被调用：

```py
 self.file1_open.assert_called_with(
      Path('file1'), 'r', encoding='utf-8', newline=''
    ) 
```

`assert_called_with()`接受任意数量的位置参数和关键字参数，并检查模拟对象的最后调用是否包含那些确切的参数。我们期望`file1_open()`以包含文件名`file1`的`Path`对象、模式为`r`、`newline`设置为空字符串以及`encoding`值为`utf-8`的方式被调用。通过确认模拟函数是否以正确的参数被调用，并假设真实函数（在这种情况下是内置的`open()`函数）的正确性，我们可以避免测试实际的结果。

注意，对于此方法，关键字参数的传递顺序并不重要。

### 在`save_record()`中测试文件保存

为了演示如何使用`mock_open`测试文件写入，让我们测试`save_record()`。首先创建一个测试方法，定义一些数据：

```py
 @mock.patch('abq_data_entry.models.Path.exists')
  def test_save_record(self, mock_path_exists):
    record = {
      "Date": '2021-07-01', "Time": '12:00',
      "Technician": 'Test Technician', "Lab": 'C',
      "Plot": '17', "Seed Sample": 'test sample',
      "Humidity": '10', "Light": '99',
      "Temperature": '20', "Equipment Fault": False,
      "Plants": '10', "Blossoms": '200',
      "Fruit": '250', "Min Height": '40',
      "Max Height": '50', "Med Height": '55',
      "Notes": 'Test Note\r\nTest Note\r\n'
    }
    record_as_csv = (
      '2021-07-01,12:00,Test Technician,C,17,test sample,10,99,'
      '20,False,10,200,250,40,50,55,"Test Note\r\nTest Note\r\n"'
      '\r\n') 
```

此方法首先再次模拟`Path.exists`并创建一个数据字典，以及以 CSV 数据行表示的相同数据。

你可能会想通过代码生成记录或其预期的 CSV 输出，但始终最好在测试中坚持使用字面值；这样做使得测试的期望变得明确，并避免测试中的逻辑错误。

现在，对于我们的第一个测试场景，让我们通过使用`file2_open`和`model2`来模拟向一个空但存在的文件写入：

```py
 mock_path_exists.return_value = True
    with mock.patch('abq_data_entry.models.open', self.file2_open):
      self.model2.save_record(record, None) 
```

将我们的`mock_path_exists.return_value`设置为`True`以告诉我们的方法文件已经存在，然后我们用我们的第二个`mock_open`对象（表示一个空文件）覆盖`open()`，并调用`CSVModel.save_record()`方法。由于我们传递了一个没有行号的记录（这表示记录插入），这应该导致我们的代码尝试以追加模式打开`file2`并写入 CSV 格式的记录。

`assert_called_with()`将按以下方式测试这个假设：

```py
 self.file2_open.assert_called_with(
        Path('file2'), 'a', encoding='utf-8', newline=''
      ) 
```

虽然此方法可以告诉我们`file2_open`是否以预期的参数被调用，但我们如何访问其实际的文件句柄，以便我们可以看到写入的内容？

结果我们只需调用我们的`mock_open`对象并检索模拟的文件句柄对象，如下所示：

```py
 file2_handle = self.file2_open()
      file2_handle.write.assert_called_with(record_as_csv) 
```

一旦我们有了模拟的文件句柄（它本身也是一个`Mock`对象），我们就可以在其`write()`成员上运行测试方法，以找出它是否以预期的 CSV 数据被调用。在这种情况下，文件句柄的`write()`方法应该被调用，带有 CSV 格式的记录字符串。

让我们进行一组类似的测试，传递一个行号来模拟记录更新：

```py
 with mock.patch('abq_data_entry.models.open', self.file1_open):
      self.model1.save_record(record, 1)
      self.file1_open.assert_called_with(
        Path('file1'), 'w', encoding='utf-8'
      ) 
```

检查我们的更新是否正确完成会带来问题：`assert_called_with()` 只检查对模拟函数的最后调用。当我们更新 CSV 文件时，整个 CSV 文件都会更新，每行有一个 `write()` 调用。我们不能仅仅检查最后一个调用是否正确；我们需要确保所有行的 `write()` 调用都是正确的。为了完成这个任务，`Mock` 包含一个名为 `assert_has_calls()` 的方法，我们可以使用它来测试对对象进行的调用历史。

要使用它，我们需要创建一个 `Call` 对象列表。每个 `Call` 对象代表对模拟对象的调用。我们使用 `mock.call()` 函数创建 `Call` 对象，如下所示：

```py
 file1_handle = self.file1_open()
      file1_handle.write.assert_has_calls([
        mock.call(
          'Date,Time,Technician,Lab,Plot,Seed Sample,'
          'Humidity,Light,Temperature,Equipment Fault,Plants,'
          'Blossoms,Fruit,Min Height,Max Height,Med Height,Notes'
          '\r\n'),
        mock.call(
          '2021-06-01,8:00,J Simms,A,2,AX478,24.47,1.01,21.44,'
          'False,14,27,1,2.35,9.2,5.09,\r\n'),
        mock.call(
          '2021-07-01,12:00,Test Technician,C,17,test sample,'
          '10,99,20,False,10,200,250,40,50,55,'
          '"Test Note\r\nTest Note\r\n"\r\n')
        ]) 
```

`mock.call()` 的参数代表应该传递给函数调用的参数，在我们的情况下应该是单行的 CSV 数据字符串。我们传递给 `assert_has_calls()` 的 `Call` 对象列表代表了对模拟文件句柄的 `write()` 方法应该进行的每个调用，*按顺序*。`assert_has_calls()` 方法的 `in_order` 参数也可以设置为 `False`，在这种情况下，顺序不需要匹配。在我们的情况下，顺序很重要，因为错误的顺序会导致 CSV 文件损坏。

### 对模型进行更多测试

测试 `CSVModel` 类和 `SettingsModel` 类的其他方法应该与这两个方法基本相同。示例代码中包含了一些额外的测试，但请尝试自己想出一些。

## 测试我们的应用程序对象

我们将应用程序实现为一个 Tk 对象，它不仅作为主窗口，还作为控制器，将应用程序中其他地方定义的模型和视图连接起来。因此，正如你所期望的，`patch()` 将在我们的测试代码中发挥重要作用，因为我们模拟了所有其他组件以隔离 `Application` 对象。

在 `test` 目录下打开一个名为 `test_application.py` 的新文件，我们将从导入开始：

```py
# test_application.py
from unittest import TestCase
from unittest.mock import patch
from .. import application 
```

现在，让我们以这种方式开始我们的测试用例类：

```py
class TestApplication(TestCase):
  records = [
    {'Date': '2018-06-01', 'Time': '8:00', 'Technician': 'J Simms',
     'Lab': 'A', 'Plot': '1', 'Seed Sample': 'AX477',
     'Humidity': '24.09', 'Light': '1.03', 'Temperature': '22.01',
     'Equipment Fault': False,  'Plants': '9', 'Blossoms': '21',
     'Fruit': '3', 'Max Height': '8.7', 'Med Height': '2.73',
     'Min Height': '1.67', 'Notes': '\n\n',
    },
    {'Date': '2018-06-01', 'Time': '8:00', 'Technician': 'J Simms',
     'Lab': 'A', 'Plot': '2', 'Seed Sample': 'AX478',
     'Humidity': '24.47', 'Light': '1.01', 'Temperature': '21.44',
     'Equipment Fault': False, 'Plants': '14', 'Blossoms': '27',
     'Fruit': '1', 'Max Height': '9.2', 'Med Height': '5.09',
     'Min Height': '2.35', 'Notes': ''
     }
  ]
  settings = {
    'autofill date': {'type': 'bool', 'value': True},
    'autofill sheet data': {'type': 'bool', 'value': True},
    'font size': {'type': 'int', 'value': 9},
    'font family': {'type': 'str', 'value': ''},
    'theme': {'type': 'str', 'value': 'default'}
  } 
```

由于我们的 `TestApplication` 类将使用模拟数据代替数据和设置模型，我们在其中创建了一些类属性来存储 `Application` 预期从这些模型检索的数据样本。`setUp()` 方法将使用模拟替换所有外部类，配置模拟模型以返回我们的样本数据，然后创建一个 `Application` 实例，以便我们的测试可以使用。

注意，虽然测试记录中的布尔值是 `bool` 对象，但数值是字符串。实际上，`CSVModel` 就是这么返回数据的，因为在模型的这个点上没有进行实际的数据类型转换。

现在，让我们创建我们的 `setUp()` 方法，它看起来像这样：

```py
# test_application.py, inside TestApplication class
  def setUp(self):
    with \
      patch(
        'abq_data_entry.application.m.CSVModel'
      ) as csvmodel,\
      patch(
        'abq_data_entry.application.m.SettingsModel'
      ) as settingsmodel,\
      patch(
       'abq_data_entry.application.Application._show_login'
      ) as show_login,\
      patch('abq_data_entry.application.v.DataRecordForm'),\
      patch('abq_data_entry.application.v.RecordList'),\
      patch('abq_data_entry.application.ttk.Notebook'),\
      patch('abq_data_entry.application.get_main_menu_for_os')\
    :
      show_login.return_value = True
      settingsmodel().fields = self.settings
      csvmodel().get_all_records.return_value = self.records
      self.app = application.Application() 
```

在这里，我们使用七个 `patch()` 上下文管理器创建了一个 `with` 块，每个类、方法或函数都有一个，包括：

+   CSV 和设置模型。这些已经被别名替换，这样我们就可以配置它们以返回适当的数据。

+   `show_login()` 方法，我们将返回值硬编码为 `True` 以确保登录始终成功。注意，如果我们打算编写这个类的完整测试覆盖率，我们还想测试这个函数，但现在我们只是模拟它。

+   记录表和记录列表类，因为我们不希望这些类的问题导致我们的 `Application` 测试代码出现错误。这些类将有自己的测试用例，所以我们在这个案例中不感兴趣测试它们。我们不需要对它们进行任何配置，因此我们没有对这些模拟对象进行别名设置。

+   `Notebook` 类。如果不进行模拟，我们将在其 `add()` 方法中传递 `Mock` 对象，从而引发不必要的错误。我们可以假设 Tkinter 类可以正常工作，因此我们模拟这一部分。

+   `get_main_menu_for_os` 类，因为我们不想处理实际的菜单对象。就像记录表和记录列表一样，我们的菜单类将有自己的测试用例，所以我们最好在这里将它们排除在外。

自 Python 3.2 以来，您可以通过在每次上下文管理器调用之间使用逗号来创建包含多个上下文管理器的块。不幸的是，在 Python 3.9 或更低版本中，您不能将它们放在括号中，因此我们使用相对丑陋的转义换行方法将这个巨大的调用拆分成多行。如果您使用 Python 3.10 或更高版本，您可以在上下文管理器列表周围使用括号以获得更整洁的布局。

注意我们正在创建 `settingsmodel` 和 `csvmodel` 对象的实例，并在模拟对象的 *返回值* 上配置方法，而不是在模拟对象本身上。记住，我们的模拟正在替换 *类*，而不是 *对象*，并且是对象将包含 `Application` 对象将要调用的方法。因此，我们需要调用模拟类来访问 `Application` 将用作数据或设置模型的实际 `Mock` 对象。

与它所替代的实际类不同，当作为函数调用的 `Mock` 对象每次调用时都会返回相同的对象。因此，我们不需要保存由调用模拟类创建的对象的引用；我们只需重复调用模拟类即可访问该对象。然而，请注意，每次 `Mock` 类本身都会创建一个唯一的 `Mock` 对象。

由于 `Application` 是 Tk 的子类，因此在使用后安全地销毁它是我们的好主意；即使我们重新分配了它的变量名，Tcl/Tk 对象仍然会存在，并可能对我们的测试造成问题。为了解决这个问题，在 `TestApplication` 中创建一个 `tearDown()` 方法：

```py
 def tearDown(self):
    self.app.update()
    self.app.destroy() 
```

注意对 `app.update()` 的调用。如果我们不在销毁 `app` 之前调用它，事件队列中可能有任务会在它消失后尝试访问它。这不会破坏我们的代码，但它会在我们的测试输出中添加错误消息。

现在我们已经处理好了我们的固定值，让我们编写一个测试：

```py
 def test_show_recordlist(self):
    self.app._show_recordlist()
    self.app.notebook.select.assert_called_with(self.app.recordlist) 
```

`Application._show_recordlist()` 只有一行代码，它只是调用 `self.notebook.select()`。因为我们把 `recordlist` 设置为一个模拟对象，所以它的所有成员（包括 `select`）也都是模拟对象。因此，我们可以使用模拟断言方法来检查 `select()` 是否被调用以及调用的参数。

我们可以使用类似的技术来检查 `_populate_recordlist()`，如下所示：

```py
 def test_populate_recordlist(self):
    self.app._populate_recordlist()
    self.app.model.get_all_records.assert_called()
    self.app.recordlist.populate.assert_called_with(self.records) 
```

在这种情况下，我们还在使用 `assert_called()` 方法来查看 `CSVModel.get_all_records()` 是否被调用，它应该已经被调用来填充记录列表。与 `assert_called_with()` 不同，`assert_called()` 只检查一个函数是否被调用，因此对于没有参数的函数来说是有用的。

在某些情况下，`get_all_records()` 可以引发一个异常，在这种情况下，我们应该显示一个错误消息框。但由于我们已经模拟了我们的数据模型，我们如何让 `Mock` 对象引发一个异常呢？解决方案是使用模拟的 `side_effect` 属性，如下所示：

```py
 self.app.model.get_all_records.side_effect = Exception(
      'Test message'
    ) 
```

`side_effect` 可以用来模拟在模拟函数或方法中的更复杂的功能。它可以设置为一个函数，在这种情况下，模拟将在被调用时运行该函数并返回结果；它可以设置为一个可迭代对象，在这种情况下，模拟将在每次被调用时返回可迭代对象中的下一个项目；或者，正如这个例子中所示，它可以设置为一个异常，当模拟被调用时将引发该异常。

在我们能够使用它之前，我们需要按照以下方式修补 `messagebox`：

```py
 with patch('abq_data_entry.application.messagebox'):
      self.app._populate_recordlist()
      application.messagebox.showerror.assert_called_with(
        title='Error', message='Problem reading file',
        detail='Test message'
      ) 
```

这次当我们调用 `_populate_recordlist()` 时，我们的模拟 `CSVModel` 对象引发了一个异常，这应该会导致方法调用 `messagebox.showerror()`。由于我们已经模拟了 `showerror()`，我们可以使用 `assert_called_with()` 断言它以预期的参数被调用。

显然，测试我们的 `Application` 对象最困难的部分是修补所有模拟组件并确保它们足够像真实的东西以使 `Application` 满意。一旦我们做到了这一点，编写实际的测试就相对简单了。

## 测试我们的小部件

到目前为止，我们已经很好地使用 `patch()`、`Mock` 和默认的 `TestCase` 类测试了我们的组件，但测试我们的小部件模块将带来一些新的挑战。首先，我们的小部件需要一个 Tk 实例作为它们的根窗口。我们可以在每个案例的 `setUp()` 方法中创建这个实例，但这会显著减慢测试速度，而且这并不是真正必要的：我们的测试不会修改根窗口，所以每个测试用例只需要一个根窗口。为了保持测试以合理的速度运行，我们可以利用 `setUpClass()` 方法在测试用例实例创建时只创建一个 Tk 实例一次。

其次，我们有大量的小部件要测试，每个小部件都需要自己的 `TestCase` 类。因此，我们需要创建大量需要相同 Tk 设置和清理的测试用例。为了解决这个问题，我们将创建一个自定义的 `TestCase` 基类来处理根窗口的设置和清理，然后为每个小部件测试用例子类化它。在 `test` 目录下打开一个新文件，命名为 `test_widgets.py`，并从以下代码开始：

```py
# test_widgets.py
from .. import widgets
from unittest import TestCase
from unittest.mock import Mock
import tkinter as tk
from tkinter import ttk
class TkTestCase(TestCase):
  """A test case designed for Tkinter widgets and views"""
  @classmethod
  def setUpClass(cls):
    cls.root = tk.Tk()
    cls.root.wait_visibility()
  @classmethod
  def tearDownClass(cls):
    cls.root.update()
    cls.root.destroy() 
```

`setUpClass()` 方法创建 Tk 对象并调用 `wait_visibility()` 以确保在测试开始工作之前，根窗口是可见的并且完全绘制。我们还提供了一个互补的清理方法，该方法更新 Tk 实例（以完成队列中的任何事件）并销毁它。

现在，对于每个小部件测试用例，我们将子类化 `TkTestCase` 以确保我们为小部件有一个合适的测试环境。

### ValidatedSpinbox 小部件的单元测试

`ValidatedSpinbox` 是我们为应用程序创建的较为复杂的小部件之一，因此它是开始编写测试的好地方。

将 `TkTestCase` 类子类化以创建 `ValidatedSpinbox` 的测试用例，如下所示：

```py
class TestValidatedSpinbox(TkTestCase):
  def setUp(self):
    self.value = tk.DoubleVar()
    self.vsb = widgets.ValidatedSpinbox(
      self.root,
      textvariable=self.value,
      from_=-10, to=10, increment=1
    )
    self.vsb.pack()
    self.vsb.wait_visibility()
  def tearDown(self):
    self.vsb.destroy() 
```

我们的 `setUp()` 方法创建一个控制变量来存储小部件的值，然后创建一个具有一些基本设置的 `ValidatedSpinbox` 小部件实例：最小值为 `-10`，最大值为 `10`，增量值为 `1`。创建后，我们将其打包并等待其变得可见。对于我们的清理方法，我们只需销毁小部件。

现在，让我们开始编写测试。我们将从 `_key_validate()` 方法的单元测试开始：

```py
 def test_key_validate(self):
    for x in range(10):
      x = str(x)
      p_valid = self.vsb._key_validate(x, 'end', '', x, '1')
      n_valid = self.vsb._key_validate(x, 'end', '-', '-' + x, '1')
      self.assertTrue(p_valid)
      self.assertTrue(n_valid) 
```

在这个测试中，我们只是从 0 迭代到 9 并测试数字的正负值与 `_key_validate()`，它应该对所有这些值返回 `True`。

注意到 `_key_validate()` 方法接受许多位置参数，其中大部分是冗余的；可能有一个包装方法来使其更容易调用会很好，因为对这个函数的适当测试可能需要调用它几十次。

让我们称这个方法为 `key_validate()` 并将其添加到我们的 `TestValidatedSpinbox` 类中，如下所示：

```py
 def key_validate(self, new, current=''):
    return self.vsb._key_validate(
      new,  # inserted char
      'end',  # position to insert
      current,  # current value
      current + new,  # proposed value
      '1'  # action code (1 == insert)
    ) 
```

这将使未来对该方法的调用更短且更不容易出错。现在让我们使用这个方法来测试一些无效的输入，如下所示：

```py
 def test_key_validate_letters(self):
    valid = self.key_validate('a')
    self.assertFalse(valid)
  def test_key_validate_increment(self):
    valid = self.key_validate('1', '0.')
    self.assertFalse(valid)
  def test_key_validate_high(self):
    valid = self.key_validate('0', '10')
    self.assertFalse(valid)) 
```

在第一个例子中，我们输入字母 `a`；在第二个例子中，当框中已有 `0.` 时输入一个 `1` 字符（导致建议值为 `0.1`）；在第三个例子中，当框中有 `10` 时输入一个 `0` 字符（导致建议值为 `100`）。所有这些场景都应该使验证方法失败，导致它返回 `False`。

### 集成测试 ValidatedSpinbox 小部件

在前面的测试中，我们实际上并没有向小部件输入任何数据；我们只是直接调用键验证方法并评估其输出。这是好的单元测试，但作为对我们小部件功能性的测试，它并不令人满意，对吧？鉴于我们的自定义小部件与 Tkinter 的验证 API 深度交互，我们希望测试我们是否正确地与这个 API 接口。毕竟，*那个*方面的代码比我们验证方法中的实际逻辑更具挑战性。

我们可以通过创建一些模拟实际用户操作的集成测试来完成这项任务，然后检查这些操作的结果。为了干净利落地完成这项任务，我们首先需要创建一些支持方法。

首先，我们需要一种方法来模拟在窗口中输入文本。让我们在`TkTestCase`类中开始一个新的`type_in_widget()`方法来完成这个任务：

```py
# test_widgets.py, in TkTestCase
  def type_in_widget(self, widget, string):
    widget.focus_force() 
```

这种方法的第一步是迫使注意力集中在小部件上；回想一下，`focus_force()`即使在包含窗口没有焦点的情况下也会给小部件分配焦点；我们需要使用这个方法，因为我们的测试 Tk 窗口在测试运行时很可能没有焦点。

一旦我们获得焦点，我们就需要遍历字符串中的字符，并将原始字符转换为适当的事件序列键符号。回想一下，一些字符，尤其是符号，必须表示为名称字符串，例如`minus`或`colon`。为了使这可行，我们需要一种方法在字符和它们的键符号之间进行转换。我们可以通过添加一个类属性字典来实现这一点，如下所示：

```py
# test_widgets.py, in TkTestCase
  keysyms = {
    '-': 'minus',
    ' ': 'space',
    ':': 'colon',
  } 
```

更多键符号可以在[`www.tcl.tk/man/tcl8.4/TkCmd/keysyms.htm`](http://www.tcl.tk/man/tcl8.4/TkCmd/keysyms.htm)找到，但这些都足够了。让我们这样完成`type_in_widget()`方法：

```py
# test_widgets.py, in TkTestCase.type_in_widget()
    for char in string:
      char = self.keysyms.get(char, char)
      widget.event_generate(f'<KeyPress-{char}>')
      self.root.update_idletasks() 
```

在这个循环中，我们首先检查我们的`char`值在`keysyms`中是否有名称字符串。然后我们在小部件上生成一个带有给定字符或键符号的`KeyPress`事件。请注意，我们在生成按键事件后调用`self.root.update_idletasks()`。这确保了生成的按键字符在生成后能够注册。

除了模拟键盘输入外，我们还需要能够模拟鼠标点击。我们可以创建一个类似的方法，`click_on_widget()`，来模拟鼠标按钮点击，如下所示：

```py
 def click_on_widget(self, widget, x, y, button=1):
    widget.focus_force()
    widget.event_generate(f'<ButtonPress-{button}>', x=x, y=y)
    self.root.update_idletasks() 
```

此方法接受一个小部件、一个点击的`x`和`y`坐标，以及可选的将被点击的鼠标按钮（默认为`1`，即左鼠标按钮）。就像我们处理按键方法一样，我们首先强制焦点，生成我们的事件，然后更新应用程序。鼠标点击的`x`和`y`坐标指定了相对于小部件左上角的点击位置。

在这些方法就绪后，回到`TestValidatedSpinbox`类并编写一个新的测试：

```py
# test_widgets.py, in TestValidatedSpinbox
  def test__key_validate_integration(self):
    self.vsb.delete(0, 'end')
    self.type_in_widget(self.vsb, '10')
    self.assertEqual(self.vsb.get(), '10') 
```

此方法首先清除小部件，然后使用`type_in_widget()`模拟一些有效输入。然后我们使用`get()`从控件中检索值，检查它是否与预期值匹配。请注意，在这些集成测试中，我们需要每次都清除控件，因为我们正在模拟在真实控件中的按键操作并触发该操作的所有副作用。

接下来，让我们测试一些无效输入；在测试方法中添加以下内容：

```py
 self.vsb.delete(0, 'end')
    self.type_in_widget(self.vsb, 'abcdef')
    self.assertEqual(self.vsb.get(), '')
    self.vsb.delete(0, 'end')
    self.type_in_widget(self.vsb, '200')
    self.assertEqual(self.vsb.get(), '2') 
```

这次，我们模拟在控件中输入非数字或超出范围的值，并检查控件以确保它已正确拒绝无效的按键。在第一个例子中，`ValidatedSpinbox`应该拒绝所有按键，因为它们都是字母；在第二个例子中，只有初始的`2`应该被接受，因为随后的`0`按键会使数字超出范围。

我们可以使用我们的鼠标点击方法来测试`ValidatedSpinbox`小部件箭头按钮的功能。为了简化这个过程，我们可以在测试用例类中创建一个辅助方法来点击我们想要的箭头。当然，要点击特定的箭头，我们必须找出如何在小部件内定位该元素。

一种方法就是简单地估计一个硬编码的像素数。在大多数默认主题中，箭头位于框的右侧，框的高度大约为 20 像素。因此，这种方法可能可行：

```py
# test_widgets.py, inside TestValidatedSpinbox
  def click_arrow_naive(self, arrow='inc', times=1):
    x = self.vsb.winfo_width() – 5
    y = 5 if arrow == 'inc' else 15
    for _ in range(times):
      self.click_on_widget(self.vsb, x=x, y=y) 
```

这种方法实际上效果相当不错，可能足以满足您的需求。然而，由于它对您的主题和屏幕分辨率做出了假设，因此它有点脆弱。对于更复杂的自定义小部件，您可能很难通过这种方式定位元素。更好的方法可能是找到小部件元素的实际坐标。

不幸的是，Tkinter 小部件没有提供一种方法来定位小部件内元素的*x*和*y*坐标；然而，Ttk 元素提供了一个使用`identify()`方法查看给定坐标集下哪个元素的方法。使用这种方法，我们可以编写一个方法，遍历小部件以查找特定元素，并返回找到的第一个*x*和*y*坐标集。

让我们将这个方法作为静态方法添加到`TkTestCase`类中，如下所示：

```py
# test_widgets.py, inside TkTestCase
  @staticmethod
  def find_element(widget, element):
    widget.update_idletasks()
    x_coords = range(widget.winfo_width())
    y_coords = range(widget.winfo_height())
    for x in x_coords:
      for y in y_coords:
        if widget.identify(x, y) == element:
          return (x + 1, y + 1)
    raise Exception(f'{element} was not found in widget') 
```

此方法首先更新小部件的空闲任务。如果没有这个调用，所有元素可能还没有被绘制，`identify()`将返回一个空字符串。接下来，我们通过将小部件的宽度和高度传递给`range()`函数来获取小部件中所有*x*和*y*坐标的列表。我们遍历这些列表，在小部件的每个像素坐标上调用`widget.identify()`。如果返回的元素名称与我们正在寻找的元素名称匹配，我们就返回当前坐标作为一个元组。如果我们遍历整个小部件而没有返回，我们将引发一个异常，指出未找到元素。

注意，我们对每个 *x* 和 *y* 坐标都加上了 1；这是因为该元素返回小部件的左上角坐标。在某些情况下，点击这些角落坐标不会注册为对小部件的点击。为了确保我们实际上是在小部件内部点击，我们从角落向右和向下返回 1 像素的坐标。

当然，这里有一个问题：我们正在寻找的元素名称是什么？回想一下 *第九章*，*通过样式和主题改进外观*，组成小部件的元素由主题确定，不同的主题可能有完全不同的元素。例如，如果你正在寻找增加箭头元素，Windows 上的默认主题将其称为 `Spinbox.uparrow`。然而，Linux 上的默认主题简单地将其称为 `uparrow`，而 macOS 上的默认主题甚至没有为它提供单独的元素（两个箭头都是一个名为 `Spinbox.spinbutton` 的单个元素）！

为了解决这个问题，我们需要强制我们的测试窗口使用特定的主题，这样我们就可以依赖名称的一致性。在 `TestValidatedSpinbox.setUp()` 方法中，我们将添加一些代码来强制显式主题：

```py
# test_widgets.py, inside TestValidatedSpinbox.setUp()
    ttk.Style().theme_use('classic')
    self.vsb.update_idletasks() 
```

`classic` 主题应在所有平台上都可用，并且它使用简单的元素名称 `uparrow` 和 `downarrow` 作为 `Spinbox` 箭头元素。我们已经添加了对 `update_idletasks()` 的调用，以确保在测试开始之前主题更改已经在小部件中生效。

现在，我们可以为 `TestValidatedSpinbox` 编写一个更好的 `click_arrow()` 方法，该方法依赖于元素名称而不是硬编码的像素值。将此方法添加到类中：

```py
# test_widgets.py, inside TestValidatedSpinbox
  def click_arrow(self, arrow, times=1):
    element = f'{arrow}arrow'
    x, y = self.find_element(self.vsb, element)
    for _ in range(times):
      self.click_on_widget(self.vsb, x=x, y=y) 
```

就像我们的原始版本一样，这个方法接受一个箭头方向和次数。我们使用箭头方向来构建元素名称，然后使用我们的 `find_element()` 方法在 `ValidatedSpinbox` 小部件内定位适当的箭头。一旦我们有了坐标，我们就可以使用我们编写的 `click_on_widget()` 方法来点击它。

让我们将这种方法付诸实践，并在新的测试方法中测试我们的箭头键功能：

```py
# test_widgets.py, inside TestValidatedSpinbox
  def test_arrows(self):
    self.value.set(0)
    self.click_arrow('up', times=1)
    self.assertEqual(self.vsb.get(), '1')
    self.click_arrow('up', times=5)
    self.assertEqual(self.vsb.get(), '6')
    self.click_arrow(arrow='down', times=1)
    self.assertEqual(self.vsb.get(), '5') 
```

通过设置小部件的值，然后点击适当的箭头指定次数，我们可以测试箭头是否按照我们在小部件类中创建的规则完成了工作。

## 测试我们的混合类

我们还没有着手解决的一个额外挑战是测试我们的混合类。与我们的其他小部件类不同，我们的混合类不能独立存在：它依赖于与它结合的 Ttk 小部件中找到的方法和属性。

测试这个类的一个方法是将它与一个 `Mock` 对象混合，该对象模拟了任何继承的方法。这种方法有其优点，但一个更简单（如果理论纯度较低）的方法是用最简单的 `Ttk` 小部件子类化它，并测试生成的子类。

我们将创建一个使用后一种方法的测试用例。在 `test_widgets.py` 中启动它，如下所示：

```py
# test_widgets.py
class TestValidatedMixin(TkTestCase):
  def setUp(self):
    class TestClass(widgets.ValidatedMixin, ttk.Entry):
      pass
    self.vw1 = TestClass(self.root) 
```

在这里，`setUp()` 方法仅创建了一个 `ValidatedMixin` 和 `ttk.Entry` 的基本子类，没有其他修改，然后创建了其实例。

现在，让我们编写一个针对 `_validate()` 方法的测试用例，如下所示：

```py
 def test__validate(self):
    args = {
      'proposed': 'abc',
      'current': 'ab',
      'char': 'c',
      'event': 'key',
      'index': '2',
      'action': '1'
    }
    self.assertTrue(
      self.vw1._validate(**args)
    ) 
```

因为我们向 `_validate()` 发送了一个按键事件，它将请求路由到 `_key_validate()`，该函数默认简单地返回 `True`。我们需要验证当 `_key_validate()` 返回 `False` 时，`_validate()` 是否执行了所需操作。

我们将使用 `Mock` 来完成这个任务：

```py
 fake_key_val = Mock(return_value=False)
    self.vw1._key_validate = fake_key_val
    self.assertFalse(
      self.vw1._validate(**args)
    )
    fake_key_val.assert_called_with(**args) 
```

通过测试返回 `False` 并验证 `_key_validate()` 是否以正确的参数被调用，我们已经证明了 `_validate()` 正确地将事件路由到正确的验证方法。

通过更新 `args` 中的 `event` 值，我们可以检查焦点移出事件是否也正常工作：

```py
 args['event'] = 'focusout'
    self.assertTrue(self.vw1._validate(**args))
    fake_focusout_val = Mock(return_value=False)
    self.vw1._focusout_validate = fake_focusout_val
    self.assertFalse(self.vw1._validate(**args))
    fake_focusout_val.assert_called_with(event='focusout') 
```

我们在这里采取了相同的方法，只是模拟了 `_focusout_validate()` 函数，使其返回 `False`。

正如你所见，一旦我们创建了测试类，测试 `ValidatedMixin` 就像测试任何其他小部件类一样。包含的源代码中有其他测试方法示例；这些应该足以帮助你开始创建完整的测试套件。

# 摘要

在本章中，你了解了自动化测试的好处以及 Python 的 `unittest` 库提供的能力。你学习了如何使用 `Mock` 和 `patch()` 来替换外部模块、类和函数，从而隔离代码单元。你还学习了控制 Tkinter 事件队列和模拟用户输入以自动化测试我们的 GUI 组件的策略，并针对 ABQ 应用程序的部分编写了单元测试和集成测试。

在下一章中，我们将升级我们的后端以使用关系数据库。在这个过程中，你将了解关系数据库设计和数据规范化的知识。你还将学习如何与 PostgreSQL 数据库服务器以及 Python 的 `psycopg2` PostgreSQL 接口库一起工作。
