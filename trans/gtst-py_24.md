# 测试面向对象的程序

技术娴熟的 Python 程序员一致认为测试是软件开发中最重要的方面之一。即使这一章放在书的最后，它也不是一个事后补充；到目前为止我们学习的一切都将帮助我们在编写测试时。在本章中，我们将讨论以下主题：

+   单元测试和测试驱动开发的重要性

+   标准的`unittest`模块

+   `pytest`自动化测试套件

+   `mock`模块

+   代码覆盖率

+   使用`tox`进行跨平台测试

# 为什么要测试？

许多程序员已经知道测试他们的代码有多重要。如果你是其中之一，请随意略过本节。你会发现下一节——我们实际上如何在 Python 中创建测试——更加有趣。如果你还不相信测试的重要性，我保证你的代码是有问题的，只是你不知道而已。继续阅读！

有人认为在 Python 代码中测试更重要，因为它的动态特性；而像 Java 和 C++这样的编译语言偶尔被认为在编译时强制执行类型检查，所以在某种程度上更“安全”。然而，Python 测试很少检查类型。它们检查值。它们确保正确的属性在正确的时间设置，或者序列具有正确的长度、顺序和值。这些更高级的概念需要在任何语言中进行测试。

Python 程序员测试比其他语言的程序员更多的真正原因是在 Python 中测试是如此容易！

但是为什么要测试？我们真的需要测试吗？如果我们不测试会怎样？要回答这些问题，从头开始编写一个没有任何测试的井字棋游戏。在完全编写完成之前不要运行它，从头到尾。如果让两个玩家都是人类玩家（没有人工智能），井字棋实现起来相当简单。你甚至不必尝试计算谁是赢家。现在运行你的程序。然后修复所有的错误。有多少错误？我在我的井字棋实现中记录了八个，我不确定是否都捕捉到了。你呢？

我们需要测试我们的代码以确保它正常工作。像我们刚才做的那样运行程序并修复错误是一种粗糙的测试形式。Python 的交互式解释器和几乎零编译时间使得编写几行代码并运行程序以确保这些行正在按预期工作变得容易。但是改变几行代码可能会影响我们没有意识到会受到更改影响的程序的部分，因此忽略测试这些部分。此外，随着程序的增长，解释器可以通过代码的路径数量也在增加，手动测试所有这些路径很快就变得不可能。

为了解决这个问题，我们编写自动化测试。这些是自动运行某些输入通过其他程序或程序部分的程序。我们可以在几秒钟内运行这些测试程序，并覆盖比一个程序员每次更改某些东西时想到的潜在输入情况要多得多。

有四个主要原因要编写测试：

+   确保代码按照开发人员的预期工作

+   确保在进行更改时代码仍然正常工作

+   确保开发人员理解了需求

+   确保我们正在编写的代码具有可维护的接口

第一点真的不能证明写测试所花费的时间；我们可以在交互式解释器中直接测试代码，用同样或更少的时间。但是当我们必须多次执行相同的测试操作序列时，自动化这些步骤一次，然后在需要时运行它们需要的时间更少。每次更改代码时运行测试是个好主意，无论是在初始开发阶段还是在维护版本发布时。当我们有一套全面的自动化测试时，我们可以在代码更改后运行它们，并知道我们没有无意中破坏任何被测试的东西。

前面两点更有趣。当我们为代码编写测试时，它有助于设计代码所采用的 API、接口或模式。因此，如果我们误解了需求，编写测试可以帮助突出这种误解。另一方面，如果我们不确定如何设计一个类，我们可以编写一个与该类交互的测试，这样我们就可以知道与之交互的最自然方式。事实上，通常在编写我们要测试的代码之前编写测试是有益的。

# 测试驱动开发

*先写测试*是测试驱动开发的口头禅。测试驱动开发将*未经测试的代码是有问题的代码*的概念推进了一步，并建议只有未编写的代码才应该未经测试。在我们编写测试之前，我们不会编写任何代码来证明它有效。第一次运行测试时，它应该失败，因为代码还没有被编写。然后，我们编写确保测试通过的代码，然后为下一段代码编写另一个测试。

测试驱动开发很有趣；它允许我们构建小谜题来解决。然后，我们实现解决这些谜题的代码。然后，我们制作一个更复杂的谜题，然后编写解决新谜题的代码，而不会解决以前的谜题。

测试驱动方法有两个目标。第一个是确保测试真的被编写。在我们编写代码之后，很容易说：

嗯，看起来好像可以。我不需要为这个写任何测试。这只是一个小改变；什么都不可能出错。

如果测试在我们编写代码之前已经编写好了，我们将确切地知道它何时有效（因为测试将通过），并且在将来，如果我们或其他人对其进行了更改，我们将知道它是否被破坏。

其次，先编写测试迫使我们考虑代码将如何使用。它告诉我们对象需要具有哪些方法，以及如何访问属性。它帮助我们将初始问题分解为更小的、可测试的问题，然后将经过测试的解决方案重新组合成更大的、也经过测试的解决方案。编写测试因此可以成为设计过程的一部分。通常，当我们为一个新对象编写测试时，我们会发现设计中的异常，这迫使我们考虑软件的新方面。

作为一个具体的例子，想象一下编写使用对象关系映射器将对象属性存储在数据库中的代码。在这种对象中使用自动分配的数据库 ID 是很常见的。我们的代码可能会为各种目的使用这个 ID。如果我们为这样的代码编写测试，在我们编写测试之前，我们可能会意识到我们的设计有缺陷，因为对象在保存到数据库之前不会被分配 ID。如果我们想在测试中操作一个对象而不保存它，那么在我们基于错误的前提编写代码之前，它会突出显示这个问题。

测试使软件更好。在发布软件之前编写测试可以使软件在最终用户看到或购买有错误的版本之前变得更好（我曾为那些以*用户可以测试它*为理念的公司工作过；这不是一个健康的商业模式）。在编写软件之前编写测试可以使软件第一次编写时变得更好。

# 单元测试

让我们从 Python 内置的测试库开始探索。这个库为**单元测试**提供了一个通用的面向对象的接口。单元测试专注于在任何一个测试中测试尽可能少的代码。每个测试都测试可用代码的一个单元。

这个 Python 库的名称是`unittest`，毫不奇怪。它提供了几个用于创建和运行单元测试的工具，其中最重要的是`TestCase`类。这个类提供了一组方法，允许我们比较值，设置测试，并在测试完成时进行清理。

当我们想要为特定任务编写一组单元测试时，我们创建一个`TestCase`的子类，并编写单独的方法来进行实际测试。这些方法都必须以`test`开头的名称。遵循这个约定时，测试会自动作为测试过程的一部分运行。通常，测试会在对象上设置一些值，然后运行一个方法，并使用内置的比较方法来确保正确的结果被计算出来。这里有一个非常简单的例子：

```py
import unittest

class CheckNumbers(unittest.TestCase):
    def test_int_float(self):
        self.assertEqual(1, 1.0)
```

```py
if __name__ == "__main__":
    unittest.main()
```

这段代码简单地继承了`TestCase`类，并添加了一个调用`TestCase.assertEqual`方法的方法。这个方法将根据两个参数是否相等而成功或引发异常。如果我们运行这段代码，`unittest`的`main`函数将给出以下输出：

```py
.
--------------------------------------------------------------
Ran 1 test in 0.000s

OK  
```

你知道浮点数和整数可以被比较为相等吗？让我们添加一个失败的测试，如下：

```py
    def test_str_float(self): 
        self.assertEqual(1, "1") 
```

这段代码的输出更加阴险，因为整数和字符串不是

被认为是相等的：

```py
.F
============================================================
FAIL: test_str_float (__main__.CheckNumbers)
--------------------------------------------------------------
Traceback (most recent call last):
 File "first_unittest.py", line 9, in test_str_float
 self.assertEqual(1, "1")
AssertionError: 1 != '1'

--------------------------------------------------------------
Ran 2 tests in 0.001s

FAILED (failures=1)  
```

第一行的点表示第一个测试（我们之前写的那个）成功通过；其后的字母`F`表示第二个测试失败。然后，在最后，它会给出一些信息性的输出，告诉我们测试失败的原因和位置，以及失败的数量总结。

我们可以在一个`TestCase`类上有尽可能多的测试方法。只要方法名以`test`开头，测试运行器就会将每个方法作为一个单独的、隔离的测试执行。每个测试应该完全独立于其他测试。先前测试的结果或计算不应该对当前测试产生影响。编写良好的单元测试的关键是尽可能保持每个测试方法的长度短小，每个测试用例测试一小部分代码。如果我们的代码似乎无法自然地分解成这样可测试的单元，这可能是代码需要重新设计的迹象。

# 断言方法

测试用例的一般布局是将某些变量设置为已知的值，运行一个或多个函数、方法或进程，然后使用`TestCase`的断言方法*证明*正确的预期结果是通过的或者被计算出来的。

有几种不同的断言方法可用于确认已经实现了特定的结果。我们刚刚看到了`assertEqual`，如果两个参数不能通过相等检查，它将导致测试失败。相反，`assertNotEqual`如果两个参数比较为相等，则会失败。`assertTrue`和`assertFalse`方法分别接受一个表达式，并且如果表达式不能通过`if`测试，则会失败。这些测试不检查布尔值`True`或`False`。相反，它们测试与使用`if`语句相同的条件：`False`、`None`、`0`或空列表、字典、字符串、集合或元组会通过调用`assertFalse`方法。非零数、包含值的容器，或值`True`在调用`assertTrue`方法时会成功。

有一个`assertRaises`方法，可以用来确保特定的函数调用引发特定的异常，或者可以选择作为上下文管理器来包装内联代码。如果`with`语句内的代码引发了正确的异常，则测试通过；否则，测试失败。以下代码片段是两个版本的示例：

```py
import unittest

def average(seq):
    return sum(seq) / len(seq)

class TestAverage(unittest.TestCase):
    def test_zero(self):
        self.assertRaises(ZeroDivisionError, average, [])

    def test_with_zero(self):
        with self.assertRaises(ZeroDivisionError):
            average([])

if __name__ == "__main__":
    unittest.main()
```

上下文管理器允许我们以通常的方式编写代码（通过调用函数或直接执行代码），而不必在另一个函数调用中包装函数调用。

还有几种其他断言方法，总结在下表中：

| **方法** | **描述** |
| --- | --- |
| `assertGreater``assertGreaterEqual``assertLess``assertLessEqual` | 接受两个可比较的对象，并确保命名的不等式成立。 |
| `assertIn``assertNotIn` | 确保元素是（或不是）容器对象中的一个元素。 |
| `assertIsNone``assertIsNotNone` | 确保一个元素是（或不是）确切的`None`值（而不是其他假值）。 |
| `assertSameElements` | 确保两个容器对象具有相同的元素，忽略顺序。 |
| `assertSequenceEqualassertDictEqual``assertSetEqual``assertListEqual``assertTupleEqual` | 确保两个容器以相同的顺序具有相同的元素。如果失败，显示一个比较两个列表的代码差异，以查看它们的不同之处。最后四种方法还测试了列表的类型。 |

每个断言方法都接受一个名为`msg`的可选参数。如果提供了，它将包含在错误消息中，如果断言失败，这对于澄清预期的内容或解释可能导致断言失败的错误的地方非常有用。然而，我很少使用这种语法，更喜欢为测试方法使用描述性的名称。

# 减少样板代码和清理

编写了一些小测试之后，我们经常发现我们必须为几个相关的测试编写相同的设置代码。例如，以下`list`子类有三种用于统计计算的方法：

```py
from collections import defaultdict 

class StatsList(list): 
    def mean(self): 
        return sum(self) / len(self) 

    def median(self): 
        if len(self) % 2: 
            return self[int(len(self) / 2)] 
        else: 
            idx = int(len(self) / 2) 
            return (self[idx] + self[idx-1]) / 2 

    def mode(self): 
        freqs = defaultdict(int) 
        for item in self: 
            freqs[item] += 1 
        mode_freq = max(freqs.values()) 
        modes = [] 
        for item, value in freqs.items(): 
            if value == mode_freq: 
                modes.append(item) 
        return modes 
```

显然，我们将要测试这三种方法中的每一种情况，这些情况具有非常相似的输入。我们将要看到空列表、包含非数字值的列表，或包含正常数据集的列表等情况下会发生什么。我们可以使用`TestCase`类上的`setUp`方法来为每个测试执行初始化。这个方法不接受任何参数，并允许我们在每个测试运行之前进行任意的设置。例如，我们可以在相同的整数列表上测试所有三种方法，如下所示：

```py
from stats import StatsList
import unittest

class TestValidInputs(unittest.TestCase):
    def setUp(self):
        self.stats = StatsList([1, 2, 2, 3, 3, 4])

    def test_mean(self):
        self.assertEqual(self.stats.mean(), 2.5)

    def test_median(self):
        self.assertEqual(self.stats.median(), 2.5)
        self.stats.append(4)
        self.assertEqual(self.stats.median(), 3)

    def test_mode(self):
        self.assertEqual(self.stats.mode(), [2, 3])
        self.stats.remove(2)
        self.assertEqual(self.stats.mode(), [3])

if __name__ == "__main__":
    unittest.main()
```

如果我们运行这个例子，它表明所有测试都通过了。首先注意到`setUp`方法从未在三个`test_*`方法中显式调用过。测试套件会代表我们执行这个操作。更重要的是，注意`test_median`如何改变了列表，通过向其中添加一个额外的`4`，但是当随后调用`test_mode`时，列表已经恢复到了`setUp`中指定的值。如果没有恢复，列表中将会有两个四，而`mode`方法将会返回三个值。这表明`setUp`在每个测试之前都会被单独调用，确保测试类从一个干净的状态开始。测试可以以任何顺序执行，一个测试的结果绝不能依赖于其他测试。

除了`setUp`方法，`TestCase`还提供了一个无参数的`tearDown`方法，它可以用于在类的每个测试运行后进行清理。如果清理需要除了让对象被垃圾回收之外的其他操作，这个方法就很有用。

例如，如果我们正在测试进行文件 I/O 的代码，我们的测试可能会在测试的副作用下创建新文件。`tearDown`方法可以删除这些文件，并确保系统处于与测试运行之前相同的状态。测试用例绝不能有副作用。通常，我们根据它们共同的设置代码将测试方法分组到单独的`TestCase`子类中。需要相同或相似设置的几个测试将被放置在一个类中，而需要不相关设置的测试将被放置在另一个类中。

# 组织和运行测试

对于一个单元测试集合来说，很快就会变得非常庞大和难以控制。一次性加载和运行所有测试可能会变得非常复杂。这是单元测试的主要目标：在程序上轻松运行所有测试，并快速得到一个“是”或“否”的答案，来回答“我的最近的更改是否有问题？”的问题。

与正常的程序代码一样，我们应该将测试类分成模块和包，以保持它们的组织。如果您将每个测试模块命名为以四个字符*test*开头，就可以轻松找到并运行它们。Python 的`discover`模块会查找当前文件夹或子文件夹中以`test`开头命名的任何模块。如果它在这些模块中找到任何`TestCase`对象，就会执行测试。这是一种无痛的方式来确保我们不会错过运行任何测试。要使用它，请确保您的测试模块命名为`test_<something>.py`，然后运行`python3 -m unittest discover`命令。

大多数 Python 程序员选择将他们的测试放在一个单独的包中（通常命名为`tests/`，与他们的源目录并列）。但这并不是必需的。有时，将不同包的测试模块放在该包旁边的子包中是有意义的，例如。

# 忽略损坏的测试

有时，我们知道测试会失败，但我们不希望测试套件报告失败。这可能是因为一个损坏或未完成的功能已经编写了测试，但我们目前并不专注于改进它。更常见的情况是，因为某个功能仅在特定平台、Python 版本或特定库的高级版本上可用。Python 为我们提供了一些装饰器，用于标记测试为预期失败或在已知条件下跳过。

这些装饰器如下：

+   `expectedFailure()`

+   `skip(reason)`

+   `skipIf(condition, reason)`

+   `skipUnless(condition, reason)`

这些是使用 Python 装饰器语法应用的。第一个不接受参数，只是告诉测试运行器在测试失败时不记录测试失败。`skip`方法更进一步，甚至不会运行测试。它期望一个描述为什么跳过测试的字符串参数。另外两个装饰器接受两个参数，一个是布尔表达式，指示是否应该运行测试，另一个是类似的描述。在使用时，这三个装饰器可能会像下面的代码中所示一样应用：

```py
import unittest
import sys

class SkipTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_fails(self):
        self.assertEqual(False, True)

    @unittest.skip("Test is useless")
    def test_skip(self):
        self.assertEqual(False, True)

    @unittest.skipIf(sys.version_info.minor == 4, "broken on 3.4")
    def test_skipif(self):
        self.assertEqual(False, True)

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "broken unless on linux"
    )
    def test_skipunless(self):
        self.assertEqual(False, True)

if __name__ == "__main__":
    unittest.main()
```

第一个测试失败，但被报告为预期的失败；第二个测试从未运行。其他两个测试可能会运行，也可能不会，这取决于当前的 Python 版本和操作系统。在我的 Linux 系统上，运行 Python 3.7，输出如下：

```py
xssF
======================================================================
FAIL: test_skipunless (__main__.SkipTests)
----------------------------------------------------------------------
Traceback (most recent call last):
 File "test_skipping.py", line 22, in test_skipunless
 self.assertEqual(False, True)
AssertionError: False != True

----------------------------------------------------------------------
Ran 4 tests in 0.001s

FAILED (failures=1, skipped=2, expected failures=1)
```

第一行上的`x`表示预期的失败；两个`s`字符表示跳过的测试，`F`表示真正的失败，因为在我的系统上`skipUnless`的条件为`True`。

# 使用 pytest 进行测试

Python 的`unittest`模块需要大量样板代码来设置和初始化测试。它基于非常流行的 Java 的 JUnit 测试框架。它甚至使用相同的方法名称（您可能已经注意到它们不符合 PEP-8 命名标准，该标准建议使用 snake_case 而不是 CamelCase 来表示方法名称）和测试布局。虽然这对于在 Java 中进行测试是有效的，但不一定是 Python 测试的最佳设计。我实际上发现`unittest`框架是过度使用面向对象原则的一个很好的例子。

因为 Python 程序员喜欢他们的代码简洁而简单，所以在标准库之外开发了其他测试框架。其中两个较受欢迎的是`pytest`和`nose`。前者更为健壮，并且支持 Python 3 的时间更长，因此我们将在这里讨论它。

由于`pytest`不是标准库的一部分，您需要自己下载并安装它。您可以从[`pytest.org/`](http://pytest.org/)的`pytest`主页获取它。该网站提供了各种解释器和平台的全面安装说明，但通常您可以使用更常见的 Python 软件包安装程序 pip。只需在命令行上输入`pip install pytest`，就可以开始使用了。

`pytest`的布局与`unittest`模块有很大不同。它不要求测试用例是类。相反，它利用了 Python 函数是对象的事实，并允许任何命名正确的函数像测试一样行为。它不是提供一堆用于断言相等的自定义方法，而是使用`assert`语句来验证结果。这使得测试更易读和易维护。

当我们运行`pytest`时，它会从当前文件夹开始搜索以`test_`开头的任何模块或子包。如果该模块中的任何函数也以`test`开头，它们将作为单独的测试执行。此外，如果模块中有任何以`Test`开头的类，该类上以`test_`开头的任何方法也将在测试环境中执行。

使用以下代码，让我们将之前编写的最简单的`unittest`示例移植到`pytest`：

```py
def test_int_float(): 
    assert 1 == 1.0 
```

对于完全相同的测试，我们写了两行更易读的代码，而不是我们第一个`unittest`示例中需要的六行。

但是，我们并没有禁止编写基于类的测试。类可以用于将相关测试分组在一起，或者用于需要访问类上相关属性或方法的测试。下面的示例显示了一个扩展类，其中包含一个通过和一个失败的测试；我们将看到错误输出比`unittest`模块提供的更全面：

```py
class TestNumbers: 
    def test_int_float(self): 
        assert 1 == 1.0 

    def test_int_str(self): 
        assert 1 == "1" 
```

请注意，类不必扩展任何特殊对象才能被识别为测试（尽管`pytest`可以很好地运行标准的`unittest TestCases`）。如果我们运行`pytest <filename>`，输出如下所示：

```py
============================== test session starts ==============================
platform linux -- Python 3.7.0, pytest-3.8.0, py-1.6.0, pluggy-0.7.1
rootdir: /home/dusty/Py3OOP/Chapter 24: Testing Object-oriented Programs, inifile:
collected 3 items

test_with_pytest.py ..F [100%]

=================================== FAILURES ====================================
___________________________ TestNumbers.test_int_str ____________________________

self = <test_with_pytest.TestNumbers object at 0x7fdb95e31390>

 def test_int_str(self):
> assert 1 == "1"
E AssertionError: assert 1 == '1'

test_with_pytest.py:10: AssertionError
====================== 1 failed, 2 passed in 0.03 seconds =======================
```

输出以有关平台和解释器的一些有用信息开始。这对于在不同系统之间共享或讨论错误很有用。第三行告诉我们正在测试的文件的名称（如果有多个测试模块被识别，它们都将显示出来），然后是在`unittest`模块中看到的熟悉的`.F`；`.`字符表示通过的测试，而字母`F`表示失败。

所有测试运行完毕后，将显示每个测试的错误输出。它呈现了局部变量的摘要（在本例中只有一个：传递给函数的`self`参数），发生错误的源代码以及错误消息的摘要。此外，如果引发的异常不是`AssertionError`，`pytest`将向我们呈现完整的回溯，包括源代码引用。

默认情况下，如果测试成功，`pytest`会抑制`print`语句的输出。这对于测试调试很有用；当测试失败时，我们可以向测试中添加`print`语句来检查特定变量和属性的值。如果测试失败，这些值将被输出以帮助诊断。但是，一旦测试成功，`print`语句的输出就不会显示出来，很容易被忽略。我们不必通过删除`print`语句来*清理*输出。如果由于将来的更改而再次失败，调试输出将立即可用。

# 进行设置和清理的一种方法

`pytest`支持类似于`unittest`中使用的设置和拆卸方法，但它提供了更多的灵活性。我们将简要讨论这些，因为它们很熟悉，但它们并没有像在`unittest`模块中那样被广泛使用，因为`pytest`为我们提供了一个强大的固定设施，我们将在下一节中讨论。

如果我们正在编写基于类的测试，我们可以使用两个名为`setup_method`和`teardown_method`的方法，就像在`unittest`中调用`setUp`和`tearDown`一样。它们在类中的每个测试方法之前和之后被调用，以执行设置和清理任务。但是，与`unittest`方法不同的是，这两种方法都接受一个参数：表示被调用的方法的函数对象。

此外，`pytest`提供了其他设置和拆卸函数，以便更好地控制设置和清理代码的执行时间。`setup_class`和`teardown_class`方法预期是类方法；它们接受一个表示相关类的单个参数（没有`self`参数）。这些方法仅在类被初始化时运行，而不是在每次测试运行时运行。

最后，我们有`setup_module`和`teardown_module`函数，它们在该模块中的所有测试（在函数或类中）之前和之后立即运行。这些可以用于*一次性*设置，例如创建一个将被模块中所有测试使用的套接字或数据库连接。对于这一点要小心，因为如果对象存储了在测试之间没有正确清理的状态，它可能会意外地引入测试之间的依赖关系。

这个简短的描述并没有很好地解释这些方法究竟在什么时候被调用，所以让我们看一个例子，确切地说明了它们何时被调用：

```py
def setup_module(module):
    print("setting up MODULE {0}".format(module.__name__))

def teardown_module(module):
    print("tearing down MODULE {0}".format(module.__name__))

def test_a_function():
    print("RUNNING TEST FUNCTION")

class BaseTest:
    def setup_class(cls):
        print("setting up CLASS {0}".format(cls.__name__))

    def teardown_class(cls):
        print("tearing down CLASS {0}\n".format(cls.__name__))

    def setup_method(self, method):
        print("setting up METHOD {0}".format(method.__name__))

    def teardown_method(self, method):
        print("tearing down METHOD {0}".format(method.__name__))

class TestClass1(BaseTest):
    def test_method_1(self):
        print("RUNNING METHOD 1-1")

    def test_method_2(self):
        print("RUNNING METHOD 1-2")

class TestClass2(BaseTest):
    def test_method_1(self):
        print("RUNNING METHOD 2-1")

    def test_method_2(self):
        print("RUNNING METHOD 2-2")
```

`BaseTest`类的唯一目的是提取四个方法，否则这些方法与测试类相同，并使用继承来减少重复代码的数量。因此，从`pytest`的角度来看，这两个子类不仅每个有两个测试方法，还有两个设置和两个拆卸方法（一个在类级别，一个在方法级别）。

如果我们使用`pytest`运行这些测试，并且禁用了`print`函数的输出抑制（通过传递`-s`或`--capture=no`标志），它们会告诉我们各种函数在与测试本身相关的时候被调用：

```py
setup_teardown.py
setting up MODULE setup_teardown
RUNNING TEST FUNCTION
.setting up CLASS TestClass1
setting up METHOD test_method_1
RUNNING METHOD 1-1
.tearing down  METHOD test_method_1
setting up METHOD test_method_2
RUNNING METHOD 1-2
.tearing down  METHOD test_method_2
tearing down CLASS TestClass1
setting up CLASS TestClass2
setting up METHOD test_method_1
RUNNING METHOD 2-1
.tearing down  METHOD test_method_1
setting up METHOD test_method_2
RUNNING METHOD 2-2
.tearing down  METHOD test_method_2
tearing down CLASS TestClass2

tearing down MODULE setup_teardown  
```

模块的设置和拆卸方法在会话开始和结束时执行。然后运行单个模块级别的测试函数。接下来，执行第一个类的设置方法，然后是该类的两个测试。这些测试分别包装在单独的`setup_method`和`teardown_method`调用中。测试执行完毕后，调用类的拆卸方法。在第二个类之前，发生了相同的顺序，最后调用`teardown_module`方法，确切地一次。

# 设置变量的完全不同的方法

各种设置和拆卸函数的最常见用途之一是确保在运行每个测试方法之前，某些类或模块变量可用且具有已知值。

`pytest`提供了一个完全不同的设置变量的方法，使用所谓的**fixtures**。Fixture 基本上是预定义在测试配置文件中的命名变量。这允许我们将配置与测试的执行分开，并允许 fixtures 在多个类和模块中使用。

为了使用它们，我们向我们的测试函数添加参数。参数的名称用于在特别命名的函数中查找特定的参数。例如，如果我们想测试我们在演示`unittest`时使用的`StatsList`类，我们再次想要重复测试一个有效整数列表。但是，我们可以编写我们的测试如下，而不是使用设置方法：

```py
import pytest
from stats import StatsList

@pytest.fixture
def valid_stats():
    return StatsList([1, 2, 2, 3, 3, 4])

def test_mean(valid_stats):
    assert valid_stats.mean() == 2.5

def test_median(valid_stats):
    assert valid_stats.median() == 2.5
    valid_stats.append(4)
    assert valid_stats.median() == 3

def test_mode(valid_stats):
    assert valid_stats.mode() == [2, 3]
    valid_stats.remove(2)
    assert valid_stats.mode() == [3]
```

这三个测试方法中的每一个都接受一个名为`valid_stats`的参数；这个参数是通过调用`valid_stats`函数创建的，该函数被装饰为`@pytest.fixture`。

Fixture 可以做的远不止返回基本变量。可以将`request`对象传递到 fixture 工厂中，以提供非常有用的方法和属性来修改 funcarg 的行为。`module`、`cls`和`function`属性允许我们准确地查看请求 fixture 的测试。`config`属性允许我们检查命令行参数和大量其他配置数据。

如果我们将 fixture 实现为生成器，我们可以在每次测试运行后运行清理代码。这提供了类似于拆卸方法的功能，但是在每个 fixture 的基础上。我们可以用它来清理文件、关闭连接、清空列表或重置队列。例如，以下代码测试了`os.mkdir`功能，通过创建一个临时目录 fixture：

```py
import pytest
import tempfile
import shutil
import os.path

@pytest.fixture
def temp_dir(request):
    dir = tempfile.mkdtemp()
    print(dir)
    yield dir
    shutil.rmtree(dir)

def test_osfiles(temp_dir):
    os.mkdir(os.path.join(temp_dir, "a"))
    os.mkdir(os.path.join(temp_dir, "b"))
    dir_contents = os.listdir(temp_dir)
    assert len(dir_contents) == 2
    assert "a" in dir_contents
    assert "b" in dir_contents
```

该 fixture 为文件创建一个新的空临时目录。它将此目录提供给测试使用，但在测试完成后删除该目录（使用`shutil.rmtree`，递归删除目录及其中的所有内容）。文件系统将保持与开始时相同的状态。

我们可以传递一个`scope`参数来创建一个持续时间超过一个测试的 fixture。当设置一个昂贵的操作，可以被多个测试重复使用时，这是很有用的，只要资源重用不会破坏测试的原子性或单元性（以便一个测试不依赖于前一个测试，也不受其影响）。例如，如果我们要测试以下回显服务器，我们可能只想在单独的进程中运行一个服务器实例，然后让多个测试连接到该实例：

```py
import socket 

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
s.bind(('localhost',1028)) 
s.listen(1) 

    while True: 
        client, address = s.accept() 
        data = client.recv(1024) 
        client.send(data) 
        client.close() 
```

这段代码的作用只是监听特定端口，并等待来自客户端 socket 的输入。当它接收到输入时，它会将相同的值发送回去。为了测试这个，我们可以在单独的进程中启动服务器，并缓存结果供多个测试使用。测试代码可能如下所示：

```py
import subprocess
import socket
import time
import pytest

@pytest.fixture(scope="session")
def echoserver():
    print("loading server")
    p = subprocess.Popen(["python3", "echo_server.py"])
    time.sleep(1)
    yield p
    p.terminate()

@pytest.fixture
def clientsocket(request):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("localhost", 1028))
    yield s
    s.close()

def test_echo(echoserver, clientsocket):
    clientsocket.send(b"abc")
    assert clientsocket.recv(3) == b"abc"

def test_echo2(echoserver, clientsocket):
    clientsocket.send(b"def")
    assert clientsocket.recv(3) == b"def"
```

我们在这里创建了两个 fixtures。第一个在单独的进程中运行回显服务器，并在完成时清理进程对象。第二个为每个测试实例化一个新的 socket 对象，并在测试完成时关闭 socket。

第一个 fixture 是我们目前感兴趣的。通过传递给装饰器构造函数的`scope="session"`关键字参数，`pytest`知道我们只希望在单元测试会话期间初始化和终止一次这个 fixture。

作用域可以是字符串`class`、`module`、`package`或`session`中的一个。它决定了参数将被缓存多长时间。在这个例子中，我们将其设置为`session`，因此它将在整个`pytest`运行期间被缓存。进程将在所有测试运行完之前不会被终止或重新启动。当然，`module`作用域仅为该模块中的测试缓存，`class`作用域将对象视为普通的类设置和拆卸。

在本书第三版印刷时，`pytest`中的`package`作用域被标记为实验性质。请小心使用，并要求您提供 bug 报告。

# 使用 pytest 跳过测试

与`unittest`模块一样，经常需要在`pytest`中跳过测试，原因各种各样：被测试的代码尚未编写，测试仅在某些解释器或操作系统上运行，或者测试耗时且只应在特定情况下运行。

我们可以在代码的任何地方跳过测试，使用`pytest.skip`函数。它接受一个参数：描述为什么要跳过的字符串。这个函数可以在任何地方调用。如果我们在测试函数内调用它，测试将被跳过。如果我们在模块级别调用它，那个模块中的所有测试都将被跳过。如果我们在 fixture 内调用它，所有调用该 funcarg 的测试都将被跳过。

当然，在所有这些位置，通常希望只有在满足或不满足某些条件时才跳过测试。由于我们可以在 Python 代码的任何地方执行`skip`函数，我们可以在`if`语句内执行它。因此，我们可能编写一个如下所示的测试：

```py
import sys 
import pytest 

def test_simple_skip(): 
    if sys.platform != "fakeos": 
        pytest.skip("Test works only on fakeOS") 

    fakeos.do_something_fake() 
    assert fakeos.did_not_happen 
```

这实际上是一些相当愚蠢的代码。没有名为`fakeos`的 Python 平台，因此这个测试将在所有操作系统上跳过。它展示了我们如何有条件地跳过测试，由于`if`语句可以检查任何有效的条件，我们对测试何时被跳过有很大的控制权。通常，我们检查`sys.version_info`来检查 Python 解释器版本，`sys.platform`来检查操作系统，或者`some_library.__version__`来检查我们是否有足够新的给定 API 版本。

由于基于某个条件跳过单个测试方法或函数是测试跳过的最常见用法之一，`pytest`提供了一个方便的装饰器，允许我们在一行中执行此操作。装饰器接受一个字符串，其中可以包含任何可执行的 Python 代码，该代码求值为布尔值。例如，以下测试只在 Python 3 或更高版本上运行：

```py
@pytest.mark.skipif("sys.version_info <= (3,0)") 
def test_python3(): 
    assert b"hello".decode() == "hello" 
```

`pytest.mark.xfail`装饰器的行为类似，只是它标记一个测试预期失败，类似于`unittest.expectedFailure()`。如果测试成功，它将被记录为失败。如果失败，它将被报告为预期行为。在`xfail`的情况下，条件参数是可选的。如果没有提供，测试将被标记为在所有条件下都预期失败。

`pytest`除了这里描述的功能之外，还有很多其他功能，开发人员不断添加创新的新方法，使您的测试体验更加愉快。他们在网站上有详尽的文档[`docs.pytest.org/`](https://docs.pytest.org/)。

`pytest`可以找到并运行使用标准`unittest`库定义的测试，除了它自己的测试基础设施。这意味着如果你想从`unittest`迁移到`pytest`，你不必重写所有旧的测试。

# 模拟昂贵的对象

有时，我们想要测试需要提供一个昂贵或难以构建的对象的代码。在某些情况下，这可能意味着您的 API 需要重新思考，以具有更可测试的接口（通常意味着更可用的接口）。但我们有时发现自己编写的测试代码有大量样板代码来设置与被测试代码只是偶然相关的对象。

例如，想象一下我们有一些代码，它在外部键值存储中（如`redis`或`memcache`）跟踪航班状态，以便我们可以存储时间戳和最新状态。这样的基本版本代码可能如下所示：

```py
import datetime
import redis

class FlightStatusTracker:
    ALLOWED_STATUSES = {"CANCELLED", "DELAYED", "ON TIME"}

    def __init__(self):
        self.redis = redis.StrictRedis()

    def change_status(self, flight, status):
        status = status.upper()
        if status not in self.ALLOWED_STATUSES:
            raise ValueError("{} is not a valid status".format(status))

        key = "flightno:{}".format(flight)
        value = "{}|{}".format(
            datetime.datetime.now().isoformat(), status
        )
        self.redis.set(key, value)
```

有很多我们应该为`change_status`方法测试的事情。我们应该检查如果传入了错误的状态，它是否引发了适当的错误。我们需要确保它将状态转换为大写。我们可以看到当在`redis`对象上调用`set()`方法时，键和值的格式是否正确。

然而，在我们的单元测试中，我们不必检查`redis`对象是否正确存储数据。这是绝对应该在集成或应用程序测试中进行测试的事情，但在单元测试级别，我们可以假设 py-redis 开发人员已经测试过他们的代码，并且这个方法可以按我们的要求工作。一般来说，单元测试应该是自包含的，不应依赖于外部资源的存在，比如运行中的 Redis 实例。

相反，我们只需要测试`set()`方法被调用的次数和使用的参数是否正确。我们可以在测试中使用`Mock()`对象来替换麻烦的方法，以便我们可以内省对象。以下示例说明了`Mock`的用法：

```py
from flight_status_redis import FlightStatusTracker
from unittest.mock import Mock
import pytest

@pytest.fixture
def tracker():
    return FlightStatusTracker()

def test_mock_method(tracker):
 tracker.redis.set = Mock()
    with pytest.raises(ValueError) as ex:
        tracker.change_status("AC101", "lost")
    assert ex.value.args[0] == "LOST is not a valid status"
 assert tracker.redis.set.call_count == 0

```

这个使用`pytest`语法编写的测试断言在传入不合适的参数时会引发正确的异常。此外，它为`set`方法创建了一个`Mock`对象，并确保它从未被调用。如果被调用了，这意味着我们的异常处理代码中存在错误。

在这种情况下，简单地替换方法效果很好，因为被替换的对象最终被销毁了。然而，我们经常希望仅在测试期间替换函数或方法。例如，如果我们想测试`Mock`方法中的时间戳格式，我们需要确切地知道`datetime.datetime.now()`将返回什么。然而，这个值会随着运行的不同而改变。我们需要一种方法将其固定到一个特定的值，以便我们可以进行确定性测试。

临时将库函数设置为特定值是猴子补丁的少数有效用例之一。模拟库提供了一个补丁上下文管理器，允许我们用模拟对象替换现有库上的属性。当上下文管理器退出时，原始属性会自动恢复，以免影响其他测试用例。以下是一个例子：

```py
import datetime
from unittest.mock import patch

def test_patch(tracker):
    tracker.redis.set = Mock()
    fake_now = datetime.datetime(2015, 4, 1)
 with patch("datetime.datetime") as dt:
        dt.now.return_value = fake_now
        tracker.change_status("AC102", "on time")
    dt.now.assert_called_once_with()
    tracker.redis.set.assert_called_once_with(
        "flightno:AC102", "2015-04-01T00:00:00|ON TIME"
    )
```

在前面的例子中，我们首先构造了一个名为`fake_now`的值，我们将其设置为`datetime.datetime.now`函数的返回值。我们必须在补丁`datetime.datetime`之前构造这个对象，否则我们会在构造它之前调用已经补丁的`now`函数。

`with`语句邀请补丁用模拟对象替换`datetime.datetime`模块，返回为`dt`值。模拟对象的好处是，每次访问该对象的属性或方法时，它都会返回另一个模拟对象。因此，当我们访问`dt.now`时，它会给我们一个新的模拟对象。我们将该对象的`return_value`设置为我们的`fake_now`对象。现在，每当调用`datetime.datetime.now`函数时，它将返回我们的对象，而不是一个新的模拟对象。但是当解释器退出上下文管理器时，原始的`datetime.datetime.now()`功能会被恢复。

在使用已知值调用我们的`change_status`方法后，我们使用`Mock`类的`assert_called_once_with`函数来确保`now`函数确实被调用了一次，且没有参数。然后我们再次调用它，以证明`redis.set`方法被调用时，参数的格式与我们期望的一样。

模拟日期以便获得确定性的测试结果是一个常见的补丁场景。如果你处于这种情况，你可能会喜欢 Python 包索引中提供的`freezegun`和`pytest-freezegun`项目。

前面的例子很好地说明了编写测试如何指导我们的 API 设计。`FlightStatusTracker`对象乍一看似乎很合理；我们在对象构造时构建了一个`redis`连接，并在需要时调用它。然而，当我们为这段代码编写测试时，我们发现即使我们在`FlightStatusTracker`上模拟了`self.redis`变量，`redis`连接仍然必须被构造。如果没有运行 Redis 服务器，这个调用实际上会失败，我们的测试也会失败。

我们可以通过在`setUp`方法中模拟`redis.StrictRedis`类来解决这个问题，以返回一个模拟对象。然而，一个更好的想法可能是重新思考我们的实现。与其在`__init__`中构造`redis`实例，也许我们应该允许用户传入一个，就像下面的例子一样：

```py
    def __init__(self, redis_instance=None): 
        self.redis = redis_instance if redis_instance else redis.StrictRedis() 
```

这样我们就可以在测试时传入一个模拟对象，这样`StrictRedis`方法就不会被构造。此外，它允许任何与`FlightStatusTracker`交互的客户端代码传入他们自己的`redis`实例。他们可能有各种原因这样做：他们可能已经为代码的其他部分构造了一个；他们可能已经创建了`redis` API 的优化实现；也许他们有一个将指标记录到内部监控系统的实现。通过编写单元测试，我们发现了一个使用案例，使我们的 API 从一开始就更加灵活，而不是等待客户要求我们支持他们的异类需求。

这是对模拟代码奇迹的简要介绍。自 Python 3.3 以来，模拟是标准的`unittest`库的一部分，但正如你从这些例子中看到的，它们也可以与`pytest`和其他库一起使用。模拟还有其他更高级的功能，你可能需要利用这些功能，因为你的代码变得更加复杂。例如，你可以使用`spec`参数邀请模拟模仿现有类，以便在尝试访问模仿类上不存在的属性时引发错误。你还可以构造模拟方法，每次调用时返回不同的参数，通过将列表作为`side_effect`参数。`side_effect`参数非常灵活；你还可以使用它在调用模拟时执行任意函数或引发异常。

一般来说，我们应该对模拟非常吝啬。如果我们发现自己在给定的单元测试中模拟了多个元素，我们可能最终测试的是模拟框架而不是我们的真实代码。这毫无用处；毕竟，模拟已经经过了充分测试！如果我们的代码做了很多这样的事情，这可能是另一个迹象，表明我们正在测试的 API 设计得很糟糕。模拟应该存在于被测试代码和它们接口的库之间的边界上。如果这种情况没有发生，我们可能需要改变 API，以便在不同的地方重新划定边界。

# 测试多少是足够的？

我们已经确定了未经测试的代码是有问题的代码。但我们如何知道我们的代码被测试得有多好？我们如何知道我们的代码有多少被测试，有多少是有问题的？第一个问题更重要，但很难回答。即使我们知道我们已经测试了应用程序中的每一行代码，我们也不知道我们是否已经适当地测试了它。例如，如果我们编写了一个只检查当我们提供一个整数列表时会发生什么的统计测试，如果用于浮点数、字符串或自制对象的列表，它可能仍然会失败得很惨。设计完整测试套件的责任仍然在程序员身上。

第二个问题——我们的代码有多少被测试——很容易验证。**代码覆盖率**是程序执行的代码行数的估计。如果我们知道这个数字和程序中的代码行数，我们就可以估算出实际被测试或覆盖的代码百分比。如果我们另外有一个指示哪些行没有被测试的指标，我们就可以更容易地编写新的测试来确保这些行不会出错。

用于测试代码覆盖率的最流行的工具叫做`coverage.py`。它可以像大多数其他第三方库一样安装，使用`pip install coverage`命令。

我们没有空间来涵盖覆盖 API 的所有细节，所以我们只看一些典型的例子。如果我们有一个运行所有单元测试的 Python 脚本（例如，使用`unittest.main`、`discover`、`pytest`或自定义测试运行器），我们可以使用以下命令执行覆盖分析：

```py
$coverage run coverage_unittest.py  
```

这个命令将正常退出，但它会创建一个名为`.coverage`的文件，其中保存了运行的数据。现在我们可以使用`coverage report`命令来获取代码覆盖的分析：

```py
$coverage report  
```

生成的输出应该如下所示：

```py
Name                           Stmts   Exec  Cover
--------------------------------------------------
coverage_unittest                  7      7   100%
stats                             19      6    31%
--------------------------------------------------
TOTAL                             26     13    50%  
```

这份基本报告列出了执行的文件（我们的单元测试和一个导入的模块）。还列出了每个文件中的代码行数以及测试执行的代码行数。然后将这两个数字合并以估算代码覆盖量。如果我们在`report`命令中传递`-m`选项，它还会添加一个如下所示的列：

```py
Missing
-----------
8-12, 15-23  
```

这里列出的行范围标识了在测试运行期间未执行的`stats`模块中的行。

我们刚刚对代码覆盖工具运行的示例使用了我们在本章早些时候创建的相同的 stats 模块。但是，它故意使用了一个失败的测试来测试文件中的大量代码。以下是测试：

```py
from stats import StatsList 
import unittest 

class TestMean(unittest.TestCase): 
    def test_mean(self): 
        self.assertEqual(StatsList([1,2,2,3,3,4]).mean(), 2.5) 

if __name__ == "__main__": 

    unittest.main() 
```

这段代码没有测试中位数或模式函数，这些函数对应于覆盖输出告诉我们缺失的行号。

文本报告提供了足够的信息，但如果我们使用`coverage html`命令，我们可以获得一个更有用的交互式 HTML 报告，我们可以在 Web 浏览器中查看。网页甚至会突出显示源代码中哪些行已经测试过，哪些行没有测试过。看起来是这样的：

![](img/f42ff938-8ab2-424a-bce5-445480a4d0a2.png)

我们也可以使用`pytest`模块的`coverage.py`模块。我们需要安装`pytest`插件以进行代码覆盖率，使用`pip install pytest-coverage`。该插件为`pytest`添加了几个命令行选项，其中最有用的是`--cover-report`，可以设置为`html`，`report`或`annotate`（后者实际上修改了原始源代码以突出显示未覆盖的任何行）。

不幸的是，如果我们可以在本章的这一部分上运行覆盖率报告，我们会发现我们并没有覆盖大部分关于代码覆盖率的知识！可以使用覆盖 API 来从我们自己的程序（或测试套件）中管理代码覆盖率，`coverage.py`接受了许多我们没有涉及的配置选项。我们还没有讨论语句覆盖和分支覆盖之间的区别（后者更有用，并且是最近版本的`coverage.py`的默认值），或者其他风格的代码覆盖。

请记住，虽然 100％的代码覆盖率是我们所有人都应该努力追求的一个远大目标，但 100％的覆盖率是不够的！仅仅因为一个语句被测试了并不意味着它被正确地测试了所有可能的输入。

# 案例研究

让我们通过编写一个小的、经过测试的密码应用程序来了解测试驱动开发。不用担心-您不需要了解复杂的现代加密算法（如 AES 或 RSA）背后的数学。相反，我们将实现一个称为 Vigenère 密码的 16 世纪算法。该应用程序只需要能够使用此密码对消息进行编码和解码，给定一个编码关键字。

如果您想深入了解 RSA 算法的工作原理，我在我的博客上写了一篇文章[`dusty.phillips.codes/`](https://dusty.phillips.codes/)。

首先，我们需要了解密码是如何工作的，如果我们手动应用它（没有计算机）。我们从以下表格开始：

```py
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 
B C D E F G H I J K L M N O P Q R S T U V W X Y Z A 
C D E F G H I J K L M N O P Q R S T U V W X Y Z A B 
D E F G H I J K L M N O P Q R S T U V W X Y Z A B C 
E F G H I J K L M N O P Q R S T U V W X Y Z A B C D 
F G H I J K L M N O P Q R S T U V W X Y Z A B C D E 
G H I J K L M N O P Q R S T U V W X Y Z A B C D E F 
H I J K L M N O P Q R S T U V W X Y Z A B C D E F G 
I J K L M N O P Q R S T U V W X Y Z A B C D E F G H 
J K L M N O P Q R S T U V W X Y Z A B C D E F G H I 
K L M N O P Q R S T U V W X Y Z A B C D E F G H I J 
L M N O P Q R S T U V W X Y Z A B C D E F G H I J K 
M N O P Q R S T U V W X Y Z A B C D E F G H I J K L 
N O P Q R S T U V W X Y Z A B C D E F G H I J K L M 
O P Q R S T U V W X Y Z A B C D E F G H I J K L M N 
P Q R S T U V W X Y Z A B C D E F G H I J K L M N O 
Q R S T U V W X Y Z A B C D E F G H I J K L M N O P 
R S T U V W X Y Z A B C D E F G H I J K L M N O P Q 
S T U V W X Y Z A B C D E F G H I J K L M N O P Q R 
T U V W X Y Z A B C D E F G H I J K L M N O P Q R S 
U V W X Y Z A B C D E F G H I J K L M N O P Q R S T 
V W X Y Z A B C D E F G H I J K L M N O P Q R S T U 
W X Y Z A B C D E F G H I J K L M N O P Q R S T U V 
X Y Z A B C D E F G H I J K L M N O P Q R S T U V W 
Y Z A B C D E F G H I J K L M N O P Q R S T U V W X 
Z A B C D E F G H I J K L M N O P Q R S T U V W X Y 
```

给定关键字 TRAIN，我们可以对消息 ENCODED IN PYTHON 进行编码如下：

1.  将关键字和消息一起重复，这样很容易将一个字母映射到另一个字母：

```py
E N C O D E D I N P Y T H O N
T R A I N T R A I N T R A I N 
```

1.  对于明文中的每个字母，找到以该字母开头的表中的行。

1.  找到与所选明文字母的关键字字母相关联的列。

1.  编码字符位于该行和列的交点处。

例如，以 E 开头的行与以 T 开头的列相交于字符 X。因此，密文中的第一个字母是 X。以 N 开头的行与以 R 开头的列相交于字符 E，导致密文 XE。C 与 A 相交于 C，O 与 I 相交于 W。D 和 N 映射到 Q，而 E 和 T 映射到 X。完整的编码消息是 XECWQXUIVCRKHWA。

解码遵循相反的过程。首先，找到具有共享关键字字符（T 行）的行，然后找到该行中编码字符（X）所在的位置。明文字符位于该行的列顶部（E）。

# 实施它

我们的程序将需要一个`encode`方法，该方法接受关键字和明文并返回密文，以及一个`decode`方法，该方法接受关键字和密文并返回原始消息。

但我们不只是写这些方法，让我们遵循测试驱动开发策略。我们将使用`pytest`进行单元测试。我们需要一个`encode`方法，我们知道它必须做什么；让我们首先为该方法编写一个测试，如下所示：

```py
def test_encode():
    cipher = VigenereCipher("TRAIN")
    encoded = cipher.encode("ENCODEDINPYTHON")
    assert encoded == "XECWQXUIVCRKHWA"
```

这个测试自然会失败，因为我们没有在任何地方导入`VigenereCipher`类。让我们创建一个新的模块来保存该类。

让我们从以下`VigenereCipher`类开始：

```py
class VigenereCipher:
    def __init__(self, keyword):
        self.keyword = keyword

    def encode(self, plaintext):
        return "XECWQXUIVCRKHWA"

```

如果我们在测试类的顶部添加一行`from``vigenere_cipher``import``VigenereCipher`并运行`pytest`，前面的测试将通过！我们完成了第一个测试驱动开发周期。

这可能看起来像一个荒谬的测试，但实际上它验证了很多东西。第一次我实现它时，在类名中我把 cipher 拼错成了*cypher*。即使是我基本的单元测试也帮助捕捉了一个错误。即便如此，返回一个硬编码的字符串显然不是密码类的最明智的实现，所以让我们添加第二个测试，如下所示：

```py
def test_encode_character(): 
    cipher = VigenereCipher("TRAIN") 
    encoded = cipher.encode("E") 
    assert encoded == "X" 
```

啊，现在那个测试会失败。看来我们要更加努力了。但我突然想到了一件事：如果有人尝试用空格或小写字符对字符串进行编码会怎么样？在我们开始实现编码之前，让我们为这些情况添加一些测试，这样我们就不会忘记它们。预期的行为是去除空格，并将小写字母转换为大写，如下所示：

```py
def test_encode_spaces(): 
    cipher = VigenereCipher("TRAIN") 
    encoded = cipher.encode("ENCODED IN PYTHON") 
    assert encoded == "XECWQXUIVCRKHWA" 

def test_encode_lowercase(): 
    cipher = VigenereCipher("TRain") 
    encoded = cipher.encode("encoded in Python") 
    assert encoded == "XECWQXUIVCRKHWA" 
```

如果我们运行新的测试套件，我们会发现新的测试通过了（它们期望相同的硬编码字符串）。但如果我们忘记考虑这些情况，它们以后应该会失败。

现在我们有了一些测试用例，让我们考虑如何实现我们的编码算法。编写代码使用像我们在早期手动算法中使用的表是可能的，但考虑到每一行只是一个按偏移字符旋转的字母表，这似乎很复杂。事实证明（我问了维基百科），我们可以使用模运算来组合字符，而不是进行表查找。

给定明文和关键字字符，如果我们将这两个字母转换为它们的数字值（根据它们在字母表中的位置，A 为 0，Z 为 25），将它们相加，并取余数模 26，我们就得到了密文字符！这是一个简单的计算，但由于它是逐个字符进行的，我们应该把它放在自己的函数中。在我们这样做之前，我们应该为新函数编写一个测试，如下所示：

```py
from vigenere_cipher import combine_character 
def test_combine_character(): 
    assert combine_character("E", "T") == "X" 
    assert combine_character("N", "R") == "E" 
```

现在我们可以编写代码使这个函数工作。老实说，我在完全正确地编写这个函数之前，不得不多次运行测试。首先，我不小心返回了一个整数，然后我忘记将字符从基于零的比例转换回正常的 ASCII 比例。有了测试可用，很容易测试和调试这些错误。这是测试驱动开发的另一个好处。代码的最终工作版本如下所示：

```py
def combine_character(plain, keyword): 
    plain = plain.upper() 
    keyword = keyword.upper() 
    plain_num = ord(plain) - ord('A') 
    keyword_num = ord(keyword) - ord('A') 
    return chr(ord('A') + (plain_num + keyword_num) % 26) 
```

现在`combine_characters`已经经过测试，我以为我们准备好实现我们的`encode`函数了。然而，在该函数内部我们首先需要一个与明文长度相同的关键字字符串的重复版本。让我们首先实现一个函数。哎呀，我是说让我们首先实现测试，如下所示：

```py
def test_extend_keyword(): cipher = VigenereCipher("TRAIN") extended = cipher.extend_keyword(16) assert extended == "TRAINTRAINTRAINT" 
```

在编写这个测试之前，我原本打算将`extend_keyword`作为一个独立的函数，接受一个关键字和一个整数。但当我开始起草测试时，我意识到更合理的做法是将它作为`VigenereCipher`类的辅助方法，这样它就可以访问`self.keyword`属性。这显示了测试驱动开发如何帮助设计更合理的 API。以下是方法的实现：

```py
    def extend_keyword(self, number):
        repeats = number // len(self.keyword) + 1
        return (self.keyword * repeats)[:number]
```

再次，这需要几次运行测试才能做对。我最终添加了一个修改后的测试副本，一个有十五个字母，一个有十六个字母，以确保它在整数除法有偶数的情况下也能工作。

现在我们终于准备好编写我们的`encode`方法了，如下所示：

```py
    def encode(self, plaintext): 
        cipher = [] 
        keyword = self.extend_keyword(len(plaintext)) 
        for p,k in zip(plaintext, keyword): 
            cipher.append(combine_character(p,k)) 
        return "".join(cipher) 
```

看起来正确。我们的测试套件现在应该通过了，对吗？

实际上，如果我们运行它，我们会发现仍然有两个测试失败。先前失败的编码测试实际上已经通过了，但我们完全忘记了空格和小写字符！幸好我们写了这些测试来提醒我们。我们将不得不在方法的开头添加以下行：

```py
        plaintext = plaintext.replace(" ", "").upper() 
```

如果我们在实现某些功能的过程中想到一个边界情况，我们可以创建一个描述该想法的测试。我们甚至不必实现测试；我们只需运行`assert False`来提醒我们以后再实现它。失败的测试永远不会让我们忘记边界情况，它不像问题跟踪器中的工单那样容易被忽视。如果花费一段时间来修复实现，我们可以将测试标记为预期失败。

现在所有的测试都通过了。这一章非常长，所以我们将压缩解码的示例。以下是一些测试：

```py
def test_separate_character(): 
    assert separate_character("X", "T") == "E" 
    assert separate_character("E", "R") == "N" 

def test_decode(): 
    cipher = VigenereCipher("TRAIN") 
    decoded = cipher.decode("XECWQXUIVCRKHWA") 
    assert decoded == "ENCODEDINPYTHON" 
```

以下是`separate_character`函数：

```py
def separate_character(cypher, keyword): 
    cypher = cypher.upper() 
    keyword = keyword.upper() 
    cypher_num = ord(cypher) - ord('A') 
    keyword_num = ord(keyword) - ord('A') 
    return chr(ord('A') + (cypher_num - keyword_num) % 26) 
```

现在我们可以添加`decode`方法：

```py
    def decode(self, ciphertext): 
        plain = [] 
        keyword = self.extend_keyword(len(ciphertext)) 
        for p,k in zip(ciphertext, keyword): 
            plain.append(separate_character(p,k)) 
        return "".join(plain) 
```

这些方法与编码所使用的方法非常相似。有了所有这些编写并通过的测试，我们现在可以回过头修改我们的代码，知道它仍然安全地通过测试。例如，如果我们用以下重构后的方法替换现有的`encode`和`decode`方法，我们的测试仍然通过：

```py
    def _code(self, text, combine_func): 
        text = text.replace(" ", "").upper() 
        combined = [] 
        keyword = self.extend_keyword(len(text)) 
        for p,k in zip(text, keyword): 
            combined.append(combine_func(p,k)) 
        return "".join(combined) 

    def encode(self, plaintext): 
        return self._code(plaintext, combine_character) 

    def decode(self, ciphertext): 
        return self._code(ciphertext, separate_character) 
```

这是测试驱动开发的最终好处，也是最重要的。一旦测试编写完成，我们可以尽情改进我们的代码，而且可以确信我们的更改没有破坏我们一直在测试的任何东西。此外，我们确切地知道我们的重构何时完成：当所有测试都通过时。

当然，我们的测试可能并不全面测试我们需要的一切；维护或代码重构仍然可能导致未经诊断的错误，这些错误在测试中不会显示出来。自动化测试并不是绝对可靠的。然而，如果出现错误，仍然可以按照测试驱动的计划进行，如下所示：

1.  编写一个测试（或多个测试），复制或*证明*出现的错误。当然，这将失败。

1.  然后编写代码使测试停止失败。如果测试全面，错误将被修复，我们将知道它是否再次发生，只要运行测试套件。

最后，我们可以尝试确定我们的测试在这段代码上的运行情况。安装了`pytest`覆盖插件后，`pytest -coverage-report=report`告诉我们，我们的测试套件覆盖了 100%的代码。这是一个很好的统计数据，但我们不应该对此过于自负。我们的代码在对包含数字的消息进行编码时还没有经过测试，因此其行为是未定义的。

# 练习

练习测试驱动开发。这是你的第一个练习。如果你开始一个新项目，这样做会更容易，但如果你有现有的代码需要处理，你可以通过为每个新功能编写测试来开始。随着你对自动化测试的热爱增加，这可能会变得令人沮丧。未经测试的旧代码将开始感觉僵化和紧密耦合，并且维护起来会变得不舒服；你会开始感觉自己的更改正在破坏代码，而你却无法知道，因为没有测试。但是如果你从小处开始，随着时间的推移，为代码库添加测试会改进它。

因此，要开始尝试测试驱动开发，可以开始一个全新的项目。一旦你开始意识到这些好处（你会的），并意识到编写测试所花费的时间很快就能以更易维护的代码来回报，你就会想要开始为现有代码编写测试。这就是你应该开始做的时候，而不是之前。为我们*知道*有效的代码编写测试是无聊的。在意识到我们认为有效的代码实际上有多破碎之前，很难对项目产生兴趣。

尝试使用内置的`unittest`模块和`pytest`编写相同的一组测试。您更喜欢哪个？`unittest`更类似于其他语言中的测试框架，而`pytest`可以说更符合 Python 的风格。两者都允许我们编写面向对象的测试，并轻松测试面向对象的程序。

在我们的案例研究中，我们使用了`pytest`，但我们没有涉及任何使用`unittest`不容易进行测试的功能。尝试调整测试以使用测试跳过或固定装置（`VignereCipher`的一个实例将会很有帮助）。尝试各种设置和拆卸方法，并将它们的使用与 funcargs 进行比较。哪种对您来说更自然？

尝试对您编写的测试运行覆盖报告。您是否错过了测试任何代码行？即使您有 100％的覆盖率，您是否测试了所有可能的输入？如果您正在进行测试驱动的开发，100％的覆盖率应该是很自然的，因为您会在满足该测试的代码之前编写测试。但是，如果为现有代码编写测试，很可能会有未经测试的边缘条件。

仔细考虑一下那些在某种程度上不同的值，例如：

+   当您期望完整列表时得到空列表

+   负数、零、一或无穷大与正整数相比

+   不能精确舍入到小数位的浮点数

+   当您期望数字时得到字符串

+   当您期望 ASCII 时得到 Unicode 字符串

+   当您期望有意义的东西时得到无处不在的`None`值

如果您的测试涵盖了这些边缘情况，您的代码将会很完善。

# 总结

我们最终涵盖了 Python 编程中最重要的主题：自动化测试。测试驱动开发被认为是最佳实践。标准库`unittest`模块提供了一个出色的开箱即用的测试解决方案，而`pytest`框架具有一些更符合 Python 风格的语法。模拟可以用于在我们的测试中模拟复杂的类。代码覆盖率给我们一个估计，我们的代码有多少被我们的测试运行，但它并不告诉我们我们已经测试了正确的东西。

感谢阅读《Python 入门指南》。我希望您享受了这段旅程，并渴望开始在未来的所有项目中实现面向对象的软件！
