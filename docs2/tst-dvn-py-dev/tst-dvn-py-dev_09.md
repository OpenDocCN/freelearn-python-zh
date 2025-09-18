# 第九章 单元测试模式

在整本书中，我们探讨了 TDD 中的各种模式和反模式。在本章中，你将了解一些本书之前未讨论过的额外模式。在这个过程中，你还将了解 Python `unittest`模块提供的更多高级功能，例如测试加载器、测试运行器和跳过测试。

# 模式 - 快速测试

TDD 的一个关键目标是编写执行快速的测试。在进行 TDD 时，我们会频繁地运行测试——可能每隔几分钟就会运行一次。TDD 的习惯是在开发代码、重构、提交前和部署前多次运行测试。如果测试运行时间过长，我们就不愿意频繁运行它们，这样就违背了测试的目的。

考虑到这一点，以下是一些保持测试快速运行的技术：

+   **禁用不需要的外部服务**：有些服务并非应用目的的核心，可以被禁用。例如，我们可能使用一个服务来收集用户如何使用我们应用的分析数据。我们的应用可能在每次操作时都会调用这个服务。这样的服务可以被禁用，从而使得测试运行得更快。

+   **模拟外部服务**：其他外部服务，如服务器、数据库、缓存等，可能对应用的功能至关重要。外部服务需要时间来启动、关闭和通信。我们想要模拟这些服务，并让我们的测试在模拟服务上运行。

+   **使用服务的快速变体**：如果我们必须使用服务，那么请确保它是快速的。例如，用一个内存数据库替换数据库，它更快，启动和关闭所需时间更少。同样，我们可以用一个记录要发送的电子邮件的内存电子邮件服务器替换对电子邮件服务器的调用，而不实际发送电子邮件。

+   **外部化配置**：配置与单元测试有什么关系？简单来说：如果我们需要启用或禁用服务，或者用模拟服务替换服务，那么我们需要为常规应用和运行单元测试时设计不同的配置。这要求我们以允许我们轻松地在多个配置之间切换的方式设计应用。

+   **仅运行当前模块的测试**：`unittest`测试运行器和第三方运行器都允许我们运行测试子集——特定模块、类或单个测试的测试。这对于拥有数千个测试的大型测试套件来说是一个很好的功能，因为它允许我们只运行正在工作的模块的测试。

# 模式 - 运行测试子集

我们已经看到了一种简单的方法来运行测试子集，只需在命令行上指定模块或测试类，如下所示：

```py
python -m unittest stock_alerter.tests.test_stock
python -m unittest stock_alerter.tests.test_stock.StockTest

```

这适用于我们想要基于模块运行子集的常见情况。如果我们想根据其他参数运行测试怎么办？也许我们想运行一组基本烟雾测试，或者我们只想运行集成测试，或者我们想在特定平台或 Python 版本上运行时跳过测试。

`unittest` 模块允许我们创建测试套件。**测试套件**是一组要运行的测试类。默认情况下，`unittest` 会自动发现测试并在内部创建一个包含所有匹配发现模式的测试的测试套件。然而，我们也可以手动创建不同的测试套件并运行它们。

测试套件是通过使用 `unittest.TestSuite` 类创建的。`TestSuite` 类有以下两个感兴趣的方法：

+   `addTest`: 此方法接受一个 `TestCase` 或另一个 `TestSuite` 并将其添加到套件中

+   `addTests`: 与 `addTest` 类似，此方法接受一个 `TestCase` 或 `TestSuite` 列表并将其添加到套件中

那么，我们如何使用这个函数呢？

首先，我们编写一个函数来创建套件并返回它，如下所示：

```py
def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(StockTest("test_stock_update"))
    return test_suite
```

我们可以选择套件中想要的特定测试。我们在这里向套件中添加了一个单个测试。

接下来，我们需要编写一个脚本来运行此套件，如下所示：

```py
import unittest

from stock_alerter.tests import test_stock

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(test_stock.suite())
```

在这里，我们创建了一个 `TextTestRunner`，它将运行测试并将套件或测试传递给它。`unittest.TextTestRunner` 是一个测试运行器，它接受一个测试套件并运行它，在控制台上显示测试的运行结果。

### 注意

`unittest.TextTestRunner` 是我们迄今为止一直在使用的默认测试运行器。我们可以编写自己的测试运行器。例如，我们可能会编写一个自定义测试运行器来实现 GUI 接口，或者一个将测试输出写入 XML 文件的测试运行器。

当我们运行此脚本时，我们得到以下输出：

```py
.
------------------------
Ran 1 test in 0.000s

OK

```

同样，我们可以为不同的测试子集创建不同的套件——例如，一个只包含集成测试的单独套件——并根据我们的需求只运行特定的套件。

## 测试加载器

套件函数的一个问题是，我们必须将每个测试单独添加到套件中。如果我们有很多测试，这是一个繁琐的过程。幸运的是，我们可以通过使用 `unittest.TestLoader` 对象来简化这个过程，如下所示：

```py
def suite():
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    test_suite.addTest(StockTest("test_stock_update"))
    test_suite.addTest(
        loader.loadTestsFromTestCase(StockCrossOverSignalTest))
    return test_suite
```

在这里，加载器从 `StockCrossOverSignalTest` 类中提取所有测试并创建一个套件。如果我们只想返回套件，我们可以直接返回套件，或者我们可以创建一个新的套件并添加额外的测试。在上面的例子中，我们创建了一个包含 `StockTest` 类中的一个测试和 `StockCrossOverSignalTest` 类中所有测试的套件。

`unittest.TestLoader` 还包含一些其他用于加载测试的方法：

+   `loadTestsFromModule`: 此方法接受一个模块并返回该模块中所有测试的测试套件。

+   `loadTestsFromName`：此方法接受一个指向模块、类或函数的字符串引用，并从中提取测试。如果是一个函数，则调用该函数，并返回函数返回的测试套件。字符串引用采用点格式，这意味着我们可以传递类似`stock_alerter.tests.test_stock`或`stock_alerter.tests.test_stock.StockTest`，甚至`stock_alerter.tests.test_stock.suite`的内容。

+   `discover`：此方法执行默认的自动发现过程，并将收集到的测试作为套件返回。该方法接受三个参数：起始目录、查找`test`模块的模式（默认`test*.py`）和顶级目录。

使用这些方法，我们可以仅创建我们想要的测试套件。我们可以为不同的目的创建不同的套件，并从测试脚本中执行它们。

## 使用 load_tests 协议

创建测试套件的一个更简单的方法是使用`load_tests`函数。正如我们在第七章中看到的，“使用 doctest 的可执行文档”，如果测试模块中存在`load_tests`函数，`unittest`框架会调用该函数。该函数应返回一个包含要运行的测试的`TestSuite`对象。当我们只想稍微修改默认的自动发现过程时，`load_tests`是一个更好的解决方案。

`load_tests`传递三个参数：用于加载测试的加载器、默认将要加载的测试套件以及为搜索指定的测试模式。

假设我们不想在当前平台是 Windows 时运行`StockCrossOverSignalTest`测试。我们可以编写一个如下所示的`load_tests`函数：

```py
def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(StockTest))
    if not sys.platform.startswith("win"):
        suite.addTest(
            loader.loadTestsFromTestCase(StockCrossOverSignalTest))
    return suite
```

现在，`StockCrossOverSignalTest`测试将仅在非 Windows 平台上运行。当使用`load_tests`方法时，我们不需要编写单独的脚本来运行测试或创建测试运行器。它挂钩到自动发现过程，因此使用起来更简单。

## 跳过测试

在上一节中，我们使用`load_tests`机制在平台是 Windows 时跳过一些测试。`unittest`模块提供了一个更简单的使用`skip`装饰器来完成相同任务的方法。只需用装饰器装饰一个类或方法，测试就会跳过，如下所示：

```py
@unittest.skip("skip this test for now")
def test_stock_update(self):
    self.goog.update(datetime(2014, 2, 12), price=10)
    assert_that(self.goog.price, equal_to(10))
```

装饰器接受一个参数，我们可以在其中指定跳过测试的原因。当我们运行所有测试时，我们会得到如下所示的输出：

```py
........................................................s..
-------------------------------------------------------------
Ran 59 tests in 0.094s

OK (skipped=1)

```

当以详细模式运行测试时，我们会得到如下所示的输出：

```py
test_stock_update (stock_alerter.tests.test_stock.StockTest) ... skipped 'skip this test for now'

```

`skip`装饰器无条件地跳过测试，但`unittest`提供了两个额外的装饰器`skipIf`和`skipUnless`，允许我们指定一个条件来跳过测试。这些装饰器将布尔值作为第一个参数，将消息作为第二个参数。`skipIf`如果布尔值为`True`则跳过测试，而`skipUnless`如果布尔值为`False`则跳过测试。

以下测试将在所有平台（除了 Windows）上运行：

```py
@unittest.skipIf(sys.platform.startswith("win"), "skip on windows")
def test_stock_price_should_give_the_latest_price(self):
    self.goog.update(datetime(2014, 2, 12), price=10)
    self.goog.update(datetime(2014, 2, 13), price=8.4)
    self.assertAlmostEqual(8.4, self.goog.price, delta=0.0001)
```

而下面的测试只会在 Windows 上运行：

```py
@unittest.skipUnless(sys.platform.startswith("win"), "only run on windows")
def test_price_is_the_latest_even_if_updates_are_made_out_of_order(self):
    self.goog.update(datetime(2014, 2, 13), price=8)
    self.goog.update(datetime(2014, 2, 12), price=10)
    self.assertEqual(8, self.goog.price)
```

`skip`、`skipIf`和`skipUnless`装饰器可以用于测试方法和测试类。当应用于类时，类中的所有测试都将被跳过。

# 模式 - 使用属性

`nose2`测试运行器有一个有用的`attrib`插件，允许我们在测试用例上设置属性并选择匹配特定属性的测试。

例如，以下测试设置了三个属性：

```py
def test_stock_update(self):
    self.goog.update(datetime(2014, 2, 12), price=10)
    self.assertEqual(10, self.goog.price)

test_stock_update.slow = True
test_stock_update.integration = True
test_stock_update.python = ["2.6", "3.4"]
```

当通过以下命令运行 nose2 时，插件将被启用，并且只有设置了`integration`属性为`True`的测试将被执行：

```py
nose2 --plugin=nose2.plugins.attrib -A "integration"

```

插件还可以运行所有在列表中具有特定值的测试。以下是一个命令示例：

```py
nose2 --plugin=nose2.plugins.attrib -A "python=2.6"

```

上述命令将运行所有将`python`属性设置为`2.6`或包含列表中的值`2.6`的测试。它将选择并运行之前显示的`test_stock_update`测试。

插件还可以运行所有没有设置属性的测试。以下是一个命令示例：

```py
nose2 --plugin=nose2.plugins.attrib -A "!slow"

```

上述命令将运行所有未标记为慢速的测试。

插件还可以接受复杂条件，因此我们可以给出以下命令：

```py
nose2 --plugin=nose2.plugins.attrib -E "integration and '2.6' in python"

```

此测试运行所有具有`integration`属性以及`python`属性列表中的`2.6`的测试。请注意，我们使用了`-E`开关来指定我们正在提供一个`python`条件表达式。

属性插件是一个很好的方法，可以在不手动从每个可能的组合中创建测试套件的情况下运行特定的测试子集。

## 使用 vanilla unittests 的属性

`attrib`插件需要 nose2 才能工作。如果我们正在使用常规的`unittest`模块怎么办？`unittest`模块的设计允许我们仅用几行代码轻松编写一个简化版本，如下所示：

```py
import unittest

class AttribLoader(unittest.TestLoader):
    def __init__(self, attrib):
        self.attrib = attrib

    def loadTestsFromModule(self, module, use_load_tests=False):
        return super().loadTestsFromModule(module, use_load_tests=False)

    def getTestCaseNames(self, testCaseClass):
        test_names = super().getTestCaseNames(testCaseClass)
        filtered_test_names = [test
                               for test in test_names
                               if hasattr(getattr(testCaseClass, test), self.attrib)]
        return filtered_test_names

if __name__ == "__main__":
    loader = AttribLoader("slow")
    test_suite = loader.discover(".")
    runner = unittest.TextTestRunner()
    runner.run(test_suite)
```

这段小代码将只运行那些在测试函数上设置了`integration`属性的测试。让我们更深入地看看代码。

首先，我们继承默认的`unittest.TestLoader`类，并创建我们自己的加载器，称为`AttribLoader`。记住，**加载器**是负责从类或模块中加载测试的类。

接下来，我们重写`getTestCaseNames`方法。此方法从一个类中返回一个测试用例名称列表。在这里，我们调用父方法以获取默认的测试列表，然后选择具有所需属性的测试函数。这个过滤后的列表将被返回，并且只有这些测试将被执行。

那么，为什么我们还要重写`loadTestsFromModule`方法呢？简单来说：加载测试的默认行为是按方法上的`test`前缀进行匹配，但如果存在`load_tests`函数，则所有操作都将委托给`load_tests`函数。因此，所有定义了`load_tests`函数的模块都将优先于我们的属性过滤方案。

当使用我们的加载器时，我们调用默认实现，但将`use_load_tests`参数设置为`False`。这意味着将不会执行任何`load_tests`函数，要加载的测试将由我们返回的过滤列表确定。如果我们想优先考虑`load_tests`（这是默认行为），那么我们只需从`AttribLoader`中移除此方法。

好的，现在加载器准备好了，我们修改我们的测试运行脚本以使用这个加载器，而不是默认加载器。我们通过调用`discover`方法来获取加载的测试套件，该方法反过来调用我们重写的`getTestCaseNames`方法。我们将这个套件传递给运行器并运行测试。

加载器可以很容易地修改以支持选择没有给定属性或支持更复杂条件的测试。然后我们可以添加对脚本的支持，以接受命令行上的属性并将其传递给加载器。

# 模式 - 预期失败

有时候，我们有一些失败的测试，但由于某种原因，我们不想立即修复它们。这可能是因为我们找到了一个错误并编写了一个失败的测试来演示该错误（这是一个非常好的做法），但我们决定稍后修复错误。现在，整个测试套件都在失败。

一方面，我们不希望套件失败，因为我们知道这个错误并想稍后修复它。另一方面，我们不想从套件中移除测试，因为它提醒我们需要修复错误。我们该怎么办？

Python 的`unittest`模块提供了一个解决方案：将测试标记为预期失败。我们可以通过将`unittest.expectedFailure`装饰器应用于测试来实现这一点。以下是一个实际应用的示例：

```py
class AlertTest(unittest.TestCase):
    @unittest.expectedFailure
    def test_action_is_executed_when_rule_matches(self):
        goog = mock.MagicMock(spec=Stock)
        goog.updated = Event()
        goog.update.side_effect = \
            lambda date, value: goog.updated.fire(self)
        exchange = {"GOOG": goog}
        rule = mock.MagicMock(spec=PriceRule)
        rule.matches.return_value = True
        rule.depends_on.return_value = {"GOOG"}
        action = mock.MagicMock()
        alert = Alert("sample alert", rule, action)
        alert.connect(exchange)
        exchange["GOOG"].update(datetime(2014, 2, 10), 11)
        action.execute.assert_called_with("sample alerts")
```

当执行测试时，我们得到以下输出：

```py
......x....................................................
------------------------------------------------------------
Ran 59 tests in 0.188s

OK (expected failures=1)

```

以下是其详细输出：

```py
test_action_is_executed_when_rule_matches (stock_alerter.tests.test_alert.AlertTest) ... expected failure

```

# 模式 - 数据驱动测试

我们之前简要探讨了数据驱动测试。数据驱动测试通过允许我们编写单个测试执行流程并使用不同的数据组合运行它来减少样板测试代码的数量。

以下是一个使用我们在这本书前面提到的 nose2 参数化插件的示例：

```py
from nose2.tools.params import params

def given_a_series_of_prices(stock, prices):
    timestamps = [datetime(2014, 2, 10), datetime(2014, 2, 11),
                  datetime(2014, 2, 12), datetime(2014, 2, 13)]
    for timestamp, price in zip(timestamps, prices):
        stock.update(timestamp, price)

@params(
    ([8, 10, 12], True),
    ([8, 12, 10], False),
    ([8, 10, 10], False)
)
def test_stock_trends(prices, expected_output):
    goog = Stock("GOOG")
    given_a_series_of_prices(goog, prices)
    assert goog.is_increasing_trend() == expected_output
```

运行此类测试需要使用 nose2。是否有方法使用常规的`unittest`模块做类似的事情？长期以来，没有不使用元类就能做到这一点的方法，但 Python 3.4 新增的一个特性使得这成为可能。

这个新特性是`unittest.subTest`上下文管理器。上下文管理器块内的所有代码都将被视为一个单独的测试，任何失败都将独立报告。以下是一个示例：

```py
class StockTrendTest(unittest.TestCase):
    def given_a_series_of_prices(self, stock, prices):
        timestamps = [datetime(2014, 2, 10), datetime(2014, 2, 11),
                      datetime(2014, 2, 12), datetime(2014, 2, 13)]
        for timestamp, price in zip(timestamps, prices):
            stock.update(timestamp, price)

    def test_stock_trends(self):
        dataset = [
            ([8, 10, 12], True),
            ([8, 12, 10], False),
            ([8, 10, 10], False)
        ]
        for data in dataset:
            prices, output = data
            with self.subTest(prices=prices, output=output):
                goog = Stock("GOOG")
                self.given_a_series_of_prices(goog, prices)
                self.assertEqual(output, goog.is_increasing_trend())
```

在这个例子中，测试遍历不同的场景并对每个场景进行断言。整个 Arrange-Act-Assert 模式都发生在`subTest`上下文管理器内部。上下文管理器接受任何关键字参数作为参数，并在显示错误消息时使用这些参数。

当我们运行测试时，我们得到如下输出：

```py
.
------------------------
Ran 1 test in 0.000s

OK

```

如我们所见，整个测试被视为单个测试，并且它显示测试通过了。

假设我们将测试改为使其在三个案例中的两个案例中失败，如下所示：

```py
class StockTrendTest(unittest.TestCase):
    def given_a_series_of_prices(self, stock, prices):
        timestamps = [datetime(2014, 2, 10), datetime(2014, 2, 11),
                      datetime(2014, 2, 12), datetime(2014, 2, 13)]
        for timestamp, price in zip(timestamps, prices):
            stock.update(timestamp, price)

    def test_stock_trends(self):
        dataset = [
            ([8, 10, 12], True),
            ([8, 12, 10], True),
            ([8, 10, 10], True)
        ]
        for data in dataset:
            prices, output = data
            with self.subTest(prices=prices, output=output):
                goog = Stock("GOOG")
                self.given_a_series_of_prices(goog, prices)
                self.assertEqual(output, goog.is_increasing_trend())
```

然后，输出变为以下内容：

```py
======================================================================
FAIL: test_stock_trends (stock_alerter.tests.test_stock.StockTrendTest) (output=True, prices=[8, 12, 10])
----------------------------------------------------------------------
Traceback (most recent call last):
 File "c:\Projects\tdd_with_python\src\stock_alerter\tests\test_stock.py", line 78, in test_stock_trends
 self.assertEqual(output, goog.is_increasing_trend())
AssertionError: True != False

======================================================================
FAIL: test_stock_trends (stock_alerter.tests.test_stock.StockTrendTest) (output=True, prices=[8, 10, 10])
----------------------------------------------------------------------
Traceback (most recent call last):
 File "c:\Projects\tdd_with_python\src\stock_alerter\tests\test_stock.py", line 78, in test_stock_trends
 self.assertEqual(output, goog.is_increasing_trend())
AssertionError: True != False

----------------------------------------------------------------------
Ran 1 test in 0.000s

FAILED (failures=2)

```

如前所述输出所示，它显示只运行了一个测试，但每个失败都会单独报告。此外，当测试失败时使用的值被附加到测试名称的末尾，这使得可以很容易地看到哪个条件失败了。这里显示的参数是传递给`subTest`上下文管理器的参数。

# 模式 - 集成和系统测试

在整本书中，我们强调了单元测试不是集成测试的事实。它们有不同的目的，即验证系统在集成时是否工作。话虽如此，集成测试也很重要，不应被忽视。集成测试可以使用我们用于编写单元测试的相同`unittest`框架来编写。编写集成测试时需要记住的关键点如下：

+   **仍然禁用非核心服务**：保持非核心服务，如分析或日志记录，禁用。这些不会影响应用程序的功能。

+   **启用所有核心服务**：每个其他服务都应该处于活动状态。我们不希望模拟或伪造这些服务，因为这违背了集成测试的全部目的。

+   **使用属性标记集成测试**：通过这样做，我们可以在开发期间轻松选择仅运行单元测试，同时允许在持续集成或部署之前运行集成测试。

+   **尽量减少设置和拆卸时间**：例如，不要为每个测试启动和停止服务器。相反，使用模块或包级别的固定装置在所有测试中只启动和停止一次服务。在这样做的时候，我们必须小心，确保我们的测试不会在测试之间破坏服务的状态。特别是，失败的测试或测试错误不应该使服务处于不一致的状态。

# 模式 - 间谍

模拟允许我们用一个虚拟模拟对象替换一个对象或类。我们已经看到我们如何使模拟返回预定义的值，这样被测试的类甚至不知道它已经调用了一个模拟对象。然而，有时我们可能只想记录对对象的调用，但允许执行流程继续到真实对象并返回。这样的对象被称为**间谍**。间谍保留了记录调用和在之后对调用进行断言的功能，但它并不像常规模拟那样替换真实对象。

创建`mock.Mock`对象时的`wraps`参数允许我们在代码中创建间谍行为。它接受一个对象作为值，所有对模拟的调用都转发到我们传递的对象，并将返回值发送回调用者。以下是一个示例：

```py
def test_action_doesnt_fire_if_rule_doesnt_match(self):
    goog = Stock("GOOG")
    exchange = {"GOOG": goog}
    rule = PriceRule("GOOG", lambda stock: stock.price > 10)
    rule_spy = mock.MagicMock(wraps=rule)
    action = mock.MagicMock()
    alert = Alert("sample alert", rule_spy, action)
    alert.connect(exchange)
    alert.check_rule(goog)
    rule_spy.matches.assert_called_with(exchange)
    self.assertFalse(action.execute.called)
```

在上面的例子中，我们为`rule`对象创建了一个间谍。间谍只是一个普通的模拟对象，它包装了在`wraps`参数中指定的真实对象。然后我们将间谍传递给警报。当`alert.check_rule`执行时，该方法在间谍上调用`matches`方法。间谍记录调用细节，然后将调用转发到真实规则对象并返回真实对象的值。然后我们可以对间谍进行断言以验证调用。

间谍通常用于我们想要避免过度模拟并使用真实对象，但又想对特定的调用进行断言的情况。它们也用于难以手动计算模拟返回值时，最好是进行实际计算并返回值。

## 模式 - 断言一系列调用

有时候，我们想要断言在多个对象之间发生了特定的调用序列。考虑以下测试用例：

```py
def test_action_fires_when_rule_matches(self):
    goog = Stock("GOOG")
    exchange = {"GOOG": goog}
    rule = mock.MagicMock()
    rule.matches.return_value = True
    rule.depends_on.return_value = {"GOOG"}
    action = mock.MagicMock()
    alert = Alert("sample alert", rule, action)
    alert.connect(exchange)
    goog.update(datetime(2014, 5, 14), 11)
    rule.matches.assert_called_with(exchange)
    self.assertTrue(action.execute.called)
```

在这个测试中，我们正在断言调用了`rule.matches`方法，以及调用了`action.execute`方法。我们编写断言的方式并没有检查这两个调用之间的顺序。即使`matches`方法在`execute`方法之后被调用，这个测试仍然会通过。如果我们想特别检查`matches`方法的调用发生在`execute`方法调用之前，该怎么办呢？

在回答这个问题之前，让我们看看这个交互式 Python 会话。首先，我们创建一个模拟对象，如下所示：

```py
>>> from unittest import mock
>>> obj = mock.Mock()

```

然后，我们得到两个作为模拟对象属性的子对象，如下所示：

```py
>>> child_obj1 = obj.child1
>>> child_obj2 = obj.child2

```

默认情况下，模拟对象在访问没有配置`return_value`的属性时，会返回新的模拟对象。所以`child_obj1`和`child_obj2`也将是模拟对象。

接下来，我们在模拟对象上调用一些方法，如下所示：

```py
>>> child_obj1.method1()
<Mock name='mock.child1.method1()' id='56161448'>
>>> child_obj2.method1()
<Mock name='mock.child2.method1()' id='56161672'>
>>> child_obj2.method2()
<Mock name='mock.child2.method2()' id='56162008'>
>>> obj.method()
<Mock name='mock.method()' id='56162232'>

```

再次，没有配置`return_value`，所以方法调用的默认行为是返回新的模拟对象。在这个例子中，我们可以忽略这些。

现在，让我们看一下子对象的`mock_calls`属性。这个属性包含了对模拟对象上记录的所有调用的列表，如下所示：

```py
>>> child_obj1.mock_calls
[call.method1()]
>>> child_obj2.mock_calls
[call.method1(), call.method2()]

```

模拟对象有记录的适当方法调用，正如预期的那样。现在，让我们看一下主`obj`模拟对象上的属性，如下所示：

```py
>>> obj.mock_calls
[call.child1.method1(),
 call.child2.method1(),
 call.child2.method2(),
 call.method()]

```

现在令人惊讶的是！主模拟对象似乎不仅有自己的调用细节，还有子模拟对象的所有调用！

那么，我们如何在测试中使用这个特性来断言不同模拟对象之间调用的顺序呢？

好吧，如果我们把上面的测试写成以下这样呢：

```py
def test_action_fires_when_rule_matches(self):
    goog = Stock("GOOG")
    exchange = {"GOOG": goog}
    main_mock = mock.MagicMock()
    rule = main_mock.rule
    rule.matches.return_value = True
    rule.depends_on.return_value = {"GOOG"}
    action = main_mock.action
    alert = Alert("sample alert", rule, action)
    alert.connect(exchange)
    goog.update(datetime(2014, 5, 14), 11)
    main_mock.assert_has_calls(
        [mock.call.rule.matches(exchange),
         mock.call.action.execute("sample alert")])
```

在这里，我们创建了一个主模拟对象，称为`main_mock`，而`rule`和`action`模拟则是这个主模拟的子模拟。然后我们像往常一样使用这些模拟。区别在于我们在断言部分使用`main_mock`。因为`main_mock`记录了调用子模拟的顺序，所以这个断言可以检查对`rule`和`action`模拟的调用顺序。

让我们更进一步。`assert_has_calls`方法只断言调用了调用，并且它们按照特定的顺序进行。该方法*不*保证这些是*唯一*的调用。在第一个调用之前或最后一个调用之后，甚至在这两个调用之间，可能还有其他调用。只要我们断言的调用被调用，并且它们之间保持了顺序，断言就会通过。

为了严格匹配调用，我们可以在`mock_calls`属性上简单地执行`assertEqual`，如下所示：

```py
def test_action_fires_when_rule_matches(self):
    goog = Stock("GOOG")
    exchange = {"GOOG": goog}
    main_mock = mock.MagicMock()
    rule = main_mock.rule
    rule.matches.return_value = True
    rule.depends_on.return_value = {"GOOG"}
    action = main_mock.action
    alert = Alert("sample alert", rule, action)
    alert.connect(exchange)
    goog.update(datetime(2014, 5, 14), 11)
    self.assertEqual([mock.call.rule.depends_on(),
                      mock.call.rule.matches(exchange),
                      mock.call.action.execute("sample alert")],
                     main_mock.mock_calls)
```

在上面，我们使用预期调用列表断言`mock_calls`。列表必须完全匹配——没有缺失的调用，没有多余的调用，没有任何不同。需要注意的一点是，我们必须列出*每一个*调用。有一个调用`rule.depends_on`，这是在`alert.connect`方法中完成的。我们必须指定这个调用，即使它与我们要测试的功能无关。

通常，匹配每一个调用会导致测试变得冗长，因为所有与被测试功能无关的调用也需要放入预期的输出中。这也导致测试变得脆弱，因为即使其他地方的调用略有变化，这可能会在这个特定测试中导致行为变化，也会导致测试失败。这就是为什么`assert_has_calls`的默认行为是只确定预期的调用是否存在，而不是检查调用是否完全匹配。在需要完全匹配的罕见情况下，我们总是可以直接在`mock_calls`属性上断言。

## 模式 - 打开函数的修补

模拟中最常见的用例之一是模拟文件访问。这实际上有点繁琐，因为`open`函数可以用多种方式使用。它可以作为一个普通函数使用，也可以作为一个上下文管理器使用。数据可以通过`read`、`readlines`等方法进行读取。反过来，其中一些函数返回可以迭代的迭代器。为了在测试中使用它们，必须逐一模拟所有这些，这很痛苦。

幸运的是，模拟库提供了一个极其有用的`mock_open`函数，它可以返回一个处理所有这些情况的模拟。让我们看看我们如何使用这个函数。

下面的代码是`FileReader`的代码：

```py
class FileReader:
    """Reads a series of stock updates from a file"""
    def __init__(self, filename):
        self.filename = filename

    def get_updates(self):
        """Returns the next update everytime the method is called"""

        with open(self.filename, "r") as fp:
            for line in fp:
                symbol, time, price = line.split(",")
                yield (symbol, datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f"), int(price))
```

这个类从文件中读取股票更新，并逐个返回每个更新。这个方法是一个生成器，并使用`yield`关键字逐个返回更新。

### 注意

**关于生成器的一个快速入门**

**生成器**是使用`yield`语句而不是`return`语句来返回值的函数。每次执行生成器时，执行不会从函数的开始处开始，而是从上一个`yield`语句继续运行。在上面的例子中，当生成器被执行时，它会解析文件的第一个行，然后返回值。下一次执行时，它会再次通过循环继续运行，返回第二个值，然后是第三个值，依此类推，直到循环结束。每次执行生成器返回一个股票更新。有关生成器的更多信息，请查看 Python 文档或在线文章。这样一篇文章可以在[`www.jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/`](http://www.jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/)找到。

为了测试`get_update`方法，我们需要创建不同类型的文件数据，并验证该方法是否正确读取它们并返回预期的值。为了做到这一点，我们将模拟打开函数。以下是一个这样的测试：

```py
class FileReaderTest(unittest.TestCase):
    @mock.patch("builtins.open",
                mock.mock_open(read_data="""\
                GOOG,2014-02-11T14:10:22.13,10"""))
    def test_FileReader_returns_the_file_contents(self):
        reader = FileReader("stocks.txt")
        updater = reader.get_updates()
        update = next(updater)
        self.assertEqual(("GOOG",
                          datetime(2014, 2, 11, 14, 10, 22, 130000),
                          10), update)
```

在上述测试中，我们是从修补`builtins.open`函数开始的。`patch`装饰器可以接受第二个参数，其中我们可以指定修补后要使用的模拟对象。我们调用`mock.mock_open`函数来创建一个适当的模拟对象，并将其传递给`patch`装饰器。

`mock_open`函数接受一个`read_data`参数，其中我们可以指定当模拟文件被读取时应返回什么数据。我们使用此参数来指定我们想要测试的文件数据。

测试的其余部分相当简单。需要注意的是以下一行：

```py
updater = reader.get_updates()
```

由于`get_updates`是一个生成器函数，对`get_updates`方法的调用实际上并不返回股票更新，而是返回生成器对象。这个生成器对象存储在`updater`变量中。我们使用内置的`next`函数从生成器中获取股票更新，并断言它符合预期。

## 模式 - 使用可变参数进行模拟

一个可能会咬我们的是当模拟对象的参数是可变的时候。看看下面的例子：

```py
>>> from unittest import mock
>>> param = ["abc"]
>>> obj = mock.Mock()
>>> _ = obj(param)
>>> param[0] = "123"

>>> obj.assert_called_with(["abc"])
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "C:\Python34\lib\unittest\mock.py", line 760, in assert_called_with
 raise AssertionError(_error_message()) from cause
AssertionError: Expected call: mock(['abc'])
Actual call: mock(['123'])

```

哇！那里发生了什么？错误信息如下：

```py
AssertionError: Expected call: mock(['abc'])
Actual call: mock(['123'])

```

实际调用是`mock(['123'])`？但我们调用模拟的方式如下：

```py
>>> param = ["abc"]
>>> obj = mock.Mock()
>>> _ = obj(param)

```

很明显，我们是用`["abc"]`调用的它。那么为什么这个会失败？

答案是模拟对象只存储了对调用参数的引用。因此，当执行`param[0] = "123"`这一行时，它影响了存储在模拟中的调用参数的值。在断言中，它查看保存的调用参数，并看到调用使用了数据`["123"]`，所以断言失败。

显然的问题是：为什么模拟存储了参数的引用？为什么它不复制参数，这样如果稍后传递给参数的对象被更改，存储的副本就不会改变？答案是，复制创建了一个新对象，所以所有在参数列表中比较对象身份的断言都会失败。

那我们现在该怎么做？如何让这个测试工作起来？

简单：我们只是从`Mock`或`MagicMock`继承，并更改行为以复制参数，如下所示：

```py
>>> from copy import deepcopy
>>>
>>> class CopyingMock(mock.MagicMock):
...     def __call__(self, *args, **kwargs):
...         args = deepcopy(args)
...         kwargs = deepcopy(kwargs)
...         return super().__call__(*args, **kwargs)

```

这个模拟只是复制了参数，然后调用默认行为，传入复制的内容。

断言现在通过了，如下所示：

```py
>>> param = ["abc"]
>>> obj = CopyingMock()
>>> _ = obj(param)
>>> param[0] = "123"
>>> obj.assert_called_with(["abc"])

```

请记住，当我们使用`CopyingMock`时，我们不能使用任何对象身份比较作为参数，因为它们现在会失败，如下所示：

```py
>>> class MyObj:
...     pass
...
>>> param = MyObj()
>>> obj = CopyingMock()
>>> _ = obj(param)

>>> obj.assert_called_with(param)
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "C:\Python34\lib\unittest\mock.py", line 760, in assert_called_with
 raise AssertionError(_error_message()) from cause
AssertionError: Expected call: mock(<__main__.MyObj object at 0x00000000026BAB70>)
Actual call: mock(<__main__.MyObj object at 0x00000000026A8E10>)

```

# 摘要

在本章中，你研究了单元测试的一些其他模式。你研究了如何加快测试速度以及如何运行特定的测试子集。你研究了运行测试子集的各种模式，包括创建自己的测试套件和使用`load_tests`协议。你看到了如何使用 nose2 attrib 插件根据测试属性运行测试子集，以及如何使用默认单元测试运行器实现该功能。然后我们检查了跳过测试和标记测试为预期失败的功能。最后，你研究了如何编写数据驱动测试。

接下来，我们转向了一些模拟模式，首先是实现间谍功能的方法。你也研究了在多个模拟之间验证模拟调用序列的问题。然后你研究了`mock_open`函数，以帮助我们轻松模拟文件系统访问，在这个过程中你瞥了一眼如何与生成器函数一起工作。最后，你研究了在参数可变时使用模拟的问题。

下一章是这本书的最后一章，你将了解我们可以在我们的 TDD 实践中使用的其他工具。
