# 第七章. 使用 doctest 的可执行文档

在整本书中，我们强调了代码尽可能自文档化的必要性。我们提到了 Python 的酷炫 docstring 特性如何帮助我们实现这一目标。一般来说，文档存在一个问题——它很容易与代码不同步。很多时候我们看到代码有所改变，但相应的文档更改并未进行，导致新开发者对代码的实际工作方式感到困惑。现在，`doctest`模块来拯救我们。

`doctest`模块允许我们在 docstrings 中指定示例。然后模块提取示例，运行它们，并验证它们是否仍然有效。

# 我们的第一份 doctest

以下是在`Stock`类中`price`方法的当前版本：

```py
    def price(self):
        try:
            return self.history[-1].value
        except IndexError:
            return None
```

现在，在 docstring 中，我们添加了如何使用此方法的示例。这些示例基本上是 Python 交互式 shell 的复制粘贴。因此，包含要执行的输入的行以`>>>`提示符开头，而没有提示符的行表示输出，如下所示：

```py
    def price(self):
        """Returns the current price of the Stock

        >>> from datetime import datetime
        >>> stock = Stock("GOOG")
        >>> stock.update(datetime(2011, 10, 3), 10)
        >>> stock.price
        10
        """
        try:
            return self.history[-1].value
        except IndexError:
            return None
```

现在我们有了 docstring，我们需要一种执行它的方法。将以下行添加到文件底部：

```py
if __name__ == "__main__":
    import doctest
    doctest.testmod()
```

## 运行 doctest

现在，我们可以通过将文件作为模块执行来运行测试。我们需要将文件作为模块执行，以便相对导入工作。如果这是一个独立的脚本，或者我们使用了绝对导入而不是相对导入，那么我们就可以直接执行该文件。由于上面的代码位于`stock.py`文件中，我们必须执行`stock_alerter.stock`模块。以下是要执行的命令：

+   Windows: `python.exe -m stock_alerter.stock`

+   Linux/Mac: `python3 -m stock_alerter.stock`

当我们运行上述命令时，我们会得到没有任何输出的输出。是的，什么都没有。如果没有输出，那么这意味着所有的 doctests 都通过了。我们可以传递`-v`命令行参数（用于详细输出）来查看测试确实通过了，如下所示：

+   Windows: `python.exe -m stock_alerter.stock -v`

+   Linux/Mac: `python3 -m stock_alerter.stock -v`

当我们这样做时，我们会得到以下输出：

```py
Trying:
 from datetime import datetime
Expecting nothing
ok
Trying:
 stock = Stock("GOOG")
Expecting nothing
ok
Trying:
 stock.update(datetime(2011, 10, 3), 10)
Expecting nothing
ok
Trying:
 stock.price
Expecting:
 10
ok
8 items had no tests:
 __main__
 __main__.Stock
 __main__.Stock.__init__
 __main__.Stock._is_crossover_below_to_above
 __main__.Stock.get_crossover_signal
 __main__.Stock.is_increasing_trend
 __main__.Stock.update
 __main__.StockSignal
1 items passed all tests:
 4 tests in __main__.Stock.price
4 tests in 9 items.
4 passed and 0 failed.
Test passed.

```

让我们更详细地看看这个输出。

如果我们看我们的示例的第一行，它包含以下内容：

```py
>>> from datetime import datetime

```

`doctest`会提取这一行并评估输出：

```py
Trying:
 from datetime import datetime
Expecting nothing
ok

```

示例的下一行是另一个以`>>>`提示符开始的输入行，因此 doctest 认为执行第一行不应有任何输出，因此有语句*Expecting nothing*。

当测试的第一行执行时，确实没有打印任何内容，所以`doctest`给出输出*ok*，这意味着该行按预期执行。然后`doctest`继续到下一行，并遵循相同的程序，直到遇到以下行：

```py
>>> stock.price
10

```

我们的测试表明，当这一行执行时，应该打印出`10`。这正是 doctest 所检查的，如下所示：

```py
Trying:
 stock.price
Expecting:
 10
ok

```

### 注意

注意，我们没有明确调用`print(stock.price)`。我们只是放置了`stock.price`并期望输出为`10`。这正是我们在 Python 交互式外壳中看到的行为；`doctest`使用相同的行为。

在这一行之后，我们的示例结束，`doctest`继续到下一个方法，如下所示：

```py
8 items had no tests:
 __main__
 __main__.Stock
 __main__.Stock.__init__
 __main__.Stock._is_crossover_below_to_above
 __main__.Stock.get_crossover_signal
 __main__.Stock.is_increasing_trend
 __main__.Stock.update
 __main__.StockSignal
1 items passed all tests:
 4 tests in __main__.Stock.price
4 tests in 9 items.
4 passed and 0 failed.
Test passed.

```

它告诉我们剩余的方法没有测试，并且所有的测试都通过了。请注意，`doctest`将示例的每一行都视为一个单独的测试，这就是为什么它识别出四个测试的原因。由于 Python 支持模块级和类级文档字符串，我们也可以在那些地方放置一些示例，比如如何使用整个模块或类的示例。这就是为什么`doctest`也告诉我们`__main__`和`__main__.Stock`没有任何测试。

doctests 的美丽之处在于它允许我们在示例之间混合文档。这允许我们像以下这样扩展`price`方法的文档字符串：

```py
    def price(self):
        """Returns the current price of the Stock

        >>> from datetime import datetime
        >>> stock = Stock("GOOG")
        >>> stock.update(datetime(2011, 10, 3), 10)
        >>> stock.price
        10

        The method will return the latest price by timestamp, so even if updates are out of order, it will return the latest one

        >>> stock = Stock("GOOG")
        >>> stock.update(datetime(2011, 10, 3), 10)

        Now, let us do an update with a date that is earlier than the
        previous one

        >>> stock.update(datetime(2011, 10, 2), 5)

        And the method still returns the latest price

        >>> stock.price
        10

        If there are no updates, then the method returns None

        >>> stock = Stock("GOOG")
        >>> print(stock.price)
        None
        """
        try:
            return self.history[-1].value
        except IndexError:
            return None
```

运行上面的代码，它应该通过以下新的输出：

```py
Trying:
 stock = Stock("GOOG")
Expecting nothing
ok
Trying:
 stock.update(datetime(2011, 10, 3), 10)
Expecting nothing
ok
Trying:
 stock.update(datetime(2011, 10, 2), 5)
Expecting nothing
ok
Trying:
 stock.price
Expecting:
 10
ok
Trying:
 stock = Stock("GOOG")
Expecting nothing
ok
Trying:
 print(stock.price)
Expecting:
 None
ok

```

如我们所见，`doctest`遍历文档并识别出需要执行的确切行。这允许我们在代码片段之间放置解释和文档。结果？良好的解释文档加上可测试的代码。这是一个绝佳的组合！

让我们快速看一下最后一个例子：

```py
>>> stock = Stock("GOOG")
>>> print(stock.price)
None

```

如果你注意到，我们明确地打印了输出。原因是 Python 交互式外壳通常在值是`None`时不会给出任何输出。由于 doctest 模仿了交互式外壳的行为，我们本可以只留一个空行，测试就会通过，但这并不清楚发生了什么。所以，我们调用 print 来明确表示我们期望输出为`None`。

# 测试失败

现在我们来看一下测试失败的样子。以下是对`is_increasing_trend`方法的 doctest：

```py
    def is_increasing_trend(self):
        """Returns True if the past three values have been strictly
        increasing

        Returns False if there have been less than three updates so far

        >>> stock = Stock("GOOG")
        >>> stock.is_increasing_trend()
        False
        """

        return self.history[-3].value < \
            self.history[-2].value < self.history[-1].value
```

运行测试时，我们得到以下结果：

```py
Failed example:
 stock.is_increasing_trend()
Exception raised:
 Traceback (most recent call last):
 File "C:\Python34\lib\doctest.py", line 1324, in __run
 compileflags, 1), test.globs)
 File "<doctest __main__.Stock.is_increasing_trend[1]>", line 1, in <module>
 stock.is_increasing_trend()
 File "c:\Projects\tdd_with_python\src\stock_alerter\stock.py", line 91, in is_increasing_trend
 return self.history[-3].value < \
 File "c:\Projects\tdd_with_python\src\stock_alerter\timeseries.py", line 13, in __getitem__
 return self.series[index]
 IndexError: list index out of range
**********************************************************************
1 items had failures:
 1 of   2 in __main__.Stock.is_increasing_trend
***Test Failed*** 1 failures.

```

`doctest`告诉我们导致失败的是哪一行。它还告诉我们执行了哪个命令，以及发生了什么。我们可以看到，一个意外的异常导致测试失败。

我们现在可以像以下这样修复代码：

```py
    def is_increasing_trend(self):
        """Returns True if the past three values have been strictly increasing

        Returns False if there have been less than three updates so far

        >>> stock = Stock("GOOG")
        >>> stock.is_increasing_trend()
        False
        """

        try:
            return self.history[-3].value < \
               self.history[-2].value < self.history[-1].value
        except IndexError:
            return True
```

异常现在已经消失了，但我们在修复中有一个 bug，因为它已经被替换为失败，如下所示：

```py
Failed example:
 stock.is_increasing_trend()
Expected:
 False
Got:
 True

```

让我们修复它：

```py
    def is_increasing_trend(self):
        """Returns True if the past three values have been strictly increasing
        Returns False if there have been less than three updates so far

        >>> stock = Stock("GOOG")
        >>> stock.is_increasing_trend()
        False
        """

        try:
            return self.history[-3].value < \
               self.history[-2].value < self.history[-1].value
        except IndexError:
            return False
```

通过这次修复，所有的测试又都通过了。

# 测试异常

`update`方法应该在价格小于零时也引发`ValueError`。以下是在 doctest 中验证这一点的方法：

```py
    def update(self, timestamp, price):
        """Updates the stock with the price at the given timestamp

        >>> from datetime import datetime
        >>> stock = Stock("GOOG")
        >>> stock.update(datetime(2014, 10, 2), 10)
        >>> stock.price
        10

        The method raises a ValueError exception if the price is negative

        >>> stock.update(datetime(2014, 10, 2), -1)
        Traceback (most recent call last):
            ...
        ValueError: price should not be negative
        """

        if price < 0:
            raise ValueError("price should not be negative")
        self.history.update(timestamp, price)
        self.updated.fire(self)
```

下一个部分展示了`doctest`期望查看的内容：

```py
Traceback (most recent call last):
    ...
ValueError: price should not be negative
```

预期的输出以常规的 traceback 输出开始。这一行告诉`doctest`期望一个异常。之后是实际的 traceback。由于输出通常包含可能会改变的文件路径，因此很难完全匹配。幸运的是，我们不需要这样做。`doctest`允许我们使用三个缩进的点来表示 traceback 的中间部分。最后，最后一行显示了预期的异常和异常信息。这是匹配的行，用来查看测试是否通过。

# 包级别 doctests

如我们所见，doctests 可以针对方法、类和模块编写。然而，它们也可以针对整个包编写。通常，这些会放在包的`__init__.py`文件中，并展示整个包应该如何工作，包括多个相互作用的类。以下是从我们的`__init__.py`文件中的一个这样的 doctests 集：

```py
r"""
The stock_alerter module allows you to set up rules and get alerted when those rules are met.

>>> from datetime import datetime

First, we need to setup an exchange that contains all the stocks that are going to be processed. A simple dictionary will do.

>>> from stock_alerter.stock import Stock
>>> exchange = {"GOOG": Stock("GOOG"), "AAPL": Stock("AAPL")}

Next, we configure the reader. The reader is the source from where the stock updates are coming. The module provides two readers out of the box: A FileReader for reading updates from a comma separated file, and a ListReader to get updates from a list. You can create other readers, such as an HTTPReader, to get updates from a remote server.
Here we create a simple ListReader by passing in a list of 3-tuples containing the stock symbol, timestamp and price.

>>> from stock_alerter.reader import ListReader
>>> reader = ListReader([("GOOG", datetime(2014, 2, 8), 5)])

Next, we set up an Alert. We give it a rule, and an action to be taken when the rule is fired.

>>> from stock_alerter.alert import Alert
>>> from stock_alerter.rule import PriceRule
>>> from stock_alerter.action import PrintAction
>>> alert = Alert("GOOG > $3", PriceRule("GOOG", lambda s: s.price > 3),\
...               PrintAction())

Connect the alert to the exchange

>>> alert.connect(exchange)

Now that everything is setup, we can start processing the updates

>>> from stock_alerter.processor import Processor
>>> processor = Processor(reader, exchange)
>>> processor.process()
GOOG > $3
"""

if __name__ == "__main__":
    import doctest
    doctest.testmod()
```

我们可以像下面这样运行它：

+   **Windows**: `python.exe -m stock_alerter.__init__`

+   **Linux/Mac**: `python3 -m stock_alerter.__init__`

当我们这样做时，测试会通过。

关于这个测试，有几个需要注意的事项：

在 doctests 中，我们使用绝对导入而不是相对导入。例如，我们说`from stock_alerter.stock import Stock`而不是`from .stock import Stock`。这使我们能够轻松地从命令行运行 doctests。运行此 doctests 的另一种方法是：

+   **Windows**: `python.exe -m doctest stock_alerter\__init__.py`

+   **Linux/Mac**: `python3 -m doctest stock_alerter\__init__.py`

这种语法仅在文件使用绝对导入时才有效。否则，我们会得到错误`SystemError: Parent module '' not loaded, cannot perform relative import`。

通常，在进行包级别的 doctests 时，建议使用绝对导入。

除了这些，一些示例也跨越了多行。以下是一个这样的示例：

```py
>>> alert = Alert("GOOG > $3", PriceRule("GOOG", lambda s: s.price > 3),\
...               PrintAction())

```

支持多行的方法与交互式 shell 中的方法相同。在行尾使用反斜杠`\`，并在下一行开头使用三个点`...`。这被`doctest`解释为行续行，并将这两行合并为单个输入。

### 注意

**一个重要的注意事项**：注意，文档字符串以`r`前缀开始，如下所示`r"""`。这表示原始字符串。如上所述，我们在几个地方使用了反斜杠来表示输入的续行。当 Python 在字符串中找到一个反斜杠时，它将其解释为转义字符而不是字面反斜杠。解决方案是使用双反斜杠`\\`来转义反斜杠，或者使用不进行反斜杠解释的原始字符串。与其在所有地方都使用双反斜杠，不如使用带有`r`前缀标记的原始字符串来标记文档字符串的开始，这样更可取。

# 维护 doctests

Doctests 可能会非常冗长，通常包含大量的解释和示例混合在一起。这些 doctests 很容易扩展到多页。有时，可能会有很多行 doctests 后面只跟着几行代码。我们可以在`update`方法中看到这种情况。这可能会使代码导航变得更加困难。

我们可以通过将 doctests 放入单独的文件来解决这个问题。假设，我们将文档字符串的内容放入一个名为`readme.txt`的文件中。然后我们像下面这样更改我们的`__init__.py`文件：

```py
if __name__ == "__main__":
    import doctest
    doctest.testfile("readme.txt")
```

现在将加载`readme.txt`的内容并将其作为 doctests 运行。

当在外部文件中编写测试时，没有必要像在 Python 文件中那样在内容周围放置引号。整个文件内容都被视为 doctests。同样，我们也不需要转义反斜杠。

这个特性使得将所有 doctests 放入单独的文件变得实用。这些文件应该作为用户文档使用，并包含其中的 doctests。这样，我们就可以避免在代码中添加大量 doctrings 而造成混乱。

## 运行一系列 doctests

`doctest`模块缺失的一个特性是有效的自动发现机制。与`unittest`模块不同，后者会搜索所有文件以查找测试并运行它们，而 doctest 则需要我们显式地在命令行上执行每个文件。这对大型项目来说是一个大麻烦。

虽然有一些方法可以实现这一点。最直接的方法是将 doctests 包装在`unittest.TestCase`类中，如下所示：

```py
import doctest
import unittest
from stock_alerter import stock

class PackageDocTest(unittest.TestCase):
    def test_stock_module(self):
        doctest.testmod(stock)

    def test_doc(self):
        doctest.testfile(r"..\readme.txt")
```

这些 doctests 可以像通常一样与单元测试一起运行。

这确实可行，但问题是如果 doctests 中发生失败，测试不会失败。错误会被打印出来，但不会记录失败。如果手动运行测试，这没问题，但如果以自动化的方式运行测试，例如作为构建或部署过程的一部分，就会造成问题。

`doctest`还有一个特性，可以通过它将 doctests 包装在`unittest`中：

```py
import doctest
from stock_alerter import stock

def load_tests(loader, tests, pattern):
    tests.addTests(doctest.DocTestSuite(stock))
    tests.addTests(doctest.DocFileSuite("../readme.txt"))
    return tests
```

我们之前没有看过`load_tests`，现在让我们快速看一下。`load_tests`是由`unittest`模块用来从当前模块加载单元测试套件的。当这个函数不存在时，`unittest`会使用其默认方法通过查找继承自`unittest.TestCase`的类来加载测试。然而，当这个函数存在时，它会被调用，并且可以返回一个与默认不同的测试套件。然后返回的套件会被运行。

由于 doctests 不是`unittest.TestCase`的一部分，因此默认情况下不会在执行单元测试时运行。我们做的是实现`load_tests`函数，并在该函数中将 doctests 添加到测试套件中。我们使用`doctest.DocTestSuite`和`doctest.DocFileSuite`方法从 doctests 创建与`unittest`兼容的测试套件。然后我们将这些测试套件追加到`load_tests`函数中要执行的总体测试中。

`doctest.DocTestSuite` 接收包含测试的模块作为参数。

### 注意

注意，我们必须传入实际的模块对象，而不仅仅是字符串。

`doctest.DocFileSuite` 接收包含 doctests 的文件名。文件名相对于当前测试模块的目录。例如，如果我们的目录结构如下所示：

```py
src
|
+- stock_alerter
   |
   +- readme.txt
   +- tests
      |
      +- test_doctest.py
```

然后，我们将在 `test_doctest.py` 中使用路径 `../readme.txt` 来引用此文件。

或者，我们可以指定一个包名，路径可以相对于该包，如下所示：

```py
tests.addTests(doctest.DocFileSuite("readme.txt",
                                    package="stock_alerter"))
```

## 设置和拆卸

doctests 的一个问题是我们必须显式设置 docstring 内部的所有内容。例如，以下是我们之前编写的 `update` 方法的 doctest：

```py
>>> from datetime import datetime
>>> stock = Stock("GOOG")
>>> stock.update(datetime(2011, 10, 3), 10)
>>> stock.price
10

```

在第一行，我们导入 `datetime` 模块。这与示例无关，会使示例变得杂乱，但我们必须添加它，否则我们将得到以下错误：

```py
Failed example:
 stock.update(datetime(2011, 10, 3), 10)
Exception raised:
 Traceback (most recent call last):
 ...
 NameError: name 'datetime' is not defined

```

有没有避免这些行重复的方法？是的，有。

`DocFileSuite` 和 `DocTestSuite` 都接受一个 `globs` 参数。此参数接受一个字典，其中包含用于 doctests 的全局变量项，它们可以通过示例访问。以下是我们如何做到这一点：

```py
import doctest
from datetime import datetime
from stock_alerter import stock

def load_tests(loader, tests, pattern):
    tests.addTests(doctest.DocTestSuite(stock, globs={
        "datetime": datetime,
        "Stock": stock.Stock
    }))
    tests.addTests(doctest.DocFileSuite("readme.txt", package="stock_alerter"))
    return tests
```

### 注意

注意，我们必须传入的不仅是 `datetime` 模块，还有 `Stock` 类。默认情况下，`doctest` 使用模块自己的全局变量在执行上下文中。这就是为什么我们之前能够在 doctests 中使用 `Stock` 类。当我们通过 `globs` 参数替换执行上下文时，我们必须显式设置 `Stock` 对象为执行上下文的一部分。

`DocFileSuite` 和 `DocTestSuite` 也接受 `setUp` 和 `tearDown` 参数。这些参数接受一个函数，该函数将在每个 doctest 之前和之后被调用。这是一个执行任何测试所需的环境设置或拆卸的好地方。该函数还传递了一个 `DocTest` 对象，可以在设置和拆卸过程中使用。`DocTest` 对象有许多属性，但最常用的是 `globs` 属性。这是执行上下文的字典，可以在设置中添加以实例化将在对象之间重用的对象。以下是一个这样的使用示例：

```py
import doctest
from datetime import datetime
from stock_alerter import stock

def setup_stock_doctest(doctest):
    s = stock.Stock("GOOG")
    doctest.globs.update({"stock": s})

def load_tests(loader, tests, pattern):
    tests.addTests(doctest.DocTestSuite(stock, globs={
        "datetime": datetime,
        "Stock": stock.Stock
    }, setUp=setup_stock_doctest))
    tests.addTests(doctest.DocFileSuite("readme.txt", package="stock_alerter"))
    return tests
```

通过实例化和将股票传递给 doctests，我们可以消除在单个测试中实例化它的需要，因此测试最初如下所示：

```py
    def is_increasing_trend(self):
        """Returns True if the past three values have been strictly
        increasing

        Returns False if there have been less than three updates so far

        >>> stock = Stock("GOOG")
        >>> stock.is_increasing_trend()
        False
        """
```

现在测试变为以下内容：

```py
    def is_increasing_trend(self):
        """Returns True if the past three values have been strictly
        increasing

        Returns False if there have been less than three updates so far

        >>> stock.is_increasing_trend()
        False
        """
```

为什么我们通过 `setUp` 函数实例化和传递 `stock` 而不是使用 `glob` 参数？原因是我们想要为每个测试创建一个新的 `Stock` 实例。由于 `setUp` 和 `tearDown` 在每个测试之前被调用，因此每次都会将一个新的 `stock` 实例添加到 `doctest.glob` 中。

# doctest 的限制

`doctest` 的最大限制是它只比较打印的输出。这意味着任何可能变化的输出都会导致测试失败。以下是一个示例：

```py
>>> exchange
{'GOOG': <stock_alerter.stock.Stock object at 0x00000000031F8550>, 'AAPL': <stock_alerter.stock.Stock object at 0x00000000031F8588>}

```

此 doctest 有可能因为两个原因而失败：

+   Python 不保证字典对象打印的顺序，这意味着它可能以相反的顺序打印出来，有时会导致失败

+   `Stock` 对象的地址每次可能都不同，因此这部分在下次测试运行时将无法匹配

第一个问题的解决方案是确保输出是确定的。例如，以下方法将有效：

```py
>>> for key in sorted(exchange.keys()):
...    print(key, exchange[key])
...
AAPL <stock_alerter.stock.Stock object at 0x00000000031F8550>
GOOG <stock_alerter.stock.Stock object at 0x00000000031F8588>

```

尽管如此，仍然存在对象地址的问题。为了解决这个问题，我们需要使用 doctest 指令。

## Doctest 指令

`doctest` 支持许多指令，这些指令会改变模块的行为。

我们将要查看的第一个指令是 `ELLIPSIS`。此指令允许我们使用三个点 `...` 来匹配任何文本。我们可以使用它来匹配对象地址，如下所示：

```py
>>> for key in sorted(exchange.keys()): #doctest: +ELLIPSIS
...    print(key, exchange[key])
...
AAPL <stock_alerter.stock.Stock object at 0x0...>
GOOG <stock_alerter.stock.Stock object at 0x0...>

```

现在示例将通过。

`...` 将匹配运行时打印的任何地址。我们通过在示例中添加注释 `#doctest: +ELLIPSIS` 来启用此指令。这将仅为此示例启用指令。同一 doctest 中的后续示例将关闭，除非它们被特别启用。

一些常用的指令包括：

+   `NORMALIZE_WHITESPACE`: 默认情况下，doctest 会精确匹配空白。一个空格不会与制表符匹配，并且换行符不会匹配，除非它们位于完全相同的位置。有时，我们可能想要通过换行或缩进来美化预期的输出，使其更容易阅读。在这种情况下，可以将 `NORMALIZE_WHITESPACE` 指令设置为 doctest 将所有空白视为相等。

+   `IGNORE_EXCEPTION_DETAIL`: 当匹配异常时，`doctest` 会查看异常的类型以及异常消息。当此指令启用时，仅检查类型是否匹配。

+   `SKIP`: 带有此指令的示例将被完全跳过。这可能是因为文档有意显示一个不工作或输出随机的示例。它也可以用来注释掉不工作的 doctests。

+   `REPORT_ONLY_FIRST_FAILURE`: 默认情况下，`doctest` 在失败后将继续执行后续示例，并将报告这些示例的失败。很多时候，一个示例的失败会导致后续示例的失败，并可能导致许多错误报告，这使得很难识别导致所有其他失败的第一个失败的示例。此指令将仅报告第一个失败。

这不是指令的完整列表，但它们涵盖了最常用的指令。

可以在单独的行上给出多个指令，或者用逗号分隔。以下将有效：

```py
>>> for key in sorted(exchange.keys()):
...    print(key, exchange[key])
...    #doctest: +ELLIPSIS
...    #doctest: +NORMALIZE_WHITESPACE
AAPL       <stock_alerter.stock.Stock object at 0x0...>
GOOG       <stock_alerter.stock.Stock object at 0x0...>

```

或者，以下也可以工作：

```py
>>> for key in sorted(exchange.keys()):
...    print(key, exchange[key])
...    #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
AAPL       <stock_alerter.stock.Stock object at 0x0...>
GOOG       <stock_alerter.stock.Stock object at 0x0...>

```

指令也可以通过 `optionflags` 参数传递给 `DocFileSuite` 和 `DocTestSuite`。当以以下方式传递时，指令对整个文件或模块生效：

```py
options = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
tests.addTests(doctest.DocFileSuite("readme.txt",
                                    package="stock_alerter",
                                    optionflags=options))
```

在 doctest 中，我们可以根据需要关闭某些指令，如下所示：

```py
>>> for key in sorted(exchange.keys()):
...    print(key, exchange[key])
... #doctest: -NORMALIZE_WHITESPACE
AAPL <stock_alerter.stock.Stock object at 0x0...>
GOOG <stock_alerter.stock.Stock object at 0x0...>

```

使用指令是选择性地启用或禁用 doctests 中特定行为的好方法。

# doctests 如何与 TDD 流程相结合？

现在我们对 doctests 有了一个相当好的了解，接下来的问题是：这如何与 TDD 流程相结合？记住，在 TDD 流程中，我们首先编写测试，然后编写实现。doctests 是否适合这个流程？

在某种程度上，是的。doctests 并不适合用于单个方法的 TDD。对于这些，`unittest` 模块是更好的选择。`doctest` 发挥作用的地方在于包级别的交互。穿插着示例的解释真正展示了包内不同模块和类之间的交互。这样的 doctests 可以在开始时编写出来，为我们想要整个包如何工作提供一个高级概述。这些测试将会失败。随着单个类和方法被编写，测试将开始通过。

# 摘要

在本章中，你了解了 Python 的 `doctest` 模块。你看到了它是如何帮助你将示例嵌入到文档字符串中的。你查看了几种编写 doctests 的方法，包括方法和包文档字符串。你还看到了如何将包级别的 doctests 移动到单独的文件中并运行它们。维护 doctests 很重要，你查看了一些更好的维护 doctests 的方法，包括使用设置和清理以及将它们包含在常规测试套件中。最后，你查看了一些限制以及如何使用指令来克服一些限制。

在下一章中，你将通过查看 `nose2` 包来首次了解第三方工具。
