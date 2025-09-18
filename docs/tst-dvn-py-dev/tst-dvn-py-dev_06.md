# 第六章：维护您的测试套件

如果我们定期进行 TDD，我们可能会很容易地拥有一个包含数千个测试的大测试套件。这很好——它给了我们很大的信心，大胆地添加新功能而不用担心破坏旧功能。然而，我们使测试维护变得容易是至关重要的，否则我们很快就会陷入仅仅管理测试的混乱之中。

没有编写为维护的测试很快就会带来许多头疼问题。散布在文件系统中的测试将使得定位特定测试变得不可能。难以阅读的测试在测试需要因功能更改而更改时，理解和修复将变得困难。长而编写不佳的测试将带来与低质量生产代码相同的问题。脆弱的测试将确保即使是微小的更改也会破坏大量测试。

记住，测试代码仍然是代码。就像任何生产代码一样，我们必须尽力保持其可读性、可维护性和易于更改。我们很容易陷入编写测试用例后忘记它们的陷阱。一年后，我们发现维护测试是一个巨大的头疼问题，添加新功能比以前更困难。

# 测试维护的目标

正如我们在整本书中看到的那样，单元测试服务于多种不同的目的：

+   **作为测试套件**：这是单元测试最明显的目标。一个全面的测试套件可以减少可能逃逸到生产环境的错误数量。

+   **作为文档**：当我们试图了解一个类或方法试图做什么时，查看测试套件是有用的。一个编写良好的测试套件将说明代码片段应该如何表现。

+   **作为安全网**：这使我们重构或清理代码时的压力得以释放。

+   **展示设计**：使用模拟，我们可以描绘不同类或模块之间的交互。

一套编写良好的单元测试的目标是尽可能实现以下目的。

例如，如果我们想了解一个方法的作用，那么我们需要考虑以下问题：

+   我们能否轻松地找到它的单元测试套件？

+   一旦找到，理解测试及其测试内容是否容易？

+   我们能否理解这个方法如何与其他类交互？

+   一旦我们了解了方法的作用，那么重构它是否容易？进行小的重构是否会破坏所有测试？

+   是否容易识别由于我们的重构而失败的测试，以及哪些失败是重构中的错误？

+   如果因为重构需要修改测试用例，那么理解测试并做出所需更改有多困难？

任何大型、长期项目都将涉及回答所有这些问题。

我们在本章中的目标是探讨使执行上述活动更容易的方法。

# 组织测试

简化测试维护的第一步是有一个系统化的测试组织方式。当我们实现一个新功能时，我们需要能够快速轻松地找到给定方法或类的现有测试代码。为此，有三个步骤。我们必须决定：

+   文件在文件系统中的存储位置

+   文件被称为什么

+   测试类的名称

## 文件系统布局

决定将我们的测试代码放在哪里的主要考虑因素是，我们能够多容易地找到特定类或模块的测试。除此之外，还有两个其他考虑因素要记住：

+   这个模块将如何打包和分发？

+   这段代码将如何投入生产？

对于第一个考虑因素，我们必须记住，我们希望将单元测试与主代码一起分发。对于投入生产的代码，我们可能并不总是希望部署测试，因此我们正在寻找一种将代码和测试分离的方法。

考虑到所有这些因素，有两种流行的测试代码布局方式。

在第一种模式中，测试代码被放置在主代码的子模块中，如下所示：

```py
module
|
+- file
+- file
+- tests
   |
   + test_file
   + test_file
```

这允许我们通过仅压缩模块目录并将其放入`egg`文件中来打包模块和测试。整个包是自包含的，可以直接使用。模块通过`import module`语句访问，测试通过使用`import module.tests`访问。

运行测试也不需要配置，因为测试可以通过相对导入（如`from ..file import class`）访问测试代码中的类。

这种模式非常适合将独立模块打包成`egg`文件并分发。

另一种模式是将测试保留在完全独立的文件夹层次结构中。文件布局可能如下所示：

```py
root
|
+- module
|  |
|  +- file
|  +- file
|
+- tests
   |
   + test_file
   + test_file
```

前面的模式适用于需要将测试从代码中分离出来的情况。这可能是因为我们正在制作一个产品，不希望向客户发送测试，或者可能是因为我们不希望将测试部署到生产服务器。这种模式适合，因为它很容易将模块单独压缩成`egg`文件并部署。

这种模式的缺点是，如果项目涉及许多模块，那么分离不同模块的测试可能会很麻烦。一种模式是使用每个目录的不同根，如下所示：

```py
root
|
+- src
|  |
|  +- module1
|     |
|     +- file
|     +- file
|
+- tests
   |
   +- module1
      |
      + test_file
      + test_file
```

前面的模式在 Java 等语言中很常见，但总体来说相当繁琐且冗长。然而，它可能是解决前面提到的特定需求的一种解决方案。

我们绝对不想做的两种模式如下：

+   将测试放在与代码相同的文件中

+   将测试文件放在与代码相同的目录中

将测试放在与代码相同的文件中对于单文件脚本来说是可行的，但对于更大的项目来说则显得杂乱无章。问题是如果文件很长，那么在文件中导航就变得非常困难。而且由于测试通常需要导入其他类，这会污染命名空间。它还增加了循环导入的可能性。最后，由于测试运行器会在文件名中寻找模式，这使得一次性执行所有测试变得困难。

至于将文件放在同一个目录中，这又一次打乱了模块，使得在目录中查找文件变得困难。除了不需要创建单独的测试目录之外，它没有任何特定的优势。避免这种模式，而是创建一个子模块。

## 命名约定

下一步是确定测试文件、测试类和测试方法的命名约定。

对于测试文件，最好使用`test_`前缀。例如，名为`file1.py`的文件将会有其测试在`test_file1.py`中。这使得我们在查看生产代码时能够轻松地找到相应的测试代码。使用`test_`前缀是首选的，因为这是大多数测试运行器搜索的默认模式。如果我们使用其他前缀，或者使用后缀如`file1_test.py`，那么我们很可能需要将额外的配置传递给测试运行器以找到和执行测试。通过坚持大多数常用工具期望的默认约定，我们可以轻松避免这种额外的配置。

例如，我们可以使用以下命令来运行测试：

```py
python3 -m unittest discover

```

但如果我们想要用后缀`_test`来命名我们的测试，那么我们必须使用以下命令：

```py
python3 -m unittest discover -p *_test.py

```

它是可行的，但这只是可以避免的额外配置。我们只有在需要保留旧命名约定的情况下才应该使用它。

那么关于测试类和测试方法呢？`unittest`模块会查看所有继承自`unittest.TestCase`的类。因此，类的名称并不重要。然而，其他测试运行器，如`nose2`，也会根据类的名称来选择测试类。默认模式是搜索以`Test`结尾的类。因此，将所有测试类命名为以`Test`结尾是有意义的。这也很有描述性，所以实际上没有很好的理由去做其他的事情。

同样，测试方法应该以`test`开头。这是测试运行器搜索的默认模式，因此坚持这个约定是有意义的。不以`test`开头的方法可以用作辅助方法。

## 测试套件分组

最后，我们来到了一个关于一个类应该包含什么的问题——一个测试类是否应该包含目标类的测试，或者我们应该将所有方法的测试都存储在一个测试类中？在这里，这更多的是一个个人偏好的问题。两者在客观上没有好坏之分，只是取决于哪个更易读。我个人的偏好是在单个代码库中使用这两种模式，根据测试的数量和哪种模式更容易找到我正在寻找的测试。一般来说，如果一个方法有很多测试，那么我会将它们重构到一个单独的测试类中。

# 使测试可读

在上一节中，我们探讨了相对平凡的问题，即文件布局和命名约定。我们现在将探讨我们可以改进测试用例本身的方法。

我们的首要目标是使理解测试本身变得更简单。没有什么比找到测试用例然后很难弄清楚测试试图做什么更糟糕的了。

我相信我不会是第一个承认自己多次回到一年前自己写的测试，并努力理解自己当时试图做什么的人。

这通常是一个被忽视的领域，因为当我们编写测试时，似乎很明显测试在做什么。我们需要设身处地地想象自己是一个第一次或几年后查看测试的人，试图在没有我们编写测试时的上下文知识的情况下理解测试。这是在处理大型代码库时经常出现的问题。

## 使用文档字符串

对于一个难以阅读的测试，第一道防线是使用文档字符串。**文档字符串**是 Python 的一个伟大特性，因为它们在运行时可用。测试运行器通常会拾取文档字符串，并在测试错误和失败时显示它们，这使得从测试报告中直接看到失败内容变得容易。

有些人会说，一份写得好的测试不需要额外的解释。实际上，我们在第三章*代码异味和重构*中讨论注释的价值时，也说过类似的话。为了重复我们当时的话：解释正在发生什么的注释没有价值，但解释为什么我们以这种方式实现代码的注释是有价值的。同样的原则也适用于文档字符串。

例如，看看以下代码：

```py
    def get_closing_price_list(self, on_date, num_days):
        closing_price_list = []
        for i in range(num_days):
            chk = on_date.date() - timedelta(i)
            for price_event in reversed(self.series):
                if price_event.timestamp.date() > chk:
                    pass
                if price_event.timestamp.date() == chk:
                    closing_price_list.insert(0, price_event)
                    break
                if price_event.timestamp.date() < chk:
                    closing_price_list.insert(0, price_event)
                    break
        return closing_price_list
```

这是来自`TimeSeries`类的`get_closing_price_list`方法，我们在第三章*代码异味和重构*中对其进行了重构。以下是对该方法的测试：

```py
class TimeSeriesTest(unittest.TestCase):
    def test_closing_price_list_before_series_start_date(self):
        series = TimeSeries()
        series.update(datetime(2014, 3, 10), 5)
        on_date = datetime(2014, 3, 9)
        self.assertEqual([], series.get_closing_price_list(on_date, 1))
```

此测试检查，如果传递的日期早于时间序列的开始，则返回空列表。从测试中可以清楚地看出这一点。但为什么它返回空列表而不是抛出异常呢？文档字符串是解释这种设计决策的好地方，如下所示：

```py
    def test_closing_price_list_before_series_start_date(self):
        """
        Empty list is returned if on_date is before the start of the
        series
        The moving average calculation might be done before any data
        has been added to the stock. We return an empty list so that
        the calculation can still proceed as usual.
        """
        series = TimeSeries()
        series.update(datetime(2014, 3, 10), 5)
        on_date = datetime(2014, 3, 9)
        self.assertEqual([], series.get_closing_price_list(on_date, 1))
```

## 使用固定装置

在查看文档字符串后，我们现在可以将注意力转向测试本身。

如果我们查看单元测试的一般结构，它们通常遵循安排-行动-断言的结构。在这些中，行动部分通常只有几行，断言部分最多也只有几行。到目前为止，测试的最大部分是在安排部分。对于更复杂的测试，其中特定场景可能需要多行来设置，安排部分可能占整个测试的 75%到 80%。

避免重复代码的一种方法是将所有内容都移动到适当的`setUp`和`tearDown`方法中。正如我们在第二章中看到的，“红-绿-重构 – TDD 循环”，`unittest`提供了三个级别的设置和清理方法：

+   在每个测试之前和之后运行的`setUp`和`tearDown`

+   在每个测试类之前和之后运行的`setUpClass`和`tearDownClass`

+   在每个测试文件之前和之后运行的`setUpModule`和`tearDownModule`

这种为测试设置数据的方法被称为**固定装置**。使用固定装置可以减少测试之间的代码重复，这是一个好主意。然而，也有一些需要注意的缺点：

+   有时，需要做很多设置，但每个测试只使用整体固定装置的一小部分。在这种情况下，对于新开发者来说，弄清楚每个测试使用固定装置的哪一部分可能会很困惑。

+   使用类和模块级别的固定装置时，我们必须小心。因为固定装置在多个测试之间共享，我们必须小心不要改变固定装置的状态。如果我们这样做，那么一个测试的结果可能会改变下一个测试的固定装置状态。这可能导致在执行顺序不同时出现非常奇怪的错误。

需要注意的一件事是，如果`setUp`方法抛出异常，则不会调用`tearDown`方法。以下是一个示例：

```py
class SomeTest(unittest.TestCase):
    def setUp(self):
        connect_to_database()
        connect_to_server()

    def tearDown(self):
        disconnect_from_database()
        disconnect_from_server()
```

如果在`connect_to_server`调用中抛出异常，则不会调用`tearDown`方法。这将导致数据库连接保持打开状态。当为下一个测试调用`setUp`时，第一行可能会失败（因为连接已经打开），导致所有其他测试失败。

为了避免这种情况，`unittest`模块提供了`addCleanup`方法。此方法接受一个回调函数，无论设置是否通过都会调用，如下所示：

```py
class SomeTest2(unittest.TestCase):
    def setUp(self):
        connect_to_database()
        self.addCleanup(self.disconnect_database)
        connect_to_server()
        self.addCleanup(self.disconnect_server)

    def disconnect_database(self):
        disconnect_from_database()

    def disconnect_server(self):
        disconnect_from_server()
```

使用这种结构，执行流程如下：

+   如果数据库调用失败，则不会执行清理操作

+   如果数据库调用成功了但服务器调用失败了，那么在清理过程中将调用`disconnect_database`。

+   如果两个调用都成功了，那么在清理过程中将调用`disconnect_database`和`disconnect_server`。

我们在什么时候使用`addCleanup`而不是`tearDown`？一般来说，当我们访问必须关闭的资源时，`addCleanup`是最佳选择。`tearDown`是一个放置其他类型清理的好地方，或者在`setUp`无法抛出异常的情况下。

## 固定和打补丁

在使用打补丁的模拟和固定一起使用时，有一个复杂的问题。看看以下代码：

```py
@mock.patch.object(smtplib, "SMTP")
class EmailActionTest(unittest.TestCase):
    def setUp(self):
        self.action = EmailAction(to="siddharta@silverstripesoftware.com")

    def test_connection_closed_after_sending_mail(self, mock_smtp_class):
        mock_smtp = mock_smtp_class.return_value
        self.action.execute("MSFT has crossed $10 price level")
        mock_smtp.send_message.assert_called_with(mock.ANY)
        self.assertTrue(mock_smtp.quit.called)
        mock_smtp.assert_has_calls([
            mock.call.send_message(mock.ANY),
            mock.call.quit()])
```

这是之前我们查看的`EmailAction`类的一个测试。在类级别使用`patch`装饰器来打补丁`smtplib.SMTP`类，并将模拟对象作为参数传递给所有测试用例。由于`patch`装饰器的工作方式，它只将模拟对象传递给测试用例方法，这意味着我们无法在`setUp`方法中访问它。

如果我们看看这个测试，它使用了从`mock_smtp_class`派生的`mock_smtp`对象。获取`mock_smtp`对象的行可以移动到`setUp`方法中，如果我们能访问到`mock_smtp_class`的话。有没有一种方法可以在`setUp`方法中应用补丁，这样我们就可以做一些常见的设置了？

幸运的是，`unittest`模块为我们提供了完成这项任务的工具。我们不会使用装饰器语法来打补丁，而是会使用以下类似的常规对象语法：

```py
    def setUp(self):
        patcher = mock.patch("smtplib.SMTP")
        self.addCleanup(patcher.stop)
        self.mock_smtp_class = patcher.start()
        self.mock_smtp = self.mock_smtp_class.return_value
        self.action = EmailAction(to="siddharta@silverstripesoftware.com")
```

我们在这里所做的是将需要打补丁的对象——在这种情况下是`smtplib.SMTP`——传递给`patch`函数。这返回一个具有两个方法：`start`和`stop`的补丁器对象。当我们调用`start`方法时，补丁被应用，当我们调用`stop`方法时，补丁被移除。

我们通过将其传递给`addCleanup`函数来设置`patcher.stop`方法在测试清理阶段执行。然后我们开始打补丁。`start`方法返回模拟对象，我们将其用于剩余的设置。

使用这种设置，我们可以在测试中直接使用`self.mock_smtp`，而无需在每次测试中都从`mock_smtp_class`获取它。现在的测试看起来如下：

```py
    def test_connection_closed_after_sending_mail(self):
        self.action.execute("MSFT has crossed $10 price level")
        self.mock_smtp.send_message.assert_called_with(mock.ANY)
        self.assertTrue(self.mock_smtp.quit.called)
        self.mock_smtp.assert_has_calls([
            mock.call.send_message(mock.ANY),
            mock.call.quit()])
```

将这个测试与这一节中较早的测试进行比较。由于我们不再使用装饰器补丁语法，我们不再需要额外的参数。我们也不需要在每个测试中都从`mock_smtp_class`派生`mock_smtp`。相反，所有这些工作都在`setUp`中完成。然后测试可以访问`self.mock_smtp`并直接使用它。

## 使用自定义测试用例类层次结构

减少代码重复的另一种方法是我们创建自己的测试类层次结构。例如，如果一个辅助方法在许多测试类中经常被使用，那么我们可以将它提升到更高一级的类中，并从该类继承测试类。以下是一个使概念更清晰的示例：

```py
class MyTestCase(unittest.TestCase):
    def create_start_object(self, value):
        do_something(value)

class SomeTest(MyTestCase):
    def test_1(self):
        create_start_object("value 1")

class SomeTest2(MyTestCase):
    def test_2(self):
        create_start_object("value 2")

    def test_3(self):
        create_start_object("value 3")
```

在这个例子中，我们创建了一个名为`MyTestCase`的类，它继承自`unittest.TestCase`，并在该类中放入了一些辅助方法。实际的测试类继承自`MyTestCase`，可以访问父类中的辅助方法。

使用这种技术，我们可以将常见的辅助方法组合放入可重用的父类中。层次结构不必只有一层深；有时，我们可能需要为测试的具体应用区域创建更进一步的子类。

# 在领域附近编写测试

使测试更容易阅读的另一种方法是使用领域语言编写测试，而不是仅仅使用`unittest`提供的通用函数。在本节中，我们将探讨一些实现这一目标的方法。

## 编写辅助方法

第一种技术是编写辅助方法。我们在本书中较早地使用了这种方法。以下是一些没有使用辅助方法的测试用例：

```py
    def test_increasing_trend_is_false_if_price_decreases(self):
        timestamps = [datetime(2014, 2, 11), datetime(2014, 2, 12),
                      datetime(2014, 2, 13)]
        prices = [8, 12, 10]
        for timestamp, price in zip(timestamps, prices):
            self.goog.update(timestamp, price)
        self.assertFalse(self.goog.is_increasing_trend())

    def test_increasing_trend_is_false_if_price_equal(self):
        timestamps = [datetime(2014, 2, 11), datetime(2014, 2, 12),
                      datetime(2014, 2, 13)]
        prices = [8, 10, 10]
        for timestamp, price in zip(timestamps, prices):
            self.goog.update(timestamp, price)
        self.assertFalse(self.goog.is_increasing_trend())
```

虽然测试用例很短，但测试中实际发生的事情并不清晰。让我们将其中一些代码移动到辅助方法中，如下所示：

```py
    def given_a_series_of_prices(self, prices):
        timestamps = [datetime(2014, 2, 10), datetime(2014, 2, 11),
                      datetime(2014, 2, 12), datetime(2014, 2, 13)]
        for timestamp, price in zip(timestamps, prices):
        self.goog.update(timestamp, price)
```

以下是用新辅助方法编写的相同两个测试用例：

```py
    def test_increasing_trend_is_false_if_price_decreases(self):
        self.given_a_series_of_prices([8, 12, 10])
        self.assertFalse(self.goog.is_increasing_trend())

    def test_increasing_trend_is_false_if_price_equal(self):
        self.given_a_series_of_prices([8, 10, 10])
        self.assertFalse(self.goog.is_increasing_trend())
```

如我们所见，测试现在清晰多了。这是因为辅助方法清楚地表达了计算的意图，使得新开发者能够将测试用例中的步骤与他们对需求的心理模型相对应。

## 编写更好的断言

提高测试可读性的简单方法是编写我们自己的断言方法，这些方法比`unittest`提供的通用断言级别更高。例如，假设我们想要编写一个测试来验证股票的价格历史。以下是这样测试可能看起来：

```py
class TimeSeriesEqualityTest(unittest.TestCase):
    def test_timeseries_price_history(self):
        series = TimeSeries()
        series.update(datetime(2014, 3, 10), 5)
        series.update(datetime(2014, 3, 11), 15)
        self.assertEqual(5, series[0].value)
        self.assertEqual(15, series[1].value)
```

现在，编写测试的另一种方式如下：

```py
class TimeSeriesTestCase(unittest.TestCase):
    def assert_has_price_history(self, price_list, series):
        for index, expected_price in enumerate(price_list):
            actual_price = series[index].value
            if actual_price != expected_price:
                raise self.failureException("[%d]: %d != %d".format(index, expected_price, actual_price))

class TimeSeriesEqualityTest(TimeSeriesTestCase):
    def test_timeseries_price_history(self):
        series = TimeSeries()
        series.update(datetime(2014, 3, 10), 5)
        series.update(datetime(2014, 3, 11), 15)
        self.assert_has_price_history([5, 15], series)
```

在前面的例子中，我们创建了自己的基测试类和一个自定义的断言方法。测试用例继承自这个基测试类，并在测试中使用这个断言。

`assert_has_price_history`方法的实现给出了一种如何简单编写我们自己的断言方法的思路。我们只需实现我们的断言逻辑，并在断言应该表示测试失败时引发`self.failureException`。`self.failureException`是`unittest.TestCase`的一个属性，通常设置为`AssertionError`异常。我们可以自己引发`AssertionError`，但`unittest`模块允许我们使用不同的异常进行配置，因此最好引发`self.failureException`，它总是设置为正确的值用于使用。

当在多个测试中反复使用相同的断言序列时，我们应该看看是否有机会用更清晰地表达我们意图的高级断言来替换内置断言的调用。

## 使用自定义的相等性检查器

`assertEqual` 方法的酷特性是它根据被比较的对象类型给出自定义的失败消息。如果我们尝试断言两个整数，我们会得到以下结果：

```py
>>> import unittest
>>> testcase = unittest.TestCase()
>>> testcase.assertEqual(1, 2)
Traceback (most recent call last):
 ...
AssertionError: 1 != 2

```

另一方面，断言列表会给我们另一个消息，显示预期列表和实际列表之间的差异，如下所示：

```py
>>> import unittest
>>> testcase = unittest.TestCase()
>>> testcase.assertEqual([1, 2], [1, 3])
Traceback (most recent call last):
 ...
AssertionError: Lists differ: [1, 2] != [1, 3]

First differing element 1:
2
3

- [1, 2]
?     ^

+ [1, 3]
?     ^

```

在幕后，`assertEqual` 根据被比较的对象类型委托给不同的函数。这就是我们如何为大多数常见的内置数据结构（如字符串、序列、列表、元组、集合和字典）获得具体和相关的相等性检查。

幸运的是，这个灵活的系统对开发者开放，这意味着我们可以为我们的应用程序对象添加自己的相等性检查器。以下是我们尝试比较两个 `Stock` 对象的默认场景：

```py
>>> import unittest
>>> from stock_alerter.stock import Stock
>>> test_case = unittest.TestCase()
>>> stock_1 = Stock("GOOG")
>>> stock_2 = Stock("GOOG")
>>> test_case.assertEqual(stock_1, stock_2)
Traceback (most recent call last):
 ...
AssertionError: <stock_alerter.stock.Stock object at 0x000000000336EDD8> != <stock_alerter.stock.Stock object at 0x00000000033E9588>

```

断言失败，因为尽管两个对象包含相同的数据，但在内存中它们仍然是不同的对象。现在让我们尝试为 `Stock` 类注册我们自己的相等性函数，该函数仅比较符号以识别 `Stock` 对象之间的相等性。我们只需使用 `addTypeEqualityFunc` 方法注册我们的检查器，如下所示：

```py
>>> test_case.addTypeEqualityFunc(Stock, lambda stock_1, stock_2, msg: stock_1.symbol == stock_2.symbol)
>>> test_case.assertEqual(stock_1, stock_2)
>>> print(test_case.assertEqual(stock_1, stock_2))
None
>>>

```

检查相等性的函数接受三个参数：第一个对象、第二个对象以及用户传递给 `assertEqual` 的可选消息。一旦我们以这种方式注册了函数，我们就可以调用 `assertEqual` 并传入两个 `Stock` 对象，`assertEqual` 将将比较委托给我们所注册的函数。

以这种方式使用相等性函数是断言单元测试代码中应用域对象的一种好方法。尽管如此，这种方法有两个限制：

+   我们必须为给定类型使用相同的比较函数。我们无法在某些测试中使用一个比较函数，而在其他测试中使用另一个比较函数。

+   `assertEqual` 的两个参数都必须是该类型的对象。我们无法传入不同类型的两个对象。

这两个限制都可以通过使用匹配器来克服，这就是我们现在将注意力转向的地方。

## 使用匹配器

使断言更易读的第三种方法是创建自定义匹配器对象，以便在断言期间使比较更易读。我们在之前为 `EmailAction` 类编写测试时看到了使用匹配器的一瞥。以下是对该匹配器再次的查看：

```py
class MessageMatcher:
    def __init__(self, expected):
        self.expected = expected

    def __eq__(self, other):
        return self.expected["Subject"] == other["Subject"] and \
            self.expected["From"] == other["From"] and \
            self.expected["To"] == other["To"] and \
            self.expected["Message"] == other._payload
```

匹配器可以是任何实现了 `__eq__` 方法的类。该方法将实际对象作为参数，并且该方法可以实现所需的任何比较逻辑。使用这种方法，我们可以在断言中直接比较域对象，而无需用多个单独的断言来杂乱无章。

匹配器不需要比较完整的域对象。我们可以仅比较我们感兴趣的属性。实际上，我们可以创建不同的匹配器来匹配特定的对象子集。例如，我们可能会创建一个 `AlertMessageMatcher`，如下所示：

```py
class AlertMessageMatcher:
    def __init__(self, expected):
        self.expected = expected

    def __eq__(self, other):
        return self.expected["Subject"] == "New Stock Alert" and \
            self.expected["From"] == other["From"] and \
            self.expected["To"] == other["To"] and \
            self.expected["Message"] == other._payload
```

这个匹配器只会匹配具有给定主题的警报消息，同时从预期对象中获取其他参数。

# 摘要

在本章中，你更详细地探讨了保持测试可维护性的重要但常被忽视的主题。你研究了保持一致的测试文件布局方案的重要性以及各种替代方案的优缺点。你研究了测试的命名和分组，然后转向使测试更容易理解的主题。我们讨论的一些策略包括使用文档字符串、创建自定义测试类层次结构和利用固定装置。最后，你研究了通过使用辅助函数、自定义断言、等价函数和编写自定义匹配器来使代码更易于阅读。

在下一章中，你将学习如何使用`doctest`模块将测试纳入你的文档中。
