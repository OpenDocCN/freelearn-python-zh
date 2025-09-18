# 第二章 红-绿-重构 – TDD 循环

在上一章中，我们通过创建一个失败的测试然后使其通过，进行了一个小的 TDD 循环。在这一章中，我们将通过编写更多测试来完善`Stock`类的其余部分。在这个过程中，我们将更深入地了解 TDD 循环和`unittest`模块。

# 测试是可执行的要求

在第一次测试中，我们编写了一个非常简单的测试，用来检查新的`Stock`类是否将`price`属性初始化为`None`。我们现在可以思考接下来想要实现哪些要求。

一个细心的读者可能会注意到上一句中使用的术语，我说我们可以思考接下来要实现的要求，而不是说我们可以思考接下来要编写的测试。这两个陈述是等效的，因为在 TDD（测试驱动开发）中，测试不过是要求。每次我们编写一个测试并实现代码使其通过，我们实际上是在使代码满足某些要求。从另一个角度来看，测试只是可执行的规格说明。需求文档往往与实际实现脱节，但测试则不可能出现这种情况，因为一旦它们脱节，测试就会失败。

在上一章中，我们提到`Stock`类将用于存储股票符号的价格信息和价格历史。这表明我们需要一种方法来设置价格，每次更新时都要使用。让我们实现一个满足以下要求的`update`方法：

+   它应该接受时间戳和价格值，并在对象上设置它

+   价格不能为负

+   经过多次更新后，该对象会给我们提供最新的价格

# 安排-行动-断言

让我们从第一个要求开始。以下是测试代码：

```py
    def test_stock_update(self):
        """An update should set the price on the stock object
        We will be using the `datetime` module for the timestamp
        """
        goog = Stock("GOOG")
        goog.update(datetime(2014, 2, 12), price=10)
        self.assertEqual(10, goog.price)
```

在这里，我们调用`update`方法（目前还不存在）并传入时间戳和价格，然后检查价格是否已正确设置。我们使用`unittest.TestCase`类提供的`assertEqual`方法来断言值。

由于我们使用`datetime`模块来设置时间戳，我们将在文件顶部添加`from datetime import datetime`这一行，以便它能够运行。

这个测试遵循 Arrange-Act-Assert 模式。

1.  **安排**：为测试设置上下文。在这种情况下，我们创建一个`Stock`对象。在其他测试中，可能需要创建多个对象或将一些东西连接起来，以便特定测试需要。

1.  **行动**：执行我们想要测试的操作。在这里，我们使用适当的参数调用`update`方法。

1.  **断言**：最后我们断言结果符合预期。

在这个测试中，模式的每一部分都占用了一行代码，但这并不总是如此。通常，测试的每一部分会有多行代码。

# 记录我们的测试

当我们运行测试时，我们得到以下输出：

```py
.E
==================================================================
ERROR: test_stock_update (__main__.StockTest)
An update should set the price on the stock object
------------------------------------------------------------------
Traceback (most recent call last):
 File "stock_alerter\stock.py", line 22, in test_stock_update
 goog.update(datetime(2014, 2, 12), price=10)
AttributeError: 'Stock' object has no attribute 'update'

------------------------------------------------------------------
Ran 2 tests in 0.001s
FAILED (errors=1)

```

测试如预期失败，但有趣的是，文档字符串的第一行在第四行打印出来。这很有用，因为它提供了更多关于哪个案例失败的信息。这显示了使用第一行作为简短摘要，其余的文档字符串作为更详细解释的另一种记录测试的方法。详细的解释在测试失败时不会打印出来，所以不会影响测试失败的输出。

我们使用了两种测试文档的方法：

+   编写描述性的测试方法名称

+   在文档字符串中添加解释

哪个更好？大多数情况下，测试是自我解释的，不需要很多背景解释。在这种情况下，一个命名良好的测试方法就足够了。

然而，有时测试方法名称变得非常长，以至于变得笨拙，实际上降低了代码的可读性。在其他时候，我们可能想要更详细地解释我们在测试什么以及为什么测试。在这种情况下，缩短方法名称并在文档字符串中添加解释是一个好主意。

这里是实现通过这个测试的方法：

```py
    def update(self, timestamp, price):
        self.price = price
```

这个最小化实现通过了测试。就像第一次实现一样，我们并不是试图实现全部功能。我们只想实现足够的部分来通过测试。记住，当测试通过时，意味着需求得到了满足。在这个阶段，我们有两个通过测试，实际上我们没有太多需要重构的地方，所以让我们继续前进。

# 测试异常

另一个要求是价格不能为负。如果价格是负数，我们希望抛出一个 `ValueError`。我们如何在测试中检查这个期望呢？这里有一种方法可以做到：

```py
    def test_negative_price_should_throw_ValueError(self):
        goog = Stock("GOOG")
        try:
            goog.update(datetime(2014, 2, 13), -1)
        except ValueError:
            return
        self.fail("ValueError was not raised")
```

在前面的代码中，我们使用负价格调用 `update` 方法。这个调用被 `try...except` 块包裹，以捕获 `ValueError`。如果异常被正确抛出，控制将进入 `except` 块，我们在那里从测试中返回。由于测试方法成功返回，它被标记为通过。如果没有抛出异常，则调用 `fail` 方法。这是 `unittest.TestCase` 提供的另一个方法，当它被调用时，会抛出一个测试失败异常。我们可以传递一个消息来提供一些解释，说明为什么它失败了。

这里是通过这个测试的代码：

```py
    def update(self, timestamp, price):
        if price < 0:
            raise ValueError("price should not be negative")
        self.price = price
```

使用这段代码，到目前为止的所有三个测试都通过了。

由于检查异常是一个相当常见的案例，`unittest` 提供了一种更简单的方式来处理：

```py
    def test_negative_price_should_throw_ValueError(self):
        goog = Stock("GOOG")
        self.assertRaises(ValueError, goog.update, datetime(2014, 2, 13), -1)
```

`assertRaises` 方法将期望的异常作为第一个参数，将需要调用的函数作为第二个参数，函数的参数通过剩余的参数传递。如果你需要使用关键字参数调用函数，那么它们可以作为关键字参数传递给 `assertRaises` 方法。

### 注意

注意，`assertRaises` 的第二个参数是对要调用的函数的引用。这就是为什么我们在函数名称后面不放置括号的原因。

如果传入函数引用和参数列表感觉不自然，那么 `assertRaises` 提供了另一种我们可以使用的语法：

```py
    def test_negative_price_should_throw_ValueError(self):
        goog = Stock("GOOG")
        with self.assertRaises(ValueError):
            goog.update(datetime(2014, 2, 13), -1)
```

这里发生了什么？当我们只向 `assertRaises` 传递一个参数时，会返回一个上下文管理器。我们可以使用 `with` 语句，并将我们的操作放在该块中。如果该块抛出了预期的异常，那么上下文管理器会匹配它并退出块而不会出现错误。然而，如果块中没有抛出预期的异常，那么上下文管理器在块退出时会引发失败。

# 探索断言方法

现在我们对 `update` 的要求只剩下一个：

+   **-Done-** 它应该接受一个时间戳和价格值，并将其设置在对象上

+   **-Done-** 价格不能为负

+   经过多次更新后，对象给出了最新的价格

让我们来看剩下的要求。以下是测试：

```py
    def test_stock_price_should_give_the_latest_price(self):
        goog = Stock("GOOG")
        goog.update(datetime(2014, 2, 12), price=10)
        goog.update(datetime(2014, 2, 13), price=8.4)
        self.assertAlmostEqual(8.4, goog.price, delta=0.0001)
```

这个测试所做的只是简单地调用 `update` 两次，并在我们请求价格时提供最新的价格。测试的有趣之处在于我们在这里使用了 `assertAlmostEqual` 方法。这种方法通常用于检查浮点数的相等性。我们为什么不使用普通的 `assertEqual` 呢？原因是由于浮点数的存储方式，结果可能不会正好是您期望的数字。您期望的值和实际存储的值之间可能存在一个非常小的差异。考虑到这一点，`assertAlmostEqual` 方法允许我们在比较中指定公差。例如，如果我们期望 8.4 但实际值是 8.39999999，测试仍然会通过。

`assertAlmostEqual` 方法有两种指定公差的方式。我们上面使用的方法涉及传递一个 `delta` 参数，表示预期值和实际值之间的差异应在 delta 范围内。我们上面指定的 `delta` 参数是 `0.0001`，这意味着任何在 8.3999 和 8.4001 之间的值都会通过测试。

指定公差的其他方法是使用以下代码中所示的 `places` 参数：

```py
        self.assertAlmostEqual(8.4, goog.price, places=4)
```

如果使用此参数，则在比较之前，预期的值和实际的值都会四舍五入到指定的十进制位数。请注意，您需要传递 `delta` 参数或 `places` 参数。同时传递这两个参数是错误的。

到目前为止，我们已经使用了以下断言方法：

+   `assertIsNone`

+   `assertEqual`

+   `assertRaises`

+   `assertAlmostEqual`

+   `fail`

`unittest` 模块提供了大量我们可以用于各种条件的断言方法。以下列出了一些常见的：

+   `assertFalse(x, msg)`，`assertTrue(x, msg)`

+   `assertIsNone(x, msg)`，`assertIsNotNone(x, msg)`

+   `assertEqual(x, y, msg)`，`assertNotEqual(x, y, msg)`

+   `assertAlmostEqual(x, y, places, msg, delta)`，`assertNotAlmostEqual(x, y, places, msg, delta)`

+   `assertGreater(x, y, msg)`，`assertGreaterEqual(x, y, msg)`

+   `assertLess(x, y, msg)`，`assertLessEqual(x, y, msg)`

+   `assertIs(x, y, msg)`，`assertIsNot(x, y, msg)`

+   `assertIn(x, seq, msg)`，`assertNotIn(x, seq, msg)`

+   `assertIsInstance(x, cls, msg)`，`assertNotIsInstance(x, cls, msg)`

+   `assertRegex(text, regex, msg)`，`assertNotRegex(text, regex, msg)`

+   `assertRaises(exception, callable, *args, **kwargs)`

+   `fail(msg)`

大多数前面的函数都是自解释的。以下是一些需要一些解释的点：

+   `msg`参数：大多数断言方法都接受一个可选的消息参数。可以在这里传递一个字符串，如果断言失败，它将被打印出来。通常，默认消息已经非常描述性，因此不需要此参数。大多数时候它与`fail`方法一起使用，就像我们刚才看到的那样。

+   `assertEqual`与`assertIs`：这两组断言非常相似。关键的区别在于前者检查的是`*相等性*`，而后者断言用于检查对象的`*身份*`。第二个断言在之前的例子中失败，因为尽管两个对象相等，但它们仍然是两个不同的对象，因此它们的身份是不同的：

    ```py
    >>> test = unittest.TestCase()
    >>> test.assertEqual([1, 2], [1, 2])  # Assertion Passes
    >>> test.assertIs([1, 2], [1, 2])     # Assertion Fails
    Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
     File "C:\Python34\lib\unittest\case.py", line 1067, in assertIs
     self.fail(self._formatMessage(msg, standardMsg))
     File "C:\Python34\lib\unittest\case.py", line 639, in fail
     raise self.failureException(msg)
    AssertionError: [1, 2] is not [1, 2]

    ```

+   `assertIn`/`assertNotIn`: 这些断言用于检查一个元素是否在序列中。这包括字符串、列表、集合以及任何支持`in`操作符的其他对象。

+   `assertIsInstance`/`assertNotIsInstance`: 它们检查一个对象是否是给定类的实例。`cls`参数也可以是一个类元组，用于断言对象是这些类中的任何一个的实例。

`unittest`模块还提供了一些不太常用的断言：

+   `assertRaisesRegex(exception, regex, callable, *args, **kwargs)`: 这个断言与`assertRaises`类似，但它还额外接受一个`regex`参数。可以在这里传递一个正则表达式，断言将检查是否抛出了正确的异常，以及异常消息是否与正则表达式匹配。

+   `assertWarns(warning, callable, *args, **kwargs)`: 它与`assertRaises`类似，但检查是否抛出了警告。

+   `assertWarnsRegex(warning, callable, *args, **kwargs)`: 它是`assertRaisesRegex`的警告等效。

# 特定断言与通用断言

可能会有人问一个问题，为什么有这么多不同的断言方法。为什么我们不能像以下代码中所示的那样使用`assertTrue`而不是更具体的断言呢：

```py
assertInSeq(x, seq)
assertTrue(x in seq)

assertEqual(10, x)
assertTrue(x == 10)
```

虽然它们确实等价，但使用特定断言的一个动机是，如果断言失败，你会得到更好的错误消息。当比较列表和字典等对象时，错误消息将显示差异的确切位置，这使得理解更容易。因此，建议尽可能使用更具体的断言。

# 设置和清理

让我们看看我们迄今为止所做的测试：

```py
    def test_price_of_a_new_stock_class_should_be_None(self):
        stock = Stock("GOOG")
        self.assertIsNone(stock.price)

    def test_stock_update(self):
        """An update should set the price on the stock object
        We will be using the `datetime` module for the timestamp
        """
        goog = Stock("GOOG")
        goog.update(datetime(2014, 2, 12), price=10)
        self.assertEqual(10, goog.price)

    def test_negative_price_should_throw_ValueError(self):
        goog = Stock("GOOG")
        with self.assertRaises(ValueError):
            goog.update(datetime(2014, 2, 13), -1)

    def test_stock_price_should_give_the_latest_price(self):
        goog = Stock("GOOG")
        goog.update(datetime(2014, 2, 12), price=10)
        goog.update(datetime(2014, 2, 13), price=8.4)
        self.assertAlmostEqual(8.4, goog.price, delta=0.0001)
```

如果你注意到，每个测试都通过实例化一个 `Stock` 对象来进行相同的设置，该对象随后用于测试。在这种情况下，设置只是一行代码，但有时我们可能需要在运行测试之前执行多个步骤。我们可以在每个测试中重复设置代码，而不是使用 `TestCase` 类提供的 `setUp` 方法：

```py
    def setUp(self):
        self.goog = Stock("GOOG")

    def test_price_of_a_new_stock_class_should_be_None(self):
        self.assertIsNone(self.goog.price)

    def test_stock_update(self):
        """An update should set the price on the stock object We will be  using the `datetime` module for the timestamp """
        self.goog.update(datetime(2014, 2, 12), price=10)
        self.assertEqual(10, self.goog.price)

    def test_negative_price_should_throw_ValueError(self):
        with self.assertRaises(ValueError):
            self.goog.update(datetime(2014, 2, 13), -1)

    def test_stock_price_should_give_the_latest_price(self):
        self.goog.update(datetime(2014, 2, 12), price=10)
        self.goog.update(datetime(2014, 2, 13), price=8.4)
        self.assertAlmostEqual(8.4, self.goog.price, delta=0.0001)
```

在前面的代码中，我们正在用我们自己的方法覆盖默认的 `setUp` 方法。我们将设置代码放在这个方法中。这个方法在每个测试之前执行，因此在这里完成的初始化可用于我们的测试方法。请注意，我们必须更改我们的测试以使用 `self.goog`，因为它现在已成为实例变量。

与 `setUp` 类似，还有一个 `tearDown` 方法，它在测试执行后立即执行。我们可以在该方法中执行任何必要的清理操作。

`setUp` 和 `tearDown` 方法在每个测试前后执行。如果我们想为测试组只执行一次设置，怎么办？可以将 `setUpClass` 和 `tearDownClass` 方法实现为类方法，并且它们将只在每个测试类中执行一次。同样，`setUpModule` 和 `tearDownModule` 函数可用于在整个模块中只初始化一次。以下示例显示了执行顺序：

```py
import unittest

def setUpModule():
    print("setUpModule")

def tearDownModule():
    print("tearDownModule")

class Class1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("  setUpClass")

    @classmethod
    def tearDownClass(cls):
        print("  tearDownClass")

    def setUp(self):
        print("       setUp")

    def tearDown(self):
        print("       tearDown")

    def test_1(self):
        print("         class 1 test 1")

    def test_2(self):
        print("         class 1 test 2")

class Class2Test(unittest.TestCase):
    def test_1(self):
        print("         class 2 test 1")
```

当运行此代码时，输出如下：

```py
setUpModule
  setUpClass
       setUp
         class 1 test 1
       tearDown
       setUp
         class 1 test 2
       tearDown
  tearDownClass
         class 2 test 1
tearDownModule
```

如我们所见，模块级别的设置首先执行，然后是类级别，最后是测试用例级别。清理操作以相反的顺序执行。在实际使用中，测试用例级别的 `setUp` 和 `tearDown` 方法非常常用，而类级别和模块级别的设置则不需要太多。类级别和模块级别的设置仅在存在昂贵的设置步骤时使用，例如连接数据库或远程服务器，并且最好只设置一次并共享给所有测试。

### 注意

**使用类级别和模块级别设置时的警告**

在类和模块级别进行的任何初始化都是在测试之间共享的。因此，确保在一个测试中进行的修改不会影响另一个测试非常重要。例如，如果我们已经在 `setUpClass` 中初始化了 `self.goog = Stock("GOOG")`，那么如果其他测试在它之前执行并改变了对象的状态，那么第一个测试检查新 `Stock` 对象的价格应该是 `None` 将会失败。

记住，测试运行的顺序是不确定的。测试应该是独立的，并且无论执行顺序如何，都应该通过。因此，谨慎使用`setUpClass`和`setUpModule`至关重要，以确保只在测试之间可以重用的状态下设置。

# 脆弱的测试

我们已经实现了`update`方法的三项要求：

+   **-完成-** 它应该接受时间戳和价格值，并将它们设置在对象上

+   **-完成-** 经过多次更新后，对象会给出最新的价格

+   **-完成-** 价格不能为负

现在，让我们假设出现了一个我们之前不知道的新要求：

+   `Stock`类需要一个方法来检查股票是否有上升趋势。上升趋势是指最新的三个更新值都比前一个更新值高。

到目前为止，我们的`Stock`实现只是存储最新的价格。为了实现这一功能，我们需要存储一些过去的价格历史值。一种方法是将`price`变量改为列表。问题是当我们改变实现内部结构时，它将破坏我们所有的测试，因为它们都直接访问`price`变量并断言它具有特定的值。

我们看到的是测试脆弱性的一个例子。

当实现细节的改变需要更改测试用例时，测试就变得脆弱。理想情况下，测试应该测试接口而不是直接测试实现。毕竟，接口是其他单元将用来与这个单元交互的。当我们通过接口进行测试时，它允许我们有自由地更改代码实现，而不必担心破坏测试。

### 注意

**测试可能失败的三种方式：**

+   如果在测试的代码中引入了错误

+   如果测试与实现紧密耦合，并且我们对代码进行更改以修改实现，但没有引入错误（例如，重命名变量或修改内部设计）

+   如果测试需要一些不可用的资源（例如，连接到外部服务器，但服务器已关闭）

理想情况下，第一个情况应该是测试失败的唯一情况。我们应该尽可能避免第二个和第三个。

有时测试特定的实现细节可能很重要。例如，假设我们有一个类，它预期执行复杂的计算并将结果缓存以供将来使用。测试缓存功能唯一的方法是验证计算值是否存储在缓存中。如果我们后来更改缓存方法（例如，从文件缓存切换到 memcache），那么我们也必须更改测试。

碎片化测试可能比没有测试更糟糕，因为每次实现变更都需要修复十个或二十个测试的维护开销可能会让开发者远离测试驱动开发（TDD），增加挫败感，并导致团队禁用或跳过测试。以下是一些关于如何考虑测试碎片化的指南：

+   如果可能的话，避免在测试中使用实现细节，只使用公开的接口。这包括在设置代码和断言中只使用接口方法。

+   如果测试需要检查被测试单元内部的特定功能，并且这是一个重要的功能，那么检查特定的实现动作可能是有意义的。

+   如果使用外部接口设置我们想要的确切状态很麻烦，或者没有接口方法可以检索我们想要断言的特定值，那么我们可能需要在测试中查看实现细节。

+   如果我们相当有信心，实现细节在未来不太可能发生变化，那么我们可能会继续在测试中使用特定于实现的细节。

对于第二种和第三种情况，重要的是要理解在便利性、测试可读性和碎片化之间存在权衡。没有正确答案，这是一个主观决定，需要权衡每种具体情况的利弊。

# 重新设计

在上一节中，我们讨论了检查股票是否有上升趋势的新要求。

让我们先从编写一个测试开始：

```py
class StockTrendTest(unittest.TestCase):
    def setUp(self):
        self.goog = Stock("GOOG")
    def test_increasing_trend_is_true_if_price_increase_for_3_updates(self):
        timestamps = [datetime(2014, 2, 11), datetime(2014, 2, 12), datetime(2014, 2, 13)]
        prices = [8, 10, 12]
        for timestamp, price in zip(timestamps, prices):
            self.goog.update(timestamp, price)
        self.assertTrue(self.goog.is_increasing_trend())
```

这个测试接受三个时间戳和价格，并对每个价格执行更新。由于所有三个价格都在增加，`is_increasing_trend`方法应该返回`True`。

要使这个测试通过，我们首先需要添加支持存储价格历史的功能。

在初始化器中，让我们将`price`属性替换为`price_history`列表。这个列表将存储价格更新的历史，每个新的更新都添加到列表的末尾：

```py
    def __init__(self, symbol):
        self.symbol = symbol
        self.price_history = []
```

### 注意

在进行这个更改后，现在所有测试都将失败，包括之前通过的那些。只有在我们完成几个步骤后，我们才能再次使测试通过。当测试通过时，我们可以不断运行测试，以确保我们的更改没有破坏任何功能。当测试失败时，我们没有这样的安全网。某些设计更改，就像我们现在所做的，在完成一系列更改之前，不可避免地会使许多测试暂时失败。我们应该尽量减少在测试失败时进行更改的时间，一次只做小改动。这允许我们在进行过程中验证我们的更改。

我们现在可以将`update`方法更改为将新价格存储在这个列表中。

```py
    def update(self, timestamp, price):
        if price < 0:
            raise ValueError("price should not be negative")
            self.price_history.append(price)
```

我们将保留当前通过访问`price`属性获取最新价格的用户界面。然而，由于我们已经用`price_history`列表替换了`price`属性，我们需要创建一个属性来模拟现有的界面：

```py
    @property
    def price(self):
        return self.price_history[-1] \
        if self.price_history else None
```

通过这个更改，我们可以再次运行测试，并看到我们所有的先前测试仍然通过，只有新的趋势功能测试失败。

新的设计现在允许我们实现代码以通过趋势测试：

```py
    def is_increasing_trend(self):
        return self.price_history[-3] < \
        self.price_history[-2] < self.price_history[-1]
```

该方法的实现只是简单地检查最后三个价格更新是否在增加。代码实现后，包括新的测试在内，我们所有的测试都将通过。

### 注意

**关于属性的快速入门**

属性是 Python 的一个特性，我们可以将属性访问委托给一个函数。由于我们将价格声明为属性，访问`Stock.price`将导致调用该方法而不是搜索属性。在我们的实现中，它允许我们创建一个接口，这样其他模块可以像属性一样引用股票价格，尽管在对象中实际上没有这样的属性。

# 重构测试

第一个测试通过后，我们可以继续进行第二个测试：

```py
    def test_increasing_trend_is_false_if_price_decreases(self):
        timestamps = [datetime(2014, 2, 11), datetime(2014, 2, 12), \ 
            datetime(2014, 2, 13)]
        prices = [8, 12, 10]
        for timestamp, price in zip(timestamps, prices):
            self.goog.update(timestamp, price)
        self.assertFalse(self.goog.is_increasing_trend())
```

我们的实现已经通过了这个测试，所以让我们继续进行第三个测试：

```py
    def test_increasing_trend_is_false_if_price_equal(self):
        timestamps = [datetime(2014, 2, 11), datetime(2014, 2, 12), \ 
            datetime(2014, 2, 13)]
        prices = [8, 10, 10]
        for timestamp, price in zip(timestamps, prices):
            self.goog.update(timestamp, price)
        self.assertFalse(self.goog.is_increasing_trend())
```

当前的代码也通过了这个测试。但让我们在这里暂停一下。如果我们看看到目前为止的测试用例，我们可以看到在测试之间有很多代码被重复使用。设置代码也不是很易读。这里最重要的行是价格列表，它被隐藏在混乱中。我们需要清理一下。我们要做的是将公共代码放入辅助方法中：

```py
class StockTrendTest(unittest.TestCase):
    def setUp(self):
        self.goog = Stock("GOOG")

    def given_a_series_of_prices(self, prices):
        timestamps = [datetime(2014, 2, 10), datetime(2014, 2, \ 
            11), datetime(2014, 2, 12), datetime(2014, 2, 13)]
        for timestamp, price in zip(timestamps, prices):
            self.goog.update(timestamp, price)

    def test_increasing_trend_is_true_if_price_increase_for_3_updates(self):
        self.given_a_series_of_prices([8, 10, 12])
        self.assertTrue(self.goog.is_increasing_trend())

    def test_increasing_trend_is_false_if_price_decreases(self):
        self.given_a_series_of_prices([8, 12, 10])
        self.assertFalse(self.goog.is_increasing_trend())

    def test_increasing_trend_is_false_if_price_equal(self):
        self.given_a_series_of_prices([8, 10, 10])
        self.assertFalse(self.goog.is_increasing_trend())
```

太好了！不仅消除了重复，测试的可读性也大大提高。默认情况下，`unittest`模块会寻找以单词`test`开头的方法，并且只执行这些方法作为测试，因此我们的辅助方法被误认为是测试用例的风险很小。

### 注意

记住，测试用例也是代码。所有关于编写干净、可维护和可读代码的规则也适用于测试用例。

# 探索规则类

到目前为止，我们一直专注于`Stock`类。现在让我们将注意力转向规则类。

### 注意

从这本书的这个点开始，我们将查看实现代码，然后展示我们如何有效地测试它。注意，这并不意味着先写代码，然后写单元测试。TDD 过程仍然是先测试，然后是实现。是测试用例将驱动实现策略。我们首先展示实现代码，只是为了更容易理解接下来的测试概念。所有这些代码最初都是先写测试！

规则类跟踪用户想要跟踪的规则，并且它们可以是不同类型的。例如，当股票价格超过某个值或符合某种趋势时发送警报。

下面是一个`PriceRule`实现的示例：

```py
class PriceRule:
        """PriceRule is a rule that triggers when a stock price
        satisfies a condition (usually greater, equal or lesser
        than a given value)"""

    def __init__(self, symbol, condition):
        self.symbol = symbol
        self.condition = condition

    def matches(self, exchange):
        try:
            stock = exchange[self.symbol]
        except KeyError:
            return False
        return self.condition(stock) if stock.price else False

    def depends_on(self):
        return {self.symbol}
```

此类使用股票符号和条件进行初始化。条件可以是一个 lambda 或函数，它接受一个股票作为参数并返回`True`或`False`。规则匹配当股票匹配条件时。此类的关键方法是`matches`方法。此方法根据规则是否匹配返回`True`或`False`。`matches`方法接受一个交易所作为参数。这只是一个包含所有可用于应用程序的股票的字典。

我们还没有讨论`depends_on`方法。此方法仅返回哪些股票更新依赖于规则。这将在稍后用于检查任何特定股票更新时的规则。对于`PriceRule`，它仅依赖于在初始化器中传递的股票。一个细心的读者会注意到它返回一个集合（花括号），而不是列表。

将此规则代码放入`stock_alerter`目录下的`rule.py`文件中。

下面是如何使用`PriceRule`的示例：

```py
>>> from datetime import datetime
>>> from stock_alerter.stock import Stock
>>> from stock_alerter.rule import PriceRule
>>>
>>> # First, create the exchange
>>> exchange = {"GOOG": Stock("GOOG"), "MSFT": Stock("MSFT")}
>>>
>>> # Next, create the rule, checking if GOOG price > 100
>>> rule = PriceRule("GOOG", lambda stock: stock.price > 100)
>>>
>>> # No updates? The rule is False
>>> rule.matches(exchange)
False
>>>
>>> # Price does not match the rule? Rule is False
>>> exchange["GOOG"].update(datetime(2014, 2, 13), 50)
>>> rule.matches(exchange)
False
>>>
>>> # Price matches the rule? Rule is True
>>> exchange["GOOG"].update(datetime(2014, 2, 13), 101)
>>> rule.matches(exchange)
True
>>>

```

下面是一些测试的示例：

```py
class PriceRuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        goog = Stock("GOOG")
        goog.update(datetime(2014, 2, 10), 11)
        cls.exchange = {"GOOG": goog}

    def test_a_PriceRule_matches_when_it_meets_the_condition(self):
        rule = PriceRule("GOOG", lambda stock: stock.price > 10)
        self.assertTrue(rule.matches(self.exchange))

    def test_a_PriceRule_is_False_if_the_condition_is_not_met(self):
        rule = PriceRule("GOOG", lambda stock: stock.price < 10)
        self.assertFalse(rule.matches(self.exchange))

    def test_a_PriceRule_is_False_if_the_stock_is_not_in_the_exchange(self):
        rule = PriceRule("MSFT", lambda stock: stock.price > 10)
        self.assertFalse(rule.matches(self.exchange))

    def test_a_PriceRule_is_False_if_the_stock_hasnt_got_an_update_yet(self):
        self.exchange["AAPL"] = Stock("AAPL")
        rule = PriceRule("AAPL", lambda stock: stock.price > 10)
        self.assertFalse(rule.matches(self.exchange))

    def test_a_PriceRule_only_depends_on_its_stock(self):
        rule = PriceRule("MSFT", lambda stock: stock.price > 10)
        self.assertEqual({"MSFT"}, rule.depends_on())
```

需要注意的一点是我们如何使用`setupClass`方法来进行设置。如前所述，此方法只为整个测试系列调用一次。我们使用此方法来设置交易所并存储它。请记住在`setupClass`方法上放置`@classmethod`装饰器。我们在类中存储交易所，并在测试中使用`self.exchange`来访问它。

否则，测试只是构建一个规则并检查匹配方法。

### 注意

**装饰器（非常）快速入门**

装饰器是接受一个函数作为输入并返回另一个函数作为输出的函数。Python 有一个简写语法，我们可以通过在函数或方法上方使用`@decorator`来应用装饰器。有关更多详细信息，请参阅 Python 文档或教程。一个好的教程是[`simeonfranklin.com/blog/2012/jul/1/python-decorators-in-12-steps/`](http://simeonfranklin.com/blog/2012/jul/1/python-decorators-in-12-steps/)。

现在我们来看另一个规则类，即`AndRule`。当您想要组合两个或更多规则时，会使用`AndRule`，例如，`AAPL > 10 AND GOOG > 15`。

下面是如何为它编写测试的示例：

```py
class AndRuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        goog = Stock("GOOG")
        goog.update(datetime(2014, 2, 10), 8)
        goog.update(datetime(2014, 2, 11), 10)
        goog.update(datetime(2014, 2, 12), 12)
        msft = Stock("MSFT")
        msft.update(datetime(2014, 2, 10), 10)
        msft.update(datetime(2014, 2, 11), 10)
        msft.update(datetime(2014, 2, 12), 12)
        redhat = Stock("RHT")
        redhat.update(datetime(2014, 2, 10), 7)
        cls.exchange = {"GOOG": goog, "MSFT": msft, "RHT": redhat}

    def test_an_AndRule_matches_if_all_component_rules_are_true(self):
        rule = AndRule(PriceRule("GOOG", lambda stock: stock.price > 8), PriceRule("MSFT", lambda stock: stock.price > 10))
        self.assertTrue(rule.matches(self.exchange))
```

编写测试的第一件事是它让我们思考类将如何被使用。

例如，我们应该如何将各种子规则传递给`AndRule`？我们应该有一个设置它们的方法吗？我们应该将它们作为列表传递？我们应该将它们作为单独的参数传递？这是一个设计决策，首先创建测试允许我们作为类的用户实际编写代码，并确定哪种选择最佳。在上面的测试中，我们决定将每个子规则作为单独的参数传递给`AndRule`构造函数。

现在决策已经做出，我们可以编写一些代码来通过测试：

```py
class AndRule:
    def __init__(self, *args):
        self.rules = args

    def matches(self, exchange):
        return all([rule.matches(exchange) for rule in self.rules])
```

在这里，我们可以看到测试驱动过程如何帮助我们驱动代码的设计。

### 注意

**`all`函数**

`all`函数是一个内置函数，它接受一个列表，并且只有当列表中的每个元素都是`True`时才返回`True`。

# 练习

现在是时候将我们新学的技能付诸实践了。以下是向`Stock`类中添加的新要求：

+   有时，更新可能会出现顺序问题，我们可能会先收到一个较新时间戳的更新，然后是较旧时间戳的更新。这可能是由于随机网络延迟，或者有时我们可能从不同的来源收到更新，其中一个可能稍微领先于另一个。

+   `Stock`类应该能够处理这种情况，并且`price`属性应该根据时间戳返回最新的价格。

+   `is_increasing_trend`也应该根据其时间戳处理最新的三个价格。

尝试实现这个要求。不要对这些方法的现有接口进行任何更改，但请随意根据需要更改实现。以下是一些需要考虑的事项：

+   我们现有的设计支持这个新特性吗？我们需要对当前设计做出任何更改吗？

+   我们将为此要求编写什么样的测试？

+   在我们让一切正常运行之后，我们是否可以进行一些清理工作，使代码更易于阅读或维护？

+   我们在做出这个更改后是否需要更改现有的测试，或者它们是否无需修改就能继续工作？

在练习结束时，你应该让所有现有的测试通过，以及为这个要求编写的任何新测试。完成后，你可以查看附录 A，*练习答案*，以获取这个练习的一个可能的解决方案。

# 摘要

在本章中，我们更详细地研究了 TDD 周期。我们学习了 Arrange-Act-Assert 模式，更详细地研究了提供的各种断言，以及设置测试和之后的清理的一些不同方法。最后，我们探讨了如何防止测试过于脆弱，并进行了一些基本的重构。
